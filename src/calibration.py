"""
PredictOS — Platt Scaling Calibration

XGBoost's raw output is NOT a calibrated probability.
It's a score that correlates with probability but is systematically biased.

Example of the problem:
  XGBoost says 0.72 for a market
  That market actually resolves YES only 58% of the time
  → Your return estimate says +$X, actual expected value is much lower
  → You overpay for positions

Platt scaling fixes this by fitting a logistic regression on top of
XGBoost's output using held-out predictions and known outcomes.

After calibration:
  When the model says 0.72 → it actually means ~0.72
  When the model says 0.60 → it actually means ~0.60
  Return estimates become trustworthy

Also implements isotonic regression as an alternative calibration method.
The calibration engine in learning.py picks the one with lower Brier score.

Usage in the pipeline:
  1. After XGBoost training, fit Platt scaler on validation fold
  2. Wrap model.predict_single() with calibrated_prob()
  3. Return estimates now use true probability, not raw XGBoost score
"""

import json
import math
import logging
from pathlib import Path
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

log = logging.getLogger("predictos.calibration")

PLATT_FILE = "models/platt_scaler.json"


# ─── Platt scaler ─────────────────────────────────────────────────────────────

class PlattScaler:
    """
    Logistic regression calibration layer.
    Maps raw model scores → calibrated probabilities.

    Fitted using gradient descent on:
      loss = -y*log(σ(a*s + b)) - (1-y)*log(1 - σ(a*s + b))
    where s = raw score, y = true label, a and b are learned.
    """

    def __init__(self):
        self.a = 1.0    # slope
        self.b = 0.0    # intercept
        self.fitted = False
        self.fit_stats = {}

    def fit(self, scores: list[float], labels: list[int],
            lr: float = 0.01, n_iter: int = 1000) -> dict:
        """
        Fit Platt scaler using gradient descent.
        scores : raw model outputs (0–1)
        labels : true binary outcomes (0 or 1)
        """
        if len(scores) < 10:
            log.warning("Platt scaling: need ≥10 samples")
            return {"status": "insufficient_data"}

        n   = len(scores)
        a   = self.a
        b   = self.b

        for iteration in range(n_iter):
            grad_a = 0.0
            grad_b = 0.0
            loss   = 0.0

            for s, y in zip(scores, labels):
                p     = _sigmoid(a * s + b)
                p     = max(1e-7, min(1 - 1e-7, p))
                err   = p - y
                grad_a += err * s
                grad_b += err
                loss   += -y * math.log(p) - (1 - y) * math.log(1 - p)

            grad_a /= n
            grad_b /= n
            loss   /= n

            a -= lr * grad_a
            b -= lr * grad_b

            if iteration % 100 == 0:
                log.debug(f"  Iter {iteration}: loss={loss:.4f} a={a:.4f} b={b:.4f}")

        self.a      = a
        self.b      = b
        self.fitted = True

        # Compute calibrated Brier score
        raw_preds  = scores
        cal_preds  = [self.predict(s) for s in scores]
        raw_brier  = sum((p - y) ** 2 for p, y in zip(raw_preds,  labels)) / n
        cal_brier  = sum((p - y) ** 2 for p, y in zip(cal_preds,  labels)) / n

        self.fit_stats = {
            "n_samples":         n,
            "a":                 round(a, 4),
            "b":                 round(b, 4),
            "raw_brier":         round(raw_brier, 4),
            "calibrated_brier":  round(cal_brier, 4),
            "improvement":       round(raw_brier - cal_brier, 4),
            "fitted_at":         __import__("datetime").datetime.utcnow().isoformat(),
        }

        log.info(
            f"Platt scaler fitted: Brier {raw_brier:.4f} → {cal_brier:.4f} "
            f"(improvement: {raw_brier - cal_brier:+.4f})"
        )
        return self.fit_stats

    def predict(self, raw_score: float) -> float:
        """Transform raw score to calibrated probability."""
        if not self.fitted:
            return raw_score
        p = _sigmoid(self.a * raw_score + self.b)
        return round(max(0.001, min(0.999, p)), 4)

    def predict_batch(self, scores: list[float]) -> list[float]:
        return [self.predict(s) for s in scores]

    def save(self, path: str = PLATT_FILE):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "a": self.a,
                "b": self.b,
                "fitted": self.fitted,
                "fit_stats": self.fit_stats,
            }, f, indent=2)
        log.info(f"Platt scaler saved → {path}")

    def load(self, path: str = PLATT_FILE) -> bool:
        if not Path(path).exists():
            return False
        with open(path) as f:
            data = json.load(f)
        self.a         = data.get("a", 1.0)
        self.b         = data.get("b", 0.0)
        self.fitted    = data.get("fitted", False)
        self.fit_stats = data.get("fit_stats", {})
        log.info(f"Platt scaler loaded ← {path}")
        return True


# ─── Isotonic regression (non-parametric alternative) ─────────────────────────

class IsotonicCalibrator:
    """
    Non-parametric calibration using isotonic regression.
    More flexible than Platt scaling but needs more data (≥50 samples).

    Monotone increasing: maps scores to calibrated probabilities
    while preserving the ordering.
    """

    def __init__(self):
        self.mapping: list[tuple[float, float]] = []   # (score, prob) pairs
        self.fitted = False

    def fit(self, scores: list[float], labels: list[int]) -> dict:
        if len(scores) < 20:
            return {"status": "insufficient_data"}

        # Sort by score
        pairs  = sorted(zip(scores, labels), key=lambda x: x[0])
        xs, ys = zip(*pairs)
        xs     = list(xs)
        ys     = list(float(y) for y in ys)

        # Pool adjacent violators algorithm (PAVA)
        result = _pava(ys)

        # Store as lookup table (sample 20 representative points)
        step = max(1, len(xs) // 20)
        self.mapping = [(xs[i], result[i]) for i in range(0, len(xs), step)]
        self.mapping.append((xs[-1], result[-1]))
        self.fitted = True

        # Stats
        raw_brier = sum((s - y) ** 2 for s, y in zip(xs, ys)) / len(xs)
        cal_preds = [self.predict(s) for s in xs]
        cal_brier = sum((p - y) ** 2 for p, y in zip(cal_preds, ys)) / len(xs)

        return {
            "n_samples":        len(scores),
            "raw_brier":        round(raw_brier, 4),
            "calibrated_brier": round(cal_brier, 4),
            "improvement":      round(raw_brier - cal_brier, 4),
        }

    def predict(self, score: float) -> float:
        if not self.fitted or not self.mapping:
            return score
        # Linear interpolation in lookup table
        if score <= self.mapping[0][0]:
            return self.mapping[0][1]
        if score >= self.mapping[-1][0]:
            return self.mapping[-1][1]
        for i in range(len(self.mapping) - 1):
            x0, y0 = self.mapping[i]
            x1, y1 = self.mapping[i + 1]
            if x0 <= score <= x1:
                t = (score - x0) / (x1 - x0) if x1 > x0 else 0
                return round(y0 + t * (y1 - y0), 4)
        return score


def _pava(y: list[float]) -> list[float]:
    """Pool Adjacent Violators Algorithm for isotonic regression."""
    n      = len(y)
    result = list(y)
    i      = 0
    while i < n - 1:
        if result[i] > result[i + 1]:
            # Pool: replace with average
            j = i
            while j < n - 1 and result[j] > result[j + 1]:
                j += 1
            mean = sum(result[i:j + 1]) / (j - i + 1)
            for k in range(i, j + 1):
                result[k] = mean
            i = max(0, i - 1)
        else:
            i += 1
    return result


# ─── Auto-select best calibrator ──────────────────────────────────────────────

class AutoCalibrator:
    """
    Tries both Platt scaling and isotonic regression,
    selects the one with lower Brier score on held-out data.
    """

    def __init__(self):
        self.platt    = PlattScaler()
        self.isotonic = IsotonicCalibrator()
        self.method   = "none"
        self.loaded   = False
        self.stats    = {}

    def fit_and_select(self, scores: list[float], labels: list[int],
                       val_scores: list[float] = None,
                       val_labels: list[int]   = None) -> dict:
        """
        Fit both methods. Use validation set to pick the better one.
        If no validation set, use training set (less reliable but ok for small n).
        """
        vs = val_scores if val_scores else scores
        vl = val_labels if val_labels else labels

        platt_stats = self.platt.fit(scores, labels)
        iso_stats   = self.isotonic.fit(scores, labels)

        if platt_stats.get("status") == "insufficient_data":
            self.method = "none"
            return {"status": "insufficient_data"}

        # Evaluate on validation set
        platt_brier = _brier(vs, vl, self.platt.predict_batch(vs))
        iso_brier   = _brier(vs, vl, self.isotonic.predict_batch(vs)) \
                      if self.isotonic.fitted else float("inf")

        if platt_brier <= iso_brier or not self.isotonic.fitted:
            self.method = "platt"
            log.info(f"AutoCalibrator selected Platt scaling (Brier: {platt_brier:.4f} vs iso: {iso_brier:.4f})")
        else:
            self.method = "isotonic"
            log.info(f"AutoCalibrator selected isotonic (Brier: {iso_brier:.4f} vs platt: {platt_brier:.4f})")

        self.loaded = True
        self.stats  = {
            "method":        self.method,
            "platt_brier":   round(platt_brier, 4),
            "iso_brier":     round(iso_brier, 4) if self.isotonic.fitted else None,
            "platt_stats":   platt_stats,
            "iso_stats":     iso_stats,
        }
        self.platt.save()
        return self.stats

    def predict(self, raw_score: float) -> float:
        if not self.loaded:
            self.load()
        if self.method == "platt":
            return self.platt.predict(raw_score)
        elif self.method == "isotonic":
            return self.isotonic.predict(raw_score)
        return raw_score

    def load(self):
        if self.platt.load():
            self.method = "platt"
            self.loaded = True


def _brier(scores: list[float], labels: list[int],
           preds: list[float]) -> float:
    if not preds:
        return 0.25
    return sum((p - y) ** 2 for p, y in zip(preds, labels)) / len(preds)


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-max(-500, min(500, x))))


# ─── Singleton ────────────────────────────────────────────────────────────────

_CALIBRATOR: Optional[AutoCalibrator] = None

def get_calibrator() -> AutoCalibrator:
    global _CALIBRATOR
    if _CALIBRATOR is None:
        _CALIBRATOR = AutoCalibrator()
        _CALIBRATOR.load()
    return _CALIBRATOR
