"""
PredictOS — Closed-Loop Learning System

Every resolved trade is a training sample. The model retrains nightly.
Over time it learns which signals actually predict resolution outcomes,
not just which signals correlate with market prices.

Architecture:
  1. TrainingStore   — append-only ledger of every pick + its outcome
  2. CalibrationEngine — measures how accurate probability estimates are
  3. FeatureEvolution — tracks which features are gaining/losing importance
  4. ModelRetrainer  — triggers nightly retrain when new outcomes arrive
  5. LearningReport  — exposes calibration curves and accuracy metrics

Why 98% is the wrong target
────────────────────────────
Prediction markets price in ALL public information. If a model had 98%
accuracy, the market would immediately reprice to reflect that — making
the edge disappear. The correct target is:

  • Calibration accuracy: when model says 70%, it should resolve YES ~70%
  • Edge over market: model prob beats market prob as a predictor
  • Return accuracy: predicted EV should match realised EV over 50+ picks

Realistic accuracy curve on prediction markets:
  0–20 outcomes  : ~52–58% (noise dominates, learning begins)
  20–50 outcomes : ~58–63% (features start to matter)
  50–100 outcomes: ~62–67% (category-specific patterns emerge)
  100+ outcomes  : 65–70%  (ceiling — market is already semi-efficient)

A 65% win rate with proper Kelly sizing turns a $1,000 bankroll into
meaningful profits. 98% would require predicting the future.
"""

import json
import math
import logging
import os
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger("predictos.learning")

# ─── Paths ────────────────────────────────────────────────────────────────────

TRAINING_STORE_PATH  = "data/training_store.json"
CALIBRATION_PATH     = "data/calibration.json"
LEARNING_REPORT_PATH = "data/learning_report.json"
MODEL_PERF_PATH      = "data/model_performance.json"


def _load(path: str, default=None):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return default if default is not None else {}

def _save(path: str, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  1. TRAINING STORE — append-only ledger
# ══════════════════════════════════════════════════════════════════════════════

class TrainingStore:
    """
    Every pick that resolves becomes a training record.
    Fields captured at pick time → label added at resolution time.
    This is the ground truth that makes the model smarter.
    """

    def __init__(self, path: str = TRAINING_STORE_PATH):
        self.path = path
        self.records: list[dict] = _load(path, [])

    def add_pick(self, pick: dict, model_prob: float,
                 sentiment_score: float, composite_score: float) -> str:
        """
        Record a pick at signal time (before resolution).
        Returns a record ID.
        """
        record_id = f"{pick['id']}::{datetime.now(timezone.utc).date().isoformat()}"

        # Don't duplicate
        if any(r["record_id"] == record_id for r in self.records):
            return record_id

        record = {
            "record_id":       record_id,
            "market_id":       pick.get("id"),
            "source":          pick.get("source"),
            "question":        pick.get("question"),
            "category":        pick.get("category"),

            # Features at signal time (X)
            "yes_price_at_signal":  pick.get("yes_price"),
            "model_prob_at_signal": model_prob,
            "sentiment_score":      sentiment_score,
            "composite_score":      composite_score,
            "momentum_24h":         pick.get("yes_price", 0.5) - pick.get("yes_price_24h_ago", pick.get("yes_price", 0.5)),
            "momentum_72h":         pick.get("yes_price", 0.5) - pick.get("yes_price_72h_ago", pick.get("yes_price", 0.5)),
            "volume_24h":           pick.get("volume_24h", 0),
            "liquidity":            pick.get("liquidity", 0),
            "days_to_res_at_signal": pick.get("days_to_res", 30),
            "spread":               pick.get("spread"),
            "book_depth":           pick.get("book_depth", 0),
            "pillars":              pick.get("pillars", {}),
            "direction":            pick.get("direction", "YES"),
            "confidence":           pick.get("confidence"),
            "flags":                pick.get("flags", []),

            # Signal metadata
            "signal_date":     datetime.now(timezone.utc).date().isoformat(),
            "signal_ts":       datetime.now(timezone.utc).isoformat(),
            "end_date":        pick.get("end_date"),

            # Outcome (filled in at resolution)
            "resolved":        False,
            "resolved_yes":    None,
            "won":             None,
            "resolved_at":     None,
            "resolution_date": None,

            # P&L (filled in at resolution)
            "stake":           None,
            "gross_pnl":       None,
            "predicted_pnl":   None,
            "predicted_ev":    None,
        }
        self.records.append(record)
        self._save()
        return record_id

    def resolve_record(self, market_id: str, resolved_yes: bool,
                       stake: float = None, predicted_pnl: float = None,
                       predicted_ev: float = None):
        """Mark a record as resolved with outcome. Computes actual P&L."""
        for r in self.records:
            if r["market_id"] == market_id and not r["resolved"]:
                direction  = r.get("direction", "YES")
                won        = (direction == "YES" and resolved_yes) or \
                             (direction == "NO" and not resolved_yes)

                r["resolved"]        = True
                r["resolved_yes"]    = resolved_yes
                r["won"]             = won
                r["resolved_at"]     = datetime.now(timezone.utc).isoformat()
                r["resolution_date"] = datetime.now(timezone.utc).date().isoformat()

                if stake:
                    entry_price = r.get("yes_price_at_signal", 0.5)
                    if direction == "NO":
                        entry_price = 1 - entry_price
                    shares = stake / max(entry_price, 0.001)
                    r["stake"]         = stake
                    r["gross_pnl"]     = round(shares - stake if won else -stake, 2)
                    r["predicted_pnl"] = predicted_pnl
                    r["predicted_ev"]  = predicted_ev

        self._save()

    def get_resolved(self) -> list[dict]:
        return [r for r in self.records if r["resolved"]]

    def get_unresolved(self) -> list[dict]:
        return [r for r in self.records if not r["resolved"]]

    def to_ml_format(self) -> tuple[list[dict], list[int]]:
        """
        Export as (feature_dicts, labels) for ML training.
        label = 1 if the direction we picked was correct.
        """
        resolved = self.get_resolved()
        feature_dicts, labels = [], []
        for r in resolved:
            feature_dicts.append({
                "yes_price":       r["yes_price_at_signal"],
                "volume_24h":      r["volume_24h"],
                "sentiment_score": r["sentiment_score"],
                "days_to_res":     r["days_to_res_at_signal"],
                "category":        config.TOPIC_CATEGORIES.get(r.get("category", "other"), 7),
                "momentum_24h":    r["momentum_24h"],
                "momentum_72h":    r["momentum_72h"],
                "liquidity":       r["liquidity"],
                "composite_score": r["composite_score"],
                "spread":          r.get("spread") or 0.05,
                "book_depth":      r.get("book_depth") or 0,
                "_odds_rank":      0.5,  # filled by caller
            })
            labels.append(int(r["won"]))
        return feature_dicts, labels

    def _save(self):
        _save(self.path, self.records)


# ══════════════════════════════════════════════════════════════════════════════
#  2. CALIBRATION ENGINE — measures how accurate probabilities are
# ══════════════════════════════════════════════════════════════════════════════

class CalibrationEngine:
    """
    Checks: when we say 70% probability, do things resolve YES ~70% of the time?
    This is the correct accuracy metric for prediction markets.

    Perfect calibration = the diagonal line on a calibration plot.
    Overconfidence = curve below the diagonal.
    Underconfidence = curve above the diagonal.
    """

    N_BUCKETS = 10  # 0–10%, 10–20%, ... 90–100%

    def compute(self, store: TrainingStore) -> dict:
        resolved = store.get_resolved()
        if len(resolved) < 5:
            return {
                "status":         "insufficient_data",
                "n_resolved":     len(resolved),
                "brier_score":    None,
                "log_loss":       None,
                "calibration_error": 0.15,
                "by_bucket":      [],
                "by_category":    {},
                "accuracy_curve": [],
                "message": "Need at least 5 resolved outcomes to calibrate. Keep tracking!"
            }

        probs   = np.array([r["model_prob_at_signal"] for r in resolved])
        actuals = np.array([int(r["resolved_yes"]) for r in resolved])

        # ── Brier score (lower = better, 0 = perfect) ────────────────────────
        # Measures probability calibration quality
        brier = float(np.mean((probs - actuals) ** 2))

        # ── Log loss ─────────────────────────────────────────────────────────
        eps    = 1e-7
        logloss = float(-np.mean(
            actuals * np.log(probs + eps) +
            (1 - actuals) * np.log(1 - probs + eps)
        ))

        # ── Calibration buckets ───────────────────────────────────────────────
        buckets = []
        for b in range(self.N_BUCKETS):
            lo, hi = b / self.N_BUCKETS, (b + 1) / self.N_BUCKETS
            mask   = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            mean_pred   = float(probs[mask].mean())
            actual_freq = float(actuals[mask].mean())
            buckets.append({
                "bucket":       f"{lo:.0%}–{hi:.0%}",
                "mean_pred":    round(mean_pred, 3),
                "actual_freq":  round(actual_freq, 3),
                "n":            int(mask.sum()),
                "error":        round(abs(mean_pred - actual_freq), 3),
            })

        # Expected calibration error (ECE): weighted avg bucket error
        total_n = sum(b["n"] for b in buckets)
        ece     = sum(b["n"] / total_n * b["error"] for b in buckets) if total_n > 0 else 0.15

        # ── Per-category calibration ──────────────────────────────────────────
        by_category = {}
        cats = set(r.get("category", "other") for r in resolved)
        for cat in cats:
            cat_recs = [r for r in resolved if r.get("category") == cat]
            if len(cat_recs) < 3:
                continue
            cat_probs   = np.array([r["model_prob_at_signal"] for r in cat_recs])
            cat_actuals = np.array([int(r["resolved_yes"]) for r in cat_recs])
            cat_brier   = float(np.mean((cat_probs - cat_actuals) ** 2))

            # Direction win rate (did our pick direction win?)
            direction_wins = [int(r["won"]) for r in cat_recs]
            win_rate       = sum(direction_wins) / len(direction_wins)

            by_category[cat] = {
                "n":           len(cat_recs),
                "win_rate":    round(win_rate, 4),
                "brier_score": round(cat_brier, 4),
                "cal_error":   round(float(np.mean(abs(cat_probs - cat_actuals))), 4),
            }

        # ── Accuracy over time (rolling 10-pick window) ───────────────────────
        sorted_resolved = sorted(resolved, key=lambda r: r.get("signal_ts", ""))
        acc_curve = []
        window    = 10
        for i in range(window, len(sorted_resolved) + 1):
            window_recs = sorted_resolved[i - window:i]
            wr = sum(int(r["won"]) for r in window_recs) / window
            acc_curve.append({
                "pick_n":   i,
                "win_rate": round(wr, 4),
                "date":     window_recs[-1].get("signal_date", ""),
            })

        # ── P&L accuracy ──────────────────────────────────────────────────────
        # How accurate were predicted P&L vs actual P&L?
        pnl_records = [r for r in resolved if r.get("gross_pnl") is not None
                       and r.get("predicted_pnl") is not None]
        pnl_accuracy = None
        if pnl_records:
            errors      = [abs(r["gross_pnl"] - r["predicted_pnl"]) for r in pnl_records]
            mean_error  = sum(errors) / len(errors)
            avg_stake   = sum(r["stake"] for r in pnl_records) / len(pnl_records)
            pnl_accuracy = {
                "n":                len(pnl_records),
                "mean_abs_error":   round(mean_error, 2),
                "mean_abs_error_pct": round(mean_error / avg_stake * 100, 1),
            }

        result = {
            "status":             "ok",
            "n_resolved":         len(resolved),
            "brier_score":        round(brier, 4),
            "log_loss":           round(logloss, 4),
            "calibration_error":  round(ece, 4),
            "overall_win_rate":   round(sum(int(r["won"]) for r in resolved) / len(resolved), 4),
            "by_bucket":          buckets,
            "by_category":        by_category,
            "accuracy_curve":     acc_curve,
            "pnl_accuracy":       pnl_accuracy,
            "computed_at":        datetime.now(timezone.utc).isoformat(),
            "interpretation":     _interpret_calibration(brier, ece, len(resolved)),
        }

        _save(CALIBRATION_PATH, result)
        return result


def _interpret_calibration(brier: float, ece: float, n: int) -> dict:
    """Human-readable interpretation of calibration metrics."""
    if n < 20:
        quality = "bootstrapping"
        note    = f"Only {n} outcomes — keep going. The model needs ~50 to calibrate well."
    elif brier < 0.15 and ece < 0.05:
        quality = "excellent"
        note    = "Well-calibrated. Return estimates are reliable."
    elif brier < 0.20 and ece < 0.08:
        quality = "good"
        note    = "Good calibration. Return estimates are approximately correct."
    elif brier < 0.25 and ece < 0.12:
        quality = "fair"
        note    = "Fair calibration. Use wide confidence intervals for return estimates."
    else:
        quality = "poor"
        note    = "Poor calibration — model probabilities are off. Return estimates are uncertain."

    # Reference: random guesser has Brier = 0.25; perfect = 0.0
    skill_pct = max(0, (0.25 - brier) / 0.25 * 100)

    return {
        "quality":          quality,
        "skill_score_pct":  round(skill_pct, 1),
        "note":             note,
        "brier_benchmark":  "0.25 = random, 0.0 = perfect",
        "realistic_ceiling": "0.12–0.18 on liquid prediction markets",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  3. MODEL RETRAINER — triggers retrain when enough new data arrives
# ══════════════════════════════════════════════════════════════════════════════

class ModelRetrainer:
    """
    Monitors the training store. When enough new resolved outcomes
    arrive since the last retrain, triggers a fresh XGBoost training run.

    Retrain triggers:
      - First retrain: 20 resolved outcomes
      - Subsequent: every 10 new outcomes, or weekly
    """

    MIN_FOR_FIRST_TRAIN   = 20
    RETRAIN_EVERY_N_NEW   = 10

    def should_retrain(self, store: TrainingStore) -> tuple[bool, str]:
        resolved   = store.get_resolved()
        n_resolved = len(resolved)

        perf = _load(MODEL_PERF_PATH, {})
        last_n_at_train = perf.get("n_at_last_train", 0)
        new_since_train = n_resolved - last_n_at_train

        if n_resolved < self.MIN_FOR_FIRST_TRAIN:
            return False, f"Need {self.MIN_FOR_FIRST_TRAIN - n_resolved} more outcomes before first retrain"

        if new_since_train >= self.RETRAIN_EVERY_N_NEW:
            return True, f"{new_since_train} new outcomes since last retrain"

        return False, f"Only {new_since_train} new outcomes since last retrain (need {self.RETRAIN_EVERY_N_NEW})"

    def retrain(self, store: TrainingStore) -> dict:
        """Full retrain + walk-forward validation cycle."""
        from src.ml_model import PredictOSModel, build_feature_matrix, walk_forward_backtest

        feature_dicts, labels = store.to_ml_format()
        if len(feature_dicts) < self.MIN_FOR_FIRST_TRAIN:
            return {"status": "insufficient_data"}

        # Build numpy arrays from stored feature dicts
        from src.ml_model import build_features
        import numpy as np

        X_list = []
        for fd in feature_dicts:
            market_stub = {
                "yes_price":        fd["yes_price"],
                "volume_24h":       fd["volume_24h"],
                "liquidity":        fd["liquidity"],
                "days_to_res":      fd["days_to_res"],
                "category":         _cat_from_enc(fd["category"]),
                "yes_price_24h_ago": fd["yes_price"] - fd["momentum_24h"],
                "yes_price_72h_ago": fd["yes_price"] - fd["momentum_72h"],
                "_odds_rank":        fd.get("_odds_rank", 0.5),
            }
            X_list.append(build_features(market_stub, fd["sentiment_score"]))

        X = np.vstack(X_list)
        y = np.array(labels, dtype=np.float32)

        # Walk-forward backtest first (validation)
        records_for_wf = [
            {"market": _fd_to_market(fd), "sentiment_score": fd["sentiment_score"],
             "resolved_yes": bool(labels[i]),
             "resolution_date": store.get_resolved()[i].get("resolution_date", "2020-01-01")}
            for i, fd in enumerate(feature_dicts)
        ]
        wf_result = walk_forward_backtest(records_for_wf)

        # Train final model on ALL data
        model = PredictOSModel()
        train_stats = model.train(X, y)
        model.save()

        perf = {
            "n_at_last_train":   len(labels),
            "train_stats":       train_stats,
            "walkforward":       wf_result,
            "feature_importance": model.feature_importance,
            "retrained_at":      datetime.now(timezone.utc).isoformat(),
        }
        _save(MODEL_PERF_PATH, perf)

        log.info(f"Retrain complete: {len(labels)} samples, "
                 f"WF accuracy={wf_result.get('avg_accuracy', 0):.2%}")
        return perf


def _cat_from_enc(enc: int) -> str:
    rev = {v: k for k, v in config.TOPIC_CATEGORIES.items()}
    return rev.get(enc, "other")

def _fd_to_market(fd: dict) -> dict:
    return {
        "yes_price":         fd["yes_price"],
        "volume_24h":        fd["volume_24h"],
        "liquidity":         fd["liquidity"],
        "days_to_res":       fd["days_to_res"],
        "category":          _cat_from_enc(fd["category"]),
        "yes_price_24h_ago": fd["yes_price"] - fd["momentum_24h"],
        "yes_price_72h_ago": fd["yes_price"] - fd["momentum_72h"],
        "_odds_rank":        0.5,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  4. LEARNING REPORT — daily summary of model improvement
# ══════════════════════════════════════════════════════════════════════════════

def generate_learning_report(store: TrainingStore,
                              calibration: dict) -> dict:
    """
    Full learning system status. Written to data/learning_report.json
    and consumed by the dashboard.
    """
    resolved   = store.get_resolved()
    unresolved = store.get_unresolved()

    # Overall win rate trend (last 10 vs prior 10)
    sorted_res = sorted(resolved, key=lambda r: r.get("signal_ts", ""))
    recent_10  = sorted_res[-10:] if len(sorted_res) >= 10 else sorted_res
    prior_10   = sorted_res[-20:-10] if len(sorted_res) >= 20 else []

    recent_wr = sum(int(r["won"]) for r in recent_10) / max(len(recent_10), 1)
    prior_wr  = sum(int(r["won"]) for r in prior_10)  / max(len(prior_10),  1) if prior_10 else None

    trend = None
    if prior_wr is not None:
        delta = recent_wr - prior_wr
        trend = "improving" if delta > 0.03 else ("declining" if delta < -0.03 else "stable")

    # Feature importance (from last retrain)
    perf = _load(MODEL_PERF_PATH, {})
    fi   = perf.get("feature_importance", {})

    # P&L accuracy
    pnl_records = [r for r in resolved if r.get("gross_pnl") is not None
                   and r.get("predicted_pnl") is not None]
    pnl_mae = None
    if pnl_records:
        errors  = [abs(r["gross_pnl"] - r["predicted_pnl"]) for r in pnl_records]
        pnl_mae = round(sum(errors) / len(errors), 2)

    # Prediction accuracy: was model_prob a good predictor?
    # Compute accuracy at various thresholds
    threshold_accuracy = {}
    if len(resolved) >= 10:
        for threshold in [0.55, 0.60, 0.65, 0.70]:
            high_conf = [r for r in resolved if r["model_prob_at_signal"] >= threshold]
            if high_conf:
                wr = sum(int(r["resolved_yes"]) for r in high_conf) / len(high_conf)
                threshold_accuracy[f">={threshold:.0%}"] = {
                    "n": len(high_conf),
                    "win_rate": round(wr, 4),
                }

    report = {
        "summary": {
            "total_tracked":     len(resolved) + len(unresolved),
            "total_resolved":    len(resolved),
            "total_unresolved":  len(unresolved),
            "overall_win_rate":  round(recent_wr, 4) if resolved else 0,
            "win_trend":         trend,
            "recent_10_wr":      round(recent_wr, 4),
            "prior_10_wr":       round(prior_wr, 4) if prior_wr else None,
        },
        "model_quality": {
            "brier_score":        calibration.get("brier_score"),
            "log_loss":           calibration.get("log_loss"),
            "calibration_error":  calibration.get("calibration_error"),
            "interpretation":     calibration.get("interpretation", {}),
            "last_retrained_at":  perf.get("retrained_at"),
            "n_at_last_train":    perf.get("n_at_last_train"),
            "walkforward_accuracy": perf.get("walkforward", {}).get("avg_accuracy"),
        },
        "feature_importance":    fi,
        "threshold_accuracy":    threshold_accuracy,
        "pnl_accuracy": {
            "n_with_pnl_data":  len(pnl_records),
            "mean_abs_error":   pnl_mae,
            "note": "Difference between predicted and actual P&L per trade",
        },
        "calibration_buckets":   calibration.get("by_bucket", []),
        "accuracy_curve":        calibration.get("accuracy_curve", []),
        "by_category":           calibration.get("by_category", {}),
        "honest_accuracy_note": {
            "realistic_ceiling":   "65–70% win rate on liquid prediction markets",
            "how_to_read_brier":   "0.0 = perfect, 0.25 = random coin flip",
            "what_drives_returns": "Calibration quality × Kelly sizing × edge detection",
            "why_not_98pct":       (
                "Prediction markets price in all public information. "
                "98% accuracy would require predicting the future. "
                "A well-calibrated 65% win rate with proper Kelly sizing "
                "is genuinely profitable and what professional traders target."
            ),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    _save(LEARNING_REPORT_PATH, report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
#  5. MASTER LOOP — called from main.py after resolution step
# ══════════════════════════════════════════════════════════════════════════════

def run_learning_cycle(newly_resolved_picks: list[dict] = None) -> dict:
    """
    Main entry point called nightly by the pipeline.
    1. Load training store
    2. Resolve any newly-closed picks
    3. Calibrate model
    4. Retrain if threshold met
    5. Generate learning report
    """
    store      = TrainingStore()
    calibrator = CalibrationEngine()
    retrainer  = ModelRetrainer()

    # Mark newly resolved picks in the store
    if newly_resolved_picks:
        for pick in newly_resolved_picks:
            store.resolve_record(
                market_id    = pick.get("id"),
                resolved_yes = pick.get("resolved_yes", False),
                stake        = pick.get("sizing", {}).get("position_usd"),
                predicted_pnl= _compute_predicted_pnl(pick),
                predicted_ev = pick.get("sizing", {}).get("position_usd", 0) *
                               pick.get("ev_pct", 0) / 100 if pick.get("ev_pct") else None,
            )

    # Calibrate
    calibration = calibrator.compute(store)

    # Retrain check
    should, reason = retrainer.should_retrain(store)
    retrain_result = None
    if should:
        log.info(f"Retraining: {reason}")
        retrain_result = retrainer.retrain(store)
    else:
        log.info(f"No retrain needed: {reason}")

    # Learning report
    report = generate_learning_report(store, calibration)

    return {
        "calibration":    calibration,
        "retrained":      should,
        "retrain_result": retrain_result,
        "report":         report,
        "n_resolved":     len(store.get_resolved()),
        "n_unresolved":   len(store.get_unresolved()),
    }


def record_new_picks(picks: dict, scored_markets: list[dict],
                     sentiment_scores: dict, model_probs: dict):
    """
    Record today's picks in the training store BEFORE they resolve.
    Called from main.py after picks are generated.
    """
    store = TrainingStore()
    all_picks = (
        picks.get("yes_picks", []) +
        picks.get("no_picks", []) +
        picks.get("macro_plays", [])
    )
    recorded = 0
    for pick in all_picks:
        mid = pick.get("id", "")
        store.add_pick(
            pick            = pick,
            model_prob      = model_probs.get(mid, pick.get("model_prob", 0.5)),
            sentiment_score = sentiment_scores.get(mid, 0.5),
            composite_score = pick.get("composite", 0),
        )
        recorded += 1

    log.info(f"Recorded {recorded} new picks in training store "
             f"(total: {len(store.records)})")
    return recorded


def _compute_predicted_pnl(pick: dict) -> Optional[float]:
    """Compute what we predicted the P&L would be at signal time."""
    sizing = pick.get("sizing", {})
    stake  = sizing.get("position_usd", 0)
    if not stake:
        return None
    yes_price = pick.get("yes_price", 0.5)
    direction = pick.get("direction", "YES")
    entry     = yes_price if direction == "YES" else (1 - yes_price)
    shares    = stake / max(entry, 0.001)
    return round(shares - stake, 2)  # net profit if win
