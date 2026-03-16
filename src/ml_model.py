"""
PredictOS — ML Layer
XGBoost model trained on historical prediction market data.
Target: did YES resolve? Binary classification.
Walk-forward backtesting — no lookahead bias.
"""

import os
import json
import math
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Optional
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger("predictos.ml")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    log.warning("XGBoost not installed — ML layer will use logistic fallback")

# ─── Feature Engineering ──────────────────────────────────────────────────────

TOPIC_ENC = config.TOPIC_CATEGORIES

def build_features(market: dict, sentiment_score: float = 0.5) -> np.ndarray:
    """
    Build feature vector from a scored market dict.
    Must match config.FEATURE_COLUMNS order exactly.
    """
    yes_price    = market.get("yes_price", 0.5)
    volume_24h   = math.log1p(market.get("volume_24h", 0))
    vol_total    = math.log1p(market.get("volume_total", 0))
    liquidity    = math.log1p(market.get("liquidity", 0))
    days_to_res  = market.get("days_to_res", 30)
    category     = TOPIC_ENC.get(market.get("category", "other"), 7)

    # Momentum (price changes)
    yes_24h = market.get("yes_price_24h_ago", yes_price)
    yes_72h = market.get("yes_price_72h_ago", yes_price)
    momentum_24h = yes_price - yes_24h
    momentum_72h = yes_price - yes_72h

    # Odds rank in universe (placeholder — caller should pass in)
    odds_rank = market.get("_odds_rank", 0.5)  # percentile 0–1

    features = np.array([
        yes_price,        # odds_at_open
        volume_24h,       # volume (log)
        sentiment_score,  # sentiment score 0–1
        days_to_res,      # days to resolution
        category,         # topic category (encoded)
        momentum_24h,     # 24h odds delta
        momentum_72h,     # 72h odds delta
        liquidity,        # liquidity depth (log)
        odds_rank,        # RS equivalent
    ], dtype=np.float32)

    return features


def build_feature_matrix(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build X, y from a list of historical records.
    Each record: {market: {...}, sentiment_score: float, resolved_yes: bool}
    """
    X_list, y_list = [], []
    for rec in records:
        market = rec.get("market", {})
        sent   = rec.get("sentiment_score", 0.5)
        label  = int(rec.get("resolved_yes", False))
        try:
            feats = build_features(market, sent)
            X_list.append(feats)
            y_list.append(label)
        except Exception as e:
            log.debug(f"Feature build error: {e}")

    if not X_list:
        return np.array([]).reshape(0, len(config.FEATURE_COLUMNS)), np.array([])

    return np.vstack(X_list), np.array(y_list, dtype=np.float32)


# ─── Model ────────────────────────────────────────────────────────────────────

class PredictOSModel:
    """Wraps XGBoost (with logistic fallback)."""

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_importance = {}
        self.train_stats = {}

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        if not XGB_AVAILABLE or len(X) < config.TRAIN_MIN_SAMPLES:
            log.warning(f"Skipping XGB training (samples={len(X)}, xgb={XGB_AVAILABLE})")
            self.is_trained = False
            return {"skipped": True}

        dtrain = xgb.DMatrix(X, label=y, feature_names=config.FEATURE_COLUMNS)
        params = {
            "objective":        "binary:logistic",
            "eval_metric":      "logloss",
            "max_depth":        4,
            "eta":              0.1,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "seed":             42,
        }
        self.model = xgb.train(
            params, dtrain,
            num_boost_round=200,
            verbose_eval=False,
        )
        self.is_trained = True

        # Feature importance
        fi = self.model.get_score(importance_type="gain")
        self.feature_importance = {k: round(v, 2) for k, v in fi.items()}

        self.train_stats = {
            "n_samples": len(X),
            "n_pos":     int(y.sum()),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        log.info(f"XGBoost trained on {len(X)} samples")
        return self.train_stats

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return YES probability for each row."""
        if self.is_trained and self.model and XGB_AVAILABLE:
            dtest = xgb.DMatrix(X, feature_names=config.FEATURE_COLUMNS)
            return self.model.predict(dtest)
        else:
            # Logistic fallback: use odds + sentiment as simple prior
            # X[:, 0] = yes_price, X[:, 2] = sentiment_score
            if len(X) == 0:
                return np.array([])
            yes_price  = X[:, 0]
            sent_score = X[:, 2]
            # Weighted blend
            return 0.6 * yes_price + 0.4 * sent_score

    def predict_single(self, market: dict, sentiment_score: float = 0.5) -> float:
        """Return YES probability (0–1) for a single market."""
        X = build_features(market, sentiment_score).reshape(1, -1)
        return float(self.predict_proba(X)[0])

    def save(self, path: str = None):
        path = path or config.ML_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.is_trained and self.model:
            self.model.save_model(path)
            meta = {
                "feature_importance": self.feature_importance,
                "train_stats":        self.train_stats,
            }
            with open(path.replace(".json", "_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            log.info(f"Model saved → {path}")

    def load(self, path: str = None):
        path = path or config.ML_MODEL_PATH
        if not os.path.exists(path):
            log.info("No saved model found — will use fallback")
            return
        if not XGB_AVAILABLE:
            return
        try:
            self.model = xgb.Booster()
            self.model.load_model(path)
            self.is_trained = True
            meta_path = path.replace(".json", "_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                self.feature_importance = meta.get("feature_importance", {})
                self.train_stats        = meta.get("train_stats", {})
            log.info(f"Model loaded ← {path}")
        except Exception as e:
            log.error(f"Model load failed: {e}")


# ─── Walk-Forward Backtest ────────────────────────────────────────────────────

def walk_forward_backtest(records: list[dict],
                          n_folds: int = None) -> dict:
    """
    Temporal walk-forward cross-validation.
    Records must be sorted by resolution date (oldest first).
    No lookahead bias: train on past, test on future.
    """
    n_folds = n_folds or config.WALKFORWARD_FOLDS

    if len(records) < config.TRAIN_MIN_SAMPLES:
        return {"error": "insufficient_data", "n_records": len(records)}

    # Sort by resolution date
    records_sorted = sorted(
        records,
        key=lambda r: r.get("resolution_date", "1970-01-01")
    )

    fold_size = len(records_sorted) // (n_folds + 1)
    results   = []

    for fold in range(n_folds):
        train_end   = (fold + 1) * fold_size
        test_start  = train_end
        test_end    = test_start + fold_size

        train_recs = records_sorted[:train_end]
        test_recs  = records_sorted[test_start:test_end]

        if len(train_recs) < 50 or len(test_recs) < 10:
            continue

        X_train, y_train = build_feature_matrix(train_recs)
        X_test,  y_test  = build_feature_matrix(test_recs)

        m = PredictOSModel()
        m.train(X_train, y_train)
        preds = m.predict_proba(X_test)

        # Metrics
        binary_preds = (preds >= 0.5).astype(int)
        accuracy     = float((binary_preds == y_test).mean())
        brier        = float(np.mean((preds - y_test) ** 2))

        # Log loss
        eps = 1e-7
        logloss = float(-np.mean(
            y_test * np.log(preds + eps) +
            (1 - y_test) * np.log(1 - preds + eps)
        ))

        results.append({
            "fold":        fold + 1,
            "train_n":     len(train_recs),
            "test_n":      len(test_recs),
            "accuracy":    round(accuracy, 4),
            "brier_score": round(brier, 4),
            "log_loss":    round(logloss, 4),
        })
        log.info(f"Fold {fold+1}: accuracy={accuracy:.2%} brier={brier:.4f}")

    if not results:
        return {"error": "no_valid_folds"}

    avg_accuracy = np.mean([r["accuracy"] for r in results])
    avg_brier    = np.mean([r["brier_score"] for r in results])
    avg_logloss  = np.mean([r["log_loss"] for r in results])

    return {
        "n_folds":        len(results),
        "avg_accuracy":   round(float(avg_accuracy), 4),
        "avg_brier":      round(float(avg_brier), 4),
        "avg_log_loss":   round(float(avg_logloss), 4),
        "folds":          results,
        "ran_at":         datetime.now(timezone.utc).isoformat(),
    }


# ─── Category Win Rates ───────────────────────────────────────────────────────

def compute_category_win_rates(outcomes: list[dict]) -> dict:
    """
    Compute historical win rates per topic category from resolved picks.
    Used to calibrate Kelly sizing.
    """
    cat_results: dict[str, list[int]] = {}
    for outcome in outcomes:
        cat = outcome.get("category", "other")
        res = outcome.get("resolved_yes")
        if res is None:
            continue
        direction = outcome.get("direction", "YES")
        # Win = direction matches resolution
        win = (direction == "YES" and res) or (direction == "NO" and not res)
        cat_results.setdefault(cat, []).append(int(win))

    return {
        cat: round(sum(wins) / len(wins), 4)
        for cat, wins in cat_results.items()
        if len(wins) >= 5  # minimum sample
    }


# ─── Singleton ────────────────────────────────────────────────────────────────

_MODEL: Optional[PredictOSModel] = None

def get_model() -> PredictOSModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = PredictOSModel()
        _MODEL.load()
    return _MODEL
