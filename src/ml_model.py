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
            # Fallback: structured multi-signal model before XGBoost trains.
            #
            # DESIGN PRINCIPLE: The spec requires market_price ≤ model_prob × 0.5
            # This means model_prob must be ≥ 2× market_price for a buy signal.
            # A fallback that just adds ±20% to market price can NEVER satisfy this.
            #
            # Solution: treat sentiment + momentum as votes on the BASE RATE,
            # not adjustments to the market price. The fallback outputs a probability
            # anchored to the sentiment signal, tempered by market price as a prior.
            #
            # Strong bullish sentiment (0.85) on a 30% market → model says ~65%
            # Neutral sentiment (0.50) → model closely tracks market price
            # Strong bearish sentiment (0.15) → model says much lower than market
            if len(X) == 0:
                return np.array([])

            yes_price    = X[:, 0]   # market probability
            volume_24h   = X[:, 1]   # log-scaled volume (log10 of USD)
            sent_score   = X[:, 2]   # 0-1 sentiment (0.5=neutral)
            days_to_res  = X[:, 3]   # days to resolution
            momentum_24h = X[:, 5]   # 24h price change
            momentum_72h = X[:, 6]   # 72h price change

            # Signal strength: how far from neutral is each signal?
            sent_strength  = np.abs(sent_score - 0.5) * 2   # 0-1
            mom_strength   = np.clip(np.abs(momentum_24h) * 20, 0, 1)  # 0-1

            # Signal direction: are sentiment and momentum agreeing?
            sent_bullish = sent_score > 0.5
            mom_bullish  = momentum_24h > 0.005
            signals_agree = (sent_bullish == mom_bullish) | (mom_strength < 0.1)

            # Confluence multiplier: 1.5× when signals agree, 0.6× when they conflict
            confluence = np.where(signals_agree, 1.5, 0.6)

            # Base sentiment probability: maps sentiment score to a probability
            # sent=0.85 → base_prob=0.70, sent=0.5 → 0.50, sent=0.15 → 0.30
            sent_base_prob = 0.30 + sent_score * 0.40   # range: 0.30-0.70

            # Momentum boost: consistent momentum adds up to ±0.15
            mom_consistent = np.where(
                np.sign(momentum_24h) == np.sign(momentum_72h),
                momentum_24h + momentum_72h * 0.4,
                momentum_24h * 0.3
            )
            mom_boost = np.clip(mom_consistent * 3.0, -0.15, 0.15)

            # Volume weight: high volume = market is more informed = trust market price more
            # vol log10($2k)=3.3, log10($50k)=4.7, log10($500k)=5.7
            vol_market_weight = np.clip((volume_24h - 3.0) / 3.0, 0.1, 0.8)

            # Raw signal probability (from sentiment + momentum)
            raw_signal_prob = np.clip(sent_base_prob + mom_boost, 0.05, 0.95)

            # Apply confluence multiplier toward signal
            # When signals agree strongly: move further from market toward signal
            signal_distance = raw_signal_prob - yes_price
            amplified_signal = yes_price + signal_distance * confluence

            # Blend: weight between market price and signal based on volume
            # Low volume → trust signal more (market may be inefficient)
            # High volume → trust market more (market is informed)
            signal_weight = (1.0 - vol_market_weight) * sent_strength * confluence * 0.6
            signal_weight = np.clip(signal_weight, 0.05, 0.65)

            model_prob = (1 - signal_weight) * yes_price + signal_weight * amplified_signal

            # Apply momentum push
            model_prob = model_prob + mom_boost * (1 - vol_market_weight) * 0.5

            return np.clip(model_prob, 0.05, 0.95)

    def predict_single(self, market: dict, sentiment_score: float = 0.5) -> float:
        """Return YES probability (0–1) for a single market."""
        X = build_features(market, sentiment_score).reshape(1, -1)
        return float(self.predict_proba(X)[0])

    def compute_confidence(self, market: dict, model_prob: float,
                           sentiment_score: float = 0.5) -> float:
        """
        Compute ML confidence (0-1) based on signal agreement.
        High confidence = multiple signals agree on direction.
        Low confidence = signals conflict or model is untrained.

        This replaces the naive abs(prob - 0.5) * 2 calculation.
        """
        yes_price = market.get("yes_price", 0.5)
        momentum_24h = market.get("yes_price", yes_price) - market.get("yes_price_24h_ago", yes_price)
        momentum_72h = market.get("yes_price", yes_price) - market.get("yes_price_72h_ago", yes_price)

        edge = model_prob - yes_price  # positive = model sees YES edge

        # Signal 1: How far is model from market? (larger = more confident)
        edge_conf = min(abs(edge) / 0.30, 1.0)  # 30% edge = max confidence

        # Signal 2: Does sentiment agree with model direction?
        sent_agrees = (sentiment_score > 0.5 and edge > 0) or \
                      (sentiment_score < 0.5 and edge < 0) or \
                      (abs(sentiment_score - 0.5) < 0.05)  # neutral = no disagreement
        sent_conf = 0.8 if sent_agrees else 0.3

        # Signal 3: Does momentum agree with model direction?
        mom_agrees = (momentum_24h > 0 and edge > 0) or \
                     (momentum_24h < 0 and edge < 0) or \
                     abs(momentum_24h) < 0.01  # flat = no disagreement
        mom_conf = 0.8 if mom_agrees else 0.3

        # Signal 4: Is model trained? Trained model gets base confidence boost
        trained_boost = 0.3 if self.is_trained else 0.0

        # Combined: weighted average of signals
        raw_conf = (
            0.40 * edge_conf +
            0.25 * sent_conf +
            0.20 * mom_conf +
            0.15 * (0.5 + trained_boost)
        )
        return round(min(raw_conf, 1.0), 4)

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
