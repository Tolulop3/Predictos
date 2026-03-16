"""
PredictOS — Scoring Engine
Scores each open market 0–100 across 5 pillars:
  Momentum · Volume · Sentiment · Edge · Decay
Then applies signal filters and produces final ranked list.
"""

import math
import logging
from datetime import datetime, timezone
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger("predictos.score")

# ─── Momentum Pillar ──────────────────────────────────────────────────────────

def score_momentum(market: dict, history: list[dict] = None) -> float:
    """
    0–100. Measures directional movement in YES price over 24h and 72h.
    Analogous to price momentum in InvestOS.
    """
    yes_now = market.get("yes_price", 0.5)

    # Fallback: use stored history snapshots
    yes_24h = market.get("yes_price_24h_ago", yes_now)
    yes_72h = market.get("yes_price_72h_ago", yes_now)

    delta_24h = yes_now - yes_24h  # e.g. +0.05 = moved 5% toward YES
    delta_72h = yes_now - yes_72h

    # Apply trend half-life decay to 72h signal
    decay_72h = math.exp(-math.log(2) * 2 / config.TREND_HALF_LIFE_DAYS)

    combined_delta = 0.6 * delta_24h + 0.4 * delta_72h * decay_72h

    # Map to 0–100: ±0.15 movement = ±50 score change
    raw = 50 + combined_delta * (50 / 0.15)
    return max(0.0, min(100.0, raw))


# ─── Volume Pillar ────────────────────────────────────────────────────────────

def score_volume(market: dict, universe: list[dict] = None) -> float:
    """
    0–100. Ranks market volume+liquidity vs universe (like avg_volume in stocks).
    """
    vol_24h   = market.get("volume_24h", 0)
    liquidity = market.get("liquidity", 0)

    # Combined volume score
    combined = vol_24h * 0.7 + liquidity * 0.3

    # Log scale: 0 → 0, $10k → 50, $1M → 85, $10M → 100
    if combined <= 0:
        return 0.0
    log_score = math.log10(max(combined, 1)) * (100 / 7)  # $10M = log10 7
    return max(0.0, min(100.0, log_score))


# ─── Sentiment Pillar ─────────────────────────────────────────────────────────

def score_sentiment(sentiment_result: dict) -> float:
    """
    0–100. Converts news sentiment score to pillar score.
    0.5 sentiment → 50 (neutral), 1.0 → 100 (fully YES), 0.0 → 0 (fully NO).
    """
    s = sentiment_result.get("score", 0.5)
    # Boost when many articles fire
    art_count = sentiment_result.get("article_count", 0)
    coverage_boost = min(art_count / 20, 1.0) * 10  # up to +10 for high coverage

    base = s * 100
    return max(0.0, min(100.0, base + (coverage_boost if s > 0.5 else -coverage_boost)))


# ─── Edge Pillar ──────────────────────────────────────────────────────────────

def score_edge(market: dict, model_prob: float) -> tuple[float, float]:
    """
    0–100. Measures mispricing between market price and model probability.
    Returns (pillar_score, edge_pct) where edge_pct = model_prob - market_yes_price.
    Analogous to Graham margin-of-safety.
    """
    market_prob = market.get("yes_price", 0.5)
    edge = model_prob - market_prob  # positive = market underprices YES

    edge_abs = abs(edge)

    # Map: 0% edge → 50 (neutral), 20%+ edge → 100 (YES) or 0 (NO)
    if edge > 0:
        score = 50 + (edge_abs / 0.20) * 50
    else:
        score = 50 - (edge_abs / 0.20) * 50

    return max(0.0, min(100.0, score)), round(edge, 4)


# ─── Decay Pillar ─────────────────────────────────────────────────────────────

def score_decay(market: dict, momentum_score: float) -> float:
    """
    0–100. Closer resolution WITH a clear trend = higher score.
    Sweet spot: 3–21 days. Steep penalty beyond 30 days.
    """
    days = market.get("days_to_res", 30)

    if days < 0.1:
        return 0.0   # already closed
    if days < 2:
        return 30.0  # too close (binary event risk)

    # Sweet spot: 3–21 days → high score
    if days <= 7:
        proximity = 85.0
    elif days <= 14:
        proximity = 70.0
    elif days <= 21:
        proximity = 55.0
    elif days <= 30:
        proximity = 35.0
    elif days <= 45:
        proximity = 15.0
    elif days <= 60:
        proximity = 5.0
    else:
        proximity = 0.0

    # Trend confirmation: momentum amplifies decay score
    trend_multiplier = 0.5 + (momentum_score / 100) * 0.5  # 0.5–1.0

    return max(0.0, min(100.0, proximity * trend_multiplier))


# ─── Composite Score ──────────────────────────────────────────────────────────

def compute_composite(pillar_scores: dict) -> float:
    """Weighted average of 5 pillars → 0–100."""
    w = config.PILLAR_WEIGHTS
    return (
        w["momentum"]  * pillar_scores["momentum"]  +
        w["volume"]    * pillar_scores["volume"]     +
        w["sentiment"] * pillar_scores["sentiment"]  +
        w["edge"]      * pillar_scores["edge"]       +
        w["decay"]     * pillar_scores["decay"]
    )


# ─── Signal Quality Filters ───────────────────────────────────────────────────

def apply_filters(market: dict, sentiment: dict, model_prob: float,
                  ml_confidence: float) -> tuple[bool, list[str]]:
    """
    Returns (passes, [list of flags]).
    Mirrors InvestOS filter logic.
    """
    flags = []
    passes = True

    # 0. Manifold = play-money only — never a real pick, used for sentiment signal only
    if market.get("sentiment_only") or market.get("source") == "manifold":
        flags.append("sentiment_only")
        return False, flags

    # 1. Binary event risk: <48h to resolution
    if market.get("days_to_res", 99) < config.MIN_RESOLUTION_HOURS / 24:
        flags.append("binary_event_risk")
        passes = False

    # 1b. Too far out: >60 days general, >30 days for sports (season markets = noisy)
    days     = market.get("days_to_res", 0)
    category = market.get("category", "other")
    max_days = config.MAX_RESOLUTION_DAYS_SPORTS if category == "sports" else config.MAX_RESOLUTION_DAYS
    if days > max_days:
        flags.append("too_far_out")
        passes = False

    # 2. Liquidity filter
    if market.get("volume_24h", 0) < config.MIN_DAILY_VOLUME_USD:
        flags.append("low_liquidity")
        if market.get("source") != "manifold":  # Manifold is play-money, exempt
            passes = False

    # 3. Probability bounds — reject extreme longshots (<5%) and near-certainties (>95%)
    #    These produce unstable Kelly sizes and unreliable sentiment signals
    yes_price = market.get("yes_price", 0.5)
    if yes_price < config.MIN_MARKET_PROB or yes_price > config.MAX_MARKET_PROB:
        flags.append("extreme_probability")
        passes = False

    # 4. Graham equivalent: must have >8% edge
    edge_abs = abs(model_prob - market.get("yes_price", 0.5))
    if edge_abs < config.GRAHAM_EDGE_THRESHOLD:
        flags.append("insufficient_edge")
        passes = False

    # 4. Factor conflict: high momentum + low volume = thin market
    momentum_score = market.get("_momentum_score", 50)
    if momentum_score > 75 and market.get("volume_24h", 0) < 5000:
        flags.append("thin_market_momentum_conflict")
        passes = False

    # 5. ML confidence below threshold
    if ml_confidence < config.ML_CONFIDENCE_THRESHOLD:
        flags.append("low_ml_confidence")
        # Don't fail, but caller should reduce size

    # 6. No news coverage (can't compute sentiment edge)
    if sentiment.get("article_count", 0) == 0:
        flags.append("no_news_coverage")

    return passes, flags


# ─── Kelly Position Sizing ────────────────────────────────────────────────────

def kelly_size(win_prob: float, odds: float,
               bankroll: float = None,
               category_win_rate: float = None) -> dict:
    """
    Kelly Criterion: f* = (bp - q) / b
    where b = decimal odds - 1, p = win prob, q = 1 - p.
    Returns recommended position in USD.
    """
    import config as cfg
    bankroll = bankroll or cfg.BANKROLL

    # Use category win rate if better calibrated
    p = category_win_rate if category_win_rate else win_prob
    p = max(0.01, min(0.99, p))

    # Binary market: win = 1/price - 1
    b = (1 / max(odds, 0.01)) - 1
    q = 1 - p

    kelly_f = (b * p - q) / b if b > 0 else 0
    kelly_f = max(0.0, kelly_f)

    # Fractional Kelly (conservative)
    frac_kelly = kelly_f * cfg.KELLY_FRACTION

    # Cap at max position
    max_usd  = bankroll * cfg.MAX_POSITION_PCT
    kelly_usd = bankroll * frac_kelly

    position = min(kelly_usd, max_usd)

    return {
        "kelly_f":          round(kelly_f, 4),
        "frac_kelly":       round(frac_kelly, 4),
        "position_usd":     round(position, 2),
        "max_usd":          round(max_usd, 2),
        "pct_bankroll":     round(position / bankroll * 100, 2) if bankroll else 0,
    }


# ─── Full Market Scoring ──────────────────────────────────────────────────────

def score_market(market: dict,
                 sentiment_result: dict,
                 model_prob: float,
                 ml_confidence: float,
                 category_win_rates: dict = None) -> dict:
    """
    Score a single market end-to-end. Returns enriched dict.
    """
    # Pillar scores
    mom   = score_momentum(market)
    vol   = score_volume(market)
    sent  = score_sentiment(sentiment_result)
    edge_score, edge_pct = score_edge(market, model_prob)
    dec   = score_decay(market, mom)

    market["_momentum_score"] = mom  # used in filter

    pillars = {
        "momentum":  round(mom, 1),
        "volume":    round(vol, 1),
        "sentiment": round(sent, 1),
        "edge":      round(edge_score, 1),
        "decay":     round(dec, 1),
    }

    composite = round(compute_composite(pillars), 1)

    # Filters
    passes, flags = apply_filters(market, sentiment_result, model_prob, ml_confidence)

    # Position sizing
    yes_price = market.get("yes_price", 0.5)
    sizing = kelly_size(
        win_prob=model_prob,
        odds=yes_price,
        category_win_rate=(category_win_rates or {}).get(market.get("category"), None),
    )

    # Confidence label
    if composite >= 80 and passes:
        confidence = "HIGH"
    elif composite >= 60 and passes:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Direction
    direction = "YES" if model_prob > yes_price else "NO"

    return {
        **market,
        "pillars":       pillars,
        "composite":     composite,
        "edge_pct":      round(edge_pct * 100, 1),  # as percentage
        "model_prob":    round(model_prob, 4),
        "ml_confidence": round(ml_confidence, 4),
        "direction":     direction,
        "passes_filter": passes,
        "flags":         flags,
        "confidence":    confidence,
        "sizing":        sizing,
        "sentiment":     sentiment_result,
        "scored_at":     datetime.now(timezone.utc).isoformat(),
    }


def rank_markets(scored_markets: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Separate YES and NO picks, rank by composite score.
    Returns (yes_picks, no_picks).
    """
    yes_picks = sorted(
        [m for m in scored_markets if m["direction"] == "YES" and m["passes_filter"]],
        key=lambda x: x["composite"], reverse=True
    )
    no_picks = sorted(
        [m for m in scored_markets if m["direction"] == "NO" and m["passes_filter"]],
        key=lambda x: x["composite"], reverse=True
    )
    return yes_picks, no_picks
