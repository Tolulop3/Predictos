"""
PredictOS — Live Market Validator

Before publishing any pick, this module re-fetches the current market state
from Polymarket CLOB and checks whether the signal is still valid at the
current live price.

The problem without this:
  - GitHub Actions runs at 07:00 UTC
  - By 09:00 UTC the odds may have moved from 0.40 → 0.48
  - The edge was 21%, now it's 13% — below the Graham threshold
  - The dashboard still shows the pick as if nothing changed

This module runs as the LAST step before baking the dashboard:
  1. Re-fetch live price from CLOB /last-trade-price
  2. Re-fetch live order book depth from CLOB /book
  3. Re-check all filters against current price
  4. Flag picks where signal has deteriorated
  5. Compute actual current edge (not edge at scoring time)
  6. Reject picks where edge fell below threshold
  7. Show price drift on dashboard ("was 40¢, now 43¢")
"""

import time
import logging
import requests
from datetime import datetime, timezone
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger("predictos.live_validator")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "PredictOS/1.0",
    "Accept":     "application/json",
})


# ─── CLOB live price fetch ─────────────────────────────────────────────────────

def _clob_get(url: str, params: dict = None) -> Optional[dict | list]:
    try:
        r = SESSION.get(url, params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug(f"CLOB GET {url}: {e}")
        return None


def fetch_live_price(pick: dict) -> Optional[float]:
    """
    Get the most recent trade price for a Polymarket market's YES token.
    Returns None if unavailable (Kalshi/Manifold don't support this endpoint).
    """
    if pick.get("source") != "polymarket":
        return None

    token_ids = pick.get("token_ids", [])
    yes_token = token_ids[0] if token_ids else None

    if yes_token:
        data = _clob_get(
            f"{config.POLYMARKET_CLOB_BASE}/last-trade-price",
            params={"token_id": yes_token}
        )
        if data and data.get("price"):
            return float(data["price"])

    # Fallback: re-fetch full market from CLOB
    condition_id = pick.get("condition_id", "")
    if condition_id:
        data = _clob_get(f"{config.POLYMARKET_CLOB_BASE}/markets/{condition_id}")
        if data:
            tokens   = data.get("tokens", [])
            yes_tok  = next((t for t in tokens if t.get("outcome","").upper()=="YES"), None)
            if yes_tok and yes_tok.get("price"):
                return float(yes_tok["price"])

    return None


def fetch_live_book(pick: dict) -> dict:
    """
    Fetch live order book for a pick. Returns spread + depth.
    Used to check that liquidity is still adequate.
    """
    if pick.get("source") != "polymarket":
        return {}

    token_ids = pick.get("token_ids", [])
    yes_token = token_ids[0] if token_ids else None
    if not yes_token:
        return {}

    book = _clob_get(
        f"{config.POLYMARKET_CLOB_BASE}/book",
        params={"token_id": yes_token}
    )
    if not book:
        return {}

    bids = book.get("bids", [])
    asks = book.get("asks", [])
    if not bids or not asks:
        return {"spread": None, "depth": 0}

    best_bid = float(bids[0]["price"])
    best_ask = float(asks[0]["price"])
    spread   = round(best_ask - best_bid, 4)
    depth    = sum(float(b.get("size",0)) for b in bids[:5]) + \
               sum(float(a.get("size",0)) for a in asks[:5])

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread":   spread,
        "depth":    round(depth, 2),
    }


# ─── Signal staleness check ────────────────────────────────────────────────────

def validate_pick_live(pick: dict, model_prob: float) -> dict:
    """
    Re-validate a pick against current live market state.

    Returns enriched pick with:
      - live_price         : current YES price from CLOB
      - price_drift        : how much price moved since scoring
      - live_edge          : recalculated edge at current price
      - live_passes        : whether pick still passes all filters
      - validation_status  : 'valid' | 'stale' | 'edge_gone' | 'unavailable'
      - validation_note    : human-readable explanation
    """
    scored_price = pick.get("yes_price", 0.5)
    direction    = pick.get("direction", "YES")

    # Block extreme longshots at the validation gate
    # entry_price = yes_price for YES bets, (1-yes_price) for NO bets
    entry_price = scored_price if direction == "YES" else (1 - scored_price)
    if entry_price < 0.05:
        return {
            **pick,
            "live_price":        scored_price,
            "price_drift":       0.0,
            "live_edge":         pick.get("edge_pct", 0),
            "live_passes":       False,
            "validation_status": "extreme_longshot",
            "validation_note":   f"Entry price {entry_price:.1%} below 5% minimum — rejected",
            "validated_at":      datetime.now(timezone.utc).isoformat(),
        }

    # Fetch live price
    live_price = fetch_live_price(pick)

    if live_price is None:
        # Can't validate — treat as valid but flag
        return {
            **pick,
            "live_price":        scored_price,
            "price_drift":       0.0,
            "live_edge":         pick.get("edge_pct", 0),
            "live_passes":       True,
            "validation_status": "unavailable",
            "validation_note":   "Live price unavailable — using scored price",
            "validated_at":      datetime.now(timezone.utc).isoformat(),
        }

    live_price   = max(0.001, min(0.999, live_price))
    price_drift  = round(live_price - scored_price, 4)

    # Recalculate edge at live price
    if direction == "YES":
        live_edge = round((model_prob - live_price) * 100, 1)
    else:
        live_edge = round((live_price - model_prob) * 100, 1)

    # Check live book
    book = fetch_live_book(pick)
    live_spread = book.get("spread")
    live_depth  = book.get("depth", 0)

    # Determine status
    edge_threshold_pct = config.GRAHAM_EDGE_THRESHOLD * 100  # e.g. 6.0

    # Check exit signal (spec: exit when market_price >= model_prob * 0.9)
    from src.returns import should_exit
    days_remaining = pick.get("days_to_res", 30)
    exit_signal = should_exit(live_price, model_prob, days_remaining, direction)

    if exit_signal["should_exit"] and exit_signal["reason"] != "near_expiry":
        # Market has corrected to 90% of our model value — take profit signal
        status = "exit_signal"
        note   = (
            f"EXIT SIGNAL: Market ({live_price:.1%}) reached {exit_signal['pct_of_target']:.0f}% "
            f"of model target ({exit_signal['target_price']:.1%}). "
            f"Reason: {exit_signal['reason']}. Consider closing position."
        )
        passes = True  # still a valid pick but flagged for exit

    elif abs(live_edge) < edge_threshold_pct:
        status = "edge_gone"
        note   = (
            f"Edge fell to {live_edge:+.1f}% (need ≥{edge_threshold_pct:.0f}%). "
            f"Price moved {price_drift:+.1%} since signal."
        )
        passes = False

    elif abs(price_drift) > 0.08:
        status = "stale"
        note   = (
            f"Price drifted {price_drift:+.1%} since scoring "
            f"({scored_price:.1%} → {live_price:.1%}). Edge still {live_edge:+.1f}%."
        )
        passes = True  # still valid but marked stale for dashboard display

    elif live_spread is not None and live_spread > 0.12:
        status = "wide_spread"
        note   = f"Spread widened to {live_spread:.3f} — thin liquidity right now."
        passes = False

    else:
        status = "valid"
        drift_str = f"{price_drift:+.1%}" if abs(price_drift) > 0.01 else "no drift"
        note   = f"Signal valid. Live: {live_price:.1%} ({drift_str}). Edge: {live_edge:+.1f}%."
        passes = True

    return {
        **pick,
        # Update to live prices
        "yes_price":         live_price,
        "no_price":          round(1 - live_price, 4),
        "price_at_signal":   scored_price,
        "price_drift":       round(price_drift * 100, 2),  # in pct points
        "live_edge":         live_edge,
        "live_spread":       live_spread,
        "live_depth":        live_depth,
        "live_passes":       passes,
        "validation_status": status,
        "validation_note":   note,
        "validated_at":      datetime.now(timezone.utc).isoformat(),
    }


# ─── Portfolio-level validation ───────────────────────────────────────────────

def validate_all_picks(picks: dict, model_probs: dict) -> dict:
    """
    Run live validation on every pick in the picks dict.
    Removes picks that no longer pass. Flags stale ones.
    Also runs portfolio correlation guard.
    """
    log.info("Live validation: checking picks against Polymarket CLOB…")

    validated_yes   = []
    validated_no    = []
    validated_macro = []
    rejected        = []

    for pick in picks.get("yes_picks", []):
        mid  = pick.get("id", "")
        mprob = model_probs.get(mid, pick.get("model_prob", 0.5))
        result = validate_pick_live(pick, mprob)
        time.sleep(0.2)   # rate limit courtesy
        if result["live_passes"]:
            validated_yes.append(result)
        else:
            rejected.append({**result, "rejected_reason": result["validation_note"]})
            log.info(f"  REJECTED: {pick['question'][:50]} — {result['validation_note']}")

    for pick in picks.get("no_picks", []):
        mid  = pick.get("id", "")
        mprob = model_probs.get(mid, pick.get("model_prob", 0.5))
        result = validate_pick_live(pick, mprob)
        time.sleep(0.2)
        if result["live_passes"]:
            validated_no.append(result)
        else:
            rejected.append({**result, "rejected_reason": result["validation_note"]})

    for pick in picks.get("macro_plays", []):
        mid  = pick.get("id", "")
        mprob = model_probs.get(mid, pick.get("model_prob", 0.5))
        result = validate_pick_live(pick, mprob)
        time.sleep(0.2)
        validated_macro.append(result)  # macro plays shown even if stale

    # Portfolio correlation guard
    validated_yes, correlation_flags = _correlation_guard(validated_yes)

    log.info(f"  Valid YES: {len(validated_yes)}, NO: {len(validated_no)}, "
             f"Rejected: {len(rejected)}")

    return {
        **picks,
        "yes_picks":         validated_yes,
        "no_picks":          validated_no,
        "macro_plays":       validated_macro,
        "rejected_picks":    rejected,
        "correlation_flags": correlation_flags,
        "validated_at":      datetime.now(timezone.utc).isoformat(),
    }


# ─── Portfolio correlation guard ──────────────────────────────────────────────

def _correlation_guard(yes_picks: list[dict],
                       max_same_category: int = 2) -> tuple[list[dict], list[str]]:
    """
    Prevent the portfolio from being dominated by one correlated theme.

    Rules:
    - Max 2 picks from the same category (e.g. only 2 "politics" picks)
    - Max 2 picks sharing dominant keyword (e.g. "election", "fed", "bitcoin")
    - If a category is over-represented, drop the lowest-scored pick(s)

    Returns (filtered_picks, [warning_messages])
    """
    if len(yes_picks) <= 2:
        return yes_picks, []

    flags   = []
    cat_count: dict[str, list] = {}

    for p in yes_picks:
        cat = p.get("category", "other")
        cat_count.setdefault(cat, []).append(p)

    final_picks = []
    for cat, group in cat_count.items():
        if len(group) > max_same_category:
            # Keep top N by composite score
            group_sorted = sorted(group, key=lambda x: x.get("composite", 0), reverse=True)
            kept    = group_sorted[:max_same_category]
            dropped = group_sorted[max_same_category:]
            final_picks.extend(kept)
            for d in dropped:
                flags.append(
                    f"Correlation guard: dropped '{d['question'][:45]}…' "
                    f"(3rd {cat} pick — portfolio already has {max_same_category})"
                )
                log.info(f"  Correlation guard dropped: {d['question'][:50]}")
        else:
            final_picks.extend(group)

    # Keyword clustering check
    keyword_clusters = _keyword_clusters(final_picks)
    for kw, cluster in keyword_clusters.items():
        if len(cluster) > max_same_category:
            flags.append(
                f"Keyword cluster '{kw}': {len(cluster)} picks share this theme — "
                f"consider reducing exposure"
            )

    # Re-sort by composite
    final_picks.sort(key=lambda x: x.get("composite", 0), reverse=True)
    return final_picks, flags


def _keyword_clusters(picks: list[dict]) -> dict[str, list]:
    """Group picks by dominant keyword cluster."""
    clusters: dict[str, list] = {}
    cluster_keywords = [
        ["election", "vote", "president", "congress", "senate", "ballot"],
        ["fed", "rate", "inflation", "cpi", "fomc", "interest"],
        ["bitcoin", "crypto", "btc", "eth", "ethereum"],
        ["ukraine", "russia", "war", "nato", "ceasefire"],
        ["china", "taiwan", "trade", "tariff"],
        ["trump", "harris", "biden", "democrat", "republican"],
    ]
    for pick in picks:
        q = pick.get("question", "").lower()
        for kw_group in cluster_keywords:
            matched = next((kw for kw in kw_group if kw in q), None)
            if matched:
                label = kw_group[0]  # use first keyword as cluster label
                clusters.setdefault(label, []).append(pick)
                break
    return clusters
