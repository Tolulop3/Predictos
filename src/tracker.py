"""
PredictOS — Outcome Tracker
Auto-resolves picks when markets close.
Tracks win rate, updates history, feeds back into ML training data.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import requests
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger("predictos.tracker")


# ─── File I/O ─────────────────────────────────────────────────────────────────

def _load_json(path: str, default=None):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return default if default is not None else {}


def _save_json(path: str, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─── Resolution Fetchers ──────────────────────────────────────────────────────

def _resolve_polymarket(market_id: str) -> dict | None:
    """
    Check Polymarket resolution using the three-API cascade:
      1. data-api /resolution  — most reliable, explicit outcome field
      2. CLOB /markets         — fallback via closed + winner token flag
    """
    from src.fetcher import check_polymarket_resolution
    condition_id = market_id.split("::")[-1]
    market_stub  = {"condition_id": condition_id}
    try:
        return check_polymarket_resolution(market_stub)
    except Exception as e:
        log.debug(f"Polymarket resolve check failed {market_id}: {e}")
    return None


def _resolve_kalshi(market_id: str) -> dict | None:
    """Check Kalshi market resolution."""
    ticker = market_id.split("::")[-1]
    try:
        r = requests.get(
            f"{config.KALSHI_BASE}/markets/{ticker}",
            timeout=10
        )
        r.raise_for_status()
        data = r.json().get("market", {})
        status = data.get("status", "")
        if status != "finalized":
            return None
        result = data.get("result", "")
        return {"resolved": True, "resolved_yes": result.upper() == "YES"}
    except Exception as e:
        log.debug(f"Kalshi resolve check failed {market_id}: {e}")
    return None


def _resolve_manifold(market_id: str) -> dict | None:
    """Check Manifold market resolution."""
    mid = market_id.split("::")[-1]
    try:
        r = requests.get(
            f"{config.MANIFOLD_BASE}/market/{mid}",
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("isResolved", False):
            return None
        resolution = data.get("resolution", "")
        return {"resolved": True, "resolved_yes": resolution == "YES"}
    except Exception as e:
        log.debug(f"Manifold resolve check failed {market_id}: {e}")
    return None


def check_resolution(market: dict) -> dict | None:
    """Check if a market has resolved. Returns resolution info or None."""
    mid    = market.get("id", "")
    source = market.get("source", "")

    if source == "polymarket":
        return _resolve_polymarket(mid)
    elif source == "kalshi":
        return _resolve_kalshi(mid)
    elif source == "manifold":
        return _resolve_manifold(mid)
    return None


# ─── Pick Tracking ────────────────────────────────────────────────────────────

def save_picks(picks: dict):
    """Save today's picks to file."""
    picks["saved_at"] = datetime.now(timezone.utc).isoformat()
    _save_json(config.PICKS_FILE, picks)
    log.info(f"Picks saved → {config.PICKS_FILE}")


def load_picks() -> dict:
    return _load_json(config.PICKS_FILE, {"yes_picks": [], "no_picks": [], "macro_plays": []})


def load_outcomes() -> list:
    return _load_json(config.OUTCOMES_FILE, [])


def save_outcomes(outcomes: list):
    _save_json(config.OUTCOMES_FILE, outcomes)


def load_history() -> list:
    return _load_json(config.HISTORY_FILE, [])


def save_history(history: list):
    _save_json(config.HISTORY_FILE, history)


# ─── Auto-Resolve ─────────────────────────────────────────────────────────────

def auto_resolve_picks() -> dict:
    """
    Check all unresolved picks. For any that have closed, record outcome.
    Returns summary stats.
    """
    picks    = load_picks()
    outcomes = load_outcomes()
    history  = load_history()

    resolved_ids = {o["id"] for o in outcomes if o.get("resolved")}

    all_picks = (
        picks.get("yes_picks", []) +
        picks.get("no_picks", []) +
        picks.get("macro_plays", [])
    )

    newly_resolved = []
    for pick in all_picks:
        pid = pick.get("id")
        if pid in resolved_ids:
            continue

        resolution = check_resolution(pick)
        if not resolution or not resolution.get("resolved"):
            continue

        resolved_yes = resolution["resolved_yes"]
        direction    = pick.get("direction", "YES")
        won          = (direction == "YES" and resolved_yes) or \
                       (direction == "NO" and not resolved_yes)

        outcome = {
            **pick,
            "resolved":     True,
            "resolved_yes": resolved_yes,
            "won":          won,
            "resolved_at":  datetime.now(timezone.utc).isoformat(),
        }
        outcomes.append(outcome)
        newly_resolved.append(outcome)
        log.info(f"Resolved: {pick.get('question', pid)[:60]} → "
                 f"YES={resolved_yes} | {'WIN' if won else 'LOSS'}")

    save_outcomes(outcomes)

    # Compute stats
    stats = compute_stats(outcomes)

    # Append to history
    history.append({
        "date":            datetime.now(timezone.utc).date().isoformat(),
        "newly_resolved":  len(newly_resolved),
        "total_resolved":  len(outcomes),
        "stats":           stats,
    })
    save_history(history)

    return {
        "newly_resolved": len(newly_resolved),
        "total_resolved": len(outcomes),
        "stats":          stats,
        "picks":          newly_resolved,
    }


# ─── Stats ────────────────────────────────────────────────────────────────────

def compute_stats(outcomes: list = None) -> dict:
    """Compute overall and per-category win rates."""
    if outcomes is None:
        outcomes = load_outcomes()

    resolved = [o for o in outcomes if o.get("resolved")]
    if not resolved:
        return {
            "total": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "by_category": {}, "by_source": {}
        }

    wins   = sum(1 for o in resolved if o.get("won"))
    losses = len(resolved) - wins

    # By category
    by_cat: dict[str, dict] = {}
    for o in resolved:
        cat = o.get("category", "other")
        by_cat.setdefault(cat, {"wins": 0, "total": 0})
        by_cat[cat]["total"] += 1
        if o.get("won"):
            by_cat[cat]["wins"] += 1

    # By source
    by_src: dict[str, dict] = {}
    for o in resolved:
        src = o.get("source", "unknown")
        by_src.setdefault(src, {"wins": 0, "total": 0})
        by_src[src]["total"] += 1
        if o.get("won"):
            by_src[src]["wins"] += 1

    return {
        "total":    len(resolved),
        "wins":     wins,
        "losses":   losses,
        "win_rate": round(wins / len(resolved), 4) if resolved else 0.0,
        "by_category": {
            cat: {
                **v,
                "win_rate": round(v["wins"] / v["total"], 4)
            }
            for cat, v in by_cat.items()
        },
        "by_source": {
            src: {
                **v,
                "win_rate": round(v["wins"] / v["total"], 4)
            }
            for src, v in by_src.items()
        },
    }


# ─── Content Engine ───────────────────────────────────────────────────────────

def generate_daily_tweet(picks: dict, stats: dict) -> str:
    """Generate daily tweet summarising top signal."""
    yes_picks = picks.get("yes_picks", [])
    no_picks  = picks.get("no_picks", [])

    if not yes_picks:
        return "📊 PredictOS daily scan complete — no high-confidence picks today. Staying patient. #PredictionMarkets"

    top = yes_picks[0]
    q   = top.get("question", "")[:60]
    score = top.get("composite", 0)
    edge  = top.get("edge_pct", 0)
    conf  = top.get("confidence", "")
    wr    = stats.get("win_rate", 0)

    tweet = (
        f"📊 PredictOS Signal #{datetime.now().strftime('%b%d')}\n\n"
        f"🟢 TOP YES: \"{q}...\"\n"
        f"Score: {score:.0f}/100 | Edge: {edge:+.1f}% | {conf} confidence\n"
    )

    if no_picks:
        no_top = no_picks[0]
        tweet += f"\n🔴 TOP NO: \"{no_top.get('question','')[:45]}...\"\n"

    tweet += (
        f"\nTracked win rate: {wr:.1%} ({stats.get('total', 0)} resolved)\n"
        "#PredictionMarkets #Polymarket #Kalshi"
    )
    return tweet[:280]  # Twitter limit
