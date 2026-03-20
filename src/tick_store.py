"""
PredictOS — Tick Store

Saves odds snapshots every 4 hours so the momentum pillar uses
real intraday price history instead of a single API lookup.

Without this: momentum = today's price minus yesterday's price (one data point)
With this:    momentum = true time-series curve across 7 days of 4h candles

Structure:
  data/ticks/
    polymarket/
      {condition_id}.jsonl    ← one JSON line per snapshot
    kalshi/
      {ticker}.jsonl
    manifold/
      {market_id}.jsonl

Each line:
  {"ts": 1712345678, "yes": 0.42, "vol_24h": 45000, "liquidity": 180000, "spread": 0.04}

The scorer reads the last 7 days of ticks to compute:
  - momentum_1h   : price change in last 1 hour
  - momentum_4h   : price change in last 4 hours
  - momentum_24h  : price change in last 24 hours (replaces snapshot delta)
  - momentum_72h  : price change in last 72 hours
  - vol_trend     : is volume increasing or decreasing?
  - price_velocity: rate of change (slope of last 6 ticks)
"""

import json
import math
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger("predictos.ticks")

TICK_DIR     = Path("data/ticks")
TICK_DAYS    = 7      # keep 7 days of history
TICK_INTERVAL_HOURS = 4


# ─── Write ────────────────────────────────────────────────────────────────────

def save_tick(market: dict):
    """
    Append a price snapshot for one market.
    Enforces TICK_INTERVAL_HOURS minimum gap — duplicate runs within the
    same hour must not pollute the store with near-identical entries.
    Without this guard, two ticks 9 minutes apart both land in the cache,
    and get_price_at(now-24h) finds neither because both are "too recent",
    returning None and killing momentum enrichment for that market.
    """
    source = market.get("source", "unknown")
    raw_id = _raw_id(market)
    if not raw_id:
        return

    tick_path = TICK_DIR / source / f"{_safe_filename(raw_id)}.jsonl"
    tick_path.parent.mkdir(parents=True, exist_ok=True)

    now_ts  = int(time.time())
    min_gap = TICK_INTERVAL_HOURS * 3600

    # Skip if last saved tick is too recent
    if tick_path.exists():
        try:
            last_line = None
            with open(tick_path) as f:
                for line in f:
                    if line.strip():
                        last_line = line.strip()
            if last_line:
                last_ts = json.loads(last_line).get("ts", 0)
                if now_ts - last_ts < min_gap:
                    return False  # too soon — skip
        except Exception:
            pass  # corrupt file — safe to overwrite

    tick = {
        "ts":        now_ts,
        "yes":       market.get("yes_price", 0.5),
        "vol_24h":   market.get("volume_24h", 0),
        "liquidity": market.get("liquidity", 0),
        "spread":    market.get("spread"),
        "depth":     market.get("book_depth", 0),
    }

    with open(tick_path, "a") as f:
        f.write(json.dumps(tick) + "\n")
    return True


def save_ticks_batch(markets: list[dict]):
    """Save ticks for all markets in one pass."""
    saved = 0
    skipped = 0
    for m in markets:
        try:
            wrote = save_tick(m)
            if wrote:
                saved += 1
            else:
                skipped += 1
        except Exception as e:
            log.debug(f"Tick save failed {m.get('id','?')}: {e}")
    if skipped:
        log.info(f"Tick store: saved {saved} snapshots ({skipped} skipped — interval guard)")
    else:
        log.info(f"Tick store: saved {saved} snapshots")


def prune_old_ticks():
    """Remove tick entries older than TICK_DAYS. Run weekly to keep disk clean."""
    cutoff = time.time() - TICK_DAYS * 86400
    pruned = 0
    for path in TICK_DIR.rglob("*.jsonl"):
        lines = []
        try:
            with open(path) as f:
                lines = [l for l in f if json.loads(l).get("ts", 0) >= cutoff]
            with open(path, "w") as f:
                f.writelines(l + "\n" if not l.endswith("\n") else l for l in lines)
            pruned += 1
        except Exception:
            pass
    log.info(f"Tick prune: cleaned {pruned} files")


# ─── Read ─────────────────────────────────────────────────────────────────────

def load_ticks(market: dict, days: int = 7) -> list[dict]:
    """
    Load tick history for a market.
    Returns list of {ts, yes, vol_24h, ...} sorted oldest-first.
    """
    source = market.get("source", "unknown")
    raw_id = _raw_id(market)
    if not raw_id:
        return []

    tick_path = TICK_DIR / source / f"{_safe_filename(raw_id)}.jsonl"
    if not tick_path.exists():
        return []

    cutoff = time.time() - days * 86400
    ticks  = []
    try:
        with open(tick_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                if t.get("ts", 0) >= cutoff:
                    ticks.append(t)
    except Exception as e:
        log.debug(f"Tick load failed {tick_path}: {e}")

    return sorted(ticks, key=lambda t: t["ts"])


def get_price_at(ticks: list[dict], target_ts: float) -> Optional[float]:
    """Get the YES price closest to (but not after) target_ts."""
    candidates = [t for t in ticks if t["ts"] <= target_ts]
    if not candidates:
        return None
    return candidates[-1]["yes"]


# ─── Momentum from ticks ──────────────────────────────────────────────────────

def compute_tick_momentum(market: dict) -> dict:
    """
    Replace snapshot-based momentum with real time-series momentum.
    Returns dict of momentum values to merge into the market dict.
    """
    ticks = load_ticks(market, days=TICK_DAYS)
    if len(ticks) < 2:
        # Not enough history yet — fall back to snapshot deltas
        return {}

    now_ts  = time.time()
    yes_now = ticks[-1]["yes"]

    # Time windows
    p_1h  = get_price_at(ticks, now_ts - 3_600)
    p_4h  = get_price_at(ticks, now_ts - 14_400)
    p_24h = get_price_at(ticks, now_ts - 86_400)
    p_72h = get_price_at(ticks, now_ts - 259_200)

    result = {}
    if p_1h  is not None: result["yes_price_1h_ago"]  = p_1h
    if p_4h  is not None: result["yes_price_4h_ago"]  = p_4h
    if p_24h is not None: result["yes_price_24h_ago"] = p_24h
    if p_72h is not None: result["yes_price_72h_ago"] = p_72h

    # Price velocity: slope of last 6 ticks (linear regression)
    recent = ticks[-6:]
    if len(recent) >= 3:
        n      = len(recent)
        xs     = [t["ts"] for t in recent]
        ys     = [t["yes"] for t in recent]
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        num    = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den    = sum((x - x_mean) ** 2 for x in xs)
        slope  = num / den if den > 0 else 0
        # Convert to "% per hour"
        result["price_velocity"] = round(slope * 3600 * 100, 4)

    # Volume trend: is 24h volume higher than 48h-ago volume?
    vol_recent = ticks[-6:]
    vol_older  = ticks[-12:-6] if len(ticks) >= 12 else []
    if vol_recent and vol_older:
        avg_recent = sum(t.get("vol_24h", 0) for t in vol_recent) / len(vol_recent)
        avg_older  = sum(t.get("vol_24h", 0) for t in vol_older)  / len(vol_older)
        result["vol_trend"] = round((avg_recent - avg_older) / max(avg_older, 1) * 100, 1)

    return result


def enrich_markets_from_ticks(markets: list[dict]) -> list[dict]:
    """
    Enrich all markets with tick-based momentum.
    Called in main pipeline after fetching markets.
    """
    enriched = 0
    for i, m in enumerate(markets):
        tick_data = compute_tick_momentum(m)
        if tick_data:
            markets[i] = {**m, **tick_data}
            enriched += 1
    log.info(f"Tick momentum: enriched {enriched}/{len(markets)} markets")
    return markets


# ─── Utility ──────────────────────────────────────────────────────────────────

def _raw_id(market: dict) -> str:
    """Extract the raw source ID from a market dict."""
    mid = market.get("id", "")
    if "::" in mid:
        return mid.split("::", 1)[1]
    return market.get("condition_id", "") or mid


def _safe_filename(s: str) -> str:
    """Make a string safe for use as a filename."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:80]


def tick_store_stats() -> dict:
    """Summary of what's in the tick store."""
    if not TICK_DIR.exists():
        return {"total_markets": 0, "total_ticks": 0, "sources": {}}

    by_source = {}
    total_ticks = 0
    for source_dir in TICK_DIR.iterdir():
        if not source_dir.is_dir():
            continue
        files = list(source_dir.glob("*.jsonl"))
        n_ticks = 0
        for f in files:
            try:
                with open(f) as fh:
                    n_ticks += sum(1 for _ in fh)
            except Exception:
                pass
        by_source[source_dir.name] = {"markets": len(files), "ticks": n_ticks}
        total_ticks += n_ticks

    return {
        "total_markets": sum(v["markets"] for v in by_source.values()),
        "total_ticks":   total_ticks,
        "sources":       by_source,
    }
