"""
PredictOS — Paper Trading Engine

Simulates trades with virtual money to validate the model before
committing real capital. Runs in parallel with the signal engine.

How it works:
  1. Every pick that passes filters gets a simulated paper trade
  2. Stake = Kelly-sized position from virtual $1,000 bankroll
  3. When market resolves, record outcome vs prediction
  4. Track paper P&L, calibration, and Sharpe ratio
  5. After 30+ paper trades with positive EV, the model is validated

Paper trading vs live trading:
  Paper trading  → no real money, no execution risk, no slippage
  Live trading   → real money, requires Polymarket API key + CLOB execution
                   Only appropriate AFTER paper trading proves the edge

The paper account is stored in data/paper_trades.json.
It runs automatically in parallel with every daily pipeline run.

Paper trading does NOT guarantee live trading will work:
  - Real markets have slippage (you don't always get the displayed price)
  - Order size moves the market on thin books
  - The paper engine assumes fills at mid price
These will reduce real returns vs paper returns.
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger("predictos.paper")

PAPER_FILE          = "data/paper_trades.json"
PAPER_BANKROLL_INIT = 1000.0   # virtual starting bankroll
MAX_PAPER_POSITION  = 0.02     # 2% per trade
KELLY_FRACTION      = 0.25


# ─── State ────────────────────────────────────────────────────────────────────

def _load() -> dict:
    p = Path(PAPER_FILE)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {
        "bankroll":            PAPER_BANKROLL_INIT,
        "peak_bankroll":       PAPER_BANKROLL_INIT,
        "total_pnl":           0.0,
        "open_trades":         [],
        "closed_trades":       [],
        "stats":               {},
        "validation_status":   "accumulating",  # accumulating | validated | failed
        "created_at":          datetime.now(timezone.utc).isoformat(),
    }


def _save(state: dict):
    Path(PAPER_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(PAPER_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ─── Open a paper trade ───────────────────────────────────────────────────────

def open_paper_trade(pick: dict, model_prob: float) -> Optional[dict]:
    """
    Open a simulated paper trade for a pick.
    Returns the trade record, or None if bankroll check fails.
    """
    state   = _load()
    bankroll = state["bankroll"]
    direction = pick.get("direction", "YES")
    yes_price = pick.get("yes_price", 0.5)

    entry_price = yes_price if direction == "YES" else (1 - yes_price)
    entry_price = max(0.001, min(0.999, entry_price))

    # Block bets where entry price is below 5% — these are extreme longshots
    # regardless of direction. A 1.2% entry = betting on a 98.8% certainty NOT happening.
    if entry_price < 0.05:
        return None

    # Kelly position size
    p_win  = model_prob if direction == "YES" else (1 - model_prob)
    p_win  = max(0.01, min(0.99, p_win))
    b      = (1 / entry_price) - 1
    kelly  = max(0, (b * p_win - (1 - p_win)) / b) * KELLY_FRACTION
    stake  = min(bankroll * kelly, bankroll * MAX_PAPER_POSITION)
    stake  = round(max(stake, 0.50), 2)  # min $0.50 paper trade

    if stake <= 0 or bankroll < 10:
        return None

    shares    = stake / entry_price
    win_pnl   = round(shares - stake, 2)
    lose_pnl  = round(-stake, 2)

    trade = {
        "id":            f"paper::{pick.get('id', '')}::{datetime.now(timezone.utc).date()}",
        "market_id":     pick.get("id", ""),
        "source":        pick.get("source", ""),
        "question":      pick.get("question", "")[:80],
        "category":      pick.get("category", "other"),
        "direction":     direction,
        "entry_price":   entry_price,
        "stake":         stake,
        "shares":        round(shares, 4),
        "win_pnl":       win_pnl,
        "lose_pnl":      lose_pnl,
        "model_prob":    round(model_prob, 4),
        "yes_price":     yes_price,
        "composite":     pick.get("composite", 0),
        "edge_pct":      pick.get("edge_pct", 0),
        "days_to_res":   pick.get("days_to_res", 30),
        "opened_at":     datetime.now(timezone.utc).isoformat(),
        "resolved":      False,
        "outcome":       None,
        "actual_pnl":    None,
        "pnl_vs_pred":   None,   # actual_pnl - predicted EV
    }

    # Deduct stake from paper bankroll immediately (committed capital)
    state["bankroll"]      = round(bankroll - stake, 2)
    state["open_trades"].append(trade)
    _save(state)

    log.info(
        f"Paper trade opened: {direction} {pick.get('question','')[:40]}… "
        f"stake=${stake:.2f} entry={entry_price:.1%} "
        f"win=+${win_pnl:.2f} lose=${lose_pnl:.2f}"
    )
    return trade


# ─── Close a paper trade ──────────────────────────────────────────────────────

def close_paper_trade(market_id: str, resolved_yes: bool) -> Optional[dict]:
    """
    Close an open paper trade at resolution.
    Returns the closed trade with actual P&L.
    """
    state = _load()

    trade_idx = next(
        (i for i, t in enumerate(state["open_trades"])
         if t["market_id"] == market_id and not t["resolved"]),
        None
    )
    if trade_idx is None:
        return None

    trade     = state["open_trades"].pop(trade_idx)
    direction = trade["direction"]
    won       = (direction == "YES" and resolved_yes) or \
                (direction == "NO" and not resolved_yes)

    actual_pnl = trade["win_pnl"] if won else trade["lose_pnl"]

    # Return stake + winnings (or just lose the stake)
    if won:
        state["bankroll"] = round(state["bankroll"] + trade["stake"] + actual_pnl, 2)
    # If lost, stake was already deducted at open

    state["peak_bankroll"] = max(state["peak_bankroll"], state["bankroll"])
    state["total_pnl"]     = round(state["total_pnl"] + actual_pnl, 2)

    # Expected P&L at entry (model-based)
    p_win      = trade["model_prob"] if direction == "YES" else (1 - trade["model_prob"])
    expected   = p_win * trade["win_pnl"] + (1 - p_win) * trade["lose_pnl"]

    trade.update({
        "resolved":    True,
        "resolved_yes": resolved_yes,
        "won":         won,
        "actual_pnl":  round(actual_pnl, 2),
        "expected_pnl": round(expected, 2),
        "pnl_vs_pred": round(actual_pnl - expected, 2),
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    })

    state["closed_trades"].append(trade)
    state["closed_trades"] = state["closed_trades"][-200:]  # keep last 200

    # Recompute stats
    state["stats"] = _compute_paper_stats(state)

    # Update validation status
    state["validation_status"] = _validation_status(state["stats"])

    _save(state)

    log.info(
        f"Paper trade closed: {'WIN' if won else 'LOSS'} "
        f"actual=${actual_pnl:+.2f} expected=${expected:+.2f} | "
        f"Paper bankroll: ${state['bankroll']:.2f}"
    )
    return trade


# ─── Stats ────────────────────────────────────────────────────────────────────

def _compute_paper_stats(state: dict) -> dict:
    closed = state.get("closed_trades", [])
    if not closed:
        return {"n": 0}

    n       = len(closed)
    wins    = sum(1 for t in closed if t.get("won"))
    win_rate = wins / n

    pnls    = [t.get("actual_pnl", 0) for t in closed]
    total   = sum(pnls)
    avg     = total / n

    # Sharpe ratio (annualised, assuming 1 trade/day)
    if n >= 5:
        import statistics
        std = statistics.stdev(pnls) if len(pnls) > 1 else 0.001
        sharpe = (avg / std) * math.sqrt(252) if std > 0 else 0
    else:
        sharpe = 0

    # Return vs initial
    initial = PAPER_BANKROLL_INIT
    total_return = (state["bankroll"] - initial) / initial * 100

    # EV accuracy: how close was predicted EV to actual average PnL?
    ev_errors = [abs(t.get("pnl_vs_pred", 0)) for t in closed if t.get("pnl_vs_pred") is not None]
    ev_mae    = sum(ev_errors) / len(ev_errors) if ev_errors else None

    # By category
    by_cat: dict = {}
    for t in closed:
        cat = t.get("category", "other")
        by_cat.setdefault(cat, {"n": 0, "wins": 0, "pnl": 0})
        by_cat[cat]["n"]    += 1
        by_cat[cat]["wins"] += int(t.get("won", False))
        by_cat[cat]["pnl"]  += t.get("actual_pnl", 0)

    return {
        "n":             n,
        "wins":          wins,
        "losses":        n - wins,
        "win_rate":      round(win_rate, 4),
        "total_pnl":     round(total, 2),
        "avg_pnl":       round(avg, 2),
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio":  round(sharpe, 3),
        "ev_mae":        round(ev_mae, 2) if ev_mae else None,
        "by_category":   {
            cat: {
                **v,
                "win_rate": round(v["wins"] / v["n"], 3),
                "pnl":      round(v["pnl"], 2),
            }
            for cat, v in by_cat.items()
        },
    }


def _validation_status(stats: dict) -> str:
    """
    Determine if the model is validated for live trading.

    Criteria (all must be met):
      - 30+ resolved paper trades
      - Win rate >= 55%
      - Total return > 0% (positive P&L)
      - Sharpe ratio >= 0.5
    """
    n        = stats.get("n", 0)
    win_rate = stats.get("win_rate", 0)
    ret      = stats.get("total_return_pct", 0)
    sharpe   = stats.get("sharpe_ratio", 0)

    if n < 10:
        return "accumulating"
    if n < 30:
        if win_rate >= 0.55 and ret > 0:
            return "promising"
        return "accumulating"
    # 30+ trades
    if win_rate >= 0.55 and ret > 0 and sharpe >= 0.5:
        return "validated"
    if win_rate < 0.45 or ret < -15:
        return "failed"
    return "inconclusive"


# ─── Full paper account summary ───────────────────────────────────────────────

def paper_summary() -> dict:
    state = _load()
    stats = state.get("stats", {})

    return {
        "bankroll":          state.get("bankroll", PAPER_BANKROLL_INIT),
        "initial_bankroll":  PAPER_BANKROLL_INIT,
        "peak_bankroll":     state.get("peak_bankroll", PAPER_BANKROLL_INIT),
        "total_pnl":         state.get("total_pnl", 0),
        "open_trades":       len(state.get("open_trades", [])),
        "closed_trades":     len(state.get("closed_trades", [])),
        "validation_status": state.get("validation_status", "accumulating"),
        "stats":             stats,
        "recent_trades":     state.get("closed_trades", [])[-5:],
        "validation_criteria": {
            "required_trades":  30,
            "min_win_rate":     "55%",
            "min_return":       "positive",
            "min_sharpe":       0.5,
            "note": (
                "Paper trading proves the edge is real before any live capital is risked. "
                "Even after validation, live results will differ due to slippage and "
                "execution risk. Start live trading with minimum stakes only."
            ),
        },
    }


def sync_paper_resolutions(outcomes: list[dict]):
    """
    Sync paper trade resolutions with the outcome tracker.
    Called from main pipeline after auto_resolve_picks().
    """
    resolved_count = 0
    for outcome in outcomes:
        if not outcome.get("resolved"):
            continue
        mid = outcome.get("id", "").replace("polymarket::", "").replace("kalshi::", "").replace("manifold::", "")
        result = close_paper_trade(outcome.get("id", ""), outcome.get("resolved_yes", False))
        if result:
            resolved_count += 1
    if resolved_count:
        log.info(f"Paper trading: synced {resolved_count} resolutions")
