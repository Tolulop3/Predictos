"""
PredictOS — Drawdown Circuit Breaker & Bankroll Manager

Tracks real bankroll changes over time and halts new positions
when drawdown exceeds safety thresholds.

Why this matters:
  Without this, a losing streak compounds:
    - Model has 35% win rate for 2 weeks (can happen)
    - System keeps betting at same Kelly sizes
    - Bankroll drops 40% before you notice
  
  With this:
    - After 10% drawdown → reduce position sizes by 50%
    - After 15% drawdown → halt all new positions
    - After recovery    → gradually re-enable

Three separate safeguards:
  1. Drawdown guard:    tracks peak bankroll, triggers on % fall
  2. Streak guard:      halts after N consecutive losses
  3. Velocity guard:    halts if losing >X% per day for 3 days

Bankroll state is persisted in data/bankroll.json.
Updated automatically when picks resolve via the learning store.
"""

import json
import logging
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger("predictos.bankroll")

BANKROLL_FILE = "data/bankroll.json"

# Thresholds
DRAWDOWN_REDUCE_PCT  = 0.10   # 10% drawdown → halve position sizes
DRAWDOWN_HALT_PCT    = 0.15   # 15% drawdown → halt all new positions
LOSS_STREAK_HALT     = 6      # 6 consecutive losses → halt
DAILY_LOSS_HALT_PCT  = 0.05   # losing >5%/day for 3 days → halt
RECOVERY_WIN_STREAK  = 3      # 3 consecutive wins to re-enable after halt


def _load() -> dict:
    p = Path(BANKROLL_FILE)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    # Initialise from config
    initial = config.BANKROLL
    return {
        "initial_bankroll":    initial,
        "peak_bankroll":       initial,
        "current_bankroll":    initial,
        "all_time_pnl":        0.0,
        "consecutive_losses":  0,
        "consecutive_wins":    0,
        "daily_pnl":           {},        # date → daily P&L
        "status":              "active",  # active | reduced | halted
        "status_reason":       None,
        "status_since":        None,
        "trade_history":       [],        # last 50 trades
        "last_updated":        datetime.now(timezone.utc).isoformat(),
    }


def _save(state: dict):
    Path(BANKROLL_FILE).parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(BANKROLL_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ─── Public API ───────────────────────────────────────────────────────────────

def get_state() -> dict:
    return _load()


def record_trade_result(stake: float, pnl: float, pick: dict) -> dict:
    """
    Record the result of a resolved trade and update circuit breaker state.
    Called by the learning/tracking system when a pick resolves.

    stake : amount wagered
    pnl   : net profit (positive = won, negative = lost)
    pick  : the original pick dict
    """
    state = _load()

    won  = pnl > 0
    date = datetime.now(timezone.utc).date().isoformat()

    # Update bankroll
    state["current_bankroll"]  = round(state["current_bankroll"] + pnl, 2)
    state["all_time_pnl"]      = round(state["all_time_pnl"] + pnl, 2)
    state["peak_bankroll"]     = max(state["peak_bankroll"], state["current_bankroll"])

    # Streak counters
    if won:
        state["consecutive_wins"]   += 1
        state["consecutive_losses"]  = 0
    else:
        state["consecutive_losses"] += 1
        state["consecutive_wins"]    = 0

    # Daily P&L tracking
    state["daily_pnl"][date] = round(
        state["daily_pnl"].get(date, 0) + pnl, 2
    )

    # Trade history (keep last 50)
    state["trade_history"].append({
        "date":      date,
        "question":  pick.get("question", "")[:60],
        "direction": pick.get("direction", "YES"),
        "category":  pick.get("category", "other"),
        "stake":     stake,
        "pnl":       round(pnl, 2),
        "won":       won,
    })
    state["trade_history"] = state["trade_history"][-50:]

    # Run circuit breaker evaluation
    state = _evaluate_circuit_breaker(state)

    _save(state)
    log.info(
        f"Trade recorded: {'WIN' if won else 'LOSS'} ${abs(pnl):.2f} | "
        f"Bankroll: ${state['current_bankroll']:.2f} | "
        f"Status: {state['status']}"
    )
    return state


def get_position_multiplier() -> tuple[float, str]:
    """
    Returns (multiplier, reason).
    multiplier = 1.0 → full Kelly
    multiplier = 0.5 → half size (drawdown mode)
    multiplier = 0.0 → halted (no new positions)
    """
    state = _load()
    status = state.get("status", "active")

    if status == "halted":
        return 0.0, state.get("status_reason", "Circuit breaker active")
    elif status == "reduced":
        return 0.5, state.get("status_reason", "Drawdown protection: half size")
    else:
        return 1.0, "Normal"


def get_adjusted_kelly(kelly_usd: float) -> tuple[float, str]:
    """
    Apply circuit breaker multiplier to Kelly-sized position.
    Returns (adjusted_usd, reason).
    """
    multiplier, reason = get_position_multiplier()
    adjusted = round(kelly_usd * multiplier, 2)
    return adjusted, reason


def bankroll_summary() -> dict:
    """Full bankroll status for dashboard display."""
    state = _load()

    peak    = state.get("peak_bankroll", config.BANKROLL)
    current = state.get("current_bankroll", config.BANKROLL)
    initial = state.get("initial_bankroll", config.BANKROLL)

    drawdown_from_peak = round((current - peak) / peak * 100, 2) if peak > 0 else 0
    total_return       = round((current - initial) / initial * 100, 2) if initial > 0 else 0

    # Daily P&L last 7 days
    today = datetime.now(timezone.utc).date()
    daily_pnl = state.get("daily_pnl", {})
    last_7 = []
    for i in range(7):
        d = (today - timedelta(days=i)).isoformat()
        last_7.append({"date": d, "pnl": daily_pnl.get(d, 0)})
    last_7.reverse()

    return {
        "current_bankroll":    current,
        "initial_bankroll":    initial,
        "peak_bankroll":       peak,
        "all_time_pnl":        state.get("all_time_pnl", 0),
        "total_return_pct":    total_return,
        "drawdown_from_peak":  drawdown_from_peak,
        "status":              state.get("status", "active"),
        "status_reason":       state.get("status_reason"),
        "consecutive_losses":  state.get("consecutive_losses", 0),
        "consecutive_wins":    state.get("consecutive_wins", 0),
        "daily_pnl_7d":        last_7,
        "trade_history":       state.get("trade_history", [])[-10:],
        "thresholds": {
            "reduce_at_drawdown": f"{DRAWDOWN_REDUCE_PCT:.0%}",
            "halt_at_drawdown":   f"{DRAWDOWN_HALT_PCT:.0%}",
            "halt_at_streak":     LOSS_STREAK_HALT,
            "recovery_streak":    RECOVERY_WIN_STREAK,
        },
    }


# ─── Circuit breaker logic ────────────────────────────────────────────────────

def _evaluate_circuit_breaker(state: dict) -> dict:
    """
    Evaluate all three circuit breaker conditions and update status.
    """
    peak    = state.get("peak_bankroll", config.BANKROLL)
    current = state.get("current_bankroll", config.BANKROLL)
    losses  = state.get("consecutive_losses", 0)
    wins    = state.get("consecutive_wins", 0)

    drawdown = (peak - current) / peak if peak > 0 else 0

    # ── Recovery check (re-enable halted/reduced status) ──────────────────────
    if state.get("status") in ("halted", "reduced"):
        if wins >= RECOVERY_WIN_STREAK and drawdown < DRAWDOWN_REDUCE_PCT:
            state["status"]        = "active"
            state["status_reason"] = f"Recovered: {wins} consecutive wins, drawdown {drawdown:.1%}"
            state["status_since"]  = datetime.now(timezone.utc).isoformat()
            log.info(f"Circuit breaker reset: {state['status_reason']}")
            return state

    # ── Halt conditions ────────────────────────────────────────────────────────
    if drawdown >= DRAWDOWN_HALT_PCT:
        if state.get("status") != "halted":
            state["status"]        = "halted"
            state["status_reason"] = (
                f"Drawdown halt: bankroll fell {drawdown:.1%} from peak "
                f"(${current:.2f} vs peak ${peak:.2f}). "
                f"Need {RECOVERY_WIN_STREAK} consecutive wins to resume."
            )
            state["status_since"]  = datetime.now(timezone.utc).isoformat()
            log.warning(f"CIRCUIT BREAKER HALTED: {state['status_reason']}")
        return state

    if losses >= LOSS_STREAK_HALT:
        if state.get("status") != "halted":
            state["status"]        = "halted"
            state["status_reason"] = (
                f"Streak halt: {losses} consecutive losses. "
                f"Need {RECOVERY_WIN_STREAK} consecutive wins to resume."
            )
            state["status_since"]  = datetime.now(timezone.utc).isoformat()
            log.warning(f"CIRCUIT BREAKER HALTED: {state['status_reason']}")
        return state

    # ── Velocity check (losing >5%/day for 3 consecutive days) ───────────────
    if _velocity_triggered(state):
        if state.get("status") != "halted":
            state["status"]        = "halted"
            state["status_reason"] = (
                f"Velocity halt: losing >{DAILY_LOSS_HALT_PCT:.0%}/day "
                f"for 3 consecutive days."
            )
            state["status_since"]  = datetime.now(timezone.utc).isoformat()
            log.warning(f"CIRCUIT BREAKER HALTED: {state['status_reason']}")
        return state

    # ── Reduce conditions ─────────────────────────────────────────────────────
    if drawdown >= DRAWDOWN_REDUCE_PCT:
        if state.get("status") == "active":
            state["status"]        = "reduced"
            state["status_reason"] = (
                f"Drawdown reduce: {drawdown:.1%} from peak — "
                f"position sizes halved until recovery."
            )
            state["status_since"]  = datetime.now(timezone.utc).isoformat()
            log.warning(f"CIRCUIT BREAKER REDUCED: {state['status_reason']}")
        return state

    # ── All clear ─────────────────────────────────────────────────────────────
    if state.get("status") != "active":
        state["status"]        = "active"
        state["status_reason"] = None
    return state


def _velocity_triggered(state: dict) -> bool:
    """Check if daily loss velocity exceeds threshold for 3 consecutive days."""
    daily_pnl = state.get("daily_pnl", {})
    bankroll  = state.get("current_bankroll", config.BANKROLL)

    today = datetime.now(timezone.utc).date()
    consecutive_bad_days = 0

    for i in range(1, 4):  # check yesterday, day before, day before that
        d    = (today - timedelta(days=i)).isoformat()
        pnl  = daily_pnl.get(d, 0)
        loss_pct = abs(pnl) / max(bankroll, 1) if pnl < 0 else 0
        if loss_pct >= DAILY_LOSS_HALT_PCT:
            consecutive_bad_days += 1

    return consecutive_bad_days >= 3


# ─── Paper trading integration ────────────────────────────────────────────────

def apply_to_picks(picks: dict) -> dict:
    """
    Apply circuit breaker to all picks — adjust or zero out position sizes.
    Returns modified picks with adjusted sizing.
    """
    multiplier, reason = get_position_multiplier()
    state = bankroll_summary()

    if multiplier == 0.0:
        # Halted — flag all picks, zero all sizes
        log.warning(f"Circuit breaker HALTED — zeroing all position sizes")
        for section in ("yes_picks", "no_picks", "macro_plays"):
            for pick in picks.get(section, []):
                if "sizing" in pick:
                    pick["sizing"]["position_usd"] = 0
                    pick["sizing"]["circuit_breaker"] = reason
        picks["circuit_breaker"] = {
            "status": "halted",
            "reason": reason,
            "all_sizes_zeroed": True,
        }
    elif multiplier < 1.0:
        # Reduced — halve all sizes
        log.warning(f"Circuit breaker REDUCED — halving position sizes")
        for section in ("yes_picks", "no_picks", "macro_plays"):
            for pick in picks.get(section, []):
                if "sizing" in pick:
                    original = pick["sizing"].get("position_usd", 0)
                    pick["sizing"]["position_usd"]    = round(original * multiplier, 2)
                    pick["sizing"]["original_usd"]    = original
                    pick["sizing"]["circuit_breaker"] = reason
        picks["circuit_breaker"] = {
            "status":     "reduced",
            "reason":     reason,
            "multiplier": multiplier,
        }
    else:
        picks["circuit_breaker"] = {
            "status":     "active",
            "reason":     "Normal operation",
            "multiplier": 1.0,
        }

    picks["bankroll_summary"] = state
    return picks
