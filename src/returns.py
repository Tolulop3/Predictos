"""
PredictOS — Return Calculator
Computes accurate, mathematically grounded return estimates for every pick.

The key formula for a binary prediction market:
  - You buy YES shares at price P (e.g. $0.40 = 40¢ per share)
  - Each share pays $1.00 if YES resolves, $0 if NO resolves
  - Your stake buys (stake / P) shares
  - Gross return if WIN:  (stake / P) * 1.00
  - Net profit if WIN:    stake * (1/P - 1)  =  stake * (1 - P) / P
  - Net profit if LOSE:   -stake

Example: $50 stake on YES at 40¢
  → Buy 125 shares
  → Win: $125 gross, $75 net profit (+150% on stake)
  → Lose: -$50

The model adds:
  - Expected value (EV) using model probability, not market price
  - Confidence interval on return using calibration error
  - Break-even probability (what prob needed to make this +EV?)
  - Calibration-adjusted return (shrinks toward base rate as uncertainty rises)
"""

import math
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─── Core Return Math ─────────────────────────────────────────────────────────

def compute_return(stake: float, yes_price: float,
                   direction: str = "YES") -> dict:
    """
    Exact binary market return calculation.

    stake      : dollars you are committing
    yes_price  : market implied probability (e.g. 0.40)
    direction  : 'YES' (buying YES shares) or 'NO' (buying NO shares)
    """
    yes_price = max(0.001, min(0.999, yes_price))
    no_price  = 1 - yes_price

    if direction == "YES":
        entry_price   = yes_price
        payout_per_share = 1.0       # each YES share pays $1 if YES resolves
    else:
        entry_price   = no_price
        payout_per_share = 1.0       # each NO share pays $1 if NO resolves

    shares         = stake / entry_price
    gross_win      = shares * payout_per_share
    net_profit_win = gross_win - stake       # what you keep above your stake
    pct_return_win = net_profit_win / stake * 100

    # Decimal odds format: e.g. 2.5x means stake × 2.5 returned if win
    decimal_odds = gross_win / stake

    return {
        "stake":            round(stake, 2),
        "entry_price":      round(entry_price, 4),
        "shares":           round(shares, 4),
        "gross_win":        round(gross_win, 2),
        "net_profit_win":   round(net_profit_win, 2),
        "net_loss":         round(-stake, 2),
        "pct_return_win":   round(pct_return_win, 1),
        "decimal_odds":     round(decimal_odds, 3),
        "direction":        direction,
    }


def compute_expected_value(stake: float, yes_price: float, model_prob: float,
                           direction: str = "YES") -> dict:
    """
    Expected value calculation using MODEL probability (not market price).
    This is the key edge signal: if model_prob > market_price, there is +EV.

    EV = P(win) × net_profit_win + P(lose) × net_loss
    """
    ret = compute_return(stake, yes_price, direction)

    if direction == "YES":
        p_win  = model_prob
        p_lose = 1 - model_prob
    else:
        p_win  = 1 - model_prob  # NO wins when YES does NOT resolve
        p_lose = model_prob

    ev         = p_win * ret["net_profit_win"] + p_lose * ret["net_loss"]
    ev_pct     = ev / stake * 100
    roi        = ev / stake

    # Break-even probability: minimum win rate needed for +EV
    # 0 = p_be * net_profit_win + (1 - p_be) * net_loss
    # p_be = stake / (stake + net_profit_win) = entry_price
    break_even_prob = ret["entry_price"]

    # Edge: how much better is model vs market?
    if direction == "YES":
        edge = model_prob - yes_price
    else:
        edge = (1 - model_prob) - (1 - yes_price)  # = yes_price - model_prob

    return {
        **ret,
        "model_prob":       round(model_prob, 4),
        "p_win":            round(p_win, 4),
        "p_lose":           round(p_lose, 4),
        "expected_value":   round(ev, 2),
        "ev_pct":           round(ev_pct, 2),
        "roi":              round(roi, 4),
        "break_even_prob":  round(break_even_prob, 4),
        "edge":             round(edge, 4),
        "is_positive_ev":   ev > 0,
    }


def compute_confidence_interval(ev_result: dict, calibration_error: float = 0.08,
                                n_outcomes: int = 50) -> dict:
    """
    Add a confidence interval around the return estimate.

    calibration_error : typical model probability error (starts high, shrinks over time)
    n_outcomes        : number of resolved outcomes (more = tighter CI)

    Uses normal approximation. The interval widens when:
      - calibration_error is large (model not yet well-calibrated)
      - n_outcomes is small (not enough historical data)
    """
    # Effective calibration error shrinks as we accumulate outcomes
    # Bayesian shrinkage: after N outcomes, error reduces by sqrt(N/50)
    shrinkage    = min(1.0, math.sqrt(max(n_outcomes, 1) / 50))
    eff_cal_err  = calibration_error * (2 - shrinkage)  # starts 2× base, converges to base

    p_win    = ev_result["p_win"]
    stake    = ev_result["stake"]
    net_win  = ev_result["net_profit_win"]
    net_lose = ev_result["net_loss"]

    # Variance of outcome
    ev_sq = p_win * (net_win ** 2) + (1 - p_win) * (net_lose ** 2)
    variance_outcome = ev_sq - ev_result["expected_value"] ** 2

    # Add calibration uncertainty: if p_win is off by eff_cal_err
    # how much does EV shift?
    ev_high = compute_expected_value(
        stake, ev_result["entry_price"],
        min(0.99, p_win + eff_cal_err), ev_result["direction"]
    )["expected_value"]
    ev_low = compute_expected_value(
        stake, ev_result["entry_price"],
        max(0.01, p_win - eff_cal_err), ev_result["direction"]
    )["expected_value"]

    return {
        "ev_low":           round(ev_low, 2),
        "ev_central":       round(ev_result["expected_value"], 2),
        "ev_high":          round(ev_high, 2),
        "return_low":       round(ev_low / stake * 100, 1),
        "return_central":   round(ev_result["ev_pct"], 1),
        "return_high":      round(ev_high / stake * 100, 1),
        "calibration_error": round(eff_cal_err, 4),
        "confidence_note":  _confidence_note(n_outcomes, eff_cal_err),
    }


def _confidence_note(n: int, err: float) -> str:
    if n < 10:
        return f"Wide CI — only {n} resolved outcomes. Return range will tighten over time."
    elif n < 30:
        return f"Moderate CI — {n} outcomes. Model calibrating."
    elif err < 0.06:
        return f"Tight CI — {n} outcomes, well-calibrated model."
    else:
        return f"Good CI — {n} outcomes."


def full_return_estimate(stake: float, pick: dict,
                         n_resolved_outcomes: int = 0,
                         category_calibration_error: float = 0.08) -> dict:
    """
    Master function: compute complete return estimate for a pick.
    Returns everything the dashboard needs to display.

    pick must have: yes_price, model_prob, direction, question, category
    """
    yes_price  = pick.get("yes_price", 0.5)
    model_prob = pick.get("model_prob", yes_price)
    direction  = pick.get("direction", "YES")

    # Core EV
    ev = compute_expected_value(stake, yes_price, model_prob, direction)

    # Confidence interval
    ci = compute_confidence_interval(ev, category_calibration_error,
                                     n_resolved_outcomes)

    # Scenario table (what if model prob is wrong by various amounts)
    scenarios = _scenario_table(stake, yes_price, direction, model_prob)

    # Plain-English summary
    summary = _plain_english(stake, yes_price, model_prob, direction, ev, ci)

    return {
        "stake":           stake,
        "direction":       direction,
        "entry_price":     ev["entry_price"],
        "shares":          ev["shares"],
        # If win
        "gross_win":       ev["gross_win"],
        "net_profit":      ev["net_profit_win"],
        "pct_return":      ev["pct_return_win"],
        "decimal_odds":    ev["decimal_odds"],
        # If lose
        "net_loss":        ev["net_loss"],
        # EV
        "expected_value":  ev["expected_value"],
        "ev_pct":          ev["ev_pct"],
        "is_positive_ev":  ev["is_positive_ev"],
        "edge":            ev["edge"],
        "break_even_prob": ev["break_even_prob"],
        # Confidence interval
        "ci":              ci,
        # Scenarios
        "scenarios":       scenarios,
        # Summary
        "summary":         summary,
    }


def _scenario_table(stake: float, yes_price: float, direction: str,
                    model_prob: float) -> list[dict]:
    """What the return looks like under different model probability assumptions."""
    rows = []
    for label, p_delta in [
        ("Bear case",    -0.15),
        ("Base case",    0.0),
        ("Bull case",    +0.15),
        ("Strong bull",  +0.25),
    ]:
        p = max(0.01, min(0.99, model_prob + p_delta))
        ev = compute_expected_value(stake, yes_price, p, direction)
        rows.append({
            "label":          label,
            "model_prob":     round(p, 3),
            "expected_value": ev["expected_value"],
            "ev_pct":         ev["ev_pct"],
            "is_positive_ev": ev["is_positive_ev"],
        })
    return rows


def _plain_english(stake: float, yes_price: float, model_prob: float,
                   direction: str, ev: dict, ci: dict) -> str:
    """Human-readable summary of the trade."""
    net_win  = ev["net_profit_win"]
    net_lose = ev["net_loss"]
    ev_val   = ev["expected_value"]
    edge     = ev["edge"]
    be_prob  = ev["break_even_prob"]

    dir_str  = direction
    prob_str = f"{model_prob:.0%}"
    mkt_str  = f"{yes_price:.0%}"

    lines = [
        f"You stake ${stake:.2f} on {dir_str} at {mkt_str} market price.",
        f"Model estimates {prob_str} chance of {dir_str} resolving — vs {mkt_str} market price.",
        f"If correct: +${net_win:.2f} net profit ({ev['pct_return_win']:.0f}% return).",
        f"If wrong: -${abs(net_lose):.2f} (your full stake).",
        f"Expected value: {'+'if ev_val>=0 else ''}{ev_val:.2f} per trade at this stake.",
        f"Break-even: you need {be_prob:.0%} win rate just to not lose money.",
    ]
    if abs(edge) > 0.05:
        lines.append(
            f"Edge: model sees {abs(edge)*100:.1f}% {'discount' if edge>0 else 'premium'} "
            f"vs market — {'positive' if edge>0 else 'negative'} expected value."
        )
    lines.append(ci["confidence_note"])
    return " ".join(lines)


# ─── Log Returns & Exit Signals (from spec) ───────────────────────────────────

def log_return(price_entry: float, price_exit: float) -> float:
    """
    Log return: ln(P1 / P0).
    Additive over time — correct for compound return calculation.
    Log returns are symmetric: -50% and +100% have equal magnitude.
    """
    price_entry = max(0.001, min(0.999, price_entry))
    price_exit  = max(0.001, min(0.999, price_exit))
    return round(math.log(price_exit / price_entry), 6)


def should_exit(market_price: float, model_prob: float,
                days_to_expiry: float, direction: str = "YES") -> dict:
    """
    Exit rule from spec:
      hit_target = market_price >= model_probability * 0.9
      hit_expiry = days_to_expiry <= 7

    Returns dict with exit signal and reason.
    For NO bets: use no_price and model_no_prob.
    """
    if direction == "YES":
        entry_price   = market_price
        model_p       = model_prob
    else:
        entry_price   = 1 - market_price   # NO price
        model_p       = 1 - model_prob      # model NO prob

    hit_target = entry_price >= model_p * 0.9
    hit_expiry = days_to_expiry <= 7

    if hit_target and hit_expiry:
        reason = "target_and_expiry"
        should = True
    elif hit_target:
        reason = "target_reached"   # market corrected to 90% of model value
        should = True
    elif hit_expiry:
        reason = "near_expiry"      # 7 days left — close to avoid binary event risk
        should = True
    else:
        reason = "hold"
        should = False

    pct_of_target = round(entry_price / (model_p * 0.9) * 100, 1) if model_p > 0 else 0

    return {
        "should_exit":    should,
        "reason":         reason,
        "hit_target":     hit_target,
        "hit_expiry":     hit_expiry,
        "pct_of_target":  pct_of_target,
        "current_price":  round(entry_price, 4),
        "target_price":   round(model_p * 0.9, 4),
        "days_remaining": round(days_to_expiry, 1),
    }


def sharpe_contribution(log_ret: float, position_size: float,
                        risk_free_daily: float = 0.0001) -> dict:
    """
    Contribution of a single trade to overall Sharpe Ratio.
    SR = (Rp - Rf) / σ  where Rp and Rf are log returns.

    risk_free_daily: daily risk-free rate (default ~3.6% annual / 365)
    """
    excess_return = log_ret - risk_free_daily
    # Size-weighted contribution
    weighted = excess_return * position_size
    return {
        "log_return":       round(log_ret, 6),
        "excess_return":    round(excess_return, 6),
        "weighted_contrib": round(weighted, 6),
        "position_size":    round(position_size, 2),
    }
