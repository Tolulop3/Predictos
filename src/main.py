"""
PredictOS — Main Orchestrator  (v3 — live validation + tick store + circuit breaker + paper trading)

Pipeline:
  0a. Auto-resolve previous picks (outcome tracker)
  0b. Sync paper trade resolutions
  0c. Run learning cycle (calibrate + retrain if ready)
  1.  Fetch markets from all sources
  2.  Save tick snapshots (builds momentum history over time)
  3.  Enrich markets with tick-based momentum (replaces snapshot deltas)
  4.  Fetch news articles
  5.  Load ML model + calibrator + win rates
  6.  Score each market (5 pillars)
  7.  Rank picks
  8.  Live validation — re-check every pick against Polymarket CLOB right now
  9.  Portfolio correlation guard (built into live_validator)
  10. Apply circuit breaker to position sizes
  11. Open paper trades for all validated picks
  12. Save picks + scores
  13. Record picks in learning store
  14. Bake dashboard
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from src.fetcher       import fetch_all_markets, fetch_news_articles
from src.sentiment     import compute_sentiment, sentiment_to_probability
from src.scorer        import score_market, rank_markets
from src.ml_model      import get_model, compute_category_win_rates
from src.calibration   import get_calibrator
from src.returns       import full_return_estimate
from src.learning      import run_learning_cycle, record_new_picks
from src.tick_store    import save_ticks_batch, enrich_markets_from_ticks, tick_store_stats
from src.live_validator import validate_all_picks
from src.bankroll      import apply_to_picks, bankroll_summary, record_trade_result
from src.paper_trading import open_paper_trade, sync_paper_resolutions, paper_summary
from src.tracker       import (
    save_picks, load_outcomes, auto_resolve_picks,
    compute_stats, generate_daily_tweet
)
from src.dashboard          import bake_dashboard

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predictos.main")


def run_pipeline(save_scores: bool = True) -> dict:
    log.info("═" * 60)
    log.info(f"PredictOS v{config.VERSION} — Daily Run")
    log.info(f"Bankroll: ${config.BANKROLL:,.2f}")
    log.info("═" * 60)

    # ── Step 0a: Auto-resolve previous picks ──────────────────────────────────
    log.info("Step 0a: Resolving previous picks…")
    try:
        resolve_report  = auto_resolve_picks()
        newly_resolved  = resolve_report.get("picks", [])
        log.info(f"  Resolved: {resolve_report['newly_resolved']} new, "
                 f"{resolve_report['total_resolved']} total")
    except Exception as e:
        log.error(f"Resolution step failed: {e}")
        resolve_report = {}
        newly_resolved = []

    # ── Step 0b: Sync paper trade resolutions ─────────────────────────────────
    try:
        outcomes_for_paper = load_outcomes()
        sync_paper_resolutions(outcomes_for_paper)
    except Exception as e:
        log.error(f"Paper trade sync failed: {e}")

    # Record actual P&L in bankroll manager for resolved picks
    for pick in newly_resolved:
        try:
            sizing = pick.get("sizing", {})
            stake  = sizing.get("position_usd", 0)
            if stake and stake > 0:
                yes_price = pick.get("yes_price", 0.5)
                direction = pick.get("direction", "YES")
                entry     = yes_price if direction == "YES" else (1 - yes_price)
                won       = pick.get("won", False)
                pnl       = round((stake / max(entry, 0.001)) - stake, 2) if won else -stake
                record_trade_result(stake, pnl, pick)
        except Exception as e:
            log.debug(f"Bankroll record failed: {e}")

    # ── Step 0c: Learning cycle ────────────────────────────────────────────────
    log.info("Step 0c: Running learning cycle…")
    try:
        learning_result   = run_learning_cycle(newly_resolved_picks=newly_resolved)
        calibration_error = learning_result["calibration"].get("calibration_error", 0.08)
        n_resolved        = learning_result["n_resolved"]
        log.info(f"  Learning: {n_resolved} outcomes, "
                 f"cal_error={calibration_error:.3f}, "
                 f"retrained={learning_result['retrained']}")
    except Exception as e:
        log.error(f"Learning cycle failed: {e}")
        learning_result   = {}
        calibration_error = 0.08
        n_resolved        = 0

    # ── Step 1: Fetch markets ──────────────────────────────────────────────────
    log.info("Step 1: Fetching markets…")
    markets = fetch_all_markets()
    log.info(f"  {len(markets)} markets fetched")
    if not markets:
        log.error("No markets fetched — aborting")
        return {"error": "no_markets"}

    # ── Step 2: Save tick snapshots ────────────────────────────────────────────
    log.info("Step 2: Saving tick snapshots…")
    try:
        save_ticks_batch(markets)
        ts = tick_store_stats()
        log.info(f"  Tick store: {ts['total_markets']} markets, {ts['total_ticks']} total ticks")
    except Exception as e:
        log.error(f"Tick store failed: {e}")

    # ── Step 3: Enrich momentum from tick history ──────────────────────────────
    log.info("Step 3: Enriching momentum from tick history…")
    try:
        markets = enrich_markets_from_ticks(markets)
    except Exception as e:
        log.error(f"Tick momentum enrichment failed: {e}")

    # ── Step 4: Fetch news ─────────────────────────────────────────────────────
    log.info("Step 4: Fetching news…")
    all_articles = fetch_news_articles(days=3)
    log.info(f"  {len(all_articles)} articles fetched")

    # ── Step 5: Load ML model + calibrator ────────────────────────────────────
    log.info("Step 5: Loading ML model + calibrator…")
    model      = get_model()
    calibrator = get_calibrator()
    outcomes   = load_outcomes()
    category_win_rates = compute_category_win_rates(outcomes)
    log.info(f"  Calibrator method: {calibrator.method}, "
             f"Category win rates: {category_win_rates}")

    # ── Step 6: Score each market ──────────────────────────────────────────────
    log.info("Step 6: Scoring markets…")
    scored = []
    sentiment_scores_by_id: dict[str, float] = {}
    model_probs_by_id:      dict[str, float] = {}

    for market in markets:
        try:
            sent_result = compute_sentiment(
                all_articles,
                market["question"],
                category=market.get("category", "other"),
            )
            sent_score  = sent_result.get("score", 0.5)

            # Raw ML score → calibrated probability
            raw_ml_prob      = model.predict_single(market, sent_score)
            calibrated_prob  = calibrator.predict(raw_ml_prob)

            # When no trained model: calibrated_prob ≈ yes_price + sentiment/momentum adj
            # Don't dilute it back toward market price — use it directly as model_prob
            # Only blend with news_prob once we have a real trained model
            if model.is_trained:
                news_prob  = sentiment_to_probability(sent_score, prior=market.get("yes_price", 0.5))
                model_prob = 0.65 * calibrated_prob + 0.35 * news_prob
            else:
                # Fallback: use calibrated_prob directly (already incorporates sentiment+momentum)
                model_prob = calibrated_prob

            result = score_market(
                market             = market,
                sentiment_result   = sent_result,
                model_prob         = model_prob,
                ml_confidence      = abs(calibrated_prob - 0.5) * 2,
                category_win_rates = category_win_rates,
                model_is_trained   = model.is_trained,
            )

            # Return estimate with per-category calibration error
            cat_cal_error = _get_category_cal_error(
                market.get("category", "other"),
                learning_result.get("calibration", {}).get("by_category", {}),
                default=calibration_error,
            )
            stake = result.get("sizing", {}).get("position_usd", 50)
            result["return_estimate"] = full_return_estimate(
                stake                      = max(stake, 1),
                pick                       = result,
                n_resolved_outcomes        = n_resolved,
                category_calibration_error = cat_cal_error,
            )

            scored.append(result)
            mid = market.get("id", "")
            sentiment_scores_by_id[mid] = sent_score
            model_probs_by_id[mid]      = model_prob

        except Exception as e:
            log.debug(f"Score failed for {market.get('id','?')}: {e}")

    log.info(f"  {len(scored)} markets scored")

    # ── Step 7: Rank picks ─────────────────────────────────────────────────────
    log.info("Step 7: Ranking picks…")
    yes_picks, no_picks = rank_markets(scored)
    macro_plays         = _identify_macro_plays(yes_picks + no_picks)

    picks_pre_validation = {
        "yes_picks":   [_serialise(m) for m in yes_picks[:8]],   # oversample for validation
        "no_picks":    [_serialise(m) for m in no_picks[:5]],
        "macro_plays": [_serialise(m) for m in macro_plays[:4]],
        "universe":    len(scored),
        "filtered_in": len(yes_picks) + len(no_picks),
        "run_date":    datetime.now(timezone.utc).date().isoformat(),
    }

    # ── Step 8: Live validation ────────────────────────────────────────────────
    log.info("Step 8: Live validation against Polymarket CLOB…")
    try:
        validated_picks = validate_all_picks(picks_pre_validation, model_probs_by_id)
        n_rejected = len(validated_picks.get("rejected_picks", []))
        n_flags    = len(validated_picks.get("correlation_flags", []))
        log.info(f"  Rejected by live validation: {n_rejected}, correlation flags: {n_flags}")
    except Exception as e:
        log.error(f"Live validation failed: {e}")
        validated_picks = picks_pre_validation
        validated_picks.setdefault("rejected_picks", [])
        validated_picks.setdefault("correlation_flags", [])

    # Trim to final pick counts after validation
    validated_picks["yes_picks"]   = validated_picks.get("yes_picks", [])[:5]
    validated_picks["no_picks"]    = validated_picks.get("no_picks", [])[:3]
    validated_picks["macro_plays"] = validated_picks.get("macro_plays", [])[:3]

    # ── Step 9: Apply circuit breaker to position sizes ───────────────────────
    log.info("Step 9: Applying circuit breaker…")
    picks = apply_to_picks(validated_picks)
    cb_status = picks.get("circuit_breaker", {}).get("status", "active")
    log.info(f"  Circuit breaker: {cb_status}")

    # ── Step 10: Open paper trades ─────────────────────────────────────────────
    log.info("Step 10: Opening paper trades…")
    paper_opened = 0
    all_final_picks = (
        picks.get("yes_picks", []) +
        picks.get("no_picks", []) +
        picks.get("macro_plays", [])
    )
    for pick in all_final_picks:
        try:
            mid   = pick.get("id", "")
            mprob = model_probs_by_id.get(mid, pick.get("model_prob", 0.5))
            trade = open_paper_trade(pick, mprob)
            if trade:
                paper_opened += 1
        except Exception as e:
            log.debug(f"Paper trade open failed: {e}")
    log.info(f"  Paper trades opened: {paper_opened}")

    # ── Step 11: Save picks ────────────────────────────────────────────────────
    save_picks(picks)
    log.info(f"  Picks: {len(picks.get('yes_picks',[]))} YES, "
             f"{len(picks.get('no_picks',[]))} NO, "
             f"{len(picks.get('macro_plays',[]))} macro")

    # ── Step 12: Record in learning store ──────────────────────────────────────
    try:
        record_new_picks(picks, scored, sentiment_scores_by_id, model_probs_by_id)
    except Exception as e:
        log.error(f"Learning store record failed: {e}")

    # ── Step 13: Save scores ───────────────────────────────────────────────────
    if save_scores:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(config.SCORES_FILE, "w") as f:
            json.dump([_serialise(m) for m in scored[:100]], f, indent=2)

    # ── Step 14: Bake dashboard ────────────────────────────────────────────────
    log.info("Step 14: Baking dashboard…")
    stats = compute_stats(outcomes)
    tweet = generate_daily_tweet(picks, stats)

    dashboard_data = {
        "picks":           picks,
        "stats":           stats,
        "tweet":           tweet,
        "model_info":      model.feature_importance,
        "learning":        learning_result.get("report", {}),
        "calibration":     learning_result.get("calibration", {}),
        "bankroll":        bankroll_summary(),
        "paper_trading":   paper_summary(),
        "tick_store":      tick_store_stats(),
        "run_date":        datetime.now(timezone.utc).isoformat(),
    }
    bake_dashboard(dashboard_data)
    log.info(f"  Dashboard → {config.DASHBOARD_FILE}")

    # ── Done ──────────────────────────────────────────────────────────────────
    top_yes = picks.get("yes_picks", [])
    log.info("═" * 60)
    log.info("Run complete!")
    log.info(f"  Top YES: {top_yes[0]['question'][:50] if top_yes else 'none'}")
    log.info(f"  CB status: {cb_status}")
    log.info(f"  Win rate: {stats.get('win_rate', 0):.1%} ({stats.get('total',0)} resolved)")
    log.info("═" * 60)

    return {"picks": picks, "stats": stats, "tweet": tweet,
            "resolve_report": resolve_report}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_category_cal_error(cat: str, cal_by_cat: dict, default: float = 0.08) -> float:
    return cal_by_cat.get(cat, {}).get("cal_error", default)

def _identify_macro_plays(picks: list[dict]) -> list[dict]:
    kw = config.MACRO_KEYWORDS
    return [{**p, "macro": True} for p in picks if any(k in p.get("question","").lower() for k in kw)]

def _serialise(m: dict) -> dict:
    skip = {"_momentum_score", "_odds_rank"}
    return {k: v for k, v in m.items() if k not in skip and not k.startswith("_compiled")}


if __name__ == "__main__":
    result = run_pipeline()
    print(f"\nDone. Win rate: {result['stats'].get('win_rate', 0):.1%}")
