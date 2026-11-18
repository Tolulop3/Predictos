"""
PredictOS — Event-odds resolver (Phase 1, scoping plan 2026-10-29 §3).

Repurposed from the old picks tracker (src/tracker.py): the same job —
"ask the source whether a market we logged has resolved, and record the
outcome" — pointed at feed events instead of picks. Plan §4 needs
resolved event outcomes to score the market's own calibration and to
label evaluation windows; this module produces them.

How resolution is checked (same cascade the tracker uses):
  src.tracker._resolve_polymarket() delegates to
  src.fetcher.check_polymarket_resolution(), which tries the data-api
  resolution stub (currently a no-op — data-api /resolution 404s, see
  fetcher.py) and falls back to the CLOB market state:
  GET clob.polymarket.com/markets/{condition_id} -> closed flag +
  winner flag on the YES token. This module reuses that exact function
  rather than re-implementing it, so the feed inherits the tracker's
  behaviour (and its limits).

Known limitations (honest, per plan §6.2 source fragility):
  * The CLOB cascade reports the YES-token winner for each leg market.
    Every feed leg IS its own binary Polymarket market (ladders are
    groups of binary markets), so resolved_yes maps directly to
    "this outcome occurred" for that row. There is no separate
    multi-outcome resolution endpoint in the cascade.
  * If CLOB delists/ages out a condition_id before this runs, the check
    returns None and the event stays unresolved in the log — it is
    retried on the next run, never guessed.
  * Only Polymarket rows exist in Phase 1, so only the Polymarket leg
    of the cascade is exercised. Kalshi/Manifold checks remain in
    src/tracker.py if those sources are ever added to the feed.

Inputs / outputs (all under data/event_odds/ by default):
  reads   event_odds_history.jsonl   (distinct events past resolution_date)
          resolutions.json           (already-recorded outcomes, if any)
  writes  resolutions.json           (appended with newly resolved rows)

A resolutions row records: event_id, event_family, outcome_label,
source, source_market_id, slug, question, resolution_date, resolved_at,
resolved_yes, outcome_occurred (== resolved_yes for a binary leg), and
the last implied_prob the feed logged for it (context for calibration).

Run:  python -m src.event_resolver [--data-dir data/event_odds] [--dry-run]
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger("predictos.event_resolver")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        log.warning("Skipping corrupt jsonl line")
    return rows


def due_events(history: list[dict], already: set, today: str) -> list[dict]:
    """Distinct feed rows whose resolution_date has passed, not yet recorded.

    Keeps the LAST logged row per (event_id, outcome_label, source) —
    that row's implied_prob is the feed's final pre-resolution reading.
    """
    last: dict[tuple, dict] = {}
    for r in history:
        if r.get("resolution_date", "9999") < today:
            key = (r["event_id"], r["outcome_label"], r["source"])
            if key not in already:
                if key not in last or r["feed_date"] > last[key]["feed_date"]:
                    last[key] = r
    return sorted(last.values(),
                  key=lambda r: (r["resolution_date"], r["event_id"]))


def resolve_one(row: dict) -> dict | None:
    """Check one feed row's market via the tracker's Polymarket cascade."""
    # Imported lazily so --help / dry imports never touch the network stack.
    from src.fetcher import check_polymarket_resolution

    condition_id = row["source_market_id"].split("::")[-1]
    try:
        result = check_polymarket_resolution({"condition_id": condition_id})
    except Exception as e:
        log.debug(f"Resolution check failed {row['source_market_id']}: {e}")
        return None
    if not result or not result.get("resolved"):
        return None
    resolved_yes = bool(result.get("resolved_yes"))
    return {
        "event_id": row["event_id"],
        "event_family": row["event_family"],
        "outcome_label": row["outcome_label"],
        "source": row["source"],
        "source_market_id": row["source_market_id"],
        "slug": row["slug"],
        "question": row["question"],
        "resolution_date": row["resolution_date"],
        "resolved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "resolved_yes": resolved_yes,
        # Each feed leg is a binary market: YES winning == outcome occurred.
        "outcome_occurred": resolved_yes,
        "last_implied_prob": row["implied_prob"],
        "last_feed_date": row["feed_date"],
    }


def run(data_dir: Path, dry_run: bool = False) -> dict:
    today = datetime.now(timezone.utc).date().isoformat()
    history = load_jsonl(data_dir / "event_odds_history.jsonl")

    res_path = data_dir / "resolutions.json"
    existing: list[dict] = []
    if res_path.exists():
        with open(res_path) as f:
            existing = json.load(f)
    already = {(r["event_id"], r["outcome_label"], r["source"])
               for r in existing}

    due = due_events(history, already, today)
    log.info(f"Resolver: {len(due)} feed outcomes past resolution, unchecked")

    newly = []
    for row in due:
        rec = resolve_one(row)
        if rec:
            newly.append(rec)
            log.info(f"Resolved: {rec['event_id']} / {rec['outcome_label']} "
                     f"-> occurred={rec['outcome_occurred']}")

    if newly and not dry_run:
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(res_path, "w") as f:
            json.dump(existing + newly, f, indent=2)
    return {"checked": len(due), "newly_resolved": len(newly),
            "total_resolved": len(existing) + len(newly), "dry_run": dry_run}


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="PredictOS event-odds resolver")
    ap.add_argument("--data-dir", default="data/event_odds",
                    help="Feed data directory (default: data/event_odds)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Check only; do not write resolutions.json")
    args = ap.parse_args(argv)
    summary = run(Path(args.data_dir), dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
