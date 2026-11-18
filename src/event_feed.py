"""
PredictOS — Event-Odds Feed emitter (Phase 1, scoping plan 2026-10-29 §1–§2).

What this module is:
  A NARROWED, log-only fetcher that discovers Polymarket markets for the
  hand-curated canonical event map (src/event_map.json), maps them to
  event families, and writes the daily event-odds snapshot:

    data/event_odds/latest.json                  today's snapshot (overwritten)
    data/event_odds/history/YYYY-MM-DD.json      that day's snapshot
    data/event_odds/event_odds_history.jsonl     append-only evaluation dataset

What it is NOT:
  No scorer, no ML, no sentiment, no picks. This module deliberately does
  not import src.scorer / src.ml_model / src.sentiment / src.learning.
  The only probability it emits is the market's own implied_prob — the
  YES/outcome price from Gamma, clamped 0.01–0.99 exactly as
  src/fetcher.py::_parse_gamma_market clamps it. Predictos's old
  `model_prob` (a fallback heuristic that was never trained) is never
  read, computed, or emitted. That distinction is the whole pivot.

Semantics that matter (plan §2.1):
  * fetched_at is the point-in-time stamp of this run (UTC). Evaluation
    code may only join odds to an InvestOS brief if fetched_at precedes
    the brief's generation time (as-of rule).
  * delta_1d_pp / delta_7d_pp are percentage-POINT changes computed ONLY
    from this feed's own committed history (event_odds_history.jsonl):
    delta_1d vs the most recent prior feed-day's row for the same
    (event_id, outcome_label, source); delta_7d vs the most recent row
    at least 7 feed-days old. No prior row -> null. Deltas are NEVER
    reconstructed from data-api price history and presented as logged.
  * Missing is a value (plan §1.1): a canonical slot with no qualifying
    market today is recorded in the snapshot's `missing` list with a
    reason. It is never filled with a US proxy or a neighbouring market.
  * quality_flag: "ok" = meets the §1.3 admission floor on the day
    (volume_24h >= $10k OR liquidity >= $25k, and spread <= 0.04 where a
    book exists); "thin" = matched a canonical slot but below the floor
    (kept in the feed so thinness is visible, per §1.3's
    stay-in-with-thin rule); "missing" slots appear only in `missing`.
  * Manifold is play money: rows carry play_money = null in Phase 1.
    A Manifold divergence block is a later, separate addition and is
    never averaged into implied_prob (plan §1.3, absolute rule).

Discovery (plan §1, "discovered, not hardcoded"):
  Markets are found at run time via the same Gamma API the existing
  fetcher uses (config.POLYMARKET_GAMMA_BASE, same params shape as
  src/fetcher.py::gamma_fetch_markets):
    1. Gamma /public-search?q=<slot search terms> — per-slot discovery.
    2. Gamma /markets bulk scan, ordered by 24h volume — supplement so a
       search miss does not silently zero a family.
  No slugs are hardcoded anywhere; the map holds search terms + regex
  question patterns only.

Same-day rerun semantics:
  latest.json and history/<today>.json are overwritten (they are the
  day's snapshot, not yet committed upstream until the workflow's commit
  step). In the jsonl, today's lines are REPLACED on a same-day rerun so
  (feed_date, event_id, outcome_label, source) stays unique; across days
  the file is strictly append-only. A committed day is never edited by a
  later day's run.

Run:  python -m src.event_feed [--out-dir data/event_odds]
                               [--bulk-pages 8] [--no-bulk] [--no-search]
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.fetcher import _get, _parse_date  # noqa: E402  (reuse fetcher HTTP + date plumbing)

log = logging.getLogger("predictos.event_feed")

GAMMA = config.POLYMARKET_GAMMA_BASE
MAP_PATH = Path(__file__).resolve().parent / "event_map.json"

# Row fields, in plan §2.1 order. Rows carry exactly these fields.
ROW_FIELDS = [
    "feed_date", "fetched_at", "event_id", "event_family", "question",
    "outcome_label", "source", "source_market_id", "slug",
    "implied_prob", "delta_1d_pp", "delta_7d_pp",
    "volume_24h_usd", "liquidity_usd", "spread",
    "resolution_date", "days_to_resolution", "quality_flag", "play_money",
]


# ─── Small parsing helpers (mirroring src/fetcher.py conventions) ────────────

def _as_list(value):
    """Gamma sometimes returns outcomes/outcomePrices as JSON strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _f(value, default=0.0):
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _clamp_prob(p: float) -> float:
    # Same clamp the old fetcher applies to yes_price (fetcher.py).
    return max(0.01, min(0.99, p))


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s[:60] or "yes"


def _outcome_label(mk: dict) -> str:
    """Ladder legs are named by groupItemTitle; plain binaries are 'yes'.

    Takes the NORMALISED market dict (key: group_item_title).
    Parenthetical qualifiers are dropped first ("0 (0 bps)" -> "0",
    "25 bps decrease" stays itself) so labels stay short and stable.
    """
    git = (mk.get("group_item_title") or "").strip()
    if git:
        git = re.sub(r"\([^)]*\)", "", git).strip()
        return _slugify(git)
    return "yes"


# ─── Gamma discovery (same endpoints/params as src/fetcher.py) ───────────────

def gamma_search_markets(term: str, limit_per_type: int = 10) -> list[dict]:
    """
    Gamma /public-search — returns events with embedded markets.
    Each embedded market is annotated with its parent event slug/title
    (as _event_slug / _event_title) so legs can be grouped into ladders.
    """
    data = _get(f"{GAMMA}/public-search",
                params={"q": term, "limit_per_type": limit_per_type})
    if not data or not isinstance(data, dict):
        return []
    out = []
    for ev in data.get("events") or []:
        for m in ev.get("markets") or []:
            m = dict(m)
            m["_event_slug"] = ev.get("slug", "")
            m["_event_title"] = ev.get("title", "")
            out.append(m)
    return out


def gamma_bulk_markets(pages: int = 8, page_size: int = 100) -> list[dict]:
    """
    Gamma /markets bulk scan — same params as fetcher.gamma_fetch_markets,
    plus order=volume24hr so the highest-signal markets come first.
    Embedded event info (market.events[0]) is used for ladder grouping.
    """
    out = []
    for page in range(pages):
        batch = _get(f"{GAMMA}/markets", params={
            "limit": page_size,
            "offset": page * page_size,
            "active": "true",
            "closed": "false",
            "archived": "false",
            "order": "volume24hr",
            "ascending": "false",
        })
        if not batch or not isinstance(batch, list):
            break
        for m in batch:
            m = dict(m)
            evs = m.get("events") or []
            if evs:
                m["_event_slug"] = evs[0].get("slug", "")
                m["_event_title"] = evs[0].get("title", "")
            out.append(m)
        if len(batch) < page_size:
            break
        time.sleep(0.25)
    return out


# ─── Market normalisation ────────────────────────────────────────────────────

def normalise_market(m: dict, now: datetime) -> dict | None:
    """Raw Gamma market -> the fields the feed needs. None if unusable."""
    question = (m.get("question") or "").strip()
    if not question:
        return None
    if not m.get("active", True) or m.get("closed", False) or m.get("archived", False):
        return None

    outcomes = _as_list(m.get("outcomes"))
    prices = _as_list(m.get("outcomePrices"))
    if not prices:
        return None
    try:
        yes_prob = _clamp_prob(float(prices[0]))
    except (TypeError, ValueError):
        return None

    end_dt = _parse_date(m.get("endDate", "") or m.get("endDateIso", ""))
    if end_dt is None:
        return None
    days = (end_dt - now).total_seconds() / 86400.0

    spread = m.get("spread")
    if spread is None and m.get("bestBid") is not None and m.get("bestAsk") is not None:
        spread = round(_f(m["bestAsk"]) - _f(m["bestBid"]), 4)
    spread = round(_f(spread), 4) if spread is not None else None

    condition_id = m.get("conditionId", "") or m.get("condition_id", "")
    slug = m.get("slug", "") or condition_id
    return {
        "question": question,
        "slug": slug,
        "condition_id": condition_id,
        # Plan §2.1: source_market_id is "polymarket::<condition_id>".
        "source_market_id": f"polymarket::{condition_id or slug}",
        "implied_prob": round(yes_prob, 4),
        "outcomes": outcomes,
        "group_item_title": (m.get("groupItemTitle") or "").strip(),
        "event_slug": m.get("_event_slug", ""),
        "event_title": m.get("_event_title", ""),
        "volume_24h_usd": round(_f(m.get("volume24hr", m.get("volume24hrClob"))), 2),
        "liquidity_usd": round(_f(m.get("liquidityNum", m.get("liquidity"))), 2),
        "spread": spread,
        "resolution_date": end_dt.date().isoformat(),
        "days_to_resolution": round(days, 2),
        "end_dt": end_dt,
    }


# ─── Slot matching ───────────────────────────────────────────────────────────

def _matches_slot(mk: dict, slot: dict) -> bool:
    hay = f"{mk['question']} {mk['event_title']}".lower()
    if any(x.lower() in hay for x in slot.get("exclude_terms", [])):
        return False
    return any(re.search(p, hay) for p in slot["question_patterns"])


def _group_key(mk: dict) -> str:
    return mk["event_slug"] or mk["slug"]


def select_slot_group(candidates: list[dict], slot: dict, admission: dict,
                      now: datetime):
    """
    Group candidates into Gamma-event ladders and pick the nearest group
    that has at least one leg inside the resolution window.
    Returns (legs, group_meta) or (None, reason).
    """
    if not candidates:
        return None, "no_matching_market"

    groups: dict[str, list[dict]] = {}
    for mk in candidates:
        groups.setdefault(_group_key(mk), []).append(mk)

    min_d = admission["min_days_to_resolution"]
    max_d = admission["max_days_to_resolution"]

    scored = []
    nearest_out_of_window = None
    for key, legs in groups.items():
        in_window = [l for l in legs if min_d <= l["days_to_resolution"] <= max_d]
        # Group's resolution horizon = its earliest leg end (decision date
        # for ladders; the single date for binaries / date ladders' first leg).
        horizon = min(l["days_to_resolution"] for l in legs)
        if in_window:
            scored.append((horizon, key, in_window))
        elif nearest_out_of_window is None or horizon < nearest_out_of_window[0]:
            nearest_out_of_window = (horizon, legs[0]["resolution_date"])

    if not scored:
        if nearest_out_of_window is not None:
            horizon, rdate = nearest_out_of_window
            return None, (f"no_market_in_resolution_window "
                          f"(nearest resolves {rdate}, {horizon:.0f}d out; "
                          f"window is {min_d:.0f}-{max_d:.0f}d)")
        return None, "no_matching_market"

    scored.sort(key=lambda t: t[0])
    _, key, legs = scored[0]
    # Cap ladder size: keep the most liquid legs (plan expects a curated
    # board, not a 40-leg sweep). Deterministic order after the cap.
    legs.sort(key=lambda l: l["liquidity_usd"], reverse=True)
    legs = legs[: admission["max_legs_per_event"]]
    legs.sort(key=lambda l: (l["resolution_date"], l["group_item_title"], l["slug"]))
    return legs, {"group_key": key}


def _event_id(slot: dict, legs: list[dict]) -> str:
    """
    Stable event_id for a chosen group: slot + resolution date when all
    legs share one date (decision ladders, binaries); slot + Gamma event
    slug for date ladders whose legs resolve on different dates. Stable
    across days for the same listed group — never silently re-pointed.
    """
    dates = {l["resolution_date"] for l in legs}
    if len(dates) == 1:
        return f"{slot['slot_id']}_{legs[0]['resolution_date']}"
    slug = _slugify(legs[0]["event_slug"] or legs[0]["slug"]).replace("_", "-")
    return f"{slot['slot_id']}_{slug}"[:110]


def _quality(mk: dict, admission: dict) -> str:
    liquid = (mk["volume_24h_usd"] >= admission["volume_24h_min_usd"]
              or mk["liquidity_usd"] >= admission["liquidity_min_usd"])
    spread_ok = mk["spread"] is None or mk["spread"] <= admission["spread_max"]
    return "ok" if (liquid and spread_ok) else "thin"


# ─── History / deltas (committed jsonl only — plan §2.3) ─────────────────────

def load_history(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.exists():
        return []
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    log.warning("Skipping corrupt jsonl line")
    return rows


def compute_deltas(history: list[dict], feed_date: str, event_id: str,
                   outcome_label: str, source: str, current_prob: float):
    """(delta_1d_pp, delta_7d_pp) vs this feed's own prior rows, else nulls."""
    today = datetime.strptime(feed_date, "%Y-%m-%d").date()
    prior = [r for r in history
             if r.get("event_id") == event_id
             and r.get("outcome_label") == outcome_label
             and r.get("source") == source
             and r.get("feed_date", "") < feed_date]
    if not prior:
        return None, None
    prior.sort(key=lambda r: r["feed_date"])
    d1 = round((current_prob - prior[-1]["implied_prob"]) * 100, 2)
    cutoff = (today - timedelta(days=7)).isoformat()
    week = [r for r in prior if r["feed_date"] <= cutoff]
    d7 = round((current_prob - week[-1]["implied_prob"]) * 100, 2) if week else None
    return d1, d7


def write_outputs(out_dir: Path, snapshot: dict, rows: list[dict],
                  feed_date: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history").mkdir(exist_ok=True)

    latest = out_dir / "latest.json"
    day_file = out_dir / "history" / f"{feed_date}.json"
    for path in (latest, day_file):
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)

    jsonl_path = out_dir / "event_odds_history.jsonl"
    prior_lines = []
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    if json.loads(line).get("feed_date") != feed_date:
                        prior_lines.append(line)
                except Exception:
                    prior_lines.append(line)
    with open(jsonl_path, "w") as f:
        for line in prior_lines:
            f.write(line + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return jsonl_path


# ─── Main run ────────────────────────────────────────────────────────────────

def run(out_dir: Path, bulk_pages: int = 8, use_search: bool = True,
        use_bulk: bool = True) -> dict:
    now = datetime.now(timezone.utc)
    feed_date = now.date().isoformat()
    fetched_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(MAP_PATH) as f:
        event_map = json.load(f)
    admission = event_map["admission"]
    slots = event_map["events"]

    # ── Discovery: per-slot Gamma search + one bulk volume scan ──
    raw_by_id: dict[str, dict] = {}

    def _add(markets):
        for m in markets:
            key = m.get("conditionId") or m.get("slug")
            if key and key not in raw_by_id:
                raw_by_id[key] = m

    searched_terms: set[str] = set()
    if use_search:
        for slot in slots:
            for term in slot.get("search_terms", []):
                if term in searched_terms:
                    continue
                searched_terms.add(term)
                try:
                    _add(gamma_search_markets(term))
                except Exception as e:
                    log.warning(f"Search '{term}' failed: {e}")
                time.sleep(0.2)

    markets_scanned = 0
    if use_bulk:
        bulk = gamma_bulk_markets(pages=bulk_pages)
        markets_scanned = len(bulk)
        _add(bulk)

    normalised = []
    for m in raw_by_id.values():
        mk = normalise_market(m, now)
        if mk:
            normalised.append(mk)
    log.info(f"Discovery: {len(raw_by_id)} raw markets, "
             f"{len(normalised)} usable (active, priced, dated)")

    # History is read BEFORE writing anything: deltas come only from
    # prior committed rows (plan §2.1/§2.3).
    jsonl_path = out_dir / "event_odds_history.jsonl"
    history = load_history(jsonl_path)

    rows: list[dict] = []
    missing: list[dict] = []
    claimed: set[str] = set()  # one primary per market (plan §1.3)

    for slot in slots:
        candidates = [mk for mk in normalised
                      if mk["source_market_id"] not in claimed
                      and _matches_slot(mk, slot)]
        legs, meta = select_slot_group(candidates, slot, admission, now)
        if legs is None:
            missing.append({
                "slot_id": slot["slot_id"],
                "event_family": slot["event_family"],
                "title": slot["title"],
                "reason": meta,
                "quality_flag": "missing",
                "implied_prob": None,
            })
            continue
        event_id = _event_id(slot, legs)
        for leg in legs:
            claimed.add(leg["source_market_id"])
            outcome_label = _outcome_label(leg)
            d1, d7 = compute_deltas(history, feed_date, event_id,
                                    outcome_label, "polymarket",
                                    leg["implied_prob"])
            rows.append({
                "feed_date": feed_date,
                "fetched_at": fetched_at,
                "event_id": event_id,
                "event_family": slot["event_family"],
                "question": leg["question"],
                "outcome_label": outcome_label,
                "source": "polymarket",
                "source_market_id": leg["source_market_id"],
                "slug": leg["slug"],
                "implied_prob": leg["implied_prob"],
                "delta_1d_pp": d1,
                "delta_7d_pp": d7,
                "volume_24h_usd": leg["volume_24h_usd"],
                "liquidity_usd": leg["liquidity_usd"],
                "spread": leg["spread"],
                "resolution_date": leg["resolution_date"],
                "days_to_resolution": int(round(leg["days_to_resolution"])),
                "quality_flag": _quality(leg, admission),
                # Phase 1 is Polymarket-only (plan §1.3). Manifold, when
                # added, goes here as a separate block — never averaged.
                "play_money": None,
            })

    snapshot = {
        "feed": "predictos_event_odds",
        "feed_version": 1,
        "feed_date": feed_date,
        "fetched_at": fetched_at,
        "source": "polymarket",
        "markets_scanned_bulk": markets_scanned,
        "markets_discovered": len(normalised),
        "events_emitted": len({r["event_id"] for r in rows}),
        "rows_emitted": len(rows),
        "missing_count": len(missing),
        "events": rows,
        "missing": missing,
    }
    write_outputs(out_dir, snapshot, rows, feed_date)

    by_family: dict[str, int] = {}
    for r in rows:
        by_family[r["event_family"]] = by_family.get(r["event_family"], 0) + 1
    log.info(f"Feed {feed_date}: {len(rows)} rows across "
             f"{snapshot['events_emitted']} events; {len(missing)} slots missing")
    log.info(f"By family (rows): {by_family}")
    return snapshot


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="PredictOS event-odds feed emitter")
    ap.add_argument("--out-dir", default="data/event_odds",
                    help="Output directory (default: data/event_odds)")
    ap.add_argument("--bulk-pages", type=int, default=8,
                    help="Gamma /markets pages of 100 to scan (default 8)")
    ap.add_argument("--no-search", action="store_true",
                    help="Skip per-slot Gamma /public-search discovery")
    ap.add_argument("--no-bulk", action="store_true",
                    help="Skip the Gamma /markets bulk scan")
    args = ap.parse_args(argv)

    snapshot = run(Path(args.out_dir), bulk_pages=args.bulk_pages,
                   use_search=not args.no_search, use_bulk=not args.no_bulk)
    print(json.dumps({k: snapshot[k] for k in
                      ("feed_date", "fetched_at", "markets_scanned_bulk",
                       "markets_discovered", "events_emitted",
                       "rows_emitted", "missing_count")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
