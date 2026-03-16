"""
PredictOS — Data Fetcher  (v2 — full three-API Polymarket integration)

Three Polymarket APIs, each used for what it does best:

  gamma-api.polymarket.com   — Market discovery + metadata + event grouping
  ├─ /markets                  Paginated list of all markets (best for bulk scan)
  ├─ /events                   Events with embedded markets (category/tag data)
  └─ /markets/{slug}           Single market detail, tags, description

  data-api.polymarket.com    — Historical data + time-series
  ├─ /prices                   OHLCV candle data per market/token
  ├─ /activity                 Trade activity (buys/sells) for volume breakdown
  └─ /resolution               Resolved market results for outcome tracking

  clob.polymarket.com        — Live order book + real-time pricing
  ├─ /markets                  Live market state, best bid/ask, spread
  ├─ /prices-history           Minute-level price ticks for momentum calc
  ├─ /book                     Full order book depth (liquidity scoring)
  └─ /last-trade-price         Single most recent trade price (fast poll)

Pipeline per market:
  1. gamma-api  →  discover markets, get slugs, tags, descriptions, event context
  2. clob        →  enrich with live bid/ask, spread, order book depth
  3. data-api    →  pull 24h + 72h price history for momentum calculation
  4. data-api    →  pull 24h volume candles for volume pillar accuracy
"""

import time
import logging
import requests
import feedparser
from datetime import datetime, timezone, timedelta
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger("predictos.fetch")

# ─── HTTP session ─────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "PredictOS/1.0 (github-actions-bot)",
    "Accept":     "application/json",
})

def _get(url: str, params: dict = None, timeout: int = 15,
         headers: dict = None) -> Optional[dict | list]:
    try:
        r = SESSION.get(url, params=params, timeout=timeout, headers=headers or {})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"GET {url} — {e}")
        return None

def _market_id(source: str, raw_id: str) -> str:
    return f"{source}::{raw_id}"


# ══════════════════════════════════════════════════════════════════════════════
#  GAMMA-API  — discovery + metadata
#  Best endpoint for bulk market scanning: richer metadata than CLOB /markets
# ══════════════════════════════════════════════════════════════════════════════

def gamma_fetch_markets(limit: int = 100, offset: int = 0,
                        active: bool = True) -> list[dict]:
    """
    Gamma /markets — paginated market list with full metadata.
    Returns richer data than CLOB: tags, description, event context, images.
    """
    params = {
        "limit":    limit,
        "offset":   offset,
        "active":   str(active).lower(),
        "closed":   "false",
        "archived": "false",
    }
    data = _get(f"{config.POLYMARKET_GAMMA_BASE}/markets", params=params)
    if not data:
        return []
    # Gamma returns a list directly (not wrapped in {"data": [...]})
    return data if isinstance(data, list) else data.get("data", [])


def gamma_fetch_events(limit: int = 50, offset: int = 0,
                       active: bool = True) -> list[dict]:
    """
    Gamma /events — events that group multiple related markets.
    Useful for identifying multi-outcome events and macro plays.
    Each event contains an embedded list of markets.
    """
    params = {
        "limit":  limit,
        "offset": offset,
        "active": str(active).lower(),
        "closed": "false",
    }
    data = _get(f"{config.POLYMARKET_GAMMA_BASE}/events", params=params)
    if not data:
        return []
    return data if isinstance(data, list) else data.get("data", [])


def gamma_fetch_market_detail(market_slug: str) -> Optional[dict]:
    """Gamma /markets/{slug} — single market with full tag/description data."""
    return _get(f"{config.POLYMARKET_GAMMA_BASE}/markets/{market_slug}")


def _parse_gamma_market(m: dict) -> Optional[dict]:
    """Parse a Gamma API market into PredictOS standard schema."""
    mid      = m.get("id", "")
    slug     = m.get("slug", mid)
    question = m.get("question", "") or m.get("title", "")
    if not question:
        return None

    # Price: gamma exposes bestYesBid or outcomePrices
    outcome_prices = m.get("outcomePrices", [])  # e.g. ["0.62", "0.38"]
    try:
        yes_price = float(outcome_prices[0]) if outcome_prices else float(m.get("bestYesBid", 0.5) or 0.5)
    except Exception:
        yes_price = 0.5

    yes_price = max(0.01, min(0.99, yes_price))

    volume_24h = float(m.get("volume24hr", 0) or m.get("oneDayVolume", 0) or 0)
    liquidity  = float(m.get("liquidity", 0) or 0)
    volume_tot = float(m.get("volume", 0) or 0)

    # Tags (richer in Gamma than CLOB)
    tags = [t.get("label", "") for t in (m.get("tags") or []) if t.get("label")]

    end_date    = _parse_date(m.get("endDate", "") or m.get("end_date_iso", ""))
    days_to_res = _days_until(end_date)

    # Condition / token IDs needed for CLOB + data-api enrichment
    condition_id = m.get("conditionId", "") or m.get("condition_id", "")
    token_ids    = [t.get("token_id", "") for t in (m.get("tokens") or []) if t.get("token_id")]

    return {
        "id":              _market_id("polymarket", condition_id or slug),
        "source":          "polymarket",
        "slug":            slug,
        "condition_id":    condition_id,
        "token_ids":       token_ids,
        "question":        question,
        "description":     m.get("description", ""),
        "tags":            tags,
        "category":        _classify_topic(question + " " + " ".join(tags)),
        "yes_price":       round(yes_price, 4),
        "no_price":        round(1 - yes_price, 4),
        "volume_24h":      volume_24h,
        "volume_total":    volume_tot,
        "liquidity":       liquidity,
        "end_date":        end_date.isoformat() if end_date else None,
        "days_to_res":     days_to_res,
        "active":          m.get("active", True),
        "closed":          m.get("closed", False),
        "url":             f"https://polymarket.com/event/{slug}",
        "fetched_at":      datetime.now(timezone.utc).isoformat(),
        # Placeholders enriched by CLOB + data-api below
        "yes_price_24h_ago": yes_price,
        "yes_price_72h_ago": yes_price,
        "spread":            None,
        "best_bid":          None,
        "best_ask":          None,
        "book_depth":        0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLOB  — live order book + real-time pricing
# ══════════════════════════════════════════════════════════════════════════════

def clob_enrich_market(market: dict) -> dict:
    """
    Enrich a Gamma-parsed market with live CLOB data:
      • Real-time best bid/ask (more accurate than Gamma's snapshot)
      • Bid-ask spread as a liquidity quality signal
      • Order-book depth for volume pillar augmentation
    """
    condition_id = market.get("condition_id", "")
    if not condition_id:
        return market

    # 1. Live market state from CLOB
    clob_data = _get(f"{config.POLYMARKET_CLOB_BASE}/markets/{condition_id}")
    if clob_data:
        tokens   = clob_data.get("tokens", [])
        yes_tok  = next((t for t in tokens if t.get("outcome","").upper()=="YES"), None)
        if yes_tok:
            live_price          = float(yes_tok.get("price", market["yes_price"]) or market["yes_price"])
            market["yes_price"] = round(max(0.01, min(0.99, live_price)), 4)
            market["no_price"]  = round(1 - market["yes_price"], 4)

    # 2. Order book depth for YES token
    yes_token_id = next(iter(market.get("token_ids", [])), None)
    if yes_token_id:
        book = _get(f"{config.POLYMARKET_CLOB_BASE}/book",
                    params={"token_id": yes_token_id})
        if book:
            bids     = book.get("bids", [])
            asks     = book.get("asks", [])
            best_bid = float(bids[0]["price"]) if bids else None
            best_ask = float(asks[0]["price"]) if asks else None
            spread   = round(best_ask - best_bid, 4) if (best_bid and best_ask) else None

            # Depth: sum top-5 bid + ask sizes (a real liquidity quality signal)
            depth_bids = sum(float(b.get("size", 0)) for b in bids[:5])
            depth_asks = sum(float(a.get("size", 0)) for a in asks[:5])
            book_depth = round(depth_bids + depth_asks, 2)

            market["best_bid"]   = best_bid
            market["best_ask"]   = best_ask
            market["spread"]     = spread
            market["book_depth"] = book_depth
            # Augment Gamma liquidity figure with order-book depth signal
            if book_depth > 0:
                market["liquidity"] = max(market["liquidity"], book_depth * 100)

    return market


def clob_fetch_last_price(token_id: str) -> Optional[float]:
    """Fast single-price poll — used for real-time momentum snapshots."""
    data = _get(f"{config.POLYMARKET_CLOB_BASE}/last-trade-price",
                params={"token_id": token_id})
    if data:
        return float(data.get("price", 0) or 0) or None
    return None


def clob_bulk_prices(token_ids: list[str]) -> dict[str, float]:
    """
    Batch price fetch for up to 50 tokens in one call.
    Returns {token_id: price}.
    """
    if not token_ids:
        return {}
    data = _get(f"{config.POLYMARKET_CLOB_BASE}/prices",
                params={"token_ids": ",".join(token_ids[:50])})
    if not data or not isinstance(data, dict):
        return {}
    return {k: float(v) for k, v in data.items() if v}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA-API  — historical time-series + volume candles
# ══════════════════════════════════════════════════════════════════════════════

def data_api_price_history(market_id: str,
                           interval: str = "1h",
                           fidelity: int = 60) -> list[dict]:
    """
    data-api /prices — OHLCV candle data.
    interval: '1m','5m','1h','6h','1d'
    fidelity: granularity in minutes

    Returns list of {t, open, high, low, close, volume} dicts.
    """
    data = _get(
        f"{config.POLYMARKET_DATA_BASE}/prices",
        params={"market": market_id, "interval": interval, "fidelity": fidelity},
    )
    if not data:
        return []
    return data if isinstance(data, list) else data.get("history", [])


def data_api_activity(market_id: str, limit: int = 100) -> list[dict]:
    """
    data-api /activity — individual trades for volume decomposition.
    Returns recent buys/sells with side, size, price.
    """
    data = _get(
        f"{config.POLYMARKET_DATA_BASE}/activity",
        params={"market": market_id, "limit": limit},
    )
    if not data:
        return []
    return data if isinstance(data, list) else data.get("data", [])


def data_api_resolution(condition_id: str) -> Optional[dict]:
    """
    data-api /resolution — most reliable source for resolved market outcomes.
    Returns {resolved: bool, outcome: 'YES'|'NO'|None, resolved_yes: bool|None}
    """
    data = _get(
        f"{config.POLYMARKET_DATA_BASE}/resolution",
        params={"conditionId": condition_id},
    )
    if not data:
        return None
    resolved = data.get("resolved", False)
    outcome  = data.get("outcome", None)
    return {
        "resolved":     resolved,
        "outcome":      outcome,
        "resolved_yes": (outcome == "YES") if resolved else None,
    }


def enrich_momentum(market: dict) -> dict:
    """
    Pull 4-day hourly price history from data-api and compute:
      • yes_price_24h_ago  (for momentum pillar)
      • yes_price_72h_ago  (for momentum pillar)
      • volume_24h from candles (more accurate than Gamma snapshot)

    Falls back to CLOB prices-history if data-api returns nothing.
    """
    condition_id = market.get("condition_id", "")
    if not condition_id:
        return market

    # Prefer data-api (cleaner OHLCV candles)
    history = data_api_price_history(condition_id, interval="1h", fidelity=60)

    if not history:
        # Fallback: CLOB prices-history
        end_ts   = int(time.time())
        start_ts = end_ts - 4 * 86400
        clob_hist = _get(
            f"{config.POLYMARKET_CLOB_BASE}/prices-history",
            params={"market": condition_id,
                    "startTs": start_ts, "endTs": end_ts, "fidelity": 60},
        )
        if clob_hist:
            history = clob_hist.get("history", [])

    if not history:
        return market

    history = sorted(history, key=lambda h: h.get("t", h.get("ts", 0)))

    now_ts = time.time()
    ts_24h = now_ts - 86_400
    ts_72h = now_ts - 3 * 86_400

    def _price_at(target_ts: float) -> Optional[float]:
        candidates = [h for h in history if h.get("t", h.get("ts", 0)) <= target_ts]
        if not candidates:
            return None
        closest = max(candidates, key=lambda h: h.get("t", h.get("ts", 0)))
        return float(closest.get("c", closest.get("p", 0)) or 0) or None

    p24 = _price_at(ts_24h)
    p72 = _price_at(ts_72h)
    if p24:
        market["yes_price_24h_ago"] = round(max(0.01, min(0.99, p24)), 4)
    if p72:
        market["yes_price_72h_ago"] = round(max(0.01, min(0.99, p72)), 4)

    # Volume from candles (sum of candle volumes in last 24h)
    vol_24h = sum(
        float(h.get("v", 0) or 0)
        for h in history if h.get("t", h.get("ts", 0)) >= ts_24h
    )
    if vol_24h > 0:
        market["volume_24h"] = max(market.get("volume_24h", 0), vol_24h)

    return market


# ══════════════════════════════════════════════════════════════════════════════
#  Unified Polymarket pipeline  (Gamma → CLOB → data-api)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_polymarket_markets(max_markets: int = 500,
                             enrich_top_n: int = 100) -> list[dict]:
    """
    Three-stage pipeline:
      Stage 1 — Gamma:    bulk discovery, tags, event grouping
      Stage 2 — CLOB:     live bid/ask + order-book depth enrichment
      Stage 3 — data-api: 24h/72h price history for momentum pillars

    enrich_top_n: only stages 2-3 for highest-volume markets
    (keeps GitHub Actions runtime inside ~10 min)
    """
    # ── Stage 1: Gamma discovery ───────────────────────────────────────────────
    raw_markets: list[dict] = []
    page_size = 100

    for page in range(max_markets // page_size + 1):
        batch = gamma_fetch_markets(limit=page_size, offset=page * page_size)
        if not batch:
            break
        for m in batch:
            parsed = _parse_gamma_market(m)
            if parsed:
                raw_markets.append(parsed)
        if len(batch) < page_size:
            break
        time.sleep(0.25)

    log.info(f"Gamma: discovered {len(raw_markets)} markets")

    # Tag macro events using Gamma /events
    try:
        events = gamma_fetch_events(limit=50)
        _tag_macro_events(raw_markets, events)
        log.info(f"Gamma: tagged {len(events)} events")
    except Exception as e:
        log.debug(f"Event tagging skipped: {e}")

    # Filter to active markets with meaningful time left
    markets = [
        m for m in raw_markets
        if m.get("active") and not m.get("closed") and m.get("days_to_res", 0) > 0.05
    ]

    # Sort by volume descending — enrich top-N with live data
    markets.sort(key=lambda m: m.get("volume_24h", 0), reverse=True)

    # ── Stage 2: CLOB live enrichment ─────────────────────────────────────────
    top_n = min(enrich_top_n, len(markets))
    log.info(f"CLOB: enriching top {top_n} markets with live order book…")
    for i in range(top_n):
        try:
            markets[i] = clob_enrich_market(markets[i])
        except Exception as e:
            log.debug(f"CLOB enrich {markets[i]['id']}: {e}")
        time.sleep(0.15)

    # ── Stage 3: data-api momentum history ────────────────────────────────────
    log.info(f"data-api: fetching price history for top {top_n} markets…")
    for i in range(top_n):
        try:
            markets[i] = enrich_momentum(markets[i])
        except Exception as e:
            log.debug(f"Momentum enrich {markets[i]['id']}: {e}")
        time.sleep(0.15)

    log.info(f"Polymarket pipeline complete: {len(markets)} markets ready")
    return markets


def _tag_macro_events(markets: list[dict], events: list[dict]):
    """Cross-reference markets with Gamma events for macro tagging & richer categories."""
    event_map: dict[str, dict] = {}
    for ev in events:
        for em in (ev.get("markets") or []):
            slug = em.get("slug", "")
            if slug:
                event_map[slug] = ev

    for m in markets:
        ev = event_map.get(m.get("slug", ""))
        if ev:
            m["event_title"] = ev.get("title", "")
            m["event_id"]    = ev.get("id", "")
            ev_tags          = [t.get("label","") for t in (ev.get("tags") or [])]
            m["tags"]        = list(set(m.get("tags", []) + ev_tags))
            # Re-classify with all available context
            m["category"]    = _classify_topic(
                m["question"] + " " + m.get("event_title","") + " " + " ".join(m["tags"])
            )


# ══════════════════════════════════════════════════════════════════════════════
#  Resolution check  (data-api preferred → CLOB fallback)
# ══════════════════════════════════════════════════════════════════════════════

def check_polymarket_resolution(market: dict) -> Optional[dict]:
    """
    data-api /resolution is more reliable than CLOB for final outcomes.
    Falls back to CLOB market state if data-api returns nothing.
    """
    condition_id = market.get("condition_id", "")

    # 1. data-api (most reliable)
    if condition_id:
        res = data_api_resolution(condition_id)
        if res and res.get("resolved"):
            return res

    # 2. CLOB fallback
    if condition_id:
        clob_data = _get(f"{config.POLYMARKET_CLOB_BASE}/markets/{condition_id}")
        if clob_data and clob_data.get("closed"):
            tokens    = clob_data.get("tokens", [])
            yes_token = next((t for t in tokens if t.get("outcome","").upper()=="YES"), None)
            if yes_token:
                winner = yes_token.get("winner", False)
                return {"resolved": True,
                        "resolved_yes": bool(winner),
                        "outcome": "YES" if winner else "NO"}
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Kalshi
# ══════════════════════════════════════════════════════════════════════════════

def fetch_kalshi_markets(limit: int = 200) -> list[dict]:
    markets = []
    cursor  = None
    for _ in range(5):
        params = {"limit": limit, "status": "open"}
        if cursor:
            params["cursor"] = cursor
        data = _get(f"{config.KALSHI_BASE}/markets", params=params)
        if not data:
            break
        for m in data.get("markets", []):
            try:
                parsed = _parse_kalshi(m)
                if parsed:
                    markets.append(parsed)
            except Exception as e:
                log.debug(f"Kalshi parse: {e}")
        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(0.3)
    log.info(f"Kalshi: {len(markets)} markets")
    return markets


def _parse_kalshi(m: dict) -> Optional[dict]:
    ticker  = m.get("ticker", "")
    title   = m.get("title", "")
    if not title:
        return None
    yes_price   = float(m.get("yes_bid", 50)) / 100
    volume      = float(m.get("volume", 0) or 0)
    liquidity   = float(m.get("open_interest", 0) or 0)
    end_date    = _parse_date(m.get("close_time", "") or m.get("expiration_time", ""))
    days_to_res = _days_until(end_date)
    return {
        "id":              _market_id("kalshi", ticker),
        "source":          "kalshi",
        "question":        title,
        "category":        _classify_topic(title),
        "yes_price":       round(max(0.01, min(0.99, yes_price)), 4),
        "no_price":        round(1 - yes_price, 4),
        "volume_24h":      volume,
        "volume_total":    volume,
        "liquidity":       liquidity,
        "end_date":        end_date.isoformat() if end_date else None,
        "days_to_res":     days_to_res,
        "active":          True,
        "closed":          False,
        "url":             f"https://kalshi.com/markets/{ticker}",
        "fetched_at":      datetime.now(timezone.utc).isoformat(),
        "yes_price_24h_ago": yes_price,
        "yes_price_72h_ago": yes_price,
        "condition_id":    "",
        "token_ids":       [],
        "tags":            [],
        "spread":          None,
        "book_depth":      0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Manifold
# ══════════════════════════════════════════════════════════════════════════════

def fetch_manifold_markets(limit: int = 500) -> list[dict]:
    markets = []
    before  = None
    for _ in range(4):
        params = {"limit": limit, "outcomeType": "BINARY",
                  "sort": "score", "filter": "open"}
        if before:
            params["before"] = before
        data = _get(f"{config.MANIFOLD_BASE}/markets", params=params)
        if not data or not isinstance(data, list):
            break
        for m in data:
            try:
                parsed = _parse_manifold(m)
                if parsed:
                    markets.append(parsed)
            except Exception as e:
                log.debug(f"Manifold parse: {e}")
        if len(data) < limit:
            break
        before = data[-1].get("id")
        time.sleep(0.3)
    log.info(f"Manifold: {len(markets)} markets")
    return markets


def _parse_manifold(m: dict) -> Optional[dict]:
    mid   = m.get("id", "")
    title = m.get("question", "")
    if not title:
        return None
    prob        = float(m.get("probability", 0.5))
    vol         = float(m.get("volume", 0) or 0)
    close_ts    = m.get("closeTime")
    end_date    = datetime.fromtimestamp(close_ts / 1000, tz=timezone.utc) if close_ts else None
    days_to_res = _days_until(end_date)
    return {
        "id":              _market_id("manifold", mid),
        "source":          "manifold",
        "question":        title,
        "category":        _classify_topic(title),
        "yes_price":       round(max(0.01, min(0.99, prob)), 4),
        "no_price":        round(1 - prob, 4),
        "volume_24h":      vol / max(days_to_res, 1),
        "volume_total":    vol,
        "liquidity":       vol * 0.1,
        "end_date":        end_date.isoformat() if end_date else None,
        "days_to_res":     days_to_res,
        "active":          not m.get("isResolved", False),
        "closed":          m.get("isResolved", False),
        "url":             m.get("url", f"https://manifold.markets/{mid}"),
        "fetched_at":      datetime.now(timezone.utc).isoformat(),
        "yes_price_24h_ago": prob,
        "yes_price_72h_ago": prob,
        "condition_id":    "",
        "token_ids":       [],
        "tags":            [],
        "spread":          None,
        "book_depth":      0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  News
# ══════════════════════════════════════════════════════════════════════════════

def fetch_news_articles(query: str = "", days: int = 3) -> list[dict]:
    articles = []

    if config.NEWS_API_KEY:
        from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        params = {
            "apiKey":   config.NEWS_API_KEY,
            "sources":  ",".join(config.NEWS_SOURCES),
            "from":     from_date,
            "pageSize": 100,
            "language": "en",
        }
        if query:
            params["q"] = query
        data = _get("https://newsapi.org/v2/everything", params=params)
        if data and data.get("status") == "ok":
            for a in data.get("articles", []):
                articles.append({
                    "title":        a.get("title", ""),
                    "description":  a.get("description", ""),
                    "content":      a.get("content", ""),
                    "source":       a.get("source", {}).get("name", ""),
                    "published_at": a.get("publishedAt", ""),
                    "url":          a.get("url", ""),
                })

    for feed_url in config.RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:30]:
                articles.append({
                    "title":        entry.get("title", ""),
                    "description":  entry.get("summary", ""),
                    "content":      entry.get("summary", ""),
                    "source":       feed.feed.get("title", feed_url),
                    "published_at": entry.get("published", ""),
                    "url":          entry.get("link", ""),
                })
        except Exception as e:
            log.debug(f"RSS {feed_url}: {e}")

    log.info(f"News: {len(articles)} articles (query='{query}')")
    return articles


# ══════════════════════════════════════════════════════════════════════════════
#  Unified fetch entry point
# ══════════════════════════════════════════════════════════════════════════════

def fetch_all_markets() -> list[dict]:
    all_markets = []

    try:
        all_markets += fetch_polymarket_markets()
    except Exception as e:
        log.error(f"Polymarket pipeline: {e}")

    try:
        all_markets += fetch_kalshi_markets()
    except Exception as e:
        log.error(f"Kalshi: {e}")

    try:
        all_markets += fetch_manifold_markets()
    except Exception as e:
        log.error(f"Manifold: {e}")

    filtered = [
        m for m in all_markets
        if m.get("active") and not m.get("closed") and m.get("days_to_res", 0) > 0
    ]
    log.info(f"Total universe: {len(filtered)} markets")
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
#  Utility
# ══════════════════════════════════════════════════════════════════════════════

def _parse_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s[:26], fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _days_until(dt: Optional[datetime]) -> float:
    if not dt:
        return 30
    now = datetime.now(timezone.utc)
    return max((dt - now).total_seconds() / 86400, 0)


def _classify_topic(text: str) -> str:
    text_l = text.lower()
    rules = {
        "politics":    ["election","president","senate","congress","vote","democrat",
                        "republican","white house","ballot","polls","campaign"],
        "economics":   ["fed","interest rate","inflation","gdp","recession","unemployment",
                        "cpi","rate cut","earnings","revenue","jobs report"],
        "sports":      ["nfl","nba","mlb","nhl","soccer","fifa","super bowl",
                        "world cup","champion","playoffs","tournament"],
        "science":     ["nasa","space","climate","drug","vaccine","ai ","artificial intelligence",
                        "cancer","fda","study","research"],
        "crypto":      ["bitcoin","ethereum","crypto","btc","eth","blockchain",
                        "defi","nft","altcoin","solana"],
        "geopolitics": ["ukraine","russia","china","taiwan","nato","war","military",
                        "sanction","israel","iran","north korea","conflict"],
        "weather":     ["hurricane","tornado","earthquake","storm","flood","wildfire"],
    }
    for cat, keywords in rules.items():
        if any(k in text_l for k in keywords):
            return cat
    return "other"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    markets = fetch_all_markets()
    print(f"\nTotal: {len(markets)}")
    for m in markets[:5]:
        spread   = f" spread={m['spread']:.3f}" if m.get("spread") else ""
        momentum = m["yes_price"] - m.get("yes_price_24h_ago", m["yes_price"])
        print(f"  [{m['source']}] {m['question'][:55]}"
              f"  YES={m['yes_price']:.1%} Δ24h={momentum:+.2%}{spread}")
