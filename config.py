"""
PredictOS — Configuration
Central config for all modules. Edit BANKROLL and API keys here.
"""

import os

# ─── Identity ────────────────────────────────────────────────────────────────
APP_NAME = "PredictOS"
VERSION  = "1.0.0"

# ─── Capital ─────────────────────────────────────────────────────────────────
BANKROLL = float(os.getenv("BANKROLL", "1000"))   # USD — set via GitHub secret
MAX_POSITION_PCT = 0.02                            # 2% max per market
KELLY_FRACTION   = 0.25                            # fractional Kelly (conservative)

# ─── API Keys (set as GitHub Actions secrets) ────────────────────────────────
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY", "")   # free — register at polymarket.com
NEWS_API_KEY       = os.getenv("NEWS_API_KEY", "")          # free tier at newsapi.org

# ─── Data Sources ─────────────────────────────────────────────────────────────
# ── Polymarket — three distinct APIs, each with a different job ──────────────
# gamma-api  : market discovery, metadata, categories, events (REST, no auth)
# data-api   : historical time-series, price history, volume candles (REST, no auth)
# clob       : live order-book, real-time prices, spreads, depth (REST + WS, optional auth)
POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"
POLYMARKET_DATA_BASE  = "https://data-api.polymarket.com"
POLYMARKET_CLOB_BASE  = "https://clob.polymarket.com"

KALSHI_BASE   = "https://trading-api.kalshi.com/trade-api/v2"
MANIFOLD_BASE = "https://api.manifold.markets/v0"

NEWS_SOURCES = [
    "reuters", "associated-press", "bloomberg", "the-wall-street-journal",
    "financial-times", "bbc-news", "the-guardian-uk", "politico",
    "axios", "the-hill", "nbc-news", "abc-news",
    "cbs-news", "usa-today", "time", "newsweek",
]

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/topNews",
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.politico.com/politics-news.xml",
    "https://feeds.nbcnews.com/nbcnews/public/news",
]

# ─── Scoring Weights ──────────────────────────────────────────────────────────
PILLAR_WEIGHTS = {
    "momentum":  0.25,
    "volume":    0.20,
    "sentiment": 0.25,
    "edge":      0.20,
    "decay":     0.10,
}

# ─── Signal Filters ───────────────────────────────────────────────────────────
MIN_RESOLUTION_HOURS    = 48      # skip markets resolving in <48h (binary event risk)
MAX_RESOLUTION_DAYS     = 120     # skip markets resolving >120 days out (too speculative)
MIN_DAILY_VOLUME_USD    = 10_000  # liquidity floor
GRAHAM_EDGE_THRESHOLD   = 0.08   # 8% min diff: market prob vs model prob
MIN_MARKET_PROB         = 0.08   # ignore markets priced below 8% (extreme longshots)
MAX_MARKET_PROB         = 0.92   # ignore markets priced above 92% (near certain)
NEWS_HALF_LIFE_DAYS     = 5
TREND_HALF_LIFE_DAYS    = 10
ML_CONFIDENCE_THRESHOLD = 0.55   # below this → scale-down position

# ─── ML ───────────────────────────────────────────────────────────────────────
ML_MODEL_PATH      = "models/xgb_predictos.json"
FEATURE_COLUMNS    = [
    "odds_at_open", "volume_24h", "sentiment_score",
    "days_to_resolution", "topic_category_enc",
    "momentum_24h", "momentum_72h", "liquidity_depth",
    "odds_rank_universe",                             # RS equivalent
]
WALKFORWARD_FOLDS  = 5
TRAIN_MIN_SAMPLES  = 200

# ─── Output ───────────────────────────────────────────────────────────────────
DATA_DIR       = "data"
PICKS_FILE     = "data/picks.json"
HISTORY_FILE   = "data/history.json"
SCORES_FILE    = "data/scores.json"
OUTCOMES_FILE  = "data/outcomes.json"
DASHBOARD_FILE = "docs/index.html"

# ─── Categories (regime equivalent) ──────────────────────────────────────────
TOPIC_CATEGORIES = {
    "politics":    0,
    "economics":   1,
    "sports":      2,
    "science":     3,
    "crypto":      4,
    "geopolitics": 5,
    "weather":     6,
    "other":       7,
}

MACRO_KEYWORDS = [
    "fed", "federal reserve", "interest rate", "cpi", "inflation",
    "election", "president", "senate", "gdp", "recession",
    "ukraine", "israel", "china", "taiwan", "nato",
    "bitcoin", "crypto", "sec",
]

SPORTS_KEYWORDS = [
    # Leagues / competitions
    "nfl", "nba", "mlb", "nhl", "premier league", "la liga", "serie a",
    "bundesliga", "ligue 1", "champions league", "europa league",
    "super bowl", "world cup", "stanley cup", "nba finals", "world series",
    "march madness", "ncaa", "ufc", "mma", "formula 1", "f1",
    # Actions
    "win the", "championship", "playoffs", "finals", "title", "trophy",
    "golden boot", "mvp", "ballon d'or", "relegated", "promoted",
    # Sports RSS feeds (added to RSS_FEEDS in fetcher)
]

SPORTS_RSS_FEEDS = [
    "https://www.espn.com/espn/rss/news",
    "https://feeds.bbci.co.uk/sport/rss.xml",
    "https://www.skysports.com/rss/12040",
]
