# PredictOS

**Automated Prediction Market Signal System**

What InvestOS does for stocks, PredictOS does for prediction markets.

---

## Architecture

```
PredictOS/
├── src/
│   ├── fetcher.py      — Polymarket + Kalshi + Manifold + News APIs
│   ├── sentiment.py    — 22-signal news sentiment engine
│   ├── scorer.py       — 5-pillar scoring (0–100), filters, Kelly sizing
│   ├── ml_model.py     — XGBoost + walk-forward backtest
│   ├── tracker.py      — Outcome tracking, win rates, tweet generator
│   ├── dashboard.py    — Static HTML dashboard baker
│   └── main.py         — Daily pipeline orchestrator
├── tests/
│   └── test_predictos.py
├── data/               — JSON data files (auto-generated)
│   ├── picks.json
│   ├── scores.json
│   ├── outcomes.json
│   └── history.json
├── models/             — XGBoost model files (auto-generated)
├── docs/
│   └── index.html      — GitHub Pages dashboard
├── .github/workflows/
│   └── daily_run.yml   — GitHub Actions daily runner
├── config.py
└── requirements.txt
```

---

## Quick Start

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR_USERNAME/predictos
cd predictos
pip install -r requirements.txt
```

### 2. Add GitHub Secrets

In your repo: **Settings → Secrets → Actions → New secret**

| Secret | Value | Notes |
|--------|-------|-------|
| `POLYMARKET_API_KEY` | your key | Free — register at polymarket.com |
| `NEWS_API_KEY` | your key | Free tier at newsapi.org |
| `BANKROLL` | e.g. `1000` | Your USD bankroll for Kelly sizing |

### 3. Enable GitHub Pages

**Settings → Pages → Source: Deploy from branch → main → /docs**

### 4. Run Manually

```bash
# Local test run (no API keys required for Manifold/Kalshi)
python -m src.main
```

Or trigger manually in **Actions → PredictOS Daily Run → Run workflow**.

---

## Scoring System

Each open market is scored **0–100** across 5 pillars:

| Pillar | Weight | What it measures |
|--------|--------|-----------------|
| **Momentum** | 25% | Odds direction over 24h/72h with trend half-life decay |
| **Volume** | 20% | Daily volume + liquidity depth (log-scaled) |
| **Sentiment** | 25% | 22-signal news engine, relevance-weighted, decayed |
| **Edge** | 20% | Market price vs model probability (margin of safety) |
| **Decay** | 10% | Proximity to resolution × trend confirmation |

### Signal Filters (same logic as InvestOS)

- **Binary event risk**: Skip markets resolving within 48 hours
- **Liquidity floor**: Min $10k daily volume (Polymarket/Kalshi only)
- **Graham equivalent**: Min 8% edge (market prob vs model prob)
- **Factor conflict**: High momentum + low volume = thin market, penalised
- **Half-life decay**: News signal: 5 days, trend signal: 10 days

---

## ML Layer

- **Model**: XGBoost binary classifier
- **Features**: odds_at_open, volume, sentiment, days_to_resolution, topic_category, momentum_24h, momentum_72h, liquidity_depth, odds_rank
- **Target**: Did YES resolve?
- **Validation**: Walk-forward backtest (5 folds), no lookahead bias
- **Fallback**: Logistic blend of market odds + sentiment when insufficient training data

---

## Position Sizing

- **Kelly Criterion** (25% fractional): `f* = (bp - q) / b`
- **Max 2% of bankroll** per market
- **Scale down** when ML confidence < 55%
- **Never enter** markets under $10k daily volume

---

## Output

Daily dashboard: `https://YOUR_USERNAME.github.io/predictos/`

- Top 5 YES picks: score, edge %, confidence, Kelly position size
- Top 3 NO picks: fading overpriced markets
- Macro event plays: Fed/election/geopolitical signals
- Win rate tracker: auto-resolved outcomes
- Daily tweet: copy-paste signal summary

---

## Data Sources

| Source | API | Key Required |
|--------|-----|-------------|
| Polymarket | `clob.polymarket.com` | Optional (higher rate limits) |
| Kalshi | `trading-api.kalshi.com` | No |
| Manifold Markets | `manifold.markets/api` | No |
| NewsAPI | `newsapi.org` | Free tier |
| RSS Feeds | Reuters, BBC, Politico, NBC | No |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Configuration

All settings in `config.py`:

```python
BANKROLL              = 1000      # Set via GitHub secret
MAX_POSITION_PCT      = 0.02      # 2% max per market
KELLY_FRACTION        = 0.25      # Conservative fractional Kelly
GRAHAM_EDGE_THRESHOLD = 0.08      # 8% min edge required
MIN_DAILY_VOLUME_USD  = 10_000    # Liquidity floor
ML_CONFIDENCE_THRESHOLD = 0.55   # Below = reduce position
NEWS_HALF_LIFE_DAYS   = 5
TREND_HALF_LIFE_DAYS  = 10
```

---

*Not financial advice. All prediction markets carry significant risk.*
