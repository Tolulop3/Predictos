"""
PredictOS — Test Suite
Tests all modules without live API calls.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import math
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import config
from src.sentiment import score_text, compute_sentiment, sentiment_to_probability
from src.scorer import (
    score_momentum, score_volume, score_sentiment,
    score_edge, score_decay, compute_composite,
    apply_filters, kelly_size, rank_markets, score_market
)
from src.ml_model import PredictOSModel, build_features
from src.tracker import compute_stats, generate_daily_tweet


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_market(**kwargs) -> dict:
    defaults = {
        "id": "test::abc123",
        "source": "polymarket",
        "question": "Will the Fed cut rates in December 2025?",
        "category": "economics",
        "yes_price": 0.55,
        "no_price": 0.45,
        "volume_24h": 50_000,
        "volume_total": 500_000,
        "liquidity": 200_000,
        "end_date": (datetime.now(timezone.utc) + timedelta(days=10)).isoformat(),
        "days_to_res": 10,
        "active": True,
        "closed": False,
        "url": "https://polymarket.com/event/test",
        "yes_price_24h_ago": 0.50,
        "yes_price_72h_ago": 0.48,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return {**defaults, **kwargs}


def make_articles(text: str, n: int = 3) -> list[dict]:
    return [
        {
            "title": text,
            "description": text,
            "content": text,
            "source": "reuters",
            "published_at": datetime.now(timezone.utc).isoformat(),
            "url": "https://reuters.com/test",
        }
        for _ in range(n)
    ]


# ─── Sentiment Tests ──────────────────────────────────────────────────────────

class TestSentiment(unittest.TestCase):

    def test_positive_text(self):
        score, fired = score_text("The Fed confirmed a rate cut, officially announcing the decision.")
        self.assertGreater(score, 0)
        self.assertIn("S01", fired)  # confirms

    def test_negative_text(self):
        score, fired = score_text("The proposal was rejected and collapsed amid opposition.")
        self.assertLess(score, 0)

    def test_neutral_text(self):
        score, fired = score_text("The weather is partly cloudy today.")
        # Neutral text should fire few signals
        self.assertEqual(len(fired), 0)

    def test_negation(self):
        # "not confirmed" fires the confirm signal weakly but negation also fires
        # The key property: negation dampens the score vs purely positive text
        pos_score, _ = score_text("confirmed officially approved")
        neg_score, fired = score_text("The deal was not confirmed and won't be approved.")
        # Negation text should score meaningfully lower than pure positive
        self.assertLess(neg_score, pos_score)
        # And S22 (negation signal) should fire
        self.assertIn("S22", fired)

    def test_compute_sentiment_relevant(self):
        articles = make_articles("Fed confirmed rate cut officially announced")
        result = compute_sentiment(articles, "Will the Fed cut rates?")
        self.assertIn("score", result)
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 1)

    def test_compute_sentiment_irrelevant(self):
        articles = make_articles("Football match results from last night")
        result = compute_sentiment(articles, "Will the Fed cut rates?")
        # Irrelevant articles should produce ~neutral
        self.assertAlmostEqual(result["score"], 0.5, delta=0.15)

    def test_sentiment_to_probability(self):
        # Strongly positive sentiment should push above prior
        prob = sentiment_to_probability(0.8, prior=0.5)
        self.assertGreater(prob, 0.5)

        # Strongly negative should pull below prior
        prob = sentiment_to_probability(0.2, prior=0.5)
        self.assertLess(prob, 0.5)

    def test_half_life_decay(self):
        from src.sentiment import _decay_weight
        # Fresh article = weight ~1
        now = datetime.now(timezone.utc).isoformat()
        w_fresh = _decay_weight(now, 5)
        self.assertGreater(w_fresh, 0.9)

        # 5-day-old article = weight ~0.5 (half-life)
        old = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        w_old = _decay_weight(old, 5)
        self.assertAlmostEqual(w_old, 0.5, delta=0.05)


# ─── Scorer Tests ─────────────────────────────────────────────────────────────

class TestScorer(unittest.TestCase):

    def test_momentum_bullish(self):
        m = make_market(yes_price=0.65, yes_price_24h_ago=0.55, yes_price_72h_ago=0.50)
        score = score_momentum(m)
        self.assertGreater(score, 50)  # bullish momentum → above neutral

    def test_momentum_bearish(self):
        m = make_market(yes_price=0.40, yes_price_24h_ago=0.55, yes_price_72h_ago=0.60)
        score = score_momentum(m)
        self.assertLess(score, 50)

    def test_momentum_flat(self):
        m = make_market(yes_price=0.50, yes_price_24h_ago=0.50, yes_price_72h_ago=0.50)
        score = score_momentum(m)
        self.assertAlmostEqual(score, 50, delta=5)

    def test_volume_high(self):
        m = make_market(volume_24h=1_000_000, liquidity=500_000)
        score = score_volume(m)
        self.assertGreater(score, 70)

    def test_volume_zero(self):
        m = make_market(volume_24h=0, liquidity=0)
        score = score_volume(m)
        self.assertEqual(score, 0)

    def test_edge_positive(self):
        m = make_market(yes_price=0.40)
        edge_score, edge_pct = score_edge(m, model_prob=0.60)
        self.assertGreater(edge_score, 50)  # model says 60%, market says 40%
        self.assertAlmostEqual(edge_pct, 0.20, delta=0.01)

    def test_edge_negative(self):
        m = make_market(yes_price=0.70)
        edge_score, edge_pct = score_edge(m, model_prob=0.45)
        self.assertLess(edge_score, 50)  # market overpriced
        self.assertAlmostEqual(edge_pct, -0.25, delta=0.01)

    def test_decay_close_with_momentum(self):
        m = make_market(days_to_res=5)
        score = score_decay(m, momentum_score=80)
        self.assertGreater(score, 40)

    def test_decay_binary_event_risk(self):
        # Markets < 2 days out get a fixed penalty score of 30 (not 0)
        # to reflect binary-event risk while keeping them in the universe
        m = make_market(days_to_res=0.5)
        score = score_decay(m, momentum_score=80)
        self.assertLessEqual(score, 30)   # penalised heavily vs optimal 8-day window

    def test_composite_weights(self):
        pillars = {"momentum": 80, "volume": 70, "sentiment": 60, "edge": 90, "decay": 50}
        score = compute_composite(pillars)
        import config as cfg
        expected = sum(cfg.PILLAR_WEIGHTS[k] * v for k, v in pillars.items())
        self.assertAlmostEqual(score, expected, delta=0.1)

    def test_filter_binary_event_risk(self):
        m = make_market(days_to_res=1)  # <48h
        passes, flags = apply_filters(m, {"article_count": 5}, 0.65, 0.7)
        self.assertFalse(passes)
        self.assertIn("binary_event_risk", flags)

    def test_filter_low_liquidity(self):
        m = make_market(source="polymarket", volume_24h=100)
        passes, flags = apply_filters(m, {"article_count": 5}, 0.70, 0.7)
        self.assertFalse(passes)
        self.assertIn("low_liquidity", flags)

    def test_filter_graham_edge(self):
        m = make_market(yes_price=0.50)
        # Edge < 8%: model says 0.55, market says 0.50 → only 5% edge
        passes, flags = apply_filters(m, {"article_count": 5}, 0.55, 0.7)
        self.assertFalse(passes)
        self.assertIn("insufficient_edge", flags)

    def test_filter_passes(self):
        m = make_market(yes_price=0.40, volume_24h=50_000, days_to_res=10)
        passes, flags = apply_filters(m, {"article_count": 10}, 0.65, 0.7)
        self.assertTrue(passes)

    def test_kelly_size(self):
        sizing = kelly_size(win_prob=0.65, odds=0.50, bankroll=1000)
        self.assertIn("kelly_f", sizing)
        self.assertIn("position_usd", sizing)
        self.assertLessEqual(sizing["position_usd"], 20)  # max 2% of $1000

    def test_kelly_no_edge(self):
        sizing = kelly_size(win_prob=0.50, odds=0.50, bankroll=1000)
        self.assertAlmostEqual(sizing["kelly_f"], 0.0, delta=0.01)  # no edge = no bet


# ─── ML Tests ─────────────────────────────────────────────────────────────────

class TestML(unittest.TestCase):

    def test_feature_shape(self):
        m = make_market()
        feats = build_features(m, 0.6)
        self.assertEqual(len(feats), len(config.FEATURE_COLUMNS))

    def test_fallback_predict(self):
        model = PredictOSModel()
        m = make_market(yes_price=0.60)
        prob = model.predict_single(m, sentiment_score=0.7)
        self.assertGreaterEqual(prob, 0)
        self.assertLessEqual(prob, 1)

    def test_predict_range(self):
        model = PredictOSModel()
        for yes_price in [0.1, 0.3, 0.5, 0.7, 0.9]:
            m = make_market(yes_price=yes_price)
            prob = model.predict_single(m, sentiment_score=0.6)
            self.assertTrue(0 <= prob <= 1, f"Prob {prob} out of range for yes_price={yes_price}")


# ─── Tracker Tests ────────────────────────────────────────────────────────────

class TestTracker(unittest.TestCase):

    def test_compute_stats_empty(self):
        stats = compute_stats([])
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["win_rate"], 0.0)

    def test_compute_stats(self):
        outcomes = [
            {"resolved": True, "won": True,  "category": "politics", "source": "polymarket"},
            {"resolved": True, "won": True,  "category": "politics", "source": "polymarket"},
            {"resolved": True, "won": False, "category": "economics","source": "kalshi"},
        ]
        stats = compute_stats(outcomes)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["wins"], 2)
        self.assertAlmostEqual(stats["win_rate"], 2/3, places=3)

    def test_tweet_generation(self):
        picks = {
            "yes_picks": [make_market(
                composite=82.5, edge_pct=15.3, confidence="HIGH",
                direction="YES", question="Will the Fed cut rates in September?"
            )],
            "no_picks": [],
            "macro_plays": [],
        }
        stats = {"win_rate": 0.65, "total": 20}
        tweet = generate_daily_tweet(picks, stats)
        self.assertIsInstance(tweet, str)
        self.assertLessEqual(len(tweet), 280)
        self.assertIn("PredictOS", tweet)

    def test_tweet_no_picks(self):
        picks = {"yes_picks": [], "no_picks": [], "macro_plays": []}
        stats = {"win_rate": 0, "total": 0}
        tweet = generate_daily_tweet(picks, stats)
        self.assertIn("no high-confidence", tweet.lower())


# ─── Integration Test (offline) ───────────────────────────────────────────────

class TestIntegration(unittest.TestCase):

    def test_full_score_pipeline(self):
        """Score a market end-to-end using offline data."""
        market = make_market(
            yes_price=0.42,
            yes_price_24h_ago=0.35,
            yes_price_72h_ago=0.32,
            volume_24h=75_000,
            liquidity=300_000,
            days_to_res=8,
        )
        articles = make_articles(
            "The Federal Reserve officially confirmed a rate cut decision announced today",
            n=5
        )
        from src.sentiment import compute_sentiment, sentiment_to_probability
        from src.scorer import score_market
        from src.ml_model import PredictOSModel

        sent = compute_sentiment(articles, market["question"])
        model = PredictOSModel()
        ml_prob = model.predict_single(market, sent["score"])
        news_prob = sentiment_to_probability(sent["score"], prior=market["yes_price"])
        model_prob = 0.6 * ml_prob + 0.4 * news_prob

        result = score_market(
            market=market,
            sentiment_result=sent,
            model_prob=model_prob,
            ml_confidence=abs(ml_prob - 0.5) * 2,
        )

        self.assertIn("composite", result)
        self.assertIn("pillars", result)
        self.assertIn("sizing", result)
        self.assertIn("direction", result)
        self.assertTrue(0 <= result["composite"] <= 100)
        self.assertIn(result["direction"], ["YES", "NO"])

    def test_rank_markets(self):
        """rank_markets correctly separates YES and NO picks."""
        scored = [
            {**make_market(id="test::1", yes_price=0.40), "composite": 80, "direction": "YES",
             "passes_filter": True},
            {**make_market(id="test::2", yes_price=0.40), "composite": 75, "direction": "YES",
             "passes_filter": True},
            {**make_market(id="test::3", yes_price=0.75), "composite": 70, "direction": "NO",
             "passes_filter": True},
            {**make_market(id="test::4", yes_price=0.50), "composite": 60, "direction": "YES",
             "passes_filter": False},  # filtered out
        ]
        yes_picks, no_picks = rank_markets(scored)
        self.assertEqual(len(yes_picks), 2)  # third filtered out
        self.assertEqual(len(no_picks), 1)
        self.assertEqual(yes_picks[0]["composite"], 80)  # sorted by composite


if __name__ == "__main__":
    unittest.main(verbosity=2)
