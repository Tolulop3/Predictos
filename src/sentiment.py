"""
PredictOS — Sentiment Engine
22-signal news analysis, identical architecture to InvestOS.
Maps news sentiment to YES/NO side of prediction markets.
"""

import re
import math
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

log = logging.getLogger("predictos.sentiment")

# ─── Signal Definitions (22 signals) ─────────────────────────────────────────

# Polarity: +1 = bullish for YES, -1 = bearish for YES
SIGNALS: list[dict] = [
    # ── Positive (YES-bullish) ────────────────────────────────────────────────
    {"id": "S01", "name": "Strong Confirmation",   "polarity": +1, "weight": 2.0,
     "patterns": [r"\bconfirm(ed|s)?\b", r"\bverif(ied|y)\b", r"\bproven?\b"]},

    {"id": "S02", "name": "Official Announcement", "polarity": +1, "weight": 1.8,
     "patterns": [r"\bannounce[sd]?\b", r"\bofficial(ly)?\b", r"\bdeclare[sd]?\b"]},

    {"id": "S03", "name": "Record/All-time",        "polarity": +1, "weight": 1.5,
     "patterns": [r"\brecord\b", r"\ball.time (high|best)\b", r"\bunprecedented\b"]},

    {"id": "S04", "name": "Surge/Rally Signal",     "polarity": +1, "weight": 1.4,
     "patterns": [r"\bsurge[sd]?\b", r"\bskyrock(et)?\b", r"\brallie[sd]?\b", r"\bsoar(ed|s)?\b"]},

    {"id": "S05", "name": "Approval/Pass",          "polarity": +1, "weight": 1.6,
     "patterns": [r"\bapprove[sd]?\b", r"\bpass(ed|es)?\b", r"\bsign(ed)? into law\b", r"\bapproval\b"]},

    {"id": "S06", "name": "Win/Victory",            "polarity": +1, "weight": 1.5,
     "patterns": [r"\bwins?\b", r"\bvictory\b", r"\bdefeats?\b", r"\bclinch(es|ed)?\b"]},

    {"id": "S07", "name": "Ahead/Leading",          "polarity": +1, "weight": 1.2,
     "patterns": [r"\bleads?\b", r"\bahead\b", r"\bfavou?rite\b", r"\bfrontrunner\b"]},

    {"id": "S08", "name": "Breakthrough",           "polarity": +1, "weight": 1.3,
     "patterns": [r"\bbreakthrough\b", r"\bpivot(s|ed)?\b", r"\bmilestone\b", r"\bmajor step\b"]},

    {"id": "S09", "name": "Consensus/Agreement",    "polarity": +1, "weight": 1.3,
     "patterns": [r"\bconsensus\b", r"\bagreem?ent\b", r"\bdeal\b", r"\bcompromise\b"]},

    {"id": "S10", "name": "Deadline Met",            "polarity": +1, "weight": 1.1,
     "patterns": [r"\bon track\b", r"\bmet (the )?deadline\b", r"\bschedule[d]?\b"]},

    {"id": "S11", "name": "Momentum",               "polarity": +1, "weight": 1.0,
     "patterns": [r"\bmomentum\b", r"\bgaining traction\b", r"\bprogress\b"]},

    # ── Negative (YES-bearish) ────────────────────────────────────────────────
    {"id": "S12", "name": "Denial/Rejection",       "polarity": -1, "weight": 2.0,
     "patterns": [r"\bdenies?\b", r"\breject(s|ed)?\b", r"\bveto(ed)?\b", r"\brefuse[sd]?\b"]},

    {"id": "S13", "name": "Collapse/Fail",          "polarity": -1, "weight": 1.9,
     "patterns": [r"\bcollapse[sd]?\b", r"\bfail(s|ed)?\b", r"\bfalls? through\b", r"\bcrash(es|ed)?\b"]},

    {"id": "S14", "name": "Delay/Postpone",         "polarity": -1, "weight": 1.4,
     "patterns": [r"\bdelay(s|ed)?\b", r"\bpostpon(e|ed)\b", r"\bpush(ed)? back\b"]},

    {"id": "S15", "name": "Reversal/Walkback",      "polarity": -1, "weight": 1.6,
     "patterns": [r"\breverse[sd]?\b", r"\bwalk(s|ed)? back\b", r"\bU.turn\b", r"\bscrap(s|ed|ped)?\b"]},

    {"id": "S16", "name": "Surprise Negative",      "polarity": -1, "weight": 1.5,
     "patterns": [r"\bshock(ing|ed)?\b", r"\bunexpected(ly)?\b", r"\bsurprising(ly)?\b"]},

    {"id": "S17", "name": "Crisis/Emergency",       "polarity": -1, "weight": 1.8,
     "patterns": [r"\bcrisis\b", r"\bemergency\b", r"\bcatastrophe\b", r"\bdisaster\b"]},

    {"id": "S18", "name": "Doubt/Uncertainty",      "polarity": -1, "weight": 1.1,
     "patterns": [r"\buncertain(ty)?\b", r"\bquestion(s|ed)?\b", r"\bdoubt(ful|s|ed)?\b"]},

    {"id": "S19", "name": "Opposition Strong",      "polarity": -1, "weight": 1.3,
     "patterns": [r"\boppos(es|ition|ed)\b", r"\bblocks?\b", r"\bobstruct(s|ed)?\b"]},

    {"id": "S20", "name": "Loses/Trailing",         "polarity": -1, "weight": 1.4,
     "patterns": [r"\bloses?\b", r"\btrailing\b", r"\bbehind (in )?(polls|market)\b"]},

    # ── Neutral modifiers ────────────────────────────────────────────────────
    {"id": "S21", "name": "Hedge/Qualifier",        "polarity":  0, "weight": 0.5,
     "patterns": [r"\bmight\b", r"\bcould\b", r"\bpossibly\b", r"\bpotentially\b"]},

    {"id": "S22", "name": "Negation",               "polarity": -1, "weight": 0.8,
     "patterns": [r"\bno(t| longer)\b", r"\bnever\b", r"\bwon.t\b", r"\bdoesn.t\b"]},
]

# Precompile patterns
for sig in SIGNALS:
    sig["_compiled"] = [re.compile(p, re.IGNORECASE) for p in sig["patterns"]]

# ─── Core ─────────────────────────────────────────────────────────────────────

def score_text(text: str) -> tuple[float, list[str]]:
    """
    Score raw text. Returns (raw_score, fired_signal_ids).
    raw_score ∈ (-∞, +∞) — positive = YES-bullish
    """
    if not text:
        return 0.0, []

    fired = []
    raw = 0.0

    for sig in SIGNALS:
        for pat in sig["_compiled"]:
            if pat.search(text):
                raw += sig["polarity"] * sig["weight"]
                fired.append(sig["id"])
                break  # one fire per signal per text

    return raw, fired


def _decay_weight(published_at: str, half_life_days: float) -> float:
    """Exponential decay weight based on article age."""
    try:
        pub = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except Exception:
        return 0.5

    age_days = (datetime.now(timezone.utc) - pub).total_seconds() / 86400
    return math.exp(-math.log(2) * age_days / half_life_days)


def compute_sentiment(articles: list[dict], question: str,
                      half_life_days: float = 5.0,
                      category: str = "other") -> dict:
    """
    Compute a sentiment score ∈ [0, 1] for a prediction market question.
    >0.5 = news favours YES; <0.5 = news favours NO

    When no articles match the question, falls back to category-level
    sentiment rather than neutral 0.5 — so all markets get a signal.
    """
    import config as cfg

    # Category baseline sentiments — derived from historical news tone
    # Economics/crypto news is structurally positive (growth, rally, beat)
    # Geopolitics/crisis news is structurally negative (conflict, risk)
    CATEGORY_BASELINES = {
        "economics":   0.54,   # slightly bullish — growth/beat coverage
        "crypto":      0.57,   # bullish — crypto news skews optimistic
        "politics":    0.50,   # neutral — balanced coverage
        "sports":      0.52,   # slight YES lean — wins/records coverage
        "geopolitics": 0.46,   # slightly bearish — conflict/risk coverage
        "science":     0.53,   # slight YES lean — discovery/progress coverage
        "weather":     0.48,   # slight NO lean — warnings/risk coverage
        "other":       0.50,
    }

    question_tokens = set(re.findall(r"\w+", question.lower()))

    weighted_score = 0.0
    total_weight   = 0.0
    fired_signals  = {}
    article_count  = 0

    for art in articles:
        # Relevance: how many question tokens appear in the article
        combined = " ".join([
            art.get("title", ""),
            art.get("description", ""),
            art.get("content", "") or "",
        ])
        art_tokens = set(re.findall(r"\w+", combined.lower()))
        overlap = len(question_tokens & art_tokens) / max(len(question_tokens), 1)

        # Filter: must share at least ~2 meaningful words (>1 overlap weighted)
        if overlap < 0.12:
            continue

        raw, fired = score_text(combined)
        age_weight = _decay_weight(art.get("published_at", ""), half_life_days)
        relevance_weight = overlap ** 0.5   # sqrt damping

        effective_weight = age_weight * relevance_weight
        weighted_score += raw * effective_weight
        total_weight   += effective_weight
        article_count  += 1

        for s in fired:
            fired_signals[s] = fired_signals.get(s, 0) + 1

    if total_weight == 0:
        # Fix 1: use category baseline instead of neutral 0.5
        baseline = CATEGORY_BASELINES.get(category, 0.50)
        return {"score": baseline, "raw": 0.0, "article_count": 0,
                "signals": {}, "category_fallback": True}

    raw_norm = weighted_score / total_weight

    # Sigmoid transform to [0, 1]
    score = 1 / (1 + math.exp(-raw_norm * 0.6))

    return {
        "score":            round(score, 4),
        "raw":              round(raw_norm, 4),
        "article_count":    article_count,
        "signals":          fired_signals,
        "category_fallback": False,
    }


def sentiment_to_probability(sentiment_score: float,
                             prior: float = 0.5,
                             weight: float = 0.4) -> float:
    """
    Blend sentiment signal with market prior to get model probability estimate.
    weight: how much to move from prior towards sentiment (0 = ignore news, 1 = pure news)
    """
    return round(prior + weight * (sentiment_score - 0.5), 4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fake_articles = [
        {"title": "Fed confirms rate cut in September meeting",
         "description": "Officials announced a 25bp reduction amid cooling inflation.",
         "content": "", "published_at": datetime.now(timezone.utc).isoformat()},
        {"title": "Markets rally on Fed decision, stocks surge",
         "description": "Equities soared after the official Fed announcement.",
         "content": "", "published_at": datetime.now(timezone.utc).isoformat()},
    ]
    question = "Will the Fed cut rates in September 2025?"
    result = compute_sentiment(fake_articles, question)
    print(f"Sentiment: {result}")
    prob = sentiment_to_probability(result["score"], prior=0.55)
    print(f"Model prob: {prob:.1%}")
