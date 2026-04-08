"""Tests for layers.sentiment – Task 8: Sentiment Engine."""

import numpy as np
import pytest

from layers.sentiment import (
    KeywordMatcher,
    HawkesEstimator,
    SentimentScorer,
    SentimentLayer,
    compute_sentiment_velocity,
)


# ── KeywordMatcher ──────────────────────────────────────────────────────────

class TestKeywordMatcher:
    def test_matches_text_with_keywords(self):
        matcher = KeywordMatcher(["missile", "sanctions", "war"])
        assert matcher.matches("North Korea launches missile") is True
        assert matcher.matches("Today the weather is nice") is False

    def test_case_insensitive(self):
        matcher = KeywordMatcher(["WAR", "Sanctions"])
        assert matcher.matches("new war breaks out") is True
        assert matcher.matches("SANCTIONS imposed today") is True
        assert matcher.matches("peace talks continue") is False


# ── HawkesEstimator ─────────────────────────────────────────────────────────

class TestHawkesEstimator:
    def test_regular_times(self):
        """Regularly spaced events should yield a branching ratio between 0 and 1."""
        estimator = HawkesEstimator()
        times = np.linspace(0, 100, 50)  # evenly spaced
        result = estimator.estimate(times, window=100.0)
        assert 0 <= result["branching_ratio"] < 2
        assert "intensity" in result
        assert "baseline" in result
        assert "decay" in result

    def test_clustered_times_higher_branching(self):
        """Clustered events should produce a higher branching ratio than regular ones."""
        estimator = HawkesEstimator()
        regular = np.linspace(0, 100, 50)
        # Clustered: bursts of events
        clustered = np.sort(np.concatenate([
            np.random.uniform(0, 2, 15),
            np.random.uniform(30, 32, 15),
            np.random.uniform(60, 62, 15),
        ]))
        r_regular = estimator.estimate(regular, window=100.0)
        r_clustered = estimator.estimate(clustered, window=100.0)
        assert r_clustered["branching_ratio"] > r_regular["branching_ratio"]


# ── SentimentScorer ─────────────────────────────────────────────────────────

class TestSentimentScorer:
    def test_no_finbert_returns_neutral(self):
        scorer = SentimentScorer(device="cpu", use_finbert=False)
        scores = scorer.score_headlines(["Oil prices surge", "Markets crash"])
        assert scores == [0.0, 0.0]


# ── compute_sentiment_velocity ──────────────────────────────────────────────

class TestSentimentVelocity:
    def test_declining_history_negative_velocity(self):
        history = [0.5, 0.3, 0.1, -0.1, -0.3]
        velocity = compute_sentiment_velocity(history, window=3)
        assert velocity < 0.0


# ── SentimentLayer ──────────────────────────────────────────────────────────

class TestSentimentLayer:
    def test_build_output_structure(self):
        layer = SentimentLayer(
            keywords=["war", "missile"],
            device="cpu",
            llm_provider=None,
            llm_model=None,
        )
        output = layer.build_output(
            sentiment_scores=[0.5, -0.2, 0.3],
            sentiment_history=[0.1, 0.2, 0.3, 0.2],
            hawkes_result={
                "branching_ratio": 0.4,
                "intensity": 2.0,
                "baseline": 1.2,
                "decay": 0.5,
            },
            headline_count_15m=5,
            headline_count_1h=20,
            top_headlines=["War breaks out", "Missile launched"],
            llm_result=None,
            active_sources=["reuters", "bbc"],
        )
        assert "sentiment_mean" in output
        assert "velocity" in output
        assert "acceleration" in output
        assert "hawkes" in output
        assert "headline_count_15m" in output
        assert "headline_count_1h" in output
        assert "top_headlines" in output
        assert "active_sources" in output
        assert abs(output["sentiment_mean"] - 0.2) < 1e-6
        assert output["hawkes"]["branching_ratio"] == 0.4
