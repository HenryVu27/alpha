"""Tests for Master Decision Engine (TDD)."""

import pytest
from unittest.mock import patch
from engine.decision import DecisionEngine


class TestRegimeToSignal:
    """Test regime_to_signal for crisis/bullish/stalemate regimes."""

    def test_escalation_equity_bearish(self):
        engine = DecisionEngine()
        assert engine.regime_to_signal("escalation", "equity") == -1

    def test_crisis_equity_bearish(self):
        engine = DecisionEngine()
        assert engine.regime_to_signal("crisis", "equity") == -1

    def test_risk_on_equity_bullish(self):
        engine = DecisionEngine()
        assert engine.regime_to_signal("risk_on", "equity") == 1

    def test_escalation_gold_bullish(self):
        engine = DecisionEngine()
        assert engine.regime_to_signal("escalation", "gold") == 1

    def test_crisis_treasury_bullish(self):
        engine = DecisionEngine()
        assert engine.regime_to_signal("crisis", "treasury") == 1

    def test_risk_on_safe_haven_bearish(self):
        engine = DecisionEngine()
        assert engine.regime_to_signal("risk_on", "volatility") == -1

    def test_stalemate_returns_zero(self):
        engine = DecisionEngine()
        assert engine.regime_to_signal("stalemate", "equity") == 0


class TestComputeConvergence:
    """Test convergence calculation with mostly bearish signals."""

    def test_mostly_bearish_signals(self):
        engine = DecisionEngine()
        signals = {
            "regime": -1,
            "sentiment": -1,
            "options": -1,
            "correlation": 0,
            "mean_reversion": -1,
            "sizing": 1,
        }
        score, agreeing = engine.compute_convergence(signals)
        assert score < 0
        assert agreeing >= 3

    def test_all_bullish(self):
        engine = DecisionEngine()
        signals = {"a": 1, "b": 1, "c": 1}
        score, agreeing = engine.compute_convergence(signals)
        assert score > 0
        assert agreeing == 3

    def test_all_zero(self):
        engine = DecisionEngine()
        signals = {"a": 0, "b": 0, "c": 0}
        score, agreeing = engine.compute_convergence(signals)
        assert score == 0.0


class TestClassifyConviction:
    """Test conviction classification."""

    def test_all_bearish_high_conviction(self):
        engine = DecisionEngine(convergence_min_layers=3)
        # All bearish signals -> score=-1.0, agreeing=5
        result = engine.classify_conviction(-1.0, 5)
        assert result["conviction"] == "high"
        assert result["direction"] == "bearish"

    def test_mixed_signals_mixed_conviction(self):
        engine = DecisionEngine()
        # Low score with few agreeing -> mixed
        result = engine.classify_conviction(0.05, 1)
        assert result["conviction"] == "mixed"

    def test_moderate_conviction(self):
        engine = DecisionEngine()
        result = engine.classify_conviction(0.35, 2)
        assert result["conviction"] == "moderate"
        assert result["direction"] == "bullish"

    def test_neutral_direction(self):
        engine = DecisionEngine()
        result = engine.classify_conviction(0.03, 3)
        assert result["direction"] == "neutral"


class TestMixedSignals:
    """Test that evenly mixed signals produce mixed conviction."""

    def test_balanced_signals_mixed(self):
        engine = DecisionEngine()
        signals = {
            "regime": 1,
            "sentiment": -1,
            "options": 1,
            "correlation": -1,
        }
        score, agreeing = engine.compute_convergence(signals)
        result = engine.classify_conviction(score, agreeing)
        assert result["conviction"] == "mixed"


class TestAggregate:
    """Test full aggregate pipeline with layer outputs."""

    def test_aggregate_structure(self):
        engine = DecisionEngine()
        layer_outputs = {
            "regime": {"regime_label": "escalation"},
            "sentiment": {"mean": -0.5, "velocity": -0.2},
            "options": {"signals": ["vix_elevated", "skew_elevated"]},
            "correlation": {"corr_regime": "crisis_clustering"},
            "mean_reversion": {
                "SPY": {"z_score": -2.5, "signal_suppressed": False},
                "GLD": {"z_score": 1.0, "signal_suppressed": False},
            },
            "sizing": {
                "SPY": {"edge_positive": False},
                "GLD": {"edge_positive": True},
            },
        }
        tickers = ["SPY", "GLD"]
        asset_classes = {"SPY": "equity", "GLD": "gold"}

        result = engine.aggregate(layer_outputs, tickers, asset_classes)

        # Check structure
        assert "timestamp" in result
        assert "tickers" in result
        assert "SPY" in result["tickers"]
        assert "GLD" in result["tickers"]

        # SPY: escalation + equity -> regime=-1, sentiment=-1, options=-1,
        # correlation=-1, mean_reversion=+1 (z<-2), sizing=-1
        spy = result["tickers"]["SPY"]
        assert "signals" in spy
        assert "convergence" in spy
        assert "conviction" in spy
        assert spy["signals"]["regime"] == -1
        assert spy["signals"]["sentiment"] == -1
        assert spy["signals"]["mean_reversion"] == 1  # z_score < -2.0 -> undervalued
        assert spy["signals"]["sizing"] == -1

        # GLD: escalation + gold -> regime=+1
        gld = result["tickers"]["GLD"]
        assert gld["signals"]["regime"] == 1
        assert gld["signals"]["sizing"] == 1
