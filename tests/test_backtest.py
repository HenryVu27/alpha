"""Tests for the backtest evaluator (Task 15)."""

import numpy as np
import pytest

from backtest.evaluator import SignalEvaluator


@pytest.fixture
def evaluator():
    return SignalEvaluator()


class TestHitRate:
    def test_hit_rate_mixed_signals(self, evaluator):
        """3 signals: bearish@0 price down=hit, bullish@10 price up=hit, bearish@20 price up=miss → 2/3."""
        prices = np.zeros(30)
        # bearish@0 half_life=5: price goes down → hit
        prices[0:6] = [100, 99, 98, 97, 96, 95]
        # bullish@10 half_life=5: price goes up → hit
        prices[10:16] = [100, 101, 102, 103, 104, 105]
        # bearish@20 half_life=5: price goes up → miss
        prices[20:26] = [100, 101, 102, 103, 104, 105]
        # fill remaining with neutral values
        for i in range(30):
            if prices[i] == 0:
                prices[i] = 100.0

        signals = [
            {"timestamp_idx": 0, "half_life": 5, "direction": "bearish"},
            {"timestamp_idx": 10, "half_life": 5, "direction": "bullish"},
            {"timestamp_idx": 20, "half_life": 5, "direction": "bearish"},
        ]
        hit_rate = evaluator.compute_hit_rate(signals, prices)
        assert hit_rate == pytest.approx(2 / 3, abs=0.01)


class TestLeadTime:
    def test_lead_time_bearish(self, evaluator):
        """Signal at idx 5, price drops 20% starting at idx 8 → lead_time = 3."""
        prices = np.array([100.0] * 20)
        # Price drops 20% at idx 8
        prices[8:] = 80.0

        lead = evaluator.compute_lead_time(5, prices, "bearish", threshold_pct=0.05)
        assert lead == 3

    def test_lead_time_never_hit(self, evaluator):
        """Price never crosses threshold → returns -1."""
        prices = np.array([100.0] * 20)
        lead = evaluator.compute_lead_time(5, prices, "bearish", threshold_pct=0.05)
        assert lead == -1


class TestFalsePositiveRate:
    def test_all_false_positives(self, evaluator):
        """2 bearish signals but price goes up → fp_rate = 1.0."""
        prices = np.linspace(100, 110, 30)  # steadily rising
        signals = [
            {"timestamp_idx": 0, "half_life": 10, "direction": "bearish"},
            {"timestamp_idx": 5, "half_life": 10, "direction": "bearish"},
        ]
        fp_rate = evaluator.compute_false_positive_rate(signals, prices, threshold_pct=0.03)
        assert fp_rate == pytest.approx(1.0)

    def test_no_false_positives(self, evaluator):
        """Bearish signals with price dropping significantly → fp_rate = 0.0."""
        prices = np.linspace(100, 50, 30)  # steadily falling
        signals = [
            {"timestamp_idx": 0, "half_life": 10, "direction": "bearish"},
            {"timestamp_idx": 5, "half_life": 10, "direction": "bearish"},
        ]
        fp_rate = evaluator.compute_false_positive_rate(signals, prices, threshold_pct=0.03)
        assert fp_rate == pytest.approx(0.0)


class TestRegimeAccuracy:
    def test_regime_accuracy(self, evaluator):
        """Predicted vs actual → 4/5 = 0.8."""
        predicted = ["risk_on", "escalation", "stalemate", "risk_on", "escalation"]
        actual = ["risk_on", "escalation", "stalemate", "risk_on", "stalemate"]
        acc = evaluator.regime_accuracy(predicted, actual)
        assert acc == pytest.approx(0.8)

    def test_regime_accuracy_perfect(self, evaluator):
        predicted = ["risk_on", "risk_on"]
        actual = ["risk_on", "risk_on"]
        assert evaluator.regime_accuracy(predicted, actual) == pytest.approx(1.0)
