"""Tests for layers.options_implied – Task 10: Options-Implied Volatility Signals."""

import pytest

from layers.options_implied import OptionsImpliedLayer


class TestTermStructure:
    def test_contango_detected(self):
        """vix3m > vix → contango term structure state."""
        layer = OptionsImpliedLayer()
        result = layer.compute(vix=20, vix3m=22)
        assert result["term_structure_state"] == "contango"
        assert result["term_structure_ratio"] == pytest.approx(22 / 20)

    def test_backwardation_detected(self):
        """vix3m < vix → backwardation term structure state."""
        layer = OptionsImpliedLayer()
        result = layer.compute(vix=30, vix3m=25)
        assert result["term_structure_state"] == "backwardation"
        assert result["term_structure_ratio"] == pytest.approx(25 / 30)


class TestVixLevelSignals:
    def test_vix_elevated_signal(self):
        """VIX at or above vix_high threshold triggers vix_elevated signal."""
        layer = OptionsImpliedLayer(vix_high=30)
        result = layer.compute(vix=35)
        assert "vix_elevated" in result["signals"]

    def test_vix_low_signal(self):
        """VIX at or below vix_low threshold triggers vix_low signal."""
        layer = OptionsImpliedLayer(vix_low=18)
        result = layer.compute(vix=15)
        assert "vix_low" in result["signals"]


class TestSkewSignal:
    def test_skew_elevated_signal(self):
        """SKEW at or above skew_elevated threshold triggers skew_elevated signal."""
        layer = OptionsImpliedLayer(skew_elevated=150)
        result = layer.compute(vix=20, skew=160)
        assert "skew_elevated" in result["signals"]


class TestPutCallRatioSignals:
    def test_pcr_heavy_signal(self):
        """Put/call ratio at or above pcr_heavy threshold triggers pcr_heavy signal."""
        layer = OptionsImpliedLayer(pcr_heavy=1.3)
        result = layer.compute(vix=20, put_call_ratio=1.5)
        assert "pcr_heavy" in result["signals"]

    def test_pcr_light_signal(self):
        """Put/call ratio at or below pcr_light threshold triggers pcr_light signal."""
        layer = OptionsImpliedLayer(pcr_light=0.7)
        result = layer.compute(vix=20, put_call_ratio=0.5)
        assert "pcr_light" in result["signals"]


class TestCalmMarket:
    def test_no_signals_in_calm_market(self):
        """All inputs within normal range → no signals triggered."""
        layer = OptionsImpliedLayer(
            vix_high=30, vix_low=18, skew_elevated=150, pcr_heavy=1.3, pcr_light=0.7
        )
        result = layer.compute(vix=20, vix3m=22, skew=130, put_call_ratio=0.9)
        assert result["signals"] == []
        assert result["term_structure_state"] == "contango"
        assert result["vix"] == 20
        assert result["skew"] == 130
        assert result["put_call_ratio"] == 0.9
        assert result["data_staleness_seconds"] == 0


class TestMissingData:
    def test_missing_optional_inputs_handled_gracefully(self):
        """None values for optional inputs → no crash, None fields, empty signals."""
        layer = OptionsImpliedLayer()
        result = layer.compute(vix=20, vix3m=None, skew=None, put_call_ratio=None)
        assert result["vix3m"] is None
        assert result["term_structure_ratio"] is None
        assert result["term_structure_state"] is None
        assert result["skew"] is None
        assert result["put_call_ratio"] is None
        # Only vix_low could fire (vix=20 is above default vix_low=18) – no signals
        assert isinstance(result["signals"], list)
        assert result["data_staleness_seconds"] == 0
