"""Tests for Position Sizing layer (TDD)."""

import math
import pytest

from layers.sizing import SizingLayer


class TestInverseVolSize:
    """Tests for inverse_vol_size method."""

    def test_basic_inverse_vol(self):
        """target=0.15, realized=0.30 -> raw=0.5, not capped (max=0.60)."""
        layer = SizingLayer(target_annual_vol=0.15, max_single_pct=0.60)
        result = layer.inverse_vol_size(realized_vol=0.30)
        assert math.isclose(result, 0.5, rel_tol=1e-9)

    def test_capped_at_max_single_pct(self):
        """Very low vol produces ratio > max -> capped at max_single_pct=0.40."""
        layer = SizingLayer(target_annual_vol=0.15, max_single_pct=0.40)
        # realized_vol=0.05 -> raw = 0.15/0.05 = 3.0 > 0.40, capped
        result = layer.inverse_vol_size(realized_vol=0.05)
        assert math.isclose(result, 0.40, rel_tol=1e-9)

    def test_zero_vol_returns_max_single_pct(self):
        """vol <= 0 returns max_single_pct."""
        layer = SizingLayer(max_single_pct=0.40)
        assert math.isclose(layer.inverse_vol_size(0.0), 0.40, rel_tol=1e-9)
        assert math.isclose(layer.inverse_vol_size(-0.1), 0.40, rel_tol=1e-9)


class TestKellyCriterion:
    """Tests for kelly_criterion method."""

    def test_positive_edge(self):
        """win_prob=0.6, ratio=1.5 -> quarter of (0.6 - 0.4/1.5)."""
        layer = SizingLayer(kelly_fraction=0.25)
        full_kelly = 0.6 - (1 - 0.6) / 1.5
        expected = full_kelly * 0.25
        result = layer.kelly_criterion(win_prob=0.6, win_loss_ratio=1.5)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_negative_edge_returns_zero(self):
        """win_prob=0.3, ratio=0.8 -> full kelly negative -> return 0.0."""
        layer = SizingLayer(kelly_fraction=0.25)
        result = layer.kelly_criterion(win_prob=0.3, win_loss_ratio=0.8)
        assert result == 0.0

    def test_zero_win_loss_ratio_returns_zero(self):
        """win_loss_ratio <= 0 -> return 0.0."""
        layer = SizingLayer()
        assert layer.kelly_criterion(win_prob=0.6, win_loss_ratio=0.0) == 0.0
        assert layer.kelly_criterion(win_prob=0.6, win_loss_ratio=-1.0) == 0.0


class TestComputeMethod:
    """Tests for SizingLayer.compute method."""

    def test_compute_returns_all_fields(self):
        """Full compute call returns dict with all required keys."""
        layer = SizingLayer()
        result = layer.compute(
            ticker="AAPL",
            realized_vol=0.25,
            convergence_score=0.6,
            current_position_pct=0.10,
            current_price=150.0,
            entry_price=145.0,
            stop_loss=140.0,
        )
        required_keys = {
            "ticker",
            "vol_sized_pct",
            "kelly_cap_pct",
            "final_allocation_pct",
            "sizing_basis",
            "edge_positive",
            "current_position_pct",
            "action",
            "stop_loss_distance_pct",
            "data_staleness_seconds",
        }
        assert required_keys.issubset(result.keys())
        assert result["ticker"] == "AAPL"
        assert result["current_position_pct"] == 0.10

    def test_stop_loss_distance_correct(self):
        """stop_loss_distance_pct = (current_price - stop_loss) / current_price."""
        layer = SizingLayer()
        current_price = 100.0
        stop_loss = 97.0
        expected_distance = (current_price - stop_loss) / current_price  # 0.03

        result = layer.compute(
            ticker="TEST",
            realized_vol=0.20,
            convergence_score=0.5,
            current_position_pct=0.05,
            current_price=current_price,
            stop_loss=stop_loss,
        )
        assert math.isclose(result["stop_loss_distance_pct"], expected_distance, rel_tol=1e-9)

    def test_stop_loss_none_when_not_provided(self):
        """stop_loss_distance_pct is None when stop_loss not provided."""
        layer = SizingLayer()
        result = layer.compute(
            ticker="TEST",
            realized_vol=0.20,
            convergence_score=0.5,
            current_position_pct=0.05,
            current_price=100.0,
        )
        assert result["stop_loss_distance_pct"] is None

    def test_action_oversized(self):
        """current_position_pct > final * 1.1 -> action = 'oversized'."""
        layer = SizingLayer(target_annual_vol=0.15, max_single_pct=0.40, kelly_fraction=0.25)
        # With realized_vol=0.30 -> vol_sized=0.5 -> capped at 0.40
        # convergence_score=0.6 -> win_prob=0.65 -> full_kelly=0.65-0.35/1.5=0.417 -> kelly*0.25=0.104
        # kelly_cap < vol_sized -> final = kelly_cap ~ 0.104
        # current=0.30 > 0.104*1.1=0.114 -> oversized
        result = layer.compute(
            ticker="TEST",
            realized_vol=0.30,
            convergence_score=0.6,
            current_position_pct=0.30,
            current_price=100.0,
        )
        assert result["action"] == "oversized"

    def test_action_no_edge(self):
        """Negative convergence_score with kelly=0 -> action = 'no_edge'."""
        layer = SizingLayer()
        # convergence_score=-0.8 -> win_prob=0.5+0.8*0.25=0.7 ... wait, |score|
        # convergence_score=-0.8 -> win_prob=0.5+|-0.8|*0.25=0.7
        # That's positive edge. Use very low convergence_score and check kelly=0
        # kelly_criterion returns 0 when full_kelly <= 0
        # win_prob = 0.5 + |score|*0.25, with ratio=1.5
        # full_kelly=0 when win_prob = (1-win_prob)/1.5 => 1.5*p = 1-p => p=0.4
        # Need win_prob < 0.4 -> |score| < (0.4-0.5)/0.25 -> negative, impossible
        # So edge_positive is always true if convergence_score != 0 with ratio=1.5
        # edge_positive = convergence_score > 0 AND kelly_cap > 0
        # negative convergence_score -> edge_positive = False -> action = 'no_edge'
        result = layer.compute(
            ticker="TEST",
            realized_vol=0.20,
            convergence_score=-0.5,
            current_position_pct=0.05,
            current_price=100.0,
        )
        assert result["action"] == "no_edge"
        assert result["edge_positive"] is False

    def test_sizing_basis_kelly_cap(self):
        """When kelly_cap < vol_sized and kelly > 0, sizing_basis = 'kelly_cap'."""
        layer = SizingLayer(target_annual_vol=0.15, max_single_pct=0.40, kelly_fraction=0.25)
        # realized_vol=0.30 -> vol_sized=0.5->capped 0.40
        # convergence=0.6 -> win_prob=0.65 -> kelly_cap~0.104 < vol_sized
        result = layer.compute(
            ticker="TEST",
            realized_vol=0.30,
            convergence_score=0.6,
            current_position_pct=0.10,
            current_price=100.0,
        )
        assert result["sizing_basis"] == "kelly_cap"
