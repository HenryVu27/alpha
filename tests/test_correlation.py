"""Tests for Cross-Asset Correlation Monitor (CorrelationLayer)."""

import numpy as np
import pandas as pd
import pytest

from layers.correlation import CorrelationLayer


class TestRollingCorrelation:
    """Tests for rolling_correlation method."""

    def test_returns_3x3_matrix_with_diagonal_ones(self):
        """rolling_correlation returns a 3x3 matrix with diagonal=1 for 3-column DataFrame."""
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.randn(20, 3),
            columns=["A", "B", "C"],
        )
        layer = CorrelationLayer(window=5)
        corr = layer.rolling_correlation(df, window=5)

        assert corr.shape == (3, 3), f"Expected (3,3), got {corr.shape}"
        np.testing.assert_array_almost_equal(
            np.diag(corr), np.ones(3), decimal=6,
            err_msg="Diagonal of correlation matrix must be 1.0",
        )


class TestEigenvalueRatio:
    """Tests for eigenvalue_ratio method."""

    def test_near_one_for_highly_correlated(self):
        """eigenvalue_ratio near 1.0 for a nearly-perfect correlation matrix."""
        # Build a 3x3 correlation matrix where all off-diag ~ 0.99
        corr = np.array([
            [1.0, 0.99, 0.99],
            [0.99, 1.0, 0.99],
            [0.99, 0.99, 1.0],
        ])
        layer = CorrelationLayer()
        ratio = layer.eigenvalue_ratio(corr)

        # First eigenvalue dominates → ratio should be close to 1.0
        assert ratio > 0.9, f"Expected ratio > 0.9 for highly correlated matrix, got {ratio:.4f}"

    def test_near_one_third_for_identity(self):
        """eigenvalue_ratio near 1/3 for identity matrix (uncorrelated assets)."""
        corr = np.eye(3)
        layer = CorrelationLayer()
        ratio = layer.eigenvalue_ratio(corr)

        assert abs(ratio - 1.0 / 3.0) < 0.01, (
            f"Expected ratio ≈ 1/3 for identity matrix, got {ratio:.4f}"
        )


class TestDetectSignFlips:
    """Tests for detect_sign_flips method."""

    def test_detects_positive_to_negative_flip(self):
        """prev={A_B: 0.5}, curr={A_B: -0.2} → one flip ('A', 'B', 'pos_to_neg')."""
        layer = CorrelationLayer()
        prev = {"A_B": 0.5}
        curr = {"A_B": -0.2}

        flips = layer.detect_sign_flips(prev, curr)

        assert len(flips) == 1, f"Expected 1 flip, got {len(flips)}"
        asset_a, asset_b, direction = flips[0]
        assert asset_a == "A"
        assert asset_b == "B"
        assert direction == "pos_to_neg"

    def test_detects_negative_to_positive_flip(self):
        """prev={X_Y: -0.3}, curr={X_Y: 0.4} → one flip ('X', 'Y', 'neg_to_pos')."""
        layer = CorrelationLayer()
        prev = {"X_Y": -0.3}
        curr = {"X_Y": 0.4}

        flips = layer.detect_sign_flips(prev, curr)

        assert len(flips) == 1
        asset_a, asset_b, direction = flips[0]
        assert asset_a == "X"
        assert asset_b == "Y"
        assert direction == "neg_to_pos"

    def test_no_flip_when_same_sign(self):
        """No flips when correlation stays positive."""
        layer = CorrelationLayer()
        prev = {"A_B": 0.3}
        curr = {"A_B": 0.7}

        flips = layer.detect_sign_flips(prev, curr)
        assert len(flips) == 0


class TestComputeLeadLag:
    """Tests for compute_lead_lag method."""

    def test_detects_lead_lag_relationship(self):
        """B leads A by 1 day (a[1:] = 0.8*b[:-1] + noise) → best_lag != 0."""
        np.random.seed(123)
        n = 100
        b = np.random.randn(n)
        # A is a lagged version of B: A[t] = 0.8 * B[t-1] + small noise
        a = np.empty(n)
        a[0] = np.random.randn()
        a[1:] = 0.8 * b[:-1] + np.random.randn(n - 1) * 0.1

        layer = CorrelationLayer()
        result = layer.compute_lead_lag(a, b, max_lag=3)

        assert "best_lag" in result
        assert "correlation" in result
        assert result["best_lag"] != 0, (
            f"Expected non-zero best_lag for lagged relationship, got {result['best_lag']}"
        )

    def test_returns_correct_keys(self):
        """compute_lead_lag returns dict with best_lag and correlation keys."""
        np.random.seed(0)
        a = np.random.randn(50)
        b = np.random.randn(50)
        layer = CorrelationLayer()
        result = layer.compute_lead_lag(a, b)

        assert "best_lag" in result
        assert "correlation" in result
        assert isinstance(result["best_lag"], (int, np.integer))
        assert isinstance(result["correlation"], float)


class TestCompute:
    """Tests for the full compute() method."""

    def _make_price_df(self, n_rows=50, n_assets=3, seed=42):
        """Helper: create a price DataFrame with n_assets columns."""
        np.random.seed(seed)
        prices = 100 * np.exp(
            np.cumsum(np.random.randn(n_rows, n_assets) * 0.01, axis=0)
        )
        return pd.DataFrame(prices, columns=["SPY", "GLD", "TLT"])

    def test_compute_returns_all_required_fields(self):
        """Full compute() returns all required fields."""
        df = self._make_price_df(n_rows=60)
        layer = CorrelationLayer(window=5, slow_window=21, crisis_threshold=0.55)
        result = layer.compute(df, focus_pairs=[("SPY", "GLD")])

        required_keys = {
            "eigenvalue_ratio",
            "eigenvalue_ratio_21d",
            "correlation_regime",
            "sign_flips",
            "lead_lag",
            "focus_pair_correlations",
            "data_staleness_seconds",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_compute_correlation_regime_valid(self):
        """correlation_regime is a known string value."""
        df = self._make_price_df(n_rows=60)
        layer = CorrelationLayer(window=5, slow_window=21)
        result = layer.compute(df)

        assert result["correlation_regime"] in ("crisis_clustering", "normal"), (
            f"Unexpected regime: {result['correlation_regime']}"
        )

    def test_compute_focus_pair_correlations(self):
        """focus_pair_correlations contains entry for each requested pair."""
        df = self._make_price_df(n_rows=60)
        layer = CorrelationLayer(window=5, slow_window=21)
        focus = [("SPY", "GLD"), ("SPY", "TLT")]
        result = layer.compute(df, focus_pairs=focus)

        fp = result["focus_pair_correlations"]
        assert ("SPY", "GLD") in fp, "Expected SPY-GLD pair in focus_pair_correlations"
        assert ("SPY", "TLT") in fp, "Expected SPY-TLT pair in focus_pair_correlations"
        for pair, val in fp.items():
            assert -1.0 <= val <= 1.0, f"Correlation out of range for {pair}: {val}"

    def test_compute_sign_flips_and_lead_lag_types(self):
        """sign_flips is a list and lead_lag is a dict."""
        df = self._make_price_df(n_rows=60)
        layer = CorrelationLayer(window=5, slow_window=21)
        result = layer.compute(df, focus_pairs=[("SPY", "GLD")])

        assert isinstance(result["sign_flips"], list)
        assert isinstance(result["lead_lag"], dict)

    def test_compute_updates_prev_pair_corrs(self):
        """_prev_pair_corrs is populated after calling compute()."""
        df = self._make_price_df(n_rows=60)
        layer = CorrelationLayer(window=5, slow_window=21)
        assert layer._prev_pair_corrs == {}

        layer.compute(df, focus_pairs=[("SPY", "GLD")])
        assert len(layer._prev_pair_corrs) > 0, (
            "_prev_pair_corrs should be populated after compute()"
        )
