"""Tests for regime detection (HMM + BOCPD)."""

import numpy as np
import pytest

from layers.regime import (
    BayesianChangepoint,
    RegimeDetector,
    RegimeHMM,
    select_n_states_bic,
)


class TestBayesianChangepoint:
    """Tests for Bayesian Online Changepoint Detection."""

    def test_bocpd_detects_changepoint(self):
        """Generate 100 samples from N(0,1) then 50 from N(5,1).
        Max changepoint prob around index 100 should be > 0.3."""
        np.random.seed(42)
        signal = np.concatenate([
            np.random.normal(0, 1, 100),
            np.random.normal(5, 1, 50),
        ])
        bocpd = BayesianChangepoint(hazard_lambda=250)
        probs = bocpd.run(signal)

        assert probs.shape == (len(signal),)
        # Max changepoint probability in the region around index 100
        region = probs[90:110]
        assert np.max(region) > 0.3, (
            f"Expected changepoint prob > 0.3 near index 100, got {np.max(region):.4f}"
        )

    def test_bocpd_no_changepoint_in_stable(self):
        """200 samples from N(0,1), max prob < 0.5."""
        np.random.seed(42)
        signal = np.random.normal(0, 1, 200)
        bocpd = BayesianChangepoint(hazard_lambda=250)
        probs = bocpd.run(signal)

        assert probs.shape == (len(signal),)
        assert np.max(probs) < 0.5, (
            f"Expected max prob < 0.5 for stable signal, got {np.max(probs):.4f}"
        )


class TestRegimeHMM:
    """Tests for the HMM wrapper."""

    def test_hmm_fit_and_predict(self):
        """300 samples from 3 distinct multivariate normals (100 each),
        predict should detect at least 2 regimes."""
        np.random.seed(42)
        data = np.vstack([
            np.random.multivariate_normal([0, 0], np.eye(2) * 0.1, 100),
            np.random.multivariate_normal([5, 5], np.eye(2) * 0.1, 100),
            np.random.multivariate_normal([-5, -5], np.eye(2) * 0.1, 100),
        ])
        hmm = RegimeHMM(n_states=3, covariance_type="diag")
        hmm.fit(data)
        labels = hmm.predict(data)

        assert labels.shape == (300,)
        n_unique = len(np.unique(labels))
        assert n_unique >= 2, f"Expected at least 2 regimes, got {n_unique}"

    def test_hmm_transition_matrix_rows_sum_to_one(self):
        """Transition matrix rows should sum to 1.0."""
        np.random.seed(42)
        data = np.vstack([
            np.random.multivariate_normal([0, 0], np.eye(2) * 0.1, 100),
            np.random.multivariate_normal([5, 5], np.eye(2) * 0.1, 100),
            np.random.multivariate_normal([-5, -5], np.eye(2) * 0.1, 100),
        ])
        hmm = RegimeHMM(n_states=3, covariance_type="diag")
        hmm.fit(data)
        trans_mat = hmm.transition_matrix()

        assert trans_mat.shape == (3, 3)
        np.testing.assert_allclose(
            trans_mat.sum(axis=1), np.ones(3), atol=1e-6
        )


class TestBICSelection:
    """Tests for BIC-based state selection."""

    def test_bic_state_selection(self):
        """2-cluster data, select_n_states_bic should return 2 or 3."""
        np.random.seed(42)
        data = np.vstack([
            np.random.multivariate_normal([0, 0], np.eye(2) * 0.1, 150),
            np.random.multivariate_normal([10, 10], np.eye(2) * 0.1, 150),
        ])
        best_n = select_n_states_bic(data, max_states=5)
        assert best_n in (2, 3), f"Expected 2 or 3 states, got {best_n}"


class TestRegimeDetector:
    """Tests for the full RegimeDetector pipeline."""

    def test_regime_detector_full_pipeline(self):
        """Fit on 6-dim data with two regimes (calm/crisis),
        detect() returns dict with all required fields."""
        np.random.seed(42)
        calm = np.random.multivariate_normal(
            np.zeros(6), np.eye(6) * 0.1, 150
        )
        crisis = np.random.multivariate_normal(
            np.ones(6) * 5, np.eye(6) * 0.5, 150
        )
        data = np.vstack([calm, crisis])

        detector = RegimeDetector(n_states=3, hazard_lambda=250)
        detector.fit(data)
        result = detector.detect(data)

        required_keys = {
            "regime_label",
            "regime_label_idx",
            "regime_probabilities",
            "changepoint_probability",
            "transition_matrix",
            "hmm_log_likelihood",
            "data_staleness_seconds",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )
        assert isinstance(result["regime_label"], str)
        assert isinstance(result["regime_label_idx"], (int, np.integer))
        assert isinstance(result["regime_probabilities"], np.ndarray)
        assert isinstance(result["changepoint_probability"], float)
        assert isinstance(result["transition_matrix"], np.ndarray)
