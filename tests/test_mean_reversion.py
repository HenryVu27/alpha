"""Tests for Factor-Residual Mean Reversion layer."""

import math

import numpy as np
import pandas as pd
import pytest

from layers.mean_reversion import FactorModel, MeanReversionLayer, OUProcess


class TestOUProcess:
    """Tests for Ornstein-Uhlenbeck process calibration."""

    def test_calibrate_recovers_parameters(self):
        """Simulate OU process (theta=0.5, mu=0, sigma=1, 500 steps), calibrate,
        verify theta within 0.3 of true value."""
        rng = np.random.default_rng(42)
        theta_true = 0.5
        mu_true = 0.0
        sigma_true = 1.0
        dt = 1 / 252
        n = 5000

        x = np.zeros(n)
        x[0] = 0.5
        for i in range(1, n):
            dx = theta_true * (mu_true - x[i - 1]) * dt + sigma_true * math.sqrt(dt) * rng.standard_normal()
            x[i] = x[i - 1] + dx

        ou = OUProcess()
        ou.calibrate(x, dt=dt)

        assert abs(ou.theta - theta_true) < 0.3, f"theta={ou.theta}, expected ~{theta_true}"

    def test_half_life(self):
        """Set theta=0.5, verify half_life = ln(2)/0.5."""
        ou = OUProcess()
        ou.theta = 0.5
        expected = math.log(2) / 0.5
        assert math.isclose(ou.half_life(), expected, rel_tol=1e-9)

    def test_z_score(self):
        """Set theta=0.5, mu=0, sigma=1, z_score(2.0) > 0."""
        ou = OUProcess()
        ou.theta = 0.5
        ou.mu = 0.0
        ou.sigma = 1.0
        z = ou.z_score(2.0)
        assert z > 0


class TestFactorModel:
    """Tests for simple OLS factor model."""

    def test_fit_recovers_betas(self):
        """200 samples, ticker = 1.2*market + 0.5*sector - 0.3*vol + noise,
        verify betas within 0.3."""
        rng = np.random.default_rng(123)
        n = 200
        market = rng.standard_normal(n)
        sector = rng.standard_normal(n)
        vol = rng.standard_normal(n)
        noise = rng.standard_normal(n) * 0.1

        returns = 1.2 * market + 0.5 * sector - 0.3 * vol + noise

        factors = pd.DataFrame({"market": market, "sector": sector, "vol": vol})

        fm = FactorModel()
        fm.fit(returns, factors)

        assert abs(fm.betas["market"] - 1.2) < 0.3
        assert abs(fm.betas["sector"] - 0.5) < 0.3
        assert abs(fm.betas["vol"] - (-0.3)) < 0.3

    def test_residuals_lower_std(self):
        """Verify residuals have lower std than raw returns."""
        rng = np.random.default_rng(456)
        n = 200
        market = rng.standard_normal(n)
        noise = rng.standard_normal(n) * 0.3
        returns = 2.0 * market + noise

        factors = pd.DataFrame({"market": market})

        fm = FactorModel()
        fm.fit(returns, factors)
        resid = fm.residuals(returns, factors)

        assert np.std(resid) < np.std(returns)


class TestMeanReversionLayer:
    """Tests for MeanReversionLayer signal computation."""

    def test_signal_not_suppressed_in_stalemate(self):
        """z=-2.5, half_life=6, regime='stalemate' -> not suppressed."""
        layer = MeanReversionLayer(half_life_suppression_days=15)
        signal = layer.compute_signal(
            ticker="TEST",
            z_score=-2.5,
            half_life=6.0,
            regime="stalemate",
            fair_value=100.0,
            current_price=95.0,
            ou_params={"theta": 0.5, "mu": 0.0, "sigma": 1.0},
            factor_betas={"market": 1.2},
        )
        assert signal["suppressed"] is False
        assert signal["ticker"] == "TEST"
        assert signal["z_score"] == -2.5

    def test_signal_suppressed_in_crisis(self):
        """z=-2.5, half_life=20, regime='escalation' -> suppressed."""
        layer = MeanReversionLayer(half_life_suppression_days=15)
        signal = layer.compute_signal(
            ticker="TEST",
            z_score=-2.5,
            half_life=20.0,
            regime="escalation",
            fair_value=100.0,
            current_price=95.0,
            ou_params={"theta": 0.5, "mu": 0.0, "sigma": 1.0},
            factor_betas={"market": 1.2},
        )
        assert signal["suppressed"] is True
