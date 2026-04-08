"""Factor-Residual Mean Reversion layer.

Decomposes returns into factor exposure + residual, fits an Ornstein-Uhlenbeck
process on the residual, and generates mean-reversion signals gated by regime.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


class OUProcess:
    """Ornstein-Uhlenbeck process calibration via MLE (OLS on dX = a + b*X)."""

    def __init__(self) -> None:
        self.theta: float = 0.0
        self.mu: float = 0.0
        self.sigma: float = 0.0

    def calibrate(self, series: np.ndarray, dt: float = 1 / 252) -> None:
        """Calibrate OU parameters from a time series.

        Runs OLS regression: dX = a + b * X(t) and maps coefficients to OU params.
        """
        dx = np.diff(series)
        x = series[:-1]

        # OLS: dX = a + b*X  =>  [ones, X] @ [a, b]' = dX
        A = np.column_stack([np.ones_like(x), x])
        result, _, _, _ = np.linalg.lstsq(A, dx, rcond=None)
        a, b = result[0], result[1]

        self.theta = max(-b / dt, 1e-6)
        self.mu = a / (self.theta * dt) if self.theta > 1e-6 else float(np.mean(series))

        residuals = dx - A @ result
        self.sigma = float(np.std(residuals)) / math.sqrt(dt)

    def half_life(self) -> float:
        """Return the half-life in the same time units as dt."""
        if self.theta <= 0:
            return float("inf")
        return math.log(2) / self.theta

    def z_score(self, current_value: float) -> float:
        """Compute z-score of current value relative to OU equilibrium."""
        if self.sigma <= 0 or self.theta <= 0:
            return 0.0
        return (current_value - self.mu) / (self.sigma / math.sqrt(2 * self.theta))


class FactorModel:
    """Simple OLS factor model."""

    def __init__(self) -> None:
        self.betas: dict[str, float] = {}
        self.intercept: float = 0.0

    def fit(self, returns: np.ndarray, factors: pd.DataFrame) -> None:
        """Fit factor model via OLS: returns = intercept + sum(beta_i * factor_i)."""
        A = np.column_stack([np.ones(len(returns)), factors.values])
        result, _, _, _ = np.linalg.lstsq(A, returns, rcond=None)
        self.intercept = float(result[0])
        self.betas = {col: float(result[i + 1]) for i, col in enumerate(factors.columns)}

    def predict(self, factors: pd.DataFrame) -> np.ndarray:
        """Predict returns from factor exposures."""
        pred = np.full(len(factors), self.intercept)
        for col, beta in self.betas.items():
            pred = pred + beta * factors[col].values
        return pred

    def residuals(self, returns: np.ndarray, factors: pd.DataFrame) -> np.ndarray:
        """Compute residuals: returns - predicted."""
        return returns - self.predict(factors)


class MeanReversionLayer:
    """Mean reversion signal layer with regime-based suppression."""

    SUPPRESSION_REGIMES = {"escalation", "crisis"}

    def __init__(self, half_life_suppression_days: float = 15) -> None:
        self.half_life_suppression_days = half_life_suppression_days
        self.factor_models: dict[str, FactorModel] = {}
        self.ou_models: dict[str, dict[str, OUProcess]] = {}

    def fit_factor_model(
        self, ticker: str, returns: np.ndarray, factors: pd.DataFrame
    ) -> FactorModel:
        """Fit and store a factor model for a ticker."""
        fm = FactorModel()
        fm.fit(returns, factors)
        self.factor_models[ticker] = fm
        return fm

    def fit_ou_per_regime(
        self,
        ticker: str,
        residuals: np.ndarray,
        regime_labels: np.ndarray,
        regime_names: list[str] | None = None,
        dt: float = 1 / 252,
    ) -> dict[str, OUProcess]:
        """Fit separate OU process per regime.

        If a regime has < 20 samples, use the full sample instead.
        """
        if regime_names is None:
            regime_names = [str(u) for u in np.unique(regime_labels)]

        ou_dict: dict[str, OUProcess] = {}
        for name in regime_names:
            mask = regime_labels == name
            subset = residuals[mask] if np.sum(mask) >= 20 else residuals
            ou = OUProcess()
            ou.calibrate(subset, dt=dt)
            ou_dict[name] = ou

        self.ou_models[ticker] = ou_dict
        return ou_dict

    def compute_signal(
        self,
        ticker: str,
        z_score: float,
        half_life: float,
        regime: str,
        fair_value: float,
        current_price: float,
        ou_params: dict[str, float],
        factor_betas: dict[str, float],
    ) -> dict[str, Any]:
        """Compute a mean-reversion signal dict.

        Suppresses entry signals when regime is escalation/crisis AND
        half_life exceeds the suppression threshold.
        """
        suppressed = (
            regime in self.SUPPRESSION_REGIMES
            and half_life > self.half_life_suppression_days
        )

        return {
            "ticker": ticker,
            "z_score": z_score,
            "half_life": half_life,
            "regime": regime,
            "fair_value": fair_value,
            "current_price": current_price,
            "ou_params": ou_params,
            "factor_betas": factor_betas,
            "suppressed": suppressed,
        }
