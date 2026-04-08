"""Cross-Asset Correlation Monitor.

Implements CorrelationLayer, which tracks rolling correlations across assets,
detects crisis clustering via eigenvalue concentration, identifies sign flips,
and measures lead-lag relationships between pairs.
"""

import time
from typing import Optional

import numpy as np
import pandas as pd


class CorrelationLayer:
    """Monitor cross-asset correlations and detect regime changes.

    Parameters
    ----------
    window : int
        Fast rolling window in days (default: 5).
    slow_window : int
        Slow rolling window in days (default: 21).
    crisis_threshold : float
        Eigenvalue ratio above which the regime is classified as
        'crisis_clustering' (default: 0.55).
    """

    def __init__(
        self,
        window: int = 5,
        slow_window: int = 21,
        crisis_threshold: float = 0.55,
    ) -> None:
        self.window = window
        self.slow_window = slow_window
        self.crisis_threshold = crisis_threshold
        self._prev_pair_corrs: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def rolling_correlation(self, data: pd.DataFrame, window: int) -> np.ndarray:
        """Compute a correlation matrix from recent price data.

        Parameters
        ----------
        data : pd.DataFrame
            Price (or any raw) data, one column per asset.
        window : int
            Number of most-recent return observations to use.

        Returns
        -------
        np.ndarray
            Square correlation matrix (n_assets × n_assets).
        """
        returns = data.pct_change().dropna()
        recent = returns.iloc[-window:]
        corr = recent.corr().values
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    def eigenvalue_ratio(self, corr_matrix: np.ndarray) -> float:
        """Return the fraction of variance explained by the largest eigenvalue.

        A ratio near 1.0 indicates assets are highly co-moving (crisis);
        a ratio near 1/n indicates uncorrelated assets.

        Parameters
        ----------
        corr_matrix : np.ndarray
            Symmetric correlation matrix.

        Returns
        -------
        float
            Largest eigenvalue divided by the sum of all eigenvalues.
        """
        eigvals = np.linalg.eigvalsh(corr_matrix)
        # Sort descending
        eigvals = np.sort(eigvals)[::-1]
        total = eigvals.sum()
        if total == 0.0:
            return 0.0
        return float(eigvals[0] / total)

    def detect_sign_flips(
        self,
        prev_corrs: dict[str, float],
        curr_corrs: dict[str, float],
    ) -> list[tuple[str, str, str]]:
        """Identify pairs whose correlation changed sign between two snapshots.

        Parameters
        ----------
        prev_corrs : dict
            Previous correlations keyed as "AssetA_AssetB".
        curr_corrs : dict
            Current correlations keyed as "AssetA_AssetB".

        Returns
        -------
        list of (asset_a, asset_b, direction)
            direction is "pos_to_neg" or "neg_to_pos".
        """
        flips: list[tuple[str, str, str]] = []
        for key in set(prev_corrs) & set(curr_corrs):
            prev_val = prev_corrs[key]
            curr_val = curr_corrs[key]
            if prev_val > 0 and curr_val < 0:
                direction = "pos_to_neg"
            elif prev_val < 0 and curr_val > 0:
                direction = "neg_to_pos"
            else:
                continue
            asset_a, asset_b = key.split("_", 1)
            flips.append((asset_a, asset_b, direction))
        return flips

    def compute_lead_lag(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray,
        max_lag: int = 3,
    ) -> dict:
        """Find the lag at which series_a and series_b are most correlated.

        Positive lag means series_a leads series_b; negative lag means
        series_b leads series_a.

        Parameters
        ----------
        series_a, series_b : array-like
            1-D time series of equal length.
        max_lag : int
            Maximum absolute lag to test.

        Returns
        -------
        dict
            ``best_lag`` (int) and ``correlation`` (float) at that lag.
        """
        a = np.asarray(series_a, dtype=float)
        b = np.asarray(series_b, dtype=float)

        best_lag = 0
        best_corr = 0.0

        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                corr_val = float(np.corrcoef(a, b)[0, 1])
            elif lag > 0:
                # a leads b: align a[lag:] with b[:-lag]
                corr_val = float(np.corrcoef(a[lag:], b[:-lag])[0, 1])
            else:
                # lag < 0 → b leads a: align a[:lag] with b[-lag:]
                corr_val = float(np.corrcoef(a[:lag], b[-lag:])[0, 1])

            if np.isnan(corr_val):
                corr_val = 0.0

            if abs(corr_val) > abs(best_corr):
                best_corr = corr_val
                best_lag = lag

        return {"best_lag": int(best_lag), "correlation": float(best_corr)}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        data: pd.DataFrame,
        focus_pairs: Optional[list[tuple[str, str]]] = None,
    ) -> dict:
        """Run the full correlation monitor pipeline.

        Parameters
        ----------
        data : pd.DataFrame
            Price data (rows = time, columns = assets).
        focus_pairs : list of (str, str), optional
            Specific asset pairs for detailed correlation and lead-lag analysis.

        Returns
        -------
        dict
            Contains:
            - eigenvalue_ratio          : float  (fast window)
            - eigenvalue_ratio_21d      : float  (slow window)
            - correlation_regime        : str    ("crisis_clustering" | "normal")
            - sign_flips                : list
            - lead_lag                  : dict   (keyed by pair tuple)
            - focus_pair_correlations   : dict
            - data_staleness_seconds    : float
        """
        focus_pairs = focus_pairs or []
        columns = list(data.columns)

        # ---- Correlation matrices ----------------------------------------
        fast_corr = self.rolling_correlation(data, self.window)
        slow_corr = self.rolling_correlation(data, self.slow_window)

        # ---- Eigenvalue ratios -------------------------------------------
        fast_ev_ratio = self.eigenvalue_ratio(fast_corr)
        slow_ev_ratio = self.eigenvalue_ratio(slow_corr)

        # ---- Correlation regime ------------------------------------------
        if fast_ev_ratio > self.crisis_threshold:
            regime = "crisis_clustering"
        else:
            regime = "normal"

        # ---- Focus pair correlations from fast matrix --------------------
        focus_pair_corrs: dict[tuple[str, str], float] = {}
        for asset_a, asset_b in focus_pairs:
            if asset_a in columns and asset_b in columns:
                idx_a = columns.index(asset_a)
                idx_b = columns.index(asset_b)
                focus_pair_corrs[(asset_a, asset_b)] = float(
                    fast_corr[idx_a, idx_b]
                )

        # ---- Build current pair correlations dict (all upper-triangle) ---
        curr_pair_corrs: dict[str, float] = {}
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                key = f"{columns[i]}_{columns[j]}"
                curr_pair_corrs[key] = float(fast_corr[i, j])

        # ---- Sign flip detection ----------------------------------------
        sign_flips = self.detect_sign_flips(self._prev_pair_corrs, curr_pair_corrs)
        # Update state for next call
        self._prev_pair_corrs = curr_pair_corrs

        # ---- Lead-lag analysis on focus pairs ---------------------------
        returns = data.pct_change().dropna()
        lead_lag: dict[tuple[str, str], dict] = {}
        for asset_a, asset_b in focus_pairs:
            if asset_a in returns.columns and asset_b in returns.columns:
                a_vals = returns[asset_a].values
                b_vals = returns[asset_b].values
                lead_lag[(asset_a, asset_b)] = self.compute_lead_lag(a_vals, b_vals)

        return {
            "eigenvalue_ratio": fast_ev_ratio,
            "eigenvalue_ratio_21d": slow_ev_ratio,
            "correlation_regime": regime,
            "sign_flips": sign_flips,
            "lead_lag": lead_lag,
            "focus_pair_correlations": focus_pair_corrs,
            "data_staleness_seconds": 0.0,
        }
