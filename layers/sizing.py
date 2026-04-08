"""Position Sizing layer.

Combines inverse-volatility sizing (primary) with Kelly criterion (as a cap)
to determine per-position allocation percentages.
"""

from __future__ import annotations

import time
from typing import Any


class SizingLayer:
    """Position sizing using inverse-vol as primary and Kelly as a cap.

    Parameters
    ----------
    kelly_fraction:
        Fraction of full Kelly to use (default 0.25 = quarter-Kelly).
    target_annual_vol:
        Annualised volatility target for each position (default 15%).
    max_single_pct:
        Maximum allocation for a single position (default 40%).
    max_total_pct:
        Maximum total portfolio allocation (default 85%).
    max_drawdown_pct:
        Maximum allowed drawdown before sizing is cut (default 5%).
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        target_annual_vol: float = 0.15,
        max_single_pct: float = 0.40,
        max_total_pct: float = 0.85,
        max_drawdown_pct: float = 0.05,
    ) -> None:
        self.kelly_fraction = kelly_fraction
        self.target_annual_vol = target_annual_vol
        self.max_single_pct = max_single_pct
        self.max_total_pct = max_total_pct
        self.max_drawdown_pct = max_drawdown_pct

    # ------------------------------------------------------------------
    # Core sizing methods
    # ------------------------------------------------------------------

    def inverse_vol_size(self, realized_vol: float) -> float:
        """Compute inverse-volatility position size.

        Parameters
        ----------
        realized_vol:
            Annualised realised volatility of the asset.

        Returns
        -------
        float
            Allocation percentage, capped at ``max_single_pct``.
        """
        if realized_vol <= 0:
            return self.max_single_pct

        raw = self.target_annual_vol / realized_vol
        return min(raw, self.max_single_pct)

    def kelly_criterion(self, win_prob: float, win_loss_ratio: float) -> float:
        """Compute fractional Kelly allocation.

        Parameters
        ----------
        win_prob:
            Estimated probability of a winning trade (0–1).
        win_loss_ratio:
            Ratio of average win size to average loss size.

        Returns
        -------
        float
            Fractional Kelly allocation (>= 0).  Returns 0.0 when edge is
            non-positive or ``win_loss_ratio`` is invalid.
        """
        if win_loss_ratio <= 0:
            return 0.0

        full_kelly = win_prob - (1.0 - win_prob) / win_loss_ratio
        if full_kelly <= 0:
            return 0.0

        return full_kelly * self.kelly_fraction

    # ------------------------------------------------------------------
    # Main compute entry-point
    # ------------------------------------------------------------------

    def compute(
        self,
        ticker: str,
        realized_vol: float,
        convergence_score: float,
        current_position_pct: float,
        current_price: float,
        entry_price: float | None = None,
        stop_loss: float | None = None,
    ) -> dict[str, Any]:
        """Compute position sizing recommendation for a single ticker.

        Parameters
        ----------
        ticker:
            Asset identifier.
        realized_vol:
            Annualised realised volatility.
        convergence_score:
            Signal convergence score (positive = bullish edge).
        current_position_pct:
            Current position as a fraction of portfolio.
        current_price:
            Current market price.
        entry_price:
            Price at which the position was entered (optional).
        stop_loss:
            Hard stop-loss price level (optional).

        Returns
        -------
        dict
            Sizing recommendation and diagnostic fields.
        """
        # 1. Inverse-vol size
        vol_sized = self.inverse_vol_size(realized_vol)

        # 2. Kelly cap
        win_prob = 0.5 + abs(convergence_score) * 0.25
        win_loss_ratio = 1.5
        kelly_cap = self.kelly_criterion(win_prob, win_loss_ratio)

        # 3. Final allocation
        if kelly_cap > 0:
            final = min(vol_sized, kelly_cap)
        else:
            final = 0.0

        # Cap at max_single_pct
        final = min(final, self.max_single_pct)

        # 4. Sizing basis
        if kelly_cap == 0:
            sizing_basis = "no_edge"
        elif kelly_cap < vol_sized:
            sizing_basis = "kelly_cap"
        else:
            sizing_basis = "vol_target"

        # 5. Edge flag
        edge_positive = convergence_score > 0 and kelly_cap > 0

        # 6. Stop-loss distance
        if stop_loss is not None:
            stop_loss_distance_pct = (current_price - stop_loss) / current_price
        else:
            stop_loss_distance_pct = None

        # 7. Action recommendation
        if current_position_pct > final * 1.1:
            action = "oversized"
        elif current_position_pct < final * 0.9 and edge_positive:
            action = "undersized"
        elif not edge_positive:
            action = "no_edge"
        else:
            action = "appropriately_sized"

        return {
            "ticker": ticker,
            "vol_sized_pct": vol_sized,
            "kelly_cap_pct": kelly_cap,
            "final_allocation_pct": final,
            "sizing_basis": sizing_basis,
            "edge_positive": edge_positive,
            "current_position_pct": current_position_pct,
            "action": action,
            "stop_loss_distance_pct": stop_loss_distance_pct,
            "data_staleness_seconds": time.time(),
        }
