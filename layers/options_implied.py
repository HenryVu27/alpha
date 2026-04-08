"""Options-Implied Volatility Signals – Task 10: Layer 4.

Extracts threshold-based signals from options market data:
VIX level, VIX term structure, CBOE SKEW Index, and put/call ratio.
"""

from __future__ import annotations

from typing import Optional


class OptionsImpliedLayer:
    """Extract signals from options-market implied volatility data.

    Parameters
    ----------
    vix_high:
        VIX level at or above which 'vix_elevated' is triggered.
    vix_low:
        VIX level at or below which 'vix_low' is triggered.
    skew_elevated:
        CBOE SKEW Index level at or above which 'skew_elevated' is triggered.
    pcr_heavy:
        Put/call ratio at or above which 'pcr_heavy' is triggered.
    pcr_light:
        Put/call ratio at or below which 'pcr_light' is triggered.
    """

    def __init__(
        self,
        vix_high: float = 30,
        vix_low: float = 18,
        skew_elevated: float = 150,
        pcr_heavy: float = 1.3,
        pcr_light: float = 0.7,
    ) -> None:
        self.vix_high = vix_high
        self.vix_low = vix_low
        self.skew_elevated = skew_elevated
        self.pcr_heavy = pcr_heavy
        self.pcr_light = pcr_light
        self._prev_term_state: Optional[str] = None

    def compute(
        self,
        vix: float,
        vix3m: Optional[float] = None,
        skew: Optional[float] = None,
        put_call_ratio: Optional[float] = None,
    ) -> dict:
        """Compute options-implied volatility signals.

        Parameters
        ----------
        vix:
            Current VIX (spot, 1-month implied volatility index).
        vix3m:
            3-month VIX level. When provided, term structure is calculated.
        skew:
            CBOE SKEW Index value.
        put_call_ratio:
            Equity put/call ratio.

        Returns
        -------
        dict with keys:
            vix, vix3m, term_structure_ratio, term_structure_state,
            skew, put_call_ratio, signals, data_staleness_seconds.
        """
        signals: list[str] = []

        # ── VIX term structure ────────────────────────────────────────────────
        term_structure_ratio: Optional[float] = None
        term_structure_state: Optional[str] = None

        if vix3m is not None:
            term_structure_ratio = vix3m / vix
            if term_structure_ratio > 1.0:
                term_structure_state = "contango"
            else:
                term_structure_state = "backwardation"

            # Detect flip from previous state
            if (
                self._prev_term_state is not None
                and self._prev_term_state != term_structure_state
            ):
                signals.append("term_structure_flip")

            self._prev_term_state = term_structure_state

        # ── VIX level ─────────────────────────────────────────────────────────
        if vix >= self.vix_high:
            signals.append("vix_elevated")
        elif vix <= self.vix_low:
            signals.append("vix_low")

        # ── SKEW ──────────────────────────────────────────────────────────────
        if skew is not None and skew >= self.skew_elevated:
            signals.append("skew_elevated")

        # ── Put/call ratio ────────────────────────────────────────────────────
        if put_call_ratio is not None:
            if put_call_ratio >= self.pcr_heavy:
                signals.append("pcr_heavy")
            elif put_call_ratio <= self.pcr_light:
                signals.append("pcr_light")

        return {
            "vix": vix,
            "vix3m": vix3m,
            "term_structure_ratio": term_structure_ratio,
            "term_structure_state": term_structure_state,
            "skew": skew,
            "put_call_ratio": put_call_ratio,
            "signals": signals,
            "data_staleness_seconds": 0,
        }
