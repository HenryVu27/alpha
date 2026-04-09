"""Master Decision Engine — aggregates all layer outputs into convergence signals."""

from datetime import datetime, timezone
from statistics import mean

CRISIS_REGIMES = {"escalation", "crisis"}
BULLISH_REGIMES = {"risk_on"}
SAFE_HAVEN_ASSETS = {"GLD", "TLT", "^VIX"}


class DecisionEngine:
    """Central integration point that consumes all 6 layers and produces trading signals."""

    def __init__(
        self,
        convergence_min_layers: int = 3,
        high_threshold: float = 0.5,
        moderate_threshold: float = 0.3,
    ):
        self.convergence_min_layers = convergence_min_layers
        self.high_threshold = high_threshold
        self.moderate_threshold = moderate_threshold

    # ------------------------------------------------------------------
    # Individual layer signal converters
    # ------------------------------------------------------------------

    def regime_to_signal(self, regime_label: str, asset_class: str = "equity") -> int:
        """Convert regime label to directional signal."""
        is_safe_haven = asset_class in ("gold", "treasury", "volatility")
        if regime_label in CRISIS_REGIMES:
            return 1 if is_safe_haven else -1
        if regime_label in BULLISH_REGIMES:
            return -1 if is_safe_haven else 1
        return 0

    def sentiment_to_signal(self, sentiment_mean: float, velocity: float) -> int:
        """Convert sentiment mean and velocity to directional signal."""
        if sentiment_mean < -0.3 or velocity < -0.1:
            return -1
        if sentiment_mean > 0.3 or velocity > 0.1:
            return 1
        return 0

    def options_to_signal(self, signals: list[str]) -> int:
        """Convert options-implied signals to directional signal."""
        bearish = {"vix_elevated", "term_structure_backwardation", "skew_elevated", "pcr_heavy"}
        bullish = {"vix_low", "pcr_light"}
        bearish_count = sum(1 for s in signals if s in bearish)
        bullish_count = sum(1 for s in signals if s in bullish)
        if bearish_count > bullish_count:
            return -1
        if bullish_count > bearish_count:
            return 1
        return 0

    def correlation_to_signal(self, corr_regime: str) -> int:
        """Convert correlation regime to directional signal."""
        return -1 if corr_regime == "crisis_clustering" else 0

    def mean_reversion_to_signal(self, ticker_data: dict) -> int:
        """Convert mean-reversion z-score to directional signal."""
        if ticker_data.get("signal_suppressed", False):
            return 0
        z_score = ticker_data.get("z_score", 0.0)
        if z_score < -2.0:
            return 1  # undervalued
        if z_score > 2.0:
            return -1  # overvalued
        return 0

    def sizing_to_signal(self, ticker_data: dict) -> int:
        """Convert sizing edge to directional signal.

        Returns 0 (neutral) when no sizing data exists — uncalibrated
        layers should not vote.
        """
        if not ticker_data:
            return 0  # no data = abstain, not bearish
        if ticker_data.get("edge_positive", False):
            return 1
        return -1

    # ------------------------------------------------------------------
    # Convergence & conviction
    # ------------------------------------------------------------------

    def compute_convergence(self, signals: dict[str, int]) -> tuple[float, int]:
        """Compute convergence score and count of agreeing layers.

        Returns (score, agreeing) where score is the mean of signal values
        and agreeing is the count of signals matching the majority sign.
        """
        values = list(signals.values())
        if not values:
            return (0.0, 0)
        score = mean(values)

        # Determine majority sign
        positive = sum(1 for v in values if v > 0)
        negative = sum(1 for v in values if v < 0)
        if positive > negative:
            majority_sign = 1
        elif negative > positive:
            majority_sign = -1
        else:
            majority_sign = 0

        if majority_sign == 0:
            agreeing = sum(1 for v in values if v == 0)
        else:
            agreeing = sum(1 for v in values if (v > 0) == (majority_sign > 0))

        return (score, agreeing)

    def classify_conviction(self, score: float, agreeing: int) -> dict:
        """Classify conviction level and direction from convergence metrics."""
        abs_score = abs(score)

        # Conviction level
        if abs_score >= self.high_threshold and agreeing >= self.convergence_min_layers:
            conviction = "high"
        elif abs_score >= self.moderate_threshold and agreeing >= 2:
            conviction = "moderate"
        elif agreeing < 2 or abs_score < 0.1:
            conviction = "mixed"
        else:
            conviction = "low"

        # Direction
        if abs_score < 0.05:
            direction = "neutral"
        elif score > 0:
            direction = "bullish"
        else:
            direction = "bearish"

        return {"conviction": conviction, "direction": direction}

    # ------------------------------------------------------------------
    # Full aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        layer_outputs: dict,
        tickers: list[str],
        asset_classes: dict | None = None,
    ) -> dict:
        """Aggregate all layer outputs into per-ticker convergence signals.

        Args:
            layer_outputs: Dict with keys for each layer (regime, sentiment,
                options, correlation, mean_reversion, sizing).
            tickers: List of ticker symbols to evaluate.
            asset_classes: Optional mapping of ticker -> asset class string.

        Returns:
            Full output dict with timestamp, market-wide info, and per-ticker signals.
        """
        if asset_classes is None:
            asset_classes = {}

        regime_label = layer_outputs.get("regime", {}).get("regime_label", "unknown")

        # Market-wide signals
        sentiment_data = layer_outputs.get("sentiment", {})
        sentiment_signal = self.sentiment_to_signal(
            sentiment_data.get("mean", 0.0),
            sentiment_data.get("velocity", 0.0),
        )

        options_data = layer_outputs.get("options", {})
        options_signal = self.options_to_signal(options_data.get("signals", []))

        corr_data = layer_outputs.get("correlation", {})
        correlation_signal = self.correlation_to_signal(corr_data.get("corr_regime", "normal"))

        # Per-ticker signals
        ticker_results = {}
        for ticker in tickers:
            asset_class = asset_classes.get(ticker, "equity")
            regime_signal = self.regime_to_signal(regime_label, asset_class)

            mr_data = layer_outputs.get("mean_reversion", {}).get(ticker, {})
            mr_signal = self.mean_reversion_to_signal(mr_data)

            sizing_data = layer_outputs.get("sizing", {}).get(ticker, {})
            sizing_signal = self.sizing_to_signal(sizing_data)

            signals = {
                "regime": regime_signal,
                "sentiment": sentiment_signal,
                "options": options_signal,
                "correlation": correlation_signal,
                "mean_reversion": mr_signal,
                "sizing": sizing_signal,
            }

            score, agreeing = self.compute_convergence(signals)
            conviction = self.classify_conviction(score, agreeing)

            ticker_results[ticker] = {
                "signals": signals,
                "convergence": {"score": score, "agreeing": agreeing},
                "conviction": conviction,
            }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regime_label": regime_label,
            "market_wide": {
                "sentiment": sentiment_signal,
                "options": options_signal,
                "correlation": correlation_signal,
            },
            "tickers": ticker_results,
        }
