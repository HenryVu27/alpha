"""Signal evaluation metrics for backtest framework."""

import numpy as np


class SignalEvaluator:
    """Evaluates the quality of trading signals against actual price data."""

    def compute_hit_rate(self, signals: list[dict], prices: np.ndarray) -> float:
        """Compute the fraction of signals where price moved in the predicted direction.

        For each signal: get timestamp_idx, half_life, direction.
        end_idx = min(idx + half_life, len(prices) - 1).
        Check if price moved in the signal's direction within the window.
        """
        if not signals:
            return 0.0

        hits = 0
        for signal in signals:
            idx = signal["timestamp_idx"]
            half_life = signal["half_life"]
            direction = signal["direction"]
            end_idx = min(idx + half_life, len(prices) - 1)

            price_start = prices[idx]
            price_end = prices[end_idx]

            if direction == "bearish" and price_end < price_start:
                hits += 1
            elif direction == "bullish" and price_end > price_start:
                hits += 1

        return hits / len(signals)

    def compute_lead_time(
        self,
        signal_idx: int,
        prices: np.ndarray,
        direction: str,
        threshold_pct: float = 0.05,
    ) -> int:
        """Scan forward from signal_idx for price crossing threshold.

        For bearish: when pct_change < -threshold.
        For bullish: when pct_change > threshold.
        Returns number of periods ahead, or -1 if never hit.
        """
        base_price = prices[signal_idx]

        for i in range(signal_idx + 1, len(prices)):
            pct_change = (prices[i] - base_price) / base_price

            if direction == "bearish" and pct_change < -threshold_pct:
                return i - signal_idx
            elif direction == "bullish" and pct_change > threshold_pct:
                return i - signal_idx

        return -1

    def compute_false_positive_rate(
        self,
        signals: list[dict],
        prices: np.ndarray,
        threshold_pct: float = 0.03,
    ) -> float:
        """Compute fraction of signals where max favorable move didn't cross threshold.

        For each signal, check the max favorable move within the half_life window.
        If max favorable move doesn't cross threshold, it's a false positive.
        """
        if not signals:
            return 0.0

        false_positives = 0
        for signal in signals:
            idx = signal["timestamp_idx"]
            half_life = signal["half_life"]
            direction = signal["direction"]
            end_idx = min(idx + half_life, len(prices) - 1)

            base_price = prices[idx]
            window = prices[idx : end_idx + 1]

            if direction == "bearish":
                # Max favorable move is max price drop
                max_favorable = (base_price - np.min(window)) / base_price
            else:
                # Max favorable move is max price rise
                max_favorable = (np.max(window) - base_price) / base_price

            if max_favorable < threshold_pct:
                false_positives += 1

        return false_positives / len(signals)

    def regime_accuracy(self, predicted: list[str], actual: list[str]) -> float:
        """Simple accuracy: count matches / total."""
        if not predicted:
            return 0.0
        matches = sum(1 for p, a in zip(predicted, actual) if p == a)
        return matches / len(predicted)
