"""Price monitor that checks positions and watchlist against target levels."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.alerts import AlertManager


class PriceMonitor:
    """Monitors prices against stop-loss, profit-target, and buy-target levels."""

    def __init__(self, config, fetcher, alert_manager: AlertManager):
        self._config = config
        self._fetcher = fetcher
        self._alert_manager = alert_manager
        self._triggered: set[str] = set()

    def check_price_levels(self, prices: dict[str, float]) -> list[dict]:
        """Check prices against configured levels and return triggered alerts.

        Each alert dict has keys: type, ticker, price, level, severity.
        Uses _triggered set to suppress duplicate alerts.
        """
        alerts: list[dict] = []

        # Check owned positions for stop_loss and profit_target
        for pos in self._config.portfolio.positions:
            ticker = pos.ticker
            if ticker not in prices:
                continue
            price = prices[ticker]

            # Stop loss
            key_sl = f"{ticker}_stop_loss"
            if key_sl not in self._triggered and price <= pos.stop_loss:
                self._triggered.add(key_sl)
                alerts.append({
                    "type": "stop_loss",
                    "ticker": ticker,
                    "price": price,
                    "level": pos.stop_loss,
                    "severity": "CRITICAL",
                })

            # Profit target
            key_pt = f"{ticker}_profit_target"
            if key_pt not in self._triggered and price >= pos.profit_target:
                self._triggered.add(key_pt)
                alerts.append({
                    "type": "profit_target",
                    "ticker": ticker,
                    "price": price,
                    "level": pos.profit_target,
                    "severity": "WARNING",
                })

        # Check watchlist for buy_target
        for item in self._config.watchlist:
            ticker = item.ticker
            if ticker not in prices:
                continue
            price = prices[ticker]

            key_bt = f"{ticker}_buy_target"
            if key_bt not in self._triggered and price <= item.buy_target:
                self._triggered.add(key_bt)
                alerts.append({
                    "type": "buy_target",
                    "ticker": ticker,
                    "price": price,
                    "level": item.buy_target,
                    "severity": "INFO",
                })

        return alerts

    def reset_trigger(self, ticker: str, alert_type: str):
        """Remove a trigger so the alert can fire again."""
        key = f"{ticker}_{alert_type}"
        self._triggered.discard(key)

    async def run_owned_loop(self, interval_seconds: int = 60):
        """Async loop that periodically checks owned position prices."""
        while True:
            try:
                tickers = [p.ticker for p in self._config.portfolio.positions]
                prices = self._fetcher.get_current_prices(tickers)
                alerts = self.check_price_levels(prices)
                for alert in alerts:
                    from engine.alerts import AlertSeverity
                    severity = AlertSeverity[alert["severity"]]
                    await self._alert_manager.send_alert(
                        severity, alert["type"], f"{alert['ticker']} at {alert['price']} (level: {alert['level']})"
                    )
            except Exception:
                pass  # logged elsewhere
            await asyncio.sleep(interval_seconds)

    async def run_watchlist_loop(self, interval_seconds: int = 300):
        """Async loop that periodically checks watchlist prices."""
        while True:
            try:
                tickers = [w.ticker for w in self._config.watchlist]
                prices = self._fetcher.get_current_prices(tickers)
                alerts = self.check_price_levels(prices)
                for alert in alerts:
                    from engine.alerts import AlertSeverity
                    severity = AlertSeverity[alert["severity"]]
                    await self._alert_manager.send_alert(
                        severity, alert["type"], f"{alert['ticker']} at {alert['price']} (level: {alert['level']})"
                    )
            except Exception:
                pass
            await asyncio.sleep(interval_seconds)
