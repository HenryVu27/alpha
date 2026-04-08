"""Tests for the PriceMonitor."""

from unittest.mock import MagicMock

import pytest

from engine.price_monitor import PriceMonitor


def _make_config():
    """Create a mock config with positions and watchlist."""
    config = MagicMock()

    pos_dal = MagicMock()
    pos_dal.ticker = "DAL"
    pos_dal.stop_loss = 65.55
    pos_dal.profit_target = 82.0

    pos_nvda = MagicMock()
    pos_nvda.ticker = "NVDA"
    pos_nvda.stop_loss = 173.85
    pos_nvda.profit_target = 200.0

    config.portfolio.positions = [pos_dal, pos_nvda]

    watch_vti = MagicMock()
    watch_vti.ticker = "VTI"
    watch_vti.buy_target = 333.0

    config.watchlist = [watch_vti]

    return config


class TestPriceMonitor:
    """Tests for PriceMonitor.check_price_levels."""

    def test_stop_loss_triggered(self):
        """Alert when price drops to or below stop_loss."""
        config = _make_config()
        monitor = PriceMonitor(config=config, fetcher=MagicMock(), alert_manager=MagicMock())

        alerts = monitor.check_price_levels({"DAL": 65.00, "NVDA": 185.0})

        dal_alerts = [a for a in alerts if a["ticker"] == "DAL"]
        assert len(dal_alerts) == 1
        assert dal_alerts[0]["type"] == "stop_loss"
        assert dal_alerts[0]["severity"] == "CRITICAL"

    def test_profit_target_triggered(self):
        """Alert when price reaches or exceeds profit_target."""
        config = _make_config()
        monitor = PriceMonitor(config=config, fetcher=MagicMock(), alert_manager=MagicMock())

        alerts = monitor.check_price_levels({"DAL": 70.0, "NVDA": 201.0})

        nvda_alerts = [a for a in alerts if a["ticker"] == "NVDA"]
        assert len(nvda_alerts) == 1
        assert nvda_alerts[0]["type"] == "profit_target"
        assert nvda_alerts[0]["severity"] == "WARNING"

    def test_buy_target_triggered(self):
        """Alert when watchlist price drops to or below buy_target."""
        config = _make_config()
        monitor = PriceMonitor(config=config, fetcher=MagicMock(), alert_manager=MagicMock())

        alerts = monitor.check_price_levels({"VTI": 330.0})

        assert len(alerts) == 1
        assert alerts[0]["type"] == "buy_target"
        assert alerts[0]["ticker"] == "VTI"
        assert alerts[0]["severity"] == "INFO"

    def test_no_alert_when_price_in_range(self):
        """No alerts when all prices are within normal range."""
        config = _make_config()
        monitor = PriceMonitor(config=config, fetcher=MagicMock(), alert_manager=MagicMock())

        alerts = monitor.check_price_levels({
            "DAL": 70.0,
            "NVDA": 185.0,
            "VTI": 340.0,
        })

        assert len(alerts) == 0

    def test_duplicate_alert_suppressed(self):
        """Same alert should not fire twice (triggered set prevents duplicates)."""
        config = _make_config()
        monitor = PriceMonitor(config=config, fetcher=MagicMock(), alert_manager=MagicMock())

        alerts1 = monitor.check_price_levels({"DAL": 65.0})
        alerts2 = monitor.check_price_levels({"DAL": 64.0})

        assert len(alerts1) == 1
        assert len(alerts2) == 0

    def test_reset_trigger(self):
        """reset_trigger allows alert to fire again."""
        config = _make_config()
        monitor = PriceMonitor(config=config, fetcher=MagicMock(), alert_manager=MagicMock())

        monitor.check_price_levels({"DAL": 65.0})
        monitor.reset_trigger("DAL", "stop_loss")
        alerts = monitor.check_price_levels({"DAL": 64.0})

        assert len(alerts) == 1
