"""Tests for the AlertManager and AlertSeverity."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engine.alerts import AlertManager, AlertSeverity


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_emoji_values(self):
        """Each severity level maps to the correct emoji."""
        assert AlertSeverity.INFO.emoji == "\U0001f7e2"       # 🟢
        assert AlertSeverity.WARNING.emoji == "\u26a0\ufe0f"   # ⚠️
        assert AlertSeverity.CRITICAL.emoji == "\U0001f6a8"    # 🚨

    def test_enum_values(self):
        """Enum string values are correct."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestAlertManager:
    """Tests for AlertManager."""

    def test_format_alert_includes_emoji_and_type(self):
        """format_alert returns string with emoji and alert_type."""
        manager = AlertManager(bot_token="fake", chat_id="123", store=None)
        result = manager.format_alert(
            AlertSeverity.WARNING, "Volatility Spike", "VIX above 30"
        )
        assert "\u26a0\ufe0f" in result
        assert "*Volatility Spike*" in result
        assert "VIX above 30" in result
        assert "UTC" in result

    def test_batch_non_critical_via_flush_queue(self):
        """Non-critical alerts are batched and flush_queue returns one combined string."""
        manager = AlertManager(bot_token="fake", chat_id="123", store=None)
        manager.queue_alert(AlertSeverity.INFO, "Price Update", "AAPL at 150")
        manager.queue_alert(AlertSeverity.WARNING, "Stop Loss Near", "DAL close to SL")

        result = manager.flush_queue()
        assert len(result) == 1
        assert "Price Update" in result[0]
        assert "Stop Loss Near" in result[0]
        # Queue should be empty after flush
        assert len(manager._alert_queue) == 0

    def test_flush_queue_empty(self):
        """flush_queue returns empty list when no alerts queued."""
        manager = AlertManager(bot_token="fake", chat_id="123", store=None)
        result = manager.flush_queue()
        assert result == []

    @pytest.mark.asyncio
    async def test_send_alert_critical_sends_immediately(self):
        """CRITICAL alerts call _send_telegram immediately, not queued."""
        manager = AlertManager(bot_token="fake", chat_id="123", store=None)
        manager._send_telegram = AsyncMock()

        await manager.send_alert(
            AlertSeverity.CRITICAL, "Stop Loss Hit", "DAL dropped below stop"
        )

        manager._send_telegram.assert_awaited_once()
        call_text = manager._send_telegram.call_args[0][0]
        assert "Stop Loss Hit" in call_text
        # Should NOT be in queue
        assert len(manager._alert_queue) == 0

    @pytest.mark.asyncio
    async def test_send_alert_non_critical_queues(self):
        """Non-critical alerts go to queue, not sent immediately."""
        manager = AlertManager(bot_token="fake", chat_id="123", store=None)
        manager._send_telegram = AsyncMock()

        await manager.send_alert(
            AlertSeverity.INFO, "Price Update", "AAPL at 150"
        )

        manager._send_telegram.assert_not_awaited()
        assert len(manager._alert_queue) == 1

    @pytest.mark.asyncio
    async def test_send_alert_saves_to_store(self):
        """send_alert saves to store when store is provided."""
        mock_store = MagicMock()
        manager = AlertManager(bot_token="fake", chat_id="123", store=mock_store)
        manager._send_telegram = AsyncMock()

        await manager.send_alert(
            AlertSeverity.WARNING, "Test Alert", "test message"
        )

        mock_store.save_alert.assert_called_once_with(
            "warning", "Test Alert", "test message"
        )
