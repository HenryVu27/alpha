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

    def _make_manager(self, store=None):
        return AlertManager(
            account_sid="fake_sid",
            auth_token="fake_token",
            from_number="whatsapp:+14155238886",
            to_number="whatsapp:+14696496967",
            store=store,
        )

    def test_format_alert_includes_emoji_and_type(self):
        """format_alert returns string with emoji and alert_type (plain text)."""
        manager = self._make_manager()
        result = manager.format_alert(
            AlertSeverity.WARNING, "Volatility Spike", "VIX above 30"
        )
        assert "\u26a0\ufe0f" in result
        assert "Volatility Spike" in result
        assert "VIX above 30" in result
        assert "UTC" in result
        # Should NOT contain Markdown bold markers
        assert "*Volatility Spike*" not in result

    def test_batch_non_critical_via_flush_queue(self):
        """Non-critical alerts are batched and flush_queue returns combined string(s)."""
        manager = self._make_manager()
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
        manager = self._make_manager()
        result = manager.flush_queue()
        assert result == []

    def test_flush_queue_splits_long_messages(self):
        """flush_queue splits messages that exceed WhatsApp char limit."""
        manager = self._make_manager()
        # Queue many alerts to exceed 1600 chars
        for i in range(50):
            manager.queue_alert(
                AlertSeverity.INFO, f"Alert {i}", "X" * 30
            )
        result = manager.flush_queue()
        assert len(result) > 1
        for msg in result:
            assert len(msg) <= 1600

    @pytest.mark.asyncio
    async def test_send_alert_critical_sends_immediately(self):
        """CRITICAL alerts call _send_whatsapp immediately, not queued."""
        manager = self._make_manager()
        send_mock = MagicMock()
        manager._send_whatsapp = send_mock

        with patch("engine.alerts.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(return_value=None)
            await manager.send_alert(
                AlertSeverity.CRITICAL, "Stop Loss Hit", "DAL dropped below stop"
            )
            mock_asyncio.to_thread.assert_awaited_once()
            call_args = mock_asyncio.to_thread.call_args
            assert call_args[0][0] is send_mock
            assert "Stop Loss Hit" in call_args[0][1]

        # Should NOT be in queue
        assert len(manager._alert_queue) == 0

    @pytest.mark.asyncio
    async def test_send_alert_non_critical_queues(self):
        """Non-critical alerts go to queue, not sent immediately."""
        manager = self._make_manager()

        with patch("engine.alerts.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(return_value=None)
            await manager.send_alert(
                AlertSeverity.INFO, "Price Update", "AAPL at 150"
            )
            mock_asyncio.to_thread.assert_not_awaited()

        assert len(manager._alert_queue) == 1

    @pytest.mark.asyncio
    async def test_send_alert_saves_to_store(self):
        """send_alert saves to store when store is provided."""
        mock_store = MagicMock()
        manager = self._make_manager(store=mock_store)

        with patch("engine.alerts.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(return_value=None)
            await manager.send_alert(
                AlertSeverity.WARNING, "Test Alert", "test message"
            )

        mock_store.save_alert.assert_called_once_with(
            "warning", "Test Alert", "test message"
        )

    @patch("engine.alerts.Client")
    def test_send_whatsapp_creates_client_and_sends(self, mock_client_cls):
        """_send_whatsapp creates Twilio client and calls messages.create."""
        manager = self._make_manager()
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        manager._send_whatsapp("Hello WhatsApp")

        mock_client_cls.assert_called_once_with("fake_sid", "fake_token")
        mock_client.messages.create.assert_called_once_with(
            body="Hello WhatsApp",
            from_="whatsapp:+14155238886",
            to="whatsapp:+14696496967",
        )
