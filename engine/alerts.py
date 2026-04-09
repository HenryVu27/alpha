"""Twilio WhatsApp alert manager for the trading system."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from twilio.rest import Client


WHATSAPP_CHAR_LIMIT = 1600


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def emoji(self) -> str:
        return {
            AlertSeverity.INFO: "\U0001f7e2",       # 🟢
            AlertSeverity.WARNING: "\u26a0\ufe0f",   # ⚠️
            AlertSeverity.CRITICAL: "\U0001f6a8",    # 🚨
        }[self]


class AlertManager:
    """Manages alert formatting, batching, and Twilio WhatsApp delivery."""

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        to_number: str,
        store=None,
    ):
        self._account_sid = account_sid
        self._auth_token = auth_token
        self._from_number = from_number
        self._to_number = to_number
        self._store = store
        self._alert_queue: list[tuple[AlertSeverity, str, str]] = []
        self._client: Optional[Client] = None

    def format_alert(self, severity: AlertSeverity, alert_type: str, message: str) -> str:
        """Format an alert message with emoji, type, timestamp, and message (plain text)."""
        now = datetime.now(timezone.utc).strftime("%H:%M")
        return f"{severity.emoji} {alert_type} [{now} UTC]\n{message}"

    def queue_alert(self, severity: AlertSeverity, alert_type: str, message: str):
        """Add an alert to the batch queue."""
        self._alert_queue.append((severity, alert_type, message))

    def flush_queue(self) -> list[str]:
        """Combine all queued alerts into messages respecting WhatsApp char limit."""
        if not self._alert_queue:
            return []
        parts = [
            self.format_alert(sev, atype, msg)
            for sev, atype, msg in self._alert_queue
        ]
        self._alert_queue.clear()

        # Split into multiple messages if combined text exceeds limit
        messages: list[str] = []
        current: list[str] = []
        current_len = 0

        for part in parts:
            # Account for separator between parts
            separator_len = len("\n\n") if current else 0
            if current and current_len + separator_len + len(part) > WHATSAPP_CHAR_LIMIT:
                messages.append("\n\n".join(current))
                current = [part]
                current_len = len(part)
            else:
                current.append(part)
                current_len += separator_len + len(part)

        if current:
            messages.append("\n\n".join(current))

        return messages

    def _send_whatsapp(self, text: str):
        """Send a message via Twilio WhatsApp API."""
        if self._client is None:
            self._client = Client(self._account_sid, self._auth_token)
        self._client.messages.create(
            body=text,
            from_=self._from_number,
            to=self._to_number,
        )

    async def send_alert(
        self, severity: AlertSeverity, alert_type: str, message: str
    ):
        """Send or queue an alert. CRITICAL sends immediately; others are queued."""
        # Always save to store if available
        if self._store is not None:
            self._store.save_alert(severity.value, alert_type, message)

        if severity == AlertSeverity.CRITICAL:
            text = self.format_alert(severity, alert_type, message)
            await asyncio.to_thread(self._send_whatsapp, text)
        else:
            self.queue_alert(severity, alert_type, message)

    async def flush_and_send(self):
        """Flush the queue and send each combined message."""
        messages = self.flush_queue()
        for msg in messages:
            await asyncio.to_thread(self._send_whatsapp, msg)

    async def send_daily_summary(self, summary: dict):
        """Format and send a daily summary via WhatsApp (plain text)."""
        lines = ["\U0001f4ca Daily Summary"]
        if "regime" in summary:
            lines.append(f"Regime: {summary['regime']}")
        if "positions" in summary:
            for ticker, state in summary["positions"].items():
                lines.append(f"  {ticker}: {state}")
        if "sentiment_velocity" in summary:
            lines.append(f"Sentiment velocity: {summary['sentiment_velocity']:.3f}")
        text = "\n".join(lines)
        await asyncio.to_thread(self._send_whatsapp, text)
