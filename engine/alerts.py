"""Telegram alert manager for the trading system."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional

from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes


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
    """Manages alert formatting, batching, and Telegram delivery."""

    def __init__(self, bot_token: str, chat_id: str, store=None):
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._store = store
        self._alert_queue: list[tuple[AlertSeverity, str, str]] = []
        self._bot: Optional[Bot] = None

    def format_alert(self, severity: AlertSeverity, alert_type: str, message: str) -> str:
        """Format an alert message with emoji, type, timestamp, and message."""
        now = datetime.now(timezone.utc).strftime("%H:%M")
        return f"{severity.emoji} *{alert_type}* [{now} UTC]\n{message}"

    def queue_alert(self, severity: AlertSeverity, alert_type: str, message: str):
        """Add an alert to the batch queue."""
        self._alert_queue.append((severity, alert_type, message))

    def flush_queue(self) -> list[str]:
        """Combine all queued alerts into a single message, clear queue, return list."""
        if not self._alert_queue:
            return []
        parts = [
            self.format_alert(sev, atype, msg)
            for sev, atype, msg in self._alert_queue
        ]
        self._alert_queue.clear()
        return ["\n\n".join(parts)]

    async def _send_telegram(self, text: str):
        """Send a message via Telegram Bot API."""
        if self._bot is None:
            self._bot = Bot(token=self._bot_token)
        await self._bot.send_message(
            chat_id=self._chat_id, text=text, parse_mode="Markdown"
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
            await self._send_telegram(text)
        else:
            self.queue_alert(severity, alert_type, message)

    async def flush_and_send(self):
        """Flush the queue and send each combined message."""
        messages = self.flush_queue()
        for msg in messages:
            await self._send_telegram(msg)

    async def send_daily_summary(self, summary: dict):
        """Format and send a daily summary via Telegram."""
        lines = ["\U0001f4ca *Daily Summary*"]
        if "regime" in summary:
            lines.append(f"Regime: {summary['regime']}")
        if "positions" in summary:
            for ticker, state in summary["positions"].items():
                lines.append(f"  {ticker}: {state}")
        if "sentiment_velocity" in summary:
            lines.append(f"Sentiment velocity: {summary['sentiment_velocity']:.3f}")
        text = "\n".join(lines)
        await self._send_telegram(text)

    def setup_commands(self, app: Application, config_updater=None):
        """Register Telegram command handlers on the Application."""

        async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text("System is running.")

        async def cmd_regime(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text("Regime info not yet available.")

        async def cmd_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if config_updater and context.args:
                key, value = context.args[0], context.args[1] if len(context.args) > 1 else None
                config_updater("set", key, value)
                await update.message.reply_text(f"Set {key} = {value}")
            else:
                await update.message.reply_text("Usage: /set <key> <value>")

        async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if config_updater and context.args:
                config_updater("add", context.args[0], context.args[1] if len(context.args) > 1 else None)
                await update.message.reply_text(f"Added {context.args[0]}")
            else:
                await update.message.reply_text("Usage: /add <item> [value]")

        async def cmd_remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if config_updater and context.args:
                config_updater("remove", context.args[0], None)
                await update.message.reply_text(f"Removed {context.args[0]}")
            else:
                await update.message.reply_text("Usage: /remove <item>")

        app.add_handler(CommandHandler("status", cmd_status))
        app.add_handler(CommandHandler("regime", cmd_regime))
        app.add_handler(CommandHandler("set", cmd_set))
        app.add_handler(CommandHandler("add", cmd_add))
        app.add_handler(CommandHandler("remove", cmd_remove))
