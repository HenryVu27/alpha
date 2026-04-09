"""SQLite-backed data store for the trading system."""

import json
import os
import sqlite3
from typing import Optional


class DataStore:
    """Durable persistence layer using SQLite."""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def init_db(self):
        """Create all tables and indexes."""
        cur = self.conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS layer_outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                layer TEXT NOT NULL,
                payload TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ticker TEXT NOT NULL,
                price REAL NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                severity TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS decision_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                snapshot TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS historical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                UNIQUE(ticker, date)
            )
        """)

        # Indexes
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_layer_outputs_layer_ts
            ON layer_outputs (layer, timestamp DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_prices_ticker_ts
            ON prices (ticker, timestamp DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_ts
            ON alerts (timestamp DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_decision_snapshots_ts
            ON decision_snapshots (timestamp DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_data_ticker_date
            ON historical_data (ticker, date)
        """)

        self.conn.commit()

    # -- layer_outputs --

    def save_layer_output(self, layer: str, payload_dict: dict):
        """Insert a layer output with JSON-serialized payload."""
        self.conn.execute(
            "INSERT INTO layer_outputs (layer, payload) VALUES (?, ?)",
            (layer, json.dumps(payload_dict)),
        )
        self.conn.commit()

    def get_latest_layer_output(self, layer: str) -> Optional[dict]:
        """Get the most recent layer output, JSON-parsed."""
        row = self.conn.execute(
            "SELECT payload FROM layer_outputs WHERE layer = ? ORDER BY id DESC LIMIT 1",
            (layer,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload"])

    # -- prices --

    def save_price(self, ticker: str, price: float):
        """Insert a price record."""
        self.conn.execute(
            "INSERT INTO prices (ticker, price) VALUES (?, ?)",
            (ticker, price),
        )
        self.conn.commit()

    def get_prices(self, ticker: str, limit: int = 100) -> list[dict]:
        """Get prices in chronological (ASC) order."""
        rows = self.conn.execute(
            "SELECT ticker, price, timestamp FROM prices WHERE ticker = ? ORDER BY id ASC LIMIT ?",
            (ticker, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- alerts --

    def save_alert(self, severity: str, alert_type: str, message: str):
        """Insert an alert."""
        self.conn.execute(
            "INSERT INTO alerts (severity, alert_type, message) VALUES (?, ?, ?)",
            (severity, alert_type, message),
        )
        self.conn.commit()

    def get_alerts(self, limit: int = 50) -> list[dict]:
        """Get alerts in reverse chronological order."""
        rows = self.conn.execute(
            "SELECT severity, alert_type, message, timestamp FROM alerts ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- decision_snapshots --

    def save_decision_snapshot(self, snapshot_dict: dict):
        """Insert a decision snapshot with JSON-serialized data."""
        self.conn.execute(
            "INSERT INTO decision_snapshots (snapshot) VALUES (?)",
            (json.dumps(snapshot_dict),),
        )
        self.conn.commit()

    def get_latest_decision_snapshot(self) -> Optional[dict]:
        """Get the most recent decision snapshot, JSON-parsed."""
        row = self.conn.execute(
            "SELECT snapshot FROM decision_snapshots ORDER BY id DESC LIMIT 1",
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["snapshot"])

    # -- historical_data --

    def save_historical_data(self, ticker: str, rows: list[dict]):
        """Bulk insert/replace historical OHLCV data."""
        self.conn.executemany(
            """INSERT OR REPLACE INTO historical_data
               (ticker, date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    ticker,
                    r["date"],
                    r.get("open"),
                    r.get("high"),
                    r.get("low"),
                    r.get("close"),
                    r.get("volume"),
                )
                for r in rows
            ],
        )
        self.conn.commit()

    def get_historical_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict]:
        """Get historical data with optional date filters, ordered ASC."""
        query = "SELECT ticker, date, open, high, low, close, volume FROM historical_data WHERE ticker = ?"
        params: list = [ticker]

        if start_date is not None:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date is not None:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # -- lifecycle --

    def close(self):
        """Close the database connection."""
        self.conn.close()
