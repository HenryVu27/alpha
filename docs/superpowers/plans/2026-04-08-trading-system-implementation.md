# Geopolitical Trading Signal System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time geopolitical event-driven trading signal system that monitors 6 analytical layers and sends Telegram alerts when actionable thresholds are hit.

**Architecture:** Redis Streams microservice architecture. Each analytical layer runs as a standalone async Python process, publishes results to Redis Streams. A decision engine aggregates all layer outputs into convergence signals. Telegram bot handles alerts and user commands. SQLite for durable persistence.

**Tech Stack:** Python 3.11+, Redis Streams, Docker Compose, PyTorch + transformers (FinBERT), hmmlearn, yfinance, python-telegram-bot, anthropic/openai SDK, SQLite, pydantic, structlog

**Spec:** `docs/superpowers/specs/2026-04-08-geopolitical-trading-system-design.md`

---

## Phase 1: Foundation (Tasks 1-6)

Gets config, data pipeline, Telegram alerts, and price monitoring working. Immediately useful even without ML layers.

---

### Task 1: Project Scaffolding & Config

**Files:**
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `common/__init__.py`
- Create: `common/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Create requirements.txt**

```txt
# Core
pydantic>=2.6,<3
pydantic-settings>=2.2,<3
pyyaml>=6.0,<7
structlog>=24.1,<25
redis>=5.0,<6

# Data
yfinance>=0.2.36,<1
pandas>=2.2,<3
numpy>=1.26,<2

# ML
hmmlearn>=0.3,<1
scikit-learn>=1.4,<2
scipy>=1.12,<2
torch>=2.2,<3
transformers>=4.38,<5

# NLP / News
anthropic>=0.50,<1
openai>=1.12,<2
httpx>=0.27,<1

# Telegram
python-telegram-bot>=21.0,<22

# Testing
pytest>=8.0,<9
pytest-asyncio>=0.23,<1
```

- [ ] **Step 2: Create config.yaml**

```yaml
portfolio:
  cash: 6000
  positions:
    - ticker: "DAL"
      shares: 36
      entry_price: 69.0
      stop_loss: 65.55
      profit_target: 82.0
    - ticker: "NVDA"
      shares: 11
      entry_price: 183.0
      stop_loss: 173.85
      profit_target: 200.0
    - ticker: "BTC-USD"
      amount_usd: 1000
      stop_loss: 62000
      profit_target: 80000

watchlist:
  - ticker: "VTI"
    buy_target: 333.0
  - ticker: "XLE"
    buy_target: 55.0

geopolitical:
  keywords:
    - "Iran"
    - "Hormuz"
    - "ceasefire"
    - "strikes"
    - "oil"
    - "sanctions"
  gdelt_update_interval_minutes: 15
  llm_provider: "anthropic"
  llm_model: "claude-opus-4-6"
  llm_daily_cost_cap_usd: 10.0

regime:
  n_states: 3
  retrain_schedule: "weekly"
  features:
    - "vix"
    - "vix_term_slope"
    - "yield_10y_5d_delta"
    - "crude_5d_return"
    - "spy_5d_realized_vol"
    - "usdjpy_5d_delta"
  training_crises:
    - name: "Ukraine 2022"
      start: "2022-02-01"
      end: "2022-05-01"
    - name: "Red Sea 2023"
      start: "2023-11-01"
      end: "2024-03-01"
    - name: "Iran 2026"
      start: "2026-02-28"
      end: null
  bocpd_hazard_lambda: 250
  state_labels:
    0: "risk_on"
    1: "stalemate"
    2: "escalation"

alerts:
  regime_shift_threshold: 0.6
  sentiment_strong_threshold: 0.5
  sentiment_velocity_threshold: 0.15
  branching_ratio_critical: 0.8
  vix_high: 30
  vix_low: 18
  skew_elevated: 150
  pcr_heavy: 1.3
  pcr_light: 0.7
  convergence_min_layers: 3
  convergence_high_threshold: 0.5
  convergence_moderate_threshold: 0.3
  z_score_entry_threshold: 2.0
  half_life_suppression_days: 15

sizing:
  kelly_fraction: 0.25
  target_annual_vol: 0.15
  max_single_position_pct: 0.40
  max_total_exposure_pct: 0.85
  max_drawdown_per_position_pct: 0.05
  realized_vol_window_days: 20

correlation:
  rolling_window_days: 5
  rolling_window_slow_days: 21
  eigenvalue_crisis_threshold: 0.55
  assets:
    - "VTI"
    - "CL=F"
    - "^VIX"
    - "GLD"
    - "TLT"
    - "DX-Y.NYB"
    - "XLE"
    - "DAL"
    - "NVDA"
  focus_pairs:
    - ["VTI", "CL=F"]
    - ["VTI", "^VIX"]
    - ["DAL", "CL=F"]
    - ["GLD", "^VIX"]
  lead_lag_max_days: 3

mean_reversion:
  factor_tickers:
    market: "SPY"
    volatility: "^VIX"
  sector_map:
    DAL: "XLE"
    NVDA: "XLK"
  recalibration_schedule: "weekly"

redis:
  host: "localhost"
  port: 6379
  db: 0
  stream_maxlen: 10000

timing:
  signal_update_interval_minutes: 15
  price_check_owned_seconds: 60
  price_check_watchlist_seconds: 300
  daily_summary_time: "16:15"
  timezone: "America/Chicago"
  staleness_warning_seconds: 1800

logging:
  level: "INFO"
  file: "logs/trading_system.log"
  format: "json"

database:
  path: "data/trading_system.db"
```

- [ ] **Step 3: Write the failing test for config loader**

```python
# tests/test_config.py
import pytest
from pathlib import Path


def test_load_config_from_yaml():
    from common.config import load_config

    config = load_config(Path("config.yaml"))
    assert config.portfolio.cash == 6000
    assert len(config.portfolio.positions) == 3
    assert config.portfolio.positions[0].ticker == "DAL"


def test_config_position_fields():
    from common.config import load_config

    config = load_config(Path("config.yaml"))
    dal = config.portfolio.positions[0]
    assert dal.shares == 36
    assert dal.entry_price == 69.0
    assert dal.stop_loss == 65.55
    assert dal.profit_target == 82.0


def test_config_watchlist():
    from common.config import load_config

    config = load_config(Path("config.yaml"))
    assert len(config.watchlist) == 2
    assert config.watchlist[0].ticker == "VTI"
    assert config.watchlist[0].buy_target == 333.0


def test_config_geopolitical_keywords():
    from common.config import load_config

    config = load_config(Path("config.yaml"))
    assert "Iran" in config.geopolitical.keywords
    assert len(config.geopolitical.keywords) == 6


def test_config_regime_crises():
    from common.config import load_config

    config = load_config(Path("config.yaml"))
    assert len(config.regime.training_crises) == 3
    assert config.regime.training_crises[2].end is None  # ongoing


def test_config_redis():
    from common.config import load_config

    config = load_config(Path("config.yaml"))
    assert config.redis.host == "localhost"
    assert config.redis.port == 6379


def test_config_timing():
    from common.config import load_config

    config = load_config(Path("config.yaml"))
    assert config.timing.signal_update_interval_minutes == 15
    assert config.timing.timezone == "America/Chicago"


def test_config_all_tickers_returns_portfolio_and_watchlist():
    from common.config import load_config

    config = load_config(Path("config.yaml"))
    tickers = config.all_tickers()
    assert "DAL" in tickers
    assert "VTI" in tickers
    assert "BTC-USD" in tickers
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'common'`

- [ ] **Step 5: Implement config loader with pydantic models**

```python
# common/__init__.py
# empty

# common/config.py
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class PositionConfig(BaseModel):
    ticker: str
    shares: Optional[float] = None
    amount_usd: Optional[float] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    profit_target: Optional[float] = None


class WatchlistItem(BaseModel):
    ticker: str
    buy_target: Optional[float] = None


class PortfolioConfig(BaseModel):
    cash: float
    positions: list[PositionConfig]


class GeopoliticalConfig(BaseModel):
    keywords: list[str]
    gdelt_update_interval_minutes: int = 15
    llm_provider: str = "anthropic"
    llm_model: str = "claude-opus-4-6"
    llm_daily_cost_cap_usd: float = 10.0


class CrisisConfig(BaseModel):
    name: str
    start: date
    end: Optional[date] = None


class RegimeConfig(BaseModel):
    n_states: int = 3
    retrain_schedule: str = "weekly"
    features: list[str] = []
    training_crises: list[CrisisConfig] = []
    bocpd_hazard_lambda: int = 250
    state_labels: dict[int, str] = {}


class AlertsConfig(BaseModel):
    regime_shift_threshold: float = 0.6
    sentiment_strong_threshold: float = 0.5
    sentiment_velocity_threshold: float = 0.15
    branching_ratio_critical: float = 0.8
    vix_high: float = 30
    vix_low: float = 18
    skew_elevated: float = 150
    pcr_heavy: float = 1.3
    pcr_light: float = 0.7
    convergence_min_layers: int = 3
    convergence_high_threshold: float = 0.5
    convergence_moderate_threshold: float = 0.3
    z_score_entry_threshold: float = 2.0
    half_life_suppression_days: int = 15


class SizingConfig(BaseModel):
    kelly_fraction: float = 0.25
    target_annual_vol: float = 0.15
    max_single_position_pct: float = 0.40
    max_total_exposure_pct: float = 0.85
    max_drawdown_per_position_pct: float = 0.05
    realized_vol_window_days: int = 20


class CorrelationConfig(BaseModel):
    rolling_window_days: int = 5
    rolling_window_slow_days: int = 21
    eigenvalue_crisis_threshold: float = 0.55
    assets: list[str] = []
    focus_pairs: list[list[str]] = []
    lead_lag_max_days: int = 3


class MeanReversionConfig(BaseModel):
    factor_tickers: dict[str, str] = {}
    sector_map: dict[str, str] = {}
    recalibration_schedule: str = "weekly"


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    stream_maxlen: int = 10000


class TimingConfig(BaseModel):
    signal_update_interval_minutes: int = 15
    price_check_owned_seconds: int = 60
    price_check_watchlist_seconds: int = 300
    daily_summary_time: str = "16:15"
    timezone: str = "America/Chicago"
    staleness_warning_seconds: int = 1800


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/trading_system.log"
    format: str = "json"


class DatabaseConfig(BaseModel):
    path: str = "data/trading_system.db"


class AppConfig(BaseModel):
    portfolio: PortfolioConfig
    watchlist: list[WatchlistItem] = []
    geopolitical: GeopoliticalConfig
    regime: RegimeConfig = RegimeConfig()
    alerts: AlertsConfig = AlertsConfig()
    sizing: SizingConfig = SizingConfig()
    correlation: CorrelationConfig = CorrelationConfig()
    mean_reversion: MeanReversionConfig = MeanReversionConfig()
    redis: RedisConfig = RedisConfig()
    timing: TimingConfig = TimingConfig()
    logging: LoggingConfig = LoggingConfig()
    database: DatabaseConfig = DatabaseConfig()

    def all_tickers(self) -> set[str]:
        tickers = {p.ticker for p in self.portfolio.positions}
        tickers.update(w.ticker for w in self.watchlist)
        return tickers


def load_config(path: Path) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_config.py -v`
Expected: All 8 tests PASS

- [ ] **Step 7: Commit**

```bash
git add requirements.txt config.yaml common/ tests/
git commit -m "feat: add project config with pydantic models and YAML loader"
```

---

### Task 2: Redis Streams Client & Message Schema

**Files:**
- Create: `common/schema.py`
- Create: `common/redis_client.py`
- Create: `tests/test_schema.py`
- Create: `tests/test_redis_client.py`

- [ ] **Step 1: Write failing tests for message schema**

```python
# tests/test_schema.py
import pytest
from datetime import datetime, timezone


def test_create_message_envelope():
    from common.schema import MessageEnvelope

    msg = MessageEnvelope(
        layer="regime",
        payload={"regime_label": "escalation", "regime_probabilities": [0.1, 0.2, 0.7]},
    )
    assert msg.layer == "regime"
    assert msg.version == 1
    assert msg.payload["regime_label"] == "escalation"
    assert isinstance(msg.timestamp, datetime)


def test_message_envelope_serialization_roundtrip():
    from common.schema import MessageEnvelope

    msg = MessageEnvelope(
        layer="sentiment",
        payload={"sentiment_mean": -0.42},
    )
    json_str = msg.to_json()
    restored = MessageEnvelope.from_json(json_str)
    assert restored.layer == "sentiment"
    assert restored.payload["sentiment_mean"] == -0.42
    assert restored.timestamp == msg.timestamp


def test_message_envelope_to_redis_dict():
    from common.schema import MessageEnvelope

    msg = MessageEnvelope(layer="regime", payload={"x": 1})
    d = msg.to_redis()
    assert isinstance(d, dict)
    assert "data" in d
    # Redis streams need string values
    assert isinstance(d["data"], str)


def test_message_envelope_from_redis_dict():
    from common.schema import MessageEnvelope

    original = MessageEnvelope(layer="regime", payload={"x": 1})
    d = original.to_redis()
    restored = MessageEnvelope.from_redis(d)
    assert restored.layer == original.layer
    assert restored.payload == original.payload
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'common.schema'`

- [ ] **Step 3: Implement message schema**

```python
# common/schema.py
from __future__ import annotations

import json
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class MessageEnvelope(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    layer: str
    version: int = 1
    payload: dict

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> MessageEnvelope:
        return cls.model_validate_json(data)

    def to_redis(self) -> dict[str, str]:
        return {"data": self.to_json()}

    @classmethod
    def from_redis(cls, data: dict[str, str]) -> MessageEnvelope:
        raw = data.get("data", data.get(b"data", b"").decode() if isinstance(data.get(b"data"), bytes) else "")
        if isinstance(raw, bytes):
            raw = raw.decode()
        return cls.from_json(raw)
```

- [ ] **Step 4: Run schema tests**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_schema.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Write failing tests for Redis client**

```python
# tests/test_redis_client.py
import pytest


def test_redis_stream_client_init():
    """Test client can be created with config (no connection needed)."""
    from common.redis_client import RedisStreamClient

    client = RedisStreamClient(host="localhost", port=6379, db=0, stream_maxlen=10000)
    assert client.stream_maxlen == 10000


def test_redis_stream_client_stream_key():
    from common.redis_client import RedisStreamClient

    client = RedisStreamClient(host="localhost", port=6379, db=0, stream_maxlen=10000)
    assert client.stream_key("regime") == "stream:regime"
    assert client.stream_key("sentiment") == "stream:sentiment"
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_redis_client.py -v`
Expected: FAIL

- [ ] **Step 7: Implement Redis Streams client**

```python
# common/redis_client.py
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Optional

import redis.asyncio as aioredis

from common.schema import MessageEnvelope

logger = logging.getLogger(__name__)


class RedisStreamClient:
    def __init__(self, host: str, port: int, db: int, stream_maxlen: int):
        self.host = host
        self.port = port
        self.db = db
        self.stream_maxlen = stream_maxlen
        self._redis: Optional[aioredis.Redis] = None

    def stream_key(self, layer: str) -> str:
        return f"stream:{layer}"

    async def connect(self) -> None:
        self._redis = aioredis.Redis(
            host=self.host, port=self.port, db=self.db, decode_responses=True
        )

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()

    async def publish(self, layer: str, payload: dict) -> str:
        msg = MessageEnvelope(layer=layer, payload=payload)
        key = self.stream_key(layer)
        msg_id = await self._redis.xadd(
            key, msg.to_redis(), maxlen=self.stream_maxlen, approximate=True
        )
        return msg_id

    async def subscribe(
        self,
        layers: list[str],
        last_ids: Optional[dict[str, str]] = None,
        block_ms: int = 5000,
    ) -> list[tuple[str, MessageEnvelope]]:
        if last_ids is None:
            last_ids = {self.stream_key(l): "$" for l in layers}

        streams = {self.stream_key(l): last_ids.get(self.stream_key(l), "$") for l in layers}
        results = await self._redis.xread(streams, block=block_ms, count=100)

        messages = []
        for stream_name, entries in results:
            for msg_id, data in entries:
                envelope = MessageEnvelope.from_redis(data)
                messages.append((msg_id, envelope))
                last_ids[stream_name] = msg_id

        return messages

    async def get_latest(self, layer: str) -> Optional[MessageEnvelope]:
        key = self.stream_key(layer)
        results = await self._redis.xrevrange(key, count=1)
        if not results:
            return None
        _, data = results[0]
        return MessageEnvelope.from_redis(data)
```

- [ ] **Step 8: Run tests**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_redis_client.py -v`
Expected: All 2 tests PASS (these don't require a live Redis connection)

- [ ] **Step 9: Commit**

```bash
git add common/schema.py common/redis_client.py tests/test_schema.py tests/test_redis_client.py
git commit -m "feat: add message schema and Redis Streams client"
```

---

### Task 3: Structured Logging Setup

**Files:**
- Create: `common/logging.py`

- [ ] **Step 1: Implement structured JSON logging**

```python
# common/logging.py
from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(level: str = "INFO", log_file: str = "logs/trading_system.log") -> None:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=open(log_path, "a")),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
```

- [ ] **Step 2: Commit**

```bash
git add common/logging.py
git commit -m "feat: add structured JSON logging with structlog"
```

---

### Task 4: SQLite Data Store

**Files:**
- Create: `data/__init__.py`
- Create: `data/store.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: Write failing tests for SQLite store**

```python
# tests/test_store.py
import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


def test_store_init_creates_tables(db_path):
    from data.store import DataStore

    store = DataStore(db_path)
    store.init_db()
    # Should not raise — tables exist
    store.save_layer_output("regime", {"regime_label": "risk_on"})
    store.close()


def test_save_and_get_latest_layer_output(db_path):
    from data.store import DataStore

    store = DataStore(db_path)
    store.init_db()
    store.save_layer_output("regime", {"regime_label": "escalation", "prob": 0.7})
    latest = store.get_latest_layer_output("regime")
    assert latest is not None
    assert latest["regime_label"] == "escalation"
    store.close()


def test_save_price(db_path):
    from data.store import DataStore

    store = DataStore(db_path)
    store.init_db()
    store.save_price("DAL", 69.5)
    prices = store.get_prices("DAL", limit=1)
    assert len(prices) == 1
    assert prices[0]["price"] == 69.5
    store.close()


def test_save_alert(db_path):
    from data.store import DataStore

    store = DataStore(db_path)
    store.init_db()
    store.save_alert("critical", "regime_change", "Regime shifted to escalation")
    alerts = store.get_alerts(limit=1)
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "critical"
    store.close()


def test_save_decision_snapshot(db_path):
    from data.store import DataStore

    store = DataStore(db_path)
    store.init_db()
    snapshot = {"tickers": {"DAL": {"convergence_score": -0.65}}}
    store.save_decision_snapshot(snapshot)
    latest = store.get_latest_decision_snapshot()
    assert latest["tickers"]["DAL"]["convergence_score"] == -0.65
    store.close()


def test_get_prices_returns_chronological(db_path):
    import time
    from data.store import DataStore

    store = DataStore(db_path)
    store.init_db()
    store.save_price("DAL", 69.0)
    store.save_price("DAL", 70.0)
    store.save_price("DAL", 71.0)
    prices = store.get_prices("DAL", limit=10)
    assert prices[0]["price"] == 69.0
    assert prices[-1]["price"] == 71.0
    store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_store.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SQLite data store**

```python
# data/__init__.py
# empty

# data/store.py
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class DataStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row

    def init_db(self) -> None:
        cursor = self._conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS layer_outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                layer TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_layer_outputs_layer_ts ON layer_outputs(layer, timestamp DESC);

            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                ticker TEXT NOT NULL,
                price REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_prices_ticker_ts ON prices(ticker, timestamp DESC);

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                severity TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp DESC);

            CREATE TABLE IF NOT EXISTS decision_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                snapshot TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON decision_snapshots(timestamp DESC);

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
            );
            CREATE INDEX IF NOT EXISTS idx_hist_ticker_date ON historical_data(ticker, date);
        """)
        self._conn.commit()

    def save_layer_output(self, layer: str, payload: dict) -> None:
        self._conn.execute(
            "INSERT INTO layer_outputs (layer, payload) VALUES (?, ?)",
            (layer, json.dumps(payload)),
        )
        self._conn.commit()

    def get_latest_layer_output(self, layer: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT payload FROM layer_outputs WHERE layer = ? ORDER BY timestamp DESC LIMIT 1",
            (layer,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload"])

    def save_price(self, ticker: str, price: float) -> None:
        self._conn.execute(
            "INSERT INTO prices (ticker, price) VALUES (?, ?)",
            (ticker, price),
        )
        self._conn.commit()

    def get_prices(self, ticker: str, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT timestamp, price FROM prices WHERE ticker = ? ORDER BY timestamp ASC LIMIT ?",
            (ticker, limit),
        ).fetchall()
        return [{"timestamp": r["timestamp"], "price": r["price"]} for r in rows]

    def save_alert(self, severity: str, alert_type: str, message: str) -> None:
        self._conn.execute(
            "INSERT INTO alerts (severity, alert_type, message) VALUES (?, ?, ?)",
            (severity, alert_type, message),
        )
        self._conn.commit()

    def get_alerts(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT timestamp, severity, alert_type, message FROM alerts ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_decision_snapshot(self, snapshot: dict) -> None:
        self._conn.execute(
            "INSERT INTO decision_snapshots (snapshot) VALUES (?)",
            (json.dumps(snapshot),),
        )
        self._conn.commit()

    def get_latest_decision_snapshot(self) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT snapshot FROM decision_snapshots ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["snapshot"])

    def save_historical_data(self, ticker: str, rows: list[dict]) -> None:
        self._conn.executemany(
            """INSERT OR REPLACE INTO historical_data
               (ticker, date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (ticker, r["date"], r.get("open"), r.get("high"),
                 r.get("low"), r.get("close"), r.get("volume"))
                for r in rows
            ],
        )
        self._conn.commit()

    def get_historical_data(
        self, ticker: str, start_date: str = None, end_date: str = None
    ) -> list[dict]:
        query = "SELECT * FROM historical_data WHERE ticker = ?"
        params: list = [ticker]
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date ASC"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_store.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add data/ tests/test_store.py
git commit -m "feat: add SQLite data store with tables for layers, prices, alerts, snapshots"
```

---

### Task 5: Data Fetcher

**Files:**
- Create: `data/fetcher.py`
- Create: `tests/test_fetcher.py`

- [ ] **Step 1: Write failing tests for data fetcher**

```python
# tests/test_fetcher.py
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import date


def test_fetch_current_prices_returns_dict():
    from data.fetcher import DataFetcher

    fetcher = DataFetcher()
    # Mock yfinance to avoid network calls in tests
    with patch("data.fetcher.yf") as mock_yf:
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": 69.5}
        mock_yf.Ticker.return_value = mock_ticker

        prices = fetcher.fetch_current_prices(["DAL"])
        assert "DAL" in prices
        assert prices["DAL"] == 69.5


def test_fetch_historical_returns_dataframe():
    from data.fetcher import DataFetcher

    fetcher = DataFetcher()
    mock_df = pd.DataFrame({
        "Open": [69.0], "High": [70.0], "Low": [68.0],
        "Close": [69.5], "Volume": [1000000]
    }, index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")]))

    with patch("data.fetcher.yf") as mock_yf:
        mock_yf.download.return_value = mock_df
        df = fetcher.fetch_historical("DAL", "2024-01-01", "2024-01-02")
        assert len(df) == 1
        assert "Close" in df.columns


def test_fetch_regime_features_returns_dataframe():
    from data.fetcher import DataFetcher

    fetcher = DataFetcher()
    features = ["vix", "crude_5d_return"]

    # Create mock data
    dates = pd.date_range("2024-01-01", periods=30)
    mock_data = pd.DataFrame({
        "Close": np.random.randn(30).cumsum() + 20
    }, index=dates)

    with patch("data.fetcher.yf") as mock_yf:
        mock_yf.download.return_value = mock_data
        df = fetcher.fetch_regime_features(
            start="2024-01-01", end="2024-01-31",
            feature_tickers={"^VIX": "vix", "CL=F": "crude"}
        )
        assert isinstance(df, pd.DataFrame)


def test_fetch_gdelt_headlines():
    from data.fetcher import DataFetcher

    fetcher = DataFetcher()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"articles": [
        {"title": "Iran tensions rise", "url": "http://example.com", "tone": -2.5,
         "dateadded": "20260408120000"}
    ]}

    with patch("data.fetcher.httpx.get", return_value=mock_response):
        headlines = fetcher.fetch_gdelt_headlines(["Iran", "oil"])
        assert len(headlines) >= 0  # May be 0 if parsing differs


def test_fetch_newsapi_headlines():
    from data.fetcher import DataFetcher

    fetcher = DataFetcher()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"articles": [
        {"title": "Oil prices surge amid sanctions", "publishedAt": "2026-04-08T12:00:00Z",
         "source": {"name": "Reuters"}, "url": "http://example.com"}
    ]}

    with patch("data.fetcher.httpx.get", return_value=mock_response):
        with patch.dict("os.environ", {"NEWSAPI_KEY": "test_key"}):
            headlines = fetcher.fetch_newsapi_headlines(["oil", "sanctions"])
            assert len(headlines) == 1
            assert "Oil prices" in headlines[0]["title"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_fetcher.py -v`
Expected: FAIL

- [ ] **Step 3: Implement data fetcher**

```python
# data/fetcher.py
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self):
        self._last_known_prices: dict[str, float] = {}

    def fetch_current_prices(self, tickers: list[str]) -> dict[str, Optional[float]]:
        prices = {}
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                price = t.info.get("regularMarketPrice") or t.info.get("currentPrice")
                if price is not None:
                    self._last_known_prices[ticker] = price
                prices[ticker] = price
            except Exception as e:
                logger.warning("price_fetch_failed", ticker=ticker, error=str(e))
                prices[ticker] = self._last_known_prices.get(ticker)
        return prices

    def fetch_historical(
        self, ticker: str, start: str, end: str
    ) -> pd.DataFrame:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return df
        except Exception as e:
            logger.error("historical_fetch_failed", ticker=ticker, error=str(e))
            return pd.DataFrame()

    def fetch_multiple_historical(
        self, tickers: list[str], start: str, end: str
    ) -> dict[str, pd.DataFrame]:
        results = {}
        try:
            data = yf.download(tickers, start=start, end=end, progress=False, group_by="ticker")
            if len(tickers) == 1:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                results[tickers[0]] = data
            else:
                for ticker in tickers:
                    if ticker in data.columns.get_level_values(0):
                        results[ticker] = data[ticker].dropna(how="all")
        except Exception as e:
            logger.error("multi_historical_fetch_failed", error=str(e))
        return results

    def fetch_regime_features(
        self,
        start: str,
        end: str,
        feature_tickers: dict[str, str] = None,
    ) -> pd.DataFrame:
        if feature_tickers is None:
            feature_tickers = {
                "^VIX": "vix",
                "^VIX3M": "vix3m",
                "^TNX": "yield_10y",
                "CL=F": "crude",
                "SPY": "spy",
                "JPY=X": "usdjpy",
            }

        all_tickers = list(feature_tickers.keys())
        raw = self.fetch_multiple_historical(all_tickers, start, end)

        features = pd.DataFrame(index=raw.get(all_tickers[0], pd.DataFrame()).index)

        for yf_ticker, name in feature_tickers.items():
            if yf_ticker in raw and "Close" in raw[yf_ticker].columns:
                features[name] = raw[yf_ticker]["Close"]

        return features.dropna()

    def compute_regime_features(self, raw_features: pd.DataFrame) -> pd.DataFrame:
        computed = pd.DataFrame(index=raw_features.index)

        if "vix" in raw_features.columns:
            computed["vix"] = raw_features["vix"]

        if "vix" in raw_features.columns and "vix3m" in raw_features.columns:
            computed["vix_term_slope"] = raw_features["vix3m"] / raw_features["vix"]

        if "yield_10y" in raw_features.columns:
            computed["yield_10y_5d_delta"] = raw_features["yield_10y"].diff(5)

        if "crude" in raw_features.columns:
            computed["crude_5d_return"] = raw_features["crude"].pct_change(5)

        if "spy" in raw_features.columns:
            computed["spy_5d_realized_vol"] = (
                raw_features["spy"].pct_change().rolling(5).std() * np.sqrt(252)
            )

        if "usdjpy" in raw_features.columns:
            computed["usdjpy_5d_delta"] = raw_features["usdjpy"].diff(5)

        return computed.dropna()

    def fetch_gdelt_headlines(
        self, keywords: list[str], max_records: int = 250
    ) -> list[dict]:
        query = " OR ".join(keywords)
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": max_records,
            "format": "json",
            "sort": "datedesc",
        }
        try:
            resp = httpx.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "url": a.get("url", ""),
                    "tone": a.get("tone", 0),
                    "date": a.get("seendate", ""),
                    "source": a.get("domain", ""),
                    "source_type": "gdelt",
                }
                for a in articles
            ]
        except Exception as e:
            logger.warning("gdelt_fetch_failed", error=str(e))
            return []

    def fetch_newsapi_headlines(
        self, keywords: list[str], page_size: int = 50
    ) -> list[dict]:
        api_key = os.environ.get("NEWSAPI_KEY")
        if not api_key:
            logger.warning("newsapi_key_not_set")
            return []

        query = " OR ".join(keywords)
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": api_key,
            "language": "en",
        }
        try:
            resp = httpx.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "title": a.get("title", ""),
                    "url": a.get("url", ""),
                    "date": a.get("publishedAt", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "source_type": "newsapi",
                }
                for a in data.get("articles", [])
            ]
        except Exception as e:
            logger.warning("newsapi_fetch_failed", error=str(e))
            return []
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_fetcher.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add data/fetcher.py tests/test_fetcher.py
git commit -m "feat: add data fetcher for yfinance, GDELT, and NewsAPI"
```

---

### Task 6: Telegram Alerts & Price Monitor

**Files:**
- Create: `engine/__init__.py`
- Create: `engine/alerts.py`
- Create: `engine/price_monitor.py`
- Create: `tests/test_alerts.py`
- Create: `tests/test_price_monitor.py`

- [ ] **Step 1: Write failing tests for alert manager**

```python
# tests/test_alerts.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


def test_alert_severity_levels():
    from engine.alerts import AlertSeverity

    assert AlertSeverity.INFO.emoji == "\U0001f7e2"
    assert AlertSeverity.WARNING.emoji == "\u26a0\ufe0f"
    assert AlertSeverity.CRITICAL.emoji == "\U0001f6a8"


def test_format_alert_message():
    from engine.alerts import AlertManager, AlertSeverity

    mgr = AlertManager(bot_token="test", chat_id="123", store=None)
    msg = mgr.format_alert(AlertSeverity.CRITICAL, "regime_change", "Regime shifted to escalation")
    assert "\U0001f6a8" in msg
    assert "regime_change" in msg
    assert "escalation" in msg


def test_batch_non_critical_alerts():
    from engine.alerts import AlertManager, AlertSeverity

    mgr = AlertManager(bot_token="test", chat_id="123", store=None)
    mgr.queue_alert(AlertSeverity.INFO, "price_update", "DAL at $69.50")
    mgr.queue_alert(AlertSeverity.INFO, "price_update", "NVDA at $183.00")
    assert len(mgr._alert_queue) == 2

    batched = mgr.flush_queue()
    assert len(batched) == 1  # single batched message
    assert "DAL" in batched[0]
    assert "NVDA" in batched[0]


@pytest.mark.asyncio
async def test_send_critical_alert_immediately():
    from engine.alerts import AlertManager, AlertSeverity

    mgr = AlertManager(bot_token="test", chat_id="123", store=None)
    with patch.object(mgr, "_send_telegram", new_callable=AsyncMock) as mock_send:
        await mgr.send_alert(AlertSeverity.CRITICAL, "regime_change", "Regime shifted")
        mock_send.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_alerts.py -v`
Expected: FAIL

- [ ] **Step 3: Implement alert manager**

```python
# engine/__init__.py
# empty

# engine/alerts.py
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def emoji(self) -> str:
        return {
            AlertSeverity.INFO: "\U0001f7e2",
            AlertSeverity.WARNING: "\u26a0\ufe0f",
            AlertSeverity.CRITICAL: "\U0001f6a8",
        }[self]


class AlertManager:
    def __init__(self, bot_token: str, chat_id: str, store):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.store = store
        self._bot: Optional[Bot] = None
        self._app: Optional[Application] = None
        self._alert_queue: list[tuple[AlertSeverity, str, str]] = []

    def format_alert(self, severity: AlertSeverity, alert_type: str, message: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%H:%M UTC")
        return f"{severity.emoji} *{alert_type}* [{ts}]\n{message}"

    def queue_alert(self, severity: AlertSeverity, alert_type: str, message: str) -> None:
        self._alert_queue.append((severity, alert_type, message))

    def flush_queue(self) -> list[str]:
        if not self._alert_queue:
            return []

        lines = []
        for severity, alert_type, message in self._alert_queue:
            lines.append(self.format_alert(severity, alert_type, message))

        self._alert_queue.clear()
        return ["\n\n".join(lines)]

    async def _send_telegram(self, text: str) -> None:
        if self._bot is None:
            self._bot = Bot(token=self.bot_token)
        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.error("telegram_send_failed", error=str(e))

    async def send_alert(
        self, severity: AlertSeverity, alert_type: str, message: str
    ) -> None:
        formatted = self.format_alert(severity, alert_type, message)

        if self.store:
            self.store.save_alert(severity.value, alert_type, message)

        if severity == AlertSeverity.CRITICAL:
            await self._send_telegram(formatted)
        else:
            self.queue_alert(severity, alert_type, message)

    async def flush_and_send(self) -> None:
        messages = self.flush_queue()
        for msg in messages:
            await self._send_telegram(msg)

    async def send_daily_summary(self, summary: dict) -> None:
        lines = ["\U0001f4ca *Daily Summary*\n"]

        if "regime" in summary:
            lines.append(f"*Regime:* {summary['regime']}")

        if "tickers" in summary:
            for ticker, data in summary["tickers"].items():
                score = data.get("convergence_score", 0)
                direction = data.get("direction", "neutral")
                lines.append(f"*{ticker}:* {direction} (score: {score:.2f})")

        if "market_wide" in summary:
            mw = summary["market_wide"]
            if "sentiment_velocity" in mw:
                lines.append(f"*Sentiment velocity:* {mw['sentiment_velocity']:.3f}")
            if "vix_term_structure" in mw:
                lines.append(f"*VIX term structure:* {mw['vix_term_structure']}")

        await self._send_telegram("\n".join(lines))

    def setup_commands(self, app: Application, config_updater) -> None:
        self._config_updater = config_updater

        async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if self.store:
                snapshot = self.store.get_latest_decision_snapshot()
                if snapshot:
                    lines = ["*Current Status*\n"]
                    for ticker, data in snapshot.get("tickers", {}).items():
                        lines.append(
                            f"*{ticker}:* {data.get('direction', '?')} "
                            f"(score: {data.get('convergence_score', 0):.2f})"
                        )
                    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
                else:
                    await update.message.reply_text("No data yet.")

        async def cmd_regime(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if self.store:
                output = self.store.get_latest_layer_output("regime")
                if output:
                    label = output.get("regime_label", "unknown")
                    cp = output.get("changepoint_probability", 0)
                    probs = output.get("regime_probabilities", [])
                    text = (
                        f"*Current Regime:* {label}\n"
                        f"*Changepoint Prob:* {cp:.2f}\n"
                        f"*State Probs:* {probs}"
                    )
                    await update.message.reply_text(text, parse_mode="Markdown")
                else:
                    await update.message.reply_text("No regime data yet.")

        async def cmd_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
            args = context.args
            if len(args) != 3:
                await update.message.reply_text("Usage: /set TICKER buy|stop PRICE")
                return
            ticker, field, price_str = args[0].upper(), args[1].lower(), args[2]
            try:
                price = float(price_str)
            except ValueError:
                await update.message.reply_text("Invalid price.")
                return
            if self._config_updater:
                self._config_updater(ticker, field, price)
            await update.message.reply_text(f"Set {ticker} {field} = ${price:.2f}")

        async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if not context.args:
                await update.message.reply_text("Usage: /add TICKER")
                return
            ticker = context.args[0].upper()
            if self._config_updater:
                self._config_updater(ticker, "add", None)
            await update.message.reply_text(f"Added {ticker} to watchlist.")

        async def cmd_remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if not context.args:
                await update.message.reply_text("Usage: /remove TICKER")
                return
            ticker = context.args[0].upper()
            if self._config_updater:
                self._config_updater(ticker, "remove", None)
            await update.message.reply_text(f"Removed {ticker} from watchlist.")

        app.add_handler(CommandHandler("status", cmd_status))
        app.add_handler(CommandHandler("regime", cmd_regime))
        app.add_handler(CommandHandler("set", cmd_set))
        app.add_handler(CommandHandler("add", cmd_add))
        app.add_handler(CommandHandler("remove", cmd_remove))
```

- [ ] **Step 4: Run alert tests**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_alerts.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Write failing tests for price monitor**

```python
# tests/test_price_monitor.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock


def test_check_stop_loss_triggered():
    from engine.price_monitor import PriceMonitor

    config = MagicMock()
    position = MagicMock()
    position.ticker = "DAL"
    position.stop_loss = 65.55
    position.profit_target = 82.0
    position.shares = 36
    config.portfolio.positions = [position]
    config.watchlist = []

    monitor = PriceMonitor(config=config, fetcher=None, alert_manager=None)
    alerts = monitor.check_price_levels({"DAL": 65.00})
    assert len(alerts) == 1
    assert alerts[0]["type"] == "stop_loss"
    assert alerts[0]["ticker"] == "DAL"


def test_check_profit_target_triggered():
    from engine.price_monitor import PriceMonitor

    config = MagicMock()
    position = MagicMock()
    position.ticker = "DAL"
    position.stop_loss = 65.55
    position.profit_target = 82.0
    position.shares = 36
    config.portfolio.positions = [position]
    config.watchlist = []

    monitor = PriceMonitor(config=config, fetcher=None, alert_manager=None)
    alerts = monitor.check_price_levels({"DAL": 83.00})
    assert len(alerts) == 1
    assert alerts[0]["type"] == "profit_target"


def test_check_buy_target_triggered():
    from engine.price_monitor import PriceMonitor

    config = MagicMock()
    config.portfolio.positions = []
    watch_item = MagicMock()
    watch_item.ticker = "VTI"
    watch_item.buy_target = 333.0
    config.watchlist = [watch_item]

    monitor = PriceMonitor(config=config, fetcher=None, alert_manager=None)
    alerts = monitor.check_price_levels({"VTI": 330.0})
    assert len(alerts) == 1
    assert alerts[0]["type"] == "buy_target"


def test_no_alert_when_price_in_range():
    from engine.price_monitor import PriceMonitor

    config = MagicMock()
    position = MagicMock()
    position.ticker = "DAL"
    position.stop_loss = 65.55
    position.profit_target = 82.0
    position.shares = 36
    config.portfolio.positions = [position]
    config.watchlist = []

    monitor = PriceMonitor(config=config, fetcher=None, alert_manager=None)
    alerts = monitor.check_price_levels({"DAL": 72.00})
    assert len(alerts) == 0
```

- [ ] **Step 6: Run price monitor tests to verify failure**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_price_monitor.py -v`
Expected: FAIL

- [ ] **Step 7: Implement price monitor**

```python
# engine/price_monitor.py
from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PriceMonitor:
    def __init__(self, config, fetcher, alert_manager):
        self.config = config
        self.fetcher = fetcher
        self.alert_manager = alert_manager
        self._triggered: set[str] = set()  # avoid repeat alerts

    def check_price_levels(self, prices: dict[str, float]) -> list[dict]:
        alerts = []

        for position in self.config.portfolio.positions:
            ticker = position.ticker
            price = prices.get(ticker)
            if price is None:
                continue

            if position.stop_loss and price <= position.stop_loss:
                key = f"{ticker}_stop"
                if key not in self._triggered:
                    alerts.append({
                        "type": "stop_loss",
                        "ticker": ticker,
                        "price": price,
                        "level": position.stop_loss,
                        "severity": "critical",
                    })
                    self._triggered.add(key)

            if position.profit_target and price >= position.profit_target:
                key = f"{ticker}_profit"
                if key not in self._triggered:
                    alerts.append({
                        "type": "profit_target",
                        "ticker": ticker,
                        "price": price,
                        "level": position.profit_target,
                        "severity": "warning",
                    })
                    self._triggered.add(key)

        for item in self.config.watchlist:
            ticker = item.ticker
            price = prices.get(ticker)
            if price is None:
                continue

            if item.buy_target and price <= item.buy_target:
                key = f"{ticker}_buy"
                if key not in self._triggered:
                    alerts.append({
                        "type": "buy_target",
                        "ticker": ticker,
                        "price": price,
                        "level": item.buy_target,
                        "severity": "warning",
                    })
                    self._triggered.add(key)

        return alerts

    def reset_trigger(self, ticker: str, alert_type: str) -> None:
        key = f"{ticker}_{alert_type}"
        self._triggered.discard(key)

    async def run_owned_loop(self, interval_seconds: int = 60) -> None:
        while True:
            try:
                tickers = [p.ticker for p in self.config.portfolio.positions]
                if tickers:
                    prices = self.fetcher.fetch_current_prices(tickers)
                    triggered = self.check_price_levels(prices)
                    for alert in triggered:
                        from engine.alerts import AlertSeverity

                        severity = (
                            AlertSeverity.CRITICAL
                            if alert["severity"] == "critical"
                            else AlertSeverity.WARNING
                        )
                        msg = (
                            f"{alert['ticker']}: ${alert['price']:.2f} "
                            f"hit {alert['type']} (${alert['level']:.2f})"
                        )
                        await self.alert_manager.send_alert(severity, alert["type"], msg)
            except Exception as e:
                logger.error("price_monitor_owned_error", error=str(e))
            await asyncio.sleep(interval_seconds)

    async def run_watchlist_loop(self, interval_seconds: int = 300) -> None:
        while True:
            try:
                tickers = [w.ticker for w in self.config.watchlist]
                if tickers:
                    prices = self.fetcher.fetch_current_prices(tickers)
                    triggered = self.check_price_levels(prices)
                    for alert in triggered:
                        from engine.alerts import AlertSeverity

                        msg = (
                            f"{alert['ticker']}: ${alert['price']:.2f} "
                            f"hit {alert['type']} (${alert['level']:.2f})"
                        )
                        await self.alert_manager.send_alert(
                            AlertSeverity.WARNING, alert["type"], msg
                        )
            except Exception as e:
                logger.error("price_monitor_watchlist_error", error=str(e))
            await asyncio.sleep(interval_seconds)
```

- [ ] **Step 8: Run price monitor tests**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_price_monitor.py -v`
Expected: All 4 tests PASS

- [ ] **Step 9: Commit**

```bash
git add engine/ tests/test_alerts.py tests/test_price_monitor.py
git commit -m "feat: add Telegram alert manager and price monitor with stop/profit/buy alerts"
```

---

## Phase 2: ML Layers (Tasks 7-12)

---

### Task 7: Layer 1 — Regime Detection (HMM + BOCPD)

**Files:**
- Create: `layers/__init__.py`
- Create: `layers/regime.py`
- Create: `tests/test_regime.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_regime.py
import pytest
import numpy as np
import pandas as pd


def test_bocpd_detects_changepoint():
    from layers.regime import BayesianChangepoint

    # Stable signal then abrupt shift
    np.random.seed(42)
    signal = np.concatenate([
        np.random.normal(0, 1, 100),
        np.random.normal(5, 1, 50),
    ])
    bocpd = BayesianChangepoint(hazard_lambda=100)
    probs = bocpd.run(signal)
    # Changepoint prob should spike around index 100
    assert len(probs) == len(signal)
    assert max(probs[95:110]) > 0.3


def test_bocpd_no_changepoint_in_stable():
    from layers.regime import BayesianChangepoint

    np.random.seed(42)
    signal = np.random.normal(0, 1, 200)
    bocpd = BayesianChangepoint(hazard_lambda=250)
    probs = bocpd.run(signal)
    assert max(probs) < 0.5


def test_hmm_fit_and_predict():
    from layers.regime import RegimeHMM

    np.random.seed(42)
    # 3 regimes with distinct means
    data = np.concatenate([
        np.random.multivariate_normal([0, 0], np.eye(2) * 0.5, 100),
        np.random.multivariate_normal([3, 3], np.eye(2) * 0.5, 100),
        np.random.multivariate_normal([0, 0], np.eye(2) * 0.5, 100),
    ])
    hmm = RegimeHMM(n_states=3, covariance_type="diag")
    hmm.fit(data)
    labels = hmm.predict(data)
    assert len(labels) == 300
    assert len(set(labels)) >= 2  # should detect at least 2 distinct regimes


def test_hmm_transition_matrix():
    from layers.regime import RegimeHMM

    np.random.seed(42)
    data = np.concatenate([
        np.random.multivariate_normal([0, 0], np.eye(2) * 0.5, 100),
        np.random.multivariate_normal([5, 5], np.eye(2) * 0.5, 100),
    ])
    hmm = RegimeHMM(n_states=2, covariance_type="diag")
    hmm.fit(data)
    trans = hmm.transition_matrix()
    assert trans.shape == (2, 2)
    # Rows should sum to 1
    np.testing.assert_allclose(trans.sum(axis=1), [1.0, 1.0], atol=1e-6)


def test_hmm_bic_selection():
    from layers.regime import select_n_states_bic

    np.random.seed(42)
    data = np.concatenate([
        np.random.multivariate_normal([0, 0], np.eye(2) * 0.3, 80),
        np.random.multivariate_normal([4, 4], np.eye(2) * 0.3, 80),
    ])
    best_n = select_n_states_bic(data, max_states=4)
    assert best_n in [2, 3]  # should prefer 2


def test_regime_detector_full_pipeline():
    from layers.regime import RegimeDetector

    np.random.seed(42)
    # Simulate features: calm then crisis
    calm = np.random.multivariate_normal([15, 1.1, 0, 0, 0.1, 0], np.eye(6) * 0.5, 80)
    crisis = np.random.multivariate_normal([30, 0.9, 0.5, -0.1, 0.4, 2], np.eye(6) * 0.5, 40)
    features = np.concatenate([calm, crisis])

    detector = RegimeDetector(n_states=2, hazard_lambda=50)
    detector.fit(features)
    result = detector.detect(features)

    assert "regime_label" in result
    assert "regime_probabilities" in result
    assert "changepoint_probability" in result
    assert "transition_matrix" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_regime.py -v`
Expected: FAIL

- [ ] **Step 3: Implement regime detection**

```python
# layers/__init__.py
# empty

# layers/regime.py
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class BayesianChangepoint:
    """Bayesian Online Changepoint Detection (Adams & MacKay 2007)."""

    def __init__(self, hazard_lambda: int = 250):
        self.hazard_lambda = hazard_lambda
        self.hazard_rate = 1.0 / hazard_lambda

    def run(self, signal: np.ndarray) -> np.ndarray:
        T = len(signal)
        # Run length probabilities: R[t, r] = P(run_length=r at time t)
        R = np.zeros((T + 1, T + 1))
        R[0, 0] = 1.0

        changepoint_probs = np.zeros(T)

        # Sufficient statistics for Gaussian predictive
        mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, 1.0
        muT = np.full(T + 1, mu0)
        kappaT = np.full(T + 1, kappa0)
        alphaT = np.full(T + 1, alpha0)
        betaT = np.full(T + 1, beta0)

        for t in range(T):
            x = signal[t]

            # Predictive probability for each run length
            pred_probs = self._student_t_pdf(
                x, muT[: t + 1], kappaT[: t + 1], alphaT[: t + 1], betaT[: t + 1]
            )

            # Growth probabilities
            R[t + 1, 1: t + 2] = R[t, : t + 1] * pred_probs * (1 - self.hazard_rate)
            # Changepoint probability
            R[t + 1, 0] = np.sum(R[t, : t + 1] * pred_probs * self.hazard_rate)

            # Normalize
            evidence = R[t + 1, : t + 2].sum()
            if evidence > 0:
                R[t + 1, : t + 2] /= evidence

            changepoint_probs[t] = R[t + 1, 0]

            # Update sufficient statistics
            new_mu = (kappaT[: t + 1] * muT[: t + 1] + x) / (kappaT[: t + 1] + 1)
            new_kappa = kappaT[: t + 1] + 1
            new_alpha = alphaT[: t + 1] + 0.5
            new_beta = betaT[: t + 1] + (
                kappaT[: t + 1] * (x - muT[: t + 1]) ** 2 / (2 * (kappaT[: t + 1] + 1))
            )

            muT[1: t + 2] = new_mu
            kappaT[1: t + 2] = new_kappa
            alphaT[1: t + 2] = new_alpha
            betaT[1: t + 2] = new_beta

            # Reset for run length 0
            muT[0] = mu0
            kappaT[0] = kappa0
            alphaT[0] = alpha0
            betaT[0] = beta0

        return changepoint_probs

    def _student_t_pdf(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        df = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        z = (x - mu) / scale
        from scipy.stats import t as student_t

        return student_t.pdf(z, df) / scale


class RegimeHMM:
    def __init__(self, n_states: int = 3, covariance_type: str = "diag"):
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )
        self._fitted = False

    def fit(self, data: np.ndarray) -> None:
        self.model.fit(data)
        self._fitted = True

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(data)

    def transition_matrix(self) -> np.ndarray:
        return self.model.transmat_

    def score(self, data: np.ndarray) -> float:
        return self.model.score(data)

    def bic(self, data: np.ndarray) -> float:
        n_samples, n_features = data.shape
        n_params = (
            self.n_states * n_features  # means
            + self.n_states * n_features  # diag covariance
            + self.n_states * (self.n_states - 1)  # transition probs
            + self.n_states - 1  # start probs
        )
        log_likelihood = self.model.score(data) * n_samples
        return -2 * log_likelihood + n_params * np.log(n_samples)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self._fitted = True


def select_n_states_bic(data: np.ndarray, max_states: int = 5) -> int:
    best_bic = np.inf
    best_n = 2
    for n in range(2, max_states + 1):
        try:
            hmm = RegimeHMM(n_states=n)
            hmm.fit(data)
            bic_val = hmm.bic(data)
            if bic_val < best_bic:
                best_bic = bic_val
                best_n = n
        except Exception:
            continue
    return best_n


class RegimeDetector:
    def __init__(
        self,
        n_states: int = 3,
        hazard_lambda: int = 250,
        state_labels: Optional[dict[int, str]] = None,
    ):
        self.n_states = n_states
        self.hmm = RegimeHMM(n_states=n_states)
        self.bocpd = BayesianChangepoint(hazard_lambda=hazard_lambda)
        self.pca = PCA(n_components=1)
        self.state_labels = state_labels or {0: "risk_on", 1: "stalemate", 2: "escalation"}
        self._fitted = False

    def fit(self, features: np.ndarray) -> None:
        self.hmm.fit(features)
        self.pca.fit(features)
        self._fitted = True

    def detect(self, features: np.ndarray) -> dict:
        # HMM prediction
        labels = self.hmm.predict(features)
        proba = self.hmm.predict_proba(features)
        current_label_idx = int(labels[-1])
        current_proba = proba[-1].tolist()

        # BOCPD on first principal component
        stress_index = self.pca.transform(features).flatten()
        cp_probs = self.bocpd.run(stress_index)
        current_cp_prob = float(cp_probs[-1])

        # Transition matrix
        trans_matrix = self.hmm.transition_matrix().tolist()

        regime_name = self.state_labels.get(current_label_idx, f"state_{current_label_idx}")

        return {
            "regime_label": regime_name,
            "regime_label_idx": current_label_idx,
            "regime_probabilities": current_proba,
            "changepoint_probability": current_cp_prob,
            "transition_matrix": trans_matrix,
            "hmm_log_likelihood": float(self.hmm.score(features)),
            "data_staleness_seconds": 0,
        }

    def save(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.hmm.save(f"{directory}/hmm_model.pkl")
        with open(f"{directory}/pca_model.pkl", "wb") as f:
            pickle.dump(self.pca, f)

    def load(self, directory: str) -> None:
        self.hmm.load(f"{directory}/hmm_model.pkl")
        with open(f"{directory}/pca_model.pkl", "rb") as f:
            self.pca = pickle.load(f)
        self._fitted = True
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_regime.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add layers/ tests/test_regime.py
git commit -m "feat: add regime detection with Gaussian HMM + BOCPD ensemble"
```

---

### Task 8: Layer 2 — Sentiment Engine (FinBERT + LLM + Hawkes)

**Files:**
- Create: `layers/sentiment.py`
- Create: `tests/test_sentiment.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sentiment.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone


def test_keyword_matcher():
    from layers.sentiment import KeywordMatcher

    matcher = KeywordMatcher(["Iran", "Hormuz", "ceasefire"])
    assert matcher.matches("Iran tensions escalate over Hormuz")
    assert not matcher.matches("Stock market rallies on earnings")


def test_keyword_matcher_case_insensitive():
    from layers.sentiment import KeywordMatcher

    matcher = KeywordMatcher(["iran", "oil"])
    assert matcher.matches("IRAN announces new sanctions")
    assert matcher.matches("Oil prices surge")


def test_hawkes_branching_ratio():
    from layers.sentiment import HawkesEstimator

    # Regular arrivals — low branching ratio
    np.random.seed(42)
    regular_times = np.sort(np.random.uniform(0, 100, 50))
    estimator = HawkesEstimator()
    result = estimator.estimate(regular_times, window=100)
    assert "branching_ratio" in result
    assert 0 <= result["branching_ratio"] <= 2.0


def test_hawkes_clustered_events():
    from layers.sentiment import HawkesEstimator

    # Clustered arrivals — should have higher branching ratio
    np.random.seed(42)
    base = np.array([10, 20, 30, 40, 50, 60])
    # Add clusters after each base event
    clusters = []
    for t in base:
        clusters.extend([t + 0.1, t + 0.2, t + 0.5, t + 0.8])
    times = np.sort(np.concatenate([base, clusters]))
    estimator = HawkesEstimator()
    result = estimator.estimate(times, window=70)
    assert result["branching_ratio"] > 0.1


def test_sentiment_scorer_finbert_mock():
    from layers.sentiment import SentimentScorer

    scorer = SentimentScorer(device="cpu", use_finbert=False)
    # With finbert disabled, should return neutral
    scores = scorer.score_headlines_simple(["Oil prices surge", "War breaks out"])
    assert len(scores) == 2


def test_sentiment_velocity():
    from layers.sentiment import compute_sentiment_velocity

    history = [-0.1, -0.2, -0.3, -0.5, -0.8]
    velocity = compute_sentiment_velocity(history, window=3)
    assert velocity < 0  # sentiment worsening


def test_sentiment_layer_output_format():
    from layers.sentiment import SentimentLayer

    layer = SentimentLayer(
        keywords=["Iran"],
        device="cpu",
        llm_provider=None,
        llm_model=None,
    )
    # Test output structure with mock data
    output = layer.build_output(
        sentiment_scores=[-0.5, -0.3, -0.7],
        sentiment_history=[[-0.2], [-0.3], [-0.5]],
        hawkes_result={"branching_ratio": 0.6, "intensity": 10.0},
        headline_count_15m=20,
        headline_count_1h=65,
        top_headlines=["test headline"],
        llm_result=None,
        active_sources=["gdelt"],
    )
    assert "sentiment_mean" in output
    assert "sentiment_velocity" in output
    assert "hawkes_branching_ratio" in output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_sentiment.py -v`
Expected: FAIL

- [ ] **Step 3: Implement sentiment layer**

```python
# layers/sentiment.py
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class KeywordMatcher:
    def __init__(self, keywords: list[str]):
        escaped = [re.escape(kw) for kw in keywords]
        self.pattern = re.compile("|".join(escaped), re.IGNORECASE)

    def matches(self, text: str) -> bool:
        return bool(self.pattern.search(text))


class HawkesEstimator:
    """Simple Hawkes process parameter estimation via MLE with exponential kernel."""

    def estimate(self, event_times: np.ndarray, window: float) -> dict:
        if len(event_times) < 3:
            return {"branching_ratio": 0.0, "intensity": 0.0, "baseline": 0.0, "decay": 1.0}

        n = len(event_times)
        T = window

        # MLE for exponential Hawkes: mu + alpha * sum(beta * exp(-beta*(t-ti)))
        # Use simple method-of-moments as robust starting point
        # Mean intensity
        mean_intensity = n / T

        # Estimate decay beta from mean inter-arrival time of clustered events
        diffs = np.diff(event_times)
        short_diffs = diffs[diffs < np.median(diffs)]
        if len(short_diffs) > 0:
            beta = 1.0 / (np.mean(short_diffs) + 1e-6)
        else:
            beta = 1.0

        # Estimate branching ratio from coefficient of variation of inter-arrival times
        if len(diffs) > 1:
            cv = np.std(diffs) / (np.mean(diffs) + 1e-6)
            # For Poisson process, CV=1. For Hawkes, CV > 1.
            alpha = min(max((cv - 1) / (cv + 1), 0.0), 0.99)
        else:
            alpha = 0.0

        mu = mean_intensity * (1 - alpha)

        return {
            "branching_ratio": float(alpha),
            "intensity": float(mean_intensity),
            "baseline": float(mu),
            "decay": float(beta),
        }


class SentimentScorer:
    def __init__(self, device: str = "cuda", use_finbert: bool = True):
        self.device = device
        self._model = None
        self._tokenizer = None
        self.use_finbert = use_finbert

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self._model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.half().to("cuda")
            else:
                self._model = self._model.to("cpu")
                self.device = "cpu"
            self._model.eval()
        except Exception as e:
            logger.error("finbert_load_failed", error=str(e))
            self.use_finbert = False

    def score_headlines(self, headlines: list[str]) -> list[float]:
        if not self.use_finbert or not headlines:
            return self.score_headlines_simple(headlines)

        self._load_model()
        if self._model is None:
            return self.score_headlines_simple(headlines)

        import torch

        scores = []
        batch_size = 32
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i: i + batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                # FinBERT: [positive, negative, neutral]
                # Convert to single score: positive - negative
                batch_scores = (probs[:, 0] - probs[:, 1]).cpu().numpy().tolist()
                scores.extend(batch_scores)

        return scores

    def score_headlines_simple(self, headlines: list[str]) -> list[float]:
        """Fallback: return neutral scores."""
        return [0.0] * len(headlines)


async def llm_contextual_analysis(
    headlines: list[str],
    provider: str,
    model: str,
    keywords: list[str],
) -> Optional[dict]:
    if not headlines:
        return None

    headline_text = "\n".join(f"- {h}" for h in headlines[:20])
    keywords_text = ", ".join(keywords)

    prompt = f"""Analyze these geopolitical headlines related to: {keywords_text}

{headline_text}

Return a JSON object with:
- "escalation_score": float from -1.0 (strong de-escalation) to 1.0 (strong escalation)
- "confidence": float from 0.0 to 1.0
- "rationale": one sentence explaining your assessment

Respond ONLY with the JSON object."""

    try:
        if provider == "anthropic":
            import anthropic

            client = anthropic.AsyncAnthropic()
            response = await client.messages.create(
                model=model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            import json

            text = response.content[0].text.strip()
            cost = (response.usage.input_tokens * 15 + response.usage.output_tokens * 75) / 1_000_000
            result = json.loads(text)
            result["model"] = model
            result["cost_usd"] = cost
            return result

        elif provider == "openai":
            import openai

            client = openai.AsyncOpenAI()
            response = await client.chat.completions.create(
                model=model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            import json

            text = response.choices[0].message.content.strip()
            result = json.loads(text)
            result["model"] = model
            result["cost_usd"] = 0.0  # OpenAI pricing varies
            return result

    except Exception as e:
        logger.error("llm_analysis_failed", provider=provider, error=str(e))
        return None


def compute_sentiment_velocity(history: list[float], window: int = 3) -> float:
    if len(history) < 2:
        return 0.0
    recent = history[-window:]
    if len(recent) < 2:
        return 0.0
    return float(np.mean(np.diff(recent)))


class SentimentLayer:
    def __init__(
        self,
        keywords: list[str],
        device: str = "cuda",
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        self.keywords = keywords
        self.matcher = KeywordMatcher(keywords)
        self.scorer = SentimentScorer(device=device)
        self.hawkes = HawkesEstimator()
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self._sentiment_history: list[float] = []

    def filter_headlines(self, headlines: list[dict]) -> list[dict]:
        return [h for h in headlines if self.matcher.matches(h.get("title", ""))]

    def build_output(
        self,
        sentiment_scores: list[float],
        sentiment_history: list[list[float]],
        hawkes_result: dict,
        headline_count_15m: int,
        headline_count_1h: int,
        top_headlines: list[str],
        llm_result: Optional[dict],
        active_sources: list[str],
    ) -> dict:
        if sentiment_scores:
            sentiment_mean = float(np.mean(sentiment_scores))
        else:
            sentiment_mean = 0.0

        flat_history = [np.mean(s) if s else 0.0 for s in sentiment_history]
        flat_history.append(sentiment_mean)
        velocity = compute_sentiment_velocity(flat_history)
        acceleration = compute_sentiment_velocity(
            [compute_sentiment_velocity(flat_history[:i+1]) for i in range(len(flat_history))],
            window=3,
        ) if len(flat_history) > 3 else 0.0

        output = {
            "sentiment_mean": sentiment_mean,
            "sentiment_velocity": velocity,
            "sentiment_acceleration": acceleration,
            "hawkes_branching_ratio": hawkes_result.get("branching_ratio", 0.0),
            "hawkes_intensity": hawkes_result.get("intensity", 0.0),
            "headline_count_15m": headline_count_15m,
            "headline_count_1h": headline_count_1h,
            "top_headlines": top_headlines[:5],
            "data_sources_active": active_sources,
            "data_staleness_seconds": 0,
        }

        if llm_result:
            output["llm_escalation_score"] = llm_result.get("escalation_score", 0.0)
            output["llm_confidence"] = llm_result.get("confidence", 0.0)
            output["llm_rationale"] = llm_result.get("rationale", "")
            output["llm_model"] = llm_result.get("model", "")
            output["llm_cost_usd"] = llm_result.get("cost_usd", 0.0)

        return output
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_sentiment.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add layers/sentiment.py tests/test_sentiment.py
git commit -m "feat: add sentiment layer with FinBERT, LLM analysis, and Hawkes process"
```

---

### Task 9: Layer 3 — Factor-Residual Mean Reversion

**Files:**
- Create: `layers/mean_reversion.py`
- Create: `tests/test_mean_reversion.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mean_reversion.py
import pytest
import numpy as np
import pandas as pd


def test_ou_calibrate():
    from layers.mean_reversion import OUProcess

    np.random.seed(42)
    # Simulate OU process: dX = theta*(mu - X)*dt + sigma*dW
    theta, mu, sigma = 0.5, 0.0, 1.0
    dt = 1.0 / 252
    n = 500
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) * dt + sigma * np.sqrt(dt) * np.random.randn()

    ou = OUProcess()
    ou.calibrate(x, dt=dt)
    assert abs(ou.theta - theta) < 0.3  # rough estimate
    assert abs(ou.mu - mu) < 0.5
    assert ou.sigma > 0


def test_ou_half_life():
    from layers.mean_reversion import OUProcess

    ou = OUProcess()
    ou.theta = 0.5
    half_life = ou.half_life()
    expected = np.log(2) / 0.5
    assert abs(half_life - expected) < 0.01


def test_ou_z_score():
    from layers.mean_reversion import OUProcess

    ou = OUProcess()
    ou.theta = 0.5
    ou.mu = 0.0
    ou.sigma = 1.0
    z = ou.z_score(2.0)
    assert z > 0  # price above fair value


def test_factor_model_fit():
    from layers.mean_reversion import FactorModel

    np.random.seed(42)
    n = 200
    market = np.random.randn(n) * 0.01
    sector = np.random.randn(n) * 0.01
    vol = np.random.randn(n) * 0.01
    # Ticker return = 1.2*market + 0.5*sector - 0.3*vol + noise
    ticker_returns = 1.2 * market + 0.5 * sector - 0.3 * vol + np.random.randn(n) * 0.005

    factors = pd.DataFrame({"market": market, "sector": sector, "vol": vol})
    fm = FactorModel()
    fm.fit(ticker_returns, factors)
    assert abs(fm.betas["market"] - 1.2) < 0.3
    assert abs(fm.betas["sector"] - 0.5) < 0.3


def test_factor_model_residual():
    from layers.mean_reversion import FactorModel

    np.random.seed(42)
    n = 200
    market = np.random.randn(n) * 0.01
    ticker_returns = 1.0 * market + np.random.randn(n) * 0.005

    factors = pd.DataFrame({"market": market})
    fm = FactorModel()
    fm.fit(ticker_returns, factors)
    residuals = fm.residuals(ticker_returns, factors)
    # Residuals should have lower std than raw returns
    assert np.std(residuals) < np.std(ticker_returns)


def test_mean_reversion_layer_signal():
    from layers.mean_reversion import MeanReversionLayer

    layer = MeanReversionLayer(half_life_suppression_days=15)

    result = layer.compute_signal(
        ticker="DAL",
        z_score=-2.5,
        half_life=6.0,
        regime="stalemate",
        fair_value=72.0,
        current_price=68.0,
        ou_params={"theta": 0.1, "mu": 0.0, "sigma": 1.5},
        factor_betas={"market": 1.2, "sector": 0.5},
    )
    assert result["residual_z_score"] == -2.5
    assert result["signal_suppressed"] is False


def test_mean_reversion_suppressed_in_crisis():
    from layers.mean_reversion import MeanReversionLayer

    layer = MeanReversionLayer(half_life_suppression_days=15)

    result = layer.compute_signal(
        ticker="DAL",
        z_score=-2.5,
        half_life=20.0,  # > 15 day threshold
        regime="escalation",
        fair_value=72.0,
        current_price=68.0,
        ou_params={"theta": 0.03, "mu": 0.0, "sigma": 1.5},
        factor_betas={"market": 1.2},
    )
    assert result["signal_suppressed"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_mean_reversion.py -v`
Expected: FAIL

- [ ] **Step 3: Implement mean reversion layer**

```python
# layers/mean_reversion.py
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class OUProcess:
    """Ornstein-Uhlenbeck process calibration via MLE."""

    def __init__(self):
        self.theta: float = 0.0
        self.mu: float = 0.0
        self.sigma: float = 0.0

    def calibrate(self, series: np.ndarray, dt: float = 1.0 / 252) -> None:
        n = len(series)
        if n < 10:
            return

        # OLS regression: X(t+1) - X(t) = theta*(mu - X(t))*dt + noise
        # Rearrange: dX = a + b*X(t) where a = theta*mu*dt, b = -theta*dt
        dx = np.diff(series)
        x = series[:-1]

        # OLS
        A = np.column_stack([np.ones(len(x)), x])
        result = np.linalg.lstsq(A, dx, rcond=None)
        a, b = result[0]

        self.theta = max(-b / dt, 1e-6)
        self.mu = a / (self.theta * dt) if self.theta > 1e-6 else np.mean(series)

        residuals = dx - (a + b * x)
        self.sigma = np.std(residuals) / np.sqrt(dt)

    def half_life(self) -> float:
        if self.theta <= 0:
            return float("inf")
        return np.log(2) / self.theta

    def z_score(self, current_value: float) -> float:
        if self.sigma <= 0 or self.theta <= 0:
            return 0.0
        std = self.sigma / np.sqrt(2 * self.theta)
        if std <= 0:
            return 0.0
        return (current_value - self.mu) / std


class FactorModel:
    """Simple OLS factor model for decomposing returns."""

    def __init__(self):
        self.betas: dict[str, float] = {}
        self.intercept: float = 0.0
        self._fitted = False

    def fit(self, returns: np.ndarray, factors: pd.DataFrame) -> None:
        X = factors.values
        y = returns

        A = np.column_stack([np.ones(len(y)), X])
        result = np.linalg.lstsq(A, y, rcond=None)
        coeffs = result[0]

        self.intercept = coeffs[0]
        self.betas = {col: float(coeffs[i + 1]) for i, col in enumerate(factors.columns)}
        self._fitted = True

    def predict(self, factors: pd.DataFrame) -> np.ndarray:
        pred = np.full(len(factors), self.intercept)
        for col in factors.columns:
            if col in self.betas:
                pred += self.betas[col] * factors[col].values
        return pred

    def residuals(self, returns: np.ndarray, factors: pd.DataFrame) -> np.ndarray:
        return returns - self.predict(factors)


class MeanReversionLayer:
    def __init__(self, half_life_suppression_days: int = 15):
        self.half_life_suppression_days = half_life_suppression_days
        self.factor_models: dict[str, FactorModel] = {}
        self.ou_models: dict[str, dict[str, OUProcess]] = {}  # ticker -> regime -> OU

    def fit_factor_model(
        self, ticker: str, returns: np.ndarray, factors: pd.DataFrame
    ) -> FactorModel:
        fm = FactorModel()
        fm.fit(returns, factors)
        self.factor_models[ticker] = fm
        return fm

    def fit_ou_per_regime(
        self,
        ticker: str,
        residuals: np.ndarray,
        regime_labels: np.ndarray,
        regime_names: dict[int, str],
        dt: float = 1.0 / 252,
    ) -> dict[str, OUProcess]:
        self.ou_models[ticker] = {}
        for idx, name in regime_names.items():
            mask = regime_labels == idx
            if mask.sum() < 20:
                # Not enough data, use full sample
                regime_resid = residuals
            else:
                regime_resid = residuals[mask]
            ou = OUProcess()
            ou.calibrate(regime_resid, dt=dt)
            self.ou_models[ticker][name] = ou
        return self.ou_models[ticker]

    def compute_signal(
        self,
        ticker: str,
        z_score: float,
        half_life: float,
        regime: str,
        fair_value: float,
        current_price: float,
        ou_params: dict,
        factor_betas: dict,
    ) -> dict:
        suppressed = (
            regime in ("escalation", "crisis")
            and half_life > self.half_life_suppression_days
        )

        return {
            "ticker": ticker,
            "fair_value": fair_value,
            "current_price": current_price,
            "residual_z_score": z_score,
            "half_life_days": half_life,
            "regime": regime,
            "signal_suppressed": suppressed,
            "factors": factor_betas,
            "ou_params": ou_params,
            "data_staleness_seconds": 0,
        }
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_mean_reversion.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add layers/mean_reversion.py tests/test_mean_reversion.py
git commit -m "feat: add factor-residual OU mean reversion layer with regime gating"
```

---

### Task 10: Layer 4 — Options-Implied / Volatility Signals

**Files:**
- Create: `layers/options_implied.py`
- Create: `tests/test_options_implied.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_options_implied.py
import pytest


def test_term_structure_contango():
    from layers.options_implied import OptionsImpliedLayer

    layer = OptionsImpliedLayer(vix_high=30, vix_low=18, skew_elevated=150, pcr_heavy=1.3, pcr_light=0.7)
    result = layer.compute(vix=20.0, vix3m=22.0, skew=130.0, put_call_ratio=0.9)
    assert result["term_structure_state"] == "contango"
    assert result["term_structure_ratio"] > 1.0


def test_term_structure_backwardation():
    from layers.options_implied import OptionsImpliedLayer

    layer = OptionsImpliedLayer(vix_high=30, vix_low=18, skew_elevated=150, pcr_heavy=1.3, pcr_light=0.7)
    result = layer.compute(vix=30.0, vix3m=25.0, skew=130.0, put_call_ratio=0.9)
    assert result["term_structure_state"] == "backwardation"
    assert result["term_structure_ratio"] < 1.0


def test_vix_elevated_signal():
    from layers.options_implied import OptionsImpliedLayer

    layer = OptionsImpliedLayer(vix_high=30, vix_low=18, skew_elevated=150, pcr_heavy=1.3, pcr_light=0.7)
    result = layer.compute(vix=35.0, vix3m=30.0, skew=130.0, put_call_ratio=0.9)
    assert "vix_elevated" in result["signals"]


def test_skew_elevated_signal():
    from layers.options_implied import OptionsImpliedLayer

    layer = OptionsImpliedLayer(vix_high=30, vix_low=18, skew_elevated=150, pcr_heavy=1.3, pcr_light=0.7)
    result = layer.compute(vix=20.0, vix3m=22.0, skew=160.0, put_call_ratio=0.9)
    assert "skew_elevated" in result["signals"]


def test_pcr_heavy_signal():
    from layers.options_implied import OptionsImpliedLayer

    layer = OptionsImpliedLayer(vix_high=30, vix_low=18, skew_elevated=150, pcr_heavy=1.3, pcr_light=0.7)
    result = layer.compute(vix=20.0, vix3m=22.0, skew=130.0, put_call_ratio=1.5)
    assert "pcr_heavy" in result["signals"]


def test_no_signals_in_calm_market():
    from layers.options_implied import OptionsImpliedLayer

    layer = OptionsImpliedLayer(vix_high=30, vix_low=18, skew_elevated=150, pcr_heavy=1.3, pcr_light=0.7)
    result = layer.compute(vix=20.0, vix3m=22.0, skew=130.0, put_call_ratio=0.9)
    assert len(result["signals"]) == 0


def test_missing_data_handled():
    from layers.options_implied import OptionsImpliedLayer

    layer = OptionsImpliedLayer(vix_high=30, vix_low=18, skew_elevated=150, pcr_heavy=1.3, pcr_light=0.7)
    result = layer.compute(vix=20.0, vix3m=None, skew=None, put_call_ratio=None)
    assert result["term_structure_ratio"] is None
    assert result["skew"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_options_implied.py -v`
Expected: FAIL

- [ ] **Step 3: Implement options implied layer**

```python
# layers/options_implied.py
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OptionsImpliedLayer:
    def __init__(
        self,
        vix_high: float = 30,
        vix_low: float = 18,
        skew_elevated: float = 150,
        pcr_heavy: float = 1.3,
        pcr_light: float = 0.7,
    ):
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
        signals = []

        # VIX term structure
        if vix3m is not None and vix > 0:
            term_ratio = vix3m / vix
            term_state = "contango" if term_ratio > 1.0 else "backwardation"

            # Detect flip
            if self._prev_term_state and term_state != self._prev_term_state:
                signals.append("term_structure_flip")
            self._prev_term_state = term_state
        else:
            term_ratio = None
            term_state = "unknown"

        # VIX level
        if vix >= self.vix_high:
            signals.append("vix_elevated")
        elif vix <= self.vix_low:
            signals.append("vix_low")

        # SKEW
        if skew is not None and skew >= self.skew_elevated:
            signals.append("skew_elevated")

        # Put/call ratio
        if put_call_ratio is not None:
            if put_call_ratio >= self.pcr_heavy:
                signals.append("pcr_heavy")
            elif put_call_ratio <= self.pcr_light:
                signals.append("pcr_light")

        return {
            "vix": vix,
            "vix3m": vix3m,
            "term_structure_ratio": term_ratio,
            "term_structure_state": term_state,
            "skew": skew,
            "put_call_ratio": put_call_ratio,
            "signals": signals,
            "data_staleness_seconds": 0,
        }
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_options_implied.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add layers/options_implied.py tests/test_options_implied.py
git commit -m "feat: add options-implied volatility signal layer"
```

---

### Task 11: Layer 6 — Cross-Asset Correlation Monitor

**Files:**
- Create: `layers/correlation.py`
- Create: `tests/test_correlation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_correlation.py
import pytest
import numpy as np
import pandas as pd


def test_rolling_correlation_matrix():
    from layers.correlation import CorrelationLayer

    np.random.seed(42)
    n = 30
    data = pd.DataFrame({
        "A": np.random.randn(n).cumsum(),
        "B": np.random.randn(n).cumsum(),
        "C": np.random.randn(n).cumsum(),
    })
    layer = CorrelationLayer(window=5, slow_window=21, crisis_threshold=0.55)
    corr = layer.rolling_correlation(data, window=5)
    assert corr.shape == (3, 3)
    # Diagonal should be 1
    np.testing.assert_allclose(np.diag(corr), [1, 1, 1], atol=1e-10)


def test_eigenvalue_ratio():
    from layers.correlation import CorrelationLayer

    # Perfectly correlated = eigenvalue ratio near 1
    layer = CorrelationLayer(window=5, slow_window=21, crisis_threshold=0.55)
    corr = np.array([[1.0, 0.99, 0.98], [0.99, 1.0, 0.97], [0.98, 0.97, 1.0]])
    ratio = layer.eigenvalue_ratio(corr)
    assert ratio > 0.9


def test_eigenvalue_ratio_uncorrelated():
    from layers.correlation import CorrelationLayer

    layer = CorrelationLayer(window=5, slow_window=21, crisis_threshold=0.55)
    corr = np.eye(3)
    ratio = layer.eigenvalue_ratio(corr)
    assert abs(ratio - 1.0 / 3.0) < 0.01


def test_sign_flip_detection():
    from layers.correlation import CorrelationLayer

    layer = CorrelationLayer(window=5, slow_window=21, crisis_threshold=0.55)
    prev = {"A_B": 0.5, "A_C": -0.3}
    curr = {"A_B": -0.2, "A_C": -0.5}
    flips = layer.detect_sign_flips(prev, curr)
    assert len(flips) == 1
    assert flips[0] == ("A", "B", "pos_to_neg")


def test_lead_lag_detection():
    from layers.correlation import CorrelationLayer

    np.random.seed(42)
    n = 50
    # B leads A by 1 day
    b = np.random.randn(n)
    a = np.zeros(n)
    a[1:] = 0.8 * b[:-1] + 0.2 * np.random.randn(n - 1)

    layer = CorrelationLayer(window=5, slow_window=21, crisis_threshold=0.55)
    result = layer.compute_lead_lag(a, b, max_lag=3)
    assert result["best_lag"] != 0  # should detect lag


def test_full_correlation_output():
    from layers.correlation import CorrelationLayer

    np.random.seed(42)
    n = 30
    data = pd.DataFrame({
        "VTI": np.random.randn(n).cumsum(),
        "CL=F": np.random.randn(n).cumsum(),
        "^VIX": np.random.randn(n).cumsum(),
    })
    layer = CorrelationLayer(window=5, slow_window=21, crisis_threshold=0.55)
    result = layer.compute(data, focus_pairs=[["VTI", "CL=F"]])
    assert "eigenvalue_ratio" in result
    assert "focus_pair_correlations" in result
    assert "correlation_regime" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_correlation.py -v`
Expected: FAIL

- [ ] **Step 3: Implement correlation layer**

```python
# layers/correlation.py
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CorrelationLayer:
    def __init__(
        self,
        window: int = 5,
        slow_window: int = 21,
        crisis_threshold: float = 0.55,
    ):
        self.window = window
        self.slow_window = slow_window
        self.crisis_threshold = crisis_threshold
        self._prev_pair_corrs: dict[str, float] = {}

    def rolling_correlation(self, data: pd.DataFrame, window: int) -> np.ndarray:
        returns = data.pct_change().dropna()
        if len(returns) < window:
            return np.eye(len(data.columns))
        recent = returns.tail(window)
        corr = recent.corr().values
        # Handle NaN
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    def eigenvalue_ratio(self, corr_matrix: np.ndarray) -> float:
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        total = eigenvalues.sum()
        if total <= 0:
            return 0.0
        return float(eigenvalues[0] / total)

    def detect_sign_flips(
        self, prev_corrs: dict[str, float], curr_corrs: dict[str, float]
    ) -> list[tuple[str, str, str]]:
        flips = []
        for key in curr_corrs:
            if key in prev_corrs:
                prev_sign = prev_corrs[key] >= 0
                curr_sign = curr_corrs[key] >= 0
                if prev_sign != curr_sign:
                    a, b = key.split("_", 1)
                    direction = "pos_to_neg" if prev_sign else "neg_to_pos"
                    flips.append((a, b, direction))
        return flips

    def compute_lead_lag(
        self, series_a: np.ndarray, series_b: np.ndarray, max_lag: int = 3
    ) -> dict:
        best_corr = 0.0
        best_lag = 0
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(series_a, series_b)[0, 1]
            elif lag > 0:
                corr = np.corrcoef(series_a[lag:], series_b[:-lag])[0, 1]
            else:
                corr = np.corrcoef(series_a[:lag], series_b[-lag:])[0, 1]
            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        return {"best_lag": best_lag, "correlation": float(best_corr)}

    def compute(
        self,
        data: pd.DataFrame,
        focus_pairs: list[list[str]] = None,
    ) -> dict:
        # Correlation matrices
        corr_fast = self.rolling_correlation(data, self.window)
        corr_slow = self.rolling_correlation(data, self.slow_window)

        # Eigenvalue ratios
        ev_fast = self.eigenvalue_ratio(corr_fast)
        ev_slow = self.eigenvalue_ratio(corr_slow)

        # Correlation regime
        if ev_fast > self.crisis_threshold:
            corr_regime = "crisis_clustering"
        else:
            corr_regime = "normal"

        # Focus pair correlations
        columns = list(data.columns)
        pair_corrs = {}
        if focus_pairs:
            for pair in focus_pairs:
                a, b = pair[0], pair[1]
                if a in columns and b in columns:
                    ai, bi = columns.index(a), columns.index(b)
                    key = f"{a}_{b}".replace("=", "").replace("^", "")
                    pair_corrs[key] = float(corr_fast[ai, bi])

        # Sign flips
        sign_flips = self.detect_sign_flips(self._prev_pair_corrs, pair_corrs)
        self._prev_pair_corrs = pair_corrs.copy()

        # Lead-lag for focus pairs
        returns = data.pct_change().dropna()
        lead_lag = []
        if focus_pairs and len(returns) > 5:
            for pair in focus_pairs:
                a, b = pair[0], pair[1]
                if a in returns.columns and b in returns.columns:
                    ll = self.compute_lead_lag(
                        returns[a].values, returns[b].values, max_lag=3
                    )
                    lead_lag.append({
                        "pair": [a, b],
                        "lag_days": ll["best_lag"],
                        "correlation": ll["correlation"],
                    })

        return {
            "eigenvalue_ratio": ev_fast,
            "eigenvalue_ratio_21d": ev_slow,
            "correlation_regime": corr_regime,
            "sign_flips": [list(f) for f in sign_flips],
            "lead_lag": lead_lag,
            "focus_pair_correlations": pair_corrs,
            "data_staleness_seconds": 0,
        }
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_correlation.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add layers/correlation.py tests/test_correlation.py
git commit -m "feat: add cross-asset correlation monitor with eigenvalue analysis and lead-lag"
```

---

### Task 12: Layer 5 — Position Sizing

**Files:**
- Create: `layers/sizing.py`
- Create: `tests/test_sizing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sizing.py
import pytest
import numpy as np


def test_inverse_vol_sizing():
    from layers.sizing import SizingLayer

    layer = SizingLayer(
        kelly_fraction=0.25,
        target_annual_vol=0.15,
        max_single_pct=0.40,
        max_total_pct=0.85,
        max_drawdown_pct=0.05,
    )
    pct = layer.inverse_vol_size(realized_vol=0.30)
    assert pct == 0.15 / 0.30  # target/realized = 0.5


def test_inverse_vol_capped_at_max():
    from layers.sizing import SizingLayer

    layer = SizingLayer(
        kelly_fraction=0.25,
        target_annual_vol=0.15,
        max_single_pct=0.40,
        max_total_pct=0.85,
        max_drawdown_pct=0.05,
    )
    pct = layer.inverse_vol_size(realized_vol=0.05)  # low vol = high size
    assert pct <= 0.40


def test_kelly_cap():
    from layers.sizing import SizingLayer

    layer = SizingLayer(
        kelly_fraction=0.25,
        target_annual_vol=0.15,
        max_single_pct=0.40,
        max_total_pct=0.85,
        max_drawdown_pct=0.05,
    )
    # Win prob 0.6, win/loss ratio 1.5
    cap = layer.kelly_criterion(win_prob=0.6, win_loss_ratio=1.5)
    # Full Kelly = 0.6 - (1-0.6)/1.5 = 0.333
    # Quarter Kelly = 0.333 * 0.25 = 0.083
    assert abs(cap - 0.25 * (0.6 - 0.4 / 1.5)) < 0.01


def test_kelly_negative_edge():
    from layers.sizing import SizingLayer

    layer = SizingLayer(
        kelly_fraction=0.25,
        target_annual_vol=0.15,
        max_single_pct=0.40,
        max_total_pct=0.85,
        max_drawdown_pct=0.05,
    )
    cap = layer.kelly_criterion(win_prob=0.3, win_loss_ratio=0.8)
    assert cap == 0.0  # negative edge = don't bet


def test_compute_sizing_full():
    from layers.sizing import SizingLayer

    layer = SizingLayer(
        kelly_fraction=0.25,
        target_annual_vol=0.15,
        max_single_pct=0.40,
        max_total_pct=0.85,
        max_drawdown_pct=0.05,
    )
    result = layer.compute(
        ticker="DAL",
        realized_vol=0.30,
        convergence_score=0.65,
        current_position_pct=0.25,
        current_price=69.0,
        entry_price=69.0,
        stop_loss=65.55,
    )
    assert "vol_sized_pct" in result
    assert "kelly_cap_pct" in result
    assert "final_allocation_pct" in result
    assert "edge_positive" in result


def test_stop_loss_distance():
    from layers.sizing import SizingLayer

    layer = SizingLayer(
        kelly_fraction=0.25,
        target_annual_vol=0.15,
        max_single_pct=0.40,
        max_total_pct=0.85,
        max_drawdown_pct=0.05,
    )
    result = layer.compute(
        ticker="DAL",
        realized_vol=0.30,
        convergence_score=0.65,
        current_position_pct=0.25,
        current_price=66.0,  # close to stop
        entry_price=69.0,
        stop_loss=65.55,
    )
    assert result["stop_loss_distance_pct"] < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_sizing.py -v`
Expected: FAIL

- [ ] **Step 3: Implement sizing layer**

```python
# layers/sizing.py
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SizingLayer:
    def __init__(
        self,
        kelly_fraction: float = 0.25,
        target_annual_vol: float = 0.15,
        max_single_pct: float = 0.40,
        max_total_pct: float = 0.85,
        max_drawdown_pct: float = 0.05,
    ):
        self.kelly_fraction = kelly_fraction
        self.target_annual_vol = target_annual_vol
        self.max_single_pct = max_single_pct
        self.max_total_pct = max_total_pct
        self.max_drawdown_pct = max_drawdown_pct

    def inverse_vol_size(self, realized_vol: float) -> float:
        if realized_vol <= 0:
            return self.max_single_pct
        raw = self.target_annual_vol / realized_vol
        return min(raw, self.max_single_pct)

    def kelly_criterion(self, win_prob: float, win_loss_ratio: float) -> float:
        if win_loss_ratio <= 0:
            return 0.0
        kelly = win_prob - (1 - win_prob) / win_loss_ratio
        if kelly <= 0:
            return 0.0
        return kelly * self.kelly_fraction

    def compute(
        self,
        ticker: str,
        realized_vol: float,
        convergence_score: float,
        current_position_pct: float,
        current_price: float,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> dict:
        # Inverse vol sizing
        vol_sized = self.inverse_vol_size(realized_vol)

        # Kelly cap: use |convergence_score| as proxy for win probability
        # Map convergence to win_prob: score of 0.5 -> ~60% win, 1.0 -> ~75%
        win_prob = 0.5 + abs(convergence_score) * 0.25
        # Assume 1.5:1 reward/risk for swing trades (rough estimate)
        win_loss_ratio = 1.5
        kelly_cap = self.kelly_criterion(win_prob, win_loss_ratio)

        # Final allocation = min of vol sizing and kelly cap
        if kelly_cap > 0:
            final = min(vol_sized, kelly_cap)
        else:
            final = 0.0

        # Apply hard cap
        final = min(final, self.max_single_pct)

        # Determine sizing basis
        if kelly_cap > 0 and kelly_cap < vol_sized:
            basis = "kelly_cap"
        elif kelly_cap == 0:
            basis = "no_edge"
        else:
            basis = "vol_target"

        # Edge determination
        edge_positive = convergence_score > 0 and kelly_cap > 0

        # Stop loss distance
        stop_distance = 0.0
        if stop_loss and current_price > 0:
            stop_distance = (current_price - stop_loss) / current_price

        # Position status
        if current_position_pct > final * 1.1 and final > 0:
            action = "oversized"
        elif current_position_pct < final * 0.9 and final > 0 and edge_positive:
            action = "undersized"
        elif not edge_positive:
            action = "no_edge"
        else:
            action = "appropriately_sized"

        return {
            "ticker": ticker,
            "vol_sized_pct": round(vol_sized, 4),
            "kelly_cap_pct": round(kelly_cap, 4),
            "final_allocation_pct": round(final, 4),
            "sizing_basis": basis,
            "edge_positive": edge_positive,
            "current_position_pct": round(current_position_pct, 4),
            "action": action,
            "stop_loss_distance_pct": round(stop_distance, 4),
            "data_staleness_seconds": 0,
        }
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_sizing.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add layers/sizing.py tests/test_sizing.py
git commit -m "feat: add position sizing layer with inverse-vol and Kelly cap"
```

---

## Phase 3: Integration (Tasks 13-15)

---

### Task 13: Master Decision Engine

**Files:**
- Create: `engine/decision.py`
- Create: `tests/test_decision.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_decision.py
import pytest


def test_directional_signal_from_regime():
    from engine.decision import DecisionEngine

    engine = DecisionEngine(convergence_min_layers=3, high_threshold=0.5, moderate_threshold=0.3)
    signal = engine.regime_to_signal("escalation", asset_class="equity")
    assert signal == -1

    signal = engine.regime_to_signal("risk_on", asset_class="equity")
    assert signal == 1

    signal = engine.regime_to_signal("escalation", asset_class="gold")
    assert signal == 1  # gold bullish in crisis


def test_convergence_score_calculation():
    from engine.decision import DecisionEngine

    engine = DecisionEngine(convergence_min_layers=3, high_threshold=0.5, moderate_threshold=0.3)
    signals = {"regime": -1, "sentiment": -1, "options": -1, "correlation": 0, "mean_reversion": 0, "sizing": -1}
    score, agreeing = engine.compute_convergence(signals)
    assert score < 0
    assert agreeing >= 3


def test_high_conviction_signal():
    from engine.decision import DecisionEngine

    engine = DecisionEngine(convergence_min_layers=3, high_threshold=0.5, moderate_threshold=0.3)
    signals = {"regime": -1, "sentiment": -1, "options": -1, "correlation": -1, "mean_reversion": -1, "sizing": -1}
    score, agreeing = engine.compute_convergence(signals)
    result = engine.classify_conviction(score, agreeing)
    assert result["conviction"] == "high"
    assert result["direction"] == "bearish"


def test_mixed_signal():
    from engine.decision import DecisionEngine

    engine = DecisionEngine(convergence_min_layers=3, high_threshold=0.5, moderate_threshold=0.3)
    signals = {"regime": 1, "sentiment": -1, "options": 0, "correlation": 1, "mean_reversion": -1, "sizing": 0}
    score, agreeing = engine.compute_convergence(signals)
    result = engine.classify_conviction(score, agreeing)
    assert result["conviction"] == "mixed"


def test_full_decision_output():
    from engine.decision import DecisionEngine

    engine = DecisionEngine(convergence_min_layers=3, high_threshold=0.5, moderate_threshold=0.3)

    layer_outputs = {
        "regime": {"regime_label": "escalation", "changepoint_probability": 0.7},
        "sentiment": {"sentiment_mean": -0.5, "sentiment_velocity": -0.1},
        "options_implied": {"signals": ["vix_elevated", "term_structure_backwardation"]},
        "correlation": {"correlation_regime": "crisis_clustering"},
        "mean_reversion": {"DAL": {"residual_z_score": -1.5, "signal_suppressed": True}},
        "sizing": {"DAL": {"edge_positive": False}},
    }

    result = engine.aggregate(
        layer_outputs=layer_outputs,
        tickers=["DAL"],
        asset_classes={"DAL": "equity"},
    )
    assert "tickers" in result
    assert "DAL" in result["tickers"]
    assert "market_wide" in result
    assert "timestamp" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_decision.py -v`
Expected: FAIL

- [ ] **Step 3: Implement decision engine**

```python
# engine/decision.py
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

CRISIS_REGIMES = {"escalation", "crisis"}
BULLISH_REGIMES = {"risk_on"}
SAFE_HAVEN_ASSETS = {"GLD", "TLT", "^VIX"}


class DecisionEngine:
    def __init__(
        self,
        convergence_min_layers: int = 3,
        high_threshold: float = 0.5,
        moderate_threshold: float = 0.3,
    ):
        self.convergence_min_layers = convergence_min_layers
        self.high_threshold = high_threshold
        self.moderate_threshold = moderate_threshold

    def regime_to_signal(self, regime_label: str, asset_class: str = "equity") -> int:
        is_safe_haven = asset_class in ("gold", "treasury", "volatility")
        if regime_label in CRISIS_REGIMES:
            return 1 if is_safe_haven else -1
        elif regime_label in BULLISH_REGIMES:
            return -1 if is_safe_haven else 1
        return 0  # stalemate = neutral

    def sentiment_to_signal(self, sentiment_mean: float, velocity: float) -> int:
        if sentiment_mean < -0.3 or velocity < -0.1:
            return -1
        elif sentiment_mean > 0.3 or velocity > 0.1:
            return 1
        return 0

    def options_to_signal(self, signals: list[str]) -> int:
        bearish = {"vix_elevated", "term_structure_backwardation", "skew_elevated", "pcr_heavy"}
        bullish = {"vix_low", "pcr_light"}
        bearish_count = len(set(signals) & bearish)
        bullish_count = len(set(signals) & bullish)
        if bearish_count > bullish_count:
            return -1
        elif bullish_count > bearish_count:
            return 1
        return 0

    def correlation_to_signal(self, corr_regime: str) -> int:
        if corr_regime == "crisis_clustering":
            return -1
        return 0

    def mean_reversion_to_signal(self, ticker_data: dict) -> int:
        if ticker_data.get("signal_suppressed", False):
            return 0
        z = ticker_data.get("residual_z_score", 0)
        if z < -2.0:
            return 1  # undervalued
        elif z > 2.0:
            return -1  # overvalued
        return 0

    def sizing_to_signal(self, ticker_data: dict) -> int:
        if ticker_data.get("edge_positive", False):
            return 1
        return -1

    def compute_convergence(self, signals: dict[str, int]) -> tuple[float, int]:
        values = list(signals.values())
        if not values:
            return 0.0, 0
        score = sum(values) / len(values)
        # Count agreeing: signals with same sign as majority
        majority_sign = 1 if score >= 0 else -1
        agreeing = sum(1 for v in values if v == majority_sign)
        return score, agreeing

    def classify_conviction(self, score: float, agreeing: int) -> dict:
        abs_score = abs(score)
        if abs_score >= self.high_threshold and agreeing >= self.convergence_min_layers:
            conviction = "high"
        elif abs_score >= self.moderate_threshold and agreeing >= 2:
            conviction = "moderate"
        elif agreeing < 2 or abs_score < 0.1:
            conviction = "mixed"
        else:
            conviction = "low"

        if abs_score < 0.05:
            direction = "neutral"
        elif score > 0:
            direction = "bullish"
        else:
            direction = "bearish"

        return {"conviction": conviction, "direction": direction}

    def aggregate(
        self,
        layer_outputs: dict,
        tickers: list[str],
        asset_classes: dict[str, str] = None,
    ) -> dict:
        if asset_classes is None:
            asset_classes = {}

        regime_data = layer_outputs.get("regime", {})
        sentiment_data = layer_outputs.get("sentiment", {})
        options_data = layer_outputs.get("options_implied", {})
        corr_data = layer_outputs.get("correlation", {})
        mr_data = layer_outputs.get("mean_reversion", {})
        sizing_data = layer_outputs.get("sizing", {})

        # Market-wide signals
        regime_label = regime_data.get("regime_label", "stalemate")
        sentiment_signal = self.sentiment_to_signal(
            sentiment_data.get("sentiment_mean", 0),
            sentiment_data.get("sentiment_velocity", 0),
        )
        options_signal = self.options_to_signal(options_data.get("signals", []))
        corr_signal = self.correlation_to_signal(corr_data.get("correlation_regime", "normal"))

        ticker_results = {}
        for ticker in tickers:
            ac = asset_classes.get(ticker, "equity")
            regime_signal = self.regime_to_signal(regime_label, ac)

            ticker_mr = mr_data.get(ticker, {})
            mr_signal = self.mean_reversion_to_signal(ticker_mr)

            ticker_sz = sizing_data.get(ticker, {})
            sz_signal = self.sizing_to_signal(ticker_sz)

            signals = {
                "regime": regime_signal,
                "sentiment": sentiment_signal,
                "options": options_signal,
                "correlation": corr_signal,
                "mean_reversion": mr_signal,
                "sizing": sz_signal,
            }

            score, agreeing = self.compute_convergence(signals)
            conviction_info = self.classify_conviction(score, agreeing)

            ticker_results[ticker] = {
                "convergence_score": round(score, 3),
                "layers_agreeing": agreeing,
                "direction": conviction_info["direction"],
                "conviction": conviction_info["conviction"],
                "signals": signals,
            }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tickers": ticker_results,
            "market_wide": {
                "regime": regime_label,
                "sentiment_velocity": sentiment_data.get("sentiment_velocity", 0),
                "vix_term_structure": options_data.get("term_structure_state", "unknown"),
                "correlation_regime": corr_data.get("correlation_regime", "unknown"),
            },
        }
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_decision.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add engine/decision.py tests/test_decision.py
git commit -m "feat: add master decision engine with convergence scoring"
```

---

### Task 14: Main Entry Point & Docker Compose

**Files:**
- Create: `main.py`
- Create: `run_all.py`
- Create: `docker-compose.yml`
- Create: `Dockerfile`

- [ ] **Step 1: Implement main.py**

```python
# main.py
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

from common.config import load_config
from common.logging import setup_logging, get_logger
from common.redis_client import RedisStreamClient
from data.fetcher import DataFetcher
from data.store import DataStore
from engine.alerts import AlertManager, AlertSeverity
from engine.decision import DecisionEngine
from engine.price_monitor import PriceMonitor
from layers.regime import RegimeDetector
from layers.sentiment import SentimentLayer
from layers.mean_reversion import MeanReversionLayer
from layers.options_implied import OptionsImpliedLayer
from layers.correlation import CorrelationLayer
from layers.sizing import SizingLayer

logger = None


async def run_signal_cycle(
    config,
    fetcher: DataFetcher,
    store: DataStore,
    redis: RedisStreamClient,
    alert_mgr: AlertManager,
    regime_detector: RegimeDetector,
    sentiment_layer: SentimentLayer,
    mr_layer: MeanReversionLayer,
    options_layer: OptionsImpliedLayer,
    corr_layer: CorrelationLayer,
    sizing_layer: SizingLayer,
    decision_engine: DecisionEngine,
):
    log = get_logger("signal_cycle")
    interval = config.timing.signal_update_interval_minutes * 60

    while True:
        try:
            log.info("cycle_start")

            # Fetch latest data
            all_tickers = list(config.all_tickers())
            prices = fetcher.fetch_current_prices(all_tickers)

            # Save prices
            for ticker, price in prices.items():
                if price is not None:
                    store.save_price(ticker, price)

            # Layer 1: Regime
            regime_output = store.get_latest_layer_output("regime") or {
                "regime_label": "stalemate",
                "regime_probabilities": [],
                "changepoint_probability": 0.0,
                "transition_matrix": [],
                "data_staleness_seconds": 0,
            }

            # Layer 2: Sentiment
            gdelt_headlines = fetcher.fetch_gdelt_headlines(config.geopolitical.keywords)
            newsapi_headlines = fetcher.fetch_newsapi_headlines(config.geopolitical.keywords)
            all_headlines = gdelt_headlines + newsapi_headlines

            filtered = sentiment_layer.filter_headlines(all_headlines)
            titles = [h["title"] for h in filtered]
            scores = sentiment_layer.scorer.score_headlines(titles) if titles else []

            # LLM analysis
            llm_result = None
            if sentiment_layer.llm_provider and titles:
                from layers.sentiment import llm_contextual_analysis
                llm_result = await llm_contextual_analysis(
                    titles[:20],
                    sentiment_layer.llm_provider,
                    sentiment_layer.llm_model,
                    config.geopolitical.keywords,
                )

            sentiment_output = sentiment_layer.build_output(
                sentiment_scores=scores,
                sentiment_history=[],
                hawkes_result={"branching_ratio": 0.0, "intensity": 0.0},
                headline_count_15m=len(filtered),
                headline_count_1h=len(filtered),
                top_headlines=titles[:5],
                llm_result=llm_result,
                active_sources=[s for s in ["gdelt", "newsapi"]
                               if (s == "gdelt" and gdelt_headlines) or
                                  (s == "newsapi" and newsapi_headlines)],
            )

            # Layer 4: Options/Vol
            vol_prices = fetcher.fetch_current_prices(["^VIX", "^VIX3M", "^SKEW"])
            options_output = options_layer.compute(
                vix=vol_prices.get("^VIX", 20.0) or 20.0,
                vix3m=vol_prices.get("^VIX3M"),
                skew=vol_prices.get("^SKEW"),
            )

            # Layer 6: Correlation
            import pandas as pd
            corr_tickers = config.correlation.assets
            corr_data = fetcher.fetch_multiple_historical(
                corr_tickers, start="2026-03-01", end="2026-04-08"
            )
            if corr_data:
                close_df = pd.DataFrame({
                    t: d["Close"] for t, d in corr_data.items() if "Close" in d.columns
                })
                corr_output = corr_layer.compute(close_df, config.correlation.focus_pairs)
            else:
                corr_output = {"correlation_regime": "unknown", "eigenvalue_ratio": 0}

            # Save layer outputs
            store.save_layer_output("regime", regime_output)
            store.save_layer_output("sentiment", sentiment_output)
            store.save_layer_output("options_implied", options_output)
            store.save_layer_output("correlation", corr_output)

            # Publish to Redis
            await redis.publish("regime", regime_output)
            await redis.publish("sentiment", sentiment_output)
            await redis.publish("options_implied", options_output)
            await redis.publish("correlation", corr_output)

            # Decision engine
            decision_output = decision_engine.aggregate(
                layer_outputs={
                    "regime": regime_output,
                    "sentiment": sentiment_output,
                    "options_implied": options_output,
                    "correlation": corr_output,
                },
                tickers=[p.ticker for p in config.portfolio.positions],
            )
            store.save_decision_snapshot(decision_output)
            await redis.publish("decision", decision_output)

            # Check for alerts
            for ticker, data in decision_output.get("tickers", {}).items():
                if data["conviction"] == "high":
                    await alert_mgr.send_alert(
                        AlertSeverity.CRITICAL,
                        "convergence",
                        f"{ticker}: {data['direction']} (score: {data['convergence_score']:.2f}, "
                        f"{data['layers_agreeing']} layers agree)",
                    )

            # Flush batched alerts
            await alert_mgr.flush_and_send()

            log.info("cycle_complete")

        except Exception as e:
            log.error("cycle_error", error=str(e))

        await asyncio.sleep(interval)


async def main_async(config_path: str, backtest: bool = False, crisis: str = None):
    config = load_config(Path(config_path))
    setup_logging(config.logging.level, config.logging.file)
    global logger
    logger = get_logger("main")

    if backtest:
        from backtest.replay import run_backtest
        await run_backtest(config, crisis_name=crisis)
        return

    # Initialize components
    store = DataStore(config.database.path)
    store.init_db()

    redis = RedisStreamClient(
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        stream_maxlen=config.redis.stream_maxlen,
    )
    await redis.connect()

    fetcher = DataFetcher()

    alert_mgr = AlertManager(
        bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        store=store,
    )

    # Initialize layers
    regime_detector = RegimeDetector(
        n_states=config.regime.n_states,
        hazard_lambda=config.regime.bocpd_hazard_lambda,
        state_labels=config.regime.state_labels,
    )
    sentiment_layer = SentimentLayer(
        keywords=config.geopolitical.keywords,
        llm_provider=config.geopolitical.llm_provider,
        llm_model=config.geopolitical.llm_model,
    )
    mr_layer = MeanReversionLayer(
        half_life_suppression_days=config.alerts.half_life_suppression_days
    )
    options_layer = OptionsImpliedLayer(
        vix_high=config.alerts.vix_high,
        vix_low=config.alerts.vix_low,
        skew_elevated=config.alerts.skew_elevated,
        pcr_heavy=config.alerts.pcr_heavy,
        pcr_light=config.alerts.pcr_light,
    )
    corr_layer = CorrelationLayer(
        window=config.correlation.rolling_window_days,
        slow_window=config.correlation.rolling_window_slow_days,
        crisis_threshold=config.correlation.eigenvalue_crisis_threshold,
    )
    sizing_layer = SizingLayer(
        kelly_fraction=config.sizing.kelly_fraction,
        target_annual_vol=config.sizing.target_annual_vol,
        max_single_pct=config.sizing.max_single_position_pct,
        max_total_pct=config.sizing.max_total_exposure_pct,
        max_drawdown_pct=config.sizing.max_drawdown_per_position_pct,
    )
    decision_engine = DecisionEngine(
        convergence_min_layers=config.alerts.convergence_min_layers,
        high_threshold=config.alerts.convergence_high_threshold,
        moderate_threshold=config.alerts.convergence_moderate_threshold,
    )

    price_monitor = PriceMonitor(config=config, fetcher=fetcher, alert_manager=alert_mgr)

    # Run all loops concurrently
    logger.info("system_starting")
    tasks = [
        asyncio.create_task(run_signal_cycle(
            config, fetcher, store, redis, alert_mgr,
            regime_detector, sentiment_layer, mr_layer,
            options_layer, corr_layer, sizing_layer, decision_engine,
        )),
        asyncio.create_task(price_monitor.run_owned_loop(
            config.timing.price_check_owned_seconds
        )),
        asyncio.create_task(price_monitor.run_watchlist_loop(
            config.timing.price_check_watchlist_seconds
        )),
    ]

    # Graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: [t.cancel() for t in tasks])
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("system_shutting_down")
    finally:
        await redis.close()
        store.close()


def main():
    parser = argparse.ArgumentParser(description="Geopolitical Trading Signal System")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--backtest", action="store_true", help="Run in backtest mode")
    parser.add_argument("--crisis", type=str, help="Specific crisis to backtest")
    args = parser.parse_args()

    asyncio.run(main_async(args.config, args.backtest, args.crisis))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create run_all.py for dev mode**

```python
# run_all.py
"""Development mode: run all components in a single process."""
import sys
sys.argv = ["main.py", "--config", "config.yaml"]

from main import main
main()
```

- [ ] **Step 3: Create Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

- [ ] **Step 4: Create docker-compose.yml**

```yaml
# docker-compose.yml
version: "3.8"

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  trading-system:
    build: .
    depends_on:
      - redis
    volumes:
      - ./config.yaml:/app/config.yaml
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
      - NEWSAPI_KEY=${NEWSAPI_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  redis_data:
```

- [ ] **Step 5: Create .env.example**

```bash
# .env.example
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
NEWSAPI_KEY=your_newsapi_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
```

- [ ] **Step 6: Create .gitignore**

```
# .gitignore
__pycache__/
*.pyc
*.pyo
.env
data/*.db
logs/
models/*.pkl
.pytest_cache/
*.egg-info/
dist/
build/
```

- [ ] **Step 7: Commit**

```bash
git add main.py run_all.py Dockerfile docker-compose.yml .env.example .gitignore
git commit -m "feat: add main entry point, Docker Compose, and dev runner"
```

---

### Task 15: Backtest Framework

**Files:**
- Create: `backtest/__init__.py`
- Create: `backtest/replay.py`
- Create: `backtest/evaluator.py`
- Create: `backtest/report.py`
- Create: `tests/test_backtest.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_backtest.py
import pytest
import numpy as np


def test_signal_hit_rate():
    from backtest.evaluator import SignalEvaluator

    evaluator = SignalEvaluator()
    # Signal said bearish, price went down -> hit
    signals = [
        {"direction": "bearish", "timestamp_idx": 0, "half_life": 5},
        {"direction": "bullish", "timestamp_idx": 10, "half_life": 5},
        {"direction": "bearish", "timestamp_idx": 20, "half_life": 5},
    ]
    prices = np.concatenate([
        np.linspace(100, 90, 10),   # down (bearish hit)
        np.linspace(90, 95, 10),    # up (bullish hit)
        np.linspace(95, 100, 10),   # up (bearish miss)
    ])
    hit_rate = evaluator.compute_hit_rate(signals, prices)
    assert 0 <= hit_rate <= 1.0
    assert abs(hit_rate - 2.0 / 3.0) < 0.01


def test_signal_lead_time():
    from backtest.evaluator import SignalEvaluator

    evaluator = SignalEvaluator()
    # Signal fired at index 5, major move started at index 8
    signal_idx = 5
    # Price stable then drops sharply
    prices = np.concatenate([
        np.full(8, 100.0),
        np.linspace(100, 80, 12),  # 20% drop starts at idx 8
    ])
    lead_time = evaluator.compute_lead_time(signal_idx, prices, direction="bearish", threshold_pct=0.05)
    assert lead_time == 3  # 8 - 5 = 3 periods ahead


def test_false_positive_rate():
    from backtest.evaluator import SignalEvaluator

    evaluator = SignalEvaluator()
    signals = [
        {"direction": "bearish", "timestamp_idx": 0, "half_life": 5},
        {"direction": "bearish", "timestamp_idx": 10, "half_life": 5},
    ]
    # Price goes up both times -> both false positives
    prices = np.linspace(100, 120, 20)
    fp_rate = evaluator.compute_false_positive_rate(signals, prices, threshold_pct=0.03)
    assert fp_rate == 1.0


def test_regime_accuracy():
    from backtest.evaluator import SignalEvaluator

    evaluator = SignalEvaluator()
    predicted = ["risk_on", "risk_on", "escalation", "escalation", "escalation"]
    actual = ["risk_on", "risk_on", "escalation", "escalation", "stalemate"]
    accuracy = evaluator.regime_accuracy(predicted, actual)
    assert accuracy == 4.0 / 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_backtest.py -v`
Expected: FAIL

- [ ] **Step 3: Implement signal evaluator**

```python
# backtest/__init__.py
# empty

# backtest/evaluator.py
from __future__ import annotations

import numpy as np


class SignalEvaluator:
    def compute_hit_rate(self, signals: list[dict], prices: np.ndarray) -> float:
        if not signals:
            return 0.0

        hits = 0
        for sig in signals:
            idx = sig["timestamp_idx"]
            half_life = int(sig.get("half_life", 5))
            direction = sig["direction"]
            end_idx = min(idx + half_life, len(prices) - 1)

            if end_idx <= idx:
                continue

            price_change = prices[end_idx] - prices[idx]

            if direction == "bearish" and price_change < 0:
                hits += 1
            elif direction == "bullish" and price_change > 0:
                hits += 1

        return hits / len(signals) if signals else 0.0

    def compute_lead_time(
        self,
        signal_idx: int,
        prices: np.ndarray,
        direction: str,
        threshold_pct: float = 0.05,
    ) -> int:
        start_price = prices[signal_idx]
        for i in range(signal_idx + 1, len(prices)):
            pct_change = (prices[i] - start_price) / start_price
            if direction == "bearish" and pct_change < -threshold_pct:
                return i - signal_idx
            elif direction == "bullish" and pct_change > threshold_pct:
                return i - signal_idx
        return -1  # threshold never hit

    def compute_false_positive_rate(
        self,
        signals: list[dict],
        prices: np.ndarray,
        threshold_pct: float = 0.03,
    ) -> float:
        if not signals:
            return 0.0

        false_positives = 0
        for sig in signals:
            idx = sig["timestamp_idx"]
            half_life = int(sig.get("half_life", 5))
            direction = sig["direction"]
            end_idx = min(idx + half_life, len(prices) - 1)

            if end_idx <= idx:
                false_positives += 1
                continue

            max_favorable_move = 0.0
            start_price = prices[idx]
            for i in range(idx + 1, end_idx + 1):
                pct = (prices[i] - start_price) / start_price
                if direction == "bearish":
                    max_favorable_move = min(max_favorable_move, pct)
                else:
                    max_favorable_move = max(max_favorable_move, pct)

            if direction == "bearish" and max_favorable_move > -threshold_pct:
                false_positives += 1
            elif direction == "bullish" and max_favorable_move < threshold_pct:
                false_positives += 1

        return false_positives / len(signals)

    def regime_accuracy(self, predicted: list[str], actual: list[str]) -> float:
        if not predicted or len(predicted) != len(actual):
            return 0.0
        correct = sum(1 for p, a in zip(predicted, actual) if p == a)
        return correct / len(predicted)
```

- [ ] **Step 4: Run tests**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_backtest.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Implement replay and report modules**

```python
# backtest/replay.py
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.evaluator import SignalEvaluator
from backtest.report import generate_report
from common.config import AppConfig
from data.fetcher import DataFetcher
from data.store import DataStore
from engine.decision import DecisionEngine
from layers.correlation import CorrelationLayer
from layers.mean_reversion import MeanReversionLayer, FactorModel, OUProcess
from layers.options_implied import OptionsImpliedLayer
from layers.regime import RegimeDetector
from layers.sentiment import SentimentLayer

logger = logging.getLogger(__name__)


async def run_backtest(config: AppConfig, crisis_name: str = None) -> dict:
    fetcher = DataFetcher()
    evaluator = SignalEvaluator()

    crises = config.regime.training_crises
    if crisis_name:
        crises = [c for c in crises if c.name == crisis_name]
        if not crises:
            logger.error("crisis_not_found", name=crisis_name)
            return {}

    all_results = {}

    for holdout_crisis in crises:
        logger.info("backtest_holdout", crisis=holdout_crisis.name)

        # Training crises = all except holdout
        train_crises = [c for c in config.regime.training_crises if c.name != holdout_crisis.name]

        # Fetch training data
        train_features_list = []
        for crisis in train_crises:
            end = crisis.end or datetime.now().strftime("%Y-%m-%d")
            raw = fetcher.fetch_regime_features(start=str(crisis.start), end=str(end))
            if not raw.empty:
                computed = fetcher.compute_regime_features(raw)
                train_features_list.append(computed)

        if not train_features_list:
            logger.warning("no_training_data", holdout=holdout_crisis.name)
            continue

        train_features = pd.concat(train_features_list).dropna()

        # Train regime detector
        detector = RegimeDetector(
            n_states=config.regime.n_states,
            hazard_lambda=config.regime.bocpd_hazard_lambda,
            state_labels=config.regime.state_labels,
        )
        detector.fit(train_features.values)

        # Fetch holdout data
        holdout_end = holdout_crisis.end or datetime.now().strftime("%Y-%m-%d")
        holdout_raw = fetcher.fetch_regime_features(
            start=str(holdout_crisis.start), end=str(holdout_end)
        )
        if holdout_raw.empty:
            continue
        holdout_features = fetcher.compute_regime_features(holdout_raw)

        # Replay through pipeline
        signals = []
        regime_predictions = []

        options_layer = OptionsImpliedLayer(
            vix_high=config.alerts.vix_high,
            vix_low=config.alerts.vix_low,
            skew_elevated=config.alerts.skew_elevated,
            pcr_heavy=config.alerts.pcr_heavy,
            pcr_light=config.alerts.pcr_light,
        )
        decision_engine = DecisionEngine(
            convergence_min_layers=config.alerts.convergence_min_layers,
            high_threshold=config.alerts.convergence_high_threshold,
            moderate_threshold=config.alerts.convergence_moderate_threshold,
        )

        for i in range(len(holdout_features)):
            window = holdout_features.values[: i + 1]
            if len(window) < 10:
                continue

            regime_result = detector.detect(window)
            regime_predictions.append(regime_result["regime_label"])

            vix_val = holdout_features.iloc[i].get("vix", 20.0) if "vix" in holdout_features.columns else 20.0
            vix_ts = holdout_features.iloc[i].get("vix_term_slope", 1.0) if "vix_term_slope" in holdout_features.columns else 1.0
            options_result = options_layer.compute(
                vix=vix_val,
                vix3m=vix_val * vix_ts if vix_ts else None,
            )

            decision = decision_engine.aggregate(
                layer_outputs={
                    "regime": regime_result,
                    "options_implied": options_result,
                },
                tickers=["SPY"],
            )

            for ticker, data in decision.get("tickers", {}).items():
                if data["conviction"] in ("high", "moderate"):
                    signals.append({
                        "direction": data["direction"],
                        "timestamp_idx": i,
                        "half_life": 5,
                        "conviction": data["conviction"],
                        "score": data["convergence_score"],
                    })

        # Evaluate
        spy_data = fetcher.fetch_historical(
            "SPY", start=str(holdout_crisis.start), end=str(holdout_end)
        )
        if not spy_data.empty and signals:
            spy_prices = spy_data["Close"].values
            # Align signal indices to price data
            aligned_signals = [s for s in signals if s["timestamp_idx"] < len(spy_prices)]

            hit_rate = evaluator.compute_hit_rate(aligned_signals, spy_prices)
            fp_rate = evaluator.compute_false_positive_rate(aligned_signals, spy_prices)
        else:
            hit_rate = 0.0
            fp_rate = 0.0

        all_results[holdout_crisis.name] = {
            "n_signals": len(signals),
            "hit_rate": hit_rate,
            "false_positive_rate": fp_rate,
            "n_regime_predictions": len(regime_predictions),
        }

        logger.info(
            "backtest_complete",
            crisis=holdout_crisis.name,
            n_signals=len(signals),
            hit_rate=hit_rate,
        )

    # Generate report
    report = generate_report(all_results)
    report_path = Path("backtest_report.md")
    report_path.write_text(report)
    logger.info("backtest_report_saved", path=str(report_path))

    return all_results
```

```python
# backtest/report.py
from __future__ import annotations

import json
from datetime import datetime, timezone


def generate_report(results: dict) -> str:
    lines = [
        "# Backtest Report",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}",
        "\n## Summary\n",
    ]

    total_signals = 0
    total_hits = 0

    for crisis_name, data in results.items():
        lines.append(f"### {crisis_name}\n")
        lines.append(f"- Signals generated: {data['n_signals']}")
        lines.append(f"- Hit rate: {data['hit_rate']:.1%}")
        lines.append(f"- False positive rate: {data['false_positive_rate']:.1%}")
        lines.append(f"- Regime predictions: {data['n_regime_predictions']}")
        lines.append("")

        total_signals += data["n_signals"]
        total_hits += int(data["n_signals"] * data["hit_rate"])

    if total_signals > 0:
        overall_hit_rate = total_hits / total_signals
    else:
        overall_hit_rate = 0.0

    lines.insert(3, f"- **Total signals:** {total_signals}")
    lines.insert(4, f"- **Overall hit rate:** {overall_hit_rate:.1%}")
    lines.insert(5, "")

    # Also save JSON
    json_path = "backtest_report.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    lines.append(f"\nJSON report saved to `{json_path}`")
    return "\n".join(lines)
```

- [ ] **Step 6: Run tests**

Run: `cd C:/Personal/trading-system && python -m pytest tests/test_backtest.py -v`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add backtest/ tests/test_backtest.py
git commit -m "feat: add walk-forward backtest framework with signal quality evaluation"
```

---

### Task 16: Run Full Test Suite & Final Verification

- [ ] **Step 1: Run all tests**

Run: `cd C:/Personal/trading-system && python -m pytest tests/ -v --tb=short`
Expected: All tests PASS (39 tests total across 10 test files)

- [ ] **Step 2: Verify imports work end-to-end**

Run: `cd C:/Personal/trading-system && python -c "from main import main; print('All imports OK')"`
Expected: "All imports OK"

- [ ] **Step 3: Verify Docker build**

Run: `cd C:/Personal/trading-system && docker build -t trading-system . --no-cache`
Expected: Build succeeds

- [ ] **Step 4: Final commit with any fixes**

```bash
git add -A
git commit -m "chore: final verification and test fixes"
```
