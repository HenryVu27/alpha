"""Tests for configuration loader."""

from datetime import date
from pathlib import Path

import pytest

from common.config import load_config, AppConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@pytest.fixture
def config() -> AppConfig:
    return load_config(CONFIG_PATH)


def test_load_config_returns_app_config(config: AppConfig):
    """load_config should return a valid AppConfig instance."""
    assert isinstance(config, AppConfig)


def test_position_fields(config: AppConfig):
    """Portfolio positions should have correct ticker, shares, and entry_price."""
    dal = config.portfolio.positions[0]
    assert dal.ticker == "DAL"
    assert dal.shares == 36
    assert dal.entry_price == 69.0
    assert dal.stop_loss == 65.55
    assert dal.profit_target == 82.0


def test_watchlist(config: AppConfig):
    """Watchlist should contain VTI and XLE with buy targets."""
    tickers = [w.ticker for w in config.watchlist]
    assert "VTI" in tickers
    assert "XLE" in tickers
    vti = next(w for w in config.watchlist if w.ticker == "VTI")
    assert vti.buy_target == 333.0


def test_geopolitical_keywords(config: AppConfig):
    """Geopolitical config should contain expected keywords."""
    kw = config.geopolitical.keywords
    assert "Iran" in kw
    assert "Hormuz" in kw
    assert len(kw) == 6


def test_regime_crises_null_end(config: AppConfig):
    """Ongoing crisis (Iran 2026) should have end=None."""
    iran = next(c for c in config.regime.training_crises if c.name == "Iran 2026")
    assert iran.start == date(2026, 2, 28)
    assert iran.end is None

    ukraine = next(c for c in config.regime.training_crises if c.name == "Ukraine 2022")
    assert ukraine.end == date(2022, 5, 1)


def test_redis_config(config: AppConfig):
    """Redis config should have expected defaults."""
    assert config.redis.host == "localhost"
    assert config.redis.port == 6379
    assert config.redis.db == 0
    assert config.redis.stream_maxlen == 10000


def test_timing_config(config: AppConfig):
    """Timing config should have correct timezone and intervals."""
    assert config.timing.timezone == "America/Chicago"
    assert config.timing.signal_update_interval_minutes == 15
    assert config.timing.daily_summary_time == "16:15"


def test_all_tickers(config: AppConfig):
    """all_tickers() should return set of portfolio + watchlist tickers."""
    tickers = config.all_tickers()
    assert isinstance(tickers, set)
    assert "DAL" in tickers
    assert "NVDA" in tickers
    assert "BTC-USD" in tickers
    assert "VTI" in tickers
    assert "XLE" in tickers
    assert len(tickers) == 5
