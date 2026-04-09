"""Tests for the SQLite DataStore."""

import time

import pytest

from data.store import DataStore


@pytest.fixture
def store(tmp_path):
    """Create a DataStore with a temporary database."""
    db_path = str(tmp_path / "test.db")
    ds = DataStore(db_path)
    ds.init_db()
    yield ds
    ds.close()


def test_init_db_creates_tables(store):
    """init_db creates tables — verify by saving a layer output without error."""
    store.save_layer_output("sentiment", {"score": 0.8})


def test_save_and_get_latest_layer_output(store):
    """Round-trip: save layer output then retrieve the latest."""
    store.save_layer_output("sentiment", {"score": 0.5})
    store.save_layer_output("sentiment", {"score": 0.9})

    result = store.get_latest_layer_output("sentiment")
    assert result is not None
    assert result["score"] == 0.9

    # Non-existent layer returns None
    assert store.get_latest_layer_output("missing") is None


def test_save_price_and_get_prices(store):
    """Save prices and retrieve them."""
    store.save_price("AAPL", 150.0)
    store.save_price("AAPL", 151.5)

    prices = store.get_prices("AAPL")
    assert len(prices) == 2
    assert prices[0]["price"] == 150.0
    assert prices[1]["price"] == 151.5


def test_save_alert_and_get_alerts(store):
    """Save alerts and retrieve them in reverse chronological order."""
    store.save_alert("INFO", "price_change", "AAPL moved 2%")
    store.save_alert("WARNING", "volatility", "VIX spike detected")

    alerts = store.get_alerts()
    assert len(alerts) == 2
    # Most recent first
    assert alerts[0]["severity"] == "WARNING"
    assert alerts[1]["severity"] == "INFO"


def test_save_and_get_latest_decision_snapshot(store):
    """Round-trip: save decision snapshot then retrieve the latest."""
    store.save_decision_snapshot({"action": "BUY", "confidence": 0.7})
    store.save_decision_snapshot({"action": "HOLD", "confidence": 0.4})

    result = store.get_latest_decision_snapshot()
    assert result is not None
    assert result["action"] == "HOLD"
    assert result["confidence"] == 0.4


def test_get_prices_chronological_order(store):
    """Prices are returned in chronological (ASC) order."""
    store.save_price("TSLA", 200.0)
    time.sleep(0.01)
    store.save_price("TSLA", 205.0)
    time.sleep(0.01)
    store.save_price("TSLA", 202.0)

    prices = store.get_prices("TSLA")
    assert len(prices) == 3
    assert prices[0]["price"] == 200.0
    assert prices[1]["price"] == 205.0
    assert prices[2]["price"] == 202.0
