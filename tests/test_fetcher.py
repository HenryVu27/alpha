"""Tests for data.fetcher – all mocked, no network calls."""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.fetcher import DataFetcher


@pytest.fixture
def fetcher():
    return DataFetcher()


# ── 1. fetch_current_prices ──────────────────────────────────────────────────

@patch("data.fetcher.yf")
def test_fetch_current_prices(mock_yf, fetcher):
    """Mocked yf.Ticker().info returns correct price dict."""
    mock_ticker_spy = MagicMock()
    mock_ticker_spy.info = {"regularMarketPrice": 450.0}
    mock_ticker_aapl = MagicMock()
    mock_ticker_aapl.info = {"currentPrice": 175.5}

    def _ticker(symbol):
        return {"SPY": mock_ticker_spy, "AAPL": mock_ticker_aapl}[symbol]

    mock_yf.Ticker.side_effect = _ticker

    result = fetcher.fetch_current_prices(["SPY", "AAPL"])

    assert result == {"SPY": 450.0, "AAPL": 175.5}
    # Values should be cached
    assert fetcher._last_known_prices["SPY"] == 450.0
    assert fetcher._last_known_prices["AAPL"] == 175.5


# ── 2. fetch_historical ─────────────────────────────────────────────────────

@patch("data.fetcher.yf")
def test_fetch_historical(mock_yf, fetcher):
    """Mocked yf.download returns a DataFrame."""
    expected_df = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    mock_yf.download.return_value = expected_df

    result = fetcher.fetch_historical("SPY", "2024-01-01", "2024-01-04")

    mock_yf.download.assert_called_once_with("SPY", start="2024-01-01", end="2024-01-04", progress=False)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


# ── 3. fetch_regime_features ────────────────────────────────────────────────

@patch("data.fetcher.yf")
def test_fetch_regime_features(mock_yf, fetcher):
    """fetch_regime_features returns DataFrame with renamed columns."""
    dates = pd.date_range("2024-01-01", periods=5)
    # Simulate yf.download with multiple tickers returning MultiIndex columns
    tickers = ["^VIX", "^VIX3M", "^TNX", "CL=F", "SPY", "JPY=X"]
    arrays = []
    for t in tickers:
        arrays.append(
            pd.DataFrame(
                {"Close": range(10, 15)},
                index=dates,
            )
        )

    # Build a MultiIndex DataFrame grouped by ticker
    combined = pd.concat(
        {t: df for t, df in zip(tickers, arrays)}, axis=1
    )

    mock_yf.download.return_value = combined

    result = fetcher.fetch_regime_features("2024-01-01", "2024-01-06")

    assert isinstance(result, pd.DataFrame)
    # Should have the renamed columns
    for col in ["vix", "vix3m", "yield_10y", "crude", "spy", "usdjpy"]:
        assert col in result.columns


# ── 4. fetch_gdelt_headlines ────────────────────────────────────────────────

@patch("data.fetcher.httpx")
def test_fetch_gdelt_headlines(mock_httpx, fetcher):
    """Mocked httpx.get returns GDELT articles."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "articles": [
            {
                "title": "Market rallies",
                "url": "https://example.com/1",
                "tone": 2.5,
                "seendate": "20240101T120000Z",
                "domain": "example.com",
            }
        ]
    }
    mock_httpx.get.return_value = mock_response

    result = fetcher.fetch_gdelt_headlines(["market", "rally"])

    assert len(result) == 1
    assert result[0]["title"] == "Market rallies"
    assert result[0]["source_type"] == "gdelt"
    assert result[0]["source"] == "example.com"


# ── 5. fetch_newsapi_headlines ──────────────────────────────────────────────

@patch("data.fetcher.httpx")
def test_fetch_newsapi_headlines(mock_httpx, fetcher, monkeypatch):
    """Mocked httpx.get + env var returns NewsAPI articles."""
    monkeypatch.setenv("NEWSAPI_KEY", "test-key-123")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "articles": [
            {
                "title": "Tech stocks surge",
                "url": "https://news.example.com/2",
                "publishedAt": "2024-01-02T08:00:00Z",
                "source": {"name": "ExampleNews"},
            }
        ]
    }
    mock_httpx.get.return_value = mock_response

    result = fetcher.fetch_newsapi_headlines(["tech", "stocks"])

    assert len(result) == 1
    assert result[0]["title"] == "Tech stocks surge"
    assert result[0]["source_type"] == "newsapi"
    assert result[0]["source"] == "ExampleNews"
