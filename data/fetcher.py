"""DataFetcher – ingestion layer for market data, news, and regime features."""

from __future__ import annotations

import logging
import os
from math import sqrt
from typing import Optional

import httpx
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_TICKERS: dict[str, str] = {
    "^VIX": "vix",
    "^VIX3M": "vix3m",
    "^TNX": "yield_10y",
    "CL=F": "crude",
    "SPY": "spy",
    "JPY=X": "usdjpy",
}


class DataFetcher:
    """Unified data-fetching interface for the trading system."""

    def __init__(self) -> None:
        self._last_known_prices: dict[str, float] = {}

    # ── Live prices ──────────────────────────────────────────────────────

    def fetch_current_prices(
        self, tickers: list[str]
    ) -> dict[str, Optional[float]]:
        """Fetch current prices via yfinance, with graceful degradation."""
        prices: dict[str, Optional[float]] = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                price = info.get("regularMarketPrice") or info.get("currentPrice")
                if price is not None:
                    self._last_known_prices[ticker] = price
                prices[ticker] = price
            except Exception:
                logger.warning(
                    "Failed to fetch price for %s, using last known", ticker
                )
                prices[ticker] = self._last_known_prices.get(ticker)
        return prices

    # ── Historical OHLCV ────────────────────────────────────────────────

    def fetch_historical(
        self, ticker: str, start: str, end: str
    ) -> pd.DataFrame:
        """Download historical data for a single ticker."""
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            # Handle MultiIndex columns (yfinance sometimes adds ticker level)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return df
        except Exception:
            logger.exception("Failed to fetch historical data for %s", ticker)
            return pd.DataFrame()

    def fetch_multiple_historical(
        self, tickers: list[str], start: str, end: str
    ) -> dict[str, pd.DataFrame]:
        """Download historical data for multiple tickers in one call."""
        result: dict[str, pd.DataFrame] = {}
        try:
            df = yf.download(tickers, start=start, end=end, progress=False, group_by="ticker")
            if len(tickers) == 1:
                # Single ticker – no top-level ticker index
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                result[tickers[0]] = df
            else:
                for ticker in tickers:
                    try:
                        ticker_df = df[ticker].copy()
                        result[ticker] = ticker_df
                    except KeyError:
                        logger.warning("No data returned for %s", ticker)
                        result[ticker] = pd.DataFrame()
        except Exception:
            logger.exception("Failed to fetch multiple historical data")
            for ticker in tickers:
                result[ticker] = pd.DataFrame()
        return result

    # ── Regime features ─────────────────────────────────────────────────

    def fetch_regime_features(
        self,
        start: str,
        end: str,
        feature_tickers: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Fetch raw regime-feature time series."""
        if feature_tickers is None:
            feature_tickers = DEFAULT_FEATURE_TICKERS

        tickers = list(feature_tickers.keys())
        data = self.fetch_multiple_historical(tickers, start, end)

        frames: dict[str, pd.Series] = {}
        for ticker, col_name in feature_tickers.items():
            df = data.get(ticker, pd.DataFrame())
            if not df.empty and "Close" in df.columns:
                frames[col_name] = df["Close"]

        if not frames:
            return pd.DataFrame()

        result = pd.DataFrame(frames)
        result.dropna(inplace=True)
        return result

    def compute_regime_features(
        self, raw_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Derive regime-detection inputs from raw feature series."""
        computed = pd.DataFrame(index=raw_features.index)
        computed["vix"] = raw_features["vix"]
        computed["vix_term_slope"] = raw_features["vix3m"] / raw_features["vix"]
        computed["yield_10y_5d_delta"] = raw_features["yield_10y"].diff(5)
        computed["crude_5d_return"] = raw_features["crude"].pct_change(5)
        computed["spy_5d_realized_vol"] = (
            raw_features["spy"].pct_change().rolling(5).std() * sqrt(252)
        )
        computed["usdjpy_5d_delta"] = raw_features["usdjpy"].diff(5)
        computed.dropna(inplace=True)
        return computed

    # ── News / sentiment data ───────────────────────────────────────────

    def fetch_gdelt_headlines(
        self, keywords: list[str], max_records: int = 250
    ) -> list[dict]:
        """Query GDELT v2 doc API for recent headlines."""
        query = " OR ".join(keywords)
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "sort": "datedesc",
            "maxrecords": max_records,
        }
        try:
            resp = httpx.get(url, params=params, timeout=30)
            resp_data = resp.json()
            articles = resp_data.get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "url": a.get("url", ""),
                    "tone": a.get("tone", 0.0),
                    "date": a.get("seendate", ""),
                    "source": a.get("domain", ""),
                    "source_type": "gdelt",
                }
                for a in articles
            ]
        except Exception:
            logger.exception("GDELT headline fetch failed")
            return []

    def fetch_newsapi_headlines(
        self, keywords: list[str], page_size: int = 50
    ) -> list[dict]:
        """Query NewsAPI for recent headlines."""
        api_key = os.environ.get("NEWSAPI_KEY")
        if not api_key:
            logger.warning("NEWSAPI_KEY not set – skipping NewsAPI fetch")
            return []

        query = " OR ".join(keywords)
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "pageSize": page_size,
            "apiKey": api_key,
        }
        try:
            resp = httpx.get(url, params=params, timeout=30)
            resp_data = resp.json()
            articles = resp_data.get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "url": a.get("url", ""),
                    "date": a.get("publishedAt", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "source_type": "newsapi",
                }
                for a in articles
            ]
        except Exception:
            logger.exception("NewsAPI headline fetch failed")
            return []
