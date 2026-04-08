"""Configuration loader with pydantic validation."""

from datetime import date
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class PositionConfig(BaseModel):
    ticker: str
    shares: Optional[int] = None
    entry_price: Optional[float] = None
    amount_usd: Optional[float] = None
    stop_loss: float
    profit_target: float


class WatchlistItem(BaseModel):
    ticker: str
    buy_target: float


class GeopoliticalConfig(BaseModel):
    keywords: list[str]
    gdelt_update_interval_minutes: int
    llm_provider: str
    llm_model: str
    llm_daily_cost_cap_usd: float


class CrisisConfig(BaseModel):
    name: str
    start: date
    end: Optional[date] = None


class RegimeConfig(BaseModel):
    n_states: int
    retrain_schedule: str
    features: list[str]
    training_crises: list[CrisisConfig]
    bocpd_hazard_lambda: int
    state_labels: dict[int, str]


class AlertsConfig(BaseModel):
    regime_shift_threshold: float
    sentiment_strong_threshold: float
    sentiment_velocity_threshold: float
    branching_ratio_critical: float
    vix_high: float
    vix_low: float
    skew_elevated: float
    pcr_heavy: float
    pcr_light: float
    convergence_min_layers: int
    convergence_high_threshold: float
    convergence_moderate_threshold: float
    z_score_entry_threshold: float
    half_life_suppression_days: int


class SizingConfig(BaseModel):
    kelly_fraction: float
    target_annual_vol: float
    max_single_position_pct: float
    max_total_exposure_pct: float
    max_drawdown_per_position_pct: float
    realized_vol_window_days: int


class CorrelationConfig(BaseModel):
    rolling_window_days: int
    rolling_window_slow_days: int
    eigenvalue_crisis_threshold: float
    assets: list[str]
    focus_pairs: list[list[str]]
    lead_lag_max_days: int


class MeanReversionConfig(BaseModel):
    factor_tickers: dict[str, str]
    sector_map: dict[str, str]
    recalibration_schedule: str


class RedisConfig(BaseModel):
    host: str
    port: int
    db: int
    stream_maxlen: int


class TimingConfig(BaseModel):
    signal_update_interval_minutes: int
    price_check_owned_seconds: int
    price_check_watchlist_seconds: int
    daily_summary_time: str
    timezone: str
    staleness_warning_seconds: int


class LoggingConfig(BaseModel):
    level: str
    file: str
    format: str


class DatabaseConfig(BaseModel):
    path: str


class PortfolioConfig(BaseModel):
    cash: float
    positions: list[PositionConfig]


class AppConfig(BaseModel):
    portfolio: PortfolioConfig
    watchlist: list[WatchlistItem]
    geopolitical: GeopoliticalConfig
    regime: RegimeConfig
    alerts: AlertsConfig
    sizing: SizingConfig
    correlation: CorrelationConfig
    mean_reversion: MeanReversionConfig
    redis: RedisConfig
    timing: TimingConfig
    logging: LoggingConfig
    database: DatabaseConfig

    def all_tickers(self) -> set[str]:
        """Return set of all tickers from portfolio positions and watchlist."""
        tickers: set[str] = set()
        for pos in self.portfolio.positions:
            tickers.add(pos.ticker)
        for item in self.watchlist:
            tickers.add(item.ticker)
        return tickers


def load_config(path: Path) -> AppConfig:
    """Load and validate configuration from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
