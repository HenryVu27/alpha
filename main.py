"""Main entry point for the trading system."""

import argparse
import asyncio
import json
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

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


async def run_signal_cycle(
    config,
    fetcher: DataFetcher,
    store: DataStore,
    redis: RedisStreamClient,
    alert_mgr: AlertManager,
    all_layers: dict,
    decision_engine: DecisionEngine,
):
    """Main 15-minute signal cycle loop."""
    logger = get_logger("signal_cycle")
    interval = config.timing.signal_update_interval_minutes * 60

    # Collect all tickers from positions + watchlist
    owned_tickers = [p.ticker for p in config.portfolio.positions]
    watchlist_tickers = [w.ticker for w in config.watchlist]
    all_tickers = owned_tickers + watchlist_tickers

    while True:
        try:
            logger.info("signal_cycle_start", tickers=all_tickers)

            # --- 1. Fetch and save current prices ---
            prices = fetcher.fetch_current_prices(all_tickers)
            for ticker, price in prices.items():
                if price is not None:
                    store.save_price(ticker, price)

            # --- 2. Get latest regime output (or default) ---
            regime_output = store.get_latest_layer_output("regime")
            if regime_output is None:
                regime_output = {"regime_label": "stalemate"}
            regime_label = regime_output.get("regime_label", "stalemate")

            # --- 3. Sentiment layer ---
            sentiment_layer: SentimentLayer = all_layers["sentiment"]
            keywords = config.geopolitical.keywords
            gdelt_headlines = fetcher.fetch_gdelt_headlines(keywords)
            newsapi_headlines = fetcher.fetch_newsapi_headlines(keywords)
            all_headlines = gdelt_headlines + newsapi_headlines

            filtered = sentiment_layer.filter_headlines(all_headlines)
            scores = sentiment_layer.scorer.score_headlines(
                [h.get("title", "") for h in filtered]
            )

            # Build sentiment history from store
            prev_sentiment = store.get_latest_layer_output("sentiment")
            sentiment_history = []
            if prev_sentiment and "sentiment_mean" in prev_sentiment:
                sentiment_history.append(prev_sentiment["sentiment_mean"])
            if scores:
                sentiment_history.append(float(sum(scores) / len(scores)))

            import numpy as np
            event_times = np.array([float(i) for i in range(len(filtered))]) if filtered else np.array([])
            hawkes_result = sentiment_layer.hawkes.estimate(
                event_times, window=max(len(filtered), 1)
            )

            top_headlines = [h.get("title", "") for h in filtered[:5]]
            active_sources = list(
                {h.get("source_type", "") for h in all_headlines}
            )

            sentiment_output = sentiment_layer.build_output(
                sentiment_scores=scores,
                sentiment_history=sentiment_history,
                hawkes_result=hawkes_result,
                headline_count_15m=len(filtered),
                headline_count_1h=len(all_headlines),
                top_headlines=top_headlines,
                llm_result=None,
                active_sources=active_sources,
            )
            store.save_layer_output("sentiment", sentiment_output)

            # --- 4. Options-implied layer ---
            options_layer: OptionsImpliedLayer = all_layers["options"]
            vix_data = fetcher.fetch_current_prices(["^VIX", "^VIX3M"])
            vix_val = vix_data.get("^VIX") or 20.0
            vix3m_val = vix_data.get("^VIX3M")

            # Try to get SKEW
            skew_data = fetcher.fetch_current_prices(["^SKEW"])
            skew_val = skew_data.get("^SKEW")

            options_output = options_layer.compute(
                vix=vix_val,
                vix3m=vix3m_val,
                skew=skew_val,
            )
            store.save_layer_output("options", options_output)

            # --- 5. Correlation layer ---
            corr_layer: CorrelationLayer = all_layers["correlation"]
            corr_assets = config.correlation.assets
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            start_date = pd.Timestamp(end_date) - pd.Timedelta(
                days=config.correlation.rolling_window_slow_days + 10
            )
            start_str = start_date.strftime("%Y-%m-%d")

            hist_data = fetcher.fetch_multiple_historical(
                corr_assets, start_str, end_date
            )
            # Build combined close-price DataFrame
            close_frames = {}
            for ticker, df in hist_data.items():
                if not df.empty and "Close" in df.columns:
                    close_frames[ticker] = df["Close"]

            if close_frames:
                corr_df = pd.DataFrame(close_frames).dropna()
                focus_pairs = [
                    tuple(p) for p in config.correlation.focus_pairs
                ]
                corr_output = corr_layer.compute(corr_df, focus_pairs=focus_pairs)
                # Serialize tuple keys for JSON
                serializable_corr = {
                    k: v
                    for k, v in corr_output.items()
                    if k not in ("lead_lag", "focus_pair_correlations")
                }
                serializable_corr["lead_lag"] = {
                    f"{a}_{b}": v
                    for (a, b), v in corr_output.get("lead_lag", {}).items()
                }
                serializable_corr["focus_pair_correlations"] = {
                    f"{a}_{b}": v
                    for (a, b), v in corr_output.get(
                        "focus_pair_correlations", {}
                    ).items()
                }
                store.save_layer_output("correlation", serializable_corr)
            else:
                corr_output = {"correlation_regime": "normal"}
                serializable_corr = corr_output

            # --- 6. Publish all outputs to Redis (if available) ---
            if redis is not None:
                try:
                    await redis.publish("regime", regime_output)
                    await redis.publish("sentiment", sentiment_output)
                    await redis.publish("options", options_output)
                    await redis.publish("correlation", serializable_corr)
                except Exception as e:
                    logger.warning("redis_publish_failed", error=str(e))

            # --- 7. Decision engine aggregation ---
            layer_outputs = {
                "regime": regime_output,
                "sentiment": {
                    "mean": sentiment_output.get("sentiment_mean", 0.0),
                    "velocity": sentiment_output.get("velocity", 0.0),
                },
                "options": options_output,
                "correlation": {
                    "corr_regime": corr_output.get(
                        "correlation_regime", "normal"
                    )
                },
                "mean_reversion": {},
                "sizing": {},
            }

            decision = decision_engine.aggregate(
                layer_outputs=layer_outputs,
                tickers=all_tickers,
            )

            # --- 8. Save decision snapshot ---
            store.save_decision_snapshot(decision)
            if redis is not None:
                try:
                    await redis.publish("decision", decision)
                except Exception:
                    pass
            logger.info(
                "decision_snapshot",
                regime=decision.get("regime_label"),
                tickers=list(decision.get("tickers", {}).keys()),
            )

            # --- 9. Check for high conviction alerts ---
            for ticker, info in decision.get("tickers", {}).items():
                conviction = info.get("conviction", {})
                conv_data = info.get("convergence", {})
                if conviction.get("conviction") == "high":
                    direction = conviction.get("direction", "neutral")
                    score = conv_data.get("score", 0)
                    agreeing = conv_data.get("agreeing", 0)
                    await alert_mgr.send_alert(
                        AlertSeverity.CRITICAL,
                        "high_conviction",
                        f"{ticker}: {direction} (score={score:.2f}, agreeing={agreeing})",
                    )

            # --- 10. Flush batched alerts ---
            await alert_mgr.flush_and_send()
            logger.info("signal_cycle_complete")

        except Exception:
            logger.exception("signal_cycle_error")

        await asyncio.sleep(interval)


async def main_async(config_path: str, backtest: bool = False, crisis: str = None):
    """Initialize all components and run the main event loop."""
    config = load_config(Path(config_path))

    setup_logging(
        level=config.logging.level,
        log_file=config.logging.file,
    )
    logger = get_logger("main")
    logger.info("system_starting", config_path=config_path, backtest=backtest)

    # --- Backtest mode ---
    if backtest:
        from backtest.replay import run_backtest

        result = run_backtest(config, crisis_name=crisis)
        logger.info("backtest_complete", result=result)
        return

    # --- Initialize components ---
    store = DataStore(config.database.path)
    store.init_db()

    redis = None
    try:
        redis = RedisStreamClient(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            stream_maxlen=config.redis.stream_maxlen,
        )
        await redis.connect()
        await redis._redis.ping()
        logger.info("redis_connected")
    except Exception as e:
        logger.warning("redis_unavailable_running_without", error=str(e))
        redis = None

    fetcher = DataFetcher()

    alert_mgr = AlertManager(
        account_sid=os.environ.get("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.environ.get("TWILIO_AUTH_TOKEN", ""),
        from_number=os.environ.get("TWILIO_WHATSAPP_FROM", ""),
        to_number=os.environ.get("TWILIO_WHATSAPP_TO", ""),
        store=store,
    )

    # --- Initialize layers ---
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

    mean_reversion_layer = MeanReversionLayer(
        half_life_suppression_days=config.alerts.half_life_suppression_days
        if hasattr(config.alerts, "half_life_suppression_days")
        else 15,
    )

    options_layer = OptionsImpliedLayer(
        vix_high=config.alerts.vix_high,
        vix_low=config.alerts.vix_low,
        skew_elevated=config.alerts.skew_elevated,
        pcr_heavy=config.alerts.pcr_heavy,
        pcr_light=config.alerts.pcr_light,
    )

    correlation_layer = CorrelationLayer(
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

    all_layers = {
        "regime": regime_detector,
        "sentiment": sentiment_layer,
        "mean_reversion": mean_reversion_layer,
        "options": options_layer,
        "correlation": correlation_layer,
        "sizing": sizing_layer,
    }

    # --- Initialize decision engine ---
    decision_engine = DecisionEngine(
        convergence_min_layers=config.alerts.convergence_min_layers,
        high_threshold=config.alerts.convergence_high_threshold,
        moderate_threshold=config.alerts.convergence_moderate_threshold,
    )

    # --- Initialize price monitor ---
    price_monitor = PriceMonitor(
        config=config,
        fetcher=fetcher,
        alert_manager=alert_mgr,
    )

    # --- Graceful shutdown ---
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        logger.info("shutdown_signal_received")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    try:
        loop.add_signal_handler(signal.SIGINT, shutdown_handler)
        loop.add_signal_handler(signal.SIGTERM, shutdown_handler)
    except NotImplementedError:
        # Windows does not support add_signal_handler
        pass

    # --- Run all loops concurrently ---
    logger.info("system_started", components=list(all_layers.keys()))
    try:
        await asyncio.gather(
            run_signal_cycle(
                config, fetcher, store, redis, alert_mgr, all_layers, decision_engine
            ),
            price_monitor.run_owned_loop(
                interval_seconds=config.timing.price_check_owned_seconds
            ),
            price_monitor.run_watchlist_loop(
                interval_seconds=config.timing.price_check_watchlist_seconds
            ),
        )
    except asyncio.CancelledError:
        logger.info("tasks_cancelled")
    finally:
        if redis is not None:
            await redis.close()
        store.close()
        logger.info("system_shutdown_complete")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Trading System")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run in backtest mode",
    )
    parser.add_argument(
        "--crisis",
        type=str,
        default=None,
        help="Crisis scenario name for backtest",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.config, backtest=args.backtest, crisis=args.crisis))


if __name__ == "__main__":
    main()
