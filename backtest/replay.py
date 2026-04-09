"""Walk-forward leave-one-crisis-out backtest replay."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from backtest.evaluator import SignalEvaluator
from backtest.report import generate_report
from common.config import AppConfig
from data.fetcher import DataFetcher
from engine.decision import DecisionEngine
from layers.options_implied import OptionsImpliedLayer
from layers.regime import RegimeDetector

logger = logging.getLogger(__name__)


async def run_backtest(config: AppConfig, crisis_name: str | None = None) -> dict:
    """Run leave-one-crisis-out walk-forward backtest.

    For each holdout crisis:
      - Train on all other crises
      - Walk through holdout data running regime detector + decision engine
      - Record and evaluate signals

    Args:
        config: Application configuration with training_crises list.
        crisis_name: If specified, only backtest this crisis as holdout.

    Returns:
        Dict with per-crisis results and summary.
    """
    fetcher = DataFetcher()
    evaluator = SignalEvaluator()
    crises = config.regime.training_crises

    if crisis_name:
        crises_to_test = [c for c in crises if c.name == crisis_name]
        if not crises_to_test:
            raise ValueError(f"Crisis '{crisis_name}' not found in config")
    else:
        crises_to_test = list(crises)

    all_results: dict = {"crises": {}}

    for holdout in crises_to_test:
        logger.info("Backtesting holdout crisis: %s", holdout.name)

        # Training crises = all except holdout
        training_crises = [c for c in crises if c.name != holdout.name]

        # Build regime detector trained on other crises
        detector = RegimeDetector(
            n_states=config.regime.n_states,
            hazard_lambda=config.regime.bocpd_hazard_lambda,
            state_labels=config.regime.state_labels,
        )

        # Fetch and concatenate training data
        from datetime import date as date_type
        training_features_list = []
        for tc in training_crises:
            end_date = tc.end or date_type.today()
            # Ensure start < end
            if tc.start >= end_date:
                logger.warning("Skipping crisis %s: start >= end", tc.name)
                continue
            try:
                raw = fetcher.fetch_regime_features(
                    start=str(tc.start),
                    end=str(end_date),
                )
                if raw.empty:
                    logger.warning("No data for training crisis %s", tc.name)
                    continue
                computed = fetcher.compute_regime_features(raw)
                if not computed.empty:
                    training_features_list.append(computed.values)
            except Exception:
                logger.exception("Failed to fetch data for crisis %s", tc.name)
                continue

        if not training_features_list:
            logger.warning(
                "No training data available for holdout %s, skipping", holdout.name
            )
            all_results["crises"][holdout.name] = {
                "num_signals": 0,
                "hit_rate": None,
                "false_positive_rate": None,
                "regime_predictions_count": 0,
            }
            continue

        training_data = np.concatenate(training_features_list, axis=0)
        detector.fit(training_data)

        # Fetch holdout crisis data
        holdout_end = holdout.end or date_type.today()
        try:
            raw_holdout = fetcher.fetch_regime_features(
                start=str(holdout.start),
                end=str(holdout_end),
            )
            if raw_holdout.empty:
                logger.warning("No holdout data for %s", holdout.name)
                all_results["crises"][holdout.name] = {
                    "num_signals": 0,
                    "hit_rate": None,
                    "false_positive_rate": None,
                    "regime_predictions_count": 0,
                }
                continue
            holdout_features = fetcher.compute_regime_features(raw_holdout)
        except Exception:
            logger.exception("Failed to fetch holdout data for %s", holdout.name)
            all_results["crises"][holdout.name] = {
                "num_signals": 0,
                "hit_rate": None,
                "false_positive_rate": None,
                "regime_predictions_count": 0,
            }
            continue

        # Walk through holdout data
        decision_engine = DecisionEngine(
            convergence_min_layers=config.alerts.convergence_min_layers,
            high_threshold=config.alerts.convergence_high_threshold,
            moderate_threshold=config.alerts.convergence_moderate_threshold,
        )
        options_layer = OptionsImpliedLayer(
            vix_high=config.alerts.vix_high,
            vix_low=config.alerts.vix_low,
            skew_elevated=config.alerts.skew_elevated,
            pcr_heavy=config.alerts.pcr_heavy,
            pcr_light=config.alerts.pcr_light,
        )

        signals_recorded: list[dict] = []
        regime_predictions: list[str] = []
        holdout_values = holdout_features.values

        # Use SPY close as price proxy if available in raw data
        if "spy" in raw_holdout.columns:
            # Align with computed features (which drops NaN rows)
            spy_prices = raw_holdout["spy"].loc[holdout_features.index].values
        else:
            # Fallback: use first feature column
            spy_prices = holdout_values[:, 0]

        min_window = max(10, config.regime.n_states * 2)

        for t in range(min_window, len(holdout_values)):
            window = holdout_values[: t + 1]

            try:
                regime_result = detector.detect(window)
            except Exception:
                continue

            regime_label = regime_result["regime_label"]
            regime_predictions.append(regime_label)

            # Build minimal layer outputs for decision engine
            vix_val = float(holdout_values[t, 0]) if holdout_values.shape[1] > 0 else 20.0
            options_result = options_layer.compute(vix=vix_val)

            layer_outputs = {
                "regime": regime_result,
                "sentiment": {"mean": 0.0, "velocity": 0.0},
                "options": options_result,
                "correlation": {"corr_regime": "normal"},
                "mean_reversion": {},
                "sizing": {},
            }

            tickers = list(config.all_tickers())
            agg = decision_engine.aggregate(layer_outputs, tickers)

            # Record signals when conviction is high or moderate
            for ticker, ticker_data in agg.get("tickers", {}).items():
                conviction = ticker_data.get("conviction", {})
                if conviction.get("conviction") in ("high", "moderate"):
                    signals_recorded.append(
                        {
                            "timestamp_idx": t,
                            "half_life": config.alerts.half_life_suppression_days,
                            "direction": conviction.get("direction", "neutral"),
                            "ticker": ticker,
                            "conviction": conviction.get("conviction"),
                            "regime": regime_label,
                        }
                    )

        # Evaluate signals
        if signals_recorded and len(spy_prices) > 0:
            hit_rate = evaluator.compute_hit_rate(signals_recorded, spy_prices)
            fp_rate = evaluator.compute_false_positive_rate(
                signals_recorded, spy_prices
            )
        else:
            hit_rate = None
            fp_rate = None

        all_results["crises"][holdout.name] = {
            "num_signals": len(signals_recorded),
            "hit_rate": hit_rate,
            "false_positive_rate": fp_rate,
            "regime_predictions_count": len(regime_predictions),
            "signals": signals_recorded,
        }

        logger.info(
            "Crisis %s: %d signals, hit_rate=%s, fp_rate=%s",
            holdout.name,
            len(signals_recorded),
            f"{hit_rate:.2%}" if hit_rate is not None else "N/A",
            f"{fp_rate:.2%}" if fp_rate is not None else "N/A",
        )

    # Generate report
    report_md = generate_report(all_results)

    report_path = Path("backtest_report.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    logger.info("Backtest report saved to %s", report_path)

    return all_results
