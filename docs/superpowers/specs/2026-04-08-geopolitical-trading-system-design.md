# Geopolitical Event-Driven Trading Signal System — Design Spec

## Overview

A real-time trading signal system for swing trading that monitors geopolitical events, market regime, cross-asset correlations, and news sentiment, then sends Telegram alerts when actionable thresholds are hit. Decision support only — no automated execution.

**Constraints:**
- Runs locally on a single machine with consumer GPU (RTX 3090/5080 class)
- Free or very cheap data sources (no Bloomberg)
- 15-minute signal update cadence
- Swing trading horizon (days to weeks)
- Must handle non-stationary, regime-switching markets
- Graceful degradation on data source failures
- Primary concern: avoid overfitting

## Architecture

### Communication: Redis Streams

Each layer runs as a standalone async Python process. Layers communicate via Redis Streams — not pub/sub — because Streams provide:
- Message persistence and replay (critical for backtest mode)
- Consumer groups for future scaling
- Guaranteed delivery (no lost messages if a consumer restarts)

Stream channels: `stream:regime`, `stream:sentiment`, `stream:mean_reversion`, `stream:options_implied`, `stream:sizing`, `stream:correlation`, `stream:decision`.

### Process Management

- `docker-compose.yml` defines all services: Redis + 6 layer processes + decision engine + price monitor + Telegram bot
- `run_all.py` for local dev: spawns everything in-process on a single event loop for easier debugging
- Each layer has its own SQLite write path for durability

### Message Schema

All inter-layer messages use a common JSON envelope:
```json
{
  "timestamp": "ISO-8601",
  "layer": "regime|sentiment|...",
  "version": 1,
  "payload": { ... }
}
```

### Backtest Mode

A replay service reads historical data from SQLite and publishes to the same Redis Stream channels at accelerated speed. Layers consume identically to live mode — same code path, different data source.

---

## Layer 1: Regime Detection

### Model: Gaussian HMM + Bayesian Online Changepoint Detection (BOCPD)

**Why both:**
- HMM classifies *which* regime you're in but flip-flops during transitions
- BOCPD detects *that* something changed but doesn't label what
- BOCPD acts as a gatekeeper: only re-evaluate HMM state when changepoint probability exceeds threshold

**Why not alternatives:**
- Markov-switching GARCH: `statsmodels` implementation fragile with >2 states, slow to fit, complexity unjustified given small training sample
- Spectral clustering on correlation matrices: no transition probabilities, no online updating — better suited for Layer 6
- Full ensemble with fusion rule: overfit risk on 3-4 crisis episodes

### Implementation Details

**HMM (via `hmmlearn`):**
- Gaussian emissions, diagonal covariance matrices (not full) to reduce parameters from ~70 to ~30 for 3 states × 6 features
- State count selected by BIC (test 3 vs 4 states)
- Features (5-6 max): VIX level, VIX term structure slope (VIX/VIX3M), 10Y yield 5d change, crude oil 5d return, SPY 5d realized vol, USD/JPY 5d change
- Training data: past geopolitical crises via `yfinance` (Ukraine 2022, Red Sea 2023, Iran 2026, etc.)
- Re-train weekly per config schedule

**BOCPD (Adams & MacKay 2007):**
- Run on a composite stress index (first principal component of the feature set)
- Constant hazard prior, λ=250 (expected run length ~1 year)
- Changepoint probability > configurable threshold (default 0.6) triggers HMM re-evaluation

### Overfitting Mitigation
- Diagonal covariance (not full) cuts parameters significantly
- 5-6 features max, each capturing a distinct risk dimension
- Leave-one-crisis-out cross-validation for regime labels
- BIC for state count selection

### Output (`stream:regime`)
```json
{
  "regime_label": "escalation",
  "regime_probabilities": [0.1, 0.2, 0.7],
  "changepoint_probability": 0.45,
  "transition_matrix": [[...], [...], [...]],
  "hmm_log_likelihood": -342.1,
  "data_staleness_seconds": 0
}
```

### Alerts
- Regime change detected
- Changepoint probability above threshold (imminent shift)

---

## Layer 2: Geopolitical NLP Sentiment Engine

### Sentiment: FinBERT + LLM Contextual Analysis

**FinBERT (`ProsusAI/finbert`)** for bulk headline scoring:
- Runs on GPU with FP16 quantization, ~500 headlines/second
- Scores every headline matching configured geopolitical keywords
- Produces sentiment mean, velocity (rate of change), acceleration

**LLM API call (Claude Opus or GPT) once per 15-min cycle:**
- Feed top 10-20 highest-impact headlines
- Returns structured JSON: escalation score (-1 to 1), confidence, one-sentence rationale
- Captures nuance FinBERT misses (e.g., "talks collapsed but parties agreed to meet again")
- Provider/model configurable in YAML, with cost tracking
- API keys from env vars: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` depending on configured provider
- ~$5/day at Opus pricing, less with GPT-4o

**Why not VADER ensemble:** VADER misreads financial/geopolitical language ("strikes", "sanctions relief"). Adds noise, not signal.

**Why not LLM-only:** Too slow for bulk scoring (1-3s per headline × 200 headlines). FinBERT handles volume, LLM handles depth.

### News Arrival: Standard Hawkes Process

- Exponential kernel, 3 parameters (baseline μ, branching ratio α, decay β)
- MLE fit on rolling 24-hour window via `tick` library (fallback: custom MLE if Windows compatibility issues)
- Key signal: branching ratio α approaching 1.0 = self-exciting cascade
- **Not** neural Hawkes: 10-50 events/hour is far too few to train a neural model

### Data Sources
- **Primary: GDELT GKG API** — free, 15-min updates, global coverage, includes tone scores as cross-check
- **Secondary: NewsAPI** — English-language headlines for FinBERT scoring. Free tier 100 req/day; aggregate keywords into 2-3 queries per cycle to stay under limit
- **Fallback:** If NewsAPI down, use GDELT tone scores as degraded sentiment signal

### Keyword Matching
- All keywords from config compiled into single regex at startup
- Exact/fuzzy string match, not semantic similarity — more predictable and auditable

### Output (`stream:sentiment`)
```json
{
  "sentiment_mean": -0.42,
  "sentiment_velocity": -0.08,
  "sentiment_acceleration": -0.02,
  "hawkes_branching_ratio": 0.73,
  "hawkes_intensity": 12.4,
  "headline_count_15m": 34,
  "headline_count_1h": 89,
  "top_headlines": ["...", "..."],
  "llm_escalation_score": -0.6,
  "llm_confidence": 0.8,
  "llm_rationale": "Ceasefire talks collapsed but ...",
  "llm_model": "claude-opus-4-6",
  "llm_cost_usd": 0.047,
  "data_sources_active": ["gdelt", "newsapi"],
  "data_staleness_seconds": 0
}
```

### Alerts
- Large sentiment shift (|velocity| > threshold)
- Rapid sentiment reversal
- Headline cascade (branching ratio > critical threshold)
- LLM escalation score extreme

---

## Layer 3: Factor-Residual Mean Reversion

### Model: Factor-Residual Ornstein-Uhlenbeck Process

**Why factor-residual, not raw OU:**
- Decompose each ticker's returns into systematic factors (market, sector, vol) and an idiosyncratic residual
- The residual mean-reverts faster and more reliably because systematic trends are stripped out
- Same parameter count as raw OU, meaningfully better signal quality

**Why not simpler approaches:**
- Bollinger z-score: special case of OU with arbitrary window length as implicit mean-reversion speed. OU estimates this from data.
- Regime-conditional ARIMA: models trend/seasonality, not relevant for deviation-from-fair-value signals

### Implementation Details

**Factor model:**
- Simple OLS regression per ticker against 2-3 factors: SPY return (market), sector ETF return, VIX change (vol)
- Re-estimate weekly
- Take residual = actual return - factor-predicted return

**OU process on residual:**
- 3 parameters per ticker per regime: θ (mean reversion speed), μ (fair value of residual), σ (volatility)
- Half-life = ln(2)/θ
- Z-score = (residual - μ) / (σ / √(2θ))
- Fair value tracked via simple rolling estimate, not Kalman filter (unless backtest shows Kalman adds value)

**Regime gating (critical):**
- Separate OU parameters per regime from Layer 1
- In crisis/escalation regime: suppress entry signals when half-life > 15 trading days (configurable)
- Exit signals always active regardless of regime
- Implementation: simple boolean gate `if regime == "crisis" and half_life > threshold: suppress_entry = True`

### Overfitting Mitigation
- OU has 3 parameters per ticker per regime. ~5 tickers × 3 regimes = 45 parameters total. Very low risk.
- Factor model is 2-3 coefficients per ticker. Re-estimated weekly.
- No Kalman filter complexity unless backtest justifies it.

### Output (`stream:mean_reversion`)
```json
{
  "ticker": "DAL",
  "fair_value": 72.3,
  "current_price": 68.5,
  "residual_z_score": -1.8,
  "half_life_days": 6.2,
  "regime": "stalemate",
  "signal_suppressed": false,
  "factors": {"market_beta": 1.2, "sector_beta": 0.8, "vol_beta": -0.3},
  "ou_params": {"theta": 0.112, "mu": 0.0, "sigma": 2.1}
}
```

### Alerts
- Large deviation from fair value (|z_score| > 2.0) — entry signal
- Z-score reverting toward 0 from extreme — exit signal
- Signal suppressed due to regime (informational)

---

## Layer 4: Options-Implied / Volatility Signals

### Approach: Pure Signal Extraction (No Model)

The options market already prices in expectations. This layer reads, not models.

**Signals:**
- **VIX term structure slope:** `VIX3M / VIX`. >1.0 = contango (normal). <1.0 = backwardation (near-term fear). Backwardation historically precedes/coincides with large drawdowns.
- **SKEW index:** OTM put demand on SPY. >150 = elevated tail hedging. Confirms regime signals.
- **Put/call ratio (equity-only):** >1.3 = heavy put buying (fear), <0.7 = complacency. Contrarian signal.

**Data:**
- VIX, VIX3M, SKEW via `yfinance` (`^VIX`, `^VIX3M`, `^SKEW`)
- Put/call ratio: scrape CBOE or `yfinance`. If unavailable, drop this signal — least important of the three.

**No model = no overfitting risk.** Thresholds are configurable in YAML, based on documented historical norms, not fitted to data.

### Output (`stream:options_implied`)
```json
{
  "vix": 27.3,
  "vix3m": 25.1,
  "term_structure_ratio": 0.92,
  "term_structure_state": "backwardation",
  "skew": 143.2,
  "put_call_ratio": 1.15,
  "signals": ["vix_elevated", "term_structure_backwardation"]
}
```

### Alerts
- VIX term structure flip (contango ↔ backwardation)
- VIX above/below configured thresholds
- SKEW above elevated threshold
- Put/call ratio at extremes

---

## Layer 5: Position Sizing

### Approach: Inverse-Volatility Sizing with Kelly Cap

**Primary: Inverse-volatility sizing.**
- Size each position proportional to `target_vol / realized_vol`
- No edge estimation required — purely risk-based
- If target portfolio vol is 15% annualized and a ticker's 20-day realized vol is 30%, size at half max allocation

**Cap: Quarter-Kelly upper bound.**
- Win probability proxied by convergence score from decision engine
- Kelly is a ceiling, not a target
- If inverse-vol says 30% but Kelly says 15%, use 15%

**Hard constraints from config override everything:**
- Max single position % of portfolio
- Max total exposure % of portfolio
- Max drawdown per position (stop loss)

**Why not pure Kelly:** Requires accurate win probability and payoff ratio, which we don't have — our estimates are noisy. Overestimating edge by 10% with full Kelly leads to catastrophic drawdowns.

**Why not risk parity:** Portfolio construction method for balanced baskets. Not applicable to directional swing trades.

### Output (`stream:sizing`)
```json
{
  "ticker": "DAL",
  "vol_sized_pct": 0.22,
  "kelly_cap_pct": 0.18,
  "final_allocation_pct": 0.18,
  "sizing_basis": "kelly_cap",
  "edge_positive": true,
  "current_position_pct": 0.25,
  "action": "slightly_oversized",
  "stop_loss_distance_pct": 0.05
}
```

### Alerts
- Strong positive edge (entry opportunity)
- Negative edge (stay out)
- Stop loss trigger
- Position oversized relative to sizing model

---

## Layer 6: Cross-Asset Correlation Monitor

### Approach: Rolling Correlation + Eigenvalue Regime Detection

**Signals:**
- **Rolling Pearson correlation matrix** on daily returns: 5-day (fast-reacting) and 21-day (smooth). Dual window gives two views.
- **Lead-lag detection:** Cross-correlation at lags -3 to +3 days for configured focus pairs. Actionable when one asset consistently leads another during a crisis.
- **Eigenvalue ratio:** `λ1 / Σλ` of correlation matrix. During crises, first eigenvalue dominates (everything correlates). Single continuous number that spikes during crisis clustering and normalizes during calm. Better than spectral clustering because it provides confirmation signal, not another regime label.
- **Correlation sign flips:** For each focus pair, track 5-day rolling correlation sign. Sign flip generates alert.

**Why not spectral clustering:** Layer 1 already handles regime labeling. This layer provides confirmation and early warning via continuous metrics, not categorical labels.

### Output (`stream:correlation`)
```json
{
  "eigenvalue_ratio": 0.61,
  "eigenvalue_ratio_21d": 0.45,
  "correlation_regime": "crisis_clustering",
  "sign_flips": [["DAL", "CL=F", "neg_to_pos"]],
  "lead_lag": [
    {"pair": ["CL=F", "DAL"], "lag_days": -1, "correlation": -0.72}
  ],
  "focus_pair_correlations": {
    "VTI_CLF": -0.65,
    "VTI_VIX": -0.81,
    "DAL_CLF": -0.58,
    "GLD_VIX": 0.32
  }
}
```

### Alerts
- Correlation sign flip on focus pairs
- Eigenvalue ratio crossing crisis/normal thresholds
- Lead-lag relationship emergence or disappearance

---

## Master Decision Engine

### Approach: Equal-Weighted Convergence Scoring

Each layer produces a directional signal per ticker: bullish (+1), bearish (-1), neutral (0).

- Market-wide layers (regime, sentiment, options, correlation) apply to all tickers with asset-class overrides (e.g., crisis = bearish equities, bullish gold)
- Ticker-specific layers (mean reversion, sizing) apply per ticker

**Convergence score** = mean of layer signals, normalized to [-1, +1].

**No adaptive weighting.** Estimating "which layer was most accurate" on 3-4 historical crises and weighting by that is overfitting. Equal weights or user-specified fixed weights.

**Thresholds:**
- `|score| >= 0.5` with 3+ layers agreeing → high conviction alert (critical)
- `|score| >= 0.3` with 2+ layers → moderate conviction (info)
- Layers disagreeing → "mixed signal" flag, no directional alert

**State snapshots:** Every 15-min cycle, full state written to SQLite as JSON blob — all layer outputs, convergence scores, alerts triggered. This is the backtest dataset and audit trail.

### Output (`stream:decision`)
```json
{
  "timestamp": "2026-04-08T10:30:00-05:00",
  "tickers": {
    "DAL": {
      "convergence_score": -0.65,
      "layers_agreeing": 4,
      "direction": "bearish",
      "conviction": "high",
      "signals": {
        "regime": -1,
        "sentiment": -1,
        "mean_reversion": 0,
        "options": -1,
        "correlation": -1,
        "sizing": {"edge_positive": false}
      }
    }
  },
  "market_wide": {
    "regime": "escalation",
    "sentiment_velocity": -0.08,
    "vix_term_structure": "backwardation",
    "correlation_regime": "crisis_clustering"
  }
}
```

---

## Backtest Framework

### Walk-Forward with Leave-One-Crisis-Out

**Protocol:**
1. Download historical data for all configured assets across all training crisis periods via `yfinance`. Store in SQLite.
2. For N crisis periods, run N iterations:
   - Train regime HMM + OU parameters on N-1 crises
   - Replay held-out crisis through full pipeline via Redis Streams (same code path as live)
   - Record all signals, convergence scores, hypothetical entry/exit points
3. Evaluate signal quality, not PnL.

**Why signal quality, not PnL:**
- PnL backtests require assumptions about execution (slippage, fill price, timing) that add fake precision
- PnL backtests are the #1 way people fool themselves into thinking a system works
- Signal quality metrics are harder to game

**Metrics:**
- **Signal hit rate:** When convergence said "bearish high conviction," did price move in that direction within the half-life window?
- **Signal lead time:** How many hours/days before a major move did the signal fire?
- **False positive rate:** High conviction signals that led to no meaningful move
- **Regime detection accuracy:** Did HMM correctly identify crisis periods on held-out data?

**Report:** JSON (programmatic) + markdown (readable), per-layer and per-crisis statistics.

**CLI:**
- `python main.py --backtest` — full walk-forward
- `python main.py --backtest --crisis "Ukraine 2022"` — single hold-out

---

## Telegram Bot

### Library: `python-telegram-bot`

**Configuration:** Bot token and chat ID from environment variables.

**Severity levels:**
- Info: routine updates, moderate conviction signals
- Warning: elevated signals, approaching thresholds
- Critical: regime changes, high conviction convergence, stop loss triggers

**Batching:**
- Non-critical alerts batched into single message per 15-min cycle
- Critical alerts sent immediately

**Daily summary:** After market close (configurable time), all layer states + day's signals.

**Commands:**
- `/set TICKER buy PRICE` — set buy target
- `/set TICKER stop PRICE` — set stop loss
- `/add TICKER` — add to watchlist
- `/remove TICKER` — remove from watchlist
- `/status` — current portfolio state + all layer signals
- `/regime` — current regime + shift probability

---

## Price Alert Module

Independent of ML layers. Basic price-level monitoring.

- Track configurable watchlist with buy targets, stop losses, profit targets
- Check prices every 60s for owned positions, every 300s for watchlist
- Update watchlist via Telegram commands
- Uses `yfinance` for price data

---

## Data Sources Summary

| Source | Data | Cost | Update Frequency |
|--------|------|------|------------------|
| yfinance | Prices, VIX, VIX3M, SKEW, historical data | Free | Real-time quotes |
| GDELT GKG API | Global events, tone scores | Free | 15 minutes |
| NewsAPI | English-language headlines | Free tier (100 req/day) | On request |
| CBOE (scrape) | Put/call ratio | Free | Daily |
| Claude/GPT API | Contextual sentiment analysis | ~$5/day | Per cycle |

---

## Error Handling & Graceful Degradation

- If any data source fails, use last known value and note staleness in all outputs and alerts
- `data_staleness_seconds` field in every layer output
- If staleness > 30 minutes, downweight that layer's signal in convergence scoring
- If Redis is down, each layer continues independently and writes to SQLite; decision engine reads from SQLite as fallback
- Structured JSON logging to file for all errors

---

## File Structure

```
trading-system/
├── config.yaml
├── DESIGN.md
├── docker-compose.yml
├── run_all.py              # in-process mode for dev/debug
├── main.py                 # entry point for single-process mode
├── requirements.txt
├── layers/
│   ├── regime.py
│   ├── sentiment.py
│   ├── mean_reversion.py
│   ├── options_implied.py
│   ├── sizing.py
│   └── correlation.py
├── engine/
│   ├── decision.py
│   ├── alerts.py
│   └── price_monitor.py
├── data/
│   ├── fetcher.py
│   └── store.py
├── common/
│   ├── config.py           # YAML config loader
│   ├── redis_client.py     # Redis Streams wrapper
│   └── schema.py           # message envelope + validation
├── backtest/
│   ├── replay.py           # historical data replay service
│   ├── evaluator.py        # signal quality metrics
│   └── report.py           # report generation
├── models/                 # saved model weights/params
├── tests/
├── logs/
└── docs/
```

---

## Tech Stack

- Python 3.11+
- Redis (Streams)
- Docker + Docker Compose
- PyTorch + transformers (FinBERT, FP16, CUDA)
- hmmlearn (HMM)
- tick (Hawkes process)
- yfinance (market data)
- numpy, scipy, pandas, scikit-learn
- python-telegram-bot
- anthropic / openai SDK (LLM sentiment)
- SQLite (persistence)
- pydantic (config + message validation)
- structlog (JSON logging)

---

## Build Order

1. DESIGN.md — model selection justifications (this document, reformatted)
2. `config.yaml` — full default config
3. `common/` — config loader, Redis client, message schema
4. `data/fetcher.py` + `data/store.py` — data pipeline
5. `engine/alerts.py` + `engine/price_monitor.py` — Telegram + price alerts (immediately useful)
6. `layers/regime.py` — regime detection
7. `layers/sentiment.py` — NLP + news clustering
8. `layers/mean_reversion.py` — factor-residual fair value
9. `layers/options_implied.py` — vol signals
10. `layers/correlation.py` — cross-asset monitor
11. `layers/sizing.py` — position sizing
12. `engine/decision.py` — master aggregation
13. `main.py` + `run_all.py` + `docker-compose.yml` — orchestration
14. `backtest/` — walk-forward framework
15. `tests/` — test suite
