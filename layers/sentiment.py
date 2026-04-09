"""Sentiment Engine – FinBERT scoring, Hawkes cascade detection, LLM analysis.

Task 8: Layer 2 of the trading system.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Optional

import numpy as np


# ── KeywordMatcher ──────────────────────────────────────────────────────────


class KeywordMatcher:
    """Fast regex-based keyword matching on headline text."""

    def __init__(self, keywords: list[str]) -> None:
        pattern = "|".join(re.escape(kw) for kw in keywords)
        self._regex = re.compile(pattern, re.IGNORECASE)

    def matches(self, text: str) -> bool:
        return bool(self._regex.search(text))


# ── HawkesEstimator ─────────────────────────────────────────────────────────


class HawkesEstimator:
    """Simple Hawkes process parameter estimation via method-of-moments."""

    def estimate(self, event_times: np.ndarray, window: float) -> dict:
        n = len(event_times)
        if n < 3:
            return {
                "branching_ratio": 0.0,
                "intensity": 0.0,
                "baseline": 0.0,
                "decay": 0.0,
            }

        T = window
        mean_intensity = n / T

        diffs = np.diff(event_times)
        median_diff = np.median(diffs)
        short_diffs = diffs[diffs <= median_diff]

        # Decay beta from mean of short inter-arrival times
        beta = 1.0 / np.mean(short_diffs) if len(short_diffs) > 0 and np.mean(short_diffs) > 0 else 1.0

        # Branching ratio alpha from coefficient of variation
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        if mean_diff > 0:
            cv = std_diff / mean_diff
            alpha = min(max((cv - 1) / (cv + 1), 0.0), 0.99)
        else:
            alpha = 0.0

        mu = mean_intensity * (1 - alpha)

        return {
            "branching_ratio": alpha,
            "intensity": mean_intensity,
            "baseline": mu,
            "decay": beta,
        }


# ── SentimentScorer ─────────────────────────────────────────────────────────


class SentimentScorer:
    """Score headlines with FinBERT or a neutral fallback."""

    def __init__(self, device: str = "cuda", use_finbert: bool = True) -> None:
        self._device = device
        self._use_finbert = use_finbert
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy-load FinBERT model on first use."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._device == "cuda" and not torch.cuda.is_available():
            self._device = "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self._model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self._model = self._model.to(self._device)

        if self._device == "cuda":
            self._model = self._model.half()

        self._model.eval()

    def score_headlines(self, headlines: list[str]) -> list[float]:
        """Score headlines; returns (positive_prob - negative_prob) per headline."""
        if not self._use_finbert:
            return self.score_headlines_simple(headlines)

        try:
            if self._model is None:
                self._load_model()
            return self._score_batch(headlines)
        except Exception:
            return self.score_headlines_simple(headlines)

    def _score_batch(self, headlines: list[str]) -> list[float]:
        """Run FinBERT inference in batches."""
        import torch

        scores: list[float] = []
        batch_size = 32

        for i in range(0, len(headlines), batch_size):
            batch = headlines[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                # FinBERT labels: 0=positive, 1=negative, 2=neutral
                positive = probs[:, 0].cpu().numpy()
                negative = probs[:, 1].cpu().numpy()
                scores.extend((positive - negative).tolist())

        return scores

    def score_headlines_simple(self, headlines: list[str]) -> list[float]:
        """Neutral fallback – returns 0.0 for every headline."""
        return [0.0] * len(headlines)


# ── LLM Contextual Analysis ────────────────────────────────────────────────


async def llm_contextual_analysis(
    headlines: list[str],
    provider: Optional[str],
    model: Optional[str],
    keywords: list[str],
) -> Optional[dict]:
    """Call an LLM for contextual geopolitical sentiment analysis."""
    if not provider or not model:
        return None

    top = headlines[:20]
    keyword_str = ", ".join(keywords)

    prompt = (
        f"Analyze these geopolitical headlines related to [{keyword_str}]. "
        "Return JSON with exactly these fields:\n"
        '- "escalation_score": float from -1 (de-escalation) to 1 (escalation)\n'
        '- "confidence": float from 0 to 1\n'
        '- "rationale": one sentence explaining your assessment\n\n'
        "Headlines:\n" + "\n".join(f"- {h}" for h in top)
    )

    try:
        if provider == "gemini":
            return await _call_gemini(prompt, model)
        elif provider == "openai":
            return await _call_openai(prompt, model)
        else:
            return None
    except Exception:
        return None


async def _call_gemini(prompt: str, model: str) -> Optional[dict]:
    """Call Google Gemini API for analysis."""
    import os

    from google import genai

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.models.generate_content(model=model, contents=prompt),
    )

    text = response.text
    parsed = json.loads(text)

    # Estimate cost from token usage (Gemini 2.5 Flash pricing)
    cost_usd = 0.0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
        cost_usd = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000

    return {
        "escalation_score": parsed["escalation_score"],
        "confidence": parsed["confidence"],
        "rationale": parsed["rationale"],
        "model": model,
        "cost_usd": cost_usd,
    }


async def _call_openai(prompt: str, model: str) -> Optional[dict]:
    """Call OpenAI API for analysis."""
    import openai

    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        max_tokens=256,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.choices[0].message.content
    parsed = json.loads(text)

    cost_usd = 0.0
    if response.usage:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost_usd = (input_tokens * 0.01 + output_tokens * 0.03) / 1000

    return {
        "escalation_score": parsed["escalation_score"],
        "confidence": parsed["confidence"],
        "rationale": parsed["rationale"],
        "model": model,
        "cost_usd": cost_usd,
    }


# ── Sentiment Velocity ─────────────────────────────────────────────────────


def compute_sentiment_velocity(history: list[float], window: int = 3) -> float:
    """Mean of np.diff of last `window` entries. Returns 0.0 if insufficient data."""
    if len(history) < 2:
        return 0.0
    recent = history[-(window + 1) :]  # need window+1 entries to get window diffs
    if len(recent) < 2:
        return 0.0
    diffs = np.diff(recent)
    return float(np.mean(diffs))


# ── SentimentLayer ──────────────────────────────────────────────────────────


class SentimentLayer:
    """Orchestrates keyword filtering, FinBERT scoring, Hawkes estimation, and LLM analysis."""

    def __init__(
        self,
        keywords: list[str],
        device: str = "cpu",
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        self.keywords = keywords
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.matcher = KeywordMatcher(keywords)
        self.scorer = SentimentScorer(device=device, use_finbert=(device != "cpu" or False))
        self.hawkes = HawkesEstimator()

    def filter_headlines(self, headlines: list[dict]) -> list[dict]:
        """Filter headlines by keyword matcher on the 'title' field."""
        return [h for h in headlines if self.matcher.matches(h.get("title", ""))]

    def build_output(
        self,
        sentiment_scores: list[float],
        sentiment_history: list[float],
        hawkes_result: dict,
        headline_count_15m: int,
        headline_count_1h: int,
        top_headlines: list[str],
        llm_result: Optional[dict],
        active_sources: list[str],
    ) -> dict:
        """Assemble the full sentiment output dict."""
        sentiment_mean = float(np.mean(sentiment_scores)) if sentiment_scores else 0.0
        velocity = compute_sentiment_velocity(sentiment_history)

        # Acceleration = velocity of velocity (diff of diffs)
        if len(sentiment_history) >= 3:
            diffs = list(np.diff(sentiment_history))
            acceleration = compute_sentiment_velocity(diffs)
        else:
            acceleration = 0.0

        return {
            "sentiment_mean": sentiment_mean,
            "velocity": velocity,
            "acceleration": acceleration,
            "hawkes": hawkes_result,
            "headline_count_15m": headline_count_15m,
            "headline_count_1h": headline_count_1h,
            "top_headlines": top_headlines,
            "llm_result": llm_result,
            "active_sources": active_sources,
        }
