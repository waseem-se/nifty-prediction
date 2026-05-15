"""End-to-end pipeline: scrape news -> LLM analysis -> NIFTY prediction.

Typical usage::

    from pipeline import predict_with_news

    result = predict_with_news(
        market_inputs={
            "us_market_change_pct": 0.7,
            "europe_market_change_pct": 0.3,
            "asia_market_change_pct": 0.2,
            "crude_oil_change_pct": -1.1,
            "usd_inr_change_pct": -0.1,
            "vix_change_pct": -3.0,
            "fii_flow_crore": 1400,
        },
    )

If ``OPENAI_API_KEY`` is not set, the pipeline still runs but the LLM
analysis step falls back to a keyword-based signal so callers always get
a prediction.
"""

from __future__ import annotations

import logging
import os
from typing import Mapping, Sequence

from llm_layer import (
    GeminiClient,
    LLMClient,
    NewsSignal,
    OpenAIClient,
    analyze_with_fallback,
)
from news_scraper import DEFAULT_FEEDS, NewsItem, scrape_news
from nifty_world_agent import MarketContext, NiftyNextDayAgent


logger = logging.getLogger(__name__)


def build_default_client() -> LLMClient | None:
    """Return a configured LLM client based on env vars, else ``None``.

    Preference order:
    1. ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` -> :class:`GeminiClient`
       (default model ``gemini-2.5-flash``).
    2. ``OPENAI_API_KEY`` -> :class:`OpenAIClient`
       (default model ``gpt-4o-mini``).
    """
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        try:
            return GeminiClient()
        except Exception as exc:  # pragma: no cover - depends on optional dep
            logger.warning("Could not create GeminiClient (%s); trying OpenAI.", exc)
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIClient()
        except Exception as exc:  # pragma: no cover - depends on optional dep
            logger.warning("Could not create OpenAIClient (%s); using fallback.", exc)
    return None


def predict_with_news(
    market_inputs: Mapping[str, object] | None = None,
    *,
    hours: int = 24,
    max_items: int = 40,
    feeds: Sequence[tuple[str, str]] = DEFAULT_FEEDS,
    client: LLMClient | None = None,
    items: Sequence[NewsItem] | None = None,
    agent: NiftyNextDayAgent | None = None,
) -> dict:
    """Run the full scrape -> analyze -> predict pipeline.

    Parameters
    ----------
    market_inputs:
        Optional dict of values matching :class:`MarketContext` field names.
    hours:
        Look-back window for news scraping.
    items:
        Optionally inject pre-fetched news items (skips scraping). Useful
        for tests or batch jobs that already have news in hand.
    client:
        Optional LLM client. If omitted, an :class:`OpenAIClient` is built
        from ``OPENAI_API_KEY``; if that env var is missing, the keyword
        fallback signal is used.
    """
    if items is None:
        scrape = scrape_news(feeds=feeds, lookback_hours=hours, max_items=max_items)
        items = scrape.items

    if client is None:
        client = build_default_client()

    signal: NewsSignal = analyze_with_fallback(items, client=client)

    inputs = dict(market_inputs or {})
    inputs["news_signal"] = signal
    context = MarketContext(**inputs)
    prediction = (agent or NiftyNextDayAgent()).predict(context)
    prediction["news_signal"] = {
        "overall_sentiment": signal.overall_sentiment,
        "confidence": signal.confidence,
        "summary": signal.summary,
        "source": signal.source,
        "bullish_themes": list(signal.bullish_themes),
        "bearish_themes": list(signal.bearish_themes),
    }
    prediction["news_item_count"] = len(items)
    return prediction


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    sample_inputs = {
        "us_market_change_pct": 0.7,
        "europe_market_change_pct": 0.3,
        "asia_market_change_pct": 0.2,
        "crude_oil_change_pct": -1.1,
        "usd_inr_change_pct": -0.1,
        "vix_change_pct": -3.0,
        "fii_flow_crore": 1400,
    }
    print(predict_with_news(sample_inputs))
