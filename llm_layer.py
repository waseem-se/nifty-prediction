"""LLM layer that converts scraped news into a structured market signal.

The layer exposes a small, pluggable :class:`LLMClient` interface so that
swapping providers (OpenAI, Gemini, Anthropic) is a one-class change.
Default implementation: :class:`OpenAIClient` using ``gpt-4o-mini`` —
a cheap, fast model that handles JSON-mode output well for this task.

Design goals:
- Strict JSON output via the model's ``response_format`` JSON mode.
- Deterministic parsing into :class:`NewsSignal`.
- Graceful fallback: if the LLM call fails, callers can use
  :func:`fallback_news_signal` to compute a keyword-based signal so the
  prediction agent never crashes.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from news_scraper import NewsItem


logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 2

SYSTEM_PROMPT = (
    "You are a financial markets analyst focused on the Indian NIFTY 50 index. "
    "You read recent news headlines and short summaries and assess their "
    "likely impact on NIFTY 50 over the next trading session. "
    "Always reply with strict JSON matching the requested schema. "
    "Do not include commentary outside JSON."
)

USER_PROMPT_TEMPLATE = """\
Analyse the following news items and return a JSON object with this exact schema:
{{
  "overall_sentiment": number between -1.0 (very bearish) and 1.0 (very bullish),
  "confidence": number between 0.0 and 1.0,
  "bullish_themes": [string, ...],
  "bearish_themes": [string, ...],
  "key_headlines": [
    {{"title": string, "impact": "bullish"|"bearish"|"neutral", "weight": number 0..1}}
  ],
  "summary": "2-3 sentence explanation of the net market impact"
}}

News items (newest first):
{news_block}
"""


# --- Bullish / bearish keyword fallback (used when the LLM is unavailable) ---

_POSITIVE_KEYWORDS = {
    "rate cut", "cooling inflation", "ceasefire", "stimulus",
    "strong earnings", "trade deal", "rally", "record high",
}
_NEGATIVE_KEYWORDS = {
    "rate hike", "hot inflation", "war", "sanctions",
    "banking stress", "recession", "selloff", "crash",
}


@dataclass(frozen=True)
class HeadlineImpact:
    title: str
    impact: str  # "bullish" | "bearish" | "neutral"
    weight: float = 0.5


@dataclass(frozen=True)
class NewsSignal:
    """Structured output of the LLM layer."""

    overall_sentiment: float  # -1..1
    confidence: float  # 0..1
    summary: str = ""
    bullish_themes: tuple[str, ...] = ()
    bearish_themes: tuple[str, ...] = ()
    key_headlines: tuple[HeadlineImpact, ...] = ()
    source: str = "llm"  # "llm" | "fallback" | "empty"


def _format_news_block(items: Sequence[NewsItem]) -> str:
    lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        when = item.published_at.strftime("%Y-%m-%d %H:%M UTC")
        summary = (item.summary or "").strip()
        if len(summary) > 400:
            summary = summary[:400].rstrip() + "..."
        lines.append(
            f"{idx}. [{item.source} | {when}] {item.title}"
            + (f"\n   {summary}" if summary else "")
        )
    return "\n".join(lines) if lines else "(no news items)"


def _clamp(value: float, low: float, high: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(low, min(high, v))


def parse_signal_json(payload: str | dict) -> NewsSignal:
    """Parse a raw LLM JSON payload into a :class:`NewsSignal`.

    Tolerant of missing / malformed fields — anything unparseable becomes
    a sane default rather than raising.
    """
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc
    else:
        data = payload or {}

    headlines_raw = data.get("key_headlines") or []
    headlines: list[HeadlineImpact] = []
    for item in headlines_raw:
        if not isinstance(item, dict):
            continue
        impact = str(item.get("impact", "neutral")).lower()
        if impact not in {"bullish", "bearish", "neutral"}:
            impact = "neutral"
        headlines.append(
            HeadlineImpact(
                title=str(item.get("title", ""))[:300],
                impact=impact,
                weight=_clamp(item.get("weight", 0.5), 0.0, 1.0),
            )
        )

    def _str_tuple(value: object) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()
        return tuple(str(v) for v in value if isinstance(v, (str, int, float)))

    return NewsSignal(
        overall_sentiment=_clamp(data.get("overall_sentiment", 0.0), -1.0, 1.0),
        confidence=_clamp(data.get("confidence", 0.0), 0.0, 1.0),
        summary=str(data.get("summary", "")).strip(),
        bullish_themes=_str_tuple(data.get("bullish_themes")),
        bearish_themes=_str_tuple(data.get("bearish_themes")),
        key_headlines=tuple(headlines),
        source="llm",
    )


def fallback_news_signal(items: Sequence[NewsItem]) -> NewsSignal:
    """Compute a keyword-based signal when the LLM is unavailable.

    Mirrors the heuristic used by the original rule-based agent so callers
    always get a usable :class:`NewsSignal`.
    """
    if not items:
        return NewsSignal(
            overall_sentiment=0.0,
            confidence=0.0,
            summary="No news available.",
            source="empty",
        )

    pos = neg = 0
    headlines: list[HeadlineImpact] = []
    for item in items:
        text = f"{item.title} {item.summary}".lower()
        has_pos = any(token in text for token in _POSITIVE_KEYWORDS)
        has_neg = any(token in text for token in _NEGATIVE_KEYWORDS)
        if has_pos and not has_neg:
            pos += 1
            headlines.append(HeadlineImpact(item.title, "bullish", 0.5))
        elif has_neg and not has_pos:
            neg += 1
            headlines.append(HeadlineImpact(item.title, "bearish", 0.5))

    total_signal = pos + neg
    if total_signal == 0:
        sentiment = 0.0
        confidence = 0.1
    else:
        sentiment = (pos - neg) / total_signal
        # Confidence grows with sample size but caps modestly — keyword
        # matching is a weak signal compared to the LLM.
        confidence = min(0.5, 0.1 + 0.05 * total_signal)

    return NewsSignal(
        overall_sentiment=sentiment,
        confidence=confidence,
        summary=(
            f"Keyword fallback over {len(items)} headlines: "
            f"{pos} bullish vs {neg} bearish matches."
        ),
        key_headlines=tuple(headlines[:10]),
        source="fallback",
    )


# ----------------------------- Client interface -----------------------------


class LLMClient(ABC):
    """Abstract LLM client. Implementations must return a NewsSignal."""

    @abstractmethod
    def analyze_news(self, items: Sequence[NewsItem]) -> NewsSignal:
        ...


class OpenAIClient(LLMClient):
    """OpenAI-backed LLM client.

    Defaults to ``gpt-4o-mini``: cheap, fast, supports strict JSON output,
    and good enough for headline-level financial sentiment. Switch to
    ``gpt-4o`` only if signal quality is insufficient.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client: object | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        if client is not None:
            self._client = client
            return

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it or pass api_key explicitly."
            )
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - dep-dependent
            raise RuntimeError(
                "The 'openai' package is required for OpenAIClient. "
                "Install with: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=key, timeout=timeout)

    def analyze_news(self, items: Sequence[NewsItem]) -> NewsSignal:
        if not items:
            return NewsSignal(
                overall_sentiment=0.0,
                confidence=0.0,
                summary="No news available.",
                source="empty",
            )

        user_prompt = USER_PROMPT_TEMPLATE.format(news_block=_format_news_block(items))
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(  # type: ignore[attr-defined]
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                content = response.choices[0].message.content  # type: ignore[index]
                return parse_signal_json(content or "{}")
            except Exception as exc:  # pragma: no cover - network-dependent
                last_error = exc
                logger.warning(
                    "OpenAI call failed (attempt %s/%s): %s",
                    attempt + 1, self.max_retries + 1, exc,
                )
        raise RuntimeError(f"OpenAI request failed: {last_error}")


class GeminiClient(LLMClient):
    """Google Gemini-backed LLM client.

    Defaults to ``gemini-2.5-flash`` (GA, fast, generous context window,
    free tier available). Pass ``model="gemini-3-flash"`` to use the newer
    Gemini 3 Flash model when available on your account. Uses the
    unified ``google-genai`` SDK with JSON-mode output.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client: object | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        if client is not None:
            self._client = client
            return

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get(
            "GEMINI_API_KEY"
        )
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. "
                "Export it or pass api_key explicitly."
            )
        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - dep-dependent
            raise RuntimeError(
                "The 'google-genai' package is required for GeminiClient. "
                "Install with: pip install google-genai"
            ) from exc
        self._client = genai.Client(api_key=key)

    def analyze_news(self, items: Sequence[NewsItem]) -> NewsSignal:
        if not items:
            return NewsSignal(
                overall_sentiment=0.0,
                confidence=0.0,
                summary="No news available.",
                source="empty",
            )

        prompt = (
            SYSTEM_PROMPT
            + "\n\n"
            + USER_PROMPT_TEMPLATE.format(news_block=_format_news_block(items))
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.models.generate_content(  # type: ignore[attr-defined]
                    model=self.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.2,
                    },
                )
                content = getattr(response, "text", None) or ""
                return parse_signal_json(content or "{}")
            except Exception as exc:  # pragma: no cover - network-dependent
                last_error = exc
                logger.warning(
                    "Gemini call failed (attempt %s/%s): %s",
                    attempt + 1, self.max_retries + 1, exc,
                )
        raise RuntimeError(f"Gemini request failed: {last_error}")


def analyze_with_fallback(
    items: Sequence[NewsItem],
    client: LLMClient | None = None,
) -> NewsSignal:
    """Run the LLM analysis but fall back to keyword scoring on any error."""
    if client is None:
        return fallback_news_signal(items)
    try:
        return client.analyze_news(items)
    except Exception as exc:
        logger.warning("LLM analysis failed, using keyword fallback: %s", exc)
        return fallback_news_signal(items)
