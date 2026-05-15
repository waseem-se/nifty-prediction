import json
import unittest
from datetime import datetime, timezone
from typing import Sequence

from llm_layer import (
    GeminiClient,
    LLMClient,
    NewsSignal,
    analyze_with_fallback,
    fallback_news_signal,
    parse_signal_json,
)
from news_scraper import NewsItem


def _items() -> list[NewsItem]:
    return [
        NewsItem(
            title="RBI signals possible rate cut amid cooling inflation",
            summary="Governor hints at easing.",
            source="test",
            url="https://e/1",
            published_at=datetime.now(timezone.utc),
        ),
        NewsItem(
            title="Geopolitical war fears push crude higher",
            summary="Oil up on conflict.",
            source="test",
            url="https://e/2",
            published_at=datetime.now(timezone.utc),
        ),
    ]


class FakeLLMClient(LLMClient):
    def __init__(self, payload: dict | str | Exception):
        self.payload = payload
        self.calls = 0

    def analyze_news(self, items: Sequence[NewsItem]) -> NewsSignal:
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return parse_signal_json(self.payload)


class TestParseSignalJson(unittest.TestCase):
    def test_parses_valid_payload(self) -> None:
        payload = {
            "overall_sentiment": 0.6,
            "confidence": 0.8,
            "bullish_themes": ["rate cut", "earnings"],
            "bearish_themes": ["geopolitics"],
            "key_headlines": [
                {"title": "RBI dovish", "impact": "bullish", "weight": 0.9}
            ],
            "summary": "Net bullish.",
        }
        signal = parse_signal_json(json.dumps(payload))
        self.assertAlmostEqual(signal.overall_sentiment, 0.6)
        self.assertAlmostEqual(signal.confidence, 0.8)
        self.assertEqual(signal.bullish_themes, ("rate cut", "earnings"))
        self.assertEqual(signal.key_headlines[0].impact, "bullish")
        self.assertEqual(signal.source, "llm")

    def test_clamps_out_of_range_values(self) -> None:
        signal = parse_signal_json(
            {"overall_sentiment": 5.0, "confidence": -1.0, "summary": ""}
        )
        self.assertEqual(signal.overall_sentiment, 1.0)
        self.assertEqual(signal.confidence, 0.0)

    def test_invalid_json_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_signal_json("{not-json")

    def test_missing_fields_default_safely(self) -> None:
        signal = parse_signal_json("{}")
        self.assertEqual(signal.overall_sentiment, 0.0)
        self.assertEqual(signal.bullish_themes, ())
        self.assertEqual(signal.key_headlines, ())

    def test_invalid_impact_normalized_to_neutral(self) -> None:
        signal = parse_signal_json(
            {"key_headlines": [{"title": "x", "impact": "weird", "weight": 0.5}]}
        )
        self.assertEqual(signal.key_headlines[0].impact, "neutral")


class TestFallbackSignal(unittest.TestCase):
    def test_empty_items_returns_zero_signal(self) -> None:
        signal = fallback_news_signal([])
        self.assertEqual(signal.overall_sentiment, 0.0)
        self.assertEqual(signal.source, "empty")

    def test_keyword_scoring(self) -> None:
        signal = fallback_news_signal(_items())
        # one bullish (rate cut/cooling inflation), one bearish (war)
        self.assertAlmostEqual(signal.overall_sentiment, 0.0)
        self.assertEqual(signal.source, "fallback")
        self.assertGreater(signal.confidence, 0.0)


class TestAnalyzeWithFallback(unittest.TestCase):
    def test_uses_client_when_provided(self) -> None:
        fake = FakeLLMClient(
            {"overall_sentiment": 0.5, "confidence": 0.7, "summary": "ok"}
        )
        signal = analyze_with_fallback(_items(), client=fake)
        self.assertEqual(fake.calls, 1)
        self.assertEqual(signal.source, "llm")
        self.assertAlmostEqual(signal.overall_sentiment, 0.5)

    def test_falls_back_on_client_error(self) -> None:
        fake = FakeLLMClient(RuntimeError("boom"))
        signal = analyze_with_fallback(_items(), client=fake)
        self.assertEqual(fake.calls, 1)
        self.assertEqual(signal.source, "fallback")

    def test_no_client_uses_keyword_fallback(self) -> None:
        signal = analyze_with_fallback(_items(), client=None)
        self.assertEqual(signal.source, "fallback")


class _FakeGeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeGeminiModels:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)
        return _FakeGeminiResponse(self.payload)


class _FakeGeminiSDK:
    def __init__(self, payload: str) -> None:
        self.models = _FakeGeminiModels(payload)


class TestGeminiClient(unittest.TestCase):
    def test_returns_parsed_signal(self) -> None:
        fake_sdk = _FakeGeminiSDK(
            '{"overall_sentiment": 0.4, "confidence": 0.6, "summary": "mild bull"}'
        )
        client = GeminiClient(model="gemini-2.5-flash", client=fake_sdk)
        signal = client.analyze_news(_items())
        self.assertEqual(signal.source, "llm")
        self.assertAlmostEqual(signal.overall_sentiment, 0.4)
        self.assertEqual(len(fake_sdk.models.calls), 1)
        self.assertEqual(fake_sdk.models.calls[0]["model"], "gemini-2.5-flash")
        self.assertEqual(
            fake_sdk.models.calls[0]["config"]["response_mime_type"],
            "application/json",
        )

    def test_empty_items_short_circuits(self) -> None:
        fake_sdk = _FakeGeminiSDK("{}")
        client = GeminiClient(client=fake_sdk)
        signal = client.analyze_news([])
        self.assertEqual(signal.source, "empty")
        self.assertEqual(fake_sdk.models.calls, [])

    def test_supports_gemini_3_flash_model(self) -> None:
        fake_sdk = _FakeGeminiSDK(
            '{"overall_sentiment": -0.2, "confidence": 0.5, "summary": "x"}'
        )
        client = GeminiClient(model="gemini-3-flash", client=fake_sdk)
        client.analyze_news(_items())
        self.assertEqual(fake_sdk.models.calls[0]["model"], "gemini-3-flash")


if __name__ == "__main__":
    unittest.main()
