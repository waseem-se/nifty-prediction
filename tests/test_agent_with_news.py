import unittest
from datetime import datetime, timezone

from llm_layer import NewsSignal, parse_signal_json
from news_scraper import NewsItem
from nifty_world_agent import MarketContext, NiftyNextDayAgent
from pipeline import predict_with_news


class TestAgentWithNewsSignal(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = NiftyNextDayAgent()

    def test_bullish_news_signal_pushes_score_up(self) -> None:
        signal = NewsSignal(
            overall_sentiment=0.9, confidence=0.9, summary="Strong bullish backdrop."
        )
        baseline = self.agent.predict(MarketContext())
        with_news = self.agent.predict(MarketContext(news_signal=signal))
        self.assertGreater(with_news["score"], baseline["score"])
        self.assertEqual(with_news["movement"], "up")
        self.assertTrue(
            any("LLM news contribution" in r for r in with_news["reasons"])
        )
        self.assertTrue(any("News summary" in r for r in with_news["reasons"]))

    def test_bearish_news_signal_pushes_score_down(self) -> None:
        signal = NewsSignal(
            overall_sentiment=-0.9, confidence=0.9, summary="Risk-off."
        )
        result = self.agent.predict(MarketContext(news_signal=signal))
        self.assertEqual(result["movement"], "down")

    def test_low_confidence_signal_has_minimal_impact(self) -> None:
        signal = NewsSignal(overall_sentiment=0.9, confidence=0.05, summary="weak")
        result = self.agent.predict(MarketContext(news_signal=signal))
        self.assertEqual(result["movement"], "sideways")

    def test_backward_compatible_without_signal(self) -> None:
        # The original heuristic path must still work unchanged.
        result = self.agent.predict(
            MarketContext(macro_headlines=("rate cut expected",))
        )
        self.assertIn("movement", result)


class TestPredictWithNews(unittest.TestCase):
    def test_pipeline_with_injected_items_no_client(self) -> None:
        items = [
            NewsItem(
                title="Markets rally on rate cut hopes",
                summary="",
                source="t",
                url="https://e/1",
                published_at=datetime.now(timezone.utc),
            )
        ]
        result = predict_with_news(
            market_inputs={"us_market_change_pct": 0.5},
            items=items,
            client=None,
        )
        self.assertIn("movement", result)
        self.assertIn("news_signal", result)
        self.assertEqual(result["news_item_count"], 1)
        # No LLM client + items present => keyword fallback.
        self.assertEqual(result["news_signal"]["source"], "fallback")

    def test_pipeline_with_fake_client(self) -> None:
        from tests.test_llm_layer import FakeLLMClient

        fake = FakeLLMClient(
            {"overall_sentiment": 0.7, "confidence": 0.8, "summary": "Bullish."}
        )
        items = [
            NewsItem(
                title="x",
                summary="",
                source="t",
                url="https://e/1",
                published_at=datetime.now(timezone.utc),
            )
        ]
        result = predict_with_news(market_inputs={}, items=items, client=fake)
        self.assertEqual(result["news_signal"]["source"], "llm")
        self.assertEqual(fake.calls, 1)


if __name__ == "__main__":
    unittest.main()
