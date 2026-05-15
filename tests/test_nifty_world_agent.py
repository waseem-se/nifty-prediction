import unittest

from nifty_world_agent import MarketContext, NiftyNextDayAgent


class TestNiftyNextDayAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = NiftyNextDayAgent()

    def test_predicts_up_when_global_context_is_supportive(self) -> None:
        result = self.agent.predict(
            MarketContext(
                us_market_change_pct=1.2,
                europe_market_change_pct=0.8,
                asia_market_change_pct=0.5,
                crude_oil_change_pct=-2.0,
                usd_inr_change_pct=-0.3,
                vix_change_pct=-5.0,
                fii_flow_crore=2200,
                macro_headlines=("Possible ceasefire and strong earnings",),
            )
        )

        self.assertEqual(result["movement"], "up")
        self.assertGreaterEqual(result["confidence"], 60.0)

    def test_predicts_down_when_risk_signals_dominate(self) -> None:
        result = self.agent.predict(
            MarketContext(
                us_market_change_pct=-1.5,
                europe_market_change_pct=-1.0,
                asia_market_change_pct=-1.2,
                crude_oil_change_pct=3.0,
                usd_inr_change_pct=0.6,
                vix_change_pct=8.0,
                fii_flow_crore=-3000,
                macro_headlines=("War and recession fears with hot inflation",),
            )
        )

        self.assertEqual(result["movement"], "down")
        self.assertGreaterEqual(result["confidence"], 60.0)

    def test_predicts_sideways_for_mixed_signals(self) -> None:
        result = self.agent.predict(
            MarketContext(
                us_market_change_pct=0.1,
                europe_market_change_pct=-0.1,
                asia_market_change_pct=0.0,
                crude_oil_change_pct=0.2,
                usd_inr_change_pct=0.0,
                vix_change_pct=0.2,
                fii_flow_crore=50,
                macro_headlines=("Markets await central bank cues",),
            )
        )

        self.assertEqual(result["movement"], "sideways")

    def test_mixed_sentiment_headline_is_treated_as_neutral(self) -> None:
        result = self.agent.predict(
            MarketContext(
                macro_headlines=("Ceasefire hopes amid war concerns",),
            )
        )

        self.assertEqual(result["movement"], "sideways")
        self.assertEqual(result["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
