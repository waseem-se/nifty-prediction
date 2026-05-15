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

    def test_mixed_headline_does_not_shift_nonzero_market_score(self) -> None:
        base_context = MarketContext(
            us_market_change_pct=0.6,
            europe_market_change_pct=0.4,
            asia_market_change_pct=0.2,
            crude_oil_change_pct=0.0,
            usd_inr_change_pct=0.0,
            vix_change_pct=0.0,
            fii_flow_crore=0.0,
        )
        with_mixed_headline = self.agent.predict(
            MarketContext(
                us_market_change_pct=base_context.us_market_change_pct,
                europe_market_change_pct=base_context.europe_market_change_pct,
                asia_market_change_pct=base_context.asia_market_change_pct,
                crude_oil_change_pct=base_context.crude_oil_change_pct,
                usd_inr_change_pct=base_context.usd_inr_change_pct,
                vix_change_pct=base_context.vix_change_pct,
                fii_flow_crore=base_context.fii_flow_crore,
                macro_headlines=("Ceasefire hopes amid war concerns",),
            )
        )
        without_headline = self.agent.predict(base_context)

        self.assertEqual(with_mixed_headline["score"], without_headline["score"])


if __name__ == "__main__":
    unittest.main()
