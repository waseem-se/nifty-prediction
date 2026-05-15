"""Rule-based next-day NIFTY 50 movement prediction agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_layer import NewsSignal


@dataclass(frozen=True)
class MarketContext:
    """Inputs capturing global context likely to affect NIFTY 50."""

    us_market_change_pct: float = 0.0
    europe_market_change_pct: float = 0.0
    asia_market_change_pct: float = 0.0
    crude_oil_change_pct: float = 0.0
    usd_inr_change_pct: float = 0.0
    vix_change_pct: float = 0.0
    fii_flow_crore: float = 0.0
    macro_headlines: tuple[str, ...] = ()
    news_signal: Optional["NewsSignal"] = None


class NiftyNextDayAgent:
    """Simple explainable agent that predicts next-day NIFTY direction."""

    MIN_CONFIDENCE = 35.0
    MAX_CONFIDENCE = 95.0
    BASE_CONFIDENCE = 50.0
    CONFIDENCE_MULTIPLIER = 12.0
    # Weight applied to the LLM-derived news signal when present. Tuned so
    # a high-confidence, strong sentiment (e.g. +0.8 sentiment * 0.9 conf)
    # contributes ~1.1 to score — comparable to a single macro factor.
    NEWS_SIGNAL_MULTIPLIER = 1.5

    POSITIVE_KEYWORDS = {
        "rate cut",
        "cooling inflation",
        "ceasefire",
        "stimulus",
        "strong earnings",
        "trade deal",
    }
    NEGATIVE_KEYWORDS = {
        "rate hike",
        "hot inflation",
        "war",
        "sanctions",
        "banking stress",
        "recession",
    }

    def predict(self, context: MarketContext) -> dict[str, object]:
        score = 0.0
        reasons: list[str] = []

        world_indices_score = (
            context.us_market_change_pct
            + context.europe_market_change_pct
            + context.asia_market_change_pct
        ) * 0.8
        score += world_indices_score
        if world_indices_score:
            reasons.append(f"Global equity momentum contribution: {world_indices_score:.2f}")

        oil_score = -0.5 * context.crude_oil_change_pct
        score += oil_score
        if context.crude_oil_change_pct:
            reasons.append(f"Crude oil impact contribution: {oil_score:.2f}")

        fx_score = -1.2 * context.usd_inr_change_pct
        score += fx_score
        if context.usd_inr_change_pct:
            reasons.append(f"USD/INR impact contribution: {fx_score:.2f}")

        vix_score = -0.4 * context.vix_change_pct
        score += vix_score
        if context.vix_change_pct:
            reasons.append(f"Volatility (VIX) contribution: {vix_score:.2f}")

        fii_score = 0.001 * context.fii_flow_crore
        score += fii_score
        if context.fii_flow_crore:
            reasons.append(f"FII flow contribution: {fii_score:.2f}")

        headline_score = self._headline_score(context.macro_headlines)
        score += headline_score
        if headline_score:
            reasons.append(f"Headline sentiment contribution: {headline_score:.2f}")

        if context.news_signal is not None:
            news_contribution = (
                context.news_signal.overall_sentiment
                * context.news_signal.confidence
                * self.NEWS_SIGNAL_MULTIPLIER
            )
            score += news_contribution
            reasons.append(
                f"LLM news contribution: {news_contribution:.2f} "
                f"(sentiment={context.news_signal.overall_sentiment:+.2f}, "
                f"confidence={context.news_signal.confidence:.2f}, "
                f"source={context.news_signal.source})"
            )
            if context.news_signal.summary:
                reasons.append(f"News summary: {context.news_signal.summary}")

        if score > 1.0:
            movement = "up"
        elif score < -1.0:
            movement = "down"
        else:
            movement = "sideways"

        confidence = min(
            self.MAX_CONFIDENCE,
            max(self.MIN_CONFIDENCE, self.BASE_CONFIDENCE + abs(score) * self.CONFIDENCE_MULTIPLIER),
        )

        return {
            "movement": movement,
            "score": round(score, 3),
            "confidence": round(confidence, 1),
            "reasons": reasons,
        }

    def _headline_score(self, headlines: Iterable[str]) -> float:
        score = 0.0
        for headline in headlines:
            normalized = headline.lower()
            has_positive = any(token in normalized for token in self.POSITIVE_KEYWORDS)
            has_negative = any(token in normalized for token in self.NEGATIVE_KEYWORDS)
            if has_positive and not has_negative:
                score += 0.8
            elif has_negative and not has_positive:
                score -= 0.8
        return score


if __name__ == "__main__":
    sample_context = MarketContext(
        us_market_change_pct=0.7,
        europe_market_change_pct=0.3,
        asia_market_change_pct=0.2,
        crude_oil_change_pct=-1.1,
        usd_inr_change_pct=-0.1,
        vix_change_pct=-3.0,
        fii_flow_crore=1400,
        macro_headlines=("US inflation cooling", "Middle East ceasefire talks advance"),
    )
    prediction = NiftyNextDayAgent().predict(sample_context)
    print("Predicted NIFTY next-day movement:", prediction)
