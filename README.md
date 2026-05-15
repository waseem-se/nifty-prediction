# nifty-prediction

A lightweight rule-based agent that predicts **next-day NIFTY 50 movement** (`up`, `down`, or `sideways`) based on global market context.

## What it analyzes
- US, Europe, and Asia market moves
- Crude oil movement
- USD/INR movement
- Volatility trend (VIX)
- FII flows
- Macro/world headlines sentiment

## Quick usage
```python
from nifty_world_agent import MarketContext, NiftyNextDayAgent

context = MarketContext(
    us_market_change_pct=0.7,
    europe_market_change_pct=0.3,
    asia_market_change_pct=0.2,
    crude_oil_change_pct=-1.1,
    usd_inr_change_pct=-0.1,
    vix_change_pct=-3.0,
    fii_flow_crore=1400,
    macro_headlines=("US inflation cooling", "Ceasefire talks advance"),
)

prediction = NiftyNextDayAgent().predict(context)
print(prediction)
```

Run the built-in tests:
```bash
python -m unittest discover -s tests -q
```

## News + LLM layer (optional)

You can now augment predictions with a **news web-scraping + LLM** pipeline that turns recent market headlines into a structured sentiment signal which feeds into the same agent.

### Recommended model

**Default: `gpt-4o-mini` (OpenAI)** — cheap (~$0.15 / 1M input tokens), fast, supports strict JSON output, and good enough for headline-level financial sentiment. This is the model we recommend you take a subscription / pay-as-you-go for.

Alternatives wired through the same `LLMClient` interface:
- `gemini-1.5-flash` (Google) — generous free tier, 1M-token context.
- `gpt-4o` or `claude-3-5-sonnet` — higher quality, higher cost. Use only if `gpt-4o-mini` quality isn't sufficient.

### Setup
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

### Usage
```python
from pipeline import predict_with_news

result = predict_with_news({
    "us_market_change_pct": 0.7,
    "europe_market_change_pct": 0.3,
    "asia_market_change_pct": 0.2,
    "crude_oil_change_pct": -1.1,
    "usd_inr_change_pct": -0.1,
    "vix_change_pct": -3.0,
    "fii_flow_crore": 1400,
})
print(result)
```

### What happens under the hood
1. **Scrape** — `news_scraper.py` pulls fresh headlines from Moneycontrol, ET Markets, Livemint, Reuters, and Yahoo Finance RSS feeds. Items are deduped, filtered to the last 24 hours, capped at 40, and cached on disk for 30 minutes.
2. **Analyze** — `llm_layer.py` sends the headlines to `gpt-4o-mini` with a strict JSON-mode prompt and parses the response into a `NewsSignal` (`overall_sentiment`, `confidence`, themes, key headlines, summary).
3. **Predict** — the existing `NiftyNextDayAgent` consumes the `NewsSignal` as an additional, weighted input and includes the LLM summary in the explanation.

### Graceful degradation
If `OPENAI_API_KEY` is not set, or the LLM call fails, the pipeline falls back to a keyword-based sentiment signal so a prediction is always returned. The original keyword-only path (`MarketContext(macro_headlines=...)`) continues to work unchanged.

### Swapping providers
Implement a new subclass of `LLMClient` (see the `GeminiClient` stub in `llm_layer.py`) and pass it as `client=` to `predict_with_news(...)`.
