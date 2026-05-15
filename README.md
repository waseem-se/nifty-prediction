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

You can use either provider — pick whichever you have access to:

- **`gemini-2.5-flash` (Google)** — recommended default if you have a Google AI Studio key. Fast, cheap, generous free tier, ~1M-token context. Set `model="gemini-3-flash"` to use Gemini 3 Flash when available on your account.
- **`gpt-4o-mini` (OpenAI)** — strong alternative (~$0.15 / 1M input tokens), reliable JSON mode.
- **`gpt-4o` / `claude-3-5-sonnet`** — higher quality, higher cost. Use only if Flash/mini quality isn't sufficient.

The pipeline auto-selects a client at runtime: it prefers **Gemini** if `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) is set, then falls back to **OpenAI** if `OPENAI_API_KEY` is set, and finally to a keyword-only signal if neither is configured.

### Setup
```bash
pip install -r requirements.txt

# Option A: Gemini (recommended)
export GOOGLE_API_KEY=...

# Option B: OpenAI
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
2. **Analyze** — `llm_layer.py` sends the headlines to the configured LLM (Gemini 2.5/3 Flash or `gpt-4o-mini`) with a strict JSON-mode prompt and parses the response into a `NewsSignal` (`overall_sentiment`, `confidence`, themes, key headlines, summary).
3. **Predict** — the existing `NiftyNextDayAgent` consumes the `NewsSignal` as an additional, weighted input and includes the LLM summary in the explanation.

### Graceful degradation
If neither `GOOGLE_API_KEY` nor `OPENAI_API_KEY` is set, or the LLM call fails, the pipeline falls back to a keyword-based sentiment signal so a prediction is always returned. The original keyword-only path (`MarketContext(macro_headlines=...)`) continues to work unchanged.

### Swapping providers
Both `OpenAIClient` and `GeminiClient` implement the same `LLMClient` interface — pass either as `client=` to `predict_with_news(...)`, or implement your own subclass for Anthropic/etc.
