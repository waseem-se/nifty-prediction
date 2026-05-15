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
