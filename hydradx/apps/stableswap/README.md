# Stableswap Streamlit Apps

## EUR/USD Arbitrage Smoothing

This app loads EUR/USD prices from Binance and Kraken for a date range and
builds a smoothed series by choosing the value closest to the last accepted
price at each Binance timestamp.

### Run

```zsh
streamlit run hydradx/apps/stableswap/eur_usd_arbitrage_sim.py
```

### Notes

- Data is cached per day under `hydradx/apps/stableswap/cached data/`.
- The app also caches loaded ranges in-memory via Streamlit.

