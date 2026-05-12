# Hydration Simulation Artifacts

Use this reference when a research task needs context from the previous Hydration economic research workflow in this repository.

## Repository Shape

Observed May 12, 2026:

- 111 Jupyter notebooks.
- Core Python model under `hydradx/model/`.
- AMM primitives under `hydradx/model/amm/`.
- 17 Python test files under `hydradx/tests/`.
- Legacy cadCAD work under `old_model/`.
- Production data workflows using Hydration RPC, `hydradx-api`, `substrate-interface`, Web3, Binance/Kraken-style order books, SQLPad, cached chain state, and historical Omnipool data.

## Main Prior Research Areas

- Omnipool math, liquidity, LRNA accounting, swaps, add/remove liquidity, and token onboarding.
- Dynamic fees and fee parameter changes.
- StableSwap amplification, pegs, repegging, imbalance behavior, and rebasing-token behavior.
- HOLLAR/HSM stability mechanics.
- Money-market liquidation and HDX price-drop scenarios.
- Routing through Omnipool and StableSwap subpools.
- Arbitrage, LP behavior, impermanent loss, LP returns, and price divergence.
- Slippage, order batching, and XYK vs Omnipool comparisons.
- Manipulation/stress scenarios such as toxic assets, depegs, price manipulation, HDX crash, AUSD attack, and FTT crash.

## Important Model Components

- `hydradx/model/amm/omnipool_amm.py`: `OmnipoolState`, `DynamicFee`, Omnipool swaps, add/remove liquidity, fees, LRNA, oracles, limits.
- `hydradx/model/amm/stableswap_amm.py`: `StableSwapPoolState`, D invariant, amplification, pegs, target pegs, fees, liquidity operations.
- `hydradx/model/hollar.py`: `StabilityModule`, HOLLAR buy/sell flows, buyback speed, max buy price, HSM/stableswap arbitrage.
- `hydradx/model/amm/money_market.py`: `MoneyMarket`, `CDP`, `MoneyMarketAsset`, liquidations, health factor, thresholds, close factor.
- `hydradx/model/amm/omnipool_router.py`: routing through Omnipool and StableSwap subpools.
- `hydradx/model/amm/trade_strategies.py`: random/steady/scheduled swaps, LP strategies, arbitrage, toxic asset attack, price manipulation, liquidations.
- `hydradx/model/amm/global_state.py`: global state, price shocks, historical price processes, OTC settlement.
- `hydradx/model/processing.py`: production/current/historical chain and market data loading.

## Notebook Groups

- `hydradx/spec/`: mathematical specs for Omnipool, Swap, SwapLRNA, AddLiquidity, WithdrawLiquidity, AddToken, plus algebraic checks.
- `hydradx/notebooks/Omnipool/`: dynamic fees, LP analysis, arbitrage, LRNA imbalance, HDX crash, manipulation profitability, historical arbitrage.
- `hydradx/notebooks/Stableswap/`: amplification, peg manipulation, repegging, rebasing tokens, impermanent loss/gain.
- `hydradx/notebooks/Money Market/`: CDP data, HDX liquidations, HDX price drop.
- `analysis/`: slippage, order batching, XYK comparison.
- `hydradx/derivations/`: impermanent loss and optimal arbitrage derivations.
- `hydradx/notebooks/Hackathon/`: AUSD attack and FTT crash scenarios.

## Tests As Economic Assertions

Use existing tests as examples for what research outputs should eventually encode:

- HOLLAR/HSM: constructor validation, supported collateral, buy/sell flows, buyback limits, max buy price, fees, insufficient liquidity, max liquidity, same-block peg updates, arbitrage loops, rebalance cases.
- Money market: CDP validation, liquidation eligibility, full liquidation threshold, liquidation amount, partial/full liquidation, undercollateralized toxic debt, fuzzed LTV/penalty/threshold combinations, borrow/repay/withdraw, save/load.
- Stableswap: invariant behavior, swap/add/remove, fees, peg updates, repegging arbitrage, LP outcomes, constructor failures, arbitrary peg/fee fuzz cases.
- Omnipool: constructor/oracle/fee behavior, cash-out valuation, dynamic fees, swap/add/remove, routing, arbitrage, DCA-with-LPing, exploit/manipulation scenarios, production-state save/load.

## Research Rule

When giving parameter recommendations, also state:

- What model would justify them.
- What data should calibrate them.
- What scenarios should be swept.
- What metrics define success/failure.
- What economic assertions should become tests.
