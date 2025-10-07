import copy
import math

import pytest
from hypothesis import given, strategies as st, reproduce_failure
from mpmath import mp, mpf

import os
os.chdir('../..')

from hydradx.model.hollar import StabilityModule, fast_hollar_arb_and_dump, get_hollar_sell_amount
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.stableswap_amm import StableSwapPoolState

mp.dps = 50

def test_stability_module_constructor():
    # stability module should work with this params
    liquidity = {'USDT': 1_000_000, 'SUSDS': 1_000_000}
    buyback_speed = 1/10_000
    usdt_pool = StableSwapPoolState(tokens={'USDT': 1_000_000, 'HOLLAR': 1_000_000}, amplification=1000, trade_fee=0.0001)
    susds_price = 1.05
    susds_pool = StableSwapPoolState(tokens={'HOLLAR': 1_000_000, 'SUSDS': 1_000_000/susds_price}, amplification=1000,
                                     trade_fee=0.0001, peg=susds_price)
    pools = [usdt_pool, susds_pool]
    sell_fee = 0.0002
    max_buy_price_coef = 0.999
    buy_fee = 0.0001
    native_stable = 'HOLLAR'
    StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee, native_stable)

    # stability module should allow differing params for different assets
    usdt_buyback_speed = 1/10_000
    susds_buyback_speed = 1/11_000
    buyback_speed = [usdt_buyback_speed, susds_buyback_speed]
    sell_fee = [0.0001, 0.0002]
    usdt_max_buy_price_coef = 0.999
    susds_max_buy_price_coef = 0.998
    max_buy_price_coef = [usdt_max_buy_price_coef, susds_max_buy_price_coef]
    usdt_buy_fee = 0.0001
    usdc_buy_fee = 0.0002
    buy_fee = [usdt_buy_fee, usdc_buy_fee]
    StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee, native_stable)

    # stability module should fail if buyback_speed is not in [0, 1]
    bad_buyback_speed = 1.1
    with pytest.raises(ValueError):
        StabilityModule(liquidity, bad_buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee, native_stable)
    bad_buyback_speed = -0.1
    with pytest.raises(ValueError):
        StabilityModule(liquidity, bad_buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee, native_stable)

    # stability module should fail if sell_fee is < 0
    bad_sell_fee = -.0001
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, pools, bad_sell_fee, max_buy_price_coef, buy_fee, native_stable)

    # stability module should fail if max_buy_price_coef is not in (0, 1]
    bad_max_buy_price_coef = 1.1
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, pools, sell_fee, bad_max_buy_price_coef, buy_fee, native_stable)
    bad_max_buy_price_coef = 0
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, pools, sell_fee, bad_max_buy_price_coef, buy_fee, native_stable)

    # stability module should fail if buy_fee is not in [0, 1]
    bad_buy_fee = 1.1
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, bad_buy_fee, native_stable)
    bad_buy_fee = -0.1
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, bad_buy_fee, native_stable)

    # stability module should fail if native_stablecoin is in liquidity
    bad_native_stable = 'USDT'
    with pytest.raises(AssertionError):
        StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee, bad_native_stable)

    # stability module should fail if a pool does not have correct stablecoin
    bad_usdt_pool = StableSwapPoolState(tokens={'aUSDT': 1_000_000, 'HOLLAR': 1_000_000}, amplification=1000, trade_fee=0.0001)
    bad_pools = [bad_usdt_pool, susds_pool]
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, bad_pools, sell_fee, max_buy_price_coef, buy_fee, native_stable)

    # stability module should fail if a pool does not have native stablecoin
    bad_usdt_pool = StableSwapPoolState(tokens={'USDT': 1_000_000, 'USDC': 1_000_000}, amplification=1000, trade_fee=0.0001)
    bad_pools = [bad_usdt_pool, susds_pool]
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, bad_pools, sell_fee, max_buy_price_coef, buy_fee, native_stable)


def test_overlapping_tokens():
    three_pool_tokens = {'A': 1_000_000, 'HOLLAR': 1_000_000, 'B': 1_000_000}
    three_pool = StableSwapPoolState(tokens=three_pool_tokens, amplification=1000)
    pools = [three_pool, three_pool]

    liquidity = {'A': 1_000_000, 'B': 1_000_000}
    sell_fee = 0.001
    max_buy_price_coef = 0.999
    buy_fee = 0.0001
    native_stable = 'HOLLAR'
    usdt_buyback_speed = 1/10_000
    usdc_buyback_speed = 1/11_000
    buyback_speed = [usdt_buyback_speed, usdc_buyback_speed]
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee, native_stable)

    pool1 = StableSwapPoolState(tokens=three_pool_tokens, amplification=1000)
    pool2 = StableSwapPoolState(tokens=three_pool_tokens, amplification=1000)
    pools = [pool1, pool2]

    StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee, native_stable)


@given(
    ratios = st.lists(st.floats(min_value=0.01, max_value=0.1), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/1_000_000, max_value=1),
    max_buy_price_coef = st.floats(min_value=0.99, max_value=1),
    buy_fee = st.floats(min_value=0, max_value=0.01),
    sell_ratio = st.floats(min_value=0.01, max_value=1),
    susds_price = st.floats(min_value=1, max_value=2)
)
def test_sell_hollar_to_stability_module(ratios, buyback_speed, max_buy_price_coef, buy_fee, sell_ratio, susds_price):
    liquidity = {'USDT': 1_000_000, 'SUSDS': 1_000_000}
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100,
                                    trade_fee=0.0001, precision=1e-8)
    susds_pool = StableSwapPoolState(tokens={'SUSDS': ratios[1] * 1_000_000/susds_price, 'HOLLAR': 1_000_000},
                                     amplification=100, trade_fee=0.0001, peg=1/susds_price, precision=1e-8)
    pools = [usdt_pool, susds_pool]

    sell_fee = 0.001

    tkn_sell = 'HOLLAR'
    init_hsm = StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee)

    for buy_tkn_i, tkn_buy in enumerate(liquidity.keys()):
        pool_buy = pools[buy_tkn_i]
        max_sell_amt, buy_price = init_hsm.get_buy_params(tkn_buy)
        assert max_sell_amt > 0  # this test case focuses on parameters in which HSM can buy HOLLAR
        peg = init_hsm.get_peg(tkn_buy)
        liq_diff = pool_buy.liquidity['HOLLAR'] - pool_buy.liquidity[tkn_buy] * peg
        if max_sell_amt != pytest.approx(buyback_speed * liq_diff / 2, rel=1e-15):
            raise ValueError(f"Expected {buyback_speed * liq_diff / 2}, got {max_sell_amt}")

        # test sell Hollar to HSM
        stability_module = copy.deepcopy(init_hsm)
        sell_amt = max_sell_amt * sell_ratio
        agent = Agent(holdings = {tkn_sell: sell_amt})
        stability_module.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=sell_amt)
        if agent.validate_holdings(tkn_sell):
            raise ValueError("Agent should have 0 holdings after selling")
        buy_amt = agent.get_holdings(tkn_buy)
        exec_price = buy_amt / sell_amt
        if exec_price > max_buy_price_coef:
            raise ValueError("Execution price should be less than max buy price")
        # execution price should be higher than pool price of closing arb
        pool_sell_amt = pool_buy.calculate_sell_from_buy(tkn_buy=tkn_sell, tkn_sell=tkn_buy, buy_quantity=sell_amt)
        if (pool_sell_amt - buy_amt)/pool_buy.liquidity[tkn_buy] > 1e-12:
            raise AssertionError("Stability module not giving arbitragable prices")
        if abs(buy_amt - (init_hsm.liquidity[tkn_buy] - stability_module.liquidity[tkn_buy])) / stability_module.liquidity[tkn_buy] >= 1e-15:
            raise ValueError("Stability module did not deduct correct amount of tokens")

        # test buy USDT from HSM
        buy_amt *= 0.99999  # make sure to stay under max buy amount
        sell_amt *= 0.99999
        buy_hsm = copy.deepcopy(init_hsm)
        buy_agent = Agent(enforce_holdings=False)
        buy_hsm.swap(buy_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_amt)
        if buy_agent.get_holdings(tkn_buy) != buy_amt:
            raise ValueError("Agent has incorrect amount of buy token")
        if abs(-buy_agent.get_holdings(tkn_sell) - sell_amt)/sell_amt >= 1e-15:
            raise ValueError("Agent sold incorrect amount")
        if abs(buy_amt - (init_hsm.liquidity[tkn_buy] - buy_hsm.liquidity[tkn_buy])) / buy_hsm.liquidity[tkn_buy] >= 1e-15:
            raise ValueError("Stability module did not deduct correct amount of tokens")


@given(
    ratios = st.lists(st.floats(min_value=0.999, max_value=1.001), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/1_000_000, max_value=1),
    max_buy_price_coef = st.floats(min_value=0.99, max_value=1, exclude_max=True),
    buy_fee = st.floats(min_value=0, max_value=0.01),
    susds_price = st.floats(min_value=1, max_value=2)
)
def test_sell_hollar_fails_when_balanced(ratios, buyback_speed, max_buy_price_coef, buy_fee, susds_price):
    liquidity = {'USDT': 1_000_000, 'SUSDS': 1_000_000}
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100,
                                    trade_fee=0.0001, precision=1e-8)
    susds_pool = StableSwapPoolState(tokens={'SUSDS': ratios[1] * 1_000_000/susds_price, 'HOLLAR': 1_000_000},
                                     amplification=100, trade_fee=0.0001, peg=1/susds_price, precision=1e-8)
    pools = [usdt_pool, susds_pool]
    sell_fee = 0.001
    tkn_sell = 'HOLLAR'
    init_hsm = StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee)

    for buy_tkn_i, tkn_buy in enumerate(liquidity.keys()):
        max_sell_amt, buy_price = init_hsm.get_buy_params(tkn_buy)
        if max_sell_amt != 0:
            raise ValueError("Should not be able to sell HOLLAR when pools are balanced")

        # test sell Hollar to HSM
        hsm = copy.deepcopy(init_hsm)
        sell_amt = 10
        agent = Agent(holdings = {tkn_sell: sell_amt})
        hsm.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=sell_amt)
        if not hsm.fail:
            raise ValueError("Should not be able to sell HOLLAR when pools are balanced")

        # test buy USDT from HSM
        buy_hsm = copy.deepcopy(init_hsm)
        buy_agent = Agent(enforce_holdings=False)
        buy_hsm.swap(buy_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=10)
        if not buy_hsm.fail:
            raise ValueError("Should not be able to buy USDT when pools are balanced")


@given(
    ratios = st.lists(st.floats(min_value=0.01, max_value=0.1), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/10_000, max_value=1),
    max_buy_price_coef = st.floats(min_value=0.99, max_value=1),
    buy_fee = st.floats(min_value=0.00001, max_value=0.01),
    susds_price = st.floats(min_value=1, max_value=2),
)
def test_sell_hollar_to_stability_module_fee(ratios, buyback_speed, max_buy_price_coef, buy_fee, susds_price):
    liquidity = {'USDT': 1_000_000, 'SUSDS': 1_000_000}
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100,
                                    trade_fee=0.0001, precision=1e-8)
    susds_pool = StableSwapPoolState(tokens={'HOLLAR': 1_000_000, 'SUSDS': ratios[1] * 1_000_000/susds_price},
                                     amplification=100, trade_fee=0.0001, peg=susds_price, precision=1e-8)
    sell_fee = 0.001
    tkn_sell = 'HOLLAR'
    for buy_tkn_i, tkn_buy in enumerate(liquidity.keys()):
        pools = [usdt_pool.copy(), susds_pool.copy()]
        hsm = StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee)
        pool_buy = pools[buy_tkn_i]
        max_sell_amt, buy_price = hsm.get_buy_params(tkn_buy)
        if max_sell_amt <= 0:  # this test case focuses on parameters in which stability module can buy HOLLAR
            raise AssertionError("Stability module should be able to buy HOLLAR")
        agent = Agent(holdings = {tkn_sell: max_sell_amt})
        hsm.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=max_sell_amt)
        buy_amt = agent.get_holdings(tkn_buy)
        # execution price should be higher than pool price of closing arb
        pool_sell_amt = pool_buy.calculate_sell_from_buy(tkn_buy=tkn_sell, tkn_sell=tkn_buy, buy_quantity=max_sell_amt)
        if (pool_sell_amt - buy_amt)/pool_sell_amt > 1e-15:
            raise AssertionError("Stability module not giving arbitragable prices")
        # execution price should only be higher than pool by fee, since we are trading max_sell_amt
        if (buy_amt * (1 - buy_fee) - pool_sell_amt)/pool_sell_amt > 1e-15:
            raise AssertionError("Stability module not giving correct buy fee")


@given(
    ratios = st.lists(st.floats(min_value=0.01, max_value=0.1), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/1_000_000, max_value=1),
    sell_fee = st.floats(min_value=0, max_value=0.1),
    susds_price=st.floats(min_value=1, max_value=2),
)
def test_buy_hollar_from_stability_module(ratios, buyback_speed, sell_fee, susds_price):
    # stability module should work with this params
    liquidity = {'USDT': 1_000_000, 'SUSDS': 1_000_000}
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=1000,
                                    trade_fee=0.0001)
    susds_pool = StableSwapPoolState(tokens={'HOLLAR': 1_000_000, 'SUSDS': ratios[1] * 1_000_000/susds_price},
                                     amplification=100, trade_fee=0.0001, peg=susds_price)
    tkn_buy = 'HOLLAR'
    max_buy_price_coef = 0.999
    buy_fee = 0.0001
    for tkn_sell in liquidity.keys():
        pools = [usdt_pool.copy(), susds_pool.copy()]
        init_hsm = StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee)
        hsm = copy.deepcopy(init_hsm)
        # test sell tkn_sell
        sell_amt = 1_000
        agent = Agent(holdings = {tkn_sell: sell_amt})
        hsm.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=sell_amt)
        buy_amt = agent.get_holdings(tkn_buy)
        peg = hsm.get_peg(tkn_sell)
        if agent.validate_holdings(tkn_sell):
            raise ValueError("Agent should have 0 holdings after selling")
        if agent.get_holdings(tkn_buy) != pytest.approx(sell_amt / (1+sell_fee) * peg, rel=1e-15):
            raise ValueError("Agent has incorrect amount of HOLLAR")
        if abs(sell_amt - (hsm.liquidity[tkn_sell] - init_hsm.liquidity[tkn_sell])) / hsm.liquidity[tkn_sell] >= 1e-15:
            raise ValueError("Stability module did not deduct correct amount of tokens")

        # test buy HOLLAR
        buy_hsm = copy.deepcopy(init_hsm)
        buy_agent = Agent(enforce_holdings=False)
        buy_hsm.swap(buy_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_amt)
        if buy_agent.get_holdings(tkn_buy) != buy_amt:
            raise ValueError("Agent has incorrect amount of HOLLAR")
        if abs(-buy_agent.get_holdings(tkn_sell) - sell_amt)/sell_amt >= 1e-15:
            raise ValueError("Agent sold incorrect amount")
        if abs(sell_amt - (hsm.liquidity[tkn_sell] - init_hsm.liquidity[tkn_sell])) / buy_hsm.liquidity[tkn_sell] >= 1e-15:
            raise ValueError("Stability module did not deduct correct amount of tokens")


@given(
    ratios = st.lists(st.floats(min_value=0.01, max_value=1), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/1_000_000, max_value=1),
    max_buy_price_coef = st.floats(min_value=0.99, max_value=1),
    sell_extra = st.floats(min_value=1e-15, max_value=10),
    susds_price = st.floats(min_value=1, max_value=2),
    n = st.integers(min_value=1, max_value=10),
)
def test_large_trade_fails(ratios, buyback_speed, max_buy_price_coef, sell_extra, susds_price, n):
    liquidity = {'USDT': 1_000_000, 'SUSDS': 1_000_000}
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100,
                                    trade_fee=0.0001, precision=1e-8)
    susds_pool = StableSwapPoolState(tokens={'HOLLAR': 1_000_000, 'SUSDS': ratios[1] * 1_000_000/susds_price},
                                     amplification=100, trade_fee=0.0001, peg=susds_price)
    sell_fee = 0.001
    tkn_sell = 'HOLLAR'
    buy_fee = 0.0001
    for buy_tkn_i, tkn_buy in enumerate(liquidity.keys()):
        pools = [usdt_pool.copy(), susds_pool.copy()]
        hsm = StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee)

        max_sell_amt, buy_price = hsm.get_buy_params(tkn_buy)
        sell_amt = max_sell_amt * (1 + sell_extra)
        sell_amt_n = sell_amt / n
        agent = Agent(holdings={tkn_sell: sell_amt})
        for i in range(n):
            hsm.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=sell_amt_n)
        if not hsm.fail:
            raise AssertionError("Swap should have failed")


@given(
    ratios = st.lists(st.floats(min_value=0.01, max_value=1), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/1_000_000, max_value=1),
    max_buy_price_coef = st.floats(min_value=0.99, max_value=1),
    susds_price = st.floats(min_value=1, max_value=2),
    liquidity = st.lists(st.floats(min_value=1, max_value=1_000_000), min_size=2, max_size=2),
)
def test_insufficient_liquidity(ratios, buyback_speed, max_buy_price_coef, susds_price, liquidity):
    liquidity = {'USDT': liquidity[0], 'SUSDS': liquidity[1]}
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100,
                                    trade_fee=0.0001, precision=1e-8)
    susds_pool = StableSwapPoolState(tokens={'HOLLAR': 1_000_000, 'SUSDS': ratios[1] * 1_000_000/susds_price},
                                     amplification=100, trade_fee=0.0001, peg=susds_price)
    sell_fee = 0.001
    buy_fee = 0.0001
    for buy_tkn_i, tkn in enumerate(liquidity.keys()):
        pools = [usdt_pool.copy(), susds_pool.copy()]
        hsm = StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee)
        max_sell_amt, buy_price = hsm.get_buy_params(tkn)
        if (max_sell_amt * buy_price - hsm.liquidity[tkn]) / hsm.liquidity[tkn] > 1e-15:
            raise ValueError("Stability module should not be able to sell more than it has")


@given(
    ratios = st.lists(st.floats(min_value=0.01, max_value=0.1), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/10_000, max_value=1),
    max_buy_price_coef = st.floats(min_value=0.999, max_value=1),
    susds_price = st.floats(min_value=1, max_value=2),
    buy_fee = st.floats(min_value=0.0001, max_value=0.01),
)
def test_arb_loop_known_profitable(ratios, buyback_speed, max_buy_price_coef, susds_price, buy_fee):
    sell_fee = 0.001

    liquidity = {'USDT': 1_000_000, 'SUSDS': 1_000_000}
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100,
                                    trade_fee=0.0001, precision=1e-8)
    susds_pool = StableSwapPoolState(tokens={'HOLLAR': 1_000_000, 'SUSDS': ratios[1] * 1_000_000/susds_price},
                                     amplification=100, trade_fee=0.0001, peg=susds_price)
    for buy_tkn_i, tkn in enumerate(liquidity.keys()):
        pools = [usdt_pool.copy(), susds_pool.copy()]
        hsm = StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee)

        agent = Agent()
        hsm.arb(agent, tkn)
        if not agent.validate_holdings(tkn):
            raise ValueError("Agent should have positive USDT holdings after arb")
        if agent.validate_holdings('HOLLAR'):
            raise ValueError("Agent should have 0 HOLLAR holdings after arb")


@given(
    ratios = st.lists(st.floats(min_value=0.01, max_value=1), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/1_000_000, max_value=1),
    max_buy_price_coef = st.floats(min_value=0.99, max_value=1),
    susds_price = st.floats(min_value=1, max_value=2),
    buy_fee = st.floats(min_value=0, max_value=0.01),
)
def test_arb_loop_known(ratios, buyback_speed, max_buy_price_coef, susds_price, buy_fee):
    sell_fee = 0.001

    liquidity = {'USDT': 1_000_000, 'SUSDS': 1_000_000}
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * 1_000_000, 'HOLLAR': 1_000_000}, amplification=100,
                                    trade_fee=0.0001, precision=1e-8)
    susds_pool = StableSwapPoolState(tokens={'HOLLAR': 1_000_000, 'SUSDS': ratios[1] * 1_000_000 / susds_price},
                                     amplification=100, trade_fee=0.0001, peg=susds_price)
    for buy_tkn_i, tkn in enumerate(liquidity.keys()):
        pools = [usdt_pool.copy(), susds_pool.copy()]
        hsm = StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee)

        # signature of arb function is arb(self, agent: Agent, tkn: str) -> None:
        agent = Agent()
        hsm.arb(agent, tkn)
        # agent should have no HOLLAR after arb
        if agent.validate_holdings('HOLLAR'):
            raise ValueError("Agent should have 0 HOLLAR holdings after arb")

@given(
    ratios = st.lists(st.floats(min_value=0.01, max_value=0.5), min_size=2, max_size=2),
    buyback_speed = st.floats(min_value=1/1_000_000, max_value=1),
    susds_price = st.floats(min_value=1, max_value=2),
)
def test_swap_does_not_change_params(ratios, buyback_speed, susds_price):
    liquidity = {'USDT': mpf(1_000_000), 'SUSDS': mpf(1_000_000)}
    usdt_pool = StableSwapPoolState(tokens={'USDT': ratios[0] * mpf(1_000_000), 'HOLLAR': mpf(1_000_000)},
                                    amplification=100, trade_fee=0.0001, precision=1e-8)
    susds_pool = StableSwapPoolState(tokens={'HOLLAR': 1_000_000, 'SUSDS': ratios[1] * 1_000_000 / susds_price},
                                     amplification=100, trade_fee=0.0001, peg=susds_price)
    pools = [usdt_pool, susds_pool]
    sell_fee = 0.001
    max_buy_price_coef = 0.999
    tkn_sell = 'HOLLAR'
    buy_fee = 0.0001
    hsm = StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee)
    for tkn in ['USDT', 'SUSDS']:
        if tkn == 'USDT':
            pool = usdt_pool
        else:
            pool = susds_pool
        max_buy_amt, buy_price = hsm.get_buy_params(tkn)
        agent = Agent(enforce_holdings=False)
        sell_quantity = max_buy_amt / 2
        hsm.swap(agent, tkn_buy=tkn, tkn_sell=tkn_sell, sell_quantity=sell_quantity)
        pool.swap(agent, tkn_buy=tkn_sell, tkn_sell=tkn, buy_quantity=sell_quantity)
        if hsm.fail or pool.fail:
            raise ValueError("Swap failed")
        max_buy_amt2, buy_price2 = hsm.get_buy_params(tkn)
        remaining_capacity = max_buy_amt - sell_quantity
        if max_buy_amt2 != pytest.approx(remaining_capacity, rel=1e-20):
            raise ValueError("Max buy amount changed")
        if buy_price2 != buy_price:
            raise ValueError("Buy price changed")
        # check that update resets block
        hsm.update()
        if hsm.fail:
            raise ValueError("Update failed")
        max_buy_amt3, buy_price3 = hsm.get_buy_params(tkn)
        if max_buy_amt3 >= max_buy_amt:
            raise ValueError("Max buy amount should be less than before swap")
        if max_buy_amt3 <= max_buy_amt2:
            raise ValueError("Max buy amount should be greater than after swap but before update")
        if buy_price3 <= buy_price:
            raise ValueError("Buy price should be greater than before swap")


def test_peg_updated_during_block():
    peg_raise_n = 10
    daily_return = 0.01  # 1% return in a day
    daily_blocks = 7200
    mult_inc = math.pow(1 + daily_return, peg_raise_n / daily_blocks)
    ratios = [mpf(0.5), mpf(0.5)]
    susds_price = 1.5
    buyback_speed = mpf(1/10_000)
    liquidity = {'USDT': mpf(1_000_000), 'SUSDS': mpf(1_000_000)}
    usdt_pool = StableSwapPoolState({'USDT': ratios[0] * mpf(1_000_000), 'HOLLAR': mpf(1_000_000)},100, trade_fee=0.0001,)
    susds_pool = StableSwapPoolState({'HOLLAR': 1_000_000, 'SUSDS': ratios[1] * 1_000_000 / susds_price},100,
                                     trade_fee=0.0001, peg=susds_price)
    pools = [usdt_pool, susds_pool]
    sell_fee = 0.001
    max_buy_price_coef = 0.999
    buy_fee = 0.0001
    hsm = StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee)
    hsm2 = copy.deepcopy(hsm)
    profit_diff = []
    for t in range(daily_blocks):
        if t % peg_raise_n == 0:
            # do arb in first pool
            agent = Agent()
            hsm.arb(agent, 'SUSDS')
            # update peg target for both pools
            old_peg_target = hsm.pools['SUSDS'].peg_target[1]
            assert old_peg_target == hsm2.pools['SUSDS'].peg_target[1]
            hsm.pools['SUSDS'].set_peg_target(old_peg_target * mult_inc)
            hsm2.pools['SUSDS'].set_peg_target(old_peg_target * mult_inc)
            hsm.update()  # pushes update to internal pool_state object
            hsm2.update()
            # do arb in second pool
            agent2 = Agent()
            hsm2.arb(agent2, 'SUSDS')
            # calculate profit difference
            profit_diff.append(agent2.get_holdings('SUSDS') - agent.get_holdings('SUSDS'))
    total_profit_diff = sum(profit_diff)
    print('done')


def test_hsm_max_liquidity():
    liquidity = {'USDT': mpf(1_000_000), 'SUSDS': mpf(1_000_000)}
    max_liquidity = {'SUSDS': 2_000_000}
    susds_price = 1.05
    usdt_pool = StableSwapPoolState(tokens={'USDT': mpf(1_000_000), 'HOLLAR': mpf(1_000_000)}, amplification=100,
                                    trade_fee=0.0002)
    susds_pool = StableSwapPoolState(tokens={'HOLLAR': 1_000_000, 'SUSDS': 1_000_000 / susds_price}, amplification=100,
                                     trade_fee=0.0002, peg=susds_price)
    pools = [usdt_pool, susds_pool]
    sell_fee = 0.001
    max_buy_price_coef = 0.999
    buy_fee = 0.0001
    buyback_speed = mpf(1/10_000)
    # test that setting max liquidity works
    StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee, max_liquidity=max_liquidity)
    # test that initial liquidity above max liquidity fails
    max_liquidity = {'SUSDS': 500_000}
    with pytest.raises(ValueError):
        StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee,
                        max_liquidity=max_liquidity)
    # test that trade exceeding max liquidity fails
    max_liquidity = {'SUSDS': 2_000_000}
    hsm = StabilityModule(liquidity, buyback_speed, pools, sell_fee, max_buy_price_coef, buy_fee,
                          max_liquidity=max_liquidity)
    agent = Agent(enforce_holdings=False)
    hsm.swap(agent, 'HOLLAR', 'SUSDS', sell_quantity=2_000_000)
    if not hsm.fail:
        raise ValueError("Swap should have failed")


@given(
    liq_ratio = st.floats(min_value=0.0001, max_value=0.8),
    sell_amt = st.floats(min_value=1, max_value = 10_000)
)
def test_fast_hollar_arb_and_dump(liq_ratio, sell_amt):
    liquidity = {'USDT': mpf(1_000_000)}
    buyback_speed = 0.0002
    hollar_liq = mpf(1_000_000)
    stable_tokens = {'USDT': liq_ratio * hollar_liq, 'HOLLAR': hollar_liq}
    amp = 100
    swap_fee = 0.0  # our netting of opposite trades only works with 0 fee
    peg = 1
    ss = StableSwapPoolState(stable_tokens, amp, trade_fee=swap_fee, peg=peg, precision=0.00000001)
    pools_list = [ss]
    hsm = StabilityModule(liquidity, buyback_speed, pools_list)
    agent = Agent(enforce_holdings=False)

    simulated_hsm = copy.deepcopy(hsm)
    simulated_agent = agent.copy()
    fast_hollar_arb_and_dump(simulated_hsm, simulated_agent, sell_amt, 'USDT')

    full_hsm = copy.deepcopy(hsm)
    full_agent = agent.copy()
    full_hsm.arb(full_agent, 'USDT')
    full_hsm.pools['USDT'].swap(full_agent, tkn_sell='HOLLAR', tkn_buy='USDT', sell_quantity=sell_amt)

    if full_agent.get_holdings('HOLLAR') != simulated_agent.get_holdings('HOLLAR'):
        raise ValueError("Agent should have same holdings as simulated agent")
    if full_agent.get_holdings('USDT') != pytest.approx(simulated_agent.get_holdings('USDT'), rel=1e-15):
        raise ValueError("Agent should have same holdings as simulated agent")
    if full_hsm.liquidity['USDT'] != simulated_hsm.liquidity['USDT']:
        raise ValueError("HSM should have same liquidity as simulated hsm")
    if full_hsm.pools['USDT'].liquidity['USDT'] != pytest.approx(simulated_hsm.pools['USDT'].liquidity['USDT'], rel=1e-15):
        raise ValueError("Pool should have same liquidity as simulated pool")
    if full_hsm.pools['USDT'].liquidity['HOLLAR'] != simulated_hsm.pools['USDT'].liquidity['HOLLAR']:
        raise ValueError("Pool should have same liquidity as simulated pool")
    if full_hsm.pools['USDT'].peg[1] != simulated_hsm.pools['USDT'].peg[1]:
        raise ValueError("Pool peg should have same value as simulated pool")


@given(st.floats(min_value=0.5, max_value=1.5))
def test_arb_high_hollar(liq_ratio):
    liquidity = {'USDT': mpf(1_000_000)}
    buyback_speed = 0.0002
    hollar_liq = mpf(1_000_000)
    stable_tokens = {'USDT': liq_ratio * hollar_liq, 'HOLLAR': hollar_liq}
    amp = 100
    swap_fee = 0.001
    peg = 1
    ss = StableSwapPoolState(stable_tokens, amp, trade_fee=swap_fee, peg=peg, precision=0.00000001)
    pools_list = [ss]
    hsm = StabilityModule(liquidity, buyback_speed, pools_list)
    agent = Agent(enforce_holdings=False)
    hsm.arb(agent, 'USDT')
    if agent.get_holdings("HOLLAR") != 0:
        raise ValueError("Agent should have 0 HOLLAR holdings after arb")
    if agent.get_holdings("USDT") < 0:
        raise ValueError("Agent should have positive USDT holdings after arb")
    if agent.get_holdings("UDST") > 0 and liq_ratio < 1:
        raise ValueError("Arb should not have been completed")
    if agent.get_holdings("UDST") > 0 and 1/ss.buy_spot(tkn_buy="USDT", tkn_sell="HOLLAR") < 1 + hsm.sell_price_fee["USDT"]:
        raise ValueError("arb should not be eliminated entirely")


@given(st.floats(min_value=0.5, max_value=1.5), st.floats(min_value=0.9, max_value=1.1))
def test_get_hollar_sell_amount(liq_ratio, peg):
    liquidity = {'USDT': mpf(1_000_000)}
    buyback_speed = 0.0002
    hollar_liq = mpf(1_000_000)
    stable_tokens = {'USDT': liq_ratio * hollar_liq, 'HOLLAR': hollar_liq}
    amp = 100
    swap_fee = 0.001
    ss = StableSwapPoolState(stable_tokens, amp, trade_fee=swap_fee, peg=peg, precision=0.00000001)
    pools_list = [ss]
    hsm = StabilityModule(liquidity, buyback_speed, pools_list)
    agent = Agent(enforce_holdings=False)
    sell_amount = get_hollar_sell_amount(hsm, 'USDT', 50)
    hsm.arb(agent, 'USDT', sell_amount)
    if agent.get_holdings("HOLLAR") != 0:
        raise ValueError("Agent should have 0 HOLLAR holdings after arb")
    if agent.get_holdings("USDT") < 0:
        raise ValueError("Agent should have positive USDT holdings after arb")
    if (sell_amount > 0) and ((1 + hsm.sell_price_fee["USDT"]) * peg != pytest.approx(1 / ss.buy_spot(tkn_buy="USDT", tkn_sell="HOLLAR"), rel=1e-15)):
        raise ValueError("arb should make price converge")


def test_rebalance():
    def test_hsm_rebalance():
        liquidity = {'aUSDT': mpf(2e-06), 'sUSDS': mpf(37646.77909600129), 'sUSDe': mpf(34448.524988323006),
                     'aUSDC': mpf(382803.657686)}
        buyback_speed = 0.0000001

        pools_list = []

        peg = 1
        stable_tokens = {'aUSDT': mpf('462749.517821'), 'HOLLAR': mpf('1083761.886781065')}
        amp = 222
        swap_fee = 0.0002
        ss = StableSwapPoolState(stable_tokens, amp, trade_fee=swap_fee, peg=peg, precision=0.00000001)
        pools_list.append(ss)

        peg = 1
        stable_tokens = {'sUSDS': mpf('300371.9121634992'), 'HOLLAR': mpf('457456.8469612452')}
        amp = 111
        swap_fee = 0.0004
        ss = StableSwapPoolState(stable_tokens, amp, trade_fee=swap_fee, peg=peg, precision=0.00000001)
        pools_list.append(ss)

        peg = 1
        stable_tokens = {'sUSDe': mpf('306855.16433681524'), 'HOLLAR': mpf('541716.1096157538')}
        amp = 111
        swap_fee = 0.0004
        ss = StableSwapPoolState(stable_tokens, amp, trade_fee=swap_fee, peg=peg, precision=0.00000001)
        pools_list.append(ss)

        peg = 1
        stable_tokens = {'aUSDC': mpf('516836.454873'), 'HOLLAR': mpf('1093001.7589157932')}
        amp = 222
        swap_fee = 0.0002
        ss = StableSwapPoolState(stable_tokens, amp, trade_fee=swap_fee, peg=peg, precision=0.00000001)
        pools_list.append(ss)

        hsm = StabilityModule(liquidity, buyback_speed, pools_list, sell_price_fee=0.)

        hsm.arb()
