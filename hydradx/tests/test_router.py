import pytest
from hypothesis import given, strategies as st, settings  # , reproduce_failure
from mpmath import mpf, mp
from datetime import timedelta
from typing import Literal

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.omnipool_router import OmnipoolRouter, Trade
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.fixed_price import FixedPriceExchange
from hydradx.tests.strategies_omnipool import fee_strategy

settings.register_profile("long", deadline=timedelta(milliseconds=500), print_blob=True)
settings.load_profile("long")

mp.dps = 50

asset_quantity_strategy = st.floats(min_value=100000, max_value=10000000)


def test_price_route_example():
    # This test exists to make sure we are not making same mistake in test_price and in the price function
    tokens = {
        "HDX": {'liquidity': 10000000, 'LRNA': 1000000},
        "USDT": {'liquidity': 1000000, 'LRNA': 1000000},
        "DOT": {'liquidity': 100000, 'LRNA': 1000000},
        "stablepool": {'liquidity': 1000000, 'LRNA': 1000000},
    }
    omnipool = OmnipoolState(
        tokens=tokens,
        preferred_stablecoin="USDT",
        asset_fee=0,
        lrna_fee=0,
    )
    sp_tokens = {
        "stable1": 320000,
        "stable2": 330000,
        "stable3": 350000,
    }
    stablepool = StableSwapPoolState(
        tokens=sp_tokens,
        amplification=1000,
        trade_fee=0,
        unique_id="stablepool"
    )
    agent = Agent(enforce_holdings=False)
    router = OmnipoolRouter({"omnipool": omnipool, "stablepool": stablepool})
    router_price_outside_stablepool = router.price_route([
        Trade(exchange="omnipool", tkn_sell="DOT", tkn_buy="stablepool"),
        Trade(exchange="stablepool", tkn_sell="stablepool", tkn_buy="stable1"),
    ], direction="sell")
    router.swap(agent, tkn_sell="DOT", tkn_buy="stable1", sell_quantity=1)
    holdings = agent.get_holdings("stable1")
    # if we have the math right, the DOT price denominated in stable1 should be in the ballpark of 10.
    if router_price_outside_stablepool != pytest.approx(10, rel=1e-3):
        raise ValueError(f"router price {router_price_outside_stablepool} is not correct")


@given(
    st.lists(asset_quantity_strategy, min_size=6, max_size=6),
    st.floats(min_value=0.0001, max_value=0.1),
)
def test_swap_omnipool(assets: list[float], trade_size_mult: float):
    tokens = {
        "HDX": {'liquidity': assets[0], 'LRNA': assets[1]},
        "USDT": {'liquidity': assets[2], 'LRNA': assets[3]},
        "DOT": {'liquidity': assets[4], 'LRNA': assets[5]},
    }
    omnipool = OmnipoolState(tokens, preferred_stablecoin="USDT", asset_fee=0.0025, lrna_fee=0.0005)
    router = OmnipoolRouter([omnipool])
    omnipool2 = omnipool.copy()
    agent1 = Agent(holdings={"DOT": 1000000, "USDT": 1000000})
    agent2 = Agent(holdings={"DOT": 1000000, "USDT": 1000000})
    trade_size = trade_size_mult * min(assets[4], assets[2])

    # test buy
    router.swap(agent1, tkn_buy="DOT", tkn_sell="USDT", buy_quantity=10000,)
    omnipool2.swap(agent2, "DOT", "USDT", buy_quantity=10000)

    for token in omnipool.liquidity:
        if omnipool.liquidity[token] != omnipool2.liquidity[token]:
            raise ValueError(f"omnipool liquidity {omnipool.liquidity[token]} != {omnipool2.liquidity[token]}")
        if omnipool.lrna[token] != omnipool2.lrna[token]:
            raise ValueError(f"omnipool lrna {omnipool.lrna[token]} != {omnipool2.lrna[token]}")

    # test sell
    router.swap(agent1, tkn_buy="DOT", tkn_sell="USDT", sell_quantity=trade_size,)
    omnipool2.swap(agent2, "DOT", "USDT", sell_quantity=trade_size)

    for token in omnipool.liquidity:
        if omnipool.liquidity[token] != omnipool2.liquidity[token]:
            raise ValueError(f"omnipool liquidity {omnipool.liquidity[token]} != {omnipool2.liquidity[token]}")
        if omnipool.lrna[token] != omnipool2.lrna[token]:
            raise ValueError(f"omnipool lrna {omnipool.lrna[token]} != {omnipool2.lrna[token]}")


@given(
    st.lists(asset_quantity_strategy, min_size=9, max_size=9),
    st.floats(min_value=0.0001, max_value=0.1),
)
def test_swap_stableswap(assets: list[float], trade_size_mult: float):
    tokens = {
        "HDX": {'liquidity': assets[0], 'LRNA': assets[1]},
        "DOT": {'liquidity': assets[2], 'LRNA': assets[3]},
        "stablepool": {'liquidity': assets[4], 'LRNA': assets[5]},
    }
    omnipool = OmnipoolState(tokens, asset_fee=0.0025, lrna_fee=0.0005)
    stablepool = StableSwapPoolState(
        {"stable2": assets[6], "stable3": assets[7], "USDT": assets[8]},
        amplification=1000,
        trade_fee=0.0004,
        unique_id="stablepool"
    )
    router = OmnipoolRouter([omnipool, stablepool])
    omnipool_copy = omnipool.copy()
    stablepool_copy = stablepool.copy()
    agent1 = Agent(enforce_holdings=False)
    agent2 = Agent(enforce_holdings=False)
    trade_size = trade_size_mult * min(assets[4], assets[2])

    buy_quantity = 100

    # test buy
    router.swap(agent1, tkn_sell="DOT", tkn_buy="USDT", buy_quantity=buy_quantity)
    # calculate how many shares to buy
    delta_shares = stablepool_copy.calculate_withdraw_asset("USDT", buy_quantity)
    # buy shares
    omnipool_copy.swap(agent2, tkn_buy="stablepool", tkn_sell="DOT", buy_quantity=delta_shares)
    # withdraw USDT
    stablepool_copy.withdraw_asset(agent2, buy_quantity, "USDT")

    for token in omnipool.asset_list:
        if omnipool.liquidity[token] != omnipool_copy.liquidity[token]:
            raise ValueError(f"omnipool liquidity {omnipool.liquidity[token]} != {omnipool_copy.liquidity[token]}")
        if omnipool.lrna[token] != omnipool_copy.lrna[token]:
            raise ValueError(f"omnipool lrna {omnipool.lrna[token]} != {omnipool_copy.lrna[token]}")
    for token in stablepool.asset_list:
        if stablepool.liquidity[token] != stablepool_copy.liquidity[token]:
            raise ValueError(
                f"stablepool2 liquidity {stablepool.liquidity[token]} != {stablepool_copy.liquidity[token]}")
    for token in agent1.holdings:
        if agent1.holdings[token] != agent2.holdings[token]:
            raise ValueError(f"agent1 holdings {agent1.holdings[token]} != {agent2.holdings[token]}")

    # test sell
    router.swap(agent=agent1, tkn_buy="USDT", tkn_sell="DOT", sell_quantity=trade_size)
    # sell DOT for shares
    omnipool_copy.swap(agent2, tkn_buy="stablepool", tkn_sell="DOT", sell_quantity=trade_size)
    # withdraw USDT
    stablepool_copy.remove_liquidity(agent2, agent2.get_holdings("stablepool"), "USDT")

    for token in omnipool.asset_list:
        if omnipool.liquidity[token] != omnipool_copy.liquidity[token]:
            raise ValueError(f"omnipool liquidity {omnipool.liquidity[token]} != {omnipool_copy.liquidity[token]}")
        if omnipool.lrna[token] != omnipool_copy.lrna[token]:
            raise ValueError(f"omnipool lrna {omnipool.lrna[token]} != {omnipool_copy.lrna[token]}")
    for token in stablepool.asset_list:
        if stablepool.liquidity[token] != stablepool_copy.liquidity[token]:
            raise ValueError(
                f"stablepool liquidity {stablepool.liquidity[token]} != {stablepool_copy.liquidity[token]}")
    for token in agent1.holdings:
        if agent1.holdings[token] != agent2.holdings[token]:
            raise ValueError(f"agent1 holdings {agent1.holdings[token]} != {agent2.holdings[token]}")


@given(st.lists(asset_quantity_strategy, min_size=12, max_size=12))
def test_swap_stableswap2(assets: list[float]):
    tokens = {
        "HDX": {'liquidity': assets[0], 'LRNA': assets[1]},
        "stablepool": {'liquidity': assets[2], 'LRNA': assets[3]},
        "stablepool2": {'liquidity': assets[4], 'LRNA': assets[5]},
    }

    tkn_buy = "USDT"
    tkn_sell = "stable1"

    omnipool = OmnipoolState(tokens, asset_fee=0.0025, lrna_fee=0.0005)
    stablepool = StableSwapPoolState(
        tokens={"stable1": assets[6], "stable2": assets[7], "stable3": assets[8]},
        amplification=900, trade_fee=0.0100, unique_id="stablepool"
    )
    stablepool2 = StableSwapPoolState(
        tokens={"stable2": assets[9], "stable3": assets[10], "USDT": assets[11]},
        amplification=1000, trade_fee=0.0002, unique_id="stablepool2"
    )
    router = OmnipoolRouter([omnipool, stablepool, stablepool2])
    trade_size = 100

    agent1 = Agent(enforce_holdings=False)

    # try buy
    router.swap(
        agent=agent1,
        tkn_sell=tkn_sell, tkn_buy=tkn_buy,
        buy_quantity=trade_size
    )
    if router.fail:
        raise ValueError(f"trade failed")

    for tkn in agent1.holdings.keys():
        holdings = agent1.get_holdings(tkn)
        if tkn == tkn_buy:
            if holdings != pytest.approx(trade_size, rel=1e-12):
                raise ValueError(f"agent1 holdings {holdings} != {trade_size}")
        elif tkn == tkn_sell:
            if holdings >= 0:
                raise ValueError(f"agent1 holdings {holdings} >= 0")
        elif holdings != 0:
                raise ValueError(f"agent1 holdings {holdings} != 0")

    # test sell
    agent1 = Agent(enforce_holdings=False)
    router.swap(
        agent=agent1,
        tkn_sell="stable1", tkn_buy="USDT",
        sell_quantity=trade_size
    )
    if router.fail:
        raise ValueError(f"trade failed")

    for tkn in agent1.holdings.keys():
        holdings = agent1.get_holdings(tkn)
        if tkn == tkn_buy:
            if holdings <= 0:
                raise ValueError(f"agent1 holdings {holdings} <= 0")
        elif tkn == tkn_sell:
            if holdings + trade_size != 0:
                raise ValueError(f"agent1 holdings {holdings + trade_size} != 0")
        elif holdings != 0:
            raise ValueError(f"agent1 holdings {holdings} != 0")


def test_swap():
    sp_tokens = {"stable1": 400000, "stable3": 400000}
    stablepool = StableSwapPoolState(sp_tokens, 1000, trade_fee=0.0000, unique_id="stablepool")
    sp_tokens2 = {"stable2": 300000, "stable3": 300000, "USDT": 400000}  # USDT cheaper in stablepool2
    stablepool2 = StableSwapPoolState(sp_tokens2, 1000, trade_fee=0.0000, unique_id="stablepool2")

    tokens = {
        "HDX": {'liquidity': 10000000, 'LRNA': 1000000},
        "USDT": {'liquidity': 1000000, 'LRNA': 1000000},
        "DOT": {'liquidity': 100000, 'LRNA': 1000000},
        "stablepool": {'liquidity': stablepool.shares, 'LRNA': 1000000},
        "stablepool2": {'liquidity': stablepool2.shares, 'LRNA': 1000000},
    }

    buy_tkn = "USDT"
    sell_tkn = "stable1"

    omnipool = OmnipoolState(tokens, preferred_stablecoin="USDT", asset_fee=0.0000, lrna_fee=0.0000)
    router = OmnipoolRouter([omnipool, stablepool, stablepool2])
    trade_size = 1
    agent1 = Agent(holdings={sell_tkn: trade_size})

    best_route = router.find_best_route(buy_tkn, sell_tkn, sell_quantity=trade_size)
    if best_route[0].exchange != "stablepool" or best_route[-1].exchange != "stablepool2":
        raise ValueError(f"best route {best_route} != ('stablepool', 'stablepool2')")

    new_router, new_agent = router.simulate_swap(
        agent=agent1,
        tkn_buy=buy_tkn,
        tkn_sell=sell_tkn,
        sell_quantity=trade_size
    )
    new_router2, new_agent2 = router.simulate_swap_route(
        agent=agent1,
        route=best_route,
        sell_quantity=trade_size
    )
    for tkn in new_agent.holdings:
        if new_agent.holdings[tkn] != new_agent2.holdings[tkn]:
            raise ValueError(f"new_agent holdings {new_agent.holdings[tkn]} != {new_agent2.holdings[tkn]}")
    for ex_id in new_router.exchanges:
        ex = new_router.exchanges[ex_id]
        for tkn in ex.liquidity:
            if ex.liquidity[tkn] != new_router2.exchanges[ex_id].liquidity[tkn]:
                raise ValueError(f"ex liquidity {ex.liquidity[tkn]} != {new_router2.exchanges[ex_id].liquidity[tkn]}")


def test_swap2():
    sp_tokens = {"stable1": 400000, "stable3": 400000}
    stablepool = StableSwapPoolState(sp_tokens, 1000, trade_fee=0.0000, unique_id="stablepool")
    sp_tokens2 = {"stable2": 300000, "stable3": 300000, "USDT": 200000}  # USDT more expensive in stablepool2
    stablepool2 = StableSwapPoolState(sp_tokens2, 1000, trade_fee=0.0000, unique_id="stablepool2")

    tokens = {
        "HDX": {'liquidity': 10000000, 'LRNA': 1000000},
        "USDT": {'liquidity': 1000000, 'LRNA': 1000000},
        "DOT": {'liquidity': 100000, 'LRNA': 1000000},
        "stablepool": {'liquidity': stablepool.shares, 'LRNA': 1000000},
        "stablepool2": {'liquidity': stablepool2.shares, 'LRNA': 1000000},
    }

    buy_tkn = "USDT"
    sell_tkn = "stable1"

    omnipool = OmnipoolState(tokens, preferred_stablecoin="USDT", asset_fee=0.0000, lrna_fee=0.0000)
    exchanges = {"omnipool": omnipool, "stablepool": stablepool, "stablepool2": stablepool2}
    router = OmnipoolRouter(exchanges)
    trade_size = 1
    agent1 = Agent(holdings={sell_tkn: trade_size})

    best_route = router.find_best_route(buy_tkn, sell_tkn, buy_quantity=trade_size)
    if best_route[0].exchange != "omnipool" or best_route[-1].exchange != "stablepool":
        raise ValueError(f"best route {best_route} != ('stablepool', 'omnipool')")

    new_router, new_agent = router.simulate_swap(
        agent=agent1,
        tkn_buy=buy_tkn,
        tkn_sell=sell_tkn,
        buy_quantity=trade_size
    )
    new_router2, new_agent2 = router.simulate_swap_route(
        agent=agent1,
        route=best_route,
        buy_quantity=trade_size,
    )
    for tkn in new_agent.holdings:
        if new_agent.holdings[tkn] != new_agent2.holdings[tkn]:
            raise ValueError(f"new_agent holdings {new_agent.holdings[tkn]} != {new_agent2.holdings[tkn]}")
    for ex_id in new_router.exchanges:
        ex = new_router.exchanges[ex_id]
        for tkn in ex.liquidity:
            if ex.liquidity[tkn] != new_router2.exchanges[ex_id].liquidity[tkn]:
                raise ValueError(f"ex liquidity {ex.liquidity[tkn]} != {new_router2.exchanges[ex_id].liquidity[tkn]}")


def check_agent_holdings_equal(agent1, agent2):
    for tkn in agent1.holdings:
        if agent1.holdings[tkn] != agent2.holdings[tkn]:
            raise ValueError(f"agent holdings {agent1.holdings[tkn]} != {agent2.holdings[tkn]}")


def check_liquidity_equal(pool1, pool2):
    for tkn in pool1.liquidity:
        if pool1.liquidity[tkn] != pool2.liquidity[tkn]:
            raise ValueError(f"pool liquidity {pool1.liquidity[tkn]} != {pool2.liquidity[tkn]}")


@given(st.lists(asset_quantity_strategy, min_size=6, max_size=6))
def test_swap_shares(assets: list[float]):
    tokens = {
        "HDX": {'liquidity': 1000000, 'LRNA': 1000000},
        "USDT": {'liquidity': assets[0], 'LRNA': assets[1]},
        "DOT": {'liquidity': 1000000, 'LRNA': 1000000},
        "stablepool": {'liquidity': 1000000, 'LRNA': assets[2]},
    }

    tkn_buy = "USDT"
    tkn_sell = "stablepool"

    stablepool = StableSwapPoolState(
        tokens={"stable1": assets[0], "stable2": assets[1], "USDT": assets[2]},
        amplification=1000, trade_fee=0.0100, unique_id="stablepool"
    )
    router = OmnipoolRouter([stablepool])
    agent1 = Agent(enforce_holdings=False)
    agent2 = Agent(enforce_holdings=False)
    stablepool_copy = stablepool.copy()
    trade_size = 1000

    # test buy tkn
    router.swap(
        agent1,
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        buy_quantity=trade_size,
    )
    stablepool_copy.withdraw_asset(agent2, trade_size, "USDT")
    check_agent_holdings_equal(agent1, agent2)
    check_liquidity_equal(stablepool, stablepool_copy)

    # test sell shares
    router.swap(
        agent1,
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        sell_quantity=trade_size
    )
    stablepool_copy.remove_liquidity(agent2, trade_size, "USDT")
    check_agent_holdings_equal(agent1, agent2)
    check_liquidity_equal(stablepool, stablepool_copy)

    # try the other way around
    tkn_sell, tkn_buy = tkn_buy, tkn_sell

    # test sell tkn
    router.swap(
        agent1,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        sell_quantity=trade_size
    )
    stablepool_copy.add_liquidity(agent2, trade_size, "USDT")
    check_agent_holdings_equal(agent1, agent2)
    check_liquidity_equal(stablepool, stablepool_copy)

    # test buy shares
    router.swap(
        agent1,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        buy_quantity=trade_size
    )
    stablepool_copy.buy_shares(agent2, trade_size, "USDT")
    check_agent_holdings_equal(agent1, agent2)
    check_liquidity_equal(stablepool, stablepool_copy)


@given(st.lists(asset_quantity_strategy, min_size=6, max_size=6))
def test_spot_prices_omnipool_to_self(assets: list[float]):
    tokens = {
        "HDX": {'liquidity': mpf(assets[0]), 'LRNA': mpf(assets[1])},
        "USDT": {'liquidity': mpf(assets[2]), 'LRNA': mpf(assets[3])},
        "DOT": {'liquidity': mpf(assets[4]), 'LRNA': mpf(assets[5])},
    }

    tkn_buy = "USDT"
    tkn_sell = "DOT"

    omnipool = OmnipoolState(tokens, preferred_stablecoin="USDT", asset_fee=0.0025, lrna_fee=0.0005)
    exchanges = {"omnipool": omnipool}
    router = OmnipoolRouter(exchanges)
    initial_agent = Agent(
        holdings={"DOT": mpf(1000000), "USDT": mpf(1000000)}
    )
    trade_size = 1e-12
    buy_spot = router.buy_spot(tkn_buy=tkn_buy, tkn_sell=tkn_sell)
    sell_spot = router.sell_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
    test_router, test_agent = router.simulate_swap(initial_agent, tkn_buy, tkn_sell, buy_quantity=trade_size)
    execution_price = ((initial_agent.holdings[tkn_sell] - test_agent.holdings[tkn_sell])
                       / (test_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy]))
    if buy_spot != pytest.approx(execution_price, rel=1e-12):
        raise ValueError(f"spot price {buy_spot} != execution price {execution_price}")
    if sell_spot != pytest.approx(1 / execution_price, rel=1e-12):
        raise ValueError(f"spot price {sell_spot} != execution price {1 / execution_price}")


@given(
    assets=st.lists(asset_quantity_strategy, min_size=2, max_size=2),
    lrna_fee=fee_strategy,
    asset_fee=fee_strategy,
    trade_fee=fee_strategy
)
def test_spot_prices_stableswap_to_self(assets, lrna_fee, asset_fee, trade_fee):
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': 1000000, 'LRNA': 1000000},
            "USDT": {'liquidity': 1000000, 'LRNA': 1000000}
        },
        preferred_stablecoin="USDT",
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )
    stablepool1 = StableSwapPoolState(
        tokens={"stable1": mpf(assets[0]), "stable2": mpf(assets[1])},
        amplification=100,
        trade_fee=trade_fee, unique_id="stablepool1",
        precision=1e-08
    )
    router = OmnipoolRouter({"omnipool": omnipool, "stablepool1": stablepool1})

    tkn_sell = "stable1"
    tkn_buy = "stable2"
    trade_size = 0.001
    initial_agent = Agent(
        holdings={"stable1": mpf(1), "stable2": mpf(0)}
    )
    test_router, test_agent = router.simulate_swap(
        initial_agent, tkn_buy, tkn_sell, sell_quantity=trade_size
    )

    sell_spot = router.sell_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
    buy_spot = router.buy_spot(tkn_buy=tkn_buy, tkn_sell=tkn_sell)
    sell_quantity = initial_agent.holdings[tkn_sell] - test_agent.holdings[tkn_sell]
    buy_quantity = test_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy]
    sell_ex = buy_quantity / sell_quantity
    buy_ex = sell_quantity / buy_quantity

    if sell_quantity != trade_size:
        raise ValueError(f"actually bought {sell_quantity} != trade size {trade_size}")
    if sell_spot != pytest.approx(sell_ex, rel=1e-08):
        raise ValueError(f"spot price {sell_spot} != execution price {sell_ex}")
    if buy_spot != pytest.approx(buy_ex, rel=1e-08):
        raise ValueError(f"spot price {buy_spot} != execution price {buy_ex}")


@given(
    assets=st.lists(asset_quantity_strategy, min_size=6, max_size=6),
    lrna_fee=fee_strategy,
    asset_fee=fee_strategy,
    trade_fee=fee_strategy
)
def test_buy_spot_buy_stableswap_sell_stableswap(assets, lrna_fee, asset_fee, trade_fee):
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': 1000000, 'LRNA': 1000000},
            "USDT": {'liquidity': 1000000, 'LRNA': 1000000},
            "stablepool1": {'liquidity': mpf(1000000), 'LRNA': mpf(assets[0])},
            "stablepool2": {'liquidity': mpf(1000000), 'LRNA': mpf(assets[1])}
        },
        preferred_stablecoin="USDT",
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )
    stablepool1 = StableSwapPoolState(
        tokens={"stable1": mpf(1000000), "stable2": mpf(assets[2])},
        amplification=100,
        trade_fee=trade_fee, unique_id="stablepool1",
        precision=1e-08
    )
    stablepool2 = StableSwapPoolState(
        tokens={"stable3": mpf(1000000), "stable4": mpf(assets[3])},
        amplification=1000,
        trade_fee=trade_fee, unique_id="stablepool2",
        precision=1e-08
    )
    initial_agent = Agent(enforce_holdings=False)
    tkn_sell = "stable1"
    tkn_buy = "stable3"
    trade_size = mpf(1e-07)

    router = OmnipoolRouter({"omnipool": omnipool, "stablepool1": stablepool1, "stablepool2": stablepool2})

    test_router, test_agent = router.simulate_swap(
        initial_agent, tkn_buy, tkn_sell, buy_quantity=trade_size
    )

    # debugging stuff
    # shares_bought = stablepool2.calculate_withdrawal_shares(tkn_buy, trade_size)
    # shares_sold = omnipool.calculate_sell_from_buy(
    #     tkn_buy=stablepool2.unique_id,
    #     tkn_sell=stablepool1.unique_id,
    #     buy_quantity=shares_bought
    # )
    # step_1_buy_spot = stablepool1.buy_shares_spot(tkn_add=tkn_sell)
    # step_1_agent = initial_agent.copy()
    # stablepool1.copy().buy_shares(step_1_agent, quantity=shares_sold, tkn_add=tkn_sell)
    # step_1_buy_ex = (initial_agent.holdings[tkn_sell] - step_1_agent.holdings[tkn_sell]) / step_1_agent.holdings['stablepool1']
    #
    # step_2_buy_spot = omnipool.buy_spot(tkn_sell='stablepool1', tkn_buy='stablepool2')
    # step_2_agent = step_1_agent.copy()
    # omnipool.copy().swap(step_2_agent, tkn_sell='stablepool1', tkn_buy='stablepool2', buy_quantity=shares_bought)
    # step_2_buy_ex = step_1_agent.holdings['stablepool1'] / step_2_agent.holdings['stablepool2']
    #
    # step_3_buy_spot = 1 / stablepool2.withdraw_asset_spot(tkn_remove=tkn_buy)
    # step_3_agent = step_2_agent.copy()
    # stablepool2.copy().withdraw_asset(step_3_agent, quantity=trade_size, tkn_remove=tkn_buy)
    # step_3_buy_ex = step_2_agent.holdings['stablepool2'] / step_3_agent.holdings[tkn_buy]

    buy_spot = router.buy_spot(tkn_buy=tkn_buy, tkn_sell=tkn_sell)
    sell_quantity = initial_agent.get_holdings(tkn_sell) - test_agent.get_holdings(tkn_sell)
    buy_quantity = test_agent.get_holdings(tkn_buy) - initial_agent.get_holdings(tkn_buy)
    buy_ex = sell_quantity / buy_quantity

    if buy_quantity != trade_size:
        raise ValueError(f"actually bought {buy_quantity} != trade size {trade_size}")
    if buy_spot != pytest.approx(buy_ex, rel=1e-06):
        raise ValueError(f"spot price {buy_spot} != execution price {buy_ex}")


@given(
    assets=st.lists(asset_quantity_strategy, min_size=4, max_size=4),
    lrna_fee=fee_strategy,
    asset_fee=fee_strategy,
    trade_fee=fee_strategy
)
def test_sell_spot_buy_stableswap_sell_stableswap(assets, lrna_fee, asset_fee, trade_fee):
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': 1000000, 'LRNA': 1000000},
            "USDT": {'liquidity': 1000000, 'LRNA': 1000000},
            "stablepool1": {'liquidity': mpf(1000000), 'LRNA': mpf(assets[0])},
            "stablepool2": {'liquidity': mpf(1000000), 'LRNA': mpf(assets[1])}
        },
        preferred_stablecoin="USDT",
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )
    stablepool1 = StableSwapPoolState(
        tokens={"stable1": mpf(1000000), "stable2": mpf(1000000)},
        amplification=100,
        trade_fee=trade_fee, unique_id="stablepool1",
        precision=1e-08
    )
    stablepool2 = StableSwapPoolState(
        tokens={"stable3": mpf(1000000), "stable4": mpf(1000000)},
        amplification=1000,
        trade_fee=trade_fee, unique_id="stablepool2",
        precision=1e-08
    )
    initial_agent = Agent(
        holdings={"stable1": mpf(1), "stable3": mpf(0)}
    )
    tkn_sell = "stable1"
    tkn_buy = "stable3"
    trade_size = mpf(1e-07)

    # debugging stuff
    # step_1_sell_spot = 1 / stablepool1.add_liquidity_spot(tkn_add=tkn_sell)
    # step_1_agent = initial_agent.copy()
    # stablepool1.copy().add_liquidity(step_1_agent, quantity=trade_size, tkn_add=tkn_sell)
    # step_1_sell_ex = step_1_agent.holdings['stablepool1'] / trade_size
    #
    # step_2_sell_spot = omnipool.sell_spot(tkn_sell='stablepool1', tkn_buy='stablepool2')
    # step_2_agent = step_1_agent.copy()
    # omnipool.copy().swap(step_2_agent, tkn_sell='stablepool1', tkn_buy='stablepool2', sell_quantity=step_1_agent.holdings['stablepool1'])
    # step_2_sell_ex = step_2_agent.holdings['stablepool2'] / step_1_agent.holdings['stablepool1']
    #
    # step_3_sell_spot = stablepool2.remove_liquidity_spot(tkn_remove=tkn_buy)
    # step_3_agent = step_2_agent.copy()
    # stablepool2.copy().remove_liquidity(step_3_agent, shares_removed=step_2_agent.holdings['stablepool2'], tkn_remove=tkn_buy)
    # step_3_sell_ex = step_3_agent.holdings[tkn_buy] / step_2_agent.holdings['stablepool2']

    router = OmnipoolRouter({"omnipool": omnipool, "stablepool1": stablepool1, "stablepool2": stablepool2})

    test_router, test_agent = router.simulate_swap(
        initial_agent, tkn_buy, tkn_sell, sell_quantity=trade_size
    )

    sell_spot = router.sell_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
    sell_quantity = initial_agent.holdings[tkn_sell] - test_agent.holdings[tkn_sell]
    buy_quantity = test_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy]
    sell_ex = buy_quantity / sell_quantity

    if sell_quantity != trade_size:
        raise ValueError(f"actually sold {sell_quantity} != trade size {trade_size}")
    if sell_spot != pytest.approx(sell_ex, rel=1e-06):
        raise ValueError(f"spot price {sell_spot} != execution price {sell_ex}")


@given(
    assets=st.lists(asset_quantity_strategy, min_size=4, max_size=4),
    lrna_fee=fee_strategy,
    asset_fee=fee_strategy,
    trade_fee=fee_strategy
)
def test_buy_spot_buy_stableswap_sell_omnipool(assets, lrna_fee, asset_fee, trade_fee):
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            "USDT": {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            "DOT": {'liquidity': 1000000, 'LRNA': mpf(assets[0])},
            "stablepool": {'liquidity': 1000000, 'LRNA': mpf(assets[1])},
        },
        preferred_stablecoin="USDT",
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )
    stablepool = StableSwapPoolState(
        tokens={"stable1": mpf(assets[2]), "stable2": mpf(assets[3])},
        amplification=1000,
        trade_fee=trade_fee, unique_id="stablepool",
        precision=1e-08,
        spot_price_precision=1e-12
    )
    router = OmnipoolRouter({"omnipool": omnipool, "stablepool": stablepool})

    tkn_sell = "DOT"
    tkn_buy = "stable1"
    trade_size = 1e-08
    initial_agent = Agent(enforce_holdings=False)
    test_router, test_agent = router.simulate_swap(
        initial_agent, tkn_buy, tkn_sell, buy_quantity=trade_size
    )

    buy_spot = router.buy_spot(tkn_buy=tkn_buy, tkn_sell=tkn_sell)
    buy_quantity = test_agent.get_holdings(tkn_buy) - initial_agent.get_holdings(tkn_buy)
    sell_quantity = initial_agent.get_holdings(tkn_sell) - test_agent.get_holdings(tkn_sell)
    buy_ex = sell_quantity / buy_quantity

    if buy_quantity != trade_size:
        raise ValueError(f"actually bought {buy_quantity} != trade size {trade_size}")
    if buy_spot != pytest.approx(buy_ex, rel=1e-08):
        raise ValueError(f"spot price {buy_spot} != execution price {buy_ex}")
    if test_agent.holdings['stablepool'] > 0:
        # 1e-50 would be still acceptable, but we're leaving this at 0 because it seems to work
        raise ValueError(f"leftover stablepool shares: {test_agent.holdings['stablepool']}")


@given(
    assets=st.lists(asset_quantity_strategy, min_size=4, max_size=4),
    lrna_fee=fee_strategy,
    asset_fee=fee_strategy,
    trade_fee=fee_strategy
)
def test_sell_spot_buy_stableswap_sell_omnipool(assets, lrna_fee, asset_fee, trade_fee):
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            "USDT": {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            "DOT": {'liquidity': 1000000, 'LRNA': mpf(assets[0])},
            "stablepool": {'liquidity': 1000000, 'LRNA': mpf(assets[1])},
        },
        preferred_stablecoin="USDT",
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )
    stablepool = StableSwapPoolState(
        tokens={"stable1": mpf(assets[2]), "stable2": mpf(assets[3])},
        amplification=1000,
        trade_fee=trade_fee, unique_id="stablepool",
        precision=1e-08,
        spot_price_precision=1e-12
    )
    router = OmnipoolRouter({"omnipool": omnipool, "stablepool": stablepool})

    tkn_sell = "DOT"
    tkn_buy = "stable1"
    trade_size = 1e-08
    initial_agent = Agent(
        holdings={"DOT": mpf(1), "stable1": mpf(0)}
    )
    test_router, test_agent = router.simulate_swap(
        initial_agent, tkn_buy, tkn_sell, sell_quantity=trade_size
    )

    sell_spot = router.sell_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
    buy_quantity = test_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy]
    sell_quantity = initial_agent.holdings[tkn_sell] - test_agent.holdings[tkn_sell]
    sell_ex = buy_quantity / sell_quantity

    if sell_quantity != trade_size:
        raise ValueError(f"actually sold {sell_quantity} != trade size {trade_size}")
    if sell_spot != pytest.approx(sell_ex, rel=1e-08):
        raise ValueError(f"spot price {sell_spot} != execution price {sell_ex}")


@given(
    assets=st.lists(asset_quantity_strategy, min_size=4, max_size=4),
    lrna_fee=fee_strategy,
    asset_fee=fee_strategy,
    trade_fee=fee_strategy
)
def test_sell_spot_sell_stableswap_buy_omnipool(
        assets: list[float], lrna_fee: float, asset_fee: float, trade_fee: float
):
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            "USDT": {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            "DOT": {'liquidity': mpf(1000000), 'LRNA': mpf(assets[0])},
            "stablepool": {'liquidity': mpf(1000000), 'LRNA': mpf(assets[1])},
        },
        preferred_stablecoin="USDT",
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )
    stablepool = StableSwapPoolState(
        tokens={"stable1": mpf(assets[2]), "stable2": mpf(assets[3])},
        amplification=1000,
        trade_fee=trade_fee, unique_id="stablepool",
        precision=1e-08,
        spot_price_precision=1e-12
    )
    router = OmnipoolRouter({"omnipool": omnipool, "stablepool": stablepool})

    tkn_sell = "stable1"
    tkn_buy = "DOT"
    trade_size = 1e-08
    initial_agent = Agent(
        holdings={"DOT": mpf(0), "stable1": mpf(1)}
    )
    test_router, test_agent = router.simulate_swap(
        initial_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=trade_size
    )

    sell_spot = router.sell_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
    sell_quantity = initial_agent.holdings[tkn_sell] - test_agent.holdings[tkn_sell]
    buy_quantity = test_agent.holdings[tkn_buy] - initial_agent.holdings[tkn_buy]
    sell_ex = buy_quantity / sell_quantity

    if sell_quantity != trade_size:
        raise ValueError(f"actually sold {sell_quantity} != trade size {trade_size}")
    if sell_spot != pytest.approx(sell_ex, rel=1e-08):
        raise ValueError(f"spot price {sell_spot} != execution price {sell_ex}")


@given(
    assets=st.lists(asset_quantity_strategy, min_size=4, max_size=4),
    lrna_fee=fee_strategy,
    asset_fee=fee_strategy,
    trade_fee=fee_strategy
)
@settings(deadline=timedelta(milliseconds=500))
def test_buy_spot_sell_stableswap_buy_omnipool(
        assets: list[float], lrna_fee: float, asset_fee: float, trade_fee: float
):
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            "USDT": {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            "DOT": {'liquidity': mpf(1000000), 'LRNA': mpf(assets[0])},
            "stablepool": {'liquidity': mpf(1000000), 'LRNA': mpf(assets[1])},
        },
        preferred_stablecoin="USDT",
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )
    stablepool = StableSwapPoolState(
        tokens={"stable1": mpf(assets[2]), "stable2": mpf(assets[3])},
        amplification=1000,
        trade_fee=trade_fee, unique_id="stablepool",
        precision=1e-08,
        spot_price_precision=1e-12
    )
    router = OmnipoolRouter({"omnipool": omnipool, "stablepool": stablepool})

    tkn_sell = "stable1"
    tkn_buy = "DOT"
    trade_size = 1e-08
    initial_agent = Agent(enforce_holdings=False)
    test_router, test_agent = router.simulate_swap(
        initial_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=trade_size
    )

    buy_spot = router.buy_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
    sell_quantity = initial_agent.get_holdings(tkn_sell) - test_agent.get_holdings(tkn_sell)
    buy_quantity = test_agent.get_holdings(tkn_buy) - initial_agent.get_holdings(tkn_buy)
    buy_ex = sell_quantity / buy_quantity

    if buy_quantity != trade_size:
        raise ValueError(f"actually bought {buy_quantity} != trade size {trade_size}")
    if buy_spot != pytest.approx(buy_ex, rel=1e-08):
        raise ValueError(f"spot price {buy_spot} != execution price {buy_ex}")


def test_calculate_buy_from_sell():
    stable1 = StableSwapPoolState(
        tokens={
            'BTC': mpf(1000),
            'WBTC': mpf(1001)
        }, amplification=100,
        unique_id='btc pool',
        trade_fee=0.000123
    )
    stable2 = StableSwapPoolState(
        tokens={
            'ETH': mpf(1001),
            'WETH': mpf(999)
        }, amplification=222,
        unique_id='eth pool',
        trade_fee=0.00011
    )
    omnipool = OmnipoolState(
        tokens={
            'eth pool': {'liquidity': 101, 'LRNA': 1999},
            'btc pool': {'liquidity': 102, 'LRNA': 20202},
            'HDX': {'liquidity': 1003333, 'LRNA': 1234},
            'USD': {'liquidity': 100000, 'LRNA': 5000},
            'ETH': {'liquidity': 100, 'LRNA': 2000}
        },
        asset_fee=0.0022, lrna_fee=0.0011
    )
    router = OmnipoolRouter(
        [stable1, stable2, omnipool]
    )
    agent = Agent(enforce_holdings=False)
    routes = router.find_routes(tkn_buy='ETH', tkn_sell='BTC', direction='sell')
    eth_btc_route = router.find_best_route(tkn_buy='ETH', tkn_sell='BTC')
    print("route prices:", {", ".join([swap.exchange for swap in route]): router.price_route(route=route, direction="sell") for route in routes})
    print("route calcs:", {", ".join([swap.exchange for swap in route]): router.calculate_buy_from_sell(
        tkn_sell="BTC", tkn_buy="ETH", route=route, sell_quantity=1
    ) for route in routes})
    if eth_btc_route[0].exchange != 'btc pool' or eth_btc_route[-1].exchange != 'eth pool':
        # this is to make sure we test the case where one of the assets exists in two different pools.
        raise AssertionError('ETH -> BTC trade should go through both subpools.')
    btc_eth_route = router.find_best_route(tkn_buy='BTC', tkn_sell='ETH')
    if btc_eth_route[0].exchange != 'omnipool' or btc_eth_route[-1].exchange != 'btc pool':
        # this is to test the case where buy route and sell route are different.
        raise AssertionError('BTC -> ETH trade should go through omnipool -> btc pool.')

    trades = [
        {'tkn_buy': 'BTC', 'tkn_sell': 'ETH', 'sell_quantity': 10},
        {'tkn_buy': 'BTC', 'tkn_sell': 'HDX', 'sell_quantity': 11},
        {'tkn_buy': 'HDX', 'tkn_sell': 'ETH', 'sell_quantity': 12},
        {'tkn_buy': 'USD', 'tkn_sell': 'HDX', 'sell_quantity': 13},
        {'tkn_buy': 'BTC', 'tkn_sell': 'ETH', 'sell_quantity': 14},
        {'tkn_buy': 'ETH', 'tkn_sell': 'BTC', 'sell_quantity': 15},
    ]
    for trade in trades:
        tkn_buy, tkn_sell, sell_quantity = trade.values()
        buy_prediction = router.calculate_buy_from_sell(
            tkn_sell=tkn_sell, tkn_buy=tkn_buy, sell_quantity=sell_quantity
        )
        agent = Agent(enforce_holdings=False)
        router.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=sell_quantity)
        buy_result = agent.get_holdings(tkn_buy)
        if buy_prediction != pytest.approx(buy_result, rel=1e-15):
            raise AssertionError('Trade did not come out as predicted.')


def test_price_route():
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': mpf(10001910), 'LRNA': mpf(1000900)},
            "USDT": {'liquidity': mpf(1003000), 'LRNA': mpf(1000060)},
            "DOT": {'liquidity': mpf(100000), 'LRNA': mpf(1020600)},
            "stablepool1": {'liquidity': mpf(1500000), 'LRNA': mpf(1567000)},
            "stablepool2": {'liquidity': mpf(1054000), 'LRNA': mpf(1045675)},
        },
        preferred_stablecoin="USDT",
        asset_fee=0.0025,
        lrna_fee=0.0005
    )
    stablepool1 = StableSwapPoolState(
        tokens={"stable1": mpf(1230000), "stable2": mpf(1320000)},
        amplification=1002,
        trade_fee=0.0100, unique_id="stablepool1",
        precision=1e-12,
        spot_price_precision=1e-12
    )
    stablepool2 = StableSwapPoolState(
        tokens={"stable2": mpf(330000), "stable3": mpf(345000)},
        amplification=101,
        trade_fee=0.0001, unique_id="stablepool2",
        precision=1e-12,
        spot_price_precision=1e-12
    )
    router = OmnipoolRouter(
        exchanges=[omnipool, stablepool1, stablepool2],
    )
    routes = [
        [
            Trade(exchange="omnipool", tkn_sell="DOT", tkn_buy="stablepool1"),
            Trade(exchange="stablepool1", tkn_sell="stablepool1", tkn_buy="stable1")
        ], [
            Trade(exchange="stablepool2", tkn_sell="stable2", tkn_buy="stablepool2"),
            Trade(exchange="omnipool", tkn_sell="stablepool2", tkn_buy="DOT")
        ], [
            Trade(exchange="stablepool1", tkn_sell="stable1", tkn_buy="stablepool1"),
            Trade(exchange="omnipool", tkn_sell="stablepool1", tkn_buy="stablepool2"),
            Trade(exchange="stablepool2", tkn_sell="stablepool2", tkn_buy="stable3")
        ], [
            Trade(exchange="stablepool1", tkn_sell="stable1", tkn_buy="stable2"),
            Trade(exchange="stablepool2", tkn_sell="stable2", tkn_buy="stable3"),
        ]
    ]
    for route in routes:
        direction: Literal['buy', 'sell']
        for direction in ['sell', 'buy']:
            agent = Agent(enforce_holdings=False)
            tkn_buy = route[-1].tkn_buy
            tkn_sell = route[0].tkn_sell
            if direction == 'buy':
                route.reverse()
            trade_size = mpf(1) / 10 ** 12
            router_copy = router.copy()
            router_copy.swap_route(
                agent=agent, route=route,
                buy_quantity=trade_size if direction == 'buy' else None,
                sell_quantity=trade_size if direction == 'sell' else None
            )
            ex_price = (-agent.get_holdings(tkn_sell) if direction == 'buy' else agent.get_holdings(tkn_buy)) / trade_size
            route_price = router.price_route(route, direction)
            if ex_price != pytest.approx(route_price, rel=1e-12):
                raise ValueError(
                    f"Execution price {ex_price} does not match route price {route_price} for route {route} in direction {direction}"
                )
            if direction == 'buy' and agent.get_holdings(tkn_buy) != trade_size:
                raise ValueError(
                    f"Agent holdings {agent.get_holdings(tkn_buy)} do not match expected buy quantity {trade_size} for route {route} in direction {direction}"
                )
            if direction == 'sell' and agent.get_holdings(tkn_sell) != -trade_size:
                raise ValueError(
                    f"Agent holdings {agent.get_holdings(tkn_sell)} do not match expected sell quantity {-trade_size} for route {route} in direction {direction}"
                )
        for tkn in set(agent.holdings) - {tkn_buy, tkn_sell}:
            if agent.get_holdings(tkn) != 0:
                raise ValueError(
                    f"Agent holdings {agent.get_holdings(tkn)} for token {tkn} are not zero after trade {route} in direction {direction}"
                )


def test_sell_spot_intermediate():
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': mpf(1000001910), 'LRNA': mpf(100009000)},
            "USDT": {'liquidity': mpf(1003000), 'LRNA': mpf(1000060)},
            "DOT": {'liquidity': mpf(10000000), 'LRNA': mpf(102060000)},
        },
        preferred_stablecoin="USDT",
        asset_fee=0.0025,
        lrna_fee=0.0005
    )
    stablepool1 = StableSwapPoolState(
        tokens={"stable1": mpf(123000000), "stable2": mpf(132000000)},
        amplification=1002,
        trade_fee=0.0100,
        unique_id="stablepool1",
        precision=1e-18,
        spot_price_precision=1e-18,
        peg=1.111
    )
    binance = FixedPriceExchange(
        tokens={"stablepool1": mpf(1), "DOT": mpf(10.2)},
        unique_id="binance",
        fee=0.0005
    )
    router = OmnipoolRouter([omnipool, stablepool1, binance])
    agent = Agent(enforce_holdings=False)
    tkn_sell = "stable1"
    tkn_buy = "HDX"
    trade_size = mpf(1) / 100000
    assert router.find_best_route(
        tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=trade_size
    )[1].exchange == "binance"
    expected_price = router.sell_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
    router.swap(agent, tkn_sell=tkn_sell, tkn_buy=tkn_buy, sell_quantity=trade_size)
    actual_price = agent.get_holdings(tkn_buy) / -agent.get_holdings(tkn_sell)
    if actual_price != pytest.approx(expected_price, rel=1e-12):
        raise ValueError(f"Actual price {actual_price} does not match expected price {expected_price}")
    if agent.get_holdings(tkn_sell) != -trade_size:
        raise ValueError(f"Agent holdings {agent.get_holdings(tkn_buy)} do not match expected buy quantity {trade_size}")


def test_buy_spot_intermediate():
    omnipool= OmnipoolState(
        tokens={
            "HDX": {'liquidity': mpf(100019010), 'LRNA': mpf(10009000)},
            "USDT": {'liquidity': mpf(1003000), 'LRNA': mpf(1000060)},
            "DOT": {'liquidity': mpf(10000000), 'LRNA': mpf(102060000)},
        },
        preferred_stablecoin="USDT",
        asset_fee=0.0025,
        lrna_fee=0.0005
    )
    stablepool1 = StableSwapPoolState(
        tokens={"stable1": mpf(123000000), "stable2": mpf(132000000)},
        amplification=1002,
        trade_fee=0.0101,
        unique_id="stablepool1",
        precision=1e-15,
        spot_price_precision=1e-15,
        peg=1.111,
    )
    binance = FixedPriceExchange(
        tokens={"stablepool1": mpf(1), "DOT": mpf(10.2)},
        unique_id="binance",
        fee=0.0005
    )
    router = OmnipoolRouter([omnipool, stablepool1, binance])
    agent = Agent(enforce_holdings=False)
    tkn_sell = "stable1"
    tkn_buy = "HDX"
    trade_size = mpf(1) / 100000
    assert router.find_best_route(
        tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=trade_size
    )[1].exchange == "binance"
    expected_price = router.buy_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
    router.swap(agent, tkn_sell=tkn_sell, tkn_buy=tkn_buy, buy_quantity=trade_size)
    actual_price = -agent.get_holdings(tkn_sell) / agent.get_holdings(tkn_buy)
    if actual_price != pytest.approx(expected_price, rel=1e-12):
        raise ValueError(f"Actual price {actual_price} does not match expected price {expected_price}")
    if agent.get_holdings(tkn_buy) != trade_size:
        raise ValueError(f"Agent holdings {agent.get_holdings(tkn_buy)} do not match expected buy quantity {trade_size}")


def test_calculate_swap():
    omnipool = OmnipoolState(
        tokens={
            "HDX": {'liquidity': mpf(10001910), 'LRNA': mpf(1000900)},
            "USDT": {'liquidity': mpf(1003000), 'LRNA': mpf(1000060)},
            "DOT": {'liquidity': mpf(100000), 'LRNA': mpf(1020600)},
            "stablepool1": {'liquidity': mpf(1500000), 'LRNA': mpf(1567000)},
            "stablepool2": {'liquidity': mpf(1054000), 'LRNA': mpf(1045675)},
        },
        preferred_stablecoin="USDT",
        asset_fee=0.0025,
        lrna_fee=0.0005
    )
    stablepool1 = StableSwapPoolState(
        tokens={"stable1": mpf(1230000), "stable2": mpf(1320000)},
        amplification=1002,
        trade_fee=0.0100, unique_id="stablepool1",
        precision=1e-12,
        spot_price_precision=1e-12
    )
    stablepool2 = StableSwapPoolState(
        tokens={"stable2": mpf(330000), "stable3": mpf(345000)},
        amplification=101,
        trade_fee=0.0001, unique_id="stablepool2",
        precision=1e-12,
        spot_price_precision=1e-12
    )
    router = OmnipoolRouter(
        exchanges=[omnipool, stablepool1, stablepool2],
    )
    routes = [
        [
            Trade(exchange="omnipool", tkn_sell="DOT", tkn_buy="stablepool1"),
            Trade(exchange="stablepool1", tkn_sell="stablepool1", tkn_buy="stable1")
        ], [
            Trade(exchange="stablepool2", tkn_sell="stable2", tkn_buy="stablepool2"),
            Trade(exchange="omnipool", tkn_sell="stablepool2", tkn_buy="DOT")
        ], [
            Trade(exchange="stablepool1", tkn_sell="stable1", tkn_buy="stablepool1"),
            Trade(exchange="omnipool", tkn_sell="stablepool1", tkn_buy="stablepool2"),
            Trade(exchange="stablepool2", tkn_sell="stablepool2", tkn_buy="stable3")
        ], [
            Trade(exchange="stablepool1", tkn_sell="stable1", tkn_buy="stable2"),
            Trade(exchange="stablepool2", tkn_sell="stable2", tkn_buy="stable3"),
        ]
    ]
    for route in routes:
        direction: Literal['buy', 'sell']
        for direction in ['sell', 'buy']:
            agent = Agent(enforce_holdings=False)
            tkn_buy = route[-1].tkn_buy
            tkn_sell = route[0].tkn_sell
            if direction == 'buy':
                route.reverse()
            trade_size = mpf(100)
            router_copy = router.copy()
            router_copy.swap_route(
                agent=agent, route=route,
                buy_quantity=trade_size if direction == 'buy' else None,
                sell_quantity=trade_size if direction == 'sell' else None
            )
            actual = (-agent.get_holdings(tkn_sell) if direction == 'buy' else agent.get_holdings(
                tkn_buy))
            if direction == 'buy':
                expected = router.calculate_sell_from_buy(
                    tkn_sell=tkn_sell, tkn_buy=tkn_buy, buy_quantity=trade_size, route=route
                )
            else:
                expected = router.calculate_buy_from_sell(
                    tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=trade_size, route=route
                )
            if actual != pytest.approx(expected, rel=1e-12):
                raise ValueError(
                    f"Actual {round(actual, 12)} does not match expected {round(expected, 12)} for tkn_buy={tkn_buy}, tkn_sell={tkn_sell} in direction {direction}"
                )
            if direction == 'buy' and agent.get_holdings(tkn_buy) != trade_size:
                raise ValueError(
                    f"Agent holdings {round(agent.get_holdings(tkn_buy), 12)} do not match expected buy quantity {trade_size} for tkn_buy={tkn_buy}, tkn_sell={tkn_sell} in direction {direction}"
                )
            if direction == 'sell' and agent.get_holdings(tkn_sell) != -trade_size:
                raise ValueError(
                    f"Agent holdings {round(agent.get_holdings(tkn_sell), 12)} do not match expected sell quantity {-trade_size} for tkn_buy={tkn_buy}, tkn_sell={tkn_sell} in direction {direction}"
                )
            for tkn in set(agent.holdings) - {tkn_buy, tkn_sell}:
                if agent.get_holdings(tkn) != 0:
                    raise ValueError(
                        f"Agent holdings {agent.get_holdings(tkn)} for token {tkn} are not zero after trade {route} in direction {direction}"
                    )
