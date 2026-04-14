import copy
import functools
from datetime import timedelta

import pytest
from hypothesis import given, strategies as st, settings, reproduce_failure
from mpmath import mp, mpf

from hydradx.model import run
from hydradx.model.amm import stableswap_amm as stableswap
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.stableswap_amm import StableSwapPoolState, simulate_swap, simulate_remove_uniform, \
    simulate_add_liquidity, balance_ratio_at_price
from hydradx.model.amm.trade_strategies import random_swaps, stableswap_arbitrage
from hydradx.tests.strategies_omnipool import stableswap_config

mp.dps = 50

asset_price_strategy = st.floats(min_value=0.01, max_value=1000)
asset_quantity_strategy = st.floats(min_value=1000, max_value=1000000)
fee_strategy = st.floats(min_value=0, max_value=0.1, allow_nan=False)
trade_quantity_strategy = st.floats(min_value=-1000, max_value=1000)
asset_number_strategy = st.integers(min_value=2, max_value=5)


def stable_swap_equation(pool: StableSwapPoolState):  # d: float, a: float, n: int, reserves: list):
    """
    this is the equation that should remain true at all times within a stableswap pool
    """
    a = pool.ann
    d = pool.d
    n = pool.n_coins
    reserves = list(pool.liquidity.values())
    side1 = a * sum(reserves) + d
    side2 = a * d + d ** (n + 1) / (n ** n * functools.reduce(lambda i, j: i * j, reserves))
    return side1 == pytest.approx(side2)


def test_big_buy_fails():
    initial_pool = StableSwapPoolState(
        tokens={'A': 1000000, 'B': 1000000},
        amplification=1000,
        trade_fee=0.0
    )
    agent = Agent(enforce_holdings=False)
    new_pool, new_agent = simulate_swap(
        initial_pool, agent, tkn_sell='A', tkn_buy='B', buy_quantity=10000000
    )
    if not new_pool.fail:
        raise AssertionError('Swap should have failed due to insufficient funds.')


@given(stableswap_config(trade_fee=0))
def test_swap_invariant(initial_pool: StableSwapPoolState):
    # print(f'testing with {len(initial_pool.asset_list)} assets')
    initial_state = GlobalState(
        pools={
            'stableswap': initial_pool
        },
        agents={
            'trader': Agent(
                holdings={tkn: 100000 for tkn in initial_pool.asset_list},
                trade_strategy=random_swaps(
                    pool_id='stableswap',
                    amount={tkn: 1000 for tkn in initial_pool.asset_list},
                    randomize_amount=True
                )
            )
        }
    )

    new_state = initial_state.copy()
    d = new_state.pools['stableswap'].calculate_d()
    for n in range(10):
        new_state = new_state.agents['trader'].trade_strategy.execute(new_state, agent_id='trader')
        new_d = new_state.pools['stableswap'].calculate_d()
        if new_d != pytest.approx(d):
            raise AssertionError('Invariant has varied.')
    if initial_state.total_wealth() != pytest.approx(new_state.total_wealth()):
        raise AssertionError('Some assets were lost along the way.')


def test_swap_agent_changes():
    trade_fee = 0.0001
    sell_amt = 1000
    pool = StableSwapPoolState(
        tokens={'A': 1000000, 'B': 1000000},
        amplification=1000,
        trade_fee=trade_fee
    )
    agent = Agent(holdings={'A': 1000000, 'B': 1000000})
    new_pool, new_agent = simulate_swap(
        pool, agent, tkn_sell='A', tkn_buy='B', sell_quantity=sell_amt
    )
    if new_agent.holdings['A'] != agent.holdings['A'] - sell_amt:
        raise AssertionError('Agent holdings not updated properly.')
    if new_agent.holdings['B'] + new_pool.liquidity['B'] != agent.holdings['B'] + pool.liquidity['B']:
        raise AssertionError('Agent holdings not updated properly.')

    empty_agent = Agent(enforce_holdings=True)
    fail_pool, fail_agent = simulate_swap(pool, empty_agent, tkn_sell='A', tkn_buy='B', sell_quantity=sell_amt)
    if not fail_pool.fail:
        raise AssertionError('Swap should have failed due to insufficient funds.')

    empty_agent = Agent(enforce_holdings=False)
    new_pool, new_agent = simulate_swap(pool, empty_agent, tkn_sell='A', tkn_buy='B', sell_quantity=sell_amt)
    if new_agent.get_holdings('A') != -sell_amt:
        raise AssertionError('Agent holdings not updated properly.')
    if new_agent.get_holdings('B') + new_pool.liquidity['B'] != pool.liquidity['B']:
        raise AssertionError('Agent holdings not updated properly.')


@given(st.integers(min_value=1000, max_value=1000000),
       st.integers(min_value=1000, max_value=1000000),
       st.integers(min_value=10, max_value=1000)
       )
def test_spot_price_two_assets(token_a: int, token_b: int, amp: int):
    initial_pool = StableSwapPoolState(
        tokens={"A": token_a, "B": token_b},
        amplification=amp,
        trade_fee=0.0,
        unique_id='stableswap'
    )
    spot_price_initial = initial_pool.price("B", "A")

    trade_size = 1
    initial_agent = Agent(holdings={"A": 1, "B": 1})
    swap_state, swap_agent = stableswap.simulate_swap(
        old_state=initial_pool,
        old_agent=initial_agent,
        tkn_sell="A", tkn_buy="B", sell_quantity=trade_size
    )
    delta_a = swap_state.liquidity["A"] - initial_pool.liquidity["A"]
    delta_b = initial_pool.liquidity["B"] - swap_state.liquidity["B"]
    exec_price = delta_a / delta_b

    spot_price_final = swap_state.price("B", "A")

    if spot_price_initial > exec_price and (spot_price_initial - exec_price) / spot_price_initial > 10e-10:
        raise AssertionError('Initial spot price should be lower than execution price.')
    if exec_price > spot_price_final and (exec_price - spot_price_final) / spot_price_final > 10e-10:
        raise AssertionError('Execution price should be lower than final spot price.')


@given(st.integers(min_value=1000, max_value=1000000),
       st.integers(min_value=1000, max_value=1000000),
       st.integers(min_value=1000, max_value=1000000),
       st.integers(min_value=1000, max_value=1000000),
       st.integers(min_value=10, max_value=1000),
       st.integers(min_value=1, max_value=3),
       st.floats(min_value=0.0001, max_value=1000),
       st.floats(min_value=0.0001, max_value=1000),
       st.floats(min_value=0.0001, max_value=1000)
       )
def test_spot_price(token_a: int, token_b: int, token_c: int, token_d: int, amp: int, i: int, peg1: float, peg2:float,
                    peg3: float):
    initial_pool = StableSwapPoolState(
        tokens={"A": mpf(token_a), "B": mpf(token_b), "C": mpf(token_c), "D": mpf(token_d)},
        amplification=amp,
        trade_fee=0.0,
        unique_id='stableswap',
        peg=[peg1, peg2, peg3]
    )
    tkns = ["A", "B", "C", "D"]
    spot_price_initial = initial_pool.price(tkns[i], "A")

    trade_size = 1
    initial_agent = Agent(holdings={"A": 1.1, "B": 1.1, "C": 1.1, "D": 1.1})
    swap_agent = initial_agent.copy()
    swap_pool = initial_pool.copy().swap(swap_agent, tkn_sell="A", tkn_buy=tkns[i], sell_quantity=trade_size)
    delta_a = swap_pool.liquidity["A"] - initial_pool.liquidity["A"]
    delta_b = initial_pool.liquidity[tkns[i]] - swap_pool.liquidity[tkns[i]]
    exec_price = delta_a / delta_b

    spot_price_final = swap_pool.price(tkns[i], "A")

    if spot_price_initial > exec_price and (spot_price_initial - exec_price) / spot_price_initial > 1e-10:
        raise AssertionError('Initial spot price should be lower than execution price.')
    if exec_price > spot_price_final and (exec_price - spot_price_final) / spot_price_final > 1e-10:
        raise AssertionError('Execution price should be lower than final spot price.')


@given(st.integers(min_value=1000, max_value=1000000),
       st.integers(min_value=1000, max_value=1000000),
       st.integers(min_value=10, max_value=1000)
       )
def test_share_price(token_a: int, token_b: int, amp: int):
    tokens = {"A": token_a, "B": token_b}
    initial_pool = StableSwapPoolState(
        tokens=tokens,
        amplification=amp,
        trade_fee=0.0,
        unique_id='stableswap'
    )

    share_price_initial = initial_pool.share_price()

    agent = Agent(holdings={"A": 100000000, "B": 100000000})
    delta_tkn = 1
    shares_initial = initial_pool.shares
    add_pool = initial_pool.copy()
    add_pool.add_liquidity(agent, quantity=delta_tkn, tkn_add="A")
    shares_final = add_pool.shares
    delta_a = add_pool.liquidity["A"] - tokens["A"]
    delta_s = shares_final - shares_initial
    exec_price = delta_a / delta_s

    if share_price_initial > exec_price and (share_price_initial - exec_price) / share_price_initial > 10e-10:
        raise AssertionError('Initial share price should be lower than execution price.')

    # now we test withdraw

    delta_s = agent.holdings['stableswap']
    share_price_initial = add_pool.share_price()
    a_initial = add_pool.liquidity['A']
    withdraw_pool = add_pool.copy()
    withdraw_pool.remove_liquidity(agent, shares_removed=delta_s, tkn_remove='A')
    a_final = withdraw_pool.liquidity['A']
    exec_price = (a_initial - a_final) / delta_s

    if share_price_initial < exec_price and (exec_price - share_price_initial) / share_price_initial > 10e-10:
        raise AssertionError('Initial share price should be higher than execution price.')


@given(stableswap_config(trade_fee=0))
def test_round_trip_dy(initial_pool: StableSwapPoolState):
    d = initial_pool.calculate_d()
    asset_a = initial_pool.asset_list[0]
    other_reserves = {tkn: initial_pool.liquidity[tkn] for tkn in initial_pool.liquidity}
    other_reserves.pop(asset_a)
    y = initial_pool.calculate_y(reserves=other_reserves, d=d)
    if y != pytest.approx(initial_pool.liquidity[asset_a]) or y < initial_pool.liquidity[asset_a]:
        raise AssertionError('Round-trip calculation incorrect.')
    balances_list = list(initial_pool.modified_balances(delta={asset_a: 1}).values())
    modified_d = initial_pool.calculate_d(balances_list)
    if initial_pool.calculate_y(reserves=other_reserves, d=modified_d) != pytest.approx(y + 1):
        raise AssertionError('Round-trip calculation incorrect.')


@given(stableswap_config(precision=0.000000001, trade_fee=0))
def test_remove_asset(initial_pool: StableSwapPoolState):
    initial_agent = Agent(
        holdings={tkn: 0 for tkn in initial_pool.asset_list}
    )
    # agent holds all the shares
    tkn_remove = initial_pool.asset_list[0]
    pool_name = initial_pool.unique_id
    delta_shares = min(initial_pool.shares / 2, 100)
    initial_agent.holdings.update({initial_pool.unique_id: delta_shares + 1})
    withdraw_shares_pool, withdraw_shares_agent = stableswap.simulate_remove_liquidity(
        initial_pool, initial_agent, delta_shares, tkn_remove
    )
    delta_tkn = withdraw_shares_agent.holdings[tkn_remove] - initial_agent.holdings[tkn_remove]
    withdraw_asset_pool, withdraw_asset_agent = initial_pool.copy(), initial_agent.copy()
    withdraw_asset_pool.withdraw_asset(
        withdraw_asset_agent, delta_tkn, tkn_remove
    )
    if (
            withdraw_asset_agent.holdings[tkn_remove] != pytest.approx(withdraw_shares_agent.holdings[tkn_remove])
            or withdraw_asset_agent.holdings[pool_name] != pytest.approx(withdraw_shares_agent.holdings[pool_name])
            or withdraw_shares_pool.liquidity[tkn_remove] != pytest.approx(withdraw_asset_pool.liquidity[tkn_remove])
            or withdraw_shares_pool.shares != pytest.approx(withdraw_asset_pool.shares)
    ):
        raise AssertionError("Asset values don't match.")


@given(liq = st.lists(st.floats(min_value=100000, max_value=1000000), min_size=3, max_size=3),
       amp = st.floats(min_value=5, max_value=1000),
       pegs = st.lists(st.floats(min_value=0.1, max_value=10), min_size=2, max_size=2))
def test_buy_shares_with_add_liquidity(liq: list[float], amp: float, pegs: list[float]):
    tokens = {"A": liq[0], "B": liq[1], "C": liq[2]}
    initial_pool = StableSwapPoolState(tokens, mpf(amp), trade_fee=mpf(0), peg=pegs)
    initial_agent = Agent(holdings={tkn: 0 for tkn in initial_pool.asset_list + [initial_pool.unique_id]})
    tkn_add = initial_pool.asset_list[0]
    pool_name = initial_pool.unique_id
    delta_tkn = 10
    initial_agent.holdings.update({tkn_add: 2 * delta_tkn})

    add_liquidity_pool, add_liquidity_agent = stableswap.simulate_add_liquidity(
        initial_pool, initial_agent, delta_tkn, tkn_add
    )
    delta_shares = add_liquidity_agent.holdings[pool_name] - initial_agent.holdings[pool_name]
    buy_shares_pool, buy_shares_agent = stableswap.simulate_buy_shares(
        initial_pool.copy(), initial_agent.copy(), delta_shares, tkn_add
    )

    if delta_shares != buy_shares_agent.holdings[pool_name]:
        raise AssertionError("Agent shares don't match.")
    if add_liquidity_agent.holdings[tkn_add] != pytest.approx(buy_shares_agent.holdings[tkn_add], rel=1e-12):
        raise AssertionError("Agent tkn remaining doesn't match.")
    if add_liquidity_pool.liquidity[tkn_add] != pytest.approx(buy_shares_pool.liquidity[tkn_add], rel=1e-12):
        raise AssertionError("Pool liquidity doesn't match.")
    if add_liquidity_pool.shares != pytest.approx(buy_shares_pool.shares, rel=1e-12):
        raise AssertionError("Pool shares don't match.")
    if add_liquidity_pool.calculate_d() != pytest.approx(buy_shares_pool.calculate_d(), rel=1e-12):
        raise AssertionError("Pool d doesn't match.")


@given(liq = st.lists(st.floats(min_value=100000, max_value=1000000), min_size=3, max_size=3),
       amp = st.floats(min_value=5, max_value=1000),
       pegs = st.lists(st.floats(min_value=0.1, max_value=10), min_size=2, max_size=2),
       fee = st.floats(min_value=0, max_value=0.1),
       add_pct = st.floats(min_value=1e-7, max_value=0.5))
def test_buy_shares_increases_invariant_to_shares_ratio(liq, amp, pegs, fee, add_pct):
    tokens = {"A": liq[0], "B": liq[1], "C": liq[2]}
    initial_pool = StableSwapPoolState(tokens, mpf(amp), trade_fee=mpf(fee), peg=pegs)
    tkn_add = initial_pool.asset_list[0]
    add_amt = add_pct * initial_pool.liquidity[tkn_add]
    initial_agent = Agent(holdings={tkn_add: add_amt * 2})

    # make sure we have delta_shares amount that agent holdings can handle
    temp_pool, temp_agent = stableswap.simulate_add_liquidity(initial_pool, initial_agent, add_amt, tkn_add)
    delta_shares = temp_agent.holdings[temp_pool.unique_id]

    init_ratio = initial_pool.d / initial_pool.shares
    buy_shares_pool, buy_shares_agent = stableswap.simulate_buy_shares(
        initial_pool, initial_agent, delta_shares, tkn_add
    )

    if buy_shares_pool.fail:
        raise AssertionError("Agent has insufficient holdings.")

    final_ratio = buy_shares_pool.d / buy_shares_pool.shares

    if buy_shares_agent.holdings[initial_pool.unique_id] != delta_shares:
        raise AssertionError("Agent shares don't match.")
    if (init_ratio - final_ratio)/init_ratio > 1e-15:
        raise AssertionError("Invariant not held.")


@given(
    st.floats(min_value=0.0001, max_value=.9999),
    st.integers(min_value=1, max_value=1000000),
    st.floats(min_value=0.0001, max_value=0.1)
)
def test_arbitrage_profitability(trade_fraction, amp, fee):
    initial_pool = StableSwapPoolState(
        tokens={
            tkn: mpf(10000) for tkn in ['R' + str(n) for n in range(1, 5)]
        },
        amplification=amp,
        trade_fee=fee
    )
    trader = Agent(
        holdings={tkn: 10000000000 for tkn in initial_pool.asset_list}
    )
    arbitrageur = Agent(
        holdings={tkn: 10000000000 for tkn in initial_pool.asset_list},
        trade_strategy=stableswap_arbitrage(
            pool_id='stableswap', minimum_profit=0, precision=1e-6
        )
    )
    swapped_pool: StableSwapPoolState = initial_pool.copy().swap(
        agent=trader,
        tkn_sell='R1', tkn_buy='R2',
        buy_quantity=trade_fraction * initial_pool.liquidity['R1']
    )
    pre_arb_state = GlobalState(
        pools={'stableswap': swapped_pool},
        agents={'Arbitrageur': arbitrageur},
        external_market={'R1': 1, 'R2': 1, 'R3': 1, 'R4': 1}
    )
    arbitrageur.trade_strategy.execute(pre_arb_state, 'Arbitrageur')
    if (
            sum(arbitrageur.holdings.values()) < sum(arbitrageur.initial_holdings.values())
    ):
        raise AssertionError(
            f"Arbitrageur lost money. "
            f"(start: {sum(arbitrageur.initial_holdings.values())}, end: {sum(arbitrageur.holdings.values())})"
        )


@given(
    st.floats(min_value=0.0001, max_value=.9999),
    st.integers(min_value=1, max_value=10000)
)
def test_arbitrage_efficacy(trade_fraction, amp):
    initial_pool = StableSwapPoolState(
        tokens={
            tkn: mpf(10000) for tkn in ['R' + str(n) for n in range(1, 5)]
        },
        amplification=amp,
        trade_fee=0
    )
    trader = Agent(
        holdings={tkn: 10000000000 for tkn in initial_pool.asset_list}
    )
    arbitrageur = Agent(
        holdings={tkn: 10000000000 for tkn in initial_pool.asset_list},
        trade_strategy=stableswap_arbitrage(
            pool_id='stableswap', minimum_profit=0, precision=1e-6
        )
    )
    swapped_pool: StableSwapPoolState = initial_pool.copy().swap(
        agent=trader,
        tkn_sell='R1', tkn_buy='R2',
        buy_quantity=trade_fraction * initial_pool.liquidity['R1']
    )
    pre_arb_state = GlobalState(
        pools={'stableswap': swapped_pool},
        agents={'Arbitrageur': arbitrageur},
        external_market={'R1': 1, 'R2': 1, 'R3': 1, 'R4': 1}
    )
    final_state = arbitrageur.trade_strategy.execute(pre_arb_state.copy(), 'Arbitrageur')
    final_pool = final_state.pools['stableswap']

    if (
            initial_pool.price('R1', 'R2')
            != pytest.approx(final_pool.price('R1', 'R2'), abs=1e-6)
    ):
        raise AssertionError(f"Arbitrageur didn't keep the price stable."
                             f"({initial_pool.price('R1', 'R2')},"
                             f"{final_pool.price('R1', 'R2')}")


@given(stableswap_config(trade_fee=0))
def test_add_remove_liquidity(initial_pool: StableSwapPoolState):
    lp_tkn = initial_pool.asset_list[0]
    lp = Agent(
        holdings={lp_tkn: 10000}
    )

    add_liquidity_state, add_liquidity_agent = stableswap.simulate_add_liquidity(
        initial_pool, old_agent=lp, quantity=10000, tkn_add=lp_tkn
    )
    if not stable_swap_equation(add_liquidity_state):
        raise AssertionError('Stableswap equation does not hold after add liquidity operation.')

    remove_liquidity_state, remove_liquidity_agent = stableswap.simulate_remove_liquidity(
        add_liquidity_state,
        add_liquidity_agent,
        quantity=add_liquidity_agent.holdings[initial_pool.unique_id],
        tkn_remove=lp_tkn
    )
    if not stable_swap_equation(remove_liquidity_state):
        raise AssertionError('Stableswap equation does not hold after remove liquidity operation.')
    if remove_liquidity_agent.holdings[lp_tkn] != pytest.approx(lp.holdings[lp_tkn]):
        raise AssertionError('LP did not get the same balance back when withdrawing liquidity.')


#
# def test_curve_style_withdraw_fees():
#     initial_state = stableswap.StableSwapPoolState(
#         tokens={
#             'USDA': 1000000,
#             'USDB': 1000000,
#             'USDC': 1000000,
#             'USDD': 1000000,
#         }, amplification=100, trade_fee=0.003,
#         unique_id='test_pool'
#     )
#     initial_agent = Agent(
#         holdings={'USDA': 100000}
#     )
#     test_state, test_agent = stableswap.simulate_add_liquidity(
#         old_state=initial_state.copy(),
#         old_agent=initial_agent.copy(),
#         quantity=initial_agent.holdings['USDA'],
#         tkn_add='USDA',
#     )
#
#     stable_state, stable_agent = stableswap.simulate_remove_liquidity(
#         old_state=test_state.copy(),
#         old_agent=test_agent.copy(),
#         quantity=test_agent.holdings['test_pool'],
#         tkn_remove='USDB'
#     )
#     effective_fee_withdraw = 1 - stable_agent.holdings['USDB'] / initial_agent.holdings['USDA']
#
#     swap_state, swap_agent = stableswap.simulate_swap(
#         initial_state.copy(),
#         initial_agent.copy(),
#         tkn_sell='USDA',
#         tkn_buy='USDB',
#         sell_quantity=initial_agent.holdings['USDA']
#     )
#     effective_fee_swap = 1 - swap_agent.holdings['USDB'] / initial_agent.holdings['USDA']
#
#     if effective_fee_withdraw <= effective_fee_swap:
#         raise AssertionError('Withdraw fee is not higher than swap fee.')


@given(
    st.integers(min_value=1, max_value=1000000),
    st.integers(min_value=10000, max_value=10000000),
)
def test_exploitability(initial_lp: int, trade_size: int):
    assets = ['USDA', 'USDB']
    initial_tvl = 1000000

    initial_state = StableSwapPoolState(
        tokens={tkn: mpf(initial_tvl / len(assets)) for tkn in assets},
        amplification=1000,
        trade_fee=0
    )
    initial_agent = Agent(
        holdings={tkn: mpf(initial_lp / len(assets)) for tkn in assets},
    )

    lp_state, lp_agent = initial_state.copy(), initial_agent.copy()
    for tkn in initial_state.asset_list:
        lp_state.add_liquidity(
            agent=lp_agent,
            quantity=lp_agent.get_holdings(tkn),
            tkn_add=tkn
        )

    trade_state, trade_agent = lp_state.copy(), lp_agent.copy()
    trade_agent.holdings['USDA'] = trade_size
    init_buy_tkn = trade_agent.get_holdings('USDB')
    trade_state.swap(
        agent=trade_agent,
        tkn_sell='USDA',
        tkn_buy='USDB',
        sell_quantity=trade_size
    )
    buy_amt = trade_agent.get_holdings('USDB') - init_buy_tkn

    withdraw_state, withdraw_agent = trade_state.copy(), trade_agent.copy()
    withdraw_state.remove_liquidity(
        agent=withdraw_agent,
        shares_removed=trade_agent.get_holdings('stableswap'),
        tkn_remove='USDA'
    )

    max_arb_size = buy_amt
    min_arb_size = 0

    for i in range(10):
        final_state, final_agent = withdraw_state.copy(), withdraw_agent.copy()
        arb_size = (max_arb_size - min_arb_size) / 2 + min_arb_size
        final_state.swap(
            agent=final_agent,
            tkn_sell='USDB',
            tkn_buy='USDA',
            sell_quantity=arb_size
        )

        profit = sum(final_agent.holdings.values()) - trade_size - initial_lp
        if profit > 0:
            raise AssertionError(f'Agent profited by exploit ({profit}).')

        if initial_state.price('USDA', 'USDB') < final_state.price('USDA', 'USDB'):
            min_arb_size = arb_size
        elif initial_state.price('USDA', 'USDB') > final_state.price('USDA', 'USDB'):
            max_arb_size = arb_size
        else:
            break


@given(
    st.integers(min_value=1, max_value=1000000),
    st.floats(min_value=0.00001, max_value=0.99999)
)
def test_swap_one(amplification, swap_fraction):
    initial_state = StableSwapPoolState(
        tokens={
            'USDA': mpf(1000000),
            'USDB': mpf(1000000),
            'USDC': mpf(1000000),
            'USDD': mpf(1000000),
        }, amplification=amplification, trade_fee=0,
    )
    stablecoin = initial_state.asset_list[-1]
    tkn_sell = initial_state.asset_list[0]
    buy_quantity = initial_state.liquidity[tkn_sell] * swap_fraction
    initial_agent = Agent(enforce_holdings=False)
    sell_agent = initial_agent.copy()
    sell_state = initial_state.copy().swap_one(
        agent=sell_agent,
        tkn_sell=tkn_sell,
        quantity=buy_quantity
    )
    if not stable_swap_equation(sell_state):
        raise AssertionError('Stableswap equation does not hold after swap operation.')

    for tkn in initial_state.asset_list:
        if (
                sell_state.price(tkn, stablecoin)
                != pytest.approx(initial_state.price(tkn, stablecoin))
                and tkn != tkn_sell
        ):
            raise AssertionError('Spot price changed for non-swapped token.')

    if sell_state.price(tkn_sell, stablecoin) >= initial_state.price(tkn_sell, stablecoin):
        raise AssertionError('Spot price increased for swapped token.')

    if sell_state.d != pytest.approx(initial_state.d):
        raise AssertionError('D changed after sell operation.')

    tkn_buy = sell_state.asset_list[0]
    buy_agent = sell_agent.copy()
    buy_state = sell_state.copy().swap_one(
        agent=buy_agent,
        tkn_buy=tkn_buy,
        quantity=buy_quantity
    )
    if not stable_swap_equation(buy_state):
        raise AssertionError('Stableswap equation does not hold after swap operation.')

    for tkn in initial_state.asset_list:
        if (
                buy_state.price(tkn, stablecoin)
                != pytest.approx(initial_state.price(tkn, stablecoin))
                and tkn != tkn_buy
        ):
            raise AssertionError('Spot price changed for non-swapped token.')

    if buy_state.price(tkn_buy, stablecoin) <= sell_state.price(tkn_buy, stablecoin):
        raise AssertionError('Spot price decreased for swapped token.')

    if buy_state.d != pytest.approx(initial_state.d):
        raise AssertionError('D changed after buy operation.')


# @given(st.integers(min_value=1, max_value=999))
def test_amplification_change_exploit():  # (end_amp):
    start_amp = 1000
    end_amp = 100
    initial_pool = StableSwapPoolState(
        tokens={
            'USDA': mpf(1000000),
            'USDB': mpf(1000000),
            'USDC': mpf(1000000),
            'USDD': mpf(1000000),
        },
        amplification=start_amp,
        trade_fee=0.001,
    )
    initial_agent = Agent(
        holdings={'USDA': 10000000000000, 'USDB': 10000000000000},
        # trade_strategy=stableswap_arbitrage(pool_id='stableswap', minimum_profit=0, precision=0.000001)
    )
    initial_state = GlobalState(
        pools={
            'stableswap': initial_pool
        },
        agents={
            'trader': initial_agent,
            'arbitrageur': Agent(
                holdings={tkn: 10000000000000 for tkn in initial_pool.asset_list},
                trade_strategy=stableswap_arbitrage(pool_id='stableswap', minimum_profit=1, precision=0.0001)
            )
        },
        external_market={
            'USDA': 1,
            'USDB': 1,
            'USDC': 1,
            'USDD': 1,
        }
    )
    sell_quantity = initial_pool.liquidity['USDA'] * 10
    sell_state = initial_state.copy()
    sell_state.pools['stableswap'].swap(
        agent=sell_state.agents['trader'],
        tkn_sell='USDA',
        tkn_buy='USDB',
        sell_quantity=sell_quantity
    )
    duration = int((start_amp - end_amp) / end_amp * 99.9)
    sell_state.pools['stableswap'].set_amplification(end_amp, duration)
    events = run.run(sell_state, time_steps=duration, silent=True)
    final_pool: StableSwapPoolState = events[-1].pools['stableswap']
    final_agent: Agent = events[-1].agents['trader']
    final_pool.swap(
        agent=final_agent,
        tkn_sell='USDB',
        tkn_buy='USDA',
        buy_quantity=sell_quantity
    )
    loss = sum(initial_pool.liquidity.values()) - sum(final_pool.liquidity.values())
    if loss > 0:
        raise AssertionError(F"Pool lost money. loss: {round(loss / sum(initial_pool.liquidity.values()) * 100, 5)}%")


@given(
    liquidity_stepdown=st.integers(min_value=-1000000, max_value=1000000),
    amplification=st.integers(min_value=1, max_value=10000),
    peg=st.lists(st.floats(min_value=0.1, max_value=10), min_size=2, max_size=2),
    blocks_since_update=st.integers(min_value=0, max_value=100)
)
def test_buy_sell_spot_feeless(
        liquidity_stepdown: int,
        amplification: int,
        peg: list,
        blocks_since_update: int
):
    assets_number = 3
    trade_fee = 0.0
    base_liquidity = mpf(10000000)
    max_peg_update = 0.001
    initial_state = StableSwapPoolState(
        tokens={
            tkn: base_liquidity + liquidity_stepdown * n
            for n, tkn in enumerate(['R' + str(n) for n in range(1, assets_number + 1)])
        },
        amplification=amplification,
        trade_fee=trade_fee,
        peg=peg,
        peg_target=peg,
        max_peg_update=max_peg_update
    )
    initial_state.time_step = blocks_since_update
    tkn_sell = 'R1'
    tkn_buy = 'R2'
    r1_per_r2 = initial_state.buy_spot(tkn_buy=tkn_buy, tkn_sell=tkn_sell)
    r2_per_r1 = initial_state.sell_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
    if r2_per_r1 != pytest.approx(1 / r1_per_r2, rel=1e-20):
        raise AssertionError('Inconsistent spot prices.')


@given(
    liquidity_stepdown=st.integers(min_value=-1000000, max_value=1000000),
    amplification=st.integers(min_value=1, max_value=10000),
    trade_fee=st.floats(min_value=0, max_value=0.1),
    peg=st.lists(st.floats(min_value=0.1, max_value=10), min_size=2, max_size=2),
    peg_target=st.lists(st.floats(min_value=0.1, max_value=10), min_size=2, max_size=2),
    blocks_since_update=st.integers(min_value=0, max_value=100),
    max_peg_update=st.floats(min_value=0.0001, max_value=0.01)
)
def test_buy_sell_spot(
        liquidity_stepdown: int,
        amplification: int,
        trade_fee: float,
        peg: list,
        peg_target: list,
        blocks_since_update: int,
        max_peg_update: float
):
    assets_number = 3
    base_liquidity = mpf(10000000)
    initial_state = StableSwapPoolState(
        tokens={
            tkn: base_liquidity + liquidity_stepdown * n
            for n, tkn in enumerate(['R' + str(n) for n in range(1, assets_number + 1)])
        },
        amplification=amplification,
        trade_fee=trade_fee,
        peg=peg,
        peg_target=peg_target,
        max_peg_update=max_peg_update
    )
    initial_state.time_step = blocks_since_update
    tkn_sell = 'R1'
    tkn_buy = 'R2'

    # first we test sell_spot
    sell_state = initial_state.copy()
    sell_quantity = mpf(0.001)
    sell_agent = Agent(holdings={tkn_sell: sell_quantity, tkn_buy: 0})
    r2_per_r1 = sell_state.sell_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy)
    sell_state.swap(
        agent=sell_agent,
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        sell_quantity=sell_quantity
    )
    actual_sell_quantity = sell_agent.initial_holdings[tkn_sell] - sell_agent.holdings[tkn_sell]
    actual_buy_quantity = sell_agent.holdings[tkn_buy] - sell_agent.initial_holdings[tkn_buy]
    ex_price_r1 = actual_buy_quantity / actual_sell_quantity
    if r2_per_r1 < ex_price_r1:
        raise AssertionError(f'Sell spot R1 ({r2_per_r1}) < execution price ({ex_price_r1}), implying negative slippage')
    if r2_per_r1 != pytest.approx(ex_price_r1):
        raise AssertionError(f'Sell spot R1 ({r2_per_r1}) != execution price ({ex_price_r1}).')

    # then we test sell_spot
    buy_state = initial_state.copy()
    buy_quantity = mpf(0.001)
    r1_per_r2 = initial_state.buy_spot(tkn_buy=tkn_buy, tkn_sell=tkn_sell)
    buy_agent = Agent(holdings={tkn_sell: buy_quantity * r1_per_r2 * 2, tkn_buy: 0})
    buy_state.swap(
        agent=buy_agent,
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        buy_quantity=buy_quantity
    )
    actual_sell_quantity = buy_agent.initial_holdings[tkn_sell] - buy_agent.holdings[tkn_sell]
    actual_buy_quantity = buy_agent.holdings[tkn_buy] - buy_agent.initial_holdings[tkn_buy]
    ex_price_r2 = actual_sell_quantity / actual_buy_quantity

    if r1_per_r2 > ex_price_r2:
        raise AssertionError(f'Buy spot R2 ({r1_per_r2}) > execution price ({ex_price_r2}), implying negative slippage')
    if r1_per_r2 != pytest.approx(ex_price_r2):
        raise AssertionError(f'Buy spot R2 ({r1_per_r2}) != execution price ({ex_price_r2}).')


@settings(deadline=timedelta(milliseconds=500))
@given(
    st.lists(asset_quantity_strategy, min_size=2, max_size=2),
    st.floats(min_value=0.0001, max_value=0.50),
    st.integers(min_value=10, max_value=100000)
)
def test_share_prices(assets, fee, amp):
    tokens = {'USDT': mpf(assets[0]), 'USDC': mpf(assets[1])}
    pool = StableSwapPoolState(tokens, mpf(amp), trade_fee=mpf(fee))
    spot = pool.share_price('USDT')
    add_liq_price = pool.add_liquidity_spot('USDT')
    if add_liq_price <= spot:
        raise AssertionError('Add liquidity price should be higher than spot price.')
    buy_shares_price = pool.buy_shares_spot('USDT')
    if buy_shares_price <= spot:
        raise AssertionError('Buy shares price should be higher than spot price.')
    remove_liq_price = pool.remove_liquidity_spot('USDT')
    if remove_liq_price >= spot:
        raise AssertionError('Remove liquidity price should be lower than spot price.')
    withdraw_asset_price = pool.withdraw_asset_spot('USDT')
    if withdraw_asset_price >= spot:
        raise AssertionError('Withdraw asset price should be lower than spot price.')


def test_arbitrary_peg_feeless():
    # we'll test that slippage is lowest around the peg.
    amp = 1000
    fee = 0.0
    tvl = 2000000
    trade_size = 100

    # first, with peg = 1
    slippage = {}
    for r in [1, 2, 0.5]:
        tokens = {'USDT': mpf(r / (r + 1) * tvl), 'USDC': mpf(1 / (r + 1) * tvl)}
        pool = StableSwapPoolState(tokens, mpf(amp), trade_fee=mpf(fee))
        spot = pool.sell_spot('USDT', 'USDC')

        agent = Agent(holdings={'USDT': mpf(trade_size)})
        test_state, test_agent = simulate_swap(pool, agent, 'USDT', 'USDC', sell_quantity=trade_size)
        execution_price = test_agent.holdings['USDC'] / trade_size
        slippage[r] = abs(spot - execution_price)/spot
    assert slippage[1] < slippage[2] and slippage[1] < slippage[0.5]

    # then, with peg = 2
    slippage = {}
    for r in [1, 2, 3]:
        tokens = {'USDT': mpf(r / (r + 1) * tvl), 'USDC': mpf(1 / (r + 1) * tvl)}
        pool = StableSwapPoolState(tokens, mpf(amp), trade_fee=mpf(fee), peg=2)
        spot = pool.sell_spot('USDT', 'USDC')

        agent = Agent(holdings={'USDT': mpf(trade_size)})
        test_state, test_agent = simulate_swap(pool, agent, 'USDT', 'USDC', sell_quantity=trade_size)
        execution_price = test_agent.holdings['USDC'] / trade_size
        slippage[r] = abs(spot - execution_price)/spot
    assert max(slippage.values()) < 1e-5
    assert slippage[2] < slippage[1] and slippage[2] < slippage[3]

    # finally, with peg = 0.5
    slippage = {}
    for r in [0.25, 0.5, 1]:
        tokens = {'USDT': mpf(r / (r + 1) * tvl), 'USDC': mpf(1 / (r + 1) * tvl)}
        pool = StableSwapPoolState(tokens, mpf(amp), trade_fee=mpf(fee), peg=0.5)
        spot = pool.sell_spot('USDT', 'USDC')

        agent = Agent(holdings={'USDT': mpf(trade_size)})
        test_state, test_agent = simulate_swap(pool, agent, 'USDT', 'USDC', sell_quantity=trade_size)
        execution_price = test_agent.holdings['USDC'] / trade_size
        slippage[r] = abs(spot - execution_price)/spot
    assert max(slippage.values()) < 1e-5
    assert slippage[0.5] < slippage[0.25] and slippage[0.5] < slippage[1]


@given(
    st.floats(min_value=0.0001, max_value=1000),
    st.floats(min_value=0.0001, max_value=1000),
    st.floats(min_value=0.01, max_value=100),
    st.floats(min_value=0.01, max_value=100),
)
def test_fuzz_arbitrary_peg_remove_uniform(peg1, peg2, r1, r2):
    # we'll test that the asset/share ratio does not decrease
    amp = 1000
    fee = 0.0
    tvl = 2000000
    remove_pct_size = 0.0001

    tokens = {'USDT': mpf(r1 / (r1 + r2 + 1) * tvl), 'USDC': mpf(1 / (r1 + r2 + 1) * tvl),
              'DAI': mpf(r2 / (r1 + r2 + 1) * tvl)}
    pool = StableSwapPoolState(tokens, mpf(amp), trade_fee=mpf(fee), peg=[peg1, peg2])
    usdc_ratio = pool.liquidity['USDC'] / pool.shares
    usdt_ratio = pool.liquidity['USDT'] / pool.shares

    agent = Agent(holdings={pool.unique_id: mpf(pool.shares * remove_pct_size)})
    test_state, test_agent = simulate_remove_uniform(pool, agent, agent.holdings[pool.unique_id])
    new_usdc_ratio = test_state.liquidity['USDC'] / test_state.shares
    new_usdt_ratio = test_state.liquidity['USDT'] / test_state.shares
    err_usdc = (new_usdc_ratio - usdc_ratio)/usdc_ratio
    err_usdt = (new_usdt_ratio - usdt_ratio)/usdt_ratio
    if err_usdc < -1e-20:
        raise  # exploitable
    elif err_usdc > 1e-20:
        raise  # insufficiently accurate
    if err_usdt < -1e-20:
        raise  # exploitable
    elif err_usdt > 1e-20:
        raise  # insufficiently accurate


@given(
    st.booleans(),
    st.booleans(),
    st.floats(min_value=0.000001, max_value=100),
    st.floats(min_value=0.000001, max_value=100),
    st.integers(min_value=10, max_value=100000),
    st.floats(min_value=0.000001, max_value = 1000000),
    st.floats(min_value=0.000001, max_value = 1000000),
    st.floats(min_value=0.000001, max_value = 1000000),
    st.floats(min_value=0.000001, max_value = 1000000)
)
def test_fuzz_exploit_loop(add_tkn_usdt, remove_tkn_usdt, trade_pct_size, add_pct_size, amp, peg1, peg2, r1, r2):
    fee = 0.0
    tvl = 2000000
    add_tkn = 'USDT' if add_tkn_usdt else 'USDC'
    remove_tkn = 'USDT' if remove_tkn_usdt else 'USDC'
    sell_tkn = 'USDT'
    buy_tkn = 'USDC'

    tokens = {'USDT': mpf(r1 / (r1 + r2 + 1) * tvl), 'USDC': mpf(1 / (r1 + r2 + 1) * tvl),
              'DAI': mpf(r2 / (r1 + r2 + 1) * tvl)}
    pool = StableSwapPoolState(tokens, mpf(amp), trade_fee=mpf(fee), peg=[peg1, peg2])
    add_amt = pool.liquidity[add_tkn] * add_pct_size
    sell_amt = pool.liquidity[sell_tkn] * trade_pct_size

    init_holdings = {add_tkn: add_amt}
    if add_tkn == sell_tkn:
        init_holdings[add_tkn] += sell_amt
        init_holdings[buy_tkn] = 0
    else:
        init_holdings[sell_tkn] = sell_amt
    agent = Agent(holdings={tkn: init_holdings[tkn] for tkn in init_holdings})

    # add liquidity
    pool.add_liquidity(agent, add_amt, add_tkn)
    # trade
    pool.swap(agent, sell_tkn, buy_tkn, sell_quantity=agent.holdings[sell_tkn])
    # remove liquidity
    pool.remove_liquidity(agent, agent.holdings[pool.unique_id], remove_tkn)
    # trade back
    pool.swap(agent, buy_tkn, sell_tkn, sell_quantity=agent.holdings[buy_tkn] - init_holdings[buy_tkn])
    if agent.holdings[pool.unique_id] != 0:
        raise AssertionError('Agent should have no shares left')
    if agent.holdings[buy_tkn] != init_holdings[buy_tkn]:
        raise AssertionError('By design of test, agent should have starting quantity of buy_tkn')
    if agent.holdings[sell_tkn] > init_holdings[sell_tkn]:
        raise AssertionError('Agent has successfully exploited the pool')
    profit_pct = (init_holdings[sell_tkn] - agent.holdings[sell_tkn])/tokens[sell_tkn]
    if profit_pct >= 1e9:
        raise AssertionError('Agent lost too much money')


def test_stableswap_fee():
    amp = 1000
    fee = 0.01  # very high fee of 1% to exaggerate impact
    tvl = 2000000
    trade_size = 100

    for peg1 in [1, 0.5, 2]:
        for peg2 in [1, 0.5, 2]:
            r1, r2 = peg1, peg2  # this makes pool evenly balanced at peg
            tokens = {'TKN1': mpf(r1 / (r1 + r2 + 1) * tvl), 'TKN2': mpf(1 / (r1 + r2 + 1) * tvl),
                      'TKN3': mpf(r2 / (r1 + r2 + 1) * tvl)}
            init_asset_sum_adj = tokens['TKN1'] + peg1 * tokens['TKN2']
            pool = StableSwapPoolState(tokens, mpf(amp), trade_fee=mpf(fee), peg=[peg1, peg2])
            agent = Agent(holdings={'TKN1': mpf(trade_size)})
            test_state, test_agent = simulate_swap(pool, agent, 'TKN1', 'TKN2', sell_quantity=trade_size)
            asset_sum_adj = test_state.liquidity['TKN1'] + peg1 * test_state.liquidity['TKN2']
            correct_fee = fee * trade_size
            # we're pretty permissive with the error bar here because we are taking the difference of asset sums
            if asset_sum_adj - init_asset_sum_adj != pytest.approx(correct_fee, rel=1e-3):
                raise AssertionError('Fee not correctly applied')


@given(st.floats(min_value=0.0, max_value=0.01))
def test_fuzz_stableswap_fee_invariant(fee):
    # we'll test that slippage is lowest around the peg.
    amp = 1000
    tvl = 2000000
    trade_size = 100

    for r in [1, 2, 0.5]:
        for peg_m in [0.5, 1, 2]:
            peg = r * peg_m
            tokens = {'USDT': mpf(r / (r + 1) * tvl), 'USDC': mpf(1 / (r + 1) * tvl)}
            pool = StableSwapPoolState(tokens, mpf(amp), trade_fee=mpf(fee), peg=peg)
            init_d = pool.d

            agent = Agent(holdings={'USDT': mpf(trade_size)})
            test_state, test_agent = simulate_swap(pool, agent, 'USDT', 'USDC', sell_quantity=trade_size)
            correct_fee = fee * trade_size
            # these quantities should *not* be exactly equal when the pool is not balanced, so we use high error
            rel_error = 1e-8 if peg_m == 1 else 1e-2
            assert test_state.d - init_d == pytest.approx(correct_fee, rel=rel_error)


@given(
    st.floats(min_value=0.0001, max_value=0.01),
    st.floats(min_value=0.0001, max_value=10000)
)
@settings(print_blob=True)
def test_stableswap_withdraw_fee_arbitrary_peg(fee, peg):
    # we'll compare adding USDT and withdrawing USDC to just swapping USDT for USDC
    amp = 1000
    tvl = 2000000
    trade_size = 100

    r = peg
    tokens = {'USDT': mpf(r / (r + 1) * tvl), 'USDC': mpf(1 / (r + 1) * tvl)}
    pool = StableSwapPoolState(tokens, mpf(amp), trade_fee=mpf(fee), peg=peg)

    agent = Agent(holdings={'USDT': mpf(trade_size)})
    liq_state, liq_agent = simulate_add_liquidity(pool, agent, agent.holdings['USDT'], 'USDT')
    liq_state.remove_liquidity(liq_agent, liq_agent.holdings[liq_state.unique_id], 'USDC')

    swap_state, swap_agent = simulate_swap(pool, agent, 'USDT', 'USDC', sell_quantity=trade_size)
    pct_diff = (swap_agent.holdings['USDC'] - liq_agent.holdings['USDC'])/swap_agent.holdings['USDC']
    assert pct_diff > -1e-4  # withdraw liquidity is sometimes slightly better than swapping
    assert pct_diff < 1e-4


@given(
    st.floats(min_value=0.00001, max_value=0.0010),
    st.floats(min_value=0.0001, max_value=10000),
    st.floats(min_value=10, max_value=100000),
    st.floats(min_value=-1, max_value=1, exclude_min=True),
    st.floats(min_value=0, max_value=0.01)
)
@settings(print_blob=True)
def test_fuzz_arb_repegging(fee, balance_pct, amp, repeg_pct, max_repeg):
    init_vDOT_price = 1

    balanced_tokens = {'DOT': init_vDOT_price * 1000000, 'vDOT': 1000000}
    tokens = {'DOT': balance_pct / (balance_pct + 1) * balanced_tokens['DOT'],
              'vDOT': 1 / (balance_pct + 1) * balanced_tokens['vDOT']}

    arb_size = 1
    agent = Agent(holdings={'DOT': arb_size})

    peg_target = init_vDOT_price * (1 + repeg_pct)
    pool = StableSwapPoolState(tokens, amp, trade_fee=fee, peg=init_vDOT_price, max_peg_update=max_repeg)
    pool.swap(agent, 'DOT', 'vDOT', sell_quantity=arb_size)
    pool.set_peg_target(peg_target)
    pool.swap(agent, 'vDOT', 'DOT', sell_quantity=agent.holdings['vDOT'])
    profit = agent.holdings['DOT'] - arb_size
    if profit > 0:
        raise AssertionError(f'Attack successful')


@given(
    st.floats(min_value=0.00001, max_value=0.0010),
    st.floats(min_value=0.0001, max_value=10000),
    st.floats(min_value=10, max_value=100000),
    st.floats(min_value=-1, max_value=1, exclude_min=True),
    st.floats(min_value=0, max_value=0.01)
)
@settings(print_blob=True)
def test_fuzz_arb_repegging_lp(fee, balance_pct, amp, repeg_pct, max_repeg):
    init_vDOT_price = 1
    for liq_tkn in ['DOT', 'vDOT']:
        balanced_tokens = {'DOT': init_vDOT_price * 1000000, 'vDOT': 1000000}
        tokens = {'DOT': balance_pct / (balance_pct + 1) * balanced_tokens['DOT'],
                  'vDOT': 1 / (balance_pct + 1) * balanced_tokens['vDOT']}

        liq_size = 1000000
        agent = Agent(holdings={liq_tkn: liq_size})

        peg_target = init_vDOT_price * (1 + repeg_pct)
        pool = StableSwapPoolState(tokens, amp, trade_fee=fee, peg=init_vDOT_price, max_peg_update=max_repeg)

        pool.add_liquidity(agent, liq_size, liq_tkn)
        pool.set_peg_target(peg_target)
        pool.remove_liquidity(agent, agent.holdings[pool.unique_id], liq_tkn)
        profit = agent.holdings[liq_tkn] - liq_size
        if profit > 0:
            raise AssertionError(f'Attack successful')


@given(
    st.floats(min_value=0.01, max_value=100),
    st.floats(min_value=0.01, max_value=100),
    st.floats(min_value=10, max_value=10000),
    st.floats(min_value=-0.5, max_value=0.5, exclude_min=True),
    st.floats(min_value=-0.5, max_value=0.5, exclude_min=True)
)
def test_fuzz_arb_repegging_3pool(
        ratio1,
        ratio2,
        amp,
        repeg_pct1,
        repeg_pct2
):
    init_vDOT_price = 1
    init_lstDOT_price = 1
    arb_size = 1
    fee = 0.0002
    max_repeg = 0.001

    dot_liq = mpf(1000000)
    tokens = {
        'DOT': dot_liq,
        'vDOT': ratio1 * dot_liq / init_vDOT_price,
        'lstDOT': ratio2 * dot_liq / init_lstDOT_price
    }

    peg_target = [init_vDOT_price * (1 + repeg_pct1), init_lstDOT_price * (1 + repeg_pct2)]

    for [tkn1, tkn2] in [['DOT', 'vDOT'], ['DOT', 'lstDOT'], ['vDOT', 'lstDOT']]:
        for [tkn_buy, tkn_sell] in [[tkn1, tkn2], [tkn2, tkn1]]:
            agent = Agent(holdings={tkn_sell: arb_size})

            pool = StableSwapPoolState(copy.deepcopy(tokens), amp, trade_fee=fee,
                                       peg=[init_vDOT_price, init_lstDOT_price], max_peg_update=max_repeg,
                                       precision=0.0000000001)
            pool.swap(agent, tkn_sell, tkn_buy, sell_quantity=arb_size)
            pool.set_peg_target(peg_target)
            pool.swap(agent, tkn_buy, tkn_sell, sell_quantity=agent.holdings[tkn_buy])
            profit = agent.holdings[tkn_sell] - arb_size
            if profit > 0:
                raise AssertionError(f'Attack successful')


@given(
    st.floats(min_value=0.00001, max_value=0.0010),
    st.floats(min_value=.01, max_value=100),
    st.floats(min_value=.01, max_value=100),
    st.floats(min_value=10, max_value=100000),
    st.floats(min_value=-1, max_value=1, exclude_min=True),
    st.floats(min_value=-1, max_value=1, exclude_min=True),
    st.floats(min_value=0, max_value=0.01)
)
def test_fuzz_arb_repegging_lp_3pool(fee, ratio1, ratio2, amp, repeg_pct1, repeg_pct2, max_repeg):
    init_vDOT_price = 1
    init_lstDOT_price = 1

    dot_liq = 1000000
    tokens = {
        'DOT': dot_liq,
        'vDOT': ratio1 * dot_liq / init_vDOT_price,
        'lstDOT': ratio2 * dot_liq / init_lstDOT_price
    }

    peg_target = [init_vDOT_price * (1 + repeg_pct1), init_lstDOT_price * (1 + repeg_pct2)]

    for liq_tkn in ['DOT', 'vDOT']:
        liq_size = tokens[liq_tkn] / 2
        agent = Agent(holdings={liq_tkn: liq_size})

        pool = StableSwapPoolState(tokens, amp, trade_fee=fee, peg=[init_vDOT_price, init_lstDOT_price], max_peg_update=max_repeg)

        pool.add_liquidity(agent, liq_size, liq_tkn)
        pool.set_peg_target(peg_target)
        pool.remove_liquidity(agent, agent.holdings[pool.unique_id], liq_tkn)
        profit = agent.holdings[liq_tkn] - liq_size
        if profit > 0:
            raise AssertionError(f'Attack successful')


@given(
    st.floats(min_value=0.00001, max_value=0.0010),
    st.floats(min_value=.01, max_value=100),
    st.floats(min_value=.01, max_value=100),
    st.floats(min_value=10, max_value=100000),
    st.floats(min_value=-1, max_value=1, exclude_min=True),
    st.floats(min_value=-1, max_value=1, exclude_min=True),
    st.floats(min_value=0, max_value=0.01)
)
def test_fuzz_arb_repegging_lp_uniform_3pool(fee, ratio1, ratio2, amp, repeg_pct1, repeg_pct2, max_repeg):
    init_vDOT_price = 1
    init_lstDOT_price = 1

    dot_liq = 1000000
    tokens = {
        'DOT': dot_liq,
        'vDOT': ratio1 * dot_liq / init_vDOT_price,
        'lstDOT': ratio2 * dot_liq / init_lstDOT_price
    }

    peg_target = [init_vDOT_price * (1 + repeg_pct1), init_lstDOT_price * (1 + repeg_pct2)]

    init_holdings = {liq_tkn: tokens[liq_tkn] / 2 for liq_tkn in tokens}
    agent = Agent(holdings=init_holdings)

    pool = StableSwapPoolState(tokens, amp, trade_fee=fee, peg=[init_vDOT_price, init_lstDOT_price], max_peg_update=max_repeg)

    for liq_tkn in tokens:
        pool.add_liquidity(agent, init_holdings[liq_tkn], liq_tkn)
    pool.set_peg_target(peg_target)
    pool.remove_uniform(agent, agent.holdings[pool.unique_id])
    profit_dict = {tkn: agent.holdings[tkn] - init_holdings[tkn] for tkn in init_holdings}
    profit = sum([pool.peg_target[i] * profit_dict[pool.asset_list[i]] for i in range(pool.n_coins)])
    if profit > 0:
        raise AssertionError(f'Attack successful')


def test_stableswap_constructor_peg_success():
    # n = 2
    tokens = {'DOT': 1400000, 'vDOT': 1000000}
    a = 100
    trade_fee = 0.0005
    peg = 1.4

    pool = StableSwapPoolState(tokens=tokens, amplification=a, trade_fee=trade_fee, peg=peg)
    assert pool.peg == [1, 1.4]
    assert pool.peg_target == [1, 1.4]

    # different peg target
    peg_target = 1.5
    pool = StableSwapPoolState(tokens=tokens, amplification=a, trade_fee=trade_fee, peg=peg, peg_target=peg_target)
    assert pool.peg == [1, 1.4]
    assert pool.peg_target == [1, 1.5]

    # n = 3
    tokens = {'DOT': 1400000, 'vDOT': 1000000, 'lstDOT': 1100000}
    peg = [1.4, 1.3]
    pool = StableSwapPoolState(tokens=tokens, amplification=a, trade_fee=trade_fee, peg=peg)
    assert pool.peg == [1, 1.4, 1.3]
    assert pool.peg_target == [1, 1.4, 1.3]

    # different peg target
    peg_target = [1.5, 1.4]
    pool = StableSwapPoolState(tokens=tokens, amplification=a, trade_fee=trade_fee, peg=peg, peg_target=peg_target)
    assert pool.peg == [1, 1.4, 1.3]
    assert pool.peg_target == [1, 1.5, 1.4]


def test_stableswap_constructor_peg_failure():
    tokens = {'DOT': 1400000, 'vDOT': 1000000, 'lstDOT': 1100000}
    a = 100
    trade_fee = 0.0005
    peg = 1.4

    with pytest.raises(Exception):
        StableSwapPoolState(tokens=tokens, amplification=a, trade_fee=trade_fee, peg=peg)

    # different peg target
    peg = [1.4, 1.3]
    peg_target = 1.5
    with pytest.raises(Exception):
        StableSwapPoolState(tokens=tokens, amplification=a, trade_fee=trade_fee, peg=peg, peg_target=peg_target)


@given(
    st.floats(min_value=0.00001, max_value=0.0010),
    st.floats(min_value=0.01, max_value=100),
    st.floats(min_value=0.01, max_value=100),
    st.floats(min_value=10, max_value=100000),
    st.floats(min_value=-1, max_value=1, exclude_min=True),
    st.floats(min_value=-1, max_value=1, exclude_min=True),
    st.floats(min_value=0.000001, max_value=0.01),
    st.integers(min_value=1, max_value=1000),
    st.floats(min_value=1, max_value=100000)
)
def test_peg_update(fee, ratio1, ratio2, amp, repeg_pct1, repeg_pct2, max_repeg, block_ct, sell_size):
    init_vDOT_price = 1
    init_lstDOT_price = 1

    dot_liq = 1000000
    tokens = {
        'DOT': dot_liq,
        'vDOT': ratio1 * dot_liq / init_vDOT_price,
        'lstDOT': ratio2 * dot_liq / init_lstDOT_price
    }

    peg_target = [init_vDOT_price * (1 + repeg_pct1), init_lstDOT_price * (1 + repeg_pct2)]

    for [tkn1, tkn2] in [['DOT', 'vDOT'], ['DOT', 'lstDOT'], ['vDOT', 'lstDOT']]:
        for [tkn_buy, tkn_sell] in [[tkn1, tkn2], [tkn2, tkn1]]:
            agent = Agent(holdings={tkn_sell: sell_size})

            pool = StableSwapPoolState(copy.deepcopy(tokens), amp, trade_fee=fee,
                                       peg=[init_vDOT_price, init_lstDOT_price], max_peg_update=max_repeg)
            pool.set_peg_target(peg_target)
            pool.time_step += block_ct  # fast forward some blocks
            pool.swap(agent, tkn_sell, tkn_buy, sell_quantity=sell_size)
            peg_diff = [pool.peg[i] - 1 for i in range(len(pool.peg))]
            for i in range(1,len(tokens)):
                if peg_target[i-1] != pool.peg_target[i]:
                    raise AssertionError(f'Peg target update unsuccessful')
                # peg change should be in correct direction
                is_expected_peg_move_pos = peg_target[i-1] > 1
                if is_expected_peg_move_pos and peg_diff[i] < 0:
                    raise AssertionError(f'Peg of asset {pool.asset_list[i]} not updated in correct direction')
                elif not is_expected_peg_move_pos and peg_diff[i] > 0:
                    raise AssertionError(f'Peg of asset {pool.asset_list[i]} not updated in correct direction')
                if abs(peg_diff[i])/block_ct - max_repeg > 1e-15:  # check that max per-block is respected
                    raise AssertionError(f'Peg diff of asset {pool.asset_list[i]} exceeds max_repeg {max_repeg}')
                dir_sign = 1 if is_expected_peg_move_pos else -1
                max_total_repeg = dir_sign * block_ct * max_repeg
                # if peg target not hit, 1 + max_total_repeg = peg < peg_target
                if pool.peg[i] != peg_target[i-1]:
                    if 1 + max_total_repeg != pool.peg[i]:
                        raise AssertionError(f'Peg of asset {pool.asset_list[i]} not updated sufficiently')


def test_cash_out():
    prices = {'USDT': 1, 'UDSC': 1.003, 'USDX': 0.7}
    stableswap = StableSwapPoolState(
        tokens={'USDT': 1000000, 'USDC': 1000300, 'USDX': 700000},
        amplification=1000,
        trade_fee=0.0005,
        peg=list(prices.values())[1:],
    )
    agent = Agent(holdings={'USDT': 1000})
    stableswap.add_liquidity(agent, 1000, 'USDT')
    value = stableswap.cash_out(agent, prices)
    stableswap.remove_uniform(agent, agent.holdings[stableswap.unique_id])
    if value != sum([agent.holdings[tkn] * prices[tkn] if tkn in prices else 0 for tkn in agent.holdings]):
        raise AssertionError('Cash out value not calculated correctly')


@given(
    st.floats(min_value=0.01, max_value=100),
    st.floats(min_value=10, max_value=10_000)
)
def test_balance_ratio_at_price(price, amp):
    r = balance_ratio_at_price(amp, price, mpf(1_000_000))
    tvl = 1_000_000
    tokens = {"HOLLAR": r * tvl, "USDT": (1 - r) * tvl}
    pool = StableSwapPoolState(tokens, amp)
    spot = pool.price('HOLLAR', 'USDT')
    assert spot == pytest.approx(price, rel=1e-15)


def test_calculate_buy_from_sell():
    stableswap = StableSwapPoolState(
        tokens={'a': 1000, 'b': 2000},
        amplification=100,
        peg=0.5
    )
    for i in range(1000):
        agent = Agent()
        stableswap.set_peg_target(0.5 + i * 0.0001)
        stableswap.update()
        buy_quantity = stableswap.calculate_buy_from_sell(tkn_buy='a', tkn_sell='b', sell_quantity=1)
        stableswap.swap(
            agent=agent,
            tkn_buy='a',
            tkn_sell='b',
            sell_quantity=1
        )
        real_buy_quantity = agent.holdings['a']
        if buy_quantity != pytest.approx(real_buy_quantity, rel=1e-12):
            print('wrong price')
    pass
