import copy
import math
from datetime import timedelta

import pytest
from hypothesis import given, strategies as st, reproduce_failure, settings

# from hydradx.model import run
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import GlobalState, historical_prices
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.trade_strategies import omnipool_arbitrage, back_and_forth, invest_all, dca_with_lping
from hydradx.model.amm.trade_strategies import schedule_swaps
from hydradx.tests.strategies_omnipool import omnipool_reasonable_config, reasonable_market
from hydradx.model.run import run
from hydradx.tests.utils import randomize_object
from mpmath import mp, mpf

settings.register_profile("long", deadline=timedelta(milliseconds=500), print_blob=True)
settings.load_profile("long")

mp.dps = 50

asset_price_strategy = st.floats(min_value=0.0001, max_value=100000)
asset_price_bounded_strategy = st.floats(min_value=0.1, max_value=10)
asset_number_strategy = st.integers(min_value=3, max_value=5)
arb_precision_strategy = st.integers(min_value=1, max_value=5)
asset_quantity_strategy = st.floats(min_value=100, max_value=10000000)
asset_quantity_bounded_strategy = st.floats(min_value=1000000, max_value=10000000)
percentage_of_liquidity_strategy = st.floats(min_value=0.0000001, max_value=0.10)
fee_strategy = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)


@given(omnipool_reasonable_config(asset_fee=0.0, lrna_fee=0.0, token_count=3), percentage_of_liquidity_strategy)
def test_back_and_forth_trader_feeless(omnipool: OmnipoolState, pct: float):
    holdings = {'LRNA': 1000000000}
    for asset in omnipool.asset_list:
        holdings[asset] = 1000000000
    agent = Agent(holdings=holdings, trade_strategy=back_and_forth)
    state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': agent})
    strat = back_and_forth('omnipool', pct)

    old_agent = copy.deepcopy(agent)

    strat.execute(state, 'agent')
    new_agent = state.agents['agent']
    assert new_agent.holdings['LRNA'] == old_agent.holdings['LRNA']  # LRNA holdings should be *exact*
    for asset in omnipool.asset_list:
        assert new_agent.holdings[asset] == pytest.approx(old_agent.holdings[asset], rel=1e-15)


@given(omnipool_reasonable_config(token_count=3), percentage_of_liquidity_strategy)
def test_back_and_forth_trader(omnipool: OmnipoolState, pct: float):
    holdings = {'LRNA': 1000000000}
    for asset in omnipool.asset_list:
        holdings[asset] = 1000000000
    agent = Agent(holdings=holdings, trade_strategy=back_and_forth)
    state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': agent})
    strat = back_and_forth('omnipool', pct)

    old_agent = copy.deepcopy(agent)

    strat.execute(state, 'agent')
    new_agent = state.agents['agent']
    if new_agent.holdings['LRNA'] != old_agent.holdings['LRNA']:
        raise
    for asset in omnipool.asset_list:
        if new_agent.holdings[asset] > old_agent.holdings[asset]:
            if new_agent.holdings[asset] != pytest.approx(old_agent.holdings[asset], rel=1e-15):
                raise


@given(omnipool_reasonable_config(asset_fee=0.0, lrna_fee=0.0, token_count=3), reasonable_market(token_count=3),
       arb_precision_strategy)
def test_omnipool_arbitrager_feeless(omnipool: OmnipoolState, market: list, arb_precision: int):
    agent = Agent(enforce_holdings=False)
    assets = list(omnipool.liquidity)
    external_market = {assets[i]: market[i] for i in range(len(assets))}
    external_market[omnipool.stablecoin] = 1.0
    state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': agent}, external_market=external_market)
    strat = omnipool_arbitrage('omnipool', arb_precision)

    strat.execute(state, 'agent')
    assert not omnipool.fail
    new_holdings = {tkn: agent.get_holdings(tkn) for tkn in omnipool.liquidity}

    old_value, new_value = 0, 0

    # Trading should result in net zero LRNA trades
    if agent.get_holdings("LRNA") / omnipool.lrna_total != pytest.approx(0, abs=1e-15):
        raise

    for asset in omnipool.asset_list:
        new_value += new_holdings[asset] * external_market[asset]

        # Trading should bring pool to market price
        if omnipool.usd_price(asset) != pytest.approx(external_market[asset], rel=1e-15):
            raise

    # Trading should be profitable
    if 0 > new_value:
        if new_value != pytest.approx(0, abs=1e-12):
            raise


@given(
    hdx_value=st.floats(min_value=0.5, max_value=2),
    dot_value=st.floats(min_value=0.5, max_value=2)
)
def test_omnipool_arbitrager(hdx_value, dot_value):
    omnipool = OmnipoolState(
        tokens={tkn: {
            'liquidity': mpf(1000),
            'LRNA': mpf(1000)
        } for tkn in ('HDX', 'USD', 'DOT')}
    )
    omnipool.trade_limit_per_block = float('inf')
    agent = Agent(enforce_holdings=False)
    external_market = {'USD': 1, 'HDX': hdx_value, 'DOT': dot_value}
    state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': agent}, external_market=external_market)
    strat = omnipool_arbitrage('omnipool')

    strat.execute(state, 'agent')
    new_holdings = {tkn: state.agents['agent'].get_holdings(tkn) for tkn in omnipool.asset_list + ['LRNA']}

    new_value = 0

    if not omnipool.fail:  # high fees can break arbitrager, which initially calculates values without fees
        # Trading should result in net zero LRNA trades
        if abs(new_holdings['LRNA']) >= 1e-10:
            raise AssertionError(f'Arbitrageur traded LRNA. new: {new_holdings["LRNA"]}')

        for asset in omnipool.liquidtity:
            new_value += new_holdings[asset] * external_market[asset]

            # Trading should bring pool to market price
            # if omnipool.usd_price(asset) != pytest.approx(external_market[asset], rel=1e-15):
            #     raise

        # Trading should be profitable
        if 0 > new_value:
            raise AssertionError(f'Arbitrageur lost money. new_value: {new_value}')


@given(frequency=st.integers(min_value=1, max_value=10))
def test_omnipool_arbitrager_periodic(frequency: int):
    omnipool = OmnipoolState(
        tokens={tkn: {
            'liquidity': 1000,
            'LRNA': 1000
        } for tkn in ('HDX', 'USD', 'DOT')}
    )
    omnipool.trade_limit_per_block = float('inf')
    agent = Agent(
        enforce_holdings=False,
        trade_strategy=omnipool_arbitrage('omnipool', frequency=frequency)
    )
    external_market = {'USD': 1, 'HDX': 0.02, 'DOT': 7}

    timesteps = 10
    price_list = [copy.deepcopy(external_market)]
    for i in range(timesteps):
        prices = {asset: price * .8 if asset != "USD" else 1 for asset, price in price_list[-1].items()}
        price_list.append(prices)

    initial_state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': agent}, external_market=external_market,
                        evolve_function=historical_prices(price_list))
    events = [initial_state] + run(initial_state, time_steps=timesteps, silent=True)

    for i, event in enumerate(events):
        if i == 0: continue
        if event.time_step % frequency == 0:  # there should be trade between steps i and i-1
            for asset in omnipool.liquidity:
                if event.pools['omnipool'].liquidity[asset] == events[i-1].pools['omnipool'].liquidity[asset]:
                    raise AssertionError(f'Arbitrageur did not trade at timestep {event.time_step}.')
        else:  # there should be no trade between steps i and i-1
            for asset in omnipool.liquidity:
                if event.pools['omnipool'].liquidity[asset] != events[i-1].pools['omnipool'].liquidity[asset]:
                    raise AssertionError(f'Arbitrageur traded at timestep {event.time_step}.')


@given(omnipool_reasonable_config(token_count=3))
def test_omnipool_LP(omnipool: OmnipoolState):
    holdings = {asset: 10000 for asset in omnipool.liquidity}
    initial_agent = Agent(holdings=holdings, trade_strategy=invest_all('omnipool', when=4))
    initial_state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': initial_agent})

    new_state = invest_all('omnipool').execute(initial_state.copy(), 'agent')
    for tkn in omnipool.liquidity:
        if new_state.agents['agent'].holdings[tkn] != 0:
            raise AssertionError(f'Failed to LP {tkn}')
        if new_state.agents['agent'].holdings[('omnipool', tkn)] == 0:
            raise AssertionError(f'Did not receive shares for {tkn}')

    hdx_state = invest_all('omnipool', 'HDX').execute(initial_state.copy(), 'agent')

    if hdx_state.agents['agent'].holdings['HDX'] != 0:
        raise AssertionError('HDX not reinvested.')
    if hdx_state.agents['agent'].holdings[('omnipool', 'HDX')] == 0:
        raise AssertionError('HDX shares not received.')

    for tkn in omnipool.liquidity:
        if tkn == 'HDX':
            continue
        if hdx_state.agents['agent'].holdings[tkn] == 0:
            raise AssertionError(f'{tkn} missing from holdings.')
        if ('omnipool', tkn) in hdx_state.agents['agent'].holdings \
                and hdx_state.agents['agent'].holdings[('omnipool', tkn)] != 0:
            raise AssertionError(f'Agent has shares of {tkn}, but should not.')

    events = run(initial_state, time_steps=4, silent=True)
    final_state = events[-1]
    if final_state.agents['agent'].holdings['HDX'] != 0 or events[-2].agents['agent'].holdings['HDX'] == 0:
        raise AssertionError('HDX not invested at the right time.')


def test_agent_copy():
    init_agent = randomize_object(
        Agent(holdings={'HDX': 100, 'USD': 100}, share_prices={'HDX': 1, 'USD': 1})
    )
    copy_agent = init_agent.copy()

    for member in copy_agent.__dict__:
        if (
                getattr(init_agent, member) != getattr(copy_agent, member)
        ):
            raise AssertionError(f'Copy failed for {member}.\n'
                                 f'original: {getattr(init_agent, member)}\n'
                                 f'copy: {getattr(copy_agent, member)}')


@given(
    omnipool_reasonable_config(asset_fee=0.0, lrna_fee=0.0, token_count=5),
    st.lists(st.floats(min_value=0.0001, max_value=0.5), min_size=2, max_size=2),
    st.lists(st.floats(min_value=0.9, max_value=1.1), min_size=2, max_size=2),
    st.floats(min_value=0.00001, max_value=1.1),
    st.lists(st.floats(min_value=10000, max_value=100000), min_size=2, max_size=2),
    st.lists(st.booleans(), min_size=2, max_size=2)
)
def test_dca_with_lping(
        omnipool: OmnipoolState,
        init_lp_pcts: list[float],
        price_mults: list[float],
        shares_mult: float,
        assets: list[float],
        has_assets: list[bool]
):
    buy_tkn = omnipool.asset_list[3]
    sell_tkn = omnipool.asset_list[4]
    trader_id = 'trader'

    for tkn in omnipool.liquidity:
        omnipool.liquidity[tkn] = mpf(omnipool.liquidity[tkn])
        omnipool.lrna[tkn] = mpf(omnipool.lrna[tkn])

    holdings = {
        ('omnipool', sell_tkn): mpf(init_lp_pcts[0] * omnipool.shares[sell_tkn]),
        ('omnipool', buy_tkn): mpf(init_lp_pcts[1] * omnipool.shares[buy_tkn]),
        sell_tkn: mpf(assets[0]) if has_assets[0] else 0,
        buy_tkn: mpf(assets[1]) if has_assets[1] else 0
    }
    max_shares_per_block = holdings[('omnipool', sell_tkn)] * shares_mult
    share_prices = {
        ('omnipool', sell_tkn): omnipool.lrna_price(sell_tkn) * price_mults[0],
        ('omnipool', buy_tkn): omnipool.lrna_price(buy_tkn) * price_mults[1]
    }

    agent = Agent(holdings=holdings, unique_id=trader_id, share_prices=share_prices)

    state = GlobalState(pools={'omnipool': omnipool}, agents={trader_id: agent})

    strategy = dca_with_lping('omnipool', sell_tkn, buy_tkn, max_shares_per_block)

    init_buy_tkn_lped = agent.holdings[('omnipool', buy_tkn)] if ('omnipool', buy_tkn) in agent.holdings else 0
    init_sell_tkn_lped = agent.holdings[('omnipool', sell_tkn)] if ('omnipool', sell_tkn) in agent.holdings else 0
    init_buy_tkn = agent.holdings[buy_tkn]
    init_sell_tkn = agent.holdings[sell_tkn]

    strategy.execute(state, trader_id)

    if init_sell_tkn_lped > 0:
        if len(agent.nfts) == 0:
            raise AssertionError('Agent does not have shares for buy_tkn.')

        lp_diff = init_sell_tkn_lped
        if ('omnipool', sell_tkn) in agent.holdings:
            lp_diff -= agent.holdings[('omnipool', sell_tkn)]

        if lp_diff != pytest.approx(min(max_shares_per_block, init_sell_tkn_lped), rel=1e-20):
            raise AssertionError('Agent traded incorrect amount of shares.')
    elif (agent.holdings[('omnipool', sell_tkn)] != init_sell_tkn_lped or
            agent.holdings[('omnipool', buy_tkn)] != init_buy_tkn_lped):
        raise AssertionError('Agent LP shares changed.')
    if agent.holdings[sell_tkn] != pytest.approx(init_sell_tkn, rel=1e-20):
        raise AssertionError('Agent sell token changed.')
    if agent.holdings[buy_tkn] != pytest.approx(init_buy_tkn, rel=1e-20):
        raise AssertionError('Agent buy token changed.')

def test_schedule_swaps():
    """
    Test the schedule_swaps function.
    """
    omnipool = OmnipoolState(
        tokens={tkn: {'liquidity': 1000, 'LRNA': 1000} for tkn in ('HDX', 'USD', 'DOT')}
    )
    omnipool.trade_limit_per_block = float('inf')
    agent = Agent(enforce_holdings=False)
    external_market = {'USD': 1, 'HDX': 0.02, 'DOT': 7}
    state = GlobalState(pools={'omnipool': omnipool}, agents={'agent': agent}, external_market=external_market)

    trades = [
        [{'tkn_sell': 'HDX', 'tkn_buy': 'USD', 'sell_quantity': 100}],
        [{'tkn_sell': 'DOT', 'tkn_buy': 'HDX', 'buy_quantity': 150}]
    ]
    strat = schedule_swaps('omnipool', trades)
    expected_usd = omnipool.calculate_buy_from_sell(tkn_buy='USD', tkn_sell='HDX', sell_quantity=100)
    strat.execute(state, 'agent')
    state.evolve()
    expected_dot = -omnipool.calculate_sell_from_buy(tkn_buy='HDX', tkn_sell='DOT', buy_quantity=150)
    strat.execute(state, 'agent')
    expected_hdx = 50

    assert not omnipool.fail
    assert agent.get_holdings('HDX') == pytest.approx(expected_hdx)
    assert agent.get_holdings('USD') == pytest.approx(expected_usd, rel=1e-15)
    assert agent.get_holdings('DOT') == pytest.approx(expected_dot, rel=1e-15)
