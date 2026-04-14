import copy
from datetime import timedelta
import os
import pytest
from hypothesis import given, strategies as st, settings, Phase

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.arbitrage_agent import calculate_profit, calculate_arb_amount, \
    process_next_swap, execute_arb, get_arb_swaps, combine_swaps
from hydradx.model.amm.centralized_market import OrderBook, CentralizedMarket
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.processing import get_omnipool_data_from_file, get_orderbooks_from_file
from hydradx.model.processing import load_state
from mpmath import mp, mpf

from hydradx.tests.utils import find_test_directory

mp.dps = 50


def test_calculate_profit():
    init_agent = Agent(holdings={'USDT': 1000000, 'DOT': 2000000, 'HDX': 3000000}, unique_id='bot')
    agent = init_agent.copy()
    profits = {'USDT': 100, 'DOT': 200, 'HDX': 0}
    agent.holdings['USDT'] += profits['USDT']
    agent.holdings['DOT'] += profits['DOT']
    agent.holdings['HDX'] += profits['HDX']
    calculated_profit = calculate_profit(init_agent, agent)
    for tkn in init_agent.holdings:
        assert calculated_profit[tkn] == profits[tkn]


def test_calculate_profit_with_mapping():
    init_agent = Agent(holdings={'USDT': 1000000, 'DOT': 2000000, 'HDX': 3000000, 'DAI': 4000000}, unique_id='bot')
    agent = init_agent.copy()
    profits = {'USDT': 100, 'DOT': 200, 'HDX': 0, 'DAI': 100}
    agent.holdings['USDT'] += profits['USDT']
    agent.holdings['DAI'] += profits['DAI']
    agent.holdings['DOT'] += profits['DOT']
    agent.holdings['HDX'] += profits['HDX']
    mapping = {'DAI': 'USD', 'USDT': 'USD'}
    calculated_profit = calculate_profit(init_agent, agent, mapping)

    assert calculated_profit['USD'] == 200
    assert calculated_profit['DOT'] == 200
    assert calculated_profit['HDX'] == 0


# @settings(max_examples=1)
@settings(phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target], deadline=timedelta(milliseconds=500))
@given(
    usdt_amt=st.floats(min_value=100000, max_value=1000000),
    dot_price=st.floats(min_value=0.01, max_value=1000),
    hdx_price=st.floats(min_value=0.01, max_value=1000),
    dot_wt=st.floats(min_value=0.05, max_value=0.50),
    hdx_wt=st.floats(min_value=0.01, max_value=0.20),
    price_mult=st.floats(min_value=1.1, max_value=5.0),
    lrna_fee=st.floats(min_value=0.0001, max_value=0.001),
    asset_fee=st.floats(min_value=0.0001, max_value=0.004),
    cex_fee=st.floats(min_value=0.0001, max_value=0.005),
)
def test_calculate_arb_amount_bid(
        usdt_amt: float,
        dot_price: float,
        hdx_price: float,
        dot_wt: float,
        hdx_wt: float,
        price_mult: float,
        lrna_fee: float,
        asset_fee: float,
        cex_fee: float
):
    usdt_wt = 1 - dot_wt - hdx_wt
    dot_lrna = dot_wt / usdt_wt * usdt_amt
    dot_amt = dot_lrna / dot_price
    hdx_lrna = hdx_wt / usdt_wt * usdt_amt
    hdx_amt = hdx_lrna / hdx_price

    tokens = {
        'USDT': {'liquidity': usdt_amt, 'LRNA': usdt_amt},
        'DOT': {'liquidity': dot_amt, 'LRNA': dot_lrna},
        'HDX': {'liquidity': hdx_amt, 'LRNA': hdx_lrna}
    }

    initial_dex = OmnipoolState(
        tokens=tokens,
        lrna_fee=lrna_fee,
        asset_fee=asset_fee,
        preferred_stablecoin='USDT',
    )
    orig_price = initial_dex.price('DOT', 'USDT')
    dex_price = orig_price / ((1 - lrna_fee) * (1 - asset_fee))
    bid_price = dex_price * (1 + cex_fee) * price_mult
    bid_quantity = 100000
    p = 1e-10

    tkn = 'DOT'
    numeraire = 'USDT'

    initial_cex = CentralizedMarket(
        order_book={
            ('DOT', 'USDT'): OrderBook(bids=[[bid_price, bid_quantity]], asks=[]),
        }, trade_fee=cex_fee
    )

    init_agent = Agent(holdings={'USDT': 1000000, 'DOT': 1000000, 'HDX': 1000000}, unique_id='bot')
    amt = calculate_arb_amount(
        buy_ex=initial_dex,
        sell_ex=initial_cex,
        buy_ex_tkn_pair=(tkn, numeraire),
        sell_ex_tkn_pair=(tkn, numeraire),
        sell_ex_max_sell=init_agent.holdings[tkn],
        buy_ex_max_sell=init_agent.holdings[numeraire],
        precision=p,
        max_iters=50
    )
    test_agent = Agent(holdings={'USDT': 1000000, 'DOT': 1000000})
    test_dex = initial_dex.copy()
    test_cex = initial_cex.copy()
    test_dex.swap(test_agent, tkn_buy=tkn, tkn_sell=numeraire, buy_quantity=amt)
    test_cex.swap(test_agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=amt)
    dex_price = test_dex.buy_spot(tkn, numeraire)
    cex_price = initial_cex.sell_spot(tkn, numeraire)

    if 1 - dex_price / cex_price > p and amt != bid_quantity:
        raise

    if amt == bid_quantity and dex_price > cex_price:
        raise

    profit = calculate_profit(init_agent, test_agent)
    for tkn in profit:
        assert profit[tkn] >= 0


# @settings(max_examples=1)
@settings(phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target], deadline=timedelta(milliseconds=500))
@given(
    usdt_amt=st.floats(min_value=100000, max_value=1000000),
    dot_price=st.floats(min_value=0.01, max_value=1000),
    hdx_price=st.floats(min_value=0.01, max_value=1000),
    dot_wt=st.floats(min_value=0.05, max_value=0.50),
    hdx_wt=st.floats(min_value=0.01, max_value=0.20),
    price_mult=st.floats(min_value=1.01, max_value=5.0),
    lrna_fee=st.floats(min_value=0.0001, max_value=0.001),
    asset_fee=st.floats(min_value=0.0001, max_value=0.004),
    cex_fee=st.floats(min_value=0.0001, max_value=0.005),
    max_trade=st.floats(min_value=1, max_value=100),
)
def test_calculate_arb_amount_bid_max_liquidity(
        usdt_amt: float,
        dot_price: float,
        hdx_price: float,
        dot_wt: float,
        hdx_wt: float,
        price_mult: float,
        lrna_fee: float,
        asset_fee: float,
        cex_fee: float,
        max_trade: float
):
    usdt_wt = 1 - dot_wt - hdx_wt
    dot_lrna = dot_wt / usdt_wt * usdt_amt
    dot_amt = dot_lrna / dot_price
    hdx_lrna = hdx_wt / usdt_wt * usdt_amt
    hdx_amt = hdx_lrna / hdx_price

    tokens = {
        'USDT': {'liquidity': usdt_amt, 'LRNA': usdt_amt},
        'DOT': {'liquidity': dot_amt, 'LRNA': dot_lrna},
        'HDX': {'liquidity': hdx_amt, 'LRNA': hdx_lrna}
    }

    initial_dex = OmnipoolState(
        tokens=tokens,
        lrna_fee=lrna_fee,
        asset_fee=asset_fee,
        preferred_stablecoin='USDT',
    )
    orig_price = initial_dex.price('DOT', 'USDT')
    dex_price = orig_price / ((1 - lrna_fee) * (1 - asset_fee))
    bid_price = dex_price / (1 - cex_fee) * price_mult
    bid_quantity = 100000
    p = 1e-10

    tkn = 'DOT'
    numeraire = 'USDT'
    init_holding = 1000000

    initial_cex = CentralizedMarket(
        order_book={
            ('DOT', 'USDT'): OrderBook(bids=[[bid_price, bid_quantity]], asks=[]),
        }, trade_fee=cex_fee
    )

    init_agent = Agent(holdings={'USDT': init_holding, 'DOT': init_holding, 'HDX': init_holding}, unique_id='bot')
    amt = calculate_arb_amount(
        buy_ex=initial_dex,
        sell_ex=initial_cex,
        buy_ex_tkn_pair=(tkn, numeraire),
        sell_ex_tkn_pair=(tkn, numeraire),
        sell_ex_max_sell=max_trade,
        buy_ex_max_sell=max_trade,
        precision=p,
        max_iters=50
    )
    test_agent = init_agent.copy()
    test_dex = initial_dex.copy()
    test_cex = initial_cex.copy()
    test_dex.swap(test_agent, tkn_buy=tkn, tkn_sell=numeraire, buy_quantity=amt)
    dex_price = test_dex.buy_spot(tkn, numeraire)
    cex_price = initial_cex.sell_spot(tkn, numeraire)

    if abs(init_holding - test_agent.holdings[tkn]) / max_trade - 1 > p:
        raise
    if abs(init_holding - test_agent.holdings[numeraire]) / max_trade - 1 > p:
        raise

    # checks if the cex price and spot price have been brought into alignment
    if abs(dex_price - cex_price) / cex_price > p and abs(dex_price - cex_price) > p and amt != bid_quantity:
        # if cex price and spot price aren't in alignment, it should be because of trade size limit
        if ((max_trade - abs(init_holding - test_agent.holdings[tkn])) / init_holding) > 1e-10:
            if ((max_trade - abs(init_holding - test_agent.holdings[numeraire])) / init_holding) > 1e-10:
                raise

    test_cex.swap(test_agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=amt)

    if amt == bid_quantity and dex_price > cex_price:
        raise

    profit = calculate_profit(init_agent, test_agent)
    for tkn in profit:
        assert profit[tkn] >= 0


@settings(phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target], deadline=timedelta(milliseconds=500))
# @settings(max_examples=1)
@given(
    usdt_amt=st.floats(min_value=100000, max_value=1000000),
    dot_price=st.floats(min_value=0.01, max_value=1000),
    hdx_price=st.floats(min_value=0.01, max_value=1000),
    dot_wt=st.floats(min_value=0.05, max_value=0.50),
    hdx_wt=st.floats(min_value=0.01, max_value=0.20),
    price_mult=st.floats(min_value=0.2, max_value=0.99),
    lrna_fee=st.floats(min_value=0.0001, max_value=0.001),
    asset_fee=st.floats(min_value=0.0001, max_value=0.004),
    cex_fee=st.floats(min_value=0.0001, max_value=0.005),
)
def test_calculate_arb_amount_ask(
        usdt_amt: float,
        dot_price: float,
        hdx_price: float,
        dot_wt: float,
        hdx_wt: float,
        price_mult: float,
        lrna_fee: float,
        asset_fee: float,
        cex_fee: float
):
    usdt_wt = 1 - dot_wt - hdx_wt
    dot_lrna = dot_wt / usdt_wt * usdt_amt
    dot_amt = dot_lrna / dot_price
    hdx_lrna = hdx_wt / usdt_wt * usdt_amt
    hdx_amt = hdx_lrna / hdx_price

    tokens = {
        'USDT': {'liquidity': mpf(usdt_amt), 'LRNA': mpf(usdt_amt)},
        'DOT': {'liquidity': mpf(dot_amt), 'LRNA': mpf(dot_lrna)},
        'HDX': {'liquidity': mpf(hdx_amt), 'LRNA': mpf(hdx_lrna)}
    }

    initial_dex = OmnipoolState(
        tokens=tokens,
        lrna_fee=lrna_fee,
        asset_fee=asset_fee,
        preferred_stablecoin='USDT',
    )
    sell_spot = initial_dex.sell_spot('DOT', 'USDT') / ((1 - lrna_fee) * (1 - asset_fee))
    ask_price = sell_spot / (1 - cex_fee) * price_mult
    ask_quantity = 10000
    p = 1e-10

    tkn = 'DOT'
    numeraire = 'USDT'

    initial_cex = CentralizedMarket(
        order_book={
            ('DOT', 'USDT'): OrderBook(bids=[], asks=[[mpf(ask_price), ask_quantity]]),
        }, trade_fee=cex_fee
    )

    init_agent = Agent(holdings={'USDT': 10000000, 'DOT': 10000000, 'HDX': 10000000}, unique_id='bot')
    amt = calculate_arb_amount(
        buy_ex=initial_cex,
        sell_ex=initial_dex,
        buy_ex_tkn_pair=(tkn, numeraire),
        sell_ex_tkn_pair=(tkn, numeraire),
        sell_ex_max_sell=init_agent.holdings[tkn],
        buy_ex_max_sell=init_agent.holdings[numeraire],
        precision=p,
        max_iters=50
    )
    test_agent = init_agent.copy()
    test_dex = initial_dex.copy()
    test_cex = initial_cex.copy()
    test_dex.swap(test_agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=amt)
    test_cex.swap(test_agent, tkn_sell=numeraire, tkn_buy=tkn, buy_quantity=amt)
    dex_price = test_dex.sell_spot(tkn, numeraire)
    cex_price = initial_cex.buy_spot(tkn, numeraire)

    if 1 - cex_price / dex_price > p and amt != ask_quantity:
        raise

    if amt == ask_quantity and cex_price > dex_price:
        raise

    profit = calculate_profit(init_agent, test_agent)
    for tkn in profit:
        assert profit[tkn] >= 0


@settings(phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target], deadline=timedelta(milliseconds=500))
# @settings(max_examples=1)
@given(
    usdt_amt=st.floats(min_value=100000, max_value=1000000),
    dot_price=st.floats(min_value=0.01, max_value=1000),
    hdx_price=st.floats(min_value=0.01, max_value=1000),
    dot_wt=st.floats(min_value=0.05, max_value=0.50),
    hdx_wt=st.floats(min_value=0.01, max_value=0.20),
    price_mult=st.floats(min_value=0.2, max_value=0.95),
    lrna_fee=st.floats(min_value=0.0001, max_value=0.001),
    asset_fee=st.floats(min_value=0.0001, max_value=0.004),
    cex_fee=st.floats(min_value=0.0001, max_value=0.005),
    max_trade=st.floats(min_value=1, max_value=100),
)
def test_calculate_arb_amount_ask_max_liquidity(
        usdt_amt: float,
        dot_price: float,
        hdx_price: float,
        dot_wt: float,
        hdx_wt: float,
        price_mult: float,
        lrna_fee: float,
        asset_fee: float,
        cex_fee: float,
        max_trade: float
):
    usdt_wt = 1 - dot_wt - hdx_wt
    dot_lrna = dot_wt / usdt_wt * usdt_amt
    dot_amt = dot_lrna / dot_price
    hdx_lrna = hdx_wt / usdt_wt * usdt_amt
    hdx_amt = hdx_lrna / hdx_price

    tokens = {
        'USDT': {'liquidity': usdt_amt, 'LRNA': usdt_amt},
        'DOT': {'liquidity': dot_amt, 'LRNA': dot_lrna},
        'HDX': {'liquidity': hdx_amt, 'LRNA': hdx_lrna}
    }

    initial_dex = OmnipoolState(
        tokens=tokens,
        lrna_fee=lrna_fee,
        asset_fee=asset_fee,
        preferred_stablecoin='USDT',
    )
    sell_spot = initial_dex.sell_spot('DOT', 'USDT')
    ask_price = sell_spot * (1 - cex_fee) * price_mult
    ask_quantity = 100000
    p = 1e-10

    tkn = 'DOT'
    numeraire = 'USDT'
    init_holding = 1000000

    initial_cex = CentralizedMarket(
        order_book={
            ('DOT', 'USDT'): OrderBook(bids=[], asks=[[ask_price, ask_quantity]]),
        }, trade_fee=cex_fee
    )

    init_agent = Agent(holdings={'USDT': init_holding, 'DOT': init_holding, 'HDX': init_holding}, unique_id='bot')
    amt = calculate_arb_amount(
        buy_ex=initial_cex,
        sell_ex=initial_dex,
        buy_ex_tkn_pair=(tkn, numeraire),
        sell_ex_tkn_pair=(tkn, numeraire),
        sell_ex_max_sell=max_trade,
        buy_ex_max_sell=max_trade,
        precision=p,
        max_iters=50
    )
    test_agent = init_agent.copy()
    test_dex = initial_dex.copy()
    test_cex = initial_cex.copy()
    test_dex.swap(test_agent, tkn_buy=numeraire, tkn_sell=tkn, sell_quantity=amt)
    dex_price = test_dex.sell_spot(tkn, numeraire)
    cex_price = initial_cex.buy_spot(tkn, numeraire)

    if abs(init_holding - test_agent.holdings[tkn]) / max_trade - 1 > p:
        raise

    # check if the cex price and spot price have been brought into alignment
    if abs(dex_price - cex_price) / cex_price > p and abs(dex_price - cex_price) > p and amt != ask_quantity:
        # if cex price and spot price aren't in alignment, it should be because of trade size limit
        if ((max_trade - abs(init_holding - test_agent.holdings[tkn])) / init_holding) > 1e-10:
            if ((max_trade - abs(init_holding - test_agent.holdings[numeraire])) / init_holding) > 1e-10:
                raise

    mid_holding = test_agent.holdings[numeraire]
    test_cex.swap(test_agent, tkn_buy=tkn, tkn_sell=numeraire, buy_quantity=amt)
    if abs(mid_holding - test_agent.holdings[numeraire]) / max_trade - 1 > p:
        raise

    if amt == ask_quantity and dex_price < cex_price:
        raise

    profit = calculate_profit(init_agent, test_agent)
    for tkn in profit:
        assert profit[tkn] >= 0


@given(
    dotusd_price_mult=st.floats(min_value=0.8, max_value=1.2),
    dot_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=4, max_size=4),
    hdxusd_price_mult=st.floats(min_value=0.8, max_value=1.2),
    hdxusd_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=4, max_size=4),
)
def test_process_next_swap(
        dotusd_price_mult: float,
        dot_amts: list,
        hdxusd_price_mult: float,
        hdxusd_amts: list
):
    initial_dex = OmnipoolState(
        tokens={
            'USDT': {'liquidity': mpf(2062772), 'LRNA': mpf(2062772)},
            'DOT': {'liquidity': mpf(350000), 'LRNA': mpf(1456248)},
            'HDX': {'liquidity': mpf(108000000), 'LRNA': mpf(494896)}
        },
        lrna_fee=0.0005,
        asset_fee=0.0025,
        preferred_stablecoin='USDT',
    )

    bid_multiples = [0.999, 0.99, 0.9, 0.8]
    ask_multiples = [1.001, 1.01, 1.1, 1.2]

    hdxusd_spot = initial_dex.price('HDX', 'USDT') * hdxusd_price_mult
    dotusd_spot = initial_dex.price('DOT', 'USDT') * dotusd_price_mult

    initial_cex = CentralizedMarket(
        order_book={
            ('DOT', 'USDT'): OrderBook(
                    bids=[[dotusd_spot * bid_multiples[i], dot_amts[i]] for i in range(2)],
                    asks=[[dotusd_spot * ask_multiples[i], dot_amts[i + 2]] for i in range(2)]
                ),
            ('HDX', 'USDT'): OrderBook(
                    bids=[[hdxusd_spot * bid_multiples[i], hdxusd_amts[i]] for i in range(2)],
                    asks=[[hdxusd_spot * ask_multiples[i], hdxusd_amts[i + 2]] for i in range(2)]
            )
        }, trade_fee=0.0016
    )

    initial_agent_dex = Agent(holdings={'USDT': mpf(1000000000), 'DOT': mpf(1000000000), 'HDX': mpf(1000000000)})
    initial_agent_cex = Agent(holdings={'USDT': mpf(1000000000), 'DOT': mpf(1000000000), 'HDX': mpf(1000000000)})

    test_dex = initial_dex.copy()
    test_agent_dex = initial_agent_dex.copy()
    test_agent_cex = initial_agent_cex.copy()
    test_cex = initial_cex.copy()
    buffer = 0.0
    init_max_liquidity = {
        'dex': {'USDT': mpf(1000), 'DOT': mpf(100), 'HDX': mpf(100000)},
        'cex': {'USDT': mpf(1000), 'DOT': mpf(100), 'HDX': mpf(100000)},
    }
    max_liquidity = copy.deepcopy(init_max_liquidity)
    tkn_pair = ('DOT', 'USDT')

    swap = process_next_swap(
        exchanges={'dex': test_dex, 'cex': test_cex},
        max_liquidity=max_liquidity,
        swap_config={'exchanges': {'dex': tkn_pair, 'cex': tkn_pair}, 'buffer': buffer},
        max_iters=20
    )

    if swap:

        diff_dex = {
            'DOT': test_dex.liquidity['DOT'] - initial_dex.liquidity['DOT'],
            'USDT': test_dex.liquidity['USDT'] - initial_dex.liquidity['USDT']
        }

        diff_max = {
            'DOT': (max_liquidity['dex']['DOT'] - init_max_liquidity['dex']['DOT']
                    + max_liquidity['cex']['DOT'] - init_max_liquidity['cex']['DOT']),
            'USDT': (max_liquidity['dex']['USDT'] - init_max_liquidity['dex']['USDT']
                     + max_liquidity['cex']['USDT'] - init_max_liquidity['cex']['USDT'])
        }

        diff_cex = {
            'DOT': -diff_max['DOT'] - diff_dex['DOT'],
            'USDT': -diff_max['USDT'] - diff_dex['USDT'],
            'HDX': 0
        }

        cex_swap, dex_swap = swap['cex'], swap['dex']
        dex_spot = initial_dex.price('DOT', 'USDT')
        if cex_swap['buy_asset'] != dex_swap['sell_asset'] or cex_swap['sell_asset'] != dex_swap['buy_asset']:
            raise AssertionError('Cex and dex swaps are not in the same pair.')
        cex_numeraire_amt = cex_swap['amount'] * cex_swap['price']
        if dex_swap['trade'] == 'sell':
            dex_numeraire_amt = dex_swap['min_buy']
            if dex_numeraire_amt < cex_numeraire_amt:  # check profitability
                raise AssertionError('Swap is not profitable.')
            if dex_swap['min_buy'] / dex_swap['amount'] > dex_spot:  # check dex slippage direction
                raise AssertionError('Dex slippage is not in the right direction.')
            if cex_swap['price'] < initial_cex.order_book[tkn_pair].bids[0][0]:  # check cex slippage direction
                raise AssertionError('Cex slippage is not in the right direction.')
        elif dex_swap['trade'] == 'buy':
            dex_numeraire_amt = dex_swap['max_sell']
            if dex_numeraire_amt > cex_numeraire_amt:  # check profitability
                raise AssertionError('Swap is not profitable.')
            if dex_swap['max_sell'] / dex_swap['amount'] < dex_spot:  # check dex slippage direction
                raise AssertionError('Dex slippage is not in the right direction.')
            if cex_swap['price'] > initial_cex.order_book[tkn_pair].asks[0][0]:  # check cex slippage direction
                raise AssertionError('Cex slippage is not in the right direction.')

        dex_profit = calculate_profit(initial_agent_dex, test_agent_dex)
        cex_profit = calculate_profit(initial_agent_cex, test_agent_cex)
        total_profit = {tkn: dex_profit[tkn] + cex_profit[tkn] for tkn in dex_profit}
        for tkn in total_profit:
            if total_profit[tkn] < 0:
                raise AssertionError('Profit is negative.')

        for tkn in initial_dex.asset_list:
            if tkn in max_liquidity['dex']:
                if test_dex.liquidity[tkn] - initial_dex.liquidity[tkn] != pytest.approx(
                        init_max_liquidity['dex'][tkn] - max_liquidity['dex'][tkn], 1e-20):
                    raise
        for tkn in initial_cex.asset_list:
            if tkn in max_liquidity['cex']:
                if diff_cex[tkn] != pytest.approx(
                        init_max_liquidity['cex'][tkn] - max_liquidity['cex'][tkn], 1e-20):
                    raise


@given(
    dotusd_price_mult=st.floats(min_value=0.8, max_value=1.2),
    dot_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=8, max_size=8),
    hdxdot_price_mult=st.floats(min_value=0.8, max_value=1.2),
    hdxdot_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=4, max_size=4),
    hdxusd_price_mult=st.floats(min_value=0.8, max_value=1.2),
    hdxusd_amts=st.lists(st.floats(min_value=10, max_value=10000), min_size=8, max_size=8),
)
def test_get_arb_swaps(
        dotusd_price_mult: float,
        dot_amts: list,
        hdxdot_price_mult: float,
        hdxdot_amts: list,
        hdxusd_price_mult: float,
        hdxusd_amts: list
):
    tokens = {
        'USDT': {
            'liquidity': 2062772,
            'LRNA': 2062772
        },
        'DOT': {
            'liquidity': 350000,
            'LRNA': 1456248
        },
        'HDX': {
            'liquidity': 108000000,
            'LRNA': 494896
        }
    }

    lrna_fee = 0.0005
    asset_fee = 0.0025
    cex_fee = 0.0016

    op_state = OmnipoolState(
        tokens=tokens,
        lrna_fee=lrna_fee,
        asset_fee=asset_fee,
        preferred_stablecoin='USDT',
    )

    dotusd_spot = op_state.price('DOT', 'USDT')
    dotusd_spot_adj = dotusd_spot * dotusd_price_mult

    dot_usdt_order_book = {
        'bids': [{'price': dotusd_spot_adj * 0.999, 'amount': dot_amts[0]},
                 {'price': dotusd_spot_adj * 0.99, 'amount': dot_amts[1]},
                 {'price': dotusd_spot_adj * 0.9, 'amount': dot_amts[2]},
                 {'price': dotusd_spot_adj * 0.8, 'amount': dot_amts[3]}],
        'asks': [{'price': dotusd_spot_adj * 1.001, 'amount': dot_amts[4]},
                 {'price': dotusd_spot_adj * 1.01, 'amount': dot_amts[5]},
                 {'price': dotusd_spot_adj * 1.1, 'amount': dot_amts[6]},
                 {'price': dotusd_spot_adj * 1.2, 'amount': dot_amts[7]}]
    }

    dot_usdt_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in dot_usdt_order_book['bids']],
                                        [[ask['price'], ask['amount']] for ask in dot_usdt_order_book['asks']])

    hdxusd_spot = op_state.price('HDX', 'USDT')
    hdxusd_spot_adj = hdxusd_spot * hdxusd_price_mult

    hdx_usdt_order_book = {
        'bids': [{'price': hdxusd_spot_adj * 0.999, 'amount': hdxusd_amts[0]},
                 {'price': hdxusd_spot_adj * 0.99, 'amount': hdxusd_amts[1]},
                 {'price': hdxusd_spot_adj * 0.9, 'amount': hdxusd_amts[2]},
                 {'price': hdxusd_spot_adj * 0.8, 'amount': hdxusd_amts[3]}],
        'asks': [{'price': hdxusd_spot_adj * 1.001, 'amount': hdxusd_amts[4]},
                 {'price': hdxusd_spot_adj * 1.01, 'amount': hdxusd_amts[5]},
                 {'price': hdxusd_spot_adj * 1.1, 'amount': hdxusd_amts[6]},
                 {'price': hdxusd_spot_adj * 1.2, 'amount': hdxusd_amts[7]}]
    }

    hdx_usdt_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in hdx_usdt_order_book['bids']],
                                        [[ask['price'], ask['amount']] for ask in hdx_usdt_order_book['asks']])

    hdxdot_spot = op_state.price('HDX', 'DOT')
    hdxdot_spot_adj = hdxdot_spot * hdxdot_price_mult

    hdx_dot_order_book = {
        'bids': [{'price': hdxdot_spot_adj * 0.99, 'amount': hdxdot_amts[0]},
                 {'price': hdxdot_spot_adj * 0.9, 'amount': hdxdot_amts[1]}],
        'asks': [{'price': hdxdot_spot_adj * 1.01, 'amount': hdxdot_amts[2]},
                 {'price': hdxdot_spot_adj * 1.1, 'amount': hdxdot_amts[3]}]
    }

    hdx_dot_order_book_obj = OrderBook([[bid['price'], bid['amount']] for bid in hdx_dot_order_book['bids']],
                                       [[ask['price'], ask['amount']] for ask in hdx_dot_order_book['asks']])

    order_book = {
        ('DOT', 'USDT'): dot_usdt_order_book_obj,
        ('HDX', 'USDT'): hdx_usdt_order_book_obj,
        ('HDX', 'DOT'): hdx_dot_order_book_obj
    }

    cex = CentralizedMarket(
        order_book=order_book,
        asset_list=['USDT', 'DOT', 'HDX'],
        trade_fee=cex_fee
    )

    cfg = [{"exchanges": {'dex': k, "cex": k}, "buffer": 0.0} for k in order_book]
    arb_swaps = get_arb_swaps({'dex': op_state, 'cex': cex}, cfg)
    initial_agent = Agent(holdings={'USDT': mpf(1000000), 'DOT': mpf(1000000), 'HDX': mpf(1000000)}, unique_id='bot')
    agent = initial_agent.copy()

    execute_arb({'dex': op_state, 'cex': cex}, agent, arb_swaps)

    profit = calculate_profit(initial_agent, agent)
    for tkn in profit:
        if profit[tkn] / initial_agent.holdings[tkn] < -1e-10:
            raise


def test_combine_step():

    cfg = [
        {'exchanges': {'omnipool': ('HDX', 'USDT'), 'kraken': ('HDX', 'USD')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('DOT', 'USDT'), 'kraken': ('DOT', 'USDT')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('WETH', 'USDT'), 'kraken': ('ETH', 'USDT')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('DOT', 'WETH'), 'kraken': ('DOT', 'ETH')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('WBTC', 'USDT'), 'kraken': ('BTC', 'USDT')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('iBTC', 'USDT'), 'kraken': ('BTC', 'USDT')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('DOT', 'WBTC'), 'kraken': ('DOT', 'BTC')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('DOT', 'iBTC'), 'kraken': ('DOT', 'BTC')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('WETH', 'WBTC'), 'kraken': ('ETH', 'BTC')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('WETH', 'iBTC'), 'kraken': ('ETH', 'BTC')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('ASTR', 'USDT'), 'kraken': ('ASTR', 'USD')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('CFG', 'USDT'), 'kraken': ('CFG', 'USD')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('BNC', 'USDT'), 'kraken': ('BNC', 'USD')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('GLMR', 'USDT'), 'kraken': ('GLMR', 'USD')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('INTR', 'USDT'), 'kraken': ('INTR', 'USD')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('DOT', 'USDT'), 'binance': ('DOT', 'USDT')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('DOT', 'WETH'), 'binance': ('DOT', 'ETH')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('DOT', 'WBTC'), 'binance': ('DOT', 'BTC')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('DOT', 'iBTC'), 'binance': ('DOT', 'BTC')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('WETH', 'USDT'), 'binance': ('ETH', 'USDT')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('WETH', 'WBTC'), 'binance': ('ETH', 'BTC')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('WETH', 'iBTC'), 'binance': ('ETH', 'BTC')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('WBTC', 'USDT'), 'binance': ('BTC', 'USDT')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('iBTC', 'USDT'), 'binance': ('BTC', 'USDT')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('ASTR', 'USDT'), 'binance': ('ASTR', 'USDT')}, 'buffer': 0.001},
        {'exchanges': {'omnipool': ('GLMR', 'USDT'), 'binance': ('GLMR', 'USDT')}, 'buffer': 0.001}
     ]

    # asset_list, asset_numbers, tokens, fees = get_omnipool_data(rpc='wss://rpc.hydradx.cloud', archive=False)
    #
    # kraken = get_centralized_market(config=cfg, exchange_name='kraken', trade_fee=0.0016, archive=False)
    # binance = get_centralized_market(config=cfg, exchange_name='binance', trade_fee=0.001, archive=False)
    # cex = {
    #     'kraken': kraken,
    #     'binance': binance
    # }
    # uncomment above to test with live data, below for archived data

    input_path = find_test_directory()
    input_path = os.path.join(input_path, 'data')
    asset_list, asset_numbers, tokens, fees = get_omnipool_data_from_file(input_path)

    cex = {}
    for exchange in ('kraken', 'binance'):
        cex[exchange] = CentralizedMarket(
            order_book=get_orderbooks_from_file(input_path=input_path)[exchange],
            unique_id=exchange,
            trade_fee={'kraken': 0.0016, 'binance': 0.001}[exchange]
        )
    kraken = cex['kraken']
    binance = cex['binance']

    omnipool = OmnipoolState(
        tokens=tokens,
        lrna_fee={asset: fees[asset]['protocol_fee'] for asset in asset_list},
        asset_fee={asset: fees[asset]['asset_fee'] for asset in asset_list},
        preferred_stablecoin='USDT',
        unique_id='dex'
    )

    equivalency_map = {
        'WETH': 'ETH',
        'XETH': 'ETH',
        'XXBT': 'BTC',
        'WBTC': 'BTC',
        'ZUSD': 'USD',
        'USDT': 'USD',
        'USDT10': 'USD',
        'USDT23': 'USD',
        'USDC': 'USD',
        'DAI': 'USD',
        'iBTC': 'BTC',
        'XBT': 'BTC',
    }

    exchanges_per_tkn = {
        tkn: sum(1 if tkn in exchange.asset_list else 0 for exchange in (omnipool, kraken, binance))
        for tkn in omnipool.asset_list + kraken.asset_list + binance.asset_list + list(asset_numbers.values())
    }
    exchanges = {
        'omnipool': omnipool, **cex
    }
    initial_agent = Agent(
        holdings={
            tkn: mpf(1000000) * exchanges_per_tkn[tkn]
            / (
                (binance.buy_spot(tkn, 'USDT') or kraken.buy_spot(tkn, 'USD') or omnipool.buy_spot(tkn, 'USDT')) or 1
            )
            for tkn in omnipool.asset_list + kraken.asset_list + binance.asset_list + list(asset_numbers.values())
        }
    )
    max_liquidity = {
        ex_name: {tkn: initial_agent.holdings[tkn] / exchanges_per_tkn[tkn] for tkn in initial_agent.holdings}
        for ex_name in exchanges
    }

    arb_swaps = get_arb_swaps(
        exchanges=exchanges,
        config=cfg,
        max_liquidity=max_liquidity,
        max_iters=20
    )
    test_exchanges = {ex_name: ex.copy() for ex_name, ex in exchanges.items()}
    test_agent_2 = initial_agent.copy()
    execute_arb(test_exchanges, test_agent_2, arb_swaps)
    profit = calculate_profit(initial_agent, test_agent_2, equivalency_map)
    profit_total = test_exchanges['binance'].value_assets(profit, equivalency_map)

    combine_exchanges, combine_agent = {ex_name: ex.copy() for ex_name, ex in exchanges.items()}, initial_agent.copy()
    combined_swaps = combine_swaps(
        exchanges=exchanges,
        agent=initial_agent,
        swaps=arb_swaps,
        equivalency_map=equivalency_map,
        max_liquidity=max_liquidity
    )
    execute_arb(exchanges=combine_exchanges, agent=combine_agent, swaps=combined_swaps)
    combined_profit = calculate_profit(initial_agent, combine_agent, equivalency_map)
    combined_profit_total = combine_exchanges['binance'].value_assets(combined_profit, equivalency_map)

    # see if iterating on that can get any extra profit
    iter_exchanges, iter_agent = {ex_name: ex.copy() for ex_name, ex in exchanges.items()}, initial_agent.copy()
    optimized_arb_swaps = get_arb_swaps(
        iter_exchanges, cfg, max_liquidity=max_liquidity
    )
    itered_swaps = combine_swaps(
        iter_exchanges, iter_agent, optimized_arb_swaps, equivalency_map,
        max_liquidity=max_liquidity
    )
    execute_arb(iter_exchanges, iter_agent, itered_swaps)
    iter_profit = calculate_profit(initial_agent, iter_agent, equivalency_map)
    iter_profit_total = iter_exchanges['binance'].value_assets(iter_profit, equivalency_map)

    for tkn in profit:
        if profit[tkn] / initial_agent.holdings[tkn] < -1e-10:
            raise AssertionError('Loss detected.')

    for tkn in combined_profit:
        if combined_profit[tkn] < -1e-10:
            raise AssertionError('Loss detected.')

    if profit_total > combined_profit_total:
        raise AssertionError('Loss detected.')
    else:
        print()
        print(f"Profit without combining: {profit_total}")
        print(f"Profit after combining: {combined_profit_total}")
        print(
            f"extra profit obtained by combining swaps: {combined_profit_total - profit_total}"
            f" ({(combined_profit_total - profit_total) / profit_total * 100}%)"
        )
        if iter_profit_total > combined_profit_total:
            print(f'Second iteration also gained {iter_profit_total - combined_profit_total}.')
        else:
            print('Iteration did not improve profit.')


def test_stableswap_router_arbitrage():
    # router = get_current_omnipool_router()
    # save_omnipool(omnipool)
    archive_path = os.path.join(find_test_directory(), 'archive')
    router = load_state(archive_path)
    omnipool = router.exchanges['omnipool']
    fourpool = router.exchanges['4-Pool']
    input_path = os.path.join(find_test_directory(), 'data')
    if not os.path.exists(input_path):
        input_path = 'hydradx/tests/data/'
    cex = {}
    for exchange in ('kraken', 'binance'):
        cex[exchange] = CentralizedMarket(
            order_book=get_orderbooks_from_file(input_path=input_path)[exchange],
            unique_id=exchange,
            trade_fee={'kraken': 0.0016, 'binance': 0.001}[exchange]
        )
    kraken = cex['kraken']
    binance = cex['binance']

    cfg = [
        {'exchanges': {'router': ('iBTC', 'USDT10'), 'kraken': ('BTC', 'USDT')}, 'buffer': 0.001},
        {'exchanges': {'router': ('WBTC', 'USDT23'), 'kraken': ('BTC', 'USDT')}, 'buffer': 0.001},
        {'exchanges': {'router': ('iBTC', 'USDT23'), 'kraken': ('BTC', 'USDT')}, 'buffer': 0.001},
        {'exchanges': {'router': ('WBTC', 'USDT10'), 'kraken': ('BTC', 'USDT')}, 'buffer': 0.001},
    ]
    exchanges = {'router': router, 'kraken': kraken, 'binance': binance}
    max_liquidity = {
        'router': {'WBTC': 10, 'iBTC': 10, 'USDT10': 1000000, 'USDT23': 1000000},
        'kraken': {'BTC': 10, 'USDT': 1000000}
    }
    swaps = get_arb_swaps(
        exchanges=exchanges,
        config=cfg,
        max_liquidity=max_liquidity,
        max_iters=20
    )
    agent = Agent(
        holdings={k: v for exchange in max_liquidity.values() for k, v in exchange.items()}
    )
    equivalency_map = {
        'WETH': 'ETH',
        'XETH': 'ETH',
        'XXBT': 'BTC',
        'WBTC': 'BTC',
        'ZUSD': 'USD',
        'USDT': 'USD',
        'USDT10': 'USD',
        'USDT23': 'USD',
        'USDC': 'USD',
        'DAI': 'USD',
        'iBTC': 'BTC',
        'XBT': 'BTC',
    }
    initial_agent = agent.copy()
    execute_arb(exchanges, agent, swaps)
    profit = calculate_profit(initial_agent, agent, equivalency_map)
    for asset in profit:
        if asset not in exchanges['binance'].asset_list and (
                asset not in equivalency_map or equivalency_map[asset] not in exchanges['binance'].asset_list):
            if 'USD' not in profit:
                profit['USD'] = 0
            if asset in omnipool.asset_list:
                # estimate value from omnipool price
                profit['USD'] += (
                    profit[asset] * omnipool.lrna_price(asset)

                    / omnipool.lrna['4-Pool'] * sum(fourpool.liquidity.values())
                )
    profit_total = exchanges['binance'].value_assets(profit, equivalency_map)
    print(profit_total)
    if profit_total <= 0:
        raise AssertionError('Arbitrageur should definitely make money.')
    er = 1
