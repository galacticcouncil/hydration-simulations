import numpy as np

from hydradx.apps.gigadot_modeling.utils import simulate_route, get_omnipool_minus_vDOT, get_slippage_dict
from hydradx.model.amm.money_market import MoneyMarket, MoneyMarketAsset, CDP
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.agents import Agent
from hypothesis import given, strategies as strat, assume, settings, reproduce_failure
import os

from hydradx.tests.utils import find_test_directory


def test_liquidity():
    import hydradx.apps.gigadot_modeling.liquidity  # throws error if liquidity.py has error


@given(strat.floats(min_value=0.01, max_value=100))
def test_get_omnipool_minus_vDOT(dot_mult):
    assets = {
        'HDX': {'liquidity': 1000000, 'LRNA': 1000000},
        'USDT': {'liquidity': 1000000, 'LRNA': 1000000},
        'DOT': {'liquidity': 1000000, 'LRNA': 1000000},
        'vDOT': {'liquidity': 1000000, 'LRNA': 1000000},
    }
    omnipool = OmnipoolState(assets)
    new_op = get_omnipool_minus_vDOT(omnipool, op_dot_tvl_mult=dot_mult)
    for tkn in assets:
        if tkn == 'vDOT':
            assert tkn not in new_op.asset_list
        elif tkn == 'DOT':
            assert new_op.liquidity[tkn] == omnipool.liquidity[tkn] * dot_mult
            assert new_op.lrna[tkn] == omnipool.lrna[tkn] * dot_mult
        else:
            assert new_op.liquidity[tkn] == omnipool.liquidity[tkn]
            assert new_op.lrna[tkn] == omnipool.lrna[tkn]


def test_simulate_route():
    assets = {
        'HDX': {'liquidity': 1000000, 'LRNA': 1000000},
        'USDT': {'liquidity': 1000000, 'LRNA': 1000000},
        'DOT': {'liquidity': 1000000, 'LRNA': 1000000}
    }
    omnipool = OmnipoolState(assets)
    ss_assets = {'DOT': 1000000, 'vDOT': 800000, 'aDOT': 1000000}
    peg = ss_assets['DOT'] / ss_assets['vDOT']
    stableswap = StableSwapPoolState(ss_assets, 100, peg=[peg, 1])
    agent = Agent(enforce_holdings=False)

    # within Omnipool

    buy_amt = 1
    routes = [
        [{'tkn_sell': 'HDX', 'tkn_buy': 'USDT', 'pool': "omnipool"}],  # within Omnipool
        [{'tkn_sell': 'DOT', 'tkn_buy': 'vDOT', 'pool': "gigaDOT"}],  # within StableSwap
        [{'tkn_sell': 'DOT', 'tkn_buy': 'aDOT', 'pool': "money market"}],  # within Money Market
        [
            {'tkn_sell': 'DOT', 'tkn_buy': 'aDOT', 'pool': "money market"},
            {'tkn_sell': 'aDOT', 'tkn_buy': 'vDOT', 'pool': "gigaDOT"}
        ],
        [
            {'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'pool': "omnipool"},
            {'tkn_sell': 'DOT', 'tkn_buy': 'aDOT', 'pool': "money market"},
            {'tkn_sell': 'aDOT', 'tkn_buy': 'vDOT', 'pool': "gigaDOT"}
        ]
    ]

    expected_sells = [1, 10/8, 1, 10/8, 10/8]

    for i, route in enumerate(routes):
        new_omnipool, new_stableswap, new_agent = simulate_route(omnipool, stableswap, agent, buy_amt, route)
        does_route_use_moneymarket = False
        for step in route:
            if step['pool'] == "money market":
                does_route_use_moneymarket = True
                break
        # assert abs(new_agent.get_holdings('HDX') + buy_amt) < 1e-4
        tkn_sell = route[0]['tkn_sell']
        tkn_buy = route[-1]['tkn_buy']
        assert new_agent.get_holdings(tkn_buy) == buy_amt
        for tkn in list(assets.keys()) + list(ss_assets.keys()):
            if tkn in ['DOT', 'aDOT'] and does_route_use_moneymarket:  # combine DOT and aDOT
                tkn_init = agent.get_holdings('DOT') + agent.get_holdings('aDOT')
                tkn_after = new_agent.get_holdings('DOT') + new_agent.get_holdings('aDOT')
                for tkn in ['DOT', 'aDOT']:
                    if tkn in omnipool.liquidity:
                        tkn_init += omnipool.liquidity[tkn]
                    if tkn in stableswap.liquidity:
                        tkn_init += stableswap.liquidity[tkn]
                    if tkn in new_omnipool.liquidity:
                        tkn_after += new_omnipool.liquidity[tkn]
                    if tkn in new_stableswap.liquidity:
                        tkn_after += new_stableswap.liquidity[tkn]
            else:
                tkn_init = agent.get_holdings(tkn)
                if tkn in omnipool.liquidity:
                    tkn_init += omnipool.liquidity[tkn]
                if tkn in stableswap.liquidity:
                    tkn_init += stableswap.liquidity[tkn]
                tkn_after = new_agent.get_holdings(tkn)
                if tkn in new_omnipool.liquidity:
                    tkn_after += new_omnipool.liquidity[tkn]
                if tkn in new_stableswap.liquidity:
                    tkn_after += new_stableswap.liquidity[tkn]
            assert tkn_init == tkn_after

        sell_amt = -1 * new_agent.get_holdings(tkn_sell)
        assert abs(sell_amt - expected_sells[i]) < 1e-5


def test_get_slippage_dict():

    def assert_slippage_matches(slippage, sell_amts_dicts, buy_sizes):
        for route_key in sell_amts_dicts:
            for tkn_pair in sell_amts_dicts[route_key]:
                init_price = sell_amts_dicts[route_key][tkn_pair][0] / buy_sizes[0]
                for i in range(len(sell_amts_dicts[route_key][tkn_pair])):
                    sell_amt = sell_amts_dicts[route_key][tkn_pair][i]
                    spot_sell_amt = buy_sizes[i] * init_price
                    slip = sell_amt / spot_sell_amt - 1
                    if slip == 0:
                        if slippage[route_key][tkn_pair][i] != 0:
                            raise AssertionError("Slippage doesn't match")
                    elif abs(slippage[route_key][tkn_pair][i] - slip) / slip > 1e-14:
                        raise AssertionError("Slippage doesn't match")

    sell_amts_dicts = {
        'route1': {('USDT', 'DOT'): [5, 10, 110]}
    }
    buy_sizes = [1, 2, 20]
    slippage = get_slippage_dict(sell_amts_dicts, buy_sizes)
    assert_slippage_matches(slippage, sell_amts_dicts, buy_sizes)

    sell_amts_dicts = {
        'route1': {
            ('USDT', 'DOT'): [5, 10, 110],
            ('ABC', 'DEF'): [7, 77, 777]
        },
        'route2': {
            ('tkn1', 'tkn2'): [6, 12, 120]
        }
    }
    buy_sizes = [1, 2, 3]
    slippage = get_slippage_dict(sell_amts_dicts, buy_sizes)
    assert_slippage_matches(slippage, sell_amts_dicts, buy_sizes)


def test_hsm():
    from hydradx.apps.hollar.hsm import hollar_burned


def test_hollar_init_distro():
    from hydradx.apps.hollar.hollar_init_distro import run_script
    run_script()


def test_changing_amp():
    from hydradx.apps.gigadot_modeling import changing_amp


def test_fees_volume_comp():
    from hydradx.apps.fees import fees_volume_comp


def test_hdx_buybacks():
    from hydradx.apps.fees import hdx_buybacks


def test_hdx_fees():
    from hydradx.apps.fees import hdx_fees


def test_oracle_comparison():
    from hydradx.apps.fees.oracle_comparison import run_app
    run_app(7_200_000, 7_201_000, 'AAVE')


def test_arb_oracle_comp():
    from hydradx.apps.fees import arb_oracle_comp


def test_eth_params():
    os.chdir(find_test_directory())
    os.chdir('../apps/money_market')
    from hydradx.apps.money_market.eth_params import run_app
    run_app()


def test_add_withdraw():
    from hydradx.apps.everything_is_collateral import add_withdraw_losses
    # add_withdraw_losses.scenario_1()
    add_withdraw_losses.run_and_plot()


def test_slip_fees():
    from hydradx.apps.fees import slip_fees_comparison
    slip_fees_comparison.run_and_plot()


def test_slip_fees_chart():
    from hydradx.apps.fees import slip_fees_chart
    router = slip_fees_chart.load_omnipool_router()
    omnipool = router.exchanges['omnipool']
    omnipool.asset_fee = 0
    omnipool.lrna_fee = 0
    omnipool.max_lrna_fee = 1
    omnipool.max_asset_fee = 1
    omnipool.slip_factor = 1.0
    slip_fees_chart.plot_trade_sizes("HDX", "DOT", router, omnipool)


def test_hdx_h2o():
    from hydradx.apps.omnipool import hdx_h2o

