import copy
import math

import pytest
from hypothesis import given, strategies as st, assume, settings, reproduce_failure
from mpmath import mp, mpf
import os

os.chdir('../..')

from hydradx.model import run, processing
from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.omnipool_amm import DynamicFee, OmnipoolState, OmnipoolLiquidityPosition
from hydradx.model.amm.trade_strategies import constant_swaps, omnipool_arbitrage
from hydradx.tests.strategies_omnipool import omnipool_reasonable_config, omnipool_config, assets_config
import hydradx.model.production_settings as production_settings

def test_price_change_from_trade():
    liquidity = {'HDX': mpf(10000000), 'USD': mpf(1000000)}
    lrna = {'HDX': mpf(1000000), 'USD': mpf(1000000)}
    initial_state = oamm.OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=0.0025,
        lrna_fee=0.0005,
        slip_factor=0,
        lrna_mint_pct=1.0,
        lrna_fee_burn=0
    )
    initial_state.max_lrna_fee = 0.01
    agent = Agent(enforce_holdings=False)
    sell_quantity = mpf(10000)
    omnipool = initial_state.copy()
    for i in range(1000):
        omnipool.swap(
            agent=agent,
            tkn_sell='USD',
            tkn_buy='HDX',
            sell_quantity=sell_quantity
        )
        omnipool.swap(
            agent=agent,
            tkn_sell='HDX',
            tkn_buy='USD',
            sell_quantity=agent.holdings['HDX']
        )
    omnipool.swap(tkn_sell="HDX", tkn_buy="USD", buy_quantity=-agent.holdings['USD'] / 2, agent=agent)
    start_price_usd = initial_state.liquidity['USD'] / initial_state.lrna['USD']
    end_price_usd = omnipool.liquidity['USD'] / omnipool.lrna['USD']
    start_price_hdx = initial_state.liquidity['HDX'] / initial_state.lrna['HDX']
    end_price_hdx = omnipool.liquidity['HDX'] / omnipool.lrna['HDX']
    omnipool.lrna['USD'] += omnipool.lrna_fee_destination.holdings['LRNA'] / 2
    omnipool.lrna['HDX'] += omnipool.lrna_fee_destination.holdings['LRNA'] / 2
    projected_end_price_usd = omnipool.liquidity['USD'] / omnipool.lrna['USD']
    projected_end_price_hdx = omnipool.liquidity['HDX'] / omnipool.lrna['HDX']
    pass


def test_price_after_fees():
    omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(10000000), 'LRNA': mpf(10000)},
            'USD': {'liquidity': mpf(100000), 'LRNA': mpf(10000)}
        },
        asset_fee=0.00125,
        lrna_fee=0.00025,
        slip_factor=1.0,  # 1.0,
        lrna_fee_burn=1.0,
        lrna_mint_pct=1.0
    )
    omnipool.max_lrna_fee = 0.01
    agent = Agent(enforce_holdings=False)
    start_price = omnipool.lrna_price("HDX")
    for _ in range(10):
        omnipool.swap(agent, tkn_sell='HDX', tkn_buy='USD', sell_quantity=mpf(100000))

    for _ in range(10):
        omnipool.swap(agent, tkn_sell='USD', tkn_buy='HDX', buy_quantity=mpf(100000))
    end_price = omnipool.lrna_price("HDX")
    pass


def test_adot_minting_lp():
    initial_omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(100000000), 'LRNA': mpf(1000000)},
            'aDOT': {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            'USD': {'liquidity': mpf(20000000), 'LRNA': mpf(20000000)}
        },
        preferred_stablecoin='USD'
    )
    omnipool = initial_omnipool.copy()
    no_lp_omnipool = omnipool.copy()
    yield_per_block = mpf(1) / 100
    sim_length = 1
    lp = Agent(holdings={'aDOT': mpf(10000)})
    holder = lp.copy()

    omnipool.add_liquidity(
        agent = lp,
        tkn_add = 'aDOT',
        quantity = lp.holdings['aDOT']
    )
    for _ in range(sim_length):
        omnipool.liquidity['aDOT'] *= 1 + yield_per_block  # simulate aDOT appreciation
        no_lp_omnipool.liquidity['aDOT'] *= 1 + yield_per_block  # simulate aDOT appreciation
        holder.holdings['aDOT'] *= 1 + yield_per_block
        omnipool.lrna['aDOT'] *= 1 + yield_per_block  # simulate LRNA appreciation

    arbitrageur = Agent(
        enforce_holdings=False,
        trade_strategy=omnipool_arbitrage('omnipool')
    )
    early_exit_lp = lp.copy()
    no_exit_omnipool = omnipool.copy()
    omnipool.copy().remove_liquidity(
        agent=early_exit_lp,
        tkn_remove='aDOT'
    )
    end_state = GlobalState(
        pools=[omnipool],
        agents=[arbitrageur],
        external_market={tkn: initial_omnipool.usd_price(tkn) for tkn in initial_omnipool.liquidity}
    ).evolve()
    omnipool.remove_liquidity(
        agent = lp,
        tkn_remove = 'aDOT'
    )
    diff = lp.holdings['aDOT'] - holder.holdings['aDOT']
    pass


def test_lp_results_with_slip_fees():
    initial_omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(90000), 'LRNA': mpf(900)},
            'USD': {'liquidity': mpf(90), 'LRNA': mpf(90)}
        },
        preferred_stablecoin='USD',
        slip_factor=0,
        lrna_fee=0.0005,
        asset_fee=0.00125,
        lrna_mint_pct=0,
        lrna_fee_burn=0
    )
    initial_omnipool.max_lrna_fee = 0.05
    initial_omnipool.max_asset_fee = 0.05
    lp_usd = Agent(holdings={'USD': mpf(10)})
    lp_hdx = Agent(holdings={'HDX': mpf(10000)})
    initial_omnipool.add_liquidity(
        agent=lp_usd,
        tkn_add='USD',
        quantity=lp_usd.holdings['USD']
    )
    initial_omnipool.add_liquidity(
        agent=lp_hdx,
        tkn_add='HDX',
        quantity=lp_hdx.holdings['HDX']
    )
    omnipool = initial_omnipool.copy()
    treasury_agent = Agent(enforce_holdings=False, unique_id='treasury')
    # omnipool.lrna_fee_destination = treasury_agent

    trader = Agent(enforce_holdings=False, unique_id='trader')
    for i in range(100):
        omnipool.swap(
            agent=trader,
            tkn_sell='USD',
            tkn_buy='HDX',
            sell_quantity=mpf(1)
        )
        hdx_sell_quantity = initial_omnipool.liquidity['HDX'] - omnipool.liquidity['HDX']
        delta = hdx_sell_quantity / 2
        target_price = initial_omnipool.usd_price('HDX')
        for j in range(200):
            usd_buy, delta_q_hdx, delta_q_usd, asset_fee_total, lrna_fee_total, slip_fee_buy, slip_fee_sell = omnipool.calculate_out_given_in(
                tkn_buy='USD', tkn_sell='HDX', sell_quantity=hdx_sell_quantity
            )
            after_state = {
                'HDX': {
                    'liquidity': omnipool.liquidity['HDX'] + hdx_sell_quantity,
                    'LRNA': omnipool.lrna['HDX'] + delta_q_hdx + slip_fee_sell
                },
                'USD': {
                    'liquidity': omnipool.liquidity['USD'] - usd_buy,
                    'LRNA': omnipool.lrna['USD'] + delta_q_usd + slip_fee_buy
                }
            }
            after_price = after_state['HDX']['LRNA'] / after_state['HDX']['liquidity'] * after_state['USD']['liquidity'] / after_state['USD']['LRNA']
            if after_price < target_price:
                hdx_sell_quantity -= delta
            else:
                hdx_sell_quantity += delta
            delta /= 2

        omnipool.swap(
            agent=trader,
            tkn_sell='HDX',
            tkn_buy='USD',
            sell_quantity=hdx_sell_quantity
        )
    pre_withdraw_omnipool = omnipool.copy()
    print(f"""
        after swaps:
        (HDX: {round(omnipool.liquidity['HDX'], 6)}, LRNA: {round(omnipool.lrna['HDX'], 6)})
        (USD: {round(omnipool.liquidity['USD'], 6)}, LRNA: {round(omnipool.lrna['USD'], 6)})
    """)
    omnipool.remove_liquidity(
        agent=lp_usd,
        tkn_remove='USD'
    )
    omnipool.remove_liquidity(
        agent=lp_hdx,
        tkn_remove='HDX'
    )
    diff_usd = lp_usd.get_holdings('USD') - lp_usd.initial_holdings['USD']
    diff_hdx = (lp_hdx.get_holdings('HDX') - lp_hdx.initial_holdings['HDX']) * omnipool.usd_price('HDX')
    print(f"""
        LPs withdraw:
        USD: {lp_usd.holdings['USD']}
        HDX: {lp_hdx.holdings['HDX'] * omnipool.usd_price('HDX')}
    """)
    pass


def test_out_given_in():
    omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(9000000), 'LRNA': mpf(900000)},
            'USD': {'liquidity': mpf(90000), 'LRNA': mpf(90000)}
        },
        preferred_stablecoin='USD',
        # slip_factor=1,
        lrna_fee_burn=0,
        lrna_mint_pct=0,
        asset_fee=0.0025,
        lrna_fee=0.0005
    )
    omnipool.max_lrna_fee = 0.1

    trader = Agent(enforce_holdings=False, unique_id='trader')
    hdx_sell_quantity = 1002340
    usd_buy, delta_q_hdx, delta_q_usd, asset_fee_total, lrna_fee_total, slip_fee_buy, slip_fee_sell = omnipool.calculate_out_given_in(
        tkn_buy='USD', tkn_sell='HDX', sell_quantity=hdx_sell_quantity
    )
    after_state = {
        'HDX': {
            'liquidity': omnipool.liquidity['HDX'] + hdx_sell_quantity,
            'LRNA': omnipool.lrna['HDX'] + delta_q_hdx + slip_fee_sell
        },
        'USD': {
            'liquidity': omnipool.liquidity['USD'] - usd_buy,
            'LRNA': omnipool.lrna['USD'] + delta_q_usd + slip_fee_buy
        }
    }

    omnipool.swap(
        agent=trader,
        tkn_sell='HDX',
        tkn_buy='USD',
        sell_quantity=hdx_sell_quantity
    )
    assert omnipool.lrna['HDX'] == pytest.approx(after_state['HDX']['LRNA'], rel=1e-12)
    assert omnipool.liquidity['HDX'] == pytest.approx(after_state['HDX']['liquidity'], rel=1e-12)
    assert omnipool.lrna['USD'] == pytest.approx(after_state['USD']['LRNA'], rel=1e-12)
    assert omnipool.liquidity['USD'] == pytest.approx(after_state['USD']['liquidity'], rel=1e-12)
    pass


def test_lp_loss_with_deposit():
    omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(10000000), 'LRNA': mpf(1000000)},
            'USD': {'liquidity': mpf(100000), 'LRNA': mpf(100000)}
        },
        preferred_stablecoin='USD',
        slip_factor=1,
        lrna_fee=0.0001,
        asset_fee=0.00125
    )
    lp = Agent(holdings={'HDX': mpf(10000)})
    omnipool.add_liquidity(
        agent=lp,
        tkn_add='HDX',
        quantity=lp.holdings['HDX']
    )
    omnipool.liquidity