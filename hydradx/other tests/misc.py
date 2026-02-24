import copy
import datetime
from operator import truediv

import math

import pytest
from hypothesis import given, strategies as st, assume, settings, reproduce_failure
from mpmath import mp, mpf
import os
from pathlib import Path


os.chdir('../..')

from hydradx.model.indexer_utils import get_current_omnipool, get_current_omnipool_router, get_blocks_at_timestamps \
    , get_omnipool_trades, get_omnipool_asset_data, get_asset_info_by_ids, get_omnipool_liquidity_at_intervals \
    , get_latest_stableswap_data, get_omnipool_liquidity, get_stableswap_pools
from hydradx.model import run
from hydradx.model.processing import load_state
from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import GlobalState
from hydradx.model.run import run
from hydradx.model.amm.omnipool_amm import OmnipoolState, OmnipoolLiquidityPosition
from hydradx.model.amm.trade_strategies import omnipool_arbitrage

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
        # omnipool.lrna['aDOT'] *= 1 + yield_per_block  # simulate LRNA appreciation

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
    holder_gain = holder.holdings['aDOT'] - holder.initial_holdings['aDOT']
    reward_gain_pct = (holder_gain + diff) / holder_gain * 100
    pass


def calculate_arb(
        omnipool: OmnipoolState,
        tkn_sell: str,
        tkn_buy: str,
        target_price: float
):
    overshot = False
    sell_quantity = 1
    delta = 0.5
    for j in range(200):
        buy_quantity, delta_q_hdx, delta_q_usd, asset_fee_total, lrna_fee_total, slip_fee_buy, slip_fee_sell = omnipool.calculate_out_given_in(
            tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=sell_quantity
        )
        after_state = {
            tkn_sell: {
                'liquidity': omnipool.liquidity[tkn_sell] + sell_quantity,
                'LRNA': omnipool.lrna[tkn_sell] + delta_q_hdx
            },
            tkn_buy: {
                'liquidity': omnipool.liquidity[tkn_buy] - buy_quantity,
                'LRNA': omnipool.lrna[tkn_buy] + delta_q_usd
            }
        }
        after_price = after_state[tkn_sell]['LRNA'] / after_state[tkn_sell]['liquidity'] * after_state[tkn_buy]['liquidity'] / \
                      after_state[tkn_buy]['LRNA']
        if after_price < target_price:
            if not overshot:
                delta /= 2
                overshot = True
            sell_quantity -= delta
        else:
            sell_quantity += delta
        if overshot:
            delta /= 2
        else:
            delta *= 2

    return sell_quantity


def test_lp_results_with_slip_fees():
    initial_omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(90000), 'LRNA': mpf(900)},
            'USD': {'liquidity': mpf(90), 'LRNA': mpf(90)}
        },
        preferred_stablecoin='USD',
        slip_factor=1.0,
        lrna_fee=0.0005,
        asset_fee=0.00125,
        # lrna_mint_pct=0,
        # lrna_fee_burn=0,
        withdrawal_fee=False
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
    omnipool.lrna_fee_destination = treasury_agent

    trader = Agent(enforce_holdings=False, unique_id='trader')
    for i in range(1):
        omnipool.swap(
            agent=trader,
            tkn_sell='USD',
            tkn_buy='HDX',
            sell_quantity=mpf(10)
        )
        target_price = initial_omnipool.usd_price('HDX')
        sell_quantity = calculate_arb(
            omnipool=omnipool,
            tkn_sell='HDX',
            tkn_buy='USD',
            target_price=target_price
        )
        omnipool.swap(
            agent=trader,
            tkn_sell='HDX',
            tkn_buy='USD',
            sell_quantity=sell_quantity
        )

    print("adding LRNA back into pools...")
    usd_lrna_ratio = omnipool.lrna['USD'] / sum(omnipool.lrna.values())
    omnipool.lrna['USD'] += treasury_agent.holdings['LRNA'] * usd_lrna_ratio
    omnipool.lrna['HDX'] += treasury_agent.holdings['LRNA'] * (1 - usd_lrna_ratio)

    pre_withdraw_omnipool = omnipool.copy()
    print(f"""
        after swaps:
        (HDX: {round(omnipool.liquidity['HDX'], 6)}, LRNA: {round(omnipool.lrna['HDX'], 6)})
        (USD: {round(omnipool.liquidity['USD'], 6)}, LRNA: {round(omnipool.lrna['USD'], 6)})
        HDX/USD price: {pre_withdraw_omnipool.usd_price('HDX')}
    """)

    omnipool.remove_liquidity(
        agent=lp_hdx,
        tkn_remove='HDX'
    )
    omnipool.remove_liquidity(
        agent=lp_usd,
        tkn_remove='USD'
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
        slip_factor=1,
        lrna_fee_burn=0.5,
        lrna_mint_pct=0,
        asset_fee=0.0025,
        lrna_fee=0.0005,
        # lrna_fee_destination=Agent(enforce_holdings=False, unique_id='treasury')
    )
    omnipool.max_lrna_fee = 0.1
    omnipool.max_asset_fee = 0.1

    trader = Agent(enforce_holdings=False, unique_id='trader')
    hdx_sell_quantity = mpf(1002340)
    usd_buy, delta_q_hdx, delta_q_usd, asset_fee_total, lrna_fee_total, slip_fee_buy, slip_fee_sell \
        = omnipool.calculate_out_given_in(
            tkn_buy='USD', tkn_sell='HDX', sell_quantity=hdx_sell_quantity
        )
    after_state = {
        'HDX': {
            'liquidity': omnipool.liquidity['HDX'] + hdx_sell_quantity,
            'LRNA': omnipool.lrna['HDX'] + delta_q_hdx
        },
        'USD': {
            'liquidity': omnipool.liquidity['USD'] - usd_buy,
            'LRNA': omnipool.lrna['USD'] + delta_q_usd
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
    treasury_agent = Agent(enforce_holdings=False, unique_id='treasury')
    initial_omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(10000000), 'LRNA': mpf(1000000)},
            'USD': {'liquidity': mpf(100000), 'LRNA': mpf(1000000)}
        },
        preferred_stablecoin='USD',
        # slip_factor=1,
        lrna_fee=0,
        asset_fee=0,
        lrna_fee_destination=treasury_agent,
        withdrawal_fee=False,
        lrna_mint_pct=0,
        lrna_fee_burn=0
    )
    omnipool = initial_omnipool.copy()
    lp_hdx = Agent(holdings={'HDX': mpf(10000)})
    lp_usd = Agent(holdings={'USD': mpf(1000)})
    start_value_hdx = lp_hdx.holdings['HDX'] * omnipool.usd_price('HDX')
    omnipool.add_liquidity(
        agent=lp_hdx,
        tkn_add='HDX',
        quantity=lp_hdx.holdings['HDX']
    )
    omnipool.add_liquidity(
        agent=lp_usd,
        tkn_add='USD',
        quantity=lp_usd.holdings['USD']
    )
    omnipool.lrna['HDX'] *= 2  # dump some free LRNA into HDX pool
    sell_quantity = calculate_arb(
        omnipool=omnipool,
        tkn_sell='HDX',
        tkn_buy='USD',
        target_price=initial_omnipool.usd_price('HDX')
    )
    omnipool.swap(
        agent=Agent(enforce_holdings=False),
        tkn_sell='HDX',
        tkn_buy='USD',
        sell_quantity=sell_quantity
    )
    end_value_hdx = omnipool.cash_out(lp_hdx, denomination='USD')  # gains, but less than 50% of the liquidity increase
    end_value_usd = omnipool.cash_out(lp_usd, denomination='USD') # loses a bit
    pass


def test_lrna_price_on_withdrawal():
    initial_omnipool = OmnipoolState(
        tokens={
            **{f"token{i}": {'liquidity': mpf(1100000), 'LRNA': mpf(1100000)} for i in range(1, 10)},
            'USD': {'liquidity': mpf(10000000), 'LRNA': mpf(100000000)},
            'HDX': {'liquidity': mpf(10000000), 'LRNA': mpf(1000000)}
        },
        preferred_stablecoin='USD',
        slip_factor=1,
        lrna_fee=0,
        asset_fee=0,
        lrna_mint_pct=0.0,
        lrna_fee_burn=0.0,
        withdrawal_fee=False
    )
    lps = {
        i: Agent(enforce_holdings=False, unique_id=f'lp_{i}') for i in range(1, 10)
    }
    trader = Agent(enforce_holdings=False, unique_id='trader')
    omnipool = initial_omnipool.copy()
    # lp_hdx = Agent(holdings={'HDX': omnipool.liquidity['HDX']})
    initial_lrna_price = omnipool.usd_price('LRNA')
    for i in range(1, 10):
        omnipool.add_liquidity(
            agent=lps[i],
            tkn_add=f'token{i}',
            quantity=omnipool.liquidity[f'token{i}']
        )
    for i in range(1, 10):
        omnipool.swap(
            agent=trader,
            tkn_sell=f'token{i}',
            tkn_buy='USD',
            sell_quantity=omnipool.liquidity[f'token{i}'] / 10
        )
    for i in range(1, 10):
        omnipool.remove_liquidity(
            agent=lps[i],
            tkn_remove=f'token{i}'
        )
    final_lrna_price = omnipool.usd_price('LRNA')
    lrna_price_drop = 1 - final_lrna_price / initial_lrna_price

    tkn_price_drop = {
        f'token{i}': 1 - omnipool.usd_price(f'token{i}') / initial_omnipool.usd_price(f'token{i}') for i in range(1, 10)
    }
    pass


def test_bitcoin_pump():
    router = get_current_omnipool_router(block_number=6600000)
    omnipool = router.exchanges['omnipool']
    initial_omnipool = omnipool.copy()
    arbitrageur = Agent(
        enforce_holdings=False,
        trade_strategy=omnipool_arbitrage('omnipool'),
        unique_id='arbitrageur'
    )
    trader = Agent(enforce_holdings=False, unique_id='trader')
    initial_state = GlobalState(
        pools=[omnipool],
        agents=[arbitrageur, trader],
        external_market={tkn: router.price(tkn, 'USDT') for tkn in router.asset_list}
    )
    router.swap(
        agent=trader,
        tkn_sell='USD',
        tkn_buy='BTC',
        buy_quantity=mpf(10000)
    )
    # simulate a BTC pump
    omnipool.liquidity['BTC'] *= 1.5

    sell_quantity = calculate_arb(
        omnipool=omnipool,
        tkn_sell='BTC',
        tkn_buy='USD',
        target_price=initial_omnipool.usd_price('BTC')
    )
    omnipool.swap(
        agent=trader,
        tkn_sell='BTC',
        tkn_buy='USD',
        sell_quantity=sell_quantity
    )
    pass


def test_bitcoin_pump_2():
    omnipool = OmnipoolState(
        tokens={
            'BTC': {'liquidity': 1, 'LRNA': 1000},
            'HDX': {'liquidity': 100000, 'LRNA': 1000},
            'USD': {'liquidity': 1000, 'LRNA': 1000}
        },
        asset_fee=0,
        lrna_fee=0,
        lrna_mint_pct=0,
        lrna_fee_burn=0,
        slip_factor=0,
        preferred_stablecoin='USD'
    )
    arbitrageur = Agent(
        trade_strategy=omnipool_arbitrage('omnipool', skip_assets='HDX'),
        enforce_holdings=False,
        unique_id='arbitrageur'
    )
    initial_lrna_price = omnipool.usd_price('LRNA')
    initial_state = GlobalState(
        pools=[omnipool],
        agents=[arbitrageur],
        external_market={
            'USD': 1,
            'HDX': 0.01,
            'BTC': 500
        }
    )
    end_state = run(initial_state, 1)[0]
    final_lrna_price = end_state.pools['omnipool'].usd_price('LRNA')
    pass


def test_slip_fees_november():
    dates = [
        datetime.datetime(2025, 11,i+1) for i in range(30)
    ] + [
        datetime.datetime(year=2025, month=12, day=i+1) for i in range(31)
    ]
    block_numbers = list(get_blocks_at_timestamps(dates).values())
    slip_fees_lrna = []
    regular_fees_lrna = []
    trade_volume = []
    for i in range(len(block_numbers) - 1):
        slip_fees_today = mpf(0)
        regular_fees_today = mpf(0)
        trade_volume_today = mpf(0)
        start_block = block_numbers[i]
        end_block = block_numbers[i + 1]
        omnipool = get_current_omnipool(block_number=start_block)
        omnipool.slip_factor = 1.0
        omnipool.max_asset_fee = 0.05
        omnipool.max_lrna_fee = 0.01
        lrna_price_in_usd = 1 / omnipool.lrna_price('2-Pool-Stbl')
        trades = get_omnipool_trades(
            min_block=start_block,
            max_block=end_block
        )
        for trade in trades:
            tkn_buy = trade['assetOut']
            tkn_sell = trade['assetIn']
            sell_quantity = trade['amountIn']

            if tkn_sell == 'H2O':
                tkn_sell = 'LRNA'
            if tkn_sell in omnipool.asset_list and tkn_buy in omnipool.asset_list:
                asset_fee_lrna = trade['assetFeeAmount'] * omnipool.lrna_price(tkn_buy)
                lrna_fee = trade['protocolFeeAmount']
                regular_fees_today += asset_fee_lrna + lrna_fee
                trade_volume_today += sell_quantity * omnipool.lrna_price(tkn_sell) * lrna_price_in_usd
                outputs = omnipool.calculate_out_given_in(
                    tkn_buy=tkn_buy,
                    tkn_sell=tkn_sell,
                    sell_quantity=sell_quantity
                )
                slip_fees_today += outputs[-2] + outputs[-1]
            else:
                pass

        print(f"{dates[i]} slip fees (USD) {slip_fees_today * lrna_price_in_usd}")
        print(f"regular fees: {regular_fees_today * lrna_price_in_usd} ({round(slip_fees_today / regular_fees_today * 100, 3)}%)")
        regular_fees_lrna.append(regular_fees_today)
        slip_fees_lrna.append(slip_fees_today)
        trade_volume.append(trade_volume_today)
    pass

def test_dot_crash():
    import datetime
    from hydradx.model.indexer_utils import get_omnipool_liquidity_at_intervals
    interval = datetime.timedelta(days=365)
    end_date = datetime.datetime.today()
    blocks = list(get_blocks_at_timestamps([end_date - interval, end_date]).values())
    balances = [get_omnipool_liquidity(block) for block in blocks]
    omnipool_2024 = OmnipoolState(
        tokens={tkn: {'liquidity': balances[0][tkn]['liquidity'], 'LRNA': balances[0][tkn]['LRNA']} for tkn in balances[0]}
    )
    omnipool_2025 = OmnipoolState(
        tokens={tkn: {'liquidity': balances[1][tkn]['liquidity'], 'LRNA': balances[1][tkn]['LRNA']} for tkn in balances[1]}
    )
    stablepool_2024 = get_stableswap_pools(blocks[0], 102)['102']
    stablepool_2025 = get_stableswap_pools(blocks[1], 102)['102']
    share_price_2024 = sum(stablepool_2024.liquidity.values()) / stablepool_2024.shares
    share_price_2025 = sum(stablepool_2025.liquidity.values()) / stablepool_2025.shares
    omnipool_2024.liquidity['USD'] = 1 / omnipool_2024.lrna_price('2-Pool-Stbl') * share_price_2024
    omnipool_2025.liquidity['USD'] = 1 / omnipool_2025.lrna_price('2-Pool-Stbl') * share_price_2025
    omnipool_2024.lrna['USD'] = 1
    omnipool_2025.lrna['USD'] = 1
    omnipool_2024.stablecoin = 'USD'
    omnipool_2025.stablecoin = 'USD'
    dot_price_2024 = omnipool_2024.price('DOT', '2-Pool-Stbl') * share_price_2024
    dot_price_2025 = omnipool_2025.price('DOT', '2-Pool-Stbl') * share_price_2025
    sell_quantity = calculate_arb(
        omnipool_2024, tkn_sell="DOT", tkn_buy="2-Pool-Stbl", target_price=omnipool_2025.price("DOT", "2-Pool-Stbl")
    )
    omnipool_dot_selloff = omnipool_2024.copy().swap(
        agent = Agent(enforce_holdings=False),
        tkn_sell="DOT",
        tkn_buy="2-Pool-Stbl",
        sell_quantity=sell_quantity
    )
    hdx_price_2024 = omnipool_2024.usd_price("HDX")
    hdx_price_2025 = omnipool_2025.usd_price("HDX")
    hdx_price_dot_selloff = omnipool_dot_selloff.price("HDX", "2-Pool-Stbl") * share_price_2025
    avg_asset_price_2024 = sum([
        omnipool_2024.usd_price(tkn) * omnipool_2024.lrna[tkn] / sum(omnipool_2024.lrna.values())
        for tkn in omnipool_2024.liquidity
    ])
    avg_asset_price_2025 = sum([
        omnipool_2025.usd_price(tkn) * omnipool_2025.lrna[tkn] / sum(omnipool_2025.lrna.values())
        for tkn in omnipool_2025.liquidity
    ])
    pass


def test_progressive_liquidity_removal():
    agents = [Agent(holdings={'USD': 1000, 'HDX': 1000}) for i in range(10)]
    omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': 10000, 'LRNA': 10},
            'USD': {'liquidity': 1000, 'LRNA': 100},
            'DOT': {'liquidity': 500, 'LRNA': 100}
        },
        preferred_stablecoin='USD',
        withdrawal_fee=False
    )
    start_hdx_price = omnipool.usd_price('HDX')
    for agent in agents:
        omnipool.add_liquidity(
            agent, quantity=agent.holdings['HDX'], tkn_add='HDX'
        )
        omnipool.add_liquidity(
            agent, quantity=agent.holdings['USD'], tkn_add='USD'
        )
    omnipool.swap(
        agent=Agent(),
        tkn_buy='HDX',
        tkn_sell='DOT',
        buy_quantity=10000
    )
    mid_hdx_price = omnipool.usd_price('HDX')
    lrna_holdings = []
    for agent in agents:
        omnipool.remove_liquidity(
            agent, quantity=agent.holdings[('omnipool', 'HDX')], tkn_remove='HDX'
        )
        if agent.get_holdings('LRNA') > 0:
            lrna_holdings.append(agent.get_holdings('LRNA'))
            omnipool.swap(agent, tkn_buy='USD', tkn_sell='LRNA', sell_quantity=agent.get_holdings('LRNA'))

    last_hdx_price = omnipool.usd_price('HDX')
    pass

    start_lrna_price = omnipool.usd_price('LRNA')
    omnipool.update()

    lrna_holdings = []
    for agent in agents:
        omnipool.remove_liquidity(
            agent, quantity=agent.holdings[('omnipool', 'USD')], tkn_remove='USD'
        )
        if agent.get_holdings('LRNA') > 0:
            lrna_holdings.append(agent.get_holdings('LRNA'))
            omnipool.swap(agent, tkn_buy='USD', tkn_sell='LRNA', sell_quantity=agent.get_holdings('LRNA'))

    last_lrna_price = omnipool.usd_price('LRNA')
    pass


def test_hdx_lrna_balance():
    assets = get_asset_info_by_ids()
    balances = get_omnipool_liquidity_at_intervals(
        interval=datetime.timedelta(days=1),
        asset_ids=['0'],
        start_time=datetime.datetime(2023, 12, 1),
        end_time=datetime.datetime(2025, 1, 1)
    )
    from matplotlib import pyplot as plt
    plt.figure(figsize=(20, 5))
    plt.xticks([i * 30 for i in range(14)], ['December', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'January'])
    plt.plot([b['HDX']['LRNA'] / b['HDX']['liquidity'] for b in balances.values()])
    plt.show()
    pass


@given(
    pct_hdx_pool_bought=st.floats(min_value=0.001, max_value=0.99),
    lrna_holdings=st.integers(min_value=100, max_value=50000)
)
def test_hdx_sandwich(pct_hdx_pool_bought: float, lrna_holdings: int):
    pct_hdx_pool_bought = 0.5
    lrna_holdings = 50000
    output_token = 'tBTC'

    path = Path(__file__).parent / 'cached data' / 'omnipools'
    router = load_state(
        path=path, filename='omnipool-2026-02-03.json'
    )

    # router = get_current_omnipool_router()
    # save_state(
    #     router, Path(__file__).parent / 'cached data' / 'omnipools',
    #     filename=f"omnipool-{datetime.date.today().strftime('%Y-%m-%d')}.json"
    # )

    omnipool: OmnipoolState = router.exchanges['omnipool']
    omnipool.lrna_mint_pct = 0.0  # disable LRNA minting for test
    omnipool.lrna_fee_burn = 0.0  # disable LRNA burning for test
    initial_omnipool = omnipool.copy()

    sell_quantity = omnipool.calculate_sell_from_buy(
        tkn_buy='HDX', tkn_sell=output_token, buy_quantity=omnipool.liquidity['HDX'] * pct_hdx_pool_bought
    )
    lrna_seller = Agent(enforce_holdings=False, holdings={'LRNA': lrna_holdings, output_token: sell_quantity})

    hdx_initial_price = router.price('HDX', 'Tether')
    output_initial_price = router.price(output_token, 'Tether')

    router.swap(
        agent=lrna_seller,
        tkn_buy='HDX',
        tkn_sell=output_token,
        sell_quantity=sell_quantity
    )

    hdx_buy_price = router.price('HDX', 'Tether')
    output_sell_price = router.price(output_token, 'Tether')

    router.swap(
        agent=lrna_seller,
        tkn_sell='LRNA',
        tkn_buy=output_token,
        sell_quantity=lrna_holdings
    )

    # checkpoint what the price *would* be without the extra mechanism
    hdx_otherwise_price = router.price('HDX', 'Tether')
    output_otherwise_price = router.price(output_token, 'Tether')

    # move LRNA into HDX pool as per proposed mechanism
    omnipool.lrna[output_token] -= lrna_holdings
    omnipool.lrna['HDX'] += lrna_holdings

    hdx_price_after_transfer = router.price('HDX', 'Tether')
    output_price_after_transfer = router.price(output_token, 'Tether')

    gain_from_lrna = lrna_seller.get_holdings(output_token)

    router.swap(lrna_seller, tkn_sell='HDX', tkn_buy=output_token, sell_quantity=lrna_seller.get_holdings('HDX'))
    buy_quantity = lrna_seller.get_holdings(output_token) - gain_from_lrna
    gain_from_arb = lrna_seller.get_holdings(output_token) - gain_from_lrna - lrna_seller.initial_holdings[output_token]

    hdx_final_price = router.price('HDX', 'Tether')
    output_final_price = router.price(output_token, 'Tether')
    if gain_from_arb > 0:
        profit = gain_from_arb * router.price(output_token, 'Tether')
        test_summary = f"""
        LRNA holdings: {lrna_holdings}
        {output_token} holdings: {sell_quantity}

        bought {pct_hdx_pool_bought * 100:.2f}% of HDX pool for {sell_quantity} {output_token}
        sold {lrna_holdings} LRNA for {gain_from_lrna} {output_token}
        {lrna_holdings} LRNA moved into HDX pool, raising price by {round(hdx_final_price / hdx_buy_price, 6)}%
        sold all HDX for {buy_quantity} {output_token}

        HDX initial price: {hdx_initial_price}
        HDX price after buying {pct_hdx_pool_bought * 100}% of pool: {hdx_buy_price}
        HDX price after LRNA transfer: {hdx_price_after_transfer}
        HDX price if no transfer: {hdx_otherwise_price}
        HDX final price: {hdx_final_price}

        {output_token} initial price: {output_initial_price}
        {output_token} price after buying HDX: {output_sell_price}
        {output_token} price after LRNA transfer: {output_price_after_transfer}
        {output_token} price if no transfer: {output_otherwise_price}
        {output_token} final price: {output_final_price}

        trader profit: {profit}
        HDX pools gained LRNA: {omnipool.lrna['HDX'] - initial_omnipool.lrna['HDX']}
        """
        print(test_summary)
        pass
    pass
