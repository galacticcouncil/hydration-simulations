from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import sys, os
import streamlit as st
import copy
import pytest
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.agents import Agent
from hydradx.model.plot_utils import color_gradient
from hydradx.model.indexer_utils import get_current_omnipool, get_omnipool_trades, get_current_omnipool_assets, \
    get_asset_info_by_ids

st.markdown("""
    <style>
        .stNumberInput button {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)
st.set_page_config(layout="wide")  # Put this at the very top of your script


def trade_to_price(pool: OmnipoolState, agent: Agent, tkn: str, target_price: float) -> float:
    trade_size = pool.calculate_trade_to_price('HDX', target_price)
    if trade_size > 0:
        tkn_sell = tkn
        tkn_buy = 'LRNA'
        sell_quantity = trade_size
        buy_quantity = None
        lrna_at_spot_price = sell_quantity * pool.lrna_price(tkn)  # how much LRNA would it get at spot price
        print(f"Trader sells {sell_quantity} {tkn}, pays {lrna_at_spot_price * pool.asset_fee(tkn)} LRNA in fees")
    elif trade_size < 0:
        tkn_sell = 'LRNA'
        tkn_buy = tkn
        buy_quantity = -trade_size * (1 - pool.asset_fee(tkn))
        sell_quantity = None
        lrna_at_spot_price = buy_quantity / pool.lrna_price(tkn)  # how much LRNA would it cost at spot price
        print(f"Trader buys {buy_quantity} {tkn}, pays {lrna_at_spot_price * pool.asset_fee(tkn)} LRNA in fees")
    else:
        return 0

    pool.swap(
        agent=agent, tkn_sell=tkn_sell, tkn_buy=tkn_buy, buy_quantity=buy_quantity, sell_quantity=sell_quantity
    )
    return sell_quantity or -buy_quantity


def remove_readd(pool: OmnipoolState, agent: Agent):
    pool.remove_liquidity(
        agent=agent,
        tkn_remove='HDX'
    )
    # print(f"LP removed liquidity as {agent.all_holdings()}")
    if agent.get_holdings('LRNA') > 0:
        start_hdx = agent.get_holdings('HDX')
        start_lrna = agent.get_holdings('LRNA')
        hdx_at_spot_price = 1 / pool.lrna_price('HDX') * start_lrna
        pool.swap(agent=agent, tkn_sell='LRNA', tkn_buy='HDX', sell_quantity=agent.get_holdings('LRNA'))
        hdx_back = agent.get_holdings('HDX') - start_hdx
        # print(f"LP swapped {start_lrna} LRNA to HDX, losing {hdx_at_spot_price - hdx_back} HDX in fees and slippage")
    # print(
    #     f"LP adds {round(agent.get_holdings('HDX'), 3)} HDX back to pool ({round(pool.liquidity['HDX'], 3)} HDX) as liquidity")
    pool.add_liquidity(
        agent=agent,
        tkn_add='HDX',
        quantity=agent.get_holdings('HDX')
    )


def linear_increasing_sequence(start_value, end_value, sequence_length):
    weights = list(range(1, sequence_length))
    total_weight = sum(weights)
    scale = (end_value - start_value) / total_weight

    seq = [start_value]
    acc = start_value
    for w in weights:
        acc += w * scale
        seq.append(acc)
    return seq


def scenario_1():
    print()

    lp_liquidity_pct = 0.25
    print(f"LP liquidity pct: {lp_liquidity_pct * 100}%")
    initial_omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': 1000000, 'LRNA': 1000000},
            'WETH': {'liquidity': 1400, 'LRNA': 2276599},
            'USD': {'liquidity': 1, 'LRNA': 1},
            'DAI': {'liquidity': 22682620000000, 'LRNA': 22682620000000},
            'DOT': {'liquidity': 88000, 'LRNA': 546461},
            'WBTC': {'liquidity': 47, 'LRNA': 1145210},
        },
        withdrawal_fee=False,
        lrna_fee_burn=0.5,
        lrna_mint_pct=1.0,
        asset_fee=0,
        lrna_fee=0,
        # asset_fee=0.25,
        # lrna_fee=0.25,
        preferred_stablecoin='USD'
    )
    print(f"asset fee: {initial_omnipool.asset_fee('HDX') * 100}%, lrna fee: {initial_omnipool.lrna_fee('HDX') * 100}%")

    trade_agent = Agent(enforce_holdings=False)
    initial_hdx_price = initial_omnipool.lrna_price('HDX')

    low_price = 0.98
    high_price = 1.02
    steps = 30
    buys = [1 - (1 - low_price) / 1.5 ** i for i in range(steps // 2)]
    sells = [1 + (high_price - 1) / 1.5 ** i for i in range(steps // 2)]
    prices = buys + [1] + sells[::-1]

    for price in prices:
        trade_agent.holdings = {}
        omnipool = initial_omnipool.copy()
        omnipool.liquidity['HDX'] *= (1 - lp_liquidity_pct)
        omnipool.lrna['HDX'] *= (1 - lp_liquidity_pct)
        omnipool.shares['HDX'] *= (1 - lp_liquidity_pct)
        pool_agent = Agent(
            enforce_holdings=False,
            holdings={'HDX': omnipool.liquidity['HDX'] * lp_liquidity_pct / (1 - lp_liquidity_pct)}
        )
        omnipool.add_liquidity(
            agent=pool_agent,
            tkn_add='HDX',
            quantity=pool_agent.get_holdings('HDX')
        )
        pool_agent.holdings['HDX'] = 0
        initial_value = omnipool.cash_out(pool_agent)
        print()
        print(f"price change: {round((price - 1) * 100, 3)}%")
        print(f"Initial agent holdings: {pool_agent.all_holdings()} ({round(pool_agent.get_holdings(('omnipool', 'HDX')) / omnipool.shares['HDX'] * 100, 3)}% of pool)")
        print(f"Step 1: trader trades to new price of {round(price, 6)}")
        trade_to_price(omnipool, trade_agent, 'HDX', price)
        price_change = (omnipool.lrna_price('HDX') - initial_hdx_price) / initial_hdx_price
        # remove and re-add liquidity
        print(f"Step 2: LP removes and re-adds liquidity")
        remove_readd(omnipool, pool_agent)
        print(f"Agent has {pool_agent.all_holdings()} ({round(pool_agent.get_holdings(('omnipool', 'HDX')) / omnipool.shares['HDX'] * 100, 3)}% of pool)")
        # print(f"Agent removes liquidity: {pool_agent.all_holdings()}")
        # omnipool.remove_liquidity(
        #     agent=pool_agent,
        #     tkn_remove='HDX',
        #     quantity=pool_agent.get_holdings(('omnipool', 'HDX'))
        # )
        # print(f"Agent has {pool_agent.all_holdings()}")
        # continue
        intermediate_value_1 = omnipool.cash_out(pool_agent, prices={'HDX': initial_hdx_price})
        intermediate_loss_1 = intermediate_value_1 - initial_value
        intermediate_holdings_1 = copy.copy(pool_agent.holdings)
        # swap back
        print("Step 3: trader swaps back to initial price")
        trade_to_price(omnipool, trade_agent,'HDX', initial_hdx_price)
        # fee_gained_1 = omnipool.cash_out(pool_agent) - intermediate_value_1

        remove_readd(omnipool, pool_agent)
        print(f"Step 4: LP removes and re-adds liquidity, new holdings: {pool_agent.all_holdings()} ({round(pool_agent.get_holdings(('omnipool', 'HDX')) / omnipool.shares['HDX'] * 100, 3)}% of pool)")
        intermediate_value_2 = omnipool.cash_out(pool_agent, prices={'HDX': initial_hdx_price})
        # intermediate_loss_2 = intermediate_value_2 - intermediate_value_1
        # intermediate_holdings_2 = copy.copy(pool_agent.holdings)
        trade_to_price(omnipool, trade_agent, 'HDX', initial_hdx_price)
        final_value = omnipool.cash_out(pool_agent)
        loss = final_value - pool_agent.initial_holdings['HDX'] * omnipool.lrna_price('HDX')
        # fee_gained_2 = final_value - intermediate_value_2
        if omnipool.lrna_price('HDX') != pytest.approx(initial_hdx_price, rel=1e-12):
            print("ERROR: final price not equal to initial price")
            print(f"Initial value: {initial_value}, final value: {final_value}")

        print(f"price change {round(price_change * 100, 3)}%, loss {round(loss, 7)} HDX ({round(loss / initial_value * 100, 7)}%)")


def scenario_2():
    col1, col2 = st.columns([1, 8])
    with col1:
        low_price = st.number_input(
            label="low price",
            min_value=0.01,
            max_value=1.0,
            value=0.98
        )
        high_price = st.number_input(
            label="high price",
            min_value=1.0,
            max_value=100.0,
            value=1.02
        )
        default_asset_fee = st.number_input(
            label="asset fee %",
            min_value=0.0,
            max_value=50.0,
            value=0.25,
        ) / 100

    # default_asset_fee = 0.25
    default_lrna_fee = 0.25
    lp_share_pct = 0.25
    omnipool = OmnipoolState(
        tokens={"HDX": {'liquidity': 1_000_000 * (1 - lp_share_pct), 'LRNA': 1_000_000 * (1 - lp_share_pct)},
                "USDT": {'liquidity': 1_000_000, 'LRNA': 1_000_000},
                "DOT": {'liquidity': 1_000_000, 'LRNA': 1_000_000}
        },
        asset_fee=default_asset_fee,
        lrna_fee=default_lrna_fee,
        withdrawal_fee=False
    )
    init_agent = Agent(holdings={"HDX": 1_000_000 * lp_share_pct})
    omnipool.add_liquidity(
        agent=init_agent,
        tkn_add='HDX',
        quantity=init_agent.holdings['HDX']
    )

    # low_price = 0.5
    # high_price = 2
    steps = 50
    buys = [1 - (1 - low_price) / 1.5 ** i for i in range(steps // 2)]
    sells = [1 + (high_price - 1) / 1.5 ** i for i in range(steps // 2)]
    prices = buys + [1] + sells[::-1]
    fee_gained_trend = {}
    fee_paid_trend = {}
    net_fee_trend = {}
    il_trend = {}

    for price in prices:
        omnipool1 = omnipool.copy()  # with fees
        omnipool2 = omnipool.copy()  # no fees
        omnipool2.asset_fee = 0
        omnipool2.lrna_fee = 0
        omnipool3 = omnipool.copy()  # fees for trader but not LP
        omnipool4 = omnipool.copy()  # no remove/readd
        temp_pool: OmnipoolState = omnipool.copy()

        agent1 = init_agent.copy()
        agent2 = init_agent.copy()
        agent3 = init_agent.copy()
        agent4 = init_agent.copy()
        trader = Agent(enforce_holdings=False)
        temp_agent = agent1

        print()
        print(f"price change: {round((price - 1) * 100, 3)}%")
        print("--------------------------------")

        for i, (pool, agent) in enumerate([
            (omnipool1, agent1), (omnipool2, agent2), (omnipool3, agent3), (omnipool4, agent4)
        ]):
            trade_size = trade_to_price(pool, trader, tkn='HDX', target_price=price)

            if i == 0:
                # save a snapshot
                temp_pool = pool.copy()
                temp_agent = agent.copy()
            elif i == 1 and price > 1:
                # measure fee gained by LP from trader
                assert trade_size < 0
                temp_pool.asset_fee = 0
                fee_gained_trend[price] = temp_pool.cash_out(temp_agent) - pool.cash_out(agent)
            elif i == 2:
                # no fee for LP
                pool.asset_fee = 0
            elif i == 3:
                # no remove/readd
                continue

            remove_readd(pool, agent)

            # if i == 2:
                # pool.asset_fee = default_asset_fee

            trade_to_price(pool, trader, tkn='HDX', target_price=price)
            print('--')

        value_with_fee = omnipool1.cash_out(agent1, denomination='LRNA')
        value_no_fee = omnipool2.cash_out(agent2, denomination='LRNA')
        fee_net = value_with_fee - value_no_fee
        net_fee_trend[price] = fee_net
        print(f"omnipool price with fee: {omnipool1.lrna_price('HDX')}, no fee: {omnipool2.lrna_price('HDX')}")
        print(f"agent cash out with fee: {value_with_fee}, no fee: {value_no_fee}, net = {fee_net}")
        if price in fee_gained_trend:
            print(f"fee gained by LP from trader: {fee_gained_trend[price]}")
        else:
            fee_gained_trend[price] = 0
        fee_losses = omnipool3.cash_out(agent3, denomination='LRNA') - value_with_fee
        # print(omnipool3.cash_out(agent3, denomination='LRNA'))
        print(f"agent loses {fee_losses} in fees by selling LRNA.")
        fee_paid_trend[price] = fee_losses
        il_trend[price] = omnipool4.value_assets(init_agent.initial_holdings, numeraire='LRNA') - value_no_fee

    import numpy as np
    ticks = [low_price]
    updown_ratio = np.log10(high_price) / (np.log10(high_price) - np.log10(low_price))
    total_ticks = 8
    high_ticks = round(total_ticks * updown_ratio)
    low_ticks = total_ticks - high_ticks
    print(f"low ticks: {low_ticks}, high ticks: {high_ticks}")
    ticks += list(np.logspace(np.log10(low_price), np.log10(1), num=low_ticks + 1)[1:-1])
    ticks.append(1)
    ticks += list(np.logspace(np.log10(1), np.log10(high_price), num=high_ticks + 1)[1:-1])
    ticks.append(high_price)
    ticks = [round(t, 3) for t in ticks]
    ticklabels = [f"{round(t, 3)}" if t != 1 else "1.0" for t in ticks]

    with col2:
        fig1, ax1 = plt.subplots(figsize=(16, 7))

        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.3f}"))
        ax1.plot(list(fee_gained_trend.keys()), list(fee_gained_trend.values()), label='fee gained from trader')
        ax1.plot(
            list(fee_paid_trend.keys()),
            list(fee_paid_trend.values()),
            label='fee paid by LP',
            color='orange'
        )
        ax1.set_xscale('log')
        ax1.set_xticks(ticks)
        ax1.set_xticklabels([f"{t:.3f}" for t in ticks])
        ax1.xaxis.set_major_locator(FixedLocator(ticks))
        ax1.xaxis.set_major_formatter(FixedFormatter(ticklabels))
        ax1.xaxis.set_minor_locator(plt.NullLocator())

        ax1.legend(loc="upper center")
        st.pyplot(fig1)


        fig2, ax2 = plt.subplots(figsize=(16, 7))
        ax2.plot(list(net_fee_trend.keys()), list(net_fee_trend.values()), label='net fees', color='red')
        ax2.plot(
            list(il_trend.keys()),
            list(il_trend.values()),
            label='IL losses, in H2O',
            color='purple'
        )
        ax2.set_xscale('log')
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f"{t:.3f}" for t in ticks])
        ax2.xaxis.set_major_locator(FixedLocator(ticks))
        ax2.xaxis.set_major_formatter(FixedFormatter(ticklabels))
        ax2.xaxis.set_minor_locator(plt.NullLocator())

        ax2.legend(loc="upper center")
        st.pyplot(fig2)


def scenario_3():

    with st.sidebar:
        low_price = st.number_input('low price:', min_value=0.01, max_value=1.0, value=0.98)
        high_price = st.number_input('high price:', min_value=1.0, max_value=100.0, value=1.02)
        max_lp_holdings = st.number_input('max lp holdings %:', min_value=10, max_value=99, value=50)
    steps = 50
    buys = linear_increasing_sequence(start_value=1, end_value=low_price, sequence_length=steps // 2)[:1:-1]
    sells = linear_increasing_sequence(start_value=1, end_value=high_price, sequence_length=steps // 2)[1:]
    prices = buys + [1] + sells
    loss_trend = {}

    initial_omnipool = OmnipoolState(
        tokens={"HDX": {'liquidity': 1_000_000, 'LRNA': 1_000_000},
                "USDT": {'liquidity': 1_000_000, 'LRNA': 1_000_000},
                "DOT": {'liquidity': 1_000_000, 'LRNA': 1_000_000}
        },
        asset_fee=0.25,
        lrna_fee=0.25,
        withdrawal_fee=False
    )
    trade_agent = Agent(enforce_holdings=False)
    lp_percent_array = [0.01 * i for i in range(1, max_lp_holdings + 1)]
    for lp_percent in lp_percent_array:
        loss_trend[lp_percent] = []
        for price in prices:
            omnipool = initial_omnipool.copy()
            lp_agent = Agent(holdings={"HDX": omnipool.liquidity['HDX'] * lp_percent})
            omnipool.liquidity['HDX'] *= (1 - lp_percent)
            omnipool.lrna['HDX'] *= (1 - lp_percent)
            omnipool.add_liquidity(
                agent=lp_agent,
                tkn_add='HDX',
                quantity=lp_agent.holdings['HDX']
            )
            assert omnipool.lrna['HDX'] == pytest.approx(1_000_000, rel=1e-12)
            assert omnipool.liquidity['HDX'] == pytest.approx(1_000_000, rel=1e-12)
            initial_value = omnipool.cash_out(lp_agent, denomination='LRNA')
            omnipool.trade_to_price(agent=trade_agent, tkn='HDX', target_price=price)
            remove_readd(omnipool, lp_agent)
            omnipool.trade_to_price(agent=trade_agent, tkn='HDX', target_price=1)
            remove_readd(omnipool, lp_agent)
            omnipool.trade_to_price(agent=trade_agent, tkn='HDX', target_price=1)
            # remove_readd(omnipool, lp_agent)
            if omnipool.lrna_price('HDX') != pytest.approx(1, rel=1e-12):
                raise ValueError(f"hdx price == {omnipool.lrna_price('HDX')}, not 1")
            final_value = omnipool.cash_out(lp_agent, denomination='LRNA')
            loss = (initial_value - final_value) / lp_agent.initial_holdings['HDX']
            loss_trend[lp_percent].append(loss)

    fig, ax = plt.subplots(figsize=(16, 7))
    colors = color_gradient(length = len(lp_percent_array), color1=(255, 0, 198), color2=(0, 0, 192))
    for i, lp_percent in enumerate(lp_percent_array):
        ax.plot(
            prices, loss_trend[lp_percent],
            label=f"LP pct: {round(lp_percent * 100, 1)}%" if lp_percent in [0.01, 0.25, 0.5, 0.75, 0.99] else None,
            color=colors[i]
        )
    ax.set_xlabel('price change')
    ax.set_ylabel('LP loss as a percent of deposit')
    ax.legend()
    st.pyplot(fig)

def scenario_4(block_number=None):
    max_block_number = 9300000  # get_current_block_height() - 1000
    block_number = st.number_input(
        'block number:', min_value=1, max_value=max_block_number, value=block_number or 9_000_000
    )
    trades = get_omnipool_trades(
        min_block=block_number, max_block=block_number + 1000
    )
    pass
    initial_omnipool = get_current_omnipool(block_number)
    trade_agent_1 = Agent(enforce_holdings=False, unique_id="trader1")
    trade_agent_2 = Agent(enforce_holdings=False, unique_id="trader2")
    initial_lp_agent = Agent(holdings={"HDX": initial_omnipool.liquidity['HDX'] / 4})
    initial_omnipool.liquidity["HDX"] *= 3/4
    initial_omnipool.lrna["HDX"] *= 3/4
    initial_omnipool.add_liquidity(
        agent=initial_lp_agent,
        tkn_add='HDX',
        quantity=initial_lp_agent.holdings['HDX']
    )
    lp_agent_1 = initial_lp_agent.copy()  # agent now holds 25% of the pool
    lp_agent_2 = initial_lp_agent.copy()
    lp_agent_1.unique_id = "lp1"
    lp_agent_2.unique_id = "lp2"
    omnipool1 = initial_omnipool.copy()
    omnipool2 = initial_omnipool.copy()
    events = []
    fails = []
    divergence = {pool.unique_id: {tkn: {} for tkn in omnipool1.liquidity.keys()} for pool in (omnipool1, omnipool2)}
    next_trade = trades.pop(0)
    for block in range(block_number, block_number + 1000):
        while next_trade and next_trade['block_number'] == block:
            trade = next_trade
            next_trade = trades.pop(0) if trades else None
            tkn_sell = trade['assetIn']
            tkn_buy = trade['assetOut']
            if tkn_buy == "H2O":
                tkn_buy = "LRNA"
            if tkn_sell == "H2O":
                tkn_sell = "LRNA"
            for omnipool, trade_agent in [(omnipool1, trade_agent_1), (omnipool2, trade_agent_2)]:
                omnipool.fail = ""
                agent_start_holdings = trade_agent.get_holdings(tkn_buy)
                omnipool.swap(
                    agent=trade_agent,
                    tkn_buy=tkn_buy,
                    tkn_sell=tkn_sell,
                    sell_quantity=trade['amountIn']
                )
                buy_quantity = trade_agent.get_holdings(tkn_buy) - agent_start_holdings
                divergence[omnipool.unique_id][tkn_buy][block] = abs(buy_quantity - trade['amountOut']) / trade['amountOut']
                if omnipool.fail:
                    fails.append((omnipool.unique_id, trade))

        remove_readd(omnipool1, lp_agent_1)
        if omnipool2.current_withdrawal_fee == omnipool2.min_withdrawal_fee:
            remove_readd(omnipool2, lp_agent_2)
        events.append({
            'pools': {pool.unique_id: pool.copy() for pool in (omnipool1, omnipool2)},
            'agents': {agent.unique_id: agent.copy() for agent in (lp_agent_1, lp_agent_2, trade_agent_1, trade_agent_2)},
            'block': block
        })
        pass
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot([e['block'] for e in events], [e['agents']['lp1'].get_holdings('HDX') for e in events], label='LP1 HDX')
    ax.plot([e['block'] for e in events], [e['agents']['lp2'].get_holdings('HDX') for e in events], label='LP1 HDX')
    ax.legend()
    st.pyplot(fig)
    fig2, ax2 = plt.subplots(figsize=(16, 7))
    for tkn in divergence:
        ax2.plot(list(divergence[tkn].keys()), list(divergence[tkn].values()), label=tkn)
    ax.legend()
    st.pyplot(fig2)
    pass
