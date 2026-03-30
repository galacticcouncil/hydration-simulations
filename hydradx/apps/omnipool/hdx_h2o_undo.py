import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.agents import Agent
from hydradx.model.indexer_utils import get_blocks_at_timestamps, get_omnipool_trades, get_date_of_block, \
    get_current_omnipool_router, get_asset_info_by_ids, get_current_block_height, get_current_omnipool, \
    get_hollar_liquidity_at, get_block_at_timestamp, query_indexer
from hydradx.model.processing import load_state, save_state
import datetime, json
from pathlib import Path
from hydradx.apps.omnipool.trade_downloader import get_trades_for_dates, save_trades_to_cache
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from hydradx.model.processing import query_sqlPad

st.set_page_config(layout="wide")

def simulate_trade_before_and_after():
    cache_directory = Path(__file__).parent / 'cached data' / 'omnipools'
    date = datetime.datetime(2026, 2, 1)
    omnipool_filename = f'omnipool-{date.isoformat()}.json'
    if Path.exists(cache_directory / omnipool_filename):
        omnipool_router = load_state(path=str(cache_directory), filename=omnipool_filename)
    else:
        omnipool_router = get_current_omnipool_router()
        save_state(omnipool_router, path=str(cache_directory), filename=omnipool_filename)


    lp = Agent()
    omnipool: OmnipoolState = omnipool_router.exchanges['omnipool']
    lp.holdings = {'LRNA': 1000}
    omnipool_router.swap(
        lp, tkn_sell='LRNA', tkn_buy='Hydrated Dollar', sell_quantity=lp.holdings['LRNA']
    )
    pass


def get_omnipools(start_date: datetime.datetime, end_date: datetime.datetime) -> dict[datetime.datetime, OmnipoolState]:
    cache_directory = Path(__file__).parent / 'cached data' / 'omnipools'
    omnipools = {}
    all_dates = [
        start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)
    ]
    for date in all_dates:
        block_number = get_block_at_timestamp(date)
        omnipool_filename = f'omnipool-{date.strftime('%Y-%m-%d')}.json'
        if Path.exists(cache_directory / omnipool_filename):
            omnipool_router = load_state(path=str(cache_directory), filename=omnipool_filename)
        else:
            omnipool_router = get_current_omnipool_router(block_number=block_number)
        omnipool: OmnipoolState = omnipool_router.exchanges['omnipool']
        if "HOLLAR" not in omnipool.liquidity:
            hollar_liquidity = get_hollar_liquidity_at(block_number)
            omnipool.liquidity['HOLLAR'] = hollar_liquidity['liquidity']
            omnipool.lrna['HOLLAR'] = hollar_liquidity['LRNA']
            omnipool.shares['HOLLAR'] = hollar_liquidity['shares']
            omnipool.protocol_shares['HOLLAR'] = hollar_liquidity['shares']

        omnipool.add_token(
            tkn='HOLLAR',
            liquidity=omnipool.liquidity['HOLLAR'],
            lrna=omnipool.lrna['HOLLAR'],
            shares=omnipool.shares['HOLLAR'],
            protocol_shares=omnipool.protocol_shares['HOLLAR'],
            weight_cap=1.0
        ).update()
        omnipool.time_step = block_number

        if not Path.exists(cache_directory / omnipool_filename):
            save_state(omnipool_router, path=str(cache_directory), filename=omnipool_filename)

        omnipools[date] = omnipool

    return omnipools

@st.cache_data(show_spinner=True)
def _build_hollar_trade_summary(start_date: datetime.datetime, end_date: datetime.datetime):
    trades = get_trades_for_dates(
        start_date=start_date,
        end_date=end_date,
        save_cache=True,
        cache_directory=Path(__file__).parent / 'cached data' / 'hollar trades',
        extra_filter='args: {includes: ":222,"}'
    )
    omnipools: dict[datetime.datetime, OmnipoolState] = get_omnipools(start_date, end_date)

    all_dates = [
        start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)
    ]

    # line them up by block numbers to account for any timezone difference
    for i in range(len(all_dates) - 1):
        date = all_dates[i]
        min_block = omnipools[date].time_step
        max_block = omnipools[all_dates[i + 1]].time_step
        trades_on_date = [
            trade for trade in trades
            if 'block_number' in trade and min_block < trade['block_number'] <= max_block
        ]
        for trade in trades_on_date:
            if trade['date'] != date.strftime('%Y-%m-%d'):
                trade['date'] = date.strftime('%Y-%m-%d')

    hollar_h2o_trades = [trade for trade in trades if "H2O" in trade.values()]
    hollar_per_day = {
        datetime.datetime.strptime(date, '%Y-%m-%d'): sum(
            [trade['amountOut'] for trade in hollar_h2o_trades if trade['date'] == date]
        )
        for date in set([trade['date'] for trade in hollar_h2o_trades])
    }
    hollar_per_day = dict(sorted(hollar_per_day.items(), key=lambda item: item[0]))

    hollar_percentage_per_day = {
        date: hollar_per_day[date] / omnipools[date].liquidity['HOLLAR'] if date in hollar_per_day and date in omnipools else 0
        for date in set(list(hollar_per_day.keys()) + list(omnipools.keys()))
    }
    hollar_percentage_per_day = dict(sorted(hollar_percentage_per_day.items(), key=lambda item: item[0]))

    return all_dates, omnipools, hollar_percentage_per_day


def find_hollar_trades():
    start_date = datetime.datetime(2026, 2, 16)
    end_date = datetime.datetime(2026, 3, 18)  # datetime.datetime.today() - datetime.timedelta(days=1)

    all_dates, omnipools, hollar_percentage_per_day = _build_hollar_trade_summary(start_date, end_date)

    selected_range = st.select_slider(
        "Date range",
        options=all_dates,
        value=(all_dates[0], all_dates[-1]),
        format_func=lambda d: d.strftime('%Y-%m-%d')
    )
    lp_shares = st.number_input("LP shares", min_value=0.0, value=1000.0, step=1.0)

    percentage_series = [hollar_percentage_per_day.get(date, 0) for date in all_dates]
    slider_start_idx = all_dates.index(selected_range[0])
    slider_end_idx = all_dates.index(selected_range[1])
    lp_losses = (
        sum(percentage_series[slider_start_idx: slider_end_idx])
        * lp_shares
        / omnipools[all_dates[slider_start_idx]].shares['HOLLAR']
        * omnipools[all_dates[slider_start_idx]].liquidity['HOLLAR']
    )
    st.metric("LP losses in dollars", f"{lp_losses:,.2f}")

    fig, ax = plt.subplots(figsize=(6.4, 3.36))
    ax.plot(list(hollar_percentage_per_day.keys()), list(hollar_percentage_per_day.values()))
    ax.set_title("Percentage of HOLLAR liquidity sold for H2O per day", fontsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    st.pyplot(fig)
    return
    pass

    # for i, trade in enumerate(hollar_h2o_trades):
    #     if trade['who'] == '0x6d6f646c726f7574657265780000000000000000000000000000000000000000':
    #         for j in range(len(hollar_trades)):
    #             other_trade = hollar_trades[j]
    #             if other_trade['block_number'] < trade['block_number']:
    #                 continue
    #             if other_trade['who'] != '0x6d6f646c726f7574657265780000000000000000000000000000000000000000':
    #                 continue
    #             if other_trade['assetIn'] == 'HOLLAR':
    #                 if other_trade['who'] == '0x6d6f646c726f7574657265780000000000000000000000000000000000000000':
    #                     if abs(float(other_trade['amountIn']) - float(trade['amountOut'])) / trade['amountIn'] < 0.01:
    #                         trade['match'] = j
    #                         trade['separation'] = other_trade['block_number'] - trade['block_number']
    #                         break

    #
    # with open (Path(__file__).parent / 'cached data' / 'sqlpad_hollar_h2o.json', 'r') as f:
    #     sqlpad_data = json.load(f)
    #
    # sum([float(sqlpad_data[i]['amount_in']) if sqlpad_data[i]['asset_out_symbol'] == "HOLLAR" and sqlpad_data[i]["amount_in"] != sqlpad_data[i + 1]["amount_in"] else 0 for i in range(len(sqlpad_data) - 1)])
    #
    # matches = 0
    # for trade in hollar_h2o_trades:
    #     amount = trade['amountIn']
    #     for i, other_trade in enumerate(sqlpad_data):
    #         if abs(float(other_trade['amount_in']) - amount) < 0.000001:
    #             trade['match'] = i
    #             matches += 1
    #             break
    #     if 'match' not in trade:
    #         trade['match'] = None

    plt.rcParams.update({'font.size': 4})
    fig, ax = plt.subplots()
    cumulative_hollar_sold = [sum([hollar_per_day[day] for day in sim_days if day <= date]) for date in sim_days]
    ax.plot(list(hollar_per_day.keys()), cumulative_hollar_sold)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.title("Cumulative HOLLAR Bought with H2O")
    plt.show()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(sim_days, [omnipools[date].liquidity['HOLLAR'] for date in sim_days])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.title("Hollar liquidity in Omnipool")
    plt.show()
    st.pyplot(fig)

    starting_omnipool = omnipools[sim_days[0]]
    lp = Agent(holdings={'HOLLAR': 1000})
    lp_loss_estimate = lp.holdings['HOLLAR'] * sum([hollar_per_day[day] for day in sim_days]) / starting_omnipool.liquidity['HOLLAR'] / 2

    omnipool = starting_omnipool.copy()
    omnipool.add_liquidity(lp, tkn_add="HOLLAR", quantity=1000)
    lp_shares = lp.get_holdings(('omnipool', 'HOLLAR'))
    lp_share_percentage = lp_shares / starting_omnipool.shares['HOLLAR']
    trader = Agent()
    hollar_unaccounted_for = 0
    hollar_unaccounted_for_per_day = {}
    hollar_price_per_day = {}
    for day in sim_days:
        hollar_liquidity = omnipools[day].liquidity['HOLLAR']
        hollar_lrna = omnipools[day].lrna['HOLLAR']
        # omnipool.liquidity['HOLLAR'] = hollar_liquidity
        # omnipool.lrna['HOLLAR'] = hollar_lrna
        hollar_unaccounted_for_per_day[day] = omnipool.liquidity['HOLLAR'] - omnipools[day].liquidity['HOLLAR'] - hollar_unaccounted_for
        hollar_unaccounted_for += hollar_unaccounted_for_per_day[day]
        hollar_price_today = hollar_liquidity / omnipools[day].lrna['HOLLAR']
        hollar_price_per_day[day] = hollar_price_today
        hollar_bought_today = hollar_per_day[day]
        omnipool.swap(
            trader, tkn_sell='LRNA', tkn_buy='HOLLAR', buy_quantity=hollar_bought_today
        )
        omnipool.lrna['HOLLAR'] -= hollar_bought_today / hollar_price_today
        omnipool.swap(
            trader, tkn_sell='HOLLAR', tkn_buy='LRNA', sell_quantity=hollar_bought_today / 2
        )
        final_price = omnipool.liquidity['HOLLAR'] / omnipool.lrna['HOLLAR']
        pass
    omnipool.remove_liquidity(lp, quantity=lp.get_holdings(('omnipool', 'HOLLAR')), tkn_remove='HOLLAR')
    final_holdings = lp.get_holdings('HOLLAR')
    pass


def simulate_lp_experience():
    start_date = datetime.datetime(2026, 2, 16)
    end_date = datetime.datetime.today() - datetime.timedelta(days=1)

    trades = get_trades_for_dates(
        start_date=start_date,
        end_date=end_date,
        save_cache=True,
        cache_directory=Path(__file__).parent / 'cached data' / 'hollar trades',
        extra_filter='args: {includes: ":222,"}'
    )

    trade_types = set([trade['name'] for trade in trades])
    trades_by_type = {
        trade_type: [trade for trade in trades if trade['name'] == trade_type]
        for trade_type in trade_types
    }
    trades_by_type['H2O Sells'] = [trade for trade in trades if 'H2O' in trade.values()]
    trades_by_type['Hollar Out'] = [
        trade for trade in trades
        if (trade['name'] == 'Omnipool.SellExecuted' or trade['name'] == 'Omnipool.BuyExecuted')
        and trade['assetOut'] == 'HOLLAR'
    ]
    trades_by_type['Hollar In'] = [
        trade for trade in trades
        if (trade['name'] == 'Omnipool.SellExecuted' or trade['name'] == 'Omnipool.BuyExecuted')
        and trade['assetIn'] == 'HOLLAR'
    ]

    relevant_quantity = {
        'Omnipool.LiquidityAdded': 'amount',
        'Omnipool.LiquidityRemoved': 'sharesRemoved',
        'H2O Sells': 'amountOut',
        'Omnipool.PositionCreated': 'shares',
        'Hollar Out': 'amountOut',
        'Hollar In': 'amountIn'
    }
    quantities_by_type = {
        trade_type: sum([float(trade[relevant_quantity[trade_type]]) for trade in trades_by_type[trade_type]])
        for trade_type in relevant_quantity
    }
    quantities_by_type = {
        trade_type: quantity / 10 ** 18 if quantity > 10 ** 18 else quantity
        for trade_type, quantity in quantities_by_type.items()
    }

    omnipools = list(get_omnipools(start_date, end_date).values())
    starting_omnipool = omnipools[0].copy()
    starting_omnipool.withdrawal_fee = False
    starting_omnipool.max_withdrawal_per_block = float('inf')
    starting_omnipool.max_lp_per_block = float('inf')
    lp = Agent(holdings={'HOLLAR': 1000})
    trade_agent = Agent()
    starting_omnipool.add_liquidity(lp, tkn_add="HOLLAR", quantity=lp.get_holdings('HOLLAR'))
    actual_omnipool = starting_omnipool.copy()
    alternate_omnipool = starting_omnipool.copy()
    lp_shares = lp.get_holdings(('omnipool', 'HOLLAR'))
    trades = sorted(trades, key=lambda trade: trade['date'])
    for omnipool in [actual_omnipool, alternate_omnipool]:
        for trade in trades:
            if trade['name'] == 'Omnipool.SellExecuted' or trade['name'] == 'Omnipool.BuyExecuted':

                if trade['assetIn'] == 'H2O':
                    assetIn = "LRNA"
                else:
                    assetIn = trade['assetIn']
                assetOut = trade['assetOut']

                if assetIn not in omnipool.asset_list:
                    if assetOut != 'HOLLAR':
                        continue
                    omnipool.liquidity['HOLLAR'] -= trade['amountOut']
                    omnipool.lrna['HOLLAR'] += trade['hubAmountIn']
                    continue

                elif assetOut not in omnipool.asset_list:
                    if assetIn != 'HOLLAR':
                        continue
                    omnipool.liquidity['HOLLAR'] += trade['amountIn']
                    omnipool.lrna['HOLLAR'] -= trade['hubAmountOut']
                    continue

                omnipool.liquidity[assetOut] -= trade['amountOut']
                if assetIn == 'LRNA':
                    if omnipool == actual_omnipool:
                        omnipool.lrna['HDX'] += trade['amountIn']
                    else:
                        omnipool.lrna[assetOut] += trade['amountIn']
                else:
                    omnipool.liquidity[assetIn] += trade['amountIn']
                    omnipool.lrna[assetOut] += trade['hubAmountIn']
                    omnipool.lrna[assetIn] -= trade['hubAmountOut']

            elif trade['name'] == 'Omnipool.LiquidityAdded' or trade['name'] == 'Omnipool.PositionCreated':
                omnipool.add_liquidity(
                    trade_agent,
                    tkn_add='HOLLAR',
                    quantity=int(trade['amount']) / 10 ** 18
                )
            elif trade['name'] == 'Omnipool.LiquidityRemoved':
                omnipool.remove_liquidity(
                    trade_agent,
                    quantity=int(trade['sharesRemoved']) / 10 ** 18,
                    tkn_remove='HOLLAR'
                )

    lp1, lp2 = lp.copy(), lp.copy()
    actual_omnipool.remove_liquidity(lp1, quantity=lp1.get_holdings(('omnipool', 'HOLLAR')), tkn_remove='HOLLAR')
    alternate_omnipool.remove_liquidity(lp2, quantity=lp2.get_holdings(('omnipool', 'HOLLAR')), tkn_remove='HOLLAR')


    pass


if __name__ == "__main__":
    find_hollar_trades()

# question: how is a sell: H2O, buy: HOLLAR trade actually routed?
# question: is there a simpler way to determine the difference between an LP withdrawing and putting their H2O in the pool vs diverting it to HDX?
# note: to a HOLLAR LP who is withdrawing, there is actually no downside to H2O price falling.
# the situation where H2O price falls is strictly better for them.
# it sounds like trades which went through the router probably do count as selling H2O for Hollar. I will proceed with that.
# I would like to graph: Hollar going out of the pool, H2O going in over time, and the likely impact on LPs from that.
#
# methodology will be:
# the share of hollar pool owned by LP when the change went in.
# the percentage of HOLLAR in the pool that was sold for H2O.
# LP shares / total shares * percentage of HOLLAR sold / 2 = amount of HOLLAR LP lost out on
#
# I want to take one LP through the whole date range and see if my estimate for how much they lost
# matches up decently well with the simulation
#
# so far, no. I'm losing like half the liquidity along the way for some reason, and I don't really know where it went.
# I need to find out.
#
#
#
#
#
#
#
#
