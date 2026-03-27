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


def find_hollar_trades():
    start_date = datetime.datetime(2026, 2, 17)
    end_date = datetime.datetime(2026, 2, 24)
    trades = get_trades_for_dates(
        start_date=start_date,
        end_date=datetime.datetime.today(),
        extra_filter='args: {includes: "222,"}'
    )
    hollar_trades = [trade for trade in trades if "HOLLAR" in trade.values()]
    hollar_h2o_trades = [trade for trade in hollar_trades if "H2O" in trade.values()]
    shift_hrs = 6
    all_dates = [
        start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)
    ]
    for i in range(len(all_dates) - 1):
        date = all_dates[i]
        date_str = date.strftime('%Y-%m-%d')
        trades_on_date = [trade for trade in hollar_h2o_trades if trade['date'] == date_str]
        min_block = get_blocks_at_timestamps([date])[date]
        max_block = get_blocks_at_timestamps([all_dates[i + 1]])[all_dates[i + 1]]
        block_per_day = max_block - min_block
        min_block -= block_per_day * shift_hrs / 24
        max_block -= block_per_day * shift_hrs / 24
        for trade in trades_on_date:
            trade_time = (trade['block_number'] - min_block) / (max_block - min_block) * datetime.timedelta(hours=24) + date
            trade['date'] = trade_time.strftime('%Y-%m-%d')

    non_router_trades = [trade for trade in hollar_h2o_trades if
                         trade['who'] != '0x6d6f646c726f7574657265780000000000000000000000000000000000000000']
    h2o_per_day = {
        date: sum([trade['amountIn'] for trade in hollar_h2o_trades if trade['date'] == date])
        for date in set([trade['date'] for trade in hollar_h2o_trades])
    }
    h2o_per_day = dict(sorted(h2o_per_day.items(), key=lambda item: item[0]))

    hollar_per_day = {
        date: sum([trade['amountOut'] for trade in hollar_h2o_trades if trade['date'] == date])
        for date in set([trade['date'] for trade in hollar_h2o_trades])
    }
    hollar_per_day = dict(sorted(hollar_per_day.items(), key=lambda item: item[0]))

    hollar_withdraws = {}

    big_trades = sorted(non_router_trades, key=lambda trade: -trade['amountOut'])[:5]
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
    sim_days = list(hollar_per_day.keys())
    cumulative_hollar_sold = [sum([hollar_per_day[day] for day in sim_days if day <= date]) for date in sim_days]
    ax.plot(list(hollar_per_day.keys()), cumulative_hollar_sold)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.title("Cumulative HOLLAR Bought with H2O")
    plt.show()
    st.pyplot(fig)

    cache_directory = Path(__file__).parent / 'cached data' / 'omnipools'
    omnipools = {}
    for date in sim_days:
        block_number = 0
        omnipool_filename = f'omnipool-{date}.json'
        if Path.exists(cache_directory / omnipool_filename):
            omnipool_router = load_state(path=str(cache_directory), filename=omnipool_filename)
        else:
            block_number = get_block_at_timestamp(datetime.datetime.strptime(date, '%Y-%m-%d'))
            omnipool_router = get_current_omnipool_router(block_number=block_number)
        omnipool: OmnipoolState = omnipool_router.exchanges['omnipool']
        if "HOLLAR" not in omnipool.liquidity:
            if block_number == 0:
                block_number = get_block_at_timestamp(datetime.datetime.strptime(date, '%Y-%m-%d'))
            hollar_liquidity = get_hollar_liquidity_at(block_number)
            omnipool.liquidity['HOLLAR'] = hollar_liquidity['liquidity']
            omnipool.lrna['HOLLAR'] = hollar_liquidity['LRNA']
            omnipool.shares['HOLLAR'] = hollar_liquidity['shares']
            omnipool.protocol_shares['HOLLAR'] = hollar_liquidity['shares']
            save_state(omnipool_router, path=str(cache_directory), filename=omnipool_filename)

        omnipools[date] = omnipool

    fig, ax = plt.subplots()
    ax.plot(sim_days, [omnipools[date].liquidity['HOLLAR'] for date in sim_days])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.title("Hollar liquidity in Omnipool")
    plt.show()
    st.pyplot(fig)

    hollar_removes_by_day = {}
    all_blocks = get_blocks_at_timestamps(all_dates)
    if not(Path.exists(Path(__file__).parent / "cached data" / "hollar_withdraws.json")):
        for i in range(len(all_blocks) - 1):
            min_block = all_blocks[all_dates[i]]
            max_block = all_blocks[all_dates[i + 1]]
            url = "https://unified-main-aggr-indx.indexer.hydration.cloud/graphql"
            query = rf"""
                    query OmnipoolTransactionQuery($first: Int!) {{
                      events(
                        first: $first
                        orderBy: PARA_BLOCK_HEIGHT_ASC
                        filter: {{
                          name: {{ equalTo: "Stableswap.LiquidityRemoved" }}
                          paraBlockHeight: {{
                            greaterThanOrEqualTo: {min_block}
                            lessThanOrEqualTo: {max_block}
                          }}
                          args: {{ includes: "\"assetId\":222" }}
                        }}
                      ) {{
                        nodes {{
                          name
                          args
                          id
                          paraBlockHeight
                        }}
                        pageInfo {{
                          endCursor
                          hasNextPage
                        }}
                      }}
                    }}
                    """
            variables = {"first": 10000}
            result = query_indexer(url, query, variables)
            hollar_removes_by_day[all_dates[i]] = result['data']['events']['nodes']
            for event in hollar_removes_by_day[all_dates[i]]:
                event.update(json.loads(event['args']))
                event['amount'] = int(event['amounts'][0]['amount']) / 10 ** 18

            json.dump(
                {datetime.datetime.strftime(day, '%Y-%m-%d'): hollar_removes_by_day[day] for day in hollar_removes_by_day},
                open(Path(__file__).parent / "cached data" / "hollar_withdraws.json", 'w')
            )
    else:
        with open(Path(__file__).parent / "cached data" / "hollar_withdraws.json", 'r') as f:
            hollar_removes_by_day = json.load(f)
        hollar_removes_by_day = {datetime.datetime.strptime(day, '%Y-%m-%d'): hollar_removes_by_day[day] for day in hollar_removes_by_day}

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