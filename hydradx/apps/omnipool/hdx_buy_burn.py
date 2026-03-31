import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.indexer_utils import get_blocks_at_timestamps, get_omnipool_trades, get_dates_of_blocks, \
    get_current_omnipool_router, get_omnipool_liquidity_at_intervals, get_asset_info_by_ids
from hydradx.model.processing import load_state, save_state
from hydradx.model.amm.agents import Agent
import datetime, json
from pathlib import Path
import streamlit as st
from matplotlib import pyplot as plt


st.set_page_config(layout="wide")
print("App start")


cache_directory = Path(__file__).parent / 'cached data' / 'omnipools'
omnipool_filename = f'omnipool-{datetime.date.today().isoformat()}.json'
if Path.exists(cache_directory / omnipool_filename):
    omnipool_router = load_state(path=str(cache_directory), filename=omnipool_filename)
else:
    omnipool_router = get_current_omnipool_router()
    save_state(omnipool_router, path=str(cache_directory), filename=omnipool_filename)

old_router = omnipool_router.copy()

lrna_trades = [trade for trade in trades if trade['assetIn'] == 'H2O']
lrna_sellers = {
    seller: sum([trade['amountIn'] for trade in lrna_trades if trade['who'] == seller])
    for seller in set([trade['who'] for trade in lrna_trades])
}
lrna_sellers = dict(sorted(lrna_sellers.items(), key=lambda item: -item[1]))
temp_account = list(lrna_sellers.keys())[0]
last_trade = None
dupes = []
dupe_dates = set()
temp_acct_trades = []

omnipool = omnipool_router.exchanges['omnipool']
daily_balances = {}
if Path.exists(Path(__file__).parent / 'cached data' / 'omnipool_liquidity.json'):

    with open(Path(__file__).parent / 'cached data' / 'omnipool_liquidity.json', 'r') as f:
        daily_balances = json.load(f)
    daily_balances = {int(block): daily_balances[block] for block in daily_balances}  # convert keys back to int
else:
    asset_info = get_asset_info_by_ids()
    asset_ids = [asset_info[asset].id for asset in asset_info]
    start_time = datetime.datetime.today() - datetime.timedelta(days=366)
    end_time = datetime.datetime.today()
    daily_balances = {}
    blocks = []
    for asset_id in asset_ids:
        asset_name = asset_info[asset_id].unique_id
        asset_balances = get_omnipool_liquidity_at_intervals(
            start_time=start_time,
            end_time=end_time,
            interval=datetime.timedelta(days=1),
            asset_ids=[asset_id]
        )
        if blocks == []:
            blocks = sorted(list(asset_balances.keys()))

        for block in blocks:
            if block not in daily_balances:
                daily_balances[block] = {}
                daily_balances[block]['time'] = asset_balances[block]['time']
            daily_balances[block][asset_name] = {
                'liquidity': asset_balances[block][asset_name]['liquidity'],
                'lrna': asset_balances[block][asset_name]['LRNA']
            }
    daily_balances = {int(block): daily_balances[block] for block in daily_balances}
    with open(Path(__file__).parent / 'cached data' / 'omnipool_liquidity.json', 'w') as f:
        json.dump(daily_balances, f, indent=4)

treasury_agent = Agent()
block_per_day = list(daily_balances.keys())
price_gain_overall = 1
trade_index = 0
for todays_block in block_per_day:
    hdx_bought_back = 0
    lrna_sold_back = 0

    current_hdx_balance = daily_balances[todays_block]['HDX']['liquidity']
    current_hdx_lrna = daily_balances[todays_block]['HDX']['lrna']
    hdx_lrna_price = current_hdx_lrna / current_hdx_balance

    while trade_index < len(trades) and trades[trade_index]['block_number'] < todays_block:
        trade = trades[trade_index]
        trade_index += 1

        if last_trade:
            if trade == last_trade:
                dupes.append(trade)
                dupe_dates.add(trade['date'])
                continue
        last_trade = trade
        # half of fees go to HDX buyback
        if trade['protocolFeeAmount'] > 0:
            lrna_sold_back += trade['protocolFeeAmount'] * 0.5
            hdx_bought_back += trade['protocolFeeAmount'] * 0.5 / hdx_lrna_price
        if trade['assetFeeAmount'] > 0:
            if (daily_balances[todays_block][trade['assetOut']]['liquidity']) == 0:
                if trade['assetIn'] == 'H2O':
                    asset_lrna_price = trade['amountIn'] / trade['amountOut']
                elif daily_balances[todays_block][trade['assetIn']]['lrna'] > 0:
                    other_asset_lrna_price = daily_balances[todays_block][trade['assetIn']]['lrna'] /\
                                       daily_balances[todays_block][trade['assetIn']]['liquidity']
                    asset_lrna_price = other_asset_lrna_price * trade['amountIn'] / trade['amountOut']
                elif 'hubAmountOut' in trade and trade['hubAmountOut'] > 0:
                    asset_lrna_price = trade['hubAmountOut'] / trade['amountOut']
                else:
                    continue
            else:
                asset_lrna_price = daily_balances[todays_block][trade['assetOut']]['lrna'] /\
                                   daily_balances[todays_block][trade['assetOut']]['liquidity']
            if asset_lrna_price < 0:
                pass
            if trade['assetFeeAmount'] < 0:
                pass
            lrna_this_trade = trade['assetFeeAmount'] * 0.5 * asset_lrna_price / price_gain_overall
            hdx_this_trade = trade['assetFeeAmount'] * 0.5 * asset_lrna_price / hdx_lrna_price / price_gain_overall
            effective_protocol_fee = lrna_this_trade / (trade['hubAmountIn'] if 'hubAmountIn' in trade and trade['hubAmountIn'] > 0 else trade['amountIn'])
            effective_asset_fee = trade['assetFeeAmount'] / trade['amountOut']
            if effective_asset_fee > 0.005 or effective_protocol_fee > 0.005:
                pass
            if effective_asset_fee < 0.0025 or effective_protocol_fee < 0.0005:
                pass

            if ((current_hdx_lrna + lrna_this_trade) / (current_hdx_balance - hdx_this_trade)) / hdx_lrna_price < 0:
                pass
            if hdx_bought_back < 0:
                pass
            if lrna_sold_back < 0:
                pass
            if hdx_bought_back < 0:
                pass
            lrna_sold_back += lrna_this_trade
            hdx_bought_back += hdx_this_trade
    if lrna_sold_back > 0:
        # 1. Calculate the 'k' for the HDX sub-pool at the start of the day
        k = current_hdx_lrna * current_hdx_balance

        # 2. How much HDX is actually removed from the pool by this LRNA?
        # This accounts for slippage (the price moving as you buy)
        new_hdx_balance = k / (current_hdx_lrna + lrna_sold_back)
        hdx_actually_burned = current_hdx_balance - new_hdx_balance

        # 3. The new price is (New LRNA) / (New HDX Balance)
        new_price = (current_hdx_lrna + lrna_sold_back) / new_hdx_balance

        # 4. Today's gain relative to the starting price
        gains_today = new_price / hdx_lrna_price
        price_gain_overall *= gains_today

    # gains_today = ((current_hdx_lrna + lrna_sold_back) / (current_hdx_balance - hdx_bought_back)) / hdx_lrna_price
    # price_gain_overall *= gains_today

fig, ax = plt.subplots(figsize=(16, 6))
real_trades = [trade for trade in lrna_trades if trade['who'] != temp_account]
trades_by_date = sorted(real_trades, key=lambda trade: datetime.datetime.strptime(trade['date'], '%Y-%m-%d'))
date_strings = sorted(list(set([trade['date'] for trade in trades_by_date])))
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in date_strings]
sells = [sum([trade['amountIn'] for trade in trades_by_date if trade['date'] == date]) for date in date_strings]
ax.plot(dates, sells)
st.pyplot(fig)
pass
