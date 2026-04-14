import os, sys

from hydradx.model.amm.omnipool_amm import OmnipoolState

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.agents import Agent
from hydradx.model.indexer_utils import get_blocks_at_timestamps, get_omnipool_trades, get_dates_of_blocks, \
    get_current_omnipool_router
from hydradx.model.processing import load_state, save_state
import datetime, json
from pathlib import Path
import streamlit as st
from matplotlib import pyplot as plt


st.set_page_config(layout="wide")
print("App start")

def get_trades_for_dates(start_date: datetime.datetime, end_date: datetime.datetime):
    loaded_trades = []
    block_dates = [
        start_date
        + datetime.timedelta(days=i)
        for i in range((end_date - start_date).days + 1)
    ]
    dates_to_download = []
    for date in block_dates:
        date_str = date.strftime('%Y-%m-%d')
        cache_file = Path(__file__).parent / 'cached data' / 'trades' / f'trades_{date_str}.txt'
        if Path.exists(cache_file):
            cached_trades = json.load(open(cache_file, 'r'))
            loaded_trades.extend(cached_trades)
            # block_dates.remove(date)
        else:
            dates_to_download.append((date, date + datetime.timedelta(days=1)))

    for i in range(len(dates_to_download) - 1):
        first_block, last_block = list(get_blocks_at_timestamps([dates_to_download[i][0], dates_to_download[i][1]]).values())
        new_trades = get_omnipool_trades(
            min_block=first_block, max_block=last_block - 1
        )
        new_trades = [{**trade, 'date': dates_to_download[i][0].strftime('%Y-%m-%d')} for trade in new_trades]
        loaded_trades.extend(new_trades)
    return loaded_trades

def save_trades_to_cache(trades):
    dates = [trade['date'] for trade in trades]
    unique_dates = sorted(list(set(dates)))
    for date in unique_dates:
        savefile = Path(__file__).parent / 'cached data' / 'trades' / f'trades_{date}.txt'
        if not Path.exists(savefile):
            trades_on_date = [trade for trade in trades if trade['date'] == date]
            with open(savefile, 'w') as f:
                json.dump(trades_on_date, f)

# get one year's worth of trades
trades = get_trades_for_dates(
    start_date=datetime.datetime.today() - datetime.timedelta(days=365),
    end_date=datetime.datetime.today()
)
save_trades_to_cache(trades)

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
for trade in trades:
    if last_trade:
        if trade == last_trade:
            dupes.append(trade)
            dupe_dates.add(trade['date'])
            continue
    last_trade = trade
    if trade['assetIn'] == 'H2O':
        # send sold H2O to HDX reserves
        if not trade['who'] == temp_account:
            tkn_buy = trade['assetOut']
            if tkn_buy in omnipool.lrna:
                omnipool.lrna[tkn_buy] -= trade['amountIn']
            omnipool.lrna['HDX'] += trade['amountIn']
        else:
            temp_acct_trades.append(trade)
    else:
        # send protocol fee to HDX reserves
        omnipool.lrna['HDX'] += trade['protocolFeeAmount']

hdx_price_gain = omnipool_router.price('HDX', 'Tether') / old_router.price('HDX', 'Tether') - 1
lrna_price_gain = omnipool_router.price('LRNA', 'Tether') / old_router.price('LRNA', 'Tether') - 1

fig, ax = plt.subplots(figsize=(16, 6))
real_trades = [trade for trade in lrna_trades if trade['who'] != temp_account]
trades_by_date = sorted(real_trades, key=lambda trade: datetime.datetime.strptime(trade['date'], '%Y-%m-%d'))
date_strings = sorted(list(set([trade['date'] for trade in trades_by_date])))
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in date_strings]
sells = [sum([trade['amountIn'] for trade in trades_by_date if trade['date'] == date]) for date in date_strings]
ax.plot(dates, sells)
st.pyplot(fig)
pass
