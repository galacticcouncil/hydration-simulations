import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.indexer_utils import get_blocks_at_timestamps, get_omnipool_trades, get_all_trades
from pathlib import Path
import datetime
import json

def get_trades_for_dates(
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        save_cache=True,
        extra_filter: str | None = 'name: {startsWith: "Omnipool", endsWith: "Executed}'
):
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
        new_trades = get_all_trades(
            min_block=first_block, max_block=last_block - 1,
            extra_filter=extra_filter
        )
        new_trades = [{**trade, 'date': dates_to_download[i][0].strftime('%Y-%m-%d')} for trade in new_trades]
        loaded_trades.extend(new_trades)

    if save_cache:
        save_trades_to_cache(loaded_trades)
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

