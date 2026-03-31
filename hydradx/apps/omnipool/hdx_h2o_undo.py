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
import time
from pathlib import Path
from hydradx.apps.omnipool.trade_downloader import get_trades_for_dates, save_trades_to_cache
from hydradx.apps.s3_utils import download_file_from_s3, upload_file_to_s3, sync_dir_from_s3, upload_dir_to_s3, s3_join, get_s3_client, list_s3_keys
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from hydradx.model.processing import query_sqlPad

st.set_page_config(layout="wide")

CACHE_ROOT = Path(__file__).parent / "cached data"
S3_CACHE_PREFIX = "apps/omnipool/hdx_h2o_undo"


def _show_cloud_connected_banner() -> None:
    if st.session_state.get("cloud_connected_banner_shown"):
        return

    client, _cfg = get_s3_client()
    if client is None:
        return

    st.session_state["cloud_connected_banner_shown"] = True
    banner = st.empty()
    banner.markdown(
        """
        <div style="position: sticky; top: 0; z-index: 1000; background: #2e7d32; color: #ffffff; padding: 8px 12px; border-radius: 4px; text-align: center;">
            cloud cache connected
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(1)
    banner.empty()


_show_cloud_connected_banner()


def _trade_bucket() -> str | None:
    try:
        cfg = st.secrets.get("s3", {})
        return cfg.get("trade_bucket") or os.environ.get("S3_TRADE_BUCKET") or "trade-storage"
    except Exception:
        return os.environ.get("S3_TRADE_BUCKET") or "trade-storage"


def _omnipool_bucket() -> str | None:
    try:
        cfg = st.secrets.get("s3", {})
        return cfg.get("omnipool_bucket") or os.environ.get("S3_OMNIPOOL_BUCKET") or "omnipool-storage"
    except Exception:
        return os.environ.get("S3_OMNIPOOL_BUCKET") or "omnipool-storage"


def _cache_key_for(path: Path) -> str:
    rel_path = path.relative_to(CACHE_ROOT).as_posix()
    return s3_join(S3_CACHE_PREFIX, rel_path)


def _ensure_cached_file(path: Path) -> None:
    if path.exists():
        return
    download_file_from_s3(_cache_key_for(path), path)


def _upload_cached_file(path: Path) -> None:
    if not path.exists():
        return
    upload_file_to_s3(path, _cache_key_for(path))


def _sync_cache_dir(dir_path: Path) -> None:
    prefix = s3_join(S3_CACHE_PREFIX, dir_path.relative_to(CACHE_ROOT).as_posix())
    sync_dir_from_s3(dir_path, prefix)


def _upload_cache_dir(dir_path: Path) -> None:
    prefix = s3_join(S3_CACHE_PREFIX, dir_path.relative_to(CACHE_ROOT).as_posix())
    upload_dir_to_s3(dir_path, prefix)


def _omnipool_cache_paths_for_range(
    dir_path: Path,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> list[Path]:
    days = (end_date - start_date).days + 1
    return [
        dir_path / f"omnipool-{(start_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d')}.json"
        for i in range(days)
    ]


def _ensure_omnipool_cached_file(path: Path) -> None:
    if path.exists():
        return
    key = s3_join("omnipools", path.name)
    download_file_from_s3(key, path, bucket=_omnipool_bucket())


def _upload_omnipool_cached_file(path: Path) -> None:
    if not path.exists():
        return
    key = s3_join("omnipools", path.name)
    upload_file_to_s3(path, key, bucket=_omnipool_bucket())


def _sync_omnipool_cache_dir(
    dir_path: Path,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> None:
    for path in _omnipool_cache_paths_for_range(dir_path, start_date, end_date):
        _ensure_omnipool_cached_file(path)


def _upload_omnipool_cache_dir(
    dir_path: Path,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> None:
    for path in _omnipool_cache_paths_for_range(dir_path, start_date, end_date):
        _upload_omnipool_cached_file(path)


def _trade_cache_paths_for_range(
    dir_path: Path,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> list[Path]:
    days = (end_date - start_date).days + 1
    return [
        dir_path / f"trades_{(start_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d')}.txt"
        for i in range(days)
    ]


def _sync_trades_cache_dir(
    dir_path: Path,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    show_progress: bool = False,
) -> None:
    prefix = dir_path.relative_to(CACHE_ROOT).as_posix()
    trade_paths = _trade_cache_paths_for_range(dir_path, start_date, end_date)
    missing_paths = [path for path in trade_paths if not path.exists()]
    total = len(trade_paths)
    progress = st.progress(0, text="loading hollar trades") if show_progress and missing_paths else None

    for idx, path in enumerate(trade_paths, start=1):
        if not path.exists():
            key = s3_join(prefix, path.name)
            download_file_from_s3(key, path, bucket=_trade_bucket())
        if progress is not None:
            progress.progress(min(idx / total, 1.0), text=f"loading hollar trades ({idx}/{total})")

    if progress is not None:
        if total == 0:
            progress.empty()
        else:
            progress.progress(1.0, text=f"loading hollar trades ({total}/{total})")
            time.sleep(0.2)
            progress.empty()


def _upload_trades_cache_dir(
    dir_path: Path,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> None:
    if st.session_state.get("trade_upload_done"):
        return

    prefix = dir_path.relative_to(CACHE_ROOT).as_posix()
    existing_keys = set(list_s3_keys(prefix, bucket=_trade_bucket()))
    trade_paths = _trade_cache_paths_for_range(dir_path, start_date, end_date)
    for path in trade_paths:
        if not path.exists():
            continue
        key = s3_join(prefix, path.name)
        if key in existing_keys:
            continue
        upload_file_to_s3(path, key, bucket=_trade_bucket())

    st.session_state["trade_upload_done"] = True

def simulate_trade_before_and_after():
    cache_directory = CACHE_ROOT / "omnipools"
    date = datetime.datetime(2026, 2, 1)
    omnipool_filename = f'omnipool-{date.isoformat()}.json'
    _ensure_omnipool_cached_file(cache_directory / omnipool_filename)
    if Path.exists(cache_directory / omnipool_filename):
        omnipool_router = load_state(path=str(cache_directory), filename=omnipool_filename)
    else:
        omnipool_router = get_current_omnipool_router()
        save_state(omnipool_router, path=str(cache_directory), filename=omnipool_filename)
        _upload_omnipool_cached_file(cache_directory / omnipool_filename)


    lp = Agent()
    omnipool: OmnipoolState = omnipool_router.exchanges['omnipool']
    lp.holdings = {'LRNA': 1000}
    omnipool_router.swap(
        lp, tkn_sell='LRNA', tkn_buy='Hydrated Dollar', sell_quantity=lp.holdings['LRNA']
    )
    pass


def get_omnipools(start_date: datetime.datetime, end_date: datetime.datetime) -> dict[datetime.datetime, OmnipoolState]:
    cache_directory = CACHE_ROOT / "omnipools"
    _sync_omnipool_cache_dir(cache_directory, start_date, end_date)
    omnipools = {}
    all_dates = [
        start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)
    ]
    for date in all_dates:
        block_number = get_block_at_timestamp(date)
        omnipool_filename = f"omnipool-{date.strftime('%Y-%m-%d')}.json"
        _ensure_omnipool_cached_file(cache_directory / omnipool_filename)
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
            _upload_omnipool_cached_file(cache_directory / omnipool_filename)

        omnipools[date] = omnipool

    return omnipools

@st.cache_data(show_spinner="loading hollar trades")
def _build_hollar_trade_summary(start_date: datetime.datetime, end_date: datetime.datetime):
    trades = get_trades_for_dates(
        start_date=start_date,
        end_date=end_date,
        save_cache=True,
        cache_directory=CACHE_ROOT / "hollar trades",
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

    trades_cache_dir = CACHE_ROOT / "hollar trades"
    range_key = f"{start_date.date()}_{end_date.date()}"
    if not st.session_state.get(f"trades_loaded_{range_key}"):
        _sync_trades_cache_dir(trades_cache_dir, start_date, end_date, show_progress=True)
        st.session_state[f"trades_loaded_{range_key}"] = True
    all_dates, omnipools, hollar_percentage_per_day = _build_hollar_trade_summary(start_date, end_date)
    _upload_trades_cache_dir(trades_cache_dir, start_date, end_date)

    selected_range = st.select_slider(
        "Date range",
        options=all_dates,
        value=(all_dates[0], all_dates[-1]),
        format_func=lambda d: d.strftime('%Y-%m-%d')
    )
    lp_shares = 1000.0
    input_col, text_col = st.columns([1, 4])
    with input_col:
        lp_shares = st.number_input(
            "LP shares",
            min_value=0.0,
            value=lp_shares,
            step=1.0,
            key="lp_shares_input",
            label_visibility="collapsed",
        )
    with text_col:
        st.markdown(
            f"LP shares in Hollar = ${st.session_state.get('lp_shares_input', lp_shares) / omnipools[selected_range[0]].shares['HOLLAR'] * omnipools[selected_range[0]].liquidity['HOLLAR']:.2f}"
        )

    percentage_series = [hollar_percentage_per_day.get(date, 0) for date in all_dates]
    slider_start_idx = all_dates.index(selected_range[0])
    slider_end_idx = all_dates.index(selected_range[1])
    # lp_losses = (
    #     sum(percentage_series[slider_start_idx: slider_end_idx])
    #     * lp_shares
    #     / omnipools[all_dates[slider_start_idx]].shares['HOLLAR']
    #     * omnipools[all_dates[slider_start_idx]].liquidity['HOLLAR']
    #     / 2
    # )
    lp_losses = 0
    lp_holdings_pct = 1
    for i in range(slider_start_idx, slider_end_idx):
        daily_loss = percentage_series[i]
        lp_holdings_pct *= (1 - daily_loss / 2)
    lp_losses = (1 - lp_holdings_pct) * lp_shares / omnipools[all_dates[slider_start_idx]].shares['HOLLAR'] * omnipools[all_dates[slider_start_idx]].liquidity['HOLLAR']
    st.metric("Estimated LP losses in dollars", f"{lp_losses:,.2f}")

    fig, ax = plt.subplots(figsize=(6.4, 3.36))
    ax.plot(list(hollar_percentage_per_day.keys()), [pct * 100 for pct in list(hollar_percentage_per_day.values())])
    ax.set_title("Percentage of HOLLAR liquidity sold for H2O per day", fontsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)
    st.pyplot(fig)
    return


def simulate_lp_experience():
    start_date = datetime.datetime(2026, 2, 16)
    end_date = datetime.datetime.today() - datetime.timedelta(days=1)
    trades_cache_dir = CACHE_ROOT / "hollar trades"
    _sync_trades_cache_dir(trades_cache_dir, start_date, end_date)
    trades = get_trades_for_dates(
        start_date=start_date,
        end_date=end_date,
        save_cache=True,
        cache_directory=trades_cache_dir,
        extra_filter='args: {includes: ":222,"}'
    )
    _upload_trades_cache_dir(trades_cache_dir, start_date, end_date)

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
# we look through each day when this change was active and see what percentage of the HOLLAR liquidity was sold for H2O specifically on that day.
# then we apply that percentage to the LP's share of the pool to see how much of their HOLLAR was effectively sold for H2O each day, and sum that up over the date range.
# I find that there aren't any impermanent losses avoided for the LPs, because when the price of H2O falls, the value of their withdrawals actually increases
# (They receive the same amount of Hollar when they withdraw plus more H2O.)
#
#
#
