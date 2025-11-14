import copy
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
import sys, os, math
import streamlit as st
import time

from streamlit import form_submit_button

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.omnipool_router import OmnipoolRouter
from hydradx.model.amm.money_market import MoneyMarket, MoneyMarketAsset, CDP
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.trade_strategies import liquidate_cdps, schedule_swaps, omnipool_arbitrage, sell_all
from hydradx.model.processing import get_current_money_market, save_money_market, load_money_market as load_money_market_from_file
from hydradx.model.amm.agents import Agent
from hydradx.model.indexer_utils import get_current_omnipool_router, get_current_block_height
from hydradx.apps.display_utils import get_distribution, one_line_markdown, bell_distribute
from hydradx.model.amm.fixed_price import FixedPriceExchange

st.markdown("""
    <style>
        .stNumberInput button {
            display: none;
        }
        section[data-testid="stSidebar"] { min-width: 380px !important; width: 380px !important; }
        section[data-testid="stSidebar"] > div { width: 380px !important; }
        
        section[data-testid="stSidebar"] div.stButton > button {
          padding: 0 6px !important;
          min-height: 0 !important;
          height: 22px !important;
          line-height: 1 !important;
          font-size: 14px !important;
          border-radius: 6px !important;
        }
        
        section[data-testid="stSidebar"] div.stButton { text-align: right; }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(layout="wide")
print("App start")

@st.cache_data(ttl=3600, show_spinner="Loading Omnipool data (cached for 1 hour)...")
def load_omnipool_router() -> tuple[OmnipoolRouter, str]:
    block_number = get_current_block_height() - 10000
    # Add timestamp to verify caching
    import datetime
    cache_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"Cache miss! Loading omnipool at {cache_time}")
    load_router = get_current_omnipool_router(block_number)
    load_omnipool = load_router.exchanges['omnipool']
    if block_number is None:
        block_number = load_omnipool.time_step
    stableswap_pools = [
        pool for pool in load_router.exchanges.values()
        if isinstance(pool, StableSwapPoolState)
        and min(pool.liquidity.values()) > 0
    ]
    usd_price_lrna = (
        1 / load_omnipool.lrna_price('2-Pool-Stbl') / 1.01  # fudging this because I can't get the stableswap pool shares
    )
    load_omnipool.add_token(
        'USD', liquidity = usd_price_lrna * 1000000, lrna=1000000
    ).stablecoin = 'USD'
    print('finished downloading omnipool data')
    return OmnipoolRouter(exchanges=[load_omnipool, *stableswap_pools], unique_id='router'), cache_time


def load_money_market(block_number: int) -> MoneyMarket:
    print("Loading money market data...")
    try:
        # if '/' in __file__:
        #     os.chdir(__file__[:__file__.rfind('/')])
        print(f"attempting to load money market from: {os.getcwd()}")
        load_mm = load_money_market_from_file()
    except FileNotFoundError:
        print('No local money market save file found - downloading from chain...')
        load_mm = get_current_money_market()
        load_mm.time_step = block_number

    if load_mm is None:
        print('Money market could not be loaded - check internet connection.')
        quit()

    # toss out any existing toxic CDPs
    load_mm.cdps = [cdp for cdp in load_mm.cdps if not load_mm.is_liquidatable(cdp)]
    load_mm.borrowed = {tkn: sum([cdp.debt[tkn] for cdp in load_mm.cdps if tkn in cdp.debt]) for tkn in load_mm.borrowed}
    try:
        save_money_market(load_mm, filename=f"money_market_savefile_{block_number}")
    except FileNotFoundError:
        pass

    print("Finished downloading money_market data.")
    return load_mm


def distribute_cdps(
        money_market: MoneyMarket,
        extra_collateral: dict[str, float],
        extra_debt: dict[str, float],
        num_cdps: int
) -> list[CDP]:
    """
    Create new CDPs to distribute the added collateral and debt
    """

    # this is simple because all values are in USD
    extra_collateral_total = sum(extra_collateral.values())
    extra_debt_total = sum(extra_debt.values())

    if extra_collateral_total == 0:
        new_cdps = [
            CDP(
                debt={tkn: extra_debt[tkn] / money_market.price(tkn) for tkn in extra_debt},
                collateral={}
            )
        ]
    elif extra_debt_total == 0:
        new_cdps = [
            CDP(
                collateral={tkn: extra_collateral[tkn] / money_market.price(tkn) for tkn in extra_collateral},
                debt={}
            ),
        ]
    else:
        # check the liquidation threshold of the collateral and debt assets
        liquidation_threshold = money_market.cdp_liquidation_threshold(CDP(
            collateral={tkn: extra_collateral[tkn] / money_market.price(tkn) for tkn in extra_collateral},
            debt={tkn: extra_debt[tkn] / money_market.price(tkn) for tkn in extra_debt}
        ))
        hf_min = 1 / liquidation_threshold if liquidation_threshold > 0 else float('inf')
        # collateral_weights = truncated_bell_curve(
        #     peak_x=hf_min if 1 < hf_min < 1_000_000 else 1, x_max=2, dist_length=num_cdps, sigma_scale=0.3, left_compression=4
        # )[1]
        collateral_weights = np.logspace(2, 1, num_cdps) * np.linspace(0.02, 0.1, num_cdps)
        collateral_weights /= collateral_weights.sum()
        print("Minimum collateral/debt ratio:", hf_min)

        collateral_remaining = extra_collateral_total
        debt_remaining = extra_debt_total
        new_cdps = []
        cdp_collateral_ratio = hf_min * 1.01
        for i in range(num_cdps - 1):
            new_cdps.append(
                CDP(
                    collateral={
                        tkn: collateral_weights[i] * extra_collateral[tkn] / money_market.price(tkn)
                        for tkn in extra_collateral
                    },
                    debt={
                        tkn: min((collateral_weights[i] * extra_collateral_total / cdp_collateral_ratio)
                             / money_market.price(tkn), extra_debt[tkn])
                        for tkn in extra_debt
                    }
                )
            )
            collateral_remaining -= money_market.value_assets(new_cdps[-1].collateral)
            debt_remaining -= money_market.value_assets(new_cdps[-1].debt)
            cdp_collateral_ratio = (cdp_collateral_ratio * (num_cdps - i - 1) + collateral_remaining / debt_remaining) / (num_cdps - i)


        # correct rounding errors in the last CDP
        collateral_distributed = {
            tkn: sum([cdp.collateral.get(tkn, 0) for cdp in new_cdps])
            for tkn in extra_collateral
        }
        debt_distributed = {
            tkn: sum([cdp.debt.get(tkn, 0) for cdp in new_cdps]) for tkn in extra_debt
        }
        collateral_intended = {tkn: extra_collateral[tkn] / money_market.price(tkn) for tkn in extra_collateral}
        debt_intended = {tkn: extra_debt[tkn] / money_market.price(tkn) for tkn in extra_debt}
        # last one perfectly takes up what's left
        new_cdps.append(
            CDP(
                collateral={
                    tkn: (collateral_intended[tkn] - collateral_distributed.get(tkn, 0))
                    for tkn in extra_collateral
                },
                debt={
                    tkn: (debt_intended[tkn] - debt_distributed.get(tkn, 0))
                    for tkn in extra_debt
                }
            )
        )
    return new_cdps


def rebuild_money_market():
    new_mm = initial_mm.copy()
    # apply all changes from session_state
    new_mm.assets = {}
    new_mm.asset_list = []
    new_mm.cdps = []
    old_prices = {asset.name: asset.price for asset in initial_mm.assets.values()}
    for asset in st.session_state.get("assets", []):
        new_mm.add_new_asset(asset)

    changed_prices = {asset.name for asset in st.session_state.get("assets", []) if asset.price != old_prices.get(asset.name, None)}
    for cdp in initial_cdps:
        new_cdp = cdp.copy()
        new_cdp.health_factor = None
        new_cdp.liquidation_threshold = None
        for tkn in changed_prices:
            if tkn in new_cdp.collateral:
                new_cdp.collateral[tkn] *= old_prices[tkn] / new_mm.assets[tkn].price
            if tkn in new_cdp.debt:
                new_cdp.debt[tkn] *= old_prices[tkn] / new_mm.assets[tkn].price
        if st.session_state.include_original_cdps and new_mm.get_health_factor(cdp) >= 1:
            new_mm.add_cdp(new_cdp)

    new_mm.add_cdps(distribute_cdps(
        extra_collateral=st.session_state.get("add_collateral", {}),
        extra_debt=st.session_state.get("add_debt", {}),
        money_market=new_mm,
        num_cdps=st.session_state.get("number_of_cdps", 20)
    ))
    st.session_state.refresh_graphs = True
    return new_mm

initial_router, cache_timestamp = load_omnipool_router()
initial_omnipool = initial_router.exchanges['omnipool']
initial_mm = load_money_market(initial_omnipool.time_step)
initial_cdps = [cdp.copy() for cdp in initial_mm.cdps]
initial_cdps = sorted(initial_cdps, key=lambda cdp: initial_mm.value_assets(cdp.collateral))[::-1]
initial_mm.cdps = initial_cdps
initial_router.exchanges['money_market'] = initial_mm
initial_router.asset_list = list(set(initial_router.asset_list) | set(initial_mm.asset_list))
initial_stableswaps = [exchange for exchange in initial_router.exchanges.values() if isinstance(exchange, StableSwapPoolState)]

equivalency_map = {
    'Wrapped staked ETH': 'ETH',
    'Wrapped ETH (Moonbeam Wormhole)': 'ETH',
    'aETH': 'ETH',
    'aDOT': 'DOT',
    '2-Pool-GDOT': 'DOT',
    'Bifrost Voucher DOT': 'DOT',
    'vDOT': 'DOT',
    'WETH': 'ETH',
    '2-Pool-GETH': 'ETH',
    '2-Pool-WETH': 'ETH',
    'Tether (Ethereum native)': 'USD',
    'Tether (Moonbeam Wormhole)': 'USD',
    'Tether': 'USD',
    'USDC (Ethereum native)': 'USD',
    'USDC (Moonbeam Wormhole)': 'USD',
    'DAI (Moonbeam Wormhole)': 'USD',
    'aUSDT': 'USD',
    'USDC': 'USD',
    'USDT': 'USD',
    '2-Pool-Stbl': 'USD',
    '3-Pool': 'USD',
    '4-Pool': 'USD',
    'WBTC': 'BTC',
    'interBTC': 'BTC',
    'Wrapped BTC (Moonbeam Wormhole)': 'BTC',
    'tBTC': 'BTC',
    '2-Pool': 'BTC',
}
main_tokens = [
    tkn for tkn in set(initial_router.asset_list + list(equivalency_map.values()))
    if tkn not in equivalency_map.keys()
]
main_tokens = sorted(main_tokens, key=lambda x: initial_router.liquidity[x] if x in initial_router.liquidity else float('inf'), reverse=True)
print(f"Main tokens: {main_tokens}")
# determine start prices for all tokens
start_price = {
    tkn: initial_omnipool.usd_price(tkn) if tkn in initial_omnipool.asset_list
    else initial_mm.price(tkn) if tkn in initial_mm.asset_list
    else (
        initial_omnipool.usd_price(equivalency_map[tkn]) if equivalency_map[tkn] in initial_omnipool.asset_list
        else initial_mm.price(equivalency_map[tkn]) if equivalency_map[tkn] in initial_mm.asset_list else 0
    ) if tkn in equivalency_map
    else 0
    for tkn in sorted(set(initial_router.asset_list) | set(equivalency_map.values()), key=lambda x: x in initial_mm.asset_list)
}
hdx = MoneyMarketAsset(
    name="HDX",
    price=start_price["HDX"],
    liquidation_threshold=0.7,
    liquidation_bonus=0.05,
    supply_cap=0
)
initial_mm.add_new_asset(hdx)
initial_mm.asset_list.sort(key=lambda x: -1 if x == "HDX" else 0)
for tkn in start_price:
    if start_price[tkn] == 0:
        eqs = [key for key in equivalency_map if equivalency_map[key] == tkn and key in initial_mm.asset_list]
        if eqs:
            start_price[tkn] = sum(start_price[eq] for eq in eqs) / len(eqs)

for exchange in initial_stableswaps:
    priced_tokens = [tkn for tkn in exchange.asset_list if start_price[tkn] != 0]
    if priced_tokens == []:
        for tkn in exchange.asset_list:
            start_price[tkn] = 1  # default price for USD
    else:
        for tkn in exchange.asset_list:
            if tkn not in priced_tokens:
                start_price[tkn] = exchange.price(tkn, start_price[priced_tokens[0]]) * start_price[priced_tokens[0]]

st.session_state.setdefault("time_steps", 20)
st.session_state.setdefault("include_original_cdps", True)
st.session_state.setdefault("resolution", 20)
st.session_state.setdefault("add_collateral", {"HDX": 1_000_000})
st.session_state.setdefault("add_debt", {"USDT": 600_000})
st.session_state.setdefault("price_change", {
    "HDX": -20
})
st.session_state.setdefault("number_of_cdps", 20)
st.session_state.setdefault("assets", [asset.copy() for asset in initial_mm.assets.values()])
if "money_market" not in st.session_state:
    for asset in st.session_state.get("assets", []):
        asset.price = start_price[asset.name]
st.session_state["money_market"] = rebuild_money_market()
st.session_state.setdefault("asset_changed", False)
st.session_state["refresh_graphs"] = True
st.session_state["run_simulation"] = False

with st.sidebar:
    def sidebar_builder():
        label_col, input_col = st.columns([2, 1], vertical_alignment="center")
        with label_col:
            st.subheader("Time steps:")
        with input_col:
            st.session_state["time_steps"] = st.number_input(
                label="time steps",
                min_value=2,
                max_value=100,
                value=st.session_state["time_steps"],
                label_visibility="collapsed"
            )

        def change_param_form(
                key: str,
                title: str,
                tokens: list[str] or Callable,
                default_value: float,
                default_token: str,
                min_value: float,
                max_value: float,
                on_change: Callable,
                expanded: bool = False,
                number_format: str = ''
        ):
            if callable(tokens):
                tokens = tokens()
            def set_delete_flag(asset, flag):
                st.session_state.__setitem__(flag, asset)
            expanded = st.session_state.get(f"{key}_expander", expanded)
            st.session_state[f"{key}_expander"] = expanded

            delete_flag = f"{key}_pending_delete"
            asset_to_delete = st.session_state.pop(delete_flag, None)
            if asset_to_delete is not None:
                st.session_state[key].pop(asset_to_delete, None)
                on_change()
            expander_section = st.expander(title, expanded=expanded)
            with expander_section:
                with st.form(f"{key}_change_form", border=False):
                    if number_format == "$":
                        label_column, dollar_column, input_column = st.columns([12, 1, 8], vertical_alignment="center")
                    else:
                        label_column, input_column = st.columns([3, 2], vertical_alignment="center")
                    with label_column:
                        st.selectbox(" ", tokens, index=tokens.index(default_token), key=f"{key}_asset",
                                     label_visibility="collapsed")
                    if number_format == "$":
                        with dollar_column:
                            st.write("$")
                    with input_column:
                        st.number_input(
                            label=key,
                            min_value=min_value,
                            max_value=max_value,
                            value=default_value,
                            key=f"{key}_amt",
                            label_visibility="collapsed"
                        )
                    with st.columns([1, 3, 1])[1]:
                        add_p = form_submit_button("Add", use_container_width=True)
                    if add_p:
                        if st.session_state[f"{key}_asset"] not in st.session_state[key]:
                            st.session_state[key][st.session_state[f"{key}_asset"]] = 0
                        new_val = st.session_state[key][st.session_state[f"{key}_asset"]] \
                                  + float(st.session_state.get(f"{key}_amt", 0) or 0)

                        if number_format == "$":
                            new_val = max(new_val, 0)

                        st.session_state[key][st.session_state[f"{key}_asset"]] = new_val
                        on_change()

                for asset, val in sorted(st.session_state[key].items()):
                    name_col, amt_col, del_col = st.columns([5, 3, 2])
                    with name_col:
                        st.markdown(f":red[{asset}]")
                    with amt_col:
                        if number_format == "%":
                            st.markdown(f":red[{'+' if val > 0 else ''}{val:,.0f}%]")
                        elif number_format == "$":
                            st.markdown(f":red[${val:,.0f}]")
                        else:
                            if number_format != '':
                                print(f"WARNING: unrecognized number format in change_param_form() ({number_format})")
                            st.markdown(f":red[{val:,.0f}]")
                    with del_col:
                        st.button(
                            "✕",
                            key=f"del_{key}_{asset}",
                            on_click=lambda asset=asset, flag=delete_flag: set_delete_flag(asset, flag),
                        )
            return expander_section

        def show_asset_config(asset: MoneyMarketAsset, title: str=None):
            name = asset.name
            title = title or name

            def update_asset(name, attr, value):
                asset = [asset for asset in st.session_state["assets"] if asset.name == name][0]
                setattr(asset, attr, value)
                rebuild_money_market()

            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        text-align: center;
                        border-top: 1px dotted #ccc;
                    ">
                        <h3 style="color: #886b6b; margin: 0;">{title}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                label_col, input_col = st.columns([3, 2], vertical_alignment="center")
                with label_col:
                    st.write("Price (USD):")
                with input_col:
                    asset.price = st.number_input(
                        label=f"price ({name})",
                        value=asset.price,
                        min_value=0.0000001,
                        max_value=1_000_000.0,
                        format=f"%.{min(6, 6-math.floor(math.log10(asset.price)))}f",
                        label_visibility="collapsed",
                        key=f"price_input_{name}",
                        on_change=lambda n=name: update_asset(n, 'price', st.session_state[f"price_input_{n}"])
                    )
                label_col, input_col = st.columns([3, 2], vertical_alignment="center")
                with label_col:
                    st.write("Liquidation threshold:")
                with input_col:
                    asset.liquidation_threshold = st.number_input(
                        label=f"liquidation threshold ({name})",
                        max_value=0.99,
                        value=asset.liquidation_threshold,
                        label_visibility="collapsed",
                        key=f"lt_input_{name}",
                        on_change=lambda n=name: update_asset(n, 'liquidation_threshold', st.session_state[f"lt_input_{n}"])
                    )
                label_col, input_col = st.columns([3, 2], vertical_alignment="center")
                with label_col:
                    st.write("Liquidation bonus:")
                with input_col:
                    asset.liquidation_bonus = st.number_input(
                        label=f"liquidation bonus ({name})",
                        max_value=0.2,
                        value=asset.liquidation_bonus,
                        label_visibility="collapsed",
                        key=f"lb_input_{name}",
                        on_change=lambda n=name: update_asset(n, 'liquidation_bonus', st.session_state[f"lb_input_{n}"])
                    )
                label_col, input_col = st.columns([3, 2], vertical_alignment="center")
                with label_col:
                    st.write("E-mode label:")
                with input_col:
                    asset.emode_label = st.text_input(
                        label=f"e-mode label ({name})",
                        value=asset.emode_label,
                        label_visibility="collapsed",
                        key=f"el_input_{name}",
                        on_change=lambda n=name: update_asset(n, 'emode_label', st.session_state[f"el_input_{n}"])
                    )
                if asset.emode_label:
                    label_col, input_col = st.columns([3, 2], vertical_alignment="center")
                    with label_col:
                        st.write("E-mode liquidation threshold:")
                    with input_col:
                        asset.emode_liquidation_threshold = st.number_input(
                            label=f"e-mode liquidation threshold ({name})",
                            max_value=0.99,
                            value=asset.emode_liquidation_threshold,
                            label_visibility="collapsed",
                            key=f"elt_input_{name}",
                            on_change=lambda n=name: update_asset(n, 'emode_liquidation_threshold', st.session_state[f"elt_input_{n}"])
                        )
                    label_col, input_col = st.columns([3, 2], vertical_alignment="center")
                    with label_col:
                        st.write("E-mode liquidation bonus:")
                    with input_col:
                        asset.emode_liquidation_bonus = st.number_input(
                            label=f"e-mode liquidation bonus ({name})",
                            max_value=0.2,
                            value=asset.emode_liquidation_bonus,
                            label_visibility="collapsed",
                            key=f"elb_input_{name}",
                            on_change=lambda n=name: update_asset(n, 'emode_liquidation_bonus', st.session_state[f"elb_input_{n}"])
                        )

        def money_market_config_section():
            title = "Money Market Parameters"
            mm = st.session_state["money_market"]

            with st.expander(title):
                text_col, input_col = st.columns([3, 1], vertical_alignment="center")
                with text_col:
                    st.write("New asset: ")
                with input_col:
                    new_asset_name = st.text_input(
                        label="new asset name",
                        value="",
                        placeholder="name?",
                        label_visibility="collapsed",
                    )
                if new_asset_name != "" and new_asset_name not in mm.asset_list:
                    print(f"!!new asset detected: {new_asset_name}")
                    new_asset = MoneyMarketAsset(
                        name=new_asset_name,
                        price=start_price[new_asset_name] if new_asset_name in start_price else 1.0,
                        liquidation_threshold=0.7,
                        liquidation_bonus=0.05,
                    )
                    st.session_state["assets"] = [new_asset] + st.session_state.get("assets", [])
                    print("money market:")
                    for asset in mm.assets:
                        print(f" - {asset}: {mm.assets[asset].name}")
                    print("session_state assets:")
                    for asset in st.session_state["assets"]:
                        print(f" - {asset.name}")
                    rebuild_money_market()

                mm = st.session_state["money_market"]
                for name in mm.asset_list:
                    show_asset_config(mm.assets[name])

        change_param_form(
            key="price_change",
            number_format="%",
            tokens=main_tokens,
            default_token="HDX",
            default_value=-20,
            min_value=-99,
            max_value=1000,
            title=f"Price change over {st.session_state['time_steps']} steps",
            on_change=lambda: None,
            expanded=True
        )

        with change_param_form(
            key="add_collateral",
            title="Add collateral",
            tokens=st.session_state["money_market"].asset_list,
            default_token="HDX",
            default_value=1_000_000,
            min_value=-1_000_000_000,
            max_value=1_000_000_000,
            on_change=lambda: st.session_state.__setitem__("asset_changed", True),
            number_format="$",
            expanded=True
        ):
            st.checkbox("include original CDPs", key="include_original_cdps", on_change=rebuild_money_market)
            if st.session_state.get("add_collateral", {}) != {}:
                column1, column2, column3 = st.columns([2, 1, 1], vertical_alignment="center")
                with column1:
                    st.write("Distribute over:")
                with column2:
                    st.number_input(
                        label="Number of new CDPs to create",
                        min_value=1,
                        max_value=100,
                        key="number_of_cdps",
                        on_change=lambda: st.session_state.__setitem__("asset_changed", True),
                        label_visibility="collapsed"
                    )
                with column3:
                    st.write("CDPs")

        change_param_form(
            key="add_debt",
            title="Add debt",
            tokens=st.session_state["money_market"].asset_list,
            default_token="USDT",
            default_value=600_000,
            min_value=-1_000_000_000,
            max_value=1_000_000_000,
            on_change=lambda: st.session_state.__setitem__("asset_changed", True),
            number_format="$",
            expanded=True
        )
        money_market_config_section()
        if st.session_state.asset_changed:
            st.session_state["money_market"] = rebuild_money_market()
            st.session_state["asset_changed"] = False

    sidebar_builder()


@st.fragment
def plot_health_factor_distribution(mm):
    with st.expander("CDP Health Factor Distribution"):

        res_column, graph_by_column = st.columns([1, 1])
        with res_column:
            resolution = st.radio(
                label="distribution resolution:",
                options=["low", "high"],
                key=f"resolution {mm.time_step}"
            )
        with graph_by_column:
            graph_by = st.radio(
                label="Graph health factor distribution by:",
                options=["debt", "collateral"],
                index=1,
                key=f"graph by {mm.time_step}"
            )

        cdps = mm.cdps
        health_factors = [cdp.health_factor for cdp in cdps]
        fig, ax = plt.subplots(figsize=(16, 6))

        if resolution == "low":
            smoothing = 3.0
            bins, dist = get_distribution(
                health_factors,
                [mm.value_assets(cdp.debt) for cdp in cdps] if graph_by == "debt" else [mm.value_assets(cdp.collateral) for cdp in cdps],
                resolution=20,
                minimum=min(1, min(health_factors)),
                maximum=2,
                smoothing=smoothing
            )
            ax.plot(bins, dist, label="Health Factor Distribution")
            ax.set_xlabel("Health Factor")
            ax.set_ylabel("Debt-weighted density" if graph_by == "debt" else "Collateral-weighted density")
        else:
            bins, dist = get_distribution(
                health_factors,
                [mm.value_assets(cdp.debt) for cdp in cdps] if graph_by == "debt" else [mm.value_assets(cdp.collateral) for cdp in cdps],
                resolution=200,
                minimum=min(1, min(health_factors)),
                maximum=2,
                smoothing=1.0
            )
            ax.set_xlabel("Health Factor")
            ax.set_ylabel("collateral")
            ax.plot(bins, dist, label="Health Factor Distribution")
            ok_cdps = []
            toxic_cdps = []
            for cdp in cdps:
                if mm.is_toxic(cdp):
                    toxic_cdps.append(cdp)
                else:
                    ok_cdps.append(cdp)

            for cdp in ok_cdps:
                if graph_by == "debt":
                    plt.scatter(x=cdp.health_factor, y=mm.value_assets(cdp.debt), color='#009', alpha=0.1)
                else:
                    plt.scatter(x=cdp.health_factor, y=mm.value_assets(cdp.collateral), color='#009', alpha=0.1)
            for cdp in toxic_cdps:
                if graph_by == "debt":
                    plt.scatter(x=cdp.health_factor, y=mm.value_assets(cdp.debt), color='#F00', alpha=0.5, marker="x")
                else:
                    plt.scatter(x=cdp.health_factor, y=mm.value_assets(cdp.collateral), color='#F00', alpha=0.5, marker="x")
        st.pyplot(fig)

def run_app():
    price_factor = {
        tkn: st.session_state.price_change.get(tkn, 0) for tkn in main_tokens
    }
    # update price change defaults for equivalent tokens
    price_factor.update({
        tkn: price_factor[equivalency_map[tkn]] for tkn in equivalency_map.keys()
        if equivalency_map[tkn] in main_tokens
    })
    # trade_agent = Agent(enforce_holdings=False)
    def update_prices(state: GlobalState):
        if state.time_step >= len(price_paths[main_tokens[0]]):
            return
        for price_tkn in price_paths:
            relevant_pools = [pool for pool in state.pools['router'].exchanges.values() if price_tkn in pool.asset_list]
            if price_tkn == "HDX":
                # there is no external market for HDX, so we set the price in the money market to the omnipool price
                price_paths[price_tkn][state.time_step] = state.pools['router'].exchanges['omnipool'].usd_price('HDX')
            for pool in relevant_pools:
                if isinstance(pool, FixedPriceExchange):
                    pool.prices[price_tkn] = price_paths[price_tkn][state.time_step]
                elif isinstance(pool, MoneyMarket):
                    pool.assets[price_tkn].price = price_paths[price_tkn][state.time_step]
                    pool.prices[price_tkn] = price_paths[price_tkn][state.time_step]
                # elif isinstance(pool, OmnipoolState):
                #     # execute trades to move price towards target
                #     target_price = price_paths[price_tkn][state.time_step - 1] * pool.lrna_price('USD')
                #     pool.trade_to_price(trade_agent, tkn=price_tkn, target_price=target_price)

            state.external_market[price_tkn] = price_paths[price_tkn][state.time_step]

    time_steps = st.session_state["time_steps"]
    # calculate price path from start price to final price
    all_price_change_tokens = [tkn for tkn in price_factor if price_factor[tkn] != 0]
    final_price = {tkn: start_price[tkn] * (1 + price_factor[tkn] / 100) for tkn in start_price}
    price_paths = {
        tkn: [
            start_price[tkn]] + [start_price[tkn] - (start_price[tkn] - final_price[tkn]) * ((i + 1) / time_steps)
            for i in range(time_steps)
        ] + [final_price[tkn]] for tkn in start_price
    }
    lrna_price_usd = initial_omnipool.lrna_price('USD')
    time_steps += 1
    omnipool_swap_quantities = {
        tkn: [
            initial_omnipool.calculate_trade_to_price(tkn=tkn, target_price=price_paths[tkn][i] * lrna_price_usd)
            for i in range(time_steps)
        ] for tkn in all_price_change_tokens if tkn in initial_omnipool.asset_list and price_paths[tkn][-1] != initial_omnipool.usd_price(tkn)
    }
    swap_schedule = []
    for i in range(time_steps):
        swap_schedule.append([])
        for tkn in omnipool_swap_quantities:
            net_swap = omnipool_swap_quantities[tkn][i] - (omnipool_swap_quantities[tkn][i - 1] if i > 0 else 0)
            swap_schedule[i].append({
                'tkn_sell': tkn if net_swap > 0 else 'LRNA',
                'tkn_buy': 'LRNA' if net_swap > 0 else tkn,
                'sell_quantity': net_swap if net_swap >= 0 else None,
                'buy_quantity': -net_swap if net_swap < 0 else None,
            })

    omnipool = copy.deepcopy(initial_omnipool)
    stableswaps = [exchange.copy() for exchange in initial_stableswaps]
    router = initial_router.copy()
    router.exchanges = {pool.unique_id: pool for pool in [omnipool, st.session_state["money_market"], *stableswaps]}

    initial_state = GlobalState(
        pools=[router],
        agents={
            'liquidator': Agent(
                enforce_holdings=False,
                trade_strategy=liquidate_cdps('router') + sell_all('router', tkn_buy='USDT')
            ),
            'trader': Agent(
                enforce_holdings=False,
                trade_strategy=schedule_swaps(
                    swaps=swap_schedule,
                    pool_id='router'
                )
            ),
            'arbitrageur': Agent(
                enforce_holdings=False,
                trade_strategy=omnipool_arbitrage(
                    pool_id='omnipool',
                    arb_precision=20,
                    skip_assets=['HDX'],
                )
            )
        },
        evolve_function=update_prices,
        external_market=start_price
    )

    with st.spinner(f"Running {time_steps} simulation steps..."):
        sim_start = time.time()
        new_state = initial_state.copy()
        events = []
        while new_state.time_step < time_steps:
            new_state.time_step += 1
            agent_ids = list(new_state.agents.keys())
            for agent_id in agent_ids:
                new_state.agents[agent_id].trade_strategy.execute(new_state, agent_id)
            for pool in new_state.pools['router'].exchanges.values():
                pool.update()
            update_prices(new_state)
            events.append(new_state.archive())
            if new_state.time_step >= time_steps and max([
                abs(events[-1].pools['router'].price(tkn, 'USD') - events[-2].pools['router'].price(tkn, 'USD'))
                / (events[-2].pools['router'].price(tkn, 'USD') or 1)
                for tkn in router.asset_list
            ]) > 0.001:
                time_steps += 1  # extend simulation if prices are still changing significantly
                print(f"extending simulation to {time_steps} steps")
        events.pop()  # drop the last one because we know the prices didn't change much
        sim_time = time.time() - sim_start
        st.sidebar.info(f"Simulation completed in {sim_time:.2f} seconds")
    st.header("Simulation Results")

    for event in events:
        # add the individual exchange pools to the pools dict for easy access
        event.pools = {pool.unique_id: pool for pool in [event.pools['router'], *event.pools['router'].exchanges.values()]}

    final_mm = events[-1].pools['money_market']
    for cdp in final_mm.cdps:
        cdp.health_factor = final_mm.get_health_factor(cdp)

    router.find_routes('aDOT', 'USD', direction='buy')
    router.find_routes('aDOT', 'USD', direction='sell')
    events[-1].pools['router'].price('DOT', 'USD')
    with st.expander(f"Prices over {len(events)} steps"):
        for tkn in sorted(start_price, key=lambda x: equivalency_map[x] if x in equivalency_map else x):
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.set_xlabel("Time Steps")
            ax.set_ylabel(f"{tkn} price (USD)")
            tkn_price_path = [
                event.pools['omnipool'].usd_price(tkn)
                if tkn in event.pools['omnipool'].asset_list
                else event.pools['money_market'].price(tkn)
                if tkn in event.pools['money_market'].asset_list
                else event.pools['router'].price_route(events[0].pools['router'].find_best_route(tkn, 'USD'), 'buy')
                if tkn in event.pools['router'].asset_list
                else price_paths[tkn][i] if i < len(price_paths[tkn])
                else price_paths[tkn][-1]
                for i, event in enumerate(events)
            ]
            if max(tkn_price_path) == 0:
                continue
            if abs(min(tkn_price_path) / max(tkn_price_path) - 1) < 0.01:
                continue
            # if abs(1 - tkn_price_path[-1] / final_price[tkn]) > 0.01:
            #     st.warning(f"{tkn} price did not reach expected final price of {final_price[tkn]} USD")
            ax.plot(tkn_price_path, label=f"{tkn} price")
            d = max(0, 2 - int(math.floor(math.log10(abs(tkn_price_path[0])))))
            p = tkn_price_path[0]
            ax.annotate(
                f"${(int(p) if d==0 else f'{p:.2f}' if d<=2 else round(p, d))}",
                xy=(0, tkn_price_path[0]),
                xytext=(0, min(tkn_price_path) if tkn_price_path[0] > tkn_price_path[-1] else max(tkn_price_path))
            )
            ax.scatter(marker='o', s=20, x=0, y=tkn_price_path[0], color='#009')
            p = tkn_price_path[-1]
            ax.annotate(
                f"${(int(p) if d==0 else f'{p:.2f}' if d<=2 else round(p, d))}\n({((p - tkn_price_path[0]) / tkn_price_path[0] * 100):+.1f}%)",
                xy=(len(tkn_price_path)-1, tkn_price_path[-1]),
                xytext=(
                    len(tkn_price_path) - 1.4, min(tkn_price_path)
                    if tkn_price_path[0] < tkn_price_path[-1]
                    else max(tkn_price_path) - 0.05 * (max(tkn_price_path) - min(tkn_price_path))
                ),
                arrowprops=dict(arrowstyle="->")
            )
            ax.scatter(marker='o', s=20, x=len(tkn_price_path)-1, y=tkn_price_path[-1], color='#009')
            st.pyplot(fig)

    @st.fragment
    def plot_agent_holdings(agent_id):
        with st.expander(f"{agent_id} holdings"):
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.set_xlabel("Time Steps")
            ax.set_ylabel(f"{agent_id} holdings (normalized to initial USD price)")
            for tkn in events[-1].agents[agent_id].holdings:
                liquidator_holdings = [
                    event.agents[agent_id].get_holdings(tkn) * start_price[tkn] / start_price['USD']
                    for event in events
                ]
                if max(liquidator_holdings) == 0:
                    continue
                ax.plot(liquidator_holdings, label=f"{tkn}")
            ax.legend()
            st.pyplot(fig)

    plot_agent_holdings('liquidator')
    # plot_agent_holdings('arbitrageur')

    @st.fragment
    def plot_toxic_debt():
        with (st.expander(f"Toxic debt")):
            st.radio(
                label="View toxic debt as:",
                options=["absolute (USD)", "percentage", "USD at initial prices"],
                index=1,
                key="toxic_debt_view"
            )
            st.checkbox(label="breakdown by debt token", key="toxic_debt_breakdown", value=True)
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Toxic Debt")
            toxic_debt = {tkn: [0] * len(events) for tkn in st.session_state["money_market"].borrowed}
            for i, event in enumerate(events):
                for cdp in event.pools['money_market'].cdps:
                    if event.pools['money_market'].is_toxic(cdp):
                        for tkn in cdp.debt:
                            toxic_debt[tkn][i] += cdp.debt[tkn]
            toxic_debt.update(
                {"all": [
                    sum([
                        toxic_debt[tkn][i] * (
                            events[i].pools['money_market'].assets[tkn].price
                            if st.session_state["toxic_debt_view"] != "USD at initial prices"
                            else start_price[tkn]
                        ) for tkn in toxic_debt
                    ]) for i in range(len(events))
                ]}
            )
            if not st.session_state.toxic_debt_breakdown:
                toxic_debt = {'all': toxic_debt['all']}

            if max(toxic_debt.values(), key=lambda value: max(value)) == 0:
                st.info("No toxic debt at any point in the simulation.")
                return

            money_market_values = [
                sum([
                    event.pools['money_market'].value_assets(cdp.debt)
                    for cdp in event.pools['money_market'].cdps
                ]) for event in events
            ]

            for i, (tkn, debt) in enumerate(toxic_debt.items()):
                if max(debt) == 0:
                    continue
                if st.session_state.toxic_debt_view == "percentage":
                    ax.plot(
                        [
                            debt[i] * (
                                event.pools['money_market'].assets[tkn].price
                                if tkn in st.session_state["money_market"].asset_list else 1
                            )
                            / money_market_values[i] * 100
                            for i, event in enumerate(events)
                        ], label=f"{tkn}" if st.session_state.toxic_debt_breakdown else None
                    )
                    ax.set_title("Toxic debt as a percentage of total debt")
                elif st.session_state.toxic_debt_view == "USD at initial prices":
                    ax.plot(
                        [debt[i] * (start_price[tkn] if tkn in start_price else 1) for i in range(len(debt))],
                        label=f"{tkn}" if st.session_state.toxic_debt_breakdown else None
                    )
                    ax.set_title("Toxic debt in USD valued at initial prices")
                else:
                    ax.plot(
                        [
                            debt[i] * (
                                event.pools['money_market'].assets[tkn].price
                                if tkn in st.session_state["money_market"].asset_list else 1
                            )
                            for i, event in enumerate(events)
                        ], label=f"{tkn}" if st.session_state.toxic_debt_breakdown else None
                    )
                    ax.set_title("Toxic debt in USD")
            ax.legend()
            st.pyplot(fig)

    plot_toxic_debt()


    @st.fragment
    def plot_collateral():
        with (st.expander(f"Collateral")):
            st.radio(
                label="View collateral as:",
                options=["absolute USD", "USD at initial prices"],
                index=1,
                key="collateral_view"
            )
            st.checkbox(label="breakdown by token", key="collateral_breakdown", value=True)
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Collateral value (USD")
            collateral_tokens = {
                tkn: [
                    sum([
                        cdp.collateral[tkn] if tkn in cdp.collateral else 0
                        for cdp in event.pools['money_market'].cdps
                    ])
                    for event in events
                ] for tkn in st.session_state["money_market"].asset_list
            }
            collateral_tokens.update(
                {"all": [
                    sum([
                        collateral_tokens[tkn][i] * (
                            events[i].pools['money_market'].assets[tkn].price
                            if st.session_state.collateral_view != "USD at initial prices"
                            else start_price[tkn]
                        ) for tkn in collateral_tokens
                    ])
                    for i in range(len(events))
                ]}
            )
            if not st.session_state.collateral_breakdown:
                collateral_tokens = {'all': collateral_tokens['all']}

            for i, (tkn, collateral) in enumerate(collateral_tokens.items()):
                if max(collateral) == 0:
                    continue
                if st.session_state.collateral_view == "USD at initial prices":
                    ax.plot(
                        [
                            collateral[i] * (start_price[tkn] if tkn in start_price else 1) for i in range(len(events))
                        ], label=f"{tkn}" if st.session_state.collateral_breakdown else None
                    )
                    ax.set_title("Collateral in USD valued at initial prices")
                else:
                    ax.plot(
                        [
                            collateral[i] * (
                                event.pools['money_market'].assets[tkn].price
                                if tkn in st.session_state["money_market"].asset_list else 1
                            ) for i, event in enumerate(events)
                        ], label=f"{tkn}" if st.session_state.collateral_breakdown else None
                    )
                    ax.set_title("Collateral in USD")
            ax.legend()
            st.pyplot(fig)

    plot_collateral()

    @st.fragment
    def plot_omnipool_assets():
        omnipool = events[-1].pools['router'].exchanges['omnipool']
        with st.expander(f"Omnipool asset balances"):
            asset_name = st.selectbox(
                label="Select asset",
                options=sorted(list(omnipool.liquidity.keys()), key=lambda x: omnipool.lrna[x], reverse=True),
                index=0,
                key="omnipool_asset_select"
            )
            liquidity_or_lrna = st.radio(
                label="View as:",
                options=["liquidity", "LRNA"],
                index=0,
                key="omnipool_asset_view"
            )
            if liquidity_or_lrna == "LRNA":
                balances = [
                    event.pools['router'].exchanges['omnipool'].lrna[asset_name]
                    for event in events
                ]
            else:
                balances = [
                    event.pools['router'].exchanges['omnipool'].liquidity[asset_name]
                    for event in events
                ]
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.set_xlabel("Time Steps")
            ax.set_ylabel(f"{asset_name if liquidity_or_lrna == 'liquidity' else f'{asset_name} H2O'} balance")

            ax.plot(balances, label=f"{asset_name} balance")
            st.pyplot(fig)

    plot_omnipool_assets()

    def get_hdx_total(state: GlobalState):
        total = {}
        for pool in state.pools['router'].exchanges.values():
            if 'HDX' in pool.liquidity:
                total[pool.unique_id] = pool.liquidity['HDX']
        total['money_market_collateral'] = sum([
            cdp.collateral.get('HDX', 0) for cdp in state.pools['router'].exchanges['money_market'].cdps
        ])
        for key, agent in state.agents.items():
            total[f"agent: {key}"] = agent.get_holdings('HDX')
        return total

    initial_hdx = get_hdx_total(initial_state)
    final_hdx = get_hdx_total(events[-1])
    hdx_change = sum(final_hdx.values()) - sum(initial_hdx.values())
    pass

    plot_health_factor_distribution(events[-1].pools['money_market'])

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Run simulation", use_container_width=True):
        st.session_state.run_simulation = True

if st.session_state.refresh_graphs:
    plot_health_factor_distribution(st.session_state["money_market"])


    def plot_cdp_value_distribution():
        mm = st.session_state["money_market"]
        with st.expander(f"Debt and collateral token amounts"):
            total_debt = {tkn: sum([cdp.debt[tkn] for cdp in mm.cdps if tkn in cdp.debt]) for tkn in mm.borrowed}
            total_collateral = {tkn: sum([cdp.collateral[tkn] for cdp in mm.cdps if tkn in cdp.collateral]) for tkn in mm.asset_list}
            debt_values_usd = {tkn: total_debt[tkn] * mm.assets[tkn].price for tkn in total_debt}
            collateral_values_usd = {tkn: total_collateral[tkn] * mm.assets[tkn].price for tkn in total_collateral}
            total_debt_usd = sum(debt_values_usd.values())
            total_collateral_usd = sum(collateral_values_usd.values())
            (token_name, collateral_amt, debt_amt) = st.columns(3)
            token_name.markdown("**Token**")
            collateral_amt.markdown("**Total Collateral**")
            debt_amt.markdown("**Total Debt**")
            for tkn in mm.asset_list:
                with token_name:
                    one_line_markdown(f"**{tkn}**")
                    one_line_markdown("-------------------")
                with debt_amt:
                    if tkn in total_debt and total_debt[tkn] > 0:
                        one_line_markdown(
                            f"{total_debt[tkn]:,.2f} "
                            f"({debt_values_usd[tkn]:,.2f} USD) "
                            f"({debt_values_usd[tkn] / total_debt_usd * 100:.2f}%)"
                        )
                    else:
                        one_line_markdown("0")
                    one_line_markdown("-------------------")

                with collateral_amt:
                    if tkn in total_collateral and total_collateral[tkn] > 0:
                        one_line_markdown(
                            f"{total_collateral[tkn]:,.2f} "
                            f"({collateral_values_usd[tkn]:,.2f} USD) "
                            f"({collateral_values_usd[tkn] / total_collateral_usd * 100:.2f}%)"
                        )
                    else:
                        one_line_markdown("0")
                    one_line_markdown("-------------------")


    plot_cdp_value_distribution()


if st.session_state.run_simulation:
    # plot_health_factor_distribution(mm_sim)
    # plot_cdp_value_distribution(mm_sim)
    run_app()
    st.session_state.run_simulation = False