from matplotlib import pyplot as plt
import sys, os
import streamlit as st
import copy

from matplotlib.lines import lineStyles
from streamlit import session_state

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

import hydradx.model.production_settings as settings
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.agents import Agent
from hydradx.model.indexer_utils import get_omnipool_liquidity, get_omnipool_trades, get_current_block_height

st.markdown("""
    <style>
        .stNumberInput button {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)
st.set_page_config(layout="wide")

st.session_state.setdefault("events", [])
st.session_state.setdefault("block_start", 8_996_500)
st.session_state.setdefault("sim_length", 10_000)
st.session_state.setdefault("omnipool", None)
st.session_state.setdefault("trades", {})
st.session_state.setdefault("regraph", False)
st.session_state.setdefault("lp_pct", 0.25)
st.session_state.setdefault("show_fee_regions", "none")
st.session_state.setdefault("show_scenario_1", True)
st.session_state.setdefault("show_scenario_2", True)
st.session_state.setdefault("show_scenario_3", True)
st.session_state.setdefault("show_scenario_4", True)


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


def load_omnipool_data():
    with st.spinner("Loading Omnipool data..."):
        block_start = st.session_state.block_start
        sim_length = st.session_state.sim_length
        trades = [trade for trade in get_omnipool_trades(
            min_block=block_start, max_block=block_start + sim_length
        ) if trade['assetIn'] == 'HDX' or trade['assetOut'] == 'HDX']
        tokens = get_omnipool_liquidity(block_start)
        initial_omnipool = OmnipoolState(
            tokens=tokens,
            asset_fee=settings.omnipool_asset_fee,
            lrna_fee=settings.omnipool_lrna_fee,
            withdrawal_fee=True,
            unique_id="omnipool"
        )
        # filter out trades that don't involve assets in the pool
        # this can happen if an asset was added to the pool after the start block
        trades = [trade for trade in trades if trade['assetIn'] in initial_omnipool.asset_list and trade['assetOut'] in initial_omnipool.asset_list]
        st.session_state["omnipool"] = initial_omnipool
        st.session_state["trades"] = trades
        pass


# Add this function definition near the top of your script
@st.cache_data
def get_max_block():
    """Fetches and calculates the max block number once and caches the result."""
    return get_current_block_height() - 100000

# Replace the original line with a call to your new cached function
max_block_number = get_max_block()

controls_col, _, text_col = st.columns([3, 1, 6])
with controls_col:
    label_col, input_col = st.columns([3, 1], vertical_alignment="center")
    with label_col:
        st.write("start at block number:")
    with input_col:
        st.number_input(
            "block number", min_value=1, max_value=max_block_number, key="block_start",
            label_visibility="collapsed",
        )
    print(f"max block number is {max_block_number}")
    label_col, input_col = st.columns([3, 1], vertical_alignment="center")
    with label_col:
        st.write("simulate # of blocks:")
    with input_col:
        st.number_input(
            "sim length", min_value=100, max_value=20000, key="sim_length",
            label_visibility="collapsed",
        )
with text_col:
    st.markdown(f"""
        Four LP agents are simulated:
        - LP1 simply holds their liquidity in the pool
        - LP2 withdraws and re-adds their liquidity every block
        - LP3 withdraws and re-adds their liquidity only when the withdrawal fee is at or below the initial fee level
        - LP4 does the same as LP2 but without withdrawal fees
    """)
def run_sim():
    load_omnipool_data()

    initial_omnipool = st.session_state.omnipool.copy()
    sim_length = st.session_state.sim_length
    block_start = st.session_state.block_start

    initial_lp = Agent(
        enforce_holdings=False,
        holdings={"HDX": initial_omnipool.liquidity["HDX"] / 4},
        unique_id="lp1"
    )
    initial_omnipool.liquidity["HDX"] *= 3 / 4
    initial_omnipool.lrna["HDX"] *= 3 / 4
    initial_omnipool.shares["HDX"] *= 3 / 4
    initial_omnipool.protocol_shares["HDX"] *= 3 / 4
    initial_omnipool.withdrawal_fee = True
    initial_omnipool.min_withdrawal_fee = 0.0001
    initial_omnipool.add_liquidity(
        agent=initial_lp,
        tkn_add='HDX',
        quantity=initial_lp.holdings['HDX']
    )

    fails = []
    print(f"{initial_omnipool.asset_fee("HDX")}), {initial_omnipool.lrna_fee("HDX")}")

    scenarios = [[] for _ in [1, 2, 3, 4]]
    with st.spinner("Running simulation..."):
        for i in [1, 2, 3, 4]:
            events = scenarios[i - 1]
            omnipool = initial_omnipool.copy()
            if i == 4:
                omnipool.withdrawal_fee = False
            lp_agent = initial_lp.copy()

            trade_agent = Agent(enforce_holdings=False, unique_id="trader2")
            trades = copy.deepcopy(st.session_state.trades)
            next_trade = trades.pop(0)
            for block in range(block_start, block_start + sim_length + 1):
                omnipool.update()
                while next_trade and next_trade['block_number'] == block:
                    trade = next_trade
                    tkn_sell = trade['assetIn']
                    tkn_buy = trade['assetOut']
                    if tkn_buy == "H2O":
                        tkn_buy = "LRNA"
                    if tkn_sell == "H2O":
                        tkn_sell = "LRNA"
                    if "assetFee" in trade:
                        omnipool.set_asset_fee(tkn_buy, trade['assetFee'])
                    if "protocolFee" in trade:
                        omnipool.set_lrna_fee(tkn_sell, trade['protocolFee'])
                    if tkn_buy in omnipool.asset_list and tkn_sell in omnipool.asset_list:
                        omnipool.fail = ""
                        omnipool.swap(
                            agent=trade_agent,
                            tkn_buy=tkn_buy,
                            tkn_sell=tkn_sell,
                            sell_quantity=trade['amountIn']
                        )
                        if omnipool.fail:
                            fails.append((omnipool.unique_id, trade))
                    next_trade = trades.pop(0) if trades else None

                if i == 1:
                    # first agent just holds
                    pass
                if i == 2 or i == 4:
                    # second agent always removes and re-adds liquidity
                    remove_readd(omnipool, lp_agent)
                elif (
                    i == 3
                    and omnipool.asset_fee("HDX") <= initial_omnipool.lrna_fee("HDX")
                    and omnipool.lrna_fee("HDX") <= initial_omnipool.lrna_fee("HDX")
                ):
                    # third agent only rebalances if fees are low
                    remove_readd(omnipool, lp_agent)

                events.append({
                    'pool': omnipool.copy(),
                    'agent': lp_agent.copy(),
                    'block': block
                })
    st.session_state.events = scenarios
    st.session_state.regraph = True

with controls_col:
    st.button("run simulation", on_click=run_sim, use_container_width=True)


# @st.fragment
def plot_scenario():
    initial_omnipool = st.session_state.omnipool
    scenarios = st.session_state.events

    initial_hdx_price = initial_omnipool.lrna_price('HDX')
    initial_hdx_fee = initial_omnipool.asset_fee('HDX')
    hdx_price_from_baseline = [[e['pool'].lrna_price('HDX') / initial_hdx_price  for e in events] for events in scenarios]

    strats = ['hold', 'withdraw and re-add every block', 'withdraw and re-add when fees are low', 'withdraw and re-add no fees']
    colors = ['blue', 'orange', 'red', 'purple']
    @st.fragment
    def draw_graph():
        fig, ax = plt.subplots(figsize=(16, 6))
        block_start, block_end = st.slider(
            min_value=st.session_state.block_start,
            max_value=st.session_state.block_start + st.session_state.sim_length,
            value=(st.session_state.block_start, st.session_state.block_start + st.session_state.sim_length),
            label="block range"
        )
        radio_column, checks_column = st.columns([1, 3])
        with radio_column:
            st.radio(label="show elevated fees", options=["none", "scenario 1", "scenario 2", "scenario 3", "scenario 4"], key="show_fee_regions")
        with checks_column:
            st.checkbox(label="scenario 1", key="show_scenario_1", value=True)
            st.checkbox(label="scenario 2", key="show_scenario_2", value=True)
            st.checkbox(label="scenario 3", key="show_scenario_3", value=True)
            st.checkbox(label="scenario 4", key="show_scenario_4", value=True)
        block_start -= st.session_state.block_start
        block_end -= st.session_state.block_start
        print_columns = st.columns(len(strats))
        # start with a flat line at 1
        print(f"drawing baseline from {block_start} to {block_end}")
        ax.plot(
            [scenarios[1][block_start]['block'], scenarios[1][block_end]['block']],
            [1, 1], label='baseline', color="grey", linestyle='--', alpha=0.5
        )
        lp_hold_value = [
            [
                e['pool'].cash_out(e['agent']) / initial_omnipool.cash_out(events[0]['agent'])
                for j, e in enumerate(events)
            ] for events in scenarios
        ]
        for i, events in enumerate(scenarios):
            if not st.session_state[f"show_scenario_{i + 1}"]:
                continue
            initial_value = events[0]['pool'].cash_out(events[0]['agent'])
            initial_holdings = events[0]['agent'].holdings[('omnipool', 'HDX')]

            with print_columns[i]:
                gain_loss = round((1 - lp_hold_value[i][-1]) * 100, 4)
                st.write(f"Scenario {i + 1} final {"losses" if gain_loss < 0 else "gains"}: {abs(gain_loss)}%")
                st.write(f"Final holdings: {round(events[-1]['agent'].holdings[('omnipool', 'HDX')], 3)} HDX in LP tokens")
                st.write(f"Final cash out value: {round(events[-1]['pool'].cash_out(events[-1]['agent']), 3)}")
                st.write(f"Final HDX price: {round(events[-1]['pool'].lrna_price('HDX'), 9)}")
                st.write(f"Final H2O/liquidity in HDX pool: {round(events[-1]['pool'].lrna['HDX'], 3)}/{round(events[-1]['pool'].liquidity['HDX'], 3)}")

            graph_events = events[block_start: block_end + 1]
            # lp_hold_value = [
            #     # e['pool'].cash_out(e['agent']) / initial_value
            #     # / hdx_price_from_baseline[i][j + block_start]
            #     e['agent'].holdings[('omnipool', 'HDX')] / initial_holdings
            #     for j, e in enumerate(graph_events)
            # ]
            block_numbers = [e['block'] for e in graph_events]
            ax.plot(
                block_numbers, [lp_hold_value[i][j] / lp_hold_value[0][j] for j in range(block_start, block_end + 1)],
                label=f'LP ({strats[i]})', linestyle='--', color=colors[i]
            )
            ax.set_title("HDX shares value relative to LP1 (hold)")

        if st.session_state.show_fee_regions != "none":
            selected = int(st.session_state.show_fee_regions[-1]) - 1
            # highlight regions where fee is above initial fee
            graph_events = scenarios[selected][block_start: block_end + 1]
            region_color = ""
            region_start = graph_events[0]['block']
            for event in graph_events:
                if event['pool'].asset_fee("HDX") > initial_hdx_fee:
                    if region_color != "red":
                        region_color = "red"
                        region_start = event['block']
                else:
                    if region_color == "red":
                        ax.axvspan(region_start, event['block'], color="red", alpha=0.3)
                        region_color = ""
            if region_color == "red":
                ax.axvspan(region_start, graph_events[-1]['block'], color="red", alpha=0.3)
            ax.set_xlabel('block number')
            ax.set_ylabel('relative to initial')
        ax.legend()
        st.pyplot(fig)

        # show HDX price
        fig, ax = plt.subplots(figsize=(16, 6))
        for i, events in enumerate(scenarios):
            graph_events = events[block_start: block_end + 1]
            block_numbers = [e['block'] for e in graph_events]
            ax.plot(
                block_numbers,
                hdx_price_from_baseline[i][block_start: block_end + 1],
                label=f"LP ({strats[i]})", linestyle='--'
            )
            ax.set_title("HDX price from baseline")
        ax.legend()
        st.pyplot(fig)
        pass
    draw_graph()


if st.session_state.regraph:
    plot_scenario()


def run_and_plot():
    run_sim()
    plot_scenario()
