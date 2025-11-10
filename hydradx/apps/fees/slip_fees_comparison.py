import random

from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import sys, os
import streamlit as st
import copy

from matplotlib.lines import lineStyles
from streamlit import session_state

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

import hydradx.model.production_settings as settings
from hydradx.model.amm.omnipool_amm import OmnipoolState, DynamicFee
from hydradx.model.amm.agents import Agent
from hydradx.model.indexer_utils import get_omnipool_liquidity, get_omnipool_trades, get_current_block_height
from hydradx.model.amm.trade_strategies import schedule_swaps

st.markdown("""
    <style>
        .stNumberInput button {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)
st.set_page_config(layout="wide")

st.session_state.setdefault("events", [])
st.session_state.setdefault("sim_length", 20)
st.session_state.setdefault("omnipool", None)
st.session_state.setdefault("trades", {})
st.session_state.setdefault("regraph", False)
st.session_state.setdefault("lp_pct", 0.25)
st.session_state.setdefault("show_fee_regions", "none")
st.session_state.setdefault("show_scenario_1", True)
st.session_state.setdefault("show_scenario_2", True)
st.session_state.setdefault("show_scenario_3", True)
st.session_state.setdefault("show_scenario_4", True)


def run_sim():
    sim_length = st.session_state.sim_length
    initial_omnipool = OmnipoolState(
        tokens={
            "HDX": {"liquidity": 10000000, "LRNA": 5000},
            "USD": {"liquidity": 200000, "LRNA": 10000},
        },
        asset_fee=DynamicFee(
            minimum=settings.omnipool_asset_fee_minimum,
            maximum=settings.omnipool_asset_fee_maximum,
            amplification=settings.omnipool_asset_fee.amplification,
            decay=settings.omnipool_asset_fee_decay
        ),
        lrna_fee=DynamicFee(
            minimum=settings.omnipool_lrna_fee_minimum,
            maximum=settings.omnipool_asset_fee_maximum, # note: using asset fee maximum here
            amplification=settings.omnipool_lrna_fee.amplification,
            decay=settings.omnipool_lrna_fee_decay
        ),
        withdrawal_fee=True,
        unique_id="omnipool",
        preferred_stablecoin="USD",
    )
    st.session_state.omnipool = initial_omnipool
    initial_hdx_price = initial_omnipool.lrna_price("HDX")

    scenarios = []  # 0-1: price down, 2-3: price up, 4-5: volatile, 6-7: random
    with st.spinner("Running simulation..."):
        for i in range(1, 9):
            if i > 6:
                random.seed(42)
                price_progression = [initial_hdx_price] + [
                    initial_hdx_price * (1 + random.random() / 50 - 0.005)
                    for i in range(1, sim_length + 1)
                ]
            elif i > 4:
                price_progression = [initial_hdx_price] + [
                    initial_hdx_price * 1.01 if i % 2 == 0 else initial_hdx_price * 0.99
                    for i in range(1, sim_length + 1)
                ]
            elif i > 2:
                price_progression = [initial_hdx_price] + [
                    initial_hdx_price * (1 + 0.25 / sim_length * i) for i in range(1, sim_length + 1)
                ]
            else:
                price_progression = [initial_hdx_price] + [
                    initial_hdx_price * (1 - 0.2 / sim_length * i) for i in range(1, sim_length + 1)
                ]
            scenarios.append([])
            events = scenarios[i - 1]
            omnipool = initial_omnipool.copy()
            if i % 2 == 0:
                omnipool.slip_factor = 2
                omnipool.min_lrna_fee = 0
            else:
                omnipool.slip_factor = 0
                omnipool.min_lrna_fee = settings.omnipool_lrna_fee_minimum

            trade_agent = Agent(
                enforce_holdings=False,
                unique_id="trader",
            )
            for block in range(sim_length + 1):
                omnipool.update()
                feeless_pool = omnipool.copy()
                feeless_pool.asset_fee = 0
                feeless_pool.lrna_fee = 0
                feeless_pool.slip_factor = None
                agent_start_usd = trade_agent.get_holdings("USD")
                agent_start_hdx = trade_agent.get_holdings("HDX")
                target_price = price_progression[block] if block < sim_length else price_progression[-1]
                omnipool.trade_to_price(
                    agent=trade_agent,
                    tkn="HDX",
                    target_price=target_price,
                )
                if trade_agent.get_holdings("LRNA") < 0:
                    omnipool.swap(trade_agent, tkn_buy="LRNA", tkn_sell="USD", buy_quantity=-trade_agent.holdings["LRNA"])
                else:
                    omnipool.swap(trade_agent, tkn_sell="LRNA", tkn_buy="USD", sell_quantity=trade_agent.get_holdings("LRNA"))
                agent_end_usd = trade_agent.get_holdings("USD")
                agent_end_hdx = trade_agent.get_holdings("HDX")
                feeless_agent = Agent(enforce_holdings=False)

                if agent_end_hdx > agent_start_hdx:
                    feeless_pool.swap(
                        agent=feeless_agent,
                        tkn_sell="USD",
                        tkn_buy="HDX",
                        buy_quantity=agent_end_hdx - agent_start_hdx,
                    )
                    usd_sold = -feeless_agent.get_holdings("USD")
                    fee = 1 - usd_sold / (agent_start_usd - agent_end_usd)
                elif agent_start_hdx > agent_end_hdx:
                    feeless_pool.swap(
                        agent=feeless_agent,
                        tkn_sell="HDX",
                        tkn_buy="USD",
                        sell_quantity=agent_start_hdx - agent_end_hdx,
                    )
                    usd_bought = feeless_agent.get_holdings("USD")
                    fee = 1 - (agent_end_usd - agent_start_usd) / usd_bought
                else:
                    fee = 0.0

                events.append({
                    'pool': omnipool.copy(),
                    'agent': trade_agent.copy(),
                    'block': block,
                    'fee': fee,
                })
    st.session_state.events = scenarios
    st.session_state.regraph = True
    print("simulation complete.")


controls_col, _, text_col = st.columns([3, 1, 6])
with controls_col:
    label_col, input_col = st.columns([3, 1], vertical_alignment="center")
    with label_col:
        st.write("simulate # of blocks:")
    with input_col:
        st.number_input(
            "sim length", min_value=5, max_value=250, key="sim_length",
            label_visibility="collapsed"
        )
with text_col:
    st.markdown(f"""
        Comparing dynamic fee alone vs dynamic fee + slip fee. The minimums and maximums are the same,
        but the response mechanism is different.
    """)
#
# with controls_col:
#     st.button("run simulation", on_click=run_sim, use_container_width=True)


# @st.fragment
def plot_scenario():
    initial_omnipool = st.session_state.omnipool
    scenarios = st.session_state.events
    print("plotting scenario...")
    colors = ['blue', 'orange', 'red', 'purple', 'green', 'brown', 'pink', 'gray']

    for scenario_start in range(0, len(scenarios), 2):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
        # plot hdx price in first two scenarios
        for i, events in enumerate(scenarios[scenario_start: scenario_start + 2]):
            if (i == 0 and st.session_state.show_scenario_1) or (i == 1 and st.session_state.show_scenario_2):
                prices = [e['pool'].usd_price('HDX') for e in events]
                ax1.plot(
                    prices, color=colors[i + scenario_start], linestyle='dashed' if i == 1 else 'solid',
                    label=["dynamic fees", "slip fees + dynamic fees"][i]
                )
        ax1.set_ylabel('HDX Price (USD)')
        ax1.set_xlabel('Blocks since start')
        ticks_list = list(range(0, st.session_state.sim_length + 1, 5)) + [st.session_state.sim_length]
        ax1.set_xticks(ticks_list)
        ax1.set_xticklabels([str(tick) for tick in ticks_list])

        # plot hdx fees
        for i, events in enumerate(scenarios[scenario_start: scenario_start + 2]):
            if (i == 0 and st.session_state.show_scenario_1) or (i == 1 and st.session_state.show_scenario_2):
                fees = [e['fee'] for e in events]
                ax2.plot(
                    fees, color=colors[i + scenario_start], linestyle='dashed' if i == 1 else 'solid',
                    label = ["dynamic fees", "slip fees + dynamic fees"][i]
                )
        ax2.set_ylabel('Trade fee total (%)')
        ax2.legend()
        ax2.set_xticks(ticks_list)
        ax2.set_xticklabels([str(tick) for tick in ticks_list])
        st.pyplot(fig)

# if st.session_state.regraph:
#     plot_scenario()


def run_and_plot():
    run_sim()
    plot_scenario()

run_and_plot()