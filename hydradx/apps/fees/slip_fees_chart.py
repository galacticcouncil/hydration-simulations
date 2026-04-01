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

from hydradx.model import production_settings
from hydradx.model.indexer_utils import get_current_omnipool_router

st.markdown("""
    <style>
        .stNumberInput button {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=True)
def load_omnipool_router():
    router = get_current_omnipool_router()
    return router

def run_app():
    st.session_state.router = load_omnipool_router()
    st.session_state.omnipool = st.session_state.router.exchanges['omnipool']
    omnipool = st.session_state.omnipool
    omnipool.asset_fee = 0
    omnipool.lrna_fee = 0
    omnipool.max_lrna_fee = 1
    omnipool.max_asset_fee = 1
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.tkn_buy = st.selectbox("Select token to buy:", options=omnipool.asset_list, index=omnipool.asset_list.index('HDX'))
    with col2:
        st.session_state.tkn_sell = st.selectbox("Select token to sell:", options=omnipool.asset_list, index=omnipool.asset_list.index('DOT'))
    with col3:
        omnipool.slip_factor = st.number_input("Slip factor:", min_value=0.0, max_value=10.0, value=1.0)
    plot_trade_sizes(st.session_state.tkn_buy, st.session_state.tkn_sell, st.session_state.router, st.session_state.omnipool)

def plot_trade_sizes(tkn_buy, tkn_sell, router, omnipool):
    trade_sizes = [10 ** (i / 5) for i in range(0, 26)]
    fees = []
    for trade_size in trade_sizes:
        sell_quantity = trade_size * router.price('Tether', tkn_sell)
        outputs = omnipool.calculate_out_given_in(tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=sell_quantity)
        buy_quantity, delta_qi, delta_qj, asset_fee_total, lrna_fee_total, slip_fee_buy, slip_fee_sell = outputs
        slip_fee_total = slip_fee_buy + slip_fee_sell
        slip_fee_percent = slip_fee_total / -delta_qi
        print(f"${trade_size:.2f} worth of {tkn_sell} sold for {tkn_buy} = {slip_fee_percent * 100:.4f}% slip fee")
        fees.append(slip_fee_percent)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trade_sizes, [fee * 100 for fee in fees], label='Slip Fee %', color='orange')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(f'Value of {tkn_sell} sold for {tkn_buy} (USD)')
    ax.set_xticks([1, 10, 100, 1000, 10000, 100000], ['$1', '$10', '$100', '$1000', '$10k', '$100k'])
    ax.set_ylabel('Slip Fee (%)')
    ax.set_yticks(
        [0.0001, 0.001, 0.01, 0.1, 1] + ([10] if max(fees) * 100 > 1 else []),
        ['0.0001%', '0.001%', '0.01%', '0.1%', '1%'] + (['10%'] if max(fees) * 100 > 1 else [])
    )
    # label the slip fee values at each dollar-value tick
    for i, trade_size in enumerate(trade_sizes):
        if trade_size in [1, 10, 100, 1000, 10000, 100000]:
            ax.text(trade_size, fees[i] * 100, f"{fees[i] * 100:.4f}%", fontsize=8, ha='center', va='bottom')
    ax.set_title('Total Slip Fee')
    ax.grid(True, which="both", ls="--", linewidth=0.5, color="gray")
    ax.legend()
    st.pyplot(fig)

st.set_page_config(layout="wide")
st.title("Omnipool Slip Fees Chart")
run_app()
