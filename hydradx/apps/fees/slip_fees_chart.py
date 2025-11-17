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


def plot_slip_fees():
    router = get_current_omnipool_router(9370000)
    omnipool = router.exchanges['omnipool']
    omnipool.asset_fee = production_settings.omnipool_asset_fee
    omnipool.lrna_fee = production_settings.omnipool_lrna_fee
    omnipool.current_asset_fee = {tkn: 0 for tkn in omnipool.liquidity}
    omnipool.current_lrna_fee = {tkn: 0 for tkn in omnipool.liquidity}
    omnipool.max_lrna_fee = 1 # disable LRNA fee cap for this test
    trade_sizes = [10 ** (i / 5) for i in range(1, 26)]
    fees = []
    for trade_size in trade_sizes:
        hdx_sell = trade_size * router.price('Tether', 'HDX')
        omnipool.slip_factor = 1.0
        outputs = omnipool.calculate_out_given_in(tkn_buy='LRNA', tkn_sell='HDX', sell_quantity=hdx_sell)
        buy_quantity, asset_fee_total, lrna_fee_total, slip_fee_total = outputs
        slip_fee_percent = slip_fee_total / (buy_quantity + lrna_fee_total + slip_fee_total)
        print(f"${trade_size:.2f} worth of HDX sold = {slip_fee_percent * 100:.4f}% slip fee")
        fees.append(slip_fee_percent)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trade_sizes, [fee * 100 for fee in fees], label='Slip Fee %', color='blue')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Value of HDX Sold')
    ax.set_xticks([1, 10, 100, 1000, 10000, 100000], ['$1', '$10', '$100', '$1000', '$10k', '$100k'])
    ax.set_ylabel('Slip Fee (%)')
    ax.set_yticks([0.0001, 0.001, 0.01, 0.1, 1], ['0.0001%', '0.001%', '0.01%', '0.1%', '1%'])
    ax.set_title('Slip Fees vs Trade Size (Selling HDX for LRNA)')
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend()
    st.pyplot(fig)

st.set_page_config(layout="wide")
st.title("Omnipool Slip Fees Chart")
st.markdown("This chart shows the slip fees incurred when selling HDX for LRNA in the Omnipool.")
plot_slip_fees()
