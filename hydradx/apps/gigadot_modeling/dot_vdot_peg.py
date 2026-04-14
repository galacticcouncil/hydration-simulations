import copy
import math

from matplotlib import pyplot as plt
import sys, os
import streamlit as st
from mpmath import mpf

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.stableswap_amm import StableSwapPoolState, simulate_swap
from hydradx.model.amm.agents import Agent
from hydradx.apps.gigadot_modeling.utils import get_omnipool_minus_vDOT, set_up_gigaDOT_3pool, set_up_gigaDOT_2pool, \
    create_custom_scenario, simulate_route, get_slippage_dict
from hydradx.apps.display_utils import display_liquidity, display_op_and_ss
from hydradx.model.indexer_utils import get_latest_stableswap_data

peg = st.number_input(
    label="peg value:",
    value=1.578818,
    step=0.000001, format="%.6f"
)
pool = StableSwapPoolState(
    tokens={'DOT': 1, 'vDOT': 1},
    amplification=222,
    trade_fee=0,
    peg=peg
)
test_cases = [20, 30, 40, 50, 60, 70, 80]
for balance in test_cases:
    pool.liquidity['DOT'] = mpf(balance)
    pool.liquidity['vDOT'] = mpf(100 - balance) / peg
    price = pool.price('vDOT', 'DOT')
    st.write(f"price at {balance}% DOT, {100 - balance}% vDOT: {round(price, 6)}")
