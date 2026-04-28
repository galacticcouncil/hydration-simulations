import copy
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
import sys, os, math
import streamlit as st
import time, datetime
from pathlib import Path

from streamlit import form_submit_button

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.omnipool_router import OmnipoolRouter
from hydradx.model.amm.money_market import MoneyMarket, MoneyMarketAsset, CDP
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.trade_strategies import liquidate_cdps, schedule_swaps, omnipool_arbitrage, sell_all
from hydradx.model.indexer_utils import get_current_money_market
from hydradx.model.processing import save_money_market, load_money_market, \
     save_state, load_state
from hydradx.model.indexer_utils import get_current_omnipool_router, get_current_block_height

def load_market_data():
    path = Path(__file__).parent / "archive"
    try:
        money_market = load_money_market(path=path)
    except FileNotFoundError:
        money_market = get_current_money_market()
        save_money_market(money_market, path=path)

    try:
        omnipool_router = load_state(path=path)
    except FileNotFoundError:
        omnipool_router = get_current_omnipool_router()
        save_state(omnipool_router)
    return money_market, omnipool_router


def simulate_husde_depeg(money_market: MoneyMarket, omnipool_router: OmnipoolRouter):
    # simulate a depeg event where the price of hUSDE drops to 0.5 USD
    # we assume that the price of hUSDE in the omnipool is determined by the ratio of hUSDE to USDC
    # and that the price of USDC is stable at 1 USD
    # we also assume that the stableswap pool has a peg of 1:1 between hUSDE and USDC

    # set up initial state
    initial_husde_price = 1.0
    depeg_price = 0.99
    affected_cdps = [cdp for cdp in money_market.cdps if '2-Pool-HUSDe' in cdp.collateral]
    sum([money_market.value_assets(cdp.debt) for cdp in affected_cdps])
