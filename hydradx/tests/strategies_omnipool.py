from hypothesis import strategies as st

import hydradx.model.amm.omnipool_amm as oamm
import hydradx.model.amm.stableswap_amm as ssamm
from mpmath import mp, mpf
mp.dps = 50

asset_price_strategy = st.floats(min_value=0.0001, max_value=100000)
asset_price_bounded_strategy = st.floats(min_value=0.1, max_value=10)
asset_number_strategy = st.integers(min_value=3, max_value=5)
arb_precision_strategy = st.integers(min_value=1, max_value=5)
asset_quantity_strategy = st.floats(min_value=100, max_value=10000000)
asset_quantity_bounded_strategy = st.floats(min_value=1000000, max_value=10000000)
percentage_of_liquidity_strategy = st.floats(min_value=0.0000001, max_value=0.10)
reasonable_percentage_of_liquidity_strategy = st.floats(min_value=0.01, max_value=0.10)
fee_strategy = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)
amplification_strategy = st.floats(min_value=1, max_value=10000)


def get_tkn_name(i):
    return f"{'abcdefghijklmnopqrstuvwxyz'[i % 26]}{i // 26}"


@st.composite
def reasonable_holdings(draw, token_count: int = 0):
    token_count = token_count or draw(asset_number_strategy)
    return [draw(asset_quantity_bounded_strategy) for _ in range(token_count)]


@st.composite
def reasonable_market(draw, token_count: int = 0):
    token_count = token_count or draw(asset_number_strategy)
    return [draw(asset_price_bounded_strategy) for _ in range(token_count)]


@st.composite
def reasonable_pct(draw, token_count: int = 0):
    token_count = token_count or draw(asset_number_strategy)
    return [draw(reasonable_percentage_of_liquidity_strategy) for _ in range(token_count)]


@st.composite
def reasonable_market_dict(draw, token_count: int = 0):
    price_list = draw(reasonable_market(token_count))
    price_dict = {'HDX': price_list[1], 'USD': 1.0}
    price_dict.update({
        get_tkn_name(i): price_list[i + 2] for i in
        range(token_count - 2)
    })
    return price_dict


@st.composite
def assets_reasonable_config(draw, token_count: int = 0):
    token_count = token_count or draw(asset_number_strategy)
    usd_price_lrna = draw(asset_price_bounded_strategy)
    return_dict = {
        'HDX': {
            'liquidity': draw(asset_quantity_bounded_strategy),
            'LRNA': draw(asset_quantity_bounded_strategy)
        },
        'USD': {
            'liquidity': draw(asset_quantity_bounded_strategy),
            'LRNA_price': usd_price_lrna
        }
    }
    return_dict.update({
        get_tkn_name(i): {
            'liquidity': draw(asset_quantity_bounded_strategy),
            'LRNA': draw(asset_quantity_bounded_strategy)
        } for i in range(token_count - 2)
    })
    return return_dict


@st.composite
def omnipool_reasonable_config(
        draw,
        asset_dict=None,
        token_count=0,
        lrna_fee=None,
        asset_fee=None,
        tvl_cap_usd=0,
        imbalance=None,
        remove_liquidity_volatility_threshold: float = 0
):
    asset_dict: dict = asset_dict or draw(assets_reasonable_config(token_count))

    test_state = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=tvl_cap_usd or float('inf'),
        asset_fee=draw(st.floats(min_value=0, max_value=0.1)) if asset_fee is None else asset_fee,
        lrna_fee=draw(st.floats(min_value=0, max_value=0.1)) if lrna_fee is None else lrna_fee,
        remove_liquidity_volatility_threshold=remove_liquidity_volatility_threshold,
        withdrawal_fee=True,
        min_withdrawal_fee=0.0001,
    )

    test_state.lrna_imbalance = -draw(asset_quantity_strategy) if imbalance is None else imbalance
    test_state.update()
    return test_state


@st.composite
def assets_config(draw, token_count: int = 0):
    token_count = token_count or draw(asset_number_strategy)
    usd_price_lrna = draw(asset_price_strategy)
    return_dict = {
        'HDX': {
            'liquidity': mpf(draw(asset_quantity_strategy)),
            'LRNA': mpf(draw(asset_quantity_strategy))
        },
        'USD': {
            'liquidity': mpf(draw(asset_quantity_strategy)),
            'LRNA_price': mpf(usd_price_lrna)
        }
    }
    return_dict.update({
        get_tkn_name(i): {
            'liquidity': mpf(draw(asset_quantity_strategy)),
            'LRNA': mpf(draw(asset_quantity_strategy))
        } for i in range(token_count - 2)
    })
    return return_dict


@st.composite
def omnipool_config(
        draw,
        asset_dict=None,
        token_count=0,
        lrna_fee=None,
        asset_fee=None,
        tvl_cap_usd=0,
        withdrawal_fee=True
):
    asset_dict: dict = asset_dict or draw(assets_config(token_count))

    test_state = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=tvl_cap_usd or float('inf'),
        asset_fee=draw(st.floats(min_value=0, max_value=0.1)) if asset_fee is None else asset_fee,
        lrna_fee=draw(st.floats(min_value=0, max_value=0.1)) if lrna_fee is None else lrna_fee,
        withdrawal_fee=withdrawal_fee
    )

    test_state.update()
    return test_state


@st.composite
def stableswap_config(
        draw,
        asset_dict=None,
        token_count: int = None,
        trade_fee: float = None,
        amplification: float = None,
        precision: float = 0.00001,
        unique_id: str = '',
        base_token: str = 'USD'
):
    token_count = token_count or draw(asset_number_strategy)
    asset_dict = asset_dict or {
        f"{base_token}-{'abcdefghijklmnopqrstuvwxyz'[i % 26]}{i // 26}": mpf(draw(asset_quantity_strategy))
        for i in range(token_count)
    }
    test_state = ssamm.StableSwapPoolState(
        tokens=asset_dict,
        amplification=draw(amplification_strategy) if amplification is None else amplification,
        precision=precision,
        trade_fee=draw(st.floats(min_value=0, max_value=0.1)) if trade_fee is None else trade_fee,
        unique_id=unique_id or '/'.join(asset_dict.keys())
    )
    return test_state
