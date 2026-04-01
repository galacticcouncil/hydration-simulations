import random
from pathlib import Path

import pytest
from hydradx.model.amm.omnipool_amm import OmnipoolState, DynamicFee

import os

from hydradx.model.amm.stableswap_amm import StableSwapPoolState

os.chdir('../..')

from hydradx.model.indexer_utils import get_latest_stableswap_data, get_current_block_height, get_current_omnipool, \
    get_current_omnipool_asset_ids, get_stableswap_pools, get_current_omnipool_router, get_fee_history, \
    get_executed_trades, get_stableswap_liquidity_events, get_fee_pcts, get_omnipool_asset_data, get_omnipool_liquidity, \
    get_blocks_at_timestamps, is_decimal


def test_get_latest_stableswap_data():
    """
    Test the get_latest_stableswap_data function.
    """
    pool_id = 102  # 2-Pool
    block_number = 9100000
    pool_data = get_latest_stableswap_data(pool_id, block_number)
    assert len(pool_data['liquidity']) == 2
    pool_id = 999
    with pytest.raises(IndexError):
        get_latest_stableswap_data(pool_id)


def test_get_current_stableswap_pools():
    from hydradx.model.indexer_utils import get_stableswap_pools
    pools = get_stableswap_pools(block_number=8450000)
    assert len(pools) > 0


def test_get_omnipool_state():
    omnipool = get_current_omnipool(8400000)
    assert isinstance(omnipool, OmnipoolState)
    assert isinstance([omnipool.asset_fee(tkn) for tkn in omnipool.liquidity], list)
    assert isinstance([omnipool.lrna_fee(tkn) for tkn in omnipool.liquidity], list)


def test_get_omnipool_router():
    router = get_current_omnipool_router(8400000)
    assert router is not None


def test_get_fee_history():
    asset_id = 0
    min_block_id = 8400000
    max_block_id = 8400100
    data = get_fee_history(asset_id, min_block_id, max_block_id)
    x = get_fee_pcts(data, asset_id)
    assert x[0][0] == 8400001
    assert 0.0015 < x[0][1] < 0.0016


def test_get_executed_trades():
    asset_ids = ["5", "10"]
    min_block_id = 8400000
    max_block_id = 8401000
    data = get_executed_trades(min_block_id, max_block_id, asset_ids)
    assert data[0]['block_number'] == 8400025
    assert 1920 < data[0]['input_amount'] < 1930


def test_get_stableswap_liquidity_events():
    pool_id = 102
    min_block_id = 8400000
    max_block_id = 8401000
    data = get_stableswap_liquidity_events(pool_id, min_block_id, max_block_id)
    assert data[0]['paraBlockHeight'] == 8400001
    assert data[0]['stableswapAssetLiquidityAmountsByLiquidityActionId']['nodes'][0]['assetId'] == '22'


def test_download_stableswap_exec_prices():
    from hydradx.model.indexer_utils import download_stableswap_exec_prices
    pool_id = 102
    tkn_id = "10"
    min_block_id = 6400000
    max_block_id = min_block_id + 1000

    base_dir = Path(__file__).resolve().parents[1]
    path = str(base_dir / "apps" / "fees" / "data") + "/"

    download_stableswap_exec_prices(pool_id, tkn_id, min_block_id, max_block_id, path)


def test_get_stableswap_pools():
    stableswap_pools = get_stableswap_pools(8400000)
    for pool in stableswap_pools.values():
        assert isinstance(pool, StableSwapPoolState)
    price = stableswap_pools['690'].price(
        tkn='Bifrost Voucher DOT',
        denomination='aDOT'
    )
    if price <= 1:
        raise AssertionError("vDOT should be priced higher than aDOT.")

def test_download_omnipool_spot_prices():
    from hydradx.model.indexer_utils import download_omnipool_spot_prices
    denom_id = "102"
    # tkn_id = 1000765  # tBTC
    tkn_id = "1000624"  # AAVE
    min_block_id = 7111661
    max_block_id = min_block_id + 1000
    step_size = 50000

    base_dir = Path(__file__).resolve().parents[1]
    path = str(base_dir / "apps" / "fees" / "data") + "/"

    temp_max_block_id = min(min_block_id + step_size, max_block_id)
    while min_block_id <= max_block_id:
        download_omnipool_spot_prices(tkn_id, denom_id, min_block_id, temp_max_block_id, path)
        min_block_id = temp_max_block_id + 1
        temp_max_block_id = min(temp_max_block_id + step_size, max_block_id)


def test_get_omnipool_swap_fees():
    from hydradx.model.indexer_utils import get_omnipool_swap_fees
    tkn_id = "5"
    min_block_id = 8000000
    max_block_id = 8001000

    asset_fee_data, hub_fee_data = get_omnipool_swap_fees(tkn_id, min_block_id, max_block_id)
    assert len(asset_fee_data) > 0, "Asset fee data should not be empty"
    assert len(hub_fee_data) > 0, "Hub fee data should not be empty"


def test_get_asset_info_by_ids():
    from hydradx.model.indexer_utils import get_asset_info_by_ids
    ids = get_asset_info_by_ids()
    assert isinstance(ids, dict)
    assert len(ids) > 0, "Asset info should not be empty"


def test_download_acct_trades():
    from hydradx.model.indexer_utils import download_acct_trades
    tkn_id = "5"
    base_dir = Path(__file__).resolve().parents[1]
    path = str(base_dir / "apps" / "fees") + "/"
    acct = "0x7279fcf9694718e1234d102825dccaf332f0ea36edf1ca7c0358c4b68260d24b"
    download_acct_trades(tkn_id, acct, path, 7000000, 7001000)


def test_bucket_values():
    from hydradx.model.indexer_utils import bucket_values

    min_block = 0
    max_block = 99
    values = []
    blocks = []
    for i in range(min_block, max_block + 1):
        values.append(random.randint(0, 1000))
        blocks.append(random.randint(min_block, max_block))
    blocks[0] = min_block
    blocks[1] = max_block
    data = list(zip(blocks, values))

    results1 = bucket_values(10, data)
    results2 = bucket_values(15, data)
    results3 = bucket_values(20, data)

    # different bucket counts should not affect sum of values
    s1 = sum([x['value'] for x in results1])
    s2 = sum([x['value'] for x in results2])
    s3 = sum([x['value'] for x in results3])
    assert s1 == s2 == s3, f"Sum of values should be equal, got {s1}, {s2}, {s3}"

    for i in range(len(results1)):
        s1 = results3[2*i]['value'] + results3[2*i + 1]['value']
        s2 = results1[i]['value']
        assert s1 == s2, f"Sum of values in bucket {i} should be equal, got {s1} and {s2}"


def test_get_current_omnipool_assets():
    ids_str = get_current_omnipool_asset_ids()
    ids = [int(x) for x in ids_str if is_decimal(x)]
    print(ids)
    assert 0 in ids


def test_get_current_omnipool_fees():
    from hydradx.model.indexer_utils import get_current_omnipool_fees
    asset_fees, hub_fees = get_current_omnipool_fees()
    assert isinstance(asset_fees, DynamicFee) and isinstance(hub_fees, DynamicFee)
    assert 'HDX' in asset_fees.current
    assert 'HDX' in hub_fees.current


def test_get_omnipool_asset_data():
    data = get_omnipool_asset_data(min_block_id=8400000, max_block_id=8401000)


def test_get_omnipool_trades():
    from hydradx.model.indexer_utils import get_omnipool_trades
    current_block = get_current_block_height()
    data = get_omnipool_trades(min_block=current_block - 10000, max_block=current_block - 9000)
    pass

