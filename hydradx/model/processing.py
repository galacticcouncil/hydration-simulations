import base64
import datetime
import dateutil.parser
import json
import os
import time
from csv import reader
from zipfile import ZipFile

import requests
from dotenv import load_dotenv
from hydradxapi import HydraDX
from hydradxapi.pallets.stableswap import Pool
from hydradxapi.pallets.fees import Fees
from hydradxapi.pallets.omnipool import AssetState

from .amm.centralized_market import OrderBook, CentralizedMarket
from .amm.global_state import GlobalState, value_assets
from .amm.money_market import MoneyMarket, MoneyMarketAsset, CDP
from .amm.omnipool_amm import OmnipoolState
from .amm.stableswap_amm import StableSwapPoolState
from .amm.omnipool_router import OmnipoolRouter

cash_out = GlobalState.cash_out
impermanent_loss = GlobalState.impermanent_loss
pool_val = GlobalState.pool_val
deposit_val = GlobalState.deposit_val


def postprocessing(events: list, optional_params: list[str] = ()) -> list:
    """
    Definition:
    Compute more abstract metrics from the simulation

    Optional parameters:
    'withdraw_val': tracks the actual value of each agent's assets if they were withdrawn from the pool at each step
    'deposit_val': tracks the theoretical value of each agent's original assets at each step's current spot prices,
        if they had been held outside the pool from the beginning
    'holdings_val': the total value of the agent's outside holdings
    'pool_val': tracks the value of all assets held in the pool
    'impermanent_loss': computes loss for LPs due to price movements in either direction
    """
    # save initial state
    initial_state: GlobalState = events[0]
    withdraw_state: GlobalState = initial_state.copy()

    optional_params = set(optional_params)
    if 'impermanent_loss' in optional_params:
        optional_params.add('deposit_val')
        optional_params.add('withdraw_val')

    agent_params = {
        'deposit_val',
        'withdraw_val',
        'holdings_val',
        'impermanent_loss',
        'token_count',
        'trade_volume'
    }
    exchange_params = {
        'pool_val',
        # 'usd_price'
    }
    unrecognized_params = optional_params.difference(agent_params | exchange_params)
    if unrecognized_params:
        raise ValueError(f'Unrecognized parameter {unrecognized_params}')

    for step in events:
        state: GlobalState = step

        for pool in state.pools.values():
            if 'pool_val' in optional_params:
                pool.pool_val = state.pool_val(pool)

        # agents
        for agent in state.agents.values():
            if 'deposit_val' in optional_params:
                # what are this agent's original holdings theoretically worth at current spot prices?
                agent.deposit_val = value_assets(
                    state.market_prices(agent.holdings),
                    withdraw_state.agents[agent.unique_id].holdings
                )
            if 'withdraw_val' in optional_params:
                # what are this agent's holdings worth if sold?
                agent.withdraw_val = state.cash_out(agent)
            if 'holdings_val' in optional_params:
                agent.holdings_val = sum([quantity * state.price(asset) for asset, quantity in agent.holdings.items()])
            if 'impermanent_loss' in optional_params:
                agent.impermanent_loss = agent.withdraw_val / agent.deposit_val - 1
            if 'token_count' in optional_params:
                agent.token_count = sum(agent.holdings.values())
            if 'trade_volume' in optional_params:
                agent.trade_volume = 0
                if state.time_step > 0:
                    previous_agent = events[state.time_step - 1].agents[agent.unique_id]
                    agent.trade_volume += (
                        sum([
                            abs(previous_agent.holdings[tkn] - agent.holdings[tkn]) * state.price(tkn)
                            for tkn in agent.holdings])
                    )

    return events


def import_binance_prices(
        assets: list[str] or str, start_date: str, days: int=1, interval: int = 12,
        stablecoin: str = 'USDT', return_as_dict: bool = False
) -> dict[str: list[float]]:
    start_date = dateutil.parser.parse(start_date)
    dates = [datetime.datetime.strftime(start_date + datetime.timedelta(days=i), "%Y-%m-%d") for i in range(days)]
    if isinstance(assets, str): assets = [assets]
    # find the data folder
    while not os.path.exists("./data"):
        cwd = os.getcwd()
        os.chdir("..")
        if cwd == os.getcwd():
            raise FileNotFoundError("Could not find the data folder")

    # check that the files are all there, and if not, download them
    for tkn in assets:
        for date in dates:
            file = f"{tkn}{stablecoin}-1s-{date}"
            if os.path.exists(f'./data/{file}.csv'):
                continue
            else:
                print(f'Downloading {file}')
                url = f"https://data.binance.vision/data/spot/daily/klines/{tkn}{stablecoin}/1s/{file}.zip"
                response = requests.get(url)
                with open(f'./data/{file}.zip', 'wb') as f:
                    f.write(response.content)
                with ZipFile(f"./data/{file}.zip", 'r') as zipObj:
                    zipObj.extractall(path='./data')
                os.remove(f"./data/{file}.zip")
                # strip out everything except close price
                with open(f'./data/{file}.csv', 'r') as input_file:
                    rows = input_file.readlines()
                with open(f'./data/{file}.csv', 'w', newline='') as output_file:
                    output_file.write('\n'.join([row.split(',')[1] for row in rows]))

    # now that we have all the files, read them in
    price_data = {tkn: [] for tkn in assets}
    for tkn in assets:
        for date in dates:
            file = f"{tkn}{stablecoin}-1s-{date}"
            with open(f'./data/{file}.csv', 'r') as input_file:
                csvreader = reader(input_file)
                price_data[tkn] += [float(row[0]) for row in csvreader][::interval]

    if not return_as_dict:
        data_length = min([len(price_data[tkn]) for tkn in assets])
        price_data = [
            {tkn: price_data[tkn][i] for tkn in assets} for i in range(data_length)
        ]
    return price_data


def import_monthly_binance_prices(
        assets: list[str], start_month: str, months: int, interval: int = 12, return_as_dict: bool = False
) -> dict[str: list[float]]:
    start_mth, start_year = start_month.split(' ')

    start_date = datetime.datetime.strptime(start_mth + ' 15 ' + start_year, "%b %d %Y")
    dates = [datetime.datetime.strftime(start_date + datetime.timedelta(days=i * 30), "%Y-%m") for i in range(months)]

    # find the data folder
    while not os.path.exists("./data"):
        os.chdir("..")

    # check that the files are all there, and if not, download them
    for tkn in assets:
        for date in dates:
            file = f"{tkn}BUSD-1s-{date}"
            if os.path.exists(f'./data/{file}.csv'):
                continue
            else:
                print(f'Downloading {file}')
                url = f"https://data.binance.vision/data/spot/monthly/klines/{tkn}BUSD/1s/{file}.zip"
                response = requests.get(url)
                with open(f'./data/{file}.zip', 'wb') as f:
                    f.write(response.content)
                with ZipFile(f"./data/{file}.zip", 'r') as zipObj:
                    zipObj.extractall(path='./data')
                os.remove(f"./data/{file}.zip")
                # strip out everything except close price
                with open(f'./data/{file}.csv', 'r') as input_file:
                    rows = input_file.readlines()
                with open(f'./data/{file}.csv', 'w', newline='') as output_file:
                    output_file.write('\n'.join([row.split(',')[1] for row in rows]))

    # now that we have all the files, read them in
    price_data = {tkn: [] for tkn in assets}
    for tkn in assets:
        for date in dates:
            file = f"{tkn}BUSD-1s-{date}"
            with open(f'./data/{file}.csv', 'r') as input_file:
                csvreader = reader(input_file)
                price_data[tkn] += [float(row[0]) for row in csvreader][::interval]

    if not return_as_dict:
        price_data = [
            {tkn: price_data[tkn][i] for tkn in assets} for i in range(len(price_data[assets[0]]))
        ]
    return price_data


def convert_kraken_orderbook(x: dict) -> OrderBook:
    result = x['result']
    ks = list(result.keys())
    if len(ks) > 1:
        raise ValueError('Multiple keys in result')
    k = ks[0]
    ob = x['result'][k]

    ob_obj = OrderBook(
        bids=[[float(bid[0]), float(bid[1])] for bid in ob['bids']],
        asks=[[float(ask[0]), float(ask[1])] for ask in ob['asks']]
    )
    return ob_obj


def convert_binance_orderbook(tkn_pair: tuple, x: dict) -> OrderBook:
    orderbook = x

    ob_obj = OrderBook(
        bids=[[float(bid[0]), float(bid[1])] for bid in orderbook['bids']],
        asks=[[float(ask[0]), float(ask[1])] for ask in orderbook['asks']]
    )
    return ob_obj


def get_orderbooks_from_file(input_path: str) -> dict:
    file_ls = os.listdir(input_path)
    ob_dict = {'kraken': {}, 'binance': {}}
    for filename in file_ls:
        if filename.startswith('kraken_orderbook'):
            tkn_pair = tuple(filename.split('_')[2].split('-'))
            filepath = os.path.join(input_path, filename)
            with open(filepath, newline='') as input_file:
                y = json.load(input_file)
                ob_dict['kraken'][tkn_pair] = convert_kraken_orderbook(y)
        elif filename.startswith('binance_orderbook'):
            tkn_pair = tuple(filename.split('_')[2].split('-'))
            filepath = os.path.join(input_path, filename)
            with open(filepath, newline='') as input_file:
                y = json.load(input_file)
                ob_dict['binance'][tkn_pair] = convert_binance_orderbook(tkn_pair, y)

    return ob_dict


def get_kraken_orderbook(tkn_pair: tuple, archive: bool = False) -> OrderBook:
    orderbook_url = 'https://api.kraken.com/0/public/Depth?pair=' + tkn_pair[0] + tkn_pair[1]
    resp = requests.get(orderbook_url)
    y = resp.json()
    if 'error' in y and y['error']:
        print(y['error'])
    elif archive:
        ts = time.time()
        with open(f'./archive/kraken_orderbook_{tkn_pair[0]}-{tkn_pair[1]}_{ts}.json', 'w') as output_file:
            json.dump(y, output_file)
    return convert_kraken_orderbook(y)


def get_binance_orderbook(tkn_pair: tuple, archive: bool = False) -> OrderBook:
    orderbook_url = 'https://api.binance.com/api/v3/depth?symbol=' + tkn_pair[0] + tkn_pair[1]
    resp = requests.get(orderbook_url)
    y = resp.json()
    if 'msg' in y:
        print(y['msg'])
    elif archive:
        ts = time.time()
        with open(f'./archive/binance_orderbook_{tkn_pair[0]}-{tkn_pair[1]}_{ts}.json', 'w') as output_file:
            json.dump(y, output_file)
    return convert_binance_orderbook(tkn_pair, y)


def get_unique_name(ls: list[str], name: str) -> str:
    if name not in ls:
        return name
    else:
        c = 1
        while name + str(c).zfill(3) in ls:
            c += 1
        return name + str(c).zfill(3)


def get_omnipool_data(rpc: str = 'wss://rpc.hydradx.cloud', archive: bool = False):
    with HydraDX(rpc) as chain:

        asset_list = []
        fees = {}
        tokens = {}
        asset_map = {}

        op_state = chain.api.omnipool.state()

        for asset_id in op_state:
            fee = op_state[asset_id].fees
            decimals = op_state[asset_id].asset.decimals
            symbol = op_state[asset_id].asset.symbol

            tkn = get_unique_name(asset_list, symbol)
            asset_list.append(tkn)
            asset_map[asset_id] = tkn
            tokens[tkn] = {
                'liquidity': op_state[asset_id].reserve / 10 ** decimals,
                'LRNA': op_state[asset_id].hub_reserve / 10 ** 12
            }
            fees[tkn] = {"asset_fee": fee.asset_fee / 100, "protocol_fee": fee.protocol_fee / 100}

    if archive:
        ts = time.time()
        with open(f'./archive/omnipool_data_tokens_{ts}.json', 'w') as output_file:
            json.dump(tokens, output_file)
        with open(f'./archive/omnipool_data_fees_{ts}.json', 'w') as output_file:
            json.dump(fees, output_file)
        with open(f'./archive/omnipool_data_assetmap_{ts}.json', 'w') as output_file:
            json.dump(asset_map, output_file)

    return asset_list, asset_map, tokens, fees


def get_stableswap_data(rpc: str = 'wss://rpc.hydradx.cloud', archive: bool = False) -> dict[int, Pool]:
    with HydraDX(rpc) as chain:
        data = chain.api.stableswap.pools()
    return data


def save_stableswap_data(pools: list[StableSwapPoolState], path: str = './archive/'):
    ts = time.time()
    for state in pools:
        json_state = {
            'tokens': state.liquidity,
            'amplification': state.amplification,
            'precision': state.precision,
            'trade_fee': state.trade_fee,
            'peg': state.peg,
        }
        with open(f'{path}stableswap_data_{state.unique_id}_{ts}.json', 'w') as output_file:
            json.dump(json_state, output_file)


def load_stableswap_data(path: str = './archive/') -> list[StableSwapPoolState]:
    file_ls = os.listdir(path)
    pools = []
    pool_names = ['4-Pool', '2-Pool']
    for filename in file_ls:
        # return with the first likely-looking file
        if filename.startswith('stableswap_data'):
            pool_name = filename.split('_')[2]
            if pool_name in pool_names:
                pool_names.remove(pool_name)
                with open(path + filename, 'r') as input_file:
                    json_state = json.load(input_file)
                pools.append(StableSwapPoolState(
                    tokens=json_state['tokens'],
                    amplification=json_state['amplification'],
                    precision=json_state['precision'],
                    trade_fee=json_state['trade_fee']
                ))
        if len(pool_names) == 0:
            return pools
    raise FileNotFoundError(f'Stableswap data not found: {pool_names}')


def get_omnipool_data_from_file(path: str):
    file_ls = os.listdir(path)
    tokens = {}
    asset_map = {}
    fees = {}
    for filename in file_ls:
        if filename.startswith('omnipool_data'):
            if filename.split('_')[2] == 'tokens':
                with open(os.path.join(path, filename), newline='') as json_file:
                    tokens = json.load(json_file)
            elif filename.split('_')[2] == 'fees':
                with open(os.path.join(path, filename), newline='') as json_file:
                    fees = json.load(json_file)
            elif filename.split('_')[2] == 'assetmap':
                with open(os.path.join(path, filename), newline='') as json_file:
                    asset_map_str = json.load(json_file)
                    # print(asset_map_str)
                    asset_map = {int(k): v for k, v in asset_map_str.items()}

    asset_list = list(asset_map.values())
    return asset_list, asset_map, tokens, fees


def get_current_omnipool_router(rpc='wss://rpc.hydradx.cloud') -> OmnipoolRouter:
    stableswaps = []
    with (HydraDX(rpc) as chain):
        # get omnipool and subpool data
        pool_api = chain.api.omnipool
        entries = pool_api.query_entries(pool_api.MODULE_NAME, pool_api.ASSET_STATE_STORAGE)
        op_state = {}
        for entry in entries:
            asset_id = int(entry[0].value)
            entry = entry[1].value.copy()
            entry["reserve"] = pool_api._asset_reserve(asset_id)
            asset = pool_api._registry.asset_metadata(asset_id)
            # fees = pool_api._fees.asset_fees(asset_id)
            fee_entry = pool_api._fees.query_entry(pool_api._fees.MODULE_NAME, "AssetFee", params=[asset_id])
            try:
                fees = Fees(
                    fee_entry["asset_fee"].value / 10_000,
                    fee_entry["protocol_fee"].value / 10_000,
                    int(fee_entry["timestamp"].value),
                )
            except Exception as e:
                fees = Fees(
                    0.0025, 0.0005, 0
                )
            op_state[asset_id] = AssetState.from_entry(asset, entry, fees)

        # op_state = chain.api.omnipool.state()
        sub_pools = chain.api.stableswap.pools()
        # collect assets
        assets = [(tkn.asset.asset_id, tkn.asset.symbol) for tkn in op_state.values()]
        sub_pool_assets = [(tkn.asset_id, tkn.symbol) for pool in sub_pools.values() for tkn in pool.assets]
        # get a unique symbol for each asset
        symbols = [asset[1] for asset in assets + sub_pool_assets]
        repeats = set([symbol for symbol in symbols if symbols.count(symbol) > 1])
        symbol_map = {
            tkn_id: f"{tkn_name}{tkn_id}" if tkn_name in repeats else tkn_name
            for tkn_id, tkn_name in assets + sub_pool_assets
        }
        omnipool = OmnipoolState(
            tokens={
                symbol_map[tkn_id]: {
                    'liquidity': int(op_state[tkn_id].reserve) / 10 ** op_state[tkn_id].asset.decimals,
                    'LRNA': int(op_state[tkn_id].hub_reserve) / 10 ** 12
                }
                for tkn_id in op_state.keys()  # base_tkn_ids
            },
            asset_fee={
                symbol_map[tkn_id]: op_state[tkn_id].fees.asset_fee / 100 if tkn_id in op_state else 0
                for tkn_id, tkn in [asset for asset in assets]
            },
            lrna_fee={
                symbol_map[tkn_id]: op_state[tkn_id].fees.protocol_fee / 100 if tkn_id in op_state else 0
                for tkn_id, tkn in [asset for asset in assets]
            },
            unique_id='omnipool'
        )
        for pool_id, pool_data in sub_pools.items():
            if pool_data.reserves:
                pool = StableSwapPoolState(
                    tokens={
                        symbol_map[asset.asset_id]: (
                            int(pool_data.reserves[asset.asset_id]) / 10 ** asset.decimals
                            if asset.asset_id in pool_data.reserves else 0
                        ) for asset in pool_data.assets
                    },
                    amplification=float(pool_data.final_amplification),
                    trade_fee=float(pool_data.fee) / 100,
                    unique_id=symbol_map[pool_id] if pool_id in symbol_map else f"stableswap{len(stableswaps):02}",
                    shares=pool_data.shares / 10 ** (op_state[pool_id].asset.decimals if pool_id in op_state else 0)
                )
                stableswaps.append(pool)

    router = OmnipoolRouter([omnipool, *stableswaps])
    return router


def save_omnipool(omnipool_router: OmnipoolRouter, path: str = './archive'):
    ts = time.time()
    omnipool = omnipool_router.exchanges.get('omnipool', None)
    stableswap_pools = []
    for exchange_id in omnipool_router.exchanges:
        if isinstance(omnipool_router.exchanges[exchange_id], StableSwapPoolState):
            stableswap_pools.append(omnipool_router.exchanges[exchange_id])
    with open(os.path.join(path, f'omnipool_savefile_{ts}.json'), 'w+') as output_file:
        json.dump(
            {
                'liquidity': omnipool.liquidity,
                'LRNA': omnipool.lrna,
                'shares': omnipool.shares,
                'protocol_shares': omnipool.protocol_shares,
                'asset_fee': {tkn: omnipool.asset_fee(tkn) for tkn in omnipool.liquidity},
                'lrna_fee': {tkn: omnipool.lrna_fee(tkn) for tkn in omnipool.liquidity},
                'stableswap_pools': [
                    {
                        'tokens': pool.liquidity,
                        'amplification': pool.amplification,
                        'trade_fee': float(pool.trade_fee),
                        'unique_id': pool.unique_id,
                        'shares': pool.shares,
                        'peg': pool.peg[1:]
                    } for pool in stableswap_pools
                ]
            },
            output_file
        )


def load_omnipool(path: str = './archive', filename: str = '') -> OmnipoolRouter:
    if filename:
        file_ls = [filename]
    else:
        file_ls = list(filter(lambda file: file.startswith('omnipool_savefile'), os.listdir(path)))
    for filename in reversed(sorted(file_ls)):  # by default, load the latest first
        with open(os.path.join(path, filename), 'r') as input_file:
            json_state = json.load(input_file)
            # pprint(json_state)
        omnipool = OmnipoolState(
            tokens={
                tkn: {
                    'liquidity': json_state['liquidity'][tkn],
                    'LRNA': json_state['LRNA'][tkn],
                    'shares': json_state['shares'][tkn] if 'shares' in json_state else None,
                    'protocol_shares': json_state['protocol_shares'][tkn] if 'protocol_shares' in json_state else None
                }
                for tkn in json_state['liquidity']
            },
            asset_fee={tkn: float(fee) for tkn, fee in json_state['asset_fee'].items()},
            lrna_fee={tkn: float(fee) for tkn, fee in json_state['lrna_fee'].items()}
        )
        stableswaps = []
        k = 'stableswap_pools' if 'stableswap_pools' in json_state else 'sub_pools'
        for pool in json_state[k]:
            ss = StableSwapPoolState(
                tokens=pool['tokens'],
                amplification=pool['amplification'],
                trade_fee=float(pool['trade_fee']),
                unique_id=pool['unique_id'],
                shares=pool['shares'],
                peg=pool.get('peg', None)
            )
            stableswaps.append(ss)
        router = OmnipoolRouter([omnipool, *stableswaps])
        return router
    raise FileNotFoundError(f'Omnipool file not found in {path}.')


def get_centralized_market(
        config: list,
        exchange_name: str,
        trade_fee: float,
        archive: bool
) -> CentralizedMarket:
    order_books = {}
    for arb_cfg in config:
        exchanges = arb_cfg['exchanges'].keys()
        if exchange_name in exchanges:
            tkn_pair = tuple(arb_cfg['exchanges'][exchange_name])
            if tkn_pair not in order_books:
                if exchange_name == 'kraken':
                    order_books[tkn_pair] = get_kraken_orderbook(tkn_pair, archive=archive)
                elif exchange_name == 'binance':
                    order_books[tkn_pair] = get_binance_orderbook(tkn_pair, archive=archive)
                else:
                    raise ValueError(f"Exchange {exchange_name} not supported")

    return CentralizedMarket(
        unique_id=exchange_name,
        order_book=order_books,
        trade_fee=trade_fee
    )


def convert_config(cfg: list[dict]) -> list[dict]:
    """
    Convert the config from the format used in the UI to the format used in the backend.
    """

    asset_map = {
        100: '4-Pool',
        0: 'HDX',
        10: 'USDT',
        20: 'WETH',
        16: 'GLMR',
        11: 'iBTC',
        14: 'BNC',
        19: 'WBTC',
        15: 'vDOT',
        13: 'CFG',
        5: 'DOT',
        8: 'PHA',
        12: 'ZTG',
        17: 'INTR',
        9: 'ASTR'
    }
    # asset_map = get_omnipool_data("wss://rpc.hydradx.cloud")[1]

    return [
        {
            'exchanges': {
                'omnipool': tuple(asset_map[tkn_id] for tkn_id in cfg_item['tkn_ids']),
                cfg_item['exchange']: cfg_item['order_book']
            },
            'buffer': cfg_item['buffer']
        }
        for cfg_item in cfg
    ]


def load_config(filename, path='archive'):
    with open(os.path.join(path, filename), 'r') as input_file:
        config = json.load(input_file)
    for cfg_item in config:
        for exchange in cfg_item['exchanges']:
            cfg_item['exchanges'][exchange] = tuple(cfg_item['exchanges'][exchange])
    return config


def get_omnipool_balance_history(download_new = False):
    chunk_size = 10000
    chunks_per_file = 10

    def insert_data_chunk(position: int, data_list: list):
        query = f"""
            with hdx_changes as (
              select
                block_id,
                '0' as asset_id,
                (args->>'amount')::numeric as amount
              from event
              where
                name like 'Balances.Transfer'
                and args->>'to' = '0x6d6f646c6f6d6e69706f6f6c0000000000000000000000000000000000000000'
              union all
              select
                block_id,
                '0' as asset_id,
                -(args->>'amount')::numeric as amount
              from event
              where
                name like 'Balances.Transfer'
                and args->>'from' = '0x6d6f646c6f6d6e69706f6f6c0000000000000000000000000000000000000000'
            ),
            tokens_changes as (
              select
                block_id,
                args->>'currencyId' as asset_id,
                (args->>'amount')::numeric as amount
              from event
              where
                name = 'Tokens.Transfer'
                and args->>'to' = '0x6d6f646c6f6d6e69706f6f6c0000000000000000000000000000000000000000'
              union all
              select
                block_id,
                args->>'currencyId' as asset_id,
                -(args->>'amount')::numeric as amount
              from event
              where
                name = 'Tokens.Transfer'
                and args->>'from' = '0x6d6f646c6f6d6e69706f6f6c0000000000000000000000000000000000000000'
              union all
              select
                block_id,
                args->>'currencyId' as asset_id,
                (args->>'amount')::numeric as amount
              from event
              where
                name = 'Tokens.Deposited'
                and args->>'who' = '0x6d6f646c6f6d6e69706f6f6c0000000000000000000000000000000000000000'
              union all
              select
                block_id,
                args->>'currencyId' as asset_id,
                -(args->>'amount')::numeric as amount
              from event
              where
                name = 'Tokens.Withdrawn'
                and args->>'who' = '0x6d6f646c6f6d6e69706f6f6c0000000000000000000000000000000000000000'
            ),
            balance_changes as (
              select * from hdx_changes
              union all
              select * from tokens_changes
            ),
            balance_history as (
              select
                height,
                timestamp,
                block_id,
                asset_id,
                symbol,
                sum(amount) over (partition by asset_id order by block_id) / 10 ^ decimals as balance
              from balance_changes
              inner join block on block_id = block.id
              inner join token_metadata on asset_id = token_metadata.id::text
            )
            select timestamp, symbol, balance as liquidity 
            from balance_history 
            order by timestamp asc 
            limit {chunk_size} offset {chunk_size * position}
            """
        new_data = []
        while not new_data: # and position * chunk_size < len(data_list):
            new_data = query_sqlPad(query)
        # append a line number to each
        new_data = [[position * chunk_size + i] + line for i, line in enumerate(new_data)]
        data_list = data_list[:position * chunk_size] + new_data + data_list[position * chunk_size:]
        return data_list

    def load_history_file(file_name: str):
        with open(f'./{file_name}', 'r') as file:
            file_data = json.loads(file.read())
        return file_data

    def save_history_file(data_list: list, n: int):
        # navigate to right directory
        while not os.path.exists("./model"):
            os.chdir("..")
        os.chdir("model/data/Omnipool Balance History")
        filename = f'./omnipool_history_{str(n).zfill(3)}.json'
        with open(filename, 'w') as file:
            file.write('['+',\n'.join([
                json.dumps(line)
                for line in data_list[chunk_size * chunks_per_file * n: chunk_size * chunks_per_file * (n + 1)]
            ])+']')
        print(f'Saved {filename}')

    def check_errors(data_list):
        errors = []
        for i in range(int(len(data_list) / chunk_size)):
            correctIndex = (i + len(errors)) * chunk_size
            if correctIndex >= len(data_list):
                break
            if data_list[i * chunk_size][0] != correctIndex:
                errors.append(i)
        return errors

    def fix_errors(data_list):
        # error checking and correction
        # this works in the specific case where a piece of chunk_size length failed to download, which is typical
        # other types of errors would require different handling
        errors = check_errors(data_list)
        if not errors:
            print('Data looks error-free.')
        while errors:
            print(f'Error detected at: {errors[0]}')
            data_list = insert_data_chunk(position=errors[0], data_list=data_list)
            errors = check_errors(data_list)

        return data_list

    # navigate to model directory
    while not os.path.exists("./model"):
        os.chdir("..")
    os.chdir("model/data/Omnipool Balance History")

    # load what we have so far
    all_data = []
    file_ls = os.listdir('.')
    file_ls.sort()
    for filename in file_ls:
        if filename.startswith('omnipool_history'):
            print(f'loading {filename}')
            all_data += load_history_file(filename)

    all_data = fix_errors(all_data)
    if not download_new:
        return all_data
    print("Downloading recent transactions...")

    # continue downloading and check for errors
    while True:
        file_number = int(len(all_data) / chunk_size / chunks_per_file)
        start_at = int(len(all_data) / chunk_size)
        for n in range(start_at, (file_number + 1) * chunks_per_file):
            data_length = len(all_data)
            all_data = insert_data_chunk(position=n, data_list=all_data[:n * chunk_size])
            print(f"{len(all_data)} records retrieved.")
            if len(all_data) % chunk_size != 0 or len(all_data) == data_length:
                # probably means we're finished. There might be a better way to detect this, but I think it'll do
                save_history_file(all_data, file_number)
                return all_data

        print(f'saving omnipool_history_{str(file_number).zfill(2)}')
        save_history_file(all_data, file_number)


def query_sqlPad(query: str):
    """
    input: sql query string
    output: result of the query as a list of lists
    requirement: .env file with SQLPAD_USERNAME and PASSWORD in /model/ folder
    """
    # navigate to model directory
    while not os.path.exists("./model"):
        os.chdir("..")
    os.chdir("model")
    load_dotenv()
    username = os.getenv('SQLPAD_USERNAME')
    password = os.getenv('PASSWORD')
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')

    headers = {
        'Authorization': f'Basic {encoded_credentials}',
        'Content-Type': 'application/json'  # This is typically required for JSON payloads
    }

    request = {
        'connectionId': "4a34594e-efa6-4f6e-a594-655ca20f2881",
        'batchText': query
    }

    try:
        response = requests.post(
            url='https://sqlpad.play.hydration.cloud/api/batches',
            headers=headers,
            data=json.dumps(request)
        )

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            data = response.json()
            batchID = data['statements'][0]['batchId']
            try:
                statement = requests.get(
                    url=f'https://sqlpad.play.hydration.cloud/api/batches/{batchID}/statements',
                    headers=headers
                ).json()
                statementID = statement[0]['id']

                print('waiting for query to finish...')
                if response.status_code == 200:
                    data = {'title': 'Not found'}
                    while 'title' in data and data['title'] == 'Not found':
                        data = requests.get(
                            url=f'https://sqlpad.play.hydration.cloud/api/statements/{statementID}/results',
                            headers=headers
                        ).json()
                    return data

            except Exception as e:
                print(f"There was a problem with your request: {str(e)}")
        else:
            print(f"Request failed with status code {response.status_code}")
            return []

    except Exception as e:
        print(f"There was a problem with your request: {str(e)}")


def download_history_files():
    import gdown
    # navigate to model directory
    while not os.path.exists("./model"):
        os.chdir("..")
    os.chdir("model/data")
    if not os.path.exists('Omnipool Balance History'):
        ID = '1pfDcjr5a6kuVUvArk-aX_uXMSL0_QePw'
        gdown.download_folder(id=ID, quiet=False)
        os.chdir('Omnipool Balance History')
        with ZipFile('omnipool_balance_history.zip', 'r') as zipObj:
            zipObj.extractall()
        os.remove('omnipool_balance_history.zip')
        os.chdir('..')
        # the zip file includes a folder, so we need to move the contents up one level
        source_folder = 'Omnipool Balance History/Omnipool Balance History/'
        destination_folder = 'Omnipool Balance History/'

        # Move all files from the inner folder to the outer folder
        for filename in os.listdir(source_folder):
            source_file = os.path.join(source_folder, filename)
            destination_file = os.path.join(destination_folder, filename)
            os.rename(source_file, destination_file)

        # Remove the now-empty inner folder
        os.rmdir(source_folder)


def get_historical_omnipool_balance(tkn, date=None, end_date=None) -> float or dict[str: float]:
    """
    get the balance of a particular token on a particular date or range of dates
    (hopefully) without having to load the entire history or download anything
    but also load or download what's necessary to get the data
    """

    # first make sure we have downloaded the available files
    download_history_files()

    # find the date
    date = dateutil.parser.parse(f"{date}")
    end_date = dateutil.parser.parse(f"{end_date}") if end_date else None
    # navigate to model directory
    while not os.path.exists("./model"):
        os.chdir("..")
    os.chdir("model/data/Omnipool Balance History")

    # load what we have available
    file_data = []
    def find_data_files():
        return list(filter(
            lambda filename: filename.startswith('omnipool_history'),
            os.listdir('.')
        ))
    file_ls = find_data_files()
    file_ls.sort()
    file_name = ''
    # see if the date or block they want is included in existing data
    with open(file_ls[-1]) as f:
        last_line = f.readlines()[-1][:-1]
        last_date = dateutil.parser.parse(json.loads(last_line)[1])
        last_block = json.loads(last_line)[0]
        if date is not None and last_date < date or end_date is not None and last_date < end_date:
            get_omnipool_balance_history()
            file_data = json.load(open(find_data_files()[-1]))
        else:
            for name in file_ls[::-1]:
                file_name = name
                with open(f'./{file_name}') as f:
                    first_line = f.readline()[1:]
                    start_date = dateutil.parser.parse(json.loads(first_line.strip()[:-1])[1])
                    start_block= json.loads(first_line.strip()[:-1])[0]
                    if date is not None and start_date < date:
                        print(f'loading {file_name}')
                        with open(f'./{file_name}', 'r') as file:
                            file_data = json.load(file)
                        break

    tkn_data = [line for line in file_data if line[2] == tkn]
    # find the closest date using a binary search
    left = 0
    right = len(tkn_data) - 1
    while left < right:
        mid = (left + right) // 2
        if date and dateutil.parser.parse(tkn_data[mid][1]) < date :
            left = mid + 1
        else:
            right = mid
    if not end_date:
        print(f"Retrieved balance of {tkn} on {date.strftime('%Y-%m-%d')}: {tkn_data[left][3]}")
        return tkn_data[left][3]
    else:
        # find the end block
        end = left
        while dateutil.parser.parse(tkn_data[end][1]) < end_date:
            end += 1
            if end >= len(tkn_data):
                # load the next file
                file_name = file_ls[file_ls.index(file_name) + 1]
                with open(file_name) as f:
                    tkn_data += [line for line in json.load(f) if line[2] == tkn]
        print(f"Retrieved balance of {tkn} from {date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}.")
        return {data[1]: data[3] for data in tkn_data[left:end]}


def get_current_money_market():
    from substrateinterface import SubstrateInterface
    from .abi.pool_address_provider import POOL_ADDRESS_PROVIDER_ABI
    from .abi.ui_pool_data_provider import UI_POOL_DATA_PROVIDER_ABI, UI_POOL_DATA_PROVIDER_ADDRESS

    # --- Configuration ---
    RPC_URL = "wss://rpc.hydradx.cloud"
    POOL_ADDRESS_PROVIDER_ADDRESS = "0xf3Ba4D1b50f78301BDD7EAEa9B67822A15FCA691"
    BORROWERS_API_ENDPOINT = "https://omniwatch.play.hydration.cloud/api/borrowers/by-health"
    emode_labels = ['None', 'Stablecoins', 'DOT', 'ETH']
    class CustomPoaMiddleware:
        def __init__(self, w3):
            self.w3 = w3

        def wrap_make_request(self, make_request):
            def middleware(method, params):
                response = make_request(method, params)
                result = response.get('result')
                if result and isinstance(result, dict) and 'extraData' in result:
                    # Just ignore extraData validation issues
                    pass
                return response

            return middleware

    def substrate_to_checksum(substrate_address):
        """Converts a Substrate address to an Ethereum checksum address, or returns input if already checksummed"""
        if substrate_address[:2] == '0x':
            try:
                return Web3.to_checksum_address(substrate_address)  # handles hex and checksum
            except:
                return None  # Handles invalid hex.
        try:
            # Use substrateinterface to directly convert to an Ethereum address
            substrate = SubstrateInterface(url="wss://rpc.hydradx.cloud")
            # Use decode_ss58 to get the public key in hex format
            public_key_hex = substrate.ss58_decode(substrate_address)

            # Create Ethereum address from the public key
            address = '0x' + public_key_hex[2:]  # Correctly skip the first byte (0x2a or similar)

            return Web3.to_checksum_address(address)

        except Exception as e:
            print(f"Error converting address {substrate_address}: {e}")
            return None

    from web3 import Web3, LegacyWebSocketProvider

    provider = LegacyWebSocketProvider(RPC_URL)
    w3 = Web3(provider)
    w3.middleware_onion.add(CustomPoaMiddleware)
    if not w3.is_connected():
        raise Exception("Failed to connect to RPC endpoint.")

    address_provider_contract = w3.eth.contract(address=Web3.to_checksum_address(POOL_ADDRESS_PROVIDER_ADDRESS),
                                                abi=POOL_ADDRESS_PROVIDER_ABI)
    pool_address = address_provider_contract.functions.getPool().call()
    pool_address = substrate_to_checksum(pool_address)  # Convert the pool address
    print(f"Pool Address: {pool_address}")
    data_provider_contract = w3.eth.contract(address=UI_POOL_DATA_PROVIDER_ADDRESS, abi=UI_POOL_DATA_PROVIDER_ABI)

    fields = [entry['name'] for entry in UI_POOL_DATA_PROVIDER_ABI[4]['outputs'][0]['components']]

    try:
        raw_data = data_provider_contract.functions.getReservesData(
            Web3.to_checksum_address(POOL_ADDRESS_PROVIDER_ADDRESS),
            # reserves_list
        ).call()
        reserves_data = {
            tkn[2]: {
                fields[i]: tkn[i]
                for i in range(len(fields))
            }
            for tkn in raw_data[0]
        }
        # correct decimals for python
        for tkn in reserves_data:
            reserves_data[tkn]['baseLTVasCollateral'] /= 10000
            reserves_data[tkn]['reserveLiquidationThreshold'] /= 10000
            reserves_data[tkn]['reserveLiquidationBonus'] /= 10000
            reserves_data[tkn]['reserveFactor'] /= 10000
            reserves_data[tkn]['eModeLtv'] /= 10000
            reserves_data[tkn]['eModeLiquidationThreshold'] /= 10000
            reserves_data[tkn]['eModeLiquidationBonus'] /= 10000
            reserves_data[tkn]['liquidityIndex'] /= 1e27
            reserves_data[tkn]['variableBorrowIndex'] /= 1e27
            reserves_data[tkn]['priceInMarketReferenceCurrency'] /= 1e8

    except Exception as e:
        print(f"Error getting reserve list: {e}")
        reserves_data = {}
        return None

    borrowers = []

    try:
        response = requests.get(BORROWERS_API_ENDPOINT)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        borrowers_data = response.json()["borrowers"]
    except requests.exceptions.RequestException as e:
        # raise ConnectionError(f"Error fetching borrowers from API: {e}")
        borrowers_data = []

    # remove duplicated entries
    seen = set()
    borrowers_data_no_dupes = []
    for borrower in borrowers_data:
        k = borrower[0].lower()
        if k in seen:
            print("WARNING: Duplicate borrower detected:", k)
        else:
            seen.add(k)
            borrowers_data_no_dupes.append(borrower)

    for borrower_entry in borrowers_data_no_dupes:
        borrower_info = borrower_entry[1]
        borrower_address = borrower_entry[0]  # borrower_info["account"]
        asset_map = {reserves_data[tkn]['underlyingAsset']: tkn for tkn in reserves_data}

        # Convert to checksummed address *before* passing to Web3.py
        checksummed_borrower_address = substrate_to_checksum(borrower_address)
        user_config = data_provider_contract.functions.getUserReservesData(
            Web3.to_checksum_address(POOL_ADDRESS_PROVIDER_ADDRESS),
            checksummed_borrower_address
        ).call()
        borrowers.append({
            "borrower": checksummed_borrower_address,  # Store the checksummed address
            "config": {
                'assets': {
                    asset_map[tkn[0]]: {
                        # "underlyingAsset": tkn[0],
                        "balance": (
                            tkn[1] * reserves_data[asset_map[tkn[0]]]['liquidityIndex']
                            / 10 ** reserves_data[asset_map[tkn[0]]]['decimals']
                        ),
                        "usageAsCollateralEnabledOnUser": tkn[2],
                        # "stableBorrowRate": tkn[3],
                        "debt": (
                            tkn[4] * reserves_data[asset_map[tkn[0]]]['variableBorrowIndex']
                            / 10 ** reserves_data[asset_map[tkn[0]]]['decimals']
                        ),
                        # "principalStableDebt": tkn[5] / 10 ** reserves_data[asset_map[tkn[0]]]['decimals'],
                        # "stableBorrowLastUpdateTimestamp": tkn[6],
                    }
                    for tkn in user_config[0]
                },
                'e-mode': user_config[1]
            },
            "totalCollateralBase": borrower_info["totalCollateralBase"],
            "totalDebtBase": borrower_info["totalDebtBase"],
            "healthFactor": borrower_info["healthFactor"],
            "availableBorrowsBase": borrower_info["availableBorrowsBase"],
            "currentLiquidationThreshold": borrower_info["currentLiquidationThreshold"],
            "ltv": borrower_info["ltv"],
            "updated": borrower_info["updated"],
            "account": borrower_info["account"],
            # "pool": borrower_info["pool"]
        })


    mm = MoneyMarket(
        assets=[MoneyMarketAsset(
            name=tkn,
            price=reserves_data[tkn]['priceInMarketReferenceCurrency'],
            liquidity=reserves_data[tkn]['availableLiquidity'],
            liquidation_bonus=reserves_data[tkn]['reserveLiquidationBonus'] - 1,
            liquidation_threshold=reserves_data[tkn]['reserveLiquidationThreshold'],
            ltv=reserves_data[tkn]['baseLTVasCollateral'],
            emode_liquidation_bonus=reserves_data[tkn]['eModeLiquidationBonus'] - 1,
            emode_liquidation_threshold=reserves_data[tkn]['eModeLiquidationThreshold'],
            emode_ltv=reserves_data[tkn]['eModeLtv'],
            emode_label=reserves_data[tkn]['eModeLabel'],
        ) for tkn in reserves_data],
        cdps=[CDP(
            debt={
                tkn: position['config']['assets'][tkn]['debt']
                for tkn in position['config']['assets']
                if position['config']['assets'][tkn]['debt'] > 0
            },
            collateral={
                tkn: position['config']['assets'][tkn]['balance']
                for tkn in position['config']['assets']
                if position['config']['assets'][tkn]['usageAsCollateralEnabledOnUser']
            },
            liquidation_threshold=position['currentLiquidationThreshold'],
            health_factor=position['healthFactor'],
            e_mode=emode_labels[position['config']['e-mode']]
        ) for position in borrowers]
    )
    return mm


def get_omnipool_price_history(asset_name: str):
    asset_dict = {
        'HDX': 0,
        'USDT': 10
    }
    if asset_name not in asset_dict:
        raise ValueError("Asset not found")
    else:
        asset_id = asset_dict[asset_name]

    while not os.path.exists('./model'):
        os.chdir('..')
    os.chdir('model')

    endpoint = "https://galacticcouncil.squids.live/hydration-storage-dictionary:omnipool/api/graphql"
    headers = {"Content-Type": "application/json"}

    query = """
    query GetOmnipoolAssetData($first: Int!, $blockID: Int!, $assetId: Int!) {
      omnipoolAssetData(
        first: $first,
        filter: {
          assetId: { equalTo: $assetId },
          paraChainBlockHeight: { greaterThan: $blockID }
        },
        orderBy: PARA_CHAIN_BLOCK_HEIGHT_ASC
      ) {
        edges {
          node {
            paraChainBlockHeight
            balances
            assetState
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
    """

    # Check for existing files
    files = sorted([f for f in os.listdir("./data/prices") if f.lower().startswith(f"{asset_name.lower()} asset state ")])

    start_block_height = 0  # Default to start from the beginning

    all_data = {}
    if files:
        print("Resuming from saved data...")
        for file in files:
            with open(f"./data/prices/{file}", "r") as f:
                saved_data = json.load(f)
            all_data.update(saved_data)

        start_block_height = int(max(all_data.keys()))
        print(f"Found {len(files)} files")
        print(f"Resuming from block height: {start_block_height}")

    variables = {"first": 10000, "blockID": start_block_height, "assetId": asset_id}

    n = len(files) + 1 if files else 1

    while True:
        payload = {"query": query, "variables": variables}
        response = requests.post(endpoint, headers=headers, json=payload)
        response_json = response.json()

        if "errors" in response_json:
            print(f"GraphQL errors: {response_json['errors']}")
            break

        data = response_json["data"]["omnipoolAssetData"]

        balances = {
            d['node']['paraChainBlockHeight']: {
                'liquidity': d['node']['balances']['free'],
                'LRNA': d['node']['assetState']['hubReserve']
            } for d in data['edges']
        }

        print(f"Page {n} fetched: {len(data['edges'])} records")
        with open(f"./data/prices/{asset_name} asset state {n:03}.json", "w") as f:
            f.write(json.dumps(balances))

        all_data.update(balances)

        if data["pageInfo"]["hasNextPage"]:
            last_block_in_page = max(balances.keys())
            variables["blockID"] = int(last_block_in_page)
            n += 1
        else:
            break

    print(f"Total records fetched: {len(all_data)}")


def save_money_market(mm: MoneyMarket, path: str = './archive', filename: str = None):
    if filename is None:
        filename = f"money_market_savefile_{time.time()}.json"
    elif not filename.endswith('.json'):
        filename += '.json'

    json_state = {
        'unique_id': mm.unique_id,
        'close_factor': mm.partial_liquidation_pct,
        'full_liquidation_threshold': mm.full_liquidation_threshold,
        'time_step': mm.time_step,
        'assets': [
            {
                'name': asset.name,
                'emode_label': asset.emode_label,
                'liquidity': asset.liquidity,
                'price': asset.price,
                'supply_cap': asset.supply_cap,
                'liquidation_threshold': asset.liquidation_threshold,
                'liquidation_bonus': asset.liquidation_bonus,
                'ltv': asset.ltv,
                'emode_liquidation_threshold': asset.emode_liquidation_threshold,
                'emode_liquidation_bonus': asset.emode_liquidation_bonus,
                'emode_ltv': asset.emode_ltv
            } for asset in mm.assets.values()
        ],
        'cdps': [
            {
                'debt': {tkn: cdp.debt[tkn] for tkn in cdp.debt},
                'collateral': {tkn: cdp.collateral[tkn] for tkn in cdp.collateral}
            } for cdp in mm.cdps
        ]
    }
    with open(os.path.join(path, filename), 'w') as savefile:
        json.dump(json_state, savefile)


def load_money_market(path: str = './archive', filename: str = None):
    if filename is None:
        file_ls = os.listdir(path)
        for name in file_ls:
            # return with the first likely-looking file
            if name.startswith('money_market_savefile') and name.endswith('.json'):
                filename = name
    elif not(filename.endswith('.json')):
        filename += '.json'
    if filename is None or not os.path.exists(os.path.join(path, filename)):
        raise FileNotFoundError(f'Money market file not found in {path}.')
    with open(os.path.join(path, filename), 'r') as loadfile:
        json_data = json.load(loadfile)
    mm = MoneyMarket(
        unique_id=json_data['unique_id'],
        close_factor=json_data['close_factor'],
        full_liquidation_threshold=json_data['full_liquidation_threshold'],
        assets=[
            MoneyMarketAsset(
                name=asset['name'],
                liquidity=asset['liquidity'],
                supply_cap=asset['supply_cap'],
                price=asset['price'],
                emode_label=asset['emode_label'],
                liquidation_threshold=asset['liquidation_threshold'],
                liquidation_bonus=asset['liquidation_bonus'],
                ltv=asset['ltv'],
                emode_liquidation_threshold=asset['emode_liquidation_threshold'],
                emode_liquidation_bonus=asset['emode_liquidation_bonus'],
                emode_ltv=asset['emode_ltv']
            ) for asset in json_data['assets']
        ],
        cdps=[
            CDP(
                debt={tkn: quantity for tkn, quantity in cdp['debt'].items()},
                collateral={tkn: quantity for tkn, quantity in cdp['collateral'].items()}
            ) for cdp in json_data['cdps']
        ]
    )
    mm.time_step = json_data['time_step']
    print(f"loaded {os.path.join(path, filename)}")
    return mm
