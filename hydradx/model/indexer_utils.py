import json
import requests

from hydradx.model.amm.omnipool_amm import OmnipoolState, DynamicFee
from hydradx.model.amm.omnipool_router import OmnipoolRouter
from hydradx.model.amm.stableswap_amm import StableSwapPoolState


URL_UNIFIED_PROD = 'https://galacticcouncil.squids.live/hydration-pools:unified-prod/api/graphql'
URL_OMNIPOOL_STORAGE = 'https://galacticcouncil.squids.live/hydration-storage-dictionary:omnipool-v2/api/graphql'
URL_STABLESWAP_STORAGE = 'https://galacticcouncil.squids.live/hydration-storage-dictionary:stablepool-v2/api/graphql'
URL_GENERIC_DATA = 'https://galacticcouncil.squids.live/hydration-storage-dictionary:generic-data-v2/api/graphql'

class AssetInfo:
    def __init__(
            self,
            asset_type: str = None,
            decimals: int = None,
            existential_deposit: int = None,
            id: str = None,
            is_sufficient: bool = None,
            name: str = None,
            symbol: str = None,
            xcm_rate_limit = None
    ):
        self.asset_type = asset_type
        self.decimals = decimals
        self.existential_deposit = existential_deposit
        self.id = id
        self.is_sufficient = is_sufficient
        self.name = name
        self.symbol = symbol
        self.xcm_rate_limit = xcm_rate_limit


def query_indexer(url: str, query: str, variables: dict = None) -> dict:
    response = requests.post(url, json={'query': query, 'variables': variables})
    while response.status_code == 504:
        print("504 error, retrying...")
        response = requests.post(url, json={'query': query, 'variables': variables})
    if response.status_code != 200:
        raise ValueError(f"Query failed with status code {response.status_code}")
    return_val = response.json()
    if 'errors' in return_val:
        raise ValueError(return_val['errors'][0]['message'])
    return return_val


def get_asset_info_by_ids(asset_ids: list = None) -> dict[str: AssetInfo]:

    asset_query = f"""
    query assetInfoByAssetIds{'($assetIds: [String!]!)' if asset_ids else ''} {{
        assets(filter: {{
            {'id: {in: $assetIds}, ' if asset_ids else ''}
            symbol: {{notEqualTo: "none"}}
        }}) {{
            nodes {{
                assetType
                decimals
                id
                isSufficient
                name
                symbol
                xcmRateLimit
            }}
        }}
    }}
    """

    # Send POST request to the GraphQL API
    variables = {'assetIds': [f"{asset_id}" for asset_id in asset_ids]} if asset_ids else None
    return_val = query_indexer(URL_OMNIPOOL_STORAGE, asset_query, variables)
    asset_data = return_val['data']['assets']['nodes']
    dict_data = {}
    for asset in asset_data:
        asset_info = AssetInfo(
            asset_type=asset['assetType'],
            decimals=int(asset['decimals'] if asset['decimals'] else 0),
            id=asset['id'],
            is_sufficient=asset['isSufficient'],
            name=asset['name'],
            symbol=asset['symbol']
        )
        dict_data[asset['id']] = asset_info
    return dict_data


def get_omnipool_asset_data(
        min_block_id: int,
        max_block_id: int,
        asset_ids: list[str] or list[int] = None
) -> list:

    variables: dict[str, object] = {
        "minBlock": min_block_id,
        "maxBlock": max_block_id
    }

    query = f"""
    query AssetBalancesByBlockHeight($first: Int!, $after: Cursor, $minBlock: Int!, $maxBlock: Int!{
    ', $assetIds: [Int!]!' if asset_ids else ''
    }) {{
      omnipoolAssetData(
        first: $first,
        after: $after,
        orderBy: PARA_BLOCK_HEIGHT_ASC,
        filter: {{
          paraBlockHeight: {{ greaterThanOrEqualTo: $minBlock, lessThanOrEqualTo: $maxBlock }} {
            'assetId: { in: $assetIds }' if asset_ids else ''
          }
        }}
      ) {{
        nodes {{
          paraBlockHeight
          assetId
          balances
          assetState
        }}
        pageInfo {{
          hasNextPage
          endCursor
        }}
      }}
    }}
    """

    data_all = []
    has_next_page = True
    after_cursor = None
    page_size = 2000
    variables["first"] = page_size
    if asset_ids:
        variables['assetIds'] = [int(asset_id) for asset_id in asset_ids]

    i = 0
    while has_next_page:
        print(f'Fetching page {i}...')
        variables["after"] = after_cursor
        data = query_indexer(URL_OMNIPOOL_STORAGE, query, variables)
        print(f'Processing page {i}...')
        page_data = data['data']['omnipoolAssetData']['nodes']
        data_all.extend(page_data)
        page_info = data['data']['omnipoolAssetData']['pageInfo']
        has_next_page = page_info['hasNextPage']
        after_cursor = page_info['endCursor']
        i += 1
    return data_all


def get_omnipool_data_by_asset(
        min_block_id: int,
        max_block_id: int,
        asset_ids: list = None,
        validate: bool = True
) -> dict:
    data = get_omnipool_asset_data(min_block_id, max_block_id, asset_ids)
    data_by_asset = {}
    for item in data:
        if item['assetId'] not in data_by_asset:
            data_by_asset[item['assetId']] = []
        elif validate:
            last_block_id = data_by_asset[item['assetId']][-1]['paraBlockHeight']
            next_block_id = item['paraBlockHeight']
            assert last_block_id + 1 == next_block_id
        data_by_asset[item['assetId']].append(item)
    return data_by_asset


def get_current_block_height():
    url = 'https://galacticcouncil.squids.live/hydration-pools:unified-prod/api/graphql'

    latest_block_query = """
        query BlockHeight {
            blocks(last: 1) {
                nodes {
                    height
                }
            }
        }
    """

    data = query_indexer(url, latest_block_query)
    latest_block = data['data']['blocks']['nodes'][0]['height']
    return latest_block


def get_stableswap_asset_data(
        pool_id: int,
        min_block_id: int,
        max_block_id: int
) -> list:

    stableswap_query = """
    query MyQuery($pool_id: Int!, $min_block: Int!, $max_block: Int!) {
      stableswaps(
        orderBy: PARA_BLOCK_HEIGHT_ASC,
        filter: {poolId: {equalTo: $pool_id}, paraBlockHeight: {greaterThanOrEqualTo: $min_block, lessThanOrEqualTo: $max_block}}
      ) {
        nodes {
          id
          poolId
          stableswapAssetDataByPoolId {
            nodes {
              assetId
              balances
              id
            }
          }
          fee
          finalAmplification
          finalBlock
          initialAmplification
          initialBlock
        }
      }
    }
    """
    variables = {"pool_id": pool_id, "min_block": min_block_id, "max_block": max_block_id}
    data = query_indexer(URL_STABLESWAP_STORAGE, stableswap_query, variables)
    return data['data']['stableswaps']['nodes']


def get_stableswap_data_by_block(
        pool_id: int,
        block_no: int
):
    return get_stableswap_asset_data(pool_id, block_no, block_no)[0]


def get_latest_stableswap_data(
        pool_id: int,
        block_number: int = None
):
    if block_number is None:
        latest_block_query = """
        query MaxHeightQuery {
          maxHeightResult: stableswaps(first: 1, orderBy: PARA_BLOCK_HEIGHT_DESC) {
            nodes {
              paraBlockHeight
            }
          }
        }
        """

        data = query_indexer(URL_STABLESWAP_STORAGE, latest_block_query)

        block_number = int(data['data']['maxHeightResult']['nodes'][0]['paraBlockHeight'])
    pool_data = get_stableswap_data_by_block(pool_id, block_number)
    pool_data_formatted = {
        "pool_id": pool_data['poolId'],
        "block": block_number,
        'fee': pool_data['fee'] / 1000000,
        'finalAmplification': pool_data['finalAmplification'],
        'finalBlock': pool_data['finalBlock'],
        'initialAmplification': pool_data['initialAmplification'],
        'initialBlock': pool_data['initialBlock'],
        'liquidity': {}
    }

    asset_id_list = [asset['assetId'] for asset in pool_data['stableswapAssetDataByPoolId']['nodes']]
    asset_dict = get_asset_info_by_ids(asset_id_list)

    for asset in pool_data['stableswapAssetDataByPoolId']['nodes']:
        asset_id = asset['assetId']
        balance = int(asset['balances']['d'][0]) / (10 ** asset_dict[asset_id].decimals)
        pool_data_formatted['liquidity'][asset_id] = balance
    return pool_data_formatted


def get_current_omnipool_assets() -> list[str]:
    query = """
    query assetInfoByAssetIds {
      omnipoolAssets(filter: {isRemoved: {equalTo: false}}) {
        nodes {
          assetId
        }
      }
    }
    """
    data = query_indexer(URL_UNIFIED_PROD, query)
    ids = [node['assetId'] for node in data['data']['omnipoolAssets']['nodes']]
    for i in ids:
        try:
            int(i)
        except ValueError:
            ids.remove(i)
    # hub token doesn't count
    ids.remove('1')
    return ids


def get_stableswap_ids():

    stableswap_query = """
    query MyQuery {
      stableswaps {
        groupedAggregates(groupBy: POOL_ID) {
          keys
        }
      }
    }
    """

    data = query_indexer(URL_STABLESWAP_STORAGE, stableswap_query)
    pool_ids = [int(pool['keys'][0]) for pool in data['data']['stableswaps']['groupedAggregates']]
    return pool_ids


def get_fee_history(asset_id: int, min_block: int, max_block: int = None):

    if max_block is None:
        latest_block_query = """
        query MaxHeightQuery {
          maxHeightResult: swaps(first: 1, orderBy: PARA_BLOCK_HEIGHT_DESC) {
            nodes {
              paraBlockHeight
            }
          }
        }
        """
        data = query_indexer(URL_UNIFIED_PROD, latest_block_query)
        max_block = int(data['data']['maxHeightResult']['nodes'][0]['paraBlockHeight'])

    fee_query = """
    query fees_query($first: Int!, $after: Cursor, $assetId: String!, $minBlock: Int!, $maxBlock: Int!) {
      swaps(
        first: $first,
        after: $after,
        filter: {
          allInvolvedAssetIds: {contains: [$assetId, "1"]},
          paraBlockHeight:{greaterThanOrEqualTo: $minBlock, lessThanOrEqualTo: $maxBlock}
        }
        orderBy: PARA_BLOCK_HEIGHT_ASC
      ) {
          nodes {
            id
            swapOutputs {
              nodes {
                amount
                assetId
              }
            }
            swapInputs {
              nodes {
                amount
                assetId
              }
            }
            swapFees {
              nodes {
                amount
                assetId
                recipientId
              }
            }
            paraBlockHeight
          }
          pageInfo {
            hasNextPage
            endCursor
          }
        }
    }
    """

    variables = {"assetId": str(asset_id), "minBlock": min_block, "maxBlock": max_block}
    # data = query_indexer(url, hdx_fee_query, variables)

    data_all = []
    has_next_page = True
    after_cursor = None
    page_size = 10000
    variables["first"] = page_size

    while has_next_page:
        variables["after"] = after_cursor
        data = query_indexer(URL_UNIFIED_PROD, fee_query, variables)
        page_data = data['data']['swaps']['nodes']
        data_all.extend(page_data)
        page_info = data['data']['swaps']['pageInfo']
        has_next_page = page_info['hasNextPage']
        after_cursor = page_info['endCursor']

    return data_all


def get_fee_pcts(data, asset_id):
    fee_pcts = [
        [int(x['paraBlockHeight']),
        (sum([int(y['amount']) for y in x['swapFees']['nodes'] if y['assetId'] == str(asset_id)])
         / int(x['swapOutputs']['nodes'][0]['amount']))]
        for x in data if x['swapOutputs']['nodes'][0]['assetId'] == str(asset_id)
    ]
    return fee_pcts


def get_current_stableswap_pools(block_number):
    all_assets = get_asset_info_by_ids()
    stableswap_ids = [asset.id for asset in all_assets.values() if asset.asset_type == 'StableSwap']
    stableswap_asset_query = """
        query assetInfoByAssetIds {
            stableswapAssets {
                nodes {
                    pool {
                        id
                    }
                    asset {
                        name
                        symbol
                        id
                        decimals
                    }
                }
            }
        }
    """
    stableswap_query_data = query_indexer(URL_UNIFIED_PROD, stableswap_asset_query)['data']['stableswapAssets']['nodes']
    stableswap_asset_data = {}
    for node in sorted(
        stableswap_query_data,
        key=lambda n: n['asset']['id']
    ):
        asset_id = str(node['asset']['id'])
        if asset_id not in stableswap_asset_data:
            stableswap_asset_data[asset_id] = {
                'pool_id': [node['pool']['id']],
                'decimals': node['asset']['decimals'],
                'symbol': node['asset']['symbol'],
                'name': node['asset']['name']
            }
        else:
            # same asset can be in more than one pool
            stableswap_asset_data[asset_id]['pool_id'].append(node['pool']['id'])

    stableswap_pools = {
        str(asset.id): StableSwapPoolState(
            unique_id=asset.name,
            tokens={node['name']: 0 for node in stableswap_asset_data.values() if asset.id in node['pool_id']},
            amplification=0
        ) for asset in get_asset_info_by_ids(stableswap_ids).values()
    }

    current_block = block_number
    max_queries = 20
    blocks_per_query = 100
    queries = 0
    asset_ids_remaining = list(stableswap_asset_data.keys())
    while len(asset_ids_remaining) > 0 and queries < max_queries:
        query = f"""
            query StableSwapPoolData {{
                stableswapAssetData(
                    filter: {{
                        paraBlockHeight: {{
                            lessThanOrEqualTo: {current_block}, 
                            greaterThan: {current_block - blocks_per_query}
                        }}
                    }}, orderBy: PARA_BLOCK_HEIGHT_DESC
                ) {{
                    nodes {{
                        balances
                        assetId
                        pool {{
                            poolId
                            paraBlockHeight
                            pegs
                            fee
                            finalAmplification
                        }}
                    }}
                }}
            }}
        """

        # variables = {'assetIds': asset_ids_remaining}
        data = query_indexer(URL_STABLESWAP_STORAGE, query)
        for node in data['data']['stableswapAssetData']['nodes']:
            asset_id = str(node['assetId'])
            if asset_id not in asset_ids_remaining:
                continue
            asset_name = stableswap_asset_data[asset_id]['name']
            pool_id = str(node['pool']['poolId'])
            block = node['pool']['paraBlockHeight']
            pool = stableswap_pools[pool_id]
            pool.liquidity[asset_name] = int(node['balances']['d'][0]) / 10 ** stableswap_asset_data[asset_id]['decimals']
            if block > pool.time_step:
                pool.trade_fee = float(node['pool']['fee'] / 1000000)
                pool.amplification = node['pool']['finalAmplification']
                pegs = [
                    int(node['pool']['pegs'][i][0]) / int(node['pool']['pegs'][i][1])
                    for i in range(len(node['pool']['pegs']) - 1)
                ]
                if len(pegs) != len(pool.asset_list) - 1:
                    pass
                pool.set_peg_target(pegs)
                pool.set_peg(pegs)
                pool.time_step = block
            asset_ids_remaining.remove(asset_id)

        queries += 1
        current_block -= blocks_per_query

    # there may be some tokens we can't get liquidity values for... in that case, guesstimate
    for pool in stableswap_pools.values():
        for tkn in pool.liquidity:
            if pool.liquidity[tkn] == 0:
                pool.liquidity[tkn] = sum(
                    [
                        pool.liquidity[t] * (1 if i == 0 else pool.peg[i])
                        for i, t in enumerate(pool.asset_list) if t != tkn
                    ]
                ) / (len(pool.asset_list) - 1)

    # get total issuance of shares
    stableswap_share_assets = {tkn: all_assets[tkn] for tkn in all_assets if all_assets[tkn].id in stableswap_ids}
    for pool_id in stableswap_ids:
        query = f"""
            query GetTotalIssuance {{
              assetHistoricalData(
                first: 1
                orderBy: PARA_BLOCK_HEIGHT_DESC
                filter: {{assetId: {{equalTo: "{pool_id}"}}}}
              ) {{
                nodes {{
                  assetId
                  totalIssuance
                  paraBlockHeight
                }}
              }}
            }}
        """
        shares = int(query_indexer(URL_GENERIC_DATA, query)['data']['assetHistoricalData']['nodes'][0]['totalIssuance'])
        shares /= 10 ** stableswap_share_assets[pool_id].decimals
        stableswap_pools[pool_id].shares = shares

    return stableswap_pools


def get_current_omnipool(block_number = None):
    asset_info = get_asset_info_by_ids()
    asset_ids = get_current_omnipool_assets()
    for asset in asset_info.values():
        if asset.asset_type == 'StableSwap':
            asset.symbol = asset.name
    max_block = get_current_block_height() if block_number is None else block_number
    asset_ids_remaining = asset_ids.copy()
    liquidity = {}
    lrna = {}
    shares = {}
    protocol_shares = {}
    current_block = max_block
    blocks_per_query = 100
    max_queries = 10
    queries = 0
    while asset_ids_remaining and queries < max_queries:
        omnipool_data = get_omnipool_asset_data(
            min_block_id=current_block - blocks_per_query,
            max_block_id=current_block,
            asset_ids=asset_ids_remaining
        )
        for item in reversed(omnipool_data):
            asset = asset_info[str(item['assetId'])]
            if asset.symbol not in liquidity:
                liquidity[asset.symbol] = int(item['balances']['d'][0]) / 10 ** asset.decimals
                lrna[asset.symbol] = int(item['assetState']['d'][0]) / 10 ** asset_info['1'].decimals
                shares[asset.symbol] = int(item['assetState']['d'][1]) / 10 ** asset.decimals
                protocol_shares[asset.symbol] = int(item['assetState']['d'][2]) / 10 ** asset.decimals
                asset_ids_remaining.remove(asset.id)
        current_block -= blocks_per_query
        queries += 1

    for tkn in asset_ids_remaining:
        print(f"{asset_info[tkn].name} not found in {blocks_per_query * max_queries} blocks.")
        asset_ids.remove(tkn)

    asset_fee, lrna_fee = get_current_omnipool_fees(
        asset_info={tkn: asset_info[tkn] for tkn in asset_ids},
        block_number=block_number
    )

    omnipool = OmnipoolState(
        tokens={
            tkn: {
                'liquidity': liquidity[tkn],
                'LRNA': lrna[tkn],
                'shares': shares[tkn],
                'protocol_shares': protocol_shares[tkn],
            } for tkn in liquidity
        },
        asset_fee=asset_fee,
        lrna_fee=lrna_fee
    )
    omnipool.time_step = max_block
    return omnipool


def get_current_omnipool_fees(
    asset_info: dict[str: AssetInfo] = None,
    block_number: int = None
) -> tuple[DynamicFee, DynamicFee]:

    if asset_info and isinstance(list(asset_info.keys())[0], int):
        raise TypeError("Asset info keys must be str.")

    if block_number is None:
        block_number = get_current_block_height()
    if asset_info is None:
        asset_info = get_asset_info_by_ids(get_current_omnipool_assets())

    asset_fee = DynamicFee(
        minimum=0.0015,
        maximum=0.05,
        amplification=2,
        decay=0.001
    )
    lrna_fee = DynamicFee(
        minimum=0.0005,
        maximum=0.01,
        amplification=1,
        decay=0.005
    )
    blocks_per_query = 100
    max_queries = int(max(
        (lrna_fee.maximum - lrna_fee.minimum) / lrna_fee.decay,
        (asset_fee.maximum - asset_fee.minimum) / asset_fee.decay
    ) / 100) + 1  # beyond that, fees will have decayed to minimum anyway
    queries = 0
    current_block = block_number
    asset_ids_remaining = list(asset_info.keys())

    url = URL_UNIFIED_PROD
    while len(asset_ids_remaining) > 0 and queries < max_queries:
        query = f"""
            query MyQuery {{
                events(
                    orderBy: PARA_BLOCK_HEIGHT_DESC
                    filter: {{
                        name: {{includes: "Omnipool"}}, 
                        paraBlockHeight: {{
                            greaterThan: {current_block - blocks_per_query}, 
                            lessThanOrEqualTo: {current_block}
                        }}
                    }}
                ) {{
                    nodes {{
                      name
                      args
                      id
                      paraBlockHeight
                    }}
                }}
            }}
        """
        transaction_data = query_indexer(url, query)['data']['events']['nodes']
        for trade in transaction_data:
            args = {
                arg[:arg.index(':')].strip('"'): arg[arg.index(':') + 1:].strip('"')
                for arg in trade['args'].strip('}').strip('{').split(',')
            }
            block = trade['paraBlockHeight']
            sell_id = int(args['assetIn'])
            buy_id = int(args['assetOut'])
            tkn_sell = asset_info[sell_id] if sell_id in asset_info else None
            tkn_buy = asset_info[buy_id] if buy_id in asset_info else None
            if tkn_sell and int(tkn_sell.id) in asset_ids_remaining:
                if tkn_sell.symbol not in lrna_fee.current and float(args['hubAmountOut']) > 0:
                    lrna_fee.current[tkn_sell.symbol] = float(args['protocolFeeAmount']) / float(args['hubAmountOut'])
                    lrna_fee.last_updated[tkn_sell.symbol] = block
                    if tkn_sell.symbol in asset_fee.current:
                        asset_ids_remaining.remove(int(args['assetIn']))
            if tkn_buy and int(tkn_buy.id) in asset_ids_remaining:
                if tkn_buy.symbol not in asset_fee.current and float(args['amountIn']) > 0:
                    asset_fee.current[tkn_buy.symbol] = float(args['assetFeeAmount']) / (float(args['amountOut']) + float(args['assetFeeAmount']))
                    asset_fee.last_updated[tkn_buy.symbol] = block
                    if tkn_buy.symbol in lrna_fee.current:
                        asset_ids_remaining.remove(int(args['assetOut']))

        queries += 1
        current_block -= blocks_per_query

    for tkn in asset_ids_remaining:
        symbol = asset_info[tkn].symbol
        # print(f"{asset_info[tkn].name} fee not found in {blocks_per_query * max_queries} blocks.")
        if symbol not in lrna_fee.current:
            lrna_fee.current[symbol] = lrna_fee.minimum
            lrna_fee.last_updated[symbol] = current_block
        if symbol not in asset_fee.current:
            asset_fee.current[symbol] = asset_fee.minimum
            asset_fee.last_updated[symbol] = current_block

    return asset_fee, lrna_fee


def get_current_omnipool_router(block_number: int = None):
    if block_number is None:
        block_number = get_current_block_height()
    omnipool = get_current_omnipool(block_number)
    stableswap_pools = get_current_stableswap_pools(block_number).values()

    return OmnipoolRouter(
        exchanges=[
            omnipool, *stableswap_pools
        ]
    )


def get_executed_trades(min_block: int, max_block: int, asset_ids: list[str] = None) -> list:

    asset_info = get_asset_info_by_ids(asset_ids)

    executed_trades_query = """
    query executed_trade_query(
        $first: Int!, $after: Cursor,
        $assetIds: [String!]!, $minBlock: Int!, $maxBlock: Int!
    ) {
      routedTrades(
        first: $first,
        after: $after,
        filter: {
            allInvolvedAssetIds: {contains: $assetIds},
            paraBlockHeight: { greaterThanOrEqualTo: $minBlock, lessThanOrEqualTo: $maxBlock }
        }
        orderBy: PARA_BLOCK_HEIGHT_ASC
      ) {
        nodes {
          routeTradeInputs {
            nodes {
              amount
              assetId
            }
          }
          routeTradeOutputs {
            nodes {
              amount
              assetId
            }
          }
          paraBlockHeight
          allInvolvedAssetIds
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
    """

    variables = {"assetIds": [str(id) for id in asset_ids], "minBlock": min_block, "maxBlock": max_block}
    # data = query_indexer(url, hdx_fee_query, variables)

    data_all = []
    has_next_page = True
    after_cursor = None
    page_size = 10000
    variables["first"] = page_size

    while has_next_page:
        variables["after"] = after_cursor
        data = query_indexer(URL_UNIFIED_PROD, executed_trades_query, variables)
        page_data = data['data']['routedTrades']['nodes']
        data_all.extend(page_data)
        page_info = data['data']['routedTrades']['pageInfo']
        has_next_page = page_info['hasNextPage']
        after_cursor = page_info['endCursor']

    trade_data = [
        {
            'block_number': int(x['paraBlockHeight']),
            'input_asset_id': x['routeTradeInputs']['nodes'][0]['assetId'],
            'input_amount': int(x['routeTradeInputs']['nodes'][0]['amount']) / (10 ** asset_info[x['routeTradeInputs']['nodes'][0]['assetId']].decimals),
            'output_asset_id': x['routeTradeOutputs']['nodes'][0]['assetId'],
            'output_amount': int(x['routeTradeOutputs']['nodes'][0]['amount']) / (10 ** asset_info[x['routeTradeOutputs']['nodes'][0]['assetId']].decimals),
            'all_involved_asset_ids': [y for y in x['allInvolvedAssetIds']]
        }
        for x in data_all
        if x['routeTradeOutputs']['nodes'][0]['assetId'] in asset_info
        and x['routeTradeInputs']['nodes'][0]['assetId'] in asset_info
    ]

    return trade_data


def get_stableswap_liquidity_events(pool_id: int, min_block: int, max_block: int):

    events_query = """
    query MyQuery(
      $first: Int!, $after: Cursor,
      $poolId: String!, $minBlock: Int!, $maxBlock: Int!
    ) {
      stableswapLiquidityEvents(
        first: $first,
        after: $after,
        orderBy: PARA_BLOCK_HEIGHT_ASC
        filter: {poolId: {equalTo: $poolId}, paraBlockHeight: {greaterThanOrEqualTo: $minBlock, lessThanOrEqualTo: $maxBlock}}
      ) {
        nodes {
          sharesAmount
          actionType
          paraBlockHeight
          stableswapAssetLiquidityAmountsByLiquidityActionId {
            nodes {
              amount
              assetId
            }
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
    """

    variables = {"poolId": str(pool_id), "minBlock": min_block, "maxBlock": max_block}

    data_all = []
    has_next_page = True
    after_cursor = None
    page_size = 1000
    variables["first"] = page_size

    i = 0
    while has_next_page:
        print(f'Fetching page {i}...')
        variables["after"] = after_cursor
        data = query_indexer(URL_UNIFIED_PROD, events_query, variables)
        page_data = data['data']['stableswapLiquidityEvents']['nodes']
        data_all.extend(page_data)
        page_info = data['data']['stableswapLiquidityEvents']['pageInfo']
        has_next_page = page_info['hasNextPage']
        after_cursor = page_info['endCursor']
        i += 1

    return data_all


def download_stableswap_exec_prices(pool_id: int, tkn_id: int, min_block: int, max_block: int, path: str):
    data = get_stableswap_liquidity_events(pool_id, min_block, max_block)
    asset_info = get_asset_info_by_ids([tkn_id, pool_id])
    decimals_tkn_id = asset_info[tkn_id].decimals
    decimals_pool_id = asset_info[pool_id].decimals
    prices_by_block = {}
    for tx in data:
        liq_data = tx['stableswapAssetLiquidityAmountsByLiquidityActionId']['nodes']
        if len(liq_data) == 1 and int(liq_data[0]['assetId']) == tkn_id and int(liq_data[0]['amount']) > 0:
            block_no = int(tx['paraBlockHeight'])
            if block_no not in prices_by_block:
                prices_by_block[block_no] = []
            price = int(liq_data[0]['amount']) / int(tx['sharesAmount']) * (10 ** (decimals_pool_id - decimals_tkn_id))
            prices_by_block[block_no].append(price)
    avg_price_by_block = {}
    for block_no in prices_by_block:
        avg_price_by_block[block_no] = sum(prices_by_block[block_no]) / len(prices_by_block[block_no])
    with open(f"{path}stableswap_exec_prices_{pool_id}_{tkn_id}_{min_block}_{max_block}.json", "w") as f:
        json.dump(avg_price_by_block, f)


def download_omnipool_spot_prices(tkn_id: int, denom_id: int, min_block: int, max_block: int, path: str):
    data = get_omnipool_asset_data(min_block, max_block, [denom_id, tkn_id])
    # assert len(data) == 2 * (max_block - min_block + 1)
    if len(data) != 2 * (max_block - min_block + 1):
        print(f'Warning: len(data) is {len(data)}, expected {2 * (max_block - min_block + 1)}')
    asset_info = get_asset_info_by_ids([tkn_id, denom_id])
    tkn_decimals = asset_info[tkn_id].decimals
    denom_decimals = asset_info[denom_id].decimals
    prices = {}
    prev_block = None
    for i in range(max_block - min_block + 1):
        tkn_i = 2 * i + (0 if data[2 * i]['assetId'] == tkn_id else 1)
        denom_i = 2 * i + (0 if data[2 * i]['assetId'] == denom_id else 1)
        data_tkn = data[tkn_i]
        data_denom = data[denom_i]
        assert data_tkn['paraBlockHeight'] == data_denom['paraBlockHeight'], "Block number must match"
        block_no = int(data_tkn['paraBlockHeight'])
        assert block_no == min_block + i, "Block number must be sequential"
        hub_price_of_tkn = int(data_tkn['assetState']['d'][1]) / int(data_tkn['balances']['d'][0])
        hub_price_of_denom = int(data_denom['assetState']['d'][1]) / int(data_denom['balances']['d'][0])
        price = hub_price_of_tkn / hub_price_of_denom * 10 ** (tkn_decimals - denom_decimals)
        if prev_block is not None and prices[prev_block] == price:
            continue
        prices[block_no] = price
        prev_block = block_no

    with open(f"{path}omnipool_spot_prices_{denom_id}_{tkn_id}_{min_block}_{max_block}.json", "w") as f:
        json.dump(prices, f)


def get_omnipool_swap_fees_one_query(tkn_id: int, min_block: int, max_block: int):

    query = """
    query OmnipoolSwapsByAssetId(
      $assetId: String!,
      $minBlock: Int!, $maxBlock: Int!
    ) {
      swaps(
        filter: {
          allInvolvedAssetIds: {contains: ["1", $assetId], containedBy: ["1", $assetId]},
          paraBlockHeight: { greaterThanOrEqualTo: $minBlock, lessThanOrEqualTo: $maxBlock }
        }
        orderBy: PARA_BLOCK_HEIGHT_ASC
      ) {
        nodes {
          swapOutputs {
            nodes {
              amount
              assetId
            }
          }
          swapFees {
            nodes {
              amount
              assetId
            }
          }
          paraBlockHeight
        }
      }
    }

    """

    variables = {"assetId": str(tkn_id), "minBlock": min_block, "maxBlock": max_block}
    data = query_indexer(URL_UNIFIED_PROD, query, variables)

    asset_info = get_asset_info_by_ids([tkn_id, 1])  # 1 is the hub asset ID

    asset_fee_data = []
    hub_fee_data = []
    for x in data['data']['swaps']['nodes']:
        block_number = int(x['paraBlockHeight'])
        asset_id = x['swapOutputs']['nodes'][0]['assetId']
        decimals = asset_info[asset_id].decimals
        output_amount = int(x['swapOutputs']['nodes'][0]['amount']) / (10 ** decimals)
        fee_amount = sum([int(y['amount']) for y in x['swapFees']['nodes']]) / (10 ** decimals)
        if output_amount > 0:
            fee_dict = {
                    'block_number': block_number,
                    'output_amount': output_amount,
                    'fee_amount': fee_amount
                }
            if asset_id == tkn_id:
                fee_pct = fee_amount / (output_amount + fee_amount)
                fee_dict['fee_pct'] = fee_pct
                asset_fee_data.append(fee_dict)
            elif asset_id == 1:  # 1 is the hub asset ID
                fee_pct = fee_amount / output_amount if output_amount > 0 else 0
                fee_dict['fee_pct'] = fee_pct
                hub_fee_data.append(fee_dict)
    return asset_fee_data, hub_fee_data

# TODO: ^v these two functions could be merged into one

def get_omnipool_swap_fees(tkn_id: int, min_block: int, max_block: int, max_block_count: int = 50000):
    min_block_temp = min_block
    max_block_temp = min(min_block + max_block_count - 1, max_block)
    asset_fee_data = []
    hub_fee_data = []
    while min_block_temp <= max_block:
        print(f"Querying from block {min_block_temp} to {max_block_temp}...")
        asset_data, hub_data = get_omnipool_swap_fees_one_query(tkn_id, min_block_temp, max_block_temp)
        asset_fee_data.extend(asset_data)
        hub_fee_data.extend(hub_data)
        min_block_temp = max_block_temp + 1
        max_block_temp = min(min_block_temp + max_block_count - 1, max_block)
    return asset_fee_data, hub_fee_data


def bucket_values(bucket_ct: int, data: list, min_block: int = None, max_block: int = None):
    """Aggregates values in data into buckets based on block numbers.

    Args:
        bucket_ct (int): Number of buckets to create.
        data (list): List of tuples where each tuple contains (block_number, value).
        min_block (int, optional): Minimum block number to consider.
        max_block (int, optional): Maximum block number to consider.
    """

    if min_block == None or max_block == None:
        min_block = float('inf')
        max_block = -1
        for x in data:
            if x[0] < min_block:
                min_block = x[0]
            if x[0] > max_block:
                max_block = x[0]
    bucket_size = (max_block - min_block) // bucket_ct + 1
    bucketed_data = [
        {'start_block': min_block + i * bucket_size, 'end_block': min_block + (i + 1) * bucket_size - 1, 'value': 0}
        for i in range(bucket_ct)
    ]
    if bucketed_data[-1]['end_block'] < max_block:
        raise ValueError("Last bucket does not cover the max_block, increase bucket_ct or decrease bucket_size")
    bucketed_data[-1]['end_block'] = max_block  # ensure last bucket ends at max_block
    for x in data:
        bucket_index = (x[0] - min_block) // bucket_size
        bucketed_data[bucket_index]['value'] += x[1]
    return bucketed_data


def bucket_values_per_block(bucket_ct: int, data, min_block: int = None, max_block: int = None):
    values = bucket_values(bucket_ct, data, min_block, max_block)
    return [
        {
            'start_block': x['start_block'], 'end_block': x['end_block'],
            'value': x['value'] / (x['end_block'] - x['start_block'] + 1)
        }
        for x in values
    ]


def download_acct_trades(asset_id: int, acct: str, path: str, min_block: int = None, max_block: int = None):

    if min_block is None:
        min_block = 0
    if max_block is None:
        max_block = get_current_block_height()

    query = """
    query assetInfoByAssetIds($acct: String!, $assetId: String!, $minBlock: Int!, $maxBlock: Int!) {
      routedTrades(
        filter: {
          participantSwappers: {anyEqualTo: $acct},
          allInvolvedAssetIds: {
            contains: [$assetId, "1", "102", "10"], containedBy: [$assetId, "1", "102", "10"]
          },
          paraBlockHeight: { greaterThanOrEqualTo: $minBlock, lessThanOrEqualTo: $maxBlock }
        }
      ) {
        nodes {
          routeTradeInputs {
            nodes {
              amount
              assetId
            }
          }
          routeTradeOutputs {
            nodes {
              amount
              assetId
            }
          }
          paraBlockHeight
          swaps {
            nodes {
              swapFees {
                nodes {
                  amount
                  assetId
                }
              }
              swapInputs {
                nodes {
                  amount
                  assetId
                }
              }
              swapOutputs {
                nodes {
                  amount
                  assetId
                }
              }
            }
          }
        }
      }
    }
    """

    variables = {'acct': acct, 'assetId': str(asset_id), 'minBlock': min_block, 'maxBlock': max_block}
    data = query_indexer(URL_UNIFIED_PROD, query, variables)
    asset_info = get_asset_info_by_ids([asset_id, 10, 1])
    trades = []
    for x in data['data']['routedTrades']['nodes']:
        if (len(x['routeTradeInputs']['nodes']) == 1 and len(x['routeTradeOutputs']['nodes']) == 1
            and x['routeTradeInputs']['nodes'][0]['assetId'] in [str(asset_id), '10']
            and x['routeTradeOutputs']['nodes'][0]['assetId'] in [str(asset_id), '10']):
            input_asset_id = int(x['routeTradeInputs']['nodes'][0]['assetId'])
            output_asset_id = int(x['routeTradeOutputs']['nodes'][0]['assetId'])
            found_hub = False
            for y in x['swaps']['nodes']:
                outputs = y['swapOutputs']['nodes']
                if len(outputs) == 1 and outputs[0]['assetId'] == '1':
                    if found_hub == True:
                        raise ValueError("Multiple hub outputs found in a single trade")
                    found_hub = True
                    hub_fees = sum([int(z['amount']) for z in y['swapFees']['nodes'] if z['assetId'] == '1'])
                    hub_amt = int(outputs[0]['amount'])
                    trade = {
                        'block_number': int(x['paraBlockHeight']),
                        'input_asset_id': input_asset_id,
                        'input_amount': int(x['routeTradeInputs']['nodes'][0]['amount']) / (10 ** asset_info[input_asset_id].decimals),
                        'output_asset_id': output_asset_id,
                        'output_amount': int(x['routeTradeOutputs']['nodes'][0]['amount']) / (10 ** asset_info[output_asset_id].decimals),
                        'hub_fee': hub_fees / (10 ** asset_info[1].decimals),
                        'hub_amount': hub_amt / (10 ** asset_info[1].decimals),
                    }
                    trades.append(trade)

    with open(f"{path}acct_swaps_{asset_id}_{acct}.json", "w") as f:
        json.dump(trades, f)
