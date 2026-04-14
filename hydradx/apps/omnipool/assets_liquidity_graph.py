from hydradx.model.indexer_utils import query_indexer, get_blocks_at_timestamps, get_asset_info_by_ids
import datetime
from matplotlib import pyplot as plt
from pathlib import Path
import streamlit as st
import json
import math

LIQUIDITY_GRAPH_TARGET_POINTS = 1000


def _sort_key(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def compress_liquidity_series(liquidity_series: dict, target_points: int) -> dict:
    if target_points <= 0:
        raise ValueError("target_points must be > 0")
    if len(liquidity_series) <= target_points:
        return dict(liquidity_series)

    items = sorted(liquidity_series.items(), key=lambda item: _sort_key(item[0]))
    total = len(items)
    batch_size = max(1, math.ceil(total / target_points))
    compressed = {}

    for idx in range(0, total, batch_size):
        batch = items[idx: idx + batch_size]
        first_key = batch[0][0]
        avg_value = sum(value for _, value in batch) / len(batch)
        compressed[first_key] = avg_value

    last_key, last_value = items[-1]
    compressed[last_key] = last_value
    return compressed


def _to_int_keyed(series: dict) -> dict:
    return {int(k): v for k, v in series.items()}


def get_liquidity_over_time():
    dates = [
                datetime.datetime(2025, 11, day=i + 1) for i in range(30)
            ] + [
                datetime.datetime(year=2025, month=12, day=i + 1) for i in range(31)
            ]
    block_map = get_blocks_at_timestamps(dates)
    ordered_blocks = sorted(block_map.items(), key=lambda item: item[0])
    ordered_dates = [item[0] for item in ordered_blocks]
    block_numbers = [item[1] for item in ordered_blocks]
    liquidity = {}

    if not Path.exists(Path(__file__).parent / 'cached data' / 'liquidity.json'):
        for i, start_block in enumerate(block_numbers[:-1]):
            date = dates[i]
            print(f"scanning {date}")
            blocks_per_query = 1000
            end_block = block_numbers[1 + 1]
            for block in range(start_block, end_block, blocks_per_query):
                query_start = block
                query_end = min(block + blocks_per_query, end_block)
                query = f"""
                    query AssetBalancesByBlockHeight {{
                        omnipoolAssetHistoricalData(
                            filter: {{paraBlockHeight: {{greaterThanOrEqualTo: {query_start}, lessThan: {query_end}}}}}
                        ) {{
                            nodes
                            {{
                                freeBalance
                                assetId
                                paraBlockHeight
                            }}
                        }}
                    }}
                """

                results = query_indexer("https://galacticcouncil.squids.live/hydration-pools:unified-prod/api/graphql", query)
                for result in results["data"]["omnipoolAssetHistoricalData"]["nodes"]:
                    asset_id = result["assetId"]
                    free_balance = int(result["freeBalance"])
                    result_block = int(result["paraBlockHeight"])
                    if asset_id not in liquidity:
                        liquidity[asset_id] = {}
                    liquidity[asset_id][result_block] = free_balance

        with open (Path(__file__).parent / 'cached data' / 'liquidity.json', 'w') as f:
            json.dump(liquidity, f)
    else:
        with open (Path(__file__).parent / 'cached data' / 'liquidity.json', 'r') as f:
            liquidity = json.load(f)

    asset_names = {tkn.id: tkn.unique_id for tkn in get_asset_info_by_ids(list(liquidity.keys())).values()}
    graph_liquidity = {}
    for tkn in liquidity:
        raw_series = _to_int_keyed(liquidity[tkn])
        if len(raw_series) > LIQUIDITY_GRAPH_TARGET_POINTS:
            graph_liquidity[tkn] = compress_liquidity_series(
                raw_series,
                target_points=LIQUIDITY_GRAPH_TARGET_POINTS,
            )
        else:
            graph_liquidity[tkn] = dict(raw_series)
        graph_liquidity[tkn][block_numbers[0]] = list(raw_series.values())[0]
        graph_liquidity[tkn][block_numbers[-1]] = list(raw_series.values())[-1]
        sorted_items = sorted(graph_liquidity[tkn].items(), key=lambda item: _sort_key(item[0]))
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot([item[0] for item in sorted_items], [item[1] for item in sorted_items])
        ax.set_title(f"Liquidity of {asset_names[tkn]} in Omnipool")
        ax.set_xlabel("Date")
        ax.set_ylabel("Free Balance")
        ax.set_xticks(block_numbers)
        ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in ordered_dates])
        ax.tick_params(axis="x", labelsize=8)

        plt.tight_layout()
        plt.show()
        st.pyplot(fig)
    pass

