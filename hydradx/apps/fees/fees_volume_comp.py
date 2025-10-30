from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from hydradx.model.indexer_utils import get_fee_history, get_asset_info_by_ids

c1, c2, c3 = st.columns(3)

with c1:
    asset_id = str(st.number_input(
        "Asset ID",
        min_value=0, max_value=99999999, value=5, step=1, key="asset_id", format="%d",
    ))

with c2:
    min_block = st.number_input(
        "Minimum block ID",
        min_value=0, max_value=99999999, value=7480000, step=1, key="min_block", format="%d"
    )

with c3:
    max_block = st.number_input(
        "Maximum block ID",
        min_value=0, max_value=99999999, value=7500000, step=1, key="max_block", format="%d",
        help="set to 0 to get latest data available"
    )

asset_info = get_asset_info_by_ids([asset_id])
decimals = asset_info[asset_id].decimals

data = get_fee_history(asset_id, min_block, max_block if max_block > 0 else None)
fee_amts = {}
swap_amts = {}
fee_pcts = {}
for x in data:
    if int(x['swapOutputs']['nodes'][0]['assetId']) == asset_id:
        block_no = int(x['paraBlockHeight'])
        if block_no not in fee_amts:
            fee_amts[block_no] = 0
        if block_no not in swap_amts:
            swap_amts[block_no] = 0
        fee_amts[block_no] += sum([int(y['amount']) for y in x['swapFees']['nodes'] if y['assetId'] == str(asset_id)]) / (10 ** decimals)
        swap_amts[block_no] += int(x['swapOutputs']['nodes'][0]['amount']) / (10 ** decimals)
        fee_pcts[block_no] = fee_amts[block_no] / swap_amts[block_no]

graph_data = {}
for block_no in swap_amts:
    fee_bucket = ((fee_pcts[block_no] * 10000) // 1) / 10000
    if fee_bucket not in graph_data:
        graph_data[fee_bucket] = {
            'fee_amts': [],
            'swap_amts': []
        }
    graph_data[fee_bucket]['fee_amts'].append(fee_amts[block_no])
    graph_data[fee_bucket]['swap_amts'].append(swap_amts[block_no])
for fee_bucket in graph_data:
    graph_data[fee_bucket]['average_fee_amt'] = sum(graph_data[fee_bucket]['fee_amts']) / len(graph_data[fee_bucket]['fee_amts'])
    graph_data[fee_bucket]['average_swap_amt'] = sum(graph_data[fee_bucket]['swap_amts']) / len(graph_data[fee_bucket]['swap_amts'])

fig, ax = plt.subplots()
ax.scatter(list(graph_data.keys()), [graph_data[x]['average_swap_amt'] for x in graph_data], s=5)
ax.set_title("asset fees history")
ax.set_xlabel("fee percentage")
ax.set_ylabel("swap amount in block")
st.pyplot(fig)

fig, ax = plt.subplots()
ax.scatter(list(graph_data.keys()), [graph_data[x]['average_fee_amt'] for x in graph_data], s=5)
ax.set_title("fee amounts history")
ax.set_xlabel("fee percentage")
ax.set_ylabel("fee amount in block")
st.pyplot(fig)