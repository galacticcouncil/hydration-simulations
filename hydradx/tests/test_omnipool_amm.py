import copy
import math

import pytest
from hypothesis import given, strategies as st, assume, settings, reproduce_failure
from mpmath import mp, mpf
import os

os.chdir('../..')

from hydradx.model import run, processing
from hydradx.model.amm import omnipool_amm as oamm
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.omnipool_amm import DynamicFee, OmnipoolState, OmnipoolLiquidityPosition
from hydradx.model.amm.trade_strategies import constant_swaps, omnipool_arbitrage
from hydradx.tests.strategies_omnipool import omnipool_reasonable_config, omnipool_config, assets_config
import hydradx.model.production_settings as production_settings

mp.dps = 50

asset_price_strategy = st.floats(min_value=0.0001, max_value=100000)
asset_price_bounded_strategy = st.floats(min_value=0.1, max_value=10)
asset_number_strategy = st.integers(min_value=3, max_value=5)
asset_quantity_strategy = st.floats(min_value=100, max_value=10000000)
asset_quantity_bounded_strategy = st.floats(min_value=1000000, max_value=10000000)
fee_strategy = st.floats(min_value=0.0001, max_value=0.1, allow_nan=False, allow_infinity=False)


@given(omnipool_config(asset_fee=0, lrna_fee=0, token_count=3), asset_quantity_strategy)
def test_swap_lrna_delta_Qi_respects_invariant(d: oamm.OmnipoolState, delta_ri: float):
    i = d.asset_list[-1]
    assume(i in d.asset_list)
    assume(d.liquidity[i] > delta_ri > -d.liquidity[i])
    d2 = copy.deepcopy(d)
    delta_Qi = oamm.swap_lrna_delta_Qi(d, delta_ri, i)
    d2.liquidity[i] += delta_ri
    d2.lrna[i] += delta_Qi

    # Test basics
    for j in d2.liquidity:
        assert d2.liquidity[j] > 0
        assert d2.lrna[j] > 0
    assert not (delta_ri > 0 and delta_Qi > 0)
    assert not (delta_ri < 0 and delta_Qi < 0)

    # Test that the pool invariant is respected
    assert oamm.asset_invariant(d2, i) == pytest.approx(oamm.asset_invariant(d, i))


@given(omnipool_config(asset_fee=0, lrna_fee=0, token_count=3), asset_quantity_strategy)
def test_swap_lrna_delta_Ri_respects_invariant(d: oamm.OmnipoolState, delta_qi: float):
    i = d.asset_list[-1]
    assume(i in d.asset_list)
    assume(d.lrna[i] > delta_qi > -d.lrna[i])
    d2 = copy.deepcopy(d)
    delta_Ri = oamm.swap_lrna_delta_Ri(d, delta_qi, i)
    d2.lrna[i] += delta_qi
    d2.liquidity[i] += delta_Ri

    # Test basics
    for j in d.liquidity:
        assert d2.liquidity[j] > 0
        assert d2.lrna[j] > 0
    assert not (delta_Ri > 0 and delta_qi > 0)
    assert not (delta_Ri < 0 and delta_qi < 0)

    # Test that the pool invariant is respected
    assert oamm.asset_invariant(d2, i) == pytest.approx(oamm.asset_invariant(d, i))


@given(omnipool_config())
def test_sell_accuracy(initial_state):
    # Test that the sell function is exactly accurate
    initial_agent = Agent(
        holdings=initial_state.liquidity.copy()
    )
    tkn_sell = initial_state.asset_list[1]
    tkn_buy = initial_state.asset_list[2]
    sell_quantity = initial_state.liquidity[tkn_sell] / 10
    swap_state, swap_agent = oamm.simulate_swap(
        initial_state, initial_agent,
        tkn_buy, tkn_sell,
        sell_quantity=sell_quantity
    )
    asset_sold = initial_agent.holdings[tkn_sell] - swap_agent.holdings[tkn_sell]
    if asset_sold != pytest.approx(sell_quantity, rel=1e40):
        raise AssertionError('Asset sold is wrong.')


@given(omnipool_config(asset_fee=0, lrna_fee=0))
def test_weights(initial_state: oamm.OmnipoolState):
    old_state = initial_state
    for i in old_state.liquidity:
        assert oamm.weight_i(old_state, i) >= 0
    assert sum([oamm.weight_i(old_state, i) for i in old_state.liquidity]) == pytest.approx(1.0)


@given(omnipool_config())
def test_prices(market_state: oamm.OmnipoolState):
    for i in market_state.asset_list:
        assert market_state.lrna_price(i) > 0


@given(omnipool_config(token_count=3, lrna_fee=0, asset_fee=0))
def test_add_liquidity(initial_state: oamm.OmnipoolState):
    old_state = initial_state
    old_agent = Agent(
        holdings={i: 1000 for i in old_state.asset_list}
    )
    i = old_state.asset_list[-1]
    delta_R = 1000

    new_state, new_agents = oamm.simulate_add_liquidity(old_state, old_agent, delta_R, i)
    for j in initial_state.asset_list:
        assert old_state.lrna_price(j) == pytest.approx(new_state.lrna_price(j))
    if old_state.liquidity[i] / old_state.shares[i] != pytest.approx(new_state.liquidity[i] / new_state.shares[i]):
        raise AssertionError(f'Price change in {i}'
                             f'({old_state.liquidity[i] / old_state.shares[i]}) -->'
                             f'({pytest.approx(new_state.liquidity[i] / new_state.shares[i])})'
                             )

    # check enforcement of weight caps
    # first assign some weight caps
    for i in initial_state.liquidity:
        initial_state.weight_cap[i] = min(initial_state.lrna[i] / initial_state.lrna_total * 1.1, 1)

    if old_state.weight_cap[i] < 1:
        # calculate what should be the maximum allowable liquidity provision
        max_amount = ((old_state.weight_cap[i] / (1 - old_state.weight_cap[i])
                       * old_state.lrna_total - old_state.lrna[i] / (1 - old_state.weight_cap[i]))
                      / old_state.lrna_price(i))

        if max_amount < 0:
            raise AssertionError('This calculation makes no sense.')  # but actually, it works :)

        # make sure agent has enough funds
        old_agent.holdings[i] = max_amount * 2
        # eliminate general tvl cap, so we can test just the weight cap
        old_state.tvl_cap = float('inf')

        # try one just above and just below the maximum allowable amount
        illegal_state, illegal_agents = oamm.simulate_add_liquidity(old_state, old_agent, max_amount * 1.0000001, i)
        if not illegal_state.fail:
            raise AssertionError(f'illegal transaction passed against weight limit in {i}')
        legal_state, legal_agents = oamm.simulate_add_liquidity(old_state, old_agent, max_amount * 0.9999999, i)
        if legal_state.fail:
            raise AssertionError(f'legal transaction failed against weight limit in {i} ({new_state.fail})')


@settings(max_examples=1)
@given(omnipool_config(token_count=3, lrna_fee=0, asset_fee=0))
def test_add_liquidity_with_existing_position_fails(initial_state: oamm.OmnipoolState):
    old_state = initial_state
    tkn = old_state.asset_list[1]
    old_agent = Agent(
        holdings={tkn: old_state.liquidity[tkn] / 10, (old_state.unique_id, tkn): old_state.shares[tkn] / 10}
    )

    delta_R = old_agent.holdings[tkn]

    new_state, new_agents = oamm.simulate_add_liquidity(old_state, old_agent, delta_R, tkn)

    if not new_state.fail:
        raise AssertionError(f'Adding liquidity to an existing position should fail.')


@settings(max_examples=1)
@given(omnipool_config(token_count=3, lrna_fee=0, asset_fee=0))
def test_add_liquidity_with_existing_position_succeeds(initial_state: oamm.OmnipoolState):
    old_state = initial_state
    tkn = old_state.asset_list[1]
    old_agent = Agent(
        holdings={tkn: old_state.liquidity[tkn], (old_state.unique_id, tkn): old_state.shares[tkn] / 10}
    )

    delta_R = old_agent.holdings[tkn] / 10

    nft_id = "first_position"
    new_state, new_agent = oamm.simulate_add_liquidity(old_state, old_agent, delta_R, tkn, nft_id)

    if new_state.fail:
        raise AssertionError(f'Adding liquidity to an existing position should not fail.')
    if nft_id not in new_agent.nfts:
        raise AssertionError(f'LP position not added to agent NFTs.')
    if len(new_agent.holdings) != len(old_agent.holdings):
        raise AssertionError(f'Agent holdings have wrong length.')

    nft_id2 = "second_position"
    new_state2, new_agent2 = oamm.simulate_add_liquidity(new_state, new_agent, delta_R, tkn, nft_id2)
    if new_state2.fail:
        raise AssertionError(f'Adding liquidity to an existing position should not fail.')
    if nft_id2 not in new_agent2.nfts:
        raise AssertionError(f'LP position not added to agent NFTs.')
    if len(new_agent2.holdings) != len(old_agent.holdings):
        raise AssertionError(f'Agent holdings have wrong length.')


@given(st.integers(min_value=1, max_value=10))
def test_compare_several_lp_adds_to_single(n):
    liquidity = {'HDX': mpf(10000000), 'USD': mpf(1000000), 'DOT': mpf(100000)}
    lrna = {'HDX': mpf(1000000), 'USD': mpf(1000000), 'DOT': mpf(1000000)}
    initial_state = oamm.OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        }
    )
    tkn = initial_state.asset_list[1]
    init_agent = Agent(holdings={tkn: initial_state.liquidity[tkn]})
    delta_R = init_agent.holdings[tkn] / 2

    single_add_state_1, single_add_agent_1 = oamm.simulate_add_liquidity(initial_state, init_agent, delta_R, tkn)
    single_add_state_2, single_add_agent_2 = oamm.simulate_add_liquidity(initial_state, init_agent, delta_R, tkn,
                                                                         nft_id="id001")
    multi_add_state, multi_add_agent = initial_state, init_agent
    for i in range(n):
        nft_id = str(i)
        multi_add_state, multi_add_agent = oamm.simulate_add_liquidity(multi_add_state, multi_add_agent, delta_R / n,
                                                                       tkn, nft_id=nft_id)

    if single_add_state_1.liquidity[tkn] != pytest.approx(multi_add_state.liquidity[tkn], rel=1e-20):
        raise AssertionError(f'Adding liquidity in one go should be equivalent to adding it in {n} steps.')
    if single_add_state_1.shares[tkn] != pytest.approx(multi_add_state.shares[tkn], rel=1e-20):
        raise AssertionError(f'Adding liquidity in one go should be equivalent to adding it in {n} steps.')
    if single_add_state_1.lrna[tkn] != pytest.approx(multi_add_state.lrna[tkn], rel=1e-20):
        raise AssertionError(f'Adding liquidity in one go should be equivalent to adding it in {n} steps.')
    total_multi_shares = sum([multi_add_agent.nfts[nft_id].shares for nft_id in multi_add_agent.nfts])
    shares_1 = single_add_agent_1.holdings[(single_add_state_1.unique_id, tkn)]
    if total_multi_shares != pytest.approx(shares_1, rel=1e-20):
        raise AssertionError(f'Adding liquidity in one go should be equivalent to adding it in {n} steps.')

    if single_add_state_2.liquidity[tkn] != pytest.approx(multi_add_state.liquidity[tkn], rel=1e-20):
        raise AssertionError(f'Adding liquidity in one go should be equivalent to adding it in {n} steps.')
    if single_add_state_2.shares[tkn] != pytest.approx(multi_add_state.shares[tkn], rel=1e-20):
        raise AssertionError(f'Adding liquidity in one go should be equivalent to adding it in {n} steps.')
    if single_add_state_2.lrna[tkn] != pytest.approx(multi_add_state.lrna[tkn], rel=1e-20):
        raise AssertionError(f'Adding liquidity in one go should be equivalent to adding it in {n} steps.')
    shares_2 = single_add_agent_2.nfts["id001"].shares
    if shares_1 != pytest.approx(shares_2, rel=1e-20):
        raise AssertionError(f'Adding liquidity in one go should be equivalent to adding it in {n} steps.')


@settings(max_examples=1)
@given(omnipool_config(token_count=3, lrna_fee=0, asset_fee=0))
def test_add_liquidity_with_quantity_zero_should_fail(initial_state: oamm.OmnipoolState):
    old_state = initial_state
    tkn = old_state.asset_list[1]
    old_agent = Agent(
        holdings={tkn: old_state.liquidity[tkn] / 10, (old_state.unique_id, tkn): old_state.shares[tkn] / 10}
    )

    delta_R = 0

    new_state, new_agents = oamm.simulate_add_liquidity(old_state, old_agent, delta_R, tkn)

    if not new_state.fail:
        raise AssertionError(f'Adding liquidity with quantity zero should fail.')


def test_remove_liquidity_exact():
    liquidity = {'HDX': mpf(10000000), 'USD': mpf(1000000), 'DOT': mpf(100000)}
    lrna = {'HDX': mpf(1000000), 'USD': mpf(1000000), 'DOT': mpf(1000000)}
    initial_state = oamm.OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        }
    )
    tkn = 'DOT'

    p = initial_state.price(tkn, 'LRNA')
    s = initial_state.shares[tkn] / 10
    expected_r = initial_state.liquidity[tkn] / 10 * (1 - initial_state.min_withdrawal_fee)
    position = OmnipoolLiquidityPosition(tkn, p, s, 0, initial_state.unique_id)
    init_agent = Agent(nfts={'position': position})
    new_state, new_agent = oamm.simulate_remove_liquidity(initial_state, init_agent, s, tkn, 'position')
    delta_r = initial_state.liquidity[tkn] - new_state.liquidity[tkn]
    if delta_r != pytest.approx(expected_r, rel=1e-20):
        raise AssertionError(f'Removed liquidity should be equal to initial liquidity minus final liquidity.')

    p = initial_state.price(tkn, 'LRNA') / 2
    s = initial_state.shares[tkn] / 10
    position = OmnipoolLiquidityPosition(tkn, p, s, 0, initial_state.unique_id)
    init_agent = Agent(nfts={'position': position})
    new_state, new_agent = oamm.simulate_remove_liquidity(initial_state, init_agent, s, tkn, 'position')

    expected_dq_pct = mpf(1) / 10 * (1 - initial_state.min_withdrawal_fee)
    expected_agent_dq_pct = mpf(1) / 30 * (1 - initial_state.min_withdrawal_fee)
    expected_dr_pct = mpf(1) / 10 * (1 - initial_state.min_withdrawal_fee)
    expected_ds_pct = mpf(1) / 10

    actual_dq_pct = (initial_state.lrna[tkn] - new_state.lrna[tkn]) / initial_state.lrna[tkn]
    actual_agent_dq_pct = new_agent.holdings['LRNA'] / initial_state.lrna[tkn]
    actual_dr_pct = (initial_state.liquidity[tkn] - new_state.liquidity[tkn]) / initial_state.liquidity[tkn]
    actual_ds_pct = (initial_state.shares[tkn] - new_state.shares[tkn]) / initial_state.shares[tkn]
    actual_db = initial_state.protocol_shares[tkn] - new_state.protocol_shares[tkn]

    if actual_dq_pct != pytest.approx(expected_dq_pct, rel=1e-20):
        raise AssertionError(f'LRNA change incorrect')
    if actual_agent_dq_pct != pytest.approx(expected_agent_dq_pct, rel=1e-20):
        raise AssertionError(f'LRNA given to agent incorrect')
    if actual_dr_pct != pytest.approx(expected_dr_pct, rel=1e-20):
        raise AssertionError(f'Liquidity change incorrect')
    if actual_ds_pct != pytest.approx(expected_ds_pct, rel=1e-20):
        raise AssertionError(f'Shares change incorrect')
    if actual_db != 0:
        raise AssertionError(f'Protocol should not earn shares')

    p = initial_state.price(tkn, 'LRNA') * 2
    s = initial_state.shares[tkn] / 10
    position = OmnipoolLiquidityPosition(tkn, p, s, 0, initial_state.unique_id)
    init_agent = Agent(nfts={'position': position})
    new_state, new_agent = oamm.simulate_remove_liquidity(initial_state, init_agent, s, tkn, 'position')

    expected_dq_pct = mpf(2) / 30 * (1 - initial_state.min_withdrawal_fee)
    expected_dr_pct = mpf(2) / 30 * (1 - initial_state.min_withdrawal_fee)
    expected_ds_pct = mpf(2) / 30
    expected_db_pct = mpf(1) / 30

    actual_dq_pct = (initial_state.lrna[tkn] - new_state.lrna[tkn]) / initial_state.lrna[tkn]
    actual_agent_dq = new_agent.holdings['LRNA'] if 'LRNA' in new_agent.holdings else 0
    actual_dr_pct = (initial_state.liquidity[tkn] - new_state.liquidity[tkn]) / initial_state.liquidity[tkn]
    actual_ds_pct = (initial_state.shares[tkn] - new_state.shares[tkn]) / initial_state.shares[tkn]
    actual_db_pct = (new_state.protocol_shares[tkn] - initial_state.protocol_shares[tkn]) / initial_state.shares[tkn]

    if actual_dq_pct != pytest.approx(expected_dq_pct, rel=1e-20):
        raise AssertionError(f'LRNA change incorrect')
    if actual_agent_dq != pytest.approx(0, rel=1e-20):
        raise AssertionError(f'LRNA given to agent incorrect')
    if actual_dr_pct != pytest.approx(expected_dr_pct, rel=1e-20):
        raise AssertionError(f'Liquidity change incorrect')
    if actual_ds_pct != pytest.approx(expected_ds_pct, rel=1e-20):
        raise AssertionError(f'Shares change incorrect')
    if actual_db_pct != pytest.approx(expected_db_pct, rel=1e-20):
        raise AssertionError(f'Protocol shares incorrect')


@given(st.floats(min_value=0.1, max_value=10))
def test_remove_liquidity_specified_quantity_unspecified_nft(price_mult: float):
    liquidity = {'HDX': mpf(10000000), 'USD': mpf(1000000), 'DOT': mpf(100000)}
    lrna = {'HDX': mpf(1000000), 'USD': mpf(1000000), 'DOT': mpf(1000000)}
    initial_state = oamm.OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        }
    )
    tkn = 'DOT'

    p = price_mult * initial_state.price(tkn, 'LRNA')
    s = initial_state.shares[tkn] / 10
    holdings = {(initial_state.unique_id, tkn): s}
    share_prices = {(initial_state.unique_id, tkn): p}
    init_agent = Agent(holdings=holdings, share_prices=share_prices)

    position = OmnipoolLiquidityPosition(tkn, p, s, 0, initial_state.unique_id)
    base_agent = Agent(nfts={'position': position})

    quantity = s
    new_state, new_agent = oamm.simulate_remove_liquidity(initial_state, init_agent, quantity, tkn)
    comp_state, comp_agent = oamm.simulate_remove_liquidity(initial_state, base_agent, quantity, tkn, 'position')
    if new_state.liquidity[tkn] != pytest.approx(comp_state.liquidity[tkn], rel=1e-20):
        raise AssertionError(f"Remaining liquidity doesn't match.")
    if new_state.shares[tkn] != pytest.approx(comp_state.shares[tkn], rel=1e-20):
        raise AssertionError(f"Remaining shares doesn't match.")
    if new_state.lrna[tkn] != pytest.approx(comp_state.lrna[tkn], rel=1e-20):
        raise AssertionError(f"Remaining LRNA doesn't match.")
    if new_state.protocol_shares[tkn] != pytest.approx(comp_state.protocol_shares[tkn], rel=1e-20):
        raise AssertionError(f"Remaining protocol shares doesn't match.")

    quantity = s / 2
    new_state, new_agent = oamm.simulate_remove_liquidity(initial_state, init_agent, quantity, tkn)
    comp_state, comp_agent = oamm.simulate_remove_liquidity(initial_state, base_agent, quantity, tkn, 'position')
    if new_state.liquidity[tkn] != pytest.approx(comp_state.liquidity[tkn], rel=1e-20):
        raise AssertionError(f'Remaining liquidity doesn\'t match.')
    if new_state.shares[tkn] != pytest.approx(comp_state.shares[tkn], rel=1e-20):
        raise AssertionError(f'Remaining shares doesn\'t match.')
    if new_state.lrna[tkn] != pytest.approx(comp_state.lrna[tkn], rel=1e-20):
        raise AssertionError(f'Remaining LRNA doesn\'t match.')
    if new_state.protocol_shares[tkn] != pytest.approx(comp_state.protocol_shares[tkn], rel=1e-20):
        raise AssertionError(f'Remaining protocol shares doesn\'t match.')

    quantity = s * 2
    new_state, new_agent = oamm.simulate_remove_liquidity(initial_state, init_agent, quantity, tkn)
    if not new_state.fail:
        raise AssertionError(f'Removing liquidity with quantity greater than holdings should fail.')


@given(st.floats(min_value=0.1, max_value=10))
def test_remove_liquidity_unspecified_quantity_specified_nft(price_mult: float):
    liquidity = {'HDX': mpf(10000000), 'USD': mpf(1000000), 'DOT': mpf(100000)}
    lrna = {'HDX': mpf(1000000), 'USD': mpf(1000000), 'DOT': mpf(1000000)}
    initial_state = oamm.OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        }
    )
    tkn = 'DOT'

    p = price_mult * initial_state.price(tkn, 'LRNA')
    s = initial_state.shares[tkn] / 10
    position = OmnipoolLiquidityPosition(tkn, p, s, 0, initial_state.unique_id)
    init_agent = Agent(nfts={'position': position})

    new_state, new_agent = oamm.simulate_remove_liquidity(initial_state, init_agent, tkn_remove=tkn, nft_id='position')
    comp_state, comp_agent = oamm.simulate_remove_liquidity(initial_state, init_agent, s, tkn, 'position')
    if new_state.liquidity[tkn] != pytest.approx(comp_state.liquidity[tkn], rel=1e-20):
        raise AssertionError(f'Remaining liquidity doesn\'t match.')
    if new_state.shares[tkn] != pytest.approx(comp_state.shares[tkn], rel=1e-20):
        raise AssertionError(f'Remaining shares doesn\'t match.')
    if new_state.lrna[tkn] != pytest.approx(comp_state.lrna[tkn], rel=1e-20):
        raise AssertionError(f'Remaining LRNA doesn\'t match.')
    if new_state.protocol_shares[tkn] != pytest.approx(comp_state.protocol_shares[tkn], rel=1e-20):
        raise AssertionError(f'Remaining protocol shares doesn\'t match.')


@given(st.floats(min_value=0.1, max_value=10))
def test_remove_liquidity_unspecified_quantity_unspecified_nft(price_mult: float):
    liquidity = {'HDX': mpf(10000000), 'USD': mpf(1000000), 'DOT': mpf(100000)}
    lrna = {'HDX': mpf(1000000), 'USD': mpf(1000000), 'DOT': mpf(1000000)}
    initial_state = oamm.OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        }
    )
    tkn = 'DOT'

    p = price_mult * initial_state.price(tkn, 'LRNA')
    s = initial_state.shares[tkn] / 10
    holdings = {(initial_state.unique_id, tkn): s / 2}
    share_prices = {(initial_state.unique_id, tkn): p}
    position = OmnipoolLiquidityPosition(tkn, p, s / 2, 0, initial_state.unique_id)
    init_agent = Agent(holdings=holdings, share_prices=share_prices, nfts={'position': position})

    position = OmnipoolLiquidityPosition(tkn, p, s, 0, initial_state.unique_id)
    base_agent = Agent(nfts={'position': position})

    new_state, new_agent = oamm.simulate_remove_liquidity(initial_state, init_agent, tkn_remove=tkn)
    comp_state, comp_agent = oamm.simulate_remove_liquidity(initial_state, base_agent, s, tkn, 'position')
    if new_state.liquidity[tkn] != pytest.approx(comp_state.liquidity[tkn], rel=1e-20):
        raise AssertionError(f'Remaining liquidity doesn\'t match.')
    if new_state.shares[tkn] != pytest.approx(comp_state.shares[tkn], rel=1e-20):
        raise AssertionError(f'Remaining shares doesn\'t match.')
    if new_state.lrna[tkn] != pytest.approx(comp_state.lrna[tkn], rel=1e-20):
        raise AssertionError(f'Remaining LRNA doesn\'t match.')
    if new_state.protocol_shares[tkn] != pytest.approx(comp_state.protocol_shares[tkn], rel=1e-20):
        raise AssertionError(f'Remaining protocol shares doesn\'t match.')


@given(omnipool_config(token_count=3, withdrawal_fee=False))
def test_remove_liquidity_no_fee(initial_state: oamm.OmnipoolState):
    i = initial_state.asset_list[2]
    initial_agent = Agent(
        holdings={token: 1000 for token in initial_state.asset_list + ['LRNA']},
    )
    # add LP shares to the pool
    old_state, old_agent = oamm.simulate_add_liquidity(
        old_agent=initial_agent,
        old_state=initial_state,
        tkn_add=i,
        quantity=1000
    )

    p_init = old_state.lrna_price(i)
    delta_S = -old_agent.holdings[('omnipool', i)]

    new_state, new_agent = oamm.simulate_remove_liquidity(old_state, old_agent, delta_S, i)
    for j in new_state.asset_list:
        if old_state.price(j) != pytest.approx(new_state.price(j)):
            raise AssertionError(f'Price change in asset {j}')
    if old_state.liquidity[i] / old_state.shares[i] != pytest.approx(new_state.liquidity[i] / new_state.shares[i]):
        raise AssertionError('Ratio of liquidity to shares changed')
    delta_r = new_agent.holdings[i] - old_agent.holdings[i]
    delta_q = new_agent.holdings['LRNA'] - old_agent.holdings['LRNA']
    if delta_q <= 0 and delta_q != pytest.approx(0):
        raise AssertionError('Delta Q < 0')
    if delta_r <= 0 and delta_r != pytest.approx(0):
        raise AssertionError('Delta R < 0')
    if initial_agent.holdings[i] != pytest.approx(new_agent.holdings[i], rel=1e-20):
        raise AssertionError('Agent did not get correct shares back.')

    piq = old_state.lrna_price(i)
    val_withdrawn = piq * delta_r + delta_q
    if (-2 * piq / (piq + p_init) * delta_S / old_state.shares[i] * piq
            * old_state.liquidity[i] != pytest.approx(val_withdrawn)
            and not new_state.fail):
        raise AssertionError('something is wrong')


@given(omnipool_config(token_count=3))
def test_remove_liquidity_min_fee(initial_state: oamm.OmnipoolState):
    min_fee = 0.0001
    i = initial_state.asset_list[2]
    initial_agent = Agent(
        holdings={token: 1000 for token in initial_state.asset_list + ['LRNA']},
    )
    # add LP shares to the pool
    old_state, old_agent = oamm.simulate_add_liquidity(initial_state, initial_agent, 1000, i)
    p_init = old_state.lrna_price(i)

    delta_S = -old_agent.holdings[('omnipool', i)]

    new_state, new_agent = oamm.simulate_remove_liquidity(old_state, old_agent, delta_S, i)
    for j in new_state.asset_list:
        if old_state.price(j) != pytest.approx(new_state.price(j)):
            raise AssertionError(f'Price change in asset {j}')
    if old_state.liquidity[i] / old_state.shares[i] >= new_state.liquidity[i] / new_state.shares[i]:
        raise AssertionError('Ratio of liquidity to shares decreased')
    delta_r = new_agent.holdings[i] - old_agent.holdings[i]
    delta_q = new_agent.holdings['LRNA'] - old_agent.holdings['LRNA']
    if delta_q <= 0 and delta_q != pytest.approx(0):
        raise AssertionError('Delta Q < 0')
    if delta_r <= 0 and delta_r != pytest.approx(0):
        raise AssertionError('Delta R < 0')

    piq = old_state.lrna_price(i)
    val_withdrawn = piq * delta_r + delta_q
    if (-2 * piq / (piq + p_init) * delta_S / old_state.shares[i] * piq
            * old_state.liquidity[i] * (1 - min_fee) != pytest.approx(val_withdrawn)
            and not new_state.fail):
        raise AssertionError('something is wrong')


@given(
    st.floats(min_value=0.0002, max_value=0.05),
    assets_config(token_count=3)
)
def test_remove_liquidity_dynamic_fee(price_diff: float, asset_dict: dict):
    i = list(asset_dict.keys())[2]

    test_state = oamm.OmnipoolState(
        tokens=asset_dict,
        asset_fee=0.0025,
        lrna_fee=0.0005,
        withdrawal_fee=True,
    )

    test_state.oracles['price'].price[i] /= (1 + price_diff)

    initial_agent = Agent(
        holdings={token: 1000 for token in test_state.asset_list + ['LRNA']},
    )
    # add LP shares to the pool
    old_state, old_agent = oamm.simulate_add_liquidity(test_state, initial_agent, 1000, i)
    p_init = old_state.lrna_price(i)

    delta_S = -old_agent.holdings[('omnipool', i)]

    new_state, new_agent = oamm.simulate_remove_liquidity(old_state, old_agent, delta_S, i)
    for j in new_state.asset_list:
        if old_state.price(j) != pytest.approx(new_state.price(j)):
            raise AssertionError(f'Price change in asset {j}')
    if old_state.liquidity[i] / old_state.shares[i] >= new_state.liquidity[i] / new_state.shares[i]:
        raise AssertionError('Ratio of liquidity to shares decreased')
    delta_r = new_agent.holdings[i] - old_agent.holdings[i]
    delta_q = new_agent.holdings['LRNA'] - old_agent.holdings['LRNA']
    if delta_q <= 0 and delta_q != pytest.approx(0):
        raise AssertionError('Delta Q < 0')
    if delta_r <= 0 and delta_r != pytest.approx(0):
        raise AssertionError('Delta R < 0')

    piq = old_state.lrna_price(i)
    val_withdrawn = piq * delta_r + delta_q

    x = -2 * piq / (piq + p_init)
    share_ratio = delta_S / old_state.shares[i]
    feeless_val = x * share_ratio * piq * old_state.liquidity[i]
    theoretical_val = feeless_val * (1 - price_diff)
    if theoretical_val != pytest.approx(val_withdrawn) and not new_state.fail:
        raise AssertionError('something is wrong')


@given(omnipool_config(token_count=3, withdrawal_fee=False),
       st.floats(min_value=0.001, max_value=0.2))
def test_remove_liquidity_no_fee_different_price(initial_state: oamm.OmnipoolState, trade_size_ratio: float):
    i = initial_state.asset_list[2]
    initial_agent = Agent(
        holdings={token: 1000 for token in initial_state.asset_list + ['LRNA']},
    )
    # add LP shares to the pool
    init_contrib = 1000
    old_state, old_agent = oamm.simulate_add_liquidity(initial_state, initial_agent, init_contrib, i)
    p_init = old_state.lrna_price(i)

    trader_agent = Agent(
        holdings={token: 1000 for token in initial_state.asset_list + ['LRNA']},
    )
    tkn2 = initial_state.asset_list[1]
    trade_state, _ = oamm.simulate_swap(old_state, trader_agent, tkn_buy=tkn2, tkn_sell=i,
                                        sell_quantity=initial_state.liquidity[i] * trade_size_ratio)

    delta_S = -old_agent.holdings[('omnipool', i)]

    new_state, new_agent = oamm.simulate_remove_liquidity(trade_state, old_agent, delta_S, i)
    for j in new_state.asset_list:
        if trade_state.price(j) != pytest.approx(new_state.price(j)):
            raise AssertionError(f'Price change in asset {j}')
    if trade_state.liquidity[i] / trade_state.shares[i] != pytest.approx(new_state.liquidity[i] / new_state.shares[i]):
        raise AssertionError('Ratio of liquidity to shares changed')
    delta_r = new_agent.holdings[i] - old_agent.holdings[i]
    delta_q = new_agent.holdings['LRNA'] - old_agent.holdings['LRNA']
    if delta_q <= 0 and delta_q != pytest.approx(0):
        raise AssertionError('Delta Q < 0')
    if delta_r <= 0 and delta_r != pytest.approx(0):
        raise AssertionError('Delta R < 0')

    piq = trade_state.lrna_price(i)
    val_withdrawn = piq * delta_r + delta_q
    value_percent = 2 * piq / (piq + p_init) * math.sqrt(piq / p_init)
    theoretical_val = value_percent * p_init * init_contrib
    if theoretical_val != pytest.approx(val_withdrawn) and not new_state.fail:
        raise AssertionError('something is wrong')


@given(st.floats(min_value=1, max_value=100),
       st.floats(min_value=0.1, max_value=0.9))
def test_remove_liquidity_split(price: float, split: float):
    liquidity = {'HDX': mpf(10000000), 'USD': mpf(1000000), 'DOT': mpf(100000)}
    lrna = {'HDX': mpf(1000000), 'USD': mpf(1000000), 'DOT': mpf(1000000)}
    initial_state = oamm.OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        withdrawal_fee=False
    )
    tkn = 'DOT'
    amt1 = initial_state.shares[tkn] / 5
    amt2 = amt1 * split
    amt3 = amt1 - amt2
    holdings1 = {(initial_state.unique_id, tkn): amt1}
    prices1 = {(initial_state.unique_id, tkn): price}
    agent1 = Agent(holdings=holdings1, share_prices=prices1)

    nft1 = OmnipoolLiquidityPosition(tkn, price, amt2, 0, initial_state.unique_id)
    nft2 = OmnipoolLiquidityPosition(tkn, price, amt3, 0, initial_state.unique_id)
    nfts = {'nft001': nft1, 'nft002': nft2}
    agent2 = Agent(nfts=nfts)

    state1 = initial_state.copy()
    state2 = initial_state.copy()
    state1.remove_liquidity(agent1, tkn_remove=tkn)
    state2.remove_liquidity(agent2, tkn_remove=tkn)

    if state1.liquidity[tkn] != pytest.approx(state2.liquidity[tkn], rel=1e-20):
        raise AssertionError('liquidity should match')
    if state1.shares[tkn] != pytest.approx(state2.shares[tkn], rel=1e-20):
        raise AssertionError('shares should match')
    if agent1.holdings[tkn] != pytest.approx(agent2.holdings[tkn], rel=1e-20):
        raise AssertionError('holdings should match')
    if agent1.holdings[(initial_state.unique_id, tkn)] != 0:
        raise AssertionError('holdings of shares should be zero')
    if len(agent2.nfts) != 0:
        raise AssertionError('LP positions should be removed')


@given(
    delta_qa=st.floats(min_value=-10000, max_value=-1),
    buy_index=st.integers(min_value=0, max_value=2)
)
@settings(print_blob=True)
def test_swap_lrna(delta_qa: float, buy_index: int):
    initial_state = oamm.OmnipoolState(
        tokens={
            "HDX": {'liquidity': mpf(1_000_000), 'LRNA': mpf(100_000)},
            "DOT": {'liquidity': mpf(500_000), 'LRNA': mpf(50_000)},
            "USD": {'liquidity': mpf(200_000), 'LRNA': mpf(20_000)},
        },
        asset_fee=0.0025,
        lrna_fee=0.0005,
        slip_factor=1.0
    )
    old_state = initial_state
    old_agent = Agent(enforce_holdings=False)
    i = old_state.asset_list[buy_index + 1]

    # Test with trader buying asset i
    feeless_state = initial_state.copy()
    feeless_state.lrna_fee = 0
    feeless_state.asset_fee = 0
    for asset in feeless_state.asset_list:
        feeless_state.last_lrna_fee[asset] = 0
        feeless_state.last_fee[asset] = 0

    # Test with trader selling LRNA
    new_state, new_agent = oamm.simulate_swap(
        old_state, old_agent,
        sell_quantity=-delta_qa,
        tkn_buy=i,
        tkn_sell='LRNA'
    )
    feeless_swap_state, feeless_swap_agent = oamm.simulate_swap(
        feeless_state, old_agent,
        sell_quantity=-delta_qa,
        tkn_buy=i,
        tkn_sell='LRNA'
    )
    if oamm.asset_invariant(feeless_swap_state, i) != pytest.approx(oamm.asset_invariant(old_state, i)):
        raise AssertionError('Invariant not respected in feeless trade.')
    for j in old_state.liquidity:
        if min(new_state.liquidity[j] - feeless_swap_state.liquidity[j], 0) != pytest.approx(0):
            raise AssertionError('Liquidity decreased.')
    if min(oamm.asset_invariant(new_state, i) / oamm.asset_invariant(old_state, i), 1) != pytest.approx(1):
        raise AssertionError('Invariant decreased.')

    delta_qi = new_state.lrna[i] - old_state.lrna[i]
    if new_state.liquidity[i] + new_agent.get_holdings(i) != pytest.approx(old_state.liquidity[i], rel=1e-16):
        raise AssertionError('System-wide asset total is wrong.')
    if new_state.lrna[i] + new_agent.get_holdings('LRNA') < old_state.lrna[i]:
        raise AssertionError('System-wide LRNA decreased.')

    # try swapping into LRNA and back to see if that's equivalent
    reverse_state, reverse_agent = oamm.simulate_swap(
        old_state=feeless_swap_state,
        old_agent=feeless_swap_agent,
        buy_quantity=-delta_qa,
        tkn_sell=i,
        tkn_buy='LRNA'
    )

    if reverse_agent.get_holdings(i) != pytest.approx(0, abs=1e-16):
        raise AssertionError('Agent holdings are wrong.')


@given(st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=0.0001, max_value=0.01),
       st.floats(min_value=0.0001, max_value=0.01))
def test_lrna_swap_buy_with_lrna_mint(
        hdx_liquidity: float,
        dot_liquidity: float,
        usd_liquidity: float,
        hdx_lrna: float,
        dot_lrna: float,
        usd_lrna: float,
        asset_fee: float,
        lrna_fee: float
):
    asset_dict = {
        'HDX': {'liquidity': hdx_liquidity, 'LRNA': hdx_lrna},
        'DOT': {'liquidity': dot_liquidity, 'LRNA': dot_lrna},
        'USD': {'liquidity': usd_liquidity, 'LRNA': usd_lrna},
    }

    initial_state = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=float('inf'),
        asset_fee=asset_fee,
        lrna_fee=lrna_fee,
        lrna_mint_pct=1.0
    )

    old_agent = Agent(
        holdings={token: 10000 for token in initial_state.asset_list}
    )

    i = 'DOT'

    delta_ra = 1000
    delta_ra_feeless = delta_ra / (1 - asset_fee)

    feeless_state = initial_state.copy()
    feeless_state.asset_fee = 0
    for asset in feeless_state.liquidity:
        feeless_state.last_fee[asset] = 0

    # Test with trader buying asset i
    swap_state = initial_state.copy().swap(
        old_agent.copy(),
        tkn_buy=i,
        tkn_sell='LRNA',
        buy_quantity=delta_ra
    )
    feeless_swap_state = feeless_state.copy().swap(
        old_agent.copy(),
        tkn_buy=i,
        tkn_sell='LRNA',
        buy_quantity=delta_ra_feeless
    )
    feeless_spot_price = feeless_swap_state.price(i)
    spot_price = swap_state.price(i)
    if feeless_swap_state.fail == '' and swap_state.fail == '':
        if feeless_spot_price != pytest.approx(spot_price, rel=1e-16):
            raise AssertionError('Spot price is wrong.')


@given(st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=0.0001, max_value=0.01),
       st.floats(min_value=0.0001, max_value=0.01))
def test_lrna_swap_sell_with_lrna_mint(
        hdx_liquidity: float,
        dot_liquidity: float,
        usd_liquidity: float,
        hdx_lrna: float,
        dot_lrna: float,
        usd_lrna: float,
        asset_fee: float,
        lrna_fee: float
):
    asset_dict = {
        'HDX': {'liquidity': hdx_liquidity, 'LRNA': hdx_lrna},
        'DOT': {'liquidity': dot_liquidity, 'LRNA': dot_lrna},
        'USD': {'liquidity': usd_liquidity, 'LRNA': usd_lrna},
    }

    initial_state = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=float('inf'),
        asset_fee=asset_fee,
        lrna_fee=lrna_fee,
        lrna_mint_pct=1.0
    )

    old_agent = Agent(
        holdings={token: 10000 for token in initial_state.asset_list + ['LRNA']}
    )

    i = 'DOT'

    delta_qa = -1000

    feeless_state = initial_state.copy()
    feeless_state.asset_fee = 0
    for asset in feeless_state.asset_list:
        feeless_state.last_fee[asset] = 0

    # Test with trader buying asset i
    swap_state, swap_agent = oamm.simulate_swap(
        initial_state, old_agent,
        tkn_buy=i,
        tkn_sell='LRNA',
        sell_quantity=delta_qa
    )
    feeless_swap_state, feeless_swap_agent = oamm.simulate_swap(
        feeless_state, old_agent,
        tkn_buy=i,
        tkn_sell='LRNA',
        sell_quantity=delta_qa
    )
    feeless_spot_price = feeless_swap_state.price(i)
    spot_price = swap_state.price(i)
    if feeless_swap_state.fail == '' and swap_state.fail == '':
        if feeless_spot_price != pytest.approx(spot_price, rel=1e-16):
            raise AssertionError('Spot price is wrong.')


@given(st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=0.0001, max_value=0.01),
       st.floats(min_value=0.0001, max_value=0.01), )
def test_sell_with_lrna_mint(
        hdx_liquidity: float,
        dot_liquidity: float,
        usd_liquidity: float,
        hdx_lrna: float,
        dot_lrna: float,
        usd_lrna: float,
        asset_fee: float,
        lrna_fee: float,
):
    asset_dict = {
        'HDX': {'liquidity': hdx_liquidity, 'LRNA': hdx_lrna},
        'DOT': {'liquidity': dot_liquidity, 'LRNA': dot_lrna},
        'USD': {'liquidity': usd_liquidity, 'LRNA': usd_lrna},
    }

    initial_state = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=float('inf'),
        asset_fee=asset_fee,
        lrna_fee=lrna_fee,
        lrna_mint_pct=1.0
    )

    old_agent = Agent(
        holdings={token: 10000 for token in initial_state.asset_list + ['LRNA']}
    )

    i = 'DOT'
    j = 'USD'

    delta_ri = 1000

    feeless_state = initial_state.copy()
    feeless_state.asset_fee = 0
    for asset in feeless_state.asset_list:
        feeless_state.last_fee[asset] = 0

    # Test with trader buying asset i
    swap_state, swap_agent = oamm.simulate_swap(initial_state, old_agent, j, i, 0, delta_ri)
    feeless_swap_state, feeless_swap_agent = oamm.simulate_swap(feeless_state, old_agent, j, i, 0, delta_ri)
    feeless_spot_price = feeless_swap_state.price(j)
    spot_price = swap_state.price(j)
    if feeless_swap_state.fail == '' and swap_state.fail == '':
        if feeless_spot_price != pytest.approx(spot_price, rel=1e-14):
            raise AssertionError('Spot price is wrong.')


@given(st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=0.0001, max_value=0.01), )
def test_buy_with_lrna_mint(
        hdx_liquidity: float,
        dot_liquidity: float,
        usd_liquidity: float,
        hdx_lrna: float,
        dot_lrna: float,
        usd_lrna: float,
        asset_fee: float
):
    asset_dict = {
        'HDX': {'liquidity': mpf(hdx_liquidity), 'LRNA': mpf(hdx_lrna)},
        'DOT': {'liquidity': mpf(dot_liquidity), 'LRNA': mpf(dot_lrna)},
        'USD': {'liquidity': mpf(usd_liquidity), 'LRNA': mpf(usd_lrna)},
    }

    initial_state = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=float('inf'),
        asset_fee=asset_fee,
        lrna_fee=0.0,
        lrna_mint_pct=1.0
    )

    old_agent = Agent(
        holdings={token: 10000 for token in initial_state.asset_list + ['LRNA']}
    )

    i = 'DOT'
    j = 'USD'

    delta_rj = 1000
    delta_rj_feeless = delta_rj / (1 - asset_fee)

    feeless_state = initial_state.copy()
    feeless_state.asset_fee = 0
    for asset in feeless_state.asset_list:
        feeless_state.last_fee[asset] = 0

    # Test with trader buying asset i
    swap_state, swap_agent = oamm.simulate_swap(initial_state, old_agent, j, i, delta_rj, 0)
    feeless_swap_state, feeless_swap_agent = oamm.simulate_swap(feeless_state, old_agent, j, i, delta_rj_feeless, 0)
    feeless_spot_price = feeless_swap_state.price(j)
    spot_price = swap_state.price(j)
    if feeless_swap_state.fail == '' and swap_state.fail == '':
        if feeless_spot_price != pytest.approx(spot_price, rel=1e-16):
            raise AssertionError('Spot price is wrong.')


@given(st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=10000, max_value=10000000),
       st.floats(min_value=0.0001, max_value=0.01),
       st.floats(min_value=0.0001, max_value=0.01), )
def test_sell_with_partial_lrna_mint(
        hdx_liquidity: float,
        dot_liquidity: float,
        usd_liquidity: float,
        hdx_lrna: float,
        dot_lrna: float,
        usd_lrna: float,
        asset_fee: float,
        lrna_fee: float,
):
    asset_dict = {
        'HDX': {'liquidity': hdx_liquidity, 'LRNA': hdx_lrna},
        'DOT': {'liquidity': dot_liquidity, 'LRNA': dot_lrna},
        'USD': {'liquidity': usd_liquidity, 'LRNA': usd_lrna},
    }

    initial_state_0 = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=float('inf'),
        asset_fee=asset_fee,
        lrna_fee=lrna_fee,
        lrna_mint_pct=0.0
    )

    initial_state_50 = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=float('inf'),
        asset_fee=asset_fee,
        lrna_fee=lrna_fee,
        lrna_mint_pct=0.5
    )

    initial_state_100 = oamm.OmnipoolState(
        tokens=asset_dict,
        tvl_cap=float('inf'),
        asset_fee=asset_fee,
        lrna_fee=lrna_fee,
        lrna_mint_pct=1.0
    )

    old_agent = Agent(
        holdings={token: 10000 for token in initial_state_0.asset_list + ['LRNA']}
    )

    i = 'DOT'
    j = 'USD'

    delta_ri = 1000

    # Test with trader buying asset i
    swap_state_100, swap_agent_100 = oamm.simulate_swap(
        initial_state_100, copy.deepcopy(old_agent), j, i, 0, delta_ri
    )
    swap_state_50, swap_agent_50 = oamm.simulate_swap(
        initial_state_50, copy.deepcopy(old_agent), j, i, 0, delta_ri
    )
    swap_state_0, swap_agent_0 = oamm.simulate_swap(
        initial_state_0, copy.deepcopy(old_agent), j, i, 0, delta_ri
    )

    spot_price_100 = swap_state_100.price(j)
    spot_price_50 = swap_state_50.price(j)
    spot_price_0 = swap_state_0.price(j)

    if swap_state_100.fail == '' and swap_state_50.fail == '' and swap_state_0.fail == '':
        if spot_price_100 <= spot_price_50:
            raise AssertionError('Spot price is wrong.')
        if spot_price_50 <= spot_price_0:
            raise AssertionError('Spot price is wrong.')


@given(omnipool_reasonable_config(token_count=3, lrna_fee=0.0005, asset_fee=0.0025))
def test_lrna_buy_nonzero_fee(initial_state: oamm.OmnipoolState):
    initial_state.lrna_fee_burn = 0
    old_state = initial_state
    old_agent = Agent(
        holdings={token: 1000000 for token in initial_state.asset_list + ['LRNA']}
    )
    delta_qa = 10
    i = old_state.asset_list[2]

    # Test with trader selling asset i
    new_state, new_agent = oamm.simulate_swap(
        old_state, old_agent,
        tkn_sell=i,
        tkn_buy='LRNA',
        buy_quantity=delta_qa
    )

    expected_delta_qi = -delta_qa / (1 - initial_state.lrna_fee(i))
    expected_lrna_fee = -(delta_qa + expected_delta_qi)

    if old_state.lrna[i] - new_state.lrna[i] != pytest.approx(
            new_agent.holdings['LRNA'] - old_agent.holdings['LRNA'] + expected_lrna_fee):
        raise AssertionError('Delta Qi is wrong.')

    if old_state.lrna_total - new_state.lrna_total != pytest.approx(
            new_agent.holdings['LRNA'] - old_agent.holdings['LRNA'] + expected_lrna_fee):
        raise AssertionError('Some LRNA is being incorrectly burned or minted.')


@given(omnipool_config(token_count=3), st.integers(min_value=1, max_value=2))
def test_swap_assets(initial_state: oamm.OmnipoolState, i):
    initial_state.lrna_mint_pct = 0
    i_buy = initial_state.asset_list[i]
    old_state = initial_state

    old_agent = Agent(
        holdings={token: 10000 for token in initial_state.asset_list + ['LRNA']}
    )
    sellable_tokens = len(old_state.asset_list) - 1
    i_sell = old_state.asset_list[i % sellable_tokens + 1]
    delta_R = min(1000, old_state.liquidity[i_sell] / 2, old_state.liquidity[i_buy] / 2)

    # Test with trader selling asset i, no LRNA fee... price should match feeless
    new_state, new_agent = \
        oamm.simulate_swap(old_state, old_agent, i_buy, i_sell, sell_quantity=delta_R)

    # create copies of the old state with fees removed
    asset_fee_only_state = old_state.copy()
    asset_fee_only_state.lrna_fee = 0
    feeless_state = asset_fee_only_state.copy()
    feeless_state.asset_fee = 0
    for asset in feeless_state.asset_list:
        feeless_state.last_lrna_fee[asset] = 0
        feeless_state.last_fee[asset] = 0

    asset_fee_only_state, asset_fee_only_agent = \
        oamm.simulate_swap(asset_fee_only_state, old_agent, i_buy, i_sell, sell_quantity=delta_R)
    feeless_state, feeless_agent = \
        oamm.simulate_swap(feeless_state, old_agent, i_buy, i_sell, sell_quantity=delta_R)

    for j in old_state.liquidity:
        # assets in pools only go up compared to asset_fee_only_state
        if min(asset_fee_only_state.liquidity[j] - feeless_state.liquidity[j], 0) != pytest.approx(0):
            raise AssertionError("asset in pool {j} is lesser when compared with no-fee case")
        # asset in pool goes up from asset_fee_only_state -> new_state (i.e. introduction of LRNA fee)
        if min(new_state.liquidity[j] - asset_fee_only_state.liquidity[j], 0) != pytest.approx(0):
            raise AssertionError("asset in pool {j} is lesser when LRNA fee is added vs only asset fee")
        # invariant does not decrease
        if min(oamm.asset_invariant(new_state, j) / oamm.asset_invariant(old_state, j), 1) != pytest.approx(1):
            raise AssertionError("invariant ratio less than zero")
        # total quantity of R_i remains unchanged
        if (old_state.liquidity[j] + old_agent.holdings[j]
                != pytest.approx(new_state.liquidity[j] + new_agent.holdings[j])):
            raise AssertionError("total quantity of R[{j}] changed")

    # test that no LRNA is lost
    delta_Qi = feeless_state.lrna[i_sell] - old_state.lrna[i_sell]
    delta_Qj = feeless_state.lrna[i_buy] - old_state.lrna[i_buy]
    if delta_Qj + delta_Qi != pytest.approx(0, rel=1e-12):
        raise AssertionError('Some LRNA was lost along the way.')

    delta_out_new = feeless_agent.holdings[i_buy] - old_agent.holdings[i_buy]

    # Test with trader buying asset i, no LRNA fee... price should match feeless
    buy_state = old_state.copy()
    buy_state.lrna_fee = 0
    buy_state.asset_fee = 0
    buy_state, buy_agent = oamm.simulate_swap(
        buy_state, old_agent, i_buy, i_sell, buy_quantity=delta_out_new
    )

    for j in old_state.liquidity:
        if not buy_state.liquidity[j] == pytest.approx(feeless_state.liquidity[j]):
            raise AssertionError(f'Liquidity mismatch in {j}')
        if not buy_state.lrna[j] == pytest.approx(feeless_state.lrna[j]):
            raise AssertionError(f'LRNA mismatch in {j}')
        if not (
                old_state.liquidity[j] + old_agent.holdings[j] ==
                pytest.approx(buy_state.liquidity[j] + buy_agent.holdings[j])
        ):
            raise AssertionError('Change in the total quantity of {j}.')
        # assert buy_agent.holdings[j] == pytest.approx(feeless_agent.holdings[j])
        # assert buy_agent.holdings['LRNA'] == pytest.approx(feeless_agent.holdings['LRNA'])


@given(
    list_offset=st.integers(min_value=1, max_value=4),
    slip_factor = st.floats(min_value=0.1, max_value=2)
)
def test_slip_fees(list_offset: int, slip_factor: float):
    omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': 2000000, 'LRNA': 100000},
            'USD': {'liquidity': 1000000, 'LRNA': 200000},
            'R1': {'liquidity': 30000, 'LRNA': 300000},
            'R2': {'liquidity': 9000, 'LRNA': 400000},
        },
        slip_factor=slip_factor
    )
    omnipool.max_lrna_fee = 1.0
    n = len(omnipool.asset_list)
    trades = [
        # trade every possible pair in both directions
        {"tkn_buy": omnipool.asset_list[i % n], "tkn_sell": omnipool.asset_list[j % n], "sell_quantity": 100 - j - i}
        for i in range(1, n)
        for j in range(1, n)
        if i != j
    ]
    offset_trades = []
    for i in range(0, len(trades) * list_offset, list_offset):
        offset_trades.append(trades.pop(i % len(trades)))
    trades = offset_trades

    agent = Agent(enforce_holdings=False)
    for trade in trades:
        tkn_buy = trade['tkn_buy']
        tkn_sell = trade['tkn_sell']
        sell_quantity = trade['sell_quantity']
        split_sell_state, split_sell_agent = omnipool.copy(), agent.copy()
        for i in range(2):
            split_sell_state.swap(
                agent=split_sell_agent,
                tkn_sell=tkn_sell,
                tkn_buy=tkn_buy,
                sell_quantity=sell_quantity / 2
            )
        omnipool.copy().swap(
            agent, tkn_buy, tkn_sell, sell_quantity=sell_quantity
        )
        if split_sell_agent.get_holdings(tkn_buy) <= agent.get_holdings(tkn_buy):
            raise AssertionError('Agent failed to save money by splitting the sell order.')

        buy_quantity = sell_quantity
        tkn_sell, tkn_buy = tkn_buy, tkn_sell
        agent.holdings = {}
        split_buy_state, split_buy_agent = omnipool.copy(), agent.copy()
        slip_fee_omnipool = omnipool.compute_slip_fee(tkn_sell, buy_quantity)
        slip_fee_split = []
        for i in range(2):
            slip_fee_split.append(split_buy_state.compute_slip_fee(tkn_sell, buy_quantity / 2))
            split_buy_state.swap(
                agent=split_buy_agent,
                tkn_sell=tkn_sell,
                tkn_buy=tkn_buy,
                buy_quantity=buy_quantity / 2
            )
        omnipool.copy().swap(agent, tkn_sell=tkn_sell, tkn_buy=tkn_buy, buy_quantity=buy_quantity)
        if split_buy_agent.get_holdings(tkn_sell) <= agent.get_holdings(tkn_sell):
            raise AssertionError('Agent failed to save money by splitting the buy order.')


def test_trade_limit():
    initial_state = oamm.OmnipoolState(
        tokens={
            'HDX': {'liquidity': 1000000, 'LRNA': 1000000},
            'USD': {'liquidity': 1000000, 'LRNA': 1000000},
            'R1': {'liquidity': 1000000, 'LRNA': 1000000},
        },
        trade_limit_per_block=0.25
    )
    agent = Agent(
        holdings={'HDX': 1000000, 'USD': 1000000, 'R1': 1000000, 'LRNA': 1000000}
    )
    new_state = initial_state.copy()
    trades_allowed = 0
    while not new_state.fail:
        new_state, new_agent = oamm.simulate_swap(
            new_state, agent, 'USD', 'R1', sell_quantity=100000
        )
        if new_state.fail:
            break
        trades_allowed += 1

    assert trades_allowed == 2

    new_state = initial_state.copy()
    for i in range(26):
        if new_state.fail:
            raise AssertionError('Not enough trades allowed')
        new_state, new_agent = oamm.simulate_swap(
            new_state, agent, 'R1', 'USD', buy_quantity=1000
        )
        new_state, new_agent = oamm.simulate_swap(
            new_state, agent, 'USD', 'R1', sell_quantity=11000
        )

    if not new_state.fail:
        raise AssertionError('Too many trades allowed')


@given(
    st.floats(min_value=1e-5, max_value=1e5),
)
def test_dynamic_fees(hdx_price: float):
    initial_state = oamm.OmnipoolState(
        tokens={
            'HDX': {'liquidity': 100000 / hdx_price, 'LRNA': 100000},
            'USD': {'liquidity': 100000, 'LRNA': 100000},
            'R1': {'liquidity': 100000, 'LRNA': 100000},
        },
        asset_fee=DynamicFee(
            minimum=0.0025,
            amplification=10,
            decay=0.0005,
            maximum=0.40,
            current={'R1': 0.1, 'HDX': 0.0025, 'USD': 0.0025}
        ),
        lrna_fee=DynamicFee(
            minimum=0.0005,
            amplification=10,
            decay=0.0001,
            maximum=0.10,
            current={'R1': 0.1, 'HDX': 0.0005, 'USD': 0.0005}
        )
    )
    initial_hdx_fee = initial_state.asset_fee('HDX')
    initial_usd_fee = initial_state.asset_fee('USD')
    initial_usd_lrna_fee = initial_state.lrna_fee('USD')
    initial_hdx_lrna_fee = initial_state.lrna_fee('HDX')
    initial_R1_fee = initial_state.asset_fee('R1')
    initial_R1_lrna_fee = initial_state.lrna_fee('R1')
    test_agent = Agent(
        holdings={tkn: initial_state.liquidity[tkn] / 100 for tkn in initial_state.liquidity}
    )
    test_state = initial_state.copy()
    test_state.swap(
        agent=test_agent,
        tkn_sell='USD',
        tkn_buy='HDX',
        sell_quantity=test_agent.holdings['USD']
    )
    test_state.update()

    if test_state.asset_fee('R1') >= initial_R1_fee:
        raise AssertionError('R1 fee should be decreasing due to decay.')
    if test_state.lrna_fee('R1') >= initial_R1_lrna_fee:
        raise AssertionError('R1 LRNA fee should be decreasing due to decay.')

    test_state.update()
    intermediate_hdx_fee = test_state.asset_fee('HDX')
    intermediate_usd_fee = test_state.asset_fee('USD')
    intermediate_usd_lrna_fee = test_state.lrna_fee('USD')
    intermediate_hdx_lrna_fee = test_state.lrna_fee('HDX')
    if not intermediate_hdx_fee > initial_hdx_fee:
        raise AssertionError('Fee should increase when price increases.')
    if not intermediate_usd_lrna_fee > initial_usd_lrna_fee:
        raise AssertionError('LRNA fee should increase when price decreases.')
    if not intermediate_usd_fee == initial_usd_fee:
        raise AssertionError('Asset fee should not change.')
    if not intermediate_hdx_lrna_fee == initial_hdx_lrna_fee:
        raise AssertionError('LRNA fee should not change.')

    test_state.swap(
        agent=test_agent,
        tkn_sell='HDX',
        tkn_buy='USD',
        sell_quantity=test_agent.holdings['HDX']
    )
    test_state.update()
    final_hdx_fee = test_state.asset_fee('HDX')
    final_usd_fee = test_state.asset_fee('USD')
    final_usd_lrna_fee = test_state.lrna_fee('USD')
    final_hdx_lrna_fee = test_state.lrna_fee('HDX')
    if not final_usd_fee > intermediate_usd_fee:
        raise AssertionError('Fee should increase when price increases.')
    if not final_hdx_lrna_fee > intermediate_hdx_lrna_fee:
        raise AssertionError('LRNA fee should increase when price decreases.')
    if not final_hdx_fee < intermediate_hdx_fee:
        raise AssertionError('Asset fee should decrease with time.')
    if not final_usd_lrna_fee < intermediate_usd_lrna_fee:
        raise AssertionError('LRNA fee should decrease with time.')


@given(num_blocks = st.integers(min_value=0, max_value=1000))
def test_dynamic_fee_multiple_block_update(num_blocks):
    init_vol_out = mpf(100)
    init_vol_in = 0
    W = mpf(0.2)
    amplification = 1
    decay = 1 / mpf(100000)
    init_liq = mpf(10000)
    init_liq_oracle = mpf(10000)
    R = init_liq - init_vol_out
    # num_blocks = 1000
    init_fee = mpf(0.0025)
    fee_min = mpf(0.0025)
    fee_max = mpf(0.1)
    fee = init_fee
    vol_out_oracle = init_vol_out
    vol_in_oracle = init_vol_in
    liquidity_oracle = W * (init_liq - init_vol_out) + (1-W) * init_liq_oracle
    for j in range(num_blocks):
        x = (vol_out_oracle - vol_in_oracle) / liquidity_oracle
        delta_fee = amplification * x - decay
        fee = min(max(fee + delta_fee, fee_min), fee_max)
        # oracle updates
        vol_out_oracle = vol_out_oracle * (1 - W)
        vol_in_oracle = vol_in_oracle * (1 - W)
        liquidity_oracle = W * R + (1 - W) * liquidity_oracle

    liquidity_oracle = W * (init_liq - init_vol_out) + (1-W) * init_liq_oracle

    fee_precision = 30
    test_state = OmnipoolState(
        tokens={
            'HDX': {'liquidity': 1, 'LRNA': 1},
            'R1':{'liquidity': R, 'LRNA': 1}
        },
        lrna_fee=DynamicFee(
            minimum=fee_min,
            amplification=amplification,
            decay=decay,
            maximum=fee_max,
            current={'R1': init_fee, 'HDX': 0},
            liquidity={'R1': liquidity_oracle, 'HDX': 0},
            net_volume={'R1': init_vol_out - init_vol_in, 'HDX': 0}
        ),
        dynamic_fee_precision=fee_precision
    )
    test_state.time_step = num_blocks
    fee2 = test_state.lrna_fee('R1')
    if num_blocks < fee_precision:
        if fee != pytest.approx(fee2, rel=1e-20):
            raise AssertionError('Fee is not correct within precision range.')
    else:
        if fee2 != pytest.approx(fee, rel=1e-03):
            raise AssertionError('Fee approximation is not correct.')


@given(
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_bounded_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_price_strategy, min_size=2, max_size=2),
    st.integers(min_value=10, max_value=1000),
)
def test_oracle_one_empty_block(liquidity: list[float], lrna: list[float], oracle_liquidity: list[float],
                                oracle_volume_in: list[float], oracle_volume_out: list[float],
                                oracle_prices: list[float], n):
    alpha = 2 / (n + 1)

    init_liquidity = {
        'HDX': {'liquidity': liquidity[0], 'LRNA': lrna[0]},
        'USD': {'liquidity': liquidity[1], 'LRNA': lrna[1]},
        'DOT': {'liquidity': liquidity[2], 'LRNA': lrna[2]},
    }

    init_oracle = {
        'liquidity': {'HDX': oracle_liquidity[0], 'USD': oracle_liquidity[1], 'DOT': oracle_liquidity[2]},
        'volume_in': {'HDX': oracle_volume_in[0], 'USD': oracle_volume_in[1], 'DOT': oracle_volume_in[2]},
        'volume_out': {'HDX': oracle_volume_out[0], 'USD': oracle_volume_out[1], 'DOT': oracle_volume_out[2]},
        'price': {'HDX': oracle_prices[0], 'USD': 1, 'DOT': oracle_prices[1]},
    }

    initial_omnipool = oamm.OmnipoolState(
        tokens=copy.deepcopy(init_liquidity),
        oracles={
            'price': n
        },
        asset_fee=0.0025,
        lrna_fee=0.0005,
        last_oracle_values={
            'price': copy.deepcopy(init_oracle)
        }
    )

    initial_state = GlobalState(
        pools={'omnipool': initial_omnipool},
        agents={}
    )

    events = run.run(initial_state=initial_state, time_steps=1, silent=True)
    omnipool_oracle = events[0].pools['omnipool'].oracles['price']
    for tkn in ['HDX', 'USD', 'DOT']:
        expected_liquidity = init_oracle['liquidity'][tkn] * (1 - alpha) + alpha * init_liquidity[tkn]['liquidity']
        if omnipool_oracle.liquidity[tkn] != expected_liquidity:
            raise AssertionError('Liquidity is not correct.')

        expected_vol_in = init_oracle['volume_in'][tkn] * (1 - alpha)
        if omnipool_oracle.volume_in[tkn] != expected_vol_in:
            raise AssertionError('Volume is not correct.')

        expected_vol_out = init_oracle['volume_out'][tkn] * (1 - alpha)
        if omnipool_oracle.volume_out[tkn] != expected_vol_out:
            raise AssertionError('Volume is not correct.')

        init_price = init_liquidity[tkn]['LRNA'] / init_liquidity[tkn]['liquidity']
        expected_price = init_oracle['price'][tkn] * (1 - alpha) + alpha * init_price
        if omnipool_oracle.price[tkn] != expected_price:
            raise AssertionError('Price is not correct.')


@given(
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_bounded_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_price_strategy, min_size=2, max_size=2),
    st.lists(st.floats(min_value=10, max_value=1000), min_size=2, max_size=2),
    st.integers(min_value=10, max_value=1000),
)
@settings(
    print_blob=True
)
def test_oracle_one_block_with_swaps(liquidity: list[float], lrna: list[float], oracle_liquidity: list[float],
                                     oracle_volume_in: list[float], oracle_volume_out: list[float],
                                     oracle_prices: list[float], trade_sizes: list[float], n):
    alpha = 2 / (n + 1)

    init_liquidity = {
        'HDX': {'liquidity': liquidity[0], 'LRNA': lrna[0]},
        'USD': {'liquidity': liquidity[1], 'LRNA': lrna[1]},
        'DOT': {'liquidity': liquidity[2], 'LRNA': lrna[2]},
    }

    init_oracle = {
        'liquidity': {'HDX': oracle_liquidity[0], 'USD': oracle_liquidity[1], 'DOT': oracle_liquidity[2]},
        'volume_in': {'HDX': oracle_volume_in[0], 'USD': oracle_volume_in[1], 'DOT': oracle_volume_in[2]},
        'volume_out': {'HDX': oracle_volume_out[0], 'USD': oracle_volume_out[1], 'DOT': oracle_volume_out[2]},
        'price': {'HDX': oracle_prices[0], 'USD': 1, 'DOT': oracle_prices[1]},
    }

    initial_omnipool = oamm.OmnipoolState(
        tokens=copy.deepcopy(init_liquidity),
        oracles={
            'price': n
        },
        asset_fee=0.0025,
        lrna_fee=0.0005,
        last_oracle_values={
            'price': copy.deepcopy(init_oracle)
        }
    )

    trader1_holdings = {'HDX': 1000000000, 'USD': 1000000000, 'LRNA': 1000000000, 'DOT': 1000000000}
    trader2_holdings = {'HDX': 1000000000, 'USD': 1000000000, 'LRNA': 1000000000, 'DOT': 1000000000}

    initial_state = GlobalState(
        pools={'omnipool': initial_omnipool},
        agents={
            'Trader1': Agent(
                holdings=trader1_holdings,
                trade_strategy=constant_swaps(
                    pool_id='omnipool',
                    sell_quantity=trade_sizes[0],
                    sell_asset='LRNA',
                    buy_asset='DOT'
                )
            ),
            'Trader2': Agent(
                holdings=trader2_holdings,
                trade_strategy=constant_swaps(
                    pool_id='omnipool',
                    sell_quantity=trade_sizes[1],
                    sell_asset='DOT',
                    buy_asset='LRNA'
                )
            ),
        }
    )

    events = run.run(initial_state=initial_state, time_steps=2, silent=True)
    omnipool_oracle_0 = events[0].pools['omnipool'].oracles['price']

    vol_in = {
        'HDX': 0,
        'USD': 0,
        'DOT': trader2_holdings['DOT'] - events[0].agents['Trader2'].holdings['DOT']
    }

    vol_out = {
        'HDX': 0,
        'USD': 0,
        'DOT': events[0].agents['Trader1'].holdings['DOT'] - trader1_holdings['DOT'],
    }

    for tkn in ['HDX', 'USD', 'DOT']:
        expected_liquidity = init_oracle['liquidity'][tkn] * (1 - alpha) + alpha * init_liquidity[tkn]['liquidity']
        if omnipool_oracle_0.liquidity[tkn] != expected_liquidity:
            raise AssertionError('Liquidity is not correct.')

        expected_vol_in = init_oracle['volume_in'][tkn] * (1 - alpha)
        if omnipool_oracle_0.volume_in[tkn] != expected_vol_in:
            raise AssertionError('Volume is not correct.')

        expected_vol_out = init_oracle['volume_out'][tkn] * (1 - alpha)
        if omnipool_oracle_0.volume_out[tkn] != expected_vol_out:
            raise AssertionError('Volume is not correct.')

        init_price = init_liquidity[tkn]['LRNA'] / init_liquidity[tkn]['liquidity']
        expected_price = init_oracle['price'][tkn] * (1 - alpha) + alpha * init_price
        if omnipool_oracle_0.price[tkn] != expected_price:
            raise AssertionError('Price is not correct.')

    omnipool_oracle_1 = events[1].pools['omnipool'].oracles['price']
    for tkn in ['HDX', 'USD', 'DOT']:
        expected_liquidity = omnipool_oracle_0.liquidity[tkn] * (1 - alpha) + alpha * init_liquidity[tkn]['liquidity']
        if omnipool_oracle_1.liquidity[tkn] != pytest.approx(expected_liquidity, 1e-10):
            raise AssertionError('Liquidity is not correct.')

        expected_vol_in = omnipool_oracle_0.volume_in[tkn] * (1 - alpha) + alpha * vol_in[tkn]
        if omnipool_oracle_1.volume_in[tkn] != pytest.approx(expected_vol_in, 1e-9):
            raise AssertionError('Volume is not correct.')

        expected_vol_out = omnipool_oracle_0.volume_out[tkn] * (1 - alpha) + alpha * vol_out[tkn]
        if omnipool_oracle_1.volume_out[tkn] != pytest.approx(expected_vol_out, 1e-9):
            raise AssertionError('Volume out is not correct.')

        price_1 = events[0].pools['omnipool'].price(tkn)
        expected_price = omnipool_oracle_0.price[tkn] * (1 - alpha) + alpha * price_1
        if omnipool_oracle_1.price[tkn] != pytest.approx(expected_price, 1e-10):
            raise AssertionError('Price is not correct.')


@given(
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_bounded_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    st.lists(asset_price_strategy, min_size=2, max_size=2),
    st.integers(min_value=10, max_value=1000),
)
def test_dynamic_fees_empty_block(liquidity: list[float], lrna: list[float], oracle_liquidity: list[float],
                                  oracle_volume_in: list[float], oracle_volume_out: list[float],
                                  oracle_prices: list[float], n):

    init_liquidity = {
        'HDX': {'liquidity': liquidity[0], 'LRNA': lrna[0]},
        'USD': {'liquidity': liquidity[1], 'LRNA': lrna[1]},
        'DOT': {'liquidity': liquidity[2], 'LRNA': lrna[2]},
    }

    init_oracle = {
        'liquidity': {'HDX': oracle_liquidity[0], 'USD': oracle_liquidity[1], 'DOT': oracle_liquidity[2]},
        'volume_in': {'HDX': oracle_volume_in[0], 'USD': oracle_volume_in[1], 'DOT': oracle_volume_in[2]},
        'volume_out': {'HDX': oracle_volume_out[0], 'USD': oracle_volume_out[1], 'DOT': oracle_volume_out[2]},
        'price': {'HDX': oracle_prices[0], 'USD': 1, 'DOT': oracle_prices[1]},
    }

    init_lrna_fees = {
        'HDX': 0.0005,
        'USD': 0.0010,
        'DOT': 0.0050,
    }

    init_asset_fees = {
        'HDX': 0.01,
        'USD': 0.0025,
        'DOT': 0.0040,
    }

    asset_fee_params = {
        'minimum': 0.0025,
        'amplification': 0.2,
        'decay': 0.00005,
        'fee_max': 0.4,
    }

    lrna_fee_params = {
        'minimum': 0.0005,
        'amplification': 0.04,
        'decay': 0.00001,
        'fee_max': 0.1,
    }

    initial_omnipool = oamm.OmnipoolState(
        tokens=copy.deepcopy(init_liquidity),
        oracles={
            'price': n
        },
        asset_fee=DynamicFee(
            minimum=asset_fee_params['minimum'],
            amplification=asset_fee_params['amplification'],
            decay=asset_fee_params['decay'],
            maximum=asset_fee_params['fee_max'],
            current=copy.deepcopy(init_asset_fees)
        ),
        lrna_fee=DynamicFee(
            minimum=lrna_fee_params['minimum'],
            amplification=lrna_fee_params['amplification'],
            decay=lrna_fee_params['decay'],
            maximum=lrna_fee_params['fee_max'],
            current=copy.deepcopy(init_lrna_fees)
        ),
        last_oracle_values={
            'price': copy.deepcopy(init_oracle)
        },
        update_function=lambda self: [self.lrna_fee(tkn) + self.asset_fee(tkn) for tkn in self.liquidity]
    )

    initial_state = GlobalState(
        pools={'omnipool': initial_omnipool},
        agents={}
    )

    events = run.run(initial_state=initial_state, time_steps=1, silent=True)
    omnipool = events[0].pools['omnipool']
    omnipool_oracle = omnipool.oracles['price']
    for tkn in ['HDX', 'USD', 'DOT']:
        x = (omnipool_oracle.volume_out[tkn] - omnipool_oracle.volume_in[tkn]) / omnipool_oracle.liquidity[tkn]

        df = -lrna_fee_params['amplification'] * x - lrna_fee_params['decay']
        expected_lrna_fee = min(max(init_lrna_fees[tkn] + df, lrna_fee_params['minimum']), lrna_fee_params['fee_max'])
        if omnipool.last_lrna_fee[tkn] != pytest.approx(expected_lrna_fee, rel=1e-15):
            raise AssertionError('LRNA fee is not correct.')

        df = asset_fee_params['amplification'] * x - asset_fee_params['decay']
        expected_asset_fee = min(max(init_asset_fees[tkn] + df, asset_fee_params['minimum']),
                                 asset_fee_params['fee_max'])
        if omnipool.last_fee[tkn] != pytest.approx(expected_asset_fee, rel=1e-15):
            raise AssertionError('Asset fee is not correct.')


@given(
    liquidity=st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    lrna=st.lists(asset_quantity_bounded_strategy, min_size=3, max_size=3),
    oracle_liquidity=st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    oracle_volume_in=st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    oracle_volume_out=st.lists(asset_quantity_strategy, min_size=3, max_size=3),
    oracle_period=st.integers(min_value=10, max_value=1000),
    trade_size=st.floats(min_value=-1000, max_value=1000),
    lrna_fees=st.lists(st.floats(min_value=0.0005, max_value=0.10), min_size=3, max_size=3),
    asset_fees=st.lists(st.floats(min_value=0.0025, max_value=0.40), min_size=3, max_size=3),
    amp=st.lists(st.floats(min_value=0.001, max_value=100), min_size=2, max_size=2),
    decay=st.lists(st.floats(min_value=0.000001, max_value=0.0001), min_size=2, max_size=2),
)
def test_dynamic_fees_with_trade(liquidity: list[float], lrna: list[float], oracle_liquidity: list[float],
                                 oracle_volume_in: list[float], oracle_volume_out: list[float],
                                 oracle_period, trade_size: float, lrna_fees: list[float],
                                 asset_fees: list[float], amp: list[float], decay: list[float]):
    assume(trade_size != 0)
    init_liquidity = {
        'HDX': {'liquidity': liquidity[0], 'LRNA': lrna[0]},
        'USD': {'liquidity': liquidity[1], 'LRNA': lrna[1]},
        'DOT': {'liquidity': liquidity[2], 'LRNA': lrna[2]},
    }

    init_oracle = {
        'liquidity': {'HDX': oracle_liquidity[0], 'USD': oracle_liquidity[1], 'DOT': oracle_liquidity[2]},
        'volume_in': {'HDX': oracle_volume_in[0], 'USD': oracle_volume_in[1], 'DOT': oracle_volume_in[2]},
        'volume_out': {'HDX': oracle_volume_out[0], 'USD': oracle_volume_out[1], 'DOT': oracle_volume_out[2]},
        'price': {'HDX': 1, 'USD': 1, 'DOT': 1},  # this is not relevant to the fee calculation
    }

    init_lrna_fees = {
        'HDX': lrna_fees[0],
        'USD': lrna_fees[1],
        'DOT': lrna_fees[2],
    }

    init_asset_fees = {
        'HDX': asset_fees[0],
        'USD': asset_fees[1],
        'DOT': asset_fees[2],
    }

    asset_fee_params = {
        'minimum': 0.0025,
        'amplification': amp[0],
        'raise_oracle_name': 'price',
        'decay': decay[0],
        'fee_max': 0.4,
    }

    lrna_fee_params = {
        'minimum': 0.0005,
        'amplification': amp[1],
        'decay': decay[1],
        'fee_max': 0.1,
    }

    initial_omnipool = oamm.OmnipoolState(
        tokens=copy.deepcopy(init_liquidity),
        oracles={
            'price': oracle_period
        },
        asset_fee=DynamicFee(
            minimum=asset_fee_params['minimum'],
            amplification=asset_fee_params['amplification'],
            decay=asset_fee_params['decay'],
            maximum=asset_fee_params['fee_max'],
            current=copy.deepcopy(init_asset_fees)
        ),
        lrna_fee=DynamicFee(
            minimum=lrna_fee_params['minimum'],
            amplification=lrna_fee_params['amplification'],
            decay=lrna_fee_params['decay'],
            maximum=lrna_fee_params['fee_max'],
            current=copy.deepcopy(init_lrna_fees)
        ),
        last_oracle_values={
            'price': copy.deepcopy(init_oracle)
        },
        lrna_fee_burn=0,
        update_function=lambda self: [self.lrna_fee(tkn) + self.asset_fee(tkn) for tkn in self.liquidity]
    )

    trader_holdings = {'HDX': 1000000000, 'USD': 1000000000, 'LRNA': 1000000000, 'DOT': 1000000000}

    initial_state = GlobalState(
        pools={'omnipool': initial_omnipool},
        agents={
            'trader': Agent(
                holdings=trader_holdings,
                trade_strategy=constant_swaps(
                    pool_id='omnipool',
                    sell_quantity=trade_size,
                    sell_asset='USD',
                    buy_asset='DOT'
                )
            ),
        }
    )

    events = run.run(initial_state=initial_state, time_steps=3, silent=True)

    # test non-empty block fee dynamics

    omnipool = events[1].pools['omnipool']
    prev_lrna_fees = events[0].pools['omnipool'].last_lrna_fee
    prev_asset_fees = events[0].pools['omnipool'].last_fee
    omnipool_oracle = omnipool.oracles['price']
    for tkn in ['HDX', 'USD', 'DOT']:
        x = (omnipool_oracle.volume_out[tkn] - omnipool_oracle.volume_in[tkn]) / omnipool_oracle.liquidity[tkn]

        df = -lrna_fee_params['amplification'] * x - lrna_fee_params['decay']
        expected_lrna_fee = min(max(prev_lrna_fees[tkn] + df, lrna_fee_params['minimum']), lrna_fee_params['fee_max'])
        if omnipool.last_lrna_fee[tkn] != pytest.approx(expected_lrna_fee, rel=1e-15):
            raise AssertionError('LRNA fee is not correct.')

        df = asset_fee_params['amplification'] * x - asset_fee_params['decay']
        expected_asset_fee = min(max(prev_asset_fees[tkn] + df, asset_fee_params['minimum']),
                                 asset_fee_params['fee_max'])
        if omnipool.last_fee[tkn] != pytest.approx(expected_asset_fee, rel=1e-15):
            raise AssertionError('Asset fee is not correct.')


@given(asset_quantity_strategy, omnipool_config())
def test_LP_delta_r(lp_amount, omnipool: oamm.OmnipoolState):
    agent = Agent(
        holdings={
            tkn: 100000000 for tkn in omnipool.asset_list
        }
    )
    initial_asset_holdings = copy.deepcopy(agent.holdings)
    omnipool.add_liquidity(
        agent=agent,
        quantity=lp_amount,
        tkn_add='HDX'
    )
    if agent.holdings['HDX'] != initial_asset_holdings['HDX'] - agent.delta_r[('omnipool', 'HDX')]:
        raise AssertionError('Delta_r is not correct.')


@given(omnipool_reasonable_config(remove_liquidity_volatility_threshold=0.01, asset_fee=0, lrna_fee=0))
def test_volatility_limit(initial_state: oamm.OmnipoolState):
    agent = Agent(holdings={'HDX': 1000000000})
    omnipool=initial_state.copy()
    omnipool.add_liquidity(agent, quantity=1000, tkn_add='HDX')
    omnipool.swap(agent, tkn_sell='HDX', tkn_buy='LRNA', sell_quantity=mpf(omnipool.liquidity['HDX'] / 200))
    omnipool.remove_liquidity(agent, quantity=1000, tkn_remove='HDX')

    if not omnipool.fail:
        raise ValueError("Volatility limit should be exceeded")

    # go forward one block, which should be enough for the volatility to decay
    updated_pool = omnipool.copy().update()

    updated_pool.remove_liquidity(agent, agent.holdings[('omnipool', 'HDX')], tkn_remove='HDX')
    if updated_pool.fail:
        raise ValueError("Volatility limit should not be exceeded")


@given(omnipool_reasonable_config(), st.floats(min_value=0.01, max_value=0.1), st.floats(min_value=0.01, max_value=0.1))
def test_LP_limits(omnipool: oamm.OmnipoolState, max_withdrawal_per_block, max_lp_per_block):
    omnipool.max_withdrawal_per_block = max_withdrawal_per_block
    omnipool.max_lp_per_block = max_lp_per_block
    state = omnipool.copy()
    initial_agent = Agent(holdings={'HDX': 10000000000})
    agent = initial_agent.copy()
    state.add_liquidity(
        agent=agent,
        tkn_add='HDX',
        quantity=state.liquidity['HDX'] * max_lp_per_block
    )
    if state.fail:
        raise AssertionError('Valid LP operation failed.')
    state = omnipool.copy()
    agent = initial_agent.copy()
    state.add_liquidity(
        agent=agent,
        tkn_add='HDX',
        quantity=state.liquidity['HDX'] * max_lp_per_block + 1
    )
    if not state.fail:
        raise AssertionError('Invalid LP operation succeeded.')
    state = omnipool.copy()
    agent = initial_agent.copy()
    # add liquidity again to test remove liquidity
    state.add_liquidity(
        agent=agent,
        tkn_add='HDX',
        quantity=state.liquidity['HDX'] * max_lp_per_block
    )
    if state.fail:
        raise AssertionError('Second LP operation failed.')
    withdraw_quantity = agent.holdings[('omnipool', 'HDX')]
    total_shares = state.shares['HDX']
    state.remove_liquidity(
        agent=agent,
        tkn_remove='HDX',
        quantity=withdraw_quantity  # agent.holdings[('omnipool', 'HDX')]
    )
    if withdraw_quantity / total_shares > max_withdrawal_per_block and not state.fail:
        raise AssertionError('Agent was able to remove too much liquidity.')
    state.update()
    state.remove_liquidity(
        agent=agent,
        tkn_remove='HDX',
        quantity=state.shares['HDX'] * max_withdrawal_per_block
    )
    if agent.validate_holdings(('omnipool', 'HDX')) and state.fail:
        raise AssertionError('Agent was not able to remove liquidity.')


@given(
    st.floats(min_value=0.50, max_value=1.5)
)
def test_liquidity_operations_and_spot_prices(oracle_mult):
    tokens = {
        'HDX': {'liquidity': 44000000, 'LRNA': 275143},
        'WETH': {'liquidity': 1400, 'LRNA': 2276599},
        'DAI': {'liquidity': 2268262, 'LRNA': 2268262},
        'DOT': {'liquidity': 88000, 'LRNA': 546461},
        'WBTC': {'liquidity': 47, 'LRNA': 1145210},
    }

    prices = {tkn: tokens[tkn]['LRNA'] / tokens[tkn]['liquidity'] for tkn in tokens}

    init_oracle = {
        'liquidity': {tkn: tokens[tkn]['liquidity'] for tkn in tokens},
        'volume_in': {tkn: 0 for tkn in tokens},
        'volume_out': {tkn: 0 for tkn in tokens},
        'price': {tkn: oracle_mult * prices[tkn] for tkn in tokens},
    }

    omnipool: oamm.OmnipoolState = oamm.OmnipoolState(
        tokens={
            'HDX': {'liquidity': 44000000, 'LRNA': 275143},
            'WETH': {'liquidity': 1400, 'LRNA': 2276599},
            'DAI': {'liquidity': 2268262, 'LRNA': 2268262},
            'DOT': {'liquidity': 88000, 'LRNA': 546461},
            'WBTC': {'liquidity': 47, 'LRNA': 1145210},
        },
        preferred_stablecoin='DAI',
        oracles={'price': 19, 'volatility': 19},
        last_oracle_values={
            'price': copy.deepcopy(init_oracle),
            'volatility': copy.deepcopy(init_oracle),
        },
    )

    agent = Agent(holdings={'DOT': 10000})
    add_state, add_agent = oamm.simulate_add_liquidity(
        old_state=omnipool.copy(),
        old_agent=agent.copy(),
        tkn_add='DOT',
        quantity=agent.holdings['DOT']
    )

    remove_state, remove_agent = oamm.simulate_remove_liquidity(
        old_state=add_state.copy(),
        old_agent=add_agent.copy(),
        tkn_remove='DOT',
        quantity=add_agent.holdings[('omnipool', 'DOT')]
    )

    if add_agent.holdings[('omnipool', 'DOT')] == 0:
        raise

    for tkn in omnipool.asset_list:
        initial_price = omnipool.price(tkn)
        add_price = add_state.price(tkn)
        remove_price = remove_state.price(tkn)
        if initial_price != pytest.approx(add_price, rel=1e-15):
            raise AssertionError('Price is not correct after add liquidity.')

        if initial_price != pytest.approx(remove_price, rel=1e-15):
            raise AssertionError('Price is not correct after remove liquidity.')


# @settings(max_examples=10000)
@given(
    st.floats(min_value=0, max_value=0.1, exclude_min=True),
    st.floats(min_value=-0.02, max_value=0, exclude_max=True),
    st.floats(min_value=0.50, max_value=1.5)
)
def test_lowering_price(lp_multiplier, price_movement, oracle_mult):
    # def test_lowering_price(lp_multiplier, price_movement):
    # lp_multiplier = 0.1
    # price_movement = -0.1
    # mult = 0.99

    tokens = {
        'HDX': {'liquidity': 44000000, 'LRNA': 275143},
        'WETH': {'liquidity': 1400, 'LRNA': 2276599},
        'DAI': {'liquidity': 2268262, 'LRNA': 2268262},
        'DOT': {'liquidity': 88000, 'LRNA': 546461},
        'WBTC': {'liquidity': 47, 'LRNA': 1145210},
    }

    trade_size = tokens['DOT']['liquidity'] * 1 / math.sqrt(1 + price_movement) - tokens['DOT']['liquidity']

    prices = {tkn: tokens[tkn]['LRNA'] / tokens[tkn]['liquidity'] for tkn in tokens}

    init_oracle = {
        'liquidity': {tkn: tokens[tkn]['liquidity'] for tkn in tokens},
        'volume_in': {tkn: 0 for tkn in tokens},
        'volume_out': {tkn: 0 for tkn in tokens},
        'price': {tkn: oracle_mult * prices[tkn] for tkn in tokens},
    }

    omnipool: oamm.OmnipoolState = oamm.OmnipoolState(
        tokens=tokens,
        preferred_stablecoin='DAI',
        oracles={'price': 19, 'volatility': 19},
        last_oracle_values={
            'price': copy.deepcopy(init_oracle),
            'volatility': copy.deepcopy(init_oracle),
        },
        withdrawal_fee=True,
        min_withdrawal_fee=0.0001,
    )

    market_prices = {tkn: omnipool.usd_price(tkn) for tkn in omnipool.asset_list}

    holdings = {tkn: 1000000000 for tkn in omnipool.asset_list + ['LRNA']}
    agent = Agent(holdings=holdings)

    swap_state, swap_agent = oamm.simulate_swap(
        old_state=omnipool.copy(),
        old_agent=agent.copy(),
        tkn_sell='DOT',
        tkn_buy='DAI',
        sell_quantity=trade_size
    )

    add_state, add_agent = oamm.simulate_add_liquidity(
        old_state=swap_state.copy(),
        old_agent=swap_agent.copy(),
        tkn_add='DOT',
        quantity=swap_state.liquidity['DOT'] * lp_multiplier
    )

    global_state = GlobalState(
        pools={'omnipool': add_state},
        agents={'attacker': add_agent},
        external_market=market_prices
    )

    arb_state = omnipool_arbitrage('omnipool', 20).execute(
        state=global_state.copy(),
        agent_id='attacker'
    )

    arbed_pool = arb_state.pools['omnipool']
    arbed_agent = arb_state.agents['attacker']

    remove_state, remove_agent = oamm.simulate_remove_liquidity(
        old_state=arbed_pool.copy(),
        old_agent=arbed_agent.copy(),
        tkn_remove='DOT',
        quantity=arbed_agent.holdings[('omnipool', 'DOT')]
    )

    initial_value = omnipool.cash_out(agent, market_prices)
    final_value = remove_state.cash_out(remove_agent, market_prices)
    profit = final_value - initial_value
    if profit > 0:
        raise


def test_add_and_remove_liquidity():
    lp_multiplier = 0.1
    oracle_mult = 0.99

    tokens = {
        'HDX': {'liquidity': 44000000, 'LRNA': 275143},
        'WETH': {'liquidity': 1400, 'LRNA': 2276599},
        'DAI': {'liquidity': 2268262, 'LRNA': 2268262},
        'DOT': {'liquidity': 88000, 'LRNA': 546461},
        'WBTC': {'liquidity': 47, 'LRNA': 1145210},
    }

    prices = {tkn: tokens[tkn]['LRNA'] / tokens[tkn]['liquidity'] for tkn in tokens}

    init_oracle = {
        'liquidity': {tkn: tokens[tkn]['liquidity'] for tkn in tokens},
        'volume_in': {tkn: 0 for tkn in tokens},
        'volume_out': {tkn: 0 for tkn in tokens},
        'price': {tkn: oracle_mult * prices[tkn] for tkn in tokens},
    }

    omnipool: oamm.OmnipoolState = oamm.OmnipoolState(
        tokens=tokens,
        preferred_stablecoin='DAI',
        oracles={'price': 19, 'volatility': 19},
        last_oracle_values={
            'price': copy.deepcopy(init_oracle),
            'volatility': copy.deepcopy(init_oracle),
        },
    )

    market_prices = {tkn: omnipool.usd_price(tkn) for tkn in omnipool.asset_list}

    holdings = {tkn: 1000000000 for tkn in omnipool.asset_list}
    agent = Agent(holdings=holdings)

    add_state, add_agent = oamm.simulate_add_liquidity(
        old_state=omnipool.copy(),
        old_agent=agent.copy(),
        tkn_add='DOT',
        quantity=omnipool.liquidity['DOT'] * lp_multiplier
    )

    remove_state, remove_agent = oamm.simulate_remove_liquidity(
        old_state=add_state.copy(),
        old_agent=add_agent.copy(),
        tkn_remove='DOT',
        quantity=add_agent.holdings[('omnipool', 'DOT')]
    )

    initial_value = omnipool.cash_out(agent, market_prices)
    final_value = remove_state.cash_out(remove_agent, market_prices)

    profit = final_value - initial_value
    if profit > 0:
        raise


@given(
    list_offset = st.integers(min_value=0, max_value=3),
    buy_quantities = st.lists(st.floats(min_value=1, max_value=1000), min_size=4, max_size=4)
)
def test_calculate_sell_from_buy(list_offset: int, buy_quantities: list[float]):
    swaps = [
        {"tkn_sell": "USDT", "tkn_buy": "LRNA"},
        {"tkn_sell": "LRNA", "tkn_buy": "HDX"},
        {"tkn_sell": "HDX", "tkn_buy": "USDT"},
        {"tkn_sell": "USDT", "tkn_buy": "HDX"}
    ]
    swaps = swaps[list_offset:] + swaps[:list_offset]
    omnipool = OmnipoolState(
        tokens={
            "HDX": {"liquidity": mpf(10000000), "LRNA": mpf(1000000)},
            "USDT": {"liquidity": mpf(250000), "LRNA": mpf(2000000)},
        },
        lrna_fee=mpf(1) / 2000,  # 0.0005
        asset_fee=mpf(1) / 400,  # 0.0025
        slip_factor=mpf(1.0)
    )
    omnipool.max_lrna_fee = 0.01
    for i, swap in enumerate(swaps):
        agent = Agent(enforce_holdings=False)
        buy_quantity = mpf(buy_quantities[i])
        tkn_sell = swap["tkn_sell"]
        tkn_buy = swap["tkn_buy"]
        sell_quantity = omnipool.calculate_sell_from_buy(
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            buy_quantity=buy_quantity
        )
        omnipool.copy().swap(
            agent=agent,
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            buy_quantity=buy_quantity
        )

        actual_sell_quantity = -agent.get_holdings(tkn_sell)
        actual_buy_quantity = agent.get_holdings(tkn_buy)
        if sell_quantity != pytest.approx(actual_sell_quantity, rel=1e-40):
            raise AssertionError(f'sell quantity {actual_sell_quantity} != calculated {sell_quantity}')
        if buy_quantity != pytest.approx(actual_buy_quantity, rel=1e-40):
            raise AssertionError(f'buy quantity {buy_quantity} != actual quantity {agent.holdings[tkn_buy]}')

        agent.holdings = {}
        omnipool.swap( # don't copy this time: accumulate trade volume
            agent=agent,
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            sell_quantity=sell_quantity
        )
        actual_buy_quantity = agent.get_holdings(tkn_buy)
        actual_sell_quantity = -agent.get_holdings(tkn_sell)
        if buy_quantity != pytest.approx(actual_buy_quantity, rel=1e-40):
            raise AssertionError(f'expected buy quantity {buy_quantity} != actual quantity {agent.holdings[tkn_buy]}')
        if sell_quantity != pytest.approx(actual_sell_quantity, rel=1e-40):
            raise AssertionError(f'expected sell quantity {sell_quantity} != actual quantity {-agent.holdings[tkn_sell]}')


def test_calculate_sell_from_buy_low_liq_sell_asset():
    tokens = {
        "HDX": {"liquidity": mpf(10000000), "LRNA": mpf(1000000)},
        "USDT": {"liquidity": mpf(1000000), "LRNA": mpf(1000000)},
        "DOT": {"liquidity": mpf(100000), "LRNA": mpf(1000000)},
        "TKN": {"liquidity": mpf(100), "LRNA": mpf(100)}  # spot price of $1, TVL in Omnipool $100
    }
    omnipool = OmnipoolState(
        tokens=tokens,
        lrna_fee=0.0005,
        asset_fee=0.0025,
    )
    omnipool.max_lrna_fee = 0.01
    sell_quantity = omnipool.calculate_sell_from_buy(tkn_sell='TKN', tkn_buy='USDT', buy_quantity=100)
    if sell_quantity != float('inf'):
        raise AssertionError(f'buy_amt {sell_quantity} != inf')


@given(
    list_offset = st.integers(min_value=0, max_value=3),
    sell_quantities = st.lists(st.floats(min_value=1, max_value=1000), min_size=4, max_size=4)
)
def test_calculate_buy_from_sell(list_offset: int, sell_quantities: list[float]):
    swaps = [
        {"tkn_sell": "USDT", "tkn_buy": "LRNA"},
        {"tkn_sell": "LRNA", "tkn_buy": "HDX"},
        {"tkn_sell": "HDX", "tkn_buy": "USDT"},
        {"tkn_sell": "USDT", "tkn_buy": "HDX"}
    ]
    swaps = swaps[list_offset:] + swaps[:list_offset]
    omnipool = OmnipoolState(
        tokens={
            "HDX": {"liquidity": mpf(10000000), "LRNA": mpf(1000000)},
            "USDT": {"liquidity": mpf(250000), "LRNA": mpf(2000000)},
        },
        lrna_fee=mpf(1) / 2000,  # 0.0005
        asset_fee=mpf(1) / 400,  # 0.0025
        slip_factor=mpf(1.0)
    )
    omnipool.max_lrna_fee = 0.01
    for i, swap in enumerate(swaps):
        agent = Agent(enforce_holdings=False)
        sell_quantity = mpf(sell_quantities[i])
        tkn_sell = swap["tkn_sell"]
        tkn_buy = swap["tkn_buy"]
        buy_quantity = omnipool.calculate_buy_from_sell(
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            sell_quantity=sell_quantity
        )
        omnipool.copy().swap(
            agent=agent,
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            sell_quantity=sell_quantity
        )

        if buy_quantity != pytest.approx(agent.get_holdings(tkn_buy), rel=1e-40):
            raise AssertionError(f'expected buy quantity {buy_quantity} != actual quantity {agent.holdings[tkn_buy]}')
        if sell_quantity != pytest.approx(-agent.get_holdings(tkn_sell), rel=1e-40):
            raise AssertionError(f'expected sell quantity {sell_quantity} != actual quantity {-agent.holdings[tkn_sell]}')

        agent.holdings = {}
        omnipool.swap(
            agent=agent,
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            buy_quantity=buy_quantity
        )
        actual_buy_quantity = agent.get_holdings(tkn_buy)
        actual_sell_quantity = -agent.get_holdings(tkn_sell)
        if buy_quantity != pytest.approx(actual_buy_quantity, rel=1e-40):
            raise AssertionError(f'expected buy quantity {buy_quantity} != actual quantity {agent.holdings[tkn_buy]}')
        if sell_quantity != pytest.approx(actual_sell_quantity, rel=1e-40):
            raise AssertionError(f'expected sell quantity {sell_quantity} != actual quantity {-agent.holdings[tkn_sell]}')

@given(
    tkn_liquidity = st.floats(min_value=100000, max_value=10000000),
    tkn_lrna = st.floats(min_value=10000, max_value=1000000)
)
def test_calculate_lrna_swaps(tkn_liquidity, tkn_lrna):
    omnipool = OmnipoolState(
        tokens={
            "HDX": {"liquidity": mpf(10000000), "LRNA": mpf(1000000)},
            "USDT": {"liquidity": mpf(1000000), "LRNA": mpf(1000000)},
            "TKN": {"liquidity": mpf(tkn_liquidity), "LRNA": mpf(tkn_liquidity)}
        },
        lrna_fee=0.0005,
        asset_fee=0.0025,
        slip_factor=1
    )
    agent = Agent(enforce_holdings=False)
    for swap in [{"tkn_sell": "LRNA", "tkn_buy": "TKN"}, {"tkn_sell": "TKN", "tkn_buy": "LRNA"}]:
        tkn_sell = swap["tkn_sell"]
        tkn_buy = swap["tkn_buy"]
        sell_quantity = 1
        buy_quantity = omnipool.calculate_buy_from_sell(
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            sell_quantity=sell_quantity
        )
        agent.holdings[tkn_buy] = 0
        omnipool.swap(tkn_sell=tkn_sell, tkn_buy=tkn_buy, sell_quantity=sell_quantity, agent=agent)
        actual_buy_quantity = agent.get_holdings(tkn_buy)
        if buy_quantity != pytest.approx(actual_buy_quantity, rel=1e-40):
            raise AssertionError(f'buy_quantity {buy_quantity} != actual_buy_quantity {actual_buy_quantity}')
        sell_quantity = omnipool.calculate_sell_from_buy(
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            buy_quantity=buy_quantity
        )
        agent.holdings[tkn_sell] = 0
        omnipool.swap(tkn_sell=tkn_sell, tkn_buy=tkn_buy, buy_quantity=buy_quantity, agent=agent)
        actual_sell_quantity = -agent.get_holdings(tkn_sell)
        if sell_quantity != pytest.approx(actual_sell_quantity, rel=1e-40):
            raise AssertionError(f'sell_quantity {sell_quantity} != actual_sell_quantity {actual_sell_quantity}')


@given(
    hdx_lrna=st.floats(min_value=100000000, max_value=1000000000),
    usd_lrna=st.floats(min_value=100000000, max_value=1000000000),
    hdx_asset_fee=st.floats(min_value=0, max_value=0.1),
    hdx_lrna_fee=st.floats(min_value=0, max_value=0.1),
    usd_asset_fee=st.floats(min_value=0, max_value=0.1),
    usd_lrna_fee=st.floats(min_value=0, max_value=0.1)
)
def test_buy_sell_spot(
        hdx_lrna: float, usd_lrna: float, hdx_asset_fee: float, hdx_lrna_fee: float, usd_asset_fee: float,
        usd_lrna_fee: float
):
    tokens = {
        'HDX': {'liquidity': mpf(1000000000), 'LRNA': hdx_lrna},
        'USD': {'liquidity': mpf(1000000000), 'LRNA': usd_lrna},
    }
    initial_state = oamm.OmnipoolState(
        tokens=tokens,
        lrna_fee={'HDX': hdx_lrna_fee, 'USD': usd_lrna_fee},
        asset_fee={'HDX': hdx_asset_fee, 'USD': usd_asset_fee},
    )
    agent = Agent(holdings={tkn: mpf(1000) for tkn in initial_state.asset_list})
    test_state, test_agent = initial_state.copy(), agent.copy()
    buy_quantity = 0.001
    hdx_per_usd = initial_state.sell_spot(tkn_sell='USD', tkn_buy='HDX')
    usd_per_hdx = initial_state.buy_spot(tkn_buy='HDX', tkn_sell='USD')
    test_state.swap(
        agent=test_agent,
        tkn_sell='USD',
        tkn_buy='HDX',
        buy_quantity=buy_quantity
    )
    actual_sell_quantity = test_agent.initial_holdings['USD'] - test_agent.holdings['USD']
    actual_buy_quantity = test_agent.holdings['HDX'] - test_agent.initial_holdings['HDX']
    ex_price_hdx = actual_sell_quantity / actual_buy_quantity
    ex_price_usd = actual_buy_quantity / actual_sell_quantity
    if usd_per_hdx != pytest.approx(ex_price_hdx, rel=1e-08):
        raise AssertionError(f'sell_spot_hdx {usd_per_hdx} != ex_price_hdx {ex_price_hdx}')
    if hdx_per_usd != pytest.approx(ex_price_usd, rel=1e-08):
        raise AssertionError(f'sell_spot_usd {hdx_per_usd} != ex_price_usd {ex_price_usd}')


def test_lrna_price_lrna():
    """Test that we can call lrna_price with input LRNA and get 1"""
    initial_state = oamm.OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(10000000000), 'LRNA': mpf(5000000)},
            'USD': {'liquidity': mpf(1000000000), 'LRNA': mpf(3333333333)},
            'DOT': {'liquidity': mpf(100000000), 'LRNA': mpf(1111111111)},
        },
        lrna_fee=0.0005,
        asset_fee=0.0025,
        preferred_stablecoin='USD'
    )

    lrna_price = initial_state.lrna_price('LRNA')
    if lrna_price != pytest.approx(1, rel=1e-15):
        raise AssertionError(f'lrna_price {lrna_price} != 1')


@given(st.lists(asset_quantity_strategy, min_size=6, max_size=6),
       st.floats(min_value=0.0001, max_value=0.1, exclude_min=True))
def test_price_lrna(amts: list, asset_fee: float):
    '''Tests the price function with LRNA as each input'''

    hdx_amt, usd_amt, dot_amt = mpf(amts[0]), mpf(amts[1]), mpf(amts[2])
    hdx_lrna, usd_lrna, dot_lrna = mpf(amts[3]), mpf(amts[4]), mpf(amts[5])

    initial_state = oamm.OmnipoolState(
        tokens={
            'HDX': {'liquidity': hdx_amt, 'LRNA': hdx_lrna},
            'USD': {'liquidity': usd_amt, 'LRNA': usd_lrna},
            'DOT': {'liquidity': dot_amt, 'LRNA': dot_lrna},
        },
        lrna_fee=0.0005,
        asset_fee=asset_fee,
        preferred_stablecoin='USD'
    )

    lrna_price = initial_state.price('LRNA', 'USD')
    usd_price = initial_state.price('USD', 'LRNA')
    if lrna_price != pytest.approx(usd_amt / usd_lrna, rel=1e-15):
        raise AssertionError(f'lrna_price {lrna_price} != {usd_amt / usd_lrna}')
    if usd_price != pytest.approx(usd_lrna / usd_amt, rel=1e-15):
        raise AssertionError(f'lrna_price {usd_price} != {usd_lrna / usd_amt}')


@given(st.lists(asset_quantity_strategy, min_size=6, max_size=6),
       st.floats(min_value=0.0001, max_value=0.1, exclude_min=True))
def test_sell_spot_lrna(amts: list, asset_fee: float):
    '''Tests sell_spot with LRNA as the sell_tkn'''

    hdx_amt, usd_amt, dot_amt = mpf(amts[0]), mpf(amts[1]), mpf(amts[2])
    hdx_lrna, usd_lrna, dot_lrna = mpf(amts[3]), mpf(amts[4]), mpf(amts[5])

    initial_state = oamm.OmnipoolState(
        tokens={
            'HDX': {'liquidity': hdx_amt, 'LRNA': hdx_lrna},
            'USD': {'liquidity': usd_amt, 'LRNA': usd_lrna},
            'DOT': {'liquidity': dot_amt, 'LRNA': dot_lrna},
        },
        lrna_fee=0.0005,
        asset_fee=asset_fee,
        preferred_stablecoin='USD'
    )

    price = initial_state.sell_spot('LRNA', 'USD')
    if price != pytest.approx(usd_amt / usd_lrna * (1 - asset_fee), rel=1e-15):
        raise AssertionError(f'price {price} is incorrect')


@given(st.lists(asset_quantity_strategy, min_size=6, max_size=6),
       st.floats(min_value=0.0001, max_value=0.1, exclude_min=True))
def test_buy_spot_lrna(amts: list, asset_fee: float):
    '''Tests buy_spot with LRNA as the sell_tkn'''

    hdx_amt, usd_amt, dot_amt = mpf(amts[0]), mpf(amts[1]), mpf(amts[2])
    hdx_lrna, usd_lrna, dot_lrna = mpf(amts[3]), mpf(amts[4]), mpf(amts[5])

    initial_state = oamm.OmnipoolState(
        tokens={
            'HDX': {'liquidity': hdx_amt, 'LRNA': hdx_lrna},
            'USD': {'liquidity': usd_amt, 'LRNA': usd_lrna},
            'DOT': {'liquidity': dot_amt, 'LRNA': dot_lrna},
        },
        lrna_fee=0.0005,
        asset_fee=asset_fee,
        preferred_stablecoin='USD'
    )

    price = initial_state.buy_spot('USD', 'LRNA')
    exp_price = usd_lrna / usd_amt
    if price != pytest.approx(exp_price / (1 - asset_fee), rel=1e-15):
        raise AssertionError(f'price {price} is incorrect')


def test_value_assets_without_equivalency_map():
    initial_state = oamm.OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(1000000000), 'LRNA': mpf(100000000)},
            'USD': {'liquidity': mpf(1000000000), 'LRNA': mpf(1000000000)},
            'DOT': {'liquidity': mpf(1000000000), 'LRNA': mpf(10000000000)},
        },
        lrna_fee=0.0025,
        asset_fee=0.0005,
        preferred_stablecoin='USD'
    )

    assets = {'HDX': mpf(1000), 'USD': mpf(2000), 'DOT': mpf(3000), 'LRNA': mpf(4000)}
    val = initial_state.value_assets(assets, numeraire='USD')
    if val != 100 + 2000 + 30000 + 4000:
        raise AssertionError(f'val {val} is incorrect')


def test_no_preferred_stablecoin():
    """Tests Omnipool initialization, as well as value_assets and usd_price, with no preferred_stablecoin"""
    initial_state = oamm.OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(1000000000), 'LRNA': mpf(100000000)},
            'USD': {'liquidity': mpf(1000000000), 'LRNA': mpf(1000000000)},
            'DOT': {'liquidity': mpf(1000000000), 'LRNA': mpf(10000000000)},
        },
        lrna_fee=0.0025,
        asset_fee=0.0005,
    )

    # assets = {'HDX': mpf(1000), 'USD': mpf(2000), 'DOT': mpf(3000), 'LRNA': mpf(4000)}
    assets = {'HDX': mpf(1000), 'USD': mpf(2000), 'DOT': mpf(3000)}
    val = initial_state.value_assets(assets, numeraire='USD')
    # if val != 100 + 2000 + 30000 + 4000:
    if val != 100 + 2000 + 30000:
        raise AssertionError(f'val {val} is incorrect')

    usd_p = initial_state.usd_price('HDX', usd_asset='USD')
    if usd_p != pytest.approx(0.1, rel=1e-15):
        raise AssertionError(f'usd_p {usd_p} is incorrect')

    initial_state.__repr__()


def test_fee_application():
    initial_state = OmnipoolState(
        tokens={'HDX': {'liquidity': 1000000, 'LRNA': 1000}, 'USD': {'liquidity': 3000, 'LRNA': 150}},
        lrna_fee={'HDX': 0.0005, 'USD': 0.001},
        asset_fee={'HDX': 0.007, 'USD': 0.0025}
    )
    initial_agent = Agent(
        holdings={'HDX': 1000000}
    )
    sell_quantity = 1
    sell_agent = initial_agent.copy()
    initial_state.copy().swap(
        agent=sell_agent,
        tkn_sell='HDX',
        tkn_buy='USD',
        sell_quantity=sell_quantity
    )
    buy_quantity = sell_agent.holdings['USD']
    sell_lrna_agent = initial_agent.copy()
    sell_lrna_state = initial_state.copy()
    sell_lrna_state.asset_fee = 0
    sell_lrna_state.lrna_fee = 0
    sell_lrna_state.swap(
        agent=sell_lrna_agent,
        sell_quantity=sell_quantity,
        tkn_sell='HDX',
        tkn_buy='LRNA'
    )
    lrna_fee = sell_lrna_agent.holdings['LRNA'] * initial_state.lrna_fee('HDX')
    sell_lrna_state.lrna['HDX'] += lrna_fee
    sell_lrna_agent.holdings['LRNA'] -= lrna_fee
    sell_lrna_state.swap(
        agent=sell_lrna_agent,
        sell_quantity=sell_lrna_agent.holdings['LRNA'],
        tkn_buy='USD',
        tkn_sell='LRNA'
    )
    asset_fee = sell_lrna_agent.holdings['USD'] * initial_state.asset_fee('USD')
    sell_lrna_state.liquidity['USD'] += asset_fee
    sell_lrna_agent.holdings['USD'] -= asset_fee
    buy_quantity_2 = sell_lrna_agent.holdings['USD']
    if buy_quantity != pytest.approx(buy_quantity_2, rel=1e-12):
        raise AssertionError("Direct swap was not equivalent to LRNA swap with fees applied manually.")


@given(st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=10))
def test_lrna_swap_equivalency(lrna_burn_rate, min_fee_fraction):
    initial_state = OmnipoolState(
        tokens={'HDX': {'liquidity': mpf(1000000), 'LRNA': mpf(1000)}, 'USD': {'liquidity': mpf(3000), 'LRNA': mpf(150)}},
        lrna_fee=DynamicFee(
            current={'HDX': mpf(1) / 2000, 'USD': mpf(1) / 1000},
            minimum=mpf(1) / 2000 / min_fee_fraction,
            maximum=mpf(1) / 1000 / min_fee_fraction
        ),
        asset_fee=DynamicFee(
            current={'HDX': mpf(1) / 1000 * 7, 'USD': mpf(1) / 400}
        ),
        lrna_fee_burn=mpf(1) / lrna_burn_rate / min_fee_fraction / 2000,
        slip_factor=1.0
    )

    agent = Agent(enforce_holdings=False)
    sell_quantity = 1000

    sell_agent = agent.copy()
    sell_state = initial_state.copy().swap(
        sell_quantity=sell_quantity,
        agent=sell_agent,
        tkn_sell='HDX',
        tkn_buy='LRNA'
    )
    mid_sell_agent = sell_agent.copy()
    sell_state.swap(
        sell_quantity=sell_agent.holdings['LRNA'],
        agent=sell_agent,
        tkn_buy='USD',
        tkn_sell='LRNA'
    )
    direct_sell_agent = agent.copy()
    direct_sell_state = initial_state.copy().swap(
        agent=direct_sell_agent,
        tkn_sell='HDX',
        tkn_buy='USD',
        sell_quantity=sell_quantity
    )
    buy_quantity = direct_sell_agent.holdings['USD']
    if sell_state.liquidity['USD'] != pytest.approx(direct_sell_state.liquidity['USD'], rel=1e-40):
        raise AssertionError("Direct sell was not equivalent to two LRNA swaps (USD liquidity).")
    elif sell_state.lrna['USD'] != pytest.approx(direct_sell_state.lrna['USD'], rel=1e-40):
        raise AssertionError("Direct sell was not equivalent to two LRNA swaps (USD LRNA).")
    elif sell_state.lrna['HDX'] != pytest.approx(direct_sell_state.lrna['HDX'], rel=1e-40):
        raise AssertionError("Direct sell was not equivalent to two LRNA swaps (HDX LRNA).")
    elif sell_state.liquidity['HDX'] != pytest.approx(direct_sell_state.liquidity['HDX'], rel=1e-40):
        raise AssertionError("Direct sell was not equivalent to two LRNA swaps (HDX liquidity).")
    elif sell_agent.holdings['USD'] != pytest.approx(direct_sell_agent.holdings['USD'], rel=1e-40):
        raise AssertionError("Direct sell was not equivalent to two LRNA swaps (agent USD).")
    elif sell_agent.holdings['HDX'] != pytest.approx(direct_sell_agent.holdings['HDX'], rel=1e-40):
        raise AssertionError("Direct sell was not equivalent to two LRNA swaps (agent HDX).")
    elif sell_agent.get_holdings('LRNA') != pytest.approx(direct_sell_agent.get_holdings('LRNA'), rel=1e-40):
        raise AssertionError("Direct sell was not equivalent to two LRNA swaps (agent LRNA).")
    elif sell_state.lrna_fee_destination.holdings['LRNA'] != pytest.approx(
            direct_sell_state.lrna_fee_destination.holdings['LRNA'], rel=1e-40):
        raise AssertionError("Direct sell was not equivalent to two LRNA swaps (fee destination LRNA).")
    elif direct_sell_state.fail:
        raise AssertionError("Sell failed.")
    else:
        er = 'no problem'

    buy_agent = agent.copy()
    buy_state = initial_state.copy().swap(
        buy_quantity=mid_sell_agent.holdings['LRNA'],
        agent=buy_agent,
        tkn_sell='HDX',
        tkn_buy='LRNA'
    ).swap(
        buy_quantity=buy_quantity,
        agent=buy_agent,
        tkn_buy='USD',
        tkn_sell='LRNA'
    )
    buy_quantity = buy_agent.holdings['USD']
    direct_buy_agent = agent.copy()
    direct_buy_state = initial_state.copy().swap(
        agent=direct_buy_agent,
        tkn_sell='HDX',
        tkn_buy='USD',
        buy_quantity=buy_quantity
    )
    if buy_state.liquidity['USD'] != pytest.approx(direct_buy_state.liquidity['USD'], rel=1e-12):
        raise AssertionError("Direct buy was not equivalent to two LRNA swaps (USD liquidity).")
    elif buy_state.lrna['USD'] != pytest.approx(direct_buy_state.lrna['USD'], rel=1e-12):
        raise AssertionError("Direct buy was not equivalent to two LRNA swaps (USD LRNA).")
    elif buy_state.liquidity['HDX'] != pytest.approx(direct_buy_state.liquidity['HDX'], rel=1e-12):
        raise AssertionError("Direct buy was not equivalent to two LRNA swaps (HDX liquidity).")
    elif buy_state.lrna['HDX'] != pytest.approx(direct_buy_state.lrna['HDX'], rel=1e-12):
        raise AssertionError("Direct buy was not equivalent to two LRNA swaps (HDX lrna).")
    elif buy_agent.holdings['USD'] != pytest.approx(direct_buy_agent.holdings['USD'], rel=1e-12):
        raise AssertionError("Direct buy was not equivalent to two LRNA swaps (agent USD).")
    elif buy_agent.holdings['HDX'] != pytest.approx(direct_buy_agent.holdings['HDX'], rel=1e-12):
        raise AssertionError("Direct buy was not equivalent to two LRNA swaps (agent HDX).")
    elif buy_agent.holdings['LRNA'] != pytest.approx(direct_buy_agent.holdings['LRNA'], rel=1e-12):
        raise AssertionError("Direct buy was not equivalent to two LRNA swaps (agent LRNA).")
    elif buy_state.lrna_fee_destination.holdings['LRNA'] != pytest.approx(
            direct_buy_state.lrna_fee_destination.holdings['LRNA'], rel=1e-12):
        raise AssertionError("Direct buy was not equivalent to two LRNA swaps (fee destination LRNA).")
    elif direct_buy_state.fail:
        raise AssertionError("Buy failed.")
    else:
        er = 'no problem'


def test_cash_out_omnipool_exact():
    liquidity = {'HDX': mpf(10000000), 'USD': mpf(1000000), 'DOT': mpf(100000)}
    lrna = {'HDX': mpf(1000000), 'USD': mpf(1000000), 'DOT': mpf(1000000)}
    initial_state = oamm.OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        }
    )
    tkn = 'DOT'

    p = initial_state.price(tkn, 'LRNA')
    s = initial_state.shares[tkn] / 10
    prices = {tkn: initial_state.price(tkn, 'USD') for tkn in initial_state.liquidity}
    expected_r = initial_state.liquidity[tkn] / 10 * (1 - initial_state.min_withdrawal_fee)
    expected_cash = expected_r * prices[tkn]
    position = OmnipoolLiquidityPosition(tkn, p, s, 0, initial_state.unique_id)
    init_agent = Agent(nfts={'position': position})
    cash = initial_state.cash_out(init_agent, prices)
    if cash != pytest.approx(expected_cash, rel=1e-20):
        raise AssertionError(f'Removed liquidity should be equal to initial liquidity minus final liquidity.')

    p = initial_state.price(tkn, 'LRNA') / 2
    s = initial_state.shares[tkn] / 10
    position = OmnipoolLiquidityPosition(tkn, p, s, 0, initial_state.unique_id)
    init_agent = Agent(nfts={'position': position})
    cash = initial_state.cash_out(init_agent, prices)

    expected_agent_dq_pct = mpf(1) / 30 * (1 - initial_state.min_withdrawal_fee)
    expected_agent_dq = expected_agent_dq_pct * initial_state.lrna[tkn]

    expected_dr_pct = mpf(1) / 10 * (1 - initial_state.min_withdrawal_fee)
    expected_dr = expected_dr_pct * initial_state.liquidity[tkn]
    expected_min_cash = expected_dr * prices[tkn]
    expected_max_cash = expected_min_cash + expected_agent_dq + initial_state.price('LRNA', 'USD')
    expected_cash_out_lrna = 32954

    if expected_agent_dq <= 0:
        raise AssertionError(f'LRNA change incorrect')
    if cash <= expected_min_cash:
        raise AssertionError(f'Cash out should be at least the minimum amount')
    if cash >= expected_max_cash:
        raise AssertionError(f'Cash out should be at most the maximum amount')
    if cash - expected_min_cash != pytest.approx(expected_cash_out_lrna, rel=1e-4):
        raise AssertionError(f'cash incorrect')

    p = initial_state.price(tkn, 'LRNA') * 2
    s = initial_state.shares[tkn] / 10
    position = OmnipoolLiquidityPosition(tkn, p, s, 0, initial_state.unique_id)
    init_agent = Agent(nfts={'position': position})
    cash = initial_state.cash_out(init_agent, prices)

    expected_dr_pct = mpf(2) / 30 * (1 - initial_state.min_withdrawal_fee)
    expected_dr = expected_dr_pct * initial_state.liquidity[tkn]
    expected_cash = expected_dr * prices[tkn]

    if cash != pytest.approx(expected_cash, rel=1e-20):
        raise AssertionError(f'cash incorrect')


@given(st.floats(min_value=10.1, max_value=100))
def test_cash_out_nft_position(price1: float):
    liquidity = {'HDX': mpf(10000000), 'USD': mpf(1000000), 'DOT': mpf(100000)}
    lrna = {'HDX': mpf(1000000), 'USD': mpf(1000000), 'DOT': mpf(1000000)}
    initial_state = oamm.OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        withdrawal_fee=False
    )
    dot_spot_price = initial_state.price('DOT', 'USD')
    tkn = 'DOT'
    amt1 = initial_state.shares[tkn] / 5
    delta_r = initial_state.liquidity[tkn] / 5
    nft = OmnipoolLiquidityPosition(tkn, price1, amt1, delta_r, initial_state.unique_id)
    agent = Agent(holdings={}, nfts={'pos1': nft})
    cash_out = initial_state.cash_out(agent, {'DOT': dot_spot_price})

    state = initial_state.copy()
    state.remove_liquidity(agent, tkn_remove=tkn)
    dot_value = agent.holdings['DOT'] * dot_spot_price
    assert cash_out == pytest.approx(dot_value, rel=1e-20)


@given(st.floats(min_value=10.1, max_value=100),
       st.floats(min_value=10.1, max_value=100),
       st.floats(min_value=0.1, max_value=0.9))
def test_cash_out_nft_position_with_holdings(price1: float, price2: float, r: float):
    liquidity = {'HDX': mpf(10000000), 'USD': mpf(1000000), 'DOT': mpf(100000)}
    lrna = {'HDX': mpf(1000000), 'USD': mpf(1000000), 'DOT': mpf(1000000)}
    initial_state = oamm.OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        withdrawal_fee=False
    )
    dot_spot_price = initial_state.price('DOT', 'USD')
    tkn = 'DOT'
    amt1 = r * initial_state.shares[tkn] / 5
    amt2 = initial_state.shares[tkn] / 5 - amt1
    holdings1 = {(initial_state.unique_id, tkn): amt1}
    prices1 = {(initial_state.unique_id, tkn): price1}
    nft = OmnipoolLiquidityPosition(tkn, price2, amt2, 0, initial_state.unique_id)
    agent = Agent(holdings=holdings1, share_prices=prices1, nfts={'pos1': nft})
    cash_out = initial_state.cash_out(agent, {'DOT': dot_spot_price})

    state = initial_state.copy()
    state.remove_liquidity(agent, tkn_remove=tkn)
    dot_value = agent.holdings['DOT'] * dot_spot_price
    assert cash_out == pytest.approx(dot_value, rel=1e-20)


@given(st.floats(min_value=0.1, max_value=9.9),
       st.floats(min_value=0.1, max_value=9.9),
       st.floats(min_value=0.1, max_value=0.9))
def test_cash_out_multiple_positions_works_with_lrna(price1: float, price2: float, r: float):
    liquidity = {'HDX': mpf(10000000), 'USD': mpf(1000000), 'DOT': mpf(100000)}
    lrna = {'HDX': mpf(1000000), 'USD': mpf(1000000), 'DOT': mpf(1000000)}
    initial_state = oamm.OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        withdrawal_fee=False
    )
    tkn = 'DOT'
    amt1 = r * initial_state.shares[tkn] / 5
    amt2 = initial_state.shares[tkn] / 5 - amt1
    holdings1 = {(initial_state.unique_id, tkn): amt1}
    prices1 = {(initial_state.unique_id, tkn): price1}
    nft = OmnipoolLiquidityPosition(tkn, price2, amt2, 0, initial_state.unique_id)
    agent = Agent(holdings=holdings1, share_prices=prices1, nfts={'pos1': nft})
    spot_prices = {tkn: initial_state.price(tkn, 'USD') for tkn in initial_state.liquidity}
    cash_out = initial_state.cash_out(agent, spot_prices)

    state = initial_state.copy()
    state.remove_liquidity(agent, tkn_remove=tkn)
    dot_value = agent.holdings['DOT'] * spot_prices['DOT']
    lrna_value = agent.holdings['LRNA'] * initial_state.price('LRNA', 'USD')
    assert dot_value < cash_out < dot_value + lrna_value  # cash_out will be less than dot + lrna due to slippage


@given(st.lists(st.floats(min_value=-100000, max_value=100000), min_size=3, max_size=3))
def test_cash_out_multiple_positions(trade_sizes: list[float]):
    liquidity = {'HDX': mpf(1000000), 'USD': mpf(1000000), 'DOT': mpf(100000)}
    lrna = {'HDX': mpf(1000000), 'USD': mpf(1000000), 'DOT': mpf(1000000)}
    initial_state = oamm.OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        withdrawal_fee=False
    )

    lp_quantity = mpf(10000)
    agent1 = Agent(holdings={'DOT': lp_quantity * len(trade_sizes)})
    agent2 = Agent(holdings={'DOT': mpf(10000000), 'HDX': mpf(10000000)})
    for i, trade in enumerate(trade_sizes):
        initial_state.add_liquidity(agent1, tkn_add='DOT', quantity=lp_quantity, nft_id=str(i))
        if trade > 0:
            initial_state.swap(agent2, tkn_buy='HDX', tkn_sell='DOT', sell_quantity=mpf(trade))
        elif trade < 0:
            initial_state.swap(agent2, tkn_buy='DOT', tkn_sell='HDX', sell_quantity=-mpf(trade))

    spot_prices = {tkn: initial_state.price(tkn, 'USD') for tkn in initial_state.asset_list}
    spot_prices['LRNA'] = initial_state.usd_price('LRNA')

    cash_out_value = initial_state.cash_out(agent1, spot_prices)
    cash_out_state = initial_state.copy()
    cash_out_agent = agent1.copy()
    cash_out_state.remove_liquidity(cash_out_agent, tkn_remove='DOT')
    reference_value = oamm.value_assets(spot_prices, cash_out_agent.holdings)
    if cash_out_value != pytest.approx(reference_value, 1e-20):
        raise AssertionError("Cash out not computed correctly.")


@given(
    lrna_fee=st.floats(min_value=0.0005, max_value=0.001),
    burn_rate=st.floats(min_value=0, max_value=1)
)
def test_lrna_fee_burn(lrna_fee, burn_rate):
    initial_state = OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(1000000), 'LRNA': mpf(100000)},
            'USD': {'liquidity': mpf(1000000), 'LRNA': mpf(100000)},
            'DOT': {'liquidity': mpf(100000), 'LRNA': mpf(100000)}
        },
        lrna_fee=lrna_fee,
        asset_fee=0.0025,
        lrna_fee_burn=burn_rate,
        lrna_mint_pct=0
    )
    tkn_sell = 'USD'
    tkn_buy = 'DOT'
    sell_quantity = mpf(10)
    initial_agent = Agent(holdings={tkn_sell: sell_quantity * 2})
    sell_tkn_state, sell_tkn_agent = oamm.simulate_swap(
        old_state=initial_state,
        old_agent=initial_agent,
        tkn_sell=tkn_sell,
        tkn_buy='LRNA',
        sell_quantity=sell_quantity
    )
    lrna_received_1 = sell_tkn_agent.holdings['LRNA']
    lrna_deposited_1 = sell_tkn_state.lrna_fee_destination.holdings['LRNA']
    lrna_burned_1 = (
            sum(initial_state.lrna.values())
            - sum(sell_tkn_state.lrna.values())
            - sell_tkn_agent.holdings['LRNA']
            - lrna_deposited_1
    )
    lrna_paid_out_1 = initial_state.lrna[tkn_sell] - sell_tkn_state.lrna[tkn_sell]
    lrna_fee_total_1 = lrna_paid_out_1 * lrna_fee
    if lrna_received_1 + lrna_fee_total_1 != pytest.approx(lrna_paid_out_1, rel=1e-20):
        raise AssertionError(f'LRNA fee not calculated correctly.')
    if lrna_burned_1 / lrna_fee_total_1 != pytest.approx(burn_rate, rel=1e-20):
        raise AssertionError(f'LRNA burn rate not calculated correctly.')

    buy_lrna_state, buy_lrna_agent = oamm.simulate_swap(
        old_state=initial_state,
        old_agent=initial_agent,
        tkn_sell=tkn_sell,
        tkn_buy='LRNA',
        buy_quantity=lrna_received_1
    )
    if buy_lrna_state.fail:
        raise AssertionError('buy LRNA swap failed.')
    lrna_received_2 = buy_lrna_agent.holdings['LRNA']
    lrna_deposited_2 = buy_lrna_state.lrna_fee_destination.holdings['LRNA']
    lrna_burned_2 = (
            sum(initial_state.lrna.values())
            - sum(buy_lrna_state.lrna.values())
            - buy_lrna_agent.holdings['LRNA']
            - lrna_deposited_2
    )
    lrna_paid_out_2 = initial_state.lrna[tkn_sell] - buy_lrna_state.lrna[tkn_sell]
    lrna_fee_total_2 = lrna_paid_out_2 * lrna_fee
    if lrna_received_2 + lrna_fee_total_2 != pytest.approx(lrna_paid_out_2, rel=1e-20):
        raise AssertionError(f'LRNA fee not calculated correctly.')
    if lrna_burned_2 / lrna_fee_total_2 != pytest.approx(burn_rate, rel=1e-20):
        raise AssertionError(f'LRNA burn rate not calculated correctly.')

    buy_quantity = initial_state.calculate_buy_from_sell(
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
        sell_quantity=sell_quantity
    )
    buy_state, buy_agent = oamm.simulate_swap(
        old_state=initial_state,
        old_agent=initial_agent,
        tkn_sell=tkn_sell,
        tkn_buy=tkn_buy,
        buy_quantity=buy_quantity
    )
    lrna_deposited_3 = buy_state.lrna_fee_destination.holdings['LRNA']
    lrna_burned_3 = (
            sum(initial_state.lrna.values())
            - sum(buy_state.lrna.values())
            - lrna_deposited_3
    )
    lrna_paid_out_3 = initial_state.lrna[tkn_sell] - buy_state.lrna[tkn_sell]
    lrna_fee_total_3 = lrna_paid_out_3 * lrna_fee
    if buy_state.fail:
        raise AssertionError('buy swap failed.')
    if lrna_burned_3 / lrna_fee_total_3 != pytest.approx(burn_rate, rel=1e-20):
        raise AssertionError(f'LRNA burn rate not calculated correctly.')
    if lrna_received_1 + lrna_fee_total_3 != pytest.approx(lrna_paid_out_3, rel=1e-20):
        raise AssertionError(f'LRNA fee not calculated correctly.')


def test_price_after_trade():
    setup1 = OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(2000000000000000), 'LRNA': mpf(2000000000000000)},
            'USD': {'liquidity': mpf(2000000000000000), 'LRNA': mpf(2000000000000000)}
        },
        lrna_mint_pct=1,
        asset_fee=0,
        lrna_fee=0.1
    )
    setup2 = setup1.copy()
    setup2.asset_fee = 0.1
    setup3 = setup2.copy()
    setup3.lrna_mint_pct = 0
    setup4 = setup2.copy()

    agent = Agent(
        holdings={'USD': mpf(58_823_529_411_766)}
    )
    print()
    print(setup1.usd_price('HDX'))
    print(setup2.usd_price('HDX'))
    setup1.swap(
        agent=agent.copy(),
        tkn_buy='HDX',
        tkn_sell='USD',
        sell_quantity=agent.holdings['USD']
    )
    setup2.swap(
        agent=agent.copy(),
        tkn_buy='HDX',
        tkn_sell='USD',
        sell_quantity=agent.holdings['USD']
    )
    sell_agent = agent.copy()
    setup3.swap(
        agent=sell_agent,
        tkn_buy='HDX',
        tkn_sell='USD',
        sell_quantity=agent.holdings['USD']
    )
    buy_agent = agent.copy()
    setup4.swap(
        agent=buy_agent,
        tkn_buy='HDX',
        tkn_sell='USD',
        buy_quantity=sell_agent.holdings['HDX']
    )
    print(setup1.liquidity['HDX'] / setup1.lrna['HDX'])
    print(setup2.liquidity['HDX'] / setup2.lrna['HDX'])
    print(setup3.liquidity['HDX'] / setup3.lrna['HDX'])
    print(setup4.liquidity['HDX'] / setup4.lrna['HDX'])
    lrna_minted = setup2.lrna['HDX'] - setup3.lrna['HDX']
    print(lrna_minted)
    print(f"agent4 sell quantity: {buy_agent.initial_holdings['USD'] - buy_agent.holdings['USD']}")


@given(
    sell_amt=st.floats(min_value=1, max_value=10000),
    asset_fee = st.floats(min_value=0.00001, max_value=0.1)
)
@settings(print_blob=True)
def test_fee_against_invariant_spec(sell_amt, asset_fee):
    fA = asset_fee
    omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            'USD': {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)}
        },
        lrna_mint_pct=1,
        asset_fee=fA,
        lrna_fee=0.0
    )

    q, r = omnipool.lrna['USD'], omnipool.liquidity['USD']

    delta_q = -sell_amt
    agent = Agent(holdings={'LRNA': -delta_q})

    F = 0  # python implementation doesn't remove any fee

    omnipool.swap(agent, 'USD', 'LRNA', sell_quantity=-delta_q)

    q_plus, r_plus = omnipool.lrna['USD'], omnipool.liquidity['USD']
    rho = -delta_q / q
    lhs = q_plus * r_plus - q * r
    rhs = delta_q * (r / (1 + rho) * (1 - fA) + F / rho + r_plus * (-1 - fA * (1 + rho)))
    assert lhs == pytest.approx(rhs, rel=1e-10)

    lhs2 = (q_plus * r_plus + (F - r) * q) / delta_q
    rhs2 = r * q * (1 - fA) / (q - delta_q) - (1 + fA) * r_plus + fA * r_plus * delta_q / q
    assert lhs2 == pytest.approx(rhs2, rel=1e-10)


def test_trade_to_price():
    omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            'USD': {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)}
        },
        asset_fee=0.0025,
        lrna_fee=0.0005
    )
    agent = Agent(enforce_holdings=False)

    for expected_price in [0.9, 1.0, 1.1]:
        omnipool.trade_to_price(agent, tkn='USD', other_tkn='HDX', target_price=expected_price)
        price = omnipool.lrna_price('USD')
        assert price == pytest.approx(expected_price, rel=1e-15)


def test_slip_fee_works():
    omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            'USD': {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)}
        },
        asset_fee=0.0025,
        lrna_fee=0.0005,
        slip_factor=1
    )
    omnipool.max_lrna_fee = 1.0  # disable LRNA fee cap for this test

    feeless_pool = omnipool.copy()
    feeless_pool.slip_factor = 0
    agent1 = Agent(enforce_holdings=False)
    agent2 = Agent(enforce_holdings=False)
    omnipool.swap(agent1, tkn_sell='USD', tkn_buy='HDX', buy_quantity=mpf(1000))
    # agent2 should get a better result due to no slip fee (higher holdings of USD after the trade)
    feeless_pool.swap(agent2, tkn_sell='USD', tkn_buy='HDX', buy_quantity=mpf(1000))
    if agent1.holdings['USD'] >= agent2.holdings['USD']:
        raise AssertionError('Slip fee did not reduce output amount.')


def test_buy_sell_equivalency():
    omnipool = oamm.OmnipoolState(
        tokens={
            'USD': {'liquidity': mpf(1000000000), 'LRNA': mpf(1000000000)},
            'HDX': {'liquidity': mpf(10000000000), 'LRNA': mpf(100000000)}
        },
        lrna_fee=0.0025,
        asset_fee=0.0025,
        slip_factor=1.0
    )
    omnipool.max_asset_fee = 0.05
    omnipool.max_lrna_fee = 0.05
    buy_quantity = mpf(10000)
    sell_quantity = omnipool.calculate_sell_from_buy(
        tkn_buy='HDX',
        tkn_sell='USD',
        buy_quantity=buy_quantity
    )
    reverse_buy = omnipool.calculate_buy_from_sell(
        tkn_buy='HDX',
        tkn_sell='USD',
        sell_quantity=sell_quantity
    )
    if reverse_buy != pytest.approx(buy_quantity, rel=1e-12):
        raise AssertionError('Buy/sell equivalency failed.')


def test_calculate_buy_vs_sell():
    omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)},
            'USD': {'liquidity': mpf(1000000), 'LRNA': mpf(1000000)}
        },
        asset_fee=0,
        lrna_fee=0,
        slip_factor=1.0
    )
    omnipool.max_lrna_fee = 0.01  # set LRNA fee cap for this test
    sell_quantity = mpf(1000)
    outputs = omnipool.calculate_out_given_in(tkn_buy='HDX', tkn_sell='USD', sell_quantity=sell_quantity)
    buy_quantity, asset_fee_total, lrna_fee_total, slip_fee_total = outputs
    outputs = omnipool.calculate_in_given_out(tkn_buy='HDX', tkn_sell='USD', buy_quantity=buy_quantity)
    sell_quantity_2, asset_fee_total_2, lrna_fee_total_2, slip_fee_total_2 = outputs
    if sell_quantity != pytest.approx(sell_quantity_2, rel=1e-12):
        raise AssertionError('Calculate buy vs sell quantities do not match.')
    if asset_fee_total != pytest.approx(asset_fee_total_2, rel=1e-12):
        raise AssertionError('Calculate buy vs sell asset fees do not match.')
    if lrna_fee_total != pytest.approx(lrna_fee_total_2, rel=1e-12):
        raise AssertionError('Calculate buy vs sell LRNA fees do not match.')
    if slip_fee_total != pytest.approx(slip_fee_total_2, rel=1e-12):
        raise AssertionError('Calculate buy vs sell slip fees do not match.')


def test_price_after_fees():
    omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': mpf(10000000), 'LRNA': mpf(10000)},
            'USD': {'liquidity': mpf(100000), 'LRNA': mpf(10000)}
        },
        asset_fee=0.00125,
        lrna_fee=0.00025,
        slip_factor=1.0,  # 1.0,
        lrna_fee_burn=1.0,
        lrna_mint_pct=1.0
    )
    omnipool.max_lrna_fee = 0.01
    agent = Agent(enforce_holdings=False)
    start_price = omnipool.lrna_price("HDX")
    for _ in range(10):
        omnipool.swap(agent, tkn_sell='HDX', tkn_buy='USD', sell_quantity=mpf(100000))

    for _ in range(10):
        omnipool.swap(agent, tkn_sell='USD', tkn_buy='HDX', buy_quantity=mpf(100000))
    end_price = omnipool.lrna_price("HDX")
    pass
