import pytest
from hypothesis import given, strategies as st, settings, reproduce_failure
from mpmath import mp, mpf
import hydradx.model.run as run
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.global_state import GlobalState, value_assets, historical_prices
from hydradx.model.amm.money_market import CDP, MoneyMarket, MoneyMarketAsset
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.trade_strategies import liquidate_cdps
from hydradx.model.processing import save_money_market, load_money_market
from hydradx.tests.utils import find_test_directory

mp.dps = 50


def test_cdp_validate():
    debt_asset = "USDT"
    collateral_asset = "DOT"
    init_debt_amt = 1000
    init_collat_amt = 200
    cdp = CDP(debt={debt_asset: init_debt_amt}, collateral={collateral_asset: init_collat_amt})
    if not cdp.validate():
        raise
    cdp.debt["USDT"] = -1
    if cdp.validate():
        raise AssertionError('CDP with negative collateral is invalid.')
    cdp.debt["USDT"] = init_debt_amt
    cdp.collateral["DOT"] = -1
    if cdp.validate():
        raise AssertionError('CDP with negative debt is invalid.')

    cdp = CDP(debt={debt_asset: 0}, collateral={collateral_asset: init_collat_amt})
    if not cdp.validate():
        raise AssertionError('CDP with 0 debt should be valid.')
    cdp = CDP(debt={debt_asset: init_debt_amt}, collateral={collateral_asset: 0})
    if not cdp.validate():  # note that toxic debt does not fail validation
        raise AssertionError('CDP with 0 collateral should be valid.')


# money_market tests

def test_get_oracle_price():
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset('USDT', 1, 100000, 0, 0, 0),
            MoneyMarketAsset('DOT', 10, 100000, 0, 0, 0)
        ]
    )
    if mm.price("DOT", "USDT") != 10:
        raise
    if mm.price("USDT", "DOT") != 0.1:
        raise
    with pytest.raises(Exception):
        mm.price("DOT", "ETH")


def test_is_liquidatable():
    cdp = CDP(debt={"USDT": 2000 * 0.7 - 0.00001}, collateral={"DOT": 200})
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset('USDT', 1, liquidation_threshold=0.7),
            MoneyMarketAsset('DOT', 10, liquidation_threshold=0.7)
        ], cdps=[cdp]
    )
    if mm.is_liquidatable(cdp):
        raise
    cdp.debt['USDT'] = 2000 * 0.7 + 0.00001
    if not mm.is_liquidatable(cdp):
        raise


def test_is_fully_liquidatable():
    liquidation_threshold = 0.7
    full_liquidation_threshold = liquidation_threshold / 0.95
    cdp = CDP(debt={"USDT": 2000 * full_liquidation_threshold - 0.00001}, collateral={"DOT": 200})
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset('USDT', 1, liquidation_threshold=liquidation_threshold),
            MoneyMarketAsset('DOT', 10, liquidation_threshold=liquidation_threshold)
        ], cdps=[cdp],
        full_liquidation_threshold=0.95
    )
    if mm.is_fully_liquidatable(cdp):
        raise AssertionError('CDP should be fully liquidatable')
    cdp.debt["USDT"] = 2000 * full_liquidation_threshold + 0.00001
    if not mm.is_fully_liquidatable(cdp):
        raise


def test_get_liquidate_collateral_amt():
    debt_amt = 1000
    collateral_amt = 200
    spot_price = 10
    penalty = 0.01
    collat_liquidated_exp = 101
    cdp = CDP(debt={"USDT": debt_amt}, collateral={"DOT": collateral_amt})
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset("USDT", 1, liquidation_bonus=penalty, liquidation_threshold=0.1),
            MoneyMarketAsset("DOT", spot_price, liquidation_bonus=penalty, liquidation_threshold=0.1)
        ]
    )

    collat_liquidated, debt_repaid = mm.calculate_liquidation(cdp, "DOT", "USDT", debt_amt)
    if collat_liquidated != pytest.approx(collat_liquidated_exp, rel=1e-15):
        raise ValueError('Collateral liquidated was not the expected amount.')
    if debt_repaid != debt_amt:
        raise ValueError('Debt repaid was not the expected amount.')


def test_liquidate():
    debt_amt = 1000
    collat_amt = 100
    # borrow_agent = Agent()
    cdp = CDP(debt={"USDT": debt_amt}, collateral={"DOT": collat_amt})
    collat_price = 11
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset("USDT", 1, liquidation_bonus=0.01, liquidation_threshold=0.7),
            MoneyMarketAsset("DOT", collat_price, liquidation_bonus=0.01, liquidation_threshold=0.7)
        ], cdps=[cdp]
    )
    repay_amt = debt_amt / 2
    collat_sold_expected = repay_amt / collat_price * 1.01
    agent = Agent(holdings={"USDT": repay_amt})
    mm.liquidate(cdp, agent, debt_asset="USDT", collateral_asset="DOT", repay_amount=repay_amt)
    if debt_amt - repay_amt != cdp.debt["USDT"]:
        raise
    if agent.holdings["USDT"] != 0:
        raise
    if agent.holdings["DOT"] != collat_sold_expected:
        raise
    if cdp.collateral["DOT"] != collat_amt - collat_sold_expected:
        raise
    if not mm.validate():
        raise


def test_liquidate_not_liquidatable():
    debt_amt = 1000
    collat_amt = 500
    cdp = CDP(debt={"USDT": debt_amt}, collateral={"DOT": collat_amt})
    collat_price = 11
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset("USDT", 1, 100000, 0.01, 0.7, 0),
            MoneyMarketAsset("DOT", collat_price, 100000, 0.01, 0.7, 0)
        ], cdps=[cdp]
    )
    repay_amount = debt_amt / 2
    agent = Agent(holdings={"USDT": repay_amount})
    mm.liquidate(cdp, agent, debt_asset="USDT", collateral_asset="DOT", repay_amount=repay_amount)
    if debt_amt != cdp.debt["USDT"]:
        raise
    if agent.holdings["USDT"] != repay_amount:
        raise
    if agent.validate_holdings("DOT"):
        raise
    if cdp.collateral["DOT"] != collat_amt:
        raise
    if not mm.validate():
        raise


def test_liquidate_fails():
    debt_asset = "USDT"
    collateral_asset = "DOT"
    init_debt_amt = 1000
    init_collat_amt = 100
    cdp = CDP(debt={debt_asset: init_debt_amt}, collateral={collateral_asset: init_collat_amt})
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset("USDT", 1, 100000, 0.01, 0.7, 0),
            MoneyMarketAsset("DOT", 11, 100000, 0.01, 0.7, 0)
        ], cdps=[cdp.copy()]
    )
    agent = Agent(holdings={"USDT": 10000, "DOT": 10000})

    with pytest.raises(Exception):  # trying to liquidate more than debt amount, should fail
        mm.liquidate(cdp, agent, debt_asset="USDT", collateral_asset="DOT", repay_amount=init_debt_amt * 2)


def test_liquidate_undercollateralized():
    debt_asset = "USDT"
    collateral_asset = "DOT"
    init_debt_amt = 1000
    init_collat_amt = 10
    penalty = 0.01

    # not enough collateral, should liquidate all collateral and have debt left over
    cdp = CDP(debt={debt_asset: init_debt_amt}, collateral={collateral_asset: init_collat_amt})
    dot_price = 11
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset("USDT", 1, liquidation_bonus=penalty, liquidation_threshold=0.7),
            MoneyMarketAsset("DOT", dot_price, liquidation_bonus=penalty, liquidation_threshold=0.7)
        ], cdps=[cdp.copy()]
    )
    agent = Agent(holdings={"USDT": 10000, "DOT": 0})

    mm.liquidate(cdp, agent, debt_asset="USDT", collateral_asset="DOT", repay_amount=init_debt_amt)
    if cdp.collateral["DOT"] != 0:
        raise
    debt_liq = 110 / (1 + penalty)
    if cdp.debt["USDT"] != init_debt_amt - debt_liq:
        raise
    if agent.holdings["USDT"] != 10000 - debt_liq:
        raise
    if agent.holdings["DOT"] != init_collat_amt:
        raise


@given(
    st.floats(min_value=0.1, max_value=1.0),
    st.floats(min_value=0.01, max_value=0.05),
    st.floats(min_value=0.6, max_value=0.75),
    st.floats(min_value=0.75, max_value=0.9),
    st.floats(min_value=0.2, max_value=0.8),

)
def test_liquidate_fuzz_ltv(liq_pct: float, penalty: float, liq_threshold: float,
                            full_liq_threshold: float, partial_liq_pct: float):
    ltv_ratio = 1

    collat_amt = mpf(100)
    collat_price = mpf(11)

    debt_amt = collat_amt * collat_price * ltv_ratio
    cdp = CDP(debt={"USDT": debt_amt}, collateral={"DOT": collat_amt})
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset("USDT", 1, liquidation_bonus=penalty, liquidation_threshold=liq_threshold),
            MoneyMarketAsset("DOT", collat_price, liquidation_bonus=penalty, liquidation_threshold=liq_threshold)
        ], cdps=[cdp],
        full_liquidation_threshold=liq_threshold / full_liq_threshold,
        close_factor=partial_liq_pct,
    )
    repay_amount = debt_amt * liq_pct
    collat_sold_expected = repay_amount / collat_price * (1 + penalty)
    agent = Agent(holdings={"USDT": repay_amount})
    mm.liquidate(cdp, agent, debt_asset="USDT", collateral_asset="DOT", repay_amount=repay_amount)

    if debt_amt - repay_amount == cdp.debt["USDT"]:  # case 1: we liquidated
        if ltv_ratio < liq_threshold:
            raise
        if ltv_ratio < full_liq_threshold and liq_pct > partial_liq_pct:
            raise
        if cdp.collateral["DOT"] != pytest.approx(collat_amt - collat_sold_expected, rel=1e-15):
            raise
    elif cdp.collateral["DOT"] == 0:  # case 2: we liquidated but left some toxic debt
        if ltv_ratio <= 1 / (1 + penalty):
            raise
    else:
        if cdp.debt["USDT"] != debt_amt:  # shouldn't liquidate some partial amount
            raise
        if ltv_ratio >= full_liq_threshold:
            raise
        if ltv_ratio >= liq_threshold and liq_pct <= partial_liq_pct:
            raise


def test_borrow():
    borrow_asset = "USDT"
    collateral_asset = "DOT"
    borrow_amt = 500
    collat_amt = 100
    agent = Agent(holdings={collateral_asset: collat_amt})
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset(
                borrow_asset, price=1, liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=0.7
            ),
            MoneyMarketAsset(
                collateral_asset, price=11, liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=0.7
            )
        ]
    )
    cdp = mm.borrow(agent, borrow_asset, collateral_asset, borrow_amt, collat_amt)
    if agent.holdings[borrow_asset] != borrow_amt:
        raise AssertionError("Agent didn't borrow the correct amount.")
    if agent.validate_holdings(collateral_asset):
        raise AssertionError("Agent should have no collateral left.")
    if len(mm.cdps) != 1:
        raise AssertionError("CDP not created in money market.")
    if cdp.debt != {"USDT": borrow_amt}:
        raise AssertionError("CDP debt not correct.")
    if cdp.collateral != {"DOT": collat_amt}:
        raise AssertionError("CDP collateral not correct.")
    if not mm.validate():
        raise AssertionError("Money market invalid.")

    agent2 = Agent(holdings={collateral_asset: collat_amt})
    mm.borrow(agent2, borrow_asset, collateral_asset, borrow_amt, collat_amt)
    if len(mm.cdps) != 2:
        raise AssertionError("Second CDP not created.")
    if not mm.validate():
        raise AssertionError("Money market invalid.")


def test_borrow_fails():
    borrow_asset = "USDT"
    collateral_asset = "DOT"
    collat_amt = 100
    agent = Agent(holdings={collateral_asset: collat_amt})
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset(borrow_asset, price=1, liquidation_bonus=0.01, liquidation_threshold=0.7),
            MoneyMarketAsset(collateral_asset, price=10, liquidation_bonus=0.01, liquidation_threshold=0.7),
            MoneyMarketAsset("BTC", price=100000, liquidation_bonus=0.01, liquidation_threshold=0.7),
        ]
    )
    borrow_amt = collat_amt * 10 * 0.75
    mm.borrow(agent, borrow_asset, collateral_asset, borrow_amt, collat_amt)
    if not mm.fail:
        raise AssertionError('Borrowing more than LTV limit should fail.')
    borrow_amt = collat_amt * 10 * 0.50
    with pytest.raises(Exception):  # should fail because debt asset == collateral asset
        mm.borrow(agent, collateral_asset, collateral_asset, borrow_amt, collat_amt)
    with pytest.raises(Exception):  # should fail because collateral asset is not in agent holdings
        mm.borrow(agent, borrow_asset, "ETH", borrow_amt, collat_amt)
    with pytest.raises(Exception):  # should fail because borrow asset is not in liquidity
        mm.borrow(agent, "ETH", collateral_asset, borrow_amt, collat_amt)
    with pytest.raises(Exception):  # should fail because of missing oracle
        mm.borrow(agent, "BTC", collateral_asset, borrow_amt, collat_amt)


# def test_repay():
#     agent = Agent(holdings={"USDT": 1000})
#     cdp = CDP(debt={"USDT": 1000}, collateral={"DOT": 200})
#     mm = MoneyMarket(
#         assets=[
#             MoneyMarketAsset("USDT", 1, 1000000, 0.01, 0.7, 0),
#             MoneyMarketAsset("DOT", 10, 1000000, 0.01, 0.7, 0)
#         ], cdps=[cdp]
#     )
#     mm.repay(0)
#     if agent.holdings["USDT"] != 0:
#         raise
#     if mm.borrowed["USDT"] != 0:
#         raise
#     if not mm.validate():
#         raise
#
#
# def test_repay_fails():
#     agent = Agent(holdings={"USDT": 500})
#     cdp = CDP(debt={"USDT": 1000}, collateral={"DOT": 200})
#     mm = MoneyMarket(
#         assets=[
#             MoneyMarketAsset("USDT", 1, 1000000, 0.01, 0.7, 0),
#             MoneyMarketAsset("DOT", 10, 1000000, 0.01, 0.7, 0)
#         ], cdps=[cdp]
#     )
#     with pytest.raises(Exception):  # should fail because agent does not have enough funds
#         mm.repay(agent, 0)


def omnipool_setup_for_liquidation_testing() -> OmnipoolState:
    prices = {'DOT': 7, 'HDX': 0.02, 'USDT': 1, 'WETH': 2500, 'iBTC': 45000}

    assets = {
        'DOT': {'usd price': prices['DOT'], 'weight': 0.40},
        'HDX': {'usd price': prices['HDX'], 'weight': 0.10},
        'USDT': {'usd price': prices['USDT'], 'weight': 0.30},
        'WETH': {'usd price': prices['WETH'], 'weight': 0.10},
        'iBTC': {'usd price': prices['iBTC'], 'weight': 0.10}
    }

    lrna_price_usd = 35
    initial_omnipool_tvl = 20000000
    liquidity = {}
    lrna = {}

    for tkn, info in assets.items():
        liquidity[tkn] = mpf(initial_omnipool_tvl * info['weight'] / info['usd price'])
        lrna[tkn] = mpf(initial_omnipool_tvl * info['weight'] / lrna_price_usd)

    omnipool = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in assets
        },
        preferred_stablecoin='USDT',
        asset_fee=0.0025,
        lrna_fee=0.0005
    )

    return omnipool


@given(
    st.floats(min_value=200, max_value=2000)
)
def test_omnipool_liquidate_cdp_oracle_equals_spot_small_cdp(collateral_amt: float):
    omnipool = omnipool_setup_for_liquidation_testing()
    init_pool = omnipool.copy()
    dot_price = omnipool.price("DOT", "USDT")

    collat_ratio = 0.75

    debt_amt = collat_ratio * dot_price * collateral_amt
    init_cdp = CDP(debt={'USDT': mpf(debt_amt)}, collateral={'DOT': mpf(collateral_amt)})
    cdp = init_cdp.copy()
    penalty = 0.01

    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset(
                name="USDT", price=1, liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=0.7, ltv=0.6
            ), MoneyMarketAsset(
                name="DOT", price=dot_price, liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=0.7, ltv=0.6
            )
        ], cdps=[cdp]
    )

    liquidation_amount = debt_amt
    init_agent = Agent(
        enforce_holdings=False
    )
    treasury_agent = init_agent.copy()
    mm.liquidate(cdp, treasury_agent, 'USDT', 'DOT', liquidation_amount)
    omnipool.swap(treasury_agent, 'USDT', 'DOT', buy_quantity=-treasury_agent.get_holdings('USDT'))

    before_DOT = init_agent.get_holdings('DOT') + init_cdp.collateral['DOT'] + init_pool.liquidity['DOT']
    before_USDT = init_agent.get_holdings('USDT') - init_cdp.debt['USDT'] + init_pool.liquidity['USDT']
    final_DOT = treasury_agent.get_holdings('DOT') + cdp.collateral['DOT'] + omnipool.liquidity['DOT']
    final_USDT = treasury_agent.get_holdings('USDT') - cdp.debt['USDT'] + omnipool.liquidity['USDT']
    if before_DOT != pytest.approx(final_DOT, rel=1e-20):  # check that total collateral asset amounts are correct
        raise ValueError('System-wide DOT amounts do not match')
    if before_USDT != pytest.approx(final_USDT, rel=1e-20):  # check that total debt asset amounts are correct
        raise ValueError('System-wide USDT amounts do not match')

    if not cdp.validate():
        raise ValueError('CDP is not valid after liquidation')
    if not mm.validate():
        raise ValueError('Money market is not valid after liquidation')

    if treasury_agent.holdings['DOT'] > penalty * (init_cdp.collateral['DOT'] - cdp.collateral['DOT']):
        raise ValueError("Treasury should collect at most penalty")
    for tkn in treasury_agent.holdings:
        if tkn not in cdp.collateral and treasury_agent.get_holdings(tkn) != 0:
            raise ValueError("Treasury should not have any holdings of other tokens")


def test_omnipool_liquidate_cdp_delta_debt_too_large():
    collat_ratio = 5.0
    collateral_amt = 200

    debt_amt = collat_ratio * collateral_amt
    init_cdp = CDP(debt={'USDT': mpf(debt_amt)}, collateral={'DOT': mpf(collateral_amt)})
    cdp = init_cdp.copy()

    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset(
                name="USDT", price=1, liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=0.7, ltv=0.6
            ), MoneyMarketAsset(
                name="DOT", price=2, liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=0.7, ltv=0.6
            )
        ], cdps=[cdp]
    )
    treasury_agent = Agent(holdings={"USDT": mpf(0), "DOT": mpf(0)})
    with pytest.raises(Exception):  # liquidation should fail because delta_debt > cdp.debt_amt
        mm.liquidate(cdp, treasury_agent, 'USDT', 'DOT', cdp.debt_amt * 1.01)


def test_omnipool_liquidate_cdp_not_profitable():
    collat_ratio = 5.0
    collateral_amt = 2000
    omnipool = omnipool_setup_for_liquidation_testing()
    dot_price = omnipool.price("DOT", "USDT")
    omnipool.lrna['DOT'] /= 2  # manipulate omnipool so that liquidation is not profitable

    debt_amt = collat_ratio * collateral_amt
    init_cdp = CDP(debt={'USDT': mpf(debt_amt)}, collateral={'DOT': mpf(collateral_amt)})
    cdp = init_cdp.copy()

    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset(
                name="USDT", price=1, liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=0.7, ltv=0.6
            ), MoneyMarketAsset(
                name="DOT", price=dot_price, liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=0.7, ltv=0.6
            )
        ], cdps=[cdp]
    )

    init_agent = Agent(enforce_holdings=False, trade_strategy=liquidate_cdps('omnipool'))
    treasury_agent = init_agent.copy()
    initial_state = GlobalState(
        pools={"omnipool": omnipool, "money_market": mm},
        agents={"agent": treasury_agent}
    )
    new_state = initial_state.copy().evolve()
    new_agent = new_state.agents["agent"]
    if sum(new_agent.holdings.values()) != 0:
        raise ValueError('Agent holdings should not change after failed liquidation')


#################################################
# tests for liquidate_against_omnipool          #
#################################################

# test_liquidate_against_omnipool_full_liquidation
# - tests full liquidation
# - tests liquidation of some CDPs but not others
# - test with CDP list with different assets

# test_liquidate_against_omnipool_partial_liquidation
# - test partial liquidation due to higher full liquidation threshold
# - test partial liquidation due to limited Omnipool trade profitability

# test_liquidate_against_omnipool_no_liquidation
# - test no liquidation due to overcollateralization
# - test no liquidation due to undercollateralization
# - test no liquidation due to unprofitability of liquidation against Omnipool

# fuzz test logic that determines how much is liquidated

# For fuzz test of omnipool_liquidate_cdp, we will want to see that
# - any liquidation results in treasury profit
# - if liquidation doesn't happen, buying delta_debt requires too much collateral


@given(st.floats(min_value=0.7, max_value=0.9), st.floats(min_value=0.5, max_value=0.7),
       st.floats(min_value=2.0, max_value=3.0))
def test_liquidate_against_omnipool_full_liquidation(ratio1: float, ratio2: float, ratio3: float):
    omnipool = omnipool_setup_for_liquidation_testing()

    # CDP1 should be fully liquidated
    collateral_amt1 = 100
    debt_amt1 = ratio1 * collateral_amt1 * omnipool.price("DOT", "USDT")
    cdp1 = CDP(debt={'USDT': debt_amt1}, collateral={'DOT': collateral_amt1})

    # CDP2 should not be liquidated at all
    collateral_amt2 = 101
    debt_amt2 = ratio2 * collateral_amt2 * omnipool.price("USDT", "HDX")
    cdp2 = CDP(debt={'HDX': debt_amt2}, collateral={'USDT': collateral_amt2})

    # CDP3 should be fully liquidated, due to lower liquidation threshold for HDX
    collateral_amt3 = 102
    debt_amt3 = ratio2 * collateral_amt3 / 0.95 * omnipool.price("HDX", "USDT")
    cdp3 = CDP(debt={'USDT': debt_amt3}, collateral={'HDX': collateral_amt3})

    # CDP4 should be fully liquidated, with debt left over
    collateral_amt4 = 103
    debt_amt4 = ratio3 * collateral_amt4 * omnipool.price("HDX", "USDT")
    cdp4 = CDP(debt={'USDT': debt_amt4}, collateral={'HDX': collateral_amt4})

    liq_agent = Agent(enforce_holdings=False, trade_strategy=liquidate_cdps('omnipool'))
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset(
                name="USDT", price=1, liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=0.8, ltv=0.6
            ), MoneyMarketAsset(
                name="DOT", price=omnipool.price("DOT", "USDT"),
                liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=0.6, ltv=0.6
            ), MoneyMarketAsset(
                name="HDX", price=omnipool.price("HDX", "USDT"),
                liquidity=100000000, liquidation_bonus=0.01, liquidation_threshold=0.5, ltv=0.6
            )
        ], cdps=[cdp1, cdp2, cdp3, cdp4]
    )

    state = GlobalState(agents={"liq_agent": liq_agent},
                        pools=[omnipool, mm])

    state.evolve()

    if cdp1.debt['USDT'] != 0:
        raise ValueError("CDP1 should be fully liquidated.")
    if cdp2.debt['HDX'] != debt_amt2:
        raise ValueError("CDP2 should not be liquidated.")
    if cdp3.debt['USDT'] != 0:
        raise ValueError("CDP3 should be fully liquidated.")
    if cdp4.collateral['HDX'] != 0:
        raise ValueError("CDP4 should be fully liquidated.")
    if cdp4.debt['USDT'] == 0:
        raise ValueError("CDP4 should still have debt left over.")
    if cdp4.collateral['HDX'] > 0:
        raise ValueError("CDP4 should be fully liquidated.")


def test_liquidate_against_omnipool_partial_liquidation():
    omnipool = omnipool_setup_for_liquidation_testing()

    dot_liq_threshold = 0.6
    dot_full_liq_threshold = dot_liq_threshold / 0.95
    collateral_amt1 = 200
    collat_ratio1 = dot_full_liq_threshold - 0.0001
    debt_amt1 = collat_ratio1 * collateral_amt1 * omnipool.price("DOT", "USDT")
    cdp1 = CDP(debt={'USDT': debt_amt1}, collateral={'DOT': collateral_amt1})

    hdx_full_liq_threshold = 0.7
    hdx_liq_threshold = 0.7
    collateral_amt2 = 10000000  # too much to fully liquidate on Omnipool
    collat_ratio2 = 0.7
    debt_amt2 = collat_ratio2 * collateral_amt2 * omnipool.price("HDX", "USDT")
    cdp2 = CDP(debt={"USDT": debt_amt2}, collateral={"HDX": collateral_amt2})

    liq_agent = Agent(enforce_holdings=False, trade_strategy=liquidate_cdps('omnipool', iters=100))
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset(
                name="USDT", price=1, liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=0.7, ltv=0.6
            ), MoneyMarketAsset(
                name="DOT", price=omnipool.price("DOT", "USDT"),
                liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=dot_liq_threshold, ltv=0.6
            ), MoneyMarketAsset(
                name="HDX", price=omnipool.price("HDX", "USDT"),
                liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=hdx_liq_threshold, ltv=0.6
            )
        ], cdps=[cdp1, cdp2]
    )

    GlobalState(
        agents={"liq_agent": liq_agent},
        pools={"omnipool": omnipool, "money market": mm}
    ).evolve()

    if cdp1.debt["USDT"] != debt_amt1 * (1 - mm.partial_liquidation_pct):
        raise ValueError("CDP1 should be partially liquidated by partial_liquidation_pct")
    if cdp2.debt["USDT"] == 0:
        raise ValueError("CDP2 should not be fully liquidated")
    if cdp2.debt["USDT"] == debt_amt2:
        raise ValueError("CDP2 should be partially liquidated")
    hdx_liquidated = collateral_amt2 - cdp2.collateral['HDX']
    # if liq_agent.holdings["HDX"] / hdx_liquidated > 1e-25:
    #     raise ValueError("If liquidation agent is profitable, they should have liquidated more")


@given(st.floats(min_value=0.0, max_value=0.7, exclude_min=True, exclude_max=True))
def test_liquidate_against_omnipool_no_liquidation(ratio1: float):
    omnipool = omnipool_setup_for_liquidation_testing()

    collateral_amt1 = 200
    debt_amt1 = ratio1 * collateral_amt1 * omnipool.price("DOT", "USDT")
    cdp1 = CDP(debt={'USDT': debt_amt1}, collateral={'DOT': collateral_amt1})

    collateral_amt2 = 10
    price_mult = 10  # this offsets the oracle price from Omnipool price, making liquidation unprofitable
    debt_amt2 = 0.7 * collateral_amt2 * price_mult * omnipool.price("WETH", "USDT")
    cdp2 = CDP(debt={'USDT': debt_amt2}, collateral={'WETH': collateral_amt2})

    liq_agent = Agent(enforce_holdings=False, trade_strategy=liquidate_cdps(pool_id='omnipool'))
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset(
                name='USDT', price=1, liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=0.7, ltv=0.6
            ), MoneyMarketAsset(
                name='DOT', price=omnipool.price("DOT", "USDT"),
                liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=0.7, ltv=0.6
            ), MoneyMarketAsset(
                name='WETH', price=price_mult * omnipool.price("WETH", "USDT"),
                liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=0.7, ltv=0.6
            ), MoneyMarketAsset(
                name='HDX', price=omnipool.price("HDX", "USDT"),
                liquidity=1000000, liquidation_bonus=0.01, liquidation_threshold=0.7, ltv=0.6
            )
        ], cdps=[cdp1, cdp2]
    )

    state = GlobalState(
        agents={"liq_agent": liq_agent},
        pools={"omnipool": omnipool, "money_market": mm}
    )

    state.evolve()

    if debt_amt1 != cdp1.debt['USDT']:
        raise ValueError("No liquidation should occur")
    if debt_amt2 != cdp2.debt['USDT']:
        raise ValueError("No liquidation should occur")


@settings(print_blob=True)
@given(
    st.floats(min_value=100, max_value=100000),
    # st.floats(min_value=0.1, max_value=10.0),
    st.floats(min_value=0.8, max_value=1.0),
    st.floats(min_value=0.5, max_value=1.5)
)
def test_liquidate_against_omnipool_fuzz(collateral_amt1: float, ratio1: float, price_mult: float):
    omnipool = omnipool_setup_for_liquidation_testing()

    # collateral_amt1 = 516949.1872315724
    # ratio1 = 0.8699232790986524
    # price_mult = 0.5238060155349247
    liq_threshold = mpf(0.7)
    full_liq_threshold = mpf(0.8)

    debt_amt1 = ratio1 * mpf(collateral_amt1) * price_mult * omnipool.price("DOT", "USDT")
    cdp1 = CDP({'USDT': debt_amt1}, {'DOT': mpf(collateral_amt1)})

    liq_agent = Agent(enforce_holdings=False, trade_strategy=liquidate_cdps('omnipool'))
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset(
                name="DOT",
                price=omnipool.price("DOT", "USDT") * price_mult,
                liquidity=omnipool.liquidity["DOT"],
                liquidation_bonus=0.01,
                liquidation_threshold=liq_threshold,
                ltv=0.6
            ),
            MoneyMarketAsset(
                name="USDT",
                price=1,
                liquidity=omnipool.liquidity["USDT"],
                liquidation_bonus=0.01,
                liquidation_threshold=liq_threshold,
                ltv=0.6
            )
        ],
        full_liquidation_threshold=full_liq_threshold,
        close_factor=0.5,
        cdps=[cdp1],
    )

    # evolve_function = liquidate_against_omnipool("omnipool", "liq_agent", 100)
    state = GlobalState(
        agents={"liq_agent": liq_agent},
        pools=[omnipool, mm],
    )

    cdp1_archive = cdp1.copy()
    mm_archive = mm.copy()

    state.evolve()
    liq_agent.trade_strategy.execute(state, 'liq_agent')

    if cdp1.debt['USDT'] == 0:  # fully liquidated
        assert ratio1 >= full_liq_threshold
    elif cdp1.collateral['DOT'] == 0:  # fully liquidated, bad debt remaining
        assert ratio1 > 1/(1 + mm.get_liquidation_bonus('DOT', 'USDT'))
    elif cdp1.debt['USDT'] == debt_amt1:  # not liquidated
        if ratio1 <= liq_threshold:  # 1. overcollateralized
            pass
        elif price_mult >= 1:  # 3. not profitable to liquidate
            pass
        else:
            raise ValueError("CDP should be liquidated")
    elif 0 < cdp1.debt['USDT'] < debt_amt1:  # partially liquidated
        assert ratio1 >= liq_threshold
        if (
                ratio1 * mm.full_liquidation_threshold - mm.get_liquidation_threshold('DOT', 'USDT') <= 1e-20
                and cdp1.debt['USDT'] / debt_amt1 == 1 - mm.partial_liquidation_pct
        ):
            pass  # partially liquidated due to partial_liquidation_pct
        elif ratio1 > 1 - mm.get_liquidation_bonus('DOT', 'USDT'):  # 2. undercollateralized, partially but not fully liquidated
            pass
        # elif liq_agent.holdings["DOT"] / (collateral_amt1 - cdp1.collateral['DOT']) > 1e-25:
        #     raise ValueError("If liquidation agent is profitable, they should have liquidated more")
        elif liq_agent.holdings["DOT"] < 0:
            raise ValueError("Liquidation agent should not have negative holdings")
    else:
        raise ValueError("CDP debt amount should not go above initial debt_amt or below 0")


def test_set_mm_oracles_to_external_market():
    omnipool = omnipool_setup_for_liquidation_testing()
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset(
                name=tkn,
                price=omnipool.price(tkn, "USDT"),
                liquidity=omnipool.liquidity[tkn],
                liquidation_bonus=0.05,
                liquidation_threshold=0.7,
                ltv=0.6
            )
            for tkn in omnipool.liquidity
        ], oracle_source=omnipool.usd_price
    )
    prices = {tkn: omnipool.price(tkn, "USDT") for tkn in omnipool.liquidity}
    mm.update()
    for tkn in mm.liquidity:
        if mm.price(tkn) != pytest.approx(prices[tkn], rel=1e-40):
            raise ValueError("Oracle price should be set to Omnipool spot price")


def test_liquidations():
    initial_price = {'DOT': 7, 'HDX': 0.02, 'USDT': 1, 'WETH': 2500, 'iBTC': 45000}
    initial_liquidity = {'USDT': 1000000, 'DOT': 1000000, 'HDX': 100000000}
    default_liquidation_threshold = 0.7
    default_liquidation_bonus = 0.02
    default_ltv = 0.6
    assets = [
        MoneyMarketAsset(
            name=tkn,
            price=initial_price[tkn],
            liquidity=initial_liquidity[tkn],
            liquidation_bonus=default_liquidation_bonus,
            liquidation_threshold=default_liquidation_threshold,
            ltv=default_ltv,
        ) for tkn in ['DOT', 'HDX', 'USDT']
    ]

    # cdp1 should be liquidated fully
    cdp1 = CDP(
        {'USDT': initial_price['DOT'] * default_liquidation_threshold / 0.95 + 0.00001},
        {'DOT': 1}
    )
    # cdp2 should not be liquidated
    cdp2 = CDP({'USDT': 1}, {'HDX': 1 / initial_price['HDX'] / default_liquidation_threshold + 0.00001})
    # cdp3 should be partially liquidated
    cdp3 = CDP(
        {'HDX': 20 * initial_price['DOT'] / initial_price['HDX'] * default_liquidation_threshold / 0.95 - 0.00001},
        {'DOT': 20}
    )

    cdps = [cdp1, cdp2, cdp3]
    mm = MoneyMarket(
        assets=assets,
        cdps=cdps.copy(),
    )

    liquidator = Agent(enforce_holdings=False)
    debt_amount = cdp1.debt['USDT']
    mm.liquidate(cdp1, liquidator, 'USDT', 'DOT')
    agent_value_1 = value_assets(mm.prices, liquidator.holdings)
    if agent_value_1 != pytest.approx(debt_amount * default_liquidation_bonus, rel=1e-20):
        raise ValueError("Liquidator should have profited by an amount equal to liquidation bonus.")

    mm.liquidate(cdp2, liquidator,'USDT', 'HDX')
    if value_assets(mm.prices, liquidator.holdings) != agent_value_1:
        raise ValueError("Liquidating a healthy CDP should have no effect")
    cdp3_debt_amount = cdp3.debt['HDX']
    mm.liquidate(cdp3, liquidator, 'HDX', 'DOT')
    if cdp3.debt['HDX'] == 0:
        raise ValueError("CDP3 should not be fully liquidated")
    elif cdp3.debt['HDX'] == cdp3_debt_amount:
        raise ValueError("CDP3 should be partially liquidated")
    gains = value_assets(mm.prices, liquidator.holdings) - agent_value_1
    if gains != pytest.approx(cdp3_debt_amount / 2 * default_liquidation_bonus * initial_price['HDX'], rel=1e-20):
        raise ValueError("Liquidator should have profited by an amount equal to liquidation bonus.")


def test_trade_strategy():
    omnipool = omnipool_setup_for_liquidation_testing()
    initial_price = {'DOT': 7, 'HDX': 0.02, 'USDT': 1, 'WETH': 2500, 'iBTC': 45000}
    final_price = {'DOT': 5, 'HDX': 0.04, 'USDT': 1, 'WETH': 2700, 'iBTC': 47000}
    time_steps = 3
    price_path = {
        tkn: (price for price in [
            initial_price[tkn] + (final_price[tkn] - initial_price[tkn]) * ((i + 1) / time_steps)
            for i in range(time_steps)
        ]) for tkn in initial_price
    }

    initial_liquidity = {'USDT': 1000000, 'DOT': 1000000, 'HDX': 100000000}
    default_liquidation_threshold = 0.7
    default_liquidation_bonus = 0.02
    default_ltv = 0.6
    assets = [
        MoneyMarketAsset(
            name=tkn,
            price=initial_price[tkn],
            liquidity=initial_liquidity[tkn],
            liquidation_bonus=default_liquidation_bonus,
            liquidation_threshold=default_liquidation_threshold,
            ltv=default_ltv,
        ) for tkn in ['DOT', 'HDX', 'USDT']
    ]

    # cdp1 should be liquidated fully
    cdp1 = CDP(
        {'USDT': final_price['DOT'] * default_liquidation_threshold / 0.95 + 0.00001},
        {'DOT': 1}
    )
    # cdp2 should not be liquidated
    cdp2 = CDP({'USDT': 1}, {'HDX': 1 / initial_price['HDX'] / default_liquidation_threshold + 0.00001})
    # cdp3 should be partially liquidated
    cdp3 = CDP(
        {'HDX': 1000},
        {'DOT': 1000 / final_price['DOT'] * final_price['HDX'] / default_liquidation_threshold * 0.95 + 0.00001}
    )

    cdps = [cdp1, cdp2, cdp3]
    mm = MoneyMarket(
        assets=assets,
        cdps=cdps,
        oracle_source=lambda tkn: next(price_path[tkn])
    )
    liquidator = Agent(
        trade_strategy=liquidate_cdps('omnipool'),
        enforce_holdings=False
    )
    state = GlobalState(
        agents={'liquidator': liquidator},
        pools={'omnipool': omnipool, 'money_market': mm}
    )
    final_state = run.run(state, time_steps)[-1]
    final_cdp1_debt = final_state.pools['money_market'].cdps[0].debt['USDT']
    final_cdp2_debt = final_state.pools['money_market'].cdps[1].debt['USDT']
    final_cdp3_debt = final_state.pools['money_market'].cdps[2].debt['HDX']
    final_mm = final_state.pools['money_market']
    final_liquidator = final_state.agents['liquidator']

    if final_cdp1_debt != 0:
        raise ValueError("CDP1 should be fully liquidated")
    if final_cdp2_debt != 1:
        raise ValueError("CDP2 should not be liquidated")
    if final_cdp3_debt != 500:
        raise ValueError("CDP3 should be half liquidated")
    if final_mm.price("HDX") != 0.04:
        raise ValueError("Oracle price for HDX should be 0.04")
    if final_mm.price("DOT") != 5:
        raise ValueError("Oracle price for DOT should be 5")
    profit = final_liquidator.holdings


def test_full_liquidation_not_profitable():
    omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': 20, 'LRNA': 20},
            'USD': {'liquidity': 20, 'LRNA': 20}
        }
    )
    mm = MoneyMarket(
        assets=[
            MoneyMarketAsset(
                name='HDX',
                price=1,
                liquidity=10,
                liquidation_bonus=0.02,
                liquidation_threshold=0.7,
                ltv=0.6
            ),
            MoneyMarketAsset(
                name='USD',
                price=1,
                liquidity=10,
                liquidation_bonus=0.02,
                liquidation_threshold=0.7,
                ltv=0.6
            )
        ]
    )
    cdp1 = CDP(
        {'USD': 10},
        {'HDX': 10}
    )
    mm.cdps = [cdp1]
    liquidator = Agent(
        trade_strategy=liquidate_cdps('omnipool'),
        enforce_holdings=False
    )
    initial_state = GlobalState(
        agents={'liquidator': liquidator},
        pools={'omnipool': omnipool, 'money_market': mm},
    )
    liquidator.trade_strategy.execute(initial_state, 'liquidator')
    if liquidator.holdings['USD'] != 0:
        raise ValueError("Liquidator should not have USDT")
    if liquidator.holdings['HDX'] == 0:
        raise ValueError("Liquidator should liquidate some HDX")
    if mm.cdps[0].debt['USD'] == 0:
        raise ValueError("CDP should not be fully liquidated")
    er = 1


def test_get_money_market():
    from hydradx.model.processing import get_current_money_market
    mm = get_current_money_market()
    if not isinstance(mm, MoneyMarket):
        raise ValueError("MoneyMarket should be returned")


def test_multiple_collateral():
    omnipool = omnipool_setup_for_liquidation_testing()
    initial_price = {tkn: omnipool.usd_price(tkn, 'USDT') for tkn in ['DOT', 'HDX', 'USDT']}
    final_price = {'DOT': 5, 'HDX': 0.02, 'USDT': 1}
    price_path = {
        tkn: (price for price in [final_price[tkn]])
        for tkn in initial_price
    }
    initial_liquidity = {'USDT': 1000000, 'DOT': 1000000, 'HDX': 100000000}
    default_liquidation_threshold = 0.7
    liquidation_bonus = {
        'DOT': 0.05,
        'HDX': 0.05,
        'USDT': 0.015,
    }
    default_ltv = 0.6
    assets = [
        MoneyMarketAsset(
            name=tkn,
            price=initial_price[tkn],
            liquidity=initial_liquidity[tkn],
            liquidation_bonus=liquidation_bonus[tkn],
            liquidation_threshold=default_liquidation_threshold,
            ltv=default_ltv,
        ) for tkn in initial_liquidity
    ]
    mm = MoneyMarket(
        assets=assets,
        oracle_source=lambda tkn: next(price_path[tkn]),
    )
    liquidator = Agent(
        trade_strategy=liquidate_cdps('omnipool'),
        enforce_holdings=False
    )
    initial_state = GlobalState(
        agents={'liquidator': liquidator},
        pools={'omnipool': omnipool, 'money_market': mm},
    )

    collateral = {'USDT': final_price['DOT'], 'DOT': 1}
    mm.prices = final_price
    hdx_debt_half_liquidation_point = mm.value_assets(collateral) / final_price['HDX'] * default_liquidation_threshold
    hdx_debt_full_liquidation_point = hdx_debt_half_liquidation_point / 0.95
    # liquidate half
    cdp1 = CDP(
        {'HDX': hdx_debt_full_liquidation_point - 0.00001},
        collateral
    )
    # liquidate all
    cdp2 = CDP(
        {'HDX': hdx_debt_full_liquidation_point + 0.00001},
        collateral
    )
    # liquidate half
    cdp3 = CDP(
        {'HDX': hdx_debt_half_liquidation_point + 0.00001},
        collateral
    )
    # liquidate none
    cdp4 = CDP(
        {'HDX': hdx_debt_half_liquidation_point - 0.00001},
        collateral
    )
    mm.prices = initial_price
    mm.cdps = [cdp1, cdp2, cdp3, cdp4]
    final_state = run.run(initial_state, 1)[-1]
    final_mm = final_state.pools['money_market']
    final_cdp1 = final_mm.cdps[0]
    final_cdp2 = final_mm.cdps[1]
    final_cdp3 = final_mm.cdps[2]
    final_cdp4 = final_mm.cdps[3]
    if final_cdp2.collateral['DOT'] > 0:
        raise ValueError("CDP2 DOT should be fully liquidated")
    if (final_mm.value_assets({'DOT': cdp1.collateral['DOT'] - final_cdp1.collateral['DOT']})
            / final_mm.value_assets({'HDX': final_cdp1.debt['HDX']}) - 1
            != pytest.approx(final_mm.get_liquidation_bonus('DOT', 'HDX'), rel=1e-12)
    ):
        raise ValueError("CDP1 DOT should be liquidated at half the value of HDX debt + liquidation bonus")
    if (final_mm.value_assets({'DOT': cdp3.collateral['DOT'] - final_cdp3.collateral['DOT']})
            / final_mm.value_assets({'HDX': final_cdp3.debt['HDX']}) - 1
            != pytest.approx(final_mm.get_liquidation_bonus('DOT', 'HDX'), rel=1e-12)
    ):
        raise ValueError("CDP3 should be half liquidated just like CDP1")
    if final_cdp4.debt != cdp4.debt or final_cdp4.collateral != cdp4.collateral:
        raise ValueError("CDP4 should not be liquidated")
    er = 1


def test_calculate_health_factor():
    assets = [
        MoneyMarketAsset(
            'USDC', 1, liquidation_threshold=0.9, emode_liquidation_threshold=0.93, emode_label="USD"
        ),
        MoneyMarketAsset(
            'USD', 1, liquidation_threshold=0.9, emode_liquidation_threshold=0.92, emode_label="USD"
        ),
        MoneyMarketAsset(
            'DAI', 0.9999, liquidation_threshold=0.9, emode_liquidation_threshold=0.92, emode_label="USD"
        ),
        MoneyMarketAsset(
            'ETH', 3000, liquidation_threshold=0.8, emode_liquidation_threshold=0.85, emode_label="ETH"
        ),
        MoneyMarketAsset(
            'WBTC', 80000, liquidation_threshold=0.7, emode_liquidation_threshold=0.8, emode_label="BTC"
        ),
        MoneyMarketAsset(
            'BTC', 80000, liquidation_threshold=0.75, emode_liquidation_threshold=0.85, emode_label="BTC"
        ),
    ]
    cdp1 = CDP(
        collateral={"USDC": 1897, "ETH": 1.05},
        debt={"DAI": 4000, "USD": 150},
        e_mode="USD"
    )
    cdp2 = CDP(
        collateral={"DAI": 80000, "WBTC": 0.5},
        debt={"BTC": 0.56, "USD": 10000},
        e_mode="BTC"
    )
    MoneyMarket(
        assets=assets,
        cdps=[cdp1, cdp2]
    )
    if cdp1.health_factor != pytest.approx(1.032439271, rel=1e-9):
        raise AssertionError("CDP 1 wrong health factor")
    elif cdp2.health_factor != pytest.approx(1.884358996, rel=1e-9):
        raise AssertionError("CDP 2 wrong health factor")
    print (f"CDP1: {cdp1.health_factor}")
    print(f"CDP2: {cdp2.liquidation_threshold}")


def test_save_load():
    mm = MoneyMarket(
        close_factor=0.55,
        full_liquidation_threshold=0.95,
        unique_id='test_mm',
        assets=[
            MoneyMarketAsset(
                name='ETH',
                liquidity=112,
                supply_cap=11,
                price=1000,
                ltv=.8,
                liquidation_bonus=0.1,
                liquidation_threshold=0.88,
                emode_label='ETH',
                emode_liquidation_bonus=0.05,
                emode_liquidation_threshold=0.91,
                emode_ltv=0.85
            ),
            MoneyMarketAsset(
                name='USD',
                liquidity=1122,
                supply_cap=115,
                price=1,
                ltv=.85,
                liquidation_bonus=0.04,
                liquidation_threshold=0.92,
                emode_label='stablecoin',
                emode_liquidation_bonus=0.02,
                emode_liquidation_threshold=0.95,
                emode_ltv=0.9
            )
        ],
        cdps=[
            CDP(
                collateral={'USD': 1000},
                debt={'ETH': 100}
            ),
            CDP(
                collateral={'ETH': 2000},
                debt={'USD': 150}
            )
        ]
    )
    filepath = find_test_directory()
    save_money_market(mm, path=filepath, filename='money_market_save_test')
    load_mm = load_money_market(path=filepath, filename='money_market_save_test')
    if (
        mm.asset_list != load_mm.asset_list
        or mm.full_liquidation_threshold != load_mm.full_liquidation_threshold
        or mm.partial_liquidation_pct != load_mm.partial_liquidation_pct
        or mm.time_step != load_mm.time_step
        or mm.unique_id != load_mm.unique_id
        or mm.borrowed != load_mm.borrowed
        or sum([
            1 if (mm.cdps[i].debt != load_mm.cdps[i].debt or mm.cdps[i].collateral != load_mm.cdps[i].collateral) else 0
            for i in range(len(mm.cdps))
        ]) != 0
    ):
        raise AssertionError('objects should be identical.')

    for name, asset in mm.assets.items():
        load_asset = load_mm.assets[name]
        if (
            asset.name != load_asset.name
            or asset.liquidity != load_asset.liquidity
            or asset.price != load_asset.price
            or asset.emode_label != load_asset.emode_label
            or asset.supply_cap != load_asset.supply_cap
            or asset.liquidation_threshold != load_asset.liquidation_threshold
            or asset.liquidation_bonus != load_asset.liquidation_bonus
            or asset.ltv != load_asset.ltv
            or asset.emode_ltv != load_asset.emode_ltv
            or asset.emode_liquidation_threshold != load_asset.emode_liquidation_threshold
            or asset.emode_liquidation_bonus != load_asset.emode_liquidation_bonus
        ):
            raise AssertionError('assets should be identical')
