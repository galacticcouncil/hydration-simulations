from hypothesis import given

from hydradx.model.amm.global_state import value_assets, GlobalState
from hydradx.model.amm.trade_strategies import omnipool_arbitrage, invest_all, withdraw_all
from hydradx.tests.strategies_omnipool import reasonable_market_dict, reasonable_holdings


# def value_assets(prices: dict, assets: dict) -> float:
@given(reasonable_market_dict(token_count=5), reasonable_holdings(token_count=5))
def test_value_assets(market: dict, holdings: list):
    asset_list = list(market.keys())
    assets = {asset_list[i]: holdings[i] for i in range(5)}
    value = value_assets(market, assets)
    if value != sum([holdings[i] * market[asset_list[i]] for i in range(5)]):
        raise


def test_adding_usd_to_external_market():
    state = GlobalState(agents={}, pools={}, external_market={'USD': 1, 'DOT': 10})
    state2 = GlobalState(agents={}, pools={}, external_market={'DOT': 10})
    assert state.external_market == state2.external_market


def test_example():
    from hydradx.model.amm.agents import Agent
    from hydradx.model.amm.omnipool_amm import OmnipoolState
    from hydradx.model.amm.global_state import GlobalState, fluctuate_prices
    from hydradx.model.run import run
    from pytest import approx

    state = GlobalState(
        agents=[
            Agent(
                holdings={'USD': 100},
                enforce_holdings=False,
                trade_strategy=(
                    invest_all(pool_id='omnipool', when=0)
                    + omnipool_arbitrage(pool_id='omnipool', arb_precision=2)
                    + withdraw_all(when=9)
                ),
                unique_id='agent1'
            )
        ],
        pools=[
            OmnipoolState(
                tokens={
                    'USD': {'liquidity': 1000, 'LRNA': 50},
                    'DOT': {'liquidity': 200, 'LRNA': 50},
                    'HDX': {'liquidity': 100000, 'LRNA': 50}
                },
                lrna_fee=0, asset_fee=0, slip_factor=0, unique_id='omnipool'
            )
        ],
        external_market={'USD': 1, 'DOT': 5, 'HDX': 0.02},
        update_function=fluctuate_prices(volatility={'HDX': 1}, trend={'HDX': 0.02})
    )
    events = run(state, time_steps=10)
    final_state = events[-1]
    agent1 = final_state.agents['agent1']
    omnipool = final_state.pools['omnipool']
    # agent should be buying HDX
    assert agent1.get_holdings('HDX') > agent1.get_initial_holdings('HDX')
    # price of HDX in omnipool should be consistent with external market
    assert omnipool.price('HDX', 'USD') == approx(final_state.price('HDX'), rel=1e-8)