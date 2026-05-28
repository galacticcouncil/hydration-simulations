# README

The main HydraDX model can be found in "hydradx".

Installation: 
* Install Python 3.10 or higher
* Clone the repository and navigate to the root folder, HydraDx-simulations
* In terminal, enter 'pip install -r requirements.txt'
* Alternatively, open the project folder in PyCharm, and you'll be prompted to create a virtual environment for this project. 
* If you want to pull data from SQLPad, add a .env file to ./hydradx/model/ with the following contents:
```
SQLPAD_USERNAME=<your username>
PASSWORD=<your password>
```

This codebase contains simulations for multiple exchange types, including Omnipool, centralized markets, stableswaps, OmnipoolRouter, money market, etc. All these extend the Exchange class, which defines common methods such as swap, price, etc. They can be found in the hydraDX/model/amm folder.

The OmnipoolRouter class in particular contains its own list of exchanges and its swap function will automatically find the cheapest route to swap one token for another. Use model/indexer utils/get_current_omnipool_router(block_number) to download the state at a particular point in time. Use model/processing.save/load_state to serialize this into a file for faster access and to load from that file. 

MoneyMarket class has similar functionality: get_current_money_market in indexer utils and save/load_money_market in processing.

The other most important class is the Agent, which holds assets and executes TradeStrategies.

These can be combined into a GlobalState class, which is the most straightforward way to run a simulation.

A TradeStrategy mutates a GobalState object and is designed to run once per simulation step. They can be combined using the + operator. 

The basic process: define a GlobalState with exchanges, agents and an optional evolution function. (Some evolution functions are defined in hydraDX/model/amm/global_state; in general, they consist of a function that takes some parameters and return a function that accepts a GobalState object and mutates it according to those parameters.) Run.run(state, time_steps) then runs the simulation for the specified number of steps. At each step, each exchange’s update function, each agent’s trade strategy and the state’s evolution function all run. The output is an array of GlobalState objects, each representing the state at a particular time step. 

Example:
This codebase contains simulations for multiple exchange types, including Omnipool, centralized markets, stableswaps, OmnipoolRouter, money market, etc. All these extend the Exchange class, which defines common methods such as swap, price, etc. They can be found in the hydraDX/model/amm folder.

The OmnipoolRouter class in particular, contains its own list of exchanges and its swap function will automatically find the cheapest route to swap one token for another. Use model/indexer utils/get_current_omnipool_router(block_number) to download the state at a particular point in time. Use model/processing.save/load_state to serialize this into a file for faster access and to load from that file. 

MoneyMarket class has similar functionality: get_current_money_market in indexer utils and save/load_money_market in processing.

The other most important class is the Agent, which holds assets and executes TradeStrategies.

These can be combined into a GlobalState class, which is the most straightforward way to run a simulation.

A TradeStrategy mutates a GobalState object and is designed to run once per simulation step. They can be combined using the + operator. 

The basic process: define a GlobalState with exchanges, agents and an optional evolution function. (Some evolution functions are defined in hydraDX/model/amm/global_state; in general, they consist of a function that takes some parameters and return a function that accepts a GobalState object and mutates it according to those parameters.) Run.run(state, time_steps) then runs the simulation for the specified number of steps. At each step, each exchange’s update function, each agent’s trade strategy and the state’s evolution function all run. The output is an array of GlobalState objects, each representing the state at a particular time step. 

Example:

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
