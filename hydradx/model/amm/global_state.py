import copy
import random
from typing import Callable

from .agents import Agent
from .agents import AgentArchiveState
from .exchange import Exchange
from .money_market import MoneyMarket
from .omnipool_amm import OmnipoolState, simulate_swap
from .otc import OTC


class GlobalState:
    def __init__(self,
                 agents: dict[str, Agent] | list[Agent],
                 pools: dict[str, Exchange] | list[Exchange],
                 otcs: list[OTC] = None,
                 external_market: dict[str, float] = None,
                 evolve_function: Callable = None,
                 save_data: dict = None,
                 archive_all: bool = True,
                 ):
        self.external_market = external_market or {}
        if otcs is None:
            otcs = []
        if 'USD' not in self.external_market:
            self.external_market = {'USD': 1, **self.external_market}

        if isinstance(agents, list):
            self.agents = {agent.unique_id: agent for agent in agents}
        else:
            self.agents = agents
            for agent_name in self.agents:
                self.agents[agent_name].unique_id = agent_name
        # get a list of all assets contained in any member of the state
        if isinstance(pools, list):
            self.pools = {pool.unique_id: pool for pool in pools}
        else:
            self.pools = pools
            for pool_name in self.pools:
                self.pools[pool_name].unique_id = pool_name
        self.asset_list = list(set(
            [asset for pool in self.pools.values() for asset in pool.asset_list]
            + [asset for agent in self.agents.values() for asset in agent.asset_list]
            + list(self.external_market.keys())
        ))

        # for agent in self.agents.values():
        #     for asset in self.asset_list:
        #         if asset not in agent.holdings:
        #             agent.holdings[asset] = 0
        self.evolve_function = evolve_function
        self.datastreams = save_data
        self.save_data = {
            tag: save_data[tag].assemble(self)
            for tag in save_data
        } if save_data else {}
        self.time_step = 0
        self.archive_all = archive_all
        self.otcs = otcs

    def price(self, tkn: str, numeraire: str = 'USD') -> float:
        if tkn in self.external_market:
            return self.external_market[tkn]
        else:
            return 0

    def total_wealth(self):
        return sum([
            sum([
                agent.holdings[tkn] * self.price(tkn) for tkn in agent.holdings
            ])
            for agent in self.agents.values()
        ]) + sum([
            sum([
                pool.liquidity[tkn] * self.price(tkn) for tkn in pool.liquidity
            ])
            for pool in self.pools.values()
        ])

    def total_asset(self, tkn):
        return (
                sum([pool.liquidity[tkn] if tkn in pool.liquidity else 0 for pool in self.pools.values()])
                + sum([agent.holdings[tkn] if tkn in agent.holdings else 0 for agent in self.agents.values()])
        )

    def total_assets(self):
        return {tkn: self.total_asset(tkn) for tkn in self.asset_list}

    def copy(self):
        copy_state = GlobalState(
            agents={agent_id: self.agents[agent_id].copy() for agent_id in self.agents},
            pools={pool_id: self.pools[pool_id].copy() for pool_id in self.pools},
            otcs=[otc.copy() for otc in self.otcs],
            external_market=self.external_market.copy(),
            evolve_function=copy.copy(self.evolve_function),
            save_data=self.datastreams,
            archive_all=self.archive_all
        )
        copy_state.time_step = self.time_step
        return copy_state

    def archive(self):
        if self.archive_all and not self.save_data:
            return self.copy()
        elif self.save_data:
            return {
                datastream: self.save_data[datastream](self)
                for datastream in self.save_data
            }
        else:
            return ArchiveState(self)

    def evolve(self):
        self.time_step += 1
        for pool in self.pools.values():
            pool.update()
        if self.evolve_function:
            self.evolve_function(self)
        # agent actions
        agent_ids = list(self.agents.keys())
        for agent_id in agent_ids:
            agent = self.agents[agent_id]
            if agent.trade_strategy and self.time_step % agent.trade_strategy.frequency == 0:
                agent.trade_strategy.execute(self, agent_id)
        return self

    def execute_swap(
            self,
            pool_id: str,
            agent_id: str,
            tkn_sell: str,
            tkn_buy: str,
            buy_quantity: float = 0,
            sell_quantity: float = 0,
            **kwargs
    ):
        self.pools[pool_id].swap(
            agent=self.agents[agent_id],
            tkn_sell=tkn_sell,
            tkn_buy=tkn_buy,
            buy_quantity=buy_quantity,
            sell_quantity=sell_quantity,
            **kwargs  # pass any additional arguments to the pool
        )
        return self

    # functions for calculating extra parameters we may want to track

    def market_prices(self, shares: dict) -> dict:
        """
        return the market price of each asset in state.asset_list, as well as the price of each asset in shares
        """
        prices: dict = {tkn: self.price(tkn) for tkn in self.asset_list}
        for share_id in shares:
            # if shares are for a specific asset in a specific pool, get prices according to that pool
            if isinstance(share_id, tuple):
                pool_id = share_id[0]
                tkn_id = share_id[1]
                prices[share_id] = self.pools[pool_id].usd_price(self.pools[pool_id], tkn_id)

        return prices

    def cash_out(self, agent_id: str) -> float:
        """
        return the value of the agent's holdings if they withdraw all liquidity
        and then sell at current spot prices
        """
        agent = self.agents[agent_id]
        if 'LRNA' not in agent.holdings:
            agent.holdings['LRNA'] = 0

        withdraw_holdings = {tkn: agent.holdings[tkn] for tkn in list(agent.holdings.keys())}

        for key in agent.holdings.keys():
            # shares.keys might just be the pool name, or it might be a tuple (pool, token)
            if isinstance(key, tuple):
                pool_id = key[0]
                tkn = key[1]
            else:
                pool_id = key
                tkn = key
            if pool_id in self.pools:
                if isinstance(self.pools[pool_id], OmnipoolState):
                    # optimized for omnipool, no copy operations
                    delta_qa, delta_r, delta_q, \
                        delta_s, delta_b, delta_l = self.pools[pool_id].calculate_remove_liquidity(
                        agent,
                        agent.holdings[key],
                        tkn_remove=tkn
                    )
                    withdraw_holdings[key] = 0
                    withdraw_holdings['LRNA'] += delta_qa
                    withdraw_holdings[tkn] -= delta_r
                else:
                    # much less efficient, but works for any pool
                    new_state = self.copy()
                    new_pool: Exchange = new_state.pools[pool_id]
                    new_agent = new_state.agents[agent.unique_id]
                    new_pool.remove_liquidity(agent=new_agent, quantity=agent.holdings[key], tkn_remove=tkn)
                    withdraw_holdings = {
                        tkn: withdraw_holdings[tkn] + new_agent.holdings[tkn] - agent.holdings[tkn]
                        for tkn in agent.holdings
                    }

        prices = self.market_prices(withdraw_holdings)
        return value_assets(prices, withdraw_holdings)

    def pool_val(self, pool: Exchange):
        """ get the total value of all liquidity in the pool. """
        total = 0
        for asset in pool.asset_list:
            total += pool.liquidity[asset] * self.price(asset)
        return total

    def impermanent_loss(self, agent: Agent) -> float:
        return self.cash_out(agent.unique_id) / self.deposit_val(agent) - 1

    def deposit_val(self, agent: Agent) -> float:
        return value_assets(
            self.market_prices(agent.holdings),
            agent.initial_holdings
        )

    def withdraw_val(self, agent: Agent) -> float:
        return self.cash_out(agent.unique_id)

    def external_market_trade(
            self,
            agent_id: str,
            tkn_buy: str,
            tkn_sell: str,
            buy_quantity: float = 0,
            sell_quantity: float = 0
    ):
        # do a trade at spot price on the external market
        agent = self.agents[agent_id]
        if buy_quantity:
            sell_quantity = buy_quantity * self.price(tkn_buy) / self.price(tkn_sell)
        elif sell_quantity:
            buy_quantity = sell_quantity * self.price(tkn_sell) / self.price(tkn_buy)
        else:
            # raise TypeError('Must include either buy_quantity or sell_quantity.')
            return self

        if tkn_buy not in agent.holdings:
            agent.holdings[tkn_buy] = 0

        if agent.get_holdings(tkn_sell) - sell_quantity < 0:
            # insufficient funds, reduce quantity to match
            sell_quantity = agent.get_holdings(tkn_sell)
        elif agent.get_holdings(tkn_buy) + buy_quantity < 0:
            # also insufficient funds
            buy_quantity = -agent.get_holdings(tkn_buy)

        # there could probably be a fee or something here, but for now you can sell infinite quantities for free
        agent.add(tkn_buy, buy_quantity)
        agent.remove(tkn_sell, sell_quantity)

        return self

    def __repr__(self):
        newline = "\n"
        indent = '    '
        return (
                f'global state {newline}'
                f'pools: {newline + newline + indent}' +
                ((newline + indent).join([
                    (newline + indent).join(pool_desc.split('\n'))
                    for pool_desc in [repr(pool) for pool in self.pools.values()]
                ])) +
                newline + newline +
                f'agents: {newline + newline}    ' +
                ((newline + indent).join([
                    (newline + indent).join(agent_desc.split('\n'))
                    for agent_desc in [repr(agent) for agent in self.agents.values()]
                ])) + newline +
                f'market prices: {newline + newline}    ' +
                ((newline + indent).join([
                    f'{indent}{tkn}: ${price}' for tkn, price in self.external_market.items()
                ])) +
                f'{newline}{newline}'
                f'evolution function: {self.evolve_function.__name__ if self.evolve_function else "None"}'
                f'{newline}'
        )


class ArchiveState:
    def __init__(self, state: GlobalState):
        self.time_step = state.time_step
        self.external_market = copy.copy(state.external_market)
        self.pools = {k: v.archive() for (k, v) in state.pools.items()}
        self.agents = {k: AgentArchiveState(v) for (k, v) in state.agents.items()}


def value_assets(prices: dict, assets: dict) -> float:
    """
    return the value of the agent's assets if they were sold at current spot prices
    """
    return sum([
        assets[i] * prices[i] if i in prices else 0
        for i in assets.keys()
    ])

def fluctuate_prices(volatility: dict[str: float], trend: dict[str: float] = None):
    """
        Volatility is in the form of
        {tkn: percentage for tkn in asset_list}
        and is the maximum percentage by which the asset price will vary per step.

        Trend is also in the form
        {tkn: percentage for tkn in asset_list}
        and is the percentage by the which the price will predictably move, every step.

        Approximately:
        price[tkn] += random.random() * volatility[tkn] / 100 + trend[tkn] / 100
    """
    trend = trend or {}

    def transform(state: GlobalState) -> GlobalState:
        new_state = state  # .copy()
        for asset in new_state.external_market:
            change = volatility[asset] if asset in volatility else 0
            bias = trend[asset] / 100 if asset in trend else 0
            # not exactly the same as above, because adding 1% and then subtracting 1% does not get us back to 100%
            # instead, we need to subtract (100/101)% to avoid a long-term downward trend
            new_state.external_market[asset] *= (
                    (1 + random.random() * change / 100 if random.choice([True, False])
                     else 1 - random.random() * (1 - 100 / (100 + change)))
                    # 1 / (1 + change / 100)
                    # + random.random() * (1 - 1 / (1 + change / 100) + change / 100)
                    + bias / 100
            )
        return new_state

    return transform


def oscillate_prices(volatility: dict[str: float], trend: dict[str: float] = None, period: int = 1) -> Callable:
    # steadily oscillate, no unpredictable motion
    class UpDown:
        def __init__(self, magnitude, wavelength, bias):
            self.bias = bias
            self.inertia = 0
            self.wavelength = wavelength
            self.direction = magnitude

    trend = trend or {}
    updown: dict[str: UpDown] = {}
    for token in volatility:
        updown[token] = UpDown(
            wavelength=period,
            magnitude=volatility[token],
            bias=trend[token] if token in trend else 0
        )

    def transform(state: GlobalState) -> GlobalState:
        for tkn in updown:
            if abs(updown[tkn].inertia) >= updown[tkn].wavelength:
                # reverse trend
                updown[tkn].direction = (updown[tkn].direction + 1) * -1 + 1
                updown[tkn].inertia = 0
            state.external_market[tkn] += (
                    updown[tkn].direction / 100 / updown[tkn].wavelength
                    + updown[tkn].bias / 100 / updown[tkn].wavelength
            )
            updown[tkn].inertia += 1
        return state

    return transform


def historical_prices(price_list: list[dict[str, float]]) -> Callable:
    def transform(state: GlobalState) -> GlobalState:
        for tkn in price_list[state.time_step]:
            state.external_market[tkn] = price_list[state.time_step][tkn]
        return state

    return transform


def settle_otc_against_omnipool(pool_id: str, agent_id: str):
    def transform(state: GlobalState) -> GlobalState:
        omnipool = state.pools[pool_id]
        agent = state.agents[agent_id]
        otcs_to_remove = []
        for otc in state.otcs:
            sell_amt = find_partial_otc_sell_amount(omnipool, otc)
            if sell_amt > 0:
                omnipool = state.pools[pool_id]
                omnipool_settle_otc(omnipool, otc, agent, sell_amt)
                if otc.sell_amount == 0:
                    otcs_to_remove.append(otc)
        for otc in otcs_to_remove:
            state.otcs.remove(otc)

        return state

    return transform


def find_partial_otc_sell_amount(omnipool, otc):
    buy_asset = otc.buy_asset
    sell_asset = otc.sell_asset

    # if no arbitrage can happen at spot price, we cannot even partially satisfy the OTC order
    if omnipool.price(buy_asset, sell_asset) > otc.price:
        return 0

    sell_amt = otc.sell_amount
    sell_amt_up = sell_amt
    sell_amt_down = 0
    agent = Agent(holdings={buy_asset: 0, sell_asset: sell_amt})

    iters = 20
    if not otc.partially_fillable:
        iters = 1

    # binary search
    for i in range(iters):
        temp_state, temp_agent = simulate_swap(omnipool, agent, tkn_buy=buy_asset,
                                               tkn_sell=sell_asset, sell_quantity=sell_amt)
        amt_bought = agent.holdings[buy_asset] - temp_agent.holdings[buy_asset]
        spot_after = temp_state.buy_spot(buy_asset, sell_asset)
        # execution_price = amt_sold / buy_amt if buy_amt != 0 else float('inf')

        if spot_after > otc.price or amt_bought == 0:  # trade amount too high
            sell_amt_up = sell_amt
        else:  # trade amt can be executed
            sell_amt_down = sell_amt
            if sell_amt == sell_amt_up:
                break
        sell_amt = (sell_amt_up + sell_amt_down) / 2
    return sell_amt_down


def omnipool_settle_otc(omnipool: OmnipoolState, otc: OTC, treasury_agent: Agent, buy_from_otc_amt: float) -> None:
    # flash mint otc.sell_asset to treasury_agent
    treasury_agent.holdings[otc.sell_asset] += buy_from_otc_amt
    omnipool.swap(treasury_agent, tkn_buy=otc.buy_asset, tkn_sell=otc.sell_asset,
                  sell_quantity=buy_from_otc_amt)
    otc.buy(treasury_agent, buy_from_otc_amt)
    # burn assets that were flash minted
    treasury_agent.holdings[otc.sell_asset] -= buy_from_otc_amt
