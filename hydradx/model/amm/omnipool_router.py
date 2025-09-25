import copy

from .agents import Agent
from .exchange import Exchange
from .stableswap_amm import StableSwapPoolState
from typing import Literal


class Trade:
    def __init__(
            self,
            exchange: str,
            tkn_sell: str,
            tkn_buy: str,
            buy_quantity: float = None,
            sell_quantity: float = None
    ):
        self.exchange = exchange
        self.tkn_sell = tkn_sell
        self.tkn_buy = tkn_buy
        self.buy_quantity = buy_quantity
        self.sell_quantity = sell_quantity


class OmnipoolRouter(Exchange):
    """
    Handles routing between Omnipool and Stableswap subpools that have LP assets in Omnipool
    """

    def __init__(self, exchanges: dict or list, unique_id: str = 'omnipool_router'):
        super().__init__()
        self.exchanges = exchanges if type(exchanges) == dict else {ex.unique_id: ex for ex in exchanges}
        self.asset_list = list(set([tkn for exchange in self.exchanges.values() for tkn in exchange.asset_list]))
        self.fail = ''
        self.unique_id = unique_id

    def copy(self):
        copy_self = copy.deepcopy(self)
        return copy_self

    def buy_limit(self, tkn_buy, tkn_sell=None):
        return sum([exchange.buy_limit(tkn_buy, tkn_sell) for exchange in self.exchanges.values()])

    def sell_limit(self, tkn_buy, tkn_sell):
        return float('inf')

    @property
    def liquidity(self):
        return {
            tkn: sum([exchange.liquidity[tkn] if tkn in exchange.liquidity else 0 for exchange in self.exchanges.values()])
            for tkn in [tkn for exchange in self.exchanges.values() for tkn in exchange.asset_list]
        }

    def buy_spot(self, tkn_buy: str, tkn_sell: str, fee: float = None):
        return self.price_route(self.find_best_route(tkn_buy, tkn_sell, buy_quantity=1), direction='buy', fee=fee)

    def sell_spot(self, tkn_sell: str, tkn_buy: str, fee: float = None):
        return self.price_route(self.find_best_route(tkn_buy, tkn_sell, sell_quantity=1), direction='sell', fee=fee)

    def price(self, tkn: str, numeraire: str = ''):
        """
        Price of tkn in terms of numéraire (feeless)
        """
        if numeraire and numeraire not in self.asset_list:
            raise ValueError(f'Denomination {numeraire} not in exchange')
        if tkn == numeraire:
            return 1.0
        if not numeraire:
            # default to price in USD
            numeraire = sorted([t for t in self.asset_list if t.startswith('USD')])[0]

        return self.buy_spot(tkn_buy=numeraire, tkn_sell=tkn, fee=0)

    def calculate_buy_from_sell(self, tkn_sell: str, tkn_buy: str, sell_quantity: float, route: list[Trade] = None):
        """
        Calculates how much tkn_buy you would get for selling sell_quantity of tkn_sell
        If route is None, all routes are calculated and the best one is returned
        Returns quantity of tkn_buy
        """
        if route is None:
            route = self.find_best_route(tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=sell_quantity)
        quantity = sell_quantity
        for trade in route:
            quantity = self.calculate_trade(trade, sell_quantity=quantity)
        return quantity

    def calculate_sell_from_buy(self, tkn_buy: str, tkn_sell: str, buy_quantity: float, route: list[Trade] = None):
        if route is None:
            route = self.find_best_route(tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_quantity)
        quantity = buy_quantity
        for trade in route:
            quantity = self.calculate_trade(trade, buy_quantity=quantity)
        return quantity

    def fail_transaction(self, fail_message: str):
        self.fail = fail_message
        return self

    def validate_route(
        self,
        route: list[Trade]
    ):
        """Checks that the specified route is valid"""
        for i in range(len(route) - 1):
            exchange = self.exchanges[route[i].exchange]
            if route[i].tkn_buy != route[i + 1].tkn_sell:
                raise ValueError(f'Invalid route: {route[i].tkn_buy} != {route[i + 1].tkn_sell}')
            if exchange.unique_id not in self.exchanges:
                raise ValueError(f'Exchange {exchange.unique_id} not in router exchanges')
            if route[i].tkn_buy not in exchange.asset_list and route[i].tkn_buy != exchange.unique_id:
                raise ValueError(f'Token {route[i].tkn_buy} not in exchange {exchange.unique_id} asset list')
            if route[i].tkn_sell not in exchange.asset_list and route[i].tkn_sell != exchange.unique_id:
                raise ValueError(f'Token {route[i].tkn_sell} not in exchange {exchange.unique_id} asset list')
        return self

    def price_route(
            self,
            route: list[Trade],
            direction: Literal['buy', 'sell'],
            fee: float = None
    ) -> float:
        if len(route) == 0:
            return 0.0
        price = 1.0
        for trade in route:
            price *= self.price_trade(trade, direction, fee=fee)
        return price

    def swap_route(
        self,
        agent: Agent,
        route: list[Trade],
        buy_quantity: float = None,
        sell_quantity: float = None
    ):
        """Does swap along specified route"""
        for swap in route:
            exchange = self.exchanges[swap.exchange]
            if buy_quantity:
                buy_quantity = self.execute_trade(
                    agent,
                    swap,
                    buy_quantity=buy_quantity,
                )
            else:
                sell_quantity = self.execute_trade(
                    agent,
                    swap,
                    sell_quantity=sell_quantity
                )
            if exchange.fail:
                return self.fail_transaction(exchange.fail)
        return self

    def get_exchanges(self, tkn) -> list[Exchange]:
        """Returns list of exchanges that have tkn in their asset list"""
        return [exchange for exchange in self.exchanges.values() if tkn in exchange.asset_list or exchange.unique_id == tkn]

    def find_routes(self, tkn_buy: str, tkn_sell: str, direction: Literal['buy', 'sell']) -> list[list[Trade]]:
        """
        Finds all possible routes to swap between tkn_buy and tkn_sell
        """
        routes: list[list[Trade]] = []
        tkn_buy_pools = self.get_exchanges(tkn_buy)
        tkn_sell_pools = self.get_exchanges(tkn_sell)

        if len(tkn_buy_pools) == 0:
            raise ValueError(f'No pool with {tkn_buy} in asset list')
        if len(tkn_sell_pools) == 0:
            raise ValueError(f'No pool with {tkn_sell} in asset list')

        for intermediate_pool in list(set(tkn_buy_pools) & set(tkn_sell_pools)):
            routes.append([Trade(
                exchange=intermediate_pool.unique_id,
                tkn_sell=tkn_sell,
                tkn_buy=tkn_buy
            )])
        # options for two pool routes
        for sell_pool, buy_pool in [
            (sell_pool, buy_pool)
            for buy_pool in tkn_buy_pools for sell_pool in tkn_sell_pools
            if sell_pool != buy_pool
        ]:
            if sell_pool.unique_id in buy_pool.asset_list and tkn_sell != sell_pool.unique_id:
                routes.append([
                    Trade(
                        exchange=sell_pool.unique_id,
                        tkn_sell=tkn_sell,
                        tkn_buy=sell_pool.unique_id
                    ), Trade(
                        exchange=buy_pool.unique_id,
                        tkn_sell=sell_pool.unique_id,
                        tkn_buy=tkn_buy
                    )
                ])
            elif buy_pool.unique_id in sell_pool.asset_list and tkn_buy != buy_pool.unique_id:
                routes.append([
                    Trade(
                        exchange=sell_pool.unique_id,
                        tkn_sell=tkn_sell,
                        tkn_buy=buy_pool.unique_id
                    ), Trade(
                        exchange=buy_pool.unique_id,
                        tkn_sell=buy_pool.unique_id,
                        tkn_buy=tkn_buy
                    )
                ])
            else:
                intermediaries = set(sell_pool.asset_list) & set(buy_pool.asset_list) - {tkn_buy, tkn_sell}
                if len(intermediaries) == 0:
                    continue
                best_tkn = min(intermediaries, key=lambda x: sell_pool.price(tkn_sell, x) / buy_pool.price(tkn_buy, x))
                routes.append([
                    Trade(
                        exchange=sell_pool.unique_id,
                        tkn_sell=tkn_sell,
                        tkn_buy=best_tkn
                    ), Trade(
                        exchange=buy_pool.unique_id,
                        tkn_sell=best_tkn,
                        tkn_buy=tkn_buy
                    )
                ])

        # possible three-way routes
        for intermediate_pool in self.exchanges.values():
            for sell_pool in tkn_sell_pools:
                for buy_pool in tkn_buy_pools:
                    if sell_pool == buy_pool or sell_pool == intermediate_pool or buy_pool == intermediate_pool:
                        continue
                    intermediate_sell_tkns = (
                        set(sell_pool.asset_list + [sell_pool.unique_id]) & set(intermediate_pool.asset_list)
                        - {tkn_buy, tkn_sell}
                    )
                    intermediate_buy_tkns = (
                        set(buy_pool.asset_list + [buy_pool.unique_id]) & set(intermediate_pool.asset_list)
                        - {tkn_buy, tkn_sell} - intermediate_sell_tkns
                    )
                    if len(intermediate_sell_tkns) == 0 or len(intermediate_buy_tkns) == 0:
                        continue
                    for intermediate_sell_tkn in intermediate_sell_tkns:
                        for intermediate_buy_tkn in intermediate_buy_tkns:
                            routes.append([
                                Trade(
                                    exchange=sell_pool.unique_id,
                                    tkn_sell=tkn_sell,
                                    tkn_buy=intermediate_sell_tkn
                                ), Trade(
                                    exchange=intermediate_pool.unique_id,
                                    tkn_sell=intermediate_sell_tkn,
                                    tkn_buy=intermediate_buy_tkn
                                ), Trade(
                                    exchange=buy_pool.unique_id,
                                    tkn_sell=intermediate_buy_tkn,
                                    tkn_buy=tkn_buy
                                )
                            ])

        if direction == 'buy':
            for route in routes:
                route.reverse()

        return routes

    def find_best_route(self, tkn_buy, tkn_sell, buy_quantity: float = 0, sell_quantity: float = 0) -> list[Trade]:
        """
        Finds route to swap between tkn_buy and tkn_sell with the lowest spot price
        Returns tuple in the order of (sell_pool_id, buy_pool_id)
        """
        routes = self.find_routes(tkn_buy, tkn_sell, direction = 'buy' if buy_quantity else 'sell')
        if len(routes) == 0:
            # raise ValueError(f'No route found for {tkn_buy} to {tkn_sell}')
            return []
        if buy_quantity:
            # pay the least
            return min(
                routes, key=lambda x: self.calculate_sell_from_buy(
                    tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=buy_quantity, route=x
                )
            )
        else:
            # receive the most
            return max(
                routes, key=lambda x: self.calculate_buy_from_sell(
                    tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=sell_quantity, route=x
                )
            )

    def swap(self, agent, tkn_buy, tkn_sell, buy_quantity: float = None, sell_quantity: float = None):
        """Does swap along whatever route has best spot price"""
        route = self.find_best_route(tkn_buy, tkn_sell, sell_quantity=sell_quantity, buy_quantity=buy_quantity)
        return self.swap_route(
            agent=agent,
            route=route,
            buy_quantity=buy_quantity,
            sell_quantity=sell_quantity
        )

    def simulate_swap_route(self,
        agent: Agent,
        route: list[Trade],
        buy_quantity: float = None,
        sell_quantity: float = None
    ):
        """Does swap along specified route, returning new router and agent"""
        new_state = self.copy()
        new_agent = agent.copy()
        new_state.swap_route(new_agent, route, buy_quantity, sell_quantity)
        return new_state, new_agent

    def simulate_swap(self, agent, tkn_buy, tkn_sell, buy_quantity=0, sell_quantity=0):
        """Does swap along whatever route has best spot price"""
        new_state = self.copy()
        new_agent = agent.copy()
        new_state.swap(
            agent=new_agent,
            tkn_buy=tkn_buy,
            tkn_sell=tkn_sell,
            buy_quantity=buy_quantity,
            sell_quantity=sell_quantity
        )
        return new_state, new_agent

    def price_trade(self, trade: Trade, direction: Literal['buy', 'sell'], fee: float = None) -> float:
        exchange = self.exchanges[trade.exchange]
        tkn_buy = trade.tkn_buy
        tkn_sell = trade.tkn_sell
        if tkn_sell == exchange.unique_id and isinstance(exchange, StableSwapPoolState):
            if direction == "buy":
                return 1 / exchange.withdraw_asset_spot(tkn_remove=tkn_buy, fee=fee)
            else:
                return exchange.remove_liquidity_spot(tkn_remove=tkn_buy, fee=fee)
        elif tkn_buy == exchange.unique_id and isinstance(exchange, StableSwapPoolState):
            if direction == "buy":
                return exchange.buy_shares_spot(tkn_add=tkn_sell)
            else:
                return 1 / exchange.add_liquidity_spot(tkn_add=tkn_sell, fee=fee)
        else:
            if direction == "buy":
                return exchange.buy_spot(tkn_buy=tkn_buy, tkn_sell=tkn_sell, fee=fee)
            else:
                return exchange.sell_spot(tkn_sell=tkn_sell, tkn_buy=tkn_buy, fee=fee)

    def execute_trade(self, agent: Agent, trade: Trade, buy_quantity: float=None, sell_quantity: float=None) -> float:
        trade_type = "buy" if buy_quantity is not None else "sell"
        quantity = buy_quantity if buy_quantity is not None else sell_quantity
        exchange = self.exchanges[trade.exchange]
        tkn_buy = trade.tkn_buy
        tkn_sell = trade.tkn_sell
        start_quantity = agent.get_holdings(tkn_buy) if trade_type == "sell" else agent.get_holdings(tkn_sell)
        if tkn_sell == exchange.unique_id and isinstance(exchange, StableSwapPoolState):
            if trade_type == "buy":
                exchange.withdraw_asset(
                    agent=agent,
                    quantity=quantity,
                    tkn_remove=tkn_buy
                )
            else:
                exchange.remove_liquidity(
                    agent=agent,
                    shares_removed=quantity,
                    tkn_remove=tkn_buy
                )
        elif tkn_buy == exchange.unique_id and isinstance(exchange, StableSwapPoolState):
            if trade_type == "buy":
                exchange.buy_shares(
                    agent=agent,
                    quantity=quantity,
                    tkn_add=tkn_sell
                )
            else:
                exchange.add_liquidity(
                    agent=agent,
                    quantity=quantity,
                    tkn_add=tkn_sell
                )
        else:
            exchange.swap(
                agent=agent,
                tkn_buy=tkn_buy, tkn_sell=tkn_sell,
                buy_quantity=buy_quantity, sell_quantity=sell_quantity
            )

        if trade_type == "sell":
            return agent.get_holdings(tkn_buy) - start_quantity
        else:
            return -agent.get_holdings(tkn_sell) + start_quantity

    def calculate_trade(self, trade: Trade, buy_quantity: float=None, sell_quantity: float=None) -> float:
        quantity = buy_quantity if buy_quantity is not None else sell_quantity
        trade_type = "buy" if buy_quantity is not None else "sell"
        exchange = self.exchanges[trade.exchange]
        tkn_buy = trade.tkn_buy
        tkn_sell = trade.tkn_sell
        if tkn_sell == exchange.unique_id and isinstance(exchange, StableSwapPoolState):
            if trade_type == "buy":
                return exchange.calculate_withdraw_asset(tkn_remove=tkn_buy, quantity=quantity)
            else:
                return exchange.calculate_remove_liquidity(tkn_remove=tkn_buy, quantity=quantity)
        elif tkn_buy == exchange.unique_id and isinstance(exchange, StableSwapPoolState):
            if trade_type == "buy":
                return exchange.calculate_buy_shares(tkn_add=tkn_sell, quantity=quantity)
            else:
                return exchange.calculate_add_liquidity(tkn_add=tkn_sell, quantity=quantity)
        else:
            if trade_type == "buy":
                return exchange.calculate_sell_from_buy(tkn_sell=tkn_sell, tkn_buy=tkn_buy, buy_quantity=quantity)
            else:
                return exchange.calculate_buy_from_sell(tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=quantity)
