from .exchange import Exchange
from .agents import Agent

class FixedPriceExchange(Exchange):
    def __init__(
            self,
            tokens: dict[str: float],
            fee: float = 0.0,
            unique_id='dummy exchange'
    ):
        """
        Mock exchange with infinite liquidity and no fees
        """
        super().__init__()
        self.prices = tokens
        self.liquidity = {tkn: 0 for tkn in tokens}
        self.asset_list = list(tokens.keys())
        self.fee = fee
        self.unique_id = unique_id

    def price(self, tkn: str, numeraire: str = ''):
        if numeraire and numeraire not in self.prices:
            raise ValueError(f'Denomination {numeraire} not in exchange')
        return self.prices[tkn] / (self.prices[numeraire] if numeraire in self.prices else 1)

    def buy_spot(self, tkn_buy, tkn_sell, fee=0):
        return self.price(tkn_buy) / self.price(tkn_sell) / (1 - self.fee)

    def sell_spot(self, tkn_sell, tkn_buy, fee=0):
        return self.price(tkn_sell) / self.price(tkn_buy) * (1 - self.fee)

    def buy_limit(self, tkn_buy, tkn_sell):
        return float('inf')

    def sell_limit(self, tkn_buy, tkn_sell):
        return float('inf')

    def swap(
            self,
            agent: Agent,
            tkn_buy: str,
            tkn_sell: str,
            buy_quantity: float = 0,
            sell_quantity: float = 0
    ):
        if buy_quantity:
            sell_quantity = self.calculate_sell_from_buy(tkn_sell=tkn_sell, tkn_buy=tkn_buy, buy_quantity=buy_quantity)
        elif sell_quantity:
            buy_quantity = self.calculate_buy_from_sell(tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=sell_quantity)

        agent.remove(tkn_sell, sell_quantity)
        agent.add(tkn_buy, buy_quantity)
        self.liquidity[tkn_sell] += sell_quantity
        self.liquidity[tkn_buy] -= buy_quantity
        return self

    def calculate_buy_from_sell(self, tkn_buy, tkn_sell, sell_quantity):
        return sell_quantity * self.price(tkn_sell) / self.price(tkn_buy) * (1 - self.fee)

    def calculate_sell_from_buy(self, tkn_sell, tkn_buy, buy_quantity):
        return buy_quantity * self.price(tkn_buy) / self.price(tkn_sell) / (1 - self.fee)
