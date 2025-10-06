import copy


class Agent:
    unique_id: str = ''

    def __init__(self,
                 holdings: dict[str: float] = None,
                 share_prices: dict[str: float] = None,
                 delta_r: dict[str: float] = None,
                 trade_strategy: any = None,
                 unique_id: str = 'agent',
                 nfts: dict[str: any] = None,
                 enforce_holdings: bool = True,
                 immune_to_fees: bool = False
                 ):
        """
        holdings should be in the form of:
        {
            asset_name: quantity
        }
        share_prices should be in the form of:
        {
            asset_name: price
        }
        The values of share_prices reflect the price at which those shares were acquired.
        If enforce_holdings is False, validate_holdings will always return True.
        """
        self.holdings = {tkn: val for tkn, val in holdings.items()} if holdings is not None else {}
        self.initial_holdings = {tkn: val for tkn, val in holdings.items()} if holdings is not None else {}
        self.share_prices = {k: val for k, val in share_prices.items()} if share_prices is not None else {}
        self.delta_r = {k: val for k, val in delta_r.items()} if delta_r is not None else {}
        self.trade_strategy = trade_strategy
        self.asset_list = list(self.holdings.keys())
        self.unique_id = unique_id
        self.nfts = nfts or {}
        self.enforce_holdings = enforce_holdings
        self.immune_to_fees = immune_to_fees

    def __repr__(self):
        precision = 10
        holdings = {tkn: round(self.holdings[tkn], precision) for tkn in self.holdings}
        share_prices = {tkn: round(self.share_prices[tkn], precision) for tkn in self.share_prices}
        return (
                f'Agent: {self.unique_id}\n'
                f'********************************\n'
                f'trade strategy: {self.trade_strategy.name if self.trade_strategy else "None"}\n' +
                f'holdings: (\n\n' +
                f'\n'.join(
                    [(
                            f'    *{tkn}*: {holdings[tkn]}\n' +
                            (f'    price: {share_prices[tkn]}\n' if tkn in share_prices else '')
                    ) for tkn in self.holdings]
                ) + ')\n')

    def copy(self):
        copy_self = Agent(
            holdings={k: v for k, v in self.holdings.items()},
            share_prices={k: v for k, v in self.share_prices.items()},
            delta_r={k: v for k, v in self.delta_r.items()},
            trade_strategy=self.trade_strategy,
            unique_id=self.unique_id,
            nfts={id: copy.deepcopy(nft) for id, nft in self.nfts.items()},
            enforce_holdings=self.enforce_holdings,
            immune_to_fees=self.immune_to_fees
        )
        copy_self.initial_holdings = {k: v for k, v in self.initial_holdings.items()}
        copy_self.asset_list = [tkn for tkn in self.asset_list]
        return copy_self

    def get_holdings(self, tkn) -> float:
        if tkn not in self.holdings:
            return 0
        return self.holdings[tkn]

    def get_initial_holdings(self, tkn) -> float:
        if tkn not in self.initial_holdings:
            return 0
        return self.initial_holdings[tkn]

    def all_holdings(self) -> dict[str, float]:
        return {k: v for k, v in self.holdings.items() if v != 0}

    def validate_holdings(self, tkn, amt=None) -> bool:
        if not self.enforce_holdings:
            return True
        if amt is None:
            return self.get_holdings(tkn) > 0
        else:
            return self.get_holdings(tkn) >= amt

    def add(self, tkn: str, amt: float) -> None:
        if tkn not in self.holdings:
            self.holdings[tkn] = 0
        self.holdings[tkn] += amt

    def remove(self, tkn: str, amt: float) -> None:
        if not self.enforce_holdings or self.validate_holdings(tkn, amt):
            if tkn not in self.holdings:
                self.holdings[tkn] = 0
            self.holdings[tkn] -= amt


class AgentArchiveState:
    def __init__(self, agent: Agent):
        self.unique_id = agent.unique_id
        self.holdings = {k: v for k, v in agent.holdings.items()}
        self.share_prices = {k: v for k, v in agent.share_prices.items()}
        self.asset_list = [tkn for tkn in agent.asset_list]
