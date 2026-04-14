import copy

from .agents import Agent
from .exchange import Exchange


class StableSwapPoolState(Exchange):
    unique_id: str = 'stableswap'
    peg: list[float] = []

    def __init__(
            self,
            tokens: dict,
            amplification: float,
            precision: float = 0.0001,
            trade_fee: float = 0.0,
            unique_id: str = '',
            spot_price_precision: float = 1e-07,
            shares: float = None,
            peg: float or list = None,
            peg_target: float or list = None,
            max_peg_update: float = float('inf')
    ):
        """
        Tokens should be in the form of:
        {
            token1: quantity,
            token2: quantity
        }
        There can be up to five.
        """
        super().__init__()
        if len(tokens.keys()) > 5:
            raise ValueError('Too many tokens (limit 5)')

        self.amplification = amplification
        self.amp_change_step = 0
        self.target_amp_block = 0
        self.time_step = 0
        self.peg_target_updated_at = 0
        self.precision = precision
        self.spot_price_precision = spot_price_precision
        self.liquidity = dict()
        self.asset_list: list[str] = []
        self.trade_fee = trade_fee
        if unique_id:
            self.unique_id = unique_id

        for token, quantity in tokens.items():
            self.asset_list.append(token)
            self.liquidity[token] = quantity

        self.n_coins = len(self.asset_list)

        self.set_peg(peg)
        self.set_peg_target(peg_target)
        self.max_peg_update = max_peg_update

        self.shares = self.calculate_d() if shares is None else shares
        self.conversion_metrics = {}

    @property
    def d(self) -> float:
        return self.calculate_d()

    @property
    def ann(self) -> float:
        return self.amplification * self.n_coins

    def fail_transaction(self, error: str, **kwargs):
        self.fail = error
        return self

    def update(self):
        self.time_step += 1
        if self.target_amp_block >= self.time_step:
            self.amplification += self.amp_change_step

    def set_peg(self, peg=None):
        if peg is None:
            self.peg = [1] * len(self.asset_list)
        elif isinstance(peg, (int, float)):  # peg is price of second asset denominated in first asset
            assert len(self.asset_list) == 2
            self.peg = [1, peg]
        else:  # peg is list of prices of all assets except the first, denominate in first asset
            assert len(peg) == len(self.asset_list) - 1
            self.peg = [1] + peg

    def set_peg_target(self, peg_target=None):
        if peg_target is None:
            self.peg_target = [p for p in self.peg]
        elif isinstance(peg_target, (int, float)):
            assert len(self.asset_list) == 2
            self.peg_target = [1, peg_target]
        else:
            assert len(peg_target) == len(self.asset_list) - 1
            self.peg_target = [1] + peg_target

        assert len(self.peg) == len(self.peg_target)
        self.peg_target_updated_at = self.time_step

    def set_amplification(self, amplification: float, duration: float):
        self.target_amp_block = self.time_step + duration
        self.amp_change_step = (amplification - self.amplification) / duration

    def get_adjusted_liquidity(self, balances=None):
        if balances is None:
            balances = self.liquidity
        adjusted_liquidity = {}
        for tkn in balances:
            i = self.asset_list.index(tkn)
            adjusted_liquidity[tkn] = balances[tkn] * self.peg[i]
        return adjusted_liquidity

    def calculate_d(self, reserves=(), max_iterations=128, peg_deltas=None) -> float:
        if not reserves:
            reserves = list(self.liquidity.values())
        n = self.n_coins
        if peg_deltas is None:
            peg = [self.peg[i] for i in range(n)]
        else:
            peg = [self.peg[i] + peg_deltas[i] for i in range(n)]
        xp_sorted = sorted([reserves[i] * peg[i] for i in range(n)])
        s = sum(xp_sorted)
        if s == 0:
            return 0

        d = s
        for i in range(max_iterations):

            d_p = d
            for x in xp_sorted:
                d_p *= d / (x * n)

            d_prev = d
            d = (self.ann * s + d_p * n) * d / ((self.ann - 1) * d + (n + 1) * d_p)

            if (d <= d_prev and d_prev - d < self.precision) or (d > d_prev and d - d_prev <= self.precision):
                return d
        return d

    def calculate_y(self, reserves: dict, d: float, max_iterations=128):
        """
        Given a value for D and the balances of all tokens except 1, calculate what the balance of the final token should be
        """
        peg_dict = {}
        peg_tkn_omitted = None
        for i, tkn in enumerate(self.asset_list):
            if tkn in reserves:
                peg_dict[tkn] = self.peg[i]
            else:
                if peg_tkn_omitted is not None:
                    raise AssertionError("reserves missing more than one token")
                peg_tkn_omitted = self.peg[i]

        # get all the balances except tkn_out and sort them from low to high
        balances = sorted([reserves[tkn] * peg_dict[tkn] for tkn in reserves])

        s = sum(balances)
        c = d
        for reserve in balances:
            c *= d / reserve / self.n_coins
        c *= d / self.ann / self.n_coins

        b = s + d / self.ann
        y = d

        for _ in range(max_iterations):
            y_prev = y
            y = (y ** 2 + c) / (2 * y + b - d)

            if (y <= y_prev and y_prev - y < self.precision) or (y > y_prev and y - y_prev <= self.precision):
                return y / peg_tkn_omitted

        return y / peg_tkn_omitted

    def sell_spot(self, tkn_sell, tkn_buy: str, fee: float = None):
        if tkn_buy not in self.liquidity or tkn_sell not in self.liquidity:
            return 0
        if fee is None:
            fee = 0
        fee_min = self.calculate_fee()
        fee = max(fee, fee_min)
        return self.price(tkn_sell, tkn_buy) * (1 - fee)

    def buy_spot(self, tkn_buy: str, tkn_sell, fee: float = None):
        if tkn_buy not in self.liquidity or tkn_sell not in self.liquidity:
            return 0
        if fee is None:
            fee = 0
        fee_min = self.calculate_fee()
        fee = max(fee, fee_min)
        return self.price(tkn_buy, tkn_sell) / (1 - fee)

    def sell_limit(self, tkn_buy, tkn_sell):  # TODO: fix this
        return self.liquidity[tkn_buy]

    def buy_limit(self, tkn_buy, tkn_sell):
        if tkn_buy not in self.liquidity:
            return 0
        return self.liquidity[tkn_buy]

    def calculate_buy_from_sell(self, tkn_buy, tkn_sell, sell_quantity):
        # 1. Calculate what the new peg and fee will be at the time of the swap
        new_peg, fee = self._calculate_new_peg()

        # 2. Store original peg to restore it later
        original_peg = self.peg
        self.peg = new_peg

        try:
            # 3. Now self.d and calculate_y will use the updated peg, exactly matching swap()
            reserves = self.modified_balances(delta={tkn_sell: sell_quantity}, omit=[tkn_buy])
            return (self.liquidity[tkn_buy] - self.calculate_y(reserves, self.d)) * (1 - fee)
        finally:
            # 4. Revert the state
            self.peg = original_peg

    def calculate_sell_from_buy(self, tkn_buy, tkn_sell, buy_quantity):
        new_peg, fee = self._calculate_new_peg()

        original_peg = self.peg
        self.peg = new_peg

        try:
            reserves = self.modified_balances(delta={tkn_buy: -buy_quantity}, omit=[tkn_sell])
            return (self.calculate_y(reserves, self.d) - self.liquidity[tkn_sell]) / (1 - fee)
        finally:
            self.peg = original_peg

    def price(self, tkn, denomination: str = ''):
        """
        return the price of TKN denominated in NUMÉRAIRE
        """
        if tkn == denomination:
            return 1
        if tkn not in self.liquidity or denomination not in self.liquidity:
            return 0
        i = list(self.liquidity.keys()).index(tkn)
        j = list(self.liquidity.keys()).index(denomination)
        peg_deltas = self._calculate_peg_deltas()
        d = self.calculate_d(peg_deltas=peg_deltas)
        return self.price_at_balance(balances=list(self.liquidity.values()), d=d, i=i, j=j, peg_deltas=peg_deltas)

    def price_at_balance(self, balances: list, d: float, i: int = 1, j: int = 0, peg_deltas = None):
        n = self.n_coins
        ann = self.ann

        if peg_deltas is None:
            peg = [self.peg[i] for i in range(n)]
        else:
            peg = [self.peg[i] + peg_deltas[i] for i in range(n)]

        c = d
        adj_balances = [balances[k] * peg[k] for k in range(n)]
        sorted_bal = sorted(adj_balances)
        for x in sorted_bal:
            c = c * d / (n * x)

        xi = adj_balances[i]
        xj = adj_balances[j]

        p = xj * (ann * xi + c) / (ann * xj + c) / xi

        p_adj = p * peg[i] / peg[j]

        return p_adj

    def share_price(self, numeraire: str = ''):
        i = 0 if numeraire == '' else list(self.liquidity.keys()).index(numeraire)
        d = self.calculate_d()
        s = self.shares
        a = self.amplification
        n = self.n_coins

        c = d
        sorted_liq = sorted([self.liquidity[self.asset_list[i]] * self.peg[i] for i in range(n)])
        for x in sorted_liq:
            c = c * d / (n * x)
        xi = self.liquidity[self.asset_list[i]] * self.peg[i]
        ann = self.ann
        p = (d * xi * ann + xi * (n + 1) * c - xi * d) / (xi * ann + c) / s
        return p / self.peg[i]

    def modified_balances(self, delta: dict = None, omit: list = ()) -> dict:
        balances = copy.copy(self.liquidity)
        if delta:
            for tkn, value in delta.items():
                balances[tkn] += value
        if omit:
            for tkn in omit:
                balances.pop(tkn)
        return balances

    def copy(self):
        new_pool = StableSwapPoolState(
            {k: v for k,v in self.liquidity.items()},
            amplification=self.amplification,
            precision=self.precision,
            trade_fee=self.trade_fee,
            unique_id=self.unique_id,
            spot_price_precision=self.spot_price_precision,
            shares=self.shares,
            peg=[v for v in self.peg[1:]],
            peg_target=[v for v in self.peg_target[1:]],
            max_peg_update=self.max_peg_update
        )
        new_pool.amp_change_step = self.amp_change_step
        new_pool.target_amp_block = self.target_amp_block
        new_pool.time_step = self.time_step
        new_pool.peg_target_updated_at = self.peg_target_updated_at
        new_pool.conversion_metrics = self.conversion_metrics
        return new_pool

    def __repr__(self):
        # round to given precision
        precision = 10
        liquidity = {tkn: round(self.liquidity[tkn], precision) for tkn in self.asset_list}
        shares = round(self.shares, precision)
        return (
            f'Stable Swap Pool: {self.unique_id}\n'
            f'********************************\n'
            f'trade fee: {self.trade_fee}\n'
            f'shares: {shares}\n'
            f'peg: {self.peg}\n'
            f'amplification constant: {self.amplification}\n'
            f'tokens: (\n\n'
        ) + '\n'.join(
            [(
                    f'    {token}\n'
                    f'    quantity: {liquidity[token]}\n'
                    + (
                        f'    conversion metrics:\n'
                        f'        price: {self.conversion_metrics[token]["price"]}\n'
                        f'        old shares: {self.conversion_metrics[token]["old_shares"]}\n'
                        f'        Omnipool shares: {self.conversion_metrics[token]["omnipool_shares"]}\n'
                        f'        subpool shares: {self.conversion_metrics[token]["subpool_shares"]}\n'
                        if token in self.conversion_metrics else ""
                    )
            ) for token in self.asset_list]
        ) + '\n)\n' + (
            f'error message:{self.fail or "none"}'
        )

    def _calculate_peg_deltas(self):
        peg_deltas = []
        block_ct = max([self.time_step - self.peg_target_updated_at, 1])
        for i in range(len(self.peg)):
            peg_diff = self.peg_target[i] - self.peg[i]
            max_peg_move = self.max_peg_update * self.peg[i] * block_ct
            if abs(peg_diff) <= max_peg_move:
                peg_deltas.append(peg_diff)
            elif self.peg[i] < self.peg_target[i]:
                peg_deltas.append(max_peg_move)
            else:
                peg_deltas.append(-max_peg_move)
        return peg_deltas

    def _calculate_fee_from_peg_deltas(self, peg_deltas):
        block_ct = max([self.time_step - self.peg_target_updated_at, 1])
        peg_relative_changes = [peg_deltas[i] / block_ct / self.peg[i] for i in range(len(self.peg))]
        peg_diff_per_block = (1 + max(peg_relative_changes)) / (1 + min(peg_relative_changes)) - 1
        return max(2 * peg_diff_per_block, self.trade_fee)

    def calculate_fee(self):
        peg_deltas = self._calculate_peg_deltas()
        return self._calculate_fee_from_peg_deltas(peg_deltas)

    def _calculate_new_peg(self) -> tuple:
        peg_deltas = self._calculate_peg_deltas()
        fee = self._calculate_fee_from_peg_deltas(peg_deltas)
        new_peg = [self.peg[i] + peg_deltas[i] for i in range(len(self.peg))]
        return new_peg, fee

    def _update_peg(self) -> float:
        new_peg, fee = self._calculate_new_peg()
        self.peg = new_peg
        return fee

    def swap(
            self,
            agent: Agent,
            tkn_sell: str,
            tkn_buy: str,
            buy_quantity: float = 0,
            sell_quantity: float = 0
    ):
        fee = self._update_peg()

        if buy_quantity:
            reserves = self.modified_balances(delta={tkn_buy: -buy_quantity}, omit=[tkn_sell])
            if min(reserves.values()) <= 0:
                return self.fail_transaction('Pool has insufficient liquidity.')
            sell_quantity = (self.calculate_y(reserves, self.d) - self.liquidity[tkn_sell]) / (1 - fee)
        elif sell_quantity:
            reserves = self.modified_balances(delta={tkn_sell: sell_quantity}, omit=[tkn_buy])
            buy_quantity = (self.liquidity[tkn_buy] - self.calculate_y(reserves, self.d)) * (1 - fee)

        if not agent.validate_holdings(tkn_sell, sell_quantity):
            return self.fail_transaction('Agent has insufficient funds.')

        agent.remove(tkn_sell, sell_quantity)
        agent.add(tkn_buy, buy_quantity)
        self.liquidity[tkn_buy] -= buy_quantity
        self.liquidity[tkn_sell] += sell_quantity

        return self

    def swap_one(
            self,
            agent: Agent,
            quantity: float,
            tkn_sell: str = '',
            tkn_buy: str = ''
    ):
        """
        This can be used when you want to change the price of one asset without changing the price of the others.
        Specify one asset to buy or sell, and the quantity of each of the *other* assets to sell or buy.
        The quantity of the specified asset to trade will be determined.
        Caution: this will only work correctly if the pool is initially balanced (spot prices equal on all assets).
        """
        if tkn_sell and tkn_buy:
            raise ValueError('Cannot specify both buy and sell quantities.')

        if tkn_buy:
            fee = self._update_peg()
            tkns_sell = list(filter(lambda t: t != tkn_buy, self.asset_list))
            for tkn in tkns_sell:
                if not agent.validate_holdings(tkn, quantity):
                    self.fail_transaction(f'Agent does not have any {tkn}.')

            sell_quantity = quantity
            buy_quantity = (self.liquidity[tkn_buy] - self.calculate_y(
                self.modified_balances(delta={tkn: quantity for tkn in tkns_sell}, omit=[tkn_buy]),
                self.d
            )) * (1 - fee)

            if self.liquidity[tkn_buy] < buy_quantity:
                return self.fail_transaction('Pool has insufficient liquidity.')

            for tkn in tkns_sell:
                self.liquidity[tkn] += sell_quantity
                agent.remove(tkn, sell_quantity)
            self.liquidity[tkn_buy] -= buy_quantity
            agent.add(tkn_buy, buy_quantity)

        elif tkn_sell:
            fee = self._update_peg()
            tkns_buy = list(filter(lambda t: t != tkn_sell, self.asset_list))
            buy_quantity = quantity

            if min([self.liquidity[tkn] for tkn in tkns_buy]) < buy_quantity:
                return self.fail_transaction('Pool has insufficient liquidity.')

            sell_quantity = (self.calculate_y(
                self.modified_balances(delta={tkn: -quantity for tkn in tkns_buy}, omit=[tkn_sell]),
                self.d
            ) - self.liquidity[tkn_sell]) / (1 - fee)
            if not agent.validate_holdings(tkn_sell, sell_quantity):
                return self.fail_transaction(f'Agent has insufficient funds.')
            for tkn in tkns_buy:
                self.liquidity[tkn] -= buy_quantity
                agent.add(tkn, buy_quantity)
            self.liquidity[tkn_sell] += sell_quantity
            agent.remove(tkn_sell, sell_quantity)

        return self

    def calculate_remove_liquidity(
            self,
            quantity: float,
            tkn_remove: str,
            fee: float = None
    ):
        """
        return the quantity of tkn_remove the agent will receive when withdrawing shares_removed
        """
        if fee is None:
            _fee = self.calculate_fee()
        else:
            _fee = fee
        _fee *= self.n_coins / 4 / (self.n_coins - 1)

        initial_d = self.calculate_d()
        reduced_d = initial_d - quantity * initial_d / self.shares

        xp_reduced = copy.copy(self.liquidity)
        xp_reduced.pop(tkn_remove)

        reduced_y = self.calculate_y(self.modified_balances(omit=[tkn_remove]), reduced_d)
        asset_reserve = self.liquidity[tkn_remove]

        for tkn in self.asset_list:
            if tkn == tkn_remove:
                dx_expected = self.liquidity[tkn] * reduced_d / initial_d - reduced_y
                asset_reserve -= _fee * dx_expected
                assert asset_reserve > 0
            else:
                dx_expected = self.liquidity[tkn] - self.liquidity[tkn] * reduced_d / initial_d
                xp_reduced[tkn] -= _fee * dx_expected
                assert xp_reduced[tkn] > 0

        dy = asset_reserve - self.calculate_y(xp_reduced, reduced_d)
        return dy

    def remove_liquidity(
            self,
            agent: Agent,
            shares_removed: float,
            tkn_remove: str
    ):
        """
        shares to remove is specified, quantity of tkn_remove is calculated
        """
        if shares_removed > agent.holdings[self.unique_id]:
            return self.fail_transaction('Agent has insufficient funds.')
        elif shares_removed <= 0:
            return self.fail_transaction('Withdraw quantity must be > 0.')

        fee = self._update_peg()
        dy = self.calculate_remove_liquidity(shares_removed, tkn_remove, fee)

        agent.holdings[self.unique_id] -= shares_removed
        self.shares -= shares_removed
        self.liquidity[tkn_remove] -= dy
        if tkn_remove not in agent.holdings:
            agent.holdings[tkn_remove] = 0
        agent.holdings[tkn_remove] += dy
        return self

    def calculate_add_liquidity(
            self,
            quantity: float,
            tkn_add: str,
            fee: float = None
    ):
        """
        return the number of shares the agent will receive when adding quantity of tkn_add
        """
        if fee is None:
            fee = self.trade_fee

        updated_reserves = {
            tkn: self.liquidity[tkn] + (quantity if tkn == tkn_add else 0) for tkn in self.asset_list
        }
        initial_d = self.calculate_d()
        updated_d = self.calculate_d(tuple(updated_reserves.values()))
        if updated_d < initial_d:
            return self.fail_transaction('invariant decreased for some reason')

        fixed_fee = fee
        fee = fixed_fee * self.n_coins / (4 * (self.n_coins - 1))

        d0, d1 = initial_d, updated_d

        adjusted_balances = (
            [
                updated_reserves[tkn] -
                abs(updated_reserves[tkn] - d1 * self.liquidity[tkn] / d0) * fee
                for tkn in self.asset_list
            ]
            if self.shares > 0 else updated_reserves
        )

        adjusted_d = self.calculate_d(adjusted_balances)
        if self.shares == 0:
            shares_return = updated_d
        else:
            d_diff = adjusted_d - initial_d
            shares_return = self.shares * d_diff / initial_d
        return shares_return

    def add_liquidity(
            self,
            agent: Agent,
            quantity: float,
            tkn_add: str
    ):
        """
        quantity of tkn_add to add is specified, shares to receive is calculated
        """
        if not agent.validate_holdings(tkn_add, quantity):
            return self.fail_transaction(f"Agent doesn't have enough {tkn_add}.")
        fee = self._update_peg()
        shares_return = self.calculate_add_liquidity(quantity, tkn_add, fee)
        self.shares += shares_return
        self.liquidity[tkn_add] += quantity
        agent.add(self.unique_id, shares_return)
        agent.remove(tkn_add, quantity)
        return self

    def calculate_buy_shares(
            self,
            quantity: float,
            tkn_add: str,
            fee: float = None
    ):
        if fee is None:
            fee = self.calculate_fee()
        initial_d = self.d
        d1 = initial_d + initial_d * quantity / self.shares

        xp = self.modified_balances(omit=[tkn_add])
        y = self.calculate_y(xp, d1)

        fee *= self.n_coins / (4 * (self.n_coins - 1))
        reserves_reduced = {}
        asset_reserve = 0
        for tkn, balance in self.liquidity.items():
            dx_expected = (
                    balance * d1 / initial_d - balance
            ) if tkn != tkn_add else (
                    y - balance * d1 / initial_d
            )
            reduced_balance = balance - fee * dx_expected
            if tkn == tkn_add:
                asset_reserve = reduced_balance
            else:
                reserves_reduced[tkn] = reduced_balance

        y1 = self.calculate_y(reserves_reduced, d1)
        delta_tkn = y1 - asset_reserve
        return delta_tkn

    def buy_shares(
            self,
            agent: Agent,
            quantity: float,
            tkn_add: str
    ):
        """
        quantity of shares to buy is specified, quantity of tkn_add to pay is calculated
        """
        trade_fee = self._update_peg()
        delta_tkn = self.calculate_buy_shares(quantity, tkn_add, trade_fee)
        if not agent.validate_holdings(tkn_add, delta_tkn):
            return self.fail_transaction(f"Agent doesn't have enough {tkn_add}.")

        self.liquidity[tkn_add] += delta_tkn
        self.shares += quantity
        agent.remove(tkn_add, delta_tkn)
        agent.add(self.unique_id, quantity)
        return self

    def calculate_withdraw_asset(self, tkn_remove, quantity, fee = None):
        if fee is None:
            fee = self.calculate_fee()
        balances_list = list(self.modified_balances(delta={tkn_remove: -quantity}).values())
        updated_d = self.calculate_d(balances_list)
        return self.shares * (1 - updated_d / self.d) / (1 - fee)

    def withdraw_asset(
            self,
            agent: Agent,
            quantity: float,
            tkn_remove: str
    ):
        """
        quantity of tkn_remove to withdraw is specified, shares to remove is calculated
        """
        if quantity >= self.liquidity[tkn_remove]:
            return self.fail_transaction(f'Not enough liquidity in {tkn_remove}.')
        if quantity <= 0:
            raise ValueError('Withdraw quantity must be > 0.')

        fee = self._update_peg()
        shares_removed = self.calculate_withdraw_asset(tkn_remove, quantity, fee)

        if not agent.validate_holdings(self.unique_id, shares_removed):
            return self.fail_transaction('Agent tried to remove more shares than it owns.')

        self.shares -= shares_removed
        self.liquidity[tkn_remove] -= quantity
        agent.remove(self.unique_id, shares_removed)
        agent.add(tkn_remove, quantity)
        return self

    def remove_uniform(
            self,
            agent: Agent,
            shares_removed: float
    ):
        if shares_removed > agent.holdings[self.unique_id]:
            raise ValueError('Agent tried to remove more shares than it owns.')
        elif shares_removed <= 0:
            raise ValueError('Withdraw quantity must be > 0.')

        self._update_peg()
        share_fraction = shares_removed / self.shares

        delta_tkns = {}
        for tkn in self.asset_list:
            delta_tkns[tkn] = share_fraction * self.liquidity[tkn]  # delta_tkn is positive

            if delta_tkns[tkn] >= self.liquidity[tkn]:
                return self.fail_transaction(f'Not enough liquidity in {tkn}.')

            if tkn not in agent.holdings:
                agent.holdings[tkn] = 0

        self.shares -= shares_removed
        agent.holdings[self.unique_id] -= shares_removed

        for tkn in self.asset_list:
            self.liquidity[tkn] -= delta_tkns[tkn]
            agent.holdings[tkn] += delta_tkns[tkn]  # agent is receiving funds, because delta_tkn is a negative number
        return self

    def add_liquidity_spot(self, tkn_add: str, precision: float = None, fee: float=None):
        """Calculates spot price of adding liquidity as shares denominated in liquidity"""
        if precision is None: precision = self.spot_price_precision
        trade_size = self.liquidity[tkn_add] * precision
        return trade_size / self.calculate_add_liquidity(quantity=trade_size, tkn_add=tkn_add, fee=fee)

    def buy_shares_spot(self, tkn_add: str, precision: float = None, fee: float=None):
        """Calculates spot price of buying shares as shares denominated in liquidity"""
        if precision is None: precision = self.spot_price_precision
        trade_size = self.liquidity[tkn_add] * precision * self.share_price(tkn_add)
        return self.calculate_buy_shares(quantity=trade_size, tkn_add=tkn_add, fee=fee) / trade_size

    def remove_liquidity_spot(self, tkn_remove: str, precision: float = None, fee: float = None):
        """Calculates spot price of removing liquidity as shares denominated in liquidity"""
        if precision is None: precision = self.spot_price_precision
        trade_size = self.liquidity[tkn_remove] * precision
        return self.calculate_remove_liquidity(quantity=trade_size, tkn_remove=tkn_remove, fee=fee) / trade_size

    def withdraw_asset_spot(self, tkn_remove: str, precision: float = None, fee: float = None):
        """Calculates spot price of withdrawing asset as shares denominated in liquidity"""
        if precision is None: precision = self.spot_price_precision
        trade_size = self.liquidity[tkn_remove] * precision
        delta_shares = self.calculate_withdraw_asset(tkn_remove, trade_size, fee=fee)
        return trade_size / delta_shares

    def cash_out(self, agent: Agent, prices: dict[str: float]) -> float:
        if self.unique_id not in agent.holdings:
            print(f'Error: agent does not have any shares in {self.unique_id}.')
            return 0

        new_state, new_agent = simulate_remove_uniform(
            old_state=self,
            old_agent=agent,
            shares_removed=agent.holdings[self.unique_id]
        )

        return sum([
            (new_agent.holdings[tkn] - (agent.holdings[tkn] if tkn in agent.holdings else 0)) * prices[tkn]
            if tkn in prices else 0
            for tkn in new_agent.holdings.keys()
        ])


def simulate_swap(
        old_state: StableSwapPoolState,
        old_agent: Agent,
        tkn_sell: str,
        tkn_buy: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0
):
    new_state = old_state.copy()
    new_agent = old_agent.copy()
    return new_state.swap(new_agent, tkn_sell, tkn_buy, buy_quantity, sell_quantity), new_agent


def simulate_add_liquidity(
        old_state: StableSwapPoolState,
        old_agent: Agent,
        quantity: float,  # quantity of asset to be added
        tkn_add: str
):
    new_state = old_state.copy()
    new_agent = old_agent.copy()
    return new_state.add_liquidity(new_agent, quantity, tkn_add), new_agent


def simulate_withdraw_asset(
        old_state: StableSwapPoolState,
        old_agent: Agent,
        quantity: float,
        tkn_remove: str
):
    new_state = old_state.copy()
    new_agent = old_agent.copy()
    return new_state.withdraw_asset(new_agent, quantity, tkn_remove), new_agent


def simulate_remove_liquidity(
        old_state: StableSwapPoolState,
        old_agent: Agent,
        quantity: float,  # in this case, quantity refers to a number of shares, not quantity of asset
        tkn_remove: str
):
    new_state = old_state.copy()
    new_agent = old_agent.copy()
    return new_state.remove_liquidity(new_agent, quantity, tkn_remove), new_agent


def simulate_remove_uniform(
        old_state: StableSwapPoolState,
        old_agent: Agent,
        shares_removed: float
):
    new_state = old_state.copy()
    new_agent = old_agent.copy()
    return new_state.remove_uniform(new_agent, shares_removed), new_agent


def simulate_buy_shares(
        old_state: StableSwapPoolState,
        old_agent: Agent,
        quantity: float,
        tkn_add: str
):
    new_state = old_state.copy()
    new_agent = old_agent.copy()
    return new_state.buy_shares(
        agent=new_agent,
        quantity=quantity,
        tkn_add=tkn_add
    ), new_agent


def balance_ratio_at_price(
        amplification: float,
        price: float,
        tkn_quantity: float = 1_000_000
):
    init_quantity = 1_000_000
    # find quantity that is too high
    spot = 0
    for i in range(100):
        tokens = {"A": init_quantity, "B": tkn_quantity}
        pool = StableSwapPoolState(tokens, amplification)
        spot = pool.price("A", "B")
        if spot == price:
            return init_quantity / (init_quantity + tkn_quantity)
        elif spot < price:
            break
        init_quantity *= 10
    if spot > price:
        raise ValueError('Price is too low.')
    # do binary search to identify quantity of asset 0.
    max_quantity = init_quantity
    min_quantity = 0
    mid_quantity = 0
    for i in range(100):
        mid_quantity = (max_quantity + min_quantity) / 2
        tokens = {"A": mid_quantity, "B": tkn_quantity}
        pool = StableSwapPoolState(tokens, amplification)
        spot = pool.price("A", "B")
        if spot == price:
            break
        elif spot < price:
            max_quantity = mid_quantity
        else:
            min_quantity = mid_quantity
    return mid_quantity / (mid_quantity + tkn_quantity)
