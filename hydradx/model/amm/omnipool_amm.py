import copy
from typing import Callable, Protocol, Literal
import math

from .agents import Agent
from .exchange import Exchange
from .oracle import Oracle, Block, OracleArchiveState

class AssetFeeCallable(Protocol):
    def __call__(self, tkn: str, delta_tkn: float = 0) -> float: ...


class DynamicFee:
    current: dict[str: float]
    def __init__(
        self,
        minimum: float = 0,
        maximum: float = float('inf'),
        amplification: float = 0,
        decay: float = 0,
        # ^^ if these four are provided, we can figure the rest out.
        current: dict[str: float] = None,
        liquidity: dict = None,
        net_volume: dict = None,
        last_updated: dict = None
    ):
        self.amplification = amplification
        self.decay = decay
        self.minimum = minimum
        self.maximum = maximum
        if current is None:
            self.current = {}
        else:
            self.current = current
        self.last_updated = last_updated if last_updated is not None else {}
        self.liquidity_at_last_update = liquidity if liquidity is not None else {}
        self.volume_at_last_update = net_volume if net_volume is not None else {}

    def update(self, time_step: int, volume: dict, liquidity: dict):
        for tkn in self.last_updated:
            if self.last_updated[tkn] == time_step:
                # update only when fee[tkn] has been accessed this block
                self.liquidity_at_last_update[tkn] = liquidity[tkn]
                self.volume_at_last_update[tkn] = volume[tkn]


class OmnipoolState(Exchange):
    unique_id: str = 'omnipool'
    fee_accumulator: dict[str: float]

    def __init__(self,
                 tokens: dict[str: dict],
                 tvl_cap: float = float('inf'),
                 preferred_stablecoin: str = None,
                 asset_fee: dict or DynamicFee or float = 0.0,
                 lrna_fee: dict or DynamicFee or float = 0.0,
                 oracles: dict[str: int] = None,
                 trade_limit_per_block: float = float('inf'),
                 update_function: Callable = None,
                 last_oracle_values: dict = None,
                 max_withdrawal_per_block: float = 1,
                 max_lp_per_block: float = float('inf'),
                 remove_liquidity_volatility_threshold: float = 0.0,
                 withdrawal_fee: bool = True,
                 min_withdrawal_fee: float = 0.0001,
                 lrna_mint_pct: float = 1.0,
                 unique_id: str = 'omnipool',
                 lrna_fee_burn: float = 0.5,
                 lrna_fee_destination: Agent = None,
                 dynamic_fee_precision: int = 20,
                 slip_factor: float = None,
                 minimum_slip_fee: float = 0.0,
                 ):
        """
        tokens should be a dict in the form of [str: dict]
        the nested dict needs the following parameters:
        {
          'liquidity': float  # starting risk asset liquidity in the pool
          (
          'LRNA': float  # starting LRNA on the other side of the pool
          or
          'LRNA_price': float  # price of the asset denominated in LRNA
          )

          optional:
          'weight_cap': float  # maximum fraction of TVL that may be held in this pool
          'oracle': dict  {name: period}  # name of the oracle its period, i.e. how slowly it decays
        }
        """

        super().__init__()

        if 'HDX' not in tokens:
            raise ValueError('HDX not included in tokens.')
        if preferred_stablecoin is not None and preferred_stablecoin not in tokens:
            raise ValueError(f'{preferred_stablecoin} is preferred stablecoin, but not included in tokens.')

        self.asset_list: list[str] = []
        self.liquidity = {}
        self.lrna = {}
        self.shares = {}
        self.protocol_shares = {}
        self.weight_cap = {}
        self.tvl_cap = tvl_cap
        if preferred_stablecoin is None and "USD" in tokens:
            self.stablecoin = "USD"
        else:
            self.stablecoin = preferred_stablecoin
        self.fail = ''
        self.update_function = update_function
        self.max_withdrawal_per_block = max_withdrawal_per_block
        self.max_lp_per_block = max_lp_per_block
        self.remove_liquidity_volatility_threshold = remove_liquidity_volatility_threshold
        self.lrna_mint_pct = lrna_mint_pct
        self.withdrawal_fee = withdrawal_fee
        self.oracles = {}
        self.asset_list = ['LRNA']

        for token, values in tokens.items():
            assert values['liquidity'], f'token {token} missing required parameter: liquidity'
            if 'LRNA' in values:
                lrna = values['LRNA']
            elif 'LRNA_price' in values:
                lrna = values['liquidity'] * values['LRNA_price']
            else:
                raise ValueError("token {name} missing required parameter: ('LRNA' or 'LRNA_price)")
            self.add_token(
                token,
                liquidity=values['liquidity'],
                lrna=lrna,
                shares=values['shares'] if 'shares' in values else values['liquidity'],
                protocol_shares=(
                    values['protocol_shares'] if 'protocol_shares' in values else (
                        values['shares'] if 'shares' in values else values['liquidity']
                    )
                ),
                weight_cap=values['weight_cap'] if 'weight_cap' in values else 1
            )

        if oracles is None or 'price' not in oracles:
            if last_oracle_values is None or 'price' not in last_oracle_values:
                self.oracles['price'] = Oracle(sma_equivalent_length=9, first_block=Block(self))
            else:
                self.oracles['price'] = Oracle(sma_equivalent_length=9, first_block=Block(self),
                                               last_values=last_oracle_values['price'])
        if last_oracle_values is not None and oracles is not None:
            self.oracles.update({
                name: Oracle(
                    sma_equivalent_length=period,
                    last_values=last_oracle_values[name] if name in last_oracle_values else None
                )
                for name, period in oracles.items()
            })
        elif oracles is not None:
            self.oracles.update({
                name: Oracle(sma_equivalent_length=period, first_block=Block(self))
                for name, period in oracles.items()
            })

        if withdrawal_fee:
            self.min_withdrawal_fee = min_withdrawal_fee

        # trades per block cannot exceed this fraction of the pool's liquidity
        self.trade_limit_per_block = trade_limit_per_block

        self.time_step = 0
        self.asset_fee = asset_fee
        self.lrna_fee = lrna_fee
        self.lrna_fee_burn = lrna_fee_burn
        if lrna_fee_burn > 1 or lrna_fee_burn < 0:
            raise ValueError('lrna_fee_burn must be >= 0 and <= 1')
        if lrna_fee_destination is None:
            lrna_fee_destination = Agent(holdings={'LRNA': 0})
        self.lrna_fee_destination = lrna_fee_destination
        self.dynamic_fee_precision = dynamic_fee_precision

        self.slip_factor = slip_factor
        self.minimum_slip_fee = minimum_slip_fee

        self.current_block = Block(self)
        self.unique_id = unique_id

    def _create_dynamic_fee(self, value: DynamicFee or dict or float, fee_type: Literal['lrna', 'asset']) -> DynamicFee:
        raise_oracle = 'price'
        def get_last_volume():
            return {
                tkn: (self.oracles[raise_oracle].volume_in[tkn]
                - self.oracles[raise_oracle].volume_out[tkn])
                if fee_type == 'lrna' else
                (self.oracles[raise_oracle].volume_out[tkn]
                - self.oracles[raise_oracle].volume_in[tkn])
                for tkn in self.liquidity
            }
        if isinstance(value, DynamicFee):
            return_val = DynamicFee(
                amplification=value.amplification,
                decay=value.decay,
                minimum=value.minimum,
                maximum=value.maximum,
                current={tkn: value.current[tkn] if tkn in value.current else value.minimum for tkn in self.liquidity},
                liquidity={tkn: value.liquidity_at_last_update[tkn] if tkn in value.liquidity_at_last_update else self.liquidity[tkn] for tkn in self.liquidity},
                net_volume={tkn: value.volume_at_last_update[tkn] for tkn in self.liquidity} if value.volume_at_last_update else get_last_volume(),
                last_updated={tkn: value.last_updated[tkn] if tkn in value.last_updated else self.time_step for tkn in self.liquidity }
            )
            return return_val
        elif isinstance(value, dict):
            current = {
                tkn: value[tkn] if tkn in value else (
                    (self.last_lrna_fee[tkn] if tkn in self.last_lrna_fee else self._lrna_fee.minimum)
                    if fee_type == 'lrna' else
                    (self.last_fee[tkn] if tkn in self.last_fee else self._asset_fee.minimum)
                ) for tkn in self.liquidity}
            return DynamicFee(
                current=current,
                minimum=min(current.values()),
                maximum=max(current.values()),
                liquidity={tkn: self.liquidity[tkn] for tkn in self.liquidity},
                net_volume=get_last_volume()
            )
        else:  # value is a number
            return DynamicFee(
                current={tkn: value for tkn in self.liquidity},
                maximum=value,
                minimum=value,
                liquidity={tkn: self.liquidity[tkn] for tkn in self.liquidity},
                net_volume=get_last_volume()
            )

    @property
    def lrna_fee(self) -> AssetFeeCallable:
        return self._get_lrna_fee

    @lrna_fee.setter
    def lrna_fee(self, value: DynamicFee or dict or float):
        self._lrna_fee = self._create_dynamic_fee(value, 'lrna')

    def set_lrna_fee(self, tkn: str, value: float):
        if tkn not in self._lrna_fee.current:
            raise ValueError(f'Token {tkn} not in pool')
        self._lrna_fee.current[tkn] = value
        self._lrna_fee.last_updated[tkn] = self.time_step

    def _get_lrna_fee(self, tkn, delta_q: float = 0) -> float:
        fee = self.compute_dynamic_fee(
            fee=self._lrna_fee,
            tkn=tkn
        )
        return min(fee, self._lrna_fee.maximum)

    @property
    def last_lrna_fee(self):
        return self._lrna_fee.current

    @property
    def asset_fee(self) -> AssetFeeCallable:
        return self._get_asset_fee

    @asset_fee.setter
    def asset_fee(self, value):
        self._asset_fee = self._create_dynamic_fee(value, 'asset')

    def set_asset_fee(self, tkn: str, value: float):
        if tkn not in self._asset_fee.current:
            raise ValueError(f'Token {tkn} not in pool')
        self._asset_fee.current[tkn] = value
        self._asset_fee.last_updated[tkn] = self.time_step

    def _get_asset_fee(self, tkn, delta_q: float = 0) -> float:
        fee = self.compute_dynamic_fee(
            fee=self._asset_fee,
            tkn=tkn
        )
        return min(fee, self._asset_fee.maximum)

    @property
    def last_fee(self):
        return self._asset_fee.current

    def compute_dynamic_fee(self, fee: DynamicFee, tkn: str) -> float:
        if fee.amplification == 0:
            fee.last_updated[tkn] = self.time_step
        if tkn in fee.last_updated:
            if fee.last_updated[tkn] == self.time_step:
                # return the last fee if it's already been computed for this tkn and block
                return fee.current[tkn]
        else:
            fee.last_updated[tkn] = self.time_step - 1

        # use this approximation to catch up to where we think the fee should be,
        # knowing there have been no trades until now
        num_blocks = int(self.time_step - fee.last_updated[tkn])
        m = min(self.dynamic_fee_precision, num_blocks)
        prev_volume = fee.volume_at_last_update[tkn] if tkn in fee.volume_at_last_update else 0
        prev_liquidity = fee.liquidity_at_last_update[tkn] if tkn in fee.liquidity_at_last_update else self.liquidity[tkn]
        x = fee.amplification * prev_volume / self.liquidity[tkn]
        j_sum = 0
        w = 1 - self.oracles['price'].decay_factor
        w_j = 1
        for j in range(m):
            j_sum += w_j / (
                    1 + (prev_liquidity - self.liquidity[tkn]) / self.liquidity[tkn] * w_j
            )
            w_j *= w
        w_term = (math.pow(w, m + 1) - math.pow(w, num_blocks + 1)) / self.oracles['price'].decay_factor
        delta = x * (j_sum + w_term) - num_blocks * fee.decay
        fee_value = min(max((fee.current[tkn] if tkn in fee.current else 0) + delta, fee.minimum), fee.maximum)
        fee.current[tkn] = fee_value
        fee.last_updated[tkn] = self.time_step
        return fee_value

    def compute_slip_fee(self, tkn: str, delta_q: float) -> float:
        if self.slip_factor is None:
            return 0.0
        else:
            delta_q += self.lrna[tkn] - self.current_block.lrna[tkn]
            return self.slip_factor * abs(delta_q) / (self.current_block.lrna[tkn] + delta_q) + self.minimum_slip_fee
        # calc for selling:
        # qi = -li * ri / (li + ri)
        # f = s * -qi / (ln - qi)
        # qj = qi * (1 - f)
        # rj = - lj * qj / (lj + qj)
        # f = s * q / (ln + q)
        # dr = ln * -r / (li + r)

    def add_token(
            self,
            tkn: str,
            liquidity: float,
            lrna: float,
            shares: float = 0,
            protocol_shares: float = 0,
            weight_cap: float = 1
    ):
        self.asset_list.append(tkn)
        self.liquidity[tkn] = liquidity
        self.lrna[tkn] = lrna
        self.shares[tkn] = shares or liquidity
        self.protocol_shares[tkn] = protocol_shares or self.shares[tkn]
        self.weight_cap[tkn] = weight_cap
        # self.fee_accumulator[tkn] = 0
        if hasattr(self, '_lrna_fee'):
            if tkn not in self._lrna_fee.current: self._lrna_fee.current[tkn] = self._lrna_fee.minimum
            if tkn not in self._asset_fee.current: self._asset_fee.current[tkn] = self._asset_fee.minimum
            if tkn not in self._lrna_fee.liquidity_at_last_update: self._lrna_fee.liquidity_at_last_update[tkn] = liquidity
            if tkn not in self._asset_fee.liquidity_at_last_update: self._asset_fee.liquidity_at_last_update[tkn] = liquidity
            if tkn not in self._lrna_fee.volume_at_last_update: self._lrna_fee.volume_at_last_update[tkn] = 0
            if tkn not in self._asset_fee.volume_at_last_update: self._asset_fee.volume_at_last_update[tkn] = 0
            if tkn not in self._lrna_fee.last_updated: self._lrna_fee.last_updated[tkn] = 0
            if tkn not in self._asset_fee.last_updated: self._asset_fee.last_updated[tkn] = 0
        if hasattr(self, 'current_block'):
            self.current_block.price[tkn] = self.lrna[tkn] / self.liquidity[tkn]
            self.current_block.liquidity[tkn] = self.liquidity[tkn]
            self.current_block.lps[tkn] = 0
            self.current_block.withdrawals[tkn] = 0
            self.current_block.volume_in[tkn] = 0
            self.current_block.volume_out[tkn] = 0
            self.current_block.lrna[tkn] = self.lrna[tkn]
        for oracle in self.oracles.values() if self.oracles else []:
            oracle.liquidity[tkn] = self.liquidity[tkn]
            oracle.price[tkn] = self.lrna[tkn] / self.liquidity[tkn]
            oracle.volume_in[tkn] = 0
            oracle.volume_out[tkn] = 0
        return self

    def remove_token(self, tkn: str):
        self.asset_list.remove(tkn)
        return self

    def update(self):
        # update oracles
        self.current_block.price['HDX'] = self.lrna['HDX'] / self.liquidity['HDX']

        for name, oracle in self.oracles.items():
            oracle.update(self.current_block)

        # update fees
        self._lrna_fee.update(
            time_step=self.time_step,
            volume={
                tkn: (self.oracles['price'].volume_in[tkn] - self.oracles['price'].volume_out[tkn])
                for tkn in self._lrna_fee.current
            },
            liquidity=self.oracles['price'].liquidity
        )
        self._asset_fee.update(
            time_step=self.time_step,
            volume={
                tkn: (self.oracles['price'].volume_out[tkn] - self.oracles['price'].volume_in[tkn])
                for tkn in self._asset_fee.current
            },
            liquidity=self.oracles['price'].liquidity
        )

        # update current block
        self.time_step += 1
        self.current_block = Block(self)

        self.fail = ''
        if self.update_function:
            self.update_function(self)

        return self

    @property
    def lrna_total(self):
        return sum(self.lrna.values())

    @property
    def total_value_locked(self):
        if self.stablecoin is None:
            raise ValueError('No stablecoin defined')
        return self.liquidity[self.stablecoin] * self.lrna_total / self.lrna[self.stablecoin]

    def sell_limit(self, tkn_buy: str, tkn_sell: str):
        return float('inf')

    def buy_limit(self, tkn_buy: str, tkn_sell: str):
        if tkn_buy not in self.liquidity:
            return 0
        diff = 1
        fee = self.asset_fee(tkn_buy, self.liquidity[tkn_buy])
        limit = self.liquidity[tkn_buy] * (1 - fee)
        while abs(diff) > 0.000001:
            fee = self.asset_fee(tkn_buy, limit)
            new_limit = self.liquidity[tkn_buy] * (1 - fee)
            diff = limit - new_limit
        return limit

    def copy(self):
        copy_state = copy.deepcopy(self)
        copy_state.fail = ''
        return copy_state

    def archive(self):
        return OmnipoolArchiveState(self)

    def __repr__(self):
        # don't go overboard with the precision here
        precision = 12
        lrna = {tkn: round(self.lrna[tkn], precision) for tkn in self.lrna}
        lrna_total = round(self.lrna_total, precision)
        liquidity = {tkn: round(self.liquidity[tkn], precision) for tkn in self.liquidity}
        weight_cap = {tkn: round(self.weight_cap[tkn], precision) for tkn in self.weight_cap}
        usd_prices = (
            {tkn: round(self.usd_price(tkn), precision) for tkn in self.asset_list}
            if self.stablecoin is not None
            else {tkn: 'N/A' for tkn in self.asset_list}
        )
        newline = '\n'
        return (
            f'Omnipool: {self.unique_id}\n'
            f'********************************\n'
            f'tvl cap: {self.tvl_cap}\n'
            f'lrna fee:\n\n'
            f'{newline.join([f"    {tkn}: {self.last_lrna_fee[tkn]}" for tkn in self.liquidity])}\n\n'
            f'asset fee:\n\n'
            f'{newline.join([f"    {tkn}: {self.last_fee[tkn]}" for tkn in self.liquidity])}\n\n'
            f'asset pools: (\n\n'
        ) + '\n'.join(
            [(
                    f'    *{tkn}*\n'
                    f'    asset quantity: {liquidity[tkn]}\n'
                    f'    lrna quantity: {lrna[tkn]}\n'
                    f'    USD price: {usd_prices[tkn]}\n' +
                    # f'    tvl: ${lrna[tkn] * liquidity[self.stablecoin] / lrna[self.stablecoin]}\n'
                    f'    weight: {lrna[tkn]}/{lrna_total} ({lrna[tkn] / lrna_total})\n'
                    f'    weight cap: {weight_cap[tkn]}\n'
                    f'    total shares: {self.shares[tkn]}\n'
                    f'    protocol shares: {self.protocol_shares[tkn]}\n'
            ) for tkn in self.liquidity]
        ) + '\n)\n' + f'oracles: (\n' + '\n\n'.join([
            f'    name: {name}\n    length: {oracle.length}\n'
            for name, oracle in self.oracles.items()
        ]) + f'\n)\n\nerror message: {self.fail or "None"}'

    def calculate_sell_from_buy(self, tkn_buy: str, tkn_sell: str, buy_quantity: float):
        if tkn_buy == "LRNA":
            D = buy_quantity
        else:
            A = self.liquidity[tkn_buy]
            Qb = self.lrna[tkn_buy]
            af = self.asset_fee(tkn_buy)
            b = buy_quantity / (1.0 - af)
            D = (b * Qb) / (A - b)

        if tkn_sell == "LRNA":
            return D

        L = self.liquidity[tkn_sell]
        last_block_h2o = self.current_block.lrna[tkn_sell]
        current_h2o = self.lrna[tkn_sell]
        k = 1.0 - self.lrna_fee(tkn_sell) - self.minimum_slip_fee
        s = (self.slip_factor or 0.0)
        C = current_h2o - last_block_h2o

        if abs(D - C * k) < 1e-15:
            x = C
        else:
            p = (k - s) if (D < C * k) else (k + s)
            q = (k * last_block_h2o + D - p * C)
            r = last_block_h2o * (D - C * k)

            disc = q * q - 4 * p * r
            if disc < 0:
                return math.inf  # infeasible: no real solution
            sd = disc ** 0.5
            u = (2 * r) / (-q - (sd if q >= 0 else -sd))
            x = C - u

        if not (0 < x < current_h2o):
            return math.inf
        if (last_block_h2o + (C - x)) <= 0:
            return math.inf
        if tkn_sell != "LRNA":
            # in case we would like to do something in particular with this fee
            slip_rate = self.compute_slip_fee(tkn_sell, -x)

        sell_quantity = (L * x) / (current_h2o - x)
        return sell_quantity

    def calculate_buy_from_sell(
            self,
            tkn_buy: str,
            tkn_sell: str,
            sell_quantity: float
    ):
        """
        Given a sell quantity, calculate the effective price, so we can execute it as a buy
        """
        asset_fee = self.asset_fee(tkn_buy) if tkn_buy in self.liquidity else 0
        lrna_fee = self.lrna_fee(tkn_sell) if tkn_sell in self.liquidity else 0
        if tkn_sell != "LRNA":
            delta_qi = self.lrna[tkn_sell] * sell_quantity / (self.liquidity[tkn_sell] + sell_quantity)
            slip_fee = self.compute_slip_fee(tkn_sell, -delta_qi)
            lrna_fee_total = delta_qi * (lrna_fee + slip_fee)
            delta_q = delta_qi - lrna_fee_total
        else:
            delta_q = sell_quantity

        if tkn_buy != "LRNA":
            delta_ra = self.liquidity[tkn_buy] * delta_q / (delta_q + self.lrna[tkn_buy]) * (1 - asset_fee)
            return delta_ra
        else:
            return delta_q

    def buy_spot(self, tkn_buy: str, tkn_sell: str, fee: float = None):
        if tkn_buy == tkn_sell:
            return 1
        if tkn_buy not in self.asset_list and tkn_buy != 'LRNA':
            raise ValueError(f'Invalid token: {tkn_buy}')
        if tkn_sell not in self.asset_list and tkn_sell != 'LRNA':
            raise ValueError(f'Invalid token: {tkn_sell}')
        if fee is None:
            fee = {}
            if tkn_buy == 'LRNA':
                fee['asset'] = 0
            elif tkn_sell == 'LRNA':
                fee['lrna'] = 0
            if 'lrna' not in fee:
                fee['lrna'] = self.lrna_fee(tkn_sell, 0)
            if 'asset' not in fee:
                fee['asset'] = self.asset_fee(tkn_buy, 0)
        elif not isinstance(fee, dict):
            fee = {
                'lrna': fee,
                'asset': fee
            }
        return self.price(tkn_buy, tkn_sell) / (1 - fee['lrna']) / (1 - fee['asset'])

    def sell_spot(self, tkn_sell: str, tkn_buy: str, fee: float = None):
        if tkn_buy == tkn_sell:
            return 1
        if tkn_buy not in self.asset_list and tkn_buy != 'LRNA':
            raise ValueError(f'Invalid token: {tkn_buy}')
        if tkn_sell not in self.asset_list and tkn_sell != 'LRNA':
            raise ValueError(f'Invalid token: {tkn_sell}')
        if fee is None:
            fee = {}
            if tkn_buy == 'LRNA':
                fee['asset'] = 0
            elif tkn_sell == 'LRNA':
                fee['lrna'] = 0
            if 'lrna' not in fee:
                fee['lrna'] = self.lrna_fee(tkn_sell, 0)
            if 'asset' not in fee:
                fee['asset'] = self.asset_fee(tkn_buy, 0)
        elif not isinstance(fee, dict):
            fee = {
                'lrna': fee,
                'asset': fee
            }
        return self.price(tkn_sell, tkn_buy) * (1 - fee['lrna']) * (1 - fee['asset'])

    def swap(
            self,
            agent: Agent,
            tkn_buy: str, tkn_sell: str,
            buy_quantity: float = None,
            sell_quantity: float = None
    ):
        """
        execute swap in place (modify and return self and agent)
        all swaps, LRNA, sub-pool, and asset swaps, are executed through this function
        """
        if buy_quantity is None:
            buy_quantity = 0
        if sell_quantity is None:
            sell_quantity = 0
        old_buy_liquidity = self.liquidity[tkn_buy] if tkn_buy in self.liquidity else 0
        old_sell_liquidity = self.liquidity[tkn_sell] if tkn_sell in self.liquidity else 0

        if (tkn_buy not in self.asset_list and tkn_buy != 'LRNA') or (tkn_sell not in self.asset_list and tkn_sell != 'LRNA'):
            raise ValueError(f'Invalid token pair: {tkn_buy}, {tkn_sell}')
        elif tkn_sell == 'LRNA':
            return_val = self._lrna_swap(agent, buy_quantity, -sell_quantity, tkn_buy)
        elif tkn_buy == 'LRNA':
            return_val = self._lrna_swap(agent, -sell_quantity, buy_quantity, tkn_sell)

        elif buy_quantity and not sell_quantity:
            # back into correct delta_Ri, then execute sell
            delta_Ri = self.calculate_sell_from_buy(tkn_buy, tkn_sell, buy_quantity)
            if delta_Ri < 0:
                return self.fail_transaction(f'insufficient LRNA in {tkn_sell}')
            if delta_Ri == float('inf'):
                return self.fail_transaction('not enough liquidity in sell pool to buy that much')
            # including both buy_quantity and sell_quantity potentially introduces a 'hack'
            # where you could include both and *not* have them match, but we're not worried about that
            # because this is not a production environment. Just don't do it.
            # Here, we are doing it to ensure that buy_quantity is exactly what the user wanted despite rounding error.
            return self.swap(
                agent=agent,
                tkn_buy=tkn_buy,
                tkn_sell=tkn_sell,
                buy_quantity=buy_quantity,
                sell_quantity=delta_Ri
            )
        else:
            # basic Omnipool swap
            i = tkn_sell
            j = tkn_buy
            delta_Ri = sell_quantity
            if delta_Ri <= 0:
                return self.fail_transaction('sell amount must be greater than zero')
            if not agent.validate_holdings(i, delta_Ri):
                return self.fail_transaction(f"Agent doesn't have enough {i}")

            # get the fees we will be using
            asset_fee = self.asset_fee(tkn_buy)
            lrna_fee = self.lrna_fee(tkn_sell)
            # also update both fees for each asset, because that's what they do in production
            self.asset_fee(tkn_sell)
            self.lrna_fee(tkn_buy)

            delta_qi = self.lrna[tkn_sell] * -delta_Ri / (self.liquidity[tkn_sell] + delta_Ri)
            lrna_fee_total = -delta_qi * lrna_fee
            lrna_fee_burn = self.lrna_fee_burn * lrna_fee_total
            fee_deposit = lrna_fee_total - lrna_fee_burn

            slip_fee_total = -delta_qi * self.compute_slip_fee(tkn_sell, delta_qi)
            # we could do something different with this fee, but for now it just stays in the pool

            delta_Qt = -delta_qi - lrna_fee_total - slip_fee_total
            delta_Qm = (self.lrna[tkn_buy] + delta_Qt) * delta_Qt * asset_fee / self.lrna[
                tkn_buy] * self.lrna_mint_pct
            delta_Qj = delta_Qt + delta_Qm
            delta_Rj = self.liquidity[tkn_buy] * -delta_Qt / (self.lrna[tkn_buy] + delta_Qt) * (1 - asset_fee)
            delta_QH = 0  # -lrna_fee * delta_Qi
            if self.lrna_fee_destination:
                self.lrna_fee_destination.holdings['LRNA'] += fee_deposit

            # per-block trade limits
            if (
                    -delta_Rj - self.current_block.volume_in[tkn_buy] + self.current_block.volume_out[tkn_buy]
                    > self.trade_limit_per_block * self.current_block.liquidity[tkn_buy]
            ):
                return self.fail_transaction(
                    f'{self.trade_limit_per_block * 100}% per block trade limit exceeded in {tkn_buy}.'
                )
            elif (
                    delta_Ri + self.current_block.volume_in[tkn_sell] - self.current_block.volume_out[tkn_sell]
                    > self.trade_limit_per_block * self.current_block.liquidity[tkn_sell]
            ):
                return self.fail_transaction(
                    f'{self.trade_limit_per_block * 100}% per block trade limit exceeded in {tkn_sell}.'
                )
            self.lrna[i] += delta_qi
            self.lrna[j] += delta_Qj
            self.liquidity[i] += delta_Ri
            self.liquidity[j] += -buy_quantity or delta_Rj
            self.lrna['HDX'] += delta_QH

            agent.remove(i, delta_Ri)
            agent.add(j, buy_quantity or -delta_Rj)

            return_val = self

        # update oracle
        if tkn_buy in self.liquidity:
            buy_quantity = old_buy_liquidity - self.liquidity[tkn_buy]
            self.current_block.volume_out[tkn_buy] += buy_quantity
            self.current_block.price[tkn_buy] = self.lrna[tkn_buy] / self.liquidity[tkn_buy]
        if tkn_sell in self.liquidity:
            sell_quantity = self.liquidity[tkn_sell] - old_sell_liquidity
            self.current_block.volume_in[tkn_sell] += sell_quantity
            self.current_block.price[tkn_sell] = self.lrna[tkn_sell] / self.liquidity[tkn_sell]
        return return_val

    def _lrna_swap(
            self,
            agent: Agent,
            delta_ra: float = 0,
            delta_qa: float = 0,
            tkn: str = ''
    ):
        """
        Execute LRNA swap in place (modify and return)
        """
        asset_fee = self.asset_fee(tkn) if tkn in self.liquidity else 0
        lrna_fee = self.lrna_fee(tkn) if tkn in self.liquidity else 0

        if delta_qa < 0:
            # selling LRNA
            if not agent.validate_holdings('LRNA', -delta_qa):
                return self.fail_transaction('Agent has insufficient lrna')
            delta_ra = -self.liquidity[tkn] * delta_qa / (-delta_qa + self.lrna[tkn]) * (1 - asset_fee)
            delta_qm = asset_fee * (-delta_qa) / self.lrna[tkn] * (self.lrna[tkn] - delta_qa) * self.lrna_mint_pct
            delta_q = delta_qm - delta_qa

            self.lrna[tkn] += delta_q
            self.liquidity[tkn] -= delta_ra

        elif delta_ra > 0:
            # buying asset
            if -delta_ra + self.liquidity[tkn] <= 0:
                return self.fail_transaction('insufficient assets in pool')
            denom = (self.liquidity[tkn] * (1 - asset_fee) - delta_ra)
            delta_qa = -self.lrna[tkn] * delta_ra / denom
            delta_qm = -asset_fee * (1 - asset_fee) * (self.liquidity[tkn] / denom) * delta_qa * self.lrna_mint_pct
            delta_q = -delta_qa + delta_qm

            self.lrna[tkn] += delta_q
            self.liquidity[tkn] -= delta_ra

        # buying LRNA
        elif delta_qa > 0:
            # buying LRNA
            lrna_fee_total = delta_qa / (1 - lrna_fee) - delta_qa
            slip_fee_total = delta_qa * self.compute_slip_fee(tkn, -delta_qa)
            delta_qi = -delta_qa - lrna_fee_total - slip_fee_total
            lrna_fee_burn = self.lrna_fee_burn * lrna_fee_total
            fee_deposit = lrna_fee_total - lrna_fee_burn
            if delta_qi + self.lrna[tkn] <= 0:
                return self.fail_transaction('insufficient lrna in pool')
            delta_ra = -self.liquidity[tkn] * -delta_qi / (delta_qi + self.lrna[tkn])
            if not agent.validate_holdings(tkn, -delta_ra):
                return self.fail_transaction('Agent has insufficient assets')
            self.lrna[tkn] += delta_qi  # burn the LRNA fee
            self.liquidity[tkn] += -delta_ra
            if self.lrna_fee_destination:
                self.lrna_fee_destination.holdings['LRNA'] += fee_deposit

        elif delta_ra < 0:
            # selling asset
            if not agent.validate_holdings(tkn, delta_ra):
                return self.fail_transaction('agent has insufficient assets')
            delta_qi = self.lrna[tkn] * delta_ra / (self.liquidity[tkn] - delta_ra)
            lrna_fee_total = -delta_qi * lrna_fee
            slip_fee_total = -delta_qi * self.compute_slip_fee(tkn, -delta_qi)
            lrna_fee_burn = lrna_fee_total * self.lrna_fee_burn
            fee_deposit = lrna_fee_total - lrna_fee_burn
            delta_qa = -delta_qi - lrna_fee_total
            self.lrna[tkn] += delta_qi
            self.liquidity[tkn] -= delta_ra
            if self.lrna_fee_destination:
                self.lrna_fee_destination.holdings['LRNA'] += fee_deposit

        else:
            return self.fail_transaction('All deltas are zero.')

        if delta_qa > 0:
            agent.add('LRNA', delta_qa)
            agent.remove(tkn, -delta_ra)
        else:
            agent.remove('LRNA', -delta_qa)
            agent.add(tkn, delta_ra)

        return self

    def calculate_remove_liquidity(self, agent: Agent, quantity: float = None, tkn_remove: str = None,
                                   nft_id: str = None):
        """
        If quantity is specified and nft_id is specified, remove specified quantity of shares from specified position.
        If quantity is specified and nft_id is unspecified, remove specified quantity of shares from holdings.
        If quantity is unspecified and nft_id is specified, remove specified position.
        If quantity is unspecified and nft_id is unspecified, remove all liquidity.
        """

        if tkn_remove is None:
            if nft_id is None:
                raise AssertionError('tkn_remove must be specified if nft_id is not provided.')
            else:
                tkn_remove = agent.nfts[nft_id].tkn
        reset_withdrawal_fee = self.withdrawal_fee
        if agent.immune_to_fees:
            self.withdrawal_fee = False

        delta_qa, delta_r, delta_q, delta_s, delta_b, nft_ids = 0, 0, 0, 0, 0, []
        if quantity is not None:
            if nft_id is None:  # remove specified quantity of shares from holdings
                k = (self.unique_id, tkn_remove)
                delta_qa, delta_r, delta_q, delta_s, delta_b = self._calculate_remove_one_position(
                    quantity=quantity, tkn_remove=tkn_remove, share_price=agent.share_prices[k]
                )
            else:  # remove specified quantity of shares from specified position
                delta_qa, delta_r, delta_q, delta_s, delta_b = self._calculate_remove_one_position(
                    quantity=quantity, tkn_remove=tkn_remove, share_price=agent.nfts[nft_id].price
                )
                nft_ids = [nft_id]
        else:
            if nft_id is not None:  # remove specified position
                delta_qa, delta_r, delta_q, delta_s, delta_b = self._calculate_remove_one_position(
                    quantity=agent.nfts[nft_id].shares, tkn_remove=tkn_remove, share_price=agent.nfts[nft_id].price
                )
                nft_ids = [nft_id]
            else:  # remove all liquidity
                for nft_id in agent.nfts:
                    nft = agent.nfts[nft_id]
                    if isinstance(nft, OmnipoolLiquidityPosition):
                        if nft.pool_id == self.unique_id and nft.tkn == tkn_remove:
                            nft_ids.append(nft_id)
                            dqa, dr, dq, ds, db = self._calculate_remove_one_position(
                                quantity=nft.shares, tkn_remove=tkn_remove, share_price=nft.price
                            )
                            delta_qa += dqa
                            delta_r += dr
                            delta_q += dq
                            delta_s += ds
                            delta_b += db
                if (self.unique_id, tkn_remove) in agent.holdings:
                    dqa, dr, dq, ds, db = self._calculate_remove_one_position(
                        quantity=agent.holdings[(self.unique_id, tkn_remove)], tkn_remove=tkn_remove,
                        share_price=agent.share_prices[(self.unique_id, tkn_remove)]
                    )
                    delta_qa += dqa
                    delta_r += dr
                    delta_q += dq
                    delta_s += ds
                    delta_b += db
        if agent.immune_to_fees:
            self.withdrawal_fee = reset_withdrawal_fee
        return delta_qa, delta_r, delta_q, delta_s, delta_b, nft_ids

    def _calculate_remove_one_position(self, quantity, tkn_remove, share_price):
        """
        calculated the pool and agent deltas for removing liquidity from a sub pool
        return as a tuple in this order:
        delta_qa, delta_r, delta_q, delta_s, delta_b, delta_l

        delta_qa (agent LRNA)
        delta_r (pool/agent liquidity)
        delta_q (pool LRNA)
        delta_s (pool shares)
        delta_b (protocol shares)
        delta_l (LRNA imbalance)
        """
        k = (self.unique_id, tkn_remove)

        quantity = -abs(quantity)
        assert quantity <= 0, f"delta_S cannot be positive: {quantity}"
        if tkn_remove not in self.liquidity:
            raise AssertionError(f"Invalid share token name: {tkn_remove}")

        # determine if they should get some LRNA back as well as the asset they invested
        piq = self.lrna_price(tkn_remove)
        p0 = share_price
        mult = (piq - p0) / (piq + p0)

        # Share update
        delta_b = max(mult * quantity, 0)
        delta_s = quantity + delta_b

        # Token amounts update
        delta_q = self.lrna[tkn_remove] / self.shares[tkn_remove] * delta_s
        delta_r = delta_q / piq

        if piq > p0:  # prevents rounding errors
            delta_qa = -piq * (
                    2 * piq / (piq + p0) * quantity / self.shares[tkn_remove]
                    * self.liquidity[tkn_remove] - delta_r
            )
        else:
            delta_qa = 0

        if self.withdrawal_fee:
            fee = self.current_withdrawal_fee(tkn_remove)
            delta_r *= 1 - fee
            delta_qa *= 1 - fee
            delta_q *= 1 - fee

        # L update: LRNA fees to be burned before they will start to accumulate again
        return delta_qa, delta_r, delta_q, delta_s, delta_b

    def current_withdrawal_fee(self, tkn):
        diff = abs(self.oracles['price'].price[tkn] - self.lrna_price(tkn)) / self.oracles['price'].price[tkn]
        fee = max(min(diff, 1), self.min_withdrawal_fee)
        return fee

    def add_liquidity(
            self,
            agent: Agent = None,
            quantity: float = 0,
            tkn_add: str = '',
            nft_id: str = None
    ):
        """Compute new state after liquidity addition"""

        if quantity <= 0:
            return self.fail_transaction('Quantity must be non-negative.')

        delta_Q = self.lrna_price(tkn_add) * quantity

        if nft_id is None and agent.get_holdings((self.unique_id, tkn_add)) > 0:
            return self.fail_transaction(
                'Agent already has liquidity in this pool. Try using nft_id input.'
            )

        if nft_id is not None and nft_id in agent.nfts:
            raise AssertionError('Agent already has an NFT with this ID.')

        if not agent.validate_holdings(tkn_add, quantity):
            return self.fail_transaction(
                f'Agent has insufficient funds ({agent.holdings[tkn_add]} < {quantity}).'
            )

        if (self.lrna[tkn_add] + delta_Q) / (self.lrna_total + delta_Q) > self.weight_cap[tkn_add]:
            return self.fail_transaction(
                'Transaction rejected because it would exceed the weight cap in pool[{i}].'
            )

        if self.tvl_cap < float('inf'):
            if (self.total_value_locked() + quantity * self.usd_price(tkn_add)) > self.tvl_cap:
                return self.fail_transaction('Transaction rejected because it would exceed the TVL cap.')

        # assert quantity > 0, f"delta_R must be positive: {quantity}"
        if quantity > self.max_lp_per_block * self.liquidity[tkn_add] - self.current_block.lps[tkn_add]:
            return self.fail_transaction(
                'Transaction rejected because it would exceed the max LP per block.'
            )

        # Share update
        if self.shares[tkn_add]:
            shares_added = (delta_Q / self.lrna[tkn_add]) * self.shares[tkn_add]
        else:
            shares_added = delta_Q
        self.shares[tkn_add] += shares_added

        # LRNA add (mint)
        self.lrna[tkn_add] += delta_Q

        # Token amounts update
        self.liquidity[tkn_add] += quantity

        # agent update
        agent.remove(tkn_add, quantity)
        if nft_id is None:
            k = (self.unique_id, tkn_add)
            # shares go to provisioning agent
            agent.holdings[k] = shares_added
            # set price at which liquidity was added
            agent.share_prices[k] = self.lrna_price(tkn_add)
            agent.delta_r[k] = quantity
        else:
            lp_position = OmnipoolLiquidityPosition(tkn_add, self.lrna_price(tkn_add), shares_added, quantity,
                                                    self.unique_id)
            agent.nfts[nft_id] = lp_position

        # update block
        self.current_block.lps[tkn_add] += quantity
        # update fees
        self.asset_fee(tkn_add, 0)
        self.lrna_fee(tkn_add, 0)

        return self

    def remove_liquidity(self, agent: Agent, quantity: float = None, tkn_remove: str = '', nft_id: str = None):
        """
        Remove liquidity from a sub pool.
        If quantity is specified and nft_id is specified, remove specified quantity of shares from specified position.
        If quantity is specified and nft_id is unspecified, remove specified quantity of shares from holdings.
        If quantity is unspecified and nft_id is specified, remove specified position.
        If quantity is unspecified and nft_id is unspecified, remove all liquidity.
        """

        k = (self.unique_id, tkn_remove)
        if nft_id is not None:
            if nft_id not in agent.nfts:
                return self.fail_transaction('Agent does not have liquidity position with specified nft_id.')
            tkn_remove = agent.nfts[nft_id].tkn

        if tkn_remove not in self.asset_list:
            raise AssertionError(f"invalid value for tkn_remove: {tkn_remove}")

        if quantity == 0:
            return self
        # if nft_id is None and not agent.is_holding(k):
        #     return self.fail_transaction('Agent does not have liquidity in this pool.')
        if nft_id is None:
            if quantity is not None and not agent.validate_holdings(k, quantity):
                return self.fail_transaction('Agent does not have enough liquidity in this pool.')
        else:
            if nft_id not in agent.nfts:
                return self.fail_transaction('Agent does not have liquidity position with specified nft_id.')
            elif agent.nfts[nft_id].pool_id != self.unique_id:
                return self.fail_transaction('Specified position is wrong pool.')
            elif agent.nfts[nft_id].tkn != tkn_remove:
                return self.fail_transaction('Specified position is wrong asset.')
            elif quantity is not None and agent.nfts[nft_id].shares < quantity:
                return self.fail_transaction('Agent does not have enough shares in specified position.')

        if self.remove_liquidity_volatility_threshold and self.remove_liquidity_volatility_threshold < float('inf'):
            if self.oracles['price']:
                volatility = abs(
                    self.oracles['price'].price[tkn_remove] / self.current_block.price[tkn_remove] - 1
                )
                if volatility > self.remove_liquidity_volatility_threshold:
                    return self.fail_transaction(
                        f"Withdrawal rejected because the oracle volatility is too high: {volatility} > "
                        f"{self.remove_liquidity_volatility_threshold}"
                    )

        val = self.calculate_remove_liquidity(agent, quantity, tkn_remove, nft_id)
        delta_qa, delta_r, delta_q, delta_s, delta_b, nft_ids = val[:6]

        max_remove = (
                self.max_withdrawal_per_block * self.shares[tkn_remove] - self.current_block.withdrawals[tkn_remove]
        )
        if abs(delta_s) > max_remove:
            return self.fail_transaction(
                f"Transaction rejected because it would exceed the withdrawal limit: {abs(delta_s)} > {max_remove}"
            )

        if delta_r + self.liquidity[tkn_remove] < 0:
            return self.fail_transaction('Cannot remove more liquidity than exists in the pool.')

        self.liquidity[tkn_remove] += delta_r
        self.shares[tkn_remove] += delta_s
        self.protocol_shares[tkn_remove] += delta_b
        self.lrna[tkn_remove] += delta_q

        # distribute tokens to agent
        if delta_qa > 0:
            if 'LRNA' not in agent.holdings:
                agent.holdings['LRNA'] = 0
            agent.holdings['LRNA'] += delta_qa
        if tkn_remove not in agent.holdings:
            agent.holdings[tkn_remove] = 0
        agent.holdings[tkn_remove] -= delta_r

        # remove lp position(s)
        if nft_id is None:
            if quantity is not None:
                agent.holdings[k] -= quantity
                if agent.holdings[k] == 0:
                    agent.share_prices[k] = 0
            else:
                if k in agent.holdings:
                    agent.holdings[k] = 0
                    agent.share_prices[k] = 0
                for nft_id in nft_ids:
                    del agent.nfts[nft_id]
        else:
            if quantity is not None:
                agent.nfts[nft_id].shares -= quantity
            if quantity is None or agent.nfts[nft_id].shares == 0:
                del agent.nfts[nft_id]

        self.current_block.withdrawals[tkn_remove] += abs(delta_s)
        # update fees
        self.asset_fee(tkn_remove, 0)
        self.lrna_fee(tkn_remove, 0)
        return self

    def price(self, tkn: str, numeraire: str = '') -> float:
        """
        price of an asset i denominated in j, according to current market conditions in the omnipool
        """
        if tkn not in self.asset_list + ['LRNA']:
            raise ValueError(f'{tkn} is not in the Omnipool')
        elif tkn == numeraire:
            return 1
        elif not numeraire or numeraire == 'LRNA':
            return self.lrna_price(tkn)
        elif numeraire not in self.asset_list:
            raise ValueError(f'{numeraire} is not in the Omnipool')
        elif tkn == 'LRNA':
            return 1 / self.lrna_price(numeraire)
        elif self.liquidity[tkn] == 0:
            return 0
        return self.lrna[tkn] / self.liquidity[tkn] / self.lrna[numeraire] * self.liquidity[numeraire]

    def usd_price(self, tkn, usd_asset=None):
        if usd_asset is None:
            if self.stablecoin is None:
                raise ValueError('no stablecoin set or provided as argument')
            else:
                usd_asset = self.stablecoin
        if tkn == 'LRNA':
            return 1 / self.lrna_price(usd_asset)
        else:
            return self.lrna_price(tkn) / self.lrna_price(usd_asset)

    def lrna_price(self, tkn: str, fee: float = 0) -> float:
        """Price of i denominated in LRNA"""
        if tkn == "LRNA":
            return 1
        elif self.liquidity[tkn] == 0:
            return 0
        else:
            return (self.lrna[tkn] / self.liquidity[tkn]) * (1 - fee)

    def value_assets(self, assets: dict[str, float], equivalency_map: dict[str, str] = None,
                     numeraire: str = None) -> float:
        # assets is a dict of token: quantity
        # returns the value of the assets in USD
        if numeraire is None:
            if self.stablecoin is None:
                raise ValueError('no stablecoin set or provided as argument')
            else:
                numeraire = self.stablecoin
        if equivalency_map is None:
            equivalency_map = {}
        numeraire_synonyms = [numeraire]
        for eq in equivalency_map:
            if equivalency_map[eq] == 'USD':
                numeraire_synonyms.append(eq)
        value = 0
        for tkn in assets:
            equivalents = [tkn]
            if tkn in equivalency_map:
                equivalents += [equivalency_map[tkn]]
            for eq in equivalency_map:
                if equivalency_map[eq] in equivalents:
                    equivalents.append(eq)
            tkn_value = 0
            for numeraire_tkn in numeraire_synonyms:
                for eq in equivalents:
                    if self.sell_spot(eq, numeraire_tkn) > 0:
                        tkn_value += assets[tkn] * self.price(eq, numeraire_tkn)
                        break
                if tkn_value != 0:
                    break
            value += tkn_value
        return value

    def cash_out(self, agent: Agent, prices: dict[str: float] = None, denomination: str = None) -> float:
        """
        return the value of the agent's holdings if they withdraw all liquidity
        and then sell at current spot prices
        """
        if denomination is None:
            denomination = 'LRNA'
        if prices is None:
            prices = {tkn: self.price(tkn, denomination) for tkn in self.asset_list}

        delta_qa, delta_r, delta_q, delta_s, delta_b, delta_l = 0, {}, {}, {}, {}, 0
        nft_ids = []

        for k in agent.holdings:
            if isinstance(k, tuple) and len(k) == 2 and k[0] == self.unique_id:  # LP shares of correct pool
                tkn = k[1]
                dqa, dr, dq, ds, db, ids = self.calculate_remove_liquidity(agent, tkn_remove=tkn)
                delta_qa += dqa
                delta_r[tkn] = dr + (delta_r[tkn] if tkn in delta_r else 0)
                delta_q[tkn] = dq + (delta_q[tkn] if tkn in delta_q else 0)
                delta_s[tkn] = ds + (delta_s[tkn] if tkn in delta_s else 0)
                delta_b[tkn] = db + (delta_b[tkn] if tkn in delta_b else 0)
                nft_ids += ids

        for nft_id in agent.nfts:
            if nft_id not in nft_ids and agent.nfts[nft_id].pool_id == self.unique_id:
                tkn = agent.nfts[nft_id].tkn
                dqa, dr, dq, ds, db, _ = self.calculate_remove_liquidity(agent, nft_id=nft_id)
                delta_qa += dqa
                delta_r[tkn] = dr + (delta_r[tkn] if tkn in delta_r else 0)
                delta_q[tkn] = dq + (delta_q[tkn] if tkn in delta_q else 0)
                delta_s[tkn] = ds + (delta_s[tkn] if tkn in delta_s else 0)
                delta_b[tkn] = db + (delta_b[tkn] if tkn in delta_b else 0)

        # agent_holdings = new_agent.holdings
        lrna_removed = {tkn: -delta_q[tkn] if tkn in delta_q else 0 for tkn in self.liquidity}
        liquidity_removed = {tkn: -delta_r[tkn] if tkn in delta_r else 0 for tkn in self.liquidity}

        # if 'LRNA' in prices:
        #     raise ValueError('LRNA price should not be given.')
        agent_lrna = delta_qa
        if 'LRNA' in agent.holdings:
            agent_lrna += agent.holdings['LRNA']

        if 'LRNA' not in prices and agent_lrna > 0:
            lrna_total = self.lrna_total - sum(lrna_removed.values())
            lrna_sells = {
                tkn: -(self.lrna[tkn] - lrna_removed[tkn]) / lrna_total * agent_lrna
                for tkn in self.liquidity
            }

            lrna_profits = dict()

            # sell LRNA optimally back to the pool
            for tkn, delta_qa in lrna_sells.items():
                asset_fee = (
                    self.asset_fee(tkn)
                    if hasattr(self, 'asset_fee')
                    else self.last_fee[tkn]
                )
                lrna_profits[tkn] = (
                        -(self.liquidity[tkn] - liquidity_removed[tkn]) * delta_qa
                        / (-delta_qa + self.lrna[tkn] - lrna_removed[tkn])
                        * (1 - asset_fee)
                )
                liquidity_removed[tkn] += lrna_profits[tkn]

        new_holdings = {tkn: agent.holdings[tkn] for tkn in agent.holdings}
        new_holdings['LRNA'] = agent_lrna
        for tkn in liquidity_removed:
            if tkn not in new_holdings:
                new_holdings[tkn] = 0
            new_holdings[tkn] += liquidity_removed[tkn]

        return value_assets(prices, new_holdings)

    def cash_out_mod(self, agent: Agent, prices: dict[str: float] = None, denomination: str = None) -> float:
        """
        return the value of the agent's holdings if they withdraw all liquidity
        and then sell at current spot prices
        """
        if denomination is None:
            denomination = 'LRNA'
        if prices is None:
            prices = {tkn: self.price(tkn, denomination) for tkn in self.asset_list}

        delta_qa, delta_r, delta_q, delta_s, delta_b, delta_l = 0, {}, {}, {}, {}, 0
        nft_ids = []

        for k in agent.holdings:
            if isinstance(k, tuple) and len(k) == 2 and k[0] == self.unique_id:  # LP shares of correct pool
                tkn = k[1]
                dqa, dr, dq, ds, db, ids = self.calculate_remove_liquidity(agent, tkn_remove=tkn)
                delta_qa += dqa
                delta_r[tkn] = dr + (delta_r[tkn] if tkn in delta_r else 0)
                delta_q[tkn] = dq + (delta_q[tkn] if tkn in delta_q else 0)
                delta_s[tkn] = ds + (delta_s[tkn] if tkn in delta_s else 0)
                delta_b[tkn] = db + (delta_b[tkn] if tkn in delta_b else 0)
                nft_ids += ids

        for nft_id in agent.nfts:
            if nft_id not in nft_ids and agent.nfts[nft_id].pool_id == self.unique_id:
                tkn = agent.nfts[nft_id].tkn
                dqa, dr, dq, ds, db, _ = self.calculate_remove_liquidity(agent, nft_id=nft_id)
                delta_qa += dqa
                delta_r[tkn] = dr + (delta_r[tkn] if tkn in delta_r else 0)
                delta_q[tkn] = dq + (delta_q[tkn] if tkn in delta_q else 0)
                delta_s[tkn] = ds + (delta_s[tkn] if tkn in delta_s else 0)
                delta_b[tkn] = db + (delta_b[tkn] if tkn in delta_b else 0)

        # agent_holdings = new_agent.holdings
        lrna_removed = {tkn: -delta_q[tkn] if tkn in delta_q else 0 for tkn in self.asset_list}
        liquidity_removed = {tkn: -delta_r[tkn] if tkn in delta_r else 0 for tkn in self.asset_list}

        # if 'LRNA' in prices:
        #     raise ValueError('LRNA price should not be given.')
        agent_lrna = delta_qa
        if 'LRNA' in agent.holdings:
            agent_lrna += agent.holdings['LRNA']

        if 'LRNA' not in prices and agent_lrna > 0:
            lrna_total = self.lrna_total - sum(lrna_removed.values())
            lrna_sells = {
                tkn: -(self.lrna[tkn] - lrna_removed[tkn]) / lrna_total * agent_lrna
                for tkn in self.liquidity
            }

            lrna_profits = dict()

            # sell LRNA optimally back to the pool
            for tkn, delta_qa in lrna_sells.items():
                lrna_profits[tkn] = (
                        -(self.liquidity[tkn] - liquidity_removed[tkn]) * delta_qa
                        / (-delta_qa + self.lrna[tkn] - lrna_removed[tkn])
                )
                asset_fee = (
                    self.asset_fee(tkn, lrna_profits[tkn])
                    if hasattr(self, 'asset_fee')
                    else self.last_fee[tkn]
                )
                lrna_profits[tkn] *= (1 - asset_fee)
                liquidity_removed[tkn] += lrna_profits[tkn]

        new_holdings = {tkn: agent.holdings[tkn] for tkn in agent.holdings}
        new_holdings['LRNA'] = agent_lrna
        for tkn in liquidity_removed:
            if tkn not in new_holdings:
                new_holdings[tkn] = 0
            new_holdings[tkn] += liquidity_removed[tkn]

        return value_assets(prices, new_holdings)

    def calculate_trade_to_price(self, tkn: str, target_price: float):
        """
        target_price should be the price in LRNA
        """
        if tkn not in self.liquidity:
            return 0
        k = self.lrna[tkn] * self.liquidity[tkn]
        target_x = math.sqrt(k / target_price)
        dx = target_x - self.liquidity[tkn]
        if dx < 0:
            # TODO: implement version that subtracts fees from output
            dx *= (1 - self.asset_fee(tkn, dx))
        return dx

    def trade_to_price(self, agent: Agent, tkn: str, target_price: float, other_tkn: str = None):
        """
        trade with agent to move price of tkn to target_price (in LRNA)
        """
        if other_tkn is None:
            other_tkn = 'LRNA'
        dx = self.calculate_trade_to_price(tkn, target_price)
        if dx > 0:
            self.swap(agent=agent, tkn_sell=tkn, tkn_buy=other_tkn, sell_quantity=dx)
        elif dx < 0:
            self.swap(agent=agent, tkn_buy=tkn, tkn_sell=other_tkn, buy_quantity=-dx)
        return self


class OmnipoolLiquidityPosition:
    def __init__(self, tkn: str, price: float, shares: float, delta_r: float, pool_id: str = None):
        self.tkn = tkn
        self.price = price
        self.shares = shares
        self.delta_r = delta_r
        self.pool_id = pool_id

    def copy(self):
        return OmnipoolLiquidityPosition(self.tkn, self.price, self.shares, self.delta_r, self.pool_id)


class OmnipoolArchiveState:
    def __init__(self, state: OmnipoolState):
        self.asset_list = [tkn for tkn in state.asset_list] + ['LRNA']
        self.liquidity = {k: v for (k, v) in state.liquidity.items()}
        self.lrna = {k: v for (k, v) in state.lrna.items()}
        self.lrna_total = sum(state.lrna.values())
        self.shares = {k: v for (k, v) in state.shares.items()}
        self.protocol_shares = {k: v for (k, v) in state.protocol_shares.items()}
        self.fail = state.fail
        self.stablecoin = state.stablecoin
        self.oracles = {k: OracleArchiveState(v) for (k, v) in state.oracles.items()}
        self.unique_id = state.unique_id
        self.volume_in = state.current_block.volume_in
        self.volume_out = state.current_block.volume_out
        # record these for analysis later
        self.last_fee = {k: v for (k, v) in state.last_fee.items()}
        self.last_lrna_fee = {k: v for (k, v) in state.last_lrna_fee.items()}
        # borrow some methods from parent
        self.lrna_price = OmnipoolState.lrna_price
        self.price = OmnipoolState.price
        self.usd_price = OmnipoolState.usd_price


def asset_invariant(state: OmnipoolState, i: str) -> float:
    """Invariant for specific asset"""
    return state.liquidity[i] * state.lrna[i]


def swap_lrna_delta_Qi(state: OmnipoolState, delta_ri: float, i: str) -> float:
    return state.lrna[i] * (- delta_ri / (state.liquidity[i] + delta_ri))


def swap_lrna_delta_Ri(state: OmnipoolState, delta_qi: float, i: str) -> float:
    return state.liquidity[i] * (- delta_qi / (state.lrna[i] + delta_qi))


def weight_i(state: OmnipoolState, i: str) -> float:
    return state.lrna[i] / state.lrna_total


def simulate_swap(
        old_state: OmnipoolState,
        old_agent: Agent,
        tkn_buy: str,
        tkn_sell: str,
        buy_quantity: float = 0,
        sell_quantity: float = 0
) -> tuple[OmnipoolState, Agent]:
    """
    execute swap on a copy of old_state and old_agent, and return the copies
    """
    new_state = old_state.copy()
    new_agent = old_agent.copy()

    new_state.swap(
        agent=new_agent,
        sell_quantity=sell_quantity,
        buy_quantity=buy_quantity,
        tkn_buy=tkn_buy,
        tkn_sell=tkn_sell,
    )
    return new_state, new_agent


def simulate_add_liquidity(
        old_state: OmnipoolState,
        old_agent: Agent,
        quantity: float = 0,
        tkn_add: str = '',
        nft_id: str = None
) -> tuple[OmnipoolState, Agent]:
    """Copy state, then add liquidity and return new state"""
    new_state = old_state.copy()
    new_agent = old_agent.copy()

    new_state.add_liquidity(new_agent, quantity, tkn_add, nft_id)
    return new_state, new_agent


def simulate_remove_liquidity(
        old_state: OmnipoolState,
        old_agent: Agent,
        quantity: float = None,
        tkn_remove: str = '',
        nft_id: str = None
) -> tuple[OmnipoolState, Agent]:
    """Compute new state after liquidity removal"""
    new_state = old_state.copy()
    new_agent = old_agent.copy()

    new_state.remove_liquidity(new_agent, quantity, tkn_remove, nft_id)
    return new_state, new_agent


def value_assets(prices: dict, assets: dict) -> float:
    """
    return the value of the agent's assets if they were sold at current spot prices
    """
    return sum([
        assets[i] * prices[i] if i in prices else 0
        for i in assets.keys()
    ])

