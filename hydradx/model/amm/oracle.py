from .exchange import Exchange


class Block:
    def __init__(self, input_state: Exchange):
        self.liquidity = {tkn: input_state.liquidity[tkn] for tkn in input_state.liquidity}
        self.price = {tkn: input_state.price(tkn) for tkn in input_state.liquidity}
        self.volume_in = {tkn: 0 for tkn in input_state.liquidity}
        self.volume_out = {tkn: 0 for tkn in input_state.liquidity}
        self.withdrawals = {tkn: 0 for tkn in input_state.liquidity}
        self.lps = {tkn: 0 for tkn in input_state.liquidity}
        self.asset_list = input_state.asset_list.copy()


class Oracle:
    def __init__(self, first_block: Block = None, decay_factor: float = 0, sma_equivalent_length: int = 0,
                 last_values: dict = None):
        if decay_factor:
            self.decay_factor = decay_factor
        elif sma_equivalent_length:
            self.decay_factor = 2 / (sma_equivalent_length + 1)
        else:
            raise ValueError('Either decay_factor or sma_equivalent_length must be specified')
        self.length = sma_equivalent_length or 2 / self.decay_factor - 1
        if last_values is not None:
            self.asset_list = list((last_values['liquidity']).keys())
            self.liquidity = {k: v for (k, v) in last_values['liquidity'].items()}
            self.price = {k: v for (k, v) in last_values['price'].items()}
            self.volume_in = {k: v for (k, v) in last_values['volume_in'].items()}
            self.volume_out = {k: v for (k, v) in last_values['volume_out'].items()}
        elif first_block is not None:
            self.asset_list = first_block.asset_list
            self.liquidity = first_block.liquidity
            self.price = first_block.price
            self.volume_in = first_block.volume_in
            self.volume_out = first_block.volume_out
        else:
            raise ValueError('Either last_values or first_block must be specified')
        self.age = 0

    def add_asset(self, tkn: str, liquidity: float):
        self.liquidity[tkn] = liquidity
        self.volume_in[tkn] = 0
        self.volume_out[tkn] = 0
        self.asset_list.append(tkn)

    def update(self, block: Block):
        self.age += 1
        for tkn in block.liquidity:
            self.liquidity[tkn] = (
                (1 - self.decay_factor) * self.liquidity[tkn] + self.decay_factor * block.liquidity[tkn]
            ) if tkn in self.liquidity else block.liquidity[tkn]
            self.price[tkn] = (
                (1 - self.decay_factor) * self.price[tkn] + self.decay_factor * block.price[tkn]
            ) if tkn in self.price else block.price[tkn]
            self.volume_in[tkn] = (
                (1 - self.decay_factor) * self.volume_in[tkn] + self.decay_factor * block.volume_in[tkn]
            ) if tkn in self.volume_in else block.volume_in[tkn]
            self.volume_out[tkn] = (
                (1 - self.decay_factor) * self.volume_out[tkn] + self.decay_factor * block.volume_out[tkn]
            ) if tkn in self.volume_out else block.volume_out[tkn]
        return self


class OracleArchiveState:
    def __init__(self, oracle: Oracle):
        self.liquidity = {tkn: oracle.liquidity[tkn] for tkn in oracle.liquidity}
        self.price = {tkn: oracle.price[tkn] for tkn in oracle.price}
        self.volume_in = {tkn: oracle.volume_in[tkn] for tkn in oracle.volume_in}
        self.volume_out = {tkn: oracle.volume_out[tkn] for tkn in oracle.volume_out}
        self.age = oracle.age
