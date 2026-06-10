"""
M1 — HOLLAR peg & HSM drain: model components and step loop.

World view (one floating price):
  Every counter-asset in a HOLLAR venue is externally elastic — aUSDT/aUSDC
  redeem 1:1, sUSDS/sUSDe/apyUSD/aEURC have external NAV, PRIME mints/redeems
  at oracle NAV with deep Solana liquidity. HOLLAR is the only asset with no
  external sink except the HSM. So arbitrageurs equalize the marginal USD
  price of HOLLAR across all venues; the system behaves like one aggregate
  HOLLAR/USD market whose level p_H is set by HOLLAR conservation, and the
  HSM is the only mechanism that actually burns HOLLAR (buyback, ≤ max_buy
  price, rate-limited per block, spending finite collateral inventory) or
  mints it (sell side, which caps p_H near 1 from above).

Per step (N blocks):
  1. Propeller flow: deploy sells HOLLAR→PRIME into pool 143 in tranches,
     enforcing the contract's min-out (HOLLAR priced at FIXED $1, PRIME at
     oracle NAV, band = dcaSlippagePpm). If the pool can't fill at the band,
     the tranche is SKIPPED (the on-chain ramp stalls) — unwind is the mirror.
  2. Arb equalization: find p_H s.t. arbs moving HOLLAR between venues (and
     counter-assets to/from external markets) equalize prices; net HOLLAR
     across pools conserved. Pools within `arb_threshold` of p_H untouched.
  3. HSM: per collateral pool, the pallet's execute_arbitrage (rate-limited
     buyback if pool HOLLAR-heavy & price ≤ coef; mint side if HOLLAR-poor).
"""
import sys, os, math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.apps.propeller.m1_state import (
    build_pools, build_hsm, USD_NAV, OMNIPOOL_HOLLAR, OMNIPOOL_LRNA, hollar_usd_price,
)

BLOCKS_PER_DAY = 14_400  # 6s blocks


class OmnipoolVenue:
    """HOLLAR/LRNA leg of the Omnipool as a constant-product venue with fee."""

    def __init__(self, hollar: float, lrna: float, hollar_usd0: float, fee: float = 0.0025):
        self.hollar = hollar
        self.lrna = lrna
        self.fee = fee
        self.lrna_usd = hollar_usd0 * hollar / lrna  # anchor LRNA so t0 price matches

    def price(self) -> float:  # marginal USD price of HOLLAR
        return (self.lrna / self.hollar) * self.lrna_usd

    def delta_hollar_for_price(self, p: float) -> float:
        """HOLLAR amount the pool must GAIN (+) / LOSE (−) for price → p."""
        target_h = math.sqrt(self.hollar * self.lrna * self.lrna_usd / p)
        return target_h - self.hollar

    def apply_delta(self, dh: float):
        if dh == 0:
            return 0.0
        k = self.hollar * self.lrna
        new_h = self.hollar + dh
        new_l = k / new_h
        d_lrna = self.lrna - new_l  # LRNA out (+) if HOLLAR sold in
        self.hollar, self.lrna = new_h, new_l
        return d_lrna * self.lrna_usd  # USD value arb receives (+) or pays (−)


class PropellerFlow:
    """Propeller's HOLLAR flow into/out of pool 143 with the contract's min-out stall."""

    def __init__(self, pool_143: StableSwapPoolState, slippage_band: float = 0.01,
                 prime_nav: float = USD_NAV['PRIME']):
        self.pool = pool_143
        self.band = slippage_band
        self.prime_nav = prime_nav
        self.deployed = 0.0       # cumulative HOLLAR sold into 143
        self.unwound = 0.0        # cumulative HOLLAR bought back
        self.stalled_steps = 0
        self.agent = Agent(enforce_holdings=False)

    def deploy(self, hollar_amount: float, tranche: float = 5_000.0) -> float:
        """Sell `hollar_amount` HOLLAR → PRIME in tranches; returns amount actually sold."""
        sold = 0.0
        while sold < hollar_amount - 1e-9:
            amt = min(tranche, hollar_amount - sold)
            # contract min-out: HOLLAR @ $1 fixed, PRIME @ oracle NAV, band
            fair_prime = amt * 1.0 / self.prime_nav
            min_out = fair_prime * (1 - self.band)
            received = self.pool.calculate_buy_from_sell(tkn_buy='PRIME', tkn_sell='HOLLAR', sell_quantity=amt)
            if received < min_out:
                self.stalled_steps += 1
                break  # ramp stalls this step
            self.pool.swap(self.agent, tkn_buy='PRIME', tkn_sell='HOLLAR', sell_quantity=amt)
            sold += amt
        self.deployed += sold
        return sold

    def unwind(self, hollar_amount: float, tranche: float = 5_000.0) -> float:
        """Sell PRIME → HOLLAR (loop unwind); min-out mirrors pokeRepay."""
        bought = 0.0
        while bought < hollar_amount - 1e-9:
            amt = min(tranche, hollar_amount - bought)
            prime_in = amt * 1.0 / self.prime_nav  # PRIME to sell for ~amt HOLLAR at fair
            min_out = amt * (1 - self.band)
            received = self.pool.calculate_buy_from_sell(tkn_buy='HOLLAR', tkn_sell='PRIME', sell_quantity=prime_in)
            if received < min_out:
                self.stalled_steps += 1
                break
            self.pool.swap(self.agent, tkn_buy='HOLLAR', tkn_sell='PRIME', sell_quantity=prime_in)
            bought += received
        self.unwound += bought
        return bought


class M1World:
    def __init__(self, hsm_holdings='current', arb_threshold: float = 0.001,
                 slippage_band: float = 0.01, omnipool_fee: float = 0.0025):
        self.pools = build_pools()
        self.hsm = build_hsm(self.pools, holdings=hsm_holdings)
        self.flow = PropellerFlow(self.pools['2-Pool-PRIME'], slippage_band=slippage_band)
        self.arb_threshold = arb_threshold
        self.arb_agent = Agent(enforce_holdings=False)
        p0 = self._median_price()
        self.omnipool = OmnipoolVenue(OMNIPOOL_HOLLAR, OMNIPOOL_LRNA, p0, fee=omnipool_fee)
        self.hollar_burned = 0.0
        self.hollar_minted = 0.0
        self.block = 0

    # ── prices ────────────────────────────────────────────────────────────
    def venue_prices(self) -> dict:
        out = {uid: hollar_usd_price(p) for uid, p in self.pools.items()}
        out['omnipool'] = self.omnipool.price()
        return out

    def _median_price(self) -> float:
        ps = sorted(hollar_usd_price(p) for p in self.pools.values())
        return ps[len(ps) // 2]

    def market_price(self) -> float:
        """Depth-weighted HOLLAR price across venues."""
        tot_w, tot = 0.0, 0.0
        for uid, p in self.pools.items():
            w = self.pools[uid].liquidity['HOLLAR']
            tot += hollar_usd_price(self.pools[uid]) * w
            tot_w += w
        tot += self.omnipool.price() * self.omnipool.hollar
        tot_w += self.omnipool.hollar
        return tot / tot_w

    # ── arb equalization (pairwise, HOLLAR-conserving by construction) ───
    def _venue_price(self, uid: str) -> float:
        if uid == 'omnipool':
            return self.omnipool.price()
        return hollar_usd_price(self.pools[uid])

    def _venue_hollar(self, uid: str) -> float:
        return self.omnipool.hollar if uid == 'omnipool' else self.pools[uid].liquidity['HOLLAR']

    def _move_hollar(self, src: str, dst: str, x: float):
        """Arb buys exactly x HOLLAR from `src` (paying counter), sells the same x into `dst`."""
        if src == 'omnipool':
            self.omnipool.apply_delta(-x)
        else:
            pool = self.pools[src]
            tkn = [t for t in pool.asset_list if t != 'HOLLAR'][0]
            pool.swap(self.arb_agent, tkn_buy='HOLLAR', tkn_sell=tkn, buy_quantity=x)
            if pool.fail:
                pool.fail = ''
                return False
        if dst == 'omnipool':
            self.omnipool.apply_delta(x)
        else:
            pool = self.pools[dst]
            tkn = [t for t in pool.asset_list if t != 'HOLLAR'][0]
            pool.swap(self.arb_agent, tkn_buy=tkn, tkn_sell='HOLLAR', sell_quantity=x)
            if pool.fail:  # roll back the source leg
                pool.fail = ''
                if src == 'omnipool':
                    self.omnipool.apply_delta(x)
                else:
                    spool = self.pools[src]
                    stkn = [t for t in spool.asset_list if t != 'HOLLAR'][0]
                    spool.swap(self.arb_agent, tkn_buy=stkn, tkn_sell='HOLLAR', sell_quantity=x)
                    spool.fail = ''
                return False
        return True

    def equalize(self, max_iters: int = 400):
        """Pairwise arbitrage until marginal HOLLAR USD prices agree within threshold.

        Each iteration moves x HOLLAR from the cheapest venue to the dearest —
        exactly what a profit-seeking arb does (profit accrues in counter-assets,
        sourced/sunk externally). HOLLAR units are conserved exactly.
        """
        uids = list(self.pools.keys()) + ['omnipool']
        for _ in range(max_iters):
            prices = {u: self._venue_price(u) for u in uids}
            cheap = min(prices, key=prices.get)
            dear = max(prices, key=prices.get)
            gap = prices[dear] - prices[cheap]
            # arbs need to clear both legs' fees; threshold on top
            fee_floor = 0.0008 + self.arb_threshold
            if gap <= fee_floor:
                break
            # step: a slice of the smaller venue, scaled down as the gap closes
            x = min(self._venue_hollar(cheap), self._venue_hollar(dear)) * min(0.05, gap * 10)
            if x < 1.0:
                break
            if not self._move_hollar(cheap, dear, x):
                break

    # ── HSM ───────────────────────────────────────────────────────────────
    def hsm_step(self, n_blocks: int):
        """Run the pallet's per-block arbitrage n_blocks times (rate-limit faithful)."""
        for tkn in self.hsm.asset_list:
            inv_before = self.hsm.liquidity[tkn]
            for _ in range(n_blocks):
                hollar_before = self.hsm.pools[tkn].liquidity['HOLLAR']
                self.hsm.arb(self.arb_agent, tkn)
                self.hsm.update()
                d = hollar_before - self.hsm.pools[tkn].liquidity['HOLLAR']
                if abs(d) < 1e-12:
                    break  # nothing more this imbalance level can do; skip remaining blocks
                if d > 0:
                    self.hollar_burned += d
                else:
                    self.hollar_minted += -d

    # ── step ──────────────────────────────────────────────────────────────
    def step(self, n_blocks: int, deploy_hollar: float = 0.0, unwind_hollar: float = 0.0,
             tranche: float = 5_000.0):
        if deploy_hollar > 0:
            self.flow.deploy(deploy_hollar, tranche=tranche)
        if unwind_hollar > 0:
            self.flow.unwind(unwind_hollar, tranche=tranche)
        self.equalize()
        self.hsm_step(n_blocks)
        self.equalize()
        self.block += n_blocks

    def snapshot(self) -> dict:
        prices = self.venue_prices()
        return dict(
            block=self.block,
            day=self.block / BLOCKS_PER_DAY,
            market_price=self.market_price(),
            min_price=min(prices.values()),
            prices=prices,
            hsm_inventory={t: self.hsm.liquidity[t] for t in self.hsm.asset_list},
            hsm_inventory_usd=sum(self.hsm.liquidity[t] * USD_NAV[t] for t in self.hsm.asset_list),
            hollar_burned=self.hollar_burned,
            hollar_minted=self.hollar_minted,
            deployed=self.flow.deployed,
            unwound=self.flow.unwound,
            stalled=self.flow.stalled_steps,
            pool143_hollar_share=self.pools['2-Pool-PRIME'].liquidity['HOLLAR'] /
                (self.pools['2-Pool-PRIME'].liquidity['HOLLAR'] +
                 self.pools['2-Pool-PRIME'].liquidity['PRIME'] * USD_NAV['PRIME']),
        )
