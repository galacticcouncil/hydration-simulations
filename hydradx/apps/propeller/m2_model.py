"""
M2 — Propeller loop solvency & backing-gap model.

Mirrors the propeller-vault Solidity state machine (money-market@propeller):
  CollateralVault: deposit (borrow at max LTV, synth = debt/synthLT ×1.005,
    seed loop), maintainPeg (re-top synth to cover accrued debt), redemption
    queue (snapshot debtShare incl. accrued interest; settle from freed equity;
    FIFO; partial release pro-rata to repaid debt) — F1 behavior switchable.
  SubLoop: pokeBorrow ramp (borrow to deployHfFloor, capped deployTranche,
    swap HOLLAR→PRIME at pool-143 execution), pokeRepay spiral (HF-safe sliver,
    STEP_HF_FLOOR 1.02, capped unwindTranche), deLever (fires at HF ≤ 1.10,
    x = (targetHf·debt − lt·coll)/(targetHf − lt)), harvest (skim
    equity − principalEquity − unwindTargetEquity once above threshold).
  Aave: loop liquidation when HF < 1 (close factor 50%, liq bonus 7%);
    Main position liquidatable ONLY if the synth floor is broken
    (synth·LT < debt — e.g. maintainPeg outage) AND collateral crashes.

The solvency number: backing_gap = Σ MainDebt − loopEquity (USD).

PRIME process: price = NAV(t) × (1 − discount(t)); NAV drifts up at the
exogenous HELOC yield; `discount` is the market price-vs-NAV gap (depeg).
The oracle (PRIMEoracleMRL mirror) tracks the MARKET price, so loop HF and
swap min-outs see the depeg. Pool-143 execution comes from the M1 stableswap
state (repegged to oracle each step); the PRIME side is externally elastic.

Units: USD throughout. HOLLAR assumed $1 on the Main ledger (its own peg is
M1's subject); blocks of 6s; steps of `step_blocks`.
"""
import sys, os, math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.stableswap_amm import StableSwapPoolState

BLOCKS_PER_DAY = 14_400
BLOCK_SEC = 6.0

# Aave reserve params (mainnet-verified, spec §3)
PRIME_LT = 0.88
PRIME_LIQ_BONUS = 0.07
CLOSE_FACTOR = 0.50
SYNTH_LT = 0.98
SYNTH_BUFFER = 1.005
ETH_LTV, ETH_LT = 0.75, 0.85

TARGET_HF = 1.05          # deployHfFloor
DELEVER_TRIGGER = 1.10    # spec band (deLever() HealthyEnough gate)
STEP_HF_FLOOR = 1.02      # pokeRepay sliver floor


class Pool143:
    """Pool-143 execution venue, repegged to the PRIME oracle; PRIME side elastic."""

    def __init__(self, hollar=703_162.0, prime=344_626.0, prime_price=1.037, fee=0.0004, amp=100):
        self.state = StableSwapPoolState(
            tokens={'HOLLAR': hollar, 'PRIME': prime}, amplification=amp,
            trade_fee=fee, peg=[prime_price], unique_id='2-Pool-PRIME')
        self.agent = Agent(enforce_holdings=False)
        self.prime_price = prime_price

    def repeg(self, prime_price: float):
        # the pool converges peg → peg_target inside swaps (_calculate_new_peg);
        # the oracle drives the TARGET (pallet: PegSource = MMOracle)
        self.state.update()  # advance pool time step
        self.state.set_peg_target([prime_price])
        self.prime_price = prime_price

    def rebalance_external(self, threshold=0.001):
        """Arbs restore the pool toward 50/50 value via external PRIME mint/redeem."""
        h = self.state.liquidity['HOLLAR']
        p_usd = self.state.liquidity['PRIME'] * self.prime_price
        skew = h - p_usd
        if abs(skew) / (h + p_usd) < threshold:
            return
        if skew > 0:   # HOLLAR-heavy: arb buys HOLLAR with freshly-minted PRIME
            self.state.swap(self.agent, tkn_buy='HOLLAR', tkn_sell='PRIME', buy_quantity=skew / 2)
        else:          # PRIME-heavy: arb buys PRIME (redeems externally at NAV)
            self.state.swap(self.agent, tkn_buy='PRIME', tkn_sell='HOLLAR', buy_quantity=(-skew / 2) / self.prime_price)
        self.state.fail = ''

    def sell_hollar(self, amount: float) -> float:
        """HOLLAR → PRIME; returns PRIME out (or 0 on failure)."""
        out = self.state.calculate_buy_from_sell(tkn_buy='PRIME', tkn_sell='HOLLAR', sell_quantity=amount)
        self.state.swap(self.agent, tkn_buy='PRIME', tkn_sell='HOLLAR', sell_quantity=amount)
        if self.state.fail:
            self.state.fail = ''
            return 0.0
        return out

    def sell_prime(self, amount: float) -> float:
        """PRIME → HOLLAR; returns HOLLAR out (or 0 on failure)."""
        out = self.state.calculate_buy_from_sell(tkn_buy='HOLLAR', tkn_sell='PRIME', sell_quantity=amount)
        self.state.swap(self.agent, tkn_buy='HOLLAR', tkn_sell='PRIME', sell_quantity=amount)
        if self.state.fail:
            self.state.fail = ''
            return 0.0
        return out


class PropellerM2:
    """One collateral vault + the shared loop, faithful flows, USD ledger."""

    def __init__(self,
                 deposit_usd=5_000_000.0,
                 prime_yield=0.0634,        # exogenous HELOC APR (NAV drift)
                 hollar_rate=0.044,         # governance-set borrow APR
                 deploy_tranche=5_000.0,
                 unwind_tranche=5_000.0,
                 pokes_per_step=None,       # keeper pokes per step (None = every block)
                 harvest_threshold=0.005,   # skim once surplus > 0.5% of basis
                 slippage_band=0.01,
                 f1_fixed=False,            # True: harvest reserves accrued Main interest
                 ltv=ETH_LTV):
        self.pool = Pool143()
        self.prime_nav = 1.037
        self.prime_discount = 0.0          # market price-vs-NAV gap
        self.prime_yield = prime_yield
        self.hollar_rate = hollar_rate
        self.band = slippage_band
        self.deploy_tranche = deploy_tranche
        self.unwind_tranche = unwind_tranche
        self.pokes_per_step = pokes_per_step
        self.harvest_threshold = harvest_threshold
        self.f1_fixed = f1_fixed

        # ── Main position (vault) ──
        self.collateral_usd = deposit_usd
        self.main_debt = deposit_usd * ltv           # HOLLAR, accrues hollar_rate
        self.synth = self.main_debt / SYNTH_LT * SYNTH_BUFFER
        self.synth_floor_ok = True

        # ── SubLoop ──
        self.loop_prime = 0.0          # aPRIME units
        self.loop_debt = 0.0           # HOLLAR
        self.pending_deploy = self.main_debt   # seed HOLLAR waiting to enter the loop
        self.principal_equity = self.main_debt # cost basis (seed) — shrinks on unwind
        self.unwind_target = 0.0       # equity HOLLAR the spiral must free
        self.freed = 0.0               # HOLLAR freed for the vault
        self.delever_debt_target = 0.0
        self.loop_idle = 0.0           # idle HOLLAR sitting in the loop (contract: folded into next avail)
        self.compounded = 0.0          # harvested carry supplied as collateral (USD)
        self.harvest_skimmed_hollar = 0.0
        self.interest_reserve = 0.0    # F1-fixed: carry earmarked for Main accrued interest
        self.liquidated_loss = 0.0     # equity destroyed by loop liquidations
        self.liquidation_events = 0
        self.block = 0

    # ── prices / health ──────────────────────────────────────────────────
    @property
    def prime_price(self):
        return self.prime_nav * (1 - self.prime_discount)

    def loop_collateral_usd(self):
        return self.loop_prime * self.prime_price  # oracle = market

    def loop_hf(self):
        if self.loop_debt <= 0:
            return float('inf')
        return self.loop_collateral_usd() * PRIME_LT / self.loop_debt

    def loop_equity(self):
        return max(self.loop_collateral_usd() - self.loop_debt, 0.0) + self.pending_deploy + self.freed + self.loop_idle

    def main_hf(self, collateral_price_mult=1.0):
        if self.main_debt <= 0:
            return float('inf')
        synth_part = self.synth * SYNTH_LT if self.synth_floor_ok else self.synth * SYNTH_LT
        return ((self.collateral_usd * collateral_price_mult) * ETH_LT + synth_part) / self.main_debt

    def backing_gap(self):
        """Σ MainDebt − loop equity attributable (USD). Positive = unbacked HOLLAR."""
        return self.main_debt - (self.loop_equity() + self.interest_reserve)

    # ── keeper ops (per the contracts) ───────────────────────────────────
    def poke_borrow(self):
        """Borrow to deployHfFloor (capped deployTranche), swap HOLLAR→PRIME in 143."""
        # first, deploy pending seed
        budget = 0.0
        coll_lt = self.loop_collateral_usd() * PRIME_LT
        max_debt = coll_lt / TARGET_HF
        if max_debt > self.loop_debt:
            budget = min(max_debt - self.loop_debt, self.deploy_tranche)
        spend = 0.0
        if self.pending_deploy > 0:
            spend = min(self.pending_deploy, self.deploy_tranche)
        elif budget > 0:
            spend = budget
        if spend <= 0:
            return False
        # contract min-out: HOLLAR @ $1, PRIME @ oracle (market) price
        fair_prime = spend / self.prime_price
        out = self.pool.sell_hollar(spend)
        if out < fair_prime * (1 - self.band) or out == 0.0:
            return False  # stalled (swap reverted on min-out)
        if self.pending_deploy > 0:
            self.pending_deploy -= spend
        else:
            self.loop_debt += spend
        self.loop_prime += out
        return True

    def poke_repay(self):
        """Unwind spiral step: HF-safe aPRIME sliver → HOLLAR; repay/credit."""
        if self.unwind_target <= 0 and self.delever_debt_target <= 0:
            return False
        coll = self.loop_collateral_usd()
        if self.loop_debt > 0:
            min_coll = STEP_HF_FLOOR * self.loop_debt / PRIME_LT
            headroom = max(coll - min_coll, 0.0) * 0.9
        else:
            headroom = coll
        sell_usd = min(headroom, self.unwind_tranche)
        if sell_usd <= 0:
            return False
        prime_amt = min(sell_usd / self.prime_price, self.loop_prime)
        fair_hollar = prime_amt * self.prime_price
        out = self.pool.sell_prime(prime_amt)
        if out < fair_hollar * (1 - self.band) or out == 0.0:
            return False
        self.loop_prime -= prime_amt
        out += self.loop_idle  # contract: idle balance folds into avail
        self.loop_idle = 0.0
        # safety de-lever first: full proceeds repay debt
        if self.delever_debt_target > 0:
            repay = min(out, self.delever_debt_target, self.loop_debt)
            self.loop_debt -= repay
            self.delever_debt_target -= repay
            out -= repay
            if out <= 0:
                return True
            if self.unwind_target <= 0:
                self.loop_idle += out  # park the residual (contract keeps it as balance)
                return True
        # proportional shrink: repay debt slice, free the equity remainder
        pre_coll = self.loop_collateral_usd() + out
        repay = min(out * (self.loop_debt / pre_coll) if pre_coll > 0 else 0.0, self.loop_debt)
        self.loop_debt -= repay
        freed = out - repay
        credited = min(freed, self.unwind_target)
        self.freed += credited
        self.unwind_target -= credited
        self.loop_idle += freed - credited  # rounding/overshoot stays as balance
        return True

    def de_lever_check(self):
        # SubLoop.deLever gates: HealthyEnough if hf > trigger OR hf >= targetHf
        # (SubLoop.sol:513,518) — so it only acts when hf < targetHf (1.05).
        hf = self.loop_hf()
        if hf > DELEVER_TRIGGER or hf >= TARGET_HF:
            return
        if self.loop_debt > 0:
            coll = self.loop_collateral_usd()
            lt = PRIME_LT
            if TARGET_HF <= lt:
                return
            x = (TARGET_HF * self.loop_debt - lt * coll) / (TARGET_HF - lt)
            if x > self.delever_debt_target:
                self.delever_debt_target = x

    def harvest(self):
        equity = self.loop_equity()
        reserved = self.principal_equity + self.unwind_target
        if self.f1_fixed:
            # fix: reserve the accrued Main interest before skimming
            reserved += max(self.main_debt - self.principal_equity, 0.0) - self.interest_reserve
        surplus = equity - reserved
        if surplus <= 0 or surplus < self.principal_equity * self.harvest_threshold:
            return 0.0
        if self.f1_fixed:
            accrued = max(self.main_debt - self.principal_equity, 0.0)
            top_up = min(surplus, max(accrued - self.interest_reserve, 0.0))
            self.interest_reserve += top_up
            surplus -= top_up
            if surplus <= 0:
                return 0.0
        # withdraw surplus PRIME at oracle, swap to collateral (~50bp compound cost)
        prime_amt = min(surplus / self.prime_price, self.loop_prime)
        self.loop_prime -= prime_amt
        usd = prime_amt * self.prime_price * (1 - 0.005)
        self.compounded += usd
        self.collateral_usd += usd
        self.harvest_skimmed_hollar += surplus
        return usd

    def maintain_peg(self, keeper_online=True):
        need = self.main_debt / SYNTH_LT * SYNTH_BUFFER
        if keeper_online and need > self.synth:
            self.synth = need
        self.synth_floor_ok = self.synth * SYNTH_LT >= self.main_debt

    def aave_liquidate_loop(self):
        """While loop HF < 1: liquidator repays close-factor of debt, takes PRIME +7%."""
        while self.loop_debt > 0 and self.loop_hf() < 1.0:
            repay = self.loop_debt * CLOSE_FACTOR
            seized_usd = repay * (1 + PRIME_LIQ_BONUS)
            seized_prime = min(seized_usd / self.prime_price, self.loop_prime)
            self.loop_prime -= seized_prime
            self.loop_debt -= repay
            self.liquidated_loss += repay * PRIME_LIQ_BONUS
            self.liquidation_events += 1
            if self.loop_prime <= 0:
                break

    # ── step ─────────────────────────────────────────────────────────────
    def step(self, n_blocks: int, keeper_online=True, harvest_now=False,
             prime_discount=None, prime_yield=None, hollar_rate=None):
        if prime_discount is not None:
            self.prime_discount = prime_discount
        if prime_yield is not None:
            self.prime_yield = prime_yield
        if hollar_rate is not None:
            self.hollar_rate = hollar_rate
        dt = n_blocks / BLOCKS_PER_DAY / 365.0
        # accruals: NAV drift + interest on every debt leg
        self.prime_nav *= math.exp(self.prime_yield * dt)
        self.main_debt *= math.exp(self.hollar_rate * dt)
        self.loop_debt *= math.exp(self.hollar_rate * dt)
        self.pool.repeg(self.prime_price)
        self.pool.rebalance_external()
        # liquidation check happens regardless of keeper
        self.aave_liquidate_loop()
        if keeper_online:
            pokes = self.pokes_per_step if self.pokes_per_step is not None else n_blocks
            for _ in range(pokes):
                acted = self.poke_repay() if (self.unwind_target > 0 or self.delever_debt_target > 0) \
                    else self.poke_borrow()
                if not acted:
                    break
                self.pool.rebalance_external()
            self.de_lever_check()
            if harvest_now:
                self.harvest()
        self.maintain_peg(keeper_online)
        self.block += n_blocks

    def request_full_redeem(self):
        """Redeem 100%: snapshot debtShare (with accrued interest), unwind all loop equity."""
        equity = max(self.loop_collateral_usd() - self.loop_debt, 0.0)
        self.unwind_target = equity + self.pending_deploy
        self.freed += self.pending_deploy
        self.unwind_target -= self.pending_deploy
        self.pending_deploy = 0.0
        self.principal_equity = 0.0
        return self.main_debt  # debtShare snapshot

    def settle_redeem(self, debt_share: float):
        """pokeSettle semantics: repay from freed (+ interest_reserve if F1-fixed);
        release collateral pro-rata. Returns (collateral_out, stuck_fraction)."""
        available = self.freed + (self.interest_reserve if self.f1_fixed else 0.0)
        repaid = min(available, debt_share)
        frac = repaid / debt_share if debt_share > 0 else 1.0
        out = self.collateral_usd * frac
        return out, 1.0 - frac

    def snapshot(self):
        return dict(
            day=self.block / BLOCKS_PER_DAY,
            prime_nav=self.prime_nav, prime_price=self.prime_price,
            loop_hf=self.loop_hf(), main_debt=self.main_debt,
            loop_prime_usd=self.loop_collateral_usd(), loop_debt=self.loop_debt,
            loop_equity=self.loop_equity(), backing_gap=self.backing_gap(),
            compounded=self.compounded, interest_reserve=self.interest_reserve,
            synth_floor_ok=self.synth_floor_ok,
            liquidation_events=self.liquidation_events, liquidated_loss=self.liquidated_loss,
            leverage=(self.loop_collateral_usd() / max(self.loop_collateral_usd() - self.loop_debt, 1e-9))
            if self.loop_prime > 0 else 0.0,
        )
