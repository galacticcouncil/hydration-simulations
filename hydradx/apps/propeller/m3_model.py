"""
M3 — Propeller exit & run dynamics (multi-depositor, FIFO queue, cost socialization).

Extends the M2 state machine with per-depositor pVault shares and the contract's
redemption mechanics (CollateralVault.requestRedeem/pokeSettle/claim):

  * requestRedeem snapshots the request's proportional collateralOwed, debtShare
    (incl. accrued interest) and loop-equity slice at CURRENT oracle NAV;
  * the unwind spiral frees equity over time (throughput bounded by tranche ×
    keeper cadence AND by external arb capacity restoring pool-143);
  * pokeSettle repays each request's debtShare FIFO from freed HOLLAR and
    releases collateral pro-rata (F1 as-implemented or F1-fixed);
  * all loop-leg execution costs land on SHARED loop equity (F2: socialized) —
    exiters are made whole at their snapshot, remaining holders absorb costs.

Entry/exit fee: charged on deposit/claim as a fraction retained by the vault
(accruing to remaining holders) — the M3 design knob.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from hydradx.apps.propeller.m2_model import PropellerM2, BLOCKS_PER_DAY


class M3World(PropellerM2):
    """Multi-depositor vault on the shared loop. Shares are USD-of-collateral based
    (collateral price held at 1.0 — we study exit dynamics, not collateral beta)."""

    def __init__(self, entry_fee=0.0, exit_fee=0.0, arb_capacity_per_hour=None, **kw):
        super().__init__(deposit_usd=1.0, **kw)  # dust seed; real deposits below
        # zero out the dust position
        self.collateral_usd = 0.0
        self.main_debt = 0.0
        self.synth = 0.0
        self.pending_deploy = 0.0
        self.principal_equity = 0.0
        self.entry_fee = entry_fee
        self.exit_fee = exit_fee
        self.arb_capacity_per_hour = arb_capacity_per_hour  # USD of pool-143 restoration per hour (None = unlimited)
        self._arb_budget = 0.0
        self.shares = {}          # depositor -> vault shares
        self.total_shares = 0.0
        self.fee_pot = 0.0        # fees retained (accrue to remaining holders via collateral)
        self.queue = []           # FIFO: dicts(owner, shares, coll_owed, debt_share, repaid, coll_settled)

    # ── deposits / redemptions (CollateralVault semantics) ───────────────
    def vault_assets(self):
        return self.collateral_usd

    def deposit(self, owner, usd):
        fee = usd * self.entry_fee
        self.fee_pot += fee
        usd_net = usd - fee
        new_shares = usd_net if self.total_shares == 0 else \
            usd_net * self.total_shares / self.vault_assets()
        self.shares[owner] = self.shares.get(owner, 0.0) + new_shares
        self.total_shares += new_shares
        self.collateral_usd += usd_net + fee  # fee stays in the vault as collateral
        borrow = usd_net * 0.75
        self.main_debt += borrow
        self.synth = self.main_debt / 0.98 * 1.005
        self.pending_deploy += borrow
        self.principal_equity += borrow
        return new_shares

    def request_redeem(self, owner, frac=1.0):
        sh = self.shares.get(owner, 0.0) * frac
        if sh <= 0:
            return None
        supply = self.total_shares
        coll_owed = self.collateral_usd * sh / supply
        debt_share = self.main_debt * sh / supply
        equity = max(self.loop_collateral_usd() - self.loop_debt, 0.0) + self.pending_deploy
        loop_slice = equity * sh / supply
        # seed still pending deploys instantly to freed (matches vault netting of undeployed seed)
        pend_cut = min(self.pending_deploy * sh / supply, loop_slice)
        self.pending_deploy -= pend_cut
        self.freed += pend_cut
        self.unwind_target += loop_slice - pend_cut
        self.principal_equity -= min(self.principal_equity * sh / supply, self.principal_equity)
        self.shares[owner] -= sh   # escrowed: stays in total_shares until claim (contract burns at claim)
        req = dict(owner=owner, shares=sh, coll_owed=coll_owed, debt_share=debt_share,
                   repaid=0.0, coll_settled=0.0, done=False, requested_block=self.block,
                   done_block=None)
        self.queue.append(req)
        return req

    def poke_settle(self):
        """FIFO settlement from freed HOLLAR (+ interest reserve when F1-fixed)."""
        available = self.freed
        if self.f1_fixed and self.interest_reserve > 0:
            available += self.interest_reserve
        for req in self.queue:
            if req['done'] or available <= 0:
                continue
            remaining = req['debt_share'] - req['repaid']
            if remaining <= 0:
                req['done'] = True
                continue
            pay = min(available, remaining)
            available -= pay
            req['repaid'] += pay
            rel = req['coll_owed'] * pay / req['debt_share']
            req['coll_settled'] += rel
            self.collateral_usd -= rel
            self.main_debt -= pay
            if req['repaid'] >= req['debt_share'] - 1e-6:
                req['done'] = True
                req['done_block'] = self.block
            else:
                break  # FIFO head not fully repaid → wait (contract `else break`)
        used = self.freed + (self.interest_reserve if self.f1_fixed else 0.0) - available
        from_freed = min(used, self.freed)
        self.freed -= from_freed
        if self.f1_fixed:
            self.interest_reserve -= (used - from_freed)
        if self.main_debt > 0:
            self.synth = self.main_debt / 0.98 * 1.005

    def claim(self, req):
        self.total_shares -= req['shares']  # burn escrowed shares
        req['shares'] = 0.0
        out = req['coll_settled'] * (1 - self.exit_fee)
        self.fee_pot += req['coll_settled'] * self.exit_fee
        self.collateral_usd += req['coll_settled'] * self.exit_fee  # exit fee stays with holders
        req['coll_settled'] = 0.0
        return out

    # ── throughput limit: external arbs restore pool-143 at finite capacity ──
    def step(self, n_blocks, **kw):
        if self.arb_capacity_per_hour is not None:
            self._arb_budget += self.arb_capacity_per_hour * n_blocks / 600.0
            orig = self.pool.rebalance_external

            def capped(threshold=0.001):
                h = self.pool.state.liquidity['HOLLAR']
                p_usd = self.pool.state.liquidity['PRIME'] * self.pool.prime_price
                skew = h - p_usd
                move = min(abs(skew) / 2, self._arb_budget)
                if move < 1.0:
                    return
                self._arb_budget -= move
                if skew > 0:
                    self.pool.state.swap(self.pool.agent, tkn_buy='HOLLAR', tkn_sell='PRIME', buy_quantity=move)
                else:
                    self.pool.state.swap(self.pool.agent, tkn_buy='PRIME', tkn_sell='HOLLAR',
                                         buy_quantity=move / self.pool.prime_price)
                self.pool.state.fail = ''
            self.pool.rebalance_external = capped
            try:
                super().step(n_blocks, **kw)
            finally:
                self.pool.rebalance_external = orig
        else:
            super().step(n_blocks, **kw)
        self.poke_settle()

    def holder_value(self, owner):
        """Mark-to-NAV value of a holder's remaining shares."""
        if self.total_shares <= 0:
            return 0.0
        frac = self.shares.get(owner, 0.0) / self.total_shares
        equity = max(self.loop_collateral_usd() - self.loop_debt, 0.0) + self.pending_deploy
        # holder value = collateral share − debt share + loop slice (all USD)
        return frac * (self.collateral_usd - self.main_debt + equity)
