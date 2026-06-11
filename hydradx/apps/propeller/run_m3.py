"""
M3 scenario runner — stress-run exiter gap, churn-attack fee sizing, exit latency.

Usage: python3 run_m3.py
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from hydradx.apps.propeller.m3_model import M3World, BLOCKS_PER_DAY

H = 24


def setup(n_depositors=10, each=500_000.0, ramp_days=2, **kw):
    w = M3World(**kw)
    for i in range(n_depositors):
        w.deposit(f'u{i}', each)
    for _ in range(ramp_days * H):
        w.step(600)
    return w


def s1_stress_run(reverts: bool):
    """2% PRIME discount appears at day 30; all 10 depositors run, one per hour, FIFO.
    Discount persists 14d then reverts (or stays). Realized value by queue position."""
    w = setup()
    for _ in range(30 * H):
        w.step(600)
    reqs = []
    hour = 0
    revert_hour = 14 * 24
    # run: one redemption request per hour while the discount holds
    for hour in range(60 * 24):
        disc = 0.02 if (hour < revert_hour or not reverts) else 0.0
        if hour < 10:
            reqs.append(w.request_redeem(f'u{hour}'))
        w.step(600, prime_discount=disc)
        if all(r['done'] for r in reqs) and len(reqs) == 10:
            break
    outs = []
    for i, r in enumerate(reqs):
        out = w.claim(r)
        stuck = r['coll_owed'] * (1 - r['repaid'] / r['debt_share']) if r['debt_share'] > 0 else 0
        wait_d = (r['done_block'] - r['requested_block']) / BLOCKS_PER_DAY if r['done_block'] else None
        outs.append((i, out, stuck, wait_d))
    print(f"S1 run, discount {'reverts at d14' if reverts else 'persists'}:")
    for i, out, stuck, wait_d in outs:
        if i in (0, 2, 4, 6, 9):
            w_str = f"{wait_d:.2f}d" if wait_d is not None else 'NEVER'
            print(f"   queue pos {i}: realized ${out:,.0f} / $500k  stuck ${stuck:,.0f}  settled in {w_str}")
    gap = (outs[0][1] - outs[-1][1]) / 500_000 * 100
    print(f"   first-vs-last gap: {gap:.2f}pp of deposit")


def s2_churn_attack():
    """Attacker cycles $500k deposit→redeem against $4.5M of holders.
    Measure holder loss + attacker loss per cycle at various entry/exit fees."""
    for fee_bps in (0, 10, 20, 30, 50):
        fee = fee_bps / 1e4
        w = setup(n_depositors=9, each=500_000, entry_fee=fee, exit_fee=fee)
        hv0 = sum(w.holder_value(f'u{i}') for i in range(9))
        attacker_spent = attacker_got = 0.0
        for cycle in range(10):
            attacker_spent += 500_000
            w.deposit('attacker', 500_000)
            for _ in range(6):       # let the seed deploy
                w.step(600)
            r = w.request_redeem('attacker')
            for _ in range(48):
                w.step(600)
                if r['done']:
                    break
            attacker_got += w.claim(r)
        hv1 = sum(w.holder_value(f'u{i}') for i in range(9))
        holder_loss = hv0 - hv1
        att_loss = attacker_spent - attacker_got
        print(f"S2 fee {fee_bps:>2}bp: 10 cycles — holders {'-' if holder_loss>0 else '+'}${abs(holder_loss):,.0f}, "
              f"attacker cost ${att_loss:,.0f} ({att_loss/attacker_spent*100:.2f}% per cycle notional)")


def s3_exit_latency():
    """50% of TVL exits at once; time to full settlement vs external arb capacity."""
    for cap, label in ((None, 'unlimited'), (500_000, '$500k/h'), (100_000, '$100k/h'), (25_000, '$25k/h')):
        w = setup(n_depositors=10, each=500_000, arb_capacity_per_hour=cap)
        for _ in range(7 * H):
            w.step(600)
        reqs = [w.request_redeem(f'u{i}') for i in range(5)]
        for hours in range(1, 30 * 24):
            w.step(600)
            if all(r['done'] for r in reqs):
                break
        done = sum(1 for r in reqs if r['done'])
        times = [f"{(r['done_block']-r['requested_block'])/BLOCKS_PER_DAY:.2f}d" if r['done_block'] else 'NEVER' for r in reqs]
        print(f"S3 arb capacity {label:>9}: settled {done}/5; per-request {times}")


def s4_crash_rebalance_vs_queue():
    """Collateral −40% + permissionless rebalance() while 5 redemptions are queued:
    the vault's deleverTarget settles AHEAD of the queue (CollateralVault.sol:334).
    Compare queue settle times with vs without the rebalance call."""
    for do_rebalance in (False, True):
        w = setup(n_depositors=10, each=500_000)
        for _ in range(7 * H):
            w.step(600)
        reqs = [w.request_redeem(f'u{i}') for i in range(5)]
        w.step(600)
        if do_rebalance:
            repay = w.crash_and_rebalance(0.40)
        for _ in range(30 * 24):
            w.step(600)
            if all(r['done'] for r in reqs):
                break
        times = [f"{(r['done_block']-r['requested_block'])/BLOCKS_PER_DAY:.2f}d" if r['done_block'] else 'NEVER' for r in reqs]
        label = f"crash+rebalance (deleverTarget ${repay:,.0f})" if do_rebalance else "no rebalance (control)"
        print(f"S4 {label}: settle times {times}, residual deleverTarget ${w.vault_delever_target:,.0f}")


if __name__ == '__main__':
    s1_stress_run(reverts=False)
    s1_stress_run(reverts=True)
    s2_churn_attack()
    s3_exit_latency()
    s4_crash_rebalance_vs_queue()
