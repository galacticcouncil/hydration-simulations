"""
M2 scenario runner — backing gap, carry inversion, depeg race, keeper outage.

Usage: python3 run_m2.py
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from hydradx.apps.propeller.m2_model import (
    PropellerM2, BLOCKS_PER_DAY, SYNTH_LT, ETH_LT,
)

H = 24  # hourly steps


def ramp(m, days=2):
    for _ in range(days * H):
        m.step(600)


def s1_baseline_year(f1_fixed):
    """1 year at live rates, monthly harvest, then full redeem."""
    m = PropellerM2(deposit_usd=5_000_000, f1_fixed=f1_fixed)
    ramp(m)
    gap_path = []
    for day in range(365):
        for h in range(H):
            m.step(600, harvest_now=(h == 0 and day % 30 == 29))
        gap_path.append(m.backing_gap())
    # full redemption
    debt_share = m.request_full_redeem()
    for _ in range(60 * H):  # spiral up to 60 days
        m.step(600)
        if m.unwind_target <= 1.0:
            break
    out, stuck = m.settle_redeem(debt_share)
    s = m.snapshot()
    label = 'F1-fixed' if f1_fixed else 'as-implemented'
    print(f"S1 baseline 1y [{label}]:")
    print(f"  backing gap: day30 ${gap_path[29]:,.0f}  day180 ${gap_path[179]:,.0f}  day365 ${gap_path[-1]:,.0f}")
    print(f"  compounded yield ${s['compounded']:,.0f} ({s['compounded']/5e6*100:.2f}% of deposit)")
    print(f"  redemption: collateral released {out/ (5e6 + s['compounded'])*100:.2f}%  STUCK {stuck*100:.2f}%"
          f"  (≈ ${stuck*(5e6+s['compounded']):,.0f} of user value)")


def s2_carry_inversion():
    """HOLLAR rate hike at day 30: spread → negative; watch gap + deLever."""
    for new_rate, label in ((0.07, 'HOLLAR 7.0% (spread −0.66%)'), (0.10, 'HOLLAR 10% (spread −3.66%)')):
        m = PropellerM2(deposit_usd=5_000_000)
        ramp(m)
        for day in range(30):
            for h in range(H):
                m.step(600, harvest_now=(h == 0 and day % 30 == 29))
        for day in range(180):
            for h in range(H):
                m.step(600, hollar_rate=new_rate)
        s = m.snapshot()
        print(f"S2 {label}: after 180d inverted — gap ${s['backing_gap']:,.0f} "
              f"({s['backing_gap']/s['main_debt']*100:.2f}% of Main debt), HF {s['loop_hf']:.3f}, "
              f"liqs {s['liquidation_events']}")


def s3_depeg_race():
    """PRIME market discount develops at various speeds; who wins — spiral or liquidation?"""
    cases = [
        ('1%/day to 5%', 0.05, 5 * 24),     # (label, final discount, hours to reach)
        ('5% over 12h', 0.05, 12),
        ('5% over 2h', 0.05, 2),
        ('5% instant', 0.05, 0),
        ('8% over 12h', 0.08, 12),
        ('8% instant', 0.08, 0),
        ('12% over 24h', 0.12, 24),
    ]
    for label, target, hours in cases:
        m = PropellerM2(deposit_usd=5_000_000)
        ramp(m)
        eq0 = m.loop_equity()
        if hours == 0:
            m.step(1, prime_discount=target)
        else:
            blocks = hours * 600
            # block-level resolution during the race (keeper polls ~every block)
            chunk = 10
            for b in range(0, blocks, chunk):
                m.step(chunk, prime_discount=target * (b + chunk) / blocks)
        for _ in range(48 * H):
            m.step(600)
            if m.delever_debt_target <= 0 and m.loop_hf() > 1.04:
                break
        s = m.snapshot()
        eq_loss = (eq0 - m.loop_equity()) + 0.0
        print(f"S3 {label:<14}: liqs {s['liquidation_events']:>3}  liq-loss ${s['liquidated_loss']:>9,.0f}  "
              f"equity Δ ${-eq_loss:>11,.0f}  final HF {min(s['loop_hf'],9.99):.3f}  gap ${s['backing_gap']:,.0f}")


def s4_keeper_outage():
    """maintainPeg offline; how long until the synth floor breaks, and what ETH crash then liquidates Main?"""
    for rate, label in ((0.044, '4.4%'), (0.10, '10%'), (0.20, '20%')):
        m = PropellerM2(deposit_usd=5_000_000, hollar_rate=rate)
        ramp(m)
        broke_day = None
        for day in range(120):
            for _ in range(H):
                m.step(600, keeper_online=False)
            if not m.synth_floor_ok and broke_day is None:
                broke_day = day + 1
                break
        if broke_day is None:
            print(f"S4 rate {label}: floor intact ≥120d offline")
            continue
        # crash threshold degrades continuously after the break — report at +30d offline
        for _ in range(30 * H):
            m.step(600, keeper_online=False)
        mult = (m.main_debt - m.synth * SYNTH_LT) / (m.collateral_usd * ETH_LT)
        print(f"S4 rate {label}: synth floor breaks after {broke_day}d keeper outage; "
              f"at outage+30d Main liquidates if collateral falls {(1-mult)*100:.1f}%")


def s5_unwind_under_depeg():
    """Full exit during a 3% PRIME discount — realized value vs par (R8)."""
    m = PropellerM2(deposit_usd=5_000_000)
    ramp(m)
    for _ in range(30 * H):
        m.step(600)
    m.step(600, prime_discount=0.03)
    debt_share = m.request_full_redeem()
    for _ in range(60 * H):
        m.step(600)
        if m.unwind_target <= 1.0:
            break
    out, stuck = m.settle_redeem(debt_share)
    s = m.snapshot()
    print(f"S5 exit during 3% PRIME discount: released {out/5e6*100:.2f}% of deposit, stuck {stuck*100:.2f}%, "
          f"liqs {s['liquidation_events']}, gap ${s['backing_gap']:,.0f}")


if __name__ == '__main__':
    s1_baseline_year(False)
    s1_baseline_year(True)
    s2_carry_inversion()
    s3_depeg_race()
    s4_keeper_outage()
    s5_unwind_under_depeg()
