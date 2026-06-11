"""
M4 — Propeller capacity & yield floor (analytic, anchored to on-chain facts).

On-chain anchors (2026-06-10):
  HOLLAR facilitator buckets (GhoToken.getFacilitatorBucket):
    Aave pool 0x8c0f...108e : capacity 12,000,000 / minted 11,091,942  → ~0.91M headroom
    HSM       modlpy/hsmod  : capacity 18,000,000 / minted    282,485
    FlashMint 0xb328...8cf4 : capacity    100,000
  HSM inventory $123k aUSDT; max_in_holding caps Σ $22.4M (M1 snapshot)
  PRIME isolation debt ceiling $12M; HELOC yield 6.34%; HOLLAR rate 4.4% (governance)

Per $1 of TVL (75% LTV, loop HF 1.05 → 6.18× / 5.18×):
  Main HOLLAR borrow  = 0.75
  loop HOLLAR debt    = 0.75 × 5.18 = 3.89
  total HOLLAR minted = 4.64        (all sold into pool-143 on the way in)
  loop PRIME held     = 0.75 × 6.18 = 4.64
"""
LTV = 0.75
LEV = 6.18
HOLLAR_PER_TVL = LTV * (1 + (LEV - 1))      # 0.75 × 6.18 = 4.635 (main + loop debt)
LOOP_DEBT_PER_TVL = LTV * (LEV - 1)         # 3.885 — counts against PRIME isolation ceiling? no:
# the isolation ceiling counts HOLLAR debt against PRIME collateral = loop debt only.

BUCKET_CAP = 12_000_000.0
BUCKET_MINTED = 11_091_942.0
HSM_INV_NOW = 123_372.0
HSM_MAX_HOLDING = 22_400_000.0
PRIME_CEILING = 12_000_000.0
VENUE_DEPTH = 1_400_000.0   # M1: organic absorption down to the ~0.99 stall floor
HELOC = 0.0634
HOLLAR_RATE = 0.044
DRAG = 0.001                # steady-state swap drag (measured ~5bp/yr + buffer)


def capacity_stack():
    rows = []
    bucket_headroom = BUCKET_CAP - BUCKET_MINTED
    rows.append(('HOLLAR facilitator bucket headroom (TODAY)', bucket_headroom,
                 bucket_headroom / HOLLAR_PER_TVL))
    absorb_now = HSM_INV_NOW + VENUE_DEPTH
    rows.append(('HSM absorption, current inventory (M1)', absorb_now, absorb_now / HOLLAR_PER_TVL))
    absorb_full = HSM_MAX_HOLDING + VENUE_DEPTH
    rows.append(('HSM absorption at max_in_holding (M1 rule ≈1.0×)', absorb_full, absorb_full / HOLLAR_PER_TVL))
    rows.append(('PRIME isolation debt ceiling', PRIME_CEILING, PRIME_CEILING / LOOP_DEBT_PER_TVL))
    print('CAPACITY STACK — binding constraint = min row')
    print(f"{'constraint':<52} {'HOLLAR':>13} {'max TVL':>11}")
    for name, hollar, tvl in rows:
        print(f"{name:<52} ${hollar:>11,.0f} ${tvl:>9,.0f}")
    print(f"\n→ binding TODAY: facilitator bucket — max TVL ≈ ${min(r[2] for r in rows):,.0f}")
    print("→ governance staircase to TVL X: bucket ≥ 11.09M + 4.64X · HSM inventory ≥ ~4.6X (≤22.4M cap)")
    print("  · PRIME ceiling ≥ 3.9X · plus M1 deploy-rate pacing\n")


def apy_floor():
    print('APY vs HELOC yield (r_HOLLAR = 4.4% governance, drag 10bp):')
    print(f"{'HELOC':>7} {'spread':>8} {'net APY':>8}")
    for y_bps in (634, 580, 540, 500, 460, 440, 400):
        y = y_bps / 1e4
        apy = LTV * LEV * (y - HOLLAR_RATE) - DRAG
        print(f"{y*100:>6.2f}% {(y-HOLLAR_RATE)*100:>7.2f}% {apy*100:>7.2f}%")
    # thresholds
    for target, label in ((0.05, 'APY 5% (≈ stables — product loses edge)'),
                          (0.03, 'APY 3%'), (0.0, 'APY 0% (carry inversion)')):
        y_star = (target + DRAG) / (LTV * LEV) + HOLLAR_RATE
        print(f"  HELOC floor for {label}: {y_star*100:.2f}%")
    print("  HELOC has declined 8.00% → 6.34%; ~80bp more erases the edge vs plain stables.")
    print("  Governance response is one-sided: r_HOLLAR can be CUT to defend the spread,")
    print("  but that weakens HSM-era peg defense and treasury revenue (R11).\n")


def ramp_schedule():
    print('Suggested cap-ramp (gates from M1/M2, all governance-observable):')
    steps = [
        ('Phase 0 (now)', 200_000, 'fits today\'s bucket headroom + HSM; no gov action'),
        ('Phase 1', 1_000_000, 'bucket +5M; HSM inventory ≥ $5M; deploy ≤ $250k/day'),
        ('Phase 2', 2_500_000, 'bucket +12M; HSM ≥ $12M; peg never <0.995 for 7d in Phase 1'),
        ('Phase 3 (ceiling)', 3_000_000, 'PRIME ceiling binds (3.9×TVL ≤ $12M); raise REQ-CAPS beyond'),
    ]
    for name, tvl, gate in steps:
        print(f"  {name:<18} TVL ≤ ${tvl:>9,.0f} — {gate}")
    print("  Backing-gap monitor (M2) red-line: gap > 1% of Σ Main debt ⇒ pause deposits.")


if __name__ == '__main__':
    capacity_stack()
    apy_floor()
    ramp_schedule()
