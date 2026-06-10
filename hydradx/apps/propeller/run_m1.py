"""
M1 scenario runner — Propeller deploy/unwind flow vs HOLLAR peg and HSM reserves.

Usage: python3 run_m1.py [outdir]
Writes one CSV per scenario + a summary report to stdout / report.txt.
"""
import sys, os, csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from hydradx.apps.propeller.m1_model import M1World, BLOCKS_PER_DAY

STEP_BLOCKS = 600        # 1 hour
STEPS_PER_DAY = BLOCKS_PER_DAY // STEP_BLOCKS


def run(name, days, deploy_per_day=0.0, unwind_per_day=0.0, hsm_holdings='current',
        deploy_cap=None, unwind_after_day=None, outdir='.'):
    """deploy_per_day in HOLLAR; optional total cap; optional switch to unwind after a day."""
    w = M1World(hsm_holdings=hsm_holdings)
    rows = []
    for step in range(days * STEPS_PER_DAY):
        day = step / STEPS_PER_DAY
        dep = unw = 0.0
        if unwind_after_day is not None and day >= unwind_after_day:
            open_position = w.flow.deployed - w.flow.unwound
            unw = min(unwind_per_day / STEPS_PER_DAY, max(open_position, 0.0))
        else:
            if deploy_cap is None or w.flow.deployed < deploy_cap:
                dep = deploy_per_day / STEPS_PER_DAY
        w.step(STEP_BLOCKS, deploy_hollar=dep, unwind_hollar=unw)
        s = w.snapshot()
        rows.append(s)
    # CSV
    path = os.path.join(outdir, f'm1_{name}.csv')
    with open(path, 'w', newline='') as f:
        cw = csv.writer(f)
        venue_keys = list(rows[0]['prices'].keys())
        hsm_keys = list(rows[0]['hsm_inventory'].keys())
        cw.writerow(['day', 'market_price', 'min_price', 'deployed', 'unwound', 'stalled',
                     'hollar_burned', 'hollar_minted', 'hsm_inventory_usd', 'pool143_hollar_share']
                    + [f'p_{k}' for k in venue_keys] + [f'hsm_{k}' for k in hsm_keys])
        for s in rows:
            cw.writerow([round(s['day'], 3), s['market_price'], s['min_price'], s['deployed'],
                         s['unwound'], s['stalled'], s['hollar_burned'], s['hollar_minted'],
                         s['hsm_inventory_usd'], s['pool143_hollar_share']]
                        + [s['prices'][k] for k in venue_keys] + [s['hsm_inventory'][k] for k in hsm_keys])
    # summary line
    final = rows[-1]
    worst = min(r['market_price'] for r in rows)
    worst_day = [r['day'] for r in rows if r['market_price'] == worst][0]
    days_below_995 = sum(1 for r in rows if r['market_price'] < 0.995) / STEPS_PER_DAY
    days_below_99 = sum(1 for r in rows if r['market_price'] < 0.99) / STEPS_PER_DAY
    return dict(name=name, rows=rows, summary=dict(
        deployed=final['deployed'], unwound=final['unwound'],
        worst_price=worst, worst_day=worst_day,
        final_price=final['market_price'],
        days_below_0995=days_below_995, days_below_099=days_below_99,
        hsm_spent_usd=rows[0]['hsm_inventory_usd'] - final['hsm_inventory_usd'],
        hollar_burned=final['hollar_burned'], hollar_minted=final['hollar_minted'],
        stalled_steps=final['stalled'],
    ))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__) or '.'
    scenarios = [
        dict(name='A_baseline_7d', days=7),
        dict(name='B_deploy250k_28d', days=28, deploy_per_day=250_000),
        dict(name='C_deploy1M_14d', days=14, deploy_per_day=1_000_000),
        dict(name='D_deploy1M_14d_fullHSM', days=14, deploy_per_day=1_000_000, hsm_holdings='full'),
        dict(name='E_flash7M_2d', days=7, deploy_per_day=3_500_000, deploy_cap=7_000_000),
        dict(name='F_ramp500k_then_unwind2M', days=21, deploy_per_day=500_000, deploy_cap=7_000_000,
             unwind_per_day=2_000_000, unwind_after_day=14),
    ]
    results = []
    for sc in scenarios:
        print(f"running {sc['name']} ...", flush=True)
        results.append(run(outdir=outdir, **sc))
    lines = []
    hdr = f"{'scenario':<28} {'deployed':>11} {'worst p':>8} {'@day':>5} {'final p':>8} {'d<.995':>7} {'d<.99':>6} {'HSM spent':>10} {'burned':>10} {'stalls':>6}"
    lines.append(hdr); lines.append('-' * len(hdr))
    for r in results:
        s = r['summary']
        lines.append(f"{r['name']:<28} {s['deployed']:>11,.0f} {s['worst_price']:>8.4f} {s['worst_day']:>5.1f} "
                     f"{s['final_price']:>8.4f} {s['days_below_0995']:>7.2f} {s['days_below_099']:>6.2f} "
                     f"{s['hsm_spent_usd']:>10,.0f} {s['hollar_burned']:>10,.0f} {s['stalled_steps']:>6}")
    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(outdir, 'm1_report.txt'), 'w') as f:
        f.write(report + '\n')


if __name__ == '__main__':
    main()
