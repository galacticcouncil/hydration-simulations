"""
M1 — HOLLAR peg & HSM drain model: mainnet state snapshot + builders.

Snapshot taken 2026-06-10, para block ~12,699,668, validated against the HOLLAR
ERC20 contract (balanceOf per pool account) and HSM pallet storage (raw RPC).

HOLLAR (asset 222, 18dp, Erc20/GHO) venues:
  pool 105  3-Pool-MRL   HOLLAR + USDC(21) + USDT(23)        amp 222, fee 2bp
  pool 110  2-Pool-HUSDC HOLLAR + aUSDC(1003)                amp 222, fee 2bp   [HSM]
  pool 111  2-Pool-HUSDT HOLLAR + aUSDT(1002)                amp 222, fee 2bp   [HSM]
  pool 112  2-Pool-HUSDS HOLLAR + sUSDS(1000745) peg 1.0991  amp 111, fee 4bp   [HSM]
  pool 113  2-Pool-HUSDe HOLLAR + sUSDe(1000625) peg 1.2331  amp 111, fee 4bp   [HSM]
  pool 143  2-Pool-PRIME HOLLAR + PRIME(43)      peg 1.0370  amp 100, fee 4bp
  pool 146  2-Pool-apyUSD HOLLAR + apyUSD(46)    peg 1.3672  amp 100, fee 4bp
  pool 10044 2-Pool-HEURC HOLLAR + aEURC(1044)   peg 1.1549  amp  50, fee 5bp
  omnipool  HOLLAR/LRNA  1,648,344 HOLLAR : 305,573 LRNA

HSM (pallet account modl+py/hsmod, runtime index 82):
  collateral  pool  purchase_fee  max_buy_price_coef  buyback_rate  buy_back_fee  max_in_holding  current holding
  aUSDT 1002   111      0             0.998             1e-4/block      1bp          8,000,000       123,372
  aUSDC 1003   110      0             0.998             1e-4/block      1bp          8,000,000             0
  sUSDS        112      0             0.995             1e-4/block      1bp          2,000,000             0
  sUSDe        113      0             0.995             1e-4/block      1bp          2,000,000             0

HOLLAR total supply: 11,434,606.
Counter-asset USD values (external NAV anchors): aUSDT/aUSDC/USDT/USDC = 1.0,
sUSDS = 1.0991, sUSDe = 1.2331, PRIME = 1.0370, apyUSD = 1.3672, aEURC = 1.1549
(EURUSD). PRIME mints/redeems freely at oracle NAV with deep Solana liquidity,
so the PRIME side of 143 is externally elastic.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.agents import Agent
from hydradx.model.hollar import StabilityModule

SNAPSHOT_BLOCK = 12_699_668
SNAPSHOT_DATE = "2026-06-10"
HOLLAR_TOTAL_SUPPLY = 11_434_606.0

# external USD value of each counter-asset (oracle NAV / FX) — HOLLAR deliberately absent
USD_NAV = {
    'USDT': 1.0, 'USDC': 1.0, 'aUSDT': 1.0, 'aUSDC': 1.0,
    'sUSDS': 1.099089, 'sUSDe': 1.233054,
    'PRIME': 1.037, 'apyUSD': 1.367232, 'aEURC': 1.154925,
    'LRNA': None,  # set so omnipool HOLLAR spot matches the stable venues at t0
}

POOLS = {
    # unique_id: (tokens, amplification, trade_fee, peg list for assets after the first)
    '3-Pool-MRL':   ({'HOLLAR': 289_280.0, 'USDC': 159_291.0, 'USDT': 154_786.0}, 222, 0.0002, [1.0, 1.0]),
    '2-Pool-HUSDC': ({'HOLLAR': 1_475_918.0, 'aUSDC': 790_705.0}, 222, 0.0002, [1.0]),
    '2-Pool-HUSDT': ({'HOLLAR': 1_413_570.0, 'aUSDT': 928_204.0}, 222, 0.0002, [1.0]),
    '2-Pool-HUSDS': ({'HOLLAR': 91_599.0, 'sUSDS': 56_623.0}, 111, 0.0004, [1.099089]),
    '2-Pool-HUSDe': ({'HOLLAR': 138_220.0, 'sUSDe': 67_051.0}, 111, 0.0004, [1.233054]),
    '2-Pool-PRIME': ({'HOLLAR': 703_162.0, 'PRIME': 344_626.0}, 100, 0.0004, [1.037]),
    '2-Pool-apyUSD': ({'HOLLAR': 126_247.0, 'apyUSD': 641_776.0}, 100, 0.0004, [1.367232]),
    '2-Pool-HEURC': ({'HOLLAR': 706_889.0, 'aEURC': 517_119.0}, 50, 0.0005, [1.154925]),
}

# HSM: collateral asset -> (pool unique_id, params). buyback_rate is per BLOCK.
HSM_CONFIG = {
    'aUSDT': dict(pool='2-Pool-HUSDT', max_buy_price_coef=0.998, buyback_rate=1e-4, buy_back_fee=1e-4,
                  max_in_holding=8_000_000.0, holding=123_372.0),
    'aUSDC': dict(pool='2-Pool-HUSDC', max_buy_price_coef=0.998, buyback_rate=1e-4, buy_back_fee=1e-4,
                  max_in_holding=8_000_000.0, holding=0.0),
    'sUSDS': dict(pool='2-Pool-HUSDS', max_buy_price_coef=0.995, buyback_rate=1e-4, buy_back_fee=1e-4,
                  max_in_holding=2_000_000.0, holding=0.0),
    'sUSDe': dict(pool='2-Pool-HUSDe', max_buy_price_coef=0.995, buyback_rate=1e-4, buy_back_fee=1e-4,
                  max_in_holding=2_000_000.0, holding=0.0),
}

OMNIPOOL_HOLLAR = 1_648_344.0
OMNIPOOL_LRNA = 305_573.0


def build_pools() -> dict:
    """All HOLLAR stableswap venues as StableSwapPoolState keyed by unique_id."""
    pools = {}
    for uid, (tokens, amp, fee, peg) in POOLS.items():
        pools[uid] = StableSwapPoolState(
            tokens=dict(tokens), amplification=amp, trade_fee=fee, peg=list(peg), unique_id=uid,
        )
    return pools


def build_hsm(pools: dict, holdings: str = 'current'):
    """StabilityModule wired to the four HSM pools.

    holdings: 'current' (123k aUSDT only) or 'full' (max_in_holding everywhere),
    or a dict collateral->amount.
    """
    order = list(HSM_CONFIG.keys())
    if holdings == 'current':
        liq = {c: HSM_CONFIG[c]['holding'] for c in order}
    elif holdings == 'full':
        liq = {c: HSM_CONFIG[c]['max_in_holding'] for c in order}
    else:
        liq = {c: float(holdings.get(c, 0.0)) for c in order}
    # StabilityModule requires liquidity > 0 entries to be meaningful; zero is allowed.
    return StabilityModule(
        liquidity=liq,
        buyback_speed=[HSM_CONFIG[c]['buyback_rate'] for c in order],
        pools=[pools[HSM_CONFIG[c]['pool']] for c in order],
        max_buy_price_coef=[HSM_CONFIG[c]['max_buy_price_coef'] for c in order],
        buy_fee=[HSM_CONFIG[c]['buy_back_fee'] for c in order],
        sell_price_fee=[0.0001 for _ in order],  # purchase_fee is 0 on-chain; small >0 for module validity
        native_stable='HOLLAR',
        max_liquidity={c: HSM_CONFIG[c]['max_in_holding'] for c in order},
    )


def hollar_usd_price(pool: StableSwapPoolState) -> float:
    """Marginal USD price of HOLLAR in a venue = price of HOLLAR in counter-asset × counter NAV."""
    others = [t for t in pool.asset_list if t != 'HOLLAR']
    tkn = others[0]
    return pool.price('HOLLAR', tkn) * USD_NAV[tkn]


if __name__ == '__main__':
    pools = build_pools()
    print(f"snapshot {SNAPSHOT_DATE} block {SNAPSHOT_BLOCK}")
    for uid, p in pools.items():
        print(f"{uid:>15}: HOLLAR ${hollar_usd_price(p):.4f}  reserves {{{', '.join(f'{t}: {q:,.0f}' for t, q in p.liquidity.items())}}}")
