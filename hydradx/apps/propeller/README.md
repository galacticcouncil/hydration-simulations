# Propeller M1 — HOLLAR peg & HSM drain model

Flow model for Propeller's HOLLAR sell-pressure vs the peg and HSM reserves.
See garden wiki `note-propeller-econ-models` (M1) for the research brief.

- `m1_state.py` — mainnet snapshot (2026-06-10, block ~12,699,668) of every
  HOLLAR venue (8 stableswap pools + Omnipool), HSM collateral configs +
  inventory, validated against the HOLLAR ERC20 and HSM pallet storage.
- `m1_model.py` — M1World: PropellerFlow (deploy/unwind with the contract's
  fixed-$1 min-out stall), pairwise HOLLAR-conserving arb equalization across
  all venues (every counter-asset externally elastic; HOLLAR the only floating
  price), pallet-faithful HSM execute_arbitrage per block, Omnipool venue.
- `run_m1.py` — scenario suite → CSV + report.

Run: `python3 hydradx/apps/propeller/run_m1.py` (pure python, no deps).

v1 caveats: no organic HOLLAR demand/borrow growth; frictionless arbs (8bp
floor); PRIME side infinitely elastic at NAV; HSM refills only via mint side;
hourly step granularity (rate limits applied per block within steps).
