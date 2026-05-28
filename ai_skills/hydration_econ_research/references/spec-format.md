# Research Brief Format

Use this reference when producing a full mechanism research brief.

## Required Sections

# Research Brief - <Mechanism / Feature>

## Goal
State the product/protocol objective and the decision the brief is meant to support.

## Context
Describe relevant protocol context, user flow, market environment, and why the mechanism matters now.

## Current Hydration Design
Summarize current runtime, pallet, parameter, or simulation behavior. Cite local files when possible.

## Prior Simulation Artifacts
List relevant notebooks, Python models, tests, or data loaders from `hydration-simulations`.

## Proposed Mechanism
Describe the mechanism in implementation-neutral terms first.

## Economic Model
Define variables, formulas, conservation properties, prices, liabilities, and value flows.

## Actors And Incentives
List actors and explain what each actor is expected to do voluntarily.

## State Variables
List protocol state and derived state. Distinguish on-chain state from simulation-only state.

## Parameters
Use a table with type, unit, controller, range, default/recommendation, and reason.

## Core Flows
Describe user/protocol flows step by step: mint, redeem, swap, add liquidity, withdraw, liquidate, rebalance, update parameter, etc.

## Invariants
List safety/economic invariants and whether they should be checked in runtime tests, model tests, or simulations.

## Failure Modes
Include depeg, thin liquidity, blocked arbitrage, oracle lag, collateral crash, whale trades, governance delay, and paused markets where relevant.

## Data Sources
Identify current chain state, historical chain state, CEX data, order books, oracle feeds, SQLPad, or synthetic assumptions.

## Model Components Needed
List existing simulation components to reuse and missing components to build.

## Agent Strategies
Define trader, LP, arbitrageur, liquidator, borrower, governance, and treasury behavior where relevant.

## Scenario Matrix
Provide scenarios, shocks, parameter sweeps, and expected observations.

## Metrics And Pass/Fail Criteria
Define metrics such as peg deviation, surplus/deficit, bad debt, fee revenue, LP return, arbitrage profit, liquidity depth, user execution price, and time to recovery.

## Economic Tests
Propose Python model tests, Rust unit tests, runtime integration tests, property/fuzz tests, or notebooks.

## Runtime Design Implications
Map research to storage, calls, hooks, traits, events, errors, governance origins, migrations, benchmarks, and tests.

## Open Questions
List unresolved assumptions and required decisions.

## Recommendation
State the recommended path and what must happen before implementation or launch.
