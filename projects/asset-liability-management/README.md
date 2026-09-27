# Asset-Liability Management Optimization

A synthetic **prescriptive ALM** project with two optimization layers:

1. a compact one-period LP for transparent balance-sheet allocation;
2. a three-period stochastic ALM model on a scenario tree with adaptive rebalancing.

The project is designed as the flagship banking example in this repository. It treats the balance sheet, funding mix, liquidity posture, duration gap, and hedge position as coupled decisions rather than as separate reporting metrics.

## Implementations

### 1. One-period ALM

Module:

```text
banking_optimization.alm
```

The model chooses:

- asset-bucket allocations;
- funding-source allocations;
- an interest-rate hedge;
- scenario NII shortfall variables.

It enforces:

- balance-sheet identities;
- bucket limits;
- a capital-ratio proxy;
- a liquidity proxy;
- an absolute duration-gap limit;
- scenario-level NII downside accounting.

Run:

```bash
python -m banking_optimization.alm
```

### 2. Multi-period stochastic ALM

Module:

```text
banking_optimization.alm_stochastic
```

The stochastic version uses a three-period scenario tree:

```text
                 root
                /    \
              up      down
             /  \     /  \
         up_up up_down down_up down_down
```

Every node has synthetic:

- asset yields;
- funding costs;
- funding-runoff stress;
- hedge payoff;
- probability.

The optimizer can adapt the asset mix, funding mix, and hedge after uncertainty is revealed. Decisions are node-based, so non-anticipativity is built into the scenario-tree representation: nodes sharing the same observed history share the same decision.

The objective balances:

- expected multi-period NII;
- NII downside penalties;
- asset-rebalancing costs;
- funding-rebalancing costs;
- hedge costs.

Run:

```bash
python -m banking_optimization.alm_stochastic
```

## Static-policy benchmark

The stochastic model can also be solved with:

```python
solve_stochastic(problem, adaptive=False)
```

In this mode, future asset allocations, funding allocations, and hedge positions are constrained to remain equal to the root decision.

This provides a directly comparable benchmark for the adaptive stochastic policy.

```python
from banking_optimization.alm_stochastic import compare_policies

comparison = compare_policies()
print(comparison)
```

The reported `objective_improvement_vs_static` is the model-implied value of allowing the balance sheet to adapt to observed scenario states under the same synthetic assumptions.

## Sensitivity analysis

The experiment layer evaluates the stochastic ALM policy over a grid of:

- maximum duration-gap limits;
- NII shortfall penalties.

Run:

```bash
python projects/asset-liability-management/run_experiments.py
```

It writes reproducible CSV outputs under:

```text
projects/asset-liability-management/outputs/
├── policy_comparison.csv
├── sensitivity_grid.csv
├── adaptive_asset_policy.csv
├── adaptive_funding_policy.csv
└── adaptive_node_metrics.csv
```

The `outputs/` directory is intentionally git-ignored because these files are generated artifacts.

## Validation

The test suite checks:

- balance-sheet equality at every scenario-tree node;
- asset/funding bucket bounds;
- capital-ratio feasibility;
- liquidity feasibility;
- duration-gap feasibility;
- hedge bounds;
- NII shortfall accounting;
- static-policy invariance;
- adaptive-policy dominance over the constrained static benchmark;
- sensitivity-grid reproducibility.

Run:

```bash
pytest tests/test_alm.py tests/test_alm_stochastic.py tests/test_alm_experiments.py
```

## Model architecture

The project follows:

```text
scenario assumptions
        ↓
rates / runoff / hedge economics
        ↓
multi-period optimization
        ↓
adaptive balance-sheet policy
        ↓
feasibility validation
        ↓
baseline + sensitivity comparison
```

See [MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md) for the formulation.

## Why this is an optimization project

ALM is often shown as a measurement exercise: calculate duration gap, NII sensitivity, or liquidity ratios for a fixed balance sheet.

Here those exposures are decision variables. Capital, liquidity, earnings, funding structure, interest-rate exposure, and adjustment costs therefore interact inside one optimization system.

The stochastic version adds a second decision layer: not only **what balance sheet to hold**, but also **how the policy should react after uncertainty is partially observed**.

## Limitations

This remains an educational synthetic model. It is not a regulatory ALM, IRRBB, EVE, LCR, NSFR, FTP, hedge-accounting, treasury, behavioral-deposit, or prepayment engine.

A production implementation would require, among other things:

- contractual cash-flow ladders;
- yield-curve construction and repricing conventions;
- behavioral non-maturity-deposit models;
- prepayment and early-redemption models;
- currency segmentation;
- transfer-pricing assumptions;
- instrument-level optionality;
- hedge accounting and market-value treatment;
- capital/liquidity regulation specific to the institution and jurisdiction;
- model governance, validation, controls, and auditability.

## Disclaimer

Educational use only. This project is not financial, investment, accounting, legal, credit, treasury, or regulatory advice.
