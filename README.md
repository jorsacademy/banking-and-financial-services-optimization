# Banking and Financial Services Optimization

A portfolio of reproducible Operations Research and decision-intelligence projects for banking and financial services.

The repository focuses on **prescriptive analytics**: models that choose actions under capital, liquidity, risk, capacity, and regulatory-style constraints. It complements forecasting and machine-learning work by turning estimates into explicit decisions.

## Project map

| Project | Core decision | Methods | Status |
|---|---|---|---|
| [Asset-Liability Management](projects/asset-liability-management/) | Balance-sheet allocation across assets, deposits, wholesale funding, and hedges | LP, multi-period optimization, scenario analysis | Implemented |
| Credit & Capital Allocation | Which exposures to approve/fund under capital and concentration limits | MILP, robust optimization | Planned |
| Loan Pricing & Limit Optimization | Customer-level rate and credit-limit decisions | Nonlinear/MILP, predict-then-optimize | Planned |
| Liquidity & Funding Optimization | Funding mix and liquidity buffer decisions | Stochastic LP, robust optimization | Planned |
| Collections Optimization | Which delinquent account receives which treatment and when | Assignment, MDP/bandits | Planned |
| Collateral Allocation | Allocate eligible collateral under haircuts and concentration rules | LP, min-cost flow | Planned |
| Fraud Alert Triage | Select alerts for investigation under scarce analyst capacity | Knapsack, assignment | Planned |
| Intraday Liquidity | Payment release/sequencing under liquidity limits | Scheduling, network optimization | Planned |
| ATM Cash Replenishment | Replenishment timing and quantities | Inventory-routing, stochastic optimization | Planned |
| Payment Routing | Route transactions across processors/rails | Multi-objective routing, online optimization | Planned |

## Design principles

Each project is intended to contain:

1. a banking/business decision problem;
2. a transparent mathematical formulation;
3. synthetic, reproducible data;
4. an optimization implementation;
5. automated feasibility and accounting tests;
6. scenario/sensitivity analysis;
7. explicit limitations and a non-advice disclaimer.

The examples use synthetic data and simplified constraints. They are educational decision-modeling systems, not production banking systems or regulatory calculators.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run all tests:

```bash
pytest
```

Run the first project:

```bash
python -m banking_optimization.alm
```

## Repository structure

```text
.
├── docs/
├── projects/
│   └── asset-liability-management/
├── src/
│   └── banking_optimization/
├── tests/
├── pyproject.toml
└── README.md
```

## Methodological note

A recurring architecture in this repository is **estimate → optimize → validate**. Predictive estimates such as demand, default probability, prepayment, utilization, recovery, or transaction success rates are treated as model inputs—not as decisions themselves.

Optimization outputs are checked for feasibility, accounting consistency, and sensitivity to assumptions. Historical or synthetic performance is not presented as evidence of future financial performance.

## Disclaimer

This repository is for research and educational use. It does not provide investment, financial, credit, legal, accounting, or regulatory advice. Models are simplified and must not be used for real banking decisions without independent validation, governance, controls, and applicable regulatory review.
