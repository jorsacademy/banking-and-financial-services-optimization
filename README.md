# Banking and Financial Services Optimization

A portfolio of reproducible Operations Research and decision-intelligence projects for banking and financial services.

The repository focuses on **prescriptive analytics**: models that choose actions under capital, liquidity, risk, capacity, and regulatory-style constraints. It complements forecasting and machine-learning work by turning estimates into explicit decisions.

## Project map

| Project | Core decision | Methods | Status |
|---|---|---|---|
| [Asset-Liability Management](projects/asset-liability-management/) | Adaptive balance-sheet, funding, and hedge policy on an interest-rate scenario tree | LP, multi-period stochastic programming, sensitivity analysis | Flagship |
| [Credit & Capital Allocation](projects/credit-capital-allocation/) | Which exposures to approve/fund under capital and concentration limits | MILP | Implemented |
| [Loan Pricing & Limit Optimization](projects/loan-pricing-and-limit-optimization/) | Customer-level rate and credit-limit decisions | MILP, predict-then-optimize | Implemented |
| [Liquidity & Funding Optimization](projects/liquidity-and-funding-optimization/) | Funding mix and liquidity buffer decisions | LP, stress scenarios | Implemented |
| [Collections Optimization](projects/collections-optimization/) | Which delinquent account receives which treatment | MILP, resource allocation | Implemented |
| [Collateral Allocation](projects/collateral-allocation/) | Allocate eligible collateral under haircuts and eligibility rules | LP, network allocation | Implemented |
| [Fraud Alert Triage](projects/fraud-alert-triage/) | Select alerts for investigation under scarce analyst capacity | Binary optimization | Implemented |
| [Intraday Liquidity](projects/intraday-liquidity/) | Payment release/sequencing under liquidity limits | MILP, scheduling | Implemented |
| [ATM Cash Replenishment](projects/atm-cash-replenishment/) | Replenishment timing and quantities | Multi-period MILP | Implemented |
| [Cash Supply Chain Optimization](projects/cash-supply-chain-optimization/) | ATM/branch cash replenishment and CIT routing under multistage demand uncertainty | Multistage stochastic IRP, CVaR risk aversion, rolling horizon, route-column MILP | Flagship |
| [Payment Routing](projects/payment-routing/) | Route payment segments across processors | Assignment MILP | Implemented |
| [Deposit Pricing & Retention](projects/deposit-pricing-and-retention/) | Segment-level deposit rate offers under a funding target | MILP, retention-response optimization | Implemented |

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
│   ├── asset-liability-management/
│   ├── credit-capital-allocation/
│   ├── loan-pricing-and-limit-optimization/
│   ├── liquidity-and-funding-optimization/
│   ├── collections-optimization/
│   ├── collateral-allocation/
│   ├── fraud-alert-triage/
│   ├── intraday-liquidity/
│   ├── atm-cash-replenishment/
│   ├── cash-supply-chain-optimization/
│   ├── payment-routing/
│   └── deposit-pricing-and-retention/
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
