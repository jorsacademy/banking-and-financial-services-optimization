# Credit and Capital Allocation Optimization

A binary mixed-integer optimization model for selecting a synthetic set of loan opportunities under scarce lending capacity and risk limits.

The predictive layer is deliberately outside the optimizer: PD, LGD, pricing, exposure, and risk weights are treated as supplied estimates. The optimization layer decides **which opportunities to fund**.

## Objective

For each candidate exposure, expected profit is approximated as:

```text
interest margin
- expected credit loss
- capital charge
```

The MILP maximizes total expected profit.

## Constraints

- total lending exposure budget;
- total risk-weighted-asset budget;
- expected-loss budget;
- per-sector exposure concentration cap;
- binary approve/fund decision for every opportunity.

## Run

```bash
python -m banking_optimization.credit_capital
pytest tests/test_credit_capital.py
```

## Interpretation

This project illustrates a common distinction between prediction and prescription:

- a credit model may estimate PD/LGD;
- a pricing model may estimate economics;
- the optimizer chooses a portfolio consistent with scarce capital and portfolio-level limits.

## Limitations

The model is synthetic and simplified. It is not a credit underwriting system, regulatory capital engine, IFRS 9/CECL implementation, fair-lending assessment, or production portfolio manager. Real deployment requires validated risk models, policy rules, customer-level eligibility controls, governance, monitoring, and jurisdiction-specific legal/regulatory review.
