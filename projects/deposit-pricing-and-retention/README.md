# Deposit Pricing and Retention Optimization

A mixed-integer offer-selection model for balancing deposit retention and funding cost across synthetic customer segments.

Each segment has a balance, a baseline retention rate, and a synthetic sensitivity to the offered deposit rate. The optimizer chooses one rate per segment.

## Objective

Minimize expected annual interest cost.

## Constraints

- exactly one rate offer per segment;
- portfolio-level minimum expected retained deposit balance.

## Run

```bash
python -m banking_optimization.deposit_pricing
pytest tests/test_deposit_pricing.py
```

## Why this is an optimization problem

Offering the highest rate to every segment maximizes modeled retention but overpays insensitive segments. Offering the lowest rate everywhere minimizes nominal pricing but can miss the funding target. The optimizer spends rate selectively where marginal retention is valuable.

## Extension path

Discrete-choice or causal retention models, competitor rates, relationship value, repricing lags, deposit beta, non-maturity deposit behavior, liquidity value, transfer pricing, and robust optimization.

## Limitations

Synthetic educational model only. It is not a customer-specific pricing system, deposit recommendation, treasury model, or production behavioral-deposit model.
