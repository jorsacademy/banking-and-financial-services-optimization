# Collateral Allocation Optimization

A linear program that allocates a finite inventory of synthetic collateral to multiple obligations while respecting eligibility and haircut rules.

## Objective

Minimize collateral opportunity cost.

## Constraints

- available inventory by collateral type;
- obligation-specific eligibility;
- haircut-adjusted effective coverage for every obligation.

## Run

```bash
python -m banking_optimization.collateral
pytest tests/test_collateral.py
```

## Why optimization matters

The cheapest collateral is not automatically usable everywhere. Eligibility and haircut differences couple the allocation decisions across obligations, creating a small but genuine network allocation problem.

## Limitations

Educational synthetic model only. Real collateral management includes legal agreements, currencies, settlement timing, substitutions, wrong-way risk, concentration limits, rehypothecation, encumbrance, margin period of risk, operational buffers, and venue-specific eligibility.
