# Intraday Liquidity and Payment Scheduling

A mixed-integer model that decides when eligible outgoing payments should be released and how much intraday liquidity buffer is required.

Payments have earliest release times, deadlines, amounts, and delay penalties. Incoming cash arrives during the day.

## Objective

Minimize:

- payment delay penalties;
- cost of the intraday liquidity buffer.

## Constraints

- every payment is released exactly once;
- release must occur between earliest slot and deadline;
- cumulative liquidity cannot become negative;
- each time slot has a processing-count capacity;
- liquidity buffer has an upper bound.

## Run

```bash
python -m banking_optimization.intraday_liquidity
pytest tests/test_intraday_liquidity.py
```

## Extension path

Add RTGS/ACH rails, payment priorities, bilateral limits, queues, collateralized intraday credit, stochastic incoming flows, gridlock resolution, and event-driven re-optimization.

## Limitations

Synthetic educational model only. It is not a payment-settlement, treasury, RTGS, or regulatory intraday-liquidity engine.
