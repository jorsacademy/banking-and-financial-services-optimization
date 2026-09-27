# Liquidity and Funding Optimization

A linear program for choosing a synthetic bank funding mix and liquidity buffer under multi-horizon stressed runoff assumptions.

This project is intentionally narrower than the ALM model. It focuses on **funding economics and liquidity resilience** rather than the entire balance sheet.

## Decisions

- amount raised from each funding source;
- size of the liquidity buffer.

## Objective

Minimize:

- annualized funding cost;
- liquidity-buffer carry cost.

## Constraints

- funding raised must cover the core funding need plus the buffer;
- buffer must cover stressed fixed outflows plus funding-source runoff at 1, 3, 6, and 12 months;
- stable funding must exceed a minimum share;
- each funding source has capacity bounds;
- buffer size has an upper bound.

## Run

```bash
python -m banking_optimization.liquidity_funding
pytest tests/test_liquidity_funding.py
```

## Modeling note

The stress runoff coefficients are synthetic proxies. Their purpose is to demonstrate a tractable optimization architecture in which cheap-but-unstable funding creates additional liquidity-buffer requirements.

## Limitations

This is not an LCR, NSFR, liquidity stress-testing, recovery-planning, treasury, or regulatory reporting engine. Real implementations require contractual cash-flow ladders, behavioral assumptions, collateral and central-bank eligibility, encumbrance, intraday liquidity, currency mismatches, contingency funding plans, and institution-specific regulatory interpretation.
