# Asset-Liability Management Optimization

A synthetic bank balance-sheet allocation model that chooses asset mix, funding mix, and a simple interest-rate hedge under multiple rate/funding scenarios.

The purpose is to demonstrate **prescriptive ALM**: the model does not merely calculate balance-sheet risk; it chooses a feasible balance-sheet configuration.

## Decision variables

- amount allocated to each asset bucket;
- amount raised from each funding source;
- positive/negative hedge notional;
- scenario-specific NII shortfall variables.

## Objective

Minimize the negative of probability-weighted net interest income, plus penalties for:

- NII falling below a target in individual scenarios;
- gross hedge notional.

Equivalently, the model maximizes expected NII while discouraging scenario downside and unnecessary hedging.

## Constraints

The implementation includes:

- asset-side balance-sheet identity;
- liability/funding-side balance-sheet identity;
- minimum/maximum bucket allocations;
- a capital-ratio proxy using risk-weighted assets;
- a liquidity-buffer proxy using liquid-asset weights and funding runoff rates;
- an absolute duration-gap limit;
- scenario NII shortfall accounting;
- a maximum hedge notional.

## Synthetic scenarios

The default instance uses four scenarios:

- base;
- rates up;
- rates down;
- funding stress.

All inputs are synthetic and intentionally small enough to inspect.

## Run

From the repository root:

```bash
pip install -e ".[dev]"
python -m banking_optimization.alm
```

Run tests:

```bash
pytest tests/test_alm.py
```

## Why this is an optimization project

ALM is often presented as a measurement exercise. Here the exposures themselves are decision variables. Capital, liquidity, duration, funding mix, and earnings are therefore coupled inside one optimization model.

## Limitations

This is not a regulatory ALM, IRRBB, LCR, NSFR, FTP, EVE, behavioral-deposit, prepayment, or hedge-accounting engine. The ratios are educational proxies. A real implementation requires institution-specific cash-flow models, behavioral assumptions, curve construction, transfer pricing, accounting treatment, governance, validation, and regulatory interpretation.

## Disclaimer

Educational use only. This project is not financial, investment, accounting, legal, credit, or regulatory advice.
