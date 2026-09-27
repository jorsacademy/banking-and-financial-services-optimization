# ATM Cash Replenishment Optimization

A multi-period mixed-integer inventory model for deciding when and how much cash to deliver to a small synthetic ATM network.

## Objective

Minimize:

- fixed replenishment/visit cost;
- cash handling cost;
- end-of-day cash holding cost.

## Constraints

- daily inventory balance;
- ATM safety stock;
- ATM cash capacity;
- maximum cash per delivery;
- delivery requires a visit;
- limited number of ATM visits per day.

## Run

```bash
python -m banking_optimization.atm_cash
pytest tests/test_atm_cash.py
```

## Extension path

Demand uncertainty, cash-out penalties, vehicle routing, armored-car depot constraints, denomination mix, holiday effects, multi-day stochastic programming, and rolling-horizon re-optimization.

## Limitations

Synthetic educational model only. It is not a physical-cash forecasting, security, logistics, or ATM operating system.
