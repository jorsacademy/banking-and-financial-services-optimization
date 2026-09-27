# Cash Supply Chain Optimization

An integrated optimization and simulation project for daily cash operations across a synthetic ATM and branch network.

The project combines four decision layers:

1. cash replenishment planning;
2. cash-in-transit resource planning;
3. small-network route construction;
4. forecast-error simulation and what-if analysis.

The goal is to minimize total operating cost while maintaining service levels and respecting physical cash, fleet, vault, and cashpoint constraints.

## Why this is broader than ATM inventory control

A cashpoint should not be optimized in isolation. Replenishment decisions compete for:

- central-vault preparation capacity;
- CIT vehicle capacity;
- available daily vehicle count;
- route/cluster visit capacity;
- cashpoint storage capacity;
- cash tied up as idle inventory.

A lower inventory target can reduce holding cost but increase visit frequency and cash-out risk. Larger deliveries reduce visit frequency but increase idle cash. The model represents these trade-offs jointly.

## Optimization decisions

For each cashpoint and day:

- replenishment amount;
- whether a CIT visit occurs;
- end-of-day cash inventory;
- service shortfall.

For each day:

- number of deployed CIT vehicles.

The optimizer minimizes:

- cash handling cost;
- end-of-day idle-cash holding cost;
- cashpoint visit cost;
- deployed-vehicle cost;
- service-shortfall penalty.

## Operational constraints

The MILP includes:

- daily inventory conservation;
- cashpoint capacity;
- cashpoint-specific maximum delivery;
- safety-stock targets;
- replenishment/visit linking;
- central-vault daily dispatch capacity;
- vehicle cash-carrying capacity;
- maximum stops per vehicle;
- maximum available vehicles per day;
- cluster-level visit limits.

## CIT routing

After the replenishment plan is solved, daily planned visits are routed.

For the small synthetic instance, route generation is exact:

- each visited cashpoint is assigned to one route;
- vehicle load cannot exceed capacity;
- route stop count cannot exceed the operational limit;
- stop order minimizes Euclidean travel distance;
- the final partition minimizes total route distance for the available vehicles.

The routing implementation is deliberately designed for small educational networks. Larger networks would require scalable VRP decomposition, heuristics, metaheuristics, or commercial/open-source routing solvers.

## Forecast-error simulation

The optimized plan is also evaluated under stochastic demand errors.

A Monte Carlo layer perturbs forecast net withdrawals and reports:

- mean cash-out volume;
- 95th-percentile cash-out volume;
- mean idle cash;
- mean service-event rate.

This separates deterministic planning quality from robustness to forecast error.

## What-if analysis

The current experiment layer includes fleet sensitivity:

- re-solve with alternative maximum daily vehicle counts;
- compare feasibility;
- total cost;
- planned service shortage;
- total visits;
- vehicle-days.

The architecture can be extended to safety-stock, interest-cost, CIT-cost, cashpoint-capacity, and demand-volatility scenarios.

## Repository structure

```text
cash-supply-chain-optimization/
├── README.md
├── MATHEMATICAL_MODEL.md
├── run.py
└── src/
    └── cash_supply_chain/
        ├── __init__.py
        ├── model.py
        ├── routing.py
        └── simulation.py
```

Tests live in the umbrella repository's `tests/` directory and run in CI.

## Run

From the repository root:

```bash
pip install -e ".[dev]"
python projects/cash-supply-chain-optimization/run.py
```

Generated outputs are written to:

```text
projects/cash-supply-chain-optimization/outputs/
├── deliveries.csv
├── visits.csv
├── end_inventory.csv
├── planned_shortage.csv
├── vehicles.csv
├── cost_breakdown.csv
├── cit_routes.csv
├── monte_carlo_simulation.csv
├── simulation_summary.csv
└── fleet_sensitivity.csv
```

The outputs directory is git-ignored because it contains generated artifacts.

Run the dedicated tests with:

```bash
pytest tests/test_cash_supply_chain.py tests/test_cash_supply_chain_routing.py
```

## Extension path

Natural next steps include:

- cash deposit and withdrawal forecasts as separate stochastic processes;
- recycling ATM behavior;
- cassette/denomination constraints;
- branch teller and lobby-ATM pooling;
- multiple cash management centers;
- collection as well as replenishment decisions;
- vehicle insurance limits;
- route time windows and SLAs;
- travel-time uncertainty;
- emergency/ad-hoc visits;
- rolling-horizon dynamic re-optimization;
- joint replenishment-routing optimization;
- predictive maintenance interactions.

## Limitations

This is a synthetic educational decision model. It is not a production cash-management, vault, ATM, branch, CIT, security, routing, or treasury system.

Real deployment would require institution-specific forecasting, security rules, vehicle and crew constraints, geographic data, cash-center processes, contractual SLAs, denomination/cassette logic, operating calendars, insurance rules, and governance controls.
