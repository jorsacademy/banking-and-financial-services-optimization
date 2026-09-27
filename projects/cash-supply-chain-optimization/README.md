# Cash Supply Chain Optimization

An integrated Operations Research project for daily cash operations across a synthetic ATM and branch network.

The project now contains two architectures:

1. a **staged baseline** that solves replenishment/fleet decisions first and routing second;
2. a **joint Inventory Routing Problem (IRP)** that chooses replenishment quantities and CIT routes in the same MILP.

The joint IRP is the primary model.

## Decision layers

The project combines:

- cash replenishment planning;
- cashpoint inventory control;
- CIT vehicle deployment;
- route selection;
- route-specific delivered cash;
- central-vault dispatch capacity;
- service-level protection;
- forecast-error simulation;
- what-if analysis.

The model is designed to expose the trade-off among idle cash, visit frequency, route distance, fleet cost, handling cost, and cash-out risk.

## Joint Inventory Routing Problem

Module:

```text
cash_supply_chain.joint_irp
```

For every planning day, the model selects from an enumerated catalog of feasible CIT routes.

Each candidate route:

- starts and ends at the depot;
- contains at most the maximum stops per vehicle;
- uses the exact minimum-distance stop order for that subset;
- has an explicit route distance;
- has a route-specific visit cost.

The MILP jointly chooses:

- which routes to operate;
- how much cash each selected route delivers to each included cashpoint;
- end-of-day inventory;
- safety-stock shortfall.

This means route distance is no longer calculated after replenishment decisions have already been fixed. It is part of the replenishment objective itself.

## Objective

The joint model minimizes:

- cash handling cost;
- end-of-day idle-cash holding cost;
- cashpoint visit cost;
- fixed CIT vehicle/route cost;
- route-distance cost;
- service-shortfall penalty.

A longer or geographically inefficient replenishment plan can therefore lose against a slightly different inventory policy with better route economics.

## Constraints

The integrated MILP includes:

- multi-period inventory conservation;
- cashpoint capacity;
- cashpoint-specific maximum delivery;
- safety-stock targets;
- central-vault daily dispatch capacity;
- route-specific vehicle cash capacity;
- maximum stops embedded in route generation;
- maximum routes/vehicles per day;
- at most one route visit per cashpoint/day;
- cluster-level daily visit limits.

## Route catalog

For the small synthetic network, all cashpoint subsets up to the stop limit are enumerated.

For each subset, the best depot tour is computed exactly.

With six cashpoints and at most three stops per route, the route catalog contains:

```text
C(6,1) + C(6,2) + C(6,3) = 41
```

candidate routes per day.

The optimization then selects a subset of these route columns.

This is a set-partitioning / route-selection formulation rather than an arc-by-arc VRP formulation.

## Staged baseline

The previous architecture remains available as a benchmark:

```text
replenishment + fleet MILP
            ↓
post-optimization exact routing
```

Its routing distance is added to the staged planning cost to obtain a comparable integrated cost.

The project reports:

```text
cost_improvement_vs_staged
    = staged integrated cost - joint IRP cost
```

Because the staged plan is route-feasible for the synthetic instance, it provides a valid benchmark for the joint formulation.

## Forecast-error simulation

Both staged and joint plans are evaluated under the same Monte Carlo demand perturbations.

Simulation reports:

- mean cash-out volume;
- 95th-percentile cash-out volume;
- mean idle cash;
- mean service-event rate.

This makes it possible to distinguish optimization-cost improvement from out-of-sample service robustness.

## Fleet sensitivity

The staged planning layer is also re-solved across alternative maximum fleet sizes to show:

- feasibility;
- cost;
- planned shortage;
- total visits;
- vehicle-days.

The same pattern can be extended to joint-IRP sensitivities.

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
        ├── joint_irp.py
        └── simulation.py
```

## Run

From the repository root:

```bash
pip install -e ".[dev]"
python projects/cash-supply-chain-optimization/run.py
```

The runner solves both architectures and writes:

```text
outputs/
├── staged_deliveries.csv
├── staged_cit_routes.csv
├── joint_deliveries.csv
├── joint_visits.csv
├── joint_end_inventory.csv
├── joint_planned_shortage.csv
├── joint_cit_routes.csv
├── joint_cost_breakdown.csv
├── staged_vs_joint.csv
├── staged_monte_carlo_simulation.csv
├── joint_monte_carlo_simulation.csv
├── staged_simulation_summary.csv
├── joint_simulation_summary.csv
└── fleet_sensitivity.csv
```

Generated outputs are git-ignored.

## Tests

```bash
pytest \
  tests/test_cash_supply_chain.py \
  tests/test_cash_supply_chain_routing.py \
  tests/test_cash_supply_chain_joint_irp.py
```

The tests validate:

- inventory conservation;
- vault capacity;
- route-specific vehicle capacity;
- maximum daily route count;
- cluster visit limits;
- delivery/visit linking;
- route catalog stop limits;
- objective/cost-breakdown consistency;
- joint-vs-staged benchmark consistency;
- Monte Carlo reproducibility.

## Scaling path

The current route-column formulation is exact and appropriate for a deliberately small research instance.

Larger networks would require methods such as:

- column generation;
- branch-and-price;
- route-generation heuristics;
- decomposition by cash center/geography;
- rolling-horizon optimization;
- neighborhood search;
- adaptive large neighborhood search;
- commercial VRP engines;
- stochastic or robust IRP formulations.

## Next research extensions

Natural extensions include:

- separate deposit and withdrawal processes;
- cash collection as well as delivery;
- recycling ATM behavior;
- cassette and denomination constraints;
- multiple cash management centers;
- route time windows;
- crew shift limits;
- insurance/vehicle value limits;
- travel-time uncertainty;
- emergency visits;
- stochastic demand scenarios;
- rolling-horizon re-optimization;
- scenario-dependent route recourse.

## Limitations

This is a synthetic educational optimization system. It is not a production cash-management, ATM, branch, vault, CIT, security, routing, treasury, or operational-risk platform.

Real deployment requires institution-specific forecasting, security policy, geographic data, cash-center processes, crew/vehicle rules, denomination logic, SLAs, calendars, insurance constraints, auditability, and governance.
