# Cash Supply Chain Optimization

An integrated Operations Research project for daily cash operations across a synthetic ATM and branch network.

The project now contains three architectures:

1. a **staged baseline** that solves replenishment/fleet decisions first and routing second;
2. a **deterministic joint Inventory Routing Problem (IRP)** that chooses replenishment quantities and CIT routes in the same MILP;
3. a **stochastic rolling-horizon IRP** that repeatedly re-optimizes short-horizon route and replenishment decisions under demand scenarios.

The stochastic rolling-horizon layer is the most advanced model in the project.

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

## Stochastic rolling-horizon IRP

Module:

```text
cash_supply_chain.stochastic_rolling_horizon
```

The rolling-horizon controller operates as a receding-horizon policy:

```text
observe current cash state
        ↓
generate short-horizon demand scenarios
        ↓
solve stochastic route-column MILP
        ↓
execute only today's routes and deliveries
        ↓
observe realized withdrawals
        ↓
update inventory and cash-out
        ↓
shift horizon and re-optimize
```

Each stochastic subproblem is a **two-stage stochastic IRP**.

Today's route selections and delivery quantities are here-and-now decisions and are shared across every demand scenario. This is the non-anticipativity condition for the first stage.

Future days use scenario-specific recourse variables for:

- route selection;
- route-specific delivery quantities;
- inventory;
- safety-stock shortfall;
- lost demand / cash-out.

Cash-out is modeled explicitly rather than making high-demand scenarios infeasible. Lost demand receives a high penalty in the expected-cost objective.

The rolling controller then executes only the shared first-day decision. Future scenario-specific decisions are discarded, new information is observed, and the optimization is solved again from the updated state.

### Demand scenarios

Synthetic scenarios use multiplicative lognormal forecast error around the deterministic withdrawal forecast.

The implementation supports configurable:

- planning horizon;
- number of scenarios;
- demand volatility;
- scenario seed.

### Deterministic rolling benchmark

The same receding-horizon framework can also run with the deterministic joint IRP.

Both policies can therefore be evaluated on the **same realized out-of-sample demand path**:

```text
deterministic rolling IRP
vs.
stochastic rolling IRP
```

The comparison reports:

- total realized operating cost;
- realized cash-out;
- average end inventory;
- number of operated routes;
- cost difference versus deterministic rolling;
- cash-out difference versus deterministic rolling.

This is intentionally an out-of-sample policy comparison rather than a claim that the stochastic policy must dominate on every single realized path.

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
        ├── stochastic_rolling_horizon.py
        └── simulation.py
```

## Run

From the repository root:

```bash
pip install -e ".[dev]"
python projects/cash-supply-chain-optimization/run.py

# stochastic receding-horizon experiment
python projects/cash-supply-chain-optimization/run_rolling_horizon.py
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
├── fleet_sensitivity.csv
├── rolling_policy_comparison.csv
├── deterministic_rolling_deliveries.csv
├── deterministic_rolling_routes.csv
├── deterministic_rolling_daily_summary.csv
├── stochastic_rolling_deliveries.csv
├── stochastic_rolling_routes.csv
└── stochastic_rolling_daily_summary.csv
```

Generated outputs are git-ignored.

## Tests

```bash
pytest \
  tests/test_cash_supply_chain.py \
  tests/test_cash_supply_chain_routing.py \
  tests/test_cash_supply_chain_joint_irp.py \
  tests/test_cash_supply_chain_stochastic_rolling.py
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
- stochastic scenario reproducibility;
- first-stage stochastic-route feasibility;
- scenario terminal-state feasibility;
- rolling inventory-state transitions;
- deterministic/stochastic comparison on the same realized path;
- Monte Carlo reproducibility.

## Scaling path

The current route-column formulation is exact and appropriate for a deliberately small research instance.

Larger networks would require methods such as:

- column generation;
- branch-and-price;
- route-generation heuristics;
- decomposition by cash center/geography;
- multistage scenario-tree non-anticipativity beyond the current two-stage look-ahead;
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
- correlated cashpoint demand scenarios;
- regime-dependent forecast errors;
- scenario reduction;
- CVaR / downside-risk objectives;
- multistage scenario-tree recourse;
- distributionally robust demand sets.

## Limitations

This is a synthetic educational optimization system. It is not a production cash-management, ATM, branch, vault, CIT, security, routing, treasury, or operational-risk platform.

Real deployment requires institution-specific forecasting, security policy, geographic data, cash-center processes, crew/vehicle rules, denomination logic, SLAs, calendars, insurance constraints, auditability, and governance.
