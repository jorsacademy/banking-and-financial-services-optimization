"""Two-stage stochastic IRP inside a rolling-horizon controller.

At each replanning date:
1. generate demand scenarios for a short look-ahead horizon;
2. choose today's routes and deliveries here-and-now;
3. allow scenario-specific future route/delivery recourse;
4. execute only today's shared decision;
5. observe realized demand and update inventory;
6. roll the horizon forward and solve again.

The stochastic subproblem is a route-column MILP. Future recourse is scenario-
specific, so the model is two-stage within each rolling-horizon solve rather
than a full multistage scenario tree.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .joint_irp import generate_route_catalog, solve_joint_irp
from .model import CashSupplyChainProblem, default_problem


@dataclass(frozen=True)
class StochasticHorizonResult:
    first_day_deliveries: pd.Series
    first_day_visits: pd.Series
    first_day_routes: pd.DataFrame
    expected_total_cost: float
    expected_cashout: float
    expected_safety_shortfall: float
    scenario_terminal_inventory: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "first_day_deliveries": self.first_day_deliveries.round(6).to_dict(),
            "first_day_visits": self.first_day_visits.astype(int).to_dict(),
            "first_day_routes": self.first_day_routes.round(
                {"load": 6, "distance": 6}
            ).to_dict(orient="records"),
            "expected_total_cost": round(self.expected_total_cost, 6),
            "expected_cashout": round(self.expected_cashout, 6),
            "expected_safety_shortfall": round(
                self.expected_safety_shortfall, 6
            ),
            "scenario_terminal_inventory": self.scenario_terminal_inventory.round(
                6
            ).to_dict(orient="index"),
        }


@dataclass(frozen=True)
class RollingHorizonResult:
    policy: str
    deliveries: pd.DataFrame
    realized_demand: pd.DataFrame
    end_inventory: pd.DataFrame
    cashout: pd.DataFrame
    selected_routes: pd.DataFrame
    daily_summary: pd.DataFrame
    total_realized_cost: float

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "deliveries": self.deliveries.round(6).to_dict(),
            "realized_demand": self.realized_demand.round(6).to_dict(),
            "end_inventory": self.end_inventory.round(6).to_dict(),
            "cashout": self.cashout.round(6).to_dict(),
            "selected_routes": self.selected_routes.round(
                {"load": 6, "distance": 6}
            ).to_dict(orient="records"),
            "daily_summary": self.daily_summary.round(6).to_dict(orient="index"),
            "total_realized_cost": round(self.total_realized_cost, 6),
        }


def generate_demand_scenarios(
    problem: CashSupplyChainProblem,
    days: list[int],
    scenario_count: int = 4,
    demand_sigma: float = 0.12,
    seed: int = 0,
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Generate equiprobable lognormal demand scenarios around the forecast."""
    if scenario_count < 1:
        raise ValueError("scenario_count must be positive")
    if demand_sigma < 0:
        raise ValueError("demand_sigma must be nonnegative")

    rng = np.random.default_rng(seed)
    base = problem.forecast_net_withdrawal.loc[:, days].astype(float)

    scenarios: dict[str, pd.DataFrame] = {}
    for k in range(scenario_count):
        factors = rng.lognormal(
            mean=-0.5 * demand_sigma**2,
            sigma=demand_sigma,
            size=base.shape,
        )
        scenarios[f"S{k + 1:02d}"] = pd.DataFrame(
            base.to_numpy() * factors,
            index=base.index,
            columns=base.columns,
        )

    probabilities = pd.Series(
        1.0 / scenario_count,
        index=list(scenarios),
        name="probability",
    )
    return scenarios, probabilities


def generate_realized_demand(
    problem: CashSupplyChainProblem,
    demand_sigma: float = 0.12,
    seed: int = 17,
) -> pd.DataFrame:
    """Create one out-of-sample realized demand path."""
    rng = np.random.default_rng(seed)
    base = problem.forecast_net_withdrawal.astype(float)
    factors = rng.lognormal(
        mean=-0.5 * demand_sigma**2,
        sigma=demand_sigma,
        size=base.shape,
    )
    return pd.DataFrame(
        base.to_numpy() * factors,
        index=base.index,
        columns=base.columns,
    )


def solve_stochastic_horizon(
    problem: CashSupplyChainProblem,
    current_inventory: pd.Series,
    days: list[int],
    demand_scenarios: dict[str, pd.DataFrame],
    probabilities: pd.Series | None = None,
) -> StochasticHorizonResult:
    """Solve a two-stage stochastic route-column IRP for one look-ahead window.

    Today's route and delivery variables are shared across scenarios. Decisions
    from the second horizon day onward are scenario-specific recourse.
    """
    if not days:
        raise ValueError("days cannot be empty")
    if not demand_scenarios:
        raise ValueError("demand_scenarios cannot be empty")

    p = problem
    cashpoints = list(p.cashpoints.index)
    first_day = days[0]
    future_days = days[1:]
    scenario_names = list(demand_scenarios)

    if probabilities is None:
        probabilities = pd.Series(
            1.0 / len(scenario_names),
            index=scenario_names,
        )
    probabilities = probabilities.loc[scenario_names].astype(float)
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("scenario probabilities must sum to one")

    current_inventory = current_inventory.loc[cashpoints].astype(float)

    for name, demand in demand_scenarios.items():
        if list(demand.index) != cashpoints:
            raise ValueError(f"scenario {name} cashpoints do not match problem")
        if list(demand.columns) != days:
            raise ValueError(f"scenario {name} days do not match planning horizon")

    catalog = generate_route_catalog(p)
    routes = list(catalog.index)

    route_stop_keys = [
        (route, cashpoint)
        for route in routes
        for cashpoint in catalog.loc[route, "stops"]
    ]

    # Variable blocks:
    # shared first-day z[r], q[r,i]
    # scenario-specific future z[s,t,r], q[s,t,r,i]
    # scenario-specific inventory I[s,i,t]
    # safety shortfall h[s,i,t]
    # lost demand / cashout l[s,i,t]
    z1_0 = 0
    q1_0 = z1_0 + len(routes)

    zf_keys = [
        (scenario, day, route)
        for scenario in scenario_names
        for day in future_days
        for route in routes
    ]
    zf_0 = q1_0 + len(route_stop_keys)

    qf_keys = [
        (scenario, day, route, cashpoint)
        for scenario in scenario_names
        for day in future_days
        for route in routes
        for cashpoint in catalog.loc[route, "stops"]
    ]
    qf_0 = zf_0 + len(zf_keys)

    state_keys = [
        (scenario, cashpoint, day)
        for scenario in scenario_names
        for cashpoint in cashpoints
        for day in days
    ]
    inv_0 = qf_0 + len(qf_keys)
    safe_0 = inv_0 + len(state_keys)
    lost_0 = safe_0 + len(state_keys)
    n_vars = lost_0 + len(state_keys)

    route_i = {route: k for k, route in enumerate(routes)}
    q1_i = {key: k for k, key in enumerate(route_stop_keys)}
    zf_i = {key: k for k, key in enumerate(zf_keys)}
    qf_i = {key: k for k, key in enumerate(qf_keys)}
    state_i = {key: k for k, key in enumerate(state_keys)}

    c = np.zeros(n_vars)

    # Shared first-day operating costs.
    for route in routes:
        c[z1_0 + route_i[route]] = (
            p.vehicle_fixed_cost
            + float(catalog.loc[route, "visit_cost"])
            + p.distance_cost_per_unit * float(catalog.loc[route, "distance"])
        )
    for key, k in q1_i.items():
        c[q1_0 + k] = p.handling_cost_per_unit

    # Scenario-weighted future recourse and state costs.
    for scenario in scenario_names:
        prob = float(probabilities.loc[scenario])

        for day in future_days:
            for route in routes:
                c[zf_0 + zf_i[(scenario, day, route)]] = prob * (
                    p.vehicle_fixed_cost
                    + float(catalog.loc[route, "visit_cost"])
                    + p.distance_cost_per_unit * float(
                        catalog.loc[route, "distance"]
                    )
                )

            for route in routes:
                for cashpoint in catalog.loc[route, "stops"]:
                    c[
                        qf_0 + qf_i[(scenario, day, route, cashpoint)]
                    ] = prob * p.handling_cost_per_unit

        for cashpoint in cashpoints:
            for day in days:
                key = (scenario, cashpoint, day)
                c[inv_0 + state_i[key]] = (
                    prob * p.holding_cost_per_unit_day
                )
                c[safe_0 + state_i[key]] = (
                    prob * p.shortage_penalty_per_unit
                )
                c[lost_0 + state_i[key]] = (
                    prob * p.cashout_penalty_per_unit
                )

    eq_rows: list[np.ndarray] = []
    eq_rhs: list[float] = []

    # Scenario inventory balance. First-day delivery is shared, demand is not.
    for scenario in scenario_names:
        demand = demand_scenarios[scenario]

        for cashpoint in cashpoints:
            for day_pos, day in enumerate(days):
                row = np.zeros(n_vars)

                if day_pos == 0:
                    for route in routes:
                        if cashpoint in catalog.loc[route, "stops"]:
                            row[
                                q1_0 + q1_i[(route, cashpoint)]
                            ] = 1.0
                else:
                    for route in routes:
                        if cashpoint in catalog.loc[route, "stops"]:
                            row[
                                qf_0
                                + qf_i[(scenario, day, route, cashpoint)]
                            ] = 1.0
                    previous_day = days[day_pos - 1]
                    row[
                        inv_0
                        + state_i[(scenario, cashpoint, previous_day)]
                    ] = 1.0

                row[
                    lost_0 + state_i[(scenario, cashpoint, day)]
                ] = 1.0
                row[
                    inv_0 + state_i[(scenario, cashpoint, day)]
                ] = -1.0

                rhs = float(demand.loc[cashpoint, day])
                if day_pos == 0:
                    rhs -= float(current_inventory.loc[cashpoint])

                eq_rows.append(row)
                eq_rhs.append(rhs)

    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []

    # Shared first-day route assignment: at most one route per cashpoint.
    for cashpoint in cashpoints:
        row = np.zeros(n_vars)
        for route in routes:
            if cashpoint in catalog.loc[route, "stops"]:
                row[z1_0 + route_i[route]] = 1.0
        ub_rows.append(row)
        ub_rhs.append(1.0)

    # Shared first-day route capacity and delivery linking.
    for route in routes:
        z_idx = z1_0 + route_i[route]
        capacity_row = np.zeros(n_vars)

        for cashpoint in catalog.loc[route, "stops"]:
            q_idx = q1_0 + q1_i[(route, cashpoint)]
            capacity_row[q_idx] = 1.0

            row = np.zeros(n_vars)
            row[q_idx] = 1.0
            row[z_idx] = -float(
                p.cashpoints.loc[cashpoint, "maximum_delivery"]
            )
            ub_rows.append(row)
            ub_rhs.append(0.0)

        capacity_row[z_idx] = -p.vehicle_capacity
        ub_rows.append(capacity_row)
        ub_rhs.append(0.0)

    # Shared first-day vault, fleet, cluster, and cashpoint capacity.
    row = np.zeros(n_vars)
    for route, cashpoint in route_stop_keys:
        row[q1_0 + q1_i[(route, cashpoint)]] = 1.0
    ub_rows.append(row)
    ub_rhs.append(float(p.vault_dispatch_limit.loc[first_day]))

    row = np.zeros(n_vars)
    for route in routes:
        row[z1_0 + route_i[route]] = 1.0
    ub_rows.append(row)
    ub_rhs.append(float(p.maximum_vehicles_per_day))

    for cluster, limit in p.cluster_visit_limit.items():
        members = set(
            p.cashpoints.index[p.cashpoints["cluster"] == cluster]
        )
        row = np.zeros(n_vars)
        for route in routes:
            count = len(members.intersection(catalog.loc[route, "stops"]))
            if count:
                row[z1_0 + route_i[route]] = float(count)
        ub_rows.append(row)
        ub_rhs.append(float(limit))

    for cashpoint in cashpoints:
        row = np.zeros(n_vars)
        for route in routes:
            if cashpoint in catalog.loc[route, "stops"]:
                row[q1_0 + q1_i[(route, cashpoint)]] = 1.0
        ub_rows.append(row)
        ub_rhs.append(
            float(p.cashpoints.loc[cashpoint, "capacity"])
            - float(current_inventory.loc[cashpoint])
        )

    # Scenario-specific future operational constraints.
    for scenario in scenario_names:
        for day_pos, day in enumerate(days):
            # State service constraints and lost-demand upper bounds.
            demand = demand_scenarios[scenario]
            for cashpoint in cashpoints:
                state_key = (scenario, cashpoint, day)

                row = np.zeros(n_vars)
                row[inv_0 + state_i[state_key]] = -1.0
                row[safe_0 + state_i[state_key]] = -1.0
                ub_rows.append(row)
                ub_rhs.append(
                    -float(p.cashpoints.loc[cashpoint, "safety_stock"])
                )

                row = np.zeros(n_vars)
                row[lost_0 + state_i[state_key]] = 1.0
                ub_rows.append(row)
                ub_rhs.append(float(demand.loc[cashpoint, day]))

            if day_pos == 0:
                continue

            # Route assignment.
            for cashpoint in cashpoints:
                row = np.zeros(n_vars)
                for route in routes:
                    if cashpoint in catalog.loc[route, "stops"]:
                        row[
                            zf_0 + zf_i[(scenario, day, route)]
                        ] = 1.0
                ub_rows.append(row)
                ub_rhs.append(1.0)

            # Route delivery linking and route-specific vehicle capacity.
            for route in routes:
                z_idx = zf_0 + zf_i[(scenario, day, route)]
                capacity_row = np.zeros(n_vars)

                for cashpoint in catalog.loc[route, "stops"]:
                    q_idx = (
                        qf_0
                        + qf_i[(scenario, day, route, cashpoint)]
                    )
                    capacity_row[q_idx] = 1.0

                    row = np.zeros(n_vars)
                    row[q_idx] = 1.0
                    row[z_idx] = -float(
                        p.cashpoints.loc[cashpoint, "maximum_delivery"]
                    )
                    ub_rows.append(row)
                    ub_rhs.append(0.0)

                capacity_row[z_idx] = -p.vehicle_capacity
                ub_rows.append(capacity_row)
                ub_rhs.append(0.0)

            # Vault dispatch.
            row = np.zeros(n_vars)
            for route in routes:
                for cashpoint in catalog.loc[route, "stops"]:
                    row[
                        qf_0
                        + qf_i[(scenario, day, route, cashpoint)]
                    ] = 1.0
            ub_rows.append(row)
            ub_rhs.append(float(p.vault_dispatch_limit.loc[day]))

            # Maximum routes/vehicles.
            row = np.zeros(n_vars)
            for route in routes:
                row[zf_0 + zf_i[(scenario, day, route)]] = 1.0
            ub_rows.append(row)
            ub_rhs.append(float(p.maximum_vehicles_per_day))

            # Cluster visit limits.
            for cluster, limit in p.cluster_visit_limit.items():
                members = set(
                    p.cashpoints.index[
                        p.cashpoints["cluster"] == cluster
                    ]
                )
                row = np.zeros(n_vars)
                for route in routes:
                    count = len(
                        members.intersection(catalog.loc[route, "stops"])
                    )
                    if count:
                        row[
                            zf_0 + zf_i[(scenario, day, route)]
                        ] = float(count)
                ub_rows.append(row)
                ub_rhs.append(float(limit))

            # Cashpoint capacity before demand realization.
            previous_day = days[day_pos - 1]
            for cashpoint in cashpoints:
                row = np.zeros(n_vars)
                row[
                    inv_0 + state_i[(scenario, cashpoint, previous_day)]
                ] = 1.0
                for route in routes:
                    if cashpoint in catalog.loc[route, "stops"]:
                        row[
                            qf_0
                            + qf_i[(scenario, day, route, cashpoint)]
                        ] = 1.0
                ub_rows.append(row)
                ub_rhs.append(
                    float(p.cashpoints.loc[cashpoint, "capacity"])
                )

    constraints = [
        LinearConstraint(
            np.vstack(eq_rows),
            lb=np.asarray(eq_rhs, dtype=float),
            ub=np.asarray(eq_rhs, dtype=float),
        ),
        LinearConstraint(
            np.vstack(ub_rows),
            lb=np.full(len(ub_rows), -np.inf),
            ub=np.asarray(ub_rhs, dtype=float),
        ),
    ]

    lower = np.zeros(n_vars)
    upper = np.full(n_vars, np.inf)

    # Route variables are binary.
    upper[z1_0:q1_0] = 1.0
    upper[zf_0:qf_0] = 1.0

    # Inventory capacity.
    for scenario, cashpoint, day in state_keys:
        upper[
            inv_0 + state_i[(scenario, cashpoint, day)]
        ] = float(p.cashpoints.loc[cashpoint, "capacity"])

    integrality = np.zeros(n_vars, dtype=int)
    integrality[z1_0:q1_0] = 1
    integrality[zf_0:qf_0] = 1

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(
            f"stochastic rolling-horizon IRP failed: {result.message}"
        )

    first_day_deliveries = pd.Series(
        0.0,
        index=cashpoints,
        name="delivery",
    )
    first_day_visits = pd.Series(
        0,
        index=cashpoints,
        dtype=int,
        name="visit",
    )
    route_rows = []

    for route in routes:
        if result.x[z1_0 + route_i[route]] <= 0.5:
            continue

        load = 0.0
        for cashpoint in catalog.loc[route, "stops"]:
            amount = result.x[q1_0 + q1_i[(route, cashpoint)]]
            first_day_deliveries.loc[cashpoint] += amount
            first_day_visits.loc[cashpoint] = 1
            load += amount

        route_rows.append(
            {
                "day": first_day,
                "route_id": route,
                "sequence": catalog.loc[route, "sequence"],
                "load": float(load),
                "distance": float(catalog.loc[route, "distance"]),
                "stop_count": int(catalog.loc[route, "stop_count"]),
            }
        )

    first_day_routes = pd.DataFrame(
        route_rows,
        columns=[
            "day",
            "route_id",
            "sequence",
            "load",
            "distance",
            "stop_count",
        ],
    )

    expected_cashout = 0.0
    expected_safety_shortfall = 0.0
    terminal = pd.DataFrame(
        0.0,
        index=scenario_names,
        columns=cashpoints,
    )

    last_day = days[-1]
    for scenario in scenario_names:
        prob = float(probabilities.loc[scenario])
        for cashpoint in cashpoints:
            for day in days:
                key = (scenario, cashpoint, day)
                expected_cashout += (
                    prob * result.x[lost_0 + state_i[key]]
                )
                expected_safety_shortfall += (
                    prob * result.x[safe_0 + state_i[key]]
                )

            terminal.loc[scenario, cashpoint] = result.x[
                inv_0 + state_i[(scenario, cashpoint, last_day)]
            ]

    return StochasticHorizonResult(
        first_day_deliveries=first_day_deliveries,
        first_day_visits=first_day_visits,
        first_day_routes=first_day_routes,
        expected_total_cost=float(result.fun),
        expected_cashout=float(expected_cashout),
        expected_safety_shortfall=float(expected_safety_shortfall),
        scenario_terminal_inventory=terminal,
    )


def _realized_day_cost(
    problem: CashSupplyChainProblem,
    deliveries: pd.Series,
    routes: pd.DataFrame,
    end_inventory: pd.Series,
    cashout: pd.Series,
) -> tuple[float, dict[str, float]]:
    handling = float(deliveries.sum() * problem.handling_cost_per_unit)
    holding = float(end_inventory.sum() * problem.holding_cost_per_unit_day)
    safety_shortfall = float(
        (
            problem.cashpoints["safety_stock"] - end_inventory
        ).clip(lower=0.0).sum()
    )
    safety_cost = safety_shortfall * problem.shortage_penalty_per_unit
    cashout_cost = float(cashout.sum() * problem.cashout_penalty_per_unit)

    if routes.empty:
        route_cost = 0.0
    else:
        route_visit_cost = 0.0
        for sequence in routes["sequence"]:
            stops = [
                token.strip()
                for token in sequence.split("->")
                if token.strip() != "DEPOT"
            ]
            route_visit_cost += float(
                problem.cashpoints.loc[stops, "visit_cost"].sum()
            )

        route_cost = (
            len(routes) * problem.vehicle_fixed_cost
            + float(routes["distance"].sum())
            * problem.distance_cost_per_unit
            + route_visit_cost
        )

    breakdown = {
        "handling_cost": handling,
        "holding_cost": holding,
        "safety_shortfall_cost": safety_cost,
        "cashout_cost": cashout_cost,
        "route_cost": route_cost,
    }
    return float(sum(breakdown.values())), breakdown


def _truncate_problem(
    problem: CashSupplyChainProblem,
    current_inventory: pd.Series,
    days: list[int],
) -> CashSupplyChainProblem:
    cashpoints = problem.cashpoints.copy()
    cashpoints.loc[:, "initial_cash"] = current_inventory.loc[
        cashpoints.index
    ].to_numpy()

    return replace(
        problem,
        cashpoints=cashpoints,
        forecast_net_withdrawal=problem.forecast_net_withdrawal.loc[:, days],
        vault_dispatch_limit=problem.vault_dispatch_limit.loc[days],
    )


def run_rolling_horizon(
    problem: CashSupplyChainProblem | None = None,
    policy: str = "stochastic",
    horizon_days: int = 3,
    scenario_count: int = 4,
    demand_sigma: float = 0.12,
    scenario_seed: int = 101,
    realized_demand: pd.DataFrame | None = None,
    realized_seed: int = 17,
) -> RollingHorizonResult:
    """Execute a deterministic or stochastic receding-horizon policy."""
    p = problem or default_problem()

    if policy not in {"deterministic", "stochastic"}:
        raise ValueError("policy must be deterministic or stochastic")
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")

    days = list(p.forecast_net_withdrawal.columns)
    cashpoints = list(p.cashpoints.index)

    realized = (
        realized_demand.loc[cashpoints, days].astype(float)
        if realized_demand is not None
        else generate_realized_demand(
            p,
            demand_sigma=demand_sigma,
            seed=realized_seed,
        )
    )

    current_inventory = p.cashpoints.loc[
        cashpoints, "initial_cash"
    ].astype(float).copy()

    deliveries = pd.DataFrame(0.0, index=cashpoints, columns=days)
    end_inventory = pd.DataFrame(0.0, index=cashpoints, columns=days)
    cashout = pd.DataFrame(0.0, index=cashpoints, columns=days)

    route_frames = []
    summary_rows = []
    total_realized_cost = 0.0

    for day_pos, day in enumerate(days):
        window = days[day_pos : day_pos + horizon_days]
        truncated = _truncate_problem(p, current_inventory, window)

        if policy == "stochastic":
            scenarios, probabilities = generate_demand_scenarios(
                truncated,
                window,
                scenario_count=scenario_count,
                demand_sigma=demand_sigma,
                seed=scenario_seed + day_pos,
            )
            plan = solve_stochastic_horizon(
                truncated,
                current_inventory=current_inventory,
                days=window,
                demand_scenarios=scenarios,
                probabilities=probabilities,
            )
            today_delivery = plan.first_day_deliveries
            today_routes = plan.first_day_routes.copy()
            planned_cost = plan.expected_total_cost
            planned_cashout = plan.expected_cashout
        else:
            deterministic = solve_joint_irp(truncated)
            today_delivery = deterministic.deliveries[window[0]].copy()
            today_routes = deterministic.selected_routes[
                deterministic.selected_routes["day"] == window[0]
            ].copy()
            planned_cost = deterministic.total_cost
            planned_cashout = 0.0

        before = current_inventory + today_delivery
        demand_today = realized[day]
        realized_cashout = (demand_today - before).clip(lower=0.0)
        after = (before - demand_today).clip(lower=0.0)

        day_cost, breakdown = _realized_day_cost(
            p,
            today_delivery,
            today_routes,
            after,
            realized_cashout,
        )
        total_realized_cost += day_cost

        deliveries.loc[:, day] = today_delivery
        end_inventory.loc[:, day] = after
        cashout.loc[:, day] = realized_cashout

        if not today_routes.empty:
            today_routes = today_routes.copy()
            today_routes["rolling_day"] = day
            route_frames.append(today_routes)

        summary_rows.append(
            {
                "day": day,
                "planned_horizon_cost": planned_cost,
                "planned_expected_cashout": planned_cashout,
                "realized_delivery": float(today_delivery.sum()),
                "realized_cashout": float(realized_cashout.sum()),
                "end_inventory": float(after.sum()),
                "realized_day_cost": day_cost,
                **breakdown,
            }
        )

        current_inventory = after

    selected_routes = (
        pd.concat(route_frames, ignore_index=True)
        if route_frames
        else pd.DataFrame(
            columns=[
                "day",
                "route_id",
                "sequence",
                "load",
                "distance",
                "stop_count",
                "rolling_day",
            ]
        )
    )

    daily_summary = pd.DataFrame(summary_rows).set_index("day")

    return RollingHorizonResult(
        policy=policy,
        deliveries=deliveries,
        realized_demand=realized,
        end_inventory=end_inventory,
        cashout=cashout,
        selected_routes=selected_routes,
        daily_summary=daily_summary,
        total_realized_cost=float(total_realized_cost),
    )


def compare_rolling_policies(
    problem: CashSupplyChainProblem | None = None,
    horizon_days: int = 3,
    scenario_count: int = 4,
    demand_sigma: float = 0.12,
    seed: int = 17,
) -> tuple[pd.DataFrame, dict[str, RollingHorizonResult]]:
    """Evaluate deterministic and stochastic rolling policies on one path."""
    p = problem or default_problem()
    realized = generate_realized_demand(
        p,
        demand_sigma=demand_sigma,
        seed=seed,
    )

    deterministic = run_rolling_horizon(
        p,
        policy="deterministic",
        horizon_days=horizon_days,
        demand_sigma=demand_sigma,
        realized_demand=realized,
    )
    stochastic = run_rolling_horizon(
        p,
        policy="stochastic",
        horizon_days=horizon_days,
        scenario_count=scenario_count,
        demand_sigma=demand_sigma,
        scenario_seed=seed + 10_000,
        realized_demand=realized,
    )

    rows = []
    for result in [deterministic, stochastic]:
        rows.append(
            {
                "policy": result.policy,
                "total_realized_cost": result.total_realized_cost,
                "total_cashout": float(result.cashout.to_numpy().sum()),
                "average_end_inventory": float(
                    result.end_inventory.to_numpy().mean()
                ),
                "routes": len(result.selected_routes),
            }
        )

    comparison = pd.DataFrame(rows).set_index("policy")
    comparison["cost_difference_vs_deterministic"] = (
        comparison.loc["deterministic", "total_realized_cost"]
        - comparison["total_realized_cost"]
    )
    comparison["cashout_difference_vs_deterministic"] = (
        comparison.loc["deterministic", "total_cashout"]
        - comparison["total_cashout"]
    )

    return comparison, {
        "deterministic": deterministic,
        "stochastic": stochastic,
    }
