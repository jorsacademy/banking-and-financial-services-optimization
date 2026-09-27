"""Joint inventory-routing optimization for cash supply chains.

Unlike the staged baseline, this model chooses replenishment quantities and
CIT routes in one MILP. Route distance therefore affects inventory decisions
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .model import (
    CashSupplyChainProblem,
    CashSupplyChainResult,
    default_problem,
    solve as solve_staged,
)
from .routing import _best_tour, route_plan


@dataclass(frozen=True)
class JointIRPResult:
    deliveries: pd.DataFrame
    visits: pd.DataFrame
    end_inventory: pd.DataFrame
    shortage: pd.DataFrame
    selected_routes: pd.DataFrame
    cost_breakdown: pd.Series
    total_cost: float

    def to_dict(self) -> dict:
        return {
            "deliveries": self.deliveries.round(6).to_dict(),
            "visits": self.visits.astype(int).to_dict(),
            "end_inventory": self.end_inventory.round(6).to_dict(),
            "shortage": self.shortage.round(6).to_dict(),
            "selected_routes": self.selected_routes.round(
                {"load": 6, "distance": 6}
            ).to_dict(orient="records"),
            "cost_breakdown": self.cost_breakdown.round(6).to_dict(),
            "total_cost": round(self.total_cost, 6),
        }


def generate_route_catalog(
    problem: CashSupplyChainProblem | None = None,
) -> pd.DataFrame:
    """Enumerate all stop subsets up to the per-vehicle stop limit.

    For each subset, the minimum-distance depot tour is computed exactly.
    """
    p = problem or default_problem()
    cashpoints = list(p.cashpoints.index)

    rows = []
    route_number = 1

    for size in range(1, min(p.stops_per_vehicle, len(cashpoints)) + 1):
        for subset in combinations(cashpoints, size):
            distance, order = _best_tour(tuple(subset), p)
            rows.append(
                {
                    "route_id": f"R{route_number:03d}",
                    "stops": tuple(subset),
                    "sequence": " -> ".join(("DEPOT", *order, "DEPOT")),
                    "distance": float(distance),
                    "stop_count": len(subset),
                    "visit_cost": float(
                        p.cashpoints.loc[list(subset), "visit_cost"].sum()
                    ),
                }
            )
            route_number += 1

    return pd.DataFrame(rows).set_index("route_id")


def solve_joint_irp(
    problem: CashSupplyChainProblem | None = None,
) -> JointIRPResult:
    """Solve the integrated replenishment-routing MILP."""
    p = problem or default_problem()
    cashpoints = list(p.cashpoints.index)
    days = list(p.forecast_net_withdrawal.columns)
    catalog = generate_route_catalog(p)
    routes = list(catalog.index)

    route_day_pairs = [(day, route) for day in days for route in routes]
    route_index = {pair: k for k, pair in enumerate(route_day_pairs)}
    n_z = len(route_day_pairs)

    delivery_keys: list[tuple[int, str, str]] = []
    for day in days:
        for route in routes:
            for cashpoint in catalog.loc[route, "stops"]:
                delivery_keys.append((day, route, cashpoint))
    delivery_index = {key: k for k, key in enumerate(delivery_keys)}
    n_q = len(delivery_keys)

    inv_keys = [(cashpoint, day) for cashpoint in cashpoints for day in days]
    inv_index = {key: k for k, key in enumerate(inv_keys)}
    n_i = len(inv_keys)

    # Variable blocks:
    # z[day, route] binary route-selection
    # q[day, route, cashpoint] cash delivered on that route
    # I[cashpoint, day] end inventory
    # s[cashpoint, day] safety-stock shortfall
    z0 = 0
    q0 = n_z
    i0 = q0 + n_q
    s0 = i0 + n_i
    n_vars = s0 + n_i

    c = np.zeros(n_vars)

    for day, route in route_day_pairs:
        idx = z0 + route_index[(day, route)]
        c[idx] = (
            p.vehicle_fixed_cost
            + float(catalog.loc[route, "visit_cost"])
            + p.distance_cost_per_unit * float(catalog.loc[route, "distance"])
        )

    for key, k in delivery_index.items():
        c[q0 + k] = p.handling_cost_per_unit

    for key, k in inv_index.items():
        c[i0 + k] = p.holding_cost_per_unit_day
        c[s0 + k] = p.shortage_penalty_per_unit

    eq_rows: list[np.ndarray] = []
    eq_rhs: list[float] = []

    # Inventory conservation.
    for cashpoint in cashpoints:
        for day_pos, day in enumerate(days):
            row = np.zeros(n_vars)

            for route in routes:
                if cashpoint in catalog.loc[route, "stops"]:
                    key = (day, route, cashpoint)
                    row[q0 + delivery_index[key]] = 1.0

            row[i0 + inv_index[(cashpoint, day)]] = -1.0

            if day_pos == 0:
                rhs = (
                    float(p.forecast_net_withdrawal.loc[cashpoint, day])
                    - float(p.cashpoints.loc[cashpoint, "initial_cash"])
                )
            else:
                previous_day = days[day_pos - 1]
                row[i0 + inv_index[(cashpoint, previous_day)]] = 1.0
                rhs = float(p.forecast_net_withdrawal.loc[cashpoint, day])

            eq_rows.append(row)
            eq_rhs.append(rhs)

    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []

    # Each cashpoint can be visited by at most one selected route per day.
    for day in days:
        for cashpoint in cashpoints:
            row = np.zeros(n_vars)
            for route in routes:
                if cashpoint in catalog.loc[route, "stops"]:
                    row[z0 + route_index[(day, route)]] = 1.0
            ub_rows.append(row)
            ub_rhs.append(1.0)

    # Route-specific delivery linking and vehicle capacity.
    for day, route in route_day_pairs:
        z_idx = z0 + route_index[(day, route)]

        route_capacity_row = np.zeros(n_vars)
        for cashpoint in catalog.loc[route, "stops"]:
            q_idx = q0 + delivery_index[(day, route, cashpoint)]
            route_capacity_row[q_idx] = 1.0

            link_row = np.zeros(n_vars)
            link_row[q_idx] = 1.0
            link_row[z_idx] = -float(
                p.cashpoints.loc[cashpoint, "maximum_delivery"]
            )
            ub_rows.append(link_row)
            ub_rhs.append(0.0)

        route_capacity_row[z_idx] = -p.vehicle_capacity
        ub_rows.append(route_capacity_row)
        ub_rhs.append(0.0)

    # Safety stock, vault capacity, fleet count, cluster visits, and
    # pre-withdrawal cashpoint capacity.
    for cashpoint in cashpoints:
        for day_pos, day in enumerate(days):
            inv_idx = i0 + inv_index[(cashpoint, day)]
            short_idx = s0 + inv_index[(cashpoint, day)]

            row = np.zeros(n_vars)
            row[inv_idx] = -1.0
            row[short_idx] = -1.0
            ub_rows.append(row)
            ub_rhs.append(-float(p.cashpoints.loc[cashpoint, "safety_stock"]))

            capacity_row = np.zeros(n_vars)
            for route in routes:
                if cashpoint in catalog.loc[route, "stops"]:
                    capacity_row[
                        q0 + delivery_index[(day, route, cashpoint)]
                    ] = 1.0

            if day_pos == 0:
                rhs = (
                    float(p.cashpoints.loc[cashpoint, "capacity"])
                    - float(p.cashpoints.loc[cashpoint, "initial_cash"])
                )
            else:
                previous_day = days[day_pos - 1]
                capacity_row[
                    i0 + inv_index[(cashpoint, previous_day)]
                ] = 1.0
                rhs = float(p.cashpoints.loc[cashpoint, "capacity"])

            ub_rows.append(capacity_row)
            ub_rhs.append(rhs)

    for day in days:
        # Vault dispatch.
        row = np.zeros(n_vars)
        for route in routes:
            for cashpoint in catalog.loc[route, "stops"]:
                row[
                    q0 + delivery_index[(day, route, cashpoint)]
                ] = 1.0
        ub_rows.append(row)
        ub_rhs.append(float(p.vault_dispatch_limit.loc[day]))

        # Maximum number of routes/vehicles.
        row = np.zeros(n_vars)
        for route in routes:
            row[z0 + route_index[(day, route)]] = 1.0
        ub_rows.append(row)
        ub_rhs.append(float(p.maximum_vehicles_per_day))

        # Cluster visit limits.
        for cluster, limit in p.cluster_visit_limit.items():
            members = set(
                p.cashpoints.index[p.cashpoints["cluster"] == cluster]
            )
            row = np.zeros(n_vars)
            for route in routes:
                count = len(members.intersection(catalog.loc[route, "stops"]))
                if count:
                    row[z0 + route_index[(day, route)]] = float(count)
            ub_rows.append(row)
            ub_rhs.append(float(limit))

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

    # Route binaries.
    upper[z0:q0] = 1.0

    # Inventory capacity.
    for cashpoint, day in inv_keys:
        upper[i0 + inv_index[(cashpoint, day)]] = float(
            p.cashpoints.loc[cashpoint, "capacity"]
        )

    integrality = np.zeros(n_vars, dtype=int)
    integrality[z0:q0] = 1

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(f"joint cash IRP failed: {result.message}")

    deliveries = pd.DataFrame(0.0, index=cashpoints, columns=days)
    visits = pd.DataFrame(0, index=cashpoints, columns=days)
    inventory = pd.DataFrame(0.0, index=cashpoints, columns=days)
    shortage = pd.DataFrame(0.0, index=cashpoints, columns=days)

    for cashpoint, day in inv_keys:
        inventory.loc[cashpoint, day] = result.x[
            i0 + inv_index[(cashpoint, day)]
        ]
        shortage.loc[cashpoint, day] = result.x[
            s0 + inv_index[(cashpoint, day)]
        ]

    selected_route_rows = []
    for day, route in route_day_pairs:
        if result.x[z0 + route_index[(day, route)]] <= 0.5:
            continue

        load = 0.0
        for cashpoint in catalog.loc[route, "stops"]:
            amount = result.x[
                q0 + delivery_index[(day, route, cashpoint)]
            ]
            deliveries.loc[cashpoint, day] += amount
            visits.loc[cashpoint, day] = 1
            load += amount

        selected_route_rows.append(
            {
                "day": day,
                "route_id": route,
                "sequence": catalog.loc[route, "sequence"],
                "load": float(load),
                "distance": float(catalog.loc[route, "distance"]),
                "stop_count": int(catalog.loc[route, "stop_count"]),
            }
        )

    selected_routes = pd.DataFrame(selected_route_rows)

    handling_cost = float(
        deliveries.to_numpy().sum() * p.handling_cost_per_unit
    )
    holding_cost = float(
        inventory.to_numpy().sum() * p.holding_cost_per_unit_day
    )
    shortage_cost = float(
        shortage.to_numpy().sum() * p.shortage_penalty_per_unit
    )

    if selected_routes.empty:
        vehicle_cost = 0.0
        distance_cost = 0.0
        visit_cost = 0.0
    else:
        vehicle_cost = float(
            len(selected_routes) * p.vehicle_fixed_cost
        )
        distance_cost = float(
            selected_routes["distance"].sum() * p.distance_cost_per_unit
        )
        visit_cost = float(
            sum(
                p.cashpoints.loc[cashpoint, "visit_cost"]
                for day in days
                for cashpoint in cashpoints
                if visits.loc[cashpoint, day] > 0.5
            )
        )

    cost_breakdown = pd.Series(
        {
            "handling_cost": handling_cost,
            "holding_cost": holding_cost,
            "visit_cost": visit_cost,
            "vehicle_cost": vehicle_cost,
            "distance_cost": distance_cost,
            "shortage_cost": shortage_cost,
        }
    )

    return JointIRPResult(
        deliveries=deliveries,
        visits=visits,
        end_inventory=inventory,
        shortage=shortage,
        selected_routes=selected_routes,
        cost_breakdown=cost_breakdown,
        total_cost=float(result.fun),
    )


def compare_staged_and_joint(
    problem: CashSupplyChainProblem | None = None,
    staged: CashSupplyChainResult | None = None,
    joint: JointIRPResult | None = None,
    staged_routes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare the staged replenishment+routing baseline to the joint IRP.

    Precomputed solutions can be supplied to avoid re-solving expensive models.
    """
    p = problem or default_problem()

    staged = staged or solve_staged(p)
    staged_routes = (
        staged_routes
        if staged_routes is not None
        else route_plan(staged, p)
    )
    staged_distance = (
        float(staged_routes["distance"].sum())
        if not staged_routes.empty
        else 0.0
    )
    staged_full_cost = (
        staged.total_cost + p.distance_cost_per_unit * staged_distance
    )

    joint = joint or solve_joint_irp(p)
    joint_distance = (
        float(joint.selected_routes["distance"].sum())
        if not joint.selected_routes.empty
        else 0.0
    )

    comparison = pd.DataFrame(
        [
            {
                "method": "staged",
                "integrated_total_cost": staged_full_cost,
                "routing_distance": staged_distance,
                "planned_shortage": float(staged.shortage.to_numpy().sum()),
                "total_delivered": float(staged.deliveries.to_numpy().sum()),
                "routes": len(staged_routes),
            },
            {
                "method": "joint_irp",
                "integrated_total_cost": joint.total_cost,
                "routing_distance": joint_distance,
                "planned_shortage": float(joint.shortage.to_numpy().sum()),
                "total_delivered": float(joint.deliveries.to_numpy().sum()),
                "routes": len(joint.selected_routes),
            },
        ]
    ).set_index("method")

    comparison["cost_improvement_vs_staged"] = (
        comparison.loc["staged", "integrated_total_cost"]
        - comparison["integrated_total_cost"]
    )
    return comparison
