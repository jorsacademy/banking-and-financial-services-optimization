"""Integrated cash replenishment and CIT resource planning.

The model jointly chooses cash deliveries, visit timing, end-of-day inventory,
service shortfall, and daily vehicle count for a synthetic ATM/branch network.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class CashSupplyChainProblem:
    cashpoints: pd.DataFrame
    forecast_net_withdrawal: pd.DataFrame
    vault_dispatch_limit: pd.Series
    cluster_visit_limit: dict[str, int]
    vehicle_capacity: float = 300.0
    stops_per_vehicle: int = 3
    maximum_vehicles_per_day: int = 3
    vehicle_fixed_cost: float = 5.0
    distance_cost_per_unit: float = 0.35
    handling_cost_per_unit: float = 0.010
    holding_cost_per_unit_day: float = 0.007
    shortage_penalty_per_unit: float = 4.0
    cashout_penalty_per_unit: float = 20.0


@dataclass(frozen=True)
class CashSupplyChainResult:
    deliveries: pd.DataFrame
    visits: pd.DataFrame
    end_inventory: pd.DataFrame
    shortage: pd.DataFrame
    vehicles: pd.Series
    cost_breakdown: pd.Series
    total_cost: float

    def to_dict(self) -> dict:
        return {
            "deliveries": self.deliveries.round(6).to_dict(),
            "visits": self.visits.astype(int).to_dict(),
            "end_inventory": self.end_inventory.round(6).to_dict(),
            "shortage": self.shortage.round(6).to_dict(),
            "vehicles": self.vehicles.round(6).to_dict(),
            "cost_breakdown": self.cost_breakdown.round(6).to_dict(),
            "total_cost": round(self.total_cost, 6),
        }


def default_problem() -> CashSupplyChainProblem:
    cashpoints = pd.DataFrame(
        [
            ("ATM_A", "atm", "north", 95.0, 210.0, 20.0, 130.0, 1.8, 2.0, 8.0),
            ("ATM_B", "atm", "north", 85.0, 190.0, 18.0, 120.0, 1.6, 5.0, 9.0),
            ("BRANCH_C", "branch", "central", 150.0, 320.0, 35.0, 180.0, 2.8, 6.0, 4.0),
            ("ATM_D", "atm", "central", 100.0, 220.0, 22.0, 135.0, 1.9, 9.0, 5.0),
            ("BRANCH_E", "branch", "south", 170.0, 340.0, 40.0, 190.0, 3.0, 7.0, 1.0),
            ("ATM_F", "atm", "south", 90.0, 200.0, 20.0, 125.0, 1.7, 3.0, 2.0),
        ],
        columns=[
            "cashpoint",
            "type",
            "cluster",
            "initial_cash",
            "capacity",
            "safety_stock",
            "maximum_delivery",
            "visit_cost",
            "x_coord",
            "y_coord",
        ],
    ).set_index("cashpoint")

    days = [1, 2, 3, 4, 5, 6, 7]
    forecast = pd.DataFrame(
        [
            [42.0, 48.0, 55.0, 38.0, 60.0, 64.0, 46.0],
            [34.0, 40.0, 43.0, 36.0, 48.0, 52.0, 39.0],
            [62.0, 70.0, 75.0, 58.0, 84.0, 90.0, 68.0],
            [45.0, 49.0, 53.0, 41.0, 61.0, 65.0, 47.0],
            [72.0, 78.0, 82.0, 66.0, 94.0, 101.0, 76.0],
            [37.0, 42.0, 46.0, 35.0, 51.0, 56.0, 41.0],
        ],
        index=cashpoints.index,
        columns=days,
    )

    vault_dispatch_limit = pd.Series(
        [360.0, 320.0, 360.0, 300.0, 420.0, 430.0, 360.0],
        index=days,
        name="vault_dispatch_limit",
    )

    return CashSupplyChainProblem(
        cashpoints=cashpoints,
        forecast_net_withdrawal=forecast,
        vault_dispatch_limit=vault_dispatch_limit,
        cluster_visit_limit={"north": 2, "central": 2, "south": 2},
    )


def solve(
    problem: CashSupplyChainProblem | None = None,
) -> CashSupplyChainResult:
    p = problem or default_problem()

    cashpoints = list(p.cashpoints.index)
    days = list(p.forecast_net_withdrawal.columns)
    pairs = [(i, t) for i in cashpoints for t in days]
    n = len(pairs)
    pair_index = {pair: k for k, pair in enumerate(pairs)}

    # Variable blocks:
    # q: delivery, y: visit binary, I: end inventory, s: service shortfall,
    # v: integer number of CIT vehicles by day.
    q0 = 0
    y0 = n
    i0 = 2 * n
    s0 = 3 * n
    v0 = 4 * n
    n_vars = v0 + len(days)

    day_index = {t: k for k, t in enumerate(days)}

    c = np.zeros(n_vars)
    for cashpoint, day in pairs:
        k = pair_index[(cashpoint, day)]
        c[q0 + k] = p.handling_cost_per_unit
        c[y0 + k] = float(p.cashpoints.loc[cashpoint, "visit_cost"])
        c[i0 + k] = p.holding_cost_per_unit_day
        c[s0 + k] = p.shortage_penalty_per_unit
    for day in days:
        c[v0 + day_index[day]] = p.vehicle_fixed_cost

    eq_rows: list[np.ndarray] = []
    eq_rhs: list[float] = []

    # Inventory conservation:
    # I_it = I_i,t-1 + q_it - demand_it.
    for cashpoint in cashpoints:
        for day_pos, day in enumerate(days):
            k = pair_index[(cashpoint, day)]
            row = np.zeros(n_vars)
            row[q0 + k] = 1.0
            row[i0 + k] = -1.0
            if day_pos == 0:
                rhs = (
                    float(p.forecast_net_withdrawal.loc[cashpoint, day])
                    - float(p.cashpoints.loc[cashpoint, "initial_cash"])
                )
            else:
                prev_day = days[day_pos - 1]
                prev = pair_index[(cashpoint, prev_day)]
                row[i0 + prev] = 1.0
                rhs = float(p.forecast_net_withdrawal.loc[cashpoint, day])
            eq_rows.append(row)
            eq_rhs.append(rhs)

    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []

    # Delivery can happen only if the cashpoint is visited.
    for cashpoint, day in pairs:
        k = pair_index[(cashpoint, day)]
        row = np.zeros(n_vars)
        row[q0 + k] = 1.0
        row[y0 + k] = -float(
            p.cashpoints.loc[cashpoint, "maximum_delivery"]
        )
        ub_rows.append(row)
        ub_rhs.append(0.0)

    # Safety-stock service constraint:
    # I_it + s_it >= safety_i.
    for cashpoint, day in pairs:
        k = pair_index[(cashpoint, day)]
        row = np.zeros(n_vars)
        row[i0 + k] = -1.0
        row[s0 + k] = -1.0
        ub_rows.append(row)
        ub_rhs.append(-float(p.cashpoints.loc[cashpoint, "safety_stock"]))

    for day in days:
        # Central cash preparation / dispatch capacity.
        row = np.zeros(n_vars)
        for cashpoint in cashpoints:
            row[q0 + pair_index[(cashpoint, day)]] = 1.0
        ub_rows.append(row)
        ub_rhs.append(float(p.vault_dispatch_limit.loc[day]))

        # Fleet cash-carrying capacity.
        row = np.zeros(n_vars)
        for cashpoint in cashpoints:
            row[q0 + pair_index[(cashpoint, day)]] = 1.0
        row[v0 + day_index[day]] = -p.vehicle_capacity
        ub_rows.append(row)
        ub_rhs.append(0.0)

        # Maximum stops that deployed vehicles can serve.
        row = np.zeros(n_vars)
        for cashpoint in cashpoints:
            row[y0 + pair_index[(cashpoint, day)]] = 1.0
        row[v0 + day_index[day]] = -float(p.stops_per_vehicle)
        ub_rows.append(row)
        ub_rhs.append(0.0)

        # Cluster-level operational visit limits.
        for cluster, limit in p.cluster_visit_limit.items():
            row = np.zeros(n_vars)
            members = p.cashpoints.index[
                p.cashpoints["cluster"] == cluster
            ]
            for cashpoint in members:
                row[y0 + pair_index[(cashpoint, day)]] = 1.0
            ub_rows.append(row)
            ub_rhs.append(float(limit))

    # Capacity immediately after replenishment and before forecast withdrawal.
    for cashpoint in cashpoints:
        for day_pos, day in enumerate(days):
            k = pair_index[(cashpoint, day)]
            row = np.zeros(n_vars)
            row[q0 + k] = 1.0
            if day_pos == 0:
                rhs = (
                    float(p.cashpoints.loc[cashpoint, "capacity"])
                    - float(p.cashpoints.loc[cashpoint, "initial_cash"])
                )
            else:
                prev = pair_index[(cashpoint, days[day_pos - 1])]
                row[i0 + prev] = 1.0
                rhs = float(p.cashpoints.loc[cashpoint, "capacity"])
            ub_rows.append(row)
            ub_rhs.append(rhs)

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

    # Visit binaries.
    upper[y0:i0] = 1.0

    # Cashpoint end inventory cannot exceed capacity.
    for cashpoint, day in pairs:
        k = pair_index[(cashpoint, day)]
        upper[i0 + k] = float(p.cashpoints.loc[cashpoint, "capacity"])

    # Fleet bounds.
    for day in days:
        upper[v0 + day_index[day]] = p.maximum_vehicles_per_day

    integrality = np.zeros(n_vars, dtype=int)
    integrality[y0:i0] = 1
    integrality[v0:] = 1

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(
            f"cash supply-chain optimization failed: {result.message}"
        )

    deliveries = pd.DataFrame(0.0, index=cashpoints, columns=days)
    visits = pd.DataFrame(0, index=cashpoints, columns=days)
    inventory = pd.DataFrame(0.0, index=cashpoints, columns=days)
    shortage = pd.DataFrame(0.0, index=cashpoints, columns=days)

    for cashpoint, day in pairs:
        k = pair_index[(cashpoint, day)]
        deliveries.loc[cashpoint, day] = result.x[q0 + k]
        visits.loc[cashpoint, day] = int(result.x[y0 + k] > 0.5)
        inventory.loc[cashpoint, day] = result.x[i0 + k]
        shortage.loc[cashpoint, day] = result.x[s0 + k]

    vehicles = pd.Series(
        {
            day: result.x[v0 + day_index[day]]
            for day in days
        },
        name="vehicles",
    )

    handling_cost = float(deliveries.to_numpy().sum() * p.handling_cost_per_unit)
    holding_cost = float(inventory.to_numpy().sum() * p.holding_cost_per_unit_day)
    visit_cost = float(
        sum(
            visits.loc[cashpoint, day]
            * p.cashpoints.loc[cashpoint, "visit_cost"]
            for cashpoint, day in pairs
        )
    )
    vehicle_cost = float(vehicles.sum() * p.vehicle_fixed_cost)
    shortage_cost = float(
        shortage.to_numpy().sum() * p.shortage_penalty_per_unit
    )

    cost_breakdown = pd.Series(
        {
            "handling_cost": handling_cost,
            "holding_cost": holding_cost,
            "visit_cost": visit_cost,
            "vehicle_cost": vehicle_cost,
            "shortage_cost": shortage_cost,
        }
    )

    return CashSupplyChainResult(
        deliveries=deliveries,
        visits=visits,
        end_inventory=inventory,
        shortage=shortage,
        vehicles=vehicles,
        cost_breakdown=cost_breakdown,
        total_cost=float(result.fun),
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
