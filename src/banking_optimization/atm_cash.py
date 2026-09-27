"""Multi-period ATM cash replenishment with visit capacity."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class ATMCashProblem:
    demand: pd.DataFrame
    initial_cash: pd.Series
    cash_capacity: pd.Series
    safety_stock: pd.Series
    maximum_delivery: float = 160.0
    maximum_visits_per_day: int = 2
    fixed_visit_cost: float = 2.0
    handling_cost_per_unit: float = 0.010
    holding_cost_per_unit_day: float = 0.005


@dataclass(frozen=True)
class ATMCashResult:
    deliveries: pd.DataFrame
    visits: pd.DataFrame
    end_inventory: pd.DataFrame
    total_cost: float

    def to_dict(self) -> dict:
        return {
            "deliveries": self.deliveries.round(6).to_dict(),
            "visits": self.visits.astype(int).to_dict(),
            "end_inventory": self.end_inventory.round(6).to_dict(),
            "total_cost": round(self.total_cost, 6),
        }


def default_problem() -> ATMCashProblem:
    atms = ["ATM_A", "ATM_B", "ATM_C", "ATM_D"]
    days = [1, 2, 3, 4, 5]
    demand = pd.DataFrame(
        [
            [40.0, 45.0, 50.0, 35.0, 55.0],
            [30.0, 35.0, 40.0, 45.0, 40.0],
            [55.0, 50.0, 45.0, 60.0, 50.0],
            [35.0, 30.0, 40.0, 35.0, 45.0],
        ],
        index=atms,
        columns=days,
    )
    return ATMCashProblem(
        demand=demand,
        initial_cash=pd.Series([110.0, 90.0, 130.0, 100.0], index=atms),
        cash_capacity=pd.Series([220.0, 200.0, 230.0, 200.0], index=atms),
        safety_stock=pd.Series([15.0, 15.0, 20.0, 15.0], index=atms),
    )


def solve(problem: ATMCashProblem | None = None) -> ATMCashResult:
    p = problem or default_problem()
    atms = list(p.demand.index)
    days = list(p.demand.columns)
    pairs = [(a, d) for a in atms for d in days]
    n = len(pairs)

    # Variable blocks: delivery q | visit y | end inventory I
    q0 = 0
    y0 = n
    i0 = 2 * n
    n_vars = 3 * n

    index = {pair: k for k, pair in enumerate(pairs)}

    c = np.zeros(n_vars)
    c[q0:y0] = p.handling_cost_per_unit
    c[y0:i0] = p.fixed_visit_cost
    c[i0:] = p.holding_cost_per_unit_day

    eq_rows = []
    eq_rhs = []

    # Inventory balance for every ATM/day.
    for atm in atms:
        for day_pos, day in enumerate(days):
            k = index[(atm, day)]
            row = np.zeros(n_vars)
            row[q0 + k] = 1.0
            row[i0 + k] = -1.0
            if day_pos > 0:
                prev = index[(atm, days[day_pos - 1])]
                row[i0 + prev] = 1.0
                rhs = float(p.demand.loc[atm, day])
            else:
                rhs = float(p.demand.loc[atm, day] - p.initial_cash.loc[atm])
            eq_rows.append(row)
            eq_rhs.append(rhs)

    ub_rows = []
    ub_rhs = []

    # A delivery can occur only when a visit is opened.
    for k in range(n):
        row = np.zeros(n_vars)
        row[q0 + k] = 1.0
        row[y0 + k] = -p.maximum_delivery
        ub_rows.append(row)
        ub_rhs.append(0.0)

    # Daily route/crew capacity.
    for day in days:
        row = np.zeros(n_vars)
        for atm in atms:
            row[y0 + index[(atm, day)]] = 1.0
        ub_rows.append(row)
        ub_rhs.append(float(p.maximum_visits_per_day))

    # Pre-demand machine cash capacity:
    # previous end inventory + delivery <= capacity.
    for atm in atms:
        for day_pos, day in enumerate(days):
            k = index[(atm, day)]
            row = np.zeros(n_vars)
            row[q0 + k] = 1.0
            if day_pos == 0:
                rhs = float(
                    p.cash_capacity.loc[atm] - p.initial_cash.loc[atm]
                )
            else:
                prev = index[(atm, days[day_pos - 1])]
                row[i0 + prev] = 1.0
                rhs = float(p.cash_capacity.loc[atm])
            ub_rows.append(row)
            ub_rhs.append(rhs)

    constraints = [
        LinearConstraint(
            np.vstack(eq_rows),
            lb=np.asarray(eq_rhs),
            ub=np.asarray(eq_rhs),
        ),
        LinearConstraint(
            np.vstack(ub_rows),
            lb=np.full(len(ub_rows), -np.inf),
            ub=np.asarray(ub_rhs),
        ),
    ]

    lower = np.zeros(n_vars)
    upper = np.full(n_vars, np.inf)
    upper[q0:y0] = p.maximum_delivery
    upper[y0:i0] = 1.0
    for (atm, day), k in index.items():
        lower[i0 + k] = float(p.safety_stock.loc[atm])
        upper[i0 + k] = float(p.cash_capacity.loc[atm])

    integrality = np.zeros(n_vars, dtype=int)
    integrality[y0:i0] = 1

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(f"ATM cash optimization failed: {result.message}")

    deliveries = pd.DataFrame(0.0, index=atms, columns=days)
    visits = pd.DataFrame(0, index=atms, columns=days)
    inventory = pd.DataFrame(0.0, index=atms, columns=days)

    for (atm, day), k in index.items():
        deliveries.loc[atm, day] = result.x[q0 + k]
        visits.loc[atm, day] = int(result.x[y0 + k] > 0.5)
        inventory.loc[atm, day] = result.x[i0 + k]

    return ATMCashResult(
        deliveries=deliveries,
        visits=visits,
        end_inventory=inventory,
        total_cost=float(result.fun),
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
