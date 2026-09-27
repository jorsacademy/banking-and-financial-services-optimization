"""Collections and recovery treatment allocation."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class CollectionsProblem:
    accounts: pd.DataFrame
    actions: pd.DataFrame
    capacity: dict[str, int]
    treatment_budget: float = 18.0


@dataclass(frozen=True)
class CollectionsResult:
    assignments: pd.DataFrame
    total_cost: float
    expected_incremental_recovery: float

    def to_dict(self) -> dict:
        return {
            "assignments": self.assignments.round(6).to_dict(orient="records"),
            "total_cost": round(self.total_cost, 6),
            "expected_incremental_recovery": round(
                self.expected_incremental_recovery, 6
            ),
        }


def default_problem() -> CollectionsProblem:
    accounts = pd.DataFrame(
        [
            ("A01", 12.0, "early", 0.75),
            ("A02", 18.0, "early", 0.65),
            ("A03", 25.0, "mid", 0.60),
            ("A04", 32.0, "late", 0.50),
            ("A05", 16.0, "mid", 0.70),
            ("A06", 28.0, "late", 0.55),
            ("A07", 14.0, "early", 0.80),
            ("A08", 40.0, "default", 0.45),
            ("A09", 22.0, "mid", 0.62),
            ("A10", 35.0, "late", 0.48),
            ("A11", 19.0, "early", 0.68),
            ("A12", 45.0, "default", 0.40),
        ],
        columns=["account_id", "balance", "stage", "contactability"],
    ).set_index("account_id")

    actions = pd.DataFrame(
        {
            "cost": [0.10, 0.65, 1.80, 3.50],
            "early": [0.05, 0.12, 0.18, 0.04],
            "mid": [0.03, 0.14, 0.25, 0.10],
            "late": [0.01, 0.10, 0.28, 0.22],
            "default": [0.00, 0.04, 0.20, 0.30],
        },
        index=["sms", "agent_call", "restructure", "legal"],
    )

    return CollectionsProblem(
        accounts=accounts,
        actions=actions,
        capacity={"sms": 12, "agent_call": 6, "restructure": 4, "legal": 3},
    )


def economics(problem: CollectionsProblem) -> pd.DataFrame:
    rows = []
    for account_id, account in problem.accounts.iterrows():
        for action, action_data in problem.actions.iterrows():
            response = float(action_data[account["stage"]])
            expected_recovery = (
                account["balance"]
                * account["contactability"]
                * response
            )
            cost = float(action_data["cost"])
            rows.append(
                (
                    account_id,
                    action,
                    cost,
                    expected_recovery,
                    expected_recovery - cost,
                )
            )
    return pd.DataFrame(
        rows,
        columns=[
            "account_id",
            "action",
            "cost",
            "expected_incremental_recovery",
            "net_value",
        ],
    )


def solve(problem: CollectionsProblem | None = None) -> CollectionsResult:
    p = problem or default_problem()
    table = economics(p)
    n = len(table)

    rows = []
    upper = []

    account_values = table["account_id"].to_numpy()
    for account_id in p.accounts.index:
        rows.append((account_values == account_id).astype(float))
        upper.append(1.0)

    action_values = table["action"].to_numpy()
    for action, cap in p.capacity.items():
        rows.append((action_values == action).astype(float))
        upper.append(float(cap))

    rows.append(table["cost"].to_numpy(float))
    upper.append(p.treatment_budget)

    result = milp(
        c=-table["net_value"].to_numpy(float),
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(
            np.vstack(rows),
            lb=np.full(len(rows), -np.inf),
            ub=np.asarray(upper, dtype=float),
        ),
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(f"collections optimization failed: {result.message}")

    selected = table.loc[result.x > 0.5].copy().reset_index(drop=True)
    return CollectionsResult(
        assignments=selected,
        total_cost=float(selected["cost"].sum()),
        expected_incremental_recovery=float(
            selected["expected_incremental_recovery"].sum()
        ),
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
