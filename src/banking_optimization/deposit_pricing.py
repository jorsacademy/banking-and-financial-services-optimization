"""Deposit pricing optimization with synthetic retention curves."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class DepositPricingProblem:
    segments: pd.DataFrame
    rate_options: tuple[float, ...] = (0.020, 0.025, 0.030, 0.035)
    target_expected_retained_balance: float = 620.0


@dataclass(frozen=True)
class DepositPricingResult:
    selected_offers: pd.DataFrame
    expected_retained_balance: float
    expected_interest_cost: float
    weighted_average_rate: float

    def to_dict(self) -> dict:
        return {
            "selected_offers": self.selected_offers.round(6).to_dict(orient="records"),
            "expected_retained_balance": round(self.expected_retained_balance, 6),
            "expected_interest_cost": round(self.expected_interest_cost, 6),
            "weighted_average_rate": round(self.weighted_average_rate, 6),
        }


def default_problem() -> DepositPricingProblem:
    segments = pd.DataFrame(
        [
            ("mass_stable", 150.0, 0.72, 7.0),
            ("mass_sensitive", 120.0, 0.55, 14.0),
            ("affluent_stable", 180.0, 0.76, 6.0),
            ("affluent_sensitive", 100.0, 0.58, 13.0),
            ("sme_operating", 140.0, 0.68, 8.0),
            ("sme_rate_sensitive", 160.0, 0.52, 15.0),
        ],
        columns=[
            "segment",
            "balance",
            "base_retention",
            "rate_sensitivity",
        ],
    ).set_index("segment")
    return DepositPricingProblem(segments=segments)


def offer_table(problem: DepositPricingProblem) -> pd.DataFrame:
    rows = []
    for segment, row in problem.segments.iterrows():
        for rate in problem.rate_options:
            retention = np.clip(
                row["base_retention"]
                + row["rate_sensitivity"] * (rate - problem.rate_options[0]),
                0.0,
                0.98,
            )
            expected_balance = row["balance"] * retention
            expected_cost = expected_balance * rate
            rows.append(
                (
                    segment,
                    rate,
                    float(retention),
                    float(expected_balance),
                    float(expected_cost),
                )
            )

    return pd.DataFrame(
        rows,
        columns=[
            "segment",
            "offered_rate",
            "retention_probability",
            "expected_retained_balance",
            "expected_interest_cost",
        ],
    )


def solve(problem: DepositPricingProblem | None = None) -> DepositPricingResult:
    p = problem or default_problem()
    table = offer_table(p)
    n = len(table)

    eq_rows = []
    segment_values = table["segment"].to_numpy()
    for segment in p.segments.index:
        eq_rows.append((segment_values == segment).astype(float))

    retained = table["expected_retained_balance"].to_numpy(float)
    constraints = [
        LinearConstraint(
            np.vstack(eq_rows),
            lb=np.ones(len(eq_rows)),
            ub=np.ones(len(eq_rows)),
        ),
        LinearConstraint(
            retained.reshape(1, -1),
            lb=np.array([p.target_expected_retained_balance]),
            ub=np.array([np.inf]),
        ),
    ]

    result = milp(
        c=table["expected_interest_cost"].to_numpy(float),
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(f"deposit pricing optimization failed: {result.message}")

    selected = table.loc[result.x > 0.5].copy().reset_index(drop=True)
    total_balance = float(selected["expected_retained_balance"].sum())
    total_cost = float(selected["expected_interest_cost"].sum())
    weighted_rate = (
        total_cost / total_balance if total_balance > 0 else 0.0
    )

    return DepositPricingResult(
        selected_offers=selected,
        expected_retained_balance=total_balance,
        expected_interest_cost=total_cost,
        weighted_average_rate=weighted_rate,
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
