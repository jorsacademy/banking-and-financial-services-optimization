"""Collateral allocation under haircuts and eligibility rules."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class CollateralProblem:
    collateral: pd.DataFrame
    obligations: pd.Series
    eligibility: pd.DataFrame


@dataclass(frozen=True)
class CollateralResult:
    allocation: pd.DataFrame
    effective_coverage: pd.Series
    total_opportunity_cost: float

    def to_dict(self) -> dict:
        nonzero = self.allocation.stack()
        nonzero = nonzero[nonzero > 1e-8]
        return {
            "allocation": {
                f"{i}->{j}": round(float(v), 6)
                for (i, j), v in nonzero.items()
            },
            "effective_coverage": self.effective_coverage.round(6).to_dict(),
            "total_opportunity_cost": round(self.total_opportunity_cost, 6),
        }


def default_problem() -> CollateralProblem:
    collateral = pd.DataFrame(
        {
            "inventory": [180.0, 220.0, 160.0, 120.0],
            "haircut": [0.00, 0.03, 0.08, 0.18],
            "opportunity_cost": [0.008, 0.012, 0.018, 0.030],
        },
        index=["cash", "gov_bonds", "covered_bonds", "corporate_bonds"],
    )

    obligations = pd.Series(
        [160.0, 190.0, 140.0],
        index=["clearing", "derivatives", "secured_funding"],
        name="required_effective_collateral",
    )

    eligibility = pd.DataFrame(
        [
            [1, 1, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 1],
        ],
        index=collateral.index,
        columns=obligations.index,
        dtype=int,
    )

    return CollateralProblem(
        collateral=collateral,
        obligations=obligations,
        eligibility=eligibility,
    )


def solve(problem: CollateralProblem | None = None) -> CollateralResult:
    p = problem or default_problem()
    assets = list(p.collateral.index)
    obligations = list(p.obligations.index)

    pairs = [
        (a, o)
        for a in assets
        for o in obligations
        if int(p.eligibility.loc[a, o]) == 1
    ]
    n = len(pairs)

    c = np.array(
        [p.collateral.loc[a, "opportunity_cost"] for a, _ in pairs],
        dtype=float,
    )

    a_ub = []
    b_ub = []

    # Inventory constraints.
    for asset in assets:
        row = np.array([1.0 if a == asset else 0.0 for a, _ in pairs])
        a_ub.append(row)
        b_ub.append(float(p.collateral.loc[asset, "inventory"]))

    # Effective collateral coverage >= requirement.
    for obligation in obligations:
        row = np.array(
            [
                -(1.0 - p.collateral.loc[a, "haircut"])
                if o == obligation
                else 0.0
                for a, o in pairs
            ],
            dtype=float,
        )
        a_ub.append(row)
        b_ub.append(-float(p.obligations.loc[obligation]))

    result = linprog(
        c,
        A_ub=np.vstack(a_ub),
        b_ub=np.asarray(b_ub),
        bounds=[(0.0, None)] * n,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"collateral allocation failed: {result.message}")

    allocation = pd.DataFrame(0.0, index=assets, columns=obligations)
    for value, (asset, obligation) in zip(result.x, pairs):
        allocation.loc[asset, obligation] = value

    effective_coverage = pd.Series(index=obligations, dtype=float)
    for obligation in obligations:
        effective_coverage.loc[obligation] = float(
            sum(
                allocation.loc[asset, obligation]
                * (1.0 - p.collateral.loc[asset, "haircut"])
                for asset in assets
            )
        )

    return CollateralResult(
        allocation=allocation,
        effective_coverage=effective_coverage,
        total_opportunity_cost=float(result.fun),
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
