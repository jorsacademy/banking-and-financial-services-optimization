"""Credit and capital allocation with a binary mixed-integer model."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class CreditCapitalProblem:
    opportunities: pd.DataFrame
    lending_budget: float = 650.0
    rwa_budget: float = 360.0
    expected_loss_budget: float = 16.0
    sector_exposure_cap: float = 260.0
    funding_rate: float = 0.035
    capital_charge_rate: float = 0.015


@dataclass(frozen=True)
class CreditCapitalResult:
    selected: pd.DataFrame
    total_exposure: float
    total_rwa: float
    total_expected_loss: float
    total_expected_profit: float
    sector_exposure: pd.Series

    def to_dict(self) -> dict:
        return {
            "selected_ids": list(self.selected.index),
            "total_exposure": round(self.total_exposure, 6),
            "total_rwa": round(self.total_rwa, 6),
            "total_expected_loss": round(self.total_expected_loss, 6),
            "total_expected_profit": round(self.total_expected_profit, 6),
            "sector_exposure": self.sector_exposure.round(6).to_dict(),
        }


def default_problem() -> CreditCapitalProblem:
    rows = [
        ("L001", "manufacturing", 120, 0.090, 0.020, 0.45, 0.75),
        ("L002", "services",       80, 0.082, 0.015, 0.40, 0.55),
        ("L003", "retail",         95, 0.105, 0.045, 0.50, 0.85),
        ("L004", "technology",    110, 0.088, 0.018, 0.42, 0.60),
        ("L005", "manufacturing", 70, 0.110, 0.060, 0.55, 0.95),
        ("L006", "services",      130, 0.075, 0.012, 0.35, 0.50),
        ("L007", "retail",         60, 0.125, 0.075, 0.60, 1.00),
        ("L008", "technology",     90, 0.095, 0.030, 0.45, 0.70),
        ("L009", "manufacturing", 85, 0.085, 0.025, 0.50, 0.80),
        ("L010", "services",       75, 0.115, 0.050, 0.45, 0.90),
        ("L011", "retail",        105, 0.080, 0.020, 0.40, 0.65),
        ("L012", "technology",     65, 0.120, 0.055, 0.50, 0.90),
        ("L013", "manufacturing",100, 0.078, 0.010, 0.35, 0.55),
        ("L014", "services",       55, 0.130, 0.070, 0.60, 1.00),
        ("L015", "retail",         90, 0.092, 0.028, 0.48, 0.75),
        ("L016", "technology",     75, 0.100, 0.022, 0.40, 0.65),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "application_id",
            "sector",
            "exposure",
            "rate",
            "pd",
            "lgd",
            "risk_weight",
        ],
    ).set_index("application_id")
    return CreditCapitalProblem(opportunities=df)


def economics(problem: CreditCapitalProblem) -> pd.DataFrame:
    df = problem.opportunities.copy()
    df["expected_loss"] = df["exposure"] * df["pd"] * df["lgd"]
    df["rwa"] = df["exposure"] * df["risk_weight"]
    df["net_interest_margin"] = df["exposure"] * (
        df["rate"] - problem.funding_rate
    )
    df["capital_charge"] = (
        problem.capital_charge_rate * df["rwa"]
    )
    df["expected_profit"] = (
        df["net_interest_margin"]
        - df["expected_loss"]
        - df["capital_charge"]
    )
    return df


def solve(problem: CreditCapitalProblem | None = None) -> CreditCapitalResult:
    p = problem or default_problem()
    df = economics(p)

    n = len(df)
    profit = df["expected_profit"].to_numpy(float)
    exposure = df["exposure"].to_numpy(float)
    rwa = df["rwa"].to_numpy(float)
    expected_loss = df["expected_loss"].to_numpy(float)

    rows = [exposure, rwa, expected_loss]
    upper = [p.lending_budget, p.rwa_budget, p.expected_loss_budget]

    for sector in sorted(df["sector"].unique()):
        rows.append(
            np.where(df["sector"].to_numpy() == sector, exposure, 0.0)
        )
        upper.append(p.sector_exposure_cap)

    constraints = LinearConstraint(
        np.vstack(rows),
        lb=np.full(len(rows), -np.inf),
        ub=np.asarray(upper, dtype=float),
    )

    result = milp(
        c=-profit,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(f"credit-capital optimization failed: {result.message}")

    chosen = result.x > 0.5
    selected = df.loc[chosen].copy()

    sector_exposure = (
        selected.groupby("sector")["exposure"].sum()
        if not selected.empty
        else pd.Series(dtype=float)
    )

    return CreditCapitalResult(
        selected=selected,
        total_exposure=float(selected["exposure"].sum()),
        total_rwa=float(selected["rwa"].sum()),
        total_expected_loss=float(selected["expected_loss"].sum()),
        total_expected_profit=float(selected["expected_profit"].sum()),
        sector_exposure=sector_exposure,
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
