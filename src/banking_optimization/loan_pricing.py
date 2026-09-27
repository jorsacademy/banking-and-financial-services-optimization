"""Discrete loan pricing and credit-limit optimization."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class LoanPricingProblem:
    offers: pd.DataFrame
    expected_exposure_budget: float = 260.0
    expected_rwa_budget: float = 165.0
    expected_loss_budget: float = 5.5
    funding_rate: float = 0.035
    capital_charge_rate: float = 0.015
    origination_cost: float = 0.60


@dataclass(frozen=True)
class LoanPricingResult:
    selected_offers: pd.DataFrame
    expected_exposure: float
    expected_rwa: float
    expected_loss: float
    expected_profit: float

    def to_dict(self) -> dict:
        cols = ["customer_id", "rate", "limit", "accept_prob", "pd", "expected_profit"]
        return {
            "selected_offers": self.selected_offers[cols]
            .round(6)
            .to_dict(orient="records"),
            "expected_exposure": round(self.expected_exposure, 6),
            "expected_rwa": round(self.expected_rwa, 6),
            "expected_loss": round(self.expected_loss, 6),
            "expected_profit": round(self.expected_profit, 6),
        }


def default_problem() -> LoanPricingProblem:
    customers = pd.DataFrame(
        [
            ("C01", 0.018, 0.40, 0.55, 45.0),
            ("C02", 0.028, 0.45, 0.70, 55.0),
            ("C03", 0.012, 0.35, 0.50, 60.0),
            ("C04", 0.040, 0.50, 0.85, 50.0),
            ("C05", 0.022, 0.40, 0.65, 70.0),
            ("C06", 0.032, 0.45, 0.75, 65.0),
            ("C07", 0.015, 0.35, 0.55, 50.0),
            ("C08", 0.045, 0.55, 0.95, 45.0),
        ],
        columns=["customer_id", "base_pd", "lgd", "risk_weight", "base_limit"],
    )

    rows = []
    rate_levels = [0.075, 0.090, 0.105]
    limit_multipliers = [0.75, 1.00, 1.25]

    for row in customers.itertuples(index=False):
        for k, (rate, lm) in enumerate(zip(rate_levels, limit_multipliers), start=1):
            limit = row.base_limit * lm
            accept_prob = np.clip(
                0.88 - 4.0 * (rate - 0.075) + 0.04 * (lm - 0.75),
                0.35,
                0.95,
            )
            pd_est = np.clip(
                row.base_pd + 0.012 * (lm - 0.75) + 0.08 * (rate - 0.075),
                0.005,
                0.15,
            )
            rows.append(
                (
                    f"{row.customer_id}_O{k}",
                    row.customer_id,
                    rate,
                    limit,
                    float(accept_prob),
                    float(pd_est),
                    row.lgd,
                    row.risk_weight,
                    0.70,
                )
            )

    offers = pd.DataFrame(
        rows,
        columns=[
            "offer_id",
            "customer_id",
            "rate",
            "limit",
            "accept_prob",
            "pd",
            "lgd",
            "risk_weight",
            "utilization",
        ],
    ).set_index("offer_id")

    return LoanPricingProblem(offers=offers)


def economics(problem: LoanPricingProblem) -> pd.DataFrame:
    df = problem.offers.copy()
    df["drawn_exposure"] = df["limit"] * df["utilization"]
    df["expected_exposure"] = df["accept_prob"] * df["drawn_exposure"]
    df["expected_loss"] = (
        df["expected_exposure"] * df["pd"] * df["lgd"]
    )
    df["expected_rwa"] = (
        df["expected_exposure"] * df["risk_weight"]
    )
    df["expected_margin"] = (
        df["expected_exposure"] * (df["rate"] - problem.funding_rate)
    )
    df["capital_charge"] = (
        problem.capital_charge_rate * df["expected_rwa"]
    )
    df["expected_profit"] = (
        df["expected_margin"]
        - df["expected_loss"]
        - df["capital_charge"]
        - problem.origination_cost * df["accept_prob"]
    )
    return df


def solve(problem: LoanPricingProblem | None = None) -> LoanPricingResult:
    p = problem or default_problem()
    df = economics(p)
    n = len(df)

    rows = [
        df["expected_exposure"].to_numpy(float),
        df["expected_rwa"].to_numpy(float),
        df["expected_loss"].to_numpy(float),
    ]
    lower = [-np.inf, -np.inf, -np.inf]
    upper = [
        p.expected_exposure_budget,
        p.expected_rwa_budget,
        p.expected_loss_budget,
    ]

    customer_ids = df["customer_id"].to_numpy()
    for customer in df["customer_id"].unique():
        rows.append((customer_ids == customer).astype(float))
        lower.append(-np.inf)
        upper.append(1.0)

    constraints = LinearConstraint(
        np.vstack(rows),
        lb=np.asarray(lower, dtype=float),
        ub=np.asarray(upper, dtype=float),
    )

    result = milp(
        c=-df["expected_profit"].to_numpy(float),
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(f"loan-pricing optimization failed: {result.message}")

    selected = df.loc[result.x > 0.5].copy()

    return LoanPricingResult(
        selected_offers=selected,
        expected_exposure=float(selected["expected_exposure"].sum()),
        expected_rwa=float(selected["expected_rwa"].sum()),
        expected_loss=float(selected["expected_loss"].sum()),
        expected_profit=float(selected["expected_profit"].sum()),
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
