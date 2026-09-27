"""Fraud alert investigation triage under scarce analyst capacity."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class FraudTriageProblem:
    alerts: pd.DataFrame
    investigator_minutes: float = 300.0
    maximum_alerts: int = 8
    review_cost_per_minute: float = 0.015


@dataclass(frozen=True)
class FraudTriageResult:
    selected_alerts: pd.DataFrame
    minutes_used: float
    expected_prevented_loss: float
    net_value: float

    def to_dict(self) -> dict:
        return {
            "selected_alerts": list(self.selected_alerts.index),
            "minutes_used": round(self.minutes_used, 6),
            "expected_prevented_loss": round(self.expected_prevented_loss, 6),
            "net_value": round(self.net_value, 6),
        }


def default_problem() -> FraudTriageProblem:
    alerts = pd.DataFrame(
        [
            ("F01", "card", 8.0, 0.62, 0.80, 25),
            ("F02", "transfer", 16.0, 0.40, 0.90, 45),
            ("F03", "card", 4.5, 0.78, 0.75, 20),
            ("F04", "account_takeover", 22.0, 0.35, 0.95, 55),
            ("F05", "transfer", 11.0, 0.58, 0.90, 40),
            ("F06", "card", 6.5, 0.50, 0.80, 18),
            ("F07", "account_takeover", 30.0, 0.28, 0.95, 60),
            ("F08", "transfer", 9.0, 0.72, 0.88, 35),
            ("F09", "card", 3.5, 0.85, 0.70, 16),
            ("F10", "account_takeover", 18.0, 0.48, 0.92, 50),
            ("F11", "transfer", 14.0, 0.44, 0.90, 42),
            ("F12", "card", 5.0, 0.68, 0.78, 22),
            ("F13", "transfer", 20.0, 0.32, 0.92, 48),
            ("F14", "account_takeover", 26.0, 0.30, 0.96, 58),
            ("F15", "card", 7.0, 0.55, 0.82, 24),
        ],
        columns=[
            "alert_id",
            "category",
            "amount_at_risk",
            "fraud_probability",
            "preventable_fraction",
            "review_minutes",
        ],
    ).set_index("alert_id")
    return FraudTriageProblem(alerts=alerts)


def economics(problem: FraudTriageProblem) -> pd.DataFrame:
    df = problem.alerts.copy()
    df["expected_prevented_loss"] = (
        df["amount_at_risk"]
        * df["fraud_probability"]
        * df["preventable_fraction"]
    )
    df["review_cost"] = (
        df["review_minutes"] * problem.review_cost_per_minute
    )
    df["net_value"] = df["expected_prevented_loss"] - df["review_cost"]
    return df


def solve(problem: FraudTriageProblem | None = None) -> FraudTriageResult:
    p = problem or default_problem()
    df = economics(p)
    n = len(df)

    constraints = LinearConstraint(
        np.vstack([
            df["review_minutes"].to_numpy(float),
            np.ones(n),
        ]),
        lb=np.array([-np.inf, -np.inf]),
        ub=np.array([p.investigator_minutes, p.maximum_alerts], dtype=float),
    )

    result = milp(
        c=-df["net_value"].to_numpy(float),
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(f"fraud triage optimization failed: {result.message}")

    selected = df.loc[result.x > 0.5].copy()
    return FraudTriageResult(
        selected_alerts=selected,
        minutes_used=float(selected["review_minutes"].sum()),
        expected_prevented_loss=float(
            selected["expected_prevented_loss"].sum()
        ),
        net_value=float(selected["net_value"].sum()),
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
