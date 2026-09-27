"""Intraday payment scheduling with endogenous liquidity buffer."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class IntradayLiquidityProblem:
    payments: pd.DataFrame
    incoming_cash: pd.Series
    initial_liquidity: float = 100.0
    maximum_buffer: float = 250.0
    buffer_cost_per_unit: float = 0.05
    max_payments_per_slot: int = 3


@dataclass(frozen=True)
class IntradayLiquidityResult:
    schedule: pd.DataFrame
    liquidity_buffer: float
    liquidity_profile: pd.Series
    total_cost: float

    def to_dict(self) -> dict:
        return {
            "schedule": self.schedule.to_dict(orient="records"),
            "liquidity_buffer": round(self.liquidity_buffer, 6),
            "liquidity_profile": self.liquidity_profile.round(6).to_dict(),
            "total_cost": round(self.total_cost, 6),
        }


def default_problem() -> IntradayLiquidityProblem:
    payments = pd.DataFrame(
        [
            ("PAY01", 100.0, 1, 2, 4.0),
            ("PAY02", 80.0, 1, 3, 2.0),
            ("PAY03", 120.0, 2, 4, 3.0),
            ("PAY04", 60.0, 2, 2, 8.0),
            ("PAY05", 90.0, 3, 5, 2.5),
            ("PAY06", 70.0, 3, 6, 1.5),
            ("PAY07", 110.0, 4, 6, 3.0),
            ("PAY08", 50.0, 5, 6, 4.0),
        ],
        columns=[
            "payment_id",
            "amount",
            "earliest_slot",
            "deadline_slot",
            "delay_penalty_per_slot",
        ],
    ).set_index("payment_id")

    incoming_cash = pd.Series(
        [50.0, 100.0, 60.0, 120.0, 80.0, 100.0],
        index=[1, 2, 3, 4, 5, 6],
        name="incoming_cash",
    )
    return IntradayLiquidityProblem(
        payments=payments,
        incoming_cash=incoming_cash,
    )


def solve(
    problem: IntradayLiquidityProblem | None = None,
) -> IntradayLiquidityResult:
    p = problem or default_problem()
    slots = list(p.incoming_cash.index)
    payments = list(p.payments.index)

    pairs = []
    for payment in payments:
        earliest = int(p.payments.loc[payment, "earliest_slot"])
        deadline = int(p.payments.loc[payment, "deadline_slot"])
        for slot in slots:
            if earliest <= slot <= deadline:
                pairs.append((payment, slot))

    n_x = len(pairs)
    idx_buffer = n_x
    n_vars = n_x + 1

    c = np.zeros(n_vars)
    for k, (payment, slot) in enumerate(pairs):
        earliest = int(p.payments.loc[payment, "earliest_slot"])
        penalty = float(p.payments.loc[payment, "delay_penalty_per_slot"])
        c[k] = penalty * (slot - earliest)
    c[idx_buffer] = p.buffer_cost_per_unit

    # Each payment must be released exactly once.
    eq_rows = []
    for payment in payments:
        row = np.zeros(n_vars)
        for k, (candidate, _) in enumerate(pairs):
            if candidate == payment:
                row[k] = 1.0
        eq_rows.append(row)

    constraints = [
        LinearConstraint(
            np.vstack(eq_rows),
            lb=np.ones(len(payments)),
            ub=np.ones(len(payments)),
        )
    ]

    ub_rows = []
    ub_values = []

    cumulative_incoming = p.incoming_cash.cumsum()
    for slot in slots:
        # cumulative outgoing - buffer <= initial + cumulative incoming
        row = np.zeros(n_vars)
        for k, (payment, release_slot) in enumerate(pairs):
            if release_slot <= slot:
                row[k] = float(p.payments.loc[payment, "amount"])
        row[idx_buffer] = -1.0
        ub_rows.append(row)
        ub_values.append(
            p.initial_liquidity + float(cumulative_incoming.loc[slot])
        )

        # Processing-count capacity in each slot.
        row = np.zeros(n_vars)
        for k, (_, release_slot) in enumerate(pairs):
            if release_slot == slot:
                row[k] = 1.0
        ub_rows.append(row)
        ub_values.append(float(p.max_payments_per_slot))

    constraints.append(
        LinearConstraint(
            np.vstack(ub_rows),
            lb=np.full(len(ub_rows), -np.inf),
            ub=np.asarray(ub_values, dtype=float),
        )
    )

    lower = np.zeros(n_vars)
    upper = np.ones(n_vars)
    upper[idx_buffer] = p.maximum_buffer
    integrality = np.ones(n_vars, dtype=int)
    integrality[idx_buffer] = 0

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(
            f"intraday liquidity optimization failed: {result.message}"
        )

    schedule_rows = []
    for chosen, (payment, slot) in zip(result.x[:n_x] > 0.5, pairs):
        if chosen:
            schedule_rows.append(
                {
                    "payment_id": payment,
                    "slot": int(slot),
                    "amount": float(p.payments.loc[payment, "amount"]),
                }
            )
    schedule = pd.DataFrame(schedule_rows).sort_values(
        ["slot", "payment_id"]
    ).reset_index(drop=True)

    buffer = float(result.x[idx_buffer])
    outgoing_by_slot = (
        schedule.groupby("slot")["amount"].sum()
        .reindex(slots, fill_value=0.0)
    )
    profile = (
        p.initial_liquidity
        + buffer
        + p.incoming_cash.cumsum()
        - outgoing_by_slot.cumsum()
    )
    profile.name = "end_of_slot_liquidity"

    return IntradayLiquidityResult(
        schedule=schedule,
        liquidity_buffer=buffer,
        liquidity_profile=profile,
        total_cost=float(result.fun),
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
