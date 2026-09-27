"""Route payment segments across processors under capacity constraints."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class PaymentRoutingProblem:
    segments: pd.DataFrame
    routes: pd.DataFrame
    eligibility: pd.DataFrame
    failure_penalty_per_transaction: float = 2.50
    latency_penalty_per_second: float = 0.002


@dataclass(frozen=True)
class PaymentRoutingResult:
    assignments: pd.DataFrame
    route_volume: pd.Series
    total_expected_cost: float

    def to_dict(self) -> dict:
        return {
            "assignments": self.assignments.to_dict(orient="records"),
            "route_volume": self.route_volume.round(6).to_dict(),
            "total_expected_cost": round(self.total_expected_cost, 6),
        }


def default_problem() -> PaymentRoutingProblem:
    segments = pd.DataFrame(
        [
            ("P01", 90, "domestic"),
            ("P02", 120, "domestic"),
            ("P03", 70, "cross_border"),
            ("P04", 110, "domestic"),
            ("P05", 80, "cross_border"),
            ("P06", 65, "domestic"),
            ("P07", 95, "cross_border"),
            ("P08", 75, "domestic"),
        ],
        columns=["segment_id", "volume", "type"],
    ).set_index("segment_id")

    routes = pd.DataFrame(
        {
            "capacity": [300.0, 280.0, 260.0],
            "fee_per_transaction": [0.22, 0.18, 0.26],
            "success_probability": [0.985, 0.972, 0.992],
            "latency_seconds": [1.2, 0.8, 1.8],
        },
        index=["processor_a", "processor_b", "processor_c"],
    )

    eligibility = pd.DataFrame(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        index=segments.index,
        columns=routes.index,
        dtype=int,
    )

    return PaymentRoutingProblem(
        segments=segments,
        routes=routes,
        eligibility=eligibility,
    )


def solve(problem: PaymentRoutingProblem | None = None) -> PaymentRoutingResult:
    p = problem or default_problem()
    segments = list(p.segments.index)
    routes = list(p.routes.index)

    pairs = [
        (s, r)
        for s in segments
        for r in routes
        if p.eligibility.loc[s, r] == 1
    ]
    n = len(pairs)

    unit_cost = []
    for segment, route in pairs:
        route_data = p.routes.loc[route]
        expected_failure_cost = (
            (1.0 - route_data["success_probability"])
            * p.failure_penalty_per_transaction
        )
        latency_cost = (
            route_data["latency_seconds"]
            * p.latency_penalty_per_second
        )
        unit_cost.append(
            route_data["fee_per_transaction"]
            + expected_failure_cost
            + latency_cost
        )

    c = np.array(
        [
            p.segments.loc[s, "volume"] * cost
            for (s, _), cost in zip(pairs, unit_cost)
        ],
        dtype=float,
    )

    eq_rows = []
    for segment in segments:
        eq_rows.append(
            np.array([1.0 if s == segment else 0.0 for s, _ in pairs])
        )

    ub_rows = []
    ub_values = []
    for route in routes:
        ub_rows.append(
            np.array(
                [
                    p.segments.loc[s, "volume"] if r == route else 0.0
                    for s, r in pairs
                ],
                dtype=float,
            )
        )
        ub_values.append(float(p.routes.loc[route, "capacity"]))

    constraints = [
        LinearConstraint(
            np.vstack(eq_rows),
            lb=np.ones(len(segments)),
            ub=np.ones(len(segments)),
        ),
        LinearConstraint(
            np.vstack(ub_rows),
            lb=np.full(len(routes), -np.inf),
            ub=np.asarray(ub_values),
        ),
    ]

    result = milp(
        c=c,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(f"payment routing optimization failed: {result.message}")

    rows = []
    for chosen, (segment, route), cost in zip(result.x > 0.5, pairs, unit_cost):
        if chosen:
            rows.append(
                {
                    "segment_id": segment,
                    "route": route,
                    "volume": float(p.segments.loc[segment, "volume"]),
                    "unit_expected_cost": float(cost),
                }
            )
    assignments = pd.DataFrame(rows)
    route_volume = (
        assignments.groupby("route")["volume"].sum()
        .reindex(routes, fill_value=0.0)
    )

    return PaymentRoutingResult(
        assignments=assignments,
        route_volume=route_volume,
        total_expected_cost=float(result.fun),
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
