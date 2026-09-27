"""Liquidity-buffer and funding-mix optimization."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class LiquidityFundingProblem:
    funding_sources: pd.DataFrame
    stress_outflows: pd.Series
    runoff_matrix: pd.DataFrame
    core_funding_need: float = 1_000.0
    maximum_buffer: float = 350.0
    buffer_carry_cost: float = 0.012
    minimum_stable_funding_share: float = 0.55


@dataclass(frozen=True)
class LiquidityFundingResult:
    funding_mix: pd.Series
    liquidity_buffer: float
    annualized_cost: float
    stressed_requirements: pd.Series
    coverage_surplus: pd.Series
    stable_funding_share: float

    def to_dict(self) -> dict:
        return {
            "funding_mix": self.funding_mix.round(6).to_dict(),
            "liquidity_buffer": round(self.liquidity_buffer, 6),
            "annualized_cost": round(self.annualized_cost, 6),
            "stressed_requirements": self.stressed_requirements.round(6).to_dict(),
            "coverage_surplus": self.coverage_surplus.round(6).to_dict(),
            "stable_funding_share": round(self.stable_funding_share, 6),
        }


def default_problem() -> LiquidityFundingProblem:
    funding_sources = pd.DataFrame(
        {
            "minimum": [250.0, 100.0, 0.0, 0.0, 0.0],
            "maximum": [650.0, 400.0, 250.0, 300.0, 250.0],
            "annual_cost": [0.026, 0.034, 0.041, 0.046, 0.052],
            "stable": [1.0, 1.0, 0.0, 0.0, 0.0],
        },
        index=[
            "retail_deposits",
            "term_deposits",
            "certificates",
            "secured_wholesale",
            "unsecured_wholesale",
        ],
    )

    horizons = ["1m", "3m", "6m", "12m"]
    stress_outflows = pd.Series(
        [55.0, 85.0, 115.0, 150.0],
        index=horizons,
        name="fixed_stress_outflow",
    )

    runoff_matrix = pd.DataFrame(
        [
            [0.02, 0.02, 0.05, 0.08, 0.12],
            [0.04, 0.04, 0.12, 0.18, 0.30],
            [0.06, 0.08, 0.22, 0.30, 0.48],
            [0.10, 0.15, 0.40, 0.55, 0.75],
        ],
        index=horizons,
        columns=funding_sources.index,
    )

    return LiquidityFundingProblem(
        funding_sources=funding_sources,
        stress_outflows=stress_outflows,
        runoff_matrix=runoff_matrix,
    )


def solve(
    problem: LiquidityFundingProblem | None = None,
) -> LiquidityFundingResult:
    p = problem or default_problem()

    sources = list(p.funding_sources.index)
    horizons = list(p.stress_outflows.index)
    n_f = len(sources)
    idx_buffer = n_f
    n_vars = n_f + 1

    # Minimize funding cost plus the carry cost of the liquidity buffer.
    c = np.zeros(n_vars)
    c[:n_f] = p.funding_sources.loc[sources, "annual_cost"].to_numpy(float)
    c[idx_buffer] = p.buffer_carry_cost

    # Funding raised finances the core book plus the liquidity buffer itself.
    a_eq = np.ones((1, n_vars))
    a_eq[0, idx_buffer] = -1.0
    b_eq = np.array([p.core_funding_need], dtype=float)

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    # For every horizon:
    # buffer >= fixed stress outflow + runoff(funding mix).
    for horizon in horizons:
        row = np.zeros(n_vars)
        row[:n_f] = p.runoff_matrix.loc[horizon, sources].to_numpy(float)
        row[idx_buffer] = -1.0
        a_ub.append(row)
        b_ub.append(-float(p.stress_outflows.loc[horizon]))

    # Stable funding share based on the core funding need:
    # stable funding >= minimum_share * core funding need.
    row = np.zeros(n_vars)
    row[:n_f] = -p.funding_sources.loc[sources, "stable"].to_numpy(float)
    a_ub.append(row)
    b_ub.append(-p.minimum_stable_funding_share * p.core_funding_need)

    bounds = list(
        zip(
            p.funding_sources.loc[sources, "minimum"].astype(float),
            p.funding_sources.loc[sources, "maximum"].astype(float),
        )
    )
    bounds.append((0.0, p.maximum_buffer))

    result = linprog(
        c,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"liquidity-funding optimization failed: {result.message}")

    funding_mix = pd.Series(result.x[:n_f], index=sources, name="funding")
    buffer = float(result.x[idx_buffer])

    stressed_requirements = (
        p.runoff_matrix.loc[horizons, sources].mul(funding_mix, axis=1).sum(axis=1)
        + p.stress_outflows.loc[horizons]
    )
    coverage_surplus = buffer - stressed_requirements

    stable_amount = float(
        (
            funding_mix
            * p.funding_sources.loc[sources, "stable"]
        ).sum()
    )
    stable_share = stable_amount / p.core_funding_need

    return LiquidityFundingResult(
        funding_mix=funding_mix,
        liquidity_buffer=buffer,
        annualized_cost=float(result.fun),
        stressed_requirements=stressed_requirements,
        coverage_surplus=coverage_surplus,
        stable_funding_share=stable_share,
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
