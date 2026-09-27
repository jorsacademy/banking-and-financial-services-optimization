"""Curve-aware IRRBB / EVE hedge optimization.

This module is a synthetic research example. It represents the banking-book
interest-rate exposure with key-rate PV01 buckets and chooses a bounded hedge
portfolio to reduce the worst EVE loss across stylized curve shocks.

It is not a regulatory IRRBB calculator.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class IRRBBProblem:
    base_key_rate_pv01: pd.Series
    hedge_key_rate_pv01: pd.DataFrame
    scenario_shocks_bps: pd.DataFrame
    hedge_cost_per_unit: pd.Series
    maximum_hedge_notional: pd.Series
    gross_hedge_limit: float = 70.0
    hedge_cost_weight: float = 0.10


@dataclass(frozen=True)
class IRRBBResult:
    hedge_notionals: pd.Series
    residual_key_rate_pv01: pd.Series
    scenario_eve_change: pd.Series
    worst_eve_loss: float
    unhedged_worst_eve_loss: float
    hedge_cost: float
    gross_hedge_notional: float
    objective_value: float

    @property
    def worst_loss_reduction(self) -> float:
        return self.unhedged_worst_eve_loss - self.worst_eve_loss

    def to_dict(self) -> dict:
        return {
            "hedge_notionals": self.hedge_notionals.round(6).to_dict(),
            "residual_key_rate_pv01": self.residual_key_rate_pv01.round(6).to_dict(),
            "scenario_eve_change": self.scenario_eve_change.round(6).to_dict(),
            "worst_eve_loss": round(self.worst_eve_loss, 6),
            "unhedged_worst_eve_loss": round(self.unhedged_worst_eve_loss, 6),
            "worst_loss_reduction": round(self.worst_loss_reduction, 6),
            "hedge_cost": round(self.hedge_cost, 6),
            "gross_hedge_notional": round(self.gross_hedge_notional, 6),
            "objective_value": round(self.objective_value, 6),
        }


def default_problem() -> IRRBBProblem:
    """Return a deterministic synthetic key-rate hedge instance."""
    tenors = ["2Y", "5Y", "10Y", "20Y"]

    base_key_rate_pv01 = pd.Series(
        [0.42, 0.88, 1.36, 1.05],
        index=tenors,
        name="base_pv01",
    )

    hedge_key_rate_pv01 = pd.DataFrame(
        [
            [-0.018, -0.004, 0.000, 0.000],
            [-0.003, -0.020, -0.006, 0.000],
            [0.000, -0.005, -0.024, -0.006],
            [0.000, 0.000, -0.008, -0.028],
        ],
        index=[
            "pay_2y_swap",
            "pay_5y_swap",
            "pay_10y_swap",
            "pay_20y_swap",
        ],
        columns=tenors,
    )

    scenario_shocks_bps = pd.DataFrame(
        [
            [200.0, 200.0, 200.0, 200.0],
            [-200.0, -200.0, -200.0, -200.0],
            [250.0, 150.0, 50.0, -25.0],
            [-25.0, 50.0, 150.0, 250.0],
            [150.0, 100.0, 0.0, -75.0],
            [-75.0, 0.0, 100.0, 150.0],
        ],
        index=[
            "parallel_up",
            "parallel_down",
            "short_end_up",
            "long_end_up",
            "steepener",
            "flattener",
        ],
        columns=tenors,
    )

    hedges = hedge_key_rate_pv01.index
    hedge_cost_per_unit = pd.Series(
        [0.05, 0.06, 0.08, 0.10],
        index=hedges,
        name="cost",
    )
    maximum_hedge_notional = pd.Series(
        [150.0, 150.0, 150.0, 150.0],
        index=hedges,
        name="maximum_notional",
    )

    return IRRBBProblem(
        base_key_rate_pv01=base_key_rate_pv01,
        hedge_key_rate_pv01=hedge_key_rate_pv01,
        scenario_shocks_bps=scenario_shocks_bps,
        hedge_cost_per_unit=hedge_cost_per_unit,
        maximum_hedge_notional=maximum_hedge_notional,
    )


def _validate(problem: IRRBBProblem) -> None:
    tenors = list(problem.base_key_rate_pv01.index)
    hedges = list(problem.hedge_key_rate_pv01.index)

    if list(problem.hedge_key_rate_pv01.columns) != tenors:
        raise ValueError("hedge key-rate columns must match base PV01 tenors")
    if list(problem.scenario_shocks_bps.columns) != tenors:
        raise ValueError("scenario shock columns must match base PV01 tenors")
    if list(problem.hedge_cost_per_unit.index) != hedges:
        raise ValueError("hedge costs must match hedge instruments")
    if list(problem.maximum_hedge_notional.index) != hedges:
        raise ValueError("hedge limits must match hedge instruments")
    if problem.gross_hedge_limit <= 0:
        raise ValueError("gross_hedge_limit must be positive")
    if problem.hedge_cost_weight < 0:
        raise ValueError("hedge_cost_weight must be nonnegative")
    if (problem.maximum_hedge_notional < 0).any():
        raise ValueError("maximum hedge notionals must be nonnegative")

    arrays = [
        problem.base_key_rate_pv01.to_numpy(dtype=float),
        problem.hedge_key_rate_pv01.to_numpy(dtype=float),
        problem.scenario_shocks_bps.to_numpy(dtype=float),
        problem.hedge_cost_per_unit.to_numpy(dtype=float),
        problem.maximum_hedge_notional.to_numpy(dtype=float),
    ]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("all IRRBB inputs must be finite")


def scenario_eve_change(
    problem: IRRBBProblem,
    residual_key_rate_pv01: pd.Series,
) -> pd.Series:
    """Return first-order EVE change for every curve shock scenario."""
    tenors = list(problem.base_key_rate_pv01.index)
    residual = residual_key_rate_pv01.loc[tenors].to_numpy(dtype=float)
    shocks = problem.scenario_shocks_bps.loc[:, tenors].to_numpy(dtype=float)
    values = -(shocks @ residual)
    return pd.Series(
        values,
        index=problem.scenario_shocks_bps.index,
        name="eve_change",
    )


def unhedged_worst_loss(problem: IRRBBProblem | None = None) -> float:
    """Return the worst positive EVE loss before hedging."""
    p = problem or default_problem()
    _validate(p)
    changes = scenario_eve_change(p, p.base_key_rate_pv01)
    return float(np.maximum(-changes.to_numpy(dtype=float), 0.0).max())


def solve(problem: IRRBBProblem | None = None) -> IRRBBResult:
    """Minimize worst stylized EVE loss plus a linear hedge-cost penalty."""
    p = problem or default_problem()
    _validate(p)

    tenors = list(p.base_key_rate_pv01.index)
    hedges = list(p.hedge_key_rate_pv01.index)
    scenarios = list(p.scenario_shocks_bps.index)

    n_h = len(hedges)
    idx_z = 2 * n_h
    n_vars = idx_z + 1

    # Signed hedge h_j = h_j^+ - h_j^-.
    c = np.zeros(n_vars)
    unit_cost = p.hedge_cost_per_unit.loc[hedges].to_numpy(dtype=float)
    c[:n_h] = p.hedge_cost_weight * unit_cost
    c[n_h : 2 * n_h] = p.hedge_cost_weight * unit_cost
    c[idx_z] = 1.0

    base = p.base_key_rate_pv01.loc[tenors].to_numpy(dtype=float)
    hedge_matrix = (
        p.hedge_key_rate_pv01.loc[hedges, tenors]
        .to_numpy(dtype=float)
        .T
    )

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    # z >= loss_s = residual_PV01 dot shock_s.
    for scenario in scenarios:
        shock = p.scenario_shocks_bps.loc[scenario, tenors].to_numpy(dtype=float)
        hedge_loss_coeff = shock @ hedge_matrix

        row = np.zeros(n_vars)
        row[:n_h] = hedge_loss_coeff
        row[n_h : 2 * n_h] = -hedge_loss_coeff
        row[idx_z] = -1.0
        a_ub.append(row)
        b_ub.append(-float(shock @ base))

    # Gross hedge budget.
    row = np.zeros(n_vars)
    row[: 2 * n_h] = 1.0
    a_ub.append(row)
    b_ub.append(float(p.gross_hedge_limit))

    bounds: list[tuple[float | None, float | None]] = []
    for hedge in hedges:
        bounds.append((0.0, float(p.maximum_hedge_notional.loc[hedge])))
    for hedge in hedges:
        bounds.append((0.0, float(p.maximum_hedge_notional.loc[hedge])))
    bounds.append((0.0, None))

    result = linprog(
        c,
        A_ub=np.asarray(a_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"IRRBB hedge optimization failed: {result.message}")

    signed_hedge = result.x[:n_h] - result.x[n_h : 2 * n_h]
    hedge_notionals = pd.Series(
        signed_hedge,
        index=hedges,
        name="hedge_notional",
    )

    residual = base + hedge_matrix @ signed_hedge
    residual_key_rate_pv01 = pd.Series(
        residual,
        index=tenors,
        name="residual_pv01",
    )

    changes = scenario_eve_change(p, residual_key_rate_pv01)
    worst_loss = float(
        np.maximum(-changes.to_numpy(dtype=float), 0.0).max()
    )
    hedge_cost = float(
        (
            hedge_notionals.abs()
            * p.hedge_cost_per_unit.loc[hedges]
        ).sum()
    )
    gross = float(hedge_notionals.abs().sum())

    return IRRBBResult(
        hedge_notionals=hedge_notionals,
        residual_key_rate_pv01=residual_key_rate_pv01,
        scenario_eve_change=changes,
        worst_eve_loss=worst_loss,
        unhedged_worst_eve_loss=unhedged_worst_loss(p),
        hedge_cost=hedge_cost,
        gross_hedge_notional=gross,
        objective_value=float(result.fun),
    )


def main() -> None:
    print(json.dumps(solve().to_dict(), indent=2))


if __name__ == "__main__":
    main()
