"""Experiment utilities for the stochastic ALM project."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from banking_optimization.alm_stochastic import (
    StochasticALMProblem,
    compare_policies,
    default_stochastic_problem,
    solve_stochastic,
)


def policy_summary(
    problem: StochasticALMProblem | None = None,
) -> pd.DataFrame:
    """Return a compact adaptive-vs-static policy comparison."""
    return compare_policies(problem or default_stochastic_problem()).copy()


def sensitivity_grid(
    problem: StochasticALMProblem | None = None,
    duration_gaps: tuple[float, ...] = (0.50, 0.80, 1.10),
    shortfall_penalties: tuple[float, ...] = (0.5, 2.0, 5.0),
) -> pd.DataFrame:
    """Evaluate adaptive and static ALM policies across a small parameter grid."""
    base = problem or default_stochastic_problem()
    rows = []

    for duration_gap in duration_gaps:
        for penalty in shortfall_penalties:
            candidate = replace(
                base,
                maximum_duration_gap_years=float(duration_gap),
                shortfall_penalty=float(penalty),
            )
            adaptive = solve_stochastic(candidate, adaptive=True)
            static = solve_stochastic(candidate, adaptive=False)

            rows.append(
                {
                    "maximum_duration_gap_years": float(duration_gap),
                    "shortfall_penalty": float(penalty),
                    "adaptive_expected_nii": adaptive.expected_total_nii,
                    "static_expected_nii": static.expected_total_nii,
                    "adaptive_expected_shortfall": adaptive.expected_total_shortfall,
                    "static_expected_shortfall": static.expected_total_shortfall,
                    "adaptive_objective": adaptive.objective_value,
                    "static_objective": static.objective_value,
                    "value_of_adaptivity": (
                        static.objective_value - adaptive.objective_value
                    ),
                }
            )

    return pd.DataFrame(rows)


def export_experiment_outputs(
    output_dir: str | Path,
    problem: StochasticALMProblem | None = None,
) -> dict[str, Path]:
    """Write reproducible CSV outputs for the default ALM experiment."""
    p = problem or default_stochastic_problem()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    adaptive = solve_stochastic(p, adaptive=True)
    comparison = compare_policies(p)
    sensitivity = sensitivity_grid(p)

    paths = {
        "policy_comparison": output_path / "policy_comparison.csv",
        "sensitivity_grid": output_path / "sensitivity_grid.csv",
        "adaptive_asset_policy": output_path / "adaptive_asset_policy.csv",
        "adaptive_funding_policy": output_path / "adaptive_funding_policy.csv",
        "adaptive_node_metrics": output_path / "adaptive_node_metrics.csv",
    }

    comparison.to_csv(paths["policy_comparison"])
    sensitivity.to_csv(paths["sensitivity_grid"], index=False)
    adaptive.asset_policy.to_csv(paths["adaptive_asset_policy"])
    adaptive.funding_policy.to_csv(paths["adaptive_funding_policy"])

    node_metrics = pd.DataFrame(
        {
            "period": p.tree["period"],
            "probability": p.tree["probability"],
            "nii": adaptive.node_nii,
            "nii_shortfall": adaptive.node_shortfall,
            "capital_ratio": adaptive.node_capital_ratio,
            "liquidity_surplus": adaptive.node_liquidity_surplus,
            "duration_gap_years": adaptive.node_duration_gap,
            "hedge_notional": adaptive.hedge_policy,
        }
    )
    node_metrics.to_csv(paths["adaptive_node_metrics"])

    return paths
