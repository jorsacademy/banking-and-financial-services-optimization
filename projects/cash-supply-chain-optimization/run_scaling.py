"""Run scenario reduction and Progressive Hedging scaling experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from cash_supply_chain.model import default_problem
from cash_supply_chain.multistage_cvar_irp import (
    generate_binary_scenario_tree,
    solve_multistage_cvar_irp,
)
from cash_supply_chain.progressive_hedging import (
    benchmark_progressive_hedging,
    run_progressive_hedging,
)
from cash_supply_chain.scenario_reduction import reduce_scenario_tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-leaves", type=int, default=4)
    parser.add_argument("--ph-iterations", type=int, default=10)
    parser.add_argument(
        "--full-exact",
        action="store_true",
        help="Also solve the unreduced extensive form; can be substantially slower.",
    )
    args = parser.parse_args()

    problem = default_problem()

    full_tree = generate_binary_scenario_tree(
        problem,
        days=(1, 2, 3, 4),
        demand_sigma=0.12,
        seed=2026,
    )

    t0 = perf_counter()
    reduction = reduce_scenario_tree(
        full_tree,
        target_leaves=args.target_leaves,
    )
    reduction_seconds = perf_counter() - t0

    reduced_tree = reduction.reduced_tree

    t1 = perf_counter()
    reduced_exact = solve_multistage_cvar_irp(
        problem,
        tree=reduced_tree,
        risk_aversion=0.0,
    )
    reduced_exact_seconds = perf_counter() - t1

    t2 = perf_counter()
    ph = run_progressive_hedging(
        problem,
        reduced_tree,
        max_iterations=args.ph_iterations,
        tolerance=0.02,
    )
    ph_seconds = perf_counter() - t2

    benchmark, _ = benchmark_progressive_hedging(
        problem,
        reduced_tree,
        ph_result=ph,
    )

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    reduction.summary().to_csv(
        output_dir / "scenario_reduction_summary.csv"
    )
    reduction.original_probabilities.to_csv(
        output_dir / "scenario_original_probabilities.csv"
    )
    reduction.reduced_probabilities.to_csv(
        output_dir / "scenario_reduced_probabilities.csv"
    )
    reduction.assignment.to_csv(
        output_dir / "scenario_reduction_assignment.csv"
    )
    reduced_tree.nodes.to_csv(
        output_dir / "reduced_tree_nodes.csv"
    )
    reduced_tree.demand.to_csv(
        output_dir / "reduced_tree_demand.csv"
    )

    ph.convergence_history.to_csv(
        output_dir / "ph_convergence.csv",
        index=False,
    )
    ph.consensus_routes.to_csv(
        output_dir / "ph_consensus_routes.csv"
    )
    ph.consensus_deliveries.to_csv(
        output_dir / "ph_consensus_deliveries.csv"
    )
    ph.scenario_costs.to_csv(
        output_dir / "ph_scenario_costs.csv"
    )
    benchmark.to_csv(
        output_dir / "ph_vs_exact_reduced.csv"
    )

    reduced_exact.delivery_policy.to_csv(
        output_dir / "reduced_exact_deliveries.csv"
    )
    reduced_exact.route_policy.to_csv(
        output_dir / "reduced_exact_routes.csv",
        index=False,
    )

    print("\nScenario reduction")
    print(reduction.summary().round(4).to_string())
    print(f"reduction_seconds: {reduction_seconds:.4f}")

    print("\nReduced-tree exact vs PH")
    print(benchmark.round(4).to_string())
    print(f"reduced_exact_seconds: {reduced_exact_seconds:.4f}")
    print(f"ph_seconds: {ph_seconds:.4f}")

    if args.full_exact:
        t3 = perf_counter()
        full_exact = solve_multistage_cvar_irp(
            problem,
            tree=full_tree,
            risk_aversion=0.0,
        )
        full_exact_seconds = perf_counter() - t3

        full_exact.delivery_policy.to_csv(
            output_dir / "full_exact_deliveries.csv"
        )
        full_exact.route_policy.to_csv(
            output_dir / "full_exact_routes.csv",
            index=False,
        )

        print("\nFull-tree exact solve")
        print(f"expected_cost: {full_exact.expected_cost:.4f}")
        print(f"full_exact_seconds: {full_exact_seconds:.4f}")


if __name__ == "__main__":
    main()
