"""Benchmark full route enumeration, column generation, and branch-and-price."""

from pathlib import Path
from time import perf_counter

from cash_supply_chain.column_generation import (
    delivery_requirements_from_plan,
    solve_full_catalog_route_master,
    solve_route_branch_and_price,
    solve_route_column_generation,
)
from cash_supply_chain.model import default_problem, solve


def main() -> None:
    problem = default_problem()

    # The upstream replenishment MILP does not enumerate routes. Pick the day
    # with the largest total planned cash dispatch and route those fixed loads.
    replenishment = solve(problem)
    day = int(replenishment.deliveries.sum(axis=0).idxmax())
    required = delivery_requirements_from_plan(
        replenishment,
        day,
    )

    if required.empty:
        raise RuntimeError(
            "selected benchmark day has no positive replenishment requirements"
        )

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = perf_counter()
    full_objective, full_routes = solve_full_catalog_route_master(
        problem,
        required,
    )
    full_seconds = perf_counter() - t0

    t1 = perf_counter()
    cg = solve_route_column_generation(
        problem,
        required,
    )
    cg_seconds = perf_counter() - t1

    t2 = perf_counter()
    bp = solve_route_branch_and_price(
        problem,
        required,
        max_nodes=100,
    )
    bp_seconds = perf_counter() - t2

    summary = {
        "benchmark_day": day,
        "cashpoints": len(required),
        "full_catalog_objective": full_objective,
        "column_generation_lp_bound": cg.lp_objective,
        "column_generation_integer_objective": cg.integer_objective,
        "branch_and_price_objective": bp.objective_value,
        "branch_and_price_root_bound": bp.root_lp_bound,
        "full_catalog_seconds": full_seconds,
        "column_generation_seconds": cg_seconds,
        "branch_and_price_seconds": bp_seconds,
        "cg_generated_columns": cg.generated_columns,
        "bp_generated_columns": bp.columns_generated,
        "bp_nodes_processed": bp.nodes_processed,
        "bp_exact": bp.exact,
    }

    required.to_csv(output_dir / "routing_fixed_delivery_requirements.csv")
    full_routes.to_csv(
        output_dir / "routing_full_catalog_routes.csv",
        index=False,
    )
    cg.columns.to_csv(
        output_dir / "routing_generated_columns.csv",
        index=False,
    )
    cg.selected_routes.to_csv(
        output_dir / "routing_cg_selected_routes.csv",
        index=False,
    )
    cg.pricing_history.to_csv(
        output_dir / "routing_cg_pricing_history.csv",
        index=False,
    )
    bp.selected_routes.to_csv(
        output_dir / "routing_bp_selected_routes.csv",
        index=False,
    )
    bp.search_log.to_csv(
        output_dir / "routing_bp_search_log.csv",
        index=False,
    )

    print("\nFixed-delivery routing benchmark")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
