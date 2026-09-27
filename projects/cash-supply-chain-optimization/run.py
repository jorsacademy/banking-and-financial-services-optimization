"""Run staged and joint cash-supply-chain experiments."""

from pathlib import Path

from cash_supply_chain.joint_irp import (
    compare_staged_and_joint,
    solve_joint_irp,
)
from cash_supply_chain.model import default_problem, solve
from cash_supply_chain.routing import route_plan
from cash_supply_chain.simulation import (
    fleet_sensitivity,
    simulate_plan,
    summarize_simulation,
)


if __name__ == "__main__":
    problem = default_problem()

    staged = solve(problem)
    staged_routes = route_plan(staged, problem)

    joint = solve_joint_irp(problem)
    comparison = compare_staged_and_joint(
        problem,
        staged=staged,
        joint=joint,
        staged_routes=staged_routes,
    )

    staged_simulation = simulate_plan(staged, problem)
    joint_simulation = simulate_plan(joint, problem)
    staged_summary = summarize_simulation(staged_simulation)
    joint_summary = summarize_simulation(joint_simulation)

    sensitivity = fleet_sensitivity(problem)

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    staged.deliveries.to_csv(output_dir / "staged_deliveries.csv")
    staged_routes.to_csv(output_dir / "staged_cit_routes.csv", index=False)

    joint.deliveries.to_csv(output_dir / "joint_deliveries.csv")
    joint.visits.to_csv(output_dir / "joint_visits.csv")
    joint.end_inventory.to_csv(output_dir / "joint_end_inventory.csv")
    joint.shortage.to_csv(output_dir / "joint_planned_shortage.csv")
    joint.selected_routes.to_csv(output_dir / "joint_cit_routes.csv", index=False)
    joint.cost_breakdown.to_csv(output_dir / "joint_cost_breakdown.csv")

    comparison.to_csv(output_dir / "staged_vs_joint.csv")

    staged_simulation.to_csv(
        output_dir / "staged_monte_carlo_simulation.csv",
        index=False,
    )
    joint_simulation.to_csv(
        output_dir / "joint_monte_carlo_simulation.csv",
        index=False,
    )
    staged_summary.to_csv(output_dir / "staged_simulation_summary.csv")
    joint_summary.to_csv(output_dir / "joint_simulation_summary.csv")

    sensitivity.to_csv(output_dir / "fleet_sensitivity.csv", index=False)

    print("\nStaged vs joint IRP")
    print(comparison.round(4).to_string())

    print("\nStaged simulation")
    print(staged_summary.round(4).to_string())

    print("\nJoint IRP simulation")
    print(joint_summary.round(4).to_string())

    print("\nFleet sensitivity")
    print(sensitivity.round(4).to_string(index=False))
