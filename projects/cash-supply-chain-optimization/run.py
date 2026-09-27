"""Run optimization, simulation, and fleet sensitivity."""

from pathlib import Path

from cash_supply_chain.model import default_problem, solve
from cash_supply_chain.simulation import (
    fleet_sensitivity,
    simulate_plan,
    summarize_simulation,
)


if __name__ == "__main__":
    problem = default_problem()
    result = solve(problem)
    simulation = simulate_plan(result, problem)
    summary = summarize_simulation(simulation)
    sensitivity = fleet_sensitivity(problem)

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    result.deliveries.to_csv(output_dir / "deliveries.csv")
    result.visits.to_csv(output_dir / "visits.csv")
    result.end_inventory.to_csv(output_dir / "end_inventory.csv")
    result.shortage.to_csv(output_dir / "planned_shortage.csv")
    result.vehicles.to_csv(output_dir / "vehicles.csv")
    result.cost_breakdown.to_csv(output_dir / "cost_breakdown.csv")
    simulation.to_csv(output_dir / "monte_carlo_simulation.csv", index=False)
    summary.to_csv(output_dir / "simulation_summary.csv")
    sensitivity.to_csv(output_dir / "fleet_sensitivity.csv", index=False)

    print("total_cost:", round(result.total_cost, 4))
    print(summary.round(4).to_string())
    print(sensitivity.round(4).to_string(index=False))
