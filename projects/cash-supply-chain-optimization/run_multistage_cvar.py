"""Run risk-neutral vs CVaR-averse multistage IRP experiment."""

from pathlib import Path

from cash_supply_chain.model import default_problem
from cash_supply_chain.multistage_cvar_irp import (
    compare_risk_attitudes,
    generate_binary_scenario_tree,
)


if __name__ == "__main__":
    problem = default_problem()
    tree = generate_binary_scenario_tree(
        problem,
        days=(1, 2, 3),
        demand_sigma=0.12,
        seed=2026,
    )

    comparison, results = compare_risk_attitudes(
        problem,
        tree=tree,
        cvar_alpha=0.90,
        risk_aversion=0.75,
    )

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    tree.nodes.to_csv(output_dir / "multistage_tree_nodes.csv")
    tree.demand.to_csv(output_dir / "multistage_tree_demand.csv")
    comparison.to_csv(output_dir / "multistage_risk_comparison.csv")

    for name, result in results.items():
        result.delivery_policy.to_csv(
            output_dir / f"{name}_multistage_deliveries.csv"
        )
        result.route_policy.to_csv(
            output_dir / f"{name}_multistage_routes.csv",
            index=False,
        )
        result.inventory.to_csv(
            output_dir / f"{name}_multistage_inventory.csv"
        )
        result.cashout.to_csv(
            output_dir / f"{name}_multistage_cashout.csv"
        )
        result.safety_shortfall.to_csv(
            output_dir / f"{name}_multistage_safety_shortfall.csv"
        )
        result.leaf_costs.to_csv(
            output_dir / f"{name}_leaf_costs.csv"
        )

    print("\nRisk-neutral vs CVaR-averse multistage IRP")
    print(comparison.round(4).to_string())

    print("\nRisk-averse leaf costs")
    print(results["risk_averse"].leaf_costs.round(4).to_string())
