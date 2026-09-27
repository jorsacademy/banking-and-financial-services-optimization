"""Run deterministic-vs-stochastic rolling-horizon IRP experiments."""

from pathlib import Path

from cash_supply_chain.model import default_problem
from cash_supply_chain.stochastic_rolling_horizon import compare_rolling_policies


if __name__ == "__main__":
    problem = default_problem()

    comparison, results = compare_rolling_policies(
        problem,
        horizon_days=3,
        scenario_count=4,
        demand_sigma=0.12,
        seed=2026,
    )

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(output_dir / "rolling_policy_comparison.csv")

    for policy, result in results.items():
        result.deliveries.to_csv(
            output_dir / f"{policy}_rolling_deliveries.csv"
        )
        result.realized_demand.to_csv(
            output_dir / f"{policy}_rolling_realized_demand.csv"
        )
        result.end_inventory.to_csv(
            output_dir / f"{policy}_rolling_end_inventory.csv"
        )
        result.cashout.to_csv(
            output_dir / f"{policy}_rolling_cashout.csv"
        )
        result.selected_routes.to_csv(
            output_dir / f"{policy}_rolling_routes.csv",
            index=False,
        )
        result.daily_summary.to_csv(
            output_dir / f"{policy}_rolling_daily_summary.csv"
        )

    print("\nRolling-horizon policy comparison")
    print(comparison.round(4).to_string())

    print("\nStochastic rolling daily summary")
    print(results["stochastic"].daily_summary.round(4).to_string())
