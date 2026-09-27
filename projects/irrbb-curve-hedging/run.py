from pathlib import Path

import pandas as pd

from banking_optimization.irrbb import default_problem, solve


def main() -> None:
    problem = default_problem()
    result = solve(problem)

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    result.hedge_notionals.rename("notional").to_csv(
        output_dir / "optimal_hedge_notionals.csv"
    )
    result.residual_key_rate_pv01.rename("pv01").to_csv(
        output_dir / "residual_key_rate_pv01.csv"
    )
    result.scenario_eve_change.rename("eve_change").to_csv(
        output_dir / "scenario_eve_change.csv"
    )

    pd.DataFrame(
        [
            {
                "policy": "unhedged",
                "worst_eve_loss": result.unhedged_worst_eve_loss,
                "gross_hedge_notional": 0.0,
                "hedge_cost": 0.0,
            },
            {
                "policy": "optimized_hedge",
                "worst_eve_loss": result.worst_eve_loss,
                "gross_hedge_notional": result.gross_hedge_notional,
                "hedge_cost": result.hedge_cost,
            },
        ]
    ).to_csv(output_dir / "hedge_comparison.csv", index=False)

    print(result.to_dict())


if __name__ == "__main__":
    main()
