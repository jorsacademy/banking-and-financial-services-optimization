"""Monte Carlo validation for a fixed cash replenishment plan."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .model import CashSupplyChainProblem, CashSupplyChainResult, default_problem, solve


def simulate_plan(
    result: CashSupplyChainResult,
    problem: CashSupplyChainProblem | None = None,
    replications: int = 500,
    demand_sigma: float = 0.12,
    seed: int = 7,
) -> pd.DataFrame:
    """Simulate forecast error while keeping the optimized delivery plan fixed.

    Demand is perturbed multiplicatively with a lognormal factor centered near
    one. The simulation reports realized stockout volume and idle cash.
    """
    p = problem or default_problem()
    rng = np.random.default_rng(seed)

    cashpoints = list(p.cashpoints.index)
    days = list(p.forecast_net_withdrawal.columns)
    forecast = p.forecast_net_withdrawal.to_numpy(float)
    deliveries = result.deliveries.loc[cashpoints, days].to_numpy(float)

    rows = []
    for rep in range(replications):
        factors = rng.lognormal(
            mean=-0.5 * demand_sigma**2,
            sigma=demand_sigma,
            size=forecast.shape,
        )
        realized = forecast * factors

        cashout = 0.0
        idle_cash = 0.0
        service_events = 0
        total_events = len(cashpoints) * len(days)

        inventory = p.cashpoints.loc[cashpoints, "initial_cash"].to_numpy(float)

        for t, _day in enumerate(days):
            inventory = inventory + deliveries[:, t] - realized[:, t]

            stockout = np.maximum(-inventory, 0.0)
            cashout += float(stockout.sum())
            service_events += int(np.count_nonzero(stockout <= 1e-9))

            # Unmet withdrawals are treated as lost service; inventory floors at 0.
            inventory = np.maximum(inventory, 0.0)
            idle_cash += float(inventory.sum())

        rows.append(
            {
                "replication": rep,
                "cashout_volume": cashout,
                "average_idle_cash": idle_cash / len(days),
                "service_event_rate": service_events / total_events,
            }
        )

    return pd.DataFrame(rows)


def summarize_simulation(simulation: pd.DataFrame) -> pd.Series:
    """Return compact risk/service metrics from simulation output."""
    return pd.Series(
        {
            "mean_cashout_volume": simulation["cashout_volume"].mean(),
            "p95_cashout_volume": simulation["cashout_volume"].quantile(0.95),
            "mean_idle_cash": simulation["average_idle_cash"].mean(),
            "mean_service_event_rate": simulation["service_event_rate"].mean(),
        }
    )


def fleet_sensitivity(
    problem: CashSupplyChainProblem | None = None,
    fleet_sizes: tuple[int, ...] = (1, 2, 3, 4),
) -> pd.DataFrame:
    """Re-solve the model for alternative maximum fleet sizes."""
    base = problem or default_problem()
    rows = []

    for fleet_size in fleet_sizes:
        candidate = replace(base, maximum_vehicles_per_day=int(fleet_size))
        try:
            result = solve(candidate)
            rows.append(
                {
                    "maximum_vehicles_per_day": fleet_size,
                    "feasible": True,
                    "total_cost": result.total_cost,
                    "total_shortage": float(result.shortage.to_numpy().sum()),
                    "total_visits": int(result.visits.to_numpy().sum()),
                    "vehicle_days": float(result.vehicles.sum()),
                }
            )
        except RuntimeError:
            rows.append(
                {
                    "maximum_vehicles_per_day": fleet_size,
                    "feasible": False,
                    "total_cost": np.nan,
                    "total_shortage": np.nan,
                    "total_visits": np.nan,
                    "vehicle_days": np.nan,
                }
            )

    return pd.DataFrame(rows)
