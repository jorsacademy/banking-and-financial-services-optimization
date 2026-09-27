from dataclasses import replace

import numpy as np
import pytest

from cash_supply_chain.model import default_problem
from cash_supply_chain.stochastic_rolling_horizon import (
    compare_rolling_policies,
    generate_demand_scenarios,
    generate_realized_demand,
    run_rolling_horizon,
    solve_stochastic_horizon,
)


@pytest.fixture(scope="module")
def small_problem():
    p = default_problem()
    cashpoints = ["ATM_A", "ATM_B", "BRANCH_C", "ATM_D"]
    days = [1, 2, 3]

    return replace(
        p,
        cashpoints=p.cashpoints.loc[cashpoints].copy(),
        forecast_net_withdrawal=p.forecast_net_withdrawal.loc[
            cashpoints, days
        ].copy(),
        vault_dispatch_limit=p.vault_dispatch_limit.loc[days].copy(),
        cluster_visit_limit={"north": 2, "central": 2},
        stops_per_vehicle=2,
        maximum_vehicles_per_day=2,
    )


def test_scenario_generation_is_reproducible(small_problem):
    a, pa = generate_demand_scenarios(
        small_problem,
        [1, 2],
        scenario_count=2,
        seed=99,
    )
    b, pb = generate_demand_scenarios(
        small_problem,
        [1, 2],
        scenario_count=2,
        seed=99,
    )

    assert pa.equals(pb)
    assert np.isclose(pa.sum(), 1.0)
    for name in a:
        assert a[name].equals(b[name])


def test_stochastic_horizon_first_stage_is_feasible(small_problem):
    days = [1, 2]
    scenarios, probabilities = generate_demand_scenarios(
        small_problem,
        days,
        scenario_count=2,
        demand_sigma=0.10,
        seed=123,
    )
    current = small_problem.cashpoints["initial_cash"].copy()

    result = solve_stochastic_horizon(
        small_problem,
        current_inventory=current,
        days=days,
        demand_scenarios=scenarios,
        probabilities=probabilities,
    )

    assert (
        result.first_day_deliveries
        <= small_problem.cashpoints["maximum_delivery"]
        * result.first_day_visits
        + 1e-7
    ).all()
    assert (
        current + result.first_day_deliveries
        <= small_problem.cashpoints["capacity"] + 1e-7
    ).all()
    assert len(result.first_day_routes) <= small_problem.maximum_vehicles_per_day
    assert (
        result.first_day_routes["load"]
        <= small_problem.vehicle_capacity + 1e-7
    ).all()
    assert result.expected_cashout >= -1e-8
    assert result.expected_safety_shortfall >= -1e-8


def test_stochastic_horizon_returns_scenario_terminal_states(small_problem):
    days = [1, 2]
    scenarios, probabilities = generate_demand_scenarios(
        small_problem,
        days,
        scenario_count=3,
        demand_sigma=0.15,
        seed=321,
    )
    current = small_problem.cashpoints["initial_cash"].copy()

    result = solve_stochastic_horizon(
        small_problem,
        current_inventory=current,
        days=days,
        demand_scenarios=scenarios,
        probabilities=probabilities,
    )

    assert set(result.scenario_terminal_inventory.index) == set(scenarios)
    assert set(result.scenario_terminal_inventory.columns) == set(
        small_problem.cashpoints.index
    )
    assert (result.scenario_terminal_inventory >= -1e-8).all().all()


def test_rolling_horizon_updates_inventory_with_realized_demand(small_problem):
    realized = generate_realized_demand(
        small_problem,
        demand_sigma=0.08,
        seed=77,
    )

    result = run_rolling_horizon(
        small_problem,
        policy="stochastic",
        horizon_days=2,
        scenario_count=2,
        demand_sigma=0.08,
        scenario_seed=900,
        realized_demand=realized,
    )

    previous = small_problem.cashpoints["initial_cash"].copy()

    for day in small_problem.forecast_net_withdrawal.columns:
        before = previous + result.deliveries[day]
        expected_cashout = (result.realized_demand[day] - before).clip(lower=0.0)
        expected_end = (before - result.realized_demand[day]).clip(lower=0.0)

        assert np.allclose(result.cashout[day], expected_cashout)
        assert np.allclose(result.end_inventory[day], expected_end)
        previous = expected_end

    assert result.total_realized_cost >= 0.0


def test_deterministic_and_stochastic_use_same_realized_path(small_problem):
    comparison, results = compare_rolling_policies(
        small_problem,
        horizon_days=2,
        scenario_count=2,
        demand_sigma=0.08,
        seed=44,
    )

    assert results["deterministic"].realized_demand.equals(
        results["stochastic"].realized_demand
    )
    assert set(comparison.index) == {"deterministic", "stochastic"}
    assert (comparison["total_realized_cost"] >= 0).all()
    assert (comparison["total_cashout"] >= 0).all()
