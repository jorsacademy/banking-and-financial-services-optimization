import numpy as np
import pytest

from cash_supply_chain.joint_irp import (
    compare_staged_and_joint,
    default_problem,
    generate_route_catalog,
    solve_joint_irp,
)


@pytest.fixture(scope="module")
def problem():
    return default_problem()


@pytest.fixture(scope="module")
def result(problem):
    return solve_joint_irp(problem)


@pytest.fixture(scope="module")
def comparison(problem, result):
    return compare_staged_and_joint(problem, joint=result)


def test_route_catalog_respects_stop_limit(problem):
    catalog = generate_route_catalog(problem)

    assert len(catalog) > 0
    assert (catalog["stop_count"] <= problem.stops_per_vehicle).all()
    assert (catalog["distance"] > 0).all()


def test_joint_irp_inventory_balance(problem, result):
    for cashpoint in problem.cashpoints.index:
        previous = problem.cashpoints.loc[cashpoint, "initial_cash"]
        for day in problem.forecast_net_withdrawal.columns:
            expected = (
                previous
                + result.deliveries.loc[cashpoint, day]
                - problem.forecast_net_withdrawal.loc[cashpoint, day]
            )
            assert np.isclose(result.end_inventory.loc[cashpoint, day], expected)
            previous = result.end_inventory.loc[cashpoint, day]


def test_joint_irp_route_and_vault_limits(problem, result):
    for day in problem.forecast_net_withdrawal.columns:
        day_routes = result.selected_routes[result.selected_routes["day"] == day]

        assert len(day_routes) <= problem.maximum_vehicles_per_day
        assert (day_routes["load"] <= problem.vehicle_capacity + 1e-7).all()
        assert (
            result.deliveries[day].sum()
            <= problem.vault_dispatch_limit.loc[day] + 1e-7
        )

        for cluster, limit in problem.cluster_visit_limit.items():
            members = problem.cashpoints.index[
                problem.cashpoints["cluster"] == cluster
            ]
            assert result.visits.loc[members, day].sum() <= limit + 1e-7


def test_joint_irp_cashpoint_delivery_limits(problem, result):
    for cashpoint in problem.cashpoints.index:
        max_delivery = problem.cashpoints.loc[cashpoint, "maximum_delivery"]
        assert (
            result.deliveries.loc[cashpoint].to_numpy()
            <= max_delivery * result.visits.loc[cashpoint].to_numpy() + 1e-7
        ).all()


def test_joint_irp_cost_breakdown_matches_objective(result):
    assert np.isclose(
        float(result.cost_breakdown.sum()),
        result.total_cost,
        atol=1e-6,
    )


def test_joint_irp_not_worse_than_staged_full_cost(comparison):
    assert (
        comparison.loc["joint_irp", "integrated_total_cost"]
        <= comparison.loc["staged", "integrated_total_cost"] + 1e-6
    )
    assert (
        comparison.loc["joint_irp", "cost_improvement_vs_staged"]
        >= -1e-6
    )
