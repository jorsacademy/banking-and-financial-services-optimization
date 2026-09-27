from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cash_supply_chain.column_generation import (
    solve_full_catalog_route_master,
    solve_route_branch_and_price,
    solve_route_column_generation,
)
from cash_supply_chain.model import default_problem


@pytest.fixture(scope="module")
def routing_problem():
    p = default_problem()
    cashpoints = ["ATM_A", "ATM_B", "BRANCH_C", "ATM_D"]

    return replace(
        p,
        cashpoints=p.cashpoints.loc[cashpoints].copy(),
        cluster_visit_limit={"north": 2, "central": 2},
        stops_per_vehicle=2,
        maximum_vehicles_per_day=2,
        vehicle_capacity=260.0,
    )


@pytest.fixture(scope="module")
def fixed_requirements():
    return pd.Series(
        {
            "ATM_A": 70.0,
            "ATM_B": 60.0,
            "BRANCH_C": 105.0,
            "ATM_D": 80.0,
        },
        dtype=float,
    )


def test_column_generation_matches_full_catalog(
    routing_problem,
    fixed_requirements,
):
    full_objective, _ = solve_full_catalog_route_master(
        routing_problem,
        fixed_requirements,
    )
    cg = solve_route_column_generation(
        routing_problem,
        fixed_requirements,
        max_new_columns_per_iteration=3,
    )

    assert cg.artificial_mass <= 1e-7
    assert cg.lp_objective <= cg.integer_objective + 1e-7
    assert np.isclose(
        cg.integer_objective,
        full_objective,
        atol=1e-6,
    )
    assert cg.generated_columns >= 0
    assert len(cg.selected_routes) <= routing_problem.maximum_vehicles_per_day


def test_selected_cg_routes_cover_each_cashpoint_once(
    routing_problem,
    fixed_requirements,
):
    cg = solve_route_column_generation(
        routing_problem,
        fixed_requirements,
    )

    coverage = {cashpoint: 0 for cashpoint in fixed_requirements.index}
    for stops in cg.selected_routes["stops"]:
        for cashpoint in stops.split("|"):
            coverage[cashpoint] += 1

    assert all(value == 1 for value in coverage.values())
    assert (
        cg.selected_routes["load"]
        <= routing_problem.vehicle_capacity + 1e-7
    ).all()


@pytest.fixture(scope="module")
def fractional_branch_problem():
    p = default_problem()
    cashpoints = ["ATM_A", "ATM_B", "BRANCH_C"]
    cp = p.cashpoints.loc[cashpoints].copy()
    cp.loc[:, "visit_cost"] = 0.0

    return replace(
        p,
        cashpoints=cp,
        cluster_visit_limit={"north": 2, "central": 1},
        stops_per_vehicle=2,
        maximum_vehicles_per_day=2,
        vehicle_capacity=500.0,
        vehicle_fixed_cost=100.0,
        distance_cost_per_unit=0.0,
    )


@pytest.fixture(scope="module")
def fractional_requirements():
    return pd.Series(
        {
            "ATM_A": 10.0,
            "ATM_B": 10.0,
            "BRANCH_C": 10.0,
        },
        dtype=float,
    )


def test_branch_and_price_closes_fractional_root(
    fractional_branch_problem,
    fractional_requirements,
):
    full_objective, _ = solve_full_catalog_route_master(
        fractional_branch_problem,
        fractional_requirements,
    )
    bp = solve_route_branch_and_price(
        fractional_branch_problem,
        fractional_requirements,
        max_nodes=20,
        max_new_columns_per_iteration=3,
    )

    assert bp.exact
    assert bp.nodes_processed >= 2
    assert bp.root_lp_bound < bp.objective_value - 1e-6
    assert np.isclose(
        bp.objective_value,
        full_objective,
        atol=1e-6,
    )


def test_branch_and_price_routes_are_capacity_feasible(
    fractional_branch_problem,
    fractional_requirements,
):
    bp = solve_route_branch_and_price(
        fractional_branch_problem,
        fractional_requirements,
        max_nodes=20,
    )

    assert (
        bp.selected_routes["load"]
        <= fractional_branch_problem.vehicle_capacity + 1e-7
    ).all()
    assert (
        len(bp.selected_routes)
        <= fractional_branch_problem.maximum_vehicles_per_day
    )
