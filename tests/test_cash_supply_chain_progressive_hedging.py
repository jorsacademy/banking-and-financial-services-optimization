from dataclasses import replace

import numpy as np
import pytest

from cash_supply_chain.model import default_problem
from cash_supply_chain.multistage_cvar_irp import (
    generate_binary_scenario_tree,
)
from cash_supply_chain.progressive_hedging import (
    benchmark_progressive_hedging,
    run_progressive_hedging,
)
from cash_supply_chain.scenario_reduction import reduce_scenario_tree


@pytest.fixture(scope="module")
def small_problem():
    p = default_problem()
    cashpoints = ["ATM_A", "ATM_B", "BRANCH_C"]
    days = [1, 2]

    return replace(
        p,
        cashpoints=p.cashpoints.loc[cashpoints].copy(),
        forecast_net_withdrawal=p.forecast_net_withdrawal.loc[
            cashpoints, days
        ].copy(),
        vault_dispatch_limit=p.vault_dispatch_limit.loc[days].copy(),
        cluster_visit_limit={"north": 2, "central": 1},
        stops_per_vehicle=2,
        maximum_vehicles_per_day=2,
    )


@pytest.fixture(scope="module")
def reduced_tree(small_problem):
    tree = generate_binary_scenario_tree(
        small_problem,
        days=(1, 2),
        demand_sigma=0.08,
        seed=811,
    )
    return reduce_scenario_tree(
        tree,
        target_leaves=3,
    ).reduced_tree


@pytest.fixture(scope="module")
def ph_result(small_problem, reduced_tree):
    return run_progressive_hedging(
        small_problem,
        reduced_tree,
        max_iterations=3,
        tolerance=0.03,
        route_rho=6.0,
        delivery_rho=0.06,
    )


def test_ph_produces_consensus_tables(small_problem, reduced_tree, ph_result):
    horizon = len(reduced_tree.days)
    decision_nodes = reduced_tree.nodes.index[
        reduced_tree.nodes["stage"] < horizon
    ]

    assert set(ph_result.consensus_routes.index) == set(decision_nodes)
    assert set(ph_result.consensus_deliveries.index) == set(decision_nodes)
    assert set(ph_result.consensus_deliveries.columns) == set(
        small_problem.cashpoints.index
    )


def test_ph_history_is_finite_and_bounded(ph_result):
    history = ph_result.convergence_history

    assert 1 <= len(history) <= 3
    assert np.isfinite(history["max_residual"]).all()
    assert (history["max_residual"] >= 0.0).all()
    assert np.isfinite(ph_result.expected_decomposed_cost)
    assert ph_result.expected_decomposed_cost >= 0.0


def test_ph_route_consensus_stays_in_unit_interval(ph_result):
    assert (ph_result.consensus_routes >= -1e-8).all().all()
    assert (ph_result.consensus_routes <= 1.0 + 1e-8).all().all()


def test_ph_benchmark_against_extensive_form(
    small_problem,
    reduced_tree,
    ph_result,
):
    table, reused = benchmark_progressive_hedging(
        small_problem,
        reduced_tree,
        ph_result=ph_result,
    )

    assert reused is ph_result
    assert set(table.index) == {
        "exact_extensive_form",
        "l1_progressive_hedging",
    }
    assert np.isclose(
        table.loc["exact_extensive_form", "nonanticipativity_residual"],
        0.0,
    )
    assert table.loc[
        "l1_progressive_hedging",
        "nonanticipativity_residual",
    ] >= 0.0
