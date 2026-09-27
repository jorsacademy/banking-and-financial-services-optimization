from dataclasses import replace

import numpy as np
import pytest

from cash_supply_chain.model import default_problem
from cash_supply_chain.multistage_cvar_irp import (
    generate_binary_scenario_tree,
    solve_multistage_cvar_irp,
)
from cash_supply_chain.scenario_reduction import (
    leaf_path_vectors,
    reduce_scenario_tree,
)


@pytest.fixture(scope="module")
def small_problem():
    p = default_problem()
    cashpoints = ["ATM_A", "ATM_B", "BRANCH_C"]
    days = [1, 2, 3]

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
def full_tree(small_problem):
    return generate_binary_scenario_tree(
        small_problem,
        days=(1, 2, 3),
        demand_sigma=0.10,
        seed=700,
    )


def test_leaf_vectors_and_probabilities(full_tree):
    vectors, probabilities = leaf_path_vectors(full_tree)

    assert len(vectors) == 8
    assert vectors.shape[1] == 3 * 3
    assert np.isclose(probabilities.sum(), 1.0)


def test_reduction_preserves_probability_mass_and_tree_consistency(full_tree):
    reduced = reduce_scenario_tree(full_tree, target_leaves=3)

    assert len(reduced.selected_leaves) == 3
    assert np.isclose(reduced.reduced_probabilities.sum(), 1.0)
    assert reduced.distortion >= 0.0

    tree = reduced.reduced_tree
    assert np.isclose(tree.nodes.loc["root", "probability"], 1.0)

    for parent in tree.nodes.index:
        children = tree.nodes.index[tree.nodes["parent"] == parent].tolist()
        if children:
            assert np.isclose(
                tree.nodes.loc[children, "probability"].sum(),
                tree.nodes.loc[parent, "probability"],
            )


def test_every_original_leaf_maps_to_selected_representative(full_tree):
    reduced = reduce_scenario_tree(full_tree, target_leaves=3)

    assert set(reduced.assignment.index) == set(
        reduced.original_probabilities.index
    )
    assert set(reduced.assignment.unique()).issubset(
        set(reduced.selected_leaves)
    )


def test_reduced_tree_remains_solvable(small_problem, full_tree):
    reduced = reduce_scenario_tree(full_tree, target_leaves=3)

    result = solve_multistage_cvar_irp(
        small_problem,
        tree=reduced.reduced_tree,
        cvar_alpha=0.75,
        risk_aversion=0.0,
    )

    assert np.isfinite(result.expected_cost)
    assert result.expected_cost >= 0.0
