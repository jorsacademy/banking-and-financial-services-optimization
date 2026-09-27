from dataclasses import replace

import numpy as np
import pytest

from cash_supply_chain.model import default_problem
from cash_supply_chain.multistage_cvar_irp import (
    compare_risk_attitudes,
    generate_binary_scenario_tree,
    solve_multistage_cvar_irp,
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


@pytest.fixture(scope="module")
def tree(small_problem):
    return generate_binary_scenario_tree(
        small_problem,
        days=(1, 2, 3),
        demand_sigma=0.10,
        seed=515,
    )


@pytest.fixture(scope="module")
def neutral(small_problem, tree):
    return solve_multistage_cvar_irp(
        small_problem,
        tree=tree,
        cvar_alpha=0.75,
        risk_aversion=0.0,
    )


@pytest.fixture(scope="module")
def risk_averse(small_problem, tree):
    return solve_multistage_cvar_irp(
        small_problem,
        tree=tree,
        cvar_alpha=0.75,
        risk_aversion=1.0,
    )


def test_binary_tree_probability_consistency(tree):
    nodes = tree.nodes

    assert np.isclose(nodes.loc["root", "probability"], 1.0)

    for parent in nodes.index:
        children = nodes.index[nodes["parent"] == parent].tolist()
        if children:
            assert np.isclose(
                nodes.loc[children, "probability"].sum(),
                nodes.loc[parent, "probability"],
            )

    leaves = nodes.index[nodes["stage"] == len(tree.days)]
    assert np.isclose(nodes.loc[leaves, "probability"].sum(), 1.0)


def test_multistage_policy_is_node_based_not_path_based(tree, neutral):
    decision_nodes = list(
        tree.nodes.index[tree.nodes["stage"] < len(tree.days)]
    )

    assert list(neutral.delivery_policy.index) == decision_nodes

    # There is one root decision shared by every terminal path.
    assert "root" in neutral.delivery_policy.index
    assert neutral.delivery_policy.index.tolist().count("root") == 1

    # Stage-1 decisions are exactly the two information states L and H,
    # not one separate decision per terminal leaf.
    stage_one = list(tree.nodes.index[tree.nodes["stage"] == 1])
    assert set(stage_one) == {"L", "H"}
    assert set(stage_one).issubset(neutral.delivery_policy.index)


def test_multistage_inventory_balance(small_problem, tree, neutral):
    for node in tree.demand.index:
        parent = str(tree.nodes.loc[node, "parent"])

        for cashpoint in small_problem.cashpoints.index:
            if parent == "root":
                opening = small_problem.cashpoints.loc[
                    cashpoint, "initial_cash"
                ]
            else:
                opening = neutral.inventory.loc[parent, cashpoint]

            expected = (
                opening
                + neutral.delivery_policy.loc[parent, cashpoint]
                + neutral.cashout.loc[node, cashpoint]
                - tree.demand.loc[node, cashpoint]
            )

            assert np.isclose(
                neutral.inventory.loc[node, cashpoint],
                expected,
                atol=1e-6,
            )


def test_multistage_physical_constraints(small_problem, tree, neutral):
    catalog_days = neutral.route_policy.groupby("node").size()

    for node in neutral.delivery_policy.index:
        stage = int(tree.nodes.loc[node, "stage"])
        day = tree.days[stage]

        assert (
            neutral.delivery_policy.loc[node].sum()
            <= small_problem.vault_dispatch_limit.loc[day] + 1e-7
        )
        assert catalog_days.get(node, 0) <= small_problem.maximum_vehicles_per_day

    if not neutral.route_policy.empty:
        assert (
            neutral.route_policy["load"]
            <= small_problem.vehicle_capacity + 1e-7
        ).all()

    assert (neutral.inventory >= -1e-8).all().all()
    assert (neutral.cashout >= -1e-8).all().all()


def test_leaf_risk_metrics_are_ordered(neutral):
    assert neutral.expected_cost <= neutral.cvar_cost + 1e-7
    assert neutral.cvar_cost <= neutral.worst_leaf_cost + 1e-7
    assert neutral.var_threshold <= neutral.cvar_cost + 1e-7


def test_cvar_policy_tradeoff(small_problem, tree, neutral, risk_averse):
    # Risk-neutral policy minimizes expected cost.
    assert neutral.expected_cost <= risk_averse.expected_cost + 1e-6

    # Adding positive CVaR weight cannot choose a policy with a worse CVaR
    # when the risk-neutral solution remains feasible.
    assert risk_averse.cvar_cost <= neutral.cvar_cost + 1e-6

    comparison, _ = compare_risk_attitudes(
        small_problem,
        tree=tree,
        cvar_alpha=0.75,
        risk_aversion=1.0,
    )

    assert set(comparison.index) == {"risk_neutral", "risk_averse"}
    assert (
        comparison.loc["risk_averse", "cvar_improvement_vs_neutral"]
        >= -1e-6
    )
