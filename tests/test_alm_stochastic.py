import numpy as np

from banking_optimization.alm_stochastic import (
    compare_policies,
    default_stochastic_problem,
    solve_stochastic,
)


def test_stochastic_alm_balances_every_node():
    p = default_stochastic_problem()
    r = solve_stochastic(p, adaptive=True)

    assert np.allclose(r.asset_policy.sum(axis=1), p.balance_sheet)
    assert np.allclose(r.funding_policy.sum(axis=1), p.balance_sheet)


def test_stochastic_alm_respects_node_risk_constraints():
    p = default_stochastic_problem()
    r = solve_stochastic(p, adaptive=True)

    assert (
        r.node_capital_ratio >= p.minimum_capital_ratio - 1e-8
    ).all()
    assert (r.node_liquidity_surplus >= -1e-8).all()
    assert (
        r.node_duration_gap.abs()
        <= p.maximum_duration_gap_years + 1e-8
    ).all()
    assert (
        r.hedge_policy.abs()
        <= p.maximum_hedge_notional + 1e-8
    ).all()


def test_static_policy_freezes_balance_sheet_and_hedge():
    p = default_stochastic_problem()
    r = solve_stochastic(p, adaptive=False)

    root_assets = r.asset_policy.loc["root"].to_numpy()
    root_funding = r.funding_policy.loc["root"].to_numpy()
    root_hedge = r.hedge_policy.loc["root"]

    for node in r.asset_policy.index:
        assert np.allclose(r.asset_policy.loc[node].to_numpy(), root_assets)
        assert np.allclose(r.funding_policy.loc[node].to_numpy(), root_funding)
        assert np.isclose(r.hedge_policy.loc[node], root_hedge)


def test_adaptive_policy_is_not_worse_than_static_baseline():
    p = default_stochastic_problem()
    comparison = compare_policies(p)

    assert (
        comparison.loc["adaptive", "objective_value"]
        <= comparison.loc["static", "objective_value"] + 1e-8
    )
    assert (
        comparison.loc["adaptive", "objective_improvement_vs_static"]
        >= -1e-8
    )


def test_node_shortfall_matches_nii_targets():
    p = default_stochastic_problem()
    r = solve_stochastic(p, adaptive=True)

    for node in p.tree.index:
        period = int(p.tree.loc[node, "period"])
        target = p.nii_target_by_period[period]
        expected_shortfall = max(target - r.node_nii.loc[node], 0.0)
        assert np.isclose(
            r.node_shortfall.loc[node],
            expected_shortfall,
            atol=1e-7,
        )
