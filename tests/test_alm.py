import numpy as np

from banking_optimization.alm import default_problem, solve


def test_default_problem_solves_and_balances():
    problem = default_problem()
    result = solve(problem)

    assert np.isclose(result.asset_allocation.sum(), problem.balance_sheet)
    assert np.isclose(result.funding_allocation.sum(), problem.balance_sheet)
    assert np.isfinite(result.expected_nii)


def test_default_solution_respects_bucket_bounds():
    problem = default_problem()
    result = solve(problem)

    for name, value in result.asset_allocation.items():
        assert value >= problem.assets.loc[name, "minimum"] - 1e-7
        assert value <= problem.assets.loc[name, "maximum"] + 1e-7

    for name, value in result.funding_allocation.items():
        assert value >= problem.funding.loc[name, "minimum"] - 1e-7
        assert value <= problem.funding.loc[name, "maximum"] + 1e-7


def test_default_solution_respects_risk_constraints():
    problem = default_problem()
    result = solve(problem)

    assert result.capital_ratio >= problem.minimum_capital_ratio - 1e-8
    assert result.liquidity_surplus >= -1e-8
    assert (
        abs(result.duration_gap_years)
        <= problem.maximum_duration_gap_years + 1e-8
    )
    assert abs(result.hedge_notional) <= problem.maximum_hedge_notional + 1e-8


def test_reported_expected_nii_matches_scenarios():
    problem = default_problem()
    result = solve(problem)

    recomputed = float(
        (result.scenario_nii * problem.scenario_probabilities).sum()
    )
    assert np.isclose(result.expected_nii, recomputed)
