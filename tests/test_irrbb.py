import numpy as np

from banking_optimization.irrbb import (
    default_problem,
    scenario_eve_change,
    solve,
    unhedged_worst_loss,
)


def test_irrbb_hedge_reduces_worst_eve_loss():
    problem = default_problem()
    result = solve(problem)

    assert result.worst_eve_loss < unhedged_worst_loss(problem)


def test_irrbb_respects_gross_and_instrument_limits():
    problem = default_problem()
    result = solve(problem)

    assert result.gross_hedge_notional <= problem.gross_hedge_limit + 1e-8
    for hedge, notional in result.hedge_notionals.items():
        assert abs(notional) <= problem.maximum_hedge_notional.loc[hedge] + 1e-8


def test_irrbb_residual_pv01_identity_and_scenario_values():
    problem = default_problem()
    result = solve(problem)

    recomputed_residual = (
        problem.base_key_rate_pv01
        + problem.hedge_key_rate_pv01.T @ result.hedge_notionals
    )
    assert np.allclose(
        recomputed_residual.to_numpy(),
        result.residual_key_rate_pv01.to_numpy(),
    )

    recomputed_eve = scenario_eve_change(
        problem,
        result.residual_key_rate_pv01,
    )
    assert np.allclose(
        recomputed_eve.to_numpy(),
        result.scenario_eve_change.to_numpy(),
    )
