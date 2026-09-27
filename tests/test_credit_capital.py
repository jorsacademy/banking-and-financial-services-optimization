from banking_optimization.credit_capital import (
    default_problem,
    economics,
    solve,
)


def test_credit_solution_respects_global_limits():
    problem = default_problem()
    result = solve(problem)

    assert result.total_exposure <= problem.lending_budget + 1e-7
    assert result.total_rwa <= problem.rwa_budget + 1e-7
    assert result.total_expected_loss <= problem.expected_loss_budget + 1e-7


def test_credit_solution_respects_sector_caps():
    problem = default_problem()
    result = solve(problem)

    assert (result.sector_exposure <= problem.sector_exposure_cap + 1e-7).all()


def test_selected_profit_matches_economics_table():
    problem = default_problem()
    table = economics(problem)
    result = solve(problem)

    expected = float(table.loc[result.selected.index, "expected_profit"].sum())
    assert abs(expected - result.total_expected_profit) <= 1e-8


def test_solution_selects_at_least_one_opportunity():
    result = solve(default_problem())
    assert len(result.selected) > 0
