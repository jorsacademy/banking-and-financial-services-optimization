from banking_optimization.collections import default_problem, solve


def test_collections_budget_and_capacity():
    p = default_problem()
    r = solve(p)

    assert r.total_cost <= p.treatment_budget + 1e-7
    counts = r.assignments["action"].value_counts()
    for action, cap in p.capacity.items():
        assert counts.get(action, 0) <= cap


def test_collections_assigns_at_most_one_action_per_account():
    r = solve(default_problem())
    counts = r.assignments["account_id"].value_counts()
    assert (counts <= 1).all()


def test_collections_creates_positive_recovery():
    r = solve(default_problem())
    assert r.expected_incremental_recovery > 0
