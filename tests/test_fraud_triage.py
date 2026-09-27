from banking_optimization.fraud_triage import default_problem, solve


def test_fraud_triage_respects_capacity():
    p = default_problem()
    r = solve(p)

    assert r.minutes_used <= p.investigator_minutes + 1e-7
    assert len(r.selected_alerts) <= p.maximum_alerts


def test_fraud_triage_has_positive_net_value():
    r = solve(default_problem())
    assert len(r.selected_alerts) > 0
    assert r.net_value > 0
