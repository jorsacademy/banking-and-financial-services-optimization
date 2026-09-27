from banking_optimization.loan_pricing import default_problem, solve


def test_loan_pricing_respects_portfolio_limits():
    p = default_problem()
    r = solve(p)

    assert r.expected_exposure <= p.expected_exposure_budget + 1e-7
    assert r.expected_rwa <= p.expected_rwa_budget + 1e-7
    assert r.expected_loss <= p.expected_loss_budget + 1e-7


def test_at_most_one_offer_per_customer():
    r = solve(default_problem())
    counts = r.selected_offers.groupby("customer_id").size()
    assert (counts <= 1).all()


def test_loan_pricing_selects_profitable_offers():
    r = solve(default_problem())
    assert len(r.selected_offers) > 0
    assert r.expected_profit > 0
