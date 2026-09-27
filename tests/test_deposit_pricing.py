from banking_optimization.deposit_pricing import default_problem, solve


def test_deposit_pricing_hits_retention_target():
    p = default_problem()
    r = solve(p)
    assert (
        r.expected_retained_balance
        >= p.target_expected_retained_balance - 1e-7
    )


def test_exactly_one_offer_per_segment():
    p = default_problem()
    r = solve(p)
    counts = r.selected_offers["segment"].value_counts()
    assert set(counts.index) == set(p.segments.index)
    assert (counts == 1).all()


def test_weighted_rate_is_within_offer_range():
    p = default_problem()
    r = solve(p)
    assert min(p.rate_options) <= r.weighted_average_rate <= max(p.rate_options)
