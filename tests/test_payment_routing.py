from banking_optimization.payment_routing import default_problem, solve


def test_each_segment_is_routed_once():
    p = default_problem()
    r = solve(p)

    counts = r.assignments["segment_id"].value_counts()
    assert set(counts.index) == set(p.segments.index)
    assert (counts == 1).all()


def test_route_capacities_hold():
    p = default_problem()
    r = solve(p)

    for route, volume in r.route_volume.items():
        assert volume <= p.routes.loc[route, "capacity"] + 1e-7


def test_only_eligible_routes_are_used():
    p = default_problem()
    r = solve(p)

    for row in r.assignments.itertuples(index=False):
        assert p.eligibility.loc[row.segment_id, row.route] == 1
