from cash_supply_chain.model import default_problem, solve
from cash_supply_chain.routing import route_plan


def test_routes_cover_all_planned_delivery_visits():
    p = default_problem()
    r = solve(p)
    routes = route_plan(r, p)

    for day in p.forecast_net_withdrawal.columns:
        expected = {
            cp
            for cp in p.cashpoints.index
            if r.visits.loc[cp, day] > 0.5
            and r.deliveries.loc[cp, day] > 1e-8
        }

        day_routes = routes.loc[routes["day"] == day, "sequence"].tolist()
        routed = {
            cp
            for cp in p.cashpoints.index
            if any(cp in sequence for sequence in day_routes)
        }
        assert routed == expected


def test_routes_respect_vehicle_capacity_and_count():
    p = default_problem()
    r = solve(p)
    routes = route_plan(r, p)

    assert (routes["load"] <= p.vehicle_capacity + 1e-7).all()

    route_counts = routes.groupby("day").size()
    for day, count in route_counts.items():
        assert count <= r.vehicles.loc[day] + 1e-7


def test_routes_have_positive_distance():
    p = default_problem()
    r = solve(p)
    routes = route_plan(r, p)

    assert (routes["distance"] > 0).all()
