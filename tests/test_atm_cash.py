import numpy as np

from banking_optimization.atm_cash import default_problem, solve


def test_atm_inventory_balance():
    p = default_problem()
    r = solve(p)

    for atm in p.demand.index:
        prev = p.initial_cash.loc[atm]
        for day in p.demand.columns:
            expected = prev + r.deliveries.loc[atm, day] - p.demand.loc[atm, day]
            assert np.isclose(r.end_inventory.loc[atm, day], expected)
            prev = r.end_inventory.loc[atm, day]


def test_atm_safety_and_capacity():
    p = default_problem()
    r = solve(p)

    for atm in p.demand.index:
        assert (r.end_inventory.loc[atm] >= p.safety_stock.loc[atm] - 1e-7).all()
        assert (r.end_inventory.loc[atm] <= p.cash_capacity.loc[atm] + 1e-7).all()


def test_atm_daily_visit_capacity_and_delivery_link():
    p = default_problem()
    r = solve(p)

    assert (r.visits.sum(axis=0) <= p.maximum_visits_per_day).all()
    assert (
        r.deliveries.to_numpy()
        <= p.maximum_delivery * r.visits.to_numpy() + 1e-7
    ).all()
