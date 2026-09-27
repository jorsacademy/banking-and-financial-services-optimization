import numpy as np

from cash_supply_chain.model import default_problem, solve
from cash_supply_chain.simulation import fleet_sensitivity, simulate_plan


def test_cash_supply_chain_inventory_balance():
    p = default_problem()
    r = solve(p)

    for cashpoint in p.cashpoints.index:
        previous = p.cashpoints.loc[cashpoint, "initial_cash"]
        for day in p.forecast_net_withdrawal.columns:
            expected = (
                previous
                + r.deliveries.loc[cashpoint, day]
                - p.forecast_net_withdrawal.loc[cashpoint, day]
            )
            assert np.isclose(r.end_inventory.loc[cashpoint, day], expected)
            previous = r.end_inventory.loc[cashpoint, day]


def test_cash_supply_chain_operational_limits():
    p = default_problem()
    r = solve(p)

    for day in p.forecast_net_withdrawal.columns:
        assert r.deliveries[day].sum() <= p.vault_dispatch_limit.loc[day] + 1e-7
        assert (
            r.deliveries[day].sum()
            <= p.vehicle_capacity * r.vehicles.loc[day] + 1e-7
        )
        assert r.visits[day].sum() <= p.stops_per_vehicle * r.vehicles.loc[day] + 1e-7
        assert r.vehicles.loc[day] <= p.maximum_vehicles_per_day + 1e-7

        for cluster, limit in p.cluster_visit_limit.items():
            members = p.cashpoints.index[p.cashpoints["cluster"] == cluster]
            assert r.visits.loc[members, day].sum() <= limit + 1e-7


def test_delivery_requires_visit_and_respects_cashpoint_limits():
    p = default_problem()
    r = solve(p)

    for cashpoint in p.cashpoints.index:
        max_delivery = p.cashpoints.loc[cashpoint, "maximum_delivery"]
        assert (
            r.deliveries.loc[cashpoint].to_numpy()
            <= max_delivery * r.visits.loc[cashpoint].to_numpy() + 1e-7
        ).all()


def test_simulation_is_reproducible_and_bounded():
    p = default_problem()
    r = solve(p)

    a = simulate_plan(r, p, replications=20, seed=11)
    b = simulate_plan(r, p, replications=20, seed=11)

    assert a.equals(b)
    assert (a["cashout_volume"] >= 0).all()
    assert ((a["service_event_rate"] >= 0) & (a["service_event_rate"] <= 1)).all()


def test_fleet_sensitivity_returns_all_requested_levels():
    table = fleet_sensitivity(fleet_sizes=(1, 2, 3))
    assert list(table["maximum_vehicles_per_day"]) == [1, 2, 3]
    assert table["feasible"].dtype == bool
