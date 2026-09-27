import numpy as np

from cash_supply_chain.joint_irp import (
    compare_staged_and_joint,
    default_problem,
    generate_route_catalog,
    solve_joint_irp,
)


def test_route_catalog_respects_stop_limit():
    p = default_problem()
    catalog = generate_route_catalog(p)

    assert len(catalog) > 0
    assert (catalog["stop_count"] <= p.stops_per_vehicle).all()
    assert (catalog["distance"] > 0).all()


def test_joint_irp_inventory_balance():
    p = default_problem()
    r = solve_joint_irp(p)

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


def test_joint_irp_route_and_vault_limits():
    p = default_problem()
    r = solve_joint_irp(p)

    for day in p.forecast_net_withdrawal.columns:
        day_routes = r.selected_routes[r.selected_routes["day"] == day]

        assert len(day_routes) <= p.maximum_vehicles_per_day
        assert (day_routes["load"] <= p.vehicle_capacity + 1e-7).all()
        assert r.deliveries[day].sum() <= p.vault_dispatch_limit.loc[day] + 1e-7

        for cluster, limit in p.cluster_visit_limit.items():
            members = p.cashpoints.index[p.cashpoints["cluster"] == cluster]
            assert r.visits.loc[members, day].sum() <= limit + 1e-7


def test_joint_irp_cashpoint_delivery_limits():
    p = default_problem()
    r = solve_joint_irp(p)

    for cashpoint in p.cashpoints.index:
        max_delivery = p.cashpoints.loc[cashpoint, "maximum_delivery"]
        assert (
            r.deliveries.loc[cashpoint].to_numpy()
            <= max_delivery * r.visits.loc[cashpoint].to_numpy() + 1e-7
        ).all()


def test_joint_irp_cost_breakdown_matches_objective():
    r = solve_joint_irp(default_problem())

    assert np.isclose(
        float(r.cost_breakdown.sum()),
        r.total_cost,
        atol=1e-6,
    )


def test_joint_irp_not_worse_than_staged_full_cost():
    comparison = compare_staged_and_joint(default_problem())

    assert (
        comparison.loc["joint_irp", "integrated_total_cost"]
        <= comparison.loc["staged", "integrated_total_cost"] + 1e-6
    )
    assert (
        comparison.loc["joint_irp", "cost_improvement_vs_staged"]
        >= -1e-6
    )
