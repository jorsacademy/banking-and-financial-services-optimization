"""Exact small-network CIT routing for optimized replenishment visits.

For each day, visited cashpoints are partitioned into capacity-feasible routes.
Within each route, the stop order is solved exactly by enumeration. This is
intended for small educational instances; large networks require scalable VRP
algorithms.
"""

from __future__ import annotations

from itertools import combinations, permutations
from math import hypot

import pandas as pd

from .model import CashSupplyChainProblem, CashSupplyChainResult


DEPOT = "DEPOT"
DEPOT_COORD = (0.0, 0.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _best_tour(
    stops: tuple[str, ...],
    problem: CashSupplyChainProblem,
) -> tuple[float, tuple[str, ...]]:
    if not stops:
        return 0.0, ()

    coords = {
        stop: (
            float(problem.cashpoints.loc[stop, "x_coord"]),
            float(problem.cashpoints.loc[stop, "y_coord"]),
        )
        for stop in stops
    }

    best_cost = float("inf")
    best_order: tuple[str, ...] = ()

    for order in permutations(stops):
        cost = _distance(DEPOT_COORD, coords[order[0]])
        for left, right in zip(order, order[1:]):
            cost += _distance(coords[left], coords[right])
        cost += _distance(coords[order[-1]], DEPOT_COORD)

        if cost < best_cost:
            best_cost = cost
            best_order = order

    return best_cost, best_order


def route_day(
    day: int,
    result: CashSupplyChainResult,
    problem: CashSupplyChainProblem,
) -> pd.DataFrame:
    """Create minimum-distance routes for one day's planned CIT visits."""
    active = [
        cashpoint
        for cashpoint in problem.cashpoints.index
        if result.visits.loc[cashpoint, day] > 0.5
        and result.deliveries.loc[cashpoint, day] > 1e-8
    ]

    if not active:
        return pd.DataFrame(
            columns=["day", "vehicle", "sequence", "load", "distance"]
        )

    max_routes = max(1, int(round(result.vehicles.loc[day])))
    n = len(active)
    full_mask = (1 << n) - 1

    feasible_routes: dict[int, tuple[float, tuple[str, ...], float]] = {}

    for size in range(1, min(problem.stops_per_vehicle, n) + 1):
        for idxs in combinations(range(n), size):
            stops = tuple(active[i] for i in idxs)
            load = float(
                sum(result.deliveries.loc[stop, day] for stop in stops)
            )
            if load > problem.vehicle_capacity + 1e-8:
                continue

            mask = sum(1 << i for i in idxs)
            distance, order = _best_tour(stops, problem)
            feasible_routes[mask] = (distance, order, load)

    # Dynamic programming over partitions of the active-stop bitmask.
    # dp[(mask, k)] = (distance, list_of_route_masks)
    dp: dict[tuple[int, int], tuple[float, list[int]]] = {
        (0, 0): (0.0, [])
    }

    for k in range(max_routes):
        states = [
            (mask, value)
            for (mask, used), value in list(dp.items())
            if used == k
        ]
        for mask, (current_cost, route_masks) in states:
            remaining = full_mask ^ mask
            for route_mask, (route_cost, _order, _load) in feasible_routes.items():
                if route_mask & mask:
                    continue
                if route_mask & remaining != route_mask:
                    continue

                new_mask = mask | route_mask
                key = (new_mask, k + 1)
                candidate = current_cost + route_cost

                if key not in dp or candidate < dp[key][0]:
                    dp[key] = (candidate, route_masks + [route_mask])

    candidates = [
        (value[0], used, value[1])
        for (mask, used), value in dp.items()
        if mask == full_mask and used <= max_routes
    ]
    if not candidates:
        raise RuntimeError(
            f"no feasible route partition for day {day} "
            f"with {max_routes} vehicles"
        )

    _total_distance, _used, route_masks = min(candidates, key=lambda x: x[0])

    rows = []
    for vehicle, route_mask in enumerate(route_masks, start=1):
        distance, order, load = feasible_routes[route_mask]
        rows.append(
            {
                "day": day,
                "vehicle": vehicle,
                "sequence": " -> ".join((DEPOT, *order, DEPOT)),
                "load": load,
                "distance": distance,
            }
        )

    return pd.DataFrame(rows)


def route_plan(
    result: CashSupplyChainResult,
    problem: CashSupplyChainProblem,
) -> pd.DataFrame:
    """Route all days in a solved cash-supply-chain plan."""
    frames = [
        route_day(day, result, problem)
        for day in problem.forecast_net_withdrawal.columns
    ]
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame(
            columns=["day", "vehicle", "sequence", "load", "distance"]
        )
    return pd.concat(nonempty, ignore_index=True)
