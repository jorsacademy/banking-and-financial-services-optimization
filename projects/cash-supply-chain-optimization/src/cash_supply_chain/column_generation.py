"""Route-column generation and a small Ryan-Foster branch-and-price demonstrator.

Scope
-----
This module scales the CIT routing layer for a *fixed daily replenishment
vector*. Cash quantities are supplied by an upstream inventory/replenishment
model. The routing master then covers every cashpoint exactly once while
respecting vehicle capacity, stop limits, and fleet count.

The restricted master does not pre-enumerate every feasible route. New routes
are produced by an exact pricing oracle using LP dual values.

For small educational instances the pricing oracle enumerates feasible
cashpoint subsets and solves the stop ordering exactly. In a large deployment,
this oracle is the component that would be replaced by an ESPPRC/RCSP pricing
algorithm, labeling method, or specialized shortest-path solver.

The branch-and-price routine uses Ryan-Foster pair branching. It is exact for
the fixed-delivery set-partitioning routing master when all branch nodes are
fully processed and pricing is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .model import CashSupplyChainProblem, CashSupplyChainResult, default_problem
from .routing import _best_tour


@dataclass(frozen=True)
class RouteColumn:
    route_id: str
    stops: tuple[str, ...]
    sequence: str
    load: float
    distance: float
    cost: float


@dataclass(frozen=True)
class ColumnGenerationResult:
    columns: pd.DataFrame
    selected_routes: pd.DataFrame
    lp_objective: float
    integer_objective: float
    iterations: int
    generated_columns: int
    artificial_mass: float
    pricing_history: pd.DataFrame
    required_delivery: pd.Series

    def to_dict(self) -> dict:
        return {
            "lp_objective": round(self.lp_objective, 6),
            "integer_objective": round(self.integer_objective, 6),
            "iterations": self.iterations,
            "generated_columns": self.generated_columns,
            "artificial_mass": round(self.artificial_mass, 6),
            "required_delivery": self.required_delivery.round(6).to_dict(),
            "selected_routes": self.selected_routes.round(
                {"load": 6, "distance": 6, "cost": 6}
            ).to_dict(orient="records"),
        }


@dataclass(frozen=True)
class BranchAndPriceResult:
    selected_routes: pd.DataFrame
    objective_value: float
    root_lp_bound: float
    nodes_processed: int
    nodes_pruned: int
    columns_generated: int
    incumbent_updates: int
    exact: bool
    search_log: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "objective_value": round(self.objective_value, 6),
            "root_lp_bound": round(self.root_lp_bound, 6),
            "nodes_processed": self.nodes_processed,
            "nodes_pruned": self.nodes_pruned,
            "columns_generated": self.columns_generated,
            "incumbent_updates": self.incumbent_updates,
            "exact": self.exact,
            "selected_routes": self.selected_routes.round(
                {"load": 6, "distance": 6, "cost": 6}
            ).to_dict(orient="records"),
        }


@dataclass(frozen=True)
class _MasterLPResult:
    objective: float
    route_values: pd.Series
    artificial_values: pd.Series
    coverage_duals: pd.Series
    fleet_dual: float


@dataclass(frozen=True)
class _CGNodeResult:
    columns: dict[tuple[str, ...], RouteColumn]
    lp: _MasterLPResult
    pricing_history: pd.DataFrame
    iterations: int
    added_columns: int


def _route_key(stops: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(stops))


def _route_id(stops: tuple[str, ...]) -> str:
    return "CG[" + "|".join(_route_key(stops)) + "]"


def _make_route_column(
    problem: CashSupplyChainProblem,
    required_delivery: pd.Series,
    stops: tuple[str, ...],
) -> RouteColumn:
    stops = _route_key(stops)
    distance, order = _best_tour(stops, problem)
    load = float(required_delivery.loc[list(stops)].sum())
    visit_cost = float(
        problem.cashpoints.loc[list(stops), "visit_cost"].sum()
    )
    cost = (
        problem.vehicle_fixed_cost
        + visit_cost
        + problem.distance_cost_per_unit * distance
    )
    return RouteColumn(
        route_id=_route_id(stops),
        stops=stops,
        sequence=" -> ".join(("DEPOT", *order, "DEPOT")),
        load=load,
        distance=float(distance),
        cost=float(cost),
    )


def delivery_requirements_from_plan(
    plan: CashSupplyChainResult,
    day: int,
    tolerance: float = 1e-8,
) -> pd.Series:
    """Extract positive fixed delivery requirements for one day."""
    if day not in plan.deliveries.columns:
        raise ValueError("day is not present in the replenishment plan")

    required = plan.deliveries[day].astype(float).copy()
    required[required.abs() <= tolerance] = 0.0
    return required[required > 0.0]


def _validate_requirements(
    problem: CashSupplyChainProblem,
    required_delivery: pd.Series,
) -> pd.Series:
    required = required_delivery.astype(float).copy()
    required = required[required > 1e-9]

    unknown = set(required.index) - set(problem.cashpoints.index)
    if unknown:
        raise ValueError(f"unknown cashpoints in delivery requirements: {unknown}")
    if (required < 0).any():
        raise ValueError("delivery requirements must be nonnegative")
    if (
        required
        > problem.cashpoints.loc[required.index, "maximum_delivery"] + 1e-8
    ).any():
        raise ValueError(
            "a fixed delivery requirement exceeds cashpoint maximum delivery"
        )
    if (required > problem.vehicle_capacity + 1e-8).any():
        raise ValueError(
            "a cashpoint fixed delivery exceeds vehicle capacity"
        )

    return required


def _branch_feasible(
    stops: tuple[str, ...],
    together_pairs: frozenset[tuple[str, str]],
    separate_pairs: frozenset[tuple[str, str]],
) -> bool:
    stop_set = set(stops)

    for left, right in together_pairs:
        if (left in stop_set) != (right in stop_set):
            return False

    for left, right in separate_pairs:
        if left in stop_set and right in stop_set:
            return False

    return True


def _initial_columns(
    problem: CashSupplyChainProblem,
    required_delivery: pd.Series,
    together_pairs: frozenset[tuple[str, str]] = frozenset(),
    separate_pairs: frozenset[tuple[str, str]] = frozenset(),
) -> dict[tuple[str, ...], RouteColumn]:
    """Seed with every branch-feasible singleton and a greedy partition."""
    active = list(required_delivery.index)
    columns: dict[tuple[str, ...], RouteColumn] = {}

    for cashpoint in active:
        stops = (cashpoint,)
        if _branch_feasible(stops, together_pairs, separate_pairs):
            column = _make_route_column(
                problem,
                required_delivery,
                stops,
            )
            columns[_route_key(stops)] = column

    # Greedy geographically ordered partition gives the master useful
    # multi-stop columns before dual pricing begins.
    ordered = sorted(
        active,
        key=lambda cp: (
            float(problem.cashpoints.loc[cp, "x_coord"]) ** 2
            + float(problem.cashpoints.loc[cp, "y_coord"]) ** 2
        ),
    )

    current: list[str] = []
    current_load = 0.0

    def flush() -> None:
        nonlocal current, current_load
        if not current:
            return
        stops = tuple(current)
        if _branch_feasible(stops, together_pairs, separate_pairs):
            column = _make_route_column(
                problem,
                required_delivery,
                stops,
            )
            columns[_route_key(stops)] = column
        current = []
        current_load = 0.0

    for cashpoint in ordered:
        load = float(required_delivery.loc[cashpoint])
        if (
            current
            and (
                len(current) >= problem.stops_per_vehicle
                or current_load + load > problem.vehicle_capacity + 1e-8
            )
        ):
            flush()
        current.append(cashpoint)
        current_load += load
    flush()

    return columns


def _filter_columns(
    columns: dict[tuple[str, ...], RouteColumn],
    together_pairs: frozenset[tuple[str, str]],
    separate_pairs: frozenset[tuple[str, str]],
) -> dict[tuple[str, ...], RouteColumn]:
    return {
        key: col
        for key, col in columns.items()
        if _branch_feasible(
            col.stops,
            together_pairs,
            separate_pairs,
        )
    }


def _solve_master_lp(
    problem: CashSupplyChainProblem,
    required_delivery: pd.Series,
    columns: dict[tuple[str, ...], RouteColumn],
    artificial_cost: float = 10_000.0,
) -> _MasterLPResult:
    """Solve restricted set-partitioning LP with artificial cover variables."""
    active = list(required_delivery.index)
    route_keys = list(columns)
    n_routes = len(route_keys)
    n_customers = len(active)

    # lambda routes | artificial cover variables
    n_vars = n_routes + n_customers
    c = np.zeros(n_vars)

    for k, key in enumerate(route_keys):
        c[k] = columns[key].cost
    c[n_routes:] = artificial_cost

    a_eq = np.zeros((n_customers, n_vars))
    for i, cashpoint in enumerate(active):
        for k, key in enumerate(route_keys):
            if cashpoint in columns[key].stops:
                a_eq[i, k] = 1.0
        a_eq[i, n_routes + i] = 1.0

    b_eq = np.ones(n_customers)

    a_ub = np.zeros((1, n_vars))
    a_ub[0, :n_routes] = 1.0
    b_ub = np.array([float(problem.maximum_vehicles_per_day)])

    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * n_vars,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            f"restricted routing master LP failed: {result.message}"
        )

    route_values = pd.Series(
        result.x[:n_routes],
        index=route_keys,
        dtype=float,
    )
    artificial_values = pd.Series(
        result.x[n_routes:],
        index=active,
        dtype=float,
    )
    coverage_duals = pd.Series(
        result.eqlin.marginals,
        index=active,
        dtype=float,
    )
    fleet_dual = float(result.ineqlin.marginals[0])

    return _MasterLPResult(
        objective=float(result.fun),
        route_values=route_values,
        artificial_values=artificial_values,
        coverage_duals=coverage_duals,
        fleet_dual=fleet_dual,
    )


def _price_routes(
    problem: CashSupplyChainProblem,
    required_delivery: pd.Series,
    coverage_duals: pd.Series,
    fleet_dual: float,
    existing_keys: set[tuple[str, ...]],
    together_pairs: frozenset[tuple[str, str]],
    separate_pairs: frozenset[tuple[str, str]],
    max_new_columns: int = 6,
    reduced_cost_tolerance: float = 1e-7,
) -> list[tuple[float, RouteColumn]]:
    """Exact small-instance pricing oracle.

    Pricing searches feasible cashpoint subsets without pre-loading those
    columns into the master. The reduced cost of a route is:

        route_cost - sum(coverage duals) - fleet_dual.
    """
    active = list(required_delivery.index)
    candidates: list[tuple[float, RouteColumn]] = []

    max_size = min(problem.stops_per_vehicle, len(active))
    for size in range(1, max_size + 1):
        for subset in combinations(active, size):
            key = _route_key(subset)
            if key in existing_keys:
                continue
            if not _branch_feasible(
                key,
                together_pairs,
                separate_pairs,
            ):
                continue

            load = float(required_delivery.loc[list(key)].sum())
            if load > problem.vehicle_capacity + 1e-8:
                continue

            column = _make_route_column(
                problem,
                required_delivery,
                key,
            )
            reduced_cost = (
                column.cost
                - float(coverage_duals.loc[list(key)].sum())
                - fleet_dual
            )

            if reduced_cost < -reduced_cost_tolerance:
                candidates.append((float(reduced_cost), column))

    candidates.sort(key=lambda item: item[0])
    return candidates[:max_new_columns]


def _run_column_generation_node(
    problem: CashSupplyChainProblem,
    required_delivery: pd.Series,
    seed_columns: dict[tuple[str, ...], RouteColumn] | None = None,
    together_pairs: frozenset[tuple[str, str]] = frozenset(),
    separate_pairs: frozenset[tuple[str, str]] = frozenset(),
    max_iterations: int = 50,
    max_new_columns_per_iteration: int = 6,
    artificial_cost: float = 10_000.0,
) -> _CGNodeResult:
    columns = dict(seed_columns or {})
    columns = _filter_columns(
        columns,
        together_pairs,
        separate_pairs,
    )

    if not columns:
        columns.update(
            _initial_columns(
                problem,
                required_delivery,
                together_pairs,
                separate_pairs,
            )
        )

    # Always add branch-feasible seed columns not already inherited.
    columns.update(
        {
            key: value
            for key, value in _initial_columns(
                problem,
                required_delivery,
                together_pairs,
                separate_pairs,
            ).items()
            if key not in columns
        }
    )

    history = []
    total_added = 0
    lp = None

    for iteration in range(1, max_iterations + 1):
        lp = _solve_master_lp(
            problem,
            required_delivery,
            columns,
            artificial_cost=artificial_cost,
        )

        priced = _price_routes(
            problem,
            required_delivery,
            lp.coverage_duals,
            lp.fleet_dual,
            existing_keys=set(columns),
            together_pairs=together_pairs,
            separate_pairs=separate_pairs,
            max_new_columns=max_new_columns_per_iteration,
        )

        best_reduced_cost = (
            priced[0][0] if priced else 0.0
        )

        history.append(
            {
                "iteration": iteration,
                "lp_objective": lp.objective,
                "artificial_mass": float(
                    lp.artificial_values.sum()
                ),
                "columns": len(columns),
                "new_columns": len(priced),
                "best_reduced_cost": best_reduced_cost,
            }
        )

        if not priced:
            return _CGNodeResult(
                columns=columns,
                lp=lp,
                pricing_history=pd.DataFrame(history),
                iterations=iteration,
                added_columns=total_added,
            )

        for reduced_cost, column in priced:
            key = _route_key(column.stops)
            columns[key] = column
            total_added += 1

    if lp is None:
        raise RuntimeError("column generation did not execute")

    return _CGNodeResult(
        columns=columns,
        lp=lp,
        pricing_history=pd.DataFrame(history),
        iterations=max_iterations,
        added_columns=total_added,
    )


def _solve_restricted_integer_master(
    problem: CashSupplyChainProblem,
    required_delivery: pd.Series,
    columns: dict[tuple[str, ...], RouteColumn],
) -> tuple[float, pd.Series]:
    active = list(required_delivery.index)
    route_keys = list(columns)
    n_routes = len(route_keys)

    if not active:
        return 0.0, pd.Series(dtype=float)

    c = np.array(
        [columns[key].cost for key in route_keys],
        dtype=float,
    )

    a_eq = np.zeros((len(active), n_routes))
    for i, cashpoint in enumerate(active):
        for k, key in enumerate(route_keys):
            if cashpoint in columns[key].stops:
                a_eq[i, k] = 1.0

    constraints = [
        LinearConstraint(
            a_eq,
            lb=np.ones(len(active)),
            ub=np.ones(len(active)),
        ),
        LinearConstraint(
            np.ones((1, n_routes)),
            lb=np.array([-np.inf]),
            ub=np.array([float(problem.maximum_vehicles_per_day)]),
        ),
    ]

    result = milp(
        c=c,
        integrality=np.ones(n_routes, dtype=int),
        bounds=Bounds(
            np.zeros(n_routes),
            np.ones(n_routes),
        ),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(
            f"restricted integer routing master failed: {result.message}"
        )

    return (
        float(result.fun),
        pd.Series(
            result.x,
            index=route_keys,
            dtype=float,
        ),
    )


def _selected_route_frame(
    columns: dict[tuple[str, ...], RouteColumn],
    values: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for key, value in values.items():
        if value <= threshold:
            continue
        col = columns[key]
        rows.append(
            {
                "route_id": col.route_id,
                "sequence": col.sequence,
                "stops": "|".join(col.stops),
                "load": col.load,
                "distance": col.distance,
                "cost": col.cost,
                "value": float(value),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "route_id",
            "sequence",
            "stops",
            "load",
            "distance",
            "cost",
            "value",
        ],
    )


def _columns_frame(
    columns: dict[tuple[str, ...], RouteColumn],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "route_id": col.route_id,
                "stops": "|".join(col.stops),
                "sequence": col.sequence,
                "load": col.load,
                "distance": col.distance,
                "cost": col.cost,
            }
            for col in columns.values()
        ]
    ).sort_values("route_id").reset_index(drop=True)


def solve_route_column_generation(
    problem: CashSupplyChainProblem,
    required_delivery: pd.Series,
    max_iterations: int = 50,
    max_new_columns_per_iteration: int = 6,
) -> ColumnGenerationResult:
    """Solve fixed-delivery CIT routing by column generation + final MILP."""
    required = _validate_requirements(
        problem,
        required_delivery,
    )
    if required.empty:
        empty = pd.DataFrame(
            columns=[
                "route_id",
                "sequence",
                "stops",
                "load",
                "distance",
                "cost",
                "value",
            ]
        )
        return ColumnGenerationResult(
            columns=empty.copy(),
            selected_routes=empty,
            lp_objective=0.0,
            integer_objective=0.0,
            iterations=0,
            generated_columns=0,
            artificial_mass=0.0,
            pricing_history=pd.DataFrame(),
            required_delivery=required,
        )

    cg = _run_column_generation_node(
        problem,
        required,
        max_iterations=max_iterations,
        max_new_columns_per_iteration=max_new_columns_per_iteration,
    )

    artificial_mass = float(
        cg.lp.artificial_values.sum()
    )
    if artificial_mass > 1e-7:
        raise RuntimeError(
            "column generation ended with positive artificial coverage; "
            "routing problem is infeasible under current branch/fleet limits"
        )

    integer_objective, integer_values = (
        _solve_restricted_integer_master(
            problem,
            required,
            cg.columns,
        )
    )

    return ColumnGenerationResult(
        columns=_columns_frame(cg.columns),
        selected_routes=_selected_route_frame(
            cg.columns,
            integer_values,
        ),
        lp_objective=cg.lp.objective,
        integer_objective=integer_objective,
        iterations=cg.iterations,
        generated_columns=cg.added_columns,
        artificial_mass=artificial_mass,
        pricing_history=cg.pricing_history,
        required_delivery=required,
    )


def _choose_ryan_foster_pair(
    active: list[str],
    columns: dict[tuple[str, ...], RouteColumn],
    route_values: pd.Series,
    tolerance: float = 1e-6,
) -> tuple[str, str] | None:
    best_pair = None
    best_distance = float("inf")

    for left, right in combinations(active, 2):
        together_value = float(
            sum(
                value
                for key, value in route_values.items()
                if left in columns[key].stops
                and right in columns[key].stops
            )
        )

        if (
            together_value <= tolerance
            or together_value >= 1.0 - tolerance
        ):
            continue

        distance_to_half = abs(together_value - 0.5)
        if distance_to_half < best_distance:
            best_distance = distance_to_half
            best_pair = (left, right)

    return best_pair


def solve_route_branch_and_price(
    problem: CashSupplyChainProblem,
    required_delivery: pd.Series,
    max_nodes: int = 50,
    max_cg_iterations: int = 50,
    max_new_columns_per_iteration: int = 6,
    integrality_tolerance: float = 1e-6,
) -> BranchAndPriceResult:
    """Solve fixed-delivery routing with Ryan-Foster branch-and-price."""
    required = _validate_requirements(
        problem,
        required_delivery,
    )
    active = list(required.index)

    if required.empty:
        return BranchAndPriceResult(
            selected_routes=pd.DataFrame(),
            objective_value=0.0,
            root_lp_bound=0.0,
            nodes_processed=0,
            nodes_pruned=0,
            columns_generated=0,
            incumbent_updates=0,
            exact=True,
            search_log=pd.DataFrame(),
        )

    root = _run_column_generation_node(
        problem,
        required,
        max_iterations=max_cg_iterations,
        max_new_columns_per_iteration=max_new_columns_per_iteration,
    )
    if float(root.lp.artificial_values.sum()) > 1e-7:
        raise RuntimeError("root routing master is infeasible")

    root_lp_bound = float(root.lp.objective)
    total_generated = root.added_columns

    # Initial incumbent from restricted integer master.
    incumbent_obj, incumbent_values = _solve_restricted_integer_master(
        problem,
        required,
        root.columns,
    )
    incumbent_columns = dict(root.columns)
    incumbent_updates = 1

    # DFS stack: (together, separate, inherited columns, depth)
    stack = [
        (
            frozenset(),
            frozenset(),
            dict(root.columns),
            0,
        )
    ]

    nodes_processed = 0
    nodes_pruned = 0
    log_rows = []
    exact = True

    while stack:
        if nodes_processed >= max_nodes:
            exact = False
            break

        together, separate, inherited, depth = stack.pop()
        nodes_processed += 1

        if depth == 0:
            node = root
        else:
            node = _run_column_generation_node(
                problem,
                required,
                seed_columns=inherited,
                together_pairs=together,
                separate_pairs=separate,
                max_iterations=max_cg_iterations,
                max_new_columns_per_iteration=max_new_columns_per_iteration,
            )
            total_generated += node.added_columns

        artificial_mass = float(
            node.lp.artificial_values.sum()
        )
        lower_bound = float(node.lp.objective)

        status = "open"

        if artificial_mass > 1e-7:
            status = "pruned_infeasible"
            nodes_pruned += 1
            log_rows.append(
                {
                    "node": nodes_processed,
                    "depth": depth,
                    "lower_bound": lower_bound,
                    "incumbent": incumbent_obj,
                    "artificial_mass": artificial_mass,
                    "status": status,
                }
            )
            continue

        if lower_bound >= incumbent_obj - 1e-8:
            status = "pruned_bound"
            nodes_pruned += 1
            log_rows.append(
                {
                    "node": nodes_processed,
                    "depth": depth,
                    "lower_bound": lower_bound,
                    "incumbent": incumbent_obj,
                    "artificial_mass": artificial_mass,
                    "status": status,
                }
            )
            continue

        values = node.lp.route_values
        fractional = values[
            (values > integrality_tolerance)
            & (values < 1.0 - integrality_tolerance)
        ]

        if fractional.empty:
            candidate_obj = float(
                sum(
                    node.columns[key].cost * value
                    for key, value in values.items()
                )
            )
            if candidate_obj < incumbent_obj - 1e-8:
                incumbent_obj = candidate_obj
                incumbent_values = values.copy()
                incumbent_columns = dict(node.columns)
                incumbent_updates += 1
            status = "integral"
            log_rows.append(
                {
                    "node": nodes_processed,
                    "depth": depth,
                    "lower_bound": lower_bound,
                    "incumbent": incumbent_obj,
                    "artificial_mass": artificial_mass,
                    "status": status,
                }
            )
            continue

        pair = _choose_ryan_foster_pair(
            active,
            node.columns,
            values,
            tolerance=integrality_tolerance,
        )
        if pair is None:
            # Ryan-Foster theory normally supplies such a pair for a
            # fractional set-partitioning solution. If numerical degeneracy
            # prevents finding one, terminate with a valid incumbent but mark
            # the search as non-exact rather than misrepresenting completeness.
            exact = False
            status = "stopped_no_pair"
            log_rows.append(
                {
                    "node": nodes_processed,
                    "depth": depth,
                    "lower_bound": lower_bound,
                    "incumbent": incumbent_obj,
                    "artificial_mass": artificial_mass,
                    "status": status,
                }
            )
            break

        left, right = tuple(sorted(pair))
        together_pair = (left, right)

        # Separate child: no generated route may contain both.
        separate_child = frozenset(
            set(separate) | {together_pair}
        )
        stack.append(
            (
                together,
                separate_child,
                dict(node.columns),
                depth + 1,
            )
        )

        # Together child: generated route must contain both or neither.
        together_child = frozenset(
            set(together) | {together_pair}
        )
        stack.append(
            (
                together_child,
                separate,
                dict(node.columns),
                depth + 1,
            )
        )

        status = f"branched_{left}_{right}"
        log_rows.append(
            {
                "node": nodes_processed,
                "depth": depth,
                "lower_bound": lower_bound,
                "incumbent": incumbent_obj,
                "artificial_mass": artificial_mass,
                "status": status,
            }
        )

    selected = _selected_route_frame(
        incumbent_columns,
        incumbent_values,
    )

    return BranchAndPriceResult(
        selected_routes=selected,
        objective_value=float(incumbent_obj),
        root_lp_bound=root_lp_bound,
        nodes_processed=nodes_processed,
        nodes_pruned=nodes_pruned,
        columns_generated=total_generated,
        incumbent_updates=incumbent_updates,
        exact=exact and not stack,
        search_log=pd.DataFrame(log_rows),
    )
