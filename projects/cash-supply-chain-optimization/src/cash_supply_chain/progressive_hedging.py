"""L1 Progressive Hedging heuristic for multistage cash IRP decomposition.

This module decomposes the multistage stochastic IRP by terminal scenario path.
Each leaf solves an independent route-column MILP. Decisions attached to a
shared information-history node are coordinated through probability-weighted
consensus, Lagrange multipliers, and L1 proximal penalties.

Classic Progressive Hedging uses a quadratic proximal term. SciPy/HiGHS solves
linear MILPs, so this implementation uses an L1 linearization. It should be
treated as a decomposition heuristic / educational PH variant rather than a
drop-in implementation of quadratic PH convergence theory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .joint_irp import generate_route_catalog
from .model import CashSupplyChainProblem, default_problem
from .multistage_cvar_irp import (
    ScenarioTree,
    _path_nodes,
    solve_multistage_cvar_irp,
)


@dataclass(frozen=True)
class ScenarioPathSolution:
    leaf: str
    probability: float
    route_decisions: dict[str, pd.Series]
    delivery_decisions: dict[str, pd.Series]
    base_cost: float
    total_cashout: float


@dataclass(frozen=True)
class ProgressiveHedgingResult:
    consensus_routes: pd.DataFrame
    consensus_deliveries: pd.DataFrame
    convergence_history: pd.DataFrame
    scenario_costs: pd.Series
    expected_decomposed_cost: float
    converged: bool
    iterations: int
    final_residual: float
    route_rho: float
    delivery_rho: float

    def to_dict(self) -> dict:
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "final_residual": round(self.final_residual, 6),
            "expected_decomposed_cost": round(
                self.expected_decomposed_cost, 6
            ),
            "route_rho": round(self.route_rho, 6),
            "delivery_rho": round(self.delivery_rho, 6),
            "scenario_costs": self.scenario_costs.round(6).to_dict(),
        }


def _leaf_scenarios(tree: ScenarioTree) -> list[str]:
    horizon = len(tree.days)
    return list(tree.nodes.index[tree.nodes["stage"] == horizon])


def _decision_path(tree: ScenarioTree, leaf: str) -> list[str]:
    horizon = len(tree.days)
    return [
        node
        for node in _path_nodes(tree, leaf)
        if int(tree.nodes.loc[node, "stage"]) < horizon
    ]


def _realization_path(tree: ScenarioTree, leaf: str) -> list[str]:
    return [
        node
        for node in _path_nodes(tree, leaf)
        if node != "root"
    ]


def _route_cost(
    problem: CashSupplyChainProblem,
    catalog: pd.DataFrame,
    route: str,
) -> float:
    return (
        problem.vehicle_fixed_cost
        + float(catalog.loc[route, "visit_cost"])
        + problem.distance_cost_per_unit
        * float(catalog.loc[route, "distance"])
    )


def _solve_leaf_subproblem(
    problem: CashSupplyChainProblem,
    tree: ScenarioTree,
    leaf: str,
    consensus_routes: dict[str, pd.Series] | None = None,
    consensus_deliveries: dict[str, pd.Series] | None = None,
    route_multipliers: dict[tuple[str, str], float] | None = None,
    delivery_multipliers: dict[tuple[str, str], float] | None = None,
    route_rho: float = 0.0,
    delivery_rho: float = 0.0,
) -> ScenarioPathSolution:
    """Solve one leaf-path MILP with optional L1 PH augmentation."""
    p = problem
    catalog = generate_route_catalog(p)
    routes = list(catalog.index)
    cashpoints = list(p.cashpoints.index)

    decision_nodes = _decision_path(tree, leaf)
    realization_nodes = _realization_path(tree, leaf)

    route_stop_pairs = [
        (route, cashpoint)
        for route in routes
        for cashpoint in catalog.loc[route, "stops"]
    ]

    z_keys = [(node, route) for node in decision_nodes for route in routes]
    q_keys = [
        (node, route, cashpoint)
        for node in decision_nodes
        for route, cashpoint in route_stop_pairs
    ]
    state_keys = [
        (node, cashpoint)
        for node in realization_nodes
        for cashpoint in cashpoints
    ]
    dev_keys = [
        (node, cashpoint)
        for node in decision_nodes
        for cashpoint in cashpoints
    ]

    z0 = 0
    q0 = z0 + len(z_keys)
    inv0 = q0 + len(q_keys)
    safe0 = inv0 + len(state_keys)
    lost0 = safe0 + len(state_keys)
    dplus0 = lost0 + len(state_keys)
    dminus0 = dplus0 + len(dev_keys)
    n_vars = dminus0 + len(dev_keys)

    z_i = {key: k for k, key in enumerate(z_keys)}
    q_i = {key: k for k, key in enumerate(q_keys)}
    state_i = {key: k for k, key in enumerate(state_keys)}
    dev_i = {key: k for k, key in enumerate(dev_keys)}

    route_multipliers = route_multipliers or {}
    delivery_multipliers = delivery_multipliers or {}

    c = np.zeros(n_vars)

    for node in decision_nodes:
        for route in routes:
            idx = z0 + z_i[(node, route)]
            c[idx] += _route_cost(p, catalog, route)

            if consensus_routes is not None:
                zbar = float(consensus_routes[node].loc[route])
                multiplier = float(
                    route_multipliers.get((node, route), 0.0)
                )
                c[idx] += multiplier
                # For binary z:
                # |z-zbar| = z*(1-2*zbar) + constant.
                c[idx] += route_rho * (1.0 - 2.0 * zbar)

        for route, cashpoint in route_stop_pairs:
            c[
                q0 + q_i[(node, route, cashpoint)]
            ] += p.handling_cost_per_unit

        if consensus_deliveries is not None:
            for cashpoint in cashpoints:
                multiplier = float(
                    delivery_multipliers.get(
                        (node, cashpoint),
                        0.0,
                    )
                )
                for route in routes:
                    if cashpoint in catalog.loc[route, "stops"]:
                        c[
                            q0 + q_i[(node, route, cashpoint)]
                        ] += multiplier

                key = (node, cashpoint)
                c[dplus0 + dev_i[key]] += delivery_rho
                c[dminus0 + dev_i[key]] += delivery_rho

    for node in realization_nodes:
        for cashpoint in cashpoints:
            key = (node, cashpoint)
            c[inv0 + state_i[key]] += p.holding_cost_per_unit_day
            c[safe0 + state_i[key]] += p.shortage_penalty_per_unit
            c[lost0 + state_i[key]] += p.cashout_penalty_per_unit

    eq_rows: list[np.ndarray] = []
    eq_rhs: list[float] = []

    # Inventory transition along this one terminal path.
    for node in realization_nodes:
        parent = str(tree.nodes.loc[node, "parent"])

        for cashpoint in cashpoints:
            row = np.zeros(n_vars)

            for route in routes:
                if cashpoint in catalog.loc[route, "stops"]:
                    row[
                        q0 + q_i[(parent, route, cashpoint)]
                    ] = 1.0

            row[lost0 + state_i[(node, cashpoint)]] = 1.0
            row[inv0 + state_i[(node, cashpoint)]] = -1.0

            if parent == "root":
                rhs = (
                    float(tree.demand.loc[node, cashpoint])
                    - float(p.cashpoints.loc[cashpoint, "initial_cash"])
                )
            else:
                row[inv0 + state_i[(parent, cashpoint)]] = 1.0
                rhs = float(tree.demand.loc[node, cashpoint])

            eq_rows.append(row)
            eq_rhs.append(rhs)

    # Continuous delivery consensus absolute-deviation linearization.
    if consensus_deliveries is not None:
        for node in decision_nodes:
            for cashpoint in cashpoints:
                row = np.zeros(n_vars)

                for route in routes:
                    if cashpoint in catalog.loc[route, "stops"]:
                        row[
                            q0 + q_i[(node, route, cashpoint)]
                        ] = 1.0

                key = (node, cashpoint)
                row[dplus0 + dev_i[key]] = -1.0
                row[dminus0 + dev_i[key]] = 1.0

                eq_rows.append(row)
                eq_rhs.append(
                    float(consensus_deliveries[node].loc[cashpoint])
                )

    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []

    for node in decision_nodes:
        stage = int(tree.nodes.loc[node, "stage"])
        day = tree.days[stage]

        # At most one route visit per cashpoint.
        for cashpoint in cashpoints:
            row = np.zeros(n_vars)
            for route in routes:
                if cashpoint in catalog.loc[route, "stops"]:
                    row[z0 + z_i[(node, route)]] = 1.0
            ub_rows.append(row)
            ub_rhs.append(1.0)

        # Route capacity and delivery linking.
        for route in routes:
            z_idx = z0 + z_i[(node, route)]
            capacity_row = np.zeros(n_vars)

            for cashpoint in catalog.loc[route, "stops"]:
                q_idx = q0 + q_i[(node, route, cashpoint)]
                capacity_row[q_idx] = 1.0

                row = np.zeros(n_vars)
                row[q_idx] = 1.0
                row[z_idx] = -float(
                    p.cashpoints.loc[cashpoint, "maximum_delivery"]
                )
                ub_rows.append(row)
                ub_rhs.append(0.0)

            capacity_row[z_idx] = -p.vehicle_capacity
            ub_rows.append(capacity_row)
            ub_rhs.append(0.0)

        # Vault dispatch.
        row = np.zeros(n_vars)
        for route, cashpoint in route_stop_pairs:
            row[q0 + q_i[(node, route, cashpoint)]] = 1.0
        ub_rows.append(row)
        ub_rhs.append(float(p.vault_dispatch_limit.loc[day]))

        # Fleet.
        row = np.zeros(n_vars)
        for route in routes:
            row[z0 + z_i[(node, route)]] = 1.0
        ub_rows.append(row)
        ub_rhs.append(float(p.maximum_vehicles_per_day))

        # Cluster visit limits.
        for cluster, limit in p.cluster_visit_limit.items():
            members = set(
                p.cashpoints.index[p.cashpoints["cluster"] == cluster]
            )
            row = np.zeros(n_vars)

            for route in routes:
                count = len(
                    members.intersection(catalog.loc[route, "stops"])
                )
                if count:
                    row[z0 + z_i[(node, route)]] = float(count)

            ub_rows.append(row)
            ub_rhs.append(float(limit))

        # Cashpoint pre-demand capacity.
        for cashpoint in cashpoints:
            row = np.zeros(n_vars)

            if node != "root":
                row[inv0 + state_i[(node, cashpoint)]] = 1.0

            for route in routes:
                if cashpoint in catalog.loc[route, "stops"]:
                    row[
                        q0 + q_i[(node, route, cashpoint)]
                    ] = 1.0

            rhs = float(p.cashpoints.loc[cashpoint, "capacity"])
            if node == "root":
                rhs -= float(
                    p.cashpoints.loc[cashpoint, "initial_cash"]
                )

            ub_rows.append(row)
            ub_rhs.append(rhs)

    for node in realization_nodes:
        for cashpoint in cashpoints:
            key = (node, cashpoint)

            row = np.zeros(n_vars)
            row[inv0 + state_i[key]] = -1.0
            row[safe0 + state_i[key]] = -1.0
            ub_rows.append(row)
            ub_rhs.append(
                -float(p.cashpoints.loc[cashpoint, "safety_stock"])
            )

            row = np.zeros(n_vars)
            row[lost0 + state_i[key]] = 1.0
            ub_rows.append(row)
            ub_rhs.append(float(tree.demand.loc[node, cashpoint]))

    constraints = [
        LinearConstraint(
            np.vstack(eq_rows),
            lb=np.asarray(eq_rhs, dtype=float),
            ub=np.asarray(eq_rhs, dtype=float),
        ),
        LinearConstraint(
            np.vstack(ub_rows),
            lb=np.full(len(ub_rows), -np.inf),
            ub=np.asarray(ub_rhs, dtype=float),
        ),
    ]

    lower = np.zeros(n_vars)
    upper = np.full(n_vars, np.inf)
    upper[z0:q0] = 1.0

    for node, cashpoint in state_keys:
        upper[inv0 + state_i[(node, cashpoint)]] = float(
            p.cashpoints.loc[cashpoint, "capacity"]
        )

    integrality = np.zeros(n_vars, dtype=int)
    integrality[z0:q0] = 1

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(
            f"PH leaf subproblem failed for {leaf}: {result.message}"
        )

    x = result.x

    route_decisions: dict[str, pd.Series] = {}
    delivery_decisions: dict[str, pd.Series] = {}

    for node in decision_nodes:
        route_decisions[node] = pd.Series(
            {
                route: x[z0 + z_i[(node, route)]]
                for route in routes
            },
            dtype=float,
        )

        delivery_decisions[node] = pd.Series(
            {
                cashpoint: sum(
                    x[q0 + q_i[(node, route, cashpoint)]]
                    for route in routes
                    if cashpoint in catalog.loc[route, "stops"]
                )
                for cashpoint in cashpoints
            },
            dtype=float,
        )

    # Compute original scenario-path cost without PH augmentation.
    base_cost = 0.0
    total_cashout = 0.0

    for node in decision_nodes:
        for route in routes:
            base_cost += (
                _route_cost(p, catalog, route)
                * route_decisions[node].loc[route]
            )

        base_cost += (
            delivery_decisions[node].sum()
            * p.handling_cost_per_unit
        )

    for node in realization_nodes:
        for cashpoint in cashpoints:
            key = (node, cashpoint)
            base_cost += (
                p.holding_cost_per_unit_day
                * x[inv0 + state_i[key]]
            )
            base_cost += (
                p.shortage_penalty_per_unit
                * x[safe0 + state_i[key]]
            )
            base_cost += (
                p.cashout_penalty_per_unit
                * x[lost0 + state_i[key]]
            )
            total_cashout += x[lost0 + state_i[key]]

    return ScenarioPathSolution(
        leaf=leaf,
        probability=float(tree.nodes.loc[leaf, "probability"]),
        route_decisions=route_decisions,
        delivery_decisions=delivery_decisions,
        base_cost=float(base_cost),
        total_cashout=float(total_cashout),
    )


def _compute_consensus(
    tree: ScenarioTree,
    solutions: dict[str, ScenarioPathSolution],
    routes: list[str],
    cashpoints: list[str],
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    horizon = len(tree.days)
    decision_nodes = list(
        tree.nodes.index[tree.nodes["stage"] < horizon]
    )

    route_consensus: dict[str, pd.Series] = {}
    delivery_consensus: dict[str, pd.Series] = {}

    for node in decision_nodes:
        group = [
            leaf
            for leaf, solution in solutions.items()
            if node in solution.route_decisions
        ]

        if not group:
            continue

        group_probability = sum(
            solutions[leaf].probability for leaf in group
        )
        if group_probability <= 0:
            raise RuntimeError("nonpositive PH history-group probability")

        route_value = pd.Series(0.0, index=routes)
        delivery_value = pd.Series(0.0, index=cashpoints)

        for leaf in group:
            weight = solutions[leaf].probability / group_probability
            route_value += (
                weight * solutions[leaf].route_decisions[node]
            )
            delivery_value += (
                weight * solutions[leaf].delivery_decisions[node]
            )

        route_consensus[node] = route_value
        delivery_consensus[node] = delivery_value

    return route_consensus, delivery_consensus


def _consensus_residual(
    problem: CashSupplyChainProblem,
    solutions: dict[str, ScenarioPathSolution],
    route_consensus: dict[str, pd.Series],
    delivery_consensus: dict[str, pd.Series],
) -> tuple[float, float, float]:
    route_residual = 0.0
    delivery_residual = 0.0

    for solution in solutions.values():
        for node, route_values in solution.route_decisions.items():
            route_residual = max(
                route_residual,
                float(
                    (route_values - route_consensus[node])
                    .abs()
                    .max()
                ),
            )

        for node, delivery_values in solution.delivery_decisions.items():
            scales = problem.cashpoints.loc[
                delivery_values.index,
                "maximum_delivery",
            ].clip(lower=1.0)

            normalized = (
                (delivery_values - delivery_consensus[node]).abs()
                / scales
            )
            delivery_residual = max(
                delivery_residual,
                float(normalized.max()),
            )

    return (
        route_residual,
        delivery_residual,
        max(route_residual, delivery_residual),
    )


def run_progressive_hedging(
    problem: CashSupplyChainProblem | None = None,
    tree: ScenarioTree | None = None,
    max_iterations: int = 12,
    tolerance: float = 0.02,
    route_rho: float = 8.0,
    delivery_rho: float = 0.08,
    rho_growth: float = 1.35,
    rho_update_interval: int = 4,
) -> ProgressiveHedgingResult:
    """Run probability-weighted L1 Progressive Hedging on leaf subproblems."""
    p = problem or default_problem()
    if tree is None:
        raise ValueError("tree must be supplied")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if route_rho <= 0 or delivery_rho <= 0:
        raise ValueError("PH penalty parameters must be positive")

    catalog = generate_route_catalog(p)
    routes = list(catalog.index)
    cashpoints = list(p.cashpoints.index)
    leaves = _leaf_scenarios(tree)

    # Independent scenario solves initialize the consensus.
    solutions = {
        leaf: _solve_leaf_subproblem(
            p,
            tree,
            leaf,
        )
        for leaf in leaves
    }

    route_consensus, delivery_consensus = _compute_consensus(
        tree,
        solutions,
        routes,
        cashpoints,
    )

    route_multipliers: dict[
        str, dict[tuple[str, str], float]
    ] = {leaf: {} for leaf in leaves}
    delivery_multipliers: dict[
        str, dict[tuple[str, str], float]
    ] = {leaf: {} for leaf in leaves}

    history_rows = []
    converged = False
    current_route_rho = float(route_rho)
    current_delivery_rho = float(delivery_rho)
    previous_residual = float("inf")

    for iteration in range(1, max_iterations + 1):
        new_solutions = {}

        for leaf in leaves:
            new_solutions[leaf] = _solve_leaf_subproblem(
                p,
                tree,
                leaf,
                consensus_routes=route_consensus,
                consensus_deliveries=delivery_consensus,
                route_multipliers=route_multipliers[leaf],
                delivery_multipliers=delivery_multipliers[leaf],
                route_rho=current_route_rho,
                delivery_rho=current_delivery_rho,
            )

        new_route_consensus, new_delivery_consensus = _compute_consensus(
            tree,
            new_solutions,
            routes,
            cashpoints,
        )

        route_residual, delivery_residual, residual = _consensus_residual(
            p,
            new_solutions,
            new_route_consensus,
            new_delivery_consensus,
        )

        expected_cost = float(
            sum(
                solution.probability * solution.base_cost
                for solution in new_solutions.values()
            )
        )

        history_rows.append(
            {
                "iteration": iteration,
                "route_residual": route_residual,
                "delivery_residual": delivery_residual,
                "max_residual": residual,
                "expected_decomposed_cost": expected_cost,
                "route_rho": current_route_rho,
                "delivery_rho": current_delivery_rho,
            }
        )

        solutions = new_solutions
        route_consensus = new_route_consensus
        delivery_consensus = new_delivery_consensus

        if residual <= tolerance:
            converged = True
            break

        # Multiplier update by history group.
        for leaf, solution in solutions.items():
            for node, route_values in solution.route_decisions.items():
                for route in routes:
                    key = (node, route)
                    route_multipliers[leaf][key] = (
                        route_multipliers[leaf].get(key, 0.0)
                        + current_route_rho
                        * (
                            route_values.loc[route]
                            - route_consensus[node].loc[route]
                        )
                    )

            for node, delivery_values in solution.delivery_decisions.items():
                for cashpoint in cashpoints:
                    key = (node, cashpoint)
                    delivery_multipliers[leaf][key] = (
                        delivery_multipliers[leaf].get(key, 0.0)
                        + current_delivery_rho
                        * (
                            delivery_values.loc[cashpoint]
                            - delivery_consensus[node].loc[cashpoint]
                        )
                    )

        if (
            iteration % rho_update_interval == 0
            and residual > 0.90 * previous_residual
        ):
            current_route_rho *= rho_growth
            current_delivery_rho *= rho_growth

        previous_residual = residual

    scenario_costs = pd.Series(
        {
            leaf: solution.base_cost
            for leaf, solution in solutions.items()
        },
        name="scenario_path_cost",
    )
    expected_decomposed_cost = float(
        sum(
            solution.probability * solution.base_cost
            for solution in solutions.values()
        )
    )

    consensus_routes_df = pd.DataFrame.from_dict(
        route_consensus,
        orient="index",
    ).sort_index()
    consensus_deliveries_df = pd.DataFrame.from_dict(
        delivery_consensus,
        orient="index",
    ).sort_index()

    history = pd.DataFrame(history_rows)

    final_residual = (
        float(history.iloc[-1]["max_residual"])
        if not history.empty
        else float("inf")
    )

    return ProgressiveHedgingResult(
        consensus_routes=consensus_routes_df,
        consensus_deliveries=consensus_deliveries_df,
        convergence_history=history,
        scenario_costs=scenario_costs,
        expected_decomposed_cost=expected_decomposed_cost,
        converged=converged,
        iterations=len(history),
        final_residual=final_residual,
        route_rho=current_route_rho,
        delivery_rho=current_delivery_rho,
    )


def benchmark_progressive_hedging(
    problem: CashSupplyChainProblem,
    tree: ScenarioTree,
    ph_result: ProgressiveHedgingResult | None = None,
) -> tuple[pd.DataFrame, ProgressiveHedgingResult]:
    """Compare decomposition diagnostics to the exact risk-neutral extensive form."""
    ph = ph_result or run_progressive_hedging(
        problem,
        tree,
    )
    exact = solve_multistage_cvar_irp(
        problem,
        tree=tree,
        risk_aversion=0.0,
    )

    table = pd.DataFrame(
        [
            {
                "method": "exact_extensive_form",
                "expected_cost": exact.expected_cost,
                "nonanticipativity_residual": 0.0,
                "iterations": 1,
                "converged": True,
            },
            {
                "method": "l1_progressive_hedging",
                "expected_cost": ph.expected_decomposed_cost,
                "nonanticipativity_residual": ph.final_residual,
                "iterations": ph.iterations,
                "converged": ph.converged,
            },
        ]
    ).set_index("method")

    # The decomposed cost is a diagnostic and can be optimistic before exact
    # non-anticipativity is reached. Do not label it an optimality gap unless
    # PH has converged.
    table["cost_difference_vs_exact"] = (
        table["expected_cost"]
        - table.loc["exact_extensive_form", "expected_cost"]
    )

    return table, ph
