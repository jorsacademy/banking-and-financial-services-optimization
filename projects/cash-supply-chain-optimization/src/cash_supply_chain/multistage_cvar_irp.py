"""Multistage scenario-tree cash IRP with CVaR risk aversion.

Timing convention
-----------------
The root is the state before day 1 demand. A decision node chooses routes and
deliveries for the next day. Each child node represents the demand realization
after that decision. Nodes sharing the same history are literally the same
decision node, so multistage non-anticipativity is structural rather than
enforced by pairwise equality constraints.

The objective combines expected operating cost with optional CVaR of total
path cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .joint_irp import generate_route_catalog
from .model import CashSupplyChainProblem, default_problem


@dataclass(frozen=True)
class ScenarioTree:
    nodes: pd.DataFrame
    demand: pd.DataFrame
    days: tuple[int, ...]


@dataclass(frozen=True)
class MultistageCVaRResult:
    route_policy: pd.DataFrame
    delivery_policy: pd.DataFrame
    inventory: pd.DataFrame
    cashout: pd.DataFrame
    safety_shortfall: pd.DataFrame
    leaf_costs: pd.Series
    expected_cost: float
    var_threshold: float
    cvar_cost: float
    worst_leaf_cost: float
    expected_cashout: float
    objective_value: float
    cvar_alpha: float
    risk_aversion: float

    def to_dict(self) -> dict:
        return {
            "expected_cost": round(self.expected_cost, 6),
            "var_threshold": round(self.var_threshold, 6),
            "cvar_cost": round(self.cvar_cost, 6),
            "worst_leaf_cost": round(self.worst_leaf_cost, 6),
            "expected_cashout": round(self.expected_cashout, 6),
            "objective_value": round(self.objective_value, 6),
            "cvar_alpha": self.cvar_alpha,
            "risk_aversion": self.risk_aversion,
            "leaf_costs": self.leaf_costs.round(6).to_dict(),
            "route_policy": self.route_policy.round(
                {"load": 6, "distance": 6}
            ).to_dict(orient="records"),
        }


def generate_binary_scenario_tree(
    problem: CashSupplyChainProblem | None = None,
    days: tuple[int, ...] = (1, 2, 3),
    demand_sigma: float = 0.12,
    seed: int = 2026,
) -> ScenarioTree:
    """Create a binary multistage demand tree.

    Demand is attached to non-root realization nodes. A node at stage t
    contains the demand observed after the decision made at its parent.
    """
    p = problem or default_problem()
    cashpoints = list(p.cashpoints.index)

    if not days:
        raise ValueError("days cannot be empty")
    if demand_sigma < 0:
        raise ValueError("demand_sigma must be nonnegative")
    if any(day not in p.forecast_net_withdrawal.columns for day in days):
        raise ValueError("all scenario-tree days must exist in the problem")

    rng = np.random.default_rng(seed)

    records = [
        {
            "node": "root",
            "parent": None,
            "stage": 0,
            "day": np.nan,
            "probability": 1.0,
        }
    ]
    demand_rows: dict[str, np.ndarray] = {}

    parents = ["root"]

    for stage, day in enumerate(days, start=1):
        next_parents = []

        for parent in parents:
            parent_probability = next(
                row["probability"]
                for row in records
                if row["node"] == parent
            )

            for branch in ("L", "H"):
                node = f"{parent}_{branch}" if parent != "root" else branch
                probability = parent_probability * 0.5

                # Low/high common factor creates economically interpretable
                # branch regimes while idiosyncratic noise preserves
                # cross-sectional variation.
                common = 0.90 if branch == "L" else 1.10
                idiosyncratic = rng.lognormal(
                    mean=-0.5 * demand_sigma**2,
                    sigma=demand_sigma,
                    size=len(cashpoints),
                )
                realized = (
                    p.forecast_net_withdrawal.loc[cashpoints, day].to_numpy(float)
                    * common
                    * idiosyncratic
                )

                records.append(
                    {
                        "node": node,
                        "parent": parent,
                        "stage": stage,
                        "day": day,
                        "probability": probability,
                    }
                )
                demand_rows[node] = realized
                next_parents.append(node)

        parents = next_parents

    nodes = pd.DataFrame(records).set_index("node")
    demand = pd.DataFrame.from_dict(
        demand_rows,
        orient="index",
        columns=cashpoints,
    )

    return ScenarioTree(nodes=nodes, demand=demand, days=tuple(days))


def _path_nodes(tree: ScenarioTree, leaf: str) -> list[str]:
    path = [leaf]
    node = leaf

    while node != "root":
        parent = tree.nodes.loc[node, "parent"]
        if parent is None:
            break
        node = str(parent)
        path.append(node)

    return list(reversed(path))


def _weighted_var_cvar(
    costs: pd.Series,
    probabilities: pd.Series,
    alpha: float,
) -> tuple[float, float]:
    """Compute discrete weighted VaR/CVaR via the Rockafellar-Uryasev form."""
    probabilities = probabilities.loc[costs.index].astype(float)
    probabilities = probabilities / probabilities.sum()

    candidates = sorted(set(float(v) for v in costs.to_numpy()))
    best_eta = candidates[0]
    best_value = float("inf")

    for eta in candidates:
        value = eta + float(
            (
                probabilities
                * (costs - eta).clip(lower=0.0)
            ).sum()
        ) / (1.0 - alpha)
        if value < best_value - 1e-12:
            best_eta = eta
            best_value = value

    return float(best_eta), float(best_value)


def solve_multistage_cvar_irp(
    problem: CashSupplyChainProblem | None = None,
    tree: ScenarioTree | None = None,
    cvar_alpha: float = 0.90,
    risk_aversion: float = 0.0,
) -> MultistageCVaRResult:
    """Solve a multistage route-column IRP with optional CVaR penalty."""
    p = problem or default_problem()
    tree = tree or generate_binary_scenario_tree(p)

    if not (0.0 < cvar_alpha < 1.0):
        raise ValueError("cvar_alpha must be strictly between zero and one")
    if risk_aversion < 0.0:
        raise ValueError("risk_aversion must be nonnegative")

    cashpoints = list(p.cashpoints.index)
    catalog = generate_route_catalog(p)
    routes = list(catalog.index)

    horizon = len(tree.days)
    nodes = tree.nodes
    decision_nodes = list(nodes.index[nodes["stage"] < horizon])
    realization_nodes = list(nodes.index[nodes["stage"] > 0])
    leaves = list(nodes.index[nodes["stage"] == horizon])

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

    z0 = 0
    q0 = z0 + len(z_keys)
    inv0 = q0 + len(q_keys)
    safe0 = inv0 + len(state_keys)
    lost0 = safe0 + len(state_keys)
    eta_idx = lost0 + len(state_keys)
    xi0 = eta_idx + 1
    n_vars = xi0 + len(leaves)

    z_i = {key: k for k, key in enumerate(z_keys)}
    q_i = {key: k for k, key in enumerate(q_keys)}
    state_i = {key: k for k, key in enumerate(state_keys)}
    leaf_i = {leaf: k for k, leaf in enumerate(leaves)}

    c = np.zeros(n_vars)

    def route_cost(route: str) -> float:
        return (
            p.vehicle_fixed_cost
            + float(catalog.loc[route, "visit_cost"])
            + p.distance_cost_per_unit * float(catalog.loc[route, "distance"])
        )

    # Expected operating costs.
    for node in decision_nodes:
        probability = float(nodes.loc[node, "probability"])

        for route in routes:
            c[z0 + z_i[(node, route)]] += probability * route_cost(route)

        for route, cashpoint in route_stop_pairs:
            c[
                q0 + q_i[(node, route, cashpoint)]
            ] += probability * p.handling_cost_per_unit

    for node in realization_nodes:
        probability = float(nodes.loc[node, "probability"])

        for cashpoint in cashpoints:
            key = (node, cashpoint)
            c[inv0 + state_i[key]] += (
                probability * p.holding_cost_per_unit_day
            )
            c[safe0 + state_i[key]] += (
                probability * p.shortage_penalty_per_unit
            )
            c[lost0 + state_i[key]] += (
                probability * p.cashout_penalty_per_unit
            )

    # CVaR auxiliary variables.
    c[eta_idx] = risk_aversion
    for leaf in leaves:
        probability = float(nodes.loc[leaf, "probability"])
        c[xi0 + leaf_i[leaf]] = (
            risk_aversion * probability / (1.0 - cvar_alpha)
        )

    eq_rows: list[np.ndarray] = []
    eq_rhs: list[float] = []

    # State transition on every realized edge.
    for node in realization_nodes:
        parent = str(nodes.loc[node, "parent"])

        for cashpoint in cashpoints:
            row = np.zeros(n_vars)

            # Parent decision determines delivery before node demand realizes.
            for route in routes:
                if cashpoint in catalog.loc[route, "stops"]:
                    row[
                        q0 + q_i[(parent, route, cashpoint)]
                    ] = 1.0

            # Lost demand keeps inventory nonnegative.
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

    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []

    # Decision-node operational constraints.
    for node in decision_nodes:
        stage = int(nodes.loc[node, "stage"])
        day = tree.days[stage]

        # At most one route visit per cashpoint.
        for cashpoint in cashpoints:
            row = np.zeros(n_vars)
            for route in routes:
                if cashpoint in catalog.loc[route, "stops"]:
                    row[z0 + z_i[(node, route)]] = 1.0
            ub_rows.append(row)
            ub_rhs.append(1.0)

        # Route-specific capacity and delivery linking.
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

        # Fleet count.
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
                count = len(members.intersection(catalog.loc[route, "stops"]))
                if count:
                    row[z0 + z_i[(node, route)]] = float(count)
            ub_rows.append(row)
            ub_rhs.append(float(limit))

        # Pre-demand cashpoint capacity.
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

    # Realization-node service constraints and lost-demand bounds.
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

    # CVaR excess constraints:
    # path_cost(leaf) - eta - xi_leaf <= 0.
    for leaf in leaves:
        row = np.zeros(n_vars)
        path = _path_nodes(tree, leaf)

        decision_path = [
            node for node in path
            if node in decision_nodes
        ]
        realization_path = [
            node for node in path
            if node in realization_nodes
        ]

        for node in decision_path:
            for route in routes:
                row[z0 + z_i[(node, route)]] += route_cost(route)

            for route, cashpoint in route_stop_pairs:
                row[
                    q0 + q_i[(node, route, cashpoint)]
                ] += p.handling_cost_per_unit

        for node in realization_path:
            for cashpoint in cashpoints:
                key = (node, cashpoint)
                row[inv0 + state_i[key]] += p.holding_cost_per_unit_day
                row[safe0 + state_i[key]] += p.shortage_penalty_per_unit
                row[lost0 + state_i[key]] += p.cashout_penalty_per_unit

        row[eta_idx] = -1.0
        row[xi0 + leaf_i[leaf]] = -1.0

        ub_rows.append(row)
        ub_rhs.append(0.0)

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

    # Route variables binary.
    upper[z0:q0] = 1.0

    # Inventory capacity.
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
            f"multistage CVaR IRP failed: {result.message}"
        )

    x = result.x

    delivery_policy = pd.DataFrame(
        0.0,
        index=decision_nodes,
        columns=cashpoints,
    )
    route_rows = []

    for node in decision_nodes:
        for route in routes:
            if x[z0 + z_i[(node, route)]] <= 0.5:
                continue

            load = 0.0
            for cashpoint in catalog.loc[route, "stops"]:
                amount = x[q0 + q_i[(node, route, cashpoint)]]
                delivery_policy.loc[node, cashpoint] += amount
                load += amount

            route_rows.append(
                {
                    "node": node,
                    "stage": int(nodes.loc[node, "stage"]),
                    "day": tree.days[int(nodes.loc[node, "stage"])],
                    "probability": float(nodes.loc[node, "probability"]),
                    "route_id": route,
                    "sequence": catalog.loc[route, "sequence"],
                    "load": float(load),
                    "distance": float(catalog.loc[route, "distance"]),
                }
            )

    route_policy = pd.DataFrame(
        route_rows,
        columns=[
            "node",
            "stage",
            "day",
            "probability",
            "route_id",
            "sequence",
            "load",
            "distance",
        ],
    )

    inventory = pd.DataFrame(
        0.0,
        index=realization_nodes,
        columns=cashpoints,
    )
    cashout = pd.DataFrame(
        0.0,
        index=realization_nodes,
        columns=cashpoints,
    )
    safety = pd.DataFrame(
        0.0,
        index=realization_nodes,
        columns=cashpoints,
    )

    for node in realization_nodes:
        for cashpoint in cashpoints:
            key = (node, cashpoint)
            inventory.loc[node, cashpoint] = x[inv0 + state_i[key]]
            cashout.loc[node, cashpoint] = x[lost0 + state_i[key]]
            safety.loc[node, cashpoint] = x[safe0 + state_i[key]]

    def leaf_cost(leaf: str) -> float:
        path = _path_nodes(tree, leaf)
        total = 0.0

        for node in path:
            if node in decision_nodes:
                for route in routes:
                    total += (
                        route_cost(route)
                        * x[z0 + z_i[(node, route)]]
                    )
                for route, cashpoint in route_stop_pairs:
                    total += (
                        p.handling_cost_per_unit
                        * x[q0 + q_i[(node, route, cashpoint)]]
                    )

            if node in realization_nodes:
                for cashpoint in cashpoints:
                    key = (node, cashpoint)
                    total += (
                        p.holding_cost_per_unit_day
                        * x[inv0 + state_i[key]]
                    )
                    total += (
                        p.shortage_penalty_per_unit
                        * x[safe0 + state_i[key]]
                    )
                    total += (
                        p.cashout_penalty_per_unit
                        * x[lost0 + state_i[key]]
                    )

        return float(total)

    leaf_costs = pd.Series(
        {leaf: leaf_cost(leaf) for leaf in leaves},
        name="path_cost",
    )

    # Expected cost is probability-weighted terminal path cost.
    expected_cost = float(
        sum(
            float(nodes.loc[leaf, "probability"])
            * leaf_costs.loc[leaf]
            for leaf in leaves
        )
    )

    leaf_probabilities = nodes.loc[leaves, "probability"].astype(float)
    eta, cvar_cost = _weighted_var_cvar(
        leaf_costs,
        leaf_probabilities,
        cvar_alpha,
    )

    expected_cashout = float(
        sum(
            float(nodes.loc[node, "probability"])
            * cashout.loc[node].sum()
            for node in realization_nodes
        )
    )

    return MultistageCVaRResult(
        route_policy=route_policy,
        delivery_policy=delivery_policy,
        inventory=inventory,
        cashout=cashout,
        safety_shortfall=safety,
        leaf_costs=leaf_costs,
        expected_cost=expected_cost,
        var_threshold=eta,
        cvar_cost=cvar_cost,
        worst_leaf_cost=float(leaf_costs.max()),
        expected_cashout=expected_cashout,
        objective_value=float(result.fun),
        cvar_alpha=cvar_alpha,
        risk_aversion=risk_aversion,
    )


def compare_risk_attitudes(
    problem: CashSupplyChainProblem | None = None,
    tree: ScenarioTree | None = None,
    cvar_alpha: float = 0.90,
    risk_aversion: float = 0.75,
    neutral: MultistageCVaRResult | None = None,
    risk_averse: MultistageCVaRResult | None = None,
) -> tuple[pd.DataFrame, dict[str, MultistageCVaRResult]]:
    """Compare risk-neutral and CVaR-averse policies on one scenario tree.

    Precomputed solutions may be supplied to avoid repeated MILP solves.
    """
    p = problem or default_problem()
    tree = tree or generate_binary_scenario_tree(p)

    neutral = neutral or solve_multistage_cvar_irp(
        p,
        tree=tree,
        cvar_alpha=cvar_alpha,
        risk_aversion=0.0,
    )
    risk_averse = risk_averse or solve_multistage_cvar_irp(
        p,
        tree=tree,
        cvar_alpha=cvar_alpha,
        risk_aversion=risk_aversion,
    )

    rows = []
    for name, result in [
        ("risk_neutral", neutral),
        ("risk_averse", risk_averse),
    ]:
        rows.append(
            {
                "policy": name,
                "expected_cost": result.expected_cost,
                "cvar_cost": result.cvar_cost,
                "worst_leaf_cost": result.worst_leaf_cost,
                "expected_cashout": result.expected_cashout,
                "root_delivery": float(
                    result.delivery_policy.loc["root"].sum()
                ),
                "routes_operated_weighted": float(
                    sum(
                        result.route_policy.loc[
                            result.route_policy["node"] == node
                        ].shape[0]
                        * tree.nodes.loc[node, "probability"]
                        for node in result.delivery_policy.index
                    )
                ),
            }
        )

    comparison = pd.DataFrame(rows).set_index("policy")
    comparison["expected_cost_change_vs_neutral"] = (
        comparison["expected_cost"]
        - comparison.loc["risk_neutral", "expected_cost"]
    )
    comparison["cvar_improvement_vs_neutral"] = (
        comparison.loc["risk_neutral", "cvar_cost"]
        - comparison["cvar_cost"]
    )
    comparison["worst_cost_improvement_vs_neutral"] = (
        comparison.loc["risk_neutral", "worst_leaf_cost"]
        - comparison["worst_leaf_cost"]
    )

    return comparison, {
        "risk_neutral": neutral,
        "risk_averse": risk_averse,
    }
