"""Multi-period stochastic asset-liability management on a scenario tree.

This module extends the one-period ALM example with:
- a three-period scenario tree;
- adaptive balance-sheet and hedge decisions;
- non-anticipativity through node-based decisions;
- node-specific asset/funding rates and runoff stress;
- rebalancing costs between parent/child nodes;
- a static-policy baseline for measuring the value of adaptivity.

All data are synthetic and the risk/liquidity metrics are educational proxies.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class StochasticALMProblem:
    assets: pd.DataFrame
    funding: pd.DataFrame
    tree: pd.DataFrame
    asset_rates: pd.DataFrame
    funding_rates: pd.DataFrame
    hedge_payoff: pd.Series
    runoff_multiplier: pd.Series
    balance_sheet: float = 1_000.0
    equity: float = 90.0
    minimum_capital_ratio: float = 0.11
    maximum_duration_gap_years: float = 0.80
    hedge_duration_years: float = -4.0
    maximum_hedge_notional: float = 250.0
    nii_target_by_period: tuple[float, float, float] = (22.0, 24.0, 25.0)
    shortfall_penalty: float = 2.0
    asset_rebalance_cost: float = 0.0015
    funding_rebalance_cost: float = 0.0010
    hedge_cost_rate: float = 0.0008


@dataclass(frozen=True)
class StochasticALMResult:
    asset_policy: pd.DataFrame
    funding_policy: pd.DataFrame
    hedge_policy: pd.Series
    node_nii: pd.Series
    node_shortfall: pd.Series
    node_capital_ratio: pd.Series
    node_liquidity_surplus: pd.Series
    node_duration_gap: pd.Series
    expected_total_nii: float
    expected_total_shortfall: float
    expected_rebalance_cost: float
    expected_hedge_cost: float
    objective_value: float
    adaptive: bool

    def to_dict(self) -> dict:
        return {
            "adaptive": self.adaptive,
            "asset_policy": self.asset_policy.round(6).to_dict(orient="index"),
            "funding_policy": self.funding_policy.round(6).to_dict(orient="index"),
            "hedge_policy": self.hedge_policy.round(6).to_dict(),
            "node_nii": self.node_nii.round(6).to_dict(),
            "node_shortfall": self.node_shortfall.round(6).to_dict(),
            "node_capital_ratio": self.node_capital_ratio.round(6).to_dict(),
            "node_liquidity_surplus": self.node_liquidity_surplus.round(6).to_dict(),
            "node_duration_gap": self.node_duration_gap.round(6).to_dict(),
            "expected_total_nii": round(self.expected_total_nii, 6),
            "expected_total_shortfall": round(self.expected_total_shortfall, 6),
            "expected_rebalance_cost": round(self.expected_rebalance_cost, 6),
            "expected_hedge_cost": round(self.expected_hedge_cost, 6),
            "objective_value": round(self.objective_value, 6),
        }


def default_stochastic_problem() -> StochasticALMProblem:
    assets = pd.DataFrame(
        {
            "minimum": [60.0, 100.0, 150.0, 100.0],
            "maximum": [220.0, 420.0, 450.0, 420.0],
            "risk_weight": [0.00, 0.05, 0.35, 1.00],
            "liquidity_weight": [1.00, 0.90, 0.10, 0.00],
            "duration": [0.00, 3.00, 5.00, 2.50],
        },
        index=["cash", "government_bonds", "mortgages", "corporate_loans"],
    )

    funding = pd.DataFrame(
        {
            "minimum": [300.0, 100.0, 0.0],
            "maximum": [700.0, 420.0, 320.0],
            "base_runoff": [0.05, 0.10, 0.35],
            "duration": [0.50, 1.50, 0.25],
        },
        index=["retail_deposits", "term_deposits", "wholesale_funding"],
    )

    tree = pd.DataFrame(
        [
            ("root", None, 0, 1.00),
            ("up", "root", 1, 0.50),
            ("down", "root", 1, 0.50),
            ("up_up", "up", 2, 0.25),
            ("up_down", "up", 2, 0.25),
            ("down_up", "down", 2, 0.25),
            ("down_down", "down", 2, 0.25),
        ],
        columns=["node", "parent", "period", "probability"],
    ).set_index("node")

    # Rates are annualized synthetic bucket rates observed at each decision node.
    asset_rates = pd.DataFrame(
        [
            [0.010, 0.035, 0.055, 0.065],
            [0.013, 0.043, 0.063, 0.075],
            [0.008, 0.029, 0.049, 0.057],
            [0.015, 0.050, 0.069, 0.082],
            [0.010, 0.036, 0.056, 0.066],
            [0.011, 0.038, 0.058, 0.069],
            [0.006, 0.024, 0.044, 0.052],
        ],
        index=tree.index,
        columns=assets.index,
    )

    funding_rates = pd.DataFrame(
        [
            [0.018, 0.028, 0.042],
            [0.026, 0.039, 0.056],
            [0.014, 0.022, 0.035],
            [0.032, 0.047, 0.066],
            [0.021, 0.033, 0.050],
            [0.020, 0.031, 0.047],
            [0.011, 0.018, 0.030],
        ],
        index=tree.index,
        columns=funding.index,
    )

    hedge_payoff = pd.Series(
        [0.000, 0.014, -0.010, 0.020, 0.003, 0.004, -0.016],
        index=tree.index,
        name="hedge_payoff_per_unit",
    )

    runoff_multiplier = pd.Series(
        [1.00, 1.10, 0.95, 1.25, 1.15, 1.05, 0.90],
        index=tree.index,
        name="runoff_multiplier",
    )

    return StochasticALMProblem(
        assets=assets,
        funding=funding,
        tree=tree,
        asset_rates=asset_rates,
        funding_rates=funding_rates,
        hedge_payoff=hedge_payoff,
        runoff_multiplier=runoff_multiplier,
    )


def _validate(p: StochasticALMProblem) -> None:
    nodes = list(p.tree.index)
    if p.tree.index[0] != "root":
        raise ValueError("first tree node must be root")
    if not np.isclose(float(p.tree.loc["root", "probability"]), 1.0):
        raise ValueError("root probability must equal one")
    if set(p.asset_rates.index) != set(nodes):
        raise ValueError("asset rates must be indexed by all tree nodes")
    if set(p.funding_rates.index) != set(nodes):
        raise ValueError("funding rates must be indexed by all tree nodes")
    if set(p.hedge_payoff.index) != set(nodes):
        raise ValueError("hedge payoff must be indexed by all tree nodes")
    if set(p.runoff_multiplier.index) != set(nodes):
        raise ValueError("runoff multipliers must be indexed by all tree nodes")
    if set(p.asset_rates.columns) != set(p.assets.index):
        raise ValueError("asset-rate columns must match assets")
    if set(p.funding_rates.columns) != set(p.funding.index):
        raise ValueError("funding-rate columns must match funding sources")

    periods = sorted(p.tree["period"].unique())
    if periods != list(range(len(p.nii_target_by_period))):
        raise ValueError("NII targets must cover every period")

    # Child probabilities must sum to their parent probability.
    for parent in nodes:
        children = p.tree.index[p.tree["parent"] == parent].tolist()
        if children:
            child_prob = float(p.tree.loc[children, "probability"].sum())
            parent_prob = float(p.tree.loc[parent, "probability"])
            if not np.isclose(child_prob, parent_prob):
                raise ValueError(f"child probabilities do not sum to {parent}")


def _build_and_solve(
    p: StochasticALMProblem,
    adaptive: bool,
) -> tuple[np.ndarray, dict]:
    _validate(p)

    nodes = list(p.tree.index)
    assets = list(p.assets.index)
    funding = list(p.funding.index)
    nonroot = [n for n in nodes if n != "root"]

    n_n = len(nodes)
    n_a = len(assets)
    n_f = len(funding)

    # Variable blocks:
    # x[n,a], y[n,f], hedge_plus[n], hedge_minus[n], shortfall[n],
    # asset_turnover_plus/minus[nonroot,a],
    # funding_turnover_plus/minus[nonroot,f]
    x0 = 0
    y0 = x0 + n_n * n_a
    hp0 = y0 + n_n * n_f
    hn0 = hp0 + n_n
    z0 = hn0 + n_n
    atp0 = z0 + n_n
    atm0 = atp0 + len(nonroot) * n_a
    ftp0 = atm0 + len(nonroot) * n_a
    ftm0 = ftp0 + len(nonroot) * n_f
    n_vars = ftm0 + len(nonroot) * n_f

    node_i = {n: i for i, n in enumerate(nodes)}
    asset_i = {a: i for i, a in enumerate(assets)}
    funding_i = {f: i for i, f in enumerate(funding)}
    nonroot_i = {n: i for i, n in enumerate(nonroot)}

    def ix(n: str, a: str) -> int:
        return x0 + node_i[n] * n_a + asset_i[a]

    def iy(n: str, f: str) -> int:
        return y0 + node_i[n] * n_f + funding_i[f]

    def ihp(n: str) -> int:
        return hp0 + node_i[n]

    def ihn(n: str) -> int:
        return hn0 + node_i[n]

    def iz(n: str) -> int:
        return z0 + node_i[n]

    def iatp(n: str, a: str) -> int:
        return atp0 + nonroot_i[n] * n_a + asset_i[a]

    def iatm(n: str, a: str) -> int:
        return atm0 + nonroot_i[n] * n_a + asset_i[a]

    def iftp(n: str, f: str) -> int:
        return ftp0 + nonroot_i[n] * n_f + funding_i[f]

    def iftm(n: str, f: str) -> int:
        return ftm0 + nonroot_i[n] * n_f + funding_i[f]

    c = np.zeros(n_vars)

    # Expected multi-period NII and downside penalties.
    for node in nodes:
        prob = float(p.tree.loc[node, "probability"])
        for asset in assets:
            c[ix(node, asset)] -= prob * float(p.asset_rates.loc[node, asset])
        for source in funding:
            c[iy(node, source)] += prob * float(p.funding_rates.loc[node, source])

        payoff = float(p.hedge_payoff.loc[node])
        c[ihp(node)] += prob * (-payoff + p.hedge_cost_rate)
        c[ihn(node)] += prob * (payoff + p.hedge_cost_rate)
        c[iz(node)] += prob * p.shortfall_penalty

    # Expected rebalancing cost is charged at child-node probability.
    for node in nonroot:
        prob = float(p.tree.loc[node, "probability"])
        for asset in assets:
            c[iatp(node, asset)] += prob * p.asset_rebalance_cost
            c[iatm(node, asset)] += prob * p.asset_rebalance_cost
        for source in funding:
            c[iftp(node, source)] += prob * p.funding_rebalance_cost
            c[iftm(node, source)] += prob * p.funding_rebalance_cost

    a_eq: list[np.ndarray] = []
    b_eq: list[float] = []

    # Each node has a balanced asset and funding side.
    for node in nodes:
        row = np.zeros(n_vars)
        for asset in assets:
            row[ix(node, asset)] = 1.0
        a_eq.append(row)
        b_eq.append(p.balance_sheet)

        row = np.zeros(n_vars)
        for source in funding:
            row[iy(node, source)] = 1.0
        a_eq.append(row)
        b_eq.append(p.balance_sheet)

    # Turnover identities relative to each parent node.
    for node in nonroot:
        parent = str(p.tree.loc[node, "parent"])
        for asset in assets:
            row = np.zeros(n_vars)
            row[ix(node, asset)] = 1.0
            row[ix(parent, asset)] = -1.0
            row[iatp(node, asset)] = -1.0
            row[iatm(node, asset)] = 1.0
            a_eq.append(row)
            b_eq.append(0.0)

        for source in funding:
            row = np.zeros(n_vars)
            row[iy(node, source)] = 1.0
            row[iy(parent, source)] = -1.0
            row[iftp(node, source)] = -1.0
            row[iftm(node, source)] = 1.0
            a_eq.append(row)
            b_eq.append(0.0)

    # Static baseline: freeze allocations and hedge to root at every future node.
    if not adaptive:
        for node in nonroot:
            for asset in assets:
                row = np.zeros(n_vars)
                row[ix(node, asset)] = 1.0
                row[ix("root", asset)] = -1.0
                a_eq.append(row)
                b_eq.append(0.0)

            for source in funding:
                row = np.zeros(n_vars)
                row[iy(node, source)] = 1.0
                row[iy("root", source)] = -1.0
                a_eq.append(row)
                b_eq.append(0.0)

            row = np.zeros(n_vars)
            row[ihp(node)] = 1.0
            row[ihn(node)] = -1.0
            row[ihp("root")] = -1.0
            row[ihn("root")] = 1.0
            a_eq.append(row)
            b_eq.append(0.0)

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    for node in nodes:
        # Capital adequacy proxy.
        row = np.zeros(n_vars)
        for asset in assets:
            row[ix(node, asset)] = float(p.assets.loc[asset, "risk_weight"])
        a_ub.append(row)
        b_ub.append(p.equity / p.minimum_capital_ratio)

        # Liquidity proxy with node-specific runoff stress.
        row = np.zeros(n_vars)
        for asset in assets:
            row[ix(node, asset)] = -float(p.assets.loc[asset, "liquidity_weight"])
        multiplier = float(p.runoff_multiplier.loc[node])
        for source in funding:
            row[iy(node, source)] = (
                multiplier * float(p.funding.loc[source, "base_runoff"])
            )
        a_ub.append(row)
        b_ub.append(0.0)

        # Absolute duration-gap constraint.
        row = np.zeros(n_vars)
        for asset in assets:
            row[ix(node, asset)] = float(p.assets.loc[asset, "duration"])
        for source in funding:
            row[iy(node, source)] = -float(p.funding.loc[source, "duration"])
        row[ihp(node)] = p.hedge_duration_years
        row[ihn(node)] = -p.hedge_duration_years
        max_gap = p.maximum_duration_gap_years * p.balance_sheet
        a_ub.extend([row, -row])
        b_ub.extend([max_gap, max_gap])

        # Scenario-node NII target shortfall.
        period = int(p.tree.loc[node, "period"])
        target = float(p.nii_target_by_period[period])
        row = np.zeros(n_vars)
        for asset in assets:
            row[ix(node, asset)] = -float(p.asset_rates.loc[node, asset])
        for source in funding:
            row[iy(node, source)] = float(p.funding_rates.loc[node, source])
        payoff = float(p.hedge_payoff.loc[node])
        row[ihp(node)] = -payoff
        row[ihn(node)] = payoff
        row[iz(node)] = -1.0
        a_ub.append(row)
        b_ub.append(-target)

    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * n_vars

    for node in nodes:
        for asset in assets:
            bounds[ix(node, asset)] = (
                float(p.assets.loc[asset, "minimum"]),
                float(p.assets.loc[asset, "maximum"]),
            )
        for source in funding:
            bounds[iy(node, source)] = (
                float(p.funding.loc[source, "minimum"]),
                float(p.funding.loc[source, "maximum"]),
            )
        bounds[ihp(node)] = (0.0, p.maximum_hedge_notional)
        bounds[ihn(node)] = (0.0, p.maximum_hedge_notional)
        bounds[iz(node)] = (0.0, None)

    result = linprog(
        c,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        mode = "adaptive" if adaptive else "static"
        raise RuntimeError(f"{mode} stochastic ALM failed: {result.message}")

    meta = {
        "nodes": nodes,
        "assets": assets,
        "funding": funding,
        "nonroot": nonroot,
        "ix": ix,
        "iy": iy,
        "ihp": ihp,
        "ihn": ihn,
        "iz": iz,
        "iatp": iatp,
        "iatm": iatm,
        "iftp": iftp,
        "iftm": iftm,
    }
    return result.x, {**meta, "objective": float(result.fun)}


def solve_stochastic(
    problem: StochasticALMProblem | None = None,
    adaptive: bool = True,
) -> StochasticALMResult:
    p = problem or default_stochastic_problem()
    x, m = _build_and_solve(p, adaptive=adaptive)

    nodes = m["nodes"]
    assets = m["assets"]
    funding = m["funding"]
    nonroot = m["nonroot"]

    asset_policy = pd.DataFrame(
        {
            asset: [x[m["ix"](node, asset)] for node in nodes]
            for asset in assets
        },
        index=nodes,
    )
    funding_policy = pd.DataFrame(
        {
            source: [x[m["iy"](node, source)] for node in nodes]
            for source in funding
        },
        index=nodes,
    )
    hedge_policy = pd.Series(
        [
            x[m["ihp"](node)] - x[m["ihn"](node)]
            for node in nodes
        ],
        index=nodes,
        name="hedge_notional",
    )
    node_shortfall = pd.Series(
        [x[m["iz"](node)] for node in nodes],
        index=nodes,
        name="nii_shortfall",
    )

    node_nii = pd.Series(index=nodes, dtype=float)
    capital_ratio = pd.Series(index=nodes, dtype=float)
    liquidity_surplus = pd.Series(index=nodes, dtype=float)
    duration_gap = pd.Series(index=nodes, dtype=float)

    for node in nodes:
        node_nii.loc[node] = float(
            (asset_policy.loc[node] * p.asset_rates.loc[node, assets]).sum()
            - (funding_policy.loc[node] * p.funding_rates.loc[node, funding]).sum()
            + hedge_policy.loc[node] * p.hedge_payoff.loc[node]
        )

        rwa = float(
            (asset_policy.loc[node] * p.assets.loc[assets, "risk_weight"]).sum()
        )
        capital_ratio.loc[node] = np.inf if np.isclose(rwa, 0.0) else p.equity / rwa

        liquid_assets = float(
            (
                asset_policy.loc[node]
                * p.assets.loc[assets, "liquidity_weight"]
            ).sum()
        )
        runoff = float(
            (
                funding_policy.loc[node]
                * p.funding.loc[funding, "base_runoff"]
                * p.runoff_multiplier.loc[node]
            ).sum()
        )
        liquidity_surplus.loc[node] = liquid_assets - runoff

        duration_amount = float(
            (asset_policy.loc[node] * p.assets.loc[assets, "duration"]).sum()
            - (funding_policy.loc[node] * p.funding.loc[funding, "duration"]).sum()
            + p.hedge_duration_years * hedge_policy.loc[node]
        )
        duration_gap.loc[node] = duration_amount / p.balance_sheet

    probabilities = p.tree.loc[nodes, "probability"]
    expected_total_nii = float((node_nii * probabilities).sum())
    expected_total_shortfall = float((node_shortfall * probabilities).sum())

    expected_rebalance_cost = 0.0
    for node in nonroot:
        prob = float(p.tree.loc[node, "probability"])
        asset_turnover = sum(
            x[m["iatp"](node, a)] + x[m["iatm"](node, a)]
            for a in assets
        )
        funding_turnover = sum(
            x[m["iftp"](node, f)] + x[m["iftm"](node, f)]
            for f in funding
        )
        expected_rebalance_cost += prob * (
            p.asset_rebalance_cost * asset_turnover
            + p.funding_rebalance_cost * funding_turnover
        )

    expected_hedge_cost = float(
        sum(
            p.tree.loc[node, "probability"]
            * p.hedge_cost_rate
            * (
                x[m["ihp"](node)]
                + x[m["ihn"](node)]
            )
            for node in nodes
        )
    )

    return StochasticALMResult(
        asset_policy=asset_policy,
        funding_policy=funding_policy,
        hedge_policy=hedge_policy,
        node_nii=node_nii,
        node_shortfall=node_shortfall,
        node_capital_ratio=capital_ratio,
        node_liquidity_surplus=liquidity_surplus,
        node_duration_gap=duration_gap,
        expected_total_nii=expected_total_nii,
        expected_total_shortfall=expected_total_shortfall,
        expected_rebalance_cost=float(expected_rebalance_cost),
        expected_hedge_cost=expected_hedge_cost,
        objective_value=float(m["objective"]),
        adaptive=adaptive,
    )


def compare_policies(
    problem: StochasticALMProblem | None = None,
) -> pd.DataFrame:
    p = problem or default_stochastic_problem()
    adaptive = solve_stochastic(p, adaptive=True)
    static = solve_stochastic(p, adaptive=False)

    rows = []
    for name, result in [("adaptive", adaptive), ("static", static)]:
        rows.append(
            {
                "policy": name,
                "expected_total_nii": result.expected_total_nii,
                "expected_total_shortfall": result.expected_total_shortfall,
                "expected_rebalance_cost": result.expected_rebalance_cost,
                "expected_hedge_cost": result.expected_hedge_cost,
                "objective_value": result.objective_value,
            }
        )

    comparison = pd.DataFrame(rows).set_index("policy")
    comparison["objective_improvement_vs_static"] = (
        comparison.loc["static", "objective_value"]
        - comparison["objective_value"]
    )
    return comparison


def main() -> None:
    result = solve_stochastic()
    comparison = compare_policies()
    payload = {
        "adaptive_result": result.to_dict(),
        "policy_comparison": comparison.round(6).to_dict(orient="index"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
