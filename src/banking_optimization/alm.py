"""Synthetic asset-liability management optimization example.

The model is intentionally simplified. It demonstrates how a bank-style
balance-sheet allocation problem can be formulated as a linear program with
capital, liquidity, duration-gap, and scenario NII constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class ALMProblem:
    assets: pd.DataFrame
    funding: pd.DataFrame
    scenario_probabilities: pd.Series
    asset_rates: pd.DataFrame
    funding_rates: pd.DataFrame
    hedge_payoff: pd.Series
    balance_sheet: float = 1_000.0
    equity: float = 90.0
    minimum_capital_ratio: float = 0.11
    maximum_duration_gap_years: float = 0.75
    hedge_duration_years: float = -4.0
    maximum_hedge_notional: float = 250.0
    target_nii: float = 25.0
    shortfall_penalty: float = 2.5
    hedge_cost_rate: float = 0.001


@dataclass(frozen=True)
class ALMResult:
    asset_allocation: pd.Series
    funding_allocation: pd.Series
    hedge_notional: float
    scenario_nii: pd.Series
    expected_nii: float
    expected_shortfall: float
    capital_ratio: float
    liquidity_surplus: float
    duration_gap_years: float
    objective_value: float

    def to_dict(self) -> dict:
        return {
            "asset_allocation": self.asset_allocation.round(6).to_dict(),
            "funding_allocation": self.funding_allocation.round(6).to_dict(),
            "hedge_notional": round(self.hedge_notional, 6),
            "scenario_nii": self.scenario_nii.round(6).to_dict(),
            "expected_nii": round(self.expected_nii, 6),
            "expected_shortfall": round(self.expected_shortfall, 6),
            "capital_ratio": round(self.capital_ratio, 6),
            "liquidity_surplus": round(self.liquidity_surplus, 6),
            "duration_gap_years": round(self.duration_gap_years, 6),
            "objective_value": round(self.objective_value, 6),
        }


def default_problem() -> ALMProblem:
    """Return a deterministic synthetic ALM instance."""
    assets = pd.DataFrame(
        {
            "minimum": [50.0, 100.0, 150.0, 100.0],
            "maximum": [200.0, 400.0, 450.0, 400.0],
            "risk_weight": [0.00, 0.05, 0.35, 1.00],
            "liquidity_weight": [1.00, 0.90, 0.10, 0.00],
            "duration": [0.00, 3.00, 5.00, 2.50],
        },
        index=["cash", "government_bonds", "mortgages", "corporate_loans"],
    )

    funding = pd.DataFrame(
        {
            "minimum": [300.0, 100.0, 0.0],
            "maximum": [700.0, 400.0, 300.0],
            "runoff_rate": [0.05, 0.10, 0.35],
            "duration": [0.50, 1.50, 0.25],
        },
        index=["retail_deposits", "term_deposits", "wholesale_funding"],
    )

    scenarios = ["base", "rates_up", "rates_down", "funding_stress"]
    probabilities = pd.Series([0.40, 0.20, 0.20, 0.20], index=scenarios)

    asset_rates = pd.DataFrame(
        [
            [0.010, 0.035, 0.055, 0.065],
            [0.012, 0.040, 0.060, 0.072],
            [0.008, 0.030, 0.050, 0.058],
            [0.010, 0.032, 0.052, 0.060],
        ],
        index=scenarios,
        columns=assets.index,
    )

    funding_rates = pd.DataFrame(
        [
            [0.018, 0.028, 0.042],
            [0.025, 0.038, 0.055],
            [0.014, 0.022, 0.035],
            [0.028, 0.045, 0.065],
        ],
        index=scenarios,
        columns=funding.index,
    )

    hedge_payoff = pd.Series(
        [0.000, 0.015, -0.010, 0.010],
        index=scenarios,
        name="hedge_payoff_per_unit",
    )

    return ALMProblem(
        assets=assets,
        funding=funding,
        scenario_probabilities=probabilities,
        asset_rates=asset_rates,
        funding_rates=funding_rates,
        hedge_payoff=hedge_payoff,
    )


def _validate(problem: ALMProblem) -> None:
    if problem.balance_sheet <= 0:
        raise ValueError("balance_sheet must be positive")
    if problem.minimum_capital_ratio <= 0:
        raise ValueError("minimum_capital_ratio must be positive")
    if not np.isclose(problem.scenario_probabilities.sum(), 1.0):
        raise ValueError("scenario probabilities must sum to one")

    scenarios = list(problem.scenario_probabilities.index)
    if list(problem.asset_rates.index) != scenarios:
        raise ValueError("asset-rate scenarios must match scenario probabilities")
    if list(problem.funding_rates.index) != scenarios:
        raise ValueError("funding-rate scenarios must match scenario probabilities")
    if list(problem.hedge_payoff.index) != scenarios:
        raise ValueError("hedge-payoff scenarios must match scenario probabilities")

    if set(problem.asset_rates.columns) != set(problem.assets.index):
        raise ValueError("asset-rate columns must match assets")
    if set(problem.funding_rates.columns) != set(problem.funding.index):
        raise ValueError("funding-rate columns must match funding sources")


def solve(problem: ALMProblem | None = None) -> ALMResult:
    """Solve the ALM linear program with SciPy/HiGHS."""
    p = problem or default_problem()
    _validate(p)

    assets = list(p.assets.index)
    funding = list(p.funding.index)
    scenarios = list(p.scenario_probabilities.index)

    n_a = len(assets)
    n_f = len(funding)
    n_s = len(scenarios)

    # Variable order:
    # asset allocations | funding allocations | hedge+ | hedge- | NII shortfalls
    idx_hp = n_a + n_f
    idx_hn = idx_hp + 1
    idx_z0 = idx_hn + 1
    n_vars = idx_z0 + n_s

    probabilities = p.scenario_probabilities.to_numpy(dtype=float)
    mean_asset_rates = probabilities @ p.asset_rates.loc[scenarios, assets].to_numpy()
    mean_funding_rates = probabilities @ p.funding_rates.loc[scenarios, funding].to_numpy()
    mean_hedge_payoff = float(probabilities @ p.hedge_payoff.loc[scenarios].to_numpy())

    c = np.zeros(n_vars)
    c[:n_a] = -mean_asset_rates
    c[n_a : n_a + n_f] = mean_funding_rates
    c[idx_hp] = -mean_hedge_payoff + p.hedge_cost_rate
    c[idx_hn] = mean_hedge_payoff + p.hedge_cost_rate
    c[idx_z0:] = p.shortfall_penalty * probabilities

    # Balance-sheet identities.
    a_eq = np.zeros((2, n_vars))
    b_eq = np.array([p.balance_sheet, p.balance_sheet], dtype=float)
    a_eq[0, :n_a] = 1.0
    a_eq[1, n_a : n_a + n_f] = 1.0

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    # Capital adequacy proxy: equity / RWA >= minimum ratio.
    row = np.zeros(n_vars)
    row[:n_a] = p.assets.loc[assets, "risk_weight"].to_numpy()
    a_ub.append(row)
    b_ub.append(p.equity / p.minimum_capital_ratio)

    # Liquidity proxy:
    # weighted liquid assets >= stressed funding runoff.
    row = np.zeros(n_vars)
    row[:n_a] = -p.assets.loc[assets, "liquidity_weight"].to_numpy()
    row[n_a : n_a + n_f] = p.funding.loc[funding, "runoff_rate"].to_numpy()
    a_ub.append(row)
    b_ub.append(0.0)

    # Absolute duration-gap constraint.
    duration_row = np.zeros(n_vars)
    duration_row[:n_a] = p.assets.loc[assets, "duration"].to_numpy()
    duration_row[n_a : n_a + n_f] = -p.funding.loc[funding, "duration"].to_numpy()
    duration_row[idx_hp] = p.hedge_duration_years
    duration_row[idx_hn] = -p.hedge_duration_years
    max_gap_amount = p.maximum_duration_gap_years * p.balance_sheet
    a_ub.extend([duration_row, -duration_row])
    b_ub.extend([max_gap_amount, max_gap_amount])

    # Scenario NII shortfall:
    # z_s >= target_nii - NII_s.
    for s_idx, scenario in enumerate(scenarios):
        row = np.zeros(n_vars)
        row[:n_a] = -p.asset_rates.loc[scenario, assets].to_numpy()
        row[n_a : n_a + n_f] = p.funding_rates.loc[scenario, funding].to_numpy()
        payoff = float(p.hedge_payoff.loc[scenario])
        row[idx_hp] = -payoff
        row[idx_hn] = payoff
        row[idx_z0 + s_idx] = -1.0
        a_ub.append(row)
        b_ub.append(-p.target_nii)

    bounds: list[tuple[float | None, float | None]] = []
    bounds.extend(
        zip(
            p.assets.loc[assets, "minimum"].astype(float),
            p.assets.loc[assets, "maximum"].astype(float),
        )
    )
    bounds.extend(
        zip(
            p.funding.loc[funding, "minimum"].astype(float),
            p.funding.loc[funding, "maximum"].astype(float),
        )
    )
    bounds.append((0.0, p.maximum_hedge_notional))
    bounds.append((0.0, p.maximum_hedge_notional))
    bounds.extend([(0.0, None)] * n_s)

    result = linprog(
        c,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"ALM optimization failed: {result.message}")

    x = result.x
    asset_allocation = pd.Series(x[:n_a], index=assets, name="allocation")
    funding_allocation = pd.Series(
        x[n_a : n_a + n_f], index=funding, name="allocation"
    )
    hedge_notional = float(x[idx_hp] - x[idx_hn])

    scenario_nii = (
        p.asset_rates.loc[scenarios, assets].mul(asset_allocation, axis=1).sum(axis=1)
        - p.funding_rates.loc[scenarios, funding].mul(funding_allocation, axis=1).sum(axis=1)
        + p.hedge_payoff.loc[scenarios] * hedge_notional
    )
    expected_nii = float((scenario_nii * p.scenario_probabilities).sum())
    shortfalls = (p.target_nii - scenario_nii).clip(lower=0.0)
    expected_shortfall = float((shortfalls * p.scenario_probabilities).sum())

    rwa = float(
        (asset_allocation * p.assets.loc[assets, "risk_weight"]).sum()
    )
    capital_ratio = np.inf if np.isclose(rwa, 0.0) else p.equity / rwa

    liquid_assets = float(
        (asset_allocation * p.assets.loc[assets, "liquidity_weight"]).sum()
    )
    runoff = float(
        (funding_allocation * p.funding.loc[funding, "runoff_rate"]).sum()
    )
    liquidity_surplus = liquid_assets - runoff

    duration_amount = float(
        (asset_allocation * p.assets.loc[assets, "duration"]).sum()
        - (funding_allocation * p.funding.loc[funding, "duration"]).sum()
        + p.hedge_duration_years * hedge_notional
    )
    duration_gap_years = duration_amount / p.balance_sheet

    return ALMResult(
        asset_allocation=asset_allocation,
        funding_allocation=funding_allocation,
        hedge_notional=hedge_notional,
        scenario_nii=scenario_nii,
        expected_nii=expected_nii,
        expected_shortfall=expected_shortfall,
        capital_ratio=float(capital_ratio),
        liquidity_surplus=liquidity_surplus,
        duration_gap_years=duration_gap_years,
        objective_value=float(result.fun),
    )


def main() -> None:
    result = solve()
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
