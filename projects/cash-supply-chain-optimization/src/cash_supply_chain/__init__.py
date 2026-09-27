"""Cash supply-chain optimization and simulation."""

from .model import (
    CashSupplyChainProblem,
    CashSupplyChainResult,
    default_problem,
    solve,
)

__all__ = [
    "CashSupplyChainProblem",
    "CashSupplyChainResult",
    "default_problem",
    "solve",
]

from .joint_irp import (
    JointIRPResult,
    compare_staged_and_joint,
    generate_route_catalog,
    solve_joint_irp,
)

__all__ += [
    "JointIRPResult",
    "compare_staged_and_joint",
    "generate_route_catalog",
    "solve_joint_irp",
]

from .stochastic_rolling_horizon import (
    RollingHorizonResult,
    StochasticHorizonResult,
    compare_rolling_policies,
    generate_demand_scenarios,
    generate_realized_demand,
    run_rolling_horizon,
    solve_stochastic_horizon,
)

__all__ += [
    "RollingHorizonResult",
    "StochasticHorizonResult",
    "compare_rolling_policies",
    "generate_demand_scenarios",
    "generate_realized_demand",
    "run_rolling_horizon",
    "solve_stochastic_horizon",
]

from .multistage_cvar_irp import (
    MultistageCVaRResult,
    ScenarioTree,
    compare_risk_attitudes,
    generate_binary_scenario_tree,
    solve_multistage_cvar_irp,
)

__all__ += [
    "MultistageCVaRResult",
    "ScenarioTree",
    "compare_risk_attitudes",
    "generate_binary_scenario_tree",
    "solve_multistage_cvar_irp",
]

from .scenario_reduction import (
    ScenarioReductionResult,
    leaf_path_vectors,
    reduce_scenario_tree,
    standardized_pairwise_distance,
)

__all__ += [
    "ScenarioReductionResult",
    "leaf_path_vectors",
    "reduce_scenario_tree",
    "standardized_pairwise_distance",
]

from .progressive_hedging import (
    ProgressiveHedgingResult,
    benchmark_progressive_hedging,
    run_progressive_hedging,
)

__all__ += [
    "ProgressiveHedgingResult",
    "benchmark_progressive_hedging",
    "run_progressive_hedging",
]

from .column_generation import (
    BranchAndPriceResult,
    ColumnGenerationResult,
    delivery_requirements_from_plan,
    solve_full_catalog_route_master,
    solve_route_branch_and_price,
    solve_route_column_generation,
)

__all__ += [
    "BranchAndPriceResult",
    "ColumnGenerationResult",
    "delivery_requirements_from_plan",
    "solve_full_catalog_route_master",
    "solve_route_branch_and_price",
    "solve_route_column_generation",
]
