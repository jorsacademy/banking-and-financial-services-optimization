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
