"""Scenario reduction for multistage cash-demand trees.

A greedy forward-selection procedure chooses representative terminal demand
paths. Omitted leaf probability mass is reassigned to the closest selected
leaf, and the reduced multistage tree is rebuilt from the retained histories.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .multistage_cvar_irp import ScenarioTree


@dataclass(frozen=True)
class ScenarioReductionResult:
    selected_leaves: tuple[str, ...]
    original_probabilities: pd.Series
    reduced_probabilities: pd.Series
    assignment: pd.Series
    path_vectors: pd.DataFrame
    standardized_distance: pd.DataFrame
    distortion: float
    reduced_tree: ScenarioTree

    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "original_leaves": len(self.original_probabilities),
                "selected_leaves": len(self.selected_leaves),
                "original_nodes": len(
                    set(
                        node
                        for leaf in self.original_probabilities.index
                        for node in _path_nodes_from_leaf_name(leaf)
                    )
                ),
                "reduced_nodes": len(self.reduced_tree.nodes),
                "distortion": self.distortion,
            }
        )


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


def _path_nodes_from_leaf_name(leaf: str) -> list[str]:
    """Best-effort path count helper for underscore-delimited binary names."""
    parts = leaf.split("_")
    nodes = ["root"]
    current = ""
    for part in parts:
        current = part if not current else f"{current}_{part}"
        nodes.append(current)
    return nodes


def leaf_path_vectors(tree: ScenarioTree) -> tuple[pd.DataFrame, pd.Series]:
    """Flatten every terminal demand path into one vector."""
    horizon = len(tree.days)
    leaves = list(
        tree.nodes.index[tree.nodes["stage"] == horizon]
    )

    cashpoints = list(tree.demand.columns)
    columns = [
        f"stage{stage}:{cashpoint}"
        for stage in range(1, horizon + 1)
        for cashpoint in cashpoints
    ]

    rows = {}
    probabilities = {}

    for leaf in leaves:
        path = _path_nodes(tree, leaf)
        realized_nodes = [
            node
            for node in path
            if node != "root"
        ]

        vector = []
        for node in realized_nodes:
            vector.extend(
                tree.demand.loc[node, cashpoints].to_numpy(float)
            )

        rows[leaf] = vector
        probabilities[leaf] = float(
            tree.nodes.loc[leaf, "probability"]
        )

    vectors = pd.DataFrame.from_dict(
        rows,
        orient="index",
        columns=columns,
    )
    probs = pd.Series(probabilities, name="probability")
    probs = probs / probs.sum()
    return vectors, probs


def standardized_pairwise_distance(
    vectors: pd.DataFrame,
    probabilities: pd.Series,
) -> pd.DataFrame:
    """Weighted-standardized Euclidean distance between leaf demand paths."""
    p = probabilities.loc[vectors.index].to_numpy(float)
    p = p / p.sum()
    x = vectors.to_numpy(float)

    mean = np.sum(x * p[:, None], axis=0)
    variance = np.sum(((x - mean) ** 2) * p[:, None], axis=0)
    scale = np.sqrt(variance)

    # Demand dimensions with essentially no dispersion should not dominate
    # due to numerical division.
    scale = np.where(scale > 1e-9, scale, 1.0)
    standardized = (x - mean) / scale

    diff = standardized[:, None, :] - standardized[None, :, :]
    distance = np.sqrt(np.sum(diff**2, axis=2))

    return pd.DataFrame(
        distance,
        index=vectors.index,
        columns=vectors.index,
    )


def _rebuild_reduced_tree(
    tree: ScenarioTree,
    selected_leaves: list[str],
    reduced_probabilities: pd.Series,
) -> ScenarioTree:
    keep_nodes = {"root"}
    leaf_paths: dict[str, list[str]] = {}

    for leaf in selected_leaves:
        path = _path_nodes(tree, leaf)
        leaf_paths[leaf] = path
        keep_nodes.update(path)

    records = []

    for node in tree.nodes.index:
        if node not in keep_nodes:
            continue

        if node == "root":
            probability = 1.0
        else:
            probability = float(
                sum(
                    reduced_probabilities.loc[leaf]
                    for leaf, path in leaf_paths.items()
                    if node in path
                )
            )

        parent = tree.nodes.loc[node, "parent"]
        records.append(
            {
                "node": node,
                "parent": parent,
                "stage": int(tree.nodes.loc[node, "stage"]),
                "day": tree.nodes.loc[node, "day"],
                "probability": probability,
            }
        )

    nodes = pd.DataFrame(records).set_index("node")

    demand_nodes = [
        node for node in nodes.index if node != "root"
    ]
    demand = tree.demand.loc[demand_nodes].copy()

    return ScenarioTree(
        nodes=nodes,
        demand=demand,
        days=tree.days,
    )


def reduce_scenario_tree(
    tree: ScenarioTree,
    target_leaves: int,
) -> ScenarioReductionResult:
    """Reduce terminal scenarios by greedy probability-weighted forward selection."""
    vectors, probabilities = leaf_path_vectors(tree)
    leaves = list(vectors.index)

    if target_leaves < 1:
        raise ValueError("target_leaves must be positive")
    if target_leaves > len(leaves):
        raise ValueError("target_leaves cannot exceed original leaf count")

    distance = standardized_pairwise_distance(
        vectors,
        probabilities,
    )

    if target_leaves == len(leaves):
        selected = leaves.copy()
    else:
        selected: list[str] = []
        nearest = pd.Series(
            np.inf,
            index=leaves,
            dtype=float,
        )

        for _ in range(target_leaves):
            best_candidate = None
            best_score = float("inf")

            for candidate in leaves:
                if candidate in selected:
                    continue

                candidate_distance = distance[candidate]
                improved = np.minimum(
                    nearest.to_numpy(),
                    candidate_distance.to_numpy(),
                )
                score = float(
                    np.sum(
                        probabilities.to_numpy() * improved
                    )
                )

                if score < best_score - 1e-12:
                    best_score = score
                    best_candidate = candidate

            if best_candidate is None:
                raise RuntimeError("scenario reduction selection failed")

            selected.append(best_candidate)
            nearest = pd.Series(
                np.minimum(
                    nearest.to_numpy(),
                    distance[best_candidate].to_numpy(),
                ),
                index=leaves,
            )

    assignment_map = {}
    for leaf in leaves:
        representative = min(
            selected,
            key=lambda candidate: (
                float(distance.loc[leaf, candidate]),
                selected.index(candidate),
            ),
        )
        assignment_map[leaf] = representative

    assignment = pd.Series(
        assignment_map,
        name="representative",
    )

    reduced_probabilities = pd.Series(
        0.0,
        index=selected,
        name="probability",
    )
    for leaf, representative in assignment.items():
        reduced_probabilities.loc[representative] += (
            probabilities.loc[leaf]
        )

    distortion = float(
        sum(
            probabilities.loc[leaf]
            * distance.loc[leaf, assignment.loc[leaf]]
            for leaf in leaves
        )
    )

    reduced_tree = _rebuild_reduced_tree(
        tree,
        selected,
        reduced_probabilities,
    )

    return ScenarioReductionResult(
        selected_leaves=tuple(selected),
        original_probabilities=probabilities,
        reduced_probabilities=reduced_probabilities,
        assignment=assignment,
        path_vectors=vectors,
        standardized_distance=distance,
        distortion=distortion,
        reduced_tree=reduced_tree,
    )
