from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class LineageDispersion:
    parent_index: int
    total_weight: float
    dominant_child_index: int | None
    dominant_fraction: float | None
    effective_child_count: float | None
    shannon_entropy: float | None
    normalized_entropy: float | None


@dataclass(frozen=True)
class IntervalLineageStep:
    parent_index: int
    child_index: int | None
    winner_lower_bound: float | None
    strongest_competitor_upper_bound: float | None
    robust_gap: float | None
    identifiable: bool


@dataclass(frozen=True)
class LineageFlowStep:
    transition_index: int
    descendant_weights: tuple[float, ...]
    surviving_mass: float
    extinguished_mass_increment: float


@dataclass(frozen=True)
class SourceMobiusDecomposition:
    baseline: np.ndarray
    effects: dict[tuple[str, ...], np.ndarray]
    full_response: np.ndarray
    reconstructed_full_response: np.ndarray
    reconstruction_residual: np.ndarray
    evaluation_count: int


def _validate_correspondence(correspondence: np.ndarray) -> np.ndarray:
    matrix = np.asarray(correspondence, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("correspondence must be a non-empty two-dimensional matrix")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("correspondence entries must be finite and non-negative")
    return matrix


def lineage_dispersion(correspondence: np.ndarray) -> tuple[LineageDispersion, ...]:
    """Describe splitting/mixing in each parent row without choosing a cutoff."""
    matrix = _validate_correspondence(correspondence)
    records: list[LineageDispersion] = []
    child_count = matrix.shape[1]
    for parent_index, row in enumerate(matrix):
        total = float(np.sum(row))
        if total == 0.0:
            records.append(LineageDispersion(parent_index, 0.0, None, None, None, None, None))
            continue
        p = row / total
        positive = p[p > 0.0]
        entropy = float(-np.sum(positive * np.log(positive)))
        normalized_entropy = 0.0 if child_count == 1 else float(entropy / np.log(child_count))
        effective_count = float(1.0 / np.sum(p * p))
        dominant_index = int(np.argmax(p))
        records.append(
            LineageDispersion(
                parent_index=parent_index,
                total_weight=total,
                dominant_child_index=dominant_index,
                dominant_fraction=float(p[dominant_index]),
                effective_child_count=effective_count,
                shannon_entropy=entropy,
                normalized_entropy=normalized_entropy,
            )
        )
    return tuple(records)


def interval_dominant_lineage(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[IntervalLineageStep, ...]:
    """Identify a child only when its lower bound exceeds every competitor's upper bound."""
    lower = _validate_correspondence(lower_bounds)
    upper = _validate_correspondence(upper_bounds)
    if lower.shape != upper.shape:
        raise ValueError("lower_bounds and upper_bounds must have the same shape")
    if np.any(lower > upper):
        raise ValueError("every lower bound must be less than or equal to its upper bound")

    records: list[IntervalLineageStep] = []
    for parent_index, (lo, hi) in enumerate(zip(lower, upper)):
        if np.all(hi == 0.0):
            records.append(IntervalLineageStep(parent_index, None, None, None, None, False))
            continue
        gaps: list[float] = []
        for child_index in range(lo.size):
            competitor = float(np.max(np.delete(hi, child_index))) if lo.size > 1 else 0.0
            gaps.append(float(lo[child_index] - competitor))
        winner = int(np.argmax(gaps))
        gap = gaps[winner]
        competitor = float(np.max(np.delete(hi, winner))) if lo.size > 1 else 0.0
        identifiable = gap > 0.0
        records.append(
            IntervalLineageStep(
                parent_index=parent_index,
                child_index=winner if identifiable else None,
                winner_lower_bound=float(lo[winner]) if identifiable else None,
                strongest_competitor_upper_bound=competitor if identifiable else None,
                robust_gap=gap,
                identifiable=identifiable,
            )
        )
    return tuple(records)


def propagate_relative_lineage(correspondences: Sequence[np.ndarray], start_parent: int) -> dict:
    """Propagate relative correspondence weight through a lineage without a branch cutoff."""
    if start_parent < 0:
        raise ValueError("start_parent must be non-negative")
    rule = "row normalization is descriptive relative correspondence flow, not probability or a promotion score"
    if len(correspondences) == 0:
        return {"steps": (), "threshold_applied": False, "normalization_role": rule}

    first = _validate_correspondence(correspondences[0])
    if start_parent >= first.shape[0]:
        raise ValueError("start_parent is outside the first correspondence matrix")
    current = np.zeros(first.shape[0], dtype=float)
    current[start_parent] = 1.0

    steps: list[LineageFlowStep] = []
    for transition_index, raw in enumerate(correspondences):
        matrix = _validate_correspondence(raw)
        if matrix.shape[0] != current.size:
            raise ValueError("adjacent lineage matrices have incompatible dimensions")
        row_sums = np.sum(matrix, axis=1)
        transition = np.zeros_like(matrix)
        nonzero = row_sums > 0.0
        transition[nonzero] = matrix[nonzero] / row_sums[nonzero, None]
        previous_mass = float(np.sum(current))
        next_weights = current @ transition
        surviving = float(np.sum(next_weights))
        lost = previous_mass - surviving
        if lost < 0.0 and abs(lost) < 1e-12:
            lost = 0.0
        steps.append(
            LineageFlowStep(
                transition_index=transition_index,
                descendant_weights=tuple(float(x) for x in next_weights),
                surviving_mass=surviving,
                extinguished_mass_increment=float(lost),
            )
        )
        current = next_weights
    return {"steps": tuple(steps), "threshold_applied": False, "normalization_role": rule}


def _validated_sources(sources: Sequence[str]) -> tuple[str, ...]:
    names = tuple(sources)
    if len(names) == 0 or len(set(names)) != len(names):
        raise ValueError("sources must be a non-empty sequence of unique names")
    return names


def mobius_source_decomposition(
    response_fn: Callable[[frozenset[str]], np.ndarray | Sequence[float] | float],
    sources: Sequence[str],
    max_sources: int = 10,
) -> SourceMobiusDecomposition:
    """Exhaustively separate main and all interaction orders using subset Mobius inversion."""
    names = _validated_sources(sources)
    if max_sources < 1:
        raise ValueError("max_sources must be positive")
    if len(names) > max_sources:
        raise ValueError("source count exceeds the explicit exhaustive-decomposition limit")

    baseline = np.asarray(response_fn(frozenset()), dtype=float)
    if not np.all(np.isfinite(baseline)):
        raise ValueError("response_fn must return finite values")
    responses: dict[tuple[str, ...], np.ndarray] = {(): baseline}
    effects: dict[tuple[str, ...], np.ndarray] = {}

    for size in range(1, len(names) + 1):
        for combo in combinations(names, size):
            value = np.asarray(response_fn(frozenset(combo)), dtype=float)
            if value.shape != baseline.shape:
                raise ValueError("all response_fn outputs must have the same shape")
            if not np.all(np.isfinite(value)):
                raise ValueError("response_fn must return finite values")
            responses[combo] = value
            effect = value - baseline
            combo_set = frozenset(combo)
            for prior_combo, prior_effect in effects.items():
                if frozenset(prior_combo).issubset(combo_set):
                    effect = effect - prior_effect
            effects[combo] = effect

    full = responses[tuple(names)]
    reconstructed = baseline.copy()
    for effect in effects.values():
        reconstructed = reconstructed + effect
    return SourceMobiusDecomposition(
        baseline=baseline,
        effects=effects,
        full_response=full,
        reconstructed_full_response=reconstructed,
        reconstruction_residual=full - reconstructed,
        evaluation_count=len(responses),
    )
