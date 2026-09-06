from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


VIEW_STATES = frozenset({"SUPPORTED", "DISRUPTED", "NONIDENTIFIABLE", "NOT_MEASURED"})


@dataclass(frozen=True)
class CrossViewState:
    scalar: str
    modal: str
    conglomeration: str

    def __post_init__(self) -> None:
        for name, value in (
            ("scalar", self.scalar),
            ("modal", self.modal),
            ("conglomeration", self.conglomeration),
        ):
            if value not in VIEW_STATES:
                raise ValueError(f"Unsupported {name} state: {value}")


@dataclass(frozen=True)
class ViewUncertainty:
    scalar: tuple[str, ...] = ()
    modal: tuple[str, ...] = ()
    conglomeration: tuple[str, ...] = ()


@dataclass(frozen=True)
class LineageStep:
    parent_index: int
    child_index: int | None
    dominant_score: float | None
    runner_up_score: float | None
    margin: float | None
    margin_uncertainty: float | None
    unique_dominant: bool
    identifiable: bool | None


@dataclass(frozen=True)
class SourceAttribution:
    baseline: np.ndarray
    main_effects: dict[str, np.ndarray]
    pair_interactions: dict[tuple[str, str], np.ndarray]
    full_response: np.ndarray
    reconstructed_response: np.ndarray
    higher_order_residual: np.ndarray


def relationship_pattern(state: CrossViewState) -> str:
    values = (state.scalar, state.modal, state.conglomeration)
    if any(value in {"NONIDENTIFIABLE", "NOT_MEASURED"} for value in values):
        return "RELATIONSHIP_UNRESOLVED"

    supported = tuple(value == "SUPPORTED" for value in values)
    patterns = {
        (True, True, True): "CROSS_VIEW_COHERENCE",
        (True, False, False): "SCALAR_ONLY",
        (False, True, False): "MODAL_ONLY",
        (False, False, True): "CONGLOMERATION_ONLY",
        (True, True, False): "SCALAR_MODAL_WITHOUT_CONGLOMERATION",
        (True, False, True): "SCALAR_CONGLOMERATION_WITHOUT_MODAL",
        (False, True, True): "MODAL_CONGLOMERATION_WITHOUT_SCALAR",
        (False, False, False): "NO_VIEW_SUPPORT",
    }
    return patterns[supported]


def propagate_uncertainty(uncertainty: ViewUncertainty) -> dict:
    by_view = {
        "scalar": tuple(uncertainty.scalar),
        "modal": tuple(uncertainty.modal),
        "conglomeration": tuple(uncertainty.conglomeration),
    }
    blockers = [f"{view}:{reason}" for view, reasons in by_view.items() for reason in reasons]
    return {
        "joint_identifiable": len(blockers) == 0,
        "uncertainty_by_view": by_view,
        "joint_blockers": tuple(blockers),
        "rule": "uncertainty in any required view is retained rather than erased by precision in another view",
    }


def perturbation_envelope(samples: np.ndarray | Iterable[float]) -> dict:
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("samples must contain finite values")
    flat = arr.reshape(-1)
    median = float(np.median(flat))
    return {
        "minimum": float(np.min(flat)),
        "maximum": float(np.max(flat)),
        "median": median,
        "median_absolute_deviation": float(np.median(np.abs(flat - median))),
        "range": float(np.ptp(flat)),
        "threshold_applied": False,
    }


def dominant_lineage_transition(
    correspondence: np.ndarray,
    margin_uncertainty: float | Sequence[float] | None = None,
) -> tuple[LineageStep, ...]:
    matrix = np.asarray(correspondence, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("correspondence must be a non-empty two-dimensional matrix")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("correspondence entries must be finite and non-negative")

    uncertainty: np.ndarray | None
    if margin_uncertainty is None:
        uncertainty = None
    else:
        raw = np.asarray(margin_uncertainty, dtype=float)
        if raw.ndim == 0:
            uncertainty = np.full(matrix.shape[0], float(raw))
        elif raw.shape == (matrix.shape[0],):
            uncertainty = raw
        else:
            raise ValueError("margin_uncertainty must be scalar or one value per parent row")
        if np.any(~np.isfinite(uncertainty)) or np.any(uncertainty < 0.0):
            raise ValueError("margin_uncertainty must be finite and non-negative")

    steps: list[LineageStep] = []
    for parent_index, row in enumerate(matrix):
        row_uncertainty = None if uncertainty is None else float(uncertainty[parent_index])
        if np.all(row == 0.0):
            steps.append(LineageStep(parent_index, None, None, None, None, row_uncertainty, False, False))
            continue
        order = np.argsort(row)[::-1]
        best_index = int(order[0])
        best = float(row[best_index])
        runner = float(row[int(order[1])]) if row.size > 1 else 0.0
        margin = best - runner
        unique = margin > 0.0
        if not unique:
            identifiable: bool | None = False
            child_index: int | None = None
        else:
            child_index = best_index
            identifiable = None if row_uncertainty is None else margin > row_uncertainty
        steps.append(
            LineageStep(
                parent_index=parent_index,
                child_index=child_index,
                dominant_score=best,
                runner_up_score=runner,
                margin=margin,
                margin_uncertainty=row_uncertainty,
                unique_dominant=unique,
                identifiable=identifiable,
            )
        )
    return tuple(steps)


def trace_dominant_lineage(
    correspondences: Sequence[np.ndarray],
    start_parent: int,
    margin_uncertainties: Sequence[float | Sequence[float]],
) -> tuple[LineageStep, ...]:
    if start_parent < 0:
        raise ValueError("start_parent must be non-negative")
    if len(correspondences) != len(margin_uncertainties):
        raise ValueError("one uncertainty specification is required per correspondence matrix")
    current = start_parent
    path: list[LineageStep] = []
    for correspondence, uncertainty in zip(correspondences, margin_uncertainties):
        transitions = dominant_lineage_transition(correspondence, uncertainty)
        if current >= len(transitions):
            raise ValueError("lineage index is outside the next correspondence matrix")
        step = transitions[current]
        path.append(step)
        if step.identifiable is not True or step.child_index is None:
            break
        current = step.child_index
    return tuple(path)


def source_inclusion_exclusion(
    response_fn: Callable[[frozenset[str]], np.ndarray | Sequence[float] | float],
    sources: Sequence[str],
) -> SourceAttribution:
    names = tuple(sources)
    if len(names) == 0 or len(set(names)) != len(names):
        raise ValueError("sources must be a non-empty sequence of unique names")

    def evaluate(active: frozenset[str]) -> np.ndarray:
        value = np.asarray(response_fn(active), dtype=float)
        if not np.all(np.isfinite(value)):
            raise ValueError("response_fn must return finite values")
        return value

    empty = frozenset()
    baseline = evaluate(empty)

    def evaluate_like_baseline(active: frozenset[str]) -> np.ndarray:
        value = evaluate(active)
        if value.shape != baseline.shape:
            raise ValueError("all response_fn outputs must have the same shape")
        return value

    singles = {name: evaluate_like_baseline(frozenset({name})) for name in names}
    main = {name: singles[name] - baseline for name in names}

    pair_interactions: dict[tuple[str, str], np.ndarray] = {}
    for left, right in combinations(names, 2):
        both = evaluate_like_baseline(frozenset({left, right}))
        pair_interactions[(left, right)] = both - singles[left] - singles[right] + baseline

    full = evaluate_like_baseline(frozenset(names))
    reconstructed = baseline.copy()
    for effect in main.values():
        reconstructed = reconstructed + effect
    for interaction in pair_interactions.values():
        reconstructed = reconstructed + interaction
    residual = full - reconstructed

    return SourceAttribution(
        baseline=baseline,
        main_effects=main,
        pair_interactions=pair_interactions,
        full_response=full,
        reconstructed_response=reconstructed,
        higher_order_residual=residual,
    )


def source_ablation_effects(
    response_fn: Callable[[frozenset[str]], np.ndarray | Sequence[float] | float],
    sources: Sequence[str],
) -> Mapping[str, np.ndarray]:
    names = tuple(sources)
    if len(names) == 0 or len(set(names)) != len(names):
        raise ValueError("sources must be a non-empty sequence of unique names")
    full_set = frozenset(names)
    full = np.asarray(response_fn(full_set), dtype=float)
    if not np.all(np.isfinite(full)):
        raise ValueError("response_fn must return finite values")
    effects: dict[str, np.ndarray] = {}
    for name in names:
        without = np.asarray(response_fn(full_set.difference({name})), dtype=float)
        if without.shape != full.shape:
            raise ValueError("all response_fn outputs must have the same shape")
        if not np.all(np.isfinite(without)):
            raise ValueError("response_fn must return finite values")
        effects[name] = full - without
    return effects
