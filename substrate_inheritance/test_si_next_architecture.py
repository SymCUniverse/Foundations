import numpy as np
import pytest

from substrate_inheritance.si_next_architecture import (
    CrossViewState,
    ViewUncertainty,
    dominant_lineage_transition,
    perturbation_envelope,
    propagate_uncertainty,
    relationship_pattern,
    source_ablation_effects,
    source_inclusion_exclusion,
    trace_dominant_lineage,
)
from substrate_inheritance.si_next_validation import synthetic_si_next_validation


def test_cross_view_relationship_preserves_disagreement_instead_of_averaging_it():
    state = CrossViewState("SUPPORTED", "DISRUPTED", "DISRUPTED")
    assert relationship_pattern(state) == "SCALAR_ONLY"


def test_cross_view_relationship_reports_full_coherence_only_when_all_views_support():
    state = CrossViewState("SUPPORTED", "SUPPORTED", "SUPPORTED")
    assert relationship_pattern(state) == "CROSS_VIEW_COHERENCE"


def test_nonidentifiable_view_forces_relationship_refusal():
    state = CrossViewState("SUPPORTED", "NONIDENTIFIABLE", "SUPPORTED")
    assert relationship_pattern(state) == "RELATIONSHIP_UNRESOLVED"


def test_invalid_view_state_is_rejected():
    with pytest.raises(ValueError):
        CrossViewState("SUPPORTED", "MAYBE", "SUPPORTED")


def test_uncertainty_in_one_view_is_not_erased_by_other_precise_views():
    result = propagate_uncertainty(ViewUncertainty(modal=("near_degenerate_subspace",)))
    assert result["joint_identifiable"] is False
    assert result["joint_blockers"] == ("modal:near_degenerate_subspace",)


def test_empty_uncertainty_has_no_joint_blockers():
    result = propagate_uncertainty(ViewUncertainty())
    assert result["joint_identifiable"] is True
    assert result["joint_blockers"] == ()


def test_perturbation_envelope_reports_spread_without_applying_threshold():
    result = perturbation_envelope([1.0, 1.1, 0.9])
    assert result["threshold_applied"] is False
    assert result["range"] == pytest.approx(0.2)


def test_dominant_lineage_tracks_unique_correspondence_when_uncertainty_is_resolved():
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    steps = dominant_lineage_transition(transition, margin_uncertainty=0.0)
    assert steps[0].unique_dominant is True
    assert steps[0].identifiable is True
    assert steps[0].child_index == 0
    assert steps[1].child_index == 1


def test_exact_lineage_tie_is_refused_not_broken_arbitrarily():
    step = dominant_lineage_transition(np.array([[0.5, 0.5]]), margin_uncertainty=0.0)[0]
    assert step.unique_dominant is False
    assert step.identifiable is False
    assert step.child_index is None
    assert step.margin == 0.0


def test_unique_dominant_without_uncertainty_is_not_called_identifiable():
    step = dominant_lineage_transition(np.array([[0.51, 0.49]]))[0]
    assert step.unique_dominant is True
    assert step.child_index == 0
    assert step.identifiable is None


def test_lineage_margin_below_uncertainty_is_refused():
    step = dominant_lineage_transition(np.array([[0.51, 0.49]]), margin_uncertainty=0.03)[0]
    assert step.unique_dominant is True
    assert step.margin == pytest.approx(0.02)
    assert step.identifiable is False


def test_lineage_trace_stops_at_first_nonidentifiable_transition():
    path = trace_dominant_lineage(
        (
            np.array([[0.9, 0.1], [0.2, 0.8]]),
            np.array([[0.5, 0.5], [0.1, 0.9]]),
            np.eye(2),
        ),
        start_parent=0,
        margin_uncertainties=(0.0, 0.0, 0.0),
    )
    assert len(path) == 2
    assert path[-1].identifiable is False


def test_additive_sources_have_zero_pair_interaction():
    def response(active):
        return np.array([1.0 + (2.0 if "A" in active else 0.0) + (3.0 if "B" in active else 0.0)])

    result = source_inclusion_exclusion(response, ("A", "B"))
    assert np.allclose(result.pair_interactions[("A", "B")], 0.0)
    assert np.allclose(result.higher_order_residual, 0.0)


def test_nonadditive_sources_preserve_interaction_term():
    def response(active):
        a = 1.0 if "A" in active else 0.0
        b = 1.0 if "B" in active else 0.0
        return np.array([1.0 + 2.0 * a + 3.0 * b + 4.0 * a * b])

    result = source_inclusion_exclusion(response, ("A", "B"))
    assert np.allclose(result.pair_interactions[("A", "B")], np.array([4.0]))
    assert np.allclose(result.full_response, result.reconstructed_response)


def test_ablation_effects_are_not_normalized_into_fake_percentages():
    def response(active):
        a = 1.0 if "A" in active else 0.0
        b = 1.0 if "B" in active else 0.0
        return np.array([a + b + 2.0 * a * b])

    effects = source_ablation_effects(response, ("A", "B"))
    assert effects["A"][0] == pytest.approx(3.0)
    assert effects["B"][0] == pytest.approx(3.0)


def test_three_way_source_interaction_is_retained_as_higher_order_residual():
    def response(active):
        a = 1.0 if "A" in active else 0.0
        b = 1.0 if "B" in active else 0.0
        c = 1.0 if "C" in active else 0.0
        return np.array([a + b + c + 5.0 * a * b * c])

    result = source_inclusion_exclusion(response, ("A", "B", "C"))
    assert np.allclose(result.higher_order_residual, np.array([5.0]))


def test_source_attribution_rejects_shape_drift():
    def response(active):
        if "B" in active:
            return np.array([1.0, 2.0])
        return np.array([1.0])

    with pytest.raises(ValueError, match="same shape"):
        source_inclusion_exclusion(response, ("A", "B"))


def test_source_ablation_rejects_nonfinite_response():
    def response(active):
        return np.array([np.nan if "A" not in active else 1.0])

    with pytest.raises(ValueError, match="finite"):
        source_ablation_effects(response, ("A", "B"))


def test_candidate_validation_cannot_promote_or_redefine_v02():
    result = synthetic_si_next_validation()
    assert result["scope"] == "synthetic_si_next_architecture_validation_only"
    assert result["status"] == "candidate_v0.3_non_authoritative"
    assert result["frozen_v0.2_changed"] is False
    assert result["inheritance_promotion_rule_changed"] is False
    assert result["physical_thresholds_frozen"] is False
    assert result["real_system_evidence"] is False
    assert result["lineage"]["exact_tie_refused"] is True
    assert result["multi_parent"]["effects_forced_to_sum_to_one"] is False
    assert result["multi_parent"]["interacting_pair_interaction_norm"] > 0.0
