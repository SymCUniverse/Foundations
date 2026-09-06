import numpy as np
import pytest

from substrate_inheritance.si_next_higher_order import (
    interval_dominant_lineage,
    lineage_dispersion,
    mobius_source_decomposition,
    propagate_relative_lineage,
)
from substrate_inheritance.si_next_validation import synthetic_si_next_validation


def test_lineage_dispersion_preserves_equal_split_without_cutoff():
    record = lineage_dispersion(np.array([[0.5, 0.5]]))[0]
    assert record.effective_child_count == pytest.approx(2.0)
    assert record.normalized_entropy == pytest.approx(1.0)
    assert record.dominant_fraction == pytest.approx(0.5)


def test_lineage_dispersion_recognizes_single_carrier_limit():
    record = lineage_dispersion(np.array([[1.0, 0.0, 0.0]]))[0]
    assert record.effective_child_count == pytest.approx(1.0)
    assert record.normalized_entropy == pytest.approx(0.0)


def test_interval_lineage_accepts_only_nonoverlapping_winner():
    step = interval_dominant_lineage(
        np.array([[0.8, 0.1]]),
        np.array([[0.9, 0.2]]),
    )[0]
    assert step.identifiable is True
    assert step.child_index == 0
    assert step.robust_gap == pytest.approx(0.6)


def test_interval_lineage_refuses_overlapping_candidates():
    step = interval_dominant_lineage(
        np.array([[0.45, 0.4]]),
        np.array([[0.55, 0.5]]),
    )[0]
    assert step.identifiable is False
    assert step.child_index is None


def test_relative_lineage_flow_preserves_split_then_reconvergence():
    result = propagate_relative_lineage(
        (
            np.array([[0.5, 0.5], [0.0, 1.0]]),
            np.array([[1.0, 0.0], [1.0, 0.0]]),
        ),
        start_parent=0,
    )
    assert result["steps"][0].descendant_weights == pytest.approx((0.5, 0.5))
    assert result["steps"][1].descendant_weights == pytest.approx((1.0, 0.0))
    assert result["threshold_applied"] is False


def test_relative_lineage_flow_retains_extinction():
    result = propagate_relative_lineage((np.array([[0.0, 0.0], [0.0, 1.0]]),), start_parent=0)
    assert result["steps"][0].surviving_mass == pytest.approx(0.0)
    assert result["steps"][0].extinguished_mass_increment == pytest.approx(1.0)


def test_mobius_decomposition_recovers_all_interaction_orders_exactly():
    def response(active):
        a = float("A" in active)
        b = float("B" in active)
        c = float("C" in active)
        return np.array([1.0 + 2.0 * a + 3.0 * b + 4.0 * c + 5.0 * a * b + 6.0 * a * c + 7.0 * b * c + 8.0 * a * b * c])

    result = mobius_source_decomposition(response, ("A", "B", "C"))
    assert np.allclose(result.effects[("A",)], np.array([2.0]))
    assert np.allclose(result.effects[("A", "B")], np.array([5.0]))
    assert np.allclose(result.effects[("A", "B", "C")], np.array([8.0]))
    assert np.max(np.abs(result.reconstruction_residual)) < 1e-12
    assert result.evaluation_count == 8


def test_mobius_dummy_source_has_no_spurious_attribution():
    def response(active):
        return np.array([1.0 + (2.0 if "A" in active else 0.0)])

    result = mobius_source_decomposition(response, ("A", "D"))
    assert np.allclose(result.effects[("D",)], 0.0)
    assert np.allclose(result.effects[("A", "D")], 0.0)


def test_mobius_decomposition_has_explicit_complexity_guard():
    with pytest.raises(ValueError, match="limit"):
        mobius_source_decomposition(lambda active: np.array([0.0]), ("A", "B", "C"), max_sources=2)


def test_synthetic_record_exposes_higher_order_diagnostics_without_promotion_semantics():
    result = synthetic_si_next_validation()
    assert result["physical_thresholds_frozen"] is False
    assert result["inheritance_promotion_rule_changed"] is False
