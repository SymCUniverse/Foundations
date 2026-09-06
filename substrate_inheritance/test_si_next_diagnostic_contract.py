import copy

import pytest

from substrate_inheritance.si_next_contract_validation import synthetic_valid_diagnostic_record
from substrate_inheritance.si_next_diagnostic_contract import (
    DiagnosticIngestionRefusal,
    canonical_record_sha256,
    validate_si_next_diagnostic_record,
)


def refusal_code(record):
    with pytest.raises(DiagnosticIngestionRefusal) as exc:
        validate_si_next_diagnostic_record(record)
    return exc.value.code


def test_valid_candidate_diagnostic_record_passes_without_physical_promotion_semantics():
    result = validate_si_next_diagnostic_record(synthetic_valid_diagnostic_record())
    assert result["software_admissibility_status"] == "PASS_CANDIDATE_DIAGNOSTIC_INPUT_CONTRACT"
    assert result["physical_inheritance_threshold_applied"] is False
    assert result["inheritance_promotion_label_assigned"] is False
    assert result["v0_2_result_overridden"] is False
    assert result["physical_inheritance_claim"] is False


def test_relationship_requires_all_three_views():
    record = synthetic_valid_diagnostic_record()
    del record["relationship_R"]["views"]["modal"]
    assert refusal_code(record) == "INCOMPLETE_RELATIONSHIP_VIEWS"


def test_relationship_master_score_is_refused():
    record = synthetic_valid_diagnostic_record()
    record["relationship_R"]["master_score_computed"] = True
    assert refusal_code(record) == "SCIENCE_FIREWALL_VIOLATION"


def test_primary_uncertainty_views_must_be_explicit():
    record = synthetic_valid_diagnostic_record()
    record["uncertainty_U"]["entries"] = [
        item for item in record["uncertainty_U"]["entries"] if item["view"] != "modal"
    ]
    assert refusal_code(record) == "INCOMPLETE_PRIMARY_UNCERTAINTY"


def test_lineage_identifiability_requires_uncertainty_provenance():
    record = synthetic_valid_diagnostic_record()
    record["lineage"]["transitions"][0]["uncertainty_artifact_sha256"] = None
    assert refusal_code(record) == "MISSING_LINEAGE_UNCERTAINTY"


def test_lineage_without_identifiability_request_may_explicitly_lack_uncertainty_artifact():
    record = synthetic_valid_diagnostic_record()
    record["lineage"]["transitions"][0]["identifiability_requested"] = False
    record["lineage"]["transitions"][0]["uncertainty_artifact_sha256"] = None
    result = validate_si_next_diagnostic_record(record)
    assert result["lineage"]["transitions"][0]["identifiability_requested"] is False


def test_relative_flow_cannot_be_declared_probability_or_inheritance_percentage():
    probability = synthetic_valid_diagnostic_record()
    probability["lineage"]["relative_flow_is_probability"] = True
    assert refusal_code(probability) == "SCIENCE_FIREWALL_VIOLATION"

    percentage = synthetic_valid_diagnostic_record()
    percentage["lineage"]["relative_flow_is_inheritance_percentage"] = True
    assert refusal_code(percentage) == "SCIENCE_FIREWALL_VIOLATION"


def test_exact_mobius_requires_complete_two_power_n_subset_manifest():
    record = synthetic_valid_diagnostic_record()
    record["multi_parent"]["subset_responses"] = record["multi_parent"]["subset_responses"][:-1]
    record["multi_parent"]["subset_manifest_complete_for_exact_mobius"] = False
    assert refusal_code(record) == "INCOMPLETE_EXACT_MOBIUS"


def test_partial_multi_parent_manifest_is_allowed_only_when_not_claimed_exact():
    record = synthetic_valid_diagnostic_record()
    record["multi_parent"]["exact_mobius_requested"] = False
    record["multi_parent"]["subset_responses"] = record["multi_parent"]["subset_responses"][:2]
    record["multi_parent"]["subset_manifest_complete_for_exact_mobius"] = False
    result = validate_si_next_diagnostic_record(record)
    assert result["multi_parent"]["exact_mobius_requested"] is False
    assert result["multi_parent"]["subset_manifest_complete"] is False


def test_exact_subset_responses_must_share_units_and_shape():
    unit_drift = synthetic_valid_diagnostic_record()
    unit_drift["multi_parent"]["subset_responses"][1]["response_units"] = "different_units"
    assert refusal_code(unit_drift) == "RESPONSE_UNIT_MISMATCH"

    shape_drift = synthetic_valid_diagnostic_record()
    shape_drift["multi_parent"]["subset_responses"][1]["response_shape"] = [4]
    assert refusal_code(shape_drift) == "RESPONSE_SHAPE_MISMATCH"


def test_source_effects_cannot_be_normalized_to_unit_sum():
    record = synthetic_valid_diagnostic_record()
    record["multi_parent"]["normalize_effects_to_unit_sum"] = True
    assert refusal_code(record) == "SCIENCE_FIREWALL_VIOLATION"


def test_target_leakage_in_lineage_or_source_protocol_is_refused():
    lineage = synthetic_valid_diagnostic_record()
    lineage["lineage"]["transitions"][0]["chi_used_to_choose_mapping"] = True
    assert refusal_code(lineage) == "SCIENCE_FIREWALL_VIOLATION"

    sources = synthetic_valid_diagnostic_record()
    sources["multi_parent"]["target_kinetics_used_to_choose_protocol"] = True
    assert refusal_code(sources) == "SCIENCE_FIREWALL_VIOLATION"


def test_candidate_cannot_override_v02_or_introduce_physical_threshold():
    override = synthetic_valid_diagnostic_record()
    override["science_firewall"]["candidate_output_may_override_v0_2_label"] = True
    assert refusal_code(override) == "SCIENCE_FIREWALL_VIOLATION"

    threshold = synthetic_valid_diagnostic_record()
    threshold["science_firewall"]["physical_threshold_introduced"] = True
    assert refusal_code(threshold) == "SCIENCE_FIREWALL_VIOLATION"


def test_subset_source_ids_must_reference_declared_base_records():
    record = synthetic_valid_diagnostic_record()
    record["multi_parent"]["source_ids"][0] = "unknown_source"
    assert refusal_code(record) == "UNKNOWN_SOURCE_ID"


def test_canonical_candidate_record_hash_is_stable_to_top_level_key_order():
    record = synthetic_valid_diagnostic_record()
    reordered = dict(reversed(list(copy.deepcopy(record).items())))
    assert canonical_record_sha256(record) == canonical_record_sha256(reordered)
