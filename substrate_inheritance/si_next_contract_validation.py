from __future__ import annotations

import copy
import json
from itertools import combinations
from pathlib import Path

from substrate_inheritance.si_next_diagnostic_contract import (
    BASE_SCHEMA,
    RELATIVE_FLOW_SEMANTICS,
    SCHEMA,
    STATUS,
    DiagnosticIngestionRefusal,
    canonical_record_sha256,
    validate_si_next_diagnostic_record,
)


def _hash(character: str) -> str:
    return character * 64


def synthetic_valid_diagnostic_record() -> dict:
    base_records = [
        {
            "record_id": "parent_A",
            "system_id": "SYNTHETIC/PARENT-A",
            "input_record_sha256": _hash("a"),
            "source_repository": "synthetic/si-next-contract",
            "source_ref": "parent-a",
            "source_commit": "0" * 40,
            "base_schema": BASE_SCHEMA,
        },
        {
            "record_id": "parent_B",
            "system_id": "SYNTHETIC/PARENT-B",
            "input_record_sha256": _hash("b"),
            "source_repository": "synthetic/si-next-contract",
            "source_ref": "parent-b",
            "source_commit": "1" * 40,
            "base_schema": BASE_SCHEMA,
        },
        {
            "record_id": "child_C",
            "system_id": "SYNTHETIC/CHILD-C",
            "input_record_sha256": _hash("c"),
            "source_repository": "synthetic/si-next-contract",
            "source_ref": "child-c",
            "source_commit": "2" * 40,
            "base_schema": BASE_SCHEMA,
        },
    ]

    source_ids = ["parent_A", "parent_B"]
    subset_responses = []
    for size in range(len(source_ids) + 1):
        for combo in combinations(source_ids, size):
            subset_responses.append(
                {
                    "active_source_ids": list(combo),
                    "response_artifact_sha256": f"{len(subset_responses) + 100:064x}",
                    "response_units": "synthetic_response_units",
                    "response_shape": [3],
                }
            )

    return {
        "schema": SCHEMA,
        "status": STATUS,
        "diagnostic_id": "SYNTHETIC/SI-NEXT-DIAGNOSTIC-CONTRACT",
        "base_v0_2_record_refs": base_records,
        "relationship_R": {
            "master_score_computed": False,
            "physical_threshold_applied": False,
            "views": {
                "scalar": {
                    "state": "SUPPORTED",
                    "decision_basis": "synthetic scalar record",
                    "source_artifact_sha256": _hash("d"),
                },
                "modal": {
                    "state": "NONIDENTIFIABLE",
                    "decision_basis": "synthetic crowded-sector refusal record",
                    "source_artifact_sha256": _hash("e"),
                },
                "conglomeration": {
                    "state": "SUPPORTED",
                    "decision_basis": "synthetic coupling-specificity record",
                    "source_artifact_sha256": _hash("f"),
                },
            },
        },
        "uncertainty_U": {
            "entries": [
                {
                    "view": "scalar",
                    "status": "BOUNDED",
                    "method": "synthetic perturbation envelope",
                    "source_artifact_sha256": _hash("1"),
                },
                {
                    "view": "modal",
                    "status": "UNRESOLVED",
                    "method": "synthetic subspace conditioning audit",
                    "source_artifact_sha256": _hash("2"),
                },
                {
                    "view": "conglomeration",
                    "status": "BOUNDED",
                    "method": "synthetic coupling perturbation audit",
                    "source_artifact_sha256": _hash("3"),
                },
            ]
        },
        "lineage": {
            "enabled": True,
            "relative_flow_semantics": RELATIVE_FLOW_SEMANTICS,
            "relative_flow_is_probability": False,
            "relative_flow_is_causal_fraction": False,
            "relative_flow_is_inheritance_percentage": False,
            "relative_flow_is_promotion_score": False,
            "transitions": [
                {
                    "transition_id": "parent_A_to_child_C",
                    "parent_record_sha256": _hash("a"),
                    "child_record_sha256": _hash("c"),
                    "correspondence_artifact_sha256": _hash("4"),
                    "uncertainty_artifact_sha256": _hash("5"),
                    "identifiability_requested": True,
                    "mapping_frozen_before_target_carrier_inspection": True,
                    "target_kinetics_used_to_choose_mapping": False,
                    "chi_used_to_choose_mapping": False,
                }
            ],
        },
        "multi_parent": {
            "enabled": True,
            "exact_mobius_requested": True,
            "subset_manifest_complete_for_exact_mobius": True,
            "normalize_effects_to_unit_sum": False,
            "protocol_frozen_before_target_interpretation": True,
            "target_kinetics_used_to_choose_protocol": False,
            "chi_used_to_choose_protocol": False,
            "source_ids": source_ids,
            "subset_responses": subset_responses,
        },
        "provenance": {
            "source_repository": "synthetic/si-next-contract",
            "source_ref": "synthetic",
            "source_commit": "3" * 40,
            "diagnostic_method_frozen_before_target_interpretation": True,
            "post_target_threshold_fitting": False,
            "post_target_mapping_fitting": False,
        },
        "science_firewall": {
            "v0_2_promotion_gates_changed": False,
            "v0_2_physical_thresholds_changed": False,
            "co_cu111_scoring_changed": False,
            "retroactive_reinterpretation_allowed": False,
            "candidate_output_may_override_v0_2_label": False,
            "physical_promotion_semantics": False,
            "physical_threshold_introduced": False,
            "physical_promotion_label_assigned": False,
        },
    }


def _refusal_code(record: dict) -> str:
    try:
        validate_si_next_diagnostic_record(record)
    except DiagnosticIngestionRefusal as exc:
        return exc.code
    raise AssertionError("record unexpectedly passed candidate diagnostic contract")


def synthetic_si_next_contract_validation() -> dict:
    valid = synthetic_valid_diagnostic_record()
    analysis = validate_si_next_diagnostic_record(valid)

    master_score = copy.deepcopy(valid)
    master_score["relationship_R"]["master_score_computed"] = True

    no_lineage_uncertainty = copy.deepcopy(valid)
    no_lineage_uncertainty["lineage"]["transitions"][0]["uncertainty_artifact_sha256"] = None

    probability_flow = copy.deepcopy(valid)
    probability_flow["lineage"]["relative_flow_is_probability"] = True

    incomplete_exact = copy.deepcopy(valid)
    incomplete_exact["multi_parent"]["subset_responses"] = incomplete_exact["multi_parent"]["subset_responses"][:-1]
    incomplete_exact["multi_parent"]["subset_manifest_complete_for_exact_mobius"] = False

    normalized_sources = copy.deepcopy(valid)
    normalized_sources["multi_parent"]["normalize_effects_to_unit_sum"] = True

    override_v02 = copy.deepcopy(valid)
    override_v02["science_firewall"]["candidate_output_may_override_v0_2_label"] = True

    return {
        "scope": "synthetic_si_next_diagnostic_contract_validation_only",
        "status": "candidate_v0.3_non_authoritative",
        "real_system_evidence": False,
        "physical_thresholds_frozen": False,
        "analysis": analysis,
        "canonical_input_sha256": canonical_record_sha256(valid),
        "refusal_checks": {
            "cross_view_master_score": _refusal_code(master_score),
            "lineage_identifiability_without_uncertainty": _refusal_code(no_lineage_uncertainty),
            "relative_flow_as_probability": _refusal_code(probability_flow),
            "incomplete_exact_mobius": _refusal_code(incomplete_exact),
            "normalized_source_effects": _refusal_code(normalized_sources),
            "candidate_override_of_v0_2": _refusal_code(override_v02),
        },
    }


def write_validation(path: str | Path) -> dict:
    result = synthetic_si_next_contract_validation()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = write_validation("substrate_inheritance/results/si_next_contract_validation.json")
    print(json.dumps(result, indent=2, sort_keys=True))
