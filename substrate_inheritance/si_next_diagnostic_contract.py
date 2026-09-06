from __future__ import annotations

import hashlib
import json
import re
from itertools import combinations
from typing import Any


SCHEMA = "substrate-inheritance-si-next-diagnostic-input-v0.3-candidate"
STATUS = "NON_AUTHORITATIVE_CANDIDATE_NOT_FROZEN_FOR_PHYSICAL_PROMOTION"
BASE_SCHEMA = "substrate-inheritance-real-system-input-v0.2"
ALLOWED_VIEW_STATES = {"SUPPORTED", "DISRUPTED", "NONIDENTIFIABLE", "NOT_MEASURED"}
ALLOWED_UNCERTAINTY_VIEWS = {
    "scalar",
    "modal",
    "conglomeration",
    "relationship",
    "lineage",
    "multi_parent",
}
ALLOWED_UNCERTAINTY_STATUSES = {"BOUNDED", "UNRESOLVED", "NOT_AVAILABLE"}
REQUIRED_PRIMARY_UNCERTAINTY_VIEWS = {"scalar", "modal", "conglomeration"}
RELATIVE_FLOW_SEMANTICS = "descriptive_relative_correspondence_only"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class DiagnosticIngestionRefusal(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _require(mapping: dict[str, Any], field: str, where: str) -> Any:
    if field not in mapping:
        raise DiagnosticIngestionRefusal("MISSING_FIELD", f"{where}.{field} is required")
    return mapping[field]


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DiagnosticIngestionRefusal("INVALID_STRING", f"{where} must be a nonempty string")
    return value


def _validate_sha256(value: Any, where: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise DiagnosticIngestionRefusal(
            "INVALID_SHA256",
            f"{where} must be a 64-character hexadecimal SHA-256"
            + (" or null" if allow_none else ""),
        )
    return value.lower()


def _require_bool(mapping: dict[str, Any], field: str, where: str, expected: bool) -> None:
    value = _require(mapping, field, where)
    if value is not expected:
        raise DiagnosticIngestionRefusal(
            "SCIENCE_FIREWALL_VIOLATION",
            f"{where}.{field} must be {expected}",
        )


def _validate_base_refs(raw: Any) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise DiagnosticIngestionRefusal(
            "MISSING_BASE_RECORDS",
            "base_v0_2_record_refs must be a nonempty list",
        )
    by_id: dict[str, dict[str, Any]] = {}
    hash_to_id: dict[str, str] = {}
    for index, item in enumerate(raw):
        where = f"base_v0_2_record_refs[{index}]"
        if not isinstance(item, dict):
            raise DiagnosticIngestionRefusal("INVALID_BASE_RECORD", f"{where} must be an object")
        record_id = _nonempty_string(_require(item, "record_id", where), f"{where}.record_id")
        if record_id in by_id:
            raise DiagnosticIngestionRefusal("DUPLICATE_BASE_RECORD", f"duplicate record_id {record_id!r}")
        system_id = _nonempty_string(_require(item, "system_id", where), f"{where}.system_id")
        digest = _validate_sha256(_require(item, "input_record_sha256", where), f"{where}.input_record_sha256")
        for field in ("source_repository", "source_ref", "source_commit"):
            _nonempty_string(_require(item, field, where), f"{where}.{field}")
        if "base_schema" in item and item["base_schema"] != BASE_SCHEMA:
            raise DiagnosticIngestionRefusal(
                "BASE_SCHEMA_MISMATCH",
                f"{where}.base_schema must equal {BASE_SCHEMA!r}",
            )
        if digest in hash_to_id:
            raise DiagnosticIngestionRefusal(
                "DUPLICATE_BASE_RECORD",
                f"base record hash {digest!r} is referenced more than once",
            )
        normalized = dict(item)
        normalized["system_id"] = system_id
        normalized["input_record_sha256"] = digest
        by_id[record_id] = normalized
        hash_to_id[digest] = record_id
    return by_id, hash_to_id


def _validate_relationship(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise DiagnosticIngestionRefusal("INVALID_RELATIONSHIP", "relationship_R must be an object")
    _require_bool(raw, "master_score_computed", "relationship_R", False)
    _require_bool(raw, "physical_threshold_applied", "relationship_R", False)
    views = _require(raw, "views", "relationship_R")
    if not isinstance(views, dict):
        raise DiagnosticIngestionRefusal("INVALID_RELATIONSHIP", "relationship_R.views must be an object")
    required = {"scalar", "modal", "conglomeration"}
    if set(views) != required:
        raise DiagnosticIngestionRefusal(
            "INCOMPLETE_RELATIONSHIP_VIEWS",
            "relationship_R.views must contain exactly scalar, modal, and conglomeration",
        )
    states: dict[str, str] = {}
    for view in ("scalar", "modal", "conglomeration"):
        item = views[view]
        where = f"relationship_R.views.{view}"
        if not isinstance(item, dict):
            raise DiagnosticIngestionRefusal("INVALID_RELATIONSHIP", f"{where} must be an object")
        state = _require(item, "state", where)
        if state not in ALLOWED_VIEW_STATES:
            raise DiagnosticIngestionRefusal("INVALID_VIEW_STATE", f"{where}.state={state!r} is unsupported")
        _nonempty_string(_require(item, "decision_basis", where), f"{where}.decision_basis")
        _validate_sha256(_require(item, "source_artifact_sha256", where), f"{where}.source_artifact_sha256")
        states[view] = state
    return states


def _validate_uncertainty(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise DiagnosticIngestionRefusal("INVALID_UNCERTAINTY", "uncertainty_U must be an object")
    entries = _require(raw, "entries", "uncertainty_U")
    if not isinstance(entries, list):
        raise DiagnosticIngestionRefusal("INVALID_UNCERTAINTY", "uncertainty_U.entries must be a list")
    by_view: dict[str, str] = {}
    for index, item in enumerate(entries):
        where = f"uncertainty_U.entries[{index}]"
        if not isinstance(item, dict):
            raise DiagnosticIngestionRefusal("INVALID_UNCERTAINTY", f"{where} must be an object")
        view = _require(item, "view", where)
        if view not in ALLOWED_UNCERTAINTY_VIEWS:
            raise DiagnosticIngestionRefusal("INVALID_UNCERTAINTY_VIEW", f"{where}.view={view!r} is unsupported")
        if view in by_view:
            raise DiagnosticIngestionRefusal("DUPLICATE_UNCERTAINTY_VIEW", f"uncertainty for {view!r} appears more than once")
        status = _require(item, "status", where)
        if status not in ALLOWED_UNCERTAINTY_STATUSES:
            raise DiagnosticIngestionRefusal("INVALID_UNCERTAINTY_STATUS", f"{where}.status={status!r} is unsupported")
        _nonempty_string(_require(item, "method", where), f"{where}.method")
        _validate_sha256(_require(item, "source_artifact_sha256", where), f"{where}.source_artifact_sha256")
        by_view[view] = status
    missing = REQUIRED_PRIMARY_UNCERTAINTY_VIEWS.difference(by_view)
    if missing:
        raise DiagnosticIngestionRefusal(
            "INCOMPLETE_PRIMARY_UNCERTAINTY",
            f"uncertainty_U must explicitly record scalar, modal, and conglomeration; missing {sorted(missing)}",
        )
    return by_view


def _validate_lineage(raw: Any, base_hash_to_id: dict[str, str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise DiagnosticIngestionRefusal("INVALID_LINEAGE", "lineage must be an object")
    enabled = _require(raw, "enabled", "lineage")
    if not isinstance(enabled, bool):
        raise DiagnosticIngestionRefusal("INVALID_LINEAGE", "lineage.enabled must be boolean")
    semantics = _require(raw, "relative_flow_semantics", "lineage")
    if semantics != RELATIVE_FLOW_SEMANTICS:
        raise DiagnosticIngestionRefusal(
            "INVALID_FLOW_SEMANTICS",
            f"lineage.relative_flow_semantics must equal {RELATIVE_FLOW_SEMANTICS!r}",
        )
    for field in (
        "relative_flow_is_probability",
        "relative_flow_is_causal_fraction",
        "relative_flow_is_inheritance_percentage",
        "relative_flow_is_promotion_score",
    ):
        _require_bool(raw, field, "lineage", False)
    transitions = _require(raw, "transitions", "lineage")
    if not isinstance(transitions, list):
        raise DiagnosticIngestionRefusal("INVALID_LINEAGE", "lineage.transitions must be a list")
    if enabled and not transitions:
        raise DiagnosticIngestionRefusal("EMPTY_LINEAGE", "lineage.enabled=true requires at least one transition")
    if not enabled and transitions:
        raise DiagnosticIngestionRefusal("LINEAGE_DISABLED_WITH_TRANSITIONS", "lineage.transitions must be empty when lineage.enabled=false")

    transition_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(transitions):
        where = f"lineage.transitions[{index}]"
        if not isinstance(item, dict):
            raise DiagnosticIngestionRefusal("INVALID_LINEAGE_TRANSITION", f"{where} must be an object")
        transition_id = _nonempty_string(_require(item, "transition_id", where), f"{where}.transition_id")
        if transition_id in transition_ids:
            raise DiagnosticIngestionRefusal("DUPLICATE_TRANSITION", f"duplicate transition_id {transition_id!r}")
        transition_ids.add(transition_id)
        parent_hash = _validate_sha256(_require(item, "parent_record_sha256", where), f"{where}.parent_record_sha256")
        child_hash = _validate_sha256(_require(item, "child_record_sha256", where), f"{where}.child_record_sha256")
        if parent_hash not in base_hash_to_id or child_hash not in base_hash_to_id:
            raise DiagnosticIngestionRefusal(
                "UNKNOWN_BASE_RECORD",
                f"{where} parent/child hashes must reference base_v0_2_record_refs",
            )
        correspondence_hash = _validate_sha256(
            _require(item, "correspondence_artifact_sha256", where),
            f"{where}.correspondence_artifact_sha256",
        )
        uncertainty_hash = _validate_sha256(
            _require(item, "uncertainty_artifact_sha256", where),
            f"{where}.uncertainty_artifact_sha256",
            allow_none=True,
        )
        identifiability_requested = _require(item, "identifiability_requested", where)
        if not isinstance(identifiability_requested, bool):
            raise DiagnosticIngestionRefusal(
                "INVALID_IDENTIFIABILITY_REQUEST",
                f"{where}.identifiability_requested must be boolean",
            )
        if identifiability_requested and uncertainty_hash is None:
            raise DiagnosticIngestionRefusal(
                "MISSING_LINEAGE_UNCERTAINTY",
                f"{where} cannot request identifiable lineage without uncertainty provenance",
            )
        _require_bool(item, "mapping_frozen_before_target_carrier_inspection", where, True)
        _require_bool(item, "target_kinetics_used_to_choose_mapping", where, False)
        _require_bool(item, "chi_used_to_choose_mapping", where, False)
        normalized.append(
            {
                "transition_id": transition_id,
                "parent_record_sha256": parent_hash,
                "child_record_sha256": child_hash,
                "correspondence_artifact_sha256": correspondence_hash,
                "uncertainty_artifact_sha256": uncertainty_hash,
                "identifiability_requested": identifiability_requested,
            }
        )
    return {"enabled": enabled, "transition_count": len(normalized), "transitions": normalized}


def _all_subsets(names: tuple[str, ...]) -> set[frozenset[str]]:
    expected: set[frozenset[str]] = set()
    for size in range(len(names) + 1):
        for combo in combinations(names, size):
            expected.add(frozenset(combo))
    return expected


def _validate_multi_parent(raw: Any, base_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise DiagnosticIngestionRefusal("INVALID_MULTI_PARENT", "multi_parent must be an object")
    enabled = _require(raw, "enabled", "multi_parent")
    if not isinstance(enabled, bool):
        raise DiagnosticIngestionRefusal("INVALID_MULTI_PARENT", "multi_parent.enabled must be boolean")
    exact = _require(raw, "exact_mobius_requested", "multi_parent")
    if not isinstance(exact, bool):
        raise DiagnosticIngestionRefusal("INVALID_MULTI_PARENT", "multi_parent.exact_mobius_requested must be boolean")
    complete_flag = _require(raw, "subset_manifest_complete_for_exact_mobius", "multi_parent")
    if not isinstance(complete_flag, bool):
        raise DiagnosticIngestionRefusal(
            "INVALID_MULTI_PARENT",
            "multi_parent.subset_manifest_complete_for_exact_mobius must be boolean",
        )
    _require_bool(raw, "normalize_effects_to_unit_sum", "multi_parent", False)
    _require_bool(raw, "protocol_frozen_before_target_interpretation", "multi_parent", True)
    _require_bool(raw, "target_kinetics_used_to_choose_protocol", "multi_parent", False)
    _require_bool(raw, "chi_used_to_choose_protocol", "multi_parent", False)

    source_ids_raw = _require(raw, "source_ids", "multi_parent")
    if not isinstance(source_ids_raw, list):
        raise DiagnosticIngestionRefusal("INVALID_MULTI_PARENT", "multi_parent.source_ids must be a list")
    if any(not isinstance(name, str) or not name.strip() for name in source_ids_raw):
        raise DiagnosticIngestionRefusal("INVALID_SOURCE_ID", "multi_parent.source_ids must contain nonempty strings")
    if len(set(source_ids_raw)) != len(source_ids_raw):
        raise DiagnosticIngestionRefusal("DUPLICATE_SOURCE_ID", "multi_parent.source_ids contains duplicates")
    source_ids = tuple(source_ids_raw)
    if enabled and not source_ids:
        raise DiagnosticIngestionRefusal("EMPTY_MULTI_PARENT", "multi_parent.enabled=true requires source_ids")
    if not enabled and (source_ids or exact):
        raise DiagnosticIngestionRefusal(
            "MULTI_PARENT_DISABLED_WITH_INPUTS",
            "disabled multi_parent must have no sources and cannot request exact Mobius decomposition",
        )
    unknown = [name for name in source_ids if name not in base_by_id]
    if unknown:
        raise DiagnosticIngestionRefusal(
            "UNKNOWN_SOURCE_ID",
            f"multi_parent source_ids must reference base record IDs; unknown {unknown}",
        )
    if exact and len(source_ids) > 10:
        raise DiagnosticIngestionRefusal(
            "EXHAUSTIVE_SOURCE_LIMIT",
            "exact Mobius decomposition exceeds the candidate computational source limit of 10",
        )

    manifest = _require(raw, "subset_responses", "multi_parent")
    if not isinstance(manifest, list):
        raise DiagnosticIngestionRefusal("INVALID_SUBSET_MANIFEST", "multi_parent.subset_responses must be a list")
    if not enabled and manifest:
        raise DiagnosticIngestionRefusal(
            "MULTI_PARENT_DISABLED_WITH_INPUTS",
            "disabled multi_parent must have an empty subset_responses manifest",
        )

    seen: set[frozenset[str]] = set()
    response_units: str | None = None
    response_shape: tuple[int, ...] | None = None
    for index, item in enumerate(manifest):
        where = f"multi_parent.subset_responses[{index}]"
        if not isinstance(item, dict):
            raise DiagnosticIngestionRefusal("INVALID_SUBSET_MANIFEST", f"{where} must be an object")
        active_raw = _require(item, "active_source_ids", where)
        if not isinstance(active_raw, list):
            raise DiagnosticIngestionRefusal("INVALID_SUBSET", f"{where}.active_source_ids must be a list")
        if any(not isinstance(name, str) or not name.strip() for name in active_raw):
            raise DiagnosticIngestionRefusal("INVALID_SOURCE_ID", f"{where}.active_source_ids contains invalid IDs")
        if len(set(active_raw)) != len(active_raw):
            raise DiagnosticIngestionRefusal("DUPLICATE_SOURCE_ID", f"{where}.active_source_ids contains duplicates")
        if any(name not in source_ids for name in active_raw):
            raise DiagnosticIngestionRefusal("UNKNOWN_SOURCE_ID", f"{where}.active_source_ids contains an unknown source")
        active = frozenset(active_raw)
        if active in seen:
            raise DiagnosticIngestionRefusal("DUPLICATE_SUBSET", f"subset {sorted(active)} appears more than once")
        seen.add(active)
        _validate_sha256(_require(item, "response_artifact_sha256", where), f"{where}.response_artifact_sha256")
        units = _nonempty_string(_require(item, "response_units", where), f"{where}.response_units")
        shape_raw = _require(item, "response_shape", where)
        if (
            not isinstance(shape_raw, list)
            or not shape_raw
            or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape_raw)
        ):
            raise DiagnosticIngestionRefusal(
                "INVALID_RESPONSE_SHAPE",
                f"{where}.response_shape must be a nonempty list of positive integers",
            )
        shape = tuple(shape_raw)
        if response_units is None:
            response_units = units
            response_shape = shape
        elif units != response_units:
            raise DiagnosticIngestionRefusal(
                "RESPONSE_UNIT_MISMATCH",
                "all subset responses must use common units",
            )
        elif shape != response_shape:
            raise DiagnosticIngestionRefusal(
                "RESPONSE_SHAPE_MISMATCH",
                "all subset responses must have the same shape",
            )

    expected = _all_subsets(source_ids) if enabled else set()
    actually_complete = seen == expected
    if complete_flag != actually_complete:
        raise DiagnosticIngestionRefusal(
            "SUBSET_COMPLETENESS_MISMATCH",
            "subset_manifest_complete_for_exact_mobius must match the actual subset manifest",
        )
    if exact and not actually_complete:
        raise DiagnosticIngestionRefusal(
            "INCOMPLETE_EXACT_MOBIUS",
            "exact Mobius decomposition requires every one of the 2^n source subsets",
        )
    return {
        "enabled": enabled,
        "source_count": len(source_ids),
        "subset_count": len(seen),
        "exact_mobius_requested": exact,
        "subset_manifest_complete": actually_complete,
        "response_units": response_units,
        "response_shape": response_shape,
    }


def _validate_provenance(raw: Any) -> None:
    if not isinstance(raw, dict):
        raise DiagnosticIngestionRefusal("INVALID_PROVENANCE", "provenance must be an object")
    for field in ("source_repository", "source_ref", "source_commit"):
        _nonempty_string(_require(raw, field, "provenance"), f"provenance.{field}")
    _require_bool(raw, "diagnostic_method_frozen_before_target_interpretation", "provenance", True)
    _require_bool(raw, "post_target_threshold_fitting", "provenance", False)
    _require_bool(raw, "post_target_mapping_fitting", "provenance", False)


def _validate_firewall(raw: Any) -> None:
    if not isinstance(raw, dict):
        raise DiagnosticIngestionRefusal("INVALID_FIREWALL", "science_firewall must be an object")
    for field in (
        "v0_2_promotion_gates_changed",
        "v0_2_physical_thresholds_changed",
        "co_cu111_scoring_changed",
        "retroactive_reinterpretation_allowed",
        "candidate_output_may_override_v0_2_label",
        "physical_promotion_semantics",
        "physical_threshold_introduced",
        "physical_promotion_label_assigned",
    ):
        _require_bool(raw, field, "science_firewall", False)


def validate_si_next_diagnostic_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise DiagnosticIngestionRefusal("INVALID_RECORD", "input must be a JSON object")
    if _require(record, "schema", "record") != SCHEMA:
        raise DiagnosticIngestionRefusal("SCHEMA_MISMATCH", f"record.schema must equal {SCHEMA!r}")
    if _require(record, "status", "record") != STATUS:
        raise DiagnosticIngestionRefusal("STATUS_MISMATCH", f"record.status must equal {STATUS!r}")
    diagnostic_id = _nonempty_string(_require(record, "diagnostic_id", "record"), "record.diagnostic_id")
    base_by_id, base_hash_to_id = _validate_base_refs(_require(record, "base_v0_2_record_refs", "record"))
    relationship = _validate_relationship(_require(record, "relationship_R", "record"))
    uncertainty = _validate_uncertainty(_require(record, "uncertainty_U", "record"))
    lineage = _validate_lineage(_require(record, "lineage", "record"), base_hash_to_id)
    multi_parent = _validate_multi_parent(_require(record, "multi_parent", "record"), base_by_id)
    _validate_provenance(_require(record, "provenance", "record"))
    _validate_firewall(_require(record, "science_firewall", "record"))
    return {
        "schema": "substrate-inheritance-si-next-diagnostic-admissibility-v0.1",
        "diagnostic_id": diagnostic_id,
        "software_admissibility_status": "PASS_CANDIDATE_DIAGNOSTIC_INPUT_CONTRACT",
        "base_record_count": len(base_by_id),
        "relationship_states": relationship,
        "uncertainty_status_by_view": uncertainty,
        "lineage": lineage,
        "multi_parent": multi_parent,
        "physical_inheritance_threshold_applied": False,
        "inheritance_promotion_label_assigned": False,
        "v0_2_result_overridden": False,
        "physical_inheritance_claim": False,
        "interpretation": (
            "Candidate diagnostic-ingestion admissibility only. This record can support SI-next "
            "relationship, uncertainty, lineage, and source-attribution diagnostics, but cannot "
            "alter a v0.2 physical result or assign a physical inheritance label."
        ),
    }


def canonical_record_sha256(record: dict[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
