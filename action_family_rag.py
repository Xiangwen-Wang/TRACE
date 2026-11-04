#!/usr/bin/env python3
"""Build a family-constrained action proposal via literature_rag + LLM."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from literature_rag import (
    _COMPONENT_FAMILY_ALIASES,
    _normalize_component_name as _lr_normalize_component_name,
    parse_concentration_value_strict,
    split_component_tokens,
    DecisionQuery,
    build_action_rag_payload,
    build_precedents,
    collect_component_families,
    DEFAULT_FAMILY_CLASSIFIER_MODEL,
    formulation_similarity_score,
    generate_proposal_with_openai,
    infer_context_tags_from_formulation,
    normalize_action_type,
    prepare_family_indexes_for_formulation,
    strategy_to_action_type,
    summarize_decision_in_plain_english,
)

DEFAULT_RAG_OBJECTIVE_PRIMARY = "unspecified_performance_objective"
DEFAULT_RAG_TARGET_PROPERTY = "cycling stability"
GOAL_ANALYSIS_REQUEST = (
    "Analyze how the proposed formulation change may improve adaptability for "
    "wide-temperature operation and high-voltage operation, and state possible trade-offs/risks."
)
OUTPUT_RULES_REQUEST = [
    "Use explicit, full material names in the proposal when possible; avoid ambiguous references like 'it', 'the solvent', or unspecified placeholders.",
    "Propose a small-step, conservative modification from the current formulation; avoid aggressive multi-parameter jumps.",
    "In the rationale, state which specific properties are supported to improve and which specific properties may worsen, and distinguish exact-state evidence, same-family evidence, and other similar evidence when available.",
    "For solvent_replace, output one concrete replacement target; do not output an empty change, a remove-only change, or an empty resulting_solvents object.",
    "If positive_feedback_trajectory.mode is continue_solvent_replace_trajectory, output one concrete solvent replacement for one remaining source solvent only.",
]
GROUNDING_RULES_REQUEST = [
    "Treat rag_information.family_guidance and rag_information.action_conditioned_examples as the primary prior knowledge for grounding the proposal.",
    "supporting_precedents may be an empty list; explicit citation is optional.",
    "If rag_information.family_guidance.constraint_mode is hard_allowlist, treat allowed_target_families as a hard constraint for choosing the new solvent/salt/additive.",
    "If rag_information.family_guidance.constraint_mode is advisory_from_rag, use target_family_statistics as prior evidence rather than a mandatory rule.",
    "If rag_information.family_guidance includes avoid_target_families, do not choose a new component from those families.",
    "Treat avoid_target_families as a mandatory exclusion list even when constraint_mode is advisory_from_rag.",
    "Before finalizing the JSON, verify that every newly introduced solvent/salt/additive is outside avoid_target_families.",
    "Read positive_patterns in evidence-tier order: exact_state_matches first, then secondary_same_family, then other_similar_matches. Each entry is one complete literature record containing before/after compositions, change, property, outcome, and conditions.",
    "Read negative_patterns in the same evidence-tier order to understand which complete literature records showed failure, degradation, or trade-offs.",
    "If forbidden_reversion_patterns is provided, do not propose a change that reverts the current formulation back to an older baseline already seen in literature.",
    "If positive_feedback_trajectory is present, treat it as accepted in-session experimental feedback from the current interactive optimization loop.",
    "If retry_directive is present, every change in retry_directive.do_not_repeat_rejected_changes was already proposed from the current formulation and rejected by the evaluator: do not propose any of them again (same molecule at the same or a nearly identical amount, or the same change reworded); propose a genuinely different change of the same action type.",
    "If positive_feedback_trajectory.mode is continue_solvent_replace_trajectory, continue that solvent-replacement trajectory conservatively: replace exactly one remaining source solvent, and do not modify completed or exhausted source solvents.",
]

EVIDENCE_TIER_ORDER = [
    "exact_state_matches",
    "secondary_same_family",
    "other_similar_matches",
]

EVIDENCE_TIER_DESCRIPTIONS = {
    "exact_state_matches": "Precedents whose baseline formulation exactly matches the current formulation before the action.",
    "secondary_same_family": "Precedents whose baseline is not identical, but both solvent-family set and salt-family set match the current formulation.",
    "other_similar_matches": "Other action-matched precedents considered only after removing exact and same-family baselines, then ranked by formulation similarity.",
}

EVIDENCE_TIER_LIMITS = {
    "exact_state_matches": 999,
    "secondary_same_family": 8,
    "other_similar_matches": 8,
}

PROPOSAL_CACHE_VERSION = "proposal_prompt_v2_avoid_family_enforced"


def _extract_cache_constraint_context(value: Any) -> Dict[str, Any]:
    collected: List[Dict[str, Any]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            picked: Dict[str, Any] = {}
            for key in (
                "constraint_mode",
                "avoid_target_families",
                "allowed_target_families",
                "current_component_families",
                "selection_rule",
            ):
                if key in node:
                    picked[str(key)] = node.get(key)
            if picked:
                collected.append(picked)
            for child in node.values():
                _walk(child)
        elif isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(value)
    return {"family_constraint_context": collected}


def build_cache_key(llm_input_payload: Dict[str, Any], model: str, temperature: float) -> str:
    key_obj = {
        "proposal_cache_version": PROPOSAL_CACHE_VERSION,
        "model": model,
        "temperature": float(temperature),
        "llm_input": llm_input_payload,
        "constraint_context": _extract_cache_constraint_context(llm_input_payload),
    }
    raw = json.dumps(key_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_component_for_state_key(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_component_for_state_key(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, list):
        return [_normalize_component_for_state_key(v) for v in value]
    if value is None:
        return ""
    return str(value).strip()


_STATE_KEY_DECIMALS = 2


def _canonical_component_name(name: Any) -> str:
    # Collapse aliases to one deterministic group key
    key = _lr_normalize_component_name(name)
    if not key:
        return ""
    group = {key} | {_lr_normalize_component_name(x) for x in _COMPONENT_FAMILY_ALIASES.get(key, [])}
    return min(k for k in group if k)


def _first_number(value: Any) -> Optional[float]:
    text = str(value if value is not None else "")
    pattern = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    match = re.search(pattern, text)
    return float(match.group()) if match else None


def _component_amounts(value: Any) -> Dict[str, Optional[float]]:
    items: List[Tuple[Any, Any]] = []
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                items.append((item.get("name"), item.get("ratio", item.get("amount"))))
            else:
                items.append((item, None))
    out: Dict[str, Optional[float]] = {}
    for name, amount in items:
        key = _canonical_component_name(name)
        if key:
            out[key] = _first_number(amount)
    return out


def _canonical_state(comp: Dict[str, Any]) -> Dict[str, Any]:
    """Representation-invariant view of a composition for exact state matching.
    """
    c = comp or {}
    nd = _STATE_KEY_DECIMALS

    salts = sorted({_canonical_component_name(s) for s in split_component_tokens(c.get("salt", ""))} - {""})
    conc = parse_concentration_value_strict(c.get("salt_concentration"))
    if conc is None:
        conc = _first_number(c.get("salt_concentration"))

    solvents = _component_amounts(c.get("solvents") or c.get("solvents_with_smiles") or {})
    ratios = list(solvents.values())
    total = sum(ratios) if ratios and all(r is not None for r in ratios) else 0.0
    if total > 0:
        solvent_state = {k: round(v / total, nd) for k, v in solvents.items()}
    else:
        solvent_state = {k: None for k in solvents}  # ratios unknown: match on identity only

    additives = {
        k: (None if v is None else round(v, nd))
        for k, v in _component_amounts(c.get("additives") or {}).items()
    }
    return {
        "salts": salts,
        "salt_concentration": None if conc is None else round(conc, nd),
        "solvents": solvent_state,
        "additives": additives,
    }


def _normalized_composition_key(comp: Dict[str, Any]) -> str:
    return json.dumps(_canonical_state(comp), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _as_string_map(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in value.items():
        key = str(k).strip()
        if not key:
            continue
        out[key] = str(v).strip() if v is not None else ""
    return out


def _split_component_names(value: Any) -> List[str]:
    raw = str(value).strip()
    if not raw:
        return []
    out: List[str] = []
    seen = set()
    for token in re.split(r"\s*(?:\+|/|,|;|&|\band\b)\s*", raw):
        item = str(token).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _map_items(items: Dict[str, str], names: List[str]) -> List[Dict[str, str]]:
    return [{"name": name, "value": items.get(name, "")} for name in names if name]


def _normalize_component_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").strip().lower())


def _families_for_names(names: List[str], family_index: Dict[str, List[str]]) -> List[str]:
    families = set()
    for name in names:
        key = _normalize_component_name(name)
        if not key:
            continue
        for family in family_index.get(key, []):
            families.add(str(family))
    return sorted(families)


def _family_dimension_for_action(action_type: str) -> str:
    action = normalize_action_type(action_type)
    if action in {"salt_replace", "add_new_salt", "salt_concentration_shift"}:
        return "salts"
    if action in {"solvent_replace", "add_new_solvent", "remove_solvent", "solvent_concentration_shift"}:
        return "solvents"
    if action == "additive_change":
        return "additives"
    return ""


def _component_names_for_action(
    before_comp: Dict[str, Any],
    after_comp: Dict[str, Any],
    action_type: str,
) -> Tuple[List[str], List[str]]:
    action = normalize_action_type(action_type)
    before_solvents = _as_string_map((before_comp or {}).get("solvents", {}))
    after_solvents = _as_string_map((after_comp or {}).get("solvents", {}))
    before_additives = _as_string_map((before_comp or {}).get("additives", {}))
    after_additives = _as_string_map((after_comp or {}).get("additives", {}))

    if action in {"salt_replace", "add_new_salt", "salt_concentration_shift"}:
        before_names = _split_component_names((before_comp or {}).get("salt", ""))
        after_names = _split_component_names((after_comp or {}).get("salt", ""))
        before_lower = {name.lower() for name in before_names}
        if action == "salt_replace":
            return before_names, [name for name in after_names if name.lower() not in before_lower] or after_names
        if action == "add_new_salt":
            return before_names, [name for name in after_names if name.lower() not in before_lower] or after_names
        return before_names or after_names, after_names or before_names

    if action in {"solvent_replace", "add_new_solvent", "remove_solvent", "solvent_concentration_shift"}:
        before_names = list(before_solvents.keys())
        after_names = list(after_solvents.keys())
        if action == "solvent_replace":
            source = [name for name in before_names if name not in after_solvents] or before_names
            target = [name for name in after_names if name not in before_solvents] or after_names
            return source, target
        if action == "add_new_solvent":
            return before_names, [name for name in after_names if name not in before_solvents] or after_names
        if action == "remove_solvent":
            return [name for name in before_names if name not in after_solvents] or before_names, after_names
        return before_names or after_names, after_names or before_names

    if action == "additive_change":
        before_names = list(before_additives.keys())
        after_names = list(after_additives.keys())
        changed_source = [name for name in before_names if name not in after_additives]
        changed_target = [name for name in after_names if name not in before_additives]
        adjusted = [
            name
            for name in before_names
            if name in after_additives and before_additives.get(name) != after_additives.get(name)
        ]
        source = changed_source + [name for name in adjusted if name not in changed_source]
        target = changed_target + [name for name in adjusted if name not in changed_target]
        return source or before_names, target or after_names

    return [], []


def _precedent_matches_action(precedent: Any, action_type: str) -> bool:
    normalized = normalize_action_type(action_type)
    rl_action = normalize_action_type(str(getattr(precedent, "rl_action_type", "")))
    if rl_action == normalized:
        return True
    actions = getattr(precedent, "action_types", []) or []
    normalized_actions = {normalize_action_type(str(a)) for a in actions if str(a).strip()}
    return normalized in normalized_actions


def _baseline_group_key(precedent: Any) -> str:
    key_obj = {
        "doi": str(getattr(precedent, "doi", "")).strip(),
        "before_id": str(getattr(precedent, "before_id", "")).strip(),
        "before_comp": _normalize_component_for_state_key(getattr(precedent, "before_comp", {}) or {}),
    }
    return json.dumps(key_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _change_summary_for_action(before_comp: Dict[str, Any], after_comp: Dict[str, Any], action_type: str) -> Dict[str, Any]:
    action = normalize_action_type(action_type)
    before_salt = str((before_comp or {}).get("salt", "")).strip()
    after_salt = str((after_comp or {}).get("salt", "")).strip()
    before_sc = str((before_comp or {}).get("salt_concentration", "")).strip()
    after_sc = str((after_comp or {}).get("salt_concentration", "")).strip()
    before_solvents = _as_string_map((before_comp or {}).get("solvents", {}))
    after_solvents = _as_string_map((after_comp or {}).get("solvents", {}))
    before_additives = _as_string_map((before_comp or {}).get("additives", {}))
    after_additives = _as_string_map((after_comp or {}).get("additives", {}))

    if action == "salt_replace":
        summary = {"from_salt": before_salt, "to_salt": after_salt}
        if before_sc or after_sc:
            summary["salt_concentration"] = {"from": before_sc, "to": after_sc}
        return summary

    if action == "add_new_salt":
        before_names = _split_component_names(before_salt)
        after_names = _split_component_names(after_salt)
        before_lower = {name.lower() for name in before_names}
        added = [name for name in after_names if name.lower() not in before_lower]
        summary = {
            "baseline_salt": before_salt,
            "added_salts": added or after_names,
        }
        if before_sc or after_sc:
            summary["salt_concentration"] = {"from": before_sc, "to": after_sc}
        return summary

    if action == "salt_concentration_shift":
        return {
            "salt": after_salt or before_salt,
            "concentration_change": {"from": before_sc, "to": after_sc},
        }

    if action == "solvent_replace":
        before_names = list(before_solvents.keys())
        after_names = list(after_solvents.keys())
        removed = [name for name in before_names if name not in after_solvents]
        added = [name for name in after_names if name not in before_solvents]
        replacements = [{"from": old_name, "to": new_name} for old_name, new_name in zip(removed, added)]
        summary: Dict[str, Any] = {
            "replacements": replacements,
            "removed_solvents": _map_items(before_solvents, removed),
            "added_solvents": _map_items(after_solvents, added),
        }
        common = [name for name in before_names if name in after_solvents and before_solvents.get(name) != after_solvents.get(name)]
        if common:
            summary["co_adjusted_solvents"] = [
                {"name": name, "from": before_solvents.get(name, ""), "to": after_solvents.get(name, "")}
                for name in common
            ]
        return summary

    if action == "add_new_solvent":
        added = [name for name in after_solvents.keys() if name not in before_solvents]
        return {"added_solvents": _map_items(after_solvents, added)}

    if action == "remove_solvent":
        removed = [name for name in before_solvents.keys() if name not in after_solvents]
        return {"removed_solvents": _map_items(before_solvents, removed)}

    if action == "solvent_concentration_shift":
        changed = [
            {"name": name, "from": before_solvents.get(name, ""), "to": after_solvents.get(name, "")}
            for name in before_solvents.keys()
            if name in after_solvents and before_solvents.get(name) != after_solvents.get(name)
        ]
        return {"ratio_changes": changed}

    if action == "additive_change":
        added = [name for name in after_additives.keys() if name not in before_additives]
        removed = [name for name in before_additives.keys() if name not in after_additives]
        changed = [
            {"name": name, "from": before_additives.get(name, ""), "to": after_additives.get(name, "")}
            for name in before_additives.keys()
            if name in after_additives and before_additives.get(name) != after_additives.get(name)
        ]
        return {
            "added_additives": _map_items(after_additives, added),
            "removed_additives": _map_items(before_additives, removed),
            "adjusted_additives": changed,
        }

    return {}


def _property_progress_for_precedent(precedent: Any) -> List[Dict[str, Any]]:
    props = getattr(precedent, "evaluated_properties", []) or []
    rows: List[Dict[str, Any]] = []
    for prop in props:
        if not isinstance(prop, dict):
            continue
        property_name = str(prop.get("property", "")).strip()
        outcome = str(prop.get("outcome", "")).strip()
        conditions = [str(x).strip() for x in (prop.get("conditions", []) or []) if str(x).strip()]
        if property_name or outcome or conditions:
            rows.append(
                {
                    "property": property_name or "unknown_property",
                    "outcome": outcome,
                    "conditions": conditions,
                    "polarity": str(getattr(precedent, "polarity", "mixed/uncertain")),
                }
            )
    if rows:
        return rows
    outcome = str(getattr(precedent, "outcome", "")).strip()
    if not outcome:
        return []
    return [
        {
            "property": str(getattr(precedent, "property_name", "")).strip() or "formulation_summary",
            "outcome": outcome,
            "conditions": [str(x).strip() for x in (getattr(precedent, "conditions", []) or []) if str(x).strip()],
            "polarity": str(getattr(precedent, "polarity", "mixed/uncertain")),
        }
    ]


def _escalation_target_key(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").strip().lower())


def _trim_action_precedents_for_escalation(
    matched: List[Any],
    normalized_action: str,
    exclude_targets: Any,
    trim_dominant: bool,
) -> Dict[str, Any]:
    """Drop accepted + distribution-dominant target molecules from `matched`
    """
    summary: Dict[str, Any] = {"applied": False, "accepted_dropped": [], "dominant_dropped": {}}
    accepted_keys = {_escalation_target_key(t) for t in (exclude_targets or [])}
    accepted_keys.discard("")
    if not accepted_keys and not trim_dominant:
        summary["matched"] = matched
        return summary

    per_prec_keys: List[set] = []
    counter: Dict[str, int] = {}
    label_for_key: Dict[str, str] = {}
    for p in matched:
        before_comp = getattr(p, "before_comp", {}) or {}
        after_comp = getattr(p, "after_comp", {}) or {}
        _, target_names = _component_names_for_action(before_comp, after_comp, normalized_action)
        keys = set()
        for name in (target_names or []):
            k = _escalation_target_key(name)
            if not k:
                continue
            keys.add(k)
            label_for_key.setdefault(k, str(name))
        per_prec_keys.append(keys)
        for k in keys:
            counter[k] = counter.get(k, 0) + 1

    drop_keys = set(accepted_keys)
    dominant_keys: set = set()
    if trim_dominant and len(counter) >= 4:
        counts = list(counter.values())
        mean = sum(counts) / len(counts)
        std = (sum((c - mean) ** 2 for c in counts) / len(counts)) ** 0.5
        threshold = mean + 2.0 * std
        dominant_keys = {k for k, c in counter.items() if c > threshold}
        drop_keys |= dominant_keys

    if not drop_keys:
        summary["matched"] = matched
        return summary

    filtered = [p for p, keys in zip(matched, per_prec_keys) if not (keys & drop_keys)]
    if not filtered:
        # Relax: drop only accepted molecules; if still empty, keep original.
        filtered = [p for p, keys in zip(matched, per_prec_keys) if not (keys & accepted_keys)]
        dominant_keys = set()
    summary["matched"] = filtered or matched
    summary["applied"] = bool(filtered and len(filtered) != len(matched))
    summary["accepted_dropped"] = sorted(label_for_key.get(k, k) for k in accepted_keys if k in counter)
    summary["dominant_dropped"] = {label_for_key.get(k, k): counter[k] for k in dominant_keys}
    return summary


def build_action_conditioned_examples(
    current_formulation: Dict[str, Any],
    action_type: str,
    all_precedents: List[Any],
    family_classifier_model: str = DEFAULT_FAMILY_CLASSIFIER_MODEL,
    exclude_targets: Any = None,
    trim_dominant: bool = False,
) -> Dict[str, Any]:
    current_comp = (current_formulation or {}).get("composition", {}) or {}
    normalized_action = normalize_action_type(action_type)
    matched = [p for p in (all_precedents or []) if _precedent_matches_action(p, normalized_action)]
    if not matched:
        return {
            "selected_action_type": normalized_action,
            "matched_precedent_count": 0,
            "note": "No exact action-matched precedents found for baseline-conditioned summarization.",
        }

    escalation_filter = _trim_action_precedents_for_escalation(
        matched, normalized_action, exclude_targets, trim_dominant
    )
    matched = escalation_filter["matched"]

    component_dimension = _family_dimension_for_action(normalized_action)
    family_tables, family_indexes = prepare_family_indexes_for_formulation(
        current_formulation=current_formulation,
        base_dir=Path(__file__).resolve().parent,
        model=family_classifier_model,
    )
    family_index = family_indexes.get(component_dimension, {})
    current_families = collect_component_families(current_comp, family_indexes)
    current_solvent_families = sorted(current_families.get("solvents", set()))
    current_salt_families = sorted(current_families.get("salts", set()))
    family_secondary_enabled = normalized_action not in {"salt_concentration_shift", "solvent_concentration_shift"}
    current_comp_key = _normalized_composition_key(current_comp)

    def _tier_rank(tier: str) -> int:
        try:
            return EVIDENCE_TIER_ORDER.index(tier)
        except ValueError:
            return len(EVIDENCE_TIER_ORDER)

    def _same_secondary_family(before_comp: Dict[str, Any]) -> bool:
        if not family_secondary_enabled:
            return False
        if not current_solvent_families or not current_salt_families:
            return False
        before_families = collect_component_families(before_comp, family_indexes)
        before_solvent_families = sorted(before_families.get("solvents", set()))
        before_salt_families = sorted(before_families.get("salts", set()))
        return (
            bool(before_solvent_families)
            and bool(before_salt_families)
            and before_solvent_families == current_solvent_families
            and before_salt_families == current_salt_families
        )

    def _evidence_tier(before_comp: Dict[str, Any]) -> str:
        if _normalized_composition_key(before_comp) == current_comp_key:
            return "exact_state_matches"
        if _same_secondary_family(before_comp):
            return "secondary_same_family"
        return "other_similar_matches"

    def _sorted_records(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            items,
            key=lambda row: (
                float(row.get("baseline_similarity_score", 0.0)),
                float(row.get("variant_similarity_score", 0.0)),
                1 if bool(row.get("explicitly_explored", False)) else 0,
                str(row.get("property", "")),
            ),
            reverse=True,
        )

    def _append_raw_pattern(
        store: Dict[str, List[Dict[str, Any]]],
        seen: Dict[str, set],
        tier: str,
        precedent: Any,
        before_comp: Dict[str, Any],
        after_comp: Dict[str, Any],
        change_summary: Dict[str, Any],
        property_row: Dict[str, Any],
        target_families: List[str],
        baseline_similarity: float,
        variant_similarity: float,
    ) -> None:
        entry = {
            "precedent_id": getattr(precedent, "precedent_id", ""),
            "doi": getattr(precedent, "doi", ""),
            "before_id": getattr(precedent, "before_id", ""),
            "after_id": getattr(precedent, "after_id", ""),
            "action_type": normalized_action,
            "before_composition": before_comp,
            "after_composition": after_comp,
            "change": change_summary,
            "property": str(property_row.get("property", "unknown_property")).strip() or "unknown_property",
            "outcome": str(property_row.get("outcome", "")).strip(),
            "conditions": [str(x).strip() for x in (property_row.get("conditions", []) or []) if str(x).strip()],
            "polarity": str(property_row.get("polarity", getattr(precedent, "polarity", "mixed/uncertain"))),
            "explicitly_explored": bool(getattr(precedent, "action_explicitly_explored", False)),
            "baseline_similarity_score": round(float(baseline_similarity), 4),
            "variant_similarity_score": round(float(variant_similarity), 4),
            "target_families": list(target_families),
        }
        dedupe_key = json.dumps(
            {
                "precedent_id": entry["precedent_id"],
                "before_id": entry["before_id"],
                "after_id": entry["after_id"],
                "property": entry["property"],
                "outcome": entry["outcome"],
                "conditions": entry["conditions"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if dedupe_key in seen[tier]:
            return
        seen[tier].add(dedupe_key)
        store[tier].append(entry)

    positive_patterns: Dict[str, List[Dict[str, Any]]] = {tier: [] for tier in EVIDENCE_TIER_ORDER}
    negative_patterns: Dict[str, List[Dict[str, Any]]] = {tier: [] for tier in EVIDENCE_TIER_ORDER}
    positive_seen = {tier: set() for tier in EVIDENCE_TIER_ORDER}
    negative_seen = {tier: set() for tier in EVIDENCE_TIER_ORDER}
    forbidden_reversion_patterns: List[Dict[str, Any]] = []
    seen_forbidden_keys = set()
    evidence_tier_counts = {
        tier: {
            "tier": tier,
            "definition": EVIDENCE_TIER_DESCRIPTIONS.get(tier, ""),
            "precedent_count": 0,
            "baseline_group_count": 0,
        }
        for tier in EVIDENCE_TIER_ORDER
    }
    prepared_rows: List[Dict[str, Any]] = []

    for precedent in matched:
        before_comp = getattr(precedent, "before_comp", {}) or {}
        after_comp = getattr(precedent, "after_comp", {}) or {}
        change_summary = _change_summary_for_action(before_comp, after_comp, normalized_action)
        source_names, target_names = _component_names_for_action(before_comp, after_comp, normalized_action)
        target_families = _families_for_names(target_names, family_index)
        property_progress = _property_progress_for_precedent(precedent)
        polarity = str(getattr(precedent, "polarity", "mixed/uncertain"))
        evidence_tier = _evidence_tier(before_comp)
        baseline_similarity = formulation_similarity_score(current_comp, before_comp)
        variant_similarity = formulation_similarity_score(current_comp, after_comp)
        prepared_rows.append(
            {
                "precedent": precedent,
                "before_comp": before_comp,
                "after_comp": after_comp,
                "change_summary": change_summary,
                "source_names": source_names,
                "target_names": target_names,
                "target_families": target_families,
                "property_progress": property_progress,
                "polarity": polarity,
                "evidence_tier": evidence_tier,
                "baseline_similarity_score": baseline_similarity,
                "variant_similarity_score": variant_similarity,
                "explicitly_explored": bool(getattr(precedent, "action_explicitly_explored", False)),
                "baseline_group_key": _baseline_group_key(precedent),
            }
        )
        evidence_tier_counts[evidence_tier]["precedent_count"] += 1

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in prepared_rows:
        groups.setdefault(str(row["baseline_group_key"]), []).append(row)

    for rows in groups.values():
        tier = str(rows[0].get("evidence_tier", "other_similar_matches"))
        evidence_tier_counts[tier]["baseline_group_count"] += 1

    ranked_rows = []
    for group_rows in sorted(
        groups.values(),
        key=lambda rows: (
            _tier_rank(str(rows[0].get("evidence_tier", "other_similar_matches"))),
            -max(float(row.get("baseline_similarity_score", 0.0)) for row in rows),
            -sum(1 for row in rows if str(row.get("polarity", "")) == "positive"),
            -len(rows),
        ),
    ):
        ranked_rows.extend(_sorted_records(group_rows))

    for row in ranked_rows:
        precedent = row["precedent"]
        before_comp = row["before_comp"]
        after_comp = row["after_comp"]
        change_summary = row["change_summary"]
        target_families = row["target_families"]
        property_progress = row["property_progress"]
        evidence_tier = str(row["evidence_tier"])
        baseline_similarity = float(row["baseline_similarity_score"])
        variant_similarity = float(row["variant_similarity_score"])

        if (
            _normalized_composition_key(after_comp) == current_comp_key
            and _normalized_composition_key(before_comp) != current_comp_key
        ):
            revert_change_summary = _change_summary_for_action(after_comp, before_comp, normalized_action)
            forbidden_key = json.dumps(revert_change_summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if forbidden_key not in seen_forbidden_keys:
                seen_forbidden_keys.add(forbidden_key)
                forbidden_reversion_patterns.append(
                    {
                        "reason": "current_state_matches_literature_variant",
                        "precedent_id": getattr(precedent, "precedent_id", ""),
                        "doi": getattr(precedent, "doi", ""),
                        "before_id": getattr(precedent, "before_id", ""),
                        "after_id": getattr(precedent, "after_id", ""),
                        "current_variant_composition": after_comp,
                        "baseline_composition": before_comp,
                        "do_not_revert_change": revert_change_summary,
                    }
                )

        for property_row in property_progress:
            row_polarity = str(property_row.get("polarity", row.get("polarity", "mixed/uncertain")))
            if row_polarity == "positive":
                _append_raw_pattern(
                    positive_patterns,
                    positive_seen,
                    evidence_tier,
                    precedent,
                    before_comp,
                    after_comp,
                    change_summary,
                    property_row,
                    target_families,
                    baseline_similarity,
                    variant_similarity,
                )
            elif row_polarity == "negative":
                _append_raw_pattern(
                    negative_patterns,
                    negative_seen,
                    evidence_tier,
                    precedent,
                    before_comp,
                    after_comp,
                    change_summary,
                    property_row,
                    target_families,
                    baseline_similarity,
                    variant_similarity,
                )

    result = {
        "selected_action_type": normalized_action,
        "tier_definitions": dict(EVIDENCE_TIER_DESCRIPTIONS),
        "tier_context": {
            "secondary_same_family_enabled": family_secondary_enabled,
            "required_same_families": {
                "solvents": current_solvent_families,
                "salts": current_salt_families,
            },
        },
        "tier_counts": evidence_tier_counts,
        "positive_patterns": {tier: _sorted_records(positive_patterns[tier]) for tier in EVIDENCE_TIER_ORDER},
        "negative_patterns": {tier: _sorted_records(negative_patterns[tier]) for tier in EVIDENCE_TIER_ORDER},
        "forbidden_reversion_patterns": forbidden_reversion_patterns,
    }

    if escalation_filter.get("applied"):
        result["escalation_filter"] = {k: v for k, v in escalation_filter.items() if k != "matched"}
    return result


def _normalize_allowed_target_families(allowed_target_families: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(allowed_target_families, list):
        return None
    normalized: List[Dict[str, Any]] = []
    for item in allowed_target_families:
        if isinstance(item, dict):
            family = str(item.get("family", "")).strip()
            if not family:
                continue
            normalized.append(dict(item))
            continue
        family = str(item).strip()
        if family:
            normalized.append({"family": family})
    return normalized


def _normalize_family_name_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen = set()
    for item in value:
        if isinstance(item, dict):
            family = str(item.get("family", "")).strip()
        else:
            family = str(item).strip()
        if not family or family in seen:
            continue
        seen.add(family)
        out.append(family)
    return out


def _print_action_conditioned_summary(action_type: str, action_conditioned_examples: Dict[str, Any]) -> None:
    positive_patterns = (action_conditioned_examples or {}).get("positive_patterns", {}) or {}
    negative_patterns = (action_conditioned_examples or {}).get("negative_patterns", {}) or {}
    forbidden_reversion_patterns = (action_conditioned_examples or {}).get("forbidden_reversion_patterns", []) or []

    print("LLM input item counts:")
    for tier in EVIDENCE_TIER_ORDER:
        count = len(positive_patterns.get(tier, []) or []) if isinstance(positive_patterns, dict) else 0
        print(f"positive_patterns.{tier}: {count} raw records")
    for tier in EVIDENCE_TIER_ORDER:
        count = len(negative_patterns.get(tier, []) or []) if isinstance(negative_patterns, dict) else 0
        print(f"negative_patterns.{tier}: {count} raw records")
    print(f"forbidden_reversion_patterns: {len(forbidden_reversion_patterns)} raw records")


def build_family_guidance(
    current_formulation: Dict[str, Any],
    action_type: str,
    all_precedents: List[Any],
    allowed_target_families: Any = None,
    avoid_target_families: Any = None,
    family_classifier_model: str = DEFAULT_FAMILY_CLASSIFIER_MODEL,
) -> Dict[str, Any]:
    normalized_action = normalize_action_type(action_type)
    component_dimension = _family_dimension_for_action(normalized_action)
    if not component_dimension:
        return {
            "applies": False,
            "selected_action_type": normalized_action,
            "note": "No family guidance is defined for this action type.",
        }

    family_tables, family_indexes = prepare_family_indexes_for_formulation(
        current_formulation=current_formulation,
        base_dir=Path(__file__).resolve().parent,
        model=family_classifier_model,
    )

    current_comp = (current_formulation or {}).get("composition", {}) or {}
    current_families_all = collect_component_families(current_comp, family_indexes)
    current_component_families = sorted(current_families_all.get(component_dimension, set()))
    matched = [p for p in (all_precedents or []) if _precedent_matches_action(p, normalized_action)]

    if normalized_action in {"salt_concentration_shift", "solvent_concentration_shift"}:
        return {
            "applies": False,
            "selected_action_type": normalized_action,
            "component_dimension": component_dimension,
            "note": "Family guidance is skipped for concentration/ratio-only actions because component identity does not change.",
        }

    target_family_stats: Dict[str, Dict[str, Any]] = {}
    source_family_stats: Dict[str, Dict[str, Any]] = {}
    transition_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}

    family_index = family_indexes.get(component_dimension, {})
    family_table = family_tables.get(component_dimension, {})

    for precedent in matched:
        before_comp = getattr(precedent, "before_comp", {}) or {}
        after_comp = getattr(precedent, "after_comp", {}) or {}
        source_names, target_names = _component_names_for_action(before_comp, after_comp, normalized_action)
        source_families = _families_for_names(source_names, family_index)
        target_families = _families_for_names(target_names, family_index)
        polarity = str(getattr(precedent, "polarity", "mixed/uncertain"))

        for family in source_families:
            bucket = source_family_stats.setdefault(
                family,
                {"family": family, "support_count": 0, "positive_count": 0, "example_components": []},
            )
            bucket["support_count"] += 1
            if polarity == "positive":
                bucket["positive_count"] += 1
            for name in source_names:
                if name not in bucket["example_components"] and len(bucket["example_components"]) < 4:
                    bucket["example_components"].append(name)

        for family in target_families:
            bucket = target_family_stats.setdefault(
                family,
                {"family": family, "support_count": 0, "positive_count": 0, "example_components": []},
            )
            bucket["support_count"] += 1
            if polarity == "positive":
                bucket["positive_count"] += 1
            for name in target_names:
                if name not in bucket["example_components"] and len(bucket["example_components"]) < 4:
                    bucket["example_components"].append(name)

        for source_family in source_families or ["unknown_family"]:
            for target_family in target_families or ["unknown_family"]:
                bucket = transition_stats.setdefault(
                    (source_family, target_family),
                    {
                        "from_family": source_family,
                        "to_family": target_family,
                        "count": 0,
                        "positive_count": 0,
                        "example_precedents": [],
                    },
                )
                bucket["count"] += 1
                if polarity == "positive":
                    bucket["positive_count"] += 1
                precedent_ref = {
                    "precedent_id": getattr(precedent, "precedent_id", ""),
                    "doi": getattr(precedent, "doi", ""),
                    "before_id": getattr(precedent, "before_id", ""),
                    "after_id": getattr(precedent, "after_id", ""),
                }
                if precedent_ref not in bucket["example_precedents"] and len(bucket["example_precedents"]) < 3:
                    bucket["example_precedents"].append(precedent_ref)

    allowed_target = sorted(
        target_family_stats.values(),
        key=lambda item: (
            int(item.get("positive_count", 0)),
            int(item.get("support_count", 0)),
            str(item.get("family", "")),
        ),
        reverse=True,
    )
    for item in allowed_target:
        item["known_family_members"] = list(family_table.get(str(item.get("family", "")), []) or [])[:12]

    allowed_source = sorted(
        source_family_stats.values(),
        key=lambda item: (
            int(item.get("positive_count", 0)),
            int(item.get("support_count", 0)),
            str(item.get("family", "")),
        ),
        reverse=True,
    )

    observed_transitions = sorted(
        transition_stats.values(),
        key=lambda item: (
            int(item.get("positive_count", 0)),
            int(item.get("count", 0)),
            str(item.get("from_family", "")),
            str(item.get("to_family", "")),
        ),
        reverse=True,
    )

    avoid_list = _normalize_family_name_list(avoid_target_families)
    avoid_set = set(avoid_list)
    if avoid_set:
        allowed_target = [item for item in allowed_target if str(item.get("family", "")) not in avoid_set]
        observed_transitions = [item for item in observed_transitions if str(item.get("to_family", "")) not in avoid_set]

    override = _normalize_allowed_target_families(allowed_target_families)
    constraint_mode = "advisory_from_rag"
    hard_allowed_target: List[Dict[str, Any]] = []
    if override is not None:
        constraint_mode = "hard_allowlist"
        filtered_override = [item for item in override if str(item.get("family", "")) not in avoid_set]
        for item in filtered_override:
            item["known_family_members"] = list(family_table.get(str(item.get("family", "")), []) or [])[:12]
        hard_allowed_target = filtered_override

    return {
        "applies": True,
        "constraint_mode": constraint_mode,
        "selected_action_type": normalized_action,
        "component_dimension": component_dimension,
        "current_component_families": current_component_families,
        "allowed_source_families": allowed_source[:6],
        "target_family_statistics": allowed_target[:6],
        "allowed_target_families": hard_allowed_target[:6],
        "avoid_target_families": avoid_list[:6],
        "observed_family_transitions": observed_transitions[:8],
        "selection_rule": (
            "Use target_family_statistics and observed_family_transitions as literature-derived prior evidence. "
            "Only when constraint_mode is hard_allowlist must the new component be chosen from allowed_target_families. "
            "Always avoid families listed in avoid_target_families."
        ),
    }


def build_llm_output_payload(
    payload: Dict[str, Any],
    selected_action: str,
    llm_mode: str,
    proposal: Any,
    proposal_error: Any,
) -> Dict[str, Any]:
    inp = payload.get("input", {}) or {}
    out = {
        "model": llm_mode,
        "action_strategy": selected_action,
        "action_type": inp.get("action_type"),
        "full_formulation": inp.get("current_formulation"),
        "proposal": proposal,
        "proposal_error": proposal_error,
        "hard_constraints": payload.get("hard_constraints"),
        "precedents_used": payload.get("precedents", []),
        "tendencies": payload.get("tendencies"),
    }
    out["decision_summary_plain_english"] = summarize_decision_in_plain_english(out)
    return out


def generate_action_candidate(
    current_formulation: Dict[str, Any],
    action_strategy: str,
    state_objective: Any,
    papers: List[Dict[str, Any]],
    top_k: int,
    llm_mode: str,
    llm_temperature: float,
    cache: Optional[Dict[str, str]] = None,
    force_refresh: bool = False,
    allowed_target_families: Any = None,
    avoid_target_families: Any = None,
    all_precedents: Optional[List[Any]] = None,
    family_guidance: Optional[Dict[str, Any]] = None,
    trajectory_guidance: Optional[Dict[str, Any]] = None,
    rag_objective_primary: str = DEFAULT_RAG_OBJECTIVE_PRIMARY,
    rag_target_property: str = DEFAULT_RAG_TARGET_PROPERTY,
) -> Dict[str, Any]:
    if all_precedents is None:
        all_precedents = build_precedents(papers)

    action_type = strategy_to_action_type(action_strategy)
    query = DecisionQuery(
        current_formulation=current_formulation,
        action_strategy=action_strategy,
        action_type=action_type,
        target_property=rag_target_property,
        context_tags=infer_context_tags_from_formulation(current_formulation),
        objective_primary=rag_objective_primary,
    )
    payload = build_action_rag_payload(papers=papers, query=query, top_k=top_k)
    payload_llm_input = payload.get("llm_input", {}) or {}
    if state_objective is not None:
        payload_llm_input["objective"] = state_objective
    payload_llm_input["analysis_focus"] = {
        "goals": ["wide_temperature_adaptability", "high_voltage_adaptability"],
        "request": GOAL_ANALYSIS_REQUEST,
    }
    payload_llm_input["grounding_requirements"] = {
        "rules": GROUNDING_RULES_REQUEST,
    }
    payload_llm_input["output_requirements"] = {
        "rules": OUTPUT_RULES_REQUEST,
    }
    if isinstance(trajectory_guidance, dict) and trajectory_guidance.get("mode") == "retry_same_direction":
        payload_llm_input["retry_directive"] = trajectory_guidance
    elif isinstance(trajectory_guidance, dict) and trajectory_guidance:
        payload_llm_input["positive_feedback_trajectory"] = trajectory_guidance
        if trajectory_guidance.get("mode") == "escalate_within_direction":
            payload_llm_input["escalation_directive"] = trajectory_guidance

    rag_information = payload_llm_input.get("rag_information", {}) or {}
    if isinstance(rag_information, dict):
        is_escalation = (
            isinstance(trajectory_guidance, dict)
            and trajectory_guidance.get("mode") == "escalate_within_direction"
        )
        escalation_exclude_targets = (
            list(trajectory_guidance.get("avoid_repeating_targets", []) or []) if is_escalation else None
        )
        action_conditioned_examples = build_action_conditioned_examples(
            current_formulation=current_formulation,
            action_type=action_type,
            all_precedents=all_precedents,
            family_classifier_model=llm_mode,
            exclude_targets=escalation_exclude_targets,
            trim_dominant=is_escalation,
        )
        _print_action_conditioned_summary(action_type, action_conditioned_examples)
        if is_escalation:
            ef = action_conditioned_examples.get("escalation_filter", {}) or {}
            if ef.get("applied"):
                print(
                    "[escalation RAG trim] dropped accepted="
                    + str(ef.get("accepted_dropped", []))
                    + " dominant="
                    + str(ef.get("dominant_dropped", {}))
                )
        rag_information.pop("precedents", None)
        rag_information["action_conditioned_examples"] = {
            "selected_action_type": action_conditioned_examples.get("selected_action_type"),
            "tier_definitions": action_conditioned_examples.get("tier_definitions", {}),
            "tier_context": action_conditioned_examples.get("tier_context", {}),
            "positive_patterns": action_conditioned_examples.get("positive_patterns", {}),
            "negative_patterns": action_conditioned_examples.get("negative_patterns", {}),
            "forbidden_reversion_patterns": action_conditioned_examples.get("forbidden_reversion_patterns", []),
        }
        guidance = family_guidance if isinstance(family_guidance, dict) else build_family_guidance(
            current_formulation=current_formulation,
            action_type=action_type,
            all_precedents=all_precedents,
            allowed_target_families=allowed_target_families,
            avoid_target_families=avoid_target_families,
            family_classifier_model=llm_mode,
        )
        if isinstance(guidance, dict) and (allowed_target_families is not None or avoid_target_families is not None):
            guidance = build_family_guidance(
                current_formulation=current_formulation,
                action_type=action_type,
                all_precedents=all_precedents,
                allowed_target_families=allowed_target_families,
                avoid_target_families=avoid_target_families,
                family_classifier_model=llm_mode,
            )
        if isinstance(guidance, dict) and bool(guidance.get("applies")):
            rag_information["family_guidance"] = guidance
        else:
            rag_information.pop("family_guidance", None)
        payload_llm_input["rag_information"] = rag_information
    payload["llm_input"] = payload_llm_input

    proposal: Any = None
    proposal_error: Any = None
    proposal_source = "openai"
    cache_key = build_cache_key(payload_llm_input, model=llm_mode, temperature=llm_temperature)
    cache_updated = False
    cache_value: Optional[str] = None

    cached = (cache or {}).get(cache_key)
    try:
        if cached is not None and not force_refresh:
            proposal = cached
            proposal_source = "cache"
        else:
            proposal = generate_proposal_with_openai(
                payload,
                model=llm_mode,
                temperature=llm_temperature,
            )
            if isinstance(proposal, str):
                cache_updated = True
                cache_value = proposal
    except Exception as e:
        proposal_error = str(e)

    payload["proposal"] = proposal
    payload["proposal_error"] = proposal_error
    payload["proposal_source"] = proposal_source

    return {
        "action_strategy": action_strategy,
        "proposal_source": proposal_source,
        "payload": payload,
        "llm_output": build_llm_output_payload(
            payload,
            selected_action=action_strategy,
            llm_mode=llm_mode,
            proposal=proposal,
            proposal_error=proposal_error,
        ),
        "cache_key": cache_key,
        "cache_updated": cache_updated,
        "cache_value": cache_value,
    }
