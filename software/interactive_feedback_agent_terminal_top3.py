#!/usr/bin/env python3
"""Interactive formulation agent with terminal reward input.

Flow:
1) Load initial state (given formulation).
2) Run MLP prior and take the top-3 available action directions.
3) Take the highest-ranked direction not yet attempted (directions are tried in the prior's order).
4) Generate one RAG+LLM proposal for that direction.
5) Ask user to input reward in terminal: positive / negative / unchange.
6) If positive, apply the selected proposal to the current formulation.
7) If negative or unchange, stay inside the selected direction and retry, avoiding failed target families and duplicate proposals.
8) After repeated nonpositive feedback inside the same direction:
   - if the state is still the initial state, move to the highest-ranked remaining direction from the initial MLP top-3;
   - if the state has changed, rerank the current formulation, filter attempted directions from the current MLP top-3, and take the highest-ranked remaining one, or stop if none remain.
9) Continue until user quits.

This file is intentionally standalone and does not modify prior_suggestions.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from action_family_rag import generate_action_candidate as generate_action_candidate_via_rag
from literature_rag import (
    ACTION_SPACE,
    normalize_action_type,
    prepare_family_indexes_for_formulation,
    build_precedents,
    load_papers,
)
from prior_suggestions import (
    enabled_actions_from_state,
    extract_objective_for_llm,
    infer_action_with_mlp,
    llm_ready_formulation,
    load_cache,
    normalize_current_formulation,
    save_cache,
    top_enabled_actions,
)

REWARD_ALIASES = {
    "positive": "positive",
    "pos": "positive",
    "+": "positive",
    "p": "positive",
    "negative": "negative",
    "neg": "negative",
    "-": "negative",
    "n": "negative",
    "unchange": "unchange",
    "unchanged": "unchange",
    "same": "unchange",
    "u": "unchange",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive agent with terminal reward input")
    p.add_argument("--folder", type=Path, required=True, help="Folder containing initial_state.json and feedback outputs")
    p.add_argument("--prior", type=Path, default=Path("MLP_new/mlp_prior_state_new_v5.pt"), help="Path to trained MLP prior checkpoint (SMILES/structure)")
    p.add_argument("--data", type=Path, default=Path("literature_database.json"), help="Path to literature database JSON")
    p.add_argument("--top-k", type=int, default=5, help="Top-k for literature retrieval")
    p.add_argument("--llm-mode", type=str, default="gpt-5", help="OpenAI model")
    p.add_argument("--llm-temperature", type=float, default=0.0, help="LLM temperature")
    p.add_argument("--max-steps", type=int, default=20, help="Maximum interaction steps")
    p.add_argument(
        "--block-actions",
        type=str,
        default="",
        help="Comma-separated action directions to pre-block (skip entirely), e.g. 'additive_change'",
    )
    p.add_argument(
        "--nonpositive-threshold",
        type=int,
        default=5,
        help="How many negative/unchange rewards to tolerate for the same action before blocking that action",
    )
    p.add_argument(
        "--verbose-terminal",
        action="store_true",
        help="Print full ranking/proposal JSON in terminal instead of the concise approval-focused view",
    )
    p.add_argument(
        "--top-action-candidates",
        type=int,
        default=3,
        help="Size of the MLP top-k from which untried directions are taken in prior order",
    )
    p.add_argument(
        "--duplicate-regeneration-limit",
        type=int,
        default=3,
        help="How many times to force-regenerate if a retry proposal is rejected as duplicate or invalid before blocking that action",
    )
    p.add_argument(
        "--allow-new-molecules",
        dest="allow_new_molecules",
        action="store_true",
        default=True,
        help="Allow proposals to introduce solvent/salt/additive molecules that are not already present in the current family tables",
    )
    p.add_argument(
        "--known-molecules-only",
        dest="allow_new_molecules",
        action="store_false",
        help="Reject any proposed new solvent/salt/additive unless it already exists in the current family tables",
    )
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def state_payload_from_formulation(formulation: Dict[str, Any], objective: Any = None) -> Dict[str, Any]:
    payload = {
        "state0": {
            "composition": copy.deepcopy((formulation or {}).get("composition", {}) or {}),
        }
    }
    if objective is not None:
        payload["objective"] = objective
    return payload


def parse_proposal_json(proposal: Any) -> Dict[str, Any]:
    if isinstance(proposal, dict):
        return proposal
    if not isinstance(proposal, str):
        return {}
    raw = proposal.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _extract_numeric_prefix(value: Any) -> Optional[float]:
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value or ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _is_effectively_zero(value: Any) -> bool:
    if value in (None, "", False):
        return True
    if isinstance(value, (int, float)):
        return abs(float(value)) < 1e-12
    text = str(value).strip()
    if not text:
        return True
    numeric = _extract_numeric_prefix(text)
    return numeric is not None and abs(numeric) < 1e-12


def composition_summary(formulation: Dict[str, Any]) -> str:
    comp = (formulation or {}).get("composition", {}) or {}
    salt = str(comp.get("salt", "")).strip() or "N/A"
    salt_conc = str(comp.get("salt_concentration", "")).strip() or "N/A"
    solvents = comp.get("solvents", {}) or {}
    additives = comp.get("additives", {}) or {}

    visible_solvents = (
        [(str(k), v) for k, v in solvents.items() if not _is_effectively_zero(v)]
        if isinstance(solvents, dict)
        else []
    )
    visible_additives = (
        [(str(k), v) for k, v in additives.items() if not _is_effectively_zero(v)]
        if isinstance(additives, dict)
        else []
    )
    solvent_text = ", ".join(f"{k}:{v}" for k, v in visible_solvents) if visible_solvents else "None"
    additive_text = ", ".join(f"{k}:{v}" for k, v in visible_additives) if visible_additives else "None"
    return (
        f"salt={salt}; salt_concentration={salt_conc}; "
        f"solvents={solvent_text}; additives={additive_text}"
    )



def top_enabled_unblocked_actions(
    action_ranking: List[Dict[str, float]],
    state_payload: Dict[str, Any],
    blocked_actions: List[str],
    excluded_actions: Optional[List[str]] = None,
    n: int = 3,
) -> List[str]:
    """Top enabled actions after removing directions already blocked by feedback."""
    blocked = set(blocked_actions)
    excluded = set(excluded_actions or [])
    skip = blocked | excluded
    picks = [a for a in top_enabled_actions(action_ranking, state_payload, n=max(n + len(skip), n)) if a not in skip]
    if len(picks) >= n:
        return picks[:n]
    seen = set(picks)
    for row in action_ranking:
        action = str(row.get("action", ""))
        if action and action in ACTION_SPACE and action not in skip and action not in seen:
            picks.append(action)
            seen.add(action)
        if len(picks) >= n:
            break
    if picks:
        return picks[:n]

    # Fallback: if exclusions remove every option, keep honoring blocked directions
    # but allow previously attempted actions back in so the loop can continue.
    for row in action_ranking:
        action = str(row.get("action", ""))
        if action and action in ACTION_SPACE and action not in blocked and action not in seen:
            picks.append(action)
            seen.add(action)
        if len(picks) >= n:
            break
    return picks


def proposal_signature(proposal_obj: Dict[str, Any]) -> str:
    """Stable signature for duplicate-proposal suppression during retries."""
    if not proposal_obj:
        return ""
    action_type = str(proposal_obj.get("action_type", "")).strip()
    params = proposal_obj.get("parameters", {}) or {}
    try:
        params_norm = json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        params_norm = str(params)
    return f"{action_type}|{params_norm}"


def save_candidate_cache(cache_path: Path, cache: Dict[str, str], candidate_bundle: Dict[str, Any]) -> None:
    if candidate_bundle.get("cache_updated") and candidate_bundle.get("cache_key") and candidate_bundle.get("cache_value") is not None:
        cache[str(candidate_bundle["cache_key"])] = str(candidate_bundle["cache_value"])
        save_cache(cache_path, cache)


def generate_candidate_with_duplicate_guard(
    *,
    current_formulation: Dict[str, Any],
    action_strategy: str,
    state_objective: Any,
    papers: List[Dict[str, Any]],
    top_k: int,
    llm_mode: str,
    llm_temperature: float,
    cache: Dict[str, str],
    all_precedents: List[Dict[str, Any]],
    family_indexes: Dict[str, Dict[str, List[str]]],
    allowed_target_families: Optional[List[Dict[str, Any]]],
    avoid_target_families: List[str],
    allow_new_molecules: bool,
    rejected_signatures: List[str],
    duplicate_regeneration_limit: int,
    trajectory_guidance: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    """Generate a candidate, forcing refresh if it is invalid or exactly repeats a rejected proposal."""
    last_bundle: Dict[str, Any] = {}
    last_proposal: Dict[str, Any] = {}
    was_duplicate = False
    attempts = max(1, int(duplicate_regeneration_limit) + 1)
    rejected = set(rejected_signatures)
    for attempt_idx in range(attempts):
        bundle = generate_action_candidate_via_rag(
            current_formulation=current_formulation,
            action_strategy=action_strategy,
            state_objective=state_objective,
            papers=papers,
            top_k=top_k,
            llm_mode=llm_mode,
            llm_temperature=llm_temperature,
            cache=cache,
            force_refresh=(attempt_idx > 0),
            all_precedents=all_precedents,
            allowed_target_families=allowed_target_families,
            avoid_target_families=avoid_target_families,
            trajectory_guidance=trajectory_guidance,
        )
        last_bundle = bundle
        llm_output = bundle.get("llm_output", {}) or {}
        proposal_obj = parse_proposal_json(llm_output.get("proposal"))
        last_proposal = proposal_obj
        invalid_reason = _invalid_proposal_reason(current_formulation, proposal_obj) if proposal_obj else ""
        unknown_targets = proposal_unknown_target_components(
            proposal_obj,
            family_indexes,
        ) if proposal_obj and not allow_new_molecules else []
        if unknown_targets:
            invalid_reason = (
                "proposal introduces target molecule(s) outside the current family tables: "
                + ", ".join(unknown_targets)
            )
        violating_families = proposal_violates_avoid_target_families(
            proposal_obj,
            family_indexes,
            avoid_target_families,
        ) if proposal_obj else []
        if violating_families:
            invalid_reason = (
                "proposal introduces target(s) from forbidden families: "
                + ", ".join(violating_families)
            )
        trajectory_invalid_reason = invalid_solvent_trajectory_proposal_reason(
            current_formulation,
            proposal_obj,
            trajectory_guidance,
        ) if proposal_obj else ""
        if trajectory_invalid_reason:
            invalid_reason = trajectory_invalid_reason
        if proposal_obj and not invalid_reason:
            applied_form, applicable, apply_note = apply_proposal_to_formulation(copy.deepcopy(current_formulation), proposal_obj)
            if not applicable:
                invalid_reason = f"proposal cannot be applied to the current formulation ({apply_note})"
            elif formulation_change_summary(current_formulation, applied_form) == "no change to the formulation":
                invalid_reason = "proposal does not change the current formulation"
        if invalid_reason:
            if attempt_idx + 1 < attempts:
                continue
            llm_output["proposal_error"] = f"invalid proposal: {invalid_reason}"
            llm_output["proposal"] = None
            bundle["llm_output"] = llm_output
            was_duplicate = False
            break
        sig = proposal_signature(proposal_obj)
        was_duplicate = bool(sig and sig in rejected)
        if not was_duplicate:
            break
    return last_bundle, last_proposal, was_duplicate


def generate_single_direction_decision(
    *,
    current_formulation: Dict[str, Any],
    action_strategy: str,
    objective: Any,
    papers: List[Dict[str, Any]],
    all_precedents: List[Dict[str, Any]],
    top_k: int,
    llm_mode: str,
    llm_temperature: float,
    cache: Dict[str, str],
    cache_path: Path,
    family_tables: Dict[str, Dict[str, List[str]]],
    family_indexes: Dict[str, Dict[str, List[str]]],
    allow_new_molecules: bool,
    duplicate_regeneration_limit: int,
    candidate_actions: Optional[List[str]] = None,
    raw_candidate_actions: Optional[List[str]] = None,
    excluded_actions: Optional[List[str]] = None,
    schema: str = "interactive_single_candidate_v1",
    selection_path: str = "single_direction_direct",
    selection_reason: str = "",
    trajectory_guidance: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any], List[str], Dict[str, Any]]:
    allowed_target_families = build_allowed_target_families(
        action_strategy,
        family_tables,
        [],
        force_full_allowlist=not allow_new_molecules,
    )
    candidate_bundle, proposal_obj, _ = generate_candidate_with_duplicate_guard(
        current_formulation=current_formulation,
        action_strategy=action_strategy,
        state_objective=objective,
        papers=papers,
        top_k=top_k,
        llm_mode=llm_mode,
        llm_temperature=llm_temperature,
        cache=cache,
        all_precedents=all_precedents,
        family_indexes=family_indexes,
        allowed_target_families=allowed_target_families,
        avoid_target_families=[],
        allow_new_molecules=allow_new_molecules,
        rejected_signatures=[],
        duplicate_regeneration_limit=duplicate_regeneration_limit,
        trajectory_guidance=trajectory_guidance,
    )
    save_candidate_cache(cache_path, cache, candidate_bundle)
    llm_output = candidate_bundle.get("llm_output", {}) or {}
    proposal_obj, llm_output = sanitize_proposal_against_molecule_scope(
        proposal_obj,
        llm_output,
        family_indexes,
        allow_new_molecules,
    )
    candidate_bundle["llm_output"] = llm_output
    selected_candidate = {
        "index": 0,
        "action_strategy": candidate_bundle.get("action_strategy", action_strategy),
        "proposal_source": candidate_bundle.get("proposal_source", "unknown"),
        "payload": candidate_bundle.get("payload", {}),
        "llm_output": llm_output,
    }
    candidate_action_list = list(candidate_actions or [action_strategy])
    decision_bundle = {
        "schema": schema,
        "selection_path": selection_path,
        "candidate_actions": candidate_action_list,
        "raw_candidate_actions": list(raw_candidate_actions or candidate_action_list),
        "excluded_actions": list(excluded_actions or []),
        "selected_candidate_index": 0,
        "selected_action_strategy": selected_candidate["action_strategy"],
        "selection_reason": selection_reason,
        "trajectory_guidance": copy.deepcopy(trajectory_guidance) if isinstance(trajectory_guidance, dict) else None,
        "candidates": [selected_candidate],
    }
    return (
        str(selected_candidate["action_strategy"]),
        selected_candidate,
        llm_output,
        proposal_obj,
        candidate_action_list,
        decision_bundle,
    )


def _parse_two_materials_from_combined_salt(value: Any) -> Tuple[Optional[str], Optional[str]]:
    raw = str(value or "").strip()
    if not raw:
        return None, None
    parts = [p.strip() for p in re.split(r"\+|/", raw) if p.strip()]
    if len(parts) >= 2:
        return parts[0], " + ".join(parts[1:])
    if len(parts) == 1:
        return parts[0], parts[0]
    return None, None


def _normalize_component_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").strip().lower())


def _param_dict(params: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = params.get(key)
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        # list form, e.g. resulting_solvents=[{name, fraction}] -> {name: fraction}
        return {name: val for name, val in _extract_named_values(value) if val not in (None, "")}
    return {}


def _extract_text_variants(value: Any) -> List[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    variants: List[str] = []

    def _add(item: Any) -> None:
        text = str(item or "").strip()
        if text and text not in variants:
            variants.append(text)

    _add(raw)

    paren_bits = re.findall(r"\(([^)]+)\)", raw)
    for bit in paren_bits:
        _add(bit)

    no_paren = re.sub(r"\([^)]*\)", "", raw).strip()
    _add(no_paren)

    return variants


def _family_matches_for_name(name: Any, family_index: Dict[str, List[str]]) -> List[str]:
    matches: List[str] = []
    seen: Set[str] = set()
    for variant in _extract_text_variants(name):
        key = _normalize_component_name(variant)
        for family in family_index.get(key, []):
            if family not in seen:
                seen.add(family)
                matches.append(str(family))
    return matches


def _family_dimension_for_action(action_type: str) -> str:
    action = normalize_action_type(action_type)
    if action in {"salt_replace", "add_new_salt", "salt_concentration_shift"}:
        return "salts"
    if action in {"solvent_replace", "add_new_solvent", "remove_solvent", "solvent_concentration_shift"}:
        return "solvents"
    if action == "additive_change":
        return "additives"
    return ""


def build_allowed_target_families(
    action_type: str,
    family_tables: Dict[str, Dict[str, List[str]]],
    avoid_target_families: List[str],
    force_full_allowlist: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    avoid_set = {str(item).strip() for item in (avoid_target_families or []) if str(item).strip()}
    if not avoid_set and not force_full_allowlist:
        return None
    dimension = _family_dimension_for_action(action_type)
    if not dimension:
        return None
    family_table = family_tables.get(dimension, {}) or {}
    if not isinstance(family_table, dict):
        return None
    allowed: List[Dict[str, Any]] = []
    for family_name in sorted(str(name).strip() for name in family_table.keys()):
        if family_name and family_name not in avoid_set:
            allowed.append({"family": family_name})
    return allowed


def _extract_target_component_names(action_type: str, params: Dict[str, Any]) -> List[str]:
    names: List[str] = []

    def _add(value: Any) -> None:
        item = str(value or "").strip()
        if item and item not in names:
            names.append(item)

    if action_type == "salt_replace":
        replace = _param_dict(params, "replace")
        _add(params.get("to_salt") or params.get("salt_to") or replace.get("to_salt") or replace.get("salt_to"))
        if not names:
            _, to_salt = _parse_two_materials_from_combined_salt(params.get("salt"))
            _add(to_salt)
    elif action_type == "add_new_salt":
        _add(params.get("to_salt") or params.get("salt_to") or params.get("added_salt"))
        if not names:
            _, added = _parse_two_materials_from_combined_salt(params.get("salt"))
            _add(added)
    elif action_type in {"solvent_replace", "add_new_solvent"}:
        replace = _param_dict(params, "replace")
        _add(params.get("to_solvent") or params.get("solvent") or replace.get("to_solvent") or replace.get("solvent"))
        replacements = _extract_change_pairs(params.get("replacements"))
        if replacements:
            for _, target in replacements:
                _add(target)
        added = _extract_named_values(params.get("added_solvents"))
        for name, _ in added:
            _add(name)
        resulting_solvents = _param_dict(params, "resulting_solvents")
        for name in resulting_solvents.keys():
            _add(name)
    elif action_type == "additive_change":
        change_kind = str(params.get("change") or params.get("change_type") or "").strip().lower()
        if change_kind == "replace":
            replace = _param_dict(params, "replace")
            _add(params.get("replacement_additive") or params.get("to_additive") or replace.get("to_additive"))
        else:
            _add(params.get("additive") or params.get("additive_name") or params.get("to_additive"))

    return names


def infer_failed_target_families(
    proposal_obj: Dict[str, Any],
    family_indexes: Dict[str, Dict[str, List[str]]],
) -> List[str]:
    action_type = normalize_action_type(str(proposal_obj.get("action_type", "")).strip())
    params = proposal_obj.get("parameters", {}) or {}
    if not isinstance(params, dict):
        params = {}
    dimension = _family_dimension_for_action(action_type)
    if not dimension or action_type in {"salt_concentration_shift", "solvent_concentration_shift", "remove_solvent"}:
        return []

    names = _extract_target_component_names(action_type, params)
    families: List[str] = []
    seen: Set[str] = set()
    family_index = family_indexes.get(dimension, {})
    for name in names:
        for family in _family_matches_for_name(name, family_index):
            if family not in seen:
                seen.add(family)
                families.append(str(family))
    return families


def proposal_violates_avoid_target_families(
    proposal_obj: Dict[str, Any],
    family_indexes: Dict[str, Dict[str, List[str]]],
    avoid_target_families: List[str],
) -> List[str]:
    avoid_set = {str(item).strip() for item in (avoid_target_families or []) if str(item).strip()}
    if not avoid_set or not proposal_obj:
        return []
    target_families = infer_failed_target_families(proposal_obj, family_indexes)
    return [family for family in target_families if family in avoid_set]


def proposal_unknown_target_components(
    proposal_obj: Dict[str, Any],
    family_indexes: Dict[str, Dict[str, List[str]]],
) -> List[str]:
    if not proposal_obj:
        return []
    action_type = normalize_action_type(str(proposal_obj.get("action_type", "")).strip())
    params = proposal_obj.get("parameters", {}) or {}
    if not isinstance(params, dict):
        params = {}
    dimension = _family_dimension_for_action(action_type)
    if not dimension or action_type in {"salt_concentration_shift", "solvent_concentration_shift", "remove_solvent"}:
        return []

    names = _extract_target_component_names(action_type, params)
    family_index = family_indexes.get(dimension, {})
    unknown: List[str] = []
    seen_keys: Set[str] = set()
    for name in names:
        normalized_variants = {_normalize_component_name(variant) for variant in _extract_text_variants(name)}
        normalized_variants.discard("")
        if any(variant in family_index for variant in normalized_variants):
            continue
        normalized_name = _normalize_component_name(name)
        if normalized_name and normalized_name not in seen_keys:
            seen_keys.add(normalized_name)
            unknown.append(str(name))
    return unknown


def sanitize_proposal_against_family_constraints(
    proposal_obj: Dict[str, Any],
    llm_output: Dict[str, Any],
    family_indexes: Dict[str, Dict[str, List[str]]],
    avoid_target_families: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    violating_families = proposal_violates_avoid_target_families(
        proposal_obj,
        family_indexes,
        avoid_target_families,
    ) if proposal_obj else []
    if not violating_families:
        return proposal_obj, llm_output

    sanitized = dict(llm_output or {})
    sanitized["proposal_error"] = (
        "invalid proposal: proposal introduces target(s) from forbidden families: "
        + ", ".join(violating_families)
    )
    sanitized["proposal"] = None
    return {}, sanitized


def sanitize_proposal_against_molecule_scope(
    proposal_obj: Dict[str, Any],
    llm_output: Dict[str, Any],
    family_indexes: Dict[str, Dict[str, List[str]]],
    allow_new_molecules: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if allow_new_molecules:
        return proposal_obj, llm_output
    unknown_targets = proposal_unknown_target_components(proposal_obj, family_indexes) if proposal_obj else []
    if not unknown_targets:
        return proposal_obj, llm_output

    sanitized = dict(llm_output or {})
    sanitized["proposal_error"] = (
        "invalid proposal: proposal introduces target molecule(s) outside the current family tables: "
        + ", ".join(unknown_targets)
    )
    sanitized["proposal"] = None
    return {}, sanitized


def _extract_change_pairs(items: Any) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not isinstance(items, list):
        return pairs
    def _name_of(v: Any) -> str:
        # from/to may be a plain name, or an object like {"name":..., "smiles":..., "fraction":...}
        if isinstance(v, dict):
            return str(v.get("name") or v.get("solvent") or v.get("component") or "").strip()
        return str(v or "").strip()

    for item in items:
        if not isinstance(item, dict):
            continue
        source = _name_of(item.get("from") or item.get("source") or item.get("old"))
        target = _name_of(item.get("to") or item.get("target") or item.get("new"))
        if source or target:
            pairs.append((source, target))
    return pairs


def _extract_named_values(items: Any) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, str) and item.strip():
            out.append((item.strip(), None))
            continue
        if not isinstance(item, dict):
            continue
        name = str(item.get("short_name") or item.get("name") or item.get("component") or "").strip()
        if not name:
            continue
        value = next((item.get(k) for k in ("value", "fraction", "ratio", "amount", "concentration", "to")
                      if item.get(k) not in (None, "")), None)
        if value is None:   # e.g. amount_wt_percent=0.3
            unit_key = next((k for k in item if str(k).startswith("amount_") and item.get(k) not in (None, "")), None)
            if unit_key:
                unit = {"amount_wt_percent": "wt%", "amount_vol_percent": "vol%"}.get(unit_key, "")
                value = f"{item[unit_key]} {unit}".strip()
        out.append((name, value))
    return out


def _is_inactive_component_value(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    normalized = text.lower()
    if normalized in {"0", "0.0", "0.00", "none", "absent", "false", "n/a", "na"}:
        return True
    if re.fullmatch(r"0+(?:\.0+)?(?:\s*(?:%|wt%|vol%|m))?", normalized):
        return True
    return False


def _active_mapping_items(mapping: Any) -> Dict[str, Any]:
    active: Dict[str, Any] = {}
    if not isinstance(mapping, dict):
        return active
    for key, value in mapping.items():
        name = str(key).strip()
        if not name or _is_inactive_component_value(value):
            continue
        active[name] = value
    return active


def _active_mapping_names(mapping: Any) -> List[str]:
    return list(_active_mapping_items(mapping).keys())


def active_solvent_names(formulation: Dict[str, Any]) -> List[str]:
    comp = (formulation or {}).get("composition", {}) or {}
    return _active_mapping_names(comp.get("solvents", {}) or {})


def _canonical_name_from_pool(name: Any, pool: List[str]) -> str:
    needle = _normalize_component_name(name)
    if not needle:
        return ""
    for candidate in pool:
        if _normalize_component_name(candidate) == needle:
            return str(candidate)
    return str(name).strip()


_SOLVENT_ALIAS_MAP: Optional[Dict[str, str]] = None


def _load_solvent_alias_map() -> Dict[str, str]:
    """normalized(alias-or-symbol) -> canonical short symbol, from component_aliases.json."""
    global _SOLVENT_ALIAS_MAP
    if _SOLVENT_ALIAS_MAP is None:
        mapping: Dict[str, str] = {}
        try:
            raw = json.loads(
                (Path(__file__).resolve().parent / "component_aliases.json").read_text(encoding="utf-8")
            )
            for symbol, aliases in raw.items():
                mapping[_normalize_component_name(symbol)] = symbol
                for alias in aliases or []:
                    mapping.setdefault(_normalize_component_name(alias), symbol)
        except Exception:
            mapping = {}
        _SOLVENT_ALIAS_MAP = mapping
    return _SOLVENT_ALIAS_MAP


def _trailing_parenthetical_symbol(text: str) -> str:
    """Acronym in the final parenthesised group of a name, nesting-aware:
    '4-(trifluoromethyl)-1,3-dioxolan-2-one (TFPC, a fluorinated cyclic carbonate)' -> 'TFPC',
    'propylene carbonate (PC)' -> 'PC'. Inner groups such as '(trifluoromethyl)' or
    'bis(fluorosulfonyl)' are part of the chemical name, not a symbol."""
    text = text.strip()
    if not text.endswith(")"):
        return ""
    depth = 0
    for i in range(len(text) - 1, -1, -1):
        depth += {")": 1, "(": -1}.get(text[i], 0)
        if depth == 0:
            break
    if i <= 0:
        return ""
    # the acronym is the first space-free item: "(TFPC, ...)" or "(trifluoromethyl propylene carbonate, TFPC)"
    items = [x.strip() for x in re.split(r"[,;]", text[i + 1:-1])]
    return next((x for x in items if x and " " not in x), "")


def _canonical_solvent_symbol(name: Any) -> str:
    """Best-effort short symbol for a solvent name like 'propylene carbonate (PC)' -> 'PC'.
    Different LLMs emit either the short symbol or the full descriptive name; normalize both."""
    text = str(name or "").strip()
    if not text:
        return ""
    token = _trailing_parenthetical_symbol(text)          # "full name (TFPC, ...)" -> "TFPC"
    if token:
        return token
    amap = _load_solvent_alias_map()
    stripped = re.sub(r"\([^)]*\)", "", text).strip()
    symbol = amap.get(_normalize_component_name(stripped)) or amap.get(_normalize_component_name(text))
    return symbol or text


def _resolve_solvent_key(name: Any, pool: List[str]) -> Optional[str]:
    """Match a proposal solvent name to an existing key in `pool` (the state's solvents),
    tolerating short-symbol vs full-descriptive-name mismatches across LLM outputs."""
    needle = _normalize_component_name(name)
    if not needle:
        return None
    for key in pool:
        if _normalize_component_name(key) == needle:
            return key
    sym = _normalize_component_name(_canonical_solvent_symbol(name))
    if sym:
        for key in pool:
            if _normalize_component_name(key) == sym or _normalize_component_name(_canonical_solvent_symbol(key)) == sym:
                return key
    return None


def _append_unique_name(names: List[str], value: Any, pool: Optional[List[str]] = None) -> None:
    text = str(value or "").strip()
    if not text:
        return
    canonical = _canonical_name_from_pool(text, pool or []) if pool else text
    key = _normalize_component_name(canonical)
    if not key:
        return
    if any(_normalize_component_name(existing) == key for existing in names):
        return
    names.append(canonical)


def infer_solvent_replace_sources(current_formulation: Dict[str, Any], proposal_obj: Dict[str, Any]) -> List[str]:
    action_type = normalize_action_type(str(proposal_obj.get("action_type", "")).strip())
    if action_type != "solvent_replace":
        return []
    params = proposal_obj.get("parameters", {}) or {}
    if not isinstance(params, dict):
        params = {}

    current_solvents = active_solvent_names(current_formulation)
    sources: List[str] = []
    replace = _param_dict(params, "replace")
    _append_unique_name(params.get("from_solvent") or replace.get("from_solvent"), sources, current_solvents)

    replacements = _extract_change_pairs(params.get("replacements"))
    for source, _ in replacements:
        _append_unique_name(source, sources, current_solvents)

    removed_solvents = _extract_named_values(params.get("removed_solvents"))
    for name, _ in removed_solvents:
        _append_unique_name(name, sources, current_solvents)

    resulting_solvents = _active_mapping_items(_param_dict(params, "resulting_solvents"))
    if resulting_solvents:
        resulting_keys = {_normalize_component_name(name) for name in resulting_solvents.keys()}
        for name in current_solvents:
            if _normalize_component_name(name) not in resulting_keys:
                _append_unique_name(name, sources, current_solvents)
    return sources


def infer_solvent_replace_targets(
    current_formulation: Dict[str, Any],
    proposal_obj: Dict[str, Any],
    next_formulation: Optional[Dict[str, Any]] = None,
) -> List[str]:
    action_type = normalize_action_type(str(proposal_obj.get("action_type", "")).strip())
    if action_type != "solvent_replace":
        return []
    params = proposal_obj.get("parameters", {}) or {}
    if not isinstance(params, dict):
        params = {}

    current_solvents = active_solvent_names(current_formulation)
    targets: List[str] = []
    replace = _param_dict(params, "replace")
    _append_unique_name(params.get("to_solvent") or replace.get("to_solvent"), targets)

    replacements = _extract_change_pairs(params.get("replacements"))
    for _, target in replacements:
        _append_unique_name(target, targets)

    added_solvents = _extract_named_values(params.get("added_solvents"))
    for name, value in added_solvents:
        if not _is_inactive_component_value(value):
            _append_unique_name(name, targets)

    resulting_solvents = _active_mapping_items(_param_dict(params, "resulting_solvents"))
    if resulting_solvents:
        current_keys = {_normalize_component_name(name) for name in current_solvents}
        for name in resulting_solvents.keys():
            if _normalize_component_name(name) not in current_keys:
                _append_unique_name(name, targets)

    if next_formulation is not None:
        next_solvents = active_solvent_names(next_formulation)
        current_keys = {_normalize_component_name(name) for name in current_solvents}
        for name in next_solvents:
            if _normalize_component_name(name) not in current_keys:
                _append_unique_name(name, targets, next_solvents)
    return targets


def infer_solvent_replace_step(
    current_formulation: Dict[str, Any],
    proposal_obj: Dict[str, Any],
    next_formulation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sources = infer_solvent_replace_sources(current_formulation, proposal_obj)
    targets = infer_solvent_replace_targets(current_formulation, proposal_obj, next_formulation=next_formulation)
    return {
        "sources": sources,
        "targets": targets,
        "summary": "; ".join(
            f"{src} -> {targets[idx] if idx < len(targets) else 'UNKNOWN'}"
            for idx, src in enumerate(sources)
        ) if sources else "",
    }


def _trajectory_name_list(state: Optional[Dict[str, Any]], key: str) -> List[str]:
    values = (state or {}).get(key, []) or []
    out: List[str] = []
    for value in values:
        _append_unique_name(out, value)
    return out


def solvent_trajectory_remaining_sources(state: Optional[Dict[str, Any]]) -> List[str]:
    source_solvents = _trajectory_name_list(state, "source_solvents")
    completed = {_normalize_component_name(name) for name in _trajectory_name_list(state, "completed_source_solvents")}
    exhausted = {_normalize_component_name(name) for name in _trajectory_name_list(state, "exhausted_source_solvents")}
    return [
        name
        for name in source_solvents
        if _normalize_component_name(name) not in completed and _normalize_component_name(name) not in exhausted
    ]


def solvent_trajectory_has_pending_sources(state: Optional[Dict[str, Any]]) -> bool:
    return bool(solvent_trajectory_remaining_sources(state))


def select_solvent_trajectory_source(state: Dict[str, Any], retry_threshold: int) -> str:
    remaining = solvent_trajectory_remaining_sources(state)
    if not remaining:
        return ""
    current_source = _canonical_name_from_pool(state.get("current_source"), remaining)
    retry_counts = state.get("source_retry_counts", {}) or {}
    if current_source:
        if 0 < int(retry_counts.get(current_source, 0) or 0) < int(retry_threshold):
            return current_source
    return remaining[0]


def build_positive_feedback_trajectory_guidance(state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not solvent_trajectory_has_pending_sources(state):
        return None
    remaining = solvent_trajectory_remaining_sources(state)
    completed = _trajectory_name_list(state, "completed_source_solvents")
    exhausted = _trajectory_name_list(state, "exhausted_source_solvents")
    accepted_steps = list((state or {}).get("accepted_steps", []) or [])
    preferred_source = _canonical_name_from_pool((state or {}).get("current_source"), remaining) or (remaining[0] if remaining else "")
    return {
        "mode": "continue_solvent_replace_trajectory",
        "selection_rule": (
            "Continue the accepted solvent-replacement trajectory. "
            "Replace exactly one remaining source solvent, and do not revisit completed or exhausted source solvents."
        ),
        "source_solvents": _trajectory_name_list(state, "source_solvents"),
        "completed_source_solvents": completed,
        "exhausted_source_solvents": exhausted,
        "remaining_source_solvents": remaining,
        "preferred_next_source_solvent": preferred_source,
        "accepted_steps": accepted_steps[-3:],
        "source_retry_counts": copy.deepcopy((state or {}).get("source_retry_counts", {}) or {}),
    }


# Escalation "sibling" actions are reachable from one another when escalating within an
# accepted direction. For solvents the ladder covers both introducing a bolder component
# (replace / add) AND tuning the ratio (solvent_concentration_shift): once the solvent
# identities are where they should be, escalation can sister-jump to shifting their ratio.
# This is the ONLY way a concentration_shift can surface, since the MLP prior ranks it
# dead-last and it never enters the top-k. Subtractive remove_solvent stays out.
ESCALATION_SIBLINGS: Dict[str, List[str]] = {
    "solvent_replace": ["solvent_replace", "solvent_concentration_shift"],
    "solvent_concentration_shift": ["solvent_replace", "solvent_concentration_shift"],
    "add_new_solvent": ["solvent_replace", "solvent_concentration_shift"],
    "salt_replace": ["salt_replace", "add_new_salt"],
    "add_new_salt": ["salt_replace", "add_new_salt"],
}


def escalation_sibling_actions(action: str) -> List[str]:
    """Sibling actions for escalation. Defaults to just the action itself (single-action
    escalation) for actions without an introduce-a-component sibling."""
    act = str(action or "").strip()
    return list(ESCALATION_SIBLINGS.get(act, [act] if act else []))


def describe_proposed_change(current_formulation: Dict[str, Any], proposal_obj: Dict[str, Any]) -> str:
    """One-line description of a proposed change for the generator prompt: the change summary
    plus the formulation it would produce (the summary alone can read 'UNKNOWN' when the
    proposal uses parameter keys the summary does not parse)."""
    if not proposal_obj:
        return ""
    desc = summarize_proposal_change(current_formulation, proposal_obj)
    result, applied, _ = apply_proposal_to_formulation(copy.deepcopy(current_formulation), proposal_obj)
    if applied:
        desc = f"{desc} -> resulting formulation: {composition_summary(result)}"
    return desc


def build_retry_guidance(rejected_changes: List[str]) -> Optional[Dict[str, Any]]:
    """Prompt directive for a retry in the same direction after non-positive feedback. Lists
    the changes already rejected from the CURRENT formulation; the list is reset whenever a
    change is accepted, so rejections from earlier formulations are never shown."""
    if not rejected_changes:
        return None
    return {
        "mode": "retry_same_direction",
        "directive": (
            "The changes in do_not_repeat_rejected_changes were already proposed from the current "
            "formulation and REJECTED by the evaluator. Do NOT propose any of them again, including "
            "the same molecule at the same or a nearly identical amount, or the same change worded "
            "differently. Propose a genuinely different change within the same action type."
        ),
        "do_not_repeat_rejected_changes": list(rejected_changes)[-6:],
    }


def build_escalation_guidance(
    escalation_state: Optional[Dict[str, Any]],
    current_formulation: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Prompt directive injected after a positive reward, asking for a LESS COMMON,
    MORE AGGRESSIVE next step in the same intent — building on (not repeating) what was
    already accepted. May hop across sibling actions (e.g. replace -> add a new solvent)."""
    if not isinstance(escalation_state, dict) or not escalation_state.get("sibling_actions"):
        return None
    siblings = list(escalation_state.get("sibling_actions", []) or [])
    level = int(escalation_state.get("level", 1))
    accepted_changes = list(escalation_state.get("accepted_changes", []) or [])
    accepted_targets = list(escalation_state.get("accepted_targets", []) or [])
    rejected_changes = list(escalation_state.get("rejected_changes", []) or [])
    hop = " or ".join(siblings) if len(siblings) > 1 else (siblings[0] if siblings else "the same action")
    reject_note = (
        " Some escalation attempts here were just REJECTED — do NOT propose any of the "
        "changes in do_not_repeat_rejected_changes again; pick a genuinely different next step."
        if rejected_changes else ""
    )
    return {
        "mode": "escalate_within_direction",
        "sibling_actions": siblings,
        "escalation_level": level,
        "directive": (
            "The changes in accepted_changes_in_order were ACCEPTED (positive feedback), listed "
            "oldest-to-newest. Read them as a TREND: infer what each accepted step pushed (which "
            "component was changed and in which direction) and propose the next single one-step "
            "change that CONTINUES THAT SAME TREND one notch further. You MAY keep modifying a "
            "component you already changed to push the trend further on it (e.g. take a component "
            "you just made more X and make it even more X); that is continuing the trend, not "
            "repeating. Do NOT reverse earlier progress and do NOT re-propose an identical change "
            "already made or a change already rejected. Keep it a sensible next step along the "
            "trend (not a random or exotic jump). The user's feedback will tell you whether to keep "
            "going or, after repeated rejections, to change direction. No specific chemistry is "
            "prescribed here — infer the trend from the accepted sequence yourself."
            + reject_note
        ),
        "selection_rule": "Read accepted_changes_in_order as a trend and propose the next step that continues it (further modifying an already-changed component to push the trend is allowed); do not reverse progress or re-propose an identical/rejected change.",
        "accepted_changes_in_order": accepted_changes[-8:],
        "do_not_repeat_rejected_changes": rejected_changes[-6:],
    }


def advance_escalation_state(
    escalation_state: Optional[Dict[str, Any]],
    selected_action: str,
    state_before: Dict[str, Any],
    proposal_obj: Dict[str, Any],
) -> Dict[str, Any]:
    """Create or advance the escalation ladder after a positive. The ladder continues
    while the accepted action stays within the same sibling group (intent)."""
    action_type = normalize_action_type(str(proposal_obj.get("action_type", "")).strip())
    params = proposal_obj.get("parameters", {}) or {}
    if not isinstance(params, dict):
        params = {}
    accepted_change = summarize_proposal_change(state_before, proposal_obj)
    accepted_targets = _extract_target_component_names(action_type, params)
    siblings = escalation_sibling_actions(selected_action)

    if isinstance(escalation_state, dict) and selected_action in (escalation_state.get("sibling_actions") or []):
        state = copy.deepcopy(escalation_state)
        state["level"] = int(state.get("level", 1)) + 1
        # A step was just accepted: keep exploring THIS sibling (reset it as current), give it a
        # fresh retry budget, and drop the (now stale) rejected-change / tried-sibling memory so
        # a later exhaustion re-evaluates all siblings from the newly accepted point.
        state["reject_count"] = 0
        state["rejected_changes"] = []
        state["rejected_signatures"] = []
        state["current_sibling"] = selected_action
        state["tried_siblings"] = []
    else:
        state = {"sibling_actions": siblings, "level": 1, "accepted_changes": [], "accepted_targets": []}
    state["last_action"] = selected_action
    changes = list(state.get("accepted_changes", []) or [])
    if accepted_change:
        changes.append(accepted_change)
    state["accepted_changes"] = changes[-10:]
    targets = list(state.get("accepted_targets", []) or [])
    for name in accepted_targets:
        if name and name not in targets:
            targets.append(name)
    state["accepted_targets"] = targets
    return state


def invalid_solvent_trajectory_proposal_reason(
    current_formulation: Dict[str, Any],
    proposal_obj: Dict[str, Any],
    trajectory_guidance: Optional[Dict[str, Any]],
) -> str:
    if not isinstance(trajectory_guidance, dict):
        return ""
    if trajectory_guidance.get("mode") != "continue_solvent_replace_trajectory":
        return ""
    action_type = normalize_action_type(str(proposal_obj.get("action_type", "")).strip())
    if action_type != "solvent_replace":
        return "trajectory continuation requires solvent_replace"

    remaining = list(trajectory_guidance.get("remaining_source_solvents", []) or [])
    if not remaining:
        return ""
    remaining_keys = {_normalize_component_name(name) for name in remaining}
    completed_keys = {_normalize_component_name(name) for name in (trajectory_guidance.get("completed_source_solvents", []) or [])}
    exhausted_keys = {_normalize_component_name(name) for name in (trajectory_guidance.get("exhausted_source_solvents", []) or [])}
    preferred_key = _normalize_component_name(trajectory_guidance.get("preferred_next_source_solvent"))

    sources = infer_solvent_replace_sources(current_formulation, proposal_obj)
    if not sources:
        return "trajectory continuation requires a concrete source solvent from the remaining solvent set"
    source_keys = {_normalize_component_name(name) for name in sources}
    if source_keys & completed_keys:
        return "trajectory continuation cannot modify a source solvent that already succeeded earlier in the session"
    if source_keys & exhausted_keys:
        return "trajectory continuation cannot modify a source solvent that already exhausted its retry budget"
    if not source_keys <= remaining_keys:
        return "trajectory continuation must choose the source solvent from the remaining solvent set"
    if preferred_key and preferred_key not in source_keys:
        return (
            "trajectory continuation should focus on the current remaining source solvent: "
            + str(trajectory_guidance.get("preferred_next_source_solvent"))
        )
    return ""


def update_solvent_trajectory_after_positive(
    state: Optional[Dict[str, Any]],
    current_formulation: Dict[str, Any],
    proposal_obj: Dict[str, Any],
    next_formulation: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if normalize_action_type(str(proposal_obj.get("action_type", "")).strip()) != "solvent_replace":
        return None
    source_solvents_before = active_solvent_names(current_formulation)
    if state is None:
        if len(source_solvents_before) < 2:
            return None
        trajectory_state: Dict[str, Any] = {
            "mode": "continue_solvent_replace_trajectory",
            "source_solvents": list(source_solvents_before),
            "completed_source_solvents": [],
            "exhausted_source_solvents": [],
            "accepted_steps": [],
            "source_retry_counts": {},
            "source_avoid_target_families": {},
            "source_rejected_proposal_signatures": {},
            "current_source": "",
        }
    else:
        trajectory_state = copy.deepcopy(state)

    step = infer_solvent_replace_step(current_formulation, proposal_obj, next_formulation=next_formulation)
    sources = step.get("sources", []) or []
    targets = step.get("targets", []) or []
    if not sources:
        current_source = str(trajectory_state.get("current_source", "")).strip()
        if current_source:
            sources = [current_source]

    source_pool = _trajectory_name_list(trajectory_state, "source_solvents")
    completed = _trajectory_name_list(trajectory_state, "completed_source_solvents")
    exhausted = _trajectory_name_list(trajectory_state, "exhausted_source_solvents")
    for source in sources:
        canonical = _canonical_name_from_pool(source, source_pool)
        _append_unique_name(completed, canonical, source_pool)
        exhausted = [name for name in exhausted if _normalize_component_name(name) != _normalize_component_name(canonical)]

    trajectory_state["completed_source_solvents"] = completed
    trajectory_state["exhausted_source_solvents"] = exhausted

    accepted_steps = list(trajectory_state.get("accepted_steps", []) or [])
    accepted_steps.append(
        {
            "from_solvents": sources,
            "to_solvents": targets,
            "summary": step.get("summary") or summarize_proposal_change(current_formulation, proposal_obj),
            "proposal_signature": proposal_signature(proposal_obj),
        }
    )
    trajectory_state["accepted_steps"] = accepted_steps

    retry_counts = dict(trajectory_state.get("source_retry_counts", {}) or {})
    avoid_map = dict(trajectory_state.get("source_avoid_target_families", {}) or {})
    rejected_map = dict(trajectory_state.get("source_rejected_proposal_signatures", {}) or {})
    for source in sources:
        canonical = _canonical_name_from_pool(source, source_pool)
        retry_counts[canonical] = 0
        avoid_map[canonical] = []
        rejected_map[canonical] = []
    trajectory_state["source_retry_counts"] = retry_counts
    trajectory_state["source_avoid_target_families"] = avoid_map
    trajectory_state["source_rejected_proposal_signatures"] = rejected_map
    trajectory_state["current_source"] = ""

    if not solvent_trajectory_has_pending_sources(trajectory_state):
        return None
    return trajectory_state


def mark_solvent_trajectory_source_exhausted(
    state: Optional[Dict[str, Any]],
    source_name: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(state, dict):
        return None
    trajectory_state = copy.deepcopy(state)
    source_pool = _trajectory_name_list(trajectory_state, "source_solvents")
    canonical = _canonical_name_from_pool(source_name, source_pool)
    exhausted = _trajectory_name_list(trajectory_state, "exhausted_source_solvents")
    _append_unique_name(exhausted, canonical, source_pool)
    trajectory_state["exhausted_source_solvents"] = exhausted
    trajectory_state["current_source"] = ""
    if not solvent_trajectory_has_pending_sources(trajectory_state):
        return None
    return trajectory_state


def _invalid_proposal_reason(current_formulation: Dict[str, Any], proposal_obj: Dict[str, Any]) -> str:
    action_type = str(proposal_obj.get("action_type", "")).strip()
    params = proposal_obj.get("parameters", {}) or {}
    if not isinstance(params, dict):
        params = {}

    if action_type != "solvent_replace":
        return ""

    replace = _param_dict(params, "replace")
    replacements = _extract_change_pairs(params.get("replacements"))
    if any(target for _, target in replacements):
        return ""

    from_solvent = str(params.get("from_solvent") or replace.get("from_solvent") or "").strip()
    to_solvent = str(params.get("to_solvent") or replace.get("to_solvent") or "").strip()
    if from_solvent or to_solvent:
        return "" if to_solvent else "missing target solvent"

    removed_solvents = _extract_named_values(params.get("removed_solvents"))
    added_solvents = _extract_named_values(params.get("added_solvents"))
    if removed_solvents or added_solvents:
        has_added_target = any(name and not _is_inactive_component_value(value) for name, value in added_solvents)
        return "" if has_added_target else "missing added solvent in replacement proposal"

    resulting_solvents = _active_mapping_items(_param_dict(params, "resulting_solvents"))
    if resulting_solvents:
        current_solvents = _active_mapping_items(((current_formulation or {}).get("composition", {}) or {}).get("solvents", {}))
        return "" if resulting_solvents != current_solvents else "resulting_solvents does not change formulation"

    return "solvent_replace proposal contains no usable replacement fields"


def _list_preview(items: List[str], limit: int = 3) -> str:
    if not items:
        return ""
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f", ... (+{len(items) - limit} more)"


def _compact_json_preview(value: Any, limit: int = 160) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _ratio_value(value: Any) -> Optional[float]:
    m = re.search(r"[-+]?\d*\.?\d+", str(value or ""))
    return float(m.group()) if m else None


def _normalized_solvent_fractions(solvents: Dict[str, Any]) -> Dict[str, Any]:
    """Solvent fractions normalized to 1 (e.g. EC:DMC '1':'2' -> 0.33/0.67); raw values kept
    when they cannot be parsed."""
    active = _active_mapping_items(solvents or {})
    vals = {k: _ratio_value(v) for k, v in active.items()}
    total = sum(v for v in vals.values() if v is not None)
    if total > 0 and all(v is not None for v in vals.values()):
        return {k: v / total for k, v in vals.items()}
    return dict(active)


def _fmt_fraction(value: Any) -> str:
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def formulation_change_summary(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    """What actually changes between two formulations: salt, salt concentration, solvents
    (replaced, added, removed, re-proportioned) and additives."""
    b = (before or {}).get("composition", {}) or {}
    a = (after or {}).get("composition", {}) or {}
    parts: List[str] = []
    if str(b.get("salt", "")).strip() != str(a.get("salt", "")).strip():
        parts.append(f"salt {b.get('salt') or 'none'} -> {a.get('salt') or 'none'}")
    if str(b.get("salt_concentration", "")).strip() != str(a.get("salt_concentration", "")).strip():
        parts.append(f"salt concentration {b.get('salt_concentration') or 'n/a'} -> {a.get('salt_concentration') or 'n/a'}")
    fb, fa = _normalized_solvent_fractions(b.get("solvents") or {}), _normalized_solvent_fractions(a.get("solvents") or {})
    removed = [k for k in fb if _resolve_solvent_key(k, list(fa)) is None]
    added = [k for k in fa if _resolve_solvent_key(k, list(fb)) is None]
    if removed or added:
        parts.append("solvents " + ", ".join(removed or ["none"]) + " -> " + ", ".join(added or ["none"]))
    ratio_text = []
    for k in fa:
        kb = _resolve_solvent_key(k, list(fb))
        old_v = fb.get(kb) if kb is not None else None
        new_v = fa[k]
        if kb is None:
            ratio_text.append(f"{k} {_fmt_fraction(new_v)}")
        elif (isinstance(old_v, float) and isinstance(new_v, float) and abs(old_v - new_v) > 1e-6) or (
                not isinstance(old_v, float) and str(old_v) != str(new_v)):
            ratio_text.append(f"{k} {_fmt_fraction(old_v)} -> {_fmt_fraction(new_v)}")
    if removed or added:
        parts.append("resulting solvent fractions: " + ", ".join(f"{k} {_fmt_fraction(v)}" for k, v in fa.items()))
    elif ratio_text:
        parts.append("solvent fractions " + ", ".join(ratio_text))
    ab, aa = _active_mapping_items(b.get("additives") or {}), _active_mapping_items(a.get("additives") or {})
    for k, v in aa.items():
        if k not in ab:
            parts.append(f"add additive {k} {v}")
        elif str(ab[k]) != str(v):
            parts.append(f"additive {k} {ab[k]} -> {v}")
    for k in ab:
        if k not in aa:
            parts.append(f"remove additive {k}")
    return "; ".join(parts) if parts else "no change to the formulation"


def summarize_proposal_change(current_formulation: Dict[str, Any], proposal_obj: Dict[str, Any]) -> str:
    # Describe the change by what it does to the formulation whenever the proposal can be
    # applied; the parameter-based description below is only the fallback.
    if proposal_obj:
        after, applied, _ = apply_proposal_to_formulation(copy.deepcopy(current_formulation), proposal_obj)
        if applied:
            label = str(proposal_obj.get("action_type", "")).strip().replace("_", " ")
            return f"{label}: {formulation_change_summary(current_formulation, after)}"
    comp = (current_formulation or {}).get("composition", {}) or {}
    action_type = str(proposal_obj.get("action_type", "")).strip()
    params = proposal_obj.get("parameters", {}) or {}
    if not isinstance(params, dict):
        params = {}

    if action_type == "salt_replace":
        to_salt = params.get("to_salt") or params.get("salt_to")
        if not to_salt:
            _, to_salt = _parse_two_materials_from_combined_salt(params.get("salt"))
        old_salt = str(comp.get("salt", "")).strip() or "current salt"
        return f"Replace salt: {old_salt} -> {to_salt or 'UNKNOWN'}"

    if action_type == "add_new_salt":
        added_salt = params.get("to_salt") or params.get("salt_to") or params.get("added_salt")
        if not added_salt:
            _, added_salt = _parse_two_materials_from_combined_salt(params.get("salt"))
        return f"Add new salt: {added_salt or 'UNKNOWN'}"

    if action_type == "salt_concentration_shift":
        old_conc = str(comp.get("salt_concentration", "")).strip() or "current concentration"
        new_conc = params.get("to_concentration") or params.get("concentration") or "UNKNOWN"
        return f"Shift salt concentration: {old_conc} -> {new_conc}"

    if action_type == "solvent_replace":
        replace = _param_dict(params, "replace")
        pairs = _extract_change_pairs(params.get("replacements"))
        if pairs:
            return "Replace solvent: " + "; ".join(
                f"{source or 'UNKNOWN'} -> {target or 'UNKNOWN'}" for source, target in pairs
            )
        from_solvent = str(params.get("from_solvent") or replace.get("from_solvent") or "").strip()
        to_solvent = str(params.get("to_solvent") or replace.get("to_solvent") or "").strip()
        if from_solvent or to_solvent:
            return f"Replace solvent: {from_solvent or 'UNKNOWN'} -> {to_solvent or 'UNKNOWN'}"
        removed = [name for name, _ in _extract_named_values(params.get("removed_solvents"))]
        added = [name for name, _ in _extract_named_values(params.get("added_solvents"))]
        if removed or added:
            return f"Replace solvent: remove [{_list_preview(removed)}], add [{_list_preview(added)}]"
        resulting_solvents = _active_mapping_names(_param_dict(params, "resulting_solvents"))
        current_solvents = _active_mapping_names((comp.get("solvents", {}) or {}))
        if resulting_solvents:
            removed_guess = [name for name in current_solvents if name not in resulting_solvents]
            added_guess = [name for name in resulting_solvents if name not in current_solvents]
            if removed_guess or added_guess:
                return f"Replace solvent: remove [{_list_preview(removed_guess)}], add [{_list_preview(added_guess)}]"
        invalid_reason = _invalid_proposal_reason(current_formulation, proposal_obj)
        if invalid_reason:
            return f"Replace solvent: invalid/incomplete proposal ({invalid_reason})"

    if action_type == "add_new_solvent":
        replace = _param_dict(params, "replace")
        solvent = str(params.get("solvent") or params.get("to_solvent") or replace.get("to_solvent") or "").strip()
        fraction = params.get("fraction") or params.get("ratio") or params.get("to_ratio")
        if not solvent:
            added = _extract_named_values(params.get("added_solvents"))
            if added:
                solvent, fraction = added[0]
        if not solvent:
            resulting_solvents = _active_mapping_items(_param_dict(params, "resulting_solvents"))
            current_solvents = set(_active_mapping_names((comp.get("solvents", {}) or {})))
            added_candidates = [(name, value) for name, value in resulting_solvents.items() if name not in current_solvents]
            if added_candidates:
                solvent, fraction = added_candidates[0]
        return f"Add solvent: {solvent or 'UNKNOWN'}" + (f" at {fraction}" if fraction not in (None, "") else "")

    if action_type == "remove_solvent":
        solvent = str(params.get("solvent") or params.get("from_solvent") or "").strip()
        if not solvent:
            removed = _extract_named_values(params.get("removed_solvents"))
            if removed:
                solvent = removed[0][0]
        return f"Remove solvent: {solvent or 'UNKNOWN'}"

    if action_type == "solvent_concentration_shift":
        solvent = str(params.get("solvent") or "").strip()
        to_ratio = params.get("to_ratio")
        if solvent and to_ratio not in (None, ""):
            return f"Shift solvent ratio: {solvent} -> {to_ratio}"
        ratio_updates = params.get("ratio_updates") or params.get("solvents") or {}
        if isinstance(ratio_updates, dict) and ratio_updates:
            parts = [f"{name}:{value}" for name, value in ratio_updates.items()]
            return "Shift solvent ratios: " + ", ".join(parts)

    if action_type == "additive_change":
        change_kind = str(params.get("change") or params.get("change_type") or "").strip().lower()

        def _first_named_additive(items):
            for it in (items or []):
                if isinstance(it, dict):
                    nm = it.get("name") or it.get("additive") or it.get("additive_name")
                    if nm:
                        return str(nm), (it.get("value") or it.get("amount") or it.get("concentration"))
                elif str(it).strip():
                    return str(it).strip(), None
            return None, None

        if change_kind == "remove":
            target = params.get("additive") or params.get("additive_name") or params.get("from_additive")
            if not target:
                target, _ = _first_named_additive(params.get("removed_additives"))
            return f"Remove additive: {target or 'UNKNOWN'}"
        if change_kind == "replace":
            old_add = params.get("from_additive") or params.get("additive") or params.get("additive_name")
            new_add = params.get("replacement_additive") or params.get("to_additive")
            if not new_add:
                new_add, _ = _first_named_additive(params.get("added_additives"))
            return f"Replace additive: {old_add or 'UNKNOWN'} -> {new_add or 'UNKNOWN'}"
        additive = params.get("additive") or params.get("additive_name") or params.get("to_additive")
        amount = params.get("to_amount") or params.get("amount")
        if not additive:
            additive, amt2 = _first_named_additive(params.get("added_additives") or params.get("adjusted_additives"))
            if amount in (None, "") and amt2:
                amount = amt2
        return f"Add/update additive: {additive or 'UNKNOWN'}" + (f" at {amount}" if amount not in (None, "") else "")

    return f"{action_type or 'unknown_action'}: {_compact_json_preview(params)}"


def summarize_rationale_text(text: Any, limit: int = 220) -> str:
    raw = " ".join(str(text or "").strip().split())
    if not raw:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", raw, maxsplit=1)[0]
    chosen = sentence if sentence else raw
    return chosen if len(chosen) <= limit else chosen[: limit - 3] + "..."


def summarize_reason_text(text: Any, limit: int = 320) -> str:
    return summarize_rationale_text(text, limit=limit)


def apply_proposal_to_formulation(current_formulation: Dict[str, Any], proposal_obj: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str]:
    new_formulation = copy.deepcopy(current_formulation)
    comp = new_formulation.setdefault("composition", {})
    comp.setdefault("solvents", {})
    comp.setdefault("additives", {})

    action_type = str(proposal_obj.get("action_type", "")).strip()
    params = proposal_obj.get("parameters", {}) or {}
    if not isinstance(params, dict):
        params = {}

    try:
        if action_type == "salt_replace":
            to_salt = params.get("to_salt") or params.get("salt_to") or params.get("new_salt")
            to_system = _param_dict(params, "to_salt_system")
            if isinstance(to_salt, dict) and to_salt.get("name"):
                if to_salt.get("concentration") and not params.get("to_concentration"):
                    params = {**params, "to_concentration": to_salt["concentration"]}
                to_salt = to_salt["name"]
            elif isinstance(to_salt, list):   # [{name, concentration}, ...] dual-salt target
                to_system = dict(_extract_named_values(to_salt))
                to_salt = None
            if not to_salt and to_system:
                # {salt: concentration, ...} -> "A + B", with the total as the concentration if given
                to_salt = " + ".join(str(k) for k in to_system)
                if not params.get("to_concentration"):
                    total = params.get("total_salt_concentration")
                    comp["salt_concentration"] = str(total) if total else " + ".join(
                        f"{v} {k}" for k, v in to_system.items())
            if not to_salt:
                _, to_salt = _parse_two_materials_from_combined_salt(params.get("salt"))
            if not to_salt:
                return current_formulation, False, "missing target salt"
            comp["salt"] = str(to_salt)
            if params.get("to_concentration"):
                comp["salt_concentration"] = str(params["to_concentration"])
            return new_formulation, True, "salt replaced"

        if action_type == "add_new_salt":
            base_salt = str(comp.get("salt", "")).strip()
            added_salt = (params.get("to_salt") or params.get("salt_to") or params.get("added_salt")
                          or params.get("new_salt") or params.get("new_salt_name"))
            if isinstance(added_salt, dict):
                if added_salt.get("concentration") and not params.get("concentration"):
                    params = {**params, "concentration": added_salt["concentration"]}
                added_salt = added_salt.get("name")
            if params.get("new_salt_concentration") and not params.get("concentration"):
                params = {**params, "concentration": params["new_salt_concentration"]}
            if not added_salt:
                listed = _extract_named_values(params.get("added_salts"))
                if listed:
                    added_salt = listed[0][0]
            if not added_salt:
                _, added_salt = _parse_two_materials_from_combined_salt(params.get("salt"))
            if not added_salt:
                return current_formulation, False, "missing added salt"
            if base_salt and added_salt not in base_salt:
                comp["salt"] = f"{base_salt} + {added_salt}"
            else:
                comp["salt"] = str(added_salt)
            if params.get("to_concentration"):
                comp["salt_concentration"] = str(params["to_concentration"])
            elif params.get("concentration") and comp.get("salt_concentration"):
                comp["salt_concentration"] = f"{comp['salt_concentration']} + {params['concentration']} {added_salt}"
            return new_formulation, True, "new salt added"

        if action_type == "salt_concentration_shift":
            to_conc = (params.get("to_concentration") or params.get("concentration")
                       or params.get("new_concentration") or params.get("target_concentration"))
            for k in ("to_concentration_M", "to_concentration_molar"):
                if not to_conc and params.get(k) is not None:
                    to_conc = f"{params[k]} M"
            if not to_conc:
                return current_formulation, False, "missing target concentration"
            comp["salt_concentration"] = str(to_conc)
            return new_formulation, True, "salt concentration shifted"

        if action_type == "solvent_replace":
            solvents = comp.setdefault("solvents", {})
            replace = _param_dict(params, "replace")
            from_solvent = params.get("from_solvent") or replace.get("from_solvent")
            to_solvent = params.get("to_solvent") or replace.get("to_solvent")
            replacements = _extract_change_pairs(params.get("replacements"))
            removed_solvents = _extract_named_values(params.get("removed_solvents"))
            added_solvents = _extract_named_values(params.get("added_solvents"))
            resulting_solvents = _active_mapping_items(_param_dict(params, "resulting_solvents"))

            if (not from_solvent or not to_solvent) and replacements:
                from_solvent, to_solvent = replacements[0]

            if replacements:
                applied_any = False
                for source, target in replacements:
                    if not target:
                        continue
                    old_value = None
                    src_key = _resolve_solvent_key(source, list(solvents.keys()))
                    if src_key is not None:
                        old_value = solvents.pop(src_key)
                    if old_value is None:
                        for name, value in removed_solvents:
                            if _resolve_solvent_key(source, [name]) and value not in (None, ""):
                                old_value = value
                                break
                    if old_value is None:
                        for name, value in added_solvents:
                            if _resolve_solvent_key(target, [name]) and value not in (None, ""):
                                old_value = value
                                break
                    target_key = _canonical_solvent_symbol(target) or str(target)
                    solvents[target_key] = old_value if old_value not in (None, "") else params.get("to_ratio", params.get("fraction", "1"))
                    applied_any = True
                if applied_any:
                    return new_formulation, True, "solvent replaced"

            if (not from_solvent or not to_solvent) and removed_solvents and added_solvents:
                removed_name, removed_value = removed_solvents[0]
                added_name, added_value = added_solvents[0]
                from_solvent = from_solvent or removed_name
                to_solvent = to_solvent or added_name
                replacement_value = added_value if added_value not in (None, "") else removed_value
                src_key = _resolve_solvent_key(from_solvent, list(solvents.keys()))
                if src_key is not None:
                    solvents.pop(src_key, None)
                target_key = _canonical_solvent_symbol(to_solvent) or str(to_solvent)
                solvents[target_key] = replacement_value if replacement_value not in (None, "") else params.get("to_ratio", params.get("fraction", "1"))
                return new_formulation, True, "solvent replaced"

            if resulting_solvents:
                comp["solvents"] = {
                    (_canonical_solvent_symbol(name) or str(name)): value
                    for name, value in resulting_solvents.items()
                }
                return new_formulation, True, "solvent replaced from resulting_solvents"

            invalid_reason = _invalid_proposal_reason(current_formulation, proposal_obj)
            if invalid_reason:
                return current_formulation, False, invalid_reason
            if not to_solvent:
                return current_formulation, False, "missing target solvent"
            src_key = _resolve_solvent_key(from_solvent, list(solvents.keys())) if from_solvent else None
            target_key = _canonical_solvent_symbol(to_solvent) or str(to_solvent)
            if src_key is not None:
                old_value = solvents.pop(src_key)
                solvents[target_key] = old_value
            else:
                solvents[target_key] = params.get("to_ratio", params.get("fraction", "1"))
            return new_formulation, True, "solvent replaced"

        if action_type == "add_new_solvent":
            solvents = comp.setdefault("solvents", {})
            resulting = _active_mapping_items(_param_dict(params, "resulting_solvents"))
            if resulting and any(_resolve_solvent_key(n, list(solvents.keys())) is None for n in resulting):
                # the proposal states the full new solvent blend (incl. rebalanced fractions)
                comp["solvents"] = {(_resolve_solvent_key(n, list(solvents.keys()))
                                     or _canonical_solvent_symbol(n) or str(n)): v for n, v in resulting.items()}
                return new_formulation, True, "new solvent added (resulting blend)"
            replace = _param_dict(params, "replace")
            solvent = params.get("solvent") or params.get("to_solvent") or replace.get("to_solvent")
            fraction = params.get("fraction") or params.get("ratio") or params.get("to_ratio") or "1"
            if not solvent:
                added_solvents = (_extract_named_values(params.get("added_solvents"))
                                  or _extract_named_values(params.get("new_solvents")))
                if added_solvents:
                    solvent, added_value = added_solvents[0]
                    if added_value not in (None, ""):
                        fraction = added_value
            if not solvent:
                resulting_solvents = _active_mapping_items(_param_dict(params, "resulting_solvents"))
                for name, value in resulting_solvents.items():
                    if str(name) not in solvents:
                        solvent = name
                        if value not in (None, ""):
                            fraction = value
                        break
            if not solvent:
                return current_formulation, False, "missing new solvent"
            solvents[str(solvent)] = fraction
            return new_formulation, True, "new solvent added"

        if action_type == "remove_solvent":
            solvents = comp.setdefault("solvents", {})
            solvent = params.get("solvent") or params.get("from_solvent")
            if isinstance(params.get("removed_solvent"), dict):
                solvent = solvent or params["removed_solvent"].get("short_name") or params["removed_solvent"].get("name")
            elif isinstance(params.get("removed_solvent"), str):
                solvent = solvent or params["removed_solvent"]
            if not solvent:
                removed_solvents = _extract_named_values(params.get("removed_solvents"))
                if removed_solvents:
                    solvent = removed_solvents[0][0]
            if not solvent:
                return current_formulation, False, "missing solvent to remove"
            key = _resolve_solvent_key(solvent, list(solvents.keys()))
            if key is not None:
                solvents.pop(key, None)
                resulting = _active_mapping_items(_param_dict(params, "resulting_solvents"))
                if resulting:   # rebalanced fractions of the remaining solvents, if stated
                    for n, v in resulting.items():
                        k2 = _resolve_solvent_key(n, list(solvents.keys()))
                        if k2 is not None:
                            solvents[k2] = v
                return new_formulation, True, "solvent removed"
            return current_formulation, False, "solvent not found"

        if action_type == "solvent_concentration_shift":
            solvents = comp.setdefault("solvents", {})
            solvent = params.get("solvent")
            if solvent and params.get("to_ratio") is not None:
                solvents[str(solvent)] = params["to_ratio"]
                return new_formulation, True, "single solvent ratio shifted"

            updated_any = False
            # gpt commonly emits ratio_changes=[{name, from, to}]; also accept ratio_updates /
            # solvents / resulting_solvents maps. Resolve names to the existing solvent keys so a
            # short-symbol vs full-name mismatch does not silently drop the update.
            for rc in (params.get("ratio_changes") or []):
                if isinstance(rc, dict) and rc.get("name") and rc.get("to") is not None:
                    key = _resolve_solvent_key(rc["name"], list(solvents.keys())) or _canonical_solvent_symbol(rc["name"]) or str(rc["name"])
                    solvents[key] = rc["to"]
                    updated_any = True
            ratio_updates = (params.get("new_solvent_fractions") or params.get("to_fractions") or params.get("ratio_updates")
                             or params.get("to_solvent_ratios") or params.get("to_ratios") or params.get("target_ratios")
                             or params.get("new_ratios") or params.get("solvents")
                             or _param_dict(params, "resulting_solvents") or {})
            if isinstance(ratio_updates, dict):
                for name, value in ratio_updates.items():
                    key = _resolve_solvent_key(name, list(solvents.keys())) or _canonical_solvent_symbol(name) or str(name)
                    solvents[key] = value
                    updated_any = True
            if updated_any:
                return new_formulation, True, "solvent ratios shifted"
            return current_formulation, False, "missing solvent ratio update"

        if action_type == "additive_change":
            additives = comp.setdefault("additives", {})
            additive = params.get("additive") or params.get("additive_name")
            change_kind = str(params.get("change") or params.get("change_type") or "").strip().lower()
            # List form, as in the RAG change summaries the generator is shown:
            # added_additives / removed_additives / adjusted_additives = [{name, value|concentration|to}]
            list_keys = ("added_additives", "removed_additives", "adjusted_additives")
            if not additive and any(isinstance(params.get(k), list) and params.get(k) for k in list_keys):
                touched = False
                for item in params.get("removed_additives") or []:
                    name = item.get("name") if isinstance(item, dict) else item
                    if name and str(name) in additives:
                        additives.pop(str(name), None)
                        touched = True
                for key in ("added_additives", "adjusted_additives"):
                    for item in params.get(key) or []:
                        if isinstance(item, dict) and item.get("name"):
                            amount = (item.get("to") or item.get("value") or item.get("concentration")
                                      or item.get("amount") or "1 wt%")
                            additives[str(item["name"])] = amount
                            touched = True
                if touched:
                    return new_formulation, True, "additives changed"
                return current_formulation, False, "additive list change had no effect"
            for key in ("additives", "added_additives", "new_additives"):
                mapping = params.get(key)
                if isinstance(mapping, list):
                    mapping = {n: v for n, v in _extract_named_values(mapping)}
                if not additive and isinstance(mapping, dict) and mapping:
                    for name, amount in mapping.items():
                        additives[str(name)] = amount if amount not in (None, "") else "1 wt%"
                    return new_formulation, True, "additives changed"
            if not additive and change_kind != "remove":
                return current_formulation, False, "missing additive name"

            if change_kind == "remove":
                target = additive or params.get("from_additive")
                if target and str(target) in additives:
                    additives.pop(str(target), None)
                    return new_formulation, True, "additive removed"
                return current_formulation, False, "additive to remove not found"

            if change_kind == "replace":
                old_add = params.get("from_additive") or additive
                new_add = params.get("replacement_additive") or params.get("to_additive")
                amount = params.get("to_amount") or params.get("amount") or "1 wt%"
                if old_add and str(old_add) in additives:
                    additives.pop(str(old_add), None)
                if not new_add:
                    return current_formulation, False, "missing replacement additive"
                additives[str(new_add)] = amount
                return new_formulation, True, "additive replaced"

            amount = params.get("to_amount") or params.get("amount") or "1 wt%"
            additives[str(additive)] = amount
            return new_formulation, True, "additive changed"

        return current_formulation, False, f"unsupported action_type: {action_type}"
    except Exception as e:
        return current_formulation, False, f"apply exception: {e}"


def print_candidate(
    step_idx: int,
    current_formulation: Dict[str, Any],
    selected_action: str,
    ranking: List[Dict[str, float]],
    proposal_obj: Dict[str, Any],
    llm_output: Dict[str, Any],
    verbose_terminal: bool = False,
    candidate_actions: Optional[List[str]] = None,
    decision_bundle: Optional[Dict[str, Any]] = None,
    retry_mode: bool = False,
    avoid_target_families: Optional[List[str]] = None,
    duplicate_warning: bool = False,
) -> None:
    print("\n" + "=" * 80)
    print(f"step {step_idx}")
    selection_path = str((decision_bundle or {}).get("selection_path", "")).strip()
    if retry_mode:
        retry_state_for_action = (decision_bundle or {}).get("retry_state_for_action", {}) or {}
        prior_nonpositive = int(retry_state_for_action.get("nonpositive_count", 0) or 0)
        print("mode: retry within the same direction (no new top-3 screening this step)")
        print(f"current direction: {selected_action}")
        if prior_nonpositive > 0:
            print(f"nonpositive feedbacks already received in this direction: {prior_nonpositive}")
        if avoid_target_families:
            print("avoid target families: " + ", ".join(avoid_target_families))
    else:
        excluded_actions = list((decision_bundle or {}).get("excluded_actions", []) or [])
        raw_candidate_actions = list((decision_bundle or {}).get("raw_candidate_actions", []) or [])
        selection_reason = summarize_reason_text((decision_bundle or {}).get("selection_reason"), limit=260)

        if selection_path in ("initial_top3_highest", "current_top3_highest"):
            if selection_path == "initial_top3_highest":
                print("mode: state unchanged from initial; take the highest-ranked untried direction of the initial MLP top-3")
                top_label = "initial MLP top-3: "
            else:
                print("mode: state changed; rerank the current formulation and take the highest-ranked untried direction of its MLP top-3")
                top_label = "current MLP top-3: "
            if raw_candidate_actions:
                print(top_label + ", ".join(raw_candidate_actions))
            if excluded_actions:
                print("already attempted from this top-3: " + ", ".join(excluded_actions))
            if candidate_actions:
                print("untried directions, in prior order: " + ", ".join(candidate_actions))
        elif selection_path == "positive_solvent_trajectory_continuation":
            trajectory_guidance = (decision_bundle or {}).get("trajectory_guidance", {}) or {}
            print("mode: continue accepted solvent-replacement trajectory")
            completed = list(trajectory_guidance.get("completed_source_solvents", []) or [])
            remaining = list(trajectory_guidance.get("remaining_source_solvents", []) or [])
            preferred = str((decision_bundle or {}).get("trajectory_source", "")).strip()
            if completed:
                print("already accepted in this solvent trajectory: " + ", ".join(completed))
            if remaining:
                print("remaining source solvents to try: " + ", ".join(remaining))
            if preferred:
                print("current source solvent under consideration: " + preferred)
        elif selection_path == "escalate_within_direction":
            escalation_guidance = (decision_bundle or {}).get("escalation_guidance", {}) or {}
            level = (decision_bundle or {}).get("escalation_level")
            siblings = list(escalation_guidance.get("sibling_actions", []) or [])
            transferred_from = (decision_bundle or {}).get("escalation_transferred_from")
            print("mode: escalate within the accepted direction (more aggressive / less common variant after positive)")
            if len(siblings) > 1:
                print("sibling actions explored: " + ", ".join(siblings))
            if transferred_from:
                print(f"  >> transferred to sibling action: {transferred_from} -> {selected_action}")
            print(f"escalation action: {selected_action}" + (f" (level {level})" if level else ""))
            accepted = list(escalation_guidance.get("accepted_changes_in_order", []) or [])
            if accepted:
                print("accepted trend so far in this direction: " + "; ".join(accepted))
        print(f"selected direction: {selected_action}")
        if selection_reason:
            print("why this direction was selected:")
            print(selection_reason)
    if duplicate_warning:
        print("warning: generated proposal still matched a previously rejected proposal after regeneration attempts")

    candidate_set = set(candidate_actions or [])
    alternative_rows = [
        f"{str(row.get('action', ''))} ({float(row.get('prob', 0.0)):.3f})"
        for row in ranking
        if str(row.get("action", "")) and str(row.get("action", "")) not in candidate_set
    ]
    if alternative_rows:
        print("next likely directions outside this screening: " + ", ".join(alternative_rows[:3]))

    print("current formulation:")
    print(composition_summary(current_formulation))

    if proposal_obj:
        print("proposed change in the selected direction:")
        print(summarize_proposal_change(current_formulation, proposal_obj))

        preview_formulation, preview_applied, preview_note = apply_proposal_to_formulation(current_formulation, proposal_obj)
        if preview_applied:
            print("result if approved:")
            print(composition_summary(preview_formulation))
        else:
            print(f"result if approved: unable to preview ({preview_note})")

        rationale = summarize_rationale_text(proposal_obj.get("rationale"))
        if rationale:
            print("why this specific change may help:")
            print(rationale)

        uncertainty = str(proposal_obj.get("uncertainty_level", "")).strip()
        if uncertainty:
            print(f"uncertainty: {uncertainty}")

        precedents = proposal_obj.get("supporting_precedents")
        if isinstance(precedents, list):
            print(f"literature support: {len(precedents)} precedent(s) attached")

        print("approval meaning: enter 'positive' only if you want to apply the exact change above")
    else:
        err = llm_output.get("proposal_error")
        print(f"proposal unavailable: {err or 'proposal not available'}")

    if verbose_terminal:
        print("-" * 80)
        print("full action ranking:")
        for row in ranking[:5]:
            print(json.dumps(row, ensure_ascii=False))
        print("full proposal payload:")
        if proposal_obj:
            print(json.dumps(proposal_obj, ensure_ascii=False, indent=2))
        else:
            print(json.dumps({"proposal_error": llm_output.get("proposal_error") or "proposal not available"}, ensure_ascii=False, indent=2))
    print("=" * 80)


def ask_user_reward() -> str:
    while True:
        raw = input("reward [positive=apply, negative=reject, unchange=no-change, quit]: ").strip().lower()
        if raw in {"quit", "q", "exit"}:
            return "quit"
        normalized = REWARD_ALIASES.get(raw)
        if normalized:
            return normalized
        print("invalid input, please enter: positive / negative / unchange / quit")


def save_session(path: Path, session: Dict[str, Any]) -> None:
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")


def output_paths_from_folder(folder: Path) -> Dict[str, Path]:
    return {
        "state": folder / "initial_state.json",
        "cache": folder / ".prior_suggestions_cache.json",
        "session": folder / "interactive_feedback_session.json",
        "final_state": folder / "interactive_feedback_final_state.json",
        "attempts": folder / "interactive_feedback_attempts.json",
    }


def persist_feedback_outputs(paths: Dict[str, Path], session: Dict[str, Any]) -> None:
    save_session(paths["session"], session)
    final_state = session.get("final_state", {})
    paths["final_state"].write_text(json.dumps(final_state, ensure_ascii=False, indent=2), encoding="utf-8")
    attempts = session.get("history", [])
    paths["attempts"].write_text(json.dumps(attempts, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    folder = args.folder.resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"folder not found or not a directory: {folder}")
        print("no feedback processed")
        return

    paths = output_paths_from_folder(folder)
    state_path = paths["state"]
    if not state_path.exists():
        print(f"initial_state.json not found in folder: {folder}")
        print("no feedback processed")
        return

    state_payload = load_json(state_path)
    current_formulation = normalize_current_formulation(state_payload)
    objective = extract_objective_for_llm(state_payload)
    papers = load_papers(args.data)
    all_precedents = build_precedents(papers)
    cache = load_cache(paths["cache"])

    session: Dict[str, Any] = {
        "folder": str(folder),
        "state_path": str(state_path),
        "initial_state": copy.deepcopy(current_formulation),
        "objective": objective,
        "history": [],
        "final_state": copy.deepcopy(current_formulation),
    }

    persist_feedback_outputs(paths, session)

    blocked_actions: List[str] = [
        a.strip() for a in str(getattr(args, "block_actions", "") or "").split(",") if a.strip()
    ]
    action_retry_state: Dict[str, Dict[str, Any]] = {}
    attempted_actions_in_session: List[str] = []
    active_solvent_trajectory: Optional[Dict[str, Any]] = None
    escalation_state: Optional[Dict[str, Any]] = None
    initial_state_payload = state_payload_from_formulation(
        current_formulation, objective=objective
    )
    _initial_mlp_selected_action, initial_ranking = infer_action_with_mlp(
        state_payload=initial_state_payload,
        current_formulation=current_formulation,
        prior_ckpt_path=args.prior,
    )
    initial_top_actions = [
        a for a in top_enabled_actions(
            initial_ranking,
            initial_state_payload,
            n=max(1, int(args.top_action_candidates)) + len(blocked_actions),
        )
        if a not in blocked_actions
    ][:max(1, int(args.top_action_candidates))]

    def stop_with_message(message: str) -> None:
        session["final_state"] = copy.deepcopy(current_formulation)
        session["termination_reason"] = message
        persist_feedback_outputs(paths, session)
        print(message)
        print(f"feedback outputs saved to: {folder}")

    print("initial formulation:")
    print(composition_summary(current_formulation))

    for step_idx in range(1, args.max_steps + 1):
        family_tables, family_indexes = prepare_family_indexes_for_formulation(
            current_formulation=current_formulation,
            base_dir=Path(__file__).resolve().parent,
            model=args.llm_mode,
        )
        current_state_payload = state_payload_from_formulation(
            current_formulation, objective=objective
        )
        llm_formulation = llm_ready_formulation(current_formulation)
        _mlp_selected_action, ranking = infer_action_with_mlp(
            state_payload=current_state_payload,
            current_formulation=current_formulation,
            prior_ckpt_path=args.prior,
        )
        selection_resolution_count = 0
        while True:
            selection_resolution_count += 1
            if selection_resolution_count > max(3, len(ACTION_SPACE) + 2):
                stop_with_message("Unable to find a legal proposal after exhausting the remaining action directions; stopping.")
                return

            forced_action: Optional[str] = None
            for action_name, state in action_retry_state.items():
                if action_name in blocked_actions:
                    continue
                nonpositive_count = int(state.get("nonpositive_count", 0))
                if 0 < nonpositive_count < args.nonpositive_threshold:
                    forced_action = action_name
                    break

            decision_bundle = {}
            selected_candidate = {}
            candidate_actions = []
            duplicate_warning = False
            avoid_target_families = []
            trajectory_guidance = None
            trajectory_mode = False
            trajectory_source = ""

            if escalation_state and escalation_state.get("sibling_actions"):
                # After a positive: escalate. Sibling exploration (replace <-> add) happens
                # ONLY here, in the positive-triggered escalation path.
                escalation_guidance = build_escalation_guidance(escalation_state, current_formulation)
                last_action = str(escalation_state.get("last_action", "") or "")
                enabled = set(enabled_actions_from_state(current_state_payload))
                escalation_siblings = [
                    a for a in (escalation_state.get("sibling_actions") or [])
                    if a in enabled and a not in blocked_actions
                ]
                if not escalation_siblings:
                    escalation_state = None
                    continue
                # SEQUENTIAL sibling exploration: work ONE sibling at a time, starting from the
                # last accepted action; only hop to the next sibling once the current one is
                # exhausted (see the reject handler). The ladder climbs the SAME direction to its
                # limit, THEN sister-jumps (e.g. solvent_replace -> solvent_concentration_shift).
                current_sibling = escalation_state.get("current_sibling") or last_action
                if current_sibling not in escalation_siblings:
                    current_sibling = escalation_siblings[0]
                escalation_state["current_sibling"] = current_sibling
                escalation_siblings = [current_sibling]
                selected_action = escalation_siblings[0]
                escalation_rejected = list(escalation_state.get("rejected_signatures", []) or [])
                allowed_target_families = build_allowed_target_families(
                    selected_action, family_tables, [],
                    force_full_allowlist=not args.allow_new_molecules,
                )
                if allowed_target_families == []:
                    escalation_state = None
                    continue
                candidate_bundle, proposal_obj, duplicate_warning = generate_candidate_with_duplicate_guard(
                    current_formulation=llm_formulation,
                    action_strategy=selected_action,
                    state_objective=objective,
                    papers=papers,
                    top_k=args.top_k,
                    llm_mode=args.llm_mode,
                    llm_temperature=args.llm_temperature,
                    cache=cache,
                    all_precedents=all_precedents,
                    family_indexes=family_indexes,
                    allowed_target_families=allowed_target_families,
                    avoid_target_families=[],
                    allow_new_molecules=args.allow_new_molecules,
                    rejected_signatures=escalation_rejected,
                    duplicate_regeneration_limit=args.duplicate_regeneration_limit,
                    trajectory_guidance=escalation_guidance,
                )
                save_candidate_cache(paths["cache"], cache, candidate_bundle)
                llm_output = candidate_bundle.get("llm_output", {}) or {}
                proposal_obj, llm_output = sanitize_proposal_against_molecule_scope(
                    proposal_obj, llm_output, family_indexes, args.allow_new_molecules,
                )
                candidate_bundle["llm_output"] = llm_output
                selected_candidate = {
                    "index": 0,
                    "action_strategy": selected_action,
                    "proposal_source": candidate_bundle.get("proposal_source", "unknown"),
                    "payload": candidate_bundle.get("payload", {}),
                    "llm_output": llm_output,
                }
                candidate_actions = list(escalation_siblings)
                decision_bundle = {
                    "schema": "interactive_escalation_v1",
                    "selection_path": "escalate_within_direction",
                    "selected_action_strategy": selected_action,
                    "candidate_actions": candidate_actions,
                    "escalation_guidance": copy.deepcopy(escalation_guidance),
                    "escalation_level": int(escalation_state.get("level", 1)),
                    "duplicate_warning": duplicate_warning,
                }
                # Flag (and print) when escalation hopped to a different sibling action.
                if last_action and selected_action != last_action:
                    decision_bundle["escalation_transferred_from"] = last_action
                retry_mode = False
                trajectory_mode = False
            elif solvent_trajectory_has_pending_sources(active_solvent_trajectory):
                trajectory_source = select_solvent_trajectory_source(active_solvent_trajectory or {}, args.nonpositive_threshold)
                if not trajectory_source:
                    active_solvent_trajectory = None
                    continue
                active_solvent_trajectory = copy.deepcopy(active_solvent_trajectory)
                active_solvent_trajectory["current_source"] = trajectory_source
                trajectory_guidance = build_positive_feedback_trajectory_guidance(active_solvent_trajectory)
                selected_action = "solvent_replace"
                source_avoid_map = dict((active_solvent_trajectory or {}).get("source_avoid_target_families", {}) or {})
                source_rejected_map = dict((active_solvent_trajectory or {}).get("source_rejected_proposal_signatures", {}) or {})
                avoid_target_families = list(source_avoid_map.get(trajectory_source, []) or [])
                rejected_signatures = list(source_rejected_map.get(trajectory_source, []) or [])
                allowed_target_families = build_allowed_target_families(
                    selected_action,
                    family_tables,
                    avoid_target_families,
                    force_full_allowlist=not args.allow_new_molecules,
                )
                if allowed_target_families == []:
                    active_solvent_trajectory = mark_solvent_trajectory_source_exhausted(
                        active_solvent_trajectory,
                        trajectory_source,
                    )
                    continue
                candidate_bundle, proposal_obj, duplicate_warning = generate_candidate_with_duplicate_guard(
                    current_formulation=llm_formulation,
                    action_strategy=selected_action,
                    state_objective=objective,
                    papers=papers,
                    top_k=args.top_k,
                    llm_mode=args.llm_mode,
                    llm_temperature=args.llm_temperature,
                    cache=cache,
                    all_precedents=all_precedents,
                    family_indexes=family_indexes,
                    allowed_target_families=allowed_target_families,
                    avoid_target_families=avoid_target_families,
                    allow_new_molecules=args.allow_new_molecules,
                    rejected_signatures=rejected_signatures,
                    duplicate_regeneration_limit=args.duplicate_regeneration_limit,
                    trajectory_guidance=trajectory_guidance,
                )
                save_candidate_cache(paths["cache"], cache, candidate_bundle)
                llm_output = candidate_bundle.get("llm_output", {}) or {}
                proposal_obj, llm_output = sanitize_proposal_against_family_constraints(
                    proposal_obj,
                    llm_output,
                    family_indexes,
                    avoid_target_families,
                )
                proposal_obj, llm_output = sanitize_proposal_against_molecule_scope(
                    proposal_obj,
                    llm_output,
                    family_indexes,
                    args.allow_new_molecules,
                )
                candidate_bundle["llm_output"] = llm_output
                selected_candidate = {
                    "index": 0,
                    "action_strategy": selected_action,
                    "proposal_source": candidate_bundle.get("proposal_source", "unknown"),
                    "payload": candidate_bundle.get("payload", {}),
                    "llm_output": llm_output,
                }
                decision_bundle = {
                    "schema": "interactive_positive_trajectory_v1",
                    "selection_path": "positive_solvent_trajectory_continuation",
                    "selected_action_strategy": selected_action,
                    "trajectory_guidance": copy.deepcopy(trajectory_guidance),
                    "trajectory_state": copy.deepcopy(active_solvent_trajectory),
                    "trajectory_source": trajectory_source,
                    "avoid_target_families": copy.deepcopy(avoid_target_families),
                    "duplicate_warning": duplicate_warning,
                }
                retry_mode = False
                trajectory_mode = True
            elif forced_action:
                selected_action = forced_action
                retry_state_for_action = action_retry_state.get(selected_action, {})
                avoid_target_families = list(retry_state_for_action.get("avoid_target_families", []) or [])
                allowed_target_families = build_allowed_target_families(
                    selected_action,
                    family_tables,
                    avoid_target_families,
                    force_full_allowlist=not args.allow_new_molecules,
                )
                if allowed_target_families == []:
                    retry_state_for_action = action_retry_state.setdefault(
                        selected_action,
                        {"nonpositive_count": 0, "avoid_target_families": [], "rejected_proposal_signatures": []},
                    )
                    retry_state_for_action["nonpositive_count"] = max(
                        int(retry_state_for_action.get("nonpositive_count", 0)),
                        int(args.nonpositive_threshold),
                    )
                    blocked_actions = [x for x in blocked_actions if x != selected_action] + [selected_action]
                    continue
                rejected_signatures = list(retry_state_for_action.get("rejected_proposal_signatures", []) or [])
                retry_guidance = build_retry_guidance(list(retry_state_for_action.get("rejected_changes", []) or []))
                candidate_bundle, proposal_obj, duplicate_warning = generate_candidate_with_duplicate_guard(
                    current_formulation=llm_formulation,
                    action_strategy=selected_action,
                    state_objective=objective,
                    papers=papers,
                    top_k=args.top_k,
                    llm_mode=args.llm_mode,
                    llm_temperature=args.llm_temperature,
                    cache=cache,
                    all_precedents=all_precedents,
                    family_indexes=family_indexes,
                    allowed_target_families=allowed_target_families,
                    avoid_target_families=avoid_target_families,
                    allow_new_molecules=args.allow_new_molecules,
                    rejected_signatures=rejected_signatures,
                    duplicate_regeneration_limit=args.duplicate_regeneration_limit,
                    trajectory_guidance=retry_guidance,
                )
                save_candidate_cache(paths["cache"], cache, candidate_bundle)
                llm_output = candidate_bundle.get("llm_output", {}) or {}
                proposal_obj, llm_output = sanitize_proposal_against_family_constraints(
                    proposal_obj,
                    llm_output,
                    family_indexes,
                    avoid_target_families,
                )
                proposal_obj, llm_output = sanitize_proposal_against_molecule_scope(
                    proposal_obj,
                    llm_output,
                    family_indexes,
                    args.allow_new_molecules,
                )
                candidate_bundle["llm_output"] = llm_output
                selected_candidate = {
                    "index": 0,
                    "action_strategy": selected_action,
                    "proposal_source": candidate_bundle.get("proposal_source", "unknown"),
                    "payload": candidate_bundle.get("payload", {}),
                    "llm_output": llm_output,
                }
                decision_bundle = {
                    "schema": "interactive_retry_v1",
                    "selected_action_strategy": selected_action,
                    "retry_guidance": copy.deepcopy(retry_guidance),
                    "retry_state_for_action": copy.deepcopy(retry_state_for_action),
                    "allowed_target_families": copy.deepcopy(allowed_target_families),
                    "avoid_target_families": copy.deepcopy(avoid_target_families),
                    "duplicate_warning": duplicate_warning,
                }
                retry_mode = True
            else:
                # Directions are tried in the prior's order: take the highest-ranked direction of
                # the MLP top-3 not yet attempted in this session (top-3 of the initial state while
                # the formulation is unchanged, of the current state once it has changed).
                state_changed_from_initial = current_formulation != session["initial_state"]
                if state_changed_from_initial:
                    raw_top_actions = top_enabled_actions(
                        ranking,
                        current_state_payload,
                        n=max(1, int(args.top_action_candidates)),
                    )
                    selection_path = "current_top3_highest"
                    stop_text = ("current formulation differs from the initial state, but all directions in the "
                                 "current MLP top-3 have already been attempted in this session; stopping.")
                    reason = ("Highest-ranked direction of the current MLP top-3 not yet attempted in this "
                              "session.")
                else:
                    raw_top_actions = list(initial_top_actions)
                    selection_path = "initial_top3_highest"
                    stop_text = ("state is still identical to the initial formulation and all directions in the "
                                 "initial MLP top-3 have already been attempted; stopping.")
                    reason = "Highest-ranked direction of the initial MLP top-3 not yet attempted."
                excluded_actions = [a for a in raw_top_actions if a in attempted_actions_in_session]
                candidate_actions = [a for a in raw_top_actions if a not in attempted_actions_in_session]
                if not candidate_actions:
                    stop_with_message(stop_text)
                    return
                selected_action, selected_candidate, llm_output, proposal_obj, candidate_actions, decision_bundle = generate_single_direction_decision(
                    current_formulation=llm_formulation,
                    action_strategy=str(candidate_actions[0]),
                    objective=objective,
                    papers=papers,
                    all_precedents=all_precedents,
                    top_k=args.top_k,
                    llm_mode=args.llm_mode,
                    llm_temperature=args.llm_temperature,
                    cache=cache,
                    cache_path=paths["cache"],
                    family_tables=family_tables,
                    family_indexes=family_indexes,
                    allow_new_molecules=args.allow_new_molecules,
                    duplicate_regeneration_limit=args.duplicate_regeneration_limit,
                    candidate_actions=candidate_actions,
                    raw_candidate_actions=raw_top_actions,
                    excluded_actions=excluded_actions,
                    schema="interactive_prior_order_v1",
                    selection_path=selection_path,
                    selection_reason=reason,
                    trajectory_guidance=None,
                )
                retry_mode = False

            invalid_error = str((llm_output or {}).get("proposal_error", "") or "")
            if trajectory_mode and not proposal_obj and invalid_error.startswith("invalid proposal:"):
                retry_counts = dict((active_solvent_trajectory or {}).get("source_retry_counts", {}) or {})
                retry_counts[trajectory_source] = int(retry_counts.get(trajectory_source, 0) or 0) + 1
                if active_solvent_trajectory is None:
                    active_solvent_trajectory = {}
                active_solvent_trajectory["source_retry_counts"] = retry_counts
                if int(retry_counts.get(trajectory_source, 0)) >= int(args.nonpositive_threshold):
                    active_solvent_trajectory = mark_solvent_trajectory_source_exhausted(
                        active_solvent_trajectory,
                        trajectory_source,
                    )
                else:
                    active_solvent_trajectory["current_source"] = trajectory_source
                continue
            if retry_mode and not proposal_obj and invalid_error.startswith("invalid proposal:"):
                retry_state_for_action = action_retry_state.setdefault(
                    selected_action,
                    {"nonpositive_count": 0, "avoid_target_families": [], "rejected_proposal_signatures": []},
                )
                retry_state_for_action["nonpositive_count"] = max(
                    int(retry_state_for_action.get("nonpositive_count", 0)),
                    int(args.nonpositive_threshold),
                )
                blocked_actions = [x for x in blocked_actions if x != selected_action] + [selected_action]
                continue
            break

        print_candidate(
            step_idx=step_idx,
            current_formulation=current_formulation,
            selected_action=selected_action,
            ranking=ranking,
            proposal_obj=proposal_obj,
            llm_output=llm_output,
            verbose_terminal=args.verbose_terminal,
            candidate_actions=candidate_actions,
            decision_bundle=decision_bundle,
            retry_mode=retry_mode,
            avoid_target_families=avoid_target_families,
            duplicate_warning=duplicate_warning,
        )

        state_before_step = copy.deepcopy(current_formulation)

        reward = ask_user_reward()
        if reward == "quit":
            session["final_state"] = copy.deepcopy(current_formulation)
            persist_feedback_outputs(paths, session)
            print(f"feedback outputs saved to: {folder}")
            print("stopped by user")
            return

        next_formulation = copy.deepcopy(current_formulation)
        applied = False
        apply_note = "not attempted"
        if selected_action not in attempted_actions_in_session:
            attempted_actions_in_session.append(selected_action)
        action_state = action_retry_state.setdefault(
            selected_action,
            {"nonpositive_count": 0, "avoid_target_families": [], "rejected_proposal_signatures": []},
        )
        action_state.setdefault("avoid_target_families", [])
        action_state.setdefault("rejected_proposal_signatures", [])

        if reward == "positive":
            if proposal_obj:
                next_formulation, applied, apply_note = apply_proposal_to_formulation(current_formulation, proposal_obj)
            else:
                applied = False
                apply_note = "proposal parse failed"
            blocked_actions = []
            if applied:
                if selected_action == "solvent_replace":
                    active_solvent_trajectory = update_solvent_trajectory_after_positive(
                        active_solvent_trajectory,
                        current_formulation,
                        proposal_obj,
                        next_formulation,
                    )
                # Stay in this direction next step and escalate to a bolder variant.
                escalation_state = advance_escalation_state(
                    escalation_state,
                    selected_action,
                    state_before_step,
                    proposal_obj,
                )
                current_formulation = next_formulation
                action_retry_state = {}
        else:
            sig = proposal_signature(proposal_obj)
            if decision_bundle.get("selection_path") == "escalate_within_direction":
                # RETRY within escalation instead of abandoning it on the first rejection.
                # The agent is not told the reward's structural rule (e.g. keep one cyclic +
                # one linear), so a wrong-slot / wrong-target guess is expected; give the
                # escalation several attempts (up to the nonpositive threshold), each told to
                # avoid the changes already rejected, before falling back to reranking. This
                # is what lets the ladder survive to stage-1 and then sister-jump to a ratio
                # change. Only exit the escalation once the threshold is hit.
                if isinstance(escalation_state, dict):
                    rejected = list(escalation_state.get("rejected_signatures", []) or [])
                    if sig and sig not in rejected:
                        rejected.append(sig)
                    escalation_state["rejected_signatures"] = rejected
                    rej_changes = list(escalation_state.get("rejected_changes", []) or [])
                    rej_desc = describe_proposed_change(current_formulation, proposal_obj) if proposal_obj else ""
                    if rej_desc and rej_desc not in rej_changes:
                        rej_changes.append(rej_desc)
                    escalation_state["rejected_changes"] = rej_changes
                    level = int(escalation_state.get("level", 1))
                    reject_count = int(escalation_state.get("reject_count", 0)) + 1
                    escalation_state["reject_count"] = reject_count
                else:
                    level = 1
                    reject_count = int(args.nonpositive_threshold)
                if reject_count >= int(args.nonpositive_threshold) and isinstance(escalation_state, dict):
                    # Current sibling is exhausted: HOP to the next un-tried sibling (e.g.
                    # solvent_replace -> solvent_concentration_shift) instead of abandoning the
                    # whole escalation. Only exit once every sibling has been exhausted. This is
                    # what makes the ladder climb one direction to its limit, then sister-jump.
                    tried = list(escalation_state.get("tried_siblings", []) or [])
                    cur = escalation_state.get("current_sibling") or selected_action
                    if cur and cur not in tried:
                        tried.append(cur)
                    next_sibling = next(
                        (s for s in (escalation_state.get("sibling_actions") or [])
                         if s not in tried and s not in blocked_actions),
                        None,
                    )
                    if next_sibling:
                        escalation_state["current_sibling"] = next_sibling
                        escalation_state["tried_siblings"] = tried
                        escalation_state["reject_count"] = 0
                        escalation_state["rejected_changes"] = []
                        escalation_state["rejected_signatures"] = []
                        apply_note = (
                            f"state unchanged; escalation sibling '{cur}' exhausted "
                            f"({reject_count}x); sister-jump to next sibling '{next_sibling}'"
                        )
                    else:
                        escalation_state = None
                        apply_note = (
                            "state unchanged; all escalation siblings exhausted "
                            f"(last '{cur}' rejected {reject_count}x); reranking next step"
                        )
                elif reject_count >= int(args.nonpositive_threshold):
                    escalation_state = None
                    apply_note = (
                        f"state unchanged; escalation '{selected_action}' rejected {reject_count}x; reranking"
                    )
                else:
                    apply_note = (
                        "state unchanged; escalation variant rejected "
                        f"(retry {reject_count}/{int(args.nonpositive_threshold)} at level {level}); "
                        "retrying a different bolder step within the same escalation"
                    )
            elif trajectory_mode and selected_action == "solvent_replace" and trajectory_source:
                if active_solvent_trajectory is None:
                    active_solvent_trajectory = {}
                source_rejected_map = dict((active_solvent_trajectory or {}).get("source_rejected_proposal_signatures", {}) or {})
                source_avoid_map = dict((active_solvent_trajectory or {}).get("source_avoid_target_families", {}) or {})
                source_retry_counts = dict((active_solvent_trajectory or {}).get("source_retry_counts", {}) or {})
                rejected_for_source = list(source_rejected_map.get(trajectory_source, []) or [])
                if sig and sig not in rejected_for_source:
                    rejected_for_source.append(sig)
                source_rejected_map[trajectory_source] = rejected_for_source

                failed_families = infer_failed_target_families(proposal_obj, family_indexes) if proposal_obj else []
                known_avoid = list(source_avoid_map.get(trajectory_source, []) or [])
                for family in failed_families:
                    if family not in known_avoid:
                        known_avoid.append(family)
                source_avoid_map[trajectory_source] = known_avoid
                source_retry_counts[trajectory_source] = int(source_retry_counts.get(trajectory_source, 0) or 0) + 1

                active_solvent_trajectory["source_rejected_proposal_signatures"] = source_rejected_map
                active_solvent_trajectory["source_avoid_target_families"] = source_avoid_map
                active_solvent_trajectory["source_retry_counts"] = source_retry_counts

                if int(source_retry_counts[trajectory_source]) >= int(args.nonpositive_threshold):
                    active_solvent_trajectory = mark_solvent_trajectory_source_exhausted(
                        active_solvent_trajectory,
                        trajectory_source,
                    )
                    if active_solvent_trajectory is None:
                        apply_note = (
                            "state unchanged because reward was not positive; "
                            "exhausted the final remaining solvent in the active solvent trajectory"
                        )
                    else:
                        apply_note = (
                            "state unchanged because reward was not positive; "
                            f"exhausted source solvent {trajectory_source} inside the active solvent trajectory"
                        )
                else:
                    active_solvent_trajectory["current_source"] = trajectory_source
                    apply_note = (
                        "state unchanged because reward was not positive; "
                        f"retry solvent trajectory source {trajectory_source} with avoid_target_families={known_avoid} "
                        f"and rejected_proposal_count={len(rejected_for_source)}"
                    )
            else:
                if sig and sig not in action_state["rejected_proposal_signatures"]:
                    action_state["rejected_proposal_signatures"].append(sig)
                # Rejected changes from THIS formulation, shown to the generator on the next retry.
                # action_retry_state is reset when a change is accepted, so they never carry over.
                rej_desc = describe_proposed_change(current_formulation, proposal_obj) if proposal_obj else ""
                rejected_changes = list(action_state.get("rejected_changes", []) or [])
                if rej_desc and rej_desc not in rejected_changes:
                    rejected_changes.append(rej_desc)
                action_state["rejected_changes"] = rejected_changes

                failed_families = infer_failed_target_families(proposal_obj, family_indexes) if proposal_obj else []
                known_avoid = list(action_state.get("avoid_target_families", []) or [])
                for family in failed_families:
                    if family not in known_avoid:
                        known_avoid.append(family)
                action_state["avoid_target_families"] = known_avoid
                action_state["nonpositive_count"] = int(action_state.get("nonpositive_count", 0)) + 1

                if int(action_state["nonpositive_count"]) >= args.nonpositive_threshold:
                    blocked_actions = [x for x in blocked_actions if x != selected_action] + [selected_action]
                    if current_formulation != session["initial_state"]:
                        next_step_note = "next step will rerank the current formulation and remove already attempted directions from the current MLP top-3"
                    else:
                        next_step_note = "next step will choose the highest-ranked remaining direction from the initial MLP top-3"
                    apply_note = (
                        "state unchanged because reward was not positive; "
                        f"blocked direction after {action_state['nonpositive_count']} nonpositive feedbacks; "
                        + next_step_note
                    )
                else:
                    blocked_actions = [x for x in blocked_actions if x != selected_action]
                    apply_note = (
                        "state unchanged because reward was not positive; "
                        f"retry same direction with avoid_target_families={known_avoid} "
                        f"and rejected_proposal_count={len(action_state['rejected_proposal_signatures'])}"
                    )

        step_record = {
            "step": step_idx,
            "current_state_before": copy.deepcopy(state_payload_from_formulation(state_before_step, objective=None)),
            "selection_mode": str(decision_bundle.get("selection_path") or ("retry_same_direction" if retry_mode else "unknown")),
            "candidate_action_strategies": copy.deepcopy(candidate_actions),
            "selected_action_strategy": selected_action,
            "action_ranking": ranking,
            "selection_bundle": copy.deepcopy(decision_bundle),
            "selected_candidate": copy.deepcopy(selected_candidate),
            "proposal": proposal_obj if proposal_obj else llm_output.get("proposal"),
            "proposal_signature": proposal_signature(proposal_obj),
            "proposal_error": llm_output.get("proposal_error"),
            "user_reward": reward,
            "applied": applied,
            "apply_note": apply_note,
            "retry_state_for_action": copy.deepcopy(action_retry_state.get(selected_action, {})),
            "active_solvent_trajectory": copy.deepcopy(active_solvent_trajectory),
            "blocked_actions": copy.deepcopy(blocked_actions),
            "attempted_actions_in_session": copy.deepcopy(attempted_actions_in_session),
            "current_state_equals_initial": bool(current_formulation == session["initial_state"]),
            "state_after": copy.deepcopy(state_payload_from_formulation(current_formulation, objective=None)),
        }
        session["history"].append(step_record)
        session["final_state"] = copy.deepcopy(current_formulation)
        persist_feedback_outputs(paths, session)

        print("result:")
        print(json.dumps({"reward": reward, "applied": applied, "apply_note": apply_note}, ensure_ascii=False))
        print("current formulation:")
        print(composition_summary(current_formulation))

    persist_feedback_outputs(paths, session)
    print(f"reached max steps ({args.max_steps}), feedback outputs saved to: {folder}")


if __name__ == "__main__":
    main()
