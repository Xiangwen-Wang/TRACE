from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from action_family_rag import (
    generate_action_candidate as generate_action_candidate_via_rag,
)
from literature_rag import (
    ACTION_SPACE,
    build_precedents,
    infer_context_tags_from_formulation,
    load_papers,
    strategy_to_action_type,
    summarize_decision_in_plain_english,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run prior->Action-RAG in one script")
    p.add_argument("--state", type=Path, required=False, help="Input state JSON (required for generation mode)")
    p.add_argument("--prior", type=Path, default=Path("MLP_new/mlp_prior_state_new_v5.pt"), help="Path to trained MLP prior checkpoint (SMILES/structure)")
    p.add_argument("--data", type=Path, default=Path("json_literature.json"), help="Path to literature JSON")
    p.add_argument("--top-k", type=int, default=5, help="Top-k for literature retrieval")
    p.add_argument("--llm-mode", type=str, default="gpt-4.1-mini", help="OpenAI model name, e.g. gpt-5.2 or gpt-4")
    p.add_argument("--llm-temperature", type=float, default=0.0, help="LLM sampling temperature; use 0.0 for maximum stability")
    p.add_argument("--action-override", type=str, default="", help="Optional fixed action_strategy (overrides MLP top-1)")
    p.add_argument("--force-overwrite-memory", action="store_true", help="Ignore memory/cache for this state0, regenerate via LLM, and overwrite memory entry")
    p.add_argument("--override-proposal", type=Path, default=None, help="Path to user override JSON file (feedback mode)")
    p.add_argument("--backup", type=Path, default=Path("prior_suggestion_output.json"), help="Path to write merged output JSON")
    p.add_argument("--out", type=Path, default=Path("prior_suggestion_screen.txt"), help="Path to write screen output log")
    p.add_argument("--cache", type=Path, default=Path(".prior_suggestions_cache.json"), help="Path to local LLM response cache JSON")
    p.add_argument("--memory", type=Path, default=Path(".prior_suggestions_memory.json"), help="Path to persistent feedback/override memory JSON")
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def normalize_additives_map(raw_additives: Any) -> Dict[str, str]:
    default_amount = "1 wt%"
    normalized: Dict[str, str] = {}

    def _put(name: Any, amount: Any) -> None:
        key = str(name).strip() if name is not None else ""
        if not key:
            return
        value = str(amount).strip() if amount is not None else ""
        normalized[key] = value or default_amount

    if isinstance(raw_additives, dict):
        for k, v in raw_additives.items():
            _put(k, v)
        return normalized

    if isinstance(raw_additives, list):
        for item in raw_additives:
            if isinstance(item, dict):
                for k, v in item.items():
                    _put(k, v)
            elif item not in (None, ""):
                _put(item, "")
        return normalized

    if raw_additives in (None, "", {}):
        return normalized

    raw_text = str(raw_additives).strip()
    if not raw_text:
        return normalized

    for token in [x.strip() for x in re.split(r"[,;/|]+", raw_text) if x.strip()]:
        _put(token, "")
    return normalized


def normalize_current_formulation(state_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Accept both start_point-style and decision-style payloads."""
    if isinstance(state_payload.get("current_formulation"), dict):
        base = dict(state_payload["current_formulation"])
    else:
        base = dict(state_payload)

    composition = base.get("composition")
    if not isinstance(composition, dict):
        composition = ((state_payload.get("state0", {}) or {}).get("composition", {}) or {})
    if not isinstance(composition, dict):
        composition = {}
    if not composition:
        raise ValueError("No composition found. Need either current_formulation.composition or state0.composition.")
    base["composition"] = composition

    conditions = base.get("conditions", [])
    if isinstance(conditions, list):
        cleaned = [str(x).strip() for x in conditions if str(x).strip()]
        if cleaned:
            base["conditions"] = cleaned
        else:
            base.pop("conditions", None)
    else:
        base.pop("conditions", None)

    return base


def extract_objective_for_llm(state_payload: Dict[str, Any]) -> Any:
    objective = state_payload.get("objective")
    if isinstance(objective, str):
        text = objective.strip()
        return text if text else None
    if isinstance(objective, dict):
        primary = str(objective.get("primary", "")).strip()
        secondary = objective.get("secondary")
        constraints = objective.get("constraints")
        if primary or secondary not in (None, "", [], {}) or constraints not in (None, "", [], {}):
            return objective
    return None


def llm_ready_formulation(current_formulation: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(current_formulation or {})
    comp = dict((out.get("composition", {}) or {}))
    comp["additives"] = normalize_additives_map(comp.get("additives"))
    out["composition"] = comp
    return out


def mlp_input_example(current_formulation: Dict[str, Any]) -> Dict[str, Any]:
    return {"state0": {"composition": current_formulation.get("composition", {})}}


def normalize_formulation_for_mlp(current_formulation: Dict[str, Any], obs_spec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(current_formulation or {})
    comp = dict((out.get("composition", {}) or {}))

    needs_sws = bool((obs_spec or {}).get("requires_solvents_with_smiles"))
    if needs_sws and not comp.get("solvents_with_smiles"):
        solvents = comp.get("solvents", {}) or {}
        solvent_lookup = (obs_spec or {}).get("solvent_smiles_lookup", {}) or {}
        if isinstance(solvents, dict) and solvents:
            comp["solvents_with_smiles"] = [
                {
                    "name": str(name),
                    "ratio": str(ratio),
                    "smiles": solvent_lookup.get(str(name).strip().lower(), None),
                }
                for name, ratio in solvents.items()
            ]

    out["composition"] = comp
    return out


def get_actions() -> List[str]:
    return list(ACTION_SPACE)


def enabled_actions_from_state(state_payload: Dict[str, Any]) -> List[str]:
    aa = state_payload.get("action_available", {})
    actions = get_actions()
    if not isinstance(aa, dict):
        return actions
    enabled = []
    for action in actions:
        try:
            if float(aa.get(action, 0)) > 0.0:
                enabled.append(action)
        except Exception:
            continue
    return enabled if enabled else actions


STRUCTURE_MODULE_BY_VERSION = {
    "structure_v1": "MLP3_smiles.py",
    "structure_v2_descscale": "MLP3_smiles.py",
    "structure_v3_set": "MLP3_smiles.py",
    "structure_v4": "MLP3_smiles.py",
    "structure_v5": "MLP3_smiles.py",
}


def _load_structure_mlp_module(feature_version: str = "structure_v1"):

    import importlib.util

    filename = STRUCTURE_MODULE_BY_VERSION.get(feature_version)
    if filename is None:
        raise ValueError(f"Unknown structure feature_version in checkpoint: {feature_version!r}")
    here = Path(__file__).resolve().parent
    candidates = [
        here / "MLP_new" / filename,
        here / filename,
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("mlp_smiles_prior", str(path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        f"Could not find the structure (SMILES) MLP module ({filename}). Tried: "
        + ", ".join(str(p) for p in candidates)
    )


COMPONENT_SMILES_LOOKUP_PATH = Path(__file__).resolve().parent / "supplementary" / "component_smiles_lookup.json"

_WARNED_MISSING_SMILES: set = set()


def merge_live_smiles_lookup(obs_spec: Dict[str, Any], mlp_mod, lookup_path: Path = COMPONENT_SMILES_LOOKUP_PATH) -> int:

    path = Path(lookup_path)
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0

    salt_lookup = obs_spec.setdefault("salt_smiles_lookup", {})
    solvent_lookup = obs_spec.setdefault("solvent_smiles_lookup", {})
    additive_lookup = obs_spec.setdefault("additive_smiles_lookup", {})


    generic: Dict[str, str] = {}
    salt_new: Dict[str, str] = {}
    solvent_new: Dict[str, str] = {}
    additive_new: Dict[str, str] = {}
    mlp_mod.register_component_lookup_data(data, salt_new, solvent_new, additive_new, generic)

    applied = 0
    for target, specific in (
        (salt_lookup, salt_new),
        (solvent_lookup, solvent_new),
        (additive_lookup, additive_new),
    ):
        for key, smiles in generic.items():
            target[key] = smiles
            applied += 1
        for key, smiles in specific.items():
            target[key] = smiles
            applied += 1
    return applied


def missing_smiles_components(ex: Dict[str, Any], obs_spec: Dict[str, Any], mlp_mod) -> List[Tuple[str, str]]:

    comp = mlp_mod.get_comp(ex)
    salt_lookup = obs_spec.get("salt_smiles_lookup", {}) or {}
    solvent_lookup = obs_spec.get("solvent_smiles_lookup", {}) or {}
    additive_lookup = obs_spec.get("additive_smiles_lookup", {}) or {}

    missing: List[Tuple[str, str]] = []
    for name in mlp_mod.split_known_component_names(comp.get("salt", "")):
        if not mlp_mod.lookup_smiles(name, salt_lookup):
            missing.append(("salt", name))
    for name in mlp_mod.collect_solvent_names(ex):
        if mlp_mod.is_placeholder_component_name(name):
            continue
        if not mlp_mod.lookup_smiles(name, solvent_lookup):
            missing.append(("solvent", name))
    for name in mlp_mod.collect_additive_names(ex):
        if mlp_mod.is_placeholder_component_name(name):
            continue
        if not mlp_mod.lookup_smiles(name, additive_lookup):
            missing.append(("additive", name))
    return missing


def warn_missing_smiles(missing: List[Tuple[str, str]], lookup_path: Path = COMPONENT_SMILES_LOOKUP_PATH) -> None:

    fresh = [(t, n) for (t, n) in missing if (t, n) not in _WARNED_MISSING_SMILES]
    if not fresh:
        return
    print(
        f"[SMILES missing] {len(fresh)} component(s) have no SMILES in {lookup_path.name}; "
        "they are encoded as zero vectors. Add them to keep the structure prior accurate:"
    )
    for comp_type, name in fresh:
        print(f"  - {comp_type}: {name!r}")
        _WARNED_MISSING_SMILES.add((comp_type, name))


def infer_action_with_mlp(
    state_payload: Dict[str, Any],
    current_formulation: Dict[str, Any],
    prior_ckpt_path: Path,
) -> Tuple[str, List[Dict[str, float]]]:
    import numpy as np  # lazy import for feedback-only mode
    import torch

    ckpt = torch.load(prior_ckpt_path, map_location="cpu")
    obs_spec = ckpt["obs_spec"]

    mlp_mod = _load_structure_mlp_module(str(obs_spec.get("feature_version") or "structure_v1"))
    MLP, encode_obs = mlp_mod.MLP, mlp_mod.encode_obs
    actions = list(ckpt.get("actions") or list(mlp_mod.ACTIONS))
    model = (mlp_mod.build_model(obs_spec) if hasattr(mlp_mod, "build_model")
             else MLP(in_dim=int(obs_spec["in_dim"])))

    merge_live_smiles_lookup(obs_spec, mlp_mod)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    mlp_formulation = normalize_formulation_for_mlp(current_formulation, obs_spec)
    ex = mlp_input_example(mlp_formulation)

    if state_payload.get("objective") is not None:
        ex["objective"] = state_payload.get("objective")
    warn_missing_smiles(missing_smiles_components(ex, obs_spec, mlp_mod))
    x = encode_obs(ex, obs_spec).astype(np.float32)

    with torch.no_grad():
        logits = model(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

    action_rows = []
    for i, action in enumerate(actions):
        action_rows.append(
            {
                "action": action,
                "logit": float(logits[i]),
                "prob": float(probs[i]),
            }
        )
    action_rows.sort(key=lambda r: r["prob"], reverse=True)

    enabled = set(enabled_actions_from_state(state_payload))
    for row in action_rows:
        if row["action"] in enabled:
            return row["action"], action_rows
    return action_rows[0]["action"], action_rows


def top_enabled_actions(action_ranking: List[Dict[str, float]], state_payload: Dict[str, Any], n: int = 3) -> List[str]:
    enabled = set(enabled_actions_from_state(state_payload))
    picks: List[str] = []
    for row in action_ranking:
        action = str(row.get("action", ""))
        if action and action in enabled and action not in picks:
            picks.append(action)
        if len(picks) >= n:
            break
    if not picks and action_ranking:
        picks.append(str(action_ranking[0].get("action", "unknown_action")))
    return picks


def action_ranking_from_override(action_override: str, state_payload: Dict[str, Any]) -> List[Dict[str, float]]:
    enabled = enabled_actions_from_state(state_payload)
    ranking: List[Dict[str, float]] = []
    if action_override:
        ranking.append({"action": action_override, "logit": 1.0, "prob": 1.0})
    for action in enabled:
        if action != action_override:
            ranking.append({"action": action, "logit": 0.0, "prob": 0.0})
    if not ranking and action_override:
        ranking.append({"action": action_override, "logit": 1.0, "prob": 1.0})
    return ranking


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


def summarize_state0_composition(current_formulation: Dict[str, Any]) -> str:
    comp = (current_formulation or {}).get("composition", {}) or {}
    salt = str(comp.get("salt", "")).strip() or "N/A"
    salt_conc = str(comp.get("salt_concentration", "")).strip() or "N/A"
    solvents = comp.get("solvents", {}) or {}
    additives = comp.get("additives", {}) or {}

    if isinstance(solvents, dict) and solvents:
        solvent_text = ", ".join(f"{k}:{v}" for k, v in solvents.items())
    else:
        solvent_text = "None"
    if isinstance(additives, dict) and additives:
        additive_text = ", ".join(f"{k}:{v}" for k, v in additives.items())
    else:
        additive_text = "None"

    return (
        f"state0 composition summary : salt={salt}; "
        f"salt_concentration={salt_conc}; "
        f"solvents={solvent_text}; additives={additive_text}"
    )


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


def print_proposal_from_llm(llm_output_payload: Dict[str, Any], emit) -> None:
    proposal_obj = parse_proposal_json(llm_output_payload.get("proposal"))
    if not proposal_obj:
        err = llm_output_payload.get("proposal_error")
        if err:
            emit(json.dumps({"proposal_error": err}, ensure_ascii=False))
        else:
            emit(json.dumps({"proposal_error": "LLM proposal not available."}, ensure_ascii=False))
        return

    action_type = proposal_obj.get("action_type", "unknown_action")
    params = proposal_obj.get("parameters", {})
    if not isinstance(params, dict):
        params = {}
    rationale = proposal_obj.get("rationale", "")

    emit(json.dumps({"action_type": action_type}, ensure_ascii=False))
    emit(json.dumps({"parameters": params}, ensure_ascii=False))
    if rationale:
        emit(json.dumps({"rationale": rationale}, ensure_ascii=False))


def parse_selector_json(text: Any) -> Dict[str, Any]:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return {}
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _temp_kwargs(model: Any, temperature: float) -> Dict[str, Any]:

    m = str(model or "").lower()
    if m.startswith(("gpt-5", "o1", "o3", "o4")):
        return {}
    return {"temperature": temperature}


def generate_final_choice_with_openai(
    selector_input: Dict[str, Any],
    model: str = "gpt-4.1-mini",
    temperature: float = 0.0,
) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a strict proposal selector. Given candidate proposals for the same state, pick exactly one.\n"
        "Always select one candidate unless all candidates have proposal_error or are completely unusable.\n\n"
        "Decide by COMPARING the candidates against each other on the SAME axis: expected improvement on the "
        "stated objective for this state. Do NOT treat the candidates as independently 'all plausible' — they are "
        "competing, and your job is to find what makes one better than the others.\n\n"
        "Important:\n"
        "- No action-type prior or popularity score is provided, and you must NOT assume a more common or more "
        "conventional action type is better. Judge each proposal on its concrete expected effect on the objective.\n"
        "- Be skeptical of each candidate's own rationale — every candidate argues for itself. Weigh the shared, "
        "comparable evidence (rag_summary: precedent polarity/similarity, action risk, explicitly-explored signals) "
        "rather than the persuasiveness of the self-description.\n"
        "- If two candidates look equally useful on the objective, prefer the one with stronger objective-aligned "
        "precedent support; then lower risk_level / uncertainty; then clearer, more specific parameters.\n\n"
        "In the rationale you MUST explicitly state why the chosen candidate is expected to beat EACH of the other "
        "candidates on the objective (one clause per rejected candidate). If you cannot articulate a real "
        "discriminator, say so and fall back to the evidence-based tie-breakers above.\n\n"
        "Output strict JSON with keys: selected_index, selected_action_strategy, rationale."
    )
    user_prompt = (
        "Choose one best candidate from the following JSON input by comparing them on expected improvement on the "
        "objective. selected_index must be one of the candidate indices shown in SELECTOR_INPUT.\n"
        "In rationale, compare the winner against each other candidate explicitly.\n\n"
        f"SELECTOR_INPUT:\n{json.dumps(selector_input, ensure_ascii=False)}"
    )
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **_temp_kwargs(model, max(0.0, min(float(temperature), 1.0))),
    )
    return (resp.output_text or "").strip()


def load_cache(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def save_cache(path: Path, cache: Dict[str, str]) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


SELECTOR_CACHE_VERSION = "selector_cache_v3_comparative_no_prior"


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
        "selector_cache_version": SELECTOR_CACHE_VERSION,
        "model": model,
        "temperature": float(temperature),
        "llm_input": llm_input_payload,
        "constraint_context": _extract_cache_constraint_context(llm_input_payload),
    }
    raw = json.dumps(key_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_memory(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"entries": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"entries": []}
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        return data
    if isinstance(data, list):
        return {"entries": data}
    return {"entries": []}


def save_memory(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_selector_candidate_rag_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    hard_constraints = payload.get("hard_constraints", {}) or {}
    action_info = (hard_constraints.get("action_type", {}) or {}) if isinstance(hard_constraints, dict) else {}
    precedents_compact = payload.get("precedents_compact", []) or []

    positive_count = 0
    negative_count = 0
    mixed_count = 0
    explicitly_explored_count = 0
    precedent_head: List[Dict[str, Any]] = []

    for item in precedents_compact:
        if not isinstance(item, dict):
            continue
        pol = str(item.get("polarity", "mixed/uncertain"))
        if pol == "positive":
            positive_count += 1
        elif pol == "negative":
            negative_count += 1
        else:
            mixed_count += 1
        if bool(item.get("explicitly_explored")):
            explicitly_explored_count += 1

    for item in precedents_compact[:5]:
        if not isinstance(item, dict):
            continue
        precedent_head.append(
            {
                "precedent_id": item.get("precedent_id"),
                "similarity_score": item.get("similarity_score"),
                "polarity": item.get("polarity"),
                "property": item.get("property"),
                "conditions_flags": item.get("conditions_flags"),
                "explicitly_explored": item.get("explicitly_explored"),
                "primary_objective_tag": item.get("primary_objective_tag"),
            }
        )

    return {
        "action_risk": {
            "name": action_info.get("name"),
            "risk_level": action_info.get("risk_level"),
            "positive_ratio": action_info.get("positive_ratio"),
            "negative_ratio": action_info.get("negative_ratio"),
            "explicitly_explored_ratio": action_info.get("explicitly_explored_ratio"),
        },
        "precedent_signal_summary": {
            "retrieved_precedent_count": len(precedents_compact),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "mixed_or_uncertain_count": mixed_count,
            "explicitly_explored_count": explicitly_explored_count,
        },
        "precedent_head": precedent_head,
    }


def normalize_component_for_state_key(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_component_for_state_key(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, list):
        return [normalize_component_for_state_key(v) for v in value]
    if value is None:
        return ""
    return str(value).strip()


def normalize_state0_for_memory(state0: Dict[str, Any]) -> Dict[str, Any]:
    comp = ((state0 or {}).get("state0", {}) or {}).get("composition", {}) or {}
    if not isinstance(comp, dict):
        comp = {}
    solvents = comp.get("solvents")
    if not isinstance(solvents, dict):
        solvents = {}
    if not solvents:
        sws = comp.get("solvents_with_smiles", [])
        if isinstance(sws, list):
            rebuilt = {}
            for item in sws:
                if isinstance(item, dict):
                    name = str(item.get("name", "")).strip()
                    ratio = str(item.get("ratio", "")).strip()
                    if name:
                        rebuilt[name] = ratio
            solvents = rebuilt
    state_obj = {
        "state0": {
            "composition": {
                "salt": comp.get("salt", ""),
                "salt_concentration": comp.get("salt_concentration", ""),
                "solvents": solvents,
                "additives": comp.get("additives", {}) or {},
            }
        }
    }
    return normalize_component_for_state_key(state_obj)


def state0_key(state0: Dict[str, Any]) -> str:
    normalized = normalize_state0_for_memory(state0)
    raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def find_memory_entry(memory: Dict[str, Any], state0: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    entries = memory.get("entries", [])
    if not isinstance(entries, list):
        return -1, {}
    key = state0_key(state0)
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        entry_state0 = entry.get("state0", {})
        if not isinstance(entry_state0, dict):
            continue
        wrapped_entry_state0 = entry_state0 if "state0" in entry_state0 else {"state0": entry_state0}
        if state0_key(wrapped_entry_state0) == key:
            return i, entry
    return -1, {}


def upsert_memory_entry(memory: Dict[str, Any], state0: Dict[str, Any], proposal: Any, source: str) -> None:
    entries = memory.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    idx, old = find_memory_entry(memory, state0)
    normalized_wrapper = normalize_state0_for_memory(state0)
    record = {
        "state0": normalized_wrapper.get("state0", {}),
        "proposal": proposal,
        "source": source,
    }
    if isinstance(old, dict):
        if "notes" in old:
            record["notes"] = old["notes"]
    if idx >= 0:
        entries[idx] = record
    else:
        entries.append(record)
    memory["entries"] = entries


def parse_override_proposal(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return {"raw_text": text}


def load_override_entries(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    obj: Any
    try:
        obj = json.loads(raw)
    except Exception:
        lines = [x.strip() for x in raw.splitlines() if x.strip()]
        merged: Dict[str, Any] = {}
        for line in lines:
            try:
                one = json.loads(line)
            except Exception:
                continue
            if isinstance(one, dict):
                merged.update(one)
        obj = merged if merged else []

    entries: List[Dict[str, Any]] = []
    if isinstance(obj, dict) and isinstance(obj.get("entries"), list):
        for item in obj["entries"]:
            if isinstance(item, dict):
                entries.append(item)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                entries.append(item)
    elif isinstance(obj, dict):
        entries.append(obj)
    return entries


def entry_to_state_and_proposal(entry: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    state0 = entry.get("state0", {})
    if not isinstance(state0, dict):
        state0 = {}
    if "composition" in state0:
        state0 = {"state0": state0}

    if "proposal" in entry:
        proposal = entry.get("proposal")
    else:
        proposal = {
            "action_type": entry.get("action_type", "unknown_action"),
            "parameters": entry.get("parameters", {}),
            "rationale": entry.get("rationale", ""),
        }
    return state0, proposal


def main() -> None:
    args = parse_args()
    screen_lines: List[str] = []

    def emit(line: str) -> None:
        text = str(line)
        print(text)
        screen_lines.append(text)

    if args.override_proposal is not None and args.state is None:
        memory = load_memory(args.memory)
        override_entries = load_override_entries(args.override_proposal)
        applied = 0
        for item in override_entries:
            state0, proposal = entry_to_state_and_proposal(item)
            if not state0 or not ((state0.get("state0", {}) or {}).get("composition", {})):
                continue
            upsert_memory_entry(memory, state0=state0, proposal=proposal, source="user_override")
            applied += 1
        save_memory(args.memory, memory)
        print(f"Updated memory: {args.memory} (applied overrides: {applied})")
        return

    if args.state is None:
        raise ValueError("--state is required for generation mode, or use --override-proposal with --memory for feedback mode.")

    state_payload = load_json(args.state)
    current_formulation = normalize_current_formulation(state_payload)
    current_formulation_for_llm = llm_ready_formulation(current_formulation)
    current_state0 = {"state0": {"composition": current_formulation.get("composition", {})}}
    memory = load_memory(args.memory)
    cache = load_cache(args.cache)
    _, memory_entry = find_memory_entry(memory, current_state0)

    if args.action_override:
        _selected_action = args.action_override
        action_ranking = action_ranking_from_override(args.action_override, state_payload)
    else:
        _selected_action, action_ranking = infer_action_with_mlp(
            state_payload=state_payload,
            current_formulation=current_formulation,
            prior_ckpt_path=args.prior,
        )
    decision_bundle: Dict[str, Any]
    proposal_source = "llm_top3_selector"

    if (not args.force_overwrite_memory) and isinstance(memory_entry, dict) and "proposal" in memory_entry:
        stored = memory_entry.get("proposal")
        if isinstance(stored, dict) and stored.get("schema") == "top3_v1":
            decision_bundle = stored
            proposal_source = "memory_override"
        else:
            fallback_action = str(action_ranking[0].get("action", "unknown_action")) if action_ranking else "unknown_action"
            decision_bundle = {
                "schema": "top3_v1",
                "selected_action_strategy": fallback_action,
                "selected_candidate_index": 0,
                "selector_output": {
                    "selected_index": 0,
                    "selected_action_strategy": fallback_action,
                    "rationale": "Loaded from legacy memory entry.",
                },
                "proposal_source": "memory_override_legacy",
                "candidates": [
                    {
                        "index": 0,
                        "action_strategy": fallback_action,
                        "proposal_source": "memory_override_legacy",
                        "payload": {},
                        "llm_output": {
                            "model": args.llm_mode,
                            "action_strategy": fallback_action,
                            "proposal": stored,
                            "proposal_error": None,
                        },
                    }
                ],
            }
            proposal_source = "memory_override_legacy"
    else:
        top_actions = top_enabled_actions(action_ranking, state_payload, n=3)
        if args.action_override:
            if args.action_override not in ACTION_SPACE:
                raise ValueError(f"--action-override must be one of {ACTION_SPACE}")
            top_actions = [args.action_override]

        papers = load_papers(args.data)
        all_precedents = build_precedents(papers)
        state_objective_for_llm = extract_objective_for_llm(state_payload)
        candidates: List[Dict[str, Any]] = []

        for idx, action_strategy in enumerate(top_actions):
            candidate_bundle = generate_action_candidate_via_rag(
                current_formulation=current_formulation_for_llm,
                action_strategy=action_strategy,
                state_objective=state_objective_for_llm,
                papers=papers,
                top_k=args.top_k,
                llm_mode=args.llm_mode,
                llm_temperature=args.llm_temperature,
                cache=cache,
                force_refresh=args.force_overwrite_memory,
                all_precedents=all_precedents,
            )
            if candidate_bundle.get("cache_updated") and candidate_bundle.get("cache_key") and candidate_bundle.get("cache_value") is not None:
                cache[str(candidate_bundle["cache_key"])] = str(candidate_bundle["cache_value"])
                save_cache(args.cache, cache)
            candidates.append(
                {
                    "index": idx,
                    "action_strategy": candidate_bundle.get("action_strategy", action_strategy),
                    "proposal_source": candidate_bundle.get("proposal_source", "unknown"),
                    "payload": candidate_bundle.get("payload", {}),
                    "llm_output": candidate_bundle.get("llm_output", {}),
                }
            )

        selector_input = {
            "state0_composition": current_formulation_for_llm.get("composition", {}),
            "context_tags": infer_context_tags_from_formulation(current_formulation_for_llm),
            "candidates": [],
        }
        for c in candidates:
            proposal_value = (c.get("llm_output", {}) or {}).get("proposal")
            proposal_obj = parse_proposal_json((c.get("llm_output", {}) or {}).get("proposal"))
            selector_input["candidates"].append(
                {
                    "index": c.get("index"),
                    "action_strategy": c.get("action_strategy"),
                    "action_type": (c.get("llm_output", {}) or {}).get("action_type"),
                    "proposal": proposal_obj if proposal_obj else proposal_value,
                    "proposal_error": (c.get("llm_output", {}) or {}).get("proposal_error"),
                    "rag_summary": build_selector_candidate_rag_summary(c.get("payload", {}) or {}),
                }
            )

        selector_cache_key = build_cache_key(
            {"selector_input": selector_input},
            model=args.llm_mode,
            temperature=args.llm_temperature,
        )
        selector_cached = cache.get(selector_cache_key)
        selector_output_obj: Dict[str, Any] = {}
        selector_output_text: str = ""
        if selector_cached is not None and not args.force_overwrite_memory:
            selector_output_text = selector_cached
            selector_output_obj = parse_selector_json(selector_output_text)
        else:
            try:
                selector_output_text = generate_final_choice_with_openai(
                    selector_input=selector_input,
                    model=args.llm_mode,
                    temperature=args.llm_temperature,
                )
                selector_output_obj = parse_selector_json(selector_output_text)
                cache[selector_cache_key] = selector_output_text
                save_cache(args.cache, cache)
            except Exception as e:
                selector_output_obj = {"selected_index": 0, "rationale": f"selector_failed: {e}"}

        selected_idx_raw = selector_output_obj.get("selected_index", 0)
        try:
            selected_idx = int(selected_idx_raw)
        except Exception:
            selected_idx = 0
        if selected_idx < 0 or selected_idx >= len(candidates):
            selected_idx = 0

        selected_action_strategy = str(candidates[selected_idx].get("action_strategy", "unknown_action"))
        decision_bundle = {
            "schema": "top3_v1",
            "selected_action_strategy": selected_action_strategy,
            "selected_candidate_index": selected_idx,
            "selector_output": selector_output_obj if selector_output_obj else selector_output_text,
            "proposal_source": "llm_top3_selector",
            "candidates": candidates,
        }
        upsert_memory_entry(memory, state0=current_state0, proposal=decision_bundle, source="llm_top3_selector")
        save_memory(args.memory, memory)
        proposal_source = "llm_top3_selector"

    candidates = decision_bundle.get("candidates", []) if isinstance(decision_bundle, dict) else []
    selected_idx = int(decision_bundle.get("selected_candidate_index", 0)) if isinstance(decision_bundle, dict) else 0
    if not candidates:
        raise RuntimeError("No candidates available in decision bundle.")
    if selected_idx < 0 or selected_idx >= len(candidates):
        selected_idx = 0
    chosen = candidates[selected_idx]

    candidate_llm_inputs = []
    for c in candidates:
        payload = c.get("payload", {}) or {}
        candidate_llm_inputs.append(
            {
                "index": c.get("index"),
                "action_strategy": c.get("action_strategy"),
                "rag_information": ((payload.get("llm_input", {}) or {}).get("rag_information", {}) or {}),
            }
        )

    chosen_payload = chosen.get("payload", {}) or {}
    backup_payload = {
        "selected_action_strategy": decision_bundle.get("selected_action_strategy", chosen.get("action_strategy")),
        "selected_candidate_index": selected_idx,
        "mlp_action_ranking": action_ranking,
        "chosen_rag_information": ((chosen_payload.get("llm_input", {}) or {}).get("rag_information", {}) or {}),
        "candidate_llm_inputs": candidate_llm_inputs,
        "chosen_llm_output": chosen.get("llm_output", {}),
    }
    args.backup.write_text(json.dumps(backup_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    emit(summarize_state0_composition(current_formulation_for_llm))
    emit(f"Top candidate actions from agent: {[c.get('action_strategy') for c in candidates]}")
    emit(json.dumps({"proposal_source": proposal_source}, ensure_ascii=False))
    emit("candidate suggestions:")
    for c in candidates:
        emit(f"candidate_index={c.get('index')} action_strategy={c.get('action_strategy')}")
        emit(json.dumps({"candidate_proposal_source": c.get("proposal_source", "unknown")}, ensure_ascii=False))
        print_proposal_from_llm(c.get("llm_output", {}) or {}, emit=emit)
    emit("final selection:")
    emit(
        json.dumps(
            {
                "selected_candidate_index": selected_idx,
                "selected_action_strategy": decision_bundle.get("selected_action_strategy"),
                "selector_output": decision_bundle.get("selector_output"),
            },
            ensure_ascii=False,
        )
    )
    emit("final suggestions:")
    print_proposal_from_llm(chosen.get("llm_output", {}) or {}, emit=emit)

    args.out.write_text("\n".join(screen_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
