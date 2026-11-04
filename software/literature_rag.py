#!/usr/bin/env python3
"""RAG over literature_database.json for formulation optimization.

Modes:
1) QA mode (legacy): retrieve evidence and optionally call LLM.
2) Action-RAG mode: given current formulation F_t + action_type, return
   structured guidance for downstream LLM:
   - hard_constraints
   - precedents (before -> after)
   - tendencies
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Set, Tuple
import warnings

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_+\-\.]+")
POSITIVE_HINTS = [
    "improved", "higher", "enhanced", "better", "stable", "uniform", "outperformed",
    "highest", "maintains", "maintained", "lower hysteresis", "lower residual",
    "minimal impedance growth", "no short-circuit", "improved retention", "suppressed",
]
NEGATIVE_HINTS = [
    "poor", "worse", "inferior", "unstable", "capacity fade", "short-circuit",
    "rapid decay", "lower coulombic efficiency", "higher residual oxidation current",
    "increased impedance", "degraded", "dendritic", "fracture",
]
ALLOWED_OBJECTIVES = {
    "ion_transport_temperature_dependence",
    "viscosity_transport_coupling",
    "solvation_association_temperature_stability",
    "unspecified_performance_objective",
}
OBJECTIVE_TAGS = {
    "ion_transport_temperature_dependence": [
        "ionic conductivity",
        "conductivity",
        "arrhenius",
        "vtf",
        "activation energy",
        "self-diffusion",
        "diffusion coefficient",
        "rate capability",
        "low temperature",
        "wide temperature",
    ],
    "viscosity_transport_coupling": [
        "viscosity",
        "walden",
        "green-kubo",
        "fluidity",
        "mass transport",
        "polarization",
    ],
    "solvation_association_temperature_stability": [
        "solvation",
        "coordination",
        "cip",
        "ssip",
        "agg",
        "ion pair",
        "cluster",
        "sei",
        "cei",
        "interfacial",
        "raman",
        "nmr",
    ],
}
ACTION_SPACE = [
    "salt_replace",
    "salt_concentration_shift",
    "add_new_salt",
    "solvent_replace",
    "solvent_concentration_shift",
    "add_new_solvent",
    "remove_solvent",
    "additive_change",
]
ACTION_SPACE_WITH_UNKNOWN = ACTION_SPACE + ["unknown_action"]
SMALL_CHANGE_THRESHOLD = 0.05

_EMBED_MODEL = None
_EMBED_MODEL_NAME = None
RAG_INDEX_CACHE_VERSION = "formulation_level_v1"


PROPOSAL_SYSTEM_PROMPT = (
    "You are a formulation design assistant. "
    "Use the provided literature_rag information as primary prior knowledge.\n"
    "Use retrieved precedents as guidance. Treat explicit exclusion constraints in rag_information as mandatory.\n\n"
    "Hard proposal rules (must follow):\n"
    "- Propose exactly ONE allowed action (one-step) consistent with the allowed action_type.\n"
    "- Do not change multiple major knobs at once (salt identity, salt concentration, solvent set/ratio, additive package).\n"
    "- Within the single allowed action, choose a change that is practical and likely to be informative for the objective "
    "(avoid changes so tiny that they are unlikely to produce a measurable effect).\n"
    "- If uncertainty is high or evidence is weak, keep the change modest and clearly state uncertainty.\n"
    "- Use explicit full material names where possible; avoid ambiguous references like 'it' or 'the solvent'.\n\n"
    "You are allowed to:\n"
    "- extrapolate cautiously beyond exact precedents,\n"
    "- as long as the proposal stays within the allowed action_type\n"
    "- and you explicitly describe the uncertainty or risk.\n\n"
    "Grounding requirements:\n"
    "- Prioritize rag_information in LLM_INPUT when making decisions.\n"
    "- Do not claim fabricated literature evidence; if extrapolating, label it as extrapolation.\n"
    "- supporting_precedents may be an empty list; explicit citation is optional.\n"
    "- If rag_information.family_guidance.constraint_mode is hard_allowlist, treat allowed_target_families as a hard constraint on any new solvent/salt/additive you propose.\n"
    "- If rag_information.family_guidance.constraint_mode is advisory_from_rag, use target_family_statistics and observed_family_transitions as prior evidence rather than a mandatory rule.\n"
    "- Do not choose a new solvent/salt/additive from rag_information.family_guidance.avoid_target_families.\n"
    "- avoid_target_families is a mandatory exclusion list even when constraint_mode is advisory_from_rag.\n"
    "- Before writing the final JSON, verify that every newly introduced solvent/salt/additive is outside avoid_target_families.\n"
    "- If the currently most obvious target is forbidden, choose a different target outside the forbidden families; do not return a remove-only or empty change.\n"
    "- Read rag_information.action_conditioned_examples.positive_patterns in this priority order: exact_state_matches first, then secondary_same_family, then other_similar_matches. Each entry is one complete literature record with before/after compositions, change, property, outcome, and conditions.\n"
    "- Read rag_information.action_conditioned_examples.negative_patterns in the same priority order to understand which complete literature records showed failure, degradation, or trade-offs.\n"
    "- If rag_information.action_conditioned_examples.forbidden_reversion_patterns is present, do not propose a reverse move that returns the current formulation to an older baseline already traversed in literature, even if that old baseline looked strong.\n"
    "- If LLM_INPUT.positive_feedback_trajectory is present, treat it as accepted in-session feedback from the current optimization run.\n"
    "- If LLM_INPUT.positive_feedback_trajectory.mode is continue_solvent_replace_trajectory, continue that trajectory conservatively: replace exactly one remaining source solvent, and do not modify completed or exhausted source solvents.\n"
    "- If evidence is weak, keep the proposal modest and explicitly raise uncertainty.\n\n"
    "Structural reasoning with SMILES (important):\n"
    "- Abbreviations and names are unreliable for structure. To judge a molecule's structure correctly — "
    "its degree of fluorination (the NUMBER of fluorine atoms) and whether it is a ring/cyclic carbonate or a "
    "chain/linear carbonate — you must FIND and reason from its SMILES, not from its name.\n"
    "- Before proposing, determine the SMILES of every solvent currently in the formulation AND of each candidate "
    "molecule you are considering, and use those SMILES to decide the change.\n"
    "- Include a 'smiles' field for every solvent you propose (your best structural SMILES for it).\n"
    "- If your intent is to increase fluorination, the substituted-in solvent MUST contain MORE fluorine atoms than "
    "the one it replaces (compare F counts from the SMILES). Do NOT propose a change that keeps the fluorine count "
    "the same or lower when the goal is to fluorinate.\n\n"
    "Action-format requirements:\n"
    "- For solvent_replace, parameters must contain one concrete non-empty replacement target. Valid examples are replacements=[{from,to}] or from_solvent+to_solvent.\n"
    "- For solvent_replace, do not output an empty resulting_solvents object, a remove-only change, or a target whose amount is effectively zero.\n"
    "- For add_new_solvent, name at least one concrete new solvent outside avoid_target_families.\n"
    "- For solvent_concentration_shift, express every solvent fraction on a 10% grid (multiples of 0.1, e.g. 0.5/0.5, 0.6/0.4, 0.7/0.3) and change the ratio by exactly ONE 10% step from the current one (e.g. 5:5 -> 6:4). Do not propose fractions like 0.65 or 0.73.\n\n"
    "You may propose new entities if they belong to known chemical families in the provided family tables, and you must mark them as extrapolation."
)

PROPOSAL_USER_PROMPT_PREFIX = (
    "Given full formulation, objective, RL action strategy, and retrieved priors, "
    "propose ONE single-step modification consistent with allowed action_type.\n"
    "This must be exactly one action and must not change multiple major knobs at once.\n"
    "Choose a change that is practical and likely to produce a measurable effect on the objective, "
    "while staying consistent with hard constraints and retrieved priors.\n"
    "If positive_feedback_trajectory is present in LLM_INPUT, use it as accepted interactive-session feedback when choosing the next one-step modification.\n"
    "Output strict JSON with keys: action_type, parameters, rationale, supporting_precedents, "
    "is_exploratory, uncertainty_level.\n"
    "Keep the change focused. Avoid aggressive ratio jumps, concentration jumps, or multiple simultaneous major edits.\n\n"
    "LLM_INPUT:\n"
)


@dataclass
class Chunk:
    chunk_id: str
    paper_title: str
    doi: str
    formulation_id: str
    role: str
    property_name: str
    text: str


@dataclass
class Precedent:
    precedent_id: str
    paper_title: str
    doi: str
    before_id: str
    after_id: str
    before_comp: Dict[str, Any]
    after_comp: Dict[str, Any]
    action_types: List[str]
    property_name: str
    conditions: List[str]
    outcome: str
    polarity: str
    after_role: str
    explored_dimensions: List[str]
    action_explicitly_explored: bool
    objective_tags: List[str]
    rl_action_type: str
    evaluated_properties: List[Dict[str, Any]]


@dataclass
class DecisionQuery:
    current_formulation: Dict[str, Any]
    action_strategy: str
    action_type: str
    target_property: str
    context_tags: List[str]
    objective_primary: str


class TfIdfRetriever:
    """Minimal sparse TF-IDF retriever implemented with standard library only."""

    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.vectors: List[Dict[str, float]] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return TOKEN_PATTERN.findall(text.lower())

    @staticmethod
    def _l2_normalize(vec: Dict[str, float]) -> Dict[str, float]:
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm == 0:
            return vec
        return {k: v / norm for k, v in vec.items()}

    def _to_tfidf(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        tf = Counter(tokens)
        length = len(tokens)
        vec = {}
        for token, count in tf.items():
            idf = self.idf.get(token)
            if idf is None:
                continue
            vec[token] = (count / length) * idf
        return self._l2_normalize(vec)

    def fit(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self.df = defaultdict(int)

        tokenized_docs: List[List[str]] = []
        for c in chunks:
            tokens = self._tokenize(c.text)
            tokenized_docs.append(tokens)
            for t in set(tokens):
                self.df[t] += 1

        n_docs = max(len(chunks), 1)
        self.idf = {
            t: math.log((1 + n_docs) / (1 + df_t)) + 1.0
            for t, df_t in self.df.items()
        }

        self.vectors = []
        for tokens in tokenized_docs:
            tf = Counter(tokens)
            length = max(len(tokens), 1)
            vec = {}
            for token, count in tf.items():
                vec[token] = (count / length) * self.idf[token]
            self.vectors.append(self._l2_normalize(vec))

    @staticmethod
    def _dot(a: Dict[str, float], b: Dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(k, 0.0) for k, v in a.items())

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Chunk]]:
        q_vec = self._to_tfidf(query)
        if not q_vec:
            return []

        scored: List[Tuple[float, Chunk]] = []
        for vec, chunk in zip(self.vectors, self.chunks):
            score = self._dot(q_vec, vec)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def save(self, path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "chunks": [asdict(c) for c in self.chunks],
            "df": self.df,
            "idf": self.idf,
            "vectors": self.vectors,
            "metadata": metadata or {},
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    def load(self, path: Path) -> Dict[str, Any]:
        with path.open("rb") as f:
            payload = pickle.load(f)
        self.chunks = [Chunk(**c) for c in payload["chunks"]]
        self.df = dict(payload["df"])
        self.idf = dict(payload["idf"])
        self.vectors = payload["vectors"]
        metadata = payload.get("metadata", {})
        return metadata if isinstance(metadata, dict) else {}


def _load_embedding_model():
    """Best-effort Sentence-Transformers loader with process-level cache."""
    global _EMBED_MODEL, _EMBED_MODEL_NAME
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    model_name = os.environ.get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _EMBED_MODEL = SentenceTransformer(model_name)
        _EMBED_MODEL_NAME = model_name
        return _EMBED_MODEL
    except Exception as e:
        warnings.warn(
            "Embedding model unavailable, fallback to TF-IDF only. "
            "Install with `pip install -U sentence-transformers`. "
            f"Requested model: {model_name}. Error: {e}"
        )
        _EMBED_MODEL = None
        _EMBED_MODEL_NAME = None
        return None


def _encode_sentences(model: Any, sentences: List[str]) -> Any:
    """Use the standard Sentence-Transformers encode API."""
    if not sentences:
        return []
    return model.encode(
        sentences,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def _embedding_search_scores(
    query_text: str,
    chunks: List[Chunk],
) -> Dict[str, float]:
    model = _load_embedding_model()
    if model is None or not chunks:
        return {}
    try:
        import numpy as np

        query_vec = _encode_sentences(model, [query_text])
        doc_texts = [c.text for c in chunks]
        doc_vecs = _encode_sentences(model, doc_texts)
        sims = (doc_vecs @ query_vec[0]).astype(float)
        return {c.chunk_id: float(sims[i]) for i, c in enumerate(chunks)}
    except Exception as e:
        warnings.warn(f"Embedding scoring failed, fallback to TF-IDF only: {e}")
        return {}


def flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            parts.append(f"{k}: {flatten(v)}")
        return "; ".join(parts)
    if isinstance(value, list):
        return " | ".join(flatten(v) for v in value)
    return str(value)


def normalize_action_type(action_type: str) -> str:
    a = (action_type or "").strip().lower()
    return a if a in ACTION_SPACE_WITH_UNKNOWN else "unknown_action"


def strategy_to_action_type(action_strategy: str) -> str:
    # Canonicalize all incoming strategies to the current action space.
    return normalize_action_type(action_strategy)


def action_to_dimensions(action_type: str) -> List[str]:
    a = normalize_action_type(action_type)
    mapping = {
        "additive_change": ["additive identity", "additive concentration"],
        "salt_replace": ["salt identity"],
        "salt_concentration_shift": ["salt concentration"],
        "add_new_salt": ["salt identity", "dual-salt combinations", "salt concentration"],
        "solvent_replace": ["solvent composition", "solvent identity"],
        "solvent_concentration_shift": ["solvent composition"],
        "add_new_solvent": ["solvent composition", "solvent identity"],
        "remove_solvent": ["solvent composition", "solvent identity"],
    }
    return mapping.get(a, [])


def normalize_text_tokens(text: str) -> List[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).lower()


def normalize_role(role: str) -> str:
    r = normalize_text(role)
    if "baseline" in r:
        return "baseline"
    if "reference" in r:
        return "reference"
    if "control" in r:
        return "control"
    if "standard" in r:
        return "standard"
    if "optimized" in r:
        return "optimized"
    if "variant" in r:
        return "variant"
    return r.strip() or "unknown"


def extract_additives(comp: Dict[str, Any]) -> set:
    names = set()

    def _collect(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for k, v in value.items():
                names.add(normalize_text(k).strip())
                if isinstance(v, str):
                    names.add(normalize_text(v).strip())
        elif isinstance(value, list):
            for item in value:
                _collect(item)
        else:
            names.add(normalize_text(value).strip())

    _collect(comp.get("additive"))
    _collect(comp.get("additives"))
    return {re.sub(r"\s+", " ", n) for n in names if n}


def extract_solvents(comp: Dict[str, Any]) -> Dict[str, str]:
    solvents = comp.get("solvents") or {}
    result: Dict[str, str] = {}
    for k, v in solvents.items():
        key = normalize_text(k).strip()
        val = normalize_text(v).strip()
        if key:
            result[key] = val
    return result


def split_component_tokens(value: Any) -> Set[str]:
    raw = normalize_text(value)
    if not raw.strip():
        return set()
    raw = re.sub(r"\([^)]*\)", " ", raw)
    pieces = re.split(r"\s*(?:\+|/|,|;|&|\band\b)\s*", raw)
    out: Set[str] = set()
    for piece in pieces:
        token = re.sub(r"^\d+(?:\.\d+)?\s*(?:m|mm|mol/?l|wt\.?%|vol\.?%)\s*", "", piece).strip()
        token = re.sub(r"\s+", " ", token)
        if token and not re.fullmatch(r"[-+]?\d*\.?\d+", token):
            out.add(token)
    return out


def relative_change(base: Optional[float], value: Optional[float]) -> Optional[float]:
    if base is None or value is None:
        return None
    denom = max(abs(float(base)), 1e-8)
    return abs(float(value) - float(base)) / denom


def _numeric_ratio_map(ratios: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in (ratios or {}).items():
        parsed = parse_numeric_value(value)
        if parsed is None:
            continue
        out[str(key)] = float(parsed)
    return out


def composition_total_change(before: Dict[str, Any], after: Dict[str, Any]) -> Optional[float]:
    b_num = _numeric_ratio_map(before)
    a_num = _numeric_ratio_map(after)
    if not b_num and not a_num:
        return None
    b_total = sum(b_num.values())
    a_total = sum(a_num.values())
    if b_total <= 0.0 or a_total <= 0.0:
        return None
    keys = set(before.keys()) | set(after.keys())
    return sum(abs((b_num.get(k, 0.0) / b_total) - (a_num.get(k, 0.0) / a_total)) for k in keys)


def normalize_objective_primary(value: str) -> str:
    v = (value or "").strip()
    if v in ALLOWED_OBJECTIVES:
        return v
    return "unspecified_performance_objective"


def objective_to_target_property(objective: str) -> str:
    mapping = {
        "ion_transport_temperature_dependence": "ionic conductivity and temperature dependence",
        "viscosity_transport_coupling": "viscosity-transport coupling",
        "solvation_association_temperature_stability": "solvation and interfacial stability",
        "unspecified_performance_objective": "cycling stability",
    }
    return mapping.get(objective, "cycling stability")


def infer_objective_tags_for_precedent(
    paper: Dict[str, Any],
    formulation: Dict[str, Any],
    prop: Optional[Dict[str, Any]] = None,
) -> List[str]:
    text_parts = [
        flatten(formulation.get("evaluated_properties", [])),
        flatten(paper.get("design_conclusions", [])),
        flatten(paper.get("design_boundaries", {})),
    ]
    if prop is not None:
        text_parts.extend(
            [
                str(prop.get("property", "")),
                flatten(prop.get("conditions", [])),
                str(prop.get("outcome", "")),
            ]
        )
    return infer_objective_tags_from_text(" ".join(text_parts))


def infer_rl_action_type_for_precedent(
    before_comp: Dict[str, Any],
    after_comp: Dict[str, Any],
    after_id: str,
    paper: Optional[Dict[str, Any]] = None,
) -> str:
    # Mirror json2priorreward.infer_action_type priority and outputs.
    base_add = extract_additives(before_comp)
    var_add = extract_additives(after_comp)
    base_salt = normalize_text(before_comp.get("salt", "")).strip()
    var_salt = normalize_text(after_comp.get("salt", "")).strip()
    base_sc = normalize_text(before_comp.get("salt_concentration", "")).strip()
    var_sc = normalize_text(after_comp.get("salt_concentration", "")).strip()
    base_solvents = extract_solvents(before_comp)
    var_solvents = extract_solvents(after_comp)
    base_solvent_keys = set(base_solvents.keys())
    var_solvent_keys = set(var_solvents.keys())
    base_salt_tokens = split_component_tokens(base_salt)
    var_salt_tokens = split_component_tokens(var_salt)
    salt_rel_delta = relative_change(
        parse_concentration_value_strict(base_sc),
        parse_concentration_value_strict(var_sc),
    )

    if base_add != var_add:
        return "additive_change"

    if base_salt != var_salt:
        if not base_salt and var_salt:
            if salt_rel_delta is not None and salt_rel_delta < SMALL_CHANGE_THRESHOLD:
                return "additive_change"
            if salt_rel_delta is not None and salt_rel_delta >= SMALL_CHANGE_THRESHOLD:
                return "add_new_salt"
        elif (
            base_salt_tokens
            and var_salt_tokens
            and base_salt_tokens.issubset(var_salt_tokens)
            and len(var_salt_tokens) > len(base_salt_tokens)
        ):
            if salt_rel_delta is not None and salt_rel_delta < SMALL_CHANGE_THRESHOLD:
                return "additive_change"
            if salt_rel_delta is not None and salt_rel_delta >= SMALL_CHANGE_THRESHOLD:
                return "add_new_salt"
        else:
            return "salt_replace"
    elif base_sc and var_sc and base_sc != var_sc:
        if salt_rel_delta is not None and salt_rel_delta < SMALL_CHANGE_THRESHOLD:
            return "additive_change"
        if salt_rel_delta is not None and salt_rel_delta >= SMALL_CHANGE_THRESHOLD:
            return "salt_concentration_shift"

    if base_solvent_keys != var_solvent_keys:
        solvent_total_delta = composition_total_change(base_solvents, var_solvents)
        if solvent_total_delta is not None and solvent_total_delta < SMALL_CHANGE_THRESHOLD:
            return "additive_change"
        if base_solvent_keys and var_solvent_keys and base_solvent_keys.issubset(var_solvent_keys):
            return "add_new_solvent"
        if base_solvent_keys and var_solvent_keys and var_solvent_keys.issubset(base_solvent_keys):
            return "remove_solvent"
        if not base_solvent_keys and var_solvent_keys:
            return "add_new_solvent"
        if base_solvent_keys and not var_solvent_keys:
            return "remove_solvent"
        return "solvent_replace"

    if base_solvent_keys == var_solvent_keys and base_solvent_keys:
        if any(base_solvents[k] != var_solvents.get(k, "") for k in base_solvent_keys):
            solvent_total_delta = composition_total_change(base_solvents, var_solvents)
            if solvent_total_delta is not None and solvent_total_delta < SMALL_CHANGE_THRESHOLD:
                return "additive_change"
            if solvent_total_delta is not None and solvent_total_delta >= SMALL_CHANGE_THRESHOLD:
                return "solvent_concentration_shift"

    if paper:
        exploration = paper.get("exploration_operations", {}) or {}
        boundaries = paper.get("design_boundaries", {}) or {}
        explored_text = normalize_text(" ".join(exploration.get("explored_dimensions", [])))
        not_explored_text = normalize_text(" ".join(exploration.get("not_explored", [])))
        intent_text = normalize_text(" ".join(boundaries.get("intentional_constraints", [])))
        fid = normalize_text(after_id)

        additive_only = (
            "additive" in explored_text
            and "salt" in not_explored_text
            and "solvent" in not_explored_text
        ) or ("additive-only" in intent_text or "additive only" in intent_text)

        if additive_only:
            return "additive_change"
        if any(k in fid for k in ["without additive", "additive-free", "no additive"]):
            return "additive_change"
        if "additive" in fid:
            return "additive_change"

    return "unknown_action"


def parse_numeric_value(text: Any) -> Optional[float]:
    if text is None:
        return None
    s = str(text)
    # Prefer values tied to concentration units to avoid matching salt names (e.g., PF6)
    # or unrelated units (e.g., mAh).
    molarity = re.findall(r"(\d+(?:\.\d+)?)\s*(?:M|mol/?L)\b", s)
    fraction = re.findall(r"(\d+(?:\.\d+)?)\s*(?:wt\.?%|vol\.?%|%)\b", s, flags=re.IGNORECASE)
    unit_nums = [float(x) for x in (molarity + fraction)]
    if unit_nums:
        return max(unit_nums)

    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return None
    values = [float(x) for x in nums]
    # If range exists, return max as loose upper bound.
    return max(values)


def parse_concentration_value_strict(text: Any) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if "not explicitly specified" in s.lower():
        return None

    nums: List[float] = []
    for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:M|mol/?L)\b", s):
        nums.append(float(x))
    for x in re.findall(r"(\d+(?:\.\d+)?)\s*mM\b", s):
        nums.append(float(x) / 1000.0)
    for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:wt\.?%|vol\.?%|%)\b", s, flags=re.IGNORECASE):
        nums.append(float(x))
    if not nums:
        return None
    return max(nums)


def classify_polarity(text: str) -> str:
    t = (text or "").lower()
    pos = sum(1 for k in POSITIVE_HINTS if k in t)
    neg = sum(1 for k in NEGATIVE_HINTS if k in t)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "mixed/uncertain"


def select_baseline(paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    formulations = (paper or {}).get("formulations", []) or []
    if not formulations:
        return None

    # Priority: explicit baseline -> reference/control/standard -> id keywords -> first formulation.
    for f in formulations:
        if normalize_role(f.get("role", "")) == "baseline":
            return f

    for f in formulations:
        r = normalize_role(f.get("role", ""))
        if r in {"reference", "control", "standard"}:
            return f

    id_keywords = ["standard", "pristine", "blank", "control", "reference", "std"]
    for f in formulations:
        fid = normalize_text(f.get("id", ""))
        if any(k in fid for k in id_keywords):
            return f

    conclusions = normalize_text(" ".join((paper or {}).get("design_conclusions", []) or []))
    if any(k in conclusions for k in ["standard", "pristine", "blank", "base electrolyte"]):
        return formulations[0]

    return formulations[0]


def infer_action_types(before_comp: Dict[str, Any], after_comp: Dict[str, Any]) -> List[str]:
    actions: Set[str] = set()

    b_add = extract_additives(before_comp)
    a_add = extract_additives(after_comp)
    if b_add != a_add:
        actions.add("additive_change")

    b_salt = normalize_text(before_comp.get("salt", "")).strip()
    a_salt = normalize_text(after_comp.get("salt", "")).strip()
    b_salt_tokens = split_component_tokens(b_salt)
    a_salt_tokens = split_component_tokens(a_salt)
    b_sc_raw = normalize_text(before_comp.get("salt_concentration", "")).strip()
    a_sc_raw = normalize_text(after_comp.get("salt_concentration", "")).strip()
    salt_rel_delta = relative_change(
        parse_concentration_value_strict(b_sc_raw),
        parse_concentration_value_strict(a_sc_raw),
    )

    if b_salt != a_salt:
        if not b_salt and a_salt:
            if salt_rel_delta is not None and salt_rel_delta < SMALL_CHANGE_THRESHOLD:
                actions.add("additive_change")
            elif salt_rel_delta is not None:
                actions.add("add_new_salt")
        elif (
            b_salt_tokens
            and a_salt_tokens
            and b_salt_tokens.issubset(a_salt_tokens)
            and len(a_salt_tokens) > len(b_salt_tokens)
        ):
            if salt_rel_delta is not None and salt_rel_delta < SMALL_CHANGE_THRESHOLD:
                actions.add("additive_change")
            elif salt_rel_delta is not None:
                actions.add("add_new_salt")
        else:
            actions.add("salt_replace")
    elif b_sc_raw and a_sc_raw and b_sc_raw != a_sc_raw:
        if salt_rel_delta is not None and salt_rel_delta < SMALL_CHANGE_THRESHOLD:
            actions.add("additive_change")
        elif salt_rel_delta is not None:
            actions.add("salt_concentration_shift")

    b_sol = extract_solvents(before_comp)
    a_sol = extract_solvents(after_comp)
    b_keys = set(b_sol.keys())
    a_keys = set(a_sol.keys())
    if b_keys != a_keys:
        solvent_total_delta = composition_total_change(b_sol, a_sol)
        if solvent_total_delta is not None and solvent_total_delta < SMALL_CHANGE_THRESHOLD:
            actions.add("additive_change")
        elif b_keys and a_keys and b_keys.issubset(a_keys):
            actions.add("add_new_solvent")
        elif b_keys and a_keys and a_keys.issubset(b_keys):
            actions.add("remove_solvent")
        elif not b_keys and a_keys:
            actions.add("add_new_solvent")
        elif b_keys and not a_keys:
            actions.add("remove_solvent")
        else:
            actions.add("solvent_replace")
    elif b_keys and any(b_sol[k] != a_sol.get(k, "") for k in b_keys):
        solvent_total_delta = composition_total_change(b_sol, a_sol)
        if solvent_total_delta is not None and solvent_total_delta < SMALL_CHANGE_THRESHOLD:
            actions.add("additive_change")
        elif solvent_total_delta is not None:
            actions.add("solvent_concentration_shift")

    if "additive_change" in actions and any(a != "additive_change" for a in actions) and b_add == a_add:
        actions.discard("additive_change")

    if not actions:
        return ["unknown_action"]

    ordered = [a for a in ACTION_SPACE if a in actions]
    return ordered if ordered else ["unknown_action"]


def detect_condition_flags(conditions: List[str]) -> Dict[str, bool]:
    text = " ".join(conditions).lower()
    temps = [float(x) for x in re.findall(r"(-?\d+(?:\.\d+)?)\s*°\s*c", text)]
    volts = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*v", text)]
    return {
        "low_temp": any(t <= 10.0 for t in temps),
        "high_temp": any(t >= 45.0 for t in temps),
        "high_voltage": any(v >= 4.4 for v in volts),
        "lithium_metal": ("li/li" in text) or ("lithium metal" in text) or ("li metal" in text),
    }


def infer_context_tags_from_formulation(formulation: Dict[str, Any]) -> List[str]:
    conditions = get_formulation_conditions(formulation)
    flags = detect_condition_flags(conditions)
    return [k for k, v in flags.items() if v]


def get_formulation_conditions(formulation: Dict[str, Any]) -> List[str]:
    raw = (formulation or {}).get("conditions", [])
    if not isinstance(raw, list):
        return []
    out = [str(x).strip() for x in raw]
    return [x for x in out if x]


def normalize_query_formulation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Accept direct formulation or start_point-style payload and normalize fields."""
    formulation = data if isinstance(data, dict) else {}
    normalized = dict(formulation)

    comp = (formulation.get("composition", {}) or {})
    if not isinstance(comp, dict) or not comp:
        comp = ((formulation.get("state0", {}) or {}).get("composition", {}) or {})
    if not isinstance(comp, dict):
        comp = {}
    normalized["composition"] = comp

    conditions = get_formulation_conditions(formulation)
    if conditions:
        normalized["conditions"] = conditions
    else:
        normalized.pop("conditions", None)

    return normalized


def extract_salt_from_entities(formulation: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    entities = (formulation or {}).get("entities", {}) or {}
    salts = entities.get("salts")
    if salts is None:
        return "", None

    # Expected patterns:
    # 1) [{"name": "LiPF6", "concentration": "1.0 M"}, ...]
    # 2) ["LiPF6", ...]
    # 3) {"name": "LiPF6", "concentration": "1.0 M"}
    if isinstance(salts, dict):
        name = str(salts.get("name", "")).strip()
        conc = salts.get("concentration")
        return name, (str(conc) if conc is not None else None)

    if isinstance(salts, list) and salts:
        first = salts[0]
        if isinstance(first, dict):
            name = str(first.get("name", "")).strip()
            conc = first.get("concentration")
            return name, (str(conc) if conc is not None else None)
        return str(first).strip(), None

    if isinstance(salts, str):
        return salts.strip(), None
    return "", None


def infer_flags_from_context_tags(context_tags: List[str]) -> Dict[str, bool]:
    text = " ".join(context_tags).lower()
    return {
        "low_temp": ("low_temp" in text) or ("low temperature" in text),
        "high_temp": ("high_temp" in text) or ("high temperature" in text),
        "high_voltage": ("high_voltage" in text) or ("high voltage" in text),
        "lithium_metal": ("li_metal" in text) or ("lithium metal" in text) or ("li metal" in text),
    }


def detect_solvent_family(comp: Dict[str, Any]) -> str:
    solvents = " ".join((comp.get("solvents", {}) or {}).keys()).lower()
    carbonate_tokens = {"ec", "emc", "dmc", "dec", "pc", "fec"}
    ether_tokens = {"dme", "diglyme", "triglyme", "tetraglyme", "dol", "thf"}
    has_carbonate = any(k in solvents for k in carbonate_tokens)
    has_ether = any(k in solvents for k in ether_tokens)
    if has_carbonate and has_ether:
        return "mixed"
    if has_carbonate:
        return "carbonate"
    if has_ether:
        return "ether"
    return "unknown"


def _normalize_component_name(name: str) -> str:
    s = str(name or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", s)


COMPONENT_ALIASES_FILENAME = "component_aliases.json"

_DEFAULT_COMPONENT_FAMILY_ALIASES: Dict[str, List[str]] = {
    "ec": ["ethylene carbonate"],
    "dmc": ["dimethyl carbonate"],
    "emc": ["ethyl methyl carbonate", "ethylmethyl carbonate", "ethylene methyl carbonate"],
    "dec": ["diethyl carbonate"],
    "pc": ["propylene carbonate"],
    "fec": ["fluoroethylene carbonate"],
    "vc": ["vinylene carbonate"],
    "vec": ["vinyl ethylene carbonate"],
    "tep": ["triethyl phosphate"],
    "sl": ["sulfolane"],
    "sn": ["succinonitrile"],
    "adn": ["adiponitrile"],
    "an": ["acetonitrile"],
    "thp": ["tetrahydropyran"],
    "ea": ["ethyl acetate"],
    "dee": ["diethyl ether"],
    "dme": ["1,2-dimethoxyethane"],
    "dol": ["1,3-dioxolane"],
    "dx": ["1,4-dioxane"],
    "fb": ["fluorobenzene"],
    "mp": ["methyl propionate"],
    "vn": ["valeronitrile"],
    "bl": ["butyrolactam", "butyrolactone"],
    "h2o": ["water"],
}


def load_component_aliases(base_dir: Path) -> Dict[str, List[str]]:
    path = base_dir / COMPONENT_ALIASES_FILENAME
    payload: Any = _DEFAULT_COMPONENT_FAMILY_ALIASES
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = _DEFAULT_COMPONENT_FAMILY_ALIASES

    alias_groups: List[List[str]] = []
    for canonical_name, aliases in (payload or {}).items():
        names = [str(canonical_name).strip()]
        if isinstance(aliases, str):
            names.append(aliases.strip())
        elif isinstance(aliases, list):
            names.extend(str(item).strip() for item in aliases)
        cleaned = [name for name in names if name]
        if cleaned:
            alias_groups.append(cleaned)

    normalized_aliases: Dict[str, List[str]] = {}
    for group in alias_groups:
        unique_names: List[str] = []
        seen_group_keys: Set[str] = set()
        for name in group:
            key = _normalize_component_name(name)
            if not key or key in seen_group_keys:
                continue
            seen_group_keys.add(key)
            unique_names.append(name)
        for name in unique_names:
            name_key = _normalize_component_name(name)
            linked = [other for other in unique_names if _normalize_component_name(other) != name_key]
            normalized_aliases[name_key] = linked
    return normalized_aliases


_COMPONENT_FAMILY_ALIASES = load_component_aliases(Path(__file__).resolve().parent)

LEARNED_FAMILY_TABLES_FILENAME = ".learned_family_tables.json"
DEFAULT_FAMILY_CLASSIFIER_MODEL = "gpt-5"


def _empty_family_tables() -> Dict[str, Dict[str, List[str]]]:
    return {"additives": {}, "salts": {}, "solvents": {}}


def load_learned_family_tables(base_dir: Path) -> Dict[str, Dict[str, List[str]]]:
    path = base_dir / LEARNED_FAMILY_TABLES_FILENAME
    if not path.exists():
        return _empty_family_tables()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_family_tables()

    out = _empty_family_tables()
    if not isinstance(payload, dict):
        return out
    for key in out.keys():
        bucket = payload.get(key, {})
        if not isinstance(bucket, dict):
            continue
        normalized_bucket: Dict[str, List[str]] = {}
        for family, members in bucket.items():
            family_name = str(family).strip()
            if not family_name or not isinstance(members, list):
                continue
            kept: List[str] = []
            seen_member_keys: Set[str] = set()
            for member in members:
                member_name = str(member).strip()
                member_key = _normalize_component_name(member_name)
                if not member_name or not member_key or member_key in seen_member_keys:
                    continue
                seen_member_keys.add(member_key)
                kept.append(member_name)
            if kept:
                normalized_bucket[family_name] = kept
        out[key] = normalized_bucket
    return out


def save_learned_family_tables(base_dir: Path, learned_tables: Dict[str, Dict[str, List[str]]]) -> None:
    path = base_dir / LEARNED_FAMILY_TABLES_FILENAME
    payload = _empty_family_tables()
    for key in payload.keys():
        bucket = (learned_tables or {}).get(key, {})
        if isinstance(bucket, dict):
            payload[key] = {
                str(family): [str(member).strip() for member in members if str(member).strip()]
                for family, members in bucket.items()
                if str(family).strip() and isinstance(members, list)
            }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_family_tables(
    base_tables: Dict[str, Dict[str, List[str]]],
    learned_tables: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Dict[str, List[str]]]:
    merged = _empty_family_tables()
    for key in merged.keys():
        seen_by_family: Dict[str, Set[str]] = {}
        out_bucket: Dict[str, List[str]] = {}
        for source in (base_tables or {}, learned_tables or {}):
            bucket = source.get(key, {})
            if not isinstance(bucket, dict):
                continue
            for family, members in bucket.items():
                family_name = str(family).strip()
                if not family_name:
                    continue
                seen_bucket = seen_by_family.setdefault(family_name, set())
                dst = out_bucket.setdefault(family_name, [])
                if not isinstance(members, list):
                    continue
                for member in members:
                    member_name = str(member).strip()
                    member_key = _normalize_component_name(member_name)
                    if not member_name or not member_key or member_key in seen_bucket:
                        continue
                    seen_bucket.add(member_key)
                    dst.append(member_name)
        merged[key] = out_bucket
    return merged


def _normalize_family_identifier(value: Any) -> str:
    raw = str(value or "").strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw


def _component_names_by_kind_from_formulation(current_formulation: Dict[str, Any]) -> Dict[str, List[str]]:
    comp = ((current_formulation or {}).get("composition", {}) or {})
    out = {"solvents": [], "salts": [], "additives": []}
    solvents = comp.get("solvents", {}) or {}
    if isinstance(solvents, dict):
        out["solvents"] = [str(name).strip() for name in solvents.keys() if str(name).strip()]
    salt_name = str(comp.get("salt", "")).strip()
    if salt_name:
        out["salts"] = [salt_name]
    additives = comp.get("additives", {}) or {}
    if isinstance(additives, dict):
        out["additives"] = [str(name).strip() for name in additives.keys() if str(name).strip()]
    return out


def _temp_kwargs(model: Any, temperature: float) -> Dict[str, Any]:
    """Omit `temperature` for models that reject a non-default value (gpt-5 and other
    reasoning models only accept the default), otherwise pass it through."""
    m = str(model or "").lower()
    if m.startswith(("gpt-5", "o1", "o3", "o4")):
        return {}
    return {"temperature": temperature}


def _classify_unknown_component_with_openai(
    component_name: str,
    component_kind: str,
    family_table: Dict[str, List[str]],
    model: str = DEFAULT_FAMILY_CLASSIFIER_MODEL,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    family_options = [
        {
            "family": str(family),
            "example_members": list(members or [])[:12],
        }
        for family, members in sorted((family_table or {}).items(), key=lambda item: str(item[0]))
    ]
    system_prompt = (
        "You classify battery electrolyte formulation components into chemical family tables.\n"
        "The component role is already known: solvent, salt, or additive.\n"
        "Prefer an existing family whenever chemically defensible.\n"
        "Create a new family only if none of the existing families fit.\n"
        "Output strict JSON with keys: selected_family, create_new_family, new_family_name, rationale_short."
    )
    user_prompt = json.dumps(
        {
            "task": "classify_unknown_battery_electrolyte_component",
            "component_name": component_name,
            "component_kind": component_kind,
            "existing_families": family_options,
            "rules": [
                "selected_family must be one existing family name when create_new_family is false",
                "when create_new_family is true, new_family_name must be a short lowercase snake_case family identifier",
                "new family names should describe chemistry, not a single component name, unless the chemistry is genuinely unique",
            ],
        },
        ensure_ascii=False,
    )
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **_temp_kwargs(model, 0.0),
    )
    text = (resp.output_text or "").strip()
    try:
        obj = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"family classifier returned non-JSON: {text}") from e
    if not isinstance(obj, dict):
        raise RuntimeError(f"family classifier returned invalid payload: {text}")
    return obj


def prepare_family_tables_for_formulation(
    current_formulation: Dict[str, Any],
    base_dir: Path,
    model: str = DEFAULT_FAMILY_CLASSIFIER_MODEL,
) -> Dict[str, Dict[str, List[str]]]:
    static_tables = load_family_tables(base_dir)
    learned_tables = load_learned_family_tables(base_dir)
    merged_tables = _merge_family_tables(static_tables, learned_tables)

    component_names = _component_names_by_kind_from_formulation(current_formulation)
    changed = False
    for kind, names in component_names.items():
        current_bucket = merged_tables.get(kind, {})
        current_index = build_family_member_index(current_bucket)
        learned_bucket = learned_tables.setdefault(kind, {})
        for name in names:
            if _normalize_component_name(name) in current_index:
                continue
            try:
                classification = _classify_unknown_component_with_openai(
                    component_name=name,
                    component_kind=kind[:-1] if kind.endswith("s") else kind,
                    family_table=current_bucket,
                    model=model,
                )
            except Exception:
                continue

            create_new_family = bool(classification.get("create_new_family"))
            selected_family = str(classification.get("selected_family", "")).strip()
            if create_new_family:
                selected_family = _normalize_family_identifier(
                    classification.get("new_family_name") or selected_family or f"{kind}_other"
                )
            if not selected_family:
                continue
            if not create_new_family and selected_family not in current_bucket:
                continue
            learned_members = learned_bucket.setdefault(selected_family, [])
            member_key = _normalize_component_name(name)
            if member_key not in {_normalize_component_name(x) for x in learned_members}:
                learned_members.append(name)
                changed = True
                current_bucket = _merge_family_tables(static_tables, learned_tables).get(kind, {})
                current_index = build_family_member_index(current_bucket)
                merged_tables = _merge_family_tables(static_tables, learned_tables)

    if changed:
        save_learned_family_tables(base_dir, learned_tables)
        merged_tables = _merge_family_tables(static_tables, learned_tables)
    return merged_tables


def prepare_family_indexes_for_formulation(
    current_formulation: Dict[str, Any],
    base_dir: Path,
    model: str = DEFAULT_FAMILY_CLASSIFIER_MODEL,
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]:
    family_tables = prepare_family_tables_for_formulation(current_formulation, base_dir=base_dir, model=model)
    family_indexes = {
        "additives": build_family_member_index(family_tables.get("additives", {})),
        "salts": build_family_member_index(family_tables.get("salts", {})),
        "solvents": build_family_member_index(family_tables.get("solvents", {})),
    }
    return family_tables, family_indexes


def load_family_tables(base_dir: Path) -> Dict[str, Dict[str, List[str]]]:
    files = {
        "additives": "additive_families.json",
        "salts": "salt_families.json",
        "solvents": "solvent_families.json",
    }
    out: Dict[str, Dict[str, List[str]]] = {"additives": {}, "salts": {}, "solvents": {}}
    for key, filename in files.items():
        path = base_dir / filename
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                out[key] = payload
        except Exception:
            continue
    return out


def build_family_member_index(family_table: Dict[str, List[str]]) -> Dict[str, List[str]]:
    idx: Dict[str, set] = defaultdict(set)
    for family, members in (family_table or {}).items():
        for member in (members or []):
            member_key = _normalize_component_name(str(member))
            keys = {member_key} if member_key else set()
            for alias in _COMPONENT_FAMILY_ALIASES.get(member_key, []):
                alias_key = _normalize_component_name(alias)
                if alias_key:
                    keys.add(alias_key)
            for k in keys:
                idx[k].add(str(family))
    return {k: sorted(v) for k, v in idx.items()}


def collect_component_families(
    comp: Dict[str, Any],
    family_indexes: Dict[str, Dict[str, List[str]]],
) -> Dict[str, set]:
    families = {"solvents": set(), "additives": set(), "salts": set()}
    if not comp:
        return families

    for s_name in (comp.get("solvents", {}) or {}).keys():
        k = _normalize_component_name(str(s_name))
        for fam in family_indexes.get("solvents", {}).get(k, []):
            families["solvents"].add(fam)

    for a_name in (comp.get("additives", {}) or {}).keys():
        k = _normalize_component_name(str(a_name))
        for fam in family_indexes.get("additives", {}).get(k, []):
            families["additives"].add(fam)

    salt_name = str(comp.get("salt", "")).strip()
    if salt_name:
        k = _normalize_component_name(salt_name)
        for fam in family_indexes.get("salts", {}).get(k, []):
            families["salts"].add(fam)

    return families


def actions_related(query_action_type: str, precedent_action_types: List[str]) -> bool:
    q = normalize_action_type(query_action_type)
    pa = {normalize_action_type(x) for x in (precedent_action_types or [])}
    if not q or q == "unknown_action":
        return True
    if q in pa:
        return True
    q_dims = set(action_to_dimensions(q))
    if not q_dims:
        return False
    for a in pa:
        if q_dims & set(action_to_dimensions(a)):
            return True
    return False


def normalize_action_to_rl_space(action: str) -> str:
    # Kept for backward compatibility at call sites; now directly returns the canonical action space.
    return normalize_action_type(action)


def infer_objective_tags_from_text(text: str) -> List[str]:
    t = (text or "").lower()
    matched: List[str] = []
    for tag, keys in OBJECTIVE_TAGS.items():
        if any(k in t for k in keys):
            matched.append(tag)
    if not matched:
        return ["unspecified_performance_objective"]
    return matched


def build_chunks(papers: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []

    for p_idx, paper in enumerate(papers):
        meta = paper.get("paper_metadata", {})
        title = str(meta.get("title", "unknown"))
        doi = str(meta.get("doi", "unknown"))

        paper_text = "\n".join(
            [
                f"Paper: {title}",
                f"DOI: {doi}",
                f"Design conclusions: {flatten(paper.get('design_conclusions', []))}",
                f"Design boundaries: {flatten(paper.get('design_boundaries', {}))}",
                f"Exploration operations: {flatten(paper.get('exploration_operations', {}))}",
            ]
        )
        chunks.append(
            Chunk(
                chunk_id=f"paper_{p_idx}",
                paper_title=title,
                doi=doi,
                formulation_id="N/A",
                role="paper-level",
                property_name="paper_summary",
                text=paper_text,
            )
        )

        for f_idx, formulation in enumerate(paper.get("formulations", [])):
            formulation_id = str(formulation.get("id", f"f_{f_idx}"))
            role = str(formulation.get("role", "unknown"))
            composition = flatten(formulation.get("composition", {}))
            property_name, _, _, _, _ = summarize_formulation_properties(paper, formulation)
            evaluated_properties = collect_evaluated_properties(formulation)
            property_blocks: List[str] = []
            for item in evaluated_properties:
                property_blocks.append(
                    "\n".join(
                        [
                            f"Property: {item.get('property', '')}",
                            f"Conditions: {flatten(item.get('conditions', []))}",
                            f"Outcome: {item.get('outcome', '')}",
                        ]
                    )
                )

            text = "\n".join(
                [
                    f"Paper: {title}",
                    f"DOI: {doi}",
                    f"Formulation: {formulation_id} ({role})",
                    f"Composition: {composition}",
                    "Evaluated properties:",
                    "\n\n".join(property_blocks) if property_blocks else "None",
                ]
            )

            chunks.append(
                Chunk(
                    chunk_id=f"{p_idx}_{f_idx}",
                    paper_title=title,
                    doi=doi,
                    formulation_id=formulation_id,
                    role=role,
                    property_name=property_name,
                    text=text,
                )
            )

    return chunks


def collect_evaluated_properties(formulation: Dict[str, Any]) -> List[Dict[str, Any]]:
    props = formulation.get("evaluated_properties", []) or []
    normalized: List[Dict[str, Any]] = []
    for prop in props:
        if not isinstance(prop, dict):
            continue
        property_name = str(prop.get("property", "unknown_property")).strip() or "unknown_property"
        conditions = [str(x).strip() for x in (prop.get("conditions", []) or []) if str(x).strip()]
        outcome = str(prop.get("outcome", "")).strip()
        normalized.append(
            {
                "property": property_name,
                "conditions": conditions,
                "outcome": outcome,
            }
        )
    return normalized


def summarize_formulation_properties(
    paper: Dict[str, Any],
    formulation: Dict[str, Any],
) -> Tuple[str, List[str], str, str, List[str]]:
    props = collect_evaluated_properties(formulation)
    if not props:
        objective_tags = infer_objective_tags_for_precedent(paper, formulation, None)
        return "formulation_summary", [], "", "mixed/uncertain", objective_tags

    property_names: List[str] = []
    conditions: List[str] = []
    outcome_lines: List[str] = []
    polarities: Set[str] = set()
    objective_tag_set: Set[str] = set()

    for prop in props:
        property_name = str(prop.get("property", "unknown_property")).strip() or "unknown_property"
        if property_name not in property_names:
            property_names.append(property_name)

        for cond in (prop.get("conditions", []) or []):
            cond_text = str(cond).strip()
            if cond_text and cond_text not in conditions:
                conditions.append(cond_text)

        outcome = str(prop.get("outcome", "")).strip()
        if outcome:
            outcome_lines.append(f"{property_name}: {outcome}")
            polarities.add(classify_polarity(outcome))

        for tag in infer_objective_tags_for_precedent(paper, formulation, prop):
            tag_text = str(tag).strip()
            if tag_text:
                objective_tag_set.add(tag_text)

    if polarities == {"positive"}:
        polarity = "positive"
    elif polarities == {"negative"}:
        polarity = "negative"
    elif polarities:
        polarity = "mixed/uncertain"
    else:
        polarity = "mixed/uncertain"

    priority_terms = [
        "cycling performance",
        "cycling stability",
        "capacity retention",
        "coulombic efficiency",
        "ionic conductivity",
        "conductivity",
        "rate capability",
        "electrochemical impedance",
        "impedance",
        "oxidation stability",
        "electrochemical stability",
        "viscosity",
        "solvation",
        "sei",
        "cei",
    ]

    property_summary = "formulation_summary"
    lowered_names = [(name, normalize_text(name)) for name in property_names]
    for term in priority_terms:
        term_norm = normalize_text(term)
        matched = next((name for name, lowered in lowered_names if term_norm in lowered), None)
        if matched:
            property_summary = matched
            break
    if property_summary == "formulation_summary" and property_names:
        property_summary = property_names[0]

    if not objective_tag_set:
        objective_tag_set.update(infer_objective_tags_for_precedent(paper, formulation, None))

    return (
        property_summary,
        conditions,
        "\n".join(outcome_lines),
        polarity,
        sorted(objective_tag_set),
    )


def build_precedents(papers: List[Dict[str, Any]]) -> List[Precedent]:
    precedents: List[Precedent] = []

    for p_idx, paper in enumerate(papers):
        # Records of the processed literature database carry their precedents precomputed.
        if "precedents" in paper:
            precedents.extend(Precedent(**row) for row in paper["precedents"])
            continue

        title = str((paper.get("paper_metadata") or {}).get("title", "unknown"))
        doi = str((paper.get("paper_metadata") or {}).get("doi", "unknown"))
        explored_dimensions = [str(x).lower() for x in ((paper.get("exploration_operations") or {}).get("explored_dimensions", []) or [])]
        formulations = paper.get("formulations", []) or []
        baseline = select_baseline(paper)
        if baseline is None:
            continue

        b_id = str(baseline.get("id", "baseline"))
        b_comp = baseline.get("composition", {}) or {}

        for f_idx, cand in enumerate(formulations):
            c_id = str(cand.get("id", f"f_{f_idx}"))
            if c_id == b_id:
                continue
            c_comp = cand.get("composition", {}) or {}
            actions = infer_action_types(b_comp, c_comp)
            rl_action_type = infer_rl_action_type_for_precedent(
                before_comp=b_comp,
                after_comp=c_comp,
                after_id=c_id,
                paper=paper,
            )
            after_role = str(cand.get("role", "unknown"))
            explicit = False
            for action in actions:
                dims = action_to_dimensions(action)
                if any(d.lower() in explored_dimensions for d in dims):
                    explicit = True
                    break

            property_name, conditions, outcome, polarity, objective_tags = summarize_formulation_properties(
                paper,
                cand,
            )
            precedents.append(
                Precedent(
                    precedent_id=f"{p_idx}_{f_idx}",
                    paper_title=title,
                    doi=doi,
                    before_id=b_id,
                    after_id=c_id,
                    before_comp=b_comp,
                    after_comp=c_comp,
                    action_types=actions,
                    property_name=property_name,
                    conditions=conditions,
                    outcome=outcome,
                    polarity=polarity,
                    after_role=after_role,
                    explored_dimensions=explored_dimensions,
                    action_explicitly_explored=explicit,
                    objective_tags=objective_tags,
                    rl_action_type=rl_action_type,
                    evaluated_properties=collect_evaluated_properties(cand),
                )
            )

    return precedents


def make_precedent_text(p: Precedent) -> str:
    return "\n".join(
        [
            f"Paper: {p.paper_title}",
            f"DOI: {p.doi}",
            f"Action types: {', '.join(p.action_types)}",
            f"RL action type: {p.rl_action_type}",
            f"Before: {p.before_id} | {flatten(p.before_comp)}",
            f"After: {p.after_id} | {flatten(p.after_comp)}",
            f"Formulation scope: {p.property_name}",
            f"Evaluated properties: {json.dumps(p.evaluated_properties, ensure_ascii=False)}",
            f"Conditions: {flatten(p.conditions)}",
            f"Outcome: {p.outcome}",
            f"Polarity: {p.polarity}",
        ]
    )


def extract_component_values_from_formulations(papers: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    stat = {
        "solvents": defaultdict(list),
        "additives": defaultdict(list),
        "salts": defaultdict(list),
        "salt_concentration": defaultdict(list),
    }

    for paper in papers:
        for f in paper.get("formulations", []) or []:
            comp = f.get("composition", {}) or {}
            salt = str(comp.get("salt", "")).strip()
            if salt:
                stat["salts"][salt.lower()].append(salt)

            sc = comp.get("salt_concentration")
            if sc is not None:
                stat["salt_concentration"]["all"].append(str(sc))

            for k, v in (comp.get("solvents", {}) or {}).items():
                stat["solvents"][str(k).lower()].append(str(v))

            for k, v in (comp.get("additives", {}) or {}).items():
                stat["additives"][str(k).lower()].append(str(v))

    return stat


def concentration_upper_bound(values: List[str], strict: bool = False) -> Optional[float]:
    nums: List[float] = []
    for v in values:
        x = parse_concentration_value_strict(v) if strict else parse_numeric_value(v)
        if x is not None:
            nums.append(x)
    if not nums:
        return None
    return max(nums)


def summarize_action_stats(precedents: List[Precedent]) -> Dict[str, Dict[str, float]]:
    count = Counter()
    neg = Counter()
    pos = Counter()

    for p in precedents:
        for a in p.action_types:
            count[a] += 1
            if p.polarity == "negative":
                neg[a] += 1
            elif p.polarity == "positive":
                pos[a] += 1

    out: Dict[str, Dict[str, float]] = {}
    for a, c in count.items():
        out[a] = {
            "frequency": float(c),
            "positive_ratio": float(pos[a] / c),
            "negative_ratio": float(neg[a] / c),
        }
    return out


def summarize_action_exploration_explicitness(precedents: List[Precedent]) -> Dict[str, Dict[str, float]]:
    totals = Counter()
    explicit = Counter()
    for p in precedents:
        for a in p.action_types:
            totals[a] += 1
            if p.action_explicitly_explored:
                explicit[a] += 1
    out: Dict[str, Dict[str, float]] = {}
    for a, c in totals.items():
        out[a] = {
            "explicit_count": float(explicit[a]),
            "total_count": float(c),
            "explicit_ratio": float(explicit[a] / c) if c else 0.0,
        }
    return out


def build_hard_constraints(
    query: DecisionQuery,
    comp_stats: Dict[str, Dict[str, List[str]]],
    action_stats: Dict[str, Dict[str, float]],
    action_explicitness: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    ft = query.current_formulation
    action_type = query.action_type
    comp = ft.get("composition", {}) or {}

    solvent_constraints = {}
    for name, value in (comp.get("solvents", {}) or {}).items():
        key = str(name).lower()
        seen = key in comp_stats["solvents"]
        ub = concentration_upper_bound(comp_stats["solvents"].get(key, []))
        solvent_constraints[name] = {
            "appeared_in_literature": seen,
            "observed_value_examples": comp_stats["solvents"].get(key, [])[:5],
            "approx_upper_bound_numeric": ub,
            "current_value": value,
        }

    additive_constraints = {}
    for name, value in (comp.get("additives", {}) or {}).items():
        key = str(name).lower()
        seen = key in comp_stats["additives"]
        ub = concentration_upper_bound(comp_stats["additives"].get(key, []))
        additive_constraints[name] = {
            "appeared_in_literature": seen,
            "observed_value_examples": comp_stats["additives"].get(key, [])[:5],
            "approx_upper_bound_numeric": ub,
            "current_value": value,
        }

    salt_name_entity, salt_conc_entity = extract_salt_from_entities(ft)
    salt_name = salt_name_entity or str(comp.get("salt", "")).strip()
    salt_seen = salt_name.lower() in comp_stats["salts"] if salt_name else False
    salt_ub = concentration_upper_bound(comp_stats["salt_concentration"].get("all", []), strict=True)

    action = action_stats.get(action_type, None)
    explicit = action_explicitness.get(action_type, None)
    if action is None:
        action_summary = {
            "appeared_in_literature": False,
            "frequency": 0,
            "risk_level": "unknown",
            "negative_ratio": None,
            "positive_ratio": None,
            "explicitly_explored_ratio": None,
        }
    else:
        neg_ratio = action["negative_ratio"]
        risk = "low"
        if neg_ratio >= 0.5:
            risk = "high"
        elif neg_ratio >= 0.25:
            risk = "medium"
        action_summary = {
            "appeared_in_literature": True,
            "frequency": int(action["frequency"]),
            "risk_level": risk,
            "negative_ratio": round(action["negative_ratio"], 3),
            "positive_ratio": round(action["positive_ratio"], 3),
            "explicitly_explored_ratio": round((explicit or {}).get("explicit_ratio", 0.0), 3),
        }

    return {
        "query_scope": {
            "target_property": query.target_property,
            "context_tags": query.context_tags,
            "objective_primary": query.objective_primary,
        },
        "salt": {
            "name": salt_name,
            "appeared_in_literature": salt_seen,
            "salt_concentration_current": salt_conc_entity if salt_conc_entity is not None else comp.get("salt_concentration"),
            "salt_concentration_approx_upper_bound_numeric": salt_ub,
        },
        "solvents": solvent_constraints,
        "additives": additive_constraints,
        "action_type": {
            "name": action_type,
            **action_summary,
        },
    }


def build_decision_query_text(query: DecisionQuery) -> str:
    parts = [
        f"action_type {query.action_type}",
        f"action_strategy {query.action_strategy}",
        f"target_property {query.target_property}",
        " ".join(query.context_tags),
    ]
    return "\n".join(parts)


def _extract_component_names(value: Any) -> set:
    out = set()
    if isinstance(value, dict):
        for k in value.keys():
            s = str(k).strip().lower()
            if s:
                out.add(s)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                n = str(item.get("name", "")).strip().lower()
                if n:
                    out.add(n)
            else:
                s = str(item).strip().lower()
                if s:
                    out.add(s)
    return out


def _extract_solvent_names(comp: Dict[str, Any]) -> set:
    solvents = comp.get("solvents", {})
    names = _extract_component_names(solvents)
    if names:
        return names
    # Support newer schema with solvents_with_smiles only.
    return _extract_component_names(comp.get("solvents_with_smiles", []))


def _extract_additive_names(comp: Dict[str, Any]) -> set:
    additives = comp.get("additives", {})
    return _extract_component_names(additives)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / union) if union else 0.0


def _concentration_similarity(a: Any, b: Any) -> float:
    va = parse_concentration_value_strict(a)
    vb = parse_concentration_value_strict(b)
    if va is None or vb is None:
        return 0.0
    denom = max(va, vb, 1e-8)
    return max(0.0, 1.0 - abs(va - vb) / denom)


def formulation_similarity_score(query_comp: Dict[str, Any], precedent_comp: Dict[str, Any]) -> float:
    q_salt = str((query_comp or {}).get("salt", "")).strip().lower()
    p_salt = str((precedent_comp or {}).get("salt", "")).strip().lower()
    salt_score = 1.0 if q_salt and p_salt and q_salt == p_salt else 0.0

    q_solvents = _extract_solvent_names(query_comp or {})
    p_solvents = _extract_solvent_names(precedent_comp or {})
    solvent_score = _jaccard(q_solvents, p_solvents)

    q_additives = _extract_additive_names(query_comp or {})
    p_additives = _extract_additive_names(precedent_comp or {})
    additive_score = _jaccard(q_additives, p_additives)

    conc_score = _concentration_similarity(
        (query_comp or {}).get("salt_concentration"),
        (precedent_comp or {}).get("salt_concentration"),
    )

    # Heavily weight component-level composition similarity.
    return (0.35 * salt_score) + (0.35 * solvent_score) + (0.2 * additive_score) + (0.1 * conc_score)


def retrieve_precedents_for_action(
    precedents: List[Precedent],
    query: DecisionQuery,
    top_k: int,
) -> List[Tuple[float, Precedent]]:
    def _coverage_select(pool: List[Tuple[float, Precedent]], limit: int = 20) -> List[Tuple[float, Precedent]]:
        if not pool:
            return []
        if len(pool) <= limit:
            return sorted(pool, key=lambda x: float(x[0]), reverse=True)

        def _cond_flags(p: Precedent) -> Set[str]:
            flags = detect_condition_flags(p.conditions)
            return {k for k, v in flags.items() if v}

        def _bucket_labels(p: Precedent) -> List[str]:
            labels: List[str] = []
            objective_tags = [str(tag).strip() for tag in (p.objective_tags or []) if str(tag).strip()]
            if objective_tags:
                for tag in objective_tags:
                    labels.append(f"objective:{tag}")
            else:
                labels.append("objective:unspecified_performance_objective")

            cond_flags = sorted(_cond_flags(p))
            if cond_flags:
                for flag in cond_flags:
                    labels.append(f"condition:{flag}")
            else:
                labels.append("condition:none")

            polarity = str(p.polarity or "mixed/uncertain").strip() or "mixed/uncertain"
            labels.append(f"polarity:{polarity}")
            return labels

        # Build coverage buckets from semantic facets instead of score ranges.
        buckets_by_label: Dict[str, List[int]] = {}
        bucket_order: List[str] = []
        for idx, (_, p) in enumerate(pool):
            for label in _bucket_labels(p):
                if label not in buckets_by_label:
                    buckets_by_label[label] = []
                    bucket_order.append(label)
                buckets_by_label[label].append(idx)
        buckets: List[List[int]] = [buckets_by_label[label] for label in bucket_order if buckets_by_label.get(label)]
        if not buckets:
            buckets = [list(range(len(pool)))]

        selected: List[Tuple[float, Precedent]] = []
        selected_ids: Set[str] = set()
        seen_objective_tags: Set[str] = set()
        seen_polarities: Set[str] = set()
        seen_action_types: Set[str] = set()
        seen_condition_flags: Set[str] = set()

        def _novelty_utility(score: float, p: Precedent) -> float:
            pol = str(p.polarity or "mixed/uncertain")
            action_set = {str(a) for a in (p.action_types or []) if str(a)}
            cond_set = _cond_flags(p)
            objective_set = {str(tag) for tag in (p.objective_tags or []) if str(tag)}
            objective_bonus = 0.04 * len(objective_set - seen_objective_tags)
            pol_bonus = 0.06 if pol and pol not in seen_polarities else 0.0
            action_bonus = 0.03 * len(action_set - seen_action_types)
            cond_bonus = 0.03 * len(cond_set - seen_condition_flags)
            return float(score) + objective_bonus + pol_bonus + action_bonus + cond_bonus

        def _commit(s: float, p: Precedent) -> None:
            selected.append((s, p))
            selected_ids.add(str(p.precedent_id))
            for tag in (p.objective_tags or []):
                if str(tag):
                    seen_objective_tags.add(str(tag))
            seen_polarities.add(str(p.polarity or "mixed/uncertain"))
            for a in (p.action_types or []):
                if str(a):
                    seen_action_types.add(str(a))
            seen_condition_flags.update(_cond_flags(p))

        # Guarantee coverage bucket leaders are included first.
        for bucket in buckets:
            if len(selected) >= limit:
                break
            if not bucket:
                continue
            for leader_idx in bucket:
                s, p = pool[leader_idx]
                pid = str(p.precedent_id)
                if pid in selected_ids:
                    continue
                _commit(float(s), p)
                break

        while len(selected) < limit:
            added_this_round = False
            for bucket in buckets:
                if len(selected) >= limit:
                    break
                best_idx = -1
                best_util = float("-inf")
                for idx in bucket:
                    s, p = pool[idx]
                    pid = str(p.precedent_id)
                    if pid in selected_ids:
                        continue
                    u = _novelty_utility(float(s), p)
                    if u > best_util:
                        best_util = u
                        best_idx = idx
                if best_idx >= 0:
                    s, p = pool[best_idx]
                    _commit(float(s), p)
                    added_this_round = True
            if not added_this_round:
                break

        # Fill remaining slots by pure score order.
        if len(selected) < limit:
            for s, p in pool:
                pid = str(p.precedent_id)
                if pid in selected_ids:
                    continue
                _commit(float(s), p)
                if len(selected) >= limit:
                    break

        selected.sort(key=lambda x: float(x[0]), reverse=True)
        return selected

    def _adaptive_truncate(
        cands: List[Tuple[float, Precedent]],
        requested_top_k: int,
        keep_all: bool = False,
    ) -> Tuple[List[Tuple[float, Precedent]], Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "fallback_triggered": False,
            "flat_low_mode": False,
            "mode": "normal",
        }
        if not cands:
            meta["mode"] = "empty"
            return [], meta

        if keep_all:
            meta["mode"] = "keep_all_exact_only"
            meta["median_guard_enabled"] = False
            meta["must_keep_floor"] = 0.0
            meta["must_keep_raw_count"] = len(cands)
            meta["must_keep_count"] = len(cands)
            meta["pool_size"] = len(cands)
            meta["all_large_gap_mode"] = False
            meta["selection_mode"] = "keep_all_score_sort"
            meta["selected_limit"] = len(cands)
            return list(cands), meta

        score_vals = [float(s) for s, _ in cands]
        median_floor = float(median(score_vals)) if score_vals else 0.0
        max_score = float(cands[0][0])
        # Use a narrow must-keep band near the head instead of admitting every
        # median-or-above candidate, which can explode the pool for broad action
        # classes with many similarly scored precedents.
        median_guard_enabled = median_floor > 0.25 and len(cands) >= max(10, requested_top_k * 2)
        must_keep_floor = max((0.85 * max_score), (median_floor + 0.08))
        must_keep_cap = min(60, max(15, requested_top_k * 4))
        must_keep_raw = [(s, p) for s, p in cands if median_guard_enabled and float(s) >= must_keep_floor]
        must_keep = must_keep_raw[:must_keep_cap]
        must_keep_ids = {str(p.precedent_id) for _, p in must_keep}
        meta["median_floor"] = round(median_floor, 6)
        meta["median_guard_enabled"] = median_guard_enabled
        meta["must_keep_floor"] = round(must_keep_floor, 6)
        meta["must_keep_raw_count"] = len(must_keep_raw)
        meta["must_keep_count"] = len(must_keep)
        min_keep = max(5, requested_top_k, min(len(must_keep), must_keep_cap))
        # Very low-confidence regime: keep a small minimum head for robustness.
        if max_score < 0.10:
            meta["mode"] = "very_low_confidence_median_guard"
            low_pool = cands[: min(len(cands), min_keep)]
            return _coverage_select(low_pool, limit=20), meta

        abs_floor = 0.20
        rel_floor = 0.72 * max_score
        keep_floor = max(abs_floor, rel_floor, median_floor + 0.02)

        filtered = [(s, p) for s, p in cands if float(s) >= keep_floor]

        # Flat-low regime: many similarly low candidates -> keep only a very small high-signal head.
        flat_low_mode = False
        if len(cands) >= 12:
            idx = min(10, len(cands) - 1)
            spread_head = max_score - float(cands[idx][0])
            if max_score < 0.50 and spread_head <= 0.06:
                flat_low_mode = True
                meta["flat_low_mode"] = True
                # Explicit stricter threshold for flat/low distributions.
                # Keep candidates only if score >= max(keep_floor, 0.75*top_score, top_score-0.02, 0.22)
                stricter_floor = max(keep_floor, 0.75 * max_score, max_score - 0.02, 0.22)
                filtered = [(s, p) for s, p in cands if float(s) >= stricter_floor]

        # Fallback to a small head if thresholds are too strict.
        if not filtered:
            fallback_n = min(len(cands), min_keep)
            filtered = cands[:fallback_n]
            meta["fallback_triggered"] = True

        # Build candidate pool: always include all median-or-above entries first.
        pool: List[Tuple[float, Precedent]] = list(must_keep)
        for s, p in filtered:
            pid = str(p.precedent_id)
            if pid in must_keep_ids:
                continue
            pool.append((s, p))

        # If pool is still small, backfill from head-ranked candidates.
        if len(pool) < min_keep:
            in_pool = {str(p.precedent_id) for _, p in pool}
            for s, p in cands:
                pid = str(p.precedent_id)
                if pid in in_pool:
                    continue
                pool.append((s, p))
                in_pool.add(pid)
                if len(pool) >= min_keep:
                    break

        if flat_low_mode:
            cap = min(12, max(8, requested_top_k))
        else:
            cap = min(80, max(24, requested_top_k * 4))
        pool = pool[: min(cap, len(pool))]
        meta["pool_size"] = len(pool)

        # Final stage:
        # Always enforce coverage within the high-score pool, then return
        # selected items sorted by original score descending.
        limit = 20
        head = pool[: min(limit, len(pool))]
        gaps = [float(head[i][0]) - float(head[i + 1][0]) for i in range(len(head) - 1)]
        all_large_gap_mode = bool(gaps) and all(g >= 0.10 for g in gaps)
        meta["all_large_gap_mode"] = all_large_gap_mode

        selected = _coverage_select(pool, limit=limit)
        meta["selection_mode"] = "coverage_then_score_sort"
        meta["selected_limit"] = 20
        return selected, meta

    def _comp_signature(comp: Dict[str, Any]) -> str:
        c = comp or {}
        salt = str(c.get("salt", "")).strip()
        solvents = sorted(_extract_solvent_names(c))
        additives = sorted(_extract_additive_names(c))
        parts: List[str] = []
        if salt:
            parts.append(f"salt:{salt}")
        if solvents:
            parts.append(f"solvents:{'|'.join(solvents)}")
        if additives:
            parts.append(f"additives:{'|'.join(additives)}")
        return "; ".join(parts) if parts else "none"

    # Retrieve on compact decision-relevant text, then rerank with structured matching.
    chunks = [
        Chunk(
            chunk_id=p.precedent_id,
            paper_title=p.paper_title,
            doi=p.doi,
            formulation_id=f"{p.before_id}->{p.after_id}",
            role="precedent",
            property_name=p.property_name,
            text=(
                f"scope={p.property_name}\n"
                f"rl_action={p.rl_action_type}\n"
                f"action_types={'|'.join(p.action_types)}\n"
                f"objective_tags={'|'.join(p.objective_tags)}\n"
                f"before_components={_comp_signature(p.before_comp)}\n"
                f"after_components={_comp_signature(p.after_comp)}\n"
                f"conditions_flags={json.dumps(detect_condition_flags(p.conditions), ensure_ascii=False)}\n"
                f"outcome={p.outcome}\n"
                f"polarity={p.polarity}"
            ),
        )
        for p in precedents
    ]
    retriever = TfIdfRetriever()
    retriever.fit(chunks)

    _ = top_k
    scored = retriever.search(build_decision_query_text(query), top_k=max(1, len(precedents)))
    text_score_by_id = {chunk.chunk_id: score for score, chunk in scored}
    semantic_score_by_id = _embedding_search_scores(build_decision_query_text(query), chunks)

    ctx_flag_query = infer_flags_from_context_tags(query.context_tags)
    ctx_flag_query_conditions = detect_condition_flags(get_formulation_conditions(query.current_formulation))
    merged_flags = {k: (ctx_flag_query.get(k, False) or ctx_flag_query_conditions.get(k, False)) for k in ctx_flag_query}
    ft_comp = (query.current_formulation or {}).get("composition", {}) or {}
    ft_salt = str(ft_comp.get("salt", "")).strip().lower()
    ft_family = detect_solvent_family(ft_comp)
    family_tables, family_indexes = prepare_family_indexes_for_formulation(
        current_formulation=query.current_formulation,
        base_dir=Path(__file__).resolve().parent,
        model=DEFAULT_FAMILY_CLASSIFIER_MODEL,
    )
    ft_component_families = collect_component_families(ft_comp, family_indexes)
    query_action_rl = normalize_action_to_rl_space(query.action_strategy or query.action_type)

    # Layer 1: action matching in the canonical RL action space.
    exact_action_matches: List[Precedent] = []
    related_action_matches: List[Precedent] = []
    for p in precedents:
        p_action_rl = p.rl_action_type or "unknown_action"
        exact_match = (query_action_rl == p_action_rl)
        related_match = actions_related(query.action_type, p.action_types)
        if query_action_rl == "unknown_action":
            exact_action_matches.append(p)
            continue
        if exact_match:
            exact_action_matches.append(p)
        elif related_match:
            related_action_matches.append(p)

    action_screened: List[Precedent]
    min_exact = max(8, int(top_k) * 2)
    action_screen_mode = "fallback_all"
    if exact_action_matches and len(exact_action_matches) >= min_exact:
        action_screened = list(exact_action_matches)
        action_screen_mode = "exact_only"
    elif exact_action_matches:
        action_screened = list(exact_action_matches) + list(related_action_matches)
        action_screen_mode = "exact_plus_related"
    else:
        action_screened = list(related_action_matches)
        action_screen_mode = "related_only"
    if not action_screened:
        action_screened = list(precedents)
        action_screen_mode = "fallback_all"

    ranked: List[Tuple[float, Precedent]] = []
    ft_comp = (query.current_formulation or {}).get("composition", {}) or {}
    for p in action_screened:
        text_score = text_score_by_id.get(p.precedent_id, 0.0)
        semantic_score = semantic_score_by_id.get(p.precedent_id, 0.0)
        comp_sim = max(
            formulation_similarity_score(ft_comp, p.after_comp or {}),
            formulation_similarity_score(ft_comp, p.before_comp or {}),
        )
        rule_score = 0.0

        p_after_salt = str((p.after_comp or {}).get("salt", "")).strip().lower()
        p_before_salt = str((p.before_comp or {}).get("salt", "")).strip().lower()
        if ft_salt and (ft_salt == p_after_salt or ft_salt == p_before_salt):
            rule_score += 0.12

        p_family = detect_solvent_family(p.after_comp)
        if ft_family != "unknown" and ft_family == p_family:
            rule_score += 0.08

        # Layer 3: family-based weighting on screened candidates
        p_component_families = collect_component_families(p.after_comp or {}, family_indexes)
        shared_solvent_families = ft_component_families["solvents"] & p_component_families["solvents"]
        shared_additive_families = ft_component_families["additives"] & p_component_families["additives"]
        shared_salt_families = ft_component_families["salts"] & p_component_families["salts"]
        rule_score += min(0.16, 0.04 * len(shared_solvent_families))
        rule_score += min(0.12, 0.04 * len(shared_additive_families))
        rule_score += min(0.08, 0.08 * len(shared_salt_families))

        p_flags = detect_condition_flags(p.conditions)
        overlap_flags = 0
        for k, v in merged_flags.items():
            if v and p_flags.get(k, False):
                overlap_flags += 1
        rule_score += min(0.2, 0.07 * overlap_flags)

        if p.action_explicitly_explored:
            rule_score += 0.08
        if p.after_role.lower() == "optimized":
            rule_score += 0.05
        if p.polarity == "positive":
            rule_score += 0.06
        elif p.polarity == "negative":
            rule_score -= 0.04

        # Hybrid rerank: lexical + semantic + composition + rule priors.
        total = (0.12 * text_score) + (0.23 * semantic_score) + (0.50 * comp_sim) + rule_score
        ranked.append((total, p))

    ranked.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate semantic repeats by (before_id, after_id). Keep highest-score entry.
    # Then build a score-based pool and run coverage-aware selection (up to 20).
    deduped: List[Tuple[float, Precedent]] = []
    seen_pairs = set()
    for score, p in ranked:
        pair = (p.before_id, p.after_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        deduped.append((score, p))

    keep_all_exact_only = action_screen_mode == "exact_only"
    truncated, truncate_meta = _adaptive_truncate(
        deduped,
        max(1, int(top_k)),
        keep_all=keep_all_exact_only,
    )
    print(
        "[RAG layer1] "
        f"action_screen_mode={action_screen_mode}, "
        f"exact_action_matches={len(exact_action_matches)}, "
        f"related_action_matches={len(related_action_matches)}, "
        f"screened_candidates={len(action_screened)}, "
        f"deduped_candidates={len(deduped)}, "
        f"returned={len(truncated)}, "
        f"selection_mode={truncate_meta.get('selection_mode', 'unknown')}, "
        f"mode={truncate_meta.get('mode', 'normal')}"
    )
    return truncated


def build_tendencies(
    selected_precedents: List[Precedent],
    ft_conditions: List[str],
) -> Dict[str, Any]:
    if not selected_precedents:
        return {
            "condition_bucket": {},
            "top_action_patterns": [],
            "top_component_patterns": [],
        }

    cond_flags_ft = detect_condition_flags(ft_conditions)
    action_counter = Counter()
    component_counter = Counter()

    # Condition-aware filter: use precedents with overlapping flags; if too few, use all.
    filtered = []
    for p in selected_precedents:
        pf = detect_condition_flags(p.conditions)
        overlap = any(cond_flags_ft.get(k, False) and pf.get(k, False) for k in cond_flags_ft)
        if overlap:
            filtered.append(p)
    if len(filtered) < max(3, len(selected_precedents) // 3):
        filtered = selected_precedents

    for p in filtered:
        for a in p.action_types:
            action_counter[a] += 1

        b_sol = set((p.before_comp.get("solvents", {}) or {}).keys())
        a_sol = set((p.after_comp.get("solvents", {}) or {}).keys())
        for s in sorted(a_sol - b_sol):
            component_counter[f"add_new_solvent:{s}"] += 1
        for s in sorted(b_sol - a_sol):
            component_counter[f"remove_solvent:{s}"] += 1

        b_add = set((p.before_comp.get("additives", {}) or {}).keys())
        a_add = set((p.after_comp.get("additives", {}) or {}).keys())
        changed_additives = sorted((a_add - b_add) | (b_add - a_add))
        if not changed_additives and b_add == a_add and (p.before_comp.get("additives", {}) or {}) != (p.after_comp.get("additives", {}) or {}):
            changed_additives = sorted(a_add)
        for a in changed_additives:
            component_counter[f"additive_change:{a}"] += 1

        b_salt = str(p.before_comp.get("salt", ""))
        a_salt = str(p.after_comp.get("salt", ""))
        if b_salt and a_salt and b_salt != a_salt:
            key = "add_new_salt" if normalize_action_type(p.rl_action_type) == "add_new_salt" else "salt_replace"
            component_counter[f"{key}:{b_salt}->{a_salt}"] += 1

    return {
        "condition_bucket": cond_flags_ft,
        "top_action_patterns": [{"pattern": k, "count": v} for k, v in action_counter.most_common(6)],
        "top_component_patterns": [{"pattern": k, "count": v} for k, v in component_counter.most_common(8)],
    }


def truncate_text(text: Any, limit: int = 220) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[: max(limit - 3, 0)].rstrip() + "..."


def _as_map(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def slim_composition_for_llm(comp: Dict[str, Any]) -> Dict[str, Any]:
    src = comp or {}
    out: Dict[str, Any] = {}

    salt = src.get("salt")
    if salt not in (None, ""):
        out["salt"] = salt

    salt_conc = src.get("salt_concentration")
    if salt_conc not in (None, ""):
        out["salt_concentration"] = salt_conc

    solvents = src.get("solvents")
    if isinstance(solvents, dict) and solvents:
        out["solvents"] = solvents
    else:
        # Support schema where solvents are stored as list entries with smiles.
        sws = src.get("solvents_with_smiles", [])
        if isinstance(sws, list) and sws:
            rebuilt: Dict[str, Any] = {}
            for item in sws:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                ratio = item.get("ratio", "")
                if name:
                    rebuilt[name] = ratio
            if rebuilt:
                out["solvents"] = rebuilt

    additives = src.get("additives")
    if isinstance(additives, dict) and additives:
        out["additives"] = additives
    elif isinstance(additives, list) and additives:
        out["additives"] = additives
    elif additives not in (None, "", {}, []):
        out["additives"] = additives

    return out


def compute_comp_delta(before_comp: Dict[str, Any], after_comp: Dict[str, Any]) -> Dict[str, Any]:
    b = before_comp or {}
    a = after_comp or {}

    b_salt = str(b.get("salt", "")).strip()
    a_salt = str(a.get("salt", "")).strip()
    b_sc = str(b.get("salt_concentration", "")).strip()
    a_sc = str(a.get("salt_concentration", "")).strip()

    b_sol = _as_map(b.get("solvents"))
    a_sol = _as_map(a.get("solvents"))
    b_add = _as_map(b.get("additives"))
    a_add = _as_map(a.get("additives"))

    b_sol_keys = set(str(k) for k in b_sol.keys())
    a_sol_keys = set(str(k) for k in a_sol.keys())
    b_add_keys = set(str(k) for k in b_add.keys())
    a_add_keys = set(str(k) for k in a_add.keys())

    solvent_changed_ratio = sorted(
        [k for k in (b_sol_keys & a_sol_keys) if str(b_sol.get(k, "")) != str(a_sol.get(k, ""))]
    )
    additive_changed_ratio = sorted(
        [k for k in (b_add_keys & a_add_keys) if str(b_add.get(k, "")) != str(a_add.get(k, ""))]
    )

    return {
        "salt_change": {
            "changed": b_salt != a_salt,
        },
        "salt_concentration_change": {
            "changed": b_sc != a_sc,
        },
        "solvent_change": {
            "changed": b_sol != a_sol,
            "added": sorted(a_sol_keys - b_sol_keys),
            "removed": sorted(b_sol_keys - a_sol_keys),
            "ratio_changed": solvent_changed_ratio,
        },
        "additive_change": {
            "changed": b_add != a_add,
            "added": sorted(a_add_keys - b_add_keys),
            "removed": sorted(b_add_keys - a_add_keys),
            "ratio_changed": additive_changed_ratio,
        },
    }


def build_compact_precedent_row(score: float, p: Precedent) -> Dict[str, Any]:
    before_comp = slim_composition_for_llm(p.before_comp or {})
    after_comp = slim_composition_for_llm(p.after_comp or {})
    return {
        "precedent_id": p.precedent_id,
        "similarity_score": round(float(score), 4),
        "source": {
            "doi": p.doi,
        },
        "before_id": p.before_id,
        "after_id": p.after_id,
        "before": {
            "composition": before_comp,
        },
        "after": {
            "composition": after_comp,
        },
        "action_types": p.action_types,
        "rl_action_type": p.rl_action_type,
        "delta": compute_comp_delta(before_comp, after_comp),
        "conditions_flags": detect_condition_flags(p.conditions),
        "scope": p.property_name,
        "property": p.property_name,
        "polarity": p.polarity,
        "evaluated_properties": p.evaluated_properties,
        "explicitly_explored": bool(p.action_explicitly_explored),
        "primary_objective_tag": (p.objective_tags[0] if p.objective_tags else "unspecified_performance_objective"),
    }


def build_paper_prior_index(papers: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for paper in papers:
        meta = paper.get("paper_metadata", {}) or {}
        doi = str(meta.get("doi", "")).strip()
        if not doi:
            continue
        boundaries = paper.get("design_boundaries", {}) or {}
        exploration = paper.get("exploration_operations", {}) or {}
        conclusions_raw = paper.get("design_conclusions", []) or []
        conclusions: List[str] = []
        for item in conclusions_raw[:3]:
            txt = truncate_text(item, limit=160)
            if txt:
                conclusions.append(txt)
        index[doi] = {
            "doi": doi,
            "explored_dimensions": (exploration.get("explored_dimensions", []) or [])[:5],
            "not_explored": (exploration.get("not_explored", []) or [])[:5],
            "fixed_parameters": (boundaries.get("fixed_parameters", []) or [])[:6],
            "conclusion_brief": conclusions,
        }
    return index


def build_paper_priors_for_selected(
    selected_precedents: List[Precedent],
    paper_prior_index: Dict[str, Dict[str, Any]],
    limit: int = 12,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for p in selected_precedents:
        doi = str(p.doi or "").strip()
        if not doi or doi in seen:
            continue
        seen.add(doi)
        prior = dict(paper_prior_index.get(doi, {"doi": doi}))
        if len(out) < limit:
            out.append(prior)
    return out


def compact_hard_constraints_for_llm(hard: Dict[str, Any], full_formulation: Dict[str, Any]) -> Dict[str, Any]:
    comp = (full_formulation.get("composition", {}) or {}) if isinstance(full_formulation, dict) else {}
    solvents = _as_map(comp.get("solvents"))
    additives = _as_map(comp.get("additives"))
    salt_info = (hard.get("salt", {}) or {}) if isinstance(hard, dict) else {}
    action_info = (hard.get("action_type", {}) or {}) if isinstance(hard, dict) else {}
    query_scope = (hard.get("query_scope", {}) or {}) if isinstance(hard, dict) else {}
    return {
        "query_scope": {
            "target_property": query_scope.get("target_property", ""),
            "context_tags": query_scope.get("context_tags", []),
            "objective_primary": query_scope.get("objective_primary", "unspecified_performance_objective"),
        },
        "action_type": {
            "name": action_info.get("name", "unknown_action"),
            "risk_level": action_info.get("risk_level", "unknown"),
            "negative_ratio": action_info.get("negative_ratio"),
            "positive_ratio": action_info.get("positive_ratio"),
            "explicitly_explored_ratio": action_info.get("explicitly_explored_ratio"),
        },
        "salt": {
            "name": salt_info.get("name", ""),
            "salt_concentration_current": salt_info.get("salt_concentration_current", ""),
            "salt_concentration_approx_upper_bound_numeric": salt_info.get("salt_concentration_approx_upper_bound_numeric"),
        },
        "solvents": sorted(str(k) for k in solvents.keys()),
        "additives": sorted(str(k) for k in additives.keys()),
    }


def compact_tendencies_for_llm(tendencies: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "condition_bucket": tendencies.get("condition_bucket", {}),
        "top_action_patterns": [],
        "top_component_patterns": [],
    }
    for item in (tendencies.get("top_action_patterns", []) or [])[:5]:
        if isinstance(item, dict):
            out["top_action_patterns"].append({"pattern": item.get("pattern"), "count": item.get("count")})
    for item in (tendencies.get("top_component_patterns", []) or [])[:8]:
        if isinstance(item, dict):
            out["top_component_patterns"].append({"pattern": item.get("pattern"), "count": item.get("count")})
    return out


def build_action_rag_payload(
    papers: List[Dict[str, Any]],
    query: DecisionQuery,
    top_k: int,
) -> Dict[str, Any]:
    query.action_type = normalize_action_type(query.action_type)
    precedents = build_precedents(papers)
    comp_stats = extract_component_values_from_formulations(papers)
    action_stats = summarize_action_stats(precedents)
    action_explicitness = summarize_action_exploration_explicitness(precedents)
    selected = retrieve_precedents_for_action(precedents, query, top_k=top_k)

    selected_precedents = [p for _, p in selected]
    paper_prior_index = build_paper_prior_index(papers)
    precedent_rows = []
    precedent_rows_compact = []
    for score, p in selected:
        positive_outcomes = [p.outcome] if p.polarity == "positive" and p.outcome else []
        precedent_rows.append(
            {
                "precedent_id": p.precedent_id,
                "similarity_score": round(float(score), 4),
                "paper_title": p.paper_title,
                "doi": p.doi,
                "action_types": p.action_types,
                "rl_action_type": p.rl_action_type,
                "formulation_id": p.after_id,
                "role": p.after_role,
                "composition": p.after_comp,
                "positive_outcomes": positive_outcomes,
                "before": {
                    "formulation_id": p.before_id,
                    "composition": p.before_comp,
                },
                "after": {
                    "formulation_id": p.after_id,
                    "composition": p.after_comp,
                },
                "scope": p.property_name,
                "property": p.property_name,
                "objective_tags": p.objective_tags,
                "evaluated_properties": p.evaluated_properties,
                "conditions": p.conditions,
                "outcome": p.outcome,
                "outcome_polarity": p.polarity,
                "action_explicitly_explored": p.action_explicitly_explored,
            }
        )
        precedent_rows_compact.append(build_compact_precedent_row(score, p))

    hard_constraints = build_hard_constraints(query, comp_stats, action_stats, action_explicitness)
    tendencies = build_tendencies(selected_precedents, get_formulation_conditions(query.current_formulation))
    paper_priors = build_paper_priors_for_selected(selected_precedents, paper_prior_index, limit=12)

    payload = {
        "input": {
            "current_formulation": query.current_formulation,
            "action_strategy": query.action_strategy,
            "action_type": query.action_type,
            "target_property": query.target_property,
            "context_tags": query.context_tags,
            "objective_primary": query.objective_primary,
        },
        "precedent_selection_policy": {
            "screening_layers": [
                "rl_action_exact_or_related",
            ],
            "ranking": "composition_similarity_dominant_weighted_sort",
            "truncation": "adaptive_score_threshold_and_cap",
        },
        "hard_constraints": hard_constraints,
        "precedents": precedent_rows,
        "precedents_compact": precedent_rows_compact,
        "paper_priors": paper_priors,
        "tendencies": tendencies,
    }
    payload["rag_context_text"] = build_rag_context_text(payload)
    payload["llm_input"] = build_llm_input_payload(payload)
    return payload


def build_llm_input_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    inp = payload.get("input", {}) or {}
    hard_constraints = payload.get("hard_constraints", {}) or {}
    full_formulation = inp.get("current_formulation", {}) or {}
    return {
        "full_formulation": full_formulation,
        "objective": inp.get("objective_primary", "unspecified_performance_objective"),
        "action_strategy": inp.get("action_strategy", "unknown_action"),
        "rag_information": {
            "precedent_selection_policy": payload.get("precedent_selection_policy", {}),
            "hard_constraints": compact_hard_constraints_for_llm(hard_constraints, full_formulation),
            "precedents": payload.get("precedents_compact", []),
            "tendencies": compact_tendencies_for_llm(payload.get("tendencies", {}) or {}),
        },
    }


def summarize_decision_in_plain_english(payload: Dict[str, Any]) -> str:
    """
    Generate a deterministic, human-readable English summary
    of an RL -> RAG -> LLM formulation decision.
    """
    objective_map = {
        "ion_transport_temperature_dependence": "improving ion transport across temperature",
        "viscosity_transport_coupling": "optimizing viscosity-transport coupling",
        "solvation_association_temperature_stability": "enhancing solvation and interfacial stability",
        "unspecified_performance_objective": "overall electrolyte performance optimization",
    }

    objective_key = payload.get("objective", "unspecified_performance_objective")
    objective_text = objective_map.get(objective_key, objective_key)

    action_strategy = payload.get("action_strategy", "unknown_action")
    action_text = str(action_strategy).replace("_", " ")

    proposal = payload.get("proposal", {})
    if isinstance(proposal, str):
        raw = proposal.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        try:
            proposal = json.loads(raw)
        except Exception:
            proposal = {}

    parameters = proposal.get("parameters", {}) if isinstance(proposal, dict) else {}
    if not isinstance(parameters, dict):
        parameters = {}

    salt = payload.get("hard_constraints", {}).get("salt", {}).get("name") or "LiPF6"
    solvents_map = payload.get("full_formulation", {}).get("composition", {}).get("solvents", {}) or {}
    if solvents_map:
        solvent_parts = [f"{k}({v})" for k, v in solvents_map.items()]
        solvents = " + ".join(solvent_parts)
    else:
        solvents = "carbonate solvent blend"

    uncertainty = "unknown"
    exploratory = False
    if isinstance(proposal, dict):
        uncertainty = str(proposal.get("uncertainty_level", "unknown"))
        exploratory = bool(proposal.get("is_exploratory", False))

    risk_map = {
        "low": "low-risk",
        "medium": "moderate-risk",
        "high": "high-risk",
    }
    risk_text = risk_map.get(uncertainty, uncertainty)
    exploration_text = "exploratory" if exploratory else "non-exploratory"

    def _action_detail_text(strategy: str, params: Dict[str, Any]) -> str:
        strategy = normalize_action_type(strategy)
        if strategy == "additive_change":
            additive = params.get("additive") or params.get("additive_name") or "an additive"
            old_amount = params.get("from_fraction") or params.get("from_amount")
            new_amount = params.get("to_fraction") or params.get("to_amount") or params.get("amount")
            change_kind = params.get("change") or params.get("change_type")
            if change_kind == "remove":
                return f"removing {additive}"
            if change_kind == "replace":
                replacement = params.get("replacement_additive") or params.get("to_additive") or "another additive"
                return f"replacing additive {additive} with {replacement}"
            if old_amount or new_amount:
                return f"adjusting additive {additive} from {old_amount or 'the current level'} to {new_amount or 'a new level'}"
            return f"changing additive package around {additive}"
        if strategy == "salt_replace":
            old_salt = params.get("from_salt") or params.get("salt_from") or salt
            new_salt = params.get("to_salt") or params.get("salt_to")
            # Backward-compatible parsing for proposals that return only a combined
            # "salt" field, e.g. "LiPF6 + LiDFOB".
            if not new_salt:
                salt_field = str(params.get("salt", "")).strip()
                if salt_field:
                    parts = [p.strip() for p in re.split(r"\+|/", salt_field) if p.strip()]
                    if len(parts) >= 2:
                        if not params.get("from_salt"):
                            old_salt = parts[0]
                        new_salt = " + ".join(parts[1:])
                    else:
                        new_salt = salt_field
            if not new_salt:
                new_salt = "a replacement salt"
            return f"replacing salt from {old_salt} to {new_salt}"
        if strategy == "salt_concentration_shift":
            old_c = params.get("from_concentration", "the current concentration")
            new_c = params.get("to_concentration", "a new concentration")
            return f"shifting salt concentration from {old_c} to {new_c}"
        if strategy == "add_new_salt":
            old_salt = params.get("from_salt") or params.get("salt_from") or salt
            new_salt = params.get("to_salt") or params.get("salt_to") or params.get("added_salt") or "an additional salt"
            return f"adding new salt {new_salt} alongside {old_salt}"
        if strategy == "solvent_replace":
            old_s = params.get("from_solvent", "an existing solvent")
            new_s = params.get("to_solvent", "a replacement solvent")
            return f"replacing solvent {old_s} with {new_s}"
        if strategy == "add_new_solvent":
            solvent = params.get("solvent", "a solvent")
            fraction = params.get("fraction", "a small fraction of")
            return f"adding {fraction} {solvent}"
        if strategy == "remove_solvent":
            solvent = params.get("solvent", "a solvent")
            return f"removing {solvent}"
        if strategy == "solvent_concentration_shift":
            ratio_from = params.get("from_ratio", "the current ratio")
            ratio_to = params.get("to_ratio", "a new ratio")
            return f"shifting solvent ratio from {ratio_from} to {ratio_to}"
        return "making a constrained formulation modification"

    action_detail = _action_detail_text(str(action_strategy), parameters if isinstance(parameters, dict) else {})

    summary = (
        f"Under the objective of {objective_text}, "
        f"RL restricts the search to the '{action_text}' direction. "
        f"Based on literature-informed priors, the LLM proposes {action_detail} "
        f"for the current {salt} / {solvents} electrolyte, "
        f"as a {risk_text}, {exploration_text}, but well-supported next step."
    )
    return summary


def build_rag_context_text(payload: Dict[str, Any]) -> str:
    hard = payload.get("hard_constraints", {})
    precedents = payload.get("precedents", []) or []
    tendencies = payload.get("tendencies", {})
    action = (hard.get("action_type", {}) or {}).get("name", "unknown")
    target_property = ((payload.get("input", {}) or {}).get("target_property", "unknown"))

    lines = [
        f"Action-RAG Summary for action_type={action}, target_property={target_property}",
        "",
        "[Hard constraints]",
        json.dumps(hard, ensure_ascii=False),
        "",
        "[Precedents]",
    ]
    for i, p in enumerate(precedents[:8], start=1):
        lines.append(
            f"{i}. {p.get('before', {}).get('formulation_id')} -> {p.get('after', {}).get('formulation_id')}; "
            f"actions={p.get('action_types')}; "
            f"role={p.get('role')}; "
            f"conditions={p.get('conditions')}; "
            f"outcome={p.get('outcome')}; "
            f"polarity={p.get('outcome_polarity')}"
        )
    lines.extend(
        [
            "",
            "[Tendencies]",
            json.dumps(tendencies, ensure_ascii=False),
        ]
    )
    return "\n".join(lines)


def format_context(results: List[Tuple[float, Chunk]]) -> str:
    blocks = []
    for rank, (score, chunk) in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Retrieved #{rank}] score={score:.4f}",
                    f"paper_title: {chunk.paper_title}",
                    f"doi: {chunk.doi}",
                    f"formulation_id: {chunk.formulation_id}",
                    f"role: {chunk.role}",
                    f"scope: {chunk.property_name}",
                    f"evidence:\n{chunk.text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def generate_answer_with_openai(question: str, context: str, model: str = "gpt-5") -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are an electrolyte literature assistant. "
        "Answer strictly from retrieved evidence. "
        "If evidence is insufficient, explicitly say so."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{context}\n\n"
        "Please provide:\n"
        "1) Direct answer\n"
        "2) Key supporting formulations and outcomes\n"
        "3) Any uncertainty or missing evidence\n"
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **_temp_kwargs(model, 0.1),
    )

    return resp.output_text.strip()


def generate_proposal_with_openai(
    payload: Dict[str, Any],
    model: str = "gpt-5",
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

    def _request(prompt_system: str, prompt_user: str) -> str:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
            ],
            **_temp_kwargs(model, max(0.0, min(float(temperature), 1.0))),
        )
        return (resp.output_text or "").strip()

    user_prompt = (
        PROPOSAL_USER_PROMPT_PREFIX
        + json.dumps(payload.get("llm_input", {}), ensure_ascii=False)
        + "\n"
    )
    return _request(PROPOSAL_SYSTEM_PROMPT, user_prompt)


def fallback_extractive_answer(question: str, results: List[Tuple[float, Chunk]]) -> str:
    if not results:
        return "No relevant literature snippets were retrieved."

    lines = [f"Question: {question}", "", "Based on the retrieved evidence:"]
    for i, (score, chunk) in enumerate(results, start=1):
        outcome_match = re.search(r"Outcome:\s*(.*)", chunk.text, flags=re.DOTALL)
        outcome = ""
        if outcome_match:
            outcome = " ".join(part.strip() for part in outcome_match.group(1).splitlines() if part.strip())
        lines.append(
            f"{i}. [{chunk.formulation_id} | {chunk.property_name} | score={score:.3f}] {outcome}"
        )
    lines.append("")
    lines.append("Tip: set OPENAI_API_KEY to enable automatic LLM answers with more complete responses.")
    return "\n".join(lines)


def load_papers(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("literature database must be structured as list[paper]")
    return data


def _build_index_cache_metadata(data_path: Path) -> Dict[str, Any]:
    resolved = data_path.resolve()
    stat = resolved.stat()
    return {
        "version": RAG_INDEX_CACHE_VERSION,
        "data_path": str(resolved),
        "data_mtime_ns": int(stat.st_mtime_ns),
        "data_size": int(stat.st_size),
    }


def build_or_load_index(data_path: Path, cache_path: Optional[Path]) -> TfIdfRetriever:
    retriever = TfIdfRetriever()
    expected_metadata = _build_index_cache_metadata(data_path)

    if cache_path and cache_path.exists():
        try:
            cached_metadata = retriever.load(cache_path)
            if (
                cached_metadata.get("version") == expected_metadata["version"]
                and cached_metadata.get("data_path") == expected_metadata["data_path"]
                and int(cached_metadata.get("data_mtime_ns", -1)) == expected_metadata["data_mtime_ns"]
                and int(cached_metadata.get("data_size", -1)) == expected_metadata["data_size"]
            ):
                return retriever
            print(f"[RAG] Rebuilding stale cache index: {cache_path}")
        except Exception as e:
            print(f"[RAG] Rebuilding unreadable cache index: {cache_path} ({e})")

    papers = load_papers(data_path)
    chunks = build_chunks(papers)
    retriever.fit(chunks)

    if cache_path:
        retriever.save(cache_path, metadata=expected_metadata)

    return retriever


def answer_question(
    retriever: TfIdfRetriever,
    question: str,
    top_k: int,
    llm_model: str,
    use_llm: bool,
) -> Tuple[str, str]:
    results = retriever.search(question, top_k=top_k)
    context = format_context(results)

    if use_llm:
        try:
            answer = generate_answer_with_openai(question=question, context=context, model=llm_model)
            return answer, context
        except Exception as e:
            msg = f"[LLM call failed; fell back to retrieval summary] {e}"
            answer = msg + "\n\n" + fallback_extractive_answer(question, results)
            return answer, context

    return fallback_extractive_answer(question, results), context


def parse_current_formulation(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if args.current_formulation:
        return json.loads(args.current_formulation)
    if args.current_formulation_file:
        with args.current_formulation_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def parse_decision_query(args: argparse.Namespace) -> DecisionQuery:
    if not args.query_file:
        raise ValueError("Decision mode requires --query-file")
    with args.query_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # query-file accepts either full formulation object directly
    # or wrapper {"current_formulation": {...}}.
    if isinstance(data, dict) and "current_formulation" in data and isinstance(data.get("current_formulation"), dict):
        formulation = normalize_query_formulation(data.get("current_formulation", {}))
    else:
        formulation = normalize_query_formulation(data if isinstance(data, dict) else {})

    objective_primary = normalize_objective_primary(args.objective)
    action_strategy = args.action_strategy if args.action_strategy in ACTION_SPACE_WITH_UNKNOWN else "unknown_action"
    action_type = strategy_to_action_type(action_strategy)
    target_property = objective_to_target_property(objective_primary)
    context_tags = infer_context_tags_from_formulation(formulation)

    return DecisionQuery(
        current_formulation=formulation,
        action_strategy=action_strategy,
        action_type=action_type,
        target_property=target_property,
        context_tags=context_tags,
        objective_primary=objective_primary,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG for literature_database.json")
    p.add_argument(
        "--mode",
        choices=["qa", "decision", "auto"],
        default="auto",
        help="Run mode: qa / decision / auto (infer from args)",
    )
    p.add_argument("--data", type=Path, default=Path("literature_database.json"), help="Path to literature database")
    p.add_argument("--cache-index", type=Path, default=Path("literature_rag_index.pkl"), help="Path to index cache")

    # QA mode args
    p.add_argument("--question", type=str, default=None, help="Single question to ask")
    p.add_argument("--interactive", action="store_true", help="Interactive QA mode")
    p.add_argument("--llm-mode", type=str, default="gpt-5", help="OpenAI model name, e.g. gpt-5.2 or gpt-4")
    p.add_argument("--llm-model", dest="llm_mode", type=str, help=argparse.SUPPRESS)
    p.add_argument("--no-llm", action="store_true", help="Disable LLM call and return extractive summary only")
    p.add_argument("--show-context", action="store_true", help="Print retrieved context")

    # Decision args
    p.add_argument(
        "--objective",
        type=str,
        choices=sorted(ALLOWED_OBJECTIVES),
        default="unspecified_performance_objective",
        help="Optimization objective category",
    )
    p.add_argument("--query-file", type=Path, default=None, help="JSON file for full current formulation")
    p.add_argument(
        "--action-strategy",
        type=str,
        choices=ACTION_SPACE_WITH_UNKNOWN,
        default="unknown_action",
        help="RL action strategy (canonical action space plus unknown_action)",
    )
    p.add_argument("--emit-action-rag", action="store_true", help="Emit structured action RAG payload as JSON")
    p.add_argument("--propose-with-llm", action="store_true", help=argparse.SUPPRESS)

    p.add_argument("--top-k", type=int, default=5, help="Number of chunks/precedents to retrieve")
    return p.parse_args()


def run_decision_mode(args: argparse.Namespace) -> None:
    if not args.query_file:
        raise ValueError("Decision mode requires --query-file")
    decision_query = parse_decision_query(args)
    papers = load_papers(args.data)
    payload = build_action_rag_payload(
        papers=papers,
        query=decision_query,
        top_k=args.top_k,
    )
    try:
        payload["proposal"] = generate_proposal_with_openai(payload, model=args.llm_mode)
    except Exception as e:
        payload["proposal_error"] = str(e)
    output_dir = Path.cwd()
    llm_input_path = output_dir / "llm_input.json"
    llm_output_path = output_dir / "llm_output.json"

    llm_input_payload = payload.get("llm_input", {})
    llm_output_payload = {
        "model": args.llm_mode,
        "objective": payload.get("input", {}).get("objective_primary"),
        "action_strategy": payload.get("input", {}).get("action_strategy"),
        "action_type": payload.get("input", {}).get("action_type"),
        "full_formulation": payload.get("input", {}).get("current_formulation"),
        "proposal": payload.get("proposal"),
        "proposal_error": payload.get("proposal_error"),
        "hard_constraints": payload.get("hard_constraints"),
        "precedents_used": payload.get("precedents", []),
        "tendencies": payload.get("tendencies"),
    }
    llm_output_payload["decision_summary_plain_english"] = summarize_decision_in_plain_english(llm_output_payload)

    llm_input_path.write_text(json.dumps(llm_input_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    llm_output_path.write_text(json.dumps(llm_output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(llm_output_payload["decision_summary_plain_english"])


def run_qa_mode(args: argparse.Namespace) -> None:
    retriever = build_or_load_index(args.data, args.cache_index)

    def _run_one(question: str) -> None:
        answer, context = answer_question(
            retriever=retriever,
            question=question,
            top_k=args.top_k,
            llm_model=args.llm_mode,
            use_llm=not args.no_llm,
        )
        print("\n=== Answer ===")
        print(answer)
        if args.show_context:
            print("\n=== Retrieved Context ===")
            print(context)

    if args.question:
        _run_one(args.question)
        return

    if args.interactive:
        print("RAG interactive mode. Enter q / quit to exit.")
        while True:
            q = input("\nQuestion> ").strip()
            if q.lower() in {"q", "quit", "exit"}:
                break
            if not q:
                continue
            _run_one(q)
        return

    print("For QA mode, provide --question or --interactive")


def main() -> None:
    args = parse_args()
    mode = args.mode
    if mode == "auto":
        if args.emit_action_rag or args.query_file or args.propose_with_llm:
            mode = "decision"
        else:
            mode = "qa"

    if mode == "decision":
        run_decision_mode(args)
        return

    run_qa_mode(args)


if __name__ == "__main__":
    main()
