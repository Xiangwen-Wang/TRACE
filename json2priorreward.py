from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


POSITIVE_KEYWORDS = [
    "improved",
    "higher",
    "enhanced",
    "suppressed",
    "reduced",
    "lower impedance",
    "stable",
    "uniform",
    "maintained",
    "extended",
    "highest",
    "favored",
    "outperformed",
    "better",
    "mitigated",
    "minimal impedance growth",
]

NEGATIVE_KEYWORDS = [
    "poor",
    "inferior",
    "degraded",
    "rapid decay",
    "increased impedance",
    "severe",
    "unstable",
    "short circuit",
    "worse",
    "deteriorates",
    "capacity fade",
    "lower coulombic efficiency",
]

TRADEOFF_HINTS = ["but", "however", "while", "although", "yet", "trade-off", "mixed"]

CARBONATE_TOKENS = ["ec", "emc", "dmc", "dec", "pc", "fec", "vc"]
ETHER_TOKENS = ["dme", "diglyme", "triglyme", "tetraglyme", "ether", "dioxolane", "dol", "thf"]
IONIC_TOKENS = ["ionic liquid", "pyr", "imidazolium", "fsi-ionic", "tfsi-ionic"]

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

ACTIONS = [
    "salt_replace",
    "salt_concentration_shift",
    "add_new_salt",
    "solvent_replace",
    "solvent_concentration_shift",
    "add_new_solvent",
    "remove_solvent",
    "additive_change",
]

LEGACY_ACTION_ALIASES = {
    "add_additive": "additive_change",
    "delete_additive": "additive_change",
}

SMALL_CHANGE_THRESHOLD = 0.05
OPENAI_ACTION_MODEL = "gpt-5.2"

LLM_ACTION_RULES = """You must classify the variant into one or more actions from this closed set:
- salt_replace
- salt_concentration_shift
- add_new_salt
- solvent_replace
- solvent_concentration_shift
- add_new_solvent
- remove_solvent
- additive_change

Rules:
1) additive_change: additive(s) are added/removed/changed compared to baseline.
2) salt_replace: old salt species is replaced by new salt species.
3) salt_concentration_shift: salt species unchanged, but salt concentration changes.
4) add_new_salt: old salt species still exists, new salt species is added, and salt concentration relative change is >= 5%.
5) solvent_replace: old solvent species is replaced by new solvent species.
6) solvent_concentration_shift: solvent species unchanged, but solvent ratio/concentration changes.
7) add_new_solvent: solvent species list changes by adding/removing solvent species.
8) remove_solvent: solvent species list changes by removing solvent species.

Threshold rule:
- If solvent composition change is < 5%, classify as additive_change.
- If salt concentration relative change is < 5%, classify as additive_change.

Multi-label requirements:
- A pair may have multiple actions across dimensions.
- At most one salt action from: salt_replace, salt_concentration_shift, add_new_salt.
- At most one solvent action from: solvent_replace, solvent_concentration_shift, add_new_solvent, remove_solvent.

Important output requirements:
- Return strict JSON only: {"actions": ["<action1>", "<action2>", ...]}
- Never output text outside JSON.
"""


def canonicalize_action_name(action: Any) -> Optional[str]:
    if not isinstance(action, str):
        return None
    if action in ACTIONS:
        return action
    return LEGACY_ACTION_ALIASES.get(action)


def canonicalize_action_list(value: Any) -> Set[str]:
    result: Set[str] = set()
    if isinstance(value, list):
        for item in value:
            mapped = canonicalize_action_name(item)
            if mapped:
                result.add(mapped)
        return result
    mapped = canonicalize_action_name(value)
    if mapped:
        result.add(mapped)
    return result

GENERIC_SOLVENT_KEYS = {
    "solvent",
    "solvents",
    "solvent_1",
    "solvent_2",
    "ratio",
    "electrolyte",
    "electrolyte solvent",
    "electrolyte solvents",
    "carbonate solvent(s)",
    "carbonate solvents",
    "linear carbonate (emc)",
    "linear carbonates (dec/emc/dmc)",
    "linear carbonates (e.g., dmc/emc/dec)",
    "other_carbonates",
    "inorganic_filler",
    "uncertain",
    "ec-based carbonate electrolyte",
    "unspecified carbonate/ether system",
}

COMPONENT_ALIASES_FILENAME = "component_aliases.json"

_DEFAULT_SOLVENT_ALIASES = {
    "ethylene carbonate": "EC",
    "dimethyl carbonate": "DMC",
    "ethyl methyl carbonate": "EMC",
    "ethylmethyl carbonate": "EMC",
    "ethylene methyl carbonate": "EMC",
    "diethyl carbonate": "DEC",
    "propylene carbonate": "PC",
    "fluoroethylene carbonate": "FEC",
    "vinylene carbonate": "VC",
    "triethyl phosphate": "TEP",
    "sulfolane": "SL",
    "succinonitrile": "SN",
    "adiponitrile": "ADN",
    "acetonitrile": "AN",
    "tetrahydropyran": "THP",
    "methyl propionate": "MP",
    "ethyl acetate": "EA",
    "diethyl ether": "DEE",
    "1,2-dimethoxyethane": "DME",
    "1,3-dioxolane": "DOL",
    "1,4-dioxane": "DX",
    "fluorobenzene": "FB",
    "valeronitrile": "VN",
    "butyrolactam": "BL",
    "water": "H2O",
}


def load_solvent_aliases(base_dir: Path) -> Dict[str, str]:
    aliases = {str(k).strip().lower(): str(v).strip() for k, v in _DEFAULT_SOLVENT_ALIASES.items() if str(k).strip() and str(v).strip()}
    path = base_dir / COMPONENT_ALIASES_FILENAME
    if not path.exists():
        return aliases
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return aliases
    if not isinstance(payload, dict):
        return aliases

    for canonical_name, synonym_values in payload.items():
        canonical = str(canonical_name).strip()
        if not canonical:
            continue
        synonyms: List[str] = []
        if isinstance(synonym_values, str):
            synonyms.append(synonym_values.strip())
        elif isinstance(synonym_values, list):
            synonyms.extend(str(item).strip() for item in synonym_values)
        for synonym in synonyms:
            if synonym:
                aliases[synonym.lower()] = canonical
    return aliases


SOLVENT_ALIASES = load_solvent_aliases(Path(__file__).resolve().parent / "supplementary")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).lower()


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{flatten_text(k)} {flatten_text(v)}" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(flatten_text(v) for v in value)
    return str(value)


def parse_float(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_ratio_value(text: Any) -> Optional[float]:
    s = str(text).strip()
    v = parse_float(s)
    if v is None:
        return None
    if "%" in s:
        return v / 100.0
    return v


def split_component_tokens(value: Any) -> Set[str]:
    s = normalize_key_token(value)
    if not s:
        return set()
    parts = re.split(r"\s*(?:\+|/|,|;|&|\band\b)\s*", s)
    return {p.strip() for p in parts if p.strip()}


def relative_change(base_value: Optional[float], var_value: Optional[float]) -> Optional[float]:
    if base_value is None or var_value is None:
        return None
    if abs(base_value) <= 1e-12:
        return None
    return abs(var_value - base_value) / abs(base_value)


def composition_total_change(base_values: Dict[str, str], var_values: Dict[str, str]) -> Optional[float]:
    base_norm = normalize_ratio_map(base_values)
    var_norm = normalize_ratio_map(var_values)
    keys = set(base_norm.keys()) | set(var_norm.keys())
    if not keys:
        return None

    total = 0.0
    for key in keys:
        b = parse_ratio_value(base_norm.get(key, "0"))
        v = parse_ratio_value(var_norm.get(key, "0"))
        if b is None or v is None:
            return None
        total += abs(v - b)
    # total variation distance in composition simplex.
    return total / 2.0


class UncertainActionResolver:
    """Resolve uncertain action assignments with GPT."""

    def __init__(self, cache_path: Optional[str] = None) -> None:
        self._client: Any = None
        self._cache: Dict[str, Set[str]] = {}
        self.cache_path = Path(cache_path) if cache_path else None
        if self.cache_path and self.cache_path.exists():
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self._cache = {k: set(v) for k, v in data.items()}
            print(f"Loaded {len(self._cache)} cached LLM action decisions from {self.cache_path}")
        self.resolved = 0
        self.failed = 0
        self.error_counts: Dict[str, int] = {}
        self.error_examples: List[str] = []

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for uncertain action resolution with GPT-5.2.")

        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "openai package is required. Install with `pip install openai`."
            ) from e

        self._client = OpenAI(api_key=api_key)
        return self._client

    def _build_payload(
        self,
        baseline: Dict[str, Any],
        formulation: Dict[str, Any],
        paper: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        meta = (paper or {}).get("paper_metadata", {}) if isinstance(paper, dict) else {}
        return {
            "paper": {
                "title": meta.get("title", ""),
                "doi": meta.get("doi", ""),
            },
            "baseline": {
                "id": baseline.get("id", ""),
                "composition": baseline.get("composition", {}) or {},
            },
            "variant": {
                "id": formulation.get("id", ""),
                "composition": formulation.get("composition", {}) or {},
            },
        }

    def _parse_actions(self, response: Any) -> Set[str]:
        raw = getattr(response, "output_text", None)
        if not raw:
            return set()
        raw = str(raw).strip()
        if not raw:
            return set()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return set()
        actions = canonicalize_action_list(data.get("actions"))
        if actions:
            return actions
        # Backward-compatible parse path for old cached responses.
        return canonicalize_action_list(data.get("action"))

    def classify(
        self,
        baseline: Dict[str, Any],
        formulation: Dict[str, Any],
        paper: Optional[Dict[str, Any]] = None,
    ) -> Set[str]:
        payload = self._build_payload(baseline, formulation, paper)
        cache_key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached:
                self.resolved += 1
            else:
                self.failed += 1
            return set(cached)

        client = self._ensure_client()
        try:
            response = client.responses.create(
                model=OPENAI_ACTION_MODEL,
                reasoning={"effort": "low"},
                input=[
                    {"role": "system", "content": LLM_ACTION_RULES},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "action_result",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "actions": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": ACTIONS},
                                    "minItems": 1,
                                },
                            },
                            "required": ["actions"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    }
                },
            )
            actions = self._parse_actions(response)
        except Exception as e:
            actions = set()
            err_name = type(e).__name__
            err_text = str(e).strip()
            self.error_counts[err_name] = self.error_counts.get(err_name, 0) + 1
            if len(self.error_examples) < 3:
                self.error_examples.append(f"{err_name}: {err_text}" if err_text else err_name)
        print("finish one LLM check")

        self._cache[cache_key] = set(actions)
        self.save_cache()
        if actions:
            self.resolved += 1
            return actions

        self.failed += 1
        return set()


def _resolver_save_cache(self) -> None:
    if self.cache_path:
        self.cache_path.write_text(
            json.dumps({k: sorted(v) for k, v in self._cache.items()}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )


UncertainActionResolver.save_cache = _resolver_save_cache  # type: ignore[attr-defined]


def state0_signature_from_row(state0: Dict[str, Any]) -> str:
    comp = (state0 or {}).get("composition", {}) or {}
    solvents_raw = comp.get("solvents_with_smiles", []) or []
    solvents_map: Dict[str, str] = {}
    if isinstance(solvents_raw, list):
        for item in solvents_raw:
            if not isinstance(item, dict):
                continue
            name = normalize_text(item.get("name", "")).strip()
            ratio = normalize_text(item.get("ratio", "")).strip()
            if name:
                solvents_map[name] = ratio

    additives_raw = comp.get("additives", {}) or {}
    additives_pairs: List[Tuple[str, str]] = []
    if isinstance(additives_raw, dict):
        additives_pairs = sorted(
            (
                normalize_text(k).strip(),
                normalize_text(v).strip(),
            )
            for k, v in additives_raw.items()
            if normalize_text(k).strip()
        )

    payload = {
        "salt": normalize_text(comp.get("salt", "")).strip(),
        "salt_concentration": normalize_text(comp.get("salt_concentration", "")).strip(),
        "solvents": sorted(
            (normalize_text(k).strip(), normalize_text(v).strip())
            for k, v in normalize_ratio_map(solvents_map).items()
            if normalize_text(k).strip()
        ),
        "additives": additives_pairs,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def state0_signature_from_baseline(
    baseline_comp: Dict[str, Any],
    solvent_standardizer: Optional["SolventStandardizer"] = None,
    codebook_additive_keys: Optional[Set[str]] = None,
) -> str:
    additives = collect_additives_for_state0(
        baseline_comp,
        codebook_additive_keys=codebook_additive_keys,
    )
    solvents_raw = (
        solvent_standardizer.standardize_map(
            baseline_comp.get("solvents", {}) or {},
            keep_unresolved=True,
        )
        if solvent_standardizer
        else (baseline_comp.get("solvents", {}) or {})
    )
    solvents = normalize_ratio_map(solvents_raw)
    payload = {
        "salt": normalize_text(baseline_comp.get("salt", "")).strip(),
        "salt_concentration": normalize_text(baseline_comp.get("salt_concentration", "")).strip(),
        "solvents": sorted(
            (normalize_text(k).strip(), normalize_text(v).strip())
            for k, v in solvents.items()
            if normalize_text(k).strip()
        ),
        "additives": sorted(
            (
                normalize_text(k).strip(),
                normalize_text(v).strip(),
            )
            for k, v in additives.items()
            if normalize_text(k).strip()
        ),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class CodebookAdditiveMatcher:
    def __init__(self) -> None:
        self._doi_to_additive_keys: Dict[str, Set[str]] = {}
        self._title_to_additive_keys: Dict[str, Set[str]] = {}

    def load(self, path: str) -> int:
        data = load_json_relaxed(path)
        loaded = 0
        if not isinstance(data, list):
            return loaded

        for row in data:
            if not isinstance(row, dict):
                continue

            additive_names: Set[str] = set()

            entities = row.get("entities", {}) or {}
            entity_additives = entities.get("additives", []) or []
            if isinstance(entity_additives, list):
                for item in entity_additives:
                    if isinstance(item, dict) and item.get("name"):
                        additive_names.add(str(item.get("name")))
                    elif isinstance(item, str):
                        additive_names.add(item)

            formulation = row.get("formulation", {}) or {}
            additive_fraction = formulation.get("additive_fraction", {}) or {}
            if isinstance(additive_fraction, dict):
                for name in additive_fraction.keys():
                    additive_names.add(str(name))

            additive_keys: Set[str] = set()
            for name in additive_names:
                additive_keys.update(additive_name_match_keys(name))
            additive_keys = {k for k in additive_keys if k}
            if not additive_keys:
                continue

            raw_ref = row.get("raw_ref", {}) or {}
            doi = normalize_doi(raw_ref.get("doi", ""))
            title = normalize_title(raw_ref.get("title", ""))

            if doi:
                self._doi_to_additive_keys.setdefault(doi, set()).update(additive_keys)
            if title:
                self._title_to_additive_keys.setdefault(title, set()).update(additive_keys)
            loaded += 1

        return loaded

    def get_additive_keys_for_paper(self, paper: Dict[str, Any]) -> Set[str]:
        meta = paper.get("paper_metadata", {}) or {}
        doi = normalize_doi(meta.get("doi", ""))
        title = normalize_title(meta.get("title", ""))

        result: Set[str] = set()
        if doi and doi in self._doi_to_additive_keys:
            result.update(self._doi_to_additive_keys[doi])
        if title and title in self._title_to_additive_keys:
            result.update(self._title_to_additive_keys[title])
        return result


class BackupActionLookup:
    def __init__(self) -> None:
        self._by_doi: Dict[Tuple[str, str, str], Set[str]] = {}
        self._by_title: Dict[Tuple[str, str, str], Set[str]] = {}

    def load(self, path: str) -> int:
        data = load_json_relaxed(path)
        loaded = 0
        if not isinstance(data, list):
            return loaded

        for row in data:
            if not isinstance(row, dict):
                continue
            ref_info = row.get("ref_info", {}) or {}
            objective = row.get("objective", {}) or {}
            action_prior = row.get("action_prior_reward", {}) or {}
            state0 = row.get("state0", {}) or {}
            objective_tag = normalize_text(objective.get("primary", "")).strip()
            if not objective_tag:
                tags = objective.get("tags", []) or []
                if isinstance(tags, list) and tags:
                    objective_tag = normalize_text(tags[0]).strip()
            if not objective_tag:
                continue

            actions: Set[str] = set()
            for action, value in action_prior.items():
                if value != 1:
                    continue
                mapped = canonicalize_action_name(action)
                if mapped in ACTIONS:
                    actions.add(mapped)
            if not actions:
                continue

            signature = state0_signature_from_row(state0)
            doi = normalize_doi(ref_info.get("paper_doi", ""))
            title = normalize_title(ref_info.get("paper_title", ""))

            if doi:
                key = (doi, objective_tag, signature)
                if key not in self._by_doi:
                    self._by_doi[key] = set()
                self._by_doi[key].update(actions)
            if title:
                key = (title, objective_tag, signature)
                if key not in self._by_title:
                    self._by_title[key] = set()
                self._by_title[key].update(actions)
            loaded += 1

        return loaded

    def lookup(self, paper: Dict[str, Any], objective_tag: str, state0_signature: str) -> Optional[Set[str]]:
        tag = normalize_text(objective_tag).strip()
        if not tag:
            return None
        meta = (paper or {}).get("paper_metadata", {}) if isinstance(paper, dict) else {}
        doi = normalize_doi(meta.get("doi", ""))
        title = normalize_title(meta.get("title", ""))

        if doi:
            key = (doi, tag, state0_signature)
            if key in self._by_doi:
                return set(self._by_doi[key])
        if title:
            key = (title, tag, state0_signature)
            if key in self._by_title:
                return set(self._by_title[key])
        return None


def normalize_key_token(value: Any) -> str:
    s = normalize_text(value).strip()
    s = s.replace("−", "-").replace("–", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def canonical_lookup_key(value: Any) -> str:
    s = normalize_key_token(value)
    s = s.replace("_", "")
    s = s.replace("-", "")
    s = s.replace(" ", "")
    return s


def normalize_doi(value: Any) -> str:
    s = normalize_text(value).strip()
    s = s.replace("https://doi.org/", "")
    s = s.replace("http://doi.org/", "")
    s = s.replace("doi:", "")
    return s.strip()


def normalize_title(value: Any) -> str:
    s = normalize_text(value).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def additive_name_match_keys(raw_name: Any) -> Set[str]:
    raw = normalize_key_token(raw_name)
    if not raw:
        return set()

    keys: Set[str] = set()

    def _add(token: str) -> None:
        t = normalize_key_token(token)
        if not t:
            return
        keys.add(canonical_lookup_key(t))
        alias = SOLVENT_ALIASES.get(t)
        if alias:
            keys.add(canonical_lookup_key(alias))

    _add(raw)
    _add(raw.replace("(", "").replace(")", ""))

    m = re.search(r"\(([^)]+)\)", raw)
    if m:
        _add(m.group(1))
        outside = normalize_key_token(re.sub(r"\([^)]*\)", "", raw)).strip()
        if outside:
            _add(outside)

    for token in re.split(r"[/,:;+|]", raw):
        _add(token)

    return {k for k in keys if k}


class SolventStandardizer:
    def __init__(self, vocab_names: Iterable[str]):
        self.name_lookup: Dict[str, str] = {}
        for name in vocab_names:
            name_str = str(name).strip()
            if not name_str:
                continue
            key = canonical_lookup_key(name_str)
            if key and key not in self.name_lookup:
                self.name_lookup[key] = name_str

    @classmethod
    def from_json(cls, path: str) -> "SolventStandardizer":
        data = load_json_relaxed(path)
        names: List[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    name = item.get("name")
                    if name:
                        names.append(str(name))
                elif isinstance(item, str):
                    names.append(item)
        return cls(names)

    def _resolve_single_name(self, raw_name: str) -> Optional[str]:
        raw = normalize_key_token(raw_name)
        if not raw or raw in GENERIC_SOLVENT_KEYS:
            return None

        candidates: List[str] = [raw]
        m = re.search(r"\(([^)]+)\)", raw)
        if m:
            inner = normalize_key_token(m.group(1))
            if inner:
                candidates.append(inner)
            outside = normalize_key_token(re.sub(r"\([^)]*\)", "", raw)).strip()
            if outside:
                candidates.append(outside)

        candidates.append(raw.replace("(", "").replace(")", ""))
        candidates.append(raw.replace(":", "/"))
        if raw in SOLVENT_ALIASES:
            candidates.append(normalize_key_token(SOLVENT_ALIASES[raw]))

        for cand in candidates:
            alias_target = SOLVENT_ALIASES.get(cand)
            if alias_target:
                cand = normalize_key_token(alias_target)
            key = canonical_lookup_key(cand)
            if key in self.name_lookup:
                return self.name_lookup[key]
        return None

    def resolve_name(self, raw_name: Any) -> List[str]:
        key = normalize_key_token(raw_name)
        if not key:
            return []

        direct = self._resolve_single_name(key)
        if direct:
            return [direct]

        split_tokens = [t.strip() for t in re.split(r"[/,:;+|]", key) if t.strip()]
        if len(split_tokens) >= 2:
            resolved: List[str] = []
            for token in split_tokens:
                r = self._resolve_single_name(token)
                if not r:
                    return []
                resolved.append(r)
            return sorted(set(resolved))

        return []

    def standardize_map(self, solvents: Dict[str, Any], keep_unresolved: bool = False) -> Dict[str, str]:
        if not isinstance(solvents, dict):
            return {}

        numeric_values: Dict[str, float] = {}
        text_values: Dict[str, str] = {}

        for raw_key, raw_val in solvents.items():
            resolved = self.resolve_name(raw_key)
            fallback_name = normalize_key_token(raw_key)
            if not resolved and keep_unresolved and fallback_name and fallback_name not in GENERIC_SOLVENT_KEYS:
                resolved = [fallback_name]
            if not resolved:
                continue
            raw_str = str(raw_val).strip()
            parsed = parse_float(raw_str)
            if parsed is not None:
                share = max(parsed, 0.0) / float(len(resolved))
                for name in resolved:
                    numeric_values[name] = numeric_values.get(name, 0.0) + share
            else:
                for name in resolved:
                    if name not in numeric_values and name not in text_values:
                        text_values[name] = raw_str

        if numeric_values:
            return {k: f"{v:.6g}" for k, v in sorted(numeric_values.items())}
        return {k: v for k, v in sorted(text_values.items())}


SOLVENT_SMILES_OVERRIDES: Dict[str, str] = {
    "BTC": "FC(F)(F)COC(=O)OCC(F)(F)F",
    "HFE": "C(C(C(F)F)(F)F)OC(C(F)F)(F)F",
    "DEE": "CCOCCOCC",
    "CPME": "COC1CCCC1",
    "bis(2,2,2-trifluoroethyl) ether (BTFE)": "FC(F)(F)COCC(F)(F)F",
}


def clean_smiles(smiles: Any) -> Optional[str]:
    if smiles is None:
        return None
    s = re.sub(r"\s+", "", str(smiles))
    if not s or s.lower() in {"none", "null", "nan"}:
        return None
    return s


class SolventSmilesResolver:

    def __init__(self) -> None:
        self._vocab: Dict[str, Optional[str]] = {}
        self._curated: Dict[str, str] = {}
        self._overrides: Dict[str, str] = {
            canonical_lookup_key(k): v for k, v in SOLVENT_SMILES_OVERRIDES.items()
        }

    def _set_smiles(self, name: Any, smiles: Any) -> None:
        key = canonical_lookup_key(name) if name is not None else ""
        if not key:
            return
        s = clean_smiles(smiles)
        if key not in self._vocab or (self._vocab[key] is None and s is not None):
            self._vocab[key] = s

    def load_excel_vocab(self, path: str) -> int:
        data = load_json_relaxed(path)
        loaded = 0
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                if "name" not in item:
                    continue
                self._set_smiles(item.get("name"), item.get("smiles"))
                loaded += 1
        return loaded

    def load_unmapped_vocab(self, path: str) -> int:
        data = load_json_relaxed(path)
        loaded = 0
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                if "name" not in item:
                    continue
                self._set_smiles(item.get("name"), item.get("smiles"))
                loaded += 1
        return loaded

    def load_curated_lookup(self, path: str) -> int:
        data = load_json_relaxed(path)
        loaded = 0
        if not isinstance(data, dict):
            return loaded
        for section in ("components", "solvents", "additives"):
            for name, smiles in (data.get(section, {}) or {}).items():
                key = canonical_lookup_key(name)
                s = clean_smiles(smiles)
                if key and s and key not in self._curated:
                    self._curated[key] = s
                    loaded += 1
        return loaded

    def known_names(self) -> int:
        return len(set(self._vocab) | set(self._curated) | set(self._overrides))

    def get_smiles(self, name: Any) -> Optional[str]:
        key = canonical_lookup_key(name)
        if not key:
            return None
        if key in self._overrides:
            return self._overrides[key]
        if key in self._curated:
            return self._curated[key]
        return self._vocab.get(key)


def build_solvent_entries_with_smiles(
    solvents_map: Dict[str, str],
    resolver: Optional[SolventSmilesResolver],
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for name, ratio in sorted(solvents_map.items()):
        entries.append(
            {
                "name": name,
                "ratio": ratio,
                "smiles": resolver.get_smiles(name) if resolver else None,
            }
        )
    return entries


def load_json_relaxed(path: str) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", text)
        return json.loads(cleaned)


def normalize_ratio_map(values: Dict[str, Any]) -> Dict[str, str]:
    if not values:
        return {}

    parsed: Dict[str, float] = {}
    for key, raw in values.items():
        s = str(raw).strip()
        v = parse_float(s)
        if v is None:
            continue
        if "%" in s:
            v = v / 100.0
        parsed[str(key)] = max(v, 0.0)

    if not parsed:
        return {str(k): str(v) for k, v in values.items()}

    total = sum(parsed.values())
    if total <= 0:
        return {str(k): str(v) for k, v in values.items()}

    normalized: Dict[str, str] = {}
    for key, value in parsed.items():
        normalized[key] = f"{(value / total):.6g}"
    return normalized


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


def collect_paper_text(paper: Dict[str, Any], baseline: Dict[str, Any]) -> str:
    chunks: List[str] = []
    chunks.append(flatten_text(paper.get("paper_metadata", {})))
    chunks.append(flatten_text(paper.get("design_boundaries", {})))
    chunks.append(flatten_text(paper.get("exploration_operations", {})))
    chunks.append(flatten_text(paper.get("design_conclusions", [])))

    for formulation in paper.get("formulations", []):
        chunks.append(flatten_text(formulation.get("composition", {})))
        for prop in formulation.get("evaluated_properties", []):
            chunks.append(flatten_text(prop.get("conditions", [])))
            chunks.append(flatten_text(prop.get("outcome", "")))

    chunks.append(flatten_text(baseline.get("composition", {})))
    return " ".join(chunks).lower()


def select_baseline(paper: Dict[str, Any]) -> Dict[str, Any]:
    formulations = paper.get("formulations", [])
    if not formulations:
        return {}

    
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

    conclusions = normalize_text(" ".join(paper.get("design_conclusions", [])))
    if any(k in conclusions for k in ["standard", "pristine", "blank", "base electrolyte"]):
        return formulations[0]

    return formulations[0]


def infer_high_voltage(text: str) -> int:
    if "≥4.4" in text or "geq 4.4" in text:
        return 1
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*v", text):
        try:
            if float(m.group(1)) >= 4.4:
                return 1
        except ValueError:
            continue
    return 0


def infer_wide_temperature(text: str) -> int:
    if "elevated temperature" in text:
        return 1
    temps: Set[float] = set()
    for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*°\s*c", text):
        try:
            temps.add(float(m.group(1)))
        except ValueError:
            continue
    return 1 if len(temps) >= 2 else 0


def infer_lithium_metal(text: str) -> int:
    patterns = ["li/li", "li||li", "li–cu", "li-cu", "cu/li", "li metal", "lithium metal"]
    return 1 if any(p in text for p in patterns) else 0


def infer_cathode_class(text: str) -> str:
    if "lnmo" in text or "lini0.5mn1.5o4" in text:
        return "LNMO"
    if "ncm" in text or "nmc" in text or "lini" in text and "co" in text and "mn" in text:
        return "NCM"
    if "lco" in text or "licoo2" in text:
        return "LCO"
    if "lfp" in text or "lifepo4" in text:
        return "LFP"
    if "li-rich" in text or "lithium-rich" in text:
        return "Li-rich"
    return "unknown"


def infer_electrolyte_family(text: str, baseline_comp: Dict[str, Any]) -> str:
    combined = f"{text} {flatten_text(baseline_comp).lower()}"
    has_carbonate = any(tok in combined for tok in CARBONATE_TOKENS)
    has_ether = any(tok in combined for tok in ETHER_TOKENS)
    has_ionic = any(tok in combined for tok in IONIC_TOKENS)
    has_high_conc = "high-concentration" in combined or "high concentration" in combined

    if has_ionic and not has_carbonate and not has_ether:
        return "ionic"
    if (has_carbonate and has_ether) or (has_ether and has_high_conc):
        return "mixed"
    if has_carbonate:
        return "carbonate"
    if has_ether:
        return "ether"
    if has_ionic:
        return "ionic"
    return "unknown"


def infer_design_constraint(paper: Dict[str, Any]) -> Dict[str, int]:
    boundaries = paper.get("design_boundaries", {}) or {}
    exploration = paper.get("exploration_operations", {}) or {}

    fixed_text = normalize_text(" ".join(boundaries.get("fixed_parameters", [])))
    intent_text = normalize_text(" ".join(boundaries.get("intentional_constraints", [])))
    not_explored_text = normalize_text(" ".join(exploration.get("not_explored", [])))

    additive_only = 0
    if "additive-only" in intent_text or "additive only" in intent_text:
        additive_only = 1
    elif "salt" in not_explored_text and "solvent" in not_explored_text:
        additive_only = 1

    salt_fixed = 1 if ("salt" in fixed_text or "salt" in not_explored_text) else 0
    solvent_fixed = 1 if ("solvent" in fixed_text or "solvent" in not_explored_text) else 0

    return {
        "additive_only": additive_only,
        "salt_fixed": salt_fixed,
        "solvent_fixed": solvent_fixed,
    }


def extract_additives(comp: Dict[str, Any]) -> Set[str]:
    names: Set[str] = set()

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

    cleaned: Set[str] = set()
    for n in names:
        if n:
            cleaned.add(re.sub(r"\s+", " ", n))
    return cleaned


def has_additive_hint(name: Any, amount_text: Any) -> bool:
    text = f"{normalize_text(name)} {normalize_text(amount_text)}"
    if not text.strip():
        return False
    additive_markers = [
        "additive",
        "additives",
        "co-additive",
        "coadditive",
        "dopant",
        "functional additive",
    ]
    return any(marker in text for marker in additive_markers)


def collect_additives_for_state0(
    comp: Dict[str, Any],
    codebook_additive_keys: Optional[Set[str]] = None,
) -> Dict[str, str]:
    merged: Dict[str, str] = {}

    additives = comp.get("additives")
    if isinstance(additives, dict):
        for name, amount in additives.items():
            name_s = str(name).strip()
            if not name_s:
                continue
            merged[name_s] = str(amount).strip() if amount is not None else ""
    elif isinstance(additives, list):
        for item in additives:
            if isinstance(item, dict):
                for name, amount in item.items():
                    name_s = str(name).strip()
                    if not name_s:
                        continue
                    merged[name_s] = str(amount).strip() if amount is not None else ""
            elif item not in (None, ""):
                name_s = str(item).strip()
                if name_s:
                    merged[name_s] = ""
    elif additives not in (None, "", {}):
        name_s = str(additives).strip()
        if name_s:
            merged[name_s] = ""

    solvents = comp.get("solvents") or {}
    if isinstance(solvents, dict):
        for solvent_name, solvent_amount in solvents.items():
            solvent_match_keys = additive_name_match_keys(solvent_name)
            codebook_match = bool(
                codebook_additive_keys
                and solvent_match_keys
                and (solvent_match_keys & codebook_additive_keys)
            )
            if has_additive_hint(solvent_name, solvent_amount) or codebook_match:
                name_s = str(solvent_name).strip()
                if not name_s:
                    continue
                amount_s = str(solvent_amount).strip() if solvent_amount is not None else ""
                merged[name_s] = amount_s

    return merged


def extract_solvents(comp: Dict[str, Any], solvent_standardizer: Optional[SolventStandardizer] = None) -> Dict[str, str]:
    solvents = comp.get("solvents") or {}
    if solvent_standardizer:
        solvents = solvent_standardizer.standardize_map(solvents)
    result: Dict[str, str] = {}
    for k, v in solvents.items():
        key = normalize_text(k).strip()
        val = normalize_text(v).strip()
        if key:
            result[key] = val
    return result


_ADDITIVE_LEVEL_RE = re.compile(r"%|ppm", re.IGNORECASE)
REFERENCE_ROLES = {"reference", "control", "standard"}


def added_salt_is_additive_level(base_salt: str, var_salt: str, base_sc: str, var_sc: str) -> bool:
    more_pct = len(_ADDITIVE_LEVEL_RE.findall(var_sc)) > len(_ADDITIVE_LEVEL_RE.findall(base_sc))
    named_additive = "additive" in var_salt and "additive" not in base_salt
    return more_pct or named_additive


def infer_action_types(
    baseline: Dict[str, Any],
    formulation: Dict[str, Any],
    paper: Optional[Dict[str, Any]] = None,
    solvent_standardizer: Optional[SolventStandardizer] = None,
) -> Set[str]:
    base_comp = baseline.get("composition", {}) or {}
    var_comp = formulation.get("composition", {}) or {}

    base_add = extract_additives(base_comp)
    var_add = extract_additives(var_comp)
    base_salt = normalize_text(base_comp.get("salt", "")).strip()
    var_salt = normalize_text(var_comp.get("salt", "")).strip()
    base_sc = normalize_text(base_comp.get("salt_concentration", "")).strip()
    var_sc = normalize_text(var_comp.get("salt_concentration", "")).strip()

    base_solvents = extract_solvents(base_comp, solvent_standardizer=solvent_standardizer)
    var_solvents = extract_solvents(var_comp, solvent_standardizer=solvent_standardizer)
    base_solvent_keys = set(base_solvents.keys())
    var_solvent_keys = set(var_solvents.keys())
    base_sc_v = parse_ratio_value(base_sc) if base_sc else None
    var_sc_v = parse_ratio_value(var_sc) if var_sc else None
    salt_rel_delta = relative_change(base_sc_v, var_sc_v)
    base_solvent_norm = normalize_ratio_map(base_solvents)
    var_solvent_norm = normalize_ratio_map(var_solvents)

    additive_action: Optional[str] = None
    if base_add != var_add:
        additive_action = "additive_change"

    salt_action: Optional[str] = None
    if base_salt != var_salt:
        if not base_salt and var_salt:
            if salt_rel_delta is not None and salt_rel_delta < SMALL_CHANGE_THRESHOLD:
                salt_action = "additive_change"
            elif salt_rel_delta is not None and salt_rel_delta >= SMALL_CHANGE_THRESHOLD:
                salt_action = "add_new_salt"
            else:
                salt_action = None
        else:
            base_salt_tokens = split_component_tokens(base_salt)
            var_salt_tokens = split_component_tokens(var_salt)
            if (
                base_salt_tokens
                and var_salt_tokens
                and base_salt_tokens.issubset(var_salt_tokens)
                and len(var_salt_tokens) > len(base_salt_tokens)
            ):
                if salt_rel_delta is not None and salt_rel_delta < SMALL_CHANGE_THRESHOLD:
                    salt_action = "additive_change"
                elif salt_rel_delta is not None and salt_rel_delta >= SMALL_CHANGE_THRESHOLD:
                    salt_action = "add_new_salt"
                else:
                    salt_action = None
            else:
                salt_action = "salt_replace"
    elif base_sc and var_sc and base_sc != var_sc:
        if salt_rel_delta is None:
            salt_action = None
        elif salt_rel_delta < SMALL_CHANGE_THRESHOLD:
            salt_action = "additive_change"
        else:
            salt_action = "salt_concentration_shift"


    if salt_action == "add_new_salt" and added_salt_is_additive_level(base_salt, var_salt, base_sc, var_sc):
        salt_action = None
        additive_action = "additive_change"


    solvent_action: Optional[str] = None
    if base_solvent_keys != var_solvent_keys:
        solvent_total_delta = composition_total_change(base_solvents, var_solvents)
        if solvent_total_delta is not None and solvent_total_delta < SMALL_CHANGE_THRESHOLD:
            solvent_action = "additive_change"
        elif base_solvent_keys and var_solvent_keys and base_solvent_keys.issubset(var_solvent_keys):
            solvent_action = "add_new_solvent"
        elif base_solvent_keys and var_solvent_keys and var_solvent_keys.issubset(base_solvent_keys):
            solvent_action = "remove_solvent"
        elif not base_solvent_keys and var_solvent_keys:
            solvent_action = "add_new_solvent"
        elif base_solvent_keys and not var_solvent_keys:
            solvent_action = "remove_solvent"
        else:
            solvent_action = "solvent_replace"
    elif base_solvent_keys and base_solvent_keys == var_solvent_keys:
        if any(base_solvent_norm.get(k, "") != var_solvent_norm.get(k, "") for k in base_solvent_keys):
            solvent_total_delta = composition_total_change(base_solvents, var_solvents)
            if solvent_total_delta is None:
                solvent_action = None
            elif solvent_total_delta < SMALL_CHANGE_THRESHOLD:
                solvent_action = "additive_change"
            else:
                solvent_action = "solvent_concentration_shift"

    actions: Set[str] = set()
    if additive_action is not None:
        actions.add(additive_action)
    if salt_action is not None:
        actions.add(salt_action)
    if solvent_action is not None:
        actions.add(solvent_action)


    if additive_action is None and "additive_change" in actions and any(a != "additive_change" for a in actions):
        actions.discard("additive_change")

    if actions:
        return actions

    if paper:
        exploration = paper.get("exploration_operations", {}) or {}
        boundaries = paper.get("design_boundaries", {}) or {}
        explored_text = normalize_text(" ".join(exploration.get("explored_dimensions", [])))
        not_explored_text = normalize_text(" ".join(exploration.get("not_explored", [])))
        intent_text = normalize_text(" ".join(boundaries.get("intentional_constraints", [])))
        fid = normalize_text(formulation.get("id", ""))

        additive_only = (
            "additive" in explored_text
            and "salt" in not_explored_text
            and "solvent" in not_explored_text
        ) or ("additive-only" in intent_text or "additive only" in intent_text)

        if additive_only:
            return {"additive_change"}
        if any(k in fid for k in ["without additive", "additive-free", "no additive"]):
            return {"additive_change"}
        if "additive" in fid:
            return {"additive_change"}

    return set()


def reference_predecessors(
    paper: Dict[str, Any],
    baseline: Dict[str, Any],
    solvent_standardizer: Optional[SolventStandardizer] = None,
) -> List[Tuple[Dict[str, Any], Set[str]]]:
    out: List[Tuple[Dict[str, Any], Set[str]]] = []
    seen: Set[str] = set()
    formulations = paper.get("formulations", []) or []
    targets = [f for f in formulations if normalize_role(f.get("role", "")) not in REFERENCE_ROLES]
    if baseline not in targets:
        targets.insert(0, baseline)
    for ref in formulations:
        if ref is baseline or normalize_role(ref.get("role", "")) not in REFERENCE_ROLES:
            continue
        key = json.dumps(ref.get("composition", {}) or {}, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        explored: Set[str] = set()
        for target in targets:
            if json.dumps(target.get("composition", {}) or {}, sort_keys=True, ensure_ascii=False) == key:
                continue
            actions = infer_action_types(ref, target, paper=paper, solvent_standardizer=solvent_standardizer)
            if len(actions) == 1 and "remove_solvent" not in actions:
                explored |= actions
        if explored:
            seen.add(key)
            out.append((ref, explored))
    return out


def infer_action_type(
    baseline: Dict[str, Any],
    formulation: Dict[str, Any],
    paper: Optional[Dict[str, Any]] = None,
    solvent_standardizer: Optional[SolventStandardizer] = None,
) -> Optional[str]:
    actions = infer_action_types(
        baseline,
        formulation,
        paper=paper,
        solvent_standardizer=solvent_standardizer,
    )
    if len(actions) == 1:
        return next(iter(actions))
    return None


def keyword_polarity(text: str) -> Optional[int]:
    t = normalize_text(text)
    pos = sum(1 for k in POSITIVE_KEYWORDS if k in t)
    neg = sum(1 for k in NEGATIVE_KEYWORDS if k in t)
    tradeoff = any(h in t for h in TRADEOFF_HINTS)

    if pos > 0 and neg == 0:
        return +1
    if neg > 0 and pos == 0:
        return -1
    if pos > 0 and neg > 0:
        return 0
    if tradeoff and (pos > 0 or neg > 0):
        return 0
    return None


def action_relevant_conclusion_polarity(paper: Dict[str, Any], action_type: str) -> Optional[int]:
    conclusions = " ".join(paper.get("design_conclusions", []))
    text = normalize_text(conclusions)

    action_map = {
        "additive_change": [
            "additive",
            "incorporating",
            "dop",
            "functional",
            "remove additive",
            "without additive",
            "additive-free",
        ],
        "salt_replace": ["salt", "lipf6", "lifsi", "lidfob", "salt replacement"],
        "salt_concentration_shift": ["concentration", "m", "high-concentration"],
        "add_new_salt": ["dual-salt", "multi-salt", "additional salt", "co-salt"],
        "solvent_replace": ["solvent", "carbonate", "ether", "co-solvent", "cosolvent"],
        "add_new_solvent": ["solvent", "co-solvent", "cosolvent"],
        "remove_solvent": ["solvent", "without", "removed", "remove"],
        "solvent_concentration_shift": ["ratio", "vol%", "v/v", "composition"],
    }

    tags = action_map.get(action_type, [])
    if tags and not any(tag in text for tag in tags):
        return None
    return keyword_polarity(text)


def infer_prior_reward(paper: Dict[str, Any], formulation: Dict[str, Any], action_type: str) -> int:
    role = normalize_role(formulation.get("role", ""))
    if role == "optimized":
        return +1

    outcome_scores: List[int] = []
    for prop in formulation.get("evaluated_properties", []) or []:
        score = keyword_polarity(prop.get("outcome", ""))
        if score is not None:
            outcome_scores.append(score)

    outcome_signal: Optional[int]
    if not outcome_scores:
        outcome_signal = None
    elif all(s == +1 for s in outcome_scores):
        outcome_signal = +1
    elif all(s == -1 for s in outcome_scores):
        outcome_signal = -1
    elif any(s == +1 for s in outcome_scores) and any(s == -1 for s in outcome_scores):
        outcome_signal = 0
    else:
        outcome_signal = 0 if any(s == 0 for s in outcome_scores) else outcome_scores[0]

    conclusion_signal = action_relevant_conclusion_polarity(paper, action_type)

    if outcome_signal is None and conclusion_signal is None:
        return 0
    if outcome_signal is None:
        return conclusion_signal if conclusion_signal is not None else 0
    if conclusion_signal is None:
        return outcome_signal
    if outcome_signal == conclusion_signal:
        return outcome_signal
    return 0


def infer_objective(paper: Dict[str, Any], formulation: Dict[str, Any]) -> Dict[str, Any]:
    objective_text_parts = [
        flatten_text(formulation.get("evaluated_properties", [])),
        flatten_text(paper.get("design_conclusions", [])),
        flatten_text(paper.get("design_boundaries", {})),
    ]
    text = normalize_text(" ".join(objective_text_parts))

    matched: List[str] = []
    for tag, keys in OBJECTIVE_TAGS.items():
        if any(k in text for k in keys):
            matched.append(tag)

    if not matched:
        matched = ["unspecified_performance_objective"]

    return {
        "primary": matched[0],
        "tags": matched,
    }


GOAL_HIGH_VOLTAGE = "high_voltage"
GOAL_TEMPERATURE = "wide_temperature"
GOAL_OTHER = "other"
CHANGE_GOALS = (GOAL_HIGH_VOLTAGE, GOAL_TEMPERATURE, GOAL_OTHER)

_HV_KEYWORDS = (
    "high voltage", "high-voltage", "oxidation stability", "oxidative stability", "anodic stability",
    "oxidation potential", "oxidative decomposition", "linear sweep", "lsv", "leakage current",
    "potentiostatic hold", "floating test", "float test", "transition metal dissolution",
    "transition-metal dissolution", "transition metal deposition",
)
_HV_CATHODES = ("lnmo", "lini0.5mn1.5o4", "lnmo", "lini0.5 mn1.5o4", "licopo4", "linipo4")
_TEMP_KEYWORDS = (
    "low temperature", "low-temperature", "high temperature", "high-temperature",
    "elevated temperature", "wide temperature", "wide-temperature", "subzero", "sub-zero",
    "all-climate", "all climate", "all-temperature", "cold", "freezing",
)
_TRANSPORT_PROPS = ("conductivity", "viscosity", "diffusion", "arrhenius", "activation energy", "walden")
_VOLT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*v\b")


_TEMP_RANGE_RE = re.compile(
    r"(?<![\d.])([-−]?\d+(?:\.\d+)?)\s*(?:°\s*c)?\s*(?:to|~|–|—|-)\s*([-−]?\d+(?:\.\d+)?)\s*°\s*c"
)


def _temps_in(text: str) -> List[float]:
    def num(x: str) -> Optional[float]:
        try:
            return float(x.replace("−", "-"))
        except ValueError:
            return None

    out: List[float] = []
    for m in _TEMP_RANGE_RE.finditer(text):
        out += [v for v in (num(m.group(1)), num(m.group(2))) if v is not None]
    for m in re.finditer(r"(?<![\d.\-−–~])([-−]?\d+(?:\.\d+)?)\s*°\s*c", text):
        v = num(m.group(1))
        if v is not None:
            out.append(v)
    return out


def _goals_from_text(text: str, transport: bool = False) -> Set[str]:
    goals: Set[str] = set()
    t = text.lower()
    if any(k in t for k in _HV_KEYWORDS) or any(c in t for c in _HV_CATHODES):
        goals.add(GOAL_HIGH_VOLTAGE)
    for m in _VOLT_RE.finditer(t):
        try:
            if 4.4 <= float(m.group(1)) <= 6.5:
                goals.add(GOAL_HIGH_VOLTAGE)
                break
        except ValueError:
            continue
    if any(k in t for k in _TEMP_KEYWORDS):
        goals.add(GOAL_TEMPERATURE)
    else:
        temps = _temps_in(t)
        if transport:
            if any(x <= -20 for x in temps):
                goals.add(GOAL_TEMPERATURE)
        elif any(x <= -10 or 45 <= x <= 150 for x in temps):
            goals.add(GOAL_TEMPERATURE)
    return goals


_ROUTINE_HV_MEASUREMENTS = ("lsv", "linear sweep", "electrochemical stability window", "electrochemical window",
                            "oxidation stability", "oxidative stability", "anodic stability", "oxidation potential")


def infer_paper_goals(paper: Dict[str, Any]) -> List[str]:
    title_goals = (paper.get("paper_metadata") or {}).get("title_goals") or []
    if title_goals:
        return [g for g in CHANGE_GOALS if g in title_goals]
    goals = _goals_from_text(flatten_text(paper.get("design_conclusions", [])).lower())
    if goals:
        return [g for g in CHANGE_GOALS if g in goals]
    goals = set()
    base = select_baseline(paper)
    for f in paper.get("formulations", []) or []:
        if f is base:
            continue
        for prop in f.get("evaluated_properties", []) or []:
            name = str(prop.get("property", ""))
            text = " ".join([name, flatten_text(prop.get("conditions", []))]).lower()
            if any(k in text for k in _ROUTINE_HV_MEASUREMENTS):
                continue
            goals |= _goals_from_text(text, transport=any(k in name.lower() for k in _TRANSPORT_PROPS))
    return [g for g in CHANGE_GOALS if g in goals] or [GOAL_OTHER]


def infer_change_goals(paper: Dict[str, Any], formulation: Dict[str, Any]) -> List[str]:
    return infer_paper_goals(paper)


def build_context_state(paper: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    paper_text = collect_paper_text(paper, baseline)
    baseline_comp = baseline.get("composition", {}) if baseline else {}

    return {
        "high_voltage": infer_high_voltage(paper_text),
        "wide_temperature": infer_wide_temperature(paper_text),
        "lithium_metal": infer_lithium_metal(paper_text),
        "cathode_class": infer_cathode_class(paper_text),
        "electrolyte_family": infer_electrolyte_family(paper_text, baseline_comp or {}),
        "design_constraint": infer_design_constraint(paper),
    }


def make_prior_row(
    paper: Dict[str, Any],
    state_formulation: Dict[str, Any],
    context_state: Dict[str, Any],
    objective_tag: str,
    explored_actions: Set[str],
    codebook_additive_keys: Set[str],
    solvent_standardizer: Optional[SolventStandardizer],
    smiles_resolver: Any,
    state0_source: str = "baseline",
) -> Dict[str, Any]:
    comp = state_formulation.get("composition", {}) or {}
    state0 = {
        "composition": {
            "salt": comp.get("salt", ""),
            "salt_concentration": comp.get("salt_concentration", ""),
            "solvents_with_smiles": [],
            "additives": collect_additives_for_state0(comp, codebook_additive_keys=codebook_additive_keys),
        }
    }
    standardized_solvents = normalize_ratio_map(
        solvent_standardizer.standardize_map(comp.get("solvents", {}) or {}, keep_unresolved=True)
        if solvent_standardizer
        else (comp.get("solvents", {}) or {})
    )
    state0["composition"]["solvents_with_smiles"] = build_solvent_entries_with_smiles(
        standardized_solvents,
        resolver=smiles_resolver,
    )
    return {
        "state0": state0,
        "features": context_state,
        "objective": {
            "primary": objective_tag,
            "tags": [objective_tag],
        },
        "action_prior_reward": {action: (1 if action in explored_actions else 0) for action in ACTIONS},
        "ref_info": {
            "paper_title": paper.get("paper_metadata", {}).get("title", ""),
            "paper_doi": paper.get("paper_metadata", {}).get("doi", ""),
            "state0_source": state0_source,
            "state0_id": str(state_formulation.get("id", "")),
        },
    }


def build_prior_dataset(
    papers: Iterable[Dict[str, Any]],
    solvent_standardizer: Optional[SolventStandardizer] = None,
    smiles_resolver: Optional[SolventSmilesResolver] = None,
    codebook_matcher: Optional[CodebookAdditiveMatcher] = None,
    uncertain_resolver: Optional[UncertainActionResolver] = None,
    backup_lookup: Optional[BackupActionLookup] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    skipped_uncertain = 0
    llm_resolved = 0
    llm_failed = 0
    backup_reused_unknown = 0
    backup_merged_objectives = 0
    reference_rows = 0

    for paper in papers:
        formulations = paper.get("formulations", [])
        if len(formulations) <= 1:
            continue

        baseline = select_baseline(paper)
        if not baseline:
            continue

        context_state = build_context_state(paper, baseline)
        codebook_additive_keys = (
            codebook_matcher.get_additive_keys_for_paper(paper) if codebook_matcher else set()
        )
        baseline_comp = baseline.get("composition", {}) or {}
        baseline_state0_sig = state0_signature_from_baseline(
            baseline_comp,
            solvent_standardizer=solvent_standardizer,
            codebook_additive_keys=codebook_additive_keys,
        )
        baseline_id = baseline.get("id")
        objective_to_actions: Dict[str, Set[str]] = {}
        objective_to_formulations: Dict[str, Set[str]] = {}
        objective_unknown_counts: Dict[str, int] = {}

        for formulation in formulations:
            if formulation.get("id") == baseline_id and formulation is baseline:
                continue

            objective_tags = infer_change_goals(paper, formulation)

            seen_objective_tags: List[str] = []
            seen_tags_set: Set[str] = set()
            for objective_tag in objective_tags:
                if objective_tag in seen_tags_set:
                    continue
                seen_tags_set.add(objective_tag)
                seen_objective_tags.append(objective_tag)
                if objective_tag not in objective_to_actions:
                    objective_to_actions[objective_tag] = set()
                if objective_tag not in objective_to_formulations:
                    objective_to_formulations[objective_tag] = set()
                objective_to_formulations[objective_tag].add(str(formulation.get("id", "")))

            action_types = infer_action_types(
                baseline,
                formulation,
                paper=paper,
                solvent_standardizer=solvent_standardizer,
            )
            if not action_types:
                reused_from_backup = False
                if backup_lookup is not None and seen_objective_tags:
                    reused_from_backup = all(
                        backup_lookup.lookup(paper, objective_tag, baseline_state0_sig) is not None
                        for objective_tag in seen_objective_tags
                    )
                if reused_from_backup:
                    backup_reused_unknown += 1
                    for objective_tag in seen_objective_tags:
                        objective_unknown_counts[objective_tag] = objective_unknown_counts.get(objective_tag, 0) + 1
                    continue

                if uncertain_resolver is not None:
                    action_types = uncertain_resolver.classify(
                        baseline=baseline,
                        formulation=formulation,
                        paper=paper,
                    )
                    if action_types:
                        llm_resolved += 1
                    else:
                        llm_failed += 1
                        skipped_uncertain += 1
                        for objective_tag in seen_objective_tags:
                            objective_unknown_counts[objective_tag] = objective_unknown_counts.get(objective_tag, 0) + 1
                        continue
                else:
                    skipped_uncertain += 1
                    for objective_tag in seen_objective_tags:
                        objective_unknown_counts[objective_tag] = objective_unknown_counts.get(objective_tag, 0) + 1
                    continue

            for objective_tag in seen_objective_tags:
                objective_to_actions[objective_tag].update(action_types)

        # Emit full action coverage only for objectives that appeared in this paper.
        emitted_objectives: List[str] = []
        for objective_tag, explored_actions in objective_to_actions.items():
            merged_actions = set(explored_actions)
            if backup_lookup is not None and objective_unknown_counts.get(objective_tag, 0) > 0:
                backup_actions = backup_lookup.lookup(paper, objective_tag, baseline_state0_sig)
                if backup_actions:
                    merged_actions.update(backup_actions)
                    backup_merged_objectives += 1

            if not merged_actions:
                continue
            rows.append(
                make_prior_row(
                    paper,
                    baseline,
                    context_state,
                    objective_tag,
                    merged_actions,
                    codebook_additive_keys=codebook_additive_keys,
                    solvent_standardizer=solvent_standardizer,
                    smiles_resolver=smiles_resolver,
                )
            )
            emitted_objectives.append(objective_tag)

        for reference, step_actions in reference_predecessors(
            paper, baseline, solvent_standardizer=solvent_standardizer
        ):
            reference_context = build_context_state(paper, reference)
            for objective_tag in emitted_objectives:
                rows.append(
                    make_prior_row(
                        paper,
                        reference,
                        reference_context,
                        objective_tag,
                        step_actions,
                        codebook_additive_keys=codebook_additive_keys,
                        solvent_standardizer=solvent_standardizer,
                        smiles_resolver=smiles_resolver,
                        state0_source="reference_predecessor",
                    )
                )
                reference_rows += 1

    print(f"LLM-resolved uncertain formulations: {llm_resolved}")
    print(f"LLM-failed uncertain formulations: {llm_failed}")
    print(f"Skipped uncertain formulations: {skipped_uncertain}")
    if uncertain_resolver is not None and getattr(uncertain_resolver, "error_counts", None):
        parts = [f"{k}={v}" for k, v in sorted(uncertain_resolver.error_counts.items())]
        if parts:
            print(f"LLM error summary: {', '.join(parts)}")
        for idx, msg in enumerate(getattr(uncertain_resolver, "error_examples", [])[:3], start=1):
            print(f"LLM error sample {idx}: {msg}")
    print(f"Backup-reused uncertain formulations: {backup_reused_unknown}")
    print(f"Backup-merged objective rows: {backup_merged_objectives}")
    print(f"Reference-predecessor rows: {reference_rows}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert literature JSON to prior reward dataset")
    parser.add_argument("--input", default="json_literature.json", help="Path to literature JSON")
    parser.add_argument("--output", default="prior_reward_dataset.json", help="Path to output dataset JSON")
    parser.add_argument(
        "--excel_solvent_vocab",
        default="supplementary/excel_solvent_vocabulary.json",
        help="Path to solvent vocabulary JSON (list of {name, smiles})",
    )
    parser.add_argument(
        "--unmapped_solvent_vocab",
        default="supplementary/unmapped_solvent_keys.json",
        help="Path to unmapped solvent vocabulary JSON (list of {name, count, smiles})",
    )
    parser.add_argument(
        "--smiles_lookup",
        default="supplementary/component_smiles_lookup.json",
        help="Curated name -> SMILES lookup, preferred over the vocabulary files",
    )
    parser.add_argument(
        "--excel_codebook",
        default="supplementary/excel_codebook.json",
        help="Path to anonymized codebook JSON used to map papers to additive names",
    )
    parser.add_argument(
        "--backup_dataset",
        default="prior_reward_dataset_backup.json",
        help="Path to prior reward backup dataset used to reuse unknown action outcomes",
    )
    parser.add_argument(
        "--llm_cache",
        default="llm_action_cache.json",
        help="Persistent cache of LLM action decisions for transitions the rules cannot classify",
    )
    args = parser.parse_args()

    papers = load_json_relaxed(args.input)
    if not isinstance(papers, list):
        raise ValueError("Expected top-level JSON array of papers")

    solvent_standardizer: Optional[SolventStandardizer] = None
    vocab_names: List[str] = []
    excel_vocab_path = Path(args.excel_solvent_vocab)
    if excel_vocab_path.exists():
        excel_vocab_data = load_json_relaxed(str(excel_vocab_path))
        if isinstance(excel_vocab_data, list):
            for item in excel_vocab_data:
                if isinstance(item, dict) and item.get("name"):
                    vocab_names.append(str(item.get("name")))
                elif isinstance(item, str):
                    vocab_names.append(item)
        print(f"Loaded solvent names from {args.excel_solvent_vocab}")
    else:
        print(f"Excel solvent vocabulary not found: {args.excel_solvent_vocab}")

    unmapped_vocab_path = Path(args.unmapped_solvent_vocab)
    if unmapped_vocab_path.exists():
        unmapped_vocab_data = load_json_relaxed(str(unmapped_vocab_path))
        if isinstance(unmapped_vocab_data, list):
            for item in unmapped_vocab_data:
                if isinstance(item, dict) and item.get("name"):
                    vocab_names.append(str(item.get("name")))
        print(f"Loaded solvent names from {args.unmapped_solvent_vocab}")
    else:
        print(f"Unmapped solvent vocabulary not found: {args.unmapped_solvent_vocab}")

    if vocab_names:
        solvent_standardizer = SolventStandardizer(vocab_names)
        print(f"Built solvent standardizer with {len(solvent_standardizer.name_lookup)} names")
    else:
        print("No solvent vocabulary names loaded. Continue without solvent normalization.")

    smiles_resolver = SolventSmilesResolver()
    excel_loaded = smiles_resolver.load_excel_vocab(args.excel_solvent_vocab) if excel_vocab_path.exists() else 0
    unmapped_loaded = (
        smiles_resolver.load_unmapped_vocab(args.unmapped_solvent_vocab) if unmapped_vocab_path.exists() else 0
    )
    curated_loaded = (
        smiles_resolver.load_curated_lookup(args.smiles_lookup) if Path(args.smiles_lookup).exists() else 0
    )
    print(
        "Loaded smiles records: "
        f"excel={excel_loaded}, unmapped={unmapped_loaded}, "
        f"curated={curated_loaded}, unique_names={smiles_resolver.known_names()}"
    )

    codebook_matcher: Optional[CodebookAdditiveMatcher] = None
    codebook_path = Path(args.excel_codebook)
    if codebook_path.exists():
        codebook_matcher = CodebookAdditiveMatcher()
        codebook_loaded = codebook_matcher.load(args.excel_codebook)
        print(f"Loaded codebook additive rows: {codebook_loaded} from {args.excel_codebook}")
    else:
        print(f"Codebook not found: {args.excel_codebook}. Continue without codebook additive matching.")

    backup_lookup: Optional[BackupActionLookup] = None
    backup_dataset_path = Path(args.backup_dataset)
    if backup_dataset_path.exists():
        backup_lookup = BackupActionLookup()
        loaded_backup_rows = backup_lookup.load(args.backup_dataset)
        print(f"Loaded backup action rows: {loaded_backup_rows} from {args.backup_dataset}")
    else:
        print(f"Backup dataset not found: {args.backup_dataset}. Continue without backup reuse.")

    uncertain_resolver = UncertainActionResolver(cache_path=args.llm_cache)
    dataset = build_prior_dataset(
        papers,
        solvent_standardizer=solvent_standardizer,
        smiles_resolver=smiles_resolver,
        codebook_matcher=codebook_matcher,
        uncertain_resolver=uncertain_resolver,
        backup_lookup=backup_lookup,
    )
    unknown_actions = sorted(
        {
            action
            for row in dataset
            for action in (row.get("action_prior_reward", {}) or {}).keys()
            if action not in ACTIONS
        }
    )
    if unknown_actions:
        raise ValueError(f"Found actions outside ACTIONS: {unknown_actions}")

    Path(args.output).write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Built prior dataset with {len(dataset)} samples -> {args.output}")
    print(f"Action space ({len(ACTIONS)}): {', '.join(ACTIONS)}")


if __name__ == "__main__":
    main()

