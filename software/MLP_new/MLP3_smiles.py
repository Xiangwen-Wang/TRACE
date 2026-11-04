"""
SMILES-structure MLP action prior with per-descriptor scale normalization.

The 16 RDKit descriptors span very different ranges (MolWt ~100-600, atom counts
0-10, TPSA ~0-150). Each descriptor is divided by its standard deviation over the
distinct molecules of that component kind (salt anion / solvent / additive) seen in the
training examples, so no descriptor dominates by magnitude alone. Only scaling, no
centering, so an absent component still encodes as an all-zero block. The scales are
stored in obs_spec["desc_scale"]; older specs without them (structure_v1) are encoded
with raw descriptors, exactly as they were trained.

Two encoders, selected by obs_spec["feature_version"] (training: --feature_version):
  structure_v2_descscale  pooled 3162-d input vector -> MLP (the deployed checkpoint)
  structure_v3_set        per-molecule slots -> SetPrior: shared nonlinear molecule encoder, in-network
                          pooling (fraction-weighted + unweighted mean), concentration as a
                          separate log-scale scalar, one LayerNorm per component group
  structure_v4            v2 pooled input (fraction-weighted fingerprints, so similarity follows
                          molecular structure) with one fix: the anion block is not multiplied
                          by concentration; concentration is a separate log-scale scalar and
                          co-salts are weighted by molar share
  structure_v5            v4 + one-hot goal of the modification (high_voltage / wide_temperature /
                          other), read from objective.primary (training default)
Use build_model(obs_spec) to get the matching model for a checkpoint.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.parsers.expat import model

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


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
A2I = {a: i for i, a in enumerate(ACTIONS)}
_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_COMMON_CATIONS = (
    ("li", ("li", "lithium", "[li", "[li+]")),
    ("na", ("na", "sodium", "[na", "[na+]")),
    ("k", ("k", "potassium", "[k", "[k+]")),
    ("mg", ("mg", "magnesium", "[mg", "[mg+2", "[mg++]")),
    ("ca", ("ca", "calcium", "[ca", "[ca+2", "[ca++]")),
    ("zn", ("zn", "zinc", "[zn", "[zn+2", "[zn++]")),
    ("al", ("al", "aluminum", "aluminium", "[al", "[al+3")),
    ("nh4", ("nh4", "ammonium")),
)
_DESC_NAMES = [
    "mol_wt",
    "heavy_atom_count",
    "h_acceptors",
    "h_donors",
    "rot_bonds",
    "ring_count",
    "tpsa",
    "logp",
    "fraction_csp3",
    "hetero_atoms",
    "f_count",
    "o_count",
    "n_count",
    "s_count",
    "p_count",
    "formal_charge",
]
_MORGAN_GENERATORS: Dict[int, Any] = {}

_PLACEHOLDER_COMPONENT_KEYS = {
    "",
    "?",
    "na",
    "nan",
    "none",
    "no",
    "null",
    "unknown",
    "uncertain",
    "unclear",
    "unspecified",
    "notreported",
    "notavailable",
    "notapplicable",
    "tbd",
    "commerciallithiumsaltunspecified",
    "uncertainsalt",
    "unknownsalt",
    "castingsolvent",
    "polymermatrix",
    "otherfilmformingadditives",
    "noadditive",
    "noneadditive",
    "nosolvent",
    "nonesolvent",
}


def parse_first_float(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace("−", "-")
    m = _NUM.search(s)
    return float(m.group(0)) if m else float("nan")


def norm_name(x: Any) -> str:
    return str(x).strip().lower()


def compact_name_key(x: Any) -> str:
    return _NON_ALNUM.sub("", norm_name(x))


def alias_keys(x: Any) -> List[str]:
    raw = str(x or "").strip()
    if not raw:
        return []
    raw_norm = norm_name(raw)
    cleaned = re.sub(r"\([^)]*\)", " ", raw_norm)
    cleaned = re.sub(r"\badditive\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    keys = {
        raw_norm,
        compact_name_key(raw_norm),
        cleaned,
        compact_name_key(cleaned),
    }
    for token in re.findall(r"\(([^)]+)\)", raw):
        token_norm = norm_name(token)
        keys.add(token_norm)
        keys.add(compact_name_key(token_norm))
    if "_" in raw_norm:
        keys.add(raw_norm.replace("_", " "))
        keys.add(compact_name_key(raw_norm.replace("_", " ")))
    return [k for k in keys if k]


def is_placeholder_component_name(name: Any) -> bool:
    raw = norm_name(name)
    key = compact_name_key(raw)
    return raw in _PLACEHOLDER_COMPONENT_KEYS or key in _PLACEHOLDER_COMPONENT_KEYS


def get_comp(ex: Dict[str, Any]) -> Dict[str, Any]:
    return (ex.get("state0", {}) or {}).get("composition", {}) or {}


def get_salt_and_conc(ex: Dict[str, Any]) -> Tuple[str, float]:
    comp = get_comp(ex)
    return norm_name(comp.get("salt", "")), parse_first_float(comp.get("salt_concentration", None))


def split_component_names(raw: Any) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    text = text.replace("＋", "+")
    parts = re.split(r"\s*\+\s*|\s*/\s*|\s+and\s+|,", text)
    out = [p.strip() for p in parts if p.strip()]
    return out or [text]


def split_known_component_names(raw: Any) -> List[str]:
    return [name for name in split_component_names(raw) if not is_placeholder_component_name(name)]


def safe_weight(value: Any, default: float = 1.0) -> float:
    parsed = parse_first_float(value)
    return float(parsed) if np.isfinite(parsed) else float(default)


def register_smiles(lookup: Dict[str, str], name: Any, smiles: Any) -> None:
    smiles_text = str(smiles or "").strip()
    if not smiles_text or smiles_text.lower() in {"none", "null", "nan"}:
        return
    for key in alias_keys(name):
        lookup.setdefault(key, smiles_text)


def lookup_smiles(name: Any, lookup: Dict[str, str]) -> str:
    for key in alias_keys(name):
        if key in lookup:
            return lookup[key]
    return ""


def get_solvents_with_smiles(
    ex: Dict[str, Any],
    solvent_lookup: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str, float]]:
    comp = get_comp(ex)
    lst = comp.get("solvents_with_smiles", []) or []
    out: List[Tuple[str, str, float]] = []
    if isinstance(lst, list):
        for d in lst:
            if not isinstance(d, dict):
                continue
            name = norm_name(d.get("name", ""))
            if is_placeholder_component_name(name):
                continue
            smiles = str(d.get("smiles", "") or "").strip()
            if not smiles and solvent_lookup:
                smiles = lookup_smiles(name, solvent_lookup)
            out.append((name, smiles, safe_weight(d.get("ratio", None), default=1.0)))
        if out:
            return out

    solvents = comp.get("solvents", {}) or {}
    if isinstance(solvents, dict):
        for name, ratio in solvents.items():
            nm = norm_name(name)
            if is_placeholder_component_name(nm):
                continue
            smiles = lookup_smiles(nm, solvent_lookup or {})
            out.append((nm, smiles, safe_weight(ratio, default=1.0)))
    elif solvents not in (None, "", {}):
        for name in split_known_component_names(solvents):
            smiles = lookup_smiles(name, solvent_lookup or {})
            out.append((norm_name(name), smiles, 1.0))
    return out


def get_additives_with_smiles(
    ex: Dict[str, Any],
    additive_lookup: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str, float]]:
    comp = get_comp(ex)
    adds = comp.get("additives", {}) or {}
    out: List[Tuple[str, str, float]] = []
    if isinstance(adds, dict):
        for name, amount in adds.items():
            nm = norm_name(name)
            if is_placeholder_component_name(nm):
                continue
            smiles = lookup_smiles(nm, additive_lookup or {})
            out.append((nm, smiles, safe_weight(amount, default=1.0)))
    elif isinstance(adds, list):
        for item in adds:
            if isinstance(item, dict):
                for name, amount in item.items():
                    nm = norm_name(name)
                    if is_placeholder_component_name(nm):
                        continue
                    smiles = lookup_smiles(nm, additive_lookup or {})
                    out.append((nm, smiles, safe_weight(amount, default=1.0)))
            elif item not in (None, ""):
                for name in split_known_component_names(item):
                    nm = norm_name(name)
                    smiles = lookup_smiles(nm, additive_lookup or {})
                    out.append((nm, smiles, 1.0))
    elif adds not in (None, "", {}):
        for name in split_known_component_names(adds):
            nm = norm_name(name)
            smiles = lookup_smiles(nm, additive_lookup or {})
            out.append((nm, smiles, 1.0))
    return out


def collect_solvent_names(ex: Dict[str, Any]) -> List[str]:
    comp = get_comp(ex)
    names: List[str] = []
    lst = comp.get("solvents_with_smiles", []) or []
    if isinstance(lst, list):
        for d in lst:
            if isinstance(d, dict):
                name = norm_name(d.get("name", ""))
                if name:
                    names.append(name)
    solvents = comp.get("solvents", {}) or {}
    if isinstance(solvents, dict):
        for name in solvents.keys():
            nm = norm_name(name)
            if nm:
                names.append(nm)
    elif solvents not in (None, "", {}):
        names.extend(split_component_names(solvents))
    return list(dict.fromkeys(names))


def collect_additive_names(ex: Dict[str, Any]) -> List[str]:
    comp = get_comp(ex)
    adds = comp.get("additives", {}) or {}
    names: List[str] = []
    if isinstance(adds, dict):
        names.extend(norm_name(name) for name in adds.keys() if norm_name(name))
    elif isinstance(adds, list):
        for item in adds:
            if isinstance(item, dict):
                names.extend(norm_name(name) for name in item.keys() if norm_name(name))
            elif item not in (None, ""):
                names.extend(split_component_names(item))
    elif adds not in (None, "", {}):
        names.extend(split_component_names(adds))
    return list(dict.fromkeys(names))


def canonicalize_action_name(action: Any) -> str:
    if not isinstance(action, str):
        return ""
    raw = action.strip()
    return raw if raw in A2I else ""


def action_targets(ex: Dict[str, Any]) -> np.ndarray:
    y = np.zeros(len(ACTIONS), dtype=np.float32)
    apr = ex.get("action_prior_reward", {}) or {}
    if not isinstance(apr, dict):
        return y
    merged: Dict[str, float] = {}
    for raw_name, value in apr.items():
        action = canonicalize_action_name(raw_name)
        if not action:
            continue
        merged[action] = max(float(value), merged.get(action, 0.0))
    for a, i in A2I.items():
        y[i] = float(merged.get(a, 0.0))
    return y


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected JSON list.")
    return data


def register_named_smiles_items(
    items: Any,
    specific_lookup: Dict[str, str],
    generic_lookup: Dict[str, str],
) -> None:
    """Register SMILES from either {name: smiles} or [{name, smiles}, ...] formats."""
    if isinstance(items, dict):
        for name, smiles in items.items():
            register_smiles(specific_lookup, name, smiles)
            register_smiles(generic_lookup, name, smiles)
    elif isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            register_smiles(specific_lookup, item.get("name"), item.get("smiles"))
            register_smiles(generic_lookup, item.get("name"), item.get("smiles"))


def register_component_lookup_data(
    data: Any,
    salt_lookup: Dict[str, str],
    solvent_lookup: Dict[str, str],
    additive_lookup: Dict[str, str],
    generic_lookup: Dict[str, str],
) -> int:
    """
    Load SMILES lookups from supported JSON formats.

    Supported formats:
      1. New component_smiles_lookup.json style:
         {"salts": {...}, "components": {...}}
      2. Older component_smiles_lookup.json style:
         {"salts": {...}, "solvents": {...}, "additives": {...}}
      3. excel_codebook.json style:
         [{"entities": {"salts": [{"name": "LiPF6", "smiles": "..."}], ...}}]
    """
    before = len(salt_lookup) + len(solvent_lookup) + len(additive_lookup)

    if isinstance(data, dict):
        register_named_smiles_items(data.get("salts", {}), salt_lookup, generic_lookup)

        components = data.get("components", {}) or {}
        register_named_smiles_items(components, solvent_lookup, generic_lookup)
        register_named_smiles_items(components, additive_lookup, generic_lookup)

        register_named_smiles_items(data.get("solvents", {}), solvent_lookup, generic_lookup)
        register_named_smiles_items(data.get("additives", {}), additive_lookup, generic_lookup)

        entities = data.get("entities", {}) or {}
        if isinstance(entities, dict):
            register_named_smiles_items(entities.get("salts", []), salt_lookup, generic_lookup)
            register_named_smiles_items(entities.get("solvents", []), solvent_lookup, generic_lookup)
            register_named_smiles_items(entities.get("additives", []), additive_lookup, generic_lookup)
            register_named_smiles_items(entities.get("components", []), solvent_lookup, generic_lookup)
            register_named_smiles_items(entities.get("components", []), additive_lookup, generic_lookup)

    elif isinstance(data, list):
        for rec in data:
            entities = (rec or {}).get("entities", {}) or {}
            if not isinstance(entities, dict):
                continue
            register_named_smiles_items(entities.get("salts", []), salt_lookup, generic_lookup)
            register_named_smiles_items(entities.get("solvents", []), solvent_lookup, generic_lookup)
            register_named_smiles_items(entities.get("additives", []), additive_lookup, generic_lookup)
            register_named_smiles_items(entities.get("components", []), solvent_lookup, generic_lookup)
            register_named_smiles_items(entities.get("components", []), additive_lookup, generic_lookup)

    return len(salt_lookup) + len(solvent_lookup) + len(additive_lookup) - before


def build_structure_lookups(
    train_ex: Sequence[Dict[str, Any]],
    codebook_path: Optional[str],
) -> Dict[str, Dict[str, str]]:
    generic_lookup: Dict[str, str] = {}
    salt_lookup: Dict[str, str] = {}
    solvent_lookup: Dict[str, str] = {}
    additive_lookup: Dict[str, str] = {}

    candidate_paths: List[Path] = []
    if codebook_path:
        candidate_paths.append(Path(codebook_path))
        if Path(codebook_path).name != "component_smiles_lookup.json":
            candidate_paths.append(Path("component_smiles_lookup.json"))
    else:
        candidate_paths.extend([Path("component_smiles_lookup.json"), Path("codebook/excel_codebook.json")])

    loaded_any = False
    for path in candidate_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            lookup_data = json.load(f)
        added = register_component_lookup_data(
            lookup_data,
            salt_lookup,
            solvent_lookup,
            additive_lookup,
            generic_lookup,
        )
        loaded_any = True
        print(f"Loaded structure lookup: {path} ({added} typed entries registered)")

    if candidate_paths and not loaded_any:
        print(
            "Warning: no structure lookup file found. Tried: "
            + ", ".join(str(p) for p in candidate_paths)
        )

    for ex in train_ex:
        for name, smiles, _ in get_solvents_with_smiles(ex):
            register_smiles(solvent_lookup, name, smiles)
            register_smiles(generic_lookup, name, smiles)
        for name, smiles, _ in get_additives_with_smiles(ex):
            register_smiles(additive_lookup, name, smiles)
            register_smiles(generic_lookup, name, smiles)

    merged_solvent_lookup = dict(generic_lookup)
    merged_solvent_lookup.update(solvent_lookup)
    merged_additive_lookup = dict(generic_lookup)
    merged_additive_lookup.update(additive_lookup)
    merged_salt_lookup = dict(generic_lookup)
    merged_salt_lookup.update(salt_lookup)
    return {
        "salt_smiles_lookup": merged_salt_lookup,
        "solvent_smiles_lookup": merged_solvent_lookup,
        "additive_smiles_lookup": merged_additive_lookup,
    }


def _record_missing_component(
    bucket: Dict[str, Dict[str, Any]],
    name: Any,
    row_idx: int,
    raw_value: Any,
) -> None:
    key = str(name or "").strip() or "<empty>"
    rec = bucket.setdefault(key, {"count": 0, "rows": [], "examples": []})
    rec["count"] += 1
    if len(rec["rows"]) < 20:
        rec["rows"].append(row_idx)
    raw_text = str(raw_value)
    if raw_text not in rec["examples"] and len(rec["examples"]) < 3:
        rec["examples"].append(raw_text)


def diagnose_missing_structures(
    examples: Sequence[Dict[str, Any]],
    lookups: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    report: Dict[str, Dict[str, Dict[str, Any]]] = {
        "salt": {},
        "solvent": {},
        "additive": {},
        "placeholder_salt": {},
        "placeholder_solvent": {},
        "placeholder_additive": {},
    }
    salt_lookup = lookups.get("salt_smiles_lookup", {}) or {}
    solvent_lookup = lookups.get("solvent_smiles_lookup", {}) or {}
    additive_lookup = lookups.get("additive_smiles_lookup", {}) or {}

    for row_idx, ex in enumerate(examples):
        comp = get_comp(ex)

        raw_salt = comp.get("salt", "")
        salt_names = split_component_names(raw_salt)
        if not salt_names:
            _record_missing_component(report["placeholder_salt"], "<empty>", row_idx, raw_salt)
        for name in salt_names:
            if is_placeholder_component_name(name):
                _record_missing_component(report["placeholder_salt"], name, row_idx, raw_salt)
            elif not lookup_smiles(name, salt_lookup):
                _record_missing_component(report["salt"], name, row_idx, raw_salt)

        for name in collect_solvent_names(ex):
            raw_solvents = comp.get("solvents", comp.get("solvents_with_smiles", ""))
            if is_placeholder_component_name(name):
                _record_missing_component(report["placeholder_solvent"], name, row_idx, raw_solvents)
            elif not lookup_smiles(name, solvent_lookup):
                _record_missing_component(report["solvent"], name, row_idx, raw_solvents)

        for name in collect_additive_names(ex):
            raw_additives = comp.get("additives", "")
            if is_placeholder_component_name(name):
                _record_missing_component(report["placeholder_additive"], name, row_idx, raw_additives)
            elif not lookup_smiles(name, additive_lookup):
                _record_missing_component(report["additive"], name, row_idx, raw_additives)

    return report


def print_missing_structure_report(
    report: Dict[str, Dict[str, Dict[str, Any]]],
    limit_per_section: int = 200,
) -> bool:
    printed_any = False
    section_titles = [
        ("placeholder_salt", "[PLACEHOLDER/EMPTY SALT NAMES]"),
        ("salt", "[MISSING SALT SMILES]"),
        ("placeholder_solvent", "[PLACEHOLDER/EMPTY SOLVENT NAMES]"),
        ("solvent", "[MISSING SOLVENT SMILES]"),
        ("placeholder_additive", "[PLACEHOLDER/EMPTY ADDITIVE NAMES]"),
        ("additive", "[MISSING ADDITIVE SMILES]"),
    ]
    for key, title in section_titles:
        bucket = report.get(key, {}) or {}
        if not bucket:
            continue
        printed_any = True
        total_mentions = sum(int(v.get("count", 0)) for v in bucket.values())
        print(f"\n{title} unique={len(bucket)} mentions={total_mentions}")
        sorted_items = sorted(bucket.items(), key=lambda kv: (-int(kv[1].get("count", 0)), kv[0]))
        for i, (name, rec) in enumerate(sorted_items):
            if i >= limit_per_section:
                print(
                    f"  ... {len(bucket) - limit_per_section} more not shown; "
                    "increase --print_missing_limit to show all"
                )
                break
            rows = ",".join(str(r) for r in rec.get("rows", []))
            examples = " | ".join(rec.get("examples", []))
            print(f"  - {name!r}: count={rec.get('count', 0)}, rows=[{rows}], raw_examples={examples}")
    if not printed_any:
        print("\nAll salt/solvent/additive names in the dataset have SMILES lookups.")
    return printed_any


def infer_cation_token(name: Any, smiles: Any = "") -> str:
    text = f"{norm_name(name)} {norm_name(smiles)}"
    for token, hints in _COMMON_CATIONS:
        if any(h in text for h in hints):
            return token
    return "unknown"


def split_salt_smiles(smiles: str) -> Tuple[str, str]:
    text = str(smiles or "").strip()
    if not text:
        return "", ""
    parts = [p for p in text.split(".") if p]
    if len(parts) <= 1:
        return "", text

    cation_idx = None
    for idx, part in enumerate(parts):
        lower = part.lower()
        if any(hint in lower for _, hints in _COMMON_CATIONS for hint in hints):
            cation_idx = idx
            break
        if "+" in part and "-" not in part:
            cation_idx = idx
            break
    if cation_idx is None:
        cation_idx = 0
    cation = parts[cation_idx]
    anion_parts = [part for idx, part in enumerate(parts) if idx != cation_idx]
    anion = max(anion_parts, key=len) if anion_parts else ""
    return cation, anion or text


def get_morgan_generator(fp_size: int):
    generator = _MORGAN_GENERATORS.get(fp_size)
    if generator is None:
        generator = GetMorganGenerator(radius=2, fpSize=fp_size)
        _MORGAN_GENERATORS[fp_size] = generator
    return generator


def mol_from_smiles(smiles: str):
    text = str(smiles or "").strip()
    if not text:
        return None
    try:
        return Chem.MolFromSmiles(text)
    except Exception:
        return None


def structure_fingerprint(smiles: str, fp_size: int) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid or missing SMILES for fingerprinting: {smiles!r}")
    arr = np.zeros(fp_size, dtype=np.float32)
    fp = get_morgan_generator(fp_size).GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def structure_descriptors(smiles: str) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid or missing SMILES for descriptor featurization: {smiles!r}")
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    vals = [
        Descriptors.MolWt(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        float(atoms.count("F")),
        float(atoms.count("O")),
        float(atoms.count("N")),
        float(atoms.count("S")),
        float(atoms.count("P")),
        float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
    ]
    return np.asarray(vals, dtype=np.float32)


def weighted_pool(items: Sequence[Tuple[str, float]], feature_fn, feature_dim: int) -> np.ndarray:
    vec = np.zeros(feature_dim, dtype=np.float32)
    total = 0.0
    for smiles, weight in items:
        if not smiles:
            continue
        w = float(weight)
        if not np.isfinite(w) or w <= 0:
            continue
        vec += w * feature_fn(smiles)
        total += w
    if total > 0:
        vec /= total
    return vec


def weighted_descriptor_variance(
    items: Sequence[Tuple[str, float]], desc_dim: int, desc_fn=None
) -> np.ndarray:
    desc_fn = desc_fn or structure_descriptors
    descs: List[np.ndarray] = []
    weights: List[float] = []
    for smiles, weight in items:
        if not smiles:
            continue
        w = float(weight)
        if not np.isfinite(w) or w <= 0:
            continue
        descs.append(desc_fn(smiles))
        weights.append(w)
    if len(descs) <= 1:
        return np.zeros(desc_dim, dtype=np.float32)
    D = np.stack(descs, axis=0)
    W = np.asarray(weights, dtype=np.float32)
    W = W / max(float(W.sum()), 1e-8)
    mean = (D * W[:, None]).sum(axis=0)
    return ((D - mean) ** 2 * W[:, None]).sum(axis=0).astype(np.float32)
def mean_pairwise_similarity(
    items: Sequence[Tuple[str,float]],
    fp_size:int
)->np.ndarray:

    fps=[]

    for smiles,_ in items:

        if not smiles:
            continue

        mol=mol_from_smiles(smiles)

        if mol is None:
            continue

        fp=get_morgan_generator(fp_size).GetFingerprint(
            mol
        )

        fps.append(fp)

    if len(fps)<2:
        return np.array(
            [0.0],
            dtype=np.float32
        )

    sims=[]

    for i in range(len(fps)):
        for j in range(i+1,len(fps)):

            sims.append(
                DataStructs.TanimotoSimilarity(
                    fps[i],
                    fps[j]
                )
            )

    return np.array(
        [float(np.mean(sims))],
        dtype=np.float32
    )

def compute_descriptor_scales(
    train_ex: Sequence[Dict[str, Any]], lookups: Dict[str, Dict[str, str]]
) -> Dict[str, List[float]]:
    """Per-descriptor std over the distinct training molecules of each component kind."""
    kinds: Dict[str, set] = {"salt": set(), "solvent": set(), "additive": set()}
    for ex in train_ex:
        for salt_name in split_known_component_names(get_comp(ex).get("salt", "")):
            salt_smiles = lookup_smiles(salt_name, lookups["salt_smiles_lookup"])
            if salt_smiles:
                _, anion_smiles = split_salt_smiles(salt_smiles)
                kinds["salt"].add(anion_smiles or salt_smiles)
        for _, smiles, _ in get_solvents_with_smiles(ex, lookups["solvent_smiles_lookup"]):
            if smiles:
                kinds["solvent"].add(smiles)
        for _, smiles, _ in get_additives_with_smiles(ex, lookups["additive_smiles_lookup"]):
            if smiles:
                kinds["additive"].add(smiles)
    scales: Dict[str, List[float]] = {}
    for kind, smiles_set in kinds.items():
        descs = []
        for smi in sorted(smiles_set):
            try:
                descs.append(structure_descriptors(smi))
            except ValueError:
                continue
        if len(descs) < 2:
            scales[kind] = [1.0] * len(_DESC_NAMES)
            continue
        sd = np.stack(descs).std(axis=0)
        scales[kind] = [float(v) if v > 1e-6 else 1.0 for v in sd]  # constant descriptor -> unscaled
    return scales


def build_structure_obs_spec(
    train_ex: Sequence[Dict[str, Any]],
    conc_min: float,
    conc_max: float,
    codebook_path: Optional[str],
    fp_size: int,
    prebuilt_lookups: Optional[Dict[str, Dict[str, str]]] = None,
    feature_version: str = "structure_v5",
) -> Dict[str, Any]:
    lookups = prebuilt_lookups or build_structure_lookups(train_ex, codebook_path)
    cations = Counter()
    missing_salts = Counter()
    placeholder_salt_examples = 0

    for ex in train_ex:
        comp = get_comp(ex)
        salt_names = split_known_component_names(comp.get("salt", ""))
        if not salt_names:
            placeholder_salt_examples += 1
            continue
        for salt_name in salt_names:
            salt_smiles = lookup_smiles(salt_name, lookups["salt_smiles_lookup"])
            if not salt_smiles:
                missing_salts[salt_name] += 1
                continue
            cations[infer_cation_token(salt_name, salt_smiles)] += 1

    if missing_salts:
        preview = ", ".join(f"{k} ({v})" for k, v in missing_salts.most_common(10))
        print(f"Warning: missing salt SMILES during cation-vocab build: {preview}")
    if placeholder_salt_examples:
        print(
            f"Warning: {placeholder_salt_examples} examples have placeholder/no salt names; "
            "salt features will be zero/unknown for them."
        )

    cation_vocab = [k for k, _ in cations.most_common()]
    if "unknown" not in cation_vocab:
        cation_vocab.append("unknown")

    in_dim = (
        fp_size
        + len(_DESC_NAMES)
        + len(cation_vocab)
        + 1
        + fp_size
        + len(_DESC_NAMES)
        + len(_DESC_NAMES)
        + 1
        + fp_size
        + len(_DESC_NAMES)
        + len(_DESC_NAMES)
        + 2
        
        +1
        +1
    )
    if feature_version in SET_VERSIONS:
        in_dim = v3_in_dim(fp_size, len(_DESC_NAMES), len(cation_vocab))
    elif feature_version in ("structure_v4", "structure_v5"):
        in_dim += 1  # salt scalar = [log-scale concentration, known flag] instead of [c_norm]
        if feature_version == "structure_v5":
            in_dim += len(GOALS)  # goal one-hot
    desc_scale = compute_descriptor_scales(train_ex, lookups)
    return {
        "feature_version": feature_version,
        "desc_scale": desc_scale,
        "feature_mode": "structure",
        "feature_backend": "rdkit",
        "fp_size": int(fp_size),
        "desc_names": list(_DESC_NAMES),
        "cation_vocab": cation_vocab,
        "salt_smiles_lookup": lookups["salt_smiles_lookup"],
        "solvent_smiles_lookup": lookups["solvent_smiles_lookup"],
        "additive_smiles_lookup": lookups["additive_smiles_lookup"],
        "conc_min": float(conc_min),
        "conc_max": float(conc_max),
        "in_dim": int(in_dim),
        "requires_solvents_with_smiles": True,
        **({"v3_slots": dict(V3_SLOTS)} if feature_version in SET_VERSIONS else {}),
        **({"log_conc_range": list(V3_LOG_CONC_RANGE)}
           if feature_version in SET_VERSIONS + ("structure_v4", "structure_v5") else {}),
        **({"goals": list(GOALS)} if feature_version == "structure_v5" else {}),
    }


# v4/v5: pooled v2 input with the salt fix; v5 adds the goal of the modification.
POOLED_FIXED_SALT_VERSIONS = ("structure_v4", "structure_v5")
FEATURE_V5 = "structure_v5"
GOALS = ("high_voltage", "wide_temperature", "other")


def goal_vector(objective: Any) -> np.ndarray:
    """Multi-hot goal of the modification, from a dataset objective or an agent objective.

    Dataset rows carry exactly one of GOALS as objective.primary. An agent objective such as
    {"primary": "high_voltage_oxidative_stability", ...} is mapped by keyword: voltage ->
    high_voltage, temperature -> wide_temperature, neither -> other.
    """
    vec = np.zeros(len(GOALS), dtype=np.float32)
    if isinstance(objective, dict):
        texts = [str(objective.get("primary", ""))] + [str(t) for t in (objective.get("tags") or [])]
    elif objective:
        texts = [str(objective)]
    else:
        texts = []
    keys = {t.strip().lower() for t in texts if t and t.strip()}
    if keys & set(GOALS):
        for g in keys & set(GOALS):
            vec[GOALS.index(g)] = 1.0
        return vec
    blob = " ".join(keys)
    if "voltage" in blob:
        vec[GOALS.index("high_voltage")] = 1.0
    if "temperature" in blob or "low-temp" in blob or "all-climate" in blob:
        vec[GOALS.index("wide_temperature")] = 1.0
    if not vec.any():
        vec[GOALS.index("other")] = 1.0
    return vec


def encode_obs_structure(ex: Dict[str, Any], obs_spec: Dict[str, Any]) -> np.ndarray:
    fp_size = int(obs_spec["fp_size"])
    desc_dim = len(obs_spec["desc_names"])
    cation_vocab = list(obs_spec["cation_vocab"])
    salt_lookup = obs_spec.get("salt_smiles_lookup", {}) or {}
    solvent_lookup = obs_spec.get("solvent_smiles_lookup", {}) or {}
    additive_lookup = obs_spec.get("additive_smiles_lookup", {}) or {}

    scale = obs_spec.get("desc_scale") or {}

    def _desc_fn(kind):
        s = scale.get(kind)
        if not s:
            return structure_descriptors  # v1 spec: raw descriptors
        s_arr = np.asarray(s, dtype=np.float32)
        return lambda smi: structure_descriptors(smi) / s_arr

    salt_desc_fn, solvent_desc_fn, additive_desc_fn = _desc_fn("salt"), _desc_fn("solvent"), _desc_fn("additive")

    v4 = obs_spec.get("feature_version") in POOLED_FIXED_SALT_VERSIONS
    _, conc = get_salt_and_conc(ex)
    conc_min = float(obs_spec["conc_min"])
    conc_max = float(obs_spec["conc_max"])
    if np.isfinite(conc):
        c_norm = float(np.clip((conc - conc_min) / max(conc_max - conc_min, 1e-8), 0.0, 1.0))
    else:
        c_norm = 0.0

    salt_names = split_known_component_names(get_comp(ex).get("salt", ""))
    cation_vec = np.zeros(len(cation_vocab), dtype=np.float32)
    salt_items: List[Tuple[str, float]] = []
    named_salts: List[str] = []
    for salt_name in salt_names:
        salt_smiles = lookup_smiles(salt_name, salt_lookup)
        if not salt_smiles:
            continue
        cation_token = infer_cation_token(salt_name, salt_smiles)
        if cation_token in cation_vocab:
            cation_vec[cation_vocab.index(cation_token)] += 1.0
        _, anion_smiles = split_salt_smiles(salt_smiles)
        salt_items.append((anion_smiles or salt_smiles, 1.0))
        named_salts.append(salt_name)

    if float(cation_vec.sum()) > 0:
        cation_vec /= float(cation_vec.sum())
    elif "unknown" in cation_vocab:
        cation_vec[cation_vocab.index("unknown")] = 1.0

    if v4:
        # v4: the anion block describes WHICH salt (molar-share weights for co-salts) and is
        # not scaled by concentration; concentration is one log-scale scalar + known flag.
        molar = _salt_molar_amounts(get_comp(ex).get("salt_concentration"), named_salts)
        if molar:
            salt_items = [(smi, w) for (smi, _), w in zip(salt_items, molar)]
            conc = float(sum(molar))
        lo, hi = V3_LOG_CONC_RANGE
        if np.isfinite(conc) and conc > 0:
            salt_scalar = np.asarray([float(np.clip(np.log(conc / lo) / np.log(hi / lo), 0.0, 1.0)), 1.0],
                                     dtype=np.float32)
        else:
            salt_scalar = np.asarray([0.0, 0.0], dtype=np.float32)
        salt_fp = weighted_pool(salt_items, lambda s: structure_fingerprint(s, fp_size), fp_size)
        salt_desc = weighted_pool(salt_items, salt_desc_fn, desc_dim)
    else:
        salt_fp = weighted_pool(salt_items, lambda s: structure_fingerprint(s, fp_size), fp_size) * c_norm
        salt_desc = weighted_pool(salt_items, salt_desc_fn, desc_dim) * c_norm
        salt_scalar = np.asarray([c_norm], dtype=np.float32)

    solvent_entries = get_solvents_with_smiles(ex, solvent_lookup)
    solvent_items = [(smiles, weight) for _, smiles, weight in solvent_entries]
    solvent_fp = weighted_pool(solvent_items, lambda s: structure_fingerprint(s, fp_size), fp_size)
    solvent_desc_mean = weighted_pool(solvent_items, solvent_desc_fn, desc_dim)
    solvent_desc_var = weighted_descriptor_variance(solvent_items, desc_dim, solvent_desc_fn)
    solvent_count = np.asarray([min(len(solvent_entries) / 6.0, 1.0)], dtype=np.float32)

    additive_entries = get_additives_with_smiles(ex, additive_lookup)
    additive_items = [(smiles, weight) for _, smiles, weight in additive_entries]
    additive_fp = weighted_pool(additive_items, lambda s: structure_fingerprint(s, fp_size), fp_size)
    additive_desc_mean = weighted_pool(additive_items, additive_desc_fn, desc_dim)
    additive_desc_var = weighted_descriptor_variance(additive_items, desc_dim, additive_desc_fn)
    additive_scalars = np.asarray(
        [
            1.0 if additive_entries else 0.0,
            min(len(additive_entries) / 4.0, 1.0),
        ],
        dtype=np.float32,
    )
    solvent_similarity=mean_pairwise_similarity(
        solvent_items,
        fp_size
    )

    additive_similarity=mean_pairwise_similarity(
        additive_items,
        fp_size
    )

    parts = [
        salt_fp,
        salt_desc,
        cation_vec,
        salt_scalar,
        solvent_fp,
        solvent_desc_mean,
        solvent_desc_var,
        solvent_count,
        additive_fp,
        additive_desc_mean,
        additive_desc_var,
        additive_scalars,
        solvent_similarity,
        additive_similarity,
    ]
    if obs_spec.get("feature_version") == FEATURE_V5:
        parts.append(goal_vector(ex.get("objective")))   # goal of the modification
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# structure_v3_set: per-molecule set encoding
#
# Every molecule (salt anion, solvent, additive) is represented by its own
# [Morgan fingerprint || scaled descriptors] vector, kept per slot instead of being
# pooled in the input. The model (SetPrior) passes every molecule through one shared
# molecule encoder and pools inside the network:
#   salt      sum_i w_i psi(anion_i), w = molar share (equal if not parseable); the
#             concentration does NOT multiply the anion features
#   solvents  [ sum_i x_i psi(m_i)  ||  mean_i psi(m_i) ], x = volume/mole fraction
#   additives same as solvents, weights from the reported amounts
# Concentration enters separately as one log-scale scalar (+ a "known" flag), next to
# the cation vector. Each component group is LayerNorm-ed before concatenation, so no
# group dominates by size and their relative influence is learned.
# ---------------------------------------------------------------------------
FEATURE_V3 = "structure_v3_set"   # set encoder, nonlinear molecule encoder
SET_VERSIONS = (FEATURE_V3,)
V3_SLOTS = {"salt": 2, "solvent": 8, "additive": 4}
V3_LOG_CONC_RANGE = (0.1, 6.0)  # mol/L; log-scaled to [0, 1]
_MOLAR_RE = re.compile(
    r"(\d*\.?\d+)\s*(?:m\b|M\b|mol\s*/?\s*(?:L|l|dm|kg)|mol\s*L|mol\s*kg)",
)


def _salt_molar_amounts(conc_text: Any, salt_names: Sequence[str]) -> Optional[List[float]]:
    """Molar amount of each salt from e.g. '1.0 M LiPF6 + 0.1 M LiODFB'; None if not all parse."""
    text = str(conc_text or "")
    if len(salt_names) <= 1:
        return None
    out: List[float] = []
    low = text.lower()
    for name in salt_names:
        pos = low.find(norm_name(name))
        if pos < 0:
            return None
        found = None
        for mt in _MOLAR_RE.finditer(text[:pos]):
            found = float(mt.group(1))
        if found is None:
            return None
        out.append(found)
    return out if all(v > 0 for v in out) else None


def _slot_items(items: Sequence[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
    """Valid (smiles, weight) items, at most k (largest weights kept), weights summing to 1."""
    valid = [(s, float(w)) for s, w in items if s and np.isfinite(w) and w > 0]
    valid.sort(key=lambda t: -t[1])
    valid = valid[:k]
    tot = sum(w for _, w in valid)
    return [(s, w / tot) for s, w in valid] if tot > 0 else []


def v3_mol_dim(obs_spec: Dict[str, Any]) -> int:
    return int(obs_spec["fp_size"]) + len(obs_spec["desc_names"])


def v3_in_dim(fp_size: int, desc_dim: int, n_cation: int) -> int:
    n_slots = sum(V3_SLOTS.values())
    return n_slots * (fp_size + desc_dim) + n_slots + n_cation + 2


def encode_obs_set(ex: Dict[str, Any], obs_spec: Dict[str, Any]) -> np.ndarray:
    fp_size = int(obs_spec["fp_size"])
    desc_dim = len(obs_spec["desc_names"])
    mol_dim = fp_size + desc_dim
    cation_vocab = list(obs_spec["cation_vocab"])
    scale = obs_spec.get("desc_scale") or {}

    def mol_vec(smiles: str, kind: str) -> np.ndarray:
        d = structure_descriptors(smiles)
        s = scale.get(kind)
        if s:
            d = d / np.asarray(s, dtype=np.float32)
        return np.concatenate([structure_fingerprint(smiles, fp_size), d]).astype(np.float32)

    # salt: anion identity, molar-share weights, cation vector
    comp = get_comp(ex)
    salt_names = split_known_component_names(comp.get("salt", ""))
    cation_vec = np.zeros(len(cation_vocab), dtype=np.float32)
    named: List[Tuple[str, str]] = []
    for salt_name in salt_names:
        salt_smiles = lookup_smiles(salt_name, obs_spec.get("salt_smiles_lookup", {}) or {})
        if not salt_smiles:
            continue
        token = infer_cation_token(salt_name, salt_smiles)
        if token in cation_vocab:
            cation_vec[cation_vocab.index(token)] += 1.0
        _, anion = split_salt_smiles(salt_smiles)
        named.append((salt_name, anion or salt_smiles))
    if float(cation_vec.sum()) > 0:
        cation_vec /= float(cation_vec.sum())
    elif "unknown" in cation_vocab:
        cation_vec[cation_vocab.index("unknown")] = 1.0
    molar = _salt_molar_amounts(comp.get("salt_concentration"), [n for n, _ in named])
    salt_w = molar if molar else [1.0] * len(named)
    salt_items = [(smi, w) for (_, smi), w in zip(named, salt_w)]

    solvent_items = [(s, w) for _, s, w in get_solvents_with_smiles(ex, obs_spec.get("solvent_smiles_lookup", {}) or {})]
    additive_items = [(s, w) for _, s, w in get_additives_with_smiles(ex, obs_spec.get("additive_smiles_lookup", {}) or {})]

    blocks: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    for kind, items in (("salt", salt_items), ("solvent", solvent_items), ("additive", additive_items)):
        k = V3_SLOTS[kind]
        feats = np.zeros((k, mol_dim), dtype=np.float32)
        w = np.zeros(k, dtype=np.float32)
        for j, (smi, wt) in enumerate(_slot_items(items, k)):
            feats[j] = mol_vec(smi, kind)
            w[j] = wt
        blocks.append(feats.reshape(-1))
        weights.append(w)

    # concentration: one log-scale scalar + known flag (total molar salt if parseable)
    _, conc = get_salt_and_conc(ex)
    if molar:
        conc = float(sum(molar))
    lo, hi = V3_LOG_CONC_RANGE
    if np.isfinite(conc) and conc > 0:
        c_log = float(np.clip(np.log(conc / lo) / np.log(hi / lo), 0.0, 1.0))
        known = 1.0
    else:
        c_log, known = 0.0, 0.0

    return np.concatenate(
        blocks + weights + [cation_vec, np.asarray([c_log, known], dtype=np.float32)]
    ).astype(np.float32)


def encode_obs(
    ex: Dict[str, Any],
    obs_spec: Dict[str, Any],
) -> np.ndarray:
    if obs_spec.get("feature_version") in SET_VERSIONS:
        return encode_obs_set(ex, obs_spec)
    return encode_obs_structure(ex, obs_spec)


class PriorDataset(Dataset):
    def __init__(self, examples: Sequence[Dict[str, Any]], obs_spec: Dict[str, Any]):
        self.ex = list(examples)
        self.obs_spec = obs_spec
        # encode once (RDKit featurization is deterministic), not on every epoch
        self.X = [torch.from_numpy(encode_obs(ex, obs_spec)) for ex in self.ex]
        self.Y = [torch.from_numpy(action_targets(ex)) for ex in self.ex]

    def __len__(self) -> int:
        return len(self.ex)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


class MLP(nn.Module):

    def __init__(
        self,
        in_dim
    ):

        super().__init__()

        self.net=nn.Sequential(

            nn.Linear(
                in_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(.15),

            nn.Linear(
                512,
                512
            ),

            nn.ReLU(),

            nn.Dropout(.1),

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                len(ACTIONS)
            )
        )

    def forward(self,x):
        return self.net(x)


class SetPrior(nn.Module):
    """structure_v3_set model: shared molecule encoder, in-network pooling per component
    group, one LayerNorm per group, then the action head. Input = encode_obs_set vector."""

    def __init__(self, obs_spec: Dict[str, Any], mol_hidden: int = 256, mol_out: int = 128):
        super().__init__()
        self.mol_dim = v3_mol_dim(obs_spec)
        self.slots = [V3_SLOTS["salt"], V3_SLOTS["solvent"], V3_SLOTS["additive"]]
        self.n_cation = len(obs_spec["cation_vocab"])
        self.mol_encoder = "mlp"
        self.psi = nn.Sequential(
            nn.Linear(self.mol_dim, mol_hidden), nn.ReLU(), nn.Dropout(.1),
            nn.Linear(mol_hidden, mol_out), nn.ReLU(),
        )
        self.ln_salt = nn.LayerNorm(mol_out)          # weighted anion
        self.ln_solv = nn.LayerNorm(2 * mol_out)      # weighted || unweighted mean
        self.ln_add = nn.LayerNorm(2 * mol_out)
        head_in = mol_out + 2 * mol_out + 2 * mol_out + self.n_cation + 2
        self.head = nn.Sequential(
            nn.Linear(head_in, 512), nn.ReLU(), nn.Dropout(.15),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.out = nn.Linear(256, len(ACTIONS))

    def _split(self, x):
        b, off = x.shape[0], 0
        feats = []
        for k in self.slots:
            feats.append(x[:, off:off + k * self.mol_dim].reshape(b, k, self.mol_dim))
            off += k * self.mol_dim
        weights = []
        for k in self.slots:
            weights.append(x[:, off:off + k])
            off += k
        cation = x[:, off:off + self.n_cation]
        off += self.n_cation
        conc = x[:, off:off + 2]
        return feats, weights, cation, conc

    def _pool(self, feats, w):
        mask = (w > 0).float()
        h = self.psi(feats) * mask.unsqueeze(-1)                 # absent slots -> 0 (incl. linear bias)
        weighted = (h * w.unsqueeze(-1)).sum(1)                  # weights already sum to 1
        mean = h.sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        return weighted, mean

    def embed(self, x):
        """256-d representation before the output layer (for analysis / adapters)."""
        (f_salt, f_solv, f_add), (w_salt, w_solv, w_add), cation, conc = self._split(x)
        s_w, _ = self._pool(f_salt, w_salt)
        v_w, v_m = self._pool(f_solv, w_solv)
        a_w, a_m = self._pool(f_add, w_add)
        z = torch.cat([
            self.ln_salt(s_w),
            self.ln_solv(torch.cat([v_w, v_m], -1)),
            self.ln_add(torch.cat([a_w, a_m], -1)),
            cation, conc,
        ], -1)
        return self.head(z)

    def forward(self, x):
        return self.out(self.embed(x))


def build_model(obs_spec: Dict[str, Any]) -> nn.Module:
    """Model matching the checkpoint's feature_version (v1/v2/v4/v5 -> MLP, v3 -> SetPrior)."""
    if obs_spec.get("feature_version") in SET_VERSIONS:
        return SetPrior(obs_spec)
    return MLP(in_dim=int(obs_spec["in_dim"]))


def ranking_metrics(scores: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    top1 = top3 = 0
    mrr = 0.0
    valid = 0
    for i in range(scores.shape[0]):
        pos = np.where(y[i] > 0.5)[0]
        if len(pos) == 0:
            continue
        valid += 1
        order = np.argsort(-scores[i])
        top1 += int(order[0] in pos)
        top3 += int(any(o in pos for o in order[:3]))
        ranks = {a: r + 1 for r, a in enumerate(order)}
        mrr += 1.0 / min(ranks[p] for p in pos)
    if valid == 0:
        return {"top1": np.nan, "top3": np.nan, "mrr": np.nan, "valid": 0}
    return {"top1": top1 / valid, "top3": top3 / valid, "mrr": mrr / valid, "valid": valid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="mlp_prior_stateflag.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conc_min", type=float, default=0.5)
    ap.add_argument("--conc_max", type=float, default=2.0)
    ap.add_argument(
        "--feature_version",
        type=str,
        default="structure_v5",
        choices=["structure_v2_descscale", FEATURE_V3, "structure_v4", "structure_v5"],
        help="structure_v5 (default): v4 + one-hot goal of the modification (objective.primary: "
             "high_voltage / wide_temperature / other); "
             "structure_v4: v2 pooled input with the salt fix (anion features not "
             "scaled by concentration; log-scale concentration scalar; molar-share salt weights); "
             "structure_v3_set: per-molecule set encoder; structure_v2_descscale: original pooled input.",
    )
    ap.add_argument(
        "--fp_size",
        type=int,
        default=1024,
        help="Fingerprint bit size when feature_mode=structure.",
    )
    ap.add_argument(
        "--structure_codebook",
        type=str,
        default="component_smiles_lookup.json",
        help=(
            "Optional SMILES lookup JSON. Supports new top-level salts/components, "
            "old salts/solvents/additives, or older codebook/excel_codebook.json list format."
        ),
    )
    ap.add_argument(
        "--allow_missing_structures",
        action="store_true",
        help=(
            "Continue training even if some salt/solvent/additive names are missing "
            "from the SMILES lookup. Missing structures are encoded as zero vectors."
        ),
    )
    ap.add_argument(
        "--print_missing_limit",
        type=int,
        default=200,
        help="Maximum number of missing component names to print per section.",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    raw = load_json_list(args.data)
    raw = [
        ex
        for ex in raw
        if isinstance(ex.get("action_prior_reward", None), dict) and isinstance(get_comp(ex), dict)
    ]
    if not raw:
        raise ValueError("No usable examples found.")

    structure_lookups = build_structure_lookups(raw, args.structure_codebook)
    missing_report = diagnose_missing_structures(raw, structure_lookups)
    has_missing_structures = print_missing_structure_report(
        missing_report,
        limit_per_section=max(1, int(args.print_missing_limit)),
    )
    if has_missing_structures and not args.allow_missing_structures:
        raise ValueError(
            "Some components have no SMILES lookup or are placeholder values. "
            "The missing names were printed above. Add them to component_smiles_lookup.json "
            "or rerun with --allow_missing_structures to encode them as zero vectors."
        )

    train_ex, val_ex = train_test_split(raw, test_size=0.15, random_state=args.seed, shuffle=True)
    obs_spec = build_structure_obs_spec(
        train_ex=train_ex,
        conc_min=float(args.conc_min),
        conc_max=float(args.conc_max),
        codebook_path=args.structure_codebook,
        fp_size=int(args.fp_size),
        prebuilt_lookups=structure_lookups,
        feature_version=args.feature_version,
    )

    in_dim = int(obs_spec["in_dim"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Train:", len(train_ex), "Val:", len(val_ex))
    print("Feature mode:", obs_spec.get("feature_mode"), "| backend:", obs_spec.get("feature_backend"))
    print("Input dim:", in_dim)
    print(
        "Structure lookups | salt:",
        len(obs_spec.get("salt_smiles_lookup", {})),
        "solvent:",
        len(obs_spec.get("solvent_smiles_lookup", {})),
        "additive:",
        len(obs_spec.get("additive_smiles_lookup", {})),
        "fp_size:",
        obs_spec.get("fp_size"),
        "desc_dim:",
        len(obs_spec.get("desc_names", [])),
    )

    train_ds = PriorDataset(train_ex, obs_spec)
    val_ds = PriorDataset(val_ex, obs_spec)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(obs_spec).to(device)
    print("Feature version:", obs_spec["feature_version"], "| model:", type(model).__name__,
          "| molecule encoder:", getattr(model, "mol_encoder", "-"))

    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)

    Y=np.stack([
        action_targets(x)
        for x in train_ex])

    freq=Y.sum(0)

    pos_weight=(
        len(Y)-freq
    )/(freq+1e-6)

    print(
        "\nPos weights:"
    )

    for a,w in zip(
        ACTIONS,
        pos_weight
    ):
        print(
            a,
            round(float(w),2)
        )

    pos_weight=torch.tensor(
        pos_weight,
        dtype=torch.float32
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"epoch {ep}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

    model.eval()
    Xv, Yv = [], []
    for x, y in val_dl:
        Xv.append(x.numpy())
        Yv.append(y.numpy())
    Xv = np.concatenate(Xv)
    Yv = np.concatenate(Yv)
    with torch.no_grad():
        logits = model(torch.from_numpy(Xv).to(device)).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    yhat = (probs >= 0.5).astype(np.int32)

    micro_f1 = f1_score(Yv.reshape(-1), yhat.reshape(-1), average="micro")
    rank = ranking_metrics(probs, Yv)
    metrics = {"micro_f1": float(micro_f1), **{k: float(v) for k, v in rank.items()}}

    print("\n[VAL METRICS]")
    for k, v in metrics.items():
        print(f"{k:>10s}: {v}")

    ckpt = {
        "state_dict": model.state_dict(),
        "actions": ACTIONS,
        "metrics": metrics,
        "obs_spec": obs_spec,
        "config": {
            "hidden": args.hidden,
            "seed": args.seed,
            "fp_size": args.fp_size,
            "structure_codebook": args.structure_codebook,
        },
    }
    torch.save(ckpt, args.out)
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()
