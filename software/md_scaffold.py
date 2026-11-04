"""
MD scaffold — detect a local GROMACS toolchain and, for a given formulation,
generate a ready-to-finish job directory from the user's template set.

Per the chosen scope (scaffold + job directory, generate-not-submit): this builds
the folder, a packmol box input, a topol.top with the right [ molecules ] counts,
a chained run.sh (RESP2 charges -> packmol -> gmx em/NVT/NPT/prod -> analysis) and
a MANIFEST listing the per-species structure files the user MUST drop in. It does
NOT run or submit anything, and it does NOT invent molecular structures or force-
field parameters — that step is delegated to the user's RESP2.sh / GAFF toolchain,
which is exactly where the .itp files come from.
"""
from __future__ import annotations

import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config

# --- rough scaffold constants (documented; adjust for your system) ---
N_SOLVENT_TOTAL = 500        # total solvent molecules packed in the box
SOLVENT_MOLAR_VOLUME_L = 0.10  # ~L/mol, used only to estimate salt count from molarity
VOL_PER_MOLECULE_NM3 = 0.16    # ~nm^3/molecule, used only for a starting box size


def detect_md_tools() -> Dict[str, Optional[str]]:
    """Return {gmx, packmol, multiwfn} -> path or None (first candidate found)."""
    found: Dict[str, Optional[str]] = {}
    for key, candidates in config.MD_EXECUTABLES.items():
        found[key] = next((shutil.which(c) for c in candidates if shutil.which(c)), None)
    return found


def templates_ok(templates_dir: Path) -> Tuple[bool, List[str]]:
    missing = [f for f in config.MD_REQUIRED_TEMPLATES if not (templates_dir / f).exists()]
    return (not missing), missing


def md_available(templates_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """MD is 'available' only if the gmx engine is on PATH AND templates are present."""
    tools = detect_md_tools()
    ok_tpl, missing = templates_ok(templates_dir)
    available = bool(tools.get("gmx")) and ok_tpl
    return available, {"tools": tools, "templates_ok": ok_tpl, "missing_templates": missing}


def _safe(name: str) -> str:
    """Species -> filesystem-safe token (strip parenthetical abbreviations, punctuation)."""
    base = re.sub(r"\(.*?\)", "", str(name)).strip()
    base = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")
    return base or "SP"


def _num(x: Any, default: float = 0.0) -> float:
    m = re.search(r"[-+]?\d*\.?\d+", str(x).replace("−", "-"))
    return float(m.group(0)) if m else default


def _split_salt(salt: str) -> Tuple[str, str]:
    """Very small Li-salt splitter: 'LiPF6' -> ('Li','PF6'). Falls back to (salt, '')."""
    s = str(salt).strip()
    if s.lower().startswith("li"):
        anion = s[2:] or "anion"
        return "Li", anion
    return s, ""


def parse_species(comp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn a formulation composition into a species list with packmol counts."""
    solvents = comp.get("solvents", {}) or {}
    ratios = {k: _num(v, 1.0) for k, v in solvents.items()}
    total_r = sum(ratios.values()) or 1.0
    species: List[Dict[str, Any]] = []
    for name, r in ratios.items():
        species.append({"name": name, "token": _safe(name), "role": "solvent",
                        "count": max(1, round(r / total_r * N_SOLVENT_TOTAL)), "charge": 0})

    conc = _num(comp.get("salt_concentration"), 1.0)
    # salt_count estimate: N_solvent * molarity * solvent_molar_volume  (documented heuristic)
    salt_count = max(1, round(N_SOLVENT_TOTAL * conc * SOLVENT_MOLAR_VOLUME_L))
    cation, anion = _split_salt(comp.get("salt", ""))
    species.append({"name": cation, "token": _safe(cation), "role": "cation",
                    "count": salt_count, "charge": +1})
    if anion:
        species.append({"name": anion, "token": _safe(anion), "role": "anion",
                        "count": salt_count, "charge": -1})

    for name, amt in (comp.get("additives", {}) or {}).items():
        species.append({"name": name, "token": _safe(name), "role": "additive",
                        "count": max(1, round(_num(amt, 1.0))), "charge": 0})
    return species


def _box_nm(species: List[Dict[str, Any]]) -> float:
    n_total = sum(s["count"] for s in species)
    return round((n_total * VOL_PER_MOLECULE_NM3) ** (1 / 3), 2)


def _packmol_inp(species, box_nm: float) -> str:
    box_ang = box_nm * 10.0
    lines = ["tolerance 2.0", "filetype pdb", "output box.pdb", ""]
    for s in species:
        lines += [f"structure {s['token']}.pdb",
                  f"  number {s['count']}",
                  f"  inside box 0. 0. 0. {box_ang:.1f} {box_ang:.1f} {box_ang:.1f}",
                  "end structure", ""]
    return "\n".join(lines)


def _topol_top(template_top: Path, species) -> str:
    """Reuse the template's [defaults]/[atomtypes] header, then write includes +
    [ molecules ] for THIS formulation. Per-species [atomtypes]/[moleculetype] come
    from the .itp files RESP2.sh/GAFF produce (see MANIFEST)."""
    header = template_top.read_text(encoding="utf-8").split("#include")[0].rstrip()
    includes = "\n".join(f'#include "{s["token"]}.itp"' for s in species)
    mols = "\n".join(f'{s["token"]:<8s} {s["count"]}' for s in species)
    return (f"{header}\n\n{includes}\n\n[ system ]\nelectrolyte\n\n"
            f"[ molecules ]\n; Molecule   nmols\n{mols}\n")


def _run_sh(species, templates_present: bool) -> str:
    charge_steps = "\n".join(
        f'bash RESP2.sh {s["token"]}.pdb {s["charge"]}   # -> {s["token"]}.itp (charges); then GAFF/antechamber'
        for s in species)
    return f"""#!/bin/bash
set -euo pipefail
# Auto-generated MD job. GENERATE-ONLY: review, drop in the MANIFEST files, then run on your server.

# 1) Per-species force field: RESP2 charges (Gaussian+Multiwfn) then build each .itp
{charge_steps}

# 2) Build the initial box
packmol < packmol.inp                      # -> box.pdb

# 3) GROMACS: minimise -> NVT -> NPT -> production
gmx grompp -f em.mdp    -c box.pdb -p topol.top -o em.tpr   -maxwarn 5
gmx mdrun  -deffnm em
gmx grompp -f md_NVT.mdp -c em.gro  -p topol.top -o nvt.tpr  -maxwarn 5
gmx mdrun  -deffnm nvt
gmx grompp -f md_NPT.mdp -c nvt.gro -p topol.top -o npt.tpr  -maxwarn 5
gmx mdrun  -deffnm npt
gmx grompp -f prod.mdp   -c npt.gro -p topol.top -o prod.tpr -maxwarn 5
gmx mdrun  -deffnm prod

# 4) Analysis -> solvation descriptors (AGG / CN_PF6 / SSIP / sigma ...)
#    Plug your analysis here; read the descriptors, then return positive/negative/nochange
#    to the running program.
echo "MD finished. Compute descriptors, then enter feedback in the program."
"""


def generate_job(formulation: Dict[str, Any], templates_dir: Path, out_root: Path,
                 step: int, objective: str = "") -> Path:
    comp = formulation.get("composition") or (formulation.get("state0") or {}).get("composition") or {}
    species = parse_species(comp)
    box = _box_nm(species)

    jobdir = out_root / f"step_{step:02d}"
    n = 1
    while jobdir.exists():
        jobdir = out_root / f"step_{step:02d}_{n}"; n += 1
    jobdir.mkdir(parents=True)

    # copy runnable templates
    for f in ("em.mdp", "md_NVT.mdp", "md_NPT.mdp", "prod.mdp", "RESP2.sh"):
        src = templates_dir / f
        if src.exists():
            shutil.copy(src, jobdir / f)

    (jobdir / "packmol.inp").write_text(_packmol_inp(species, box), encoding="utf-8")
    tpl_top = templates_dir / "topol.top"
    if tpl_top.exists():
        (jobdir / "topol.top").write_text(_topol_top(tpl_top, species), encoding="utf-8")
    (jobdir / "run.sh").write_text(_run_sh(species, tpl_top.exists()), encoding="utf-8")

    manifest = ["# MANIFEST — files YOU must place in this folder before running", "",
                f"Objective: {objective}", f"Box: {box} nm cube (starting guess)", "",
                "Required per-species structure files (one 3D .pdb each):"]
    for s in species:
        manifest.append(f"  - {s['token']}.pdb   ({s['role']}, {s['name']}, count={s['count']}, net charge {s['charge']:+d})")
    manifest += ["", "Force-field fragments (produced by RESP2.sh + GAFF/antechamber):"]
    manifest += [f"  - {s['token']}.itp" for s in species]
    manifest += ["", "Then: bash run.sh  (on a machine with gmx/packmol/Multiwfn).",
                 "This program does NOT submit the job — you run it and report the result."]
    (jobdir / "MANIFEST.md").write_text("\n".join(manifest), encoding="utf-8")
    return jobdir
