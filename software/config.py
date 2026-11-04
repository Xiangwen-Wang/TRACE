"""
Central paths / knobs for the packaged electrolyte-formulation program.

Everything is resolved relative to this file so the `software/` folder is
self-contained and can be moved anywhere.
"""
from __future__ import annotations

from pathlib import Path

PKG = Path(__file__).resolve().parent

# --- engine assets (copied into the package, read by module-dir relative paths) ---
PRIOR_CKPT = PKG / "MLP_new" / "mlp_prior_state_new_v5.pt"   # frozen literature prior (never modified)
BASE_CORPUS = PKG / "literature_database.json"      # processed literature RAG database
MD_TEMPLATES = PKG / "md_templates"                 # em/NVT/NPT/prod mdp, topol.top, RESP2.sh

# --- persistent learning store (survives across runs; everything learned accumulates here) ---
STORE = PKG / "store"
FEEDBACK_LEDGER = STORE / "feedback_ledger.jsonl"      # every feedback event, append-only (source of truth)
DISCOVERED_PAPERS = STORE / "discovered_papers.json"   # ledger -> precedents (positive + negative), rebuilt
MERGED_CORPUS = STORE / "merged_corpus.json"           # base corpus + discovered papers (fed to the RAG)
POSTERIOR_STATE = STORE / "action_posterior.json"      # tier 1: evidence counts behind the prior shift
MODELS = STORE / "models"                              # tier 2: versioned adapter checkpoints
REGISTRY = STORE / "learning_registry.json"            # active learner/version + update history
KEYFILE = STORE / ".openai_key"                        # optional remembered OpenAI key (chmod 600)

# --- per-run working area ---
RUNS = PKG / "runs"
MD_JOBS = PKG / "md_jobs"

# --- run defaults ---
DEFAULT_MAX_STEPS = 20
DEFAULT_LLM_MODEL = "gpt-5"

# --- feedback memory / continual learning ---
# Ask for optional evidence (MD descriptors or a note) after each feedback. It is stored in the
# ledger and written into the precedent, so later runs can read WHY a move was accepted/rejected.
ASK_EVIDENCE = True
# Write that evidence into the precedent conditions the generator reads. False keeps it in the
# ledger only, so the agent stays blind to MD descriptors (the main-text MD-in-the-loop protocol).
EVIDENCE_TO_RAG = True
# Write negative / no-change feedback back as precedents too (not only accepted moves).
WRITE_BACK_NONPOSITIVE = True
# Make written-back precedents retrievable in the SAME session (not only from the next run).
LIVE_WRITE_BACK = True
# Selection-time learning. "off": frozen literature prior only. "auto": add the active learner's
# bounded logit shift (tier 1 posterior, or the tier 2 adapter once it has been promoted).
ONLINE_PRIOR_SHIFT = "auto"
# Tier 1 (posterior): pseudo-count of each backoff level, and the cap on |logit shift|.
POSTERIOR_STRENGTH = 4.0
MAX_LOGIT_SHIFT = 1.5
UNCHANGE_WEIGHT = 0.5          # a no-change counts as half a non-success observation
# Tier 2 (adapter): data gate before training is attempted, and the time-ordered hold-out.
ADAPTER_MIN_EVENTS = 60
ADAPTER_MIN_POSITIVES = 10
ADAPTER_HOLDOUT_FRAC = 0.2     # newest events are held out for the promotion test

# MD tool executables we probe for (first found wins).
MD_EXECUTABLES = {
    "gmx": ["gmx", "gmx_mpi"],
    "packmol": ["packmol"],
    "multiwfn": ["Multiwfn"],
}

# Files a template dir must contain to count as a usable MD template set.
MD_REQUIRED_TEMPLATES = ["em.mdp", "md_NVT.mdp", "md_NPT.mdp", "prod.mdp", "topol.top"]


def ensure_dirs() -> None:
    for d in (STORE, MODELS, RUNS, MD_JOBS):
        d.mkdir(parents=True, exist_ok=True)
