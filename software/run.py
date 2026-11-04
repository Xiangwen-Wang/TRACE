"""
Electrolyte formulation optimizer — packaged entry point.

Flow (terminal):
  0. Ask for an OpenAI API key (hidden input; optionally remembered).
  1. Ask for a local initial_formulation file; read it.
  2. Run the engine: MLP ranks actions -> top-3 directions -> RAG-LLM proposal per step.
  3. Detect a local MD toolchain. If present, ask for a template folder and auto-generate a
     GROMACS job directory for each proposed formulation (generate-only, not submitted);
     you run MD on your server, read the descriptors, then enter positive/negative/nochange.
     If MD is not present, you just enter positive/negative/nochange directly.
  4. Every feedback (positive / negative / no-change, with optional MD descriptors or a note) is
     stored at once in store/feedback_ledger.jsonl, turned into a RAG precedent that the generator
     can already retrieve at the next step, and counted by the tier 1 posterior that shifts the
     MLP prior. On stop, the precedent store is rebuilt from the full ledger and the tier 2
     adapter is retrained (gated; promoted only if it beats tier 1 on held-out feedback).

Launch with the double-click start.command, or:
    ../.venv/bin/python run.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
from getpass import getpass
from pathlib import Path

import config
import md_scaffold
from learning import ledger as fb_ledger
from learning.manager import LearningManager

# import the engine (monolithic agent) and patch its I/O hooks
import interactive_feedback_agent_terminal_top3 as agent

# ----------------------------------------------------------------- helpers ---
DEFAULT_OBJECTIVE = {
    "primary": "high_voltage_oxidative_stability",
    "tags": ["high_voltage_oxidative_stability", "high_voltage_cycling_stability"],
    "constraints": {"voltage_window": "2.7-4.5 V"},
}


def ensure_openai_key() -> None:
    """Get the OpenAI key like an app: env -> remembered file -> prompt (hidden)."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key and config.KEYFILE.exists():
        key = config.KEYFILE.read_text(encoding="utf-8").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key
            print("Loaded OpenAI API key from store/.openai_key.")
    if not key:
        print("\nThe RAG-LLM proposal step needs an OpenAI API key.")
        key = getpass("Enter your OpenAI API key (sk-..., input hidden): ").strip()
        if not key:
            print("[warn] No key entered; the proposal step will fail (other features still work).")
            return
        os.environ["OPENAI_API_KEY"] = key
        if input("Remember this key on disk for next time? [y/N]: ").strip().lower() in ("y", "yes"):
            config.ensure_dirs()
            config.KEYFILE.write_text(key, encoding="utf-8")
            try:
                os.chmod(config.KEYFILE, 0o600)
            except OSError:
                pass
            print(f"Saved to {config.KEYFILE} (mode 600; delete the file to forget it).")
    # soft connectivity check — never fatal
    try:
        from openai import OpenAI
        OpenAI().models.list()
        print("OpenAI key verified.")
    except Exception as e:
        print(f"[warn] Key check failed or offline ({str(e)[:80]}); continuing, proposals may fail.")


def ask_path(prompt: str, must_exist: bool = True, default: str = "") -> Path:
    while True:
        raw = input(prompt).strip().strip('"').strip("'")
        if not raw and default:
            raw = default
        p = Path(raw).expanduser()
        if not must_exist or p.exists():
            return p
        print(f"  Path not found: {p}  — please try again")


def load_initial_state(src: Path) -> dict:
    """Accept {state0:{composition}}, {composition}, or a bare composition; ensure objective."""
    data = json.load(open(src, encoding="utf-8"))
    if "state0" in data:
        state = {"state0": data["state0"]}
    elif "composition" in data:
        state = {"state0": {"composition": data["composition"]}}
    elif "salt" in data or "solvents" in data:
        state = {"state0": {"composition": data}}
    else:
        raise ValueError("Unrecognized formulation file: expected one of state0/composition/salt")
    state["objective"] = data.get("objective", DEFAULT_OBJECTIVE)
    return state


# --------------------------------------------------------- engine I/O hooks ---
_LAST = {"current": None, "proposal": None, "step": 0, "action": "", "ranking": [], "bundle": {}}
_MD = {"enabled": False, "templates": None, "jobroot": None, "objective": ""}
_LEARN: dict = {"mgr": None}

_orig_print_candidate = agent.print_candidate
_orig_ask_reward = agent.ask_user_reward
_orig_build_precedents = agent.build_precedents
_orig_infer = agent.infer_action_with_mlp


def _patched_print_candidate(*a, **k):
    _LAST["current"] = k.get("current_formulation")
    _LAST["proposal"] = k.get("proposal_obj")
    _LAST["step"] = k.get("step_idx", _LAST["step"])
    _LAST["action"] = k.get("selected_action") or ""
    _LAST["ranking"] = k.get("ranking") or []
    _LAST["bundle"] = k.get("decision_bundle") or {}
    return _orig_print_candidate(*a, **k)


def _prior_of(action: str) -> dict:
    """Rank/prob of the action under the FROZEN prior (before any feedback shift)."""
    rows = sorted(_LAST["ranking"], key=lambda r: -float(r.get("prior_logit", r.get("logit", 0.0))))
    for i, r in enumerate(rows):
        if r.get("action") == action:
            lg = float(r.get("prior_logit", r.get("logit", 0.0)))
            return {"rank": i + 1, "prob": 1.0 / (1.0 + 2.718281828459045 ** (-lg))}
    return {}


def _patched_ask_reward() -> str:
    proposed, applied_ok, jobdir = None, False, None
    current = copy.deepcopy(_LAST["current"]) if _LAST["current"] else None
    if current and _LAST["proposal"]:
        try:
            next_form, applied_ok, _ = agent.apply_proposal_to_formulation(copy.deepcopy(current), _LAST["proposal"])
            proposed = (next_form or {}).get("composition") if applied_ok else None
        except Exception:
            proposed, applied_ok = None, False
    if _MD["enabled"] and _LAST["proposal"]:
        try:
            if applied_ok:
                jobdir = md_scaffold.generate_job(
                    next_form, _MD["templates"], _MD["jobroot"],
                    step=int(_LAST["step"]), objective=_MD["objective"])
                print(f"\n[MD] Generated a job directory for this proposal (generate-only, not submitted):\n     {jobdir}")
                print("[MD] Add the files listed in MANIFEST.md -> run `bash run.sh` on a server -> "
                      "compute the descriptors, then enter feedback below.")
            else:
                print("\n[MD] Proposal could not be applied to a concrete formulation; "
                      "skipping MD generation — enter feedback from judgment.")
        except Exception as e:  # never let MD scaffolding break the loop
            print(f"\n[MD] Failed to generate the job directory ({e}); enter feedback manually.")

    reward = _orig_ask_reward()
    mgr = _LEARN["mgr"]
    if reward == "quit" or mgr is None or not current:
        return reward
    evidence = {}
    if config.ASK_EVIDENCE:
        raw = input("evidence (optional) — MD descriptors as name=value; name=value, a .json file, "
                    "or a short note [Enter to skip]: ")
        evidence = fb_ledger.parse_evidence(raw)
    if jobdir:
        evidence["md_job"] = str(jobdir)
    try:
        proposal = _LAST["proposal"] or {}
        mgr.record(
            step=int(_LAST["step"]), action=_LAST["action"],
            selection_mode=str(_LAST["bundle"].get("selection_path") or ""),
            before=current.get("composition") or {}, proposed=proposed,
            change=agent.summarize_proposal_change(current, proposal) if proposal else "",
            rationale=str(proposal.get("rationale", ""))[:400],
            reward=reward, applied=(reward == "positive" and applied_ok),
            evidence=evidence, prior=_prior_of(_LAST["action"]))
    except Exception as e:  # storing feedback must never break the loop
        print(f"[learn] could not store this feedback ({e})")
    return reward


# ----------------------------------------------------------------- main ------
def main() -> None:
    config.ensure_dirs()
    os.chdir(config.PKG)  # module-dir relative data files resolve inside the package
    print("=" * 72)
    print("            Electrolyte Formulation Optimizer")
    print("=" * 72)

    # 0) OpenAI key (like an app: prompt up-front, optional remember)
    ensure_openai_key()

    # 1) initial formulation
    src = ask_path("\nEnter the path to your local initial_formulation file: ", must_exist=True)
    try:
        state = load_initial_state(src)
    except Exception as e:
        print(f"Failed to read formulation: {e}")
        return
    objective = str((state.get("objective") or {}).get("primary") or "high_voltage_oxidative_stability")

    run_name = f"run_{src.stem}_{len(list(config.RUNS.glob('run_*'))) + 1}"
    run_folder = config.RUNS / run_name
    run_folder.mkdir(parents=True, exist_ok=True)
    (run_folder / "initial_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Formulation loaded. Objective = {objective}")
    print(f"Run directory: {run_folder}")

    # 2) MD detection
    print("\nChecking for a local MD toolchain ...")
    templates_dir = config.MD_TEMPLATES
    available, info = md_scaffold.md_available(templates_dir)
    tools = info["tools"]
    print(f"  gmx: {tools.get('gmx') or 'not found'} | packmol: {tools.get('packmol') or 'not found'} | "
          f"Multiwfn: {tools.get('multiwfn') or 'not found'}")
    if not info["templates_ok"]:
        print(f"  Missing MD templates: {info['missing_templates']} (default template dir {templates_dir})")

    if available:
        tdir = ask_path(f"\nLocal MD detected. Enter the template folder holding the required MD files "
                        f"[Enter for default {templates_dir}]: ",
                        must_exist=True, default=str(templates_dir))
        ok, missing = md_scaffold.templates_ok(tdir)
        if not ok:
            print(f"  That folder is missing templates {missing}; switching to manual-feedback mode.")
        else:
            _MD.update(enabled=True, templates=tdir,
                       jobroot=config.MD_JOBS / run_name, objective=objective)
            _MD["jobroot"].mkdir(parents=True, exist_ok=True)
            print(f"  MD mode enabled. Jobs will be generated under: {_MD['jobroot']}")
    if not _MD["enabled"]:
        print("\nAutomatic MD not enabled: you will enter positive / negative / nochange each step.")

    # 3) feedback memory: rebuild the RAG corpus from the ledger (base + all past feedback),
    #    and start the learning manager for this run
    from learning.manager import rebuild_knowledge
    k = rebuild_knowledge(config)
    merged = config.MERGED_CORPUS
    mgr = LearningManager(config, run_name, objective, source="md" if _MD["enabled"] else "manual")
    _LEARN["mgr"] = mgr
    print("\n[learn] " + mgr.describe())
    if k["precedents"]:
        print(f"[learn] RAG corpus includes {k['precedents']} precedent(s) from past feedback "
              f"({k['positive']} accepted, {k['negative']} rejected, {k['unchange']} no-change).")

    # 4) patch engine hooks and run its loop
    agent.print_candidate = _patched_print_candidate
    agent.ask_user_reward = _patched_ask_reward
    agent.build_precedents = mgr.wrap_build_precedents(_orig_build_precedents)
    agent.infer_action_with_mlp = mgr.wrap_infer(_orig_infer, agent.enabled_actions_from_state)
    sys.argv = [
        "engine",
        "--folder", str(run_folder),
        "--prior", str(config.PRIOR_CKPT),
        "--data", str(merged),
        "--max-steps", str(config.DEFAULT_MAX_STEPS),
        "--llm-mode", config.DEFAULT_LLM_MODEL,
    ]
    print("\n" + "=" * 72)
    print(" Optimization loop (type 'quit' to stop)")
    print("=" * 72)
    try:
        agent.main()
    finally:
        agent.print_candidate = _orig_print_candidate
        agent.ask_user_reward = _orig_ask_reward
        agent.build_precedents = _orig_build_precedents
        agent.infer_action_with_mlp = _orig_infer

    # 5) continual-learning update + screening
    print("\n" + "=" * 72)
    print(" SCREENING — folding this session's feedback into the system")
    print("=" * 72)
    summary = mgr.finalize()
    k, ad, act = summary["knowledge"], summary["adapter"], summary["active"]
    print(f"[memory] {summary['run_events']} feedback event(s) this run; ledger total {k['events']}.")
    print(f"[memory] RAG precedents from feedback: {k['precedents']} "
          f"({k['positive']} accepted, {k['negative']} rejected, {k['unchange']} no-change) "
          f"-> merged corpus {k['merged_corpus']} papers")
    if ad.get("status") == "locked":
        print(f"[learn] tier 1 posterior updated ({summary['posterior_events']} events); "
              f"tier 2 adapter locked: {ad.get('reason', 'no feedback yet')}")
    else:
        h = ad.get("holdout") or {}
        print(f"[learn] tier 2 adapter candidate: {ad['status']} (held-out log-loss adapter "
              f"{h.get('logloss_adapter', float('nan')):.3f} vs tier 1 {h.get('logloss_tier1', float('nan')):.3f} "
              f"vs base rate {h.get('logloss_base_rate', float('nan')):.3f})")
    learner = (f"tier 2 adapter v{act['adapter_version']}" if act.get("learner") == "adapter"
               else "tier 1 posterior")
    print(f"[learn] active prior shift for the next run: {learner}")
    print("\nDone. Stored under store/ (feedback_ledger.jsonl, discovered_papers.json, merged_corpus.json, "
          "action_posterior.json, learning_registry.json). The next run uses them automatically; "
          "`python learning_cli.py status` shows what has been learned.")

if __name__ == "__main__":
    main()
