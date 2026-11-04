"""
Inspect and maintain the feedback memory / continual-learning store.

    python learning_cli.py status                 what has been learned so far
    python learning_cli.py update                 rebuild precedents from the ledger + retrain learners
    python learning_cli.py import PATH [--objective O] [--run NAME]
                                                  add feedback from an engine run folder or an
                                                  interactive_feedback_attempts.json (e.g. older sessions)
    python learning_cli.py activate posterior|N   choose the active learner (N = adapter version)
    python learning_cli.py shifts STATE.json      show the prior shift the active learner gives a formulation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import config

os.chdir(config.PKG)
sys.path.insert(0, str(config.PKG))

from learning import ledger  # noqa: E402
from learning.manager import (LEARNER_ADAPTER, LEARNER_POSTERIOR, LearningManager, adapter_path,  # noqa: E402
                              load_registry, rebuild_knowledge, save_registry, update_learners)
from learning.posterior import ActionPosterior  # noqa: E402


def cmd_status(_):
    events = ledger.read_events(config.FEEDBACK_LEDGER)
    reg = load_registry(config)
    print(f"ledger: {config.FEEDBACK_LEDGER}  ({len(events)} events)")
    if not events:
        print("no feedback stored yet.")
        return
    by_r = {r: sum(e["reward"] == r for e in events) for r in ledger.REWARDS}
    runs = sorted({e["run"] for e in events})
    print(f"  {by_r['positive']} positive / {by_r['negative']} negative / {by_r['unchange']} no-change "
          f"across {len(runs)} run(s); {sum(bool(e.get('evidence', {}).get('descriptors')) for e in events)} "
          f"with MD descriptors")
    disc = json.loads(config.DISCOVERED_PAPERS.read_text()) if config.DISCOVERED_PAPERS.exists() else []
    v = [p["paper_metadata"].get("verdict") for p in disc]
    print(f"RAG precedents from feedback: {len(disc)} ({v.count('positive')} accepted, "
          f"{v.count('negative')} rejected, {v.count('unchange')} no-change)")
    post = ActionPosterior(config.POSTERIOR_STRENGTH, config.MAX_LOGIT_SHIFT, config.UNCHANGE_WEIGHT).fit(events)
    for obj in sorted({e.get("objective", "") for e in events}):
        s = post.summary(obj)
        print(f"\ntier 1 posterior — objective {obj}: base success rate {s['base_rate']}")
        for a, r in sorted(s["by_action"].items(), key=lambda kv: -kv[1]["p"]):
            print(f"  {a:28s} n={r['n']:<5} success={r['success']:<5} p={r['p']}")
    act = reg.get("active", {})
    print(f"\nactive learner: {act.get('learner')}" + (f" v{act['adapter_version']}" if act.get("adapter_version") else ""))
    print(f"prior shift at selection: {config.ONLINE_PRIOR_SHIFT}")
    for ver in reg.get("versions", []):
        print(f"  adapter v{ver['version']}  {ver['time']}  n={ver['n_events']}  holdout={ver.get('holdout')}")
    if reg.get("updates"):
        last = reg["updates"][-1]["adapter"]
        print(f"last tier 2 check: {last.get('status')} {last.get('reason', '') or last.get('holdout', '')}")


def cmd_update(_):
    k = rebuild_knowledge(config)
    print(f"precedents rebuilt: {k}")
    u = update_learners(config)
    print(json.dumps(u, ensure_ascii=False, indent=2, default=str))


def cmd_import(a):
    src = Path(a.path).expanduser()
    folder = src if src.is_dir() else src.parent
    attempts_p = src / "interactive_feedback_attempts.json" if src.is_dir() else src
    attempts = json.loads(attempts_p.read_text(encoding="utf-8"))
    objective = a.objective
    sess = folder / "interactive_feedback_session.json"
    if not objective and sess.exists():
        objective = str((json.loads(sess.read_text()).get("objective") or {}).get("primary") or "")
    objective = objective or "high_voltage_oxidative_stability"
    import interactive_feedback_agent_terminal_top3 as agent
    run = a.run or f"import_{attempts_p.stem if src.is_file() else folder.name}"
    events = ledger.from_attempts(attempts, run=run, objective=objective, source="import",
                                  apply_fn=agent.apply_proposal_to_formulation)
    n = ledger.append_events(config.FEEDBACK_LEDGER, events)
    print(f"imported {n} new event(s) from {attempts_p} (objective {objective}, run {run}); "
          f"run `python learning_cli.py update` to rebuild precedents and retrain.")


def cmd_activate(a):
    reg = load_registry(config)
    if a.target == LEARNER_POSTERIOR:
        reg["active"] = {"learner": LEARNER_POSTERIOR, "adapter_version": None}
    else:
        v = int(a.target.lstrip("v"))
        if not adapter_path(config, v).exists():
            sys.exit(f"adapter v{v} not found in {config.MODELS}")
        reg["active"] = {"learner": LEARNER_ADAPTER, "adapter_version": v}
    save_registry(config, reg)
    print(f"active learner -> {reg['active']}")


def cmd_shifts(a):
    data = json.loads(Path(a.state).read_text(encoding="utf-8"))
    comp = (data.get("state0") or {}).get("composition") or data.get("composition") or data
    obj = a.objective or str((data.get("objective") or {}).get("primary") or "high_voltage_oxidative_stability")
    mgr = LearningManager(config, "inspect", obj, source="manual")
    print(mgr.describe())
    for act, d in sorted(mgr.current_shifts(comp).items(), key=lambda kv: -kv[1]):
        print(f"  {act:28s} {d:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("update").set_defaults(fn=cmd_update)
    p = sub.add_parser("import"); p.add_argument("path"); p.add_argument("--objective"); p.add_argument("--run")
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("activate"); p.add_argument("target"); p.set_defaults(fn=cmd_activate)
    p = sub.add_parser("shifts"); p.add_argument("state"); p.add_argument("--objective"); p.set_defaults(fn=cmd_shifts)
    a = ap.parse_args()
    config.ensure_dirs()
    a.fn(a)


if __name__ == "__main__":
    main()
