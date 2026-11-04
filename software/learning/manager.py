"""
Learning manager: one object per run that ties the ledger, knowledge write-back and the
continual learners to the engine.

Per feedback event (immediately, crash-safe):
  1. append the event to store/feedback_ledger.jsonl
  2. update the tier 1 posterior (counts; no training)
  3. if LIVE_WRITE_BACK, upsert the transition's precedent into the running engine's precedent
     list, so the generator can already see it at the next step
At selection time:
  the frozen prior's logits get the active learner's bounded shift (ONLINE_PRIOR_SHIFT="auto")
At the end of a run (or `python learning_cli.py update`):
  rebuild the precedent store and merged corpus from the full ledger, then try to train a tier 2
  adapter; it is saved as a new version and promoted only if it passes the held-out test.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import adapter as tier2
from . import knowledge, ledger
from .posterior import ActionPosterior

LEARNER_POSTERIOR, LEARNER_ADAPTER = "posterior", "adapter"


# ----------------------------------------------------------------- registry ---
def load_registry(cfg) -> Dict[str, Any]:
    if cfg.REGISTRY.exists():
        try:
            return json.loads(cfg.REGISTRY.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"active": {"learner": LEARNER_POSTERIOR, "adapter_version": None}, "versions": [], "updates": []}


def save_registry(cfg, reg: Dict[str, Any]) -> None:
    cfg.REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    cfg.REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def adapter_path(cfg, version: int) -> Path:
    return cfg.MODELS / f"adapter_v{version}.pt"


# --------------------------------------------------------- store rebuilding ---
def rebuild_knowledge(cfg, events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, int]:
    events = ledger.read_events(cfg.FEEDBACK_LEDGER) if events is None else events
    papers = knowledge.build_papers(events, include_nonpositive=cfg.WRITE_BACK_NONPOSITIVE,
                                   expose_evidence=cfg.EVIDENCE_TO_RAG)
    n_papers, n_merged = knowledge.write_store(papers, cfg.DISCOVERED_PAPERS, cfg.BASE_CORPUS, cfg.MERGED_CORPUS)
    pol = {"positive": 0, "negative": 0, "unchange": 0}
    for p in papers:
        pol[p["paper_metadata"]["verdict"]] += 1
    return {"events": len(events), "precedents": n_papers, "merged_corpus": n_merged, **pol}


def update_learners(cfg, events: Optional[List[Dict[str, Any]]] = None, force_version: bool = False) -> Dict[str, Any]:
    """Refresh tier 1 state, try tier 2, record the update in the registry."""
    events = ledger.read_events(cfg.FEEDBACK_LEDGER) if events is None else events
    post = ActionPosterior(cfg.POSTERIOR_STRENGTH, cfg.MAX_LOGIT_SHIFT, cfg.UNCHANGE_WEIGHT).fit(events)
    cfg.POSTERIOR_STATE.write_text(json.dumps(post.dump(), ensure_ascii=False), encoding="utf-8")

    reg = load_registry(cfg)
    report: Dict[str, Any] = {"status": "locked"}
    if events:
        enc = tier2.PriorEncoder(cfg.PRIOR_CKPT)
        report = tier2.train_candidate(events, enc, cfg)
        if report.get("model") is not None:
            version = 1 + max([v["version"] for v in reg["versions"]] or [0])
            tier2.save(report["model"], report["objectives"], adapter_path(cfg, version), report)
            reg["versions"].append({"version": version, "time": datetime.now().isoformat(timespec="seconds"),
                                    "n_events": report["n_events"], "holdout": report.get("holdout")})
            reg["active"] = {"learner": LEARNER_ADAPTER, "adapter_version": version}
            report["version"] = version
        elif report.get("status") == "rejected":
            # The adapter family does not beat tier 1 on the newest feedback: fall back to tier 1
            # rather than keep an older adapter fitted on less data.
            reg["active"] = {"learner": LEARNER_POSTERIOR, "adapter_version": None}
    reg["updates"].append({
        "time": datetime.now().isoformat(timespec="seconds"), "n_events": len(events),
        "n_positive": sum(e["reward"] == "positive" for e in events),
        "adapter": {k: v for k, v in report.items() if k not in ("model", "objectives")},
        "active": dict(reg["active"]),
    })
    save_registry(cfg, reg)
    report.pop("model", None)
    return {"posterior_events": post.n_events, "adapter": report, "active": reg["active"]}


# ------------------------------------------------------------------ manager ---
class LearningManager:
    def __init__(self, cfg, run_name: str, objective: str, source: str):
        self.cfg, self.run, self.objective, self.source = cfg, run_name, objective, source
        self.events = ledger.read_events(cfg.FEEDBACK_LEDGER)
        self.posterior = ActionPosterior(cfg.POSTERIOR_STRENGTH, cfg.MAX_LOGIT_SHIFT,
                                         cfg.UNCHANGE_WEIGHT).fit(self.events)
        self.live = knowledge.LivePrecedents()
        self.reg = load_registry(cfg)
        self.adapter = None
        act = self.reg.get("active") or {}
        if (act.get("learner") == LEARNER_ADAPTER and act.get("adapter_version")
                and adapter_path(cfg, act["adapter_version"]).exists()):
            try:
                self.adapter = tier2.LoadedAdapter(adapter_path(cfg, act["adapter_version"]),
                                                   tier2.PriorEncoder(cfg.PRIOR_CKPT))
            except Exception as e:
                print(f"[learn] could not load adapter v{act['adapter_version']} ({e}); using tier 1")

    # ------------------------------------------------------------- reporting ---
    def describe(self) -> str:
        n = len(self.events)
        npos = sum(e["reward"] == "positive" for e in self.events)
        mode = self.cfg.ONLINE_PRIOR_SHIFT
        learner = f"tier 2 adapter v{self.reg['active']['adapter_version']}" if self.adapter else "tier 1 posterior"
        return (f"feedback memory: {n} events ({npos} positive) | prior shift: "
                + ("off (frozen literature prior)" if mode == "off" else f"{learner}"))

    # ------------------------------------------------------------ engine hooks ---
    def wrap_build_precedents(self, orig: Callable) -> Callable:
        def wrapped(papers):
            precs = orig(papers)
            self.live.bind(precs)
            return precs
        return wrapped

    def current_shifts(self, comp: Dict[str, Any]) -> Dict[str, float]:
        if self.cfg.ONLINE_PRIOR_SHIFT == "off":
            return {}
        if self.adapter is not None:
            return self.adapter.shifts(self.objective, comp)
        return self.posterior.shifts(self.objective, comp)

    def wrap_infer(self, orig: Callable, enabled_fn: Callable) -> Callable:
        def wrapped(state_payload, current_formulation, prior_ckpt_path):
            _, rows = orig(state_payload=state_payload, current_formulation=current_formulation,
                           prior_ckpt_path=prior_ckpt_path)
            shifts = self.current_shifts((current_formulation or {}).get("composition") or {})
            if not shifts:
                return _, rows
            for r in rows:
                d = float(shifts.get(r["action"], 0.0))
                r["prior_logit"], r["feedback_shift"] = r["logit"], d
                r["logit"] = r["prior_logit"] + d
                r["prob"] = 1.0 / (1.0 + math.exp(-r["logit"]))
            rows.sort(key=lambda r: r["prob"], reverse=True)
            moved = [f"{r['action']} {r['feedback_shift']:+.2f}" for r in rows if abs(r["feedback_shift"]) >= 0.05]
            if moved:
                print("[learn] prior shifted by past feedback: " + ", ".join(moved))
            enabled = set(enabled_fn(state_payload))
            first = next((r["action"] for r in rows if r["action"] in enabled), rows[0]["action"])
            return first, rows
        return wrapped

    # -------------------------------------------------------------- feedback ---
    def record(self, *, step: int, action: str, selection_mode: str, before: Dict[str, Any],
               proposed: Optional[Dict[str, Any]], change: str, rationale: str, reward: str,
               applied: bool, evidence: Dict[str, Any], prior: Dict[str, Any]) -> None:
        ev = ledger.make_event(run=self.run, step=step, objective=self.objective, action=action,
                               selection_mode=selection_mode, before=before, proposed=proposed,
                               change=change, rationale=rationale, reward=reward, applied=applied,
                               evidence=evidence, source=self.source, prior=prior)
        if not ledger.append_events(self.cfg.FEEDBACK_LEDGER, [ev]):
            return
        self.events.append(ev)
        self.posterior.observe(ev)
        note = f"[learn] stored feedback #{len(self.events)} ({action}, {reward})"
        if (self.cfg.LIVE_WRITE_BACK and proposed and
                (reward == "positive" or self.cfg.WRITE_BACK_NONPOSITIVE)):
            tid = knowledge.transition_id(self.objective, before, proposed)
            same = [e for e in self.events if e.get("proposed") and e.get("before")
                    and knowledge.transition_id(e.get("objective", ""), e["before"], e["proposed"]) == tid]
            if self.live.upsert(knowledge.transition_paper(tid, same, self.cfg.EVIDENCE_TO_RAG)):
                note += " -> precedent available to the RAG from the next step"
        print(note)

    def finalize(self) -> Dict[str, Any]:
        k = rebuild_knowledge(self.cfg)
        u = update_learners(self.cfg)
        return {"knowledge": k, **u, "run_events": sum(e.get("run") == self.run for e in self.events)}
