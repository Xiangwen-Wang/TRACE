"""
Feedback ledger: one JSON line per feedback event, append-only.

Every other learned artifact (precedents, posterior counts, adapter) is DERIVED from this file
and can be rebuilt from it at any time, so nothing is lost if a learner changes or is reset.

Event schema
------------
event_id        "<run>#<step>"  (unique; re-appending the same id is a no-op)
time            ISO timestamp
run, step       where it came from
objective       primary objective of the run
action          the canonical action that was proposed (one of the 8)
selection_mode  how the controller reached it (top3 screening, retry, escalation, ...)
before          composition the proposal was made from
proposed        composition the proposal would produce (None if it could not be applied)
change          one-line description of the change
rationale       the generator's rationale (shortened)
reward          positive | negative | unchange
applied         whether the agent moved to `proposed`
evidence        {"descriptors": {name: value}, "note": str, "md_job": path} (all optional)
source          "md" | "manual" | "import"
prior           {"rank": int, "prob": float} of `action` under the frozen prior at that step
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REWARDS = ("positive", "negative", "unchange")


def read_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn last line from a crash; skip it
        if ev.get("reward") in REWARDS and ev.get("action"):
            out.append(ev)
    return out


def append_events(path: Path, events: Iterable[Dict[str, Any]]) -> int:
    """Append events whose event_id is not in the ledger yet. Returns how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = {e.get("event_id") for e in read_events(path)}
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for ev in events:
            if ev.get("event_id") in seen:
                continue
            ev.setdefault("time", datetime.now().isoformat(timespec="seconds"))
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            seen.add(ev.get("event_id"))
            n += 1
    return n


def make_event(
    *, run: str, step: int, objective: str, action: str, selection_mode: str,
    before: Dict[str, Any], proposed: Optional[Dict[str, Any]], change: str, rationale: str,
    reward: str, applied: bool, evidence: Optional[Dict[str, Any]], source: str,
    prior: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "event_id": f"{run}#{step}",
        "run": run, "step": int(step), "objective": objective,
        "action": action, "selection_mode": selection_mode,
        "before": before, "proposed": proposed, "change": change, "rationale": rationale,
        "reward": reward, "applied": bool(applied),
        "evidence": evidence or {}, "source": source, "prior": prior or {},
    }


def parse_evidence(raw: str) -> Dict[str, Any]:
    """Free-form evidence entry -> {"descriptors": {...}, "note": str}.

    Accepts a path to a JSON file (dict of descriptors), `key=value; key=value` pairs
    (numeric values parsed), or plain text (kept as a note).
    """
    raw = (raw or "").strip()
    if not raw:
        return {}
    p = Path(raw.strip('"').strip("'")).expanduser()
    if p.suffix.lower() == ".json" and p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {"descriptors": data, "source_file": str(p)}
        except Exception:
            pass
    descriptors: Dict[str, Any] = {}
    notes: List[str] = []
    for part in raw.replace("\n", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                descriptors[k] = float(v)
            except ValueError:
                descriptors[k] = v
        else:
            notes.append(part)
    out: Dict[str, Any] = {}
    if descriptors:
        out["descriptors"] = descriptors
    if notes:
        out["note"] = "; ".join(notes)
    return out


def from_attempts(attempts: List[Dict[str, Any]], run: str, objective: str, source: str = "import",
                  apply_fn=None) -> List[Dict[str, Any]]:
    """Convert an engine attempts file (interactive_feedback_attempts.json) into ledger events.

    For rejected steps the engine does not store the proposed composition; `apply_fn`
    (the engine's apply_proposal_to_formulation) recomputes it when given.
    """
    def comp(state):
        return ((state or {}).get("state0") or {}).get("composition") or (state or {}).get("composition") or {}

    events = []
    for a in attempts:
        reward, action = a.get("user_reward"), a.get("selected_action_strategy")
        before = comp(a.get("current_state_before"))
        if reward not in REWARDS or not action or not before:
            continue
        proposed = comp(a.get("state_after")) if (reward == "positive" and a.get("applied")) else None
        if proposed is None and apply_fn and isinstance(a.get("proposal"), dict):
            try:
                nf, ok, _ = apply_fn({"composition": before}, a["proposal"])
                proposed = (nf or {}).get("composition") if ok else None
            except Exception:
                proposed = None
        ranking = a.get("action_ranking") or []
        rank = next((i + 1 for i, r in enumerate(ranking) if r.get("action") == action), None)
        prob = next((float(r.get("prob", 0)) for r in ranking if r.get("action") == action), None)
        events.append(make_event(
            run=run, step=int(a.get("step", 0)), objective=objective, action=action,
            selection_mode=str(a.get("selection_mode") or ""), before=before, proposed=proposed,
            change="", rationale=str(((a.get("proposal") or {}) if isinstance(a.get("proposal"), dict) else {})
                                     .get("rationale", ""))[:400],
            reward=reward, applied=bool(a.get("applied")), evidence={}, source=source,
            prior={"rank": rank, "prob": prob} if rank else {},
        ))
    return events
