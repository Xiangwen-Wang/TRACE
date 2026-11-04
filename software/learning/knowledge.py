"""
Knowledge write-back: feedback events -> literature-schema precedents.

Each distinct transition (objective, before-state, proposed-state) becomes one "paper" with a
baseline and one variant, so it flows through the same build_precedents() as the literature and
lands in the generator's evidence:
  accepted moves  -> positive_patterns   (and, via the after-state match, forbidden reversions)
  rejected moves  -> negative_patterns   (the generator sees what already failed from this state)
  no-change       -> mixed/uncertain     (kept as context, not as a direction)
Repeated feedback on the same transition is aggregated (counts are written into the record), and
any evidence entered with the feedback (MD descriptors, notes) is attached as conditions.

The precedent store is always rebuilt from the ledger, so it can never drift from it.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from action_family_rag import _normalized_composition_key
from literature_rag import action_to_dimensions, build_precedents, load_papers

SOURCE_TAG = "trace_feedback"


def transition_id(objective: str, before: Dict[str, Any], proposed: Dict[str, Any]) -> str:
    key = json.dumps([objective, _normalized_composition_key(before), _normalized_composition_key(proposed)],
                     ensure_ascii=False)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _group(events: List[Dict[str, Any]], include_nonpositive: bool) -> "OrderedDict[str, List[Dict[str, Any]]]":
    groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for ev in events:
        if not ev.get("before") or not ev.get("proposed"):
            continue  # a proposal that could not be applied has no concrete transition
        if ev["reward"] != "positive" and not include_nonpositive:
            continue
        tid = transition_id(ev.get("objective", ""), ev["before"], ev["proposed"])
        groups.setdefault(tid, []).append(ev)
    return groups


def _verdict(evs: List[Dict[str, Any]]) -> str:
    pos = sum(e["reward"] == "positive" for e in evs)
    neg = sum(e["reward"] == "negative" for e in evs)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    if pos == neg == 0:
        return "unchange"
    return evs[-1]["reward"]  # tie: the most recent judgement wins


def _evidence_conditions(evs: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for e in evs:
        ev = e.get("evidence") or {}
        desc = ev.get("descriptors") or {}
        if desc:
            out.append(f"{e['event_id']} ({e['reward']}): " +
                       ", ".join(f"{k}={v}" for k, v in list(desc.items())[:12]))
        if ev.get("note"):
            out.append(f"{e['event_id']} note: {str(ev['note'])[:200]}")
    return out


def transition_paper(tid: str, evs: List[Dict[str, Any]], expose_evidence: bool = True) -> Dict[str, Any]:
    first, last = evs[0], evs[-1]
    objective = first.get("objective") or "unspecified_performance_objective"
    action = last.get("action") or ""
    verdict = _verdict(evs)
    counts = {r: sum(e["reward"] == r for e in evs) for r in ("positive", "negative", "unchange")}
    source = "MD" if any(e.get("source") == "md" for e in evs) else "expert"
    # Outcome wording is chosen so literature_rag.classify_polarity reads the right polarity
    # (descriptor values go into conditions, never into the outcome, so they cannot flip it).
    outcome = {
        "positive": f"Improved {objective}; accepted under in-house {source} feedback.",
        "negative": f"Worse {objective}; rejected under in-house {source} feedback.",
        "unchange": f"No measurable change in {objective} under in-house {source} feedback.",
    }[verdict]
    conditions = [
        f"in-house {source} feedback on an agent proposal ({action})",
        f"feedback count: {counts['positive']} positive, {counts['negative']} negative, "
        f"{counts['unchange']} no-change",
    ] + (_evidence_conditions(evs) if expose_evidence else [])
    change = next((e.get("change") for e in reversed(evs) if e.get("change")), "")
    return {
        "paper_metadata": {
            "title": f"In-house feedback: {change or action} [{verdict}]",
            "doi": f"feedback:{tid}",
            "source": SOURCE_TAG,
            "verdict": verdict,
            "events": [e["event_id"] for e in evs],
        },
        "formulations": [
            {"id": "baseline", "role": "baseline", "composition": first["before"]},
            {"id": "proposed", "role": "optimized" if verdict == "positive" else "variant",
             "composition": first["proposed"],
             "evaluated_properties": [{"property": objective, "conditions": conditions, "outcome": outcome}]},
        ],
        "exploration_operations": {"explored_dimensions": action_to_dimensions(action) if action else []},
        "design_conclusions": [f"{action}: {outcome}"],
    }


def build_papers(events: List[Dict[str, Any]], include_nonpositive: bool = True,
                 expose_evidence: bool = True) -> List[Dict[str, Any]]:
    return [transition_paper(tid, evs, expose_evidence) for tid, evs in _group(events, include_nonpositive).items()]


def paper_precedents(paper: Dict[str, Any]) -> List[Any]:
    """Precedent objects for one write-back paper, with a stable, collision-free id."""
    precs = build_precedents([paper])
    tid = str(paper["paper_metadata"]["doi"]).split(":", 1)[-1]
    for i, p in enumerate(precs):
        p.precedent_id = f"fb:{tid}" + (f":{i}" if i else "")
    return precs


def write_store(papers: List[Dict[str, Any]], discovered_path: Path, base_corpus: Path,
                merged_path: Path) -> Tuple[int, int]:
    """discovered = write-back papers; merged = base corpus + discovered. Returns (n_papers, n_merged)."""
    discovered_path.parent.mkdir(parents=True, exist_ok=True)
    discovered_path.write_text(json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8")
    base = load_papers(base_corpus)
    merged_path.write_text(json.dumps(base + papers, ensure_ascii=False), encoding="utf-8")
    return len(papers), len(base) + len(papers)


class LivePrecedents:
    """Keeps the engine's in-memory precedent list in sync with the ledger during a session.

    The engine builds all_precedents once and passes the same list object to every retrieval
    call, so updating it in place makes a new feedback record retrievable at the next step.
    """

    def __init__(self) -> None:
        self.ref: List[Any] | None = None

    def bind(self, precedents: List[Any]) -> None:
        # Write-back records loaded from merged_corpus.json get positional ids; give them their
        # stable fb:<transition> id so a later upsert replaces them instead of duplicating.
        for p in precedents:
            doi = str(getattr(p, "doi", ""))
            if doi.startswith("feedback:"):
                p.precedent_id = "fb:" + doi.split(":", 1)[1]
        self.ref = precedents

    def upsert(self, paper: Dict[str, Any]) -> int:
        if self.ref is None:
            return 0
        tid = str(paper["paper_metadata"]["doi"]).split(":", 1)[-1]
        prefix = f"fb:{tid}"
        self.ref[:] = [p for p in self.ref if not str(getattr(p, "precedent_id", "")).startswith(prefix)]
        new = paper_precedents(paper)
        self.ref.extend(new)
        return len(new)
