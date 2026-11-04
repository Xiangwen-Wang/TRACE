"""
Tier 1 — hierarchical Beta posterior over feedback outcomes (no training, works from event 1).

For the current objective o, state s and action a, the success probability is estimated with
empirical-Bayes shrinkage through three levels, each level's estimate being the prior of the next:

    p_o      = (S_o + 1) / (N_o + 2)                              base success rate for o
    p_oa     = (S_oa + m * p_o)  / (N_oa + m)                     action a under o
    p_osa    = (S_osa + m * p_oa) / (N_osa + m)                   action a from this state

The logit shift added to the frozen literature prior is

    shift_a = clip( logit(p_osa) - logit(p_o), -MAX, +MAX )

so with no feedback every shift is exactly 0 (the agent is the literature prior), an action that
keeps failing is pushed down and one that keeps succeeding is pulled up, and no amount of
feedback can move a logit by more than MAX. "State" means component identity (salt, solvents,
additives, ignoring amounts), so evidence transfers across ratio variants of one formulation.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from action_family_rag import _canonical_state

ACTIONS = [
    "salt_replace", "salt_concentration_shift", "add_new_salt", "solvent_replace",
    "solvent_concentration_shift", "add_new_solvent", "remove_solvent", "additive_change",
]


def state_context(comp: Dict[str, Any]) -> str:
    c = _canonical_state(comp or {})
    return json.dumps([c["salts"], sorted(c["solvents"]), sorted(c["additives"])], ensure_ascii=False)


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


class ActionPosterior:
    def __init__(self, strength: float = 4.0, max_shift: float = 1.5, unchange_weight: float = 0.5):
        self.m, self.max_shift, self.uw = float(strength), float(max_shift), float(unchange_weight)
        self.S: Dict[Tuple, float] = defaultdict(float)
        self.N: Dict[Tuple, float] = defaultdict(float)
        self.n_events = 0

    def fit(self, events: List[Dict[str, Any]]) -> "ActionPosterior":
        self.S.clear(); self.N.clear(); self.n_events = 0
        for ev in events:
            self.observe(ev)
        return self

    def observe(self, ev: Dict[str, Any]) -> None:
        a, r, o = ev.get("action"), ev.get("reward"), ev.get("objective", "")
        if a not in ACTIONS:
            return
        w = self.uw if r == "unchange" else 1.0
        s = 1.0 if r == "positive" else 0.0
        ctx = state_context(ev.get("before") or {})
        for key in ((o,), (o, a), (o, ctx, a)):
            self.N[key] += w
            self.S[key] += s * w
        self.n_events += 1

    def _p(self, objective: str, ctx: str, action: str) -> Tuple[float, float, float]:
        p_o = (self.S[(objective,)] + 1.0) / (self.N[(objective,)] + 2.0)
        p_oa = (self.S[(objective, action)] + self.m * p_o) / (self.N[(objective, action)] + self.m)
        p_osa = (self.S[(objective, ctx, action)] + self.m * p_oa) / (self.N[(objective, ctx, action)] + self.m)
        return p_o, p_oa, p_osa

    def predict(self, objective: str, comp: Dict[str, Any], action: str) -> float:
        return self._p(objective, state_context(comp), action)[2]

    def shifts(self, objective: str, comp: Dict[str, Any]) -> Dict[str, float]:
        ctx = state_context(comp)
        out = {}
        for a in ACTIONS:
            p_o, _, p_osa = self._p(objective, ctx, a)
            out[a] = max(-self.max_shift, min(self.max_shift, _logit(p_osa) - _logit(p_o)))
        return out

    def summary(self, objective: str) -> Dict[str, Any]:
        rows = {}
        for a in ACTIONS:
            n, s = self.N[(objective, a)], self.S[(objective, a)]
            if n:
                rows[a] = {"n": round(n, 2), "success": round(s, 2),
                           "p": round(self._p(objective, "", a)[1], 3)}
        return {"objective": objective, "n_events": self.n_events,
                "base_rate": round((self.S[(objective,)] + 1) / (self.N[(objective,)] + 2), 3),
                "by_action": rows}

    def dump(self) -> Dict[str, Any]:
        return {"strength": self.m, "max_shift": self.max_shift, "unchange_weight": self.uw,
                "n_events": self.n_events,
                "counts": [{"key": list(k), "N": self.N[k], "S": self.S[k]} for k in self.N]}
