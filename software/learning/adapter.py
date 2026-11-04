"""
Tier 2 — residual adapter on the frozen prior (trained on feedback only, promoted only if it wins).

The literature prior is never modified. The adapter reads the prior's own 256-d penultimate
representation of the state plus its 8 logits, and outputs a bounded per-action logit shift:

    p(success | s, a) = sigmoid( c_objective + MAX * tanh(g(h(s), l(s))_a) )
    selection score   = l(s)_a + MAX * tanh(g(...)_a)

so it predicts the same quantity as tier 1, but can generalize across states through the prior's
structure representation instead of only matching identical component sets.

Continual-learning safeguards (feedback is sparse and arrives over time):
  * data gate      no training below ADAPTER_MIN_EVENTS / ADAPTER_MIN_POSITIVES
  * anchoring      zero-initialised output, weight decay and an L2 penalty on the shift itself
  * time-ordered   the newest ADAPTER_HOLDOUT_FRAC of events are held out; the adapter is
    promotion      promoted only if its held-out log-loss beats BOTH the base rate and tier 1
  * versioning     every trained candidate is saved (store/models/adapter_vN.pt) with its
                   metrics; the registry records which version is active, and can roll back
  * refit          a promoted adapter is refit on all events before use
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .posterior import ACTIONS, ActionPosterior

A_IDX = {a: i for i, a in enumerate(ACTIONS)}


# ------------------------------------------------------------ prior features ---
class PriorEncoder:
    """Loads the frozen prior once; returns (penultimate embedding, logits) for a composition."""

    def __init__(self, ckpt_path: Path):
        import prior_suggestions as ps

        self.ps = ps
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.obs_spec = ckpt["obs_spec"]
        self.mod = ps._load_structure_mlp_module(str(self.obs_spec.get("feature_version") or "structure_v1"))
        ps.merge_live_smiles_lookup(self.obs_spec, self.mod)
        self.actions = list(ckpt.get("actions") or self.mod.ACTIONS)
        build = getattr(self.mod, "build_model", None)
        self.model = build(self.obs_spec) if build else self.mod.MLP(in_dim=int(self.obs_spec["in_dim"]))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        if hasattr(self.model, "embed"):          # v3/v4 SetPrior
            self.body, self.head = self.model.embed, self.model.out
        else:                                     # v2 MLP
            self.body, self.head = self.model.net[:-1], self.model.net[-1]
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def features(self, comp: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        import json
        key = json.dumps(comp, sort_keys=True, ensure_ascii=False, default=str)
        if key not in self._cache:
            form = self.ps.normalize_formulation_for_mlp({"composition": comp}, self.obs_spec)
            x = self.mod.encode_obs(self.ps.mlp_input_example(form), self.obs_spec).astype(np.float32)
            with torch.no_grad():
                h = self.body(torch.from_numpy(x).unsqueeze(0))
                logits = self.head(h).squeeze(0).numpy()
            order = [self.actions.index(a) for a in ACTIONS]
            self._cache[key] = (h.squeeze(0).numpy(), logits[order])
        return self._cache[key]


# ------------------------------------------------------------------ adapter ---
class ResidualAdapter(nn.Module):
    def __init__(self, emb_dim: int, n_objectives: int, hidden: int = 32, max_shift: float = 1.5):
        super().__init__()
        self.max_shift = float(max_shift)
        self.g = nn.Sequential(nn.Linear(emb_dim + len(ACTIONS), hidden), nn.ReLU(),
                               nn.Linear(hidden, len(ACTIONS)))
        nn.init.zeros_(self.g[-1].weight); nn.init.zeros_(self.g[-1].bias)
        # Per-objective base-rate logit, pinned to the empirical rate (not trained): otherwise the
        # shift and the base rate are not identifiable and the shift absorbs the overall failure
        # rate. The shift therefore only models deviations from the base rate, as in tier 1.
        self.c = nn.Embedding(max(1, n_objectives), 1)
        nn.init.zeros_(self.c.weight)
        self.c.weight.requires_grad_(False)

    def shift(self, h, l):
        return self.max_shift * torch.tanh(self.g(torch.cat([h, l], dim=-1)))

    def forward(self, h, l, obj_idx, a_idx):
        s = self.shift(h, l).gather(1, a_idx.view(-1, 1)).squeeze(1)
        return self.c(obj_idx).squeeze(1) + s  # success logit


def _tensors(events, enc: PriorEncoder, obj_index: Dict[str, int]):
    H, L, O, A, Y, W = [], [], [], [], [], []
    for ev in events:
        h, l = enc.features(ev["before"])
        H.append(h); L.append(l)
        O.append(obj_index.get(ev.get("objective", ""), 0)); A.append(A_IDX[ev["action"]])
        Y.append(1.0 if ev["reward"] == "positive" else 0.0)
        W.append(0.5 if ev["reward"] == "unchange" else 1.0)
    f = lambda v, dt=torch.float32: torch.tensor(np.array(v), dtype=dt)
    return f(H), f(L), f(O, torch.long), f(A, torch.long), f(Y), f(W)


def _fit(events, enc, obj_index, max_shift, epochs=300, wd=1e-2, anchor=0.05, seed=0) -> ResidualAdapter:
    torch.manual_seed(seed)
    H, L, O, A, Y, W = _tensors(events, enc, obj_index)
    model = ResidualAdapter(H.shape[1], len(obj_index), max_shift=max_shift)
    with torch.no_grad():
        for o, i in obj_index.items():
            m = O == i
            s, n = float((Y[m] * W[m]).sum()), float(W[m].sum())
            p = (s + 1.0) / (n + 2.0)
            model.c.weight[i, 0] = math.log(p / (1 - p))
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=3e-3, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(epochs):
        loss = (bce(model(H, L, O, A), Y) * W).sum() / W.sum()
        # anchor: penalize the shift itself, so it stays near 0 unless the data insist
        loss = loss + anchor * model.shift(H, L).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def _logloss(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum())


def eligible(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in events if e.get("before") and e.get("action") in A_IDX]


def train_candidate(events: List[Dict[str, Any]], enc: PriorEncoder, cfg) -> Dict[str, Any]:
    """Time-ordered holdout comparison. Returns a report (and the refit model if promotable)."""
    ev = sorted(eligible(events), key=lambda e: (e.get("time", ""), e.get("event_id", "")))
    n_pos = sum(e["reward"] == "positive" for e in ev)
    report: Dict[str, Any] = {"n_events": len(ev), "n_positive": n_pos}
    if len(ev) < cfg.ADAPTER_MIN_EVENTS or n_pos < cfg.ADAPTER_MIN_POSITIVES:
        report["status"] = "locked"
        report["reason"] = (f"needs >= {cfg.ADAPTER_MIN_EVENTS} events and >= {cfg.ADAPTER_MIN_POSITIVES} "
                            f"positives (have {len(ev)} / {n_pos})")
        return report

    n_hold = max(1, int(round(len(ev) * cfg.ADAPTER_HOLDOUT_FRAC)))
    train, hold = ev[:-n_hold], ev[-n_hold:]
    objectives = sorted({e.get("objective", "") for e in ev})
    obj_index = {o: i for i, o in enumerate(objectives)}

    model = _fit(train, enc, obj_index, cfg.MAX_LOGIT_SHIFT)
    H, L, O, A, Y, W = _tensors(hold, enc, obj_index)
    with torch.no_grad():
        p_ad = torch.sigmoid(model(H, L, O, A)).numpy()
    tier1 = ActionPosterior(cfg.POSTERIOR_STRENGTH, cfg.MAX_LOGIT_SHIFT, cfg.UNCHANGE_WEIGHT).fit(train)
    p_t1 = np.array([tier1.predict(e.get("objective", ""), e["before"], e["action"]) for e in hold])
    base = np.array([(tier1.S[(e.get("objective", ""),)] + 1) / (tier1.N[(e.get("objective", ""),)] + 2)
                     for e in hold])
    y, w = Y.numpy(), W.numpy()
    report["holdout"] = {"n": len(hold), "logloss_adapter": _logloss(p_ad, y, w),
                         "logloss_tier1": _logloss(p_t1, y, w), "logloss_base_rate": _logloss(base, y, w)}
    h = report["holdout"]
    promote = h["logloss_adapter"] < min(h["logloss_tier1"], h["logloss_base_rate"])
    report["status"] = "promotable" if promote else "rejected"
    if promote:
        report["model"] = _fit(ev, enc, obj_index, cfg.MAX_LOGIT_SHIFT)  # refit on everything
        report["objectives"] = objectives
    return report


def save(model: ResidualAdapter, objectives: List[str], path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "objectives": objectives, "actions": ACTIONS,
                "emb_dim": model.g[0].in_features - len(ACTIONS), "max_shift": model.max_shift,
                "report": {k: v for k, v in report.items() if k != "model"}}, path)


class LoadedAdapter:
    def __init__(self, path: Path, enc: PriorEncoder):
        ck = torch.load(path, map_location="cpu")
        self.enc = enc
        self.model = ResidualAdapter(int(ck["emb_dim"]), len(ck["objectives"]), max_shift=float(ck["max_shift"]))
        self.model.load_state_dict(ck["state_dict"]); self.model.eval()

    def shifts(self, objective: str, comp: Dict[str, Any]) -> Dict[str, float]:
        h, l = self.enc.features(comp)
        with torch.no_grad():
            s = self.model.shift(torch.from_numpy(h).unsqueeze(0), torch.from_numpy(l).unsqueeze(0)).squeeze(0)
        return {a: float(s[i]) for i, a in enumerate(ACTIONS)}
