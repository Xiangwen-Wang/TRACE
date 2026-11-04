"""
Feedback memory and continual learning for the packaged optimizer.

  ledger.py     append-only record of every feedback event (the single source of truth)
  knowledge.py  feedback events -> literature-schema precedents (positive AND negative),
                written back to the RAG corpus and injected live into the running session
  posterior.py  tier 1: hierarchical Beta posterior over (objective, state, action) -> a
                bounded, evidence-weighted shift on the frozen MLP prior logits
  adapter.py    tier 2: residual adapter on the prior's structure features, trained on
                feedback only, promoted only if it beats tier 1 on time-ordered held-out events
  manager.py    orchestrates the tiers, the model registry and the engine hooks
"""
