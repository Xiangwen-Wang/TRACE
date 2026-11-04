# Electrolyte Formulation Optimizer (packaged software)

Self-contained package: start from a local initial_formulation file -> MLP ranks actions ->
top-3 directions -> RAG-LLM concrete proposal -> (optional) auto-generated MD job -> manual/MD
feedback loop. Every feedback is stored as new knowledge at once, and a continual-learning layer
keeps improving the system as feedback accumulates across runs.

> This folder does not depend on any other file in the repo; you can move it anywhere. It needs
> the project venv (already has torch/openai/rdkit/sentence-transformers) and an `OPENAI_API_KEY`
> (used by the RAG-LLM proposal step; entered at startup).

## Run
**Option 1 (recommended, app-like):** double-click `start.command` in Finder (or `./start.command`
in a terminal). It locates the project venv automatically and starts the program.

**Option 2 (terminal):**
```bash
cd software
../.venv/bin/python run.py
```

Then follow the prompts:
0. **Enter your OpenAI API key** (`sk-...`, hidden input). Optionally "remember" it to
   `store/.openai_key` (mode 600; delete the file to forget). You may also `export OPENAI_API_KEY=...`
   beforehand to skip this. A one-time connectivity check runs at startup (failure only warns).
1. **Enter the path to a local initial_formulation file.** Accepts `{state0:{composition}}`,
   `{composition:{...}}`, or a bare `{salt, solvents, ...}`; if `objective` is missing it defaults
   to high_voltage_oxidative_stability.
2. The program reads it -> **MLP ranking -> prints the top-3 action directions -> one RAG-LLM
   concrete proposal per step**.
3. **Detects a local MD toolchain** (`gmx` / `packmol` / `Multiwfn`):
   - If available: it asks for the template folder holding the required MD files (default
     `md_templates/`), then **auto-generates a GROMACS job directory every step**
     (`md_jobs/<run>/step_XX/`: packmol.inp, topol.top, mdp files, RESP2.sh, run.sh, `MANIFEST.md`).
     **Generate-only, not submitted** — add the per-species files listed in MANIFEST, run
     `bash run.sh` on a server, compute the solvation descriptors, then return feedback.
   - If not available: enter `positive / negative / nochange` directly (`quit` to stop).
4. **After each feedback, optionally enter evidence**: MD descriptors as `name=value; name=value`,
   a path to a `.json` file of descriptors, or a short note (Enter to skip).
5. **On stop, a SCREENING step runs automatically:** the precedent store is rebuilt from the full
   feedback ledger and the learners are updated (see below).

## Feedback memory and continual learning
All learned state lives in `store/` and is derived from ONE append-only file,
`store/feedback_ledger.jsonl`, so every learner can be rebuilt or replaced without losing data.
The literature prior (`MLP_new/mlp_prior_state_new_v5.pt`) is never modified.

**1. Feedback becomes known information (every feedback, immediately).**
Each feedback event is written to the ledger: the state, the action, the concrete proposed
formulation (recomputed for rejected proposals too), the change, the generator's rationale, the
reward and any MD evidence. Each distinct transition (objective, before-state, proposed state) is
then turned into a literature-schema precedent and handled by the same `build_precedents()` as the
literature:
| feedback | RAG polarity | what the generator sees |
|---|---|---|
| positive | positive | a positive pattern; its after-state also feeds the forbidden-reversion check |
| negative | negative | a negative pattern ("this was already tried from this state and failed") |
| no change | mixed/uncertain | context only |

Repeated feedback on one transition is aggregated (counts in the record; majority verdict, the
latest wins ties), and MD descriptors/notes are attached as conditions. With
`LIVE_WRITE_BACK` the new precedent is inserted into the running engine at once, so it is
retrievable **from the next step of the same session**. At startup and on stop,
`store/discovered_papers.json` and `store/merged_corpus.json` (base corpus + feedback precedents)
are rebuilt from the ledger and fed to the RAG.

**2. Continual learning over repeated feedback (tiered by how much data there is).**
Feedback only ever adds a *bounded* logit shift to the frozen prior at selection time
(`score = prior_logit + shift`, `|shift| <= MAX_LOGIT_SHIFT`); with no feedback the agent is
exactly the literature prior. Shifts are printed whenever they move an action.

- **Tier 1: hierarchical posterior (active from the first event, no training).** Success rates
  are estimated with empirical-Bayes shrinkage objective -> action -> (state, action), where the
  state is the component identity (salt, solvents, additives, ignoring amounts), and
  `shift = logit(p_state,action) - logit(p_objective)`. Little evidence gives a small shift;
  repeated failures of a direction from a formulation push it down, and repeated successes pull it
  up. Knobs: `POSTERIOR_STRENGTH`, `MAX_LOGIT_SHIFT`, `UNCHANGE_WEIGHT`.
- **Tier 2: residual adapter (gated).** A small network on the prior's own 256-d state
  representation and logits predicts the same shift, so it can generalize to *similar*
  formulations rather than only identical ones. Safeguards for sparse, sequential data:
  - it is not trained below `ADAPTER_MIN_EVENTS` / `ADAPTER_MIN_POSITIVES`;
  - it starts at zero shift and is anchored there (weight decay plus a penalty on the shift);
  - the base rate is pinned, not learned;
  - each candidate is evaluated on the newest `ADAPTER_HOLDOUT_FRAC` of events (time-ordered)
    and **promoted only if its held-out log-loss beats both tier 1 and the base rate**;
  - promoted versions are saved as `store/models/adapter_vN.pt`, recorded in
    `store/learning_registry.json`, and can be rolled back.

  If a new candidate loses, the system falls back to tier 1.
- `ONLINE_PRIOR_SHIFT = "off"` in `config.py` restores the frozen-prior behaviour, for example
  for blind-discovery experiments. Feedback is still stored and written back to the RAG.

`EVIDENCE_TO_RAG = False` keeps entered MD evidence in the ledger only, so the generator stays blind to
descriptors (the main-text protocol). The design is written up as an SI note in
`SI_feedback_memory_continual_learning.md`.

Rewards use only positive/negative/no-change. Descriptors are stored as evidence for the RAG
records and are not a training target.

### `learning_cli.py`
```bash
../.venv/bin/python learning_cli.py status              # ledger, precedents, per-action posterior, active learner
../.venv/bin/python learning_cli.py update              # rebuild precedents + retrain/re-gate learners
../.venv/bin/python learning_cli.py import <run folder | attempts.json> [--objective O]
                                                        # add feedback from earlier engine runs
../.venv/bin/python learning_cli.py shifts state.json   # the shift the active learner gives a formulation
../.venv/bin/python learning_cli.py activate posterior  # or: activate 3  (adapter v3) — manual rollback
```

## Files
```
start.command     macOS double-click launcher (locates venv -> run.py)
run.py            orchestrator (key prompt + terminal flow + engine hooks + MD hook + feedback memory + learning update on stop)
config.py         paths and knobs
md_scaffold.py    MD tool detection + generate a job directory from a formulation (scaffold, not submitted)
learning/         feedback memory + continual learning:
                    ledger.py (event store), knowledge.py (feedback -> RAG precedents, live upsert),
                    posterior.py (tier 1), adapter.py (tier 2), manager.py (engine hooks, registry)
learning_cli.py   status / update / import / shifts / activate
md_templates/     your GROMACS templates (em/NVT/NPT/prod.mdp, topol.top, RESP2.sh, 19.inp)
store/            persistent learning store (accumulates across runs; ships empty = fresh start)
runs/  md_jobs/   per-run working area / MD job output
# engine:
interactive_feedback_agent_terminal_top3.py, literature_rag.py, action_family_rag.py,
prior_suggestions.py, MLP_new/ (model definition + checkpoint), literature_database.json, *_families.json, ...
```

## Known limits
- MD is a **scaffold**: it does not invent 3D structures or force-field parameters; molecule
  parameterization is delegated to `RESP2.sh` + GAFF/antechamber (see each job's MANIFEST). The
  packmol counts, salt count and box size are commented estimate defaults — tune them for your
  system (constants at the top of `md_scaffold.py`).
- Feedback is sparse. Tier 1 adapts per formulation (component identity) and does not transfer to
  unseen formulations; tier 2 is meant to do that once enough feedback exists and it passes the
  held-out gate.
- The success reward depends on the objective's rubric, so posterior and adapter estimates are kept
  per objective; feedback under one objective does not shift another.

## Note on the MD template files
`md_templates/` contains GROMACS template files (some `.mdp` files have Chinese comments). They are
treated as data, not program code, and are passed through to the generated jobs unchanged.
