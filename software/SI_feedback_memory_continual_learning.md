# Supplementary Note: Persistent feedback memory and continual learning

## Motivation

In TRACE as described in Methods, external feedback acts only on the control state
$x_t=(s_t,\mathcal{M}_t,z_t)$. A rejected proposal is remembered through its signature in
$\Sigma_t(a)$ and the families of its target molecule in $\mathcal{F}^{\mathrm{avoid}}_t(a)$. An
accepted proposal moves the state and starts an escalation. These records belong to a single
session, and the retry records are cleared as soon as a proposal is accepted. Two knowledge sources
never change during or across campaigns: the literature transition pool $\mathcal{T}_a$, from which
the generator's evidence $\mathcal{E}(s_t,a)$ is drawn, and the action prior $f_\theta$, which
decides the candidate directions $\mathcal{C}_t$.

Two consequences follow. First, a modification proposed by the LLM and evaluated by MD is a
transition of exactly the kind TRACE learns from in the literature: a before state, an action, an
after state and an outcome. Yet it is not available as evidence. When the agent later returns to
the same formulation, the generator cannot read that this change was already tried, or that it
worked. Second, repeated feedback never changes which directions are considered. If a direction
fails again and again from a formulation across sessions, the prior proposes it again in every new
session.

This note describes an extension, implemented in the packaged program (`software/`), that closes
both gaps without changing any component described in Methods:

1. **Feedback becomes known information.** Every evaluated proposal is stored at once as a
   transition in the same schema as the literature transitions. It is retrievable by the
   action-conditioned RAG from the next round of the same session onward.
2. **Repeated feedback drives continual learning.** A tiered learner turns the accumulated
   feedback into a bounded correction of the frozen action prior. A no-training estimator acts
   from the first feedback; a learned adapter is added only once there is enough feedback and it
   has been shown to predict held-out feedback better.

The prior's training data and weights, the retrieval scoring, the generator and selector prompts,
and the controller rules are all unchanged. With both layers switched off, the program is the agent
of Methods. The campaigns reported in the main text were run without this extension.

## Overview

```
 round t:  s_t ──f_θ──► ℓ(s_t) ──(+ δ(s_t,o), bounded)──► C_t ──RAG over T_a ∪ T^fb_a──► proposal p_t
                                         ▲                                     ▲
                                         │ continual learning                  │ live write-back
                                         │ (tier 1 posterior / tier 2 adapter) │ (same round t+1)
 feedback: e_t = (s_t, a_t, s̃_{t+1}, r_t, o, ε_t) ──► ledger L ──► feedback transitions T^fb
```

All learned state is derived from a single append-only feedback ledger $\mathcal{L}$. The feedback
transitions and both learners can therefore be rebuilt from it at any time, or replaced by a
different learner, without losing data.

## 1. Feedback events

Each feedback produces an event

$$
e_t=\big(s_t,\ a_t,\ \tilde s_{t+1},\ r_t,\ o,\ \varepsilon_t\big),
$$

with the following fields:
- $s_t$ is the formulation the proposal was made from;
- $a_t$ is the tested action;
- $\tilde s_{t+1}=\operatorname{apply}(s_t,p_t)$ is the proposed formulation. It is recomputed for
  rejected proposals too, although the state does not move to it;
- $r_t\in\{+1,0,-1\}$ is the feedback;
- $o$ is the objective;
- $\varepsilon_t$ is optional evidence entered with the feedback (MD descriptors or a short note).

The event also records the proposal's change summary, the generator's rationale, the controller
mode, and the rank of $a_t$ under the frozen prior. Events are appended to $\mathcal{L}$ as soon as
the feedback is entered, so nothing is lost if a session is interrupted.

## 2. Feedback as known information

### 2.1 Feedback transitions

Events are grouped by the transition they test,
$\big(o,\ \mathrm{key}(s_t),\ \mathrm{key}(\tilde s_{t+1})\big)$, using the canonical formulation
key $\mathrm{key}(\cdot)$ of Methods. Each group becomes one transition in the literature schema
(Methods, Eq. for $\tau$):

$$
\tau^{\mathrm{fb}}=\big(s_{\mathrm{bef}}=s_t,\ a,\ s_{\mathrm{aft}}=\tilde s_{t+1},\ P=o,\ \pi,\ \kappa\big).
$$

The polarity aggregates all feedback on that transition. Let $n_+$, $n_-$ and $n_0$ be the numbers
of positive, negative and unchanged results. Then

$$
\pi=
\begin{cases}
+, & n_+>n_-,\\
-, & n_->n_+,\\
\varnothing, & n_+=n_-=0,
\end{cases}
$$

and a tie with $n_+=n_->0$ is resolved by the most recent result. The conditions $\kappa$ record the
feedback counts and, optionally, the evidence $\varepsilon$.

Because $\tau^{\mathrm{fb}}$ has the literature schema, it enters the action-matched pool
$\mathcal{T}_a\leftarrow\mathcal{T}_a\cup\mathcal{T}^{\mathrm{fb}}_a$ through the same precedent
builder. The before-end tiers, the after-end reversion check and the neighborhood summary all apply
to it without modification.

### 2.2 Roles in the retrieved evidence

| Feedback | Polarity | Before end ($\tau.s_{\mathrm{bef}}=s_t$) | After end ($\tau.s_{\mathrm{aft}}=\tilde s_{t+1}$) |
|---|---|---|---|
| accepted | $+$ | positive pattern: a direction known to work from this formulation | once the agent stands at $\tilde s_{t+1}$, the reverse change is a forbidden reversion |
| rejected | $-$ | negative pattern: a change already tested from this formulation and found worse | — (the state never reached $\tilde s_{t+1}$) |
| unchanged | $\varnothing$ | uncertain record, kept as context | — |

When the agent revisits $s_t$, its own feedback transitions fall into the `exact` before-state
tier, which is the first tier the generator reads. They complement the controller memory in
three ways:
- $\Sigma_t(a)$ and $\mathcal{F}^{\mathrm{avoid}}_t(a)$ are prompt constraints that hold within one
  session and are cleared after an acceptance. A rejected feedback transition stays in the evidence
  permanently, for every later visit to that formulation in any session.
- An accepted change is protected against reversal across sessions by the same exact-match rule
  that protects literature transitions. Within a session, only the escalation record
  $\mathcal{H}_t$ protects it.
- Feedback transitions enter the neighborhood tendency counts in the same way, so the agent's own
  results also shape the aggregate patterns around $s_t$.

### 2.3 Immediate availability within a session

The engine builds the precedent pool once per session and recomputes $\mathcal{E}(s,a)$ from it in
every round. The extension inserts the transition (or updates it, if the transition was seen
before) into this pool when the feedback is entered. As a result, the record is retrievable from
round $t+1$ of the same session.

The case where this matters most is the retry that follows a rejection. The generator is
re-queried for the same action from the same $s_t$. Besides the prompt constraint that the
rejected signature must not be repeated, it now reads the rejected change as an `exact`-tier
negative pattern, with its outcome and evidence. At the end of a session, and at the start of the
next one, the stored transitions are rebuilt from $\mathcal{L}$ and merged with the literature
corpus that the RAG loads.

### 2.4 Evidence and blindness

The protocol of Methods keeps the agent blind to MD descriptors: it observes only $r_t$. The
extension stores optional evidence $\varepsilon_t$ in the ledger in all cases. Whether
$\varepsilon_t$ is also written into the conditions $\kappa$ that the generator reads is a separate
switch (`EVIDENCE_TO_RAG`):
- **On (default for interactive use):** the generator can read why a change was accepted or
  rejected, which is useful when a chemist runs the program as a design tool.
- **Off:** the evidence is kept only for the record, and the blind protocol of Methods holds
  exactly.

The learners in Section 3 never use $\varepsilon_t$; they use only $r_t$.

## 3. Continual learning over repeated feedback

### 3.1 Principle: a bounded correction of a frozen prior

The literature prior $f_\theta$ is never retrained on feedback. Feedback is sparse, arrives one
result at a time, and is specific to one objective's evaluation, whereas $f_\theta$ encodes 911
papers' worth of literature practice. Instead, feedback adds a bounded, objective-specific shift to
the prior's logits $\ell(s)$ (Methods, MLP output):

$$
\ell'_a(s)=\ell_a(s)+\delta_a(s,o),\qquad |\delta_a|\le\Delta,\qquad \hat y'_a=\sigma\big(\ell'_a(s)\big),
$$

and the candidate set is drawn from $\hat y'$ instead of $\hat y$ (Methods, Eq. for $\mathcal{C}_t$).
The rest of the selection step is unchanged:
- the shift acts on $\mathcal{C}_0$ and on the regular-mode candidate sets $\mathcal{C}_t$;
- retry and escalation do not consult the prior, so they are unaffected;
- the selector still does not see the scores.

The construction has three properties:
- With no feedback, $\delta\equiv0$, so the agent is exactly the literature prior.
- No amount of feedback can move a logit by more than $\Delta$ ($\Delta=1.5$ by default). A
  literature-supported direction can be demoted, but cannot be erased.
- The prior itself stays objective-free, as in Methods. The shift depends on $o$, because whether a
  change counts as an improvement depends on the objective's evaluation.

All learners model the probability that testing action $a$ from state $s$ yields an improvement.
Each event contributes a success label and a weight,

$$
y=\mathbb{I}[r=+1],\qquad w=\begin{cases}1, & r\in\{+1,-1\},\\ w_0, & r=0,\end{cases}
$$

with $w_0=0.5$, so an unchanged result counts as half a non-success. The learners are arranged in
tiers by how much feedback they need.

### 3.2 Tier 1: hierarchical posterior (from the first feedback, no training)

Success rates are estimated by empirical-Bayes shrinkage through three nested levels. Each level's
estimate serves as the prior mean of the next. Let $S$ and $N$ be the weighted success and
observation counts at each level. The levels are the objective $o$, the action $a$ under $o$, and
the action $a$ from formulations with the same component identity $\bar s$ under $o$. Here $\bar s$
is the salt, solvent and additive identities with amounts ignored, so that evidence carries over
between ratio variants of one formulation. Then

$$
p_o=\frac{S_o+1}{N_o+2},\qquad
p_{oa}=\frac{S_{oa}+m\,p_o}{N_{oa}+m},\qquad
p_{o\bar s a}=\frac{S_{o\bar s a}+m\,p_{oa}}{N_{o\bar s a}+m},
$$

$$
\delta^{(1)}_a(s,o)=\operatorname{clip}\Big(\operatorname{logit}p_{o\bar s a}-\operatorname{logit}p_o,\ -\Delta,\ \Delta\Big),
$$

with pseudo-count $m=4$.

The shift is measured against the objective's base success rate. A direction is demoted only if it
fails more often than directions fail on average under this objective, and promoted only if it
succeeds more often. A single result moves the shift only slightly; consistent results move it
toward $\pm\Delta$.

Tier 1 requires no training. It is updated with every event, so a direction that keeps failing
from a formulation is already ranked lower at the next regular-mode draw of the same session. Its
limitation is that it cannot transfer to formulations with a different component identity.

### 3.3 Tier 2: residual adapter on the prior's representation (gated)

To generalize to similar but not identical formulations, tier 2 learns the shift as a function of
the prior's own representation of the state: the penultimate activation $z_3(s)\in\mathbb{R}^{256}$
and the logits $\ell(s)$ (Methods, MLP equations):

$$
\delta^{(2)}(s)=\Delta\tanh\Big(g_\psi\big([z_3(s)\,\|\,\ell(s)]\big)\Big)\in[-\Delta,\Delta]^8,
\qquad
P(y=1\mid s,a,o)=\sigma\big(b_o+\delta^{(2)}_a(s)\big).
$$

Here $g_\psi$ is a two-layer network with 32 hidden units. The adapter is trained with

$$
\min_\psi\ \frac{\sum_e w_e\,\mathrm{BCE}\big(\sigma(b_{o_e}+\delta^{(2)}_{a_e}(s_e)),\,y_e\big)}{\sum_e w_e}
\;+\;\lambda\,\overline{\big\|\delta^{(2)}(s_e)\big\|^2}
\;+\;\text{weight decay}.
$$

Several safeguards address the small, sequential and imbalanced feedback:
- **Data gate.** No training is attempted below 60 events or 10 accepted results.
- **Anchoring.** The output layer is zero-initialised, so the adapter starts as "no shift". The
  penalty $\lambda$ on the shift itself keeps it near zero unless the data insist.
- **Pinned base rate.** $b_o$ is fixed to $\operatorname{logit}p_o$ rather than learned. Otherwise
  the base rate and the shift cannot be separated, and with mostly negative feedback the shift
  absorbs the overall failure rate, demoting every action alike. With $b_o$ pinned, the adapter
  predicts the same quantity as tier 1, a deviation from the base rate.
- **Time-ordered promotion test.** Events are ordered by time and the newest 20 % are held out. A
  candidate adapter is promoted only if its weighted held-out log-loss is lower than both that of
  tier 1, fitted on the same earlier events, and that of the base rate. A promoted adapter is
  refitted on all events before use.
- **Versioning and fallback.** Each promoted adapter is saved as a new version in a registry, and
  any version can be re-activated. If a new candidate fails the test, the system falls back to
  tier 1 rather than keep an older adapter fitted on less data.

The learners are updated at the end of every session, or on demand.

### 3.4 Summary of the tiers

| Tier | Active from | Learns | Generalizes across | Safeguards |
|---|---|---|---|---|
| Write-back (Sec. 2) | first event | evidence for the generator | identical and similar before states (RAG tiers) | no training; polarity from aggregated feedback |
| 1: posterior | first event | per-objective success rates of directions | ratio variants of one formulation | shrinkage to base rate, cap $\Delta$ |
| 2: adapter | ≥60 events, ≥10 accepted | success as a function of the prior's state representation | structurally similar formulations | gate, anchoring, pinned base rate, held-out promotion, versioning |

## 4. Mechanics on logged campaign feedback

The following check applies the extension to the feedback logged in five earlier sessions. It is a
check that the mechanics behave as designed, not an evaluation of optimization performance.

The five sessions were run under one objective (high-voltage oxidative stability). Their logs
contain 71 feedback events: 7 accepted, 59 rejected and 5 unchanged. In 24 rejected events from
the older logs, the proposal itself was not recorded, so no proposed formulation could be
reconstructed and no transition could be formed.

**Write-back.**
- The remaining events produced 21 feedback transitions: 6 positive, 10 negative and 5 uncertain.
- Every transition, after passing through the unchanged precedent builder, received the polarity of
  its feedback.
- A rejected additive proposal entered during a simulated session was returned as an `exact`-tier
  negative pattern by the next retrieval for the same action and state.

**Tier 1.**
- The base success rate was $p_o=0.11$.
- From EMC:PC (7:3) with 1.0 M LiPF$_6$, the starting formulation of these campaigns (18 events
  from this component identity), the shifts were $+1.5$ for $\textsf{solvent\_replace}$ and $+0.9$ for
  $\textsf{solvent\_conc\_shift}$.
- They were $-0.5$ for $\textsf{add\_new\_solvent}$, $-1.4$ for $\textsf{salt\_replace}$ and $-1.5$
  for $\textsf{additive\_change}$, with 0 for untested actions.
- This reflects the logged outcomes: all accepted changes were solvent replacements or ratio
  changes, and none of the 15 additive proposals was accepted.

**Tier 2.**
- With 7 accepted results, the adapter is locked at the default gate.
- When it was trained regardless, its held-out log-loss on the newest 14 events was 0.22. This
  beats the base rate (0.27) but not tier 1 (0.08), so it was not promoted, and tier 1 remained
  active.
- At the current amount of feedback the identity-level estimator is the better model, and the
  promotion test is what prevents a premature switch.

## 5. Limitations

- **Feedback quantity.** The learners can only be as informative as the feedback. With tens of
  events, tier 1 reflects the outcomes on the formulations actually visited and says nothing about
  others. Tier 2 is designed for the regime in which feedback accumulates over many campaigns.
- **Dependence on the objective's evaluation.** All estimates are kept per objective, and feedback
  under one objective does not shift another. Within an objective, a change in how feedback is
  judged, for example a revised MD acceptance rule, makes older feedback partly inconsistent with
  newer feedback. The ledger keeps every event, so such feedback can be filtered and the learners
  rebuilt.
- **Selection effects.** Feedback is collected along the agent's own trajectories, so directions
  the prior rarely proposes receive few observations. The bounded shift and the shrinkage towards
  the base rate limit the effect of this on ranking, but do not remove it.
- **Scope of write-back.** A transition is formed only when the proposal can be applied to a
  concrete formulation. Proposals that fail to apply are still counted by the learners but produce
  no precedent.

## Implementation

All files are in `software/`. Run `../.venv/bin/python run.py` for an interactive session. Use
`learning_cli.py` to inspect or update the store: `status`, `update`, `import <run folder>`,
`shifts <state.json>`, `activate posterior|<version>`.

| Component | File | Store |
|---|---|---|
| feedback ledger | `learning/ledger.py` | `store/feedback_ledger.jsonl` |
| feedback transitions, live write-back | `learning/knowledge.py` | `store/discovered_papers.json`, `store/merged_corpus.json` |
| tier 1 posterior | `learning/posterior.py` | `store/action_posterior.json` |
| tier 2 adapter | `learning/adapter.py` | `store/models/adapter_vN.pt` |
| engine hooks, registry, end-of-session update | `learning/manager.py`, `run.py` | `store/learning_registry.json` |

| Setting (`config.py`) | Default | Effect |
|---|---|---|
| `LIVE_WRITE_BACK` | `True` | feedback transitions retrievable from the next round of the same session |
| `WRITE_BACK_NONPOSITIVE` | `True` | rejected and unchanged results are written back, not only accepted ones |
| `ASK_EVIDENCE` / `EVIDENCE_TO_RAG` | `True` / `True` | ask for MD evidence; expose it to the generator (`False` keeps the agent blind) |
| `ONLINE_PRIOR_SHIFT` | `"auto"` | `"off"` uses the frozen literature prior only |
| `POSTERIOR_STRENGTH` $m$ | 4 | pseudo-count of each shrinkage level |
| `MAX_LOGIT_SHIFT` $\Delta$ | 1.5 | cap on the shift of any logit |
| `UNCHANGE_WEIGHT` $w_0$ | 0.5 | weight of an unchanged result |
| `ADAPTER_MIN_EVENTS` / `ADAPTER_MIN_POSITIVES` | 60 / 10 | data gate of tier 2 |
| `ADAPTER_HOLDOUT_FRAC` | 0.2 | newest fraction of events held out for the promotion test |

Setting `LIVE_WRITE_BACK = False`, `WRITE_BACK_NONPOSITIVE = False` and
`ONLINE_PRIOR_SHIFT = "off"`, with an empty store, reproduces the agent of Methods.
