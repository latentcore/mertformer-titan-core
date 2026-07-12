# Post-45K Research Agenda

**Purpose:** consolidates BACKLOG I.7 #66, #67, #68, #69, #70, #72, #80 — seven deep-research questions that genuinely require a trained checkpoint (or are pure research-design questions) and where building a stub "eval harness" now would be theater, not evidence. Each entry states a concrete, falsifiable acceptance criterion so the question is answerable, not just askable, once a checkpoint exists.

## #66 — Context scaling & memory wall
**Question:** how does MertFormer Titan's effective usable context degrade as sequence length approaches/exceeds `max_seq_len` (4096)? Where's the actual host-device bandwidth wall for the GQA KV-cache at canonical scale?
**Acceptance criterion:** a measured perplexity-vs-position curve on sequences up to and beyond 4096 tokens, plus a measured (not estimated) KV-cache memory/bandwidth profile on the target hardware.
**Blocked on:** real checkpoint + real GPU profiling access.

## #67 — Continual learning / catastrophic forgetting
**Question:** does a second fine-tuning pass on new data measurably degrade held-out performance on the original 45K training distribution?
**Acceptance criterion:** held-out perplexity (eval/held_out_ppl.py) measured before and after a defined continual fine-tune, on the SAME held-out set.
**Blocked on:** real checkpoint + a defined second-stage fine-tune dataset (doesn't exist yet).

## #68 — Long-horizon credit assignment
**Question:** does the model's training signal meaningfully propagate across long dependency chains, or does it effectively only learn short-range statistics?
**Acceptance criterion:** a synthetic long-range-dependency probe (e.g. needle-in-haystack style) with a measured accuracy-vs-distance curve.
**Blocked on:** real checkpoint.

## #69 — Tool-grounded planning reliability
**Question:** given the SDK's chat/tool-executor surface (`orchestrator/tool_executor.py`), how reliably does the model choose to invoke a tool vs. hallucinate an answer, once trained?
**Acceptance criterion:** measured tool-invocation precision/recall on a fixed task set requiring tool use.
**Blocked on:** real checkpoint + a defined tool-use eval set (doesn't exist yet — `eval/agentic_suite.py`'s current suite is a toy/placeholder, see BACKLOG #45's random-init-baseline work this pass).

## #70 — Causal abstraction / counterfactual & partial-observability world modeling
**Question:** does the model's internal representation support counterfactual reasoning distinguishable from surface pattern-matching?
**Acceptance criterion:** a controlled counterfactual-probe eval (e.g. paired factual/counterfactual prompts) with measured consistency.
**Blocked on:** real checkpoint + eval design (a genuinely hard research problem, not just an engineering task — likely needs its own literature review before an eval can even be designed well).

## #72 — Mechanistic interpretability → intervention
**Question:** can specific circuits/attention heads/MoE experts be identified and causally intervened on to change specific behaviors?
**Acceptance criterion:** at least one reproducible intervention (e.g. ablating a specific expert/head measurably changes a specific, predicted output property).
**Blocked on:** real checkpoint + interpretability tooling (not yet built — this is its own multi-week workstream, out of scope to stub).

## #80 — Watermarking
**Question:** should MertFormer Titan ship with statistical text watermarking (e.g. a Kirchenbauer-style green/red-list logit bias), and if so, what's the quality/detectability tradeoff?
**Acceptance criterion:** a design decision recorded in DECISIONS.md, informed by a measured quality-cost tradeoff on a real checkpoint if the decision is "yes."
**Blocked on:** real checkpoint for the quality-cost measurement; the design QUESTION itself (yes/no, and which scheme) could be decided pre-45K as a policy matter, independent of having a checkpoint.

## Why these seven are grouped here instead of getting individual probe scripts

Unlike calibration/bias/toxicity/hallucination/adversarial/membership-inference (BACKLOG #71/73/76/77/78/81, addressed this pass with real `eval/*_probe.py` harnesses because a small, honest, offline proxy methodology exists for each), these seven either (a) need eval infrastructure that doesn't exist yet and would take real design work to build correctly, or (b) are genuinely open research questions where the "eval" IS the research contribution, not a pre-existing methodology to wire up. Building a hollow stub for any of these would be exactly the "cosmetic-fix disease" this whole project has spent this session rooting out — a script that LOOKS like progress but measures nothing real.
