# Launch-Time Decisions Checklist

**Purpose:** BACKLOG I.2 #19 — "Launch-anı düzeltilmemiş regularizasyon/erken-durdurma kararları — henüz verilmedi" (launch-time uncorrected regularization/early-stopping decisions — not yet given). A pre-flight checklist to fill in, immediately before the real 45K `--go`, not before.

Fill in each row at actual launch time (not now — these are genuinely launch-time decisions, dependent on the final corpus/config state at that moment):

| Decision | Question to answer at launch time | Current default (if unset) | Filled in at launch (date + value) |
|---|---|---|---|
| Weight decay | Confirmed value for the 45K run — same as pilot (`config/config.py` default) or overridden? | `config/config.py`'s dataclass default | _pending_ |
| Gradient clipping | `max_grad_norm` — confirmed for canonical scale, not just pilot scale? | `config/config.py`'s dataclass default | _pending_ |
| Early stopping | Is there an early-stop criterion for the 45K run, or does it always run to `TITAN_MAX_STEPS`? | none configured — runs to completion or divergence-guard trip | _pending_ |
| Divergence guard | ON or OFF for this specific run (see [divergence_guard_decision_brief.md](divergence_guard_decision_brief.md)) | ON (`use_divergence_guard=True`) | _pending_ |
| Liquid/CfC inclusion | Keep or drop per [liquid_keep_or_drop_brief.md](liquid_keep_or_drop_brief.md) | included (canonical architecture) | _pending_ |
| Lane choice | `offline_clean` or `online_teacher` per [lane_cost_tradeoff_brief.md](lane_cost_tradeoff_brief.md) | neither pre-selected | _pending_ |
| `TITAN_STRICT_TOKEN_BUDGET` | Confirmed ON for the real run (now defaulted ON in both launch scripts as of 2026-07-12 — verify it wasn't overridden back to 0) | `1` (both launch scripts) | _pending_ |
| Checkpoint off-site backup | Off-site copy destination decided and reachable BEFORE the run starts (see BACKLOG #36 — the 14-May H200 checkpoint loss) | none configured | _pending_ |
| Resume policy | `TITAN_AUTO_RESUME` / `TITAN_RESUME_ALLOW_PARTIAL` confirmed for this specific launch | lane-script defaults | _pending_ |

## Why this exists as a checklist, not a script

Several of these (Liquid inclusion, lane choice, divergence-guard keep/remove) are genuinely one-time human judgment calls informed by evidence briefs already prepared (linked above) — a script cannot make them. Others (off-site backup destination, weight-decay override) depend on launch-day infrastructure that doesn't exist yet. This checklist's job is to make sure none of them get skipped silently at the moment of `--go`.
