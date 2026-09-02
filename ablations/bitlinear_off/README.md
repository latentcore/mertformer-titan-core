# Ablation: BitLinear Off

**Purpose**: Compare BitNet vs. standard dense weights.

**Config change**:
- `use_bitnet: false`

**[2026-09-02] Prerequisite bug fixed**: `use_bitnet: false` previously had **no effect on the
model at all** — `layers/ffn.py`, `layers/mla.py`, `layers/moe.py` and `layers/liquid.py` called
`BitLinear(...)` unconditionally, so this ablation would have silently produced a meaningless
zero-delta result if run as-is. Fixed via `layers/bitlinear.py::make_linear`, which all four
files now call with `cfg.use_bitnet`. See `DECISIONS.md` and `BACKLOG.md` for the full record.

**Runner**: `scripts/run_bitlinear_ablation.py` (same $0-pilot methodology as
`scripts/run_liquid_ablation.py` — see `no_liquid/README.md`). CPU-smoke-verified post-fix
(the two arms now genuinely diverge; before the fix they were byte-identical).

**Status**: Not run for a real signal yet — CPU smoke proves the toggle works, it is not a
measured result. Execute on training hardware (`python scripts/run_bitlinear_ablation.py --steps
500 --device cuda` or higher) and record results in `ablations/results.md`.
