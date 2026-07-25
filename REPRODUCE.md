# REPRODUCE — verify & launch

Canonical command surface (consolidated from `reports/final_commands.md` + `repro/`).
Turkish: [REPRODUCE_TR.md](REPRODUCE_TR.md).

## 0. Bootstrap (offline-first)
```bash
bash scripts/bootstrap_venv.sh        # creates .titan-venv with pinned deps (Python 3.11)
```

## 1. Verify the repo (no network, no GPU) — the CI `verify` gate
```bash
bash scripts/verify_all.sh            # secret-scan + pytest (578 passed, 5 skipped) +
                                      # preflight + operator-mode overfit smoke + gates + manifests
```
This runs the K4 drills (checkpoint save→restore→resume) and the CfC/MoE tolerance parity at toy scale. None of this trains the canonical model.

## 1b. Teacher-free smoke (no 70B teacher download, no GPU needed)
```bash
.titan-venv/bin/python scripts/train_smoke.py --steps 50 --cleanup   # real fwd/bwd/opt on the canonical model; no teacher
.titan-venv/bin/python scripts/checkpoint_restore_drill.py           # save → reload → allclose
.titan-venv/bin/python scripts/resume_compat_check.py                # resume from a saved step
.titan-venv/bin/python scripts/cfc_moe_tolerance_check.py            # MoE seq↔parallel + Liquid parity
```
The 70B teacher is loaded only when `distill_alpha > 0` (`train.py`). `train_smoke.py` never constructs a teacher; for a teacher-free pass through the full training entry, set `TITAN_DISTILL_ALPHA=0` (KD-off, pure cross-entropy) — that path never loads or downloads the teacher. Keep `HF_TOKEN` unset (and `TITAN_SKIP_PHASE0=1`) so the optional phase-0 precompute can't fetch it either.

## 2. Check training readiness (no training)
```bash
bash zero_touch_start.sh --check-only   # prints train_allowed + reason code + blockers
```

## 2b. Laptop-scale pre-flight probes (before spending cloud GPU-hours)

Two self-contained, zero-argument orchestrator scripts let you sanity-check a candidate LR regime on a local CUDA laptop GPU **before** paying for cloud H100/H200 hours. Neither trains the canonical 3.67B model or makes a capability/scaling claim — both are LR-safety ("does grad_norm stay finite and does loss descend") gates only. Both bootstrap their own dependencies (never `torch`/`torchvision`/`torchaudio`/`triton`/`bitsandbytes` — those must already match your local CUDA build) and package a `REPORT.md` + output ZIP on completion.

```bash
python3 scripts/preflight_run.py             # 36M param toy probe, no config overlay needed
python3 scripts/preflight_run_pilot171m.py    # 172.67M param Go/No-Go pilot (BACKLOG.md T1.5)
```

| | `preflight_run.py` | `preflight_run_pilot171m.py` |
|---|---|---|
| Params | ~36M (hardcoded toy architecture) | 172.67M (measured; `config/model/mertformer_pilot_stabilization.yaml`) |
| Config source | hardcoded constants in the script | `MERTFORMER_MODEL_CONFIG` overlay, asserted against a live `cfg` before training starts |
| Optimizer | plain AdamW | GaLore + 8-bit Adam (same as the real 45K; falls back to plain AdamW with a printed warning if `bitsandbytes` is missing) |
| Purpose | cheapest possible smoke of the LR regime | the documented Go/No-Go stabilization gate before committing to 45K |
| Output | `preflight_run_output.zip` | `pilot171m_run_output.zip` |

Both resume automatically on rerun (`TITAN_AUTO_RESUME=1` semantics); delete the output zip and work directory to start over. Self-contained, portable versions of both (repo snapshot + script, no local clone needed) are prepared as needed for target-machine handoff — see each script's own module docstring for the exact bundle contents.

## 3. Launch the real run (target hardware: H100/H200)
```bash
bash zero_touch_start.sh                # canonical 45K owned lane: readiness verdict,
                                        # run lock, resume policy, post-train autorun
```
- Step bound is `cfg.max_steps`, controlled by env `TITAN_MAX_STEPS` (default 45000). A few-step **teacher-free** smoke of the canonical architecture: `TITAN_DISTILL_ALPHA=0 TITAN_SKIP_PHASE0=1 TITAN_MAX_STEPS=2 bash zero_touch_start.sh`. Drop `TITAN_DISTILL_ALPHA` only when you actually want the 70B teacher / KD (which loads/downloads it).
- Online-teacher distillation needs `HF_TOKEN`; the strict offline lane needs precomputed top-k logits (see blockers in [STATUS.md](STATUS.md)).

## 4. Full closure ladder (regenerates artifacts; optional, slow)
```bash
bash scripts/final_one_shot.sh          # zips + SBOM + manifests; GitHub-policy step 403 on a
                                        # free private repo is expected and skipped (visibility unchanged)
```
