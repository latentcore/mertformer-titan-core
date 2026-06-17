# REPRODUCE — verify & launch

Canonical command surface (consolidated from `reports/final_commands.md` + `repro/`).
Turkish: [REPRODUCE_TR.md](REPRODUCE_TR.md).

## 0. Bootstrap (offline-first)
```bash
bash scripts/bootstrap_venv.sh        # creates .titan-venv with pinned deps (Python 3.11)
```

## 1. Verify the repo (no network, no GPU) — the CI `verify` gate
```bash
bash scripts/verify_all.sh            # secret-scan + pytest (370 passed, 4 skipped) +
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
