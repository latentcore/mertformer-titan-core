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

## 2. Check training readiness (no training)
```bash
bash zero_touch_start.sh --check-only   # prints train_allowed + reason code + blockers
```

## 3. Launch the real run (target hardware: H100/H200)
```bash
bash zero_touch_start.sh                # canonical 45K owned lane: readiness verdict,
                                        # run lock, resume policy, post-train autorun
```
- Step bound is `cfg.max_steps`, controlled by env `TITAN_MAX_STEPS` (default 45000). A few-step smoke of the canonical architecture (no behavior-param change): `TITAN_MAX_STEPS=2 bash zero_touch_start.sh`.
- Online-teacher distillation needs `HF_TOKEN`; the strict offline lane needs precomputed top-k logits (see blockers in [STATUS.md](STATUS.md)).

## 4. Full closure ladder (regenerates artifacts; optional, slow)
```bash
bash scripts/final_one_shot.sh          # zips + SBOM + manifests; GitHub-policy step 403 on a
                                        # free private repo is expected and skipped (visibility unchanged)
```
