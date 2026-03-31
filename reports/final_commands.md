# Final Commands

## Canonical 45K Start Gate

```bash
bash zero_touch_start.sh --check-only
```

## Canonical 45K Launcher

```bash
TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh
```

Optional online teacher lane:

```bash
HF_TOKEN=... TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh
```

## Canonical Closure

```bash
bash scripts/final_one_shot.sh
```

## Refresh Backlog and Readiness Reports

```bash
python3 scripts/build_master_closure_matrix.py
python3 scripts/build_train_readiness_contract.py --allow-not-ready
python3 scripts/build_max_closure_handoff.py
```

Current readiness status: `TRAIN_ALLOWED`
Recommended path: `offline_clean`
