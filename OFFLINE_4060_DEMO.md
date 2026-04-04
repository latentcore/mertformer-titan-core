# Offline RTX 4060 Demo Mode

This bundle includes a no-network laptop demo lane.

Goal:
- run a tiny training proof locally
- no teacher model
- no Hugging Face token
- no dataset download
- no WandB
- no smart runner

Command:
```bash
cd ~/MertFormer_45K_Launch_Bundle_20260401_2130
bash run.sh --offline-4060-demo
```

Optional knobs:
```bash
TITAN_DEMO_MAX_STEPS=20 TITAN_DEMO_SEQ_LEN=256 TITAN_DEMO_BATCH_SIZE=1 bash run.sh --offline-4060-demo
```

What it uses:
- `datasets/offline_demo/train.jsonl`
- `datasets/offline_demo/validation.jsonl`
- a built-in byte-level demo tokenizer
- a tiny 2-layer MertFormer config with MoE + Liquid both active in non-overlapping layers

What it writes:
- `logs/offline_4060_demo.jsonl`
- `reports/offline_4060_demo_summary.json`
- `checkpoints/offline_4060_demo/`

Notes:
- this is a local architecture proof lane, not the full 45K production run
- internet-free here means no data/teacher/token dependency during launch
- local Python/Torch availability is still required on the machine
