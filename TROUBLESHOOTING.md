# Troubleshooting — MertFormer Titan (Build 30 V2)

## 1) HF_TOKEN missing (online training / teacher)
**Symptom:** Preflight or training warns about gated teacher or dataset access.
**Fix:** Set `HF_TOKEN` in `.env` or environment, then retry.

## 2) Stage JSONL missing (offline mode)
**Symptom:** Preflight warns `Stage JSONL missing` or training fails in offline mode.
**Fix:** Run `python3 scripts/data_pipeline.py` or switch to online mode (`TITAN_OFFLINE=0`).

## 3) accelerate config mismatch
**Symptom:** Training runs on 1 GPU or ignores expected distributed setup.
**Fix:** Delete or update `~/.cache/huggingface/accelerate/default_config.yaml`, or set `TITAN_FORCE_ACCELERATE_RECONF=1` and rerun `run.sh`.

## 4) cuda.lock missing
**Symptom:** Preflight warns `CUDA lock missing` on training hardware.
**Fix:** Run `python3 scripts/write_cuda_lock.py` on the target GPU machine.

## 5) Precomputed logits path mismatch
**Symptom:** Offline distillation fails to find logits shards.
**Fix:** Ensure `TITAN_LOGITS_PATH` points to `./datasets/logits/` (or your custom path) and that shards exist.

## 6) torch.compile on MPS
**Symptom:** CfC fast path errors on macOS/MPS.
**Fix:** Keep `liquid_fast_path=false` on MPS. (Default behavior already guards this.)

## 7) OOM on training
**Symptom:** CUDA out of memory or process killed.
**Fix:** Reduce batch size/seq length, enable gradient checkpointing, or scale hardware.

## 8) Token budget runs longer than expected
**Symptom:** Training exceeds planned steps.
**Fix:** Use `TITAN_PROFILE=fixed_steps` or set `TITAN_MAX_STEPS` explicitly.
