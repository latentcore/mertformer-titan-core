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
**Fix:** Use `TITAN_TOKEN_BUDGET_MODE=fixed_steps` or set `TITAN_MAX_STEPS` explicitly.

## 9) NCCL hang / deadlock (multi-GPU)
**Symptom:** Multi-GPU training stalls with no progress; a rank appears stuck in a collective; `NCCL timeout` in logs.
**Fix:** Ensure all ranks reach every collective (NaN/skip paths must broadcast the same decision — the train loop already DDP-syncs these). Set `NCCL_TIMEOUT` higher for slow I/O, and avoid passing non-tensor metadata through Accelerate `concatenate()` (drops str/int by design — see `train/trainer_data.py`). If it persists, restart from the last checkpoint.

## 10) Checkpoint corruption / truncated checkpoint
**Symptom:** Resume fails with an unpickling/`EOFError`/size-mismatch on `best.pt`/`latest.pt`/`final.pt`.
**Fix:** Checkpoints are written atomically (`*.pt.tmp` → `os.replace`), so a provider kill leaves the previous checkpoint intact — resume from `latest.pt` (or `best.pt`). Verify integrity with the SHA256 sidecar before reuse. Never resume from a `*.pt.tmp` file.

## 11) Tokenizer mismatch (train/eval)
**Symptom:** Eval or resume produces garbage; vocab size differs; a `gpt2`-style tokenizer (vocab 50257) is loaded instead of the canonical Llama-3 tokenizer (vocab 128256).
**Fix:** The canonical path uses `utils/tokenizer_resolver.py` with **no silent fallback** and stamps a `tokenizer_identity` hash into every checkpoint/shard. If a mismatch is reported, confirm `cfg.teacher_model_id` / the resolved tokenizer and re-run; do not override with a generic tokenizer.
