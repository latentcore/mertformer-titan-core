# Environment Variables — canonical training/run surface

Single index of the environment variables that govern the **canonical 45K
training / Phase-0 precompute / orchestration** path. Defaults are the in-code
defaults (`config/config.py`, `train/`, `scripts/`).

> Scope: this lists the **training-relevant** knobs. Out-of-scope families
> (`MERTFORMER_CHESS_*`, `MERTFORMER_ONEFILE_*`, `*_DEMO_*`, signing/upload,
> kernel-tuning/benchmark-harness internals such as `MERTFORMER_CANON_COMPARE_*`
> / `MERTFORMER_FUSED_BLOCK_*`) are intentionally omitted — they do not affect
> the 45K run. Find them with
> `grep -rhoE '(TITAN|MERTFORMER)_[A-Z_]+' --include='*.py' .`.
>
> **2026-07-27 refresh:** this file was last content-updated 2026-06-28 and had
> drifted a full month behind the 2026-07-08 through 2026-07-25 stabilization
> work (LR/warmup sweep knobs, the divergence guard, re-warmup, off-site
> backup, the pre45k gate) — none of those knobs were listed. Re-synced against
> a live `grep` of every `os.environ.get`/`os.getenv` call in the repo; see
> `BACKLOG.md`/`DECISIONS.md` 2026-07-27 entries.

## Run schedule / batch
| Var | Default | Effect |
| --- | --- | --- |
| `TITAN_MAX_STEPS` | `45000` | Total optimizer steps (the 45K run). |
| `TITAN_BATCH_SIZE` | `128` | Global batch (→ ~23.6B tokens @ 4096). `micro_batch`/`grad_accum` auto-derived. |
| `TITAN_BATCH_SIZE_FALLBACKS` | _unset_ | Orchestrator OOM-only retry ladder, e.g. `1024,512,256`. |
| `TITAN_VAL_CHECK_INTERVAL` | `1000` | Validation cadence (steps). |
| `TITAN_VAL_STEPS` | `10` | Validation micro-batches per check. |
| `TITAN_SAVE_INTERVAL` | `1000` | Checkpoint cadence (steps). |
| `TITAN_LOG_INTERVAL` | `10` | Metric-log cadence (steps). |
| `TITAN_TELEMETRY_INTERVAL` | `100` | Host/GPU snapshot throttle (subprocess cost). |
| `TITAN_DATALOADER_PIN` | `1` | `pin_memory` on the DataLoader. |
| `TITAN_DATALOADER_NONBLOCKING` | `1` | Non-blocking host→device transfer. |

## Token budget
| Var | Default | Effect |
| --- | --- | --- |
| `TITAN_TARGET_TOKENS_MIN` | `23_600_000_000` | Token floor; basis for the overshoot guard. |
| `TITAN_TOKEN_BUDGET_MODE` | `fixed_steps` | `fixed_steps` or `open_ended` (saturation gate). |
| `TITAN_STRICT_TOKEN_BUDGET` | `0` | `1` → hard-fail on >5% planned-token overshoot (launch checklist). |

## LR / warmup / stability guards (pre-45K stabilization pass, 2026-07-08 onward)
| Var | Default | Effect |
| --- | --- | --- |
| `TITAN_LEARNING_RATE` | `3e-4` | Peak LR. Sweep-start value, not verified-safe — `1.5e-3` was empirically fatal at this scale (real 36M/171M divergences, `BACKLOG.md`). |
| `TITAN_ROUTER_LR_MULT` | `1.0` | Router LR differential multiplier (old ad-hoc value was `1.5`). |
| `TITAN_WARMUP_RATIO` | `0.15` | Warmup fraction of `max_steps`, used when `TITAN_WARMUP_STEPS=0`. |
| `TITAN_WARMUP_STEPS` | `0` | Explicit warmup-step override (`0` = derive from `warmup_ratio`). |
| `TITAN_LIQUID_WARMUP_STEPS` | `10000` | Steps before the Liquid/CfC mixer unfreezes. |
| `TITAN_DIVERGENCE_GUARD` | `1` | Loss/grad-norm divergence circuit breaker (the grad-norm co-trigger is not separately gated). `0` disables both. |
| `TITAN_DETERMINISTIC` | `0` | `1` = genuinely enable deterministic training (was a dead flag before the 2026-07-11 fix). |

## Post-45K continuation (re-warmup, default OFF)
| Var | Default | Effect |
| --- | --- | --- |
| `TITAN_USE_REWARMUP` | `0` | `1` = use `get_rewarmup_schedule()` instead of the base WSD schedule on resume. Deliberate post-45K-only opt-in. |
| `TITAN_REWARMUP_STEPS` | `1000` | Ramp length back to peak LR. |
| `TITAN_REWARMUP_START_RATIO` | `0.01` | LR floor ratio the re-warmup ramps from. |

## Off-site checkpoint backup (`scripts/offsite_backup_watcher.py`, default OFF)
| Var | Default | Effect |
| --- | --- | --- |
| `TITAN_OFFSITE_BACKUP_DEST` | _unset_ | `s3://...` / `gs://...` / remote-rsync target. Unset = watcher no-ops. |
| `TITAN_OFFSITE_BACKUP_SRC` | `cfg.save_dir` | Source checkpoint directory to sync. |
| `TITAN_OFFSITE_BACKUP_INTERVAL_SECONDS` | watcher default | Poll cadence. |
| `TITAN_OFFSITE_BACKUP_STABILITY_SECONDS` | watcher default | Skip a cycle if the newest file was touched more recently than this (partial-write guard). |

## Teacher / distillation
| Var | Default | Effect |
| --- | --- | --- |
| `TITAN_TEACHER_MODEL_ID` | `meta-llama/Llama-3.3-70B-Instruct` | Teacher surface. |
| `TITAN_DISTILL_ALPHA` | `0.8` | KD weight. **`0` = teacher-free smoke** (70B never downloaded; pure CE). |
| `TITAN_REQUIRE_GATED_TEACHER` | `1` | Hard teacher policy (no teacherless fallback) on the canonical lane. |
| `TITAN_USE_PRECOMPUTED_LOGITS` | `1` | Offline KD from precomputed Top-K shards (no teacher VRAM). |
| `TITAN_LOGITS_PATH` | `./datasets/logits/` | Precomputed shard dir. |
| `TITAN_ALLOW_DENSE_PRECOMPUTE` | `0` | Debug-only dense precompute (canonical is sparse Top-K). |
| `TITAN_ALLOW_DENSE_TOPK_RECONSTRUCT` | `0` | Allow large dense reconstruction from Top-K (debug). |
| `TITAN_SKIP_DISK_GATE` | `0` | Bypass Phase-0 disk pre-flight (not recommended). |
| `TITAN_TOPK_DENSE_MAX_ELEMENTS` | `8000000` | Safety ceiling before a dense Top-K reconstruction is refused. |

## Phase-0 orchestration (`zero_touch_start.sh` auto-invocation)
| Var | Default | Effect |
| --- | --- | --- |
| `TITAN_SKIP_PHASE0` | `0` | Skip optional Top-K teacher-logit precompute (also auto-skipped for `--plan-only`/`--dry-run`/`--check-only`). |
| `TITAN_TOP_K` | script default `32` | Forwarded as `--top-k` to the precompute orchestrator when `zero_touch_start.sh` auto-invokes it. |
| `TITAN_PRECOMPUTE_BATCH` | script default `4` | Forwarded as `--batch-size`. Direct script invocation (`python scripts/precompute_logits_topk.py --batch-size N`) still takes the CLI flag; this env var only applies to `zero_touch_start.sh`'s own auto-invocation path. |
| `TITAN_PRECOMPUTE_GPUS` | _unset_ | If set, routes Phase-0 through the parallel multi-GPU orchestrator (`precompute_logits_parallel.py --gpus N`) instead of the single-process script. |

## Tokenizer
| Var | Default | Effect |
| --- | --- | --- |
| `TITAN_USE_TR_TOKENIZER` | `0` | `1` = local Turkish tokenizer everywhere (incompatible with teacher Top-K KD). |
| `TITAN_TR_TOKENIZER_ID` | `tokenizer/tr` | TR tokenizer path. |
| `TITAN_LOCAL_TOKENIZER_PATH` | _unset_ | Explicit local tokenizer artifact dir. |
| `TITAN_TEACHER_TOKENIZER_PATH` | _unset_ | Offline snapshot of the **teacher** tokenizer (same family only). |

## Data / dedup / packing
| Var | Default | Effect |
| --- | --- | --- |
| `TITAN_OFFLINE` | `1` | Offline-first (no external download/login). |
| `TITAN_SEQUENCE_PACKING` | `1` | EOS-separated packing (teacher+student share `train/packing.py`). |
| `TITAN_VERIFY_LOGIT_ALIGNMENT` | `1` | Hard per-sequence identity check (no silent KD realign). |
| `TITAN_ALLOW_LEGACY_LOGIT_REALIGN` | `0` | Escape hatch to the non-packed legacy path. |
| `TITAN_DEDUP_ENABLED` | `1` | Rolling blake2b dedup. |
| `TITAN_DEDUP_SCOPE` | `global` | `global` / `stage`. |
| `TITAN_DEDUP_HASH_BYTES` | `8` | Fingerprint width. |
| `TITAN_DEDUP_MAX` | `2000000` | Bounded dedup window. |
| `TITAN_DEDUP_NORMALIZE` | `1` | Normalize before hashing. |
| `TITAN_ALLOW_OPTIONAL_SOURCES` | `0` | Include optional (gated) dataset sources. |
| `TITAN_TOKEN_PROBE_SAMPLES` | `64` | Token-probe sample count. |

## Runtime fast-paths / kernels (opt-in, default-off behavior preserved)
| Var | Default | Effect |
| --- | --- | --- |
| `TITAN_MOE_DISPATCH` | `parallel` | MoE dispatch (`parallel` / `sequential`). |
| `TITAN_LIQUID_FAST_PATH` | `1` | Liquid `torch.compile`-guarded fast path. |
| `TITAN_LIQUID_TRAIN_IMPL` | `baseline` | `baseline`/`precompute_input`/`packed_pair`/`packed_pair_compile`. |
| `TITAN_FLASH_ATTN_INFER` | `0` | Flash-Attention at inference. |
| `TITAN_FFN_PACK` / `TITAN_MOE_PACK` / `TITAN_MLA_KV_PACK` | `0` | Lossless packed gate+up / K+V projections. |
| `MERTFORMER_LOWBIT_KERNEL` | `0` | Opt-in low-bit kernel dispatch. |
| `MERTFORMER_FUSED_KERNEL` | `1` | Fused triton ternary path (when kernel enabled). |
| `MERTFORMER_TENSORCORE` | `0` | Tensor-core path in the triton kernel. |
| `MERTFORMER_KERNEL_STRICT` | `0` | Raise instead of silent torch fallback when a backend is unavailable. |
| `MERTFORMER_KERNEL_BACKEND` | _auto_ | Force a specific kernel backend. |
| `MERTFORMER_ENABLE_TORCHSCRIPT_COMPAT` | `0` | Enable TorchScript on the liquid kernel. |
| `MERTFORMER_FUSED_BACKWARD` | `1` | Use the fused Triton backward for BitLinear; `0` = eager-fallback backward. |

## Pre-45K gate / preflight (`scripts/titan_preflight.py`, `scripts/pre45k_gate.py`)
| Var | Default | Effect |
| --- | --- | --- |
| `TITAN_PREFLIGHT_MIN_DISK_GB` | `100` | Minimum free disk required to pass the offline disk gate. |
| `TITAN_PREFLIGHT_REQUIRE_STAGE_JSONL` | `0` | `1` = hard-fail (not just warn) if a curriculum stage JSONL is missing. |
| `TITAN_PREFLIGHT_REQUIRE_SECRETS` | `0` | `1` = hard-fail if `HF_TOKEN`/secrets are absent (default is soft-warn). |
| `TITAN_PREFLIGHT_STRICT_CUDA_LOCK` | profile-dependent (`1` default profile / `0` legacy profile) | Require a written `cuda.lock` before proceeding. |
| `MERTFORMER_DDP_SMOKE_SECONDS` | `240` | Budget for the real 2-GPU DDP smoke test inside `pre45k_gate.sh`. |

## Resume / checkpoint
| Var | Default | Effect |
| --- | --- | --- |
| `TITAN_AUTO_RESUME` | `1` | Auto-discover the latest checkpoint. |
| `TITAN_RESUME_FROM` | _unset_ | Explicit checkpoint path. |
| `TITAN_RESUME_ALLOW_PARTIAL` | `0` | Allow partial (key-mismatch) resume — exploratory only. |
| `TITAN_POST_CKPT` | _unset_ | Explicit checkpoint path for post-train autorun (falls back to `BENCHMARK_CKPT`, then auto-discovery). |

## Logging / claim gate
| Var | Default | Effect |
| --- | --- | --- |
| `MERTFORMER_LOGBOOK` | `1` | Append to the unified hash-chained logbook. |
| `TITAN_CLAIM_MODE` | `0` | Claim-grade run (requires validation ≥ 1000 samples). |
| `TITAN_CONFIG_VERBOSE` | `0` | Config module console output. |

## Config overlays (YAML)
| Var | Effect |
| --- | --- |
| `MERTFORMER_CONFIG` / `MERTFORMER_MODEL_CONFIG` / `MERTFORMER_TRAIN_CONFIG` / `MERTFORMER_EXPORT_CONFIG` | Optional `config/*.yaml` overlays merged over defaults. |

## Orchestration / credentials / protocol
| Var | Default | Effect |
| --- | --- | --- |
| `HF_TOKEN` | _required online_ | Gated teacher/dataset access. |
| `WANDB_API_KEY` / `TITAN_WANDB` | _unset_ / `1` | Optional Weights & Biases. |
| `CUDA_VISIBLE_DEVICES` | _all_ | GPU selection (orchestrator derives `--num_processes`). |
| `ACCELERATE_CONFIG_FILE` | _unset_ | Custom Accelerate config. |
| `TITAN_PYTHON` | _venv/auto_ | Python interpreter for orchestrator child commands. |
| `TITAN_ALLOW_EXPERIMENTAL_TPU` | `0` | Gate for the unsupported `train_tpu_turbo.py` lane. |
| `TITAN_PROFILE` | `stable` | `stable` / `max_arch` / `offline_4060_demo` runtime profile selector (`run.sh`). |
| `TITAN_INSTALL` | `1` | `run.sh` dependency-install step toggle. |
| `TITAN_FORCE_ACCELERATE_RECONF` | `0` | `run.sh`-only: force-regenerate `~/.cache/huggingface/accelerate/default_config.yaml` instead of erroring on a mismatch (see `TROUBLESHOOTING.md` #3). |
| `SOP_AUTO_COMMIT` / `SOP_AUTO_PUSH` | `1` | Auto commit/push **inside** an already-human-invoked closure SOP script (`one_command_full_sop.sh`/`final_one_shot.sh`); set `0` for a dry SOP. This does not relax the Master Protocol's own per-request commit/push authorization rule for an AI agent — running the SOP script is itself the human authorization event for its internal commit/push steps. |
