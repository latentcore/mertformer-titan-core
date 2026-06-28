# Environment Variables — canonical training/run surface

Single index of the environment variables that govern the **canonical 45K
training / Phase-0 precompute / orchestration** path. Defaults are the in-code
defaults (`config/config.py`, `train/`, `scripts/`). Turkish notes inline.

> Scope: this lists the **training-relevant** knobs. Out-of-scope families
> (`MERTFORMER_CHESS_*`, `MERTFORMER_ONEFILE_*`, `*_DEMO_*`, signing/upload) are
> intentionally omitted — they do not affect the 45K run. Find them with
> `grep -rhoE '(TITAN|MERTFORMER)_[A-Z_]+' --include='*.py' .`.

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

## Token budget
| Var | Default | Effect |
| --- | --- | --- |
| `TITAN_TARGET_TOKENS_MIN` | `23_600_000_000` | Token floor; basis for the overshoot guard. |
| `TITAN_TOKEN_BUDGET_MODE` | `fixed_steps` | `fixed_steps` or `open_ended` (saturation gate). |
| `TITAN_STRICT_TOKEN_BUDGET` | `0` | `1` → hard-fail on >5% planned-token overshoot (launch checklist). |

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

## Resume / checkpoint
| Var | Default | Effect |
| --- | --- | --- |
| `TITAN_AUTO_RESUME` | `1` | Auto-discover the latest checkpoint. |
| `TITAN_RESUME_FROM` | _unset_ | Explicit checkpoint path. |
| `TITAN_RESUME_ALLOW_PARTIAL` | `0` | Allow partial (key-mismatch) resume — exploratory only. |

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
| `SOP_AUTO_COMMIT` / `SOP_AUTO_PUSH` | `1` | Closure-protocol auto commit/push toggles (set `0` for dry SOP). |

> **Precompute batch size** is a CLI flag, not an env var:
> `python scripts/precompute_logits_topk.py --batch-size 32` (default `4`;
> raise on a B300/H100 box to cut Phase-0 wall-clock). Same for
> `scripts/precompute_logits_parallel.py --gpus N`.
