# Pre-45K Gate Report

Generated: 2026-07-25T14:55:15.105669+00:00
Verdict: **PASS_DDP_NOT_APPLICABLE**
strict_ddp: False

| Step | OK |
|---|---|
| Offline preflight (`titan_preflight.py`) | True |
| Dry-run preview (`zero_touch_start.sh --dry-run`) | True |
| DDP smoke (2-GPU) | status=skipped_not_2_gpu ok=False skipped=True |

Claim boundary: the DDP smoke step only confirms genuine dual-GPU activity when exactly 2 GPUs are present and `accelerate`/CUDA are available. On a single-GPU or CPU machine it is a clean, non-blocking skip (`PASS_DDP_NOT_APPLICABLE`) -- not a pass for DDP correctness itself.
