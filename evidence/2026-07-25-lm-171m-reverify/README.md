# 171M LM Re-Verify — 2026-07-25 (Kaggle T4×2, single-GPU) — DIVERGED, C1 grad-norm guard fired correctly

**Status: `RE-VERIFICATION RUN, FAILED infra-verdict (5/9), genuinely informative` — same divergence class and same C1 safety-brake validation as `evidence/2026-07-25-lm-36m-reverify/`, at the larger 171M pilot scale. See that folder's README for the shared background (candidate fix set, C1 design purpose); this file covers what's specific to the 171M run.**

## What this is

BACKLOG's 171M LM pilot re-verification (`scripts/preflight_run_pilot171m.py`, zero-arg, 3000-step Go/No-Go plan), run as job `04_lm_171m_reverify` of the same 5-job Kaggle batch. Preceded by the batch's own 2-GPU DDP smoke test (a batch-level step, not part of this job's own `console.log` in this folder): the manifest (`MertFormer_Kaggle_Batch_Output_manifest.json`) records `"used_ddp": false` for this job, and the live Kaggle Logs tab (observed during the run, not captured in a file that made it into the downloaded output) showed `both_active_any_sample=False -> FALL BACK TO SINGLE-GPU` — the smoke test (fixed earlier the same day, the `ddp_smoke_test()` bug) correctly detected no genuine dual-GPU activity and fell back, so this job ran single-GPU, not DDP.

## 🟢 MEASURED — what happened (full detail in `console.log`, this folder)

- Preflight infra checks: all hard requirements PASS (same as 36M — Tesla T4, bf16, disk, RAM).
- Plan: 3,000 steps × 16,384 = 49.15M token positions.
- Training progressed further than 36M before instability: MoE `MaxLoad` crossed the 0.40 alarm threshold intermittently from step ~180 onward (0.40-0.48 range), less severe than 36M's 0.72-0.76 peak.
- **Step 450:** divergence guards arm — `ce_ema=11.1653`, **`grad_norm_ema=10,581,991,948.83`** (~10.58 **billion** — already far more elevated at arming time than 36M's ~6.7 million, indicating gradients were unstable well before this point).
- **Step 500:** a real validation checkpoint saved (`✅ New best validation loss! Val Loss: 11.8515`) — the run reached at least one genuine validation cycle before the brake fired, meaning some training signal was being processed, not an immediate blowup from step 0.
- **Step ~520-530:** gradient norms in the trillions (`GradNorm: 1,752,834,913,075.2` at step 520, `5,634,968,440,012.8` at step 530).
- **Step ~530:** grad_norm_ema reaches 1.7381e12. **`🛑 SAFETY BRAKE: relative grad-norm divergence — grad_norm_ema=1.7381e+12 exceeded 100.00x the warmup-end reference (10,581,991,948.83) for 10 consecutive optimizer steps.`** — C1 firing correctly a second time, at a different scale.
- Clean shutdown: `pilot171m_stabilization_step_532.pt` and `pilot171m_stabilization_final.pt` saved, `REPORT.md` written (7.9 KiB), output archive packaged and verified (2.3 GiB, 20 members) — **not included in the consolidated batch zip, not present on this laptop** (same collection-rule gap as 36M — see that folder's README for the full explanation).
- `preflight_run_pilot171m.py`'s own infra-verdict: **FAIL (5/9 infrastructure checks passed)** — reached further than 36M's 3/9 before the brake, consistent with surviving more steps (530/3000 vs 416/1830) before the same divergence class caught up with it.

## 🔴 Negative finding — same divergence class as 36M, at larger scale

The 171M pilot (full canonical architecture at 9 layers × 512 hidden, per its own Go/No-Go design purpose — "does the candidate LR regime hold on the full architecture at 512h/9L?") diverged under the same candidate LR/z-loss/EMA-threshold fix set, confirming this is not a 36M-scale-specific artifact. Reaching a real validation checkpoint at step 500 before the step-530 brake shows the model was learning something (val loss 11.8515, close to the `ln(vocab)=11.76`-ish uniform-baseline region expected this early) before the divergence overtook it — consistent with the established pattern (2026-07-02, 2026-07-12) of onset correlating with LR ramping into the low-1e-4/2e-4 range combined with MoE imbalance, not an instant failure.

## 🟢 Positive finding — C1 validated a second time, at a different scale and reference magnitude

The reference grad_norm_ema at arming (10.58 billion) was roughly 1,580× larger than 36M's reference (6.7 million) — the guard's *relative* (100×-over-reference) design meant it still worked correctly despite this large difference in absolute scale, which is exactly the property a relative (not absolute) threshold is supposed to provide. Two real-hardware firings in one session, at two different model scales and two very different absolute gradient magnitudes, both producing a clean checkpoint instead of an uncontrolled explosion, is meaningfully stronger evidence for C1 than either firing alone.

## Checkpoints — NOT retrievable from this laptop (same gap as 36M)

`pilot171m_stabilization_step_532.pt` and `pilot171m_stabilization_final.pt` were saved during the run (see `console.log` lines ~456-457) and packaged into `pilot171m_run_output.zip` (2.3 GiB) inside the job's own Kaggle working directory, but excluded from the consolidated batch zip by the same `status == "completed"`-only collection rule described in `evidence/2026-07-25-lm-36m-reverify/README.md`. Not present anywhere on this laptop; not hashed; may still exist in the Kaggle notebook's own Output if not yet cleaned up (not checked in this pass).

## Boundary (does NOT prove)

Not a claim that the 45K canonical run is safe or ready — this is the second and larger-scale real-hardware confirmation that the candidate fixes alone are insufficient. Not a claim about C1's behavior at full 45K duration/scale (seen working correctly at 36M and 171M scale, single-GPU, T4; not yet seen at multi-thousand-step duration or multi-GPU DDP). Does not change 45K readiness (`TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP / START_ALLOWED`) directly, but is directly relevant evidence for the launch-time judgment calls in `reports/launch_time_decisions_checklist.md`.

## Reproduce

`python scripts/preflight_run_pilot171m.py` (zero-arg) on a CUDA machine with `TITAN_LEARNING_RATE=3e-4` (current default).
