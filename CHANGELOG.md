# Changelog

All notable changes to this project are tracked in this file.

> **Maintenance note (added 2026-07-25):** this file is hand-maintained, not auto-regenerated — it drifted a full month (2026-06-28 → 2026-07-25) before this note existed. Any closure pass that lands a real `BACKLOG.md`/`DECISIONS.md` entry should also add/update the current `## Unreleased - <date>` section here (EN) and in `CHANGELOG_TR.md` (TR) — a short summary is enough, full detail stays in `BACKLOG.md`/`DECISIONS.md`. See `reports/change_control_sop.md`.
> Entries are kept in strict reverse-chronological order (newest first); a 2026-07-27 pass found "Pass 7 (2026-06-13)" mis-filed after the 2026-03-13/2026-02-08 tagged releases and moved it back to its correct chronological slot — see that entry below for detail.
> **Coverage note:** this file's earliest entry is `v0.1.0-pilot-ready` (2026-02-08) — that is where changelog discipline for this project began, not where the project itself began. Earlier work is not summarized here; `git log` is the source of truth for anything before that date.

## Unreleased - 2026-08-06

### Added
- `evidence/2026-08-02-chess-searchless-5070/`: real, checkpoint-bound training + retroactive
  holdout/puzzle/elo evaluation results for `ChessFormerAI/chessformer` (independent side
  project, not part of this repo's own canonical model). Puzzle accuracy 45.78% (directly
  comparable to DeepMind's Searchless Chess, arXiv:2402.04494), Elo 1509 (internal Stockfish
  scale, not comparable to DeepMind's 2895). Does not close this repo's own 45K gap. Fold-back-
  or-deprecate decision on `scripts/chess_5080_onefile.py` made: stays separate. Full detail:
  `BACKLOG.md`, `DECISIONS.md`.
- `evidence/2026-08-02-chess-searchless-5070/retroactive_eval.py`: the script itself that
  produced the eval reports above, added to the evidence package verbatim for full
  reproducibility.

## Unreleased - 2026-08-01

### Added
- `README.md`/`README_TR.md` Quickstart: added a line pointing to the existing `scripts/train_smoke.py` sanity loop (already documented in `scripts/README.md`, previously not surfaced from the README) — a "does it even train" smoke check for first-time visitors, no new code. `ARCHITECTURE.md`: added a compact ASCII per-layer flow diagram beneath the existing "At a glance" table, complementing it. Note: `README_CHECKLIST.md` (superseded, historical, not a canonical surface) documents that ASCII/Mermaid diagrams were deliberately removed from `README.md` itself in the 178KB→4KB cut; this diagram goes into `ARCHITECTURE.md` instead, which isn't under that size constraint, so the earlier decision stands unchanged. `ARCHITECTURE.md` remains intentionally EN-only (pre-existing, matches the `TECHNICAL_REPORT_TR.md`/`MODEL_CARD_TR.md` sibling pattern in `README_TR.md`'s own Canonical surfaces list — not every canonical doc has a TR twin). No claim/code change.

### Fixed
- `tests/test_kaggle_onefile_config.py`: three tests (`test_run_config_schema_v2_defaults_ok`, `test_run_config_unknown_key_rejected_in_strict_mode`, `test_required_core_keys_present`) called `resolve_runtime_config()` unscoped, so its Colab-mirroring default `out_dir` fell back to a real `~/Downloads/content/mertformer_outputs` and got `mkdir()`'d as a side effect on every pytest run, on any OS — the same bug class already fixed once for a different test in `tests/test_kaggle_onefile_colab_math_fastproof.py`, independently uncovered here since that earlier fix only touched the one test it landed in. Fixed with the same proven pattern: `tmp_path`-scoped `out_dir`/`artifact_root` passed into the config before calling `resolve_runtime_config()`. Verified: stray directory deleted and confirmed absent after a full suite run (`721 passed, 9 skipped, 1 xfailed` — identical count, no regression), plus standalone `ruff`/`bandit`/`interrogate` (all clean; `final_one_shot.sh` itself not required for a test-file-only change). Full detail: `BACKLOG.md`.
- `scripts/check_overlay_validity.py::check_overlay()`: removed a dead, unused `env` dict (hardcoded Unix-only `PATH: "/usr/bin:/bin:/usr/local/bin"`) that was assigned but never passed to `subprocess.run()` (the call already uses `full_env` instead) — confirmed via `ruff --select F841` (repo's default lint scope, `pyproject.toml`'s `select = ["E9", "F821", "F822", "F823"]`, deliberately excludes this class to avoid low-signal churn across legacy scripts; this one instance was fixed directly since it was already hand-identified). `apps/chess_gui/play_mertformer_chess_web.py`: `--device` CLI flag's `argparse` `choices` was missing `"cuda"` (only `["cpu", "mps"]`) — the auto-detect path (`choose_device()`) already checks CUDA first, but a user could not force it explicitly. A repo-wide `ruff`/`bandit` discovery pass (130 unused-import/variable findings, 543 low/medium `bandit` findings) was run but deliberately not acted on beyond the one instance above — both fall inside the project's own documented no-broad-cleanup lint policy and outside this pass's `layers/`/`model/`/`train/`/`orchestrator/`/`mertformer_sdk/` no-touch boundary. Verified: `tests/test_chess_gui_contract.py` + `tests/test_config_overlay_strict.py` (10 passed), `scripts/check_overlay_validity.py` run directly (4/4 overlays OK), full suite `721 passed, 9 skipped, 1 xfailed` (identical count, no regression), doc-claim consistency OK (after reverting a locally-regenerated, never-committed `reports/train_readiness_decision.{json,md}` back to canonical — the known machine-local-state artifact this file's own maintenance history has hit before, not a real doc bug).

## Unreleased - 2026-07-31

### Added
- Documented a 2026-07-31 external signal on Liquid/CfC wall-clock cost (independent, small-scale test of `layers/liquid.py`/`layers/mla.py`, different hardware, ~9.4x `LiquidMixer`-vs-`GQA` at component level, plus a not-previously-written-down mechanism: recurrence cost scales with `seq_len`, attention's doesn't) across `BACKLOG.md`/`BACKLOG_TR.md`, `ABLATION.md`/`ABLATION_TR.md`, `WHITE_PAPER_LIQUIDROUTER.md`, `reports/liquid_keep_or_drop_brief.md`, `reports/blog_liquid_ablation_draft.md`, `reports/paper_outline_draft.md`, `reports/publication_readiness_kit.md`, and `ablations/no_liquid/README.md`/`README_TR.md`. Informational only — does not reopen the 2026-07-19 `DECIDED: Keep` call, does not change any measured claim, no code touched. Full detail: `BACKLOG.md`.
- Added a symmetric inference-side counterpoint to the entry above, same `BACKLOG.md`/`BACKLOG_TR.md`/`ABLATION.md`/`ABLATION_TR.md` entry: `generate()`'s stateful `LiquidMixer` decode path (`h_init`/`return_state`) is already implemented and correctness-tested (`<1e-8` parity vs. full-forward, `tests/test_liquid_generate_parity.py`, fixed 2026-07-08), and architecturally implies a per-token decode cost independent of context length unlike growing-KV-cache attention — but decode-mode speed has never been benchmarked. Folded into the same pre-45K validation item as the training-side finding; no new claim, no code touched.
- Added new `scripts/liquid_vs_gqa_canonical_benchmark.py` (component-level `GQA`-vs-`LiquidMixer` train-mode + decode-mode micro-benchmark, same conventions as `scripts/liquid_train_impl_benchmark.py`, CUDA/MPS/CPU) and actually ran both scripts at canonical `hidden_size=2048` on this machine's own RTX 4060 (8GB VRAM) — the pre-45K validation item from the two entries above, measured rather than planned. Results: train-mode (`seq_len=512`, `batch_size=1`, `--fast-path 0` — no working Triton install on this Windows box) measured `LiquidMixer` ~797-1620x slower than `GQA` (`packed_pair` slower than `baseline`), and `seq_len=2048`/`4096` OOM outright on this GPU even at `batch_size=1` (a memory-scaling cost, not just wall-clock, not previously written down). Decode-mode measured `LiquidMixer`'s per-token cost flat across a 24x context-length sweep and 8-23x faster than `GQA`, confirming the stateful-decode hypothesis from the entry above. Full numbers and hedges: `BACKLOG.md`/`BACKLOG_TR.md` (same entry), `ABLATION.md`/`ABLATION_TR.md` addenda. Raw JSON: `reports/benchmarks/liquid_train_impl_canonical_512.json`, `reports/benchmarks/liquid_vs_gqa_canonical_decode.json`. Still consumer-GPU/single-run/not-canonical-scale evidence; does not reopen `DECIDED: Keep`. No model/training code touched.

### Fixed
- `layers/moe.py::MoE._dispatch_parallel()`: `torch.arange(N, device=topk_idx.device).repeat_interleave(k)` crashed under `torch.onnx.export`'s tracer with a cuda/cpu device mismatch (`repeat_interleave`'s ONNX symbolic export doesn't propagate the source device) — first surfaced running `tests/test_comprehensive.py::TestONNXCycle::test_onnx_export_import` for real on this machine's CUDA GPU (previously always `skipif`-gated out on CPU-only/Mac machines). Fixed with the device-safe equivalent `torch.arange(N, device=topk_idx.device).unsqueeze(1).expand(N, k).reshape(-1)` — identical output. `tests/test_moe_dispatch_parallel_counts.py` (12 tests) unaffected. Also marked the same test `xfail` (not fixed, dated reason, Mert's call) for a second, deeper ONNX-export incompatibility in `LiquidMixer`'s in-place eval-cache buffer mutation (`layers/liquid.py::_set_cache`) — the same cache `generate()`'s correctness-tested decode path relies on, a real architecture question, not a same-night fix. Full detail: `BACKLOG.md`.
- `tests/test_pre45k_gate.py::test_offline_preflight_reports_the_missing_corpus_rather_than_passing`: inherited ambient `GITHUB_ACTIONS`/`CI`/`TITAN_PREFLIGHT_ALLOW_MISSING_STAGE_JSONL` env vars into its own subprocess call, passing on a bare local checkout but failing under real GitHub Actions CI (which runs via `bash scripts/verify_all.sh`). Now neutralizes all three via `monkeypatch.delenv(...)` before invoking the gate; both production escape hatches (`scripts/titan_preflight.py`, `scripts/verify_all.sh`) are unchanged. Full detail: `BACKLOG.md`.
- Windows portability pass: `bash scripts/final_one_shot.sh` had never been run on Windows before and failed repeatedly, each failure a genuine, reproducible cross-platform bug. Root-caused and fixed by category: `python3`-hardcoding in 9 test files, `.titan-venv` Windows venv-layout detection, `Path.relative_to()` backslash-vs-forward-slash key mismatches in manifest/sync scripts, missing `encoding="utf-8"` on 39 files' `subprocess.run(text=True)` calls plus 3 files needing `sys.stdout`/`sys.stderr` UTF-8 `.reconfigure()` (fixes a `wandb`-console-wrapper crash in `train/train.py`), an `os.isatty()` test assumption, Mac-only hardcoded `/Applications` paths gated behind `sys.platform == "darwin"`, a non-portable `mktemp -t` template, two unconditionally-fatal `dealroom_sync` calls made best-effort (matching the repo's own existing pattern for sibling-repo-optional steps), a `start_gate`/`check_doc_claim_consistency` redundant-re-verification-after-self-mutation ordering bug, 3 scripts passing a relative venv interpreter path into `subprocess.Popen()`/`subprocess.run()` argv (fixed to `sys.executable` / an absolute path), a missing `ruff` dev dependency (installed at the pinned version), and CUDA never being checked (only MPS-or-CPU) in 4 device-selection call sites (`scripts/train_smoke.py`, `tests/test_moe_dispatch_parallel_counts.py`, `tests/test_architecture_integrity.py`, `apps/chess_gui/play_mertformer_chess_web.py`) — all four now check CUDA first, then MPS, then CPU, matching `config/config.py`'s existing default and staying behavior-neutral on both this machine and Mac. Full root-cause detail per category: `BACKLOG.md`.
- `tests/test_kaggle_onefile_colab_math_fastproof.py::test_logging_artifacts_written_and_compare_schema` created a real, empty `~/Downloads/content/mertformer_outputs/` directory as a side effect on every pytest run (any OS) via `resolve_runtime_config()`'s Colab-path-mirroring default resolving before the test's own override took effect; fixed by pre-scoping `out_dir`/`artifact_root` into the config dict passed *into* the call.
- `scripts/final_one_shot.sh`'s `chess_5080_share_export` step rebuilt a uniquely-timestamped Desktop delivery zip on every ladder pass with no cleanup of prior bundles, silently accumulating duplicates; made opt-in via `TITAN_CHESS_5080_EXPORT=1` (default: skipped). Platform-neutral behavior change, not Windows-specific.
- `BACKLOG_TR.md`'s own Windows-portability entry translated the literal pytest count into Turkish in 3 places, breaking this file's own convention of quoting tool output verbatim; fixed.
- The Windows-portability commit's `git add -A` also swept up a regenerated `reports/train_readiness_decision.json`/`.md` reflecting this machine's own zero-corpus state (`STAGE_JSONL_MISSING`) rather than the documented canonical one (`PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE`), which broke GitHub Actions' "Documentation claim consistency gate" — `scripts/verify_all.sh` checks docs against whatever readiness snapshot is already committed, before its own refresh step runs. Reverted both files to their pre-commit content and regenerated the dependent `reports/target_machine_handoff_manifest.json`/`.md` for hash consistency. Not a code/logic bug; full detail in `BACKLOG.md`.
- Author-name spelling regression from the 2026-07-29 relicensing pass: the new `LICENSE` copyright line (and everything copied from it — `NOTICE`, `LICENSE_TR`, `README.md`/`README_TR.md`, and 51 source-file license headers) used the ASCII transliteration `Mert Yunlu`, silently reverting an earlier, deliberate decision (documented elsewhere in `DECISIONS.md`) to standardize on the Turkish spelling `Mert Yünlü`. Fixed in all 56 live occurrences; the two dated `DECISIONS.md`/`DECISIONS_TR.md` quotes of the 2026-07-29 `LICENSE` text are deliberately left as accurate historical records. Comment/text-only change, zero runtime effect.

### Validation
- Test count unchanged (`726 passed, 5 skipped` locally, per the prior entry) — this fixes an existing test's environment isolation, it adds/removes nothing. Confirmed: simulating CI ambient pollution (`GITHUB_ACTIONS=true CI=true TITAN_PREFLIGHT_ALLOW_MISSING_STAGE_JSONL=1`) reproduces the failure pre-fix and passes post-fix.
- `bash scripts/final_one_shot.sh` now completes clean end-to-end on Windows/Git-Bash: `[final] COMPLETED`, exit 0. This machine's honest measurement is `721 passed, 10 skipped`, audited skip-by-skip against the prior Mac-measured `726 passed, 5 skipped`: 2 of the 10 are genuinely platform-specific (MPS-absence skips in `test_architecture_integrity.py`/`test_comprehensive.py`, would pass on Mac), the other 8 match this repo's own already-documented fresh-clone-skip precedent (`c2c212e2`) — missing local tokenizer artifacts and the pre-existing CUDA/MPS/QINN/corpus environment skips. Deliberately not reverted to `726/5`; that would misrepresent this machine's real measurement. `md_integrity_check.py` (373 files, 0 findings) and `secret_scan.py` (953 files, clean) both pass; 5 auto-generated report files found to leak the local Windows username/absolute paths through incomplete `sanitize_path()` coverage were excluded from commit (regenerable artifacts, not source).

## Unreleased - 2026-07-30

### Added
- `CODE_OF_CONDUCT.md`/`_TR` (Contributor Covenant 2.1, plus a project-specific measured/target/vision claim-discipline clause) — closes the last gap in GitHub's community-standards checklist now that `README.md` invites external contributions.

### Fixed
- Audit waves 1-5 (independent 2026-07-27 static audit, 20+ real findings): MoE capacity host-syncs removed, `train/packing.py` resume-counter desync fixed, teacher-logit identity sidecar added, param estimators reproduce the measured count exactly, feature-flagged drift detectors made reachable, 2 dead scripts deleted, plus structure-check/config-validator/alias-guard/PoC-hashing fixes and the 45K dashboard wiring. Full pass-by-pass detail: `BACKLOG.md`.
- License-header contradictions: 50 `*.py` files and `run.sh` still carried an All-Rights-Reserved header after the Apache 2.0 relicensing; fixed, verified with a full-repo grep across every tracked file type.
- `NOTICE`: two sentences left factually wrong by the relicensing (stale "proprietary" language, stale team name) corrected; the Llama attribution layer is untouched.
- `SECURITY.md`/`_TR`: explicit contact-email fallback added alongside the primary GitHub Security Advisories channel.
- `tests/test_pre45k_gate.py::test_run_offline_preflight_against_real_repo`: depended on the gitignored training corpus and failed on every fresh clone/CI runner; now skips without the corpus, with a new sibling test pinning the honest-failure direction (see the 2026-07-31 entry above for that sibling test's own follow-up fix).

### Changed
- Relicensed the code under Apache 2.0 for public release; added Hiring and Contribution sections to `README.md`/`README_TR.md`. See `DECISIONS.md`.

### Validation
- `726 passed, 5 skipped`. `bash scripts/final_one_shot.sh` green (see `BACKLOG.md` "Public-release closure"). No training-math, readiness, or claim-boundary change.

## Unreleased - 2026-07-27

### Fixed
- `scripts/scaling_audit_math.py` + `config/config.py::_estimate_total_params()`: both independent analytical param-count estimators reused the dense-FFN `intermediate_size` for MoE experts instead of the real, larger `moe_intermediate`, and both omitted `layers/moe.py`'s always-active "shared expert" entirely — undercounting active params by ~44% and total params by ~8%. Fixed to match the real architecture; `estimate_params()` now reports ~3.698B total / ~1.886B active, matching `ARCHITECTURE.md`'s independently-stated ~1.86B active figure. 4 new regression tests (`tests/test_scaling_audit_math.py`, `tests/test_config_dynamic_param_count.py`).
- `CHANGELOG.md`/`CHANGELOG_TR.md`: "Pass 7 (2026-06-13)" was filed after the 2026-03-13 and 2026-02-08 entries, breaking reverse-chronological order — moved to its correct slot (between 2026-06-17 and 2026-05-24).
- `ENV_VARS.md`: re-synced against a live `grep` of every `os.environ.get`/`os.getenv` call — the file had drifted a full month behind the 2026-07-08→07-25 stabilization work and was missing the entire LR/warmup-sweep family (`TITAN_LEARNING_RATE`, `TITAN_ROUTER_LR_MULT`, `TITAN_WARMUP_RATIO`, `TITAN_WARMUP_STEPS`, `TITAN_LIQUID_WARMUP_STEPS`), `TITAN_DIVERGENCE_GUARD`, the re-warmup family, the off-site backup family, the `TITAN_PREFLIGHT_*` family, `TITAN_DETERMINISTIC`, dataloader flags, `TITAN_PROFILE`/`TITAN_INSTALL`, and `MERTFORMER_DDP_SMOKE_SECONDS`/`MERTFORMER_FUSED_BACKWARD`.
- `V2_BACKLOG_SEED.md` Track F: 3 items (`liquid_warmup_steps` env override, z-loss double-multiply, `mark_weights_updated()` cache question) were still listed as open even though `BACKLOG.md`/`DECISIONS.md` already record them as resolved/investigated — annotated resolved with cross-references.
- `CHESS_5080_POC_INTERNAL.md` (EN) brought to content parity with `CHESS_5080_POC_INTERNAL_TR.md`, which had flagged the EN file as stale in its own text (Windows build workspace, EXE delivery, Stockfish auto-fetch cache, curated position suite, synthetic teaching corpus were TR-only).
- `TECHNICAL_REPORT.md`/`_TR`: masthead date was frozen at 2026-06-18 while the body already carried a 2026-07-19 revision (INT-KERNEL relabel) — added an explicit last-revision note rather than silently back-dating.
- `README.md`/`README_TR.md`: `AGENTS.md` (self-declared #1 in the repo's own source-of-truth order) had zero inbound links from any root doc; `START_HERE.md`/`README_SUMMARY.md` (the external-reviewer onboarding path) were never linked from `README.md` either. Added both to the Canonical surfaces list.

### Validation
- `626 passed, 5 skipped` (was `622 passed, 5 skipped` at the last entry — the +4 is this pass's own new regression tests). Docs-only changes otherwise (Class A per the Master Protocol); `bash scripts/verify_all.sh` re-run green. No training-math, readiness, or claim-boundary change.

## Unreleased - 2026-07-25

### Added
- `scripts/pre45k_gate.py`/`.sh` + `scripts/ddp_smoke.py`: chains the offline preflight, a dry-run preview, and a real 2-GPU DDP smoke test into one pre-spend launch-readiness gate; writes `reports/pre45k_gate_report.{json,md}`.
- `scripts/kaggle_batch_runner.py`: unattended multi-job Kaggle orchestrator; produced 4 real evidence sets under `evidence/2026-07-25-*` (Nutrition5k Liquid-OFF/MoE-OFF ablations, 36M/171M LM re-verification).
- `utils/divergence_guard.py` gained an independent grad-norm EMA co-trigger ("C1") alongside the existing loss-based brake — confirmed firing correctly on real 36M/171M hardware.
- `scripts/offsite_backup_watcher.py`, `runbooks/checkpoint_offsite_backup.md`, `train/trainer_core.py::get_rewarmup_schedule()` (post-45K LR re-warmup).
- `tests/test_atomic_write_hygiene.py`: atomic (temp+`os.replace`) writes for 5 pipeline files previously trusted via a bare `.exists()` check.
- `model/nutrition_vision.py` + `scripts/{train,predict,evaluate}_nutrition5k.py`: a bounded vision side-experiment reusing the real BitLinear/MoE/Liquid trunk unmodified; a real trained + independently-re-verified checkpoint, then a real comparative ablation (see Changed).

### Fixed
- z-loss effective weight: an accidental double-multiply left it ~500x below the Switch-Transformer/ST-MoE convention; `z_loss_coef` corrected `1e-4 → 0.05`.
- `generate()` never threaded the Liquid/CfC hidden state across decode steps — a silent no-op during generation; fixed, with a full-forward↔incremental-decode parity test.
- `bigcode/the-stack-dedup` revision/sha256 finally pinned (a dataset-ref scanner false-positive had blocked it for months).
- `scripts/kaggle_batch_runner.py::run_chess()` invocation bug (wrong `sys.path`) found live during a real Kaggle run and fixed.
- `layers/moe.py` MoE dispatch-parallel `torch.bincount` → `scatter_add_` (MPS/older-torch portability).

### Changed
- LR regime (`1.5e-3 → 3e-4`, sweep start not verified-safe), Liquid spike guard (absolute → EMA-relative), WSD scheduler clamp — all candidate fixes, applied and re-tested on real RTX-5070/Kaggle hardware but not yet proven sufficient (see Validation).
- Eight launch-time decisions locked (`DECISIONS.md`): lane = `online_teacher`, Liquid = Keep, model size = 3.67B canonical, `top_k` = 32 (not 256), 2 dead Stage-5 datasets replaced with a verified live one, 3 license-TBD datasets kept-and-documented, Stage-3 TR/synthetic ratio ratified, INT-KERNEL claim relabeled honest (fp-simulation, no real ternary kernel yet).
- Public gist reorganized: Nutrition5k promoted to the front, a real z-loss arithmetic error corrected (`~50x` → `~500x`), the one-pager's pitch/investor framing replaced with research framing.

### Validation
- `622 passed, 5 skipped` (was `370 passed, 4 skipped` at the last entry). Third real-hardware confirmation (2026-07-02 / 07-12 / 07-25) that this architecture still diverges at small scale without further LR/optimizer work — the new grad-norm safety brake (C1) is now confirmed catching it cleanly at two scales instead of an uncontrolled explosion. Readiness unchanged: `TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP`. No trained/benchmark claim — the one remaining gap is a real 45K GPU run. Full pass-by-pass detail: `BACKLOG.md`, `DECISIONS.md`.

## Unreleased - 2026-06-28

### Added
- `scripts/flip_status_banner.py`: **report-only** status-banner auditor — lists tracked files carrying the pre-training banner and reports post-flip eligibility (a real, non-zero eval metric). It has **no write path**: the actual evidence-gated flip is a deliberate post-run task (a naive "checkpoint+summary exists" gate is satisfied by a stray demo checkpoint + stub summary, so a pre-built auto-writer is unsafe). See BACKLOG.
- `ENV_VARS.md`: single index of the canonical training/precompute/orchestration environment variables with defaults.

### Fixed
- `eval/gsm8k.py`: checkpoint load now uses `weights_only=False` (+ `_orig_mod.` key normalization, non-strict load), mirroring the documented `train.py` resume path — prevents a torch≥2.6 `UnpicklingError` when evaluating a real training checkpoint (optimizer/GaLore state) in the post-45K GSM8K benchmark.

### Changed
- Banner/version hygiene: normalized non-frozen `Status` / `Version` / `__version__` banners to the canonical Build-30-V2 form (`utils/logger.py`, `orchestrator/*`, `scripts/*`). Comment/metadata only, zero runtime change. Frozen-path banners (`model/`, `train/`, `layers/`) are deliberately left for the post-45K evidence-gated flip.

### Validation
- `370 passed, 4 skipped` (offline-first pytest, unchanged); readiness `TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP`. No trained/benchmark claim — the one remaining gap is a real 45K GPU run.

## Unreleased - 2026-06-17

### Added
- WHITE_PAPER_LIQUIDROUTER (EN+TR): arXiv submission section (Section 8) with proposed title — gated post-45K.
- `TITAN_DISTILL_ALPHA` env knob enabling a teacher-free pre-45K smoke (the 70B teacher is never downloaded when set to 0).
- README_SUMMARY (EN+TR): "Architecture at a glance" component table (BitNet / GQA / sparse MoE / Liquid-CfC).
- Canonical root scaffold (STATUS / TRUTH_MATRIX / BACKLOG / GOVERNANCE / REPRODUCE, EN+TR) as the single reviewer entry point.

### Changed
- Attention class renamed MLA → GQA (always grouped-query; filename `layers/mla.py` kept for manifest/SHA stability).
- 45K-pre-run operational hardening: atomic checkpoint save (`os.replace`), removed the in-forward MoE collapse-flag `all_reduce`, telemetry buffers `persistent=False`, removed the permanent grad-clip ratchet (now transient), telemetry throttle.
- Liquid ablation canonicalized to the 12-seed verdict (OFF 96.32% / ON 94.69%, Δ−1.63 pp, p=0.305, inconclusive — no measured benefit, ~30% slower); the single-seed +0.50 pilot is superseded across ablation surfaces.
- TECHNICAL_REPORT (EN+TR) clinical rewrite; §3.1 "12x" relabeled Target/estimate; §7 SHA256 step-chaining relabeled "designed"; MoE expert intermediate corrected to 8192.
- README cut 178KB → ~4KB (full snapshot archived); single-persona technical/evidence surface; commercial/GTM material moved under `private/`.
- License surface resolved to Proprietary & Confidential across README / README_TR (matching LICENSE).

### Fixed
- `pyproject.toml`: added the missing `mertformer_sdk.kernels.cpp` package.
- `.pre-commit-config.yaml`: ruff pinned to v0.15.5 (matches constraints.txt).
- `Dockerfile`: now applies `constraints.txt` for reproducible builds.
- `registry/mertformer_v0.1.json`: version synced to Build 30 V2 (was v27.0).
- `scripts/secret_scan.py` + `policy/allow_deny_policy.yaml`: GitHub-token patterns extended to gho_/ghu_/ghs_/ghr_ and fine-grained PAT.

### Validation
- `370 passed, 4 skipped` (offline-first pytest); readiness `TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP`. No trained/benchmark claim — the one remaining gap is a real 45K GPU run.

## Pass 7 (2026-06-13) — Mac-doable backlog zeroed + $0 Kaggle pilot
- Added `scripts/run_liquid_ablation.py` + `docs/KAGGLE_PILOT.md`: a free LiquidRouter ON-vs-OFF
  ablation pilot (~80–100M, pure CE, no teacher) — the single domino that unlocks the GPU-gated work.
- LatentODE per-batch reset during training (no cross-batch state leak); MoE collapse flag DDP
  all-reduce (guarded, no-op off-DDP); liquid-impl benchmark script; coverage config.
- Docs: ARCHITECTURE.md Projections + stage-3 note; CPU quickstart. Backlog dispositions in DECISIONS.md.
- Invariants held: param count locked; pytest green; ruff + scoped mypy + verify_all green.

## Unreleased - 2026-05-24

### Added
- Optional, default-off packed projection controls for FFN, MoE BitSwiGLU, and MLA K/V training paths.
- Environment-overridable training controls for batch size, log interval, validation interval, checkpoint interval, and DataLoader transfer behavior.
- Optional 8-GPU Accelerate profile under `repro/accelerate_8xgpu.yaml`.
- Equivalence coverage for packed projection and Liquid train implementation variants.

### Changed
- README, usage guide, training plan, feature-flag governance, script catalog, and verification matrix now document the optional speed-control surface with explicit claim boundaries.
- Documentation clarifies that `repro/` holds reproducibility/run configs, while `configs/` remains reserved for stable named configuration contracts.

### Validation
- Optional speed flags remain off by default and require equivalence tests plus target-machine logs before any speed claim.

## v1.0.0-build30-v2 - 2026-03-13

### Added
- Cross-dataset deduplication path in data pipeline.
- MoE parallel dispatch mode and CfC fast path toggle.
- Onefile demo CLI enhancements + training log dashboard script.
- SOP tolerance check for CfC/MoE loss parity.

### Changed
- Build 30 V2 version sync across core docs and model metadata.
- Training token budget defaulted to fixed-steps gating.

### Validation
- SOP full run (verify_all, md_quality, linkcheck, sync_manifest) PASS.

## v0.1.0-pilot-ready - 2026-02-08

### Added
- Pilot report contract: `interfaces/pilot_report_v1.schema.json`.
- SDK pilot helpers and CLI commands:
  - `mertformer verify`
  - `mertformer pilot-report --out <json>`
- SITL proof flow for drone-class offline evidence:
  - `scripts/drone_sitl_demo.py`
  - `reports/drone_sitl_demo.md`
  - `reports/drone_sitl_demo_TR.md`
  - `reports/pilots/README.md`
  - `reports/pilots/README_TR.md`
- Pilot business docs:
  - `reports/pilot_readiness_kit.md` + `_TR`
  - `reports/pilot_offer_packages.md` + `_TR`
  - `reports/sales_funnel_90d.md` + `_TR`
- Clean-room verification report:
  - `reports/cleanroom_verification.md`
  - `reports/cleanroom_verification_TR.md`
- Pilot acceptance signature template:
  - `reports/pilot_acceptance_signoff.md`
  - `reports/pilot_acceptance_signoff_TR.md`

### Changed
- Benchmark claim safety gate is strict: missing checkpoint now reports `NOT ELIGIBLE FOR CLAIM`.
- README claim language tightened: measured vs target/estimate separated.
- Absolute Desktop paths removed from tracked artifacts.
- Turkish document parity and wording normalization improved across TR docs.

### Fixed
- Strict checkpoint guard in SDK load path to prevent accidental random-weight pilot runs.
- Docs index and project structure blocks synchronized with tracked files.

### Validation
- `python3 -m pytest -q` passed.
- `ruff check .` passed.
- `bash scripts/verify_all.sh` passed.
