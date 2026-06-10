# Train/Eval Tokenizer Parity Fix

Status: `DONE_NOW` (scoped bug fix, covered by tests)
Scope: close the train/eval tokenizer-family mismatch only. No core, teacher,
dataset, or policy changes beyond making the tokenizer single-sourced.

## Bug (measured, from code)

- Train resolved the tokenizer via `load_teacher_tokenizer(prefer_local=True)`,
  which, offline/non-gated, returned the local **Turkish WordPiece**
  (`data/tokenizer/tr`, `BertTokenizer`, `do_lower_case=True`, vocab `128000`).
- Eval (`eval/gsm8k.py`, `scripts/eval.py`) loaded
  `AutoTokenizer.from_pretrained(cfg.teacher_model_id)` -> **Llama-3 BPE**
  (vocab `128256`). Two different tokenizer families -> model decoded garbage
  (`111111...`).
- `cfg.vocab_size = 128256` while the actual training tokenizer was `128000`,
  and the model had no resize -> embedding/lm_head size disagreed with the
  tokenizer and with the eval-time model build.
- `train.py` `load_teacher_tokenizer` had a silent `gpt2` fallback (vocab
  `50257`) — a catastrophic third tokenizer.

## Fix

1. **Single resolver.** New `utils/tokenizer_resolver.py::resolve_tokenizer(cfg)`
   is the one path for train, eval, and demo. The teacher-vs-TR choice is
   governed solely by `cfg.use_tr_tokenizer`:
   - `False` (default) -> teacher tokenizer everywhere (teacher inputs, KD, eval).
   - `True` (opt-in) -> local TR tokenizer everywhere. Never half-and-half.
2. **Identity stamped into checkpoints.** Train records
   `checkpoint["tokenizer_id"] = {name_or_path, vocab_size, tokenizer_class,
   use_tr_tokenizer}` (`RUNTIME_TOKENIZER_ID`, written by `save_checkpoint_smart`).
3. **Eval loads from the checkpoint identity.** `eval/gsm8k.py` and
   `scripts/eval.py` reload the tokenizer via `load_tokenizer_from_identity`;
   a checkpoint without `tokenizer_id` (or with a now-inconsistent vocab) raises
   an explicit error. **No silent teacher fallback.**
4. **`vocab_size` derived from `len(tokenizer)`** (includes added/special tokens:
   Llama-3 `vocab_size` attr `128000` vs `len` `128256`). New
   `MertFormer.resize_token_embeddings(n)` aligns embedding + lm_head (weight tie
   preserved) at both train and eval. Closes `128256/128000`.
5. **`gpt2` fallback removed.** Teacher-tokenizer failure now raises with
   guidance (HF_TOKEN / `TITAN_TEACHER_TOKENIZER_PATH` snapshot /
   `TITAN_USE_TR_TOKENIZER=1`). No silent substitution.

## Files changed

- `utils/tokenizer_resolver.py` (new): `resolve_tokenizer`, `tokenizer_identity`,
  `load_tokenizer_from_identity`.
- `model/transformers.py`: `vocab_size` property + `resize_token_embeddings`.
- `train/train.py`: import resolver; `RUNTIME_TOKENIZER_ID`; stamp
  `tokenizer_id` in `save_checkpoint_smart`; `cfg.vocab_size = len(tokenizer)` +
  `resize_token_embeddings`; `load_teacher_tokenizer` delegates to resolver;
  `gpt2` fallback deleted.
- `eval/gsm8k.py`, `scripts/eval.py`, `scripts/chat.py` (demo): tokenizer from
  checkpoint identity; hard error on missing id; model resized to it.

## Tests (added)

- `tests/test_tokenizer_parity.py`
  - missing/empty `tokenizer_id` -> `ValueError` (no silent fallback);
  - recorded-vs-actual vocab mismatch -> `ValueError`;
  - resolver single-source identity for the opt-in TR path;
  - **parity proof**: stamp identity -> torch.save -> reload -> identical
    `name_or_path`/`vocab_size`/class and identical token ids for the same text;
  - `resize_token_embeddings` grows, preserves tie, preserves existing rows,
    no-ops when aligned.
- `tests/test_checkpoint_tokenizer_id.py`
  - `save_checkpoint_smart` stamps `tokenizer_id` (and writes `None` explicitly
    when unset, so eval fails loud rather than guessing);
  - **end-to-end proof**: real train write path (`save_checkpoint_smart`) ->
    eval read path (`load_tokenizer_from_identity`) yields the same tokenizer
    (identical encodings, `len` == recorded `vocab_size`).

## Verification

- New tests: 13 passed.
- Full suite: `298 passed, 4 skipped`.
- `ruff check --select E9,F821,F822,F823` on all changed files: clean.

## Checkpoint-bound benchmark evidence paths (`DONE_NOW`, follow-up wave)

The `post_train_autorun` `checkpoint_required=True` steps (`bench-only` / `full`)
were still bypassing the resolver. Now migrated to the same pattern as
`eval/gsm8k.py` — tokenizer strictly from `checkpoint["tokenizer_id"]`, hard
error when absent, model `resize_token_embeddings` to that vocab:

- `scripts/benchmarks_internal.py` (HumanEval/MBPP). The
  `hf-internal-testing/llama-tokenizer` test fallback now lives **only** in the
  `--allow-random` smoke branch, never on the checkpoint decode path.
- `scripts/golden_score.py` — the script the autorun `golden_eval` step actually
  invokes (`post_train_autorun.py` line 339). Added beyond the two originally
  named files because it, not `golden_eval.py`, is the wired checkpoint-bound
  golden decode path; flagged explicitly in the handoff.
- `scripts/golden_eval.py` (standalone golden decode) — same migration.

Tests: `tests/test_benchmark_tokenizer_id.py` — for all three steps, (a) the
decode path passes `checkpoint["tokenizer_id"]` to the resolver (spy), and
(b) a checkpoint without `tokenizer_id` raises `ValueError` before any model
build (no silent fallback). 6 passed.

## Notes / boundaries

- KD logit-space consistency is now structural: when `use_tr_tokenizer=1`,
  teacher/KD/eval all resolve the same TR tokenizer; otherwise all resolve the
  teacher tokenizer. The resolver cannot produce a half-and-half state.
- Offline teacher use without network requires an explicit
  `TITAN_TEACHER_TOKENIZER_PATH` snapshot of the *same* teacher tokenizer; the
  TR WordPiece is never auto-substituted for the teacher.
- Still out of scope (not on the train->eval decode path; separate follow-up):
  `mertformer_sdk/api.py`, `scripts/train_tpu_turbo.py`,
  `orchestrator/distillation_manager.py`, `scripts/smart_runner.py`,
  `scripts/titan_preflight.py`.
