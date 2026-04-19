# Build30 T4 One-Cell Smoke Test Result

## Purpose

The goal of this smoke test was not to prove model quality. The goal was to verify whether the `step_00003500.pt` checkpoint could be loaded enough to perform a simple generation attempt.

## Inputs

- Checkpoint: `/kaggle/working/mertformer_onecell_outputs/checkpoints/kaggle_onecell_t4_build30/step_00003500.pt`
- Device used in the final attempt: `cuda`
- Tokenizer recovered from checkpoint: `/content/recovered_mertformer_sentencepiece.model`
- Vocabulary size: `32768`
- Train state: `step=3500`, `tokens_seen=9856000`, `best_val_loss=7.990970656275749`

## Observed Output

Prompt:

```text
2 + 2 =
```

Observed generated text:

```text
111111111111111111111111
```

Other simple prompts showed the same repeated-token behavior.

## Interpretation

The checkpoint produced a degenerate repeated-token output in this fallback generation smoke test.

This means:
- the checkpoint and tokenizer evidence are real
- generation was attempted
- this checkpoint should not be presented as a usable chat model
- this checkpoint should be presented only as provisional training evidence

## Important Loader Caveat

The smoke test used a compatibility/fallback loader rather than the exact original runtime class. The fallback loader had high `MISSING` and `UNEXPECTED` state-dict counts during earlier attempts, so generation quality cannot be treated as a final architectural evaluation.

Even with that caveat, the output is not suitable for a public chatbot demo.

## Truth Boundary

Correct statement:

```text
Build30 produced a real provisional training evidence checkpoint and logged loss improvements, but the current smoke generation does not demonstrate a usable conversational model.
```

Incorrect statement:

```text
Build30 is a working chatbot.
```
