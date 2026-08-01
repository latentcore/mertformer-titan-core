# Contributing

This repository is licensed under **Apache License 2.0** (see [LICENSE](LICENSE)) and
**is open to external contributions**. By submitting a contribution you agree it is
licensed under the same terms, per Apache 2.0 Section 5.

Every pull request must pass `bash scripts/verify_all.sh` with zero regressions.
See the "Contributions & PR Rules" section of [README.md](README.md) for the full list.

Developed by Mert Yünlü. AI coding assistants (Claude Code) were used for implementation; all architecture, design decisions, and final review are the author's own.

## Commit Message Style

Title: `<type>: <concise, imperative summary>` — type is one of
`fix|docs|feat|refactor|test|chore|harden|closure`.

Body:
1. What changed and why (the actual finding/motivation).
2. If it required investigation: how it was found/scoped (especially if
   scope could reasonably be questioned).
3. Anything deliberately left untouched, and why.
4. A `Verified:` line with concrete numbers — not "tests pass," the
   actual `N passed, M skipped` and which gates ran.
5. `Co-Authored-By: <model name> <email>` trailer if AI-assisted, naming
   the actual model/tool used for that commit.

Brevity scales with risk/ambiguity, not diff size. Example (commit
`0b4f79d6`):

```
fix: remove dead env var in check_overlay_validity.py + add cuda to chess GUI --device

scripts/check_overlay_validity.py::check_overlay(): removed a dead, unused
env dict (hardcoded Unix-only PATH: "/usr/bin:/bin:/usr/local/bin") that was
assigned but never passed to subprocess.run() (the call already uses
full_env instead) -- confirmed via `ruff --select F841` (repo's default lint
scope, pyproject.toml's select = ["E9", "F821", "F822", "F823"], deliberately
excludes this class to avoid low-signal churn across legacy scripts; this
one instance was fixed directly since it was already hand-identified).

apps/chess_gui/play_mertformer_chess_web.py: --device CLI flag's argparse
choices was missing "cuda" (only ["cpu", "mps"]) -- the auto-detect path
(choose_device()) already checks CUDA first, but a user could not force it
explicitly.

A repo-wide ruff/bandit discovery pass (130 unused-import/variable findings,
543 low/medium bandit findings) was run but deliberately not acted on beyond
the one instance above -- both fall inside the project's own documented
no-broad-cleanup lint policy and outside this pass's layers/model/train/
orchestrator/mertformer_sdk no-touch boundary.

Verified: tests/test_chess_gui_contract.py + tests/test_config_overlay_strict.py
(10 passed), scripts/check_overlay_validity.py run directly (4/4 overlays OK),
full suite 721 passed, 9 skipped, 1 xfailed (identical count, no regression),
doc-claim consistency OK (after reverting a locally-regenerated, never-committed
reports/train_readiness_decision.{json,md} back to canonical -- known
machine-local-state artifact, not a real doc bug).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
```

## Internal Workflow
- Use `feature/` and `fix/` branch prefixes.
- Keep all claims and metrics aligned with **pre-training / pending validation** status.
- Avoid committing large binaries or datasets; prefer scripts and reproducible configs.
- Document changes briefly in the relevant README/MD when they affect user-facing behavior.

## Security
Report security issues privately — see [SECURITY.md](SECURITY.md). Do not open a public issue.
