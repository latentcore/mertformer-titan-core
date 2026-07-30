# Contributing

This repository is licensed under **Apache License 2.0** (see [LICENSE](LICENSE)) and
**is open to external contributions**. By submitting a contribution you agree it is
licensed under the same terms, per Apache 2.0 Section 5.

Every pull request must pass `bash scripts/verify_all.sh` with zero regressions.
See the "Contributions & PR Rules" section of [README.md](README.md) for the full list.

Developed by Mert Yünlü. AI coding assistants (Claude Code) were used for implementation; all architecture, design decisions, and final review are the author's own.

## Internal Workflow
- Use `feature/` and `fix/` branch prefixes.
- Keep all claims and metrics aligned with **pre-training / pending validation** status.
- Avoid committing large binaries or datasets; prefer scripts and reproducible configs.
- Document changes briefly in the relevant README/MD when they affect user-facing behavior.

## Security
Report security issues privately — see [SECURITY.md](SECURITY.md). Do not open a public issue.
