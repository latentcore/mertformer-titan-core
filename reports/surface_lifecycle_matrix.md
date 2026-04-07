# Surface Lifecycle Matrix

Canonical lifecycle classes for the current closure pass.

| Surface Family | Lifecycle Class | Change Barrier | Notes |
| --- | --- | --- | --- |
| `AGENTS.md`, source-of-truth maps, truth constitution, schemas | `frozen` | high | Frozen surfaces define governance, schemas, naming, and release-truth constraints. |
| Verification gates, manifests, handoff packs, readiness contracts | `maintained` | medium | These must stay current and reproducible, but should not churn without measured reason. |
| Training, benchmark, kernel, chess, export, packaging, product, security, legal, pilot lanes | `living` | controlled | These surfaces are expected to change as implementation and measured evidence evolve. |
| Workspace hygiene reports and quarantine manifests | `maintained` | medium | Hygiene policy is stable, but item-level decisions must refresh as the workspace changes. |
| Research moonshots (`3000+ Elo`, `20 ms/move`, `10000x speedup`, AGI/ASI) | `living` | high external proof bar | Research lanes stay outside V1 release truth until independently measured. |
