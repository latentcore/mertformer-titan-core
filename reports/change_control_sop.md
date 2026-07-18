# Change Control SOP

## Required Ladder
1. define scope and affected surfaces
2. classify risk and freeze impact
3. inspect current implementation first
4. implement the smallest change that closes the gap
5. run targeted tests
6. run the canonical verification ladder
7. refresh manifests and generated governance artifacts
8. refresh docs and truth surfaces
9. verify no drift remains
10. capture artifacts and reports
11. commit with scoped message
12. push only after the verified tree is clean

## Exceptions
- Hotfixes may shrink scope, but they do not skip verification or truth-sync.
- Freeze-breach changes require an ADR or an explicit governance note.

## Commit/Push Authorization (added 2026-07-19 -- closes a documentation gap)

Steps 11-12 above (commit, push) are gated on **explicit, per-request authorization from the
repo owner** -- not implied by an earlier approval, not inferred from task momentum, and not
granted by completing steps 1-10. Concretely:
- An agent/operator must have a clear, current-turn instruction to commit and/or push before
  doing so. A prior turn's authorization does not carry forward to a new turn's changes.
- If authorization is ambiguous or was given for a different, narrower change, stop and ask
  rather than assuming it extends to the current change.
- This authority rule lives here (repo-internal) so it is visible to anyone with only repo
  access; it does not depend on any external, non-tracked document.
