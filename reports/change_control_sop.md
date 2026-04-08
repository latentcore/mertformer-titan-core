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
