# System Memory Policy

## Durable Memory Order
1. `AGENTS.md`
2. source-of-truth and truth-constitution reports
3. ADR chain
4. runbooks and SOP-aligned operator docs
5. manifests, hashes, and provenance artifacts
6. closure reports and scorecards
7. backlog classification and missing-items reports

## Rules
- Critical state must be written into repo memory, not left in chat context alone.
- Current state must be recoverable from docs plus manifests plus reports.
- Resume safety depends on written artifacts, not recollection.
