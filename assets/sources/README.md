# Asset Source Archive

This directory stores **editable source files** for visual assets used in the repository.

## Purpose
- Keep architecture visuals maintainable across releases.
- Preserve source-to-export traceability for audits.
- Avoid irreversible edits on final PNG/GIF artifacts.

## Expected Source Formats
- `.drawio` / `.xml` (diagrams)
- `.fig` (Figma exports)
- `.psd` / `.ai` (design masters)
- Optional: `.svg` editable intermediate exports

## Naming Convention
- `header_v<build>.{ext}`
- `synaptic_map_v<build>.{ext}`
- `*_source_v<build>.{ext}`

## Export Mapping Rule
Each source file should map to a committed runtime asset under `assets/`:
- `header_v*.{ext}` -> `assets/header.png`
- `synaptic_map_v*.{ext}` -> `assets/synaptic_map.png`

When updating visuals:
1. Update source file in this folder.
2. Export runtime image to `assets/`.
3. Mention the update in release notes or snapshot reports.
