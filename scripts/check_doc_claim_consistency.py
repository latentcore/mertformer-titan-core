#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

README_EN = ROOT / "README.md"
README_TR = ROOT / "README_TR.md"
SNAP_EN = ROOT / "reports/release_snapshot.md"
SNAP_TR = ROOT / "reports/release_snapshot_TR.md"
MISSION_EN = ROOT / "MISSION.md"
MISSION_TR = ROOT / "MISSION_TR.md"
USE_POLICY_EN = ROOT / "USE_POLICY.md"
USE_POLICY_TR = ROOT / "USE_POLICY_TR.md"
SECURITY_EN = ROOT / "SECURITY.md"
SECURITY_TR = ROOT / "SECURITY_TR.md"
MODEL_CARD_EN = ROOT / "MODEL_CARD.md"
MODEL_CARD_TR = ROOT / "MODEL_CARD_TR.md"
SYSTEM_PROMPT = ROOT / "prompts/system_v1.txt"

TEST_STAT_RE = re.compile(r"(\d+\s+passed,\s*\d+\s+skipped)")

# Snapshot history is intentionally frozen; skip from global claim-pattern scans.
EXCLUDED_PREFIXES = [
    ROOT / "reports" / "snapshots",
]

BANNED_PATTERNS = [
    re.compile(r"\bWorld'?s\s+First\b", re.IGNORECASE),
    re.compile(r"\bfirst\s+empirically\s+stable\b", re.IGNORECASE),
    re.compile(r"\bfirst-ever\b", re.IGNORECASE),
    re.compile(r"\bDünyada\s+bir\s+ilk\b", re.IGNORECASE),
    re.compile(r"\bCfC-based\s+router\b", re.IGNORECASE),
    re.compile(r"\bCfC\s+tabanlı\s+router\b", re.IGNORECASE),
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def is_excluded(path: Path) -> bool:
    p = path.resolve()
    return any(str(p).startswith(str(prefix.resolve())) for prefix in EXCLUDED_PREFIXES)


def iter_public_markdown() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*.md"):
        if is_excluded(path):
            continue
        files.append(path)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check documentation claim/evidence consistency.")
    parser.add_argument("--expected-test-stat", default="122 passed, 3 skipped")
    args = parser.parse_args()

    errors: list[str] = []

    en = read_text(README_EN)
    tr = read_text(README_TR)
    snap_en = read_text(SNAP_EN)
    snap_tr = read_text(SNAP_TR)
    mission_en = read_text(MISSION_EN)
    mission_tr = read_text(MISSION_TR)
    use_policy_en = read_text(USE_POLICY_EN)
    use_policy_tr = read_text(USE_POLICY_TR)
    security_en = read_text(SECURITY_EN)
    security_tr = read_text(SECURITY_TR)
    model_card_en = read_text(MODEL_CARD_EN)
    model_card_tr = read_text(MODEL_CARD_TR)
    system_prompt = read_text(SYSTEM_PROMPT)

    required_pairs = [
        ("README.md", en, "NOT ELIGIBLE FOR CLAIM"),
        ("README_TR.md", tr, "NOT ELIGIBLE FOR CLAIM"),
        ("README.md", en, "pre-training"),
        ("README.md", en, "not to claim a production-ready or certified platform"),
        ("README_TR.md", tr, "proof-of-system"),
        ("README.md", en, "MLA-labeled GQA attention (current implementation)"),
        ("README_TR.md", tr, "MLA etiketli GQA dikkat bloğu (mevcut implementasyon)"),
        ("README.md", en, "Routing policy: token-choice top-k."),
        ("README_TR.md", tr, "Yönlendirme politikası: token-choice top-k."),
        ("README.md", en, "out_of_scope_pending_ids=[8, 9, 11, 12, 51, 52, 54, 55, 56, 57]"),
        ("README_TR.md", tr, "out_of_scope_pending_ids=[8, 9, 11, 12, 51, 52, 54, 55, 56, 57]"),
        ("README.md", en, "Default mode is `verified`."),
        ("README_TR.md", tr, "Varsayılan mod `verified`"),
        ("MISSION.md", mission_en, "No claim without evidence."),
        ("MISSION_TR.md", mission_tr, "Kanıt yoksa claim yoktur."),
        ("USE_POLICY.md", use_policy_en, "creative_or_folklore"),
        ("USE_POLICY_TR.md", use_policy_tr, "creative_or_folklore"),
        ("SECURITY.md", security_en, "45K readiness is the primary ship gate for this pass."),
        ("SECURITY_TR.md", security_tr, "ana ship gate 45K readiness"),
        ("MODEL_CARD.md", model_card_en, "first serious architecture validation run"),
        ("MODEL_CARD_TR.md", model_card_tr, "ilk ciddi mimari doğrulama koşusu"),
        ("prompts/system_v1.txt", system_prompt, "Default output mode is verified mode."),
        ("prompts/system_v1.txt", system_prompt, "No claim may remain in final docs unless backed by evidence"),
    ]
    for name, text, needle in required_pairs:
        if needle not in text:
            errors.append(f"missing required phrase in {name}: {needle}")

    stat_sources = {
        "README.md": en,
        "README_TR.md": tr,
        "reports/release_snapshot.md": snap_en,
        "reports/release_snapshot_TR.md": snap_tr,
    }
    found_stats: set[str] = set()
    for name, text in stat_sources.items():
        m = TEST_STAT_RE.search(text)
        if not m:
            errors.append(f"missing test stat marker in {name}")
            continue
        stat = m.group(1)
        found_stats.add(stat)

    if len(found_stats) > 1:
        errors.append(f"inconsistent test stats across docs: {sorted(found_stats)}")
    if args.expected_test_stat not in found_stats:
        errors.append(
            f"expected stat '{args.expected_test_stat}' not found in docs (found: {sorted(found_stats)})"
        )

    pointer_marker_en = "Turkish counterparts for audits are pointer files"
    if pointer_marker_en not in snap_en:
        errors.append(
            "release_snapshot.md must explicitly mark audit EN_TR/DE_TR docs as pointer files"
        )

    pointer_marker_tr_variants = (
        "TR denetim dosyaları yönlendirme (pointer) dosyalarıdır",
        "TR denetim dosyalari yonlendirme (pointer) dosyalaridir",
    )
    if not any(x in snap_tr for x in pointer_marker_tr_variants):
        errors.append(
            "release_snapshot_TR.md must explicitly mark audit EN_TR/DE_TR docs as pointer files"
        )

    for md_path in iter_public_markdown():
        text = read_text(md_path)
        rel = md_path.relative_to(ROOT)
        for pat in BANNED_PATTERNS:
            if pat.search(text):
                errors.append(f"banned claim pattern in {rel}: /{pat.pattern}/")

    if errors:
        print("FAIL: documentation claim consistency check failed")
        for e in errors:
            print(f" - {e}")
        return 1

    print("OK: documentation claim consistency checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
