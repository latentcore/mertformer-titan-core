#!/usr/bin/env python3
"""
FACTS live-drift gate.

Distinct from scripts/check_facts_consistency.py (which checks that stale
string IDs don't leak into current-truth docs -- a text-consistency check).
This gate instead RECOMPUTES numbers and compares them to what reports/FACTS.json
and config/model/*.yaml docstrings claim, to catch the class of bug that
actually happened on 2026-07-12: mertformer_pilot_stabilization.yaml's own
"MEASURED" docstring said 171,617,923 params, but a real MertFormer()
instantiated from that exact file (once its Liquid/MoE layer collision was
fixed so it could load at all) measured 172,668,035 -- a real, silent,
un-caught drift between a docstring's claimed number and what the config
actually produces.

Two checks:
  1. Small config/model/*.yaml overlays (below PARAM_INSTANTIATE_CEILING,
     cheap to instantiate on CPU) are loaded for real via a fresh subprocess,
     a real MertFormer() is built, and its exact measured param count is
     compared against the "total parameters ... N" figure embedded in that
     yaml's own docstring comment (if present). Any mismatch is a hard
     drift -- exact check, since this is a real measured count.
  2. The canonical (no-overlay) architecture's analytical estimate
     (config._estimate_total_params, cheap, no instantiation) is compared
     against reports/FACTS.json's measured_runtime_params with a wide
     tolerance band (the estimator is documented as approximate, not
     claim-grade) -- catches gross architecture drift without paying the
     cost of instantiating the full ~3.67B canonical model on every CI run.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OVERLAY_DIR = ROOT / "config" / "model"
FACTS_PATH = ROOT / "reports" / "FACTS.json"

PARAM_INSTANTIATE_CEILING = 300_000_000  # overlays above this are too expensive to build in CI
GROSS_DRIFT_TOLERANCE = 0.15  # 15% -- the analytical estimator is approximate by design

_MEASURE_PROBE = """
import re
import model.transformers as mt
model = mt.MertFormer()
n = sum(p.numel() for p in model.parameters())
print("MEASURED_PARAMS=" + str(n))
"""

DOCSTRING_PARAM_RE = re.compile(r"total parameters\s*\.*\s*([\d,]+)")
# A yaml is skipped (not a drift target) unless it declares its own measured count.


def _approx_overlay_param_count(yaml_text: str) -> int:
    """Cheap heuristic to decide whether an overlay is small enough to instantiate:
    read hidden_size/num_layers/vocab_size-ish fields if present, else assume small."""
    def _field(name: str, default: int) -> int:
        m = re.search(rf"^{name}:\s*(\d+)", yaml_text, re.MULTILINE)
        return int(m.group(1)) if m else default

    hidden = _field("hidden_size", 512)
    layers = _field("num_layers", 9)
    vocab = 128256
    # Rough order-of-magnitude: embedding dominates at small scale.
    return vocab * hidden + layers * hidden * hidden * 12


def measure_overlay_params(yaml_name: str) -> int:
    env = dict(os.environ)
    env["MERTFORMER_MODEL_CONFIG"] = yaml_name
    proc = subprocess.run(
        [sys.executable, "-c", _MEASURE_PROBE],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to instantiate MertFormer() under {yaml_name}: {proc.stderr[-2000:]}")
    m = re.search(r"MEASURED_PARAMS=(\d+)", proc.stdout)
    if not m:
        raise RuntimeError(f"probe produced no MEASURED_PARAMS line for {yaml_name}: {proc.stdout!r}")
    return int(m.group(1))


def check_small_overlays() -> list:
    offenders = []
    for path in sorted(OVERLAY_DIR.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        claimed_match = DOCSTRING_PARAM_RE.search(text)
        if not claimed_match:
            continue  # overlay doesn't claim a measured count; nothing to drift-check
        claimed = int(claimed_match.group(1).replace(",", ""))
        if _approx_overlay_param_count(text) > PARAM_INSTANTIATE_CEILING:
            print(f"SKIP (too large to instantiate in CI): {path.name}")
            continue
        try:
            measured = measure_overlay_params(path.name)
        except RuntimeError as exc:
            offenders.append(f"{path.name}: could not measure ({exc})")
            continue
        if measured != claimed:
            offenders.append(
                f"{path.name}: docstring claims {claimed:,} params, "
                f"a live MertFormer() built from this overlay actually measures {measured:,}"
            )
        else:
            print(f"OK: {path.name} docstring ({claimed:,}) matches live measurement")
    return offenders


def check_canonical_gross_drift() -> list:
    sys.path.insert(0, str(ROOT))
    import config.config as c  # noqa: E402

    estimated = c._estimate_total_params(None)
    facts = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
    measured = int(str(facts["measured_runtime_params"]).replace(",", ""))
    ratio = abs(estimated - measured) / measured
    print(
        f"canonical: FACTS.json measured={measured:,}  analytical_estimate={estimated:,.0f}  "
        f"delta={ratio:.1%} (tolerance {GROSS_DRIFT_TOLERANCE:.0%})"
    )
    if ratio > GROSS_DRIFT_TOLERANCE:
        return [
            f"canonical architecture: analytical estimate ({estimated:,.0f}) diverges from "
            f"FACTS.json measured_runtime_params ({measured:,}) by {ratio:.1%}, "
            f"exceeding the {GROSS_DRIFT_TOLERANCE:.0%} tolerance -- possible architecture/FACTS drift"
        ]
    return []


def main() -> int:
    offenders = []
    offenders.extend(check_small_overlays())
    offenders.extend(check_canonical_gross_drift())

    if offenders:
        print(f"\nFAIL: {len(offenders)} FACTS-drift finding(s):")
        for o in offenders:
            print(f"  - {o}")
        return 1

    print("\nOK: no FACTS drift detected (small-overlay docstrings match live measurement; canonical estimate within tolerance).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
