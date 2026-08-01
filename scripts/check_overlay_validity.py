#!/usr/bin/env python3
"""
Config overlay validity gate.

config/config.py runs validate_layer_config(cfg) once, at module-import time,
against whichever single MERTFORMER_MODEL_CONFIG overlay (or none) happened to
be active in that process. Nothing in CI previously imported config.config
once per config/model/*.yaml overlay in a fresh subprocess, so a Liquid/MoE
layer-index collision in an overlay nobody happened to import that day could
sit unnoticed indefinitely -- exactly what happened to
config/model/mertformer_pilot_stabilization.yaml (liquid_layers_idx [2, 5]
collided with MoE's [2, 5, 8]; the file could never actually load, discovered
only by manual inspection on 2026-07-12, fixed in the same pass).

This gate loads EVERY config/model/*.yaml overlay, each in its own fresh
subprocess (module-level import side effects mean the same interpreter can't
safely re-validate a second overlay), and hard-fails if any overlay's config
fails to import/validate.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OVERLAY_DIR = ROOT / "config" / "model"

_PROBE = (
    "import config.config as c\n"
    "print('OVERLAY_OK total_params_field_present=' + str(hasattr(c.cfg, 'num_layers')))\n"
)


def check_overlay(yaml_name: str) -> tuple:
    import os

    full_env = dict(os.environ)
    full_env["MERTFORMER_MODEL_CONFIG"] = yaml_name
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=ROOT,
        env=full_env,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=120,
    )
    ok = proc.returncode == 0 and "OVERLAY_OK" in proc.stdout
    return ok, proc.stdout, proc.stderr


def main() -> int:
    if not OVERLAY_DIR.is_dir():
        print(f"FAIL: overlay directory missing: {OVERLAY_DIR}")
        return 2

    overlays = sorted(p.name for p in OVERLAY_DIR.glob("*.yaml"))
    if not overlays:
        print("WARN: no config/model/*.yaml overlays found; nothing to validate.")
        return 0

    failures = []
    for name in overlays:
        ok, stdout, stderr = check_overlay(name)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}")
        if not ok:
            tail = "\n".join(stderr.strip().splitlines()[-15:])
            print(f"  --- stderr tail ---\n  " + tail.replace("\n", "\n  "))
            failures.append(name)

    if failures:
        print(f"\nFAIL: {len(failures)}/{len(overlays)} overlay(s) fail to load/validate: {failures}")
        return 1

    print(f"\nOK: all {len(overlays)} config/model/*.yaml overlays load and pass validate_layer_config().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
