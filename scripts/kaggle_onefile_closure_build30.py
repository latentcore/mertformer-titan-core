#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

try:
    import torch
except Exception:  # pragma: no cover - optional at import time
    torch = None  # type: ignore


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_ARTIFACT_ROOT = ROOT / "reports" / "benchmarks" / "kaggle_runs"
DEFAULT_LOCAL_CHECKPOINT_DIR = ROOT / "checkpoints" / "kaggle_onefile_build30"
DEFAULT_KAGGLE_ARTIFACT_ROOT = Path("/kaggle/working/mertformer_outputs")
DEFAULT_KAGGLE_CHECKPOINT_DIR = DEFAULT_KAGGLE_ARTIFACT_ROOT / "checkpoints" / "kaggle_onefile_build30"
DEFAULT_LOCAL_ONECELL_ARTIFACT_ROOT = ROOT / "reports" / "benchmarks" / "kaggle_onecell_runs"
DEFAULT_LOCAL_ONECELL_CHECKPOINT_DIR = ROOT / "checkpoints" / "kaggle_onecell_t4_build30"
DEFAULT_KAGGLE_ONECELL_ARTIFACT_ROOT = Path("/kaggle/working/mertformer_onecell_outputs")
DEFAULT_KAGGLE_ONECELL_CHECKPOINT_DIR = DEFAULT_KAGGLE_ONECELL_ARTIFACT_ROOT / "checkpoints" / "kaggle_onecell_t4_build30"
SCHEMA = "kaggle_onefile_closure_build30_v1"

LEGACY_SCRIPT_MAP: dict[str, tuple[str, str]] = {
    "build30": ("scripts/kaggle_onefile_demo_build30.py", "kaggle_onefile_demo_build30"),
    "onecell_t4": ("scripts/kaggle_onecell_t4_build30.py", "kaggle_onecell_t4_build30"),
    "fastproof": (
        "scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py",
        "kaggle_onefile_demo_build30_colab_math_fastproof",
    ),
}

PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "sweetspot": {
        "legacy_lane": "build30",
        "legacy_profile": "linkedin_sweetspot",
        "description": "Balanced proof-of-learning profile for canonical Kaggle evidence.",
        "overrides": {
            "mert_enable_all_extensions": False,
            "mert_use_qinn": False,
            "allow_notebook_input": False,
            "chat_enabled": False,
            "interactive": False,
            "interactive_menu": False,
            "step_log_interval": 10,
            "max_wall_hours": 3.5,
        },
    },
    "p100_safe": {
        "legacy_lane": "build30",
        "legacy_profile": "linkedin_sweetspot",
        "description": (
            "Conservative single-GPU profile tuned for Tesla P100-class Kaggle notebooks. "
            "max_steps is a safety cap, not the canonical 45K run: at batch_size=4/seq_len=256 "
            "this profile processes ~1024 tokens/step vs the canonical 128x4096=524288 "
            "tokens/step, so reaching this step count is NOT equivalent to, and must never be "
            "reported as, the real 45K training run."
        ),
        "overrides": {
            "batch_size": 4,
            "seq_len": 256,
            "grad_accum_steps": 2,
            "mert_enable_all_extensions": False,
            "mert_use_qinn": False,
            "allow_notebook_input": False,
            "chat_enabled": False,
            "interactive": False,
            "interactive_menu": False,
            "step_log_interval": 10,
            "max_wall_hours": 10.8,
            # [2026-07-11] Was 45000 -- an exact numeric collision with the canonical 45K
            # training run that could make a P100 probe look like the real 45K in reports.
            # This profile is wall-clock-bounded (max_wall_hours above); this is a safety
            # cap, not a target step count. Kept well clear of 45000.
            "max_steps": 20000,
        },
    },
    "t4x2_dist": {
        "legacy_lane": "build30",
        "legacy_profile": "linkedin_sweetspot",
        "description": "Aggressive but claim-safe dual-T4 profile for Kaggle T4 x2 runtimes.",
        "overrides": {
            "batch_size": 8,
            "seq_len": 256,
            "grad_accum_steps": 2,
            "mert_enable_all_extensions": False,
            "mert_use_qinn": False,
            "allow_notebook_input": False,
            "chat_enabled": False,
            "interactive": False,
            "interactive_menu": False,
            "step_log_interval": 10,
            "max_wall_hours": 10.8,
            "max_steps": 60000,
        },
    },
    "onecell_t4_sweetspot": {
        "legacy_lane": "onecell_t4",
        "legacy_profile": "t4_onecell_sweetspot",
        "description": "Single-T4, single-cell Kaggle lane with structured evidence and guarded repo parity.",
        "overrides": {
            "mert_enable_all_extensions": False,
            "mert_use_qinn": False,
            "allow_notebook_input": False,
            "chat_enabled": False,
            "interactive": False,
            "interactive_menu": False,
            "step_log_interval": 10,
            "max_wall_hours": 5.5,
        },
    },
    "mini300m_probe": {
        "legacy_lane": "build30",
        "legacy_profile": "mini300m",
        "description": "Longer convergence probe centered on the 300M-class mini architecture.",
        "overrides": {
            "mert_enable_all_extensions": False,
            "mert_use_qinn": False,
            "allow_notebook_input": False,
            "chat_enabled": False,
            "interactive": False,
            "interactive_menu": False,
            "step_log_interval": 25,
            "max_wall_hours": 10.8,
            "max_steps": 120000,
        },
    },
    "fastproof_math": {
        "legacy_lane": "fastproof",
        "legacy_profile": "colab_math_fastproof",
        "description": "Fast arithmetic proof lane with strict config/evidence contracts.",
        "overrides": {
            "allow_notebook_input": False,
            "chat_enabled": False,
            "interactive": False,
            "interactive_menu": False,
            "step_log_interval": 10,
            "max_wall_hours_locked": True,
            "max_wall_hours": 1.0,
            "compile_policy": "off",
        },
    },
}

ALWAYS_ON_COMPARE_ARGS = {
    "steps": "2",
    "batch_size": "1",
    "seq_len": "16",
    "vocab_size": "256",
    "hidden": "64",
    "layers": "2",
    "heads": "4",
}


@dataclass(frozen=True)
class RuntimeMeta:
    kaggle: bool
    colab: bool
    gpu_count: int
    gpu_names: tuple[str, ...]
    gpu_label: str
    device: str


@dataclass(frozen=True)
class Layout:
    artifact_root: Path
    checkpoint_dir: Path
    run_id: str
    run_dir: Path
    auxiliary_dir: Path
    closure_dir: Path
    report_out: Path
    canonical_summary_path: Path
    canonical_summary_md_path: Path
    first100_snapshot_path: Path
    canonical_artifact_index_path: Path
    canonical_sha256_manifest_path: Path
    canonical_package_manifest_path: Path
    canonical_bundle_path: Path


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def local_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def sanitize_text(text: str) -> str:
    return text.replace(str(ROOT), "<REPO_ROOT>")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def probe_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def resolve_writable_dir(preferred: Path, fallbacks: Iterable[Path]) -> Path:
    for candidate in (preferred, *fallbacks):
        if probe_writable_dir(candidate):
            return candidate
    return Path.cwd()


def detect_runtime() -> RuntimeMeta:
    kaggle = bool(
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or os.environ.get("KAGGLE_URL_BASE")
        or os.environ.get("KAGGLE_KERNEL_RUN_ID")
    )
    colab = bool(
        os.environ.get("COLAB_GPU")
        or os.environ.get("COLAB_TPU_ADDR")
        or os.environ.get("COLAB_BACKEND_VERSION")
    )

    gpu_names: list[str] = []
    device = "cpu"
    if torch is not None and torch.cuda.is_available():
        device = "cuda"
        gpu_names = [str(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
    elif torch is not None and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"

    gpu_label = "none"
    normalized = [name.upper() for name in gpu_names]
    if gpu_names:
        if len(gpu_names) == 2 and all("T4" in name for name in normalized):
            gpu_label = "GPU T4 x2"
        elif any("P100" in name for name in normalized):
            gpu_label = "GPU P100"
        elif len(gpu_names) > 1:
            gpu_label = f"{gpu_names[0]} x{len(gpu_names)}"
        else:
            gpu_label = gpu_names[0]

    return RuntimeMeta(
        kaggle=kaggle,
        colab=colab,
        gpu_count=len(gpu_names),
        gpu_names=tuple(gpu_names),
        gpu_label=gpu_label,
        device=device,
    )


def choose_profile(requested: str, runtime: RuntimeMeta) -> str:
    requested = (requested or "auto").strip().lower()
    if requested and requested != "auto":
        if requested not in PROFILE_SPECS:
            raise ValueError(f"Unsupported profile: {requested}")
        return requested
    label = runtime.gpu_label.upper()
    if label == "GPU T4 X2":
        return "t4x2_dist"
    if "T4" in label:
        return "onecell_t4_sweetspot"
    if "P100" in label:
        return "p100_safe"
    return "sweetspot"


def build_paths(
    runtime: RuntimeMeta,
    selected_profile: str,
    artifact_root_arg: str,
    checkpoint_dir_arg: str,
    run_id: str,
    report_out: str,
) -> Layout:
    onecell_profile = selected_profile == "onecell_t4_sweetspot"
    if artifact_root_arg:
        artifact_root = Path(artifact_root_arg).expanduser()
    elif onecell_profile and runtime.kaggle:
        artifact_root = DEFAULT_KAGGLE_ONECELL_ARTIFACT_ROOT
    elif onecell_profile:
        artifact_root = DEFAULT_LOCAL_ONECELL_ARTIFACT_ROOT
    elif runtime.kaggle:
        artifact_root = DEFAULT_KAGGLE_ARTIFACT_ROOT
    else:
        artifact_root = DEFAULT_LOCAL_ARTIFACT_ROOT

    if runtime.kaggle:
        artifact_root = resolve_writable_dir(
            artifact_root,
            [Path("/kaggle/working"), Path.cwd() / "artifacts"],
        )
    else:
        fallback_root = DEFAULT_LOCAL_ONECELL_ARTIFACT_ROOT if onecell_profile else DEFAULT_LOCAL_ARTIFACT_ROOT
        artifact_root = resolve_writable_dir(artifact_root, [fallback_root, ROOT / "artifacts", Path.cwd()])

    if checkpoint_dir_arg:
        checkpoint_dir = Path(checkpoint_dir_arg).expanduser()
    elif onecell_profile and runtime.kaggle:
        checkpoint_dir = DEFAULT_KAGGLE_ONECELL_CHECKPOINT_DIR
    elif onecell_profile:
        checkpoint_dir = DEFAULT_LOCAL_ONECELL_CHECKPOINT_DIR
    elif runtime.kaggle:
        checkpoint_dir = DEFAULT_KAGGLE_CHECKPOINT_DIR
    else:
        checkpoint_dir = DEFAULT_LOCAL_CHECKPOINT_DIR

    if not checkpoint_dir.is_absolute():
        checkpoint_dir = artifact_root / checkpoint_dir
    checkpoint_name = DEFAULT_LOCAL_ONECELL_CHECKPOINT_DIR.name if onecell_profile else DEFAULT_LOCAL_CHECKPOINT_DIR.name
    checkpoint_dir = resolve_writable_dir(
        checkpoint_dir,
        [artifact_root / "checkpoints" / checkpoint_name, ROOT / "checkpoints" / checkpoint_name],
    )

    run_dir = artifact_root / "runs" / run_id if runtime.kaggle else artifact_root / run_id
    closure_dir = run_dir / "closure"
    auxiliary_dir = run_dir / "auxiliary"
    if report_out:
        report_out_path = Path(report_out).expanduser()
    else:
        report_out_path = closure_dir / "canonical_status.json"

    return Layout(
        artifact_root=artifact_root,
        checkpoint_dir=checkpoint_dir,
        run_id=run_id,
        run_dir=run_dir,
        auxiliary_dir=auxiliary_dir,
        closure_dir=closure_dir,
        report_out=report_out_path,
        canonical_summary_path=closure_dir / "canonical_closure_summary.json",
        canonical_summary_md_path=closure_dir / "canonical_closure_summary.md",
        first100_snapshot_path=closure_dir / "first_100_step_loss_snapshot.json",
        canonical_artifact_index_path=closure_dir / "canonical_artifact_index.json",
        canonical_sha256_manifest_path=closure_dir / "canonical_sha256_manifest.txt",
        canonical_package_manifest_path=closure_dir / "canonical_package_manifest.json",
        canonical_bundle_path=closure_dir / f"{run_id}_canonical_bundle.zip",
    )


@contextmanager
def patched_argv(argv: list[str]) -> Iterator[None]:
    original = sys.argv[:]
    sys.argv = argv[:]
    try:
        yield
    finally:
        sys.argv = original


@contextmanager
def patched_env(updates: dict[str, str]) -> Iterator[None]:
    original: dict[str, Optional[str]] = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def patched_run_config(module: Any, overrides: dict[str, Any]) -> Iterator[None]:
    original = dict(module.RUN_CONFIG)
    module.RUN_CONFIG.update(overrides)
    try:
        yield
    finally:
        module.RUN_CONFIG.clear()
        module.RUN_CONFIG.update(original)


_loaded_modules: dict[str, Any] = {}


def load_local_module(kind: str):
    if kind in _loaded_modules:
        return _loaded_modules[kind]
    rel_path, module_name = LEGACY_SCRIPT_MAP[kind]
    script_path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module for {kind}: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _loaded_modules[kind] = module
    return module


def canonical_overrides(profile: str, layout: Layout, quick: bool, checkpoint: str, mode: str) -> dict[str, Any]:
    spec = PROFILE_SPECS[profile]
    overrides = dict(spec["overrides"])
    overrides.update(
        {
            "profile": spec["legacy_profile"],
            "artifact_root": str(layout.artifact_root),
            "out_dir": str(layout.artifact_root),
            "checkpoint_dir": str(layout.checkpoint_dir),
            "artifact_run_id": layout.run_id,
            "bundle_out": str(layout.run_dir / f"{layout.run_id}_evidence.zip"),
            "interactive": False,
            "interactive_menu": False,
            "allow_notebook_input": False,
            "chat_enabled": False,
            "chat_interactive": False,
            "force_interactive_input": False,
            "write_files": True,
        }
    )
    if quick:
        overrides.update({"max_steps": 120, "max_wall_hours": 0.2})
    if mode == "resume" and checkpoint:
        overrides.update(
            {
                "resume_mode": "path",
                "resume_path": checkpoint,
                "checkpoint_path": checkpoint,
            }
        )
    elif checkpoint:
        overrides.setdefault("checkpoint_path", checkpoint)
    return overrides


def execute_legacy_lane(profile: str, layout: Layout, mode: str, checkpoint: str, quick: bool) -> dict[str, Any]:
    spec = PROFILE_SPECS[profile]
    module = load_local_module(str(spec["legacy_lane"]))
    overrides = canonical_overrides(profile, layout, quick=quick, checkpoint=checkpoint, mode=mode)
    env_updates = {
        "MERTFORMER_ONEFILE_PROFILE": str(spec["legacy_profile"]),
        "MERTFORMER_LOCAL_REPO_ROOT": str(ROOT),
        "MERTFORMER_OUTPUT_ROOT": str(layout.artifact_root),
    }
    if quick:
        env_updates["MERTFORMER_ONEFILE_FORCE_QUICK"] = "1"
    with patched_env(env_updates), patched_run_config(module, overrides), patched_argv([str((ROOT / LEGACY_SCRIPT_MAP[str(spec['legacy_lane'])][0]).resolve())]):
        payload = module.run_all()
    return payload


def find_checkpoint(explicit: str, layout: Layout) -> Optional[Path]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    manifest_path = layout.checkpoint_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key in ("latest", "best"):
                value = str(manifest.get(key, "")).strip()
                if value:
                    candidates.append(Path(value).expanduser())
        except Exception:
            pass
    candidates.extend([
        layout.checkpoint_dir / "latest.pt",
        layout.checkpoint_dir / "best.pt",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def infer_run_dir_from_payload(payload: dict[str, Any], fallback: Path) -> Path:
    output_files = payload.get("output_files", {}) if isinstance(payload, dict) else {}
    for key in ("json", "summary_json", "public_summary", "artifacts_index", "step_metrics_csv"):
        value = str(output_files.get(key, "")).strip()
        if value:
            return Path(value).expanduser().resolve().parent
    return fallback


def find_latest_run_dir(runtime: RuntimeMeta, artifact_root: Path, explicit: str) -> Optional[Path]:
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.exists() else None
    base = artifact_root / "runs" if runtime.kaggle else artifact_root
    if not base.exists():
        return None
    dirs = [path for path in base.iterdir() if path.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda item: item.stat().st_mtime, reverse=True)[0]


def run_command(cmd: list[str], env: Optional[dict[str, str]] = None) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, check=False)
    return {
        "cmd": " ".join(cmd),
        "return_code": proc.returncode,
        "ok": proc.returncode == 0,
        "elapsed_sec": round(time.time() - started, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def run_compare_job(layout: Layout, device: str) -> dict[str, Any]:
    compare_dir = layout.auxiliary_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/kaggle_train_compare_build30.py",
        "--quick",
        "--steps",
        os.environ.get("MERTFORMER_CANON_COMPARE_STEPS", ALWAYS_ON_COMPARE_ARGS["steps"]),
        "--batch-size",
        os.environ.get("MERTFORMER_CANON_COMPARE_BATCH_SIZE", ALWAYS_ON_COMPARE_ARGS["batch_size"]),
        "--seq-len",
        os.environ.get("MERTFORMER_CANON_COMPARE_SEQ_LEN", ALWAYS_ON_COMPARE_ARGS["seq_len"]),
        "--vocab-size",
        os.environ.get("MERTFORMER_CANON_COMPARE_VOCAB", ALWAYS_ON_COMPARE_ARGS["vocab_size"]),
        "--hidden",
        os.environ.get("MERTFORMER_CANON_COMPARE_HIDDEN", ALWAYS_ON_COMPARE_ARGS["hidden"]),
        "--layers",
        os.environ.get("MERTFORMER_CANON_COMPARE_LAYERS", ALWAYS_ON_COMPARE_ARGS["layers"]),
        "--heads",
        os.environ.get("MERTFORMER_CANON_COMPARE_HEADS", ALWAYS_ON_COMPARE_ARGS["heads"]),
        "--out-dir",
        str(compare_dir),
        "--device",
        os.environ.get("MERTFORMER_CANON_COMPARE_DEVICE", device if device in {"cpu", "mps", "cuda"} else "cpu"),
    ]
    result = run_command(cmd)
    result["out_dir"] = str(compare_dir)
    return result


def run_text_understanding_job(layout: Layout, quick: bool) -> dict[str, Any]:
    text_root = layout.auxiliary_dir / "text_understanding"
    env = os.environ.copy()
    env["MERTFORMER_OUTPUT_ROOT"] = str(text_root)
    if quick:
        env["MERTFORMER_TEXT_POC_QUICK"] = "1"
    cmd = [sys.executable, "scripts/kaggle_onefile_demo_build30_text_understanding.py"]
    if quick:
        cmd.append("--quick")
    result = run_command(cmd, env=env)
    run_base = text_root / "runs"
    latest = None
    if run_base.exists():
        dirs = [path for path in run_base.iterdir() if path.is_dir()]
        if dirs:
            latest = sorted(dirs, key=lambda item: item.stat().st_mtime, reverse=True)[0]
    result["out_dir"] = str(latest or run_base)
    return result


def _pick_loss_keys(fieldnames: Iterable[str]) -> tuple[Optional[str], Optional[str]]:
    lowered = {field.lower(): field for field in fieldnames}
    step_key = None
    for candidate in ("step", "global_step"):
        if candidate in lowered:
            step_key = lowered[candidate]
            break
    loss_key = None
    for candidate in ("train_loss", "loss", "total_loss", "smoothed_loss"):
        if candidate in lowered:
            loss_key = lowered[candidate]
            break
    if loss_key is None:
        for field in fieldnames:
            if "loss" in field.lower():
                loss_key = field
                break
    return step_key, loss_key


def build_first100_snapshot(run_dir: Path, out_path: Path) -> dict[str, Any]:
    candidates = sorted(run_dir.glob("*_step_metrics.csv"))
    if not candidates:
        candidates = sorted(run_dir.glob("*.csv"))
    rows: list[dict[str, Any]] = []
    source = None
    for candidate in candidates:
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                step_key, loss_key = _pick_loss_keys(reader.fieldnames)
                if loss_key is None:
                    continue
                for idx, row in enumerate(reader):
                    if idx >= 100:
                        break
                    try:
                        loss_val = float(row[loss_key])
                    except Exception:
                        continue
                    step_raw = row.get(step_key, idx + 1) if step_key else idx + 1
                    try:
                        step_val = int(float(step_raw))
                    except Exception:
                        step_val = idx + 1
                    rows.append({"step": step_val, "loss": loss_val})
                if rows:
                    source = candidate
                    break
        except Exception:
            continue
    payload = {
        "schema": "first_100_step_loss_snapshot_v1",
        "generated_at_utc": utc_now(),
        "source": str(source) if source else None,
        "rows": rows,
        "row_count": len(rows),
        "first_loss": rows[0]["loss"] if rows else None,
        "last_loss": rows[-1]["loss"] if rows else None,
        "loss_drop": (rows[0]["loss"] - rows[-1]["loss"]) if len(rows) >= 2 else None,
    }
    atomic_write_json(out_path, payload)
    return payload


def iter_artifact_files(run_dir: Path) -> Iterator[Path]:
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != ".DS_Store":
            yield path


def build_artifact_index(run_dir: Path, out_path: Path, extra_skip: Optional[set[Path]] = None) -> dict[str, Any]:
    skip = {path.resolve() for path in (extra_skip or set())}
    items = []
    for path in iter_artifact_files(run_dir):
        if path.resolve() in skip:
            continue
        rel = str(path.relative_to(run_dir))
        items.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema": "canonical_artifact_index_v1",
        "generated_at_utc": utc_now(),
        "root": str(run_dir),
        "count": len(items),
        "items": items,
    }
    atomic_write_json(out_path, payload)
    return payload


def build_sha256_manifest(index_payload: dict[str, Any], out_path: Path) -> None:
    lines = [f"{item['sha256']}  {item['path']}" for item in index_payload.get("items", [])]
    atomic_write_text(out_path, "\n".join(lines) if lines else "")


def build_claim_boundary_notes(checkpoint: Optional[Path]) -> list[str]:
    notes = [
        "This canonical lane is publish-ready and repo-side closure-safe; it is not an automatic product-quality claim.",
        "Floating Kaggle quotas and accelerator availability remain account-dependent and must be verified at runtime.",
        "Auxiliary compare/text surfaces are supportive evidence, not trained benchmark claims.",
    ]
    if checkpoint is None:
        notes.append("No trained checkpoint was resolved; benchmark-grade claims remain blocked.")
    else:
        notes.append(f"Checkpoint resolved at package time: {checkpoint}")
    return notes


def write_canonical_summary(
    *,
    mode: str,
    requested_profile: str,
    selected_profile: str,
    runtime: RuntimeMeta,
    layout: Layout,
    legacy_payload: Optional[dict[str, Any]],
    checkpoint: Optional[Path],
    compare_result: Optional[dict[str, Any]],
    text_result: Optional[dict[str, Any]],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    legacy_lane = PROFILE_SPECS[selected_profile]["legacy_lane"]
    legacy_profile = PROFILE_SPECS[selected_profile]["legacy_profile"]
    payload = {
        "schema": SCHEMA,
        "generated_at_utc": utc_now(),
        "mode": mode,
        "requested_profile": requested_profile,
        "selected_profile": selected_profile,
        "legacy_lane": legacy_lane,
        "legacy_profile": legacy_profile,
        "run_id": layout.run_id,
        "artifact_root": str(layout.artifact_root),
        "run_dir": str(layout.run_dir),
        "checkpoint_dir": str(layout.checkpoint_dir),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "runtime": {
            "kaggle": runtime.kaggle,
            "colab": runtime.colab,
            "device": runtime.device,
            "gpu_count": runtime.gpu_count,
            "gpu_names": list(runtime.gpu_names),
            "gpu_label": runtime.gpu_label,
        },
        "always_on_surfaces": {
            "first_100_step_loss_snapshot": str(layout.first100_snapshot_path),
            "compare_job": compare_result,
            "text_understanding_job": text_result,
        },
        "legacy_payload_files": (legacy_payload or {}).get("output_files", {}),
        "legacy_final_status": (legacy_payload or {}).get("final_status"),
        "legacy_final_reason": (legacy_payload or {}).get("final_reason"),
        "snapshot": snapshot,
        "claim_boundary": build_claim_boundary_notes(checkpoint),
    }
    atomic_write_json(layout.canonical_summary_path, payload)

    md = [
        "# Canonical Kaggle Closure Summary",
        "",
        f"- mode: `{mode}`",
        f"- requested_profile: `{requested_profile}`",
        f"- selected_profile: `{selected_profile}`",
        f"- legacy_lane: `{legacy_lane}`",
        f"- legacy_profile: `{legacy_profile}`",
        f"- run_id: `{layout.run_id}`",
        f"- runtime_gpu: `{runtime.gpu_label or 'none'}`",
        f"- checkpoint: `{checkpoint or 'none'}`",
        "",
        "## Claim Boundary",
    ]
    md.extend(f"- {line}" for line in payload["claim_boundary"])
    md.extend(
        [
            "",
            "## Auxiliary Surfaces",
            f"- first_100_step_loss_snapshot: `{layout.first100_snapshot_path.name}`",
            f"- compare_job_ok: `{(compare_result or {}).get('ok')}`",
            f"- text_understanding_ok: `{(text_result or {}).get('ok')}`",
        ]
    )
    atomic_write_text(layout.canonical_summary_md_path, "\n".join(md))
    return payload


def build_package_manifest(layout: Layout, summary: dict[str, Any], index_payload: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": "canonical_package_manifest_v1",
        "generated_at_utc": utc_now(),
        "run_id": layout.run_id,
        "run_dir": str(layout.run_dir),
        "bundle_path": str(layout.canonical_bundle_path),
        "summary_path": str(layout.canonical_summary_path),
        "artifact_count": index_payload.get("count", 0),
        "checkpoint": summary.get("checkpoint"),
        "selected_profile": summary.get("selected_profile"),
        "legacy_lane": summary.get("legacy_lane"),
    }
    atomic_write_json(layout.canonical_package_manifest_path, payload)
    return payload


def build_bundle(layout: Layout) -> None:
    layout.canonical_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(layout.canonical_bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in iter_artifact_files(layout.run_dir):
            if path.resolve() == layout.canonical_bundle_path.resolve():
                continue
            archive.write(path, arcname=str(path.relative_to(layout.run_dir)))


def verify_mode_payload(requested_profile: str, selected_profile: str, runtime: RuntimeMeta, layout: Layout) -> dict[str, Any]:
    checks = {
        "artifact_root_writable": probe_writable_dir(layout.artifact_root),
        "checkpoint_dir_writable": probe_writable_dir(layout.checkpoint_dir),
        "legacy_build30_exists": (ROOT / LEGACY_SCRIPT_MAP["build30"][0]).exists(),
        "legacy_onecell_t4_exists": (ROOT / LEGACY_SCRIPT_MAP["onecell_t4"][0]).exists(),
        "legacy_fastproof_exists": (ROOT / LEGACY_SCRIPT_MAP["fastproof"][0]).exists(),
        "compare_script_exists": (ROOT / "scripts/kaggle_train_compare_build30.py").exists(),
        "text_poc_exists": (ROOT / "scripts/kaggle_onefile_demo_build30_text_understanding.py").exists(),
    }
    payload = {
        "schema": SCHEMA,
        "generated_at_utc": utc_now(),
        "mode": "verify",
        "status": "completed" if all(checks.values()) else "failed",
        "requested_profile": requested_profile,
        "selected_profile": selected_profile,
        "runtime": {
            "kaggle": runtime.kaggle,
            "colab": runtime.colab,
            "device": runtime.device,
            "gpu_count": runtime.gpu_count,
            "gpu_names": list(runtime.gpu_names),
            "gpu_label": runtime.gpu_label,
        },
        "artifact_root": sanitize_text(str(layout.artifact_root)),
        "checkpoint_dir": sanitize_text(str(layout.checkpoint_dir)),
        "checks": checks,
        "claim_boundary": [
            "Verify mode checks runtime and canonical lane wiring only.",
            "No trained-evidence claim is produced by verify mode.",
        ],
    }
    atomic_write_json(layout.report_out, payload)
    return payload


def package_existing_run(
    *,
    requested_profile: str,
    selected_profile: str,
    runtime: RuntimeMeta,
    layout: Layout,
    checkpoint: Optional[Path],
) -> dict[str, Any]:
    snapshot = build_first100_snapshot(layout.run_dir, layout.first100_snapshot_path)
    summary = write_canonical_summary(
        mode="package-only",
        requested_profile=requested_profile,
        selected_profile=selected_profile,
        runtime=runtime,
        layout=layout,
        legacy_payload=None,
        checkpoint=checkpoint,
        compare_result=None,
        text_result=None,
        snapshot=snapshot,
    )
    index_payload = build_artifact_index(layout.run_dir, layout.canonical_artifact_index_path, extra_skip={layout.canonical_bundle_path})
    build_sha256_manifest(index_payload, layout.canonical_sha256_manifest_path)
    package_manifest = build_package_manifest(layout, summary, index_payload)
    build_bundle(layout)
    payload = {
        "schema": SCHEMA,
        "generated_at_utc": utc_now(),
        "mode": "package-only",
        "status": "completed" if checkpoint else "warning",
        "reason": None if checkpoint else "checkpoint_missing",
        "checkpoint": str(checkpoint) if checkpoint else None,
        "run_dir": str(layout.run_dir),
        "selected_profile": selected_profile,
        "package_manifest": package_manifest,
        "bundle_path": str(layout.canonical_bundle_path),
        "artifact_index": str(layout.canonical_artifact_index_path),
        "sha256_manifest": str(layout.canonical_sha256_manifest_path),
        "summary_path": str(layout.canonical_summary_path),
        "claim_boundary": build_claim_boundary_notes(checkpoint),
    }
    atomic_write_json(layout.report_out, payload)
    return payload


def maybe_refresh_repo_posttrain(checkpoint: Optional[Path]) -> dict[str, Any]:
    if checkpoint is None:
        # [2026-07-11] This used to return learning_rate/max_steps/warmup_ratio here --
        # hyperparameter fields that don't belong on a skipped-run status report and
        # silently dropped the real status fields (return_code/ok/stdout_tail) that
        # run_command() below actually produces. Restored to the same status schema
        # run_command() returns so callers can rely on one consistent shape.
        return {
            "cmd": "<skipped>",
            "return_code": 2,
            "ok": False,
            "elapsed_sec": 0.0,
            "stdout_tail": "",
            "stderr_tail": "checkpoint missing",
        }
    cmd = [
        sys.executable,
        "scripts/post_train_autorun.py",
        "--demo-only",
        "--allow-missing-checkpoint",
        "--checkpoint",
        str(checkpoint),
    ]
    return run_command(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical Kaggle closure lane for Build30.")
    parser.add_argument("--mode", default="train-end", choices=["train-end", "verify", "resume", "package-only", "bench-only"])
    parser.add_argument("--profile", default="auto", choices=["auto", *PROFILE_SPECS.keys()])
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--report-out", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--allow-missing-checkpoint", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Force quick/smoke legacy settings where supported.")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--skip-text-poc", action="store_true")
    parser.add_argument("--refresh-repo-posttrain", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = detect_runtime()
    selected_profile = choose_profile(args.profile, runtime)
    run_id = args.run_id.strip() or f"canon_{selected_profile}_{local_stamp()}"
    layout = build_paths(runtime, selected_profile, args.artifact_root, args.checkpoint_dir, run_id, args.report_out)

    if args.mode == "verify":
        payload = verify_mode_payload(args.profile, selected_profile, runtime, layout)
        print(json.dumps({"status": payload["status"], "selected_profile": selected_profile, "report": str(layout.report_out)}, ensure_ascii=False))
        return 0 if payload["status"] == "completed" else 1

    if args.mode == "package-only":
        run_dir = find_latest_run_dir(runtime, layout.artifact_root, args.run_dir)
        if run_dir is None:
            payload = {
                "schema": SCHEMA,
                "generated_at_utc": utc_now(),
                "mode": "package-only",
                "status": "failed",
                "reason": "run_dir_not_found",
            }
            atomic_write_json(layout.report_out, payload)
            print(json.dumps(payload, ensure_ascii=False))
            return 1
        layout = Layout(
            artifact_root=layout.artifact_root,
            checkpoint_dir=layout.checkpoint_dir,
            run_id=run_dir.name,
            run_dir=run_dir,
            auxiliary_dir=run_dir / "auxiliary",
            closure_dir=run_dir / "closure",
            report_out=(Path(args.report_out).expanduser() if args.report_out else (run_dir / "closure" / "canonical_status.json")),
            canonical_summary_path=run_dir / "closure" / "canonical_closure_summary.json",
            canonical_summary_md_path=run_dir / "closure" / "canonical_closure_summary.md",
            first100_snapshot_path=run_dir / "closure" / "first_100_step_loss_snapshot.json",
            canonical_artifact_index_path=run_dir / "closure" / "canonical_artifact_index.json",
            canonical_sha256_manifest_path=run_dir / "closure" / "canonical_sha256_manifest.txt",
            canonical_package_manifest_path=run_dir / "closure" / "canonical_package_manifest.json",
            canonical_bundle_path=run_dir / "closure" / f"{run_dir.name}_canonical_bundle.zip",
        )
        checkpoint = find_checkpoint(args.checkpoint, layout)
        payload = package_existing_run(
            requested_profile=args.profile,
            selected_profile=selected_profile,
            runtime=runtime,
            layout=layout,
            checkpoint=checkpoint,
        )
        print(json.dumps({"status": payload["status"], "bundle": payload["bundle_path"], "checkpoint": payload["checkpoint"]}, ensure_ascii=False))
        return 0 if checkpoint is not None or args.allow_missing_checkpoint else 2

    checkpoint = find_checkpoint(args.checkpoint, layout)
    if args.mode == "bench-only" and checkpoint is None and not args.allow_missing_checkpoint:
        payload = {
            "schema": SCHEMA,
            "generated_at_utc": utc_now(),
            "mode": "bench-only",
            "status": "failed",
            "reason": "checkpoint_missing",
        }
        atomic_write_json(layout.report_out, payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 2

    if args.mode in {"train-end", "resume"}:
        try:
            legacy_payload = execute_legacy_lane(selected_profile, layout, args.mode, args.checkpoint, args.quick)
        except Exception as exc:
            payload = {
                "schema": SCHEMA,
                "generated_at_utc": utc_now(),
                "mode": args.mode,
                "status": "failed",
                "reason": f"legacy_lane_exception:{type(exc).__name__}",
                "traceback": traceback.format_exc(),
            }
            atomic_write_json(layout.report_out, payload)
            print(json.dumps({"status": "failed", "reason": payload["reason"]}, ensure_ascii=False))
            return 1
        run_dir = infer_run_dir_from_payload(legacy_payload, layout.run_dir)
    else:
        legacy_payload = None
        run_dir = find_latest_run_dir(runtime, layout.artifact_root, args.run_dir)
        if run_dir is None:
            payload = {
                "schema": SCHEMA,
                "generated_at_utc": utc_now(),
                "mode": args.mode,
                "status": "failed",
                "reason": "run_dir_not_found",
            }
            atomic_write_json(layout.report_out, payload)
            print(json.dumps(payload, ensure_ascii=False))
            return 1

    layout = Layout(
        artifact_root=layout.artifact_root,
        checkpoint_dir=layout.checkpoint_dir,
        run_id=run_dir.name,
        run_dir=run_dir,
        auxiliary_dir=run_dir / "auxiliary",
        closure_dir=run_dir / "closure",
        report_out=(Path(args.report_out).expanduser() if args.report_out else (run_dir / "closure" / "canonical_status.json")),
        canonical_summary_path=run_dir / "closure" / "canonical_closure_summary.json",
        canonical_summary_md_path=run_dir / "closure" / "canonical_closure_summary.md",
        first100_snapshot_path=run_dir / "closure" / "first_100_step_loss_snapshot.json",
        canonical_artifact_index_path=run_dir / "closure" / "canonical_artifact_index.json",
        canonical_sha256_manifest_path=run_dir / "closure" / "canonical_sha256_manifest.txt",
        canonical_package_manifest_path=run_dir / "closure" / "canonical_package_manifest.json",
        canonical_bundle_path=run_dir / "closure" / f"{run_dir.name}_canonical_bundle.zip",
    )
    checkpoint = find_checkpoint(args.checkpoint, layout)
    compare_result = None if args.skip_compare else run_compare_job(layout, runtime.device)
    text_result = None if args.skip_text_poc else run_text_understanding_job(layout, quick=True)
    snapshot = build_first100_snapshot(layout.run_dir, layout.first100_snapshot_path)
    summary = write_canonical_summary(
        mode=args.mode,
        requested_profile=args.profile,
        selected_profile=selected_profile,
        runtime=runtime,
        layout=layout,
        legacy_payload=legacy_payload,
        checkpoint=checkpoint,
        compare_result=compare_result,
        text_result=text_result,
        snapshot=snapshot,
    )
    index_payload = build_artifact_index(layout.run_dir, layout.canonical_artifact_index_path, extra_skip={layout.canonical_bundle_path})
    build_sha256_manifest(index_payload, layout.canonical_sha256_manifest_path)
    package_manifest = build_package_manifest(layout, summary, index_payload)
    build_bundle(layout)
    repo_posttrain = maybe_refresh_repo_posttrain(checkpoint) if args.refresh_repo_posttrain else None

    payload = {
        "schema": SCHEMA,
        "generated_at_utc": utc_now(),
        "mode": args.mode,
        "status": (
            "warning"
            if args.mode == "bench-only" and checkpoint is None
            else "completed"
        ),
        "requested_profile": args.profile,
        "selected_profile": selected_profile,
        "run_dir": str(layout.run_dir),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "canonical_summary": str(layout.canonical_summary_path),
        "bundle_path": str(layout.canonical_bundle_path),
        "artifact_index": str(layout.canonical_artifact_index_path),
        "sha256_manifest": str(layout.canonical_sha256_manifest_path),
        "package_manifest": package_manifest,
        "legacy_final_status": (legacy_payload or {}).get("final_status") if isinstance(legacy_payload, dict) else None,
        "legacy_final_reason": (legacy_payload or {}).get("final_reason") if isinstance(legacy_payload, dict) else None,
        "compare_result": compare_result,
        "text_result": text_result,
        "repo_posttrain": repo_posttrain,
        "claim_boundary": build_claim_boundary_notes(checkpoint),
    }
    atomic_write_json(layout.report_out, payload)
    print(json.dumps({"status": payload["status"], "selected_profile": selected_profile, "run_dir": str(layout.run_dir), "checkpoint": payload["checkpoint"]}, ensure_ascii=False))
    if args.mode == "bench-only" and checkpoint is None and not args.allow_missing_checkpoint:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
