#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-file, zero-argument pre-flight validation run.

PURPOSE (read this first)
-------------------------
This script validates the TRAINING INFRASTRUCTURE of the repository at
REPO_DIR end-to-end on one 8 GB laptop GPU: dataset staging -> config ->
the repository's own training entry point -> checkpoints -> logs -> one
output ZIP. It does NOT validate model quality. At this scale (~36M
parameters, ~30M tokens, ternary-quantized linear layers plus sparse
expert routing plus a recurrent mixer layer) the resulting checkpoint
will generate mostly incoherent, partially-grammatical fragments, and
loss spikes / expert-routing imbalance / NaN-skip events are plausible.
Those are recorded as observations. The PASS/FAIL verdict is strictly
about whether the pipeline runs to completion and writes verifiable
artifacts (checkpoints, hash-chained logs, manifests).

WHAT IT REUSES FROM THE REPOSITORY (unchanged, source of truth)
---------------------------------------------------------------
* config.config.cfg          - the single config object. It is mutated
                               post-import, which is the repository's own
                               established pattern for small runs. No
                               architecture code is re-implemented here.
* train.train.train()        - the canonical training loop: offline
                               fallback dataset path, validation loop,
                               rotating checkpoints, auto-resume, NaN /
                               OOM / spike guards, and the repository's
                               RunLogger (hash-chained JSONL + CSV +
                               manifest) exactly as implemented there.
* utils.logger.try_git_commit- provenance stamping for the config record.

WHAT THIS FILE ADDS (glue only)
-------------------------------
* zero-argument entry point and environment staging
* dependency bootstrap: missing third-party libraries are pip-installed
  once at startup; the GPU stack (torch / torchvision / torchaudio /
  triton / bitsandbytes) is NEVER installed, upgraded, or touched - if
  torch is absent or CUDA unavailable, the script stops with instructions
* repository auto-discovery across a few sensible local locations when
  the configured REPO_DIR does not contain the repo
* a preflight report (one PASS/FAIL line per requirement) that stops
  before any training on hard failures
* optional repository clone (CLONE_REPO toggle; token via env var only)
* one-time dataset download + deterministic ~30M-token slice + manifest
* process-wide offline lock after setup (no network during training)
* artifact collection, log-chain verification, REPORT.md, output ZIP
* cleanup so nothing created by this run remains outside the ZIP once a
  terminal outcome is reached

NETWORK POLICY
--------------
The network is used at most once, at the very start, for (a) installing
any missing third-party libraries, (b) the optional repository clone,
(c) the tokenizer files, (d) the dataset slice. Before the training
phase begins, hub/dataset offline flags are set for the whole process.
Experiment-tracking uploads are disabled.

RESUME / IDEMPOTENCY
--------------------
* Output ZIP already present -> exit immediately (delete it to rerun).
* Crash or Ctrl+C            -> work directory and staged files are kept;
                               rerunning this script resumes (the trainer
                               auto-resumes from its *_latest.pt file and
                               the dataset slice is reused if its hashes
                               still match).
* Terminal outcome           -> ZIP written and verified, then every file
                               this run created (work directory + files
                               staged into the repository tree) is
                               removed, and any pre-existing files that
                               were set aside from the trainer's data
                               paths (curriculum stage files, fallback /
                               validation files) are restored verbatim.
                               Pre-existing repository files are never
                               deleted; a few small trainer-owned
                               report files may be refreshed in place
                               (disclosed in REPORT.md).

EXIT CODES
----------
0 = verdict PASS
1 = terminal run but verdict FAIL (ZIP still produced)
2 = setup/runtime error (state kept so a rerun can resume)
130 = interrupted (state kept so a rerun can resume)
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

# =====================================================================
# Repository acquisition toggle (the only intended user-editable knobs)
# =====================================================================
CLONE_REPO = False
REPO_URL = "https://example.invalid/your-org/your-repo.git"  # set before enabling CLONE_REPO
GIT_TOKEN_ENV = "GIT_TOKEN"  # env var read for private-repo auth; never hardcoded

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR / "repo"  # configured default; auto-discovery below may
                                 # rebind this (and the derived dataset paths)
                                 # to the first location containing the repo.

# =====================================================================
# Fixed run configuration (single 8 GB VRAM / 32 GB RAM laptop GPU)
# =====================================================================
MODEL_NAME = "infra_preflight"

HIDDEN_SIZE = 384
NUM_LAYERS = 6
NUM_HEADS = 6
NUM_KV_HEADS = 3          # grouped-query attention retained (6 query / 3 kv heads)
HEAD_DIM = 64             # 384 == 6 * 64
FFN_INTERMEDIATE = 1024   # dense FFN width
MOE_EXPERTS = 8
MOE_ACTIVE = 2            # top-k experts per token
MOE_EVERY_N_LAYERS = 3    # expert layers at block ids 2 and 5 (1-indexed % 3 == 0)
MOE_INTERMEDIATE = 768    # per-expert FFN width
LIQUID_LAYER_IDS = [3]    # recurrent mixer layer id (disjoint from expert layers)
LIQUID_WARMUP_STEPS = 200 # repo default (10000) exceeds this run; scaled to fit

SEQ_LEN = 512
MICRO_BATCH = 16
GRAD_ACCUM = 2
GLOBAL_BATCH = MICRO_BATCH * GRAD_ACCUM  # 32 sequences / optimizer step
TOTAL_STEPS = 1830
SAVE_INTERVAL = 500
VAL_INTERVAL = 500
LOG_INTERVAL = 10
SEED = 1453  # repository default, kept for parity

TOKENS_PER_STEP = GLOBAL_BATCH * SEQ_LEN            # 16,384 token positions
PLANNED_TOKENS = TOTAL_STEPS * TOKENS_PER_STEP      # 29,982,720 (~30M, one pass)

# One-time-online assets. These identifiers are functional requirements
# (public hub artifact ids), not endorsements.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"  # small, clean, widely used instruction data
DATASET_SPLIT = "train_sft"
TOKENIZER_ID = "hf-internal-testing/llama-tokenizer"  # public 32k SentencePiece tokenizer;
# same ungated artifact the repository's own preflight uses as its tokenizer fallback.
VAL_ROWS = 2000  # held-out conversations written to the repo's validation path

# Host requirements
PY_MIN = (3, 10)          # repository uses PEP 604 unions without a future import
MIN_VRAM_GB = 7.0
MIN_FREE_DISK_GB = 12.0
MIN_RAM_GB_WARN = 24.0

# Third-party imports the repository and this harness need. Any that are
# missing are pip-installed once at startup (import name, pip requirement).
BOOTSTRAP_LIBS = (
    ("numpy", "numpy"),
    ("psutil", "psutil"),
    ("tqdm", "tqdm"),
    ("einops", "einops"),
    ("huggingface_hub", "huggingface_hub"),
    ("tokenizers", "tokenizers"),
    ("sentencepiece", "sentencepiece"),
    ("safetensors", "safetensors"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("accelerate", "accelerate"),
)
# The GPU stack must match the local CUDA setup and is NEVER installed,
# upgraded, or otherwise touched by this script.
FORBIDDEN_INSTALLS = ("torch", "torchvision", "torchaudio", "triton", "bitsandbytes")

# Marker files that identify the repository root during auto-discovery.
REPO_MARKERS = ("train/train.py", "config/config.py")

# Repository files that must exist for the run to be meaningful.
REQUIRED_REPO_PATHS = (
    "config/config.py",
    "model/transformers.py",
    "train/train.py",
    "train/trainer_core.py",
    "train/trainer_data.py",
    "train/trainer_eval.py",
    "train/packing.py",
    "layers/mertformer_block.py",
    "layers/bitlinear.py",
    "layers/moe.py",
    "layers/liquid.py",
    "layers/mla.py",
    "layers/ffn.py",
    "utils/logger.py",
    "utils/tokenizer_resolver.py",
    "utils/liquid_safeguard.py",
    "orchestrator/distillation_manager.py",
    "orchestrator/telemetry.py",
)

# Curriculum stage files: if present, the trainer would use them instead of
# this run's controlled slice, so they are set aside automatically (renamed
# in place) and restored verbatim at cleanup. Same for pre-existing files on
# the fallback/validation paths.
STAGE_FILE_RELPATHS = (
    "datasets/stage1/stage1_data.jsonl",
    "datasets/stage2/stage2_data.jsonl",
    "datasets/stage3/stage3_data.jsonl",
    "datasets/stage4_soul/stage4_data.jsonl",
    "datasets/stage5_tools/stage5_data.jsonl",
)
SIDELINE_SUFFIX = ".preflight_sidelined"  # reversible rename marker; never deleted

# Repository directories this run may write into (cleanup is limited to
# the created-file diff inside these directories).
BASELINE_DIRS = ("datasets", "logs", "reports")
KNOWN_REPORT_FILES = (
    "system_stats.jsonl",
    "energy_baseline.json",
    "latency_baseline.json",
    "thermal_baseline.json",
    "training_runtime_manifest.json",
)

# =====================================================================
# Derived paths (everything transient lives under WORK_DIR)
# =====================================================================
WORK_DIR = SCRIPT_DIR / "preflight_work"
HF_CACHE_DIR = WORK_DIR / "hf_cache"
TOKENIZER_DIR = WORK_DIR / "tokenizer"  # plain local snapshot; the run's single
                                         # tokenizer source after the first download
                                         # (independent of hub-cache internals)
CKPT_DIR = WORK_DIR / "checkpoints"
STATE_PATH = WORK_DIR / "state.json"
CONSOLE_LOG_PATH = WORK_DIR / "console.log"
DATASET_MANIFEST_PATH = WORK_DIR / "dataset_manifest.json"
RUN_CONFIG_PATH = WORK_DIR / "run_config.json"
METRICS_PATH = WORK_DIR / "metrics.json"
REPORT_PATH = WORK_DIR / "REPORT.md"
OUTPUT_ZIP = SCRIPT_DIR / "preflight_run_output.zip"

TRAIN_JSONL = REPO_DIR / "datasets" / "training_data.jsonl"   # trainer's offline fallback path
VAL_JSONL = REPO_DIR / "datasets" / "validation.jsonl"        # trainer's validation path
# TRAIN_JSONL / VAL_JSONL are rebound by _set_repo_dir() if auto-discovery
# resolves the repository somewhere other than the configured REPO_DIR.

STATE_SCHEMA = 1


class OrchestrationError(RuntimeError):
    """User-actionable failure; message must say exactly what to do."""


# =====================================================================
# Small utilities
# =====================================================================
def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[preflight {_ts()}] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[preflight {_ts()}] WARNING: {msg}", flush=True)


def fail(msg: str) -> None:
    raise OrchestrationError(msg)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} PiB"


def human_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h else f"{m:d}m {s:02d}s"


def load_state() -> dict:
    state = read_json(STATE_PATH)
    if state.get("schema") != STATE_SCHEMA:
        state = {"schema": STATE_SCHEMA}
    return state


def save_state(state: dict) -> None:
    write_json(STATE_PATH, state)


def snapshot_repo_tree() -> dict:
    """Record which files/dirs exist under the writable repo dirs pre-run."""
    files: dict[str, list[str]] = {}
    dirs_present: list[str] = []
    for d in BASELINE_DIRS:
        root = REPO_DIR / d
        if root.exists():
            dirs_present.append(d)
            files[d] = sorted(
                str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
            )
        else:
            files[d] = []
    return {"files": files, "dirs_present": dirs_present}


# =====================================================================
# Console tee: mirror all stdout/stderr into WORK_DIR/console.log so the
# repository's own console output (progress, guard messages) is archived.
# =====================================================================
class _Tee(io.TextIOBase):
    def __init__(self, stream, fh):
        self._s = stream
        self._f = fh

    def write(self, s):  # noqa: D102 - thin passthrough
        try:
            self._s.write(s)
        except Exception:
            pass
        try:
            self._f.write(s)
        except Exception:
            pass
        return len(s)

    def flush(self):
        for t in (self._s, self._f):
            try:
                t.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


_TEE = {"installed": False, "fh": None, "out": None, "err": None}


def install_tee() -> None:
    if _TEE["installed"]:
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    CONSOLE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = CONSOLE_LOG_PATH.open("a", encoding="utf-8", errors="replace")
    _TEE.update(installed=True, fh=fh, out=sys.stdout, err=sys.stderr)
    sys.stdout = _Tee(_TEE["out"], fh)
    sys.stderr = _Tee(_TEE["err"], fh)


def uninstall_tee() -> None:
    if not _TEE["installed"]:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = _TEE["out"]
    sys.stderr = _TEE["err"]
    try:
        _TEE["fh"].close()
    except Exception:
        pass
    _TEE.update(installed=False, fh=None)


# =====================================================================
# Phase 0: environment staging (must run before torch / repo imports)
# =====================================================================
def stage_environment() -> dict[str, str]:
    """Set every switch the repository reads at import/run time.

    The trainer, config and tokenizer resolver are configured entirely via
    their documented environment switches plus a small post-import config
    mutation (phase_config). Nothing in the repository is modified.
    """
    env = {
        # Hermetic caches + no telemetry / tracker uploads.
        "HF_HOME": str(HF_CACHE_DIR),
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "WANDB_DISABLED": "true",
        "WANDB_MODE": "offline",
        # Repository switches (all documented in the repo's config/trainer).
        "MERTFORMER_LOGBOOK": "0",   # skip the shared append-only logbook so
                                      # cleanup can restore the repo exactly;
                                      # per-run hash-chained JSONL is archived.
        "TITAN_CONFIG_VERBOSE": "0",
        "TITAN_OFFLINE": "1",        # activates the single-file fallback dataset path
        "TITAN_AUTO_RESUME": "1",    # trainer resumes from *_latest.pt on rerun
        "TITAN_MAX_STEPS": str(TOTAL_STEPS),
        "TITAN_BATCH_SIZE": str(GLOBAL_BATCH),
        "TITAN_SAVE_INTERVAL": str(SAVE_INTERVAL),
        "TITAN_VAL_CHECK_INTERVAL": str(VAL_INTERVAL),
        "TITAN_LOG_INTERVAL": str(LOG_INTERVAL),
        "TITAN_VAL_STEPS": "10",
        "TITAN_TOKEN_BUDGET_MODE": "fixed_steps",
        "TITAN_USE_PRECOMPUTED_LOGITS": "0",   # no offline-logit KD in this run
        "TITAN_DISTILL_ALPHA": "0",            # tokenizer only; no teacher model load
        "TITAN_REQUIRE_GATED_TEACHER": "0",
        "TITAN_TEACHER_MODEL_ID": str(TOKENIZER_DIR),  # local tokenizer snapshot
        # (materialized in phase_tokenizer before the trainer ever reads it)
        "TITAN_USE_TR_TOKENIZER": "0",
        "TITAN_LIQUID_FAST_PATH": "0",          # no graph compile of the recurrent loop
        "TITAN_LIQUID_TRAIN_IMPL": "packed_pair",  # fused input/recurrent projections
        "TITAN_MOE_DISPATCH": "parallel",
        "TITAN_TELEMETRY_INTERVAL": "100",
        "TITAN_STRICT_TOKEN_BUDGET": "0",
        "TITAN_DATALOADER_PIN": "1",
        "TITAN_DATALOADER_NONBLOCKING": "1",
        "TITAN_WANDB": "0",
    }
    os.environ.update(env)
    if os.name != "nt":
        # Allocator hint for a small-VRAM device (set before CUDA init).
        # Windows CUDA allocator builds do not support expandable_segments.
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def lock_offline() -> None:
    """Forbid any further hub/dataset network access for this process."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    log("Offline lock engaged: no network access from this point on.")


# =====================================================================
# Phase 1a: dependency bootstrap (before any repository import)
# =====================================================================
def bootstrap_dependencies() -> None:
    """Install missing third-party libraries once; never touch the GPU stack.

    Policy: torch / torchvision / torchaudio / triton / bitsandbytes must
    match the local CUDA setup and are never installed, upgraded, or
    modified here. If torch is absent or CUDA is unavailable, this fails
    early with instructions instead of attempting an install.
    """
    if importlib.util.find_spec("torch") is None:
        fail(
            "torch is not installed. It is deliberately NOT auto-installed: the "
            "build must match your CUDA driver/toolkit. Install the correct "
            "CUDA-enabled torch wheel yourself (see pytorch.org), then rerun."
        )
    import torch  # noqa: WPS433

    if not torch.cuda.is_available():
        fail(
            "torch imports but torch.cuda.is_available() is False; this run "
            "requires the GPU. Install a CUDA-enabled torch build matching your "
            "driver (this script never modifies torch itself), then rerun."
        )

    # Defensive guard: refuse to ever place a GPU-stack package on the
    # install line, even if the list above is edited carelessly.
    for _, req in BOOTSTRAP_LIBS:
        base = re.split(r"[<>=!\[]", req, maxsplit=1)[0].strip().lower()
        if base in FORBIDDEN_INSTALLS:
            fail(f"Internal guard: refusing to auto-install GPU-stack package {base!r}.")

    missing = [req for name, req in BOOTSTRAP_LIBS if importlib.util.find_spec(name) is None]
    if not missing:
        log("Dependencies OK: all required libraries already installed.")
        return

    log(f"Installing missing libraries (one-time): {', '.join(missing)}")
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *missing],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-12:])
        fail(
            "Automatic dependency install failed. Install manually, then rerun:\n"
            f"  {sys.executable} -m pip install {' '.join(missing)}\n"
            f"pip output (tail):\n{tail}"
        )
    importlib.invalidate_caches()
    still_missing = [name for name, _ in BOOTSTRAP_LIBS if importlib.util.find_spec(name) is None]
    if still_missing:
        fail(
            "Libraries still unimportable after install: "
            + ", ".join(still_missing)
            + ". Check the active interpreter/venv and install manually."
        )
    log("Dependency install complete.")


# =====================================================================
# Phase 1b: repository auto-discovery
# =====================================================================
def _is_repo_root(path: Path) -> bool:
    try:
        return path.is_dir() and all((path / rel).exists() for rel in REPO_MARKERS)
    except OSError:
        return False


def _set_repo_dir(path: Path) -> None:
    """Rebind the repo root and every path derived from it."""
    global REPO_DIR, TRAIN_JSONL, VAL_JSONL
    REPO_DIR = path
    TRAIN_JSONL = REPO_DIR / "datasets" / "training_data.jsonl"
    VAL_JSONL = REPO_DIR / "datasets" / "validation.jsonl"


def discover_repo() -> tuple[Path | None, list[Path]]:
    """Return (first location containing the repo markers, locations searched).

    Search order: configured REPO_DIR, the script directory itself, ./repo
    next to the script, the current working directory, ./repo under it, the
    parents of both, then the immediate subdirectories of the script and
    working directories.
    """
    candidates: list[Path] = [
        REPO_DIR,
        SCRIPT_DIR,
        SCRIPT_DIR / "repo",
        Path.cwd(),
        Path.cwd() / "repo",
        SCRIPT_DIR.parent,
        Path.cwd().parent,
    ]
    for base in (SCRIPT_DIR, Path.cwd()):
        try:
            candidates.extend(sorted(p for p in base.iterdir() if p.is_dir()))
        except OSError:
            pass

    searched: list[Path] = []
    seen: set[Path] = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        searched.append(resolved)
        if _is_repo_root(resolved):
            return resolved, searched
    return None, searched


# =====================================================================
# Phase 1c: preflight requirement report (hard stop before anything runs)
# =====================================================================
def phase_preflight(state: dict, repo_path: Path | None, searched: list[Path]) -> dict:
    """Print a PASS/FAIL line per requirement, then stop on any hard failure."""
    failures: list[str] = []

    def check(ok: bool, label: str, detail: str, fix: str | None = None) -> bool:
        print(f"[preflight]   [{'PASS' if ok else 'FAIL'}] {label}: {detail}", flush=True)
        if not ok:
            failures.append(fix or f"{label}: {detail}")
        return ok

    def soft(ok: bool, label: str, detail: str) -> None:
        print(f"[preflight]   [{'PASS' if ok else 'WARN'}] {label}: {detail}", flush=True)

    def info(label: str, detail: str) -> None:
        print(f"[preflight]   [INFO] {label}: {detail}", flush=True)

    log("Preflight requirement report:")

    check(
        sys.version_info >= PY_MIN,
        "python",
        f"{platform.python_version()} (need >= {PY_MIN[0]}.{PY_MIN[1]})",
        fix=f"Install Python {PY_MIN[0]}.{PY_MIN[1]}+ (the repository uses modern typing syntax at import time).",
    )

    props = None
    total_vram_gb = 0.0
    torch_ok = importlib.util.find_spec("torch") is not None
    check(
        torch_ok,
        "torch",
        "importable" if torch_ok else "NOT installed",
        fix="Install a CUDA-enabled torch build matching your driver (see pytorch.org); this script never installs torch.",
    )
    if torch_ok:
        import torch  # noqa: WPS433

        cuda_ok = torch.cuda.is_available()
        check(
            cuda_ok,
            "cuda",
            "torch.cuda.is_available() == True" if cuda_ok else "torch.cuda.is_available() == False",
            fix="Install a CUDA-enabled torch build matching your driver; a CPU-only torch cannot run this.",
        )
        if cuda_ok:
            props = torch.cuda.get_device_properties(0)
            total_vram_gb = props.total_memory / 2**30
            check(
                total_vram_gb >= MIN_VRAM_GB,
                "gpu",
                f"{props.name} (sm_{props.major}{props.minor}, {total_vram_gb:.1f} GB VRAM; need >= {MIN_VRAM_GB:.1f})",
                fix=f"This fixed configuration is sized for an 8 GB device; found {total_vram_gb:.1f} GB.",
            )
            if MIN_VRAM_GB <= total_vram_gb < 7.8:
                soft(False, "gpu_margin", f"VRAM reports {total_vram_gb:.1f} GB (< 8.0); margins are still safe")
            smoke_ok, smoke_detail = True, "64x64 matmul on device OK"
            try:
                x = torch.randn(64, 64, device="cuda")
                if not math.isfinite(float((x @ x).float().sum().item())):
                    raise RuntimeError("non-finite result")
                del x
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001
                smoke_ok, smoke_detail = False, f"failed ({exc})"
            check(
                smoke_ok,
                "cuda_smoke",
                smoke_detail,
                fix="The installed torch build likely does not support this GPU's compute capability; install a newer CUDA torch wheel.",
            )
            check(
                torch.cuda.is_bf16_supported(),
                "bf16",
                "supported" if torch.cuda.is_bf16_supported() else "NOT supported",
                fix="This run requires bf16 mixed precision; this GPU/torch build does not support it.",
            )

    if repo_path is not None:
        markers_ok = _is_repo_root(repo_path)
        check(
            markers_ok,
            "repository",
            f"found at {repo_path} (key modules present: {', '.join(REPO_MARKERS)})",
        )
    elif CLONE_REPO:
        info("repository", f"not found locally; will be cloned from REPO_URL into {REPO_DIR}")
    else:
        check(
            False,
            "repository",
            "not found",
            fix=(
                "Repository not found. Searched:\n      "
                + "\n      ".join(str(p) for p in searched)
                + "\n    Place the repository (containing "
                + " and ".join(REPO_MARKERS)
                + ") at one of these locations, edit REPO_DIR at the top of this "
                f"file, or set CLONE_REPO=True with REPO_URL and export {GIT_TOKEN_ENV}."
            ),
        )

    free_gb = shutil.disk_usage(str(SCRIPT_DIR)).free / 2**30
    check(
        free_gb >= MIN_FREE_DISK_GB,
        "disk",
        f"{free_gb:.1f} GB free (need >= {MIN_FREE_DISK_GB:.0f} for dataset slice + checkpoints + output ZIP)",
        fix=f"Free at least {MIN_FREE_DISK_GB:.0f} GB on the disk holding {SCRIPT_DIR}.",
    )

    ram_gb = 0.0
    if importlib.util.find_spec("psutil") is not None:
        import psutil  # noqa: WPS433

        ram_gb = psutil.virtual_memory().total / 2**30
        soft(
            ram_gb >= MIN_RAM_GB_WARN,
            "ram",
            f"{ram_gb:.1f} GB system RAM"
            + ("" if ram_gb >= MIN_RAM_GB_WARN else f" (< {MIN_RAM_GB_WARN:.0f} GB; run should still fit)"),
        )

    if repo_path is None:
        info("dataset_slice", "resolved after the repository is available")
    elif state.get("dataset_ready") and TRAIN_JSONL.exists() and VAL_JSONL.exists():
        info("dataset_slice", "present from a previous attempt; will be SHA256-verified and reused")
    elif TRAIN_JSONL.exists() or VAL_JSONL.exists():
        info(
            "dataset_slice",
            "pre-existing file(s) occupy the trainer's data paths; they will be "
            "set aside automatically and restored verbatim at cleanup",
        )
    else:
        info("dataset_slice", f"absent; ~{PLANNED_TOKENS/1e6:.0f}M tokens will be downloaded once at setup")

    if failures:
        fail("Preflight failed:\n  - " + "\n  - ".join(failures))
    log("Preflight report: all hard requirements PASS.")

    versions = {}
    for mod in ("torch", "transformers", "datasets", "accelerate", "psutil"):
        try:
            versions[mod] = importlib.import_module(mod).__version__
        except Exception:
            versions[mod] = "n/a"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **versions,
        "gpu_name": props.name if props else "n/a",
        "gpu_compute_capability": f"{props.major}.{props.minor}" if props else "n/a",
        "gpu_vram_gb": round(total_vram_gb, 2),
        "system_ram_gb": round(ram_gb, 1),
        "free_disk_gb": round(free_gb, 1),
    }


# =====================================================================
# Phase 2: repository acquisition and validation
# =====================================================================
def _redact(text: str, secret: str | None) -> str:
    return text.replace(secret, "***") if secret else text


def phase_repo(repo_path: Path | None, searched: list[Path]) -> None:
    if repo_path is not None:
        _set_repo_dir(repo_path)
        if CLONE_REPO:
            log(f"Repository already present at {REPO_DIR}; skipping clone.")
        else:
            log(f"Using repository at {REPO_DIR}.")
    else:
        if not CLONE_REPO:
            # phase_preflight already reported this as a hard failure; this
            # branch only guards against being called out of order.
            fail(
                "Repository not found. Searched:\n  "
                + "\n  ".join(str(p) for p in searched)
                + "\nPlace the repository (containing "
                + " and ".join(REPO_MARKERS)
                + ") at one of these locations, edit REPO_DIR at the top of "
                f"this file, or set CLONE_REPO=True and export {GIT_TOKEN_ENV}."
            )
        if REPO_DIR.exists() and any(REPO_DIR.iterdir()):
            fail(
                f"CLONE_REPO=True but {REPO_DIR} exists and is not a valid "
                "repository. Remove it (or point REPO_DIR elsewhere) and rerun."
            )
        if "example.invalid" in REPO_URL:
            fail("CLONE_REPO=True: set REPO_URL at the top of this file first.")
        if not REPO_URL.startswith("https://"):
            fail("CLONE_REPO=True supports https:// URLs only (token auth).")
        token = os.environ.get(GIT_TOKEN_ENV, "").strip()
        if not token:
            fail(
                f"CLONE_REPO=True but the environment variable {GIT_TOKEN_ENV} "
                "is not set. Export a repository access token to it and rerun. "
                "The token is read from the environment only and never stored."
            )
        if shutil.which("git") is None:
            fail("git executable not found on PATH; install git and rerun.")
        auth_url = REPO_URL.replace("https://", f"https://{token}@", 1)
        log(f"Cloning repository into {REPO_DIR} (token redacted) ...")
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", auth_url, str(REPO_DIR)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            fail(
                "git clone failed:\n"
                + _redact(proc.stderr.strip() or proc.stdout.strip(), token)
            )
        _set_repo_dir(REPO_DIR)
        log("Clone complete.")

    missing = [rel for rel in REQUIRED_REPO_PATHS if not (REPO_DIR / rel).exists()]
    if missing:
        fail(
            "Repository is missing required files (wrong directory or partial "
            "checkout?):\n  " + "\n  ".join(missing)
        )

    sys.path.insert(0, str(REPO_DIR))
    # Confirm the key packages resolve from the repo root without executing
    # them (execution happens later, after the environment is fully staged).
    unresolvable = [
        pkg
        for pkg in ("config", "model", "train", "layers", "utils", "orchestrator")
        if importlib.util.find_spec(pkg) is None
    ]
    if unresolvable:
        fail(
            "Repository packages not importable from "
            f"{REPO_DIR}: {', '.join(unresolvable)}. Check for missing "
            "__init__.py files or a shadowing installed package."
        )
    log(f"Repository validated at {REPO_DIR} and added to sys.path.")


# =====================================================================
# Phase 3: tokenizer (one-time online, then a plain local snapshot)
# =====================================================================
def phase_tokenizer(state: dict):
    """Materialize the tokenizer as a plain local snapshot under WORK_DIR.

    The snapshot directory is the single tokenizer source for everything
    that follows - this harness and the trainer's own resolver (pointed at
    it via the teacher-model switch) - so resumes and the offline training
    phase never depend on hub-cache internals. A missing or unreadable
    snapshot self-heals by re-downloading once; no manual step is ever
    required.
    """
    from transformers import AutoTokenizer  # noqa: WPS433

    def load_snapshot():
        return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)

    tokenizer = None
    if (TOKENIZER_DIR / "tokenizer_config.json").exists():
        try:
            tokenizer = load_snapshot()
            log(f"Tokenizer loaded from local snapshot: {TOKENIZER_DIR}")
        except Exception as exc:  # noqa: BLE001
            warn(f"Local tokenizer snapshot unreadable ({exc}); re-downloading once.")
            shutil.rmtree(TOKENIZER_DIR, ignore_errors=True)

    if tokenizer is None:
        log(f"Downloading tokenizer '{TOKENIZER_ID}' (one-time) ...")
        try:
            fetched = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        except Exception as exc:  # noqa: BLE001
            fail(
                f"Tokenizer download failed for '{TOKENIZER_ID}'. Check the "
                "internet connection and simply rerun this script; it resumes "
                f"automatically from where it stopped. Detail: {exc}"
            )
        if fetched.pad_token is None:
            fetched.pad_token = fetched.eos_token
        shutil.rmtree(TOKENIZER_DIR, ignore_errors=True)
        TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
        fetched.save_pretrained(str(TOKENIZER_DIR))
        tokenizer = load_snapshot()  # verify the snapshot round-trips from disk
        log(f"Tokenizer snapshot written to {TOKENIZER_DIR}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    state["tokenizer_ready"] = True
    state["tokenizer_vocab"] = len(tokenizer)
    save_state(state)
    log(f"Tokenizer ready (vocab={len(tokenizer)}).")
    return tokenizer


# =====================================================================
# Phase 4: dataset slice (one-time online) -> trainer's expected paths
# =====================================================================
def _render_conversation(messages) -> str:
    """Flatten a multi-turn record into plain 'User:/Assistant:' text."""
    parts: list[str] = []
    for m in messages or []:
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        role = str(m.get("role") or "").strip().lower()
        label = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
            role, role.capitalize() or "Speaker"
        )
        parts.append(f"{label}: {content}")
    return "\n".join(parts)


def _sideline(path: Path, state: dict) -> None:
    """Move a pre-existing repository file out of the trainer's way.

    The file is renamed in place (same directory, atomic on the same
    filesystem) and restored verbatim by cleanup() once the run reaches a
    terminal outcome. Every move is recorded in the run state first, so
    resumed runs neither repeat it nor mistake our staged files for it.
    """
    backup = path.with_name(path.name + SIDELINE_SUFFIX)
    n = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}{SIDELINE_SUFFIX}.{n}")
        n += 1
    state.setdefault("sidelined", []).append({"orig": str(path), "backup": str(backup)})
    save_state(state)
    os.replace(path, backup)
    log(f"Pre-existing file set aside for this run (restored at cleanup): {path.name}")


def _guard_dataset_paths(state: dict) -> None:
    """Clear the trainer's data paths automatically, without touching user data.

    Pre-existing repository files on the paths the trainer reads (curriculum
    stage files, and the offline fallback / validation files) are renamed
    aside and restored verbatim by cleanup(). Partial slices left by an
    interrupted earlier attempt of this same run are removed.
    """
    sidelined_origs = {e["orig"] for e in state.get("sidelined", [])}
    for rel in STAGE_FILE_RELPATHS:
        p = REPO_DIR / rel
        if p.exists() and str(p) not in sidelined_origs:
            _sideline(p, state)
    if not state.get("dataset_ready"):
        baseline = set(
            state.get("baseline", {}).get("files", {}).get("datasets", [])
        )
        for p in (TRAIN_JSONL, VAL_JSONL):
            if not p.exists():
                continue
            rel = str(p.relative_to(REPO_DIR / "datasets"))
            if rel in baseline and str(p) not in sidelined_origs:
                _sideline(p, state)
            else:
                warn(f"Removing partial slice from an interrupted attempt: {p.name}")
                p.unlink()


def _resolve_dataset_revision(state: dict) -> str:
    """Pin the dataset to one exact upstream commit for the whole run.

    The commit sha of the dataset repository is resolved once, frozen into
    the run state, passed to every load, and recorded in the manifest and
    report, so the slice is immune to upstream changes (including across
    resumes) and byte-identical to regenerate by loading the recorded
    revision.
    """
    revision = state.get("dataset_revision")
    if revision:
        return revision
    from huggingface_hub import HfApi  # noqa: WPS433

    try:
        revision = HfApi().dataset_info(DATASET_ID).sha
    except Exception as exc:  # noqa: BLE001
        fail(
            f"Could not resolve the upstream commit of '{DATASET_ID}' "
            f"(network reachable?): {exc}"
        )
    if not revision:
        fail(f"Upstream returned no commit sha for '{DATASET_ID}'.")
    state["dataset_revision"] = revision
    save_state(state)
    log(f"Dataset revision pinned: {revision}")
    return revision


def phase_dataset(state: dict, tokenizer) -> dict:
    """Build the ~30M-token slice the trainer will consume.

    Selection rule (deterministic, stated exactly):
      * pin the dataset to one exact upstream commit (resolved once,
        frozen for the whole run, recorded in the manifest);
      * stream DATASET_ID split DATASET_SPLIT at that revision in
        upstream order (no shuffle);
      * render each conversation's messages into one plain-text row;
      * TRAIN file: accumulate rows until the sum of per-row trainable
        tokens, min(len(tokens), SEQ_LEN-1)+1 (i.e. after the trainer's
        truncate-to-511 + EOS), reaches PLANNED_TOKENS (29,982,720);
      * VALIDATION file: the next VAL_ROWS (2,000) conversations.
    The trainer samples this file uniformly with replacement, so the run is
    token-equivalent to one epoch, not a strict ordered pass.
    """
    _guard_dataset_paths(state)

    if state.get("dataset_ready"):
        hashes = state.get("dataset_hashes", {})
        if (
            TRAIN_JSONL.exists()
            and VAL_JSONL.exists()
            and sha256_file(TRAIN_JSONL) == hashes.get("train")
            and sha256_file(VAL_JSONL) == hashes.get("val")
        ):
            log("Dataset slice already staged and hash-verified; reusing it.")
            return read_json(DATASET_MANIFEST_PATH)
        warn(
            "Staged dataset slice is missing or modified; rebuilding it "
            "automatically from the pinned revision (byte-identical)."
        )
        state["dataset_ready"] = False
        state.pop("dataset_hashes", None)
        save_state(state)

    from datasets import load_dataset  # noqa: WPS433

    revision = _resolve_dataset_revision(state)
    log(
        f"Streaming '{DATASET_ID}' [{DATASET_SPLIT}] at revision {revision[:12]} "
        f"until {PLANNED_TOKENS:,} trainable tokens + {VAL_ROWS} validation rows ..."
    )
    stream = load_dataset(
        DATASET_ID, split=DATASET_SPLIT, streaming=True, revision=revision
    )

    TRAIN_JSONL.parent.mkdir(parents=True, exist_ok=True)
    st = {
        "mode": "train",
        "done": False,
        "train": {"rows": 0, "raw_tokens": 0, "trainable_tokens": 0, "truncated_rows": 0},
        "val": {"rows": 0, "raw_tokens": 0, "trainable_tokens": 0, "truncated_rows": 0},
        "last_log": time.time(),
    }
    cap = SEQ_LEN - 1  # trainer truncates to max_len-1 then appends EOS

    def consume(text: str, n_tokens: int, ftrain, fval) -> None:
        trainable = min(n_tokens, cap) + 1
        bucket = st[st["mode"]]
        line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
        (ftrain if st["mode"] == "train" else fval).write(line)
        bucket["rows"] += 1
        bucket["raw_tokens"] += n_tokens
        bucket["trainable_tokens"] += trainable
        if n_tokens > cap:
            bucket["truncated_rows"] += 1
        if st["mode"] == "train" and bucket["trainable_tokens"] >= PLANNED_TOKENS:
            st["mode"] = "val"
        elif st["mode"] == "val" and bucket["rows"] >= VAL_ROWS:
            st["done"] = True

    with TRAIN_JSONL.open("w", encoding="utf-8") as ftrain, VAL_JSONL.open(
        "w", encoding="utf-8"
    ) as fval:
        batch: list[str] = []

        def flush() -> None:
            if not batch:
                return
            encoded = tokenizer(batch, add_special_tokens=True)["input_ids"]
            for text, ids in zip(batch, encoded):
                if st["done"]:
                    break
                consume(text, len(ids), ftrain, fval)
            batch.clear()

        for example in stream:
            text = _render_conversation(example.get("messages"))
            if len(text) < 32:
                continue
            batch.append(text)
            if len(batch) >= 64:
                flush()
            if st["done"]:
                break
            if time.time() - st["last_log"] > 15:
                st["last_log"] = time.time()
                tr = st["train"]
                log(
                    f"  ... {tr['rows']:,} train rows, "
                    f"{tr['trainable_tokens']:,}/{PLANNED_TOKENS:,} trainable tokens"
                )
        flush()

    if st["train"]["trainable_tokens"] < PLANNED_TOKENS:
        fail(
            "Dataset stream ended before the token target was reached "
            f"({st['train']['trainable_tokens']:,} < {PLANNED_TOKENS:,})."
        )
    if st["val"]["rows"] < VAL_ROWS:
        warn(f"Validation slice has {st['val']['rows']} rows (< {VAL_ROWS}); continuing.")

    train_hash = sha256_file(TRAIN_JSONL)
    val_hash = sha256_file(VAL_JSONL)
    manifest = {
        "schema": "preflight_dataset_manifest_v1",
        "generated_utc": utc_now(),
        "source": {
            "dataset_id": DATASET_ID,
            "split": DATASET_SPLIT,
            "revision": revision,
            "revision_pinned": True,
            "streaming_order": "upstream order at the pinned revision, no shuffle",
            "reproduce_load": (
                f'load_dataset("{DATASET_ID}", split="{DATASET_SPLIT}", '
                f'streaming=True, revision="{revision}")'
            ),
        },
        "rendering": "messages flattened to 'User:'/'Assistant:'/'System:' lines, joined by newlines",
        "selection_rule": (
            f"train: accumulate rows until sum(min(len(tokens), {cap})+1) >= "
            f"{PLANNED_TOKENS}; validation: next {VAL_ROWS} rows"
        ),
        "tokenizer": {"id": TOKENIZER_ID, "vocab_size": len(tokenizer)},
        "planned_training_token_positions": PLANNED_TOKENS,
        "epoch_semantics": (
            "trainer samples the file uniformly with replacement; the run is "
            "token-equivalent to ~1 epoch, not an ordered pass"
        ),
        "files": {
            "train": {
                "repo_path": "datasets/training_data.jsonl",
                "bytes": TRAIN_JSONL.stat().st_size,
                "sha256": train_hash,
                **st["train"],
            },
            "validation": {
                "repo_path": "datasets/validation.jsonl",
                "bytes": VAL_JSONL.stat().st_size,
                "sha256": val_hash,
                **st["val"],
            },
        },
    }
    write_json(DATASET_MANIFEST_PATH, manifest)
    state["dataset_ready"] = True
    state["dataset_hashes"] = {"train": train_hash, "val": val_hash}
    save_state(state)
    log(
        f"Dataset staged: train {st['train']['rows']:,} rows "
        f"({human_bytes(TRAIN_JSONL.stat().st_size)}, "
        f"{st['train']['trainable_tokens']:,} trainable tokens, "
        f"{st['train']['truncated_rows']:,} rows will be truncated at {SEQ_LEN}); "
        f"val {st['val']['rows']:,} rows."
    )
    return manifest


# =====================================================================
# Phase 5: repository config (import, mutate, validate, snapshot)
# =====================================================================
def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        return repr(value)


def phase_config(device_info: dict, staged_env: dict) -> None:
    """Import the repository config singleton and shape it for this run.

    Post-import mutation of the shared config object is the repository's
    own pattern for reduced-size runs; every architectural mechanism
    (ternary linear layers, grouped-query attention, sparse expert
    routing, the recurrent mixer, gradient checkpointing) stays on and is
    executed by unmodified repository code.
    """
    from config.config import cfg, validate_layer_config  # noqa: WPS433
    from utils.logger import try_git_commit  # noqa: WPS433

    # --- identity / dimensions ---------------------------------------
    cfg.model_name = MODEL_NAME
    cfg.hidden_size = HIDDEN_SIZE
    cfg.intermediate_size = FFN_INTERMEDIATE
    cfg.num_layers = NUM_LAYERS
    cfg.num_hidden_layers = NUM_LAYERS
    cfg.num_heads = NUM_HEADS
    cfg.num_attention_heads = NUM_HEADS
    cfg.num_kv_heads = NUM_KV_HEADS
    cfg.head_dim = HEAD_DIM
    cfg.max_seq_len = SEQ_LEN
    # --- sparse expert routing (8 experts, 2 active) -------------------
    cfg.use_moe = True
    cfg.num_experts = MOE_EXPERTS
    cfg.num_experts_per_tok = MOE_ACTIVE
    cfg.active_experts = MOE_ACTIVE
    cfg.moe_every_n_layers = MOE_EVERY_N_LAYERS
    cfg.moe_intermediate = MOE_INTERMEDIATE
    # --- recurrent mixer -----------------------------------------------
    cfg.use_liquid = True
    cfg.liquid_layers_idx = list(LIQUID_LAYER_IDS)
    cfg.liquid_every_n_layers = 0
    cfg.liquid_warmup_steps = LIQUID_WARMUP_STEPS
    cfg.liquid_fast_path = False
    cfg.liquid_train_impl = "packed_pair"
    # --- disabled optional subsystems -----------------------------------
    cfg.use_qinn = False
    # [2026-07-08] OPTIMIZER PARITY (behavior change to this script, deliberate).
    # This used to force `use_galore=False, use_8bit_adam=False` ("plain fp32 AdamW is well
    # within budget here"). But the real 45K run uses GaLoreAdamW8bit, whose low-rank
    # projection (rank / scale / update_proj_gap) changes the effective step size in ways
    # that do NOT map 1:1 to plain AdamW. Sweeping an LR under plain AdamW and then running
    # 45K under GaLore verifies the wrong optimizer's dynamics. The pre-flight now inherits
    # whatever the canonical config selects, so a "stable" LR is stable under the optimizer
    # 45K will actually use. Falls back gracefully: build_optimizer() already degrades to
    # torch AdamW when galore_torch / bitsandbytes are unavailable, and logs the ACTIVE class.
    cfg.use_torch_compile = False
    # --- schedule / batching (must fit 8 GB, no CPU offload) ------------
    cfg.max_steps = TOTAL_STEPS
    cfg.batch_size = GLOBAL_BATCH
    cfg.micro_batch_size = MICRO_BATCH
    cfg.grad_accum_steps = GRAD_ACCUM
    cfg.use_gradient_checkpointing = True
    cfg.epoch_mode = False
    cfg.save_interval = SAVE_INTERVAL
    cfg.val_check_interval = VAL_INTERVAL
    cfg.log_interval = LOG_INTERVAL
    cfg.seed = SEED
    cfg.dataloader_num_workers = 0
    # --- teacher policy: tokenizer only, no distillation ----------------
    cfg.use_precomputed_logits = False
    cfg.require_gated_teacher = False
    cfg.distill_alpha = 0.0
    cfg.teacher_model_id = str(TOKENIZER_DIR)  # local snapshot from phase_tokenizer
    cfg.use_tr_tokenizer = False
    # --- outputs stay inside the work directory -------------------------
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.save_dir = str(CKPT_DIR)
    cfg.output_dir = str(CKPT_DIR)

    # Re-run the repository's own structural validation on the new shape.
    validate_layer_config(cfg)
    if cfg.hidden_size != cfg.num_heads * cfg.head_dim:
        fail("Internal geometry error: hidden_size != num_heads * head_dim.")

    snapshot = {str(k): _json_safe(v) for k, v in sorted(cfg.__dict__.items())}
    run_config = {
        "schema": "preflight_run_config_v1",
        "generated_utc": utc_now(),
        "purpose": "infrastructure validation only; not a model-quality run",
        "planned": {
            "total_steps": TOTAL_STEPS,
            "global_batch": GLOBAL_BATCH,
            "micro_batch": MICRO_BATCH,
            "grad_accum": GRAD_ACCUM,
            "seq_len": SEQ_LEN,
            "tokens_per_step": TOKENS_PER_STEP,
            "planned_token_positions": PLANNED_TOKENS,
            "checkpoint_interval": SAVE_INTERVAL,
            "validation_interval": VAL_INTERVAL,
            "precision": "bf16 mixed precision, activation checkpointing on",
        },
        "config": snapshot,
        "environment_switches": {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith(("TITAN_", "MERTFORMER_", "HF_", "WANDB_", "PYTORCH_CUDA", "TOKENIZERS_"))
        },
        "staged_env": staged_env,
        "device": device_info,
        "repository": {
            "path": str(REPO_DIR),
            "git_commit": try_git_commit(REPO_DIR),
        },
    }
    write_json(RUN_CONFIG_PATH, run_config)
    log(
        f"Config shaped: {NUM_LAYERS}L x {HIDDEN_SIZE}h | attn {NUM_HEADS}q/"
        f"{NUM_KV_HEADS}kv | experts {MOE_EXPERTS}x{MOE_ACTIVE} at layers 2,5 | "
        f"mixer at layer {LIQUID_LAYER_IDS[0]} | seq {SEQ_LEN} | "
        f"{MICRO_BATCH}x{GRAD_ACCUM} -> global {GLOBAL_BATCH} | steps {TOTAL_STEPS}"
    )


# =====================================================================
# Phase 6: run the repository's canonical training entry point
# =====================================================================
def phase_train() -> dict:
    import torch  # noqa: WPS433

    from train.train import train as run_training  # noqa: WPS433

    torch.cuda.reset_peak_memory_stats()
    outcome: dict = {"status": "returned", "exception": None, "traceback": None, "exit_code": None}
    log("Handing control to the repository trainer (train.train.train) ...")
    t0 = time.time()
    try:
        run_training()
    except SystemExit as exc:  # trainer exits explicitly on some setup errors
        outcome["status"] = "system_exit"
        outcome["exit_code"] = exc.code
    except KeyboardInterrupt:
        outcome["status"] = "interrupted"
    except Exception as exc:  # noqa: BLE001 - archived + reported verbatim
        outcome["status"] = "exception"
        outcome["exception"] = repr(exc)
        outcome["traceback"] = traceback.format_exc()
    outcome["wall_seconds"] = time.time() - t0
    try:
        torch.cuda.synchronize()
        outcome["peak_vram_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        outcome["peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    except Exception:
        pass
    log(
        f"Trainer returned ({outcome['status']}) after "
        f"{human_duration(outcome['wall_seconds'])}."
    )
    return outcome


# =====================================================================
# Artifact analysis: hash-chain verification, CSV metrics, console scan
# =====================================================================
def verify_hash_chain(jsonl_path: Path) -> dict:
    """Replay the run log's per-line hash chain exactly as the repository
    logger computes it: sha256(prev_hash_hex + line_without_hash + '\\n')."""
    genesis = hashlib.sha256(b"").hexdigest()
    prev = genesis
    lines = 0
    try:
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.rstrip("\n")
                if not raw:
                    continue
                obj = json.loads(raw)
                chain = obj.get("_chain") or {}
                stored = chain.get("hash")
                pre = dict(obj)
                pre["_chain"] = {k: v for k, v in chain.items() if k != "hash"}
                blob = json.dumps(pre, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                h = hashlib.sha256()
                h.update(prev.encode("utf-8"))
                h.update((blob + "\n").encode("utf-8"))
                if stored != h.hexdigest() or chain.get("prev") != prev:
                    return {
                        "ok": False,
                        "lines": lines,
                        "final_hash": prev,
                        "error": f"chain break at line {lines + 1}",
                    }
                prev = stored
                lines += 1
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "lines": lines, "final_hash": prev, "error": repr(exc)}
    return {"ok": True, "lines": lines, "final_hash": prev, "error": None}


def parse_step_csvs(csv_paths: list[Path]) -> dict:
    """Merge the repository's step CSVs (resumes may produce several)."""
    merged: dict[int, dict] = {}
    for path in sorted(csv_paths, key=lambda p: p.stat().st_mtime):
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        step = int(float(row.get("step") or 0))
                    except (TypeError, ValueError):
                        continue
                    if step <= 0:
                        continue
                    cur = merged.setdefault(step, {})
                    for k, v in row.items():
                        if v not in (None, ""):
                            cur[k] = v
        except Exception as exc:  # noqa: BLE001
            warn(f"CSV parse issue for {path.name}: {exc}")
    steps = sorted(merged)

    def fval(step: int, key: str):
        try:
            return float(merged[step][key])
        except (KeyError, TypeError, ValueError):
            return None

    def series(key: str):
        out = []
        for s in steps:
            v = fval(s, key)
            if v is not None:
                out.append((s, v))
        return out

    losses = series("loss")
    tok_s = [v for _, v in series("tok_s") if v > 0]
    grads = [v for _, v in series("grad_norm")]
    entropy = [v for _, v in series("moe_load_entropy")]
    max_load = [v for _, v in series("moe_max_load")]
    gpu_alloc = [v for _, v in series("gpu_allocated_gb")]

    summary = {
        "logged_rows": len(steps),
        "first_step": steps[0] if steps else None,
        "last_step": steps[-1] if steps else 0,
        "first_loss": losses[0][1] if losses else None,
        "last_loss": losses[-1][1] if losses else None,
        "min_loss": min(v for _, v in losses) if losses else None,
        "max_loss": max(v for _, v in losses) if losses else None,
        "median_tok_s": statistics.median(tok_s) if tok_s else None,
        "max_grad_norm": max(grads) if grads else None,
        "last_grad_norm": grads[-1] if grads else None,
        "moe_entropy_last": entropy[-1] if entropy else None,
        "moe_entropy_min": min(entropy) if entropy else None,
        "moe_max_load_peak": max(max_load) if max_load else None,
        "telemetry_gpu_alloc_peak_gb": max(gpu_alloc) if gpu_alloc else None,
        "loss_curve_sample": [
            {"step": s, "loss": round(v, 4)}
            for s, v in losses[:: max(1, len(losses) // 24) or 1]
        ],
    }
    return summary


_CONSOLE_PATTERNS = {
    "param_millions": re.compile(r"Model Parametreleri:\s*([0-9.]+)\s*Milyon"),
    "trainable_millions": re.compile(r"Trainable:\s*([0-9.]+)\s*Milyon"),
    "optimizer": re.compile(r"OPTIMIZER ACTIVE:\s*(\w+)"),
    "val_loss": re.compile(r"Validation Loss:\s*([0-9]+\.[0-9]+)"),
}
_CONSOLE_COUNTERS = {
    "nan_skips": "NaN detected",
    "oom_events": "OOM detected",
    "liquid_spikes": "LIQUID SPIKE",
    "router_collapse_alerts": "ROUTER COLLAPSE DETECTED",
    "imbalance_alerts": "EARLY IMBALANCE ALERT",
    "grad_norm_collapse_warnings": "Gradient norm collapse",
    "safety_brake": "SAFETY BRAKE",
    "early_stop_triggered": "Early stopping triggered",
    "liquid_unfrozen": "UNFREEZING LIQUID",
    "onnx_export_failed": "ONNX D\u00d6N\u00dc\u015e\u00dcM HATASI",
    "onnx_export_ok": "ONNX BA\u015eARIYLA",
    "new_best_val": "NEW BEST",
    "data_skip_alarms": "Veri atlama oran\u0131 y\u00fcksek",
}


def scan_console_log() -> dict:
    findings: dict = {k: 0 for k in _CONSOLE_COUNTERS}
    findings.update(
        param_millions=None, trainable_millions=None, optimizer=None, val_losses=[]
    )
    try:
        text = CONSOLE_LOG_PATH.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return findings
    for key, needle in _CONSOLE_COUNTERS.items():
        findings[key] = text.count(needle)
    m = _CONSOLE_PATTERNS["param_millions"].search(text)
    if m:
        findings["param_millions"] = float(m.group(1))
    m = _CONSOLE_PATTERNS["trainable_millions"].search(text)
    if m:
        findings["trainable_millions"] = float(m.group(1))
    m = _CONSOLE_PATTERNS["optimizer"].search(text)
    if m:
        findings["optimizer"] = m.group(1)
    findings["val_losses"] = [
        float(v) for v in _CONSOLE_PATTERNS["val_loss"].findall(text)
    ]
    return findings


# =====================================================================
# Artifact collection
# =====================================================================
_STEP_CKPT_RE = re.compile(r"_step_(\d+)\.pt$")


def collect_artifacts(state: dict) -> dict:
    logs_dir = REPO_DIR / "logs"
    start_ts = float(state.get("first_start_ts", 0)) - 5.0

    def mine(p: Path) -> bool:
        try:
            return p.stat().st_mtime >= start_ts
        except OSError:
            return False

    def grab(pattern: str) -> list[Path]:
        if not logs_dir.exists():
            return []
        return sorted(
            (p for p in logs_dir.glob(pattern) if mine(p)),
            key=lambda p: p.stat().st_mtime,
        )

    jsonl_paths = grab("run_*.jsonl")
    csv_paths = grab("run_*.csv")
    manifest_paths = grab("run_*.manifest.json")
    manifest = read_json(manifest_paths[-1]) if manifest_paths else {}

    chains = {p.name: verify_hash_chain(p) for p in jsonl_paths}
    newest_jsonl = None
    run_id = manifest.get("run_id")
    if run_id and (logs_dir / f"{run_id}.jsonl").exists():
        newest_jsonl = logs_dir / f"{run_id}.jsonl"
    elif jsonl_paths:
        newest_jsonl = jsonl_paths[-1]
    newest_chain = (
        chains.get(newest_jsonl.name)
        if newest_jsonl is not None
        else {"ok": False, "lines": 0, "final_hash": None, "error": "no run log found"}
    )

    report_paths = [
        REPO_DIR / "reports" / name
        for name in KNOWN_REPORT_FILES
        if (REPO_DIR / "reports" / name).exists()
    ]

    entries: list[dict] = []
    sha_map: dict[str, str] = {}
    final_path = latest_path = best_path = None
    step_ckpt_steps: list[int] = []
    if CKPT_DIR.exists():
        for p in sorted(CKPT_DIR.iterdir()):
            if not p.is_file() or p.suffix not in (".pt", ".onnx"):
                continue
            role = "other"
            if p.name.endswith("_final.pt"):
                role, final_path = "final", p
            elif p.name.endswith("_latest.pt"):
                role, latest_path = "latest", p
            elif p.name.endswith("_best.pt"):
                role, best_path = "best", p
            elif (m := _STEP_CKPT_RE.search(p.name)):
                role = f"step_{m.group(1)}"
                step_ckpt_steps.append(int(m.group(1)))
            elif p.suffix == ".onnx":
                role = "onnx_export"
            sha = sha256_file(p)
            sha_map[str(p)] = sha
            entries.append(
                {"name": p.name, "role": role, "bytes": p.stat().st_size, "sha256": sha}
            )

    final_check = {
        "present": final_path is not None,
        "loadable": False,
        "recorded_step": None,
        "tensor_entries": None,
        "error": None,
    }
    if final_path is not None:
        import torch  # noqa: WPS433

        try:
            payload = torch.load(final_path, map_location="cpu", weights_only=False)
            final_check["loadable"] = isinstance(payload, dict) and "model" in payload
            if isinstance(payload, dict):
                try:
                    final_check["recorded_step"] = int(payload.get("step", -1))
                except (TypeError, ValueError):
                    final_check["recorded_step"] = None
                if final_check["loadable"]:
                    final_check["tensor_entries"] = len(payload["model"])
            del payload
        except Exception as exc:  # noqa: BLE001
            final_check["error"] = repr(exc)

    return {
        "jsonl_paths": jsonl_paths,
        "csv_paths": csv_paths,
        "manifest_paths": manifest_paths,
        "manifest": manifest,
        "chains": chains,
        "newest_jsonl": newest_jsonl.name if newest_jsonl else None,
        "newest_chain": newest_chain,
        "report_paths": report_paths,
        "checkpoints": entries,
        "sha_map": sha_map,
        "final_path": final_path,
        "latest_path": latest_path,
        "best_path": best_path,
        "step_ckpt_steps": sorted(step_ckpt_steps),
        "final_check": final_check,
    }


# =====================================================================
# Verdict
# =====================================================================
def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def evaluate_verdict(
    status: str | None,
    outcome: dict,
    csv_summary: dict,
    console: dict,
    art: dict,
    dataset_ok: bool,
) -> tuple[str, list[dict], list[str]]:
    checks: list[dict] = []

    def add(cid: str, desc: str, ok: bool, detail: str) -> None:
        checks.append({"id": cid, "check": desc, "ok": bool(ok), "detail": detail})

    last_step = int(csv_summary.get("last_step") or 0)
    fc = art["final_check"]
    chain = art["newest_chain"]
    manifest_hash = art["manifest"].get("final_chain_hash")
    val_losses = console.get("val_losses") or []
    last_loss = csv_summary.get("last_loss")

    add(
        "C1",
        "trainer reached a clean terminal state",
        status == "completed" or (status == "early_stopped" and last_step >= TOTAL_STEPS),
        f"manifest status={status!r}",
    )
    add(
        "C2",
        f"target step count reached ({TOTAL_STEPS})",
        last_step >= TOTAL_STEPS,
        f"last logged step={last_step}",
    )
    add(
        "C3",
        "final checkpoint present, loadable, stamped with the final step",
        fc["present"] and fc["loadable"] and fc["recorded_step"] == TOTAL_STEPS,
        f"present={fc['present']}, loadable={fc['loadable']}, "
        f"recorded_step={fc['recorded_step']}, tensor_entries={fc['tensor_entries']}"
        + (f", error={fc['error']}" if fc["error"] else ""),
    )
    add(
        "C4",
        "periodic checkpoint rotation exercised (latest + a mid-run step file)",
        art["latest_path"] is not None
        and any(s < TOTAL_STEPS for s in art["step_ckpt_steps"]),
        f"latest={'yes' if art['latest_path'] else 'no'}, "
        f"step files at {art['step_ckpt_steps'] or 'none'}",
    )
    add(
        "C5",
        "validation-driven best checkpoint written",
        art["best_path"] is not None,
        f"best={'yes' if art['best_path'] else 'no'}, "
        f"validation checks observed={len(val_losses)}",
    )
    add(
        "C6",
        "final train and validation losses are finite",
        last_loss is not None
        and math.isfinite(last_loss)
        and bool(val_losses)
        and math.isfinite(val_losses[-1]),
        f"train last={_fmt(last_loss)}, val last="
        f"{_fmt(val_losses[-1]) if val_losses else 'n/a'}",
    )
    add(
        "C7",
        "run-log hash chain verifies and matches the manifest",
        bool(chain["ok"]) and bool(manifest_hash) and chain["final_hash"] == manifest_hash,
        f"lines={chain['lines']}, chain_ok={chain['ok']}, "
        f"matches_manifest={chain['final_hash'] == manifest_hash}"
        + (f", error={chain['error']}" if chain.get("error") else ""),
    )
    add(
        "C8",
        "staged dataset files unchanged since the manifest was written",
        dataset_ok,
        "SHA256 re-verified at packaging time" if dataset_ok else "hash mismatch",
    )
    add(
        "C9",
        "no unhandled exception, no safety-brake termination",
        outcome["status"] == "returned"
        and console.get("safety_brake", 0) == 0
        and status not in ("failed", "safety_brake_stop"),
        f"orchestrator saw status={outcome['status']!r}, "
        f"safety_brake_hits={console.get('safety_brake', 0)}",
    )

    verdict = "PASS" if all(c["ok"] for c in checks) else "FAIL"

    warnings: list[str] = []

    def note(count: int, msg: str) -> None:
        if count:
            warnings.append(msg)

    note(console.get("nan_skips", 0), f"{console.get('nan_skips')} NaN gradient step(s) skipped by the guard")
    note(console.get("oom_events", 0), f"{console.get('oom_events')} OOM event(s) recovered by the retry path")
    note(console.get("liquid_spikes", 0), f"{console.get('liquid_spikes')} recurrent-mixer spike freeze event(s)")
    note(console.get("router_collapse_alerts", 0), f"{console.get('router_collapse_alerts')} expert-router collapse alert(s)")
    note(console.get("imbalance_alerts", 0), f"{console.get('imbalance_alerts')} early expert-imbalance alert(s)")
    note(console.get("grad_norm_collapse_warnings", 0), f"{console.get('grad_norm_collapse_warnings')} gradient-norm collapse warning(s)")
    note(console.get("data_skip_alarms", 0), f"{console.get('data_skip_alarms')} high data-skip-rate alarm(s) from the loader")

    # [2026-07-08] Loss-quality observation (NON-BLOCKING; the PASS/FAIL verdict above stays
    # infrastructure-only by design — see this script's own stated purpose).
    # Why: on 2026-07-02 this script returned a clean infra PASS while the run had actually
    # diverged (10.4 -> 15.0). A human had to eyeball the loss curve to notice. The data was
    # already in csv_summary; it just was never surfaced. Now it lands in REPORT.md automatically.
    first_loss = csv_summary.get("first_loss")
    last_loss = csv_summary.get("last_loss")
    if (
        first_loss is not None
        and last_loss is not None
        and math.isfinite(first_loss)
        and math.isfinite(last_loss)
        and last_loss > first_loss
    ):
        warnings.append(
            f"training loss ended HIGHER than it started ({first_loss:.4f} -> {last_loss:.4f}) "
            "— check for divergence; infra PASS does not mean the run learned anything"
        )

    if console.get("onnx_export_failed", 0) and not console.get("onnx_export_ok", 0):
        warnings.append("final graph export failed (known-brittle path; checkpoints unaffected)")
    if status == "early_stopped":
        warnings.append("trainer flagged early-stop at the final step (patience path exercised)")
    return verdict, checks, warnings


# =====================================================================
# REPORT.md
# =====================================================================
def write_report(ctx: dict) -> None:
    art = ctx["art"]
    csvs = ctx["csv_summary"]
    console = ctx["console"]
    L: list[str] = []
    add = L.append

    add("# Pre-flight Infrastructure Validation - Report")
    add("")
    add(f"**Verdict: {ctx['verdict']}**  |  generated {utc_now()}  |  trainer status `{ctx['status']}`")
    add("")
    add("## 1. Scope (read before interpreting anything below)")
    add("")
    add(
        "This run validates the training infrastructure only: environment, "
        "dataset staging, the repository's own training loop, checkpoint "
        "rotation, resumability hooks, hash-chained logging, and packaging. "
        "It says nothing about model quality. A model of roughly "
        f"{_fmt(console.get('param_millions'), 1)}M parameters trained on "
        f"~{PLANNED_TOKENS/1e6:.0f}M tokens - with ternary weight simulation, "
        "sparse expert routing, and a recurrent mixer layer - will produce "
        "mostly incoherent, partially-grammatical fragments. Loss spikes, "
        "expert-routing imbalance, and skipped NaN steps are plausible at this "
        "scale; they are listed under Observations and are not failures unless "
        "they terminated the run."
    )
    add("")
    add("## 2. Run summary")
    add("")
    add("| Item | Value |")
    add("|---|---|")
    add(f"| Device | {ctx['device'].get('gpu_name')} ({ctx['device'].get('gpu_vram_gb')} GB VRAM, sm_{str(ctx['device'].get('gpu_compute_capability','')).replace('.','')}) |")
    add(f"| Precision | bf16 mixed precision, activation checkpointing on |")
    add(f"| Geometry | {NUM_LAYERS} layers x {HIDDEN_SIZE} hidden, attn {NUM_HEADS}q/{NUM_KV_HEADS}kv (head_dim {HEAD_DIM}) |")
    add(f"| Experts / mixer | {MOE_EXPERTS} experts, top-{MOE_ACTIVE}, at layers 2 and 5; recurrent mixer at layer {LIQUID_LAYER_IDS[0]} |")
    add(f"| Parameters | {_fmt(console.get('param_millions'),2)}M total / {_fmt(console.get('trainable_millions'),2)}M trainable (as printed by the model) |")
    add(f"| Optimizer | {console.get('optimizer') or 'n/a'} |")
    add(f"| Batching | micro {MICRO_BATCH} x accum {GRAD_ACCUM} = {GLOBAL_BATCH} sequences/step at seq_len {SEQ_LEN} |")
    add(f"| Steps completed | {_fmt(int(csvs.get('last_step') or 0))} / {TOTAL_STEPS:,} |")
    add(f"| Token positions consumed | {_fmt(ctx['tokens']['nominal'])} (steps x {TOKENS_PER_STEP:,}) |")
    add(f"| Supervised tokens (trainer count) | {_fmt(ctx['tokens']['supervised'])} |")
    add(f"| Wall-clock (this process) | {human_duration(ctx['outcome'].get('wall_seconds', 0))} |")
    add(f"| Median throughput | {_fmt(csvs.get('median_tok_s'), 0)} tok/s |")
    add(f"| Peak VRAM | {_fmt(ctx['outcome'].get('peak_vram_alloc_gb'), 2)} GB allocated / {_fmt(ctx['outcome'].get('peak_vram_reserved_gb'), 2)} GB reserved |")
    add("")
    if ctx.get("resumed_segments", 1) > 1:
        add(f"_This result was produced across {ctx['resumed_segments']} resumed segment(s); wall-clock above covers the final segment only. Loss/step data below is merged across segments._")
        add("")
    add("## 3. Losses")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add(f"| First logged train loss | {_fmt(csvs.get('first_loss'))} (step {_fmt(csvs.get('first_step'))}) |")
    add(f"| Final train loss | {_fmt(csvs.get('last_loss'))} (step {_fmt(csvs.get('last_step'))}) |")
    add(f"| Min / max train loss | {_fmt(csvs.get('min_loss'))} / {_fmt(csvs.get('max_loss'))} |")
    vls = console.get("val_losses") or []
    add(f"| Validation losses (every {VAL_INTERVAL} steps) | {', '.join(_fmt(v) for v in vls) if vls else 'n/a'} |")
    add(f"| Best validation loss | {_fmt(ctx['tokens'].get('best_val'))} |")
    add(f"| Final gradient norm / peak | {_fmt(csvs.get('last_grad_norm'))} / {_fmt(csvs.get('max_grad_norm'))} |")
    add(f"| Expert load entropy (last / min) | {_fmt(csvs.get('moe_entropy_last'))} / {_fmt(csvs.get('moe_entropy_min'))} |")
    add("")
    add("Loss curve sample (merged from the run CSVs; full series is in `logs/runs/`):")
    add("")
    add("| Step | Loss |")
    add("|---|---|")
    for pt in csvs.get("loss_curve_sample") or []:
        add(f"| {pt['step']} | {pt['loss']} |")
    add("")
    add("## 4. Checkpoints")
    add("")
    add("| File | Role | Size | SHA256 |")
    add("|---|---|---|---|")
    for e in art["checkpoints"]:
        add(f"| {e['name']} | {e['role']} | {human_bytes(e['bytes'])} | `{e['sha256']}` |")
    if not art["checkpoints"]:
        add("| (none found) | - | - | - |")
    add("")
    fc = art["final_check"]
    add(
        f"Final checkpoint verification: present={_fmt(fc['present'])}, "
        f"loadable={_fmt(fc['loadable'])}, recorded step={_fmt(fc['recorded_step'])}, "
        f"tensor entries={_fmt(fc['tensor_entries'])}."
    )
    add("")
    add("## 5. Infrastructure checks")
    add("")
    add("| ID | Check | Result | Detail |")
    add("|---|---|---|---|")
    for c in ctx["checks"]:
        add(f"| {c['id']} | {c['check']} | {'PASS' if c['ok'] else 'FAIL'} | {c['detail']} |")
    add("")
    add("## 6. Observations (non-fatal)")
    add("")
    if ctx["warnings"]:
        for w in ctx["warnings"]:
            add(f"- {w}")
    else:
        add("No guard events were recorded: no NaN skips, OOM retries, spike freezes, or routing alerts.")
    add("")
    add("## 7. Dataset slice (exact recipe)")
    add("")
    dm = ctx["dataset_manifest"]
    src = dm.get("source", {})
    add(
        f"Source `{src.get('dataset_id')}` split `{src.get('split')}`, pinned to "
        f"upstream revision `{src.get('revision')}` and streamed in upstream "
        f"order with no shuffle. Each conversation's messages were flattened to "
        f"`User:`/`Assistant:`/`System:` lines and written as one JSON row. "
        f"{dm.get('selection_rule')}. The trainer truncates each row to "
        f"{SEQ_LEN-1} tokens plus EOS and samples the file uniformly with "
        f"replacement, so the run is token-equivalent to about one epoch rather "
        f"than an ordered pass. The SHA256s below identify the exact bytes used."
    )
    add("")
    add("| File | Rows | Trainable tokens | Truncated rows | SHA256 |")
    add("|---|---|---|---|---|")
    for key in ("train", "validation"):
        f = dm.get("files", {}).get(key, {})
        add(
            f"| {f.get('repo_path')} | {_fmt(f.get('rows'))} | "
            f"{_fmt(f.get('trainable_tokens'))} | {_fmt(f.get('truncated_rows'))} | "
            f"`{f.get('sha256')}` |"
        )
    add("")
    add("## 8. Configuration and environment")
    add("")
    add(
        "The exact configuration object (post-mutation), every relevant "
        "environment switch, library versions, device identity, and the "
        "repository git commit are recorded in `config/run_config.json`. The "
        "run's hash-chained event log, step CSV, and manifest written by the "
        "repository logger are under `logs/runs/`; trainer-side runtime and "
        "telemetry reports are under `logs/repo_reports/`."
    )
    add("")
    add("## 9. Reproduce")
    add("")
    add(
        "Place the repository at `./repo` beside `preflight_run.py` (or set the "
        "clone toggle at the top of the file), delete `preflight_run_output.zip` "
        "if present, and run `python preflight_run.py`. The dataset slice is "
        "byte-identical to regenerate: load the source at the pinned revision "
        "recorded in `dataset/dataset_manifest.json` (the exact `load_dataset` "
        "call is stored there under `source.reproduce_load`) and apply the "
        "selection rule above, then compare against the recorded SHA256s."
    )
    add("")
    add("## 10. Known limitations of this harness")
    add("")
    add(
        "The shared append-only logbook feature of the repository logger was "
        "disabled for this run so cleanup can restore the repository exactly; "
        "the per-run hash-chained log preserved here carries the same records. "
        "Pre-existing repository files on the trainer's data paths (curriculum "
        "stage files and the offline fallback / validation files) were set "
        "aside for the duration of the run via a reversible in-place rename "
        "and restored verbatim at cleanup. "
        "A few small trainer-owned telemetry report files under `reports/` are "
        "refreshed in place by the trainer itself; if they pre-existed, their "
        "prior content is not restored (copies are archived here). The final "
        "graph export step is known-brittle for this architecture and is "
        "treated as non-fatal. Sampling is with replacement, so 'one epoch' is "
        "token-equivalent, not a strict pass. The console log inside this ZIP "
        "ends at packaging time; final cleanup messages appear on the console "
        "only."
    )
    add("")
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    log(f"REPORT.md written ({human_bytes(REPORT_PATH.stat().st_size)}).")


# =====================================================================
# Output ZIP (single deliverable) + cleanup
# =====================================================================
def build_zip(art: dict, verdict: str, status: str | None) -> None:
    log(f"Packaging output archive -> {OUTPUT_ZIP.name}")
    entries: dict[str, dict] = {}
    with zipfile.ZipFile(
        OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as zf:

        def add_file(src: Path | None, arc: str, sha: str | None = None) -> None:
            if src is None or not src.exists():
                return
            zf.write(src, arc)
            entries[arc] = {"bytes": src.stat().st_size, "sha256": sha or sha256_file(src)}

        add_file(REPORT_PATH, "REPORT.md")
        add_file(METRICS_PATH, "metrics.json")
        add_file(RUN_CONFIG_PATH, "config/run_config.json")
        add_file(DATASET_MANIFEST_PATH, "dataset/dataset_manifest.json")
        add_file(STATE_PATH, "logs/state.json")
        for p in art["jsonl_paths"] + art["csv_paths"] + art["manifest_paths"]:
            add_file(p, f"logs/runs/{p.name}")
        for p in art["report_paths"]:
            add_file(p, f"logs/repo_reports/{p.name}")
        for e in art["checkpoints"]:
            add_file(CKPT_DIR / e["name"], f"checkpoints/{e['name']}", sha=e["sha256"])
        sys.stdout.flush()
        sys.stderr.flush()
        add_file(CONSOLE_LOG_PATH, "logs/console.log")
        zf.writestr(
            "MANIFEST.json",
            json.dumps(
                {
                    "schema": "preflight_zip_manifest_v1",
                    "generated_utc": utc_now(),
                    "verdict": verdict,
                    "trainer_status": status,
                    "entries": entries,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    with zipfile.ZipFile(OUTPUT_ZIP) as zf:
        bad = zf.testzip()
        if bad is not None:
            fail(f"Output archive failed integrity verification on member: {bad}")
    log(
        f"Archive verified: {OUTPUT_ZIP.name} "
        f"({human_bytes(OUTPUT_ZIP.stat().st_size)}, {len(entries) + 1} members)."
    )


def cleanup(state: dict) -> tuple[int, int, bool]:
    """Remove everything this run created, restore what it set aside.

    First, files that were sidelined for the trainer are moved back to
    their original names verbatim. Then repository-side removal is limited
    to the created-file diff against the pre-run baseline inside
    BASELINE_DIRS; the work directory is removed whole. Pre-existing files
    are never deleted, and sideline backups are never treated as created
    files.
    """
    restored = 0
    for entry in state.get("sidelined", []):
        orig, backup = Path(entry["orig"]), Path(entry["backup"])
        try:
            if backup.exists():
                if orig.exists():
                    orig.unlink()  # our staged file occupying the original name
                os.replace(backup, orig)
                restored += 1
        except OSError as exc:
            print(f"[preflight] note: could not restore {orig}: {exc}")

    baseline = state.get("baseline") or {"files": {}, "dirs_present": []}
    removed = 0
    for d in BASELINE_DIRS:
        root = REPO_DIR / d
        if not root.exists():
            continue
        base = set(baseline.get("files", {}).get(d, []))
        files = sorted(
            (p for p in root.rglob("*") if p.is_file()),
            key=lambda p: len(p.parts),
            reverse=True,
        )
        for p in files:
            if SIDELINE_SUFFIX in p.name:
                continue  # a backup restore failed above; never delete user data
            rel = str(p.relative_to(root))
            if rel in base:
                continue
            try:
                p.unlink()
                removed += 1
            except OSError as exc:
                print(f"[preflight] note: could not remove {p}: {exc}")
        for p in sorted(
            (q for q in root.rglob("*") if q.is_dir()),
            key=lambda q: len(q.parts),
            reverse=True,
        ):
            try:
                p.rmdir()  # only succeeds if empty
            except OSError:
                pass
        if d not in baseline.get("dirs_present", []):
            try:
                root.rmdir()
            except OSError:
                pass

    def _onerr(func, path, _exc_info):
        try:
            os.chmod(path, 0o700)
            func(path)
        except Exception:
            pass

    shutil.rmtree(WORK_DIR, onerror=_onerr)
    return removed, restored, WORK_DIR.exists()


# =====================================================================
# Terminal-outcome routing
# =====================================================================
def finalize_run(state: dict, outcome: dict, dataset_manifest: dict, device: dict) -> int:
    save_state(state)

    if outcome["status"] == "interrupted":
        log("Interrupted. Everything kept in place; rerun this script to resume from the latest checkpoint.")
        return 130

    art = collect_artifacts(state)
    status = art["manifest"].get("status")

    if outcome["status"] == "exception":
        log("Trainer raised an exception. State kept so a rerun can resume. Traceback:")
        print(outcome.get("traceback") or outcome.get("exception"), flush=True)
        log("Address the printed cause if needed, then rerun this script.")
        return 2
    if outcome["status"] == "system_exit" and outcome.get("exit_code") not in (None, 0):
        log(
            f"Trainer exited with code {outcome['exit_code']} before completing. "
            "State kept; fix the printed cause and rerun to resume."
        )
        return 2
    if status == "aborted":
        log("Trainer recorded an interrupted run and saved a resume checkpoint. Rerun this script to continue.")
        return 130
    if status in (None, "failed"):
        log(f"Trainer ended with status {status!r}; state kept for inspection. Rerun to retry/resume.")
        return 2

    # status is terminal: completed / early_stopped / safety_brake_stop
    csv_summary = parse_step_csvs(art["csv_paths"])
    console = scan_console_log()
    hashes = state.get("dataset_hashes", {})
    dataset_ok = (
        TRAIN_JSONL.exists()
        and VAL_JSONL.exists()
        and sha256_file(TRAIN_JSONL) == hashes.get("train")
        and sha256_file(VAL_JSONL) == hashes.get("val")
    )
    verdict, checks, warnings = evaluate_verdict(
        status, outcome, csv_summary, console, art, dataset_ok
    )

    man = art["manifest"]
    extra = man.get("extra") if isinstance(man.get("extra"), dict) else {}
    supervised = extra.get("tokens_seen", man.get("tokens_seen"))
    best_val = extra.get("best_val_loss", man.get("best_val_loss"))
    vlosses = console.get("val_losses") or []
    if best_val is None and vlosses:
        best_val = min(vlosses)
    tokens = {
        "nominal": int(csv_summary.get("last_step") or 0) * TOKENS_PER_STEP,
        "supervised": supervised,
        "best_val": best_val,
    }

    metrics = {
        "schema": "preflight_metrics_v1",
        "generated_utc": utc_now(),
        "verdict": verdict,
        "trainer_status": status,
        "checks": checks,
        "observations": warnings,
        "outcome": outcome,
        "tokens": tokens,
        "csv_summary": csv_summary,
        "console_findings": console,
        "log_chains": art["chains"],
        "newest_run_log": art["newest_jsonl"],
        "run_manifest": {
            k: man.get(k)
            for k in (
                "run_id",
                "status",
                "started_at",
                "closed_at",
                "lines",
                "final_chain_hash",
                "config_hash",
            )
            if k in man
        },
        "checkpoints": art["checkpoints"],
        "dataset_integrity_ok": dataset_ok,
        "resumed_segments": len(art["manifest_paths"]) or 1,
    }
    write_json(METRICS_PATH, metrics)

    write_report(
        {
            "verdict": verdict,
            "checks": checks,
            "warnings": warnings,
            "status": status,
            "outcome": outcome,
            "csv_summary": csv_summary,
            "console": console,
            "art": art,
            "dataset_manifest": dataset_manifest,
            "tokens": tokens,
            "device": device,
            "resumed_segments": len(art["manifest_paths"]) or 1,
        }
    )
    build_zip(art, verdict, status)

    uninstall_tee()
    removed, restored, leftover = cleanup(state)

    passed = sum(1 for c in checks if c["ok"])
    print("")
    print(f"[preflight] VERDICT: {verdict}  ({passed}/{len(checks)} infrastructure checks passed)")
    print(
        f"[preflight] steps {csv_summary.get('last_step')}/{TOTAL_STEPS} | "
        f"token positions {tokens['nominal']:,} | wall {human_duration(outcome.get('wall_seconds', 0))} | "
        f"peak VRAM {outcome.get('peak_vram_alloc_gb', 'n/a')} GB"
    )
    print(f"[preflight] output: {OUTPUT_ZIP} ({human_bytes(OUTPUT_ZIP.stat().st_size)}) - REPORT.md is inside.")
    print(
        f"[preflight] cleanup: {removed} staged file(s) removed from the repository "
        f"tree; {restored} pre-existing file(s) restored."
    )
    if warnings:
        print("[preflight] observations: " + "; ".join(warnings))
    if leftover:
        print(f"[preflight] note: {WORK_DIR} could not be fully removed (open handle?); safe to delete manually.")
    return 0 if verdict == "PASS" else 1


# =====================================================================
# Entry point
# =====================================================================
def main() -> int:
    if OUTPUT_ZIP.exists():
        print(
            f"[preflight] {OUTPUT_ZIP} already exists; nothing to do.\n"
            f"[preflight] Delete it (plus {WORK_DIR}, {TRAIN_JSONL} and {VAL_JSONL} "
            "if present) to run again from scratch."
        )
        return 0

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    install_tee()
    log("=" * 72)
    log("PRE-FLIGHT INFRASTRUCTURE VALIDATION  (single run, zero arguments)")
    log("Goal: prove the pipeline runs end-to-end and writes verifiable artifacts.")
    log("Not the goal: model quality. Expect incoherent text from this checkpoint.")
    log("=" * 72)

    staged_env = stage_environment()
    bootstrap_dependencies()
    state = load_state()
    if "first_start_ts" not in state:
        state["first_start_ts"] = time.time()
        save_state(state)
    else:
        log("Existing work directory found: resuming a previous attempt.")

    repo_path, searched = discover_repo()
    saved_repo = state.get("repo_dir")
    if saved_repo and _is_repo_root(Path(saved_repo)):
        repo_path = Path(saved_repo)  # pin resumed runs to the original repo
    if repo_path is not None:
        _set_repo_dir(repo_path)

    device = phase_preflight(state, repo_path, searched)
    phase_repo(repo_path, searched)
    state["repo_dir"] = str(REPO_DIR)
    if "baseline" not in state:
        state["baseline"] = snapshot_repo_tree()
    save_state(state)

    tokenizer = phase_tokenizer(state)
    dataset_manifest = phase_dataset(state, tokenizer)
    del tokenizer

    lock_offline()
    phase_config(device, staged_env)
    log(
        f"Plan: {TOTAL_STEPS:,} steps x {TOKENS_PER_STEP:,} = {PLANNED_TOKENS:,} "
        f"token positions (~1 epoch of the slice, sampled with replacement); "
        f"checkpoint every {SAVE_INTERVAL}, validate every {VAL_INTERVAL}."
    )

    outcome = phase_train()
    return finalize_run(state, outcome, dataset_manifest, device)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OrchestrationError as exc:
        print(f"\n[preflight] ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[preflight] Interrupted. State kept; rerun this script to resume.", flush=True)
        sys.exit(130)
