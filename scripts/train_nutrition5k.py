#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-file, zero-argument Nutrition5k vision side-experiment.

PURPOSE (read this first)
--------------------------
This is a BOUNDED SIDE EXPERIMENT, separate from the canonical 45K
text-pretraining run. It trains a small vision-adapted variant of the
MertFormer trunk to predict 5 nutrition targets (calories, mass, fat, carb,
protein) from a single overhead RGB photo of a plate of food, using the real
Nutrition5k dataset (Thames et al., CVPR 2021, arXiv:2103.03375).

It is designed to run start-to-finish, single click, on one RTX 5070
laptop GPU, with or without the full mertformer-titan-core repository
present locally (see "PORTABILITY" below).

DATA PROVENANCE (verified against the live source on 2026-07-17, not
assumed from a secondhand description)
--------------------------------------------------------------------
The dataset lives entirely in a public, anonymously-readable Google Cloud
Storage bucket -- NOT in the google-research-datasets/Nutrition5k git repo
(that repo only holds a README and two shell/py scripts). No gsutil, gcloud
SDK, or auth of any kind is required; every file below was fetched with a
plain HTTPS GET during development of this script:

    https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/...

Verified real paths/filenames (some differ from casual secondhand summaries
-- corrected here against the live bucket):
    dish_ids/splits/rgb_train_ids.txt      (4,059 dish ids)
    dish_ids/splits/rgb_test_ids.txt       (  709 dish ids)
    metadata/dish_metadata_cafe1.csv
    metadata/dish_metadata_cafe2.csv
    metadata/ingredients_metadata.csv      (plural "ingredients" -- not
                                             "ingredient_metadata.csv")
    imagery/realsense_overhead/dish_<id>/rgb.png          (~400 KB each)
    imagery/realsense_overhead/dish_<id>/depth_raw.png    (not downloaded --
    imagery/realsense_overhead/dish_<id>/depth_color.png   see DEPTH below)
    imagery/side_angles/dish_<id>/camera_{A,B,C,D}.h264   (not downloaded;
                                             raw H.264, NOT .mp4 as some
                                             secondary sources claim)

Dish metadata CSV row layout (verified by reading raw bytes of
dish_metadata_cafe1.csv): each row is
    dish_id,total_calories,total_mass,total_fat,total_carb,total_protein,
    then a variable-length repeating group of 7 fields per ingredient
    (ingr_id,ingr_name,ingr_grams,ingr_calories,ingr_fat,ingr_carb,ingr_protein).
There is NO "num_ingrs" column despite that being reported in some
secondhand summaries -- ingredient count is simply the number of trailing
7-field groups, and this script never needs it (only the 5 dish-level
totals are used as regression targets).

DEPTH: SKIPPED ON PURPOSE. Nutrition5k ships a matched depth image per dish
(depth_raw.png, depth_color.png) alongside rgb.png. This script downloads
RGB only. This was a deliberate one-shot tradeoff (this experiment will not
be repeated, so reliability of the one attempt mattered more than the
depth channel's reported-but-modest accuracy gain): the paper's own Table 3
shows depth-as-4th-channel improves calorie MAE% from 26.1% to 18.8%, a
real but secondary gain relative to the complexity depth fusion adds
(16-bit depth normalization, RGB-D alignment, a second input stem) on a
pipeline that gets exactly one unattended run. RGB-only "2D Direct
Prediction" is also the most-replicated baseline in follow-up literature,
which keeps the comparison in REPORT.md meaningful.

WHAT THIS SCRIPT REUSES FROM THE REAL REPOSITORY (unmodified, source of truth)
-------------------------------------------------------------------------------
* config.config.cfg                    - mutated post-import for the small
                                          vision geometry, the repository's
                                          own established pattern (see
                                          scripts/preflight_run.py).
* layers.bitlinear.BitLinear           - ternary (1.58-bit) linear layers.
* layers.moe.MoE                       - sparse mixture-of-experts block.
* layers.liquid.LiquidMixer            - recurrent CfC mixer layer.
* layers.mertformer_block.RMSNorm      - the trunk's normalization layer.
* layers.ffn.MertFormerFFN             - the dense SwiGLU FFN block.
* model.nutrition_vision.NutritionVisionModel - the new (this-experiment-
  only) vision trunk assembled from the four reused layers above plus one
  necessarily-new component: a bidirectional attention module (GQA in
  layers/mla.py hardcodes causal masking on every code path -- verified by
  reading the file, not assumed -- which is correct for autoregressive text
  and wrong for an unordered set of image patches). See
  model/nutrition_vision.py's own docstring for the full, explicit reuse
  boundary, including the one known and accepted directional bias left in
  (MoE's LiquidRouter still runs a small causal depthwise conv over the
  flattened patch sequence).

NOT reused: train.train.train() (the canonical LM trainer). It is coupled
to token-ID input, a vocab_size embedding lookup, and (by default) KD
against a teacher LM -- none of which apply to image-to-scalar regression.
This script implements its own short, plain training loop instead
(phase_train below): plain AdamW, no GaLore/8-bit-Adam (that optimizer
stack exists for a 3.67B-parameter run; this model is ~10-15M parameters
and plain AdamW is already well inside budget on one RTX 5070).

PORTABILITY (the actual point of this script)
----------------------------------------------
Two ways to run this file:
  1. Inside the real repository (this is the normal case: this file lives
     at <repo>/scripts/train_nutrition5k.py). It imports config/layers/model
     directly from the surrounding repo tree.
  2. Standalone, on a "clean" machine with none of the rest of the repo
     present. Running this script's --package-only phase (or letting a
     completed run reach its packaging phase) writes
     scripts/nutrition5k_output.zip, which contains this script, a
     `vendor/` copy of the exact config/layers/model closure this script
     needs (verified by tracing every import transitively -- see
     REQUIRED_VENDOR_FILES below), a Windows launcher
     (RUN_NUTRITION5K.bat), and a README. Unzip that archive anywhere on
     the clean laptop and double-click RUN_NUTRITION5K.bat: this same
     script runs again, this time importing from its own bundled `vendor/`
     directory instead of a surrounding repo (auto-detected, see
     discover_or_vendor_repo()).

NETWORK POLICY
--------------
Network is used only for: (a) installing missing pure-Python dependencies
(Pillow, numpy, tqdm -- never torch/CUDA packages), (b) the one-time
dataset download from the public GCS bucket over plain HTTPS. Nothing else
touches the network; there is no telemetry, no experiment-tracker upload.

RESUME / IDEMPOTENCY
---------------------
* Dataset download skips any file already on disk with the expected size.
* Training checkpoints (latest + best) allow a rerun to resume instead of
  restarting from step 0.
* Unlike scripts/preflight_run.py (a disposable infrastructure-validation
  harness that restores the repo to its pre-run state), this script's
  dataset/checkpoints/report ARE the deliverable of a real experiment and
  are deliberately left in place -- there is no destructive cleanup phase.

EXIT CODES
----------
0   = training completed and REPORT.md / output ZIP were written
1   = training completed but produced a degenerate result (see REPORT.md)
2   = setup/runtime error (state kept so a rerun can resume)
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
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =====================================================================
# Fixed run configuration (the only intended user-editable knobs)
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_NAME = "nutrition5k_vision_side_experiment"

# --- vision trunk geometry (small: this is a side experiment, not the
#     canonical 3.67B run) ------------------------------------------------
IMAGE_SIZE = 256          # matches the paper's own InceptionV2 input resolution
PATCH_SIZE = 16           # -> 256 patches/image (16x16 grid)
HIDDEN_SIZE = 256
NUM_LAYERS = 8
NUM_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 32             # 8 * 32 == 256 == HIDDEN_SIZE
INTERMEDIATE_SIZE = 1024  # dense SwiGLU FFN width (4x hidden)
MOE_EXPERTS = 8
MOE_ACTIVE = 2
MOE_INTERMEDIATE = 512
MOE_LAYER_IDS = (3, 6)    # 0-indexed, disjoint from LIQUID_LAYER_IDS
LIQUID_LAYER_IDS = (5,)
DROPOUT = 0.1
ATTN_DROPOUT = 0.0
HEAD_HIDDEN = 128
TARGETS = ("calories", "mass", "fat", "carb", "protein")

# --- training ------------------------------------------------------------
BATCH_SIZE = 32
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 10       # epochs with no val-calorie-MAE improvement
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0
SEED = 1453                    # repository default, kept for parity
LOG_INTERVAL_STEPS = 20
DATALOADER_WORKERS = 2

# --- dataset (verified public GCS bucket; no auth, no gsutil needed) -----
GCS_BASE = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset"
SPLIT_FILES = {
    "train": f"{GCS_BASE}/dish_ids/splits/rgb_train_ids.txt",
    "test": f"{GCS_BASE}/dish_ids/splits/rgb_test_ids.txt",
}
METADATA_FILES = {
    "cafe1": f"{GCS_BASE}/metadata/dish_metadata_cafe1.csv",
    "cafe2": f"{GCS_BASE}/metadata/dish_metadata_cafe2.csv",
}
OVERHEAD_RGB_URL = GCS_BASE + "/imagery/realsense_overhead/{dish_id}/rgb.png"

# Verified primary-source baseline (Table 3, arXiv:2103.03375, "2D Direct
# Prediction" row -- RGB-only, InceptionV2 backbone, JFT-300M pretrained).
# Format: MAE (paper's units: kcal for calories, grams otherwise) / MAE%.
PAPER_BASELINE_2D_DIRECT = {
    "calories": {"mae": 70.6, "mae_pct": 26.1},
    "mass": {"mae": 40.4, "mae_pct": 18.8},
    "fat": {"mae": 5.0, "mae_pct": 34.2},
    "carb": {"mae": 6.1, "mae_pct": 31.9},
    "protein": {"mae": 5.5, "mae_pct": 29.5},
}
PAPER_ALWAYS_PREDICT_MEAN_BASELINE = {
    "calories": {"mae": 150.8, "mae_pct": 60.2},
    "mass": {"mae": 124.6, "mae_pct": 58.5},
    "fat": {"mae": 8.2, "mae_pct": 67.6},
    "carb": {"mae": 12.5, "mae_pct": 62.1},
    "protein": {"mae": 10.5, "mae_pct": 62.1},
}

# Third-party libraries this script needs beyond the standard library.
# Never includes torch/CUDA packages -- those must already be installed
# and matched to the local driver, exactly like scripts/preflight_run.py.
BOOTSTRAP_LIBS = (
    ("PIL", "Pillow"),
    ("numpy", "numpy"),
    ("tqdm", "tqdm"),
)
FORBIDDEN_INSTALLS = ("torch", "torchvision", "torchaudio", "triton", "bitsandbytes")

# Marker files that identify the real repository root during auto-discovery
# (same convention as scripts/preflight_run.py's REPO_MARKERS).
REPO_MARKERS = ("config/config.py", "layers/mertformer_block.py", "model/transformers.py")

# The exact, transitively-verified import closure this script needs from
# the repository (traced by hand on 2026-07-17: importing any layers.*
# submodule executes layers/__init__.py first, which imports every file in
# layers/; importing model.nutrition_vision executes model/__init__.py
# first, which imports model/transformers.py; config/__init__.py is empty
# and config/config.py only reads a YAML overlay if MERTFORMER_CONFIG-style
# env vars are explicitly set, which this script never does). Nothing
# outside these three packages is required -- no utils/, no orchestrator/,
# no train/, no mertformer_sdk/ (bitlinear.py's mertformer_sdk import is
# lazy, env-gated, and try/except-wrapped).
REQUIRED_VENDOR_FILES = (
    "config/__init__.py",
    "config/build_label.py",
    "config/config.py",
    "layers/__init__.py",
    "layers/bitlinear.py",
    "layers/bitnet_patch.py",
    "layers/cognitive_extensions.py",
    "layers/ffn.py",
    "layers/lifelong_safety.py",
    "layers/liquid.py",
    "layers/mertformer_block.py",
    "layers/mla.py",
    "layers/moe.py",
    "layers/qinn.py",
    "layers/world_model_head.py",
    "model/__init__.py",
    "model/transformers.py",
    "model/nutrition_vision.py",
)

VENDOR_DIR = SCRIPT_DIR / "vendor"  # present only inside the packaged ZIP

# =====================================================================
# Derived paths (everything this run creates lives under WORK_DIR /
# DATA_DIR, next to this script)
# =====================================================================
WORK_DIR = SCRIPT_DIR / "nutrition5k_work"
DATA_DIR = SCRIPT_DIR / "datasets" / "nutrition5k"
IMAGE_DIR = DATA_DIR / "images"
CKPT_DIR = WORK_DIR / "checkpoints"
STATE_PATH = WORK_DIR / "state.json"
CONSOLE_LOG_PATH = WORK_DIR / "console.log"
DATASET_MANIFEST_PATH = WORK_DIR / "dataset_manifest.json"
RUN_CONFIG_PATH = WORK_DIR / "run_config.json"
METRICS_PATH = WORK_DIR / "metrics.json"
TRAIN_LOG_CSV = WORK_DIR / "train_log.csv"
REPORT_PATH = WORK_DIR / "REPORT.md"
OUTPUT_ZIP = SCRIPT_DIR / "nutrition5k_output.zip"

TRAIN_INDEX_CSV = DATA_DIR / "train_index.csv"
TEST_INDEX_CSV = DATA_DIR / "test_index.csv"

STATE_SCHEMA = 1


class OrchestrationError(RuntimeError):
    """User-actionable failure; message must say exactly what to do."""


# =====================================================================
# Small utilities (same conventions as scripts/preflight_run.py)
# =====================================================================
def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[nutrition5k {_ts()}] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[nutrition5k {_ts()}] WARNING: {msg}", flush=True)


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


# =====================================================================
# Console tee: mirror stdout/stderr into WORK_DIR/console.log
# =====================================================================
class _Tee(io.TextIOBase):
    def __init__(self, stream, fh):
        self._s = stream
        self._f = fh

    def write(self, s):
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
# Phase 0: dependency bootstrap (before any torch/repo import)
# =====================================================================
def bootstrap_dependencies() -> None:
    """Install missing pure-Python libraries once; never touch the GPU stack.

    torch must already be installed and matched to the local CUDA driver
    (same policy as scripts/preflight_run.py) -- it is never installed,
    upgraded, or modified here.
    """
    if importlib.util.find_spec("torch") is None:
        fail(
            "torch is not installed. It is deliberately NOT auto-installed: the "
            "build must match your CUDA driver/toolkit. Install the correct "
            "CUDA-enabled torch wheel yourself (see pytorch.org), then rerun."
        )
    import torch  # noqa: WPS433

    cuda_ok = torch.cuda.is_available()
    if not cuda_ok:
        warn(
            "torch.cuda.is_available() is False; this run will use CPU and be "
            "very slow. Install a CUDA-enabled torch build matching your driver "
            "for a real run."
        )

    for _, req in BOOTSTRAP_LIBS:
        base = req.split("[", 1)[0].strip().lower()
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
        text=True, encoding="utf-8", errors="replace",
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
# Phase 1: repository discovery, with a vendored fallback for the
# "clean laptop" case (see PORTABILITY in the module docstring)
# =====================================================================
def _is_repo_root(path: Path) -> bool:
    try:
        return path.is_dir() and all((path / rel).exists() for rel in REPO_MARKERS)
    except OSError:
        return False


def discover_repo() -> Optional[Path]:
    """Search a few sensible locations for the real repository root."""
    candidates: List[Path] = [SCRIPT_DIR, SCRIPT_DIR.parent, Path.cwd(), Path.cwd().parent]
    seen: set = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_repo_root(resolved):
            return resolved
    return None


def discover_or_vendor_repo() -> str:
    """Return which import source is in effect: 'repo' or 'vendor'.

    Prefers a real, surrounding repository (the normal case: this script
    living at <repo>/scripts/train_nutrition5k.py). Falls back to this
    script's own bundled vendor/ directory, which is present only inside
    the ZIP this script's own packaging phase produces -- that is how the
    same file runs standalone on a machine with no other project files.
    """
    repo_path = discover_repo()
    if repo_path is not None:
        sys.path.insert(0, str(repo_path))
        log(f"Using the real repository at {repo_path}.")
        return "repo"

    if VENDOR_DIR.exists() and all((VENDOR_DIR / rel).exists() for rel in REQUIRED_VENDOR_FILES):
        sys.path.insert(0, str(VENDOR_DIR))
        log(f"Real repository not found; using bundled vendor/ at {VENDOR_DIR}.")
        return "vendor"

    fail(
        "Neither the real mertformer-titan-core repository nor a bundled "
        f"vendor/ directory was found next to {SCRIPT_DIR}. Run this script "
        "from inside the repository, or unzip nutrition5k_output.zip (which "
        "contains a vendor/ directory) and run it from there."
    )
    raise AssertionError("unreachable")  # fail() always raises


# =====================================================================
# Phase 2: dataset (one-time online download from the verified GCS bucket)
# =====================================================================
def _http_get(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "mertformer-nutrition5k/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_get_with_retry(url: str, attempts: int = 4, timeout: float = 30.0) -> Optional[bytes]:
    last_exc: Optional[Exception] = None
    for i in range(attempts):
        try:
            return _http_get(url, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None  # dish genuinely absent from the bucket; not a transient error
            last_exc = exc
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        time.sleep(min(2 ** i, 8))
    warn(f"Download failed after {attempts} attempts: {url} ({last_exc})")
    return None


def _download_text(url: str, dest: Path) -> str:
    if dest.exists():
        return dest.read_text(encoding="utf-8")
    data = _http_get_with_retry(url)
    if data is None:
        fail(f"Could not download required file: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return data.decode("utf-8")


def _parse_dish_metadata_csv(text: str) -> Dict[str, Dict[str, float]]:
    """dish_id -> {calories, mass, fat, carb, protein}. See module docstring
    for the verified (no num_ingrs field) row layout."""
    out: Dict[str, Dict[str, float]] = {}
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if len(row) < 6:
            continue
        dish_id = row[0].strip()
        try:
            out[dish_id] = {
                "calories": float(row[1]),
                "mass": float(row[2]),
                "fat": float(row[3]),
                "carb": float(row[4]),
                "protein": float(row[5]),
            }
        except ValueError:
            continue
    return out


def phase_dataset(state: dict, splits: tuple = ("train", "test")) -> dict:
    """Download splits + metadata + per-dish overhead RGB, build index CSVs
    for the requested splits. Resumable per requested split: an index CSV
    that already exists for a requested split is never rebuilt.

    `splits` lets a caller that only needs the held-out test set (e.g.
    evaluate_nutrition5k.py) skip downloading ~4000 train-split images
    (~1.6 GB) it will never use -- important on disk-constrained machines.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    index_paths = {"train": TRAIN_INDEX_CSV, "test": TEST_INDEX_CSV}
    if all(index_paths[s].exists() for s in splits) and DATASET_MANIFEST_PATH.exists():
        log(f"Dataset index already staged for {', '.join(splits)}; reusing it.")
        return read_json(DATASET_MANIFEST_PATH)

    log("Downloading split files and metadata (one-time, public GCS bucket, no auth) ...")
    split_ids: Dict[str, List[str]] = {}
    for name, url in SPLIT_FILES.items():
        if name not in splits:
            continue
        text = _download_text(url, DATA_DIR / f"{name}_ids.txt")
        split_ids[name] = [line.strip() for line in text.splitlines() if line.strip()]
        log(f"  split '{name}': {len(split_ids[name]):,} dish ids")

    labels: Dict[str, Dict[str, float]] = {}
    for name, url in METADATA_FILES.items():
        text = _download_text(url, DATA_DIR / f"dish_metadata_{name}.csv")
        parsed = _parse_dish_metadata_csv(text)
        labels.update(parsed)
        log(f"  metadata '{name}': {len(parsed):,} labeled dishes")

    try:
        from tqdm import tqdm  # noqa: WPS433
    except Exception:
        def tqdm(iterable, **_kw):  # type: ignore
            return iterable

    manifest = read_json(DATASET_MANIFEST_PATH) if DATASET_MANIFEST_PATH.exists() else {}
    manifest.setdefault("schema", "nutrition5k_dataset_manifest_v1")
    manifest["generated_utc"] = utc_now()
    manifest.setdefault(
        "source",
        {
            "bucket_base": GCS_BASE,
            "access": "anonymous HTTPS GET, no auth/gsutil required (verified 2026-07-17)",
            "modalities_downloaded": ["realsense_overhead/rgb.png"],
            "modalities_skipped": [
                "realsense_overhead/depth_raw.png",
                "realsense_overhead/depth_color.png",
                "side_angles/camera_{A,B,C,D}.h264",
            ],
            "skip_reason": "see module docstring 'DEPTH: SKIPPED ON PURPOSE'",
        },
    )
    manifest.setdefault("splits", {})

    for split_name, ids in split_ids.items():
        rows = []
        missing_label = 0
        missing_image = 0
        for dish_id in tqdm(ids, desc=f"images[{split_name}]"):
            lbl = labels.get(dish_id)
            if lbl is None:
                missing_label += 1
                continue
            img_path = IMAGE_DIR / f"{dish_id}.png"
            if not img_path.exists() or img_path.stat().st_size == 0:
                data = _http_get_with_retry(OVERHEAD_RGB_URL.format(dish_id=dish_id))
                if data is None:
                    missing_image += 1
                    continue
                img_path.write_bytes(data)
            rows.append((dish_id, str(img_path), lbl))

        index_path = TRAIN_INDEX_CSV if split_name == "train" else TEST_INDEX_CSV
        with index_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["dish_id", "image_path", *TARGETS])
            for dish_id, path, lbl in rows:
                writer.writerow([dish_id, path, *[lbl[t] for t in TARGETS]])

        manifest["splits"][split_name] = {
            "requested": len(ids),
            "written": len(rows),
            "missing_label": missing_label,
            "missing_image": missing_image,
            "index_path": str(index_path),
        }
        log(
            f"  {split_name}: {len(rows):,}/{len(ids):,} dishes staged "
            f"({missing_label} missing labels, {missing_image} missing images)"
        )

    if "train" in splits and manifest["splits"].get("train", {}).get("written", 0) == 0:
        fail("Dataset staging produced zero usable training rows; cannot proceed.")

    write_json(DATASET_MANIFEST_PATH, manifest)
    state["dataset_ready"] = True
    save_state(state)
    log("Dataset staged.")
    return manifest


# =====================================================================
# Phase 3: repository config (import, mutate for the vision geometry)
# =====================================================================
def phase_config() -> None:
    """Mutate the shared config singleton for this small vision geometry.

    Same post-import mutation pattern scripts/preflight_run.py uses: MoE
    and MertFormerFFN (reused unmodified by model/nutrition_vision.py) read
    their sizing from this global object, not from constructor arguments.
    """
    from config.config import cfg  # noqa: WPS433

    cfg.model_name = MODEL_NAME
    cfg.hidden_size = HIDDEN_SIZE
    cfg.intermediate_size = INTERMEDIATE_SIZE
    cfg.num_layers = NUM_LAYERS
    cfg.num_heads = NUM_HEADS
    cfg.num_kv_heads = NUM_KV_HEADS
    cfg.head_dim = HEAD_DIM
    cfg.use_moe = True
    cfg.num_experts = MOE_EXPERTS
    cfg.num_experts_per_tok = MOE_ACTIVE
    cfg.active_experts = MOE_ACTIVE
    cfg.moe_intermediate = MOE_INTERMEDIATE
    cfg.use_liquid = True
    cfg.liquid_fast_path = False
    cfg.liquid_train_impl = "baseline"
    cfg.use_qinn = False
    cfg.attention_dropout = ATTN_DROPOUT
    log(
        f"Config shaped for vision trunk: {NUM_LAYERS}L x {HIDDEN_SIZE}h | "
        f"attn {NUM_HEADS}q/{NUM_KV_HEADS}kv | experts {MOE_EXPERTS}x{MOE_ACTIVE} "
        f"at layers {list(MOE_LAYER_IDS)} | mixer at layer {list(LIQUID_LAYER_IDS)}"
    )


def build_model():
    from model.nutrition_vision import NutritionVisionModel  # noqa: WPS433

    return NutritionVisionModel(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        moe_layers=MOE_LAYER_IDS,
        liquid_layers=LIQUID_LAYER_IDS,
        dropout=DROPOUT,
        attn_dropout=ATTN_DROPOUT,
        head_hidden=HEAD_HIDDEN,
        targets=TARGETS,
    )


# =====================================================================
# Phase 4: dataset / dataloader / normalization
# =====================================================================
def _load_index(path: Path) -> List[Tuple[str, str, Dict[str, float]]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lbl = {t: float(row[t]) for t in TARGETS}
            rows.append((row["dish_id"], row["image_path"], lbl))
    return rows


def compute_normalization_stats(train_rows) -> Dict[str, Tuple[float, float]]:
    """Per-target (mean, std) computed from the TRAIN split only."""
    stats: Dict[str, Tuple[float, float]] = {}
    n = len(train_rows)
    for t in TARGETS:
        values = [lbl[t] for _, _, lbl in train_rows]
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        std = max(var ** 0.5, 1e-6)
        stats[t] = (mean, std)
    return stats


class Nutrition5kDataset:
    """Map-style dataset (duck-typed: __len__/__getitem__, no torch.utils.data.Dataset
    base class needed) -- must be a MODULE-LEVEL class, not a local class returned by a
    factory function.

    [Windows fix] DataLoader(num_workers>0) on Windows uses the 'spawn' start method,
    which PICKLES the dataset object to hand it to each worker process. A class defined
    inside a function (a "local"/nested class) has no importable qualified name, so
    pickle cannot reconstruct it in the worker process -- this crashed on the real
    RTX 5070 Windows run with `Can't pickle local object
    ...make_dataset_class.<locals>.Nutrition5kDataset` the moment phase_train() built
    its first DataLoader iterator. macOS/Linux masked this in testing here because
    those tests used DATALOADER_WORKERS=0 (no worker-process pickling at all).

    torch/PIL/numpy imports stay INSIDE __getitem__ (not at module top-level) so this
    class -- and the whole module -- still imports cleanly before
    bootstrap_dependencies() has installed Pillow/numpy on a fresh machine, and so
    `--package-only` mode keeps working with no torch installed at all (see its own
    docstring).
    """

    def __init__(self, rows, image_size: int):
        self.rows = rows
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        import numpy as np
        import torch
        from PIL import Image

        dish_id, path, lbl = self.rows[idx]
        img = Image.open(path).convert("RGB")
        # Resize so the shorter side == image_size, then center-crop the
        # square region -- the paper's own stated preprocessing ("images
        # were downsized and center cropped in order to retain the most
        # salient dish region").
        w, h = img.size
        scale = self.image_size / min(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        left = (new_w - self.image_size) // 2
        top = (new_h - self.image_size) // 2
        img = img.crop((left, top, left + self.image_size, top + self.image_size))

        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)
        targets = torch.tensor([lbl[t] for t in TARGETS], dtype=torch.float32)
        return tensor, targets


# =====================================================================
# Phase 5: training loop (plain AdamW; NOT train.train.train() -- see
# module docstring for why)
# =====================================================================
def _lr_schedule(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def phase_train(state: dict, dataset_manifest: dict) -> dict:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(SEED)

    from config.config import cfg  # noqa: WPS433

    train_rows = _load_index(TRAIN_INDEX_CSV)
    test_rows = _load_index(TEST_INDEX_CSV)
    log(f"Loaded index: {len(train_rows):,} train dishes, {len(test_rows):,} test dishes.")

    norm_stats = compute_normalization_stats(train_rows)
    state["norm_stats"] = {t: list(v) for t, v in norm_stats.items()}
    save_state(state)

    train_ds = Nutrition5kDataset(train_rows, IMAGE_SIZE)
    test_ds = Nutrition5kDataset(test_rows, IMAGE_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=DATALOADER_WORKERS,
        pin_memory=pin_memory, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=DATALOADER_WORKERS,
        pin_memory=pin_memory,
    )

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model built: {n_params:,} parameters ({n_trainable:,} trainable) on {device}.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * MAX_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _lr_schedule(s, total_steps, warmup_steps)
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CKPT_DIR / "nutrition5k_best.pt"
    latest_path = CKPT_DIR / "nutrition5k_latest.pt"

    start_epoch = 0
    best_val_calorie_mae = math.inf
    epochs_without_improve = 0
    global_step = 0
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_calorie_mae = ckpt.get("best_val_calorie_mae", math.inf)
        epochs_without_improve = ckpt.get("epochs_without_improve", 0)
        global_step = ckpt.get("global_step", 0)
        log(f"Resumed from {latest_path.name} at epoch {start_epoch}.")

    if not TRAIN_LOG_CSV.exists():
        with TRAIN_LOG_CSV.open("w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(
                ["epoch", "step", "train_loss", "lr", "val_calorie_mae", "val_calorie_mae_pct"]
            )

    def normalize(t: str, value: "torch.Tensor") -> "torch.Tensor":
        mean, std = norm_stats[t]
        return (value - mean) / std

    def run_validation() -> Dict[str, Dict[str, float]]:
        model.eval()
        sums = {t: 0.0 for t in TARGETS}
        gt_sum = {t: 0.0 for t in TARGETS}
        n = 0
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds, _ = model(images)
                bsz = images.size(0)
                n += bsz
                for i, t in enumerate(TARGETS):
                    sums[t] += (preds[t].float() - targets[:, i]).abs().sum().item()
                    gt_sum[t] += targets[:, i].sum().item()
        out = {}
        for t in TARGETS:
            mae = sums[t] / max(1, n)
            gt_mean = gt_sum[t] / max(1, n)
            mae_pct = (mae / gt_mean * 100.0) if gt_mean > 0 else float("nan")
            out[t] = {"mae": mae, "mae_pct": mae_pct}
        return out

    log(
        f"Training plan: {MAX_EPOCHS} epochs max x {steps_per_epoch} steps/epoch "
        f"(early stop patience {EARLY_STOP_PATIENCE} on val calorie MAE), "
        f"batch {BATCH_SIZE}, lr {LEARNING_RATE}, AMP={'on' if use_amp else 'off'}."
    )

    final_val: Dict[str, Dict[str, float]] = {}
    stopped_early = False
    epoch = start_epoch
    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_n = 0
        t_epoch0 = time.time()
        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                preds, aux_loss = model(images)
                task_loss = images.new_zeros(())
                for i, t in enumerate(TARGETS):
                    pred_norm = normalize(t, preds[t].float())
                    tgt_norm = normalize(t, targets[:, i].float())
                    task_loss = task_loss + torch.nn.functional.smooth_l1_loss(pred_norm, tgt_norm)
                loss = task_loss + float(cfg.router_aux_loss_coef) * aux_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            epoch_loss_sum += float(loss.detach().item())
            epoch_loss_n += 1
            if global_step % LOG_INTERVAL_STEPS == 0:
                log(
                    f"epoch {epoch + 1}/{MAX_EPOCHS} step {step + 1}/{steps_per_epoch} "
                    f"(global {global_step}) loss={loss.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        val_metrics = run_validation()
        val_calorie_mae = val_metrics["calories"]["mae"]
        train_loss_epoch = epoch_loss_sum / max(1, epoch_loss_n)
        log(
            f"epoch {epoch + 1}/{MAX_EPOCHS} done in {human_duration(time.time() - t_epoch0)} "
            f"| train_loss={train_loss_epoch:.4f} | val_calorie_MAE="
            f"{val_calorie_mae:.1f} kcal ({val_metrics['calories']['mae_pct']:.1f}%)"
        )
        with TRAIN_LOG_CSV.open("a", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(
                [
                    epoch + 1, global_step, f"{train_loss_epoch:.6f}",
                    f"{scheduler.get_last_lr()[0]:.8f}", f"{val_calorie_mae:.4f}",
                    f"{val_metrics['calories']['mae_pct']:.4f}",
                ]
            )

        is_new_best = val_calorie_mae < best_val_calorie_mae
        if is_new_best:
            best_val_calorie_mae = val_calorie_mae
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        # Built *after* best_val_calorie_mae is updated above, so a checkpoint
        # saved on a new-best epoch reports that epoch's own value, not the
        # previous best (a prior ordering bug left this one step stale for
        # every "new best" checkpoint -- weights were always correct, only
        # this metadata field lagged, e.g. predict_nutrition5k.py's printed
        # "val MAE at save time").
        ckpt_payload = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_calorie_mae": best_val_calorie_mae,
            "epochs_without_improve": epochs_without_improve,
            "norm_stats": state["norm_stats"],
        }
        torch.save(ckpt_payload, latest_path)

        final_val = val_metrics
        if is_new_best:
            torch.save(ckpt_payload, best_path)
            log(f"  new best (val calorie MAE {best_val_calorie_mae:.1f} kcal) -> {best_path.name}")
        elif epochs_without_improve >= EARLY_STOP_PATIENCE:
            log(f"Early stopping: no improvement for {EARLY_STOP_PATIENCE} epochs.")
            stopped_early = True
            break

    # Final report uses the BEST checkpoint's validation metrics, not
    # necessarily the last epoch's (early stopping may have moved past it).
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model"])
        final_val = run_validation()

    return {
        "epochs_run": epoch + 1,
        "stopped_early": stopped_early,
        "global_step": global_step,
        "n_params": n_params,
        "n_trainable": n_trainable,
        "device": str(device),
        "amp": use_amp,
        "val_metrics": final_val,
        "best_val_calorie_mae": best_val_calorie_mae,
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "latest_checkpoint": str(latest_path) if latest_path.exists() else None,
    }


# =====================================================================
# Phase 6: REPORT.md
# =====================================================================
def write_report(ctx: dict) -> None:
    train = ctx["train_result"]
    val = train["val_metrics"]
    L: List[str] = []
    add = L.append

    add("# Nutrition5k Vision Side-Experiment - Report")
    add("")
    add(f"Generated {utc_now()}  |  import source: `{ctx['import_source']}`")
    add("")
    add("## 1. Scope (read before interpreting anything below)")
    add("")
    add(
        "This is a bounded side experiment, separate from the canonical "
        "45K-step text-pretraining run. It reuses the real BitLinear / MoE / "
        "LiquidMixer / RMSNorm layers on a small (~"
        f"{train['n_params']/1e6:.1f}M parameter) vision trunk with a "
        "necessarily-new bidirectional attention module (the shared GQA "
        "attention is hardcoded causal; see model/nutrition_vision.py). It "
        "does **not** reuse train.train.train() (the canonical LM trainer is "
        "coupled to token-ID input and does not apply to image regression) - "
        "training used a short, plain AdamW loop defined in this script."
    )
    add("")
    add(
        "The comparison numbers below are against the original Nutrition5k "
        "paper's own InceptionV2 backbone (2048-d features, JFT-300M "
        "pretrained, full precision) — a much larger, pretrained, full-"
        f"precision model. This experiment's ~{train['n_params']/1e6:.1f}M "
        "parameter, from-scratch, partly-ternary-quantized trunk is **not** "
        "expected to match it; the paper's numbers are included as a "
        "reference point, not a claim of parity."
    )
    add("")
    add("## 2. Data")
    add("")
    manifest = ctx["dataset_manifest"]
    add("| Split | requested | staged | missing label | missing image |")
    add("|---|---:|---:|---:|---:|")
    for name, s in manifest.get("splits", {}).items():
        add(
            f"| {name} | {s['requested']:,} | {s['written']:,} | "
            f"{s['missing_label']:,} | {s['missing_image']:,} |"
        )
    add("")
    add(
        "Source: public GCS bucket `gs://nutrition5k_dataset/nutrition5k_dataset/` "
        "(anonymous HTTPS, no auth/gsutil). Only `imagery/realsense_overhead/*/rgb.png` "
        "was downloaded — depth and side-angle video were deliberately skipped "
        "(see this script's module docstring, 'DEPTH: SKIPPED ON PURPOSE')."
    )
    add("")
    add("## 3. Model")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Parameters | {train['n_params']:,} total / {train['n_trainable']:,} trainable |")
    add(f"| Geometry | {NUM_LAYERS} layers x {HIDDEN_SIZE} hidden, attn {NUM_HEADS}q/{NUM_KV_HEADS}kv |")
    add(f"| Experts / mixer | {MOE_EXPERTS} experts top-{MOE_ACTIVE} at layers {list(MOE_LAYER_IDS)}; LiquidMixer at layer {list(LIQUID_LAYER_IDS)} |")
    add(f"| Input | {IMAGE_SIZE}x{IMAGE_SIZE} RGB, {PATCH_SIZE}x{PATCH_SIZE} patches ({(IMAGE_SIZE // PATCH_SIZE) ** 2} patches/image) |")
    add(f"| Device | {train['device']} | AMP {'on' if train['amp'] else 'off'} |")
    add(f"| Epochs run | {train['epochs_run']} / {MAX_EPOCHS} (early stopped: {train['stopped_early']}) |")
    add("")
    add("## 4. Results vs. the paper's own baselines (Table 3, arXiv:2103.03375)")
    add("")
    add("| Target | This model MAE | This model MAE% | Paper 2D Direct MAE | Paper 2D Direct MAE% | Paper always-predict-mean MAE% |")
    add("|---|---:|---:|---:|---:|---:|")
    for t in TARGETS:
        m = val.get(t, {})
        pb = PAPER_BASELINE_2D_DIRECT[t]
        mb = PAPER_ALWAYS_PREDICT_MEAN_BASELINE[t]
        unit = "kcal" if t == "calories" else "g"
        add(
            f"| {t} | {m.get('mae', float('nan')):.1f} {unit} | {m.get('mae_pct', float('nan')):.1f}% | "
            f"{pb['mae']:.1f} {unit} | {pb['mae_pct']:.1f}% | {mb['mae_pct']:.1f}% |"
        )
    add("")
    add(
        "A result between the 'always-predict-mean' column and the paper's "
        "direct-prediction column means the model learned a real, non-trivial "
        "signal from the image, even without matching the paper's much larger "
        "pretrained backbone."
    )
    add("")
    add("## 5. Checkpoints")
    add("")
    add(f"- Best (by val calorie MAE): `{train.get('best_checkpoint')}`")
    add(f"- Latest: `{train.get('latest_checkpoint')}`")
    add(f"- Best val calorie MAE reached: {train['best_val_calorie_mae']:.1f} kcal")
    add("")
    add("## 6. Known, accepted limitations (not bugs)")
    add("")
    add(
        "- No depth channel (see Section 2). The paper's own Table 3 shows "
        "depth-as-4th-channel improves calorie MAE% from 26.1% to 18.8%; this "
        "experiment does not have that signal available."
    )
    add(
        "- The reused MoE LiquidRouter applies a causal depthwise conv over "
        "the flattened patch sequence (raster-scan order), a directional "
        "inductive bias with no natural meaning for a 2D image. Harmless "
        "(zero-padded), documented in model/nutrition_vision.py, not fixed "
        "(would require modifying the shared, sealed layers/moe.py)."
    )
    add(
        "- Single train/test split (the paper's own Nutri-Train/Nutri-Test "
        "partition), no cross-validation — this was a one-shot run by design."
    )
    add("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    log(f"Report written: {REPORT_PATH}")


# =====================================================================
# Phase 7: packaging (vendor the verified import closure + this script +
# a Windows launcher + README into one droppable ZIP)
# =====================================================================
def _find_source_root_for_vendoring() -> Optional[Path]:
    """Where to copy REQUIRED_VENDOR_FILES FROM. If we're already running
    from a vendor/ directory (i.e. this is itself the packaged/unzipped
    copy), re-package from that same vendor/ directory."""
    repo_path = discover_repo()
    if repo_path is not None:
        return repo_path
    if VENDOR_DIR.exists():
        return VENDOR_DIR
    return None


def build_zip() -> None:
    source_root = _find_source_root_for_vendoring()
    if source_root is None:
        warn("No import source found to vendor from; skipping ZIP packaging.")
        return

    missing = [rel for rel in REQUIRED_VENDOR_FILES if not (source_root / rel).exists()]
    if missing:
        warn(f"Cannot package ZIP: missing source files for vendoring: {missing}")
        return

    launcher_bat = (
        "@echo off\r\n"
        "REM Double-click to train the Nutrition5k vision side-experiment.\r\n"
        "REM Requires: Python 3.10+ with a CUDA-enabled torch already installed\r\n"
        "REM (this script never installs/upgrades torch itself).\r\n"
        "cd /d \"%~dp0\"\r\n"
        "python train_nutrition5k.py\r\n"
        "pause\r\n"
    )
    readme = (
        "Nutrition5k vision side-experiment - standalone package\n"
        "=========================================================\n\n"
        "Contents:\n"
        "  train_nutrition5k.py   - this experiment's single entry point\n"
        "  predict_nutrition5k.py - single-photo inference using the trained\n"
        "                           checkpoint (run after training completes\n"
        "                           at least one epoch): python\n"
        "                           predict_nutrition5k.py <path-to-photo.jpg>\n"
        "  vendor/                - the exact config/layers/model closure this\n"
        "                           script needs, copied from the real repo\n"
        "                           (verified import closure, see the header\n"
        "                           comment 'REQUIRED_VENDOR_FILES' inside\n"
        "                           train_nutrition5k.py)\n"
        "  RUN_NUTRITION5K.bat    - Windows double-click launcher\n\n"
        "Requirements on the target machine:\n"
        "  - Python 3.10+\n"
        "  - A CUDA-enabled torch build already installed and matching the\n"
        "    local GPU driver (this script deliberately never installs or\n"
        "    upgrades torch/CUDA packages - see its module docstring).\n"
        "  - Everything else (Pillow, numpy, tqdm) is installed automatically\n"
        "    on first run if missing.\n\n"
        "Usage: unzip anywhere, double-click RUN_NUTRITION5K.bat (or run\n"
        "`python train_nutrition5k.py` from a terminal in this folder).\n"
        "The dataset (~1.9 GB of Nutrition5k overhead RGB photos) downloads\n"
        "automatically on first run from the public Nutrition5k GCS bucket -\n"
        "no account or token needed. Progress, checkpoints, and the final\n"
        "REPORT.md are written next to this script.\n"
    )

    with zipfile.ZipFile(
        OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as zf:
        zf.write(Path(__file__).resolve(), "train_nutrition5k.py")
        for aux_script in ("predict_nutrition5k.py", "evaluate_nutrition5k.py"):
            aux_path = SCRIPT_DIR / aux_script
            if aux_path.exists():
                zf.write(aux_path, aux_script)
        for rel in REQUIRED_VENDOR_FILES:
            zf.write(source_root / rel, f"vendor/{rel}")
        zf.writestr("RUN_NUTRITION5K.bat", launcher_bat)
        zf.writestr("README.txt", readme)
        if REPORT_PATH.exists():
            zf.write(REPORT_PATH, "REPORT.md")
        if METRICS_PATH.exists():
            zf.write(METRICS_PATH, "metrics.json")
        if TRAIN_LOG_CSV.exists():
            zf.write(TRAIN_LOG_CSV, "train_log.csv")
        # The actual point of bringing this ZIP back: the trained checkpoint(s),
        # so predictions/evaluation can be reproduced independently elsewhere
        # (e.g. on a different machine) without re-running the whole training.
        for ckpt_name in ("nutrition5k_best.pt", "nutrition5k_latest.pt"):
            ckpt_path = CKPT_DIR / ckpt_name
            if ckpt_path.exists():
                zf.write(ckpt_path, f"checkpoints/{ckpt_name}")

    with zipfile.ZipFile(OUTPUT_ZIP) as zf:
        bad = zf.testzip()
        if bad is not None:
            fail(f"Output archive failed integrity verification on member: {bad}")
    log(f"Portable package written: {OUTPUT_ZIP} ({human_bytes(OUTPUT_ZIP.stat().st_size)}).")


# =====================================================================
# Entry point
# =====================================================================
def main() -> int:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    install_tee()
    log("=" * 72)
    log("NUTRITION5K VISION SIDE-EXPERIMENT (single run, zero arguments)")
    log("Bounded side experiment; does not touch the canonical 45K run.")
    log("=" * 72)

    bootstrap_dependencies()
    import_source = discover_or_vendor_repo()

    state = load_state()
    if "first_start_ts" not in state:
        state["first_start_ts"] = time.time()
        save_state(state)
    else:
        log("Existing work directory found: resuming a previous attempt.")

    dataset_manifest = phase_dataset(state)
    phase_config()

    outcome = {"status": "returned", "exception": None, "traceback": None}
    train_result: Optional[dict] = None
    t0 = time.time()
    try:
        train_result = phase_train(state, dataset_manifest)
    except KeyboardInterrupt:
        outcome["status"] = "interrupted"
    except Exception as exc:  # noqa: BLE001
        outcome["status"] = "exception"
        outcome["exception"] = repr(exc)
        outcome["traceback"] = traceback.format_exc()
    outcome["wall_seconds"] = time.time() - t0

    if outcome["status"] == "interrupted":
        log("Interrupted. State and checkpoints kept; rerun this script to resume.")
        uninstall_tee()
        return 130
    if outcome["status"] == "exception":
        log("Training raised an exception. State kept so a rerun can resume. Traceback:")
        print(outcome["traceback"], flush=True)
        uninstall_tee()
        return 2

    assert train_result is not None
    write_report(
        {
            "import_source": import_source,
            "dataset_manifest": dataset_manifest,
            "train_result": train_result,
            "outcome": outcome,
        }
    )
    write_json(
        METRICS_PATH,
        {
            "schema": "nutrition5k_metrics_v1",
            "generated_utc": utc_now(),
            "train_result": train_result,
            "outcome": outcome,
        },
    )
    build_zip()
    uninstall_tee()

    calorie_mae_pct = train_result["val_metrics"].get("calories", {}).get("mae_pct", float("nan"))
    always_mean_pct = PAPER_ALWAYS_PREDICT_MEAN_BASELINE["calories"]["mae_pct"]
    degenerate = not math.isfinite(calorie_mae_pct) or calorie_mae_pct >= always_mean_pct

    print("")
    print(
        f"[nutrition5k] DONE. epochs={train_result['epochs_run']}/{MAX_EPOCHS} "
        f"wall={human_duration(outcome['wall_seconds'])} "
        f"val_calorie_MAE={train_result['best_val_calorie_mae']:.1f} kcal "
        f"({calorie_mae_pct:.1f}%)"
    )
    print(f"[nutrition5k] REPORT.md: {REPORT_PATH}")
    if OUTPUT_ZIP.exists():
        print(f"[nutrition5k] portable package: {OUTPUT_ZIP} ({human_bytes(OUTPUT_ZIP.stat().st_size)})")
    if degenerate:
        print(
            "[nutrition5k] NOTE: final calorie MAE% did not beat the paper's "
            "'always predict the mean' baseline -- see REPORT.md Section 6/4."
        )
    return 1 if degenerate else 0


def package_only() -> int:
    """Build the portable ZIP without running bootstrap/dataset/training.

    This is the mode the module docstring's PORTABILITY section refers to:
    run from inside the real repository to produce a starter-kit ZIP (this
    script + the vendored config/layers/model closure + a Windows launcher
    + README) that a clean machine can unzip and run standalone -- before
    any dataset download or training has happened here. If an earlier full
    run already wrote a REPORT.md, that gets bundled in too; otherwise the
    ZIP is just the starter kit. Needs no third-party library (not even
    torch) and no network access.
    """
    log("Packaging only (no dataset download, no training) ...")
    build_zip()
    if not OUTPUT_ZIP.exists():
        log("Packaging failed: see the warning above (likely a missing vendor file).")
        return 2
    print(f"[nutrition5k] portable package: {OUTPUT_ZIP} ({human_bytes(OUTPUT_ZIP.stat().st_size)})")
    return 0


if __name__ == "__main__":
    try:
        if "--package-only" in sys.argv[1:]:
            sys.exit(package_only())
        sys.exit(main())
    except OrchestrationError as exc:
        print(f"\n[nutrition5k] ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[nutrition5k] Interrupted. State kept; rerun this script to resume.", flush=True)
        sys.exit(130)
