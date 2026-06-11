#!/usr/bin/env python3
"""
MertFormer Titan Phase-0 — multi-GPU data-parallel teacher logit precompute.

This is a PRODUCTION orchestrator on top of ``scripts/precompute_logits_topk.py``.
It launches one worker process per GPU; each worker re-packs the whole stream
from line 0 deterministically (identity preserved) and teacher-forwards only its
own *block-cyclic* shard of sequences (see the [PARALLEL] note in the worker
script). Because every shard is named by its first global ``seq_index``, the
union of all workers' shards reads back in exact global order, so the train-side
per-sequence identity assertion and ``scripts/validate_logit_alignment.py`` keep
holding verbatim — no change to the train reader, the validator, or the
single-process canonical lane.

Why this collapses wall-clock: a 70B teacher in 4-bit fits on a single modern
GPU, so each GPU runs an independent teacher copy over 1/N of the sequences.
Teacher forward is the dominant cost and it is embarrassingly parallel here.

Safety / compatibility guarantees:
  * Default single-process behaviour (``precompute_logits_topk.py`` with no
    sharding flags) is completely untouched.
  * Workers resume statelessly (a block whose shard exists is skipped), so a
    killed/retried worker never double-writes or corrupts a shard.
  * After all workers for a stage finish, the orchestrator verifies *coverage*
    (every block present, no gaps), writes the canonical resume state the rest
    of the pipeline reads, and (unless disabled) runs the alignment validator.

Exit codes (repo convention):
  0   all requested stages produced + verified
  1   a worker failed after retries / runtime error
  4   dataset(s) missing
  5   coverage gap or alignment validation failed
  130 interrupted (workers already checkpoint their completed blocks)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("precompute_parallel")

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent
for _ in range(4):
    if (_ROOT / "config").exists():
        break
    _ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))

from scripts import precompute_logits_topk as P  # noqa: E402

WORKER_SCRIPT = _THIS.parent / "precompute_logits_topk.py"
REPORT_DIR = _ROOT / "logs" / "precompute_parallel"

# Exit codes (kept module-level so tests/callers can reference them by name).
EXIT_OK = 0
EXIT_WORKER_FAIL = 1
EXIT_DATA_MISSING = 4
EXIT_COVERAGE_FAIL = 5
EXIT_INTERRUPT = 130


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without GPUs or subprocesses).
# ---------------------------------------------------------------------------
def resolve_worker_gpus(
    gpus: Optional[int],
    gpu_ids: Optional[str],
    detected: int,
) -> list[int]:
    """Resolve the concrete list of GPU ids the workers will pin to.

    Priority: explicit ``--gpu-ids`` > explicit ``--gpus`` count > auto-detected.
    """
    if gpu_ids:
        ids = [int(x) for x in str(gpu_ids).replace(" ", "").split(",") if x != ""]
        if not ids:
            raise ValueError("--gpu-ids was given but parsed empty")
        return ids
    if gpus is not None:
        if int(gpus) < 1:
            raise ValueError("--gpus must be >= 1")
        return list(range(int(gpus)))
    return list(range(max(0, int(detected))))


def aggregate_total_sequences(worker_states: list[dict]) -> int:
    """Consensus total-sequence count S from finished workers.

    Every worker packs the full stream, so each records the same ``total_sequences``
    once done. We take the max over workers that reported a positive value (a
    worker owning no blocks still reports the correct S).
    """
    vals = [int(ws.get("total_sequences", 0) or 0) for ws in worker_states]
    return max(vals) if vals else 0


def verify_stage_coverage(
    logits_dir: Path,
    stage_name: str,
    total_sequences: int,
    chunk_size: int,
) -> tuple[bool, list[int]]:
    """Every block tiling [0, S) must have its shard file. Returns (ok, missing)."""
    expected = P.block_first_seq_indices(total_sequences, chunk_size)
    missing = [
        first
        for first in expected
        if not (logits_dir / f"{stage_name}_{P.SUBSET}_part_{first}.pt").exists()
    ]
    return (len(missing) == 0, missing)


def build_worker_command(
    stage_num: int,
    shard_id: int,
    num_shards: int,
    *,
    top_k: int,
    chunk_size: int,
    batch_size: int,
    max_seq: int,
    logits_dir: Path,
    model_id: str,
    python_exe: Optional[str] = None,
) -> list[str]:
    """The exact argv a single GPU worker runs (one block-cyclic shard)."""
    exe = python_exe or sys.executable
    return [
        exe,
        str(WORKER_SCRIPT),
        "--stage", str(stage_num),
        "--num-shards", str(num_shards),
        "--shard-id", str(shard_id),
        "--top-k", str(top_k),
        "--chunk-size", str(chunk_size),
        "--batch-size", str(batch_size),
        "--max-seq", str(max_seq),
        "--logits-dir", str(logits_dir),
        "--model-id", str(model_id),
    ]


def _collect_worker_states(logits_dir: Path, stage_name: str) -> list[dict]:
    states: list[dict] = []
    for path in sorted(logits_dir.glob(f"{stage_name}_{P.SUBSET}_shard*of*_state.json")):
        try:
            states.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return states


# ---------------------------------------------------------------------------
# Finalize (importable; used by both the live run and the test-suite).
# ---------------------------------------------------------------------------
def finalize_stage(
    logits_dir: Path,
    stage_num: int,
    chunk_size: int,
    *,
    validate: bool = True,
    tokenizer=None,
) -> dict:
    """Verify coverage, write the canonical resume state, optionally validate.

    Returns a result dict with ``status`` in {PASS, COVERAGE_FAIL, ALIGN_FAIL,
    MISSING}. Pure on top of on-disk shards + worker state files, so a test can
    drive it after running workers in-process.
    """
    stage_name = f"stage{stage_num}"
    jsonl_path = P.STAGE_FILES[stage_num]
    if not jsonl_path.exists():
        return {"stage": stage_name, "status": "MISSING", "reason": "dataset_missing"}

    states = _collect_worker_states(logits_dir, stage_name)
    total_sequences = aggregate_total_sequences(states)
    ok, missing = verify_stage_coverage(logits_dir, stage_name, total_sequences, chunk_size)
    if not ok:
        return {
            "stage": stage_name, "status": "COVERAGE_FAIL",
            "total_sequences": total_sequences, "missing_blocks": missing[:32],
            "missing_count": len(missing),
        }

    # Canonical resume state so _stage_complete / has_precomputed_logits /
    # titan_preflight all see this stage as complete (raw-line keyed).
    total_lines = P._count_jsonl(jsonl_path)
    P._save_resume_state(
        logits_dir, stage_name,
        lines_consumed=total_lines, sequences_emitted=total_sequences,
    )

    result = {
        "stage": stage_name, "status": "PASS",
        "total_sequences": total_sequences,
        "blocks": len(P.block_first_seq_indices(total_sequences, chunk_size)),
    }

    if validate:
        try:
            from scripts.validate_logit_alignment import validate_stage
            from utils.tokenizer_resolver import resolve_tokenizer
            from config.config import cfg as _cfg
            tok = tokenizer if tokenizer is not None else resolve_tokenizer(_cfg)
            res = validate_stage(stage_num, logits_dir, tok)
            result["alignment"] = res.get("status")
            result["alignment_reason"] = res.get("reason_code")
            if res.get("status") not in ("PASS", "MISSING"):
                result["status"] = "ALIGN_FAIL"
        except Exception as exc:  # pragma: no cover - validator import/runtime guard
            result["status"] = "ALIGN_FAIL"
            result["alignment_reason"] = f"validator_error:{exc}"

    return result


# ---------------------------------------------------------------------------
# Live run (subprocess fan-out across GPUs).
# ---------------------------------------------------------------------------
_CHILDREN: list[subprocess.Popen] = []


def _terminate_children() -> None:
    for proc in _CHILDREN:
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass


def _run_stage(
    stage_num: int,
    gpu_ids: list[int],
    *,
    top_k: int,
    chunk_size: int,
    batch_size: int,
    max_seq: int,
    logits_dir: Path,
    model_id: str,
    max_retries: int,
    validate: bool,
    log_dir: Path,
    interrupted: Optional[dict] = None,
) -> dict:
    interrupted = interrupted if interrupted is not None else {"flag": False}
    stage_name = f"stage{stage_num}"
    jsonl_path = P.STAGE_FILES[stage_num]
    if not jsonl_path.exists():
        logger.warning("%s dataset missing: %s", stage_name, jsonl_path)
        return {"stage": stage_name, "status": "MISSING", "reason": "dataset_missing"}

    num_shards = len(gpu_ids)
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(range(num_shards))  # shard ids that still need to succeed

    for attempt in range(max_retries + 1):
        if interrupted["flag"]:
            # A SIGINT during a prior wave must NOT trigger a retry relaunch of
            # the workers the user just terminated.
            return {"stage": stage_name, "status": "INTERRUPTED"}
        if not pending:
            break
        logger.info(
            "%s: launching %d worker(s) on GPUs %s (attempt %d/%d)",
            stage_name, len(pending), [gpu_ids[i] for i in pending], attempt + 1, max_retries + 1,
        )
        procs: dict[int, subprocess.Popen] = {}
        logs: dict[int, "os.PathLike"] = {}
        for shard_id in pending:
            gpu = gpu_ids[shard_id]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = build_worker_command(
                stage_num, shard_id, num_shards,
                top_k=top_k, chunk_size=chunk_size, batch_size=batch_size,
                max_seq=max_seq, logits_dir=logits_dir, model_id=model_id,
            )
            log_path = log_dir / f"{stage_name}_shard{shard_id}of{num_shards}.log"
            logs[shard_id] = log_path
            handle = open(log_path, "a", encoding="utf-8")
            proc = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT)
            proc._titan_log_handle = handle  # type: ignore[attr-defined]
            procs[shard_id] = proc
            _CHILDREN.append(proc)

        failed: list[int] = []
        for shard_id, proc in procs.items():
            rc = proc.wait()
            try:
                proc._titan_log_handle.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            if proc in _CHILDREN:
                _CHILDREN.remove(proc)
            if rc != 0:
                logger.error(
                    "%s worker shard %d (GPU %d) exited rc=%s; see %s",
                    stage_name, shard_id, gpu_ids[shard_id], rc, logs[shard_id],
                )
                failed.append(shard_id)
            else:
                logger.info("%s worker shard %d (GPU %d) OK", stage_name, shard_id, gpu_ids[shard_id])
        pending = failed
        if interrupted["flag"]:
            # Workers were SIGTERM'd by the handler and surfaced as "failed";
            # do not relaunch them or finalize — honour the interrupt now.
            return {"stage": stage_name, "status": "INTERRUPTED"}

    if pending:
        return {"stage": stage_name, "status": "WORKER_FAIL", "failed_shards": pending}

    # All workers succeeded -> coverage + canonical state + alignment.
    return finalize_stage(logits_dir, stage_num, chunk_size, validate=validate)


def _status_to_exit(results: list[dict]) -> int:
    if any(r["status"] == "WORKER_FAIL" for r in results):
        return EXIT_WORKER_FAIL
    if any(r["status"] in ("COVERAGE_FAIL", "ALIGN_FAIL") for r in results):
        return EXIT_COVERAGE_FAIL
    if results and all(r["status"] == "MISSING" for r in results):
        return EXIT_DATA_MISSING
    return EXIT_OK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-GPU data-parallel Top-K teacher logit precompute orchestrator"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all-stages", action="store_true", help="Process all 5 stages.")
    group.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5], help="Process a single stage.")

    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of GPUs/workers (defaults to detected device count).")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Explicit comma-separated GPU ids, e.g. '0,1,2,3' (overrides --gpus).")
    parser.add_argument("--top-k", type=int, default=P.DEFAULT_TOP_K)
    parser.add_argument("--chunk-size", type=int, default=P.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--batch-size", type=int, default=P.DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-seq", type=int, default=P.DEFAULT_MAX_SEQ)
    parser.add_argument("--logits-dir", type=Path,
                        default=Path(getattr(P.cfg, "precomputed_logits_path", "./datasets/logits/")))
    parser.add_argument("--model-id", type=str,
                        default=getattr(P.cfg, "teacher_model_id", "meta-llama/Llama-3.3-70B-Instruct"))
    parser.add_argument("--max-retries", type=int, default=1,
                        help="Retries for a failed worker (resume is safe). Default 1.")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip the post-run alignment validation (not recommended).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the per-GPU worker plan and exit (no teacher, no GPUs needed).")
    parser.add_argument("--check-complete", action="store_true",
                        help="Exit 0 only if the requested stages already have full coverage.")
    return parser.parse_args()


def _detect_gpu_count() -> int:
    try:
        import torch
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def main() -> int:
    args = parse_args()
    stages = [1, 2, 3, 4, 5] if args.all_stages else [args.stage]
    logits_dir = args.logits_dir.resolve()

    try:
        gpu_ids = resolve_worker_gpus(args.gpus, args.gpu_ids, _detect_gpu_count())
    except ValueError as exc:
        logger.error("GPU resolution failed: %s", exc)
        return EXIT_WORKER_FAIL

    if args.dry_run:
        logger.info("DRY-RUN plan: stages=%s | workers(GPUs)=%s | chunk=%s | batch=%s | top_k=%s",
                    stages, gpu_ids, args.chunk_size, args.batch_size, args.top_k)
        if not gpu_ids:
            logger.info("(no GPUs resolved — pass --gpus N or --gpu-ids to preview a real plan)")
        for stage_num in stages:
            present = P.STAGE_FILES[stage_num].exists()
            for shard_id in range(len(gpu_ids)):
                cmd = build_worker_command(
                    stage_num, shard_id, len(gpu_ids),
                    top_k=args.top_k, chunk_size=args.chunk_size, batch_size=args.batch_size,
                    max_seq=args.max_seq, logits_dir=logits_dir, model_id=args.model_id,
                )
                logger.info("  stage%s shard %d -> GPU %d | data=%s | %s",
                            stage_num, shard_id, gpu_ids[shard_id],
                            "OK" if present else "MISSING", " ".join(cmd))
        return EXIT_OK

    if args.check_complete:
        all_ok = True
        for stage_num in stages:
            stage_name = f"stage{stage_num}"
            states = _collect_worker_states(logits_dir, stage_name)
            S = aggregate_total_sequences(states)
            ok, missing = verify_stage_coverage(logits_dir, stage_name, S, args.chunk_size)
            logger.info("%s: %s | S=%s | missing_blocks=%s",
                        stage_name, "complete" if ok and S > 0 else "incomplete", S, len(missing))
            all_ok = all_ok and ok and S > 0
        return EXIT_OK if all_ok else EXIT_COVERAGE_FAIL

    if not gpu_ids:
        logger.error("No GPUs resolved. A real parallel precompute needs >=1 GPU "
                     "(pass --gpus N / --gpu-ids, or use scripts/precompute_logits_topk.py on CPU).")
        return EXIT_WORKER_FAIL

    if not os.environ.get("HF_TOKEN", "").strip():
        logger.error("HF_TOKEN is required for real teacher logit precomputation.")
        return EXIT_WORKER_FAIL

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logits_dir.mkdir(parents=True, exist_ok=True)

    interrupted = {"flag": False}

    def _on_sigint(signum, frame):  # pragma: no cover - signal path
        interrupted["flag"] = True
        logger.warning("SIGINT received — terminating workers (completed blocks are checkpointed).")
        _terminate_children()

    signal.signal(signal.SIGINT, _on_sigint)

    started = time.time()
    results: list[dict] = []
    for stage_num in stages:
        if interrupted["flag"]:
            break
        res = _run_stage(
            stage_num, gpu_ids,
            top_k=args.top_k, chunk_size=args.chunk_size, batch_size=args.batch_size,
            max_seq=args.max_seq, logits_dir=logits_dir, model_id=args.model_id,
            max_retries=args.max_retries, validate=not args.no_validate, log_dir=REPORT_DIR,
            interrupted=interrupted,
        )
        logger.info("%s result: %s", res.get("stage"), res)
        results.append(res)
        if res["status"] == "INTERRUPTED":
            break
        if res["status"] in ("WORKER_FAIL", "COVERAGE_FAIL", "ALIGN_FAIL"):
            logger.error("Stopping: %s did not finalize cleanly.", res.get("stage"))
            break

    report = {
        "stages": stages,
        "gpu_ids": gpu_ids,
        "num_shards": len(gpu_ids),
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "top_k": args.top_k,
        "elapsed_s": round(time.time() - started, 2),
        "interrupted": interrupted["flag"],
        "results": results,
    }
    try:
        (REPORT_DIR / "parallel_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to write parallel report: %s", exc)

    if interrupted["flag"]:
        return EXIT_INTERRUPT
    code = _status_to_exit(results)
    logger.info("Parallel precompute finished: exit=%s", code)
    return code


if __name__ == "__main__":
    sys.exit(main())
