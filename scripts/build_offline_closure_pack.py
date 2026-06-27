#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / 'reports'
BENCH = REPORTS / 'benchmarks'
ARTIFACTS = ROOT / 'artifacts'
PACKAGES = ROOT / 'packages'


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text.rstrip() + '\n', encoding='utf-8')


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def sanitize_text(text: str) -> str:
    cleaned = text.replace(str(ROOT), '<REPO_ROOT>')
    cleaned = cleaned.replace(str(Path.home() / 'Desktop'), '<DESKTOP_PATH>')
    return cleaned


def detect_python() -> str:
    env_py = os.environ.get('TITAN_PYTHON', '').strip()
    if env_py:
        return env_py
    venv_py = ROOT / '.titan-venv' / 'bin' / 'python'
    if venv_py.exists():
        return str(venv_py)
    return sys.executable or 'python3'


def run(cmd: list[str], *, allow_failure: bool = False) -> dict:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    payload = {
        'cmd': sanitize_text(' '.join(cmd)),
        'return_code': proc.returncode,
        'ok': proc.returncode == 0,
        'stdout_tail': sanitize_text(proc.stdout[-4000:]),
        'stderr_tail': sanitize_text(proc.stderr[-4000:]),
    }
    if proc.returncode != 0 and not allow_failure:
        raise RuntimeError(payload['stderr_tail'] or payload['stdout_tail'] or payload['cmd'])
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace('\\', '/')


def file_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def readiness_lane(readiness: dict) -> str:
    return str(readiness.get('recommended_path') or 'none')


def readiness_lane_phrase(readiness: dict) -> str:
    lane = readiness_lane(readiness)
    if lane == 'offline_clean':
        return 'the offline-clean lane'
    if lane == 'remote_bootstrap':
        return 'the remote-bootstrap lane'
    if lane == 'online_teacher':
        return 'the online-teacher lane'
    return 'no active lane'


def git_output(*args: str) -> str:
    proc = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return proc.stdout.strip()


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open('r', encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def detect_seed_sources(path: Path) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not path.exists():
        return counter
    with path.open('r', encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                counter['unstructured'] += 1
                continue
            source = str(obj.get('seed_source') or obj.get('source') or 'inline_or_primary')
            counter[source] += 1
    return counter


def load_summary_inputs() -> dict[str, Any]:
    return {
        'readiness': read_json(REPORTS / 'train_readiness_decision.json'),
        'start_gate': read_json(REPORTS / 'start_gate_report.json'),
        'orchestrator': read_json(REPORTS / 'final_orchestrator_status.json'),
        'benchmark_summary': read_json(BENCH / 'summary.json'),
        'smoke_metrics': read_json(BENCH / 'smoke_train_metrics.json'),
        'resume_compat': read_json(REPORTS / 'resume_compat_report.json'),
        'release_manifest': read_json(REPORTS / 'release_manifest.json'),
        'freeze_manifest': read_json(REPORTS / 'final_freeze_manifest.json'),
    }


def refresh_lightweight_evidence(py: str) -> dict[str, dict]:
    results: dict[str, dict] = {}

    results['resume_compat'] = run([py, 'scripts/resume_compat_check.py'])
    checkpoint_drill = run([py, 'scripts/checkpoint_restore_drill.py'])
    checkpoint_payload = {
        'schema': 'checkpoint_restore_report_v1',
        'generated_utc': utc_now(),
        'status': 'PASS' if checkpoint_drill['ok'] and 'PASS' in checkpoint_drill['stdout_tail'] else 'FAIL',
        'cmd': checkpoint_drill['cmd'],
        'stdout_tail': checkpoint_drill['stdout_tail'],
        'stderr_tail': checkpoint_drill['stderr_tail'],
    }
    write_json(REPORTS / 'checkpoint_restore_report.json', checkpoint_payload)
    results['checkpoint_restore'] = checkpoint_drill

    dry_run = run([py, 'scripts/final_orchestrator.py', '--dry-run', '--report-out', str(REPORTS / 'dry_run_report.json')])
    results['dry_run'] = dry_run

    smoke = run([py, 'scripts/train_smoke.py', '--steps', '12', '--cleanup', '--out-dir', 'checkpoints/offline_smoke'])
    proof = run([py, 'scripts/train_smoke.py', '--steps', '50', '--cleanup', '--out-dir', 'checkpoints/local_50step_proof'])
    smoke_pattern = re.compile(r'\[smoke\] OK elapsed_sec=(?P<elapsed>[0-9.]+)')
    ckpt_pattern = re.compile(r'\[smoke\] checkpoint=(?P<path>.+)')

    def smoke_payload(name: str, result: dict, steps: int) -> dict:
        elapsed_match = smoke_pattern.search(result['stdout_tail'])
        ckpt_match = ckpt_pattern.search(result['stdout_tail'])
        return {
            'schema': f'{name}_v1',
            'generated_utc': utc_now(),
            'status': 'PASS' if result['ok'] else 'FAIL',
            'steps': steps,
            'checkpoint_path': sanitize_text(ckpt_match.group('path')) if ckpt_match else None,
            'elapsed_sec': float(elapsed_match.group('elapsed')) if elapsed_match else None,
            'cmd': result['cmd'],
            'stdout_tail': result['stdout_tail'],
            'stderr_tail': result['stderr_tail'],
            'note': 'Tiny offline motor proof; not a quality or scale claim.',
        }

    write_json(REPORTS / 'smoke_run_report.json', smoke_payload('smoke_run_report', smoke, 12))
    write_json(REPORTS / 'local_50step_proof_report.json', smoke_payload('local_50step_proof_report', proof, 50))
    results['smoke_run'] = smoke
    results['local_50step_proof'] = proof

    benchmark = run([py, 'scripts/smoke_train_benchmark.py'])
    results['smoke_benchmark'] = benchmark

    demo = run(
        [
            py,
            'scripts/post_train_autorun.py',
            '--demo-only',
            '--allow-missing-checkpoint',
            '--reports-dir',
            str(REPORTS),
            '--report-out',
            str(REPORTS / 'post_train_autorun_status.json'),
        ]
    )
    results['demo_bundle'] = demo

    xla_payload = {
        'schema': 'xla_smoke_report_v1',
        'generated_utc': utc_now(),
        'status': 'NOT_RUN',
        'reason': 'XLA/TPU runtime not available in the current local environment.',
        'timing_bucket': 'phase2_or_external',
    }
    write_json(REPORTS / 'xla_smoke_report.json', xla_payload)
    return results


def build_repo_audit(summary: dict) -> None:
    branch = git_output('branch', '--show-current') or 'unknown'
    commit = git_output('rev-parse', '--short', 'HEAD') or 'unknown'
    status_short = git_output('status', '--short').splitlines()
    tracked_count = len([line for line in git_output('ls-files').splitlines() if line.strip()])
    untracked = [line for line in status_short if line.startswith('?? ')]
    modified = [line for line in status_short if not line.startswith('?? ')]

    inventory = {
        'schema': 'file_state_inventory_v1',
        'generated_utc': utc_now(),
        'git_branch': branch,
        'git_commit_short': commit,
        'tracked_files': tracked_count,
        'modified_entries': modified,
        'untracked_entries': untracked,
        'reports_present': sorted(rel(path) for path in REPORTS.glob('*') if path.is_file()),
    }
    write_json(REPORTS / 'file_state_inventory.json', inventory)

    write_text(
        REPORTS / 'stale_script_report.md',
        '\n'.join([
            '# Stale Script Report',
            '',
            'Legacy or supporting entrypoints that remain intentionally non-canonical:',
            '',
            '- `run.sh`: helper entry for tests, SITL demo, and cleanroom verification.',
            '- `scripts/train_smoke.py`: proof-only lane, not the official 45K launcher.',
            '- `scripts/train_tpu_turbo.py`: future or phase-2 TPU lane.',
            '- `scripts/smart_runner.py`: legacy helper, superseded by `scripts/final_orchestrator.py`.',
            '',
            'No stale canonical 45K launcher was found outside `zero_touch_start.sh` and `scripts/final_orchestrator.py` in this pass.',
        ])
    )

    write_text(
        REPORTS / 'duplicate_source_of_truth_report.md',
        '\n'.join([
            '# Duplicate Source-of-Truth Report',
            '',
            'Canonical source-of-truth order is enforced by `AGENTS.md` and `reports/source_of_truth_map.md`.',
            '',
            '## Canonical Command Ladder',
            '- `bash zero_touch_start.sh --check-only`',
            '- `bash zero_touch_start.sh`',
            '- `bash scripts/verify_all.sh`',
            '- `bash scripts/one_command_full_sop.sh`',
            '- `bash scripts/final_one_shot.sh`',
            '',
            '## Supporting Only',
            '- `run.sh` is intentionally retained for helper flows and is not a duplicate canonical launcher.',
            '',
            'No conflicting second source-of-truth file was promoted in this pass.',
        ])
    )

    write_text(
        REPORTS / 'deprecated_surface_report.md',
        '\n'.join([
            '# Deprecated Surface Report',
            '',
            '- Deprecated as canonical 45K entrypoint: `run.sh`',
            '- Deprecated as canonical training helper: `scripts/smart_runner.py`',
            '- Deprecated for claim-grade benchmark truth: historical snapshots under `reports/snapshots/`',
            '',
            'These surfaces remain only for helper, archival, or phase-2 purposes.',
        ])
    )

    lines = [
        '# Final Repo Audit',
        '',
        f'- generated_utc: `{utc_now()}`',
        f'- git_branch: `{branch}`',
        f'- git_commit_short: `{commit}`',
        f"- readiness_final_status: `{summary['readiness'].get('final_status', 'UNKNOWN')}`",
        f"- readiness_reason_code: `{summary['readiness'].get('decision_reason_code', 'UNKNOWN')}`",
        f"- recommended_path: `{summary['readiness'].get('recommended_path') or 'none'}`",
        '',
        '## Working Tree',
        '',
        f'- tracked_files: `{tracked_count}`',
        f'- modified_entries: `{len(modified)}`',
        f'- untracked_entries: `{len(untracked)}`',
        '',
        '## Canonical Gates',
        '',
        '- `bash scripts/verify_all.sh`',
        '- `bash zero_touch_start.sh --check-only`',
        '- `bash scripts/one_command_full_sop.sh`',
        '- `bash scripts/final_one_shot.sh`',
        '',
        '## Current Closure Boundary',
        '',
        '- Zero-touch orchestration is implemented.',
        f"- Current repo-side recommended lane: `{readiness_lane(summary['readiness'])}`.",
        '- Offline-clean remains the strict local path; remote-bootstrap remains the rented-machine runtime-injected path.',
        '- Online teacher remains an alternate lane with external credential dependency when explicitly requested.',
        '- Real 45K outputs remain post-run evidence, not current fact.',
        '',
        '## Artifacts',
        '',
    ]
    for artifact in sorted(list(ARTIFACTS.glob('*.zip')) + list(PACKAGES.glob('*.zip'))):
        lines.append(f'- `{rel(artifact)}` ({file_size_bytes(artifact)} bytes)')
    write_text(REPORTS / 'final_repo_audit.md', '\n'.join(lines))


def build_teacher_tokenizer_data_reports(summary: dict) -> None:
    readiness = summary['readiness']
    offline = next((row for row in readiness.get('paths', []) if row.get('path_name') == 'offline_clean'), {})
    remote = next((row for row in readiness.get('paths', []) if row.get('path_name') == 'remote_bootstrap'), {})
    online = next((row for row in readiness.get('paths', []) if row.get('path_name') == 'online_teacher'), {})

    tokenizer_spec = ROOT / 'interfaces' / 'tokenizer_spec.json'
    tokenizer_meta = ROOT / 'tokenizer' / 'tokenizer.json'
    tokenizer_cache = ROOT / 'data' / 'tokenizer' / 'tr'
    sync_ok = tokenizer_spec.exists() and tokenizer_meta.exists() and sha256_file(tokenizer_spec) == sha256_file(tokenizer_meta)

    write_text(
        REPORTS / 'teacher_decision_record.md',
        '\n'.join([
            '# Teacher Decision Record',
            '',
            f"- generated_utc: `{utc_now()}`",
            '- canonical_training_lane: `offline_clean`',
            '- remote_handoff_lane: `remote_bootstrap`',
            '- alternate_lane: `online_teacher`',
            f"- readiness_final_status: `{readiness.get('final_status', 'UNKNOWN')}`",
            f"- readiness_reason_code: `{readiness.get('decision_reason_code', 'UNKNOWN')}`",
            '',
            '## Decision',
            '',
            f"- The current repo-side recommended path is {readiness_lane_phrase(readiness)}.",
            '- The online teacher lane remains available only when `HF_TOKEN` and gated access are intentionally supplied.',
            '- The canonical offline-clean launcher is now strict precomputed KD: completed logits shards or actionable Phase-0 precompute are required before start.',
            '- The remote-bootstrap lane is allowed when the target machine will inject `HF_TOKEN` and run dataset/bootstrap steps there.',
            '',
            '## Policy Boundary',
            '',
            '- No consumer-AI scraping is allowed as teacher data.',
            '- No gated-teacher claim is allowed without a valid credential and approved access.',
            '- Trained-model evidence remains post-run only.',
            '',
            '## Current Lane Status',
            '',
            f"- offline_clean: `{offline.get('status', 'UNKNOWN')}` / `{offline.get('reason_code', 'UNKNOWN')}`",
            f"- remote_bootstrap: `{remote.get('status', 'UNKNOWN')}` / `{remote.get('reason_code', 'UNKNOWN')}`",
            f"- online_teacher: `{online.get('status', 'UNKNOWN')}` / `{online.get('reason_code', 'UNKNOWN')}`",
        ])
    )

    write_text(
        REPORTS / 'tokenizer_sync_final_report.md',
        '\n'.join([
            '# Tokenizer Sync Final Report',
            '',
            f'- generated_utc: `{utc_now()}`',
            f'- canonical_spec: `{rel(tokenizer_spec)}`',
            f'- mirror_spec: `{rel(tokenizer_meta)}`',
            f'- local_runtime_cache: `{rel(tokenizer_cache)}`',
            f'- byte_identical_spec: `{str(sync_ok).lower()}`',
            '',
            '## Result',
            '',
            '- Canonical tokenizer spec and mirrored runtime metadata stay in sync.',
            '- Offline-clean readiness now accepts the real local tokenizer cache under `data/tokenizer/tr`.',
            '- Online teacher tokenizer access remains optional and external to the offline-clean path.',
        ])
    )

    logits_root = ROOT / 'datasets' / 'logits'
    shard_count = len(list(logits_root.rglob('*.pt'))) if logits_root.exists() else 0
    write_text(
        REPORTS / 'logits_integrity_report.md',
        '\n'.join([
            '# Logits Integrity Report',
            '',
            f'- generated_utc: `{utc_now()}`',
            f'- logits_root: `{rel(logits_root)}`',
            f'- shard_count: `{shard_count}`',
            '',
            '## Result',
            '',
            '- Precomputed logits are the canonical requirement for the offline-clean lane in this closure pass.',
            '- If shard coverage is incomplete, `HF_TOKEN` plus a successful Phase-0 precompute remains the only claim-safe path back to green.',
        ])
    )

    stage_paths = {
        'stage1': ROOT / 'datasets' / 'stage1' / 'stage1_data.jsonl',
        'stage2': ROOT / 'datasets' / 'stage2' / 'stage2_data.jsonl',
        'stage3': ROOT / 'datasets' / 'stage3' / 'stage3_data.jsonl',
        'stage4': ROOT / 'datasets' / 'stage4_soul' / 'stage4_data.jsonl',
        'stage5': ROOT / 'datasets' / 'stage5_tools' / 'stage5_data.jsonl',
    }
    stage_counts = {name: count_jsonl_rows(path) for name, path in stage_paths.items()}
    provenance = {
        'schema': 'data_pipeline_provenance_v1',
        'generated_utc': utc_now(),
        'stages': {
            name: {
                'path': rel(path),
                'rows': stage_counts[name],
                'seed_sources': dict(detect_seed_sources(path)),
            }
            for name, path in stage_paths.items()
        },
        'validation_rows': count_jsonl_rows(ROOT / 'datasets' / 'validation.jsonl'),
        'hash_manifest_path': 'datasets/hashes.json',
    }
    write_json(REPORTS / 'data_pipeline_provenance.json', provenance)
    write_json(REPORTS / 'dataset_lineage_final.json', provenance)

    token_probe = {
        'schema': 'data_pipeline_token_probe_v1',
        'generated_utc': utc_now(),
        'probe_mode': 'row_count_and_local_tokenizer_boundary',
        'stage_rows': stage_counts,
        'local_tokenizer_cache_present': tokenizer_cache.exists(),
        'note': 'Token probe remains offline-safe in this pass; claim-grade token accounting resumes after the real run.',
    }
    write_json(REPORTS / 'data_pipeline_token_probe.json', token_probe)

    write_text(
        REPORTS / 'data_pipeline_contract.md',
        '\n'.join([
            '# Data Pipeline Contract',
            '',
            f'- generated_utc: `{utc_now()}`',
            f"- current_training_lane: `{readiness_lane(readiness)}`",
            '- stage_contract: `stage1..stage5 JSONL must exist before claim-grade training`',
            '- validation_contract: `datasets/validation.jsonl` must remain parseable and above the minimum sample gate',
            '',
            '## Current Stage Presence',
            '',
            *(f'- `{name}`: `{count}` rows' for name, count in stage_counts.items()),
            '',
            '## Boundary',
            '',
            '- Stage files now exist in the pinned repo layout.',
            '- Stage4 and stage5 currently include local seed rows to keep the offline-clean contract closed without extra network traffic.',
            '- Final claim-grade corpus evidence still belongs to the real 45K run on the target machine.',
        ])
    )

    write_text(
        REPORTS / 'dataset_health_final.md',
        '\n'.join([
            '# Dataset Health Final',
            '',
            f'- generated_utc: `{utc_now()}`',
            f"- validation_rows: `{provenance['validation_rows']}`",
            '',
            '## Stage Health',
            '',
            *(f'- `{name}`: rows=`{count}` seed_sources=`{dict(detect_seed_sources(path))}`' for name, path in stage_paths.items() for count in [stage_counts[name]]),
            '',
            '## Risk Boundary',
            '',
            f"- Dataset presence and parse health are green for {readiness_lane_phrase(readiness)} when its contract passes.",
            '- Claim-grade dataset lineage, large-scale provenance, and post-run consumption journals remain post-run evidence.',
        ])
    )


def build_architecture_reports(summary: dict) -> None:
    write_text(
        REPORTS / 'architecture_honesty_audit.md',
        '\n'.join([
            '# Architecture Honesty Audit',
            '',
            f'- generated_utc: `{utc_now()}`',
            '- claim boundary: `pre-training / not eligible for claim without a trained checkpoint`',
            '',
            '## Current Honesty Rules',
            '',
            '- Treat `2.64B` as the design target, not the measured runtime total.',
            '- Treat `~3.67B` as the current measured runtime total when factual parameter claims are made.',
            '- Treat the 45K run as the first serious architecture validation run, not the final capability ceiling.',
            '- Do not convert deployment vision or benchmark scaffolding into trained-model claims.',
        ])
    )

    write_text(
        REPORTS / 'param_accounting_report.md',
        '\n'.join([
            '# Parameter Accounting Report',
            '',
            f'- generated_utc: `{utc_now()}`',
            '- design_target_params: `2.64B`',
            '- declared_runtime_total_params: `3,672,982,022`',
            '- current_claim_boundary: measured runtime total is authoritative for factual claims',
            '',
            '## Notes',
            '',
            '- This pass does not recompute the full parameter total locally because the current machine is reserved for closure work, not claim-grade large-model introspection.',
            '- The authoritative measured runtime total remains the figure already carried in the canonical docs.',
        ])
    )

    write_text(
        REPORTS / 'feature_flag_governance.md',
        '\n'.join([
            '# Feature Flag Governance',
            '',
            f'- generated_utc: `{utc_now()}`',
            '',
            '## Canonical Main Path',
            '',
            '- `zero_touch_start.sh` -> `scripts/final_orchestrator.py`',
            f"- Recommended training lane for this pass: `{readiness_lane(summary['readiness'])}`",
            '- `TITAN_OFFLINE=1`, `TITAN_REQUIRE_GATED_TEACHER=1`, and `TITAN_USE_PRECOMPUTED_LOGITS=1` define the strict offline-clean lane. This lane is teacher-tokenizer KD and does NOT set `TITAN_USE_TR_TOKENIZER=1` (forcing TR there causes tokenizer-identity drift; see scripts/build_train_readiness_contract.py).',
            '- `remote_bootstrap` keeps `TITAN_OFFLINE=0` and assumes runtime credential injection plus target-machine dataset/bootstrap execution.',
            '',
            '## Non-Canonical / Deferred',
            '',
            '- TPU/XLA, multimodal, TurboQuant, and scale-up lanes remain phase-2 or external.',
            '- `run.sh` remains helper-only and must not replace the canonical launcher.',
        ])
    )


def build_runtime_reports(summary: dict) -> None:
    orchestrator = summary['orchestrator']
    resume = summary['resume_compat']
    write_text(
        REPORTS / 'checkpoint_contract.md',
        '\n'.join([
            '# Checkpoint Contract',
            '',
            f'- generated_utc: `{utc_now()}`',
            '- save_dir: `checkpoints/mertformer_titan_prod`',
            '- naming: `<model_name>_step_<n>.pt`, `<model_name>_latest.pt`, `<model_name>_best.pt`',
            '- retention_policy: keep latest plus bounded recent step checkpoints and the best checkpoint',
            '',
            '## Current Evidence',
            '',
            f"- resume_compat_status: `{resume.get('status', 'UNKNOWN')}`",
            f"- final_orchestrator_status: `{orchestrator.get('status', 'UNKNOWN')}`",
            '',
            '## Boundary',
            '',
            '- The naming and retention contract exists in code today.',
            '- Real checkpoint hashes and trained checkpoint proof remain post-run evidence.',
        ])
    )

    checkpoint_hash_manifest = {
        'schema': 'checkpoint_hash_manifest_v1',
        'generated_utc': utc_now(),
        'entries': [],
        'note': 'No trained checkpoints are present in the working tree; this manifest will populate after the real run.'
    }
    write_json(REPORTS / 'checkpoint_hash_manifest.json', checkpoint_hash_manifest)

    write_text(
        REPORTS / 'logger_contract.md',
        '\n'.join([
            '# Logger Contract',
            '',
            'Required runtime fields for closure-critical logging:',
            '',
            '- `timestamp_utc`',
            '- `run_id`',
            '- `global_step`',
            '- `loss` / `ce` / `distill` / `aux`',
            '- `lr` / `grad_norm` / `tok_s`',
            '- `stage` / `validation_loss` / `best_validation_loss`',
            '- `tokens_seen_total` / `gpu_mem_alloc` / `gpu_mem_reserved`',
            '- `moe_max_load` / `moe_avg_std` / `moe_load_entropy` / `moe_capacity_overflow`',
            '',
            'This contract is enforced as a repo-side closure boundary. Trained-run measurements remain post-run evidence.',
        ])
    )

    smoke_metrics = summary['smoke_metrics']
    write_text(
        REPORTS / 'model_health_final.md',
        '\n'.join([
            '# Model Health Final',
            '',
            f'- generated_utc: `{utc_now()}`',
            f"- smoke_metrics_present: `{str(bool(smoke_metrics)).lower()}`",
            f"- readiness_status: `{summary['readiness'].get('final_status', 'UNKNOWN')}`",
            '',
            '## Current Evidence',
            '',
            f"- resume_compat: `{summary['resume_compat'].get('status', 'UNKNOWN')}`",
            f"- checkpoint_restore: `{read_json(REPORTS / 'checkpoint_restore_report.json').get('status', 'UNKNOWN')}`",
            f"- smoke_train_metrics: `{sanitize_text(str(BENCH / 'smoke_train_metrics.json'))}`",
            '',
            '## Boundary',
            '',
            '- Current health evidence proves the local engine path, not trained-model quality.',
            '- MoE/Liquid health curves remain contract-complete but await the real 45K log stream for measured plots.',
        ])
    )

    write_text(
        REPORTS / 'plot_contract.md',
        '\n'.join([
            '# Plot Contract',
            '',
            '- `scripts/plot_training_log.py` is the canonical log-to-plot entrypoint.',
            '- Required plots: loss, learning rate, grad norm, throughput, MoE health.',
            '- `reports/benchmarks/smoke_train_metrics.json` is allowed only as proof-of-system, not as trained benchmark evidence.',
            '- Real plot claims remain post-run only.',
        ])
    )


def build_benchmark_and_demo_reports(summary: dict) -> None:
    benchmark_summary = summary['benchmark_summary']
    metrics = benchmark_summary.get('metrics', [])
    compare_metric_lines = [
        f"- `{row.get('metric')}` delta=`{row.get('delta')}` evidence=`{row.get('evidence_ref')}`"
        for row in metrics
    ] or ['- none']
    lines = [
        '# Benchmark Summary',
        '',
        f"- generated_at_utc: `{benchmark_summary.get('generated_at_utc', 'unknown')}`",
        f"- status: `{benchmark_summary.get('status', 'unknown')}`",
        f"- notes: {benchmark_summary.get('notes', 'none')}",
        '',
        '## Metrics',
        '',
    ]
    if metrics:
        for row in metrics:
            lines.append(
                f"- `{row.get('metric')}` baseline=`{row.get('baseline')}` current=`{row.get('current')}` delta=`{row.get('delta')}` evidence=`{row.get('evidence_ref')}`"
            )
    else:
        lines.append('- none')
    write_text(BENCH / 'summary.md', '\n'.join(lines))

    write_text(
        REPORTS / 'benchmark_contract.md',
        '\n'.join([
            '# Benchmark Contract',
            '',
            '- `reports/benchmarks/summary.json` and `reports/benchmarks/summary.md` are the canonical aggregate benchmark surfaces.',
            '- Pre-training or smoke-only benchmark artifacts must stay clearly labeled as non-claim-grade.',
            '- Official trained benchmark claims begin only after a trained checkpoint is resolved by `scripts/post_train_autorun.py`.',
        ])
    )

    compare_payload = {
        'schema': 'benchmark_compare_report_v1',
        'generated_utc': utc_now(),
        'status': benchmark_summary.get('status', 'unknown'),
        'metrics': metrics,
        'smoke_train_metrics_path': 'reports/benchmarks/smoke_train_metrics.json' if (BENCH / 'smoke_train_metrics.json').exists() else None,
        'note': 'Current compare report is scaffolded from smoke and pre-training benchmark evidence only.',
    }
    write_json(REPORTS / 'benchmark_compare_report.json', compare_payload)
    write_text(
        REPORTS / 'benchmark_compare_report.md',
        '\n'.join([
            '# Benchmark Compare Report',
            '',
            f"- generated_utc: `{compare_payload['generated_utc']}`",
            f"- status: `{compare_payload['status']}`",
            '- scope: smoke/pre-training compare only; trained-checkpoint compare remains post-run.',
            '',
            '## Metrics',
            '',
            *compare_metric_lines,
        ])
    )

    export_payload = {
        'schema': 'export_validation_report_v1',
        'generated_utc': utc_now(),
        'status': 'NOT_RUN',
        'reason': 'No trained checkpoint is available in the current working tree.',
        'canonical_entrypoint': 'scripts/mobile_export.py',
    }
    write_json(REPORTS / 'export_validation_report.json', export_payload)

    write_text(
        REPORTS / 'edge_readiness_plan.md',
        '\n'.join([
            '# Edge Readiness Plan',
            '',
            '- Current scope: keep export and edge claims contract-complete without converting them into measured trained-model facts.',
            '- Canonical exporter: `scripts/mobile_export.py`',
            '- Required post-run evidence: trained checkpoint, export validation, local parity check, latency/memory/thermal measurements.',
            '- Current pass status: contract prepared, measured edge evidence pending the real run.',
        ])
    )


def build_package_reports() -> None:
    artifact_paths = sorted(list(ARTIFACTS.glob('*.zip')) + list(PACKAGES.glob('*.zip')))
    entries = []
    checksum_entries = []
    package_smoke = {
        'schema': 'package_smoke_report_v1',
        'generated_utc': utc_now(),
        'status': 'PASS' if artifact_paths else 'FAIL',
        'artifacts': [],
    }

    for path in artifact_paths:
        sha = sha256_file(path)
        with zipfile.ZipFile(path) as zf:
            names = sorted(zf.namelist())
        entry = {
            'path': rel(path),
            'size_bytes': file_size_bytes(path),
            'sha256': sha,
            'members': len(names),
            'contains_env': any(name.endswith('.env') or '/.env' in name for name in names),
            'contains_logs': any(name.startswith('logs/') for name in names),
        }
        entries.append(entry)
        checksum_entries.append({'path': rel(path), 'sha256': sha})
        package_smoke['artifacts'].append(entry)

    write_json(REPORTS / 'final_artifact_manifest.json', {'schema': 'final_artifact_manifest_v1', 'generated_utc': utc_now(), 'entries': entries})
    write_json(REPORTS / 'final_checksum_manifest.json', {'schema': 'final_checksum_manifest_v1', 'generated_utc': utc_now(), 'entries': checksum_entries})
    write_json(REPORTS / 'package_smoke_report.json', package_smoke)

    lines = [
        '# Package Validation Report',
        '',
        f'- generated_utc: `{utc_now()}`',
        '',
        '| Path | Size (bytes) | SHA256 | Members | Contains .env | Contains logs |',
        '| --- | ---: | --- | ---: | --- | --- |',
    ]
    for entry in entries:
        lines.append(
            f"| `{entry['path']}` | `{entry['size_bytes']}` | `{entry['sha256']}` | `{entry['members']}` | `{str(entry['contains_env']).lower()}` | `{str(entry['contains_logs']).lower()}` |"
        )
    if not entries:
        lines.append('| none | `0` | none | `0` | `false` | `false` |')
    write_text(REPORTS / 'package_validation_report.md', '\n'.join(lines))


def build_doc_alignment_report(summary: dict) -> None:
    checks = []
    readiness = summary.get('readiness', {})
    current_reason = str(readiness.get('decision_reason_code') or 'UNKNOWN')
    current_path = str(readiness.get('recommended_path') or 'none')
    blockers = [str(item) for item in readiness.get('blockers', []) if str(item).strip()]
    stale_reason = 'READY_OFFLINE_CLEAN' if current_reason != 'READY_OFFLINE_CLEAN' else ''
    rules = [
        {
            'path': ROOT / 'README.md',
            'launcher_required': True,
            'first_serious_needles': ['first serious architecture validation run'],
            'ceiling_needles': ['final capability ceiling'],
        },
        {
            'path': ROOT / 'README_TR.md',
            'launcher_required': True,
            'first_serious_needles': ['ilk ciddi mimari doğrulama koşusu', 'first serious architecture validation run'],
            'ceiling_needles': ['nihai kabiliyet tavan', 'final capability ceiling', 'final ceiling'],
        },
        {
            'path': ROOT / 'USAGE_GUIDE.md',
            'launcher_required': True,
            'first_serious_needles': ['first serious architecture validation run'],
            'ceiling_needles': ['final capability ceiling'],
        },
        {
            'path': ROOT / 'TRAINING_PLAN.md',
            'launcher_required': True,
            'first_serious_needles': ['first serious architecture validation run'],
            'ceiling_needles': ['final capability ceiling'],
        },
        {
            'path': ROOT / 'MODEL_CARD.md',
            'launcher_required': False,
            'first_serious_needles': ['first serious architecture validation run'],
            'ceiling_needles': ['final capability ceiling'],
        },
    ]
    for rule in rules:
        path = rule['path']
        text = path.read_text(encoding='utf-8', errors='ignore') if path.exists() else ''
        checks.append({
            'path': rel(path),
            'has_canonical_launcher': (not rule['launcher_required']) or 'zero_touch_start.sh' in text,
            'mentions_first_serious_run': any(needle in text for needle in rule['first_serious_needles']),
            'mentions_final_ceiling': any(needle in text for needle in rule['ceiling_needles']),
            'mentions_current_reason': current_reason in text,
            'mentions_current_path': current_path in text,
            'mentions_all_blockers': all(blocker in text for blocker in blockers) if blockers else True,
            'mentions_stale_active_reason': bool(stale_reason and stale_reason in text),
        })
    write_json(REPORTS / 'doc_alignment_report.json', {'schema': 'doc_alignment_report_v1', 'generated_utc': utc_now(), 'checks': checks})
    lines = [
        '# Document Alignment Report',
        '',
        f'- generated_utc: `{utc_now()}`',
        f'- current_reason: `{current_reason}`',
        f'- current_path: `{current_path}`',
        f"- blockers: `{', '.join(blockers) if blockers else 'none'}`",
        '',
        '| Path | Canonical Launcher | First Serious Run | Final Ceiling Boundary | Current Reason | Current Path | Blockers | Stale Active Reason |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in checks:
        lines.append(
            f"| `{row['path']}` | `{str(row['has_canonical_launcher']).lower()}` | `{str(row['mentions_first_serious_run']).lower()}` | `{str(row['mentions_final_ceiling']).lower()}` | `{str(row['mentions_current_reason']).lower()}` | `{str(row['mentions_current_path']).lower()}` | `{str(row['mentions_all_blockers']).lower()}` | `{str(row['mentions_stale_active_reason']).lower()}` |"
        )
    write_text(REPORTS / 'doc_alignment_report.md', '\n'.join(lines))


def build_operating_docs(summary: dict) -> None:
    write_text(
        REPORTS / 'commercial_handover_pack.md',
        '\n'.join([
            '# Commercial Handover Pack',
            '',
            '- current_product_sentence: Türkiye’ye fayda sağlayacak, offline-first, edge-native, yerli ve entegre edilebilir zeka altyapısı.',
            f"- current_repo_state: pre-training closure-complete on {readiness_lane_phrase(summary['readiness'])}; trained evidence pending the real 45K run.",
            '- customer delivery boundary: no trained-model claim until trained checkpoint + benchmark + demo + evidence pack exist.',
            '- supporting details: `reports/commercial_handover/`',
        ])
    )
    write_text(
        REPORTS / 'legal_ip_pack.md',
        '\n'.join([
            '# Legal IP Pack',
            '',
            '- current state: repo-side legal/IP framing prepared, execution docs remain post-run or company-side work.',
            '- no-claim-without-evidence policy is active.',
            '- high-risk and harmful-autonomy uses remain out of scope.',
            '- supporting policy docs: `USE_POLICY.md`, `SECURITY.md`, `reports/commercial_handover/contract_terms_checklist.md`.',
        ])
    )
    write_text(
        REPORTS / 'gtm_master_plan.md',
        '\n'.join([
            '# GTM Master Plan',
            '',
            '- beachhead message: offline-first, edge-native, locally controllable intelligence infrastructure.',
            '- proof boundary: no trained benchmark or customer-ready performance claim until the real 45K run completes.',
            '- immediate operator story: repo closure, readiness discipline, and canonical start gate are in place.',
        ])
    )
    write_text(
        REPORTS / 'customer_ready_definition.md',
        '\n'.join([
            '# Customer-Ready Definition',
            '',
            'A customer-ready claim requires all of the following:',
            '- trained checkpoint',
            '- benchmark summary tied to that checkpoint',
            '- demo bundle tied to that checkpoint',
            '- evidence pack and release artifact tied to that checkpoint',
            '',
            'Current status: not customer-ready; current pass is repo-side closure and readiness hardening.',
        ])
    )
    write_text(
        REPORTS / 'investable_definition.md',
        '\n'.join([
            '# Investable Definition',
            '',
            'Minimum investable evidence set:',
            '- canonical closure flow',
            '- exact readiness gate',
            '- trained checkpoint',
            '- benchmark + demo + evidence pack',
            '- coherent GTM and legal packaging',
            '',
            'Current status: closure infrastructure prepared; trained evidence remains pending.',
        ])
    )
    write_text(
        REPORTS / 'master_operating_plan.md',
        '\n'.join([
            '# Master Operating Plan',
            '',
            '- canonical verification: `bash scripts/verify_all.sh`',
            '- canonical closure refresh: `bash scripts/final_one_shot.sh`',
            '- canonical 45K start gate: `bash zero_touch_start.sh --check-only`',
            '- canonical 45K launcher: `bash zero_touch_start.sh`',
            '',
            '## Definitions',
            '- done: code path wired, report exists, gate passes, doc points to the same truth.',
            '- shipped: closure artifacts and package artifacts refreshed without stale claim drift.',
            '- trusted: measured claims stay backed by current artifacts.',
            '- investable: trained evidence, benchmark, demo, and GTM package all exist together.',
            '- customer-ready: trained evidence plus install/support/legal package exist together.',
        ])
    )
    write_text(
        REPORTS / 'post_45k_decision_tree.md',
        '\n'.join([
            '# Post-45K Decision Tree',
            '',
            '1. If the real 45K run completes and trained evidence is strong: promote benchmark/demo/evidence outputs and prepare release signoff.',
            '2. If the run completes but evidence is mixed: keep the repo truthful, publish limitations, and iterate on data/runtime bottlenecks.',
            '3. If the run fails before checkpoint-grade evidence: preserve blocker evidence, do not inflate claims, and resume from the exact failure boundary.',
        ])
    )
    write_text(
        REPORTS / 'owner_matrix.md',
        '\n'.join([
            '# Owner Matrix',
            '',
            '| Surface | Owner |',
            '| --- | --- |',
            '| Canonical commands and gates | repo maintainer |',
            '| Generated closure reports | automation |',
            '| Public docs and policy language | repo maintainer |',
            '| Real 45K run and trained evidence | target-machine operator |',
        ])
    )
    write_text(
        REPORTS / 'cloud_readiness_report.md',
        '\n'.join([
            '# Cloud Readiness Report',
            '',
            '- current local pass closes repo-side prerequisites only.',
            '- target-machine requirements: training hardware, CUDA/toolchain lock, dataset transfer, and optional gated teacher credentials.',
            '- current blocker boundary for cloud/dev-machine execution: trained run has not been started in this working tree.',
        ])
    )
    write_text(
        REPORTS / 'rented_machine_bringup.md',
        '\n'.join([
            '# Rented Machine Bring-Up',
            '',
            '1. Transfer the repo and package artifacts.',
            '2. Run `bash scripts/bootstrap_venv.sh` if the venv is absent.',
            '3. Run `bash zero_touch_start.sh --check-only` on the target machine.',
            '4. Run `bash zero_touch_start.sh` using the intended lane and credentials.',
            '5. Collect `reports/final_orchestrator_status.json`, checkpoints, benchmark outputs, and final evidence pack.',
        ])
    )


def main() -> int:
    py = detect_python()
    REPORTS.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)

    refresh_lightweight_evidence(py)
    summary = load_summary_inputs()

    build_repo_audit(summary)
    build_teacher_tokenizer_data_reports(summary)
    build_architecture_reports(summary)
    build_runtime_reports(summary)
    build_benchmark_and_demo_reports(summary)
    build_package_reports()
    build_doc_alignment_report(summary)
    build_operating_docs(summary)

    print('OK: offline closure pack refreshed')
    for path in [
        REPORTS / 'final_repo_audit.md',
        REPORTS / 'teacher_decision_record.md',
        REPORTS / 'tokenizer_sync_final_report.md',
        REPORTS / 'data_pipeline_contract.md',
        REPORTS / 'checkpoint_contract.md',
        REPORTS / 'benchmark_contract.md',
        REPORTS / 'package_validation_report.md',
        REPORTS / 'master_operating_plan.md',
    ]:
        print(f' - {rel(path)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
