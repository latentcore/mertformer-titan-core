import ast
from pathlib import Path


def _tree() -> ast.AST:
    text = Path("train/train.py").read_text(encoding="utf-8")
    return ast.parse(text)


def _is_scheduler_step_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "scheduler"
    )


def test_scheduler_step_singleton():
    tree = _tree()
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _is_scheduler_step_call(n)]
    assert len(calls) == 1


def test_global_step_singleton():
    tree = _tree()
    inc = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AugAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "global_step"
        and isinstance(n.op, ast.Add)
        and isinstance(n.value, ast.Constant)
        and n.value.value == 1
    ]
    assert len(inc) == 1


def _has_student_eval(node: ast.If) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if isinstance(n.func.value, ast.Name) and n.func.value.id == "student" and n.func.attr == "eval":
                return True
    return False


def _has_try_block(node: ast.If) -> bool:
    return any(isinstance(n, ast.Try) for n in ast.walk(node))


def test_validation_block_scoped():
    tree = _tree()
    validation_guards = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and "global_step % val_check_interval == 0" in ast.unparse(n.test)
    ]
    assert validation_guards, "Validation guard not found"
    guard = validation_guards[0]
    assert _has_student_eval(guard), "Validation should call student.eval() inside guard"
    assert _has_try_block(guard), "Validation try-block should be inside guard"


def _is_accelerator_reduce_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "reduce"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "accelerator"
    )


def _is_torch_dist_broadcast_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "broadcast":
        return False
    mid = node.func.value
    if not isinstance(mid, ast.Attribute) or mid.attr != "distributed":
        return False
    root = mid.value
    return isinstance(root, ast.Name) and root.id == "torch"


def test_validation_ddp_sync_primitives_present():
    tree = _tree()
    reduce_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _is_accelerator_reduce_call(n)]
    broadcast_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _is_torch_dist_broadcast_call(n)]
    assert reduce_calls, "Expected accelerator.reduce for global validation aggregation"
    assert broadcast_calls, "Expected torch.distributed.broadcast for global early-stop decision"


def test_seed_includes_process_index():
    source = Path("train/train.py").read_text(encoding="utf-8")
    assert "set_seed(cfg.seed + accelerator.process_index)" in source


def test_wandb_is_optional_import():
    source = Path("train/train.py").read_text(encoding="utf-8")
    assert "try:\n    import wandb\nexcept ImportError:\n    wandb = None" in source


def test_runtime_manifest_does_not_overwrite_readiness_manifest():
    source = Path("train/train.py").read_text(encoding="utf-8")
    assert "training_runtime_manifest.json" in source
    assert "write_training_runtime_manifest(" in source
    assert "write_training_readiness_manifest(" not in source
    assert 'project_root / "reports" / "training_readiness_manifest.json"' not in source


def test_resume_requires_exact_model_keys_by_default():
    source = Path("train/train.py").read_text(encoding="utf-8")
    assert "TITAN_RESUME_ALLOW_PARTIAL" in source
    assert "Default closure policy requires exact model-state compatibility" in source
    assert "missing_keys" in source
    assert "unexpected_keys" in source


def test_teacher_bitsandbytes_load_has_non_quantized_fallback():
    source = Path("train/train.py").read_text(encoding="utf-8")
    assert "import bitsandbytes" in source
    assert "quantization_config" in source
    assert "bitsandbytes missing; loading teacher without 4-bit quantization" in source


def test_sop_summary_accepts_current_secret_scan_success_line():
    source = Path("scripts/one_command_full_sop.sh").read_text(encoding="utf-8")
    assert 'line.startswith("OK: no secret patterns detected in ") and "files." in line' in source


def test_pre_zip_cache_cleanup_is_followed_by_runtime_clean_check():
    sop = Path("scripts/one_command_full_sop.sh").read_text(encoding="utf-8")
    final = Path("scripts/final_one_shot.sh").read_text(encoding="utf-8")

    assert sop.index('run_step "pre_zip_cache_cleanup"') < sop.index(
        'run_step "pre_zip_runtime_clean_check"'
    ) < sop.index('run_step "release_build30"')
    assert final.index('run_step "pre_zip_cache_cleanup"') < final.index(
        'run_step "pre_zip_runtime_clean_check"'
    ) < final.index("run_zip_with_tolerance artifacts/mertformer_release.zip")


def test_kd_loss_called_with_padding_mask():
    tree = _tree()
    kd_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "kd_loss_safe"
    ]
    assert kd_calls, "Expected kd_loss_safe call in training loop"
    assert any(any(k.arg == "mask" for k in call.keywords) for call in kd_calls), (
        "Expected kd_loss_safe to receive a mask keyword (pad exclusion)"
    )


def _is_sync_gradients_guard(node: ast.If) -> bool:
    return ast.unparse(node.test) == "accelerator.sync_gradients"


def test_grad_norm_reads_are_sync_step_guarded():
    tree = _tree()
    guards = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and _is_sync_gradients_guard(n)
    ]
    assert len(guards) == 1, "Expected one accelerator.sync_gradients optimizer-step guard"

    guard = guards[0]
    body_nodes = [n for stmt in guard.body for n in ast.walk(stmt)]
    assert any(
        isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == "grad_norm"
        for n in body_nodes
    ), "Expected grad_norm to be assigned inside the sync_gradients guard"
    assert any(isinstance(stmt, ast.Continue) for stmt in guard.orelse), (
        "Non-sync accumulation micro-steps must continue before grad_norm is read"
    )

    grad_norm_loads = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == "grad_norm"
    ]
    assert grad_norm_loads, "Expected grad_norm to be read after the optimizer step"
    assert all(n.lineno > guard.lineno for n in grad_norm_loads), (
        "grad_norm must not be read before the sync_gradients guard can assign it"
    )
