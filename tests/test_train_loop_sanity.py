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
