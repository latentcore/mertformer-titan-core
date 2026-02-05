from pathlib import Path
import re


def test_scheduler_step_singleton():
    text = Path("train/train.py").read_text(encoding="utf-8")
    assert text.count("scheduler.step()") == 1


def test_global_step_singleton():
    text = Path("train/train.py").read_text(encoding="utf-8")
    assert text.count("global_step += 1") == 1


def test_validation_block_scoped():
    text = Path("train/train.py").read_text(encoding="utf-8")
    guard = "if global_step % val_check_interval == 0"
    start = text.find(guard)
    assert start != -1, "Validation guard not found"

    # Limit scan to the validation block region
    end = text.find("if global_step % cfg.save_interval", start)
    if end == -1:
        end = text.find("micro_step += 1", start)
    segment = text[start:end]

    assert "student.eval()" in segment, "Validation should call student.eval() inside guard"
    assert "try:" in segment, "Validation try-block should be inside guard"
