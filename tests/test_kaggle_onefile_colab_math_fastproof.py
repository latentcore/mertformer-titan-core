from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

import torch
import torch.nn as nn

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import scripts.kaggle_onefile_demo_build30_colab_math_fastproof as onefile
import scripts.zip_denylist_audit as zip_audit


def test_math_generator_format():
    rows = onefile.mathfp_generate_math_records(
        n=120,
        seed=7,
        min_value=-20,
        max_value=20,
        include_negative=True,
        ops=["+", "-", "*", "/"],
    )
    assert len(rows) >= 100
    for row in rows:
        text = str(row["full_text"])
        assert "=" in text
        left, right = text.split("=", 1)
        expected = int(str(row["answer"]))
        a_s, op, b_s = left.strip().split(" ")
        a = int(a_s)
        b = int(b_s)
        if op == "+":
            assert a + b == expected
        elif op == "-":
            assert a - b == expected
        elif op == "*":
            assert a * b == expected
        elif op == "/":
            assert b != 0
            assert a // b == expected
            assert a % b == 0
        else:
            raise AssertionError(f"unexpected op: {op}")
        assert right.strip() == str(expected)


def test_division_is_integer_safe():
    rows = onefile.mathfp_generate_math_records(
        n=300,
        seed=17,
        min_value=-80,
        max_value=80,
        include_negative=True,
        ops=["/"],
    )
    assert rows
    for row in rows:
        left = str(row["prompt"]).replace("=", "").strip()
        a_s, op, b_s = left.split(" ")
        assert op == "/"
        a = int(a_s)
        b = int(b_s)
        c = int(str(row["answer"]))
        assert b != 0
        assert a % b == 0
        assert a // b == c


def test_answer_only_loss_mask():
    tok = onefile.SimpleTokenizer(vocab_size=128)
    tok.fit(["12 + 7 = 19", "1 + 1 = 2"])
    x, labels = onefile.mathfp_build_answer_only_tensors(tok, "12 + 7 = 19", seq_len=32)
    eq_id = tok.stoi["="]
    eq_positions = [i for i, t in enumerate(x.tolist()) if int(t) == int(eq_id)]
    assert eq_positions
    eq_pos = eq_positions[0]
    # labels are shifted by one; positions <= eq should stay ignored.
    for idx, value in enumerate(labels.tolist()):
        if (idx + 1) <= eq_pos:
            assert int(value) == -100


def test_architecture_selector_prompt_fallback():
    cfg = dict(onefile.RUN_CONFIG)
    cfg["startup_prompt_enabled"] = True
    cfg["allow_notebook_input"] = False
    cfg["force_interactive_input"] = False
    cfg["architecture_mode"] = "both"
    resolved = onefile.mathfp_prompt_architecture(cfg)
    assert resolved["architecture_mode"] == "both"


def test_both_mode_halves_steps():
    cfg = {
        "architecture_mode": "both",
        "max_steps": 120,
    }
    variants = ["our_mertformer", "gpt_proxy_dense", "gemini_proxy_moe"]
    alloc = onefile.mathfp_allocate_steps(cfg, variants)
    assert alloc["our_mertformer"] == 60
    assert alloc["gpt_proxy_dense"] + alloc["gemini_proxy_moe"] == 60


def test_all_extensions_default_on():
    assert bool(onefile.RUN_CONFIG.get("mert_enable_all_extensions", False)) is True
    assert bool(onefile.RUN_CONFIG.get("mert_use_qinn", False)) is True
    assert str(onefile.RUN_CONFIG.get("task_mode", "")) == "math_eq_answer"


class _TinyLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.emb = nn.Embedding(self.vocab_size, 32)
        self.head = nn.Linear(32, self.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        x = self.emb(input_ids)
        return self.head(x)


def test_logging_artifacts_written_and_compare_schema(tmp_path: Path, monkeypatch):
    def _mock_models(cfg, vocab_size):
        return {
            "our_mertformer": _TinyLM(vocab_size),
            "gpt_proxy_dense": _TinyLM(vocab_size),
            "gemini_proxy_moe": _TinyLM(vocab_size),
        }

    monkeypatch.setattr(onefile, "mathfp_build_variant_models", _mock_models)

    cfg = onefile.resolve_runtime_config(dict(onefile.RUN_CONFIG))
    cfg["out_dir"] = str(tmp_path / "out")
    cfg["artifact_root"] = str(tmp_path / "out")
    cfg["checkpoint_dir"] = str(tmp_path / "out" / "checkpoints")
    cfg["write_files"] = True
    cfg["startup_prompt_enabled"] = False
    cfg["experimental_toggle_prompt"] = False
    cfg["architecture_mode"] = "our"
    cfg["other_proxy_mode"] = "both"
    cfg["max_steps"] = 2
    cfg["batch_size"] = 2
    cfg["seq_len"] = 24
    cfg["math_num_train"] = 24
    cfg["math_num_val"] = 8
    cfg["math_num_test"] = 8

    layout = onefile.init_artifact_layout(cfg)
    payload = onefile.run_math_fastproof(
        cfg=cfg,
        layout=layout,
        device="cpu",
        gpu_meta={},
        gpu_tune={},
        total_start=time.time(),
    )

    out_files = payload["output_files"]
    must_exist = [
        "run_log_jsonl",
        "step_metrics_csv",
        "summary_json",
        "compare_json",
        "compare_csv",
        "compare_md",
        "health_txt",
        "artifact_index",
    ]
    for key in must_exist:
        assert key in out_files, key
        assert Path(out_files[key]).exists(), key

    compare = payload["compare"]
    assert payload.get("schema") == "build30_colab_math_fastproof_payload_v2"
    assert compare.get("schema") == "build30_colab_math_fastproof_compare_v2"
    assert "speedup_ratio_vs_gpt_proxy" in compare
    assert "speedup_ratio_vs_gemini_proxy" in compare
    assert "quality_delta_exact_match" in compare
    assert "exact_match_unseen_our" in compare
    assert "loss_gate_pass" in compare
    assert "accuracy_gate_pass" in compare
    assert "speed_gate_pass" in compare
    assert "feature_coverage_matrix" in payload
    assert float(payload["feature_coverage_matrix"].get("coverage_completeness_percent", 0.0)) == 100.0
    assert "compile_stall_guard" in payload


def test_readme_mentions_new_script():
    files = [
        Path(project_root) / "README.md",
        Path(project_root) / "README_TR.md",
        Path(project_root) / "scripts" / "README.md",
        Path(project_root) / "scripts" / "README_TR.md",
    ]
    for p in files:
        text = p.read_text(encoding="utf-8")
        assert "kaggle_onefile_demo_build30_colab_math_fastproof.py" in text


def test_zip_denylist_audit(tmp_path: Path):
    z = tmp_path / "demo.zip"
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("scripts/main.py", "print('ok')\n")
        zf.writestr("__pycache__/bad.pyc", "x")
    report = zip_audit.audit_zip(z)
    assert report["ok"] is False
    assert report["deny_count"] >= 1
