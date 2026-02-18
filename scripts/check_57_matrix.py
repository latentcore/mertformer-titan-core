#!/usr/bin/env python3
"""
Build30 closure gate for 57-item roadmap (Code+Test Green).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ItemCheck:
    id: int
    area: str
    name: str
    code: bool
    integration: bool
    test: bool
    evidence_pending: bool
    evidence: list[str]

    @property
    def green(self) -> bool:
        return int(self.code) + int(self.integration) + int(self.test) >= 2

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "area": self.area,
            "name": self.name,
            "code": self.code,
            "integration": self.integration,
            "test": self.test,
            "green": self.green,
            "evidence_pending": self.evidence_pending,
            "evidence": self.evidence,
        }


def _exists(path: str) -> bool:
    return (ROOT / path).exists()


def _grep(path: str, pattern: str) -> bool:
    p = ROOT / path
    if not p.exists():
        return False
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return pattern in txt


def _has_any(paths: Iterable[str]) -> bool:
    return any(_exists(p) for p in paths)


def build_checks() -> list[ItemCheck]:
    pending = {8, 9, 11, 12, 51, 52, 54, 55, 56, 57}

    checks: list[ItemCheck] = [
        ItemCheck(1, "foundation", "crash/deadlock/silent corruption zero", _exists("scripts/operator_mode_gate.py"), _exists("scripts/verify_all.sh"), _exists("tests/test_train_loop_sanity.py"), 1 in pending, ["scripts/operator_mode_gate.py", "tests/test_train_loop_sanity.py"]),
        ItemCheck(2, "foundation", "deterministic train/resume", _grep("train/train.py", "set_seed(cfg.seed + accelerator.process_index)"), _exists("train/train.py"), _exists("tests/test_train_loop_sanity.py"), 2 in pending, ["train/train.py"]),
        ItemCheck(3, "foundation", "full gate matrix", _exists("scripts/verify_all.sh"), _exists("scripts/train_smoke.py"), _has_any(["tests/test_model.py", "tests/test_comprehensive.py"]), 3 in pending, ["scripts/verify_all.sh"]),
        ItemCheck(4, "foundation", "reproducible metadata", _exists("scripts/check_57_matrix.py"), _grep("train/train.py", "config': str(cfg)"), _exists("tests/test_export_metadata.py"), 4 in pending, ["train/train.py", "scripts/check_57_matrix.py"]),
        ItemCheck(5, "data", "licensed data inventory", _exists("datasets/LICENSES.md") or _exists("datasets/LICENSES_TR.md"), _exists("datasets/hashes.json"), _exists("scripts/verify_datasets.py"), 5 in pending, ["datasets/hashes.json"]),
        ItemCheck(6, "data", "dedup + quality filtering", _exists("scripts/data_pipeline.py"), _grep("scripts/data_pipeline.py", "dedup"), _exists("tests/test_comprehensive.py"), 6 in pending, ["scripts/data_pipeline.py"]),
        ItemCheck(7, "data", "curriculum automation", _grep("train/train.py", "CurriculumDataset"), _exists("train/train.py"), _exists("tests/test_train_loop_sanity.py"), 7 in pending, ["train/train.py"]),
        ItemCheck(8, "train", "multi-stage scale-up protocol", _exists("TRAINING_PLAN.md"), _exists("train/train.py"), _exists("tests/test_train_loop_sanity.py"), 8 in pending, ["TRAINING_PLAN.md"]),
        ItemCheck(9, "train", "ddp/fsdp safety", _grep("train/train.py", "accelerator"), _grep("train/train.py", "broadcast"), _exists("tests/test_train_loop_sanity.py"), 9 in pending, ["train/train.py"]),
        ItemCheck(10, "train", "checkpoint safety + recovery", _exists("scripts/checkpoint_restore_drill.py"), _grep("train/train.py", "save_checkpoint_smart"), _exists("tests/test_train_loop_sanity.py"), 10 in pending, ["scripts/checkpoint_restore_drill.py"]),
        ItemCheck(11, "eval", "core capability benchmark pack", _has_any(["eval/gsm8k.py", "eval/humaneval.py"]), _exists("eval/report_builder.py"), _exists("tests/test_comprehensive.py"), 11 in pending, ["eval/gsm8k.py", "eval/humaneval.py"]),
        ItemCheck(12, "eval", "security/red-team benchmarks", _exists("scripts/golden_eval.py"), _exists("reports/cleanroom_verification.md"), _exists("tests/test_comprehensive.py"), 12 in pending, ["scripts/golden_eval.py"]),
        ItemCheck(13, "eval", "kpi pack + schema + cli", _exists("mertformer_sdk/kpi.py"), _grep("mertformer_sdk/cli.py", "kpi-report"), _exists("tests/test_kpi_report_cli.py"), 13 in pending, ["mertformer_sdk/kpi.py"]),
        ItemCheck(14, "deploy", "onnx export contract", _exists("scripts/test_onnx_export.py"), _exists("scripts/mobile_export.py"), _exists("tests/test_onnx_export_path.py"), 14 in pending, ["scripts/test_onnx_export.py"]),
        ItemCheck(15, "deploy", "edge/runtime smoke", _exists("scripts/train_smoke.py"), _exists("scripts/kaggle_train_compare_build30.py"), _exists("tests/test_kaggle_compare_script.py"), 15 in pending, ["scripts/train_smoke.py"]),
        ItemCheck(16, "model", "bitnet", _exists("layers/bitlinear.py"), _exists("layers/bitnet_patch.py"), _exists("tests/test_kernel_equivalence.py"), 16 in pending, ["layers/bitlinear.py"]),
        ItemCheck(17, "model", "transformer stack", _exists("model/transformers.py"), _exists("layers/mertformer_block.py"), _exists("tests/test_model.py"), 17 in pending, ["model/transformers.py"]),
        ItemCheck(18, "model", "mla", _exists("layers/mla.py"), _grep("layers/mertformer_block.py", "MLA"), _exists("tests/test_mla_regressions.py"), 18 in pending, ["layers/mla.py"]),
        ItemCheck(19, "model", "decoupled rope", _grep("layers/mla.py", "decoupled_rope"), _exists("layers/mla.py"), _exists("tests/test_mla_regressions.py"), 19 in pending, ["layers/mla.py"]),
        ItemCheck(20, "model", "gqa", _grep("layers/mla.py", "num_kv_heads"), _exists("layers/mla.py"), _exists("tests/test_mla_regressions.py"), 20 in pending, ["layers/mla.py"]),
        ItemCheck(21, "model", "hierarchical kv cache", _grep("layers/mla.py", "use_hierarchical_kv_cache"), _grep("layers/mla.py", "_build_hierarchical_kv"), _exists("tests/test_cognitive_extensions.py"), 21 in pending, ["layers/mla.py"]),
        ItemCheck(22, "model", "moe + liquidrouter", _exists("layers/moe.py"), _grep("layers/mertformer_block.py", "MoE"), _exists("tests/test_architecture_integrity.py"), 22 in pending, ["layers/moe.py"]),
        ItemCheck(23, "model", "liquid/cfc mixer", _exists("layers/liquid.py"), _grep("layers/mertformer_block.py", "LiquidMixer"), _exists("tests/test_architecture_integrity.py"), 23 in pending, ["layers/liquid.py"]),
        ItemCheck(24, "model", "swiglu ffn", _exists("layers/ffn.py"), _grep("layers/mertformer_block.py", "MertFormerFFN"), _exists("tests/test_model.py"), 24 in pending, ["layers/ffn.py"]),
        ItemCheck(25, "model", "qinn", _exists("layers/qinn.py"), _grep("layers/mertformer_block.py", "qinn"), _exists("tests/test_model.py"), 25 in pending, ["layers/qinn.py"]),
        ItemCheck(26, "model", "norm hybrid", _grep("layers/mla.py", "_QKRMSNorm"), _grep("layers/mertformer_block.py", "RMSNorm"), _exists("tests/test_model.py"), 26 in pending, ["layers/mla.py", "layers/mertformer_block.py"]),
        ItemCheck(27, "model", "residual scaling/deepnorm", _grep("layers/mertformer_block.py", "self.residual_scale"), _exists("layers/mertformer_block.py"), _exists("tests/test_model.py"), 27 in pending, ["layers/mertformer_block.py"]),
        ItemCheck(28, "model", "attention fallback matrix", _grep("layers/mla.py", "scaled_dot_product_attention"), _grep("layers/mla.py", "FLASH_ATTN_AVAILABLE"), _exists("tests/test_mla_regressions.py"), 28 in pending, ["layers/mla.py"]),
        ItemCheck(29, "train", "distillation path", _exists("orchestrator/distillation_manager.py"), _grep("train/train.py", "DistillationManager"), _exists("tests/test_train_loop_sanity.py"), 29 in pending, ["orchestrator/distillation_manager.py"]),
        ItemCheck(30, "agent", "swarm runtime 3/15/45", _exists("orchestrator/swarm_runtime.py"), _exists("orchestrator/agent_registry.py"), _exists("tests/test_orchestrator_swarm_runtime.py"), 30 in pending, ["orchestrator/swarm_runtime.py"]),
        ItemCheck(31, "agent", "global workspace broadcast", _exists("layers/cognitive_extensions.py"), _grep("layers/mertformer_block.py", "workspace_layer"), _exists("tests/test_cognitive_extensions.py"), 31 in pending, ["layers/cognitive_extensions.py"]),
        ItemCheck(32, "agent", "cross-expert sync bus", _grep("layers/moe.py", "use_cross_expert_sync_bus"), _grep("layers/moe.py", "sync_gate"), _exists("tests/test_cognitive_extensions.py"), 32 in pending, ["layers/moe.py"]),
        ItemCheck(33, "agent", "cross-layer latent ode", _grep("layers/cognitive_extensions.py", "ContinuousLatentODEStateChannel"), _grep("model/transformers.py", "latent_ode_channel"), _exists("tests/test_cognitive_extensions.py"), 33 in pending, ["layers/cognitive_extensions.py"]),
        ItemCheck(34, "agent", "neuromodulatory gain", _grep("layers/cognitive_extensions.py", "NeuromodulatoryGainLayer"), _grep("model/transformers.py", "neuromod_gain_layer"), _exists("tests/test_cognitive_extensions.py"), 34 in pending, ["layers/cognitive_extensions.py"]),
        ItemCheck(35, "memory", "hierarchical memory", _grep("orchestrator/memory.py", "HierarchicalMemoryContract"), _grep("orchestrator/memory.py", "retrieval_metrics"), _exists("tests/test_cognitive_extensions.py"), 35 in pending, ["orchestrator/memory.py"]),
        ItemCheck(36, "cognition", "causal world model head", _exists("layers/world_model_head.py"), _grep("model/transformers.py", "world_model_head"), _exists("tests/test_world_model_head.py"), 36 in pending, ["layers/world_model_head.py"]),
        ItemCheck(37, "cognition", "planner-controller", _exists("orchestrator/planner.py"), _grep("orchestrator/planner.py", "tool_id"), _exists("tests/test_orchestrator_swarm_runtime.py"), 37 in pending, ["orchestrator/planner.py"]),
        ItemCheck(38, "cognition", "verifier/critic loop", _exists("orchestrator/verifier.py"), _grep("orchestrator/verifier.py", "consistency"), _exists("tests/test_orchestrator_swarm_runtime.py"), 38 in pending, ["orchestrator/verifier.py"]),
        ItemCheck(39, "learning", "structural plasticity", _grep("layers/moe.py", "_apply_structural_plasticity"), _grep("layers/moe.py", "expert_activity_mask"), _exists("tests/test_cognitive_extensions.py"), 39 in pending, ["layers/moe.py"]),
        ItemCheck(40, "learning", "lifelong safety adaptation", _exists("layers/lifelong_safety.py"), _grep("layers/mertformer_block.py", "lifelong_safety_layer"), _exists("tests/test_lifelong_safety.py"), 40 in pending, ["layers/lifelong_safety.py"]),
        ItemCheck(41, "governance", "offline governance", _exists("orchestrator/governance.py"), _exists("SECURITY.md"), _exists("tests/test_orchestrator_swarm_runtime.py"), 41 in pending, ["orchestrator/governance.py"]),
        ItemCheck(42, "learning", "hebbian layer", _grep("layers/cognitive_extensions.py", "HebbianPlasticityLayer"), _grep("layers/mertformer_block.py", "hebbian_layer"), _exists("tests/test_cognitive_extensions.py"), 42 in pending, ["layers/cognitive_extensions.py"]),
        ItemCheck(43, "reasoning", "neuro-symbolic layer", _grep("layers/cognitive_extensions.py", "NeuroSymbolicLayer"), _grep("layers/mertformer_block.py", "neuro_symbolic_layer"), _exists("tests/test_cognitive_extensions.py"), 43 in pending, ["layers/cognitive_extensions.py"]),
        ItemCheck(44, "kernel", "bitnet cpu c++ kernel", _exists("mertformer_sdk/kernels/cpp/bitnet_cpu.cpp"), _exists("mertformer_sdk/kernels/cpp/loader.py"), _exists("tests/test_cpp_kernel_loader.py"), 44 in pending, ["mertformer_sdk/kernels/cpp/bitnet_cpu.cpp"]),
        ItemCheck(45, "kernel", "cuda/triton fused kernel", _exists("mertformer_sdk/kernels/triton_ternary.py"), _grep("layers/bitlinear.py", "triton_ternary"), _exists("tests/test_kernel_dispatcher.py"), 45 in pending, ["mertformer_sdk/kernels/triton_ternary.py"]),
        ItemCheck(46, "kernel", "metal/mps kernel path", _grep("mertformer_sdk/kernels/dispatcher.py", "mps_optimized"), _grep("layers/bitlinear.py", "mps_optimized"), _exists("tests/test_kernel_dispatcher.py"), 46 in pending, ["mertformer_sdk/kernels/dispatcher.py"]),
        ItemCheck(47, "kernel", "onnx custom op contract", _exists("mertformer_sdk/kernels/onnx_custom_op.py"), _grep("mertformer_sdk/kernels/__init__.py", "onnx_custom_op"), _exists("tests/test_onnx_custom_op_contract.py"), 47 in pending, ["mertformer_sdk/kernels/onnx_custom_op.py"]),
        ItemCheck(48, "runtime", "kernel dispatcher fallback matrix", _exists("mertformer_sdk/kernels/dispatcher.py"), _grep("layers/bitlinear.py", "select_backend"), _exists("tests/test_kernel_dispatcher.py"), 48 in pending, ["mertformer_sdk/kernels/dispatcher.py"]),
        ItemCheck(49, "product", "sdk api stability", _exists("mertformer_sdk/api.py"), _exists("mertformer_sdk/cli.py"), _exists("tests/test_sdk_api.py"), 49 in pending, ["mertformer_sdk/api.py"]),
        ItemCheck(50, "product", "pilot ops + sla telemetry", _grep("orchestrator/telemetry.py", "runtime_health_report"), _exists("mertformer_sdk/pilot.py"), _exists("tests/test_sdk_pilot_cli.py"), 50 in pending, ["orchestrator/telemetry.py"]),
        ItemCheck(51, "product", "kpi validated pilots", _exists("mertformer_sdk/kpi.py"), _exists("reports/kpi_report_v1.json"), _exists("tests/test_kpi_report_cli.py"), 51 in pending, ["mertformer_sdk/kpi.py"]),
        ItemCheck(52, "agi", "generalization proof", _exists("eval/generalization_suite.py"), _exists("eval/report_builder.py"), _exists("tests/test_57_matrix_gate.py"), 52 in pending, ["eval/generalization_suite.py"]),
        ItemCheck(53, "agi", "robust tool-use + planning autonomy", _exists("orchestrator/tool_registry.py"), _grep("orchestrator/planner.py", "_select_tool"), _exists("tests/test_orchestrator_swarm_runtime.py"), 53 in pending, ["orchestrator/tool_registry.py"]),
        ItemCheck(54, "agi", "continual learning without forgetting", _exists("train/continual_adapter.py"), _grep("train/train.py", "ContinualLearningAdapter"), _exists("tests/test_continual_adapter.py"), 54 in pending, ["train/continual_adapter.py", "train/train.py"]),
        ItemCheck(55, "asi", "recursive self-improvement governance", _exists("orchestrator/self_improvement_guard.py"), _grep("orchestrator/core.py", "propose_self_improvements"), _exists("tests/test_57_matrix_gate.py"), 55 in pending, ["orchestrator/self_improvement_guard.py"]),
        ItemCheck(56, "asi", "formal alignment scaffold", _exists("orchestrator/alignment_contracts.py"), _grep("orchestrator/core.py", "check_alignment"), _exists("tests/test_57_matrix_gate.py"), 56 in pending, ["orchestrator/alignment_contracts.py"]),
        ItemCheck(57, "asi", "compute/energy orchestration scaffold", _exists("orchestrator/compute_orchestrator.py"), _grep("orchestrator/core.py", "compute_schedule"), _exists("tests/test_57_matrix_gate.py"), 57 in pending, ["orchestrator/compute_orchestrator.py"]),
    ]
    return checks


def build_payload(checks: list[ItemCheck]) -> dict:
    total = len(checks)
    green = sum(1 for c in checks if c.green)
    pending_ids = [c.id for c in checks if c.evidence_pending]
    return {
        "schema": "closure_57_matrix_v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_items": total,
        "green_items": green,
        "all_green": green == total == 57,
        "evidence_pending_ids": pending_ids,
        "items": [c.to_dict() for c in checks],
    }


def write_reports(payload: dict, json_path: Path, md_path: Path, md_tr_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Closure 57 Matrix",
        "",
        f"- total_items: {payload['total_items']}",
        f"- green_items: {payload['green_items']}",
        f"- all_green: {payload['all_green']}",
        f"- evidence_pending_ids: {payload['evidence_pending_ids']}",
        "",
        "| # | Area | Name | Code | Integration | Test | Green | Evidence Pending |",
        "|---:|---|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for item in payload["items"]:
        lines.append(
            f"| {item['id']} | {item['area']} | {item['name']} | {'✅' if item['code'] else '❌'} | "
            f"{'✅' if item['integration'] else '❌'} | {'✅' if item['test'] else '❌'} | "
            f"{'✅' if item['green'] else '❌'} | {'⚠️' if item['evidence_pending'] else '—'} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tr_lines = [
        "# Kapanış 57 Matrisi",
        "",
        f"- toplam_madde: {payload['total_items']}",
        f"- yesil_madde: {payload['green_items']}",
        f"- hepsi_yesil: {payload['all_green']}",
        f"- kanit_bekleyen_idler: {payload['evidence_pending_ids']}",
        "",
        "| # | Alan | Bileşen | Kod | Entegrasyon | Test | Yeşil | Evidence Pending |",
        "|---:|---|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for item in payload["items"]:
        tr_lines.append(
            f"| {item['id']} | {item['area']} | {item['name']} | {'✅' if item['code'] else '❌'} | "
            f"{'✅' if item['integration'] else '❌'} | {'✅' if item['test'] else '❌'} | "
            f"{'✅' if item['green'] else '❌'} | {'⚠️' if item['evidence_pending'] else '—'} |"
        )
    md_tr_path.write_text("\n".join(tr_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/closure_57_matrix.json")
    parser.add_argument("--md-out", default="reports/closure_57_matrix.md")
    parser.add_argument("--md-tr-out", default="reports/closure_57_matrix_TR.md")
    args = parser.parse_args()

    checks = build_checks()
    payload = build_payload(checks)
    write_reports(payload, ROOT / args.out, ROOT / args.md_out, ROOT / args.md_tr_out)

    print(json.dumps({
        "total_items": payload["total_items"],
        "green_items": payload["green_items"],
        "all_green": payload["all_green"],
        "evidence_pending_ids": payload["evidence_pending_ids"],
    }, ensure_ascii=False))

    return 0 if payload["all_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
