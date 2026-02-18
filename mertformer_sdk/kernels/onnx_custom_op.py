"""ONNX custom-op contract and safe fallback hooks."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class OnnxCustomOpStatus:
    enabled: bool
    plugin_name: str
    reason: str


def detect_onnx_custom_op_plugin() -> OnnxCustomOpStatus:
    plugin = os.getenv("MERTFORMER_ONNX_PLUGIN", "").strip()
    if not plugin:
        return OnnxCustomOpStatus(enabled=False, plugin_name="", reason="env_not_set")
    return OnnxCustomOpStatus(enabled=True, plugin_name=plugin, reason="env_present")


def export_contract() -> Dict[str, str | bool]:
    st = detect_onnx_custom_op_plugin()
    return {
        "custom_op_enabled": st.enabled,
        "plugin_name": st.plugin_name,
        "reason": st.reason,
        "fallback": "standard_onnx_graph",
    }

