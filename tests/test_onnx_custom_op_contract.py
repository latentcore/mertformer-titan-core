from __future__ import annotations

import os
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from mertformer_sdk.kernels.onnx_custom_op import detect_onnx_custom_op_plugin, export_contract


def test_onnx_custom_op_contract_defaults_to_fallback():
    os.environ.pop("MERTFORMER_ONNX_PLUGIN", None)
    st = detect_onnx_custom_op_plugin()
    assert st.enabled is False
    contract = export_contract()
    assert contract["fallback"] == "standard_onnx_graph"


def test_onnx_custom_op_contract_detects_plugin_env():
    os.environ["MERTFORMER_ONNX_PLUGIN"] = "mertformer.custom.quant"
    try:
        st = detect_onnx_custom_op_plugin()
        assert st.enabled is True
        assert st.plugin_name == "mertformer.custom.quant"
    finally:
        os.environ.pop("MERTFORMER_ONNX_PLUGIN", None)

