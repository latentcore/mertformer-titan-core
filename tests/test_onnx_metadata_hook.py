from __future__ import annotations

import pytest

pytest.importorskip("onnx", reason="onnx not installed")
def test_add_bitpack_metadata(tmp_path):
    import onnx
    from onnx import helper, TensorProto
    from mertformer_sdk.utils.onnx_meta import add_bitpack_metadata

    # Minimal ONNX model
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y])
    model = helper.make_model(graph)

    out_path = tmp_path / "dummy.onnx"
    onnx.save(model, out_path)

    add_bitpack_metadata(out_path)

    loaded = onnx.load(out_path)
    meta = {p.key: p.value for p in loaded.metadata_props}
    assert meta.get("mertformer.bitpack") == "ternary5in8"
    assert meta.get("mertformer.bitpack_meta") == "titan_s25_bitpack.json"
