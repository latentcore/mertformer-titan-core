#include <torch/extension.h>
#include <vector>

// Minimal CPU reference kernel (matmul path).
// This file is intentionally simple and acts as buildable scaffold.
// HONEST NOTE: Despite the "bitnet" name, this is a plain dense
// torch::matmul reference path. It does NOT implement BitNet b1.58 /
// ternary {-1,0,+1} weights, bit-packing, or any low-bit kernel.
// It is a buildable fallback/reference only; real ternary logic is absent.
torch::Tensor bitnet_cpu_linear(torch::Tensor x, torch::Tensor w, torch::Tensor bias) {
  // Plain dense matmul (not ternary). Reference behavior only.
  auto out = torch::matmul(x, w.t());
  if (bias.defined() && bias.numel() > 0) {
    out = out + bias;
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bitnet_cpu_linear", &bitnet_cpu_linear,
        "CPU reference linear (scaffold): plain dense matmul, NOT ternary/b1.58 BitNet");
}

