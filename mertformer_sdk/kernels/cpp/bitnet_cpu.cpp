#include <torch/extension.h>
#include <vector>

// Minimal CPU reference kernel (matmul path).
// This file is intentionally simple and acts as buildable scaffold.
torch::Tensor bitnet_cpu_linear(torch::Tensor x, torch::Tensor w, torch::Tensor bias) {
  auto out = torch::matmul(x, w.t());
  if (bias.defined() && bias.numel() > 0) {
    out = out + bias;
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bitnet_cpu_linear", &bitnet_cpu_linear, "BitNet CPU linear kernel (scaffold)");
}

