# Seed Policy

- Set a global seed for Python, NumPy, and PyTorch.
- Log the seed in every run metadata file.
- Note: Full determinism is not guaranteed across GPU kernels.

## Example Log Snippet
```
2026-02-05 07:12:44,021 - [INFO] - seed=1337
2026-02-05 07:12:44,021 - [INFO] - torch.deterministic=False
2026-02-05 07:12:44,022 - [INFO] - cudnn.benchmark=True
```
