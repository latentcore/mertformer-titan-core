# Inference Contract

## Inputs
- **Text input** (UTF-8)
- **Tokenizer**: loaded at runtime from `meta-llama/Llama-3.3-70B-Instruct`
- **Max context length**: 4096 tokens (`cfg.max_seq_len`)

## Outputs
- Token IDs and decoded text
- Output length is **caller-defined** (e.g., chat defaults to 128 tokens)

## Notes
- ONNX export is provided for mobile inference.
- Contract may evolve after production runs.
