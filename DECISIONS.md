# Architecture Decisions

- **Config discipline**: Keep `config.py` as runtime default; allow optional YAML overlays.
- **Swarm architecture**: Documented as target; not enforced in core training.
- **Multimodal**: Deferred until text core is proven.
- **Canonical KD lane**: Keep the external lane name `offline_clean`, but treat it as the strict precomputed-KD path for the 45K closure run.
- **Teacher surface**: Pin the teacher to `meta-llama/Llama-3.3-70B-Instruct`; do not silently swap teacher IDs on the canonical path.
- **Fallback boundary**: Remove teacherless fallback from the canonical `offline_clean` training path. If logits are missing, fail with the exact blocker instead of silently downgrading the run.
- **Prompt surface**: Keep `prompts/system_v1.txt` as the only canonical system prompt surface until post-training behavioral evidence justifies expansion.
- **Artifact strategy**: Preserve the existing release/repo zips and add a separate training outputs bundle zip for real run retrieval (`artifacts/mertformer_training_outputs_bundle.zip` + SHA256 + manifests).
