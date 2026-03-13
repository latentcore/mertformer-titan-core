# Chain Map — Connected vs Independent

This map summarizes what is **strictly connected** in the training evidence chain vs what is **independent** (productization or deployment evidence).

## Connected Training Chain
```mermaid
flowchart TD
  A["Stage JSONL (datasets/stage*)"] --> B["Training (run.sh → train/train.py)"]
  B --> C["Logs (logs/*.jsonl)"]
  C --> D["SOP artifacts (reports + packages/artifacts zips)"]
```

## Prerequisite Gates (Before Training)
```mermaid
flowchart LR
  G1["accelerate config"] --> B
  G2["cuda.lock"] --> B
  G3["HF_TOKEN + gated teacher"] --> B
  G4["dataset access"] --> B
```

## Independent Evidence Streams
```mermaid
flowchart LR
  X["NPU/Vulkan benchmarks"]
  Y["Dockerfile labels"]
  Z["README timing estimates"]
  X -.-> D
  Y -.-> D
  Z -.-> D
```

Notes:
- The dotted edges above are **not** training blockers; they affect deployment or documentation quality.
- The connected chain is the only path that produces **claim‑eligible** training evidence.
