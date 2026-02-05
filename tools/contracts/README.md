# Tool Contracts

Defines input/output contracts for tool calls.

## Contract Template
- **name**: tool name (string)
- **inputs**: JSON object, schema-defined
- **outputs**: JSON object, schema-defined
- **errors**: list of possible error codes

## Example
```
name: "benchmark.run"
inputs:
  model_id: string
  samples: int
outputs:
  humaneval: int
  mbpp: int
errors:
  - DATASET_MISSING
  - MODEL_NOT_FOUND
```
