# Logger Contract

Required runtime fields for closure-critical logging:

- `timestamp_utc`
- `run_id`
- `global_step`
- `loss` / `ce` / `distill` / `aux`
- `lr` / `grad_norm` / `tok_s`
- `stage` / `validation_loss` / `best_validation_loss`
- `tokens_seen_total` / `gpu_mem_alloc` / `gpu_mem_reserved`
- `moe_max_load` / `moe_avg_std` / `moe_load_entropy` / `moe_capacity_overflow`

This contract is enforced as a repo-side closure boundary. Trained-run measurements remain post-run evidence.
