# Off-site checkpoint backup — runbook

**Why this exists:** the 2026-05-14 2xH200 partial run trained to step 1880 and its checkpoint
was **permanently lost** — no off-site copy existed, and the rented machine's storage was
reclaimed before it could be retrieved. This is the single concrete failure this runbook exists
to prevent. No launch script wires this automatically (checked 2026-07-19: no
`rsync`/`rclone`/`aws s3 sync`/`gsutil` call exists anywhere in `scripts/launch_*.sh`) — this is
a manual operator step, run from a second terminal on the rented machine, alongside the training
run itself.

## When to run this

Every time `TITAN_SAVE_INTERVAL` (default `1000` steps, see `scripts/launch_ocean_45k.sh`) writes
a new checkpoint under `cfg.save_dir` (default `./checkpoints/mertformer_titan_prod/`). Don't wait
until the run finishes — a checkpoint that only exists on the rented machine is one preemption
away from the same fate as the 2026-05-14 run.

## Minimum viable procedure (any lane)

1. **Before the run starts**, decide the off-site destination and confirm you can write to it
   from the rented machine (test with a small file first — don't discover a broken credential
   after the first real checkpoint lands):
   - A cloud bucket you control (S3/GCS/R2/Backblaze), **or**
   - Your own machine via `rsync`/`scp` over SSH if the rented box exposes a reachable port, **or**
   - A second, independent rented-storage volume, if the compute provider offers one separately
     from the compute instance itself (so a compute-instance reclaim doesn't also wipe storage).
2. **While the run is live**, in a separate terminal/session on the rented machine, poll for new
   checkpoint files and sync them out as they appear. A simple loop (adjust the destination for
   whichever option was chosen in step 1):
   ```bash
   # Example: sync to an S3-compatible bucket every 5 minutes.
   while true; do
     aws s3 sync ./checkpoints/mertformer_titan_prod/ s3://<your-bucket>/mertformer_titan_prod/ \
       --exclude "*.tmp"
     sleep 300
   done
   ```
   ```bash
   # Example: rsync to your own machine over SSH (run FROM the rented box).
   while true; do
     rsync -avz --exclude "*.tmp" ./checkpoints/mertformer_titan_prod/ \
       you@your-home-machine:/path/to/backup/mertformer_titan_prod/
     sleep 300
   done
   ```
3. **After each sync**, spot-check that the off-site copy's file sizes/`sha256sum` match the
   on-box copy for at least the `best.pt`/`latest.pt` files — a partially-synced checkpoint is
   worse than no checkpoint (silent corruption vs. an honest "nothing here").
4. **After the run ends** (success, preemption, or manual stop), do one final full sync and verify
   it before releasing/terminating the rented instance. Do not terminate the instance until the
   off-site copy is confirmed complete.

## What this runbook deliberately does not do

- It does not pick a specific cloud provider for you — that's a cost/access decision tied to
  whichever compute lane and budget end up being used for the real 45K run, and shouldn't be
  hardcoded here ahead of that decision.
- It does not wire this into `scripts/launch_ocean_45k.sh`/`scripts/launch_8xb300.sh` as an
  automatic background step. Doing that safely means handling credential injection, retry/backoff,
  and partial-write detection inside the launch script itself — a real engineering task, not a
  runbook. This document is the manual bridge until that automation exists (tracked in
  `BACKLOG.md`).
