# Off-site checkpoint backup — runbook

**Why this exists:** the 2026-05-14 2xH200 partial run trained to step 1880 and its checkpoint
was **permanently lost** — no off-site copy existed, and the rented machine's storage was
reclaimed before it could be retrieved. This is the single concrete failure this runbook exists
to prevent.

**Update (2026-07-19): automation now exists.** `scripts/offsite_backup_watcher.py` is wired into
both `scripts/launch_8xb300.sh` and `scripts/launch_ocean_45k.sh` and starts automatically
whenever `TITAN_OFFSITE_BACKUP_DEST` is set at launch time (off by default -- no-op otherwise, see
"Automated path" below). The manual procedure in this runbook is kept as the documented fallback
and as a spot-check cross-reference (step 3 below) -- it is not obsolete, just no longer the only
option.

## Automated path (preferred)

1. Decide the destination (same three options as the manual procedure below) and export it before
   launch:
   ```bash
   export TITAN_OFFSITE_BACKUP_DEST="s3://your-bucket/mertformer_titan_prod/"
   # or: export TITAN_OFFSITE_BACKUP_DEST="you@your-home-machine:/path/to/backup/"
   # or: export TITAN_OFFSITE_BACKUP_DEST="gs://your-bucket/mertformer_titan_prod/"
   ```
2. Launch as usual (`bash scripts/launch_8xb300.sh --go` or `bash scripts/launch_ocean_45k.sh --go`).
   The launch script prints whether the watcher started, and the watcher's own log lands under
   `logs/` (`logs/offsite_backup_watcher.log` on the b300 launcher, `logs/launch/offsite_backup_watcher_<RUN_ID>.log`
   on the ocean launcher).
3. The watcher polls every `TITAN_OFFSITE_BACKUP_INTERVAL_SECONDS` (default 300s), skips a cycle if
   the newest checkpoint file was touched within `TITAN_OFFSITE_BACKUP_STABILITY_SECONDS` (default
   30s -- a best-effort guard against syncing a file still mid-write), then syncs via
   `aws s3 sync`/`gsutil rsync`/`rsync` (chosen automatically from the destination's scheme) with
   retry+backoff on failure.
4. On `scripts/launch_ocean_45k.sh` the watcher is trap-killed cleanly when the run ends (same
   pattern as the existing `nvidia-smi dmon` telemetry process in that script). On
   `scripts/launch_8xb300.sh` the watcher is started before that script's own `exec` into
   `zero_touch_start.sh` and is **not** auto-stopped (`exec` replaces the process, so nothing after
   it can clean up) -- stop it manually once the run finishes and step 4 of the manual procedure
   below (final full sync + verify) has been done.

No launch script wired this automatically before 2026-07-19 (verified then: no
`rsync`/`rclone`/`aws s3 sync`/`gsutil` call existed anywhere in `scripts/launch_*.sh`) — that
gap is what the section below still documents as the fallback, for when the automated path isn't
usable (e.g. a destination scheme the watcher doesn't recognize) or as a manual spot-check.

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

## What this (still) deliberately does not do

- It does not pick a specific cloud provider for you — that's a cost/access decision tied to
  whichever compute lane and budget end up being used for the real 45K run, and shouldn't be
  hardcoded here ahead of that decision. `scripts/offsite_backup_watcher.py` dispatches to
  whichever scheme `TITAN_OFFSITE_BACKUP_DEST` implies (`s3://`, `gs://`, or rsync-style) — the
  choice is still yours, made at launch time via that one env var.
- Credential handling is still the operator's responsibility (e.g. `~/.aws/credentials`,
  `gcloud auth`, or an SSH key already set up for the rsync target) — the watcher shells out to
  the standard CLI tools and inherits whatever credentials are already configured in that
  environment; it does not manage or inject credentials itself.
