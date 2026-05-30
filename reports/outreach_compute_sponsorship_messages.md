# Compute Sponsorship and Referral Outreach Notes

## Boundary
Use evidence-first outreach only. Do not send bulk spam, raw logs, credentials,
private checkpoints, or provider screenshots. Do not claim a completed 45K run,
benchmark verification, production readiness, mobile readiness, or trained model
capability without direct artifacts.

## Short Compute Sponsor DM
Hi, I am looking for a short, bounded GPU window for MertFormer Titan Build30.
The repo-side start gate is `TRAIN_ALLOWED / READY_REMOTE_BOOTSTRAP`, and a
recent 2x H200 partial run captured stable DDP startup and 1,880 training steps,
but it did not recover final eval/checkpoint/archive artifacts. I am seeking
8-10 hours of 2x H100/H200 or equivalent with persistent output storage and
reliable artifact retrieval so the next run can produce a clean evidence pack.

Reference: https://gist.github.com/latentcore/dac0aa0c56b12177e4a0e8e8f684bccf

## Detailed Compute Sponsor Ask
Current positioning:

> Working repo-side 45K start readiness plus partial 2x H200 pre-45K synthetic
> arithmetic evidence; seeking reliable GPU compute to produce final
> checkpoint-bound evaluation artifacts.

Preferred proof window:

- `8-10` hours of `2x H100/H200` or equivalent.
- Persistent disk for logs, checkpoints, reports, eval outputs, and archives.
- Reliable artifact retrieval after the job exits.

Minimum useful window:

- `2-4` hours of `2x H100/H200` can validate launch and early-curriculum
  behavior, but may not reach final proof completion.

Required runtime controls:

- SSH or equivalent terminal access.
- `tmux`/`screen` or managed job logs.
- Explicit output directory.
- Checkpoint/result retention after process exit.
- Final artifact sync or download command that does not depend only on a web UI.

Proof success condition:

- Final held-out eval summary.
- Exact accuracy and parser/format validity.
- Representative generated examples.
- Checkpoint path plus SHA256 manifest.
- JSON/Markdown reports, JSONL logs, eval history CSV, fixed held-out cases,
  and zip or tar.gz archive.

Update rule after a successful proof:

- Do not add sponsor credit, capability language, or public success wording until
  the final eval, checkpoint, manifest, and archive artifacts are recovered.
- After recovery, update the public Gist, `reports/ocean_pre45k_h200_20260514_partial_evidence.md`,
  outreach packet, and relevant README/evidence references with checkpoint-bound
  measured values only.

## Short Referral DM
Hi, I am preparing a claim-safe MertFormer Titan Build30 packet for systems /
scaling roles. The strongest signal is not a capability claim; it is the
evidence discipline around repo-side readiness, closure automation, low-bit
runtime surfaces, and partial 2x H200 operational logs. If you are comfortable,
I would appreciate a referral or a quick pointer to the right systems/scaling
reviewer.

Reference: `applications/anthropic/measured_evidence_summary.md`

## Follow-Up Boundary
- Ask for a specific review, compute window, or referral path.
- Keep the message short and artifact-linked.
- If the recipient asks for details, send the Gist and the repo evidence summary first.
- Do not send the private raw terminal log unless a written review boundary is agreed.
