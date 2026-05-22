# MertFormer Titan - 1-Page Evidence Packet

Date: 2026-05-22
Owner: Mert Yunlu
Public summary: https://gist.github.com/latentcore/dac0aa0c56b12177e4a0e8e8f684bccf

## 1. Who I Am

I am Mert, a self-taught ML systems builder from Turkiye, turning 18 soon.

I am interested in the right early-career path for serious ML systems work, especially Anthropic Fellows, Infrastructure, ML Systems, or Performance Engineering if appropriate.

## 2. Project

MertFormer Titan is an evidence-first low-bit pre-training architecture project.

The current focus is not to claim model quality before evidence. The focus is to produce checkpoint-bound artifacts through a reliable training run, then evaluate honestly.

## 3. Current Evidence

- Repo-side training readiness exists through the `remote_bootstrap` path.
- Current readiness decision: `TRAIN_ALLOWED`.
- Current decision reason: `READY_REMOTE_BOOTSTRAP`.
- Partial 2x H200 distributed training logs exist from a previous provider run.
- Captured partial window: approximately 89 minutes.
- Captured steps: 1 through 1880.
- Captured supervised tokens: approximately 45.56M.
- Cleaned partial evidence window captured no OOM, NCCL failure, or traceback marker.
- The project keeps a measured / target / vision boundary for claims.

## 4. What I Am Not Claiming

- Not trained yet.
- Not benchmark-verified yet.
- Not production-ready.
- Not mobile-ready.
- Not security-certified.
- Not claiming superiority over Claude, Llama, Gemma, Phi, or any other model.
- Not claiming a completed 45K run.

## 5. Current Blocker

The next step is reliable compute and persistent artifact retrieval to produce final checkpoint/eval artifacts.

The main missing evidence is:

- final trained checkpoint
- checkpoint manifest / hash
- checkpoint-bound evaluation outputs
- final archive / artifact bundle
- any real device or deployment measurement

## 6. Why This May Be Relevant

The strongest signal is ML systems discipline rather than a premature capability claim:

- readiness gates
- explicit blocker reason codes
- distributed training evidence
- artifact and manifest planning
- honest evidence boundaries
- post-run evaluation plan

## 7. Concrete Ask

I would be grateful for guidance on the right path:

- Anthropic Fellows
- Infrastructure
- ML Systems / Performance Engineering
- another early-career route
- or the appropriate official application channel

I am not trying to bypass the official process. I am trying to understand the right evidence standard and path before asking anyone to evaluate the project technically.
