# MertFormer Build30 T4 One-Cell Evidence Pointer

This file points to the external evidence archive for the Build30 T4 one-cell run.

## External Archive

- Storage surface: Google Drive
- Drive path: `/content/drive/MyDrive/mertformer_evidence/mertformer_full_evidence_20260419_182653.zip`
- Archive size: `15.713 GB`
- SHA256: `f56db515c0c242ca4ef7468415596cd290bf5e03d70d14de4b86c262b89e4cc1`
- SHA sidecar: `/content/drive/MyDrive/mertformer_evidence/mertformer_full_evidence_20260419_182653.zip.sha256`

## Local Repository Policy

The full zip and checkpoint are not copied into this repository.

Reason:
- the archive is too large for normal source control
- the local Mac does not have enough comfortable free space for repeated large evidence copies
- this repository should contain small, readable, verifiable evidence pointers rather than raw training blobs

## Run Identity

- Runtime output dir: `/kaggle/working/mertformer_onecell_outputs`
- Checkpoint dir: `/kaggle/working/mertformer_onecell_outputs/checkpoints/kaggle_onecell_t4_build30`
- Run dir: `/kaggle/working/mertformer_onecell_outputs/runs/run_20260419_140401`
- Best observed checkpoint: `step_00003500.pt`
- Checkpoint size: approximately `2.142 GB`

## Truth Boundary

This is a provisional training evidence package.

Correct claim:
- a real T4 one-cell training run happened
- a checkpoint was produced
- training and validation loss were logged
- the full evidence archive was backed up to Google Drive
- the archive has a SHA256 integrity hash

Incorrect claim:
- production chatbot
- benchmark-validated model
- final global model
- AGI or ASI evidence
- edge deployment proven

## Next Action

Do not download the full zip unless a second local/offline backup or a Hugging Face upload workflow explicitly requires it.

The practical next step is to create a Hugging Face private evidence repository or a short README/model card that links to this evidence pointer and truth boundary.
