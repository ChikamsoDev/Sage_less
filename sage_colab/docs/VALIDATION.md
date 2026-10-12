# Validation completed on 7 October 2026

**33 tests passed** in the isolated pinned-library CPU environment. There were no failures or skips. One PEFT warning came from saving the deliberately tiny randomly initialized test model without a pretrained model-card path; it is not a failed test. Exact output is in `validation/pytest.txt`.

Executed locally:

- Full acting-loop tests: real files, evidence-based completion, unverified-answer labelling, persistent skills, consent exclusion, malicious action rejection, idempotent interrupted-action recovery, large-file evidence externalization, and grant-version provenance.
- Authority/executor tests: owner authentication, altered signatures, stale/expired grants, exact parameter binding, resource prefixes, cumulative action limits, traversal, symlink/hardlink and file-type conflicts, read/write preconditions, arithmetic and structured-data results, process restart and owner resume.
- Actual OS sandbox tests: Python assertions and artifacts, denial of host-file reads, network sockets, subprocess execution and host permission changes; revocation during in-flight Python execution.
- Persistent memory tests: provenance/scope filtering, claim corrections, deletion, separate training consent, secret-field exclusion, and index rebuilding when encoders change.
- Snapshot tests with local directories standing in for Drive: live SQLite backup, checksums, restore, corrupt-snapshot rejection, symlink rejection, plus integrated training/goal recovery in a new runtime state directory. This does not exercise Google Drive FUSE or its authorisation flow.
- Actual gradient updates and parameter/checkpoint restoration for the recurrent world/policy/controller system; isolated consented-learning candidates, retention gate, owner promotion and rollback; no operation receipt promoted into a goal-completion label.
- Actual PEFT LoRA gradients, changed parameters, saved safetensor adapter and optimiser restoration path on a **35,008-parameter random Qwen3 test model**. No pretrained LLM weights were downloaded. This checks the integration, not 4-bit GPU training or large-model quality.
- Isolated planner configuration evaluation, candidate archive, exact configuration validation, owner release gate and rollback.
- Notebook JSON/nbformat validation, compilation of every code cell and embedded Python file, and verification that the embedded ZIP matches the complete source archive by SHA-256.

The pinned local inference stack imported successfully: Transformers 4.57.1, PEFT 0.17.1, bitsandbytes 0.48.1, Accelerate 1.11.0 and PyTorch 2.8.0+cpu. Dependencies were installed in a separate project validation environment; the user's existing environment was not changed. Local IPC tests needed execution outside the development tool's outer network sandbox because that sandbox blocks even Unix-domain socket binding. Generated-code tests still ran under SAGE's own Landlock/seccomp sandbox.

## Actual small-model training experiment

The 135,799-parameter model trained for 120 optimiser steps on 2,160 labelled transitions from mechanically executed simulator branches. Training loss fell from **3.0990 to 0.5843**. All recurrent, action, world-ensemble, policy and controller parameter groups changed in tests. Saved weights, optimiser, data/source hashes and resume behaviour were checked.

| Diagnostic split | Untrained success Brier ↓ | Trained success Brier ↓ |
|---|---:|---:|
| Development episodes | 0.3491 | 0.0480 |
| Later-index / longer-delay episodes | 0.3290 | 0.0196 |
| Withheld orders composition | 0.3396 | 0.0210 |
| Retention episodes | 0.3439 | 0.0373 |

Exact metrics, losses, parameter hash and dataset hashes are in `validation/world_training.json` and `validation/world_evaluation.json`. The curriculum covers file prerequisites, consumable inventory, workflow prerequisites, and a held-out composition involving orders. These are small shared-mechanism simulators. The numbers establish learning in this curriculum; they do **not** establish broad transfer, better real-world planning, or superiority to a capable LLM agent. Only one training seed was used for the recorded experiment.

## Awaiting hosted Colab execution

This development environment has CPU-only PyTorch and no authenticated hosted Colab session. Therefore the following were **not run here**:

- Actual free-Colab GPU allocation, CUDA kernels and bitsandbytes 4-bit loading of Qwen3-8B/4B.
- Large-model download, end-to-end pretrained generation and full-size QLoRA memory/time behaviour.
- Google Drive mount authorisation, FUSE snapshot performance and recovery after Colab eviction.
- The B0–B4 matched pretrained-agent comparison on three tool tasks; its runner is implemented and enabled by default in the notebook.
- Pretrained MiniLM downloads and runtime retrieval quality.

The notebook automatically runs these available startup checks after you choose a GPU and authorise Drive. It records precise failures, actual model revision, hardware, allocations, dependency versions and diagnostic outcomes. A QLoRA resource failure is recorded as an untrained/incomplete candidate and a clean-backbone recovery is attempted. OS-isolated code execution fails closed if unavailable. There is no manual debugging or test-running step in the launch instructions.

AGI/ASI, competent-human parity, broad continual learning, cross-modal grounding, secret independent audits and general reasoning improvement remain unverified or unimplemented, as detailed in `IMPLEMENTATION_STATUS.md`.
