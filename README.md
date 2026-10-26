# Open Kernel Challenge: final solutions (team bigboateng)

GPU kernels written by team `bigboateng` for the Open Kernel Challenge, in which each submission is judged on an
NVIDIA H200 and scored as its speedup over the board's baseline implementation. This repository
holds the final scored solution on each of the four boards: a complete Muse Spark Open decoder layer, its routed
NVFP4 MoE experts, its FP8 paged attention and its dense BF16 projections. Each folder explains what its problem
computes and how the solution works, and carries the official result.

## Results

| Board | Speedup | 95% CI | Score vs baseline (us) | Board rank (2026-10-26 01:41Z) |
|---|---|---|---|---|
| [`muse_decode_layer`](muse_decode_layer) | 1.7352 | 1.7092 - 1.7612 | 13036.1 vs 22620.3 | 132 / 702 |
| [`muse_moe_w4a16`](muse_moe_w4a16) | 1.8320 | 1.8046 - 1.8595 | 7340.7 vs 13448.5 | 217 / 794 |
| [`muse_attention_decode`](muse_attention_decode) | 1.8228 | 1.7955 - 1.8564 | 2066.2 vs 3766.3 | 196 / 752 |
| [`muse_bf16_proj`](muse_bf16_proj) | 1.3066 | 1.2870 - 1.3262 | 3697.2 vs 4830.8 | 196 / 1019 |

All four passed the judge's correctness checks on an H200. Overall standing: peak **#121 of 1155** (2026-10-25
23:40Z); **#132 of 1159** at the 2026-10-26 01:41Z snapshot.

Per-board numbers come from each folder's `RESULT.json` (the official judge record: speedup, confidence interval,
scores, file hashes and the rank snapshot).

## Techniques

- NVFP4 (E2M1 + E4M3 block scales) weights decoded in registers with inline PTX and fed to tensor-core MMAs as the
  register A operand; tile-major, interleaved and gate/up-paired weight layouts made losslessly in `prepare()`.
- Swap-AB weight streaming for small token counts, with TMA tile loads from host-built tensor descriptors.
- Split-K reductions inside Hopper thread-block clusters through distributed shared memory, and last-arriver
  reductions in a fixed order with release/acquire atomics.
- Programmatic dependent launch (`griddepcontrol`) between dependent kernels, with L2 prefetch of constant weights
  before the dependency wait.
- Fused FP8 E4M3 paged attention: QK-norm, RoPE, cache store, split-K attention, an exact log-sum-exp merge and the
  output norm in one kernel; warp-specialized Gluon kernels with `cp.async`, `mbarrier` rings and `wgmma`.
- One-token GEMV paths, pinned cuBLASLt algorithms with bit-exact self-tests, and launch-feature fallbacks that are
  resolved before CUDA-graph capture.

## Tech stack

Python, PyTorch, Triton, Gluon (Triton's lower-level GPU dialect: explicit layouts, shared memory, `mma_v2`,
`warpgroup_mma`, warp specialization), inline PTX, cuBLAS / cuBLASLt (through PyTorch and `ctypes`), and SGLang's
kernels (FA3, Marlin and Muse Spark fused ops) where the solutions call them. There is no CUDA C++ or CUTLASS source
in this repository; Hopper features (TMA, wgmma, clusters, PDL) are reached through Triton, Gluon and inline PTX.

## Repository layout

    <board>/submission.py   the solution (judged code; comments and docstrings edited, see below)
    <board>/RESULT.json     the official result record
    <board>/README.md       the problem, how the solution works, the result
    SHA256SUMS              SHA-256 of each judged submission.py
    verify/check_equivalence.py
    LICENSE, NOTICE

Each `submission.py` exposes `prepare(weights)` (one-time, untimed weight preparation) where the problem has
weights, and `kernel_fn(...)`, the timed call. They need an SM90 GPU, PyTorch, Triton with Gluon, and SGLang
(`sglang`, `sgl_kernel`), as in the challenge's environment.

## Verifying the code against the judged files

`SHA256SUMS` lists the SHA-256 hashes of the files exactly as the judge ran them; they match `submission_sha256` in
each `RESULT.json`. The comments and docstrings of three files (`muse_decode_layer`, `muse_moe_w4a16`,
`muse_bf16_proj`) were edited for publication, so those three no longer match the hashes byte for byte;
`muse_attention_decode/submission.py` is unchanged and does. No executable code was changed:
`verify/check_equivalence.py` parses two files, removes every docstring and compares the syntax trees (comments are
not part of the tree):

    python verify/check_equivalence.py <judged submission.py> <board>/submission.py

It prints `EQUIVALENT` for each of the four files against its judged original.

## Author

Team `bigboateng`. The code is released under the MIT License (`LICENSE`). Parts of it reproduce or call SGLang
kernels, which remain under the Apache License 2.0; see `NOTICE`.
