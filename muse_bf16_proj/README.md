# muse_bf16_proj: dense BF16 projections

## The problem

The five dense projections of a Muse Spark Open layer on one H200 under TP8, each called on its own with a token
count `M` and a layer name:

| Layer | Weight | Notes |
|---|---|---|
| `qkv_gate` | BF16 `[2304, 8192]` | q, k, v and the output gate |
| `o` | BF16 `[8192, 1024]` | |
| `pooler` | BF16 `[4096, 8192]` | |
| `unpooler` | BF16 `[8192, 4096]` | up to 8 tokens only this GPU's 1024-row slice |
| `router` | FP32 `[256, 8192]` | FP32 output |

The reference is SGLang's path: `bf16_gemm_dispatch` (cuBLAS) for the BF16 layers, and for the router `F.linear` in
FP32 up to 8 tokens, above that two BF16 GEMMs on a hi/lo split of the FP32 weight with FP32 accumulation. Token
counts run from 1 (decode) to 256. At these token counts the cost is dominated by reading the weights, so the
kernels are built to stream weights at full bandwidth.

## How the solution works

`_cbf_dispatch` picks a kernel by `(layer, M)`; each tiling was chosen by measurement on the H200, and any case it
does not serve falls through to SGLang's `_bf16_gemm_dispatch_impl`.

- **One token: CUDA-core GEMVs.** `_cbf_gemv32` computes `out[n] = sum_k x[k] W[n, k]` with FP32 accumulation, one
  or two weight rows per program and one long load burst per row (`CBF_GEMV1`), for `qkv_gate`, `o`, the pooler and
  the unpooler slice. The router uses `_cbf_gemv_rows32` on the FP32 weight with FP32 output.
- **2-32 tokens: swap-AB weight streaming.** The weight tile (32 or 64 rows) is the MMA's M side and the few tokens
  are padded to 16-32 on its N side, instead of padding the tokens up to a large M tile. Tiles are fetched by TMA
  from host-built tensor descriptors (`_cbf_skinny_gemm_tma`); rows of `x` beyond `M` are zero-filled by the copy
  engine.
- **Split-K reduced through distributed shared memory.** For `qkv_gate` at 2-32 tokens, and for the unpooler slice at
  2-8 tokens, the K ranges of one output tile run as one thread-block cluster (3 or 4 CTAs, `_cbf_cluster*`). Ranks
  other than 0 copy their FP32 partial tile with 16-byte `st.shared::cluster` stores into rank 0's shared memory, a
  `barrier.cluster` release/acquire publishes it, and rank 0 adds the partials and stores BF16. There are no global
  partials, counters or reduce launch.
- **Tile-contiguous weight copies.** For the pooler (2-32 tokens) and the unpooler (9-32 tokens), `prepare()` stores
  the weight as `[N/64, K/256, 64, 256]` (`_cbf_tile_weights`, a lossless reorder), so every TMA box is one contiguous
  32 KB block. `qkv_gate` at 128 tokens gets the same kind of copy with 64 x 128 blocks.
- **Router at 8-256 tokens.** BF16 hi and lo products of the FP32 weight are accumulated separately in FP32 over K
  ranges (`_cbf_router_partials_pdl`), then a reduce kernel sums them. The two kernels are chained with programmatic
  dependent launch: the partials kernel issues `griddepcontrol.launch_dependents` and the reduce kernel starts with
  `griddepcontrol.wait`, so its launch overlaps the partials grid.
- **128 tokens.** `qkv_gate` runs a split-K GEMM (`_cbf_lm_partials_tiled`: 128 tokens x 64 weight rows per tile,
  K blocks of 128, 3 K ranges; `tl.dot` compiles to wgmma on sm_90) on the tile-contiguous copy. The pooler calls cuBLAS on swapped operands,
  `torch.mm(W, x^T)`, and returns the transpose, a column-major view with the required shape and dtype.
- **CUDA-graph-safe fallbacks.** PDL and cluster launches are tried at first use, which happens before graph capture;
  if one is unavailable the case permanently falls back to a plain launch of the same arithmetic.

Split-K scratch and arrival counters are allocated once in `prepare()`, and the counters reset themselves, so every
call can be replayed inside a CUDA graph.

## Official result

| | |
|---|---|
| GPU | H200 |
| Passed | true |
| Speedup | 1.3066 |
| 95% CI | 1.2870 - 1.3262 |
| Score | 3697.2 us (baseline 4830.8 us) |
| Board rank | 196 / 1019 (snapshot 2026-10-26 01:41Z) |
| Submitted / finished | 2026-10-25 02:23:05Z / 02:33:22Z |

Source: [`RESULT.json`](RESULT.json). Per-case pass counts are not recorded in it.

## Verify

`submission.py` differs from the judged file only in comments and docstrings.

- SHA-256 of the judged file: `d5d2023c75ba41956742366f5f9d8b02d5694ecdc64e90593efb7c1d2bfbcac7`
- Judge content hash: `798bebff2b12fb27bd4e91e4d7eafd62c18a19d2a936b4f8ab15bd63612dfac9`
- SHA-256 of the published `submission.py`: `0808a0f1fe5404808382f5160f482e636d1c57eb68b9e66db3cc7d35c0d99229`

With a copy of the judged file (it matches the SHA-256 above):

    python verify/check_equivalence.py judged_submission.py muse_bf16_proj/submission.py

prints `EQUIVALENT` when the two files have the same syntax tree once docstrings are removed.
