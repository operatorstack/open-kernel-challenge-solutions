# muse_attention_decode: FP8 paged attention for decode and verify

## The problem

The attention core of Muse Spark Open on one H200 under TP8: everything between the QKV projection and the output
projection (`MuseSpark1xAttention.gated_attention` in SGLang). Per GPU there are 16 query heads and 2 KV heads of
dimension 64. The input is the fused projection output `qkv` (q, k, v and the output gate's logits), a paged FP8 E4M3
KV cache (page size 1 in the scored cases) with its page table, and the step's positions and cache rows. Each case is:

- a layer kind: sliding-window (window 2048, interleaved RoPE with theta 500000) or full attention (no RoPE);
- `q_len` 1 (decode) or 8 (a speculative verify block, causal inside the block);
- a batch size `B`.

The output is `RMSNorm(attention) x sigmoid(gate)` per head, in BF16. SGLang runs this as FusedQKNorm, the E4M3 casts,
`store_cache`, FA3 (`flash_attn_with_kvcache`) and `rms_norm_gate`.

## How the solution works

`ROUTE` selects a path per (layer kind, `q_len`, batch class), chosen per case from H200 measurements of both paths,
and every entry selects the fused path: one kernel for the whole layer instead of five. The FA3 path (`_prep_kernel`
fusing the norms, RoPE and E4M3 casts, then FA3 and `rms_norm_gate`) remains in the file as the alternative route.

- **Whole-layer fused kernel (`_fused_attn`, Triton).** Each program normalizes (and on sliding layers rotates) its
  queries, rounds them to E4M3 as FA3 does, and attends one split of the cached keys: page-table gather, online
  softmax over 128-key tiles, E4M3 x E4M3 products accumulated in FP32. The program holding split 0 also normalizes,
  rotates and stores the block's new keys and values to the cache as E4M3 and attends them causally from registers.
- **Split-K over the sequence with an in-kernel merge.** Partial `(o, m, l)` results go to scratch; the last program
  of each (request, KV-head block) to finish, counted with an atomic, merges the splits exactly with log-sum-exp
  (in two levels when there are many splits) and applies RMSNorm x sigmoid(gate). The gate is loaded at kernel start
  so its DRAM latency is off the merge's critical path. The split count is derived from the SM count and the
  sequence span, never from data.
- **Block-diagonal query tile.** On decode steps one program serves both KV heads with a block-diagonal query tile,
  so it reads whole 128-byte cache rows (both heads' 64-byte halves) in one pass.
- **Gluon full-attention decode (`_g_full_decode`, sm_90, batch > 4).** Every warp owns a slice of each 128-row key
  tile and keeps its own online-softmax state; `S = K Q^T` runs on `mma_v2` with keys as M and the 8 query rows of a
  KV head as N (E4M3 operands, FP32 accumulate), and `O^T += V^T P` in FP16. Cache tiles arrive through a multi-stage
  `cp.async` pipeline; the warps' states merge once per program.
- **Gluon full-attention verify (`_g_full_verify`, sm_90, batch > 4).** A warp-specialized kernel
  (`gl.warp_specialize`): producer warps gather FP8 key/value tiles into a shared-memory ring with `cp.async`,
  synchronized by `mbarrier` full/empty barriers, and consumer warps run `warpgroup_mma` (wgmma) for `Q K^T` and
  `P V` over 64 query rows per KV head (8 tokens x 8 grouped heads), with the block's own new keys padded to one
  wgmma K step.
- Launch shapes (`FUSED_CFG`) are per layer kind, `q_len` and batch class (`<= 4`, `5-31`, `>= 32`), chosen by sweeps
  on the H200; the choice depends only on shapes and device capability.

## Official result

| | |
|---|---|
| GPU | H200 |
| Passed | true |
| Speedup | 1.8228 |
| 95% CI | 1.7955 - 1.8564 |
| Score | 2066.2 us (baseline 3766.3 us) |
| Board rank | 196 / 752 (snapshot 2026-10-26 01:41Z) |
| Submitted / finished | 2026-10-25 03:54:06Z / 04:01:34Z |

Source: [`RESULT.json`](RESULT.json). Per-case pass counts are not recorded in it.

## Verify

`submission.py` differs from the judged file only in comments.

- SHA-256 of the judged file (reference): `cbded556842cb511642b4d56fd6fb4b27c33336d41812ba0b630a3727fcac8c3`
- SHA-256 of the published `submission.py`: `52e71251fb86305d2d1fd0b9bfa8c81eb3717144bb5553ca0241d40809d5ea06`
- Judge content hash: `7a72333ffe43d9f566deec637df0d82e9e9b9c2eab50b8f2a2dcfa0da0c54029`

With a copy of the judged file (it matches the reference SHA-256 above):

    python verify/check_equivalence.py judged_submission.py muse_attention_decode/submission.py

prints `EQUIVALENT` when the two files have the same syntax tree once docstrings are removed.
