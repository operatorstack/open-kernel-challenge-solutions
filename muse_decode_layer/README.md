# muse_decode_layer: a full Muse Spark Open decoder layer

## The problem

One complete decoder layer of Muse Spark Open as one H200 runs it under TP8 / EP8, end to end: attention, the
post-attention boundary, the router, the MoE block and the post-FFN boundary with the next layer's input norm. The
reference is SGLang's `MuseSpark1xDecoderLayer.forward`:

| Stage | SGLang module |
|---|---|
| `qkv_gate` projection, BF16 `[2304, 8192]` | `bf16_gemm_dispatch` (cuBLAS) |
| per-head q/k RMSNorm (+ interleaved RoPE on sliding layers) | `FusedQKNorm` |
| FP8 E4M3 cache store, q to E4M3 | `set_kv_buffer` / `store_cache` |
| paged attention, sliding window 2048 or full | FA3 `flash_attn_with_kvcache` |
| output norm x sigmoid(gate) | `FusedRMSNormGate` |
| `o` projection, BF16 `[8192, 1024]` | `bf16_gemm_dispatch` |
| post-attention norm + residual gate + pre-FFN norm | `FusedPostNormResidualPreNorm` |
| router, FP32 `[256, 8192]`; sigmoid, biased top-8, renormalize | `MuseSpark1xRouter`, `FusedMuseSparkRouting` |
| pooler, BF16 `[4096, 8192]`; pre-expert RMSNorm | `bf16_gemm_dispatch`, `MuseSpark1xRMSNorm` |
| 32 local NVFP4 experts (W4A16) + post-expert norm and weighted sum | Marlin `moe_wna16_marlin_gemm`, Triton finalize |
| unpooler, BF16 `[8192, 4096]` (this GPU's 1024-row slice up to 8 tokens) | `bf16_gemm_dispatch` |
| post-FFN norm + residual gate + next layer's pre-attention norm | `FusedPostNormResidualPreNorm` |

A case is `B` requests of `q_len` 1 (decode) or 8 (a speculative verify block), so `T = B x q_len` tokens, on a
sliding-window or a full-attention layer. All-reduces are the identity on one GPU and the all-gather is a zero pad.

## How the solution works

`kernel_fn` keeps SGLang's structure and replaces the kernels that matter for each token count with faster ones, chosen
by fixed shape tables (no data-dependent choice). The main pieces:

**Attention**
- On sm_90, attention for most cases runs as one fused kernel per layer (q/k norm, RoPE, FP8 cache store, attention,
  RMSNorm x sigmoid(gate)), using the same techniques as the [`muse_attention_decode`](../muse_attention_decode)
  solution (the Gluon full-decode kernels have the same function bodies as that file's):
  the Triton split-K kernels behind `_fused` (`_fused_attn`, with an exact last-arriver log-sum-exp merge) on
  sliding layers, the Gluon
  `_gluon_full_decode` (`mma_v2`, `cp.async` pipeline) for full-attention decode at `B >= 16`, and the
  warp-specialized Gluon `_gluon_full_verify` (`warpgroup_mma`, `mbarrier` ring) for full-attention verify blocks at
  `B >= 16`.
- Other cases use one Triton kernel for q/k norm, RoPE, q to E4M3 and the FP8 cache write (`_qk_norm_fp8_store`,
  replacing 5 kernels), then a custom split-K full-attention kernel at batch 1 (`_s_full_attention`) or FA3.

**Dense projections**
- One token: `_gemv_kernel`, a bandwidth-bound Triton GEMV. For the o projection on sliding layers at batch 1 it
  uses programmatic dependent launch (PDL) to start before its producer finishes and prefetch its constant weight
  lines into L2 (`prefetch.global.L2`) before waiting on the producer's output.
- 8-32 tokens: swap-AB Triton GEMMs (`_gemm_swap`, weight rows on the MMA M side), with split-K into an FP32 buffer
  plus a reduce kernel where it measured faster (`SWAP_CFG`).
- The unpooler slice at 2-8 tokens: split-K reduced inside a thread-block cluster through distributed shared memory,
  written straight into the zero-padded output rows (the same technique as the cluster kernel of
  [`muse_bf16_proj`](../muse_bf16_proj), with an output row stride).
- The pooler at 128 and 256 tokens: pinned cuBLASLt algorithms called through `ctypes` (`_lt_prepare`, `_lt_dense`).
  Each plan is rebuilt in `prepare()` from its recorded configuration and used only if the rebuilt algorithm is
  byte-equal to the recorded one, `cublasLtMatmulAlgoCheck` accepts it, and a fixed self-test input reproduces the
  default call bit for bit; otherwise the shape keeps the default call.

**Router and routing**
- The router and `FusedMuseSparkRouting` run on a side stream, overlapping the pooler on the main stream. At one
  token the router is the same FP32 GEMV kernel as in `muse_bf16_proj` (`_cbf_gemv_rows32`, same function body);
  at 8 tokens a dedicated FP32 kernel (`_router_eight`); above 8 tokens a one-launch split-K kernel on BF16 hi + lo halves of the FP32 weight with FP32 accumulation
  (`_router_hilo`). For `T > 1` the expert grouping (`_grouped_route`; from 128 tokens a multi-CTA `_route2_mc`
  that reproduces the single-CTA sort exactly) also runs on the side stream.

**MoE experts**
- One token (`_experts_d1_n`): the pre-expert RMSNorm is fused into the W13 kernel's prologue with the norm kernel's
  own layout and expressions, so the result is bit-exact. W13 and W2 are split over K (16 and 8 splits); the last CTA
  to arrive (release/acquire atomics) reduces the FP32 partials in a fixed order. Weights are read from per-(expert,
  tile) contiguous k-block records (`_segf_layouts`, exact byte permutations made and inverse-checked in `prepare()`,
  with a fallback to the plain layout if the check fails).
- More tokens (`_grouped_experts`): grouped Gluon NVFP4 kernels using the same techniques as
  [`muse_moe_w4a16`](../muse_moe_w4a16), with in-register E2M1 x E4M3 decode, tile-major / interleaved / gate-up-paired weight copies per token count, W2 split
  over K at 8-32 tokens with the FP32 partials summed in the post-expert kernel, and per-item 8/16/32-row token tiles
  from 128 tokens.
- The post-expert norm and weighted sum: a slot-parallel form (rstd per slot, then a PDL elementwise combine) at
  8, 16, 32 and 128 tokens, otherwise SGLang's `_post_expert_norm_weighted_sum_kernel` body with PDL added.

**Layer boundaries**
- At 8-32 tokens on sm_90 both norm/residual boundaries call SGLang's multi-CTA cluster kernel (`fused_norm_residual`,
  RMS moments exchanged through distributed shared memory), which SGLang itself selects only on newer GPUs.

On sm_90 most of this file's Triton kernels launch with PDL, so each kernel's launch and prologue overlap its producer's tail.

## Official result

| | |
|---|---|
| GPU | H200 |
| Passed | true |
| Speedup | 1.7352 |
| 95% CI | 1.7092 - 1.7612 |
| Score | 13036.1 us (baseline 22620.3 us) |
| Board rank | 132 / 702 (snapshot 2026-10-26 01:41Z) |
| Submitted / finished | 2026-10-25 23:37:25Z / 23:40:07Z |

Source: [`RESULT.json`](RESULT.json). Per-case pass counts are not recorded in it.

## Verify

`submission.py` differs from the judged file only in comments and docstrings.

- SHA-256 of the judged file: `36d438e02e8350742fcf681d226b351159815ca5a2efb9e1e9a92e95a58e85fb`
- Judge content hash: `a1aba0a50c964dcf0bf590b67798321f2156f1678fb7f6d9aa4ba622db9d1a75`
- SHA-256 of the published `submission.py`: `f58b46f33fde7e7a744dc1c9480d4f9f90162a794554113abe61f89af6767546`

With a copy of the judged file (it matches the judged file's SHA-256 above):

    python verify/check_equivalence.py judged_submission.py muse_decode_layer/submission.py

prints `EQUIVALENT` when the two files have the same syntax tree once docstrings are removed.
