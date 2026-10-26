# muse_moe_w4a16: routed NVFP4 MoE experts

## The problem

The routed-expert block of Muse Spark Open for one expert-parallel rank (EP8) on one H200. The model has 256 experts
and routes each token to 8 of them; this GPU holds experts 0-31. The inputs are BF16 hidden states `[M, H]`, the
top-8 expert ids and weights per token, and the expert weights in NVFP4 (W4A16): packed E2M1 codes (two per byte),
an FP8 E4M3 scale per 16 values, and an FP32 global scale per expert. For each token and each slot routed to a local
expert the block computes

    h = x W13^T,  inter = silu(gate(h)) * up(h),  y = inter W2^T

and then a post-expert RMSNorm (weight applied as `1 + w`), the top-k weighting and a sum over the slots. BF16
roundings follow SGLang's path. Token counts run from one token (decode) to verify blocks of 128 tokens and more.

## How the solution works

`kernel_fn` runs three or four launches:

1. `_route` / `_route2` (Triton, one CTA): map global expert ids to local ones, group the (token, slot) pairs by
   expert in a deterministic order, and write a work list of `(expert, first row, end row)` items.
2. `_w13_silu`, `_w13_mix` or `_w2` with `GU=True` (Gluon): the gate/up GEMM with SiLU and the product fused into the
   epilogue.
3. `_w2`, `_w2_mix` or `_w2_ks` (Gluon): the down projection.
4. SGLang's `_post_expert_norm_weighted_sum_kernel`, or `_wsum_ks` at one token.

Key techniques:

- **NVFP4 decode in registers.** `prepare()` permutes the bits inside every 32-bit word of packed codes
  (`_permute_bits`, lossless), so that one rotate and one mask place two E2M1 codes in BF16 bit positions
  (value x 2^-126); a `bf16x2` multiply by the E4M3 block scale (times 2^119, exact) finishes the decode
  (`_DECODE_ASM` / `_DECODE_ASM16`, inline PTX run with `gl.inline_asm_elementwise`). The remaining factors are applied to the FP32 accumulator.
- **Weights as the register A operand.** Weight words are loaded from global memory straight into the registers of
  the `mma_v2` A operand, in a K order chosen so that each thread's K values are 32 contiguous bytes. No shared memory
  is used for weights; activations are the B operand and go through an XOR-swizzled shared buffer.
- **Tile-major weight copies.** `_tile_major` reorders codes and scales so that each (expert, 64-row tile, 256-wide K
  block) is one contiguous run, which streams from DRAM better than rows 2 KB apart. `_interleaved` merges codes and
  scales per block (one stream per CTA, used at 8 tokens), and `_gu_rows` builds a gate/up-paired copy of W13 so one
  CTA reads gate and up rows of the same output columns in one stream (used at 16 and 32 tokens).
- **One token (`_decode1_direct`).** Each top-k slot is a work item read directly from `topk_ids`, so no routing pass
  runs. W13 is split over K into 2 parts reduced by `_d1_reduce`, and W2 is split over K into 4 parts whose FP32
  partials are summed inside `_wsum_ks`, a copy of SGLang's weighted-sum kernel that rounds the sum to BF16 before the
  unchanged norm. Weight and scale loads are double-buffered in registers.
- **Per-item token tiles at verify sizes.** From 128 tokens, `_route2` builds two work lists in one pass: W13 runs on
  items of up to 32 rows (one weight decode serves up to 32 token rows; 8/16/32-row tile chosen per item) and W2 on
  items of up to 16 rows (8/16-row tile per item). The 32-row tile is compiled with a 128-register cap.
- A shared-B path that multiplies with `warpgroup_mma` (wgmma) is still in the file behind `VERIFY_SMB`; it is off,
  because the register path was faster on the H200.

## Official result

| | |
|---|---|
| GPU | H200 |
| Passed | true |
| Speedup | 1.8320 |
| 95% CI | 1.8046 - 1.8595 |
| Score | 7340.7 us (baseline 13448.5 us) |
| Board rank | 217 / 794 (snapshot 2026-10-26 01:41Z) |
| Submitted / finished | 2026-10-25 04:13:11Z / 04:16:29Z |

Source: [`RESULT.json`](RESULT.json). Per-case pass counts are not recorded in it.

## Verify

`submission.py` differs from the judged file only in comments and docstrings.

- SHA-256 of the judged file: `8a48f0f100ccf5fc9878c122037969296841d1315474d9807c218698037aa773`
- Judge content hash: `5e7f7995d62957d2b75b156be699141a14a59c8b8d1ad38fc878761c41231965`
- SHA-256 of the published `submission.py`: `281d56e94d22f7ebb440274d25c240697b4728af2586da252fb1befb1f35cc35`

With a copy of the judged file (it matches the judged file's SHA-256 above):

    python verify/check_equivalence.py judged_submission.py muse_moe_w4a16/submission.py

prints `EQUIVALENT` when the two files have the same syntax tree once docstrings are removed.
