# okc: problem=muse_decode_layer
"""SGLang's decoder layer for Muse Spark Open, as one H200 runs it under TP8/EP8, at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f (MuseSpark1xDecoderLayer.forward), with the same modules:

  qkv_gate projection              bf16_gemm_dispatch (cuBLAS on H200)
  FusedQKNorm (Triton)             per-head RMSNorm of q and k, read from the fused projection,
                                   with the interleaved RoPE fused in on sliding layers
  cache store, q to FP8            as FlashAttentionBackend and set_kv_buffer do for an FP8 cache
                                   (cast, then the store_cache JIT kernel at the step's rows)
  flash_attn_with_kvcache (FA3)    paged, causal, window (2047, 0) on sliding layers
  FusedRMSNormGate (JIT)           per-head output norm x sigmoid(gate)
  o projection                     bf16_gemm_dispatch; then the all-reduce (identity here)
  FusedPostNormResidualPreNorm     post-attention norm + residual gate + pre-FFN norm (JIT);
                                   it also writes the FP32 router input up to 8 tokens
  router                           FP32 F.linear up to 8 tokens, above that two BF16 GEMMs
                                   on a hi/lo split of the weight (MuseSpark1xRouter); on a
                                   side stream, overlapping the pooler
  FusedMuseSparkRouting (JIT)      sigmoid, biased top-8, renormalize (on the side stream)
  pooler                           bf16_gemm_dispatch, replicated on every GPU
  pre-expert norm                  MuseSpark1xRMSNorm (F.rms_norm)
  experts                          moe_align_block_size, two moe_wna16_marlin_gemm (W4A16),
                                   silu_and_mul, Triton post-expert norm weighted sum
                                   (MuseSpark1xFp4Experts._forward_marlin); then the
                                   all-reduce (identity here)
  unpooler                         up to 8 tokens, this GPU's 1/8 slice of the weight, then
                                   the all-gather (identity here: zeros in the other columns);
                                   above that, the whole weight (bf16_gemm_dispatch)
  FusedPostNormResidualPreNorm     post-FFN norm + residual gate + the next layer's pre-attention
                                   norm (JIT)

Differs from SGLang:
  * The step's positions, cache rows, FA3 lengths and FA3 scheduler metadata come in with the
    inputs; SGLang builds them once per step for all layers.
  * All-reduces are the identity; the all-gather is a zero pad of the unpooler's output."""

from typing import TypedDict

import types

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.ampere import async_copy as gcp
from triton.experimental.gluon.language.nvidia.ampere import mbarrier as gmbar
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, warpgroup_mma

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    _post_expert_norm_weighted_sum_kernel,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
# Programmatic dependent launch for this file's Triton kernels on SM90, as SGLang's kernels use.
_PDL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
_PDL_KW = {"USE_PDL": True, "launch_pdl": True} if _PDL else {}
GROUPED_MIN_TOKENS = 128  # grouped NVFP4 kernels from this many tokens (verify-16/32 shapes) on SM90
GROUPED_BT = 32
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


# ---- fused glue kernels (everything else stays SGLang's) ------------------------------------

@triton.jit
def _qk_norm_rope_fp8_store_kernel(QKV, POS, COS_SIN, Q8, KC, VC, LOC, KS, VS, eps, stride_qkv, n_rows,
                                   HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr,
                                   HAS_ROPE: tl.constexpr, SCALED: tl.constexpr, USE_PDL: tl.constexpr = False):
    """One (token, head) row per lane: per-head RMS norm of q and k (+ interleaved RoPE on sliding
    layers) exactly as FusedQKNorm, then q -> FP8 (FA3's input), k and v -> FP8 cache rows at
    cache_loc (as k.to(fp8) / v.to(fp8) + store_cache). Replaces 5 kernels with 1."""
    if USE_PDL:  # as SGLang's SM90 kernels: wait for the producer, then let the next kernel launch
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    HEADS: tl.constexpr = HQ + 2 * HKV
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows
    cols = tl.arange(0, D)
    mask = row_mask[:, None] & (cols < D)[None, :]
    token = rows // HEADS
    head = rows % HEADS
    is_q = head < HQ
    is_k = (head >= HQ) & (head < HQ + HKV)
    is_v = head >= HQ + HKV
    src = token.to(tl.int64) * stride_qkv + head * D
    raw = tl.load(QKV + src[:, None] + cols[None, :], mask=mask, other=0.0)
    x = raw.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    x = x * rstd[:, None]
    if HAS_ROPE:
        x = x.to(tl.bfloat16).to(tl.float32)
        xp = tl.load(QKV + src[:, None] + (cols ^ 1)[None, :], mask=mask, other=0.0).to(tl.float32)
        xp = (xp * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        pos = tl.load(POS + token, mask=row_mask, other=0).to(tl.int64)
        freq = cols // 2
        cos = tl.load(COS_SIN + pos[:, None] * D + freq[None, :], mask=mask, other=1.0)
        sin = tl.load(COS_SIN + pos[:, None] * D + D // 2 + freq[None, :], mask=mask, other=0.0)
        is_even = (cols % 2 == 0)[None, :]
        x = tl.where(is_even, x * cos - xp * sin, xp * sin + x * cos)
    x = x.to(tl.bfloat16).to(tl.float32)  # q and k leave the norm as BF16
    x = tl.where(is_v[:, None], raw.to(tl.float32), x)
    if SCALED:  # k / k_scale and v / v_scale on BF16 tensors round to BF16 before the FP8 cast
        ks = tl.load(KS)
        vs = tl.load(VS)
        x = tl.where(is_k[:, None], (x / ks).to(tl.bfloat16).to(tl.float32), x)
        x = tl.where(is_v[:, None], (x / vs).to(tl.bfloat16).to(tl.float32), x)
    x8 = x.to(tl.float8e4nv)
    q_dst = tl.where(is_q, (token * HQ + head) * D, 0)
    tl.store(Q8 + q_dst[:, None] + cols[None, :], x8, mask=mask & is_q[:, None])
    loc = tl.load(LOC + token, mask=row_mask, other=0).to(tl.int64)
    kv_head = tl.where(is_v, head - HQ - HKV, head - HQ)
    dst = loc * (HKV * D) + kv_head * D
    tl.store(KC + dst[:, None] + cols[None, :], x8, mask=mask & is_k[:, None])
    tl.store(VC + dst[:, None] + cols[None, :], x8, mask=mask & is_v[:, None])


def _qk_norm_fp8_store(qkv, data, cfg, cos_sin):
    tokens = cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    q8 = torch.empty((tokens, hq, d), dtype=FP8, device=qkv.device)
    n_rows = tokens * (hq + 2 * hkv)
    # Two values per thread: with four or more FP8 values per thread this Triton build packs the
    # contiguous FP8 store wrongly on SM89 (bytes 0-1 of each 4-byte group are garbage).
    block_m = 4 if n_rows < 1024 else 8
    rope = bool(cfg["rope_theta"])
    _qk_norm_rope_fp8_store_kernel[(triton.cdiv(n_rows, block_m),)](
        qkv, data["positions"], cos_sin, q8, data["k_cache"], data["v_cache"], data["cache_loc"],
        data["k_scale"], data["v_scale"], RMS_EPS, qkv.stride(0), n_rows,
        HQ=hq, HKV=hkv, D=d, BLOCK_M=block_m, HAS_ROPE=rope, SCALED=cfg["kv_scale"] != 1.0, num_warps=block_m, **_PDL_KW)
    return q8


# ---- SGLang's _post_expert_norm_weighted_sum_kernel (fused_post_expert_norm_weighted_sum_triton.py),
# body verbatim, plus PDL (as in the one-token GEMV) so the PDL-launched unpooler that follows can launch early ----
@triton.jit
def _post_expert_ws_pdl_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    ids_ptr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    eps,
    APPLY_NORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
    HAS_IDS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_PDL: tl.constexpr = False,
    KS: tl.constexpr = 1,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    tok = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    if HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for k in tl.static_range(TOPK):
        base = (tok * TOPK + k) * H
        live = True
        if HAS_IDS:
            live = tl.load(ids_ptr + tok * TOPK + k) >= 0
        if live:
            if KS > 1:  # W2's KS FP32 partials, summed and rounded as W2's BF16 output
                x = tl.zeros([BLOCK_H], dtype=tl.float32)
                for s in tl.static_range(KS):
                    x += tl.load(x_ptr + base * KS + s * H + offs, mask=mask, other=0.0)
                x = x.to(tl.bfloat16).to(tl.float32)
            else:
                x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

            if APPLY_NORM:
                if HAS_WEIGHT and WEIGHT_FIRST:
                    y = x * w
                    var = tl.sum(y * y, axis=0) / H
                    y = y * tl.rsqrt(var + eps)
                elif HAS_WEIGHT:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps) * w
                else:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps)
            else:
                y = x

            s = tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
            acc += y * s

    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


# ---- post-expert finalize, slot-parallel form: slot-parallel rstd + PDL elementwise combine
FG1R1_BH = 512
FG1R1_TS = {8, 16, 32, 128}  # token counts served by this form (None = every T); others use _post_expert_ws_pdl_kernel


@triton.jit
def _pe_rstd_kernel(x_ptr, w_ptr, r_ptr, ids_ptr, TOPK: tl.constexpr, H: tl.constexpr, eps, BLOCK_H: tl.constexpr,
                    USE_PDL: tl.constexpr = False, KS: tl.constexpr = 1):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    tok = tl.program_id(0)
    k = tl.program_id(1)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    if tl.load(ids_ptr + tok * TOPK + k) >= 0:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        base = (tok * TOPK + k) * H
        if KS > 1:  # W2's KS FP32 partials, summed in split order and rounded as W2's BF16 output (as in _post_expert_ws_pdl_kernel)
            x = tl.zeros([BLOCK_H], dtype=tl.float32)
            for s in tl.static_range(KS):
                x += tl.load(x_ptr + base * KS + s * H + offs, mask=mask, other=0.0)
            x = x.to(tl.bfloat16).to(tl.float32)
        else:
            x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * w
        var = tl.sum(y * y, axis=0) / H
        tl.store(r_ptr + tok * TOPK + k, tl.rsqrt(var + eps))


@triton.jit
def _pe_combine_kernel(x_ptr, w_ptr, s_ptr, r_ptr, out_ptr, ids_ptr, TOPK: tl.constexpr, H: tl.constexpr,
                       BH: tl.constexpr, USE_PDL: tl.constexpr = False, KS: tl.constexpr = 1):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    tok = tl.program_id(0)
    offs = tl.program_id(1) * BH + tl.arange(0, BH)
    w = tl.load(w_ptr + offs).to(tl.float32)
    acc = tl.zeros([BH], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        base = (tok * TOPK + k) * H
        if tl.load(ids_ptr + tok * TOPK + k) >= 0:
            if KS > 1:
                x = tl.zeros([BH], dtype=tl.float32)
                for s in tl.static_range(KS):
                    x += tl.load(x_ptr + base * KS + s * H + offs)
                x = x.to(tl.bfloat16).to(tl.float32)
            else:
                x = tl.load(x_ptr + base + offs).to(tl.float32)
            y = x * w
            y = y * tl.load(r_ptr + tok * TOPK + k)
            s = tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
            acc += y * s
    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty))


def _fg1r1(y, topk_weights, norm_weight, local_ids, out, tokens, topk, hidden, ks):
    assert hidden % FG1R1_BH == 0
    r = torch.empty((tokens * topk,), dtype=torch.float32, device=out.device)
    _pe_rstd_kernel[(tokens, topk)](y, norm_weight, r, local_ids, TOPK=topk, H=hidden, eps=POST_NORM_EPS,
                                    BLOCK_H=triton.next_power_of_2(hidden), KS=ks, num_warps=8, num_stages=2, **_PDL_KW)
    _pe_combine_kernel[(tokens, hidden // FG1R1_BH)](y, norm_weight, topk_weights, r, out, local_ids, TOPK=topk,
                                                     H=hidden, BH=FG1R1_BH, KS=ks, num_warps=4, **_PDL_KW)
    return out


def _post_expert_weighted_sum(y, topk_weights, norm_weight, local_ids):
    """SGLang's fused post-expert norm + weighted sum kernel at every token count (SGLang falls
    back to ~10 torch ops below 32 tokens)."""
    tokens, topk, hidden = y.shape
    out = torch.empty((tokens, hidden), dtype=y.dtype, device=y.device)
    if FG1R1_TS is None or tokens in FG1R1_TS:  # slot-parallel form (fixed-shape branch on T)
        return _fg1r1(y, topk_weights, norm_weight, local_ids, out, tokens, topk, hidden, 1)
    _post_expert_ws_pdl_kernel[(tokens,)](  # SGLang's kernel + PDL
        y, norm_weight, topk_weights, out, local_ids, TOPK=topk, H=hidden, eps=POST_NORM_EPS,
        APPLY_NORM=True, HAS_WEIGHT=True, WEIGHT_FIRST=True, HAS_IDS=True,
        BLOCK_H=triton.next_power_of_2(hidden), num_warps=8, num_stages=2, **_PDL_KW)
    return out


def _post_expert_weighted_sum_ks(y, topk_weights, norm_weight, local_ids, ks):
    """y = W2's FP32 partials [tokens * TOPK, ks, H]; BF16 output as SGLang's."""
    tokens, topk = local_ids.shape
    hidden = y.shape[-1]
    out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=y.device)
    if FG1R1_TS is None or tokens in FG1R1_TS:  # slot-parallel form (fixed-shape branch on T)
        return _fg1r1(y, topk_weights, norm_weight, local_ids, out, tokens, topk, hidden, ks)
    _post_expert_ws_pdl_kernel[(tokens,)](
        y, norm_weight, topk_weights, out, local_ids, TOPK=topk, H=hidden, eps=POST_NORM_EPS,
        APPLY_NORM=True, HAS_WEIGHT=True, WEIGHT_FIRST=True, HAS_IDS=True,
        BLOCK_H=triton.next_power_of_2(hidden), num_warps=8, num_stages=2, KS=ks, **_PDL_KW)
    return out


@triton.jit
def _rms_norm_local_ids_kernel(X, W, OUT, eps, IDS, LS, LOCAL, SORTED, EXPERT_IDS, NPOST, n_tok, block_size,
                               H: tl.constexpr, TOPK: tl.constexpr, N_EXP: tl.constexpr, N_LOCAL: tl.constexpr,
                               DO_ALIGN: tl.constexpr, NP: tl.constexpr, NB: tl.constexpr,
                               SORT_ALIGN: tl.constexpr = False, NBUCKET: tl.constexpr = 64,
                               USE_PDL: tl.constexpr = False, LSOUT=0, DO_LS: tl.constexpr = False):
    """Programs 0..n_tok-1: pre-expert F.rms_norm(x, (H,), w, eps) for one token, plus its local expert
    ids ((ids - local_start) % 256, then SGLang's expert_map: 0-31 or -1). With DO_ALIGN, program
    n_tok also runs SGLang's single-CTA moe_align_small_numel algorithm (same "+1 offset" buckets,
    num_experts = 257) on those local ids, replacing its separate launch."""
    if USE_PDL:  # as SGLang's SM90 kernels: wait for the producer, then let the next kernel launch
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    if DO_ALIGN:
        if row == n_tok:
            numel = n_tok * TOPK
            offs_p = tl.arange(0, NP)
            mask_p = offs_p < numel
            gid = tl.load(IDS + offs_p, mask=mask_p, other=0)
            rel = (gid - tl.load(LS) + N_EXP) % N_EXP
            lid = tl.where(rel < N_LOCAL, rel, -1)
            if SORT_ALIGN:
                # Same "+1 offset" buckets (0 = filtered, 1..32 = local experts) via a key sort: stable
                # rank = sorted position minus the bucket's first sorted position.
                bkt = tl.where(mask_p, (lid + 1).to(tl.int32), NBUCKET - 1)
                key = tl.sort(bkt * NP + offs_p)
                sb = key // NP
                sidx = key % NP
                cnt_b = tl.histogram(bkt, NBUCKET)  # [NBUCKET]
                offs_k = tl.arange(0, NBUCKET)
                cnt_b = tl.where(offs_k < N_LOCAL + 1, cnt_b, 0)
                pad_b = ((cnt_b + block_size - 1) // block_size) * block_size
                excl_b = tl.cumsum(pad_b, axis=0) - pad_b
                first_b = tl.cumsum(cnt_b, axis=0) - cnt_b
                total = tl.sum(pad_b, axis=0)
                tl.store(NPOST, total.to(tl.int32))
                offs_b = tl.arange(0, NB)
                block_start = offs_b * block_size
                in_b = ((block_start[:, None] >= excl_b[None, :]) & (block_start[:, None] < (excl_b + pad_b)[None, :])
                        & (pad_b[None, :] > 0))
                eid = tl.sum(in_b.to(tl.int32) * (offs_k[None, :] - 1), axis=1)
                tl.store(EXPERT_IDS + offs_b, eid.to(tl.int32), mask=block_start < total)
                n_fill = (total + NP - 1) // NP
                for it in range(n_fill):
                    f_offs = it * NP + offs_p
                    tl.store(SORTED + f_offs, tl.full([NP], 0, tl.int32) + numel, mask=f_offs < total)
                tl.debug_barrier()
                # destination of sorted position j: excl[bucket] + (j - first[bucket])
                sel = (sb[:, None] == offs_k[None, :])
                dst = tl.sum(sel.to(tl.int32) * (excl_b - first_b)[None, :], axis=1) + offs_p
                tl.store(SORTED + dst, sidx.to(tl.int32), mask=sb < N_LOCAL + 1)
                return
            bucket = tl.where(mask_p, (lid + 1).to(tl.int32), N_EXP + 1)
            same = (bucket[None, :] == bucket[:, None]) & mask_p[None, :] & mask_p[:, None]
            earlier = offs_p[None, :] < offs_p[:, None]
            rank = tl.sum((same & earlier).to(tl.int32), axis=1)
            cnt = tl.sum(same.to(tl.int32), axis=1)
            padded_cnt = ((cnt + block_size - 1) // block_size) * block_size
            is_rep = (rank == 0) & mask_p
            smaller_rep = (bucket[None, :] < bucket[:, None]) & is_rep[None, :]
            excl = tl.sum(smaller_rep.to(tl.int32) * padded_cnt[None, :], axis=1)
            total = tl.sum(tl.where(is_rep, padded_cnt, 0), axis=0)
            tl.store(NPOST, total.to(tl.int32))
            offs_b = tl.arange(0, NB)
            block_start = offs_b * block_size
            in_range = ((block_start[:, None] >= excl[None, :]) & (block_start[:, None] < (excl + padded_cnt)[None, :])
                        & is_rep[None, :])
            eid = tl.sum(in_range.to(tl.int32) * (bucket[None, :] - 1), axis=1)
            tl.store(EXPERT_IDS + offs_b, eid.to(tl.int32), mask=block_start < total)
            n_fill = (total + NP - 1) // NP
            for it in range(n_fill):
                f_offs = it * NP + offs_p
                tl.store(SORTED + f_offs, tl.full([NP], 0, tl.int32) + numel, mask=f_offs < total)
            tl.debug_barrier()
            tl.store(SORTED + excl + rank, offs_p.to(tl.int32), mask=mask_p)
            return
    if DO_LS:  # local_start into the one-token path's slot buffer (saves a separate device-to-device copy)
        if row == 0:
            tl.store(LSOUT, tl.load(LS))
    offs = tl.arange(0, H)
    x = tl.load(X + row * H + offs).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / H + eps)
    tl.store(OUT + row * H + offs, (x * rstd * tl.load(W + offs).to(tl.float32)).to(tl.bfloat16))
    k = tl.arange(0, TOPK)
    ids = tl.load(IDS + row * TOPK + k)
    rel = (ids - tl.load(LS) + N_EXP) % N_EXP
    tl.store(LOCAL + row * TOPK + k, tl.where(rel < N_LOCAL, rel, -1))


ALIGN_IN_NORM_MAX_SLOTS = 64  # SGLang's single-CTA align runs inside the norm kernel up to here
SORT_ALIGN_MAX_SLOTS = 256  # sort-based single-CTA align inside the norm kernel up to here (T <= 32)


MARLIN_BLOCK_CAP_MIN_TOKENS = 128  # at verify-16/32 sizes cap Marlin's block at 32 rows
MARLIN_BLOCK_CAP = 32
MARLIN_BLOCK_CAP_LARGE = 48  # from 256 tokens (verify-32) 48-row blocks


def _marlin_block_size(num_tokens):
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    if num_tokens >= MARLIN_BLOCK_CAP_MIN_TOKENS:
        block_size_m = min(block_size_m, MARLIN_BLOCK_CAP_LARGE if num_tokens >= 256 else MARLIN_BLOCK_CAP)
    return block_size_m


def _rms_norm_local_ids(x, w, eps, topk_ids, local_start, want_align=True, ls_out=None):
    """Returns (normed x, local ids, align) where align is (sorted_ids, block_experts, n_padded, block_size)
    for Marlin when computed here (T*8 <= 64 and not on the grouped path), else None."""
    out = torch.empty_like(x)
    local = torch.empty_like(topk_ids, dtype=torch.int32)
    n_tok = x.shape[0]
    numel = topk_ids.numel()
    sort_align = ALIGN_IN_NORM_MAX_SLOTS < numel <= SORT_ALIGN_MAX_SLOTS and n_tok < GROUPED_MIN_TOKENS
    do_align = want_align and (numel <= ALIGN_IN_NORM_MAX_SLOTS or sort_align)
    align = None
    if do_align:
        bs = _marlin_block_size(n_tok)
        # Buffer sizes exactly as moe_align_block_size(local_ids, bs, 256) allocates them.
        padded = numel * bs if numel < N_EXPERTS + 1 else numel + (N_EXPERTS + 1) * (bs - 1)
        padded = (padded + 3) // 4 * 4
        sorted_ids = torch.empty((padded,), dtype=torch.int32, device=x.device)
        block_experts = torch.empty((triton.cdiv(padded, bs),), dtype=torch.int32, device=x.device)
        n_post = torch.empty((1,), dtype=torch.int32, device=x.device)
        align = (sorted_ids, block_experts, n_post, bs)
        nb = triton.next_power_of_2(max(block_experts.numel(), 2))
    else:
        sorted_ids = block_experts = n_post = local
        bs, nb = 1, 2
    _rms_norm_local_ids_kernel[(n_tok + (1 if do_align else 0),)](
        x, w, out, eps, topk_ids, local_start, local, sorted_ids, block_experts, n_post, n_tok, bs,
        H=x.shape[1], TOPK=topk_ids.shape[1], N_EXP=N_EXPERTS, N_LOCAL=LOCAL_EXPERTS, DO_ALIGN=do_align,
        NP=triton.next_power_of_2(max(numel, 2)), NB=nb, SORT_ALIGN=sort_align, NBUCKET=64, num_warps=4,
        LSOUT=ls_out if ls_out is not None else local_start, DO_LS=ls_out is not None, **_PDL_KW)
    return out, local, align


# ---- grouped NVFP4 expert GEMMs for large token counts --------------------------------------
# Swap-AB tiles: the MMA's M side is 64 weight rows of one expert, the N side is up to 32 token-slots
# routed to it (moe_align_block_size order), so an expert with 20 tokens fills 20 of 32 columns instead
# of a padded 48/64-row Marlin block. Weights are the original NVFP4 tensors; E2M1 x E4M3 is exact in
# BF16 and the FP32 global scale is applied after the dot. Used from GROUPED_MIN_TOKENS tokens;
# Marlin (SGLang's kernel) stays below that.

@triton.jit
def _dequant_e2m1(p, sf, BN: tl.constexpr, BK: tl.constexpr):
    """p: uint8 [BN, BK//2] packed codes, sf: fp8e4nv [BN, BK//16] -> bf16 [BN, BK] (exact)."""
    pi = p.to(tl.int32)
    lo = pi & 0xF
    hi = pi >> 4
    c = tl.interleave(lo, hi)  # [BN, BK]: element 2i = low nibble
    e = (c >> 1) & 3
    m = c & 1
    mag = tl.where(e == 0, m * 0x3F00, ((e + 126) << 7) | (m << 6))
    bits = mag | ((c & 8) << 12)
    w = bits.to(tl.int16).to(tl.bfloat16, bitcast=True)
    s = sf.to(tl.bfloat16)
    w = tl.reshape(w, (BN, BK // 16, 16)) * tl.reshape(s, (BN, BK // 16, 1))
    return tl.reshape(w, (BN, BK))


# Marlin-style dequant: the 3 magnitude bits of an E2M1 code go to BF16 bits [8:6] (exponent low bits
# and mantissa MSB) and the sign to bit 15, so the BF16 value is the E2M1 value times 2^-126 (E=0 lands
# on a subnormal, still proportional). The block scale is applied in the same asm as sf * 2^60 (exact);
# the remaining 2^66 joins the FP32 global scale after the dot.
_DEQ_ASM = tl.constexpr("""
{
.reg .b32 t01, t23, m, a, b, z;
mov.b32 z, 0x80008000;
prmt.b32 t01, $4, 0, 0x4140;
prmt.b32 t23, $4, 0, 0x4342;
and.b32 m, t01, 0x00070007;
shl.b32 b, m, 6;
shl.b32 a, t01, 12;
lop3.b32 a, a, b, 0x80008000, 0xE4;
fma.rn.bf16x2 $0, a, $5, z;
and.b32 m, t23, 0x00070007;
shl.b32 b, m, 6;
shl.b32 a, t23, 12;
lop3.b32 a, a, b, 0x80008000, 0xE4;
fma.rn.bf16x2 $1, a, $6, z;
and.b32 m, t01, 0x00700070;
shl.b32 b, m, 2;
shl.b32 a, t01, 8;
lop3.b32 a, a, b, 0x80008000, 0xE4;
fma.rn.bf16x2 $2, a, $5, z;
and.b32 m, t23, 0x00700070;
shl.b32 b, m, 2;
shl.b32 a, t23, 8;
lop3.b32 a, a, b, 0x80008000, 0xE4;
fma.rn.bf16x2 $3, a, $6, z;
}
""")
BIAS_FIX = 2.0 ** 66  # E2M1 placement 2^-126 = in-asm scale 2^60 x epilogue 2^66


@triton.jit
def _dequant_asm(p, sf, BN: tl.constexpr, BK: tl.constexpr):
    """p uint8 [BN, BK//2], sf fp8e4nv [BN, BK//16] -> bf16 [BN, BK] equal to the dequant x 2^-66."""
    s = sf.to(tl.bfloat16).to(tl.int16, bitcast=True) + (60 << 7)  # x 2^60, exact
    s = s.to(tl.bfloat16, bitcast=True)
    s = tl.reshape(tl.broadcast_to(tl.reshape(s, (BN, BK // 16, 1)), (BN, BK // 16, 8)), (BN, BK // 2))
    lo, hi = tl.inline_asm_elementwise(_DEQ_ASM, "=r,=r,=r,=r,r,r,r", [p, s],
                                       dtype=(tl.bfloat16, tl.bfloat16), is_pure=True, pack=4)
    return tl.interleave(lo, hi)


@triton.jit
def _dequant(p, sf, BN: tl.constexpr, BK: tl.constexpr, DEQ: tl.constexpr):
    if DEQ == 1:
        return _dequant_asm(p, sf, BN, BK)
    else:
        return _dequant_e2m1(p, sf, BN, BK)


@triton.jit
def _moe_gate_up_kernel(X, W, SF, GS, SORTED, EXPERTS, NPOST, OUT,
                        stride_xt, stride_we, stride_wn, stride_se, stride_sn, stride_ot,
                        n_slots, I: tl.constexpr, K: tl.constexpr, TOPK: tl.constexpr,
                        BN: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, DEQ: tl.constexpr, GSCALE: tl.constexpr,
                        M_FAST: tl.constexpr, USE_PDL: tl.constexpr = False):
    if USE_PDL:  # launch early, wait for the producer; no early trigger (large grid)
        tl.extra.cuda.gdc_wait()
    if M_FAST:  # token-blocks of one weight tile run side by side, sharing it through L2
        pm = tl.program_id(0)
        pn = tl.program_id(1)
    else:
        pn = tl.program_id(0)
        pm = tl.program_id(1)
    if pm * BT >= tl.load(NPOST):
        return
    e = tl.load(EXPERTS + pm)
    if e < 0:
        return
    slots = tl.load(SORTED + pm * BT + tl.arange(0, BT))
    valid = slots < n_slots
    tok = tl.where(valid, slots // TOPK, 0)
    rn = pn * BN + tl.arange(0, BN)
    kp = tl.arange(0, BK // 2)
    ks = tl.arange(0, BK // 16)
    kk = tl.arange(0, BK)
    wbase = W + e.to(tl.int64) * stride_we
    sbase = SF + e.to(tl.int64) * stride_se
    acc_g = tl.zeros((BN, BT), tl.float32)
    acc_u = tl.zeros((BN, BT), tl.float32)
    for k0 in range(0, K, BK):
        xt = tl.load(X + tok[None, :] * stride_xt + (k0 + kk)[:, None], mask=valid[None, :], other=0.0)  # [BK, BT]
        pg = tl.load(wbase + rn[:, None] * stride_wn + (k0 // 2 + kp)[None, :])
        sg = tl.load(sbase + rn[:, None] * stride_sn + (k0 // 16 + ks)[None, :])
        acc_g = tl.dot(_dequant(pg, sg, BN, BK, DEQ), xt, acc_g)
        pu = tl.load(wbase + (I + rn)[:, None] * stride_wn + (k0 // 2 + kp)[None, :])
        su = tl.load(sbase + (I + rn)[:, None] * stride_sn + (k0 // 16 + ks)[None, :])
        acc_u = tl.dot(_dequant(pu, su, BN, BK, DEQ), xt, acc_u)
    gs = tl.load(GS + e) * GSCALE
    g = (acc_g * gs).to(tl.bfloat16).to(tl.float32)
    u = (acc_u * gs).to(tl.bfloat16).to(tl.float32)
    h = (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16)  # [BN, BT]
    tl.store(OUT + slots[None, :].to(tl.int64) * stride_ot + rn[:, None], h, mask=valid[None, :])


@triton.jit
def _moe_down_kernel(A, W, SF, GS, SORTED, EXPERTS, NPOST, OUT,
                     stride_at, stride_we, stride_wn, stride_se, stride_sn, stride_ot,
                     n_slots, K: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
                     DEQ: tl.constexpr, GSCALE: tl.constexpr, M_FAST: tl.constexpr, USE_PDL: tl.constexpr = False):
    if USE_PDL:  # launch early, wait for the producer; no early trigger (large grid)
        tl.extra.cuda.gdc_wait()
    if M_FAST:
        pm = tl.program_id(0)
        pn = tl.program_id(1)
    else:
        pn = tl.program_id(0)
        pm = tl.program_id(1)
    if pm * BT >= tl.load(NPOST):
        return
    e = tl.load(EXPERTS + pm)
    if e < 0:
        return
    slots = tl.load(SORTED + pm * BT + tl.arange(0, BT))
    valid = slots < n_slots
    srow = tl.where(valid, slots, 0)
    rn = pn * BN + tl.arange(0, BN)
    kp = tl.arange(0, BK // 2)
    ks = tl.arange(0, BK // 16)
    kk = tl.arange(0, BK)
    wbase = W + e.to(tl.int64) * stride_we
    sbase = SF + e.to(tl.int64) * stride_se
    acc = tl.zeros((BN, BT), tl.float32)
    for k0 in range(0, K, BK):
        at = tl.load(A + srow[None, :].to(tl.int64) * stride_at + (k0 + kk)[:, None], mask=valid[None, :], other=0.0)
        p = tl.load(wbase + rn[:, None] * stride_wn + (k0 // 2 + kp)[None, :])
        s = tl.load(sbase + rn[:, None] * stride_sn + (k0 // 16 + ks)[None, :])
        acc = tl.dot(_dequant(p, s, BN, BK, DEQ), at, acc)
    y = (acc * (tl.load(GS + e) * GSCALE)).to(tl.bfloat16)
    tl.store(OUT + slots[None, :].to(tl.int64) * stride_ot + rn[:, None], y, mask=valid[None, :])


def moe_nvfp4(x, w13, w13_sf, w13_gs, w2, w2_sf, w2_gs, sorted_ids, block_experts, n_post, top_k, bt,
              bn=64, bk=256, num_warps=4, num_stages=3, deq=1, m_fast=False):
    """x bf16 [T, K]; returns y bf16 [T*top_k, H] (rows of slots not routed here are unwritten)."""
    T, K = x.shape
    E, N1, _ = w13.shape
    I = N1 // 2
    H = w2.shape[1]
    n_slots = T * top_k
    grid_m = min(triton.cdiv(sorted_ids.numel(), bt), E + triton.cdiv(n_slots, bt))
    inter = torch.empty((n_slots, I), dtype=torch.bfloat16, device=x.device)
    g1 = (grid_m, I // bn) if m_fast else (I // bn, grid_m)
    g2 = (grid_m, H // bn) if m_fast else (H // bn, grid_m)
    _moe_gate_up_kernel[g1](
        x, w13, w13_sf, w13_gs, sorted_ids, block_experts, n_post, inter,
        x.stride(0), w13.stride(0), w13.stride(1), w13_sf.stride(0), w13_sf.stride(1), inter.stride(0),
        n_slots, I=I, K=K, TOPK=top_k, BN=bn, BT=bt, BK=bk, DEQ=deq, GSCALE=BIAS_FIX if deq == 1 else 1.0, M_FAST=m_fast,
        num_warps=num_warps, num_stages=num_stages, **_PDL_KW)
    y = torch.empty((n_slots, H), dtype=torch.bfloat16, device=x.device)
    _moe_down_kernel[g2](
        inter, w2, w2_sf, w2_gs, sorted_ids, block_experts, n_post, y,
        inter.stride(0), w2.stride(0), w2.stride(1), w2_sf.stride(0), w2_sf.stride(1), y.stride(0),
        n_slots, K=I, BN=bn, BT=bt, BK=bk, DEQ=deq, GSCALE=BIAS_FIX if deq == 1 else 1.0, M_FAST=m_fast,
        num_warps=num_warps, num_stages=num_stages, **_PDL_KW)
    return y


@triton.jit
def _gemv_kernel(X, W, OUT, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, USE_PDL: tl.constexpr = False,
                 PREF: tl.constexpr = False):
    """One token: out[n] = sum_k x[k] w[n, k], BF16 in, FP32 accumulation, BF16 out."""
    if USE_PDL and PREF:  # W is a constant weight, not the producer's output, so this CTA's W lines
        # are prefetched into L2 before the producer finishes (PDL early launch); the loads and arithmetic below are unchanged
        pr = tl.program_id(0) * BN + tl.arange(0, BN)
        lines = tl.arange(0, K // 64)  # one 128-byte line = 64 BF16
        pp = (W + pr[:, None] * K + lines[None, :] * 64).to(tl.int64)
        tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l", [pp], dtype=tl.int32,
                                  is_pure=False, pack=1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pn = tl.program_id(0)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + k0 + rk).to(tl.float32)
        w = tl.load(W + rn[:, None] * K + (k0 + rk)[None, :]).to(tl.float32)
        acc += w * x[None, :]
    tl.store(OUT + rn, tl.sum(acc, axis=1).to(tl.bfloat16))


GEMV_CFG = {8192: (1, 512, 8), 1024: (16, 256, 8), 4096: (1, 2048, 4)}  # K -> (BN, BK, warps), from an H200 sweep


# ---- one-launch split-K router for T > 8 (BF16 hi + lo of the FP32 router, FP32 accumulation) ----
@triton.jit
def router_hilo_kernel(X, HI, LO, OUT, T, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                       BK: tl.constexpr, KS: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    ks = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mm = rm < T
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(ks * (K // KS), (ks + 1) * (K // KS), BK):
        x = tl.load(X + rm[:, None] * K + (k0 + rk)[None, :], mask=mm[:, None], other=0.0)
        hi = tl.load(HI + (k0 + rk)[:, None] * N + rn[None, :])  # hi/lo stored K-major [K, N]
        lo = tl.load(LO + (k0 + rk)[:, None] * N + rn[None, :])
        acc = tl.dot(x, hi, acc)
        acc = tl.dot(x, lo, acc)
    tl.atomic_add(OUT + rm[:, None] * N + rn[None, :], acc, mask=mm[:, None], sem="relaxed")


def _router_hilo(x, hi_t, lo_t):
    t, k = x.shape
    n = hi_t.shape[1]
    # tiles from an H200 sweep; the 16/32-token rows were already within noise of the best
    if t <= 64:
        bm, bn, bk, ks, nw, ns = 16, 32, 128, 32, 4, 3
    elif t <= 128:
        bm, bn, bk, ks, nw, ns = 64, 64, 128, 16, 4, 4
    else:
        bm, bn, bk, ks, nw, ns = 64, 64, 128, 8, 4, 4
    if not _SWAP_SM90:  # pipelining depth only (same tiles and arithmetic): sm_120 has ~101 KB shared memory per block
        ns = min(ns, 2)
    out = torch.zeros((t, n), dtype=torch.float32, device=x.device)
    router_hilo_kernel[(triton.cdiv(t, bm), n // bn, ks)](x, hi_t, lo_t, out, t, K=k, N=n, BM=bm, BN=bn, BK=bk, KS=ks,
                                                         num_warps=nw, num_stages=ns)
    return out



# ---- split-K full-attention decode kernels (adapted from the muse_attention_decode kernels) ----

@triton.jit
def _full_partition_kernel(Q, K, V, PT, POS, KS, VS, PART, LSE,
                          HQ: tl.constexpr, HK: tl.constexpr, L: tl.constexpr,
                          PAGE: tl.constexpr, PT_STRIDE: tl.constexpr,
                          SCALE: tl.constexpr, GROUP: tl.constexpr,
                          NS: tl.constexpr, SP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    token = tl.program_id(0)
    kvhead = tl.program_id(1)
    split = tl.program_id(2)
    batch = token // L
    gh = tl.arange(0, BM)
    qheads = kvhead * GROUP + gh
    d = tl.arange(0, 64)
    pos = tl.load(POS + token).to(tl.int32)
    first = split * SP
    part_base = ((token * HK + kvhead) * NS + split) * GROUP
    if first <= pos:
        query = tl.load(Q + (token * HQ + qheads[:, None]) * 64 + d[None, :],
                        mask=(gh < GROUP)[:, None], other=0.0).to(tl.float16)
        count = tl.minimum(SP, pos - first + 1)
        maximum = tl.full((BM,), -float('inf'), tl.float32)
        denom = tl.full((BM,), 0.0, tl.float32)
        accum = tl.zeros((BM, 64), tl.float32)
        scale = SCALE * tl.load(KS)
        for block in range(tl.cdiv(count, BN)):
            logical = first + block * BN + tl.arange(0, BN)
            valid = logical <= pos
            physical = tl.load(PT + batch * PT_STRIDE + logical // PAGE, mask=valid, other=0).to(tl.int64)
            rows = physical * PAGE + logical % PAGE
            keys = tl.load(K + (rows[None, :] * HK + kvhead) * 64 + d[:, None],
                           mask=valid[None, :], other=0.0).to(tl.float16)
            logits = tl.dot(query, keys) * scale
            logits = tl.where(valid[None, :], logits, -float('inf'))
            next_max = tl.maximum(maximum, tl.max(logits, axis=1))
            alpha = tl.exp(maximum - next_max)
            probs = tl.exp(logits - next_max[:, None])
            denom = denom * alpha + tl.sum(probs, axis=1)
            values = tl.load(V + (rows[:, None] * HK + kvhead) * 64 + d[None, :],
                             mask=valid[:, None], other=0.0).to(tl.float16)
            accum = accum * alpha[:, None] + tl.dot(probs.to(tl.float16), values)
            maximum = next_max
        partial = accum * (tl.load(VS) / denom)[:, None]
        lse = maximum + tl.log(denom)
    else:
        partial = tl.zeros((BM, 64), tl.float32)
        lse = tl.full((BM,), -float('inf'), tl.float32)
    tl.store(PART + (part_base + gh[:, None]) * 64 + d[None, :], partial, mask=(gh < GROUP)[:, None])
    tl.store(LSE + part_base + gh, lse, mask=gh < GROUP)


@triton.jit
def _merge_full_kernel(PART, LSE, O, HQ: tl.constexpr, HK: tl.constexpr,
                         GROUP: tl.constexpr, BG: tl.constexpr, NS: tl.constexpr):
    token = tl.program_id(0)
    kvhead = tl.program_id(1)
    split = tl.arange(0, NS)
    gh = tl.arange(0, BG)
    d = tl.arange(0, 64)
    base = ((token * HK + kvhead) * NS + split[:, None]) * GROUP + gh[None, :]
    lse = tl.load(LSE + base, mask=gh[None, :] < GROUP, other=-float('inf'))
    maximum = tl.max(lse, axis=0)
    weights = tl.exp(lse - maximum[None, :])
    partial = tl.load(PART + base[:, :, None] * 64 + d[None, None, :],
                      mask=(gh[None, :, None] < GROUP), other=0.0)
    accum = tl.sum(partial * weights[:, :, None], axis=0)
    result = accum / tl.sum(weights, axis=0)[:, None]
    qheads = kvhead * GROUP + gh
    tl.store(O + (token * HQ + qheads[:, None]) * 64 + d[None, :], result, mask=(gh < GROUP)[:, None])


def _b_full_attention(q, data):
    """full_attention for single-token full-attention decode (64 partitions, then merge)."""
    cfg = data["config"]
    hq, hk, length = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["q_len"]
    tokens = cfg["batch"] * length
    group = hq // hk
    capacity = data["page_table"].shape[1] * cfg["page_size"]
    partitions = 64
    partition_tokens = triton.cdiv(capacity, partitions * 128) * 128
    out = torch.empty((tokens, hq, 64), device=q.device, dtype=torch.bfloat16)
    partial = torch.empty((tokens, hk, partitions, group, 64), device=q.device, dtype=torch.float32)
    lse = torch.empty((tokens, hk, partitions, group), device=q.device, dtype=torch.float32)
    _full_partition_kernel[(tokens, hk, partitions)](q, data["k_cache"], data["v_cache"],
        data["page_table"], data["positions"], data["k_scale"], data["v_scale"], partial, lse,
        hq, hk, length, cfg["page_size"], data["page_table"].stride(0),
        cfg["sm_scale"] * cfg["qk_scale_factor"], group, partitions, partition_tokens,
        max(16, triton.next_power_of_2(group)), 128, num_warps=4, num_stages=3)  # 3 stages (measured on the H200)
    _merge_full_kernel[(tokens, hk)](partial, lse, out, hq, hk, group,
        triton.next_power_of_2(group), partitions, num_warps=8)
    return out


# ---- full-attention verify blocks (q_len 8) as one 64-row tile per request and KV head ----
# _full_partition_kernel runs one program per token: at q_len 8 the 8 tokens of a request re-read the same
# keys. Here one program per (request, KV head, split) holds the request's L tokens x GROUP query heads as one
# [L * GROUP, 64] tile (64 rows at L = 8, GROUP = 8), reads each key block once, and masks per row: token t of the
# request attends to keys 0 .. positions[t] (causal within the draft). Same FP8 -> FP16 operands, FP32 softmax and
# scale handling as _full_partition_kernel; partials are written in its [token, KV head, split, group, 64] layout,
# so _merge_full_kernel merges them unchanged.
@triton.jit
def _verify_partition_kernel(Q, K, V, PT, POS, KS, VS, PART, LSE,
                             HQ: tl.constexpr, HK: tl.constexpr, L: tl.constexpr,
                             PAGE: tl.constexpr, PT_STRIDE: tl.constexpr,
                             SCALE: tl.constexpr, GROUP: tl.constexpr,
                             NS: tl.constexpr, SP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                             FP8DOT: tl.constexpr = False, EXP2: tl.constexpr = False):
    batch = tl.program_id(0)
    kvhead = tl.program_id(1)
    split = tl.program_id(2)
    r = tl.arange(0, BM)
    t = r // GROUP
    g = r % GROUP
    live = t < L
    token = batch * L + t
    qheads = kvhead * GROUP + g
    d = tl.arange(0, 64)
    pos_row = tl.load(POS + token, mask=live, other=-1).to(tl.int32)
    pos = tl.load(POS + batch * L + L - 1).to(tl.int32)  # the request's last (largest) position
    first = split * SP
    part_row = ((token * HK + kvhead) * NS + split) * GROUP + g
    if first <= pos:
        query = tl.load(Q + (token[:, None] * HQ + qheads[:, None]) * 64 + d[None, :], mask=live[:, None], other=0.0)
        if not FP8DOT:
            query = query.to(tl.float16)
        count = tl.minimum(SP, pos - first + 1)
        maximum = tl.full((BM,), -float('inf'), tl.float32)
        denom = tl.full((BM,), 0.0, tl.float32)
        accum = tl.zeros((BM, 64), tl.float32)
        scale = SCALE * tl.load(KS)
        if EXP2:
            scale = scale * 1.4426950408889634  # log2(e): exp2 on base-2 logits; lse is converted back below
        for block in range(tl.cdiv(count, BN)):
            logical = first + block * BN + tl.arange(0, BN)
            valid = logical <= pos
            physical = tl.load(PT + batch * PT_STRIDE + logical // PAGE, mask=valid, other=0).to(tl.int64)
            rows = physical * PAGE + logical % PAGE
            keys = tl.load(K + (rows[None, :] * HK + kvhead) * 64 + d[:, None], mask=valid[None, :], other=0.0)
            if not FP8DOT:
                keys = keys.to(tl.float16)
            logits = tl.dot(query, keys) * scale
            seen = logical[None, :] <= pos_row[:, None]
            logits = tl.where(seen, logits, -float('inf'))
            next_max = tl.maximum(maximum, tl.max(logits, axis=1))
            # a row whose keys all lie past its position in this block keeps -inf: guard the exponent base
            base = tl.where(next_max == -float('inf'), 0.0, next_max)
            if EXP2:
                alpha = tl.math.exp2(maximum - base)
                probs = tl.math.exp2(logits - base[:, None])
            else:
                alpha = tl.exp(maximum - base)
                probs = tl.exp(logits - base[:, None])
            denom = denom * alpha + tl.sum(probs, axis=1)
            values = tl.load(V + (rows[:, None] * HK + kvhead) * 64 + d[None, :], mask=valid[:, None], other=0.0)
            if FP8DOT:  # FA3's FP8 path: probabilities rounded to E4M3 for the P.V product
                accum = accum * alpha[:, None] + tl.dot(probs.to(tl.float8e4nv), values)
            else:
                accum = accum * alpha[:, None] + tl.dot(probs.to(tl.float16), values.to(tl.float16))
            maximum = next_max
        some = denom > 0
        partial = accum * tl.where(some, tl.load(VS) / tl.where(some, denom, 1.0), 0.0)[:, None]
        if EXP2:
            lse = tl.where(some, (maximum + tl.math.log2(tl.where(some, denom, 1.0))) * 0.6931471805599453, -float('inf'))
        else:
            lse = tl.where(some, maximum + tl.log(tl.where(some, denom, 1.0)), -float('inf'))
    else:
        partial = tl.zeros((BM, 64), tl.float32)
        lse = tl.full((BM,), -float('inf'), tl.float32)
    tl.store(PART + part_row[:, None] * 64 + d[None, :], partial, mask=live[:, None])
    tl.store(LSE + part_row, lse, mask=live)


VERIFY_ATTN_PARTITIONS = 64
VERIFY_ATTN_BN = 128
VERIFY_ATTN_STAGES = 3
VERIFY_ATTN_FP8DOT = False
VERIFY_ATTN_EXP2 = False
VERIFY_ATTN_WARPS = 4


def _v_full_attention(q, data):
    """Full-attention verify block: _verify_partition_kernel (one tile per request and KV head), then _merge_full_kernel."""
    cfg = data["config"]
    hq, hk, length, batch = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["q_len"], cfg["batch"]
    tokens = batch * length
    group = hq // hk
    capacity = data["page_table"].shape[1] * cfg["page_size"]
    partitions = VERIFY_ATTN_PARTITIONS
    partition_tokens = triton.cdiv(capacity, partitions * VERIFY_ATTN_BN) * VERIFY_ATTN_BN
    out = torch.empty((tokens, hq, 64), device=q.device, dtype=torch.bfloat16)
    partial = torch.empty((tokens, hk, partitions, group, 64), device=q.device, dtype=torch.float32)
    lse = torch.empty((tokens, hk, partitions, group), device=q.device, dtype=torch.float32)
    _verify_partition_kernel[(batch, hk, partitions)](q, data["k_cache"], data["v_cache"],
        data["page_table"], data["positions"], data["k_scale"], data["v_scale"], partial, lse,
        hq, hk, length, cfg["page_size"], data["page_table"].stride(0),
        cfg["sm_scale"] * cfg["qk_scale_factor"], group, partitions, partition_tokens,
        max(16, triton.next_power_of_2(length * group)), VERIFY_ATTN_BN, FP8DOT=VERIFY_ATTN_FP8DOT, EXP2=VERIFY_ATTN_EXP2,
        num_warps=VERIFY_ATTN_WARPS, num_stages=VERIFY_ATTN_STAGES)
    _merge_full_kernel[(tokens, hk)](partial, lse, out, hq, hk, group,
        triton.next_power_of_2(group), partitions, num_warps=8)
    return out

# ---- full attention at small batch (B < 16): more splits, and a merge that loops over them ----
# _merge_full_kernel holds all NS x GROUP x 64 partials in registers (fine at NS = 64); with NS up to 512 this
# merge streams them: one pass over the splits with a running max / sum (the same log-sum-exp combination).
@triton.jit
def _merge_loop_kernel(PART, LSE, O, HQ: tl.constexpr, HK: tl.constexpr, GROUP: tl.constexpr, BG: tl.constexpr,
                       NS: tl.constexpr):
    token = tl.program_id(0)
    kvhead = tl.program_id(1)
    gh = tl.arange(0, BG)
    d = tl.arange(0, 64)
    gm = gh < GROUP
    base = (token * HK + kvhead) * NS * GROUP
    maximum = tl.full((BG,), -float('inf'), tl.float32)
    total = tl.zeros((BG,), tl.float32)
    accum = tl.zeros((BG, 64), tl.float32)
    for s in range(NS):
        lse = tl.load(LSE + base + s * GROUP + gh, mask=gm, other=-float('inf'))
        nxt = tl.maximum(maximum, lse)
        safe = tl.where(nxt == -float('inf'), 0.0, nxt)
        alpha = tl.exp(maximum - safe)
        w = tl.exp(lse - safe)
        part = tl.load(PART + (base + s * GROUP + gh[:, None]) * 64 + d[None, :], mask=gm[:, None], other=0.0)
        accum = accum * alpha[:, None] + part * w[:, None]
        total = total * alpha + w
        maximum = nxt
    qheads = kvhead * GROUP + gh
    tl.store(O + (token * HQ + qheads[:, None]) * 64 + d[None, :], accum / total[:, None], mask=gm[:, None])


@triton.jit
def _merge_vec_kernel(PART, LSE, O, HQ: tl.constexpr, HK: tl.constexpr, GROUP: tl.constexpr, NS: tl.constexpr):
    """One program per (token, KV head, grouped query head): all NS partials of that head in one [NS, 64] load
    (the streaming merge above is serial over the splits and only has tokens x HK programs)."""
    token = tl.program_id(0)
    kvhead = tl.program_id(1)
    g = tl.program_id(2)
    s = tl.arange(0, NS)
    d = tl.arange(0, 64)
    rows = ((token * HK + kvhead) * NS + s) * GROUP + g
    lse = tl.load(LSE + rows)
    maximum = tl.max(lse, axis=0)
    w = tl.exp(lse - maximum)
    part = tl.load(PART + rows[:, None] * 64 + d[None, :])
    result = tl.sum(part * w[:, None], axis=0) / tl.sum(w, axis=0)
    tl.store(O + (token * HQ + kvhead * GROUP + g) * 64 + d, result)


SMALL_FULL_PARTITIONS = {1: 128, 8: 128}  # q_len -> splits at batch < 16 (measured on the H200)
SMALL_FULL_BN = 128


def _s_full_attention(q, data, partitions=None):
    """Full attention at small batch: the one-token split kernel (q_len 1) or the verify tile kernel (q_len 8)
    over `partitions` splits per request and KV head, then the streaming merge."""
    cfg = data["config"]
    hq, hk, length, batch = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["q_len"], cfg["batch"]
    tokens = batch * length
    group = hq // hk
    partitions = partitions or SMALL_FULL_PARTITIONS[length]
    capacity = data["page_table"].shape[1] * cfg["page_size"]
    partition_tokens = triton.cdiv(capacity, partitions * SMALL_FULL_BN) * SMALL_FULL_BN
    out = torch.empty((tokens, hq, 64), device=q.device, dtype=torch.bfloat16)
    partial = torch.empty((tokens, hk, partitions, group, 64), device=q.device, dtype=torch.float32)
    lse = torch.empty((tokens, hk, partitions, group), device=q.device, dtype=torch.float32)
    args = (q, data["k_cache"], data["v_cache"], data["page_table"], data["positions"], data["k_scale"],
            data["v_scale"], partial, lse, hq, hk, length, cfg["page_size"], data["page_table"].stride(0),
            cfg["sm_scale"] * cfg["qk_scale_factor"], group, partitions, partition_tokens)
    if length == 1:
        _full_partition_kernel[(tokens, hk, partitions)](*args, max(16, triton.next_power_of_2(group)), SMALL_FULL_BN,
                                                         num_warps=4, num_stages=3)
    else:
        _verify_partition_kernel[(batch, hk, partitions)](*args, max(16, triton.next_power_of_2(length * group)),
                                                          SMALL_FULL_BN, num_warps=4, num_stages=3)
    _merge_vec_kernel[(tokens, hk, group)](partial, lse, out, hq, hk, group, partitions,
                                           num_warps=4 if partitions <= 128 else 8)
    return out

SMALL_FULL_MAX_BATCH = 1  # full layers at q_len 1 / 8 with at most this many requests (0 = off)
VERIFY_ATTN_MIN_BATCH = 1073741824  # verify blocks of this many requests or more take this path


# ---- small-M dense projections (8-32 tokens). Swap-AB tiles: BN weight rows on the MMA M side, the tokens
# padded to BM on the N side, FP32 accumulation (split-K into an FP32 buffer + reduce where measured faster), BF16 out.
@triton.jit
def _gemm_swap_kernel(X, W, OUT, M, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr,
                      BK: tl.constexpr, USE_PDL: tl.constexpr = False, PREF: tl.constexpr = False):
    if USE_PDL and PREF:  # this CTA's constant W lines into L2 before the producer finishes (as in the one-token GEMV)
        pr = tl.program_id(0) * BN + tl.arange(0, BN)
        lines = tl.arange(0, K // 64)  # one 128-byte line = 64 BF16
        pp = (W + pr[:, None] * K + lines[None, :] * 64).to(tl.int64)
        tl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l", [pp], dtype=tl.int32,
                                  is_pure=False, pack=1)
    if USE_PDL:  # as in the one-token GEMV: wait for the producer grid, then let the next kernel launch
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pn = tl.program_id(0)
    rn = pn * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BN, BM), tl.float32)
    for k0 in range(0, K, BK):
        w = tl.load(W + rn[:, None] * K + (k0 + rk)[None, :])
        x = tl.load(X + rm[None, :] * K + (k0 + rk)[:, None], mask=rm[None, :] < M, other=0.0)
        acc = tl.dot(w, x, acc)
    tl.store(OUT + rm[None, :] * N + rn[:, None], acc.to(tl.bfloat16), mask=rm[None, :] < M)


@triton.jit
def _gemm_swap_splitk_kernel(X, W, P, M, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BM: tl.constexpr,
                             BK: tl.constexpr, SPLIT: tl.constexpr, USE_PDL: tl.constexpr = False):
    if USE_PDL:  # as in the one-token GEMV: wait for the producer grid, then let the next kernel launch
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BN, BM), tl.float32)
    KS: tl.constexpr = K // SPLIT
    for k0 in range(ps * KS, ps * KS + KS, BK):
        w = tl.load(W + rn[:, None] * K + (k0 + rk)[None, :])
        x = tl.load(X + rm[None, :] * K + (k0 + rk)[:, None], mask=rm[None, :] < M, other=0.0)
        acc = tl.dot(w, x, acc)
    tl.store(P + ps * M * N + rm[None, :] * N + rn[:, None], acc, mask=rm[None, :] < M)


@triton.jit
def _splitk_reduce_kernel(P, OUT, MN, SPLIT: tl.constexpr, BLOCK: tl.constexpr, USE_PDL: tl.constexpr = False):
    if USE_PDL:  # as in the one-token GEMV: wait for the producer grid, then let the next kernel launch
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < MN
    a = tl.zeros((BLOCK,), tl.float32)
    for s in range(SPLIT):
        a += tl.load(P + s * MN + o, mask=m, other=0.0)
    tl.store(OUT + o, a.to(tl.bfloat16), mask=m)


# (N, K, M) -> (BN, BK, SPLIT, stages), from an H200 sweep; the T = 32 pooler runs BN 64 / 4 stages (64 CTAs, same K
# order, bitwise-identical output), chosen in the full layer, where it beats (32, 256, 1, 5) next to the co-running router
SWAP_CFG = {(2304, 8192, 8): (128, 128, 4, 5), (2304, 8192, 16): (128, 128, 4, 4), (2304, 8192, 32): (128, 128, 4, 5),
            (8192, 1024, 8): (64, 256, 1, 5), (8192, 1024, 16): (64, 128, 1, 5), (8192, 1024, 32): (64, 128, 1, 5),
            (4096, 8192, 8): (64, 128, 4, 3), (4096, 8192, 16): (32, 256, 1, 5), (4096, 8192, 32): (64, 256, 1, 4),
            (8192, 4096, 16): (64, 128, 1, 5), (8192, 4096, 32): (64, 128, 1, 5)}


# Pipelining depth only (same tiles and K order, so identical arithmetic): the H200-tuned depth on SM90; two stages
# elsewhere, where shared memory per block is smaller (sm_120 ~101 KB).
_SWAP_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def _gemm_swap(x, w, cfg, pref=False):
    m, k = x.shape
    n = w.shape[0]
    bn, bk, split, stages = cfg
    if not _SWAP_SM90:
        stages = 2
    bm = max(16, triton.next_power_of_2(m))
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if split == 1:
        _gemm_swap_kernel[(n // bn,)](x, w, out, m, N=n, K=k, BN=bn, BM=bm, BK=bk, num_warps=4, num_stages=stages,
                                      PREF=pref, **_PDL_KW)  # L2 weight prefetch: o-proj at sliding batch 1 only (kernel_fn)
    else:
        p = torch.empty((split, m, n), dtype=torch.float32, device=x.device)
        _gemm_swap_splitk_kernel[(n // bn, split)](x, w, p, m, N=n, K=k, BN=bn, BM=bm, BK=bk, SPLIT=split,
                                                   num_warps=4, num_stages=stages, **_PDL_KW)
        _splitk_reduce_kernel[(triton.cdiv(m * n, 1024),)](p, out, m * n, SPLIT=split, BLOCK=1024, num_warps=4,
                                                           **_PDL_KW)
    return out



# ---- 8-token FP32 router ----

@triton.jit
def _router_eight_tokens(X, W, Y, K: tl.constexpr, N: tl.constexpr,
                         SXM: tl.constexpr, SXK: tl.constexpr,
                         SW: tl.constexpr, TILE_K: tl.constexpr):
    row = tl.program_id(0)
    tokens = tl.arange(0, 8)
    columns = tl.arange(0, TILE_K)
    accum = tl.full((8, TILE_K), 0, tl.float32)
    for tile in range(triton.cdiv(K, TILE_K)):
        k = tile * TILE_K + columns
        x = tl.load(X + tokens[:, None] * SXM + k[None, :] * SXK,
                    mask=k[None, :] < K, other=0).to(tl.float32)
        w = tl.load(W + row * SW + k, mask=k < K, other=0)
        accum += x * w[None, :]
    tl.store(Y + tokens * N + row, tl.sum(accum, axis=1))


def _router_eight(router_input, w):
    """8-token router: FP32 x . FP32 w per output column, one program per router column."""
    n = w.shape[0]
    out = torch.empty((8, n), device=w.device, dtype=torch.float32)
    _router_eight_tokens[(n,)](router_input, w, out, router_input.shape[1], n, router_input.stride(0),
                               router_input.stride(1), w.stride(0), TILE_K=1024, num_warps=8)
    return out

# ---- Gluon full-attention decode (adapted from the muse_attention_decode kernels) ----
_COUNTERS = {}


# ---------------------------------------------------------------------------------------------------------
# Full-layer decode on SM90 (Gluon): every warp owns BN / W keys of each 128-byte-row tile and keeps its own
# online-softmax state; S_h = K_h Q_h^T runs as mma with keys as M and the 8 query rows of a KV head as N
# (E4M3 operands, fp32 accumulate), O_h^T += V_h^T P_h in fp16. The warps' states merge once per program;
# the last-arriving program merges the splits exactly and applies RMSNorm x sigmoid(gate), as _fused_attn.

@gluon.jit
def _g_issue(k_cache, v_cache, k_smem, v_smem, rows, nmask, slot, ROW: gl.constexpr, ld3: gl.constexpr):
    col = gl.arange(0, ROW, layout=gl.SliceLayout(0, gl.SliceLayout(1, ld3)))
    off = rows.to(gl.int64)[:, :, None] * ROW + col[None, None, :]
    m3 = nmask[:, :, None] & (col[None, None, :] < ROW)
    gcp.async_copy_global_to_shared(k_smem.index(slot), k_cache + off, mask=m3)
    gcp.async_copy_global_to_shared(v_smem.index(slot), v_cache + off, mask=m3)
    gcp.commit_group()


@gluon.jit
def _g_head_q(qkv_ptr, qrow, h, eps, G: gl.constexpr, D: gl.constexpr, W: gl.constexpr, bl3: gl.constexpr,
              b8_l: gl.constexpr):
    """Q_h^T [W, D, G] (replicated per warp): RMSNorm, BF16, E4M3 (as _fused_attn, no RoPE on full layers)."""
    qw = gl.arange(0, W, layout=gl.SliceLayout(1, gl.SliceLayout(2, bl3)))
    qd = gl.arange(0, D, layout=gl.SliceLayout(0, gl.SliceLayout(2, bl3)))
    qg = gl.arange(0, G, layout=gl.SliceLayout(0, gl.SliceLayout(1, bl3)))
    x = gl.load(qrow + qw[:, None, None] * 0 + (h * G + qg[None, None, :]) * D + qd[None, :, None]).to(gl.float32)
    inv = gl.rsqrt(gl.sum(x * x, axis=1) / D + eps)
    x = (x * inv[:, None, :]).to(gl.bfloat16).to(gl.float32)
    return gl.convert_layout(x.to(gl.float8e4nv), b8_l)


@gluon.jit
def _g_head_loop(k_smem, v_smem, slot, q8, ok, scale, m, l, o, h: gl.constexpr, D: gl.constexpr,
                 W: gl.constexpr, KW: gl.constexpr, G: gl.constexpr, mma3: gl.constexpr, a8_l: gl.constexpr,
                 a16_l: gl.constexpr, b16_l: gl.constexpr):
    kt = k_smem.index(slot)
    vt = v_smem.index(slot)
    sc = mma_v2(kt.slice(h * D, D, dim=2).load(a8_l), q8, gl.zeros([W, KW, G], gl.float32, layout=mma3)) * scale
    sc = gl.where(ok[:, :, None], sc, -float("inf"))
    mn = gl.maximum(m, gl.max(sc, axis=1))
    mu = gl.where(mn == -float("inf"), 0.0, mn)
    p = gl.exp2(sc - mu[:, None, :])
    a = gl.exp2(m - mu)
    l = l * a + gl.sum(p, axis=1)
    v = vt.slice(h * D, D, dim=2).permute([0, 2, 1]).load(a16_l).to(gl.float16)
    o = mma_v2(v, gl.convert_layout(p.to(gl.float16), b16_l), o * a[:, None, :])
    return mn, l, o


@gluon.jit
def _g_head_finish(qkv_ptr, qrow, loc, k_cache, v_cache, k_scale, v_scale, scale, eps, m, l, o, part_o, part_ml,
                   pbase, s, h: gl.constexpr, HQ: gl.constexpr, HKV: gl.constexpr, D: gl.constexpr,
                   G: gl.constexpr, mma3: gl.constexpr):
    """Merge the warps' states, add the block's new key on split 0 (stored to the cache there), store the
    split's partial for rows h*G .. h*G+G-1."""
    o2_l: gl.constexpr = gl.SliceLayout(0, mma3)
    sg_l: gl.constexpr = gl.SliceLayout(0, o2_l)
    sd_l: gl.constexpr = gl.SliceLayout(1, o2_l)
    mt = gl.max(m, axis=0)
    mz = gl.where(mt == -float("inf"), 0.0, mt)
    w = gl.exp2(m - mz[None, :])
    lt = gl.convert_layout(gl.sum(l * w, axis=0), sg_l)
    ot = gl.sum(o * w[:, None, :], axis=0)
    mt = gl.convert_layout(mt, sg_l)
    dd = gl.arange(0, D, layout=sd_l)
    gg = gl.arange(0, G, layout=sg_l)
    if s == 0:
        kx = gl.load(qrow + (HQ + h) * D + dd).to(gl.float32)
        kinv = gl.rsqrt(gl.sum(kx * kx, axis=0) / D + eps)
        k8 = ((kx * kinv).to(gl.bfloat16).to(gl.float32) / k_scale).to(gl.float8e4nv)
        vx = gl.load(qrow + (HQ + HKV + h) * D + dd).to(gl.float32)
        v8 = (vx / v_scale).to(gl.float8e4nv)
        coff = loc * (HKV * D) + h * D + dd
        gl.store(k_cache + coff, k8)
        gl.store(v_cache + coff, v8)
        qx = gl.load(qrow + (h * G + gg[None, :]) * D + dd[:, None]).to(gl.float32)
        qinv = gl.rsqrt(gl.sum(qx * qx, axis=0) / D + eps)
        qf = (qx * qinv[None, :]).to(gl.bfloat16).to(gl.float32).to(gl.float8e4nv).to(gl.float32)
        sn = gl.sum(qf * k8.to(gl.float32)[:, None], axis=0) * scale
        mn = gl.maximum(mt, sn)
        an = gl.exp2(mt - mn)
        pn = gl.exp2(sn - mn)
        lt = lt * an + pn
        ot = ot * an[None, :] + pn[None, :] * v8.to(gl.float32)[:, None]
        mt = mn
    r = h * G + gg
    gl.store(part_o + (pbase + r[None, :]) * D + dd[:, None], ot)
    gl.store(part_ml + (pbase + r) * 2, mt)
    gl.store(part_ml + (pbase + r) * 2 + 1, lt)


@gluon.jit
def _gv_softmax_pv(q_s, k_t, v16_s, ok, scale, m, l, o, mma_s: gl.constexpr, mma_o: gl.constexpr,
                   R: gl.constexpr, BN: gl.constexpr):
    sc = warpgroup_mma(q_s, k_t.permute([1, 0]), gl.zeros([R, BN], gl.float32, layout=mma_s)) * scale
    sc = gl.where(ok, sc, -float("inf"))
    mn = gl.maximum(m, gl.max(sc, axis=1))
    mu = gl.where(mn == -float("inf"), 0.0, mn)
    p = gl.exp2(sc - mu[:, None])
    a = gl.exp2(m - mu)
    l = l * a + gl.sum(p, axis=1)
    ao = gl.convert_layout(a, gl.SliceLayout(1, mma_o))
    pa = gl.convert_layout(p.to(gl.float16), gl.DotOperandLayout(0, mma_o, 2))
    o = warpgroup_mma(pa, v16_s, o * ao[:, None])
    return mn, l, o


@gluon.jit
def _gv_producer(pt_row, start, end, ntiles, k_cache, v_cache, k_ring, v_ring, full, empty, h,
                 BN: gl.constexpr, STAGES: gl.constexpr, D: gl.constexpr, ROW: gl.constexpr):
    ld: gl.constexpr = gl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])
    kn = gl.arange(0, BN, layout=gl.SliceLayout(1, ld))
    dq = gl.arange(0, D, layout=gl.SliceLayout(0, ld))
    for t in range(0, ntiles):
        slot = t % STAGES
        ph = (t // STAGES) & 1
        gmbar.wait(empty.index(slot), ph ^ 1)
        n = start + t * BN + kn
        nm = n < end
        rows = gl.load(pt_row + n, mask=nm, other=0).to(gl.int64)
        off = rows[:, None] * ROW + h * D + dq[None, :]
        m2 = nm[:, None] & (dq[None, :] < D)
        gcp.async_copy_global_to_shared(k_ring.index(slot), k_cache + off, mask=m2)
        gcp.async_copy_global_to_shared(v_ring.index(slot), v_cache + off, mask=m2)
        gcp.mbarrier_arrive(full.index(slot), increment_count=False)


@gluon.jit
def _gv_consumer(q_s, k_ring, v_ring, v16_s, full, empty, start, end, ntiles, scale,
                 BN: gl.constexpr, STAGES: gl.constexpr, R: gl.constexpr, D: gl.constexpr):
    blk: gl.constexpr = gl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])
    mma_s: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BN, 32])
    mma_o: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, D, 16])
    sn = gl.arange(0, BN, layout=gl.SliceLayout(0, mma_s))
    m = gl.full([R], -float("inf"), gl.float32, layout=gl.SliceLayout(1, mma_s))
    l = gl.zeros([R], gl.float32, layout=gl.SliceLayout(1, mma_s))
    o = gl.zeros([R, D], gl.float32, layout=mma_o)
    for t in range(0, ntiles):
        slot = t % STAGES
        ph = (t // STAGES) & 1
        gmbar.wait(full.index(slot), ph)
        vt = v_ring.index(slot).load(blk)
        v16_s.store(vt.to(gl.float16))
        fence_async_shared()
        gl.barrier()
        ok = ((start + t * BN + sn) < end)[None, :]
        m, l, o = _gv_softmax_pv(q_s, k_ring.index(slot), v16_s, ok, scale, m, l, o, mma_s, mma_o, R, BN)
        gl.barrier()
        gmbar.arrive(empty.index(slot))
    return m, l, o


@gluon.jit
def _g_full_verify(qkv_ptr, loc_ptr, ks_ptr, vs_ptr, k_cache, v_cache, pt_ptr, seq_ptr, part_o, part_ml, cnt_ptr,
                   out_ptr, qkv_stride, pt_stride, chunk, sm_scale, eps,
                   HQ: gl.constexpr, HKV: gl.constexpr, D: gl.constexpr, L: gl.constexpr, NSPLIT: gl.constexpr,
                   NSP2: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr, PR: gl.constexpr):
    G: gl.constexpr = HQ // HKV
    R: gl.constexpr = L * G
    ROW: gl.constexpr = HKV * D
    NP: gl.constexpr = 16
    LOG2E: gl.constexpr = 1.4426950408889634
    h = gl.program_id(0)
    b = gl.program_id(1)
    s = gl.program_id(2)
    seq = gl.load(seq_ptr + b)
    k_scale = gl.load(ks_ptr)
    v_scale = gl.load(vs_ptr)
    scale = sm_scale * k_scale * LOG2E
    start = s * chunk
    end = gl.minimum(start + chunk, seq)
    ntiles = gl.maximum(gl.cdiv(end - start, BN), 0)

    blk: gl.constexpr = gl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])
    k_sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2)
    v_sl: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    mma_s: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BN, 32])
    mma_o: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, D, 16])
    mma_n: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, NP, 32])

    rq = gl.arange(0, R, layout=gl.SliceLayout(1, blk))
    dq = gl.arange(0, D, layout=gl.SliceLayout(0, blk))
    tok = b * L + rq // G
    qh = h * G + rq % G
    qrow = qkv_ptr + tok.to(gl.int64)[:, None] * qkv_stride
    qx = gl.load(qrow + qh[:, None] * D + dq[None, :]).to(gl.float32)
    qinv = gl.rsqrt(gl.sum(qx * qx, axis=1) / D + eps)
    q8 = (qx * qinv[:, None]).to(gl.bfloat16).to(gl.float32).to(gl.float8e4nv)
    q_s = gl.allocate_shared_memory(gl.float8e4nv, [R, D], k_sl, q8)
    # The gate is read only by the last-arriving program, after every split has finished; issuing its load here
    # takes that DRAM round trip off the merge's serial tail.
    gate_pre = gl.load(qrow + (HQ + 2 * HKV) * D + qh[:, None] * D + dq[None, :])
    k_ring = gl.allocate_shared_memory(gl.float8e4nv, [STAGES, BN, D], k_sl)
    v_ring = gl.allocate_shared_memory(gl.float8e4nv, [STAGES, BN, D], k_sl)
    v16_s = gl.allocate_shared_memory(gl.float16, [BN, D], v_sl)
    full = gmbar.allocate_mbarrier(batch=STAGES)
    empty = gmbar.allocate_mbarrier(batch=STAGES)
    for i in gl.static_range(STAGES):
        gmbar.init(full.index(i), count=128)
        gmbar.init(empty.index(i), count=1)
    fence_async_shared()
    pt_row = pt_ptr + b.to(gl.int64) * pt_stride
    m, l, o = gl.warp_specialize(
        [(_gv_consumer, (q_s, k_ring, v_ring, v16_s, full, empty, start, end, ntiles, scale, BN, STAGES, R, D)),
         (_gv_producer, (pt_row, start, end, ntiles, k_cache, v_cache, k_ring, v_ring, full, empty, h, BN, STAGES,
                         D, ROW))],
        [4], [PR])

    if s == 0:
        jn = gl.arange(0, NP, layout=gl.SliceLayout(1, blk))
        jm = jn < L
        ntok = b * L + jn
        nrow = qkv_ptr + ntok.to(gl.int64)[:, None] * qkv_stride
        kx = gl.load(nrow + (HQ + h) * D + dq[None, :], mask=jm[:, None], other=0.0).to(gl.float32)
        kinv = gl.rsqrt(gl.sum(kx * kx, axis=1) / D + eps)
        k8 = ((kx * kinv[:, None]).to(gl.bfloat16).to(gl.float32) / k_scale).to(gl.float8e4nv)
        vx = gl.load(nrow + (HQ + HKV + h) * D + dq[None, :], mask=jm[:, None], other=0.0).to(gl.float32)
        v8 = (vx / v_scale).to(gl.float8e4nv)
        loc = gl.load(loc_ptr + ntok, mask=jm, other=0).to(gl.int64)
        coff = loc[:, None] * ROW + h * D + dq[None, :]
        gl.store(k_cache + coff, k8, mask=jm[:, None] & (dq[None, :] < D))
        gl.store(v_cache + coff, v8, mask=jm[:, None] & (dq[None, :] < D))
        kn_s = gl.allocate_shared_memory(gl.float8e4nv, [NP, D], k_sl, k8)
        vn_s = gl.allocate_shared_memory(gl.float16, [NP, D], v_sl, v8.to(gl.float16))
        fence_async_shared()
        rr = gl.arange(0, R, layout=gl.SliceLayout(1, mma_n))
        cc = gl.arange(0, NP, layout=gl.SliceLayout(0, mma_n))
        okn = (cc[None, :] < L) & (cc[None, :] <= (rr // G)[:, None])
        m2 = gl.convert_layout(m, gl.SliceLayout(1, mma_n))
        l2 = gl.convert_layout(l, gl.SliceLayout(1, mma_n))
        mn, ln, o = _gv_softmax_pv(q_s, kn_s, vn_s, okn, scale, m2, l2, o, mma_n, mma_o, R, NP)
        m = gl.convert_layout(mn, gl.SliceLayout(1, mma_s))
        l = gl.convert_layout(ln, gl.SliceLayout(1, mma_s))

    unit = b * HKV + h
    ro = gl.arange(0, R, layout=gl.SliceLayout(1, mma_o))
    do = gl.arange(0, D, layout=gl.SliceLayout(0, mma_o))
    pbase = (unit * NSPLIT + s) * R
    gl.store(part_o + (pbase + ro[:, None]) * D + do[None, :], o)
    r2 = gl.arange(0, R, layout=gl.SliceLayout(1, mma_s))
    gl.store(part_ml + (pbase + r2) * 2, m)
    gl.store(part_ml + (pbase + r2) * 2 + 1, l)

    gl.barrier()
    done = gl.atomic_add(cnt_ptr + unit, 1)
    if done == NSPLIT - 1:
        mL: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [1, 0])
        oL: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
        si = gl.arange(0, NSP2, layout=gl.SliceLayout(1, mL))
        rm = gl.arange(0, R, layout=gl.SliceLayout(0, mL))
        smask = (si[:, None] < NSPLIT) & (rm[None, :] < R)
        mptr = part_ml + ((unit * NSPLIT + si[:, None]) * R + rm[None, :]) * 2
        ms = gl.load(mptr, mask=smask, other=-float("inf"), cache_modifier=".cg")
        ls = gl.load(mptr + 1, mask=smask, other=0.0, cache_modifier=".cg")
        mx = gl.max(ms, axis=0)
        mx = gl.where(mx == -float("inf"), 0.0, mx)
        lsum = gl.sum(gl.exp2(ms - mx[None, :]) * ls, axis=0)
        r3 = gl.arange(0, R, layout=gl.SliceLayout(1, oL))
        d3 = gl.arange(0, D, layout=gl.SliceLayout(0, oL))
        mxo = gl.convert_layout(mx, gl.SliceLayout(1, oL))
        acc = gl.zeros([R, D], gl.float32, layout=oL)
        for k in range(0, NSPLIT):
            mj = gl.load(part_ml + ((unit * NSPLIT + k) * R + r3) * 2, cache_modifier=".cg")
            oj = gl.load(part_o + ((unit * NSPLIT + k) * R + r3[:, None]) * D + d3[None, :], cache_modifier=".cg")
            acc += gl.exp2(mj - mxo)[:, None] * oj
        lso = gl.convert_layout(lsum, gl.SliceLayout(1, oL))
        acc = (acc / lso[:, None] * v_scale).to(gl.bfloat16).to(gl.float32)
        acc = acc * gl.rsqrt(gl.sum(acc * acc, axis=1) / D + eps)[:, None]
        tok3 = b * L + r3 // G
        qh3 = h * G + r3 % G
        grow = qkv_ptr + tok3.to(gl.int64)[:, None] * qkv_stride
        gate = gl.convert_layout(gate_pre, oL).to(gl.float32)
        acc = acc * (1.0 / (1.0 + gl.exp(-gate)))
        gl.store(out_ptr + (tok3.to(gl.int64)[:, None] * HQ + qh3[:, None]) * D + d3[None, :], acc.to(gl.bfloat16))
        gl.atomic_xchg(cnt_ptr + unit, 0)


@gluon.jit
def _g_full_decode(qkv_ptr, loc_ptr, ks_ptr, vs_ptr, k_cache, v_cache, pt_ptr, seq_ptr, part_o, part_ml, cnt_ptr,
                   out_ptr, qkv_stride, pt_stride, chunk, sm_scale, eps,
                   HQ: gl.constexpr, HKV: gl.constexpr, D: gl.constexpr, NSPLIT: gl.constexpr,
                   NSP2: gl.constexpr, BN: gl.constexpr, STAGES: gl.constexpr, W: gl.constexpr):
    G: gl.constexpr = HQ // HKV
    ROW: gl.constexpr = HKV * D
    ROWS: gl.constexpr = HKV * G
    KW: gl.constexpr = BN // W
    LOG2E: gl.constexpr = 1.4426950408889634
    b = gl.program_id(0)
    s = gl.program_id(1)
    seq = gl.load(seq_ptr + b)
    k_scale = gl.load(ks_ptr)
    v_scale = gl.load(vs_ptr)
    scale = sm_scale * k_scale * LOG2E
    start = s * chunk
    end = gl.minimum(start + chunk, seq)
    ntiles = gl.maximum(gl.cdiv(end - start, BN), 0)

    ld3: gl.constexpr = gl.BlockedLayout([1, 1, 16], [1, 4, 8], [W, 1, 1], [2, 1, 0])
    sm3: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [2, 1, 0])
    mma3: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[W, 1, 1], instr_shape=[1, 16, 8])
    a8_l: gl.constexpr = gl.DotOperandLayout(0, mma3, 4)
    b8_l: gl.constexpr = gl.DotOperandLayout(1, mma3, 4)
    a16_l: gl.constexpr = gl.DotOperandLayout(0, mma3, 2)
    b16_l: gl.constexpr = gl.DotOperandLayout(1, mma3, 2)
    wg_l: gl.constexpr = gl.SliceLayout(1, mma3)
    wk_l: gl.constexpr = gl.SliceLayout(2, mma3)
    bl3: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 4, 8], [W, 1, 1], [2, 1, 0])

    k_smem = gl.allocate_shared_memory(gl.float8e4nv, [STAGES, W, KW, ROW], sm3)
    v_smem = gl.allocate_shared_memory(gl.float8e4nv, [STAGES, W, KW, ROW], sm3)
    qrow = qkv_ptr + b.to(gl.int64) * qkv_stride
    q0 = _g_head_q(qkv_ptr, qrow, 0, eps, G, D, W, bl3, b8_l)
    q1 = _g_head_q(qkv_ptr, qrow, 1, eps, G, D, W, bl3, b8_l)

    pt_row = pt_ptr + b.to(gl.int64) * pt_stride
    lw = gl.arange(0, W, layout=gl.SliceLayout(1, gl.SliceLayout(2, ld3)))
    lk = gl.arange(0, KW, layout=gl.SliceLayout(0, gl.SliceLayout(2, ld3)))
    kidx = lw[:, None] * KW + lk[None, :]
    for i in gl.static_range(STAGES - 1):
        n = start + i * BN + kidx
        nm = n < end
        rows = gl.load(pt_row + n, mask=nm, other=0)
        _g_issue(k_cache, v_cache, k_smem, v_smem, rows, nm, i, ROW, ld3)
    n_nx = start + (STAGES - 1) * BN + kidx
    rows_nx = gl.load(pt_row + n_nx, mask=n_nx < end, other=0)

    sw = gl.arange(0, W, layout=gl.SliceLayout(1, wk_l))
    sk = gl.arange(0, KW, layout=gl.SliceLayout(0, wk_l))
    sidx = sw[:, None] * KW + sk[None, :]
    m0 = gl.full([W, G], -float("inf"), gl.float32, layout=wg_l)
    m1 = gl.full([W, G], -float("inf"), gl.float32, layout=wg_l)
    l0 = gl.zeros([W, G], gl.float32, layout=wg_l)
    l1 = gl.zeros([W, G], gl.float32, layout=wg_l)
    o0 = gl.zeros([W, D, G], gl.float32, layout=mma3)
    o1 = gl.zeros([W, D, G], gl.float32, layout=mma3)
    for t in range(0, ntiles):
        slot = t % STAGES
        gcp.wait_group(STAGES - 2)
        gl.barrier()
        ok = (start + t * BN + sidx) < end
        m0, l0, o0 = _g_head_loop(k_smem, v_smem, slot, q0, ok, scale, m0, l0, o0, 0, D, W, KW, G, mma3, a8_l,
                                  a16_l, b16_l)
        m1, l1, o1 = _g_head_loop(k_smem, v_smem, slot, q1, ok, scale, m1, l1, o1, 1, D, W, KW, G, mma3, a8_l,
                                  a16_l, b16_l)
        tn = t + STAGES - 1
        n = start + tn * BN + kidx
        nm = n < end
        rows = rows_nx
        n2 = n + BN
        rows_nx = gl.load(pt_row + n2, mask=n2 < end, other=0)
        _g_issue(k_cache, v_cache, k_smem, v_smem, rows, nm, tn % STAGES, ROW, ld3)
    gcp.wait_group(0)

    loc = gl.load(loc_ptr + b).to(gl.int64)
    pbase = (b * NSPLIT + s) * ROWS
    _g_head_finish(qkv_ptr, qrow, loc, k_cache, v_cache, k_scale, v_scale, scale, eps, m0, l0, o0, part_o, part_ml,
                   pbase, s, 0, HQ, HKV, D, G, mma3)
    _g_head_finish(qkv_ptr, qrow, loc, k_cache, v_cache, k_scale, v_scale, scale, eps, m1, l1, o1, part_o, part_ml,
                   pbase, s, 1, HQ, HKV, D, G, mma3)

    gl.barrier()
    done = gl.atomic_add(cnt_ptr + b, 1)
    if done == NSPLIT - 1:
        mL: gl.constexpr = gl.BlockedLayout([1, 1], [2, 16], [W, 1], [1, 0])
        oL: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [W, 1], [1, 0])
        si = gl.arange(0, NSP2, layout=gl.SliceLayout(1, mL))
        rr = gl.arange(0, ROWS, layout=gl.SliceLayout(0, mL))
        smask = (si[:, None] < NSPLIT) & (rr[None, :] < ROWS)
        mptr = part_ml + ((b * NSPLIT + si[:, None]) * ROWS + rr[None, :]) * 2
        ms = gl.load(mptr, mask=smask, other=-float("inf"), cache_modifier=".cg")
        ls = gl.load(mptr + 1, mask=smask, other=0.0, cache_modifier=".cg")
        mx = gl.max(ms, axis=0)
        mx = gl.where(mx == -float("inf"), 0.0, mx)
        lsum = gl.sum(gl.exp2(ms - mx[None, :]) * ls, axis=0)
        ro = gl.arange(0, ROWS, layout=gl.SliceLayout(1, oL))
        do = gl.arange(0, D, layout=gl.SliceLayout(0, oL))
        mxo = gl.convert_layout(mx, gl.SliceLayout(1, oL))
        o = gl.zeros([ROWS, D], gl.float32, layout=oL)
        for k in range(0, NSPLIT):
            mj = gl.load(part_ml + ((b * NSPLIT + k) * ROWS + ro) * 2, cache_modifier=".cg")
            oj = gl.load(part_o + ((b * NSPLIT + k) * ROWS + ro[:, None]) * D + do[None, :], cache_modifier=".cg")
            o += gl.exp2(mj - mxo)[:, None] * oj
        lso = gl.convert_layout(lsum, gl.SliceLayout(1, oL))
        o = (o / lso[:, None] * v_scale).to(gl.bfloat16).to(gl.float32)
        o = o * gl.rsqrt(gl.sum(o * o, axis=1) / D + eps)[:, None]
        gate = gl.load(qrow + (HQ + 2 * HKV) * D + ro[:, None] * D + do[None, :]).to(gl.float32)
        o = o * (1.0 / (1.0 + gl.exp(-gate)))
        gl.store(out_ptr + (b.to(gl.int64) * HQ + ro[:, None]) * D + do[None, :], o.to(gl.bfloat16))
        gl.atomic_xchg(cnt_ptr + b, 0)


def _counter(n, device):
    """Per-unit arrival counters; each merge resets its own to zero (shape-sized, not data-derived)."""
    key = (device, n)
    if key not in _COUNTERS:
        _COUNTERS[key] = torch.zeros(n, dtype=torch.int32, device=device)
    return _COUNTERS[key]


# Fused-path launch shape per layer kind: keys per tile, warps, pipeline stages, most splits per unit.
# Tuned on an H200 (SM90, the judge's toolchain) by per-case sweeps against the previous shapes.


# the Gluon kernel's FUSED_CFG[('full', 1)] launch fields
_G_FC = dict(g_block_n=128, g_stages=3, g_warps=4, g_per_sm=4)

_G_VFC = dict(gv_block_n=128, gv_stages=4, gv_split=8, gv_pr=24, gv_maxnreg=128,
              big_gv_split=4, big_min_batch=32)


def _gluon_full_verify(data, cfg, qkv, tokens, fc):
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    page_table = data["page_table"]
    span = page_table.shape[1]
    bn = fc["gv_block_n"]
    gv_split = fc.get("big_gv_split", fc["gv_split"]) if batch >= fc.get("big_min_batch", 1 << 30) else fc["gv_split"]
    nsplit = max(1, min(gv_split, triton.cdiv(span, bn)))
    chunk = triton.cdiv(triton.cdiv(span, nsplit), bn) * bn
    nsplit = triton.cdiv(span, chunk)
    rows = length * (hq // hkv)
    part_o = torch.empty((batch * hkv, nsplit, rows, d), dtype=torch.float32, device=qkv.device)
    part_ml = torch.empty((batch * hkv, nsplit, rows, 2), dtype=torch.float32, device=qkv.device)
    out = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=qkv.device)
    _g_full_verify[(hkv, batch, nsplit)](
        qkv, data["cache_loc"], data["k_scale"], data["v_scale"], data["k_cache"], data["v_cache"], page_table,
        data["seq_lens"], part_o, part_ml, _counter(batch * hkv, qkv.device), out, qkv.stride(0), page_table.stride(0),
        chunk, cfg["sm_scale"] * cfg["qk_scale_factor"], RMS_EPS,
        HQ=hq, HKV=hkv, D=d, L=length, NSPLIT=nsplit, NSP2=triton.next_power_of_2(nsplit), BN=bn,
        STAGES=fc["gv_stages"], PR=fc["gv_pr"], num_warps=4, maxnreg=fc["gv_maxnreg"])
    return out.view(batch, length, hq * d)


def _gluon_full_decode(data, cfg, qkv, tokens, fc):
    batch, hq, hkv, d = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    page_table = data["page_table"]
    span = page_table.shape[1]
    bn = fc["g_block_n"]
    target = fc["g_per_sm"] * torch.cuda.get_device_properties(qkv.device).multi_processor_count
    nsplit = max(1, min(16, triton.cdiv(span, bn), triton.cdiv(target, batch)))
    chunk = triton.cdiv(triton.cdiv(span, nsplit), bn) * bn
    nsplit = triton.cdiv(span, chunk)
    rows = hq
    part_o = torch.empty((batch, nsplit, rows, d), dtype=torch.float32, device=qkv.device)
    part_ml = torch.empty((batch, nsplit, rows, 2), dtype=torch.float32, device=qkv.device)
    out = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=qkv.device)
    _g_full_decode[(batch, nsplit)](
        qkv, data["cache_loc"], data["k_scale"], data["v_scale"], data["k_cache"], data["v_cache"], page_table,
        data["seq_lens"], part_o, part_ml, _counter(batch, qkv.device), out, qkv.stride(0), page_table.stride(0),
        chunk, cfg["sm_scale"] * cfg["qk_scale_factor"], RMS_EPS,
        HQ=hq, HKV=hkv, D=d, NSPLIT=nsplit, NSP2=triton.next_power_of_2(nsplit), BN=bn, STAGES=fc["g_stages"],
        W=fc["g_warps"], num_warps=fc["g_warps"])
    return out.view(batch, 1, hq * d)


# ---- fused attention for sliding and full layers (adapted from the muse_attention_decode kernels) ----
MERGE_ELEMS = tl.constexpr(16384)   # partial elements loaded per step of the last-arrival merge (raised from 4096 to 16384 on the H200)

@triton.jit
def _combine_ml(part_o, part_ml, first, count, r, dd, ROWS: tl.constexpr, D: tl.constexpr, NP2: tl.constexpr):
    """Exact log-sum-exp merge of `count` partials starting at index `first`: (unnormalized o, l, m)."""
    SB1: tl.constexpr = MERGE_ELEMS // (ROWS * D) if MERGE_ELEMS // (ROWS * D) > 0 else 1
    if SB1 >= NP2:
        # Every partial fits in one step: issue the m, l and o loads together (one L2 round trip).
        sa = tl.arange(0, NP2)
        sam = sa < count
        mptr1 = part_ml + ((first + sa[:, None]) * ROWS + r[None, :]) * 2
        ma = tl.load(mptr1, mask=sam[:, None], other=-float("inf"), cache_modifier=".cg")
        la = tl.load(mptr1 + 1, mask=sam[:, None], other=0.0, cache_modifier=".cg")
        oa = tl.load(part_o + ((first + sa[:, None, None]) * ROWS + r[None, :, None]) * D + dd[None, None, :],
                     mask=sam[:, None, None], other=0.0, cache_modifier=".cg")
        mx1 = tl.max(ma, 0)
        mx1 = tl.where(mx1 == -float("inf"), 0.0, mx1)
        wa = tl.exp2(ma - mx1[None, :])
        return tl.sum(wa[:, :, None] * oa, 0), tl.sum(wa * la, 0), mx1
    si = tl.arange(0, NP2)
    smask = si[:, None] < count
    mptr = part_ml + ((first + si[:, None]) * ROWS + r[None, :]) * 2
    ms = tl.load(mptr, mask=smask, other=-float("inf"), cache_modifier=".cg")
    ls = tl.load(mptr + 1, mask=smask, other=0.0, cache_modifier=".cg")
    mx = tl.max(ms, 0)
    mx = tl.where(mx == -float("inf"), 0.0, mx)
    lsum = tl.sum(tl.exp2(ms - mx[None, :]) * ls, 0)
    o = tl.zeros([ROWS, D], tl.float32)
    SB: tl.constexpr = MERGE_ELEMS // (ROWS * D) if MERGE_ELEMS // (ROWS * D) > 0 else 1
    sb = tl.arange(0, SB)
    for k0 in range(0, count, SB):
        sj = k0 + sb
        sjm = sj < count
        mj = tl.load(part_ml + ((first + sj[:, None]) * ROWS + r[None, :]) * 2,
                     mask=sjm[:, None], other=-float("inf"), cache_modifier=".cg")
        oj = tl.load(part_o + ((first + sj[:, None, None]) * ROWS + r[None, :, None]) * D + dd[None, None, :],
                     mask=sjm[:, None, None], other=0.0, cache_modifier=".cg")
        o += tl.sum(tl.exp2(mj - mx[None, :])[:, :, None] * oj, 0)
    return o, lsum, mx

@triton.jit
def _combine(part_o, part_ml, first, count, r, dd, ROWS: tl.constexpr, D: tl.constexpr, NP2: tl.constexpr):
    o, lsum, mx = _combine_ml(part_o, part_ml, first, count, r, dd, ROWS, D, NP2)
    return o, lsum

@triton.jit
def _emit_pre(o, lsum, v_scale, eps, gate, out_ptr, tok, qh, rvalid, dd, HQ: tl.constexpr, D: tl.constexpr):
    """_emit with the gate already in registers (loaded at kernel start, off the merge's critical path)."""
    o = (o / lsum[:, None] * v_scale).to(tl.bfloat16).to(tl.float32)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + eps)[:, None]
    o = o * tl.sigmoid(gate.to(tl.float32))
    tl.store(out_ptr + (tok.to(tl.int64)[:, None] * HQ + qh[:, None]) * D + dd[None, :], o.to(tl.bfloat16),
             mask=rvalid[:, None])

@triton.jit
def _emit(o, lsum, v_scale, eps, qkv_ptr, out_ptr, tok, qh, rvalid, dd, qkv_stride,
          HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr):
    """Normalize the merged attention, round to BF16 as FA3's output is, then RMSNorm x sigmoid(gate)."""
    o = (o / lsum[:, None] * v_scale).to(tl.bfloat16).to(tl.float32)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + eps)[:, None]
    gptr = qkv_ptr + tok.to(tl.int64)[:, None] * qkv_stride + (HQ + 2 * HKV) * D + qh[:, None] * D + dd[None, :]
    gate = tl.load(gptr, mask=rvalid[:, None], other=0.0).to(tl.float32)
    o = o * tl.sigmoid(gate)
    tl.store(out_ptr + (tok.to(tl.int64)[:, None] * HQ + qh[:, None]) * D + dd[None, :], o.to(tl.bfloat16),
             mask=rvalid[:, None])

@triton.jit
def _fused_attn(qkv_ptr, loc_ptr, cos_sin_ptr, ks_ptr, vs_ptr, k_cache, v_cache, pt_ptr, seq_ptr,
                part_o, part_ml, part2_o, part2_ml, cnt_ptr, out_ptr,
                qkv_stride, pt_stride, chunk, sm_scale, eps,
                HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, L: tl.constexpr, HPB: tl.constexpr,
                ROWS: tl.constexpr, LP: tl.constexpr, PAGE: tl.constexpr, BLOCK_N: tl.constexpr,
                NSPLIT: tl.constexpr, NSP2: tl.constexpr, SB: tl.constexpr, WINDOW: tl.constexpr,
                ROPE: tl.constexpr, GS: tl.constexpr, NG: tl.constexpr, NG2: tl.constexpr,
                PREFETCH: tl.constexpr, NRB: tl.constexpr, QK8: tl.constexpr, GPRE: tl.constexpr,
                USE_PDL: tl.constexpr = False):
    if USE_PDL:  # wait for the producer grid, then let the next kernel launch
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    b = tl.program_id(0)
    hblk = tl.program_id(1) // NRB             # block of HPB KV heads
    rb = tl.program_id(1) % NRB                # block of ROWS query rows within it
    s = tl.program_id(2)                       # split of the cached keys
    G: tl.constexpr = HQ // HKV
    NHB: tl.constexpr = HKV // HPB
    DB: tl.constexpr = HPB * D                 # columns of one block: HPB heads side by side
    HALF: tl.constexpr = D // 2
    LOG2E: tl.constexpr = 1.4426950408889634
    seq = tl.load(seq_ptr + b)                 # cached tokens; the block's tokens sit at seq .. seq+L-1
    k_scale = tl.load(ks_ptr)
    v_scale = tl.load(vs_ptr)
    scale = sm_scale * k_scale * LOG2E

    # Query rows r = (head in block, token, group); block-diagonal columns (zero off the row's head).
    r = tl.arange(0, ROWS)
    rg = rb * ROWS + r                         # row index within the head block
    hb = rg // (L * G)
    j = (rg // G) % L
    g = rg % G
    rvalid = rg < HPB * L * G
    qh = (hblk * HPB + hb) * G + g
    tok = b * L + j
    c2 = tl.arange(0, DB // 2)                 # RoPE pair index across the block
    chb = c2 // HALF
    ci = c2 % HALF
    qmask = rvalid[:, None] & (chb[None, :] == hb[:, None])
    qptr = qkv_ptr + tok.to(tl.int64)[:, None] * qkv_stride + qh[:, None] * D + 2 * ci[None, :]
    qe = tl.load(qptr, mask=qmask, other=0.0).to(tl.float32)
    qo = tl.load(qptr + 1, mask=qmask, other=0.0).to(tl.float32)
    inv = tl.rsqrt((tl.sum(qe * qe, 1) + tl.sum(qo * qo, 1)) / D + eps)[:, None]
    qe = (qe * inv).to(tl.bfloat16).to(tl.float32)
    qo = (qo * inv).to(tl.bfloat16).to(tl.float32)
    qpos = seq + j
    if ROPE:
        cs = cos_sin_ptr + qpos.to(tl.int64)[:, None] * D + ci[None, :]
        cq = tl.load(cs, mask=qmask, other=0.0)
        sq = tl.load(cs + HALF, mask=qmask, other=0.0)
        qe, qo = (qe * cq - qo * sq).to(tl.bfloat16).to(tl.float32), (qe * sq + qo * cq).to(tl.bfloat16).to(tl.float32)
    # Queries rounded to E4M3, as FA3 does.
    q8 = tl.reshape(tl.join(qe, qo), [ROWS, DB]).to(tl.float8e4nv)
    q = q8.to(tl.float16)
    # The gate is needed only by the merging program, but loading it here takes its DRAM round trip off the
    # merge's critical path (every program loads it; 2 KB).
    if GPRE:
        dgt = tl.arange(0, D)
        gate_pre = tl.load(qkv_ptr + tok.to(tl.int64)[:, None] * qkv_stride + (HQ + 2 * HKV) * D + qh[:, None] * D
                           + dgt[None, :], mask=rvalid[:, None], other=0.0)

    dcol = tl.arange(0, DB)
    m_i = tl.full([ROWS], -float("inf"), tl.float32)
    l_i = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, DB], tl.float32)

    # Cached keys of this split: [lo, seq) on full layers, the window's cached part on sliding ones.
    if WINDOW > 0:
        lo = tl.maximum(seq - WINDOW + 1, 0)
    else:
        lo = 0
    start = lo + s * chunk
    end = tl.minimum(start + chunk, seq)
    pt_row = pt_ptr + b.to(tl.int64) * pt_stride
    n = start + tl.arange(0, BLOCK_N)
    if PREFETCH:
        page = tl.load(pt_row + n // PAGE, mask=n < end, other=0)
    else:
        page = n
    for n0 in range(start, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        nm = n < end
        if PREFETCH:
            # The next tile's page indices load while this tile's rows are fetched and used.
            nn = n + BLOCK_N
            page_next = tl.load(pt_row + nn // PAGE, mask=nn < end, other=0)
            row = page.to(tl.int64) * PAGE + n % PAGE
            page = page_next
        else:
            row = tl.load(pt_row + n // PAGE, mask=nm, other=0).to(tl.int64) * PAGE + n % PAGE
        off = row[:, None] * (HKV * D) + hblk * DB + dcol[None, :]
        if QK8:
            k = tl.load(k_cache + off, mask=nm[:, None], other=0.0)
        else:
            k = tl.load(k_cache + off, mask=nm[:, None], other=0.0).to(tl.float16)
        v = tl.load(v_cache + off, mask=nm[:, None], other=0.0).to(tl.float16)
        if QK8:
            # E4M3 x E4M3 products are exact in the fp32 accumulator; q is already E4M3-rounded, as FA3 does.
            sc = tl.dot(q8, tl.trans(k), out_dtype=tl.float32) * scale
        else:
            sc = tl.dot(q, tl.trans(k)) * scale
        ok = nm[None, :]
        if WINDOW > 0:
            ok = ok & (n[None, :] > qpos[:, None] - WINDOW)
        sc = tl.where(ok, sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_use = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp2(sc - m_use[:, None])
        alpha = tl.exp2(m_i - m_use)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new

    if s == 0:
        # The block's new keys and values: normalized (+ RoPE), stored to the cache as E4M3, and attended
        # causally from registers (other programs read only rows below seq).
        jn = tl.arange(0, LP)
        jm = jn < L
        ntok = b * L + jn
        kptr = qkv_ptr + ntok.to(tl.int64)[:, None] * qkv_stride + (HQ + hblk * HPB) * D + 2 * c2[None, :]
        ke = tl.load(kptr, mask=jm[:, None], other=0.0).to(tl.float32)
        ko = tl.load(kptr + 1, mask=jm[:, None], other=0.0).to(tl.float32)
        ke3 = tl.reshape(ke, [LP, HPB, HALF])
        ko3 = tl.reshape(ko, [LP, HPB, HALF])
        kinv = tl.rsqrt((tl.sum(ke3 * ke3, 2) + tl.sum(ko3 * ko3, 2)) / D + eps)[:, :, None]
        ke = tl.reshape((ke3 * kinv).to(tl.bfloat16).to(tl.float32), [LP, DB // 2])
        ko = tl.reshape((ko3 * kinv).to(tl.bfloat16).to(tl.float32), [LP, DB // 2])
        if ROPE:
            npos = seq + jn
            csk = cos_sin_ptr + npos.to(tl.int64)[:, None] * D + ci[None, :]
            ck = tl.load(csk, mask=jm[:, None], other=0.0)
            sk = tl.load(csk + HALF, mask=jm[:, None], other=0.0)
            ke, ko = (ke * ck - ko * sk).to(tl.bfloat16).to(tl.float32), (ke * sk + ko * ck).to(tl.bfloat16).to(tl.float32)
        k8 = (tl.reshape(tl.join(ke, ko), [LP, DB]) / k_scale).to(tl.float8e4nv)
        vraw = tl.load(qkv_ptr + ntok.to(tl.int64)[:, None] * qkv_stride + (HQ + HKV + hblk * HPB) * D + dcol[None, :],
                       mask=jm[:, None], other=0.0).to(tl.float32)
        v8 = (vraw / v_scale).to(tl.float8e4nv)
        loc = tl.load(loc_ptr + ntok, mask=jm, other=0)
        coff = loc[:, None] * (HKV * D) + hblk * DB + dcol[None, :]
        tl.store(k_cache + coff, k8, mask=jm[:, None] & (rb == 0))
        tl.store(v_cache + coff, v8, mask=jm[:, None] & (rb == 0))
        sc = tl.dot(q, tl.trans(k8.to(tl.float16))) * scale
        sc = tl.where(jm[None, :] & (jn[None, :] <= j[:, None]), sc, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        p = tl.exp2(sc - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v8.to(tl.float16))
        m_i = m_new

    # This split's partial: each row keeps only its own head's columns.
    dd = tl.arange(0, D)
    if HPB == 1:
        o_part = acc
    else:
        acc3 = tl.reshape(acc, [ROWS, HPB, D])
        sel = tl.arange(0, HPB)[None, :, None] == hb[:, None, None]
        o_part = tl.sum(tl.where(sel, acc3, 0.0), 1)
    unit = ((b * NHB + hblk) * NRB + rb).to(tl.int64)
    pbase = (unit * NSPLIT + s) * ROWS
    tl.store(part_o + (pbase + r[:, None]) * D + dd[None, :], o_part)
    tl.store(part_ml + (pbase + r) * 2, m_i)
    tl.store(part_ml + (pbase + r) * 2 + 1, l_i)

    tl.debug_barrier()
    if NG == 1:
        done = tl.atomic_add(cnt_ptr + unit, 1)
        if done == NSPLIT - 1:
            # Last program of this unit: exact merge of every split, then RMSNorm x sigmoid(gate).
            o, lsum = _combine(part_o, part_ml, unit * NSPLIT, NSPLIT, r, dd, ROWS, D, NSP2)
            if GPRE:
                _emit_pre(o, lsum, v_scale, eps, gate_pre, out_ptr, tok, qh, rvalid, dd, HQ, D)
            else:
                _emit(o, lsum, v_scale, eps, qkv_ptr, out_ptr, tok, qh, rvalid, dd, qkv_stride, HQ, HKV, D)
            tl.atomic_xchg(cnt_ptr + unit, 0)
    else:
        # Two levels: the last program of each group of GS splits merges the group; the last group
        # merger merges the groups and writes the output.
        grp = s // GS
        gsize = tl.minimum(GS, NSPLIT - grp * GS)
        gcnt = cnt_ptr + unit * (NG + 1) + grp
        done = tl.atomic_add(gcnt, 1)
        if done == gsize - 1:
            o, lsum, mx = _combine_ml(part_o, part_ml, unit * NSPLIT + grp * GS, gsize, r, dd, ROWS, D, GS)
            gbase = (unit * NG + grp) * ROWS
            tl.store(part2_o + (gbase + r[:, None]) * D + dd[None, :], o)
            tl.store(part2_ml + (gbase + r) * 2, mx)
            tl.store(part2_ml + (gbase + r) * 2 + 1, lsum)
            tl.atomic_xchg(gcnt, 0)
            tl.debug_barrier()
            tcnt = cnt_ptr + unit * (NG + 1) + NG
            done2 = tl.atomic_add(tcnt, 1)
            if done2 == NG - 1:
                o, lsum = _combine(part2_o, part2_ml, unit * NG, NG, r, dd, ROWS, D, NG2)
                if GPRE:
                    _emit_pre(o, lsum, v_scale, eps, gate_pre, out_ptr, tok, qh, rvalid, dd, HQ, D)
                else:
                    _emit(o, lsum, v_scale, eps, qkv_ptr, out_ptr, tok, qh, rvalid, dd, qkv_stride, HQ, HKV, D)
                tl.atomic_xchg(tcnt, 0)

@triton.jit
def _qk_rows(qkv_ptr, rowoff, cos_sin_ptr, pos, mask, eps, D: tl.constexpr, ROPE: tl.constexpr):
    """RMSNorm (+ interleaved RoPE) of [R, D] rows at qkv_ptr + rowoff[:, None]; BF16-rounded, fp32 [R, D]."""
    HALF: tl.constexpr = D // 2
    i = tl.arange(0, HALF)[None, :]
    xe = tl.load(qkv_ptr + rowoff[:, None] + 2 * i, mask=mask[:, None], other=0.0).to(tl.float32)
    xo = tl.load(qkv_ptr + rowoff[:, None] + 2 * i + 1, mask=mask[:, None], other=0.0).to(tl.float32)
    inv = tl.rsqrt((tl.sum(xe * xe, 1) + tl.sum(xo * xo, 1)) / D + eps)[:, None]
    xe = (xe * inv).to(tl.bfloat16).to(tl.float32)
    xo = (xo * inv).to(tl.bfloat16).to(tl.float32)
    if ROPE:
        cs = cos_sin_ptr + pos.to(tl.int64)[:, None] * D + i
        c = tl.load(cs, mask=mask[:, None], other=0.0)
        sn = tl.load(cs + HALF, mask=mask[:, None], other=0.0)
        xe, xo = (xe * c - xo * sn).to(tl.bfloat16).to(tl.float32), (xe * sn + xo * c).to(tl.bfloat16).to(tl.float32)
    return tl.reshape(tl.join(xe, xo), [mask.shape[0], D])

@triton.jit
def _online(sc, v, m_i, l_i, acc):
    m_new = tl.maximum(m_i, tl.max(sc, 1))
    m_use = tl.where(m_new == -float("inf"), 0.0, m_new)
    p = tl.exp2(sc - m_use[:, None])
    alpha = tl.exp2(m_i - m_use)
    return m_new, l_i * alpha + tl.sum(p, 1), acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)

@triton.jit
def _fused_attn_pair(qkv_ptr, loc_ptr, cos_sin_ptr, ks_ptr, vs_ptr, k_cache, v_cache, pt_ptr, seq_ptr,
                     part_o, part_ml, cnt_ptr, out_ptr,
                     qkv_stride, pt_stride, chunk, sm_scale, eps,
                     HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                     RH: tl.constexpr, LP: tl.constexpr, PAGE: tl.constexpr, BLOCK_N: tl.constexpr,
                     NSPLIT: tl.constexpr, NSP2: tl.constexpr, SB: tl.constexpr, WINDOW: tl.constexpr,
                     ROPE: tl.constexpr):
    """Verify blocks: one program serves a pair of KV heads (RH query rows each), loading both 64-byte
    halves of each cache row back to back; otherwise as _fused_attn."""
    b = tl.program_id(0)
    pr = tl.program_id(1)                      # KV heads 2pr, 2pr+1
    s = tl.program_id(2)
    G: tl.constexpr = HQ // HKV
    NHB: tl.constexpr = HKV // 2
    ROWS: tl.constexpr = 2 * RH
    LOG2E: tl.constexpr = 1.4426950408889634
    seq = tl.load(seq_ptr + b)
    k_scale = tl.load(ks_ptr)
    v_scale = tl.load(vs_ptr)
    scale = sm_scale * k_scale * LOG2E

    r = tl.arange(0, RH)                       # r = token * G + group
    j = r // G
    g = r % G
    rvalid = r < L * G
    tok = b * L + j
    qpos = seq + j
    base_q = tok.to(tl.int64) * qkv_stride
    q0 = _qk_rows(qkv_ptr, base_q + ((2 * pr) * G + g) * D, cos_sin_ptr, qpos, rvalid, eps, D, ROPE)
    q1 = _qk_rows(qkv_ptr, base_q + ((2 * pr + 1) * G + g) * D, cos_sin_ptr, qpos, rvalid, eps, D, ROPE)
    q0 = q0.to(tl.float8e4nv).to(tl.float16)
    q1 = q1.to(tl.float8e4nv).to(tl.float16)

    dh = tl.arange(0, D)
    m0 = tl.full([RH], -float("inf"), tl.float32)
    m1 = tl.full([RH], -float("inf"), tl.float32)
    l0 = tl.zeros([RH], tl.float32)
    l1 = tl.zeros([RH], tl.float32)
    a0 = tl.zeros([RH, D], tl.float32)
    a1 = tl.zeros([RH, D], tl.float32)
    if WINDOW > 0:
        lo = tl.maximum(seq - WINDOW + 1, 0)
    else:
        lo = 0
    start = lo + s * chunk
    end = tl.minimum(start + chunk, seq)
    for n0 in range(start, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        nm = n < end
        page = tl.load(pt_ptr + b.to(tl.int64) * pt_stride + n // PAGE, mask=nm, other=0)
        row = page.to(tl.int64) * PAGE + n % PAGE
        off = row[:, None] * (HKV * D) + pr * (2 * D) + dh[None, :]
        k0 = tl.load(k_cache + off, mask=nm[:, None], other=0.0).to(tl.float16)
        k1 = tl.load(k_cache + off + D, mask=nm[:, None], other=0.0).to(tl.float16)
        v0 = tl.load(v_cache + off, mask=nm[:, None], other=0.0).to(tl.float16)
        v1 = tl.load(v_cache + off + D, mask=nm[:, None], other=0.0).to(tl.float16)
        ok = nm[None, :]
        if WINDOW > 0:
            ok = ok & (n[None, :] > qpos[:, None] - WINDOW)
        s0 = tl.where(ok, tl.dot(q0, tl.trans(k0)) * scale, -float("inf"))
        s1 = tl.where(ok, tl.dot(q1, tl.trans(k1)) * scale, -float("inf"))
        m0, l0, a0 = _online(s0, v0, m0, l0, a0)
        m1, l1, a1 = _online(s1, v1, m1, l1, a1)

    if s == 0:
        jn = tl.arange(0, LP)
        jm = jn < L
        ntok = b * L + jn
        nbase = ntok.to(tl.int64) * qkv_stride
        kn0 = _qk_rows(qkv_ptr, nbase + (HQ + 2 * pr) * D, cos_sin_ptr, seq + jn, jm, eps, D, ROPE)
        kn1 = _qk_rows(qkv_ptr, nbase + (HQ + 2 * pr + 1) * D, cos_sin_ptr, seq + jn, jm, eps, D, ROPE)
        k80 = (kn0 / k_scale).to(tl.float8e4nv)
        k81 = (kn1 / k_scale).to(tl.float8e4nv)
        dd = tl.arange(0, D)
        vbase = qkv_ptr + nbase[:, None] + (HQ + HKV + 2 * pr) * D + dd[None, :]
        v80 = (tl.load(vbase, mask=jm[:, None], other=0.0).to(tl.float32) / v_scale).to(tl.float8e4nv)
        v81 = (tl.load(vbase + D, mask=jm[:, None], other=0.0).to(tl.float32) / v_scale).to(tl.float8e4nv)
        loc = tl.load(loc_ptr + ntok, mask=jm, other=0)
        coff = loc[:, None] * (HKV * D) + pr * (2 * D) + dd[None, :]
        tl.store(k_cache + coff, k80, mask=jm[:, None])
        tl.store(k_cache + coff + D, k81, mask=jm[:, None])
        tl.store(v_cache + coff, v80, mask=jm[:, None])
        tl.store(v_cache + coff + D, v81, mask=jm[:, None])
        okn = jm[None, :] & (jn[None, :] <= j[:, None])
        s0 = tl.where(okn, tl.dot(q0, tl.trans(k80.to(tl.float16))) * scale, -float("inf"))
        s1 = tl.where(okn, tl.dot(q1, tl.trans(k81.to(tl.float16))) * scale, -float("inf"))
        m0, l0, a0 = _online(s0, v80.to(tl.float16), m0, l0, a0)
        m1, l1, a1 = _online(s1, v81.to(tl.float16), m1, l1, a1)

    dd = tl.arange(0, D)
    unit = (b * NHB + pr).to(tl.int64)
    pbase = (unit * NSPLIT + s) * ROWS
    tl.store(part_o + (pbase + r[:, None]) * D + dd[None, :], a0)
    tl.store(part_o + (pbase + RH + r[:, None]) * D + dd[None, :], a1)
    tl.store(part_ml + (pbase + r) * 2, m0)
    tl.store(part_ml + (pbase + r) * 2 + 1, l0)
    tl.store(part_ml + (pbase + RH + r) * 2, m1)
    tl.store(part_ml + (pbase + RH + r) * 2 + 1, l1)

    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr + unit, 1)
    if done == NSPLIT - 1:
        rr = tl.arange(0, ROWS)                # rr = head-in-pair * RH + r
        hb = rr // RH
        jr = (rr % RH) // G
        gr = rr % G
        vr = (rr % RH) < L * G
        qh = (2 * pr + hb) * G + gr
        tokr = b * L + jr
        si = tl.arange(0, NSP2)
        mptr = part_ml + ((unit * NSPLIT + si[:, None]) * ROWS + rr[None, :]) * 2
        smask = si[:, None] < NSPLIT
        ms = tl.load(mptr, mask=smask, other=-float("inf"), cache_modifier=".cg")
        ls = tl.load(mptr + 1, mask=smask, other=0.0, cache_modifier=".cg")
        mx = tl.max(ms, 0)
        mx = tl.where(mx == -float("inf"), 0.0, mx)
        lsum = tl.sum(tl.exp2(ms - mx[None, :]) * ls, 0)
        o = tl.zeros([ROWS, D], tl.float32)
        for sj in range(0, NSPLIT):
            mj = tl.load(part_ml + ((unit * NSPLIT + sj) * ROWS + rr) * 2, cache_modifier=".cg")
            oj = tl.load(part_o + ((unit * NSPLIT + sj) * ROWS + rr[:, None]) * D + dd[None, :], cache_modifier=".cg")
            o += tl.exp2(mj - mx)[:, None] * oj
        o = (o / lsum[:, None] * v_scale).to(tl.bfloat16).to(tl.float32)
        o = o * tl.rsqrt(tl.sum(o * o, 1) / D + eps)[:, None]
        gptr = qkv_ptr + tokr.to(tl.int64)[:, None] * qkv_stride + (HQ + 2 * HKV) * D + qh[:, None] * D + dd[None, :]
        gate = tl.load(gptr, mask=vr[:, None], other=0.0).to(tl.float32)
        o = o * tl.sigmoid(gate)
        tl.store(out_ptr + (tokr.to(tl.int64)[:, None] * HQ + qh[:, None]) * D + dd[None, :], o.to(tl.bfloat16),
                 mask=vr[:, None])
        tl.atomic_xchg(cnt_ptr + unit, 0)

FUSED_CFG = {
    ("sliding", 1): dict(block_n=128, warps=4, stages=2, max_split=16, per_sm=4, qk8=False,
                         small=dict(warps=8, max_split=32, per_sm=64, prefetch=True),
                         big=dict(max_split=8, stages=3), big_min_batch=32),
    ("sliding", 8): dict(block_n=128, warps=4, stages=3, max_split=8, per_sm=4, row_blocks=1,
                         small=dict(block_n=128, stages=2, max_split=16, row_blocks=4, qk8=False, prefetch=True), big=dict(max_split=4), big_min_batch=32),
    ("full", 1): dict(block_n=128, warps=4, stages=5, max_split=128, per_sm=2, prefetch=False, min_chunk=256,
                      small=dict(hpb=1, per_sm=4, stages=3, prefetch=True, gluon=False, gate_pre=False), big=dict(per_sm=4),
                      big_min_batch=32, gluon=True, g_block_n=128, g_stages=3, g_warps=4, g_per_sm=4),
    ("full", 8): dict(block_n=64, warps=4, stages=3, max_split=64, per_sm=2, pair_min_batch=16, prefetch=True,
                      min_chunk=256),
}

def _fused(data, cfg, qkv, tokens, cos_sin):
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    window = cfg["window"]
    kind = "sliding" if window else "full"
    fc = FUSED_CFG[(kind, 1 if length == 1 else 8)]
    if batch <= 4 and "small" in fc:   # shape-only choice: small batches favour more, smaller programs
        fc = {**fc, **fc["small"]}
    if batch >= fc.get("big_min_batch", 1 << 30) and "big" in fc:
        fc = {**fc, **fc["big"]}
    if (fc.get("gluon") and length == 1 and not window and page == 1 and hq == 16 and hkv == 2 and d == 64
            and torch.cuda.get_device_capability(qkv.device)[0] == 9):   # shape and device-capability choice
        return _gluon_full_decode(data, cfg, qkv, tokens, fc)
    group = hq // hkv
    pair = (fc.get("pair", False) or batch >= fc.get("pair_min_batch", 1 << 30)) and length > 1 and hkv % 2 == 0
    hpb = fc.get("hpb", 2) if ((length == 1 or pair) and hkv % 2 == 0) else 1
    rh = max(16, triton.next_power_of_2(length * group))
    rows = 2 * rh if pair else max(16, triton.next_power_of_2(hpb * length * group))
    nrb = 1 if pair else max(1, min(fc.get("row_blocks", 1), rows // 16))
    rows = rows // nrb
    page_table = data["page_table"]
    span = (window - 1) if window else page_table.shape[1] * page   # cached keys a request can attend
    units = batch * (hkv // hpb) * nrb
    target = fc["per_sm"] * torch.cuda.get_device_properties(qkv.device).multi_processor_count
    nsplit = max(1, min(fc["max_split"], triton.cdiv(span, max(fc["block_n"], fc.get("min_chunk", 1))),
                        triton.cdiv(target, units)))
    chunk = triton.cdiv(triton.cdiv(span, nsplit), fc["block_n"]) * fc["block_n"]
    nsplit = triton.cdiv(span, chunk)
    part_o = torch.empty((units, nsplit, rows, d), dtype=torch.float32, device=qkv.device)
    part_ml = torch.empty((units, nsplit, rows, 2), dtype=torch.float32, device=qkv.device)
    out = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=qkv.device)
    sb = max(1, min(triton.next_power_of_2(nsplit), 4096 // (rows * d)))
    gs = 16
    ng = triton.cdiv(nsplit, gs) if nsplit > gs else 1
    part2_o = torch.empty((units, ng, rows, d), dtype=torch.float32, device=qkv.device) if ng > 1 else part_o
    part2_ml = torch.empty((units, ng, rows, 2), dtype=torch.float32, device=qkv.device) if ng > 1 else part_ml
    if pair:
        _fused_attn_pair[(batch, hkv // 2, nsplit)](
            qkv, data["cache_loc"], cos_sin, data["k_scale"], data["v_scale"], data["k_cache"], data["v_cache"],
            page_table, data["seq_lens"], part_o, part_ml, _counter(units, qkv.device), out,
            qkv.stride(0), page_table.stride(0), chunk, cfg["sm_scale"] * cfg["qk_scale_factor"], RMS_EPS,
            HQ=hq, HKV=hkv, D=d, L=length, RH=rh, LP=max(16, triton.next_power_of_2(length)), PAGE=page,
            BLOCK_N=fc["block_n"], NSPLIT=nsplit, NSP2=triton.next_power_of_2(nsplit), SB=sb, WINDOW=window,
            ROPE=bool(cfg["rope_theta"]), num_warps=fc["warps"], num_stages=fc["stages"])
        return out.view(batch, length, hq * d)
    _fused_attn[(batch, hkv // hpb * nrb, nsplit)](
        qkv, data["cache_loc"], cos_sin, data["k_scale"], data["v_scale"], data["k_cache"], data["v_cache"],
        page_table, data["seq_lens"], part_o, part_ml, part2_o, part2_ml, _counter(units * (ng + 1), qkv.device), out,
        qkv.stride(0), page_table.stride(0), chunk, cfg["sm_scale"] * cfg["qk_scale_factor"], RMS_EPS,
        HQ=hq, HKV=hkv, D=d, L=length, HPB=hpb, ROWS=rows, LP=max(16, triton.next_power_of_2(length)), PAGE=page,
        BLOCK_N=fc["block_n"], NSPLIT=nsplit, NSP2=triton.next_power_of_2(nsplit), SB=sb, WINDOW=window,
        ROPE=bool(cfg["rope_theta"]), GS=gs, NG=ng, NG2=triton.next_power_of_2(ng), PREFETCH=fc.get("prefetch", False), NRB=nrb, QK8=fc.get("qk8", True), GPRE=fc.get("gate_pre", True),
        num_warps=fc["warps"], num_stages=fc["stages"], **(_PDL_KW if batch == 1 else {}))  # PDL at batch 1 (shape-only)
    return out.view(batch, length, hq * d)


# ---- one-token (DIRECT-mode) grouped NVFP4 W13/W2 kernels (as in the muse_moe_w4a16 solution), renamed for this layer ----
@gluon.jit
def _w13_silu_d1(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr,
              K: gl.constexpr, I: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr,
              XS: gl.constexpr, SMB: gl.constexpr = False, HOP: gl.constexpr = False,
              CHK: gl.constexpr = False, PF: gl.constexpr = False,
              DIRECT: gl.constexpr = False, NLOC: gl.constexpr = 32):
    mma: gl.constexpr = (MMA3_8 if BM == 8 else MMA3_16) if HOP else MMA_L
    a_l: gl.constexpr = (A3_8 if BM == 8 else A3_16) if HOP else A_L
    pid_n = gl.program_id(0)
    pid_w = gl.program_id(1)
    if DIRECT:  # M = 1: slot pid_w of the only token is one item of one row; no routing pass
        e = gl.load(work_ptr + pid_w) - gl.load(nwork_ptr)
        active = (e >= 0) & (e < NLOC)
        e = e * active.to(gl.int32)
        r0 = pid_w
        r1 = pid_w + active.to(gl.int32)
        if pid_n == 0:  # this slot's local expert id, or -1, for the post-expert norm
            gl.store(nwork_ptr + 1 + pid_w, e + active.to(gl.int32) - 1)
    else:
        active = pid_w < gl.load(nwork_ptr)
        e = gl.load(work_ptr + pid_w * 3 + 0)
        r0 = gl.load(work_ptr + pid_w * 3 + 1)
        r1 = gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        n0 = pid_n * BN
        rw, qw, jw = _idx3(BN, 4, 8, L3)
        KB: gl.constexpr = K // BK  # tile-major weights, see _tile_major
        w_off = (e * (2 * I // BN) + pid_n) * KB * (BN * 32) + rw * 32 + qw * 8 + jw
        rs, qs, gs_ = _idx3(BN, 4, 4, LS)
        s_off = (e * (2 * I // BN) + pid_n) * KB * (BN * 16) + rs * 16 + qs * 4 + gs_
        mx, qx, cx = _idx3(BM, 4, 64, LX8 if BM == 8 else LX)
        rv = r0 + mx
        pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
        x_off = (pair // 8) * K + qx * 64 + cx
        x_mask = (rv < r1) & (cx >= 0)
        xoff, xmsk = _xg_offsets(sorted_ptr, r0, r1, K, True, BM)
        x_s = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 64], X_SMEM)
        xb_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], XB_SMEM)
        wp = w_ptr.to(gl.pointer_type(gl.int32))
        acc_g = gl.zeros([BN, BM], gl.float32, layout=mma)
        acc_u = gl.zeros([BN, BM], gl.float32, layout=mma)
        if PF:  # register double buffer: block k0 + BK is requested before block k0 is decoded
            wg_n = gl.load(wp + w_off)
            sg_n = gl.load(sf_ptr + s_off)
            wu_n = gl.load(wp + w_off + I * (K // 8))
            su_n = gl.load(sf_ptr + s_off + I * (K // 16))
        for k0 in range(0, K, BK):
            if PF:
                wg = wg_n
                sg = sg_n
                wu = wu_n
                su = su_n
                kn = k0 + BK
                wm = (rw >= 0) & (kn < K)
                sm = (rs >= 0) & (kn < K)
                wg_n = gl.load(wp + w_off + (kn // BK) * (BN * 32), mask=wm, other=0)
                sg_n = gl.load(sf_ptr + s_off + (kn // BK) * (BN * 16), mask=sm)
                wu_n = gl.load(wp + w_off + I * (K // 8) + (kn // BK) * (BN * 32), mask=wm, other=0)
                su_n = gl.load(sf_ptr + s_off + I * (K // 16) + (kn // BK) * (BN * 16), mask=sm)
            else:
                wg = gl.load(wp + w_off + (k0 // BK) * (BN * 32))
                sg = gl.load(sf_ptr + s_off + (k0 // BK) * (BN * 16))
                wu = gl.load(wp + w_off + I * (K // 8) + (k0 // BK) * (BN * 32))
                su = gl.load(sf_ptr + s_off + I * (K // 16) + (k0 // BK) * (BN * 16))
            if SMB:
                _xb_store(x_ptr, xoff, xmsk, k0, xb_s, BM, BK)
                if CHK:
                    g0, g1, g2, g3 = _wsplit4(wg, BN)
                    h0, h1, h2, h3 = _ssplit4(sg, BN)
                    acc_g = _mma_chunk(g0, h0, xb_s, 0, acc_g, BN, a_l, HOP)
                    acc_g = _mma_chunk(g1, h1, xb_s, 1, acc_g, BN, a_l, HOP)
                    acc_g = _mma_chunk(g2, h2, xb_s, 2, acc_g, BN, a_l, HOP)
                    acc_g = _mma_chunk(g3, h3, xb_s, 3, acc_g, BN, a_l, HOP)
                    u0, u1, u2, u3 = _wsplit4(wu, BN)
                    v0, v1, v2, v3 = _ssplit4(su, BN)
                    acc_u = _mma_chunk(u0, v0, xb_s, 0, acc_u, BN, a_l, HOP)
                    acc_u = _mma_chunk(u1, v1, xb_s, 1, acc_u, BN, a_l, HOP)
                    acc_u = _mma_chunk(u2, v2, xb_s, 2, acc_u, BN, a_l, HOP)
                    acc_u = _mma_chunk(u3, v3, xb_s, 3, acc_u, BN, a_l, HOP)
                elif HOP:
                    acc_g = warpgroup_mma(_decode_tile(wg, sg, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc_g)
                    acc_u = warpgroup_mma(_decode_tile(wu, su, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc_u)
                else:
                    xb = xb_s.permute((1, 0)).load(B_L)
                    acc_g = mma_v2(_decode_tile(wg, sg, BN, BK, L3, A_L), xb, acc_g)
                    acc_u = mma_v2(_decode_tile(wu, su, BN, BK, L3, A_L), xb, acc_u)
            else:
                if XS:
                    xb = _x_shared(x_ptr, xoff, xmsk, k0, x_s, BM, BK)
                else:
                    xb = _x_tile(gl.load(x_ptr + x_off + k0, mask=x_mask, other=0.0), BM, BK, B_L)
                acc_g = mma_v2(_decode_tile(wg, sg, BN, BK, L3, A_L), xb, acc_g)
                acc_u = mma_v2(_decode_tile(wu, su, BN, BK, L3, A_L), xb, acc_u)
        g = gl.load(gs_ptr + e) * ACC_BIAS
        hg = (acc_g * g).to(gl.bfloat16).to(gl.float32)
        hu = (acc_u * g).to(gl.bfloat16).to(gl.float32)
        inter = (hg / (1.0 + gl.exp(-hg))).to(gl.bfloat16).to(gl.float32) * hu
        n_o = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
        m_o = gl.arange(0, BM, layout=gl.SliceLayout(0, mma))
        ro = r0 + m_o
        pair_o = gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
        optr = out_ptr + gl.expand_dims(pair_o, 0) * I + gl.expand_dims(n0 + n_o, 1)
        gl.store(optr, inter.to(gl.bfloat16), mask=gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1))


@gluon.jit
def _w13_silu_d1s(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr, part_ptr, cnt_ptr,
              K: gl.constexpr, I: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr,
              XS: gl.constexpr, SMB: gl.constexpr = False, HOP: gl.constexpr = False,
              CHK: gl.constexpr = False, PF: gl.constexpr = False,
              DIRECT: gl.constexpr = False, NLOC: gl.constexpr = 32, KS: gl.constexpr = 1,
              IL: gl.constexpr = False):
    mma: gl.constexpr = (MMA3_8 if BM == 8 else MMA3_16) if HOP else MMA_L
    a_l: gl.constexpr = (A3_8 if BM == 8 else A3_16) if HOP else A_L
    tl.extra.cuda.gdc_wait()  # wait for the producer's writes before any dependent read (launch_pdl=True launch)
    # no early launch_dependents trigger in this one-token kernel
    pid_n = gl.program_id(0)
    pid_w = gl.program_id(1)
    pid_k = gl.program_id(2)
    kb0 = pid_k * (K // KS)
    kb1 = kb0 + K // KS
    if DIRECT:  # M = 1: slot pid_w of the only token is one item of one row; no routing pass
        e = gl.load(work_ptr + pid_w) - gl.load(nwork_ptr)
        active = (e >= 0) & (e < NLOC)
        e = e * active.to(gl.int32)
        r0 = pid_w
        r1 = pid_w + active.to(gl.int32)
        if (pid_n == 0) & (pid_k == 0):  # this slot's local expert id, or -1, for the post-expert norm
            gl.store(nwork_ptr + 1 + pid_w, e + active.to(gl.int32) - 1)
    else:
        active = pid_w < gl.load(nwork_ptr)
        e = gl.load(work_ptr + pid_w * 3 + 0)
        r0 = gl.load(work_ptr + pid_w * 3 + 1)
        r1 = gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        n0 = pid_n * BN
        rw, qw, jw = _idx3(BN, 4, 8, L3)
        KB: gl.constexpr = K // BK  # tile-major weights, see _tile_major
        # IL = f1: per (expert, n-tile) KB contiguous k-block records
        # [gate codes BN*BK/2 | up codes BN*BK/2 | gate sf BN*BK/16 | up sf BN*BK/16], so a CTA's K-split is one contiguous
        # span instead of 4 DRAM streams (see _segf_layouts).
        RW: gl.constexpr = BN * 72 if IL else BN * 32               # int32 words per k-block record (18432 B)
        RB: gl.constexpr = BN * 288 if IL else BN * 16              # bytes per k-block record (scale stride)
        S0: gl.constexpr = BN * 256 if IL else 0                    # gate sf offset in the record
        UPW: gl.constexpr = BN * 32 if IL else I * (K // 8)         # up codes after gate codes
        UPB: gl.constexpr = BN * 16 if IL else I * (K // 16)        # up sf after gate sf
        NT: gl.constexpr = I // BN if IL else 2 * I // BN
        w_off = (e * NT + pid_n) * KB * RW + rw * 32 + qw * 8 + jw
        rs, qs, gs_ = _idx3(BN, 4, 4, LS)
        s_off = (e * NT + pid_n) * KB * RB + S0 + rs * 16 + qs * 4 + gs_
        mx, qx, cx = _idx3(BM, 4, 64, LX8 if BM == 8 else LX)
        rv = r0 + mx
        pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
        x_off = (pair // 8) * K + qx * 64 + cx
        x_mask = (rv < r1) & (cx >= 0)
        xoff, xmsk = _xg_offsets(sorted_ptr, r0, r1, K, True, BM)
        x_s = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 64], X_SMEM)
        xb_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], XB_SMEM)
        wp = w_ptr.to(gl.pointer_type(gl.int32))
        acc_g = gl.zeros([BN, BM], gl.float32, layout=mma)
        acc_u = gl.zeros([BN, BM], gl.float32, layout=mma)
        if PF:  # register double buffer: block k0 + BK is requested before block k0 is decoded
            wg_n = gl.load(wp + w_off + (kb0 // BK) * RW)
            sg_n = gl.load(sf_ptr + s_off + (kb0 // BK) * RB)
            wu_n = gl.load(wp + w_off + UPW + (kb0 // BK) * RW)
            su_n = gl.load(sf_ptr + s_off + UPB + (kb0 // BK) * RB)
        for k0 in range(kb0, kb1, BK):
            if PF:
                wg = wg_n
                sg = sg_n
                wu = wu_n
                su = su_n
                kn = k0 + BK
                wm = (rw >= 0) & (kn < kb1)
                sm = (rs >= 0) & (kn < kb1)
                wg_n = gl.load(wp + w_off + (kn // BK) * RW, mask=wm, other=0)
                sg_n = gl.load(sf_ptr + s_off + (kn // BK) * RB, mask=sm)
                wu_n = gl.load(wp + w_off + UPW + (kn // BK) * RW, mask=wm, other=0)
                su_n = gl.load(sf_ptr + s_off + UPB + (kn // BK) * RB, mask=sm)
            else:
                wg = gl.load(wp + w_off + (k0 // BK) * RW)
                sg = gl.load(sf_ptr + s_off + (k0 // BK) * RB)
                wu = gl.load(wp + w_off + UPW + (k0 // BK) * RW)
                su = gl.load(sf_ptr + s_off + UPB + (k0 // BK) * RB)
            if SMB:
                _xb_store(x_ptr, xoff, xmsk, k0, xb_s, BM, BK)
                if CHK:
                    g0, g1, g2, g3 = _wsplit4(wg, BN)
                    h0, h1, h2, h3 = _ssplit4(sg, BN)
                    acc_g = _mma_chunk(g0, h0, xb_s, 0, acc_g, BN, a_l, HOP)
                    acc_g = _mma_chunk(g1, h1, xb_s, 1, acc_g, BN, a_l, HOP)
                    acc_g = _mma_chunk(g2, h2, xb_s, 2, acc_g, BN, a_l, HOP)
                    acc_g = _mma_chunk(g3, h3, xb_s, 3, acc_g, BN, a_l, HOP)
                    u0, u1, u2, u3 = _wsplit4(wu, BN)
                    v0, v1, v2, v3 = _ssplit4(su, BN)
                    acc_u = _mma_chunk(u0, v0, xb_s, 0, acc_u, BN, a_l, HOP)
                    acc_u = _mma_chunk(u1, v1, xb_s, 1, acc_u, BN, a_l, HOP)
                    acc_u = _mma_chunk(u2, v2, xb_s, 2, acc_u, BN, a_l, HOP)
                    acc_u = _mma_chunk(u3, v3, xb_s, 3, acc_u, BN, a_l, HOP)
                elif HOP:
                    acc_g = warpgroup_mma(_decode_tile(wg, sg, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc_g)
                    acc_u = warpgroup_mma(_decode_tile(wu, su, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc_u)
                else:
                    xb = xb_s.permute((1, 0)).load(B_L)
                    acc_g = mma_v2(_decode_tile(wg, sg, BN, BK, L3, A_L), xb, acc_g)
                    acc_u = mma_v2(_decode_tile(wu, su, BN, BK, L3, A_L), xb, acc_u)
            else:
                if XS:
                    xb = _x_shared(x_ptr, xoff, xmsk, k0, x_s, BM, BK)
                else:
                    xb = _x_tile(gl.load(x_ptr + x_off + k0, mask=x_mask, other=0.0), BM, BK, B_L)
                acc_g = mma_v2(_decode_tile(wg, sg, BN, BK, L3, A_L), xb, acc_g)
                acc_u = mma_v2(_decode_tile(wu, su, BN, BK, L3, A_L), xb, acc_u)
        last = True
        if KS > 1:  # split-K partials, last arriver reduces in a fixed order
            n_p = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
            tile = pid_w * (I // BN) + pid_n
            pbase = part_ptr + (tile * KS + pid_k) * (2 * BN)
            gl.store(pbase + 0 * BN + n_p, gl.sum(acc_g, axis=1))
            gl.store(pbase + 1 * BN + n_p, gl.sum(acc_u, axis=1))
            gl.barrier()
            done = gl.atomic_add(cnt_ptr + tile, 1, sem="release")  # release only (acquire by the last arriver)
            last = done == KS - 1
            if last:
                seen = gl.atomic_add(cnt_ptr + tile, 0, sem="acquire")  # acquire RMW on the counter's release sequence
                z0 = seen - KS  # == 0 (all KS adds are in); reload addresses depend on the acquire, so it cannot be dropped
                gl.barrier()
                acc_g_s = gl.zeros([BN], gl.float32, layout=gl.SliceLayout(1, mma))
                acc_u_s = gl.zeros([BN], gl.float32, layout=gl.SliceLayout(1, mma))
                for j in gl.static_range(KS):
                    pj = part_ptr + (tile * KS + j) * (2 * BN) + z0
                    acc_g_s += gl.load(pj + 0 * BN + n_p, cache_modifier=".cg")
                    acc_u_s += gl.load(pj + 1 * BN + n_p, cache_modifier=".cg")
                acc_g = gl.expand_dims(acc_g_s, 1) + gl.zeros([BN, BM], gl.float32, layout=mma)
                acc_u = gl.expand_dims(acc_u_s, 1) + gl.zeros([BN, BM], gl.float32, layout=mma)
                gl.atomic_xchg(cnt_ptr + tile, 0)
        if last:
            g = gl.load(gs_ptr + e) * ACC_BIAS
            hg = (acc_g * g).to(gl.bfloat16).to(gl.float32)
            hu = (acc_u * g).to(gl.bfloat16).to(gl.float32)
            inter = (hg / (1.0 + gl.exp(-hg))).to(gl.bfloat16).to(gl.float32) * hu
            n_o = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
            m_o = gl.arange(0, BM, layout=gl.SliceLayout(0, mma))
            ro = r0 + m_o
            pair_o = gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
            optr = out_ptr + gl.expand_dims(pair_o, 0) * I + gl.expand_dims(n0 + n_o, 1)
            gl.store(optr, inter.to(gl.bfloat16), mask=gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1))


# ---- one-token W13 with the pre-expert norm fused into its prologue ----
@gluon.jit
def _w13_silu_d1s_n(x_ptr, nw_ptr, ls_ptr, eps, dbg_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr, part_ptr, cnt_ptr,
              K: gl.constexpr, I: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr,
              XS: gl.constexpr, SMB: gl.constexpr = False, HOP: gl.constexpr = False,
              CHK: gl.constexpr = False, PF: gl.constexpr = False,
              DIRECT: gl.constexpr = False, NLOC: gl.constexpr = 32, KS: gl.constexpr = 1,
              IL: gl.constexpr = False, DBG: gl.constexpr = False):
    mma: gl.constexpr = (MMA3_8 if BM == 8 else MMA3_16) if HOP else MMA_L
    a_l: gl.constexpr = (A3_8 if BM == 8 else A3_16) if HOP else A_L
    tl.extra.cuda.gdc_wait()  # wait for the producer's writes before any dependent read (launch_pdl=True launch)
    # no early launch_dependents trigger in this one-token kernel
    pid_n = gl.program_id(0)
    pid_w = gl.program_id(1)
    pid_k = gl.program_id(2)
    kb0 = pid_k * (K // KS)
    kb1 = kb0 + K // KS
    if DIRECT:  # M = 1: slot pid_w of the only token is one item of one row; no routing pass
        ls = gl.load(ls_ptr)  # local_start from its own pointer (the norm's DO_LS wrote buf[0])
        e = gl.load(work_ptr + pid_w) - ls
        active = (e >= 0) & (e < NLOC)
        e = e * active.to(gl.int32)
        r0 = pid_w
        r1 = pid_w + active.to(gl.int32)
        if (pid_n == 0) & (pid_w == 0) & (pid_k == 0):  # buf[0] = local_start for W2 (which waits for W13)
            gl.store(nwork_ptr, ls)
        if (pid_n == 0) & (pid_k == 0):  # this slot's local expert id, or -1, for the post-expert norm
            gl.store(nwork_ptr + 1 + pid_w, e + active.to(gl.int32) - 1)
    else:
        active = pid_w < gl.load(nwork_ptr)
        e = gl.load(work_ptr + pid_w * 3 + 0)
        r0 = gl.load(work_ptr + pid_w * 3 + 1)
        r1 = gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        gl.static_assert(XS and DIRECT and not SMB)
        # the pre-expert norm's rstd, with the norm kernel's own layout and expressions (TTGIR
        # #blocked1 = BlockedLayout([8], [32], [4], [0]) over K = 4096; num_warps 4 as the norm kernel)
        LN: gl.constexpr = gl.BlockedLayout([8], [32], [4], [0])
        xr = gl.load(x_ptr + gl.arange(0, K, layout=LN)).to(gl.float32)
        rstd = gl.rsqrt(gl.sum(xr * xr, axis=0) / K + eps)
        n0 = pid_n * BN
        rw, qw, jw = _idx3(BN, 4, 8, L3)
        KB: gl.constexpr = K // BK  # tile-major weights, see _tile_major
        # IL = f1: per (expert, n-tile) KB contiguous k-block records
        # [gate codes BN*BK/2 | up codes BN*BK/2 | gate sf BN*BK/16 | up sf BN*BK/16], so a CTA's K-split is one contiguous
        # span instead of 4 DRAM streams (see _segf_layouts).
        RW: gl.constexpr = BN * 72 if IL else BN * 32               # int32 words per k-block record (18432 B)
        RB: gl.constexpr = BN * 288 if IL else BN * 16              # bytes per k-block record (scale stride)
        S0: gl.constexpr = BN * 256 if IL else 0                    # gate sf offset in the record
        UPW: gl.constexpr = BN * 32 if IL else I * (K // 8)         # up codes after gate codes
        UPB: gl.constexpr = BN * 16 if IL else I * (K // 16)        # up sf after gate sf
        NT: gl.constexpr = I // BN if IL else 2 * I // BN
        w_off = (e * NT + pid_n) * KB * RW + rw * 32 + qw * 8 + jw
        rs, qs, gs_ = _idx3(BN, 4, 4, LS)
        s_off = (e * NT + pid_n) * KB * RB + S0 + rs * 16 + qs * 4 + gs_
        mx, qx, cx = _idx3(BM, 4, 64, LX8 if BM == 8 else LX)
        rv = r0 + mx
        pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
        x_off = (pair // 8) * K + qx * 64 + cx
        x_mask = (rv < r1) & (cx >= 0)
        xoff, xmsk = _xg_offsets(sorted_ptr, r0, r1, K, True, BM)
        x_s = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 64], X_SMEM)
        xb_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], XB_SMEM)
        wp = w_ptr.to(gl.pointer_type(gl.int32))
        acc_g = gl.zeros([BN, BM], gl.float32, layout=mma)
        acc_u = gl.zeros([BN, BM], gl.float32, layout=mma)
        if PF:  # register double buffer: block k0 + BK is requested before block k0 is decoded
            wg_n = gl.load(wp + w_off + (kb0 // BK) * RW)
            sg_n = gl.load(sf_ptr + s_off + (kb0 // BK) * RB)
            wu_n = gl.load(wp + w_off + UPW + (kb0 // BK) * RW)
            su_n = gl.load(sf_ptr + s_off + UPB + (kb0 // BK) * RB)
        for k0 in range(kb0, kb1, BK):
            if PF:
                wg = wg_n
                sg = sg_n
                wu = wu_n
                su = su_n
                kn = k0 + BK
                wm = (rw >= 0) & (kn < kb1)
                sm = (rs >= 0) & (kn < kb1)
                wg_n = gl.load(wp + w_off + (kn // BK) * RW, mask=wm, other=0)
                sg_n = gl.load(sf_ptr + s_off + (kn // BK) * RB, mask=sm)
                wu_n = gl.load(wp + w_off + UPW + (kn // BK) * RW, mask=wm, other=0)
                su_n = gl.load(sf_ptr + s_off + UPB + (kn // BK) * RB, mask=sm)
            else:
                wg = gl.load(wp + w_off + (k0 // BK) * RW)
                sg = gl.load(sf_ptr + s_off + (k0 // BK) * RB)
                wu = gl.load(wp + w_off + UPW + (k0 // BK) * RW)
                su = gl.load(sf_ptr + s_off + UPB + (k0 // BK) * RB)
            if SMB:
                _xb_store(x_ptr, xoff, xmsk, k0, xb_s, BM, BK)
                if CHK:
                    g0, g1, g2, g3 = _wsplit4(wg, BN)
                    h0, h1, h2, h3 = _ssplit4(sg, BN)
                    acc_g = _mma_chunk(g0, h0, xb_s, 0, acc_g, BN, a_l, HOP)
                    acc_g = _mma_chunk(g1, h1, xb_s, 1, acc_g, BN, a_l, HOP)
                    acc_g = _mma_chunk(g2, h2, xb_s, 2, acc_g, BN, a_l, HOP)
                    acc_g = _mma_chunk(g3, h3, xb_s, 3, acc_g, BN, a_l, HOP)
                    u0, u1, u2, u3 = _wsplit4(wu, BN)
                    v0, v1, v2, v3 = _ssplit4(su, BN)
                    acc_u = _mma_chunk(u0, v0, xb_s, 0, acc_u, BN, a_l, HOP)
                    acc_u = _mma_chunk(u1, v1, xb_s, 1, acc_u, BN, a_l, HOP)
                    acc_u = _mma_chunk(u2, v2, xb_s, 2, acc_u, BN, a_l, HOP)
                    acc_u = _mma_chunk(u3, v3, xb_s, 3, acc_u, BN, a_l, HOP)
                elif HOP:
                    acc_g = warpgroup_mma(_decode_tile(wg, sg, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc_g)
                    acc_u = warpgroup_mma(_decode_tile(wu, su, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc_u)
                else:
                    xb = xb_s.permute((1, 0)).load(B_L)
                    acc_g = mma_v2(_decode_tile(wg, sg, BN, BK, L3, A_L), xb, acc_g)
                    acc_u = mma_v2(_decode_tile(wu, su, BN, BK, L3, A_L), xb, acc_u)
            else:
                if XS:
                    xb = _x_shared_n(x_ptr, xoff, xmsk, k0, x_s, BM, BK, nw_ptr, rstd, K, dbg_ptr, DBG, pid_n == 0)
                else:
                    xb = _x_tile(gl.load(x_ptr + x_off + k0, mask=x_mask, other=0.0), BM, BK, B_L)
                acc_g = mma_v2(_decode_tile(wg, sg, BN, BK, L3, A_L), xb, acc_g)
                acc_u = mma_v2(_decode_tile(wu, su, BN, BK, L3, A_L), xb, acc_u)
        last = True
        if KS > 1:  # split-K partials, last arriver reduces in a fixed order
            n_p = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
            tile = pid_w * (I // BN) + pid_n
            pbase = part_ptr + (tile * KS + pid_k) * (2 * BN)
            gl.store(pbase + 0 * BN + n_p, gl.sum(acc_g, axis=1))
            gl.store(pbase + 1 * BN + n_p, gl.sum(acc_u, axis=1))
            gl.barrier()
            done = gl.atomic_add(cnt_ptr + tile, 1, sem="release")  # release only (acquire by the last arriver)
            last = done == KS - 1
            if last:
                seen = gl.atomic_add(cnt_ptr + tile, 0, sem="acquire")  # acquire RMW on the counter's release sequence
                z0 = seen - KS  # == 0 (all KS adds are in); reload addresses depend on the acquire, so it cannot be dropped
                gl.barrier()
                acc_g_s = gl.zeros([BN], gl.float32, layout=gl.SliceLayout(1, mma))
                acc_u_s = gl.zeros([BN], gl.float32, layout=gl.SliceLayout(1, mma))
                for j in gl.static_range(KS):
                    pj = part_ptr + (tile * KS + j) * (2 * BN) + z0
                    acc_g_s += gl.load(pj + 0 * BN + n_p, cache_modifier=".cg")
                    acc_u_s += gl.load(pj + 1 * BN + n_p, cache_modifier=".cg")
                acc_g = gl.expand_dims(acc_g_s, 1) + gl.zeros([BN, BM], gl.float32, layout=mma)
                acc_u = gl.expand_dims(acc_u_s, 1) + gl.zeros([BN, BM], gl.float32, layout=mma)
                gl.atomic_xchg(cnt_ptr + tile, 0)
        if last:
            g = gl.load(gs_ptr + e) * ACC_BIAS
            hg = (acc_g * g).to(gl.bfloat16).to(gl.float32)
            hu = (acc_u * g).to(gl.bfloat16).to(gl.float32)
            inter = (hg / (1.0 + gl.exp(-hg))).to(gl.bfloat16).to(gl.float32) * hu
            n_o = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
            m_o = gl.arange(0, BM, layout=gl.SliceLayout(0, mma))
            ro = r0 + m_o
            pair_o = gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
            optr = out_ptr + gl.expand_dims(pair_o, 0) * I + gl.expand_dims(n0 + n_o, 1)
            gl.store(optr, inter.to(gl.bfloat16), mask=gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1))


@gluon.jit
def _w2_d1(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr,
        K: gl.constexpr, N: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr, XS: gl.constexpr,
        SMB: gl.constexpr = False, HOP: gl.constexpr = False,
              CHK: gl.constexpr = False, PF: gl.constexpr = False,
              DIRECT: gl.constexpr = False, NLOC: gl.constexpr = 32):
    mma: gl.constexpr = (MMA3_8 if BM == 8 else MMA3_16) if HOP else MMA_L
    a_l: gl.constexpr = (A3_8 if BM == 8 else A3_16) if HOP else A_L
    pid_n = gl.program_id(0)
    pid_w = gl.program_id(1)
    if DIRECT:  # M = 1: slot pid_w of the only token is one item of one row; no routing pass
        e = gl.load(work_ptr + pid_w) - gl.load(nwork_ptr)
        active = (e >= 0) & (e < NLOC)
        e = e * active.to(gl.int32)
        r0 = pid_w
        r1 = pid_w + active.to(gl.int32)
    else:
        active = pid_w < gl.load(nwork_ptr)
        e = gl.load(work_ptr + pid_w * 3 + 0)
        r0 = gl.load(work_ptr + pid_w * 3 + 1)
        r1 = gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        n0 = pid_n * BN
        rw, qw, jw = _idx3(BN, 4, 8, L3)
        KB: gl.constexpr = K // BK  # tile-major weights
        w_off = (e * (N // BN) + pid_n) * KB * (BN * 32) + rw * 32 + qw * 8 + jw
        rs, qs, gs_ = _idx3(BN, 4, 4, LS)
        s_off = (e * (N // BN) + pid_n) * KB * (BN * 16) + rs * 16 + qs * 4 + gs_
        mx, qx, cx = _idx3(BM, 4, 64, LX8 if BM == 8 else LX)
        rv = r0 + mx
        pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
        x_off = pair * K + qx * 64 + cx
        x_mask = (rv < r1) & (cx >= 0)
        xoff, xmsk = _xg_offsets(sorted_ptr, r0, r1, K, False, BM)
        x_s = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 64], X_SMEM)
        xb_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], XB_SMEM)
        wp = w_ptr.to(gl.pointer_type(gl.int32))
        acc = gl.zeros([BN, BM], gl.float32, layout=mma)
        if PF:
            wq_n = gl.load(wp + w_off)
            sq_n = gl.load(sf_ptr + s_off)
        for k0 in range(0, K, BK):
            if PF:
                wq = wq_n
                sq = sq_n
                kn = k0 + BK
                wq_n = gl.load(wp + w_off + (kn // BK) * (BN * 32), mask=(rw >= 0) & (kn < K), other=0)
                sq_n = gl.load(sf_ptr + s_off + (kn // BK) * (BN * 16), mask=(rs >= 0) & (kn < K))
            else:
                wq = gl.load(wp + w_off + (k0 // BK) * (BN * 32))
                sq = gl.load(sf_ptr + s_off + (k0 // BK) * (BN * 16))
            if SMB:
                _xb_store(a_ptr, xoff, xmsk, k0, xb_s, BM, BK)
                if CHK:
                    q0, q1, q2, q3 = _wsplit4(wq, BN)
                    p0, p1, p2, p3 = _ssplit4(sq, BN)
                    acc = _mma_chunk(q0, p0, xb_s, 0, acc, BN, a_l, HOP)
                    acc = _mma_chunk(q1, p1, xb_s, 1, acc, BN, a_l, HOP)
                    acc = _mma_chunk(q2, p2, xb_s, 2, acc, BN, a_l, HOP)
                    acc = _mma_chunk(q3, p3, xb_s, 3, acc, BN, a_l, HOP)
                elif HOP:
                    acc = warpgroup_mma(_decode_tile(wq, sq, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc)
                else:
                    acc = mma_v2(_decode_tile(wq, sq, BN, BK, L3, A_L), xb_s.permute((1, 0)).load(B_L), acc)
            else:
                if XS:
                    xb = _x_shared(a_ptr, xoff, xmsk, k0, x_s, BM, BK)
                else:
                    xb = _x_tile(gl.load(a_ptr + x_off + k0, mask=x_mask, other=0.0), BM, BK, B_L)
                acc = mma_v2(_decode_tile(wq, sq, BN, BK, L3, A_L), xb, acc)
        y = acc * (gl.load(gs_ptr + e) * ACC_BIAS)
        n_o = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
        m_o = gl.arange(0, BM, layout=gl.SliceLayout(0, mma))
        ro = r0 + m_o
        pair_o = gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
        optr = out_ptr + gl.expand_dims(pair_o, 0) * N + gl.expand_dims(n0 + n_o, 1)
        gl.store(optr, y.to(gl.bfloat16), mask=gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1))
    if DIRECT:
        if r1 == r0:  # non-local slot: zero output row, which the post-expert norm maps to zero
            zl: gl.constexpr = gl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])
            zo = gl.arange(0, BN, layout=zl)
            gl.store(out_ptr + pid_w * N + pid_n * BN + zo, gl.zeros([BN], gl.bfloat16, layout=zl))


@gluon.jit
def _w2_d1s(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr, part_ptr, cnt_ptr,
        K: gl.constexpr, N: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr, XS: gl.constexpr,
        SMB: gl.constexpr = False, HOP: gl.constexpr = False,
              CHK: gl.constexpr = False, PF: gl.constexpr = False,
              DIRECT: gl.constexpr = False, NLOC: gl.constexpr = 32, KS: gl.constexpr = 1,
              OV: gl.constexpr = 0,  # 0 = the plain K loop
              IL: gl.constexpr = False):  # IL: contiguous per-block code+scale records
    mma: gl.constexpr = (MMA3_8 if BM == 8 else MMA3_16) if HOP else MMA_L
    a_l: gl.constexpr = (A3_8 if BM == 8 else A3_16) if HOP else A_L
    tl.extra.cuda.gdc_wait()  # wait for the producer's writes before any dependent read (launch_pdl=True launch)
    # no early launch_dependents trigger in this one-token kernel
    pid_n = gl.program_id(0)
    pid_w = gl.program_id(1)
    pid_k = gl.program_id(2)
    kb0 = pid_k * (K // KS)
    kb1 = kb0 + K // KS
    if DIRECT:  # M = 1: slot pid_w of the only token is one item of one row; no routing pass
        e = gl.load(work_ptr + pid_w) - gl.load(nwork_ptr)
        active = (e >= 0) & (e < NLOC)
        e = e * active.to(gl.int32)
        r0 = pid_w
        r1 = pid_w + active.to(gl.int32)
    else:
        active = pid_w < gl.load(nwork_ptr)
        e = gl.load(work_ptr + pid_w * 3 + 0)
        r0 = gl.load(work_ptr + pid_w * 3 + 1)
        r1 = gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        n0 = pid_n * BN
        rw, qw, jw = _idx3(BN, 4, 8, L3)
        KB: gl.constexpr = K // BK  # tile-major weights
        # IL = il2: codes then scales per 9216 B block (one stream per CTA instead of two)
        BLKW: gl.constexpr = BN * 36 if IL else BN * 32
        BLKB: gl.constexpr = BN * 144 if IL else BN * 16
        S0: gl.constexpr = BN * 128 if IL else 0
        w_off = (e * (N // BN) + pid_n) * KB * BLKW + rw * 32 + qw * 8 + jw
        rs, qs, gs_ = _idx3(BN, 4, 4, LS)
        s_off = (e * (N // BN) + pid_n) * KB * BLKB + S0 + rs * 16 + qs * 4 + gs_
        mx, qx, cx = _idx3(BM, 4, 64, LX8 if BM == 8 else LX)
        rv = r0 + mx
        pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
        x_off = pair * K + qx * 64 + cx
        x_mask = (rv < r1) & (cx >= 0)
        xoff, xmsk = _xg_offsets(sorted_ptr, r0, r1, K, False, BM)
        x_s = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 64], X_SMEM)
        xb_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], XB_SMEM)
        wp = w_ptr.to(gl.pointer_type(gl.int32))
        acc = gl.zeros([BN, BM], gl.float32, layout=mma)
        if PF:
            wq_n = gl.load(wp + w_off + (kb0 // BK) * BLKW)
            sq_n = gl.load(sf_ptr + s_off + (kb0 // BK) * BLKB)
        if OV > 0:  # OV schedule (one-token DIRECT, XS, PF, 2 chunks per CTA): activation staging hoisted, barrier-free decode/MMA block
            gl.static_assert((K // KS) // BK == 2)
            wq0 = wq_n
            sq0 = sq_n
            wq1 = gl.load(wp + w_off + ((kb0 + BK) // BK) * BLKW)
            sq1 = gl.load(sf_ptr + s_off + ((kb0 + BK) // BK) * BLKB)
            xb0 = _x_shared(a_ptr, xoff, xmsk, kb0, x_s, BM, BK)
            xb1 = _x_shared(a_ptr, xoff, xmsk, kb0 + BK, x_s, BM, BK)
            if OV == 1:
                acc = mma_v2(_decode_tile(wq0, sq0, BN, BK, L3, A_L), xb0, acc)
                acc = mma_v2(_decode_tile(wq1, sq1, BN, BK, L3, A_L), xb1, acc)
            else:
                t0 = _decode_tile(wq0, sq0, BN, BK, L3, A_L)
                t1 = _decode_tile(wq1, sq1, BN, BK, L3, A_L)
                acc = mma_v2(t0, xb0, acc)
                acc = mma_v2(t1, xb1, acc)
        else:
            for k0 in range(kb0, kb1, BK):
                if PF:
                    wq = wq_n
                    sq = sq_n
                    kn = k0 + BK
                    wq_n = gl.load(wp + w_off + (kn // BK) * BLKW, mask=(rw >= 0) & (kn < kb1), other=0)
                    sq_n = gl.load(sf_ptr + s_off + (kn // BK) * BLKB, mask=(rs >= 0) & (kn < kb1))
                else:
                    wq = gl.load(wp + w_off + (k0 // BK) * BLKW)
                    sq = gl.load(sf_ptr + s_off + (k0 // BK) * BLKB)
                if SMB:
                    _xb_store(a_ptr, xoff, xmsk, k0, xb_s, BM, BK)
                    if CHK:
                        q0, q1, q2, q3 = _wsplit4(wq, BN)
                        p0, p1, p2, p3 = _ssplit4(sq, BN)
                        acc = _mma_chunk(q0, p0, xb_s, 0, acc, BN, a_l, HOP)
                        acc = _mma_chunk(q1, p1, xb_s, 1, acc, BN, a_l, HOP)
                        acc = _mma_chunk(q2, p2, xb_s, 2, acc, BN, a_l, HOP)
                        acc = _mma_chunk(q3, p3, xb_s, 3, acc, BN, a_l, HOP)
                    elif HOP:
                        acc = warpgroup_mma(_decode_tile(wq, sq, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc)
                    else:
                        acc = mma_v2(_decode_tile(wq, sq, BN, BK, L3, A_L), xb_s.permute((1, 0)).load(B_L), acc)
                else:
                    if XS:
                        xb = _x_shared(a_ptr, xoff, xmsk, k0, x_s, BM, BK)
                    else:
                        xb = _x_tile(gl.load(a_ptr + x_off + k0, mask=x_mask, other=0.0), BM, BK, B_L)
                    acc = mma_v2(_decode_tile(wq, sq, BN, BK, L3, A_L), xb, acc)
        last = True
        if KS > 1:  # split-K partials, last arriver reduces in a fixed order
            n_p = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
            tile = pid_w * (N // BN) + pid_n
            pbase = part_ptr + (tile * KS + pid_k) * (1 * BN)
            gl.store(pbase + 0 * BN + n_p, gl.sum(acc, axis=1))
            gl.barrier()
            done = gl.atomic_add(cnt_ptr + tile, 1, sem="release")  # release only (acquire by the last arriver)
            last = done == KS - 1
            if last:
                seen = gl.atomic_add(cnt_ptr + tile, 0, sem="acquire")  # acquire RMW on the counter's release sequence
                z0 = seen - KS  # == 0 (all KS adds are in); reload addresses depend on the acquire, so it cannot be dropped
                gl.barrier()
                acc_s = gl.zeros([BN], gl.float32, layout=gl.SliceLayout(1, mma))
                for j in gl.static_range(KS):
                    pj = part_ptr + (tile * KS + j) * (1 * BN) + z0
                    acc_s += gl.load(pj + 0 * BN + n_p, cache_modifier=".cg")
                acc = gl.expand_dims(acc_s, 1) + gl.zeros([BN, BM], gl.float32, layout=mma)
                gl.atomic_xchg(cnt_ptr + tile, 0)
        if last:
            y = acc * (gl.load(gs_ptr + e) * ACC_BIAS)
            n_o = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
            m_o = gl.arange(0, BM, layout=gl.SliceLayout(0, mma))
            ro = r0 + m_o
            pair_o = gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
            optr = out_ptr + gl.expand_dims(pair_o, 0) * N + gl.expand_dims(n0 + n_o, 1)
            gl.store(optr, y.to(gl.bfloat16), mask=gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1))
    if DIRECT:
        if (r1 == r0) & (pid_k == 0):  # non-local slot: zero output row, which the post-expert norm maps to zero
            zl: gl.constexpr = gl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])
            zo = gl.arange(0, BN, layout=zl)
            gl.store(out_ptr + pid_w * N + pid_n * BN + zo, gl.zeros([BN], gl.bfloat16, layout=zl))


def _experts_d1(x, topk_ids, topk_weights, local_start, state):
    """_decode1_direct (one-token grouped experts) for this layer: each of the token's 8 slots is one work item read from topk_ids
    (no routing pass); slot 0 of the buffer holds this rank's first expert id, slots 1..8 receive the per-slot local
    ids (or -1) written by W13, which the post-expert finalize reads."""
    g = state["grouped"]
    H = x.shape[1]
    I = D_EXPERT
    buf = state["d1_buf"]
    # buf[0] = local_start is written by _rms_norm_local_ids_kernel (ls_out) just before this
    ids = topk_ids.to(torch.int32).contiguous()
    inter = torch.empty((TOP_K, I), dtype=torch.bfloat16, device=x.device)
    ks13, ks2 = D1_KS  # split-K over the K loop (shape constants)
    f1 = SEGF_F1 and g["segf_ok"]  # fallback: original layout and kernels if the prepare-time inverse check failed
    w13c, w13s = g["chain13"] if f1 else (g["w13"], g["w13_sf"])  # per-(expert, tile) contiguous k-block records
    w2c, w2s = g["il2"] if f1 else (g["w2"], g["w2_sf"])
    _w13_silu_d1s[(I // BN, TOP_K, ks13)](x, w13c, w13s, g["w13_gs"], state["iota8"], ids, buf, inter,
                                          state["d1s_part13"], state["d1s_cnt13"],
                                          K=H, I=I, BN=BN, BK=BK, BM=8, XS=True, SMB=False, HOP=False, CHK=False,
                                          PF=True, DIRECT=True, NLOC=LOCAL_EXPERTS, KS=ks13, IL=f1, launch_pdl=True, num_warps=NUM_WARPS)
    y = torch.empty((TOP_K, H), dtype=torch.bfloat16, device=x.device)
    _w2_d1s[(H // BN, TOP_K, ks2)](inter, w2c, w2s, g["w2_gs"], state["iota8"], ids, buf, y,
                                   state["d1s_part2"], state["d1s_cnt2"], K=I, N=H, BN=BN,
                                   BK=BK, BM=8, XS=True, SMB=False, HOP=False, CHK=False, PF=True, DIRECT=True,
                                   NLOC=LOCAL_EXPERTS, KS=ks2, OV=W2OV_MODE, IL=f1, launch_pdl=True, num_warps=NUM_WARPS)
    return _post_expert_weighted_sum(y.view(1, TOP_K, H), topk_weights.to(torch.float32).contiguous(),
                                     state["post_expert_w"], buf[1:].view(1, TOP_K))


def _experts_d1_n(x_raw, topk_ids, topk_weights, local_start, state, dbg=None):
    """_experts_d1 with the pre-expert norm inside W13's prologue (x_raw = the pooler output)."""
    g = state["grouped"]
    H = x_raw.shape[1]
    I = D_EXPERT
    buf = state["d1_buf"]
    ids = topk_ids.to(torch.int32).contiguous()
    inter = torch.empty((TOP_K, I), dtype=torch.bfloat16, device=x_raw.device)
    ks13, ks2 = D1_KS
    f1 = SEGF_F1 and g["segf_ok"]  # fallback: original layout and kernels if the prepare-time inverse check failed
    w13c, w13s = g["chain13"] if f1 else (g["w13"], g["w13_sf"])  # per-(expert, tile) contiguous k-block records
    w2c, w2s = g["il2"] if f1 else (g["w2"], g["w2_sf"])
    _w13_silu_d1s_n[(I // BN, TOP_K, ks13)](x_raw, state["pre_expert_w"], local_start, RMS_EPS, dbg if dbg is not None else buf,
                                            w13c, w13s, g["w13_gs"], state["iota8"], ids, buf, inter,
                                            state["d1s_part13"], state["d1s_cnt13"],
                                            K=H, I=I, BN=BN, BK=BK, BM=8, XS=True, SMB=False, HOP=False, CHK=False,
                                            PF=True, DIRECT=True, NLOC=LOCAL_EXPERTS, KS=ks13, IL=f1, DBG=dbg is not None,
                                            launch_pdl=True, num_warps=NUM_WARPS)
    y = torch.empty((TOP_K, H), dtype=torch.bfloat16, device=x_raw.device)
    _w2_d1s[(H // BN, TOP_K, ks2)](inter, w2c, w2s, g["w2_gs"], state["iota8"], ids, buf, y,
                                   state["d1s_part2"], state["d1s_cnt2"], K=I, N=H, BN=BN,
                                   BK=BK, BM=8, XS=True, SMB=False, HOP=False, CHK=False, PF=True, DIRECT=True,
                                   NLOC=LOCAL_EXPERTS, KS=ks2, OV=W2OV_MODE, IL=f1, launch_pdl=True, num_warps=NUM_WARPS)
    return _post_expert_weighted_sum(y.view(1, TOP_K, H), topk_weights.to(torch.float32).contiguous(),
                                     state["post_expert_w"], buf[1:].view(1, TOP_K))


def _gemm(x, w, out=None, pref=False):
    """SGLang's dispatch, except a single token goes to a bandwidth-bound Triton GEMV."""
    if x.shape[0] == 1 and x.is_contiguous() and w.is_contiguous():
        n, k = w.shape
        bn, bk, nw = GEMV_CFG.get(k, (8, 512, 4))
        if out is None:
            out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
        _gemv_kernel[(n // bn,)](x, w, out, K=k, BN=bn, BK=bk, num_warps=nw, PREF=pref, **_PDL_KW)  # L2 weight prefetch: o-proj at sliding batch 1 only (kernel_fn)
        return out
    if out is not None:
        return torch.mm(x, w.t(), out=out)
    cfg = SWAP_CFG.get((w.shape[0], w.shape[1], x.shape[0]))
    if cfg is not None and x.is_contiguous() and w.is_contiguous():  # fixed-shape branch
        return _gemm_swap(x, w, cfg, pref)
    return _bf16_gemm_dispatch_impl(x, w, None)


# ---- projection kernels adapted from the muse_bf16_proj solution (this block through "end of block"). Served only at
# LANEB_DONOR's (layer, tokens) pairs (fixed-shape branch), measured faster than this layer's own path on the H200.
# Its split-K kernels share one partials buffer and
# self-resetting counters: they run only on the main stream, one at a time (the side stream's router never uses them).
# ---- small-m streaming block (names prefixed _cbf_ / CBF_) ----
# Swap-AB MMA weight streaming with K split into ranges of whole blocks and an in-kernel last-arriver
# reduction (self-resetting counters). CBF_ENABLED lists the (layer, m bucket) cases this block serves;
# everything else returns None and runs the base path.


@triton.jit
def _cbf_skinny_gemm(x_ptr, w_ptr, w2_ptr, out_ptr, part_ptr, cnt_ptr, M, N, K,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 SPLIT_K: tl.constexpr, HILO: tl.constexpr, OUT_F32: tl.constexpr, ATOMIC: tl.constexpr = False):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    # K is cut into SPLIT_K ranges of whole BLOCK_K blocks (they may differ by one block), so the
    # program count can match the SM count for any split.
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    k0 = kb0 * BLOCK_K
    w_off = offs_n[:, None].to(tl.int64) * K + (k0 + offs_k)[None, :]
    x_ptrs = x_ptr + offs_m[None, :].to(tl.int64) * K + (k0 + offs_k)[:, None]
    m_mask = offs_m[None, :] < M
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(kb0, kb1):
        xt = tl.load(x_ptrs, mask=m_mask, other=0.0)
        w = tl.load(w_ptr + w_off)
        acc = tl.dot(w, xt, acc)
        if HILO:
            w2 = tl.load(w2_ptr + w_off)
            acc = tl.dot(w2, xt, acc)
        w_off += BLOCK_K
        x_ptrs += BLOCK_K
    out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
    if ATOMIC:
        # FP32 output zeroed by the caller: each K range adds its partial directly (no reduction tail).
        tl.atomic_add(out_ptrs, acc, mask=m_mask, sem="relaxed", scope="gpu")
    elif SPLIT_K == 1:
        if OUT_F32:
            tl.store(out_ptrs, acc, mask=m_mask)
        else:
            tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask)
    else:
        # Partials: [m_tiles, SPLIT_K, BLOCK_M, N] FP32, written once per call before being read.
        tile = pid_m * SPLIT_K
        p_base = part_ptr + (tile * BLOCK_M).to(tl.int64) * N
        p_off = (tl.arange(0, BLOCK_M)[None, :] * N + offs_n[:, None]).to(tl.int64)
        tl.store(p_base + pid_k * BLOCK_M * N + p_off, acc)
        tl.debug_barrier()
        cnt_idx = pid_m * tl.num_programs(0) + pid_n
        arrived = tl.atomic_add(cnt_ptr + cnt_idx, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                tot += tl.load(p_base + s * BLOCK_M * N + p_off, cache_modifier=".cg")
            if OUT_F32:
                tl.store(out_ptrs, tot, mask=m_mask)
            else:
                tl.store(out_ptrs, tot.to(tl.bfloat16), mask=m_mask)
            tl.atomic_xchg(cnt_ptr + cnt_idx, 0, sem="relaxed", scope="gpu")


def _cbf_launch(x, w, n, state, cfg, block_m):
    m, k = x.shape
    block_n, block_k, split_k, warps, stages = cfg
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    grid = (n // block_n, split_k, triton.cdiv(m, block_m))
    _cbf_skinny_gemm[grid](x, w, w, out, state["cbf_partials"], state["cbf_counters"], m, n, k,
                           BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k,
                           HILO=False, OUT_F32=False, ATOMIC=False, num_warps=warps, num_stages=stages)
    return out


# Streaming MMA: (layer, m bucket 16 = 1-16 tokens, 32 = 17-32 tokens) -> BLOCK_N, BLOCK_K, SPLIT_K, warps, stages.
CBF_STREAM = {
    ("qkv_gate", 16): (32, 64, 11, 4, 4),     # 72 tiles x 11 K ranges = 792 programs (H200: 1.093 vs SGLang at m=1)
    ("o", 16): (64, 128, 1, 4, 4),            # 128 tiles, no K split (H200: 1.08 / 1.10 vs SGLang at m8 / m16)
    ("o", 32): (64, 128, 1, 4, 4),            # H200: 1.08 vs SGLang at m32
}
# Cases served: qkv_gate at 1-16 tokens (one token runs the same 16-wide MMA tile).
CBF_ENABLED = {("qkv_gate", 1), ("qkv_gate", 16), ("o", 16), ("o", 32)}


try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except Exception:  # older Triton: the pointer kernel is used everywhere
    TensorDescriptor = None  # older Triton: the pooler case keeps the base path


@triton.jit
def _cbf_skinny_gemm_tma(x_desc, w_desc, w2_desc, out_ptr, part_ptr, cnt_ptr, M, N, K,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     SPLIT_K: tl.constexpr, HILO: tl.constexpr, OUT_F32: tl.constexpr):
    """_skinny_gemm with weight and activation tiles fetched by TMA (host-built tensor descriptors);
    rows of x beyond M are zero-filled by the copy engine. K is cut into SPLIT_K ranges of whole blocks."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    n0 = pid_n * BLOCK_N
    m0 = pid_m * BLOCK_M
    k = kb0 * BLOCK_K
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(kb0, kb1):
        xt = x_desc.load([m0, k]).T
        acc = tl.dot(w_desc.load([n0, k]), xt, acc)
        if HILO:
            acc = tl.dot(w2_desc.load([n0, k]), xt, acc)
        k += BLOCK_K
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_m = m0 + tl.arange(0, BLOCK_M)
    m_mask = offs_m[None, :] < M
    out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
    if SPLIT_K == 1:
        if OUT_F32:
            tl.store(out_ptrs, acc, mask=m_mask)
        else:
            tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask)
    else:
        tile = pid_m * SPLIT_K
        p_base = part_ptr + (tile * BLOCK_M).to(tl.int64) * N
        p_off = (tl.arange(0, BLOCK_M)[None, :] * N + offs_n[:, None]).to(tl.int64)
        tl.store(p_base + pid_k * BLOCK_M * N + p_off, acc)
        tl.debug_barrier()
        cnt_idx = pid_m * tl.num_programs(1) + pid_n
        arrived = tl.atomic_add(cnt_ptr + cnt_idx, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                tot += tl.load(p_base + s * BLOCK_M * N + p_off, cache_modifier=".cg")
            if OUT_F32:
                tl.store(out_ptrs, tot, mask=m_mask)
            else:
                tl.store(out_ptrs, tot.to(tl.bfloat16), mask=m_mask)
            tl.atomic_xchg(cnt_ptr + cnt_idx, 0, sem="relaxed", scope="gpu")


def _cbf_launch_tma(x, w, w2, n, state, cfg, block_m, out_dtype):
    m, k = x.shape
    block_n, block_k, split_k, warps, stages = cfg
    if split_k > 1 and triton.cdiv(m, block_m) * split_k * block_m * n > state["cbf_partials"].numel():
        raise ValueError("split-K scratch too small for this shape")
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    x_desc = TensorDescriptor.from_tensor(x, [block_m, block_k])
    w_desc = TensorDescriptor.from_tensor(w, [block_n, block_k])
    w2_desc = TensorDescriptor.from_tensor(w2 if w2 is not None else w, [block_n, block_k])
    grid = (triton.cdiv(m, block_m), n // block_n, split_k)
    _cbf_skinny_gemm_tma[grid](x_desc, w_desc, w2_desc, out, state["cbf_partials"], state["cbf_counters"], m, n, k,
                           BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k,
                           HILO=w2 is not None, OUT_F32=out_dtype == torch.float32,
                           num_warps=warps, num_stages=stages)
    return out


# layer -> BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages, served at 2-32 tokens (BLOCK_M = tokens rounded up
# to a power of two, at least 16)
# qkv_gate at 17-32 tokens on the TMA streaming kernel (H200: 1.068 vs SGLang)
CBF_TMA32 = {"qkv_gate": (64, 128, 3, 4, 4)}
CBF_TMA = {
    "pooler": (32, 256, 1, 4, 5),     # 128 programs, no K split (H200: 5-6% faster than (32, 128, 2, 4, 5))
    "o": (64, 256, 1, 4, 4),          # 2-32 tokens; on the H200 3-8% faster than the swap-AB streaming tile
    "unpooler": (64, 256, 1, 4, 4),   # 9-32 tokens (2-8 take the slice path first); H200: 17.0 vs 17.2-17.4 us for SGLang
}


@triton.jit
def _cbf_gemv_rows(X, W, Y, K: tl.constexpr, N: tl.constexpr, SX: tl.constexpr, SW: tl.constexpr,
                   ROWS: tl.constexpr, TILE_K: tl.constexpr):
    """One token on CUDA cores: ROWS weight rows per program, TILE_K columns per step, FP32 accumulation,
    weights loaded with .cg (the form of the base one-token pooler GEMV)."""
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, TILE_K)
    accum = tl.full((ROWS, TILE_K), 0, tl.float32)
    for tile in range(triton.cdiv(K, TILE_K)):
        k = tile * TILE_K + columns
        x = tl.load(X + k * SX, mask=k < K, other=0).to(tl.float32)
        w = tl.load(W + rows[:, None] * SW + k[None, :], mask=(rows[:, None] < N) & (k[None, :] < K),
                    other=0, cache_modifier=".cg").to(tl.float32)
        accum += w * x[None, :]
    tl.store(Y + rows, tl.sum(accum, axis=1).to(tl.bfloat16), mask=rows < N)


def _cbf_launch_gemv(x, w, n, cfg):
    rows, tile_k, warps = cfg
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    _cbf_gemv_rows[(triton.cdiv(n, rows),)](x, w, out, x.shape[1], n, x.stride(1), w.stride(0),
                                            ROWS=rows, TILE_K=tile_k, num_warps=warps, num_stages=1)
    return out


# One-token CUDA-core GEMV: layer -> ROWS, TILE_K, num_warps
CBF_GEMV1 = {
    "qkv_gate": (1, 4096, 16),  # 2304 programs, one row and one load burst each
    "pooler": (1, 4096, 16),    # H200: 15.5 vs 18.2 us for the base (4, 2048, 16) tiling
    "o": (2, 1024, 4),          # H200: 4.56 vs SGLang 5.50 us
    "unpooler": (1, 2048, 4),   # the 1024-row slice: 1024 programs (H200: 2.84 vs 3.49 us for (2, 4096, 4))
}


@triton.jit
def _cbf_gemv32(x_ptr, w_ptr, out_ptr, part_ptr, cnt_ptr, N, K,
          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, OUT_F32: tl.constexpr):
    """One token: out[n] = sum_k x[k] * W[n, k] on CUDA cores (W BF16 or FP32), FP32 accumulation."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_per = K // SPLIT_K
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k0 = pid_k * k_per
    w_ptrs = w_ptr + offs_n[:, None].to(tl.int64) * K + (k0 + offs_k)[None, :]
    x_ptrs = x_ptr + k0 + offs_k
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for _ in range(0, k_per, BLOCK_K):
        xv = tl.load(x_ptrs).to(tl.float32)
        w = tl.load(w_ptrs).to(tl.float32)
        acc += w * xv[None, :]
        w_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
    tot = tl.sum(acc, axis=1)
    if SPLIT_K == 1:
        if OUT_F32:
            tl.store(out_ptr + offs_n, tot)
        else:
            tl.store(out_ptr + offs_n, tot.to(tl.bfloat16))
    else:
        tl.store(part_ptr + pid_k * N + offs_n, tot)
        tl.debug_barrier()
        arrived = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            red = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                red += tl.load(part_ptr + s * N + offs_n, cache_modifier=".cg")
            if OUT_F32:
                tl.store(out_ptr + offs_n, red)
            else:
                tl.store(out_ptr + offs_n, red.to(tl.bfloat16))
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")


def _cbf_launch_gemv32(x, w, n, state, cfg, out_dtype):
    block_n, block_k, split_k, warps, stages = cfg
    out = torch.empty((1, n), dtype=out_dtype, device=x.device)
    _cbf_gemv32[(n // block_n, split_k)](x, w, out, state["cbf_partials"], state["cbf_counters"], n, x.shape[1],
                                   BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k,
                                   OUT_F32=out_dtype == torch.float32, num_warps=warps, num_stages=stages)
    return out


# Router at one token: FP32 weights as given, one row per program, 16 warps, no K split
# (H200: 2.701 against SGLang)
CBF_ROUTER1 = (1, 2048, 1, 16, 1)


@triton.jit
def _cbf_lm_partials(x_desc, w_desc, part_ptr, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    """Standard orientation (all tokens of a 128-token tile x BLOCK_N weight rows), TMA-loaded tiles, K split into
    SPLIT_K ranges of whole blocks; FP32 partials (SPLIT_K > 1) or the BF16 output (SPLIT_K == 1)."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k = kb0 * BLOCK_K
    for _ in range(kb0, kb1):
        a = x_desc.load([m0, k])
        b = w_desc.load([n0, k])
        acc = tl.dot(a, b.T, acc)
        k += BLOCK_K
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    off = offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptr + off, acc.to(tl.bfloat16), mask=mask)
    else:
        tl.store(part_ptr + pid_k.to(tl.int64) * M * N + off, acc, mask=mask)


@triton.jit
def _cbf_lm_reduce(part_ptr, out_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + s * MN + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)


def _cbf_launch_lm(x, w, n, state, cfg):
    block_n, block_k, split_k, warps, block_m = cfg
    m, k = x.shape
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if split_k > 1 and split_k * m * n > state["cbf_lm_partials"].numel():
        raise ValueError("split-K scratch too small for this shape")
    x_desc = TensorDescriptor.from_tensor(x, [block_m, block_k])
    w_desc = TensorDescriptor.from_tensor(w, [block_n, block_k])
    grid = (triton.cdiv(n, block_n), split_k, triton.cdiv(m, block_m))
    _cbf_lm_partials[grid](x_desc, w_desc, state["cbf_lm_partials"], out, m, n, k, BLOCK_M=block_m, BLOCK_N=block_n,
                           BLOCK_K=block_k, SPLIT_K=split_k, num_warps=warps, num_stages=state["cbf_lm_stages"])
    if split_k > 1:
        _cbf_lm_reduce[(triton.cdiv(m * n, 1024),)](state["cbf_lm_partials"], out, m * n, SPLIT_K=split_k,
                                                    BLOCK=1024, num_warps=4)
    return out


# 128 tokens (verify 16), split-K wgmma GEMM: layer -> BLOCK_N, BLOCK_K, SPLIT_K, num_warps, BLOCK_M
# (programs: 18 x 7 = 126, 32 x 4 = 128, 128 x 1 = 128; sm_90 cross-compile: HGMMA + UTMALDG, 96-131 KB shared)
CBF_LM128 = {
    "qkv_gate": (64, 128, 3, 4, 128),  # 36 x 3 = 108 programs (H200: 15.2 vs 16.0 us for cuBLAS)
    "pooler": (64, 128, 2, 4, 128),    # 64 x 2 = 128 programs (H200: 22.6 vs 23.3 us for (128, 64, 4, 8))
}


# The unpooler's 1024-row slice at 2-8 tokens: BLOCK_N, BLOCK_K, SPLIT_K, warps, stages (128 programs)
CBF_UNPOOL8 = (32, 128, 4, 4, 4)


@triton.jit
def _cbf_router_partials(X, HI, LO, PARTIAL, M, K: tl.constexpr, N: tl.constexpr, SXM, SXK,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    """The base router partials kernel generalised over the tiling: BF16 hi and lo products with separate FP32
    accumulators over one of SPLIT_K K ranges, summed and stored as an FP32 partial."""
    pid_n = tl.program_id(0)
    split = tl.program_id(1)
    pid_m = tl.program_id(2)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    iterations = triton.cdiv(K, SPLIT_K * BLOCK_K)
    acc_hi = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    acc_lo = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for tile in range(iterations):
        kk = (split * iterations + tile) * BLOCK_K + ks
        xx = tl.load(X + rows[:, None] * SXM + kk[None, :] * SXK, mask=(rows[:, None] < M) & (kk[None, :] < K),
                     other=0)
        hi = tl.load(HI + kk[:, None] * N + cols[None, :], mask=(kk[:, None] < K) & (cols[None, :] < N), other=0)
        lo = tl.load(LO + kk[:, None] * N + cols[None, :], mask=(kk[:, None] < K) & (cols[None, :] < N), other=0)
        acc_hi = tl.dot(xx, hi, acc_hi, input_precision="ieee")
        acc_lo = tl.dot(xx, lo, acc_lo, input_precision="ieee")
    off = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(PARTIAL + split * M * N + off, acc_hi + acc_lo, mask=mask)


@triton.jit
def _cbf_router_reduce(PARTIAL, OUT, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT_K)
    values = tl.load(PARTIAL + splits[:, None] * MN + offsets[None, :], mask=offsets[None, :] < MN, other=0)
    tl.store(OUT + offsets, tl.sum(values, axis=0), mask=offsets < MN)


def _cbf_launch_router(x, n, state, cfg):
    block_n, block_k, split_k, warps, stages, block_m = cfg
    m, k = x.shape
    if split_k * m * n > state["cbf_router_partials"].numel():
        raise ValueError("router scratch too small for this shape")
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _cbf_router_partials[(triton.cdiv(n, block_n), split_k, triton.cdiv(m, block_m))](
        x, state["router_hi_t"], state["router_lo_t"], state["cbf_router_partials"], m, k, n, x.stride(0), x.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k, num_warps=warps, num_stages=stages)
    _cbf_router_reduce[(triton.cdiv(m * n, 128),)](state["cbf_router_partials"], out, m * n, SPLIT_K=split_k, BLOCK=128,
                                                   num_warps=4)
    return out


# Router (BF16 hi/lo, two kernels) by exact token count: BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages, BLOCK_M
# (H200: m16 5.11 vs 6.29 us, m128 6.96 vs 10.28, m256 8.90 vs 10.88 for the base
# tilings).
CBF_ROUTER_HILO = {
    8: (32, 128, 16, 4, 4, 16),      # H200: 5.10 us vs ~6.0 us for an FP32 CUDA-core kernel
    16: (32, 64, 32, 4, 4, 16),
    32: (64, 64, 16, 4, 5, 16),      # H200: 5.79 us vs ~7.1 us for the base tiling
    128: (32, 128, 8, 4, 5, 64),
    256: (64, 64, 8, 4, 5, 64),
}
# TMA streaming for qkv_gate at 9-16 tokens (H200: 11.9 vs 12.4 us for the streaming kernel)
CBF_TMA16 = {"qkv_gate": (32, 128, 6, 4, 4)}


@triton.jit
def _cbf_gemv_rows32(X, W, Y, K: tl.constexpr, N: tl.constexpr, SX: tl.constexpr, SW: tl.constexpr,
                     ROWS: tl.constexpr, TILE_K: tl.constexpr):
    """One token on CUDA cores with FP32 weights and FP32 output (the router), the base one-token GEMV form."""
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, TILE_K)
    accum = tl.full((ROWS, TILE_K), 0, tl.float32)
    for tile in range(triton.cdiv(K, TILE_K)):
        k = tile * TILE_K + columns
        x = tl.load(X + k * SX, mask=k < K, other=0).to(tl.float32)
        w = tl.load(W + rows[:, None] * SW + k[None, :], mask=(rows[:, None] < N) & (k[None, :] < K), other=0,
                    cache_modifier=".cg")
        accum += w * x[None, :]
    tl.store(Y + rows, tl.sum(accum, axis=1), mask=rows < N)


def _cbf_launch_router1(x, w, n, cfg):
    rows, tile_k, warps = cfg
    out = torch.empty((1, n), dtype=torch.float32, device=x.device)
    _cbf_gemv_rows32[(triton.cdiv(n, rows),)](x, w, out, x.shape[1], n, x.stride(1), w.stride(0), ROWS=rows,
                                              TILE_K=tile_k, num_warps=warps, num_stages=1)
    return out


# Router at one token: ROWS, TILE_K, num_warps (H200: 2.65 vs 2.89 us for the 16-warp GEMV)
CBF_ROUTER1_ROWS = (1, 2048, 32)


def _cbf_prepare(state, device):
    state["cbf_partials"] = torch.empty(1 << 20, dtype=torch.float32, device=device)
    state["cbf_counters"] = torch.zeros(4096, dtype=torch.int32, device=device)
    state["cbf_lm_partials"] = torch.empty(4 << 20, dtype=torch.float32, device=device)  # 16 MiB of FP32 partials
    state["cbf_router_partials"] = torch.empty(1 << 20, dtype=torch.float32, device=device)  # 4 MiB
    # 4 pipeline stages need 131 KB of shared memory (H200: 228 KB); 3 elsewhere (e.g. 101 KB on sm_120)
    state["cbf_lm_stages"] = 4 if torch.cuda.get_device_capability(device)[0] == 9 else 3


def _cbf_dispatch(x, layer, n, state):
    """This block's output for the call, or None when the case is not one it serves."""
    m = x.shape[0]
    if layer == "router" and m in CBF_ROUTER_HILO:
        return _cbf_launch_router(x, n, state, CBF_ROUTER_HILO[m])
    if m == 128 and layer in CBF_LM128 and TensorDescriptor is not None:
        return _cbf_launch_lm(x, state[layer][:n], n, state, CBF_LM128[layer])
    if m > 32:
        return None
    if layer == "unpooler" and 1 < m <= 8:
        return _cbf_launch(x, state[layer][:n], n, state, CBF_UNPOOL8, 16)
    if m == 1 and layer in CBF_GEMV1:
        return _cbf_launch_gemv(x, state[layer][:n], n, CBF_GEMV1[layer])
    if m == 1 and layer == "router":
        return _cbf_launch_router1(x, state["router"], n, CBF_ROUTER1_ROWS)
    if 16 < m and layer in CBF_TMA32 and TensorDescriptor is not None:
        return _cbf_launch_tma(x, state[layer][:n], None, n, state, CBF_TMA32[layer], max(16, triton.next_power_of_2(m)),
                               torch.bfloat16)
    if 8 < m <= 16 and layer in CBF_TMA16 and TensorDescriptor is not None:
        return _cbf_launch_tma(x, state[layer][:n], None, n, state, CBF_TMA16[layer], 16, torch.bfloat16)
    if 1 < m and layer in CBF_TMA and TensorDescriptor is not None:
        return _cbf_launch_tma(x, state[layer][:n], None, n, state, CBF_TMA[layer], max(16, triton.next_power_of_2(m)),
                               torch.bfloat16)
    if (layer, 1 if m == 1 else 16 if m <= 16 else 32) not in CBF_ENABLED:
        return None
    cfg = CBF_STREAM.get((layer, 16 if m <= 16 else 32))
    if cfg is None:
        return None
    return _cbf_launch(x, state[layer][:n], n, state, cfg, max(16, triton.next_power_of_2(m)))
# ---- end of the muse_bf16_proj block ----

LANEB_DONOR = {('router', 1)}


@triton.jit
def _cbf_skinny_gemm_so(x_ptr, w_ptr, out_ptr, part_ptr, cnt_ptr, M, N, K, OUT_STRIDE,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    """_cbf_skinny_gemm (non-atomic path, identical arithmetic) with an output row stride, so the unpooler's
    1024-row slice at 2-8 tokens lands directly in the zero-padded [T, 8192] rows."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    k0 = kb0 * BLOCK_K
    w_off = offs_n[:, None].to(tl.int64) * K + (k0 + offs_k)[None, :]
    x_ptrs = x_ptr + offs_m[None, :].to(tl.int64) * K + (k0 + offs_k)[:, None]
    m_mask = offs_m[None, :] < M
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(kb0, kb1):
        xt = tl.load(x_ptrs, mask=m_mask, other=0.0)
        w = tl.load(w_ptr + w_off)
        acc = tl.dot(w, xt, acc)
        w_off += BLOCK_K
        x_ptrs += BLOCK_K
    out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * OUT_STRIDE + offs_n[:, None]
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask)
    else:
        tile = pid_m * SPLIT_K
        p_base = part_ptr + (tile * BLOCK_M).to(tl.int64) * N
        p_off = (tl.arange(0, BLOCK_M)[None, :] * N + offs_n[:, None]).to(tl.int64)
        tl.store(p_base + pid_k * BLOCK_M * N + p_off, acc)
        tl.debug_barrier()
        cnt_idx = pid_m * tl.num_programs(0) + pid_n
        arrived = tl.atomic_add(cnt_ptr + cnt_idx, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                tot += tl.load(p_base + s * BLOCK_M * N + p_off, cache_modifier=".cg")
            tl.store(out_ptrs, tot.to(tl.bfloat16), mask=m_mask)
            tl.atomic_xchg(cnt_ptr + cnt_idx, 0, sem="relaxed", scope="gpu")


def _laneb_prepare(state, weights):
    """State for the muse_bf16_proj block: the layer weights (the same tensors as state["dense"], no copy) and its scratch."""
    dense = state["dense"]
    cbf = {name: dense[name] for name in ("qkv_gate", "o", "pooler", "unpooler")}
    cbf.update(router=state["router"], router_hi_t=state["router_hi_t"], router_lo_t=state["router_lo_t"])
    _cbf_prepare(cbf, weights["qkv_gate"].device)
    return cbf



# ---- pinned cuBLASLt configurations at T=128/256 ----
# Chosen offline by a sweep of cuBLASLt algorithms and rebuilt in prepare() from their configuration attributes. A plan is
# used only if the rebuilt 64-byte algorithm equals the recorded one, cublasLtMatmulAlgoCheck accepts it, and a fixed
# self-test input reproduces the default call bit for bit; otherwise that shape keeps the default call
# (bitwise-safe) and _LT_OK / _LT_STATUS say so.
import ctypes as _lt_ct
import glob as _lt_glob
import os as _lt_os

LT_ENUMS = {'CUBLAS_COMPUTE_32F': 68, 'CUDA_R_32F': 0, 'CUDA_R_16BF': 14, 'CUBLAS_OP_N': 0, 'CUBLAS_OP_T': 1, 'CUBLASLT_MATMUL_DESC_TRANSA': 3, 'CUBLASLT_MATMUL_DESC_TRANSB': 4, 'CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES': 1, 'CUBLASLT_ALGO_CAP_SPLITK_SUPPORT': 0, 'CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK': 1, 'CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT': 2, 'CUBLASLT_ALGO_CAP_TILE_IDS': 6, 'CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX': 7, 'CUBLASLT_ALGO_CAP_STAGES_IDS': 13, 'CUBLASLT_ALGO_CONFIG_ID': 0, 'CUBLASLT_ALGO_CONFIG_TILE_ID': 1, 'CUBLASLT_ALGO_CONFIG_SPLITK_NUM': 2, 'CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME': 3, 'CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING': 4, 'CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION': 5, 'CUBLASLT_ALGO_CONFIG_STAGES_ID': 6, 'CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID': 7, 'CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID': 8, 'CUBLASLT_REDUCTION_SCHEME_NONE': 0, 'CUBLASLT_REDUCTION_SCHEME_INPLACE': 1, 'CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE': 2, 'CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE': 4}
LT_TABLE = {('pooler', 128): {'id': 66, 'tile': 27, 'stages': 35, 'splitk': 1, 'red': 2, 'swz': 0, 'custom': 2, 'inner': 0, 'cluster': 6, 'ws': 0, 'raw': [115964117058, 4294967331, 2, 1688849860263938, 558779540176897, 60129542158, 292058693646, 0]}, ('pooler', 256): {'id': 66, 'tile': 17, 'stages': 35, 'splitk': 1, 'red': 0, 'swz': 0, 'custom': 0, 'inner': 0, 'cluster': 5, 'ws': 0, 'raw': [73014444098, 4294967331, 0, 1407374883553280, 558779540176897, 60129542158, 292058693646, 0]}}
LT_SHAPES = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
LT_WS_BYTES = 64 << 20
_LT_OK = [False]  # every table entry active (diagnostic flag, like _B3CL_OK)
_LT_STATUS = {}  # "layer:T" -> "active" or why that shape keeps the default call
_LT_LAUNCH_FALLBACKS = [0]


class _LtAlgo(_lt_ct.Structure):
    _fields_ = [("data", _lt_ct.c_uint64 * 8)]


class _LtResult(_lt_ct.Structure):
    _fields_ = [("algo", _LtAlgo), ("workspaceSize", _lt_ct.c_size_t), ("state", _lt_ct.c_int),
                ("wavesCount", _lt_ct.c_float), ("reserved", _lt_ct.c_int * 4)]


def _lt_lib(device):
    torch.mm(torch.ones(128, 64, device=device, dtype=torch.bfloat16), torch.ones(64, 64, device=device, dtype=torch.bfloat16))
    path = next((l.split()[-1] for l in open("/proc/self/maps") if "libcublasLt.so" in l), None)
    if path is None:
        site = _lt_os.path.dirname(_lt_os.path.dirname(torch.__file__))
        hits = sorted(_lt_glob.glob(_lt_os.path.join(site, "nvidia", "**", "libcublasLt.so*"), recursive=True))
        path = hits[0] if hits else "libcublasLt.so"
    lib = _lt_ct.CDLL(path)
    vp, sz, i32, u64, i64, P = _lt_ct.c_void_p, _lt_ct.c_size_t, _lt_ct.c_int, _lt_ct.c_uint64, _lt_ct.c_int64, _lt_ct.POINTER
    sig = {"cublasLtCreate": [P(vp)], "cublasLtMatmulDescCreate": [P(vp), i32, i32],
           "cublasLtMatmulDescSetAttribute": [vp, i32, vp, sz], "cublasLtMatrixLayoutCreate": [P(vp), i32, u64, u64, i64],
           "cublasLtMatmulAlgoInit": [vp, i32, i32, i32, i32, i32, i32, i32, vp],
           "cublasLtMatmulAlgoConfigSetAttribute": [vp, i32, vp, sz],
           "cublasLtMatmulAlgoCheck": [vp, vp, vp, vp, vp, vp, vp, vp],
           "cublasLtMatmul": [vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, vp, sz, vp]}
    for fn, at in sig.items():
        getattr(lib, fn).argtypes = at
        getattr(lib, fn).restype = _lt_ct.c_int
    return lib, path


def _lt_ok(st, what):
    if st != 0:
        raise RuntimeError("LEVER-T128 cuBLASLt %s: status %d" % (what, st))


def _lt_prepare(state, device):
    """The pinned plans, built once from shape and configuration only (no value-dependent choice). A shape whose
    plan cannot be built, rebuilt byte-equal, checked, or reproduce the default call's raw bits on a fixed self-test input
    keeps the default call; _LT_OK[0] says whether every table entry is active, _LT_STATUS says why not."""
    _LT_OK[0] = False
    _LT_STATUS.clear()
    if not LT_TABLE:
        return
    plans = {}
    try:
        E = LT_ENUMS
        lib, path = _lt_lib(device)
        h = _lt_ct.c_void_p()
        _lt_ok(lib.cublasLtCreate(_lt_ct.byref(h)), "create")
        bf = E["CUDA_R_16BF"]
        types7 = (E["CUBLAS_COMPUTE_32F"], E["CUDA_R_32F"], bf, bf, bf, bf)
        lt = {"lib": lib, "path": path, "h": h, "plans": plans, "one": _lt_ct.c_float(1.0), "zero": _lt_ct.c_float(0.0),
              "ws": torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=device)}
    except Exception as exc:  # no cuBLASLt here: every shape keeps the default call
        for key in LT_TABLE:
            _LT_STATUS["%s:%d" % key] = "library: %s" % str(exc)[:120]
        return
    for (layer, t), cfg in LT_TABLE.items():
        key = "%s:%d" % (layer, t)
        try:
            n, k = LT_SHAPES[layer]
            desc = _lt_ct.c_void_p()
            _lt_ok(lib.cublasLtMatmulDescCreate(_lt_ct.byref(desc), E["CUBLAS_COMPUTE_32F"], E["CUDA_R_32F"]), "desc")
            ta, tb = _lt_ct.c_int32(E["CUBLAS_OP_T"]), _lt_ct.c_int32(E["CUBLAS_OP_N"])
            _lt_ok(lib.cublasLtMatmulDescSetAttribute(desc, E["CUBLASLT_MATMUL_DESC_TRANSA"], _lt_ct.byref(ta), 4), "transa")
            _lt_ok(lib.cublasLtMatmulDescSetAttribute(desc, E["CUBLASLT_MATMUL_DESC_TRANSB"], _lt_ct.byref(tb), 4), "transb")
            lay = []
            for rows, cols, ld in ((k, n, k), (k, t, k), (n, t, n)):
                m = _lt_ct.c_void_p()
                _lt_ok(lib.cublasLtMatrixLayoutCreate(_lt_ct.byref(m), bf, rows, cols, ld), "layout")
                lay.append(m)
            algo = _LtAlgo()
            _lt_ok(lib.cublasLtMatmulAlgoInit(h, *types7, cfg["id"], _lt_ct.byref(algo)), "init")
            for attr, name, size in (("CUBLASLT_ALGO_CONFIG_TILE_ID", "tile", 4), ("CUBLASLT_ALGO_CONFIG_STAGES_ID", "stages", 4),
                                     ("CUBLASLT_ALGO_CONFIG_SPLITK_NUM", "splitk", 4),
                                     ("CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME", "red", 4),
                                     ("CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING", "swz", 4), ("CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION", "custom", 4),
                                     ("CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID", "inner", 2),
                                     ("CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID", "cluster", 2)):
                v = (_lt_ct.c_uint32 if size == 4 else _lt_ct.c_uint16)(cfg[name])
                _lt_ok(lib.cublasLtMatmulAlgoConfigSetAttribute(_lt_ct.byref(algo), E[attr], _lt_ct.byref(v), size), name)
            if [int(v) for v in algo.data] != [int(v) for v in cfg["raw"]]:
                raise RuntimeError("rebuilt algorithm differs from the recorded one")
            chk = _LtResult()
            _lt_ok(lib.cublasLtMatmulAlgoCheck(h, desc, lay[0], lay[1], lay[2], lay[2], _lt_ct.byref(algo), _lt_ct.byref(chk)), "check")
            if chk.workspaceSize > LT_WS_BYTES:
                raise RuntimeError("workspace %d > %d" % (chk.workspaceSize, LT_WS_BYTES))
            plan = (desc, lay, algo, chk.workspaceSize, n)
            # self-test: the pinned plan must reproduce the default call bit for bit on a fixed input
            g = torch.Generator(device=device).manual_seed(1009 + t)
            x = torch.randn((t, k), generator=g, device=device).to(torch.bfloat16)
            w = state["dense"][layer]
            y = torch.empty((t, n), dtype=torch.bfloat16, device=device)
            _lt_ok(lib.cublasLtMatmul(h, desc, _lt_ct.byref(lt["one"]), w.data_ptr(), lay[0], x.data_ptr(), lay[1],
                                      _lt_ct.byref(lt["zero"]), y.data_ptr(), lay[2], y.data_ptr(), lay[2], _lt_ct.byref(algo),
                                      lt["ws"].data_ptr(), chk.workspaceSize, torch.cuda.current_stream().cuda_stream), "self-test launch")
            z = _gemm(x, w)
            torch.cuda.synchronize()
            if not torch.equal(y.view(torch.int16), z.contiguous().view(torch.int16)):
                raise RuntimeError("self-test output differs from the parent's call")
            plans[(layer, t)] = plan
            _LT_STATUS[key] = "active"
        except Exception as exc:
            _LT_STATUS[key] = "fallback to the parent's call: %s" % str(exc)[:160]
    state["lt"] = lt
    _LT_OK[0] = len(plans) == len(LT_TABLE)


def _lt_dense(x, layer, state):
    """The pinned cuBLASLt product for this (layer, T), or None (no active plan: the default call). A launch the
    library rejects (a host-side status, never a data value) drops the plan and is counted in _LT_LAUNCH_FALLBACKS."""
    lt = state.get("lt")
    plan = lt["plans"].get((layer, x.shape[0])) if lt is not None else None
    if plan is None or not x.is_contiguous():
        return None
    desc, lay, algo, ws, n = plan
    w = state["dense"][layer]
    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
    st = lt["lib"].cublasLtMatmul(lt["h"], desc, _lt_ct.byref(lt["one"]), w.data_ptr(), lay[0], x.data_ptr(), lay[1],
                                  _lt_ct.byref(lt["zero"]), out.data_ptr(), lay[2], out.data_ptr(), lay[2], _lt_ct.byref(algo),
                                  lt["ws"].data_ptr(), ws, torch.cuda.current_stream().cuda_stream)
    if st != 0:
        del lt["plans"][(layer, x.shape[0])]
        _LT_LAUNCH_FALLBACKS[0] += 1
        _LT_STATUS["%s:%d" % (layer, x.shape[0])] = "launch status %d: fallback to the parent's call" % st
        _LT_OK[0] = False
        return None
    return out
# ---- end of pinned cuBLASLt block ----


def _dense(x, layer, state, out=None, pref=False):
    """The projection for this layer: the muse_bf16_proj kernels at LANEB_DONOR's pairs, else this layer's _gemm."""
    t = x.shape[0]
    if (layer, t) in LANEB_DONOR and x.is_contiguous():
        cbf = state["cbf"]
        if layer == "unpooler" and t <= SLICED_UNPOOL_MAX_TOKENS:  # the 1024-row slice into the zero-padded rows
            w = cbf["unpooler"][:UNPOOL_SLICE]
            if t == 1:
                rows, tile_k, warps = CBF_GEMV1["unpooler"]
                _cbf_gemv_rows[(triton.cdiv(UNPOOL_SLICE, rows),)](x, w, out, x.shape[1], UNPOOL_SLICE, x.stride(1),
                                                                   w.stride(0), ROWS=rows, TILE_K=tile_k,
                                                                   num_warps=warps, num_stages=1)
            else:
                block_n, block_k, split_k, warps, stages = CBF_UNPOOL8
                _cbf_skinny_gemm_so[(UNPOOL_SLICE // block_n, split_k, 1)](
                    x, w, out, cbf["cbf_partials"], cbf["cbf_counters"], t, UNPOOL_SLICE, x.shape[1], out.stride(0),
                    BLOCK_M=16, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k, num_warps=warps, num_stages=stages)
            return out
        y = _cbf_dispatch(x, layer, cbf[layer].shape[0], cbf)
        if y is not None:
            return y
    if layer == "unpooler" and t <= SLICED_UNPOOL_MAX_TOKENS:
        if t == 1:
            return _gemm(x, state["dense"]["unpool_slice"], out=out[:, :UNPOOL_SLICE])
        if _B3CL_OK[0] and x.is_contiguous():  # cluster DSMEM split-K into the zero-padded rows
            try:
                return _b3cl_launch(x, state["dense"]["unpooler"][:UNPOOL_SLICE], UNPOOL_SLICE, B3CL_UNPOOL8, 16, out=out)
            except Exception as exc:  # first use happens before capture; torch.mm serves from then on
                _B3CL_OK[0] = False
                print("B3a: cluster kernel unavailable (%s: %s)" % (type(exc).__name__, exc))
        return torch.mm(x, state["dense"]["unpooler"][:UNPOOL_SLICE].t(), out=out[:, :UNPOOL_SLICE])
    if x.shape[0] >= 128:  # the pinned cuBLASLt plan, if this (layer, T) has one
        y = _lt_dense(x, layer, state)
        if y is not None:
            return y
    return _gemm(x, state["dense"][layer], pref=pref)


def _router_logits(x, t, state):
    """The router at LANEB_DONOR's pairs (one-token FP32 GEMV / split-K hi-lo), else None (this layer's own path)."""
    if ("router", t) not in LANEB_DONOR:
        return None
    cbf = state["cbf"]
    if t == 1:
        return _cbf_launch_router1(x, cbf["router"], cbf["router"].shape[0], CBF_ROUTER1_ROWS)
    return _cbf_launch_router(x, cbf["router"].shape[0], cbf, CBF_ROUTER_HILO[t])


# ---- cluster DSMEM split-K for small-m dense (the muse_bf16_proj cluster kernel, with an output row stride) ----
try:
    from triton.tools.tensor_descriptor import TensorDescriptor as _B3CL_TD
except Exception:  # older Triton
    _B3CL_TD = None
_B3CL_KERNELS = {}
_B3CL_OK = [torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9 and _B3CL_TD is not None]


def _b3cl_v4copy(n):
    body = ["{ .reg .u32 t, l, r; .reg .f32 a, b, c, d; mov.u32 t, %tid.x; shl.b32 t, t, 4; add.u32 l, $1, t; "
            "mapa.shared::cluster.u32 r, $2, 0; add.u32 r, r, t;"]
    for i in range(n):
        body.append(" ld.shared.v4.f32 {a, b, c, d}, [l+%d]; st.shared::cluster.v4.f32 [r+%d], {a, b, c, d};"
                    % (2048 * i, 2048 * i))
    body.append(" mov.u32 $0, 0; }")
    return "".join(body)


B3CL_V4COPY2 = tl.constexpr(_b3cl_v4copy(2))  # 4 KB tile (64 x 16 FP32), 128 threads x 16 bytes x 2
B3CL_V4COPY4 = tl.constexpr(_b3cl_v4copy(4))  # 8 KB tile (64 x 32 FP32)


@triton.jit
def _b3cl_gemm(x_desc, w_desc, out_ptr, M, N, K, OUT_STRIDE, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, SLOT_KB: tl.constexpr):
    pid_k = tl.program_id(0)  # rank in the cluster
    pid_n = tl.program_id(1)
    pid_m = tl.program_id(2)
    if SLOT_KB == 8:
        base = tl.inline_asm_elementwise(".shared .align 16 .b8 b3cl_slots[8192]; mov.u32 $0, b3cl_slots;", "=r,r",
                                         [pid_k], dtype=tl.int32, is_pure=False, pack=1)
    else:
        base = tl.inline_asm_elementwise(".shared .align 16 .b8 b3cl_slots[16384]; mov.u32 $0, b3cl_slots;", "=r,r",
                                         [pid_k], dtype=tl.int32, is_pure=False, pack=1)
    started = tl.inline_asm_elementwise("barrier.cluster.arrive.relaxed.aligned; mov.u32 $0, 0;", "=r,r", [pid_k],
                                        dtype=tl.int32, is_pure=False, pack=1)
    kb_total = K // BLOCK_K
    kb0 = pid_k * kb_total // SPLIT_K
    kb1 = (pid_k + 1) * kb_total // SPLIT_K
    n0 = pid_n * BLOCK_N
    m0 = pid_m * BLOCK_M
    k = kb0 * BLOCK_K
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for _ in range(kb0, kb1):
        xt = x_desc.load([m0, k]).T
        acc = tl.dot(w_desc.load([n0, k]), xt, acc)
        k += BLOCK_K
    idx4 = (tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]) * 4
    TILE_BYTES: tl.constexpr = BLOCK_N * BLOCK_M * 4
    if pid_k != 0:
        tl.inline_asm_elementwise("st.shared.f32 [$1], $2; mov.u32 $0, 0;", "=r,r,f", [base + idx4, acc],
                                  dtype=tl.int32, is_pure=False, pack=1)
        tl.debug_barrier()
        ready = tl.inline_asm_elementwise("barrier.cluster.wait.aligned; mov.u32 $0, 0;", "=r,r", [pid_k + started],
                                          dtype=tl.int32, is_pure=False, pack=1)
        dst = base + (pid_k - 1) * TILE_BYTES + ready
        if TILE_BYTES == 4096:
            tl.inline_asm_elementwise(B3CL_V4COPY2, "=r,r,r", [base, dst], dtype=tl.int32, is_pure=False, pack=1)
        else:
            tl.inline_asm_elementwise(B3CL_V4COPY4, "=r,r,r", [base, dst], dtype=tl.int32, is_pure=False, pack=1)
        tl.inline_asm_elementwise("barrier.cluster.arrive.release.aligned; mov.u32 $0, 0;", "=r,r", [pid_k],
                                  dtype=tl.int32, is_pure=False, pack=1)
    else:
        ready = tl.inline_asm_elementwise("barrier.cluster.wait.aligned; mov.u32 $0, 0;", "=r,r", [pid_k + started],
                                          dtype=tl.int32, is_pure=False, pack=1)
        done = tl.inline_asm_elementwise("barrier.cluster.arrive.release.aligned; "
                                         "barrier.cluster.wait.acquire.aligned; mov.u32 $0, 0;", "=r,r",
                                         [pid_k + ready], dtype=tl.int32, is_pure=False, pack=1)
        tot = acc
        for s in tl.static_range(1, SPLIT_K):
            tot += tl.inline_asm_elementwise("ld.shared.f32 $0, [$1];", "=f,r",
                                             [base + (s - 1) * TILE_BYTES + done + idx4], dtype=tl.float32,
                                             is_pure=False, pack=1)
        offs_n = n0 + tl.arange(0, BLOCK_N)
        offs_m = m0 + tl.arange(0, BLOCK_M)
        out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * OUT_STRIDE + offs_n[:, None]
        tl.store(out_ptrs, tot.to(tl.bfloat16), mask=offs_m[None, :] < M)


# BLOCK_N, BLOCK_K, SPLIT_K (= cluster size), num_warps (4: the copy assumes 128 threads), num_stages
B3CL_QKV = (64, 256, 3, 4, 4)
B3CL_UNPOOL8 = (64, 256, 4, 4, 4)  # unpooler slice (1024 rows) at 2-8 tokens: 16 tiles x 4-CTA clusters


def _b3cl_launch(x, w, n, cfg, block_m, out=None):
    from triton import knobs
    block_n, block_k, split_k, warps, stages = cfg
    m, k = x.shape
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    x_desc = _B3CL_TD.from_tensor(x, [block_m, block_k])
    w_desc = _B3CL_TD.from_tensor(w, [block_n, block_k])
    grid = (1, n // block_n, triton.cdiv(m, block_m))  # axis 0 is the cluster: split_k CTAs at launch
    slot_kb = 8 if (split_k - 1) * block_n * block_m * 4 <= 8192 else 16
    args = (x_desc, w_desc, out, m, n, k, out.stride(0), block_m, block_n, block_k, split_k, slot_kb)  # every parameter, in order
    key = (cfg, block_m, m, out.stride(0))  # m: Triton specialises integer divisibility
    kern = _B3CL_KERNELS.get(key)
    if kern is None:
        if n % block_n or k % block_k or warps != 4 or block_n * block_m * 4 not in (4096, 8192):
            raise ValueError("cluster kernel: unsupported shape or tiling")
        ck = _b3cl_gemm.warmup(*args[:7], BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                                      SPLIT_K=split_k, SLOT_KB=slot_kb, num_warps=warps, num_stages=stages, grid=grid)
        ck._init_handles()
        pm = list(ck.packed_metadata)  # (num_warps, num_ctas, shared)
        if len(pm) != 3 or pm[1] != 1 or pm[2] <= 49152:  # the 48 KB opt-in (static slots) is done by Triton above it
            raise ValueError("cluster kernel: unexpected launch metadata %r" % (pm,))
        pm[1] = split_k
        kern = _B3CL_KERNELS[key] = (ck, tuple(pm))
    ck, pm = kern
    stream = torch.cuda.current_stream().cuda_stream
    ck.run(grid[0], grid[1], grid[2], stream, ck.function, pm, ck.launch_metadata(grid, stream, *args),
           knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *args)
    return out
# ---- end of cluster DSMEM block ----


# ---- the layer boundaries on SGLang's DSMEM cluster kernel at SM90 ----
# SGLang's fused_norm_residual picks its multi-CTA cluster kernel (one row split over C CTAs, partial RMS moments
# exchanged through distributed shared memory) only at capability >= 10, and the one-CTA-per-row kernel otherwise.
# Both are SGLang's kernels; on the H200 the cluster kernel is faster at small token counts. Cluster
# size by token count (fixed-shape branch); absent counts keep the module's own call.
from sglang.kernels.ops.muse_spark_v12.norm_residual import fused_norm_residual as _fused_norm_residual
BOUNDARY_CLUSTER = {"attn": {8: 16, 16: 16, 32: 16}, "ffn": {8: 16, 16: 16, 32: 16}}
_BOUNDARY_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def _boundary(op, which, stream, branch, router_dtype=None):
    c = BOUNDARY_CLUSTER[which].get(branch.shape[0]) if _BOUNDARY_SM90 else None
    if c is None or not (op.jit_selected() and branch.is_contiguous() and stream.is_contiguous()):
        return op(stream, branch, branch_dtype=torch.bfloat16, router_dtype=router_dtype)
    op._ensure_cached_weights(torch.bfloat16)
    return _fused_norm_residual(
        branch, stream, op._post_norm.eps, op._pre_norm.eps, post_norm_is_input_scale=op._post_norm_is_input_scale,
        post_weight=op._cc_post_weight if op._post_norm.has_weight else None,
        pre_weight=op._cc_pre_weight if op._pre_norm.has_weight else None,
        alpha=op._cc_alpha if op._use_residual_gate else None, beta=op._cc_beta if op._use_residual_gate else None,
        want_router=router_dtype is not None, cluster_size=c)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {**{name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
                  "unpool_slice": weights["unpooler"][:UNPOOL_SLICE].contiguous()},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        "pre_expert_w": (weights["pre_expert_norm"] + 1.0).to(torch.bfloat16).contiguous(),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        # Hopper only: on a bandwidth-bound GPU (L40S) Marlin is at par or better at these sizes; on SM90
        # Marlin's padded 48/64-row blocks are the target (decided by device capability, not data).
        "grouped_moe": False,  # SGLang's Marlin (block capped) on every device
        "unpool_rows": torch.zeros((SLICED_UNPOOL_MAX_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device),
        "raw_experts": {name: weights[name].contiguous() for name in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")},
        "side_stream": torch.cuda.Stream(device),
        "side_stream_hi": torch.cuda.Stream(device, priority=torch.cuda.Stream.priority_range()[1]),  # one token
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    for op in (state["attn_boundary"], state["ffn_boundary"]):
        op._ensure_cached_weights(torch.bfloat16)
    state["grouped"] = _grouped_prepare(weights)  # grouped NVFP4 expert weights
    state["cbf"] = _laneb_prepare(state, weights)  # muse_bf16_proj block state
    state["d1_buf"] = torch.full((1 + TOP_K,), -1, dtype=torch.int32, device=device)  # one-token expert slot buffer
    # split-K partials (up to 16 splits) and per-tile arrival counters (each last arriver resets its own)
    state["d1s_part13"] = torch.empty(TOP_K * (D_EXPERT // BN) * 16 * 2 * BN, dtype=torch.float32, device=device)
    state["d1s_part2"] = torch.empty(TOP_K * (D_EXPERT // BN) * 16 * BN, dtype=torch.float32, device=device)
    state["d1s_cnt13"] = torch.zeros(TOP_K * (D_EXPERT // BN), dtype=torch.int32, device=device)
    state["d1s_cnt2"] = torch.zeros(TOP_K * (D_EXPERT // BN), dtype=torch.int32, device=device)
    state["iota8"] = torch.arange(TOP_K, dtype=torch.int32, device=device)
    _lt_prepare(state, device)  # pinned cuBLASLt plans
    torch.cuda.synchronize()
    return state


# ---- one-token NVFP4 expert GEMV (decode-1). Marlin runs its 8-row MMA tiles at ~60% of read bandwidth at
# one token; these two kernels read each routed expert's packed weights and scales once per slot and are
# bandwidth-bound. Same arithmetic contract as Marlin + silu_and_mul: exact E2M1 x E4M3 dequant to BF16, FP32
# accumulation, the expert's global scale in FP32, gate/up rounded to BF16 before silu(g) * u, BF16 out.
GEMV_MOE_MAX_TOKENS = 0  # off (lost 5-7% at decode-1 on the H200)


@triton.jit
def _moe_gemv_gate_up_kernel(X, W, SF, GS, IDS, OUT, stride_we, stride_wn, stride_se, stride_sn,
                             I: tl.constexpr, K: tl.constexpr, TOPK: tl.constexpr, BN: tl.constexpr,
                             BK: tl.constexpr, GSCALE: tl.constexpr, USE_PDL: tl.constexpr = False):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    slot = tl.program_id(0)
    e = tl.load(IDS + slot)
    if e < 0:
        return
    tok = slot // TOPK
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    kp = tl.arange(0, BK // 2)
    ks = tl.arange(0, BK // 16)
    kk = tl.arange(0, BK)
    wbase = W + e.to(tl.int64) * stride_we
    sbase = SF + e.to(tl.int64) * stride_se
    acc_g = tl.zeros((BN, BK), tl.float32)
    acc_u = tl.zeros((BN, BK), tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + tok * K + k0 + kk).to(tl.float32)
        pg = tl.load(wbase + rn[:, None] * stride_wn + (k0 // 2 + kp)[None, :])
        sg = tl.load(sbase + rn[:, None] * stride_sn + (k0 // 16 + ks)[None, :])
        acc_g += _dequant_asm(pg, sg, BN, BK).to(tl.float32) * x[None, :]
        pu = tl.load(wbase + (I + rn)[:, None] * stride_wn + (k0 // 2 + kp)[None, :])
        su = tl.load(sbase + (I + rn)[:, None] * stride_sn + (k0 // 16 + ks)[None, :])
        acc_u += _dequant_asm(pu, su, BN, BK).to(tl.float32) * x[None, :]
    gs = tl.load(GS + e) * GSCALE
    g = (tl.sum(acc_g, axis=1) * gs).to(tl.bfloat16).to(tl.float32)
    u = (tl.sum(acc_u, axis=1) * gs).to(tl.bfloat16).to(tl.float32)
    h = (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16)
    tl.store(OUT + slot.to(tl.int64) * I + rn, h)


@triton.jit
def _moe_gemv_down_kernel(A, W, SF, GS, IDS, OUT, stride_we, stride_wn, stride_se, stride_sn,
                          H: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                          GSCALE: tl.constexpr, USE_PDL: tl.constexpr = False):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    slot = tl.program_id(0)
    e = tl.load(IDS + slot)
    if e < 0:
        return
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    kp = tl.arange(0, BK // 2)
    ks = tl.arange(0, BK // 16)
    kk = tl.arange(0, BK)
    wbase = W + e.to(tl.int64) * stride_we
    sbase = SF + e.to(tl.int64) * stride_se
    acc = tl.zeros((BN, BK), tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(A + slot.to(tl.int64) * K + k0 + kk).to(tl.float32)
        p = tl.load(wbase + rn[:, None] * stride_wn + (k0 // 2 + kp)[None, :])
        s = tl.load(sbase + rn[:, None] * stride_sn + (k0 // 16 + ks)[None, :])
        acc += _dequant_asm(p, s, BN, BK).to(tl.float32) * a[None, :]
    y = (tl.sum(acc, axis=1) * (tl.load(GS + e) * GSCALE)).to(tl.bfloat16)
    tl.store(OUT + slot.to(tl.int64) * H + rn, y)


def _moe_gemv(x, local_ids, raw, bn1=8, bk1=512, bn2=8, bk2=512, w1=4, w2=4):
    """x bf16 [T, K] (T <= GEMV_MOE_MAX_TOKENS); local_ids int32 [T, TOPK] (-1 = not here). Returns y bf16
    [T, TOPK, H]; rows of non-local slots are left unwritten (the post-expert kernel skips them by id)."""
    T, K = x.shape
    w13, s13, w2_, s2 = raw["w13"], raw["w13_sf"], raw["w2"], raw["w2_sf"]
    I = w13.shape[1] // 2
    H = w2_.shape[1]
    slots = T * TOP_K
    ids = local_ids.reshape(-1)
    h = torch.empty((slots, I), device=x.device, dtype=torch.bfloat16)
    _moe_gemv_gate_up_kernel[(slots, I // bn1)](
        x, w13, s13, raw["w13_gs"], ids, h, w13.stride(0), w13.stride(1), s13.stride(0), s13.stride(1),
        I=I, K=K, TOPK=TOP_K, BN=bn1, BK=bk1, GSCALE=BIAS_FIX, num_warps=w1, **_PDL_KW)
    y = torch.empty((slots, H), device=x.device, dtype=torch.bfloat16)
    _moe_gemv_down_kernel[(slots, H // bn2)](
        h, w2_, s2, raw["w2_gs"], ids, y, w2_.stride(0), w2_.stride(1), s2.stride(0), s2.stride(1),
        H=H, K=I, BN=bn2, BK=bk2, GSCALE=BIAS_FIX, num_warps=w2, **_PDL_KW)
    return y.view(T, TOP_K, H)


# ---- at one token the pre-expert RMSNorm and local-id mapping run inside the gate/up GEMV (one dependent
# kernel fewer on the critical chain; Marlin's block alignment is not needed by the GEMV).
@triton.jit
def _moe_gemv_norm_gate_up_kernel(X, NW, eps, TOPK_IDS, LS, LOCAL, W, SF, GS, OUT,
                                  stride_we, stride_wn, stride_se, stride_sn,
                                  I: tl.constexpr, K: tl.constexpr, TOPK: tl.constexpr, N_EXP: tl.constexpr,
                                  N_LOCAL: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                                  GSCALE: tl.constexpr, USE_PDL: tl.constexpr = False):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    slot = tl.program_id(0)
    rel = (tl.load(TOPK_IDS + slot) - tl.load(LS) + N_EXP) % N_EXP
    e = tl.where(rel < N_LOCAL, rel, -1)
    if tl.program_id(1) == 0:
        tl.store(LOCAL + slot, e.to(tl.int32))
    if e < 0:
        return
    tok = slot // TOPK
    offs = tl.arange(0, K)
    xf = tl.load(X + tok * K + offs).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(xf * xf, axis=0) / K + eps)  # F.rms_norm, as the pre-expert norm kernel
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    kp = tl.arange(0, BK // 2)
    ks = tl.arange(0, BK // 16)
    kk = tl.arange(0, BK)
    wbase = W + e.to(tl.int64) * stride_we
    sbase = SF + e.to(tl.int64) * stride_se
    acc_g = tl.zeros((BN, BK), tl.float32)
    acc_u = tl.zeros((BN, BK), tl.float32)
    for k0 in range(0, K, BK):
        xr = tl.load(X + tok * K + k0 + kk).to(tl.float32)
        x = (xr * rstd * tl.load(NW + k0 + kk).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        pg = tl.load(wbase + rn[:, None] * stride_wn + (k0 // 2 + kp)[None, :])
        sg = tl.load(sbase + rn[:, None] * stride_sn + (k0 // 16 + ks)[None, :])
        acc_g += _dequant_asm(pg, sg, BN, BK).to(tl.float32) * x[None, :]
        pu = tl.load(wbase + (I + rn)[:, None] * stride_wn + (k0 // 2 + kp)[None, :])
        su = tl.load(sbase + (I + rn)[:, None] * stride_sn + (k0 // 16 + ks)[None, :])
        acc_u += _dequant_asm(pu, su, BN, BK).to(tl.float32) * x[None, :]
    gs = tl.load(GS + e) * GSCALE
    g = (tl.sum(acc_g, axis=1) * gs).to(tl.bfloat16).to(tl.float32)
    u = (tl.sum(acc_u, axis=1) * gs).to(tl.bfloat16).to(tl.float32)
    h = (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16)
    tl.store(OUT + slot.to(tl.int64) * I + rn, h)


def _experts_one_token(x, norm_w, eps, topk_ids, local_start, topk_weights, state, bn=8, bk=512, warps=4):
    """Pre-expert norm + local ids + expert GEMVs + post-expert finalize for one token."""
    raw = state["raw_experts"]
    T, K = x.shape
    w13, s13, w2_, s2 = raw["w13"], raw["w13_sf"], raw["w2"], raw["w2_sf"]
    I = w13.shape[1] // 2
    H = w2_.shape[1]
    slots = T * TOP_K
    local = torch.empty((T, TOP_K), device=x.device, dtype=torch.int32)
    h = torch.empty((slots, I), device=x.device, dtype=torch.bfloat16)
    _moe_gemv_norm_gate_up_kernel[(slots, I // bn)](
        x, norm_w, eps, topk_ids, local_start, local, w13, s13, raw["w13_gs"], h,
        w13.stride(0), w13.stride(1), s13.stride(0), s13.stride(1),
        I=I, K=K, TOPK=TOP_K, N_EXP=N_EXPERTS, N_LOCAL=LOCAL_EXPERTS, BN=bn, BK=bk, GSCALE=BIAS_FIX,
        num_warps=warps, **_PDL_KW)
    y = torch.empty((slots, H), device=x.device, dtype=torch.bfloat16)
    _moe_gemv_down_kernel[(slots, H // bn)](
        h, w2_, s2, raw["w2_gs"], local, y, w2_.stride(0), w2_.stride(1), s2.stride(0), s2.stride(1),
        H=H, K=I, BN=bn, BK=bk, GSCALE=BIAS_FIX, num_warps=warps, **_PDL_KW)
    return _post_expert_weighted_sum(y.view(T, TOP_K, H), topk_weights.to(torch.float32).contiguous(),
                                     state["post_expert_w"], local)


# ---- grouped NVFP4 expert kernels (_route takes local_start by pointer) ----
SCALE_BIAS = gl.constexpr(2.0 ** 119)
ACC_BIAS = gl.constexpr(2.0 ** 7)
BM = 16
D1_KS = (16, 8)
W2OV_MODE = 2  # one-token W2 K-loop schedule (1 = activation staging hoisted, 2 = also two decodes first), fixed constant; D1_KS: K splits of the direct one-token W13 / W2
BN = 64
BK = 256
NUM_WARPS = 4
XS_MIN_M = 1
PF_MAX_M = 1  # weight prefetch (register double buffer) at decode 1 only
SMB_MIN_M = 4096  # shared-B (wgmma) path off: the register path is faster on this layer's routing


@triton.jit
def _route(ids_ptr, local_ptr, sorted_ptr, work_ptr, nwork_ptr, P, LS_PTR,
           LOCAL: tl.constexpr, BM: tl.constexpr, P_BLOCK: tl.constexpr, W_MAX: tl.constexpr):
    offs = tl.arange(0, P_BLOCK)
    pm = offs < P
    gid = tl.load(ids_ptr + offs, mask=pm, other=-1)
    lid = gid - tl.load(LS_PTR)  # this layer passes local_start as an int32 [1] tensor
    live = pm & (lid >= 0) & (lid < LOCAL)
    lid = tl.where(live, lid, -1)
    tl.store(local_ptr + offs, lid, mask=pm)
    # Pairs grouped by local expert, in pair order within an expert (deterministic); non-local pairs last.
    key = tl.where(live, lid * P_BLOCK + offs, LOCAL * P_BLOCK + offs)
    skey = tl.sort(key)
    tl.store(sorted_ptr + offs, skey % P_BLOCK, mask=pm)
    counts = tl.histogram(tl.maximum(lid, 0), LOCAL, mask=live)
    starts = tl.cumsum(counts, 0) - counts
    chunks = (counts + BM - 1) // BM
    coff = tl.cumsum(chunks, 0) - chunks
    total = tl.sum(chunks, 0)
    e_ar = tl.arange(0, LOCAL)
    w = tl.arange(0, W_MAX)
    sel = (coff[None, :] <= w[:, None]) & ((coff + chunks)[None, :] > w[:, None])
    owner = tl.sum(tl.where(sel, e_ar[None, :], 0), 1)
    c_first = tl.sum(tl.where(sel, coff[None, :], 0), 1)
    e_start = tl.sum(tl.where(sel, starts[None, :], 0), 1)
    e_cnt = tl.sum(tl.where(sel, counts[None, :], 0), 1)
    r0 = e_start + (w - c_first) * BM
    r1 = tl.minimum(r0 + BM, e_start + e_cnt)
    wm = w < total
    tl.store(work_ptr + w * 3 + 0, owner, mask=wm)
    tl.store(work_ptr + w * 3 + 1, r0, mask=wm)
    tl.store(work_ptr + w * 3 + 2, r1, mask=wm)
    tl.store(nwork_ptr, total)


_DECODE_ASM = gl.constexpr("""
{
.reg .b32 t, u;
and.b32 t, $4, 0x81C081C0;
mul.rn.bf16x2 $0, t, $6;
shf.l.wrap.b32 t, $4, $4, 4;
and.b32 t, t, 0x81C081C0;
mul.rn.bf16x2 $1, t, $6;
shf.l.wrap.b32 t, $4, $4, 10;
and.b32 t, t, 0x81C081C0;
mul.rn.bf16x2 $2, t, $6;
shf.l.wrap.b32 t, $4, $4, 6;
shf.l.wrap.b32 u, $4, $4, 14;
and.b32 t, t, 0x80C080C0;
and.b32 u, u, 0x01000100;
or.b32 t, t, u;
mul.rn.bf16x2 $3, t, $6;
}
""")
_DECODE_ASM16 = gl.constexpr("""
{
.reg .b32 t, u;
and.b32 t, $8, 0x81C081C0;
mul.rn.bf16x2 $0, t, $24;
and.b32 t, $10, 0x81C081C0;
mul.rn.bf16x2 $1, t, $25;
shf.l.wrap.b32 t, $8, $8, 4;
and.b32 t, t, 0x81C081C0;
mul.rn.bf16x2 $2, t, $24;
shf.l.wrap.b32 t, $10, $10, 4;
and.b32 t, t, 0x81C081C0;
mul.rn.bf16x2 $3, t, $25;
shf.l.wrap.b32 t, $8, $8, 10;
and.b32 t, t, 0x81C081C0;
mul.rn.bf16x2 $4, t, $24;
shf.l.wrap.b32 t, $10, $10, 10;
and.b32 t, t, 0x81C081C0;
mul.rn.bf16x2 $5, t, $25;
shf.l.wrap.b32 t, $8, $8, 6;
shf.l.wrap.b32 u, $8, $8, 14;
and.b32 t, t, 0x80C080C0;
and.b32 u, u, 0x01000100;
or.b32 t, t, u;
mul.rn.bf16x2 $6, t, $24;
shf.l.wrap.b32 t, $10, $10, 6;
shf.l.wrap.b32 u, $10, $10, 14;
and.b32 t, t, 0x80C080C0;
and.b32 u, u, 0x01000100;
or.b32 t, t, u;
mul.rn.bf16x2 $7, t, $25;
}
""")


# Layouts (CTA of 4 warps, BN = 64 weight rows, BM = 16 token rows, BK = 256).
MMA_L = gl.constexpr(gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]))
A_L = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MMA_L.value, k_width=2))
B_L = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=MMA_L.value, k_width=2))
# packed words [BN rows, 4 (q), 8 (j)]: j in registers (contiguous 32 bytes), q in lanes
L3 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [8, 0, 0]],
    lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
    warp_bases=[[16, 0, 0], [32, 0, 0]], block_bases=[], shape=[64, 4, 8]))
# Decode input/output [BN rows, 4 (q), 8 (word), 2 (b0), 2 (b2), 2 (b3)]: the 16 elements of one asm call are, in
# register order, (b0, row + 8, b2, b3): the MMA A quads of two k16 steps for rows r and r + 8.
LW16 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 0, 1, 0, 0], [8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
               [0, 0, 1, 0, 0, 0], [0, 0, 2, 0, 0, 0], [0, 0, 4, 0, 0, 0]],
    lane_bases=[[0, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0]],
    warp_bases=[[16, 0, 0, 0, 0, 0], [32, 0, 0, 0, 0, 0]], block_bases=[], shape=[64, 4, 8, 2, 2, 2]))
# The same decode for one K chunk of 64: words [BN, 4 (q), 2 (word)], so a chunk's decoded A operand is 16
# registers per thread instead of 64.
LW16C = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 0, 1, 0, 0], [8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0]],
    lane_bases=[[0, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0]],
    warp_bases=[[16, 0, 0, 0, 0, 0], [32, 0, 0, 0, 0, 0]], block_bases=[], shape=[64, 4, 2, 2, 2, 2]))
# block-scale bytes [BN, 4 (q), 4 (g)]
LS = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [8, 0, 0]],
    lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
    warp_bases=[[16, 0, 0], [32, 0, 0]], block_bases=[], shape=[64, 4, 4]))
# activations [BM token rows, 4 (q), 64 (8j + i)]
LX = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [8, 0, 0]],
    lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
    warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[16, 4, 64]))
# The same activation tile viewed as [64 rows (4n + q), 64]: per-warp read layout from shared memory ...
LX2 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0]],
    lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
    warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[64, 64]))
LX8 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]],
    lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
    warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[8, 4, 64]))
LX2_8 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]],
    lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
    warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[32, 64]))
LX32 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [8, 0, 0], [16, 0, 0]],
    lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
    warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[32, 4, 64]))
LX2_32 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0], [64, 0]],
    lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
    warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[128, 64]))
# ... cooperative global load layout (each thread 16 contiguous bytes) ...
LXG = gl.constexpr(gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[4, 1], order=[1, 0]))
# ... and the shared buffer, 16-byte chunks XOR-swizzled by row so a quarter warp's rows hit distinct banks.
X_SMEM = gl.constexpr(gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0]))
# Shared-B mode: each K block of activations is stored once as [BM, BK] in the MMA's logical K order, 128-byte
# swizzled. On sm_90 (H200) the GEMMs read it with wgmma, the decoded weights being wgmma's register A operand (the
# k_width-2 operand layout below is the same register fragment); elsewhere mma_v2 reads it.
XB_SMEM = gl.constexpr(gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2))
MMA3_8 = gl.constexpr(gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 8, 16]))
MMA3_16 = gl.constexpr(gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 16, 16]))
A3_8 = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MMA3_8.value, k_width=2))
A3_16 = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MMA3_16.value, k_width=2))


@gluon.jit
def _idx3(D0: gl.constexpr, D1: gl.constexpr, D2: gl.constexpr, L: gl.constexpr):
    a = gl.arange(0, D0, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    b = gl.arange(0, D1, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    c = gl.arange(0, D2, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    return (gl.expand_dims(gl.expand_dims(a, 1), 2), gl.expand_dims(gl.expand_dims(b, 0), 2),
            gl.expand_dims(gl.expand_dims(c, 0), 1))


@gluon.jit
def _x8(v):
    v2 = gl.join(v, v)
    v4 = gl.join(v2, v2)
    return gl.join(v4, v4)


@gluon.jit
def _decode_tile(w3, sb, BN: gl.constexpr, BK: gl.constexpr, L3: gl.constexpr, A_L: gl.constexpr):
    """Packed words [BN, 4, 8] + scale bytes [BN, 4, 4] -> decoded A operand [BN, BK] (logical K
    k' = 32 w + 16 b3 + 8 b2 + 2 q + b0 <-> physical K 64 q + 8 w + 4 b3 + 2 b2 + b0)."""
    s = (sb.to(gl.float8e4nv, bitcast=True).to(gl.float32) * SCALE_BIAS).to(gl.bfloat16)
    s8 = gl.reshape(gl.join(s, s), [BN, 4, 8])
    w16 = gl.convert_layout(_x8(w3), LW16)
    s16 = gl.convert_layout(_x8(s8), LW16)
    t = gl.inline_asm_elementwise(_DECODE_ASM16, "=r,=r,=r,=r,=r,=r,=r,=r," + ",".join(["r"] * 24), [w16, s16],
                                  dtype=gl.bfloat16, is_pure=True, pack=16)
    t = gl.reshape(gl.permute(t, (0, 2, 5, 4, 1, 3)), [BN, BK])
    return gl.convert_layout(t, A_L, assert_trivial=True)


@gluon.jit
def _wsplit4(w3, BN: gl.constexpr):
    """Packed words [BN, 4, 8] (word w = 4 c1 + 2 c0 + wl) -> the four K chunks c = 2 c1 + c0, each [BN, 4, 2]."""
    lo, hi = gl.split(gl.permute(gl.reshape(w3, [BN, 4, 2, 2, 2]), (0, 1, 4, 2, 3)))
    c0, c2 = gl.split(lo)
    c1, c3 = gl.split(hi)
    return c0, c1, c2, c3


@gluon.jit
def _ssplit4(sb, BN: gl.constexpr):
    """Scale bytes [BN, 4, 4] (group g = chunk c = 2 c1 + c0) -> four [BN, 4, 1]."""
    lo, hi = gl.split(gl.reshape(sb, [BN, 4, 2, 2]))
    c0, c2 = gl.split(lo)
    c1, c3 = gl.split(hi)
    return gl.expand_dims(c0, 2), gl.expand_dims(c1, 2), gl.expand_dims(c2, 2), gl.expand_dims(c3, 2)


@gluon.jit
def _mma_chunk(wc, sc, xb_s, C: gl.constexpr, acc, BN: gl.constexpr, a_l: gl.constexpr, HOP: gl.constexpr):
    """Decode K chunk C (64 logical K) and multiply it with the matching 64 columns of the shared activation tile."""
    s = (sc.to(gl.float8e4nv, bitcast=True).to(gl.float32) * SCALE_BIAS).to(gl.bfloat16)
    s2 = gl.reshape(gl.join(s, s), [BN, 4, 2])
    w16 = gl.convert_layout(_x8(wc), LW16C)
    s16 = gl.convert_layout(_x8(s2), LW16C)
    t = gl.inline_asm_elementwise(_DECODE_ASM16, "=r,=r,=r,=r,=r,=r,=r,=r," + ",".join(["r"] * 24), [w16, s16],
                                  dtype=gl.bfloat16, is_pure=True, pack=16)
    a = gl.convert_layout(gl.reshape(gl.permute(t, (0, 2, 5, 4, 1, 3)), [BN, 64]), a_l)
    bv = xb_s.slice(64 * C, 64, dim=1).permute((1, 0))
    if HOP:
        acc = warpgroup_mma(a, bv, acc)
    else:
        acc = mma_v2(a, bv.load(B_L), acc)
    return acc


@gluon.jit
def _xg_offsets(sorted_ptr, r0, r1, K: gl.constexpr, BY_TOKEN: gl.constexpr, BM: gl.constexpr):
    """Offsets/mask [4 BM, 64] in LXG for the cooperative load: row 4n + q, column c -> x[row_n, 64q + c]."""
    rr = gl.arange(0, 4 * BM, layout=gl.SliceLayout(1, LXG))
    cc = gl.arange(0, 64, layout=gl.SliceLayout(0, LXG))
    rv = r0 + rr // 4
    pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
    row = pair // 8 if BY_TOKEN else pair
    off = gl.expand_dims(row * K + (rr % 4) * 64, 1) + gl.expand_dims(cc, 0)
    msk = gl.expand_dims(rv < r1, 1) & gl.expand_dims(cc >= 0, 0)
    return off, msk


@gluon.jit
def _x_shared(x_ptr, off, msk, k0, x_s, BM: gl.constexpr, BK: gl.constexpr):
    xg = gl.load(x_ptr + off + k0, mask=msk, other=0.0)
    gl.barrier()
    x_s.store(xg)
    gl.barrier()
    if BM == 8:
        xv = gl.reshape(x_s.load(LX2_8), [BM, 4, 64])
    elif BM == 32:
        xv = gl.reshape(x_s.load(LX2_32), [BM, 4, 64])
    else:
        xv = gl.reshape(x_s.load(LX2), [BM, 4, 64])
    return _x_tile(xv, BM, BK, B_L)

@gluon.jit
def _x_shared_n(x_ptr, off, msk, k0, x_s, BM: gl.constexpr, BK: gl.constexpr, nw_ptr, rstd, K: gl.constexpr,
                dbg_ptr, DBG: gl.constexpr, dbg_on):
    """_x_shared on the RAW pooler row: stages bf16((x * rstd) * w) exactly as the norm kernel's
    output (x and w extended from bf16, one rounding); rows outside the item stay +0.0, as the unfused path's masked load."""
    xg = gl.load(x_ptr + off + k0, mask=msk, other=0.0)
    wv = gl.load(nw_ptr + (off % K) + k0)
    xn = ((xg.to(gl.float32) * rstd) * wv.to(gl.float32)).to(gl.bfloat16)
    xn = gl.where(msk, xn, gl.zeros_like(xn))
    if DBG:
        if dbg_on:
            gl.store(dbg_ptr + (off % K) + k0, xn, mask=msk)
    gl.barrier()
    x_s.store(xn)
    gl.barrier()
    if BM == 8:
        xv = gl.reshape(x_s.load(LX2_8), [BM, 4, 64])
    elif BM == 32:
        xv = gl.reshape(x_s.load(LX2_32), [BM, 4, 64])
    else:
        xv = gl.reshape(x_s.load(LX2), [BM, 4, 64])
    return _x_tile(xv, BM, BK, B_L)



@gluon.jit
def _xb_store(x_ptr, off, msk, k0, xb_s, BM: gl.constexpr, BK: gl.constexpr):
    """Cooperative load of one K block of activations, stored to xb_s as [n, k' = 32 w + 16 b3 + 8 b2 + 2 q + b0]."""
    xg = gl.load(x_ptr + off + k0, mask=msk, other=0.0)  # [4 BM, 64]: row 4 n + q, column 8 w + 4 b3 + 2 b2 + b0
    xl = gl.reshape(gl.permute(gl.reshape(xg, [BM, 4, 8, 2, 2, 2]), (0, 2, 3, 4, 1, 5)), [BM, BK])
    gl.barrier()
    xb_s.store(xl)
    fence_async_shared()
    gl.barrier()


@gluon.jit
def _x_tile(xv, BM: gl.constexpr, BK: gl.constexpr, B_L: gl.constexpr):
    # [n, q, 64 physical = 8 w + 4 b3 + 2 b2 + b0] -> [k' = 32 w + 16 b3 + 8 b2 + 2 q + b0, n]
    xb = gl.reshape(gl.permute(gl.reshape(xv, [BM, 4, 8, 2, 2, 2]), (2, 3, 4, 1, 5, 0)), [BK, BM])
    return gl.convert_layout(xb, B_L, assert_trivial=BM != 32)


@gluon.jit
def _w13_silu(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr,
              K: gl.constexpr, I: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr,
              XS: gl.constexpr, SMB: gl.constexpr = False, HOP: gl.constexpr = False,
              CHK: gl.constexpr = False, PF: gl.constexpr = False,
              DIRECT: gl.constexpr = False, NLOC: gl.constexpr = 32, IL: gl.constexpr = False):
    mma: gl.constexpr = (MMA3_8 if BM == 8 else MMA3_16) if HOP else MMA_L
    a_l: gl.constexpr = (A3_8 if BM == 8 else A3_16) if HOP else A_L
    tl.extra.cuda.gdc_wait()  # wait for the producer's writes before any dependent read (launch_pdl=True launch)
    tl.extra.cuda.gdc_launch_dependents()  # trigger at kernel start (dependents keep their gdc_wait)
    pid_n = gl.program_id(0)
    pid_w = gl.program_id(1)
    if DIRECT:  # M = 1: slot pid_w of the only token is one item of one row; no routing pass
        e = gl.load(work_ptr + pid_w) - gl.load(nwork_ptr)
        active = (e >= 0) & (e < NLOC)
        e = e * active.to(gl.int32)
        r0 = pid_w
        r1 = pid_w + active.to(gl.int32)
        if pid_n == 0:  # this slot's local expert id, or -1, for the post-expert norm
            gl.store(nwork_ptr + 1 + pid_w, e + active.to(gl.int32) - 1)
    else:
        active = pid_w < gl.load(nwork_ptr)
        e = gl.load(work_ptr + pid_w * 3 + 0)
        r0 = gl.load(work_ptr + pid_w * 3 + 1)
        r1 = gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        n0 = pid_n * BN
        rw, qw, jw = _idx3(BN, 4, 8, L3)
        KB: gl.constexpr = K // BK  # tile-major weights, see _tile_major
        BLKW: gl.constexpr = BN * 36 if IL else BN * 32  # IL: codes then scales per 9216 B block
        BLKB: gl.constexpr = BN * 144 if IL else BN * 16
        S0: gl.constexpr = BN * 128 if IL else 0
        w_off = (e * (2 * I // BN) + pid_n) * KB * BLKW + rw * 32 + qw * 8 + jw
        rs, qs, gs_ = _idx3(BN, 4, 4, LS)
        s_off = (e * (2 * I // BN) + pid_n) * KB * BLKB + S0 + rs * 16 + qs * 4 + gs_
        mx, qx, cx = _idx3(BM, 4, 64, LX8 if BM == 8 else LX)
        rv = r0 + mx
        pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
        x_off = (pair // 8) * K + qx * 64 + cx
        x_mask = (rv < r1) & (cx >= 0)
        xoff, xmsk = _xg_offsets(sorted_ptr, r0, r1, K, True, BM)
        x_s = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 64], X_SMEM)
        xb_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], XB_SMEM)
        wp = w_ptr.to(gl.pointer_type(gl.int32))
        acc_g = gl.zeros([BN, BM], gl.float32, layout=mma)
        acc_u = gl.zeros([BN, BM], gl.float32, layout=mma)
        if PF:  # register double buffer: block k0 + BK is requested before block k0 is decoded
            wg_n = gl.load(wp + w_off)
            sg_n = gl.load(sf_ptr + s_off)
            wu_n = gl.load(wp + w_off + (I // BN) * KB * BLKW)
            su_n = gl.load(sf_ptr + s_off + (I // BN) * KB * BLKB)
        for k0 in range(0, K, BK):
            if PF:
                wg = wg_n
                sg = sg_n
                wu = wu_n
                su = su_n
                kn = k0 + BK
                wm = (rw >= 0) & (kn < K)
                sm = (rs >= 0) & (kn < K)
                wg_n = gl.load(wp + w_off + (kn // BK) * BLKW, mask=wm, other=0)
                sg_n = gl.load(sf_ptr + s_off + (kn // BK) * BLKB, mask=sm)
                wu_n = gl.load(wp + w_off + (I // BN) * KB * BLKW + (kn // BK) * BLKW, mask=wm, other=0)
                su_n = gl.load(sf_ptr + s_off + (I // BN) * KB * BLKB + (kn // BK) * BLKB, mask=sm)
            else:
                wg = gl.load(wp + w_off + (k0 // BK) * BLKW)
                sg = gl.load(sf_ptr + s_off + (k0 // BK) * BLKB)
                wu = gl.load(wp + w_off + (I // BN) * KB * BLKW + (k0 // BK) * BLKW)
                su = gl.load(sf_ptr + s_off + (I // BN) * KB * BLKB + (k0 // BK) * BLKB)
            if SMB:
                _xb_store(x_ptr, xoff, xmsk, k0, xb_s, BM, BK)
                if CHK:
                    g0, g1, g2, g3 = _wsplit4(wg, BN)
                    h0, h1, h2, h3 = _ssplit4(sg, BN)
                    acc_g = _mma_chunk(g0, h0, xb_s, 0, acc_g, BN, a_l, HOP)
                    acc_g = _mma_chunk(g1, h1, xb_s, 1, acc_g, BN, a_l, HOP)
                    acc_g = _mma_chunk(g2, h2, xb_s, 2, acc_g, BN, a_l, HOP)
                    acc_g = _mma_chunk(g3, h3, xb_s, 3, acc_g, BN, a_l, HOP)
                    u0, u1, u2, u3 = _wsplit4(wu, BN)
                    v0, v1, v2, v3 = _ssplit4(su, BN)
                    acc_u = _mma_chunk(u0, v0, xb_s, 0, acc_u, BN, a_l, HOP)
                    acc_u = _mma_chunk(u1, v1, xb_s, 1, acc_u, BN, a_l, HOP)
                    acc_u = _mma_chunk(u2, v2, xb_s, 2, acc_u, BN, a_l, HOP)
                    acc_u = _mma_chunk(u3, v3, xb_s, 3, acc_u, BN, a_l, HOP)
                elif HOP:
                    acc_g = warpgroup_mma(_decode_tile(wg, sg, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc_g)
                    acc_u = warpgroup_mma(_decode_tile(wu, su, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc_u)
                else:
                    xb = xb_s.permute((1, 0)).load(B_L)
                    acc_g = mma_v2(_decode_tile(wg, sg, BN, BK, L3, A_L), xb, acc_g)
                    acc_u = mma_v2(_decode_tile(wu, su, BN, BK, L3, A_L), xb, acc_u)
            else:
                if XS:
                    xb = _x_shared(x_ptr, xoff, xmsk, k0, x_s, BM, BK)
                else:
                    xb = _x_tile(gl.load(x_ptr + x_off + k0, mask=x_mask, other=0.0), BM, BK, B_L)
                acc_g = mma_v2(_decode_tile(wg, sg, BN, BK, L3, A_L), xb, acc_g)
                acc_u = mma_v2(_decode_tile(wu, su, BN, BK, L3, A_L), xb, acc_u)
        g = gl.load(gs_ptr + e) * ACC_BIAS
        hg = (acc_g * g).to(gl.bfloat16).to(gl.float32)
        hu = (acc_u * g).to(gl.bfloat16).to(gl.float32)
        inter = (hg / (1.0 + gl.exp(-hg))).to(gl.bfloat16).to(gl.float32) * hu
        n_o = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
        m_o = gl.arange(0, BM, layout=gl.SliceLayout(0, mma))
        ro = r0 + m_o
        pair_o = gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
        optr = out_ptr + gl.expand_dims(pair_o, 0) * I + gl.expand_dims(n0 + n_o, 1)
        gl.store(optr, inter.to(gl.bfloat16), mask=gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1))


@gluon.jit
def _gu_split(t, BN: gl.constexpr, BM: gl.constexpr):
    """[BN, BM] with rows 16 s + 8 g + i -> (g = 0 part, g = 1 part), each [BN / 16, 8, BM] (row 8 apart: one thread)."""
    return gl.split(gl.permute(gl.reshape(t, [BN // 16, 2, 8, BM]), (0, 2, 3, 1)))


@gluon.jit
def _w2(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr,
        K: gl.constexpr, N: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr, XS: gl.constexpr,
        SMB: gl.constexpr = False, HOP: gl.constexpr = False,
              CHK: gl.constexpr = False, PF: gl.constexpr = False,
              DIRECT: gl.constexpr = False, NLOC: gl.constexpr = 32, GU: gl.constexpr = False,
        IL: gl.constexpr = False, KS: gl.constexpr = 1):
    mma: gl.constexpr = (MMA3_8 if BM == 8 else MMA3_16) if HOP else MMA_L
    a_l: gl.constexpr = (A3_8 if BM == 8 else A3_16) if HOP else A_L
    tl.extra.cuda.gdc_wait()  # wait for the producer's writes before any dependent read (launch_pdl=True launch)
    tl.extra.cuda.gdc_launch_dependents()  # trigger at kernel start (dependents keep their gdc_wait)
    pid_n = gl.program_id(0)
    pid_w = gl.program_id(1)
    if DIRECT:  # M = 1: slot pid_w of the only token is one item of one row; no routing pass
        e = gl.load(work_ptr + pid_w) - gl.load(nwork_ptr)
        active = (e >= 0) & (e < NLOC)
        e = e * active.to(gl.int32)
        r0 = pid_w
        r1 = pid_w + active.to(gl.int32)
        if GU:
            if pid_n == 0:  # this slot's local expert id, or -1, for the post-expert norm
                gl.store(nwork_ptr + 1 + pid_w, e + active.to(gl.int32) - 1)
    else:
        active = pid_w < gl.load(nwork_ptr)
        e = gl.load(work_ptr + pid_w * 3 + 0)
        r0 = gl.load(work_ptr + pid_w * 3 + 1)
        r1 = gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        n0 = pid_n * BN
        rw, qw, jw = _idx3(BN, 4, 8, L3)
        KB: gl.constexpr = K // BK  # tile-major weights
        BLKW: gl.constexpr = BN * 36 if IL else BN * 32
        BLKB: gl.constexpr = BN * 144 if IL else BN * 16
        S0: gl.constexpr = BN * 128 if IL else 0
        w_off = (e * (N // BN) + pid_n) * KB * BLKW + rw * 32 + qw * 8 + jw
        rs, qs, gs_ = _idx3(BN, 4, 4, LS)
        s_off = (e * (N // BN) + pid_n) * KB * BLKB + S0 + rs * 16 + qs * 4 + gs_
        mx, qx, cx = _idx3(BM, 4, 64, LX8 if BM == 8 else LX)
        rv = r0 + mx
        pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
        x_off = (pair // 8 if GU else pair) * K + qx * 64 + cx
        x_mask = (rv < r1) & (cx >= 0)
        xoff, xmsk = _xg_offsets(sorted_ptr, r0, r1, K, GU, BM)
        x_s = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 64], X_SMEM)
        xb_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], XB_SMEM)
        wp = w_ptr.to(gl.pointer_type(gl.int32))
        acc = gl.zeros([BN, BM], gl.float32, layout=mma)
        if PF:
            wq_n = gl.load(wp + w_off)
            sq_n = gl.load(sf_ptr + s_off)
        if KS > 1:  # this CTA's K range
            k_lo = gl.program_id(2) * (K // KS)
        else:
            k_lo = 0
        for k0 in range(k_lo, k_lo + K // KS, BK):
            if PF:
                wq = wq_n
                sq = sq_n
                kn = k0 + BK
                wq_n = gl.load(wp + w_off + (kn // BK) * BLKW, mask=(rw >= 0) & (kn < K), other=0)
                sq_n = gl.load(sf_ptr + s_off + (kn // BK) * BLKB, mask=(rs >= 0) & (kn < K))
            else:
                wq = gl.load(wp + w_off + (k0 // BK) * BLKW)
                sq = gl.load(sf_ptr + s_off + (k0 // BK) * BLKB)
            if SMB:
                _xb_store(a_ptr, xoff, xmsk, k0, xb_s, BM, BK)
                if CHK:
                    q0, q1, q2, q3 = _wsplit4(wq, BN)
                    p0, p1, p2, p3 = _ssplit4(sq, BN)
                    acc = _mma_chunk(q0, p0, xb_s, 0, acc, BN, a_l, HOP)
                    acc = _mma_chunk(q1, p1, xb_s, 1, acc, BN, a_l, HOP)
                    acc = _mma_chunk(q2, p2, xb_s, 2, acc, BN, a_l, HOP)
                    acc = _mma_chunk(q3, p3, xb_s, 3, acc, BN, a_l, HOP)
                elif HOP:
                    acc = warpgroup_mma(_decode_tile(wq, sq, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc)
                else:
                    acc = mma_v2(_decode_tile(wq, sq, BN, BK, L3, A_L), xb_s.permute((1, 0)).load(B_L), acc)
            else:
                if XS:
                    xb = _x_shared(a_ptr, xoff, xmsk, k0, x_s, BM, BK)
                else:
                    xb = _x_tile(gl.load(a_ptr + x_off + k0, mask=x_mask, other=0.0), BM, BK, B_L)
                acc = mma_v2(_decode_tile(wq, sq, BN, BK, L3, A_L), xb, acc)
        y = acc * (gl.load(gs_ptr + e) * ACC_BIAS)
        n_o = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
        m_o = gl.arange(0, BM, layout=gl.SliceLayout(0, mma))
        ro = r0 + m_o
        pair_o = gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
        if GU:  # rows 16 s + i are gate column 8 s + i of this tile's 32, rows 16 s + 8 + i its up rows
            rows2 = gl.expand_dims(n_o, 1) + gl.expand_dims(m_o * 0, 0)
            pair2 = gl.expand_dims(pair_o, 0) + gl.expand_dims(n_o * 0, 1)
            msk2 = gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1)
            hg, hu = _gu_split(y, BN, BM)
            rg, _ru = _gu_split(rows2, BN, BM)
            pg, _pu = _gu_split(pair2, BN, BM)
            mg, _mu = _gu_split(msk2.to(gl.int32), BN, BM)
            hg = hg.to(gl.bfloat16).to(gl.float32)
            hu = hu.to(gl.bfloat16).to(gl.float32)
            v = (hg / (1.0 + gl.exp(-hg))).to(gl.bfloat16).to(gl.float32) * hu
            col = pid_n * (BN // 2) + (rg // 16) * 8 + rg % 8
            gl.store(out_ptr + pg * (N // 2) + col, v.to(gl.bfloat16), mask=mg != 0)
        elif KS > 1:  # FP32 partial row [pair, split, N]
            optr = out_ptr + gl.expand_dims(pair_o, 0) * (KS * N) + gl.program_id(2) * N + gl.expand_dims(n0 + n_o, 1)
            gl.store(optr, y, mask=gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1))
        else:
            optr = out_ptr + gl.expand_dims(pair_o, 0) * N + gl.expand_dims(n0 + n_o, 1)
            gl.store(optr, y.to(gl.bfloat16), mask=gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1))
    if DIRECT and not GU:
        if r1 == r0:  # non-local slot: zero output row, which the post-expert norm maps to zero
            zl: gl.constexpr = gl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])
            zo = gl.arange(0, BN, layout=zl)
            gl.store(out_ptr + pid_w * N + pid_n * BN + zo, gl.zeros([BN], gl.bfloat16, layout=zl))


def _src_bit(j, t):
    """Source bit in the stored word of target bit t of decoded output j (see _DECODE_ASM)."""
    if j == 0:
        return t
    if j == 1:
        return (t - 4) % 32
    if j == 2:
        return (t - 10) % 32
    return (t - 6) % 32 if (0x80C080C0 >> t) & 1 else (t - 14) % 32


def _permute_bits(packed):
    """uint8 [..., K/2] (two E2M1 codes per byte, low nibble first) -> the same codes with each 32-bit word's bits
    permuted (lossless): natural code c of the word goes to decoded output c // 2, half c % 2."""
    words = packed.contiguous().view(torch.int32)
    out = torch.zeros_like(words)
    for c in range(8):
        j, h = c // 2, c % 2
        for bit, tgt in enumerate((6, 7, 8, 15)):
            out |= ((words >> (4 * c + bit)) & 1) << _src_bit(j, tgt + 16 * h)
    return out.view(torch.uint8).view(packed.shape)


def _tile_major(codes, sf):
    """Codes uint8 [L, rows, K/2] and fp8 scales [L, rows, K/16] -> the same bytes ordered tile-major, so that
    each (expert, BN-row tile, BK block) is one contiguous BN x 128 B run of codes and one BN x 16 B run of scales
    (a lossless permutation of bytes; contiguous blocks stream from DRAM better than BN rows 2 KB apart)."""
    n_l, rows, half_k = codes.shape
    kb = 2 * half_k // BK
    c = codes.view(n_l, rows // BN, BN, kb, BK // 2).permute(0, 1, 3, 2, 4).contiguous()
    s = sf.contiguous().view(n_l, rows // BN, BN, kb, BK // 16).permute(0, 1, 3, 2, 4).contiguous()
    return c, s


IL_MIN_M = 1 << 30  # IL (interleaved code+scale blocks) only at T = 8 (IL was slower at 128 / 256 tokens on this layer); from this many tokens ...
IL_M = (8, 16)  # ... and at these token counts; at T = 16 only W2 reads the interleaved copy (W13 runs GU there)
GU_M = (8, 16, 32)  # token counts whose W13 uses the gate/up-paired (GU) copy


def _gu_rows(I):
    """Row order of the gate/up-paired W13 copy. New row 64 t + 16 s + 8 g + i holds gate (g = 0) or up (g = 1)
    row of output column 32 t + 8 s + i (a lossless row permutation)."""
    r = torch.arange(2 * I)
    t, s, g, i = r // 64, (r % 64) // 16, (r % 16) // 8, r % 8
    return g * I + 32 * t + 8 * s + i


def _interleaved(c, s):
    """Tile-major codes and scales merged per block (codes then scales, 9216 B), a second lossless copy
    used from IL_MIN_M tokens and at IL_M, where it streams better."""
    n_l, nt, kb = c.shape[:3]
    blk = torch.cat([c.reshape(n_l, nt, kb, -1), s.view(torch.uint8).reshape(n_l, nt, kb, -1)], dim=3).contiguous()
    return blk, blk.view(torch.float8_e4m3fn)


# ---- W13 verify kernels: items of up to 32 rows, token tile chosen per item ----
MIX_MIN_M = 128  # from this many tokens W13 runs _w13_mix on 32-row items
BM_W13 = 32
W13_MAXNREG_MIX = 128  # the 32-row tile needs 128 registers to run without spills


@triton.jit
def _route2(ids_ptr, local_ptr, sorted_ptr, work_ptr, nwork_ptr, work2_ptr, nwork2_ptr, P, local_start,
            LOCAL: tl.constexpr, BM: tl.constexpr, BM2: tl.constexpr, P_BLOCK: tl.constexpr, W_MAX: tl.constexpr,
            W_MAX2: tl.constexpr):
    """_route, plus a second work list over the same sorted pairs with items of at most BM2 rows (W13's)."""
    offs = tl.arange(0, P_BLOCK)
    pm = offs < P
    gid = tl.load(ids_ptr + offs, mask=pm, other=-1)
    lid = gid - tl.load(local_start)  # this layer passes local_start as an int32 [1] tensor
    live = pm & (lid >= 0) & (lid < LOCAL)
    lid = tl.where(live, lid, -1)
    tl.store(local_ptr + offs, lid, mask=pm)
    key = tl.where(live, lid * P_BLOCK + offs, LOCAL * P_BLOCK + offs)
    skey = tl.sort(key)
    tl.store(sorted_ptr + offs, skey % P_BLOCK, mask=pm)
    counts = tl.histogram(tl.maximum(lid, 0), LOCAL, mask=live)
    starts = tl.cumsum(counts, 0) - counts
    e_ar = tl.arange(0, LOCAL)
    chunks = (counts + BM - 1) // BM
    coff = tl.cumsum(chunks, 0) - chunks
    total = tl.sum(chunks, 0)
    w = tl.arange(0, W_MAX)
    sel = (coff[None, :] <= w[:, None]) & ((coff + chunks)[None, :] > w[:, None])
    owner = tl.sum(tl.where(sel, e_ar[None, :], 0), 1)
    c_first = tl.sum(tl.where(sel, coff[None, :], 0), 1)
    e_start = tl.sum(tl.where(sel, starts[None, :], 0), 1)
    e_cnt = tl.sum(tl.where(sel, counts[None, :], 0), 1)
    r0 = e_start + (w - c_first) * BM
    r1 = tl.minimum(r0 + BM, e_start + e_cnt)
    wm = w < total
    tl.store(work_ptr + w * 3 + 0, owner, mask=wm)
    tl.store(work_ptr + w * 3 + 1, r0, mask=wm)
    tl.store(work_ptr + w * 3 + 2, r1, mask=wm)
    tl.store(nwork_ptr, total)
    chunks2 = (counts + BM2 - 1) // BM2
    coff2 = tl.cumsum(chunks2, 0) - chunks2
    total2 = tl.sum(chunks2, 0)
    w2 = tl.arange(0, W_MAX2)
    sel2 = (coff2[None, :] <= w2[:, None]) & ((coff2 + chunks2)[None, :] > w2[:, None])
    owner2 = tl.sum(tl.where(sel2, e_ar[None, :], 0), 1)
    c_first2 = tl.sum(tl.where(sel2, coff2[None, :], 0), 1)
    e_start2 = tl.sum(tl.where(sel2, starts[None, :], 0), 1)
    e_cnt2 = tl.sum(tl.where(sel2, counts[None, :], 0), 1)
    q0 = e_start2 + (w2 - c_first2) * BM2
    q1 = tl.minimum(q0 + BM2, e_start2 + e_cnt2)
    wm2 = w2 < total2
    tl.store(work2_ptr + w2 * 3 + 0, owner2, mask=wm2)
    tl.store(work2_ptr + w2 * 3 + 1, q0, mask=wm2)
    tl.store(work2_ptr + w2 * 3 + 2, q1, mask=wm2)
    tl.store(nwork2_ptr, total2)





# ---- _route2 on LOCAL + 2 CTAs. The single-CTA sort of keys lid * P_BLOCK + offs (unique; local pairs by
# expert then pair index, non-local pairs after in pair order) is reproduced exactly: CTA e < LOCAL writes expert e's
# pairs at starts[e] + their rank in pair order, CTA LOCAL writes the non-local pairs after the n_live local ones, and
# CTA LOCAL + 1 writes local and both work lists with _route2's own expressions. Integer outputs are bit-identical.
@triton.jit
def _route2_mc(ids_ptr, local_ptr, sorted_ptr, work_ptr, nwork_ptr, work2_ptr, nwork2_ptr, P, local_start,
            LOCAL: tl.constexpr, BM: tl.constexpr, BM2: tl.constexpr, P_BLOCK: tl.constexpr, W_MAX: tl.constexpr,
            W_MAX2: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, P_BLOCK)
    pm = offs < P
    gid = tl.load(ids_ptr + offs, mask=pm, other=-1)
    lid = gid - tl.load(local_start)
    live = pm & (lid >= 0) & (lid < LOCAL)
    lid = tl.where(live, lid, -1)
    if pid < LOCAL:
        start = tl.sum((live & (lid < pid)).to(tl.int32), 0)  # no histogram for one offset
        mine = live & (lid == pid)
        rank = tl.cumsum(mine.to(tl.int32), 0) - 1
        tl.store(sorted_ptr + start + rank, offs, mask=mine)
    elif pid == LOCAL:
        dead = pm & (live == 0)
        n_live = tl.sum(live.to(tl.int32), 0)
        rank = tl.cumsum(dead.to(tl.int32), 0) - 1
        tl.store(sorted_ptr + n_live + rank, offs, mask=dead)
    elif pid == LOCAL + 1:  # local ids + W2's work list
        tl.store(local_ptr + offs, lid, mask=pm)
        counts = tl.histogram(tl.maximum(lid, 0), LOCAL, mask=live)
        starts = tl.cumsum(counts, 0) - counts
        e_ar = tl.arange(0, LOCAL)
        chunks = (counts + BM - 1) // BM
        coff = tl.cumsum(chunks, 0) - chunks
        total = tl.sum(chunks, 0)
        w = tl.arange(0, W_MAX)
        sel = (coff[None, :] <= w[:, None]) & ((coff + chunks)[None, :] > w[:, None])
        owner = tl.sum(tl.where(sel, e_ar[None, :], 0), 1)
        c_first = tl.sum(tl.where(sel, coff[None, :], 0), 1)
        e_start = tl.sum(tl.where(sel, starts[None, :], 0), 1)
        e_cnt = tl.sum(tl.where(sel, counts[None, :], 0), 1)
        r0 = e_start + (w - c_first) * BM
        r1 = tl.minimum(r0 + BM, e_start + e_cnt)
        wm = w < total
        tl.store(work_ptr + w * 3 + 0, owner, mask=wm)
        tl.store(work_ptr + w * 3 + 1, r0, mask=wm)
        tl.store(work_ptr + w * 3 + 2, r1, mask=wm)
        tl.store(nwork_ptr, total)
    else:  # W13's work list on its own CTA
        counts = tl.histogram(tl.maximum(lid, 0), LOCAL, mask=live)
        starts = tl.cumsum(counts, 0) - counts
        e_ar = tl.arange(0, LOCAL)
        chunks2 = (counts + BM2 - 1) // BM2
        coff2 = tl.cumsum(chunks2, 0) - chunks2
        total2 = tl.sum(chunks2, 0)
        w2 = tl.arange(0, W_MAX2)
        sel2 = (coff2[None, :] <= w2[:, None]) & ((coff2 + chunks2)[None, :] > w2[:, None])
        owner2 = tl.sum(tl.where(sel2, e_ar[None, :], 0), 1)
        c_first2 = tl.sum(tl.where(sel2, coff2[None, :], 0), 1)
        e_start2 = tl.sum(tl.where(sel2, starts[None, :], 0), 1)
        e_cnt2 = tl.sum(tl.where(sel2, counts[None, :], 0), 1)
        q0 = e_start2 + (w2 - c_first2) * BM2
        q1 = tl.minimum(q0 + BM2, e_start2 + e_cnt2)
        wm2 = w2 < total2
        tl.store(work2_ptr + w2 * 3 + 0, owner2, mask=wm2)
        tl.store(work2_ptr + w2 * 3 + 1, q0, mask=wm2)
        tl.store(work2_ptr + w2 * 3 + 2, q1, mask=wm2)
        tl.store(nwork2_ptr, total2)

@gluon.jit
def _w13_item(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n,
              K: gl.constexpr, I: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr,
              XS: gl.constexpr, SMB: gl.constexpr, HOP: gl.constexpr, CHK: gl.constexpr, PF: gl.constexpr):
    mma: gl.constexpr = (MMA3_8 if BM == 8 else MMA3_16) if HOP else MMA_L
    a_l: gl.constexpr = (A3_8 if BM == 8 else A3_16) if HOP else A_L
    n0 = pid_n * BN
    rw, qw, jw = _idx3(BN, 4, 8, L3)
    KB: gl.constexpr = K // BK  # tile-major weights, see _tile_major
    w_off = (e * (2 * I // BN) + pid_n) * KB * (BN * 32) + rw * 32 + qw * 8 + jw
    rs, qs, gs_ = _idx3(BN, 4, 4, LS)
    s_off = (e * (2 * I // BN) + pid_n) * KB * (BN * 16) + rs * 16 + qs * 4 + gs_
    mx, qx, cx = _idx3(BM, 4, 64, LX8 if BM == 8 else (LX32 if BM == 32 else LX))
    rv = r0 + mx
    pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
    x_off = (pair // 8) * K + qx * 64 + cx
    x_mask = (rv < r1) & (cx >= 0)
    xoff, xmsk = _xg_offsets(sorted_ptr, r0, r1, K, True, BM)
    x_s = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 64], X_SMEM)
    xb_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], XB_SMEM)
    wp = w_ptr.to(gl.pointer_type(gl.int32))
    acc_g = gl.zeros([BN, BM], gl.float32, layout=mma)
    acc_u = gl.zeros([BN, BM], gl.float32, layout=mma)
    if PF:  # register double buffer: block k0 + BK is requested before block k0 is decoded
        wg_n = gl.load(wp + w_off)
        sg_n = gl.load(sf_ptr + s_off)
        wu_n = gl.load(wp + w_off + I * (K // 8))
        su_n = gl.load(sf_ptr + s_off + I * (K // 16))
    for k0 in range(0, K, BK):
        if PF:
            wg = wg_n
            sg = sg_n
            wu = wu_n
            su = su_n
            kn = k0 + BK
            wm = (rw >= 0) & (kn < K)
            sm = (rs >= 0) & (kn < K)
            wg_n = gl.load(wp + w_off + (kn // BK) * (BN * 32), mask=wm, other=0)
            sg_n = gl.load(sf_ptr + s_off + (kn // BK) * (BN * 16), mask=sm)
            wu_n = gl.load(wp + w_off + I * (K // 8) + (kn // BK) * (BN * 32), mask=wm, other=0)
            su_n = gl.load(sf_ptr + s_off + I * (K // 16) + (kn // BK) * (BN * 16), mask=sm)
        else:
            wg = gl.load(wp + w_off + (k0 // BK) * (BN * 32))
            sg = gl.load(sf_ptr + s_off + (k0 // BK) * (BN * 16))
            wu = gl.load(wp + w_off + I * (K // 8) + (k0 // BK) * (BN * 32))
            su = gl.load(sf_ptr + s_off + I * (K // 16) + (k0 // BK) * (BN * 16))
        if SMB:
            _xb_store(x_ptr, xoff, xmsk, k0, xb_s, BM, BK)
            if CHK:
                g0, g1, g2, g3 = _wsplit4(wg, BN)
                h0, h1, h2, h3 = _ssplit4(sg, BN)
                acc_g = _mma_chunk(g0, h0, xb_s, 0, acc_g, BN, a_l, HOP)
                acc_g = _mma_chunk(g1, h1, xb_s, 1, acc_g, BN, a_l, HOP)
                acc_g = _mma_chunk(g2, h2, xb_s, 2, acc_g, BN, a_l, HOP)
                acc_g = _mma_chunk(g3, h3, xb_s, 3, acc_g, BN, a_l, HOP)
                u0, u1, u2, u3 = _wsplit4(wu, BN)
                v0, v1, v2, v3 = _ssplit4(su, BN)
                acc_u = _mma_chunk(u0, v0, xb_s, 0, acc_u, BN, a_l, HOP)
                acc_u = _mma_chunk(u1, v1, xb_s, 1, acc_u, BN, a_l, HOP)
                acc_u = _mma_chunk(u2, v2, xb_s, 2, acc_u, BN, a_l, HOP)
                acc_u = _mma_chunk(u3, v3, xb_s, 3, acc_u, BN, a_l, HOP)
            elif HOP:
                acc_g = warpgroup_mma(_decode_tile(wg, sg, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc_g)
                acc_u = warpgroup_mma(_decode_tile(wu, su, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc_u)
            else:
                xb = xb_s.permute((1, 0)).load(B_L)
                acc_g = mma_v2(_decode_tile(wg, sg, BN, BK, L3, A_L), xb, acc_g)
                acc_u = mma_v2(_decode_tile(wu, su, BN, BK, L3, A_L), xb, acc_u)
        else:
            if XS:
                xb = _x_shared(x_ptr, xoff, xmsk, k0, x_s, BM, BK)
            else:
                xb = _x_tile(gl.load(x_ptr + x_off + k0, mask=x_mask, other=0.0), BM, BK, B_L)
            acc_g = mma_v2(_decode_tile(wg, sg, BN, BK, L3, A_L), xb, acc_g)
            acc_u = mma_v2(_decode_tile(wu, su, BN, BK, L3, A_L), xb, acc_u)
    g = gl.load(gs_ptr + e) * ACC_BIAS
    hg = (acc_g * g).to(gl.bfloat16).to(gl.float32)
    hu = (acc_u * g).to(gl.bfloat16).to(gl.float32)
    inter = (hg / (1.0 + gl.exp(-hg))).to(gl.bfloat16).to(gl.float32) * hu
    n_o = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
    m_o = gl.arange(0, BM, layout=gl.SliceLayout(0, mma))
    ro = r0 + m_o
    pair_o = gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
    optr = out_ptr + gl.expand_dims(pair_o, 0) * I + gl.expand_dims(n0 + n_o, 1)
    gl.store(optr, inter.to(gl.bfloat16), mask=gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1))



@gluon.jit
def _w13_mix(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr,
             K: gl.constexpr, I: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr,
             XS: gl.constexpr, SMB: gl.constexpr = False, HOP: gl.constexpr = False,
             CHK: gl.constexpr = False, PF: gl.constexpr = False):
    tl.extra.cuda.gdc_wait()  # wait for the producer's writes before any dependent read (launch_pdl=True launch)
    tl.extra.cuda.gdc_launch_dependents()  # trigger at kernel start (dependents keep their gdc_wait)
    pid_n = gl.program_id(0)
    pid_w = gl.program_id(1)
    active = pid_w < gl.load(nwork_ptr)
    e = gl.load(work_ptr + pid_w * 3 + 0)
    r0 = gl.load(work_ptr + pid_w * 3 + 1)
    r1 = gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        if r1 - r0 <= 8:
            _w13_item(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n,
                      K=K, I=I, BN=BN, BK=BK, BM=8, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)
        elif BM == 32:
            if r1 - r0 <= 16:
                _w13_item(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n,
                          K=K, I=I, BN=BN, BK=BK, BM=16, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)
            else:
                _w13_item(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n,
                          K=K, I=I, BN=BN, BK=BK, BM=32, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)
        else:
            _w13_item(x_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n,
                      K=K, I=I, BN=BN, BK=BK, BM=BM, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)




# ---- W2 per-item tile at M >= MIX_MIN_M ----
@gluon.jit
def _w2_item(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n,
             K: gl.constexpr, N: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr,
             XS: gl.constexpr, SMB: gl.constexpr, HOP: gl.constexpr, CHK: gl.constexpr, PF: gl.constexpr):
    mma: gl.constexpr = (MMA3_8 if BM == 8 else MMA3_16) if HOP else MMA_L
    a_l: gl.constexpr = (A3_8 if BM == 8 else A3_16) if HOP else A_L
    n0 = pid_n * BN
    rw, qw, jw = _idx3(BN, 4, 8, L3)
    KB: gl.constexpr = K // BK  # tile-major weights
    w_off = (e * (N // BN) + pid_n) * KB * (BN * 32) + rw * 32 + qw * 8 + jw
    rs, qs, gs_ = _idx3(BN, 4, 4, LS)
    s_off = (e * (N // BN) + pid_n) * KB * (BN * 16) + rs * 16 + qs * 4 + gs_
    mx, qx, cx = _idx3(BM, 4, 64, LX8 if BM == 8 else (LX32 if BM == 32 else LX))
    rv = r0 + mx
    pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
    x_off = pair * K + qx * 64 + cx
    x_mask = (rv < r1) & (cx >= 0)
    xoff, xmsk = _xg_offsets(sorted_ptr, r0, r1, K, False, BM)
    x_s = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 64], X_SMEM)
    xb_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BK], XB_SMEM)
    wp = w_ptr.to(gl.pointer_type(gl.int32))
    acc = gl.zeros([BN, BM], gl.float32, layout=mma)
    if PF:
        wq_n = gl.load(wp + w_off)
        sq_n = gl.load(sf_ptr + s_off)
    for k0 in range(0, K, BK):
        if PF:
            wq = wq_n
            sq = sq_n
            kn = k0 + BK
            wq_n = gl.load(wp + w_off + (kn // BK) * (BN * 32), mask=(rw >= 0) & (kn < K), other=0)
            sq_n = gl.load(sf_ptr + s_off + (kn // BK) * (BN * 16), mask=(rs >= 0) & (kn < K))
        else:
            wq = gl.load(wp + w_off + (k0 // BK) * (BN * 32))
            sq = gl.load(sf_ptr + s_off + (k0 // BK) * (BN * 16))
        if SMB:
            _xb_store(a_ptr, xoff, xmsk, k0, xb_s, BM, BK)
            if CHK:
                q0, q1, q2, q3 = _wsplit4(wq, BN)
                p0, p1, p2, p3 = _ssplit4(sq, BN)
                acc = _mma_chunk(q0, p0, xb_s, 0, acc, BN, a_l, HOP)
                acc = _mma_chunk(q1, p1, xb_s, 1, acc, BN, a_l, HOP)
                acc = _mma_chunk(q2, p2, xb_s, 2, acc, BN, a_l, HOP)
                acc = _mma_chunk(q3, p3, xb_s, 3, acc, BN, a_l, HOP)
            elif HOP:
                acc = warpgroup_mma(_decode_tile(wq, sq, BN, BK, L3, a_l), xb_s.permute((1, 0)), acc)
            else:
                acc = mma_v2(_decode_tile(wq, sq, BN, BK, L3, A_L), xb_s.permute((1, 0)).load(B_L), acc)
        else:
            if XS:
                xb = _x_shared(a_ptr, xoff, xmsk, k0, x_s, BM, BK)
            else:
                xb = _x_tile(gl.load(a_ptr + x_off + k0, mask=x_mask, other=0.0), BM, BK, B_L)
            acc = mma_v2(_decode_tile(wq, sq, BN, BK, L3, A_L), xb, acc)
    y = acc * (gl.load(gs_ptr + e) * ACC_BIAS)
    n_o = gl.arange(0, BN, layout=gl.SliceLayout(1, mma))
    m_o = gl.arange(0, BM, layout=gl.SliceLayout(0, mma))
    ro = r0 + m_o
    pair_o = gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
    optr = out_ptr + gl.expand_dims(pair_o, 0) * N + gl.expand_dims(n0 + n_o, 1)
    gl.store(optr, y.to(gl.bfloat16), mask=gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1))



@gluon.jit
def _w2_mix(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr,
            K: gl.constexpr, N: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr, XS: gl.constexpr,
            SMB: gl.constexpr = False, HOP: gl.constexpr = False,
            CHK: gl.constexpr = False, PF: gl.constexpr = False):
    tl.extra.cuda.gdc_wait()  # wait for the producer's writes before any dependent read (launch_pdl=True launch)
    tl.extra.cuda.gdc_launch_dependents()  # trigger at kernel start (dependents keep their gdc_wait)
    pid_n = gl.program_id(0)
    pid_w = gl.program_id(1)
    active = pid_w < gl.load(nwork_ptr)
    e = gl.load(work_ptr + pid_w * 3 + 0)
    r0 = gl.load(work_ptr + pid_w * 3 + 1)
    r1 = gl.load(work_ptr + pid_w * 3 + 2)
    if active:
        if r1 - r0 <= 8:
            _w2_item(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n,
                     K=K, N=N, BN=BN, BK=BK, BM=8, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)
        else:
            _w2_item(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, out_ptr, e, r0, r1, pid_n,
                     K=K, N=N, BN=BN, BK=BK, BM=BM, XS=XS, SMB=SMB, HOP=HOP, CHK=CHK, PF=PF)





def _grouped_prepare(weights):
    """Grouped-expert prepare step for this layer's 32 local experts (weights are already local here)."""
    L = LOCAL_EXPERTS
    # tile-major layout (only this grouped copy; the Marlin bank and raw_experts keep theirs)
    w13, w13_sf = _tile_major(torch.stack([_permute_bits(weights["w13"][e]) for e in range(L)]), weights["w13_sf"][:L])
    w2, w2_sf = _tile_major(torch.stack([_permute_bits(weights["w2"][e]) for e in range(L)]), weights["w2_sf"][:L])
    gu = _gu_rows(weights["w13"].shape[1] // 2).to(weights["w13"].device)  # gate/up-paired row order
    w13g, w13g_sf = _tile_major(torch.stack([_permute_bits(weights["w13"][e][gu]) for e in range(L)]),
                                weights["w13_sf"][:L][:, gu])
    return _segf_chain({  # per-(expert, tile) contiguous k-block records
        "w13": w13, "w13_sf": w13_sf, "w13_gs": weights["w13_gs"][:L].float().contiguous(),
        "w2": w2, "w2_sf": w2_sf, "w2_gs": weights["w2_gs"][:L].float().contiguous(),
        "w13g": w13g, "w13g_sf": w13g_sf,
        "il13": _interleaved(w13, w13_sf), "il2": _interleaved(w2, w2_sf),  # interleaved code+scale copies
        "hopper": torch.cuda.get_device_capability()[0] == 9,
    })


SEGF_F1 = True  # one-token W13 reads f1, one-token W2 reads f2 (== il2)


def _segf_chain(g):
    # Fallback: a failed inverse check keeps the original layout and routes the
    # one-token kernels to the original addressing (IL=False), so outputs and the normal path are unchanged. _segf_layouts raises
    # before g is touched, so nothing partial is left behind. Decided once at prepare time (untimed), not per call.
    try:
        g.update(_segf_layouts(g))
        g["chain13"] = g["f1"]
        g["segf_ok"] = True
    except RuntimeError:
        g["segf_ok"] = False
    return g


def _segf_layouts(g):
    """Prepare-time expert layouts for the one-token and verify chain. Exact byte permutations of the
    tile-major copies, inverse-checked here (prepare is untimed). Every record is a multiple of 128 B and every
    (expert, tile) run is contiguous:
      f1   uint8 [L, I/BN, KB, 2*BN*(BK/2+BK/16)]  per (expert, W13 n-tile): k-block records
           [gate codes BN*BK/2 | up codes BN*BK/2 | gate sf BN*BK/16 | up sf BN*BK/16]  (18432 B at BN 64, BK 256)
      f1gu uint8 [L, 2I/BN, KB, BN*(BK/2+BK/16)]   gate/up-paired (GU, _gu_rows) copy: records [codes | sf]   (verify)
      f2   uint8 [L, H/BN, KB, BN*(BK/2+BK/16)]    W2 records [codes | sf]   (== il2)
      f3   fp32  [L, 2]                            (w13_gs, w2_gs) per expert
    Each array also comes as (bytes, float8 view) for the kernels' code / scale pointers."""
    c, s = g["w13"], g["w13_sf"].view(torch.uint8)
    n_l, nt, kb = c.shape[:3]
    h = nt // 2
    parts = (c[:, :h], c[:, h:], s[:, :h], s[:, h:])
    flat = [x.reshape(n_l, h, kb, -1) for x in parts]
    f1 = torch.cat(flat, dim=3).contiguous()
    o = 0
    for x in flat:
        n = x.shape[3]
        if not torch.equal(f1[..., o:o + n], x):
            raise RuntimeError("segment F f1 is not an exact permutation of the tile-major W13")
        o += n
    f1gu = _interleaved(g["w13g"], g["w13g_sf"])[0]
    cg = g["w13g"].reshape(*g["w13g"].shape[:3], -1)
    if not (torch.equal(f1gu[..., :cg.shape[3]], cg) and
            torch.equal(f1gu[..., cg.shape[3]:], g["w13g_sf"].view(torch.uint8).reshape(*cg.shape[:3], -1))):
        raise RuntimeError("segment F f1gu is not an exact permutation of the GU copy")
    f2 = g["il2"][0]
    c2 = g["w2"].reshape(*g["w2"].shape[:3], -1)
    if not (torch.equal(f2[..., :c2.shape[3]], c2) and
            torch.equal(f2[..., c2.shape[3]:], g["w2_sf"].view(torch.uint8).reshape(*c2.shape[:3], -1))):
        raise RuntimeError("segment F f2 is not an exact permutation of the tile-major W2")
    for t in (f1, f1gu, f2):
        if t.shape[-1] % 128 or t.data_ptr() % 128:
            raise RuntimeError("segment F record not 128 B aligned")
    f3 = torch.stack((g["w13_gs"], g["w2_gs"]), dim=1).contiguous()
    return {"f1": (f1, f1.view(torch.float8_e4m3fn)), "f1gu": (f1gu, f1gu.view(torch.float8_e4m3fn)),
            "f2": (f2, f2.view(torch.float8_e4m3fn)), "f3": f3}


def _grouped_route(topk_ids, local_start):
    """The grouped routing step (pairs grouped by local expert, work list); depends only on topk_ids, so this layer
    runs it on the router side stream."""
    M = topk_ids.shape[0]
    L = LOCAL_EXPERTS
    dev = topk_ids.device
    P = M * TOP_K
    ids = topk_ids.to(torch.int32).contiguous()
    p_block = triton.next_power_of_2(P)
    bm = 8 if M <= 32 else BM
    w_max = triton.next_power_of_2(triton.cdiv(P, bm) + L)
    local = torch.empty((M, TOP_K), dtype=torch.int32, device=dev)
    sorted_pairs = torch.empty((P,), dtype=torch.int32, device=dev)
    work = torch.empty((w_max, 3), dtype=torch.int32, device=dev)
    nwork = torch.empty((1,), dtype=torch.int32, device=dev)
    if M >= MIX_MIN_M:  # one routing pass, two work lists (W2's, W13's 32-row items)
        w_max13 = triton.next_power_of_2(triton.cdiv(P, BM_W13) + L)
        work13 = torch.empty((w_max13, 3), dtype=torch.int32, device=dev)
        nwork13 = torch.empty((1,), dtype=torch.int32, device=dev)
        _route2_mc[(L + 3,)](ids, local, sorted_pairs, work, nwork, work13, nwork13, P, local_start, LOCAL=L, BM=bm,  # multi-CTA routing (_route2_mc)
                      BM2=BM_W13, P_BLOCK=p_block, W_MAX=w_max, W_MAX2=w_max13, num_warps=16 if p_block >= 1024 else 4)
        return local, sorted_pairs, work, nwork, bm, w_max, work13, nwork13, w_max13
    _route[(1,)](ids, local, sorted_pairs, work, nwork, P, local_start, LOCAL=L, BM=bm, P_BLOCK=p_block, W_MAX=w_max,
                 num_warps=16 if p_block >= 1024 else 4)
    return local, sorted_pairs, work, nwork, bm, w_max


W13_MAXNREG_MIN_M = 128  # W13 register cap from this many tokens on decode's register path
W13_MAXNREG = 96


def _w13_regcap(M):
    if M >= MIX_MIN_M:  # _w13_mix's 32-row tile
        return {"maxnreg": W13_MAXNREG_MIX}
    return {"maxnreg": W13_MAXNREG} if M >= W13_MAXNREG_MIN_M else {}


W2_KS = 2  # W2 split-K factor ...
W2_KS_M = (8, 16, 32)  # ... at these token counts


def _grouped_experts(x, topk_weights, route, state):
    """Grouped expert GEMMs (gate/up + silu, down) after _grouped_route, with this file's finalize."""
    g = state["grouped"]
    M, H = x.shape
    I = D_EXPERT
    dev = x.device
    P = M * TOP_K
    tw = topk_weights.to(torch.float32).contiguous()
    local, sorted_pairs, work, nwork, bm, w_max = route[:6]
    inter = torch.empty((P, I), dtype=torch.bfloat16, device=dev)
    il = M >= IL_MIN_M or M in IL_M  # interleaved code+scale blocks
    w13c, w13s = g["il13"] if il else (g["w13"], g["w13_sf"])
    w2c, w2s = g["il2"] if il else (g["w2"], g["w2_sf"])
    if M in GU_M:  # gate/up-paired W13 through _w2 (GU), SiLU fused
        _w2[(2 * I // BN, w_max)](x, g["w13g"], g["w13g_sf"], g["w13_gs"], sorted_pairs, work, nwork,
                                  inter, K=H, N=2 * I, BN=BN, BK=BK, BM=bm, XS=M >= XS_MIN_M, SMB=False, HOP=False,
                                  CHK=True, PF=M <= PF_MAX_M, GU=True, launch_pdl=M != 16, num_warps=NUM_WARPS)
    elif M >= MIX_MIN_M:  # W13 on 32-row items, register path
        work13, nwork13, w_max13 = route[6:9]
        _w13_mix[(I // BN, w_max13)](x, g["w13"], g["w13_sf"], g["w13_gs"], sorted_pairs, work13, nwork13, inter,
                                     K=H, I=I, BN=BN, BK=BK, BM=BM_W13, XS=M >= XS_MIN_M, SMB=False, HOP=False,
                                     CHK=True, PF=M <= PF_MAX_M, launch_pdl=M != 16, num_warps=NUM_WARPS, **_w13_regcap(M))
    else:
        _w13_silu[(I // BN, w_max)](x, w13c, w13s, g["w13_gs"], sorted_pairs, work, nwork, inter,
                                    K=H, I=I, BN=BN, BK=BK, BM=bm, XS=M >= XS_MIN_M, SMB=M >= SMB_MIN_M,
                                    HOP=g["hopper"] and M >= SMB_MIN_M, CHK=True, PF=M <= PF_MAX_M,
                                    IL=il, launch_pdl=M != 16, num_warps=NUM_WARPS, **_w13_regcap(M))
    y = torch.empty((P, H), dtype=torch.bfloat16, device=dev)
    if M >= MIX_MIN_M:  # per-item 8 / 16-row tile, register path
        _w2_mix[(H // BN, w_max)](inter, g["w2"], g["w2_sf"], g["w2_gs"], sorted_pairs, work, nwork, y,
                                  K=I, N=H, BN=BN, BK=BK, BM=bm, XS=M >= XS_MIN_M, SMB=False, HOP=False,
                                  CHK=True, PF=M <= PF_MAX_M, launch_pdl=M != 16, num_warps=NUM_WARPS)
        return _post_expert_weighted_sum(y.view(M, TOP_K, H), tw, state["post_expert_w"], local)
    if M in W2_KS_M:  # W2 split over K, FP32 partials summed in the post-expert kernel
        y = torch.empty((P, W2_KS, H), dtype=torch.float32, device=dev)
        _w2[(H // BN, w_max, W2_KS)](inter, w2c, w2s, g["w2_gs"], sorted_pairs, work, nwork, y,
                                     K=I, N=H, BN=BN, BK=BK, BM=bm, XS=M >= XS_MIN_M, SMB=False, HOP=False,
                                     CHK=True, PF=False, IL=il, KS=W2_KS, launch_pdl=M != 16, num_warps=NUM_WARPS)
        return _post_expert_weighted_sum_ks(y, tw, state["post_expert_w"], local, W2_KS)
    y = torch.empty((P, H), dtype=torch.bfloat16, device=dev)
    _w2[(H // BN, w_max)](inter, w2c, w2s, g["w2_gs"], sorted_pairs, work, nwork, y,
                          K=I, N=H, BN=BN, BK=BK, BM=bm, XS=M >= XS_MIN_M, SMB=M >= SMB_MIN_M,
                          HOP=g["hopper"] and M >= SMB_MIN_M, CHK=True, PF=M <= PF_MAX_M,
                          IL=il, launch_pdl=M != 16, num_warps=NUM_WARPS)
    return _post_expert_weighted_sum(y.view(M, TOP_K, H), tw, state["post_expert_w"], local)


def _experts(x, topk_weights, local_ids, state, align=None):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize.
    local_ids: this GPU's expert per slot (0-31, -1 elsewhere), as SGLang's expert_map gives."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    topk_weights = topk_weights.to(torch.float32).contiguous()
    if num_tokens <= GEMV_MOE_MAX_TOKENS:  # bandwidth-bound one-token expert GEMV (fixed-shape branch)
        y = _moe_gemv(x, local_ids, state["raw_experts"])
        return _post_expert_weighted_sum(y, topk_weights, state["post_expert_w"], local_ids)
    if num_tokens >= GROUPED_MIN_TOKENS and state["grouped_moe"]:
        raw = state["raw_experts"]
        sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, GROUPED_BT, N_EXPERTS)
        y = moe_nvfp4(x, raw["w13"], raw["w13_sf"], raw["w13_gs"], raw["w2"], raw["w2_sf"], raw["w2_gs"],
                      sorted_ids, block_experts, n_padded, TOP_K, GROUPED_BT, m_fast=True)
        return _post_expert_weighted_sum(y.view(num_tokens, TOP_K, size_k), topk_weights, state["post_expert_w"],
                                         local_ids)
    if align is not None:
        sorted_ids, block_experts, n_padded, block_size_m = align
    else:
        block_size_m = _marlin_block_size(num_tokens)
        sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return _post_expert_weighted_sum(y, topk_weights, state["post_expert_w"], local_ids)


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _dense(data["attn_input"], "qkv_gate", state)
    # QK-norm (+ RoPE), q to FP8 and the FP8 cache write in one kernel (slot 0 is a live row here).
    # full single-token decode at batch >= 16 on SM90 runs the Gluon kernel end to end
    # (q/k norm, FP8 store of the new k/v, attention, RMSNorm x sigmoid(gate)); the other cases follow.
    if (not cfg["window"] and length == 8 and batch >= 16 and cfg["page_size"] == 1 and hq == 16 and hkv == 2
            and d == 64 and _SWAP_SM90):
        o = _gluon_full_verify(data, cfg, qkv, tokens, _G_VFC).view(tokens, hq * d)
    elif (cfg["window"] or (length == 8 and batch >= 16 and cfg["page_size"] == 1 and hq == 16 and hkv == 2 and d == 64)) and _SWAP_SM90:
        o = _fused(data, cfg, qkv, tokens, state["cos_sin"]).view(tokens, hq * d)
    elif (not cfg["window"] and length == 1 and batch >= 16 and cfg["page_size"] == 1 and hq == 16 and hkv == 2
            and d == 64 and _SWAP_SM90):
        o = _gluon_full_decode(data, cfg, qkv, tokens, _G_FC).view(tokens, hq * d)
    else:
        q8 = _qk_norm_fp8_store(qkv, data, cfg, state["cos_sin"])
        gate = qkv[:, hq * d + 2 * hkv * d:]
        k_cache, v_cache = data["k_cache"], data["v_cache"]
        scaled = cfg["kv_scale"] != 1.0
        window = cfg["window"]
        descale = {}
        if scaled:
            descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                           v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
        if not window and length == 1 and batch >= 16:  # custom full single-token decode at B >= 16; FA3 for sliding, verify and B < 16
            out = _b_full_attention(q8, data)
        elif not window and length == 8 and batch >= VERIFY_ATTN_MIN_BATCH:  # full verify blocks
            out = _v_full_attention(q8, data)
        elif not window and length in (1, 8) and batch <= SMALL_FULL_MAX_BATCH:  # small-batch full layers
            out = _s_full_attention(q8, data)
        else:
            out = flash_attn_with_kvcache(
                q=q8,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=data["page_table"],
                cache_seqlens=data["cache_seqlens"],
                cu_seqlens_q=data["cu_seqlens_q"],
                max_seqlen_q=length,
                softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
                causal=True,
                window_size=(window - 1, 0) if window else (-1, -1),
                scheduler_metadata=data["scheduler_metadata"],
                **descale,
            )
        if isinstance(out, tuple):
            out = out[0]
        o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    attn_output = _dense(o, "o", state, pref=bool(cfg["window"]) and batch == 1)  # all-reduce: identity; L2 weight prefetch before the wait only behind the attention kernel's early trigger (shape-only)

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = _boundary(state["attn_boundary"], "attn", data["stream"], attn_output,
                                                router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream_hi" if tokens == 1 else "side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _dense(ffn_input, "pooler", state)
    with torch.cuda.stream(side_stream):
        logits = _router_logits(router_input if not split_router else ffn_input, tokens, state)
        if logits is not None:
            pass
        elif split_router:
            logits = _router_hilo(ffn_input, state["router_hi_t"], state["router_lo_t"])
        else:
            if tokens == 8:  # 8-token FP32 router (fixed-shape branch)
                logits = _router_eight(router_input.contiguous(), state["router"])
            else:
                logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
        route = None
        if tokens > 1:  # the grouped route on the side stream, overlapping the pre-expert norm
            routed = torch.cuda.Event()
            routed.record(side_stream)
            route = _grouped_route(topk_ids, data["local_start"])
    if route is None:
        main_stream.wait_stream(side_stream)
    else:
        main_stream.wait_event(routed)  # routing done; the route may still be running
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    # all-reduce: identity
    if tokens == 1 and _SWAP_SM90:  # the pre-expert norm inside W13's prologue (bit-exact rstd)
        expert_output = _experts_d1_n(expert_input, topk_ids, topk_weights, data["local_start"], state)
    elif tokens == 1:  # one token without the fused path: pre-expert norm with Marlin's alignment, then Marlin
        expert_input, local_ids, align = _rms_norm_local_ids(expert_input, state["pre_expert_w"], RMS_EPS, topk_ids,
                                                             data["local_start"])
        expert_output = _experts(expert_input, topk_weights, local_ids, state, align)
    else:  # grouped NVFP4 kernels for every T > 1 (fixed-shape branch); norm without Marlin's alignment
        expert_input, _, _ = _rms_norm_local_ids(expert_input, state["pre_expert_w"], RMS_EPS, topk_ids,
                                                 data["local_start"], want_align=False)
        main_stream.wait_stream(side_stream)  # the route (side stream) before the grouped experts
        if not torch.cuda.is_current_stream_capturing():
            for tensor in [t for t in route if isinstance(t, torch.Tensor)]:  # incl. W13's list
                tensor.record_stream(main_stream)
        expert_output = _grouped_experts(expert_input, topk_weights, route, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        # The all-gather pads this GPU's slice with zeros: write the slice into a zeroed row.
        # Columns past the slice of this scratch row block are zero from prepare() and never written.
        ffn_output = state["unpool_rows"][:tokens]
        _dense(expert_output, "unpooler", state, out=ffn_output)
    else:
        ffn_output = _dense(expert_output, "unpooler", state)

    # Post-FFN boundary, with the next layer's pre-attention norm (SGLang's JIT kernel).
    next_attn_input, stream, _ = _boundary(state["ffn_boundary"], "ffn", stream, ffn_output)
    return next_attn_input, stream, topk_ids


# full-attention verify-block launch configuration
FUSED_CFG[("full", 8)] = dict(block_n=128, warps=4, stages=3, max_split=8, per_sm=4, prefetch=True, min_chunk=256,
                              row_blocks=1, qk8=True, big=dict(per_sm=8), big_min_batch=32)
