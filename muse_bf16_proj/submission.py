# okc: problem=muse_bf16_proj
"""Candidate c049c: c048 (official parent, job 6224d0f5, 1.30168) plus a bundle of five parts, each screened on c048 on the H200 plane (real SGLang comparator, same job as c048's own path): (P6a) the pooler at exactly 128 tokens computes the same product with cuBLAS on swapped operands, torch.mm(W, x^T), and returns its transpose (a column-major view of the required shape and dtype; 0.954 of c048's time, job h54b; it ran in c045's official check); (H1a) the unpooler at 9-32 tokens streams its weights from a lossless tile-contiguous copy made in prepare() ([N/64, K/256, 64, 256], so every TMA box is one contiguous 32 KB block; 0.959 / 0.961 at 16 / 32 tokens, job h62); (H1b, H1d) the pooler at 2-32 tokens does the same with 64-row tiles (0.927 at 32 tokens, job h62; 0.959 / 0.966 at 8 / 16 tokens, job h65). (H1e) qkv_gate at exactly 128 tokens runs c043's split-K wgmma GEMM (64 x 128, 3 K ranges) on the same kind of tile-contiguous copy (0.955, job h67). Same kernels, K order and FP32 accumulation as c048 for H1a/H1b/H1d/H1e.

Candidate c048: c047 plus one change -- the unpooler's 1024-row slice at 2-8 tokens runs c046's cluster split-K kernel (64 weight rows x 256-wide K blocks, 4 K ranges reduced through distributed shared memory in a 4-CTA cluster) instead of c043's swap-AB streaming kernel with a global last-arriver reduction. H200 plane (job h51, real SGLang comparator, same job): 5.47 vs 6.30 us at 8 tokens (0.868). Falls back to c043's kernel if the cluster launch is unavailable at first use (before capture).

Candidate c047: c046 plus one change -- the router at 8-256 tokens keeps c043's two kernels (BF16 hi/lo partials over K ranges, then a reduce kernel), their tilings and arithmetic, but launches the reduce kernel with programmatic dependent launch: the partials kernel signals griddepcontrol.launch_dependents (at its start; after its K loop at 256 tokens) and the reduce kernel starts with griddepcontrol.wait, which blocks until the partials grid has completed and its writes are visible. Only the kernel boundary changes. H200 plane (job h49, real SGLang comparator, same job): 0.919 / 0.957 / 0.921 / 0.933 / 0.923 of c043's router time at 8 / 16 / 32 / 128 / 256 tokens. If the launch option is unavailable at first use (before capture), c043's launches serve the router.

Candidate c046: c043 (official parent, job ead2b544, 1.27918) plus one change -- qkv_gate at 2-32 tokens runs a split-K weight-streaming kernel whose K ranges are reduced inside a Hopper thread-block cluster through distributed shared memory instead of through global partials and counters. The kernel is c043's TMA streaming kernel (64 weight rows x 256-wide K blocks, 3 K ranges, 4 stages), launched as clusters of 3 CTAs (one cluster per output tile); ranks 1-2 stage their FP32 partial tile in their own shared memory and copy it with 16-byte stores into rank 0's shared memory, one cluster barrier publishes it, and rank 0 adds the three partials and stores BF16. Nothing persists between calls. H200 plane (jobs h42-h46, real SGLang comparator, overhead 2*t1-t2): 11.9 vs 13.3 us at 8 and 16 tokens, 13.1 vs 14.1 us at 32 tokens against c043's own kernels in the same job.

Candidate c043: c042 plus two H200-plane retiles (job h26, real SGLang comparator): the router at one token on aa2862cc's one-token GEMV form with FP32 output (1 row x 2048, 32 warps; 2.65 vs 2.89 us), and o at 2-32 tokens on TMA streaming (64 weight rows x 256-wide K blocks, no K split; 3-8% faster than the swap-AB streaming tile).

Candidate c042: c041 plus the router at 8 and 32 tokens on the two-kernel hi/lo form with H200-plane tilings (job h18d, real SGLang comparator: m8 5.10 vs ~6.0 us for aa2862cc's FP32 CUDA-core kernel, m32 5.79 vs ~7.1 us for aa2862cc's split-K tiling). The hi + lo split with FP32 accumulation passes the router check at any m (TASK.md).

Candidate c041: c040 plus retuned router kernels ranked on the H200 plane (jobs h16-h17, real SGLang comparator): the router at 16/128/256 tokens keeps aa2862cc's two-kernel hi/lo structure (BF16 hi + lo products, separate FP32 accumulators, K-split partials then a reduce kernel) with deeper pipelines and tilings chosen on the H200; plus TMA streaming for the unpooler at 9-32 tokens and for qkv_gate at 9-16 tokens.

Candidate c040: c030r (official parent, job 5f485773, 1.18742) unchanged, plus one composed change ranked on the H200 plane (jobs h05, h08, h09): c039's parts with retuned tilings -- one-token CUDA-core GEMVs for qkv_gate, o, the pooler, the unpooler slice (1 x 2048, 4 warps) and the router; TMA streaming for the pooler at 2-32 tokens (32 weight rows x 256-wide K blocks, no K split); streaming for o at 2-32 tokens and the unpooler slice at 2-8 tokens; TMA streaming for qkv_gate at 17-32 tokens; and the split-K wgmma GEMM at exactly 128 tokens for qkv_gate (64 x 128, split 3) and the pooler (64 x 128, split 2).

Candidate c039: c030r (official parent, job 5f485773, 1.18742) unchanged, plus one composed change ranked on the H200 plane (jobs h02-h05) -- one-token CUDA-core GEMVs for qkv_gate (1 x 4096, 16 warps), o (2 x 1024, 4 warps), the pooler (1 x 4096, 16 warps; replaces aa2862cc's 4 x 2048 tiling) and the router (FP32, 16 warps); the swap-AB streaming MMA kernel for o at 2-32 tokens (64 weight rows x 128-wide K blocks, no K split) and for the unpooler's 1024-row slice at 2-8 tokens; a TMA streaming tiling for qkv_gate at 17-32 tokens; and the split-K wgmma GEMM for the pooler at exactly 128 tokens.

Candidate c037: c030r (official parent, job 5f485773, 1.18742) unchanged, plus one composed change (each part below or near the 0.382% margin alone) -- one-token CUDA-core GEMVs for qkv_gate (1 row x the whole K, 16 warps) and the router (FP32 weights, 16 warps), the pooler at exactly 128 tokens on the split-K wgmma GEMM, and the unpooler's 1024-row slice at 2-8 tokens on the swap-AB streaming kernel (32 weight rows x 4 K ranges).

Candidate c031b: c030r unchanged, plus one composed change (each part below the 0.382% margin alone, jointly ~+0.7%) -- one-token CUDA-core GEMVs for qkv_gate (1 row x the whole K, 16 warps) and the router (FP32 weights, 16 warps; H200 2.701 in job 7c81bace), and the pooler at exactly 128 tokens on the split-K wgmma GEMM (H200 1.042 vs 1.001 in job 70e5a3d7).

Candidate c030r: candidate c028 unchanged, plus one change -- the unpooler at one token (this GPU's 1024-row
slice, 8.4 MB) runs a CUDA-core GEMV (2 rows x the whole K per program, 512 programs, one load burst each)
instead of SGLang's path, which reads the slice at about 1.2 TB/s on H200.

c028 (kept as is): the official parent c027 (job 3a3ff4e4, 1.1598) unchanged, plus one change -- the pooler at 2-32
tokens runs claude-bf16's TMA weight-streaming MMA kernel (swap-AB, 32-row weight tiles x 2 K ranges = 256
programs, K ranges reduced in-kernel). On H200 this exact kernel and tiling read 1.091 against SGLang at 32 tokens
(job 7c81bace); at 2-16 tokens it streams the same weights with a smaller token tile.

c027 (kept as is): aa2862cc plus qkv_gate at 1-16 tokens on claude-bf16's weight-streaming MMA kernel (swap-AB,
72 weight tiles x 11 uneven K ranges = 792 programs). Every other case runs aa2862cc's path byte-for-byte.

aa2862cc's description, for the paths it keeps -- SGLang's dense projections for Muse Spark Open on H200, at
SGLang commit a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f:

  qkv_gate, o, pooler    bf16_gemm_dispatch (cuBLAS on H200)
  unpooler               bf16_gemm_dispatch: up to 8 tokens on this GPU's 1/8 slice of the
                         weight, above that on the whole weight
  router                 FP32 weight: F.linear in FP32 up to 8 tokens, above that two BF16
                         GEMMs on a hi/lo split of the weight with FP32 accumulation
                         (MuseSpark1xRouter.forward_split2)

Differs from SGLang: the sliced unpooler's all-gather is not run.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl



class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]; up to 8 tokens, only rows 0-1023 (this GPU's slice)
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [M, K]
    config: Config


# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8


# ---- claude-bf16 small-m streaming block (names prefixed _cbf_ / CBF_) ----
# Swap-AB MMA weight streaming with K split into ranges of whole blocks and an in-kernel last-arriver
# reduction (self-resetting counters). CBF_ENABLED lists the (layer, m bucket) cases this block serves;
# everything else returns None and runs aa2862cc's path.


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
    ("qkv_gate", 16): (32, 64, 11, 4, 4),     # 72 tiles x 11 K ranges = 792 programs (H200: 1.093 at m=1, job 7c81bace)
    ("o", 16): (64, 128, 1, 4, 4),            # 128 tiles, no K split (H200 plane 1.08 / 1.10 at m8 / m16, job h05)
    ("o", 32): (64, 128, 1, 4, 4),            # H200 plane 1.08 at m32 (job h05)
}
# Cases served: qkv_gate at 1-16 tokens (one token runs the same 16-wide MMA tile).
CBF_ENABLED = {("qkv_gate", 1), ("qkv_gate", 16), ("o", 16), ("o", 32)}


try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except Exception:  # older Triton: the pointer kernel is used everywhere
    TensorDescriptor = None  # older Triton: the pooler case keeps aa2862cc's path


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
# qkv_gate at 17-32 tokens on the TMA streaming kernel (H200 plane 1.068 vs SGLang, job h05)
CBF_TMA32 = {"qkv_gate": (64, 128, 3, 4, 4)}
CBF_TMA = {
    "pooler": (32, 256, 1, 4, 5),     # 128 programs, no K split (H200 plane 5-6% faster than (32, 128, 2, 4, 5), job h09)
    "o": (64, 256, 1, 4, 4),          # 2-32 tokens; H200 plane 3-8% faster than the swap-AB streaming tile, job h26
    "unpooler": (64, 256, 1, 4, 4),   # 9-32 tokens (2-8 take the slice path first); H200 plane 17.0 vs 17.2-17.4 us SGLang, job h15
}


@triton.jit
def _cbf_gemv_rows(X, W, Y, K: tl.constexpr, N: tl.constexpr, SX: tl.constexpr, SW: tl.constexpr,
                   ROWS: tl.constexpr, TILE_K: tl.constexpr):
    """One token on CUDA cores: ROWS weight rows per program, TILE_K columns per step, FP32 accumulation,
    weights loaded with .cg (the form of aa2862cc's one-token pooler GEMV)."""
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
    "pooler": (1, 4096, 16),    # H200 plane 15.5 vs 18.2 us for aa2862cc's (4, 2048, 16) (job h05)
    "o": (2, 1024, 4),          # H200 plane 4.56 vs SGLang 5.50 us (job h05)
    "unpooler": (1, 2048, 4),   # the 1024-row slice: 1024 programs (H200 plane 2.84 vs 3.49 us for (2, 4096, 4), job h09)
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
# (H200: 2.701 against SGLang, job 7c81bace)
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
    "qkv_gate": (64, 128, 3, 4, 128),  # 36 x 3 = 108 programs (H200 plane 15.2 vs 16.0 us cuBLAS, job h08)
    "pooler": (64, 128, 2, 4, 128),    # 64 x 2 = 128 programs (H200 plane 22.6 vs 23.3 us for (128, 64, 4, 8), job h08)
}


# The unpooler's 1024-row slice at 2-8 tokens: BLOCK_N, BLOCK_K, SPLIT_K, warps, stages (128 programs)
CBF_UNPOOL8 = (32, 128, 4, 4, 4)


@triton.jit
def _cbf_router_partials(X, HI, LO, PARTIAL, M, K: tl.constexpr, N: tl.constexpr, SXM, SXK,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    """aa2862cc's router partials kernel generalised over the tiling: BF16 hi and lo products with separate FP32
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
# (H200 plane, real SGLang comparator: m16 5.11 vs 6.29 us, m128 6.96 vs 10.28, m256 8.90 vs 10.88 for aa2862cc's
# tilings; jobs h16-h17; 8 and 32 tokens from job h18d).
CBF_ROUTER_HILO = {
    8: (32, 128, 16, 4, 4, 16),      # job h18d: 5.10 us vs aa2862cc's FP32 CUDA-core kernel ~6.0 us
    16: (32, 64, 32, 4, 4, 16),
    32: (64, 64, 16, 4, 5, 16),      # job h18d: 5.79 us vs aa2862cc's tiling ~7.1 us
    128: (32, 128, 8, 4, 5, 64),
    256: (64, 64, 8, 4, 5, 64),
}
# TMA streaming for qkv_gate at 9-16 tokens (H200 plane 11.9 vs 12.4 us for the streaming kernel, job h15)
CBF_TMA16 = {"qkv_gate": (32, 128, 6, 4, 4)}


@triton.jit
def _cbf_gemv_rows32(X, W, Y, K: tl.constexpr, N: tl.constexpr, SX: tl.constexpr, SW: tl.constexpr,
                     ROWS: tl.constexpr, TILE_K: tl.constexpr):
    """One token on CUDA cores with FP32 weights and FP32 output (the router), aa2862cc's one-token GEMV form."""
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


# Router at one token: ROWS, TILE_K, num_warps (H200 plane 2.65 vs 2.89 us for the 16-warp GEMV, job h26)
CBF_ROUTER1_ROWS = (1, 2048, 32)


# ---- qkv_gate 2-32 tokens: split-K reduced inside a thread-block cluster (names _cbf_cluster*) ----
# The CTAs holding the K ranges of one output tile form a cluster (launched with num_ctas = SPLIT_K in the launch
# metadata of a kernel compiled for one CTA per program, so program_id(0) is the rank). Senders stage their FP32
# partial in their own shared memory and copy it with 16-byte st.shared::cluster stores into a slot of rank 0's
# shared memory (a static array declared in PTX); barrier.cluster release/acquire publishes it. An arrive at the
# start and its wait after the K loop make sure every CTA of the cluster has started before any remote access.


def _cbf_v4copy(n):
    body = ["{ .reg .u32 t, l, r; .reg .f32 a, b, c, d; mov.u32 t, %tid.x; shl.b32 t, t, 4; add.u32 l, $1, t; "
            "mapa.shared::cluster.u32 r, $2, 0; add.u32 r, r, t;"]
    for i in range(n):
        body.append(" ld.shared.v4.f32 {a, b, c, d}, [l+%d]; st.shared::cluster.v4.f32 [r+%d], {a, b, c, d};"
                    % (2048 * i, 2048 * i))
    body.append(" mov.u32 $0, 0; }")
    return "".join(body)


CBF_V4COPY2 = tl.constexpr(_cbf_v4copy(2))  # 4 KB tile (64 x 16 FP32), 128 threads x 16 bytes x 2
CBF_V4COPY4 = tl.constexpr(_cbf_v4copy(4))  # 8 KB tile (64 x 32 FP32)


@triton.jit
def _cbf_cluster_gemm(x_desc, w_desc, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, SLOT_KB: tl.constexpr):
    pid_k = tl.program_id(0)  # rank in the cluster
    pid_n = tl.program_id(1)
    pid_m = tl.program_id(2)
    if SLOT_KB == 8:
        base = tl.inline_asm_elementwise(".shared .align 16 .b8 cbf_slots[8192]; mov.u32 $0, cbf_slots;", "=r,r",
                                         [pid_k], dtype=tl.int32, is_pure=False, pack=1)
    else:
        base = tl.inline_asm_elementwise(".shared .align 16 .b8 cbf_slots[16384]; mov.u32 $0, cbf_slots;", "=r,r",
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
            tl.inline_asm_elementwise(CBF_V4COPY2, "=r,r,r", [base, dst], dtype=tl.int32, is_pure=False, pack=1)
        else:
            tl.inline_asm_elementwise(CBF_V4COPY4, "=r,r,r", [base, dst], dtype=tl.int32, is_pure=False, pack=1)
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
        out_ptrs = out_ptr + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
        tl.store(out_ptrs, tot.to(tl.bfloat16), mask=offs_m[None, :] < M)


# BLOCK_N, BLOCK_K, SPLIT_K (= cluster size), num_warps (4: the copy assumes 128 threads), num_stages
CBF_CLUSTER = (64, 256, 3, 4, 4)
CBF_CLUSTER_UNPOOL8 = (64, 256, 4, 4, 4)  # unpooler slice (1024 rows) at 2-8 tokens: 16 tiles x 4-CTA clusters


def _cbf_launch_cluster(x, w, n, state, cfg, block_m):
    from triton import knobs
    block_n, block_k, split_k, warps, stages = cfg
    m, k = x.shape
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    x_desc = TensorDescriptor.from_tensor(x, [block_m, block_k])
    w_desc = TensorDescriptor.from_tensor(w, [block_n, block_k])
    grid = (1, n // block_n, triton.cdiv(m, block_m))  # axis 0 is the cluster: split_k CTAs at launch
    slot_kb = 8 if (split_k - 1) * block_n * block_m * 4 <= 8192 else 16
    args = (x_desc, w_desc, out, m, n, k, block_m, block_n, block_k, split_k, slot_kb)  # every parameter, in order
    key = (cfg, block_m, m)  # m: Triton specialises integer divisibility
    kern = state["cbf_cluster_kernels"].get(key)
    if kern is None:
        if n % block_n or k % block_k or warps != 4 or block_n * block_m * 4 not in (4096, 8192):
            raise ValueError("cluster kernel: unsupported shape or tiling")
        ck = _cbf_cluster_gemm.warmup(*args[:6], BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                                      SPLIT_K=split_k, SLOT_KB=slot_kb, num_warps=warps, num_stages=stages, grid=grid)
        ck._init_handles()
        pm = list(ck.packed_metadata)  # (num_warps, num_ctas, shared)
        if len(pm) != 3 or pm[1] != 1 or pm[2] <= 49152:  # the 48 KB opt-in (static slots) is done by Triton above it
            raise ValueError("cluster kernel: unexpected launch metadata %r" % (pm,))
        pm[1] = split_k
        kern = state["cbf_cluster_kernels"][key] = (ck, tuple(pm))
    ck, pm = kern
    stream = torch.cuda.current_stream().cuda_stream
    ck.run(grid[0], grid[1], grid[2], stream, ck.function, pm, ck.launch_metadata(grid, stream, *args),
           knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *args)
    return out
# ---- end of cluster block ----


# ---- router 8-256 tokens: programmatic dependent launch between c043's partials and reduce kernels ----
@triton.jit
def _cbf_router_partials_pdl(X, HI, LO, PARTIAL, M, K: tl.constexpr, N: tl.constexpr, SXM, SXK,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                             SPLIT_K: tl.constexpr, TRIG: tl.constexpr):
    """_cbf_router_partials plus griddepcontrol.launch_dependents (TRIG 1: at the start, 2: after the K loop)."""
    pid_n = tl.program_id(0)
    split = tl.program_id(1)
    pid_m = tl.program_id(2)
    t = 0
    if TRIG == 1:
        t = tl.inline_asm_elementwise("griddepcontrol.launch_dependents; mov.u32 $0, 0;", "=r,r", [pid_n],
                                      dtype=tl.int32, is_pure=False, pack=1)
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
    if TRIG == 2:
        t = tl.inline_asm_elementwise("griddepcontrol.launch_dependents; mov.u32 $0, 0;", "=r,r", [pid_n],
                                      dtype=tl.int32, is_pure=False, pack=1)
    off = rows[:, None] * N + cols[None, :] + t
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(PARTIAL + split * M * N + off, acc_hi + acc_lo, mask=mask)


@triton.jit
def _cbf_router_reduce_pdl(PARTIAL, OUT, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    """_cbf_router_reduce behind griddepcontrol.wait (the partials grid has completed and its writes are visible)."""
    w = tl.inline_asm_elementwise("griddepcontrol.wait; mov.u32 $0, 0;", "=r,r", [tl.program_id(0)],
                                  dtype=tl.int32, is_pure=False, pack=1)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK) + w
    splits = tl.arange(0, SPLIT_K)
    values = tl.load(PARTIAL + splits[:, None] * MN + offsets[None, :], mask=offsets[None, :] < MN, other=0)
    tl.store(OUT + offsets, tl.sum(values, axis=0), mask=offsets < MN)


def _cbf_launch_router_pdl(x, n, state, cfg):
    block_n, block_k, split_k, warps, stages, block_m = cfg
    m, k = x.shape
    if split_k * m * n > state["cbf_router_partials"].numel():
        raise ValueError("router scratch too small for this shape")
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _cbf_router_partials_pdl[(triton.cdiv(n, block_n), split_k, triton.cdiv(m, block_m))](
        x, state["router_hi_t"], state["router_lo_t"], state["cbf_router_partials"], m, k, n, x.stride(0), x.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k, TRIG=2 if m == 256 else 1,
        num_warps=warps, num_stages=stages)
    _cbf_router_reduce_pdl[(triton.cdiv(m * n, 128),)](state["cbf_router_partials"], out, m * n, SPLIT_K=split_k,
                                                       BLOCK=128, num_warps=4, launch_pdl=True)
    return out
# ---- end of PDL block ----


# ---- tile-contiguous weights (H1a/H1b): W [N, K] stored as [N/BN, K/BK, BN, BK] rows of BK, made in prepare() ----
@triton.jit
def _cbf_skinny_gemm_tma_tiled(x_desc, w_desc, w2_desc, out_ptr, part_ptr, cnt_ptr, M, N, K,
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
    for kb in range(kb0, kb1):
        xt = x_desc.load([m0, k]).T
        acc = tl.dot(w_desc.load([(pid_n * kb_total + kb) * BLOCK_N, 0]), xt, acc)
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


def _cbf_tile_weights(w, bn, bk):
    """Lossless repack: the same BF16 values, laid out so tile (n, k) is rows (n * K/bk + k) * bn ... + bn of BK."""
    n, k = w.shape
    return w.view(n // bn, bn, k // bk, bk).permute(0, 2, 1, 3).contiguous().view(-1, bk)


def _cbf_launch_tma_tiled(x, wt, n, state, cfg, block_m):
    m, k = x.shape
    block_n, block_k, split_k, warps, stages = cfg
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    x_desc = TensorDescriptor.from_tensor(x, [block_m, block_k])
    w_desc = TensorDescriptor.from_tensor(wt, [block_n, block_k])
    _cbf_skinny_gemm_tma_tiled[(triton.cdiv(m, block_m), n // block_n, split_k)](
        x_desc, w_desc, w_desc, out, state["cbf_partials"], state["cbf_counters"], m, n, k, BLOCK_M=block_m,
        BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k, HILO=False, OUT_F32=False, num_warps=warps,
        num_stages=stages)
    return out


CBF_TILED = {"unpooler": (64, 256, 1, 4, 4), "pooler": (64, 256, 1, 4, 4)}  # unpooler 9-32, pooler 2-32 tokens
# ---- end of tiled block ----


# ---- 128-token split-K GEMM on tile-contiguous weights (part H1e): same kernel, tiling and K split as c043's ----
@triton.jit
def _cbf_lm_partials_tiled(x_desc, w_desc, part_ptr, out_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
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
    for kb in range(kb0, kb1):
        a = x_desc.load([m0, k])
        b = w_desc.load([(pid_n * kb_total + kb) * BLOCK_N, 0])
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


def _cbf_launch_lm_tiled(x, wt, n, state, cfg):
    block_n, block_k, split_k, warps, block_m = cfg
    m, k = x.shape
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if split_k > 1 and split_k * m * n > state["cbf_lm_partials"].numel():
        raise ValueError("split-K scratch too small for this shape")
    x_desc = TensorDescriptor.from_tensor(x, [block_m, block_k])
    w_desc = TensorDescriptor.from_tensor(wt, [block_n, block_k])
    _cbf_lm_partials_tiled[(n // block_n, split_k, triton.cdiv(m, block_m))](
        x_desc, w_desc, state["cbf_lm_partials"], out, m, n, k, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        SPLIT_K=split_k, num_warps=warps, num_stages=state["cbf_lm_stages"])
    if split_k > 1:
        _cbf_lm_reduce[(triton.cdiv(m * n, 1024),)](state["cbf_lm_partials"], out, m * n, SPLIT_K=split_k,
                                                    BLOCK=1024, num_warps=4)
    return out
# ---- end of tiled split-K block ----


def _cbf_prepare(state, device):
    state["cbf_partials"] = torch.empty(1 << 20, dtype=torch.float32, device=device)
    state["cbf_counters"] = torch.zeros(4096, dtype=torch.int32, device=device)
    state["cbf_lm_partials"] = torch.empty(4 << 20, dtype=torch.float32, device=device)  # 16 MiB of FP32 partials
    state["cbf_router_partials"] = torch.empty(1 << 20, dtype=torch.float32, device=device)  # 4 MiB
    # 4 pipeline stages need 131 KB of shared memory (H200: 228 KB); 3 elsewhere (e.g. 101 KB on sm_120)
    state["cbf_lm_stages"] = 4 if torch.cuda.get_device_capability(device)[0] == 9 else 3
    state["cbf_cluster_kernels"] = {}
    if TensorDescriptor is not None:
        state["cbf_tiled"] = {layer: _cbf_tile_weights(state[layer], cfg[0], cfg[1]) for layer, cfg in CBF_TILED.items()}
        state["cbf_tiled_lm"] = {"qkv_gate": _cbf_tile_weights(state["qkv_gate"], CBF_LM128["qkv_gate"][0],
                                                                CBF_LM128["qkv_gate"][1])}
    state["cbf_pdl_ok"] = torch.cuda.get_device_capability(device)[0] >= 9  # griddepcontrol: sm_90+
    state["cbf_cluster_ok"] = torch.cuda.get_device_capability(device)[0] == 9  # Hopper (227 KB shared memory)


def _cbf_dispatch(x, layer, n, state):
    """This block's output for the call, or None when the case is not one it serves."""
    m = x.shape[0]
    if layer == "router" and m in CBF_ROUTER_HILO:
        if state["cbf_pdl_ok"]:
            try:
                return _cbf_launch_router_pdl(x, n, state, CBF_ROUTER_HILO[m])
            except Exception as exc:  # first use happens before capture; c043's launches serve the router from then on
                state["cbf_pdl_ok"] = False
                print("c047: dependent launch unavailable (%s: %s); the router keeps c043's launches" % (type(exc).__name__, exc))
        return _cbf_launch_router(x, n, state, CBF_ROUTER_HILO[m])
    if m == 128 and layer == "pooler":
        # cuBLAS on swapped operands; the column-major view has the required shape and dtype (jobs h38, h54b)
        return torch.mm(state[layer][:n], x.t()).t()
    if m == 128 and layer == "qkv_gate" and "cbf_tiled_lm" in state:
        return _cbf_launch_lm_tiled(x, state["cbf_tiled_lm"][layer], n, state, CBF_LM128[layer])
    if m == 128 and layer in CBF_LM128 and TensorDescriptor is not None:
        return _cbf_launch_lm(x, state[layer][:n], n, state, CBF_LM128[layer])
    if m > 32:
        return None
    if layer == "unpooler" and 1 < m <= 8:
        if state["cbf_cluster_ok"] and TensorDescriptor is not None:
            try:
                return _cbf_launch_cluster(x, state[layer][:n], n, state, CBF_CLUSTER_UNPOOL8, 16)
            except Exception as exc:  # first use happens before capture; c043's kernels serve from then on
                state["cbf_cluster_ok"] = False
                print("c048: cluster kernel unavailable (%s: %s); c043's kernels serve" % (type(exc).__name__, exc))
        return _cbf_launch(x, state[layer][:n], n, state, CBF_UNPOOL8, 16)
    if m == 1 and layer in CBF_GEMV1:
        return _cbf_launch_gemv(x, state[layer][:n], n, CBF_GEMV1[layer])
    if m == 1 and layer == "router":
        return _cbf_launch_router1(x, state["router"], n, CBF_ROUTER1_ROWS)
    if layer == "qkv_gate" and 1 < m <= 32 and state["cbf_cluster_ok"] and TensorDescriptor is not None:
        try:
            return _cbf_launch_cluster(x, state[layer][:n], n, state, CBF_CLUSTER, max(16, triton.next_power_of_2(m)))
        except Exception as exc:  # first use happens before capture; c043's kernels serve qkv_gate from then on
            state["cbf_cluster_ok"] = False
            print("c046: cluster kernel unavailable (%s: %s); qkv_gate keeps c043's kernels" % (type(exc).__name__, exc))
    if 16 < m and layer in CBF_TMA32 and TensorDescriptor is not None:
        return _cbf_launch_tma(x, state[layer][:n], None, n, state, CBF_TMA32[layer], max(16, triton.next_power_of_2(m)),
                               torch.bfloat16)
    if 8 < m <= 16 and layer in CBF_TMA16 and TensorDescriptor is not None:
        return _cbf_launch_tma(x, state[layer][:n], None, n, state, CBF_TMA16[layer], 16, torch.bfloat16)
    if ((layer == "unpooler" and 8 < m) or (layer == "pooler" and 1 < m)) and "cbf_tiled" in state:
        return _cbf_launch_tma_tiled(x, state["cbf_tiled"][layer], n, state, CBF_TILED[layer],
                                     max(16, triton.next_power_of_2(m)))
    if 1 < m and layer in CBF_TMA and TensorDescriptor is not None:
        return _cbf_launch_tma(x, state[layer][:n], None, n, state, CBF_TMA[layer], max(16, triton.next_power_of_2(m)),
                               torch.bfloat16)
    if (layer, 1 if m == 1 else 16 if m <= 16 else 32) not in CBF_ENABLED:
        return None
    cfg = CBF_STREAM.get((layer, 16 if m <= 16 else 32))
    if cfg is None:
        return None
    return _cbf_launch(x, state[layer][:n], n, state, cfg, max(16, triton.next_power_of_2(m)))
# ---- end of block ----


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    state["_pooler_use_library"] = torch.cuda.get_device_capability(weights["pooler"].device)[0] == 9
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    _cbf_prepare(state, router.device)
    return state


@triton.jit
def _router_single_token(X, W, Y, K: tl.constexpr, SX: tl.constexpr,
                         SW: tl.constexpr, TILE_K: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, TILE_K)
    accum = tl.full((TILE_K,), 0, tl.float32)
    for tile in range(triton.cdiv(K, TILE_K)):
        k = tile * TILE_K + columns
        x = tl.load(X + k * SX, mask=k < K, other=0).to(tl.float32)
        w = tl.load(W + row * SW + k, mask=k < K, other=0)
        accum += w * x
    tl.store(Y + row, tl.sum(accum, axis=0))


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


@triton.jit
def _router_sixteen_tokens(X, W, Y, K: tl.constexpr, N: tl.constexpr,
                         SXM: tl.constexpr, SXK: tl.constexpr,
                         SW: tl.constexpr, TILE_K: tl.constexpr):
    row = tl.program_id(0)
    tokens = tl.arange(0, 16)
    columns = tl.arange(0, TILE_K)
    accum = tl.full((16, TILE_K), 0, tl.float32)
    for tile in range(triton.cdiv(K, TILE_K)):
        k = tile * TILE_K + columns
        x = tl.load(X + tokens[:, None] * SXM + k[None, :] * SXK,
                    mask=k[None, :] < K, other=0).to(tl.float32)
        w = tl.load(W + row * SW + k, mask=k < K, other=0)
        accum += x * w[None, :]
    tl.store(Y + tokens * N + row, tl.sum(accum, axis=1))


@triton.jit
def _router_m16_hilo_partials(X, HI, LO, PARTIAL, K: tl.constexpr,
                             N: tl.constexpr, SXM: tl.constexpr,
                             SXK: tl.constexpr, BLOCK_N: tl.constexpr,
                             BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    rows = tl.arange(0, 16)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.program_id(1)
    ks = tl.arange(0, BLOCK_K)
    iterations = triton.cdiv(K, SPLIT_K * BLOCK_K)
    acc_hi = tl.full((16, BLOCK_N), 0, tl.float32)
    acc_lo = tl.full((16, BLOCK_N), 0, tl.float32)
    for tile in range(iterations):
        kk = (split * iterations + tile) * BLOCK_K + ks
        xx = tl.load(X + rows[:, None] * SXM + kk[None, :] * SXK,
                     mask=kk[None, :] < K, other=0)
        hi = tl.load(HI + kk[:, None] * N + cols[None, :],
                     mask=(kk[:, None] < K) & (cols[None, :] < N), other=0)
        lo = tl.load(LO + kk[:, None] * N + cols[None, :],
                     mask=(kk[:, None] < K) & (cols[None, :] < N), other=0)
        acc_hi = tl.dot(xx, hi, acc_hi, input_precision="ieee")
        acc_lo = tl.dot(xx, lo, acc_lo, input_precision="ieee")
    tl.store(PARTIAL + split * 16 * N + rows[:, None] * N + cols[None, :],
             acc_hi + acc_lo, mask=cols[None, :] < N)



@triton.jit
def _router_m16_hilo_reduce(PARTIAL, OUT, N: tl.constexpr,
                           SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT_K)
    values = tl.load(PARTIAL + splits[:, None] * 16 * N + offsets[None, :],
                     mask=offsets[None, :] < 16 * N, other=0)
    tl.store(OUT + offsets, tl.sum(values, axis=0), mask=offsets < 16 * N)



@triton.jit
def _router_m32_hilo_partials(X, HI, LO, PARTIAL, K: tl.constexpr,
                             N: tl.constexpr, SXM: tl.constexpr,
                             SXK: tl.constexpr, BLOCK_N: tl.constexpr,
                             BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    rows = tl.arange(0, 32)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.program_id(1)
    ks = tl.arange(0, BLOCK_K)
    iterations = triton.cdiv(K, SPLIT_K * BLOCK_K)
    acc_hi = tl.full((32, BLOCK_N), 0, tl.float32)
    acc_lo = tl.full((32, BLOCK_N), 0, tl.float32)
    for tile in range(iterations):
        kk = (split * iterations + tile) * BLOCK_K + ks
        xx = tl.load(X + rows[:, None] * SXM + kk[None, :] * SXK,
                     mask=kk[None, :] < K, other=0)
        hi = tl.load(HI + kk[:, None] * N + cols[None, :],
                     mask=(kk[:, None] < K) & (cols[None, :] < N), other=0)
        lo = tl.load(LO + kk[:, None] * N + cols[None, :],
                     mask=(kk[:, None] < K) & (cols[None, :] < N), other=0)
        acc_hi = tl.dot(xx, hi, acc_hi, input_precision="ieee")
        acc_lo = tl.dot(xx, lo, acc_lo, input_precision="ieee")
    tl.store(PARTIAL + split * 32 * N + rows[:, None] * N + cols[None, :],
             acc_hi + acc_lo, mask=cols[None, :] < N)


@triton.jit
def _router_m32_hilo_reduce(PARTIAL, OUT, N: tl.constexpr,
                           SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT_K)
    values = tl.load(PARTIAL + splits[:, None] * 32 * N + offsets[None, :],
                     mask=offsets[None, :] < 32 * N, other=0)
    tl.store(OUT + offsets, tl.sum(values, axis=0), mask=offsets < 32 * N)


@triton.jit
def _router_m128_hilo_partials(X, HI, LO, PARTIAL, K: tl.constexpr,
                             N: tl.constexpr, SXM: tl.constexpr,
                             SXK: tl.constexpr, BLOCK_N: tl.constexpr,
                             BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    rows = tl.program_id(2) * 32 + tl.arange(0, 32)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.program_id(1)
    ks = tl.arange(0, BLOCK_K)
    iterations = triton.cdiv(K, SPLIT_K * BLOCK_K)
    acc_hi = tl.full((32, BLOCK_N), 0, tl.float32)
    acc_lo = tl.full((32, BLOCK_N), 0, tl.float32)
    for tile in range(iterations):
        kk = (split * iterations + tile) * BLOCK_K + ks
        xx = tl.load(X + rows[:, None] * SXM + kk[None, :] * SXK,
                     mask=kk[None, :] < K, other=0)
        hi = tl.load(HI + kk[:, None] * N + cols[None, :],
                     mask=(kk[:, None] < K) & (cols[None, :] < N), other=0)
        lo = tl.load(LO + kk[:, None] * N + cols[None, :],
                     mask=(kk[:, None] < K) & (cols[None, :] < N), other=0)
        acc_hi = tl.dot(xx, hi, acc_hi, input_precision="ieee")
        acc_lo = tl.dot(xx, lo, acc_lo, input_precision="ieee")
    tl.store(PARTIAL + split * 128 * N + rows[:, None] * N + cols[None, :],
             acc_hi + acc_lo, mask=cols[None, :] < N)


@triton.jit
def _router_m128_hilo_reduce(PARTIAL, OUT, N: tl.constexpr,
                           SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT_K)
    values = tl.load(PARTIAL + splits[:, None] * 128 * N + offsets[None, :],
                     mask=offsets[None, :] < 128 * N, other=0)
    tl.store(OUT + offsets, tl.sum(values, axis=0), mask=offsets < 128 * N)


@triton.jit
def _router_m256_hilo_partials(X, HI, LO, PARTIAL, K: tl.constexpr,
                             N: tl.constexpr, SXM: tl.constexpr,
                             SXK: tl.constexpr, BLOCK_N: tl.constexpr,
                             BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    rows = tl.program_id(2) * 64 + tl.arange(0, 64)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.program_id(1)
    ks = tl.arange(0, BLOCK_K)
    iterations = triton.cdiv(K, SPLIT_K * BLOCK_K)
    acc_hi = tl.full((64, BLOCK_N), 0, tl.float32)
    acc_lo = tl.full((64, BLOCK_N), 0, tl.float32)
    for tile in range(iterations):
        kk = (split * iterations + tile) * BLOCK_K + ks
        xx = tl.load(X + rows[:, None] * SXM + kk[None, :] * SXK,
                     mask=kk[None, :] < K, other=0)
        hi = tl.load(HI + kk[:, None] * N + cols[None, :],
                     mask=(kk[:, None] < K) & (cols[None, :] < N), other=0)
        lo = tl.load(LO + kk[:, None] * N + cols[None, :],
                     mask=(kk[:, None] < K) & (cols[None, :] < N), other=0)
        acc_hi = tl.dot(xx, hi, acc_hi, input_precision="ieee")
        acc_lo = tl.dot(xx, lo, acc_lo, input_precision="ieee")
    tl.store(PARTIAL + split * 256 * N + rows[:, None] * N + cols[None, :],
             acc_hi + acc_lo, mask=cols[None, :] < N)


@triton.jit
def _router_m256_hilo_reduce(PARTIAL, OUT, N: tl.constexpr,
                           SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT_K)
    values = tl.load(PARTIAL + splits[:, None] * 256 * N + offsets[None, :],
                     mask=offsets[None, :] < 256 * N, other=0)
    tl.store(OUT + offsets, tl.sum(values, axis=0), mask=offsets < 256 * N)


@triton.jit
def _pooler_m16_direct(X, W, OUT, K: tl.constexpr, N: tl.constexpr,
                       SXM: tl.constexpr, SXK: tl.constexpr, SW: tl.constexpr,
                       BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.arange(0, 16)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    acc = tl.full((16, BLOCK_N), 0, tl.float32)
    for tile in range(triton.cdiv(K, BLOCK_K)):
        kk = tile * BLOCK_K + ks
        xx = tl.load(X + rows[:, None] * SXM + kk[None, :] * SXK,
                     mask=kk[None, :] < K, other=0)
        ww = tl.load(W + cols[:, None] * SW + kk[None, :],
                     mask=(cols[:, None] < N) & (kk[None, :] < K), other=0)
        acc = tl.dot(xx, tl.trans(ww), acc, input_precision="ieee")
    tl.store(OUT + rows[:, None] * N + cols[None, :], acc,
             mask=cols[None, :] < N)


@triton.jit
def _pooler_m8_direct(X, W, OUT, K: tl.constexpr, N: tl.constexpr,
                       SXM: tl.constexpr, SXK: tl.constexpr, SW: tl.constexpr,
                       BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.arange(0, 16)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    acc = tl.full((16, BLOCK_N), 0, tl.float32)
    for tile in range(triton.cdiv(K, BLOCK_K)):
        kk = tile * BLOCK_K + ks
        xx = tl.load(X + rows[:, None] * SXM + kk[None, :] * SXK,
                     mask=(rows[:, None] < 8) & (kk[None, :] < K), other=0)
        ww = tl.load(W + cols[:, None] * SW + kk[None, :],
                     mask=(cols[:, None] < N) & (kk[None, :] < K), other=0)
        acc = tl.dot(xx, tl.trans(ww), acc, input_precision="ieee")
    tl.store(OUT + rows[:, None] * N + cols[None, :], acc,
             mask=(rows[:, None] < 8) & (cols[None, :] < N))


@triton.jit
def _pooler_single_token(X, W, Y, K: tl.constexpr, N: tl.constexpr,
                         SX: tl.constexpr, SW: tl.constexpr,
                         ROWS: tl.constexpr, TILE_K: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, TILE_K)
    accum = tl.full((ROWS, TILE_K), 0, tl.float32)
    for tile in range(triton.cdiv(K, TILE_K)):
        k = tile * TILE_K + columns
        x = tl.load(X + k * SX, mask=k < K, other=0).to(tl.float32)
        w = tl.load(W + rows[:, None] * SW + k[None, :],
                    mask=(rows[:, None] < N) & (k[None, :] < K),
                    other=0, cache_modifier=".cg").to(tl.float32)
        accum += w * x[None, :]
    out = tl.sum(accum, axis=1)
    tl.store(Y + rows, out, mask=rows < N)


def kernel_fn(data, state):
    x, layer = data["x"], data["config"]["layer"]
    out = _cbf_dispatch(x, layer, data["config"]["n"], state)
    if out is not None:  # claude-bf16 qkv_gate at 1-16 tokens (CBF_ENABLED), pooler at 2-32 tokens (CBF_TMA),
        # unpooler at one token (CBF_GEMV1)
        return out
    if layer == "pooler" and x.shape[0] == 1:
        weight = state["pooler"]
        n = data["config"]["n"]
        out = torch.empty((1, n), device=x.device, dtype=torch.bfloat16)
        _pooler_single_token[(triton.cdiv(n, 4),)](
            x, weight, out, x.shape[1], n, x.stride(1), weight.stride(0),
            ROWS=4, TILE_K=2048, num_warps=16,
        )
        return out
    if layer == "router" and x.shape[0] == 1:
        weight = state["router"]
        n = data["config"]["n"]
        out = torch.empty((1, n), device=x.device, dtype=torch.float32)
        _router_single_token[(n,)](
            x, weight, out, x.shape[1], x.stride(1), weight.stride(0),
            TILE_K=2048, num_warps=4,
        )
        return out
    if layer == "router" and x.shape[0] == 8:
        weight = state["router"]
        n = data["config"]["n"]
        out = torch.empty((8, n), device=x.device, dtype=torch.float32)
        _router_eight_tokens[(n,)](
            x, weight, out, x.shape[1], n, x.stride(0), x.stride(1),
            weight.stride(0), TILE_K=1024, num_warps=8,
        )
        return out
    if layer == "router" and x.shape[0] == 16:
        n = data["config"]["n"]
        partial = torch.empty((16, 16, n), device=x.device, dtype=torch.float32)
        out = torch.empty((16, n), device=x.device, dtype=torch.float32)
        _router_m16_hilo_partials[(triton.cdiv(n, 32), 16)](
            x, state["router_hi_t"], state["router_lo_t"], partial,
            x.shape[1], n, x.stride(0), x.stride(1),
            BLOCK_N=32, BLOCK_K=64, SPLIT_K=16, num_warps=4,
        )
        _router_m16_hilo_reduce[(triton.cdiv(16 * n, 128),)](
            partial, out, n, SPLIT_K=16, BLOCK=128, num_warps=4,
        )
        return out
    if layer == "router" and x.shape[0] == 32:
        n = data["config"]["n"]
        partial = torch.empty((32, 32, n), device=x.device, dtype=torch.float32)
        out = torch.empty((32, n), device=x.device, dtype=torch.float32)
        _router_m32_hilo_partials[(triton.cdiv(n, 32), 32)](
            x, state["router_hi_t"], state["router_lo_t"], partial,
            x.shape[1], n, x.stride(0), x.stride(1),
            BLOCK_N=32, BLOCK_K=64, SPLIT_K=32, num_warps=4,
        )
        _router_m32_hilo_reduce[(triton.cdiv(32 * n, 128),)](
            partial, out, n, SPLIT_K=32, BLOCK=128, num_warps=4,
        )
        return out
    if layer == "router" and x.shape[0] == 256:
        n = data["config"]["n"]
        partial = torch.empty((8, 256, n), device=x.device, dtype=torch.float32)
        out = torch.empty((256, n), device=x.device, dtype=torch.float32)
        _router_m256_hilo_partials[(triton.cdiv(n, 64), 8, 4)](
            x, state["router_hi_t"], state["router_lo_t"], partial,
            x.shape[1], n, x.stride(0), x.stride(1),
            BLOCK_N=64, BLOCK_K=64, SPLIT_K=8, num_warps=4,
        )
        _router_m256_hilo_reduce[(triton.cdiv(256 * n, 128),)](
            partial, out, n, SPLIT_K=8, BLOCK=128, num_warps=4,
        )
        return out
    if layer == "router" and x.shape[0] == 128:
        n = data["config"]["n"]
        partial = torch.empty((8, 128, n), device=x.device, dtype=torch.float32)
        out = torch.empty((128, n), device=x.device, dtype=torch.float32)
        _router_m128_hilo_partials[(triton.cdiv(n, 64), 8, 4)](
            x, state["router_hi_t"], state["router_lo_t"], partial,
            x.shape[1], n, x.stride(0), x.stride(1),
            BLOCK_N=64, BLOCK_K=64, SPLIT_K=8, num_warps=4,
        )
        _router_m128_hilo_reduce[(triton.cdiv(128 * n, 128),)](
            partial, out, n, SPLIT_K=8, BLOCK=128, num_warps=4,
        )
        return out
    if layer == "pooler" and x.shape[0] == 8 and not state["_pooler_use_library"]:
        n = data["config"]["n"]
        weight = state["pooler"]
        out = torch.empty((8, n), device=x.device, dtype=torch.bfloat16)
        _pooler_m8_direct[(triton.cdiv(n, 16),)](
            x, weight, out, x.shape[1], n, x.stride(0), x.stride(1),
            weight.stride(0), BLOCK_N=16, BLOCK_K=128, num_warps=4,
        )
        return out
    if layer == "pooler" and x.shape[0] == 16 and not state["_pooler_use_library"]:
        n = data["config"]["n"]
        weight = state["pooler"]
        out = torch.empty((16, n), device=x.device, dtype=torch.bfloat16)
        _pooler_m16_direct[(triton.cdiv(n, 16),)](
            x, weight, out, x.shape[1], n, x.stride(0), x.stride(1),
            weight.stride(0), BLOCK_N=16, BLOCK_K=128, num_warps=4,
        )
        return out
    if layer != "router":
        # The unpooler's slice is its first N rows (all of them above 8 tokens).
        return _bf16_gemm_dispatch_impl(x, state[layer][: data["config"]["n"]], None)
    if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc
