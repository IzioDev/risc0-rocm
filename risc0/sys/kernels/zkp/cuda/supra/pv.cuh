// Copyright 2026 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Poseidon2 (BabyBear, width 24) row hashing and folding with lazy Montgomery reductions:
// fewer instructions than poseidon2.cuh for the same results (same I/O: Montgomery-form u32).
// CUDA uses PTX carry chains; ROCm the same arithmetic in 64-bit C++, which wraps identically.
//
// Value bounds, checked exhaustively over all canonical inputs (p/2^32 = 0.46875):
// - mred(t) = t * 2^-32 mod p, in [0, t/2^32 + p); exact (congruent) whenever the high word
//   of t is at most 0x87fffffe (~1.133p), e.g. for t < 2^32 * p.
// - sbox: x^2 < 1.469p, x^3 < 1.689p before its reduction, x^4 < 2.011p, x^7 < 1.915p before
//   its reduction: every product stays in mred's domain.
// - Partial rounds keep the cells below 1.87p (at most 1.8693p over all DH_i, x in [0, 2p)):
//   cells < 2p hold for any number of rounds, by induction.
//
// ROCm (RDNA) path: the 32-bit multiplies (v_mad_u64_u32, v_mad_i64_i32, v_mul_lo/hi_u32) issue at
// a quarter of the add rate, so each product costs 2 of them instead of 3, with a signed REDC:
// sredc(t) = (t - m*p) / 2^32 with m = (int32)(t * p^-1 mod 2^32), p^-1 = 2^31 + 2^27 + 1, m
// from two shift-adds (v_lshl_add_u32) and t - m*p from one v_mad_i64_i32 (m, -p, t). Since
// t - m*p == 0 mod 2^32 the high word is the exact quotient, == t * 2^-32 (mod p), whenever
// |t - m*p| < 2^63; |sredc(t)| <= (|t| + 2^31 p) / 2^32. Values are int32 in (-p, p) inside the
// S-box and the partial rounds and canonical u32 in the linear layers:
// - |a|, |b| <= p gives |a*b| <= p^2 and |sredc(a*b)| <= p^2/2^32 + p/2 < 0.96875p + 1 < p:
//   every S-box step (input x + RC - p in [-p, p)) stays in (-p, p), as does x^7, which
//   canon() = min_u32(r, r + p) maps to [0, p).
// - For even t, lo(t) * (2^27 + 1) is already t * p^-1 mod 2^32 (lo * 2^31 == 0), so products by
//   a constant use one shift-add for m: each constant is replaced by its even representative in
//   (-p, p) (DH_i or DH_i - p). The partial rounds' addend a == S * 2^32 (mod p) comes without
//   a reduction: with S = 2^24 * c + l, l = S mod 2^24, a = l * 2^32 + c * (2^56 mod p), even
//   (2^56 mod p is), |c| <= 2881 since |S| < 24p < 2^36, so |a| < 2^56 + 2^43.
// - Partial-round cells: e'_i = sredc(E_i * e_i + a) with |E_i| <= 0.93p, |e_i| < p,
//   |a| < 2^56 + 2^43: |t - m*p| < 0.93p^2 + 2^56.01 + 2^31 p < 2^63 and
//   |e'_i| < 0.95p < p, so |cells| < p holds for any number of rounds, by induction (the S-box
//   cell is canonicalized before adding its round constant).

#pragma once

#include <cstdint>

#include "pv_consts.h"

namespace pv {

constexpr uint32_t P = 0x78000001u, M0 = 0x77ffffffu; // M0 = -p^-1 mod 2^32

#ifdef __HIPCC__
// Signed-REDC helpers for the ROCm path (see the header comment for the bounds).

// An opaque copy: keeps LLVM from folding the shift-adds of sredc() into a v_mul_lo_u32.
__device__ __forceinline__ uint32_t opq(uint32_t x) {
  asm("" : "+v"(x));
  return x;
}

// (t - m*p) / 2^32 with m = (int32)(t * p^-1 mod 2^32): two v_lshl_add_u32 + one v_mad_i64_i32.
__device__ __forceinline__ int32_t sredc(int64_t t) {
  uint32_t lo = uint32_t(t);
  int32_t m = int32_t(opq(lo + (lo << 27)) + (lo << 31));
  return int32_t(uint64_t(t + int64_t(m) * -int64_t(P)) >> 32);
}

// sredc() for even t: m = lo * (2^27 + 1), one v_lshl_add_u32.
__device__ __forceinline__ int32_t sredc_even(int64_t t) {
  uint32_t lo = uint32_t(t);
  int32_t m = int32_t(lo + (lo << 27));
  return int32_t(uint64_t(t + int64_t(m) * -int64_t(P)) >> 32);
}

__device__ __forceinline__ int32_t smul(int32_t a, int32_t b) {
  return sredc(int64_t(a) * b);
}

// (-p, p) -> [0, p).
__device__ __forceinline__ uint32_t canon(int32_t r) {
  uint32_t u = uint32_t(r);
  return __builtin_elementwise_min(u, u + P);
}

// y in [-p, p) -> y^7 in (-p, p).
__device__ __forceinline__ int32_t ssbox(int32_t y) {
  int32_t y2 = smul(y, y);
  int32_t y3 = smul(y2, y);
  int32_t y4 = smul(y2, y2);
  return smul(y3, y4);
}

// PV_DH (Montgomery-form internal diagonal) as even representatives in (-p, p).
constexpr int32_t PV_DHE[24] = {
    -968772778,  271838616,   -774936828,  1384248088,  1011796206, 426221916,
    985139354,   1760666932,  -1713368876, 224647294,   -656957998, -764689148,
    1866859900,  1132319062,  1622455428,  -871607094,  1242914314, -234514942,
    1178086656,  -1482974038, -1528792916, 316128532,   1581421748, 456216014};
// 2^56 mod p (even).
constexpr int32_t PV_C56 = 0x0dddddde;
#endif

__device__ __forceinline__ uint32_t red1(uint32_t r) { // [0, 2p) -> [0, p)
  return r >= P ? r - P : r;
}

__device__ __forceinline__ uint32_t addm(uint32_t a, uint32_t b) {
  return red1(a + b);
}

// Montgomery reduction (see the bounds above).
__device__ __forceinline__ uint32_t mred(uint64_t t) {
#ifdef __HIPCC__
  // The high word of t + m*p, modulo 2^64 like the PTX carry chain below.
  uint32_t m = uint32_t(t) * M0;
  return uint32_t((t + uint64_t(m) * P) >> 32);
#else
  uint32_t r;
  asm("{\n\t.reg .u32 m, lo, hi, d;\n\tmov.b64 {lo,hi}, %1;\n\tmul.lo.u32 m, lo, %2;\n\t"
      "mad.lo.cc.u32 d, m, %3, lo;\n\tmadc.hi.u32 %0, m, %3, hi;\n\t}"
      : "=r"(r)
      : "l"(t), "n"(M0), "n"(P));
  return r;
#endif
}

__device__ __forceinline__ uint32_t mmul_lazy(uint32_t a, uint32_t b) {
  return mred((uint64_t)a * b);
}

__device__ __forceinline__ uint32_t mmul(uint32_t a, uint32_t b) {
  return red1(mmul_lazy(a, b));
}

// x canonical -> x^7 canonical.
__device__ __forceinline__ uint32_t sbox(uint32_t x) {
  uint32_t x2 = mmul_lazy(x, x);
  uint32_t x3 = mmul(x2, x);
  uint32_t x4 = mmul_lazy(x2, x2);
  return mmul(x3, x4);
}

__device__ __forceinline__ void m4(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  uint32_t t0 = addm(a, b), t1 = addm(c, d);
  uint32_t t2 = addm(addm(b, b), t1), t3 = addm(addm(d, d), t0);
  uint32_t t1_2 = addm(t1, t1), t0_2 = addm(t0, t0);
  uint32_t t4 = addm(addm(t1_2, t1_2), t3), t5 = addm(addm(t0_2, t0_2), t2);
  a = addm(t3, t5);
  b = t5;
  c = addm(t2, t4);
  d = t4;
}

// External (full-round) linear layer.
__device__ __forceinline__ void m_ext(uint32_t* x) {
  uint32_t s[4] = {0, 0, 0, 0};
#pragma unroll
  for (int g = 0; g < 6; g++) {
    m4(x[4 * g], x[4 * g + 1], x[4 * g + 2], x[4 * g + 3]);
#pragma unroll
    for (int j = 0; j < 4; j++)
      s[j] = g ? addm(s[j], x[4 * g + j]) : x[4 * g + j];
  }
#pragma unroll
  for (int i = 0; i < 24; i++)
    x[i] = addm(x[i], s[i & 3]);
}

__device__ __forceinline__ void full_round(uint32_t* x, int r) {
#ifdef __HIPCC__
  // The round constants first: one batch of scalar loads rather than one address per constant.
  uint32_t rc[24];
#pragma unroll
  for (int i = 0; i < 24; i++)
    rc[i] = PV_RCF[r * 24 + i];
#pragma unroll
  for (int i = 0; i < 24; i++)
    x[i] = canon(ssbox(int32_t(x[i] + (rc[i] - P))));
#else
#pragma unroll
  for (int i = 0; i < 24; i++)
    x[i] = sbox(addm(x[i], PV_RCF[r * 24 + i]));
#endif
  m_ext(x);
}

#ifdef __HIPCC__
// S = sum_i e_i for 24 values with |e_i| < 2^31, as its wrapping low word lo (v_add3_u32) and
// tb = sum (e_i >> 24) over the sign-extended top bytes: one v_perm_b32 packs two of them into
// int16 lanes and one v_pk_add_u16 adds them (|lane| <= 12 * 128), about 1.5 ops per term. Then
// S = 2^24 * tb + d with d = sum (e_i mod 2^24) in [0, 24 * 2^24), so d = lo - 2^24 * tb
// (mod 2^32) exactly, and floor(S / 2^24) = tb + (d >> 24).
__device__ __forceinline__ void ssum24(const int32_t* e, uint32_t& lo_out, int32_t& tb_out) {
  typedef unsigned short u16x2 __attribute__((ext_vector_type(2)));
  uint32_t lo = 0;
  u16x2 hp = {0, 0};
#pragma unroll
  for (int i = 0; i < 24; i += 2) {
    lo += uint32_t(e[i]) + uint32_t(e[i + 1]);
    // {e[i] >> 24, e[i + 1] >> 24} as int16 lanes: bytes 3, sign(3), 7, sign(7) of {e[i+1], e[i]}.
    hp += __builtin_bit_cast(u16x2, __builtin_amdgcn_perm(uint32_t(e[i + 1]), uint32_t(e[i]), 0x0b070903u));
  }
  lo_out = lo;
  tb_out = int32_t(int16_t(hp.x)) + int32_t(int16_t(hp.y));
}

// The 21 partial rounds with int32 cells in (-p, p): e'_i = sum_j e_j + DH_i * e_i * 2^-32,
// computed as sredc_even(E_i * e_i + a) with E_i == DH_i and a == S * 2^32 (mod p), both even.
__device__ __forceinline__ void partial_rounds(uint32_t* x) {
  int32_t e[24];
#pragma unroll
  for (int i = 0; i < 24; i++)
    e[i] = int32_t(x[i]);
#pragma unroll 1
  for (int r = 0; r < 21; r++) {
    e[0] = ssbox(int32_t(canon(e[0]) + (PV_RCP[r] - P)));
    uint32_t lo;
    int32_t tb;
    ssum24(e, lo, tb);
    // S = 2^24 * c + l with l = lo mod 2^24 and c = floor(S / 2^24): a = l * 2^32 + c * C56.
    int32_t c = tb + int32_t((lo - (uint32_t(tb) << 24)) >> 24);
    int64_t a = int64_t(uint64_t(lo & 0xffffffu) << 32) + int64_t(c) * PV_C56;
#pragma unroll
    for (int i = 0; i < 24; i++)
      e[i] = sredc_even(int64_t(PV_DHE[i]) * e[i] + a);
  }
#pragma unroll
  for (int i = 0; i < 24; i++)
    x[i] = canon(e[i]);
}
#else
// The 21 partial rounds, with the cells kept in [0, 1.87p) between rounds (no per-cell
// reduction): e'_i = sum_j e_j + DH_i * e_i * 2^-32, computed as mred(DH_i * e_i + w) with
// w = mred(mred(S) * R^3) == S * 2^32 mod p, S = sum_j e_j.
__device__ __forceinline__ void partial_rounds(uint32_t* x) {
#pragma unroll 1
  for (int r = 0; r < 21; r++) {
    x[0] = sbox(addm(red1(x[0]), PV_RCP[r]));
    uint32_t lo, hi;
    asm("add.cc.u32 %0, %2, %3; addc.u32 %1, 0, 0;" : "=r"(lo), "=r"(hi) : "r"(x[0]), "r"(x[1]));
#pragma unroll
    for (int i = 2; i < 24; i++)
      asm("add.cc.u32 %0, %0, %2; addc.u32 %1, %1, 0;" : "+r"(lo), "+r"(hi) : "r"(x[i]));
    uint32_t st = mred(((uint64_t)hi << 32) | lo); // < p + 22
    uint32_t w = mred((uint64_t)st * 0x12f37bfbu); // == S * 2^32 mod p, < 1.074p
#pragma unroll
    for (int i = 0; i < 24; i++)
      x[i] = mred((uint64_t)PV_DH[i] * x[i] + w);
  }
#pragma unroll
  for (int i = 0; i < 24; i++)
    x[i] = red1(x[i]);
}
#endif

__device__ __forceinline__ void permute(uint32_t* x) {
  m_ext(x);
#ifdef __HIPCC__
  // One loop over the 8 full rounds, so their body is emitted once (half the code size).
#pragma unroll 1
  for (int r = 0; r < 8; r++) {
    if (r == 4)
      partial_rounds(x);
    full_round(x, r);
  }
#else
#pragma unroll 1
  for (int r = 0; r < 4; r++)
    full_round(x, r);
  partial_rounds(x);
#pragma unroll 1
  for (int r = 4; r < 8; r++)
    full_round(x, r);
#endif
}

// out[i] = hash of the two digests in[2i], in[2i + 1] (n outputs).
__global__ __launch_bounds__(128, 6) void fold_k(uint4* out, const uint4* in, uint32_t n) {
  uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid >= n)
    return;
  uint32_t x[24];
#pragma unroll
  for (int q = 0; q < 4; q++) {
    uint4 v = in[(size_t)gid * 4 + q];
    x[4 * q] = v.x;
    x[4 * q + 1] = v.y;
    x[4 * q + 2] = v.z;
    x[4 * q + 3] = v.w;
  }
#pragma unroll
  for (int i = 16; i < 24; i++)
    x[i] = 0;
  permute(x);
  out[(size_t)gid * 2] = make_uint4(x[0], x[1], x[2], x[3]);
  out[(size_t)gid * 2 + 1] = make_uint4(x[4], x[5], x[6], x[7]);
}

#ifdef __HIPCC__
// The top layers of a fold tree in one block, n0 <= 128 threads: layers n0, n0/2, ..., 1 of the
// sppark_poseidon2_fold_tree layout (digests: layer of size n at [n, 2n), its inputs at [2n, 4n)).
// Those layers are latency-bound (one permutation each), so this saves the launch gaps.
__global__ __launch_bounds__(128) void fold_tail_k(uint4* nodes, uint32_t n0) {
  uint32_t t = threadIdx.x;
#pragma unroll 1
  for (uint32_t n = n0; n >= 1; n >>= 1) {
    if (t < n) {
      const uint4* in = nodes + 4 * (size_t)n;
      uint32_t x[24];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        uint4 v = in[(size_t)t * 4 + q];
        x[4 * q] = v.x;
        x[4 * q + 1] = v.y;
        x[4 * q + 2] = v.z;
        x[4 * q + 3] = v.w;
      }
#pragma unroll
      for (int i = 16; i < 24; i++)
        x[i] = 0;
      permute(x);
      uint4* out = nodes + 2 * (size_t)n;
      out[(size_t)t * 2] = make_uint4(x[0], x[1], x[2], x[3]);
      out[(size_t)t * 2 + 1] = make_uint4(x[4], x[5], x[6], x[7]);
    }
    __syncthreads();
  }
}
#endif

// out[r] = hash of row r of the column-major dim_x x dim_y matrix m (16 elements per absorb).
__global__ __launch_bounds__(128, 6) void rows_k(uint4* out,
                                                 const uint32_t* __restrict__ m,
                                                 uint32_t dim_x,
                                                 uint32_t dim_y) {
  uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid >= dim_x)
    return;
  uint32_t x[24];
#pragma unroll
  for (int i = 0; i < 24; i++)
    x[i] = 0;
  m += gid;
  uint32_t i = 0;
  do {
#pragma unroll
    for (int j = 0; j < 16; j++, i++)
      x[j] = i < dim_y ? __ldg(m + (size_t)i * dim_x) : 0u;
    permute(x);
  } while (i < dim_y);
  out[(size_t)gid * 2] = make_uint4(x[0], x[1], x[2], x[3]);
  out[(size_t)gid * 2 + 1] = make_uint4(x[4], x[5], x[6], x[7]);
}

} // namespace pv
