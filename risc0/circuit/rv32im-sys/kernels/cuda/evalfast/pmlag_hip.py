#!/usr/bin/env python3
# Copyright 2026 RISC Zero, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lagged poly_mix constant loads in the ROCm eval_check on gfx11 (regen.py: the default variant, after
# smem_hip.py).
#
# Each scalar chain link rk_link(_nf)_k(base, c, k) reads its FpExt constant ef_pm_raw[k] with a
# uniform (SGPR) load. On gfx1100 the AMDGPU scheduler hoists those loads far ahead in s_load_b512
# batches; with ~100 SGPRs free the batches are spilled to VGPR lanes and read back one v_readlane
# at a time (~14k v_readlane per main-phase wave). Here, on gfx11, the link W positions further on
# in the same chain loads its constant through ef_pml(): the table pointer passes through an empty
# asm (SGPR in and out, no instruction) that also takes the accumulator of this link as an operand,
# so the load cannot be issued before that link's accumulator exists: at most W links of constants
# are in flight. Only the scalar form of an EF_W((gfx12 form), scalar form) line is changed, and
# other targets read rk_link(_nf)_k itself (EF_DEP is 0 there). Measured on the ef harness at po2 20
# (interleaved medians, bit-exact): RX 7900 XTX 136.3 -> 123.5 ms (W 4).
#
# define(s) adds only the helpers (both variants carry them, so archsel_hip.py sees no conditional
# directive in a differing run); convert(s) adds the helpers and rewrites the links.
import re

HELP = r'''// pmlag_hip.py: on gfx11 each scalar chain link's constant loaded at most W links ahead.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__)
#define EF_DEP(e) (e)
__device__ __forceinline__ static ::risc0::FpExt ef_pml(uint32_t k, uint32_t dep) {
  uint64_t p = reinterpret_cast<uint64_t>(ef_pm_raw);
  asm("" : "+s"(p) : "v"(dep));
  const uint4 v = ((const __attribute__((address_space(4))) uint4*)p)[k];
  return ::risc0::FpExt(::risc0::Fp::fromRaw(v.x), ::risc0::Fp::fromRaw(v.y), ::risc0::Fp::fromRaw(v.z), ::risc0::Fp::fromRaw(v.w));
}
template <typename A> __device__ __forceinline__ static RkAcc rk_link_kl(const A& base, Fp c, uint32_t k, uint32_t dep) { return rk_link(base, c, ef_pml(k, dep)); }
template <typename A> __device__ __forceinline__ static RkAcc rk_link_nf_kl(const A& base, Fp c, uint32_t k, uint32_t dep) { return rk_link_nf(base, c, ef_pml(k, dep)); }
#else
#define EF_DEP(e) 0u
template <typename A> __device__ __forceinline__ static RkAcc rk_link_kl(const A& base, Fp c, uint32_t k, uint32_t) { return rk_link_k(base, c, k); }
template <typename A> __device__ __forceinline__ static RkAcc rk_link_nf_kl(const A& base, Fp c, uint32_t k, uint32_t) { return rk_link_nf_k(base, c, k); }
#endif
template <typename A> __device__ __forceinline__ static RkAcc rk_link_kl(const A& base, FpExt c, uint32_t k, uint32_t) { return rk_link_k(base, c, k); }
template <typename A> __device__ __forceinline__ static RkAcc rk_link_nf_kl(const A& base, FpExt c, uint32_t k, uint32_t) { return rk_link_nf_k(base, c, k); }
'''
ANCHOR = '// ---- wmma_hip.py: poly_mix chains on the gfx12 matrix units'
FN = re.compile(r'^__device__ (?:EF_NOINLINE |EF_CHAIN )?FpExt (rv32im_v2_\d+|poly_fp|rk_accblk)\(.*\) \{$')
LINK = re.compile(r'(RkAcc|FpExt) (x\d+) = (rk_fin\()?(rk_link(?:_nf)?_k)\((\w+), (\w+), (\d+u)\)(\)?);')


def define(s):
    assert s.count(ANCHOR) == 1
    return s.replace(ANCHOR, HELP + ANCHOR)


def convert(s, w=4, stats=None):
    lines = define(s).split('\n')
    n = nf = 0
    i = 0
    while i < len(lines):
        if not FN.match(lines[i]):
            i += 1
            continue
        end = lines.index('}', i + 1)
        nf += 1
        res = []
        for k in range(i + 1, end):
            line = lines[k]
            # only the scalar form of an EF_W((gfx12 form), scalar form) line
            if line.startswith('  EF_W(('):
                depth, cut = 0, None
                for q in range(line.index('EF_W(') + 5, len(line)):
                    if line[q] == '(':
                        depth += 1
                    elif line[q] == ')':
                        depth -= 1
                        if depth == 0:
                            cut = q + 1
                            break
                head, tail = line[:cut], line[cut:]
            else:
                head, tail = '', line
            m = LINK.search(tail)
            if not m:
                continue
            ty, name, fin, fn, base, c, kk, close = m.groups()
            if len(res) >= w:
                new = '%s %s = %s%sl(%s, %s, %s, EF_DEP(%s))%s;' % (ty, name, fin or '', fn, base, c, kk,
                                                                   res[-w], close)
                tail = tail[:m.start()] + new + tail[m.end():]
                n += 1
            res.append('%s.lo[0]' % name if ty == 'RkAcc' else '%s.elems[0].asRaw()' % name)
            lines[k] = head + tail
        i = end + 1
    msg = 'pmlag_hip: %d functions, %d links with lagged constants (W %d)' % (nf, n, w)
    if stats is not None:
        stats.append(msg)
    else:
        print(msg)
    return '\n'.join(lines)
