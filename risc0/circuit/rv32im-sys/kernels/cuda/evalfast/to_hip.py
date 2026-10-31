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

# The ROCm forms of the CUDA eval_check source: regen.py applies convert() to
# the chain's output after remat_lu.py to write eval_check_ef_hip.cu. Each
# rewrite must apply exactly as often as pinned, so a generator change that
# moves one fails here:
#
#  - EF_SLAB_ASM (PTX ld.local / st.local) and <cuda_runtime.h> only for CUDA,
#    with C++ ef_ldl / ef_stl for every slab access;
#  - device loads and slab accesses through address-space-qualified pointers:
#    through the generic Fp* parameters they compile to FLAT instructions,
#    which count against both vmcnt and lgkmcnt and were each waited on with
#    s_waitcnt vmcnt(0) lgkmcnt(0) (-3% on gfx1100, -1.7% on gfx1201);
#  - uniform addressing: the chain functions get steps and the column / slab
#    pointers as (per-lane) VGPR arguments, so every trace load computed a
#    64-bit VGPR address (index shift + 64-bit add, the base pointers held in
#    VGPRs and spilled). steps and the pointers are the same in every lane:
#    read with readfirstlane, a load is one global_load with an SGPR column
#    base (pointer + c * steps words, scalar ALU) and a 32-bit row offset
#    shared by the column loads of one row (EF_LDC), and the slab accesses use
#    an SGPR address; loads from the constant address space (the trace is
#    read-only for the whole kernel, as remat.py already assumes), which the
#    register allocator may re-issue instead of spilling; the few uniform
#    out/mix array reads through Fp* parameters as scalar loads (EF_LDU).
#    Fewer VGPRs let the scheduler issue the loads well ahead of their uses:
#    -16% on gfx1100, -18% on gfx1201 (ef harness, po2 20);
#  - the lazy reductions' Montgomery factor lo * Fp::M through Fp::mulM
#    (ef_fp.h: two shift-adds on ROCm, where a v_mul_lo_u32 issues at a
#    quarter of the add rate);
#  - the device-to-device symbol copies through hipGetSymbolAddress (the
#    standalone TU is not relocatable, where hipMemcpyToSymbol is unreliable);
#  - the eval_check stream always synchronized, as eval_check_combined.cu does;
#  - the header's first line names this ROCm build (regen.py appends its notes).
#
# usage: to_hip.py <in.cu> <out.cu>
import re
import sys


def convert(s):
    def sub(old, new, count=1):
        nonlocal s
        n = s.count(old)
        if n != count:
            sys.exit('to_hip: expected %d of %r, found %d' % (count, old[:80], n))
        s = s.replace(old, new)

    sub('// rv32im eval_check for CUDA (NVIDIA only; ROCm builds eval_check_combined.cu):\n',
        '// rv32im eval_check, the ROCm form of the CUDA one (eval_check_ef.cu):\n')
    sub('#define EF_SLAB_ASM 1\n', '#if !defined(__HIPCC__)\n#define EF_SLAB_ASM 1\n#endif\n')
    sub('#else\n'
        '#define RK_LD1(p, w) (static_cast<const ::risc0::Fp>(::risc0::Fp::fromRaw('
        'reinterpret_cast<const uint32_t*>(p)[(size_t)(w) * RK_STRIDE])))\n'
        '#define RK_ST1(p, w, v) (reinterpret_cast<uint32_t*>(p)[(size_t)(w) * RK_STRIDE] = '
        '::risc0::Fp(v).asRaw())\n',
        '#else\n'
        '#if defined(__HIP_DEVICE_COMPILE__)\n'
        '// The slab is the kernel\'s private array (scratch_* rather than FLAT accesses).\n'
        'typedef __attribute__((address_space(5))) uint32_t ef_pu32;\n'
        '// Its address is the same in every lane: an SGPR address (ef_uptr).\n'
        '__device__ __forceinline__ uint32_t ef_ldl(const void* p, uint32_t w) {\n'
        '  return ((const ef_pu32*)(ef_uptr(p)))[(size_t)w * RK_STRIDE];\n'
        '}\n'
        '__device__ __forceinline__ void ef_stl(void* p, uint32_t w, uint32_t v) {\n'
        '  ((ef_pu32*)(ef_uptr(p)))[(size_t)w * RK_STRIDE] = v;\n'
        '}\n'
        '#else\n'
        '__device__ __forceinline__ uint32_t ef_ldl(const void* p, uint32_t w) {\n'
        '  return reinterpret_cast<const uint32_t*>(p)[(size_t)w * RK_STRIDE];\n'
        '}\n'
        '__device__ __forceinline__ void ef_stl(void* p, uint32_t w, uint32_t v) {\n'
        '  reinterpret_cast<uint32_t*>(p)[(size_t)w * RK_STRIDE] = v;\n'
        '}\n'
        '#endif\n'
        '#define RK_LD1(p, w) (static_cast<const ::risc0::Fp>(::risc0::Fp::fromRaw(ef_ldl((p), (w)))))\n'
        '#define RK_ST1(p, w, v) ef_stl((p), (w), ::risc0::Fp(v).asRaw())\n')
    sub('#define EF_LD(p, i) (::risc0::Fp::fromRaw(__ldg(reinterpret_cast<const uint32_t*>(p) + (i))))\n',
        '#if defined(__HIP_DEVICE_COMPILE__)\n'
        '// Trace columns: global loads (the generic Fp* parameters would make them FLAT).\n'
        'typedef const __attribute__((address_space(1))) uint32_t ef_gu32;\n'
        '#define EF_LD(p, i) (::risc0::Fp::fromRaw(((ef_gu32*)(reinterpret_cast<const uint32_t*>(p)))[(i)]))\n'
        '// Uniform addressing (to_hip.py): steps and the trace / slab pointers are the same in every lane.\n'
        '#define EF_UNI32(x) ((uint32_t)__builtin_amdgcn_readfirstlane((int)(x)))\n'
        '__device__ __forceinline__ static void* ef_uptr(const void* p) {\n'
        '  const uint64_t v = reinterpret_cast<uint64_t>(p);\n'
        '  return reinterpret_cast<void*>((uint64_t(EF_UNI32(v >> 32)) << 32) | EF_UNI32(v));\n'
        '}\n'
        '// p + w words, the uniform base of an EF_LDC column (pointer arithmetic: an SGPR base and a\n'
        '// 32-bit VGPR offset make a global_load saddr; integer arithmetic made 64-bit VGPR addresses).\n'
        '__device__ __forceinline__ static const char* ef_ubase(const void* p, uint32_t w) {\n'
        '  return reinterpret_cast<const char*>(ef_uptr(p)) + uint64_t(w) * 4u;\n'
        '}\n'
        'typedef const __attribute__((address_space(4))) uint32_t ef_cu32;\n'
        '// Column c of p, row (cycle - k) mod steps (EF_LDCR: the raw word, see hoist_hip.py).\n'
        '#define EF_LDCR(p, c, k) (*(ef_cu32*)(reinterpret_cast<uint64_t>(ef_ubase((p), (c) * rk_s) + (((rk_c - (k)) & rk_m) << 2))))\n'
        '#define EF_LDC(p, c, k) (::risc0::Fp::fromRaw(EF_LDCR(p, c, k)))\n'
        '#define EF_LDU(p, i) (::risc0::Fp::fromRaw(((ef_cu32*)(reinterpret_cast<uint64_t>(ef_ubase((p), 0u))))[(i)]))\n'
        '#else\n'
        '#define EF_LD(p, i) (::risc0::Fp::fromRaw(__ldg(reinterpret_cast<const uint32_t*>(p) + (i))))\n'
        '#define EF_UNI32(x) (x)\n'
        '#define EF_LDC(p, c, k) EF_LD(p, (uint32_t)((c) * rk_s + ((rk_c - (k)) & rk_m)))\n'
        '#define EF_LDCR(p, c, k) (EF_LDC(p, c, k).asRaw())\n'
        '#define EF_LDU(p, i) ((p)[(i)])\n'
        '#endif\n')
    sub('const uint32_t rk_s = (uint32_t)steps, rk_m = rk_s - 1u, rk_c = (uint32_t)cycle;',
        'const uint32_t rk_s = EF_UNI32((uint32_t)steps), rk_m = rk_s - 1u, rk_c = (uint32_t)cycle;', 22)

    def resub(pat, new, count):
        nonlocal s
        s, n = re.subn(pat, new, s)
        if n != count:
            sys.exit('to_hip: expected %d of %r, found %d' % (count, pat, n))
    resub(r'EF_LD\((\w+|args\[\d\]), \(uint32_t\)\((\d+)u \* rk_s \+ \(\(rk_c - (\d+)u\) & rk_m\)\)\)',
          r'EF_LDC(\1, \2u, \3u)', 5973)
    resub(r'\b(arg\d+)\[(\d+)\]', r'EF_LDU(\1, \2)', 86)
    resub(r'\b((?:s\.|w\.)?lo(?:\[j\])?) \* (::risc0::)?Fp::M\b', r'\2Fp::mulM(\1)', 4)
    sub('#include <cuda_runtime.h>\n', '#if !defined(__HIPCC__)\n#include <cuda_runtime.h>\n#endif\n')
    pat = re.compile(r'CUDA_OK\(cudaMemcpyToSymbolAsync\((\w+), ([^,]+), ([^,]+), 0, '
                     r'cudaMemcpyDeviceToDevice, stream\)\);')
    n = len(pat.findall(s))
    if n != 3:
        sys.exit('to_hip: expected 3 symbol copies, found %d' % n)
    s = pat.sub(lambda m: 'EF_COPY_SYM(%s, %s, %s);' % (m.group(1), m.group(2), m.group(3)), s)
    sub('static cudaStream_t getEvalCheckStream() {',
        '#ifdef __HIPCC__\n'
        '// hipMemcpyToSymbol is unreliable without -fgpu-rdc: copy to the symbol\'s address.\n'
        '#define EF_COPY_SYM(sym, src, n)                                                  \\\n'
        '  do {                                                                            \\\n'
        '    void* p_ = nullptr;                                                           \\\n'
        '    CUDA_OK(hipGetSymbolAddress(&p_, HIP_SYMBOL(sym)));                           \\\n'
        '    CUDA_OK(hipMemcpyAsync(p_, (src), (n), hipMemcpyDeviceToDevice, stream));     \\\n'
        '  } while (0)\n'
        '#else\n'
        '#define EF_COPY_SYM(sym, src, n)                                                  \\\n'
        '  CUDA_OK(cudaMemcpyToSymbolAsync(sym, (src), (n), 0, cudaMemcpyDeviceToDevice, stream))\n'
        '#endif\n'
        'static cudaStream_t getEvalCheckStream() {')
    sub('''      static int s_cc_major = -1;
      if (s_cc_major < 0) cudaDeviceGetAttribute(&s_cc_major, cudaDevAttrComputeCapabilityMajor, getCachedDevice());
      needs_sync = s_cc_major >= 12;''', '''#ifdef __HIPCC__
      // As in eval_check_combined.cu: overlapping eval_check with the next segment is unsafe
      // on RDNA3 and, on a 16 GB card, slower for the memory it holds.
      needs_sync = true;
#else
      static int s_cc_major = -1;
      if (s_cc_major < 0) cudaDeviceGetAttribute(&s_cc_major, cudaDevAttrComputeCapabilityMajor, getCachedDevice());
      needs_sync = s_cc_major >= 12;
#endif''')
    return s


if __name__ == '__main__':
    with open(sys.argv[1], newline='') as f:
        text = f.read()
    with open(sys.argv[2], 'w', newline='') as f:
        f.write(convert(text))
