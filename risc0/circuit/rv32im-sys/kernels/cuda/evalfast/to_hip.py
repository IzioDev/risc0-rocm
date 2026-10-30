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
        '__device__ __forceinline__ uint32_t ef_ldl(const void* p, uint32_t w) {\n'
        '  return ((const ef_pu32*)(p))[(size_t)w * RK_STRIDE];\n'
        '}\n'
        '__device__ __forceinline__ void ef_stl(void* p, uint32_t w, uint32_t v) {\n'
        '  ((ef_pu32*)(p))[(size_t)w * RK_STRIDE] = v;\n'
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
        '#else\n'
        '#define EF_LD(p, i) (::risc0::Fp::fromRaw(__ldg(reinterpret_cast<const uint32_t*>(p) + (i))))\n'
        '#endif\n')
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
