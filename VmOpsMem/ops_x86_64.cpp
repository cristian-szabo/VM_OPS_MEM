#include <chrono>
#include <cstring>
#include <thread>

#include <cpuid.h>
#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "vmopsmem_export.h"

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA  18

#if defined(__AMX_TILE__) && defined(__AMX_INT8__)
#define AMX_S8_S32_SUPPORT 1
#else
#define AMX_S8_S32_SUPPORT 0
#endif

#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
#define AMX_BF16_F32_SUPPORT 1
#else
#define AMX_BF16_F32_SUPPORT 0
#endif

#if defined(__AVX512VNNI__)
#define VNN_S8_S32_SUPPORT  1
#define VNN_F16_F32_SUPPORT 1
#else
#define VNN_S8_S32_SUPPORT  0
#define VNN_F16_F32_SUPPORT 0
#endif

struct tile_config_t {
    uint8_t paletteId;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
};

struct Result {
    int64_t time;
    uint64_t ops;
    char output[1024];
};

struct CpuResult {
    int64_t time;
    uint64_t cycles;
};

std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

#ifdef __cplusplus
extern "C" {
#endif

extern VMOPSMEM_EXPORT void set_thread_affinity(int coreId);

VMOPSMEM_EXPORT bool
init() {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        return false;
    } else {
        return true;
    }

    start_time = std::chrono::high_resolution_clock::now();

    return true;
}

struct cpuid_info {
    unsigned eax;
    unsigned ebx;
    unsigned ecx;
    unsigned edx;
};

static cpuid_info
cpuid(unsigned leaf, unsigned subleaf) {
    cpuid_info info{};
    __get_cpuid_count(leaf, subleaf, &info.eax, &info.ebx, &info.ecx, &info.edx);
    return info;
}

unsigned long
read_bits(const unsigned int val, const char from, const char to) {
    unsigned long mask = (1 << (to + 1)) - 1;
    if (to == 31)
        return val >> from;

    return (val & mask) >> from;
}

struct logical_core_t {
    unsigned OrdIndexOAMsk;
    unsigned pkg_ID_APIC;
    unsigned Core_ID_APIC;
    unsigned SMT_ID_APIC;
};

VMOPSMEM_EXPORT void
logical_cores(logical_core_t *logicalCores, unsigned OSProcessorCount) {
    unsigned CoreSelectMask;
    unsigned SMTMaskWidth;
    unsigned PkgSelectMask;
    unsigned PkgSelectMaskShift;
    unsigned SMTSelectMask;
    bool hasLeafB;

    cpuid_info info = cpuid(0, 0);
    unsigned maxCPUID = info.eax;

    // cpuid leaf B detection
    if (maxCPUID >= 0xB) {
        info = cpuid(0xB, 0);
        hasLeafB = (info.ebx != 0);
    }

    info = cpuid(1, 0);

    // Use HWMT feature flag CPUID.01:EDX[28]
    // #1, Processors that support CPUID leaf 0BH
    if (read_bits(info.edx, 28, 28) && hasLeafB) {
        int wasCoreReported = 0;
        int wasThreadReported = 0;
        int subLeaf = 0, levelType, levelShift;
        unsigned long coreplusSMT_Mask = 0;

        do {   // we already tested CPUID leaf 0BH contain valid sub-leaves
            info = cpuid(0xB, subLeaf);
            // if EBX ==0 then this subleaf is not valid, we can exit the loop
            if (info.ebx == 0) {
                break;
            }
            levelType = read_bits(info.ecx, 8, 15);
            levelShift = read_bits(info.eax, 0, 4);
            switch (levelType) {
            case 1:   // level type is SMT, so levelShift is the SMT_Mask_Width
                SMTSelectMask = ~((-1) << levelShift);
                SMTMaskWidth = levelShift;
                wasThreadReported = 1;
                break;
            case 2:   // level type is Core, so levelShift is the CorePlsuSMT_Mask_Width
                coreplusSMT_Mask = ~((-1) << levelShift);
                PkgSelectMaskShift = levelShift;
                PkgSelectMask = (-1) ^ coreplusSMT_Mask;
                wasCoreReported = 1;
                break;
            default:
                // handle in the future
                break;
            }
            subLeaf++;
        } while (1);

        if (wasThreadReported && wasCoreReported) {
            CoreSelectMask = coreplusSMT_Mask ^ SMTSelectMask;
        } else if (!wasCoreReported && wasThreadReported) {
            CoreSelectMask = 0;
            PkgSelectMaskShift = SMTMaskWidth;
            PkgSelectMask = (-1) ^ SMTSelectMask;
        }
    }

    for (unsigned i = 0; i < OSProcessorCount; ++i) {
        set_thread_affinity(i);
        std::this_thread::yield();

        logical_core_t &core = logicalCores[i];

        cpuid_info info{};
        unsigned APICID{};

        if (hasLeafB) {
            info = cpuid(0xB, 0);
            APICID = info.edx;
        } else {
            info = cpuid(1, 0);
            APICID = read_bits(info.ebx, 24, 31);
        }

        core.OrdIndexOAMsk = i;
        core.pkg_ID_APIC = ((APICID & PkgSelectMask) >> PkgSelectMaskShift);
        core.Core_ID_APIC = ((APICID & CoreSelectMask) >> SMTMaskWidth);
        core.SMT_ID_APIC = (APICID & SMTSelectMask);
    }
}

VMOPSMEM_EXPORT CpuResult
cpu_time() {
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

    CpuResult r;
    r.cycles = ((uint64_t) hi << 32) | lo;
    r.time = elapsed_time.count();
    return r;
}

/* MATRIX MULTIPLY ACCUMULATE */
VMOPSMEM_EXPORT int32_t
amx_s8_s32_support() {
#if AMX_S8_S32_SUPPORT
    return 1;
#endif
    return 0;
}

VMOPSMEM_EXPORT int32_t
amx_bf16_f32_support() {
#if AMX_BF16_F32_SUPPORT
    return 1;
#endif
    return 0;
}

/* VECTOR NEURAL NETWORK */
VMOPSMEM_EXPORT int32_t
vnn_s8_s32_support() {
#if VNN_S8_S32_SUPPORT
    return 1;
#endif
    return 0;
}

VMOPSMEM_EXPORT int32_t
vnn_f16_f32_support() {
#if VNN_F16_F32_SUPPORT
    return 1;
#endif
    return 0;
}

#if AMX_S8_S32_SUPPORT
VMOPSMEM_EXPORT Result
amx_s8_s32(uint64_t steps) {
    tile_config_t tile_info{};
    tile_info.paletteId = 1;
    tile_info.cols[0] = 64;
    tile_info.rows[0] = 16;
    tile_info.cols[1] = 64;
    tile_info.rows[1] = 16;
    tile_info.cols[2] = 64;
    tile_info.rows[2] = 16;
    _tile_loadconfig(&tile_info);

    int8_t src1[1024] = {};
    int8_t src2[1024] = {};
    int32_t res[256] = {};

    _tile_loadd(1, src1, 64);
    _tile_loadd(2, src2, 64);
    _tile_loadd(0, res, 64);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(16)
    for (uint64_t k = 0; k < steps; k++) {
        _tile_dpbssd(0, 1, 2);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    _tile_stored(0, res, 64);
    _tile_release();

    uint64_t ops_per_output = 64 /* mul */ + 64 /* add */;
    uint64_t ops = steps * ops_per_output * 16 * 16 /* num outputs */;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.output, res, sizeof(res));
    return r;
};
#endif

#if AMX_BF16_F32_SUPPORT
VMOPSMEM_EXPORT Result
amx_bf16_f32(uint64_t steps) {
    tile_config_t tile_info{};
    tile_info.paletteId = 1;
    tile_info.cols[0] = 64;
    tile_info.rows[0] = 16;
    tile_info.cols[1] = 64;
    tile_info.rows[1] = 16;
    tile_info.cols[2] = 64;
    tile_info.rows[2] = 16;
    _tile_loadconfig(&tile_info);

    __bf16 src1[512] = {};
    __bf16 src2[512] = {};
    float res[256] = {};

    _tile_loadd(1, src1, 64);
    _tile_loadd(2, src2, 64);
    _tile_loadd(0, res, 64);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(1024)
    for (uint64_t k = 0; k < steps; k++) {
        _tile_dpbf16ps(0, 1, 2);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    _tile_stored(0, res, 64);
    _tile_release();

    uint64_t ops_per_output = 32 /* mul */ + 32 /* add */;
    uint64_t ops = steps * ops_per_output * 16 * 16 /* num outputs */;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.output, res, sizeof(res));
    return r;
};
#endif

#if VNN_S8_S32_SUPPORT
VMOPSMEM_EXPORT Result
vnn_s8_s32(uint64_t steps) {
    int src1[16] = {};
    int src2[16] = {};
    int res[16] = {};

    __m512i A, B, C;
    A = _mm512_loadu_si512((__m512i *) &src1);
    B = _mm512_loadu_si512((__m512i *) &src2);
    C = _mm512_loadu_si512((__m512i *) &res);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(1024)
    for (uint64_t k = 0; k < steps; k++) {
        C = _mm512_dpbusd_epi32(C, B, A);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    _mm512_storeu_si512((__m512i *) &res, C);

    uint64_t ops_per_output = 4 /* mul */ + 4 /* add */;
    uint64_t ops_per_inst = ops_per_output * 16 /* groups of 4 */;
    uint64_t ops = steps * ops_per_inst /* num outputs */;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.output, &res, sizeof(res));
    return r;
};
#endif

#if VNN_F16_F32_SUPPORT
VMOPSMEM_EXPORT Result
vnn_f16_f32(uint64_t steps) {
    __fp16 src1[32] = {};
    __fp16 src2[32] = {};
    float res[32] = {};

    __m512i A, B, C;
    A = _mm512_loadu_si512((__m512i *) &src1);
    B = _mm512_loadu_si512((__m512i *) &src2);
    C = _mm512_loadu_si512((__m512i *) &res);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(32)
    for (uint64_t k = 0; k < steps; k++) {
        C = _mm512_dpwssd_epi32(C, B, A);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    _mm512_storeu_si512((__m512i *) &res, C);

    uint64_t ops_per_output = 2 /* mul */ + 2 /* add */;
    uint64_t ops_per_inst = ops_per_output * 16 /* groups of 2 */;
    uint64_t ops = steps * ops_per_inst /* num outputs */;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.output, &res, sizeof(res));
    return r;
};
#endif

#ifdef __cplusplus
}
#endif
