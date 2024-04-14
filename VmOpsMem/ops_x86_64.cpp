#include "vm_ops_mem.h"

#include <chrono>
#include <cstring>
#include <thread>

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

#define BITBSR(_base, _index)                                                                      \
    ({                                                                                             \
        volatile unsigned char _ret;                                                               \
                                                                                                   \
        __asm__ volatile("bsr	%[base], %[index]"                                                   \
                         "\n\t"                                                                    \
                         "setz	%[ret]"                                                             \
                         : [ret] "+m"(_ret), [index] "=r"(_index)                                  \
                         : [base] "rm"(_base)                                                      \
                         : "cc", "memory");                                                        \
        _ret;                                                                                      \
    })

struct tile_config_t {
    uint8_t paletteId;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
};

std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

static std::vector<LogicalCore> processors;

inline unsigned short
FindMaskWidth(unsigned short maxCount) {
    unsigned short maskWidth = 0, count = (maxCount - 1);

    if (BITBSR(count, maskWidth) == 0)
        maskWidth++;

    return maskWidth;
}

static void
MapCpuTopology(LogicalCore &core) {
    unsigned short SMT_Mask_Width;
    unsigned short CORE_Mask_Width;
    unsigned short SMT_Select_Mask;
    unsigned short CORE_Select_Mask;
    unsigned short PKG_Select_Mask;

    struct {
        unsigned int Brand_ID : 8 - 0;
        unsigned int CLFSH_Size : 16 - 8;
        unsigned int Max_SMT_ID : 24 - 16;
        unsigned int Init_APIC_ID : 32 - 24;
    } leaf1_ebx;

    struct {
        unsigned int Type : 5 - 0;
        unsigned int Level : 8 - 5;
        unsigned int Init : 9 - 8;
        unsigned int Assoc : 10 - 9;
        unsigned int Unused : 14 - 10;
        unsigned int Cache_SMT_ID : 26 - 14;
        unsigned int Max_Core_ID : 32 - 26;
    } leaf4_eax;

    __asm__ volatile("movq	$0x1,  %%rax	\n\t"
                     "xorq	%%rbx, %%rbx	\n\t"
                     "xorq	%%rcx, %%rcx	\n\t"
                     "xorq	%%rdx, %%rdx	\n\t"
                     "cpuid			\n\t"
                     "mov	%%ebx, %0"
                     : "=r"(leaf1_ebx)
                     :
                     : "%rax", "%rbx", "%rcx", "%rdx");

    if (Features.Std.EDX.HTT) {
        SMT_Mask_Width = leaf1_ebx.Max_SMT_ID;

        __asm__ volatile("movq	$0x4,  %%rax	\n\t"
                         "xorq	%%rbx, %%rbx	\n\t"
                         "xorq	%%rcx, %%rcx	\n\t"
                         "xorq	%%rdx, %%rdx	\n\t"
                         "cpuid			\n\t"
                         "mov	%%eax, %0"
                         : "=r"(leaf4_eax)
                         :
                         : "%rax", "%rbx", "%rcx", "%rdx");

        CORE_Mask_Width = leaf4_eax.Max_Core_ID + 1;
    } else {
        SMT_Mask_Width = 0;
        CORE_Mask_Width = 1;
    }

    if (CORE_Mask_Width != 0) {
        SMT_Mask_Width = FindMaskWidth(SMT_Mask_Width) / CORE_Mask_Width;
    }

    SMT_Select_Mask = ~((-1) << SMT_Mask_Width);

    CORE_Select_Mask = (~((-1) << (CORE_Mask_Width + SMT_Mask_Width))) ^ SMT_Select_Mask;

    PKG_Select_Mask = (-1) << (CORE_Mask_Width + SMT_Mask_Width);

    core.ThreadID = leaf1_ebx.Init_APIC_ID & SMT_Select_Mask;

    core.CoreID = (leaf1_ebx.Init_APIC_ID & CORE_Select_Mask) >> SMT_Mask_Width;

    core.PackageID =
        (leaf1_ebx.Init_APIC_ID & PKG_Select_Mask) >> (CORE_Mask_Width + SMT_Mask_Width);
}

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

    for (unsigned i = 0; i < OSProcessorCount; ++i) {
        set_thread_affinity(i);
        std::this_thread::yield();

        LogicalCore &core = processors[i];
        core.Index = i;

        MapCpuTopology(core);
    }

    start_time = std::chrono::high_resolution_clock::now();

    return true;
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
