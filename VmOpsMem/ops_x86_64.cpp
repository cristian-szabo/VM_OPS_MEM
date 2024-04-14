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

struct Features {
    union {
        struct {
            unsigned int Stepping : 4 - 0, Model : 8 - 4, Family : 12 - 8, ProcType : 14 - 12,
                Reserved1 : 16 - 14, ExtModel : 20 - 16, ExtFamily : 28 - 20, Reserved2 : 32 - 28;
        } EAX;
        unsigned int Signature;
    };
    struct {
        unsigned int Brand_ID : 8 - 0, CLFSH_Size : 16 - 8, Max_SMT_ID : 24 - 16,
            Init_APIC_ID : 32 - 24;
    } EBX;
    struct {
        unsigned int SSE3 : 1 - 0, /* AMD Family 0Fh		*/
            PCLMULDQ : 2 - 1, DTES64 : 3 - 2, MONITOR : 4 - 3, DS_CPL : 5 - 4, VMX : 6 - 5,
            SMX : 7 - 6, EIST : 8 - 7, TM2 : 9 - 8, SSSE3 : 10 - 9, /* AMD Family 0Fh		*/
            CNXT_ID : 11 - 10, SDBG : 12 - 11, /* IA32_DEBUG_INTERFACE MSR support */
            FMA : 13 - 12, CMPXCHG16 : 14 - 13, xTPR : 15 - 14, PDCM : 16 - 15, Reserved : 17 - 16,
            PCID : 18 - 17, DCA : 19 - 18, SSE41 : 20 - 19, SSE42 : 21 - 20,
            x2APIC : 22 - 21, /* x2APIC capability		*/
            MOVBE : 23 - 22, POPCNT : 24 - 23, TSC_DEADLINE : 25 - 24, AES : 26 - 25,
            XSAVE : 27 - 26, OSXSAVE : 28 - 27, AVX : 29 - 28, F16C : 30 - 29, RDRAND : 31 - 30,
            Hyperv : 32 - 31; /* This bit is set by the Hypervisor */
    } ECX;
    struct { /* Most common x86					*/
        unsigned int FPU : 1 - 0, VME : 2 - 1, DE : 3 - 2, PSE : 4 - 3, TSC : 5 - 4, MSR : 6 - 5,
            PAE : 7 - 6, MCE : 8 - 7, CMPXCHG8 : 9 - 8, APIC : 10 - 9, Reserved1 : 11 - 10,
            SEP : 12 - 11, MTRR : 13 - 12, PGE : 14 - 13, MCA : 15 - 14, CMOV : 16 - 15,
            PAT : 17 - 16, PSE36 : 18 - 17, PSN : 19 - 18, /* Intel Processor Serial Number */
            CLFLUSH : 20 - 19, Reserved2 : 21 - 20, DS_PEBS : 22 - 21, ACPI : 23 - 22,
            MMX : 24 - 23, FXSR : 25 - 24,               /* FXSAVE and FXRSTOR instructions. */
            SSE : 26 - 25, SSE2 : 27 - 26, SS : 28 - 27, /* Intel			*/
            HTT : 29 - 28, TM1 : 30 - 29,                /* Intel			*/
            Reserved3 : 31 - 30, PBE : 32 - 31;          /* Intel			*/
    } EDX;
};

std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

std::vector<LogicalCore> processors;
Features features;

inline unsigned short
FindMaskWidth(unsigned short maxCount) {
    unsigned short maskWidth = 0, count = (maxCount - 1);

    if (BITBSR(count, maskWidth) == 0)
        maskWidth++;

    return maskWidth;
}

static void
QueryFeatures() {
    __asm__ volatile("movq	$0x1,  %%rax	\n\t"
                     "xorq	%%rbx, %%rbx	\n\t"
                     "xorq	%%rcx, %%rcx	\n\t"
                     "xorq	%%rdx, %%rdx	\n\t"
                     "cpuid			\n\t"
                     "mov	%%eax, %0	\n\t"
                     "mov	%%ebx, %1	\n\t"
                     "mov	%%ecx, %2	\n\t"
                     "mov	%%edx, %3"
                     : "=r"(features.EAX), "=r"(features.EBX), "=r"(features.ECX),
                       "=r"(features.EDX)
                     :
                     : "%rax", "%rbx", "%rcx", "%rdx");
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

    if (features.EDX.HTT) {
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

VMOPSMEM_EXPORT void
init() {
    syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);

    QueryFeatures();

    unsigned OSProcessorCount = sysconf(_SC_NPROCESSORS_CONF);

    processors.resize(OSProcessorCount);

    for (unsigned i = 0; i < OSProcessorCount; ++i) {
        set_thread_affinity(i);

        std::this_thread::yield();

        LogicalCore &core = processors[i];
        core.Index = i;

        MapCpuTopology(core);
    }

    start_time = std::chrono::high_resolution_clock::now();
}

VMOPSMEM_EXPORT CpuResult
cpu_time() {
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

    CpuResult r;
    r.Cycles = ((uint64_t) hi << 32) | lo;
    r.Time = elapsed_time.count();
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
    std::memcpy(r.Output, res, sizeof(res));
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
    std::memcpy(r.Output, res, sizeof(res));
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
    std::memcpy(r.Output, &res, sizeof(res));
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
    std::memcpy(r.Output, &res, sizeof(res));
    return r;
};
#endif

#ifdef __cplusplus
}
#endif
