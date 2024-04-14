#include "vm_ops_mem.h"

#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include <arm_neon.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>

#include "vmopsmem_export.h"

/* MATRIX MULTIPLY ACCUMULATE */
#if defined(__ARM_FEATURE_MATMUL_INT8)
#define MMLA_S8_S32_SUPPORT 1
#else
#define MMLA_S8_S32_SUPPORT 0
#endif

#if defined(__ARM_FEATURE_BF16) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#define MMLA_BF16_F32_SUPPORT 1
#else
#define MMLA_BF16_F32_SUPPORT 0
#endif

/* MULTIPLY ACCUMULATE */
#define MLA_F32_F32_SUPPORT 1

#if defined(__ARM_FEATURE_BF16) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#define MLA_BF16_F32_SUPPORT 1
#else
#define MLA_BF16_F32_SUPPORT 0
#endif

#define MLA_S8_S16_SUPPORT 1

/* DOT PRODUCT */
#if defined(__ARM_FEATURE_BF16) && defined(__ARM_FEATURE_DOTPROD)
#define DOT_BF16_F32_SUPPORT 1
#else
#define DOT_BF16_F32_SUPPORT 0
#endif

#if defined(__ARM_FEATURE_DOTPROD)
#define DOT_S8_S32_SUPPORT 1
#else
#define DOT_S8_S32_SUPPORT 0
#endif

/* FUSED MULTIPLY ACCUMULATE */
#if defined(__ARM_FEATURE_FMA)
#define FMA_F32_F32_SUPPORT 1
#else
#define FMA_F32_F32_SUPPORT 0
#endif

#if defined(__ARM_FEATURE_FMA) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define FMA_F16_F16_SUPPORT 1
#else
#define FMA_F16_F16_SUPPORT 0
#endif

#if defined(__ARM_FEATURE_FMA) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) &&                 \
    defined(__ARM_FEATURE_FP16_FML)
#define FMA_F16_F32_SUPPORT 1
#else
#define FMA_F16_F32_SUPPORT 0
#endif

union MPIDR {
    unsigned long long value;
    struct {
        unsigned long long Aff0 : 8 - 0;   /*	Thread ID		*/
        unsigned long long Aff1 : 16 - 8;  /*	Core ID: CPUID[12-8] L1 */
        unsigned long long Aff2 : 24 - 16; /*	Cluster ID - Level2	*/
        unsigned long long MT : 25 - 24;   /*	Multithreading		*/
        unsigned long long UNK : 30 - 25;
        unsigned long long U : 31 - 30; /*	0=Uniprocessor		*/
        unsigned long long RES1 : 32 - 31;
        unsigned long long Aff3 : 40 - 32; /*	Cluster ID - Level3	*/
        unsigned long long RES0 : 64 - 40;
    };
};

static void
MapCpuTopology(LogicalCore &core) {
    volatile MPIDR mpid;
    __asm__ volatile("mrs	%[mpid] ,	mpidr_el1"
                     "\n\t"
                     "isb"
                     : [mpid] "=r"(mpid)
                     :
                     : "memory");
    if (mpid.MT) {
        core.PackageID = mpid.Aff2;
        core.CoreID = mpid.Aff1;
        core.ThreadID = mpid.Aff0;
    } else {
        core.PackageID = mpid.Aff2;
        core.CoreID = mpid.Aff0;
    }
}

std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

std::vector<LogicalCore> processors;

#ifdef __cplusplus
extern "C" {
#endif

extern VMOPSMEM_EXPORT void set_thread_affinity(int coreId);

VMOPSMEM_EXPORT void
init() {
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
    uint64_t cycles;
    asm("mrs %0, cntvct_el0" : "=r"(cycles));

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

    CpuResult r;
    r.Cycles = cycles;
    r.Time = elapsed_time.count();
    return r;
}

/* MATRIX MULTIPLY ACCUMULATE */
VMOPSMEM_EXPORT int32_t
mmla_s8_s32_support() {
#if MMLA_S8_S32_SUPPORT
    if (getauxval(AT_HWCAP2) & HWCAP2_I8MM) {
        return 1;
    }
#endif
    return 0;
}

VMOPSMEM_EXPORT int32_t
mmla_bf16_f32_support() {
#if MMLA_BF16_F32_SUPPORT
    if (getauxval(AT_HWCAP2) & HWCAP2_BF16) {
        return 1;
    }
#endif
    return 0;
}

/* MULTIPLY ACCUMULATE */
VMOPSMEM_EXPORT int32_t
mla_f32_f32_support() {
#if MLA_F32_F32_SUPPORT
    return 1;
#endif
    return 0;
}

VMOPSMEM_EXPORT int32_t
mla_bf16_f32_support() {
#if MLA_BF16_F32_SUPPORT
    if (getauxval(AT_HWCAP2) & HWCAP2_BF16) {
        return 1;
    }
#endif
    return 0;
}

VMOPSMEM_EXPORT int32_t
mla_s8_s16_support() {
#if MLA_S8_S16_SUPPORT
    return 1;
#endif
    return 0;
}

/* DOT PRODUCT */
VMOPSMEM_EXPORT int32_t
dot_bf16_f32_support() {
#if DOT_BF16_F32_SUPPORT
    if ((getauxval(AT_HWCAP) & HWCAP_ASIMDDP) && (getauxval(AT_HWCAP2) & HWCAP2_BF16)) {
        return 1;
    }
#endif
    return 0;
}

VMOPSMEM_EXPORT int32_t
dot_s8_s32_support() {
#if DOT_S8_S32_SUPPORT
    if (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) {
        return 1;
    }
#endif
    return 0;
}

/* FUSED MULTIPLY ACCUMULATE */
VMOPSMEM_EXPORT int32_t
fma_f32_f32_support() {
#if FMA_F32_F32_SUPPORT
    return 1;
#endif
    return 0;
}

VMOPSMEM_EXPORT int32_t
fma_f16_f16_support() {
#if FMA_F16_F16_SUPPORT
    return 1;
#endif
    return 0;
}

VMOPSMEM_EXPORT int32_t
fma_f16_f32_support() {
#if FMA_F16_F32_SUPPORT
    return 1;
#endif
    return 0;
}

#if MMLA_S8_S32_SUPPORT
VMOPSMEM_EXPORT Result
mmla_s8_s32(uint64_t steps) {
    const signed char src0[16] = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8};
    const signed char src1[16] = {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8};
    signed int res[4] = {0, 0, 0, 0};

    int8x16_t a = vld1q_s8(src0);
    int8x16_t b = vld1q_s8(src1);
    int32x4_t c = vld1q_s32(res);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(1024)
    for (uint64_t k = 0; k < steps; k++) {
        c = vmmlaq_s32(c, a, b);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    vst1q_s32(res, c);

    uint64_t ops_per_output = 4 /* mul */ + 4 /* add */;
    uint64_t ops = steps * ops_per_output * 4 /* num outputs */;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.Output, res, sizeof(res));
    return r;
}
#endif

#if MMLA_BF16_F32_SUPPORT
VMOPSMEM_EXPORT Result
mmla_bf16_f32(uint64_t steps) {
    const __bf16 src0[8] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    const __bf16 src1[8] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    float res[4] = {0.0, 0.0, 0.0, 0.0};

    bfloat16x8_t a = vld1q_bf16(src0);
    bfloat16x8_t b = vld1q_bf16(src1);
    float32x4_t c = vld1q_f32(res);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(1024)
    for (uint64_t k = 0; k < steps; k++) {
        c = vbfmmlaq_f32(c, a, b);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    vst1q_f32(res, c);

    uint64_t ops_per_output = 2 /* mul */ + 2 /* add */;
    uint64_t ops = steps * ops_per_output * 4 /* num outputs */;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.Output, res, sizeof(res));
    return r;
}
#endif

#if MLA_F32_F32_SUPPORT
VMOPSMEM_EXPORT Result
mla_f32_f32(uint64_t steps) {
    const float src0[4] = {1.0, 2.0, 3.0, 4.0};
    const float src1[4] = {1.0, 2.0, 3.0, 4.0};
    float res[4] = {0.0, 0.0, 0.0, 0.0};

    float32x4_t a = vld1q_f32(src0);
    float32x4_t b = vld1q_f32(src1);
    float32x4_t c = vld1q_f32(res);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(1024)
    for (uint64_t k = 0; k < steps; k++) {
        c = vmlaq_f32(c, a, b);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    vst1q_f32(&res[0], c);

    uint64_t ops_per_output = 1 /* mul */ + 1 /* add */;
    uint64_t ops = steps * ops_per_output * 4 /* num outputs */;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.Output, res, sizeof(res));
    return r;
}
#endif

#if MLA_BF16_F32_SUPPORT
VMOPSMEM_EXPORT Result
mla_bf16_f32(uint64_t steps) {
    const __bf16 src0[8] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    const __bf16 src1[8] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    float res[4] = {0.0, 0.0, 0.0, 0.0};

    bfloat16x8_t a = vld1q_bf16(src0);
    bfloat16x8_t b = vld1q_bf16(src1);
    float32x4_t c = vld1q_f32(res);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(1024)
    for (uint64_t k = 0; k < steps; k++) {
        c = vbfmlalbq_f32(c, a, b);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    vst1q_f32(res, c);

    uint64_t ops_per_output = 2 /* mul */ + 2 /* add */;
    uint64_t ops = steps * ops_per_output * 4 /* num outputs */;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.Output, res, sizeof(res));
    return r;
}
#endif

#if MLA_S8_S16_SUPPORT
VMOPSMEM_EXPORT Result
mla_s8_s16(uint64_t steps) {
    const signed char src0[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const signed char src1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    signed short res[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    int8x8_t a = vld1_s8(src0);
    int8x8_t b = vld1_s8(src1);
    int16x8_t c = vld1q_s16(res);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(1024)
    for (uint64_t k = 0; k < steps; k++) {
        c = vmlal_s8(c, a, b);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    vst1q_s16(res, c);

    uint64_t ops_per_output = 4 /* mul */ + 4 /* add */;
    uint64_t ops = steps * ops_per_output * 4 /* num outputs */;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.Output, res, sizeof(res));
    return r;
}
#endif

#if DOT_BF16_F32_SUPPORT
VMOPSMEM_EXPORT Result
dot_bf16_f32(uint64_t steps) {
    const __bf16 src0[8] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    const __bf16 src1[8] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    float res[4] = {0.0, 0.0, 0.0, 0.0};

    bfloat16x8_t a = vld1q_bf16(src0);
    bfloat16x8_t b = vld1q_bf16(src1);
    float32x4_t c = vld1q_f32(res);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(1024)
    for (uint64_t k = 0; k < steps; k++) {
        c = vbfdotq_f32(c, a, b);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    vst1q_f32(res, c);

    uint64_t ops_per_output = 2 /* mul */ + 2 /* add */;
    uint64_t ops = steps * ops_per_output * 4 /* num outputs */;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.Output, res, sizeof(res));
    return r;
}
#endif

#if DOT_S8_S32_SUPPORT
VMOPSMEM_EXPORT Result
dot_s8_s32(uint64_t steps) {
    const signed char src0[16] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    const signed char src1[16] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    signed int res[4] = {0, 0, 0, 0};

    int8x16_t a = vld1q_s8(src0);
    int8x16_t b = vld1q_s8(src1);
    int32x4_t c = vld1q_s32(res);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(1024)
    for (uint64_t k = 0; k < steps; k++) {
        c = vdotq_s32(c, a, b);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    vst1q_s32(res, c);

    uint64_t ops_per_output = 4 /* mul */ + 4 /* add */;
    uint64_t ops = steps * ops_per_output * 4 /* num outputs */;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.Output, res, sizeof(res));
    return r;
}
#endif

#if FMA_F32_F32_SUPPORT
VMOPSMEM_EXPORT Result
fma_f32_f32(uint64_t steps) {
    const float src0[4] = {1.0, 2.0, 3.0, 4.0};
    const float src1[4] = {1.0, 2.0, 3.0, 4.0};
    float res[4] = {0.0, 0.0, 0.0, 0.0};

    float32x4_t a = vld1q_f32(src0);
    float32x4_t b = vld1q_f32(src1);
    float32x4_t c = vld1q_f32(res);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(1024)
    for (uint64_t k = 0; k < steps; k++) {
        c = vfmaq_f32(c, a, b);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    vst1q_f32(res, c);

    uint64_t ops_per_output = 1 /* mul */ + 1 /* add */;
    uint64_t ops = steps * ops_per_output * 4 /* num outputs */;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.Output, res, sizeof(res));
    return r;
}
#endif

#if FMA_F16_F32_SUPPORT
VMOPSMEM_EXPORT Result
fma_f16_f32(uint64_t steps) {
    const __fp16 src0[8] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    const __fp16 src1[8] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    float res[8] = {0.0, 0.0, 0.0, 0.0};

    float16x8_t a = vld1q_f16(src0);
    float16x8_t b = vld1q_f16(src1);
    float32x4_t c = vld1q_f32(res);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(1024)
    for (uint64_t k = 0; k < steps; k++) {
        c = vfmlalq_low_f16(c, a, b);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    vst1q_f32(res, c);

    uint64_t ops_per_output = 2 /* mul */ + 2 /* add */;
    uint64_t ops = steps * ops_per_output * 4 /* num outputs */;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.Output, res, sizeof(res));
    return r;
}
#endif

#if FMA_F16_F16_SUPPORT
VMOPSMEM_EXPORT Result
fma_f16_f16(uint64_t steps) {
    const __fp16 src0[8] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    const __fp16 src1[8] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    __fp16 res[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    float16x8_t a = vld1q_f16(src0);
    float16x8_t b = vld1q_f16(src1);
    float16x8_t c = vld1q_f16(res);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(1024)
    for (uint64_t k = 0; k < steps; k++) {
        c = vfmaq_f16(c, a, b);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    vst1q_f16(res, c);

    uint64_t ops_per_output = 1 /* mul */ + 1 /* add */;
    uint64_t ops = steps * ops_per_output * 8 /* num outputs */;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.Output, res, sizeof(res));
    return r;
}
#endif

#ifdef __cplusplus
}
#endif
