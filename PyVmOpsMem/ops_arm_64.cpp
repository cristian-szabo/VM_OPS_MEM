#include "ops_arm_64.h"

#include <arm_neon.h>

void
Ops_Arm_64(pybind11::module &m) {
    m.def("mmla_s8", [](uint64_t steps) {
        const signed char src0[16] = {};
        const signed char src1[16] = {};
        signed int res[4] = {};

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

        return pybind11::make_tuple(duration.count(), ops, res);
    });

    m.def("mmla_bf16", [](uint64_t steps) {
        const __bf16 src0[8] = {};
        const __bf16 src1[8] = {};
        float res[4] = {};

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

        return pybind11::make_tuple(duration.count(), ops, res);
    });

    m.def("dot_bf16", [](uint64_t steps) {
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

        return pybind11::make_tuple(duration.count(), ops, res);
    });

    m.def("dot_s8", [](uint64_t steps) {
        const signed char src0[16] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
        const signed char src1[16] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
        int res[4] = {0, 0};

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

        return pybind11::make_tuple(duration.count(), ops, res);
    });

#if defined(SIMDE_ARM_NEON_A32V7_NATIVE) && defined(SIMDE_ARCH_ARM_FMA)
    m.def("fma_f32", [](uint64_t steps) {
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

        return pybind11::make_tuple(duration.count(), ops, res);
    });
#endif

#if defined(SIMDE_ARM_NEON_A32V8_NATIVE) && defined(SIMDE_ARCH_ARM_FMA) &&                         \
    defined(SIMDE_ARM_NEON_FP16)
    m.def("fma_f16", [](uint64_t steps) {
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

        return pybind11::make_tuple(duration.count(), ops, res);
    });
#endif
}
