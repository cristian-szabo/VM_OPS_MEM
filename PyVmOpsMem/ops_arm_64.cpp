#include "ops_arm_64.h"

#include <arm_neon.h>

void
Ops_Arm_64(pybind11::module &m) {
    m.def("mmla_s8", [](uint64_t steps) {
        const signed char src0[16];
        const signed char src1[16];
        const signed int res[4];

        int8x16_t a = vld1q_s8(src0);
        int8x16_t b = vld1q_s8(src1);
        int32x4_t c = vld1q_s32(res);

        auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(32)
        for (uint64_t k = 0; k < steps; k++) {
            c = vmmlaq_s32(c, a, b);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        vst1q_s32(res, c);

        uint64_t ops = steps * 2 * 8 * 2;

        return pybind11::make_tuple(duration.count(), ops);
    });

    m.def("mmla_f16", [](uint64_t steps) {
        const signed short src0[16];
        const signed short src1[16];
        const signed short res[16];

        bfloat16x8_t a = vld1q_s8(src0);
        bfloat16x8_t b = vld1q_s8(src1);
        float32x4_t c = vld1q_f32(res);

        auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(32)
        for (uint64_t k = 0; k < steps; k++) {
            c = vbfmmlaq_f32(c, a, b);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        float out[4];
        vst1q_s32(out, c);

        uint64_t ops = steps * 2 * 4 * 2;

        return pybind11::make_tuple(duration.count(), ops);
    });
}
