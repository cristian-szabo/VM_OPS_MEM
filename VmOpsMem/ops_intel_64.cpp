#include <chrono>
#include <cstring>

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

#define AMX_S8_S32_SUPPORT 1

typedef struct __tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tilecfg;

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

VMOPSMEM_EXPORT Result
amx_s8_s32(uint64_t steps) {
    __tilecfg tile_info{};
    tile_info.palette_id = 1;
    tile_info.start_row = 0;
    tile_info.colsb[0] = 16;
    tile_info.rows[0] = 16;
    tile_info.colsb[1] = 64;
    tile_info.rows[1] = 16;
    tile_info.colsb[2] = 64;
    tile_info.rows[2] = 16;
    tile_info.colsb[3] = 64;
    tile_info.rows[3] = 16;
    _tile_loadconfig(&tile_info);

    int8_t src1[1024] = {};
    int8_t src2[1024] = {};
    int32_t res[256] = {};

    for (int i = 0; i < 1024; ++i) {
        src1[i] = i;
        src1[i] = i + 1;
        res[i / 4] = 0;
    }

    _tile_loadd(2, src1, 64);
    _tile_loadd(3, src2, 64);
    _tile_loadd(1, res, 64);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(16)
    for (uint64_t k = 0; k < steps; k++) {
        _tile_dpbssd(1, 2, 3);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    _tile_stored(1, res, 64);
    _tile_release();

    uint64_t ops_per_output = 64 /* mul */ + 64 /* add */;
    uint64_t ops = steps * ops_per_output * 16 * 16 /* num outputs */;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.output, res, sizeof(res));
    return r;
};

VMOPSMEM_EXPORT Result
amx_bf16_f32(uint64_t steps) {
    __tilecfg tile_info{};
    tile_info.palette_id = 1;
    tile_info.start_row = 0;
    tile_info.colsb[0] = 16;
    tile_info.rows[0] = 16;
    tile_info.colsb[1] = 32;
    tile_info.rows[1] = 16;
    tile_info.colsb[2] = 32;
    tile_info.rows[2] = 16;
    _tile_loadconfig(&tile_info);

    __bf16 src1[512]{};
    __bf16 src2[512]{};
    float res[256]{};

    _tile_loadd(2, src1, 128);
    _tile_loadd(1, src2, 128);
    _tile_loadd(0, res, 128);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(1024)
    for (uint64_t k = 0; k < steps; k++) {
        _tile_dpbf16ps(0, 1, 2);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    _tile_stored(0, res, 16);
    _tile_release();

    uint64_t ops_per_output = 32 /* mul */ + 32 /* add */;
    uint64_t ops = steps * ops_per_output * 16 * 16 /* num outputs */;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.output, res, sizeof(res));
    return r;
};

VMOPSMEM_EXPORT Result
vnni_s8_s32(uint64_t steps) {
    int8_t src1[4];
    int8_t src2[4];
    int32_t res;

    __m128i A, B, C;
    A = _mm_load_si128((__m128i *) &src1);
    B = _mm_load_si128((__m128i *) &src2);
    C = _mm_load_si128((__m128i *) &res);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(1024)
    for (uint64_t k = 0; k < steps; k++) {
        C = _mm_dpbusd_epi32(C, B, A);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    _mm_store_si128((__m128i *) &res, C);

    uint64_t ops = steps * 4;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.output, &res, sizeof(res));
    return r;
};

VMOPSMEM_EXPORT Result
vnni_f16_f32(uint64_t steps) {
    int8_t src1[4];
    int8_t src2[4];
    int32_t res;

    __m128i A, B, C;
    A = _mm_load_si128((__m128i *) &src1);
    B = _mm_load_si128((__m128i *) &src2);
    C = _mm_load_si128((__m128i *) &res);

    auto start = std::chrono::high_resolution_clock::now();

#pragma clang loop unroll_count(32)
    for (uint64_t k = 0; k < steps; k++) {
        C = _mm_dpwssd_epi32(C, B, A);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    _mm_store_si128((__m128i *) &res, C);

    uint64_t ops = steps * 2;

    auto r = Result{duration.count(), ops};
    std::memcpy(r.output, &res, sizeof(res));
    return r;
};

#ifdef __cplusplus
}
#endif
