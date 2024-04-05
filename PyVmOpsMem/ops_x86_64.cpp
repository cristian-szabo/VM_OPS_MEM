#include "ops_x86_64.h"

#include <immintrin.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA  18

typedef struct __tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tilecfg;

bool
set_tiledata_use() {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
        return false;
    } else {
        printf("\n TILE DATA USE SET - OK \n\n");
        return true;
    }

    return true;
}

void
Ops_X86_64(pybind11::module &m) {
    m.def("amx_s8", [](uint64_t steps) {
        if (!set_tiledata_use()) {
            throw std::runtime_error("Failed to enable amx");
        }

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

        int8_t src1[1024];
        int8_t src2[1024];
        int32_t res[256];

        _tile_loadd(2, src1, 64);
        _tile_loadd(3, src2, 64);
        _tile_loadd(1, res, 64);

        auto start = std::chrono::high_resolution_clock::now();

        for (uint64_t k = 0; k < steps; k++) {
            _tile_dpbssd(1, 2, 3);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        _tile_stored(1, res, 64);
        _tile_release();

        uint64_t ops = steps * 16 * 64 * 16;

        return pybind11::make_tuple(duration.count(), ops);
    });

    m.def("amx_bf16", [](uint64_t steps) {
        if (!set_tiledata_use()) {
            throw std::runtime_error("Failed to enable amx");
        }

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

        int16_t src1[512]{};
        int16_t src2[512]{};
        float res[256]{};

        _tile_loadd(2, src1, 128);
        _tile_loadd(1, src2, 128);
        _tile_loadd(0, res, 128);

        auto start = std::chrono::high_resolution_clock::now();

        for (uint64_t k = 0; k < steps; k++) {
            _tile_dpbf16ps(0, 1, 2);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        _tile_stored(0, res, 16);
        _tile_release();

        uint64_t ops = steps * 16 * 32 * 16;

        return pybind11::make_tuple(duration.count(), ops);
    });

    m.def("vnni_s8", [](uint64_t steps) {
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
            C = _mm_dpbusd_epi32(C, B, A);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        _mm_store_si128((__m128i *) &res, C);

        uint64_t ops = steps * 4;

        return pybind11::make_tuple(duration.count(), ops);
    });

    m.def("vnni_f16", [](uint64_t steps) {
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
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        _mm_store_si128((__m128i *) &res, C);

        uint64_t ops = steps * 2;

        return pybind11::make_tuple(duration.count(), ops);
    });
}
