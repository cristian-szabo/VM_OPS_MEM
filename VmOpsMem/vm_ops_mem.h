#pragma once

#include <cstdint>
#include <vector>

#include "vmopsmem_export.h"

struct Result {
    int64_t Time;
    uint64_t Ops;
    char Output[1024];
};

struct CpuResult {
    int64_t Time;
    uint64_t Cycles;
};

struct LogicalCore {
    unsigned Index;
    unsigned PackageID;
    unsigned CoreID;
    unsigned ThreadID;
};

extern std::vector<LogicalCore> processors;

#ifdef __cplusplus
extern "C" {
#endif

VMOPSMEM_EXPORT void set_thread_affinity(int coreId);

#ifdef __cplusplus
}
#endif
