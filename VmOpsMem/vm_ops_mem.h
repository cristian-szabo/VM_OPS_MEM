#pragma once

#include <cstdint>
#include <vector>

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
