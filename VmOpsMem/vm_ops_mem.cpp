
#include <chrono>
#include <cstdint>
#include <thread>

#include "vmopsmem_export.h"

#ifdef __cplusplus
extern "C" {
#endif

VMOPSMEM_EXPORT int
arm_build() {
#if __aarch64__
    return 1;
#endif
    return 0;
}

VMOPSMEM_EXPORT int
debug_build() {
#if __DEBUG
    return 1;
#endif
    return 0;
}

#ifdef __cplusplus
}
#endif
