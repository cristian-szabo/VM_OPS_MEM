
#include <cstdint>

#include <pthread.h>

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

VMOPSMEM_EXPORT void
set_thread_affinity(int coreId) {
    cpu_set_t cpuset{};
    CPU_SET(coreId, &cpuset);

    pthread_t thread = pthread_self();
    pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
}

VMOPSMEM_EXPORT void
set_thread_priority() {
    pthread_t thread = pthread_self();

    int policy;
    sched_param param;
    pthread_getschedparam(pthread_self(), &policy, &param);

    param.sched_priority = sched_get_priority_max(policy);
    pthread_setschedparam(pthread_self(), policy, &param);
}

#ifdef __cplusplus
}
#endif
