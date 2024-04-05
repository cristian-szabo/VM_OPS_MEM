#include "vm_ops_mem.h"

#include "mem.h"

#if defined(__aarch64__)
#include "ops_arm_64.h"
#elif defined(__x86_64__)
#include "ops_x86_64.h"
#endif

PYBIND11_MODULE(PyVmOpsMem, m) {
    Mem(m);

#if defined(__aarch64__)
    Ops_Arm_64(m);
#elif defined(__x86_64__)
    Ops_X86_64(m);
#endif
}
