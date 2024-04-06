import enum

import ctypes

class ArmOpsType(enum.IntEnum):
    # MATRIX MULTIPLY ACCUMULATE
    MMLA_S8_S32 = enum.auto()   # vmmlaq_s32 [SMMLA] (input 16, weight 16, output 4)
    MMLA_BF16_F32 = enum.auto() # vbfmmlaq_f32 [BFMMLA] (input 8, weight 8, output 4)

    # MULTIPLY ACCUMULATE
    MLA_F32_F32 = enum.auto()  # vmlaq_f32 [4 x ADD, 4 x MUL] (input 4, weight 4, output 4)
    MLA_BF16_F32 = enum.auto() # vbfmlalbq_f32 [BFMLALB] (input 8, weight 8, output 4)
    MLA_S8_S16 = enum.auto()   # vmlal_s8 [SMLAL] (input 8, weight 8, output 8)

    # DOT PRODUCT
    DOT_BF16_F32 = enum.auto() # vbfdotq_f32 [BFDOT] (input 8, weight 8, output 4)
    DOT_S8_S32 = enum.auto()   # vdotq_s32 [SDOT] (input 16, weight 16, output 4)

    # FUSED MULTIPLY ACCUMULATE 
    FMA_F32_F32 = enum.auto() # vfmaq_f32 [FMLA] (input 4, weight 4, output 4)
    FMA_F16_F16 = enum.auto() # vfmaq_f16 [FMLA] (input 8, weight 8, output 8)
    FMA_F16_F32 = enum.auto()  # vfmlalq_low_f16 [FMLAL] (input 8, weight 8, output 4)

class IntelOpsType(enum.IntEnum):
    # ADVANCED MATRIX EXTENSION 
    AMX_S8_S32 = enum.auto()
    AMX_BF16_F32 = enum.auto()


class Result(ctypes.Structure):
    _fields_ = [
        ('time', ctypes.c_longlong),
        ('ops', ctypes.c_ulonglong),
        ('output', ctypes.c_char*1024)
    ]

class CpuResult(ctypes.Structure):
    _fields_ = [
        ("time", ctypes.c_longlong),
        ("cycles", ctypes.c_ulonglong),
    ]

# dynlib_file = "/home/ubuntu/VM_OPS_MEM/Build/clang_64_debug/VmOpsMem/lib/libVmOpsMem.so"
dynlib_file = "/home/ubuntu/VM_OPS_MEM/Install/lib/libVmOpsMem.so"
lib = ctypes.cdll.LoadLibrary(dynlib_file)

def supported_arm_ops():
    ops = list()
    if lib.mmla_s8_s32_support():
        ops.append(ArmOpsType.MMLA_S8_S32)
    if lib.mmla_bf16_f32_support():
        ops.append(ArmOpsType.MMLA_BF16_F32)
    if lib.mla_f32_f32_support():
        ops.append(ArmOpsType.MLA_F32_F32)
    if lib.mla_bf16_f32_support():
        ops.append(ArmOpsType.MLA_BF16_F32)
    if lib.mla_s8_s16_support():
        ops.append(ArmOpsType.MLA_S8_S16)
    if lib.dot_bf16_f32_support():
        ops.append(ArmOpsType.DOT_BF16_F32)
    if lib.dot_s8_s32_support():
        ops.append(ArmOpsType.DOT_S8_S32)
    if lib.fma_f32_f32_support():
        ops.append(ArmOpsType.FMA_F32_F32)
    if lib.fma_f16_f16_support():
        ops.append(ArmOpsType.FMA_F16_F16)
    if lib.fma_f16_f32_support():
        ops.append(ArmOpsType.FMA_F16_F32)
    return ops

def measure_arm_ops(op, steps):
    result = None
    if op == ArmOpsType.MMLA_S8_S32:
        lib.mmla_s8_s32.restype = Result
        result = lib.mmla_s8_s32(steps)
    elif op == ArmOpsType.MMLA_BF16_F32:
        lib.mmla_bf16_f32.restype = Result
        result = lib.mmla_bf16_f32(steps)
    elif op == ArmOpsType.MLA_F32_F32:
        lib.mla_f32_f32.restype = Result
        result = lib.mla_f32_f32(steps)
    elif op == ArmOpsType.MLA_BF16_F32:
        lib.mla_bf16_f32.restype = Result
        result = lib.mla_bf16_f32(steps)
    elif op == ArmOpsType.MLA_S8_S16:
        lib.mla_s8_s16.restype = Result
        result = lib.mla_s8_s16(steps)
    elif op == ArmOpsType.DOT_BF16_F32:
        lib.dot_bf16_f32.restype = Result
        result = lib.dot_bf16_f32(steps)
    elif op == ArmOpsType.DOT_S8_S32:
        lib.dot_s8_s32.restype = Result
        result = lib.dot_s8_s32(steps)
    elif op == ArmOpsType.FMA_F32_F32:
        lib.fma_f32_f32.restype = Result
        result = lib.fma_f32_f32(steps)
    elif op == ArmOpsType.FMA_F16_F16:
        lib.fma_f16_f16.restype = Result
        result = lib.fma_f16_f16(steps)
    elif op == ArmOpsType.FMA_F16_F32:
        lib.fma_f16_f32.restype = Result
        result = lib.fma_f16_f32(steps)
    else:
        raise RuntimeError(f"Measure function for op `{op}` not found!")
    return result.time, result.ops

def supported_intel_ops():
    ops = list()
    if lib.amx_s8_s32_support():
        ops.append(IntelOpsType.AMX_S8_S32)
    return ops

def measure_intel_ops(op, steps):
    result = None
    if op == IntelOpsType.AMX_S8_S32:
        lib.amx_s8_s32.restype = Result
        result = lib.amx_s8_s32(steps)
    else:
        raise RuntimeError(f"Measure function for op `{op}` not found!")
    return result.time, result.ops

if lib.arm_build():
    OpsType = ArmOpsType
    supported_ops = supported_arm_ops
    measure_ops = measure_arm_ops
else:
    OpsType = IntelOpsType
    supported_ops = supported_intel_ops
    measure_ops = measure_intel_ops

lib.init()

def cpu_time():
    lib.cpu_time.restype = CpuResult
    result = lib.cpu_time()
    return result.time, result.cycles

if lib.debug_build():
    print("WARNING: Debug build of VmOpsMem is used!")
