import os
import enum

import concurrent.futures

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


class X86OpsType(enum.IntEnum):
    # ADVANCED MATRIX EXTENSION
    AMX_S8_S32 = enum.auto()
    AMX_BF16_F32 = enum.auto()

    # VECTOR NEURAL NETWORK
    VNN_S8_S32 = enum.auto()
    VNN_F16_F32 = enum.auto()


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


class LogicalCore(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint),
        ("package_id", ctypes.c_uint),
        ("core_id", ctypes.c_uint),
        ("smt_id", ctypes.c_uint),
    ]


lib = None
OpsType = None
supported_ops = None
measure_ops = None


def init():
    global lib
    global OpsType
    global supported_ops
    global measure_ops

    dynlib_file = os.path.join(os.getcwd(), "Install/lib/libVmOpsMem.so")
    lib = ctypes.cdll.LoadLibrary(dynlib_file)
    lib.init()

    if lib.debug_build():
        print("---------------------------------------------")
        print("| WARNING: Debug build of VmOpsMem is used! |")
        print("---------------------------------------------")

    if lib.arm_build():
        OpsType = ArmOpsType
        supported_ops = supported_arm_ops
        measure_ops = measure_arm_ops
    else:
        OpsType = X86OpsType
        supported_ops = supported_x86_ops
        measure_ops = measure_x86_ops


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


def supported_x86_ops():
    ops = list()
    if lib.amx_s8_s32_support():
        ops.append(X86OpsType.AMX_S8_S32)
    if lib.amx_bf16_f32_support():
        ops.append(X86OpsType.AMX_BF16_F32)
    if lib.vnn_s8_s32_support():
        ops.append(X86OpsType.VNN_S8_S32)
    if lib.vnn_f16_f32_support():
        ops.append(X86OpsType.VNN_F16_F32)
    return ops


def measure_x86_ops(op, steps):
    result = None
    if op == X86OpsType.AMX_S8_S32:
        lib.amx_s8_s32.restype = Result
        result = lib.amx_s8_s32(steps)
    elif op == X86OpsType.AMX_BF16_F32:
        lib.amx_bf16_f32.restype = Result
        result = lib.amx_bf16_f32(steps)
    elif op == X86OpsType.VNN_S8_S32:
        lib.vnn_s8_s32.restype = Result
        result = lib.vnn_s8_s32(steps)
    elif op == X86OpsType.VNN_F16_F32:
        lib.vnn_f16_f32.restype = Result
        result = lib.vnn_f16_f32(steps)
    else:
        raise RuntimeError(f"Measure function for op `{op}` not found!")
    return result.time, result.ops

def cpu_time():
    lib.cpu_time.restype = CpuResult
    result = lib.cpu_time()
    return result.time, result.cycles


def set_thread_affinity(core_id):
    lib.set_thread_affinity.argtypes = [ctypes.c_int32]
    lib.set_thread_affinity(core_id)


def set_thread_priority():
    lib.set_thread_priority()


def logical_cores():
    num_cores = os.cpu_count()
    result = (LogicalCore * num_cores)()
    lib.logical_cores.argtypes = [ctypes.POINTER(LogicalCore), ctypes.c_int]
    lib.logical_cores(result, ctypes.c_int(num_cores))
    return result


def system_topology():
    cpus = logical_cores()
    system = dict()
    for cpu_info in cpus:
        socket_key = f"Socket#{cpu_info.package_id}"
        core_key = f"Core#{cpu_info.core_id}"
        cpu_key = f"CPU#{cpu_info.smt_id}"
        thread_key = f"Thread#{cpu_info.index}"

        if socket_key not in system:
            system[socket_key] = dict()

        if core_key not in system[socket_key]:
            system[socket_key][core_key] = dict()

        if cpu_key not in system[socket_key][core_key]:
            system[socket_key][core_key][cpu_key] = dict()

        system[socket_key][core_key][cpu_key] = thread_key
    return system

def sizeof_fmt(num, suffix="Ops", steps=1024.0):
    for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
        if abs(num) < steps:
            return num, f"{unit}{suffix}"
        num /= steps
    return num, f"Y{suffix}"


class PerfReport:
    def __init__(self, name, ratio=1):
        self.name = name
        self.ratio = ratio
        self.elapsed_time = 0
        self.total_ops = 0
        self.total_freq = 0
        self.steps = 0

    def update(self, elapsed_time, total_ops, total_freq, steps):
        self.elapsed_time += elapsed_time
        self.total_ops += total_ops
        self.total_freq += total_freq
        self.steps += steps

    def __str__(self):
        peak_ops = self.total_ops / self.elapsed_time
        ai = self.total_ops / (self.elapsed_time * 1e9)
        cpu_freq = self.total_freq / self.steps
        ops_fmt, ops_unit = sizeof_fmt(self.total_ops)
        peak_fmt, peak_unit = sizeof_fmt(peak_ops)
        cpu_fmt, cpu_unit = sizeof_fmt(cpu_freq, "Hz")
        str = ""
        str += f"Name: {self.name}\n"
        str += f"Time: {self.elapsed_time / self.ratio:.2f} sec\n"
        str += f"Ops: {ops_fmt / self.ratio:.2f} {ops_unit}\n"
        str += f"Peak: {peak_fmt:.2f} {peak_unit}/sec\n"
        str += f"AI: {ai:.2f} ops/cycle\n"
        str += f"CpuFreq: {cpu_fmt:.2f} {cpu_unit}\n"
        str += f"Bench: {self.steps / self.ratio} iters/report\n"
        return str


class PerfMonitor:
    def __init__(self, num_cores=None):
        self.physical_cores = [core for core in logical_cores() if core.smt_id == 0]
        self.num_cores = len(self.physical_cores)

        if num_cores is not None:
            assert num_cores <= self.num_cores
            self.num_cores = num_cores
            self.physical_cores = self.physical_cores[0:num_cores]

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_cores)

    def measure(self, op, steps, time):
        report_futures = list(
            [
                self.executor.submit(PerfMonitor.worker, core_info, op, steps, time)
                for core_info in self.physical_cores
            ]
        )

        report = PerfReport(op.name, self.num_cores)
        for future in report_futures:
            core_report = future.result()
            report.update(
                core_report.elapsed_time,
                core_report.total_ops,
                core_report.total_freq,
                core_report.steps,
            )

        return report

    @staticmethod
    def worker(core_info, op, steps, time):
        set_thread_affinity(core_info.core_id)
        set_thread_priority()

        report = PerfReport(op.name)

        while report.elapsed_time < time:
            time_start, cycles_start = cpu_time()
            ops_time, ops_count = measure_ops(op, steps)
            time_end, cycles_end = cpu_time()

            time_elapsed = time_end - time_start
            time_elapsed = time_elapsed / 1e9
            cycles_elapsed = cycles_end - cycles_start
            freq = cycles_elapsed / time_elapsed
            ops_time = ops_time / 1e9

            report.update(ops_time, ops_count, freq, 1)

        return report
