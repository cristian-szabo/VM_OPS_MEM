import argparse

import vm_ops_mem as vom

def sizeof_fmt(num, suffix="Ops", steps=1024.0):
    for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
        if abs(num) < steps:
            return num, f"{unit}{suffix}"
        num /= steps
    return num, f"Y{suffix}"

def main():
    available_ops = [op.name.upper() for op in list(vom.OpsType)]
    convert_ops = lambda name: vom.OpsType[name]

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--report', type=int, default=60)
    parser.add_argument('-s', '--steps', type=int, default=int(1e9))
    parser.add_argument('-o', '--ops', type=convert_ops, choices=available_ops, nargs='+', default=[])
    args = parser.parse_args()

    steps_fmt, steps_unit = sizeof_fmt(args.steps, "Steps", 1000.0)
    steps_fmt = int(steps_fmt)

    print(f"Monitor started .... (Report every {args.report} seconds with {steps_fmt} {steps_unit})")
    print(f"---")

    elapsed_time = 0
    total_ops = 0
    steps = 0

    supported_ops = vom.supported_ops()

    if len(args.ops):
        supported_ops = [op for op in args.ops if op in supported_ops]

    if len(supported_ops) == 0:
        raise RuntimeError("No ops supported")

    op_id = 0

    while True:
        op = supported_ops[op_id]

        time, ops = vom.measure_ops(op, args.steps)
        time_sec = time / 1e9

        elapsed_time += time_sec
        total_ops += ops
        steps += 1

        if elapsed_time > args.report:
            peak_ops = total_ops / elapsed_time
            ai = total_ops / (elapsed_time * 1e9)
            ops_fmt, ops_unit = sizeof_fmt(total_ops)
            peak_fmt, peak_unit = sizeof_fmt(peak_ops)
            cpu_freq = peak_ops / ai
            cpu_fmt, cpu_unit = sizeof_fmt(cpu_freq, "Hz")

            print(f"Name: {op.name}")
            print(f"Time: {elapsed_time:.2f} sec")
            print(f"Ops: {ops_fmt:.2f} {ops_unit}")
            print(f"Peak: {peak_fmt:.2f} {peak_unit}/sec")
            print(f"AI: {ai:.2f} ops/cycle")
            print(f"CpuFreq: {cpu_fmt:.2f} {cpu_unit}")
            print(f"Bench: {steps} iters/report")
            print(f"---")

            elapsed_time = 0
            total_ops = 0
            steps = 0

            op_id = (op_id + 1) % len(supported_ops)

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as ex:
        print(ex)
    except KeyboardInterrupt:
        pass
