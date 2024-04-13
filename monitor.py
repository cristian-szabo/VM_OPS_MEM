import json
import argparse

import vm_ops_mem as vom

def main():
    available_ops = list(vom.OpsType)
    convert_ops = lambda name: vom.OpsType[name]

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--report', type=int, default=60)
    parser.add_argument('-s', '--steps', type=int, default=int(1e9))
    parser.add_argument("-c", "--cores", type=int, default=None)
    parser.add_argument('-o', '--ops', type=convert_ops, choices=available_ops, nargs='+', default=[])
    args = parser.parse_args()

    supported_ops = vom.supported_ops()

    if len(args.ops):
        supported_ops = [op for op in args.ops if op in supported_ops]

    if len(supported_ops) == 0:
        raise RuntimeError("No ops supported")

    steps_fmt, steps_unit = vom.sizeof_fmt(args.steps, "Steps", 1000.0)
    steps_fmt = int(steps_fmt)

    print(
        f"Monitor started .... (Report every {args.report} seconds with {steps_fmt} {steps_unit})"
    )
    print(f"System Topology")
    print(json.dumps(vom.system_topology(), indent=4))
    print(f"---")

    monitor = vom.PerfMonitor(args.cores)

    op_id = 0

    while True:
        op = supported_ops[op_id]

        report = monitor.measure(op, args.steps, args.report)
        print(report)

        op_id = (op_id + 1) % len(supported_ops)

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as ex:
        print(ex)
    except KeyboardInterrupt:
        pass
