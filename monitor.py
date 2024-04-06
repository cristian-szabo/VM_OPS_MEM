import Install.lib.PyVmOpsMem as vom

def sizeof_fmt(num, suffix="Ops"):
    for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
        if abs(num) < 1024.0:
            return num, f"{unit}{suffix}"
        num /= 1024.0
    return num, f"Y{suffix}"

def main():
    elapsed_time = 0
    total_ops = 0
    while True:
        time, ops, _ = vom.dot_s8(int(1e6))
        time_sec = time / 1e9

        elapsed_time += time_sec
        total_ops += ops

        if elapsed_time > 10.0:
            peak_ops = total_ops / elapsed_time
            ai = total_ops / (elapsed_time * 1e9)
            ops_fmt, ops_unit = sizeof_fmt(total_ops)
            peak_fmt, peak_unit = sizeof_fmt(peak_ops)

            print(f"Time: {elapsed_time:.2f} sec")
            print(f"Ops: {ops_fmt:.2f} {ops_unit}")
            print(f"Peak: {peak_fmt:.2f} {peak_unit}/sec")
            print(f"AI: {ai:.2f} ops/cycle")

            elapsed_time = 0
            total_ops = 0

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
