import Install.lib.PyVmOpsMem as vom

def sizeof_fmt(num, suffix="Ops"):
    for unit in ("", "K", "M", "G", "T", "P", "E", "Z"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def main():
    time_us, ops = vom.vnni_s8(int(1e6))
    print(f"Time (us): {time_us}")
    print(f"Ops: {sizeof_fmt(ops)}")

if __name__ == "__main__":
    main()
