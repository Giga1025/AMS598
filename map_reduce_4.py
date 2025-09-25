import os
import argparse
from collections import Counter

# ------------ Mapper ----------------
def mapper(file_path):
    counts = Counter()
    file_path = file_path if os.path.isabs(file_path) else os.path.join(INPUT_DIR, file_path)
    with open(file_path, "r") as f:
        for line in f:
            value = line.strip()
            if value:
                counts[value] += 1
    base = os.path.basename(file_path)
    out_path = os.path.join(MAP_OUT_DIR, f"map_{base}")
    with open(out_path, "w") as out:
        for k, v in counts.items():
            out.write(f"{k} {v}\n")
    return out_path

# ------------ Reducer ----------------
def reducer(file_list):
    total = Counter()
    for fpath in file_list:
        fpath = fpath if os.path.isabs(fpath) else os.path.join(MAP_OUT_DIR, fpath)
        with open(fpath, "r") as f:
            for line in f:
                k, v = line.strip().split()
                total[k] += int(v)

    reducer_id = f"r{os.getpid()}"
    out_path = os.path.join(REDUCE_OUT_DIR, f"reduce_{reducer_id}.txt")
    with open(out_path, "w") as out:
        for k, v in total.items():
            out.write(f"{k} {v}\n")
    return out_path

# ------------ Final Merge ----------------
def final_merge(reduce_files):
    total = Counter()
    for fpath in reduce_files:
        with open(fpath, "r") as f:
            for line in f:
                k, v = line.strip().split()
                total[k] += int(v)
    top6 = total.most_common(6)
    print("Top 6 integers by frequency:")
    for k, v in top6:
        print(f"{k}: {v}")

# ------------ Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["mapper", "reducer","final"], required=True)
    parser.add_argument("--files", nargs="+", help="Files to process")
    parser.add_argument("--id", type=int, default=0)
    args = parser.parse_args()

    NETID = "Bhuma_YaswanthReddy"
    INPUT_DIR = "/gpfs/projects/AMS598/projects2025_data/project1_data/"
    MAP_OUT_DIR = f"/gpfs/projects/AMS598/class2025/{NETID}/map_outputs/"
    REDUCE_OUT_DIR = f"/gpfs/projects/AMS598/class2025/{NETID}/reduce_outputs/"

    os.makedirs(MAP_OUT_DIR, exist_ok=True)
    os.makedirs(REDUCE_OUT_DIR, exist_ok=True)

    if args.stage == "mapper":
        for file in args.files:
            mapper(file)

    elif args.stage == "reducer":
        reducer(args.files)

    elif args.stage == "final":
        reduce_files = [os.path.join(REDUCE_OUT_DIR, f) for f in os.listdir(REDUCE_OUT_DIR)]
        final_merge(reduce_files)
        
