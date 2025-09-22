import os
import glob
from collections import Counter
from multiprocessing import Pool

# ------------ Mapper ----------------
def mapper(file_path):
    """Read one input file and write local counts to disk"""
    counts = Counter()
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
    """Read several mapper outputs and merge counts, then write reducer output"""
    total = Counter()
    for fpath in file_list:
        with open(fpath, "r") as f:
            for line in f:
                k, v = line.strip().split()
                total[k] += int(v)
    
    reducer_id = os.path.basename(file_list[0]).replace("map_", "").split(".")[0]
    out_path = os.path.join(REDUCE_OUT_DIR, f"reduce_{reducer_id}.txt")
    with open(out_path, "w") as out:
        for k, v in total.items():
            out.write(f"{k} {v}\n")
    return out_path


# ------------ Final Aggregation ----------------
def final_merge(reduce_files):
    """Merge all reducer outputs and print top 6"""
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

    NETID = "ybhuma"

    INPUT_DIR = "/gpfs/projects/AMS598/projects2025_data/project1_data/"
    MAP_OUT_DIR = f"/gpfs/projects/AMS598/class2025/{NETID}/map_outputs/"
    REDUCE_OUT_DIR = f"/gpfs/projects/AMS598/class2025/{NETID}/reduce_outputs/"

    os.makedirs(MAP_OUT_DIR, exist_ok=True)
    os.makedirs(REDUCE_OUT_DIR, exist_ok=True)

    files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    with Pool(processes=4) as pool:
        map_outputs = pool.map(mapper, files)

    chunk_size = len(map_outputs)//4
    reducer_inputs = [map_outputs[i:i+chunk_size] for i in range(0, len(map_outputs), chunk_size)]

    with Pool(processes=4) as pool:
        reduce_outputs = pool.map(reducer, reducer_inputs)

    # --- Final merge ---
    final_merge(reduce_outputs)
