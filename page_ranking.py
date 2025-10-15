from mpi4py import MPI
import numpy as np
import os
import resource
import time

BETA = 0.9
ITERATIONS = 4
INPUT_DIR = '/gpfs/projects/AMS598/projects2025_data/project2_data'
DATA_FILES = [f'data{i}.txt' for i in range(1, 11)]
OUTPUT_DIR = '/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/page_rank'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def log_memory_usage():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[Rank {rank}] Memory usage: {rss:.2f} MB")


def parse_graph(directory_path, filenames):
    adj_list = {}
    all_pages = set()

    for filename in filenames:
        filepath = os.path.join(directory_path, filename)
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            src, dst = map(int, line.strip().split(','))
                            adj_list.setdefault(src, []).append(dst)
                            all_pages.update([src, dst])
                        except ValueError:
                            continue
        except FileNotFoundError:
            continue

    return adj_list, sorted(list(all_pages))


def scatter_pages(all_pages):
    if rank == 0:
        chunks = np.array_split(all_pages, size)
    else:
        chunks = None
    return comm.scatter(chunks, root=0)


def compute_contributions(my_pages, adj_list, pagerank, page_to_idx, N):
    contrib = np.zeros(N)
    for page in my_pages:
        if page in adj_list:
            num_links = len(adj_list[page])
            if num_links > 0:
                share = pagerank[page_to_idx[page]] / num_links
                for dst in adj_list[page]:
                    if dst in page_to_idx:
                        contrib[page_to_idx[dst]] += share
    return contrib


def save_contributions(contrib, iter_num):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = os.path.join(OUTPUT_DIR, f'iter_{iter_num}_rank_{rank}.npy')
    np.save(fname, contrib)


def aggregate_contributions(iter_num, N):
    total = np.zeros(N)
    for r in range(size):
        fname = os.path.join(OUTPUT_DIR, f'iter_{iter_num}_rank_{r}.npy')
        part = np.load(fname)
        total += part
    return total


def print_top10(all_pages, pagerank):
    results = sorted(zip(all_pages, pagerank), key=lambda x: x[1], reverse=True)
    print("\n--- Top 10 Webpages by PageRank ---")
    for i in range(10):
        page, score = results[i]
        print(f"{i+1}. Webpage: {page}\tScore: {score:.6f}")


def run_pagerank():
    start_time = time.time()

    if rank == 0:
        print("[Rank 0] Parsing graph...")
    adj_list, all_pages = parse_graph(INPUT_DIR, DATA_FILES)
    if not all_pages:
        raise ValueError("No pages found.")

    N = len(all_pages)
    page_to_idx = {page: i for i, page in enumerate(all_pages)}
    pagerank = np.full(N, 1.0 / N)

    my_pages = scatter_pages(all_pages)

    log_memory_usage()

    for i in range(ITERATIONS):
        local_contrib = compute_contributions(my_pages, adj_list, pagerank, page_to_idx, N)
        save_contributions(local_contrib, i)

        comm.barrier()

        if rank == 0:
            total_contrib = aggregate_contributions(i, N)
            pagerank = (1 - BETA) / N + BETA * total_contrib
        pagerank = comm.bcast(pagerank, root=0)

    if rank == 0:
        print_top10(all_pages, pagerank)
        print(f"\n[Rank 0] Finished in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    run_pagerank()
