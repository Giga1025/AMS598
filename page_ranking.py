from mpi4py import MPI
from collections import defaultdict
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def load_edges_all_files(num_files=10):
    """Load all edges from data files (only called by rank 0)"""
    all_edges = []
    path = "/gpfs/projects/AMS598/projects2025_data/project2_data"
    
    for i in range(1, num_files + 1):
        filepath = f"{path}/data{i}.txt"
        try:
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            src, dst = map(int, line.split(","))
                            all_edges.append((src, dst))
                        except ValueError:
                            continue
        except FileNotFoundError:
            continue
    
    return all_edges


def distribute_edges(edges):
    """Distribute edges evenly across all processes using MPI scatter"""
    if rank == 0:
        chunk_size = len(edges) // size
        remainder = len(edges) % size
        chunks = []
        start = 0
        
        for i in range(size):
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(edges[start:end])
            start = end
    else:
        chunks = None
    
    return comm.scatter(chunks, root=0)


def build_local_graph(local_edges):
    """Build local graph structures from assigned edges"""
    out_counts = defaultdict(int)
    in_links = defaultdict(list)
    
    for src, dst in local_edges:
        out_counts[src] += 1
        in_links[dst].append(src)
    
    return out_counts, in_links


def gather_and_merge(out_counts, in_links):
    """Gather local graphs and merge into complete graph structure"""
    all_out_counts = comm.gather(out_counts, root=0)
    all_in_links = comm.gather(in_links, root=0)

    if rank == 0:
        merged_out = defaultdict(int)
        merged_in = defaultdict(list)

        for d in all_out_counts:
            for k, v in d.items():
                merged_out[k] += v

        for d in all_in_links:
            for k, v in d.items():
                merged_in[k].extend(v)

        nodes = set(merged_out.keys()).union(merged_in.keys())
        N = len(nodes)
        pagerank = {node: 1.0 / N for node in nodes}
        
        for node in nodes:
            if node not in merged_out:
                merged_out[node] = 0

        return merged_out, merged_in, pagerank, N
    else:
        return None, None, None, None


def broadcast_graph_data(merged_out, merged_in, pagerank, N):
    """Broadcast complete graph data to all processes"""
    merged_out = comm.bcast(merged_out, root=0)
    merged_in = comm.bcast(merged_in, root=0)
    pagerank = comm.bcast(pagerank, root=0)
    N = comm.bcast(N, root=0)
    return merged_out, merged_in, pagerank, N


def compute_pagerank_iteration(pagerank, merged_in, merged_out, N, beta=0.9):
    """MAP phase: Compute local PageRank contributions for assigned nodes"""
    local_contribs = defaultdict(float)
    all_nodes = sorted(pagerank.keys())
    my_nodes = all_nodes[rank::size]
    
    for node in my_nodes:
        incoming_rank = 0.0
        if node in merged_in:
            for src in merged_in[node]:
                out_degree = merged_out[src]
                if out_degree > 0:
                    incoming_rank += pagerank[src] / out_degree
        
        local_contribs[node] = (1 - beta) / N + beta * incoming_rank
    
    return local_contribs


def gather_and_update_pagerank(local_contribs):
    """REDUCE phase: Gather all local contributions and combine into new PageRank"""
    all_contribs = comm.gather(local_contribs, root=0)

    if rank == 0:
        new_pagerank = {}
        for d in all_contribs:
            for k, v in d.items():
                new_pagerank[k] = v
        return new_pagerank
    else:
        return None


def save_local_contribs(local_contribs, iter_num, save_dir):
    """Save intermediate results for this iteration and rank"""
    output_file = os.path.join(save_dir, f"iter_{iter_num}_rank_{rank}.txt")
    with open(output_file, "w") as f:
        for node, val in sorted(local_contribs.items()):
            f.write(f"{node},{val:.6f}\n")


def run_pagerank_iterations(pagerank, merged_in, merged_out, N, 
                            num_iterations=4, beta=0.9, save_dir=None):
    """Run specified number of PageRank iterations with map/reduce pattern"""
    for it in range(num_iterations):
        local_contribs = compute_pagerank_iteration(pagerank, merged_in, merged_out, N, beta)
        
        if save_dir is not None:
            save_local_contribs(local_contribs, it, save_dir)
        
        new_pagerank = gather_and_update_pagerank(local_contribs)
        pagerank = comm.bcast(new_pagerank, root=0)
    
    return pagerank


def write_top_results(pagerank, output_path, top_n=10):
    """Write top N pages by PageRank to output file"""
    if rank == 0:
        top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]
        with open(output_path, "w") as f:
            for node, pr in top_nodes:
                f.write(f"{node},{pr:.6f}\n")


def main():
    """Main execution function"""
    if rank == 0:
        edges = load_edges_all_files(10)
    else:
        edges = None

    local_edges = distribute_edges(edges)
    out_counts, in_links = build_local_graph(local_edges)
    merged_out, merged_in, pagerank, N = gather_and_merge(out_counts, in_links)
    merged_out, merged_in, pagerank, N = broadcast_graph_data(merged_out, merged_in, pagerank, N)

    if rank == 0:
        save_dir = "/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/page_rank"
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None
    save_dir = comm.bcast(save_dir, root=0)

    pagerank = run_pagerank_iterations(
        pagerank,
        merged_in,
        merged_out,
        N,
        num_iterations=4,
        beta=0.9,
        save_dir=save_dir
    )

    write_top_results(pagerank, "/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/top10.txt")


if __name__ == "__main__":
    main()