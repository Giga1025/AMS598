from mpi4py import MPI
from collections import defaultdict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def load_edges_all_files(num_files=10):
    all_edges = []
    path = "/gpfs/projects/AMS598/projects2025_data/project2_data"
    for i in range(1, num_files + 1):
        filepath = f"{path}/data{i}.txt"
        with open(filepath) as f:
            for line in f:
                src, dst = map(int, line.strip().split(","))
                all_edges.append((src, dst))
    return all_edges


def distribute_edges(edges):
    if rank == 0:
        chunks = [edges[i::size] for i in range(size)]
    else:
        chunks = None
    return comm.scatter(chunks, root=0)


def build_local_graph(local_edges):
    out_counts = defaultdict(int)
    in_links = defaultdict(list)
    
    for src, dst in local_edges:
        out_counts[src] += 1
        in_links[dst].append(src)
    
    return out_counts, in_links


def gather_and_merge(out_counts, in_links):
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
        pagerank = {node: 1 / N for node in nodes}
        
        return merged_out, merged_in, pagerank, N
    else:
        return None, None, None, None


def broadcast_graph_data(merged_out, merged_in, pagerank, N):
    pagerank = comm.bcast(pagerank, root=0)
    merged_in = comm.bcast(merged_in, root=0)
    merged_out = comm.bcast(merged_out, root=0)
    N = comm.bcast(N, root=0)
    return merged_out, merged_in, pagerank, N


def compute_pagerank_iteration(pagerank, merged_in, merged_out, N, beta=0.9):
    local_contribs = defaultdict(float)
    my_nodes = list(pagerank.keys())[rank::size]
    
    for node in my_nodes:
        total = 0
        
        if node in merged_in:
            for src in merged_in[node]:
                if merged_out[src] > 0:
                    total += pagerank[src] / merged_out[src]
        
        local_contribs[node] = (1 - beta) / N + beta * total
    
    return local_contribs


def gather_and_update_pagerank(local_contribs, pagerank):
    all_contribs = comm.gather(local_contribs, root=0)
    
    if rank == 0:
        new_pagerank = pagerank.copy()
        for d in all_contribs:
            for k, v in d.items():
                new_pagerank[k] = v
        pagerank = new_pagerank
    
    return comm.bcast(pagerank, root=0)


def run_pagerank_iterations(pagerank, merged_in, merged_out, N, num_iterations=4, beta=0.9):
    for it in range(num_iterations):
        local_contribs = compute_pagerank_iteration(pagerank, merged_in, merged_out, N, beta)
        pagerank = gather_and_update_pagerank(local_contribs, pagerank)
    return pagerank


def write_top_results(pagerank, output_path, top_n=10):
    if rank == 0:
        top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]
        with open(output_path, "w") as f:
            for node, pr in top_nodes:
                f.write(f"{node},{pr:.6f}\n")


def main():
    if rank == 0:
        edges = load_edges_all_files(10)
    else:
        edges = None
    
    local_edges = distribute_edges(edges)
    
    out_counts, in_links = build_local_graph(local_edges)
    
    merged_out, merged_in, pagerank, N = gather_and_merge(out_counts, in_links)
    
    merged_out, merged_in, pagerank, N = broadcast_graph_data(merged_out, merged_in, pagerank, N)
    
    pagerank = run_pagerank_iterations(pagerank, merged_in, merged_out, N, num_iterations=4)
    
    write_top_results(pagerank, "/gpfs/projects/AMS598/class2025/Bhuma_YaswanthReddy/top10.txt")


if __name__ == "__main__":
    main()