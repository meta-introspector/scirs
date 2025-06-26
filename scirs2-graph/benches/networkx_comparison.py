#!/usr/bin/env python3
"""
NetworkX performance benchmarks for comparison with scirs2-graph

This script runs equivalent benchmarks using NetworkX to enable
performance comparison with the Rust implementation.

Requirements:
    pip install networkx numpy pandas matplotlib
"""

import time
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json


def generate_random_graph(num_nodes: int, edge_prob: float) -> nx.Graph:
    """Generate a random graph with given number of nodes and edge probability."""
    return nx.erdos_renyi_graph(num_nodes, edge_prob, seed=42)


def generate_scale_free_graph(num_nodes: int, m: int) -> nx.Graph:
    """Generate a scale-free graph using preferential attachment."""
    return nx.barabasi_albert_graph(num_nodes, m, seed=42)


def time_function(func, *args, **kwargs) -> float:
    """Time a function execution and return elapsed time in seconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return end - start


def benchmark_graph_creation(sizes: List[int]) -> Dict[str, List[float]]:
    """Benchmark graph creation operations."""
    results = {
        'add_nodes': [],
        'add_edges_sparse': []
    }
    
    for size in sizes:
        # Benchmark node addition
        start = time.perf_counter()
        G = nx.Graph()
        G.add_nodes_from(range(size))
        results['add_nodes'].append(time.perf_counter() - start)
        
        # Benchmark edge addition (sparse graph)
        start = time.perf_counter()
        G = nx.Graph()
        G.add_nodes_from(range(size))
        edges = [(i, (i + 1) % size) for i in range(size)]
        edges.extend([(i, (i + size // 2) % size) for i in range(size)])
        G.add_edges_from(edges)
        results['add_edges_sparse'].append(time.perf_counter() - start)
    
    return results


def benchmark_traversal(sizes: List[int]) -> Dict[str, List[float]]:
    """Benchmark traversal algorithms."""
    results = {
        'bfs': [],
        'dfs': []
    }
    
    for size in sizes:
        G = generate_random_graph(size, 0.01)
        
        # BFS
        results['bfs'].append(
            time_function(lambda: list(nx.bfs_edges(G, 0)))
        )
        
        # DFS
        results['dfs'].append(
            time_function(lambda: list(nx.dfs_edges(G, 0)))
        )
    
    return results


def benchmark_shortest_paths(sizes: List[int]) -> Dict[str, List[float]]:
    """Benchmark shortest path algorithms."""
    results = {
        'dijkstra_single_source': []
    }
    
    for size in sizes:
        G = generate_random_graph(size, 0.05)
        # Add random weights
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = np.random.uniform(0, 10)
        
        results['dijkstra_single_source'].append(
            time_function(nx.single_source_dijkstra_path_length, G, 0)
        )
    
    return results


def benchmark_connectivity(sizes: List[int]) -> Dict[str, List[float]]:
    """Benchmark connectivity algorithms."""
    results = {
        'connected_components': [],
        'strongly_connected_components': []
    }
    
    for size in sizes:
        # Undirected graph for connected components
        G = generate_random_graph(size, 0.01)
        results['connected_components'].append(
            time_function(lambda: list(nx.connected_components(G)))
        )
        
        # Directed graph for strongly connected components
        DG = nx.DiGraph()
        DG.add_nodes_from(range(size))
        # Add ~2n random edges
        edges = [(np.random.randint(0, size), np.random.randint(0, size)) 
                 for _ in range(size * 2)]
        DG.add_edges_from(edges)
        
        results['strongly_connected_components'].append(
            time_function(lambda: list(nx.strongly_connected_components(DG)))
        )
    
    return results


def benchmark_centrality(sizes: List[int]) -> Dict[str, List[float]]:
    """Benchmark centrality algorithms."""
    results = {
        'pagerank': [],
        'betweenness_centrality': []
    }
    
    for size in sizes:
        G = generate_scale_free_graph(size, 3)
        
        results['pagerank'].append(
            time_function(nx.pagerank, G, alpha=0.85, max_iter=100, tol=1e-6)
        )
        
        results['betweenness_centrality'].append(
            time_function(nx.betweenness_centrality, G)
        )
    
    return results


def benchmark_mst(sizes: List[int]) -> Dict[str, List[float]]:
    """Benchmark minimum spanning tree algorithms."""
    results = {
        'minimum_spanning_tree': []
    }
    
    for size in sizes:
        G = generate_random_graph(size, 0.1)
        # Add random weights
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = np.random.uniform(0, 10)
        
        results['minimum_spanning_tree'].append(
            time_function(nx.minimum_spanning_tree, G)
        )
    
    return results


def plot_comparison(rust_results: Dict, python_results: Dict, title: str):
    """Create comparison plots between Rust and Python implementations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    benchmarks = list(python_results.keys())
    
    for idx, benchmark in enumerate(benchmarks[:6]):
        ax = axes[idx]
        
        if benchmark in rust_results and benchmark in python_results:
            sizes = list(rust_results[benchmark].keys())
            rust_times = [rust_results[benchmark][str(s)] for s in sizes]
            python_times = python_results[benchmark]
            
            x = np.arange(len(sizes))
            width = 0.35
            
            ax.bar(x - width/2, rust_times, width, label='scirs2-graph', color='#1f77b4')
            ax.bar(x + width/2, python_times, width, label='NetworkX', color='#ff7f0e')
            
            ax.set_xlabel('Graph Size')
            ax.set_ylabel('Time (seconds)')
            ax.set_title(benchmark.replace('_', ' ').title())
            ax.set_xticks(x)
            ax.set_xticklabels(sizes)
            ax.legend()
            ax.set_yscale('log')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Run all benchmarks and generate comparison report."""
    print("Running NetworkX benchmarks...")
    
    # Define test sizes
    creation_sizes = [100, 1000, 10000]
    traversal_sizes = [100, 1000, 10000]
    path_sizes = [100, 500, 1000]
    connectivity_sizes = [100, 1000, 5000]
    centrality_sizes = [50, 100, 200]
    mst_sizes = [100, 500, 1000]
    
    # Run benchmarks
    results = {}
    
    print("Graph creation...")
    creation_results = benchmark_graph_creation(creation_sizes)
    for key, values in creation_results.items():
        results[key] = dict(zip(creation_sizes, values))
    
    print("Traversal algorithms...")
    traversal_results = benchmark_traversal(traversal_sizes)
    for key, values in traversal_results.items():
        results[key] = dict(zip(traversal_sizes, values))
    
    print("Shortest paths...")
    path_results = benchmark_shortest_paths(path_sizes)
    for key, values in path_results.items():
        results[key] = dict(zip(path_sizes, values))
    
    print("Connectivity...")
    connectivity_results = benchmark_connectivity(connectivity_sizes)
    for key, values in connectivity_results.items():
        results[key] = dict(zip(connectivity_sizes, values))
    
    print("Centrality measures...")
    centrality_results = benchmark_centrality(centrality_sizes)
    for key, values in centrality_results.items():
        results[key] = dict(zip(centrality_sizes, values))
    
    print("Minimum spanning tree...")
    mst_results = benchmark_mst(mst_sizes)
    for key, values in mst_results.items():
        results[key] = dict(zip(mst_sizes, values))
    
    # Save results
    with open('networkx_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to networkx_benchmark_results.json")
    
    # Generate summary report
    print("\n" + "="*60)
    print("NetworkX Benchmark Summary")
    print("="*60)
    
    for benchmark, times in results.items():
        print(f"\n{benchmark}:")
        for size, time_val in times.items():
            print(f"  Size {size}: {time_val:.6f} seconds")
    
    # Create performance comparison table
    df_data = []
    for benchmark, times in results.items():
        for size, time_val in times.items():
            df_data.append({
                'Benchmark': benchmark,
                'Size': size,
                'Time (s)': time_val,
                'Library': 'NetworkX'
            })
    
    df = pd.DataFrame(df_data)
    df.to_csv('networkx_benchmarks.csv', index=False)
    print("\nDetailed results saved to networkx_benchmarks.csv")


if __name__ == "__main__":
    main()