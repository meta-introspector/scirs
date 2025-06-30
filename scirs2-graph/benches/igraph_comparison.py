#!/usr/bin/env python3
"""
igraph performance benchmarks for comparison with scirs2-graph

This script runs equivalent benchmarks using igraph to enable
performance comparison with the Rust implementation.

Requirements:
    pip install igraph numpy pandas matplotlib
"""

import time
import igraph as ig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import warnings
warnings.filterwarnings('ignore')


def generate_random_graph(num_nodes: int, edge_prob: float) -> ig.Graph:
    """Generate a random graph with given number of nodes and edge probability."""
    return ig.Graph.Erdos_Renyi(num_nodes, p=edge_prob, directed=False)


def generate_scale_free_graph(num_nodes: int, m: int) -> ig.Graph:
    """Generate a scale-free graph using preferential attachment."""
    return ig.Graph.Barabasi(num_nodes, m, directed=False)


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
        G = ig.Graph()
        G.add_vertices(size)
        results['add_nodes'].append(time.perf_counter() - start)
        
        # Benchmark edge addition (sparse graph)
        start = time.perf_counter()
        G = ig.Graph()
        G.add_vertices(size)
        edges = [(i, (i + 1) % size) for i in range(size)]
        edges.extend([(i, (i + size // 2) % size) for i in range(size)])
        G.add_edges(edges)
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
            time_function(lambda: G.bfs(0))
        )
        
        # DFS
        results['dfs'].append(
            time_function(lambda: G.dfs(0))
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
        weights = [np.random.uniform(0, 10) for _ in range(G.ecount())]
        G.es['weight'] = weights
        
        results['dijkstra_single_source'].append(
            time_function(G.shortest_paths_dijkstra, source=0, weights='weight')
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
            time_function(G.connected_components)
        )
        
        # Directed graph for strongly connected components
        DG = ig.Graph.Erdos_Renyi(size, p=0.01, directed=True)
        
        results['strongly_connected_components'].append(
            time_function(DG.connected_components, mode='strong')
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
            time_function(G.pagerank, damping=0.85)
        )
        
        results['betweenness_centrality'].append(
            time_function(G.betweenness)
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
        weights = [np.random.uniform(0, 10) for _ in range(G.ecount())]
        G.es['weight'] = weights
        
        results['minimum_spanning_tree'].append(
            time_function(G.spanning_tree, weights='weight')
        )
    
    return results


def benchmark_clustering(sizes: List[int]) -> Dict[str, List[float]]:
    """Benchmark clustering algorithms."""
    results = {
        'clustering_coefficient': [],
        'community_detection_louvain': []
    }
    
    for size in sizes:
        G = generate_scale_free_graph(size, 3)
        
        results['clustering_coefficient'].append(
            time_function(G.transitivity_local_undirected)
        )
        
        results['community_detection_louvain'].append(
            time_function(G.community_multilevel)
        )
    
    return results


def benchmark_graph_properties(sizes: List[int]) -> Dict[str, List[float]]:
    """Benchmark graph property calculations."""
    results = {
        'diameter': [],
        'density': []
    }
    
    for size in sizes:
        G = generate_random_graph(size, 0.05)
        
        results['diameter'].append(
            time_function(G.diameter)
        )
        
        results['density'].append(
            time_function(G.density)
        )
    
    return results


def main():
    """Run all benchmarks and generate comparison report."""
    print("Running igraph benchmarks...")
    
    # Define test sizes matching other benchmarks
    creation_sizes = [100, 1000, 10000]
    traversal_sizes = [100, 1000, 10000]
    path_sizes = [100, 500, 1000]
    connectivity_sizes = [100, 1000, 5000]
    centrality_sizes = [50, 100, 200]
    mst_sizes = [100, 500, 1000]
    clustering_sizes = [50, 100, 200]
    properties_sizes = [100, 500, 1000]
    
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
    
    print("Clustering algorithms...")
    clustering_results = benchmark_clustering(clustering_sizes)
    for key, values in clustering_results.items():
        results[key] = dict(zip(clustering_sizes, values))
    
    print("Graph properties...")
    properties_results = benchmark_graph_properties(properties_sizes)
    for key, values in properties_results.items():
        results[key] = dict(zip(properties_sizes, values))
    
    # Save results
    with open('igraph_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to igraph_benchmark_results.json")
    
    # Generate summary report
    print("\n" + "="*60)
    print("igraph Benchmark Summary")
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
                'Library': 'igraph'
            })
    
    df = pd.DataFrame(df_data)
    df.to_csv('igraph_benchmarks.csv', index=False)
    print("\nDetailed results saved to igraph_benchmarks.csv")
    
    # Performance insights
    print("\n" + "="*60)
    print("Performance Insights")
    print("="*60)
    
    # Find fastest and slowest operations
    fastest = df.loc[df['Time (s)'].idxmin()]
    slowest = df.loc[df['Time (s)'].idxmax()]
    
    print(f"Fastest operation: {fastest['Benchmark']} (size {fastest['Size']}) - {fastest['Time (s)']:.6f}s")
    print(f"Slowest operation: {slowest['Benchmark']} (size {slowest['Size']}) - {slowest['Time (s)']:.6f}s")
    
    # Algorithm performance breakdown
    print("\nAverage performance by algorithm:")
    avg_by_algo = df.groupby('Benchmark')['Time (s)'].mean().sort_values()
    for algo, avg_time in avg_by_algo.items():
        print(f"  {algo}: {avg_time:.6f}s")


if __name__ == "__main__":
    main()