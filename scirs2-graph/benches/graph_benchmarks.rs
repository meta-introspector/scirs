//! Performance benchmarks for scirs2-graph
//!
//! This benchmark suite measures the performance of various graph algorithms
//! and operations to enable comparison with other graph libraries like NetworkX.

#![allow(unused_imports)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use scirs2_graph::{
    Graph, DiGraph, Node, EdgeWeight,
    // Algorithms
    breadth_first_search, depth_first_search, shortest_path,
    connected_components, strongly_connected_components,
    pagerank_centrality, betweenness_centrality, minimum_spanning_tree,
    // Generators
    erdos_renyi_graph, barabasi_albert_graph,
};
use rand::prelude::*;
use std::collections::HashSet;


/// Benchmark graph creation and basic operations
fn bench_graph_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_creation");
    
    for size in &[100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::new("add_nodes", size), size, |b, &size| {
            b.iter(|| {
                let mut graph = Graph::<usize, ()>::new();
                for i in 0..size {
                    graph.add_node(i);
                }
                black_box(graph)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("add_edges_sparse", size), size, |b, &size| {
            b.iter(|| {
                let mut graph = Graph::<usize, f64>::new();
                for i in 0..size {
                    graph.add_node(i);
                }
                // Add ~2n edges (sparse graph)
                for i in 0..size {
                    let j = (i + 1) % size;
                    let k = (i + size/2) % size;
                    let _ = graph.add_edge(i, j, 1.0);
                    let _ = graph.add_edge(i, k, 1.0);
                }
                black_box(graph)
            });
        });
    }
    
    group.finish();
}

/// Benchmark traversal algorithms
fn bench_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("traversal");
    
    for size in &[100, 1000, 10000] {
        let graph = erdos_renyi_graph(*size, 0.01, Some(42)).unwrap();
        
        group.bench_with_input(BenchmarkId::new("bfs", size), &graph, |b, graph| {
            b.iter(|| {
                let result = breadth_first_search(graph, 0);
                black_box(result)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("dfs", size), &graph, |b, graph| {
            b.iter(|| {
                let result = depth_first_search(graph, 0);
                black_box(result)
            });
        });
    }
    
    group.finish();
}

/// Benchmark shortest path algorithms
fn bench_shortest_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("shortest_paths");
    
    for size in &[100, 500, 1000] {
        let graph = erdos_renyi_graph(*size, 0.05, Some(42)).unwrap();
        
        group.bench_with_input(BenchmarkId::new("dijkstra_single_source", size), &graph, |b, graph| {
            b.iter(|| {
                let result = shortest_path(graph, 0, Some(*size - 1));
                black_box(result)
            });
        });
    }
    
    group.finish();
}

/// Benchmark connectivity algorithms
fn bench_connectivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("connectivity");
    
    for size in &[100, 1000, 5000] {
        let graph = erdos_renyi_graph(*size, 0.01, Some(42)).unwrap();
        
        group.bench_with_input(BenchmarkId::new("connected_components", size), &graph, |b, graph| {
            b.iter(|| {
                let result = connected_components(graph);
                black_box(result)
            });
        });
        
        // For directed graphs
        let mut directed_graph = DiGraph::<usize, f64>::new();
        for i in 0..*size {
            directed_graph.add_node(i);
        }
        let mut rng = thread_rng();
        for _ in 0..(size * 2) {
            let u = rng.gen_range(0..*size);
            let v = rng.gen_range(0..*size);
            if u != v {
                let _ = directed_graph.add_edge(u, v, 1.0);
            }
        }
        
        group.bench_with_input(BenchmarkId::new("strongly_connected_components", size), &directed_graph, |b, graph| {
            b.iter(|| {
                let result = strongly_connected_components(graph);
                black_box(result)
            });
        });
    }
    
    group.finish();
}

/// Benchmark centrality algorithms
fn bench_centrality(c: &mut Criterion) {
    let mut group = c.benchmark_group("centrality");
    group.sample_size(10); // Reduce sample size for expensive algorithms
    
    for size in &[50, 100, 200] {
        let graph = barabasi_albert_graph(*size, 3, Some(42)).unwrap();
        
        group.bench_with_input(BenchmarkId::new("pagerank", size), &graph, |b, graph| {
            b.iter(|| {
                let result = pagerank_centrality(graph, Some(0.85), Some(100), Some(1e-6));
                black_box(result)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("betweenness_centrality", size), &graph, |b, graph| {
            b.iter(|| {
                let result = betweenness_centrality(graph);
                black_box(result)
            });
        });
    }
    
    group.finish();
}

/// Benchmark minimum spanning tree algorithms
fn bench_mst(c: &mut Criterion) {
    let mut group = c.benchmark_group("minimum_spanning_tree");
    
    for size in &[100, 500, 1000] {
        let graph = erdos_renyi_graph(*size, 0.1, Some(42)).unwrap();
        
        group.bench_with_input(BenchmarkId::new("kruskal", size), &graph, |b, graph| {
            b.iter(|| {
                let result = minimum_spanning_tree(graph);
                black_box(result)
            });
        });
    }
    
    group.finish();
}

/// Benchmark graph I/O operations
fn bench_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("io");
    
    for size in &[1000, 10000] {
        let graph = erdos_renyi_graph(*size, 0.01, Some(42)).unwrap();
        
        group.bench_with_input(BenchmarkId::new("adjacency_matrix", size), &graph, |b, graph| {
            b.iter(|| {
                let result = graph.adjacency_matrix();
                black_box(result)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("has_edge", size), &graph, |b, graph| {
            let mut rng = thread_rng();
            b.iter(|| {
                let u = rng.gen_range(0..*size);
                let v = rng.gen_range(0..*size);
                let result = graph.has_edge(&u, &v);
                black_box(result)
            });
        });
    }
    
    group.finish();
}

/// Benchmark graph generators
fn bench_generators(c: &mut Criterion) {
    let mut group = c.benchmark_group("generators");
    
    for size in &[100, 1000, 5000] {
        group.bench_with_input(BenchmarkId::new("erdos_renyi", size), size, |b, &size| {
            b.iter(|| {
                let result = erdos_renyi_graph(size, 0.01, Some(42));
                black_box(result)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("barabasi_albert", size), size, |b, &size| {
            b.iter(|| {
                let result = barabasi_albert_graph(size, 3, Some(42));
                black_box(result)
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_graph_creation,
    bench_traversal,
    bench_shortest_paths,
    bench_connectivity,
    bench_centrality,
    bench_mst,
    bench_io,
    bench_generators
);
criterion_main!(benches);