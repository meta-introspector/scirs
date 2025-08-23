//! Benchmarks for performance optimizations including SIMD, parallel processing,
//! memory-mapped graphs, and specialized data structures.

#![allow(unused_imports)]
#![allow(dead_code)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use scirs2_core::simd_ops::PlatformCapabilities;
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_graph::{
    generators,
    memory::{BitPackedGraph, CSRGraph, CompressedAdjacencyList, HybridGraph, MemmapGraph},
    performance::*,
    Graph,
};
use std::time::Duration;
use tempfile::NamedTempFile;

/// Benchmark SIMD operations vs scalar fallbacks
#[allow(dead_code)]
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");

    let sizes = vec![10_000, 100_000, 1_000_000];

    for &size in &sizes {
        // Create test data
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
        let c_vec: Vec<f32> = (0..size).map(|i| (i * 3) as f32).collect();

        // SIMD vector addition
        group.bench_with_input(
            BenchmarkId::new("simd_add_f32", size),
            &(a.clone(), b.clone()),
            |bench, (x, y)| {
                bench.iter(|| {
                    let result = f32::simd_add(x, y);
                    black_box(result)
                });
            },
        );

        // Scalar vector addition (fallback)
        group.bench_with_input(
            BenchmarkId::new("scalar_add_f32", size),
            &(a.clone(), b.clone()),
            |bench, (x, y)| {
                bench.iter(|| {
                    let result: Vec<f32> = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect();
                    black_box(result)
                });
            },
        );

        // SIMD dot product
        group.bench_with_input(
            BenchmarkId::new("simd_dot_product_f32", size),
            &(a.clone(), b.clone()),
            |bench, (x, y)| {
                bench.iter(|| {
                    let result = f32::simd_dot(x, y);
                    black_box(result)
                });
            },
        );

        // Scalar dot product
        group.bench_with_input(
            BenchmarkId::new("scalar_dot_product_f32", size),
            &(a.clone(), b.clone()),
            |bench, (x, y)| {
                bench.iter(|| {
                    let result: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
                    black_box(result)
                });
            },
        );

        // SIMD normalization
        group.bench_with_input(
            BenchmarkId::new("simd_normalize_f32", size),
            &a,
            |bench, x| {
                bench.iter(|| {
                    let result = f32::simd_normalize(x);
                    black_box(result)
                });
            },
        );

        // SIMD cosine similarity
        group.bench_with_input(
            BenchmarkId::new("simd_cosine_similarity", size),
            &(a.clone(), b.clone()),
            |bench, (x, y)| {
                bench.iter(|| {
                    let result = f32::simd_cosine_similarity(x, y);
                    black_box(result)
                });
            },
        );

        // SIMD euclidean distance
        group.bench_with_input(
            BenchmarkId::new("simd_euclidean_distance", size),
            &(a.clone(), b.clone()),
            |bench, (x, y)| {
                bench.iter(|| {
                    let result = f32::simd_euclidean_distance(x, y);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark parallel vs sequential algorithms
#[allow(dead_code)]
fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential");
    group.sample_size(10);

    let sizes = vec![10_000, 50_000, 100_000];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let graph = generators::barabasi_albert_graph(size, 5, &mut rng).unwrap();

        // Parallel degree computation
        group.bench_with_input(
            BenchmarkId::new("parallel_degree_computation", size),
            &graph,
            |b, g| {
                let config = ParallelConfig::default();
                b.iter(|| {
                    let result = parallel_degree_computation(g, &config);
                    black_box(result)
                });
            },
        );

        // Sequential degree computation
        group.bench_with_input(
            BenchmarkId::new("sequential_degree_computation", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let mut degrees = std::collections::HashMap::new();
                    for node in 0..g.node_count() {
                        degrees.insert(node, g.degree(node));
                    }
                    black_box(degrees)
                });
            },
        );

        // Parallel centrality computation
        group.bench_with_input(
            BenchmarkId::new("parallel_centrality_batch", size),
            &graph,
            |b, g| {
                let config = ParallelConfig::default();
                let nodes: Vec<usize> = (0..g.node_count().min(1000)).collect();
                b.iter(|| {
                    let result = parallel_centrality_batch(g, &nodes, &config);
                    black_box(result)
                });
            },
        );

        // Sequential centrality computation
        group.bench_with_input(
            BenchmarkId::new("sequential_centrality_batch", size),
            &graph,
            |b, g| {
                let nodes: Vec<usize> = (0..g.node_count().min(1000)).collect();
                b.iter(|| {
                    let mut centralities = Vec::new();
                    for &node in &nodes {
                        // Simplified centrality measure (degree centrality)
                        let centrality = g.degree(node) as f64 / (g.node_count() - 1) as f64;
                        centralities.push(centrality);
                    }
                    black_box(centralities)
                });
            },
        );

        // Parallel BFS from multiple sources
        group.bench_with_input(
            BenchmarkId::new("parallel_multi_source_bfs", size),
            &graph,
            |b, g| {
                let sources: Vec<usize> = (0..g.node_count().min(100)).step_by(10).collect();
                let config = ParallelConfig::default();
                b.iter(|| {
                    let result = parallel_multi_source_bfs(g, &sources, &config);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory-mapped graph operations
#[allow(dead_code)]
fn bench_memmap_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memmap_operations");
    group.sample_size(10);

    let sizes = vec![10_000, 50_000, 100_000];

    for &size in &sizes {
        // Create test CSR graph
        let mut rng = StdRng::seed_from_u64(42);
        let graph = generators::barabasi_albert_graph(size, 3, &mut rng).unwrap();

        let edges: Vec<(usize, usize, f64)> = (0..graph.node_count())
            .flat_map(|u| graph.neighbors(u).map(move |v| (u, v, 1.0)))
            .collect();

        let csr_graph = CSRGraph::from_edges(size, edges).unwrap();

        // Create temporary file for memory mapping
        let temp_file = NamedTempFile::new().unwrap();
        let mut memmap_graph = MemmapGraph::from_csr(&csr_graph, temp_file.path()).unwrap();

        // Benchmark CSR neighbor access
        group.bench_with_input(
            BenchmarkId::new("csr_neighbor_access", size),
            &csr_graph,
            |b, g| {
                b.iter(|| {
                    let mut total = 0;
                    for node in 0..g.n_nodes.min(1000) {
                        for neighbor in g.neighbors(node) {
                            total += neighbor;
                        }
                    }
                    black_box(total)
                });
            },
        );

        // Benchmark memory-mapped neighbor access
        group.bench_with_input(
            BenchmarkId::new("memmap_neighbor_access", size),
            &mut memmap_graph,
            |b, g| {
                b.iter(|| {
                    let mut total = 0;
                    for node in 0..g.node_count().min(1000) {
                        if let Ok(neighbors) = g.neighbors(node) {
                            for neighbor in neighbors {
                                total += neighbor;
                            }
                        }
                    }
                    black_box(total)
                });
            },
        );

        // Benchmark batch neighbor access
        group.bench_with_input(
            BenchmarkId::new("memmap_batch_neighbors", size),
            &mut memmap_graph,
            |b, g| {
                let nodes: Vec<usize> = (0..g.node_count().min(100)).collect();
                b.iter(|| {
                    let result = g.batch_neighbors(&nodes);
                    black_box(result)
                });
            },
        );

        // Benchmark streaming edge iteration
        group.bench_with_input(
            BenchmarkId::new("memmap_stream_edges", size),
            &mut memmap_graph,
            |b, g| {
                b.iter(|| {
                    let mut edge_count = 0;
                    let _ = g.stream_edges(|_u_v_w| {
                        edge_count += 1;
                        edge_count < 10000 // Stop after 10k edges for performance
                    });
                    black_box(edge_count)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark lazy graph metrics
#[allow(dead_code)]
fn bench_lazy_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("lazy_metrics");

    let sizes = vec![1_000, 5_000, 10_000];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let graph = generators::barabasi_albert_graph(size, 3, &mut rng).unwrap();

        // Test expensive computation with lazy evaluation
        group.bench_with_input(
            BenchmarkId::new("lazy_graph_metric_first_access", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let lazy_metric = LazyGraphMetric::new(|| {
                        // Simulate expensive computation (clustering coefficient calculation)
                        let mut total = 0.0;
                        for node in 0..g.node_count().min(100) {
                            let neighbors: Vec<_> = g.neighbors(node).collect();
                            let degree = neighbors.len();
                            if degree > 1 {
                                let mut triangles = 0;
                                for i in 0..neighbors.len() {
                                    for j in (i + 1)..neighbors.len() {
                                        if g.has_edge(&neighbors[i], &neighbors[j]) {
                                            triangles += 1;
                                        }
                                    }
                                }
                                let possible = degree * (degree - 1) / 2;
                                if possible > 0 {
                                    total += triangles as f64 / possible as f64;
                                }
                            }
                        }
                        Ok(total / g.node_count() as f64)
                    });

                    let result = lazy_metric.get();
                    black_box(result)
                });
            },
        );

        // Test subsequent accesses (should be fast)
        group.bench_with_input(
            BenchmarkId::new("lazy_graph_metric_cached_access", size),
            &graph,
            |b, g| {
                let lazy_metric = LazyGraphMetric::new(|| {
                    // Expensive computation
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    Ok(42.0f64)
                });

                // Trigger computation once
                let _ = lazy_metric.get();

                b.iter(|| {
                    let result = lazy_metric.get();
                    black_box(result)
                });
            },
        );

        // Benchmark thread-safe access
        group.bench_with_input(
            BenchmarkId::new("lazy_metric_concurrent_access", size),
            &graph,
            |b_g| {
                use std::sync::Arc;
                use std::thread;

                b.iter(|| {
                    let lazy_metric = Arc::new(LazyGraphMetric::new(|| {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        Ok(42.0f64)
                    }));

                    let handles: Vec<_> = (0..4)
                        .map(|_| {
                            let metric = lazy_metric.clone();
                            thread::spawn(move || metric.get())
                        })
                        .collect();

                    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark large graph iterators
#[allow(dead_code)]
fn bench_large_graph_iterators(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_graph_iterators");

    let sizes = vec![10_000, 50_000, 100_000];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let graph = generators::barabasi_albert_graph(size, 5, &mut rng).unwrap();

        // Standard iterator
        group.bench_with_input(
            BenchmarkId::new("standard_edge_iteration", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let mut count = 0;
                    for node in 0..g.node_count() {
                        for neighbor in g.neighbors(&node).unwrap_or_default() {
                            count += 1;
                            if count >= 100_000 {
                                break;
                            } // Limit for performance
                        }
                        if count >= 100_000 {
                            break;
                        }
                    }
                    black_box(count)
                });
            },
        );

        // Large graph iterator with chunking
        group.bench_with_input(
            BenchmarkId::new("chunked_edge_iteration", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let mut iterator = LargeGraphIterator::new(g, 10_000);
                    let mut total_edges = 0;
                    let mut chunk_count = 0;

                    while let Some(chunk) = iterator.next_chunk() {
                        total_edges += chunk.len();
                        chunk_count += 1;
                        if chunk_count >= 10 {
                            break;
                        } // Limit for performance
                    }

                    black_box((total_edges, chunk_count))
                });
            },
        );

        // Memory-efficient streaming
        group.bench_with_input(
            BenchmarkId::new("memory_efficient_processing", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let mut sum = 0u64;
                    let mut max_degree = 0usize;
                    let config = ParallelConfig::default();

                    // Process in chunks to maintain constant memory usage
                    let chunk_size = config.chunk_size;
                    for chunk_start in (0..g.node_count()).step_by(chunk_size) {
                        let chunk_end = (chunk_start + chunk_size).min(g.node_count());

                        for node in chunk_start..chunk_end {
                            let degree = g.degree(&node);
                            sum += degree as u64;
                            max_degree = max_degree.max(degree);
                        }

                        if chunk_start >= 10 * chunk_size {
                            break;
                        } // Limit for performance
                    }

                    black_box((sum, max_degree))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark platform-specific optimizations
#[allow(dead_code)]
fn bench_platform_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("platform_optimizations");

    // Detect platform capabilities
    let capabilities = scirs2_core::simd_ops::PlatformCapabilities::detect();

    let size = 100_000;
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

    // Benchmark with platform-specific optimizations
    group.bench_function("optimized_for_platform", |bench| {
        bench.iter(|| {
            let result = if capabilities.simd_available {
                f32::simd_add(&a, &b)
            } else {
                // Scalar fallback
                a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
            };
            black_box(result)
        });
    });

    // Benchmark forced scalar implementation
    group.bench_function("forced_scalar", |bench| {
        bench.iter(|| {
            let result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
            black_box(result)
        });
    });

    // Report platform capabilities
    group.bench_function("platform_detection", |bench| {
        bench.iter(|| {
            let caps = scirs2_core::simd_ops::PlatformCapabilities::detect();
            black_box(caps)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_simd_operations,
    bench_parallel_vs_sequential,
    bench_memmap_operations,
    bench_lazy_metrics,
    bench_large_graph_iterators,
    bench_platform_optimizations
);
criterion_main!(benches);
