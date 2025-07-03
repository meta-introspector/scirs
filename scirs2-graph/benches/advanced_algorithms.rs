//! Benchmarks for advanced graph algorithms not covered in basic benchmarks
//!
//! This benchmark suite covers community detection, motifs, embeddings, flow algorithms,
//! matching algorithms, and other specialized graph operations.

#![allow(unused_imports)]
#![allow(dead_code)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use scirs2_graph::{
    algorithms::{community::*, flow::*, matching::*, motifs::*, random_walk::*, similarity::*},
    embeddings::*,
    generators,
    memory::{CSRGraph, MemmapGraph},
    performance::*,
    DiGraph, Graph,
};
use std::collections::HashMap;
use std::time::Duration;

/// Benchmark community detection algorithms
fn bench_community_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("community_detection");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let sizes = vec![500, 1000, 2000];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let graph = generators::barabasi_albert_graph(size, 5, &mut rng).unwrap();

        // Louvain algorithm
        group.bench_with_input(BenchmarkId::new("louvain", size), &graph, |b, g| {
            b.iter(|| {
                let result = louvain_community_detection(g, 1.0, 0.0001, 100);
                black_box(result)
            });
        });

        // Label propagation
        group.bench_with_input(
            BenchmarkId::new("label_propagation", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let result = label_propagation_communities(g, 100, None);
                    black_box(result)
                });
            },
        );

        // Greedy modularity optimization
        group.bench_with_input(
            BenchmarkId::new("greedy_modularity", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let result = greedy_modularity_communities(g);
                    black_box(result)
                });
            },
        );

        // K-clique percolation
        group.bench_with_input(
            BenchmarkId::new("k_clique_percolation", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let result = k_clique_percolation(g, 3);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark motif finding algorithms  
fn bench_motif_finding(c: &mut Criterion) {
    let mut group = c.benchmark_group("motif_finding");
    group.sample_size(10);

    let sizes = vec![100, 200, 500];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let graph = generators::erdos_renyi_graph(size, 0.05, &mut rng).unwrap();

        // Triangle counting
        group.bench_with_input(BenchmarkId::new("triangle_count", size), &graph, |b, g| {
            b.iter(|| {
                let result = count_triangles(g);
                black_box(result)
            });
        });

        // All triangles enumeration
        group.bench_with_input(
            BenchmarkId::new("enumerate_triangles", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let result = enumerate_triangles(g);
                    black_box(result)
                });
            },
        );

        // 4-motif counting
        group.bench_with_input(
            BenchmarkId::new("four_motif_count", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let result = count_four_motifs(g);
                    black_box(result)
                });
            },
        );

        // K-clique finding (k=4)
        group.bench_with_input(BenchmarkId::new("find_cliques_k4", size), &graph, |b, g| {
            b.iter(|| {
                let result = find_cliques(g, 4);
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark embedding algorithms
fn bench_embeddings(c: &mut Criterion) {
    let mut group = c.benchmark_group("embeddings");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    let sizes = vec![500, 1000];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let graph = generators::barabasi_albert_graph(size, 3, &mut rng).unwrap();

        // Node2Vec
        let node2vec_config = Node2VecConfig {
            dimensions: 64,
            walk_length: 20,
            num_walks: 5,
            window_size: 5,
            epochs: 1,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::new("node2vec", size), &graph, |b, g| {
            b.iter(|| {
                let mut embedder = Node2VecEmbedder::new(node2vec_config.clone());
                let result = embedder.fit_transform(g);
                black_box(result)
            });
        });

        // DeepWalk
        let deepwalk_config = DeepWalkConfig {
            dimensions: 64,
            walk_length: 20,
            num_walks: 5,
            window_size: 5,
            epochs: 1,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::new("deepwalk", size), &graph, |b, g| {
            b.iter(|| {
                let mut embedder = DeepWalkEmbedder::new(deepwalk_config.clone());
                let result = embedder.fit_transform(g);
                black_box(result)
            });
        });

        // Random walk generation (baseline)
        group.bench_with_input(BenchmarkId::new("random_walks", size), &graph, |b, g| {
            b.iter(|| {
                let mut walks = Vec::new();
                for node in 0..g.node_count().min(100) {
                    if let Ok(walk) = generate_random_walk(g, node, 20, 1.0, 1.0, &mut rng) {
                        walks.push(walk);
                    }
                }
                black_box(walks)
            });
        });
    }

    group.finish();
}

/// Benchmark flow algorithms
fn bench_flow_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("flow_algorithms");
    group.sample_size(10);

    let sizes = vec![100, 200, 500];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let mut digraph = DiGraph::<usize, f64>::new();

        // Create flow network
        for i in 0..size {
            digraph.add_node(i);
        }

        // Add edges with random capacities
        for _ in 0..(size * 2) {
            let u = rng.random_range(0..size);
            let v = rng.random_range(0..size);
            if u != v {
                let capacity = rng.random_range(1.0..10.0);
                let _ = digraph.add_edge(u, v, capacity);
            }
        }

        let source = 0;
        let sink = size - 1;

        // Max flow algorithms
        group.bench_with_input(
            BenchmarkId::new("ford_fulkerson", size),
            &digraph,
            |b, g| {
                b.iter(|| {
                    let result = ford_fulkerson_max_flow(g, source, sink);
                    black_box(result)
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("edmonds_karp", size), &digraph, |b, g| {
            b.iter(|| {
                let result = edmonds_karp_max_flow(g, source, sink);
                black_box(result)
            });
        });

        group.bench_with_input(BenchmarkId::new("push_relabel", size), &digraph, |b, g| {
            b.iter(|| {
                let result = push_relabel_max_flow(g, source, sink);
                black_box(result)
            });
        });

        // Min cost flow
        group.bench_with_input(BenchmarkId::new("min_cost_flow", size), &digraph, |b, g| {
            b.iter(|| {
                let result = min_cost_max_flow(g, source, sink);
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark matching algorithms
fn bench_matching_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("matching_algorithms");
    group.sample_size(10);

    let sizes = vec![100, 200, 500];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let graph = generators::erdos_renyi_graph(size, 0.1, &mut rng).unwrap();

        // Maximum matching
        group.bench_with_input(BenchmarkId::new("max_matching", size), &graph, |b, g| {
            b.iter(|| {
                let result = maximum_matching(g);
                black_box(result)
            });
        });

        // Maximum weighted matching
        group.bench_with_input(
            BenchmarkId::new("max_weighted_matching", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let result = maximum_weighted_matching(g);
                    black_box(result)
                });
            },
        );

        // Perfect matching check
        group.bench_with_input(
            BenchmarkId::new("perfect_matching_check", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let result = has_perfect_matching(g);
                    black_box(result)
                });
            },
        );

        // Minimum vertex cover (via matching)
        group.bench_with_input(
            BenchmarkId::new("min_vertex_cover", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let result = minimum_vertex_cover(g);
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark similarity measures
fn bench_similarity_measures(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_measures");
    group.sample_size(10);

    let sizes = vec![100, 200, 500];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let graph = generators::barabasi_albert_graph(size, 3, &mut rng).unwrap();

        // Jaccard similarity for random node pairs
        let node_pairs: Vec<(usize, usize)> = (0..100)
            .map(|_| {
                let u = rng.random_range(0..size);
                let v = rng.random_range(0..size);
                (u, v)
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("jaccard_similarity", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let mut similarities = Vec::new();
                    for &(u, v) in &node_pairs {
                        if let Ok(sim) = jaccard_similarity(g, u, v) {
                            similarities.push(sim);
                        }
                    }
                    black_box(similarities)
                });
            },
        );

        // Adamic-Adar similarity
        group.bench_with_input(BenchmarkId::new("adamic_adar", size), &graph, |b, g| {
            b.iter(|| {
                let mut similarities = Vec::new();
                for &(u, v) in &node_pairs {
                    if let Ok(sim) = adamic_adar_similarity(g, u, v) {
                        similarities.push(sim);
                    }
                }
                black_box(similarities)
            });
        });

        // Common neighbors
        group.bench_with_input(
            BenchmarkId::new("common_neighbors", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let mut counts = Vec::new();
                    for &(u, v) in &node_pairs {
                        let count = common_neighbors_count(g, u, v);
                        counts.push(count);
                    }
                    black_box(counts)
                });
            },
        );

        // Resource allocation similarity
        group.bench_with_input(
            BenchmarkId::new("resource_allocation", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let mut similarities = Vec::new();
                    for &(u, v) in &node_pairs {
                        if let Ok(sim) = resource_allocation_similarity(g, u, v) {
                            similarities.push(sim);
                        }
                    }
                    black_box(similarities)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark random walk algorithms
fn bench_random_walks(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_walks");

    let sizes = vec![1000, 5000, 10000];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let graph = generators::barabasi_albert_graph(size, 3, &mut rng).unwrap();

        // Standard random walk
        group.bench_with_input(
            BenchmarkId::new("standard_random_walk", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let mut walks = Vec::new();
                    for _ in 0..100 {
                        let start = rng.random_range(0..g.node_count());
                        if let Ok(walk) = generate_random_walk(g, start, 50, 1.0, 1.0, &mut rng) {
                            walks.push(walk);
                        }
                    }
                    black_box(walks)
                });
            },
        );

        // Biased random walk (Node2Vec style)
        group.bench_with_input(
            BenchmarkId::new("biased_random_walk", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let mut walks = Vec::new();
                    for _ in 0..100 {
                        let start = rng.random_range(0..g.node_count());
                        if let Ok(walk) = generate_random_walk(g, start, 50, 0.5, 2.0, &mut rng) {
                            walks.push(walk);
                        }
                    }
                    black_box(walks)
                });
            },
        );

        // Multiple parallel walks
        group.bench_with_input(
            BenchmarkId::new("parallel_random_walks", size),
            &graph,
            |b, g| {
                b.iter(|| {
                    let result = generate_parallel_random_walks(g, 100, 50, 1.0, 1.0);
                    black_box(result)
                });
            },
        );

        // PageRank walks
        group.bench_with_input(BenchmarkId::new("pagerank_walks", size), &graph, |b, g| {
            b.iter(|| {
                let mut walks = Vec::new();
                for _ in 0..100 {
                    let start = rng.random_range(0..g.node_count());
                    if let Ok(walk) = generate_pagerank_walk(g, start, 50, 0.15, &mut rng) {
                        walks.push(walk);
                    }
                }
                black_box(walks)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_community_detection,
    bench_motif_finding,
    bench_embeddings,
    bench_flow_algorithms,
    bench_matching_algorithms,
    bench_similarity_measures,
    bench_random_walks
);
criterion_main!(benches);
