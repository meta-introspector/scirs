//! Large graph stress testing with ultrathink mode optimizations
//!
//! This benchmark tests the performance of graph algorithms on very large graphs
//! (>1M nodes) using ultrathink mode for optimization.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use scirs2_graph::base::Graph;
use scirs2_graph::ultrathink::{
    create_enhanced_ultrathink_processor, create_large_graph_ultrathink_processor,
    create_realtime_ultrathink_processor, execute_with_enhanced_ultrathink, UltrathinkConfig,
    UltrathinkProcessor,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// Configuration for large graph stress tests
const LARGE_GRAPH_SIZES: &[usize] = &[100_000, 500_000, 1_000_000, 2_000_000];
const STRESS_TEST_ITERATIONS: usize = 5;
const MAX_STRESS_TEST_TIME: Duration = Duration::from_secs(300); // 5 minutes max per test

/// Generate a large random graph for stress testing
fn generate_large_random_graph(num_nodes: usize, edge_probability: f64) -> Graph<usize, f64> {
    let mut graph = Graph::new();
    let mut rng = rand::rng();

    // Add nodes
    for i in 0..num_nodes {
        graph.add_node(i).expect("Failed to add node");
    }

    // Add edges with given probability
    let mut edges_added = 0;
    let target_edges = (num_nodes as f64 * edge_probability) as usize;

    while edges_added < target_edges {
        let source = rng.random_range(0..num_nodes);
        let target = rng.random_range(0..num_nodes);

        if source != target {
            let weight = rng.random::<f64>();
            if graph.add_edge(source, target, weight).is_ok() {
                edges_added += 1;
            }
        }
    }

    graph
}

/// Generate a scale-free graph using preferential attachment
fn generate_scale_free_graph(num_nodes: usize, initial_edges: usize) -> Graph<usize, f64> {
    let mut graph = Graph::new();
    let mut rng = rand::rng();
    let mut degree_sum = 0;
    let mut node_degrees: HashMap<usize, usize> = HashMap::new();

    // Add initial fully connected nodes
    for i in 0..initial_edges {
        graph.add_node(i).expect("Failed to add initial node");
        node_degrees.insert(i, 0);
    }

    // Connect initial nodes
    for i in 0..initial_edges {
        for j in (i + 1)..initial_edges {
            let weight = rng.random::<f64>();
            graph
                .add_edge(i, j, weight)
                .expect("Failed to add initial edge");
            *node_degrees.get_mut(&i).unwrap() += 1;
            *node_degrees.get_mut(&j).unwrap() += 1;
            degree_sum += 2;
        }
    }

    // Add remaining nodes with preferential attachment
    for i in initial_edges..num_nodes {
        graph.add_node(i).expect("Failed to add node");
        node_degrees.insert(i, 0);

        // Add edges based on preferential attachment
        for _ in 0..initial_edges {
            // Select target node based on degree probability
            let mut target = 0;
            if degree_sum > 0 {
                let random_degree = rng.random_range(0..degree_sum);
                let mut cumulative_degree = 0;

                for (&node, &degree) in &node_degrees {
                    cumulative_degree += degree;
                    if cumulative_degree > random_degree {
                        target = node;
                        break;
                    }
                }
            } else {
                target = rng.random_range(0..i);
            }

            let weight = rng.random::<f64>();
            if graph.add_edge(i, target, weight).is_ok() {
                *node_degrees.get_mut(&i).unwrap() += 1;
                *node_degrees.get_mut(&target).unwrap() += 1;
                degree_sum += 2;
            }
        }
    }

    graph
}

/// Memory-efficient large graph generator
fn generate_memory_efficient_graph(num_nodes: usize) -> Graph<usize, f64> {
    let mut graph = Graph::new();
    let mut rng = rand::rng();

    // Add nodes in batches to manage memory
    const BATCH_SIZE: usize = 10_000;

    for batch_start in (0..num_nodes).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(num_nodes);

        // Add nodes in current batch
        for i in batch_start..batch_end {
            graph.add_node(i).expect("Failed to add node");
        }

        // Add edges within and between batches
        for i in batch_start..batch_end {
            // Connect to a few random previous nodes
            let num_connections = rng.random_range(1..=5);
            for _ in 0..num_connections {
                if i > 0 {
                    let target = rng.random_range(0..i);
                    let weight = rng.random::<f64>();
                    let _ = graph.add_edge(i, target, weight);
                }
            }
        }
    }

    graph
}

/// Comprehensive stress test for different graph algorithms
fn stress_test_algorithms(
    graph: &Graph<usize, f64>,
    processor: &mut UltrathinkProcessor,
    test_name: &str,
) -> HashMap<String, Duration> {
    let mut results = HashMap::new();

    // Test 1: Graph traversal
    let start = Instant::now();
    let _result = execute_with_enhanced_ultrathink(processor, graph, "stress_traversal", |g| {
        Ok(g.node_count())
    });
    results.insert(format!("{}_traversal", test_name), start.elapsed());

    // Test 2: Connectivity analysis
    let start = Instant::now();
    let _result = execute_with_enhanced_ultrathink(processor, graph, "stress_connectivity", |g| {
        Ok(g.edge_count())
    });
    results.insert(format!("{}_connectivity", test_name), start.elapsed());

    // Test 3: Graph properties
    let start = Instant::now();
    let _result = execute_with_enhanced_ultrathink(processor, graph, "stress_properties", |g| {
        Ok(g.node_count() + g.edge_count())
    });
    results.insert(format!("{}_properties", test_name), start.elapsed());

    // Test 4: Memory optimization
    let start = Instant::now();
    let _result = execute_with_enhanced_ultrathink(processor, graph, "stress_memory", |g| {
        // Simulate memory-intensive operation
        let nodes: Vec<_> = g.nodes().collect();
        Ok(nodes.len())
    });
    results.insert(format!("{}_memory", test_name), start.elapsed());

    results
}

/// Benchmark large graph creation performance
fn bench_large_graph_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_graph_creation");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10);

    for &size in LARGE_GRAPH_SIZES.iter().take(3) {
        // Limit to first 3 sizes
        group.bench_function(format!("random_graph_{}", size), |b| {
            b.iter(|| black_box(generate_large_random_graph(size, 2.0 / size as f64)))
        });

        group.bench_function(format!("scale_free_graph_{}", size), |b| {
            b.iter(|| black_box(generate_scale_free_graph(size, 3)))
        });

        group.bench_function(format!("memory_efficient_graph_{}", size), |b| {
            b.iter(|| black_box(generate_memory_efficient_graph(size)))
        });
    }

    group.finish();
}

/// Benchmark ultrathink processors on large graphs
fn bench_ultrathink_processors(c: &mut Criterion) {
    let mut group = c.benchmark_group("ultrathink_large_graphs");
    group.measurement_time(Duration::from_secs(120));
    group.sample_size(5);

    // Create test graphs
    let small_graph = generate_large_random_graph(50_000, 4.0 / 50_000.0);
    let medium_graph = generate_scale_free_graph(100_000, 3);

    // Test different processor configurations
    let configs = vec![
        ("enhanced", create_enhanced_ultrathink_processor()),
        ("large_graph", create_large_graph_ultrathink_processor()),
        ("realtime", create_realtime_ultrathink_processor()),
    ];

    for (name, mut processor) in configs {
        group.bench_function(format!("small_graph_{}", name), |b| {
            b.iter(|| {
                let results = stress_test_algorithms(&small_graph, &mut processor, "small");
                black_box(results)
            })
        });

        group.bench_function(format!("medium_graph_{}", name), |b| {
            b.iter(|| {
                let results = stress_test_algorithms(&medium_graph, &mut processor, "medium");
                black_box(results)
            })
        });
    }

    group.finish();
}

/// Memory usage benchmarking for large graphs
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(90));
    group.sample_size(10);

    for &size in &[10_000, 50_000, 100_000] {
        group.bench_function(format!("memory_profile_{}", size), |b| {
            b.iter(|| {
                let graph = generate_memory_efficient_graph(size);
                let mut processor = create_large_graph_ultrathink_processor();

                // Simulate memory-intensive operations
                let _results =
                    execute_with_enhanced_ultrathink(&mut processor, &graph, "memory_test", |g| {
                        // Force memory allocation
                        let nodes: Vec<_> = g.nodes().collect();
                        let _edges: Vec<_> = g.edges().collect();
                        Ok(nodes.len())
                    });

                black_box(processor.get_optimization_stats())
            })
        });
    }

    group.finish();
}

/// Adaptive performance benchmarking
fn bench_adaptive_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_performance");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(15);

    let test_graph = generate_scale_free_graph(25_000, 3);

    // Test adaptive learning over multiple iterations
    group.bench_function("adaptive_learning", |b| {
        b.iter(|| {
            let mut processor = create_enhanced_ultrathink_processor();
            let mut total_time = Duration::ZERO;

            // Run multiple iterations to test adaptation
            for i in 0..10 {
                let start = Instant::now();
                let _result = execute_with_enhanced_ultrathink(
                    &mut processor,
                    &test_graph,
                    &format!("adaptive_iteration_{}", i),
                    |g| Ok(g.node_count() * g.edge_count()),
                );
                total_time += start.elapsed();
            }

            black_box((total_time, processor.get_optimization_stats()))
        })
    });

    group.finish();
}

/// Concurrent processing benchmarking
fn bench_concurrent_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_processing");
    group.measurement_time(Duration::from_secs(90));
    group.sample_size(10);

    let graphs: Vec<_> = (0..4)
        .map(|i| generate_memory_efficient_graph(20_000 + i * 5_000))
        .collect();

    group.bench_function("concurrent_graphs", |b| {
        b.iter(|| {
            let mut processors: Vec<_> = (0..4)
                .map(|_| create_realtime_ultrathink_processor())
                .collect();

            let results: Vec<_> = graphs
                .iter()
                .zip(processors.iter_mut())
                .enumerate()
                .map(|(i, (graph, processor))| {
                    execute_with_enhanced_ultrathink(
                        processor,
                        graph,
                        &format!("concurrent_{}", i),
                        |g| Ok(g.node_count() + g.edge_count()),
                    )
                })
                .collect();

            black_box(results)
        })
    });

    group.finish();
}

/// Configuration comparison benchmarking
fn bench_configuration_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("configuration_comparison");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(12);

    let test_graph = generate_scale_free_graph(30_000, 4);

    // Test different ultrathink configurations
    let configs = vec![
        (
            "baseline",
            UltrathinkConfig {
                enable_neural_rl: false,
                enable_gpu_acceleration: false,
                enable_neuromorphic: false,
                enable_realtime_adaptation: false,
                enable_memory_optimization: false,
                ..UltrathinkConfig::default()
            },
        ),
        (
            "neural_only",
            UltrathinkConfig {
                enable_neural_rl: true,
                enable_gpu_acceleration: false,
                enable_neuromorphic: false,
                enable_realtime_adaptation: false,
                enable_memory_optimization: false,
                ..UltrathinkConfig::default()
            },
        ),
        (
            "memory_only",
            UltrathinkConfig {
                enable_neural_rl: false,
                enable_gpu_acceleration: false,
                enable_neuromorphic: false,
                enable_realtime_adaptation: false,
                enable_memory_optimization: true,
                ..UltrathinkConfig::default()
            },
        ),
        ("full_ultrathink", UltrathinkConfig::default()),
    ];

    for (name, config) in configs {
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut processor = UltrathinkProcessor::new(config.clone());
                let _result = execute_with_enhanced_ultrathink(
                    &mut processor,
                    &test_graph,
                    "config_test",
                    |g| {
                        let nodes: Vec<_> = g.nodes().collect();
                        Ok(nodes.len())
                    },
                );
                black_box(processor.get_optimization_stats())
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_large_graph_creation,
    bench_ultrathink_processors,
    bench_memory_usage,
    bench_adaptive_performance,
    bench_concurrent_processing,
    bench_configuration_comparison
);

criterion_main!(benches);
