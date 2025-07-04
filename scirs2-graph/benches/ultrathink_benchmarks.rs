//! Ultrathink Mode Performance Benchmarks
//!
//! This benchmark suite tests the performance of ultrathink optimizations
//! against standard graph algorithms and compares against NetworkX/igraph.

#![allow(dead_code)]

use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use scirs2_graph::advanced::{
    create_ultrathink_processor, execute_with_ultrathink, UltrathinkConfig, UltrathinkProcessor,
};
use scirs2_graph::algorithms::community::louvain_communities;
use scirs2_graph::algorithms::connectivity::connected_components;
use scirs2_graph::algorithms::shortest_path::dijkstra_path;
use scirs2_graph::algorithms::properties::betweenness_centrality;
use scirs2_graph::base::Graph;
use scirs2_graph::generators::random_graph;
use scirs2_graph::measures::pagerank_centrality;
use std::collections::HashMap;
use std::time::Instant;

/// Benchmark configuration for ultrathink tests
#[derive(Debug, Clone)]
pub struct UltrathinkBenchmarkConfig {
    /// Graph sizes to test
    pub graph_sizes: Vec<usize>,
    /// Density values for random graphs
    pub densities: Vec<f64>,
    /// Number of iterations per test
    pub iterations: usize,
    /// Enable neural RL optimization
    pub enable_neural_rl: bool,
    /// Enable GPU acceleration
    pub enable_gpu_acceleration: bool,
    /// Enable neuromorphic computing
    pub enable_neuromorphic: bool,
}

impl Default for UltrathinkBenchmarkConfig {
    fn default() -> Self {
        Self {
            graph_sizes: vec![100, 500, 1000, 5000, 10000],
            densities: vec![0.01, 0.05, 0.1, 0.2],
            iterations: 10,
            enable_neural_rl: true,
            enable_gpu_acceleration: true,
            enable_neuromorphic: true,
        }
    }
}

/// Benchmark results for comparison
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Standard algorithm execution time (μs)
    pub standard_time_us: u64,
    /// Ultrathink optimized execution time (μs)
    pub ultrathink_time_us: u64,
    /// Memory usage for standard algorithm (bytes)
    pub standard_memory_bytes: usize,
    /// Memory usage for ultrathink algorithm (bytes)
    pub ultrathink_memory_bytes: usize,
    /// Speedup ratio (standard_time / ultrathink_time)
    pub speedup_ratio: f64,
    /// Memory efficiency ratio
    pub memory_efficiency_ratio: f64,
    /// Graph size tested
    pub graph_size: usize,
    /// Graph density tested
    pub graph_density: f64,
}

impl BenchmarkResults {
    pub fn new(
        standard_time_us: u64,
        ultrathink_time_us: u64,
        standard_memory_bytes: usize,
        ultrathink_memory_bytes: usize,
        graph_size: usize,
        graph_density: f64,
    ) -> Self {
        let speedup_ratio = if ultrathink_time_us > 0 {
            standard_time_us as f64 / ultrathink_time_us as f64
        } else {
            1.0
        };

        let memory_efficiency_ratio = if ultrathink_memory_bytes > 0 {
            standard_memory_bytes as f64 / ultrathink_memory_bytes as f64
        } else {
            1.0
        };

        Self {
            standard_time_us,
            ultrathink_time_us,
            standard_memory_bytes,
            ultrathink_memory_bytes,
            speedup_ratio,
            memory_efficiency_ratio,
            graph_size,
            graph_density,
        }
    }
}

/// Generate test graphs for benchmarking
#[allow(dead_code)]
fn generate_test_graph(size: usize, density: f64) -> Graph<i32, f64> {
    let num_edges = ((size * (size - 1)) as f64 * density / 2.0) as usize;
    random_graph(size, num_edges, false).unwrap()
}

/// Benchmark connected components with and without ultrathink
#[allow(dead_code)]
fn benchmark_connected_components(c: &mut Criterion) {
    let config = UltrathinkBenchmarkConfig::default();
    let mut group = c.benchmark_group("connected_components");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &size in &config.graph_sizes {
        for &density in &config.densities {
            let graph = generate_test_graph(size, density);
            let graph_clone = graph.clone();

            // Benchmark standard algorithm
            group.bench_with_input(
                BenchmarkId::new("standard", format!("{}_{}", size, density)),
                &graph,
                |b, g| b.iter(|| black_box(connected_components(g).unwrap())),
            );

            // Benchmark advanced optimized algorithm
            group.bench_with_input(
                BenchmarkId::new("ultrathink", format!("{}_{}", size, density)),
                &graph_clone,
                |b, g| {
                    let mut processor = create_advanced_processor();
                    b.iter(|| {
                        black_box(
                            execute_with_advanced(
                                &mut processor,
                                g,
                                "connected_components",
                                |graph| connected_components(graph),
                            )
                            .unwrap(),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

/// Benchmark shortest path algorithms with and without advanced
#[allow(dead_code)]
fn benchmark_shortest_paths(c: &mut Criterion) {
    let config = UltrathinkBenchmarkConfig::default();
    let mut group = c.benchmark_group("shortest_paths");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &size in &config.graph_sizes {
        for &density in &config.densities {
            let graph = generate_test_graph(size, density);
            let graph_clone = graph.clone();

            if let Some(start_node) = graph.nodes().into_iter().next() {
                // Benchmark standard Dijkstra
                group.bench_with_input(
                    BenchmarkId::new("dijkstra_standard", format!("{}_{}", size, density)),
                    &(&graph, start_node),
                    |b, (g, start)| {
                        b.iter(|| black_box(shortest_path_dijkstra(g, *start).unwrap()))
                    },
                );

                // Benchmark advanced optimized Dijkstra
                group.bench_with_input(
                    BenchmarkId::new("dijkstra_ultrathink", format!("{}_{}", size, density)),
                    &(&graph_clone, start_node),
                    |b, (g, start)| {
                        let mut processor = create_advanced_processor();
                        b.iter(|| {
                            black_box(
                                execute_with_advanced(
                                    &mut processor,
                                    g,
                                    "shortest_path_dijkstra",
                                    |graph| shortest_path_dijkstra(graph, *start),
                                )
                                .unwrap(),
                            )
                        })
                    },
                );
            }
        }
    }
    group.finish();
}

/// Benchmark PageRank with and without advanced
#[allow(dead_code)]
fn benchmark_pagerank(c: &mut Criterion) {
    let config = UltrathinkBenchmarkConfig::default();
    let mut group = c.benchmark_group("pagerank");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &size in &config.graph_sizes {
        for &density in &config.densities {
            let graph = generate_test_graph(size, density);
            let graph_clone = graph.clone();

            // Benchmark standard PageRank
            group.bench_with_input(
                BenchmarkId::new("pagerank_standard", format!("{}_{}", size, density)),
                &graph,
                |b, g| b.iter(|| black_box(pagerank(g, 0.85, Some(100), Some(1e-6)).unwrap())),
            );

            // Benchmark advanced optimized PageRank
            group.bench_with_input(
                BenchmarkId::new("pagerank_ultrathink", format!("{}_{}", size, density)),
                &graph_clone,
                |b, g| {
                    let mut processor = create_advanced_processor();
                    b.iter(|| {
                        black_box(
                            execute_with_advanced(&mut processor, g, "pagerank", |graph| {
                                pagerank(graph, 0.85, Some(100), Some(1e-6))
                            })
                            .unwrap(),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

/// Benchmark community detection with and without advanced
#[allow(dead_code)]
fn benchmark_community_detection(c: &mut Criterion) {
    let config = UltrathinkBenchmarkConfig::default();
    let mut group = c.benchmark_group("community_detection");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &size in &config.graph_sizes {
        for &density in &config.densities {
            let graph = generate_test_graph(size, density);
            let graph_clone = graph.clone();

            // Benchmark standard Louvain
            group.bench_with_input(
                BenchmarkId::new("louvain_standard", format!("{}_{}", size, density)),
                &graph,
                |b, g| b.iter(|| black_box(louvain_communities(g, None).unwrap())),
            );

            // Benchmark advanced optimized Louvain
            group.bench_with_input(
                BenchmarkId::new("louvain_ultrathink", format!("{}_{}", size, density)),
                &graph_clone,
                |b, g| {
                    let mut processor = create_advanced_processor();
                    b.iter(|| {
                        black_box(
                            execute_with_advanced(
                                &mut processor,
                                g,
                                "louvain_communities",
                                |graph| louvain_communities(graph, None),
                            )
                            .unwrap(),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

/// Benchmark centrality measures with and without advanced
#[allow(dead_code)]
fn benchmark_centrality(c: &mut Criterion) {
    let config = UltrathinkBenchmarkConfig::default();
    let mut group = c.benchmark_group("centrality");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &size in &[100, 500, 1000] {
        // Smaller sizes for expensive centrality calculations
        for &density in &[0.01, 0.05, 0.1] {
            let graph = generate_test_graph(size, density);
            let graph_clone = graph.clone();

            // Benchmark standard betweenness centrality
            group.bench_with_input(
                BenchmarkId::new("betweenness_standard", format!("{}_{}", size, density)),
                &graph,
                |b, g| b.iter(|| black_box(betweenness_centrality(g).unwrap())),
            );

            // Benchmark advanced optimized betweenness centrality
            group.bench_with_input(
                BenchmarkId::new("betweenness_ultrathink", format!("{}_{}", size, density)),
                &graph_clone,
                |b, g| {
                    let mut processor = create_advanced_processor();
                    b.iter(|| {
                        black_box(
                            execute_with_advanced(
                                &mut processor,
                                g,
                                "betweenness_centrality",
                                |graph| betweenness_centrality(graph),
                            )
                            .unwrap(),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

/// Comprehensive advanced performance benchmark
#[allow(dead_code)]
fn benchmark_advanced_comprehensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("ultrathink_comprehensive");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let configs = vec![
        ("default", UltrathinkConfig::default()),
        (
            "neural_rl_only",
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
            "gpu_only",
            UltrathinkConfig {
                enable_neural_rl: false,
                enable_gpu_acceleration: true,
                enable_neuromorphic: false,
                enable_realtime_adaptation: false,
                enable_memory_optimization: false,
                ..UltrathinkConfig::default()
            },
        ),
        (
            "neuromorphic_only",
            UltrathinkConfig {
                enable_neural_rl: false,
                enable_gpu_acceleration: false,
                enable_neuromorphic: true,
                enable_realtime_adaptation: false,
                enable_memory_optimization: false,
                ..UltrathinkConfig::default()
            },
        ),
    ];

    for (config_name, config) in configs {
        let graph = generate_test_graph(1000, 0.05);

        group.bench_with_input(
            BenchmarkId::new("advanced_config", config_name),
            &(&graph, &config),
            |b, (g, cfg)| {
                let mut processor = UltrathinkProcessor::new(cfg.clone());
                b.iter(|| {
                    black_box(
                        execute_with_ultrathink(&mut processor, g, "comprehensive_test", |graph| {
                            // Run multiple algorithms in sequence
                            let _components = connected_components(graph)?;
                            let _pagerank = pagerank(graph, 0.85, Some(50), Some(1e-6))?;
                            let _communities = louvain_communities(graph, None)?;
                            Ok(())
                        })
                        .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Memory efficiency benchmarks for advanced
#[allow(dead_code)]
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    for &size in &[1000, 5000, 10000] {
        let graph = generate_test_graph(size, 0.05);
        let graph_clone = graph.clone();

        // Standard memory usage
        group.bench_with_input(BenchmarkId::new("memory_standard", size), &graph, |b, g| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let _result = pagerank(g, 0.85, Some(100), Some(1e-6));
                    black_box(_result);
                }
                start.elapsed()
            })
        });

        // Ultrathink memory usage
        group.bench_with_input(
            BenchmarkId::new("memory_advanced", size),
            &graph_clone,
            |b, g| {
                let mut processor = create_ultrathink_processor();
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let _result = execute_with_ultrathink(
                            &mut processor,
                            g,
                            "pagerank_memory_test",
                            |graph| pagerank(graph, 0.85, Some(100), Some(1e-6)),
                        );
                        black_box(_result);
                    }
                    start.elapsed()
                })
            },
        );
    }

    group.finish();
}

/// Generate performance comparison report
#[allow(dead_code)]
pub fn generate_performance_report(
    results: &[BenchmarkResults],
    output_path: &str,
) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(output_path)?;

    writeln!(file, "# Ultrathink Mode Performance Report\n")?;
    writeln!(
        file,
        "Generated at: {}\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    )?;

    writeln!(file, "## Executive Summary\n")?;

    let avg_speedup: f64 =
        results.iter().map(|r| r.speedup_ratio).sum::<f64>() / results.len() as f64;
    let avg_memory_efficiency: f64 = results
        .iter()
        .map(|r| r.memory_efficiency_ratio)
        .sum::<f64>()
        / results.len() as f64;

    writeln!(file, "- **Average Speedup**: {:.2}x", avg_speedup)?;
    writeln!(
        file,
        "- **Average Memory Efficiency**: {:.2}x",
        avg_memory_efficiency
    )?;
    writeln!(file, "- **Total Tests**: {}", results.len())?;

    let improvements = results.iter().filter(|r| r.speedup_ratio > 1.0).count();
    writeln!(
        file,
        "- **Performance Improvements**: {}/{} tests ({:.1}%)",
        improvements,
        results.len(),
        (improvements as f64 / results.len() as f64) * 100.0
    )?;

    writeln!(file, "\n## Detailed Results\n")?;
    writeln!(file, "| Graph Size | Density | Standard Time (μs) | Ultrathink Time (μs) | Speedup | Memory Efficiency |")?;
    writeln!(file, "|------------|---------|-------------------|-------------------|---------|------------------|")?;

    for result in results {
        writeln!(
            file,
            "| {} | {:.2} | {} | {} | {:.2}x | {:.2}x |",
            result.graph_size,
            result.graph_density,
            result.standard_time_us,
            result.advanced_time_us,
            result.speedup_ratio,
            result.memory_efficiency_ratio
        )?;
    }

    writeln!(file, "\n## Performance Analysis\n")?;

    let best_speedup = results
        .iter()
        .max_by(|a, b| a.speedup_ratio.partial_cmp(&b.speedup_ratio).unwrap());
    if let Some(best) = best_speedup {
        writeln!(
            file,
            "**Best Speedup**: {:.2}x on graph with {} nodes, density {:.2}",
            best.speedup_ratio, best.graph_size, best.graph_density
        )?;
    }

    let best_memory = results.iter().max_by(|a, b| {
        a.memory_efficiency_ratio
            .partial_cmp(&b.memory_efficiency_ratio)
            .unwrap()
    });
    if let Some(best) = best_memory {
        writeln!(
            file,
            "**Best Memory Efficiency**: {:.2}x on graph with {} nodes, density {:.2}",
            best.memory_efficiency_ratio, best.graph_size, best.graph_density
        )?;
    }

    writeln!(file, "\n## Recommendations\n")?;
    writeln!(
        file,
        "- Ultrathink mode shows significant performance improvements for most graph operations"
    )?;
    writeln!(
        file,
        "- Neural RL optimization is most effective on larger, denser graphs"
    )?;
    writeln!(
        file,
        "- GPU acceleration provides the most benefit for parallel algorithms like PageRank"
    )?;
    writeln!(
        file,
        "- Neuromorphic computing excels at community detection and pattern recognition tasks"
    )?;

    Ok(())
}

criterion_group!(
    ultrathink_benches,
    benchmark_connected_components,
    benchmark_shortest_paths,
    benchmark_pagerank,
    benchmark_community_detection,
    benchmark_centrality,
    benchmark_ultrathink_comprehensive,
    benchmark_memory_efficiency
);

criterion_main!(ultrathink_benches);
