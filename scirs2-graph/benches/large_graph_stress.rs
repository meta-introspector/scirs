//! Comprehensive stress testing utilities for large graphs (>1M nodes)
//!
//! This benchmark suite provides utilities for testing scirs2-graph at scale,
//! including memory profiling, performance monitoring, and edge case testing.

#![allow(unused_imports)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_graph::{
    algorithms, generators, measures, DiGraph, Graph, Node, EdgeWeight,
};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Configuration for stress tests
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StressTestConfig {
    /// Node counts to test
    node_counts: Vec<usize>,
    /// Edge densities to test (probability for ER graphs)
    edge_densities: Vec<f64>,
    /// Algorithms to benchmark
    algorithms: Vec<String>,
    /// Memory limit in MB
    memory_limit_mb: usize,
    /// Timeout for individual tests
    timeout_seconds: u64,
    /// Number of samples for statistical algorithms
    sample_size: usize,
    /// Enable parallel processing
    parallel: bool,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            node_counts: vec![100_000, 500_000, 1_000_000, 2_000_000, 5_000_000],
            edge_densities: vec![0.00001, 0.00005, 0.0001],
            algorithms: vec![
                "bfs".to_string(),
                "connected_components".to_string(),
                "pagerank".to_string(),
                "degree_distribution".to_string(),
                "clustering_coefficient".to_string(),
            ],
            memory_limit_mb: 16_384, // 16 GB
            timeout_seconds: 300,    // 5 minutes
            sample_size: 1000,
            parallel: true,
        }
    }
}

/// Results from a stress test run
#[derive(Debug, Serialize, Deserialize)]
struct StressTestResults {
    config: StressTestConfig,
    graph_generation: Vec<GraphGenerationResult>,
    algorithm_performance: Vec<AlgorithmResult>,
    memory_profile: MemoryProfile,
    summary: TestSummary,
}

#[derive(Debug, Serialize, Deserialize)]
struct GraphGenerationResult {
    nodes: usize,
    edges: usize,
    generation_time_ms: u128,
    memory_used_mb: f64,
    graph_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct AlgorithmResult {
    algorithm: String,
    graph_size: usize,
    execution_time_ms: u128,
    memory_delta_mb: f64,
    result_summary: String,
    parallel_speedup: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MemoryProfile {
    peak_memory_mb: f64,
    average_memory_mb: f64,
    memory_efficiency_ratio: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct TestSummary {
    total_runtime_seconds: f64,
    largest_graph_tested: usize,
    failures: Vec<String>,
    warnings: Vec<String>,
}

/// Memory monitoring utilities
mod memory {
    use std::fs;
    
    pub fn get_current_memory_mb() -> f64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(status) = fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<f64>() {
                                return kb / 1024.0;
                            }
                        }
                    }
                }
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            // macOS memory monitoring would go here
        }
        
        #[cfg(target_os = "windows")]
        {
            // Windows memory monitoring would go here
        }
        
        0.0
    }
    
    pub fn check_memory_limit(limit_mb: usize) -> bool {
        get_current_memory_mb() < limit_mb as f64
    }
}

/// Benchmark large graph generation
fn bench_graph_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_graph_generation");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));
    
    let config = StressTestConfig::default();
    
    for &node_count in &config.node_counts[..3] { // Test first 3 sizes
        // Erdős-Rényi
        for &density in &config.edge_densities {
            let id = format!("erdos_renyi_{}_density_{}", node_count, density);
            group.bench_with_input(
                BenchmarkId::new("erdos_renyi", &id),
                &(node_count, density),
                |b, &(n, p)| {
                    b.iter(|| {
                        let g = generators::erdos_renyi_graph(n, p, None).unwrap();
                        black_box(g)
                    });
                },
            );
        }
        
        // Barabási-Albert
        group.bench_with_input(
            BenchmarkId::new("barabasi_albert", node_count),
            &node_count,
            |b, &n| {
                b.iter(|| {
                    let g = generators::barabasi_albert_graph(n, 3, None).unwrap();
                    black_box(g)
                });
            },
        );
        
        // Check memory
        if !memory::check_memory_limit(config.memory_limit_mb) {
            println!("Memory limit reached, stopping generation benchmarks");
            break;
        }
    }
    
    group.finish();
}

/// Benchmark algorithms on large graphs
fn bench_large_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_graph_algorithms");
    group.sample_size(10);
    
    // Pre-generate test graphs
    let test_graphs = vec![
        ("small", generators::barabasi_albert_graph(100_000, 3, None).unwrap()),
        ("medium", generators::barabasi_albert_graph(500_000, 3, None).unwrap()),
        ("large", generators::barabasi_albert_graph(1_000_000, 3, None).unwrap()),
    ];
    
    for (size_name, graph) in &test_graphs {
        let node_count = graph.node_count();
        
        // BFS from random node
        group.bench_with_input(
            BenchmarkId::new("bfs", size_name),
            graph,
            |b, g| {
                b.iter(|| {
                    let result = algorithms::breadth_first_search(g, &0).unwrap();
                    black_box(result)
                });
            },
        );
        
        // Connected components
        group.bench_with_input(
            BenchmarkId::new("connected_components", size_name),
            graph,
            |b, g| {
                b.iter(|| {
                    let result = algorithms::connected_components(g).unwrap();
                    black_box(result)
                });
            },
        );
        
        // PageRank (limited iterations)
        group.bench_with_input(
            BenchmarkId::new("pagerank_10iter", size_name),
            graph,
            |b, g| {
                b.iter(|| {
                    let result = algorithms::pagerank(g, 0.85, Some(10)).unwrap();
                    black_box(result)
                });
            },
        );
        
        // Degree distribution
        group.bench_with_input(
            BenchmarkId::new("degree_distribution", size_name),
            graph,
            |b, g| {
                b.iter(|| {
                    let degrees: Vec<usize> = (0..g.node_count())
                        .map(|i| g.degree(i))
                        .collect();
                    black_box(degrees)
                });
            },
        );
    }
    
    group.finish();
}

/// Stress test streaming operations
fn bench_streaming_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_operations");
    group.sample_size(10);
    
    // Test processing graphs in chunks
    let chunk_sizes = vec![10_000, 50_000, 100_000];
    let total_nodes = 1_000_000;
    
    for &chunk_size in &chunk_sizes {
        group.bench_with_input(
            BenchmarkId::new("streaming_degree_calc", chunk_size),
            &(total_nodes, chunk_size),
            |b, &(total, chunk)| {
                b.iter(|| {
                    let mut total_degree = 0u64;
                    let mut max_degree = 0usize;
                    
                    for chunk_start in (0..total).step_by(chunk) {
                        let chunk_end = (chunk_start + chunk).min(total);
                        let chunk_graph = generators::path_graph(chunk_end - chunk_start);
                        
                        for i in 0..chunk_graph.node_count() {
                            let degree = chunk_graph.degree(i);
                            total_degree += degree as u64;
                            max_degree = max_degree.max(degree);
                        }
                    }
                    
                    black_box((total_degree, max_degree))
                });
            },
        );
    }
    
    group.finish();
}

/// Run comprehensive stress test suite
fn run_stress_test_suite(config: StressTestConfig) -> StressTestResults {
    let start_time = Instant::now();
    let mut results = StressTestResults {
        config: config.clone(),
        graph_generation: Vec::new(),
        algorithm_performance: Vec::new(),
        memory_profile: MemoryProfile {
            peak_memory_mb: 0.0,
            average_memory_mb: 0.0,
            memory_efficiency_ratio: 0.0,
        },
        summary: TestSummary {
            total_runtime_seconds: 0.0,
            largest_graph_tested: 0,
            failures: Vec::new(),
            warnings: Vec::new(),
        },
    };
    
    let mut memory_samples = Vec::new();
    let initial_memory = memory::get_current_memory_mb();
    
    // Test graph generation
    for &node_count in &config.node_counts {
        if !memory::check_memory_limit(config.memory_limit_mb) {
            results.summary.warnings.push(format!(
                "Skipping {} nodes due to memory limit", node_count
            ));
            continue;
        }
        
        let gen_start = Instant::now();
        let pre_gen_memory = memory::get_current_memory_mb();
        
        match generators::barabasi_albert_graph(node_count, 3, None) {
            Ok(graph) => {
                let gen_time = gen_start.elapsed();
                let post_gen_memory = memory::get_current_memory_mb();
                
                results.graph_generation.push(GraphGenerationResult {
                    nodes: graph.node_count(),
                    edges: graph.edge_count(),
                    generation_time_ms: gen_time.as_millis(),
                    memory_used_mb: post_gen_memory - pre_gen_memory,
                    graph_type: "barabasi_albert".to_string(),
                });
                
                results.summary.largest_graph_tested = 
                    results.summary.largest_graph_tested.max(node_count);
                
                memory_samples.push(post_gen_memory);
                
                // Test algorithms on this graph
                for algorithm in &config.algorithms {
                    if let Some(result) = test_algorithm(&graph, algorithm, &config) {
                        results.algorithm_performance.push(result);
                    }
                }
            }
            Err(e) => {
                results.summary.failures.push(format!(
                    "Failed to generate {} node graph: {}", node_count, e
                ));
            }
        }
    }
    
    // Calculate memory profile
    if !memory_samples.is_empty() {
        results.memory_profile.peak_memory_mb = memory_samples.iter()
            .fold(0.0f64, |a, &b| a.max(b));
        results.memory_profile.average_memory_mb = memory_samples.iter()
            .sum::<f64>() / memory_samples.len() as f64;
        results.memory_profile.memory_efficiency_ratio = 
            results.memory_profile.average_memory_mb / results.memory_profile.peak_memory_mb;
    }
    
    results.summary.total_runtime_seconds = start_time.elapsed().as_secs_f64();
    results
}

fn test_algorithm(
    graph: &Graph<usize, f64>, 
    algorithm: &str,
    config: &StressTestConfig
) -> Option<AlgorithmResult> {
    let start = Instant::now();
    let pre_memory = memory::get_current_memory_mb();
    
    let result_summary = match algorithm {
        "bfs" => {
            match algorithms::breadth_first_search(graph, &0) {
                Ok(order) => format!("Visited {} nodes", order.len()),
                Err(e) => return None,
            }
        }
        "connected_components" => {
            match algorithms::connected_components(graph) {
                Ok(components) => format!("Found {} components", components.len()),
                Err(e) => return None,
            }
        }
        "pagerank" => {
            match algorithms::pagerank(graph, 0.85, Some(10)) {
                Ok(ranks) => {
                    let max_rank = ranks.iter().fold(0.0f64, |a, &b| a.max(b));
                    format!("Max PageRank: {:.6}", max_rank)
                }
                Err(e) => return None,
            }
        }
        "degree_distribution" => {
            let degrees: Vec<usize> = (0..graph.node_count())
                .map(|i| graph.degree(i))
                .collect();
            let max_degree = degrees.iter().max().unwrap_or(&0);
            format!("Max degree: {}", max_degree)
        }
        "clustering_coefficient" => {
            // Sample-based for large graphs
            use rand::prelude::*;
            let mut rng = thread_rng();
            let mut sum = 0.0;
            for _ in 0..config.sample_size.min(graph.node_count()) {
                let node = rng.gen_range(0..graph.node_count());
                if let Ok(cc) = measures::local_clustering_coefficient(graph, node) {
                    sum += cc;
                }
            }
            format!("Avg clustering (sampled): {:.4}", sum / config.sample_size as f64)
        }
        _ => return None,
    };
    
    let execution_time = start.elapsed();
    let post_memory = memory::get_current_memory_mb();
    
    Some(AlgorithmResult {
        algorithm: algorithm.to_string(),
        graph_size: graph.node_count(),
        execution_time_ms: execution_time.as_millis(),
        memory_delta_mb: post_memory - pre_memory,
        result_summary,
        parallel_speedup: None, // Could be calculated if parallel version exists
    })
}

/// Print stress test report
fn print_stress_test_report(results: &StressTestResults) {
    println!("\n========== STRESS TEST REPORT ==========");
    println!("Total runtime: {:.2}s", results.summary.total_runtime_seconds);
    println!("Largest graph tested: {} nodes", results.summary.largest_graph_tested);
    
    println!("\n--- Graph Generation Performance ---");
    for gen in &results.graph_generation {
        println!("  {} nodes, {} edges: {:.2}s, {:.1}MB",
                 gen.nodes, gen.edges,
                 gen.generation_time_ms as f64 / 1000.0,
                 gen.memory_used_mb);
    }
    
    println!("\n--- Algorithm Performance ---");
    for algo_name in &results.config.algorithms {
        println!("\n  {}:", algo_name);
        for result in results.algorithm_performance.iter()
            .filter(|r| r.algorithm == *algo_name) {
            println!("    {} nodes: {:.3}s, {:.1}MB delta | {}",
                     result.graph_size,
                     result.execution_time_ms as f64 / 1000.0,
                     result.memory_delta_mb,
                     result.result_summary);
        }
    }
    
    println!("\n--- Memory Profile ---");
    println!("  Peak memory: {:.1}MB", results.memory_profile.peak_memory_mb);
    println!("  Average memory: {:.1}MB", results.memory_profile.average_memory_mb);
    println!("  Efficiency ratio: {:.2}", results.memory_profile.memory_efficiency_ratio);
    
    if !results.summary.failures.is_empty() {
        println!("\n--- Failures ---");
        for failure in &results.summary.failures {
            println!("  ❌ {}", failure);
        }
    }
    
    if !results.summary.warnings.is_empty() {
        println!("\n--- Warnings ---");
        for warning in &results.summary.warnings {
            println!("  ⚠️  {}", warning);
        }
    }
    
    println!("\n=====================================");
}

/// Save results to JSON file
fn save_results(results: &StressTestResults, filename: &str) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(results)?;
    std::fs::write(filename, json)?;
    Ok(())
}

// Criterion benchmark groups
criterion_group!(
    benches,
    bench_graph_generation,
    bench_large_algorithms,
    bench_streaming_operations
);
criterion_main!(benches);

// Standalone stress test runner
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[ignore] // Run with: cargo test --release -- --ignored stress_test_million_nodes
    fn stress_test_million_nodes() {
        let config = StressTestConfig {
            node_counts: vec![100_000, 500_000, 1_000_000],
            edge_densities: vec![0.00001],
            algorithms: vec![
                "bfs".to_string(),
                "connected_components".to_string(),
                "pagerank".to_string(),
                "degree_distribution".to_string(),
            ],
            memory_limit_mb: 8192,
            timeout_seconds: 600,
            sample_size: 1000,
            parallel: true,
        };
        
        let results = run_stress_test_suite(config);
        print_stress_test_report(&results);
        
        // Save results
        let _ = save_results(&results, "stress_test_results.json");
    }
    
    #[test]
    #[ignore]
    fn stress_test_five_million_nodes() {
        let config = StressTestConfig {
            node_counts: vec![1_000_000, 2_000_000, 5_000_000],
            edge_densities: vec![0.000001], // Very sparse
            algorithms: vec![
                "degree_distribution".to_string(),
                "clustering_coefficient".to_string(),
            ],
            memory_limit_mb: 32768, // 32GB
            timeout_seconds: 1800,  // 30 minutes
            sample_size: 10000,
            parallel: true,
        };
        
        let results = run_stress_test_suite(config);
        print_stress_test_report(&results);
        
        // Save results
        let _ = save_results(&results, "stress_test_5m_results.json");
    }
}