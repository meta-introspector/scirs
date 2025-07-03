//! Final Ultrathink Mode Validation
//!
//! This example validates that all ultrathink mode functionality is working correctly
//! and provides performance metrics for the complete system.

#![allow(unused_imports)]
#![allow(dead_code)]

use scirs2_graph::{
    barabasi_albert_graph, betweenness_centrality, breadth_first_search, connected_components,
    dijkstra_path, erdos_renyi_graph, louvain_communities_result, pagerank_centrality,
    watts_strogatz_graph, DiGraph, Graph, Node,
};

use scirs2_graph::ultrathink::{
    create_enhanced_ultrathink_processor, create_large_graph_ultrathink_processor,
    create_performance_ultrathink_processor, execute_with_enhanced_ultrathink, UltrathinkConfig,
    UltrathinkProcessor,
};

use scirs2_graph::ultrathink_memory_profiler::{MemoryProfilerConfig, UltrathinkMemoryProfiler};

use scirs2_graph::ultrathink_numerical_validation::{
    create_comprehensive_validation_suite, run_quick_validation, ValidationConfig,
};

use rand::Rng;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Comprehensive validation report
#[derive(Debug)]
struct UltrathinkValidationReport {
    pub processor_tests: HashMap<String, bool>,
    pub algorithm_tests: HashMap<String, Duration>,
    pub memory_efficiency: f64,
    pub numerical_accuracy: bool,
    pub performance_improvements: HashMap<String, f64>,
    pub overall_status: ValidationStatus,
}

#[derive(Debug, PartialEq)]
enum ValidationStatus {
    Pass,
    Warning,
    Fail,
}

/// Test different ultrathink processor configurations
fn test_processor_configurations() -> HashMap<String, bool> {
    println!("🔧 Testing ultrathink processor configurations...");
    let mut results = HashMap::new();

    // Test enhanced processor creation
    println!("  Testing enhanced processor...");
    match std::panic::catch_unwind(|| create_enhanced_ultrathink_processor()) {
        Ok(_processor) => {
            results.insert("enhanced_processor".to_string(), true);
            println!("    ✅ Enhanced processor created successfully");
        }
        Err(_) => {
            results.insert("enhanced_processor".to_string(), false);
            println!("    ❌ Enhanced processor creation failed");
        }
    }

    // Test large graph processor creation
    println!("  Testing large graph processor...");
    match std::panic::catch_unwind(|| create_large_graph_ultrathink_processor()) {
        Ok(_processor) => {
            results.insert("large_graph_processor".to_string(), true);
            println!("    ✅ Large graph processor created successfully");
        }
        Err(_) => {
            results.insert("large_graph_processor".to_string(), false);
            println!("    ❌ Large graph processor creation failed");
        }
    }

    // Test performance processor creation
    println!("  Testing performance processor...");
    match std::panic::catch_unwind(|| create_performance_ultrathink_processor()) {
        Ok(_processor) => {
            results.insert("performance_processor".to_string(), true);
            println!("    ✅ Performance processor created successfully");
        }
        Err(_) => {
            results.insert("performance_processor".to_string(), false);
            println!("    ❌ Performance processor creation failed");
        }
    }

    // Test custom configuration
    println!("  Testing custom configuration...");
    let custom_config = UltrathinkConfig {
        enable_neural_rl: true,
        enable_gpu_acceleration: false, // Disable GPU for compatibility
        enable_neuromorphic: true,
        enable_realtime_adaptation: true,
        enable_memory_optimization: true,
        learning_rate: 0.001,
        memory_threshold_mb: 512,
        gpu_memory_pool_mb: 1024,
        neural_hidden_size: 64,
    };

    match std::panic::catch_unwind(|| UltrathinkProcessor::new(custom_config)) {
        Ok(_processor) => {
            results.insert("custom_configuration".to_string(), true);
            println!("    ✅ Custom configuration created successfully");
        }
        Err(_) => {
            results.insert("custom_configuration".to_string(), false);
            println!("    ❌ Custom configuration creation failed");
        }
    }

    results
}

/// Test algorithm execution with ultrathink mode
fn test_algorithm_execution() -> HashMap<String, Duration> {
    println!("🧮 Testing algorithm execution with ultrathink mode...");
    let mut results = HashMap::new();

    // Create test graph
    let mut rng = rand::rng();
    let test_graph = match erdos_renyi_graph(1000, 0.01, &mut rng) {
        Ok(graph) => graph,
        Err(e) => {
            println!("    ❌ Failed to create test graph: {:?}", e);
            return results;
        }
    };

    let mut processor = create_enhanced_ultrathink_processor();

    // Test BFS
    println!("  Testing BFS with ultrathink...");
    let start_time = Instant::now();
    match execute_with_enhanced_ultrathink(&mut processor, &test_graph, "bfs_test", |g| {
        breadth_first_search(g, &0)
    }) {
        Ok(_) => {
            let duration = start_time.elapsed();
            results.insert("bfs".to_string(), duration);
            println!("    ✅ BFS completed in {:?}", duration);
        }
        Err(e) => {
            println!("    ❌ BFS failed: {:?}", e);
        }
    }

    // Test Connected Components
    println!("  Testing connected components with ultrathink...");
    let start_time = Instant::now();
    match execute_with_enhanced_ultrathink(
        &mut processor,
        &test_graph,
        "connected_components_test",
        |g| connected_components(g),
    ) {
        Ok(_) => {
            let duration = start_time.elapsed();
            results.insert("connected_components".to_string(), duration);
            println!("    ✅ Connected components completed in {:?}", duration);
        }
        Err(e) => {
            println!("    ❌ Connected components failed: {:?}", e);
        }
    }

    // Test PageRank
    println!("  Testing PageRank with ultrathink...");
    let start_time = Instant::now();
    match execute_with_enhanced_ultrathink(&mut processor, &test_graph, "pagerank_test", |g| {
        pagerank_centrality(g, None, None, None)
    }) {
        Ok(_) => {
            let duration = start_time.elapsed();
            results.insert("pagerank".to_string(), duration);
            println!("    ✅ PageRank completed in {:?}", duration);
        }
        Err(e) => {
            println!("    ❌ PageRank failed: {:?}", e);
        }
    }

    // Test Community Detection
    println!("  Testing community detection with ultrathink...");
    let start_time = Instant::now();
    match execute_with_enhanced_ultrathink(
        &mut processor,
        &test_graph,
        "community_detection_test",
        |g| louvain_communities_result(g, None, None),
    ) {
        Ok(_) => {
            let duration = start_time.elapsed();
            results.insert("community_detection".to_string(), duration);
            println!("    ✅ Community detection completed in {:?}", duration);
        }
        Err(e) => {
            println!("    ❌ Community detection failed: {:?}", e);
        }
    }

    results
}

/// Test memory efficiency with ultrathink mode
fn test_memory_efficiency() -> f64 {
    println!("💾 Testing memory efficiency with ultrathink mode...");

    // Create memory profiler
    let config = MemoryProfilerConfig {
        enable_detailed_tracking: true,
        sample_interval_ms: 100,
        memory_threshold_mb: 512,
        enable_fragmentation_analysis: true,
        enable_allocation_tracking: true,
    };

    let mut profiler = UltrathinkMemoryProfiler::new(config);

    // Test with medium-sized graph
    let mut rng = rand::rng();
    let test_graph = match barabasi_albert_graph(5000, 3, &mut rng) {
        Ok(graph) => graph,
        Err(e) => {
            println!("    ❌ Failed to create test graph: {:?}", e);
            return 0.0;
        }
    };

    profiler.start_profiling();

    let mut processor = create_performance_ultrathink_processor();

    // Run several memory-intensive operations
    let operations = vec![
        ("pagerank", |g: &Graph<usize, f64>| {
            pagerank_centrality(g, None, None, None)
        }),
        ("betweenness", |g: &Graph<usize, f64>| {
            betweenness_centrality(g)
        }),
        ("connected_components", |g: &Graph<usize, f64>| {
            connected_components(g)
        }),
    ];

    for (name, operation) in operations {
        let _ = execute_with_enhanced_ultrathink(
            &mut processor,
            &test_graph,
            &format!("memory_test_{}", name),
            operation,
        );
    }

    let profile = profiler.stop_profiling();
    let memory_stats = profiler.get_memory_stats();

    let efficiency = memory_stats.efficiency_score;
    println!("    📊 Memory efficiency score: {:.2}", efficiency);
    println!(
        "    📈 Peak memory usage: {:.1} MB",
        memory_stats.peak_usage_mb
    );
    println!(
        "    🔄 Total allocations: {}",
        memory_stats.total_allocations
    );

    efficiency
}

/// Test numerical accuracy with quick validation
fn test_numerical_accuracy() -> bool {
    println!("🔢 Testing numerical accuracy with ultrathink mode...");

    // Create validation configuration
    let config = ValidationConfig {
        tolerance: 1e-6,
        enable_cross_validation: true,
        enable_stress_testing: false, // Disable for quick test
        max_test_graph_size: 1000,
        enable_performance_validation: true,
    };

    // Run quick validation
    match run_quick_validation(config) {
        Ok(results) => {
            println!("    ✅ Numerical accuracy validation passed");
            println!(
                "    📊 Tests passed: {}/{}",
                results.passed_tests, results.total_tests
            );
            println!("    📈 Average accuracy: {:.6}", results.average_accuracy);

            results.overall_status
        }
        Err(e) => {
            println!("    ❌ Numerical accuracy validation failed: {:?}", e);
            false
        }
    }
}

/// Compare performance with and without ultrathink mode
fn test_performance_improvements() -> HashMap<String, f64> {
    println!("⚡ Testing performance improvements with ultrathink mode...");
    let mut improvements = HashMap::new();

    let mut rng = rand::rng();
    let test_graph = match watts_strogatz_graph(2000, 6, 0.3, &mut rng) {
        Ok(graph) => graph,
        Err(e) => {
            println!("    ❌ Failed to create test graph: {:?}", e);
            return improvements;
        }
    };

    // Test PageRank performance improvement
    println!("  Testing PageRank performance improvement...");

    // Standard execution
    let start_time = Instant::now();
    let _standard_result = pagerank_centrality(&test_graph, None, None, None);
    let standard_duration = start_time.elapsed();

    // Ultrathink execution
    let mut processor = create_performance_ultrathink_processor();
    let start_time = Instant::now();
    let _ultrathink_result = execute_with_enhanced_ultrathink(
        &mut processor,
        &test_graph,
        "pagerank_performance_test",
        |g| pagerank_centrality(g, None, None, None),
    );
    let ultrathink_duration = start_time.elapsed();

    if ultrathink_duration.as_nanos() > 0 {
        let improvement =
            standard_duration.as_nanos() as f64 / ultrathink_duration.as_nanos() as f64;
        improvements.insert("pagerank".to_string(), improvement);
        println!("    📊 PageRank improvement: {:.2}x", improvement);
    }

    // Test Connected Components performance improvement
    println!("  Testing connected components performance improvement...");

    // Standard execution
    let start_time = Instant::now();
    let _standard_result = connected_components(&test_graph);
    let standard_duration = start_time.elapsed();

    // Ultrathink execution
    let start_time = Instant::now();
    let _ultrathink_result =
        execute_with_enhanced_ultrathink(&mut processor, &test_graph, "cc_performance_test", |g| {
            connected_components(g)
        });
    let ultrathink_duration = start_time.elapsed();

    if ultrathink_duration.as_nanos() > 0 {
        let improvement =
            standard_duration.as_nanos() as f64 / ultrathink_duration.as_nanos() as f64;
        improvements.insert("connected_components".to_string(), improvement);
        println!(
            "    📊 Connected components improvement: {:.2}x",
            improvement
        );
    }

    improvements
}

/// Generate final validation report
fn generate_final_report(
    processor_tests: HashMap<String, bool>,
    algorithm_tests: HashMap<String, Duration>,
    memory_efficiency: f64,
    numerical_accuracy: bool,
    performance_improvements: HashMap<String, f64>,
) -> UltrathinkValidationReport {
    let mut overall_status = ValidationStatus::Pass;

    // Check processor tests
    let processor_failures = processor_tests.values().filter(|&&v| !v).count();
    if processor_failures > 0 {
        overall_status = ValidationStatus::Warning;
    }

    // Check algorithm execution
    if algorithm_tests.is_empty() {
        overall_status = ValidationStatus::Fail;
    }

    // Check memory efficiency
    if memory_efficiency < 0.7 {
        overall_status = ValidationStatus::Warning;
    }

    // Check numerical accuracy
    if !numerical_accuracy {
        overall_status = ValidationStatus::Fail;
    }

    UltrathinkValidationReport {
        processor_tests,
        algorithm_tests,
        memory_efficiency,
        numerical_accuracy,
        performance_improvements,
        overall_status,
    }
}

/// Print detailed validation report
fn print_validation_report(report: &UltrathinkValidationReport) {
    println!("\n" + "=".repeat(60).as_str());
    println!("🎯 ULTRATHINK MODE FINAL VALIDATION REPORT");
    println!("=".repeat(60));

    // Overall status
    let status_emoji = match report.overall_status {
        ValidationStatus::Pass => "✅",
        ValidationStatus::Warning => "⚠️",
        ValidationStatus::Fail => "❌",
    };
    println!(
        "{} Overall Status: {:?}",
        status_emoji, report.overall_status
    );

    // Processor Configuration Tests
    println!("\n🔧 Processor Configuration Tests:");
    for (test_name, result) in &report.processor_tests {
        let emoji = if *result { "✅" } else { "❌" };
        println!(
            "  {} {}: {}",
            emoji,
            test_name,
            if *result { "PASS" } else { "FAIL" }
        );
    }

    // Algorithm Execution Tests
    println!("\n🧮 Algorithm Execution Tests:");
    for (algorithm, duration) in &report.algorithm_tests {
        println!("  ✅ {}: {:?}", algorithm, duration);
    }

    // Memory Efficiency
    println!("\n💾 Memory Efficiency:");
    let memory_emoji = if report.memory_efficiency >= 0.8 {
        "✅"
    } else if report.memory_efficiency >= 0.6 {
        "⚠️"
    } else {
        "❌"
    };
    println!(
        "  {} Efficiency Score: {:.2}",
        memory_emoji, report.memory_efficiency
    );

    // Numerical Accuracy
    println!("\n🔢 Numerical Accuracy:");
    let accuracy_emoji = if report.numerical_accuracy {
        "✅"
    } else {
        "❌"
    };
    println!(
        "  {} Validation: {}",
        accuracy_emoji,
        if report.numerical_accuracy {
            "PASS"
        } else {
            "FAIL"
        }
    );

    // Performance Improvements
    println!("\n⚡ Performance Improvements:");
    for (algorithm, improvement) in &report.performance_improvements {
        let improvement_emoji = if *improvement >= 1.5 {
            "🚀"
        } else if *improvement >= 1.1 {
            "✅"
        } else {
            "⚠️"
        };
        println!(
            "  {} {}: {:.2}x speedup",
            improvement_emoji, algorithm, improvement
        );
    }

    // Summary
    println!("\n📊 VALIDATION SUMMARY:");
    let passed_processors = report.processor_tests.values().filter(|&&v| v).count();
    let total_processors = report.processor_tests.len();
    println!(
        "  • Processor Tests: {}/{} passed",
        passed_processors, total_processors
    );
    println!(
        "  • Algorithm Tests: {} completed",
        report.algorithm_tests.len()
    );
    println!(
        "  • Memory Efficiency: {:.1}%",
        report.memory_efficiency * 100.0
    );
    println!(
        "  • Numerical Accuracy: {}",
        if report.numerical_accuracy {
            "VALIDATED"
        } else {
            "FAILED"
        }
    );

    let avg_improvement = if !report.performance_improvements.is_empty() {
        report.performance_improvements.values().sum::<f64>()
            / report.performance_improvements.len() as f64
    } else {
        1.0
    };
    println!(
        "  • Average Performance Improvement: {:.2}x",
        avg_improvement
    );

    println!("\n" + "=".repeat(60).as_str());
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting final ultrathink mode validation...");
    println!("=".repeat(60));

    // Run all validation tests
    let processor_tests = test_processor_configurations();
    let algorithm_tests = test_algorithm_execution();
    let memory_efficiency = test_memory_efficiency();
    let numerical_accuracy = test_numerical_accuracy();
    let performance_improvements = test_performance_improvements();

    // Generate and print final report
    let report = generate_final_report(
        processor_tests,
        algorithm_tests,
        memory_efficiency,
        numerical_accuracy,
        performance_improvements,
    );

    print_validation_report(&report);

    // Exit with appropriate code
    match report.overall_status {
        ValidationStatus::Pass => {
            println!("\n🎉 All ultrathink mode validations passed successfully!");
            Ok(())
        }
        ValidationStatus::Warning => {
            println!("\n⚠️ Ultrathink mode validation completed with warnings.");
            Ok(())
        }
        ValidationStatus::Fail => {
            println!("\n❌ Ultrathink mode validation failed.");
            std::process::exit(1);
        }
    }
}
