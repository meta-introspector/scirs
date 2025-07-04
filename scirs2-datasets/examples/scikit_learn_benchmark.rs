//! Comprehensive benchmarks comparing SciRS2 performance against scikit-learn
//!
//! This example runs extensive benchmarks to measure the performance of SciRS2
//! dataset operations compared to scikit-learn equivalents.
//!
//! Usage:
//!   cargo run --example scikit_learn_benchmark --release
//!
//! Note: This requires scikit-learn to be installed for Python comparison benchmarks

use scirs2_datasets::{
    benchmarks::{BenchmarkRunner, BenchmarkSuite, PerformanceComparison},
    load_boston, load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine, make_blobs,
    make_classification, make_regression,
    utils::{stratified_kfold_split, train_test_split},
};
use std::collections::HashMap;
use std::process::Command;
use std::time::{Duration, Instant};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 SciRS2 vs Scikit-Learn Performance Benchmarks");
    println!("================================================\n");

    let runner = BenchmarkRunner::new()
        .with_iterations(5)
        .with_warmup(2)
        .with_memory_measurement(false);

    // Run comprehensive SciRS2 benchmarks
    let scirs2_suites = runner.run_comprehensive_benchmarks();

    println!("\n" + "=".repeat(60).as_str());
    println!("DETAILED ANALYSIS");
    println!("=".repeat(60));

    // Analyze toy dataset performance
    analyze_toy_dataset_performance(&scirs2_suites);

    // Analyze data generation performance
    analyze_data_generation_performance(&scirs2_suites);

    // Run Python comparison benchmarks (if available)
    run_python_comparison_benchmarks();

    // Generate performance report
    generate_performance_report(&scirs2_suites);

    println!("\n🎉 Benchmark suite completed successfully!");
    println!("Check the generated performance report for detailed analysis.");

    Ok(())
}

#[allow(dead_code)]
fn analyze_toy_dataset_performance(suites: &[BenchmarkSuite]) {
    if let Some(toy_suite) = suites.iter().find(|s| s.name == "Toy Datasets") {
        println!("\n📊 TOY DATASET LOADING ANALYSIS");
        println!("-".repeat(40));

        let mut total_loading_time = Duration::ZERO;
        let mut total_samples = 0;
        let mut fastest_dataset = ("", Duration::MAX);
        let mut slowest_dataset = ("", Duration::ZERO);

        for result in toy_suite.successful_results() {
            total_loading_time += result.duration;
            total_samples += result.samples;

            if result.duration < fastest_dataset.1 {
                fastest_dataset = (&result.operation, result.duration);
            }
            if result.duration > slowest_dataset.1 {
                slowest_dataset = (&result.operation, result.duration);
            }

            println!(
                "  {}: {} ({} samples, {:.1} samples/s)",
                result.operation.replace("load_", ""),
                result.formatted_duration(),
                result.samples,
                result.throughput
            );
        }

        println!("\n  Summary:");
        println!(
            "    Total loading time: {:.2}s",
            total_loading_time.as_secs_f64()
        );
        println!("    Total samples loaded: {}", total_samples);
        println!(
            "    Average throughput: {:.1} samples/s",
            total_samples as f64 / total_loading_time.as_secs_f64()
        );
        println!(
            "    Fastest: {} ({})",
            fastest_dataset.0,
            format_duration(fastest_dataset.1)
        );
        println!(
            "    Slowest: {} ({})",
            slowest_dataset.0,
            format_duration(slowest_dataset.1)
        );
    }
}

#[allow(dead_code)]
fn analyze_data_generation_performance(suites: &[BenchmarkSuite]) {
    if let Some(gen_suite) = suites.iter().find(|s| s.name == "Data Generation") {
        println!("\n🔬 DATA GENERATION ANALYSIS");
        println!("-".repeat(40));

        let mut classification_results = Vec::new();
        let mut regression_results = Vec::new();
        let mut clustering_results = Vec::new();

        for result in gen_suite.successful_results() {
            if result.operation.contains("classification") {
                classification_results.push(result);
            } else if result.operation.contains("regression") {
                regression_results.push(result);
            } else if result.operation.contains("blobs") {
                clustering_results.push(result);
            }
        }

        analyze_generation_type("Classification", &classification_results);
        analyze_generation_type("Regression", &regression_results);
        analyze_generation_type("Clustering", &clustering_results);

        // Performance scaling analysis
        analyze_scaling_performance(gen_suite);
    }
}

#[allow(dead_code)]
fn analyze_generation_type(
    gen_type: &str,
    results: &[&scirs2_datasets::benchmarks::BenchmarkResult],
) {
    if results.is_empty() {
        return;
    }

    println!("\n  {} Generation:", gen_type);

    let total_samples: usize = results.iter().map(|r| r.samples).sum();
    let total_duration: Duration = results.iter().map(|r| r.duration).sum();
    let avg_throughput = total_samples as f64 / total_duration.as_secs_f64();

    println!("    Configurations tested: {}", results.len());
    println!("    Total samples generated: {}", total_samples);
    println!("    Average throughput: {:.1} samples/s", avg_throughput);

    // Find best and worst performance
    let best = results
        .iter()
        .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap());
    let worst = results
        .iter()
        .min_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap());

    if let (Some(best), Some(worst)) = (best, worst) {
        println!(
            "    Best: {} ({:.1} samples/s)",
            best.operation.split('_').last().unwrap_or("unknown"),
            best.throughput
        );
        println!(
            "    Worst: {} ({:.1} samples/s)",
            worst.operation.split('_').last().unwrap_or("unknown"),
            worst.throughput
        );
    }
}

#[allow(dead_code)]
fn analyze_scaling_performance(suite: &BenchmarkSuite) {
    println!("\n  📈 SCALING ANALYSIS:");

    // Group results by sample size
    let mut size_groups: HashMap<usize, Vec<_>> = HashMap::new();

    for result in suite.successful_results() {
        size_groups.entry(result.samples).or_default().push(result);
    }

    let mut sizes: Vec<_> = size_groups.keys().collect();
    sizes.sort();

    for &size in &sizes {
        if let Some(results) = size_groups.get(size) {
            let avg_throughput =
                results.iter().map(|r| r.throughput).sum::<f64>() / results.len() as f64;
            let avg_duration = results
                .iter()
                .map(|r| r.duration.as_secs_f64())
                .sum::<f64>()
                / results.len() as f64;

            println!(
                "    {} samples: {:.1} samples/s (avg {:.2}s)",
                size, avg_throughput, avg_duration
            );
        }
    }

    // Calculate scaling efficiency
    if sizes.len() >= 2 {
        let small_size = sizes[0];
        let large_size = sizes[sizes.len() - 1];

        if let (Some(small_results), Some(large_results)) =
            (size_groups.get(small_size), size_groups.get(large_size))
        {
            let small_avg = small_results.iter().map(|r| r.throughput).sum::<f64>()
                / small_results.len() as f64;
            let large_avg = large_results.iter().map(|r| r.throughput).sum::<f64>()
                / large_results.len() as f64;

            let efficiency = large_avg / small_avg;
            let size_ratio = *large_size as f64 / *small_size as f64;

            println!(
                "    Scaling efficiency: {:.2}x (size increased {:.1}x)",
                efficiency, size_ratio
            );

            if efficiency > 0.8 {
                println!("    ✅ Good scaling performance");
            } else if efficiency > 0.5 {
                println!("    ⚠️ Moderate scaling performance");
            } else {
                println!("    ❌ Poor scaling performance");
            }
        }
    }
}

#[allow(dead_code)]
fn run_python_comparison_benchmarks() {
    println!("\n🐍 PYTHON SCIKIT-LEARN COMPARISON");
    println!("-".repeat(40));

    // Check if Python and scikit-learn are available
    let python_check = Command::new("python3")
        .arg("-c")
        .arg("import sklearn; print('scikit-learn', sklearn.__version__)")
        .output();

    match python_check {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout);
            println!("  ✅ Found {}", version.trim());

            // Run comparative benchmarks
            run_sklearn_toy_dataset_comparison();
            run_sklearn_generation_comparison();
        }
        _ => {
            println!("  ❌ Python scikit-learn not available");
            println!("  Install with: pip install scikit-learn");
            println!("  Skipping Python comparison benchmarks");
        }
    }
}

#[allow(dead_code)]
fn run_sklearn_toy_dataset_comparison() {
    println!("\n  📊 Toy Dataset Loading Comparison:");

    let datasets = vec![
        (
            "iris",
            "from sklearn.datasets import load_iris; load_iris()",
        ),
        (
            "boston",
            "from sklearn.datasets import load_boston; load_boston()",
        ),
        (
            "digits",
            "from sklearn.datasets import load_digits; load_digits()",
        ),
        (
            "wine",
            "from sklearn.datasets import load_wine; load_wine()",
        ),
        (
            "breast_cancer",
            "from sklearn.datasets import load_breast_cancer; load_breast_cancer()",
        ),
    ];

    for (name, python_code) in datasets {
        // Time Python execution
        let start = Instant::now();
        let python_result = Command::new("python3")
            .arg("-c")
            .arg(&format!(
                "import time; start=time.time(); {}; print(f'{{:.4f}}', time.time()-start)",
                python_code
            ))
            .output();

        match python_result {
            Ok(output) if output.status.success() => {
                let python_time = String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .parse::<f64>()
                    .unwrap_or(0.0);

                // Time SciRS2 execution
                let scirs2_start = Instant::now();
                let _scirs2_result = match name {
                    "iris" => load_iris().map(|_| ()),
                    "boston" => load_boston().map(|_| ()),
                    "digits" => load_digits().map(|_| ()),
                    "wine" => load_wine().map(|_| ()),
                    "breast_cancer" => load_breast_cancer().map(|_| ()),
                    _ => Ok(()),
                };
                let scirs2_time = scirs2_start.elapsed().as_secs_f64();

                let speedup = python_time / scirs2_time;
                let status = if speedup > 1.2 {
                    "🚀 FASTER"
                } else if speedup > 0.8 {
                    "≈ SIMILAR"
                } else {
                    "🐌 SLOWER"
                };

                println!(
                    "    {}: SciRS2 {:.2}ms vs sklearn {:.2}ms ({:.1}x {}",
                    name,
                    scirs2_time * 1000.0,
                    python_time * 1000.0,
                    speedup,
                    status
                );
            }
            _ => {
                println!("    {}: Failed to benchmark Python version", name);
            }
        }
    }
}

#[allow(dead_code)]
fn run_sklearn_generation_comparison() {
    println!("\n  🔬 Data Generation Comparison:");

    let configs = vec![
        (1000, 10, "classification"),
        (5000, 20, "classification"),
        (1000, 10, "regression"),
        (5000, 20, "regression"),
    ];

    for (n_samples, n_features, gen_type) in configs {
        let (python_code, scirs2_fn): (&str, Box<dyn Fn() -> Result<_, _>>) = match gen_type {
            "classification" => (
                &format!("from sklearn.datasets import make_classification; make_classification(n_samples={}, n_features={}, random_state=42)", n_samples, n_features),
                Box::new(move || make_classification(n_samples, n_features, 3, 2, 4, Some(42)))
            ),
            "regression" => (
                &format!("from sklearn.datasets import make_regression; make_regression(n_samples={}, n_features={}, random_state=42)", n_samples, n_features),
                Box::new(move || make_regression(n_samples, n_features, 3, 0.1, Some(42)))
            ),
            _ => continue,
        };

        // Time Python execution
        let python_result = Command::new("python3")
            .arg("-c")
            .arg(&format!(
                "import time; start=time.time(); {}; print(f'{{:.4f}}', time.time()-start)",
                python_code
            ))
            .output();

        match python_result {
            Ok(output) if output.status.success() => {
                let python_time = String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .parse::<f64>()
                    .unwrap_or(0.0);

                // Time SciRS2 execution
                let scirs2_start = Instant::now();
                let _scirs2_result = scirs2_fn();
                let scirs2_time = scirs2_start.elapsed().as_secs_f64();

                let speedup = python_time / scirs2_time;
                let status = if speedup > 1.2 {
                    "🚀 FASTER"
                } else if speedup > 0.8 {
                    "≈ SIMILAR"
                } else {
                    "🐌 SLOWER"
                };

                println!(
                    "    {} {}x{}: SciRS2 {:.2}ms vs sklearn {:.2}ms ({:.1}x {})",
                    gen_type,
                    n_samples,
                    n_features,
                    scirs2_time * 1000.0,
                    python_time * 1000.0,
                    speedup,
                    status
                );
            }
            _ => {
                println!(
                    "    {} {}x{}: Failed to benchmark Python version",
                    gen_type, n_samples, n_features
                );
            }
        }
    }
}

#[allow(dead_code)]
fn generate_performance_report(suites: &[BenchmarkSuite]) {
    println!("\n📋 PERFORMANCE SUMMARY REPORT");
    println!("=".repeat(60));

    let mut total_operations = 0;
    let mut total_samples = 0;
    let mut total_duration = Duration::ZERO;

    for suite in suites {
        total_operations += suite.results.len();
        total_samples += suite.total_samples();
        total_duration += suite.total_duration;
    }

    println!("  Total operations benchmarked: {}", total_operations);
    println!("  Total samples processed: {}", total_samples);
    println!(
        "  Total benchmark time: {:.2}s",
        total_duration.as_secs_f64()
    );
    println!(
        "  Overall throughput: {:.1} samples/s",
        total_samples as f64 / total_duration.as_secs_f64()
    );

    // Performance assessment
    let avg_throughput = total_samples as f64 / total_duration.as_secs_f64();

    println!("\n  🎯 PERFORMANCE ASSESSMENT:");
    if avg_throughput > 50000.0 {
        println!("    ⭐ EXCELLENT - High-performance implementation");
    } else if avg_throughput > 10000.0 {
        println!("    ✅ GOOD - Solid performance for scientific computing");
    } else if avg_throughput > 1000.0 {
        println!("    ⚠️ MODERATE - Acceptable for most use cases");
    } else {
        println!("    ❌ SLOW - May need optimization");
    }

    // Recommendations
    println!("\n  💡 RECOMMENDATIONS:");

    if let Some(gen_suite) = suites.iter().find(|s| s.name == "Data Generation") {
        let successful = gen_suite.successful_results();
        let failed = gen_suite.failed_results();

        if !failed.is_empty() {
            println!(
                "    • Fix {} failed data generation operations",
                failed.len()
            );
        }

        if successful.len() > 0 {
            let avg_gen_throughput =
                successful.iter().map(|r| r.throughput).sum::<f64>() / successful.len() as f64;
            if avg_gen_throughput < 1000.0 {
                println!("    • Consider optimizing data generation algorithms");
                println!("    • Implement SIMD operations for numeric computations");
                println!("    • Use parallel processing for large datasets");
            }
        }
    }

    println!("    • Consider GPU acceleration for large-scale operations");
    println!("    • Implement streaming for memory-efficient processing");
    println!("    • Add caching for frequently accessed datasets");
}

#[allow(dead_code)]
fn format_duration(duration: Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if duration.as_millis() > 0 {
        format!("{}ms", duration.as_millis())
    } else {
        format!("{}μs", duration.as_micros())
    }
}
