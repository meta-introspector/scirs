//! Advanced Validation Framework Showcase
//!
//! This example demonstrates the comprehensive validation framework for Advanced
//! optimizations. It validates that SIMD, parallel, and other optimizations maintain
//! numerical accuracy while providing performance benefits.

use scirs2_stats::{
    create_custom_ultrathink_validator, create_ultrathink_validator, ValidationConfig,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Advanced Validation Framework Showcase");
    println!("============================================\n");

    // Demonstrate default validation
    demonstrate_default_validation()?;

    // Demonstrate custom validation configuration
    demonstrate_custom_validation()?;

    // Demonstrate detailed analysis
    demonstrate_detailed_analysis()?;

    println!("\n✅ Advanced validation showcase completed successfully!");

    Ok(())
}

/// Demonstrate validation with default configuration
#[allow(dead_code)]
fn demonstrate_default_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Default Validation Configuration");
    println!("-----------------------------------");

    let mut validator = create_ultrathink_validator();

    println!("🚀 Running comprehensive validation tests...");
    let report = validator.validate_all_operations()?;

    println!("\n📋 Validation Summary:");
    println!("  {}", report.summary);
    println!("  Total tests: {}", report.total_tests);
    println!("  Passed: {}", report.passed_tests);
    println!("  Failed: {}", report.failed_tests);
    println!("  Average speedup: {:.2}x", report.average_speedup);
    println!("  Average accuracy: {:.6}", report.average_accuracy);

    // Show detailed results for any failures
    let failed_tests: Vec<_> = report
        .test_results
        .iter()
        .filter(|r| !r.accuracy_passed || !r.performance_passed)
        .collect();

    if !failed_tests.is_empty() {
        println!("\n⚠️  Failed Test Details:");
        for test in failed_tests {
            println!("  {} (size: {}):", test.operation_name, test.data_size);
            println!("    Speedup: {:.2}x", test.speedup_ratio);
            println!("    Accuracy: {:.6}", test.numerical_accuracy);
            for error in &test.error_messages {
                println!("    Error: {}", error);
            }
        }
    }

    Ok(())
}

/// Demonstrate custom validation configuration
#[allow(dead_code)]
fn demonstrate_custom_validation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Custom Validation Configuration");
    println!("----------------------------------");

    // Create custom configuration for high-precision validation
    let custom_config = ValidationConfig {
        numerical_tolerance: 1e-15,        // Higher precision requirement
        benchmark_iterations: 50,          // Fewer iterations for faster testing
        min_performance_improvement: 1.05, // Lower performance threshold (5% improvement)
        verbose_logging: true,
        test_sizes: vec![1000, 10000], // Specific test sizes
    };

    let mut validator = create_custom_ultrathink_validator(custom_config);

    println!("🚀 Running high-precision validation tests...");
    let report = validator.validate_all_operations()?;

    println!("\n📋 High-Precision Validation Summary:");
    println!("  {}", report.summary);

    // Analyze performance trends across data sizes
    analyze_performance_trends(&report);

    Ok(())
}

/// Demonstrate detailed validation analysis
#[allow(dead_code)]
fn demonstrate_detailed_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Detailed Validation Analysis");
    println!("-------------------------------");

    let mut validator = create_ultrathink_validator();
    let report = validator.validate_all_operations()?;

    // Analyze results by operation type
    analyze_by_operation(&report);

    // Analyze results by data size
    analyze_by_data_size(&report);

    // Performance recommendations
    generate_recommendations(&report);

    Ok(())
}

/// Analyze performance trends across different data sizes
#[allow(dead_code)]
fn analyze_performance_trends(report: &scirs2_stats::ValidationReport) {
    println!("\n📈 Performance Trend Analysis:");

    let mut operations: std::collections::HashMap<String, Vec<&scirs2_stats::ValidationResult>> =
        std::collections::HashMap::new();

    for result in &report.test_results {
        operations
            .entry(result.operation_name.clone())
            .or_insert_with(Vec::new)
            .push(result);
    }

    for (operation, results) in operations {
        println!("  {}:", operation);

        let mut sorted_results = results;
        sorted_results.sort_by_key(|r| r.data_size);

        for result in sorted_results {
            let status = if result.accuracy_passed && result.performance_passed {
                "✅"
            } else {
                "❌"
            };
            println!(
                "    Size {:>8}: {:.2}x speedup, {:.6} accuracy {}",
                result.data_size, result.speedup_ratio, result.numerical_accuracy, status
            );
        }
    }
}

/// Analyze validation results by operation type
#[allow(dead_code)]
fn analyze_by_operation(report: &scirs2_stats::ValidationReport) {
    println!("\n🔬 Analysis by Operation Type:");

    let mut operation_stats: std::collections::HashMap<String, (usize, usize, f64, f64)> =
        std::collections::HashMap::new();

    for result in &report.test_results {
        let entry = operation_stats
            .entry(result.operation_name.clone())
            .or_insert((0, 0, 0.0, 0.0));

        entry.0 += 1; // Total tests
        if result.accuracy_passed && result.performance_passed {
            entry.1 += 1; // Passed tests
        }
        entry.2 += result.speedup_ratio; // Sum of speedups
        entry.3 += result.numerical_accuracy; // Sum of accuracies
    }

    for (operation, (total, passed, speedup_sum, accuracy_sum)) in operation_stats {
        let pass_rate = (passed as f64 / total as f64) * 100.0;
        let avg_speedup = speedup_sum / total as f64;
        let avg_accuracy = accuracy_sum / total as f64;

        println!(
            "  {}: {:.1}% pass rate, {:.2}x avg speedup, {:.6} avg accuracy",
            operation, pass_rate, avg_speedup, avg_accuracy
        );
    }
}

/// Analyze validation results by data size
#[allow(dead_code)]
fn analyze_by_data_size(report: &scirs2_stats::ValidationReport) {
    println!("\n📏 Analysis by Data Size:");

    let mut size_stats: std::collections::HashMap<usize, (usize, usize, f64, f64)> =
        std::collections::HashMap::new();

    for result in &report.test_results {
        let entry = size_stats
            .entry(result.data_size)
            .or_insert((0, 0, 0.0, 0.0));

        entry.0 += 1; // Total tests
        if result.accuracy_passed && result.performance_passed {
            entry.1 += 1; // Passed tests
        }
        entry.2 += result.speedup_ratio; // Sum of speedups
        entry.3 += result.numerical_accuracy; // Sum of accuracies
    }

    let mut sizes: Vec<_> = size_stats.keys().cloned().collect();
    sizes.sort();

    for size in sizes {
        if let Some((total, passed, speedup_sum, accuracy_sum)) = size_stats.get(&size) {
            let pass_rate = (*passed as f64 / *total as f64) * 100.0;
            let avg_speedup = speedup_sum / *total as f64;
            let avg_accuracy = accuracy_sum / *total as f64;

            println!(
                "  Size {:>8}: {:.1}% pass rate, {:.2}x avg speedup, {:.6} avg accuracy",
                size, pass_rate, avg_speedup, avg_accuracy
            );
        }
    }
}

/// Generate optimization recommendations based on validation results
#[allow(dead_code)]
fn generate_recommendations(report: &scirs2_stats::ValidationReport) {
    println!("\n💡 Optimization Recommendations:");

    let pass_rate = (report.passed_tests as f64 / report.total_tests as f64) * 100.0;

    if pass_rate >= 95.0 {
        println!("  ✅ Excellent validation results! Advanced optimizations are working well.");
    } else if pass_rate >= 80.0 {
        println!("  ⚠️  Good validation results, but some optimizations may need tuning.");
    } else {
        println!("  ❌ Several validation failures detected. Review optimization strategies.");
    }

    if report.average_speedup < 1.5 {
        println!(
            "  🚀 Consider increasing optimization aggressiveness (current avg speedup: {:.2}x)",
            report.average_speedup
        );
    }

    if report.average_accuracy < 0.999999 {
        println!(
            "  🎯 Review numerical stability (current avg accuracy: {:.6})",
            report.average_accuracy
        );
    }

    // Analyze failures for specific recommendations
    let failed_tests: Vec<_> = report
        .test_results
        .iter()
        .filter(|r| !r.accuracy_passed || !r.performance_passed)
        .collect();

    if !failed_tests.is_empty() {
        println!("  🔧 Specific Issues:");

        let accuracy_failures = failed_tests.iter().filter(|r| !r.accuracy_passed).count();
        let performance_failures = failed_tests
            .iter()
            .filter(|r| !r.performance_passed)
            .count();

        if accuracy_failures > 0 {
            println!(
                "    - {} accuracy failures: Consider adjusting SIMD precision or using higher precision types",
                accuracy_failures
            );
        }

        if performance_failures > 0 {
            println!(
                "    - {} performance failures: Consider adjusting optimization thresholds or parallel processing parameters",
                performance_failures
            );
        }
    }

    println!("\n📊 Validation Metrics Summary:");
    println!("  Total tests conducted: {}", report.total_tests);
    println!("  Overall success rate: {:.1}%", pass_rate);
    println!("  Average performance gain: {:.2}x", report.average_speedup);
    println!(
        "  Average numerical accuracy: {:.6}",
        report.average_accuracy
    );
}
