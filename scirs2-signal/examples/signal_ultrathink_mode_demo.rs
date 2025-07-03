//! Ultrathink Mode Demonstration
//!
//! This example shows how to use the ultrathink mode coordinator for comprehensive
//! validation and performance testing of signal processing implementations.

use scirs2_signal::ultrathink_mode_coordinator::{
    run_quick_ultrathink_validation, UltrathinkConfig, UltrathinkCoordinator,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Ultrathink Mode Signal Processing Demo");
    println!("=========================================\n");

    // Example 1: Quick validation with default settings
    println!("1. Running quick ultrathink validation...");
    match run_quick_ultrathink_validation() {
        Ok(results) => {
            println!("✅ Quick validation completed successfully!");
            println!("   Overall success score: {:.1}/100", results.success_score);
            println!(
                "   SIMD speedup: {:.1}x",
                results.performance_metrics.simd_speedup
            );
            println!(
                "   Parallel speedup: {:.1}x",
                results.performance_metrics.parallel_speedup
            );
            println!(
                "   Memory efficiency: {:.1}%",
                results.performance_metrics.memory_efficiency * 100.0
            );
            println!(
                "   Execution time: {:.1} ms",
                results.performance_metrics.execution_time_ms
            );

            if !results.issues.is_empty() {
                println!("   Issues found:");
                for issue in &results.issues {
                    println!("     - {}", issue);
                }
            }
        }
        Err(e) => {
            println!("❌ Quick validation failed: {}", e);
        }
    }

    println!();

    // Example 2: Custom configuration
    println!("2. Running validation with custom configuration...");
    let custom_config = UltrathinkConfig {
        enable_simd: true,
        enable_parallel: true,
        enable_memory_optimization: true,
        enable_numerical_stability: true,
        enable_validation: true,
        max_threads: Some(4),
        validation_tolerance: 1e-12,
    };

    let mut coordinator = UltrathinkCoordinator::with_config(custom_config);

    match coordinator.run_comprehensive_validation() {
        Ok(results) => {
            println!("✅ Custom validation completed!");
            println!("   Validation Results:");
            println!(
                "     - Multitaper: {}",
                if results.validation_results.multitaper_validation {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!(
                "     - Lomb-Scargle: {}",
                if results.validation_results.lombscargle_validation {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!(
                "     - Parametric: {}",
                if results.validation_results.parametric_validation {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!(
                "     - Wavelet: {}",
                if results.validation_results.wavelet_validation {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!(
                "     - Filter: {}",
                if results.validation_results.filter_validation {
                    "✅"
                } else {
                    "❌"
                }
            );
            println!(
                "   Overall validation score: {:.1}/100",
                results.validation_results.overall_score
            );
        }
        Err(e) => {
            println!("❌ Custom validation failed: {}", e);
        }
    }

    println!();

    // Example 3: Performance comparison
    println!("3. Performance analysis over multiple runs...");
    let mut coordinator = UltrathinkCoordinator::new();

    for i in 1..=3 {
        println!("   Run {}...", i);
        if let Ok(_results) = coordinator.run_comprehensive_validation() {
            println!("   ✅ Run {} completed", i);
        } else {
            println!("   ❌ Run {} failed", i);
        }
    }

    // Generate performance report
    let performance_report = coordinator.generate_performance_report();
    println!("\n   Performance Report:");
    for (metric, value) in performance_report {
        match metric.as_str() {
            "simd_speedup" => println!("     SIMD Speedup: {:.2}x", value),
            "parallel_speedup" => println!("     Parallel Speedup: {:.2}x", value),
            "memory_efficiency" => println!("     Memory Efficiency: {:.1}%", value * 100.0),
            "numerical_stability" => println!("     Numerical Stability: {:.1}%", value * 100.0),
            "execution_time_ms" => println!("     Execution Time: {:.1} ms", value),
            _ => {}
        }
    }

    // Example 4: Memory-constrained validation
    println!("\n4. Testing memory-constrained configuration...");
    let memory_constrained_config = UltrathinkConfig {
        enable_simd: false,
        enable_parallel: false,
        enable_memory_optimization: true,
        enable_numerical_stability: true,
        enable_validation: false, // Skip extensive validation to save memory
        max_threads: Some(1),
        validation_tolerance: 1e-8,
    };

    let mut memory_coordinator = UltrathinkCoordinator::with_config(memory_constrained_config);
    match memory_coordinator.run_comprehensive_validation() {
        Ok(results) => {
            println!("✅ Memory-constrained validation completed!");
            println!(
                "   Memory efficiency: {:.1}%",
                results.performance_metrics.memory_efficiency * 100.0
            );
            println!(
                "   Execution time: {:.1} ms",
                results.performance_metrics.execution_time_ms
            );
        }
        Err(e) => {
            println!("❌ Memory-constrained validation failed: {}", e);
        }
    }

    println!("\n🎉 Ultrathink mode demonstration completed!");
    println!("\nKey Benefits of Ultrathink Mode:");
    println!("  ⚡ Enhanced performance through SIMD and parallel processing");
    println!("  🔍 Comprehensive validation and testing");
    println!("  💾 Memory-efficient algorithms for large datasets");
    println!("  🎯 Numerical stability improvements");
    println!("  📊 Detailed performance metrics and reporting");

    Ok(())
}
