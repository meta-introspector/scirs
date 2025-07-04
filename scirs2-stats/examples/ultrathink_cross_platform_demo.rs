//! Cross-Platform Advanced Validation Demo
//!
//! This example demonstrates the comprehensive cross-platform validation
//! capabilities that ensure Advanced optimizations work consistently
//! across different platforms, architectures, and system configurations.

use scirs2_stats::{create_cross_platform_validator, CompatibilityRating, CrossPlatformValidator};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 Advanced Cross-Platform Validation");
    println!("=======================================\n");

    // Create the cross-platform validator
    let mut validator = create_cross_platform_validator();

    println!("🔍 Running comprehensive platform validation...");
    println!("This may take a few moments as we test all optimizations.\n");

    // Run the comprehensive validation
    let report = validator.validate_platform()?;

    // Display platform information
    println!("📋 Platform Information");
    println!("-----------------------");
    println!("Architecture: {}", report.platform_info.architecture);
    println!(
        "Operating System: {}",
        report.platform_info.operating_system
    );
    println!("Compiler: {}", report.platform_info.compiler_version);
    println!("Logical Cores: {}", report.platform_info.logical_cores);
    println!("Physical Cores: {}", report.platform_info.physical_cores);
    println!("Memory: {:.1} GB", report.platform_info.memory_gb);
    println!("SIMD Features: {:?}", report.platform_info.simd_features);
    println!(
        "Cache: L1: {} KB, L2: {} KB, L3: {} KB",
        report.platform_info.l1_cache_kb,
        report.platform_info.l2_cache_kb,
        report.platform_info.l3_cache_kb
    );
    println!();

    // Display overall results
    println!("📊 Validation Results");
    println!("---------------------");
    println!("Overall Score: {:.1}%", report.overall_score * 100.0);
    println!("Compatibility Rating: {:?}", report.compatibility_rating);
    println!(
        "Tests Passed: {} / {}",
        report.passed_tests,
        report.test_results.len()
    );
    println!("Tests Failed: {}", report.failed_tests);
    println!();

    // Display performance profile
    println!("⚡ Performance Profile");
    println!("---------------------");
    println!(
        "SIMD Speedup Factor: {:.2}x",
        report.performance_profile.simd_speedup_factor
    );
    println!(
        "Parallel Efficiency: {:.1}%",
        report.performance_profile.parallel_efficiency * 100.0
    );
    println!(
        "Memory Bandwidth: {:.1} GB/s",
        report.performance_profile.memory_bandwidth_gbps
    );
    println!(
        "Cache Efficiency: {:.1}%",
        report.performance_profile.cache_efficiency * 100.0
    );
    println!(
        "Thermal Throttling: {}",
        if report.performance_profile.thermal_throttling_detected {
            "⚠️  Detected"
        } else {
            "✅ None"
        }
    );
    println!(
        "Recommended Mode: {:?}",
        report.performance_profile.recommended_optimization_mode
    );
    println!();

    // Show test results summary by category
    println!("🧪 Test Results by Category");
    println!("---------------------------");

    let mut categories = std::collections::HashMap::new();
    for result in &report.test_results {
        let category = result.function_name.split('_').next().unwrap_or("unknown");
        let entry = categories.entry(category).or_insert((0, 0));
        if result.passed {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }

    for (category, (passed, failed)) in categories {
        let total = passed + failed;
        let success_rate = passed as f64 / total as f64 * 100.0;
        let status = if success_rate == 100.0 {
            "✅"
        } else if success_rate >= 80.0 {
            "⚠️ "
        } else {
            "❌"
        };
        println!(
            "{} {}: {}/{} ({:.0}%)",
            status, category, passed, total, success_rate
        );
    }
    println!();

    // Show failed tests if any
    if report.failed_tests > 0 {
        println!("❌ Failed Tests");
        println!("---------------");
        for result in &report.test_results {
            if !result.passed {
                println!("• {}: {}", result.function_name, result.test_name);
                for warning in &result.warnings {
                    println!("  ⚠️  {}", warning);
                }
            }
        }
        println!();
    }

    // Show warnings if any
    if !report.warnings.is_empty() {
        println!("⚠️  Platform Warnings");
        println!("---------------------");
        for warning in &report.warnings {
            println!("• {}", warning);
        }
        println!();
    }

    // Show platform-specific issues if any
    if !report.platform_specific_issues.is_empty() {
        println!("🔧 Platform-Specific Issues");
        println!("---------------------------");
        for issue in &report.platform_specific_issues {
            println!("• {}", issue);
        }
        println!();
    }

    // Performance recommendations
    println!("💡 Optimization Recommendations");
    println!("-------------------------------");

    match report.compatibility_rating {
        CompatibilityRating::Excellent => {
            println!(
                "✅ Excellent compatibility! All Advanced optimizations should work perfectly."
            );
            println!(
                "• Use OptimizationMode::{:?} for best results",
                report.performance_profile.recommended_optimization_mode
            );
        }
        CompatibilityRating::Good => {
            println!("✅ Good compatibility. Most optimizations work well.");
            println!(
                "• Consider using OptimizationMode::{:?}",
                report.performance_profile.recommended_optimization_mode
            );
            if report.performance_profile.simd_speedup_factor < 2.0 {
                println!("• SIMD performance could be improved - check compiler flags");
            }
        }
        CompatibilityRating::Fair => {
            println!("⚠️  Fair compatibility. Some optimizations may have reduced effectiveness.");
            println!("• Recommended to use OptimizationMode::Balanced or Accuracy");
            println!("• Monitor performance carefully and consider platform-specific tuning");
        }
        CompatibilityRating::Poor => {
            println!("❌ Poor compatibility. Significant issues detected.");
            println!("• Recommended to use OptimizationMode::Accuracy for maximum reliability");
            println!("• Consider updating system libraries or compiler");
        }
        CompatibilityRating::Incompatible => {
            println!(
                "❌ Platform incompatibility detected. Advanced optimizations may not work."
            );
            println!("• Use standard (non-Advanced) statistical functions");
            println!("• Contact support with platform details");
        }
    }

    if report.performance_profile.thermal_throttling_detected {
        println!(
            "• Thermal throttling detected - ensure adequate cooling for sustained performance"
        );
    }

    if report.platform_info.memory_gb < 8.0 {
        println!("• Low memory configuration - consider using smaller dataset chunk sizes");
    }

    if report.performance_profile.parallel_efficiency < 0.7 {
        println!("• Parallel efficiency is low - check for memory bandwidth limitations");
    }

    println!();

    // Show top performing tests
    let mut top_tests: Vec<_> = report.test_results.iter().filter(|r| r.passed).collect();
    top_tests.sort_by(|a, b| {
        b.performance_score
            .partial_cmp(&a.performance_score)
            .unwrap()
    });

    if !top_tests.is_empty() {
        println!("🏆 Top Performing Tests");
        println!("----------------------");
        for result in top_tests.iter().take(5) {
            println!(
                "• {}: {:.1}% performance, {:.4} accuracy",
                result.test_name,
                result.performance_score * 100.0,
                result.numerical_accuracy
            );
        }
        println!();
    }

    // Summary and next steps
    println!("📋 Summary");
    println!("----------");
    match report.compatibility_rating {
        CompatibilityRating::Excellent | CompatibilityRating::Good => {
            println!("✅ Your platform is well-suited for Advanced optimizations!");
            println!("✅ All major statistical operations should perform excellently.");
            println!("✅ You can confidently use performance-focused configurations.");
        }
        CompatibilityRating::Fair => {
            println!("⚠️  Your platform has some limitations but should work adequately.");
            println!("⚠️  Consider balanced optimization modes and monitor performance.");
        }
        CompatibilityRating::Poor | CompatibilityRating::Incompatible => {
            println!("❌ Significant platform compatibility issues detected.");
            println!("❌ Consider system updates or use accuracy-focused modes.");
        }
    }

    println!("\n🎯 Next Steps:");
    println!(
        "1. Use the recommended OptimizationMode::{:?} in your applications",
        report.performance_profile.recommended_optimization_mode
    );
    println!("2. Monitor performance with enable_performance_monitoring: true");
    println!("3. Run this validation periodically after system updates");

    if report.failed_tests > 0 {
        println!("4. Review failed tests and consider filing an issue if problems persist");
    }

    println!("\n🚀 Advanced cross-platform validation complete!");

    Ok(())
}
