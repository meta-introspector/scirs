//! Memory optimization and leak detection demonstration
//!
//! This example showcases the advanced memory optimization capabilities including
//! leak detection, fragmentation analysis, and optimization recommendations.

use ndarray::Array1;
use scirs2_optim::{
    adam::{Adam, AdamConfig},
    benchmarking::memory_optimizer::*,
    error::Result,
    optimizers::Optimizer,
    sgd::{SGDConfig, SGD},
};

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("🔍 SciRS2 Memory Optimization and Leak Detection Demo");
    println!("=====================================================\n");

    // Run memory optimization demonstration
    run_memory_optimization_demo()?;

    // Run memory leak detection
    run_memory_leak_detection()?;

    // Run fragmentation analysis
    run_fragmentation_analysis()?;

    // Run optimization recommendations
    run_optimization_recommendations()?;

    // Run real-time monitoring
    run_realtime_monitoring()?;

    println!("\n✅ Memory optimization demonstration completed!");
    Ok(())
}

/// Demonstrate memory optimization capabilities
#[allow(dead_code)]
fn run_memory_optimization_demo() -> Result<()> {
    println!("🚀 MEMORY OPTIMIZATION DEMONSTRATION");
    println!("====================================");

    // Create memory optimizer with comprehensive configuration
    let config = MemoryOptimizerConfig {
        enable_detailed_tracking: true,
        enable_leak_detection: true,
        enable_pattern_analysis: true,
        sampling_interval_ms: 50,
        max_history_length: 1000,
        leak_growth_threshold: 512.0, // 512 bytes
        fragmentation_threshold: 0.25,
        enable_stack_traces: false, // Disabled for performance
        alert_thresholds: AlertThresholds {
            warning_threshold: 0.75,  // 75%
            critical_threshold: 0.90, // 90%
            allocation_rate_threshold: 500.0,
            fragmentation_threshold: 0.4,
        },
    };

    let mut memory_optimizer = MemoryOptimizer::new(config);

    println!("Starting memory monitoring...");
    memory_optimizer.start_monitoring()?;

    // Simulate optimization workload with various memory patterns
    simulate_optimization_workload(&mut memory_optimizer)?;

    // Generate analysis report
    let analysis_report = memory_optimizer.analyze_and_recommend()?;

    println!("\n📊 MEMORY ANALYSIS RESULTS");
    println!("==========================");
    println!(
        "Memory Efficiency Score: {:.1}%",
        analysis_report.efficiency_score * 100.0
    );

    // Display memory usage summary
    println!("\nMemory Usage Summary:");
    let usage_summary = &analysis_report.memory_usage_summary;
    println!(
        "  Current Usage: {:.2} MB",
        usage_summary.current_usage.used_memory as f64 / 1024.0 / 1024.0
    );
    println!(
        "  Peak Usage: {:.2} MB",
        usage_summary.peak_usage.used_memory as f64 / 1024.0 / 1024.0
    );
    println!("  Usage Trend: {:?}", usage_summary.usage_trend);

    // Display usage by category
    println!("\nMemory Usage by Category:");
    for (category, usage) in &usage_summary.current_usage.by_category {
        println!(
            "  {:?}: {:.2} MB",
            category,
            *usage as f64 / 1024.0 / 1024.0
        );
    }

    Ok(())
}

/// Demonstrate memory leak detection
#[allow(dead_code)]
fn run_memory_leak_detection() -> Result<()> {
    println!("\n🔍 MEMORY LEAK DETECTION");
    println!("========================");

    let config = MemoryOptimizerConfig {
        enable_leak_detection: true,
        leak_growth_threshold: 256.0, // More sensitive for demo
        ..Default::default()
    };

    let mut memory_optimizer = MemoryOptimizer::new(config);
    memory_optimizer.start_monitoring()?;

    println!("Simulating potential memory leak scenarios...");

    // Simulate memory leak patterns
    simulate_memory_leak_patterns(&mut memory_optimizer)?;

    // Analyze for leaks
    let analysis_report = memory_optimizer.analyze_and_recommend()?;

    println!("\n🔬 LEAK DETECTION RESULTS");
    if analysis_report.detected_leaks.is_empty() {
        println!("✅ No memory leaks detected");
    } else {
        println!(
            "⚠️  {} potential memory leak(s) detected:",
            analysis_report.detected_leaks.len()
        );

        for (i, leak) in analysis_report.detected_leaks.iter().enumerate() {
            println!("\n{}. Leak Type: {:?}", i + 1, leak.leak_type);
            println!("   Severity: {:?}", leak.severity);
            println!("   Confidence: {:.1}%", leak.confidence * 100.0);
            println!("   Leak Rate: {:.2} bytes/sec", leak.leak_rate);
            println!("   Suggested Fix: {}", leak.suggested_fix);

            if let Some(source) = &leak.source_location {
                println!("   Source Location: {}", source);
            }
        }
    }

    Ok(())
}

/// Demonstrate fragmentation analysis
#[allow(dead_code)]
fn run_fragmentation_analysis() -> Result<()> {
    println!("\n🧩 FRAGMENTATION ANALYSIS");
    println!("=========================");

    let mut memory_optimizer = MemoryOptimizer::new(MemoryOptimizerConfig::default());
    memory_optimizer.start_monitoring()?;

    // Simulate fragmentation-inducing patterns
    simulate_fragmentation_patterns(&mut memory_optimizer)?;

    let analysis_report = memory_optimizer.analyze_and_recommend()?;
    let fragmentation = &analysis_report.fragmentation_analysis;

    println!("Current Fragmentation Analysis:");
    println!(
        "  External Fragmentation: {:.1}%",
        fragmentation.current_fragmentation.external_fragmentation * 100.0
    );
    println!(
        "  Internal Fragmentation: {:.1}%",
        fragmentation.current_fragmentation.internal_fragmentation * 100.0
    );
    println!(
        "  Largest Free Block: {:.2} MB",
        fragmentation.current_fragmentation.largest_free_block as f64 / 1024.0 / 1024.0
    );
    println!(
        "  Free Block Count: {}",
        fragmentation.current_fragmentation.free_block_count
    );
    println!(
        "  Average Free Block Size: {:.2} KB",
        fragmentation.current_fragmentation.average_free_block_size / 1024.0
    );

    println!(
        "\nFragmentation Trend: {:?}",
        fragmentation.fragmentation_trend
    );

    println!("\nFragmentation Causes:");
    for cause in &fragmentation.causes {
        println!("  - {}", cause);
    }

    println!("\nFragmentation Recommendations:");
    for recommendation in &fragmentation.recommendations {
        println!("  - {}", recommendation);
    }

    Ok(())
}

/// Demonstrate optimization recommendations
#[allow(dead_code)]
fn run_optimization_recommendations() -> Result<()> {
    println!("\n💡 OPTIMIZATION RECOMMENDATIONS");
    println!("===============================");

    let mut memory_optimizer = MemoryOptimizer::new(MemoryOptimizerConfig::default());
    memory_optimizer.start_monitoring()?;

    // Simulate suboptimal memory patterns
    simulate_suboptimal_patterns(&mut memory_optimizer)?;

    let analysis_report = memory_optimizer.analyze_and_recommend()?;

    if analysis_report.optimization_recommendations.is_empty() {
        println!("✅ No optimization recommendations - memory usage is already optimal!");
    } else {
        println!(
            "Found {} optimization opportunities:",
            analysis_report.optimization_recommendations.len()
        );

        for (i, recommendation) in analysis_report
            .optimization_recommendations
            .iter()
            .enumerate()
        {
            println!(
                "\n{}. {} (Priority: {:?})",
                i + 1,
                recommendation.title,
                recommendation.priority
            );
            println!("   Type: {:?}", recommendation.recommendation_type);
            println!("   Description: {}", recommendation.description);

            println!("   Expected Benefits:");
            println!(
                "     - Memory Reduction: {:.1}%",
                recommendation.expected_benefits.memory_reduction_percent
            );
            println!(
                "     - Performance Improvement: {:.1}%",
                recommendation
                    .expected_benefits
                    .performance_improvement_percent
            );
            println!(
                "     - Allocation Reduction: {} allocations",
                recommendation.expected_benefits.allocation_reduction
            );

            println!("   Implementation Effort:");
            println!(
                "     - Development: {:.1} hours",
                recommendation.estimated_effort.development_hours
            );
            println!(
                "     - Testing: {:.1} hours",
                recommendation.estimated_effort.testing_hours
            );
            println!(
                "     - Complexity: {:?}",
                recommendation.estimated_effort.deployment_complexity
            );

            println!(
                "   Risk Assessment: {:?}",
                recommendation.risk_assessment.risk_level
            );

            if !recommendation.code_examples.is_empty() {
                println!("   Code Example:");
                let example = &recommendation.code_examples[0];
                println!("     Before: {}", example.before_code);
                println!("     After:  {}", example.after_code);
                println!("     Explanation: {}", example.explanation);
            }
        }
    }

    // Display cost-benefit analysis
    let cost_benefit = &analysis_report.cost_benefit_analysis;
    println!("\n💰 COST-BENEFIT ANALYSIS");
    println!(
        "Total Potential Savings: ${:.2}",
        cost_benefit.total_potential_savings
    );
    println!(
        "Implementation Costs: ${:.2}",
        cost_benefit.implementation_costs
    );

    if !cost_benefit.roi_estimates.is_empty() {
        println!("\nROI Estimates:");
        for roi in &cost_benefit.roi_estimates {
            println!(
                "  {:?}: {:.1}x ROI, Break-even in {} days",
                roi.optimization_type,
                roi.estimated_roi,
                roi.time_to_break_even.as_secs() / (24 * 3600)
            );
        }
    }

    Ok(())
}

/// Demonstrate real-time monitoring
#[allow(dead_code)]
fn run_realtime_monitoring() -> Result<()> {
    println!("\n📡 REAL-TIME MEMORY MONITORING");
    println!("==============================");

    let config = MemoryOptimizerConfig {
        alert_thresholds: AlertThresholds {
            warning_threshold: 0.6, // Lower thresholds for demo
            critical_threshold: 0.8,
            allocation_rate_threshold: 100.0,
            fragmentation_threshold: 0.3,
        },
        ..Default::default()
    };

    let mut memory_optimizer = MemoryOptimizer::new(config);
    memory_optimizer.start_monitoring()?;

    println!("Monitoring memory usage with alerts...");

    // Simulate various scenarios that trigger alerts
    simulate_alert_scenarios(&mut memory_optimizer)?;

    // Check for alerts
    let alerts = memory_optimizer.get_alerts();

    if alerts.is_empty() {
        println!("✅ No alerts generated - all systems normal");
    } else {
        println!("🚨 {} alert(s) generated:", alerts.len());

        for (i, alert) in alerts.iter().enumerate() {
            println!(
                "\n{}. {:?} Alert (Severity: {:?})",
                i + 1,
                alert.alert_type,
                alert.severity
            );
            println!("   Message: {}", alert.message);
            println!("   Timestamp: {:?}", alert.timestamp);

            if !alert.suggested_actions.is_empty() {
                println!("   Suggested Actions:");
                for action in &alert.suggested_actions {
                    println!("     - {}", action);
                }
            }
        }
    }

    println!("\n📊 Real-time Memory Metrics:");
    let analysis_report = memory_optimizer.analyze_and_recommend()?;
    let performance_impact = &analysis_report.performance_impact_analysis;

    println!(
        "  Overall Performance Impact: {:.1}%",
        performance_impact.overall_impact_score * 100.0
    );

    if !performance_impact.bottlenecks.is_empty() {
        println!("  Performance Bottlenecks:");
        for bottleneck in &performance_impact.bottlenecks {
            println!(
                "    - {}: {:.1}% impact",
                bottleneck.bottleneck_type,
                bottleneck.impact * 100.0
            );
        }
    }

    Ok(())
}

// Helper functions to simulate different memory patterns

#[allow(dead_code)]
fn simulate_optimization_workload(memory_optimizer: &mut MemoryOptimizer) -> Result<()> {
    println!("  Running optimization workload with memory tracking...");

    // Create optimizers
    let mut adam = Adam::new(AdamConfig::default());
    let mut sgd = SGD::new(SGDConfig::default());

    // Simulate parameter arrays of various sizes
    let mut params_small = Array1::from_vec(vec![1.0; 100]);
    let mut params_medium = Array1::from_vec(vec![1.0; 1000]);
    let mut params_large = Array1::from_vec(vec![1.0; 10000]);

    for step in 0..50 {
        // Record memory snapshot
        memory_optimizer.record_snapshot()?;

        // Simulate gradient computation and optimization steps
        let gradients_small = simulate_gradients(&params_small, step);
        let gradients_medium = simulate_gradients(&params_medium, step);
        let gradients_large = simulate_gradients(&params_large, step);

        // Update parameters
        if let Ok(new_params) = adam.step(&params_small, &gradients_small) {
            params_small = new_params;
        }

        if let Ok(new_params) = sgd.step(&params_medium, &gradients_medium) {
            params_medium = new_params;
        }

        if let Ok(new_params) = adam.step(&params_large, &gradients_large) {
            params_large = new_params;
        }

        // Simulate temporary allocations
        if step % 10 == 0 {
            let _temp_data = vec![0u8; 1024 * 1024]; // 1MB temporary allocation
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    println!("    Completed {} optimization steps", 50);
    Ok(())
}

#[allow(dead_code)]
fn simulate_memory_leak_patterns(memory_optimizer: &mut MemoryOptimizer) -> Result<()> {
    println!("  Simulating potential memory leak patterns...");

    let mut accumulated_data = Vec::new();

    for step in 0..30 {
        memory_optimizer.record_snapshot()?;

        // Simulate accumulating data (potential leak)
        if step % 3 == 0 {
            let leak_data = vec![0u8; 1024 * (step + 1)]; // Growing allocation
            accumulated_data.push(leak_data);
        }

        // Simulate some normal allocations
        let _normal_data = vec![0u8; 1024 * 10]; // 10KB

        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    println!("    Simulated {} steps with potential leak pattern", 30);
    Ok(())
}

#[allow(dead_code)]
fn simulate_fragmentation_patterns(memory_optimizer: &mut MemoryOptimizer) -> Result<()> {
    println!("  Simulating memory fragmentation patterns...");

    let mut allocations = Vec::new();

    for step in 0..40 {
        memory_optimizer.record_snapshot()?;

        // Simulate mixed allocation sizes (causes fragmentation)
        let sizes = vec![64, 128, 256, 512, 1024, 2048];
        let size = sizes[step % sizes.len()];

        let allocation = vec![0u8; size];
        allocations.push(allocation);

        // Periodically free some allocations (creates holes)
        if step % 7 == 0 && !allocations.is_empty() {
            let remove_count = allocations.len() / 3;
            for _ in 0..remove_count {
                if !allocations.is_empty() {
                    allocations.remove(0);
                }
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    println!("    Simulated {} steps with fragmentation patterns", 40);
    Ok(())
}

#[allow(dead_code)]
fn simulate_suboptimal_patterns(memory_optimizer: &mut MemoryOptimizer) -> Result<()> {
    println!("  Simulating suboptimal memory patterns...");

    for step in 0..25 {
        memory_optimizer.record_snapshot()?;

        // Simulate frequent small allocations (suboptimal)
        for _ in 0..10 {
            let _small_alloc = vec![0u8; 32]; // Very small allocations
        }

        // Simulate large temporary allocation
        let _large_temp = vec![0u8; 1024 * 1024 * 2]; // 2MB

        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    println!("    Simulated {} steps with suboptimal patterns", 25);
    Ok(())
}

#[allow(dead_code)]
fn simulate_alert_scenarios(memory_optimizer: &mut MemoryOptimizer) -> Result<()> {
    println!("  Simulating scenarios that may trigger alerts...");

    let mut large_allocations = Vec::new();

    for step in 0..20 {
        memory_optimizer.record_snapshot()?;

        // Simulate growing memory usage
        let allocation_size = 1024 * 1024 * (step + 1); // Growing allocations
        let allocation = vec![0u8; allocation_size];
        large_allocations.push(allocation);

        // Simulate rapid allocation rate
        for _ in 0..50 {
            let _rapid_alloc = vec![0u8; 1024]; // Many small allocations
        }

        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    println!("    Simulated {} steps with alert-triggering patterns", 20);
    Ok(())
}

#[allow(dead_code)]
fn simulate_gradients(params: &Array1<f64>, step: usize) -> Array1<f64> {
    let noise_factor = 1.0 + 0.1 * (step as f64 * 0.1).sin();
    params.mapv(|x| 2.0 * x * noise_factor + 0.01 * (step as f64).cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_simulation() {
        let params = Array1::from_vec(vec![1.0, 2.0]);
        let gradients = simulate_gradients(&params, 0);
        assert_eq!(gradients.len(), 2);
        assert!(gradients[0] > 0.0); // Should be positive for positive params
    }

    #[test]
    fn test_memory_optimizer_creation() {
        let config = MemoryOptimizerConfig::default();
        let optimizer = MemoryOptimizer::new(config);
        assert!(optimizer.config.enable_detailed_tracking);
    }
}
