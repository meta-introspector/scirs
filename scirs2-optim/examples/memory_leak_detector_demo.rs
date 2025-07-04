//! Memory leak detector demonstration
//!
//! This example demonstrates the comprehensive memory leak detection capabilities
//! of the scirs2-optim crate, including real-time monitoring, pattern analysis,
//! and optimization recommendations.

use clap::{Arg, Command};
use ndarray::Array1;
use scirs2_optim::benchmarking::memory_leak_detector::{
    AllocationType, MemoryDetectionConfig, MemoryLeakDetector,
};
use scirs2_optim::error::Result;
use scirs2_optim::optimizers::{Adam, SGD};
use serde_json;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[allow(dead_code)]
fn main() -> Result<()> {
    let matches = Command::new("Memory Leak Detector Demo")
        .version("1.0")
        .about("Demonstrates memory leak detection capabilities")
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("FILE")
                .help("Output file for leak detection results")
                .default_value("leak_detection_results.json"),
        )
        .arg(
            Arg::new("enable-tracking")
                .long("enable-tracking")
                .action(clap::ArgAction::SetTrue)
                .help("Enable detailed allocation tracking"),
        )
        .arg(
            Arg::new("sensitivity")
                .long("sensitivity")
                .value_name("FLOAT")
                .help("Leak detection sensitivity (0.0 to 1.0)")
                .default_value("0.8"),
        )
        .arg(
            Arg::new("duration")
                .short('d')
                .long("duration")
                .value_name("SECONDS")
                .help("Duration to run memory leak detection")
                .default_value("60"),
        )
        .arg(
            Arg::new("scenario")
                .short('s')
                .long("scenario")
                .value_name("SCENARIO")
                .help("Test scenario to run")
                .value_parser([
                    "basic",
                    "optimizer-stress",
                    "memory-growth",
                    "pattern-analysis",
                ])
                .default_value("basic"),
        )
        .get_matches();

    let output_file = matches.get_one::<String>("output").unwrap();
    let enable_tracking = matches.get_flag("enable-tracking");
    let sensitivity: f64 = matches
        .get_one::<String>("sensitivity")
        .unwrap()
        .parse()
        .map_err(|_| {
            scirs2_optim::error::OptimError::InvalidConfig("Invalid sensitivity value".to_string())
        })?;
    let duration: u64 = matches
        .get_one::<String>("duration")
        .unwrap()
        .parse()
        .map_err(|_| {
            scirs2_optim::error::OptimError::InvalidConfig("Invalid duration value".to_string())
        })?;
    let scenario = matches.get_one::<String>("scenario").unwrap();

    println!("🧠 Memory Leak Detector Demo");
    println!("============================");
    println!("Output file: {}", output_file);
    println!("Sensitivity: {}", sensitivity);
    println!("Duration: {} seconds", duration);
    println!("Scenario: {}", scenario);
    println!();

    // Configure memory leak detector
    let mut config = MemoryDetectionConfig::default();
    config.enable_allocation_tracking = enable_tracking;
    config.leak_sensitivity = sensitivity;
    config.enable_real_time_monitoring = true;

    // Create memory leak detector
    let mut detector = MemoryLeakDetector::new(config)?;

    // Start monitoring
    println!("🔍 Starting memory leak detection...");
    detector.start_monitoring()?;

    match scenario {
        "basic" => run_basic_scenario(&mut detector, duration)?,
        "optimizer-stress" => run_optimizer_stress_scenario(&mut detector, duration)?,
        "memory-growth" => run_memory_growth_scenario(&mut detector, duration)?,
        "pattern-analysis" => run_pattern_analysis_scenario(&mut detector, duration)?,
        _ => unreachable!(),
    }

    // Stop monitoring
    detector.stop_monitoring()?;

    // Generate comprehensive report
    println!("\n📊 Generating memory leak analysis report...");
    let report = detector.generate_optimization_report()?;

    // Display summary
    display_leak_summary(&report);

    // Save detailed results
    let json_output = serde_json::to_string_pretty(&report)?;
    std::fs::write(output_file, json_output)?;
    println!("✅ Detailed results saved to: {}", output_file);

    // Display recommendations
    display_recommendations(&report);

    Ok(())
}

#[allow(dead_code)]
fn run_basic_scenario(detector: &mut MemoryLeakDetector, duration: u64) -> Result<()> {
    println!("Running basic memory leak detection scenario...");

    let start = Instant::now();
    let mut allocation_id = 0;

    while start.elapsed().as_secs() < duration {
        // Simulate various types of allocations
        for alloc_type in [
            AllocationType::Parameter,
            AllocationType::Gradient,
            AllocationType::OptimizerState,
            AllocationType::Temporary,
        ] {
            allocation_id += 1;
            let size = 1024 * (allocation_id % 10 + 1); // Variable sizes
            detector.record_allocation(allocation_id, size, alloc_type)?;

            // Simulate some deallocations (but not all - creating potential leaks)
            if allocation_id % 3 == 0 {
                detector.record_deallocation(allocation_id - 2)?;
            }
        }

        // Take periodic snapshots
        if allocation_id % 100 == 0 {
            detector.take_snapshot()?;
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    println!("\n✅ Basic scenario completed");
    Ok(())
}

#[allow(dead_code)]
fn run_optimizer_stress_scenario(detector: &mut MemoryLeakDetector, duration: u64) -> Result<()> {
    println!("Running optimizer stress test scenario...");

    let start = Instant::now();
    let mut optimizers = Vec::new();
    let mut allocation_id = 0;

    // Create multiple optimizers to stress test memory usage
    for i in 0..10 {
        let params = Array1::from_vec((0..1000).map(|j| (i + j) as f32 * 0.001).collect());
        if i % 2 == 0 {
            optimizers.push(Box::new(Adam::new(0.001, 0.9, 0.999, 1e-8)?));
        } else {
            optimizers.push(Box::new(SGD::new(0.01, 0.9)?));
        }

        // Record optimizer state allocations
        allocation_id += 1;
        detector.record_allocation(
            allocation_id,
            params.len() * std::mem::size_of::<f32>(),
            AllocationType::OptimizerState,
        )?;
    }

    let mut step = 0;
    while start.elapsed().as_secs() < duration {
        step += 1;

        // Simulate optimizer steps with memory allocations
        for (i, _optimizer) in optimizers.iter().enumerate() {
            allocation_id += 1;

            // Simulate parameter updates
            detector.record_allocation(
                allocation_id,
                1000 * std::mem::size_of::<f32>(),
                AllocationType::Parameter,
            )?;

            allocation_id += 1;
            // Simulate gradient allocations
            detector.record_allocation(
                allocation_id,
                1000 * std::mem::size_of::<f32>(),
                AllocationType::Gradient,
            )?;

            // Simulate some temporary allocations
            allocation_id += 1;
            detector.record_allocation(
                allocation_id,
                500 * std::mem::size_of::<f32>(),
                AllocationType::Temporary,
            )?;

            // Clean up some temporary allocations
            if step % 2 == 0 {
                detector.record_deallocation(allocation_id)?;
            }

            // But "forget" to clean up some parameter/gradient allocations
            if step % 5 == 0 && i % 3 != 0 {
                detector.record_deallocation(allocation_id - 1)?;
            }
        }

        // Take snapshots periodically
        if step % 50 == 0 {
            detector.take_snapshot()?;
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }

        std::thread::sleep(Duration::from_millis(50));
    }

    println!("\n✅ Optimizer stress test completed");
    Ok(())
}

#[allow(dead_code)]
fn run_memory_growth_scenario(detector: &mut MemoryLeakDetector, duration: u64) -> Result<()> {
    println!("Running memory growth pattern scenario...");

    let start = Instant::now();
    let mut allocation_id = 0;
    let mut growth_factor = 1;

    while start.elapsed().as_secs() < duration {
        // Simulate exponential memory growth (classic leak pattern)
        let num_allocations = growth_factor * 10;

        for _ in 0..num_allocations {
            allocation_id += 1;
            let size = 1024 * growth_factor; // Growing allocation sizes
            detector.record_allocation(allocation_id, size, AllocationType::Cache)?;

            // Only deallocate a fraction of allocations (causing growth)
            if allocation_id % 10 == 0 {
                detector.record_deallocation(allocation_id - 5)?;
            }
        }

        // Increase growth factor periodically
        if start.elapsed().as_secs() % 10 == 0 {
            growth_factor += 1;
        }

        // Take frequent snapshots to capture growth pattern
        detector.take_snapshot()?;

        print!("📈");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        std::thread::sleep(Duration::from_millis(100));
    }

    println!("\n✅ Memory growth scenario completed");
    Ok(())
}

#[allow(dead_code)]
fn run_pattern_analysis_scenario(detector: &mut MemoryLeakDetector, duration: u64) -> Result<()> {
    println!("Running memory pattern analysis scenario...");

    let start = Instant::now();
    let mut allocation_id = 0;
    let mut phase = 0;

    while start.elapsed().as_secs() < duration {
        phase = (start.elapsed().as_secs() / 10) % 4;

        match phase {
            0 => {
                // Burst allocation phase
                for _ in 0..100 {
                    allocation_id += 1;
                    detector.record_allocation(allocation_id, 1024, AllocationType::Temporary)?;
                }
                println!("💥 Burst phase");
            }
            1 => {
                // Steady allocation phase
                for _ in 0..10 {
                    allocation_id += 1;
                    detector.record_allocation(allocation_id, 2048, AllocationType::Parameter)?;
                }
                println!("🌊 Steady phase");
            }
            2 => {
                // Cleanup phase
                for i in 1..=50 {
                    if allocation_id >= i {
                        detector.record_deallocation(allocation_id - i + 1)?;
                    }
                }
                println!("🧹 Cleanup phase");
            }
            3 => {
                // Periodic allocation phase
                if start.elapsed().as_millis() % 1000 < 500 {
                    allocation_id += 1;
                    detector.record_allocation(
                        allocation_id,
                        4096,
                        AllocationType::OptimizerState,
                    )?;
                }
                println!("🔄 Periodic phase");
            }
            _ => unreachable!(),
        }

        // Take snapshots to capture patterns
        detector.take_snapshot()?;

        std::thread::sleep(Duration::from_millis(200));
    }

    println!("\n✅ Pattern analysis scenario completed");
    Ok(())
}

#[allow(dead_code)]
fn display_leak_summary(
    report: &scirs2_optim::benchmarking::memory_leak_detector::MemoryOptimizationReport,
) {
    println!("\n📋 Memory Leak Detection Summary");
    println!("================================");
    println!("{}", report.summary);

    if !report.leak_results.is_empty() {
        println!("\n🔍 Leak Detection Results:");
        for (i, leak) in report.leak_results.iter().enumerate() {
            if leak.leak_detected {
                println!(
                    "  Leak #{}: {} bytes (severity: {:.2}, confidence: {:.2})",
                    i + 1,
                    leak.leaked_memory_bytes,
                    leak.severity,
                    leak.confidence
                );

                if !leak.leak_sources.is_empty() {
                    println!("    Sources:");
                    for source in &leak.leak_sources {
                        println!(
                            "      - {:?}: {} bytes (prob: {:.2})",
                            source.source_type, source.leak_size, source.probability
                        );
                    }
                }
            }
        }
    } else {
        println!("\n✅ No memory leaks detected!");
    }

    if !report.patterns.is_empty() {
        println!("\n📊 Memory Patterns Detected:");
        for pattern in &report.patterns {
            println!(
                "  - {}: {} (confidence: {:.2})",
                pattern.pattern_type, pattern.description, pattern.confidence
            );
        }
    }

    if !report.anomalies.is_empty() {
        println!("\n⚠️  Memory Anomalies:");
        for anomaly in &report.anomalies {
            println!(
                "  - {}: {} (severity: {:.2})",
                anomaly.anomaly_type, anomaly.description, anomaly.severity
            );
        }
    }
}

#[allow(dead_code)]
fn display_recommendations(
    report: &scirs2_optim::benchmarking::memory_leak_detector::MemoryOptimizationReport,
) {
    if !report.recommendations.is_empty() {
        println!("\n💡 Optimization Recommendations");
        println!("===============================");

        for (i, rec) in report.recommendations.iter().enumerate() {
            println!(
                "{}. {:?} - {:?}",
                i + 1,
                rec.recommendation_type,
                rec.priority
            );
            println!("   {}", rec.description);
            println!("   Expected impact: {}", rec.expected_impact);

            if let Some(examples) = &rec.code_examples {
                println!("   Code examples:");
                for example in examples {
                    println!("     {}", example);
                }
            }
            println!();
        }
    }

    println!("🎯 Performance Metrics");
    println!("======================");
    println!(
        "Memory Efficiency: {:.2}%",
        report.performance_metrics.memory_efficiency * 100.0
    );
    println!(
        "Allocation Efficiency: {:.2}%",
        report.performance_metrics.allocation_efficiency * 100.0
    );
    println!(
        "Cache Hit Ratio: {:.2}%",
        report.performance_metrics.cache_hit_ratio * 100.0
    );
    println!(
        "GC Overhead: {:.2}%",
        report.performance_metrics.gc_overhead * 100.0
    );
}
