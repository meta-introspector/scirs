//! Comprehensive memory profiler for optimizer performance analysis
//!
//! This example provides advanced memory profiling capabilities including
//! real-time monitoring, allocation tracking, pattern analysis, and
//! performance optimization recommendations.

use clap::{Arg, Command};
use ndarray::{Array1, Array2};
use scirs2_optim::benchmarking::memory_leak_detector::{
    AllocationType, MemoryDetectionConfig, MemoryLeakDetector,
};
use scirs2_optim::error::Result;
use scirs2_optim::optimizers::{Adam, AdamW, RMSprop, SGD};
use serde_json;
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
struct ProfilingSession {
    session_id: String,
    start_time: SystemTime,
    optimizer_type: String,
    problem_size: usize,
    batch_size: usize,
}

#[derive(Clone, Debug)]
struct MemorySnapshot {
    timestamp: u64,
    total_allocated: usize,
    total_deallocated: usize,
    active_allocations: usize,
    peak_memory: usize,
    allocation_rate: f64,
    deallocation_rate: f64,
}

#[derive(Clone, Debug)]
struct ProfilingResults {
    session: ProfilingSession,
    snapshots: Vec<MemorySnapshot>,
    leak_analysis:
        Option<scirs2_optim::benchmarking::memory_leak_detector::MemoryOptimizationReport>,
    performance_metrics: HashMap<String, f64>,
    efficiency_score: f64,
}

#[allow(dead_code)]
fn main() -> Result<()> {
    let matches = Command::new("Comprehensive Memory Profiler")
        .version("1.0")
        .about("Advanced memory profiling for optimizer performance analysis")
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("FILE")
                .help("Output file for profiling results")
                .default_value("memory_profile.json"),
        )
        .arg(
            Arg::new("duration")
                .short('d')
                .long("duration")
                .value_name("SECONDS")
                .help("Duration to run profiling")
                .default_value("300"),
        )
        .arg(
            Arg::new("sampling-rate")
                .long("sampling-rate")
                .value_name("MILLISECONDS")
                .help("Memory sampling rate in milliseconds")
                .default_value("1000"),
        )
        .arg(
            Arg::new("optimizer")
                .long("optimizer")
                .value_name("OPTIMIZER")
                .help("Optimizer to profile")
                .value_parser(["adam", "sgd", "adamw", "rmsprop", "all"])
                .default_value("all"),
        )
        .arg(
            Arg::new("problem-size")
                .long("problem-size")
                .value_name("SIZE")
                .help("Problem size (number of parameters)")
                .default_value("10000"),
        )
        .arg(
            Arg::new("batch-size")
                .long("batch-size")
                .value_name("SIZE")
                .help("Batch size for training simulation")
                .default_value("128"),
        )
        .arg(
            Arg::new("enable-detailed-tracking")
                .long("enable-detailed-tracking")
                .action(clap::ArgAction::SetTrue)
                .help("Enable detailed allocation tracking"),
        )
        .get_matches();

    let output_file = matches.get_one::<String>("output").unwrap();
    let duration: u64 = matches
        .get_one::<String>("duration")
        .unwrap()
        .parse()
        .map_err(|_| {
            scirs2_optim::error::OptimError::InvalidConfig("Invalid duration".to_string())
        })?;
    let sampling_rate: u64 = matches
        .get_one::<String>("sampling-rate")
        .unwrap()
        .parse()
        .map_err(|_| {
            scirs2_optim::error::OptimError::InvalidConfig("Invalid sampling rate".to_string())
        })?;
    let optimizer_type = matches.get_one::<String>("optimizer").unwrap();
    let problem_size: usize = matches
        .get_one::<String>("problem-size")
        .unwrap()
        .parse()
        .map_err(|_| {
            scirs2_optim::error::OptimError::InvalidConfig("Invalid problem size".to_string())
        })?;
    let batch_size: usize = matches
        .get_one::<String>("batch-size")
        .unwrap()
        .parse()
        .map_err(|_| {
            scirs2_optim::error::OptimError::InvalidConfig("Invalid batch size".to_string())
        })?;
    let detailed_tracking = matches.get_flag("enable-detailed-tracking");

    println!("🔬 Comprehensive Memory Profiler");
    println!("=================================");
    println!("Duration: {} seconds", duration);
    println!("Sampling rate: {} ms", sampling_rate);
    println!("Optimizer: {}", optimizer_type);
    println!("Problem size: {} parameters", problem_size);
    println!("Batch size: {}", batch_size);
    println!("Detailed tracking: {}", detailed_tracking);
    println!();

    let mut results = Vec::new();

    let optimizers = if optimizer_type == "all" {
        vec!["adam", "sgd", "adamw", "rmsprop"]
    } else {
        vec![optimizer_type]
    };

    for opt_type in optimizers {
        println!("📊 Profiling optimizer: {}", opt_type);
        let result = profile_optimizer(
            opt_type,
            problem_size,
            batch_size,
            duration,
            sampling_rate,
            detailed_tracking,
        )?;

        display_profiling_summary(&result);
        results.push(result);
        println!();
    }

    // Generate comparative analysis
    if results.len() > 1 {
        generate_comparative_analysis(&results);
    }

    // Save results
    let json_output = serde_json::to_string_pretty(&results)?;
    std::fs::write(output_file, json_output)?;
    println!("💾 Results saved to: {}", output_file);

    Ok(())
}

#[allow(dead_code)]
fn profile_optimizer(
    optimizer_type: &str,
    problem_size: usize,
    batch_size: usize,
    duration: u64,
    sampling_rate: u64,
    detailed_tracking: bool,
) -> Result<ProfilingResults> {
    // Configure memory detector
    let mut config = MemoryDetectionConfig::default();
    config.enable_allocation_tracking = detailed_tracking;
    config.sampling_rate = sampling_rate;
    config.enable_real_time_monitoring = true;

    let mut detector = MemoryLeakDetector::new(config)?;
    detector.start_monitoring()?;

    let session = ProfilingSession {
        session_id: format!(
            "{}_{}",
            optimizer_type,
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
        ),
        start_time: SystemTime::now(),
        optimizer_type: optimizer_type.to_string(),
        problem_size,
        batch_size,
    };

    println!("  🚀 Starting profiling session: {}", session.session_id);

    let start_time = Instant::now();
    let mut snapshots = Vec::new();
    let mut allocation_id = 0;
    let mut step = 0;

    // Initialize optimizer-specific state
    let mut optimizer_state_size = estimate_optimizer_state_size(optimizer_type, problem_size);

    // Simulate initial optimizer allocation
    allocation_id += 1;
    detector.record_allocation(
        allocation_id,
        optimizer_state_size,
        AllocationType::OptimizerState,
    )?;

    // Parameters allocation
    allocation_id += 1;
    detector.record_allocation(
        allocation_id,
        problem_size * std::mem::size_of::<f32>(),
        AllocationType::Parameter,
    )?;

    while start_time.elapsed().as_secs() < duration {
        step += 1;

        // Simulate training step memory allocations
        simulate_training_step(
            &mut detector,
            &mut allocation_id,
            optimizer_type,
            problem_size,
            batch_size,
        )?;

        // Take memory snapshot
        if step % (sampling_rate / 100) == 0 {
            let snapshot = create_memory_snapshot(&detector, start_time.elapsed())?;
            snapshots.push(snapshot);
        }

        // Update optimizer state size (some optimizers grow)
        if optimizer_type == "adam" || optimizer_type == "adamw" {
            if step % 1000 == 0 {
                optimizer_state_size = (optimizer_state_size as f64 * 1.001) as usize;
            }
        }

        // Simulate periodic cleanup
        if step % 100 == 0 {
            simulate_cleanup(&mut detector, allocation_id, 10)?;
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    detector.stop_monitoring()?;

    // Generate leak analysis
    let leak_analysis = detector.generate_optimization_report().ok();

    // Calculate performance metrics
    let performance_metrics = calculate_performance_metrics(&snapshots, &session);

    // Calculate efficiency score
    let efficiency_score = calculate_efficiency_score(&snapshots, &performance_metrics);

    println!("  ✅ Profiling session completed");

    Ok(ProfilingResults {
        session,
        snapshots,
        leak_analysis,
        performance_metrics,
        efficiency_score,
    })
}

#[allow(dead_code)]
fn simulate_training_step(
    detector: &mut MemoryLeakDetector,
    allocation_id: &mut usize,
    optimizer_type: &str,
    problem_size: usize,
    batch_size: usize,
) -> Result<()> {
    // Forward pass allocations
    *allocation_id += 1;
    detector.record_allocation(
        *allocation_id,
        batch_size * problem_size * std::mem::size_of::<f32>(),
        AllocationType::Temporary,
    )?;

    // Gradient allocation
    *allocation_id += 1;
    detector.record_allocation(
        *allocation_id,
        problem_size * std::mem::size_of::<f32>(),
        AllocationType::Gradient,
    )?;

    // Optimizer-specific allocations
    match optimizer_type {
        "adam" | "adamw" => {
            // Momentum and velocity buffers
            *allocation_id += 1;
            detector.record_allocation(
                *allocation_id,
                problem_size * std::mem::size_of::<f32>() * 2, // m and v
                AllocationType::OptimizerState,
            )?;
        }
        "rmsprop" => {
            // Squared gradient buffer
            *allocation_id += 1;
            detector.record_allocation(
                *allocation_id,
                problem_size * std::mem::size_of::<f32>(),
                AllocationType::OptimizerState,
            )?;
        }
        "sgd" => {
            // Momentum buffer (if used)
            *allocation_id += 1;
            detector.record_allocation(
                *allocation_id,
                problem_size * std::mem::size_of::<f32>(),
                AllocationType::OptimizerState,
            )?;
        }
        _ => {}
    }

    // Clean up temporary allocations
    detector.record_deallocation(*allocation_id - 2)?; // Forward pass temps

    Ok(())
}

#[allow(dead_code)]
fn simulate_cleanup(
    detector: &mut MemoryLeakDetector,
    current_allocation_id: usize,
    cleanup_count: usize,
) -> Result<()> {
    for i in 1..=cleanup_count {
        if current_allocation_id >= i {
            detector.record_deallocation(current_allocation_id - i + 1)?;
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn create_memory_snapshot(
    detector: &MemoryLeakDetector,
    elapsed: Duration,
) -> Result<MemorySnapshot> {
    let snapshot = detector.take_snapshot()?;

    Ok(MemorySnapshot {
        timestamp: elapsed.as_millis() as u64,
        total_allocated: 0,    // Would be calculated from detector state
        total_deallocated: 0,  // Would be calculated from detector state
        active_allocations: 0, // Would be calculated from detector state
        peak_memory: snapshot.total_memory,
        allocation_rate: 0.0,   // Would be calculated from recent history
        deallocation_rate: 0.0, // Would be calculated from recent history
    })
}

#[allow(dead_code)]
fn estimate_optimizer_state_size(optimizer_type: &str, problem_size: usize) -> usize {
    let base_size = problem_size * std::mem::size_of::<f32>();

    match optimizer_type {
        "adam" | "adamw" => base_size * 3, // params + momentum + velocity
        "rmsprop" => base_size * 2,        // params + squared gradients
        "sgd" => base_size * 2,            // params + momentum
        _ => base_size,
    }
}

#[allow(dead_code)]
fn calculate_performance_metrics(
    snapshots: &[MemorySnapshot],
    session: &ProfilingSession,
) -> HashMap<String, f64> {
    let mut metrics = HashMap::new();

    if snapshots.is_empty() {
        return metrics;
    }

    // Memory growth rate
    let first_memory = snapshots.first().unwrap().peak_memory as f64;
    let last_memory = snapshots.last().unwrap().peak_memory as f64;
    let duration = (snapshots.last().unwrap().timestamp - snapshots.first().unwrap().timestamp)
        as f64
        / 1000.0;

    if duration > 0.0 {
        let growth_rate = (last_memory - first_memory) / duration;
        metrics.insert("memory_growth_rate_bytes_per_sec".to_string(), growth_rate);
    }

    // Peak memory usage
    let peak_memory = snapshots.iter().map(|s| s.peak_memory).max().unwrap_or(0) as f64;
    metrics.insert("peak_memory_bytes".to_string(), peak_memory);

    // Average memory usage
    let avg_memory =
        snapshots.iter().map(|s| s.peak_memory as f64).sum::<f64>() / snapshots.len() as f64;
    metrics.insert("average_memory_bytes".to_string(), avg_memory);

    // Memory efficiency (parameter memory / total memory)
    let param_memory = session.problem_size * std::mem::size_of::<f32>();
    let efficiency = param_memory as f64 / avg_memory;
    metrics.insert("memory_efficiency".to_string(), efficiency);

    // Memory stability (coefficient of variation)
    let memory_values: Vec<f64> = snapshots.iter().map(|s| s.peak_memory as f64).collect();
    let std_dev = calculate_std_dev(&memory_values);
    let cv = std_dev / avg_memory;
    metrics.insert("memory_stability".to_string(), 1.0 - cv.min(1.0));

    metrics
}

#[allow(dead_code)]
fn calculate_std_dev(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

#[allow(dead_code)]
fn calculate_efficiency_score(snapshots: &[MemorySnapshot], metrics: &HashMap<String, f64>) -> f64 {
    let mut score = 1.0;

    // Penalize high memory growth
    if let Some(growth_rate) = metrics.get("memory_growth_rate_bytes_per_sec") {
        if *growth_rate > 1024.0 {
            // More than 1KB/sec growth
            score *= 0.8;
        }
    }

    // Reward high memory efficiency
    if let Some(efficiency) = metrics.get("memory_efficiency") {
        score *= efficiency.min(1.0);
    }

    // Reward memory stability
    if let Some(stability) = metrics.get("memory_stability") {
        score *= stability;
    }

    // Penalize excessive peak memory
    if let Some(peak_memory) = metrics.get("peak_memory_bytes") {
        let reasonable_memory = 100 * 1024 * 1024; // 100MB baseline
        if *peak_memory > reasonable_memory as f64 {
            score *= reasonable_memory as f64 / peak_memory;
        }
    }

    score.max(0.0).min(1.0)
}

#[allow(dead_code)]
fn display_profiling_summary(results: &ProfilingResults) {
    println!("  📋 Session: {}", results.session.session_id);
    println!("  ⚙️  Optimizer: {}", results.session.optimizer_type);
    println!("  📊 Snapshots: {}", results.snapshots.len());
    println!("  🎯 Efficiency Score: {:.3}", results.efficiency_score);

    for (metric, value) in &results.performance_metrics {
        println!("  📈 {}: {:.2}", metric, value);
    }

    if let Some(leak_analysis) = &results.leak_analysis {
        println!("  🔍 Leak Analysis:");
        println!("    - Leaks detected: {}", leak_analysis.leak_results.len());
        println!("    - Patterns found: {}", leak_analysis.patterns.len());
        println!(
            "    - Recommendations: {}",
            leak_analysis.recommendations.len()
        );
    }
}

#[allow(dead_code)]
fn generate_comparative_analysis(results: &[ProfilingResults]) {
    println!("🏆 Comparative Analysis");
    println!("======================");

    // Find best and worst performers
    let mut efficiency_scores: Vec<_> = results
        .iter()
        .map(|r| (&r.session.optimizer_type, r.efficiency_score))
        .collect();
    efficiency_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("🥇 Efficiency Ranking:");
    for (i, (optimizer, score)) in efficiency_scores.iter().enumerate() {
        let medal = match i {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        println!("  {} {}: {:.3}", medal, optimizer, score);
    }

    // Memory usage comparison
    println!("\n💾 Memory Usage Comparison:");
    for result in results {
        if let Some(peak_memory) = result.performance_metrics.get("peak_memory_bytes") {
            println!(
                "  {}: {:.2} MB",
                result.session.optimizer_type,
                peak_memory / (1024.0 * 1024.0)
            );
        }
    }

    // Growth rate comparison
    println!("\n📈 Memory Growth Rate Comparison:");
    for result in results {
        if let Some(growth_rate) = result
            .performance_metrics
            .get("memory_growth_rate_bytes_per_sec")
        {
            println!(
                "  {}: {:.2} KB/s",
                result.session.optimizer_type,
                growth_rate / 1024.0
            );
        }
    }

    // Recommendations
    println!("\n💡 Recommendations:");
    let best_optimizer = &efficiency_scores[0].0;
    let worst_optimizer = &efficiency_scores.last().unwrap().0;

    println!(
        "  ✅ Best performer: {} (efficiency: {:.3})",
        best_optimizer, efficiency_scores[0].1
    );
    println!(
        "  ⚠️  Consider optimizing: {} (efficiency: {:.3})",
        worst_optimizer,
        efficiency_scores.last().unwrap().1
    );

    if efficiency_scores[0].1 - efficiency_scores.last().unwrap().1 > 0.2 {
        println!(
            "  🔧 Significant performance gap detected - investigate memory allocation patterns"
        );
    }
}
