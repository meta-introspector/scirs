//! Asynchronous parallel optimization examples
//!
//! This example demonstrates how to use asynchronous parallel optimization
//! for problems where function evaluations have highly variable execution times.

use ndarray::{Array1, ArrayView1};
use scirs2_optimize::async_parallel::{
    AsyncDifferentialEvolution, AsyncOptimizationConfig, SlowEvaluationStrategy,
};
use std::time::{Duration, Instant};
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Asynchronous Parallel Optimization Examples\n");

    // Basic async optimization demo
    basic_async_optimization_demo().await?;

    // Variable timing optimization
    variable_timing_demo().await?;

    // Timeout handling demo
    timeout_handling_demo().await?;

    // Performance comparison
    performance_comparison_demo().await?;

    Ok(())
}

/// Basic demonstration of async optimization
async fn basic_async_optimization_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic Async Optimization Demo ===");

    // Simple quadratic function with simulated computation time
    let quadratic_fn = |x: Array1<f64>| async move {
        // Simulate some computation time (50-150ms)
        let delay = 50 + (rand::random::<u64>() % 100);
        sleep(Duration::from_millis(delay)).await;

        // Return quadratic function value
        x.iter().map(|&xi| xi.powi(2)).sum::<f64>()
    };

    let bounds_lower = Array1::from_vec(vec![-5.0, -5.0, -5.0]);
    let bounds_upper = Array1::from_vec(vec![5.0, 5.0, 5.0]);

    let optimizer = AsyncDifferentialEvolution::new(3, Some(30), None)
        .with_bounds(bounds_lower, bounds_upper)?
        .with_parameters(0.8, 0.7, 20, 1e-6);

    let start_time = Instant::now();
    let (result, stats) = optimizer.optimize(quadratic_fn).await?;
    let total_time = start_time.elapsed();

    println!("Optimization Results:");
    println!(
        "  Final solution: [{:.6}, {:.6}, {:.6}]",
        result.x[0], result.x[1], result.x[2]
    );
    println!("  Final cost: {:.6e}", result.fun);
    println!("  Generations: {}", result.iterations);
    println!("  Function evaluations: {}", result.nfev);
    println!("  Success: {}", result.success);
    println!("\nPerformance Statistics:");
    println!(
        "  Total optimization time: {:.2} seconds",
        total_time.as_secs_f64()
    );
    println!("  Total evaluations submitted: {}", stats.total_submitted);
    println!("  Total evaluations completed: {}", stats.total_completed);
    println!("  Total evaluations cancelled: {}", stats.total_cancelled);
    println!(
        "  Average evaluation time: {:.2} ms",
        stats.avg_evaluation_time.as_millis()
    );
    println!(
        "  Min evaluation time: {:.2} ms",
        stats.min_evaluation_time.as_millis()
    );
    println!(
        "  Max evaluation time: {:.2} ms",
        stats.max_evaluation_time.as_millis()
    );
    println!();

    Ok(())
}

/// Demonstrate optimization with highly variable evaluation times
async fn variable_timing_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Variable Timing Demo ===");

    // Rosenbrock function with highly variable evaluation times
    let variable_rosenbrock = |x: Array1<f64>| async move {
        // Simulate highly variable computation times
        let delay = match rand::random::<f64>() {
            r if r < 0.5 => 10 + (rand::random::<u64>() % 20), // Fast: 10-30ms (50% of cases)
            r if r < 0.8 => 100 + (rand::random::<u64>() % 100), // Medium: 100-200ms (30% of cases)
            _ => 500 + (rand::random::<u64>() % 500),          // Slow: 500-1000ms (20% of cases)
        };

        sleep(Duration::from_millis(delay)).await;

        // 2D Rosenbrock function
        let a = 1.0 - x[0];
        let b = x[1] - x[0].powi(2);
        a.powi(2) + 100.0 * b.powi(2)
    };

    let bounds_lower = Array1::from_vec(vec![-2.0, -2.0]);
    let bounds_upper = Array1::from_vec(vec![2.0, 2.0]);

    // Configure for handling variable timing
    let config = AsyncOptimizationConfig {
        max_workers: 8,
        evaluation_timeout: Some(Duration::from_secs(2)),
        completion_timeout: Some(Duration::from_secs(5)),
        slow_evaluation_strategy: SlowEvaluationStrategy::UsePartial { min_fraction: 0.75 },
        min_evaluations: 8,
    };

    let optimizer = AsyncDifferentialEvolution::new(2, Some(40), Some(config))
        .with_bounds(bounds_lower, bounds_upper)?
        .with_parameters(0.8, 0.7, 25, 1e-5);

    let start_time = Instant::now();
    let (result, stats) = optimizer.optimize(variable_rosenbrock).await?;
    let total_time = start_time.elapsed();

    println!("Variable Timing Results:");
    println!("  Final solution: [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Final cost: {:.6e}", result.fun);
    println!("  Target solution: [1.0, 1.0] with cost 0.0");
    println!(
        "  Error from target: {:.6}",
        ((result.x[0] - 1.0).powi(2) + (result.x[1] - 1.0).powi(2)).sqrt()
    );
    println!("  Generations: {}", result.iterations);
    println!("\nTiming Analysis:");
    println!("  Total time: {:.2} seconds", total_time.as_secs_f64());
    println!(
        "  Completed: {} / {} evaluations",
        stats.total_completed, stats.total_submitted
    );
    println!(
        "  Cancellation rate: {:.1}%",
        stats.total_cancelled as f64 / stats.total_submitted as f64 * 100.0
    );
    println!(
        "  Average eval time: {:.2} ms",
        stats.avg_evaluation_time.as_millis()
    );
    println!(
        "  Min eval time: {:.2} ms",
        stats.min_evaluation_time.as_millis()
    );
    println!(
        "  Max eval time: {:.2} ms",
        stats.max_evaluation_time.as_millis()
    );
    println!();

    Ok(())
}

/// Demonstrate timeout handling strategies
async fn timeout_handling_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Timeout Handling Demo ===");

    // Function that sometimes hangs (simulating network calls, complex simulations, etc.)
    let unreliable_function = |x: Array1<f64>| async move {
        // 30% chance of taking way too long
        if rand::random::<f64>() < 0.3 {
            sleep(Duration::from_secs(5)).await; // Very slow evaluation
        } else {
            // Normal evaluation time
            sleep(Duration::from_millis(20 + rand::random::<u64>() % 50)).await;
        }

        // Simple sphere function
        x.iter().map(|&xi| xi.powi(2)).sum::<f64>()
    };

    let bounds_lower = Array1::from_vec(vec![-3.0, -3.0, -3.0, -3.0]);
    let bounds_upper = Array1::from_vec(vec![3.0, 3.0, 3.0, 3.0]);

    // Test different timeout strategies
    let strategies = vec![
        ("Wait All", SlowEvaluationStrategy::WaitAll),
        (
            "Cancel Slow",
            SlowEvaluationStrategy::CancelSlow {
                timeout: Duration::from_millis(200),
            },
        ),
        (
            "Use Partial",
            SlowEvaluationStrategy::UsePartial { min_fraction: 0.7 },
        ),
    ];

    for (strategy_name, strategy) in strategies {
        println!("Testing strategy: {}", strategy_name);

        let config = AsyncOptimizationConfig {
            max_workers: 6,
            evaluation_timeout: Some(Duration::from_millis(300)),
            completion_timeout: Some(Duration::from_millis(1000)),
            slow_evaluation_strategy: strategy,
            min_evaluations: 5,
        };

        let optimizer = AsyncDifferentialEvolution::new(4, Some(20), Some(config))
            .with_bounds(bounds_lower.clone(), bounds_upper.clone())?
            .with_parameters(0.8, 0.7, 10, 1e-4);

        let start_time = Instant::now();
        let (result, stats) = optimizer.optimize(unreliable_function).await?;
        let total_time = start_time.elapsed();

        println!(
            "  Time: {:.2}s, Final cost: {:.6e}, Completed: {}/{}, Cancelled: {}",
            total_time.as_secs_f64(),
            result.fun,
            stats.total_completed,
            stats.total_submitted,
            stats.total_cancelled
        );
    }
    println!();

    Ok(())
}

/// Compare async vs synchronous optimization performance
async fn performance_comparison_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Performance Comparison: Async vs Sync ===");

    // Function with moderate variable timing
    let test_function = |x: Array1<f64>| async move {
        // Simulate realistic variable computation times (10-100ms)
        let delay = 10 + (rand::random::<u64>() % 90);
        sleep(Duration::from_millis(delay)).await;

        // Extended Rosenbrock function
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            let a = 1.0 - x[i];
            let b = x[i + 1] - x[i].powi(2);
            sum += a.powi(2) + 100.0 * b.powi(2);
        }
        sum
    };

    let dimensions = 6;
    let bounds_lower = Array1::from_vec(vec![-2.0; dimensions]);
    let bounds_upper = Array1::from_vec(vec![2.0; dimensions]);

    // Test with different worker counts
    let worker_counts = vec![1, 2, 4, 8, 16];

    for &workers in &worker_counts {
        let config = AsyncOptimizationConfig {
            max_workers: workers,
            evaluation_timeout: Some(Duration::from_millis(200)),
            completion_timeout: Some(Duration::from_secs(2)),
            slow_evaluation_strategy: SlowEvaluationStrategy::UsePartial { min_fraction: 0.8 },
            min_evaluations: 10,
        };

        let optimizer = AsyncDifferentialEvolution::new(dimensions, Some(50), Some(config))
            .with_bounds(bounds_lower.clone(), bounds_upper.clone())?
            .with_parameters(0.8, 0.7, 15, 1e-5);

        let start_time = Instant::now();
        let (result, stats) = optimizer.optimize(test_function).await?;
        let total_time = start_time.elapsed();

        let efficiency = stats.total_completed as f64 / total_time.as_secs_f64();

        println!("Workers: {:2} | Time: {:5.2}s | Evaluations: {:3} | Efficiency: {:5.1} evals/sec | Final cost: {:.6e}",
                 workers, total_time.as_secs_f64(), stats.total_completed, efficiency, result.fun);
    }

    println!("\nNote: Efficiency shows evaluations completed per second.");
    println!("Higher efficiency indicates better parallelization.");
    println!();

    Ok(())
}

/// Helper function to simulate CPU-intensive computation
#[allow(dead_code)]
async fn cpu_intensive_function(x: Array1<f64>) -> f64 {
    // Simulate variable CPU load
    let iterations = 1000 + (rand::random::<usize>() % 9000); // 1k-10k iterations

    tokio::task::spawn_blocking(move || {
        let mut result = 0.0;
        for i in 0..iterations {
            result += (i as f64).sin();
        }

        // Actual objective function (Ackley function)
        let a = 20.0;
        let b = 0.2;
        let c = 2.0 * std::f64::consts::PI;
        let n = x.len() as f64;

        let sum1 = x.iter().map(|&xi| xi.powi(2)).sum::<f64>() / n;
        let sum2 = x.iter().map(|&xi| (c * xi).cos()).sum::<f64>() / n;

        -a * (-b * sum1.sqrt()).exp() - sum2.exp() + a + std::f64::consts::E + result * 1e-10
    })
    .await
    .unwrap_or(f64::INFINITY)
}

/// Simulate network-based function evaluation
#[allow(dead_code)]
async fn network_simulation_function(x: Array1<f64>) -> f64 {
    // Simulate network latency and occasional failures
    let latency = match rand::random::<f64>() {
        r if r < 0.7 => 20 + rand::random::<u64>() % 30, // Fast network: 20-50ms
        r if r < 0.9 => 100 + rand::random::<u64>() % 100, // Slow network: 100-200ms
        _ => 500 + rand::random::<u64>() % 1000,         // Very slow: 500-1500ms
    };

    sleep(Duration::from_millis(latency)).await;

    // 5% chance of "network failure" (timeout)
    if rand::random::<f64>() < 0.05 {
        sleep(Duration::from_secs(10)).await; // Will be cancelled by timeout
    }

    // Simple test function
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += (x[i] - 0.5 * (i as f64 + 1.0)).powi(2);
    }
    sum
}
