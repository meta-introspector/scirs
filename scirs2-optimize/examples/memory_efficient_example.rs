//! Memory-efficient optimization example
//!
//! This example demonstrates memory-efficient optimization algorithms
//! for large-scale problems with limited memory.

use ndarray::{Array1, ArrayView1};
use scirs2_optimize::unconstrained::{
    create_memory_efficient_optimizer, create_ultra_scale_optimizer,
    minimize_memory_efficient_lbfgs, minimize_ultra_scale, MemoryOptions, UltraScaleOptions,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Memory-Efficient Optimization Examples\n");

    // Example 1: Standard memory-efficient L-BFGS
    standard_memory_efficient_example()?;

    // Example 2: Custom memory configuration
    custom_memory_config_example()?;

    // Example 3: Ultra-large-scale optimization
    ultra_scale_example()?;

    // Example 4: Memory usage comparison
    memory_comparison_example()?;

    Ok(())
}

/// Example 1: Standard memory-efficient L-BFGS for moderately large problems
fn standard_memory_efficient_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Standard Memory-Efficient L-BFGS ===");

    // Create a large-scale quadratic problem
    let n = 10000;
    println!("Problem size: {} variables", n);

    let quadratic = |x: &ArrayView1<f64>| -> f64 {
        // Quadratic function: f(x) = sum((x_i - i/n)^2)
        let mut sum = 0.0;
        for (i, &xi) in x.iter().enumerate() {
            let target = (i as f64) / (n as f64);
            sum += (xi - target).powi(2);
        }
        sum
    };

    // Create memory-efficient optimizer
    let memory_options = create_memory_efficient_optimizer(n, 64); // 64MB available

    println!("Memory configuration:");
    println!("  Chunk size: {}", memory_options.chunk_size);
    println!("  Max history: {}", memory_options.max_history);
    println!(
        "  Max memory: {} MB",
        memory_options.max_memory_bytes / (1024 * 1024)
    );

    let x0 = Array1::zeros(n);

    let start = Instant::now();
    let result = minimize_memory_efficient_lbfgs(quadratic, x0, &memory_options)?;
    let duration = start.elapsed();

    println!("Results:");
    println!("  Success: {}", result.success);
    println!("  Iterations: {}", result.iterations);
    println!("  Function evaluations: {}", result.func_evals);
    println!("  Final function value: {:.6e}", result.fun.into());
    println!("  Optimization time: {:.3} seconds", duration.as_secs_f64());

    // Check solution quality on first few variables
    let mut max_error = 0.0;
    for i in 0..std::cmp::min(10, n) {
        let target = (i as f64) / (n as f64);
        let error = (result.x[i] - target).abs();
        max_error = max_error.max(error);
    }
    println!("  Max error in first 10 variables: {:.6e}", max_error);
    println!();

    Ok(())
}

/// Example 2: Custom memory configuration for specific constraints
fn custom_memory_config_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Custom Memory Configuration ===");

    let n = 5000;
    println!("Problem size: {} variables", n);

    // Rosenbrock-like function (more challenging)
    let rosenbrock_extended = |x: &ArrayView1<f64>| -> f64 {
        let mut sum = 0.0;
        for i in 0..(x.len() - 1) {
            let a = 1.0;
            let b = 100.0;
            sum += (a - x[i]).powi(2) + b * (x[i + 1] - x[i].powi(2)).powi(2);
        }
        sum
    };

    // Custom memory options for challenging problem
    let mut memory_options = MemoryOptions::default();
    memory_options.chunk_size = 256; // Smaller chunks for stability
    memory_options.max_history = 20; // More history for difficult problem
    memory_options.max_memory_bytes = 32 * 1024 * 1024; // 32MB limit
    memory_options.use_memory_pool = true;
    memory_options.base_options.max_iter = 1000;
    memory_options.base_options.gtol = 1e-6;

    println!("Custom memory configuration:");
    println!("  Chunk size: {}", memory_options.chunk_size);
    println!("  Max history: {}", memory_options.max_history);
    println!("  Memory pool: {}", memory_options.use_memory_pool);

    let x0 = Array1::zeros(n);

    let start = Instant::now();
    let result = minimize_memory_efficient_lbfgs(rosenbrock_extended, x0, &memory_options)?;
    let duration = start.elapsed();

    println!("Results:");
    println!("  Success: {}", result.success);
    println!("  Iterations: {}", result.iterations);
    println!("  Function evaluations: {}", result.func_evals);
    println!("  Final function value: {:.6e}", result.fun.into());
    println!("  Optimization time: {:.3} seconds", duration.as_secs_f64());
    println!();

    Ok(())
}

/// Example 3: Ultra-large-scale optimization with progressive refinement
fn ultra_scale_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Ultra-Large-Scale Optimization ===");

    let n = 50000; // Very large problem
    println!("Problem size: {} variables", n);

    // Sparse-like quadratic function
    let sparse_quadratic = |x: &ArrayView1<f64>| -> f64 {
        let mut sum = 0.0;
        for (i, &xi) in x.iter().enumerate() {
            // Most variables only interact with nearby ones (sparse structure)
            sum += xi.powi(2);

            if i > 0 && i % 10 == 0 {
                sum += 0.1 * xi * x[i - 1];
            }
            if i < x.len() - 1 && (i + 1) % 10 == 0 {
                sum += 0.1 * xi * x[i + 1];
            }
        }
        sum
    };

    // Create ultra-scale optimizer
    let ultra_options = create_ultra_scale_optimizer(
        n,   // problem size
        128, // 128MB available memory
        0.1, // 10% sparsity
    );

    println!("Ultra-scale configuration:");
    println!(
        "  Max variables in memory: {}",
        ultra_options.max_variables_in_memory
    );
    println!("  Block size: {}", ultra_options.block_size);
    println!("  Refinement passes: {}", ultra_options.refinement_passes);
    println!("  Use disk storage: {}", ultra_options.use_disk_storage);

    let x0 = Array1::ones(n) * 0.1;

    let start = Instant::now();
    let result = minimize_ultra_scale(sparse_quadratic, x0, &ultra_options)?;
    let duration = start.elapsed();

    println!("Results:");
    println!("  Success: {}", result.success);
    println!("  Iterations: {}", result.iterations);
    println!("  Function evaluations: {}", result.func_evals);
    println!("  Final function value: {:.6e}", result.fun.into());
    println!("  Optimization time: {:.3} seconds", duration.as_secs_f64());

    // Check convergence
    let solution_norm = result.x.mapv(|x| x.abs()).sum() / n as f64;
    println!("  Average |x_i|: {:.6e}", solution_norm);
    println!();

    Ok(())
}

/// Example 4: Memory usage comparison between different approaches
fn memory_comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Memory Usage Comparison ===");

    let sizes = [1000, 5000, 10000];
    let memory_limits = [16, 32, 64]; // MB

    println!("Comparing memory efficiency across problem sizes and memory limits:");
    println!();

    for &n in &sizes {
        println!("Problem size: {} variables", n);

        let simple_quadratic = |x: &ArrayView1<f64>| -> f64 { x.mapv(|xi| xi.powi(2)).sum() };

        for &mem_limit in &memory_limits {
            let memory_options = create_memory_efficient_optimizer(n, mem_limit);
            let x0 = Array1::ones(n);

            let start = Instant::now();
            let result = minimize_memory_efficient_lbfgs(&simple_quadratic, x0, &memory_options);
            let duration = start.elapsed();

            match result {
                Ok(res) => {
                    println!(
                        "  {}MB memory: {} iterations, {:.3}s, f = {:.2e}",
                        mem_limit,
                        res.iterations,
                        duration.as_secs_f64(),
                        res.fun.into()
                    );
                }
                Err(e) => {
                    println!("  {}MB memory: Failed - {}", mem_limit, e);
                }
            }
        }
        println!();
    }

    Ok(())
}

/// Helper function to demonstrate memory efficiency vs accuracy trade-offs
#[allow(dead_code)]
fn analyze_memory_vs_accuracy() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Memory vs Accuracy Analysis ===");

    let n = 10000;
    let target_solution = Array1::from_shape_fn(n, |i| (i as f64) / (n as f64));

    let quadratic = |x: &ArrayView1<f64>| -> f64 {
        target_solution
            .iter()
            .zip(x.iter())
            .map(|(&ti, &xi)| (xi - ti).powi(2))
            .sum()
    };

    let chunk_sizes = [64, 256, 1024, 4096];
    let history_sizes = [5, 10, 20];

    println!("Analyzing accuracy vs memory trade-offs:");
    println!("Target: variable i should equal i/n");
    println!();

    for &chunk_size in &chunk_sizes {
        for &max_history in &history_sizes {
            let mut memory_options = MemoryOptions::default();
            memory_options.chunk_size = chunk_size;
            memory_options.max_history = max_history;

            let x0 = Array1::zeros(n);
            let start = Instant::now();

            if let Ok(result) = minimize_memory_efficient_lbfgs(quadratic, x0, &memory_options) {
                let duration = start.elapsed();

                // Compute accuracy
                let mut total_error = 0.0;
                for i in 0..n {
                    let target = (i as f64) / (n as f64);
                    total_error += (result.x[i] - target).abs();
                }
                let avg_error = total_error / n as f64;

                println!(
                    "Chunk: {:4}, History: {:2} => Error: {:.2e}, Time: {:.3}s, Iters: {}",
                    chunk_size,
                    max_history,
                    avg_error,
                    duration.as_secs_f64(),
                    result.iterations
                );
            }
        }
    }

    Ok(())
}
