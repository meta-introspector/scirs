//! JIT optimization example
//!
//! This example demonstrates how to use just-in-time compilation
//! and auto-vectorization to accelerate optimization.

use ndarray::{Array1, ArrayView1};
use scirs2_optimize::jit_optimization::{optimize_function, JitCompiler, JitOptions};
use scirs2_optimize::unconstrained::{minimize, Method, Options};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("JIT Optimization Examples\n");

    // Example 1: Basic JIT optimization
    basic_jit_example()?;

    // Example 2: Performance comparison
    performance_comparison_example()?;

    // Example 3: Function pattern detection
    pattern_detection_example()?;

    // Example 4: Memory-efficient optimization
    memory_efficient_example()?;

    Ok(())
}

/// Demonstrate basic JIT optimization
fn basic_jit_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic JIT Optimization ===");

    // Define a complex objective function
    let complex_function = |x: &ArrayView1<f64>| -> f64 {
        let mut sum = 0.0;
        let n = x.len();

        // Complex computation that can benefit from optimization
        for i in 0..n {
            sum += x[i].powi(4) - 16.0 * x[i].powi(2) + 5.0 * x[i];

            // Add some interactions
            if i > 0 {
                sum += x[i] * x[i - 1] * (i as f64).sin();
            }
        }

        sum
    };

    let n_vars = 10;

    // Create JIT options
    let jit_options = JitOptions {
        enable_jit: true,
        enable_vectorization: true,
        optimization_level: 3,
        enable_specialization: true,
        enable_caching: true,
        ..Default::default()
    };

    // Optimize the function with JIT
    let optimized_function = Arc::new(optimize_function(
        complex_function,
        n_vars,
        Some(jit_options),
    )?);

    // Use in optimization
    let initial_guess = Array1::ones(n_vars);
    let mut options = Options::default();
    options.max_iter = 100;

    let opt_fn = optimized_function.clone();
    let result = minimize(
        move |x| opt_fn(x),
        initial_guess.as_slice().unwrap(),
        Method::BFGS,
        Some(options),
    )?;

    println!("JIT-optimized solution found:");
    println!("  Success: {}", result.success);
    println!("  Iterations: {}", result.iterations);
    println!("  Function value: {:.6}", result.fun);
    println!(
        "  Solution norm: {:.6}",
        result.x.mapv(|x| x.powi(2)).sum().sqrt()
    );
    println!();

    Ok(())
}

/// Compare performance with and without JIT
fn performance_comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Performance Comparison ===");

    let n_vars = 100;

    // Define a computationally intensive function
    let intensive_function = |x: &ArrayView1<f64>| -> f64 {
        let mut result = 0.0;
        let n = x.len();

        // Simulate expensive computation
        for i in 0..n {
            let xi = x[i];
            result += xi.powi(2) * (2.0 * std::f64::consts::PI * xi).sin().exp();

            for j in (i + 1)..n.min(i + 5) {
                result += 0.1 * xi * x[j] * ((i + j) as f64).cos();
            }
        }

        result
    };

    let test_point = Array1::from_shape_fn(n_vars, |i| (i as f64 + 1.0) * 0.1);

    // Test without JIT
    let start = Instant::now();
    let mut total_without_jit = 0.0;
    let n_evaluations = 1000;

    for _ in 0..n_evaluations {
        total_without_jit += intensive_function(&test_point.view());
    }
    let time_without_jit = start.elapsed();

    // Test with JIT
    let jit_options = JitOptions {
        enable_jit: true,
        enable_vectorization: true,
        optimization_level: 3,
        ..Default::default()
    };

    let optimized_function = optimize_function(intensive_function, n_vars, Some(jit_options))?;

    let start = Instant::now();
    let mut total_with_jit = 0.0;

    for _ in 0..n_evaluations {
        total_with_jit += optimized_function(&test_point.view());
    }
    let time_with_jit = start.elapsed();

    println!("Performance comparison ({} evaluations):", n_evaluations);
    println!(
        "  Without JIT: {:.3} ms",
        time_without_jit.as_secs_f64() * 1000.0
    );
    println!(
        "  With JIT:    {:.3} ms",
        time_with_jit.as_secs_f64() * 1000.0
    );

    if time_without_jit > time_with_jit {
        let speedup = time_without_jit.as_secs_f64() / time_with_jit.as_secs_f64();
        println!("  Speedup:     {:.2}x", speedup);
    } else {
        println!("  Note: JIT overhead may dominate for simple functions");
    }

    // Verify correctness
    println!(
        "  Results match: {}",
        (total_without_jit - total_with_jit).abs() < 1e-10
    );
    println!();

    Ok(())
}

/// Demonstrate function pattern detection
fn pattern_detection_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Function Pattern Detection ===");

    let mut compiler = JitCompiler::new(JitOptions::default());

    // Test different function types
    let functions: Vec<(&str, Box<dyn Fn(&ArrayView1<f64>) -> f64 + Send + Sync>)> = vec![
        (
            "Quadratic",
            Box::new(|x: &ArrayView1<f64>| {
                x[0] * x[0] + 2.0 * x[0] * x[1] + x[1] * x[1] + x[0] + x[1]
            }),
        ),
        (
            "Sum of squares",
            Box::new(|x: &ArrayView1<f64>| x.iter().map(|&xi| xi * xi).sum::<f64>()),
        ),
        (
            "Separable",
            Box::new(|x: &ArrayView1<f64>| {
                x.iter().map(|&xi| xi.powi(4) - xi.powi(2)).sum::<f64>()
            }),
        ),
        (
            "Polynomial (degree 3)",
            Box::new(|x: &ArrayView1<f64>| {
                x.iter()
                    .map(|&xi| xi.powi(3) + 2.0 * xi.powi(2) + xi + 1.0)
                    .sum::<f64>()
            }),
        ),
        (
            "General (transcendental)",
            Box::new(|x: &ArrayView1<f64>| {
                x.iter().map(|&xi| xi.sin() + xi.cos().exp()).sum::<f64>()
            }),
        ),
    ];

    for (name, function) in functions {
        let n_vars = 5;
        let compile_start = Instant::now();

        let compiled = compiler.compile_function(function, n_vars)?;
        let compile_time = compile_start.elapsed();

        println!("Function: {}", name);
        println!("  Pattern detected: {:?}", compiled.pattern);
        println!(
            "  Compile time: {:.3} ms",
            compile_time.as_secs_f64() * 1000.0
        );
        println!("  Vectorized: {}", compiled.metadata.is_vectorized);
        println!(
            "  Optimization flags: {:?}",
            compiled.metadata.optimization_flags
        );

        // Test the compiled function
        let test_point = Array1::from_shape_fn(n_vars, |i| (i as f64) * 0.5);
        let _result = (compiled.implementation)(&test_point.view());

        println!();
    }

    // Show compiler statistics
    let stats = compiler.get_stats();
    println!("Compiler statistics:");
    println!("  Total compiled: {}", stats.total_compiled);
    println!("  Total compile time: {} ms", stats.total_compile_time_ms);
    println!();

    Ok(())
}

/// Demonstrate memory-efficient optimization for large problems
fn memory_efficient_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Memory-Efficient Optimization ===");

    let n_vars = 1000; // Large problem

    // Create a large-scale optimization problem
    let large_scale_function = |x: &ArrayView1<f64>| -> f64 {
        let n = x.len();
        let mut sum = 0.0;

        // Use chunked processing for efficiency
        let chunk_size = 100;
        for chunk_start in (0..n).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n);
            let mut chunk_sum = 0.0;

            for i in chunk_start..chunk_end {
                chunk_sum += x[i] * x[i] + 0.01 * i as f64 * x[i];

                // Add some non-separable terms (but limited scope)
                if i > 0 && i < chunk_end {
                    chunk_sum += 0.001 * x[i - 1] * x[i];
                }
            }

            sum += chunk_sum;
        }

        sum
    };

    let jit_options = JitOptions {
        enable_jit: true,
        enable_vectorization: true,
        enable_specialization: true,
        max_cache_size: 10, // Limit cache for memory efficiency
        ..Default::default()
    };

    println!("Optimizing large-scale problem ({} variables)", n_vars);

    let compile_start = Instant::now();
    let optimized_function = Arc::new(optimize_function(
        large_scale_function,
        n_vars,
        Some(jit_options),
    )?);
    let compile_time = compile_start.elapsed();

    println!(
        "JIT compilation time: {:.3} ms",
        compile_time.as_secs_f64() * 1000.0
    );

    // Test optimization
    let initial_guess = Array1::ones(n_vars);
    let mut options = Options::default();
    options.max_iter = 50; // Limit iterations for demo
    options.gtol = 1e-4; // Relaxed tolerance for large problem

    let opt_start = Instant::now();
    let opt_fn = optimized_function.clone();
    let result = minimize(
        move |x| opt_fn(x),
        initial_guess.as_slice().unwrap(),
        Method::LBFGS, // Good for large-scale problems
        Some(options),
    )?;
    let opt_time = opt_start.elapsed();

    println!("Optimization results:");
    println!("  Success: {}", result.success);
    println!("  Iterations: {}", result.iterations);
    println!("  Function evaluations: {}", result.func_evals);
    println!(
        "  Optimization time: {:.3} ms",
        opt_time.as_secs_f64() * 1000.0
    );
    println!("  Final function value: {:.6}", result.fun);
    println!(
        "  Solution norm: {:.6}",
        result.x.mapv(|x| x.powi(2)).sum().sqrt()
    );

    // Check solution quality
    let grad_norm = result.x.mapv(|x| (2.0 * x).abs()).sum() / n_vars as f64;
    println!("  Average gradient magnitude: {:.6}", grad_norm);
    println!();

    Ok(())
}

/// Helper function to demonstrate vector operations
#[allow(dead_code)]
fn demonstrate_vectorization() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vectorization Demo ===");

    let n = 10000;
    let x = Array1::from_shape_fn(n, |i| (i as f64) * 0.001);

    // Scalar version
    let scalar_fn = |x: &ArrayView1<f64>| -> f64 {
        let mut sum = 0.0;
        for &xi in x.iter() {
            sum += xi * xi + xi.sin();
        }
        sum
    };

    // Potentially vectorized version
    let vectorized_fn = |x: &ArrayView1<f64>| -> f64 {
        // Use ndarray operations that can be vectorized
        let squares = x.mapv(|xi| xi * xi);
        let sines = x.mapv(|xi| xi.sin());
        squares.sum() + sines.sum()
    };

    let start = Instant::now();
    let scalar_result = scalar_fn(&x.view());
    let scalar_time = start.elapsed();

    let start = Instant::now();
    let vectorized_result = vectorized_fn(&x.view());
    let vectorized_time = start.elapsed();

    println!("Vectorization comparison ({} elements):", n);
    println!(
        "  Scalar version:     {:.3} ms",
        scalar_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Vectorized version: {:.3} ms",
        vectorized_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Results match: {}",
        (scalar_result - vectorized_result).abs() < 1e-10
    );

    if scalar_time > vectorized_time {
        let speedup = scalar_time.as_secs_f64() / vectorized_time.as_secs_f64();
        println!("  Speedup: {:.2}x", speedup);
    }

    Ok(())
}
