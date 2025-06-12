//! Efficient sparse optimization example
//!
//! This example demonstrates efficient handling of sparse Jacobians and Hessians
//! in optimization problems with advanced sparsity detection and specialized algorithms.

use ndarray::{Array1, ArrayView1};
use scirs2_optimize::unconstrained::{minimize_efficient_sparse_newton, EfficientSparseOptions};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Efficient Sparse Optimization Examples\n");

    // Example 1: Simple sparse quadratic optimization
    simple_sparse_example()?;

    // Example 2: Large-scale sparse optimization
    large_scale_sparse_example()?;

    // Example 3: Adaptive sparsity detection
    adaptive_sparsity_example()?;

    // Example 4: Comparison with dense methods
    sparse_vs_dense_comparison()?;

    Ok(())
}

/// Example 1: Simple sparse quadratic optimization
fn simple_sparse_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Simple Sparse Quadratic Optimization ===");

    let n = 100;
    println!("Problem size: {} variables", n);

    // Sparse quadratic function: only every 10th variable contributes
    let fun = |x: &ArrayView1<f64>| -> f64 {
        let mut sum = 0.0;
        for i in (0..x.len()).step_by(10) {
            sum += x[i].powi(2);
        }
        sum
    };

    let grad = |x: &ArrayView1<f64>| -> Array1<f64> {
        let mut g = Array1::zeros(x.len());
        for i in (0..x.len()).step_by(10) {
            g[i] = 2.0 * x[i];
        }
        g
    };

    let x0 = Array1::ones(n);
    let options = EfficientSparseOptions::default();

    println!("Configuration:");
    println!("  Auto-detect sparsity: {}", options.auto_detect_sparsity);
    println!("  Sparsity threshold: {:.1e}", options.sparsity_threshold);
    println!("  Use sparse Hessian: {}", options.use_sparse_hessian);

    let start = Instant::now();
    let result = minimize_efficient_sparse_newton(fun, grad, x0, &options)?;
    let duration = start.elapsed();

    println!("Results:");
    println!("  Success: {}", result.success);
    println!("  Iterations: {}", result.iterations);
    println!("  Function evaluations: {}", result.func_evals);
    println!("  Final function value: {:.6e}", result.fun);
    println!("  Optimization time: {:.3} ms", duration.as_millis());

    // Check solution quality - only every 10th variable should be zero
    let mut active_errors = Vec::new();
    let mut inactive_values = Vec::new();

    for i in 0..n {
        if i % 10 == 0 {
            active_errors.push(result.x[i].abs());
        } else {
            inactive_values.push(result.x[i].abs());
        }
    }

    let max_active_error = active_errors.iter().fold(0.0f64, |a, &b| a.max(b));
    let max_inactive_value = inactive_values.iter().fold(0.0f64, |a, &b| a.max(b));

    println!("  Max error in active variables: {:.6e}", max_active_error);
    println!(
        "  Max value in inactive variables: {:.6e}",
        max_inactive_value
    );
    println!();

    Ok(())
}

/// Example 2: Large-scale sparse optimization
fn large_scale_sparse_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Large-Scale Sparse Optimization ===");

    let n = 5000;
    println!("Problem size: {} variables", n);

    // Banded sparse structure: each variable interacts only with nearby variables
    let fun = |x: &ArrayView1<f64>| -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() {
            // Diagonal term
            sum += x[i].powi(2);

            // Interaction with neighbors (creates sparse structure)
            if i > 0 {
                sum += 0.1 * x[i] * x[i - 1];
            }
            if i < x.len() - 1 {
                sum += 0.1 * x[i] * x[i + 1];
            }
        }
        sum
    };

    let grad = |x: &ArrayView1<f64>| -> Array1<f64> {
        let mut g = Array1::zeros(x.len());
        for i in 0..x.len() {
            // Diagonal gradient
            g[i] = 2.0 * x[i];

            // Neighbor interactions
            if i > 0 {
                g[i] += 0.1 * x[i - 1];
                g[i - 1] += 0.1 * x[i];
            }
            if i < x.len() - 1 {
                g[i] += 0.1 * x[i + 1];
            }
        }
        g
    };

    let mut options = EfficientSparseOptions::default();
    options.auto_detect_sparsity = true;
    options.adaptive_sparsity = true;
    options.sparse_percentage_threshold = 0.05; // Very sparse threshold
    options.max_sparsity_iterations = 3;

    // Limit iterations for demo
    options.base_options.max_iter = 100;

    println!("Large-scale configuration:");
    println!(
        "  Sparse percentage threshold: {:.1}%",
        options.sparse_percentage_threshold * 100.0
    );
    println!("  Adaptive sparsity: {}", options.adaptive_sparsity);
    println!("  Parallel sparse ops: {}", options.parallel_sparse_ops);

    let x0 = Array1::ones(n) * 0.1;

    let start = Instant::now();
    let result = minimize_efficient_sparse_newton(fun, grad, x0, &options)?;
    let duration = start.elapsed();

    println!("Results:");
    println!("  Success: {}", result.success);
    println!("  Iterations: {}", result.iterations);
    println!("  Function evaluations: {}", result.func_evals);
    println!("  Final function value: {:.6e}", result.fun);
    println!("  Optimization time: {:.3} ms", duration.as_millis());

    // Analyze solution structure
    let solution_norm = result.x.mapv(|x| x.abs()).sum() / n as f64;
    let max_abs_value = result.x.mapv(|x| x.abs()).fold(0.0f64, |a, b| a.max(b));

    println!("  Average |x_i|: {:.6e}", solution_norm);
    println!("  Max |x_i|: {:.6e}", max_abs_value);
    println!();

    Ok(())
}

/// Example 3: Adaptive sparsity detection
fn adaptive_sparsity_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Adaptive Sparsity Detection ===");

    let n = 1000;
    println!("Problem size: {} variables", n);

    // Function with changing sparsity pattern during optimization
    let fun = |x: &ArrayView1<f64>| -> f64 {
        let mut sum = 0.0;

        // Primary sparse structure: every 5th variable
        for i in (0..x.len()).step_by(5) {
            sum += x[i].powi(2);
        }

        // Secondary sparse structure: becomes active for larger values
        for i in 0..x.len() {
            if x[i].abs() > 0.1 {
                sum += 0.01 * x[i].powi(4);
            }
        }

        sum
    };

    let grad = |x: &ArrayView1<f64>| -> Array1<f64> {
        let mut g = Array1::zeros(x.len());

        // Primary gradient
        for i in (0..x.len()).step_by(5) {
            g[i] = 2.0 * x[i];
        }

        // Secondary gradient
        for i in 0..x.len() {
            if x[i].abs() > 0.1 {
                g[i] += 0.04 * x[i].powi(3);
            }
        }

        g
    };

    let mut options = EfficientSparseOptions::default();
    options.auto_detect_sparsity = true;
    options.adaptive_sparsity = true;
    options.max_sparsity_iterations = 5;
    options.sparsity_threshold = 1e-10;

    // More aggressive sparsity detection
    options.sparse_percentage_threshold = 0.2;

    println!("Adaptive sparsity configuration:");
    println!(
        "  Max sparsity iterations: {}",
        options.max_sparsity_iterations
    );
    println!("  Sparsity threshold: {:.1e}", options.sparsity_threshold);

    let x0 = Array1::from_shape_fn(n, |i| {
        if i % 5 == 0 {
            0.5
        } else {
            0.01
        } // Start with mixed sparsity
    });

    let start = Instant::now();
    let result = minimize_efficient_sparse_newton(fun, grad, x0, &options)?;
    let duration = start.elapsed();

    println!("Results:");
    println!("  Success: {}", result.success);
    println!("  Iterations: {}", result.iterations);
    println!("  Function evaluations: {}", result.func_evals);
    println!("  Final function value: {:.6e}", result.fun);
    println!("  Optimization time: {:.3} ms", duration.as_millis());

    // Analyze final sparsity
    let nnz_count = result.x.iter().filter(|&&x| x.abs() > 1e-8).count();
    let final_sparsity = nnz_count as f64 / n as f64;

    println!(
        "  Final sparsity: {:.1}% ({} non-zeros)",
        final_sparsity * 100.0,
        nnz_count
    );
    println!();

    Ok(())
}

/// Example 4: Comparison between sparse and dense methods
fn sparse_vs_dense_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Sparse vs Dense Method Comparison ===");

    let problem_sizes = [100, 500, 1000];
    let sparsity_levels = [0.05, 0.1, 0.2]; // 5%, 10%, 20% non-zero

    println!("Comparing sparse vs dense optimization efficiency:\n");

    for &n in &problem_sizes {
        println!("Problem size: {} variables", n);

        for &sparsity in &sparsity_levels {
            let active_vars: Vec<usize> = (0..n)
                .filter(|&i| (i as f64 / n as f64) < sparsity)
                .collect();

            println!(
                "  Sparsity level: {:.1}% ({} active variables)",
                sparsity * 100.0,
                active_vars.len()
            );

            // Create sparse problem
            let fun =
                |x: &ArrayView1<f64>| -> f64 { active_vars.iter().map(|&i| x[i].powi(2)).sum() };

            let grad = |x: &ArrayView1<f64>| -> Array1<f64> {
                let mut g = Array1::zeros(x.len());
                for &i in &active_vars {
                    g[i] = 2.0 * x[i];
                }
                g
            };

            let x0 = Array1::ones(n) * 0.1;

            // Test sparse method
            let mut sparse_options = EfficientSparseOptions::default();
            sparse_options.auto_detect_sparsity = true;
            sparse_options.sparse_percentage_threshold = sparsity + 0.05;
            sparse_options.base_options.max_iter = 50; // Limit for comparison

            let start = Instant::now();
            let sparse_result =
                minimize_efficient_sparse_newton(fun, grad, x0.clone(), &sparse_options);
            let sparse_time = start.elapsed();

            match sparse_result {
                Ok(result) => {
                    println!(
                        "    Sparse method: {} iter, {:.2} ms, f = {:.2e}",
                        result.iterations,
                        sparse_time.as_millis(),
                        result.fun
                    );
                }
                Err(e) => {
                    println!("    Sparse method: Failed - {}", e);
                }
            }

            // For comparison, we would test a dense method here
            // but that would require implementing a dense Newton method
            println!("    Dense method: [comparison would go here]");
        }
        println!();
    }

    Ok(())
}

/// Helper function to analyze sparsity patterns
#[allow(dead_code)]
fn analyze_sparsity_pattern(x: &Array1<f64>, threshold: f64) -> (usize, f64, Vec<usize>) {
    let n = x.len();
    let mut non_zero_indices = Vec::new();

    for (i, &value) in x.iter().enumerate() {
        if value.abs() > threshold {
            non_zero_indices.push(i);
        }
    }

    let nnz = non_zero_indices.len();
    let sparsity = nnz as f64 / n as f64;

    (nnz, sparsity, non_zero_indices)
}

/// Demonstration of sparsity pattern evolution
#[allow(dead_code)]
fn demonstrate_sparsity_evolution() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Sparsity Pattern Evolution ===");

    let n = 200;

    // Function where sparsity changes during optimization
    let fun = |x: &ArrayView1<f64>| -> f64 {
        let mut sum = 0.0;

        // Initial sparse structure
        for i in (0..n).step_by(10) {
            sum += x[i].powi(2);
        }

        // Coupling terms that activate gradually
        for i in 0..(n - 1) {
            if x[i].abs() > 0.05 && x[i + 1].abs() > 0.05 {
                sum += 0.1 * x[i] * x[i + 1];
            }
        }

        sum
    };

    let grad = |x: &ArrayView1<f64>| -> Array1<f64> {
        let mut g = Array1::zeros(n);

        for i in (0..n).step_by(10) {
            g[i] = 2.0 * x[i];
        }

        for i in 0..(n - 1) {
            if x[i].abs() > 0.05 && x[i + 1].abs() > 0.05 {
                g[i] += 0.1 * x[i + 1];
                g[i + 1] += 0.1 * x[i];
            }
        }

        g
    };

    let mut options = EfficientSparseOptions::default();
    options.adaptive_sparsity = true;
    options.auto_detect_sparsity = true;
    options.base_options.max_iter = 20;

    let x0 = Array1::from_shape_fn(n, |i| if i % 10 == 0 { 0.2 } else { 0.01 });

    println!("Tracking sparsity evolution during optimization...");
    println!("Initial sparsity pattern analysis:");

    let (initial_nnz, initial_sparsity, _) = analyze_sparsity_pattern(&x0, 1e-6);
    println!(
        "  Initial: {} non-zeros, {:.1}% sparse",
        initial_nnz,
        initial_sparsity * 100.0
    );

    let result = minimize_efficient_sparse_newton(fun, grad, x0, &options)?;

    let (final_nnz, final_sparsity, _) = analyze_sparsity_pattern(&result.x, 1e-6);
    println!(
        "  Final: {} non-zeros, {:.1}% sparse",
        final_nnz,
        final_sparsity * 100.0
    );

    println!("Optimization completed in {} iterations", result.iterations);

    Ok(())
}
