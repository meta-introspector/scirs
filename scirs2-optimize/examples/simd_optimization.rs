//! SIMD-accelerated optimization example
//!
//! This example demonstrates the performance benefits of SIMD-accelerated
//! optimization algorithms for problems with many variables.

use ndarray::{Array1, ArrayView1};
use scirs2_optimize::simd_ops::{SimdConfig, SimdVectorOps};
use scirs2_optimize::unconstrained::{minimize_simd_bfgs, Options, SimdBfgsOptions};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SIMD-Accelerated Optimization Examples\n");

    // Detect SIMD capabilities
    simd_capabilities_demo();

    // Performance comparison
    performance_comparison_demo()?;

    // High-dimensional optimization
    high_dimensional_demo()?;

    // SIMD vector operations
    simd_vector_operations_demo();

    Ok(())
}

/// Demonstrate SIMD capability detection
fn simd_capabilities_demo() {
    println!("=== SIMD Capabilities Detection ===");

    let config = SimdConfig::detect();
    println!("CPU SIMD Features:");
    println!("  AVX2 Support: {}", config.avx2_available);
    println!("  SSE4.1 Support: {}", config.sse41_available);
    println!("  FMA Support: {}", config.fma_available);
    println!("  Vector Width: {} elements", config.vector_width);
    println!("  SIMD Available: {}", config.has_simd());

    if config.has_simd() {
        println!("✓ SIMD acceleration is available on this system");
    } else {
        println!("⚠ SIMD acceleration is not available - using scalar fallback");
    }
    println!();
}

/// Compare performance of SIMD vs scalar optimization
fn performance_comparison_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Performance Comparison: SIMD vs Scalar ===");

    // High-dimensional quadratic function
    let dimensions = vec![50, 100, 200, 500];

    for &n in &dimensions {
        println!("Problem size: {} variables", n);

        // Create a quadratic function with some cross terms for complexity
        let quadratic_fun = move |x: &ArrayView1<f64>| {
            let mut sum = 0.0;
            for i in 0..x.len() {
                sum += (i + 1) as f64 * x[i].powi(2);
                if i < x.len() - 1 {
                    sum += 0.1 * x[i] * x[i + 1]; // Cross terms
                }
            }
            sum
        };

        let x0 = Array1::from_shape_fn(n, |i| (i as f64 + 1.0) * 0.1);

        // Test regular BFGS
        let start = Instant::now();
        let options = Options {
            max_iter: 1000,
            gtol: 1e-8,
            ..Default::default()
        };
        let regular_result =
            scirs2_optimize::unconstrained::minimize_bfgs(quadratic_fun, x0.clone(), &options)?;
        let regular_time = start.elapsed();

        // Test SIMD BFGS
        let start = Instant::now();
        let simd_result = minimize_simd_bfgs(
            quadratic_fun,
            x0.clone(),
            Some(SimdBfgsOptions {
                base_options: Options {
                    max_iter: 1000,
                    gtol: 1e-8,
                    ..Default::default()
                },
                force_simd: true,
                ..Default::default()
            }),
        )?;
        let simd_time = start.elapsed();

        // Compare results
        let speedup = regular_time.as_millis() as f64 / simd_time.as_millis() as f64;

        println!(
            "  Regular BFGS: {:.2} ms, {} iterations, final cost: {:.2e}",
            regular_time.as_millis(),
            regular_result.iterations,
            regular_result.fun
        );
        println!(
            "  SIMD BFGS:   {:.2} ms, {} iterations, final cost: {:.2e}",
            simd_time.as_millis(),
            simd_result.iterations,
            simd_result.fun
        );
        println!("  Speedup: {:.1}x", speedup);

        // Verify solutions are similar
        let solution_diff = (&regular_result.x - &simd_result.x).mapv(|x| x.abs()).sum();
        println!("  Solution difference: {:.2e}", solution_diff);
        println!();
    }

    Ok(())
}

/// Demonstrate optimization on high-dimensional problems
fn high_dimensional_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== High-Dimensional Optimization ===");

    let n = 1000; // 1000-dimensional problem
    println!("Optimizing {}-dimensional Rosenbrock function", n);

    // Extended Rosenbrock function
    let rosenbrock = |x: &ArrayView1<f64>| {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            let a = 1.0 - x[i];
            let b = x[i + 1] - x[i].powi(2);
            sum += a.powi(2) + 100.0 * b.powi(2);
        }
        sum
    };

    // Start from a point far from the optimum
    let x0 = Array1::from_shape_fn(n, |i| if i % 2 == 0 { -1.2 } else { 1.0 });

    let start = Instant::now();
    let result = minimize_simd_bfgs(
        rosenbrock,
        x0,
        Some(SimdBfgsOptions {
            base_options: Options {
                max_iter: 500,
                gtol: 1e-6,
                ftol: 1e-9,
                ..Default::default()
            },
            force_simd: true,
            ..Default::default()
        }),
    )?;
    let elapsed = start.elapsed();

    println!(
        "Optimization completed in {:.2} seconds",
        elapsed.as_secs_f64()
    );
    println!("  Iterations: {}", result.iterations);
    println!("  Function evaluations: {}", result.nfev);
    println!("  Final cost: {:.6e}", result.fun);
    println!("  Success: {}", result.success);

    // Check how close we got to the true minimum (all ones)
    let error_from_optimum = result.x.iter().map(|&xi| (xi - 1.0).abs()).sum::<f64>() / n as f64;
    println!("  Average error from optimum: {:.6e}", error_from_optimum);

    // Show convergence for first few variables
    println!("  First 10 variables:");
    for i in 0..10.min(n) {
        println!("    x[{}] = {:.6} (target: 1.0)", i, result.x[i]);
    }

    println!();

    Ok(())
}

/// Demonstrate SIMD vector operations
fn simd_vector_operations_demo() {
    println!("=== SIMD Vector Operations ===");

    let simd_ops = SimdVectorOps::new();
    let sizes = vec![8, 16, 64, 256, 1024];

    for &n in &sizes {
        println!("Vector size: {} elements", n);

        // Create test vectors
        let a = Array1::from_shape_fn(n, |i| i as f64);
        let b = Array1::from_shape_fn(n, |i| (i * 2) as f64);

        // Benchmark different operations
        let iterations = 10000;

        // Dot product
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = simd_ops.dot_product(&a.view(), &b.view());
        }
        let dot_time = start.elapsed();

        // Vector addition
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = simd_ops.add(&a.view(), &b.view());
        }
        let add_time = start.elapsed();

        // Vector scaling
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = simd_ops.scale(2.0, &a.view());
        }
        let scale_time = start.elapsed();

        // AXPY operation
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = simd_ops.axpy(2.0, &a.view(), &b.view());
        }
        let axpy_time = start.elapsed();

        // Vector norm
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = simd_ops.norm(&a.view());
        }
        let norm_time = start.elapsed();

        println!("  Operations ({} iterations):", iterations);
        println!(
            "    Dot product: {:.2} μs",
            dot_time.as_micros() as f64 / iterations as f64
        );
        println!(
            "    Addition:    {:.2} μs",
            add_time.as_micros() as f64 / iterations as f64
        );
        println!(
            "    Scaling:     {:.2} μs",
            scale_time.as_micros() as f64 / iterations as f64
        );
        println!(
            "    AXPY:        {:.2} μs",
            axpy_time.as_micros() as f64 / iterations as f64
        );
        println!(
            "    Norm:        {:.2} μs",
            norm_time.as_micros() as f64 / iterations as f64
        );
        println!();
    }
}

/// Helper function to create complex test functions
#[allow(dead_code)]
fn create_complex_quadratic(n: usize) -> impl Fn(&ArrayView1<f64>) -> f64 {
    move |x: &ArrayView1<f64>| {
        let mut sum = 0.0;

        // Diagonal terms with varying condition numbers
        for i in 0..n {
            let weight = 1.0 + (i as f64) / (n as f64) * 1000.0; // Condition number up to 1000
            sum += weight * x[i].powi(2);
        }

        // Off-diagonal coupling terms
        for i in 0..n - 1 {
            sum += 0.1 * x[i] * x[i + 1];
        }

        // Long-range coupling
        for i in 0..n / 2 {
            if i + n / 2 < n {
                sum += 0.01 * x[i] * x[i + n / 2];
            }
        }

        sum
    }
}

/// Helper function for creating ill-conditioned problems
#[allow(dead_code)]
fn create_ill_conditioned_quadratic(
    n: usize,
    condition_number: f64,
) -> impl Fn(&ArrayView1<f64>) -> f64 {
    move |x: &ArrayView1<f64>| {
        let mut sum = 0.0;

        // Create eigenvalues that span the condition number
        for i in 0..n {
            let lambda = 1.0 + (condition_number - 1.0) * (i as f64) / ((n - 1) as f64);
            sum += lambda * x[i].powi(2);
        }

        sum
    }
}
