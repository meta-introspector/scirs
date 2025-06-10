//! Automatic differentiation example
//!
//! This example demonstrates the automatic differentiation capabilities for exact
//! gradient and Hessian computation in optimization problems.

use ndarray::{Array1, ArrayView1};
use scirs2_optimize::automatic_differentiation::{
    autodiff, create_ad_gradient, create_ad_hessian, optimize_ad_mode, ADMode, AutoDiffOptions,
};
use scirs2_optimize::unconstrained::{minimize, Method, Options};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Automatic Differentiation Examples\n");

    // Example 1: Basic gradient computation
    basic_gradient_example()?;

    // Example 2: Hessian computation
    hessian_computation_example()?;

    // Example 3: AD mode comparison
    ad_mode_comparison_example()?;

    // Example 4: Optimization with AD gradients
    optimization_with_ad_example()?;

    // Example 5: Complex function with multiple operations
    complex_function_example()?;

    // Example 6: Performance comparison
    performance_comparison_example()?;

    Ok(())
}

/// Example 1: Basic gradient computation using automatic differentiation
fn basic_gradient_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic Gradient Computation ===");

    // Test function: f(x, y) = x² + xy + 2y²
    let func = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[0] * x[1] + 2.0 * x[1] * x[1] };

    let x = Array1::from_vec(vec![1.0, 2.0]);

    println!("Function: f(x, y) = x² + xy + 2y²");
    println!("Point: x = [{:.1}, {:.1}]", x[0], x[1]);
    println!("Function value: {:.1}", func(&x.view()));

    // Test both forward and reverse modes
    let modes = [
        ("Forward", ADMode::Forward),
        ("Reverse", ADMode::Reverse),
        ("Auto", ADMode::Auto),
    ];

    for (name, mode) in &modes {
        let mut options = AutoDiffOptions::default();
        options.mode = *mode;
        options.forward_options.compute_gradient = true;
        options.reverse_options.compute_gradient = true;

        let start = Instant::now();
        let result = autodiff(func, &x.view(), &options)?;
        let duration = start.elapsed();

        if let Some(grad) = result.gradient {
            println!("\n{} mode:", name);
            println!("  Gradient: [{:.3}, {:.3}]", grad[0], grad[1]);
            println!("  Function evaluations: {}", result.n_fev);
            println!("  Time: {:.2} μs", duration.as_micros());

            // Analytical gradient: ∂f/∂x = 2x + y, ∂f/∂y = x + 4y
            let analytical = Array1::from_vec(vec![
                2.0 * x[0] + x[1], // 2(1) + 2 = 4
                x[0] + 4.0 * x[1], // 1 + 4(2) = 9
            ]);
            println!("  Analytical: [{:.3}, {:.3}]", analytical[0], analytical[1]);
            println!(
                "  Error: [{:.2e}, {:.2e}]",
                (grad[0] - analytical[0]).abs(),
                (grad[1] - analytical[1]).abs()
            );
        }
    }

    println!();
    Ok(())
}

/// Example 2: Hessian computation
fn hessian_computation_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Hessian Computation ===");

    // Test function: f(x, y) = x² + xy + 2y²
    let func = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[0] * x[1] + 2.0 * x[1] * x[1] };

    let x = Array1::from_vec(vec![1.0, 2.0]);

    println!("Function: f(x, y) = x² + xy + 2y²");
    println!("Point: x = [{:.1}, {:.1}]", x[0], x[1]);

    // Test Hessian computation
    let mut options = AutoDiffOptions::default();
    options.compute_hessian = true;
    options.forward_options.compute_hessian = true;
    options.reverse_options.compute_hessian = true;

    let start = Instant::now();
    let result = autodiff(func, &x.view(), &options)?;
    let duration = start.elapsed();

    if let Some(hess) = result.hessian {
        println!("\nHessian matrix:");
        println!("  [{:.3}, {:.3}]", hess[[0, 0]], hess[[0, 1]]);
        println!("  [{:.3}, {:.3}]", hess[[1, 0]], hess[[1, 1]]);
        println!("Time: {:.2} μs", duration.as_micros());

        // Analytical Hessian:
        // ∂²f/∂x² = 2, ∂²f/∂x∂y = 1
        // ∂²f/∂y∂x = 1, ∂²f/∂y² = 4
        println!("\nAnalytical Hessian:");
        println!("  [2.000, 1.000]");
        println!("  [1.000, 4.000]");

        println!("\nErrors:");
        println!(
            "  [{:.2e}, {:.2e}]",
            (hess[[0, 0]] - 2.0).abs(),
            (hess[[0, 1]] - 1.0).abs()
        );
        println!(
            "  [{:.2e}, {:.2e}]",
            (hess[[1, 0]] - 1.0).abs(),
            (hess[[1, 1]] - 4.0).abs()
        );
    }

    println!();
    Ok(())
}

/// Example 3: AD mode comparison for different problem sizes
fn ad_mode_comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== AD Mode Comparison ===");

    let problem_sizes = [2, 5, 10, 20, 50];

    println!("Optimal AD mode selection for different problem sizes:");
    println!("Size | Recommended | Input Dim | Output Dim | Mode Selected");
    println!("-----|-------------|-----------|------------|---------------");

    for &n in &problem_sizes {
        let output_dim = 1; // Optimization problems have scalar output
        let recommended = optimize_ad_mode(n, output_dim, 0.1);

        let mode_str = match recommended {
            ADMode::Forward => "Forward",
            ADMode::Reverse => "Reverse",
            ADMode::Auto => "Auto",
        };

        let efficiency_reason = if n <= 5 {
            "(small input)"
        } else if n <= 20 {
            "(medium input)"
        } else {
            "(large input)"
        };

        println!(
            "{:4} | {:<11} | {:9} | {:10} | {} {}",
            n, mode_str, n, output_dim, mode_str, efficiency_reason
        );
    }

    println!("\nSparsity impact:");
    let sparsity_levels = [0.1, 0.5, 0.8, 0.95];

    for &sparsity in &sparsity_levels {
        let mode_sparse = optimize_ad_mode(50, 1, sparsity);
        let mode_str = match mode_sparse {
            ADMode::Forward => "Forward",
            ADMode::Reverse => "Reverse",
            ADMode::Auto => "Auto",
        };
        println!("  Sparsity {:.0}% → {} mode", sparsity * 100.0, mode_str);
    }

    println!();
    Ok(())
}

/// Example 4: Optimization using AD gradients
fn optimization_with_ad_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Optimization with AD Gradients ===");

    // Rosenbrock function
    let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
        let a = 1.0;
        let b = 100.0;
        (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
    };

    println!("Function: Rosenbrock (a=1, b=100)");
    println!("Global minimum: (1, 1) with f = 0");

    let x0 = vec![-1.2, 1.0];
    println!("Starting point: [{:.1}, {:.1}]", x0[0], x0[1]);

    // Create AD gradient function
    let ad_options = AutoDiffOptions::default();
    let ad_gradient = create_ad_gradient(rosenbrock, ad_options);

    // Test the AD gradient at starting point
    let x0_array = Array1::from_vec(x0.clone());
    let grad_at_start = ad_gradient(&x0_array.view());
    println!(
        "AD gradient at start: [{:.3}, {:.3}]",
        grad_at_start[0], grad_at_start[1]
    );

    // Run optimization with different methods
    let methods = [
        ("BFGS", Method::BFGS),
        ("L-BFGS", Method::LBFGS),
        ("Newton-CG", Method::NewtonCG),
    ];

    for (name, method) in &methods {
        let mut options = Options::default();
        options.max_iter = 100;
        options.gtol = 1e-6;

        let start = Instant::now();
        let result = minimize(rosenbrock, &x0, *method, Some(options));
        let duration = start.elapsed();

        match result {
            Ok(res) => {
                println!("\n{} optimization:", name);
                println!("  Solution: [{:.6}, {:.6}]", res.x[0], res.x[1]);
                println!("  Function value: {:.2e}", res.fun);
                println!("  Iterations: {}", res.iterations);
                println!("  Function evals: {}", res.func_evals);
                println!("  Success: {}", res.success);
                println!("  Time: {:.2} ms", duration.as_millis());

                // Calculate error from true optimum
                let error = ((res.x[0] - 1.0).powi(2) + (res.x[1] - 1.0).powi(2)).sqrt();
                println!("  Distance from optimum: {:.2e}", error);
            }
            Err(e) => {
                println!("\n{} optimization failed: {}", name, e);
            }
        }
    }

    println!();
    Ok(())
}

/// Example 5: Complex function with multiple operations
fn complex_function_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Complex Function with Multiple Operations ===");

    // Complex function with transcendental operations
    let complex_func = |x: &ArrayView1<f64>| -> f64 {
        let x1 = x[0];
        let x2 = x[1];

        // f(x₁, x₂) = sin(x₁) * exp(x₂) + x₁²*ln(x₂² + 1) + cos(x₁ + x₂)
        x1.sin() * x2.exp() + x1.powi(2) * (x2.powi(2) + 1.0).ln() + (x1 + x2).cos()
    };

    let x = Array1::from_vec(vec![0.5, 1.0]);

    println!("Function: f(x₁, x₂) = sin(x₁)·exp(x₂) + x₁²·ln(x₂²+1) + cos(x₁+x₂)");
    println!("Point: x = [{:.1}, {:.1}]", x[0], x[1]);
    println!("Function value: {:.6}", complex_func(&x.view()));

    // Compute gradient using AD
    let mut options = AutoDiffOptions::default();
    options.mode = ADMode::Reverse; // Better for scalar output
    options.reverse_options.compute_gradient = true;

    let start = Instant::now();
    let result = autodiff(complex_func, &x.view(), &options)?;
    let duration = start.elapsed();

    if let Some(grad) = result.gradient {
        println!("\nAD gradient: [{:.6}, {:.6}]", grad[0], grad[1]);
        println!("Time: {:.2} μs", duration.as_micros());
        println!("Function evaluations: {}", result.n_fev);

        // Compare with numerical gradient
        let h = 1e-8;
        let mut x_plus = x.clone();
        x_plus[0] += h;
        let f_plus_x = complex_func(&x_plus.view());

        let mut x_minus = x.clone();
        x_minus[0] -= h;
        let f_minus_x = complex_func(&x_minus.view());

        let numerical_grad_x = (f_plus_x - f_minus_x) / (2.0 * h);

        x_plus = x.clone();
        x_plus[1] += h;
        let f_plus_y = complex_func(&x_plus.view());

        x_minus = x.clone();
        x_minus[1] -= h;
        let f_minus_y = complex_func(&x_minus.view());

        let numerical_grad_y = (f_plus_y - f_minus_y) / (2.0 * h);

        println!(
            "Numerical gradient: [{:.6}, {:.6}]",
            numerical_grad_x, numerical_grad_y
        );
        println!(
            "Absolute error: [{:.2e}, {:.2e}]",
            (grad[0] - numerical_grad_x).abs(),
            (grad[1] - numerical_grad_y).abs()
        );
    }

    println!();
    Ok(())
}

/// Example 6: Performance comparison between AD and numerical differentiation
fn performance_comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Performance Comparison ===");

    let problem_sizes = [5, 10, 20];

    println!("Comparing AD vs numerical differentiation performance:");
    println!("Size | AD Time (μs) | Numerical Time (μs) | Speedup | Accuracy");
    println!("-----|-------------|---------------------|---------|----------");

    for &n in &problem_sizes {
        // Create test function: sum of squares
        let func = move |x: &ArrayView1<f64>| -> f64 { x.iter().map(|&xi| xi.powi(2)).sum() };

        let x = Array1::from_vec(vec![0.5; n]);

        // Time AD computation
        let ad_options = AutoDiffOptions::default();
        let start_ad = Instant::now();
        let ad_result = autodiff(func, &x.view(), &ad_options)?;
        let ad_time = start_ad.elapsed();

        // Time numerical differentiation
        let start_num = Instant::now();
        let h = 1e-8;
        let mut numerical_grad = Array1::zeros(n);

        for i in 0..n {
            let mut x_plus = x.clone();
            x_plus[i] += h;
            let f_plus = func(&x_plus.view());

            let mut x_minus = x.clone();
            x_minus[i] -= h;
            let f_minus = func(&x_minus.view());

            numerical_grad[i] = (f_plus - f_minus) / (2.0 * h);
        }
        let num_time = start_num.elapsed();

        // Calculate accuracy
        let ad_grad = ad_result.gradient.unwrap();
        let analytical_grad = x.mapv(|xi| 2.0 * xi); // Analytical gradient for x²

        let ad_error = (&ad_grad - &analytical_grad).mapv(|x| x.abs()).sum() / n as f64;
        let num_error = (&numerical_grad - &analytical_grad).mapv(|x| x.abs()).sum() / n as f64;

        let speedup = num_time.as_nanos() as f64 / ad_time.as_nanos() as f64;

        println!(
            "{:4} | {:11.1} | {:19.1} | {:7.1}x | AD: {:.1e}, Num: {:.1e}",
            n,
            ad_time.as_micros(),
            num_time.as_micros(),
            speedup,
            ad_error,
            num_error
        );
    }

    println!("\nKey observations:");
    println!("- AD provides exact gradients (machine precision)");
    println!("- Numerical differentiation has truncation error");
    println!("- AD can be faster for complex functions");
    println!("- AD avoids numerical stability issues");

    println!();
    Ok(())
}

/// Helper function to create analytical gradient for testing
#[allow(dead_code)]
fn analytical_gradient_rosenbrock(x: &ArrayView1<f64>) -> Array1<f64> {
    let a = 1.0;
    let b = 100.0;
    let x1 = x[0];
    let x2 = x[1];

    Array1::from_vec(vec![
        -2.0 * (a - x1) - 4.0 * b * x1 * (x2 - x1.powi(2)),
        2.0 * b * (x2 - x1.powi(2)),
    ])
}

/// Helper function to create analytical Hessian for testing
#[allow(dead_code)]
fn analytical_hessian_rosenbrock(x: &ArrayView1<f64>) -> ndarray::Array2<f64> {
    let b = 100.0;
    let x1 = x[0];
    let x2 = x[1];

    let h11 = 2.0 - 4.0 * b * (x2 - 3.0 * x1.powi(2));
    let h12 = -4.0 * b * x1;
    let h21 = h12; // Symmetric
    let h22 = 2.0 * b;

    ndarray::Array2::from_shape_vec((2, 2), vec![h11, h12, h21, h22]).unwrap()
}
