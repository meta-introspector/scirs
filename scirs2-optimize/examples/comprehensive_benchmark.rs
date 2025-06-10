//! Comprehensive benchmark example comparing optimization methods
//!
//! This example demonstrates various optimization algorithms on standard test functions
//! and provides performance comparisons.

use ndarray::{array, Array1, ArrayView1};
use scirs2_optimize::{
    error::OptimizeError,
    global::{differential_evolution, DifferentialEvolutionOptions},
    unconstrained::{
        minimize, minimize_bfgs, minimize_lbfgs, minimize_nelder_mead, minimize_powell, Method,
        Options,
    },
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Standard test function trait
trait TestFunction: Sync {
    fn name(&self) -> &str;
    fn evaluate(&self, x: &ArrayView1<f64>) -> f64;
    fn global_minimum(&self) -> f64;
    #[allow(dead_code)]
    fn global_optimum(&self) -> Array1<f64>;
    fn bounds(&self) -> Vec<(f64, f64)>;
    fn initial_guess(&self) -> Array1<f64>;
}

/// Rosenbrock function
struct Rosenbrock {
    dimensions: usize,
}

impl Rosenbrock {
    fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl TestFunction for Rosenbrock {
    fn name(&self) -> &str {
        "Rosenbrock"
    }

    fn evaluate(&self, x: &ArrayView1<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    }

    fn global_minimum(&self) -> f64 {
        0.0
    }

    fn global_optimum(&self) -> Array1<f64> {
        Array1::ones(self.dimensions)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(-5.0, 10.0); self.dimensions]
    }

    fn initial_guess(&self) -> Array1<f64> {
        Array1::zeros(self.dimensions)
    }
}

/// Sphere function
struct Sphere {
    dimensions: usize,
}

impl Sphere {
    fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl TestFunction for Sphere {
    fn name(&self) -> &str {
        "Sphere"
    }

    fn evaluate(&self, x: &ArrayView1<f64>) -> f64 {
        x.iter().map(|&xi| xi.powi(2)).sum()
    }

    fn global_minimum(&self) -> f64 {
        0.0
    }

    fn global_optimum(&self) -> Array1<f64> {
        Array1::zeros(self.dimensions)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(-5.0, 5.0); self.dimensions]
    }

    fn initial_guess(&self) -> Array1<f64> {
        Array1::from_elem(self.dimensions, 2.0)
    }
}

/// Beale function (2D only)
struct Beale;

impl TestFunction for Beale {
    fn name(&self) -> &str {
        "Beale"
    }

    fn evaluate(&self, x: &ArrayView1<f64>) -> f64 {
        let x0 = x[0];
        let x1 = x[1];
        (1.5 - x0 + x0 * x1).powi(2)
            + (2.25 - x0 + x0 * x1.powi(2)).powi(2)
            + (2.625 - x0 + x0 * x1.powi(3)).powi(2)
    }

    fn global_minimum(&self) -> f64 {
        0.0
    }

    fn global_optimum(&self) -> Array1<f64> {
        array![3.0, 0.5]
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(-4.5, 4.5), (-4.5, 4.5)]
    }

    fn initial_guess(&self) -> Array1<f64> {
        array![1.0, 1.0]
    }
}

/// Booth function (2D only)
struct Booth;

impl TestFunction for Booth {
    fn name(&self) -> &str {
        "Booth"
    }

    fn evaluate(&self, x: &ArrayView1<f64>) -> f64 {
        let x0 = x[0];
        let x1 = x[1];
        (x0 + 2.0 * x1 - 7.0).powi(2) + (2.0 * x0 + x1 - 5.0).powi(2)
    }

    fn global_minimum(&self) -> f64 {
        0.0
    }

    fn global_optimum(&self) -> Array1<f64> {
        array![1.0, 3.0]
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        vec![(-10.0, 10.0), (-10.0, 10.0)]
    }

    fn initial_guess(&self) -> Array1<f64> {
        array![0.0, 0.0]
    }
}

/// Optimization result summary
#[derive(Debug, Clone)]
struct OptimizationResult {
    method_name: String,
    function_name: String,
    success: bool,
    final_value: f64,
    error_from_global: f64,
    num_iterations: usize,
    num_function_evals: usize,
    time_elapsed: Duration,
}

/// Run optimization with a specific method
fn run_optimization<F>(
    method_name: &str,
    function: &dyn TestFunction,
    optimizer: F,
) -> OptimizationResult
where
    F: FnOnce(
        Array1<f64>,
    ) -> Result<scirs2_optimize::unconstrained::OptimizeResult<f64>, OptimizeError>,
{
    let start_time = Instant::now();
    let initial_guess = function.initial_guess();

    match optimizer(initial_guess) {
        Ok(result) => {
            let error_from_global = (result.fun - function.global_minimum()).abs();

            OptimizationResult {
                method_name: method_name.to_string(),
                function_name: function.name().to_string(),
                success: result.success,
                final_value: result.fun,
                error_from_global,
                num_iterations: result.iterations,
                num_function_evals: result.func_evals,
                time_elapsed: start_time.elapsed(),
            }
        }
        Err(_) => OptimizationResult {
            method_name: method_name.to_string(),
            function_name: function.name().to_string(),
            success: false,
            final_value: f64::INFINITY,
            error_from_global: f64::INFINITY,
            num_iterations: 0,
            num_function_evals: 0,
            time_elapsed: start_time.elapsed(),
        },
    }
}

/// Benchmark all optimization methods on a test function
fn benchmark_function(function: &dyn TestFunction) -> Vec<OptimizationResult> {
    let mut results = Vec::new();

    println!("  Testing function: {}", function.name());

    // BFGS
    {
        let func = |x: &ArrayView1<f64>| function.evaluate(x);
        let result = run_optimization("BFGS", function, |x0| {
            minimize_bfgs(&func, x0, &Options::default())
        });
        results.push(result);
    }

    // L-BFGS
    {
        let func = |x: &ArrayView1<f64>| function.evaluate(x);
        let result = run_optimization("L-BFGS", function, |x0| {
            minimize_lbfgs(&func, x0, &Options::default())
        });
        results.push(result);
    }

    // Conjugate Gradient
    {
        let func = |x: &ArrayView1<f64>| function.evaluate(x);
        let result = run_optimization("CG", function, |x0| {
            minimize(
                &func,
                x0.as_slice().unwrap(),
                Method::CG,
                Some(Options::default()),
            )
        });
        results.push(result);
    }

    // Nelder-Mead
    {
        let func = |x: &ArrayView1<f64>| function.evaluate(x);
        let result = run_optimization("Nelder-Mead", function, |x0| {
            minimize_nelder_mead(&func, x0, &Options::default())
        });
        results.push(result);
    }

    // Powell
    {
        let func = |x: &ArrayView1<f64>| function.evaluate(x);
        let result = run_optimization("Powell", function, |x0| {
            minimize_powell(&func, x0, &Options::default())
        });
        results.push(result);
    }

    // Differential Evolution (global method)
    {
        let func = |x: &ArrayView1<f64>| function.evaluate(x);
        let result = run_optimization("Differential Evolution", function, |_x0| {
            let bounds = function.bounds();
            let options = DifferentialEvolutionOptions::default();
            differential_evolution(&func, bounds, Some(options), None)
        });
        results.push(result);
    }

    results
}

/// Print results in a nice table format
fn print_results(all_results: &[OptimizationResult]) {
    println!("\n{:-<120}", "");
    println!(
        "{:20} {:15} {:>8} {:>12} {:>15} {:>8} {:>8} {:>12}",
        "Function", "Method", "Success", "Final Value", "Error", "Iter", "FEvals", "Time (ms)"
    );
    println!("{:-<120}", "");

    for result in all_results {
        let success_str = if result.success { "✓" } else { "✗" };
        let time_ms = result.time_elapsed.as_secs_f64() * 1000.0;

        println!(
            "{:20} {:15} {:>8} {:>12.6e} {:>15.6e} {:>8} {:>8} {:>12.2}",
            result.function_name,
            result.method_name,
            success_str,
            result.final_value,
            result.error_from_global,
            result.num_iterations,
            result.num_function_evals,
            time_ms
        );
    }
    println!("{:-<120}", "");
}

/// Print summary statistics
fn print_summary(all_results: &[OptimizationResult]) {
    println!("\n=== SUMMARY STATISTICS ===");

    // Group by method
    let mut by_method: HashMap<String, Vec<&OptimizationResult>> = HashMap::new();
    for result in all_results {
        by_method
            .entry(result.method_name.clone())
            .or_insert_with(Vec::new)
            .push(result);
    }

    println!(
        "\n{:15} {:>8} {:>8} {:>15} {:>12} {:>12}",
        "Method", "Success%", "Avg Iter", "Avg Error", "Avg FEvals", "Avg Time(ms)"
    );
    println!("{:-<80}", "");

    for (method_name, results) in by_method {
        let total_runs = results.len();
        let successes = results.iter().filter(|r| r.success).count();
        let success_rate = (successes as f64 / total_runs as f64) * 100.0;

        let avg_iterations =
            results.iter().map(|r| r.num_iterations as f64).sum::<f64>() / total_runs as f64;

        let avg_error = results
            .iter()
            .filter(|r| r.error_from_global.is_finite())
            .map(|r| r.error_from_global)
            .sum::<f64>()
            / results
                .iter()
                .filter(|r| r.error_from_global.is_finite())
                .count() as f64;

        let avg_fevals = results
            .iter()
            .map(|r| r.num_function_evals as f64)
            .sum::<f64>()
            / total_runs as f64;

        let avg_time = results
            .iter()
            .map(|r| r.time_elapsed.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / total_runs as f64;

        println!(
            "{:15} {:>7.1}% {:>8.1} {:>15.6e} {:>12.1} {:>12.2}",
            method_name, success_rate, avg_iterations, avg_error, avg_fevals, avg_time
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SCIRS2-OPTIMIZE COMPREHENSIVE BENCHMARK ===");
    println!("Testing various optimization methods on standard test functions");
    println!();

    // Define test functions
    let test_functions: Vec<Box<dyn TestFunction>> = vec![
        Box::new(Sphere::new(2)),
        Box::new(Sphere::new(5)),
        Box::new(Rosenbrock::new(2)),
        Box::new(Rosenbrock::new(5)),
        Box::new(Beale),
        Box::new(Booth),
    ];

    let mut all_results = Vec::new();

    println!("Running optimization benchmarks...");
    for function in &test_functions {
        let results = benchmark_function(function.as_ref());
        all_results.extend(results);
    }

    // Print detailed results
    print_results(&all_results);

    // Print summary
    print_summary(&all_results);

    println!("\n=== PERFORMANCE INSIGHTS ===");

    // Find best method for each function
    let mut by_function: HashMap<String, Vec<&OptimizationResult>> = HashMap::new();
    for result in &all_results {
        by_function
            .entry(result.function_name.clone())
            .or_insert_with(Vec::new)
            .push(result);
    }

    for (function_name, results) in by_function {
        let successful_results: Vec<_> = results
            .iter()
            .filter(|r| r.success && r.error_from_global.is_finite())
            .collect();

        if let Some(best) = successful_results.iter().min_by(|a, b| {
            a.error_from_global
                .partial_cmp(&b.error_from_global)
                .unwrap()
        }) {
            let fastest = successful_results
                .iter()
                .min_by(|a, b| a.time_elapsed.partial_cmp(&b.time_elapsed).unwrap());

            let most_efficient = successful_results.iter().min_by(|a, b| {
                a.num_function_evals
                    .partial_cmp(&b.num_function_evals)
                    .unwrap()
            });

            println!("\n{}:", function_name);
            println!(
                "  Most accurate: {} (error: {:.2e})",
                best.method_name, best.error_from_global
            );
            if let Some(fast) = fastest {
                println!(
                    "  Fastest: {} ({:.1}ms)",
                    fast.method_name,
                    fast.time_elapsed.as_secs_f64() * 1000.0
                );
            }
            if let Some(efficient) = most_efficient {
                println!(
                    "  Most efficient: {} ({} function evaluations)",
                    efficient.method_name, efficient.num_function_evals
                );
            }
        }
    }

    println!("\n=== RECOMMENDATION ===");
    println!("• For smooth functions: BFGS or L-BFGS typically perform best");
    println!("• For non-smooth functions: Nelder-Mead or Powell are good choices");
    println!("• For global optimization: Differential Evolution can find global minima");
    println!("• For high-dimensional problems: L-BFGS is memory-efficient");
    println!("• For noisy functions: Nelder-Mead is robust to noise");

    println!("\n=== BENCHMARK COMPLETE ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_rosenbrock_global_minimum() {
        let f = Rosenbrock::new(2);
        let x_opt = f.global_optimum();
        let f_opt = f.evaluate(&x_opt.view());
        assert_abs_diff_eq!(f_opt, f.global_minimum(), epsilon = 1e-12);
    }

    #[test]
    fn test_sphere_global_minimum() {
        let f = Sphere::new(3);
        let x_opt = f.global_optimum();
        let f_opt = f.evaluate(&x_opt.view());
        assert_abs_diff_eq!(f_opt, f.global_minimum(), epsilon = 1e-12);
    }

    #[test]
    fn test_beale_global_minimum() {
        let f = Beale;
        let x_opt = f.global_optimum();
        let f_opt = f.evaluate(&x_opt.view());
        assert_abs_diff_eq!(f_opt, f.global_minimum(), epsilon = 1e-12);
    }
}
