//! Comprehensive Performance Benchmark for scirs2-optimize
//!
//! This example benchmarks all major optimization algorithms across different problem types,
//! comparing performance, convergence rates, and resource usage.

use ndarray::{Array1, Array2, ArrayView1};
use scirs2_optimize::{
    global::{
        minimize_differential_evolution, minimize_dual_annealing, DifferentialEvolutionOptions,
        DualAnnealingOptions,
    },
    least_squares::{minimize_least_squares, LeastSquaresOptions},
    stochastic::{
        minimize_adam, minimize_adamw, minimize_rmsprop, minimize_sgd, minimize_sgd_momentum,
        AdamOptions, AdamWOptions, DataProvider, InMemoryDataProvider, LearningRateSchedule,
        MomentumOptions, RMSPropOptions, SGDOptions, StochasticGradientFunction,
    },
    unconstrained::{
        minimize_bfgs, minimize_conjugate_gradient, minimize_lbfgs, minimize_newton,
        minimize_powell, BfgsOptions, ConjugateGradientOptions, LbfgsOptions, NewtonOptions,
        PowellOptions,
    },
    OptimizeResult,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Comprehensive Performance Benchmark for scirs2-optimize\n");

    // Benchmark unconstrained optimization
    benchmark_unconstrained_methods()?;

    // Benchmark stochastic optimization
    benchmark_stochastic_methods()?;

    // Benchmark global optimization
    benchmark_global_methods()?;

    // Benchmark least squares
    benchmark_least_squares_methods()?;

    // Scalability analysis
    perform_scalability_analysis()?;

    // Cross-algorithm comparison
    perform_cross_algorithm_comparison()?;

    println!("\n✅ Performance benchmark completed successfully!");
    Ok(())
}

/// Benchmark unconstrained optimization methods
fn benchmark_unconstrained_methods() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Benchmarking Unconstrained Optimization Methods");
    println!("=".repeat(60));

    let problems = [
        ("Quadratic", create_quadratic_problem()),
        ("Rosenbrock", create_rosenbrock_problem()),
        ("Himmelblau", create_himmelblau_problem()),
        ("Rastrigin", create_rastrigin_problem()),
    ];

    let methods = [
        ("BFGS", MethodType::BFGS),
        ("L-BFGS", MethodType::LBFGS),
        ("Newton", MethodType::Newton),
        ("CG", MethodType::ConjugateGradient),
        ("Powell", MethodType::Powell),
    ];

    for (problem_name, (func, grad, x0)) in &problems {
        println!("\n🎯 Problem: {}", problem_name);
        println!("-".repeat(40));

        for (method_name, method_type) in &methods {
            let start = Instant::now();
            let result =
                run_unconstrained_method(*method_type, func.clone(), grad.clone(), x0.clone())?;
            let duration = start.elapsed();

            println!(
                "  {:12}: f={:.2e}, iter={:3}, fev={:3}, time={:6.2}ms, success={}",
                method_name,
                result.fun,
                result.iterations,
                result.func_evals,
                duration.as_millis(),
                result.success
            );
        }
    }

    Ok(())
}

/// Benchmark stochastic optimization methods
fn benchmark_stochastic_methods() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Benchmarking Stochastic Optimization Methods");
    println!("=".repeat(60));

    let problems = [
        ("Quadratic ML", create_ml_quadratic_problem()),
        ("Logistic Regression", create_logistic_regression_problem()),
    ];

    let methods = [
        ("SGD", StochasticMethodType::SGD),
        ("Momentum", StochasticMethodType::Momentum),
        ("RMSProp", StochasticMethodType::RMSProp),
        ("Adam", StochasticMethodType::Adam),
        ("AdamW", StochasticMethodType::AdamW),
    ];

    for (problem_name, (grad_func_factory, x0, data_provider_factory)) in &problems {
        println!("\n🎯 Problem: {}", problem_name);
        println!("-".repeat(40));

        for (method_name, method_type) in &methods {
            let start = Instant::now();
            let grad_func = grad_func_factory();
            let data_provider = data_provider_factory();
            let result = run_stochastic_method(*method_type, grad_func, x0.clone(), data_provider)?;
            let duration = start.elapsed();

            println!(
                "  {:12}: f={:.2e}, iter={:3}, fev={:3}, time={:6.2}ms, success={}",
                method_name,
                result.fun,
                result.iterations,
                result.func_evals,
                duration.as_millis(),
                result.success
            );
        }
    }

    Ok(())
}

/// Benchmark global optimization methods
fn benchmark_global_methods() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Benchmarking Global Optimization Methods");
    println!("=".repeat(60));

    let problems = [
        ("Ackley", create_ackley_problem()),
        ("Griewank", create_griewank_problem()),
        ("Multi-modal", create_multimodal_problem()),
    ];

    let methods = [
        (
            "Differential Evolution",
            GlobalMethodType::DifferentialEvolution,
        ),
        ("Dual Annealing", GlobalMethodType::DualAnnealing),
    ];

    for (problem_name, (func, bounds)) in &problems {
        println!("\n🎯 Problem: {}", problem_name);
        println!("-".repeat(40));

        for (method_name, method_type) in &methods {
            let start = Instant::now();
            let result = run_global_method(*method_type, func.clone(), bounds.clone())?;
            let duration = start.elapsed();

            println!(
                "  {:20}: f={:.2e}, iter={:3}, fev={:4}, time={:6.2}ms, success={}",
                method_name,
                result.fun,
                result.iterations,
                result.func_evals,
                duration.as_millis(),
                result.success
            );
        }
    }

    Ok(())
}

/// Benchmark least squares methods
fn benchmark_least_squares_methods() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Benchmarking Least Squares Methods");
    println!("=".repeat(60));

    let problems = [
        ("Linear Fit", create_linear_least_squares()),
        ("Exponential Fit", create_exponential_least_squares()),
        ("Polynomial Fit", create_polynomial_least_squares()),
    ];

    for (problem_name, (residual, jacobian, x0)) in &problems {
        println!("\n🎯 Problem: {}", problem_name);
        println!("-".repeat(40));

        let start = Instant::now();
        let options = LeastSquaresOptions::default();
        let result = minimize_least_squares(
            residual.clone(),
            Some(jacobian.clone()),
            x0.clone(),
            options,
        )?;
        let duration = start.elapsed();

        println!(
            "  Levenberg-Marquardt: f={:.2e}, iter={:3}, fev={:3}, time={:6.2}ms, success={}",
            result.fun,
            result.iterations,
            result.func_evals,
            duration.as_millis(),
            result.success
        );
    }

    Ok(())
}

/// Perform scalability analysis
fn perform_scalability_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Scalability Analysis");
    println!("=".repeat(60));

    let dimensions = [10, 50, 100, 500, 1000];
    let methods = [
        ("BFGS", MethodType::BFGS),
        ("L-BFGS", MethodType::LBFGS),
        ("Adam (stoch)", MethodType::AdamStochastic),
    ];

    println!("Dimension scaling (Quadratic problem):");
    println!("Dim    Method          Time(ms)  Iterations  Success");
    println!("-".repeat(55));

    for &dim in &dimensions {
        let (func, grad, x0) = create_high_dimensional_quadratic(dim);

        for (method_name, method_type) in &methods {
            let start = Instant::now();

            let result = match method_type {
                MethodType::AdamStochastic => {
                    let grad_func = HighDimQuadraticML { dim };
                    let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));
                    let options = AdamOptions {
                        learning_rate: 0.01,
                        max_iter: 1000,
                        tol: 1e-6,
                        ..Default::default()
                    };
                    minimize_adam(grad_func, x0.clone(), data_provider, options)?
                }
                _ => {
                    run_unconstrained_method(*method_type, func.clone(), grad.clone(), x0.clone())?
                }
            };

            let duration = start.elapsed();

            println!(
                "{:4}   {:12}   {:7.2}    {:4}       {}",
                dim,
                method_name,
                duration.as_millis(),
                result.iterations,
                result.success
            );
        }
    }

    Ok(())
}

/// Perform cross-algorithm comparison on challenging problems
fn perform_cross_algorithm_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Cross-Algorithm Comparison Summary");
    println!("=".repeat(60));

    // Test on the notorious Rosenbrock function
    println!("Challenge Problem: Rosenbrock Function (2D)");
    println!("Method              Final Value    Time(ms)  Iterations  Rating");
    println!("-".repeat(65));

    let (func, grad, x0) = create_rosenbrock_problem();

    let test_methods = [
        ("BFGS", TestMethodType::BFGS),
        ("L-BFGS", TestMethodType::LBFGS),
        ("Newton", TestMethodType::Newton),
        ("Adam", TestMethodType::Adam),
        (
            "Differential Evolution",
            TestMethodType::DifferentialEvolution,
        ),
    ];

    let mut results = Vec::new();

    for (name, method) in &test_methods {
        let start = Instant::now();
        let result = match method {
            TestMethodType::BFGS => {
                run_unconstrained_method(MethodType::BFGS, func.clone(), grad.clone(), x0.clone())?
            }
            TestMethodType::LBFGS => {
                run_unconstrained_method(MethodType::LBFGS, func.clone(), grad.clone(), x0.clone())?
            }
            TestMethodType::Newton => run_unconstrained_method(
                MethodType::Newton,
                func.clone(),
                grad.clone(),
                x0.clone(),
            )?,
            TestMethodType::Adam => {
                let grad_func = SimpleRosenbrock;
                let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));
                let options = AdamOptions {
                    learning_rate: 0.001,
                    max_iter: 2000,
                    tol: 1e-6,
                    ..Default::default()
                };
                minimize_adam(grad_func, x0.clone(), data_provider, options)?
            }
            TestMethodType::DifferentialEvolution => {
                let bounds = vec![(-2.0, 2.0), (-2.0, 2.0)];
                run_global_method(
                    GlobalMethodType::DifferentialEvolution,
                    func.clone(),
                    bounds,
                )?
            }
        };
        let duration = start.elapsed();

        // Calculate a performance rating
        let rating = calculate_performance_rating(&result, duration.as_millis() as f64);

        results.push((
            *name,
            result.fun,
            duration.as_millis(),
            result.iterations,
            rating,
        ));

        println!(
            "{:18}  {:.2e}     {:5}        {:4}     {:4.1}/5",
            name,
            result.fun,
            duration.as_millis(),
            result.iterations,
            rating
        );
    }

    // Find the best performer
    results.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap());
    println!(
        "\n🏆 Best performer: {} (Rating: {:.1}/5)",
        results[0].0, results[0].4
    );

    Ok(())
}

// Helper function to calculate performance rating
fn calculate_performance_rating(result: &OptimizeResult<f64>, time_ms: f64) -> f64 {
    let accuracy_score = if result.fun < 1e-10 {
        5.0
    } else if result.fun < 1e-6 {
        4.0
    } else if result.fun < 1e-3 {
        3.0
    } else if result.fun < 1e-1 {
        2.0
    } else {
        1.0
    };

    let speed_score = if time_ms < 10.0 {
        5.0
    } else if time_ms < 50.0 {
        4.0
    } else if time_ms < 200.0 {
        3.0
    } else if time_ms < 1000.0 {
        2.0
    } else {
        1.0
    };

    let success_score = if result.success { 5.0 } else { 1.0 };

    (accuracy_score + speed_score + success_score) / 3.0
}

// Method type enumerations
#[derive(Clone, Copy)]
enum MethodType {
    BFGS,
    LBFGS,
    Newton,
    ConjugateGradient,
    Powell,
    AdamStochastic,
}

#[derive(Clone, Copy)]
enum StochasticMethodType {
    SGD,
    Momentum,
    RMSProp,
    Adam,
    AdamW,
}

#[derive(Clone, Copy)]
enum GlobalMethodType {
    DifferentialEvolution,
    DualAnnealing,
}

#[derive(Clone, Copy)]
enum TestMethodType {
    BFGS,
    LBFGS,
    Newton,
    Adam,
    DifferentialEvolution,
}

// Function type aliases for better readability
type ObjectiveFunction = Box<dyn Fn(&ArrayView1<f64>) -> f64 + Send + Sync>;
type GradientFunction = Box<dyn Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync>;
type ResidualFunction = Box<dyn Fn(&ArrayView1<f64>) -> Array1<f64> + Send + Sync>;
type JacobianFunction = Box<dyn Fn(&ArrayView1<f64>) -> Array2<f64> + Send + Sync>;

type StochasticGradientFactory = Box<dyn Fn() -> Box<dyn StochasticGradientFunction> + Send + Sync>;
type DataProviderFactory = Box<dyn Fn() -> Box<dyn DataProvider> + Send + Sync>;

// Problem creation functions
fn create_quadratic_problem() -> (ObjectiveFunction, GradientFunction, Array1<f64>) {
    let func = Box::new(|x: &ArrayView1<f64>| -> f64 { x.mapv(|xi| xi * xi).sum() });

    let grad = Box::new(|x: &ArrayView1<f64>| -> Array1<f64> { x.mapv(|xi| 2.0 * xi) });

    let x0 = Array1::from_vec(vec![2.0, -1.5]);
    (func, grad, x0)
}

fn create_rosenbrock_problem() -> (ObjectiveFunction, GradientFunction, Array1<f64>) {
    let func = Box::new(|x: &ArrayView1<f64>| -> f64 {
        let x1 = x[0];
        let x2 = x[1];
        (1.0 - x1).powi(2) + 100.0 * (x2 - x1 * x1).powi(2)
    });

    let grad = Box::new(|x: &ArrayView1<f64>| -> Array1<f64> {
        let x1 = x[0];
        let x2 = x[1];
        Array1::from_vec(vec![
            -2.0 * (1.0 - x1) - 400.0 * x1 * (x2 - x1 * x1),
            200.0 * (x2 - x1 * x1),
        ])
    });

    let x0 = Array1::from_vec(vec![-1.2, 1.0]);
    (func, grad, x0)
}

fn create_himmelblau_problem() -> (ObjectiveFunction, GradientFunction, Array1<f64>) {
    let func = Box::new(|x: &ArrayView1<f64>| -> f64 {
        let x1 = x[0];
        let x2 = x[1];
        (x1 * x1 + x2 - 11.0).powi(2) + (x1 + x2 * x2 - 7.0).powi(2)
    });

    let grad = Box::new(|x: &ArrayView1<f64>| -> Array1<f64> {
        let x1 = x[0];
        let x2 = x[1];
        Array1::from_vec(vec![
            4.0 * x1 * (x1 * x1 + x2 - 11.0) + 2.0 * (x1 + x2 * x2 - 7.0),
            2.0 * (x1 * x1 + x2 - 11.0) + 4.0 * x2 * (x1 + x2 * x2 - 7.0),
        ])
    });

    let x0 = Array1::from_vec(vec![0.0, 0.0]);
    (func, grad, x0)
}

fn create_rastrigin_problem() -> (ObjectiveFunction, GradientFunction, Array1<f64>) {
    let func = Box::new(|x: &ArrayView1<f64>| -> f64 {
        let n = x.len() as f64;
        let a = 10.0;
        a * n
            + x.iter()
                .map(|&xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    });

    let grad = Box::new(|x: &ArrayView1<f64>| -> Array1<f64> {
        let a = 10.0;
        x.mapv(|xi| {
            2.0 * xi + a * 2.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * xi).sin()
        })
    });

    let x0 = Array1::from_vec(vec![2.0, 2.0]);
    (func, grad, x0)
}

fn create_ackley_problem() -> (ObjectiveFunction, Vec<(f64, f64)>) {
    let func = Box::new(|x: &ArrayView1<f64>| -> f64 {
        let n = x.len() as f64;
        let sum_sq = x.iter().map(|&xi| xi * xi).sum::<f64>();
        let sum_cos = x
            .iter()
            .map(|&xi| (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>();

        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
            + 20.0
            + std::f64::consts::E
    });

    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
    (func, bounds)
}

fn create_griewank_problem() -> (ObjectiveFunction, Vec<(f64, f64)>) {
    let func = Box::new(|x: &ArrayView1<f64>| -> f64 {
        let sum_sq = x.iter().map(|&xi| xi * xi).sum::<f64>() / 4000.0;
        let prod_cos = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| (xi / ((i + 1) as f64).sqrt()).cos())
            .product::<f64>();
        sum_sq - prod_cos + 1.0
    });

    let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];
    (func, bounds)
}

fn create_multimodal_problem() -> (ObjectiveFunction, Vec<(f64, f64)>) {
    let func = Box::new(|x: &ArrayView1<f64>| -> f64 {
        // Modified Himmelblau with multiple minima
        let x1 = x[0];
        let x2 = x[1];
        (x1 * x1 + x2 - 11.0).powi(2)
            + (x1 + x2 * x2 - 7.0).powi(2)
            + 0.1 * ((5.0 * x1).sin() * (5.0 * x2).sin()).powi(2)
    });

    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
    (func, bounds)
}

fn create_high_dimensional_quadratic(
    dim: usize,
) -> (ObjectiveFunction, GradientFunction, Array1<f64>) {
    let func = Box::new(move |x: &ArrayView1<f64>| -> f64 { x.mapv(|xi| xi * xi).sum() });

    let grad = Box::new(|x: &ArrayView1<f64>| -> Array1<f64> { x.mapv(|xi| 2.0 * xi) });

    let x0 = Array1::from_elem(dim, 1.0);
    (func, grad, x0)
}

// Stochastic problem factories
fn create_ml_quadratic_problem() -> (StochasticGradientFactory, Array1<f64>, DataProviderFactory) {
    let grad_func_factory =
        Box::new(|| -> Box<dyn StochasticGradientFunction> { Box::new(QuadraticML) });

    let x0 = Array1::from_vec(vec![2.0, -1.5, 1.0]);

    let data_provider_factory = Box::new(|| -> Box<dyn DataProvider> {
        Box::new(InMemoryDataProvider::new(vec![1.0; 100]))
    });

    (grad_func_factory, x0, data_provider_factory)
}

fn create_logistic_regression_problem(
) -> (StochasticGradientFactory, Array1<f64>, DataProviderFactory) {
    let grad_func_factory = Box::new(|| -> Box<dyn StochasticGradientFunction> {
        Box::new(LogisticRegressionML::new())
    });

    let x0 = Array1::zeros(4); // 4 features

    let data_provider_factory = Box::new(|| -> Box<dyn DataProvider> {
        // Generate synthetic dataset indices
        let indices: Vec<f64> = (0..200).map(|i| i as f64).collect();
        Box::new(InMemoryDataProvider::new(indices))
    });

    (grad_func_factory, x0, data_provider_factory)
}

// Least squares problems
fn create_linear_least_squares() -> (ResidualFunction, JacobianFunction, Array1<f64>) {
    let residual = Box::new(|x: &ArrayView1<f64>| -> Array1<f64> {
        let m = x[0];
        let b = x[1];
        let x_data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = [2.1, 4.0, 6.1, 8.0, 9.9]; // roughly y = 2x with noise

        x_data
            .iter()
            .zip(y_data.iter())
            .map(|(&x_i, &y_i)| m * x_i + b - y_i)
            .collect::<Vec<f64>>()
            .into()
    });

    let jacobian = Box::new(|_x: &ArrayView1<f64>| -> Array2<f64> {
        let x_data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut jac = Array2::zeros((5, 2));
        for (i, &x_i) in x_data.iter().enumerate() {
            jac[[i, 0]] = x_i; // ∂residual/∂m
            jac[[i, 1]] = 1.0; // ∂residual/∂b
        }
        jac
    });

    let x0 = Array1::from_vec(vec![1.0, 0.0]);
    (residual, jacobian, x0)
}

fn create_exponential_least_squares() -> (ResidualFunction, JacobianFunction, Array1<f64>) {
    let residual = Box::new(|x: &ArrayView1<f64>| -> Array1<f64> {
        let a = x[0];
        let b = x[1];
        let x_data = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y_data = [1.0, 2.7, 7.4, 20.1, 54.6]; // roughly y = e^x

        x_data
            .iter()
            .zip(y_data.iter())
            .map(|(&x_i, &y_i)| a * (b * x_i).exp() - y_i)
            .collect::<Vec<f64>>()
            .into()
    });

    let jacobian = Box::new(|x: &ArrayView1<f64>| -> Array2<f64> {
        let a = x[0];
        let b = x[1];
        let x_data = [0.0, 1.0, 2.0, 3.0, 4.0];
        let mut jac = Array2::zeros((5, 2));
        for (i, &x_i) in x_data.iter().enumerate() {
            let exp_bx = (b * x_i).exp();
            jac[[i, 0]] = exp_bx;
            jac[[i, 1]] = a * x_i * exp_bx;
        }
        jac
    });

    let x0 = Array1::from_vec(vec![1.0, 1.0]);
    (residual, jacobian, x0)
}

fn create_polynomial_least_squares() -> (ResidualFunction, JacobianFunction, Array1<f64>) {
    let residual = Box::new(|x: &ArrayView1<f64>| -> Array1<f64> {
        let a = x[0];
        let b = x[1];
        let c = x[2];
        let x_data = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let y_data = [6.1, 2.1, 0.9, 2.1, 6.1]; // roughly y = x^2 + 1

        x_data
            .iter()
            .zip(y_data.iter())
            .map(|(&x_i, &y_i)| a * x_i * x_i + b * x_i + c - y_i)
            .collect::<Vec<f64>>()
            .into()
    });

    let jacobian = Box::new(|_x: &ArrayView1<f64>| -> Array2<f64> {
        let x_data = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut jac = Array2::zeros((5, 3));
        for (i, &x_i) in x_data.iter().enumerate() {
            jac[[i, 0]] = x_i * x_i;
            jac[[i, 1]] = x_i;
            jac[[i, 2]] = 1.0;
        }
        jac
    });

    let x0 = Array1::from_vec(vec![1.0, 0.0, 1.0]);
    (residual, jacobian, x0)
}

// Stochastic gradient function implementations
struct QuadraticML;

impl StochasticGradientFunction for QuadraticML {
    fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> Array1<f64> {
        x.mapv(|xi| 2.0 * xi)
    }

    fn compute_value(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> f64 {
        x.mapv(|xi| xi * xi).sum()
    }
}

struct HighDimQuadraticML {
    dim: usize,
}

impl StochasticGradientFunction for HighDimQuadraticML {
    fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> Array1<f64> {
        x.mapv(|xi| 2.0 * xi)
    }

    fn compute_value(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> f64 {
        x.mapv(|xi| xi * xi).sum()
    }
}

struct SimpleRosenbrock;

impl StochasticGradientFunction for SimpleRosenbrock {
    fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> Array1<f64> {
        let x1 = x[0];
        let x2 = x[1];
        Array1::from_vec(vec![
            -2.0 * (1.0 - x1) - 400.0 * x1 * (x2 - x1 * x1),
            200.0 * (x2 - x1 * x1),
        ])
    }

    fn compute_value(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> f64 {
        let x1 = x[0];
        let x2 = x[1];
        (1.0 - x1).powi(2) + 100.0 * (x2 - x1 * x1).powi(2)
    }
}

struct LogisticRegressionML {
    features: Array2<f64>,
    labels: Array1<f64>,
}

impl LogisticRegressionML {
    fn new() -> Self {
        // Generate synthetic dataset
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n_samples = 200;
        let n_features = 4;

        let mut features = Array2::zeros((n_samples, n_features));
        let mut labels = Array1::zeros(n_samples);
        let true_weights = Array1::from_vec(vec![0.5, -0.3, 0.8, -0.2]);

        for i in 0..n_samples {
            for j in 0..n_features {
                features[[i, j]] = rng.gen_range(-2.0..2.0);
            }

            let z = features.row(i).dot(&true_weights) + rng.gen_range(-0.1..0.1);
            labels[i] = if z > 0.0 { 1.0 } else { 0.0 };
        }

        Self { features, labels }
    }
}

impl StochasticGradientFunction for LogisticRegressionML {
    fn compute_gradient(&mut self, params: &ArrayView1<f64>, batch_indices: &[f64]) -> Array1<f64> {
        let indices: Vec<usize> = batch_indices.iter().map(|&x| x as usize).collect();
        let mut gradient = Array1::zeros(params.len());

        for &i in &indices {
            if i < self.labels.len() {
                let x_i = self.features.row(i);
                let y_i = self.labels[i];

                let z = x_i.dot(params);
                let prediction = 1.0 / (1.0 + (-z).exp());
                let error = prediction - y_i;

                for (j, &x_ij) in x_i.iter().enumerate() {
                    gradient[j] += error * x_ij;
                }
            }
        }

        gradient / indices.len() as f64
    }

    fn compute_value(&mut self, params: &ArrayView1<f64>, batch_indices: &[f64]) -> f64 {
        let indices: Vec<usize> = batch_indices.iter().map(|&x| x as usize).collect();
        let mut loss = 0.0;

        for &i in &indices {
            if i < self.labels.len() {
                let x_i = self.features.row(i);
                let y_i = self.labels[i];

                let z = x_i.dot(params);
                let prediction = 1.0 / (1.0 + (-z).exp());

                loss += -y_i * prediction.ln() - (1.0 - y_i) * (1.0 - prediction).ln();
            }
        }

        loss / indices.len() as f64
    }
}

// Method runners
fn run_unconstrained_method(
    method: MethodType,
    func: ObjectiveFunction,
    grad: GradientFunction,
    x0: Array1<f64>,
) -> Result<OptimizeResult<f64>, Box<dyn std::error::Error>> {
    match method {
        MethodType::BFGS => {
            let options = BfgsOptions::default();
            let mut f = func;
            let mut g = grad;
            Ok(minimize_bfgs(|x| f(x), |x| g(x), x0, options)?)
        }
        MethodType::LBFGS => {
            let options = LbfgsOptions::default();
            let mut f = func;
            let mut g = grad;
            Ok(minimize_lbfgs(|x| f(x), |x| g(x), x0, options)?)
        }
        MethodType::Newton => {
            let options = NewtonOptions::default();
            let mut f = func;
            let mut g = grad;
            Ok(minimize_newton(|x| f(x), |x| g(x), x0, options)?)
        }
        MethodType::ConjugateGradient => {
            let options = ConjugateGradientOptions::default();
            let mut f = func;
            let mut g = grad;
            Ok(minimize_conjugate_gradient(
                |x| f(x),
                |x| g(x),
                x0,
                options,
            )?)
        }
        MethodType::Powell => {
            let options = PowellOptions::default();
            let mut f = func;
            Ok(minimize_powell(|x| f(x), x0, options)?)
        }
        _ => unreachable!(),
    }
}

fn run_stochastic_method(
    method: StochasticMethodType,
    mut grad_func: Box<dyn StochasticGradientFunction>,
    x0: Array1<f64>,
    data_provider: Box<dyn DataProvider>,
) -> Result<OptimizeResult<f64>, Box<dyn std::error::Error>> {
    match method {
        StochasticMethodType::SGD => {
            let options = SGDOptions {
                learning_rate: 0.01,
                max_iter: 500,
                tol: 1e-6,
                ..Default::default()
            };
            Ok(minimize_sgd(*grad_func, x0, data_provider, options)?)
        }
        StochasticMethodType::Momentum => {
            let options = MomentumOptions {
                learning_rate: 0.01,
                momentum: 0.9,
                max_iter: 500,
                tol: 1e-6,
                ..Default::default()
            };
            Ok(minimize_sgd_momentum(
                *grad_func,
                x0,
                data_provider,
                options,
            )?)
        }
        StochasticMethodType::RMSProp => {
            let options = RMSPropOptions {
                learning_rate: 0.01,
                max_iter: 500,
                tol: 1e-6,
                ..Default::default()
            };
            Ok(minimize_rmsprop(*grad_func, x0, data_provider, options)?)
        }
        StochasticMethodType::Adam => {
            let options = AdamOptions {
                learning_rate: 0.01,
                max_iter: 500,
                tol: 1e-6,
                ..Default::default()
            };
            Ok(minimize_adam(*grad_func, x0, data_provider, options)?)
        }
        StochasticMethodType::AdamW => {
            let options = AdamWOptions {
                learning_rate: 0.01,
                max_iter: 500,
                tol: 1e-6,
                ..Default::default()
            };
            Ok(minimize_adamw(*grad_func, x0, data_provider, options)?)
        }
    }
}

fn run_global_method(
    method: GlobalMethodType,
    func: ObjectiveFunction,
    bounds: Vec<(f64, f64)>,
) -> Result<OptimizeResult<f64>, Box<dyn std::error::Error>> {
    match method {
        GlobalMethodType::DifferentialEvolution => {
            let options = DifferentialEvolutionOptions {
                max_iter: 100,
                population_size: 15,
                ..Default::default()
            };
            let mut f = func;
            Ok(minimize_differential_evolution(|x| f(x), bounds, options)?)
        }
        GlobalMethodType::DualAnnealing => {
            let options = DualAnnealingOptions {
                max_iter: 100,
                ..Default::default()
            };
            let mut f = func;
            Ok(minimize_dual_annealing(|x| f(x), bounds, options)?)
        }
    }
}
