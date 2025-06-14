//! Stochastic optimization example
//!
//! This example demonstrates the various stochastic optimization methods available
//! in the optimization library, comparing their performance on different problems.

use ndarray::{Array1, Array2, ArrayView1};
use scirs2_optimize::stochastic::{
    minimize_adam, minimize_adamw, minimize_rmsprop, minimize_sgd, minimize_sgd_momentum,
    minimize_stochastic, AdamOptions, AdamWOptions, DataProvider, InMemoryDataProvider,
    LearningRateSchedule, MomentumOptions, RMSPropOptions, SGDOptions, StochasticGradientFunction,
    StochasticMethod, StochasticOptions,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Stochastic Optimization Methods Comparison\\n");

    // Example 1: Quadratic function optimization
    quadratic_function_example()?;

    // Example 2: Rosenbrock function optimization
    rosenbrock_function_example()?;

    // Example 3: Machine learning-style logistic regression
    logistic_regression_example()?;

    // Example 4: Noisy optimization problem
    noisy_optimization_example()?;

    // Example 5: Learning rate schedule comparison
    learning_rate_schedule_comparison()?;

    // Example 6: Batch size effects
    batch_size_effects_example()?;

    Ok(())
}

/// Example 1: Simple quadratic function optimization
fn quadratic_function_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 1: Quadratic Function Optimization ===");

    // Define a simple quadratic function: f(x) = sum(x_i^2)
    struct QuadraticProblem;

    impl StochasticGradientFunction for QuadraticProblem {
        fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> Array1<f64> {
            x.mapv(|xi| 2.0 * xi)
        }

        fn compute_value(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> f64 {
            x.mapv(|xi| xi * xi).sum()
        }
    }

    let x0 = Array1::from_vec(vec![2.0, -1.5, 3.0, -2.0]);
    let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));

    println!("Function: f(x) = sum(x_i^2)");
    println!(
        "Starting point: [{:.1}, {:.1}, {:.1}, {:.1}]",
        x0[0], x0[1], x0[2], x0[3]
    );
    println!("Global minimum: [0, 0, 0, 0] with f = 0\\n");

    let methods = [
        ("SGD", StochasticMethod::SGD),
        ("Momentum", StochasticMethod::Momentum),
        ("RMSProp", StochasticMethod::RMSProp),
        ("Adam", StochasticMethod::Adam),
        ("AdamW", StochasticMethod::AdamW),
    ];

    for (name, method) in &methods {
        let options = StochasticOptions {
            learning_rate: 0.1,
            max_iter: 200,
            batch_size: Some(20),
            tol: 1e-8,
            ..Default::default()
        };

        let grad_func = QuadraticProblem;
        let x0_clone = x0.clone();
        let data_clone = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));

        let start = Instant::now();
        let result = minimize_stochastic(*method, grad_func, x0_clone, data_clone, options)?;
        let duration = start.elapsed();

        println!("{} results:", name);
        println!(
            "  Final solution: [{:.6}, {:.6}, {:.6}, {:.6}]",
            result.x[0], result.x[1], result.x[2], result.x[3]
        );
        println!("  Function value: {:.2e}", result.fun);
        println!("  Iterations: {}", result.iterations);
        println!("  Function evaluations: {}", result.func_evals);
        println!("  Success: {}", result.success);
        println!("  Time: {:.2} ms", duration.as_millis());

        // Calculate distance from true optimum
        let error = result.x.mapv(|xi| xi * xi).sum().sqrt();
        println!("  Distance from optimum: {:.2e}\\n", error);
    }

    Ok(())
}

/// Example 2: Rosenbrock function optimization
fn rosenbrock_function_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 2: Rosenbrock Function Optimization ===");

    // Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
    struct RosenbrockProblem {
        a: f64,
        b: f64,
    }

    impl StochasticGradientFunction for RosenbrockProblem {
        fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> Array1<f64> {
            let x1 = x[0];
            let x2 = x[1];

            Array1::from_vec(vec![
                -2.0 * (self.a - x1) - 4.0 * self.b * x1 * (x2 - x1 * x1),
                2.0 * self.b * (x2 - x1 * x1),
            ])
        }

        fn compute_value(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> f64 {
            let x1 = x[0];
            let x2 = x[1];
            (self.a - x1).powi(2) + self.b * (x2 - x1 * x1).powi(2)
        }
    }

    let x0 = Array1::from_vec(vec![-1.2, 1.0]);

    println!("Function: Rosenbrock with a=1, b=100");
    println!("Starting point: [{:.1}, {:.1}]", x0[0], x0[1]);
    println!("Global minimum: [1, 1] with f = 0\\n");

    // Test different optimizers with their optimal settings
    let test_cases = [
        ("Adam", "standard"),
        ("AdamW", "with weight decay"),
        ("RMSProp", "standard"),
        ("Momentum + Nesterov", "accelerated"),
    ];

    for (name, variant) in &test_cases {
        println!("{} ({}):", name, variant);

        let start = Instant::now();
        let result = match *name {
            "Adam" => {
                let options = AdamOptions {
                    learning_rate: 0.01,
                    max_iter: 1000,
                    tol: 1e-6,
                    ..Default::default()
                };
                let grad_func = RosenbrockProblem { a: 1.0, b: 100.0 };
                let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));
                minimize_adam(grad_func, x0.clone(), data_provider, options)?
            }
            "AdamW" => {
                let options = AdamWOptions {
                    learning_rate: 0.01,
                    weight_decay: 0.001,
                    max_iter: 1000,
                    tol: 1e-6,
                    ..Default::default()
                };
                let grad_func = RosenbrockProblem { a: 1.0, b: 100.0 };
                let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));
                minimize_adamw(grad_func, x0.clone(), data_provider, options)?
            }
            "RMSProp" => {
                let options = RMSPropOptions {
                    learning_rate: 0.01,
                    decay_rate: 0.99,
                    max_iter: 1000,
                    tol: 1e-6,
                    ..Default::default()
                };
                let grad_func = RosenbrockProblem { a: 1.0, b: 100.0 };
                let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));
                minimize_rmsprop(grad_func, x0.clone(), data_provider, options)?
            }
            "Momentum + Nesterov" => {
                let options = MomentumOptions {
                    learning_rate: 0.01,
                    momentum: 0.9,
                    nesterov: true,
                    max_iter: 1000,
                    tol: 1e-6,
                    ..Default::default()
                };
                let grad_func = RosenbrockProblem { a: 1.0, b: 100.0 };
                let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));
                minimize_sgd_momentum(grad_func, x0.clone(), data_provider, options)?
            }
            _ => unreachable!(),
        };
        let duration = start.elapsed();

        println!("  Solution: [{:.6}, {:.6}]", result.x[0], result.x[1]);
        println!("  Function value: {:.2e}", result.fun);
        println!("  Iterations: {}", result.iterations);
        println!("  Success: {}", result.success);
        println!("  Time: {:.2} ms", duration.as_millis());

        let error = ((result.x[0] - 1.0).powi(2) + (result.x[1] - 1.0).powi(2)).sqrt();
        println!("  Distance from optimum: {:.2e}\\n", error);
    }

    Ok(())
}

/// Example 3: Logistic regression with stochastic optimization
fn logistic_regression_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 3: Logistic Regression ===");

    // Synthetic binary classification dataset
    struct LogisticRegressionProblem {
        features: Array2<f64>,
        labels: Array1<f64>,
    }

    impl StochasticGradientFunction for LogisticRegressionProblem {
        fn compute_gradient(
            &mut self,
            params: &ArrayView1<f64>,
            batch_indices: &[f64],
        ) -> Array1<f64> {
            let indices: Vec<usize> = batch_indices.iter().map(|&x| x as usize).collect();
            let mut gradient = Array1::zeros(params.len());

            for &i in &indices {
                if i < self.labels.len() {
                    let x_i = self.features.row(i);
                    let y_i = self.labels[i];

                    // Compute prediction: sigmoid(w^T x)
                    let z = x_i.dot(params);
                    let prediction = 1.0 / (1.0 + (-z).exp());

                    // Gradient: (prediction - y) * x
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

                    // Logistic loss: -y*log(p) - (1-y)*log(1-p)
                    loss += -y_i * prediction.ln() - (1.0 - y_i) * (1.0 - prediction).ln();
                }
            }

            loss / indices.len() as f64
        }
    }

    // Generate synthetic dataset
    let n_samples = 1000;
    let n_features = 5;

    // Create random features and true weights
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut features = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);
    let true_weights =
        Array1::from_vec((0..n_features).map(|_| rng.gen_range(-1.0..1.0)).collect());

    for i in 0..n_samples {
        for j in 0..n_features {
            features[[i, j]] = rng.gen_range(-2.0..2.0);
        }

        let z = features.row(i).dot(&true_weights) + rng.gen_range(-0.5..0.5); // Add noise
        labels[i] = if z > 0.0 { 1.0 } else { 0.0 };
    }

    println!("Dataset: {} samples, {} features", n_samples, n_features);
    println!(
        "True weights: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
        true_weights[0], true_weights[1], true_weights[2], true_weights[3], true_weights[4]
    );

    let x0 = Array1::zeros(n_features); // Start from zero weights
    let indices: Vec<f64> = (0..n_samples).map(|i| i as f64).collect();
    let data_provider = Box::new(InMemoryDataProvider::new(indices));

    // Test different optimizers on this ML problem
    let ml_methods = [
        ("SGD", "basic"),
        ("Adam", "adaptive"),
        ("AdamW", "with regularization"),
    ];

    for (name, desc) in &ml_methods {
        println!("\\n{} ({}):", name, desc);

        let start = Instant::now();
        let result = match *name {
            "SGD" => {
                let options = SGDOptions {
                    learning_rate: 0.1,
                    max_iter: 500,
                    batch_size: Some(32),
                    tol: 1e-6,
                    ..Default::default()
                };
                let grad_func = LogisticRegressionProblem {
                    features: features.clone(),
                    labels: labels.clone(),
                };
                minimize_sgd(grad_func, x0.clone(), data_provider.clone(), options)?
            }
            "Adam" => {
                let options = AdamOptions {
                    learning_rate: 0.01,
                    max_iter: 500,
                    batch_size: Some(32),
                    tol: 1e-6,
                    ..Default::default()
                };
                let grad_func = LogisticRegressionProblem {
                    features: features.clone(),
                    labels: labels.clone(),
                };
                minimize_adam(grad_func, x0.clone(), data_provider.clone(), options)?
            }
            "AdamW" => {
                let options = AdamWOptions {
                    learning_rate: 0.01,
                    weight_decay: 0.01,
                    max_iter: 500,
                    batch_size: Some(32),
                    tol: 1e-6,
                    ..Default::default()
                };
                let grad_func = LogisticRegressionProblem {
                    features: features.clone(),
                    labels: labels.clone(),
                };
                minimize_adamw(grad_func, x0.clone(), data_provider.clone(), options)?
            }
            _ => unreachable!(),
        };
        let duration = start.elapsed();

        println!(
            "  Learned weights: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
            result.x[0], result.x[1], result.x[2], result.x[3], result.x[4]
        );
        println!("  Final loss: {:.4}", result.fun);
        println!("  Iterations: {}", result.iterations);
        println!("  Time: {:.2} ms", duration.as_millis());

        // Calculate weight recovery error
        let weight_error = (&result.x - &true_weights).mapv(|x| x.abs()).sum() / n_features as f64;
        println!("  Average weight error: {:.4}", weight_error);
    }

    Ok(())
}

/// Example 4: Noisy optimization problem
fn noisy_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n=== Example 4: Noisy Optimization Problem ===");

    // Function with additive noise to test robustness
    struct NoisyQuadraticProblem {
        noise_level: f64,
    }

    impl StochasticGradientFunction for NoisyQuadraticProblem {
        fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> Array1<f64> {
            use rand::Rng;
            let mut rng = rand::thread_rng();

            // True gradient: 2*x, plus noise
            x.mapv(|xi| {
                let true_grad = 2.0 * xi;
                let noise = rng.random_range(-self.noise_level..self.noise_level);
                true_grad + noise
            })
        }

        fn compute_value(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> f64 {
            use rand::Rng;
            let mut rng = rand::thread_rng();

            // True function: sum(x_i^2), plus noise
            let true_value = x.mapv(|xi| xi * xi).sum();
            let noise = rng.random_range(-self.noise_level..self.noise_level);
            true_value + noise
        }
    }

    let x0 = Array1::from_vec(vec![3.0, -2.0, 1.5]);
    let noise_levels = [0.1, 0.5, 1.0];

    println!("Function: f(x) = sum(x_i^2) + noise");
    println!(
        "Starting point: [{:.1}, {:.1}, {:.1}]\\n",
        x0[0], x0[1], x0[2]
    );

    for &noise_level in &noise_levels {
        println!("Noise level: {:.1}", noise_level);

        let robust_methods = [
            ("Adam", "adaptive moment estimation"),
            ("RMSProp", "adaptive learning rate"),
        ];

        for (name, desc) in &robust_methods {
            let start = Instant::now();
            let result = match *name {
                "Adam" => {
                    let options = AdamOptions {
                        learning_rate: 0.01,
                        max_iter: 500,
                        batch_size: Some(10),
                        gradient_clip: Some(1.0), // Clip for noise robustness
                        tol: 1e-4,                // Looser tolerance due to noise
                        ..Default::default()
                    };
                    let grad_func = NoisyQuadraticProblem { noise_level };
                    let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));
                    minimize_adam(grad_func, x0.clone(), data_provider, options)?
                }
                "RMSProp" => {
                    let options = RMSPropOptions {
                        learning_rate: 0.01,
                        max_iter: 500,
                        batch_size: Some(10),
                        gradient_clip: Some(1.0),
                        tol: 1e-4,
                        ..Default::default()
                    };
                    let grad_func = NoisyQuadraticProblem { noise_level };
                    let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));
                    minimize_rmsprop(grad_func, x0.clone(), data_provider, options)?
                }
                _ => unreachable!(),
            };
            let duration = start.elapsed();

            println!(
                "  {} ({}): final = [{:.4}, {:.4}, {:.4}], loss = {:.4}, time = {:.1}ms",
                name,
                desc,
                result.x[0],
                result.x[1],
                result.x[2],
                result.fun,
                duration.as_millis()
            );
        }
        println!();
    }

    Ok(())
}

/// Example 5: Learning rate schedule comparison
fn learning_rate_schedule_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 5: Learning Rate Schedule Comparison ===");

    struct QuadraticProblem;

    impl StochasticGradientFunction for QuadraticProblem {
        fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> Array1<f64> {
            x.mapv(|xi| 2.0 * xi)
        }

        fn compute_value(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> f64 {
            x.mapv(|xi| xi * xi).sum()
        }
    }

    let x0 = Array1::from_vec(vec![2.0, -2.0]);

    let schedules = [
        ("Constant", LearningRateSchedule::Constant),
        (
            "Exponential Decay",
            LearningRateSchedule::ExponentialDecay { decay_rate: 0.95 },
        ),
        ("Linear Decay", LearningRateSchedule::LinearDecay),
        ("Cosine Annealing", LearningRateSchedule::CosineAnnealing),
        (
            "Inverse Time",
            LearningRateSchedule::InverseTimeDecay { decay_rate: 0.01 },
        ),
    ];

    println!("Testing different learning rate schedules with Adam:");
    println!("Starting point: [{:.1}, {:.1}]\\n", x0[0], x0[1]);

    for (name, schedule) in &schedules {
        let options = AdamOptions {
            learning_rate: 0.1,
            max_iter: 200,
            lr_schedule: schedule.clone(),
            tol: 1e-8,
            ..Default::default()
        };

        let grad_func = QuadraticProblem;
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 50]));

        let start = Instant::now();
        let result = minimize_adam(grad_func, x0.clone(), data_provider, options)?;
        let duration = start.elapsed();

        println!(
            "{:15}: final = [{:.6}, {:.6}], loss = {:.2e}, iter = {}, time = {:.1}ms",
            name,
            result.x[0],
            result.x[1],
            result.fun,
            result.iterations,
            duration.as_millis()
        );
    }

    Ok(())
}

/// Example 6: Batch size effects
fn batch_size_effects_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n=== Example 6: Batch Size Effects ===");

    struct QuadraticProblem;

    impl StochasticGradientFunction for QuadraticProblem {
        fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> Array1<f64> {
            x.mapv(|xi| 2.0 * xi)
        }

        fn compute_value(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> f64 {
            x.mapv(|xi| xi * xi).sum()
        }
    }

    let x0 = Array1::from_vec(vec![1.0, -1.0]);
    let dataset_size = 1000;
    let batch_sizes = [1, 10, 32, 100, dataset_size]; // Include full batch

    println!("Testing batch size effects with SGD:");
    println!(
        "Dataset size: {}, Starting point: [{:.1}, {:.1}]\\n",
        dataset_size, x0[0], x0[1]
    );

    for &batch_size in &batch_sizes {
        let batch_desc = if batch_size == dataset_size {
            "Full batch".to_string()
        } else {
            format!("Mini-batch ({})", batch_size)
        };

        let options = SGDOptions {
            learning_rate: 0.01,
            max_iter: 200,
            batch_size: Some(batch_size),
            tol: 1e-8,
            ..Default::default()
        };

        let grad_func = QuadraticProblem;
        let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; dataset_size]));

        let start = Instant::now();
        let result = minimize_sgd(grad_func, x0.clone(), data_provider, options)?;
        let duration = start.elapsed();

        let convergence_quality = if result.success {
            "Converged"
        } else {
            "Max iter"
        };

        println!(
            "{:15}: final = [{:.6}, {:.6}], loss = {:.2e}, {} in {} iter, time = {:.1}ms",
            batch_desc,
            result.x[0],
            result.x[1],
            result.fun,
            convergence_quality,
            result.iterations,
            duration.as_millis()
        );
    }

    println!("\\nObservations:");
    println!("- Smaller batches: More stochastic, potentially faster initial progress");
    println!("- Larger batches: More stable convergence, better final accuracy");
    println!("- Full batch: Most stable, but potentially slower per iteration");

    Ok(())
}

/// Helper trait to enable cloning of Box<dyn DataProvider>
trait CloneableDataProvider: DataProvider {
    fn clone_box(&self) -> Box<dyn DataProvider>;
}

impl CloneableDataProvider for InMemoryDataProvider {
    fn clone_box(&self) -> Box<dyn DataProvider> {
        Box::new(InMemoryDataProvider::new(self.get_full_data()))
    }
}

// We need to implement Clone for the trait object
impl Clone for Box<dyn DataProvider> {
    fn clone(&self) -> Box<dyn DataProvider> {
        // This is a simplification - in real code you'd want a proper solution
        // For now, just create a new InMemoryDataProvider with the same data
        Box::new(InMemoryDataProvider::new(self.get_full_data()))
    }
}
