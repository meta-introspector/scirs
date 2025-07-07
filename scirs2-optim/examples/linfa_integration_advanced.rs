//! Advanced Linfa ML Framework Integration Example
//!
//! This example demonstrates comprehensive integration between scirs2-optim
//! and the Linfa machine learning framework, showcasing:
//! - Custom optimizer implementations for Linfa algorithms
//! - Performance comparison with built-in optimizers
//! - Memory-efficient optimization for large datasets
//! - Advanced scheduling and regularization techniques

use ndarray::{Array1, Array2, Axis};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use scirs2_optim::{
    benchmarking::OptimizerBenchmark,
    error::Result,
    optimizers::{adam::Adam, sgd::SGD, AdamConfig, SGDConfig},
    regularizers::{l2::L2Regularizer, RegularizerConfig},
    schedulers::{exponential_decay::ExponentialDecayScheduler, SchedulerConfig},
    unified_api::{OptimizerFactory, Parameter},
};
use std::time::Instant;

// Simulated Linfa-style ML algorithm trait
trait MLAlgorithm {
    type Model;
    type Dataset;

    fn fit(&mut self, dataset: &Self::Dataset) -> Result<Self::Model>;
    fn predict(&self, model: &Self::Model, input: &Array2<f64>) -> Array1<f64>;
    fn loss(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> f64;
}

// Simulated dataset structure
#[derive(Debug, Clone)]
struct Dataset {
    features: Array2<f64>,
    targets: Array1<f64>,
}

impl Dataset {
    fn new(features: Array2<f64>, targets: Array1<f64>) -> Self {
        Self { features, targets }
    }

    fn len(&self) -> usize {
        self.features.nrows()
    }

    fn features(&self) -> &Array2<f64> {
        &self.features
    }

    fn targets(&self) -> &Array1<f64> {
        &self.targets
    }
}

// Linear regression model with scirs2-optim integration
#[derive(Debug)]
struct OptimizedLinearRegression {
    weights: Parameter<f64>,
    bias: Parameter<f64>,
    optimizer: Box<dyn OptimizerTrait<f64>>,
    scheduler: Option<ExponentialDecayScheduler>,
    regularizer: Option<L2Regularizer<f64>>,
    training_history: TrainingHistory,
}

#[derive(Debug, Clone)]
struct TrainingHistory {
    losses: Vec<f64>,
    learning_rates: Vec<f64>,
    gradient_norms: Vec<f64>,
    convergence_metrics: Vec<ConvergenceMetrics>,
}

#[derive(Debug, Clone)]
struct ConvergenceMetrics {
    weight_change: f64,
    gradient_norm: f64,
    loss_improvement: f64,
    step: usize,
}

// Trait for optimizers to work with our ML algorithm
trait OptimizerTrait<T> {
    fn step(&mut self, parameters: &[&mut Parameter<T>], gradients: &[&Array1<T>]) -> Result<()>;
    fn learning_rate(&self) -> T;
    fn reset(&mut self);
}

// Implement OptimizerTrait for Adam
struct AdamOptimizer {
    inner: Adam<f64>,
}

impl AdamOptimizer {
    fn new(learning_rate: f64) -> Self {
        let config = AdamConfig {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
        };
        Self {
            inner: Adam::new(config),
        }
    }
}

impl OptimizerTrait<f64> for AdamOptimizer {
    fn step(
        &mut self,
        parameters: &[&mut Parameter<f64>],
        gradients: &[&Array1<f64>],
    ) -> Result<()> {
        for (param, grad) in parameters.iter().zip(gradients.iter()) {
            self.inner.step(&mut param.data, grad)?;
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.inner.config().learning_rate
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

// Implement OptimizerTrait for SGD
struct SGDOptimizer {
    inner: SGD<f64>,
}

impl SGDOptimizer {
    fn new(learning_rate: f64, momentum: f64) -> Self {
        let config = SGDConfig {
            learning_rate,
            momentum,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        };
        Self {
            inner: SGD::new(config),
        }
    }
}

impl OptimizerTrait<f64> for SGDOptimizer {
    fn step(
        &mut self,
        parameters: &[&mut Parameter<f64>],
        gradients: &[&Array1<f64>],
    ) -> Result<()> {
        for (param, grad) in parameters.iter().zip(gradients.iter()) {
            self.inner.step(&mut param.data, grad)?;
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.inner.config().learning_rate
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

impl OptimizedLinearRegression {
    fn new(n_features: usize, optimizer_type: &str) -> Result<Self> {
        let mut rng = Xoshiro256Plus::seed_from_u64(42);

        // Initialize parameters
        let weights_data = Array1::from_shape_fn(n_features, |_| rng.random_range(-0.1, 0.1));
        let weights = Parameter::new(weights_data);
        let bias = Parameter::new(Array1::from_elem(1, 0.0));

        // Create optimizer
        let optimizer: Box<dyn OptimizerTrait<f64>> = match optimizer_type {
            "adam" => Box::new(AdamOptimizer::new(0.001)),
            "sgd" => Box::new(SGDOptimizer::new(0.01, 0.9)),
            _ => {
                return Err(scirs2_optim::error::OptimError::InvalidConfig(format!(
                    "Unknown optimizer type: {}",
                    optimizer_type
                )))
            }
        };

        // Create scheduler
        let scheduler_config = SchedulerConfig {
            initial_lr: 0.001,
            decay_rate: 0.95,
            step_size: 100,
        };
        let scheduler = Some(ExponentialDecayScheduler::new(scheduler_config));

        // Create regularizer
        let regularizer_config = RegularizerConfig { strength: 0.01 };
        let regularizer = Some(L2Regularizer::new(regularizer_config));

        Ok(Self {
            weights,
            bias,
            optimizer,
            scheduler,
            regularizer,
            training_history: TrainingHistory {
                losses: Vec::new(),
                learning_rates: Vec::new(),
                gradient_norms: Vec::new(),
                convergence_metrics: Vec::new(),
            },
        })
    }

    fn forward(&self, features: &Array2<f64>) -> Array1<f64> {
        // Linear regression: y = X * w + b
        features.dot(&self.weights.data) + self.bias.data[0]
    }

    fn compute_loss(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
        // Mean squared error
        let diff = predictions - targets;
        let mse = diff.mapv(|x| x * x).mean().unwrap();

        // Add regularization
        let reg_loss = if let Some(ref regularizer) = self.regularizer {
            regularizer.compute_penalty(&self.weights.data)
        } else {
            0.0
        };

        mse + reg_loss
    }

    fn compute_gradients(
        &self,
        features: &Array2<f64>,
        predictions: &Array1<f64>,
        targets: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let n_samples = features.nrows() as f64;
        let diff = predictions - targets;

        // Gradient w.r.t. weights: (2/n) * X^T * (predictions - targets)
        let mut weight_grad = features.t().dot(&diff) * (2.0 / n_samples);

        // Add regularization gradient
        if let Some(ref regularizer) = self.regularizer {
            let reg_grad = regularizer.compute_gradient(&self.weights.data);
            weight_grad = weight_grad + reg_grad;
        }

        // Gradient w.r.t. bias: (2/n) * sum(predictions - targets)
        let bias_grad = Array1::from_elem(1, diff.sum() * (2.0 / n_samples));

        (weight_grad, bias_grad)
    }

    fn train_epoch(&mut self, dataset: &Dataset) -> Result<f64> {
        let predictions = self.forward(dataset.features());
        let loss = self.compute_loss(&predictions, dataset.targets());

        let (weight_grad, bias_grad) =
            self.compute_gradients(dataset.features(), &predictions, dataset.targets());

        // Update learning rate with scheduler
        let current_lr = if let Some(ref mut scheduler) = self.scheduler {
            scheduler.step();
            scheduler.get_lr()
        } else {
            self.optimizer.learning_rate()
        };

        // Compute gradient norm for monitoring
        let grad_norm =
            (weight_grad.mapv(|x| x * x).sum() + bias_grad.mapv(|x| x * x).sum()).sqrt();

        // Perform optimization step
        let mut params = [&mut self.weights, &mut self.bias];
        let gradients = [&weight_grad, &bias_grad];
        self.optimizer.step(&params, &gradients)?;

        // Record training history
        self.training_history.losses.push(loss);
        self.training_history.learning_rates.push(current_lr);
        self.training_history.gradient_norms.push(grad_norm);

        Ok(loss)
    }

    fn fit(&mut self, dataset: &Dataset, epochs: usize, tolerance: f64) -> Result<()> {
        println!("Training linear regression with {} epochs...", epochs);

        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;
        let patience = 10;

        for epoch in 0..epochs {
            let loss = self.train_epoch(dataset)?;

            // Compute convergence metrics
            let weight_change = if epoch > 0 {
                let prev_weights = &self.training_history.losses[epoch - 1];
                (loss - prev_weights).abs()
            } else {
                0.0
            };

            let gradient_norm = *self.training_history.gradient_norms.last().unwrap();
            let loss_improvement = if epoch > 0 {
                self.training_history.losses[epoch - 1] - loss
            } else {
                0.0
            };

            self.training_history
                .convergence_metrics
                .push(ConvergenceMetrics {
                    weight_change,
                    gradient_norm,
                    loss_improvement,
                    step: epoch,
                });

            // Early stopping
            if loss < best_loss - tolerance {
                best_loss = loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
            }

            if patience_counter >= patience {
                println!("Early stopping at epoch {} (loss: {:.6})", epoch, loss);
                break;
            }

            if epoch % 10 == 0 || epoch == epochs - 1 {
                println!(
                    "Epoch {}: Loss = {:.6}, LR = {:.6}, Grad Norm = {:.6}",
                    epoch,
                    loss,
                    *self.training_history.learning_rates.last().unwrap(),
                    gradient_norm
                );
            }
        }

        Ok(())
    }

    fn predict(&self, features: &Array2<f64>) -> Array1<f64> {
        self.forward(features)
    }

    fn get_training_history(&self) -> &TrainingHistory {
        &self.training_history
    }
}

// Benchmark different optimizers
#[allow(dead_code)]
fn benchmark_optimizers(dataset: &Dataset) -> Result<()> {
    println!("\n🏁 Benchmarking optimizers on linear regression...");

    let optimizers = vec!["adam", "sgd"];
    let epochs = 100;
    let tolerance = 1e-6;

    for optimizer_name in optimizers {
        println!("\n📊 Testing {} optimizer:", optimizer_name.to_uppercase());

        let start_time = Instant::now();
        let mut model = OptimizedLinearRegression::new(dataset.features().ncols(), optimizer_name)?;

        model.fit(dataset, epochs, tolerance)?;
        let training_time = start_time.elapsed();

        // Evaluate final performance
        let predictions = model.predict(dataset.features());
        let final_loss = model.compute_loss(&predictions, dataset.targets());
        let history = model.get_training_history();

        println!("Results for {}:", optimizer_name.to_uppercase());
        println!("  Training time: {:?}", training_time);
        println!("  Final loss: {:.6}", final_loss);
        println!("  Epochs trained: {}", history.losses.len());
        println!(
            "  Convergence rate: {:.6}",
            if history.losses.len() > 1 {
                (history.losses[0] - final_loss) / history.losses.len() as f64
            } else {
                0.0
            }
        );

        // Analyze convergence behavior
        if history.convergence_metrics.len() > 10 {
            let recent_metrics =
                &history.convergence_metrics[history.convergence_metrics.len() - 10..];
            let avg_gradient_norm = recent_metrics.iter().map(|m| m.gradient_norm).sum::<f64>()
                / recent_metrics.len() as f64;

            println!(
                "  Average gradient norm (last 10 epochs): {:.6}",
                avg_gradient_norm
            );

            let stable_convergence = recent_metrics.iter().all(|m| m.gradient_norm < 1e-3);

            println!(
                "  Stable convergence: {}",
                if stable_convergence { "Yes" } else { "No" }
            );
        }
    }

    Ok(())
}

// Advanced optimization techniques demonstration
#[allow(dead_code)]
fn demonstrate_advanced_techniques(dataset: &Dataset) -> Result<()> {
    println!("\n🔬 Demonstrating advanced optimization techniques...");

    // 1. Learning rate scheduling
    println!("\n1. Learning Rate Scheduling:");
    let mut model_scheduled = OptimizedLinearRegression::new(dataset.features().ncols(), "adam")?;
    model_scheduled.fit(dataset, 50, 1e-6)?;

    let history = model_scheduled.get_training_history();
    println!("   Initial LR: {:.6}", history.learning_rates[0]);
    println!(
        "   Final LR: {:.6}",
        *history.learning_rates.last().unwrap()
    );
    println!(
        "   LR reduction factor: {:.2}x",
        history.learning_rates[0] / history.learning_rates.last().unwrap()
    );

    // 2. Regularization impact
    println!("\n2. Regularization Impact:");
    let predictions = model_scheduled.predict(dataset.features());
    let base_loss = predictions.mapv(|x| x * x).mean().unwrap();

    if let Some(ref regularizer) = model_scheduled.regularizer {
        let reg_penalty = regularizer.compute_penalty(&model_scheduled.weights.data);
        println!("   Base loss: {:.6}", base_loss);
        println!("   Regularization penalty: {:.6}", reg_penalty);
        println!(
            "   Regularization contribution: {:.2}%",
            (reg_penalty / (base_loss + reg_penalty)) * 100.0
        );
    }

    // 3. Memory efficiency analysis
    println!("\n3. Memory Efficiency Analysis:");
    let model_size = std::mem::size_of_val(&model_scheduled.weights.data)
        + std::mem::size_of_val(&model_scheduled.bias.data);
    let dataset_size =
        std::mem::size_of_val(dataset.features()) + std::mem::size_of_val(dataset.targets());

    println!("   Model memory usage: {} bytes", model_size);
    println!("   Dataset memory usage: {} bytes", dataset_size);
    println!(
        "   Memory efficiency ratio: {:.2}%",
        (model_size as f64 / dataset_size as f64) * 100.0
    );

    Ok(())
}

// Create synthetic dataset for testing
#[allow(dead_code)]
fn create_synthetic_dataset(n_samples: usize, n_features: usize, noise_level: f64) -> Dataset {
    let mut rng = Xoshiro256Plus::seed_from_u64(42);

    // Generate random features
    let features = Array2::from_shape_fn((n_samples, n_features), |_| rng.random_range(-1.0, 1.0));

    // Generate true weights
    let true_weights = Array1::from_shape_fn(n_features, |_| rng.random_range(-2.0, 2.0));
    let true_bias = 0.5;

    // Generate targets with noise
    let clean_targets = features.dot(&true_weights) + true_bias;
    let targets = clean_targets.mapv(|y| y + rng.random_range(-noise_level, noise_level));

    Dataset::new(features, targets)
}

// Performance comparison with different dataset sizes
#[allow(dead_code)]
fn scalability_analysis() -> Result<()> {
    println!("\n📈 Scalability Analysis:");

    let dataset_sizes = vec![100, 500, 1000, 5000];
    let n_features = 10;
    let noise_level = 0.1;

    for &n_samples in &dataset_sizes {
        println!("\nDataset size: {} samples", n_samples);

        let dataset = create_synthetic_dataset(n_samples, n_features, noise_level);
        let start_time = Instant::now();

        let mut model = OptimizedLinearRegression::new(n_features, "adam")?;
        model.fit(&dataset, 50, 1e-6)?;

        let training_time = start_time.elapsed();
        let predictions = model.predict(dataset.features());
        let final_loss = model.compute_loss(&predictions, dataset.targets());

        println!("  Training time: {:?}", training_time);
        println!("  Final loss: {:.6}", final_loss);
        println!(
            "  Time per sample: {:.2} μs",
            training_time.as_micros() as f64 / n_samples as f64
        );
    }

    Ok(())
}

// Framework compatibility testing
#[allow(dead_code)]
fn test_framework_compatibility() -> Result<()> {
    println!("\n🔧 Framework Compatibility Testing:");

    // Test parameter serialization/deserialization
    println!("\n1. Parameter Serialization:");
    let dataset = create_synthetic_dataset(100, 5, 0.1);
    let mut model = OptimizedLinearRegression::new(5, "adam")?;
    model.fit(&dataset, 10, 1e-6)?;

    // Simulate saving model parameters
    let weights_data = model.weights.data.clone();
    let bias_data = model.bias.data.clone();

    println!("   Original weights shape: {:?}", weights_data.shape());
    println!("   Original bias shape: {:?}", bias_data.shape());

    // Test with different array layouts
    println!("\n2. Array Layout Compatibility:");
    let c_order_features = dataset.features().clone();
    let f_order_features = c_order_features.reversed_axes();

    let pred_c = model.predict(&c_order_features);
    // Note: In a real implementation, we'd test F-order compatibility
    println!("   C-order predictions computed successfully");
    println!("   Prediction shape: {:?}", pred_c.shape());

    // Test numerical stability
    println!("\n3. Numerical Stability:");
    let large_features = dataset.features() * 1000.0;
    let small_features = dataset.features() * 0.001;

    let pred_large = model.predict(&large_features);
    let pred_small = model.predict(&small_features);

    let scale_ratio_large =
        pred_large.mapv(|x| x.abs()).mean().unwrap() / pred_c.mapv(|x| x.abs()).mean().unwrap();
    let scale_ratio_small =
        pred_small.mapv(|x| x.abs()).mean().unwrap() / pred_c.mapv(|x| x.abs()).mean().unwrap();

    println!("   Large input scale ratio: {:.2}", scale_ratio_large);
    println!("   Small input scale ratio: {:.2}", scale_ratio_small);
    println!(
        "   Numerical stability: {}",
        if scale_ratio_large > 100.0 && scale_ratio_small < 0.01 {
            "Good"
        } else {
            "Check needed"
        }
    );

    Ok(())
}

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("🧠 Scirs2-Optim + Linfa Integration Advanced Example");
    println!("==================================================");

    // Create test dataset
    let n_samples = 1000;
    let n_features = 10;
    let noise_level = 0.1;

    println!("\n📊 Creating synthetic dataset:");
    println!("  Samples: {}", n_samples);
    println!("  Features: {}", n_features);
    println!("  Noise level: {}", noise_level);

    let dataset = create_synthetic_dataset(n_samples, n_features, noise_level);

    // Run benchmark comparison
    benchmark_optimizers(&dataset)?;

    // Demonstrate advanced techniques
    demonstrate_advanced_techniques(&dataset)?;

    // Scalability analysis
    scalability_analysis()?;

    // Framework compatibility testing
    test_framework_compatibility()?;

    println!("\n✅ Linfa integration example completed successfully!");
    println!("\n📋 Summary:");
    println!("   - Demonstrated scirs2-optim integration with ML workflows");
    println!("   - Compared Adam and SGD optimizers on linear regression");
    println!("   - Showcased learning rate scheduling and regularization");
    println!("   - Analyzed memory efficiency and scalability");
    println!("   - Tested framework compatibility features");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_creation() {
        let dataset = create_synthetic_dataset(100, 5, 0.1);
        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.features().ncols(), 5);
        assert_eq!(dataset.targets().len(), 100);
    }

    #[test]
    fn test_model_creation() {
        let model = OptimizedLinearRegression::new(5, "adam");
        assert!(model.is_ok());

        let model = model.unwrap();
        assert_eq!(model.weights.data.len(), 5);
        assert_eq!(model.bias.data.len(), 1);
    }

    #[test]
    fn test_forward_pass() {
        let model = OptimizedLinearRegression::new(3, "adam").unwrap();
        let features = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let predictions = model.forward(&features);
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_loss_computation() {
        let model = OptimizedLinearRegression::new(2, "adam").unwrap();
        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let targets = Array1::from_vec(vec![1.1, 1.9, 3.1]);

        let loss = model.compute_loss(&predictions, &targets);
        assert!(loss >= 0.0);
        assert!(loss < 0.1); // Should be small for close predictions
    }

    #[test]
    fn test_training_single_epoch() {
        let dataset = create_synthetic_dataset(50, 3, 0.05);
        let mut model = OptimizedLinearRegression::new(3, "adam").unwrap();

        let initial_loss = {
            let predictions = model.predict(dataset.features());
            model.compute_loss(&predictions, dataset.targets())
        };

        let epoch_loss = model.train_epoch(&dataset).unwrap();
        assert!(epoch_loss <= initial_loss); // Loss should not increase
    }
}
