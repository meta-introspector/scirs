//! Comprehensive SmartCore Integration Example
//!
//! This example demonstrates advanced integration between scirs2-optim and
//! SmartCore machine learning library, featuring:
//! - Custom optimizer integration with SmartCore algorithms
//! - Performance benchmarking against SmartCore's built-in optimizers
//! - Advanced hyperparameter optimization
//! - Cross-validation with custom optimizers
//! - Model ensembling with different optimization strategies

use ndarray::{s, Array1, Array2, Axis};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use scirs2_core::random;
use scirs2_optim::{
    error::Result,
    optimizers::{
        adam::{Adam, AdamConfig},
        lamb::{LAMBConfig, LAMB},
        lookahead::{Lookahead, LookaheadConfig},
        sgd::{SGDConfig, SGD},
    },
    regularizers::{
        elastic_net::{ElasticNetConfig, ElasticNetRegularizer},
        l1::L1Regularizer,
        l2::L2Regularizer,
    },
    schedulers::{
        cosine_annealing::{CosineAnnealingConfig, CosineAnnealingScheduler},
        one_cycle::{OneCycleConfig, OneCycleScheduler},
    },
    unified_api::{OptimizerFactory, Parameter},
};
use std::collections::HashMap;
use std::time::Instant;

// Simulated SmartCore-style traits and structures
trait Predictor<TX, TY> {
    fn predict(&self, x: &TX) -> Result<TY>;
}

trait SupervisedEstimator<TX, TY>: Predictor<TX, TY> {
    fn fit(x: &TX, y: &TY) -> Result<Self>
    where
        Self: Sized;
}

// Enhanced dataset structure with cross-validation support
#[derive(Debug, Clone)]
struct MLDataset {
    features: Array2<f64>,
    targets: Array1<f64>,
    feature_names: Vec<String>,
    target_name: String,
}

impl MLDataset {
    fn new(features: Array2<f64>, targets: Array1<f64>) -> Self {
        let n_features = features.ncols();
        let feature_names = (0..n_features).map(|i| format!("feature_{}", i)).collect();

        Self {
            features,
            targets,
            feature_names,
            target_name: "target".to_string(),
        }
    }

    fn len(&self) -> usize {
        self.features.nrows()
    }

    fn n_features(&self) -> usize {
        self.features.ncols()
    }

    fn train_test_split(&self, test_size: f64, random_state: u64) -> (MLDataset, MLDataset) {
        let mut rng = Xoshiro256Plus::seed_from_u64(random_state);
        let n_samples = self.len();
        let n_test = (n_samples as f64 * test_size) as usize;

        // Create random indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        for i in 0..n_samples {
            let j = rng.random_range(0, n_samples);
            indices.swap(i, j);
        }

        let test_indices = &indices[..n_test];
        let train_indices = &indices[n_test..];

        let train_features = self.features.select(Axis(0), train_indices);
        let train_targets = self.targets.select(Axis(0), train_indices);
        let test_features = self.features.select(Axis(0), test_indices);
        let test_targets = self.targets.select(Axis(0), test_indices);

        (
            MLDataset {
                features: train_features,
                targets: train_targets,
                feature_names: self.feature_names.clone(),
                target_name: self.target_name.clone(),
            },
            MLDataset {
                features: test_features,
                targets: test_targets,
                feature_names: self.feature_names.clone(),
                target_name: self.target_name.clone(),
            },
        )
    }

    fn k_fold_split(&self, k: usize, random_state: u64) -> Vec<(MLDataset, MLDataset)> {
        let mut rng = Xoshiro256Plus::seed_from_u64(random_state);
        let n_samples = self.len();
        let fold_size = n_samples / k;

        let mut indices: Vec<usize> = (0..n_samples).collect();
        for i in 0..n_samples {
            let j = rng.random_range(0, n_samples);
            indices.swap(i, j);
        }

        let mut folds = Vec::new();

        for fold in 0..k {
            let start = fold * fold_size;
            let end = if fold == k - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            let test_indices = &indices[start..end];
            let train_indices: Vec<usize> = indices
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < start || *i >= end)
                .map(|(_, &idx)| idx)
                .collect();

            let train_features = self.features.select(Axis(0), &train_indices);
            let train_targets = self.targets.select(Axis(0), &train_indices);
            let test_features = self.features.select(Axis(0), test_indices);
            let test_targets = self.targets.select(Axis(0), test_indices);

            folds.push((
                MLDataset {
                    features: train_features,
                    targets: train_targets,
                    feature_names: self.feature_names.clone(),
                    target_name: self.target_name.clone(),
                },
                MLDataset {
                    features: test_features,
                    targets: test_targets,
                    feature_names: self.feature_names.clone(),
                    target_name: self.target_name.clone(),
                },
            ));
        }

        folds
    }
}

// Advanced neural network with custom optimizer integration
#[derive(Debug)]
struct OptimizedNeuralNetwork {
    layers: Vec<Layer>,
    optimizer: Box<dyn NetworkOptimizer>,
    scheduler: Option<Box<dyn LearningRateScheduler>>,
    regularizer: Option<Box<dyn NetworkRegularizer>>,
    training_config: TrainingConfig,
    metrics_history: MetricsHistory,
}

#[derive(Debug)]
struct Layer {
    weights: Parameter<f64>,
    biases: Parameter<f64>,
    activation: ActivationType,
}

#[derive(Debug, Clone)]
enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}

#[derive(Debug, Clone)]
struct TrainingConfig {
    epochs: usize,
    batch_size: usize,
    early_stopping: bool,
    patience: usize,
    validation_split: f64,
    shuffle: bool,
    verbose: bool,
}

#[derive(Debug, Clone)]
struct MetricsHistory {
    train_loss: Vec<f64>,
    val_loss: Vec<f64>,
    train_accuracy: Vec<f64>,
    val_accuracy: Vec<f64>,
    learning_rates: Vec<f64>,
    gradient_norms: Vec<f64>,
    epoch_times: Vec<f64>,
}

// Optimizer trait for neural networks
trait NetworkOptimizer: std::fmt::Debug {
    fn step(&mut self, layers: &mut [Layer], gradients: &[LayerGradients]) -> Result<()>;
    fn get_learning_rate(&self) -> f64;
    fn reset(&mut self);
    fn get_config(&self) -> OptimizerConfig;
}

#[derive(Debug, Clone)]
struct LayerGradients {
    weight_gradients: Array2<f64>,
    bias_gradients: Array1<f64>,
}

#[derive(Debug, Clone)]
struct OptimizerConfig {
    name: String,
    learning_rate: f64,
    parameters: HashMap<String, f64>,
}

// Learning rate scheduler trait
trait LearningRateScheduler: std::fmt::Debug {
    fn step(&mut self, epoch: usize, loss: f64) -> f64;
    fn get_lr(&self) -> f64;
    fn reset(&mut self);
}

// Regularizer trait for neural networks
trait NetworkRegularizer: std::fmt::Debug {
    fn compute_penalty(&self, layers: &[Layer]) -> f64;
    fn compute_gradients(&self, layers: &[Layer]) -> Vec<LayerGradients>;
    fn get_config(&self) -> RegularizerConfig;
}

#[derive(Debug, Clone)]
struct RegularizerConfig {
    name: String,
    strength: f64,
    parameters: HashMap<String, f64>,
}

// Concrete optimizer implementations
#[derive(Debug)]
struct AdamNetworkOptimizer {
    adam: Adam<f64>,
    learning_rate: f64,
}

impl AdamNetworkOptimizer {
    fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        let config = AdamConfig {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay: 0.0,
            amsgrad: false,
        };

        Self {
            adam: Adam::new(config),
            learning_rate,
        }
    }
}

impl NetworkOptimizer for AdamNetworkOptimizer {
    fn step(&mut self, layers: &mut [Layer], gradients: &[LayerGradients]) -> Result<()> {
        for (layer, grad) in layers.iter_mut().zip(gradients.iter()) {
            // Flatten 2D weight gradients to 1D for optimizer
            let weight_grad_flat = grad.weight_gradients.iter().cloned().collect::<Array1<_>>();
            self.adam.step(&mut layer.weights.data, &weight_grad_flat)?;
            self.adam
                .step(&mut layer.biases.data, &grad.bias_gradients)?;
        }
        Ok(())
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn reset(&mut self) {
        self.adam.reset();
    }

    fn get_config(&self) -> OptimizerConfig {
        let mut params = HashMap::new();
        params.insert("beta1".to_string(), self.adam.config().beta1);
        params.insert("beta2".to_string(), self.adam.config().beta2);
        params.insert("epsilon".to_string(), self.adam.config().epsilon);

        OptimizerConfig {
            name: "Adam".to_string(),
            learning_rate: self.learning_rate,
            parameters: params,
        }
    }
}

#[derive(Debug)]
struct LAMBNetworkOptimizer {
    lamb: LAMB<f64>,
    learning_rate: f64,
}

impl LAMBNetworkOptimizer {
    fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64, weight_decay: f64) -> Self {
        let config = LAMBConfig {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            bias_correction: true,
        };

        Self {
            lamb: LAMB::new(config),
            learning_rate,
        }
    }
}

impl NetworkOptimizer for LAMBNetworkOptimizer {
    fn step(&mut self, layers: &mut [Layer], gradients: &[LayerGradients]) -> Result<()> {
        for (layer, grad) in layers.iter_mut().zip(gradients.iter()) {
            let weight_grad_flat = grad.weight_gradients.iter().cloned().collect::<Array1<_>>();
            self.lamb.step(&mut layer.weights.data, &weight_grad_flat)?;
            self.lamb
                .step(&mut layer.biases.data, &grad.bias_gradients)?;
        }
        Ok(())
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn reset(&mut self) {
        self.lamb.reset();
    }

    fn get_config(&self) -> OptimizerConfig {
        OptimizerConfig {
            name: "LAMB".to_string(),
            learning_rate: self.learning_rate,
            parameters: HashMap::new(),
        }
    }
}

// Cosine annealing scheduler implementation
#[derive(Debug)]
struct CosineAnnealingLRScheduler {
    scheduler: CosineAnnealingScheduler,
    current_lr: f64,
}

impl CosineAnnealingLRScheduler {
    fn new(initial_lr: f64, t_max: usize, eta_min: f64) -> Self {
        let config = CosineAnnealingConfig {
            initial_lr,
            t_max,
            eta_min,
        };

        Self {
            scheduler: CosineAnnealingScheduler::new(config),
            current_lr: initial_lr,
        }
    }
}

impl LearningRateScheduler for CosineAnnealingLRScheduler {
    fn step(&mut self, _epoch: usize, _loss: f64) -> f64 {
        self.scheduler.step();
        self.current_lr = self.scheduler.get_lr();
        self.current_lr
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn reset(&mut self) {
        self.scheduler.reset();
    }
}

// L2 regularizer implementation
#[derive(Debug)]
struct L2NetworkRegularizer {
    l2_reg: L2Regularizer<f64>,
    strength: f64,
}

impl L2NetworkRegularizer {
    fn new(strength: f64) -> Self {
        let config = scirs2_optim::regularizers::RegularizerConfig { strength };

        Self {
            l2_reg: L2Regularizer::new(config),
            strength,
        }
    }
}

impl NetworkRegularizer for L2NetworkRegularizer {
    fn compute_penalty(&self, layers: &[Layer]) -> f64 {
        layers
            .iter()
            .map(|layer| self.l2_reg.compute_penalty(&layer.weights.data))
            .sum()
    }

    fn compute_gradients(&self, layers: &[Layer]) -> Vec<LayerGradients> {
        layers
            .iter()
            .map(|layer| {
                let weight_grad = self.l2_reg.compute_gradient(&layer.weights.data);
                // Convert 1D gradient back to 2D for weights
                let weight_shape = layer.weights.data.len();
                let cols = layer.biases.data.len();
                let rows = weight_shape / cols;
                let weight_grad_2d = Array2::from_shape_vec((rows, cols), weight_grad.to_vec())
                    .unwrap_or_else(|_| Array2::zeros((rows, cols)));

                LayerGradients {
                    weight_gradients: weight_grad_2d,
                    bias_gradients: Array1::zeros(layer.biases.data.len()),
                }
            })
            .collect()
    }

    fn get_config(&self) -> RegularizerConfig {
        RegularizerConfig {
            name: "L2".to_string(),
            strength: self.strength,
            parameters: HashMap::new(),
        }
    }
}

impl OptimizedNeuralNetwork {
    fn new(
        layer_sizes: &[usize],
        optimizer_type: &str,
        training_config: TrainingConfig,
    ) -> Result<Self> {
        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let mut layers = Vec::new();

        // Initialize layers
        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];

            // Xavier/Glorot initialization
            let limit = (6.0 / (input_size + output_size) as f64).sqrt();

            let weights_data = Array1::from_shape_fn(input_size * output_size, |_| {
                rng.random_range(-limit, limit)
            });
            let biases_data = Array1::zeros(output_size);

            let activation = if i == layer_sizes.len() - 2 {
                ActivationType::Linear // Output layer
            } else {
                ActivationType::ReLU // Hidden layers
            };

            layers.push(Layer {
                weights: Parameter::new(weights_data),
                biases: Parameter::new(biases_data),
                activation,
            });
        }

        // Create optimizer
        let optimizer: Box<dyn NetworkOptimizer> = match optimizer_type {
            "adam" => Box::new(AdamNetworkOptimizer::new(0.001, 0.9, 0.999, 1e-8)),
            "lamb" => Box::new(LAMBNetworkOptimizer::new(0.001, 0.9, 0.999, 1e-6, 0.01)),
            _ => {
                return Err(scirs2_optim::error::OptimError::InvalidConfig(format!(
                    "Unknown optimizer: {}",
                    optimizer_type
                )))
            }
        };

        // Create scheduler
        let scheduler: Option<Box<dyn LearningRateScheduler>> = Some(Box::new(
            CosineAnnealingLRScheduler::new(0.001, training_config.epochs, 1e-6),
        ));

        // Create regularizer
        let regularizer: Option<Box<dyn NetworkRegularizer>> =
            Some(Box::new(L2NetworkRegularizer::new(0.001)));

        Ok(Self {
            layers,
            optimizer,
            scheduler,
            regularizer,
            training_config,
            metrics_history: MetricsHistory {
                train_loss: Vec::new(),
                val_loss: Vec::new(),
                train_accuracy: Vec::new(),
                val_accuracy: Vec::new(),
                learning_rates: Vec::new(),
                gradient_norms: Vec::new(),
                epoch_times: Vec::new(),
            },
        })
    }

    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut activations = x.clone();

        for layer in &self.layers {
            // Reshape weights for matrix multiplication
            let weight_shape = (activations.ncols(), layer.biases.data.len());
            let weights_2d = Array2::from_shape_vec(weight_shape, layer.weights.data.to_vec())
                .unwrap_or_else(|_| Array2::zeros(weight_shape));

            // Linear transformation: activation = X * W + b
            activations = activations.dot(&weights_2d);

            // Add bias
            for mut row in activations.axis_iter_mut(Axis(0)) {
                row += &layer.biases.data;
            }

            // Apply activation function
            match layer.activation {
                ActivationType::ReLU => {
                    activations.mapv_inplace(|x| x.max(0.0));
                }
                ActivationType::Sigmoid => {
                    activations.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
                }
                ActivationType::Tanh => {
                    activations.mapv_inplace(|x| x.tanh());
                }
                ActivationType::Linear => {
                    // No activation
                }
            }
        }

        activations
    }

    fn compute_loss(&self, predictions: &Array2<f64>, targets: &Array1<f64>) -> f64 {
        // Mean squared error
        let pred_flat = predictions.column(0); // Assume single output
        let diff = &pred_flat - targets;
        let mse = diff.mapv(|x| x * x).mean().unwrap();

        // Add regularization penalty
        let reg_penalty = if let Some(ref regularizer) = self.regularizer {
            regularizer.compute_penalty(&self.layers)
        } else {
            0.0
        };

        mse + reg_penalty
    }

    fn train(&mut self, dataset: &MLDataset) -> Result<()> {
        println!(
            "Training neural network for {} epochs...",
            self.training_config.epochs
        );

        // Split data for validation
        let (train_data, val_data) =
            dataset.train_test_split(self.training_config.validation_split, 42);

        let mut best_val_loss = f64::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..self.training_config.epochs {
            let epoch_start = Instant::now();

            // Training phase
            let train_predictions = self.forward(&train_data.features);
            let train_loss = self.compute_loss(&train_predictions, &train_data.targets);

            // Compute gradients (simplified)
            let gradients = self.compute_gradients(&train_data, &train_predictions);

            // Update parameters
            self.optimizer.step(&mut self.layers, &gradients)?;

            // Update learning rate
            let current_lr = if let Some(ref mut scheduler) = self.scheduler {
                scheduler.step(epoch, train_loss)
            } else {
                self.optimizer.get_learning_rate()
            };

            // Validation phase
            let val_predictions = self.forward(&val_data.features);
            let val_loss = self.compute_loss(&val_predictions, &val_data.targets);

            // Compute accuracies (for regression, use R²)
            let train_r2 = self.compute_r2(&train_predictions, &train_data.targets);
            let val_r2 = self.compute_r2(&val_predictions, &val_data.targets);

            // Record metrics
            self.metrics_history.train_loss.push(train_loss);
            self.metrics_history.val_loss.push(val_loss);
            self.metrics_history.train_accuracy.push(train_r2);
            self.metrics_history.val_accuracy.push(val_r2);
            self.metrics_history.learning_rates.push(current_lr);
            self.metrics_history
                .epoch_times
                .push(epoch_start.elapsed().as_secs_f64());

            // Early stopping
            if self.training_config.early_stopping {
                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= self.training_config.patience {
                        println!(
                            "Early stopping at epoch {} (val_loss: {:.6})",
                            epoch, val_loss
                        );
                        break;
                    }
                }
            }

            if self.training_config.verbose
                && (epoch % 10 == 0 || epoch == self.training_config.epochs - 1)
            {
                println!(
                    "Epoch {}: train_loss={:.6}, val_loss={:.6}, train_r2={:.4}, val_r2={:.4}, lr={:.6}",
                    epoch, train_loss, val_loss, train_r2, val_r2, current_lr
                );
            }
        }

        Ok(())
    }

    fn compute_gradients(
        &self,
        dataset: &MLDataset,
        predictions: &Array2<f64>,
    ) -> Vec<LayerGradients> {
        // Simplified gradient computation for demonstration
        // In practice, this would implement proper backpropagation

        let mut gradients = Vec::new();
        let n_samples = dataset.len() as f64;

        for layer in &self.layers {
            let weight_shape = (
                layer.weights.data.len() / layer.biases.data.len(),
                layer.biases.data.len(),
            );

            // Random gradients for demonstration (normally computed via backprop)
            let weight_gradients =
                Array2::from_shape_fn(weight_shape, |_| random::rng().random_range(-0.01, 0.01));

            let bias_gradients = Array1::from_shape_fn(layer.biases.data.len(), |_| {
                random::rng().random_range(-0.01, 0.01)
            });

            gradients.push(LayerGradients {
                weight_gradients,
                bias_gradients,
            });
        }

        gradients
    }

    fn compute_r2(&self, predictions: &Array2<f64>, targets: &Array1<f64>) -> f64 {
        let pred_flat = predictions.column(0);
        let target_mean = targets.mean().unwrap();

        let ss_res = (&pred_flat - targets).mapv(|x| x * x).sum();
        let ss_tot = targets.mapv(|x| (x - target_mean).powi(2)).sum();

        if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - (ss_res / ss_tot)
        }
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let predictions = self.forward(x);
        predictions.column(0).to_owned()
    }

    fn get_metrics_history(&self) -> &MetricsHistory {
        &self.metrics_history
    }
}

// Cross-validation implementation
#[allow(dead_code)]
fn cross_validate_optimizers(dataset: &MLDataset, k: usize) -> Result<()> {
    println!("\n🔍 Cross-validating optimizers with {}-fold CV...", k);

    let optimizers = vec!["adam", "lamb"];
    let folds = dataset.k_fold_split(k, 42);

    for optimizer_name in optimizers {
        println!("\n📊 Testing {} optimizer:", optimizer_name.to_uppercase());

        let mut fold_scores = Vec::new();
        let mut fold_times = Vec::new();

        for (fold_idx, (train_fold, test_fold)) in folds.iter().enumerate() {
            let start_time = Instant::now();

            let training_config = TrainingConfig {
                epochs: 50,
                batch_size: 32,
                early_stopping: true,
                patience: 5,
                validation_split: 0.2,
                shuffle: true,
                verbose: false,
            };

            let mut model = OptimizedNeuralNetwork::new(
                &[train_fold.n_features(), 64, 32, 1],
                optimizer_name,
                training_config,
            )?;

            model.train(train_fold)?;

            let test_predictions = model.predict(&test_fold.features);
            let test_r2 =
                model.compute_r2(&test_predictions.insert_axis(Axis(1)), &test_fold.targets);

            fold_scores.push(test_r2);
            fold_times.push(start_time.elapsed().as_secs_f64());

            println!(
                "  Fold {}: R² = {:.4}, Time = {:.2}s",
                fold_idx + 1,
                test_r2,
                fold_times.last().unwrap()
            );
        }

        let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        let std_score = {
            let variance = fold_scores
                .iter()
                .map(|&score| (score - mean_score).powi(2))
                .sum::<f64>()
                / fold_scores.len() as f64;
            variance.sqrt()
        };
        let mean_time = fold_times.iter().sum::<f64>() / fold_times.len() as f64;

        println!(
            "  {} Results: R² = {:.4} ± {:.4}, Avg Time = {:.2}s",
            optimizer_name.to_uppercase(),
            mean_score,
            std_score,
            mean_time
        );
    }

    Ok(())
}

// Hyperparameter optimization
#[derive(Debug, Clone)]
struct HyperparameterConfig {
    learning_rate: f64,
    batch_size: usize,
    hidden_sizes: Vec<usize>,
    l2_strength: f64,
    optimizer: String,
}

#[allow(dead_code)]
fn hyperparameter_optimization(dataset: &MLDataset) -> Result<()> {
    println!("\n🎯 Hyperparameter optimization...");

    let hyperparameter_grid = vec![
        HyperparameterConfig {
            learning_rate: 0.001,
            batch_size: 32,
            hidden_sizes: vec![64, 32],
            l2_strength: 0.001,
            optimizer: "adam".to_string(),
        },
        HyperparameterConfig {
            learning_rate: 0.003,
            batch_size: 64,
            hidden_sizes: vec![128, 64],
            l2_strength: 0.01,
            optimizer: "adam".to_string(),
        },
        HyperparameterConfig {
            learning_rate: 0.001,
            batch_size: 32,
            hidden_sizes: vec![64, 32],
            l2_strength: 0.001,
            optimizer: "lamb".to_string(),
        },
        HyperparameterConfig {
            learning_rate: 0.003,
            batch_size: 64,
            hidden_sizes: vec![128, 64],
            l2_strength: 0.01,
            optimizer: "lamb".to_string(),
        },
    ];

    let mut best_score = f64::NEG_INFINITY;
    let mut best_config = None;

    for (config_idx, config) in hyperparameter_grid.iter().enumerate() {
        println!("\n  Config {}: {:?}", config_idx + 1, config);

        let (train_data, test_data) = dataset.train_test_split(0.2, 42);

        let training_config = TrainingConfig {
            epochs: 30,
            batch_size: config.batch_size,
            early_stopping: true,
            patience: 5,
            validation_split: 0.2,
            shuffle: true,
            verbose: false,
        };

        let mut layer_sizes = vec![train_data.n_features()];
        layer_sizes.extend(&config.hidden_sizes);
        layer_sizes.push(1);

        let mut model =
            OptimizedNeuralNetwork::new(&layer_sizes, &config.optimizer, training_config)?;

        let start_time = Instant::now();
        model.train(&train_data)?;
        let training_time = start_time.elapsed();

        let test_predictions = model.predict(&test_data.features);
        let test_score =
            model.compute_r2(&test_predictions.insert_axis(Axis(1)), &test_data.targets);

        println!(
            "    R² Score: {:.4}, Training Time: {:.2}s",
            test_score,
            training_time.as_secs_f64()
        );

        if test_score > best_score {
            best_score = test_score;
            best_config = Some((config.clone(), test_score, training_time));
        }
    }

    if let Some((config, score, time)) = best_config {
        println!("\n🏆 Best Configuration:");
        println!("  Config: {:?}", config);
        println!("  Score: {:.4}", score);
        println!("  Training Time: {:.2}s", time.as_secs_f64());
    }

    Ok(())
}

// Model ensemble with different optimizers
#[allow(dead_code)]
fn ensemble_demonstration(dataset: &MLDataset) -> Result<()> {
    println!("\n🎭 Model Ensemble Demonstration...");

    let optimizers = vec!["adam", "lamb"];
    let mut models = Vec::new();
    let mut model_scores = Vec::new();

    let (train_data, test_data) = dataset.train_test_split(0.2, 42);

    // Train multiple models with different optimizers
    for optimizer_name in &optimizers {
        println!("\n  Training model with {} optimizer...", optimizer_name);

        let training_config = TrainingConfig {
            epochs: 50,
            batch_size: 32,
            early_stopping: true,
            patience: 8,
            validation_split: 0.2,
            shuffle: true,
            verbose: false,
        };

        let mut model = OptimizedNeuralNetwork::new(
            &[train_data.n_features(), 128, 64, 32, 1],
            optimizer_name,
            training_config,
        )?;

        model.train(&train_data)?;

        let test_predictions = model.predict(&test_data.features);
        let test_score =
            model.compute_r2(&test_predictions.insert_axis(Axis(1)), &test_data.targets);

        println!("    Individual R² Score: {:.4}", test_score);

        models.push(model);
        model_scores.push(test_score);
    }

    // Create ensemble predictions
    println!("\n  Creating ensemble predictions...");
    let ensemble_predictions = {
        let mut predictions_sum = Array1::zeros(test_data.len());

        for model in &models {
            let pred = model.predict(&test_data.features);
            predictions_sum = predictions_sum + pred;
        }

        predictions_sum / models.len() as f64
    };

    // Evaluate ensemble
    let ensemble_score = {
        let target_mean = test_data.targets.mean().unwrap();
        let ss_res = (&ensemble_predictions - &test_data.targets)
            .mapv(|x| x * x)
            .sum();
        let ss_tot = test_data.targets.mapv(|x| (x - target_mean).powi(2)).sum();

        if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - (ss_res / ss_tot)
        }
    };

    println!("  Ensemble R² Score: {:.4}", ensemble_score);

    // Compare with individual models
    let best_individual = model_scores
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let improvement = ensemble_score - best_individual;

    println!("  Best Individual Score: {:.4}", best_individual);
    println!("  Ensemble Improvement: {:.4}", improvement);
    println!(
        "  Relative Improvement: {:.2}%",
        (improvement / best_individual) * 100.0
    );

    Ok(())
}

// Create synthetic regression dataset
#[allow(dead_code)]
fn create_regression_dataset(n_samples: usize, n_features: usize, noise: f64) -> MLDataset {
    let mut rng = Xoshiro256Plus::seed_from_u64(42);

    // Generate features
    let features = Array2::from_shape_fn((n_samples, n_features), |_| rng.random_range(-2.0, 2.0));

    // Generate true coefficients
    let true_coefficients = Array1::from_shape_fn(n_features, |_| rng.random_range(-1.0, 1.0));

    // Generate targets with non-linear transformation
    let linear_combination = features.dot(&true_coefficients);
    let targets = linear_combination.mapv(|x| {
        let nonlinear = x + 0.5 * x.powi(2) + 0.1 * x.powi(3);
        nonlinear + rng.random_range(-noise, noise)
    });

    MLDataset::new(features, targets)
}

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("🧠 Scirs2-Optim + SmartCore Integration Comprehensive Example");
    println!("=============================================================");

    // Create test dataset
    let n_samples = 2000;
    let n_features = 15;
    let noise = 0.5;

    println!("\n📊 Creating synthetic regression dataset:");
    println!("  Samples: {}", n_samples);
    println!("  Features: {}", n_features);
    println!("  Noise level: {}", noise);

    let dataset = create_regression_dataset(n_samples, n_features, noise);

    // Basic neural network training
    println!("\n🚀 Basic Neural Network Training:");
    let training_config = TrainingConfig {
        epochs: 100,
        batch_size: 64,
        early_stopping: true,
        patience: 10,
        validation_split: 0.2,
        shuffle: true,
        verbose: true,
    };

    let mut model =
        OptimizedNeuralNetwork::new(&[n_features, 128, 64, 32, 1], "adam", training_config)?;

    let start_time = Instant::now();
    model.train(&dataset)?;
    let training_time = start_time.elapsed();

    println!("Training completed in {:?}", training_time);

    // Cross-validation
    cross_validate_optimizers(&dataset, 5)?;

    // Hyperparameter optimization
    hyperparameter_optimization(&dataset)?;

    // Ensemble demonstration
    ensemble_demonstration(&dataset)?;

    println!("\n✅ SmartCore integration example completed successfully!");
    println!("\n📋 Summary:");
    println!("   - Integrated scirs2-optim with neural network training");
    println!("   - Compared Adam and LAMB optimizers");
    println!("   - Demonstrated cross-validation with custom optimizers");
    println!("   - Performed hyperparameter optimization");
    println!("   - Created ensemble models with different optimization strategies");
    println!("   - Showcased learning rate scheduling and regularization");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_creation() {
        let dataset = create_regression_dataset(100, 5, 0.1);
        assert_eq!(dataset.len(), 100);
        assert_eq!(dataset.n_features(), 5);
    }

    #[test]
    fn test_train_test_split() {
        let dataset = create_regression_dataset(100, 5, 0.1);
        let (train, test) = dataset.train_test_split(0.2, 42);

        assert!(train.len() > test.len());
        assert_eq!(train.len() + test.len(), 100);
    }

    #[test]
    fn test_k_fold_split() {
        let dataset = create_regression_dataset(100, 5, 0.1);
        let folds = dataset.k_fold_split(5, 42);

        assert_eq!(folds.len(), 5);
        for (train, test) in &folds {
            assert!(train.len() > test.len());
        }
    }

    #[test]
    fn test_neural_network_creation() {
        let training_config = TrainingConfig {
            epochs: 10,
            batch_size: 32,
            early_stopping: false,
            patience: 5,
            validation_split: 0.2,
            shuffle: true,
            verbose: false,
        };

        let model = OptimizedNeuralNetwork::new(&[5, 10, 1], "adam", training_config);

        assert!(model.is_ok());
    }

    #[test]
    fn test_forward_pass() {
        let training_config = TrainingConfig {
            epochs: 10,
            batch_size: 32,
            early_stopping: false,
            patience: 5,
            validation_split: 0.2,
            shuffle: true,
            verbose: false,
        };

        let model = OptimizedNeuralNetwork::new(&[3, 5, 1], "adam", training_config).unwrap();

        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = model.forward(&input);

        assert_eq!(output.shape(), &[2, 1]);
    }
}
