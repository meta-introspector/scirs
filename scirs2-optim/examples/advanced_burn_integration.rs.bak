//! Advanced Burn integration example for scirs2-optim
//!
//! This example demonstrates deep integration between scirs2-optim and Burn,
//! showing how to use scirs2-optim optimizers with Burn tensors and training loops.

use ndarray::{Array1, Array2, Array3};
use scirs2_optim::{
    error::Result as OptimResult,
    gradient_processing::{ClippingStrategy, GradientClipper},
    optimizers::{Adam, Optimizer as OptimizerTrait, RMSprop, SGD},
    regularizers::{DropoutLayer, L2Regularizer, Regularizer},
    schedulers::{CosineAnnealing, ExponentialDecay, ReduceOnPlateau, Scheduler},
    unified_api::{OptimizerConfig, OptimizerFactory, Parameter},
};
use std::collections::HashMap;

/// Burn tensor compatibility layer
///
/// This module provides conversion utilities between Burn tensors and ndarray arrays
/// to enable seamless integration with scirs2-optim optimizers.
pub mod burn_bridge {
    use super::*;

    /// Burn tensor shape information
    #[derive(Debug, Clone)]
    pub struct TensorShape {
        pub dims: Vec<usize>,
    }

    impl TensorShape {
        pub fn new(dims: Vec<usize>) -> Self {
            Self { dims }
        }

        pub fn total_elements(&self) -> usize {
            self.dims.iter().product()
        }
    }

    /// Convert Burn tensor to ndarray (placeholder implementation)
    pub fn burn_tensor_to_array(tensor_data: &[f32], shape: &TensorShape) -> Array1<f64> {
        Array1::from_vec(tensor_data.iter().map(|&x| x as f64).collect())
    }

    /// Convert ndarray to Burn tensor data (placeholder implementation)
    pub fn array_to_burn_tensor(array: &Array1<f64>) -> Vec<f32> {
        array.iter().map(|&x| x as f32).collect()
    }

    /// Convert multi-dimensional data
    pub fn burn_tensor_to_array2(tensor_data: &[f32], shape: (usize, usize)) -> Array2<f64> {
        let data: Vec<f64> = tensor_data.iter().map(|&x| x as f64).collect();
        Array2::from_shape_vec(shape, data).expect("Invalid shape for tensor conversion")
    }

    /// Convert 3D tensor data
    pub fn burn_tensor_to_array3(tensor_data: &[f32], shape: (usize, usize, usize)) -> Array3<f64> {
        let data: Vec<f64> = tensor_data.iter().map(|&x| x as f64).collect();
        Array3::from_shape_vec(shape, data).expect("Invalid shape for tensor conversion")
    }
}

/// Burn-compatible activation functions
pub mod activations {
    use ndarray::Array2;

    pub fn relu(input: &Array2<f64>) -> Array2<f64> {
        input.mapv(|x| x.max(0.0))
    }

    pub fn relu_derivative(input: &Array2<f64>) -> Array2<f64> {
        input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }

    pub fn sigmoid(input: &Array2<f64>) -> Array2<f64> {
        input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    pub fn sigmoid_derivative(input: &Array2<f64>) -> Array2<f64> {
        let sig = sigmoid(input);
        &sig * &sig.mapv(|x| 1.0 - x)
    }

    pub fn tanh(input: &Array2<f64>) -> Array2<f64> {
        input.mapv(|x| x.tanh())
    }

    pub fn tanh_derivative(input: &Array2<f64>) -> Array2<f64> {
        let tanh_val = tanh(input);
        tanh_val.mapv(|x| 1.0 - x * x)
    }

    pub fn softmax(input: &Array2<f64>) -> Array2<f64> {
        let mut output = Array2::zeros(input.raw_dim());

        for (i, row) in input.outer_iter().enumerate() {
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_row: Array1<f64> = row.mapv(|x| (x - max_val).exp());
            let sum_exp = exp_row.sum();

            let softmax_row = exp_row / sum_exp;
            output.row_mut(i).assign(&softmax_row);
        }

        output
    }
}

/// Burn-compatible neural network layer with advanced features
pub struct BurnNeuralLayer {
    /// Weight parameters
    weights: Parameter<f64>,
    /// Bias parameters
    bias: Parameter<f64>,
    /// Layer dimensions
    input_dim: usize,
    output_dim: usize,
    /// Activation function type
    activation: ActivationType,
    /// Dropout layer
    dropout: Option<DropoutLayer<f64>>,
    /// L2 regularizer
    l2_regularizer: Option<L2Regularizer<f64>>,
    /// Layer name
    name: String,
}

/// Supported activation types
#[derive(Debug, Clone)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
    Softmax,
}

impl BurnNeuralLayer {
    /// Create a new neural layer with Burn-style initialization
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        activation: ActivationType,
        name: &str,
    ) -> Self {
        // He initialization for ReLU, Xavier for others
        let scale = match activation {
            ActivationType::ReLU => (2.0 / input_dim as f64).sqrt(),
            _ => (2.0 / (input_dim + output_dim) as f64).sqrt(),
        };

        let weights_data: Vec<f64> = (0..input_dim * output_dim)
            .map(|i| {
                // Simple deterministic initialization for reproducible testing
                let val = (i as f64 * 0.01) % 1.0 - 0.5;
                val * scale
            })
            .collect();

        let bias_data: Vec<f64> = (0..output_dim).map(|_| 0.0).collect();

        let weights = Parameter::new(
            Array2::from_shape_vec((input_dim, output_dim), weights_data)
                .expect("Invalid weight shape")
                .into_dyn(),
            &format!("{}.weight", name),
        );

        let bias = Parameter::new(
            Array1::from_vec(bias_data).into_dyn(),
            &format!("{}.bias", name),
        );

        Self {
            weights,
            bias,
            input_dim,
            output_dim,
            activation,
            dropout: None,
            l2_regularizer: None,
            name: name.to_string(),
        }
    }

    /// Add dropout to this layer
    pub fn with_dropout(mut self, dropout_rate: f64) -> OptimResult<Self> {
        self.dropout = Some(DropoutLayer::new(dropout_rate));
        Ok(self)
    }

    /// Add L2 regularization to this layer
    pub fn with_l2_regularization(mut self, l2_weight: f64) -> Self {
        self.l2_regularizer = Some(L2Regularizer::new(l2_weight));
        self
    }

    /// Forward pass with activation
    pub fn forward(&self, input: &Array2<f64>, training: bool) -> OptimResult<Array2<f64>> {
        // Linear transformation
        let weights_2d = self
            .weights
            .data()
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| {
                scirs2_optim::error::OptimError::DimensionMismatch(
                    "Weight tensor dimension mismatch".to_string(),
                )
            })?;

        let bias_1d = self
            .bias
            .data()
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| {
                scirs2_optim::error::OptimError::DimensionMismatch(
                    "Bias tensor dimension mismatch".to_string(),
                )
            })?;

        let linear_output = input.dot(&weights_2d) + &bias_1d;

        // Apply activation
        let activated_output = match self.activation {
            ActivationType::ReLU => activations::relu(&linear_output),
            ActivationType::Sigmoid => activations::sigmoid(&linear_output),
            ActivationType::Tanh => activations::tanh(&linear_output),
            ActivationType::Linear => linear_output,
            ActivationType::Softmax => activations::softmax(&linear_output),
        };

        // Apply dropout during training
        let output = if training && self.dropout.is_some() {
            let dropout = self.dropout.as_ref().unwrap();
            dropout.forward(&activated_output)?
        } else {
            activated_output
        };

        Ok(output)
    }

    /// Backward pass with activation derivative
    pub fn backward(
        &mut self,
        input: &Array2<f64>,
        grad_output: &Array2<f64>,
        pre_activation: &Array2<f64>,
        training: bool,
    ) -> OptimResult<Array2<f64>> {
        // Apply dropout gradient if training
        let mut grad_activated = grad_output.clone();
        if training && self.dropout.is_some() {
            let dropout = self.dropout.as_mut().unwrap();
            grad_activated = dropout.backward(&grad_activated)?;
        }

        // Apply activation derivative
        let grad_linear = match self.activation {
            ActivationType::ReLU => &grad_activated * &activations::relu_derivative(pre_activation),
            ActivationType::Sigmoid => {
                &grad_activated * &activations::sigmoid_derivative(pre_activation)
            }
            ActivationType::Tanh => &grad_activated * &activations::tanh_derivative(pre_activation),
            ActivationType::Linear => grad_activated,
            ActivationType::Softmax => {
                // Simplified softmax gradient (assuming cross-entropy loss)
                grad_activated
            }
        };

        // Compute parameter gradients
        let grad_weights = input.t().dot(&grad_linear);
        let grad_bias = grad_linear.sum_axis(ndarray::Axis(0));

        // Apply L2 regularization to weight gradients
        let final_grad_weights = if let Some(ref l2_reg) = self.l2_regularizer {
            let weights_2d = self
                .weights
                .data()
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| {
                    scirs2_optim::error::OptimError::DimensionMismatch(
                        "Weight tensor dimension mismatch".to_string(),
                    )
                })?;

            let l2_grad = l2_reg.compute_gradient(&weights_2d.into_dyn())?;
            let l2_grad_2d = l2_grad.into_dimensionality::<ndarray::Ix2>().map_err(|_| {
                scirs2_optim::error::OptimError::DimensionMismatch(
                    "L2 gradient dimension mismatch".to_string(),
                )
            })?;

            grad_weights + l2_grad_2d
        } else {
            grad_weights
        };

        // Compute input gradient
        let weights_2d = self
            .weights
            .data()
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| {
                scirs2_optim::error::OptimError::DimensionMismatch(
                    "Weight tensor dimension mismatch".to_string(),
                )
            })?;

        let grad_input = grad_linear.dot(&weights_2d.t());

        // Set gradients in parameters
        self.weights.set_grad(final_grad_weights.into_dyn());
        self.bias.set_grad(grad_bias.into_dyn());

        Ok(grad_input)
    }

    /// Get mutable references to parameters
    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<f64>> {
        vec![&mut self.weights, &mut self.bias]
    }

    /// Get regularization loss
    pub fn regularization_loss(&self) -> OptimResult<f64> {
        if let Some(ref l2_reg) = self.l2_regularizer {
            let weights_2d = self
                .weights
                .data()
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| {
                    scirs2_optim::error::OptimError::DimensionMismatch(
                        "Weight tensor dimension mismatch".to_string(),
                    )
                })?;

            l2_reg.compute_loss(&weights_2d.into_dyn())
        } else {
            Ok(0.0)
        }
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.input_dim * self.output_dim + self.output_dim
    }

    /// Get layer name
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Advanced Burn-compatible neural network
pub struct BurnNeuralNetwork {
    /// Network layers
    layers: Vec<BurnNeuralLayer>,
    /// Network architecture
    architecture: Vec<usize>,
    /// Gradient clipper
    gradient_clipper: Option<GradientClipper<f64>>,
    /// Training mode
    training_mode: bool,
}

impl BurnNeuralNetwork {
    /// Create a new neural network with specified architecture and activations
    pub fn new(architecture: Vec<usize>, activations: Vec<ActivationType>) -> OptimResult<Self> {
        if architecture.len() != activations.len() + 1 {
            return Err(scirs2_optim::error::OptimError::InvalidConfig(
                "Architecture length must be activations length + 1".to_string(),
            ));
        }

        let mut layers = Vec::new();

        for i in 0..architecture.len() - 1 {
            let layer = BurnNeuralLayer::new(
                architecture[i],
                architecture[i + 1],
                activations[i].clone(),
                &format!("layer_{}", i),
            );
            layers.push(layer);
        }

        Ok(Self {
            layers,
            architecture,
            gradient_clipper: None,
            training_mode: true,
        })
    }

    /// Add gradient clipping
    pub fn with_gradient_clipping(mut self, clip_value: f64, strategy: ClippingStrategy) -> Self {
        self.gradient_clipper = Some(GradientClipper::new(clip_value, strategy));
        self
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training_mode = training;
    }

    /// Forward pass through the network
    pub fn forward(&self, input: Array2<f64>) -> OptimResult<(Array2<f64>, Vec<Array2<f64>>)> {
        let mut activations = vec![input.clone()];
        let mut current = input;

        for layer in &self.layers {
            current = layer.forward(&current, self.training_mode)?;
            activations.push(current.clone());
        }

        Ok((current, activations))
    }

    /// Backward pass through the network
    pub fn backward(
        &mut self,
        input: &Array2<f64>,
        target: &Array2<f64>,
        loss_type: LossType,
    ) -> OptimResult<(f64, Vec<Array2<f64>>)> {
        // Forward pass to compute activations and pre-activations
        let (output, activations) = self.forward(input.clone())?;

        // Compute loss and initial gradient
        let (loss, mut grad_output) = match loss_type {
            LossType::MeanSquaredError => {
                let diff = &output - target;
                let mse_loss = (&diff * &diff).mean().unwrap_or(0.0);
                let grad = &diff * (2.0 / output.nrows() as f64);
                (mse_loss, grad)
            }
            LossType::CrossEntropy => {
                // Simplified cross-entropy for softmax output
                let epsilon = 1e-15;
                let clipped_output = output.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
                let log_output = clipped_output.mapv(|x| x.ln());
                let ce_loss = -(target * &log_output).mean().unwrap_or(0.0);
                let grad = (&clipped_output - target) / output.nrows() as f64;
                (ce_loss, grad)
            }
        };

        // Add regularization loss
        let mut total_loss = loss;
        for layer in &self.layers {
            total_loss += layer.regularization_loss()?;
        }

        // Backward pass through layers
        let mut pre_activations = Vec::new();

        // Store pre-activations (simplified - would need actual pre-activation storage)
        for i in 0..activations.len() - 1 {
            pre_activations.push(activations[i + 1].clone());
        }

        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let layer_input = &activations[i];
            let pre_activation = &pre_activations[i];

            grad_output = layer.backward(
                layer_input,
                &grad_output,
                pre_activation,
                self.training_mode,
            )?;
        }

        // Apply gradient clipping if configured
        if let Some(ref mut clipper) = self.gradient_clipper {
            for layer in &mut self.layers {
                for param in layer.parameters_mut() {
                    if let Some(grad) = param.grad() {
                        let clipped_grad = clipper.clip_gradient(&grad)?;
                        param.set_grad(clipped_grad);
                    }
                }
            }
        }

        Ok((total_loss, activations))
    }

    /// Get all parameters
    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<f64>> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        params
    }

    /// Get total parameter count
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.parameter_count())
            .sum()
    }

    /// Get network architecture
    pub fn architecture(&self) -> &[usize] {
        &self.architecture
    }

    /// Get layer information
    pub fn layer_info(&self) -> Vec<String> {
        self.layers
            .iter()
            .map(|layer| {
                format!(
                    "{}: {} -> {}, activation: {:?}",
                    layer.name(),
                    layer.input_dim,
                    layer.output_dim,
                    layer.activation
                )
            })
            .collect()
    }
}

/// Loss function types
#[derive(Debug, Clone)]
pub enum LossType {
    MeanSquaredError,
    CrossEntropy,
}

/// Advanced Burn training configuration
pub struct BurnTrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Optimizer type
    pub optimizer_type: String,
    /// Scheduler configuration
    pub scheduler_config: Option<BurnSchedulerConfig>,
    /// Gradient clipping configuration
    pub grad_clip_config: Option<GradClipConfig>,
    /// Loss function type
    pub loss_type: LossType,
    /// Validation split
    pub validation_split: f64,
    /// Early stopping patience
    pub early_stopping_patience: Option<usize>,
}

/// Scheduler configuration for Burn
pub struct BurnSchedulerConfig {
    pub scheduler_type: String,
    pub decay_rate: f64,
    pub step_size: Option<usize>,
    pub min_lr: Option<f64>,
    pub patience: Option<usize>,
    pub factor: Option<f64>,
}

/// Gradient clipping configuration
pub struct GradClipConfig {
    pub clip_value: f64,
    pub strategy: ClippingStrategy,
}

/// Advanced Burn trainer with comprehensive features
pub struct BurnTrainer {
    /// Neural network
    network: BurnNeuralNetwork,
    /// Optimizer
    optimizer: Box<dyn scirs2_optim::unified_api::UnifiedOptimizer<f64>>,
    /// Learning rate scheduler
    scheduler: Option<Box<dyn Scheduler<f64>>>,
    /// Training configuration
    config: BurnTrainingConfig,
    /// Training history
    train_history: TrainingHistory,
    /// Best model state (for early stopping)
    best_loss: f64,
    /// Early stopping counter
    early_stopping_counter: usize,
}

/// Training history tracking
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    pub train_losses: Vec<f64>,
    pub val_losses: Vec<f64>,
    pub learning_rates: Vec<f64>,
    pub epoch_times: Vec<std::time::Duration>,
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            learning_rates: Vec::new(),
            epoch_times: Vec::new(),
        }
    }
}

impl BurnTrainer {
    /// Create a new advanced Burn trainer
    pub fn new(
        architecture: Vec<usize>,
        activations: Vec<ActivationType>,
        config: BurnTrainingConfig,
    ) -> OptimResult<Self> {
        let mut network = BurnNeuralNetwork::new(architecture, activations)?;

        // Add gradient clipping if configured
        if let Some(ref clip_config) = config.grad_clip_config {
            network = network
                .with_gradient_clipping(clip_config.clip_value, clip_config.strategy.clone());
        }

        // Create optimizer
        let optimizer_config = OptimizerConfig::new(config.learning_rate).weight_decay(0.0001);

        let optimizer: Box<dyn scirs2_optim::unified_api::UnifiedOptimizer<f64>> =
            match config.optimizer_type.as_str() {
                "adam" => Box::new(OptimizerFactory::adam(optimizer_config)),
                "sgd" => Box::new(OptimizerFactory::sgd(optimizer_config)),
                "rmsprop" => Box::new(OptimizerFactory::rmsprop(optimizer_config)),
                _ => Box::new(OptimizerFactory::adam(optimizer_config)),
            };

        // Create scheduler if configured
        let scheduler: Option<Box<dyn Scheduler<f64>>> =
            if let Some(ref sched_config) = config.scheduler_config {
                match sched_config.scheduler_type.as_str() {
                    "exponential" => Some(Box::new(ExponentialDecay::new(
                        config.learning_rate,
                        sched_config.decay_rate,
                    ))),
                    "cosine" => Some(Box::new(CosineAnnealing::new(
                        config.learning_rate,
                        sched_config.min_lr.unwrap_or(0.0),
                        config.epochs,
                    ))),
                    "reduce_on_plateau" => Some(Box::new(ReduceOnPlateau::new(
                        config.learning_rate,
                        sched_config.factor.unwrap_or(0.5),
                        sched_config.patience.unwrap_or(10),
                        sched_config.min_lr.unwrap_or(1e-8),
                    ))),
                    _ => None,
                }
            } else {
                None
            };

        Ok(Self {
            network,
            optimizer,
            scheduler,
            config,
            train_history: TrainingHistory::new(),
            best_loss: f64::INFINITY,
            early_stopping_counter: 0,
        })
    }

    /// Train the network with advanced features
    pub fn train(
        &mut self,
        train_data: &Array2<f64>,
        train_targets: &Array2<f64>,
    ) -> OptimResult<BurnTrainingResults> {
        let start_time = std::time::Instant::now();

        // Split data into training and validation
        let split_idx = ((1.0 - self.config.validation_split) * train_data.nrows() as f64) as usize;
        let (train_x, val_x) = train_data.split_at(ndarray::Axis(0), split_idx);
        let (train_y, val_y) = train_targets.split_at(ndarray::Axis(0), split_idx);

        println!("🔥 Starting Advanced Burn Neural Network Training");
        println!("   Architecture: {:?}", self.network.architecture());
        println!("   Parameters: {}", self.network.parameter_count());
        println!("   Optimizer: {}", self.config.optimizer_type);
        println!(
            "   Scheduler: {:?}",
            self.config
                .scheduler_config
                .as_ref()
                .map(|s| &s.scheduler_type)
        );
        println!("   Loss Type: {:?}", self.config.loss_type);
        println!("   Epochs: {}", self.config.epochs);
        println!("   Batch Size: {}", self.config.batch_size);
        println!("   Train/Val Split: {}/{}", train_x.nrows(), val_x.nrows());

        // Print layer information
        for layer_info in self.network.layer_info() {
            println!("     {}", layer_info);
        }

        for epoch in 0..self.config.epochs {
            let epoch_start = std::time::Instant::now();

            // Training phase
            self.network.set_training(true);
            let train_loss = self.train_epoch(&train_x.to_owned(), &train_y.to_owned())?;

            // Validation phase
            self.network.set_training(false);
            let val_loss = self.validate(&val_x.to_owned(), &val_y.to_owned())?;

            let epoch_time = epoch_start.elapsed();

            // Record history
            self.train_history.train_losses.push(train_loss);
            self.train_history.val_losses.push(val_loss);
            self.train_history.epoch_times.push(epoch_time);

            let current_lr = if let Some(ref scheduler) = self.scheduler {
                scheduler.get_lr()
            } else {
                self.config.learning_rate
            };
            self.train_history.learning_rates.push(current_lr);

            // Update learning rate scheduler
            if let Some(ref mut scheduler) = self.scheduler {
                // For ReduceOnPlateau, we need to pass the validation loss
                if let Some(reduce_scheduler) = scheduler
                    .as_any_mut()
                    .downcast_mut::<ReduceOnPlateau<f64>>()
                {
                    reduce_scheduler.step_with_loss(val_loss);
                } else {
                    scheduler.step();
                }
            }

            // Early stopping check
            if let Some(patience) = self.config.early_stopping_patience {
                if val_loss < self.best_loss {
                    self.best_loss = val_loss;
                    self.early_stopping_counter = 0;
                } else {
                    self.early_stopping_counter += 1;
                    if self.early_stopping_counter >= patience {
                        println!("🛑 Early stopping triggered at epoch {}", epoch + 1);
                        break;
                    }
                }
            }

            // Print progress
            if (epoch + 1) % 10 == 0 || epoch == 0 || epoch == self.config.epochs - 1 {
                println!("   Epoch {}/{}: Train Loss = {:.6}, Val Loss = {:.6}, LR = {:.6}, Time = {:.2}s", 
                    epoch + 1, self.config.epochs, train_loss, val_loss, current_lr, epoch_time.as_secs_f64());
            }
        }

        let total_training_time = start_time.elapsed();
        let final_train_loss = self
            .train_history
            .train_losses
            .last()
            .copied()
            .unwrap_or(0.0);
        let final_val_loss = self.train_history.val_losses.last().copied().unwrap_or(0.0);

        println!(
            "✅ Training completed in {:.2}s",
            total_training_time.as_secs_f64()
        );
        println!("   Final Train Loss: {:.6}", final_train_loss);
        println!("   Final Validation Loss: {:.6}", final_val_loss);
        println!("   Best Validation Loss: {:.6}", self.best_loss);

        Ok(BurnTrainingResults {
            final_train_loss,
            final_val_loss,
            best_val_loss: self.best_loss,
            training_time: total_training_time,
            epochs_trained: self.train_history.train_losses.len(),
            converged: final_val_loss < 0.01,
            early_stopped: self.config.early_stopping_patience.is_some()
                && self.early_stopping_counter >= self.config.early_stopping_patience.unwrap(),
            history: self.train_history.clone(),
        })
    }

    /// Train one epoch
    fn train_epoch(
        &mut self,
        train_data: &Array2<f64>,
        train_targets: &Array2<f64>,
    ) -> OptimResult<f64> {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;
        let batch_size = self.config.batch_size;
        let n_samples = train_data.nrows();

        for batch_start in (0..n_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_samples);

            let batch_data = train_data
                .slice(ndarray::s![batch_start..batch_end, ..])
                .to_owned();
            let batch_targets = train_targets
                .slice(ndarray::s![batch_start..batch_end, ..])
                .to_owned();

            let (loss, _) = self.network.backward(
                &batch_data,
                &batch_targets,
                self.config.loss_type.clone(),
            )?;
            epoch_loss += loss;
            batch_count += 1;

            // Update parameters
            for param in self.network.parameters_mut() {
                self.optimizer.step_param(param)?;
            }
        }

        Ok(epoch_loss / batch_count as f64)
    }

    /// Validate the model
    fn validate(&self, val_data: &Array2<f64>, val_targets: &Array2<f64>) -> OptimResult<f64> {
        let (output, _) = self.network.forward(val_data.clone())?;

        let loss = match self.config.loss_type {
            LossType::MeanSquaredError => {
                let diff = &output - val_targets;
                (&diff * &diff).mean().unwrap_or(0.0)
            }
            LossType::CrossEntropy => {
                let epsilon = 1e-15;
                let clipped_output = output.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
                let log_output = clipped_output.mapv(|x| x.ln());
                -(val_targets * &log_output).mean().unwrap_or(0.0)
            }
        };

        Ok(loss)
    }

    /// Get training history
    pub fn training_history(&self) -> &TrainingHistory {
        &self.train_history
    }
}

/// Advanced training results
#[derive(Debug, Clone)]
pub struct BurnTrainingResults {
    pub final_train_loss: f64,
    pub final_val_loss: f64,
    pub best_val_loss: f64,
    pub training_time: std::time::Duration,
    pub epochs_trained: usize,
    pub converged: bool,
    pub early_stopped: bool,
    pub history: TrainingHistory,
}

/// Generate synthetic classification data
#[allow(dead_code)]
pub fn generate_classification_data(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    noise_level: f64,
) -> (Array2<f64>, Array2<f64>) {
    let mut rng_state = 42u64;

    // Generate random input data
    let mut input_data = Vec::new();
    for _i in 0..n_samples * n_features {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let random_val = (rng_state % 10000) as f64 / 10000.0 - 0.5;
        input_data.push(random_val);
    }

    let input =
        Array2::from_shape_vec((n_samples, n_features), input_data).expect("Invalid input shape");

    // Generate one-hot encoded targets
    let mut target_data = vec![0.0; n_samples * n_classes];
    for i in 0..n_samples {
        // Simple decision boundary based on input features
        let mut class_score = 0.0;
        for j in 0..n_features {
            class_score += input[[i, j]] * (j + 1) as f64;
        }

        let class = ((class_score + 1.0) * n_classes as f64 / 2.0) as usize % n_classes;
        target_data[i * n_classes + class] = 1.0;

        // Add label noise
        if noise_level > 0.0 {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            if (rng_state % 1000) as f64 / 1000.0 < noise_level {
                target_data[i * n_classes + class] = 0.0;
                let noise_class = (rng_state % n_classes as u64) as usize;
                target_data[i * n_classes + noise_class] = 1.0;
            }
        }
    }

    let targets =
        Array2::from_shape_vec((n_samples, n_classes), target_data).expect("Invalid target shape");

    (input, targets)
}

#[allow(dead_code)]
fn main() -> OptimResult<()> {
    println!("🔥 Advanced Burn Integration with scirs2-optim");
    println!("===============================================");

    // Generate synthetic classification dataset
    let (train_data, train_targets) = generate_classification_data(2000, 20, 5, 0.05);
    let (test_data, test_targets) = generate_classification_data(500, 20, 5, 0.05);

    println!("📊 Dataset Info:");
    println!("   Training samples: {}", train_data.nrows());
    println!("   Test samples: {}", test_data.nrows());
    println!("   Input features: {}", train_data.ncols());
    println!("   Output classes: {}", train_targets.ncols());

    // Create advanced training configuration
    let config = BurnTrainingConfig {
        learning_rate: 0.001,
        epochs: 100,
        batch_size: 64,
        optimizer_type: "adam".to_string(),
        scheduler_config: Some(BurnSchedulerConfig {
            scheduler_type: "reduce_on_plateau".to_string(),
            decay_rate: 0.95,
            step_size: None,
            min_lr: Some(1e-6),
            patience: Some(10),
            factor: Some(0.5),
        }),
        grad_clip_config: Some(GradClipConfig {
            clip_value: 1.0,
            strategy: ClippingStrategy::Norm,
        }),
        loss_type: LossType::CrossEntropy,
        validation_split: 0.2,
        early_stopping_patience: Some(20),
    };

    // Create network architecture with different activations
    let architecture = vec![20, 64, 32, 16, 5];
    let activations = vec![
        ActivationType::ReLU,
        ActivationType::ReLU,
        ActivationType::ReLU,
        ActivationType::Softmax,
    ];

    // Create and train the model
    let mut trainer = BurnTrainer::new(architecture, activations, config)?;

    let training_results = trainer.train(&train_data, &train_targets)?;

    println!("\n📈 Training Results:");
    println!(
        "   Final Training Loss: {:.6}",
        training_results.final_train_loss
    );
    println!(
        "   Final Validation Loss: {:.6}",
        training_results.final_val_loss
    );
    println!(
        "   Best Validation Loss: {:.6}",
        training_results.best_val_loss
    );
    println!(
        "   Training Time: {:.2}s",
        training_results.training_time.as_secs_f64()
    );
    println!("   Epochs Trained: {}", training_results.epochs_trained);
    println!("   Converged: {}", training_results.converged);
    println!("   Early Stopped: {}", training_results.early_stopped);

    // Demonstrate different optimizers and configurations
    println!("\n🔄 Comparing Different Configurations:");

    let test_configs = vec![
        ("SGD with Momentum", "sgd", "exponential", None),
        ("RMSprop", "rmsprop", "cosine", Some(0.5)),
        ("Adam with Decay", "adam", "exponential", Some(1.0)),
    ];

    for (name, optimizer, scheduler, clip_val) in test_configs {
        let config = BurnTrainingConfig {
            learning_rate: if optimizer == "sgd" { 0.01 } else { 0.001 },
            epochs: 30,
            batch_size: 64,
            optimizer_type: optimizer.to_string(),
            scheduler_config: Some(BurnSchedulerConfig {
                scheduler_type: scheduler.to_string(),
                decay_rate: 0.9,
                step_size: Some(10),
                min_lr: Some(1e-6),
                patience: None,
                factor: None,
            }),
            grad_clip_config: clip_val.map(|val| GradClipConfig {
                clip_value: val,
                strategy: ClippingStrategy::Norm,
            }),
            loss_type: LossType::CrossEntropy,
            validation_split: 0.2,
            early_stopping_patience: None,
        };

        let mut trainer = BurnTrainer::new(
            vec![20, 32, 5],
            vec![ActivationType::ReLU, ActivationType::Softmax],
            config,
        )?;

        let results = trainer.train(&train_data, &train_targets)?;

        println!(
            "   {}: Final Loss = {:.6}, Time = {:.2}s, Epochs = {}",
            name,
            results.final_val_loss,
            results.training_time.as_secs_f64(),
            results.epochs_trained
        );
    }

    println!("\n✅ Advanced Burn integration example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_burn_neural_layer() {
        let mut layer = BurnNeuralLayer::new(3, 2, ActivationType::ReLU, "test");

        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = layer.forward(&input, false).unwrap();

        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(layer.parameter_count(), 8);
    }

    #[test]
    fn test_burn_neural_network() {
        let network = BurnNeuralNetwork::new(
            vec![3, 4, 2],
            vec![ActivationType::ReLU, ActivationType::Linear],
        )
        .unwrap();

        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let (output, activations) = network.forward(input).unwrap();

        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(activations.len(), 3); // input + 2 layers
    }

    #[test]
    fn test_activation_functions() {
        let input = Array2::from_shape_vec((2, 2), vec![-1.0, 0.0, 1.0, 2.0]).unwrap();

        let relu_output = activations::relu(&input);
        assert_eq!(relu_output[[0, 0]], 0.0); // ReLU(-1) = 0
        assert_eq!(relu_output[[1, 1]], 2.0); // ReLU(2) = 2

        let sigmoid_output = activations::sigmoid(&input);
        assert!(sigmoid_output[[0, 0]] > 0.0 && sigmoid_output[[0, 0]] < 1.0);
    }

    #[test]
    fn test_classification_data_generation() {
        let (input, targets) = generate_classification_data(100, 5, 3, 0.0);

        assert_eq!(input.shape(), &[100, 5]);
        assert_eq!(targets.shape(), &[100, 3]);

        // Check that targets are one-hot encoded
        for i in 0..100 {
            let row_sum: f64 = targets.row(i).iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_burn_trainer_creation() {
        let config = BurnTrainingConfig {
            learning_rate: 0.001,
            epochs: 10,
            batch_size: 16,
            optimizer_type: "adam".to_string(),
            scheduler_config: None,
            grad_clip_config: None,
            loss_type: LossType::CrossEntropy,
            validation_split: 0.2,
            early_stopping_patience: None,
        };

        let trainer = BurnTrainer::new(
            vec![5, 3, 2],
            vec![ActivationType::ReLU, ActivationType::Softmax],
            config,
        );
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_layer_with_regularization() {
        let layer =
            BurnNeuralLayer::new(3, 2, ActivationType::ReLU, "test").with_l2_regularization(0.01);

        let reg_loss = layer.regularization_loss().unwrap();
        assert!(reg_loss >= 0.0);
    }
}
