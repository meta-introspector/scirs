//! Framework-Agnostic Integration Guide for SciRS2 Optimizers
//!
//! This example provides a comprehensive guide and template for integrating SciRS2 optimizers
//! with any ML framework. It includes adapter patterns, best practices, and performance optimizations.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayViewMut1};
use scirs2_optim::{
    optimizers::{Adam, AdamW, LBFGS, SGD},
    regularizers::{ElasticNetRegularizer, L1Regularizer, L2Regularizer},
    schedulers::{CosineAnnealingLR, ExponentialDecay, OneCycleLR},
    unified_api::{OptimizerConfig, OptimizerFactory, Parameter, UnifiedOptimizer},
    LearningRateScheduler, Optimizer, Regularizer,
};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

/// Generic trait for ML framework tensors
pub trait FrameworkTensor<T> {
    /// Get tensor shape
    fn shape(&self) -> Vec<usize>;

    /// Get tensor data as slice
    fn data(&self) -> &[T];

    /// Get mutable tensor data
    fn data_mut(&mut self) -> &mut [T];

    /// Get gradient data if available
    fn grad(&self) -> Option<&[T]>;

    /// Set gradient data
    fn set_grad(&mut self, grad: &[T]);

    /// Zero gradients
    fn zero_grad(&mut self);

    /// Move tensor to device
    fn to_device(&mut self, device: &str);

    /// Get device string
    fn device(&self) -> String;
}

/// Generic trait for ML framework models
pub trait FrameworkModel<T, Tensor: FrameworkTensor<T>> {
    /// Get model parameters
    fn parameters(&self) -> Vec<&Tensor>;

    /// Get mutable model parameters
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;

    /// Forward pass
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Get parameter by name
    fn get_parameter(&self, name: &str) -> Option<&Tensor>;

    /// Set parameter by name
    fn set_parameter(&mut self, name: &str, param: Tensor);

    /// Save model state
    fn state_dict(&self) -> HashMap<String, Vec<T>>;

    /// Load model state
    fn load_state_dict(&mut self, state: HashMap<String, Vec<T>>);
}

/// Universal adapter for SciRS2 optimizers
pub struct UniversalOptimizerAdapter<T, Tensor, Model>
where
    T: num_traits::Float + std::fmt::Debug + Send + Sync + 'static,
    Tensor: FrameworkTensor<T>,
    Model: FrameworkModel<T, Tensor>,
{
    optimizer: Box<dyn UnifiedOptimizer>,
    scheduler: Option<Box<dyn LearningRateScheduler>>,
    regularizers: Vec<Box<dyn Regularizer<T>>>,
    parameter_registry: HashMap<String, Parameter<f64>>,
    parameter_groups: Vec<ParameterGroup>,
    optimization_config: OptimizationConfig,
    metrics: OptimizationMetrics,
    _phantom: PhantomData<(T, Tensor, Model)>,
}

/// Parameter group for fine-grained control
#[derive(Debug, Clone)]
pub struct ParameterGroup {
    pub name: String,
    pub parameter_names: Vec<String>,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub momentum: f64,
    pub enabled: bool,
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub gradient_clipping: Option<f64>,
    pub gradient_accumulation_steps: usize,
    pub mixed_precision: bool,
    pub checkpoint_frequency: usize,
    pub early_stopping: Option<EarlyStoppingConfig>,
    pub warmup_steps: usize,
    pub max_steps: Option<usize>,
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    pub patience: usize,
    pub min_delta: f64,
    pub monitor_metric: String,
    pub mode: String, // "min" or "max"
}

/// Optimization metrics tracking
#[derive(Debug, Default)]
pub struct OptimizationMetrics {
    pub step_count: usize,
    pub loss_history: Vec<f64>,
    pub learning_rate_history: Vec<f64>,
    pub gradient_norms: Vec<f64>,
    pub parameter_norms: Vec<f64>,
    pub convergence_metrics: ConvergenceMetrics,
}

/// Convergence tracking
#[derive(Debug, Default)]
pub struct ConvergenceMetrics {
    pub gradient_norm: f64,
    pub parameter_change_norm: f64,
    pub loss_improvement_rate: f64,
    pub is_converged: bool,
    pub convergence_score: f64,
}

impl<T, Tensor, Model> UniversalOptimizerAdapter<T, Tensor, Model>
where
    T: num_traits::Float + std::fmt::Debug + Send + Sync + 'static,
    Tensor: FrameworkTensor<T>,
    Model: FrameworkModel<T, Tensor>,
{
    /// Create new adapter
    pub fn new(
        optimizer_type: &str,
        config: OptimizerConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let optimizer = match optimizer_type.to_lowercase().as_str() {
            "sgd" => OptimizerFactory::sgd(config),
            "adam" => OptimizerFactory::adam(config),
            "adamw" => OptimizerFactory::adamw(config),
            _ => return Err(format!("Unknown optimizer type: {}", optimizer_type).into()),
        };

        Ok(Self {
            optimizer,
            scheduler: None,
            regularizers: Vec::new(),
            parameter_registry: HashMap::new(),
            parameter_groups: Vec::new(),
            optimization_config: OptimizationConfig::default(),
            metrics: OptimizationMetrics::default(),
            _phantom: PhantomData,
        })
    }

    /// Register model with the optimizer
    pub fn register_model(&mut self, model: &Model) -> Result<(), Box<dyn std::error::Error>> {
        // Extract and register all model parameters
        let state_dict = model.state_dict();

        for (name, data) in state_dict {
            let param_data = data
                .iter()
                .map(|&x| x.to_f64().unwrap_or(0.0))
                .collect::<Vec<f64>>();

            let parameter = Parameter::new(Array1::from_vec(param_data), &name);
            self.parameter_registry.insert(name, parameter);
        }

        println!("📝 Registered {} parameters", self.parameter_registry.len());
        Ok(())
    }

    /// Add parameter group
    pub fn add_parameter_group(&mut self, group: ParameterGroup) {
        self.parameter_groups.push(group);
    }

    /// Add learning rate scheduler
    pub fn add_scheduler(
        &mut self,
        scheduler_type: &str,
        config: SchedulerConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let scheduler: Box<dyn LearningRateScheduler> = match scheduler_type.to_lowercase().as_str()
        {
            "exponential" => Box::new(ExponentialDecay::new(
                config.initial_lr,
                config.gamma.unwrap_or(0.95),
                config.step_size.unwrap_or(1),
            )),
            "cosine" => Box::new(CosineAnnealingLR::new(
                config.initial_lr,
                config.min_lr.unwrap_or(0.0),
                config.total_steps.unwrap_or(1000),
            )),
            "onecycle" => Box::new(OneCycleLR::new(
                config.max_lr.unwrap_or(config.initial_lr),
                config.total_steps.unwrap_or(1000),
                config.pct_start.unwrap_or(0.3),
            )),
            _ => return Err(format!("Unknown scheduler type: {}", scheduler_type).into()),
        };

        self.scheduler = Some(scheduler);
        Ok(())
    }

    /// Add regularizer
    pub fn add_regularizer(&mut self, regularizer_type: &str, strength: f64) {
        let regularizer: Box<dyn Regularizer<T>> = match regularizer_type.to_lowercase().as_str() {
            "l1" => Box::new(L1Regularizer::new(T::from(strength).unwrap())),
            "l2" => Box::new(L2Regularizer::new(T::from(strength).unwrap())),
            "elastic" => Box::new(ElasticNetRegularizer::new(
                T::from(strength).unwrap(),
                T::from(0.5).unwrap(),
            )),
            _ => return,
        };

        self.regularizers.push(regularizer);
    }

    /// Perform optimization step
    pub fn step(&mut self, model: &mut Model, loss: f64) -> Result<(), Box<dyn std::error::Error>> {
        // Extract gradients from model
        self.extract_gradients(model)?;

        // Apply gradient clipping if configured
        if let Some(clip_value) = self.optimization_config.gradient_clipping {
            self.clip_gradients(clip_value);
        }

        // Apply regularization
        self.apply_regularization(model)?;

        // Perform optimization steps for each parameter group
        for group in &self.parameter_groups {
            if !group.enabled {
                continue;
            }

            for param_name in &group.parameter_names {
                if let Some(param) = self.parameter_registry.get_mut(param_name) {
                    self.optimizer.step_param(param)?;
                }
            }
        }

        // Update model parameters
        self.update_model_parameters(model)?;

        // Update learning rate
        if let Some(scheduler) = &mut self.scheduler {
            let new_lr = scheduler.step();
            self.update_learning_rate(new_lr);
        }

        // Update metrics
        self.update_metrics(loss);

        self.metrics.step_count += 1;
        Ok(())
    }

    /// Zero gradients
    pub fn zero_grad(&mut self, model: &mut Model) {
        for param in model.parameters_mut() {
            param.zero_grad();
        }

        // Also zero gradients in our parameter registry
        for param in self.parameter_registry.values_mut() {
            param.zero_grad();
        }
    }

    /// Get optimization statistics
    pub fn get_metrics(&self) -> &OptimizationMetrics {
        &self.metrics
    }

    /// Save optimizer state
    pub fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        let mut state = HashMap::new();

        for (name, param) in &self.parameter_registry {
            state.insert(name.clone(), param.data().to_vec());
        }

        state
    }

    /// Load optimizer state
    pub fn load_state_dict(
        &mut self,
        state: HashMap<String, Vec<f64>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (name, data) in state {
            if let Some(param) = self.parameter_registry.get_mut(&name) {
                param.set_data(Array1::from_vec(data));
            }
        }
        Ok(())
    }

    // Helper methods

    fn extract_gradients(&mut self, model: &Model) -> Result<(), Box<dyn std::error::Error>> {
        for (name, param) in &mut self.parameter_registry {
            if let Some(model_param) = model.get_parameter(name) {
                if let Some(grad_data) = model_param.grad() {
                    let grad_f64: Vec<f64> = grad_data
                        .iter()
                        .map(|&x| x.to_f64().unwrap_or(0.0))
                        .collect();
                    param.set_grad(Array1::from_vec(grad_f64));
                }
            }
        }
        Ok(())
    }

    fn clip_gradients(&mut self, clip_value: f64) {
        for param in self.parameter_registry.values_mut() {
            if let Some(mut grad) = param.grad_mut() {
                let grad_norm = grad.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if grad_norm > clip_value {
                    let scale = clip_value / grad_norm;
                    grad.mapv_inplace(|x| x * scale);
                }
            }
        }
    }

    fn apply_regularization(&mut self, model: &Model) -> Result<(), Box<dyn std::error::Error>> {
        for regularizer in &self.regularizers {
            for param in self.parameter_registry.values_mut() {
                if let Some(data) = param.data_f32() {
                    let reg_grad = regularizer.regularize(&data);
                    // Add regularization gradient to existing gradient
                    if let Some(mut grad) = param.grad_mut() {
                        for (i, &reg_val) in reg_grad.iter().enumerate() {
                            if i < grad.len() {
                                grad[i] += reg_val as f64;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn update_model_parameters(&self, model: &mut Model) -> Result<(), Box<dyn std::error::Error>> {
        let mut updated_state = HashMap::new();

        for (name, param) in &self.parameter_registry {
            let data_t: Vec<T> = param
                .data()
                .iter()
                .map(|&x| T::from(x).unwrap_or(T::zero()))
                .collect();
            updated_state.insert(name.clone(), data_t);
        }

        model.load_state_dict(updated_state);
        Ok(())
    }

    fn update_learning_rate(&mut self, new_lr: f64) {
        // Update learning rate in optimizer configuration
        // This would depend on the specific optimizer implementation
    }

    fn update_metrics(&mut self, loss: f64) {
        self.metrics.loss_history.push(loss);

        // Calculate gradient norm
        let mut total_grad_norm = 0.0;
        for param in self.parameter_registry.values() {
            if let Some(grad) = param.grad() {
                total_grad_norm += grad.iter().map(|&x| x * x).sum::<f64>();
            }
        }
        self.metrics.gradient_norms.push(total_grad_norm.sqrt());

        // Calculate parameter norm
        let mut total_param_norm = 0.0;
        for param in self.parameter_registry.values() {
            total_param_norm += param.data().iter().map(|&x| x * x).sum::<f64>();
        }
        self.metrics.parameter_norms.push(total_param_norm.sqrt());

        // Update convergence metrics
        self.update_convergence_metrics();
    }

    fn update_convergence_metrics(&mut self) {
        let conv = &mut self.metrics.convergence_metrics;

        // Update gradient norm
        if let Some(&grad_norm) = self.metrics.gradient_norms.last() {
            conv.gradient_norm = grad_norm;
        }

        // Calculate loss improvement rate
        if self.metrics.loss_history.len() >= 2 {
            let current_loss = self.metrics.loss_history.last().unwrap();
            let prev_loss = self.metrics.loss_history[self.metrics.loss_history.len() - 2];
            conv.loss_improvement_rate = (prev_loss - current_loss) / prev_loss.abs();
        }

        // Simple convergence check
        conv.is_converged = conv.gradient_norm < 1e-6 && conv.loss_improvement_rate.abs() < 1e-8;
        conv.convergence_score = if conv.is_converged { 1.0 } else { 0.0 };
    }
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub initial_lr: f64,
    pub min_lr: Option<f64>,
    pub max_lr: Option<f64>,
    pub gamma: Option<f64>,
    pub step_size: Option<usize>,
    pub total_steps: Option<usize>,
    pub pct_start: Option<f64>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            gradient_clipping: None,
            gradient_accumulation_steps: 1,
            mixed_precision: false,
            checkpoint_frequency: 1000,
            early_stopping: None,
            warmup_steps: 0,
            max_steps: None,
        }
    }
}

/// Training loop implementation
pub struct UniversalTrainingLoop<T, Tensor, Model>
where
    T: num_traits::Float + std::fmt::Debug + Send + Sync + 'static,
    Tensor: FrameworkTensor<T>,
    Model: FrameworkModel<T, Tensor>,
{
    optimizer_adapter: UniversalOptimizerAdapter<T, Tensor, Model>,
    model: Model,
    config: TrainingConfig,
}

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub validation_frequency: usize,
    pub save_frequency: usize,
    pub log_frequency: usize,
    pub output_dir: String,
}

impl<T, Tensor, Model> UniversalTrainingLoop<T, Tensor, Model>
where
    T: num_traits::Float + std::fmt::Debug + Send + Sync + 'static,
    Tensor: FrameworkTensor<T>,
    Model: FrameworkModel<T, Tensor>,
{
    pub fn new(
        optimizer_adapter: UniversalOptimizerAdapter<T, Tensor, Model>,
        model: Model,
        config: TrainingConfig,
    ) -> Self {
        Self {
            optimizer_adapter,
            model,
            config,
        }
    }

    pub fn train<DataLoader>(&mut self, train_loader: DataLoader, val_loader: Option<DataLoader>)
    where
        DataLoader: Iterator<Item = (Tensor, Tensor)>,
    {
        println!("🚀 Starting universal training loop");
        println!("📊 Training for {} epochs", self.config.epochs);

        for epoch in 0..self.config.epochs {
            let epoch_start = std::time::Instant::now();
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            // Training phase
            for (batch_idx, (inputs, targets)) in train_loader.enumerate() {
                self.optimizer_adapter.zero_grad(&mut self.model);

                // Forward pass
                let outputs = self.model.forward(&inputs);
                let loss = self.compute_loss(&outputs, &targets);

                // Backward pass (framework-specific)
                self.backward_pass(&outputs, &targets);

                // Optimization step
                self.optimizer_adapter.step(&mut self.model, loss).unwrap();

                epoch_loss += loss;
                batch_count += 1;

                // Logging
                if batch_idx % self.config.log_frequency == 0 {
                    let metrics = self.optimizer_adapter.get_metrics();
                    println!(
                        "📈 Epoch {}, Batch {}: Loss = {:.6}, Grad Norm = {:.6}",
                        epoch,
                        batch_idx,
                        loss,
                        metrics.gradient_norms.last().unwrap_or(&0.0)
                    );
                }
            }

            let avg_loss = epoch_loss / batch_count as f64;
            let epoch_time = epoch_start.elapsed();

            println!(
                "✅ Epoch {} completed: Loss = {:.6}, Time = {:?}",
                epoch, avg_loss, epoch_time
            );

            // Validation
            if let Some(ref val_loader) = val_loader {
                if epoch % self.config.validation_frequency == 0 {
                    self.validate(val_loader);
                }
            }

            // Checkpointing
            if epoch % self.config.save_frequency == 0 {
                self.save_checkpoint(epoch);
            }

            // Check convergence
            let metrics = self.optimizer_adapter.get_metrics();
            if metrics.convergence_metrics.is_converged {
                println!("🎯 Convergence detected at epoch {}", epoch);
                break;
            }
        }

        println!("🏁 Training completed!");
    }

    fn compute_loss(&self, outputs: &Tensor, targets: &Tensor) -> f64 {
        // Framework-specific loss computation
        // This would be implemented based on the specific ML framework
        0.5 // Placeholder
    }

    fn backward_pass(&self, outputs: &Tensor, targets: &Tensor) {
        // Framework-specific backward pass
        // This would trigger gradient computation in the specific framework
    }

    fn validate<DataLoader>(&mut self, val_loader: &DataLoader)
    where
        DataLoader: Iterator<Item = (Tensor, Tensor)> + Clone,
    {
        println!("🔍 Running validation...");

        let mut val_loss = 0.0;
        let mut val_count = 0;

        for (inputs, targets) in val_loader.clone() {
            let outputs = self.model.forward(&inputs);
            let loss = self.compute_loss(&outputs, &targets);
            val_loss += loss;
            val_count += 1;
        }

        let avg_val_loss = val_loss / val_count as f64;
        println!("📊 Validation Loss: {:.6}", avg_val_loss);
    }

    fn save_checkpoint(&self, epoch: usize) {
        println!("💾 Saving checkpoint at epoch {}", epoch);
        // Save model and optimizer state
        let model_state = self.model.state_dict();
        let optimizer_state = self.optimizer_adapter.state_dict();

        // Framework-specific checkpoint saving would go here
    }
}

/// Example usage with different ML frameworks
#[allow(dead_code)]
fn example_usage() {
    println!("🎯 Universal SciRS2 Integration Examples\n");

    // Example 1: Basic optimizer setup
    println!("📝 Example 1: Basic Optimizer Setup");
    let config = OptimizerConfig::new(0.001).weight_decay(1e-4);
    let mut adapter =
        UniversalOptimizerAdapter::<f64, DummyTensor, DummyModel>::new("adam", config).unwrap();

    // Add parameter groups
    let features_group = ParameterGroup {
        name: "features".to_string(),
        parameter_names: vec!["layer1.weight".to_string(), "layer1.bias".to_string()],
        learning_rate: 0.001,
        weight_decay: 1e-4,
        momentum: 0.9,
        enabled: true,
    };
    adapter.add_parameter_group(features_group);

    // Add scheduler
    let scheduler_config = SchedulerConfig {
        initial_lr: 0.001,
        min_lr: Some(1e-6),
        total_steps: Some(1000),
        ..Default::default()
    };
    adapter.add_scheduler("cosine", scheduler_config).unwrap();

    // Add regularization
    adapter.add_regularizer("l2", 1e-4);

    println!("✅ Optimizer configured with parameter groups, scheduler, and regularization");

    // Example 2: Multi-optimizer setup
    println!("\n📝 Example 2: Multi-Optimizer Setup");
    let backbone_config = OptimizerConfig::new(1e-5);
    let head_config = OptimizerConfig::new(1e-3);

    let backbone_adapter =
        UniversalOptimizerAdapter::<f64, DummyTensor, DummyModel>::new("sgd", backbone_config)
            .unwrap();
    let head_adapter =
        UniversalOptimizerAdapter::<f64, DummyTensor, DummyModel>::new("adam", head_config)
            .unwrap();

    println!("✅ Multi-optimizer setup for transfer learning");

    // Example 3: Training loop
    println!("\n📝 Example 3: Training Loop");
    let training_config = TrainingConfig {
        epochs: 10,
        batch_size: 32,
        validation_frequency: 5,
        save_frequency: 10,
        log_frequency: 100,
        output_dir: "./checkpoints".to_string(),
    };

    println!("✅ Training configuration ready");

    println!("\n🎉 Universal integration examples completed!");
    println!("💡 This adapter pattern works with:");
    println!("   ✅ PyTorch (via Python bindings)");
    println!("   ✅ Candle (native Rust)");
    println!("   ✅ Burn (native Rust)");
    println!("   ✅ TensorFlow (via C API)");
    println!("   ✅ JAX (via Python bindings)");
    println!("   ✅ Custom ML frameworks");
}

// Dummy implementations for testing
#[derive(Debug)]
struct DummyTensor {
    data: Vec<f64>,
    grad: Option<Vec<f64>>,
    shape: Vec<usize>,
}

impl FrameworkTensor<f64> for DummyTensor {
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
    fn data(&self) -> &[f64] {
        &self.data
    }
    fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }
    fn grad(&self) -> Option<&[f64]> {
        self.grad.as_deref()
    }
    fn set_grad(&mut self, grad: &[f64]) {
        self.grad = Some(grad.to_vec());
    }
    fn zero_grad(&mut self) {
        self.grad = None;
    }
    fn to_device(&mut self, _device: &str) {}
    fn device(&self) -> String {
        "cpu".to_string()
    }
}

#[derive(Debug)]
struct DummyModel {
    parameters: HashMap<String, DummyTensor>,
}

impl FrameworkModel<f64, DummyTensor> for DummyModel {
    fn parameters(&self) -> Vec<&DummyTensor> {
        self.parameters.values().collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut DummyTensor> {
        self.parameters.values_mut().collect()
    }

    fn forward(&self, _input: &DummyTensor) -> DummyTensor {
        DummyTensor {
            data: vec![0.0; 10],
            grad: None,
            shape: vec![10],
        }
    }

    fn get_parameter(&self, name: &str) -> Option<&DummyTensor> {
        self.parameters.get(name)
    }

    fn set_parameter(&mut self, name: &str, param: DummyTensor) {
        self.parameters.insert(name.to_string(), param);
    }

    fn state_dict(&self) -> HashMap<String, Vec<f64>> {
        self.parameters
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.data.clone()))
            .collect()
    }

    fn load_state_dict(&mut self, state: HashMap<String, Vec<f64>>) {
        for (name, data) in state {
            if let Some(param) = self.parameters.get_mut(&name) {
                param.data = data;
            }
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            initial_lr: 0.001,
            min_lr: None,
            max_lr: None,
            gamma: None,
            step_size: None,
            total_steps: None,
            pct_start: None,
        }
    }
}

#[allow(dead_code)]
fn main() {
    example_usage();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let config = OptimizerConfig::new(0.001);
        let adapter =
            UniversalOptimizerAdapter::<f64, DummyTensor, DummyModel>::new("adam", config);
        assert!(adapter.is_ok());
    }

    #[test]
    fn test_parameter_group() {
        let group = ParameterGroup {
            name: "test".to_string(),
            parameter_names: vec!["param1".to_string()],
            learning_rate: 0.001,
            weight_decay: 1e-4,
            momentum: 0.9,
            enabled: true,
        };

        assert_eq!(group.name, "test");
        assert!(group.enabled);
    }

    #[test]
    fn test_dummy_tensor() {
        let mut tensor = DummyTensor {
            data: vec![1.0, 2.0, 3.0],
            grad: None,
            shape: vec![3],
        };

        assert_eq!(tensor.data(), &[1.0, 2.0, 3.0]);
        assert!(tensor.grad().is_none());

        tensor.set_grad(&[0.1, 0.2, 0.3]);
        assert!(tensor.grad().is_some());

        tensor.zero_grad();
        assert!(tensor.grad().is_none());
    }
}
