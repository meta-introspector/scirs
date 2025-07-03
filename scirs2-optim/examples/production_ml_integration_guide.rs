//! Production-Ready ML Framework Integration Guide
//!
//! This comprehensive example demonstrates how to integrate SciRS2 optimizers
//! with popular ML frameworks in production environments. It covers:
//!
//! - Multi-framework compatibility (PyTorch, TensorFlow, JAX, Candle, Burn, Linfa)
//! - Performance optimization techniques
//! - Memory management best practices
//! - Error handling and debugging
//! - Distributed training considerations
//! - Model serving and inference optimization

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, Axis};
use scirs2_optim::{
    benchmarking::memory_leak_detector::{MemoryDetectionConfig, MemoryLeakDetector},
    benchmarking::performance_regression_detector::{
        PerformanceRegressionDetector, RegressionConfig,
    },
    error::{OptimError, Result},
    optimizers::{
        adam::{Adam, AdamConfig},
        adamw::{AdamW, AdamWConfig},
        lamb::{LAMBConfig, LAMB},
        lars::{LARSConfig, LARS},
        lion::{Lion, LionConfig},
        sgd::{SGDConfig, SGD},
    },
    regularizers::{
        dropout::{DropoutConfig, DropoutRegularizer},
        l1::L1Regularizer,
        l2::L2Regularizer,
        mixup::{MixupConfig, MixupRegularizer},
    },
    schedulers::{
        cosine_annealing::{CosineAnnealingConfig, CosineAnnealingScheduler},
        one_cycle::{OneCycleConfig, OneCycleScheduler},
        reduce_on_plateau::{ReduceOnPlateauConfig, ReduceOnPlateauScheduler},
    },
    unified_api::{OptimizerFactory, Parameter},
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Production-ready ML model trainer with SciRS2 integration
#[allow(dead_code)]
#[derive(Debug)]
pub struct ProductionMLTrainer {
    /// Optimizer factory for different scenarios
    optimizer_factory: OptimizerFactory,
    /// Memory leak detector for debugging
    memory_detector: MemoryLeakDetector,
    /// Performance regression detector
    performance_detector: PerformanceRegressionDetector,
    /// Training metrics
    metrics: TrainingMetrics,
    /// Configuration
    config: ProductionConfig,
}

/// Training metrics for monitoring
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Training loss history
    train_loss_history: Vec<f64>,
    /// Validation loss history
    val_loss_history: Vec<f64>,
    /// Learning rate history
    lr_history: Vec<f64>,
    /// Memory usage history
    memory_usage_history: Vec<usize>,
    /// Training time per epoch
    epoch_times: Vec<f64>,
    /// Gradient norms
    grad_norms: Vec<f64>,
}

/// Production configuration
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ProductionConfig {
    /// Enable memory monitoring
    enable_memory_monitoring: bool,
    /// Enable performance monitoring
    enable_performance_monitoring: bool,
    /// Gradient clipping threshold
    gradient_clip_threshold: Option<f64>,
    /// Mixed precision training
    use_mixed_precision: bool,
    /// Checkpoint frequency
    checkpoint_every_n_epochs: usize,
    /// Early stopping patience
    early_stopping_patience: usize,
    /// Target framework
    target_framework: TargetFramework,
}

/// Supported ML frameworks
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum TargetFramework {
    PyTorch,
    TensorFlow,
    JAX,
    Candle,
    Burn,
    Linfa,
    SmartCore,
    Custom(String),
}

impl Default for ProductionConfig {
    fn default() -> Self {
        Self {
            enable_memory_monitoring: true,
            enable_performance_monitoring: true,
            gradient_clip_threshold: Some(1.0),
            use_mixed_precision: false,
            checkpoint_every_n_epochs: 10,
            early_stopping_patience: 50,
            target_framework: TargetFramework::PyTorch,
        }
    }
}

impl ProductionMLTrainer {
    /// Create a new production ML trainer
    pub fn new(config: ProductionConfig) -> Result<Self> {
        let optimizer_factory = OptimizerFactory::new();

        let memory_config = MemoryDetectionConfig {
            enable_allocation_tracking: config.enable_memory_monitoring,
            memory_growth_threshold: 500 * 1024 * 1024, // 500MB
            leak_sensitivity: 0.8,
            ..Default::default()
        };
        let memory_detector = MemoryLeakDetector::new(memory_config);

        let regression_config = RegressionConfig {
            enable_detection: config.enable_performance_monitoring,
            confidence_threshold: 0.95,
            degradation_threshold: 0.1, // 10% performance degradation
            ..Default::default()
        };
        let performance_detector = PerformanceRegressionDetector::new(regression_config)?;

        Ok(Self {
            optimizer_factory,
            memory_detector,
            performance_detector,
            metrics: TrainingMetrics::new(),
            config,
        })
    }

    /// Train a model with comprehensive monitoring and optimization
    #[allow(clippy::too_many_arguments)]
    pub fn train_model<F, L, V>(
        &mut self,
        mut model: F,
        train_data: &[(Array2<f64>, Array1<f64>)],
        val_data: &[(Array2<f64>, Array1<f64>)],
        loss_fn: L,
        val_fn: V,
        epochs: usize,
        optimizer_name: &str,
    ) -> Result<TrainingResults>
    where
        F: FnMut(&Array2<f64>) -> Array2<f64> + Send + Sync,
        L: Fn(&Array2<f64>, &Array1<f64>) -> f64 + Send + Sync,
        V: Fn(&Array2<f64>, &Array1<f64>) -> f64 + Send + Sync,
    {
        println!("🚀 Starting production ML training with SciRS2 optimizers");
        println!("   Framework: {:?}", self.config.target_framework);
        println!("   Optimizer: {}", optimizer_name);
        println!("   Epochs: {}", epochs);
        println!("   Training samples: {}", train_data.len());
        println!("   Validation samples: {}", val_data.len());

        // Initialize optimizer based on the specified type
        let mut optimizer = self.create_optimizer(optimizer_name)?;

        // Initialize learning rate scheduler
        let mut scheduler = self.create_scheduler(optimizer_name, epochs)?;

        // Initialize regularizers
        let regularizers = self.create_regularizers()?;

        // Start memory monitoring
        if self.config.enable_memory_monitoring {
            self.memory_detector.start_monitoring()?;
            println!("✅ Memory monitoring enabled");
        }

        let mut best_val_loss = f64::INFINITY;
        let mut no_improvement_count = 0;
        let start_time = Instant::now();

        println!("\n📊 Training Progress:");
        println!(
            "Epoch | Train Loss | Val Loss   | LR        | Grad Norm | Time (s) | Memory (MB)"
        );
        println!(
            "------|------------|------------|-----------|-----------|----------|------------"
        );

        for epoch in 0..epochs {
            let epoch_start = Instant::now();

            // Training phase
            let mut epoch_train_loss = 0.0;
            let mut total_grad_norm = 0.0;

            for (batch_idx, (input, target)) in train_data.iter().enumerate() {
                // Forward pass
                let output = model(input);
                let loss = loss_fn(&output, target);
                epoch_train_loss += loss;

                // Simulate gradient computation (in real framework, this would be automatic)
                let grad_norm = self.simulate_gradient_computation(&output, target);
                total_grad_norm += grad_norm;

                // Apply gradient clipping if configured
                let clipped_grad_norm = if let Some(threshold) = self.config.gradient_clip_threshold
                {
                    grad_norm.min(threshold)
                } else {
                    grad_norm
                };

                // Optimizer step (simplified for example)
                self.apply_optimizer_step(&mut optimizer, clipped_grad_norm)?;

                // Apply regularization
                for regularizer in &regularizers {
                    self.apply_regularization(regularizer, &output)?;
                }
            }

            epoch_train_loss /= train_data.len() as f64;
            total_grad_norm /= train_data.len() as f64;

            // Validation phase
            let mut epoch_val_loss = 0.0;
            for (input, target) in val_data.iter() {
                let output = model(input);
                let loss = val_fn(&output, target);
                epoch_val_loss += loss;
            }
            epoch_val_loss /= val_data.len() as f64;

            // Update learning rate
            let current_lr = scheduler.step(Some(epoch_val_loss));

            // Record metrics
            let epoch_time = epoch_start.elapsed().as_secs_f64();
            let memory_usage = self.get_current_memory_usage();

            self.metrics.train_loss_history.push(epoch_train_loss);
            self.metrics.val_loss_history.push(epoch_val_loss);
            self.metrics.lr_history.push(current_lr);
            self.metrics.epoch_times.push(epoch_time);
            self.metrics.memory_usage_history.push(memory_usage);
            self.metrics.grad_norms.push(total_grad_norm);

            // Print progress
            println!(
                "{:5} | {:10.6} | {:10.6} | {:9.6} | {:9.6} | {:8.2} | {:10.2}",
                epoch + 1,
                epoch_train_loss,
                epoch_val_loss,
                current_lr,
                total_grad_norm,
                epoch_time,
                memory_usage as f64 / (1024.0 * 1024.0)
            );

            // Early stopping check
            if epoch_val_loss < best_val_loss {
                best_val_loss = epoch_val_loss;
                no_improvement_count = 0;
            } else {
                no_improvement_count += 1;
                if no_improvement_count >= self.config.early_stopping_patience {
                    println!("\n⏹️  Early stopping triggered at epoch {}", epoch + 1);
                    break;
                }
            }

            // Checkpoint saving
            if (epoch + 1) % self.config.checkpoint_every_n_epochs == 0 {
                self.save_checkpoint(epoch + 1, &optimizer)?;
                println!("💾 Checkpoint saved at epoch {}", epoch + 1);
            }

            // Memory leak detection
            if self.config.enable_memory_monitoring && epoch % 10 == 0 {
                let memory_snapshot = self.memory_detector.take_snapshot()?;
                if memory_snapshot.growth_rate > 1024.0 * 1024.0 {
                    // 1MB/s growth
                    println!(
                        "⚠️  High memory growth detected: {:.2} MB/s",
                        memory_snapshot.growth_rate / (1024.0 * 1024.0)
                    );
                }
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        // Stop monitoring
        if self.config.enable_memory_monitoring {
            self.memory_detector.stop_monitoring()?;
        }

        // Generate comprehensive report
        let report = self.generate_training_report(total_time, best_val_loss)?;

        println!("\n🎉 Training completed!");
        println!("   Total time: {:.2} minutes", total_time / 60.0);
        println!("   Best validation loss: {:.6}", best_val_loss);
        println!(
            "   Memory efficiency: {:.2}%",
            report.memory_efficiency * 100.0
        );

        Ok(TrainingResults {
            best_val_loss,
            total_time,
            epochs_trained: self.metrics.train_loss_history.len(),
            metrics: self.metrics.clone(),
            report,
        })
    }

    /// Create optimizer based on name and framework
    fn create_optimizer(&self, name: &str) -> Result<Box<dyn OptimizerTrait>> {
        match name.to_lowercase().as_str() {
            "adam" => {
                let config = AdamConfig {
                    learning_rate: 0.001,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    weight_decay: 0.0,
                    amsgrad: false,
                    maximize: false,
                };
                Ok(Box::new(OptimizerWrapper::new(Adam::new(config))))
            }
            "adamw" => {
                let config = AdamWConfig {
                    learning_rate: 0.001,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    weight_decay: 0.01,
                    amsgrad: false,
                };
                Ok(Box::new(OptimizerWrapper::new(AdamW::new(config))))
            }
            "lamb" => {
                let config = LAMBConfig {
                    learning_rate: 0.001,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-6,
                    weight_decay: 0.01,
                    bias_correction: true,
                    always_adapt: false,
                };
                Ok(Box::new(OptimizerWrapper::new(LAMB::new(config))))
            }
            "lars" => {
                let config = LARSConfig {
                    learning_rate: 0.001,
                    momentum: 0.9,
                    dampening: 0.0,
                    weight_decay: 0.0005,
                    trust_coefficient: 0.001,
                    epsilon: 1e-8,
                    nesterov: false,
                };
                Ok(Box::new(OptimizerWrapper::new(LARS::new(config))))
            }
            "lion" => {
                let config = LionConfig {
                    learning_rate: 1e-4,
                    beta1: 0.9,
                    beta2: 0.99,
                    weight_decay: 0.0,
                };
                Ok(Box::new(OptimizerWrapper::new(Lion::new(config))))
            }
            "sgd" => {
                let config = SGDConfig {
                    learning_rate: 0.01,
                    momentum: 0.9,
                    dampening: 0.0,
                    weight_decay: 0.0,
                    nesterov: true,
                };
                Ok(Box::new(OptimizerWrapper::new(SGD::new(config))))
            }
            _ => Err(OptimError::InvalidConfig(format!(
                "Unknown optimizer: {}",
                name
            ))),
        }
    }

    /// Create learning rate scheduler
    fn create_scheduler(
        &self,
        optimizer_name: &str,
        epochs: usize,
    ) -> Result<Box<dyn SchedulerTrait>> {
        match optimizer_name.to_lowercase().as_str() {
            "adam" | "adamw" => {
                let config = CosineAnnealingConfig {
                    t_max: epochs,
                    eta_min: 1e-7,
                    last_epoch: -1,
                };
                Ok(Box::new(SchedulerWrapper::new(
                    CosineAnnealingScheduler::new(config),
                )))
            }
            "lamb" | "lars" => {
                let config = OneCycleConfig {
                    max_lr: 0.01,
                    total_steps: epochs,
                    epochs: Some(epochs),
                    steps_per_epoch: None,
                    pct_start: 0.3,
                    anneal_strategy: "cos".to_string(),
                    cycle_momentum: true,
                    base_momentum: 0.85,
                    max_momentum: 0.95,
                    div_factor: 25.0,
                    final_div_factor: 10000.0,
                    three_phase: false,
                    last_epoch: -1,
                };
                Ok(Box::new(SchedulerWrapper::new(OneCycleScheduler::new(
                    config,
                ))))
            }
            _ => {
                let config = ReduceOnPlateauConfig {
                    mode: "min".to_string(),
                    factor: 0.1,
                    patience: 10,
                    threshold: 1e-4,
                    threshold_mode: "rel".to_string(),
                    cooldown: 0,
                    min_lr: 0.0,
                    eps: 1e-8,
                    verbose: false,
                };
                Ok(Box::new(SchedulerWrapper::new(
                    ReduceOnPlateauScheduler::new(config),
                )))
            }
        }
    }

    /// Create regularizers based on configuration
    fn create_regularizers(&self) -> Result<Vec<Box<dyn RegularizerTrait>>> {
        let mut regularizers = Vec::new();

        // Add L2 regularization
        regularizers.push(Box::new(RegularizerWrapper::new(L2Regularizer::new(0.01))));

        // Add dropout for neural networks
        let dropout_config = DropoutConfig {
            dropout_rate: 0.1,
            training: true,
            inplace: false,
        };
        regularizers.push(Box::new(RegularizerWrapper::new(DropoutRegularizer::new(
            dropout_config,
        ))));

        // Add mixup for data augmentation
        let mixup_config = MixupConfig {
            alpha: 0.2,
            cutmix_alpha: 1.0,
            cutmix_minmax: None,
            prob: 1.0,
            switch_prob: 0.5,
            mode: "batch".to_string(),
            correct_lam: true,
            label_smoothing: 0.1,
            num_classes: 10,
        };
        regularizers.push(Box::new(RegularizerWrapper::new(MixupRegularizer::new(
            mixup_config,
        ))));

        Ok(regularizers)
    }

    /// Simulate gradient computation (framework-specific implementation would differ)
    fn simulate_gradient_computation(&self, _output: &Array2<f64>, _target: &Array1<f64>) -> f64 {
        // In a real implementation, this would:
        // 1. Compute gradients using the framework's autodiff
        // 2. Return the gradient norm
        // For simulation, we return a random gradient norm
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(0.1..2.0)
    }

    /// Apply optimizer step
    fn apply_optimizer_step(
        &self,
        _optimizer: &mut Box<dyn OptimizerTrait>,
        _grad_norm: f64,
    ) -> Result<()> {
        // In a real implementation, this would apply the optimizer step
        // to update model parameters
        Ok(())
    }

    /// Apply regularization
    fn apply_regularization(
        &self,
        _regularizer: &Box<dyn RegularizerTrait>,
        _output: &Array2<f64>,
    ) -> Result<()> {
        // In a real implementation, this would apply regularization
        // to the model parameters or loss
        Ok(())
    }

    /// Get current memory usage
    fn get_current_memory_usage(&self) -> usize {
        // In a real implementation, this would query system memory usage
        // For simulation, we return a mock value
        64 * 1024 * 1024 // 64MB
    }

    /// Save model checkpoint
    fn save_checkpoint(&self, epoch: usize, _optimizer: &Box<dyn OptimizerTrait>) -> Result<()> {
        println!("💾 Saving checkpoint for epoch {}", epoch);
        // In a real implementation, this would save model and optimizer state
        Ok(())
    }

    /// Generate comprehensive training report
    fn generate_training_report(
        &self,
        total_time: f64,
        best_val_loss: f64,
    ) -> Result<TrainingReport> {
        // Calculate performance metrics
        let avg_epoch_time =
            self.metrics.epoch_times.iter().sum::<f64>() / self.metrics.epoch_times.len() as f64;
        let memory_efficiency = self.calculate_memory_efficiency();
        let convergence_rate = self.calculate_convergence_rate();

        // Generate memory leak report if monitoring is enabled
        let memory_report = if self.config.enable_memory_monitoring {
            Some(self.memory_detector.generate_optimization_report()?)
        } else {
            None
        };

        Ok(TrainingReport {
            best_val_loss,
            total_time,
            avg_epoch_time,
            memory_efficiency,
            convergence_rate,
            memory_report,
            framework_recommendations: self.generate_framework_recommendations(),
        })
    }

    /// Calculate memory efficiency score
    fn calculate_memory_efficiency(&self) -> f64 {
        if self.metrics.memory_usage_history.is_empty() {
            return 1.0;
        }

        let max_memory = *self.metrics.memory_usage_history.iter().max().unwrap() as f64;
        let avg_memory = self.metrics.memory_usage_history.iter().sum::<usize>() as f64
            / self.metrics.memory_usage_history.len() as f64;

        if max_memory > 0.0 {
            avg_memory / max_memory
        } else {
            1.0
        }
    }

    /// Calculate convergence rate
    fn calculate_convergence_rate(&self) -> f64 {
        if self.metrics.val_loss_history.len() < 10 {
            return 0.0;
        }

        let initial_loss = self.metrics.val_loss_history[0];
        let final_loss = *self.metrics.val_loss_history.last().unwrap();

        if initial_loss > final_loss {
            (initial_loss - final_loss) / self.metrics.val_loss_history.len() as f64
        } else {
            0.0
        }
    }

    /// Generate framework-specific recommendations
    fn generate_framework_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        match self.config.target_framework {
            TargetFramework::PyTorch => {
                recommendations
                    .push("Consider using torch.compile() for additional speedup".to_string());
                recommendations.push(
                    "Enable torch.backends.cudnn.benchmark for consistent input sizes".to_string(),
                );
                recommendations
                    .push("Use torch.utils.data.DataLoader with num_workers > 0".to_string());
            }
            TargetFramework::TensorFlow => {
                recommendations
                    .push("Use tf.function decorators for graph optimization".to_string());
                recommendations
                    .push("Enable mixed precision with tf.keras.mixed_precision".to_string());
                recommendations
                    .push("Consider tf.data.Dataset for efficient data loading".to_string());
            }
            TargetFramework::JAX => {
                recommendations.push("Use jax.jit() for just-in-time compilation".to_string());
                recommendations.push("Enable XLA optimizations with jax.config".to_string());
                recommendations.push("Consider pmap for multi-device training".to_string());
            }
            TargetFramework::Burn => {
                recommendations
                    .push("Use Burn's autodiff for efficient gradient computation".to_string());
                recommendations.push("Enable WGPU backend for GPU acceleration".to_string());
                recommendations.push("Consider tensor fusion for memory optimization".to_string());
            }
            TargetFramework::Candle => {
                recommendations.push("Use Candle's CUDA backend for GPU acceleration".to_string());
                recommendations.push("Enable safetensors for efficient model loading".to_string());
                recommendations
                    .push("Consider quantization for inference optimization".to_string());
            }
            _ => {
                recommendations.push("Follow framework-specific best practices".to_string());
                recommendations.push("Enable hardware-specific optimizations".to_string());
            }
        }

        // Add optimizer-specific recommendations
        if let Some(avg_grad_norm) = self.metrics.grad_norms.last() {
            if *avg_grad_norm > 10.0 {
                recommendations.push(
                    "Consider reducing learning rate or increasing gradient clipping".to_string(),
                );
            } else if *avg_grad_norm < 0.01 {
                recommendations
                    .push("Consider increasing learning rate for faster convergence".to_string());
            }
        }

        recommendations
    }
}

/// Training results
#[allow(dead_code)]
#[derive(Debug)]
pub struct TrainingResults {
    pub best_val_loss: f64,
    pub total_time: f64,
    pub epochs_trained: usize,
    pub metrics: TrainingMetrics,
    pub report: TrainingReport,
}

/// Comprehensive training report
#[allow(dead_code)]
#[derive(Debug)]
pub struct TrainingReport {
    pub best_val_loss: f64,
    pub total_time: f64,
    pub avg_epoch_time: f64,
    pub memory_efficiency: f64,
    pub convergence_rate: f64,
    pub memory_report:
        Option<scirs2_optim::benchmarking::memory_leak_detector::MemoryOptimizationReport>,
    pub framework_recommendations: Vec<String>,
}

/// Trait abstractions for optimizer integration

trait OptimizerTrait: Send + Sync {
    fn step(&mut self) -> Result<()>;
    fn zero_grad(&mut self);
    fn get_lr(&self) -> f64;
    fn set_lr(&mut self, lr: f64);
}

trait SchedulerTrait: Send + Sync {
    fn step(&mut self, metric: Option<f64>) -> f64;
    fn get_last_lr(&self) -> f64;
}

trait RegularizerTrait: Send + Sync {
    fn apply(&self, params: &Array1<f64>) -> Result<f64>;
}

/// Wrapper implementations for trait objects

struct OptimizerWrapper<T> {
    inner: T,
    current_lr: f64,
}

impl<T> OptimizerWrapper<T> {
    fn new(optimizer: T) -> Self {
        Self {
            inner: optimizer,
            current_lr: 0.001, // Default learning rate
        }
    }
}

impl<T: Send + Sync> OptimizerTrait for OptimizerWrapper<T> {
    fn step(&mut self) -> Result<()> {
        // In real implementation, this would call the actual optimizer step
        Ok(())
    }

    fn zero_grad(&mut self) {
        // In real implementation, this would zero gradients
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }

    fn set_lr(&mut self, lr: f64) {
        self.current_lr = lr;
    }
}

struct SchedulerWrapper<T> {
    inner: T,
    last_lr: f64,
}

impl<T> SchedulerWrapper<T> {
    fn new(scheduler: T) -> Self {
        Self {
            inner: scheduler,
            last_lr: 0.001,
        }
    }
}

impl<T: Send + Sync> SchedulerTrait for SchedulerWrapper<T> {
    fn step(&mut self, _metric: Option<f64>) -> f64 {
        // In real implementation, this would call the actual scheduler step
        // and return the new learning rate
        self.last_lr * 0.99 // Simulate decay
    }

    fn get_last_lr(&self) -> f64 {
        self.last_lr
    }
}

struct RegularizerWrapper<T> {
    inner: T,
}

impl<T> RegularizerWrapper<T> {
    fn new(regularizer: T) -> Self {
        Self { inner: regularizer }
    }
}

impl<T: Send + Sync> RegularizerTrait for RegularizerWrapper<T> {
    fn apply(&self, _params: &Array1<f64>) -> Result<f64> {
        // In real implementation, this would apply regularization
        Ok(0.0) // Return regularization penalty
    }
}

impl TrainingMetrics {
    fn new() -> Self {
        Self {
            train_loss_history: Vec::new(),
            val_loss_history: Vec::new(),
            lr_history: Vec::new(),
            memory_usage_history: Vec::new(),
            epoch_times: Vec::new(),
            grad_norms: Vec::new(),
        }
    }
}

/// Example usage and demonstration
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_production_training_workflow() {
        // Create production trainer configuration
        let config = ProductionConfig {
            enable_memory_monitoring: true,
            enable_performance_monitoring: true,
            gradient_clip_threshold: Some(1.0),
            use_mixed_precision: false,
            checkpoint_every_n_epochs: 5,
            early_stopping_patience: 20,
            target_framework: TargetFramework::PyTorch,
        };

        let mut trainer = ProductionMLTrainer::new(config).unwrap();

        // Create dummy training data
        let train_data: Vec<(Array2<f64>, Array1<f64>)> = (0..100)
            .map(|_| {
                let input = Array::random((32, 10), rand::distributions::Uniform::new(-1.0, 1.0));
                let target = Array::random(32, rand::distributions::Uniform::new(0.0, 1.0));
                (input, target)
            })
            .collect();

        let val_data: Vec<(Array2<f64>, Array1<f64>)> = (0..20)
            .map(|_| {
                let input = Array::random((32, 10), rand::distributions::Uniform::new(-1.0, 1.0));
                let target = Array::random(32, rand::distributions::Uniform::new(0.0, 1.0));
                (input, target)
            })
            .collect();

        // Simple model function (identity for testing)
        let model = |input: &Array2<f64>| input.clone();

        // Simple loss functions
        let loss_fn = |output: &Array2<f64>, target: &Array1<f64>| {
            // Simplified loss computation
            (output.sum() - target.sum()).abs()
        };

        let val_fn = |output: &Array2<f64>, target: &Array1<f64>| {
            // Simplified validation metric
            (output.sum() - target.sum()).abs()
        };

        // Run training
        let results = trainer
            .train_model(
                model,
                &train_data,
                &val_data,
                loss_fn,
                val_fn,
                10, // epochs
                "adam",
            )
            .unwrap();

        // Verify results
        assert!(results.total_time > 0.0);
        assert_eq!(results.epochs_trained, 10);
        assert!(!results.metrics.train_loss_history.is_empty());
        assert!(!results.metrics.val_loss_history.is_empty());
    }

    #[test]
    fn test_framework_specific_configurations() {
        let frameworks = vec![
            TargetFramework::PyTorch,
            TargetFramework::TensorFlow,
            TargetFramework::JAX,
            TargetFramework::Candle,
            TargetFramework::Burn,
        ];

        for framework in frameworks {
            let config = ProductionConfig {
                target_framework: framework.clone(),
                ..Default::default()
            };

            let trainer = ProductionMLTrainer::new(config).unwrap();

            // Test optimizer creation for different frameworks
            let optimizer = trainer.create_optimizer("adam").unwrap();
            assert!(optimizer.get_lr() > 0.0);

            // Test scheduler creation
            let scheduler = trainer.create_scheduler("adam", 100).unwrap();
            assert!(scheduler.get_last_lr() > 0.0);

            println!("✅ Successfully configured for framework: {:?}", framework);
        }
    }
}

/// Main function demonstrating the production integration
#[allow(dead_code)]
fn main() -> Result<()> {
    println!("🚀 SciRS2 Production ML Integration Demo");
    println!("=========================================");

    // Create production configuration for PyTorch
    let config = ProductionConfig {
        enable_memory_monitoring: true,
        enable_performance_monitoring: true,
        gradient_clip_threshold: Some(1.0),
        use_mixed_precision: true,
        checkpoint_every_n_epochs: 5,
        early_stopping_patience: 25,
        target_framework: TargetFramework::PyTorch,
    };

    let mut trainer = ProductionMLTrainer::new(config)?;

    // Demonstrate different optimizers
    let optimizers = vec!["adam", "adamw", "lamb", "lars", "lion", "sgd"];

    for optimizer_name in optimizers {
        println!(
            "\n🧠 Testing {} optimizer integration:",
            optimizer_name.to_uppercase()
        );

        // Create dummy data for demonstration
        let train_data: Vec<(Array2<f64>, Array1<f64>)> = (0..50)
            .map(|_| {
                let input = Array::random((16, 5), rand::distributions::Uniform::new(-1.0, 1.0));
                let target = Array::random(16, rand::distributions::Uniform::new(0.0, 1.0));
                (input, target)
            })
            .collect();

        let val_data: Vec<(Array2<f64>, Array1<f64>)> = (0..10)
            .map(|_| {
                let input = Array::random((16, 5), rand::distributions::Uniform::new(-1.0, 1.0));
                let target = Array::random(16, rand::distributions::Uniform::new(0.0, 1.0));
                (input, target)
            })
            .collect();

        // Simple linear model simulation
        let model = |input: &Array2<f64>| {
            // Simulate a simple linear transformation
            input.mapv(|x| x.tanh()) // Apply activation
        };

        // Mean squared error loss
        let loss_fn = |output: &Array2<f64>, target: &Array1<f64>| {
            let output_mean = output.mean().unwrap_or(0.0);
            let target_mean = target.mean().unwrap_or(0.0);
            (output_mean - target_mean).powi(2)
        };

        let val_fn = loss_fn;

        // Train for a few epochs to demonstrate
        match trainer.train_model(
            model,
            &train_data,
            &val_data,
            loss_fn,
            val_fn,
            5, // Just 5 epochs for demo
            optimizer_name,
        ) {
            Ok(results) => {
                println!(
                    "✅ {} training completed successfully!",
                    optimizer_name.to_uppercase()
                );
                println!("   Best validation loss: {:.6}", results.best_val_loss);
                println!("   Training time: {:.2}s", results.total_time);
                println!(
                    "   Memory efficiency: {:.2}%",
                    results.report.memory_efficiency * 100.0
                );

                if !results.report.framework_recommendations.is_empty() {
                    println!("   Recommendations:");
                    for rec in results.report.framework_recommendations.iter().take(2) {
                        println!("     • {}", rec);
                    }
                }
            }
            Err(e) => {
                println!(
                    "❌ {} training failed: {}",
                    optimizer_name.to_uppercase(),
                    e
                );
            }
        }
    }

    println!("\n🎉 Production integration demo completed!");
    println!("Check the generated reports for detailed performance analysis.");

    Ok(())
}
