//! Advanced Candle integration example for scirs2-optim
//!
//! This example demonstrates deep integration between scirs2-optim and Candle,
//! showing how to use scirs2-optim optimizers with Candle tensors and neural networks.

use scirs2_optim::{
    optimizers::{Adam, SGD, Optimizer as OptimizerTrait},
    unified_api::{OptimizerConfig, OptimizerFactory, Parameter},
    schedulers::{ExponentialDecay, CosineAnnealing, Scheduler},
    error::Result as OptimResult,
};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Candle tensor compatibility layer
/// 
/// This module provides conversion utilities between Candle tensors and ndarray arrays
/// to enable seamless integration with scirs2-optim optimizers.
pub mod candle_bridge {
    use super::*;

    /// Convert Candle tensor to ndarray Array1 (placeholder implementation)
    pub fn tensor_to_array1(tensor_data: &[f32]) -> Array1<f64> {
        Array1::from_vec(tensor_data.iter().map(|&x| x as f64).collect())
    }

    /// Convert ndarray Array1 to Candle-compatible data (placeholder implementation)
    pub fn array1_to_tensor_data(array: &Array1<f64>) -> Vec<f32> {
        array.iter().map(|&x| x as f32).collect()
    }

    /// Convert Candle tensor to ndarray Array2 (placeholder implementation)
    pub fn tensor_to_array2(tensor_data: &[f32], shape: (usize, usize)) -> Array2<f64> {
        let data: Vec<f64> = tensor_data.iter().map(|&x| x as f64).collect();
        Array2::from_shape_vec(shape, data).expect("Invalid shape for tensor conversion")
    }

    /// Convert ndarray Array2 to Candle-compatible data (placeholder implementation)
    pub fn array2_to_tensor_data(array: &Array2<f64>) -> Vec<f32> {
        array.iter().map(|&x| x as f32).collect()
    }
}

/// Candle-compatible neural network layer
pub struct CandleLinearLayer {
    /// Weight parameters
    weights: Parameter<f64>,
    /// Bias parameters
    bias: Parameter<f64>,
    /// Layer dimensions
    input_dim: usize,
    output_dim: usize,
}

impl CandleLinearLayer {
    /// Create a new linear layer with Candle-style initialization
    pub fn new(input_dim: usize, output_dim: usize, name: &str) -> Self {
        // Xavier/Glorot initialization
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        
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
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Array2<f64>) -> OptimResult<Array2<f64>> {
        // input: (batch_size, input_dim)
        // weights: (input_dim, output_dim)
        // output: (batch_size, output_dim)
        
        let weights_2d = self.weights.data()
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| scirs2_optim::error::OptimError::DimensionMismatch(
                "Weight tensor dimension mismatch".to_string()
            ))?;
        
        let bias_1d = self.bias.data()
            .clone()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| scirs2_optim::error::OptimError::DimensionMismatch(
                "Bias tensor dimension mismatch".to_string()
            ))?;

        // Matrix multiplication: input @ weights
        let output = input.dot(&weights_2d);
        
        // Add bias to each row
        let output_with_bias = output + &bias_1d;
        
        Ok(output_with_bias)
    }

    /// Backward pass (simplified gradient computation)
    pub fn backward(
        &mut self,
        input: &Array2<f64>,
        grad_output: &Array2<f64>,
    ) -> OptimResult<Array2<f64>> {
        // Compute gradients
        // grad_weights = input.T @ grad_output
        let grad_weights = input.t().dot(grad_output);
        
        // grad_bias = sum(grad_output, axis=0)
        let grad_bias = grad_output.sum_axis(ndarray::Axis(0));
        
        // grad_input = grad_output @ weights.T
        let weights_2d = self.weights.data()
            .clone()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| scirs2_optim::error::OptimError::DimensionMismatch(
                "Weight tensor dimension mismatch".to_string()
            ))?;
        
        let grad_input = grad_output.dot(&weights_2d.t());
        
        // Set gradients in parameters
        self.weights.set_grad(grad_weights.into_dyn());
        self.bias.set_grad(grad_bias.into_dyn());
        
        Ok(grad_input)
    }

    /// Get mutable references to parameters
    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<f64>> {
        vec![&mut self.weights, &mut self.bias]
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.input_dim * self.output_dim + self.output_dim
    }
}

/// Candle-compatible neural network
pub struct CandleNeuralNetwork {
    /// Network layers
    layers: Vec<CandleLinearLayer>,
    /// Network architecture
    architecture: Vec<usize>,
}

impl CandleNeuralNetwork {
    /// Create a new neural network with specified architecture
    pub fn new(architecture: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        
        for i in 0..architecture.len() - 1 {
            let layer = CandleLinearLayer::new(
                architecture[i],
                architecture[i + 1],
                &format!("layer_{}", i),
            );
            layers.push(layer);
        }

        Self {
            layers,
            architecture,
        }
    }

    /// Forward pass through the network
    pub fn forward(&self, mut input: Array2<f64>) -> OptimResult<Array2<f64>> {
        for layer in &self.layers {
            input = layer.forward(&input)?;
            // Apply ReLU activation (except for the last layer)
            if layer as *const _ != self.layers.last().unwrap() as *const _ {
                input.mapv_inplace(|x| x.max(0.0));
            }
        }
        Ok(input)
    }

    /// Backward pass through the network
    pub fn backward(
        &mut self,
        input: &Array2<f64>,
        target: &Array2<f64>,
    ) -> OptimResult<f64> {
        // Forward pass to compute activations
        let mut activations = vec![input.clone()];
        let mut current = input.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            current = layer.forward(&current)?;
            if i < self.layers.len() - 1 {
                current.mapv_inplace(|x| x.max(0.0)); // ReLU
            }
            activations.push(current.clone());
        }

        // Compute loss (MSE)
        let output = activations.last().unwrap();
        let diff = output - target;
        let loss = (&diff * &diff).mean().unwrap_or(0.0);

        // Backward pass
        let mut grad_output = &diff * 2.0 / (output.nrows() as f64);

        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let layer_input = &activations[i];
            grad_output = layer.backward(layer_input, &grad_output)?;

            // Apply ReLU derivative (except for the first layer)
            if i > 0 {
                let activation = &activations[i];
                grad_output = grad_output * &activation.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            }
        }

        Ok(loss)
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
        self.layers.iter().map(|layer| layer.parameter_count()).sum()
    }

    /// Get network architecture
    pub fn architecture(&self) -> &[usize] {
        &self.architecture
    }
}

/// Candle training configuration
pub struct CandleTrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Optimizer type
    pub optimizer_type: String,
    /// Scheduler configuration
    pub scheduler_config: Option<SchedulerConfig>,
    /// Gradient clipping threshold
    pub grad_clip: Option<f64>,
}

/// Scheduler configuration
pub struct SchedulerConfig {
    pub scheduler_type: String,
    pub decay_rate: f64,
    pub step_size: Option<usize>,
    pub min_lr: Option<f64>,
}

/// Candle trainer for neural networks
pub struct CandleTrainer {
    /// Neural network
    network: CandleNeuralNetwork,
    /// Optimizer
    optimizer: Box<dyn scirs2_optim::unified_api::UnifiedOptimizer<f64>>,
    /// Learning rate scheduler
    scheduler: Option<Box<dyn Scheduler<f64>>>,
    /// Training configuration
    config: CandleTrainingConfig,
    /// Training history
    train_history: Vec<f64>,
}

impl CandleTrainer {
    /// Create a new trainer
    pub fn new(
        architecture: Vec<usize>,
        config: CandleTrainingConfig,
    ) -> OptimResult<Self> {
        let network = CandleNeuralNetwork::new(architecture);

        // Create optimizer
        let optimizer_config = OptimizerConfig::new(config.learning_rate)
            .weight_decay(0.0001);

        let optimizer: Box<dyn scirs2_optim::unified_api::UnifiedOptimizer<f64>> = 
            match config.optimizer_type.as_str() {
                "adam" => Box::new(OptimizerFactory::adam(optimizer_config)),
                "sgd" => Box::new(OptimizerFactory::sgd(optimizer_config)),
                _ => Box::new(OptimizerFactory::adam(optimizer_config)),
            };

        // Create scheduler if configured
        let scheduler: Option<Box<dyn Scheduler<f64>>> = if let Some(ref sched_config) = config.scheduler_config {
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
            train_history: Vec::new(),
        })
    }

    /// Train the network
    pub fn train(
        &mut self,
        train_data: &Array2<f64>,
        train_targets: &Array2<f64>,
    ) -> OptimResult<TrainingResults> {
        let start_time = std::time::Instant::now();
        let mut epoch_losses = Vec::new();
        let batch_size = self.config.batch_size;
        let n_samples = train_data.nrows();

        println!("🚀 Starting Candle Neural Network Training");
        println!("   Architecture: {:?}", self.network.architecture());
        println!("   Parameters: {}", self.network.parameter_count());
        println!("   Optimizer: {}", self.config.optimizer_type);
        println!("   Epochs: {}", self.config.epochs);
        println!("   Batch Size: {}", batch_size);

        for epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            // Mini-batch training
            for batch_start in (0..n_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_samples);
                
                let batch_data = train_data.slice(ndarray::s![batch_start..batch_end, ..]).to_owned();
                let batch_targets = train_targets.slice(ndarray::s![batch_start..batch_end, ..]).to_owned();

                // Forward and backward pass
                let loss = self.network.backward(&batch_data, &batch_targets)?;
                epoch_loss += loss;
                batch_count += 1;

                // Update parameters
                for param in self.network.parameters_mut() {
                    if let Some(grad) = param.grad() {
                        // Apply gradient clipping if configured
                        let clipped_grad = if let Some(clip_val) = self.config.grad_clip {
                            let grad_norm = grad.mapv(|x| x * x).sum().sqrt();
                            if grad_norm > clip_val {
                                grad * (clip_val / grad_norm)
                            } else {
                                grad.clone()
                            }
                        } else {
                            grad.clone()
                        };

                        // Convert to Array1 for optimizer compatibility
                        let grad_1d = if grad.ndim() == 1 {
                            clipped_grad.into_dimensionality::<ndarray::Ix1>()
                                .map_err(|_| scirs2_optim::error::OptimError::DimensionMismatch(
                                    "Gradient dimension mismatch".to_string()
                                ))?
                        } else {
                            // Flatten multi-dimensional gradients
                            Array1::from_vec(clipped_grad.iter().cloned().collect())
                        };

                        self.optimizer.step_param(param)?;
                    }
                }
            }

            epoch_loss /= batch_count as f64;
            epoch_losses.push(epoch_loss);
            self.train_history.push(epoch_loss);

            // Update learning rate scheduler
            if let Some(ref mut scheduler) = self.scheduler {
                scheduler.step();
            }

            // Print progress
            if (epoch + 1) % 10 == 0 || epoch == 0 {
                let current_lr = if let Some(ref scheduler) = self.scheduler {
                    scheduler.get_lr()
                } else {
                    self.config.learning_rate
                };
                
                println!("   Epoch {}/{}: Loss = {:.6}, LR = {:.6}", 
                    epoch + 1, self.config.epochs, epoch_loss, current_lr);
            }
        }

        let training_time = start_time.elapsed();
        let final_loss = epoch_losses.last().copied().unwrap_or(0.0);

        println!("✅ Training completed in {:.2}s", training_time.as_secs_f64());
        println!("   Final Loss: {:.6}", final_loss);

        Ok(TrainingResults {
            final_loss,
            epoch_losses,
            training_time,
            converged: final_loss < 0.01, // Simple convergence criterion
        })
    }

    /// Evaluate the network
    pub fn evaluate(
        &self,
        test_data: &Array2<f64>,
        test_targets: &Array2<f64>,
    ) -> OptimResult<EvaluationResults> {
        let predictions = self.network.forward(test_data)?;
        
        // Compute MSE loss
        let diff = &predictions - test_targets;
        let mse_loss = (&diff * &diff).mean().unwrap_or(0.0);
        
        // Compute MAE loss
        let mae_loss = diff.mapv(|x| x.abs()).mean().unwrap_or(0.0);
        
        // Compute accuracy for classification (if targets are one-hot)
        let accuracy = if test_targets.ncols() > 1 {
            let mut correct = 0;
            for i in 0..predictions.nrows() {
                let pred_class = predictions.row(i)
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                
                let true_class = test_targets.row(i)
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                if pred_class == true_class {
                    correct += 1;
                }
            }
            correct as f64 / predictions.nrows() as f64
        } else {
            // For regression, use R-squared
            let mean_target = test_targets.mean().unwrap_or(0.0);
            let ss_res = (&diff * &diff).sum();
            let ss_tot = test_targets.mapv(|x| (x - mean_target).powi(2)).sum();
            1.0 - (ss_res / ss_tot)
        };

        Ok(EvaluationResults {
            mse_loss,
            mae_loss,
            accuracy,
            predictions,
        })
    }

    /// Get training history
    pub fn training_history(&self) -> &[f64] {
        &self.train_history
    }

    /// Save model parameters
    pub fn save_model(&self, path: &str) -> OptimResult<()> {
        let mut model_data = HashMap::new();
        
        // This would serialize the network parameters
        // For now, just create a placeholder
        model_data.insert("architecture".to_string(), 
            serde_json::to_string(&self.network.architecture())?);
        model_data.insert("config".to_string(), 
            format!("{:?}", self.config.optimizer_type));

        let serialized = serde_json::to_string_pretty(&model_data)?;
        std::fs::write(path, serialized)?;
        
        println!("💾 Model saved to: {}", path);
        Ok(())
    }
}

/// Training results
#[derive(Debug, Clone)]
pub struct TrainingResults {
    pub final_loss: f64,
    pub epoch_losses: Vec<f64>,
    pub training_time: std::time::Duration,
    pub converged: bool,
}

/// Evaluation results
#[derive(Debug, Clone)]
pub struct EvaluationResults {
    pub mse_loss: f64,
    pub mae_loss: f64,
    pub accuracy: f64,
    pub predictions: Array2<f64>,
}

/// Generate synthetic training data for demonstration
pub fn generate_synthetic_data(
    n_samples: usize,
    input_dim: usize,
    output_dim: usize,
    noise_level: f64,
) -> (Array2<f64>, Array2<f64>) {
    let mut rng_state = 42u64; // Simple PRNG state
    
    // Generate random input data
    let mut input_data = Vec::new();
    for _i in 0..n_samples * input_dim {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let random_val = (rng_state % 10000) as f64 / 10000.0 - 0.5;
        input_data.push(random_val);
    }
    
    let input = Array2::from_shape_vec((n_samples, input_dim), input_data)
        .expect("Invalid input shape");

    // Generate synthetic targets (simple linear relationship + noise)
    let mut target_data = Vec::new();
    for i in 0..n_samples {
        for j in 0..output_dim {
            let mut target_val = 0.0;
            for k in 0..input_dim {
                target_val += input[[i, k]] * (k + j + 1) as f64 * 0.1;
            }
            
            // Add noise
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = (rng_state % 1000) as f64 / 1000.0 - 0.5;
            target_val += noise * noise_level;
            
            target_data.push(target_val);
        }
    }
    
    let targets = Array2::from_shape_vec((n_samples, output_dim), target_data)
        .expect("Invalid target shape");

    (input, targets)
}

fn main() -> OptimResult<()> {
    println!("🔥 Advanced Candle Integration with scirs2-optim");
    println!("================================================");

    // Generate synthetic dataset
    let (train_data, train_targets) = generate_synthetic_data(1000, 10, 3, 0.1);
    let (test_data, test_targets) = generate_synthetic_data(200, 10, 3, 0.1);

    println!("📊 Dataset Info:");
    println!("   Training samples: {}", train_data.nrows());
    println!("   Test samples: {}", test_data.nrows());
    println!("   Input dimension: {}", train_data.ncols());
    println!("   Output dimension: {}", train_targets.ncols());

    // Create training configuration
    let config = CandleTrainingConfig {
        learning_rate: 0.001,
        epochs: 100,
        batch_size: 32,
        optimizer_type: "adam".to_string(),
        scheduler_config: Some(SchedulerConfig {
            scheduler_type: "cosine".to_string(),
            decay_rate: 0.95,
            step_size: Some(20),
            min_lr: Some(0.0001),
        }),
        grad_clip: Some(1.0),
    };

    // Create and train the model
    let mut trainer = CandleTrainer::new(vec![10, 64, 32, 3], config)?;
    
    let training_results = trainer.train(&train_data, &train_targets)?;
    
    // Evaluate the model
    let eval_results = trainer.evaluate(&test_data, &test_targets)?;
    
    println!("\n📈 Training Results:");
    println!("   Final Training Loss: {:.6}", training_results.final_loss);
    println!("   Training Time: {:.2}s", training_results.training_time.as_secs_f64());
    println!("   Converged: {}", training_results.converged);
    
    println!("\n🎯 Evaluation Results:");
    println!("   Test MSE Loss: {:.6}", eval_results.mse_loss);
    println!("   Test MAE Loss: {:.6}", eval_results.mae_loss);
    println!("   Test R²: {:.6}", eval_results.accuracy);

    // Save the model
    trainer.save_model("candle_model.json")?;

    // Demonstrate different optimizers
    println!("\n🔄 Comparing Optimizers:");
    
    let optimizers = vec!["sgd", "adam"];
    for optimizer_name in optimizers {
        let config = CandleTrainingConfig {
            learning_rate: if optimizer_name == "sgd" { 0.01 } else { 0.001 },
            epochs: 50,
            batch_size: 32,
            optimizer_type: optimizer_name.to_string(),
            scheduler_config: None,
            grad_clip: Some(1.0),
        };

        let mut trainer = CandleTrainer::new(vec![10, 32, 3], config)?;
        let results = trainer.train(&train_data, &train_targets)?;
        
        println!("   {}: Final Loss = {:.6}, Time = {:.2}s", 
            optimizer_name.to_uppercase(), 
            results.final_loss, 
            results.training_time.as_secs_f64());
    }

    println!("\n✅ Candle integration example completed successfully!");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_linear_layer() {
        let mut layer = CandleLinearLayer::new(3, 2, "test");
        
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = layer.forward(&input).unwrap();
        
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(layer.parameter_count(), 8); // 3*2 + 2
    }

    #[test]
    fn test_candle_neural_network() {
        let network = CandleNeuralNetwork::new(vec![3, 4, 2]);
        
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = network.forward(input).unwrap();
        
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(network.parameter_count(), 3*4 + 4 + 4*2 + 2); // (3*4+4) + (4*2+2)
    }

    #[test]
    fn test_synthetic_data_generation() {
        let (input, targets) = generate_synthetic_data(100, 5, 2, 0.1);
        
        assert_eq!(input.shape(), &[100, 5]);
        assert_eq!(targets.shape(), &[100, 2]);
    }

    #[test]
    fn test_candle_trainer_creation() {
        let config = CandleTrainingConfig {
            learning_rate: 0.001,
            epochs: 10,
            batch_size: 16,
            optimizer_type: "adam".to_string(),
            scheduler_config: None,
            grad_clip: Some(1.0),
        };

        let trainer = CandleTrainer::new(vec![5, 3, 2], config);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_bridge_conversions() {
        let tensor_data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let array = candle_bridge::tensor_to_array1(&tensor_data);
        
        assert_eq!(array.len(), 4);
        assert_eq!(array[0], 1.0_f64);
        
        let converted_back = candle_bridge::array1_to_tensor_data(&array);
        assert_eq!(converted_back.len(), 4);
        assert_eq!(converted_back[0], 1.0_f32);
    }

    #[test]
    fn test_training_mini_batch() {
        let (train_data, train_targets) = generate_synthetic_data(50, 3, 2, 0.05);
        
        let config = CandleTrainingConfig {
            learning_rate: 0.01,
            epochs: 5,
            batch_size: 10,
            optimizer_type: "sgd".to_string(),
            scheduler_config: None,
            grad_clip: None,
        };

        let mut trainer = CandleTrainer::new(vec![3, 4, 2], config).unwrap();
        let results = trainer.train(&train_data, &train_targets).unwrap();
        
        assert_eq!(results.epoch_losses.len(), 5);
        assert!(results.final_loss >= 0.0);
    }
}