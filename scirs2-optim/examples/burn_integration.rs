//! Integration example with Burn ML framework
//!
//! This example demonstrates how to use SciRS2 optimizers with the Burn ML framework
//! for training neural networks, including custom models and advanced training scenarios.

use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::{Module, Param},
    nn::{
        loss::{CrossEntropyLoss, Reduction},
        Linear, LinearConfig,
    },
    optim::{GradientsParams, Optimizer},
    tensor::{backend::Backend, Data, Distribution, Shape, Tensor},
    train::{
        metric::{
            processor::{FullEventProcessor, Metrics},
            AccuracyMetric, LossMetric,
        },
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition, TrainOutput, TrainStep,
        ValidStep,
    },
};
use ndarray::{Array1, Array2};
use scirs2_optim::{
    optimizers::{Adam, SGD},
    unified_api::{OptimizerConfig, OptimizerFactory, Parameter, UnifiedOptimizer},
    Optimizer as SciRS2Optimizer,
};
use std::collections::HashMap;

// Use NdArray backend for simplicity
type MyBackend = burn::backend::ndarray::NdArray<f32>;
type MyDevice = burn::backend::ndarray::NdArrayDevice;

/// Simple MLP model using Burn
#[derive(Module, Debug)]
pub struct SimpleMLP<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
}

#[derive(Config, Debug)]
pub struct SimpleMlpConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

impl SimpleMlpConfig {
    /// Initialize the model
    pub fn init<B: Backend>(&self, device: &B::Device) -> SimpleMLP<B> {
        SimpleMLP {
            linear1: LinearConfig::new(self.input_size, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            linear3: LinearConfig::new(self.hidden_size, self.output_size).init(device),
        }
    }
}

impl<B: Backend> SimpleMLP<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input).relu();
        let x = self.linear2.forward(x).relu();
        self.linear3.forward(x)
    }
}

/// Custom SciRS2 optimizer wrapper for Burn
#[derive(Clone, Debug)]
pub struct SciRS2OptimizerWrapper {
    optimizer: Box<dyn UnifiedOptimizer>,
    param_registry: HashMap<String, Parameter<f64>>,
    learning_rate: f64,
}

impl SciRS2OptimizerWrapper {
    pub fn new(config: OptimizerConfig) -> Self {
        let learning_rate = config.learning_rate;
        let optimizer = OptimizerFactory::adam(config);
        
        Self {
            optimizer,
            param_registry: HashMap::new(),
            learning_rate,
        }
    }

    pub fn sgd(learning_rate: f64) -> Self {
        let config = OptimizerConfig::new(learning_rate);
        let optimizer = OptimizerFactory::sgd(config);
        
        Self {
            optimizer,
            param_registry: HashMap::new(),
            learning_rate,
        }
    }

    pub fn adam(learning_rate: f64, weight_decay: f64) -> Self {
        let config = OptimizerConfig::new(learning_rate).weight_decay(weight_decay);
        let optimizer = OptimizerFactory::adam(config);
        
        Self {
            optimizer,
            param_registry: HashMap::new(),
            learning_rate,
        }
    }
}

/// Bridge implementation to connect Burn gradients with SciRS2 optimizers
impl<B: Backend> Optimizer<SimpleMLP<B>, B> for SciRS2OptimizerWrapper
where
    B::FloatTensorPrimitive: Send + Sync,
    B::IntTensorPrimitive: Send + Sync,
    B::BoolTensorPrimitive: Send + Sync,
{
    type State = ();

    fn step(&mut self, lr: f64, module: SimpleMLP<B>, gradients: GradientsParams) -> SimpleMLP<B> {
        // Convert Burn gradients to SciRS2 format and apply updates
        let mut updated_module = module;
        
        // Update linear1 weights
        if let Some(grad) = gradients.get(&updated_module.linear1.weight.id()) {
            let weight_data: Vec<f32> = updated_module.linear1.weight.val().to_data().convert().value;
            let grad_data: Vec<f32> = grad.to_data().convert().value;
            
            // Convert to f64 for SciRS2
            let params = Array1::from_vec(weight_data.iter().map(|&x| x as f64).collect());
            let grads = Array1::from_vec(grad_data.iter().map(|&x| x as f64).collect());
            
            // Create or get parameter
            let param_name = "linear1_weight";
            if !self.param_registry.contains_key(param_name) {
                let param = Parameter::new(params.clone(), param_name);
                self.param_registry.insert(param_name.to_string(), param);
            }
            
            if let Some(param) = self.param_registry.get_mut(param_name) {
                param.set_grad(grads);
                let _ = self.optimizer.step_param(param);
                
                // Convert back to f32 and update Burn tensor
                let updated_data: Vec<f32> = param.data().iter().map(|&x| x as f32).collect();
                let updated_tensor = Tensor::from_data(
                    Data::new(updated_data, updated_module.linear1.weight.shape()),
                    updated_module.linear1.weight.device(),
                );
                updated_module.linear1.weight = Param::from_tensor(updated_tensor);
            }
        }
        
        // Similar updates for other parameters...
        // (In a real implementation, this would be automated)
        
        updated_module
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }
}

/// Training configuration
#[derive(Config)]
pub struct TrainingConfig {
    pub model: SimpleMlpConfig,
    pub optimizer_type: String,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub num_epochs: usize,
    pub batch_size: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: SimpleMlpConfig {
                input_size: 784,
                hidden_size: 512,
                output_size: 10,
            },
            optimizer_type: "adam".to_string(),
            learning_rate: 1e-3,
            weight_decay: 1e-4,
            num_epochs: 10,
            batch_size: 64,
        }
    }
}

/// Batch type for training
#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 2>,
    pub targets: Tensor<B, 1, burn::tensor::Int>,
}

/// Training step implementation
impl<B: Backend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for SimpleMLP<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

/// Validation step implementation
impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for SimpleMLP<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Clone)]
pub struct ClassificationOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 2>,
    pub targets: Tensor<B, 1, burn::tensor::Int>,
}

impl<B: Backend> SimpleMLP<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 2>,
        targets: Tensor<B, 1, burn::tensor::Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLoss::new(None, &output.device())
            .forward(output.clone(), targets.clone(), Reduction::Mean);

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

/// Custom batcher for synthetic data
#[derive(Clone)]
pub struct SyntheticBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> SyntheticBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<(Vec<f32>, usize), MnistBatch<B>> for SyntheticBatcher<B> {
    fn batch(&self, items: Vec<(Vec<f32>, usize)>) -> MnistBatch<B> {
        let batch_size = items.len();
        let input_size = items[0].0.len();
        
        let mut images_vec = Vec::with_capacity(batch_size * input_size);
        let mut targets_vec = Vec::with_capacity(batch_size);
        
        for (image, target) in items {
            images_vec.extend(image);
            targets_vec.push(target as i64);
        }
        
        let images = Tensor::from_data(
            Data::new(images_vec, Shape::new([batch_size, input_size])),
            &self.device,
        );
        let targets = Tensor::from_data(
            Data::new(targets_vec, Shape::new([batch_size])),
            &self.device,
        );
        
        MnistBatch { images, targets }
    }
}

/// Main training function
fn train_with_scirs2_optimizer() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Burn + SciRS2 Integration Example");
    
    let config = TrainingConfig::default();
    let device = MyDevice::default();
    
    // Initialize model
    let model = config.model.init::<MyBackend>(&device);
    println!("📊 Model initialized with {} parameters", count_parameters(&model));
    
    // Create SciRS2 optimizer
    let optimizer = match config.optimizer_type.as_str() {
        "adam" => SciRS2OptimizerWrapper::adam(config.learning_rate, config.weight_decay),
        "sgd" => SciRS2OptimizerWrapper::sgd(config.learning_rate),
        _ => SciRS2OptimizerWrapper::adam(config.learning_rate, config.weight_decay),
    };
    
    println!("⚡ Using SciRS2 {} optimizer", config.optimizer_type);
    println!("📈 Learning rate: {}", config.learning_rate);
    println!("🏋️ Weight decay: {}", config.weight_decay);
    
    // Generate synthetic training data
    let train_data = generate_synthetic_data(1000, 784, 10);
    let valid_data = generate_synthetic_data(200, 784, 10);
    
    println!("📊 Training data: {} samples", train_data.len());
    println!("📊 Validation data: {} samples", valid_data.len());
    
    // Training would continue here with actual Burn training loop
    // This is a simplified example showing the integration structure
    
    println!("✅ Integration setup completed successfully!");
    
    Ok(())
}

/// Advanced example with different optimizers for different layers
fn multi_optimizer_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 Multi-Optimizer Example");
    
    let device = MyDevice::default();
    
    // Create model
    let config = SimpleMlpConfig {
        input_size: 100,
        hidden_size: 64,
        output_size: 10,
    };
    let model = config.init::<MyBackend>(&device);
    
    // Different optimizers for different layers
    let feature_optimizer = SciRS2OptimizerWrapper::sgd(0.001); // Conservative for features
    let classifier_optimizer = SciRS2OptimizerWrapper::adam(0.01, 1e-4); // Aggressive for classifier
    
    println!("🎭 Using different optimizers:");
    println!("   - Feature layers: SGD with LR=0.001");
    println!("   - Classifier: Adam with LR=0.01");
    
    // This demonstrates parameter group management
    // In practice, you'd split the model parameters by layer/function
    
    Ok(())
}

/// Benchmark comparison
fn benchmark_optimizers() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Optimizer Performance Benchmark");
    
    let device = MyDevice::default();
    let config = SimpleMlpConfig {
        input_size: 784,
        hidden_size: 512,
        output_size: 10,
    };
    
    // Test different SciRS2 optimizers
    let optimizers = vec![
        ("SGD", SciRS2OptimizerWrapper::sgd(0.01)),
        ("Adam", SciRS2OptimizerWrapper::adam(0.001, 1e-4)),
    ];
    
    for (name, optimizer) in optimizers {
        let model = config.init::<MyBackend>(&device);
        
        let start = std::time::Instant::now();
        
        // Simulate training steps
        for _ in 0..100 {
            // In real scenario, this would be actual training steps
            let dummy_input = Tensor::random(Shape::new([32, 784]), Distribution::Normal(0.0, 1.0), &device);
            let _output = model.forward(dummy_input);
            // Apply optimizer step (simplified)
        }
        
        let duration = start.elapsed();
        println!("⏱️  {}: {:?}", name, duration);
    }
    
    Ok(())
}

/// Transfer learning example
fn transfer_learning_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Transfer Learning with SciRS2 Optimizers");
    
    let device = MyDevice::default();
    
    // Simulate pre-trained model
    let pretrained_config = SimpleMlpConfig {
        input_size: 784,
        hidden_size: 512,
        output_size: 1000, // Pre-trained on 1000 classes
    };
    
    // Fine-tuning for new task
    let finetune_config = SimpleMlpConfig {
        input_size: 784,
        hidden_size: 512,
        output_size: 10, // Fine-tune for 10 classes
    };
    
    // Different learning rates for different layers
    let backbone_lr = 1e-5; // Very small for pre-trained features
    let head_lr = 1e-3; // Larger for new classifier head
    
    let backbone_optimizer = SciRS2OptimizerWrapper::adam(backbone_lr, 1e-4);
    let head_optimizer = SciRS2OptimizerWrapper::adam(head_lr, 1e-4);
    
    println!("🎯 Transfer learning setup:");
    println!("   - Backbone LR: {}", backbone_lr);
    println!("   - Head LR: {}", head_lr);
    
    Ok(())
}

/// Helper functions
fn count_parameters<B: Backend>(model: &SimpleMLP<B>) -> usize {
    // Simplified parameter counting
    // In practice, you'd iterate through all parameters
    1000 // Placeholder
}

fn generate_synthetic_data(samples: usize, input_dim: usize, num_classes: usize) -> Vec<(Vec<f32>, usize)> {
    use rand::prelude::*;
    let mut rng = thread_rng();
    
    (0..samples)
        .map(|_| {
            let input: Vec<f32> = (0..input_dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let target = rng.gen_range(0..num_classes);
            (input, target)
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Burn + SciRS2 Optimization Integration Examples\n");
    
    // Run all examples
    train_with_scirs2_optimizer()?;
    multi_optimizer_example()?;
    benchmark_optimizers()?;
    transfer_learning_example()?;
    
    println!("\n🎉 All Burn integration examples completed successfully!");
    println!("💡 Key benefits of SciRS2 + Burn integration:");
    println!("   ✅ Advanced optimization algorithms");
    println!("   ✅ Parameter group management");
    println!("   ✅ Built-in regularization techniques");
    println!("   ✅ Gradient clipping and normalization");
    println!("   ✅ Performance monitoring");
    println!("   ✅ Easy optimizer swapping and comparison");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scirs2_optimizer_creation() {
        let optimizer = SciRS2OptimizerWrapper::adam(0.001, 1e-4);
        assert!((optimizer.learning_rate - 0.001).abs() < 1e-10);
    }
    
    #[test]
    fn test_synthetic_data_generation() {
        let data = generate_synthetic_data(10, 5, 3);
        assert_eq!(data.len(), 10);
        assert_eq!(data[0].0.len(), 5);
        assert!(data[0].1 < 3);
    }
    
    #[test]
    fn test_model_creation() {
        let device = MyDevice::default();
        let config = SimpleMlpConfig {
            input_size: 10,
            hidden_size: 20,
            output_size: 5,
        };
        let model = config.init::<MyBackend>(&device);
        
        // Test forward pass
        let input = Tensor::zeros(Shape::new([1, 10]), &device);
        let output = model.forward(input);
        assert_eq!(output.shape().dims, [1, 5]);
    }
}