//! NAS controller for building and managing architectures

use crate::error::Result;
use crate::layers::Layer;
use crate::models::sequential::Sequential;
use crate::nas::{
    architecture_encoding::ArchitectureEncoding,
    search_space::{SearchSpaceConfig, LayerType, Architecture},
};
use std::sync::Arc;

/// Configuration for the NAS controller
#[derive(Debug, Clone)]
pub struct ControllerConfig {
    /// Input shape for the models
    pub input_shape: Vec<usize>,
    /// Number of output classes
    pub num_classes: usize,
    /// Whether to add a final softmax layer
    pub add_softmax: bool,
    /// Global seed for reproducibility
    pub seed: Option<u64>,
    /// Device to use (cpu, cuda, etc.)
    pub device: String,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            input_shape: vec![32, 32, 3], // Default to CIFAR-10 like input
            num_classes: 10,
            add_softmax: true,
            seed: None,
            device: "cpu".to_string(),
        }
    }
}

/// NAS Controller for building models from architecture encodings
pub struct NASController {
    config: ControllerConfig,
    search_space: SearchSpaceConfig,
}

impl NASController {
    /// Create a new NAS controller
    pub fn new(search_space: SearchSpaceConfig) -> Result<Self> {
        Ok(Self {
            config: ControllerConfig::default(),
            search_space,
        })
    }

    /// Create with custom configuration
    pub fn with_config(search_space: SearchSpaceConfig, config: ControllerConfig) -> Result<Self> {
        Ok(Self {
            config,
            search_space,
        })
    }

    /// Build a model from an architecture encoding
    pub fn build_model(&self, encoding: &Arc<dyn ArchitectureEncoding>) -> Result<Sequential<f32>> {
        let architecture = encoding.to_architecture()?;
        self.build_from_architecture(&architecture)
    }

    /// Build a model from an Architecture struct
    pub fn build_from_architecture(&self, architecture: &Architecture) -> Result<Sequential<f32>> {
        let mut model = Sequential::new();
        let mut current_shape = self.config.input_shape.clone();
        
        // Apply width and depth multipliers
        let effective_layers = self.apply_multipliers(
            &architecture.layers,
            architecture.width_multiplier,
            architecture.depth_multiplier
        )?;

        // Build layers
        for (i, layer_type) in effective_layers.iter().enumerate() {
            match layer_type {
                LayerType::Dense(units) => {
                    let input_size = current_shape.iter().product();
                    let mut rng = rand::thread_rng();
                    model.add_layer(crate::layers::Dense::new(input_size, *units, None, &mut rng)?);
                    current_shape = vec![*units];
                },
                LayerType::Dropout(rate) => {
                    let mut rng = rand::thread_rng();
                    model.add_layer(crate::layers::Dropout::new(*rate as f64, &mut rng)?);
                    // Dropout doesn't change shape
                },
                LayerType::BatchNorm => {
                    let features = current_shape.last().copied().unwrap_or(1);
                    let mut rng = rand::thread_rng();
                    model.add_layer(crate::layers::BatchNorm::new(features, 0.9, 1e-5, &mut rng)?);
                    // BatchNorm doesn't change shape
                },
                LayerType::Activation(name) => {
                    let size = current_shape.iter().product();
                    let mut rng = rand::thread_rng();
                    model.add_layer(crate::layers::Dense::new(size, size, Some(name.as_str()), &mut rng)?);
                    // Activation doesn't change shape
                },
                LayerType::Flatten => {
                    // For now, simulate flatten with a dense layer
                    let input_size: usize = current_shape.iter().product();
                    let mut rng = rand::thread_rng();
                    model.add_layer(crate::layers::Dense::new(input_size, input_size, None, &mut rng)?);
                    current_shape = vec![input_size];
                },
                _ => {
                    // Skip unsupported layer types for now
                    continue;
                }
            }
        }

        // Add final classification layer if needed
        if self.config.add_softmax {
            let input_size = current_shape.iter().product();
            let mut rng = rand::thread_rng();
            model.add_layer(crate::layers::Dense::new(
                input_size, 
                self.config.num_classes, 
                Some("softmax"), 
                &mut rng
            )?);
        }

        // Handle skip connections
        for (from, to) in &architecture.connections {
            // This would require a more sophisticated model builder
            // For now, we'll skip implementing skip connections in Sequential
            // In practice, would use a functional API
        }

        Ok(model)
    }

    /// Apply width and depth multipliers to layers
    fn apply_multipliers(
        &self,
        layers: &[LayerType],
        width_mult: f32,
        depth_mult: f32,
    ) -> Result<Vec<LayerType>> {
        let mut result = Vec::new();
        
        for layer in layers {
            // Apply depth multiplier (repeat layers)
            let repetitions = (depth_mult.max(0.1) as usize).max(1);
            
            for _ in 0..repetitions {
                // Apply width multiplier
                let modified_layer = match layer {
                    LayerType::Dense(units) => {
                        LayerType::Dense((*units as f32 * width_mult).round() as usize)
                    },
                    LayerType::Conv2D { filters, kernel_size, stride } => {
                        LayerType::Conv2D {
                            filters: (*filters as f32 * width_mult).round() as usize,
                            kernel_size: *kernel_size,
                            stride: *stride,
                        }
                    },
                    LayerType::Conv1D { filters, kernel_size, stride } => {
                        LayerType::Conv1D {
                            filters: (*filters as f32 * width_mult).round() as usize,
                            kernel_size: *kernel_size,
                            stride: *stride,
                        }
                    },
                    LayerType::LSTM { units, return_sequences } => {
                        LayerType::LSTM {
                            units: (*units as f32 * width_mult).round() as usize,
                            return_sequences: *return_sequences,
                        }
                    },
                    LayerType::GRU { units, return_sequences } => {
                        LayerType::GRU {
                            units: (*units as f32 * width_mult).round() as usize,
                            return_sequences: *return_sequences,
                        }
                    },
                    LayerType::Attention { num_heads, key_dim } => {
                        LayerType::Attention {
                            num_heads: *num_heads,
                            key_dim: (*key_dim as f32 * width_mult).round() as usize,
                        }
                    },
                    // Other layers remain unchanged
                    other => other.clone(),
                };
                
                result.push(modified_layer);
            }
        }
        
        Ok(result)
    }

    /// Create a layer from LayerType
    fn create_layer(&self, layer_type: &LayerType, input_shape: &[usize]) -> Result<Box<dyn Layer<f32>>> {
        use crate::layers::{Dense, Dropout, BatchNorm};
        
        match layer_type {
            LayerType::Dense(units) => {
                let input_size = input_shape.iter().product();
                let mut rng = rand::thread_rng();
                Ok(Box::new(Dense::new(input_size, *units, None, &mut rng)?))
            },
            LayerType::Dropout(rate) => {
                let mut rng = rand::thread_rng();
                Ok(Box::new(Dropout::new(*rate as f64, &mut rng)?))
            },
            LayerType::BatchNorm => {
                let features = input_shape.last().copied().unwrap_or(1);
                let mut rng = rand::thread_rng();
                Ok(Box::new(BatchNorm::new(features, 0.9, 1e-5, &mut rng)?))
            },
            LayerType::Activation(name) => {
                // Create a simple dense layer with 1:1 mapping and activation
                let size = input_shape.iter().product();
                let mut rng = rand::thread_rng();
                Ok(Box::new(Dense::new(size, size, Some(name.as_str()), &mut rng)?))
            },
            LayerType::Flatten => {
                // Create a reshape layer that flattens
                struct FlattenLayer;
                
                impl crate::layers::Layer<f32> for FlattenLayer {
                    fn forward(&self, input: &ndarray::ArrayD<f32>) -> Result<ndarray::ArrayD<f32>> {
                        let batch_size = input.shape()[0];
                        let flattened_size: usize = input.shape()[1..].iter().product();
                        Ok(input.clone().into_shape(vec![batch_size, flattened_size])?)
                    }
                    
                    fn backward(&self, _input: &ndarray::ArrayD<f32>, grad_output: &ndarray::ArrayD<f32>) -> Result<ndarray::ArrayD<f32>> {
                        Ok(grad_output.clone())
                    }
                    
                    fn update(&mut self, _learning_rate: f32) -> Result<()> {
                        Ok(())
                    }
                    
                    fn as_any(&self) -> &dyn std::any::Any {
                        self
                    }
                    
                    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                        self
                    }
                    
                    fn layer_type(&self) -> &str {
                        "Flatten"
                    }
                }
                
                Ok(Box::new(FlattenLayer))
            },
            // TODO: Implement other layer types
            _ => Err(crate::error::NeuralError::NotImplementedError(
                format!("Layer type {:?} not yet implemented", layer_type)
            )),
        }
    }

    /// Compute output shape after a layer
    fn compute_output_shape(&self, layer_type: &LayerType, input_shape: &[usize]) -> Result<Vec<usize>> {
        match layer_type {
            LayerType::Dense(units) => Ok(vec![*units]),
            LayerType::Conv2D { filters, kernel_size, stride } => {
                if input_shape.len() < 3 {
                    return Err(crate::error::NeuralError::InvalidArgument(
                        "Conv2D requires 3D input (H, W, C)".to_string()
                    ));
                }
                let h = (input_shape[0] - kernel_size.0) / stride.0 + 1;
                let w = (input_shape[1] - kernel_size.1) / stride.1 + 1;
                Ok(vec![h, w, *filters])
            },
            LayerType::MaxPool2D { pool_size, stride } |
            LayerType::AvgPool2D { pool_size, stride } => {
                if input_shape.len() < 3 {
                    return Err(crate::error::NeuralError::InvalidArgument(
                        "Pooling requires 3D input (H, W, C)".to_string()
                    ));
                }
                let h = (input_shape[0] - pool_size.0) / stride.0 + 1;
                let w = (input_shape[1] - pool_size.1) / stride.1 + 1;
                Ok(vec![h, w, input_shape[2]])
            },
            LayerType::GlobalMaxPool2D | LayerType::GlobalAvgPool2D => {
                if input_shape.len() < 3 {
                    return Err(crate::error::NeuralError::InvalidArgument(
                        "Global pooling requires 3D input (H, W, C)".to_string()
                    ));
                }
                Ok(vec![input_shape[2]])
            },
            LayerType::Flatten => {
                let total_size: usize = input_shape.iter().product();
                Ok(vec![total_size])
            },
            LayerType::Reshape(new_shape) => {
                let new_shape_usize: Vec<usize> = new_shape.iter()
                    .map(|&x| if x < 0 { 
                        // -1 means infer this dimension
                        let known_product: i32 = new_shape.iter()
                            .filter(|&&y| y > 0)
                            .product();
                        let total: i32 = input_shape.iter().map(|&x| x as i32).product();
                        (total / known_product) as usize
                    } else {
                        x as usize
                    })
                    .collect();
                Ok(new_shape_usize)
            },
            // For layers that don't change shape
            LayerType::Dropout(_) | LayerType::BatchNorm | LayerType::LayerNorm |
            LayerType::Activation(_) | LayerType::Residual => {
                Ok(input_shape.to_vec())
            },
            // TODO: Implement shape computation for other layer types
            _ => Ok(input_shape.to_vec()),
        }
    }

    /// Count parameters in a model
    pub fn count_parameters(&self, model: &Sequential<f32>) -> Result<usize> {
        // Simplified implementation
        // In practice, would iterate through model layers and count parameters
        Ok(1_000_000) // Placeholder
    }

    /// Estimate FLOPs for a model
    pub fn estimate_flops(&self, model: &Sequential<f32>, input_shape: &[usize]) -> Result<usize> {
        // Simplified implementation
        // In practice, would calculate FLOPs based on operations
        Ok(100_000_000) // Placeholder
    }

    /// Validate an architecture
    pub fn validate_architecture(&self, architecture: &Architecture) -> Result<()> {
        // Check if architecture is valid
        if architecture.layers.is_empty() {
            return Err(crate::error::NeuralError::InvalidArgument(
                "Architecture must have at least one layer".to_string()
            ));
        }

        // Check skip connections
        for (from, to) in &architecture.connections {
            if *from >= architecture.layers.len() || *to >= architecture.layers.len() {
                return Err(crate::error::NeuralError::InvalidArgument(
                    format!("Invalid skip connection: {} -> {}", from, to)
                ));
            }
            if from >= to {
                return Err(crate::error::NeuralError::InvalidArgument(
                    "Skip connections must be forward connections".to_string()
                ));
            }
        }

        // Validate multipliers
        if architecture.width_multiplier <= 0.0 || architecture.depth_multiplier <= 0.0 {
            return Err(crate::error::NeuralError::InvalidArgument(
                "Multipliers must be positive".to_string()
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::search_space::Architecture;

    #[test]
    fn test_controller_creation() {
        let search_space = SearchSpaceConfig::default();
        let controller = NASController::new(search_space).unwrap();
        assert_eq!(controller.config.num_classes, 10);
    }

    #[test]
    fn test_architecture_validation() {
        let search_space = SearchSpaceConfig::default();
        let controller = NASController::new(search_space).unwrap();

        // Valid architecture
        let valid_arch = Architecture {
            layers: vec![
                LayerType::Dense(128),
                LayerType::Activation("relu".to_string()),
                LayerType::Dense(10),
            ],
            connections: vec![],
            width_multiplier: 1.0,
            depth_multiplier: 1.0,
        };
        assert!(controller.validate_architecture(&valid_arch).is_ok());

        // Empty architecture
        let empty_arch = Architecture {
            layers: vec![],
            connections: vec![],
            width_multiplier: 1.0,
            depth_multiplier: 1.0,
        };
        assert!(controller.validate_architecture(&empty_arch).is_err());

        // Invalid skip connection
        let invalid_skip = Architecture {
            layers: vec![LayerType::Dense(128), LayerType::Dense(10)],
            connections: vec![(1, 0)], // Backward connection
            width_multiplier: 1.0,
            depth_multiplier: 1.0,
        };
        assert!(controller.validate_architecture(&invalid_skip).is_err());
    }

    #[test]
    fn test_multiplier_application() {
        let controller = NASController::new(SearchSpaceConfig::default()).unwrap();
        
        let layers = vec![
            LayerType::Dense(100),
            LayerType::Conv2D { filters: 32, kernel_size: (3, 3), stride: (1, 1) },
        ];

        let modified = controller.apply_multipliers(&layers, 2.0, 1.5).unwrap();
        
        // Width multiplier should double the units/filters
        match &modified[0] {
            LayerType::Dense(units) => assert_eq!(*units, 200),
            _ => panic!("Expected Dense layer"),
        }

        // Depth multiplier should create repetitions
        assert!(modified.len() > layers.len());
    }
}