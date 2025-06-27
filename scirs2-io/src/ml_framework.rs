//! Machine learning framework compatibility layer
//!
//! Provides conversion utilities and interfaces for seamless integration with
//! popular machine learning frameworks, enabling easy data exchange and model I/O.

use crate::error::{IoError, Result};
use crate::metadata::Metadata;
use ndarray::{Array1, Array2, Array3, Array4, ArrayD, ArrayView2, ArrayViewD, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Supported ML framework formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MLFramework {
    /// PyTorch tensor format
    PyTorch,
    /// TensorFlow SavedModel format
    TensorFlow,
    /// ONNX (Open Neural Network Exchange) format
    ONNX,
    /// Core ML format (Apple)
    CoreML,
    /// JAX format
    JAX,
    /// MXNet format
    MXNet,
    /// Hugging Face format
    HuggingFace,
    /// SafeTensors format
    SafeTensors,
}

/// Tensor metadata for ML frameworks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub name: Option<String>,
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub device: Option<String>,
    pub requires_grad: bool,
    pub is_parameter: bool,
}

/// Data types supported by ML frameworks
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    Float32,
    Float64,
    Float16,
    BFloat16,
    Int32,
    Int64,
    Int16,
    Int8,
    UInt8,
    Bool,
}

impl DataType {
    /// Get byte size of the data type
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Float64 | Self::Int64 => 8,
            Self::Float32 | Self::Int32 => 4,
            Self::Float16 | Self::BFloat16 | Self::Int16 => 2,
            Self::Int8 | Self::UInt8 | Self::Bool => 1,
        }
    }
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub framework: String,
    pub framework_version: Option<String>,
    pub model_name: Option<String>,
    pub model_version: Option<String>,
    pub architecture: Option<String>,
    pub input_shapes: HashMap<String, Vec<usize>>,
    pub output_shapes: HashMap<String, Vec<usize>>,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// ML tensor container
#[derive(Debug, Clone)]
pub struct MLTensor {
    pub data: ArrayD<f32>,
    pub metadata: TensorMetadata,
}

impl MLTensor {
    /// Create new ML tensor
    pub fn new(data: ArrayD<f32>, name: Option<String>) -> Self {
        let shape = data.shape().to_vec();
        Self {
            data,
            metadata: TensorMetadata {
                name,
                shape,
                dtype: DataType::Float32,
                device: None,
                requires_grad: false,
                is_parameter: false,
            },
        }
    }

    /// Convert to different data type
    pub fn to_dtype(&self, dtype: DataType) -> Result<Self> {
        // For simplicity, we'll just handle float conversions
        match dtype {
            DataType::Float32 => Ok(self.clone()),
            DataType::Float64 => {
                let data = self.data.mapv(|x| x as f64);
                Ok(Self {
                    data: data.mapv(|x| x as f32).into_dyn(),
                    metadata: TensorMetadata {
                        dtype,
                        ..self.metadata.clone()
                    },
                })
            }
            _ => Err(IoError::UnsupportedFormat(format!("Unsupported dtype conversion: {:?}", dtype))),
        }
    }
}

/// ML model container
pub struct MLModel {
    pub metadata: ModelMetadata,
    pub weights: HashMap<String, MLTensor>,
    pub config: HashMap<String, serde_json::Value>,
}

impl MLModel {
    /// Create new ML model
    pub fn new(framework: MLFramework) -> Self {
        Self {
            metadata: ModelMetadata {
                framework: format!("{:?}", framework),
                framework_version: None,
                model_name: None,
                model_version: None,
                architecture: None,
                input_shapes: HashMap::new(),
                output_shapes: HashMap::new(),
                parameters: HashMap::new(),
            },
            weights: HashMap::new(),
            config: HashMap::new(),
        }
    }

    /// Add weight tensor
    pub fn add_weight(&mut self, name: impl Into<String>, tensor: MLTensor) {
        self.weights.insert(name.into(), tensor);
    }

    /// Get weight tensor
    pub fn get_weight(&self, name: &str) -> Option<&MLTensor> {
        self.weights.get(name)
    }

    /// Save model to file
    pub fn save(&self, framework: MLFramework, path: impl AsRef<Path>) -> Result<()> {
        let converter = get_converter(framework);
        converter.save_model(self, path.as_ref())
    }

    /// Load model from file
    pub fn load(framework: MLFramework, path: impl AsRef<Path>) -> Result<Self> {
        let converter = get_converter(framework);
        converter.load_model(path.as_ref())
    }
}

/// Trait for ML framework converters
trait MLFrameworkConverter {
    fn save_model(&self, model: &MLModel, path: &Path) -> Result<()>;
    fn load_model(&self, path: &Path) -> Result<MLModel>;
    fn save_tensor(&self, tensor: &MLTensor, path: &Path) -> Result<()>;
    fn load_tensor(&self, path: &Path) -> Result<MLTensor>;
}

/// Get appropriate converter for framework
fn get_converter(framework: MLFramework) -> Box<dyn MLFrameworkConverter> {
    match framework {
        MLFramework::PyTorch => Box::new(PyTorchConverter),
        MLFramework::ONNX => Box::new(ONNXConverter),
        MLFramework::SafeTensors => Box::new(SafeTensorsConverter),
        _ => Box::new(GenericConverter),
    }
}

/// PyTorch format converter
struct PyTorchConverter;

impl MLFrameworkConverter for PyTorchConverter {
    fn save_model(&self, model: &MLModel, path: &Path) -> Result<()> {
        // Save in a PyTorch-compatible format (simplified)
        let mut state_dict = HashMap::new();
        
        for (name, tensor) in &model.weights {
            state_dict.insert(name.clone(), tensor_to_python_dict(tensor)?);
        }
        
        let model_dict = serde_json::json!({
            "state_dict": state_dict,
            "metadata": model.metadata,
            "config": model.config,
        });
        
        let file = File::create(path).map_err(|e| IoError::Io(e))?;
        serde_json::to_writer_pretty(file, &model_dict)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }

    fn load_model(&self, path: &Path) -> Result<MLModel> {
        let file = File::open(path).map_err(|e| IoError::Io(e))?;
        let model_dict: serde_json::Value = serde_json::from_reader(file)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;
        
        let mut model = MLModel::new(MLFramework::PyTorch);
        
        if let Some(metadata) = model_dict.get("metadata") {
            model.metadata = serde_json::from_value(metadata.clone())
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
        
        if let Some(config) = model_dict.get("config") {
            model.config = serde_json::from_value(config.clone())
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
        
        if let Some(state_dict) = model_dict.get("state_dict").and_then(|v| v.as_object()) {
            for (name, tensor_data) in state_dict {
                let tensor = python_dict_to_tensor(tensor_data)?;
                model.weights.insert(name.clone(), tensor);
            }
        }
        
        Ok(model)
    }

    fn save_tensor(&self, tensor: &MLTensor, path: &Path) -> Result<()> {
        let tensor_dict = tensor_to_python_dict(tensor)?;
        let file = File::create(path).map_err(|e| IoError::Io(e))?;
        serde_json::to_writer_pretty(file, &tensor_dict)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }

    fn load_tensor(&self, path: &Path) -> Result<MLTensor> {
        let file = File::open(path).map_err(|e| IoError::Io(e))?;
        let tensor_dict: serde_json::Value = serde_json::from_reader(file)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;
        python_dict_to_tensor(&tensor_dict)
    }
}

/// ONNX format converter
struct ONNXConverter;

impl MLFrameworkConverter for ONNXConverter {
    fn save_model(&self, model: &MLModel, path: &Path) -> Result<()> {
        // Simplified ONNX-like format
        let onnx_model = serde_json::json!({
            "format": "onnx",
            "version": "1.0",
            "graph": {
                "name": model.metadata.model_name,
                "inputs": model.metadata.input_shapes,
                "outputs": model.metadata.output_shapes,
                "initializers": model.weights.iter().map(|(name, tensor)| {
                    serde_json::json!({
                        "name": name,
                        "shape": tensor.metadata.shape,
                        "dtype": tensor.metadata.dtype,
                    })
                }).collect::<Vec<_>>(),
            },
            "metadata": model.metadata,
        });
        
        let file = File::create(path).map_err(|e| IoError::Io(e))?;
        serde_json::to_writer_pretty(file, &onnx_model)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }

    fn load_model(&self, _path: &Path) -> Result<MLModel> {
        Err(IoError::UnsupportedFormat("ONNX loading not fully implemented".to_string()))
    }

    fn save_tensor(&self, _tensor: &MLTensor, _path: &Path) -> Result<()> {
        Err(IoError::UnsupportedFormat("ONNX tensor saving not implemented".to_string()))
    }

    fn load_tensor(&self, _path: &Path) -> Result<MLTensor> {
        Err(IoError::UnsupportedFormat("ONNX tensor loading not implemented".to_string()))
    }
}

/// SafeTensors format converter
struct SafeTensorsConverter;

impl MLFrameworkConverter for SafeTensorsConverter {
    fn save_model(&self, model: &MLModel, path: &Path) -> Result<()> {
        // SafeTensors-like format
        let mut tensors = HashMap::new();
        
        for (name, tensor) in &model.weights {
            tensors.insert(name.clone(), serde_json::json!({
                "shape": tensor.metadata.shape,
                "dtype": format!("{:?}", tensor.metadata.dtype),
                "data": tensor.data.as_slice().unwrap().to_vec(),
            }));
        }
        
        let safetensors = serde_json::json!({
            "tensors": tensors,
            "metadata": model.metadata,
        });
        
        let file = File::create(path).map_err(|e| IoError::Io(e))?;
        serde_json::to_writer(file, &safetensors)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }

    fn load_model(&self, path: &Path) -> Result<MLModel> {
        let file = File::open(path).map_err(|e| IoError::Io(e))?;
        let safetensors: serde_json::Value = serde_json::from_reader(file)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;
        
        let mut model = MLModel::new(MLFramework::SafeTensors);
        
        if let Some(metadata) = safetensors.get("metadata") {
            model.metadata = serde_json::from_value(metadata.clone())
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }
        
        if let Some(tensors) = safetensors.get("tensors").and_then(|v| v.as_object()) {
            for (name, tensor_data) in tensors {
                let shape: Vec<usize> = serde_json::from_value(
                    tensor_data["shape"].clone()
                ).map_err(|e| IoError::SerializationError(e.to_string()))?;
                
                let data: Vec<f32> = serde_json::from_value(
                    tensor_data["data"].clone()
                ).map_err(|e| IoError::SerializationError(e.to_string()))?;
                
                let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
                    .map_err(|e| IoError::Other(e.to_string()))?;
                
                model.weights.insert(name.clone(), MLTensor::new(array, Some(name.clone())));
            }
        }
        
        Ok(model)
    }

    fn save_tensor(&self, tensor: &MLTensor, path: &Path) -> Result<()> {
        let tensor_data = serde_json::json!({
            "shape": tensor.metadata.shape,
            "dtype": format!("{:?}", tensor.metadata.dtype),
            "data": tensor.data.as_slice().unwrap().to_vec(),
        });
        
        let file = File::create(path).map_err(|e| IoError::Io(e))?;
        serde_json::to_writer(file, &tensor_data)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }

    fn load_tensor(&self, path: &Path) -> Result<MLTensor> {
        let file = File::open(path).map_err(|e| IoError::Io(e))?;
        let tensor_data: serde_json::Value = serde_json::from_reader(file)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;
        
        let shape: Vec<usize> = serde_json::from_value(
            tensor_data["shape"].clone()
        ).map_err(|e| IoError::SerializationError(e.to_string()))?;
        
        let data: Vec<f32> = serde_json::from_value(
            tensor_data["data"].clone()
        ).map_err(|e| IoError::SerializationError(e.to_string()))?;
        
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| IoError::Other(e.to_string()))?;
        
        Ok(MLTensor::new(array, None))
    }
}

/// Generic converter for unsupported frameworks
struct GenericConverter;

impl MLFrameworkConverter for GenericConverter {
    fn save_model(&self, _model: &MLModel, _path: &Path) -> Result<()> {
        Err(IoError::UnsupportedFormat("Framework not fully supported".to_string()))
    }

    fn load_model(&self, _path: &Path) -> Result<MLModel> {
        Err(IoError::UnsupportedFormat("Framework not fully supported".to_string()))
    }

    fn save_tensor(&self, _tensor: &MLTensor, _path: &Path) -> Result<()> {
        Err(IoError::UnsupportedFormat("Framework not fully supported".to_string()))
    }

    fn load_tensor(&self, _path: &Path) -> Result<MLTensor> {
        Err(IoError::UnsupportedFormat("Framework not fully supported".to_string()))
    }
}

/// Helper functions for tensor conversions
fn tensor_to_python_dict(tensor: &MLTensor) -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "data": tensor.data.as_slice().unwrap().to_vec(),
        "shape": tensor.metadata.shape,
        "dtype": format!("{:?}", tensor.metadata.dtype),
        "requires_grad": tensor.metadata.requires_grad,
    }))
}

fn python_dict_to_tensor(dict: &serde_json::Value) -> Result<MLTensor> {
    let shape: Vec<usize> = serde_json::from_value(dict["shape"].clone())
        .map_err(|e| IoError::SerializationError(e.to_string()))?;
    
    let data: Vec<f32> = serde_json::from_value(dict["data"].clone())
        .map_err(|e| IoError::SerializationError(e.to_string()))?;
    
    let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| IoError::Other(e.to_string()))?;
    
    let mut tensor = MLTensor::new(array, None);
    
    if let Some(requires_grad) = dict.get("requires_grad").and_then(|v| v.as_bool()) {
        tensor.metadata.requires_grad = requires_grad;
    }
    
    Ok(tensor)
}

/// Conversion utilities for common data formats
pub mod converters {
    use super::*;
    
    /// Convert ndarray to ML tensor
    pub fn from_ndarray<D>(array: ArrayView2<f32>, name: Option<String>) -> MLTensor {
        let shape = array.shape().to_vec();
        let data = array.to_owned().into_dyn();
        
        MLTensor {
            data,
            metadata: TensorMetadata {
                name,
                shape,
                dtype: DataType::Float32,
                device: None,
                requires_grad: false,
                is_parameter: false,
            },
        }
    }
    
    /// Convert ML tensor to ndarray
    pub fn to_ndarray2(tensor: &MLTensor) -> Result<Array2<f32>> {
        if tensor.metadata.shape.len() != 2 {
            return Err(IoError::Other("Tensor is not 2D".to_string()));
        }
        
        let shape = (tensor.metadata.shape[0], tensor.metadata.shape[1]);
        let data = tensor.data.as_slice().unwrap().to_vec();
        
        Array2::from_shape_vec(shape, data)
            .map_err(|e| IoError::Other(e.to_string()))
    }
    
    /// Convert between different ML frameworks
    pub fn convert_model(model: &MLModel, from: MLFramework, to: MLFramework) -> Result<MLModel> {
        // For now, just copy the model with new framework metadata
        let mut new_model = MLModel::new(to);
        new_model.metadata = model.metadata.clone();
        new_model.metadata.framework = format!("{:?}", to);
        new_model.weights = model.weights.clone();
        new_model.config = model.config.clone();
        
        Ok(new_model)
    }
}

/// Dataset utilities for ML frameworks
pub mod datasets {
    use super::*;
    
    /// ML dataset container
    pub struct MLDataset {
        pub features: Vec<MLTensor>,
        pub labels: Option<Vec<MLTensor>>,
        pub metadata: HashMap<String, serde_json::Value>,
    }
    
    impl MLDataset {
        /// Create new dataset
        pub fn new(features: Vec<MLTensor>) -> Self {
            Self {
                features,
                labels: None,
                metadata: HashMap::new(),
            }
        }
        
        /// Add labels
        pub fn with_labels(mut self, labels: Vec<MLTensor>) -> Self {
            self.labels = Some(labels);
            self
        }
        
        /// Get number of samples
        pub fn len(&self) -> usize {
            self.features.len()
        }
        
        /// Check if empty
        pub fn is_empty(&self) -> bool {
            self.features.is_empty()
        }
        
        /// Split into train/test sets
        pub fn train_test_split(&self, test_ratio: f32) -> (MLDataset, MLDataset) {
            let n = self.len();
            let test_size = (n as f32 * test_ratio) as usize;
            let train_size = n - test_size;
            
            let train_features = self.features[..train_size].to_vec();
            let test_features = self.features[train_size..].to_vec();
            
            let (train_labels, test_labels) = if let Some(labels) = &self.labels {
                (
                    Some(labels[..train_size].to_vec()),
                    Some(labels[train_size..].to_vec()),
                )
            } else {
                (None, None)
            };
            
            let train_dataset = MLDataset {
                features: train_features,
                labels: train_labels,
                metadata: self.metadata.clone(),
            };
            
            let test_dataset = MLDataset {
                features: test_features,
                labels: test_labels,
                metadata: self.metadata.clone(),
            };
            
            (train_dataset, test_dataset)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_ml_tensor_creation() {
        let data = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tensor = MLTensor::new(data, Some("test_tensor".to_string()));
        
        assert_eq!(tensor.metadata.shape, vec![2, 3]);
        assert_eq!(tensor.metadata.name, Some("test_tensor".to_string()));
    }

    #[test]
    fn test_model_save_load_safetensors() {
        let mut model = MLModel::new(MLFramework::SafeTensors);
        
        let data = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let tensor = MLTensor::new(data, Some("weight1".to_string()));
        model.add_weight("weight1", tensor);
        
        // Test in-memory round trip
        let temp_file = std::env::temp_dir().join("test_model.safetensors");
        model.save(MLFramework::SafeTensors, &temp_file).unwrap();
        let loaded_model = MLModel::load(MLFramework::SafeTensors, &temp_file).unwrap();
        
        assert_eq!(loaded_model.weights.len(), 1);
        assert!(loaded_model.get_weight("weight1").is_some());
        
        // Clean up
        let _ = std::fs::remove_file(temp_file);
    }

    #[test]
    fn test_ndarray_conversion() {
        let array = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let tensor = converters::from_ndarray(array.view(), Some("test".to_string()));
        
        assert_eq!(tensor.metadata.shape, vec![2, 2]);
        
        let array_back = converters::to_ndarray2(&tensor).unwrap();
        assert_eq!(array_back, array);
    }
}