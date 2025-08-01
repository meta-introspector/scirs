//! Machine learning framework compatibility layer
//!
//! Provides conversion utilities and interfaces for seamless integration with
//! popular machine learning frameworks, enabling easy data exchange and model I/O.

#![allow(dead_code)]
#![allow(missing_docs)]

pub mod types;
pub mod converters;
pub mod datasets;
pub mod utils;
pub mod validation;
pub mod quantization;
pub mod optimization;
pub mod batch_processing;
pub mod serving;

// Re-export core types for backward compatibility
pub use types::{MLFramework, DataType, TensorMetadata, ModelMetadata, MLTensor, MLModel};
pub use converters::{
    MLFrameworkConverter, get_converter,
    PyTorchConverter, TensorFlowConverter, ONNXConverter, SafeTensorsConverter,
    JAXConverter, MXNetConverter, CoreMLConverter, HuggingFaceConverter,
};
pub use datasets::MLDataset;
pub use validation::{ModelValidator, ValidationConfig, ValidationReport, BatchValidator};
pub use quantization::{QuantizationMethod, QuantizedTensor, ModelQuantizer, QuantizedModel};
pub use optimization::{OptimizationTechnique, ModelOptimizer};
pub use batch_processing::{BatchProcessor, DataLoader};
pub use serving::{ModelServer, ServerConfig, ApiConfig, InferenceRequest, InferenceResponse, 
    ResponseStatus, ServerMetrics, HealthStatus, ModelInfo, LoadBalancer};

// Import required dependencies for the remaining modules
use crate::error::{IoError, Result};
use ndarray::{Array2, ArrayD, ArrayView2, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

// Additional imports for async support
#[cfg(feature = "async")]
use tokio::{
    fs,
    sync::{Arc, Mutex, RwLock},
    time::{sleep, Duration, Instant},
};

// Additional imports for binary operations
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

// Additional imports for networking
#[cfg(feature = "async")]
use std::collections::VecDeque;

// Include remaining modules from the parent file
// Note: The following modules are not available in the root and have been commented out
// pub use super::converters as common_converters;
// pub use super::serving;
// pub use super::model_hub;
// pub use super::pytorch_enhanced;
// pub use super::tensorflow_enhanced;
// pub use super::onnx_enhanced;
// pub use super::versioning;
// pub use super::error_handling;