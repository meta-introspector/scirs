//! Data transformation module for SciRS2
//!
//! This module provides utilities for transforming data in ways that are useful
//! for machine learning and data analysis. The main functionalities include:
//!
//! - Data normalization and standardization
//! - Feature engineering
//! - Dimensionality reduction

#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]

/// Error handling for the transformation module
pub mod error;

/// Basic normalization methods for data
pub mod normalize;

/// Feature engineering techniques
pub mod features;

/// Dimensionality reduction algorithms
pub mod reduction;

/// Matrix decomposition techniques
pub mod decomposition;

/// Advanced scaling and transformation methods
pub mod scaling;

/// Missing value imputation utilities
pub mod impute;

/// Categorical data encoding utilities
pub mod encoding;

/// Feature selection utilities
pub mod selection;

/// Time series feature extraction
pub mod time_series;

/// Pipeline API for chaining transformations
pub mod pipeline;

/// SIMD-accelerated normalization operations
#[cfg(feature = "simd")]
pub mod normalize_simd;

/// SIMD-accelerated feature engineering operations
#[cfg(feature = "simd")]
pub mod features_simd;

/// SIMD-accelerated scaling operations
#[cfg(feature = "simd")]
pub mod scaling_simd;

/// Out-of-core processing for large datasets
pub mod out_of_core;

/// Streaming transformations for continuous data
pub mod streaming;

/// Text processing transformers
pub mod text;

/// Image processing transformers
pub mod image;

/// Graph embedding transformers
pub mod graph;

/// GPU-accelerated transformations
#[cfg(feature = "gpu")]
pub mod gpu;

/// Distributed processing for multi-node transformations
#[cfg(feature = "distributed")]
pub mod distributed;

/// Automated feature engineering with meta-learning
pub mod auto_feature_engineering;

/// Production monitoring with drift detection
#[cfg(feature = "monitoring")]
pub mod monitoring;

// Re-export important types and functions
pub use encoding::{BinaryEncoder, OneHotEncoder, OrdinalEncoder, TargetEncoder};
pub use error::{Result, TransformError};
pub use features::{
    binarize, discretize_equal_frequency, discretize_equal_width, log_transform, power_transform,
    PolynomialFeatures, PowerTransformer,
};
pub use impute::{
    DistanceMetric, ImputeStrategy, IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer,
    WeightingScheme,
};
pub use normalize::{normalize_array, normalize_vector, NormalizationMethod, Normalizer};
pub use reduction::{trustworthiness, TruncatedSVD, LDA, PCA, TSNE, UMAP, Isomap, LLE, SpectralEmbedding, AffinityMethod};
pub use decomposition::{NMF, DictionaryLearning};
pub use scaling::{MaxAbsScaler, QuantileTransformer};
pub use selection::{VarianceThreshold, RecursiveFeatureElimination, MutualInfoSelector};
pub use time_series::{FourierFeatures, LagFeatures, WaveletFeatures, TimeSeriesFeatures};
pub use pipeline::{Pipeline, ColumnTransformer, RemainderOption, Transformer, make_pipeline, make_column_transformer};

#[cfg(feature = "simd")]
pub use normalize_simd::{simd_minmax_normalize_1d, simd_zscore_normalize_1d, simd_l2_normalize_1d, simd_maxabs_normalize_1d, simd_normalize_array};

#[cfg(feature = "simd")]
pub use features_simd::{SimdPolynomialFeatures, simd_power_transform, simd_binarize};

#[cfg(feature = "simd")]
pub use scaling_simd::{SimdMaxAbsScaler, SimdRobustScaler, SimdStandardScaler};

pub use out_of_core::{OutOfCoreConfig, OutOfCoreTransformer, ChunkedArrayReader, ChunkedArrayWriter, 
                      OutOfCoreNormalizer, csv_chunks};
pub use streaming::{StreamingTransformer, StreamingStandardScaler, StreamingMinMaxScaler, 
                    StreamingQuantileTracker, WindowedStreamingTransformer};
pub use text::{CountVectorizer, TfidfVectorizer, HashingVectorizer, StreamingCountVectorizer};
pub use image::{PatchExtractor, HOGDescriptor, BlockNorm, ImageNormalizer, ImageNormMethod, 
                rgb_to_grayscale, resize_images};
pub use graph::{SpectralEmbedding, LaplacianType, DeepWalk, Node2Vec, GraphAutoencoder, 
                ActivationType, edge_list_to_adjacency, adjacency_to_edge_list};

// GPU acceleration exports
#[cfg(feature = "gpu")]
pub use gpu::{GpuPCA, GpuMatrixOps, GpuTSNE};

// Distributed processing exports
#[cfg(feature = "distributed")]
pub use distributed::{DistributedConfig, DistributedCoordinator, DistributedPCA, NodeInfo, 
                      PartitioningStrategy};

// Automated feature engineering exports
pub use auto_feature_engineering::{AutoFeatureEngineer, DatasetMetaFeatures, TransformationType, 
                                   TransformationConfig};

// Production monitoring exports
#[cfg(feature = "monitoring")]
pub use monitoring::{TransformationMonitor, DriftMethod, DriftDetectionResult, PerformanceMetrics, 
                     AlertConfig, AlertType};
