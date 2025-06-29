//! Optimization configuration and auto-tuning system
//!
//! This module provides intelligent configuration systems that automatically
//! choose optimal settings for transformations based on data characteristics
//! and system resources.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, TransformError};
use crate::utils::ProcessingStrategy;

/// System resource information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResources {
    /// Available memory in MB
    pub memory_mb: usize,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Whether GPU is available
    pub has_gpu: bool,
    /// Whether SIMD instructions are available
    pub has_simd: bool,
    /// L3 cache size in KB (affects chunk sizes)
    pub l3_cache_kb: usize,
}

impl SystemResources {
    /// Detect system resources automatically
    pub fn detect() -> Self {
        SystemResources {
            memory_mb: Self::detect_memory_mb(),
            cpu_cores: num_cpus::get(),
            has_gpu: Self::detect_gpu(),
            has_simd: Self::detect_simd(),
            l3_cache_kb: Self::detect_l3_cache_kb(),
        }
    }

    /// Detect available memory
    fn detect_memory_mb() -> usize {
        // Simplified detection - in practice, use system APIs
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemAvailable:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb / 1024; // Convert to MB
                            }
                        }
                    }
                }
            }
        }
        
        // Fallback: assume 8GB
        8 * 1024
    }

    /// Detect GPU availability
    fn detect_gpu() -> bool {
        // Simplified detection
        #[cfg(feature = "gpu")]
        {
            // In practice, check for CUDA or OpenCL
            true
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Detect SIMD support
    fn detect_simd() -> bool {
        #[cfg(feature = "simd")]
        {
            true
        }
        #[cfg(not(feature = "simd"))]
        {
            false
        }
    }

    /// Detect L3 cache size
    fn detect_l3_cache_kb() -> usize {
        // Simplified - in practice, use CPUID or /sys/devices/system/cpu
        8 * 1024 // Assume 8MB L3 cache
    }

    /// Get conservative memory limit for transformations (80% of available)
    pub fn safe_memory_mb(&self) -> usize {
        (self.memory_mb as f64 * 0.8) as usize
    }

    /// Get optimal chunk size based on cache size
    pub fn optimal_chunk_size(&self, element_size: usize) -> usize {
        // Target 50% of L3 cache
        let target_bytes = (self.l3_cache_kb * 1024) / 2;
        (target_bytes / element_size).max(1000) // At least 1000 elements
    }
}

/// Data characteristics for optimization decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCharacteristics {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Data sparsity (0.0 = dense, 1.0 = all zeros)
    pub sparsity: f64,
    /// Data range (max - min)
    pub data_range: f64,
    /// Outlier ratio
    pub outlier_ratio: f64,
    /// Whether data has missing values
    pub has_missing: bool,
    /// Estimated memory footprint in MB
    pub memory_footprint_mb: f64,
    /// Data type size (e.g., 8 for f64)
    pub element_size: usize,
}

impl DataCharacteristics {
    /// Analyze data characteristics from array view
    pub fn analyze(data: &ndarray::ArrayView2<f64>) -> Result<Self> {
        let (n_samples, n_features) = data.dim();
        
        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty data".to_string()));
        }
        
        // Calculate sparsity
        let zeros = data.iter().filter(|&&x| x == 0.0).count();
        let sparsity = zeros as f64 / data.len() as f64;
        
        // Calculate data range
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        let mut finite_count = 0;
        let mut missing_count = 0;
        
        for &val in data.iter() {
            if val.is_finite() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
                finite_count += 1;
            } else {
                missing_count += 1;
            }
        }
        
        let data_range = if finite_count > 0 { max_val - min_val } else { 0.0 };
        let has_missing = missing_count > 0;
        
        // Estimate outlier ratio using IQR method (simplified)
        let outlier_ratio = if n_samples > 10 {
            let mut sample_values: Vec<f64> = data.iter()
                .filter(|&&x| x.is_finite())
                .take(1000) // Sample for efficiency
                .copied()
                .collect();
            
            if sample_values.len() >= 4 {
                sample_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = sample_values.len();
                let q1 = sample_values[n / 4];
                let q3 = sample_values[3 * n / 4];
                let iqr = q3 - q1;
                
                if iqr > 0.0 {
                    let lower_bound = q1 - 1.5 * iqr;
                    let upper_bound = q3 + 1.5 * iqr;
                    let outliers = sample_values.iter()
                        .filter(|&&x| x < lower_bound || x > upper_bound)
                        .count();
                    outliers as f64 / sample_values.len() as f64
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        let memory_footprint_mb = (n_samples * n_features * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);
        
        Ok(DataCharacteristics {
            n_samples,
            n_features,
            sparsity,
            data_range,
            outlier_ratio,
            has_missing,
            memory_footprint_mb,
            element_size: std::mem::size_of::<f64>(),
        })
    }

    /// Check if data is considered "large"
    pub fn is_large_dataset(&self) -> bool {
        self.n_samples > 100_000 || self.n_features > 10_000 || self.memory_footprint_mb > 1000.0
    }

    /// Check if data is considered "wide" (more features than samples)
    pub fn is_wide_dataset(&self) -> bool {
        self.n_features > self.n_samples
    }

    /// Check if data is sparse
    pub fn is_sparse(&self) -> bool {
        self.sparsity > 0.5
    }

    /// Check if data has significant outliers
    pub fn has_outliers(&self) -> bool {
        self.outlier_ratio > 0.05 // More than 5% outliers
    }
}

/// Optimization configuration for a specific transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Processing strategy to use
    pub processing_strategy: ProcessingStrategy,
    /// Memory limit in MB
    pub memory_limit_mb: usize,
    /// Whether to use robust statistics
    pub use_robust: bool,
    /// Whether to use parallel processing
    pub use_parallel: bool,
    /// Whether to use SIMD acceleration
    pub use_simd: bool,
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
    /// Chunk size for batch processing
    pub chunk_size: usize,
    /// Number of threads to use
    pub num_threads: usize,
    /// Additional algorithm-specific parameters
    pub algorithm_params: HashMap<String, f64>,
}

impl OptimizationConfig {
    /// Create optimization config for standardization
    pub fn for_standardization(
        data_chars: &DataCharacteristics,
        system: &SystemResources,
    ) -> Self {
        let use_robust = data_chars.has_outliers();
        let use_parallel = data_chars.n_samples > 10_000 && system.cpu_cores > 1;
        let use_simd = system.has_simd && data_chars.n_features > 100;
        let use_gpu = system.has_gpu && data_chars.memory_footprint_mb > 100.0;
        
        let processing_strategy = if data_chars.memory_footprint_mb > system.safe_memory_mb() as f64 {
            ProcessingStrategy::OutOfCore {
                chunk_size: system.optimal_chunk_size(data_chars.element_size)
            }
        } else if use_parallel {
            ProcessingStrategy::Parallel
        } else if use_simd {
            ProcessingStrategy::Simd
        } else {
            ProcessingStrategy::Standard
        };
        
        OptimizationConfig {
            processing_strategy,
            memory_limit_mb: system.safe_memory_mb(),
            use_robust,
            use_parallel,
            use_simd,
            use_gpu,
            chunk_size: system.optimal_chunk_size(data_chars.element_size),
            num_threads: if use_parallel { system.cpu_cores } else { 1 },
            algorithm_params: HashMap::new(),
        }
    }

    /// Create optimization config for PCA
    pub fn for_pca(
        data_chars: &DataCharacteristics,
        system: &SystemResources,
        n_components: usize,
    ) -> Self {
        let use_randomized = data_chars.is_large_dataset();
        let use_parallel = data_chars.n_samples > 1_000 && system.cpu_cores > 1;
        let use_gpu = system.has_gpu && data_chars.memory_footprint_mb > 500.0;
        
        // PCA memory requirements are higher due to covariance matrix
        let memory_multiplier = if data_chars.n_features > data_chars.n_samples { 3.0 } else { 2.0 };
        let estimated_memory = data_chars.memory_footprint_mb * memory_multiplier;
        
        let processing_strategy = if estimated_memory > system.safe_memory_mb() as f64 {
            ProcessingStrategy::OutOfCore {
                chunk_size: (system.safe_memory_mb() * 1024 * 1024) / (data_chars.n_features * data_chars.element_size)
            }
        } else if use_parallel {
            ProcessingStrategy::Parallel
        } else {
            ProcessingStrategy::Standard
        };
        
        let mut algorithm_params = HashMap::new();
        algorithm_params.insert("use_randomized".to_string(), if use_randomized { 1.0 } else { 0.0 });
        algorithm_params.insert("n_components".to_string(), n_components as f64);
        
        OptimizationConfig {
            processing_strategy,
            memory_limit_mb: system.safe_memory_mb(),
            use_robust: false, // PCA doesn't typically use robust statistics
            use_parallel,
            use_simd: system.has_simd,
            use_gpu,
            chunk_size: system.optimal_chunk_size(data_chars.element_size),
            num_threads: if use_parallel { system.cpu_cores } else { 1 },
            algorithm_params,
        }
    }

    /// Create optimization config for polynomial features
    pub fn for_polynomial_features(
        data_chars: &DataCharacteristics,
        system: &SystemResources,
        degree: usize,
    ) -> Result<Self> {
        // Polynomial features can explode in size
        let estimated_output_features = Self::estimate_polynomial_features(data_chars.n_features, degree)?;
        let estimated_memory = data_chars.n_samples as f64 * estimated_output_features as f64 * data_chars.element_size as f64 / (1024.0 * 1024.0);
        
        if estimated_memory > system.memory_mb as f64 * 0.9 {
            return Err(TransformError::MemoryError(
                format!(
                    "Polynomial features would require {:.1} MB, but only {} MB available",
                    estimated_memory, system.memory_mb
                )
            ));
        }
        
        let use_parallel = data_chars.n_samples > 1_000 && system.cpu_cores > 1;
        let use_simd = system.has_simd && estimated_output_features > 100;
        
        let processing_strategy = if estimated_memory > system.safe_memory_mb() as f64 {
            ProcessingStrategy::OutOfCore {
                chunk_size: (system.safe_memory_mb() * 1024 * 1024) / (estimated_output_features * data_chars.element_size)
            }
        } else if use_parallel {
            ProcessingStrategy::Parallel
        } else if use_simd {
            ProcessingStrategy::Simd
        } else {
            ProcessingStrategy::Standard
        };
        
        let mut algorithm_params = HashMap::new();
        algorithm_params.insert("degree".to_string(), degree as f64);
        algorithm_params.insert("estimated_output_features".to_string(), estimated_output_features as f64);
        
        Ok(OptimizationConfig {
            processing_strategy,
            memory_limit_mb: system.safe_memory_mb(),
            use_robust: false,
            use_parallel,
            use_simd,
            use_gpu: false, // Polynomial features typically don't benefit from GPU
            chunk_size: system.optimal_chunk_size(data_chars.element_size),
            num_threads: if use_parallel { system.cpu_cores } else { 1 },
            algorithm_params,
        })
    }

    /// Estimate number of polynomial features
    fn estimate_polynomial_features(n_features: usize, degree: usize) -> Result<usize> {
        if degree == 0 {
            return Err(TransformError::InvalidInput("Degree must be at least 1".to_string()));
        }
        
        let mut total_features = 1; // bias term
        
        for d in 1..=degree {
            // Multinomial coefficient: (n_features + d - 1)! / (d! * (n_features - 1)!)
            let mut coeff = 1;
            for i in 0..d {
                coeff = coeff * (n_features + d - 1 - i) / (i + 1);
                
                // Check for overflow
                if coeff > 1_000_000 {
                    return Err(TransformError::ComputationError(
                        "Too many polynomial features would be generated".to_string()
                    ));
                }
            }
            total_features += coeff;
        }
        
        Ok(total_features)
    }

    /// Get estimated execution time for this configuration
    pub fn estimated_execution_time(&self, data_chars: &DataCharacteristics) -> std::time::Duration {
        use std::time::Duration;
        
        let base_ops = data_chars.n_samples as u64 * data_chars.n_features as u64;
        
        let ops_per_second = match self.processing_strategy {
            ProcessingStrategy::Parallel => {
                1_000_000_000 * self.num_threads as u64 // 1 billion ops/second per thread
            },
            ProcessingStrategy::Simd => {
                2_000_000_000 // 2 billion ops/second with SIMD
            },
            ProcessingStrategy::OutOfCore { .. } => {
                100_000_000 // 100 million ops/second (I/O bound)
            },
            ProcessingStrategy::Standard => {
                500_000_000 // 500 million ops/second
            },
        };
        
        let time_ns = (base_ops * 1_000_000_000) / ops_per_second;
        Duration::from_nanos(time_ns.max(1000)) // At least 1 microsecond
    }
}

/// Auto-tuning system for optimization configurations
pub struct AutoTuner {
    /// System resources
    system: SystemResources,
    /// Performance history for different configurations
    performance_history: HashMap<String, Vec<PerformanceRecord>>,
}

/// Performance record for auto-tuning
#[derive(Debug, Clone)]
struct PerformanceRecord {
    config_hash: String,
    execution_time: std::time::Duration,
    memory_used_mb: f64,
    success: bool,
    data_characteristics: DataCharacteristics,
}

impl AutoTuner {
    /// Create a new auto-tuner
    pub fn new() -> Self {
        AutoTuner {
            system: SystemResources::detect(),
            performance_history: HashMap::new(),
        }
    }

    /// Get optimal configuration for a specific transformation
    pub fn optimize_for_transformation(
        &self,
        transformation: &str,
        data_chars: &DataCharacteristics,
        params: &HashMap<String, f64>,
    ) -> Result<OptimizationConfig> {
        match transformation {
            "standardization" => {
                Ok(OptimizationConfig::for_standardization(data_chars, &self.system))
            },
            "pca" => {
                let n_components = params.get("n_components").unwrap_or(&5.0) as &f64;
                Ok(OptimizationConfig::for_pca(data_chars, &self.system, *n_components as usize))
            },
            "polynomial" => {
                let degree = params.get("degree").unwrap_or(&2.0) as &f64;
                OptimizationConfig::for_polynomial_features(data_chars, &self.system, *degree as usize)
            },
            _ => {
                // Default configuration
                Ok(OptimizationConfig {
                    processing_strategy: if data_chars.is_large_dataset() {
                        ProcessingStrategy::Parallel
                    } else {
                        ProcessingStrategy::Standard
                    },
                    memory_limit_mb: self.system.safe_memory_mb(),
                    use_robust: data_chars.has_outliers(),
                    use_parallel: data_chars.n_samples > 10_000,
                    use_simd: self.system.has_simd,
                    use_gpu: self.system.has_gpu && data_chars.memory_footprint_mb > 100.0,
                    chunk_size: self.system.optimal_chunk_size(data_chars.element_size),
                    num_threads: self.system.cpu_cores,
                    algorithm_params: HashMap::new(),
                })
            }
        }
    }

    /// Record performance for learning
    pub fn record_performance(
        &mut self,
        transformation: &str,
        config: &OptimizationConfig,
        execution_time: std::time::Duration,
        memory_used_mb: f64,
        success: bool,
        data_chars: DataCharacteristics,
    ) {
        let config_hash = format!("{:?}", config); // Simplified hash
        
        let record = PerformanceRecord {
            config_hash: config_hash.clone(),
            execution_time,
            memory_used_mb,
            success,
            data_characteristics: data_chars,
        };
        
        self.performance_history
            .entry(transformation.to_string())
            .or_insert_with(Vec::new)
            .push(record);
        
        // Keep only recent records (last 100)
        let records = self.performance_history.get_mut(transformation).unwrap();
        if records.len() > 100 {
            records.remove(0);
        }
    }

    /// Get system resources
    pub fn system_resources(&self) -> &SystemResources {
        &self.system
    }

    /// Generate optimization report
    pub fn generate_report(&self, data_chars: &DataCharacteristics) -> OptimizationReport {
        let recommendations = vec![
            self.get_recommendation_for_transformation("standardization", data_chars),
            self.get_recommendation_for_transformation("pca", data_chars),
            self.get_recommendation_for_transformation("polynomial", data_chars),
        ];
        
        OptimizationReport {
            system_info: self.system.clone(),
            data_info: data_chars.clone(),
            recommendations,
            estimated_total_memory_mb: data_chars.memory_footprint_mb * 2.0, // Conservative estimate
        }
    }

    fn get_recommendation_for_transformation(
        &self,
        transformation: &str,
        data_chars: &DataCharacteristics,
    ) -> TransformationRecommendation {
        let config = self.optimize_for_transformation(transformation, data_chars, &HashMap::new())
            .unwrap_or_else(|_| OptimizationConfig {
                processing_strategy: ProcessingStrategy::Standard,
                memory_limit_mb: self.system.safe_memory_mb(),
                use_robust: false,
                use_parallel: false,
                use_simd: false,
                use_gpu: false,
                chunk_size: 1000,
                num_threads: 1,
                algorithm_params: HashMap::new(),
            });
        
        let estimated_time = config.estimated_execution_time(data_chars);
        
        TransformationRecommendation {
            transformation: transformation.to_string(),
            config,
            estimated_time,
            confidence: 0.8, // Placeholder
            reason: format!("Optimized for {} samples, {} features", data_chars.n_samples, data_chars.n_features),
        }
    }
}

/// Optimization report
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    /// System information
    pub system_info: SystemResources,
    /// Data characteristics
    pub data_info: DataCharacteristics,
    /// Recommendations for different transformations
    pub recommendations: Vec<TransformationRecommendation>,
    /// Estimated total memory usage
    pub estimated_total_memory_mb: f64,
}

/// Recommendation for a specific transformation
#[derive(Debug, Clone)]
pub struct TransformationRecommendation {
    /// Transformation name
    pub transformation: String,
    /// Recommended configuration
    pub config: OptimizationConfig,
    /// Estimated execution time
    pub estimated_time: std::time::Duration,
    /// Confidence in recommendation (0.0 to 1.0)
    pub confidence: f64,
    /// Human-readable reason
    pub reason: String,
}

impl OptimizationReport {
    /// Print a human-readable report
    pub fn print_report(&self) {
        println!("=== Optimization Report ===");
        println!("System Resources:");
        println!("  Memory: {} MB", self.system_info.memory_mb);
        println!("  CPU Cores: {}", self.system_info.cpu_cores);
        println!("  GPU Available: {}", self.system_info.has_gpu);
        println!("  SIMD Available: {}", self.system_info.has_simd);
        println!();
        
        println!("Data Characteristics:");
        println!("  Samples: {}", self.data_info.n_samples);
        println!("  Features: {}", self.data_info.n_features);
        println!("  Memory Footprint: {:.1} MB", self.data_info.memory_footprint_mb);
        println!("  Sparsity: {:.1}%", self.data_info.sparsity * 100.0);
        println!("  Has Outliers: {}", self.data_info.has_outliers());
        println!();
        
        println!("Recommendations:");
        for rec in &self.recommendations {
            println!("  {}:", rec.transformation);
            println!("    Strategy: {:?}", rec.config.processing_strategy);
            println!("    Estimated Time: {:.2}s", rec.estimated_time.as_secs_f64());
            println!("    Use Parallel: {}", rec.config.use_parallel);
            println!("    Use SIMD: {}", rec.config.use_simd);
            println!("    Use GPU: {}", rec.config.use_gpu);
            println!("    Reason: {}", rec.reason);
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_system_resources_detection() {
        let resources = SystemResources::detect();
        assert!(resources.cpu_cores > 0);
        assert!(resources.memory_mb > 0);
        assert!(resources.safe_memory_mb() < resources.memory_mb);
    }

    #[test]
    fn test_data_characteristics_analysis() {
        let data = Array2::from_shape_vec((100, 10), (0..1000).map(|x| x as f64).collect()).unwrap();
        let chars = DataCharacteristics::analyze(&data.view()).unwrap();
        
        assert_eq!(chars.n_samples, 100);
        assert_eq!(chars.n_features, 10);
        assert!(chars.memory_footprint_mb > 0.0);
        assert!(!chars.is_large_dataset());
    }

    #[test]
    fn test_optimization_config_for_standardization() {
        let data = Array2::ones((1000, 50));
        let chars = DataCharacteristics::analyze(&data.view()).unwrap();
        let system = SystemResources::detect();
        
        let config = OptimizationConfig::for_standardization(&chars, &system);
        assert!(config.memory_limit_mb > 0);
    }

    #[test]
    fn test_optimization_config_for_pca() {
        let data = Array2::ones((500, 20));
        let chars = DataCharacteristics::analyze(&data.view()).unwrap();
        let system = SystemResources::detect();
        
        let config = OptimizationConfig::for_pca(&chars, &system, 10);
        assert_eq!(config.algorithm_params.get("n_components"), Some(&10.0));
    }

    #[test]
    fn test_polynomial_features_estimation() {
        // Test polynomial feature estimation
        let result = OptimizationConfig::estimate_polynomial_features(5, 2);
        assert!(result.is_ok());
        
        // Should handle large degrees gracefully
        let result = OptimizationConfig::estimate_polynomial_features(100, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_auto_tuner() {
        let tuner = AutoTuner::new();
        let data = Array2::ones((100, 10));
        let chars = DataCharacteristics::analyze(&data.view()).unwrap();
        
        let config = tuner.optimize_for_transformation("standardization", &chars, &HashMap::new()).unwrap();
        assert!(config.memory_limit_mb > 0);
        
        let report = tuner.generate_report(&chars);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_large_dataset_detection() {
        let mut chars = DataCharacteristics {
            n_samples: 200_000,
            n_features: 1000,
            sparsity: 0.1,
            data_range: 100.0,
            outlier_ratio: 0.02,
            has_missing: false,
            memory_footprint_mb: 1500.0,
            element_size: 8,
        };
        
        assert!(chars.is_large_dataset());
        
        chars.n_samples = 1000;
        chars.memory_footprint_mb = 10.0;
        assert!(!chars.is_large_dataset());
    }
}
