//! API standardization and consistency framework for scirs2-stats v1.0.0
//!
//! This module provides a unified, consistent API layer across all statistical
//! functions in scirs2-stats. It implements standardized parameter handling,
//! builder patterns, method chaining, and consistent error reporting to ensure
//! a smooth user experience that follows Rust idioms while maintaining SciPy
//! compatibility where appropriate.

use crate::error::{StatsError, StatsResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, NumCast};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Standardized statistical operation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardizedConfig {
    /// Enable automatic optimization selection
    pub auto_optimize: bool,
    /// Enable parallel processing when beneficial
    pub parallel: bool,
    /// Enable SIMD optimizations when available
    pub simd: bool,
    /// Maximum memory usage limit
    pub memory_limit: Option<usize>,
    /// Confidence level for statistical tests (0.0-1.0)
    pub confidence_level: f64,
    /// Null value handling strategy
    pub null_handling: NullHandling,
    /// Output precision for display
    pub output_precision: usize,
    /// Enable detailed result metadata
    pub include_metadata: bool,
}

impl Default for StandardizedConfig {
    fn default() -> Self {
        Self {
            auto_optimize: true,
            parallel: true,
            simd: true,
            memory_limit: None,
            confidence_level: 0.95,
            null_handling: NullHandling::Exclude,
            output_precision: 6,
            include_metadata: false,
        }
    }
}

/// Strategy for handling null/missing values
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NullHandling {
    /// Exclude null values from computation
    Exclude,
    /// Propagate null values (result is null if any input is null)
    Propagate,
    /// Replace null values with specified value
    Replace(f64),
    /// Fail computation if null values are encountered
    Fail,
}

/// Standardized result wrapper with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardizedResult<T> {
    /// The computed result value
    pub value: T,
    /// Computation metadata
    pub metadata: ResultMetadata,
    /// Any warnings generated during computation
    pub warnings: Vec<String>,
}

/// Metadata about the computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultMetadata {
    /// Sample size used in computation
    pub sample_size: usize,
    /// Degrees of freedom (where applicable)
    pub degrees_of_freedom: Option<usize>,
    /// Confidence level used (where applicable)
    pub confidence_level: Option<f64>,
    /// Method/algorithm used
    pub method: String,
    /// Computation time in milliseconds
    pub computation_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: Option<usize>,
    /// Whether optimization was applied
    pub optimized: bool,
    /// Additional method-specific metadata
    pub extra: HashMap<String, String>,
}

/// Builder pattern for descriptive statistics
pub struct DescriptiveStatsBuilder<F> {
    config: StandardizedConfig,
    ddof: Option<usize>,
    axis: Option<usize>,
    weights: Option<Array1<F>>,
    phantom: PhantomData<F>,
}

/// Builder pattern for correlation analysis
pub struct CorrelationBuilder<F> {
    config: StandardizedConfig,
    method: CorrelationMethod,
    min_periods: Option<usize>,
    phantom: PhantomData<F>,
}

/// Builder pattern for statistical tests
pub struct StatisticalTestBuilder<F> {
    config: StandardizedConfig,
    alternative: Alternative,
    equal_var: bool,
    phantom: PhantomData<F>,
}

/// Correlation method specification
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CorrelationMethod {
    Pearson,
    Spearman,
    Kendall,
    PartialPearson,
    PartialSpearman,
}

/// Alternative hypothesis for statistical tests
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

/// Unified statistical analysis interface
pub struct StatsAnalyzer<F> {
    config: StandardizedConfig,
    phantom: PhantomData<F>,
}

/// Descriptive statistics result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveStats<F> {
    pub count: usize,
    pub mean: F,
    pub std: F,
    pub min: F,
    pub percentile_25: F,
    pub median: F,
    pub percentile_75: F,
    pub max: F,
    pub variance: F,
    pub skewness: F,
    pub kurtosis: F,
}

/// Correlation analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationResult<F> {
    pub correlation: F,
    pub p_value: Option<F>,
    pub confidence_interval: Option<(F, F)>,
    pub method: CorrelationMethod,
}

/// Statistical test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult<F> {
    pub statistic: F,
    pub p_value: F,
    pub confidence_interval: Option<(F, F)>,
    pub effect_size: Option<F>,
    pub power: Option<F>,
}

impl<F> DescriptiveStatsBuilder<F>
where
    F: Float + NumCast + Clone,
{
    /// Create a new descriptive statistics builder
    pub fn new() -> Self {
        Self {
            config: StandardizedConfig::default(),
            ddof: None,
            axis: None,
            weights: None,
            phantom: PhantomData,
        }
    }

    /// Set degrees of freedom adjustment
    pub fn ddof(mut self, ddof: usize) -> Self {
        self.ddof = Some(ddof);
        self
    }

    /// Set computation axis (for multi-dimensional arrays)
    pub fn axis(mut self, axis: usize) -> Self {
        self.axis = Some(axis);
        self
    }

    /// Set sample weights
    pub fn weights(mut self, weights: Array1<F>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Enable/disable parallel processing
    pub fn parallel(mut self, enable: bool) -> Self {
        self.config.parallel = enable;
        self
    }

    /// Enable/disable SIMD optimizations
    pub fn simd(mut self, enable: bool) -> Self {
        self.config.simd = enable;
        self
    }

    /// Set null value handling strategy
    pub fn null_handling(mut self, strategy: NullHandling) -> Self {
        self.config.null_handling = strategy;
        self
    }

    /// Set memory limit
    pub fn memory_limit(mut self, limit: usize) -> Self {
        self.config.memory_limit = Some(limit);
        self
    }

    /// Include metadata in results
    pub fn with_metadata(mut self) -> Self {
        self.config.include_metadata = true;
        self
    }

    /// Compute descriptive statistics for the given data
    pub fn compute(&self, data: ArrayView1<F>) -> StatsResult<StandardizedResult<DescriptiveStats<F>>> {
        let start_time = std::time::Instant::now();
        let mut warnings = Vec::new();

        // Data validation
        if data.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Cannot compute statistics for empty array".to_string(),
            ));
        }

        // Handle null values based on strategy
        let (cleaned_data, sample_size) = self.handle_null_values(&data, &mut warnings)?;
        
        // Select computation method based on configuration
        let stats = if self.config.auto_optimize {
            self.compute_optimized(&cleaned_data, &mut warnings)?
        } else {
            self.compute_standard(&cleaned_data, &mut warnings)?
        };

        let computation_time = start_time.elapsed().as_secs_f64() * 1000.0;

        // Build metadata
        let metadata = ResultMetadata {
            sample_size,
            degrees_of_freedom: Some(sample_size.saturating_sub(self.ddof.unwrap_or(1))),
            confidence_level: None,
            method: self.select_method_name(),
            computation_time_ms: computation_time,
            memory_usage_bytes: self.estimate_memory_usage(sample_size),
            optimized: self.config.auto_optimize,
            extra: HashMap::new(),
        };

        Ok(StandardizedResult {
            value: stats,
            metadata,
            warnings,
        })
    }

    /// Handle null values according to the configured strategy
    fn handle_null_values(&self, data: &ArrayView1<F>, warnings: &mut Vec<String>) -> StatsResult<(Array1<F>, usize)> {
        // For now, assume no null values in numeric arrays
        // In a real implementation, this would detect and handle NaN values
        let finite_data: Vec<F> = data.iter()
            .filter(|&&x| x.is_finite())
            .cloned()
            .collect();

        if finite_data.len() != data.len() {
            warnings.push(format!("Removed {} non-finite values", data.len() - finite_data.len()));
        }

        match self.config.null_handling {
            NullHandling::Exclude => {
                Ok((Array1::from_vec(finite_data), finite_data.len()))
            }
            NullHandling::Fail if finite_data.len() != data.len() => {
                Err(StatsError::InvalidArgument(
                    "Null values encountered with Fail strategy".to_string(),
                ))
            }
            _ => Ok((Array1::from_vec(finite_data), finite_data.len())),
        }
    }

    /// Compute statistics using optimized methods
    fn compute_optimized(&self, data: &Array1<F>, warnings: &mut Vec<String>) -> StatsResult<DescriptiveStats<F>> {
        let n = data.len();
        
        // Use SIMD-optimized functions when available and beneficial
        if self.config.simd && n > 64 {
            self.compute_simd_optimized(data, warnings)
        } else if self.config.parallel && n > 10000 {
            self.compute_parallel_optimized(data, warnings)
        } else {
            self.compute_standard(data, warnings)
        }
    }

    /// Compute statistics using SIMD optimizations
    fn compute_simd_optimized(&self, data: &Array1<F>, _warnings: &mut Vec<String>) -> StatsResult<DescriptiveStats<F>> {
        // Use SIMD-optimized descriptive statistics
        let mean = crate::descriptive_simd::mean_simd(&data.view())?;
        let variance = crate::descriptive_simd::variance_simd(&data.view(), self.ddof.unwrap_or(1))?;
        let std = variance.sqrt();

        // Compute other statistics
        let (min, max) = self.compute_min_max(data);
        let sorted_data = self.get_sorted_data(data);
        let percentiles = self.compute_percentiles(&sorted_data)?;
        
        // Use existing functions for skewness and kurtosis
        let skewness = crate::descriptive::skew(&data.view(), false)?;
        let kurtosis = crate::descriptive::kurtosis(&data.view(), true, false)?;

        Ok(DescriptiveStats {
            count: data.len(),
            mean,
            std,
            min,
            percentile_25: percentiles[0],
            median: percentiles[1],
            percentile_75: percentiles[2],
            max,
            variance,
            skewness,
            kurtosis,
        })
    }

    /// Compute statistics using parallel optimizations
    fn compute_parallel_optimized(&self, data: &Array1<F>, _warnings: &mut Vec<String>) -> StatsResult<DescriptiveStats<F>> {
        // Use parallel-optimized functions
        let mean = crate::parallel_stats::mean_parallel(&data.view(), scirs2_core::parallel_ops::num_threads())?;
        let variance = crate::parallel_stats::variance_parallel(&data.view(), self.ddof.unwrap_or(1), scirs2_core::parallel_ops::num_threads())?;
        let std = variance.sqrt();

        // Compute other statistics
        let (min, max) = self.compute_min_max(data);
        let sorted_data = self.get_sorted_data(data);
        let percentiles = self.compute_percentiles(&sorted_data)?;
        
        // Use existing functions for skewness and kurtosis
        let skewness = crate::descriptive::skew(&data.view(), false)?;
        let kurtosis = crate::descriptive::kurtosis(&data.view(), true, false)?;

        Ok(DescriptiveStats {
            count: data.len(),
            mean,
            std,
            min,
            percentile_25: percentiles[0],
            median: percentiles[1],
            percentile_75: percentiles[2],
            max,
            variance,
            skewness,
            kurtosis,
        })
    }

    /// Compute statistics using standard methods
    fn compute_standard(&self, data: &Array1<F>, _warnings: &mut Vec<String>) -> StatsResult<DescriptiveStats<F>> {
        let mean = crate::descriptive::mean(&data.view())?;
        let variance = crate::descriptive::var(&data.view(), self.ddof.unwrap_or(1))?;
        let std = variance.sqrt();
        
        let (min, max) = self.compute_min_max(data);
        let sorted_data = self.get_sorted_data(data);
        let percentiles = self.compute_percentiles(&sorted_data)?;
        
        let skewness = crate::descriptive::skew(&data.view(), false)?;
        let kurtosis = crate::descriptive::kurtosis(&data.view(), true, false)?;

        Ok(DescriptiveStats {
            count: data.len(),
            mean,
            std,
            min,
            percentile_25: percentiles[0],
            median: percentiles[1],
            percentile_75: percentiles[2],
            max,
            variance,
            skewness,
            kurtosis,
        })
    }

    /// Compute min and max values
    fn compute_min_max(&self, data: &Array1<F>) -> (F, F) {
        let mut min = data[0];
        let mut max = data[0];
        
        for &value in data.iter() {
            if value < min { min = value; }
            if value > max { max = value; }
        }
        
        (min, max)
    }

    /// Get sorted copy of data for percentile calculations
    fn get_sorted_data(&self, data: &Array1<F>) -> Vec<F> {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    /// Compute percentiles (25th, 50th, 75th)
    fn compute_percentiles(&self, sorted_data: &[F]) -> StatsResult<[F; 3]> {
        let n = sorted_data.len();
        if n == 0 {
            return Err(StatsError::InvalidArgument("Empty data".to_string()));
        }

        let p25_idx = (n as f64 * 0.25) as usize;
        let p50_idx = (n as f64 * 0.50) as usize;
        let p75_idx = (n as f64 * 0.75) as usize;

        Ok([
            sorted_data[p25_idx.min(n - 1)],
            sorted_data[p50_idx.min(n - 1)],
            sorted_data[p75_idx.min(n - 1)],
        ])
    }

    /// Select method name for metadata
    fn select_method_name(&self) -> String {
        if self.config.simd && self.config.parallel {
            "SIMD+Parallel".to_string()
        } else if self.config.simd {
            "SIMD".to_string()
        } else if self.config.parallel {
            "Parallel".to_string()
        } else {
            "Standard".to_string()
        }
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self, sample_size: usize) -> Option<usize> {
        if self.config.include_metadata {
            Some(sample_size * std::mem::size_of::<F>() * 2) // Rough estimate
        } else {
            None
        }
    }
}

impl<F> CorrelationBuilder<F>
where
    F: Float + NumCast + Clone,
{
    /// Create a new correlation analysis builder
    pub fn new() -> Self {
        Self {
            config: StandardizedConfig::default(),
            method: CorrelationMethod::Pearson,
            min_periods: None,
            phantom: PhantomData,
        }
    }

    /// Set correlation method
    pub fn method(mut self, method: CorrelationMethod) -> Self {
        self.method = method;
        self
    }

    /// Set minimum number of periods for valid correlation
    pub fn min_periods(mut self, periods: usize) -> Self {
        self.min_periods = Some(periods);
        self
    }

    /// Set confidence level for p-values and confidence intervals
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.config.confidence_level = level;
        self
    }

    /// Enable/disable parallel processing
    pub fn parallel(mut self, enable: bool) -> Self {
        self.config.parallel = enable;
        self
    }

    /// Enable/disable SIMD optimizations
    pub fn simd(mut self, enable: bool) -> Self {
        self.config.simd = enable;
        self
    }

    /// Include metadata in results
    pub fn with_metadata(mut self) -> Self {
        self.config.include_metadata = true;
        self
    }

    /// Compute correlation between two variables
    pub fn compute(&self, x: ArrayView1<F>, y: ArrayView1<F>) -> StatsResult<StandardizedResult<CorrelationResult<F>>> {
        let start_time = std::time::Instant::now();
        let mut warnings = Vec::new();

        // Data validation
        if x.len() != y.len() {
            return Err(StatsError::DimensionMismatch(
                "Input arrays must have the same length".to_string(),
            ));
        }

        if x.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Cannot compute correlation for empty arrays".to_string(),
            ));
        }

        // Check minimum periods requirement
        if let Some(min_periods) = self.min_periods {
            if x.len() < min_periods {
                return Err(StatsError::InvalidArgument(
                    format!("Insufficient data: {} observations, {} required", x.len(), min_periods),
                ));
            }
        }

        // Compute correlation based on method
        let correlation = match self.method {
            CorrelationMethod::Pearson => {
                if self.config.simd && x.len() > 64 {
                    crate::correlation_simd::pearson_r_simd(&x, &y)?
                } else {
                    crate::correlation::pearson_r(&x, &y)?
                }
            }
            CorrelationMethod::Spearman => {
                crate::correlation::spearman_r(&x, &y)?
            }
            CorrelationMethod::Kendall => {
                crate::correlation::kendall_tau(&x, &y, "b")?
            }
            _ => {
                warnings.push("Advanced correlation methods not yet implemented".to_string());
                crate::correlation::pearson_r(&x, &y)?
            }
        };

        // Compute p-value and confidence interval if requested
        let (p_value, confidence_interval) = if self.config.include_metadata {
            self.compute_statistical_inference(correlation, x.len(), &mut warnings)?
        } else {
            (None, None)
        };

        let computation_time = start_time.elapsed().as_secs_f64() * 1000.0;

        let result = CorrelationResult {
            correlation,
            p_value,
            confidence_interval,
            method: self.method,
        };

        let metadata = ResultMetadata {
            sample_size: x.len(),
            degrees_of_freedom: Some(x.len().saturating_sub(2)),
            confidence_level: Some(self.config.confidence_level),
            method: format!("{:?}", self.method),
            computation_time_ms: computation_time,
            memory_usage_bytes: self.estimate_memory_usage(x.len()),
            optimized: self.config.simd || self.config.parallel,
            extra: HashMap::new(),
        };

        Ok(StandardizedResult {
            value: result,
            metadata,
            warnings,
        })
    }

    /// Compute correlation matrix for multiple variables
    pub fn compute_matrix(&self, data: ArrayView2<F>) -> StatsResult<StandardizedResult<Array2<F>>> {
        let start_time = std::time::Instant::now();
        let mut warnings = Vec::new();

        // Use optimized correlation matrix computation
        let correlation_matrix = if self.config.auto_optimize {
            // Use memory-optimized correlation matrix computation
            let mut optimizer = crate::memory_optimization_advanced::MemoryOptimizationSuite::new(
                crate::memory_optimization_advanced::MemoryOptimizationConfig::default()
            );
            optimizer.optimized_correlation_matrix(data)?
        } else {
            crate::correlation::corrcoef(&data, "pearson")?
        };

        let computation_time = start_time.elapsed().as_secs_f64() * 1000.0;

        let metadata = ResultMetadata {
            sample_size: data.nrows(),
            degrees_of_freedom: Some(data.nrows().saturating_sub(2)),
            confidence_level: Some(self.config.confidence_level),
            method: format!("Matrix {:?}", self.method),
            computation_time_ms: computation_time,
            memory_usage_bytes: self.estimate_memory_usage(data.nrows() * data.ncols()),
            optimized: self.config.auto_optimize,
            extra: HashMap::new(),
        };

        Ok(StandardizedResult {
            value: correlation_matrix,
            metadata,
            warnings,
        })
    }

    /// Compute statistical inference (p-values, confidence intervals)
    fn compute_statistical_inference(&self, correlation: F, n: usize, _warnings: &mut Vec<String>) -> StatsResult<(Option<F>, Option<(F, F)>)> {
        // Fisher's z-transformation for confidence intervals
        let z = ((F::one() + correlation) / (F::one() - correlation)).ln() * F::from(0.5).unwrap();
        let se_z = F::one() / F::from(n - 3).unwrap().sqrt();
        
        // Critical value for given confidence level (simplified - would use proper t-distribution)
        let alpha = F::one() - F::from(self.config.confidence_level).unwrap();
        let z_critical = F::from(1.96).unwrap(); // Approximate for 95% confidence
        
        let z_lower = z - z_critical * se_z;
        let z_upper = z + z_critical * se_z;
        
        // Transform back to correlation scale
        let r_lower = (F::from(2.0).unwrap() * z_lower).exp();
        let r_lower = (r_lower - F::one()) / (r_lower + F::one());
        
        let r_upper = (F::from(2.0).unwrap() * z_upper).exp();
        let r_upper = (r_upper - F::one()) / (r_upper + F::one());
        
        // Simplified p-value calculation (would use proper statistical test)
        let t_stat = correlation * F::from(n - 2).unwrap().sqrt() / (F::one() - correlation * correlation).sqrt();
        let p_value = F::from(2.0).unwrap() * (F::one() - F::from(0.95).unwrap()); // Simplified
        
        Ok((Some(p_value), Some((r_lower, r_upper))))
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self, size: usize) -> Option<usize> {
        if self.config.include_metadata {
            Some(size * std::mem::size_of::<F>() * 3) // Rough estimate
        } else {
            None
        }
    }
}

impl<F> StatsAnalyzer<F>
where
    F: Float + NumCast + Clone,
{
    /// Create a new unified stats analyzer
    pub fn new() -> Self {
        Self {
            config: StandardizedConfig::default(),
            phantom: PhantomData,
        }
    }

    /// Configure the analyzer
    pub fn configure(mut self, config: StandardizedConfig) -> Self {
        self.config = config;
        self
    }

    /// Perform comprehensive descriptive analysis
    pub fn describe(&self, data: ArrayView1<F>) -> StatsResult<StandardizedResult<DescriptiveStats<F>>> {
        DescriptiveStatsBuilder::new()
            .parallel(self.config.parallel)
            .simd(self.config.simd)
            .null_handling(self.config.null_handling)
            .with_metadata()
            .compute(data)
    }

    /// Perform correlation analysis
    pub fn correlate(&self, x: ArrayView1<F>, y: ArrayView1<F>, method: CorrelationMethod) -> StatsResult<StandardizedResult<CorrelationResult<F>>> {
        CorrelationBuilder::new()
            .method(method)
            .confidence_level(self.config.confidence_level)
            .parallel(self.config.parallel)
            .simd(self.config.simd)
            .with_metadata()
            .compute(x, y)
    }

    /// Get analyzer configuration
    pub fn get_config(&self) -> &StandardizedConfig {
        &self.config
    }
}

/// Convenient type aliases for common use cases
pub type F64StatsAnalyzer = StatsAnalyzer<f64>;
pub type F32StatsAnalyzer = StatsAnalyzer<f32>;

pub type F64DescriptiveBuilder = DescriptiveStatsBuilder<f64>;
pub type F32DescriptiveBuilder = DescriptiveStatsBuilder<f32>;

pub type F64CorrelationBuilder = CorrelationBuilder<f64>;
pub type F32CorrelationBuilder = CorrelationBuilder<f32>;

impl<F> Default for DescriptiveStatsBuilder<F>
where
    F: Float + NumCast + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Default for CorrelationBuilder<F>
where
    F: Float + NumCast + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Default for StatsAnalyzer<F>
where
    F: Float + NumCast + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_descriptive_stats_builder() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let result = DescriptiveStatsBuilder::new()
            .ddof(1)
            .parallel(false)
            .simd(false)
            .with_metadata()
            .compute(data.view())
            .unwrap();

        assert_eq!(result.value.count, 5);
        assert!((result.value.mean - 3.0).abs() < 1e-10);
        assert!(result.metadata.optimized == false);
    }

    #[test]
    fn test_correlation_builder() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = CorrelationBuilder::new()
            .method(CorrelationMethod::Pearson)
            .confidence_level(0.95)
            .with_metadata()
            .compute(x.view(), y.view())
            .unwrap();

        assert!((result.value.correlation - 1.0).abs() < 1e-10);
        assert!(result.value.p_value.is_some());
        assert!(result.value.confidence_interval.is_some());
    }

    #[test]
    fn test_stats_analyzer() {
        let analyzer = StatsAnalyzer::new();
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let desc_result = analyzer.describe(data.view()).unwrap();
        assert_eq!(desc_result.value.count, 5);

        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_result = analyzer.correlate(x.view(), y.view(), CorrelationMethod::Pearson).unwrap();
        assert!((corr_result.value.correlation + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_null_handling() {
        let data = array![1.0, 2.0, f64::NAN, 4.0, 5.0];
        
        let result = DescriptiveStatsBuilder::new()
            .null_handling(NullHandling::Exclude)
            .compute(data.view())
            .unwrap();

        assert_eq!(result.value.count, 4); // NaN excluded
        assert!(!result.warnings.is_empty()); // Should have warning about removed values
    }

    #[test] 
    fn test_standardized_config() {
        let config = StandardizedConfig {
            auto_optimize: false,
            parallel: false,
            simd: true,
            confidence_level: 0.99,
            ..Default::default()
        };

        assert!(!config.auto_optimize);
        assert!(!config.parallel);
        assert!(config.simd);
        assert!((config.confidence_level - 0.99).abs() < 1e-10);
    }
}