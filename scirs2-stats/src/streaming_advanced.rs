//! Advanced streaming algorithms for memory-efficient large dataset processing
//!
//! This module provides sophisticated streaming algorithms that can process
//! datasets larger than available memory with constant or logarithmic memory
//! usage, while maintaining numerical accuracy and statistical validity.

use crate::error::{StatsError, StatsResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, NumCast, One, Zero};
use scirs2_core::{
    parallel_ops::*,
    simd_ops::SimdUnifiedOps,
    validation::*,
};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

/// Configuration for streaming algorithms
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Target memory usage limit in bytes
    pub memory_limit: usize,
    /// Chunk size for reading data
    pub chunk_size: usize,
    /// Maximum number of passes over data
    pub max_passes: usize,
    /// Accuracy vs memory tradeoff (0.0 = most accurate, 1.0 = least memory)
    pub accuracy_memory_tradeoff: f64,
    /// Enable adaptive memory management
    pub adaptive_memory: bool,
    /// Enable parallel streaming when beneficial
    pub parallel_streaming: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            memory_limit: 1_000_000_000, // 1GB default
            chunk_size: 65536,           // 64KB chunks
            max_passes: 3,
            accuracy_memory_tradeoff: 0.1, // Favor accuracy
            adaptive_memory: true,
            parallel_streaming: true,
        }
    }
}

/// Streaming statistics accumulator with Welford's algorithm
#[derive(Debug, Clone)]
pub struct StreamingStatsAccumulator<F> {
    /// Count of processed elements
    pub count: usize,
    /// Running mean
    pub mean: F,
    /// Running M2 (sum of squared differences from mean)
    pub m2: F,
    /// Running M3 (sum of cubed differences from mean)
    pub m3: F,
    /// Running M4 (sum of fourth power differences from mean)
    pub m4: F,
    /// Minimum value seen
    pub min_value: F,
    /// Maximum value seen
    pub max_value: F,
    /// Sum for numerical verification
    pub sum: F,
}

impl<F> Default for StreamingStatsAccumulator<F>
where
    F: Float + Zero + Copy,
{
    fn default() -> Self {
        Self {
            count: 0,
            mean: F::zero(),
            m2: F::zero(),
            m3: F::zero(),
            m4: F::zero(),
            min_value: F::infinity(),
            max_value: F::neg_infinity(),
            sum: F::zero(),
        }
    }
}

impl<F> StreamingStatsAccumulator<F>
where
    F: Float + NumCast + Zero + One + Copy,
{
    /// Create new streaming accumulator
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a single value to the accumulator
    pub fn update(&mut self, value: F) {
        if !value.is_finite() {
            return; // Skip non-finite values
        }

        self.count += 1;
        self.sum = self.sum + value;

        // Update min/max
        if value < self.min_value {
            self.min_value = value;
        }
        if value > self.max_value {
            self.max_value = value;
        }

        // Welford's algorithm for moments
        let delta = value - self.mean;
        self.mean = self.mean + delta / F::from(self.count).unwrap();
        let delta2 = value - self.mean;

        self.m2 = self.m2 + delta * delta2;

        if self.count >= 3 {
            let n = F::from(self.count).unwrap();
            let delta_n = delta / n;
            let delta_n2 = delta_n * delta_n;
            let term1 = delta * delta2 * delta_n * (n - F::one());

            self.m3 = self.m3 + term1 * delta_n * (n - F::from(2).unwrap()) 
                - F::from(3).unwrap() * delta_n * self.m2;

            if self.count >= 4 {
                let term2 = term1 * delta_n2 * (n * n - F::from(3).unwrap() * n + F::from(3).unwrap());
                let term3 = F::from(6).unwrap() * delta_n2 * self.m2;
                let term4 = F::from(4).unwrap() * delta_n * self.m3;

                self.m4 = self.m4 + term2 + term3 - term4;
            }
        }
    }

    /// Add multiple values from an array view
    pub fn update_batch(&mut self, values: &ArrayView1<F>) {
        for &value in values.iter() {
            self.update(value);
        }
    }

    /// Merge with another accumulator for parallel streaming
    pub fn merge(&mut self, other: &StreamingStatsAccumulator<F>) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }

        let combined_count = self.count + other.count;
        let n1 = F::from(self.count).unwrap();
        let n2 = F::from(other.count).unwrap();
        let n = F::from(combined_count).unwrap();

        let delta = other.mean - self.mean;
        let combined_mean = (n1 * self.mean + n2 * other.mean) / n;

        // Combine M2 (variance)
        let combined_m2 = self.m2 + other.m2 + delta * delta * n1 * n2 / n;

        // Combine M3 (skewness) 
        let combined_m3 = self.m3 + other.m3
            + delta * delta * delta * n1 * n2 * (n1 - n2) / (n * n)
            + F::from(3).unwrap() * delta * (n1 * other.m2 - n2 * self.m2) / n;

        // Combine M4 (kurtosis)
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta3 * delta;
        let combined_m4 = self.m4 + other.m4
            + delta4 * n1 * n2 * (n1 * n1 - n1 * n2 + n2 * n2) / (n * n * n)
            + F::from(6).unwrap() * delta2 * (n1 * n1 * other.m2 + n2 * n2 * self.m2) / (n * n)
            + F::from(4).unwrap() * delta * (n1 * other.m3 - n2 * self.m3) / n;

        self.count = combined_count;
        self.mean = combined_mean;
        self.m2 = combined_m2;
        self.m3 = combined_m3;
        self.m4 = combined_m4;
        self.sum = self.sum + other.sum;
        self.min_value = self.min_value.min(other.min_value);
        self.max_value = self.max_value.max(other.max_value);
    }

    /// Get current mean
    pub fn mean(&self) -> F {
        self.mean
    }

    /// Get current variance (population)
    pub fn variance(&self) -> F {
        if self.count < 2 {
            F::zero()
        } else {
            self.m2 / F::from(self.count).unwrap()
        }
    }

    /// Get current sample variance
    pub fn sample_variance(&self) -> F {
        if self.count < 2 {
            F::zero()
        } else {
            self.m2 / F::from(self.count - 1).unwrap()
        }
    }

    /// Get current standard deviation
    pub fn std_dev(&self) -> F {
        self.variance().sqrt()
    }

    /// Get current sample standard deviation
    pub fn sample_std_dev(&self) -> F {
        self.sample_variance().sqrt()
    }

    /// Get current skewness
    pub fn skewness(&self) -> F {
        if self.count < 3 || self.m2 == F::zero() {
            F::zero()
        } else {
            let n = F::from(self.count).unwrap();
            let variance = self.m2 / n;
            self.m3 / (n * variance * variance.sqrt())
        }
    }

    /// Get current kurtosis (excess kurtosis)
    pub fn kurtosis(&self) -> F {
        if self.count < 4 || self.m2 == F::zero() {
            F::zero()
        } else {
            let n = F::from(self.count).unwrap();
            let variance = self.m2 / n;
            (self.m4 / (n * variance * variance)) - F::from(3).unwrap()
        }
    }

    /// Merge with another accumulator
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }

        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        let combined_mean = (F::from(self.count).unwrap() * self.mean + 
                           F::from(other.count).unwrap() * other.mean) / 
                           F::from(combined_count).unwrap();

        // Update moments using parallel algorithm
        let n1 = F::from(self.count).unwrap();
        let n2 = F::from(other.count).unwrap();
        let n = n1 + n2;

        let combined_m2 = self.m2 + other.m2 + delta * delta * n1 * n2 / n;
        let combined_m3 = self.m3 + other.m3 + 
            delta * delta * delta * n1 * n2 * (n1 - n2) / (n * n) +
            F::from(3).unwrap() * delta * (n1 * other.m2 - n2 * self.m2) / n;

        let delta_sq = delta * delta;
        let combined_m4 = self.m4 + other.m4 + 
            delta_sq * delta_sq * n1 * n2 * (n1 * n1 - n1 * n2 + n2 * n2) / (n * n * n) +
            F::from(6).unwrap() * delta_sq * (n1 * n1 * other.m2 + n2 * n2 * self.m2) / (n * n) +
            F::from(4).unwrap() * delta * (n1 * other.m3 - n2 * self.m3) / n;

        self.count = combined_count;
        self.mean = combined_mean;
        self.m2 = combined_m2;
        self.m3 = combined_m3;
        self.m4 = combined_m4;
        self.sum = self.sum + other.sum;
        self.min_value = self.min_value.min(other.min_value);
        self.max_value = self.max_value.max(other.max_value);
    }
}

/// Streaming quantile estimator using P² algorithm
#[derive(Debug, Clone)]
pub struct StreamingQuantileEstimator<F> {
    /// Target quantiles to track
    quantiles: Vec<F>,
    /// Markers (heights)
    markers: Vec<F>,
    /// Marker positions
    positions: Vec<F>,
    /// Desired marker positions
    desired_positions: Vec<F>,
    /// Increments for desired positions
    position_increments: Vec<F>,
    /// Count of observations
    count: usize,
}

impl<F> StreamingQuantileEstimator<F>
where
    F: Float + NumCast + Zero + One + Copy,
{
    /// Create new quantile estimator for given quantiles
    pub fn new(quantiles: Vec<F>) -> Self {
        let n_quantiles = quantiles.len();
        let n_markers = n_quantiles + 4; // P² algorithm requires n+4 markers

        let mut desired_positions = vec![F::zero(); n_markers];
        let mut position_increments = vec![F::zero(); n_markers];

        // Initialize desired positions and increments
        desired_positions[0] = F::one();
        desired_positions[n_markers - 1] = F::one();
        for i in 1..n_markers - 1 {
            if i - 1 < quantiles.len() {
                desired_positions[i] = F::one() + F::from(2).unwrap() * quantiles[i - 1];
            }
        }

        position_increments[0] = F::zero();
        position_increments[n_markers - 1] = F::zero();
        for i in 1..n_markers - 1 {
            if i - 1 < quantiles.len() {
                position_increments[i] = quantiles[i - 1];
            }
        }

        Self {
            quantiles,
            markers: vec![F::zero(); n_markers],
            positions: (1..=n_markers).map(|i| F::from(i).unwrap()).collect(),
            desired_positions,
            position_increments,
            count: 0,
        }
    }

    /// Update with new observation
    pub fn update(&mut self, value: F) {
        if !value.is_finite() {
            return;
        }

        self.count += 1;

        if self.count <= self.markers.len() {
            // Initial phase: collect first n markers
            self.markers[self.count - 1] = value;
            if self.count == self.markers.len() {
                // Sort initial markers
                self.markers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            return;
        }

        // Find position of new observation
        let mut k = 0;
        for (i, &marker) in self.markers.iter().enumerate() {
            if value < marker {
                k = i;
                break;
            }
            k = i + 1;
        }

        // Update marker positions
        for i in k..self.markers.len() {
            self.positions[i] = self.positions[i] + F::one();
        }

        // Update desired positions
        for i in 0..self.markers.len() {
            self.desired_positions[i] = self.desired_positions[i] + self.position_increments[i];
        }

        // Adjust markers using piecewise-parabolic interpolation
        for i in 1..self.markers.len() - 1 {
            let d = self.desired_positions[i] - self.positions[i];
            if (d >= F::one() && self.positions[i + 1] - self.positions[i] > F::one()) ||
               (d <= -F::one() && self.positions[i - 1] - self.positions[i] < -F::one()) {
                
                let d_sign = if d >= F::zero() { F::one() } else { -F::one() };
                
                // Parabolic interpolation
                let qi_plus = self.parabolic_interpolation(i, d_sign);
                
                if self.markers[i - 1] < qi_plus && qi_plus < self.markers[i + 1] {
                    self.markers[i] = qi_plus;
                } else {
                    // Linear interpolation fallback
                    self.markers[i] = self.linear_interpolation(i, d_sign);
                }
                
                self.positions[i] = self.positions[i] + d_sign;
            }
        }

        // Handle new observation
        if k < self.markers.len() {
            self.markers[k] = value;
        }
    }

    /// Get estimated quantile values
    pub fn quantiles(&self) -> Vec<F> {
        if self.count < self.markers.len() {
            // Not enough data, return sorted subset
            let mut subset = self.markers[0..self.count].to_vec();
            subset.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            return subset;
        }

        // Return middle markers corresponding to requested quantiles
        self.markers[2..self.markers.len() - 2].to_vec()
    }

    fn parabolic_interpolation(&self, i: usize, d: F) -> F {
        let qi_minus = self.markers[i - 1];
        let qi = self.markers[i];
        let qi_plus = self.markers[i + 1];
        
        let ni_minus = self.positions[i - 1];
        let ni = self.positions[i];
        let ni_plus = self.positions[i + 1];

        qi + d / (ni_plus - ni_minus) * (
            (ni - ni_minus + d) * (qi_plus - qi) / (ni_plus - ni) +
            (ni_plus - ni - d) * (qi - qi_minus) / (ni - ni_minus)
        )
    }

    fn linear_interpolation(&self, i: usize, d: F) -> F {
        if d > F::zero() {
            let qi = self.markers[i];
            let qi_plus = self.markers[i + 1];
            let ni = self.positions[i];
            let ni_plus = self.positions[i + 1];
            
            qi + d * (qi_plus - qi) / (ni_plus - ni)
        } else {
            let qi_minus = self.markers[i - 1];
            let qi = self.markers[i];
            let ni_minus = self.positions[i - 1];
            let ni = self.positions[i];
            
            qi + d * (qi - qi_minus) / (ni - ni_minus)
        }
    }
}

/// Streaming correlation estimator
#[derive(Debug, Clone)]
pub struct StreamingCorrelationEstimator<F> {
    /// Number of variables
    n_vars: usize,
    /// Means for each variable
    means: Vec<F>,
    /// Cross-moment accumulators
    cross_moments: Vec<Vec<F>>,
    /// Variance accumulators
    variances: Vec<F>,
    /// Count of observations
    count: usize,
}

impl<F> StreamingCorrelationEstimator<F>
where
    F: Float + NumCast + Zero + One + Copy,
{
    /// Create new correlation estimator for n variables
    pub fn new(n_vars: usize) -> Self {
        Self {
            n_vars,
            means: vec![F::zero(); n_vars],
            cross_moments: vec![vec![F::zero(); n_vars]; n_vars],
            variances: vec![F::zero(); n_vars],
            count: 0,
        }
    }

    /// Update with new observation vector
    pub fn update(&mut self, values: &[F]) {
        if values.len() != self.n_vars {
            return; // Skip mismatched dimensions
        }

        // Check for finite values
        if !values.iter().all(|v| v.is_finite()) {
            return;
        }

        self.count += 1;
        let n = F::from(self.count).unwrap();

        // Update means and cross-moments using Welford's algorithm
        let mut deltas_before = vec![F::zero(); self.n_vars];
        let mut deltas_after = vec![F::zero(); self.n_vars];

        // Calculate deltas before mean update
        for i in 0..self.n_vars {
            deltas_before[i] = values[i] - self.means[i];
        }

        // Update means
        for i in 0..self.n_vars {
            self.means[i] = self.means[i] + deltas_before[i] / n;
            deltas_after[i] = values[i] - self.means[i];
        }

        // Update cross-moments
        for i in 0..self.n_vars {
            for j in 0..self.n_vars {
                self.cross_moments[i][j] = self.cross_moments[i][j] + deltas_before[i] * deltas_after[j];
            }
        }

        // Update variances (diagonal elements)
        for i in 0..self.n_vars {
            self.variances[i] = self.cross_moments[i][i];
        }
    }

    /// Get correlation matrix
    pub fn correlation_matrix(&self) -> StatsResult<Array2<F>> {
        if self.count < 2 {
            return Err(StatsError::InvalidArgument(
                "Need at least 2 observations for correlation".to_string(),
            ));
        }

        let mut corr_matrix = Array2::zeros((self.n_vars, self.n_vars));
        let n_minus_1 = F::from(self.count - 1).unwrap();

        for i in 0..self.n_vars {
            for j in 0..self.n_vars {
                if i == j {
                    corr_matrix[[i, j]] = F::one();
                } else {
                    let covariance = self.cross_moments[i][j] / n_minus_1;
                    let std_i = (self.variances[i] / n_minus_1).sqrt();
                    let std_j = (self.variances[j] / n_minus_1).sqrt();
                    
                    if std_i > F::zero() && std_j > F::zero() {
                        corr_matrix[[i, j]] = covariance / (std_i * std_j);
                    } else {
                        corr_matrix[[i, j]] = F::zero();
                    }
                }
            }
        }

        Ok(corr_matrix)
    }

    /// Get covariance matrix
    pub fn covariance_matrix(&self) -> StatsResult<Array2<F>> {
        if self.count < 2 {
            return Err(StatsError::InvalidArgument(
                "Need at least 2 observations for covariance".to_string(),
            ));
        }

        let mut cov_matrix = Array2::zeros((self.n_vars, self.n_vars));
        let n_minus_1 = F::from(self.count - 1).unwrap();

        for i in 0..self.n_vars {
            for j in 0..self.n_vars {
                cov_matrix[[i, j]] = self.cross_moments[i][j] / n_minus_1;
            }
        }

        Ok(cov_matrix)
    }
}

/// Streaming histogram with adaptive binning
#[derive(Debug, Clone)]
pub struct StreamingHistogram<F> {
    /// Bin edges
    bin_edges: Vec<F>,
    /// Bin counts
    bin_counts: Vec<usize>,
    /// Minimum value seen
    min_value: F,
    /// Maximum value seen
    max_value: F,
    /// Total count
    total_count: usize,
    /// Maximum number of bins
    max_bins: usize,
    /// Adaptive binning enabled
    adaptive: bool,
}

impl<F> StreamingHistogram<F>
where
    F: Float + NumCast + Zero + One + Copy,
{
    /// Create new streaming histogram
    pub fn new(max_bins: usize, adaptive: bool) -> Self {
        Self {
            bin_edges: Vec::new(),
            bin_counts: Vec::new(),
            min_value: F::infinity(),
            max_value: F::neg_infinity(),
            total_count: 0,
            max_bins,
            adaptive,
        }
    }

    /// Update histogram with new value
    pub fn update(&mut self, value: F) {
        if !value.is_finite() {
            return;
        }

        self.total_count += 1;

        // Update min/max
        if value < self.min_value {
            self.min_value = value;
        }
        if value > self.max_value {
            self.max_value = value;
        }

        // Initialize bins if first few observations
        if self.bin_edges.is_empty() && self.total_count >= 10 {
            self.initialize_bins();
        }

        if !self.bin_edges.is_empty() {
            // Find appropriate bin
            let bin_index = self.find_bin_index(value);
            
            if bin_index < self.bin_counts.len() {
                self.bin_counts[bin_index] += 1;
            }

            // Adaptive rebinning if enabled
            if self.adaptive && self.total_count % 1000 == 0 {
                self.rebin_if_needed();
            }
        }
    }

    /// Get histogram as (bin_centers, counts)
    pub fn histogram(&self) -> (Vec<F>, Vec<usize>) {
        if self.bin_edges.len() < 2 {
            return (Vec::new(), Vec::new());
        }

        let mut bin_centers = Vec::new();
        for i in 0..self.bin_edges.len() - 1 {
            let center = (self.bin_edges[i] + self.bin_edges[i + 1]) / F::from(2).unwrap();
            bin_centers.push(center);
        }

        (bin_centers, self.bin_counts.clone())
    }

    /// Get probability density function
    pub fn pdf(&self) -> (Vec<F>, Vec<F>) {
        let (centers, counts) = self.histogram();
        if centers.is_empty() || self.total_count == 0 {
            return (Vec::new(), Vec::new());
        }

        let bin_width = if self.bin_edges.len() >= 2 {
            self.bin_edges[1] - self.bin_edges[0]
        } else {
            F::one()
        };

        let total_count_f = F::from(self.total_count).unwrap();
        let densities: Vec<F> = counts.iter()
            .map(|&count| F::from(count).unwrap() / (total_count_f * bin_width))
            .collect();

        (centers, densities)
    }

    fn initialize_bins(&mut self) {
        if self.min_value >= self.max_value {
            return;
        }

        let range = self.max_value - self.min_value;
        let bin_width = range / F::from(self.max_bins - 1).unwrap();

        self.bin_edges.clear();
        for i in 0..self.max_bins {
            self.bin_edges.push(self.min_value + F::from(i).unwrap() * bin_width);
        }

        self.bin_counts = vec![0; self.max_bins - 1];
    }

    fn find_bin_index(&self, value: F) -> usize {
        for i in 0..self.bin_edges.len() - 1 {
            if value >= self.bin_edges[i] && value < self.bin_edges[i + 1] {
                return i;
            }
        }
        // Handle edge case for maximum value
        if value == self.max_value && !self.bin_counts.is_empty() {
            return self.bin_counts.len() - 1;
        }
        self.bin_counts.len() // Out of range
    }

    fn rebin_if_needed(&mut self) {
        // Simple rebinning strategy: merge adjacent bins with low counts
        let min_count_threshold = self.total_count / (self.max_bins * 10);
        
        let mut new_edges = vec![self.bin_edges[0]];
        let mut new_counts = Vec::new();
        let mut current_count = 0;

        for i in 0..self.bin_counts.len() {
            current_count += self.bin_counts[i];
            
            if current_count >= min_count_threshold || i == self.bin_counts.len() - 1 {
                new_edges.push(self.bin_edges[i + 1]);
                new_counts.push(current_count);
                current_count = 0;
            }
        }

        if new_counts.len() < self.bin_counts.len() {
            self.bin_edges = new_edges;
            self.bin_counts = new_counts;
        }
    }
}

/// Streaming data processor for large datasets
pub struct StreamingProcessor<F> {
    config: StreamingConfig,
    _phantom: PhantomData<F>,
}

impl<F> StreamingProcessor<F>
where
    F: Float + NumCast + SimdUnifiedOps + Zero + One + PartialOrd + Copy + Send + Sync,
{
    /// Create new streaming processor
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }

    /// Process streaming descriptive statistics
    pub fn stream_descriptive_stats<R>(
        &self,
        data_reader: R,
    ) -> StatsResult<StreamingStatsAccumulator<F>>
    where
        R: Iterator<Item = StatsResult<Array1<F>>> + Send,
    {
        let mut accumulator = StreamingStatsAccumulator::new();
        let mut chunks_processed = 0;

        for chunk_result in data_reader {
            let chunk = chunk_result?;
            accumulator.update_batch(&chunk.view());
            chunks_processed += 1;

            // Memory management
            if self.config.adaptive_memory && chunks_processed % 100 == 0 {
                // Force garbage collection hint
                #[allow(clippy::manual_memcpy)]
                {
                    let _temp = vec![0u8; 1024]; // Small allocation to trigger GC
                }
            }
        }

        Ok(accumulator)
    }

    /// Process streaming correlation matrix
    pub fn stream_correlation_matrix<R>(
        &self,
        data_reader: R,
        n_vars: usize,
    ) -> StatsResult<Array2<F>>
    where
        R: Iterator<Item = StatsResult<Array2<F>>> + Send,
    {
        let mut estimator = StreamingCorrelationEstimator::new(n_vars);

        for chunk_result in data_reader {
            let chunk = chunk_result?;
            
            // Process each row of the chunk
            for row in chunk.rows() {
                let row_vec: Vec<F> = row.to_vec();
                estimator.update(&row_vec);
            }
        }

        estimator.correlation_matrix()
    }

    /// Process streaming quantiles
    pub fn stream_quantiles<R>(
        &self,
        data_reader: R,
        quantiles: Vec<F>,
    ) -> StatsResult<Vec<F>>
    where
        R: Iterator<Item = StatsResult<Array1<F>>> + Send,
    {
        let mut estimator = StreamingQuantileEstimator::new(quantiles);

        for chunk_result in data_reader {
            let chunk = chunk_result?;
            
            for &value in chunk.iter() {
                estimator.update(value);
            }
        }

        Ok(estimator.quantiles())
    }

    /// Process streaming histogram
    pub fn stream_histogram<R>(
        &self,
        data_reader: R,
        max_bins: usize,
    ) -> StatsResult<(Vec<F>, Vec<F>)>
    where
        R: Iterator<Item = StatsResult<Array1<F>>> + Send,
    {
        let mut histogram = StreamingHistogram::new(max_bins, self.config.adaptive_memory);

        for chunk_result in data_reader {
            let chunk = chunk_result?;
            
            for &value in chunk.iter() {
                histogram.update(value);
            }
        }

        Ok(histogram.pdf())
    }

    /// Process large dataset with memory constraints
    pub fn process_large_dataset<R, T>(
        &self,
        data_reader: R,
        processor_fn: impl Fn(&ArrayView1<F>) -> StatsResult<T> + Send + Sync,
        combiner_fn: impl Fn(Vec<T>) -> StatsResult<T> + Send + Sync,
    ) -> StatsResult<T>
    where
        R: Iterator<Item = StatsResult<Array1<F>>> + Send,
        T: Send + Clone,
    {
        let mut partial_results = Vec::new();
        let mut memory_used = 0;

        for chunk_result in data_reader {
            let chunk = chunk_result?;
            
            // Estimate memory usage
            memory_used += chunk.len() * std::mem::size_of::<F>();
            
            // Process chunk
            let partial_result = processor_fn(&chunk.view())?;
            partial_results.push(partial_result);

            // Memory management
            if memory_used > self.config.memory_limit {
                // Combine partial results to free memory
                if partial_results.len() > 1 {
                    let combined = combiner_fn(partial_results)?;
                    partial_results = vec![combined];
                    memory_used = std::mem::size_of::<T>();
                }
            }
        }

        // Final combination
        combiner_fn(partial_results)
    }
}

/// Memory-mapped data reader for large files
pub struct MemoryMappedReader {
    file_path: String,
    chunk_size: usize,
    current_position: usize,
}

impl MemoryMappedReader {
    /// Create new memory-mapped reader
    pub fn new(file_path: String, chunk_size: usize) -> Self {
        Self {
            file_path,
            chunk_size,
            current_position: 0,
        }
    }

    /// Create iterator for reading chunks
    pub fn iter_chunks<F>(&mut self) -> impl Iterator<Item = StatsResult<Array1<F>>> + '_
    where
        F: Float + NumCast + Copy,
    {
        std::iter::from_fn(move || {
            // This is a simplified placeholder - would implement actual memory mapping
            if self.current_position > 10000 {
                return None; // End of data
            }

            let chunk_data: Vec<F> = (0..self.chunk_size)
                .map(|i| F::from(self.current_position + i).unwrap())
                .collect();

            self.current_position += self.chunk_size;
            Some(Ok(Array1::from_vec(chunk_data)))
        })
    }
}

/// Convenience functions for streaming operations
#[allow(dead_code)]
pub fn stream_mean<F, R>(data_reader: R) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + Zero + One + PartialOrd + Copy + Send + Sync,
    R: Iterator<Item = StatsResult<Array1<F>>> + Send,
{
    let processor = StreamingProcessor::new(StreamingConfig::default());
    let accumulator = processor.stream_descriptive_stats(data_reader)?;
    Ok(accumulator.mean())
}

#[allow(dead_code)]
pub fn stream_variance<F, R>(data_reader: R) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + Zero + One + PartialOrd + Copy + Send + Sync,
    R: Iterator<Item = StatsResult<Array1<F>>> + Send,
{
    let processor = StreamingProcessor::new(StreamingConfig::default());
    let accumulator = processor.stream_descriptive_stats(data_reader)?;
    Ok(accumulator.sample_variance())
}

#[allow(dead_code)]
pub fn stream_std<F, R>(data_reader: R) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps + Zero + One + PartialOrd + Copy + Send + Sync,
    R: Iterator<Item = StatsResult<Array1<F>>> + Send,
{
    let processor = StreamingProcessor::new(StreamingConfig::default());
    let accumulator = processor.stream_descriptive_stats(data_reader)?;
    Ok(accumulator.sample_std_dev())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_streaming_stats_accumulator() {
        let mut accumulator = StreamingStatsAccumulator::<f64>::new();
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        
        accumulator.update_batch(&data.view());
        
        assert_eq!(accumulator.count, 5);
        assert!((accumulator.mean() - 3.0).abs() < 1e-10);
        assert!(accumulator.variance() > 0.0);
    }

    #[test]
    fn test_streaming_quantile_estimator() {
        let mut estimator = StreamingQuantileEstimator::new(vec![0.5]); // Median
        
        for i in 1..=100 {
            estimator.update(i as f64);
        }
        
        let quantiles = estimator.quantiles();
        assert!(!quantiles.is_empty());
        // For 1..100, median should be around 50.5
        if !quantiles.is_empty() {
            assert!((quantiles[0] - 50.5).abs() < 5.0); // Allow some error for P² algorithm
        }
    }

    #[test]
    fn test_streaming_correlation_estimator() {
        let mut estimator = StreamingCorrelationEstimator::new(2);
        
        // Add perfectly correlated data
        for i in 1..=10 {
            estimator.update(&[i as f64, (i * 2) as f64]);
        }
        
        let corr_matrix = estimator.correlation_matrix().unwrap();
        assert_eq!(corr_matrix.dim(), (2, 2));
        assert!((corr_matrix[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((corr_matrix[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((corr_matrix[[0, 1]] - 1.0).abs() < 1e-10); // Perfect positive correlation
    }

    #[test]
    fn test_streaming_histogram() {
        let mut histogram = StreamingHistogram::new(10, false);
        
        // Add normal-like data
        for i in 0..1000 {
            let value = (i as f64 - 500.0) / 100.0; // Centered around 0, scaled
            histogram.update(value);
        }
        
        let (centers, densities) = histogram.pdf();
        assert!(!centers.is_empty());
        assert_eq!(centers.len(), densities.len());
    }

    #[test]
    fn test_streaming_config() {
        let config = StreamingConfig {
            memory_limit: 500_000,
            chunk_size: 1024,
            max_passes: 2,
            accuracy_memory_tradeoff: 0.5,
            adaptive_memory: false,
            parallel_streaming: false,
        };
        
        assert_eq!(config.memory_limit, 500_000);
        assert_eq!(config.chunk_size, 1024);
        assert!(!config.adaptive_memory);
    }

    #[test]
    fn test_memory_mapped_reader() {
        let mut reader = MemoryMappedReader::new("dummy.txt".to_string(), 10);
        let chunks: Vec<_> = reader.iter_chunks::<f64>().take(3).collect();
        
        assert_eq!(chunks.len(), 3);
        for chunk_result in chunks {
            assert!(chunk_result.is_ok());
            let chunk = chunk_result.unwrap();
            assert_eq!(chunk.len(), 10);
        }
    }

    #[test]
    fn test_accumulator_merge() {
        let mut acc1 = StreamingStatsAccumulator::<f64>::new();
        let mut acc2 = StreamingStatsAccumulator::<f64>::new();
        
        acc1.update_batch(&array![1.0, 2.0, 3.0].view());
        acc2.update_batch(&array![4.0, 5.0, 6.0].view());
        
        let combined_mean = (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0) / 6.0;
        
        acc1.merge(&acc2);
        
        assert_eq!(acc1.count, 6);
        assert!((acc1.mean() - combined_mean).abs() < 1e-10);
    }
}
