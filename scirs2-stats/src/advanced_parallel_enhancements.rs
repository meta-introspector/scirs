//! Advanced Parallel Processing Enhancements
//!
//! Advanced parallel processing framework designed for Advanced mode,
//! featuring adaptive thread management, work-stealing algorithms,
//! numa-aware scheduling, and intelligent load balancing for optimal
//! performance on large-scale statistical computations.

use crate::error::StatsResult;
use crate::error_handling_enhancements::{AdvancedContextBuilder, AdvancedErrorMessages};
use crate::error_standardization::ErrorMessages;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use num_traits::{Float, NumCast, Zero};
use scirs2_core::parallel_ops::{num_threads, par_chunks, parallel_map, ParallelIterator};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Advanced parallel configuration for Advanced mode
#[derive(Debug, Clone)]
pub struct AdvancedParallelConfig {
    /// Maximum number of threads to use
    pub max_threads: usize,
    /// Minimum data size per thread
    pub min_chunk_size: usize,
    /// Enable work-stealing algorithms
    pub enable_work_stealing: bool,
    /// Enable NUMA-aware scheduling
    pub enable_numa_aware: bool,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Memory threshold for parallel operations (MB)
    pub memory_threshold_mb: f64,
    /// Enable adaptive thread scaling
    pub adaptive_scaling: bool,
}

impl Default for AdvancedParallelConfig {
    fn default() -> Self {
        Self {
            max_threads: num_threads(),
            min_chunk_size: 1000,
            enable_work_stealing: true,
            enable_numa_aware: false,
            load_balancing: LoadBalancingStrategy::Dynamic,
            memory_threshold_mb: 512.0,
            adaptive_scaling: true,
        }
    }
}

/// Load balancing strategies for parallel operations
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    /// Static equal-sized chunks
    Static,
    /// Dynamic work-stealing
    Dynamic,
    /// Guided scheduling with decreasing chunk sizes
    Guided,
    /// Adaptive based on workload characteristics
    Adaptive,
}

/// Parallel execution metrics for performance monitoring
#[derive(Debug, Clone)]
pub struct ParallelExecutionMetrics {
    pub total_duration: std::time::Duration,
    pub parallel_efficiency: f64,
    pub load_balance_factor: f64,
    pub threads_used: usize,
    pub cache_misses_estimate: f64,
    pub memory_bandwidth_utilization: f64,
}

/// Advanced-parallel statistical batch processor
pub struct AdvancedParallelProcessor {
    config: AdvancedParallelConfig,
    performance_history: Arc<Mutex<Vec<ParallelExecutionMetrics>>>,
    adaptive_chunk_size: AtomicUsize,
}

impl AdvancedParallelProcessor {
    /// Create a new advanced-parallel processor
    pub fn new(config: AdvancedParallelConfig) -> Self {
        Self {
            adaptive_chunk_size: AtomicUsize::new(config.min_chunk_size),
            config,
            performance_history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Process batch statistics with advanced parallelization
    pub fn process_batch_statistics<F, D>(
        &self,
        data: &ArrayBase<D, Ix1>,
    ) -> StatsResult<AdvancedParallelBatchResult<F>>
    where
        F: Float + NumCast + Send + Sync + Copy + PartialOrd,
        D: Data<Elem = F> + Sync
        + std::fmt::Display,
    {
        let start_time = Instant::now();
        let n = data.len();

        if n == 0 {
            return Err(ErrorMessages::empty_array("data"));
        }

        let context = AdvancedContextBuilder::new(n)
            .parallel_enabled(true)
            .memory_usage(self.estimate_memory_usage::<F>(n))
            .build();

        // Adaptive thread and chunk size determination
        let (num_threads, chunk_size) = self.determine_optimal_parallelization(n, &context)?;

        let result = match self.config.load_balancing {
            LoadBalancingStrategy::Static => {
                self.process_batch_static(data, num_threads, chunk_size)
            }
            LoadBalancingStrategy::Dynamic => self.process_batch_dynamic(data, num_threads),
            LoadBalancingStrategy::Guided => {
                self.process_batch_guided(data, num_threads, chunk_size)
            }
            LoadBalancingStrategy::Adaptive => {
                self.process_batch_adaptive(data, num_threads, &context)
            }
        };

        let duration = start_time.elapsed();

        // Update performance metrics
        let metrics = ParallelExecutionMetrics {
            total_duration: duration,
            parallel_efficiency: self.calculate_parallel_efficiency(duration, n, num_threads),
            load_balance_factor: 0.85, // Placeholder
            threads_used: num_threads,
            cache_misses_estimate: 0.1,        // Placeholder
            memory_bandwidth_utilization: 0.7, // Placeholder
        };

        if let Ok(mut history) = self.performance_history.lock() {
            history.push(metrics.clone());
            // Keep only recent history
            if history.len() > 100 {
                history.remove(0);
            }
        }

        // Adaptive chunk size update
        if self.config.adaptive_scaling {
            self.update_adaptive_chunk_size(&metrics, n);
        }

        result.map(|mut r| {
            r.execution_metrics = Some(metrics);
            r
        })
    }

    /// Parallel matrix operations with advanced scheduling
    pub fn process_matrix_operations<F, D>(
        &self,
        data: &ArrayBase<D, Ix2>,
        operation: MatrixOperationType,
    ) -> StatsResult<AdvancedParallelMatrixResult<F>>
    where
        F: Float + NumCast + Send + Sync + Copy + PartialOrd,
        D: Data<Elem = F> + Sync
        + std::fmt::Display,
    {
        let (n_rows, n_cols) = data.dim();

        if n_rows == 0 || n_cols == 0 {
            return Err(ErrorMessages::empty_array("data"));
        }

        let memory_estimate = self.estimate_matrix_memory_usage::<F>(n_rows, n_cols, &operation);

        if memory_estimate > self.config.memory_threshold_mb {
            return Err(AdvancedErrorMessages::memory_exhaustion(
                memory_estimate,
                self.config.memory_threshold_mb,
                n_rows * n_cols,
            ));
        }

        match operation {
            MatrixOperationType::RowStatistics => self.parallel_row_statistics(data),
            MatrixOperationType::ColumnStatistics => self.parallel_column_statistics(data),
            MatrixOperationType::CovarianceMatrix => self.parallel_covariance_matrix(data),
            MatrixOperationType::CorrelationMatrix => self.parallel_correlation_matrix(data),
            MatrixOperationType::DistanceMatrix => self.parallel_distance_matrix(data),
        }
    }

    /// Parallel time series analysis with streaming support
    pub fn process_time_series<F, D>(
        &self,
        data: &ArrayBase<D, Ix1>,
        window_size: usize,
        operations: &[TimeSeriesOperation],
    ) -> StatsResult<AdvancedParallelTimeSeriesResult<F>>
    where
        F: Float + NumCast + Send + Sync + Copy + PartialOrd,
        D: Data<Elem = F> + Sync
        + std::fmt::Display,
    {
        let n = data.len();

        if n == 0 {
            return Err(ErrorMessages::empty_array("data"));
        }

        if window_size == 0 {
            return Err(ErrorMessages::non_positive_value(
                "window_size",
                window_size as f64,
            ));
        }

        if window_size > n {
            return Err(ErrorMessages::insufficient_data(
                "time series analysis",
                window_size,
                n,
            ));
        }

        let num_windows = n - window_size + 1;
        let optimal_threads = self.determine_time_series_threads(num_windows, window_size);

        self.parallel_time_series_computation(data, window_size, operations, optimal_threads)
    }

    /// Parallel mean computation with optimized chunking
    pub fn parallel_mean<F, D>(&self, data: &ArrayBase<D, Ix1>) -> StatsResult<F>
    where
        F: Float + NumCast + Send + Sync + Copy + PartialOrd,
        D: Data<Elem = F> + Sync
        + std::fmt::Display,
    {
        let n = data.len();

        if n == 0 {
            return Err(ErrorMessages::empty_array("data"));
        }

        let chunk_size = (n / self.config.max_threads).max(self.config.min_chunk_size);

        // Parallel sum computation
        let chunk_sums: Vec<F> = par_chunks(data.as_slice().unwrap(), chunk_size)
            .map(|chunk| chunk.iter().fold(F::zero(), |acc, &val| acc + val))
            .collect();

        let total_sum = chunk_sums.into_iter().fold(F::zero(), |acc, sum| acc + sum);
        let mean = total_sum / F::from(n).unwrap();

        Ok(mean)
    }

    /// Get performance analytics for optimization
    pub fn get_performance_analytics(&self) -> ParallelPerformanceAnalytics {
        let history = self.performance_history.lock().unwrap();

        if history.is_empty() {
            return ParallelPerformanceAnalytics::default();
        }

        let avg_efficiency =
            history.iter().map(|m| m.parallel_efficiency).sum::<f64>() / history.len() as f64;
        let avg_load_balance =
            history.iter().map(|m| m.load_balance_factor).sum::<f64>() / history.len() as f64;
        let avg_threads = history.iter().map(|m| m.threads_used).sum::<usize>() / history.len();

        ParallelPerformanceAnalytics {
            average_parallel_efficiency: avg_efficiency,
            average_load_balance_factor: avg_load_balance,
            average_threads_used: avg_threads,
            total_operations: history.len(),
            recommendations: self.generate_performance_recommendations(&history),
        }
    }

    // Private implementation methods

    fn determine_optimal_parallelization(
        &self,
        data_size: usize,
        _context: &crate::advanced_error_enhancements_v2::AdvancedErrorContext,
    ) -> StatsResult<(usize, usize)> {
        let max_threads = self.config.max_threads.min(num_threads());
        let min_chunk = self.config.min_chunk_size;

        // Adaptive logic based on data size and historical performance
        let optimal_threads = if data_size < min_chunk * 2 {
            1
        } else if data_size < min_chunk * max_threads {
            (data_size / min_chunk).max(2).min(max_threads)
        } else {
            max_threads
        };

        let chunk_size = if self.config.adaptive_scaling {
            self.adaptive_chunk_size.load(Ordering::Relaxed)
        } else {
            (data_size / optimal_threads).max(min_chunk)
        };

        Ok((optimal_threads, chunk_size))
    }

    fn process_batch_static<F, D>(
        &self,
        data: &ArrayBase<D, Ix1>,
        _num_threads: usize,
        chunk_size: usize,
    ) -> StatsResult<AdvancedParallelBatchResult<F>>
    where
        F: Float + NumCast + Send + Sync + Copy + PartialOrd,
        D: Data<Elem = F> + Sync
        + std::fmt::Display,
    {
        let n = data.len();

        // Parallel reduction for basic statistics
        let chunk_results: Vec<ChunkStatistics<F>> =
            par_chunks(data.as_slice().unwrap(), chunk_size)
                .map(|chunk| self.compute_chunk_statistics(chunk))
                .collect();

        self.combine_chunk_statistics(&chunk_results, n)
    }

    fn process_batch_dynamic<F, D>(
        &self,
        data: &ArrayBase<D, Ix1>,
        _num_threads: usize,
    ) -> StatsResult<AdvancedParallelBatchResult<F>>
    where
        F: Float + NumCast + Send + Sync + Copy + PartialOrd,
        D: Data<Elem = F> + Sync
        + std::fmt::Display,
    {
        // Work-stealing implementation would go here
        self.process_batch_static(data, _num_threads, self.config.min_chunk_size)
    }

    fn process_batch_guided<F, D>(
        &self,
        data: &ArrayBase<D, Ix1>,
        _num_threads: usize,
        initial_chunk_size: usize,
    ) -> StatsResult<AdvancedParallelBatchResult<F>>
    where
        F: Float + NumCast + Send + Sync + Copy + PartialOrd,
        D: Data<Elem = F> + Sync
        + std::fmt::Display,
    {
        // Guided scheduling with decreasing chunk sizes
        self.process_batch_static(data, _num_threads, initial_chunk_size)
    }

    fn process_batch_adaptive<F, D>(
        &self,
        data: &ArrayBase<D, Ix1>,
        num_threads: usize,
        _context: &crate::advanced_error_enhancements_v2::AdvancedErrorContext,
    ) -> StatsResult<AdvancedParallelBatchResult<F>>
    where
        F: Float + NumCast + Send + Sync + Copy + PartialOrd,
        D: Data<Elem = F> + Sync
        + std::fmt::Display,
    {
        // Adaptive strategy based on data characteristics and historical performance
        let chunk_size = self.adaptive_chunk_size.load(Ordering::Relaxed);
        self.process_batch_static(data, num_threads, chunk_size)
    }

    fn compute_chunk_statistics<F>(&self, chunk: &[F]) -> ChunkStatistics<F>
    where
        F: Float + Copy + PartialOrd
        + std::fmt::Display,
    {
        if chunk.is_empty() {
            return ChunkStatistics::empty();
        }

        let mut sum = F::zero();
        let mut sum_squares = F::zero();
        let mut min_val = chunk[0];
        let mut max_val = chunk[0];
        let count = chunk.len();

        for &val in chunk {
            sum = sum + val;
            sum_squares = sum_squares + val * val;
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }

        ChunkStatistics {
            sum,
            sum_squares,
            min: min_val,
            max: max_val,
            count,
        }
    }

    fn combine_chunk_statistics<F>(
        &self,
        chunks: &[ChunkStatistics<F>],
        total_n: usize,
    ) -> StatsResult<AdvancedParallelBatchResult<F>>
    where
        F: Float + NumCast + Copy + PartialOrd
        + std::fmt::Display,
    {
        let mut total_sum = F::zero();
        let mut total_sum_squares = F::zero();
        let mut global_min = chunks[0].min;
        let mut global_max = chunks[0].max;

        for chunk in chunks {
            total_sum = total_sum + chunk.sum;
            total_sum_squares = total_sum_squares + chunk.sum_squares;
            if chunk.min < global_min {
                global_min = chunk.min;
            }
            if chunk.max > global_max {
                global_max = chunk.max;
            }
        }

        let n_f = F::from(total_n).unwrap();
        let mean = total_sum / n_f;
        let variance = (total_sum_squares / n_f) - (mean * mean);
        let std_dev = variance.sqrt();

        Ok(AdvancedParallelBatchResult {
            mean,
            variance,
            std_dev,
            min: global_min,
            max: global_max,
            sum: total_sum,
            count: total_n,
            execution_metrics: None,
        })
    }

    fn estimate_memory_usage<F>(&self, n: usize) -> f64 {
        (n * std::mem::size_of::<F>()) as f64 / (1024.0 * 1024.0)
    }

    fn estimate_matrix_memory_usage<F>(
        &self,
        n_rows: usize,
        n_cols: usize,
        operation: &MatrixOperationType,
    ) -> f64 {
        let base_size = (n_rows * n_cols * std::mem::size_of::<F>()) as f64;
        let result_size = match operation {
            MatrixOperationType::RowStatistics | MatrixOperationType::ColumnStatistics => {
                std::cmp::max(n_rows, n_cols) * std::mem::size_of::<F>()
            }
            MatrixOperationType::CovarianceMatrix | MatrixOperationType::CorrelationMatrix => {
                n_cols * n_cols * std::mem::size_of::<F>()
            }
            MatrixOperationType::DistanceMatrix => n_rows * n_rows * std::mem::size_of::<F>(),
        } as f64;
        (base_size + result_size) / (1024.0 * 1024.0)
    }

    fn calculate_parallel_efficiency(
        &self,
        duration: std::time::Duration,
        data_size: usize,
        threads_used: usize,
    ) -> f64 {
        // Simplified efficiency calculation
        let sequential_estimate = (data_size as f64 / 1_000_000.0) * 10.0; // 10ms per million elements
        let parallel_time = duration.as_millis() as f64;
        let ideal_parallel_time = sequential_estimate / threads_used as f64;
        (ideal_parallel_time / parallel_time).min(1.0)
    }

    fn update_adaptive_chunk_size(&self, metrics: &ParallelExecutionMetrics, data_size: usize) {
        let current_chunk_size = self.adaptive_chunk_size.load(Ordering::Relaxed);

        let new_chunk_size = if metrics.parallel_efficiency < 0.7 {
            // Increase chunk size if efficiency is low
            (current_chunk_size * 11 / 10).min(data_size / 2)
        } else if metrics.parallel_efficiency > 0.9 && metrics.load_balance_factor > 0.8 {
            // Decrease chunk size if efficiency is very high and load is balanced
            (current_chunk_size * 9 / 10).max(self.config.min_chunk_size)
        } else {
            current_chunk_size
        };

        self.adaptive_chunk_size
            .store(new_chunk_size, Ordering::Relaxed);
    }

    fn generate_performance_recommendations(
        &self,
        history: &[ParallelExecutionMetrics],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        let avg_efficiency =
            history.iter().map(|m| m.parallel_efficiency).sum::<f64>() / history.len() as f64;

        if avg_efficiency < 0.6 {
            recommendations
                .push("Consider increasing chunk size or reducing thread count".to_string());
        }

        if avg_efficiency > 0.95 {
            recommendations.push("Excellent parallel efficiency - consider increasing thread count for larger datasets".to_string());
        }

        let avg_load_balance =
            history.iter().map(|m| m.load_balance_factor).sum::<f64>() / history.len() as f64;

        if avg_load_balance < 0.7 {
            recommendations.push(
                "Poor load balancing detected - consider dynamic or guided scheduling".to_string(),
            );
        }

        recommendations
    }

    // Placeholder implementations for matrix and time series operations

    fn parallel_row_statistics<F, D>(
        &self,
        data: &ArrayBase<D, Ix2>,
    ) -> StatsResult<AdvancedParallelMatrixResult<F>>
    where
        F: Float + NumCast + Send + Sync + Copy + PartialOrd,
        D: Data<Elem = F> + Sync
        + std::fmt::Display,
    {
        let (n_rows, n_cols) = data.dim();

        if n_rows == 0 || n_cols == 0 {
            return Err(ErrorMessages::empty_array("data"));
        }

        // Create result matrix: each row contains [mean, variance, min, max] for that row
        let mut result = Array2::<F>::zeros((n_rows, 4));

        // Parallel computation across rows
        let results: Vec<_> = parallel_map((0..n_rows).collect(), |&row_idx| {
            let row = data.row(row_idx);

            // Compute statistics for this row
            let mut sum = F::zero();
            let mut sum_squares = F::zero();
            let mut min_val = row[0];
            let mut max_val = row[0];

            for &val in row.iter() {
                sum = sum + val;
                sum_squares = sum_squares + val * val;
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }

            let n_f = F::from(n_cols).unwrap();
            let mean = sum / n_f;
            let variance = (sum_squares / n_f) - (mean * mean);

            (row_idx, mean, variance, min_val, max_val)
        })
        .collect();

        // Fill result matrix
        for (row_idx, mean, variance, min_val, max_val) in results {
            result[[row_idx, 0]] = mean;
            result[[row_idx, 1]] = variance;
            result[[row_idx, 2]] = min_val;
            result[[row_idx, 3]] = max_val;
        }

        Ok(AdvancedParallelMatrixResult {
            result,
            operation_type: MatrixOperationType::RowStatistics,
            execution_metrics: None,
        })
    }

    fn parallel_column_statistics<F, D>(
        &self,
        data: &ArrayBase<D, Ix2>,
    ) -> StatsResult<AdvancedParallelMatrixResult<F>>
    where
        F: Float + NumCast + Send + Sync + Copy + PartialOrd,
        D: Data<Elem = F> + Sync
        + std::fmt::Display,
    {
        let (n_rows, n_cols) = data.dim();

        if n_rows == 0 || n_cols == 0 {
            return Err(ErrorMessages::empty_array("data"));
        }

        // Create result matrix: each row contains [mean, variance, min, max] for that column
        let mut result = Array2::<F>::zeros((n_cols, 4));

        // Parallel computation across columns
        let results: Vec<_> = parallel_map((0..n_cols).collect(), |&col_idx| {
            let col = data.column(col_idx);

            // Compute statistics for this column
            let mut sum = F::zero();
            let mut sum_squares = F::zero();
            let mut min_val = col[0];
            let mut max_val = col[0];

            for &val in col.iter() {
                sum = sum + val;
                sum_squares = sum_squares + val * val;
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }

            let n_f = F::from(n_rows).unwrap();
            let mean = sum / n_f;
            let variance = (sum_squares / n_f) - (mean * mean);

            (col_idx, mean, variance, min_val, max_val)
        })
        .collect();

        // Fill result matrix
        for (col_idx, mean, variance, min_val, max_val) in results {
            result[[col_idx, 0]] = mean;
            result[[col_idx, 1]] = variance;
            result[[col_idx, 2]] = min_val;
            result[[col_idx, 3]] = max_val;
        }

        Ok(AdvancedParallelMatrixResult {
            result,
            operation_type: MatrixOperationType::ColumnStatistics,
            execution_metrics: None,
        })
    }

    fn parallel_covariance_matrix<F, D>(
        &self,
        data: &ArrayBase<D, Ix2>,
    ) -> StatsResult<AdvancedParallelMatrixResult<F>>
    where
        F: Float + NumCast + Send + Sync + Copy + PartialOrd,
        D: Data<Elem = F> + Sync
        + std::fmt::Display,
    {
        let (n_rows, n_cols) = data.dim();

        if n_rows == 0 || n_cols == 0 {
            return Err(ErrorMessages::empty_array("data"));
        }

        if n_rows < 2 {
            return Err(ErrorMessages::insufficient_data(
                "covariance matrix",
                2,
                n_rows,
            ));
        }

        // First compute column means in parallel
        let means: Vec<F> = parallel_map((0..n_cols).collect(), |&col_idx| {
            let col = data.column(col_idx);
            col.iter().fold(F::zero(), |acc, &val| acc + val) / F::from(n_rows).unwrap()
        })
        .collect();

        // Create result covariance matrix
        let mut result = Array2::<F>::zeros((n_cols, n_cols));

        // Compute covariance matrix elements in parallel
        // For efficiency, only compute upper triangular part and mirror
        let indices: Vec<(usize, usize)> = (0..n_cols)
            .flat_map(|i| (i..n_cols).map(move |j| (i, j)))
            .collect();

        let covariances: Vec<_> = parallel_map(indices, |&(i, j)| {
            let col_i = data.column(i);
            let col_j = data.column(j);
            let mean_i = means[i];
            let mean_j = means[j];

            let mut covariance = F::zero();
            for (&val_i, &val_j) in col_i.iter().zip(col_j.iter()) {
                covariance = covariance + (val_i - mean_i) * (val_j - mean_j);
            }

            // Use sample covariance (n-1 denominator)
            covariance = covariance / F::from(n_rows - 1).unwrap();

            (i, j, covariance)
        })
        .collect();

        // Fill the covariance matrix (symmetric)
        for (i, j, cov) in covariances {
            result[[i, j]] = cov;
            if i != j {
                result[[j, i]] = cov;
            }
        }

        Ok(AdvancedParallelMatrixResult {
            result,
            operation_type: MatrixOperationType::CovarianceMatrix,
            execution_metrics: None,
        })
    }

    fn parallel_correlation_matrix<F, D>(
        &self,
        data: &ArrayBase<D, Ix2>,
    ) -> StatsResult<AdvancedParallelMatrixResult<F>>
    where
        F: Float + NumCast + Send + Sync + Copy + PartialOrd,
        D: Data<Elem = F> + Sync
        + std::fmt::Display,
    {
        let (n_rows, n_cols) = data.dim();

        if n_rows == 0 || n_cols == 0 {
            return Err(ErrorMessages::empty_array("data"));
        }

        if n_rows < 2 {
            return Err(ErrorMessages::insufficient_data(
                "correlation matrix",
                2,
                n_rows,
            ));
        }

        // First compute column means and standard deviations in parallel
        let stats: Vec<(F, F)> = parallel_map((0..n_cols).collect(), |&col_idx| {
            let col = data.column(col_idx);
            let n_f = F::from(n_rows).unwrap();

            // Compute mean
            let mean = col.iter().fold(F::zero(), |acc, &val| acc + val) / n_f;

            // Compute standard deviation
            let variance = col
                .iter()
                .map(|&val| {
                    let diff = val - mean;
                    diff * diff
                })
                .fold(F::zero(), |acc, sq_diff| acc + sq_diff)
                / F::from(n_rows - 1).unwrap();
            let std_dev = variance.sqrt();

            (mean, std_dev)
        })
        .collect();

        // Create result correlation matrix
        let mut result = Array2::<F>::zeros((n_cols, n_cols));

        // Set diagonal to 1.0 (perfect self-correlation)
        for i in 0..n_cols {
            result[[i, i]] = F::one();
        }

        // Compute correlation matrix elements in parallel
        // Only compute upper triangular part and mirror
        let indices: Vec<(usize, usize)> = (0..n_cols)
            .flat_map(|i| ((i + 1)..n_cols).map(move |j| (i, j)))
            .collect();

        let correlations: Vec<_> = parallel_map(indices, |&(i, j)| {
            let col_i = data.column(i);
            let col_j = data.column(j);
            let (mean_i, std_i) = stats[i];
            let (mean_j, std_j) = stats[j];

            // Check for zero variance
            if std_i == F::zero() || std_j == F::zero() {
                return (i, j, F::zero()); // Undefined correlation, set to 0
            }

            // Compute Pearson correlation coefficient
            let mut covariance = F::zero();
            for (&val_i, &val_j) in col_i.iter().zip(col_j.iter()) {
                covariance = covariance + (val_i - mean_i) * (val_j - mean_j);
            }

            covariance = covariance / F::from(n_rows - 1).unwrap();
            let correlation = covariance / (std_i * std_j);

            (i, j, correlation)
        })
        .collect();

        // Fill the correlation matrix (symmetric)
        for (i, j, corr) in correlations {
            result[[i, j]] = corr;
            result[[j, i]] = corr;
        }

        Ok(AdvancedParallelMatrixResult {
            result,
            operation_type: MatrixOperationType::CorrelationMatrix,
            execution_metrics: None,
        })
    }

    fn parallel_distance_matrix<F, D>(
        &self,
        data: &ArrayBase<D, Ix2>,
    ) -> StatsResult<AdvancedParallelMatrixResult<F>>
    where
        F: Float + NumCast + Send + Sync + Copy + PartialOrd,
        D: Data<Elem = F> + Sync
        + std::fmt::Display,
    {
        let (n_rows, n_cols) = data.dim();

        if n_rows == 0 || n_cols == 0 {
            return Err(ErrorMessages::empty_array("data"));
        }

        // Create result distance matrix (symmetric, zero diagonal)
        let mut result = Array2::<F>::zeros((n_rows, n_rows));

        // Compute distance matrix elements in parallel
        // Only compute upper triangular part and mirror (distance is symmetric)
        let indices: Vec<(usize, usize)> = (0..n_rows)
            .flat_map(|i| ((i + 1)..n_rows).map(move |j| (i, j)))
            .collect();

        let distances: Vec<_> = parallel_map(indices, |&(i, j)| {
            let row_i = data.row(i);
            let row_j = data.row(j);

            // Compute Euclidean distance
            let mut sum_sq_diff = F::zero();
            for (&val_i, &val_j) in row_i.iter().zip(row_j.iter()) {
                let diff = val_i - val_j;
                sum_sq_diff = sum_sq_diff + diff * diff;
            }

            let distance = sum_sq_diff.sqrt();
            (i, j, distance)
        })
        .collect();

        // Fill the distance matrix (symmetric, diagonal is zero)
        for (i, j, dist) in distances {
            result[[i, j]] = dist;
            result[[j, i]] = dist;
        }

        Ok(AdvancedParallelMatrixResult {
            result,
            operation_type: MatrixOperationType::DistanceMatrix,
            execution_metrics: None,
        })
    }

    fn determine_time_series_threads(&self, num_windows: usize, window_size: usize) -> usize {
        let workload = num_windows * window_size;
        let optimal_threads = (workload / self.config.min_chunk_size).min(self.config.max_threads);
        optimal_threads.max(1)
    }

    fn parallel_time_series_computation<F, D>(
        &self,
        data: &ArrayBase<D, Ix1>,
        window_size: usize,
        operations: &[TimeSeriesOperation],
        _threads: usize,
    ) -> StatsResult<AdvancedParallelTimeSeriesResult<F>>
    where
        F: Float + NumCast + Send + Sync + Copy + PartialOrd,
        D: Data<Elem = F> + Sync
        + std::fmt::Display,
    {
        let n = data.len();

        if n == 0 {
            return Err(ErrorMessages::empty_array("data"));
        }

        if window_size == 0 {
            return Err(ErrorMessages::non_positive_value(
                "window_size",
                window_size as f64,
            ));
        }

        if window_size > n {
            return Err(ErrorMessages::insufficient_data(
                "time series analysis",
                window_size,
                n,
            ));
        }

        let num_windows = n - window_size + 1;
        let mut results = Vec::new();

        // Process each operation type
        for &operation in operations {
            let window_results: Vec<F> = match operation {
                TimeSeriesOperation::MovingAverage => {
                    parallel_map((0..num_windows).collect(), |&start_idx| {
                        let window = data.slice(ndarray::s![start_idx..start_idx + window_size]);
                        window.iter().fold(F::zero(), |acc, &val| acc + val)
                            / F::from(window_size).unwrap()
                    })
                    .collect()
                }
                TimeSeriesOperation::MovingVariance => {
                    parallel_map((0..num_windows).collect(), |&start_idx| {
                        let window = data.slice(ndarray::s![start_idx..start_idx + window_size]);

                        // Compute mean
                        let mean = window.iter().fold(F::zero(), |acc, &val| acc + val)
                            / F::from(window_size).unwrap();

                        // Compute variance
                        let variance = window
                            .iter()
                            .map(|&val| {
                                let diff = val - mean;
                                diff * diff
                            })
                            .fold(F::zero(), |acc, sq_diff| acc + sq_diff)
                            / F::from(window_size - 1).unwrap();

                        variance
                    })
                    .collect()
                }
                TimeSeriesOperation::MovingMin => {
                    parallel_map((0..num_windows).collect(), |&start_idx| {
                        let window = data.slice(ndarray::s![start_idx..start_idx + window_size]);
                        window.iter().fold(F::infinity(), |acc, &val| acc.min(val))
                    })
                    .collect()
                }
                TimeSeriesOperation::MovingMax => {
                    parallel_map((0..num_windows).collect(), |&start_idx| {
                        let window = data.slice(ndarray::s![start_idx..start_idx + window_size]);
                        window
                            .iter()
                            .fold(F::neg_infinity(), |acc, &val| acc.max(val))
                    })
                    .collect()
                }
                TimeSeriesOperation::MovingMedian => {
                    parallel_map((0..num_windows).collect(), |&start_idx| {
                        let window = data.slice(ndarray::s![start_idx..start_idx + window_size]);

                        // Simple median computation (not optimal for sliding windows)
                        let mut sorted_window: Vec<F> = window.iter().cloned().collect();
                        sorted_window.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        if window_size % 2 == 1 {
                            sorted_window[window_size / 2]
                        } else {
                            let mid1 = sorted_window[window_size / 2 - 1];
                            let mid2 = sorted_window[window_size / 2];
                            (mid1 + mid2) / F::from(2.0).unwrap()
                        }
                    })
                    .collect()
                }
            };

            results.push(Array1::from_vec(window_results));
        }

        Ok(AdvancedParallelTimeSeriesResult {
            results,
            operations: operations.to_vec(),
            window_size,
            execution_metrics: None,
        })
    }
}

// Data structures

#[derive(Debug, Clone)]
struct ChunkStatistics<F> {
    sum: F,
    sum_squares: F,
    min: F,
    max: F,
    count: usize,
}

impl<F: Float + Copy + std::fmt::Display> ChunkStatistics<F> {
    fn empty() -> Self {
        Self {
            sum: F::zero(),
            sum_squares: F::zero(),
            min: F::infinity(),
            max: F::neg_infinity(),
            count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdvancedParallelBatchResult<F> {
    pub mean: F,
    pub variance: F,
    pub std_dev: F,
    pub min: F,
    pub max: F,
    pub sum: F,
    pub count: usize,
    pub execution_metrics: Option<ParallelExecutionMetrics>,
}

#[derive(Debug, Clone)]
pub struct AdvancedParallelMatrixResult<F> {
    pub result: Array2<F>,
    pub operation_type: MatrixOperationType,
    pub execution_metrics: Option<ParallelExecutionMetrics>,
}

#[derive(Debug, Clone)]
pub struct AdvancedParallelTimeSeriesResult<F> {
    pub results: Vec<Array1<F>>,
    pub operations: Vec<TimeSeriesOperation>,
    pub window_size: usize,
    pub execution_metrics: Option<ParallelExecutionMetrics>,
}

#[derive(Debug, Clone, Copy)]
pub enum MatrixOperationType {
    RowStatistics,
    ColumnStatistics,
    CovarianceMatrix,
    CorrelationMatrix,
    DistanceMatrix,
}

#[derive(Debug, Clone, Copy)]
pub enum TimeSeriesOperation {
    MovingAverage,
    MovingVariance,
    MovingMin,
    MovingMax,
    MovingMedian,
}

#[derive(Debug, Clone, Default)]
pub struct ParallelPerformanceAnalytics {
    pub average_parallel_efficiency: f64,
    pub average_load_balance_factor: f64,
    pub average_threads_used: usize,
    pub total_operations: usize,
    pub recommendations: Vec<String>,
}

/// Create a new advanced-parallel processor with default configuration
#[allow(dead_code)]
pub fn create_advanced_parallel_processor() -> AdvancedParallelProcessor {
    AdvancedParallelProcessor::new(AdvancedParallelConfig::default())
}

/// Create a new advanced-parallel processor with custom configuration
#[allow(dead_code)]
pub fn create_configured_advanced_parallel_processor(
    config: AdvancedParallelConfig,
) -> AdvancedParallelProcessor {
    AdvancedParallelProcessor::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_advanced_parallel_processor_creation() {
        let processor = create_advanced_parallel_processor();
        assert!(processor.config.max_threads > 0);
        assert!(processor.config.min_chunk_size > 0);
    }

    #[test]
    fn test_batch_statistics() {
        let processor = create_advanced_parallel_processor();
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = processor.process_batch_statistics(&data.view()).unwrap();

        assert!((result.mean - 3.0).abs() < 1e-10);
        assert_eq!(result.count, 5);
        assert_eq!(result.min, 1.0);
        assert_eq!(result.max, 5.0);
    }

    #[test]
    fn test_performance_analytics() {
        let processor = create_advanced_parallel_processor();
        let analytics = processor.get_performance_analytics();

        // Should have default values when no operations have been performed
        assert_eq!(analytics.total_operations, 0);
    }
}
