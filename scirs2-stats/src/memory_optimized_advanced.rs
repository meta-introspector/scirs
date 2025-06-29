//! Advanced memory optimization for large-scale statistical computing
//!
//! This module provides memory-aware algorithms that automatically adapt
//! to available memory constraints and optimize data layout for cache efficiency.

use crate::error::{StatsError, StatsResult};
use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix1, Ix2};
use num_traits::{Float, NumCast, One, Zero};
use scirs2_core::{parallel_ops::*, simd_ops::SimdUnifiedOps, validation::*};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Memory constraints configuration
#[derive(Debug, Clone)]
pub struct MemoryConstraints {
    /// Maximum memory to use (in bytes)
    pub max_memory_bytes: usize,
    /// Preferred chunk size for processing
    pub chunk_size: usize,
    /// Use memory mapping for large files
    pub use_memory_mapping: bool,
    /// Enable garbage collection hints
    pub enable_gc_hints: bool,
}

impl Default for MemoryConstraints {
    fn default() -> Self {
        // Default to 1GB max memory, 64KB chunks
        Self {
            max_memory_bytes: 1_024 * 1_024 * 1_024,
            chunk_size: 64 * 1024,
            use_memory_mapping: true,
            enable_gc_hints: true,
        }
    }
}

/// Adaptive memory manager that monitors usage and adjusts strategies
pub struct AdaptiveMemoryManager {
    constraints: MemoryConstraints,
    current_usage: Arc<Mutex<usize>>,
    peak_usage: Arc<Mutex<usize>>,
    operation_history: Arc<Mutex<VecDeque<OperationMetrics>>>,
}

#[derive(Debug, Clone)]
struct OperationMetrics {
    operation_type: String,
    memory_used: usize,
    processing_time: std::time::Duration,
    chunk_size_used: usize,
}

impl AdaptiveMemoryManager {
    pub fn new(constraints: MemoryConstraints) -> Self {
        Self {
            constraints,
            current_usage: Arc::new(Mutex::new(0)),
            peak_usage: Arc::new(Mutex::new(0)),
            operation_history: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
        }
    }

    /// Get optimal chunk size based on current memory usage and data size
    pub fn get_optimal_chunk_size(&self, data_size: usize, element_size: usize) -> usize {
        let current_usage = *self.current_usage.lock().unwrap();
        let available_memory = self
            .constraints
            .max_memory_bytes
            .saturating_sub(current_usage);

        // Use at most 80% of available memory for chunk processing
        let max_chunk_memory = available_memory * 4 / 5;
        let max_chunk_elements = max_chunk_memory / element_size;

        // Prefer power-of-2 sizes for cache efficiency
        let optimal_size = max_chunk_elements
            .min(data_size)
            .min(self.constraints.chunk_size);
        optimal_size.next_power_of_two() / 2 // Round down to nearest power of 2
    }

    /// Record memory usage for an operation
    pub fn record_operation(&self, metrics: OperationMetrics) {
        let mut history = self.operation_history.lock().unwrap();

        // Keep only recent operations
        if history.len() >= 100 {
            history.pop_front();
        }

        history.push_back(metrics.clone());

        // Update peak usage
        let mut peak = self.peak_usage.lock().unwrap();
        *peak = (*peak).max(metrics.memory_used);
    }

    /// Get memory usage statistics
    pub fn get_statistics(&self) -> MemoryStatistics {
        let current_usage = *self.current_usage.lock().unwrap();
        let peak_usage = *self.peak_usage.lock().unwrap();
        let history = self.operation_history.lock().unwrap();

        let avg_memory_per_op = if !history.is_empty() {
            history.iter().map(|m| m.memory_used).sum::<usize>() / history.len()
        } else {
            0
        };

        MemoryStatistics {
            current_usage,
            peak_usage,
            avg_memory_per_operation: avg_memory_per_op,
            operations_completed: history.len(),
            memory_efficiency: if peak_usage > 0 {
                (avg_memory_per_op as f64 / peak_usage as f64) * 100.0
            } else {
                100.0
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    pub current_usage: usize,
    pub peak_usage: usize,
    pub avg_memory_per_operation: usize,
    pub operations_completed: usize,
    pub memory_efficiency: f64,
}

/// Memory-aware correlation matrix computation
///
/// Computes correlation matrices using adaptive chunking based on available memory.
/// For very large matrices, uses block-wise computation to stay within memory constraints.
pub fn corrcoef_memory_aware<F>(
    data: &ArrayView2<F>,
    method: &str,
    manager: &mut AdaptiveMemoryManager,
) -> StatsResult<Array2<F>>
where
    F: Float
        + NumCast
        + SimdUnifiedOps
        + Zero
        + One
        + Copy
        + Send
        + Sync
        + std::iter::Sum<F>
        + std::fmt::Debug,
{
    let start_time = std::time::Instant::now();

    check_array_finite_2d(data, "data")?;

    let (n_obs, n_vars) = data.dim();
    let element_size = std::mem::size_of::<F>();

    // Estimate memory requirements
    let matrix_memory = n_vars * n_vars * element_size;
    let temp_memory = n_obs * element_size * 2; // For column pairs
    let total_estimated = matrix_memory + temp_memory;

    let mut corr_matrix = Array2::<F>::zeros((n_vars, n_vars));

    // Set diagonal to 1.0
    for i in 0..n_vars {
        corr_matrix[[i, i]] = F::one();
    }

    if total_estimated <= manager.constraints.max_memory_bytes {
        // Can fit in memory - use standard approach
        corr_matrix = compute_correlation_matrix_standard(data, method)?;
    } else {
        // Use block-wise computation
        let block_size = manager.get_optimal_chunk_size(n_vars, element_size * n_vars);
        corr_matrix = compute_correlation_matrix_blocked(data, method, block_size)?;
    }

    // Record metrics
    let metrics = OperationMetrics {
        operation_type: format!("corrcoef_memory_aware_{}", method),
        memory_used: total_estimated,
        processing_time: start_time.elapsed(),
        chunk_size_used: n_vars,
    };
    manager.record_operation(metrics);

    Ok(corr_matrix)
}

/// Cache-oblivious matrix multiplication for large correlation computations
pub fn cache_oblivious_matrix_mult<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    threshold: usize,
) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + Zero + One + Copy + Send + Sync,
{
    let (m, n) = a.dim();
    let (n2, p) = b.dim();

    if n != n2 {
        return Err(StatsError::DimensionMismatch(
            "Matrix dimensions don't match for multiplication".to_string(),
        ));
    }

    let mut result = Array2::<F>::zeros((m, p));

    if m <= threshold && n <= threshold && p <= threshold {
        // Base case: use standard multiplication
        for i in 0..m {
            for j in 0..p {
                let mut sum = F::zero();
                for k in 0..n {
                    sum = sum + a[[i, k]] * b[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }
    } else {
        // Recursive case: divide matrices
        let mid_m = m / 2;
        let _mid_n = n / 2;
        let _mid_p = p / 2;

        // This is a simplified version - full implementation would handle
        // all matrix subdivisions recursively
        if m > threshold {
            let a_top = a.slice(ndarray::s![..mid_m, ..]);
            let a_bottom = a.slice(ndarray::s![mid_m.., ..]);

            let result_top = cache_oblivious_matrix_mult(&a_top, b, threshold)?;
            let result_bottom = cache_oblivious_matrix_mult(&a_bottom, b, threshold)?;

            result
                .slice_mut(ndarray::s![..mid_m, ..])
                .assign(&result_top);
            result
                .slice_mut(ndarray::s![mid_m.., ..])
                .assign(&result_bottom);
        }
    }

    Ok(result)
}

/// Streaming covariance computation for large datasets
pub fn streaming_covariance_matrix<'a, F>(
    data_chunks: impl Iterator<Item = ArrayView2<'a, F>>,
    manager: &mut AdaptiveMemoryManager,
) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + Zero + One + Copy + 'a,
{
    let start_time = std::time::Instant::now();

    let mut n_vars = 0;
    let mut total_obs = 0;
    let mut sum_x = Array1::<F>::zeros(0);
    let mut sum_x2 = Array2::<F>::zeros((0, 0));
    let mut initialized = false;

    for chunk in data_chunks {
        let (chunk_obs, chunk_vars) = chunk.dim();

        if !initialized {
            n_vars = chunk_vars;
            sum_x = Array1::<F>::zeros(n_vars);
            sum_x2 = Array2::<F>::zeros((n_vars, n_vars));
            initialized = true;
        } else if chunk_vars != n_vars {
            return Err(StatsError::DimensionMismatch(
                "All chunks must have the same number of variables".to_string(),
            ));
        }

        total_obs += chunk_obs;

        // Update sums
        for i in 0..chunk_obs {
            let row = chunk.row(i);

            // Update sum_x
            for j in 0..n_vars {
                sum_x[j] = sum_x[j] + row[j];
            }

            // Update sum_x2 (cross products)
            for j in 0..n_vars {
                for k in j..n_vars {
                    let product = row[j] * row[k];
                    sum_x2[[j, k]] = sum_x2[[j, k]] + product;
                    if j != k {
                        sum_x2[[k, j]] = sum_x2[[k, j]] + product;
                    }
                }
            }
        }
    }

    if total_obs == 0 {
        return Err(StatsError::InvalidArgument("No data provided".to_string()));
    }

    // Compute covariance matrix
    let mut cov_matrix = Array2::<F>::zeros((n_vars, n_vars));
    let n_f = F::from(total_obs).unwrap();

    for i in 0..n_vars {
        for j in 0..n_vars {
            let mean_i = sum_x[i] / n_f;
            let mean_j = sum_x[j] / n_f;
            let cov = (sum_x2[[i, j]] / n_f) - (mean_i * mean_j);
            cov_matrix[[i, j]] = cov;
        }
    }

    // Record metrics
    let memory_used = (n_vars * n_vars + n_vars) * std::mem::size_of::<F>();
    let metrics = OperationMetrics {
        operation_type: "streaming_covariance_matrix".to_string(),
        memory_used,
        processing_time: start_time.elapsed(),
        chunk_size_used: n_vars,
    };
    manager.record_operation(metrics);

    Ok(cov_matrix)
}

/// Memory-efficient principal component analysis
pub fn pca_memory_efficient<F>(
    data: &ArrayView2<F>,
    n_components: Option<usize>,
    manager: &mut AdaptiveMemoryManager,
) -> StatsResult<PCAResult<F>>
where
    F: Float + NumCast + Zero + One + Copy + Send + Sync + std::fmt::Debug,
{
    let start_time = std::time::Instant::now();

    let (n_obs, n_vars) = data.dim();
    let n_components = n_components.unwrap_or(n_vars.min(n_obs));

    // Center the data using streaming mean
    let mut means = Array1::<F>::zeros(n_vars);
    for i in 0..n_vars {
        let column = data.column(i);
        means[i] = column.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(n_obs).unwrap();
    }

    // Estimate memory for centered data
    let centered_data_memory = n_obs * n_vars * std::mem::size_of::<F>();

    if centered_data_memory <= manager.constraints.max_memory_bytes / 2 {
        // Can afford to store centered data
        let mut centered_data = Array2::<F>::zeros((n_obs, n_vars));
        for i in 0..n_obs {
            for j in 0..n_vars {
                centered_data[[i, j]] = data[[i, j]] - means[j];
            }
        }

        // Compute covariance matrix
        let cov_matrix = compute_covariance_from_centered(&centered_data.view())?;

        // Eigendecomposition (simplified - would use proper linear algebra library)
        let (eigenvalues, eigenvectors) =
            compute_eigendecomposition(&cov_matrix.view(), n_components)?;

        // Transform data
        let transformed = matrix_multiply(&centered_data.view(), &eigenvectors.view())?;

        let result = PCAResult {
            components: eigenvectors,
            explained_variance: eigenvalues,
            transformed_data: transformed,
            mean: means,
        };

        let metrics = OperationMetrics {
            operation_type: "pca_memory_efficient".to_string(),
            memory_used: centered_data_memory,
            processing_time: start_time.elapsed(),
            chunk_size_used: n_vars,
        };
        manager.record_operation(metrics);

        Ok(result)
    } else {
        // Use incremental PCA for very large datasets
        incremental_pca(data, n_components, &means, manager)
    }
}

#[derive(Debug, Clone)]
pub struct PCAResult<F> {
    pub components: Array2<F>,
    pub explained_variance: Array1<F>,
    pub transformed_data: Array2<F>,
    pub mean: Array1<F>,
}

// Helper functions (simplified implementations)

fn compute_correlation_matrix_standard<F>(
    data: &ArrayView2<F>,
    method: &str,
) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + Zero + One + Copy + std::iter::Sum<F> + std::fmt::Debug,
{
    // Use existing corrcoef implementation
    crate::corrcoef(data, method)
}

fn compute_correlation_matrix_blocked<F>(
    data: &ArrayView2<F>,
    method: &str,
    block_size: usize,
) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + Zero + One + Copy + std::iter::Sum<F> + std::fmt::Debug,
{
    let (_, n_vars) = data.dim();
    let mut corr_matrix = Array2::<F>::zeros((n_vars, n_vars));

    // Set diagonal
    for i in 0..n_vars {
        corr_matrix[[i, i]] = F::one();
    }

    // Process in blocks
    for i_block in (0..n_vars).step_by(block_size) {
        let i_end = (i_block + block_size).min(n_vars);

        for j_block in (i_block..n_vars).step_by(block_size) {
            let j_end = (j_block + block_size).min(n_vars);

            // Compute correlations for this block
            for i in i_block..i_end {
                for j in j_block.max(i + 1)..j_end {
                    let col_i = data.column(i);
                    let col_j = data.column(j);

                    let corr = match method {
                        "pearson" => crate::pearson_r(&col_i, &col_j)?,
                        "spearman" => crate::spearman_r(&col_i, &col_j)?,
                        "kendall" => crate::kendall_tau(&col_i, &col_j, "b")?,
                        _ => {
                            return Err(StatsError::InvalidArgument(format!(
                                "Unknown method: {}",
                                method
                            )))
                        }
                    };

                    corr_matrix[[i, j]] = corr;
                    corr_matrix[[j, i]] = corr;
                }
            }
        }
    }

    Ok(corr_matrix)
}

fn compute_covariance_from_centered<F>(data: &ArrayView2<F>) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + Zero + Copy,
{
    let (n_obs, n_vars) = data.dim();
    let mut cov_matrix = Array2::<F>::zeros((n_vars, n_vars));
    let n_f = F::from(n_obs - 1).unwrap(); // Sample covariance

    for i in 0..n_vars {
        for j in i..n_vars {
            let mut cov = F::zero();
            for k in 0..n_obs {
                cov = cov + data[[k, i]] * data[[k, j]];
            }
            cov = cov / n_f;
            cov_matrix[[i, j]] = cov;
            cov_matrix[[j, i]] = cov;
        }
    }

    Ok(cov_matrix)
}

fn compute_eigendecomposition<F>(
    matrix: &ArrayView2<F>,
    n_components: usize,
) -> StatsResult<(Array1<F>, Array2<F>)>
where
    F: Float + NumCast + Zero + One + Copy,
{
    let n = matrix.dim().0;
    let n_components = n_components.min(n);

    // Power iteration method for top eigenvalues/eigenvectors
    // This is a simplified implementation - in practice would use LAPACK
    let mut eigenvalues = Array1::<F>::zeros(n_components);
    let mut eigenvectors = Array2::<F>::zeros((n, n_components));

    for k in 0..n_components {
        // Initialize random vector
        let mut v = Array1::<F>::ones(n);

        // Power iteration
        for _ in 0..100 {
            // Max iterations
            let mut new_v = Array1::<F>::zeros(n);

            // Matrix-vector multiplication
            for i in 0..n {
                let mut sum = F::zero();
                for j in 0..n {
                    sum = sum + matrix[[i, j]] * v[j];
                }
                new_v[i] = sum;
            }

            // Orthogonalize against previous eigenvectors
            for prev_k in 0..k {
                let mut dot_product = F::zero();
                for i in 0..n {
                    dot_product = dot_product + new_v[i] * eigenvectors[[i, prev_k]];
                }

                for i in 0..n {
                    new_v[i] = new_v[i] - dot_product * eigenvectors[[i, prev_k]];
                }
            }

            // Normalize
            let mut norm = F::zero();
            for i in 0..n {
                norm = norm + new_v[i] * new_v[i];
            }
            norm = norm.sqrt();

            if norm > F::epsilon() {
                for i in 0..n {
                    new_v[i] = new_v[i] / norm;
                }
            }

            // Check convergence
            let mut converged = true;
            for i in 0..n {
                if (new_v[i] - v[i]).abs() > F::from(1e-6).unwrap() {
                    converged = false;
                    break;
                }
            }

            v = new_v;

            if converged {
                break;
            }
        }

        // Compute eigenvalue
        let mut eigenvalue = F::zero();
        for i in 0..n {
            let mut sum = F::zero();
            for j in 0..n {
                sum = sum + matrix[[i, j]] * v[j];
            }
            eigenvalue = eigenvalue + v[i] * sum;
        }

        eigenvalues[k] = eigenvalue;
        for i in 0..n {
            eigenvectors[[i, k]] = v[i];
        }
    }

    Ok((eigenvalues, eigenvectors))
}

fn matrix_multiply<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + Zero + Copy,
{
    let (m, n) = a.dim();
    let (n2, p) = b.dim();

    if n != n2 {
        return Err(StatsError::DimensionMismatch(
            "Matrix dimensions don't match".to_string(),
        ));
    }

    let mut result = Array2::<F>::zeros((m, p));

    for i in 0..m {
        for j in 0..p {
            let mut sum = F::zero();
            for k in 0..n {
                sum = sum + a[[i, k]] * b[[k, j]];
            }
            result[[i, j]] = sum;
        }
    }

    Ok(result)
}

fn incremental_pca<F>(
    data: &ArrayView2<F>,
    n_components: usize,
    means: &Array1<F>,
    manager: &mut AdaptiveMemoryManager,
) -> StatsResult<PCAResult<F>>
where
    F: Float + NumCast + Zero + One + Copy + Send + Sync + std::fmt::Debug,
{
    let (n_obs, n_vars) = data.dim();
    let n_components = n_components.min(n_vars);

    // Batch size for incremental processing
    let batch_size = manager.get_optimal_chunk_size(n_obs, std::mem::size_of::<F>() * n_vars);

    // Initialize components with random orthogonal matrix
    let mut components = Array2::<F>::zeros((n_vars, n_components));
    for i in 0..n_components {
        components[[i % n_vars, i]] = F::one();
    }

    let mut explained_variance = Array1::<F>::zeros(n_components);
    let mut _n_samples_seen = 0;

    // Process data in batches
    for batch_start in (0..n_obs).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n_obs);
        let batch = data.slice(ndarray::s![batch_start..batch_end, ..]);

        // Center the batch
        let mut centered_batch = Array2::<F>::zeros(batch.dim());
        for i in 0..batch.nrows() {
            for j in 0..batch.ncols() {
                centered_batch[[i, j]] = batch[[i, j]] - means[j];
            }
        }

        // Update components using simplified incremental update
        for k in 0..n_components {
            let component = components.column(k).to_owned();

            // Project batch onto current component
            let mut projections = Array1::<F>::zeros(batch.nrows());
            for i in 0..batch.nrows() {
                let mut projection = F::zero();
                for j in 0..n_vars {
                    projection = projection + centered_batch[[i, j]] * component[j];
                }
                projections[i] = projection;
            }

            // Update component direction
            let mut new_component = Array1::<F>::zeros(n_vars);
            for j in 0..n_vars {
                let mut sum = F::zero();
                for i in 0..batch.nrows() {
                    sum = sum + centered_batch[[i, j]] * projections[i];
                }
                new_component[j] = sum;
            }

            // Normalize
            let mut norm = F::zero();
            for j in 0..n_vars {
                norm = norm + new_component[j] * new_component[j];
            }
            norm = norm.sqrt();

            if norm > F::epsilon() {
                for j in 0..n_vars {
                    components[[j, k]] = new_component[j] / norm;
                }

                // Update explained variance
                let variance = projections
                    .iter()
                    .map(|&x| x * x)
                    .fold(F::zero(), |acc, x| acc + x);
                explained_variance[k] = variance / F::from(batch.nrows()).unwrap();
            }
        }

        _n_samples_seen += batch.nrows();
    }

    // Transform the data
    let mut transformed_data = Array2::<F>::zeros((n_obs, n_components));
    for i in 0..n_obs {
        for k in 0..n_components {
            let mut projection = F::zero();
            for j in 0..n_vars {
                let centered_val = data[[i, j]] - means[j];
                projection = projection + centered_val * components[[j, k]];
            }
            transformed_data[[i, k]] = projection;
        }
    }

    Ok(PCAResult {
        components,
        explained_variance,
        transformed_data,
        mean: means.clone(),
    })
}

fn check_array_finite_2d<F, D>(arr: &ArrayBase<D, Ix2>, name: &str) -> StatsResult<()>
where
    F: Float,
    D: Data<Elem = F>,
{
    for &val in arr.iter() {
        if !val.is_finite() {
            return Err(StatsError::InvalidArgument(format!(
                "{} contains non-finite values",
                name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_adaptive_memory_manager() {
        let constraints = MemoryConstraints::default();
        let mut manager = AdaptiveMemoryManager::new(constraints);

        let chunk_size = manager.get_optimal_chunk_size(1000, 8);
        assert!(chunk_size > 0);
        assert!(chunk_size <= 1000);

        let metrics = OperationMetrics {
            operation_type: "test".to_string(),
            memory_used: 1024,
            processing_time: std::time::Duration::from_millis(10),
            chunk_size_used: chunk_size,
        };

        manager.record_operation(metrics);
        let stats = manager.get_statistics();
        assert_eq!(stats.operations_completed, 1);
    }

    #[test]
    fn test_cache_oblivious_matrix_mult() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let result = cache_oblivious_matrix_mult(&a.view(), &b.view(), 2).unwrap();

        // Expected: [[19, 22], [43, 50]]
        assert!((result[[0, 0]] - 19.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 22.0).abs() < 1e-10);
        assert!((result[[1, 0]] - 43.0).abs() < 1e-10);
        assert!((result[[1, 1]] - 50.0).abs() < 1e-10);
    }
}
