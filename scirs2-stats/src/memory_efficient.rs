//! Memory-efficient statistical operations
//!
//! This module provides memory-optimized implementations of statistical functions
//! that minimize allocations and use streaming/chunked processing for large datasets.

use crate::error::{StatsError, StatsResult};
use ndarray::{ArrayBase, ArrayViewMut1, Data, Ix1, Ix2, s};
use num_traits::{Float, NumCast};
use std::cmp::Ordering;

/// Chunk size for streaming operations (tuned for cache efficiency)
const CHUNK_SIZE: usize = 8192;

/// Streaming mean calculation that processes data in chunks
///
/// This function computes the mean without loading the entire dataset into memory
/// at once, making it suitable for very large datasets.
///
/// # Arguments
///
/// * `data_iter` - Iterator over data chunks
/// * `total_count` - Total number of elements across all chunks
///
/// # Returns
///
/// The arithmetic mean
pub fn streaming_mean<F, I>(mut data_iter: I, total_count: usize) -> StatsResult<F>
where
    F: Float + NumCast,
    I: Iterator<Item = F>,
{
    if total_count == 0 {
        return Err(StatsError::invalid_argument("Cannot compute mean of empty dataset"));
    }

    let mut sum = F::zero();
    let mut count = 0;

    // Process in chunks to maintain precision
    while count < total_count {
        let chunk_sum = data_iter
            .by_ref()
            .take(CHUNK_SIZE)
            .fold(F::zero(), |acc, val| acc + val);
        
        sum = sum + chunk_sum;
        count += CHUNK_SIZE.min(total_count - count);
    }

    Ok(sum / F::from(total_count).unwrap())
}

/// Welford's online algorithm for variance computation
///
/// This algorithm computes variance in a single pass with minimal memory usage
/// and improved numerical stability.
///
/// # Arguments
///
/// * `x` - Input data array
/// * `ddof` - Delta degrees of freedom
///
/// # Returns
///
/// * Tuple of (mean, variance)
pub fn welford_variance<F, D>(x: &ArrayBase<D, Ix1>, ddof: usize) -> StatsResult<(F, F)>
where
    F: Float + NumCast,
    D: Data<Elem = F>,
{
    let n = x.len();
    if n <= ddof {
        return Err(StatsError::invalid_argument(
            "Not enough data points for the given degrees of freedom"
        ));
    }

    let mut mean = F::zero();
    let mut m2 = F::zero();
    let mut count = 0;

    for &value in x.iter() {
        count += 1;
        let delta = value - mean;
        mean = mean + delta / F::from(count).unwrap();
        let delta2 = value - mean;
        m2 = m2 + delta * delta2;
    }

    let variance = m2 / F::from(n - ddof).unwrap();
    Ok((mean, variance))
}

/// In-place normalization (standardization) of data
///
/// This function normalizes data in-place to have zero mean and unit variance,
/// avoiding the need for additional memory allocation.
///
/// # Arguments
///
/// * `data` - Mutable array to normalize
/// * `ddof` - Delta degrees of freedom for variance calculation
pub fn normalize_inplace<F>(data: &mut ArrayViewMut1<F>, ddof: usize) -> StatsResult<()>
where
    F: Float + NumCast,
{
    let (mean, variance) = welford_variance(&data.view(), ddof)?;
    
    if variance <= F::epsilon() {
        return Err(StatsError::invalid_argument(
            "Cannot normalize data with zero variance"
        ));
    }

    let std_dev = variance.sqrt();
    
    // Normalize in-place
    for val in data.iter_mut() {
        *val = (*val - mean) / std_dev;
    }

    Ok(())
}

/// Memory-efficient quantile computation using quickselect
///
/// This function computes quantiles without fully sorting the array,
/// which saves memory and time for large datasets.
///
/// # Arguments
///
/// * `data` - Input data array (will be partially reordered)
/// * `q` - Quantile to compute (0 to 1)
///
/// # Returns
///
/// The computed quantile value
pub fn quantile_quickselect<F>(data: &mut [F], q: F) -> StatsResult<F>
where
    F: Float + NumCast,
{
    if data.is_empty() {
        return Err(StatsError::invalid_argument("Cannot compute quantile of empty array"));
    }

    if q < F::zero() || q > F::one() {
        return Err(StatsError::domain("Quantile must be between 0 and 1"));
    }

    let n = data.len();
    let pos = q * F::from(n - 1).unwrap();
    let k = NumCast::from(pos.floor()).unwrap();
    
    // Use quickselect to find k-th element
    quickselect(data, k);
    
    let lower = data[k];
    
    // Handle interpolation if needed
    let frac = pos - pos.floor();
    if frac > F::zero() && k + 1 < n {
        quickselect(&mut data[(k + 1)..], 0);
        let upper = data[k + 1];
        Ok(lower + frac * (upper - lower))
    } else {
        Ok(lower)
    }
}

/// Quickselect algorithm for finding k-th smallest element
fn quickselect<F: Float>(data: &mut [F], k: usize) {
    let len = data.len();
    if len <= 1 {
        return;
    }

    let mut left = 0;
    let mut right = len - 1;

    while left < right {
        let pivot_idx = partition(data, left, right);
        
        match k.cmp(&pivot_idx) {
            Ordering::Less => right = pivot_idx - 1,
            Ordering::Greater => left = pivot_idx + 1,
            Ordering::Equal => return,
        }
    }
}

/// Partition function for quickselect
fn partition<F: Float>(data: &mut [F], left: usize, right: usize) -> usize {
    let pivot_idx = left + (right - left) / 2;
    let pivot = data[pivot_idx];
    
    data.swap(pivot_idx, right);
    
    let mut store_idx = left;
    for i in left..right {
        if data[i] < pivot {
            data.swap(i, store_idx);
            store_idx += 1;
        }
    }
    
    data.swap(store_idx, right);
    store_idx
}

/// Memory-efficient covariance matrix computation
///
/// Computes covariance matrix using a streaming algorithm that processes
/// data in chunks to minimize memory usage.
///
/// # Arguments
///
/// * `data` - 2D array where columns are variables
/// * `ddof` - Delta degrees of freedom
///
/// # Returns
///
/// Covariance matrix
pub fn covariance_chunked<F, D>(
    data: &ArrayBase<D, Ix2>,
    ddof: usize,
) -> StatsResult<ndarray::Array2<F>>
where
    F: Float + NumCast,
    D: Data<Elem = F>,
{
    let n_obs = data.nrows();
    let n_vars = data.ncols();
    
    if n_obs <= ddof {
        return Err(StatsError::invalid_argument(
            "Not enough observations for the given degrees of freedom"
        ));
    }

    // Compute means for each variable
    let mut means = ndarray::Array1::zeros(n_vars);
    for j in 0..n_vars {
        let col = data.slice(s![.., j]);
        means[j] = col.iter().fold(F::zero(), |acc, &val| acc + val) / F::from(n_obs).unwrap();
    }

    // Initialize covariance matrix
    let mut cov_matrix = ndarray::Array2::zeros((n_vars, n_vars));

    // Process data in chunks to compute covariance
    let chunk_size = CHUNK_SIZE / n_vars;
    for chunk_start in (0..n_obs).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(n_obs);
        let chunk = data.slice(s![chunk_start..chunk_end, ..]);

        // Update covariance for this chunk
        for i in 0..n_vars {
            for j in i..n_vars {
                let mut sum = F::zero();
                for k in 0..chunk.nrows() {
                    let xi = chunk[(k, i)] - means[i];
                    let xj = chunk[(k, j)] - means[j];
                    sum = sum + xi * xj;
                }
                cov_matrix[(i, j)] = cov_matrix[(i, j)] + sum;
            }
        }
    }

    // Normalize and fill symmetric entries
    let factor = F::from(n_obs - ddof).unwrap();
    for i in 0..n_vars {
        for j in i..n_vars {
            cov_matrix[(i, j)] = cov_matrix[(i, j)] / factor;
            if i != j {
                cov_matrix[(j, i)] = cov_matrix[(i, j)];
            }
        }
    }

    Ok(cov_matrix)
}

/// Memory-efficient histogram computation
///
/// Computes histogram without storing all data, using a streaming approach.
pub struct StreamingHistogram<F: Float> {
    bins: Vec<F>,
    counts: Vec<usize>,
    min_val: F,
    max_val: F,
    total_count: usize,
}

impl<F: Float + NumCast> StreamingHistogram<F> {
    /// Create a new streaming histogram
    pub fn new(n_bins: usize, min_val: F, max_val: F) -> Self {
        let bin_width = (max_val - min_val) / F::from(n_bins).unwrap();
        let bins: Vec<F> = (0..=n_bins)
            .map(|i| min_val + F::from(i).unwrap() * bin_width)
            .collect();
        
        Self {
            bins,
            counts: vec![0; n_bins],
            min_val,
            max_val,
            total_count: 0,
        }
    }

    /// Add a value to the histogram
    pub fn add_value(&mut self, value: F) {
        if value >= self.min_val && value <= self.max_val {
            let n_bins = self.counts.len();
            let bin_width = (self.max_val - self.min_val) / F::from(n_bins).unwrap();
            let bin_idx = ((value - self.min_val) / bin_width).floor();
            let bin_idx: usize = NumCast::from(bin_idx).unwrap_or(n_bins - 1).min(n_bins - 1);
            self.counts[bin_idx] += 1;
            self.total_count += 1;
        }
    }

    /// Add multiple values
    pub fn add_values<D>(&mut self, values: &ArrayBase<D, Ix1>)
    where
        D: Data<Elem = F>,
    {
        for &value in values.iter() {
            self.add_value(value);
        }
    }

    /// Get the histogram results
    pub fn get_histogram(&self) -> (Vec<F>, Vec<usize>) {
        (self.bins.clone(), self.counts.clone())
    }

    /// Get normalized histogram (density)
    pub fn get_density(&self) -> (Vec<F>, Vec<F>) {
        let n_bins = self.counts.len();
        let bin_width = (self.max_val - self.min_val) / F::from(n_bins).unwrap();
        let total = F::from(self.total_count).unwrap() * bin_width;
        
        let density: Vec<F> = self.counts
            .iter()
            .map(|&count| F::from(count).unwrap() / total)
            .collect();
        
        (self.bins.clone(), density)
    }
}