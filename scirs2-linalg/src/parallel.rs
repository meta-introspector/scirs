//! Parallel processing utilities for linear algebra operations
//!
//! This module provides utilities for managing worker threads across various
//! linear algebra operations, ensuring consistent behavior and optimal performance.

use std::sync::Mutex;

/// Global worker configuration
static GLOBAL_WORKERS: Mutex<Option<usize>> = Mutex::new(None);

/// Set the global worker thread count for all operations
///
/// This affects operations that don't explicitly specify a worker count.
/// If set to None, operations will use system defaults.
///
/// # Arguments
///
/// * `workers` - Number of worker threads (None = use system default)
///
/// # Examples
///
/// ```
/// use scirs2_linalg::parallel::set_global_workers;
///
/// // Use 4 threads for all operations
/// set_global_workers(Some(4));
///
/// // Reset to system default
/// set_global_workers(None);
/// ```
pub fn set_global_workers(workers: Option<usize>) {
    if let Ok(mut global) = GLOBAL_WORKERS.lock() {
        *global = workers;

        // Set OpenMP environment variable if specified
        if let Some(num_workers) = workers {
            std::env::set_var("OMP_NUM_THREADS", num_workers.to_string());
        } else {
            // Remove the environment variable to use system default
            std::env::remove_var("OMP_NUM_THREADS");
        }
    }
}

/// Get the current global worker thread count
///
/// # Returns
///
/// * Current global worker count (None = system default)
pub fn get_global_workers() -> Option<usize> {
    GLOBAL_WORKERS.lock().ok().and_then(|global| *global)
}

/// Configure worker threads for an operation
///
/// This function determines the appropriate number of worker threads to use,
/// considering both the operation-specific setting and global configuration.
///
/// # Arguments
///
/// * `workers` - Operation-specific worker count
///
/// # Returns
///
/// * Effective worker count to use
pub fn configure_workers(workers: Option<usize>) -> Option<usize> {
    match workers {
        Some(count) => {
            // Operation-specific setting takes precedence
            std::env::set_var("OMP_NUM_THREADS", count.to_string());
            Some(count)
        }
        None => {
            // Use global setting if available
            let global_workers = get_global_workers();
            if let Some(count) = global_workers {
                std::env::set_var("OMP_NUM_THREADS", count.to_string());
            }
            global_workers
        }
    }
}

/// Worker configuration for batched operations
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Number of worker threads
    pub workers: Option<usize>,
    /// Threshold for using parallel processing
    pub parallel_threshold: usize,
    /// Chunk size for batched operations
    pub chunk_size: usize,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            workers: None,
            parallel_threshold: 1000,
            chunk_size: 64,
        }
    }
}

impl WorkerConfig {
    /// Create a new worker configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of worker threads
    pub fn with_workers(mut self, workers: usize) -> Self {
        self.workers = Some(workers);
        self
    }

    /// Set the parallel processing threshold
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Set the chunk size for batched operations
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Apply this configuration for the current operation
    pub fn apply(&self) {
        configure_workers(self.workers);
    }
}

/// Scoped worker configuration
///
/// Temporarily sets worker configuration and restores the previous
/// configuration when dropped.
pub struct ScopedWorkers {
    previous_workers: Option<usize>,
}

impl ScopedWorkers {
    /// Create a scoped worker configuration
    ///
    /// # Arguments
    ///
    /// * `workers` - Number of worker threads for this scope
    ///
    /// # Returns
    ///
    /// * ScopedWorkers guard that restores previous configuration on drop
    pub fn new(workers: Option<usize>) -> Self {
        let previous_workers = get_global_workers();
        set_global_workers(workers);
        Self { previous_workers }
    }
}

impl Drop for ScopedWorkers {
    fn drop(&mut self) {
        set_global_workers(self.previous_workers);
    }
}

/// Parallel iterator utilities for matrix operations
pub mod iter {
    use rayon::prelude::*;

    /// Process chunks of work in parallel
    ///
    /// # Arguments
    ///
    /// * `items` - Items to process
    /// * `chunk_size` - Size of each chunk
    /// * `f` - Function to apply to each chunk
    ///
    /// # Returns
    ///
    /// * Vector of results from each chunk
    pub fn parallel_chunks<T, R, F>(items: &[T], chunk_size: usize, f: F) -> Vec<R>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(&[T]) -> R + Send + Sync,
    {
        items
            .chunks(chunk_size)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(f)
            .collect()
    }

    /// Process items in parallel with index information
    ///
    /// # Arguments
    ///
    /// * `items` - Items to process
    /// * `f` - Function to apply to each (index, item) pair
    ///
    /// # Returns
    ///
    /// * Vector of results
    pub fn parallel_enumerate<T, R, F>(items: &[T], f: F) -> Vec<R>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(usize, &T) -> R + Send + Sync,
    {
        items
            .par_iter()
            .enumerate()
            .map(|(i, item)| f(i, item))
            .collect()
    }
}

/// Adaptive algorithm selection based on data size and worker configuration
pub mod adaptive {
    use super::WorkerConfig;

    /// Algorithm selection strategy
    #[derive(Debug, Clone, Copy)]
    pub enum Strategy {
        /// Always use serial processing
        Serial,
        /// Always use parallel processing
        Parallel,
        /// Automatically choose based on data size
        Adaptive,
    }

    /// Choose processing strategy based on data size and configuration
    ///
    /// # Arguments
    ///
    /// * `data_size` - Size of the data to process
    /// * `config` - Worker configuration
    ///
    /// # Returns
    ///
    /// * Recommended processing strategy
    pub fn choose_strategy(data_size: usize, config: &WorkerConfig) -> Strategy {
        if data_size < config.parallel_threshold {
            Strategy::Serial
        } else {
            Strategy::Parallel
        }
    }

    /// Check if parallel processing is recommended
    ///
    /// # Arguments
    ///
    /// * `data_size` - Size of the data to process
    /// * `config` - Worker configuration
    ///
    /// # Returns
    ///
    /// * true if parallel processing is recommended
    pub fn should_use_parallel(data_size: usize, config: &WorkerConfig) -> bool {
        matches!(choose_strategy(data_size, config), Strategy::Parallel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_workers() {
        // Save initial state to restore later
        let original_state = get_global_workers();

        // Test setting and getting global workers
        set_global_workers(Some(4));
        assert_eq!(get_global_workers(), Some(4));

        set_global_workers(None);
        assert_eq!(get_global_workers(), None);

        // Restore original state to avoid test interference
        set_global_workers(original_state);
    }

    #[test]
    fn test_scoped_workers() {
        // Save initial state to restore later
        let original_state = get_global_workers();

        // Set initial global workers
        set_global_workers(Some(2));

        {
            // Create scoped configuration
            let _scoped = ScopedWorkers::new(Some(8));
            assert_eq!(get_global_workers(), Some(8));
        }

        // Should be restored after scope
        assert_eq!(get_global_workers(), Some(2));

        // Restore original state to avoid test interference
        set_global_workers(original_state);
    }

    #[test]
    fn test_worker_config() {
        let config = WorkerConfig::new()
            .with_workers(4)
            .with_threshold(2000)
            .with_chunk_size(128);

        assert_eq!(config.workers, Some(4));
        assert_eq!(config.parallel_threshold, 2000);
        assert_eq!(config.chunk_size, 128);
    }

    #[test]
    fn test_adaptive_strategy() {
        let config = WorkerConfig::default();

        // Small data should use serial
        assert!(matches!(
            adaptive::choose_strategy(100, &config),
            adaptive::Strategy::Serial
        ));

        // Large data should use parallel
        assert!(matches!(
            adaptive::choose_strategy(2000, &config),
            adaptive::Strategy::Parallel
        ));
    }
}
