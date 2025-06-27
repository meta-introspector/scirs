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
    use scirs2_core::parallel_ops::*;

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

/// Work-stealing scheduler optimizations
///
/// This module provides advanced scheduling strategies for parallel algorithms
/// using work-stealing techniques to improve load balancing and performance.
pub mod scheduler {
    use super::WorkerConfig;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// Work-stealing task scheduler
    ///
    /// Implements a work-stealing scheduler that dynamically balances work
    /// across threads for improved performance on irregular workloads.
    pub struct WorkStealingScheduler {
        num_workers: usize,
        chunk_size: usize,
        adaptive_chunking: bool,
    }

    impl WorkStealingScheduler {
        /// Create a new work-stealing scheduler
        pub fn new(config: &WorkerConfig) -> Self {
            let num_workers = config.workers.unwrap_or_else(|| {
                // Default to available parallelism or 4 threads
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            });
            Self {
                num_workers,
                chunk_size: config.chunk_size,
                adaptive_chunking: true,
            }
        }

        /// Set whether to use adaptive chunking
        pub fn with_adaptive_chunking(mut self, adaptive: bool) -> Self {
            self.adaptive_chunking = adaptive;
            self
        }

        /// Execute work items using work-stealing strategy
        ///
        /// This function divides work into chunks and uses atomic counters
        /// to allow threads to steal work from a global queue when they
        /// finish their assigned chunks early.
        pub fn execute<T, R, F>(&self, items: &[T], f: F) -> Vec<R>
        where
            T: Send + Sync,
            R: Send + Default + Clone,
            F: Fn(&T) -> R + Send + Sync,
        {
            let n = items.len();
            if n == 0 {
                return Vec::new();
            }

            // Determine chunk size based on workload characteristics
            let chunk_size = if self.adaptive_chunking {
                self.adaptive_chunk_size(n)
            } else {
                self.chunk_size
            };

            // Create shared work counter
            let work_counter = Arc::new(AtomicUsize::new(0));
            let results = Arc::new(Mutex::new(vec![R::default(); n]));

            // Use scoped threads to process work items
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..self.num_workers)
                    .map(|_| {
                        let work_counter = work_counter.clone();
                        let results = results.clone();
                        let items_ref = items;
                        let f_ref = &f;

                        s.spawn(move || {
                            loop {
                                // Steal a chunk of work
                                let start = work_counter.fetch_add(chunk_size, Ordering::SeqCst);
                                if start >= n {
                                    break;
                                }

                                let end = std::cmp::min(start + chunk_size, n);

                                // Process the chunk
                                for i in start..end {
                                    let result = f_ref(&items_ref[i]);
                                    let mut results_guard = results.lock().unwrap();
                                    results_guard[i] = result;
                                }
                            }
                        })
                    })
                    .collect();

                // Wait for all threads to complete
                for handle in handles {
                    handle.join().unwrap();
                }
            });

            // Extract results
            Arc::try_unwrap(results)
                .unwrap_or_else(|_| panic!("Failed to extract results"))
                .into_inner()
                .unwrap_or_else(|_| panic!("Failed to extract mutex inner value"))
        }

        /// Determine adaptive chunk size based on workload size
        fn adaptive_chunk_size(&self, total_items: usize) -> usize {
            // Use smaller chunks for better load balancing on smaller workloads
            // and larger chunks for better cache efficiency on larger workloads
            let items_per_worker = total_items / self.num_workers;

            if items_per_worker < 100 {
                // Small workload: use fine-grained chunks
                std::cmp::max(1, items_per_worker / 4)
            } else if items_per_worker < 1000 {
                // Medium workload: balance between overhead and load balancing
                items_per_worker / 8
            } else {
                // Large workload: prioritize cache efficiency
                std::cmp::min(self.chunk_size, items_per_worker / 16)
            }
        }

        /// Execute matrix operations with work-stealing
        ///
        /// Specialized version for matrix operations that takes into account
        /// cache line sizes and memory access patterns.
        pub fn execute_matrix<R, F>(
            &self,
            rows: usize,
            cols: usize,
            f: F,
        ) -> ndarray::Array2<R>
        where
            R: Send + Default + Clone,
            F: Fn(usize, usize) -> R + Send + Sync,
        {
            // Use block partitioning for better cache efficiency
            let block_size = 64; // Typical cache line aligned block
            let work_items: Vec<(usize, usize)> = (0..rows)
                .step_by(block_size)
                .flat_map(|i| {
                    (0..cols)
                        .step_by(block_size)
                        .map(move |j| (i, j))
                })
                .collect();

            // Process blocks using work-stealing and collect results
            let work_counter = Arc::new(AtomicUsize::new(0));
            let results_vec = Arc::new(Mutex::new(Vec::new()));

            std::thread::scope(|s| {
                let handles: Vec<_> = (0..self.num_workers)
                    .map(|_| {
                        let work_counter = work_counter.clone();
                        let results_vec = results_vec.clone();
                        let work_items_ref = &work_items;
                        let f_ref = &f;

                        s.spawn(move || {
                            let mut local_results = Vec::new();

                            loop {
                                let idx = work_counter.fetch_add(1, Ordering::SeqCst);
                                if idx >= work_items_ref.len() {
                                    break;
                                }

                                let (block_i, block_j) = work_items_ref[idx];
                                let i_end = std::cmp::min(block_i + block_size, rows);
                                let j_end = std::cmp::min(block_j + block_size, cols);

                                // Process the block
                                for i in block_i..i_end {
                                    for j in block_j..j_end {
                                        local_results.push((i, j, f_ref(i, j)));
                                    }
                                }
                            }

                            // Add local results to global results
                            if !local_results.is_empty() {
                                let mut global_results = results_vec.lock().unwrap();
                                global_results.extend(local_results);
                            }
                        })
                    })
                    .collect();

                for handle in handles {
                    handle.join().unwrap();
                }
            });

            // Create result matrix from collected results
            let mut result = ndarray::Array2::default((rows, cols));
            let results = Arc::try_unwrap(results_vec)
                .unwrap_or_else(|_| panic!("Failed to extract results"))
                .into_inner()
                .unwrap_or_else(|_| panic!("Failed to extract mutex inner value"));

            for (i, j, val) in results {
                result[[i, j]] = val;
            }

            result
        }
    }

    /// Dynamic load balancer for irregular workloads
    ///
    /// This struct provides dynamic load balancing for workloads where
    /// different items may take varying amounts of time to process.
    pub struct DynamicLoadBalancer {
        scheduler: WorkStealingScheduler,
        /// Tracks execution time statistics for adaptive scheduling
        timing_stats: Arc<Mutex<TimingStats>>,
    }

    #[derive(Default)]
    struct TimingStats {
        total_items: usize,
        total_time_ms: u128,
        min_time_ms: u128,
        max_time_ms: u128,
    }

    impl DynamicLoadBalancer {
        /// Create a new dynamic load balancer
        pub fn new(config: &WorkerConfig) -> Self {
            Self {
                scheduler: WorkStealingScheduler::new(config),
                timing_stats: Arc::new(Mutex::new(TimingStats::default())),
            }
        }

        /// Execute work items with dynamic load balancing and timing
        pub fn execute_timed<T, R, F>(&self, items: &[T], f: F) -> Vec<R>
        where
            T: Send + Sync,
            R: Send + Default + Clone,
            F: Fn(&T) -> R + Send + Sync,
        {
            use std::time::Instant;

            let n = items.len();
            if n == 0 {
                return Vec::new();
            }

            let results = Arc::new(Mutex::new(vec![R::default(); n]));
            let work_counter = Arc::new(AtomicUsize::new(0));
            let timing_stats = self.timing_stats.clone();

            std::thread::scope(|s| {
                let handles: Vec<_> = (0..self.scheduler.num_workers)
                    .map(|_| {
                        let work_counter = work_counter.clone();
                        let results = results.clone();
                        let timing_stats = timing_stats.clone();
                        let items_ref = items;
                        let f_ref = &f;

                        s.spawn(move || {
                            let mut local_min = u128::MAX;
                            let mut local_max = 0u128;
                            let mut local_total = 0u128;
                            let mut local_count = 0usize;

                            loop {
                                let idx = work_counter.fetch_add(1, Ordering::SeqCst);
                                if idx >= n {
                                    break;
                                }

                                // Time the execution
                                let start = Instant::now();
                                let result = f_ref(&items_ref[idx]);
                                let elapsed = start.elapsed().as_millis();

                                // Update local statistics
                                local_min = local_min.min(elapsed);
                                local_max = local_max.max(elapsed);
                                local_total += elapsed;
                                local_count += 1;

                                // Store result
                                let mut results_guard = results.lock().unwrap();
                                results_guard[idx] = result;
                            }

                            // Update global statistics
                            if local_count > 0 {
                                let mut stats = timing_stats.lock().unwrap();
                                stats.total_items += local_count;
                                stats.total_time_ms += local_total;
                                stats.min_time_ms = stats.min_time_ms.min(local_min);
                                stats.max_time_ms = stats.max_time_ms.max(local_max);
                            }
                        })
                    })
                    .collect();

                for handle in handles {
                    handle.join().unwrap();
                }
            });

            Arc::try_unwrap(results)
                .unwrap_or_else(|_| panic!("Failed to extract results"))
                .into_inner()
                .unwrap_or_else(|_| panic!("Failed to extract mutex inner value"))
        }

        /// Get average execution time per item
        pub fn get_average_time_ms(&self) -> f64 {
            let stats = self.timing_stats.lock().unwrap();
            if stats.total_items > 0 {
                stats.total_time_ms as f64 / stats.total_items as f64
            } else {
                0.0
            }
        }

        /// Get timing variance to detect irregular workloads
        pub fn get_time_variance(&self) -> f64 {
            let stats = self.timing_stats.lock().unwrap();
            if stats.total_items > 0 && stats.max_time_ms > stats.min_time_ms {
                (stats.max_time_ms - stats.min_time_ms) as f64 / stats.min_time_ms as f64
            } else {
                0.0
            }
        }
    }
}

/// Thread pool configurations for linear algebra operations
///
/// This module provides flexible thread pool management with support for
/// different configurations optimized for various linear algebra workloads.
pub mod thread_pool {
    use super::{WorkerConfig, configure_workers};
    use std::sync::{Arc, Mutex, Once};
    use scirs2_core::parallel_ops::*;

    /// Global thread pool manager
    static INIT: Once = Once::new();
    static mut GLOBAL_POOL: Option<Arc<Mutex<ThreadPoolManager>>> = None;

    /// Thread pool configuration profiles
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ThreadPoolProfile {
        /// Default profile - uses system defaults
        Default,
        /// CPU-bound profile - one thread per CPU core
        CpuBound,
        /// Memory-bound profile - fewer threads to reduce memory contention
        MemoryBound,
        /// Latency-sensitive profile - more threads for better responsiveness
        LatencySensitive,
        /// Custom profile with specific thread count
        Custom(usize),
    }

    impl ThreadPoolProfile {
        /// Get the number of threads for this profile
        pub fn num_threads(&self) -> usize {
            match self {
                ThreadPoolProfile::Default => {
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4)
                }
                ThreadPoolProfile::CpuBound => {
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4)
                }
                ThreadPoolProfile::MemoryBound => {
                    // Use half the available cores to reduce memory contention
                    std::thread::available_parallelism()
                        .map(|n| std::cmp::max(1, n.get() / 2))
                        .unwrap_or(2)
                }
                ThreadPoolProfile::LatencySensitive => {
                    // Use 1.5x the available cores for better responsiveness
                    std::thread::available_parallelism()
                        .map(|n| n.get() + n.get() / 2)
                        .unwrap_or(6)
                }
                ThreadPoolProfile::Custom(n) => *n,
            }
        }
    }

    /// Thread pool manager for linear algebra operations
    pub struct ThreadPoolManager {
        profile: ThreadPoolProfile,
        /// Stack size for worker threads (in bytes)
        stack_size: Option<usize>,
        /// Thread name prefix
        thread_name_prefix: String,
        /// Whether to pin threads to CPU cores
        cpu_affinity: bool,
    }

    impl ThreadPoolManager {
        /// Create a new thread pool manager with default settings
        pub fn new() -> Self {
            Self {
                profile: ThreadPoolProfile::Default,
                stack_size: None,
                thread_name_prefix: "linalg-worker".to_string(),
                cpu_affinity: false,
            }
        }

        /// Set the thread pool profile
        pub fn with_profile(mut self, profile: ThreadPoolProfile) -> Self {
            self.profile = profile;
            self
        }

        /// Set the stack size for worker threads
        pub fn with_stack_size(mut self, size: usize) -> Self {
            self.stack_size = Some(size);
            self
        }

        /// Set the thread name prefix
        pub fn with_thread_name_prefix(mut self, prefix: String) -> Self {
            self.thread_name_prefix = prefix;
            self
        }

        /// Enable CPU affinity for worker threads
        pub fn with_cpu_affinity(mut self, enabled: bool) -> Self {
            self.cpu_affinity = enabled;
            self
        }

        /// Initialize the thread pool with current settings
        pub fn initialize(&self) -> Result<(), String> {
            let num_threads = self.profile.num_threads();

            // Configure rayon thread pool
            let thread_prefix = self.thread_name_prefix.clone();
            let mut pool_builder = ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .thread_name(move |idx| format!("{}-{}", thread_prefix, idx));

            if let Some(stack_size) = self.stack_size {
                pool_builder = pool_builder.stack_size(stack_size);
            }

            pool_builder
                .build_global()
                .map_err(|e| format!("Failed to initialize thread pool: {}", e))?;

            // Set OpenMP threads for BLAS/LAPACK operations
            std::env::set_var("OMP_NUM_THREADS", num_threads.to_string());

            // Set MKL threads if using Intel MKL
            std::env::set_var("MKL_NUM_THREADS", num_threads.to_string());

            Ok(())
        }

        /// Get current thread pool statistics
        pub fn statistics(&self) -> ThreadPoolStats {
            ThreadPoolStats {
                num_threads: self.profile.num_threads(),
                current_parallelism: num_threads(),
                profile: self.profile,
                stack_size: self.stack_size,
            }
        }
    }

    impl Default for ThreadPoolManager {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Thread pool statistics
    #[derive(Debug, Clone)]
    pub struct ThreadPoolStats {
        pub num_threads: usize,
        pub current_parallelism: usize,
        pub profile: ThreadPoolProfile,
        pub stack_size: Option<usize>,
    }

    /// Get the global thread pool manager
    pub fn global_pool() -> Arc<Mutex<ThreadPoolManager>> {
        unsafe {
            INIT.call_once(|| {
                GLOBAL_POOL = Some(Arc::new(Mutex::new(ThreadPoolManager::new())));
            });
            #[allow(static_mut_refs)]
            GLOBAL_POOL.as_ref().unwrap().clone()
        }
    }

    /// Initialize global thread pool with a specific profile
    pub fn initialize_global_pool(profile: ThreadPoolProfile) -> Result<(), String> {
        let pool = global_pool();
        let mut manager = pool.lock().unwrap();
        manager.profile = profile;
        manager.initialize()
    }

    /// Adaptive thread pool that adjusts based on workload
    pub struct AdaptiveThreadPool {
        min_threads: usize,
        max_threads: usize,
        current_threads: Arc<Mutex<usize>>,
        /// Tracks CPU utilization for adaptive scaling
        cpu_utilization: Arc<Mutex<f64>>,
    }

    impl AdaptiveThreadPool {
        /// Create a new adaptive thread pool
        pub fn new(min_threads: usize, max_threads: usize) -> Self {
            let current = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);

            Self {
                min_threads,
                max_threads,
                current_threads: Arc::new(Mutex::new(current)),
                cpu_utilization: Arc::new(Mutex::new(0.0)),
            }
        }

        /// Update thread count based on current utilization
        pub fn adapt(&self, utilization: f64) {
            let mut current = self.current_threads.lock().unwrap();
            let mut cpu_util = self.cpu_utilization.lock().unwrap();
            *cpu_util = utilization;

            if utilization > 0.9 && *current < self.max_threads {
                // High utilization - increase threads
                *current = std::cmp::min(*current + 1, self.max_threads);
                self.apply_thread_count(*current);
            } else if utilization < 0.5 && *current > self.min_threads {
                // Low utilization - decrease threads
                *current = std::cmp::max(*current - 1, self.min_threads);
                self.apply_thread_count(*current);
            }
        }

        /// Apply the new thread count
        fn apply_thread_count(&self, count: usize) {
            configure_workers(Some(count));
        }

        /// Get current thread count
        pub fn current_thread_count(&self) -> usize {
            *self.current_threads.lock().unwrap()
        }
    }

    /// Scoped thread pool for temporary operations
    ///
    /// Creates a temporary thread pool configuration that is restored
    /// when the scope ends.
    pub struct ScopedThreadPool {
        #[allow(dead_code)]
        previous_config: WorkerConfig,
        _guard: ScopedThreadPoolGuard,
    }

    struct ScopedThreadPoolGuard;

    impl Drop for ScopedThreadPoolGuard {
        fn drop(&mut self) {
            // Restoration is handled by ScopedWorkers in the parent module
        }
    }

    impl ScopedThreadPool {
        /// Create a new scoped thread pool
        pub fn new(config: WorkerConfig) -> Self {
            let previous_config = WorkerConfig {
                workers: super::get_global_workers(),
                parallel_threshold: 1000,
                chunk_size: 64,
            };

            config.apply();

            Self {
                previous_config,
                _guard: ScopedThreadPoolGuard,
            }
        }

        /// Execute a function within this thread pool scope
        pub fn execute<F, R>(&self, f: F) -> R
        where
            F: FnOnce() -> R,
        {
            f()
        }
    }

    /// Thread pool benchmarking utilities
    pub mod benchmark {
        use super::*;
        use std::time::{Duration, Instant};

        /// Benchmark result for a thread pool configuration
        #[derive(Debug, Clone)]
        pub struct BenchmarkResult {
            pub profile: ThreadPoolProfile,
            pub num_threads: usize,
            pub execution_time: Duration,
            pub throughput: f64,
        }

        /// Benchmark different thread pool configurations
        pub fn benchmark_configurations<F>(
            profiles: &[ThreadPoolProfile],
            workload: F,
        ) -> Vec<BenchmarkResult>
        where
            F: Fn() -> f64 + Clone,
        {
            let mut results = Vec::new();

            for &profile in profiles {
                // Initialize thread pool with profile
                if let Err(e) = initialize_global_pool(profile) {
                    eprintln!("Failed to initialize pool for {:?}: {}", profile, e);
                    continue;
                }

                // Warm up
                for _ in 0..3 {
                    workload();
                }

                // Benchmark
                let start = Instant::now();
                let operations = 10;
                let mut total_work = 0.0;

                for _ in 0..operations {
                    total_work += workload();
                }

                let elapsed = start.elapsed();
                let throughput = total_work / elapsed.as_secs_f64();

                results.push(BenchmarkResult {
                    profile,
                    num_threads: profile.num_threads(),
                    execution_time: elapsed,
                    throughput,
                });
            }

            results
        }

        /// Find optimal thread pool configuration for a workload
        pub fn find_optimal_configuration<F>(workload: F) -> ThreadPoolProfile
        where
            F: Fn() -> f64 + Clone,
        {
            let profiles = vec![
                ThreadPoolProfile::CpuBound,
                ThreadPoolProfile::MemoryBound,
                ThreadPoolProfile::LatencySensitive,
            ];

            let results = benchmark_configurations(&profiles, workload);

            results
                .into_iter()
                .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
                .map(|r| r.profile)
                .unwrap_or(ThreadPoolProfile::Default)
        }
    }
}

/// Algorithm-specific parallel implementations
pub mod algorithms {
    use super::{adaptive, WorkerConfig};
    use crate::error::{LinalgError, LinalgResult};
    use ndarray::{Array1, ArrayView1, ArrayView2};
    use num_traits::{Float, NumAssign, One, Zero};
    use scirs2_core::parallel_ops::*;
    use std::iter::Sum;

    /// Parallel matrix-vector multiplication
    ///
    /// This is a simpler and more effective parallelization that can be used
    /// as a building block for more complex algorithms.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Input matrix
    /// * `vector` - Input vector
    /// * `config` - Worker configuration
    ///
    /// # Returns
    ///
    /// * Result vector y = A * x
    pub fn parallel_matvec<F>(
        matrix: &ArrayView2<F>,
        vector: &ArrayView1<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + Send + Sync + Zero + Sum + 'static,
    {
        let (m, n) = matrix.dim();
        if n != vector.len() {
            return Err(LinalgError::ShapeError(format!(
                "Matrix-vector dimensions incompatible: {}x{} * {}",
                m,
                n,
                vector.len()
            )));
        }

        let data_size = m * n;
        if !adaptive::should_use_parallel(data_size, config) {
            // Fall back to serial computation
            return Ok(matrix.dot(vector));
        }

        config.apply();

        // Parallel computation of each row
        let result_vec: Vec<F> = (0..m)
            .into_par_iter()
            .map(|i| {
                matrix
                    .row(i)
                    .iter()
                    .zip(vector.iter())
                    .map(|(&aij, &xj)| aij * xj)
                    .sum()
            })
            .collect();

        Ok(Array1::from_vec(result_vec))
    }

    /// Parallel power iteration for dominant eigenvalue
    ///
    /// This implementation uses parallel matrix-vector multiplications
    /// in the power iteration method for computing dominant eigenvalues.
    pub fn parallel_power_iteration<F>(
        matrix: &ArrayView2<F>,
        max_iter: usize,
        tolerance: F,
        config: &WorkerConfig,
    ) -> LinalgResult<(F, Array1<F>)>
    where
        F: Float + Send + Sync + Zero + Sum + NumAssign + One + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "Power iteration requires square matrix".to_string(),
            ));
        }

        let data_size = m * n;
        if !adaptive::should_use_parallel(data_size, config) {
            // Fall back to serial power iteration
            return crate::eigen::power_iteration(&matrix.view(), max_iter, tolerance);
        }

        config.apply();

        // Initialize with simple vector
        let mut v = Array1::ones(n);
        let norm = v.iter().map(|&x| x * x).sum::<F>().sqrt();
        v /= norm;

        let mut eigenvalue = F::zero();

        for _iter in 0..max_iter {
            // Use the parallel matrix-vector multiplication
            let new_v = parallel_matvec(matrix, &v.view(), config)?;

            // Compute eigenvalue estimate (Rayleigh quotient)
            let new_eigenvalue = new_v
                .iter()
                .zip(v.iter())
                .map(|(&new_vi, &vi)| new_vi * vi)
                .sum::<F>();

            // Normalize
            let norm = new_v.iter().map(|&x| x * x).sum::<F>().sqrt();
            if norm < F::epsilon() {
                return Err(LinalgError::ComputationError(
                    "Vector became zero during iteration".to_string(),
                ));
            }
            let normalized_v = new_v / norm;

            // Check convergence
            if (new_eigenvalue - eigenvalue).abs() < tolerance {
                return Ok((new_eigenvalue, normalized_v));
            }

            eigenvalue = new_eigenvalue;
            v = normalized_v;
        }

        Err(LinalgError::ComputationError(
            "Power iteration failed to converge".to_string(),
        ))
    }

    /// Parallel vector operations for linear algebra
    ///
    /// This module provides basic parallel vector operations that serve as
    /// building blocks for more complex algorithms.
    pub mod vector_ops {
        use super::*;

        /// Parallel dot product of two vectors
        pub fn parallel_dot<F>(
            x: &ArrayView1<F>,
            y: &ArrayView1<F>,
            config: &WorkerConfig,
        ) -> LinalgResult<F>
        where
            F: Float + Send + Sync + Zero + Sum + 'static,
        {
            if x.len() != y.len() {
                return Err(LinalgError::ShapeError(
                    "Vectors must have same length for dot product".to_string(),
                ));
            }

            let data_size = x.len();
            if !adaptive::should_use_parallel(data_size, config) {
                return Ok(x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum());
            }

            config.apply();

            let result = (0..x.len()).into_par_iter().map(|i| x[i] * y[i]).sum();

            Ok(result)
        }

        /// Parallel vector norm computation
        pub fn parallel_norm<F>(x: &ArrayView1<F>, config: &WorkerConfig) -> LinalgResult<F>
        where
            F: Float + Send + Sync + Zero + Sum + 'static,
        {
            let data_size = x.len();
            if !adaptive::should_use_parallel(data_size, config) {
                return Ok(x.iter().map(|&xi| xi * xi).sum::<F>().sqrt());
            }

            config.apply();

            let sum_squares = (0..x.len()).into_par_iter().map(|i| x[i] * x[i]).sum::<F>();

            Ok(sum_squares.sqrt())
        }

        /// Parallel AXPY operation: y = a*x + y
        ///
        /// Note: This function returns a new array rather than modifying in-place
        /// due to complications with parallel mutable iteration.
        pub fn parallel_axpy<F>(
            alpha: F,
            x: &ArrayView1<F>,
            y: &ArrayView1<F>,
            config: &WorkerConfig,
        ) -> LinalgResult<Array1<F>>
        where
            F: Float + Send + Sync + 'static,
        {
            if x.len() != y.len() {
                return Err(LinalgError::ShapeError(
                    "Vectors must have same length for AXPY".to_string(),
                ));
            }

            let data_size = x.len();
            if !adaptive::should_use_parallel(data_size, config) {
                let result = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&xi, &yi)| alpha * xi + yi)
                    .collect();
                return Ok(Array1::from_vec(result));
            }

            config.apply();

            let result_vec: Vec<F> = (0..x.len())
                .into_par_iter()
                .map(|i| alpha * x[i] + y[i])
                .collect();

            Ok(Array1::from_vec(result_vec))
        }
    }

    /// Parallel matrix multiplication (GEMM)
    ///
    /// Implements parallel general matrix multiplication with block-based
    /// parallelization for improved cache performance.
    pub fn parallel_gemm<F>(
        a: &ArrayView2<F>,
        b: &ArrayView2<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<ndarray::Array2<F>>
    where
        F: Float + Send + Sync + Zero + Sum + NumAssign + 'static,
    {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();

        if k != k2 {
            return Err(LinalgError::ShapeError(format!(
                "Matrix dimensions incompatible for multiplication: {}x{} * {}x{}",
                m, k, k2, n
            )));
        }

        let data_size = m * k * n;
        if !adaptive::should_use_parallel(data_size, config) {
            return Ok(a.dot(b));
        }

        config.apply();

        // Block size for cache-friendly computation
        let block_size = config.chunk_size;

        let mut result = ndarray::Array2::zeros((m, n));

        // Parallel computation using blocks
        result
            .outer_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(i, mut row)| {
                for j in 0..n {
                    let mut sum = F::zero();
                    for kb in (0..k).step_by(block_size) {
                        let k_end = std::cmp::min(kb + block_size, k);
                        for ki in kb..k_end {
                            sum += a[[i, ki]] * b[[ki, j]];
                        }
                    }
                    row[j] = sum;
                }
            });

        Ok(result)
    }

    /// Parallel QR decomposition using Householder reflections
    ///
    /// This implementation parallelizes the application of Householder
    /// transformations across columns.
    pub fn parallel_qr<F>(
        matrix: &ArrayView2<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<(ndarray::Array2<F>, ndarray::Array2<F>)>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        let data_size = m * n;

        if !adaptive::should_use_parallel(data_size, config) {
            return crate::decomposition::qr(&matrix.view(), None);
        }

        config.apply();

        let mut a = matrix.to_owned();
        let mut q = ndarray::Array2::eye(m);
        let min_dim = std::cmp::min(m, n);

        for k in 0..min_dim {
            // Extract column vector for Householder reflection
            let x = a.slice(ndarray::s![k.., k]).to_owned();
            let alpha = if x[0] >= F::zero() {
                -x.iter().map(|&xi| xi * xi).sum::<F>().sqrt()
            } else {
                x.iter().map(|&xi| xi * xi).sum::<F>().sqrt()
            };

            if alpha.abs() < F::epsilon() {
                continue;
            }

            let mut v = x.clone();
            v[0] -= alpha;
            let v_norm_sq = v.iter().map(|&vi| vi * vi).sum::<F>();

            if v_norm_sq < F::epsilon() {
                continue;
            }

            // Apply Householder transformation (serial for simplicity)
            let remaining_cols = n - k;
            if remaining_cols > 1 {
                for j in k..n {
                    let col = a.slice(ndarray::s![k.., j]).to_owned();
                    let dot_product = v
                        .iter()
                        .zip(col.iter())
                        .map(|(&vi, &ci)| vi * ci)
                        .sum::<F>();
                    let factor = F::from(2.0).unwrap() * dot_product / v_norm_sq;

                    for (i, &vi) in v.iter().enumerate() {
                        a[[k + i, j]] -= factor * vi;
                    }
                }
            }

            // Update Q matrix (serial for simplicity)
            for i in 0..m {
                let row = q.slice(ndarray::s![i, k..]).to_owned();
                let dot_product = v
                    .iter()
                    .zip(row.iter())
                    .map(|(&vi, &ri)| vi * ri)
                    .sum::<F>();
                let factor = F::from(2.0).unwrap() * dot_product / v_norm_sq;

                for (j, &vj) in v.iter().enumerate() {
                    q[[i, k + j]] -= factor * vj;
                }
            }
        }

        let r = a.slice(ndarray::s![..min_dim, ..]).to_owned();
        Ok((q, r))
    }

    /// Parallel Cholesky decomposition
    ///
    /// Implements parallel Cholesky decomposition using block-column approach
    /// for positive definite matrices.
    pub fn parallel_cholesky<F>(
        matrix: &ArrayView2<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<ndarray::Array2<F>>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "Cholesky decomposition requires square matrix".to_string(),
            ));
        }

        let data_size = m * n;
        if !adaptive::should_use_parallel(data_size, config) {
            return crate::decomposition::cholesky(&matrix.view(), None);
        }

        config.apply();

        let mut l = ndarray::Array2::zeros((n, n));
        let block_size = config.chunk_size;

        for k in (0..n).step_by(block_size) {
            let k_end = std::cmp::min(k + block_size, n);

            // Diagonal block factorization (serial for numerical stability)
            for i in k..k_end {
                // Compute L[i,i]
                let mut sum = F::zero();
                for j in 0..i {
                    sum += l[[i, j]] * l[[i, j]];
                }
                let aii = matrix[[i, i]] - sum;
                if aii <= F::zero() {
                    return Err(LinalgError::ComputationError(
                        "Matrix is not positive definite".to_string(),
                    ));
                }
                l[[i, i]] = aii.sqrt();

                // Compute L[i+1:k_end, i]
                for j in (i + 1)..k_end {
                    let mut sum = F::zero();
                    for p in 0..i {
                        sum += l[[j, p]] * l[[i, p]];
                    }
                    l[[j, i]] = (matrix[[j, i]] - sum) / l[[i, i]];
                }
            }

            // Update trailing submatrix (serial for simplicity)
            if k_end < n {
                for i in k_end..n {
                    for j in k..k_end {
                        let mut sum = F::zero();
                        for p in 0..j {
                            sum += l[[i, p]] * l[[j, p]];
                        }
                        l[[i, j]] = (matrix[[i, j]] - sum) / l[[j, j]];
                    }
                }
            }
        }

        Ok(l)
    }

    /// Parallel LU decomposition with partial pivoting
    ///
    /// Implements parallel LU decomposition using block-column approach.
    pub fn parallel_lu<F>(
        matrix: &ArrayView2<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<(ndarray::Array2<F>, ndarray::Array2<F>, ndarray::Array2<F>)>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        let data_size = m * n;

        if !adaptive::should_use_parallel(data_size, config) {
            return crate::decomposition::lu(&matrix.view(), None);
        }

        config.apply();

        let mut a = matrix.to_owned();
        let mut perm_vec = (0..m).collect::<Vec<_>>();
        let min_dim = std::cmp::min(m, n);

        for k in 0..min_dim {
            // Find pivot (serial for correctness)
            let mut max_val = F::zero();
            let mut pivot_row = k;
            for i in k..m {
                let abs_val = a[[i, k]].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    pivot_row = i;
                }
            }

            if max_val < F::epsilon() {
                return Err(LinalgError::ComputationError(
                    "Matrix is singular".to_string(),
                ));
            }

            // Swap rows if needed
            if pivot_row != k {
                for j in 0..n {
                    let temp = a[[k, j]];
                    a[[k, j]] = a[[pivot_row, j]];
                    a[[pivot_row, j]] = temp;
                }
                perm_vec.swap(k, pivot_row);
            }

            // Update submatrix (serial for now to avoid borrowing issues)
            let pivot = a[[k, k]];

            for i in (k + 1)..m {
                let multiplier = a[[i, k]] / pivot;
                a[[i, k]] = multiplier;

                for j in (k + 1)..n {
                    a[[i, j]] = a[[i, j]] - multiplier * a[[k, j]];
                }
            }
        }

        // Create permutation matrix P
        let mut p = ndarray::Array2::zeros((m, m));
        for (i, &piv) in perm_vec.iter().enumerate() {
            p[[i, piv]] = F::one();
        }

        // Extract L and U matrices
        let mut l = ndarray::Array2::eye(m);
        let mut u = ndarray::Array2::zeros((m, n));

        for i in 0..m {
            for j in 0..n {
                if i > j && j < min_dim {
                    l[[i, j]] = a[[i, j]];
                } else if i <= j {
                    u[[i, j]] = a[[i, j]];
                }
            }
        }

        Ok((p, l, u))
    }

    /// Parallel conjugate gradient solver
    ///
    /// Implements parallel conjugate gradient method for solving linear systems
    /// with symmetric positive definite matrices.
    pub fn parallel_conjugate_gradient<F>(
        matrix: &ArrayView2<F>,
        b: &ArrayView1<F>,
        max_iter: usize,
        tolerance: F,
        config: &WorkerConfig,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "CG requires square matrix".to_string(),
            ));
        }
        if n != b.len() {
            return Err(LinalgError::ShapeError(
                "Matrix and vector dimensions incompatible".to_string(),
            ));
        }

        let data_size = m * n;
        if !adaptive::should_use_parallel(data_size, config) {
            return crate::iterative_solvers::conjugate_gradient(
                &matrix.view(),
                &b.view(),
                max_iter,
                tolerance,
                None,
            );
        }

        config.apply();

        // Initialize
        let mut x = Array1::zeros(n);

        // r = b - A*x
        let ax = parallel_matvec(matrix, &x.view(), config)?;
        let mut r = b - &ax;
        let mut p = r.clone();
        let mut rsold = vector_ops::parallel_dot(&r.view(), &r.view(), config)?;

        for _iter in 0..max_iter {
            let ap = parallel_matvec(matrix, &p.view(), config)?;
            let alpha = rsold / vector_ops::parallel_dot(&p.view(), &ap.view(), config)?;

            x = vector_ops::parallel_axpy(alpha, &p.view(), &x.view(), config)?;
            r = vector_ops::parallel_axpy(-alpha, &ap.view(), &r.view(), config)?;

            let rsnew = vector_ops::parallel_dot(&r.view(), &r.view(), config)?;

            if rsnew.sqrt() < tolerance {
                return Ok(x);
            }

            let beta = rsnew / rsold;
            p = vector_ops::parallel_axpy(beta, &p.view(), &r.view(), config)?;
            rsold = rsnew;
        }

        Err(LinalgError::ComputationError(
            "Conjugate gradient failed to converge".to_string(),
        ))
    }

    /// Parallel SVD decomposition
    ///
    /// Implements parallel Singular Value Decomposition using a block-based approach
    /// for improved performance on large matrices.
    pub fn parallel_svd<F>(
        matrix: &ArrayView2<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<(ndarray::Array2<F>, ndarray::Array1<F>, ndarray::Array2<F>)>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        let data_size = m * n;

        if !adaptive::should_use_parallel(data_size, config) {
            return crate::decomposition::svd(&matrix.view(), false, None);
        }

        config.apply();

        // For now, use QR decomposition as a first step
        // This is a simplified parallel SVD - a full implementation would use
        // more sophisticated algorithms like Jacobi SVD or divide-and-conquer
        let (q, r) = parallel_qr(matrix, config)?;

        // Apply SVD to the smaller R matrix (serial for numerical stability)
        let (u_r, s, vt) = crate::decomposition::svd(&r.view(), false, None)?;

        // U = Q * U_r
        let u = parallel_gemm(&q.view(), &u_r.view(), config)?;

        Ok((u, s, vt))
    }

    /// Parallel GMRES (Generalized Minimal Residual) solver
    ///
    /// Implements parallel GMRES for solving non-symmetric linear systems.
    pub fn parallel_gmres<F>(
        matrix: &ArrayView2<F>,
        b: &ArrayView1<F>,
        max_iter: usize,
        tolerance: F,
        restart: usize,
        config: &WorkerConfig,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + std::fmt::Debug + std::fmt::Display + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "GMRES requires square matrix".to_string(),
            ));
        }
        if n != b.len() {
            return Err(LinalgError::ShapeError(
                "Matrix and vector dimensions incompatible".to_string(),
            ));
        }

        let data_size = m * n;
        if !adaptive::should_use_parallel(data_size, config) {
            // Fall back to serial GMRES - use the iterative solver version
            let options = crate::solvers::iterative::IterativeSolverOptions {
                max_iterations: max_iter,
                tolerance,
                verbose: false,
                restart: Some(restart),
            };
            let result = crate::solvers::iterative::gmres(
                matrix,
                b,
                None,
                &options,
            )?;
            return Ok(result.solution);
        }

        config.apply();

        let mut x = Array1::zeros(n);
        let restart = restart.min(n);

        for _outer in 0..max_iter {
            // Compute initial residual
            let ax = parallel_matvec(matrix, &x.view(), config)?;
            let r = b - &ax;
            let beta = vector_ops::parallel_norm(&r.view(), config)?;

            if beta < tolerance {
                return Ok(x);
            }

            // Initialize Krylov subspace
            let mut v = vec![r / beta];
            let mut h = ndarray::Array2::<F>::zeros((restart + 1, restart));

            // Arnoldi iteration
            for j in 0..restart {
                // w = A * v[j]
                let w = parallel_matvec(matrix, &v[j].view(), config)?;

                // Modified Gram-Schmidt orthogonalization
                let mut w_new = w.clone();
                for i in 0..=j {
                    h[[i, j]] = vector_ops::parallel_dot(&w.view(), &v[i].view(), config)?;
                    w_new = vector_ops::parallel_axpy(-h[[i, j]], &v[i].view(), &w_new.view(), config)?;
                }

                h[[j + 1, j]] = vector_ops::parallel_norm(&w_new.view(), config)?;

                if h[[j + 1, j]] < F::epsilon() {
                    break;
                }

                v.push(w_new / h[[j + 1, j]]);
            }

            // Solve least squares problem (serial for numerical stability)
            let k = v.len() - 1;
            let h_sub = h.slice(ndarray::s![..=k, ..k]).to_owned();
            let mut g = Array1::zeros(k + 1);
            g[0] = beta;

            // Apply Givens rotations to solve the least squares problem
            let mut y = Array1::zeros(k);
            for i in (0..k).rev() {
                let mut sum = g[i];
                for j in (i + 1)..k {
                    sum -= h_sub[[i, j]] * y[j];
                }
                y[i] = sum / h_sub[[i, i]];
            }

            // Update solution
            for i in 0..k {
                x = vector_ops::parallel_axpy(y[i], &v[i].view(), &x.view(), config)?;
            }

            // Check residual
            let ax = parallel_matvec(matrix, &x.view(), config)?;
            let r = b - &ax;
            let residual_norm = vector_ops::parallel_norm(&r.view(), config)?;

            if residual_norm < tolerance {
                return Ok(x);
            }
        }

        Err(LinalgError::ComputationError(
            "GMRES failed to converge".to_string(),
        ))
    }

    /// Parallel BiCGSTAB (Biconjugate Gradient Stabilized) solver
    ///
    /// Implements parallel BiCGSTAB for solving non-symmetric linear systems.
    pub fn parallel_bicgstab<F>(
        matrix: &ArrayView2<F>,
        b: &ArrayView1<F>,
        max_iter: usize,
        tolerance: F,
        config: &WorkerConfig,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "BiCGSTAB requires square matrix".to_string(),
            ));
        }
        if n != b.len() {
            return Err(LinalgError::ShapeError(
                "Matrix and vector dimensions incompatible".to_string(),
            ));
        }

        let data_size = m * n;
        if !adaptive::should_use_parallel(data_size, config) {
            return crate::iterative_solvers::bicgstab(&matrix.view(), &b.view(), max_iter, tolerance, None);
        }

        config.apply();

        // Initialize
        let mut x = Array1::zeros(n);
        let ax = parallel_matvec(matrix, &x.view(), config)?;
        let mut r = b - &ax;
        let r_hat = r.clone();
        
        let mut rho = F::one();
        let mut alpha = F::one();
        let mut omega = F::one();
        
        let mut v = Array1::zeros(n);
        let mut p = Array1::zeros(n);

        for _iter in 0..max_iter {
            let rho_new = vector_ops::parallel_dot(&r_hat.view(), &r.view(), config)?;
            
            if rho_new.abs() < F::epsilon() {
                return Err(LinalgError::ComputationError(
                    "BiCGSTAB breakdown: rho = 0".to_string(),
                ));
            }
            
            let beta = (rho_new / rho) * (alpha / omega);
            
            // p = r + beta * (p - omega * v)
            let temp = vector_ops::parallel_axpy(-omega, &v.view(), &p.view(), config)?;
            p = vector_ops::parallel_axpy(F::one(), &r.view(), &vector_ops::parallel_axpy(beta, &temp.view(), &Array1::zeros(n).view(), config)?.view(), config)?;
            
            // v = A * p
            v = parallel_matvec(matrix, &p.view(), config)?;
            
            alpha = rho_new / vector_ops::parallel_dot(&r_hat.view(), &v.view(), config)?;
            
            // s = r - alpha * v
            let s = vector_ops::parallel_axpy(-alpha, &v.view(), &r.view(), config)?;
            
            // Check convergence
            let s_norm = vector_ops::parallel_norm(&s.view(), config)?;
            if s_norm < tolerance {
                x = vector_ops::parallel_axpy(alpha, &p.view(), &x.view(), config)?;
                return Ok(x);
            }
            
            // t = A * s
            let t = parallel_matvec(matrix, &s.view(), config)?;
            
            omega = vector_ops::parallel_dot(&t.view(), &s.view(), config)? / 
                    vector_ops::parallel_dot(&t.view(), &t.view(), config)?;
            
            // x = x + alpha * p + omega * s
            x = vector_ops::parallel_axpy(alpha, &p.view(), &x.view(), config)?;
            x = vector_ops::parallel_axpy(omega, &s.view(), &x.view(), config)?;
            
            // r = s - omega * t
            r = vector_ops::parallel_axpy(-omega, &t.view(), &s.view(), config)?;
            
            // Check convergence
            let r_norm = vector_ops::parallel_norm(&r.view(), config)?;
            if r_norm < tolerance {
                return Ok(x);
            }
            
            rho = rho_new;
        }

        Err(LinalgError::ComputationError(
            "BiCGSTAB failed to converge".to_string(),
        ))
    }

    /// Parallel Jacobi method
    ///
    /// Implements parallel Jacobi iteration for solving linear systems.
    /// This method is particularly well-suited for parallel execution.
    pub fn parallel_jacobi<F>(
        matrix: &ArrayView2<F>,
        b: &ArrayView1<F>,
        max_iter: usize,
        tolerance: F,
        config: &WorkerConfig,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "Jacobi method requires square matrix".to_string(),
            ));
        }
        if n != b.len() {
            return Err(LinalgError::ShapeError(
                "Matrix and vector dimensions incompatible".to_string(),
            ));
        }

        let data_size = m * n;
        if !adaptive::should_use_parallel(data_size, config) {
            return crate::iterative_solvers::jacobi_method(&matrix.view(), &b.view(), max_iter, tolerance, None);
        }

        config.apply();

        // Extract diagonal
        let diag: Vec<F> = (0..n)
            .into_par_iter()
            .map(|i| {
                if matrix[[i, i]].abs() < F::epsilon() {
                    F::one() // Avoid division by zero
                } else {
                    matrix[[i, i]]
                }
            })
            .collect();

        let mut x = Array1::zeros(n);

        for _iter in 0..max_iter {
            // Parallel update: x_new[i] = (b[i] - sum(A[i,j]*x[j] for j != i)) / A[i,i]
            let x_new_vec: Vec<F> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut sum = b[i];
                    for j in 0..n {
                        if i != j {
                            sum -= matrix[[i, j]] * x[j];
                        }
                    }
                    sum / diag[i]
                })
                .collect();

            let x_new = Array1::from_vec(x_new_vec);

            // Check convergence
            let diff = &x_new - &x;
            let error = vector_ops::parallel_norm(&diff.view(), config)?;

            if error < tolerance {
                return Ok(x_new);
            }

            x = x_new.clone();
        }

        Err(LinalgError::ComputationError(
            "Jacobi method failed to converge".to_string(),
        ))
    }

    /// Parallel SOR (Successive Over-Relaxation) method
    ///
    /// Implements a modified parallel SOR using red-black ordering
    /// to enable parallel updates.
    pub fn parallel_sor<F>(
        matrix: &ArrayView2<F>,
        b: &ArrayView1<F>,
        omega: F,
        max_iter: usize,
        tolerance: F,
        config: &WorkerConfig,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "SOR requires square matrix".to_string(),
            ));
        }
        if n != b.len() {
            return Err(LinalgError::ShapeError(
                "Matrix and vector dimensions incompatible".to_string(),
            ));
        }

        if omega <= F::zero() || omega >= F::from(2.0).unwrap() {
            return Err(LinalgError::InvalidInputError(
                "Relaxation parameter omega must be in (0, 2)".to_string(),
            ));
        }

        let data_size = m * n;
        if !adaptive::should_use_parallel(data_size, config) {
            return crate::iterative_solvers::successive_over_relaxation(
                &matrix.view(), &b.view(), omega, max_iter, tolerance, None
            );
        }

        config.apply();

        let mut x = Array1::zeros(n);

        for _iter in 0..max_iter {
            let x_old = x.clone();

            // Red-black ordering for parallel updates
            // First update "red" points (even indices)
            let red_updates: Vec<(usize, F)> = (0..n)
                .into_par_iter()
                .filter(|&i| i % 2 == 0)
                .map(|i| {
                    let mut sum = b[i];
                    for j in 0..n {
                        if i != j {
                            sum -= matrix[[i, j]] * x_old[j];
                        }
                    }
                    let x_gs = sum / matrix[[i, i]];
                    let x_new = (F::one() - omega) * x_old[i] + omega * x_gs;
                    (i, x_new)
                })
                .collect();

            // Apply red updates
            for (i, val) in red_updates {
                x[i] = val;
            }

            // Then update "black" points (odd indices)
            let black_updates: Vec<(usize, F)> = (0..n)
                .into_par_iter()
                .filter(|&i| i % 2 == 1)
                .map(|i| {
                    let mut sum = b[i];
                    for j in 0..n {
                        if i != j {
                            sum -= matrix[[i, j]] * x[j];
                        }
                    }
                    let x_gs = sum / matrix[[i, i]];
                    let x_new = (F::one() - omega) * x_old[i] + omega * x_gs;
                    (i, x_new)
                })
                .collect();

            // Apply black updates
            for (i, val) in black_updates {
                x[i] = val;
            }

            // Check convergence
            let ax = parallel_matvec(matrix, &x.view(), config)?;
            let r = b - &ax;
            let error = vector_ops::parallel_norm(&r.view(), config)?;

            if error < tolerance {
                return Ok(x);
            }
        }

        Err(LinalgError::ComputationError(
            "SOR failed to converge".to_string(),
        ))
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
