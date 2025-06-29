//! Enhanced thread pool configurations and management
//!
//! This module provides sophisticated thread pool management with support for
//! adaptive sizing, CPU affinity, NUMA awareness, and workload-specific optimization.

use crate::error::{LinalgError, LinalgResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;

/// Thread pool profile for different types of workloads
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ThreadPoolProfile {
    /// Optimized for CPU-intensive computations
    CpuIntensive,
    /// Optimized for memory-bound operations
    MemoryBound,
    /// Balanced profile for mixed workloads
    Balanced,
    /// Low-latency profile for quick operations
    LowLatency,
    /// High-throughput profile for bulk operations
    HighThroughput,
    /// Linear algebra specific optimizations
    LinearAlgebra,
    /// Matrix multiplication optimized
    MatrixMultiplication,
    /// Eigenvalue computation optimized
    EigenComputation,
    /// Decomposition algorithms optimized
    Decomposition,
    /// Iterative solver optimized
    IterativeSolver,
    /// NUMA-aware parallel processing
    NumaOptimized,
    /// GPU-CPU hybrid processing
    HybridComputing,
    /// Custom profile with specific parameters
    Custom(String),
}

/// Thread affinity strategy
#[derive(Debug, Clone)]
pub enum AffinityStrategy {
    /// No specific affinity
    None,
    /// Pin threads to specific cores
    Pinned(Vec<usize>),
    /// Spread threads across NUMA nodes
    NumaSpread,
    /// Keep threads within same NUMA node
    NumaCompact,
    /// Custom affinity pattern
    Custom(Vec<Option<usize>>),
}

/// Thread pool configuration
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Profile for workload optimization
    pub profile: ThreadPoolProfile,
    /// Minimum number of threads
    pub min_threads: usize,
    /// Maximum number of threads
    pub max_threads: usize,
    /// Current number of active threads
    pub active_threads: usize,
    /// Thread idle timeout
    pub idle_timeout: Duration,
    /// Affinity strategy
    pub affinity: AffinityStrategy,
    /// Enable NUMA awareness
    pub numa_aware: bool,
    /// Work stealing enabled
    pub work_stealing: bool,
    /// Queue capacity
    pub queue_capacity: usize,
    /// Thread stack size
    pub stack_size: Option<usize>,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            profile: ThreadPoolProfile::Balanced,
            min_threads: 1,
            max_threads: num_cpus,
            active_threads: num_cpus,
            idle_timeout: Duration::from_secs(60),
            affinity: AffinityStrategy::None,
            numa_aware: false,
            work_stealing: true,
            queue_capacity: 1024,
            stack_size: None,
        }
    }
}

impl ThreadPoolConfig {
    /// Create optimized configuration for matrix multiplication workloads
    pub fn for_matrix_multiplication(matrix_size: (usize, usize)) -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let (m, n) = matrix_size;
        let total_ops = m * n;

        // Optimize based on problem size
        let (threads, stack_size, queue_capacity) = if total_ops > 1_000_000 {
            // Large matrices: use all cores with larger stacks and queues
            (num_cpus, Some(8 * 1024 * 1024), 4096)
        } else if total_ops > 100_000 {
            // Medium matrices: moderate parallelism
            ((num_cpus * 3) / 4, Some(4 * 1024 * 1024), 2048)
        } else {
            // Small matrices: limited parallelism to avoid overhead
            (num_cpus / 2, None, 512)
        };

        Self {
            profile: ThreadPoolProfile::MatrixMultiplication,
            min_threads: 1,
            max_threads: threads,
            active_threads: threads,
            idle_timeout: Duration::from_secs(30),
            affinity: AffinityStrategy::NumaSpread,
            numa_aware: true,
            work_stealing: true,
            queue_capacity,
            stack_size,
        }
    }

    /// Create optimized configuration for eigenvalue computation
    pub fn for_eigenvalue_computation(matrix_order: usize) -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // Eigenvalue problems benefit from fewer threads due to synchronization
        let threads = if matrix_order > 1000 {
            num_cpus
        } else if matrix_order > 100 {
            (num_cpus * 2) / 3
        } else {
            num_cpus / 2
        };

        Self {
            profile: ThreadPoolProfile::EigenComputation,
            min_threads: 1,
            max_threads: threads,
            active_threads: threads,
            idle_timeout: Duration::from_secs(120), // Longer timeout for iterative methods
            affinity: AffinityStrategy::NumaCompact,
            numa_aware: true,
            work_stealing: false, // Less beneficial for eigenvalue algorithms
            queue_capacity: 1024,
            stack_size: Some(4 * 1024 * 1024),
        }
    }

    /// Create optimized configuration for decomposition algorithms
    pub fn for_decomposition(decomp_type: DecompositionType, matrix_size: (usize, usize)) -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let (m, n) = matrix_size;
        let size = m.max(n);

        let (threads, work_stealing, affinity) = match decomp_type {
            DecompositionType::LU => {
                // LU benefits from sequential processing with some parallelism
                (
                    (num_cpus * 2) / 3,
                    false,
                    AffinityStrategy::Pinned(vec![0, 1, 2, 3]),
                )
            }
            DecompositionType::QR => {
                // QR can use more parallelism due to independent Householder reflections
                (num_cpus, true, AffinityStrategy::NumaSpread)
            }
            DecompositionType::SVD => {
                // SVD is compute-intensive and benefits from all cores
                (num_cpus, true, AffinityStrategy::NumaSpread)
            }
            DecompositionType::Cholesky => {
                // Cholesky is inherently sequential but can parallelize column operations
                (num_cpus.div_ceil(2), false, AffinityStrategy::NumaCompact)
            }
        };

        Self {
            profile: ThreadPoolProfile::Decomposition,
            min_threads: 1,
            max_threads: threads,
            active_threads: threads,
            idle_timeout: Duration::from_secs(45),
            affinity,
            numa_aware: size > 500,
            work_stealing,
            queue_capacity: if size > 1000 { 2048 } else { 1024 },
            stack_size: Some(6 * 1024 * 1024),
        }
    }

    /// Create optimized configuration for iterative solvers
    pub fn for_iterative_solver(problem_size: usize, solver_type: IterativeSolverType) -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let (threads, queue_capacity, affinity) = match solver_type {
            IterativeSolverType::ConjugateGradient => {
                // CG benefits from moderate parallelism
                ((num_cpus * 3) / 4, 1024, AffinityStrategy::NumaSpread)
            }
            IterativeSolverType::GMRES => {
                // GMRES needs more memory and moderate parallelism
                (num_cpus / 2, 2048, AffinityStrategy::NumaCompact)
            }
            IterativeSolverType::BiCGSTAB => {
                // BiCGSTAB can use more parallelism
                (num_cpus, 1024, AffinityStrategy::NumaSpread)
            }
        };

        Self {
            profile: ThreadPoolProfile::IterativeSolver,
            min_threads: 1,
            max_threads: threads,
            active_threads: threads,
            idle_timeout: Duration::from_secs(180), // Long timeout for iterative methods
            affinity,
            numa_aware: problem_size > 10_000,
            work_stealing: true,
            queue_capacity,
            stack_size: Some(8 * 1024 * 1024), // Large stack for deep recursion
        }
    }
}

/// Decomposition algorithm types for configuration optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompositionType {
    LU,
    QR,
    SVD,
    Cholesky,
}

/// Iterative solver types for configuration optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterativeSolverType {
    ConjugateGradient,
    GMRES,
    BiCGSTAB,
}

impl ThreadPoolConfig {
    /// Create a configuration for CPU-intensive workloads
    pub fn cpu_intensive() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            profile: ThreadPoolProfile::CpuIntensive,
            min_threads: num_cpus,
            max_threads: num_cpus,
            active_threads: num_cpus,
            affinity: AffinityStrategy::Pinned((0..num_cpus).collect()),
            numa_aware: true,
            work_stealing: true,
            stack_size: Some(2 * 1024 * 1024), // 2MB stack
            ..Default::default()
        }
    }

    /// Create a configuration for memory-bound workloads
    pub fn memory_bound() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            profile: ThreadPoolProfile::MemoryBound,
            min_threads: num_cpus / 2,
            max_threads: num_cpus * 2,
            active_threads: num_cpus,
            affinity: AffinityStrategy::NumaSpread,
            numa_aware: true,
            work_stealing: false, // Less beneficial for memory-bound tasks
            queue_capacity: 2048, // Larger queue for memory operations
            ..Default::default()
        }
    }

    /// Create a configuration for low-latency workloads
    pub fn low_latency() -> Self {
        Self {
            profile: ThreadPoolProfile::LowLatency,
            min_threads: 2,
            max_threads: 4,
            active_threads: 2,
            idle_timeout: Duration::from_millis(100),
            affinity: AffinityStrategy::Pinned(vec![0, 1]),
            work_stealing: false,
            queue_capacity: 64, // Small queue for low latency
            ..Default::default()
        }
    }

    /// Create a configuration for high-throughput workloads
    pub fn high_throughput() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            profile: ThreadPoolProfile::HighThroughput,
            min_threads: num_cpus,
            max_threads: num_cpus * 3,
            active_threads: num_cpus * 2,
            idle_timeout: Duration::from_secs(300), // Longer timeout
            affinity: AffinityStrategy::NumaSpread,
            numa_aware: true,
            work_stealing: true,
            queue_capacity: 4096, // Large queue for throughput
            ..Default::default()
        }
    }

    /// Set CPU affinity
    pub fn with_affinity(mut self, affinity: AffinityStrategy) -> Self {
        self.affinity = affinity;
        self
    }

    /// Enable or disable NUMA awareness
    pub fn with_numa_awareness(mut self, numa_aware: bool) -> Self {
        self.numa_aware = numa_aware;
        self
    }

    /// Set thread count
    pub fn with_threads(mut self, min: usize, max: usize) -> Self {
        self.min_threads = min;
        self.max_threads = max;
        self.active_threads = min;
        self
    }

    /// Set work stealing behavior
    pub fn with_work_stealing(mut self, enabled: bool) -> Self {
        self.work_stealing = enabled;
        self
    }

    /// Create configuration optimized for linear algebra operations
    pub fn linear_algebra() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            profile: ThreadPoolProfile::LinearAlgebra,
            min_threads: num_cpus,
            max_threads: num_cpus,
            active_threads: num_cpus,
            affinity: AffinityStrategy::Pinned((0..num_cpus).collect()),
            numa_aware: true,
            work_stealing: true,
            stack_size: Some(4 * 1024 * 1024), // 4MB stack for recursive algorithms
            queue_capacity: 1024,
            ..Default::default()
        }
    }

    /// Create configuration optimized for matrix multiplication
    pub fn matrix_multiplication() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            profile: ThreadPoolProfile::MatrixMultiplication,
            min_threads: num_cpus,
            max_threads: num_cpus,
            active_threads: num_cpus,
            affinity: AffinityStrategy::NumaSpread,
            numa_aware: true,
            work_stealing: false, // Less effective for regular matrix blocks
            stack_size: Some(2 * 1024 * 1024), // 2MB stack
            queue_capacity: 512,
            ..Default::default()
        }
    }

    /// Create configuration optimized for eigenvalue computations
    pub fn eigen_computation() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            profile: ThreadPoolProfile::EigenComputation,
            min_threads: num_cpus / 2,
            max_threads: num_cpus,
            active_threads: num_cpus,
            affinity: AffinityStrategy::Pinned((0..num_cpus).collect()),
            numa_aware: true,
            work_stealing: true,
            stack_size: Some(8 * 1024 * 1024), // 8MB stack for iterative methods
            queue_capacity: 256,
            idle_timeout: Duration::from_secs(120), // Longer timeout for iterative methods
        }
    }

    /// Create configuration optimized for matrix decompositions
    pub fn decomposition() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            profile: ThreadPoolProfile::Decomposition,
            min_threads: num_cpus,
            max_threads: num_cpus,
            active_threads: num_cpus,
            affinity: AffinityStrategy::NumaCompact,
            numa_aware: true,
            work_stealing: true,
            stack_size: Some(6 * 1024 * 1024), // 6MB stack
            queue_capacity: 768,
            ..Default::default()
        }
    }

    /// Create configuration optimized for iterative solvers
    pub fn iterative_solver() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            profile: ThreadPoolProfile::IterativeSolver,
            min_threads: num_cpus / 2,
            max_threads: num_cpus * 2,
            active_threads: num_cpus,
            affinity: AffinityStrategy::NumaSpread,
            numa_aware: true,
            work_stealing: true,
            stack_size: Some(4 * 1024 * 1024), // 4MB stack
            queue_capacity: 2048,              // Large queue for convergence iterations
            idle_timeout: Duration::from_secs(180),
        }
    }

    /// Create NUMA-optimized configuration
    pub fn numa_optimized() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            profile: ThreadPoolProfile::NumaOptimized,
            min_threads: num_cpus,
            max_threads: num_cpus,
            active_threads: num_cpus,
            affinity: AffinityStrategy::NumaSpread,
            numa_aware: true,
            work_stealing: false, // NUMA-aware scheduling instead
            stack_size: Some(3 * 1024 * 1024),
            queue_capacity: 1024,
            ..Default::default()
        }
    }

    /// Create hybrid GPU-CPU configuration
    pub fn hybrid_computing() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            profile: ThreadPoolProfile::HybridComputing,
            min_threads: num_cpus / 4, // Reserve some cores for GPU coordination
            max_threads: num_cpus / 2,
            active_threads: num_cpus / 2,
            affinity: AffinityStrategy::Custom(vec![Some(0), Some(1), None, None]), // Pin some, leave others free
            numa_aware: true,
            work_stealing: true,
            stack_size: Some(2 * 1024 * 1024),
            queue_capacity: 512,
            ..Default::default()
        }
    }
}

/// Thread pool statistics
#[derive(Debug, Default, Clone)]
pub struct ThreadPoolStats {
    /// Number of tasks completed
    pub tasks_completed: u64,
    /// Number of tasks currently queued
    pub tasks_queued: u64,
    /// Average task execution time
    pub avg_execution_time: Duration,
    /// Thread utilization (0.0 to 1.0)
    pub thread_utilization: f64,
    /// Queue utilization (0.0 to 1.0)
    pub queue_utilization: f64,
    /// Number of work steals
    pub work_steals: u64,
    /// Number of threads currently active
    pub active_threads: usize,
    /// CPU affinity effectiveness
    pub affinity_effectiveness: f64,
}

/// Enhanced thread pool manager
pub struct ThreadPoolManager {
    /// Registered thread pools by profile
    pools: RwLock<HashMap<ThreadPoolProfile, Arc<ThreadPool>>>,
    /// Global configuration
    global_config: Mutex<ThreadPoolConfig>,
    /// Pool usage statistics
    stats: Mutex<HashMap<ThreadPoolProfile, ThreadPoolStats>>,
}

impl ThreadPoolManager {
    /// Create a new thread pool manager
    pub fn new() -> Self {
        Self {
            pools: RwLock::new(HashMap::new()),
            global_config: Mutex::new(ThreadPoolConfig::default()),
            stats: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create a thread pool for the specified profile
    pub fn get_pool(&self, profile: ThreadPoolProfile) -> LinalgResult<Arc<ThreadPool>> {
        // First, try to get existing pool
        {
            let pools = self.pools.read().map_err(|_| {
                LinalgError::ComputationError("Failed to acquire pool read lock".to_string())
            })?;

            if let Some(pool) = pools.get(&profile) {
                return Ok(Arc::clone(pool));
            }
        }

        // Create new pool
        let config = self.create_config_for_profile(&profile)?;
        let pool = Arc::new(ThreadPool::new(config)?);

        // Store pool
        {
            let mut pools = self.pools.write().map_err(|_| {
                LinalgError::ComputationError("Failed to acquire pool write lock".to_string())
            })?;
            pools.insert(profile.clone(), Arc::clone(&pool));
        }

        // Initialize stats
        {
            let mut stats = self.stats.lock().map_err(|_| {
                LinalgError::ComputationError("Failed to acquire stats lock".to_string())
            })?;
            stats.insert(profile, ThreadPoolStats::default());
        }

        Ok(pool)
    }

    /// Create configuration for a specific profile
    fn create_config_for_profile(
        &self,
        profile: &ThreadPoolProfile,
    ) -> LinalgResult<ThreadPoolConfig> {
        let base_config = match profile {
            ThreadPoolProfile::CpuIntensive => ThreadPoolConfig::cpu_intensive(),
            ThreadPoolProfile::MemoryBound => ThreadPoolConfig::memory_bound(),
            ThreadPoolProfile::LowLatency => ThreadPoolConfig::low_latency(),
            ThreadPoolProfile::HighThroughput => ThreadPoolConfig::high_throughput(),
            ThreadPoolProfile::Balanced => ThreadPoolConfig::default(),
            ThreadPoolProfile::LinearAlgebra => ThreadPoolConfig::linear_algebra(),
            ThreadPoolProfile::MatrixMultiplication => ThreadPoolConfig::matrix_multiplication(),
            ThreadPoolProfile::EigenComputation => ThreadPoolConfig::eigen_computation(),
            ThreadPoolProfile::Decomposition => ThreadPoolConfig::decomposition(),
            ThreadPoolProfile::IterativeSolver => ThreadPoolConfig::iterative_solver(),
            ThreadPoolProfile::NumaOptimized => ThreadPoolConfig::numa_optimized(),
            ThreadPoolProfile::HybridComputing => ThreadPoolConfig::hybrid_computing(),
            ThreadPoolProfile::Custom(_) => {
                // Use global config for custom profiles
                self.global_config
                    .lock()
                    .map_err(|_| {
                        LinalgError::ComputationError(
                            "Failed to acquire global config lock".to_string(),
                        )
                    })?
                    .clone()
            }
        };

        Ok(base_config)
    }

    /// Set global configuration that applies to new pools
    pub fn set_global_config(&self, config: ThreadPoolConfig) -> LinalgResult<()> {
        let mut global_config = self.global_config.lock().map_err(|_| {
            LinalgError::ComputationError("Failed to acquire global config lock".to_string())
        })?;
        *global_config = config;
        Ok(())
    }

    /// Get statistics for a specific pool
    pub fn get_stats(&self, profile: &ThreadPoolProfile) -> Option<ThreadPoolStats> {
        self.stats.lock().ok()?.get(profile).cloned()
    }

    /// Get statistics for all pools
    pub fn get_all_stats(&self) -> HashMap<ThreadPoolProfile, ThreadPoolStats> {
        self.stats
            .lock()
            .map(|stats| stats.clone())
            .unwrap_or_default()
    }

    /// Shutdown all pools
    pub fn shutdown_all(&self) -> LinalgResult<()> {
        let mut pools = self.pools.write().map_err(|_| {
            LinalgError::ComputationError("Failed to acquire pool write lock".to_string())
        })?;

        for pool in pools.values() {
            pool.shutdown()?;
        }

        pools.clear();
        Ok(())
    }

    /// Adaptive pool selection based on workload characteristics
    pub fn recommend_profile(
        &self,
        matrix_size: usize,
        operation_type: OperationType,
        memory_usage: Option<usize>,
    ) -> ThreadPoolProfile {
        // Advanced recommendation logic based on matrix size and operation type
        match operation_type {
            OperationType::MatrixMultiplication => {
                if matrix_size > 2000 {
                    ThreadPoolProfile::MatrixMultiplication
                } else if matrix_size > 500 {
                    ThreadPoolProfile::LinearAlgebra
                } else {
                    ThreadPoolProfile::LowLatency
                }
            }
            OperationType::Decomposition => {
                if matrix_size > 1000 {
                    ThreadPoolProfile::Decomposition
                } else if matrix_size > 200 {
                    ThreadPoolProfile::LinearAlgebra
                } else {
                    ThreadPoolProfile::CpuIntensive
                }
            }
            OperationType::EigenSolver => {
                if matrix_size > 500 {
                    ThreadPoolProfile::EigenComputation
                } else {
                    ThreadPoolProfile::LinearAlgebra
                }
            }
            OperationType::Solve if matrix_size < 100 => ThreadPoolProfile::LowLatency,
            OperationType::Solve => ThreadPoolProfile::LinearAlgebra,
            OperationType::IterativeSolver => {
                if memory_usage.is_some_and(|mem| mem > 1_000_000_000) {
                    // > 1GB
                    ThreadPoolProfile::IterativeSolver
                } else {
                    ThreadPoolProfile::MemoryBound
                }
            }
            OperationType::BatchOperations => ThreadPoolProfile::HighThroughput,
            OperationType::ElementWise => {
                if matrix_size > 10000 {
                    ThreadPoolProfile::NumaOptimized
                } else {
                    ThreadPoolProfile::CpuIntensive
                }
            }
            OperationType::Reduction => ThreadPoolProfile::CpuIntensive,
            OperationType::HybridGpuCpu => ThreadPoolProfile::HybridComputing,
        }
    }
}

impl Default for ThreadPoolManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Operation type for thread pool recommendation
#[derive(Debug, Clone)]
pub enum OperationType {
    MatrixMultiplication,
    Decomposition,
    EigenSolver,
    Solve,
    IterativeSolver,
    BatchOperations,
    ElementWise,
    Reduction,
    HybridGpuCpu,
}

/// Individual thread pool implementation
pub struct ThreadPool {
    /// Configuration
    config: ThreadPoolConfig,
    /// Thread handles
    #[allow(dead_code)]
    threads: Vec<thread::JoinHandle<()>>,
    /// Shutdown signal
    shutdown: Arc<Mutex<bool>>,
}

impl ThreadPool {
    /// Create a new thread pool
    pub fn new(config: ThreadPoolConfig) -> LinalgResult<Self> {
        let shutdown = Arc::new(Mutex::new(false));
        let mut threads = Vec::new();

        // Create worker threads
        for i in 0..config.active_threads {
            let shutdown_clone = Arc::clone(&shutdown);
            let config_clone = config.clone();

            let handle = thread::Builder::new()
                .name(format!("worker-{}", i))
                .stack_size(config.stack_size.unwrap_or(1024 * 1024))
                .spawn(move || {
                    Self::worker_thread(i, config_clone, shutdown_clone);
                })
                .map_err(|e| {
                    LinalgError::ComputationError(format!("Failed to spawn thread: {}", e))
                })?;

            threads.push(handle);
        }

        Ok(Self {
            config,
            threads,
            shutdown,
        })
    }

    /// Worker thread main loop
    fn worker_thread(id: usize, config: ThreadPoolConfig, shutdown: Arc<Mutex<bool>>) {
        // Set thread affinity if specified
        Self::set_thread_affinity(id, &config.affinity);

        loop {
            // Check shutdown signal
            if let Ok(should_shutdown) = shutdown.lock() {
                if *should_shutdown {
                    break;
                }
            }

            // Simulate work processing
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Set CPU affinity for the current thread
    fn set_thread_affinity(thread_id: usize, affinity: &AffinityStrategy) {
        match affinity {
            AffinityStrategy::None => {
                // No affinity setting
            }
            AffinityStrategy::Pinned(cores) => {
                if let Some(&core_id) = cores.get(thread_id) {
                    Self::pin_to_core(core_id);
                }
            }
            AffinityStrategy::NumaSpread => {
                // Spread threads across NUMA nodes
                let numa_nodes = Self::get_numa_topology();
                if !numa_nodes.is_empty() {
                    let node_id = thread_id % numa_nodes.len();
                    Self::pin_to_numa_node(numa_nodes[node_id]);
                }
            }
            AffinityStrategy::NumaCompact => {
                // Keep threads on the same NUMA node (node 0)
                Self::pin_to_numa_node(0);
            }
            AffinityStrategy::Custom(cores) => {
                if let Some(Some(core_id)) = cores.get(thread_id) {
                    Self::pin_to_core(*core_id);
                }
            }
        }
    }

    /// Pin thread to specific CPU core (platform-specific implementation)
    fn pin_to_core(_core_id: usize) {
        // Platform-specific CPU affinity not implemented
        // In a production implementation, this would use platform-specific APIs
        // such as libc::sched_setaffinity on Linux or SetThreadAffinityMask on Windows
    }

    /// Pin thread to NUMA node
    fn pin_to_numa_node(node_id: usize) {
        // Simplified NUMA pinning - in practice would use numa library
        let _ = node_id; // Placeholder
    }

    /// Get NUMA topology (simplified)
    fn get_numa_topology() -> Vec<usize> {
        // Return mock topology - in practice would query actual NUMA layout
        vec![0, 1] // Assume 2 NUMA nodes
    }

    /// Get current configuration
    pub fn config(&self) -> &ThreadPoolConfig {
        &self.config
    }

    /// Shutdown the thread pool
    pub fn shutdown(&self) -> LinalgResult<()> {
        // Signal shutdown
        {
            let mut shutdown = self.shutdown.lock().map_err(|_| {
                LinalgError::ComputationError("Failed to acquire shutdown lock".to_string())
            })?;
            *shutdown = true;
        }

        // Note: In a real implementation, we would join threads here
        // For now, we just signal shutdown
        Ok(())
    }
}

/// Global thread pool manager instance
static GLOBAL_MANAGER: std::sync::OnceLock<ThreadPoolManager> = std::sync::OnceLock::new();

/// Get the global thread pool manager
pub fn get_global_manager() -> &'static ThreadPoolManager {
    GLOBAL_MANAGER.get_or_init(ThreadPoolManager::new)
}

/// Create a scoped thread pool for temporary use
pub struct ScopedThreadPool {
    pool: Arc<ThreadPool>,
    _phantom: std::marker::PhantomData<()>,
}

impl ScopedThreadPool {
    /// Create a new scoped thread pool
    pub fn new(config: ThreadPoolConfig) -> LinalgResult<Self> {
        let pool = Arc::new(ThreadPool::new(config)?);
        Ok(Self {
            pool,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get the underlying thread pool
    pub fn pool(&self) -> &ThreadPool {
        &self.pool
    }
}

impl Drop for ScopedThreadPool {
    fn drop(&mut self) {
        let _ = self.pool.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_pool_config_creation() {
        let config = ThreadPoolConfig::cpu_intensive();
        assert_eq!(config.profile, ThreadPoolProfile::CpuIntensive);
        assert!(config.numa_aware);
        assert!(config.work_stealing);
    }

    #[test]
    fn test_thread_pool_manager() {
        let manager = ThreadPoolManager::new();

        let pool = manager.get_pool(ThreadPoolProfile::Balanced);
        assert!(pool.is_ok());

        let pool2 = manager.get_pool(ThreadPoolProfile::Balanced);
        assert!(pool2.is_ok());

        // Should be the same pool instance
        assert!(Arc::ptr_eq(&pool.unwrap(), &pool2.unwrap()));
    }

    #[test]
    fn test_profile_recommendation() {
        let manager = ThreadPoolManager::new();

        let profile = manager.recommend_profile(2000, OperationType::MatrixMultiplication, None);
        assert_eq!(profile, ThreadPoolProfile::LinearAlgebra);

        let profile = manager.recommend_profile(50, OperationType::Solve, None);
        assert_eq!(profile, ThreadPoolProfile::LowLatency);
    }

    #[test]
    fn test_scoped_thread_pool() {
        let config = ThreadPoolConfig::default();
        let scoped_pool = ScopedThreadPool::new(config);
        assert!(scoped_pool.is_ok());

        // Pool should be automatically cleaned up when dropped
    }
}
