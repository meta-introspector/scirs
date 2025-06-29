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
    fn create_config_for_profile(&self, profile: &ThreadPoolProfile) -> LinalgResult<ThreadPoolConfig> {
        let base_config = match profile {
            ThreadPoolProfile::CpuIntensive => ThreadPoolConfig::cpu_intensive(),
            ThreadPoolProfile::MemoryBound => ThreadPoolConfig::memory_bound(),
            ThreadPoolProfile::LowLatency => ThreadPoolConfig::low_latency(),
            ThreadPoolProfile::HighThroughput => ThreadPoolConfig::high_throughput(),
            ThreadPoolProfile::Balanced => ThreadPoolConfig::default(),
            ThreadPoolProfile::Custom(_) => {
                // Use global config for custom profiles
                self.global_config.lock().map_err(|_| {
                    LinalgError::ComputationError("Failed to acquire global config lock".to_string())
                })?.clone()
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
        self.stats.lock().map(|stats| stats.clone()).unwrap_or_default()
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
    pub fn recommend_profile(&self, 
        matrix_size: usize, 
        operation_type: OperationType,
        _memory_usage: Option<usize>,
    ) -> ThreadPoolProfile {
        match operation_type {
            OperationType::MatrixMultiplication if matrix_size > 1000 => {
                ThreadPoolProfile::CpuIntensive
            }
            OperationType::Decomposition if matrix_size > 500 => {
                ThreadPoolProfile::CpuIntensive
            }
            OperationType::Solve if matrix_size < 100 => {
                ThreadPoolProfile::LowLatency
            }
            OperationType::IterativeSolver => {
                ThreadPoolProfile::MemoryBound
            }
            OperationType::BatchOperations => {
                ThreadPoolProfile::HighThroughput
            }
            _ => ThreadPoolProfile::Balanced,
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
    Solve,
    IterativeSolver,
    BatchOperations,
    ElementWise,
    Reduction,
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
                .map_err(|e| LinalgError::ComputationError(format!("Failed to spawn thread: {}", e)))?;
            
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
        
        let profile = manager.recommend_profile(
            2000, 
            OperationType::MatrixMultiplication, 
            None
        );
        assert_eq!(profile, ThreadPoolProfile::CpuIntensive);
        
        let profile = manager.recommend_profile(
            50, 
            OperationType::Solve, 
            None
        );
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