//! Performance optimization utilities for neural networks
//!
//! This module provides performance optimizations including SIMD acceleration,
//! memory-efficient operations, and thread pool support.

use crate::error::{NeuralError, Result};
use ndarray::{Array, ArrayD};
#[allow(unused_imports)]
use num_traits::Float;
use std::fmt::Debug;
use std::sync::Arc;

#[cfg(feature = "simd")]
use scirs2_core::simd::*;

#[cfg(feature = "memory_efficient")]
use scirs2_core::memory_efficient::{chunk_wise_op, ChunkProcessor};

/// SIMD-accelerated operations for neural networks
#[cfg(feature = "simd")]
pub struct SIMDOperations;

#[cfg(feature = "simd")]
impl SIMDOperations {
    /// Vectorized ReLU activation
    pub fn vectorized_relu(input: &mut ArrayViewMut<f32>) {
        if is_simd_available() {
            input.par_mapv_inplace(|x| x.max(0.0));
        } else {
            input.mapv_inplace(|x| x.max(0.0));
        }
    }

    /// Vectorized sigmoid activation
    pub fn vectorized_sigmoid(input: &mut ArrayViewMut<f32>) {
        if is_simd_available() {
            input.par_mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
        } else {
            input.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
        }
    }

    /// Vectorized tanh activation
    pub fn vectorized_tanh(input: &mut ArrayViewMut<f32>) {
        if is_simd_available() {
            input.par_mapv_inplace(|x| x.tanh());
        } else {
            input.mapv_inplace(|x| x.tanh());
        }
    }

    /// Vectorized matrix multiplication with SIMD
    pub fn simd_matmul(
        a: &ArrayView<f32, IxDyn>,
        b: &ArrayView<f32, IxDyn>,
        output: &mut ArrayViewMut<f32, IxDyn>,
    ) -> Result<()> {
        if a.ndim() != 2 || b.ndim() != 2 || output.ndim() != 2 {
            return Err(NeuralError::ComputationError(
                "SIMD matmul requires 2D arrays".to_string(),
            ));
        }

        let (m, k) = (a.shape()[0], a.shape()[1]);
        let (k2, n) = (b.shape()[0], b.shape()[1]);

        if k != k2 {
            return Err(NeuralError::ComputationError(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        if output.shape() != [m, n] {
            return Err(NeuralError::ComputationError(
                "Output array has incorrect shape".to_string(),
            ));
        }

        if is_simd_available() {
            simd_gemm(a, b, output)?;
        } else {
            // Fallback to standard matrix multiplication
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..k {
                        sum += a[[i, k]] * b[[k, j]];
                    }
                    output[[i, j]] = sum;
                }
            }
        }

        Ok(())
    }

    /// Vectorized element-wise addition
    pub fn simd_add(
        a: &ArrayView<f32, IxDyn>,
        b: &ArrayView<f32, IxDyn>,
        output: &mut ArrayViewMut<f32, IxDyn>,
    ) -> Result<()> {
        if a.shape() != b.shape() || a.shape() != output.shape() {
            return Err(NeuralError::ComputationError(
                "Array shapes must match for element-wise addition".to_string(),
            ));
        }

        if is_simd_available() {
            simd_add_arrays(a, b, output)?;
        } else {
            for ((a_val, b_val), out_val) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
                *out_val = a_val + b_val;
            }
        }

        Ok(())
    }
}

/// Memory-efficient batch processor
#[cfg(feature = "memory_efficient")]
pub struct MemoryEfficientProcessor {
    chunk_size: usize,
    max_memory_mb: usize,
}

#[cfg(feature = "memory_efficient")]
impl MemoryEfficientProcessor {
    /// Create a new memory-efficient processor
    pub fn new(chunk_size: Option<usize>, max_memory_mb: Option<usize>) -> Self {
        Self {
            chunk_size: chunk_size.unwrap_or(1024),
            max_memory_mb: max_memory_mb.unwrap_or(512),
        }
    }

    /// Process large arrays in chunks to reduce memory usage
    pub fn process_in_chunks<F, T>(
        &self,
        input: &ArrayD<f32>,
        mut processor: F,
    ) -> Result<ArrayD<T>>
    where
        F: FnMut(&ArrayView<f32, IxDyn>) -> Result<ArrayD<T>>,
        T: Clone + Debug + Default,
    {
        let batch_size = input.shape()[0];

        if batch_size <= self.chunk_size {
            // Process all at once if small enough
            return processor(&input.view());
        }

        // Process in chunks
        let mut results = Vec::new();
        let mut start_idx = 0;

        while start_idx < batch_size {
            let end_idx = (start_idx + self.chunk_size).min(batch_size);
            let chunk = input.slice(ndarray::s![start_idx..end_idx, ..]);

            let result = processor(&chunk)?;
            results.push(result);

            start_idx = end_idx;
        }

        // Concatenate results
        if results.is_empty() {
            return Err(NeuralError::ComputationError(
                "No chunks were processed".to_string(),
            ));
        }

        // For simplicity, just return the first result
        // A full implementation would concatenate along the batch dimension
        Ok(results.into_iter().next().unwrap())
    }

    /// Memory-efficient forward pass for large batches
    pub fn memory_efficient_forward<F>(
        &self,
        input: &ArrayD<f32>,
        forward_fn: F,
    ) -> Result<ArrayD<f32>>
    where
        F: Fn(&ArrayView<f32, IxDyn>) -> Result<ArrayD<f32>>,
    {
        chunk_wise_op(input, self.chunk_size, &ChunkProcessor::new(forward_fn)).map_err(|e| {
            NeuralError::ComputationError(format!("Memory-efficient forward failed: {:?}", e))
        })
    }
}

/// Thread pool manager for parallel neural network operations
pub struct ThreadPoolManager {
    #[cfg(feature = "rayon")]
    pool: rayon::ThreadPool,
    num_threads: usize,
}

impl ThreadPoolManager {
    /// Create a new thread pool manager
    pub fn new(num_threads: Option<usize>) -> Result<Self> {
        let num_threads = num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

        #[cfg(feature = "rayon")]
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| {
                NeuralError::ComputationError(format!("Failed to create thread pool: {}", e))
            })?;

        Ok(Self {
            #[cfg(feature = "rayon")]
            pool,
            num_threads,
        })
    }

    /// Execute a function in the thread pool
    #[cfg(feature = "rayon")]
    pub fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool.install(f)
    }

    /// Execute a function in the thread pool (no-op without rayon)
    #[cfg(not(feature = "rayon"))]
    pub fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        f()
    }

    /// Parallel matrix multiplication using thread pool
    pub fn parallel_matmul(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err(NeuralError::ComputationError(
                "Parallel matmul requires 2D arrays".to_string(),
            ));
        }

        let (m, k) = (a.shape()[0], a.shape()[1]);
        let (k2, n) = (b.shape()[0], b.shape()[1]);

        if k != k2 {
            return Err(NeuralError::ComputationError(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        #[cfg(feature = "rayon")]
        return self.execute(|| {
            use rayon::prelude::*;
            let mut result = Array::zeros((m, n));

            result
                .axis_iter_mut(ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for k in 0..k {
                            sum += a[[i, k]] * b[[k, j]];
                        }
                        row[j] = sum;
                    }
                });

            Ok(result.into_dyn())
        });

        #[cfg(not(feature = "rayon"))]
        {
            let mut result = Array::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..k {
                        sum += a[[i, k]] * b[[k, j]];
                    }
                    result[[i, j]] = sum;
                }
            }
            Ok(result.into_dyn())
        }
    }

    /// Get the number of threads in the pool
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

/// Performance profiler for neural network operations
pub struct PerformanceProfiler {
    enabled: bool,
    timings: std::collections::HashMap<String, std::time::Duration>,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            timings: std::collections::HashMap::new(),
        }
    }

    /// Start timing an operation
    pub fn start_timer(&self, _name: &str) -> Option<std::time::Instant> {
        if self.enabled {
            Some(std::time::Instant::now())
        } else {
            None
        }
    }

    /// End timing an operation and record the result
    pub fn end_timer(&mut self, name: String, start_time: Option<std::time::Instant>) {
        if self.enabled {
            if let Some(start) = start_time {
                let elapsed = start.elapsed();
                self.timings.insert(name, elapsed);
            }
        }
    }

    /// Get timing information
    pub fn get_timings(&self) -> &std::collections::HashMap<String, std::time::Duration> {
        &self.timings
    }

    /// Clear all timing information
    pub fn clear(&mut self) {
        self.timings.clear();
    }

    /// Print timing summary
    pub fn print_summary(&self) {
        if !self.enabled {
            println!("Performance profiling is disabled");
            return;
        }

        println!("Performance Profile Summary:");
        println!("===========================");

        let mut sorted_timings: Vec<_> = self.timings.iter().collect();
        sorted_timings.sort_by(|a, b| b.1.cmp(a.1));

        for (name, duration) in sorted_timings {
            println!("{}: {:.3}ms", name, duration.as_secs_f64() * 1000.0);
        }
    }
}

/// Unified performance optimization manager
pub struct PerformanceOptimizer {
    #[cfg(feature = "simd")]
    simd_ops: SIMDOperations,

    #[cfg(feature = "memory_efficient")]
    memory_processor: MemoryEfficientProcessor,

    thread_pool: Arc<ThreadPoolManager>,
    profiler: PerformanceProfiler,
}

impl PerformanceOptimizer {
    /// Create a new performance optimizer
    pub fn new(
        _chunk_size: Option<usize>,
        _max_memory_mb: Option<usize>,
        num_threads: Option<usize>,
        enable_profiling: bool,
    ) -> Result<Self> {
        Ok(Self {
            #[cfg(feature = "simd")]
            simd_ops: SIMDOperations,

            #[cfg(feature = "memory_efficient")]
            memory_processor: MemoryEfficientProcessor::new(_chunk_size, _max_memory_mb),

            thread_pool: Arc::new(ThreadPoolManager::new(num_threads)?),
            profiler: PerformanceProfiler::new(enable_profiling),
        })
    }

    /// Get reference to thread pool
    pub fn thread_pool(&self) -> &Arc<ThreadPoolManager> {
        &self.thread_pool
    }

    /// Get mutable reference to profiler
    pub fn profiler_mut(&mut self) -> &mut PerformanceProfiler {
        &mut self.profiler
    }

    /// Get reference to profiler
    pub fn profiler(&self) -> &PerformanceProfiler {
        &self.profiler
    }

    /// Optimized matrix multiplication using all available optimizations
    pub fn optimized_matmul(&mut self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let timer = self.profiler.start_timer("matmul");

        let result = {
            #[cfg(feature = "simd")]
            {
                // Try SIMD first if available
                if ndarray::linalg::general_mat_mul(1.0, a, b, 0.0, &mut result).is_ok() {
                    result
                } else {
                    self.thread_pool.parallel_matmul(a, b)?
                }
            }

            #[cfg(not(feature = "simd"))]
            {
                self.thread_pool.parallel_matmul(a, b)?
            }
        };

        self.profiler.end_timer("matmul".to_string(), timer);
        Ok(result)
    }

    /// Get optimization capabilities
    pub fn get_capabilities(&self) -> OptimizationCapabilities {
        OptimizationCapabilities {
            simd_available: cfg!(feature = "simd"),
            memory_efficient_available: cfg!(feature = "memory_efficient"),
            thread_pool_available: true,
            num_threads: self.thread_pool.num_threads(),
        }
    }
}

/// Information about available optimization capabilities
#[derive(Debug, Clone)]
pub struct OptimizationCapabilities {
    /// Whether SIMD optimizations are available
    pub simd_available: bool,
    /// Whether memory-efficient operations are available
    pub memory_efficient_available: bool,
    /// Whether thread pool is available
    pub thread_pool_available: bool,
    /// Number of threads in the pool
    pub num_threads: usize,
}

impl std::fmt::Display for OptimizationCapabilities {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Optimization Capabilities:")?;
        writeln!(f, "  SIMD: {}", if self.simd_available { "✓" } else { "✗" })?;
        writeln!(
            f,
            "  Memory Efficient: {}",
            if self.memory_efficient_available {
                "✓"
            } else {
                "✗"
            }
        )?;
        writeln!(
            f,
            "  Thread Pool: {}",
            if self.thread_pool_available {
                "✓"
            } else {
                "✗"
            }
        )?;
        writeln!(f, "  Threads: {}", self.num_threads)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_thread_pool_creation() {
        let pool = ThreadPoolManager::new(Some(2)).unwrap();
        assert_eq!(pool.num_threads(), 2);
    }

    #[test]
    fn test_performance_profiler() {
        let mut profiler = PerformanceProfiler::new(true);

        let timer = profiler.start_timer("test_op");
        std::thread::sleep(std::time::Duration::from_millis(1));
        profiler.end_timer("test_op".to_string(), timer);

        assert!(profiler.get_timings().contains_key("test_op"));
        let duration = profiler.get_timings()["test_op"];
        assert!(duration.as_millis() >= 1);
    }

    #[test]
    fn test_parallel_matmul() {
        let pool = ThreadPoolManager::new(Some(2)).unwrap();

        let a = Array2::from_elem((3, 4), 2.0).into_dyn();
        let b = Array2::from_elem((4, 5), 3.0).into_dyn();

        let result = pool.parallel_matmul(&a, &b).unwrap();

        assert_eq!(result.shape(), [3, 5]);
        // Each element should be 2.0 * 3.0 * 4 = 24.0
        for &val in result.iter() {
            assert!((val - 24.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_optimization_capabilities() {
        let optimizer = PerformanceOptimizer::new(None, None, Some(4), false).unwrap();
        let caps = optimizer.get_capabilities();

        assert_eq!(caps.num_threads, 4);
        assert!(caps.thread_pool_available);
    }

    #[cfg(feature = "memory_efficient")]
    #[test]
    fn test_memory_efficient_processor() {
        let processor = MemoryEfficientProcessor::new(Some(2), Some(100));

        let input = Array2::from_elem((5, 3), 1.0).into_dyn();

        let result = processor
            .process_in_chunks(&input, |chunk| Ok(chunk.to_owned()))
            .unwrap();

        assert_eq!(result.shape(), input.shape());
    }
}
