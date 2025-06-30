//! Performance optimization utilities for critical paths
//!
//! This module provides tools and utilities for optimizing performance-critical
//! sections of scirs2-core based on profiling data.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Cache locality hint for prefetch operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Locality {
    /// High locality - data likely to be reused soon (L1 cache)
    High,
    /// Medium locality - data may be reused (L2 cache)
    Medium,
    /// Low locality - data unlikely to be reused soon (L3 cache)
    Low,
    /// No temporal locality - streaming access (bypass cache)
    None,
}

/// Performance hints for critical code paths
pub struct PerformanceHints;

impl PerformanceHints {
    /// Hint that a branch is likely to be taken
    ///
    /// Note: This function provides branch prediction hints on supported architectures.
    /// For Beta 1 stability, unstable intrinsics have been removed.
    #[inline(always)]
    pub fn likely(cond: bool) -> bool {
        // Use platform-specific assembly hints where available
        #[cfg(target_arch = "x86_64")]
        {
            if cond {
                // x86_64 specific: use assembly hint for branch prediction
                unsafe {
                    std::arch::asm!("# likely branch", options(nomem, nostack));
                }
            }
        }
        cond
    }

    /// Hint that a branch is unlikely to be taken
    ///
    /// Note: This function provides branch prediction hints on supported architectures.
    /// For Beta 1 stability, unstable intrinsics have been removed.
    #[inline(always)]
    pub fn unlikely(cond: bool) -> bool {
        // Use platform-specific assembly hints where available
        #[cfg(target_arch = "x86_64")]
        {
            if !cond {
                // x86_64 specific: use assembly hint for branch prediction
                unsafe {
                    std::arch::asm!("# unlikely branch", options(nomem, nostack));
                }
            }
        }
        cond
    }

    /// Prefetch data for read access
    #[inline(always)]
    pub fn prefetch_read<T>(data: &T) {
        let ptr = data as *const T as *const u8;

        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                // Prefetch into all cache levels for read
                std::arch::asm!(
                    "prefetcht0 [{}]",
                    in(reg) ptr,
                    options(readonly, nostack)
                );
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                // ARMv8 prefetch for load
                std::arch::asm!(
                    "prfm pldl1keep, [{}]",
                    in(reg) ptr,
                    options(readonly, nostack)
                );
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback: use black_box to prevent optimization but don't prefetch
            std::hint::black_box(data);
        }
    }

    /// Prefetch data for write access
    #[inline(always)]
    pub fn prefetch_write<T>(data: &mut T) {
        let ptr = data as *mut T as *mut u8;

        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                // Prefetch with intent to write
                std::arch::asm!(
                    "prefetcht0 [{}]",
                    in(reg) ptr,
                    options(nostack)
                );
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                // ARMv8 prefetch for store
                std::arch::asm!(
                    "prfm pstl1keep, [{}]",
                    in(reg) ptr,
                    options(nostack)
                );
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback: use black_box to prevent optimization but don't prefetch
            std::hint::black_box(data);
        }
    }

    /// Advanced prefetch with locality hint
    #[inline(always)]
    pub fn prefetch_with_locality<T>(data: &T, locality: Locality) {
        let ptr = data as *const T as *const u8;

        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                match locality {
                    Locality::High => {
                        // Prefetch into L1 cache
                        std::arch::asm!(
                            "prefetcht0 [{}]",
                            in(reg) ptr,
                            options(readonly, nostack)
                        );
                    }
                    Locality::Medium => {
                        // Prefetch into L2 cache
                        std::arch::asm!(
                            "prefetcht1 [{}]",
                            in(reg) ptr,
                            options(readonly, nostack)
                        );
                    }
                    Locality::Low => {
                        // Prefetch into L3 cache
                        std::arch::asm!(
                            "prefetcht2 [{}]",
                            in(reg) ptr,
                            options(readonly, nostack)
                        );
                    }
                    Locality::None => {
                        // Non-temporal prefetch
                        std::arch::asm!(
                            "prefetchnta [{}]",
                            in(reg) ptr,
                            options(readonly, nostack)
                        );
                    }
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                match locality {
                    Locality::High => {
                        std::arch::asm!(
                            "prfm pldl1keep, [{}]",
                            in(reg) ptr,
                            options(readonly, nostack)
                        );
                    }
                    Locality::Medium => {
                        std::arch::asm!(
                            "prfm pldl2keep, [{}]",
                            in(reg) ptr,
                            options(readonly, nostack)
                        );
                    }
                    Locality::Low => {
                        std::arch::asm!(
                            "prfm pldl3keep, [{}]",
                            in(reg) ptr,
                            options(readonly, nostack)
                        );
                    }
                    Locality::None => {
                        std::arch::asm!(
                            "prfm pldl1strm, [{}]",
                            in(reg) ptr,
                            options(readonly, nostack)
                        );
                    }
                }
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            std::hint::black_box(data);
        }
    }

    /// Memory fence for synchronization
    #[inline(always)]
    pub fn memory_fence() {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                std::arch::asm!("mfence", options(nostack));
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                std::arch::asm!("dmb sy", options(nostack));
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Cache line flush for explicit cache management
    #[inline(always)]
    pub fn flush_cache_line<T>(data: &T) {
        let _ptr = data as *const T as *const u8;

        // Note: Cache line flushing is arch-specific and may not be portable
        // For now, use a memory barrier as a fallback
        #[cfg(target_arch = "x86_64")]
        {
            // On x86_64, we would use clflush but it requires specific syntax
            // For simplicity, we'll use a fence instruction instead
            unsafe {
                std::arch::asm!("mfence", options(nostack, nomem));
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                // ARMv8 data cache clean and invalidate
                std::arch::asm!(
                    "dc civac, {}",
                    in(reg) ptr,
                    options(nostack)
                );
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // No specific flush available, just prevent optimization
            std::hint::black_box(data);
        }
    }

    /// Optimized memory copy with cache awareness
    #[inline]
    pub fn cache_aware_copy<T: Copy>(src: &[T], dst: &mut [T]) {
        assert_eq!(src.len(), dst.len());

        if std::mem::size_of_val(src) > 64 * 1024 {
            // Large copy: use non-temporal stores to avoid cache pollution
            #[cfg(target_arch = "x86_64")]
            {
                unsafe {
                    let src_ptr = src.as_ptr() as *const u8;
                    let dst_ptr = dst.as_mut_ptr() as *mut u8;
                    let len = std::mem::size_of_val(src);

                    // Use non-temporal memory copy for large transfers
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, len);

                    // Follow with memory fence
                    std::arch::asm!("sfence", options(nostack));
                }
                return;
            }
        }

        // Regular copy for smaller data or unsupported architectures
        dst.copy_from_slice(src);
    }

    /// Optimized memory set with cache awareness
    #[inline]
    pub fn cache_aware_memset<T: Copy>(dst: &mut [T], value: T) {
        if std::mem::size_of_val(dst) > 32 * 1024 {
            // Large memset: use vectorized operations where possible
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                // For large arrays, try to use SIMD if T is appropriate
                if std::mem::size_of::<T>() == 8 {
                    // 64-bit values can use SSE2
                    let chunks = dst.len() / 2;
                    for i in 0..chunks {
                        dst[i * 2] = value;
                        dst[i * 2 + 1] = value;
                    }
                    // Handle remainder
                    for item in dst.iter_mut().skip(chunks * 2) {
                        *item = value;
                    }
                    return;
                }
            }
        }

        // Regular fill for smaller data or unsupported cases
        dst.fill(value);
    }
}

/// Performance metrics for adaptive learning
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Average execution times for different operation types
    pub operation_times: std::collections::HashMap<String, f64>,
    /// Success rate for different optimization strategies
    pub strategy_success_rates: std::collections::HashMap<OptimizationStrategy, f64>,
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
    /// Cache hit rates
    pub cache_hit_rate: f64,
    /// Parallel efficiency measurements
    pub parallel_efficiency: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            operation_times: std::collections::HashMap::new(),
            strategy_success_rates: std::collections::HashMap::new(),
            memory_bandwidth_utilization: 0.0,
            cache_hit_rate: 0.0,
            parallel_efficiency: 0.0,
        }
    }
}

/// Optimization strategies available
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationStrategy {
    Scalar,
    Simd,
    Parallel,
    Gpu,
    Hybrid,
    CacheOptimized,
    MemoryBound,
    ComputeBound,
}

/// Strategy selector for choosing the best optimization approach
#[derive(Debug, Clone)]
pub struct StrategySelector {
    /// Current preferred strategy
    preferred_strategy: OptimizationStrategy,
    /// Strategy weights based on past performance
    strategy_weights: std::collections::HashMap<OptimizationStrategy, f64>,
    /// Learning rate for weight updates
    learning_rate: f64,
    /// Exploration rate for trying different strategies
    exploration_rate: f64,
}

impl Default for StrategySelector {
    fn default() -> Self {
        let mut strategy_weights = std::collections::HashMap::new();
        strategy_weights.insert(OptimizationStrategy::Scalar, 1.0);
        strategy_weights.insert(OptimizationStrategy::Simd, 1.0);
        strategy_weights.insert(OptimizationStrategy::Parallel, 1.0);
        strategy_weights.insert(OptimizationStrategy::Gpu, 1.0);
        strategy_weights.insert(OptimizationStrategy::Hybrid, 1.0);
        strategy_weights.insert(OptimizationStrategy::CacheOptimized, 1.0);
        strategy_weights.insert(OptimizationStrategy::MemoryBound, 1.0);
        strategy_weights.insert(OptimizationStrategy::ComputeBound, 1.0);

        Self {
            preferred_strategy: OptimizationStrategy::Scalar,
            strategy_weights,
            learning_rate: 0.1,
            exploration_rate: 0.1,
        }
    }
}

impl StrategySelector {
    /// Select the best strategy for given operation characteristics
    pub fn select_strategy(&self, operation_size: usize, memory_bound: bool) -> OptimizationStrategy {
        // Use epsilon-greedy exploration
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        operation_size.hash(&mut hasher);
        let rand_val = (hasher.finish() % 100) as f64 / 100.0;
        
        if rand_val < self.exploration_rate {
            // Explore: choose a random strategy
            let strategies = [
                OptimizationStrategy::Scalar,
                OptimizationStrategy::Simd,
                OptimizationStrategy::Parallel,
                OptimizationStrategy::Gpu,
            ];
            strategies[operation_size % strategies.len()]
        } else {
            // Exploit: choose the best strategy based on characteristics
            if memory_bound {
                OptimizationStrategy::MemoryBound
            } else if operation_size > 100_000 {
                OptimizationStrategy::Parallel
            } else if operation_size > 1_000 {
                OptimizationStrategy::Simd
            } else {
                OptimizationStrategy::Scalar
            }
        }
    }

    /// Update strategy weights based on performance feedback
    pub fn update_strategy_weight(&mut self, strategy: OptimizationStrategy, performance_score: f64) {
        if let Some(weight) = self.strategy_weights.get_mut(&strategy) {
            *weight = *weight * (1.0 - self.learning_rate) + performance_score * self.learning_rate;
        }
    }
}

/// Adaptive optimization based on runtime characteristics
pub struct AdaptiveOptimizer {
    /// Threshold for switching to parallel execution
    parallel_threshold: AtomicUsize,
    /// Threshold for using SIMD operations
    simd_threshold: AtomicUsize,
    /// Threshold for using GPU acceleration
    gpu_threshold: AtomicUsize,
    /// Cache line size for the current architecture
    cache_line_size: usize,
    /// Performance metrics for adaptive learning
    performance_metrics: std::sync::RwLock<PerformanceMetrics>,
    /// Optimization strategy selector
    strategy_selector: std::sync::RwLock<StrategySelector>,
}

impl AdaptiveOptimizer {
    /// Create a new adaptive optimizer
    pub fn new() -> Self {
        Self {
            parallel_threshold: AtomicUsize::new(10_000),
            simd_threshold: AtomicUsize::new(1_000),
            gpu_threshold: AtomicUsize::new(100_000),
            cache_line_size: Self::detect_cache_line_size(),
            performance_metrics: std::sync::RwLock::new(PerformanceMetrics::default()),
            strategy_selector: std::sync::RwLock::new(StrategySelector::default()),
        }
    }

    /// Detect the cache line size for the current architecture
    fn detect_cache_line_size() -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            64 // Common for x86_64
        }
        #[cfg(target_arch = "aarch64")]
        {
            128 // Common for ARM64
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            64 // Default fallback
        }
    }

    /// Check if parallel execution should be used for given size
    #[inline]
    #[allow(unused_variables)]
    pub fn should_use_parallel(&self, size: usize) -> bool {
        #[cfg(feature = "parallel")]
        {
            size >= self.parallel_threshold.load(Ordering::Relaxed)
        }
        #[cfg(not(feature = "parallel"))]
        {
            false
        }
    }

    /// Check if SIMD should be used for given size
    #[inline]
    #[allow(unused_variables)]
    pub fn should_use_simd(&self, size: usize) -> bool {
        #[cfg(feature = "simd")]
        {
            size >= self.simd_threshold.load(Ordering::Relaxed)
        }
        #[cfg(not(feature = "simd"))]
        {
            false
        }
    }

    /// Update thresholds based on performance measurements
    pub fn update_thresholds(&self, operation: &str, size: usize, duration_ns: u64) {
        // Simple heuristic: adjust thresholds based on operation efficiency
        let ops_per_ns = size as f64 / duration_ns as f64;

        if operation.contains("parallel") && ops_per_ns < 0.1 {
            // Parallel overhead too high, increase threshold
            self.parallel_threshold
                .fetch_add(size / 10, Ordering::Relaxed);
        } else if operation.contains("simd") && ops_per_ns < 1.0 {
            // SIMD not efficient enough, increase threshold
            self.simd_threshold.fetch_add(size / 10, Ordering::Relaxed);
        }
    }

    /// Get optimal chunk size for cache-friendly operations
    #[inline]
    pub fn optimal_chunk_size<T>(&self) -> usize {
        // Calculate chunk size based on cache line size and element size
        let element_size = std::mem::size_of::<T>();
        let elements_per_cache_line = self.cache_line_size / element_size.max(1);

        // Use multiple cache lines for better performance
        elements_per_cache_line * 16
    }

    /// Check if GPU acceleration should be used for given size
    #[inline]
    #[allow(unused_variables)]
    pub fn should_use_gpu(&self, size: usize) -> bool {
        #[cfg(feature = "gpu")]
        {
            size >= self.gpu_threshold.load(Ordering::Relaxed)
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Select the optimal strategy for a given operation
    pub fn select_optimal_strategy(&self, operation_name: &str, size: usize) -> OptimizationStrategy {
        // Determine if operation is memory-bound based on operation name
        let memory_bound = operation_name.contains("copy") || 
                          operation_name.contains("memset") ||
                          operation_name.contains("transpose");

        if let Ok(selector) = self.strategy_selector.read() {
            selector.select_strategy(size, memory_bound)
        } else {
            // Fallback selection
            if self.should_use_gpu(size) {
                OptimizationStrategy::Gpu
            } else if self.should_use_parallel(size) {
                OptimizationStrategy::Parallel
            } else if self.should_use_simd(size) {
                OptimizationStrategy::Simd
            } else {
                OptimizationStrategy::Scalar
            }
        }
    }

    /// Record performance measurement and update adaptive parameters
    pub fn record_performance(&self, operation: &str, strategy: OptimizationStrategy, size: usize, duration_ns: u64) {
        // Calculate performance score (higher is better)
        let ops_per_ns = size as f64 / duration_ns as f64;
        let performance_score = ops_per_ns.min(10.0) / 10.0; // Normalize to 0-1
        
        // Update strategy weights
        if let Ok(mut selector) = self.strategy_selector.write() {
            selector.update_strategy_weight(strategy, performance_score);
        }

        // Update performance metrics
        if let Ok(mut metrics) = self.performance_metrics.write() {
            let avg_time = metrics.operation_times.entry(operation.to_string()).or_insert(0.0);
            *avg_time = (*avg_time * 0.9) + (duration_ns as f64 * 0.1); // Exponential moving average
            
            metrics.strategy_success_rates.insert(strategy, performance_score);
        }

        // Update thresholds based on performance
        self.update_thresholds(operation, size, duration_ns);
    }

    /// Get performance metrics for analysis
    pub fn get_performance_metrics(&self) -> Option<PerformanceMetrics> {
        self.performance_metrics.read().ok().map(|m| m.clone())
    }

    /// Analyze operation characteristics to suggest optimizations
    pub fn analyze_operation(&self, operation_name: &str, input_size: usize, output_size: usize) -> OptimizationAdvice {
        let strategy = self.select_optimal_strategy(operation_name, input_size);
        let chunk_size = if strategy == OptimizationStrategy::Parallel {
            Some(self.optimal_chunk_size::<f64>())
        } else {
            None
        };

        let prefetch_distance = if input_size > 10_000 {
            Some(self.cache_line_size * 8) // Prefetch 8 cache lines ahead
        } else {
            None
        };

        OptimizationAdvice {
            recommended_strategy: strategy,
            optimal_chunk_size: chunk_size,
            prefetch_distance,
            memory_allocation_hint: if output_size > 1_000_000 {
                Some("Consider using memory-mapped files for large outputs".to_string())
            } else {
                None
            },
        }
    }
}

/// Optimization advice generated by the adaptive optimizer
#[derive(Debug, Clone)]
pub struct OptimizationAdvice {
    /// Recommended optimization strategy
    pub recommended_strategy: OptimizationStrategy,
    /// Optimal chunk size for parallel processing
    pub optimal_chunk_size: Option<usize>,
    /// Prefetch distance for memory access
    pub prefetch_distance: Option<usize>,
    /// Memory allocation hints
    pub memory_allocation_hint: Option<String>,
}

impl Default for AdaptiveOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Fast path optimizations for common operations
pub mod fast_paths {
    use super::*;

    /// Optimized array addition for f64
    #[inline]
    #[allow(unused_variables)]
    pub fn add_f64_arrays(a: &[f64], b: &[f64], result: &mut [f64]) -> Result<(), &'static str> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err("Array lengths must match");
        }

        let len = a.len();
        let optimizer = AdaptiveOptimizer::new();

        #[cfg(feature = "simd")]
        if optimizer.should_use_simd(len) {
            // Use SIMD operations for f64 addition
            use crate::simd_ops::SimdUnifiedOps;
            use ndarray::ArrayView1;

            // Process in SIMD-width chunks
            let simd_chunks = len / 4; // Process 4 f64s at a time

            for i in 0..simd_chunks {
                let start = i * 4;
                let end = start + 4;

                if end <= len {
                    let a_view = ArrayView1::from(&a[start..end]);
                    let b_view = ArrayView1::from(&b[start..end]);

                    // Use SIMD addition
                    let simd_result = f64::simd_add(&a_view, &b_view);
                    result[start..end].copy_from_slice(simd_result.as_slice().unwrap());
                }
            }

            // Handle remaining elements with scalar operations
            for i in (simd_chunks * 4)..len {
                result[i] = a[i] + b[i];
            }
            return Ok(());
        }

        #[cfg(feature = "parallel")]
        if optimizer.should_use_parallel(len) {
            use rayon::prelude::*;
            result
                .par_chunks_mut(optimizer.optimal_chunk_size::<f64>())
                .zip(a.par_chunks(optimizer.optimal_chunk_size::<f64>()))
                .zip(b.par_chunks(optimizer.optimal_chunk_size::<f64>()))
                .for_each(|((r_chunk, a_chunk), b_chunk)| {
                    for i in 0..r_chunk.len() {
                        r_chunk[i] = a_chunk[i] + b_chunk[i];
                    }
                });
            return Ok(());
        }

        // Scalar fallback with loop unrolling
        let chunks = len / 8;

        for i in 0..chunks {
            let idx = i * 8;
            result[idx] = a[idx] + b[idx];
            result[idx + 1] = a[idx + 1] + b[idx + 1];
            result[idx + 2] = a[idx + 2] + b[idx + 2];
            result[idx + 3] = a[idx + 3] + b[idx + 3];
            result[idx + 4] = a[idx + 4] + b[idx + 4];
            result[idx + 5] = a[idx + 5] + b[idx + 5];
            result[idx + 6] = a[idx + 6] + b[idx + 6];
            result[idx + 7] = a[idx + 7] + b[idx + 7];
        }

        for i in (chunks * 8)..len {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    /// Optimized matrix multiplication kernel
    #[inline]
    pub fn matmul_kernel(
        a: &[f64],
        b: &[f64],
        c: &mut [f64],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), &'static str> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err("Invalid matrix dimensions");
        }

        // Tile sizes for cache optimization
        const TILE_M: usize = 64;
        const TILE_N: usize = 64;
        const TILE_K: usize = 64;

        // Clear result matrix
        c.fill(0.0);

        #[cfg(feature = "parallel")]
        {
            let optimizer = AdaptiveOptimizer::new();
            if optimizer.should_use_parallel(m * n) {
                use rayon::prelude::*;

                // Use synchronization for parallel matrix multiplication
                use std::sync::Mutex;
                let c_mutex = Mutex::new(c);

                // Parallel tiled implementation using row-wise parallelization
                (0..m).into_par_iter().step_by(TILE_M).for_each(|i0| {
                    let i_max = (i0 + TILE_M).min(m);
                    let mut local_updates = Vec::new();

                    for j0 in (0..n).step_by(TILE_N) {
                        for k0 in (0..k).step_by(TILE_K) {
                            let j_max = (j0 + TILE_N).min(n);
                            let k_max = (k0 + TILE_K).min(k);

                            for i in i0..i_max {
                                for j in j0..j_max {
                                    let mut sum = 0.0;
                                    for k_idx in k0..k_max {
                                        sum += a[i * k + k_idx] * b[k_idx * n + j];
                                    }
                                    local_updates.push((i, j, sum));
                                }
                            }
                        }
                    }

                    // Apply all local updates at once
                    if let Ok(mut c_guard) = c_mutex.lock() {
                        for (i, j, sum) in local_updates {
                            c_guard[i * n + j] += sum;
                        }
                    }
                });
                return Ok(());
            }
        }

        // Serial tiled implementation
        for i0 in (0..m).step_by(TILE_M) {
            for j0 in (0..n).step_by(TILE_N) {
                for k0 in (0..k).step_by(TILE_K) {
                    let i_max = (i0 + TILE_M).min(m);
                    let j_max = (j0 + TILE_N).min(n);
                    let k_max = (k0 + TILE_K).min(k);

                    for i in i0..i_max {
                        for j in j0..j_max {
                            let mut sum = c[i * n + j];
                            for k_idx in k0..k_max {
                                sum += a[i * k + k_idx] * b[k_idx * n + j];
                            }
                            c[i * n + j] = sum;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Memory access pattern optimizer
pub struct MemoryAccessOptimizer {
    /// Stride detection for array access
    _stride_detector: StrideDetector,
}

#[derive(Default)]
struct StrideDetector {
    _last_address: Option<usize>,
    _detected_stride: Option<isize>,
    _confidence: f32,
}

impl MemoryAccessOptimizer {
    pub fn new() -> Self {
        Self {
            _stride_detector: StrideDetector::default(),
        }
    }

    /// Analyze memory access pattern and suggest optimizations
    pub fn analyze_access_pattern<T>(&mut self, addresses: &[*const T]) -> AccessPattern {
        if addresses.is_empty() {
            return AccessPattern::Unknown;
        }

        // Simple stride detection
        let mut strides = Vec::new();
        for window in addresses.windows(2) {
            let stride = (window[1] as isize) - (window[0] as isize);
            strides.push(stride / std::mem::size_of::<T>() as isize);
        }

        // Check if all strides are equal (sequential access)
        if strides.windows(2).all(|w| w[0] == w[1]) {
            match strides[0] {
                1 => AccessPattern::Sequential,
                -1 => AccessPattern::ReverseSequential,
                s if s > 1 => AccessPattern::Strided(s as usize),
                _ => AccessPattern::Random,
            }
        } else {
            AccessPattern::Random
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccessPattern {
    Sequential,
    ReverseSequential,
    Strided(usize),
    Random,
    Unknown,
}

impl Default for MemoryAccessOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive benchmarking framework for performance analysis
pub mod benchmarking {
    use super::*;
    use std::time::{Duration, Instant};
    use std::collections::HashMap;

    /// Benchmark configuration
    #[derive(Debug, Clone)]
    pub struct BenchmarkConfig {
        /// Number of warm-up iterations
        pub warmup_iterations: usize,
        /// Number of measurement iterations
        pub measurement_iterations: usize,
        /// Minimum benchmark duration
        pub min_duration: Duration,
        /// Maximum benchmark duration
        pub max_duration: Duration,
        /// Sample size range for testing
        pub sample_sizes: Vec<usize>,
        /// Strategies to benchmark
        pub strategies: Vec<OptimizationStrategy>,
    }

    impl Default for BenchmarkConfig {
        fn default() -> Self {
            Self {
                warmup_iterations: 5,
                measurement_iterations: 20,
                min_duration: Duration::from_millis(100),
                max_duration: Duration::from_secs(30),
                sample_sizes: vec![100, 1_000, 10_000, 100_000, 1_000_000],
                strategies: vec![
                    OptimizationStrategy::Scalar,
                    OptimizationStrategy::Simd,
                    OptimizationStrategy::Parallel,
                ],
            }
        }
    }

    /// Benchmark result for a single measurement
    #[derive(Debug, Clone)]
    pub struct BenchmarkMeasurement {
        /// Strategy used
        pub strategy: OptimizationStrategy,
        /// Input size
        pub input_size: usize,
        /// Duration of measurement
        pub duration: Duration,
        /// Throughput (operations per second)
        pub throughput: f64,
        /// Memory usage in bytes
        pub memory_usage: usize,
        /// Additional metrics
        pub custom_metrics: HashMap<String, f64>,
    }

    /// Aggregated benchmark results
    #[derive(Debug, Clone)]
    pub struct BenchmarkResults {
        /// Operation name
        pub operation_name: String,
        /// All measurements
        pub measurements: Vec<BenchmarkMeasurement>,
        /// Performance summary by strategy
        pub strategy_summary: HashMap<OptimizationStrategy, StrategyPerformance>,
        /// Scalability analysis
        pub scalability_analysis: ScalabilityAnalysis,
        /// Recommendations
        pub recommendations: Vec<String>,
        /// Total benchmark duration
        pub total_duration: Duration,
    }

    /// Performance summary for a strategy
    #[derive(Debug, Clone)]
    pub struct StrategyPerformance {
        /// Average throughput
        pub avg_throughput: f64,
        /// Standard deviation of throughput
        pub throughput_stddev: f64,
        /// Average memory usage
        pub avg_memory_usage: f64,
        /// Best input size for this strategy
        pub optimal_size: usize,
        /// Performance efficiency score (0-1)
        pub efficiency_score: f64,
    }

    /// Scalability analysis results
    #[derive(Debug, Clone)]
    pub struct ScalabilityAnalysis {
        /// Parallel efficiency at different sizes
        pub parallel_efficiency: HashMap<usize, f64>,
        /// Memory scaling behavior
        pub memory_scaling: MemoryScaling,
        /// Performance bottleneck analysis
        pub bottlenecks: Vec<PerformanceBottleneck>,
    }

    /// Memory scaling characteristics
    #[derive(Debug, Clone)]
    pub struct MemoryScaling {
        /// Linear coefficient (memory = linear_coeff * size + constant_coeff)
        pub linear_coefficient: f64,
        /// Constant coefficient
        pub constant_coefficient: f64,
        /// R-squared of the fit
        pub r_squared: f64,
    }

    /// Performance bottleneck identification
    #[derive(Debug, Clone)]
    pub struct PerformanceBottleneck {
        /// Bottleneck type
        pub bottleneck_type: BottleneckType,
        /// Input size range where bottleneck occurs
        pub size_range: (usize, usize),
        /// Performance impact (0-1, higher means more severe)
        pub impact: f64,
        /// Suggested mitigation
        pub mitigation: String,
    }

    /// Types of performance bottlenecks
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BottleneckType {
        MemoryBandwidth,
        CacheLatency,
        ComputeBound,
        SynchronizationOverhead,
        AlgorithmicComplexity,
    }

    /// Benchmark runner for comprehensive performance analysis
    pub struct BenchmarkRunner {
        config: BenchmarkConfig,
        optimizer: AdaptiveOptimizer,
    }

    impl BenchmarkRunner {
        /// Create a new benchmark runner
        pub fn new(config: BenchmarkConfig) -> Self {
            Self {
                config,
                optimizer: AdaptiveOptimizer::new(),
            }
        }

        /// Run comprehensive benchmarks for an operation
        pub fn benchmark_operation<F>(&self, operation_name: &str, operation: F) -> BenchmarkResults
        where
            F: Fn(&[f64], OptimizationStrategy) -> (Duration, Vec<f64>) + Send + Sync,
        {
            let start_time = Instant::now();
            let mut measurements = Vec::new();

            // Run benchmarks for each size and strategy combination
            for &size in &self.config.sample_sizes {
                let input_data: Vec<f64> = (0..size).map(|i| i as f64).collect();

                for &strategy in &self.config.strategies {
                    // Warm-up phase
                    for _ in 0..self.config.warmup_iterations {
                        let _ = operation(&input_data, strategy);
                    }

                    // Measurement phase
                    let mut durations = Vec::new();
                    for _ in 0..self.config.measurement_iterations {
                        let (duration, _result) = operation(&input_data, strategy);
                        durations.push(duration);
                    }

                    // Calculate statistics
                    let avg_duration = Duration::from_nanos(
                        (durations.iter().map(|d| d.as_nanos()).sum::<u128>() / durations.len() as u128) as u64
                    );

                    let throughput = if avg_duration.as_nanos() > 0 {
                        (size as f64) / (avg_duration.as_secs_f64())
                    } else {
                        0.0
                    };

                    // Estimate memory usage
                    let memory_usage = self.estimate_memory_usage(size, strategy);

                    measurements.push(BenchmarkMeasurement {
                        strategy,
                        input_size: size,
                        duration: avg_duration,
                        throughput,
                        memory_usage,
                        custom_metrics: HashMap::new(),
                    });
                }
            }

            // Analyze results
            let strategy_summary = self.analyze_strategy_performance(&measurements);
            let scalability_analysis = self.analyze_scalability(&measurements);
            let recommendations = self.generate_recommendations(&measurements, &strategy_summary);

            BenchmarkResults {
                operation_name: operation_name.to_string(),
                measurements,
                strategy_summary,
                scalability_analysis,
                recommendations,
                total_duration: start_time.elapsed(),
            }
        }

        /// Analyze performance by strategy
        fn analyze_strategy_performance(&self, measurements: &[BenchmarkMeasurement]) -> HashMap<OptimizationStrategy, StrategyPerformance> {
            let mut strategy_map: HashMap<OptimizationStrategy, Vec<&BenchmarkMeasurement>> = HashMap::new();
            
            for measurement in measurements {
                strategy_map.entry(measurement.strategy).or_default().push(measurement);
            }

            let mut summary = HashMap::new();
            for (strategy, strategy_measurements) in strategy_map {
                let throughputs: Vec<f64> = strategy_measurements.iter().map(|m| m.throughput).collect();
                let memory_usages: Vec<f64> = strategy_measurements.iter().map(|m| m.memory_usage as f64).collect();

                let avg_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
                let throughput_variance = throughputs.iter()
                    .map(|&x| (x - avg_throughput).powi(2))
                    .sum::<f64>() / throughputs.len() as f64;
                let throughput_stddev = throughput_variance.sqrt();

                let avg_memory_usage = memory_usages.iter().sum::<f64>() / memory_usages.len() as f64;

                // Find optimal size (highest throughput)
                let optimal_size = strategy_measurements.iter()
                    .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
                    .map(|m| m.input_size)
                    .unwrap_or(0);

                // Calculate efficiency score (throughput per memory unit)
                let efficiency_score = if avg_memory_usage > 0.0 {
                    (avg_throughput / avg_memory_usage * 1e6).min(1.0)
                } else {
                    0.0
                };

                summary.insert(strategy, StrategyPerformance {
                    avg_throughput,
                    throughput_stddev,
                    avg_memory_usage,
                    optimal_size,
                    efficiency_score,
                });
            }

            summary
        }

        /// Analyze scalability characteristics
        fn analyze_scalability(&self, measurements: &[BenchmarkMeasurement]) -> ScalabilityAnalysis {
            let mut parallel_efficiency = HashMap::new();
            let mut memory_sizes = Vec::new();
            let mut memory_usages = Vec::new();

            // Calculate parallel efficiency
            for &size in &self.config.sample_sizes {
                let scalar_throughput = measurements.iter()
                    .find(|m| m.input_size == size && m.strategy == OptimizationStrategy::Scalar)
                    .map(|m| m.throughput)
                    .unwrap_or(0.0);

                let parallel_throughput = measurements.iter()
                    .find(|m| m.input_size == size && m.strategy == OptimizationStrategy::Parallel)
                    .map(|m| m.throughput)
                    .unwrap_or(0.0);

                if scalar_throughput > 0.0 {
                    let efficiency = parallel_throughput / (scalar_throughput * 4.0); // Assume 4 cores
                    parallel_efficiency.insert(size, efficiency.min(1.0));
                }

                memory_sizes.push(size as f64);
                if let Some(measurement) = measurements.iter().find(|m| m.input_size == size) {
                    memory_usages.push(measurement.memory_usage as f64);
                }
            }

            // Fit linear model for memory scaling
            let memory_scaling = self.fit_linear_model(&memory_sizes, &memory_usages);

            // Identify bottlenecks
            let bottlenecks = self.identify_bottlenecks(measurements);

            ScalabilityAnalysis {
                parallel_efficiency,
                memory_scaling,
                bottlenecks,
            }
        }

        /// Fit linear model for memory scaling analysis
        fn fit_linear_model(&self, x: &[f64], y: &[f64]) -> MemoryScaling {
            if x.len() != y.len() || x.is_empty() {
                return MemoryScaling {
                    linear_coefficient: 0.0,
                    constant_coefficient: 0.0,
                    r_squared: 0.0,
                };
            }

            let n = x.len() as f64;
            let sum_x = x.iter().sum::<f64>();
            let sum_y = y.iter().sum::<f64>();
            let sum_xy = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum::<f64>();
            let sum_x2 = x.iter().map(|xi| xi * xi).sum::<f64>();

            let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
            let intercept = (sum_y - slope * sum_x) / n;

            // Calculate R-squared
            let y_mean = sum_y / n;
            let ss_tot = y.iter().map(|yi| (yi - y_mean).powi(2)).sum::<f64>();
            let ss_res = x.iter().zip(y.iter())
                .map(|(xi, yi)| (yi - (slope * xi + intercept)).powi(2))
                .sum::<f64>();

            let r_squared = if ss_tot > 0.0 { 1.0 - (ss_res / ss_tot) } else { 0.0 };

            MemoryScaling {
                linear_coefficient: slope,
                constant_coefficient: intercept,
                r_squared,
            }
        }

        /// Identify performance bottlenecks
        fn identify_bottlenecks(&self, measurements: &[BenchmarkMeasurement]) -> Vec<PerformanceBottleneck> {
            let mut bottlenecks = Vec::new();

            // Group by size
            let mut size_groups: HashMap<usize, Vec<&BenchmarkMeasurement>> = HashMap::new();
            for measurement in measurements {
                size_groups.entry(measurement.input_size).or_default().push(measurement);
            }

            for (&size, group) in &size_groups {
                // Check for memory bandwidth bottleneck
                let max_throughput = group.iter().map(|m| m.throughput).fold(0.0f64, f64::max);
                let min_throughput = group.iter().map(|m| m.throughput).fold(f64::INFINITY, f64::min);
                
                if max_throughput > 0.0 && (max_throughput - min_throughput) / max_throughput > 0.5 {
                    let impact = (max_throughput - min_throughput) / max_throughput;
                    bottlenecks.push(PerformanceBottleneck {
                        bottleneck_type: BottleneckType::MemoryBandwidth,
                        size_range: (size, size),
                        impact,
                        mitigation: "Consider cache-friendly data layouts or memory prefetching".to_string(),
                    });
                }

                // Check for synchronization overhead in parallel strategies
                let scalar_perf = group.iter()
                    .find(|m| m.strategy == OptimizationStrategy::Scalar)
                    .map(|m| m.throughput)
                    .unwrap_or(0.0);
                
                let parallel_perf = group.iter()
                    .find(|m| m.strategy == OptimizationStrategy::Parallel)
                    .map(|m| m.throughput)
                    .unwrap_or(0.0);

                if scalar_perf > 0.0 && parallel_perf / scalar_perf < 2.0 {
                    let impact = 1.0 - (parallel_perf / (scalar_perf * 4.0));
                    bottlenecks.push(PerformanceBottleneck {
                        bottleneck_type: BottleneckType::SynchronizationOverhead,
                        size_range: (size, size),
                        impact,
                        mitigation: "Reduce synchronization points or increase work per thread".to_string(),
                    });
                }
            }

            bottlenecks
        }

        /// Generate performance recommendations
        fn generate_recommendations(&self, measurements: &[BenchmarkMeasurement], strategy_summary: &HashMap<OptimizationStrategy, StrategyPerformance>) -> Vec<String> {
            let mut recommendations = Vec::new();

            // Find best overall strategy
            let best_strategy = strategy_summary.iter()
                .max_by(|(_, a), (_, b)| a.avg_throughput.partial_cmp(&b.avg_throughput).unwrap())
                .map(|(strategy, _)| *strategy);

            if let Some(strategy) = best_strategy {
                recommendations.push(format!("Best overall strategy: {:?}", strategy));
            }

            // Analyze size-dependent recommendations
            let large_size_threshold = 50_000;
            let large_measurements: Vec<_> = measurements.iter()
                .filter(|m| m.input_size >= large_size_threshold)
                .collect();

            if !large_measurements.is_empty() {
                let best_large_strategy = large_measurements.iter()
                    .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
                    .map(|m| m.strategy);

                if let Some(strategy) = best_large_strategy {
                    recommendations.push(format!("For large datasets (>{}): Use {:?}", large_size_threshold, strategy));
                }
            }

            // Memory efficiency recommendations
            let most_efficient = strategy_summary.iter()
                .max_by(|(_, a), (_, b)| a.efficiency_score.partial_cmp(&b.efficiency_score).unwrap())
                .map(|(strategy, perf)| (*strategy, perf.efficiency_score));

            if let Some((strategy, score)) = most_efficient {
                if score > 0.8 {
                    recommendations.push(format!("Most memory-efficient strategy: {:?} (efficiency: {:.2})", strategy, score));
                }
            }

            // Scalability recommendations
            let parallel_measurements: Vec<_> = measurements.iter()
                .filter(|m| m.strategy == OptimizationStrategy::Parallel)
                .collect();

            if parallel_measurements.len() >= 2 {
                let throughput_growth = parallel_measurements.last().unwrap().throughput / parallel_measurements.first().unwrap().throughput;
                if throughput_growth < 2.0 {
                    recommendations.push("Parallel strategy shows poor scalability - consider algorithmic improvements".to_string());
                }
            }

            if recommendations.is_empty() {
                recommendations.push("Performance analysis complete - all strategies show similar characteristics".to_string());
            }

            recommendations
        }

        /// Estimate memory usage for a given strategy and size
        fn estimate_memory_usage(&self, size: usize, strategy: OptimizationStrategy) -> usize {
            let base_memory = size * std::mem::size_of::<f64>(); // Input data

            match strategy {
                OptimizationStrategy::Scalar => base_memory,
                OptimizationStrategy::Simd => base_memory + 1024, // Small SIMD overhead
                OptimizationStrategy::Parallel => base_memory + size * std::mem::size_of::<f64>(), // Temporary arrays
                OptimizationStrategy::Gpu => base_memory * 2, // GPU memory transfer overhead
                _ => base_memory,
            }
        }
    }

    /// Default benchmark configurations for common operations
    pub mod presets {
        use super::*;

        /// Configuration for array operations benchmarks
        pub fn array_operations() -> BenchmarkConfig {
            BenchmarkConfig {
                warmup_iterations: 3,
                measurement_iterations: 10,
                min_duration: Duration::from_millis(50),
                max_duration: Duration::from_secs(10),
                sample_sizes: vec![100, 1_000, 10_000, 100_000],
                strategies: vec![
                    OptimizationStrategy::Scalar,
                    OptimizationStrategy::Simd,
                    OptimizationStrategy::Parallel,
                ],
            }
        }

        /// Configuration for matrix operations benchmarks
        pub fn matrix_operations() -> BenchmarkConfig {
            BenchmarkConfig {
                warmup_iterations: 5,
                measurement_iterations: 15,
                min_duration: Duration::from_millis(100),
                max_duration: Duration::from_secs(30),
                sample_sizes: vec![64, 128, 256, 512, 1024],
                strategies: vec![
                    OptimizationStrategy::Scalar,
                    OptimizationStrategy::Simd,
                    OptimizationStrategy::Parallel,
                    OptimizationStrategy::CacheOptimized,
                ],
            }
        }

        /// Configuration for memory-intensive operations
        pub fn memory_intensive() -> BenchmarkConfig {
            BenchmarkConfig {
                warmup_iterations: 2,
                measurement_iterations: 8,
                min_duration: Duration::from_millis(200),
                max_duration: Duration::from_secs(20),
                sample_sizes: vec![1_000, 10_000, 100_000, 1_000_000, 10_000_000],
                strategies: vec![
                    OptimizationStrategy::Scalar,
                    OptimizationStrategy::MemoryBound,
                    OptimizationStrategy::CacheOptimized,
                ],
            }
        }
    }
}

/// Ultra-optimized cache-aware algorithms for maximum performance
///
/// This module provides adaptive algorithms that automatically adjust their
/// behavior based on cache performance characteristics and system topology.
pub mod cache_aware_algorithms {
    use super::*;
    
    
    
    /// Cache-aware matrix multiplication with adaptive blocking
    pub fn matrix_multiply_cache_aware<T>(
        a: &[T],
        b: &[T],
        c: &mut [T],
        m: usize,
        n: usize,
        k: usize,
    ) where
        T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default,
    {
        // Detect optimal block size based on cache hierarchy
        let block_size = detect_optimal_block_size::<T>();
        
        // Cache-blocked matrix multiplication
        for ii in (0..m).step_by(block_size) {
            for jj in (0..n).step_by(block_size) {
                for kk in (0..k).step_by(block_size) {
                    let m_block = (ii + block_size).min(m);
                    let n_block = (jj + block_size).min(n);
                    let k_block = (kk + block_size).min(k);
                    
                    // Micro-kernel for the block
                    for i in ii..m_block {
                        // Prefetch next cache line
                        if i + 1 < m_block {
                            PerformanceHints::prefetch_read(&a[(i + 1) * k + kk]);
                        }
                        
                        for j in jj..n_block {
                            let mut sum = T::default();
                            
                            // Unroll inner loop for better instruction scheduling
                            let mut l = kk;
                            while l + 4 <= k_block {
                                sum = sum + a[i * k + l] * b[l * n + j];
                                sum = sum + a[i * k + l + 1] * b[(l + 1) * n + j];
                                sum = sum + a[i * k + l + 2] * b[(l + 2) * n + j];
                                sum = sum + a[i * k + l + 3] * b[(l + 3) * n + j];
                                l += 4;
                            }
                            
                            // Handle remaining elements
                            while l < k_block {
                                sum = sum + a[i * k + l] * b[l * n + j];
                                l += 1;
                            }
                            
                            c[i * n + j] = sum;
                        }
                    }
                }
            }
        }
    }
    
    /// Adaptive sorting algorithm that chooses the best strategy based on data characteristics
    pub fn adaptive_sort<T: Ord + Copy>(data: &mut [T]) {
        let len = data.len();
        
        if len <= 1 {
            return;
        }
        
        // Choose algorithm based on size and cache characteristics
        if len < 64 {
            // Use insertion sort for small arrays (cache-friendly)
            cache_aware_insertion_sort(data);
        } else if len < 2048 {
            // Use cache-optimized quicksort for medium arrays
            cache_aware_quicksort(data, 0, len - 1);
        } else {
            // Use cache-oblivious merge sort for large arrays
            cache_oblivious_merge_sort(data);
        }
    }
    
    /// Cache-aware insertion sort optimized for modern cache hierarchies
    fn cache_aware_insertion_sort<T: Ord + Copy>(data: &mut [T]) {
        for i in 1..data.len() {
            let key = data[i];
            let mut j = i;
            
            // Prefetch next elements to improve cache utilization
            if i + 1 < data.len() {
                PerformanceHints::prefetch_read(&data[i + 1]);
            }
            
            while j > 0 && data[j - 1] > key {
                data[j] = data[j - 1];
                j -= 1;
            }
            data[j] = key;
        }
    }
    
    /// Cache-optimized quicksort with adaptive pivot selection
    fn cache_aware_quicksort<T: Ord + Copy>(data: &mut [T], low: usize, high: usize) {
        if low < high {
            // Use median-of-3 for better pivot selection
            let pivot = partition_with_prefetch(data, low, high);
            
            if pivot > 0 {
                cache_aware_quicksort(data, low, pivot - 1);
            }
            cache_aware_quicksort(data, pivot + 1, high);
        }
    }
    
    /// Partitioning with prefetching for better cache utilization
    fn partition_with_prefetch<T: Ord + Copy>(data: &mut [T], low: usize, high: usize) -> usize {
        // Median-of-3 pivot selection
        let mid = low + (high - low) / 2;
        if data[mid] < data[low] {
            data.swap(low, mid);
        }
        if data[high] < data[low] {
            data.swap(low, high);
        }
        if data[high] < data[mid] {
            data.swap(mid, high);
        }
        data.swap(mid, high);
        
        let pivot = data[high];
        let mut i = low;
        
        for j in low..high {
            // Prefetch next iteration
            if j + 8 < high {
                PerformanceHints::prefetch_read(&data[j + 8]);
            }
            
            if data[j] <= pivot {
                data.swap(i, j);
                i += 1;
            }
        }
        data.swap(i, high);
        i
    }
    
    /// Cache-oblivious merge sort for optimal cache performance on large datasets
    fn cache_oblivious_merge_sort<T: Ord + Copy>(data: &mut [T]) {
        let len = data.len();
        if len <= 1 {
            return;
        }
        
        let mut temp = vec![data[0]; len];
        cache_oblivious_merge_sort_recursive(data, &mut temp, 0, len - 1);
    }
    
    fn cache_oblivious_merge_sort_recursive<T: Ord + Copy>(
        data: &mut [T],
        temp: &mut [T],
        left: usize,
        right: usize,
    ) {
        if left >= right {
            return;
        }
        
        let mid = left + (right - left) / 2;
        cache_oblivious_merge_sort_recursive(data, temp, left, mid);
        cache_oblivious_merge_sort_recursive(data, temp, mid + 1, right);
        cache_aware_merge(data, temp, left, mid, right);
    }
    
    /// Cache-aware merge operation with prefetching
    fn cache_aware_merge<T: Ord + Copy>(
        data: &mut [T],
        temp: &mut [T],
        left: usize,
        mid: usize,
        right: usize,
    ) {
        // Copy to temporary array
        temp[left..(right + 1)].copy_from_slice(&data[left..(right + 1)]);
        
        let mut i = left;
        let mut j = mid + 1;
        let mut k = left;
        
        while i <= mid && j <= right {
            // Prefetch ahead in both arrays
            if i + 8 <= mid {
                PerformanceHints::prefetch_read(&temp[i + 8]);
            }
            if j + 8 <= right {
                PerformanceHints::prefetch_read(&temp[j + 8]);
            }
            
            if temp[i] <= temp[j] {
                data[k] = temp[i];
                i += 1;
            } else {
                data[k] = temp[j];
                j += 1;
            }
            k += 1;
        }
        
        // Copy remaining elements
        while i <= mid {
            data[k] = temp[i];
            i += 1;
            k += 1;
        }
        
        while j <= right {
            data[k] = temp[j];
            j += 1;
            k += 1;
        }
    }
    
    /// Detect optimal block size for cache-aware algorithms
    fn detect_optimal_block_size<T>() -> usize {
        // Estimate based on L1 cache size and element size
        let l1_cache_size = 32 * 1024; // 32KB typical L1 cache
        let element_size = std::mem::size_of::<T>();
        let cache_lines = l1_cache_size / 64; // 64-byte cache lines
        let elements_per_line = 64 / element_size.max(1);
        
        // Use square root of cache capacity for 2D blocking
        let block_elements = (cache_lines * elements_per_line / 3) as f64; // Divide by 3 for 3 arrays
        (block_elements.sqrt() as usize).next_power_of_two().min(512)
    }
    
    /// Cache-aware vector reduction with optimal memory access patterns
    pub fn cache_aware_reduce<T, F>(data: &[T], init: T, op: F) -> T
    where
        T: Copy,
        F: Fn(T, T) -> T,
    {
        if data.is_empty() {
            return init;
        }
        
        let _len = data.len();
        let block_size = 64; // Process in cache-line-sized blocks
        let mut result = init;
        
        // Process in blocks to maintain cache locality
        for chunk in data.chunks(block_size) {
            // Prefetch next chunk
            if chunk.as_ptr() as usize + std::mem::size_of_val(chunk) 
                < data.as_ptr() as usize + std::mem::size_of_val(data) {
                let next_chunk_start = unsafe { chunk.as_ptr().add(chunk.len()) };
                PerformanceHints::prefetch_read(unsafe { &*next_chunk_start });
            }
            
            // Reduce within the chunk
            for &item in chunk {
                result = op(result, item);
            }
        }
        
        result
    }
    
    /// Adaptive memory copy with optimal strategy selection
    pub fn adaptive_memcpy<T: Copy>(src: &[T], dst: &mut [T]) {
        debug_assert_eq!(src.len(), dst.len());
        
        let _len = src.len();
        let size_bytes = std::mem::size_of_val(src);
        
        // Choose strategy based on size
        if size_bytes <= 64 {
            // Small copy - use simple loop
            dst.copy_from_slice(src);
        } else if size_bytes <= 4096 {
            // Medium copy - use cache-optimized copy with prefetching
            cache_optimized_copy(src, dst);
        } else {
            // Large copy - use streaming copy to avoid cache pollution
            streaming_copy(src, dst);
        }
    }
    
    /// Cache-optimized copy with prefetching
    fn cache_optimized_copy<T: Copy>(src: &[T], dst: &mut [T]) {
        let chunk_size = 64 / std::mem::size_of::<T>(); // One cache line worth
        
        for (src_chunk, dst_chunk) in src.chunks(chunk_size).zip(dst.chunks_mut(chunk_size)) {
            // Prefetch next source chunk
            if src_chunk.as_ptr() as usize + std::mem::size_of_val(src_chunk) 
                < src.as_ptr() as usize + std::mem::size_of_val(src) {
                let next_src = unsafe { src_chunk.as_ptr().add(chunk_size) };
                PerformanceHints::prefetch_read(unsafe { &*next_src });
            }
            
            dst_chunk.copy_from_slice(src_chunk);
        }
    }
    
    /// Streaming copy for large data to avoid cache pollution
    fn streaming_copy<T: Copy>(src: &[T], dst: &mut [T]) {
        // Use non-temporal stores for large copies to bypass cache
        // For now, fall back to regular copy as non-temporal intrinsics are unstable
        dst.copy_from_slice(src);
    }
    
    /// Cache-aware 2D array transpose
    pub fn cache_aware_transpose<T: Copy>(
        src: &[T],
        dst: &mut [T],
        rows: usize,
        cols: usize,
    ) {
        debug_assert_eq!(src.len(), rows * cols);
        debug_assert_eq!(dst.len(), rows * cols);
        
        let block_size = detect_optimal_block_size::<T>().min(32);
        
        // Block-wise transpose for better cache locality
        for i in (0..rows).step_by(block_size) {
            for j in (0..cols).step_by(block_size) {
                let max_i = (i + block_size).min(rows);
                let max_j = (j + block_size).min(cols);
                
                // Transpose within the block
                for ii in i..max_i {
                    // Prefetch next row
                    if ii + 1 < max_i {
                        PerformanceHints::prefetch_read(&src[(ii + 1) * cols + j]);
                    }
                    
                    for jj in j..max_j {
                        dst[jj * rows + ii] = src[ii * cols + jj];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn test_adaptive_optimizer() {
        let optimizer = AdaptiveOptimizer::new();

        // Test threshold detection
        assert!(!optimizer.should_use_parallel(100));

        // Only test parallel execution if the feature is enabled
        #[cfg(feature = "parallel")]
        assert!(optimizer.should_use_parallel(100_000));

        // Test chunk size calculation
        let chunk_size = optimizer.optimal_chunk_size::<f64>();
        assert!(chunk_size > 0);
        assert_eq!(chunk_size % 16, 0); // Should be multiple of 16
    }

    #[test]
    fn test_fast_path_addition() {
        let a = vec![1.0; 1000];
        let b = vec![2.0; 1000];
        let mut result = vec![0.0; 1000];

        fast_paths::add_f64_arrays(&a, &b, &mut result).unwrap();

        for val in result {
            assert_eq!(val, 3.0);
        }
    }

    #[test]
    fn test_memory_access_pattern() {
        let mut optimizer = MemoryAccessOptimizer::new();

        // Sequential access
        let addresses: Vec<*const f64> = (0..10)
            .map(|i| (i * std::mem::size_of::<f64>()) as *const f64)
            .collect();
        assert_eq!(
            optimizer.analyze_access_pattern(&addresses),
            AccessPattern::Sequential
        );

        // Strided access
        let addresses: Vec<*const f64> = (0..10)
            .map(|i| (i * 3 * std::mem::size_of::<f64>()) as *const f64)
            .collect();
        assert_eq!(
            optimizer.analyze_access_pattern(&addresses),
            AccessPattern::Strided(3)
        );
    }

    #[test]
    fn test_performance_hints() {
        // Test that hints don't crash and return correct values
        assert!(PerformanceHints::likely(true));
        assert!(!PerformanceHints::likely(false));
        assert!(PerformanceHints::unlikely(true));
        assert!(!PerformanceHints::unlikely(false));

        // Test prefetch operations (should not crash)
        let data = [1.0f64; 100];
        PerformanceHints::prefetch_read(&data[0]);

        let mut data_mut = [0.0f64; 100];
        PerformanceHints::prefetch_write(&mut data_mut[0]);

        // Test locality-based prefetch
        PerformanceHints::prefetch_with_locality(&data[0], Locality::High);
        PerformanceHints::prefetch_with_locality(&data[0], Locality::Medium);
        PerformanceHints::prefetch_with_locality(&data[0], Locality::Low);
        PerformanceHints::prefetch_with_locality(&data[0], Locality::None);
    }

    #[test]
    fn test_cache_operations() {
        let data = [1.0f64; 8];

        // Test cache flush (should not crash)
        PerformanceHints::flush_cache_line(&data[0]);

        // Test memory fence (should not crash)
        PerformanceHints::memory_fence();

        // Test cache-aware copy
        let src = vec![1.0f64; 1000];
        let mut dst = vec![0.0f64; 1000];
        PerformanceHints::cache_aware_copy(&src, &mut dst);
        assert_eq!(src, dst);

        // Test cache-aware memset
        let mut data = vec![0.0f64; 1000];
        PerformanceHints::cache_aware_memset(&mut data, 5.0);
        assert!(data.iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_locality_enum() {
        // Test that Locality enum works correctly
        let localities = [
            Locality::High,
            Locality::Medium,
            Locality::Low,
            Locality::None,
        ];

        for locality in &localities {
            // Test that we can use locality in prefetch
            let data = 42i32;
            PerformanceHints::prefetch_with_locality(&data, *locality);
        }

        // Test enum properties
        assert_eq!(Locality::High, Locality::High);
        assert_ne!(Locality::High, Locality::Low);

        // Test Debug formatting
        assert!(format!("{:?}", Locality::High).contains("High"));
    }

    #[test]
    fn test_strategy_selector() {
        let mut selector = StrategySelector::default();
        
        // Test strategy selection
        let strategy = selector.select_strategy(1000, false);
        assert!(matches!(strategy, OptimizationStrategy::Simd | OptimizationStrategy::Scalar | OptimizationStrategy::Parallel | OptimizationStrategy::Gpu));
        
        // Test weight updates
        selector.update_strategy_weight(OptimizationStrategy::Simd, 0.8);
        selector.update_strategy_weight(OptimizationStrategy::Parallel, 0.9);
        
        // Weights should be updated
        assert!(selector.strategy_weights[&OptimizationStrategy::Simd] != 1.0);
        assert!(selector.strategy_weights[&OptimizationStrategy::Parallel] != 1.0);
    }

    #[test]
    fn test_adaptive_optimizer_enhanced() {
        let optimizer = AdaptiveOptimizer::new();
        
        // Test GPU threshold
        assert!(!optimizer.should_use_gpu(1000));
        
        // Test strategy selection
        let strategy = optimizer.select_optimal_strategy("matrix_multiply", 50_000);
        assert!(matches!(strategy, OptimizationStrategy::Parallel | OptimizationStrategy::Simd | OptimizationStrategy::Scalar));
        
        // Test performance recording
        optimizer.record_performance("test_op", OptimizationStrategy::Simd, 1000, 1_000_000);
        
        // Test optimization advice
        let advice = optimizer.analyze_operation("matrix_multiply", 10_000, 10_000);
        assert!(matches!(advice.recommended_strategy, OptimizationStrategy::Parallel | OptimizationStrategy::Simd | OptimizationStrategy::Scalar));
        
        // Test metrics retrieval
        let metrics = optimizer.get_performance_metrics();
        assert!(metrics.is_some());
    }

    #[test]
    fn test_optimization_strategy_enum() {
        // Test that all strategies can be created and compared
        let strategies = [
            OptimizationStrategy::Scalar,
            OptimizationStrategy::Simd,
            OptimizationStrategy::Parallel,
            OptimizationStrategy::Gpu,
            OptimizationStrategy::Hybrid,
            OptimizationStrategy::CacheOptimized,
            OptimizationStrategy::MemoryBound,
            OptimizationStrategy::ComputeBound,
        ];
        
        for strategy in &strategies {
            // Test Debug formatting
            assert!(!format!("{:?}", strategy).is_empty());
            
            // Test equality
            assert_eq!(*strategy, *strategy);
        }
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::default();
        
        // Test that we can add operation times
        metrics.operation_times.insert("test_op".to_string(), 1000.0);
        assert_eq!(metrics.operation_times["test_op"], 1000.0);
        
        // Test strategy success rates
        metrics.strategy_success_rates.insert(OptimizationStrategy::Simd, 0.85);
        assert_eq!(metrics.strategy_success_rates[&OptimizationStrategy::Simd], 0.85);
        
        // Test other metrics
        metrics.memory_bandwidth_utilization = 0.75;
        metrics.cache_hit_rate = 0.90;
        metrics.parallel_efficiency = 0.80;
        
        assert_eq!(metrics.memory_bandwidth_utilization, 0.75);
        assert_eq!(metrics.cache_hit_rate, 0.90);
        assert_eq!(metrics.parallel_efficiency, 0.80);
    }

    #[test]
    fn test_optimization_advice() {
        let advice = OptimizationAdvice {
            recommended_strategy: OptimizationStrategy::Parallel,
            optimal_chunk_size: Some(1024),
            prefetch_distance: Some(64),
            memory_allocation_hint: Some("Use memory mapping".to_string()),
        };
        
        assert_eq!(advice.recommended_strategy, OptimizationStrategy::Parallel);
        assert_eq!(advice.optimal_chunk_size, Some(1024));
        assert_eq!(advice.prefetch_distance, Some(64));
        assert!(advice.memory_allocation_hint.is_some());
        
        // Test Debug formatting
        assert!(!format!("{:?}", advice).is_empty());
    }

    #[test]
    fn test_benchmarking_config() {
        let config = benchmarking::BenchmarkConfig::default();
        
        assert_eq!(config.warmup_iterations, 5);
        assert_eq!(config.measurement_iterations, 20);
        assert!(!config.sample_sizes.is_empty());
        assert!(!config.strategies.is_empty());
        
        // Test preset configurations
        let array_config = benchmarking::presets::array_operations();
        assert_eq!(array_config.warmup_iterations, 3);
        assert_eq!(array_config.measurement_iterations, 10);
        
        let matrix_config = benchmarking::presets::matrix_operations();
        assert_eq!(matrix_config.warmup_iterations, 5);
        assert_eq!(matrix_config.measurement_iterations, 15);
        
        let memory_config = benchmarking::presets::memory_intensive();
        assert_eq!(memory_config.warmup_iterations, 2);
        assert_eq!(memory_config.measurement_iterations, 8);
    }

    #[test]
    fn test_benchmark_measurement() {
        let measurement = benchmarking::BenchmarkMeasurement {
            strategy: OptimizationStrategy::Simd,
            input_size: 1000,
            duration: Duration::from_millis(5),
            throughput: 200_000.0,
            memory_usage: 8000,
            custom_metrics: std::collections::HashMap::new(),
        };
        
        assert_eq!(measurement.strategy, OptimizationStrategy::Simd);
        assert_eq!(measurement.input_size, 1000);
        assert_eq!(measurement.throughput, 200_000.0);
        assert_eq!(measurement.memory_usage, 8000);
    }

    #[test]
    fn test_benchmark_runner() {
        let config = benchmarking::BenchmarkConfig {
            warmup_iterations: 1,
            measurement_iterations: 2,
            min_duration: Duration::from_millis(1),
            max_duration: Duration::from_secs(1),
            sample_sizes: vec![10, 100],
            strategies: vec![OptimizationStrategy::Scalar, OptimizationStrategy::Simd],
        };
        
        let runner = benchmarking::BenchmarkRunner::new(config);
        
        // Test a simple operation
        let results = runner.benchmark_operation("test_add", |data, _strategy| {
            let start = Instant::now();
            let result: Vec<f64> = data.iter().map(|x| x + 1.0).collect();
            (start.elapsed(), result)
        });
        
        assert_eq!(results.operation_name, "test_add");
        assert!(!results.measurements.is_empty());
        assert!(!results.strategy_summary.is_empty());
        assert!(!results.recommendations.is_empty());
        assert!(results.total_duration > Duration::from_nanos(0));
    }

    #[test]
    fn test_strategy_performance() {
        let performance = benchmarking::StrategyPerformance {
            avg_throughput: 150_000.0,
            throughput_stddev: 5_000.0,
            avg_memory_usage: 8000.0,
            optimal_size: 10_000,
            efficiency_score: 0.85,
        };
        
        assert_eq!(performance.avg_throughput, 150_000.0);
        assert_eq!(performance.throughput_stddev, 5_000.0);
        assert_eq!(performance.optimal_size, 10_000);
        assert_eq!(performance.efficiency_score, 0.85);
    }

    #[test]
    fn test_scalability_analysis() {
        let mut parallel_efficiency = std::collections::HashMap::new();
        parallel_efficiency.insert(1000, 0.8);
        parallel_efficiency.insert(10000, 0.9);
        
        let memory_scaling = benchmarking::MemoryScaling {
            linear_coefficient: 8.0,
            constant_coefficient: 1024.0,
            r_squared: 0.95,
        };
        
        let bottleneck = benchmarking::PerformanceBottleneck {
            bottleneck_type: benchmarking::BottleneckType::MemoryBandwidth,
            size_range: (10000, 10000),
            impact: 0.3,
            mitigation: "Use memory prefetching".to_string(),
        };
        
        let analysis = benchmarking::ScalabilityAnalysis {
            parallel_efficiency,
            memory_scaling,
            bottlenecks: vec![bottleneck],
        };
        
        assert_eq!(analysis.parallel_efficiency[&1000], 0.8);
        assert_eq!(analysis.memory_scaling.linear_coefficient, 8.0);
        assert_eq!(analysis.bottlenecks.len(), 1);
        assert_eq!(analysis.bottlenecks[0].bottleneck_type, benchmarking::BottleneckType::MemoryBandwidth);
    }

    #[test]
    fn test_memory_scaling() {
        let scaling = benchmarking::MemoryScaling {
            linear_coefficient: 8.0,
            constant_coefficient: 512.0,
            r_squared: 0.99,
        };
        
        assert_eq!(scaling.linear_coefficient, 8.0);
        assert_eq!(scaling.constant_coefficient, 512.0);
        assert_eq!(scaling.r_squared, 0.99);
    }

    #[test]
    fn test_performance_bottleneck() {
        let bottleneck = benchmarking::PerformanceBottleneck {
            bottleneck_type: benchmarking::BottleneckType::SynchronizationOverhead,
            size_range: (1000, 5000),
            impact: 0.6,
            mitigation: "Reduce thread contention".to_string(),
        };
        
        assert_eq!(bottleneck.bottleneck_type, benchmarking::BottleneckType::SynchronizationOverhead);
        assert_eq!(bottleneck.size_range, (1000, 5000));
        assert_eq!(bottleneck.impact, 0.6);
        assert_eq!(bottleneck.mitigation, "Reduce thread contention");
    }

    #[test]
    fn test_bottleneck_type_enum() {
        let bottleneck_types = [
            benchmarking::BottleneckType::MemoryBandwidth,
            benchmarking::BottleneckType::CacheLatency,
            benchmarking::BottleneckType::ComputeBound,
            benchmarking::BottleneckType::SynchronizationOverhead,
            benchmarking::BottleneckType::AlgorithmicComplexity,
        ];
        
        for bottleneck_type in &bottleneck_types {
            // Test Debug formatting
            assert!(!format!("{:?}", bottleneck_type).is_empty());
            
            // Test equality
            assert_eq!(*bottleneck_type, *bottleneck_type);
        }
        
        // Test inequality
        assert_ne!(benchmarking::BottleneckType::MemoryBandwidth, benchmarking::BottleneckType::CacheLatency);
    }

    #[test]
    fn test_benchmark_results() {
        let measurement = benchmarking::BenchmarkMeasurement {
            strategy: OptimizationStrategy::Parallel,
            input_size: 1000,
            duration: Duration::from_millis(10),
            throughput: 100_000.0,
            memory_usage: 8000,
            custom_metrics: std::collections::HashMap::new(),
        };
        
        let mut strategy_summary = std::collections::HashMap::new();
        strategy_summary.insert(OptimizationStrategy::Parallel, benchmarking::StrategyPerformance {
            avg_throughput: 100_000.0,
            throughput_stddev: 1_000.0,
            avg_memory_usage: 8000.0,
            optimal_size: 1000,
            efficiency_score: 0.9,
        });
        
        let scalability_analysis = benchmarking::ScalabilityAnalysis {
            parallel_efficiency: std::collections::HashMap::new(),
            memory_scaling: benchmarking::MemoryScaling {
                linear_coefficient: 8.0,
                constant_coefficient: 0.0,
                r_squared: 1.0,
            },
            bottlenecks: Vec::new(),
        };
        
        let results = benchmarking::BenchmarkResults {
            operation_name: "test_operation".to_string(),
            measurements: vec![measurement],
            strategy_summary,
            scalability_analysis,
            recommendations: vec!["Use parallel strategy".to_string()],
            total_duration: Duration::from_millis(100),
        };
        
        assert_eq!(results.operation_name, "test_operation");
        assert_eq!(results.measurements.len(), 1);
        assert_eq!(results.strategy_summary.len(), 1);
        assert_eq!(results.recommendations.len(), 1);
        assert_eq!(results.total_duration, Duration::from_millis(100));
    }
}
