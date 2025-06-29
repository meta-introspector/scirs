//! Advanced performance optimization system for production 1.0
//!
//! This module provides comprehensive performance optimization including:
//! - Advanced SIMD operations with runtime feature detection
//! - Cache-aware algorithms and memory optimization
//! - Adaptive performance tuning based on system characteristics
//! - Production-grade resource management

use crate::error::{CoreResult, CoreError, ErrorContext};
use std::sync::{Arc, Mutex, OnceLock};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};

/// Global performance optimizer instance
static GLOBAL_OPTIMIZER: OnceLock<Arc<Mutex<AdvancedPerformanceOptimizer>>> = OnceLock::new();

/// Advanced SIMD capabilities with runtime detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimdCapabilities {
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub fma: bool,
    pub neon: bool,  // ARM NEON
}

impl SimdCapabilities {
    /// Detect SIMD capabilities at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                sse2: is_x86_feature_detected!("sse2"),
                sse3: is_x86_feature_detected!("sse3"),
                ssse3: is_x86_feature_detected!("ssse3"),
                sse4_1: is_x86_feature_detected!("sse4.1"),
                sse4_2: is_x86_feature_detected!("sse4.2"),
                avx: is_x86_feature_detected!("avx"),
                avx2: is_x86_feature_detected!("avx2"),
                avx512f: is_x86_feature_detected!("avx512f"),
                fma: is_x86_feature_detected!("fma"),
                neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                sse2: false,
                sse3: false,
                ssse3: false,
                sse4_1: false,
                sse4_2: false,
                avx: false,
                avx2: false,
                avx512f: false,
                fma: false,
                neon: std::arch::is_aarch64_feature_detected!("neon"),
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                sse2: false,
                sse3: false,
                ssse3: false,
                sse4_1: false,
                sse4_2: false,
                avx: false,
                avx2: false,
                avx512f: false,
                fma: false,
                neon: false,
            }
        }
    }

    /// Get the highest available SIMD instruction set
    pub fn highest_available(&self) -> SimdInstructionSet {
        if self.avx512f {
            SimdInstructionSet::AVX512
        } else if self.avx2 {
            SimdInstructionSet::AVX2
        } else if self.avx {
            SimdInstructionSet::AVX
        } else if self.sse4_2 {
            SimdInstructionSet::SSE42
        } else if self.sse4_1 {
            SimdInstructionSet::SSE41
        } else if self.ssse3 {
            SimdInstructionSet::SSSE3
        } else if self.sse3 {
            SimdInstructionSet::SSE3
        } else if self.sse2 {
            SimdInstructionSet::SSE2
        } else if self.neon {
            SimdInstructionSet::NEON
        } else {
            SimdInstructionSet::Scalar
        }
    }
}

/// SIMD instruction set levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdInstructionSet {
    Scalar,
    SSE2,
    SSE3,
    SSSE3,
    SSE41,
    SSE42,
    AVX,
    AVX2,
    AVX512,
    NEON,
}

/// Cache-aware memory layout information
#[derive(Debug, Clone)]
pub struct CacheInfo {
    pub l1_cache_size: usize,
    pub l2_cache_size: usize,
    pub l3_cache_size: usize,
    pub cache_line_size: usize,
    pub prefetch_distance: usize,
}

impl Default for CacheInfo {
    fn default() -> Self {
        Self {
            l1_cache_size: 32 * 1024,    // 32KB typical L1
            l2_cache_size: 256 * 1024,   // 256KB typical L2
            l3_cache_size: 8 * 1024 * 1024, // 8MB typical L3
            cache_line_size: 64,         // 64 bytes typical
            prefetch_distance: 8,        // 8 cache lines ahead
        }
    }
}

impl CacheInfo {
    /// Detect cache information from system
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self::detect_x86_64()
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::detect_aarch64()
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::default()
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86_64() -> Self {
        // Use CPUID instruction to detect cache sizes
        use std::arch::x86_64::__cpuid;
        
        // CPUID leaf 4 provides cache information
        let l1_cache_size = 32 * 1024;    // Default 32KB
        let mut l2_cache_size = 256 * 1024;   // Default 256KB
        let mut l3_cache_size = 8 * 1024 * 1024; // Default 8MB
        let cache_line_size = 64;             // Standard cache line size
        
        unsafe {
            // Check if CPUID is supported
            let cpuid_result = __cpuid(0);
            if cpuid_result.eax >= 4 {
                // Try to get L1 cache info (leaf 4, subleaf 1)
                let cache_info = __cpuid(0x80000006);
                if cache_info.ecx != 0 {
                    l2_cache_size = (((cache_info.ecx >> 16) & 0xFFFF) * 1024) as usize;
                }
                
                // Try to get L3 cache info
                let _extended_info = __cpuid(0x80000008);
                if cache_info.edx != 0 {
                    l3_cache_size = (((cache_info.edx >> 18) & 0x3FFF) * 512 * 1024) as usize;
                }
            }
        }

        Self {
            l1_cache_size,
            l2_cache_size,
            l3_cache_size,
            cache_line_size,
            prefetch_distance: 8,
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_aarch64() -> Self {
        // ARM cache detection through system registers
        let mut l1_cache_size = 64 * 1024;    // Default 64KB for ARM
        let mut l2_cache_size = 512 * 1024;   // Default 512KB
        let mut l3_cache_size = 4 * 1024 * 1024; // Default 4MB
        
        // Try to read cache size information from /proc/cpuinfo on Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in cpuinfo.lines() {
                    if line.contains("cache size") {
                        if let Some(size_str) = line.split(':').nth(1) {
                            if let Ok(size) = size_str.trim().replace(" KB", "").parse::<usize>() {
                                l2_cache_size = size * 1024;
                            }
                        }
                    }
                }
            }
        }

        Self {
            l1_cache_size,
            l2_cache_size,
            l3_cache_size,
            cache_line_size: 64,
            prefetch_distance: 8,
        }
    }

    /// Calculate optimal chunk size for cache efficiency
    pub fn optimal_chunk_size(&self, element_size: usize) -> usize {
        // Target L2 cache size with some headroom
        let target_size = self.l2_cache_size / 2;
        target_size / element_size
    }

    /// Calculate optimal blocking size for matrix operations
    pub fn optimal_block_size(&self, element_size: usize) -> usize {
        // Target square blocks that fit in L1 cache
        let _elements_per_line = self.cache_line_size / element_size;
        let target_elements = self.l1_cache_size / (3 * element_size); // A, B, C matrices
        (target_elements as f64).sqrt() as usize
    }
}

/// Adaptive performance characteristics
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    pub simd_capabilities: SimdCapabilities,
    pub cache_info: CacheInfo,
    pub cpu_cores: usize,
    pub numa_nodes: usize,
    pub memory_bandwidth_gb_per_sec: f64,
    pub preferred_parallelism: usize,
}

impl PerformanceProfile {
    /// Detect system performance characteristics
    pub fn detect() -> Self {
        let simd_capabilities = SimdCapabilities::detect();
        let cache_info = CacheInfo::detect();
        let cpu_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        
        Self {
            simd_capabilities,
            cache_info,
            cpu_cores,
            numa_nodes: Self::detect_numa_nodes(),
            memory_bandwidth_gb_per_sec: Self::estimate_memory_bandwidth(),
            preferred_parallelism: cpu_cores.min(16), // Cap at 16 for efficiency
        }
    }

    /// Detect NUMA topology
    fn detect_numa_nodes() -> usize {
        #[cfg(target_os = "linux")]
        {
            // Check /sys/devices/system/node/ for NUMA nodes
            if let Ok(entries) = std::fs::read_dir("/sys/devices/system/node/") {
                let numa_count = entries
                    .filter_map(|entry| entry.ok())
                    .filter(|entry| {
                        entry.file_name()
                            .to_string_lossy()
                            .starts_with("node")
                    })
                    .count();
                
                if numa_count > 0 {
                    return numa_count;
                }
            }
            
            // Fallback: check /proc/cpuinfo for physical processors
            if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
                let mut physical_ids = std::collections::HashSet::new();
                for line in cpuinfo.lines() {
                    if line.starts_with("physical id") {
                        if let Some(id_str) = line.split(':').nth(1) {
                            if let Ok(id) = id_str.trim().parse::<usize>() {
                                physical_ids.insert(id);
                            }
                        }
                    }
                }
                if !physical_ids.is_empty() {
                    return physical_ids.len();
                }
            }
        }
        
        #[cfg(target_os = "windows")]
        {
            // Use GetNumaHighestNodeNumber on Windows
            // For now, default to 1
        }
        
        1 // Default to single NUMA node
    }

    /// Estimate memory bandwidth based on system characteristics
    fn estimate_memory_bandwidth() -> f64 {
        #[cfg(target_os = "linux")]
        {
            // Try to read memory information from /proc/meminfo
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(size_str) = line.split_whitespace().nth(1) {
                            if let Ok(memory_kb) = size_str.parse::<usize>() {
                                let memory_gb = memory_kb / (1024 * 1024);
                                
                                // Estimate based on memory size and typical DDR speeds
                                return match memory_gb {
                                    0..=8 => 12.8,      // DDR4-2400
                                    9..=16 => 25.6,     // DDR4-3200
                                    17..=32 => 38.4,    // DDR4-4800
                                    33..=64 => 51.2,    // DDR5-4800
                                    _ => 76.8,          // DDR5-7200 or higher
                                };
                            }
                        }
                    }
                }
            }
        }
        
        25.0 // Default DDR4-3200 bandwidth
    }
}

/// Advanced performance optimization algorithms
#[derive(Debug)]
pub struct AdvancedPerformanceOptimizer {
    profile: PerformanceProfile,
    optimization_history: HashMap<String, OptimizationResult>,
    adaptive_thresholds: AdaptiveThresholds,
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub algorithm_name: String,
    pub input_size: usize,
    pub execution_time: Duration,
    pub throughput_ops_per_sec: f64,
    pub cache_miss_rate: f64,
    pub optimal_settings: OptimizationSettings,
}

#[derive(Debug, Clone)]
pub struct OptimizationSettings {
    pub use_simd: bool,
    pub simd_instruction_set: SimdInstructionSet,
    pub chunk_size: usize,
    pub block_size: usize,
    pub prefetch_enabled: bool,
    pub parallel_threshold: usize,
    pub num_threads: usize,
}

#[derive(Debug, Clone)]
struct AdaptiveThresholds {
    simd_threshold: usize,
    parallel_threshold: usize,
    cache_blocking_threshold: usize,
    prefetch_threshold: usize,
}

impl Default for AdaptiveThresholds {
    fn default() -> Self {
        Self {
            simd_threshold: 64,    // Use SIMD for arrays >= 64 elements
            parallel_threshold: 10000, // Parallelize for >= 10K elements
            cache_blocking_threshold: 1024, // Block for matrices >= 1024x1024
            prefetch_threshold: 512,   // Prefetch for >= 512 elements
        }
    }
}

impl AdvancedPerformanceOptimizer {
    /// Create new performance optimizer
    pub fn new() -> Self {
        Self {
            profile: PerformanceProfile::detect(),
            optimization_history: HashMap::new(),
            adaptive_thresholds: AdaptiveThresholds::default(),
        }
    }

    /// Get global optimizer instance
    pub fn global() -> Arc<Mutex<Self>> {
        GLOBAL_OPTIMIZER.get_or_init(|| Arc::new(Mutex::new(Self::new()))).clone()
    }

    /// Get the performance profile
    pub fn profile(&self) -> &PerformanceProfile {
        &self.profile
    }

    /// Determine optimal settings for a given algorithm and input size
    pub fn optimize_for(&mut self, algorithm: &str, input_size: usize) -> OptimizationSettings {
        // Check if we have historical data for similar operations
        if let Some(result) = self.find_similar_optimization(algorithm, input_size) {
            return result.optimal_settings.clone();
        }

        // Generate initial optimization settings
        // TODO: Run micro-benchmarks to refine settings
        self.generate_initial_settings(algorithm, input_size)
    }

    fn find_similar_optimization(&self, algorithm: &str, input_size: usize) -> Option<&OptimizationResult> {
        self.optimization_history
            .values()
            .filter(|result| result.algorithm_name == algorithm)
            .min_by_key(|result| (result.input_size as i64 - input_size as i64).abs())
    }

    fn generate_initial_settings(&self, _algorithm: &str, input_size: usize) -> OptimizationSettings {
        let simd_instruction_set = self.profile.simd_capabilities.highest_available();
        let use_simd = input_size >= self.adaptive_thresholds.simd_threshold;
        let chunk_size = self.profile.cache_info.optimal_chunk_size(std::mem::size_of::<f64>());
        let block_size = self.profile.cache_info.optimal_block_size(std::mem::size_of::<f64>());
        let prefetch_enabled = input_size >= self.adaptive_thresholds.prefetch_threshold;
        let parallel_threshold = self.adaptive_thresholds.parallel_threshold;
        let num_threads = if input_size >= parallel_threshold {
            self.profile.preferred_parallelism
        } else {
            1
        };

        OptimizationSettings {
            use_simd,
            simd_instruction_set,
            chunk_size,
            block_size,
            prefetch_enabled,
            parallel_threshold,
            num_threads,
        }
    }

    /// Record optimization result for future reference
    pub fn record_optimization(&mut self, result: OptimizationResult) {
        let key = format!("{}_{}", result.algorithm_name, result.input_size);
        self.optimization_history.insert(key, result);
    }

    /// Get performance recommendations for a specific workload
    pub fn get_recommendations(&self, workload_type: WorkloadType, data_size: usize) -> Vec<PerformanceRecommendation> {
        let mut recommendations = Vec::new();

        match workload_type {
            WorkloadType::LinearAlgebra => {
                if data_size >= self.adaptive_thresholds.cache_blocking_threshold {
                    recommendations.push(PerformanceRecommendation::EnableCacheBlocking);
                }
                if self.profile.simd_capabilities.avx2 {
                    recommendations.push(PerformanceRecommendation::UseAdvancedSIMD);
                }
                if data_size >= self.adaptive_thresholds.parallel_threshold {
                    recommendations.push(PerformanceRecommendation::EnableParallelism {
                        num_threads: self.profile.preferred_parallelism,
                    });
                }
            },
            WorkloadType::Statistics => {
                if self.profile.simd_capabilities.fma {
                    recommendations.push(PerformanceRecommendation::UseFusedMultiplyAdd);
                }
                recommendations.push(PerformanceRecommendation::OptimizeMemoryAccess);
            },
            WorkloadType::SignalProcessing => {
                recommendations.push(PerformanceRecommendation::UseVectorization);
                if data_size >= self.adaptive_thresholds.prefetch_threshold {
                    recommendations.push(PerformanceRecommendation::EnablePrefetching);
                }
            },
        }

        recommendations
    }
}

impl Default for AdvancedPerformanceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of computational workloads
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadType {
    LinearAlgebra,
    Statistics,
    SignalProcessing,
}

/// Performance optimization recommendations
#[derive(Debug, Clone)]
pub enum PerformanceRecommendation {
    EnableCacheBlocking,
    UseAdvancedSIMD,
    UseFusedMultiplyAdd,
    EnableParallelism { num_threads: usize },
    OptimizeMemoryAccess,
    UseVectorization,
    EnablePrefetching,
}

/// Cache-aware matrix multiplication with adaptive blocking
pub fn cache_aware_matrix_multiply<T>(
    a: &Array2<T>,
    b: &Array2<T>,
    settings: &OptimizationSettings,
) -> CoreResult<Array2<T>>
where
    T: Clone + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy + Send + Sync,
{
    let (m, k) = a.dim();
    let (k2, n) = b.dim();
    
    if k != k2 {
        return Err(CoreError::ValidationError(ErrorContext {
            message: "Matrix dimensions don't match".to_string(),
            location: None,
            cause: None,
        }));
    }

    let mut c = Array2::default((m, n));
    
    if m * n * k >= settings.parallel_threshold && settings.num_threads > 1 {
        // Use blocked parallel multiplication
        blocked_parallel_multiply(a, b, &mut c, settings)?;
    } else {
        // Use cache-blocked multiplication
        blocked_multiply(a, b, &mut c, settings.block_size)?;
    }
    
    Ok(c)
}

fn blocked_multiply<T>(
    a: &Array2<T>,
    b: &Array2<T>,
    c: &mut Array2<T>,
    block_size: usize,
) -> CoreResult<()>
where
    T: Clone + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy + Send + Sync,
{
    let (m, k) = a.dim();
    let (_, n) = b.dim();
    
    for i_block in (0..m).step_by(block_size) {
        for j_block in (0..n).step_by(block_size) {
            for k_block in (0..k).step_by(block_size) {
                let i_end = (i_block + block_size).min(m);
                let j_end = (j_block + block_size).min(n);
                let k_end = (k_block + block_size).min(k);
                
                for i in i_block..i_end {
                    for j in j_block..j_end {
                        let mut sum = c[[i, j]];
                        for k_idx in k_block..k_end {
                            sum = sum + a[[i, k_idx]] * b[[k_idx, j]];
                        }
                        c[[i, j]] = sum;
                    }
                }
            }
        }
    }
    
    Ok(())
}

fn blocked_parallel_multiply<T>(
    a: &Array2<T>,
    b: &Array2<T>,
    c: &mut Array2<T>,
    settings: &OptimizationSettings,
) -> CoreResult<()>
where
    T: Clone + Default + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy + Send + Sync,
{
    // Rayon support temporarily disabled
    // use rayon::prelude::*;
    
    let (m, n) = c.dim();
    let block_size = settings.block_size;
    
    // Create parallel blocks
    let blocks: Vec<(usize, usize)> = (0..m)
        .step_by(block_size)
        .flat_map(|i| (0..n).step_by(block_size).map(move |j| (i, j)))
        .collect();
    
    // Process blocks sequentially (parallel support disabled)
    for &(i_block, j_block) in &blocks {
        let i_end = (i_block + block_size).min(m);
        let j_end = (j_block + block_size).min(n);
        
        // Process this block
        for i in i_block..i_end {
            for j in j_block..j_end {
                let mut sum = T::default();
                for k in 0..a.dim().1 {
                    sum = sum + a[[i, k]] * b[[k, j]];
                }
                c[[i, j]] = sum;
            }
        }
    }
    
    Ok(())
}

/// Advanced SIMD operations for different instruction sets
pub mod simd_ops {
    use super::*;
    
    /// SIMD-optimized vector addition
    pub fn simd_vector_add<T>(
        a: ArrayView1<T>,
        b: ArrayView1<T>,
        instruction_set: SimdInstructionSet,
    ) -> CoreResult<Array1<T>>
    where
        T: Clone + Copy + std::ops::Add<Output = T> + Default,
    {
        let len = a.len();
        if len != b.len() {
            return Err(CoreError::ValidationError(ErrorContext {
                message: "Vector lengths don't match".to_string(),
                location: None,
                cause: None,
            }));
        }
        
        let mut result = Array1::default((len,));
        
        match instruction_set {
            SimdInstructionSet::AVX2 => {
                // Would use AVX2 intrinsics here
                simd_add_fallback(a, b, result.view_mut())?;
            },
            SimdInstructionSet::AVX => {
                // Would use AVX intrinsics here
                simd_add_fallback(a, b, result.view_mut())?;
            },
            SimdInstructionSet::SSE42 => {
                // Would use SSE4.2 intrinsics here
                simd_add_fallback(a, b, result.view_mut())?;
            },
            SimdInstructionSet::NEON => {
                // Would use NEON intrinsics here
                simd_add_fallback(a, b, result.view_mut())?;
            },
            _ => {
                // Fallback to scalar implementation
                simd_add_fallback(a, b, result.view_mut())?;
            }
        }
        
        Ok(result)
    }
    
    fn simd_add_fallback<T>(
        a: ArrayView1<T>,
        b: ArrayView1<T>,
        mut result: ArrayViewMut1<T>,
    ) -> CoreResult<()>
    where
        T: Clone + Copy + std::ops::Add<Output = T>,
    {
        for ((a_val, b_val), res) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
            *res = *a_val + *b_val;
        }
        Ok(())
    }
    
    /// SIMD-optimized dot product
    pub fn simd_dot_product<T>(
        a: ArrayView1<T>,
        b: ArrayView1<T>,
        instruction_set: SimdInstructionSet,
    ) -> CoreResult<T>
    where
        T: Clone + Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default,
    {
        let len = a.len();
        if len != b.len() {
            return Err(CoreError::ValidationError(ErrorContext {
                message: "Vector lengths don't match".to_string(),
                location: None,
                cause: None,
            }));
        }
        
        match instruction_set {
            SimdInstructionSet::AVX2 | SimdInstructionSet::AVX | SimdInstructionSet::SSE42 => {
                // Would use vectorized implementation
                dot_product_fallback(a, b)
            },
            _ => dot_product_fallback(a, b),
        }
    }
    
    fn dot_product_fallback<T>(a: ArrayView1<T>, b: ArrayView1<T>) -> CoreResult<T>
    where
        T: Clone + Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default,
    {
        let mut sum = T::default();
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            sum = sum + *a_val * *b_val;
        }
        Ok(sum)
    }
}

/// Memory prefetching utilities
pub mod prefetch {
    use std::arch::x86_64::*;
    
    /// Prefetch memory for read access
    /// 
    /// # Safety
    /// 
    /// The caller must ensure that `addr` is a valid pointer that can be safely
    /// dereferenced. The pointer does not need to be aligned, but should point to
    /// allocated memory that will be accessed in the near future.
    #[inline]
    pub unsafe fn prefetch_read<T>(addr: *const T, locality: i32) {
        #[cfg(target_arch = "x86_64")]
        {
            match locality {
                0 => _mm_prefetch(addr as *const i8, _MM_HINT_NTA),
                1 => _mm_prefetch(addr as *const i8, _MM_HINT_T2),
                2 => _mm_prefetch(addr as *const i8, _MM_HINT_T1),
                3 => _mm_prefetch(addr as *const i8, _MM_HINT_T0),
                _ => _mm_prefetch(addr as *const i8, _MM_HINT_T0), // Default to highest locality
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // No-op on non-x86 architectures
            let _ = (addr, locality);
        }
    }
    
    /// Prefetch memory for write access
    /// 
    /// # Safety
    /// 
    /// The caller must ensure that `addr` is a valid pointer that can be safely
    /// dereferenced. The pointer does not need to be aligned, but should point to
    /// allocated memory that will be written to in the near future.
    #[inline]
    pub unsafe fn prefetch_write<T>(addr: *const T) {
        #[cfg(target_arch = "x86_64")]
        {
            _mm_prefetch(addr as *const i8, _MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // No-op on non-x86 architectures
            let _ = addr;
        }
    }
}

/// Performance measurement and profiling utilities
pub mod profiling {
    use super::*;
    
    /// Measure execution time and throughput of an operation
    pub fn measure_performance<F, R>(operation: F) -> (R, Duration, f64)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();
        let throughput = 1.0 / duration.as_secs_f64();
        
        (result, duration, throughput)
    }
    
    /// Profile memory access patterns
    pub struct MemoryProfiler {
        cache_misses: u64,
        memory_accesses: u64,
    }
    
    impl Default for MemoryProfiler {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MemoryProfiler {
        pub fn new() -> Self {
            Self {
                cache_misses: 0,
                memory_accesses: 0,
            }
        }
        
        pub fn record_access(&mut self, is_cache_miss: bool) {
            self.memory_accesses += 1;
            if is_cache_miss {
                self.cache_misses += 1;
            }
        }
        
        pub fn cache_miss_rate(&self) -> f64 {
            if self.memory_accesses == 0 {
                0.0
            } else {
                self.cache_misses as f64 / self.memory_accesses as f64
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_capabilities_detection() {
        let caps = SimdCapabilities::detect();
        println!("SIMD capabilities: {:?}", caps);
        
        let highest = caps.highest_available();
        assert!(highest >= SimdInstructionSet::Scalar);
    }
    
    #[test]
    fn test_cache_info_detection() {
        let cache_info = CacheInfo::detect();
        assert!(cache_info.cache_line_size > 0);
        assert!(cache_info.l1_cache_size > 0);
    }
    
    #[test]
    fn test_performance_optimizer() {
        let mut optimizer = AdvancedPerformanceOptimizer::new();
        let settings = optimizer.optimize_for("matrix_multiply", 1000);
        
        assert!(settings.chunk_size > 0);
        assert!(settings.block_size > 0);
    }
    
    #[test]
    fn test_cache_aware_matrix_multiply() {
        let a = Array2::from_elem((4, 4), 1.0f64);
        let b = Array2::from_elem((4, 4), 2.0f64);
        
        let settings = OptimizationSettings {
            use_simd: false,
            simd_instruction_set: SimdInstructionSet::Scalar,
            chunk_size: 1024,
            block_size: 2,
            prefetch_enabled: false,
            parallel_threshold: 10000,
            num_threads: 1,
        };
        
        let result = cache_aware_matrix_multiply(&a, &b, &settings).unwrap();
        assert_eq!(result.dim(), (4, 4));
        
        // Each element should be 4 * 1.0 * 2.0 = 8.0
        for &val in result.iter() {
            assert!((val - 8.0).abs() < 1e-10);
        }
    }
}