//! GPU acceleration interfaces and stubs for clustering algorithms
//!
//! This module provides GPU acceleration capabilities for clustering algorithms.
//! Currently implements stubs and interfaces that can be extended with actual
//! GPU implementations using CUDA, OpenCL, or other GPU computing frameworks.

use crate::error::{ClusteringError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// GPU acceleration backends
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GpuBackend {
    /// NVIDIA CUDA backend
    Cuda,
    /// OpenCL backend (cross-platform)
    OpenCl,
    /// AMD ROCm backend
    Rocm,
    /// Intel OneAPI backend
    OneApi,
    /// Apple Metal Performance Shaders
    Metal,
    /// CPU fallback (no GPU acceleration)
    CpuFallback,
}

/// GPU device information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuDevice {
    /// Device ID
    pub device_id: u32,
    /// Device name
    pub name: String,
    /// Total memory in bytes
    pub total_memory: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Compute capability or equivalent
    pub compute_capability: String,
    /// Number of compute units
    pub compute_units: u32,
    /// Backend type
    pub backend: GpuBackend,
    /// Whether device supports double precision
    pub supports_double_precision: bool,
}

/// GPU memory allocation strategy
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MemoryStrategy {
    /// Use unified memory (CUDA/HIP)
    Unified,
    /// Explicit host-device transfers
    Explicit,
    /// Memory pooling for reuse
    Pooled { pool_size_mb: usize },
    /// Zero-copy memory (if supported)
    ZeroCopy,
}

/// GPU acceleration configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuConfig {
    /// Preferred backend
    pub backend: GpuBackend,
    /// Device selection strategy
    pub device_selection: DeviceSelection,
    /// Memory allocation strategy
    pub memory_strategy: MemoryStrategy,
    /// Block size for GPU kernels
    pub block_size: u32,
    /// Grid size for GPU kernels
    pub grid_size: u32,
    /// Enable automatic tuning
    pub auto_tune: bool,
    /// Fallback to CPU if GPU fails
    pub cpu_fallback: bool,
}

/// Device selection strategy
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DeviceSelection {
    /// Use device with most memory
    MostMemory,
    /// Use device with highest compute capability
    HighestCompute,
    /// Use specific device by ID
    Specific(u32),
    /// Automatically select best device
    Automatic,
    /// Use multiple devices (multi-GPU)
    MultiGpu(Vec<u32>),
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::CpuFallback,
            device_selection: DeviceSelection::Automatic,
            memory_strategy: MemoryStrategy::Explicit,
            block_size: 256,
            grid_size: 1024,
            auto_tune: true,
            cpu_fallback: true,
        }
    }
}

/// GPU context for clustering operations
#[derive(Debug)]
pub struct GpuContext {
    /// Active devices
    devices: Vec<GpuDevice>,
    /// Current configuration
    config: GpuConfig,
    /// Backend-specific context
    backend_context: BackendContext,
    /// Performance statistics
    stats: GpuStats,
}

/// Backend-specific context (placeholder for actual implementations)
#[derive(Debug)]
enum BackendContext {
    /// CUDA context
    Cuda {
        #[allow(dead_code)]
        context_handle: usize,
    },
    /// OpenCL context
    OpenCl {
        #[allow(dead_code)]
        context_handle: usize,
    },
    /// CPU fallback (no context needed)
    CpuFallback,
}

/// GPU performance statistics
#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuStats {
    /// Total GPU memory allocations
    pub total_allocations: usize,
    /// Current memory usage
    pub current_memory_usage: usize,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Number of kernel launches
    pub kernel_launches: usize,
    /// Total GPU computation time (seconds)
    pub gpu_compute_time: f64,
    /// Total memory transfer time (seconds)
    pub memory_transfer_time: f64,
    /// Host-to-device transfers
    pub h2d_transfers: usize,
    /// Device-to-host transfers
    pub d2h_transfers: usize,
}

impl GpuContext {
    /// Initialize GPU context with configuration
    pub fn new(config: GpuConfig) -> Result<Self> {
        let devices = Self::detect_devices(&config.backend)?;
        
        if devices.is_empty() && !config.cpu_fallback {
            return Err(ClusteringError::ComputationError(
                "No GPU devices found and CPU fallback disabled".to_string(),
            ));
        }
        
        let backend_context = Self::initialize_backend(&config.backend)?;
        
        Ok(Self {
            devices,
            config,
            backend_context,
            stats: GpuStats::default(),
        })
    }
    
    /// Detect available GPU devices
    fn detect_devices(backend: &GpuBackend) -> Result<Vec<GpuDevice>> {
        match backend {
            GpuBackend::Cuda => Self::detect_cuda_devices(),
            GpuBackend::OpenCl => Self::detect_opencl_devices(),
            GpuBackend::Rocm => Self::detect_rocm_devices(),
            GpuBackend::OneApi => Self::detect_oneapi_devices(),
            GpuBackend::Metal => Self::detect_metal_devices(),
            GpuBackend::CpuFallback => Ok(vec![]),
        }
    }
    
    /// Detect CUDA devices (stub implementation)
    fn detect_cuda_devices() -> Result<Vec<GpuDevice>> {
        // Stub implementation - would use actual CUDA device detection
        #[cfg(feature = "cuda")]
        {
            // Actual CUDA device detection would go here
            Ok(vec![])
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(vec![])
        }
    }
    
    /// Detect OpenCL devices (stub implementation)
    fn detect_opencl_devices() -> Result<Vec<GpuDevice>> {
        // Stub implementation - would use actual OpenCL device detection
        #[cfg(feature = "opencl")]
        {
            // Actual OpenCL device detection would go here
            Ok(vec![])
        }
        #[cfg(not(feature = "opencl"))]
        {
            Ok(vec![])
        }
    }
    
    /// Detect ROCm devices (stub implementation)
    fn detect_rocm_devices() -> Result<Vec<GpuDevice>> {
        // Stub implementation - would use actual ROCm device detection
        Ok(vec![])
    }
    
    /// Detect OneAPI devices (stub implementation)
    fn detect_oneapi_devices() -> Result<Vec<GpuDevice>> {
        // Stub implementation - would use actual OneAPI device detection
        Ok(vec![])
    }
    
    /// Detect Metal devices (stub implementation)
    fn detect_metal_devices() -> Result<Vec<GpuDevice>> {
        // Stub implementation - would use actual Metal device detection
        #[cfg(target_os = "macos")]
        {
            // Actual Metal device detection would go here
            Ok(vec![])
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(vec![])
        }
    }
    
    /// Initialize backend context
    fn initialize_backend(backend: &GpuBackend) -> Result<BackendContext> {
        match backend {
            GpuBackend::Cuda => Ok(BackendContext::Cuda { context_handle: 0 }),
            GpuBackend::OpenCl => Ok(BackendContext::OpenCl { context_handle: 0 }),
            _ => Ok(BackendContext::CpuFallback),
        }
    }
    
    /// Get available devices
    pub fn get_devices(&self) -> &[GpuDevice] {
        &self.devices
    }
    
    /// Get current configuration
    pub fn get_config(&self) -> &GpuConfig {
        &self.config
    }
    
    /// Get performance statistics
    pub fn get_stats(&self) -> &GpuStats {
        &self.stats
    }
    
    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        !self.devices.is_empty()
    }
}

/// GPU-accelerated K-means clustering (stub implementation)
pub struct GpuKMeans<F: Float> {
    /// GPU context
    context: GpuContext,
    /// Current cluster centers on GPU
    gpu_centers: Option<GpuArray<F>>,
    /// Configuration
    config: GpuKMeansConfig,
}

/// Configuration for GPU K-means
#[derive(Debug, Clone)]
pub struct GpuKMeansConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Batch size for processing
    pub batch_size: usize,
    /// Use shared memory optimization
    pub use_shared_memory: bool,
}

impl Default for GpuKMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            max_iterations: 300,
            tolerance: 1e-4,
            batch_size: 1024,
            use_shared_memory: true,
        }
    }
}

/// GPU array abstraction
#[derive(Debug)]
pub struct GpuArray<F: Float> {
    /// Device pointer (platform-specific)
    device_ptr: usize,
    /// Array dimensions
    shape: Vec<usize>,
    /// Element count
    size: usize,
    /// Data type marker
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + FromPrimitive> GpuArray<F> {
    /// Allocate GPU memory for array
    pub fn allocate(shape: &[usize]) -> Result<Self> {
        let size = shape.iter().product();
        
        // Stub implementation - would allocate actual GPU memory
        Ok(Self {
            device_ptr: 0,
            shape: shape.to_vec(),
            size,
            _phantom: std::marker::PhantomData,
        })
    }
    
    /// Copy data from host to device
    pub fn copy_from_host(&mut self, _host_data: ArrayView2<F>) -> Result<()> {
        // Stub implementation - would perform actual host-to-device transfer
        Ok(())
    }
    
    /// Copy data from device to host
    pub fn copy_to_host(&self, _host_data: &mut Array2<F>) -> Result<()> {
        // Stub implementation - would perform actual device-to-host transfer
        Ok(())
    }
    
    /// Get array shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get element count
    pub fn size(&self) -> usize {
        self.size
    }
}

impl<F: Float + FromPrimitive> Drop for GpuArray<F> {
    fn drop(&mut self) {
        // Stub implementation - would free actual GPU memory
    }
}

impl<F: Float + FromPrimitive> GpuKMeans<F> {
    /// Create new GPU K-means instance
    pub fn new(gpu_config: GpuConfig, kmeans_config: GpuKMeansConfig) -> Result<Self> {
        let context = GpuContext::new(gpu_config)?;
        
        Ok(Self {
            context,
            gpu_centers: None,
            config: kmeans_config,
        })
    }
    
    /// Initialize cluster centers on GPU
    pub fn initialize_centers(&mut self, initial_centers: ArrayView2<F>) -> Result<()> {
        let shape = initial_centers.shape();
        let mut gpu_centers = GpuArray::allocate(shape)?;
        gpu_centers.copy_from_host(initial_centers)?;
        self.gpu_centers = Some(gpu_centers);
        Ok(())
    }
    
    /// Perform K-means clustering on GPU
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<(Array2<F>, Array1<usize>)> {
        if self.gpu_centers.is_none() {
            return Err(ClusteringError::InvalidInput(
                "Centers not initialized".to_string(),
            ));
        }
        
        if !self.context.is_gpu_available() {
            // Fallback to CPU implementation
            return self.fit_cpu_fallback(data);
        }
        
        // Stub implementation - would perform actual GPU clustering
        let n_samples = data.nrows();
        let centers = Array2::zeros((self.config.n_clusters, data.ncols()));
        let labels = Array1::zeros(n_samples);
        
        Ok((centers, labels))
    }
    
    /// CPU fallback implementation
    fn fit_cpu_fallback(&self, data: ArrayView2<F>) -> Result<(Array2<F>, Array1<usize>)> {
        // Use CPU-based K-means as fallback
        use crate::vq::kmeans;
        
        let (centers, labels) = kmeans(
            data,
            self.config.n_clusters,
            None,
            Some(self.config.max_iterations),
            Some(self.config.tolerance),
            None,
        )?;
        
        Ok((centers, labels))
    }
    
    /// Get current cluster centers
    pub fn get_centers(&self) -> Result<Array2<F>> {
        if let Some(ref gpu_centers) = self.gpu_centers {
            let mut host_centers = Array2::zeros(gpu_centers.shape());
            gpu_centers.copy_to_host(&mut host_centers)?;
            Ok(host_centers)
        } else {
            Err(ClusteringError::InvalidInput(
                "Centers not available".to_string(),
            ))
        }
    }
    
    /// Predict cluster assignments
    pub fn predict(&self, data: ArrayView2<F>) -> Result<Array1<usize>> {
        // Stub implementation - would use GPU for prediction
        let n_samples = data.nrows();
        Ok(Array1::zeros(n_samples))
    }
}

/// GPU-accelerated distance matrix computation
pub struct GpuDistanceMatrix<F: Float> {
    /// GPU context
    context: GpuContext,
    /// Distance metric
    metric: DistanceMetric,
}

/// Supported distance metrics for GPU computation
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Cosine distance
    Cosine,
    /// Minkowski distance with parameter p
    Minkowski(f64),
}

impl<F: Float + FromPrimitive> GpuDistanceMatrix<F> {
    /// Create new GPU distance matrix computer
    pub fn new(gpu_config: GpuConfig, metric: DistanceMetric) -> Result<Self> {
        let context = GpuContext::new(gpu_config)?;
        
        Ok(Self { context, metric })
    }
    
    /// Compute pairwise distances on GPU
    pub fn compute_distances(&self, data: ArrayView2<F>) -> Result<Array2<F>> {
        if !self.context.is_gpu_available() {
            // Fallback to CPU implementation
            return self.compute_distances_cpu(data);
        }
        
        // Stub implementation - would perform actual GPU distance computation
        let n_samples = data.nrows();
        Ok(Array2::zeros((n_samples, n_samples)))
    }
    
    /// CPU fallback for distance computation
    fn compute_distances_cpu(&self, data: ArrayView2<F>) -> Result<Array2<F>> {
        let n_samples = data.nrows();
        let mut distances = Array2::zeros((n_samples, n_samples));
        
        for i in 0..n_samples {
            for j in i..n_samples {
                let dist = match self.metric {
                    DistanceMetric::Euclidean => {
                        crate::vq::euclidean_distance(data.row(i), data.row(j))
                    }
                    DistanceMetric::Manhattan => {
                        data.row(i)
                            .iter()
                            .zip(data.row(j).iter())
                            .map(|(a, b)| (*a - *b).abs())
                            .sum()
                    }
                    DistanceMetric::Cosine => {
                        let dot_product = data.row(i)
                            .iter()
                            .zip(data.row(j).iter())
                            .map(|(a, b)| *a * *b)
                            .sum::<F>();
                        
                        let norm_i = data.row(i).iter().map(|x| *x * *x).sum::<F>().sqrt();
                        let norm_j = data.row(j).iter().map(|x| *x * *x).sum::<F>().sqrt();
                        
                        F::one() - dot_product / (norm_i * norm_j)
                    }
                    DistanceMetric::Minkowski(p) => {
                        let sum = data.row(i)
                            .iter()
                            .zip(data.row(j).iter())
                            .map(|(a, b)| (*a - *b).abs().to_f64().unwrap().powf(p))
                            .sum::<f64>();
                        F::from(sum.powf(1.0 / p)).unwrap()
                    }
                };
                
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }
        
        Ok(distances)
    }
}

/// GPU memory manager for efficient allocation and deallocation
#[derive(Debug)]
pub struct GpuMemoryManager {
    /// Memory pools by size
    pools: HashMap<usize, Vec<usize>>,
    /// Total allocated memory
    total_allocated: usize,
    /// Peak memory usage
    peak_usage: usize,
    /// Configuration
    config: MemoryManagerConfig,
}

/// Configuration for GPU memory manager
#[derive(Debug, Clone)]
pub struct MemoryManagerConfig {
    /// Enable memory pooling
    pub enable_pooling: bool,
    /// Pool sizes to maintain
    pub pool_sizes: Vec<usize>,
    /// Maximum memory usage (bytes)
    pub max_memory: usize,
    /// Automatic garbage collection threshold
    pub gc_threshold: f64,
}

impl Default for MemoryManagerConfig {
    fn default() -> Self {
        Self {
            enable_pooling: true,
            pool_sizes: vec![
                1024 * 1024,      // 1MB
                16 * 1024 * 1024, // 16MB
                64 * 1024 * 1024, // 64MB
                256 * 1024 * 1024, // 256MB
            ],
            max_memory: 1024 * 1024 * 1024, // 1GB
            gc_threshold: 0.8,
        }
    }
}

impl GpuMemoryManager {
    /// Create new GPU memory manager
    pub fn new(config: MemoryManagerConfig) -> Self {
        let mut pools = HashMap::new();
        
        if config.enable_pooling {
            for &size in &config.pool_sizes {
                pools.insert(size, Vec::new());
            }
        }
        
        Self {
            pools,
            total_allocated: 0,
            peak_usage: 0,
            config,
        }
    }
    
    /// Allocate GPU memory
    pub fn allocate(&mut self, size: usize) -> Result<usize> {
        // Check memory limit
        if self.total_allocated + size > self.config.max_memory {
            if self.should_gc() {
                self.garbage_collect()?;
            }
            
            if self.total_allocated + size > self.config.max_memory {
                return Err(ClusteringError::ComputationError(
                    "GPU memory limit exceeded".to_string(),
                ));
            }
        }
        
        // Try to reuse from pool
        if self.config.enable_pooling {
            if let Some(pool) = self.pools.get_mut(&size) {
                if let Some(ptr) = pool.pop() {
                    return Ok(ptr);
                }
            }
        }
        
        // Allocate new memory (stub implementation)
        let ptr = self.total_allocated + 1; // Fake pointer
        self.total_allocated += size;
        self.peak_usage = self.peak_usage.max(self.total_allocated);
        
        Ok(ptr)
    }
    
    /// Deallocate GPU memory
    pub fn deallocate(&mut self, ptr: usize, size: usize) -> Result<()> {
        if self.config.enable_pooling {
            if let Some(pool) = self.pools.get_mut(&size) {
                pool.push(ptr);
                return Ok(());
            }
        }
        
        // Actually free memory (stub implementation)
        self.total_allocated = self.total_allocated.saturating_sub(size);
        Ok(())
    }
    
    /// Check if garbage collection should be triggered
    fn should_gc(&self) -> bool {
        let usage_ratio = self.total_allocated as f64 / self.config.max_memory as f64;
        usage_ratio > self.config.gc_threshold
    }
    
    /// Perform garbage collection
    fn garbage_collect(&mut self) -> Result<()> {
        // Clear memory pools to free unused allocations
        for pool in self.pools.values_mut() {
            pool.clear();
        }
        
        Ok(())
    }
    
    /// Get memory usage statistics
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            total_allocated: self.total_allocated,
            peak_usage: self.peak_usage,
            pool_count: self.pools.values().map(|p| p.len()).sum(),
            fragmentation_ratio: self.calculate_fragmentation(),
        }
    }
    
    /// Calculate memory fragmentation ratio
    fn calculate_fragmentation(&self) -> f64 {
        let pooled_memory: usize = self
            .pools
            .iter()
            .map(|(&size, pool)| size * pool.len())
            .sum();
        
        if self.total_allocated == 0 {
            0.0
        } else {
            pooled_memory as f64 / self.total_allocated as f64
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MemoryStats {
    /// Total allocated memory (bytes)
    pub total_allocated: usize,
    /// Peak memory usage (bytes)
    pub peak_usage: usize,
    /// Number of objects in memory pools
    pub pool_count: usize,
    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,
}

/// GPU benchmark utilities
pub mod benchmark {
    use super::*;
    use std::time::Instant;
    
    /// Benchmark result for GPU operations
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct BenchmarkResult {
        /// Operation name
        pub operation: String,
        /// GPU execution time (seconds)
        pub gpu_time: f64,
        /// CPU execution time (seconds)
        pub cpu_time: f64,
        /// Speedup factor (cpu_time / gpu_time)
        pub speedup: f64,
        /// Memory usage (bytes)
        pub memory_usage: usize,
        /// Data size processed
        pub data_size: usize,
        /// Throughput (operations per second)
        pub throughput: f64,
    }
    
    /// Benchmark GPU vs CPU performance
    pub fn benchmark_kmeans<F: Float + FromPrimitive>(
        data: ArrayView2<F>,
        n_clusters: usize,
    ) -> Result<BenchmarkResult> {
        let data_size = data.len() * std::mem::size_of::<F>();
        
        // Benchmark GPU implementation
        let gpu_start = Instant::now();
        let gpu_config = GpuConfig::default();
        let kmeans_config = GpuKMeansConfig {
            n_clusters,
            ..Default::default()
        };
        
        let mut gpu_kmeans = GpuKMeans::new(gpu_config, kmeans_config)?;
        let initial_centers = Array2::zeros((n_clusters, data.ncols()));
        gpu_kmeans.initialize_centers(initial_centers.view())?;
        let _gpu_result = gpu_kmeans.fit(data)?;
        let gpu_time = gpu_start.elapsed().as_secs_f64();
        
        // Benchmark CPU implementation
        let cpu_start = Instant::now();
        let _cpu_result = crate::vq::kmeans(data, n_clusters, None, None, None, None)?;
        let cpu_time = cpu_start.elapsed().as_secs_f64();
        
        let speedup = if gpu_time > 0.0 { cpu_time / gpu_time } else { 0.0 };
        let throughput = if gpu_time > 0.0 { data.nrows() as f64 / gpu_time } else { 0.0 };
        
        Ok(BenchmarkResult {
            operation: "kmeans".to_string(),
            gpu_time,
            cpu_time,
            speedup,
            memory_usage: data_size * 2, // Estimate for data + centers
            data_size,
            throughput,
        })
    }
    
    /// Benchmark distance matrix computation
    pub fn benchmark_distance_matrix<F: Float + FromPrimitive>(
        data: ArrayView2<F>,
    ) -> Result<BenchmarkResult> {
        let data_size = data.len() * std::mem::size_of::<F>();
        
        // Benchmark GPU implementation
        let gpu_start = Instant::now();
        let gpu_config = GpuConfig::default();
        let gpu_distances = GpuDistanceMatrix::new(gpu_config, DistanceMetric::Euclidean)?;
        let _gpu_result = gpu_distances.compute_distances(data)?;
        let gpu_time = gpu_start.elapsed().as_secs_f64();
        
        // Benchmark CPU implementation
        let cpu_start = Instant::now();
        let _cpu_result = gpu_distances.compute_distances_cpu(data)?;
        let cpu_time = cpu_start.elapsed().as_secs_f64();
        
        let speedup = if gpu_time > 0.0 { cpu_time / gpu_time } else { 0.0 };
        let throughput = if gpu_time > 0.0 { 
            (data.nrows() * data.nrows()) as f64 / gpu_time 
        } else { 
            0.0 
        };
        
        Ok(BenchmarkResult {
            operation: "distance_matrix".to_string(),
            gpu_time,
            cpu_time,
            speedup,
            memory_usage: data_size + data.nrows() * data.nrows() * std::mem::size_of::<F>(),
            data_size,
            throughput,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.backend, GpuBackend::CpuFallback);
        assert!(config.cpu_fallback);
        assert!(config.auto_tune);
    }
    
    #[test]
    fn test_gpu_context_creation() {
        let config = GpuConfig::default();
        let context = GpuContext::new(config);
        assert!(context.is_ok());
        
        let ctx = context.unwrap();
        assert!(!ctx.is_gpu_available()); // Should be CPU fallback
    }
    
    #[test]
    fn test_gpu_array_allocation() {
        let shape = vec![10, 5];
        let array = GpuArray::<f64>::allocate(&shape);
        assert!(array.is_ok());
        
        let arr = array.unwrap();
        assert_eq!(arr.shape(), &shape);
        assert_eq!(arr.size(), 50);
    }
    
    #[test]
    fn test_gpu_kmeans_creation() {
        let gpu_config = GpuConfig::default();
        let kmeans_config = GpuKMeansConfig::default();
        
        let gpu_kmeans = GpuKMeans::<f64>::new(gpu_config, kmeans_config);
        assert!(gpu_kmeans.is_ok());
    }
    
    #[test]
    fn test_gpu_distance_matrix() {
        let gpu_config = GpuConfig::default();
        let distance_matrix = GpuDistanceMatrix::<f64>::new(gpu_config, DistanceMetric::Euclidean);
        assert!(distance_matrix.is_ok());
        
        let data = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let distances = distance_matrix.unwrap().compute_distances(data.view());
        assert!(distances.is_ok());
        
        let dist_matrix = distances.unwrap();
        assert_eq!(dist_matrix.shape(), &[4, 4]);
    }
    
    #[test]
    fn test_memory_manager() {
        let config = MemoryManagerConfig::default();
        let mut manager = GpuMemoryManager::new(config);
        
        let ptr = manager.allocate(1024);
        assert!(ptr.is_ok());
        
        let ptr_val = ptr.unwrap();
        let dealloc_result = manager.deallocate(ptr_val, 1024);
        assert!(dealloc_result.is_ok());
        
        let stats = manager.get_stats();
        assert!(stats.total_allocated >= 0);
    }
}