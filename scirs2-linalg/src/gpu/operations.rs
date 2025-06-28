//! GPU-accelerated linear algebra operations

use crate::error::{LinalgError, LinalgResult};
use super::{GpuContext, GpuLinalgOps, AutoGpuSelector};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, NumAssign, Zero};
use std::fmt::Debug;

/// Default GPU threshold for switching from CPU to GPU (number of elements)
pub const DEFAULT_GPU_THRESHOLD: usize = 50_000;

/// GPU operation dispatcher that automatically selects CPU or GPU
pub struct GpuOperationDispatcher<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    gpu_threshold: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> GpuOperationDispatcher<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    /// Create a new GPU operation dispatcher
    pub fn new() -> Self {
        Self {
            gpu_threshold: DEFAULT_GPU_THRESHOLD,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Create dispatcher with custom GPU threshold
    pub fn with_threshold(threshold: usize) -> Self {
        Self {
            gpu_threshold: threshold,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Set the GPU threshold
    pub fn set_threshold(&mut self, threshold: usize) {
        self.gpu_threshold = threshold;
    }
    
    /// Get the current GPU threshold
    pub fn threshold(&self) -> usize {
        self.gpu_threshold
    }
}

impl<T> Default for GpuOperationDispatcher<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> GpuLinalgOps<T> for GpuOperationDispatcher<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    fn gpu_matvec(
        &self,
        ctx: &dyn GpuContext,
        a: &ArrayView2<T>,
        x: &ArrayView1<T>,
    ) -> LinalgResult<Array1<T>> {
        let (m, n) = a.dim();
        
        if n != x.len() {
            return Err(LinalgError::ShapeError(format!(
                "Matrix columns ({}) must match vector length ({})",
                n, x.len()
            )));
        }
        
        // For now, fall back to CPU implementation
        // In a real implementation, this would use GPU kernels
        self.cpu_matvec(a, x)
    }
    
    fn gpu_matmul(
        &self,
        ctx: &dyn GpuContext,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
    ) -> LinalgResult<Array2<T>> {
        let (m, k1) = a.dim();
        let (k2, n) = b.dim();
        
        if k1 != k2 {
            return Err(LinalgError::ShapeError(format!(
                "Matrix dimensions mismatch: {}x{} * {}x{}",
                m, k1, k2, n
            )));
        }
        
        // For now, fall back to CPU implementation
        self.cpu_matmul(a, b)
    }
    
    fn gpu_dot(
        &self,
        ctx: &dyn GpuContext,
        x: &ArrayView1<T>,
        y: &ArrayView1<T>,
    ) -> LinalgResult<T> {
        if x.len() != y.len() {
            return Err(LinalgError::ShapeError(format!(
                "Vector lengths must match: {} != {}",
                x.len(), y.len()
            )));
        }
        
        // For now, fall back to CPU implementation
        Ok(self.cpu_dot(x, y))
    }
    
    fn gpu_norm(
        &self,
        ctx: &dyn GpuContext,
        x: &ArrayView1<T>,
    ) -> LinalgResult<T> {
        // For now, fall back to CPU implementation
        Ok(self.cpu_norm(x))
    }
    
    fn gpu_elementwise_add(
        &self,
        ctx: &dyn GpuContext,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
    ) -> LinalgResult<Array2<T>> {
        if a.shape() != b.shape() {
            return Err(LinalgError::ShapeError(format!(
                "Matrix shapes must match: {:?} != {:?}",
                a.shape(), b.shape()
            )));
        }
        
        // For now, fall back to CPU implementation
        self.cpu_elementwise_add(a, b)
    }
    
    fn gpu_elementwise_mul(
        &self,
        ctx: &dyn GpuContext,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
    ) -> LinalgResult<Array2<T>> {
        if a.shape() != b.shape() {
            return Err(LinalgError::ShapeError(format!(
                "Matrix shapes must match: {:?} != {:?}",
                a.shape(), b.shape()
            )));
        }
        
        // For now, fall back to CPU implementation
        self.cpu_elementwise_mul(a, b)
    }
}

impl<T> GpuOperationDispatcher<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    /// CPU fallback for matrix-vector multiplication
    fn cpu_matvec(&self, a: &ArrayView2<T>, x: &ArrayView1<T>) -> LinalgResult<Array1<T>> {
        let (m, n) = a.dim();
        let mut result = Array1::zeros(m);
        
        for i in 0..m {
            let mut sum = T::zero();
            for j in 0..n {
                sum += a[[i, j]] * x[j];
            }
            result[i] = sum;
        }
        
        Ok(result)
    }
    
    /// CPU fallback for matrix-matrix multiplication
    fn cpu_matmul(&self, a: &ArrayView2<T>, b: &ArrayView2<T>) -> LinalgResult<Array2<T>> {
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        let mut result = Array2::zeros((m, n));
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum += a[[i, l]] * b[[l, j]];
                }
                result[[i, j]] = sum;
            }
        }
        
        Ok(result)
    }
    
    /// CPU fallback for dot product
    fn cpu_dot(&self, x: &ArrayView1<T>, y: &ArrayView1<T>) -> T {
        let mut result = T::zero();
        for (a, b) in x.iter().zip(y.iter()) {
            result += *a * *b;
        }
        result
    }
    
    /// CPU fallback for vector norm
    fn cpu_norm(&self, x: &ArrayView1<T>) -> T {
        let mut sum_sq = T::zero();
        for &val in x.iter() {
            sum_sq += val * val;
        }
        sum_sq.sqrt()
    }
    
    /// CPU fallback for element-wise addition
    fn cpu_elementwise_add(&self, a: &ArrayView2<T>, b: &ArrayView2<T>) -> LinalgResult<Array2<T>> {
        let mut result = Array2::zeros(a.dim());
        for ((i, j), &val_a) in a.indexed_iter() {
            result[[i, j]] = val_a + b[[i, j]];
        }
        Ok(result)
    }
    
    /// CPU fallback for element-wise multiplication
    fn cpu_elementwise_mul(&self, a: &ArrayView2<T>, b: &ArrayView2<T>) -> LinalgResult<Array2<T>> {
        let mut result = Array2::zeros(a.dim());
        for ((i, j), &val_a) in a.indexed_iter() {
            result[[i, j]] = val_a * b[[i, j]];
        }
        Ok(result)
    }
}

impl<T> AutoGpuSelector<T> for GpuOperationDispatcher<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    fn auto_matvec(
        &self,
        a: &ArrayView2<T>,
        x: &ArrayView1<T>,
        gpu_context: Option<&dyn GpuContext>,
    ) -> LinalgResult<Array1<T>> {
        let elements = a.len();
        
        if let Some(ctx) = gpu_context {
            if elements > self.gpu_threshold {
                // Use GPU implementation
                return self.gpu_matvec(ctx, a, x);
            }
        }
        
        // Use CPU implementation
        self.cpu_matvec(a, x)
    }
    
    fn auto_matmul(
        &self,
        a: &ArrayView2<T>,
        b: &ArrayView2<T>,
        gpu_context: Option<&dyn GpuContext>,
    ) -> LinalgResult<Array2<T>> {
        let elements = a.len() + b.len();
        
        if let Some(ctx) = gpu_context {
            if elements > self.gpu_threshold {
                // Use GPU implementation
                return self.gpu_matmul(ctx, a, b);
            }
        }
        
        // Use CPU implementation
        self.cpu_matmul(a, b)
    }
}

/// Kernel management for GPU operations
pub struct GpuKernelManager {
    kernel_cache: std::collections::HashMap<String, String>,
}

impl GpuKernelManager {
    /// Create a new kernel manager
    pub fn new() -> Self {
        Self {
            kernel_cache: std::collections::HashMap::new(),
        }
    }
    
    /// Load a kernel from source
    pub fn load_kernel(&mut self, name: &str, source: &str) -> LinalgResult<()> {
        self.kernel_cache.insert(name.to_string(), source.to_string());
        Ok(())
    }
    
    /// Get kernel source by name
    pub fn get_kernel(&self, name: &str) -> Option<&str> {
        self.kernel_cache.get(name).map(|s| s.as_str())
    }
    
    /// List available kernels
    pub fn list_kernels(&self) -> Vec<&str> {
        self.kernel_cache.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for GpuKernelManager {
    fn default() -> Self {
        let mut manager = Self::new();
        
        // Load default kernels
        let _ = manager.load_kernel("matvec_f32", include_str!("../../../kernels/matvec_f32.cl"));
        let _ = manager.load_kernel("matmul_f32", include_str!("../../../kernels/matmul_f32.cl"));
        
        manager
    }
}

/// Performance profiler for GPU operations
pub struct GpuPerformanceProfiler {
    measurements: std::collections::HashMap<String, Vec<f64>>,
}

impl GpuPerformanceProfiler {
    /// Create a new performance profiler
    pub fn new() -> Self {
        Self {
            measurements: std::collections::HashMap::new(),
        }
    }
    
    /// Record a performance measurement
    pub fn record(&mut self, operation: &str, time_seconds: f64) {
        self.measurements
            .entry(operation.to_string())
            .or_insert_with(Vec::new)
            .push(time_seconds);
    }
    
    /// Get average time for an operation
    pub fn average_time(&self, operation: &str) -> Option<f64> {
        self.measurements.get(operation).map(|times| {
            times.iter().sum::<f64>() / times.len() as f64
        })
    }
    
    /// Get best time for an operation
    pub fn best_time(&self, operation: &str) -> Option<f64> {
        self.measurements.get(operation)
            .and_then(|times| times.iter().min_by(|a, b| a.partial_cmp(b).unwrap()))
            .copied()
    }
    
    /// Get all recorded operations
    pub fn operations(&self) -> Vec<&str> {
        self.measurements.keys().map(|s| s.as_str()).collect()
    }
    
    /// Clear all measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
    }
}

impl Default for GpuPerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gpu_operation_dispatcher() {
        let dispatcher = GpuOperationDispatcher::<f64>::new();
        assert_eq!(dispatcher.threshold(), DEFAULT_GPU_THRESHOLD);
        
        let mut dispatcher = GpuOperationDispatcher::with_threshold(1000);
        assert_eq!(dispatcher.threshold(), 1000);
        
        dispatcher.set_threshold(2000);
        assert_eq!(dispatcher.threshold(), 2000);
    }

    #[test]
    fn test_cpu_fallback_operations() {
        let dispatcher = GpuOperationDispatcher::<f64>::new();
        
        // Test matrix-vector multiplication
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let x = array![1.0, 2.0];
        let result = dispatcher.cpu_matvec(&a.view(), &x.view()).unwrap();
        assert_eq!(result, array![5.0, 11.0]);
        
        // Test matrix-matrix multiplication
        let b = array![[1.0, 0.0], [0.0, 1.0]];
        let result = dispatcher.cpu_matmul(&a.view(), &b.view()).unwrap();
        assert_eq!(result, a);
        
        // Test dot product
        let y = array![2.0, 3.0];
        let dot_result = dispatcher.cpu_dot(&x.view(), &y.view());
        assert_eq!(dot_result, 8.0);
        
        // Test norm
        let norm_result = dispatcher.cpu_norm(&x.view());
        assert!((norm_result - (5.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_manager() {
        let mut manager = GpuKernelManager::new();
        
        manager.load_kernel("test_kernel", "kernel void test() {}").unwrap();
        assert!(manager.get_kernel("test_kernel").is_some());
        assert!(manager.get_kernel("nonexistent").is_none());
        
        let kernels = manager.list_kernels();
        assert!(kernels.contains(&"test_kernel"));
    }

    #[test]
    fn test_performance_profiler() {
        let mut profiler = GpuPerformanceProfiler::new();
        
        profiler.record("matmul", 0.1);
        profiler.record("matmul", 0.2);
        profiler.record("matvec", 0.05);
        
        assert_eq!(profiler.average_time("matmul"), Some(0.15));
        assert_eq!(profiler.best_time("matmul"), Some(0.1));
        assert_eq!(profiler.average_time("matvec"), Some(0.05));
        
        let ops = profiler.operations();
        assert!(ops.contains(&"matmul"));
        assert!(ops.contains(&"matvec"));
        
        profiler.clear();
        assert!(profiler.operations().is_empty());
    }
}