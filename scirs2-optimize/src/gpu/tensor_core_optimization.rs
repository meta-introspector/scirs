//! Tensor Core optimizations for high-performance GPU acceleration
//!
//! This module leverages NVIDIA Tensor Cores for accelerated matrix operations
//! in optimization algorithms, providing significant speedup for suitable workloads.

use crate::error::{ScirsError, ScirsResult};
use ndarray::{Array1, Array2};
use scirs2_core::gpu::{GpuContext, GpuKernel};
use std::sync::Arc;

/// Tensor Core acceleration configuration
#[derive(Debug, Clone)]
pub struct TensorCoreOptimizationConfig {
    /// Use mixed precision (FP16 for computation, FP32 for accumulation)
    pub mixed_precision: bool,
    /// Tile size for matrix operations
    pub tile_size: usize,
    /// Whether to use automatic mixed precision (AMP)
    pub use_amp: bool,
    /// Loss scaling for numerical stability in mixed precision
    pub loss_scale: f32,
    /// Gradient clipping threshold
    pub gradient_clip_threshold: Option<f32>,
}

impl Default for TensorCoreOptimizationConfig {
    fn default() -> Self {
        Self {
            mixed_precision: true,
            tile_size: 16, // Optimal for most Tensor Core operations
            use_amp: true,
            loss_scale: 65536.0,
            gradient_clip_threshold: Some(1.0),
        }
    }
}

/// Tensor Core-accelerated matrix operations for optimization
pub struct TensorCoreOptimizer {
    context: Arc<GpuContext>,
    config: TensorCoreOptimizationConfig,
    gemm_kernel: GpuKernel,
    batch_gemm_kernel: GpuKernel,
    gradient_kernel: GpuKernel,
}

impl TensorCoreOptimizer {
    /// Create a new Tensor Core optimizer
    pub fn new(
        context: Arc<GpuContext>,
        config: TensorCoreOptimizationConfig,
    ) -> ScirsResult<Self> {
        // Check Tensor Core capability
        if !context.supports_tensor_cores()? {
            return Err(ScirsError::NotSupported(
                "Tensor Cores not available on this device".to_string(),
            ));
        }

        let gemm_kernel = Self::create_gemm_kernel(&context, &config)?;
        let batch_gemm_kernel = Self::create_batch_gemm_kernel(&context, &config)?;
        let gradient_kernel = Self::create_gradient_kernel(&context, &config)?;

        Ok(Self {
            context,
            config,
            gemm_kernel,
            batch_gemm_kernel,
            gradient_kernel,
        })
    }

    /// Create optimized GEMM kernel using Tensor Cores
    fn create_gemm_kernel(
        context: &Arc<GpuContext>,
        config: &TensorCoreOptimizationConfig,
    ) -> ScirsResult<GpuKernel> {
        let kernel_source = if config.mixed_precision {
            format!(
                r#"
                #include <cuda_fp16.h>
                #include <mma.h>
                
                using namespace nvcuda;
                
                extern "C" __global__ void tensor_core_gemm_mixed(
                    const half* A,
                    const half* B,
                    float* C,
                    int M, int N, int K,
                    float alpha, float beta
                ) {{
                    const int WMMA_M = 16;
                    const int WMMA_N = 16;
                    const int WMMA_K = 16;
                    
                    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
                    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
                    
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                    
                    wmma::fill_fragment(acc_frag, 0.0f);
                    
                    for (int i = 0; i < K; i += WMMA_K) {{
                        int aRow = warpM * WMMA_M;
                        int aCol = i;
                        int bRow = i;
                        int bCol = warpN * WMMA_N;
                        
                        if (aRow < M && aCol < K && bRow < K && bCol < N) {{
                            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
                            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
                            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                        }}
                    }}
                    
                    int cRow = warpM * WMMA_M;
                    int cCol = warpN * WMMA_N;
                    
                    if (cRow < M && cCol < N) {{
                        wmma::load_matrix_sync(c_frag, C + cRow * N + cCol, N, wmma::mem_row_major);
                        for (int i = 0; i < c_frag.num_elements; i++) {{
                            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
                        }}
                        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
                    }}
                }}
            "#,
            )
        } else {
            format!(
                r#"
                #include <mma.h>
                
                using namespace nvcuda;
                
                extern "C" __global__ void tensor_core_gemm_fp32(
                    const float* A,
                    const float* B,
                    float* C,
                    int M, int N, int K,
                    float alpha, float beta
                ) {{
                    // Standard FP32 Tensor Core implementation
                    const int WMMA_M = 16;
                    const int WMMA_N = 16;
                    const int WMMA_K = 8;
                    
                    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
                    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
                    
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag;
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                    
                    wmma::fill_fragment(acc_frag, 0.0f);
                    
                    for (int i = 0; i < K; i += WMMA_K) {{
                        int aRow = warpM * WMMA_M;
                        int aCol = i;
                        int bRow = i;
                        int bCol = warpN * WMMA_N;
                        
                        if (aRow < M && aCol < K && bRow < K && bCol < N) {{
                            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
                            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
                            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                        }}
                    }}
                    
                    int cRow = warpM * WMMA_M;
                    int cCol = warpN * WMMA_N;
                    
                    if (cRow < M && cCol < N) {{
                        wmma::load_matrix_sync(c_frag, C + cRow * N + cCol, N, wmma::mem_row_major);
                        for (int i = 0; i < c_frag.num_elements; i++) {{
                            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
                        }}
                        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
                    }}
                }}
            "#,
            )
        };

        let kernel_name = if config.mixed_precision {
            "tensor_core_gemm_mixed"
        } else {
            "tensor_core_gemm_fp32"
        };

        context.compile_kernel(kernel_name, &kernel_source)
    }

    /// Create batch GEMM kernel for multiple matrix multiplications
    fn create_batch_gemm_kernel(
        context: &Arc<GpuContext>,
        config: &TensorCoreOptimizationConfig,
    ) -> ScirsResult<GpuKernel> {
        let kernel_source = r#"
            #include <cuda_fp16.h>
            #include <mma.h>
            
            using namespace nvcuda;
            
            extern "C" __global__ void tensor_core_batch_gemm(
                const half** A_array,
                const half** B_array,
                float** C_array,
                int* M_array,
                int* N_array,
                int* K_array,
                float* alpha_array,
                float* beta_array,
                int batch_count
            ) {
                int batch_id = blockIdx.z;
                if (batch_id >= batch_count) return;
                
                const half* A = A_array[batch_id];
                const half* B = B_array[batch_id];
                float* C = C_array[batch_id];
                int M = M_array[batch_id];
                int N = N_array[batch_id];
                int K = K_array[batch_id];
                float alpha = alpha_array[batch_id];
                float beta = beta_array[batch_id];
                
                const int WMMA_M = 16;
                const int WMMA_N = 16;
                const int WMMA_K = 16;
                
                int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
                int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
                
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                
                wmma::fill_fragment(acc_frag, 0.0f);
                
                for (int i = 0; i < K; i += WMMA_K) {
                    int aRow = warpM * WMMA_M;
                    int aCol = i;
                    int bRow = i;
                    int bCol = warpN * WMMA_N;
                    
                    if (aRow < M && aCol < K && bRow < K && bCol < N) {
                        wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
                        wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
                        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                    }
                }
                
                int cRow = warpM * WMMA_M;
                int cCol = warpN * WMMA_N;
                
                if (cRow < M && cCol < N) {
                    wmma::load_matrix_sync(c_frag, C + cRow * N + cCol, N, wmma::mem_row_major);
                    for (int i = 0; i < c_frag.num_elements; i++) {
                        c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
                    }
                    wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
                }
            }
        "#;

        context.compile_kernel("tensor_core_batch_gemm", kernel_source)
    }

    /// Create gradient computation kernel with Tensor Core acceleration
    fn create_gradient_kernel(
        context: &Arc<GpuContext>,
        config: &TensorCoreOptimizationConfig,
    ) -> ScirsResult<GpuKernel> {
        let kernel_source = r#"
            #include <cuda_fp16.h>
            #include <mma.h>
            
            using namespace nvcuda;
            
            extern "C" __global__ void tensor_core_gradient_computation(
                const half* jacobian,
                const half* residuals,
                float* gradients,
                int n_points,
                int n_dims,
                float loss_scale
            ) {
                // Use Tensor Cores to compute J^T * r efficiently
                const int WMMA_M = 16;
                const int WMMA_N = 16;
                const int WMMA_K = 16;
                
                int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
                int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
                
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> jt_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> r_frag;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
                
                wmma::fill_fragment(acc_frag, 0.0f);
                
                // Compute J^T * r using Tensor Cores
                for (int k = 0; k < n_points; k += WMMA_K) {
                    if (warpM * WMMA_M < n_dims && k < n_points) {
                        // Load transposed Jacobian and residuals
                        wmma::load_matrix_sync(jt_frag, jacobian + k * n_dims + warpM * WMMA_M, n_dims);
                        wmma::load_matrix_sync(r_frag, residuals + k, 1);
                        wmma::mma_sync(acc_frag, jt_frag, r_frag, acc_frag);
                    }
                }
                
                // Store result with loss scaling
                if (warpM * WMMA_M < n_dims) {
                    for (int i = 0; i < WMMA_M && warpM * WMMA_M + i < n_dims; i++) {
                        gradients[warpM * WMMA_M + i] = acc_frag.x[i] / loss_scale;
                    }
                }
            }
        "#;

        context.compile_kernel("tensor_core_gradient_computation", kernel_source)
    }

    /// Perform optimized matrix multiplication using Tensor Cores
    pub fn gemm(
        &self,
        a: &Array2<f64>,
        b: &Array2<f64>,
        c: &mut Array2<f64>,
        alpha: f64,
        beta: f64,
    ) -> ScirsResult<()> {
        let (m, k1) = a.dim();
        let (k2, n) = b.dim();

        if k1 != k2 {
            return Err(ScirsError::InvalidInput(
                "Matrix dimensions don't match for multiplication".to_string(),
            ));
        }

        if self.config.mixed_precision {
            // Convert to FP16 for computation
            let a_fp16 = self.context.convert_to_fp16(a)?;
            let b_fp16 = self.context.convert_to_fp16(b)?;

            // Calculate grid and block dimensions
            let tile_size = self.config.tile_size;
            let grid_x = (m + tile_size - 1) / tile_size;
            let grid_y = (n + tile_size - 1) / tile_size;
            let block_size = 256;

            self.gemm_kernel.launch(
                (grid_x as u32, grid_y as u32, 1),
                (block_size as u32, 1, 1),
                &[
                    a_fp16.as_ptr(),
                    b_fp16.as_ptr(),
                    c.as_ptr(),
                    &(m as i32),
                    &(n as i32),
                    &(k1 as i32),
                    &(alpha as f32),
                    &(beta as f32),
                ],
                None,
            )?;
        } else {
            // Use FP32 Tensor Cores
            let tile_size = self.config.tile_size;
            let grid_x = (m + tile_size - 1) / tile_size;
            let grid_y = (n + tile_size - 1) / tile_size;
            let block_size = 256;

            self.gemm_kernel.launch(
                (grid_x as u32, grid_y as u32, 1),
                (block_size as u32, 1, 1),
                &[
                    a.as_ptr(),
                    b.as_ptr(),
                    c.as_ptr(),
                    &(m as i32),
                    &(n as i32),
                    &(k1 as i32),
                    &(alpha as f32),
                    &(beta as f32),
                ],
                None,
            )?;
        }

        Ok(())
    }

    /// Perform batch matrix multiplication using Tensor Cores
    pub fn batch_gemm(
        &self,
        a_batch: &[&Array2<f64>],
        b_batch: &[&Array2<f64>],
        c_batch: &mut [&mut Array2<f64>],
        alpha_batch: &[f64],
        beta_batch: &[f64],
    ) -> ScirsResult<()> {
        let batch_count = a_batch.len();

        if batch_count != b_batch.len() || batch_count != c_batch.len() {
            return Err(ScirsError::InvalidInput(
                "Batch sizes must match".to_string(),
            ));
        }

        // Prepare batch arrays on GPU
        let mut a_ptrs = Vec::new();
        let mut b_ptrs = Vec::new();
        let mut c_ptrs = Vec::new();
        let mut m_array = Vec::new();
        let mut n_array = Vec::new();
        let mut k_array = Vec::new();

        for i in 0..batch_count {
            let (m, k1) = a_batch[i].shape();
            let (k2, n) = b_batch[i].shape();

            if k1 != k2 {
                return Err(ScirsError::InvalidInput(format!(
                    "Matrix dimensions don't match for batch {}",
                    i
                )));
            }

            a_ptrs.push(a_batch[i].as_ptr());
            b_ptrs.push(b_batch[i].as_ptr());
            c_ptrs.push(c_batch[i].as_ptr());
            m_array.push(m as i32);
            n_array.push(n as i32);
            k_array.push(k1 as i32);
        }

        // Upload batch data to GPU
        let gpu_a_ptrs = self.context.upload_ptr_array(&a_ptrs)?;
        let gpu_b_ptrs = self.context.upload_ptr_array(&b_ptrs)?;
        let gpu_c_ptrs = self.context.upload_ptr_array(&c_ptrs)?;
        let gpu_m_array = self.context.upload_array(&Array1::from(m_array))?;
        let gpu_n_array = self.context.upload_array(&Array1::from(n_array))?;
        let gpu_k_array = self.context.upload_array(&Array1::from(k_array))?;
        let gpu_alpha_array = self.context.upload_array(&Array1::from(
            alpha_batch.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        ))?;
        let gpu_beta_array = self.context.upload_array(&Array1::from(
            beta_batch.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        ))?;

        // Launch batch kernel
        let tile_size = self.config.tile_size;
        let max_dim = m_array
            .iter()
            .zip(n_array.iter())
            .map(|(&m, &n)| m.max(n))
            .max()
            .unwrap_or(1) as usize;
        let grid_x = (max_dim + tile_size - 1) / tile_size;
        let grid_y = grid_x;
        let block_size = 256;

        self.batch_gemm_kernel.launch(
            (grid_x as u32, grid_y as u32, batch_count as u32),
            (block_size as u32, 1, 1),
            &[
                gpu_a_ptrs.as_ptr(),
                gpu_b_ptrs.as_ptr(),
                gpu_c_ptrs.as_ptr(),
                gpu_m_array.as_ptr(),
                gpu_n_array.as_ptr(),
                gpu_k_array.as_ptr(),
                gpu_alpha_array.as_ptr(),
                gpu_beta_array.as_ptr(),
                &(batch_count as i32),
            ],
            None,
        )?;

        Ok(())
    }

    /// Compute gradients using Tensor Core acceleration
    pub fn compute_gradients(
        &self,
        jacobian: &Array2<f64>,
        residuals: &Array1<f64>,
    ) -> ScirsResult<Array1<f64>> {
        let (n_points, n_dims) = jacobian.shape();
        let gradients = self.context.allocate_array::<f64>(&[n_dims, 1])?;

        if self.config.mixed_precision {
            // Convert to FP16 for computation
            let jacobian_fp16 = self.context.convert_to_fp16(jacobian)?;
            let residuals_fp16 = self.context.convert_to_fp16(residuals)?;

            let tile_size = self.config.tile_size;
            let grid_x = (n_dims + tile_size - 1) / tile_size;
            let block_size = 256;

            self.gradient_kernel.launch(
                (grid_x as u32, 1, 1),
                (block_size as u32, 1, 1),
                &[
                    jacobian_fp16.as_ptr(),
                    residuals_fp16.as_ptr(),
                    gradients.as_ptr(),
                    &(n_points as i32),
                    &(n_dims as i32),
                    &self.config.loss_scale,
                ],
                None,
            )?;
        }

        Ok(gradients)
    }

    /// Check if gradient clipping is needed and apply it
    pub fn clip_gradients(&self, gradients: &mut Array1<f64>) -> ScirsResult<()> {
        if let Some(threshold) = self.config.gradient_clip_threshold {
            // Compute gradient norm
            let grad_norm_squared = self.context.dot(gradients, gradients)?;
            let grad_norm = grad_norm_squared.sqrt();

            if grad_norm > threshold as f64 {
                let scale_factor = threshold as f64 / grad_norm;
                self.context.scale_array(gradients, scale_factor)?;
            }
        }
        Ok(())
    }

    /// Get the current configuration
    pub fn config(&self) -> &TensorCoreOptimizationConfig {
        &self.config
    }

    /// Update loss scale for automatic mixed precision
    pub fn update_loss_scale(&mut self, loss_scale: f32) {
        self.config.loss_scale = loss_scale;
    }

    /// Check if computation overflowed (for AMP)
    pub fn check_overflow(&self, tensor: &Array2<f64>) -> ScirsResult<bool> {
        // Check for infinite or NaN values
        self.context.has_nan_or_inf(tensor)
    }
}

/// Automatic Mixed Precision (AMP) manager for optimization
pub struct AMPManager {
    loss_scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: u32,
    consecutive_unskipped: u32,
}

impl AMPManager {
    /// Create a new AMP manager
    pub fn new() -> Self {
        Self {
            loss_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            consecutive_unskipped: 0,
        }
    }

    /// Update loss scale based on overflow detection
    pub fn update(&mut self, found_overflow: bool) -> f32 {
        if found_overflow {
            self.loss_scale *= self.backoff_factor;
            self.consecutive_unskipped = 0;
        } else {
            self.consecutive_unskipped += 1;
            if self.consecutive_unskipped >= self.growth_interval {
                self.loss_scale *= self.growth_factor;
                self.consecutive_unskipped = 0;
            }
        }

        // Clamp loss scale to reasonable bounds
        self.loss_scale = self.loss_scale.max(1.0).min(2_f32.powi(20));
        self.loss_scale
    }

    /// Get current loss scale
    pub fn loss_scale(&self) -> f32 {
        self.loss_scale
    }
}

impl Default for AMPManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_core_config() {
        let config = TensorCoreOptimizationConfig::default();
        assert!(config.mixed_precision);
        assert_eq!(config.tile_size, 16);
        assert!(config.use_amp);
        assert_eq!(config.loss_scale, 65536.0);
    }

    #[test]
    fn test_amp_manager() {
        let mut manager = AMPManager::new();
        assert_eq!(manager.loss_scale(), 65536.0);

        // Test overflow handling
        let new_scale = manager.update(true);
        assert_eq!(new_scale, 32768.0);

        // Test growth
        for _ in 0..2000 {
            manager.update(false);
        }
        let grown_scale = manager.loss_scale();
        assert!(grown_scale > 32768.0);
    }

    #[test]
    #[ignore = "Requires Tensor Core capable GPU"]
    fn test_tensor_core_optimizer() {
        // This would test the actual Tensor Core optimizer
        // Implementation depends on the actual scirs2-core GPU infrastructure
    }
}
