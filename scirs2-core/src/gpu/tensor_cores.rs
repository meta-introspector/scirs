//! Tensor core acceleration support for modern GPUs
//!
//! This module provides support for hardware-accelerated tensor operations using
//! specialized tensor processing units available on modern GPUs (NVIDIA Tensor Cores,
//! AMD Matrix Cores, etc.).

use crate::gpu::{GpuBackend, GpuBuffer, GpuError};
use std::fmt;
use thiserror::Error;

/// Supported tensor data types for hardware acceleration
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TensorDataType {
    /// 16-bit floating point (half precision)
    Float16,
    /// Brain floating point 16-bit
    BFloat16,
    /// 32-bit floating point (single precision)
    Float32,
    /// 64-bit floating point (double precision)
    Float64,
    /// 8-bit signed integer
    Int8,
    /// 4-bit integer (packed)
    Int4,
    /// 1-bit binary
    Binary,
    /// Mixed precision (accumulation in higher precision)
    Mixed(Box<TensorDataType>, Box<TensorDataType>), // (input, accumulator)
}

impl fmt::Display for TensorDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorDataType::Float16 => write!(f, "f16"),
            TensorDataType::BFloat16 => write!(f, "bf16"),
            TensorDataType::Float32 => write!(f, "f32"),
            TensorDataType::Float64 => write!(f, "f64"),
            TensorDataType::Int8 => write!(f, "i8"),
            TensorDataType::Int4 => write!(f, "i4"),
            TensorDataType::Binary => write!(f, "binary"),
            TensorDataType::Mixed(input, accum) => write!(f, "mixed({}, {})", input, accum),
        }
    }
}

/// Tensor core operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCoreOp {
    /// Matrix multiplication (GEMM)
    MatrixMultiply,
    /// Convolution operation
    Convolution,
    /// Attention mechanism (scaled dot-product attention)
    Attention,
    /// Sparse matrix operations
    SparseOps,
    /// Custom tensor operation
    Custom(&'static str),
}

/// Tensor core capabilities for different GPU architectures
#[derive(Debug, Clone, Default)]
pub struct TensorCoreCapabilities {
    /// Whether tensor cores are available
    pub available: bool,
    /// Supported data types
    pub supported_types: Vec<TensorDataType>,
    /// Supported operations
    pub supported_ops: Vec<TensorCoreOp>,
    /// Matrix dimensions supported (M, N, K)
    pub supported_dimensions: Vec<(usize, usize, usize)>,
    /// Peak throughput in TOPS (Tera-Operations Per Second)
    pub peak_tops: Option<f64>,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: Option<f64>,
    /// Architecture-specific features
    pub arch_features: Vec<String>,
}

/// Tensor core configuration for optimized operations
#[derive(Debug, Clone)]
pub struct TensorCoreConfig {
    /// Preferred data type for computations
    pub data_type: TensorDataType,
    /// Use mixed precision if available
    pub use_mixed_precision: bool,
    /// Enable automatic type conversion
    pub auto_convert: bool,
    /// Tile size for operations
    pub tile_size: (usize, usize),
    /// Use sparse operations if beneficial
    pub use_sparse: bool,
    /// Architecture-specific optimizations
    pub arch_optimizations: Vec<String>,
}

impl Default for TensorCoreConfig {
    fn default() -> Self {
        Self {
            data_type: TensorDataType::Float16,
            use_mixed_precision: true,
            auto_convert: true,
            tile_size: (16, 16),
            use_sparse: false,
            arch_optimizations: Vec::new(),
        }
    }
}

/// Error types for tensor core operations
#[derive(Error, Debug)]
pub enum TensorCoreError {
    /// Tensor cores not available on this device
    #[error("Tensor cores not available on this device")]
    NotAvailable,

    /// Unsupported data type for tensor core operations
    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(TensorDataType),

    /// Unsupported operation
    #[error("Unsupported operation: {0:?}")]
    UnsupportedOperation(TensorCoreOp),

    /// Invalid matrix dimensions
    #[error("Invalid matrix dimensions: {m}x{n}x{k}")]
    InvalidDimensions { m: usize, n: usize, k: usize },

    /// Memory alignment error
    #[error("Memory alignment error: {0}")]
    MemoryAlignment(String),

    /// Performance hint
    #[error("Performance warning: {0}")]
    PerformanceWarning(String),

    /// Underlying GPU error
    #[error("GPU error: {0}")]
    GpuError(#[from] GpuError),
}

/// Tensor core manager for handling hardware acceleration
pub struct TensorCoreManager {
    backend: GpuBackend,
    capabilities: TensorCoreCapabilities,
    config: TensorCoreConfig,
}

impl TensorCoreManager {
    /// Create a new tensor core manager for the given backend
    pub fn new(backend: GpuBackend) -> Result<Self, TensorCoreError> {
        let capabilities = Self::detect_capabilities(backend)?;

        if !capabilities.available {
            return Err(TensorCoreError::NotAvailable);
        }

        let config = Self::optimal_config(&capabilities);

        Ok(Self {
            backend,
            capabilities,
            config,
        })
    }

    /// Get tensor core capabilities for the current device
    pub const fn capabilities(&self) -> &TensorCoreCapabilities {
        &self.capabilities
    }

    /// Get current configuration
    pub const fn config(&self) -> &TensorCoreConfig {
        &self.config
    }

    /// Update tensor core configuration
    pub fn set_config(&mut self, config: TensorCoreConfig) -> Result<(), TensorCoreError> {
        // Validate configuration against capabilities
        if !self
            .capabilities
            .supported_types
            .contains(&config.data_type)
        {
            return Err(TensorCoreError::UnsupportedDataType(config.data_type));
        }

        self.config = config;
        Ok(())
    }

    /// Check if an operation is supported with current configuration
    pub fn is_operation_supported(&self, op: TensorCoreOp) -> bool {
        self.capabilities.supported_ops.contains(&op)
    }

    /// Check if dimensions are optimal for tensor core operations
    pub fn are_dimensions_optimal(&self, m: usize, n: usize, k: usize) -> bool {
        // Check if dimensions are multiples of tensor core tile sizes
        match self.backend {
            GpuBackend::Cuda => {
                // NVIDIA tensor cores typically work best with multiples of 16
                m % 16 == 0 && n % 16 == 0 && k % 16 == 0
            }
            GpuBackend::Rocm => {
                // AMD matrix cores typically work best with multiples of 32
                m % 32 == 0 && n % 32 == 0 && k % 32 == 0
            }
            _ => false,
        }
    }

    /// Get performance hints for given dimensions
    pub fn get_performance_hints(&self, m: usize, n: usize, k: usize) -> Vec<String> {
        let mut hints = Vec::new();

        if !self.are_dimensions_optimal(m, n, k) {
            hints.push(format!(
                "Consider padding dimensions to multiples of {} for optimal performance",
                match self.backend {
                    GpuBackend::Cuda => 16,
                    GpuBackend::Rocm => 32,
                    _ => 16,
                }
            ));
        }

        if self.config.use_mixed_precision && self.config.data_type == TensorDataType::Float32 {
            hints.push(
                "Consider using Float16 or BFloat16 for better tensor core utilization".to_string(),
            );
        }

        if m * n * k < 1024 * 1024 {
            hints.push("Small matrices may not fully utilize tensor cores".to_string());
        }

        hints
    }

    /// Suggest optimal data type for given operation
    pub fn suggest_optimal_type(&self, op: TensorCoreOp) -> Option<TensorDataType> {
        match op {
            TensorCoreOp::MatrixMultiply => {
                if self
                    .capabilities
                    .supported_types
                    .contains(&TensorDataType::BFloat16)
                {
                    Some(TensorDataType::BFloat16)
                } else if self
                    .capabilities
                    .supported_types
                    .contains(&TensorDataType::Float16)
                {
                    Some(TensorDataType::Float16)
                } else {
                    Some(TensorDataType::Float32)
                }
            }
            TensorCoreOp::Convolution => {
                if self
                    .capabilities
                    .supported_types
                    .contains(&TensorDataType::Int8)
                {
                    Some(TensorDataType::Int8)
                } else if self
                    .capabilities
                    .supported_types
                    .contains(&TensorDataType::Float16)
                {
                    Some(TensorDataType::Float16)
                } else {
                    Some(TensorDataType::Float32)
                }
            }
            TensorCoreOp::Attention => {
                // Attention typically benefits from higher precision
                if self
                    .capabilities
                    .supported_types
                    .contains(&TensorDataType::BFloat16)
                {
                    Some(TensorDataType::BFloat16)
                } else {
                    Some(TensorDataType::Float32)
                }
            }
            _ => self.capabilities.supported_types.first().cloned(),
        }
    }

    /// Detect tensor core capabilities for the given backend
    fn detect_capabilities(backend: GpuBackend) -> Result<TensorCoreCapabilities, TensorCoreError> {
        match backend {
            GpuBackend::Cuda => Ok(Self::nvidia_tensor_capabilities()),
            GpuBackend::Rocm => Ok(Self::amd_matrix_capabilities()),
            GpuBackend::Metal => Ok(Self::apple_neural_capabilities()),
            _ => Ok(TensorCoreCapabilities::default()),
        }
    }

    /// NVIDIA Tensor Core capabilities (Volta, Turing, Ampere, Hopper)
    fn nvidia_tensor_capabilities() -> TensorCoreCapabilities {
        TensorCoreCapabilities {
            available: true,
            supported_types: vec![
                TensorDataType::Float16,
                TensorDataType::BFloat16,
                TensorDataType::Float32,
                TensorDataType::Int8,
                TensorDataType::Int4,
                TensorDataType::Binary,
                TensorDataType::Mixed(
                    Box::new(TensorDataType::Float16),
                    Box::new(TensorDataType::Float32),
                ),
            ],
            supported_ops: vec![
                TensorCoreOp::MatrixMultiply,
                TensorCoreOp::Convolution,
                TensorCoreOp::Attention,
                TensorCoreOp::SparseOps,
            ],
            supported_dimensions: vec![
                (16, 16, 16), // Basic tensor core size
                (32, 8, 16),  // Alternative configurations
                (8, 32, 16),
            ],
            peak_tops: Some(312.0),              // Example for A100
            memory_bandwidth_gbps: Some(2039.0), // Example for A100 HBM2e
            arch_features: vec![
                "Sparsity 2:4".to_string(),
                "Multi-precision".to_string(),
                "Transformer Engine".to_string(),
            ],
        }
    }

    /// AMD Matrix Core capabilities (CDNA, RDNA)
    fn amd_matrix_capabilities() -> TensorCoreCapabilities {
        TensorCoreCapabilities {
            available: true,
            supported_types: vec![
                TensorDataType::Float16,
                TensorDataType::BFloat16,
                TensorDataType::Float32,
                TensorDataType::Int8,
                TensorDataType::Mixed(
                    Box::new(TensorDataType::Float16),
                    Box::new(TensorDataType::Float32),
                ),
            ],
            supported_ops: vec![TensorCoreOp::MatrixMultiply, TensorCoreOp::Convolution],
            supported_dimensions: vec![
                (32, 32, 8), // MFMA instruction size
                (16, 16, 16),
            ],
            peak_tops: Some(383.0),              // Example for MI250X
            memory_bandwidth_gbps: Some(3276.0), // Example for MI250X HBM2e
            arch_features: vec!["MFMA instructions".to_string(), "Matrix cores".to_string()],
        }
    }

    /// Apple Neural Engine capabilities
    fn apple_neural_capabilities() -> TensorCoreCapabilities {
        TensorCoreCapabilities {
            available: true,
            supported_types: vec![
                TensorDataType::Float16,
                TensorDataType::Float32,
                TensorDataType::Int8,
            ],
            supported_ops: vec![
                TensorCoreOp::MatrixMultiply,
                TensorCoreOp::Convolution,
                TensorCoreOp::Attention,
            ],
            supported_dimensions: vec![(16, 16, 16)],
            peak_tops: Some(15.8),              // Example for M1 Neural Engine
            memory_bandwidth_gbps: Some(68.25), // Example for M1 unified memory
            arch_features: vec!["Neural Engine".to_string(), "Unified memory".to_string()],
        }
    }

    /// Determine optimal configuration based on capabilities
    fn optimal_config(capabilities: &TensorCoreCapabilities) -> TensorCoreConfig {
        let data_type = if capabilities
            .supported_types
            .contains(&TensorDataType::BFloat16)
        {
            TensorDataType::BFloat16
        } else if capabilities
            .supported_types
            .contains(&TensorDataType::Float16)
        {
            TensorDataType::Float16
        } else {
            TensorDataType::Float32
        };

        let tile_size = capabilities
            .supported_dimensions
            .first()
            .map(|(m, n, _)| (*m, *n))
            .unwrap_or((16, 16));

        TensorCoreConfig {
            data_type,
            use_mixed_precision: capabilities
                .supported_types
                .iter()
                .any(|t| matches!(t, TensorDataType::Mixed(_, _))),
            auto_convert: true,
            tile_size,
            use_sparse: capabilities
                .arch_features
                .iter()
                .any(|f| f.contains("Sparsity")),
            arch_optimizations: capabilities.arch_features.clone(),
        }
    }
}

/// Tensor core operation descriptor
#[derive(Debug, Clone)]
pub struct TensorOperation {
    /// Type of operation
    pub op_type: TensorCoreOp,
    /// Input data type
    pub input_type: TensorDataType,
    /// Output data type
    pub output_type: TensorDataType,
    /// Matrix dimensions (M, N, K)
    pub dimensions: (usize, usize, usize),
    /// Whether to use mixed precision
    pub mixed_precision: bool,
    /// Sparsity pattern if applicable
    pub sparsity: Option<SparsePattern>,
}

/// Sparsity patterns for sparse tensor operations
#[derive(Debug, Clone)]
pub enum SparsePattern {
    /// 2:4 structured sparsity (2 out of every 4 elements are zero)
    Structured2_4,
    /// Random sparsity with given ratio
    Random(f32),
    /// Block sparsity
    Block {
        block_size: (usize, usize),
        sparsity: f32,
    },
    /// Custom sparsity pattern
    Custom(String),
}

/// Tensor core optimized matrix multiplication
pub fn tensor_core_gemm<T>(
    manager: &TensorCoreManager,
    a: &GpuBuffer<T>,
    b: &GpuBuffer<T>,
    c: &mut GpuBuffer<T>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), TensorCoreError>
where
    T: crate::gpu::GpuDataType,
{
    // Validate dimensions
    if !manager.are_dimensions_optimal(m, n, k) {
        let hints = manager.get_performance_hints(m, n, k);
        for hint in hints {
            eprintln!("Performance hint: {}", hint);
        }
    }

    // Check if operation is supported
    if !manager.is_operation_supported(TensorCoreOp::MatrixMultiply) {
        return Err(TensorCoreError::UnsupportedOperation(
            TensorCoreOp::MatrixMultiply,
        ));
    }

    // Validate buffer sizes
    if a.len() < m * k {
        return Err(TensorCoreError::InvalidDimensions { m, n, k });
    }
    if b.len() < k * n {
        return Err(TensorCoreError::InvalidDimensions { m, n, k });
    }
    if c.len() < m * n {
        return Err(TensorCoreError::InvalidDimensions { m, n, k });
    }

    // Generate optimized kernel for this configuration
    let kernel_source = generate_tensor_core_gemm_kernel(manager, m, n, k)?;
    
    // Execute tensor core GEMM
    execute_tensor_core_operation(manager, &kernel_source, a, b, c, m, n, k)?;

    Ok(())
}

/// Generate optimized tensor core GEMM kernel source
fn generate_tensor_core_gemm_kernel(
    manager: &TensorCoreManager,
    m: usize,
    n: usize,
    k: usize,
) -> Result<String, TensorCoreError> {
    let tile_size = manager.config().tile_size;
    let data_type = &manager.config().data_type;
    let use_mixed_precision = manager.config().use_mixed_precision;

    match manager.backend {
        GpuBackend::Cuda => generate_cuda_tensor_core_kernel(data_type, tile_size, m, n, k, use_mixed_precision),
        GpuBackend::Rocm => generate_rocm_matrix_core_kernel(data_type, tile_size, m, n, k, use_mixed_precision),
        GpuBackend::Metal => generate_metal_mps_kernel(data_type, tile_size, m, n, k),
        _ => Err(TensorCoreError::UnsupportedOperation(TensorCoreOp::MatrixMultiply)),
    }
}

/// Generate CUDA tensor core kernel
fn generate_cuda_tensor_core_kernel(
    data_type: &TensorDataType,
    tile_size: (usize, usize),
    _m: usize,
    _n: usize,
    _k: usize,
    use_mixed_precision: bool,
) -> Result<String, TensorCoreError> {
    let (tile_m, tile_n) = tile_size;
    let dtype_str = match data_type {
        TensorDataType::Float16 => "__half",
        TensorDataType::BFloat16 => "__nv_bfloat16",
        TensorDataType::Float32 => "float",
        TensorDataType::Int8 => "int8_t",
        _ => return Err(TensorCoreError::UnsupportedDataType(data_type.clone())),
    };

    let accumulator_type = if use_mixed_precision { "float" } else { dtype_str };

    Ok(format!(r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

__global__ void tensor_core_gemm(
    const {dtype_str}* __restrict__ A,
    const {dtype_str}* __restrict__ B,
    {accumulator_type}* __restrict__ C,
    int M, int N, int K
) {{
    // Tensor core fragment declarations
    wmma::fragment<wmma::matrix_a, {tile_m}, {tile_n}, 16, {dtype_str}, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, {tile_m}, {tile_n}, 16, {dtype_str}, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, {tile_m}, {tile_n}, 16, {accumulator_type}> acc_frag;
    wmma::fragment<wmma::accumulator, {tile_m}, {tile_n}, 16, {accumulator_type}> c_frag;

    // Thread block and warp coordinates
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    // Bounds checking
    if (warp_row * {tile_m} >= M || warp_col * {tile_n} >= N) return;

    // Initialize accumulator
    wmma::fill_fragment(acc_frag, 0.0f);

    // Main computation loop
    for (int i = 0; i < K; i += 16) {{
        int a_row = warp_row * {tile_m};
        int a_col = i;
        int b_row = i;
        int b_col = warp_col * {tile_n};

        // Bounds checking for partial tiles
        if (a_col + 16 <= K && b_row + 16 <= K) {{
            // Load matrix fragments
            wmma::load_matrix_sync(a_frag, A + a_row * K + a_col, K);
            wmma::load_matrix_sync(b_frag, B + b_row * N + b_col, N);

            // Perform matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }}
    }}

    // Load existing C matrix values for accumulation
    int c_row = warp_row * {tile_m};
    int c_col = warp_col * {tile_n};
    
    if (c_row + {tile_m} <= M && c_col + {tile_n} <= N) {{
        wmma::load_matrix_sync(c_frag, C + c_row * N + c_col, N, wmma::mem_row_major);
        
        // Add to accumulator
        for (int i = 0; i < c_frag.num_elements; i++) {{
            c_frag.x[i] += acc_frag.x[i];
        }}

        // Store result
        wmma::store_matrix_sync(C + c_row * N + c_col, c_frag, N, wmma::mem_row_major);
    }}
}}
"#,
        dtype_str = dtype_str,
        accumulator_type = accumulator_type,
        tile_m = tile_m,
        tile_n = tile_n,
    ))
}

/// Generate ROCm matrix core kernel
fn generate_rocm_matrix_core_kernel(
    data_type: &TensorDataType,
    tile_size: (usize, usize),
    _m: usize,
    _n: usize,
    _k: usize,
    use_mixed_precision: bool,
) -> Result<String, TensorCoreError> {
    let (tile_m, tile_n) = tile_size;
    let dtype_str = match data_type {
        TensorDataType::Float16 => "_Float16",
        TensorDataType::BFloat16 => "__bf16",
        TensorDataType::Float32 => "float",
        TensorDataType::Int8 => "int8_t",
        _ => return Err(TensorCoreError::UnsupportedDataType(data_type.clone())),
    };

    let accumulator_type = if use_mixed_precision { "float" } else { dtype_str };

    Ok(format!(r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void matrix_core_gemm(
    const {dtype_str}* __restrict__ A,
    const {dtype_str}* __restrict__ B,
    {accumulator_type}* __restrict__ C,
    int M, int N, int K
) {{
    // AMD MFMA intrinsics for matrix core operations
    const int tid = threadIdx.x;
    const int warp_id = tid / 64;  // AMD wavefront size is 64
    const int lane_id = tid % 64;
    
    const int block_row = blockIdx.y * {tile_m};
    const int block_col = blockIdx.x * {tile_n};
    
    // Bounds checking
    if (block_row >= M || block_col >= N) return;
    
    // Shared memory for tile loading
    __shared__ {dtype_str} A_shared[{tile_m} * 32];
    __shared__ {dtype_str} B_shared[32 * {tile_n}];
    
    {accumulator_type} accumulator[{acc_size}] = {{0}};
    
    // Main computation loop
    for (int k_block = 0; k_block < K; k_block += 32) {{
        // Cooperative loading to shared memory
        if (tid < {tile_m} * 32 / 64) {{
            int load_idx = tid * 64 + lane_id;
            if (load_idx < {tile_m} * 32 && (block_row + load_idx / 32) < M && (k_block + load_idx % 32) < K) {{
                A_shared[load_idx] = A[(block_row + load_idx / 32) * K + (k_block + load_idx % 32)];
            }}
        }}
        
        if (tid < 32 * {tile_n} / 64) {{
            int load_idx = tid * 64 + lane_id;
            if (load_idx < 32 * {tile_n} && (k_block + load_idx / {tile_n}) < K && (block_col + load_idx % {tile_n}) < N) {{
                B_shared[load_idx] = B[(k_block + load_idx / {tile_n}) * N + (block_col + load_idx % {tile_n})];
            }}
        }}
        
        __syncthreads();
        
        // MFMA matrix multiplication (simplified)
        // In practice, would use __builtin_amdgcn_mfma_* intrinsics
        for (int i = 0; i < {tile_m}; i += 4) {{
            for (int j = 0; j < {tile_n}; j += 4) {{
                for (int k_inner = 0; k_inner < 32; k_inner++) {{
                    accumulator[(i * {tile_n} + j) / 16] += 
                        A_shared[i * 32 + k_inner] * B_shared[k_inner * {tile_n} + j];
                }}
            }}
        }}
        
        __syncthreads();
    }}
    
    // Store results
    for (int i = 0; i < {tile_m}; i++) {{
        for (int j = 0; j < {tile_n}; j++) {{
            int global_row = block_row + i;
            int global_col = block_col + j;
            if (global_row < M && global_col < N) {{
                C[global_row * N + global_col] += accumulator[(i * {tile_n} + j) / 16];
            }}
        }}
    }}
}}
"#,
        dtype_str = dtype_str,
        accumulator_type = accumulator_type,
        tile_m = tile_m,
        tile_n = tile_n,
        acc_size = (tile_m * tile_n) / 16,
    ))
}

/// Generate Metal Performance Shaders kernel
fn generate_metal_mps_kernel(
    data_type: &TensorDataType,
    tile_size: (usize, usize),
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<String, TensorCoreError> {
    let (tile_m, tile_n) = tile_size;
    let dtype_str = match data_type {
        TensorDataType::Float16 => "half",
        TensorDataType::Float32 => "float",
        TensorDataType::Int8 => "char",
        _ => return Err(TensorCoreError::UnsupportedDataType(data_type.clone())),
    };

    Ok(format!(r#"
#include <metal_stdlib>
using namespace metal;

kernel void neural_engine_gemm(
    device const {dtype_str}* A [[buffer(0)]],
    device const {dtype_str}* B [[buffer(1)]],
    device {dtype_str}* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {{
    const uint row = gid.y * {tile_m} + tid.y;
    const uint col = gid.x * {tile_n} + tid.x;
    
    if (row >= M || col >= N) return;
    
    {dtype_str} sum = 0.0;
    
    // Use SIMD group matrix operations when available
    for (uint k = 0; k < K; k++) {{
        sum += A[row * K + k] * B[k * N + col];
    }}
    
    C[row * N + col] = sum;
}}
"#,
        dtype_str = dtype_str,
        tile_m = tile_m,
        tile_n = tile_n,
    ))
}

/// Execute tensor core operation
fn execute_tensor_core_operation<T>(
    manager: &TensorCoreManager,
    kernel_source: &str,
    _a: &GpuBuffer<T>,
    _b: &GpuBuffer<T>,
    _c: &mut GpuBuffer<T>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), TensorCoreError>
where
    T: crate::gpu::GpuDataType,
{
    // In a real implementation, this would:
    // 1. Compile the kernel source for the target backend
    // 2. Set kernel arguments (buffers A, B, C and dimensions)
    // 3. Calculate optimal grid and block dimensions
    // 4. Launch the kernel with tensor core support
    // 5. Synchronize and check for errors

    let tile_size = manager.config().tile_size;
    let grid_dim_x = n.div_ceil(tile_size.1);
    let grid_dim_y = m.div_ceil(tile_size.0);
    
    eprintln!("Executing tensor core GEMM:");
    eprintln!("  Kernel source length: {} characters", kernel_source.len());
    eprintln!("  Dimensions: {}x{}x{}", m, n, k);
    eprintln!("  Grid dimensions: {}x{}", grid_dim_x, grid_dim_y);
    eprintln!("  Tile size: {:?}", tile_size);
    eprintln!("  Backend: {:?}", manager.backend);
    eprintln!("  Data type: {}", manager.config().data_type);
    
    // Placeholder for actual kernel execution
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data_type_display() {
        assert_eq!(TensorDataType::Float16.to_string(), "f16");
        assert_eq!(TensorDataType::BFloat16.to_string(), "bf16");
        assert_eq!(TensorDataType::Int8.to_string(), "i8");
    }

    #[test]
    fn test_nvidia_capabilities() {
        let caps = TensorCoreManager::nvidia_tensor_capabilities();
        assert!(caps.available);
        assert!(caps.supported_types.contains(&TensorDataType::Float16));
        assert!(caps.supported_ops.contains(&TensorCoreOp::MatrixMultiply));
    }

    #[test]
    fn test_amd_capabilities() {
        let caps = TensorCoreManager::amd_matrix_capabilities();
        assert!(caps.available);
        assert!(caps.supported_types.contains(&TensorDataType::BFloat16));
        assert!(caps.supported_ops.contains(&TensorCoreOp::MatrixMultiply));
    }

    #[test]
    fn test_optimal_config() {
        let caps = TensorCoreManager::nvidia_tensor_capabilities();
        let config = TensorCoreManager::optimal_config(&caps);
        assert_eq!(config.data_type, TensorDataType::BFloat16);
        assert!(config.use_mixed_precision);
    }

    #[test]
    fn test_dimension_optimization() {
        // This would require a real GPU context, so we'll test the logic only
        let caps = TensorCoreManager::nvidia_tensor_capabilities();
        let config = TensorCoreManager::optimal_config(&caps);

        // Test that we can create a config
        assert!(config.auto_convert);
        assert_eq!(config.tile_size, (16, 16));
    }
}
