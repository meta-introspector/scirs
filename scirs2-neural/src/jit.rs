//! Just-In-Time (JIT) compilation system for neural networks
//!
//! This module provides JIT compilation capabilities for optimizing neural network
//! operations at runtime. It includes:
//! - Dynamic code generation for optimized kernels
//! - Operation fusion for reducing memory overhead
//! - Platform-specific optimizations
//! - SIMD and vectorization hints
//! - Cache-friendly memory access patterns

use crate::error::{NeuralError, Result};
use ndarray::{Array, ArrayD, Dimension};
use num_traits::Float;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

/// JIT compilation target architecture
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum TargetArchitecture {
    /// x86-64 with specific feature sets
    X86_64 {
        /// AVX support level (0, 1, 2, 512)
        avx_level: u8,
        /// FMA support
        fma: bool,
        /// BMI support
        bmi: bool,
    },
    /// ARM64 with NEON support
    ARM64 {
        /// NEON support
        neon: bool,
        /// SVE support
        sve: bool,
    },
    /// RISC-V with vector extensions
    RISCV {
        /// Vector extension support
        vector: bool,
    },
    /// GPU targets
    GPU {
        /// GPU architecture type
        arch: GPUArchitecture,
    },
    /// WebAssembly target
    WASM {
        /// SIMD support
        simd: bool,
        /// Threading support
        threads: bool,
    },
    /// Generic fallback
    Generic,
}

/// GPU architecture types
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum GPUArchitecture {
    /// NVIDIA CUDA
    CUDA {
        /// Compute capability
        compute_capability: (u8, u8),
    },
    /// AMD ROCm
    ROCm {
        /// GFX version
        gfx_version: String,
    },
    /// Intel GPU
    Intel {
        /// Generation
        generation: u8,
    },
    /// Apple Metal
    Metal {
        /// Family
        family: u8,
    },
    /// Vulkan compute
    Vulkan,
    /// OpenCL
    OpenCL {
        /// Version
        version: String,
    },
}

/// JIT operation types that can be compiled
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum JITOperation {
    /// Matrix multiplication with specific dimensions
    MatMul {
        /// Input A dimensions
        a_shape: Vec<usize>,
        /// Input B dimensions
        b_shape: Vec<usize>,
        /// Transpose flags
        transpose_a: bool,
        transpose_b: bool,
    },
    /// Element-wise operations
    ElementWise {
        /// Operation type
        op: ElementWiseOp,
        /// Input shapes
        shapes: Vec<Vec<usize>>,
    },
    /// Convolution operations
    Convolution {
        /// Input shape [N, C, H, W]
        input_shape: Vec<usize>,
        /// Weight shape [out_channels, in_channels, kH, kW]
        weight_shape: Vec<usize>,
        /// Stride
        stride: Vec<usize>,
        /// Padding
        padding: Vec<usize>,
        /// Dilation
        dilation: Vec<usize>,
    },
    /// Pooling operations
    Pooling {
        /// Pooling type
        pool_type: PoolingType,
        /// Input shape
        input_shape: Vec<usize>,
        /// Kernel size
        kernel_size: Vec<usize>,
        /// Stride
        stride: Vec<usize>,
        /// Padding
        padding: Vec<usize>,
    },
    /// Normalization operations
    Normalization {
        /// Normalization type
        norm_type: NormalizationType,
        /// Input shape
        input_shape: Vec<usize>,
        /// Axes to normalize over
        axes: Vec<usize>,
    },
    /// Activation functions
    Activation {
        /// Activation type
        activation: ActivationType,
        /// Input shape
        input_shape: Vec<usize>,
    },
    /// Reduction operations
    Reduction {
        /// Reduction type
        reduction: ReductionType,
        /// Input shape
        input_shape: Vec<usize>,
        /// Reduction axes
        axes: Vec<usize>,
        /// Keep dimensions
        keep_dims: bool,
    },
    /// Custom fused operations
    FusedOp {
        /// Sequence of operations to fuse
        operations: Vec<Box<JITOperation>>,
        /// Fusion strategy
        fusion_strategy: FusionStrategy,
    },
}

/// Element-wise operation types
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum ElementWiseOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
    Equal,
    Greater,
    Less,
    Abs,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    /// Custom operation with code
    Custom(String),
}

/// Pooling operation types
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum PoolingType {
    Max,
    Average,
    Global,
    Adaptive,
}

/// Normalization types
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum NormalizationType {
    BatchNorm,
    LayerNorm,
    InstanceNorm,
    GroupNorm,
}

/// Activation function types
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    GELU,
    Swish,
    Mish,
    LeakyReLU(f64),
    ELU(f64),
    SELU,
}

/// Reduction operation types
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum ReductionType {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
    Std,
    Var,
}

/// Fusion strategies for combining operations
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum FusionStrategy {
    /// Vertical fusion (producer-consumer)
    Vertical,
    /// Horizontal fusion (independent operations)
    Horizontal,
    /// Loop fusion
    Loop,
    /// Memory fusion (share memory buffers)
    Memory,
}

/// JIT compilation context and cache
pub struct JITCompiler {
    /// Target architecture for compilation
    target_arch: TargetArchitecture,
    /// Compiled kernel cache
    kernel_cache: Arc<RwLock<HashMap<JITKernelKey, CompiledKernel>>>,
    /// Optimization settings
    optimization_level: OptimizationLevel,
    /// Code generation settings
    codegen_settings: CodeGenSettings,
    /// Runtime statistics
    stats: Arc<RwLock<JITStatistics>>,
    /// Operation fusion optimizer
    fusion_optimizer: FusionOptimizer,
}

/// Key for identifying compiled kernels
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JITKernelKey {
    /// Operation description
    operation: JITOperation,
    /// Target architecture
    target: TargetArchitecture,
    /// Optimization level
    opt_level: OptimizationLevel,
}

/// Compiled kernel with metadata
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    /// Generated code (platform-specific)
    code: String,
    /// Entry point function name
    entry_point: String,
    /// Memory requirements
    memory_requirements: MemoryRequirements,
    /// Performance characteristics
    performance_hints: PerformanceHints,
    /// Compilation timestamp
    timestamp: std::time::Instant,
    /// Usage count
    usage_count: Arc<RwLock<u64>>,
}

/// Memory requirements for a kernel
#[derive(Debug, Clone)]
pub struct MemoryRequirements {
    /// Minimum required memory (bytes)
    min_memory: usize,
    /// Optimal memory for performance (bytes)
    optimal_memory: usize,
    /// Memory access pattern
    access_pattern: MemoryAccessPattern,
    /// Alignment requirements
    alignment: usize,
}

/// Memory access patterns
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryAccessPattern {
    Sequential,
    Random,
    Strided,
    Blocked,
}

/// Performance hints for kernel execution
#[derive(Debug, Clone)]
pub struct PerformanceHints {
    /// Estimated FLOPS
    estimated_flops: u64,
    /// Memory bandwidth utilization (0-1)
    memory_bandwidth_util: f64,
    /// Compute intensity (FLOPS per byte)
    compute_intensity: f64,
    /// Vectorization factor
    vectorization_factor: u8,
    /// Parallelization level
    parallelization_level: u8,
}

/// JIT compilation optimization levels
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OptimizationLevel {
    /// No optimization, fast compilation
    O0,
    /// Basic optimizations
    O1,
    /// Standard optimizations
    O2,
    /// Aggressive optimizations
    O3,
    /// Size optimization
    Os,
    /// Custom optimization with specific flags
    Custom(Vec<String>),
}

/// Code generation settings
#[derive(Debug, Clone)]
pub struct CodeGenSettings {
    /// Enable vectorization
    pub vectorize: bool,
    /// Unroll loops
    pub unroll_loops: bool,
    /// Use lookup tables for functions
    pub use_lookup_tables: bool,
    /// Inline functions aggressively
    pub aggressive_inlining: bool,
    /// Use platform-specific intrinsics
    pub use_intrinsics: bool,
    /// Generate debug information
    pub debug_info: bool,
    /// Target specific features
    pub target_features: HashSet<String>,
}

/// JIT compilation and execution statistics
#[derive(Debug, Clone, Default)]
pub struct JITStatistics {
    /// Number of kernels compiled
    pub kernels_compiled: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Total compilation time (ms)
    pub total_compile_time_ms: f64,
    /// Total execution time (ms)
    pub total_execution_time_ms: f64,
    /// Average compilation time per kernel (ms)
    pub avg_compile_time_ms: f64,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Most frequently used operations
    pub popular_operations: HashMap<String, u64>,
}

/// Operation fusion optimizer
pub struct FusionOptimizer {
    /// Fusion rules database
    fusion_rules: HashMap<(String, String), FusionRule>,
    /// Maximum fusion depth
    max_fusion_depth: usize,
    /// Memory threshold for fusion
    memory_threshold: usize,
}

/// Fusion rule for combining operations
#[derive(Debug, Clone)]
pub struct FusionRule {
    /// Can these operations be fused?
    pub can_fuse: bool,
    /// Expected performance improvement (ratio)
    pub performance_gain: f64,
    /// Memory savings (bytes)
    pub memory_savings: usize,
    /// Fusion strategy to use
    pub strategy: FusionStrategy,
}

impl JITCompiler {
    /// Create a new JIT compiler for the target architecture
    pub fn new(target_arch: TargetArchitecture) -> Self {
        let codegen_settings = CodeGenSettings {
            vectorize: true,
            unroll_loops: true,
            use_lookup_tables: false,
            aggressive_inlining: true,
            use_intrinsics: true,
            debug_info: false,
            target_features: HashSet::new(),
        };

        let fusion_optimizer = FusionOptimizer::new();

        Self {
            target_arch,
            kernel_cache: Arc::new(RwLock::new(HashMap::new())),
            optimization_level: OptimizationLevel::O2,
            codegen_settings,
            stats: Arc::new(RwLock::new(JITStatistics::default())),
            fusion_optimizer,
        }
    }

    /// Detect the best target architecture for the current system
    pub fn detect_target_architecture() -> TargetArchitecture {
        #[cfg(target_arch = "x86_64")]
        {
            TargetArchitecture::X86_64 {
                avx_level: detect_avx_level(),
                fma: is_x86_feature_detected!("fma"),
                bmi: is_x86_feature_detected!("bmi1"),
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            TargetArchitecture::ARM64 {
                neon: true, // NEON is standard on ARM64
                sve: false, // SVE detection would require runtime checks
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            TargetArchitecture::WASM {
                simd: true,     // Assume SIMD support in modern browsers
                threads: false, // Threading support varies
            }
        }
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )))]
        {
            TargetArchitecture::Generic
        }
    }

    /// Compile a JIT operation to optimized code
    pub fn compile_operation(&self, operation: &JITOperation) -> Result<CompiledKernel> {
        let start_time = std::time::Instant::now();

        // Create kernel key
        let key = JITKernelKey {
            operation: operation.clone(),
            target: self.target_arch.clone(),
            opt_level: self.optimization_level.clone(),
        };

        // Check cache first
        if let Some(cached_kernel) = self.get_cached_kernel(&key) {
            self.update_cache_stats(true);
            return Ok(cached_kernel);
        }

        // Apply fusion optimization if applicable
        let optimized_operation = self.fusion_optimizer.optimize_operation(operation)?;

        // Generate code for the operation
        let code = self.generate_code(&optimized_operation)?;
        let entry_point = format!("kernel_{}", self.generate_kernel_id(&key));

        // Analyze memory requirements
        let memory_requirements = self.analyze_memory_requirements(&optimized_operation)?;

        // Estimate performance characteristics
        let performance_hints = self.estimate_performance(&optimized_operation)?;

        let kernel = CompiledKernel {
            code,
            entry_point,
            memory_requirements,
            performance_hints,
            timestamp: std::time::Instant::now(),
            usage_count: Arc::new(RwLock::new(0)),
        };

        // Cache the compiled kernel
        self.cache_kernel(key, kernel.clone());

        // Update statistics
        let compile_time = start_time.elapsed().as_millis() as f64;
        self.update_compile_stats(compile_time);
        self.update_cache_stats(false);

        Ok(kernel)
    }

    /// Execute a compiled kernel with given inputs
    pub fn execute_kernel<F: Float + Debug>(
        &self,
        kernel: &CompiledKernel,
        inputs: &[&ArrayD<F>],
        output_shape: &[usize],
    ) -> Result<ArrayD<F>> {
        let start_time = std::time::Instant::now();

        // Validate inputs
        self.validate_kernel_inputs(kernel, inputs)?;

        // For now, we'll provide a placeholder execution
        // In a real implementation, this would call the compiled code
        let output = self.execute_kernel_placeholder(inputs, output_shape)?;

        // Update usage statistics
        if let Ok(mut count) = kernel.usage_count.write() {
            *count += 1;
        }

        let execution_time = start_time.elapsed().as_millis() as f64;
        self.update_execution_stats(execution_time);

        Ok(output)
    }

    /// Compile and execute an operation in one step
    pub fn compile_and_execute<F: Float + Debug>(
        &self,
        operation: &JITOperation,
        inputs: &[&ArrayD<F>],
        output_shape: &[usize],
    ) -> Result<ArrayD<F>> {
        let kernel = self.compile_operation(operation)?;
        self.execute_kernel(&kernel, inputs, output_shape)
    }

    /// Get cached kernel if available
    fn get_cached_kernel(&self, key: &JITKernelKey) -> Option<CompiledKernel> {
        if let Ok(cache) = self.kernel_cache.read() {
            cache.get(key).cloned()
        } else {
            None
        }
    }

    /// Cache a compiled kernel
    fn cache_kernel(&self, key: JITKernelKey, kernel: CompiledKernel) {
        if let Ok(mut cache) = self.kernel_cache.write() {
            cache.insert(key, kernel);
        }
    }

    /// Generate unique kernel ID
    fn generate_kernel_id(&self, key: &JITKernelKey) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Generate optimized code for an operation
    fn generate_code(&self, operation: &JITOperation) -> Result<String> {
        match operation {
            JITOperation::MatMul {
                a_shape,
                b_shape,
                transpose_a,
                transpose_b,
            } => self.generate_matmul_code(a_shape, b_shape, *transpose_a, *transpose_b),
            JITOperation::ElementWise { op, shapes } => self.generate_elementwise_code(op, shapes),
            JITOperation::Convolution {
                input_shape,
                weight_shape,
                stride,
                padding,
                dilation,
            } => {
                self.generate_convolution_code(input_shape, weight_shape, stride, padding, dilation)
            }
            JITOperation::Activation {
                activation,
                input_shape,
            } => self.generate_activation_code(activation, input_shape),
            JITOperation::FusedOp {
                operations,
                fusion_strategy,
            } => self.generate_fused_code(operations, fusion_strategy),
            _ => {
                // For other operations, generate generic code
                Ok(self.generate_generic_code(operation))
            }
        }
    }

    /// Generate matrix multiplication code
    fn generate_matmul_code(
        &self,
        a_shape: &[usize],
        b_shape: &[usize],
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<String> {
        let m = if transpose_a { a_shape[1] } else { a_shape[0] };
        let k = if transpose_a { a_shape[0] } else { a_shape[1] };
        let n = if transpose_b { b_shape[0] } else { b_shape[1] };

        let mut code = String::new();
        code.push_str(&format!("// Optimized MatMul: {}x{} * {}x{}\n", m, k, k, n));
        code.push_str("void kernel_matmul(const float* A, const float* B, float* C) {\n");

        if self.codegen_settings.vectorize && self.target_arch_supports_simd() {
            // Generate vectorized code
            code.push_str(&self.generate_vectorized_matmul(m, k, n)?);
        } else {
            // Generate scalar code
            code.push_str(&self.generate_scalar_matmul(m, k, n, transpose_a, transpose_b));
        }

        code.push_str("}\n");
        Ok(code)
    }

    /// Generate element-wise operation code
    fn generate_elementwise_code(
        &self,
        op: &ElementWiseOp,
        shapes: &[Vec<usize>],
    ) -> Result<String> {
        let mut code = String::new();
        let total_elements = shapes[0].iter().product::<usize>();

        code.push_str(&format!("// Element-wise operation: {:?}\n", op));
        code.push_str("void kernel_elementwise(");

        // Generate input parameters
        for i in 0..shapes.len() {
            code.push_str(&format!("const float* input{}, ", i));
        }
        code.push_str("float* output) {\n");

        if self.codegen_settings.vectorize && self.target_arch_supports_simd() {
            code.push_str(&self.generate_vectorized_elementwise(op, total_elements)?);
        } else {
            code.push_str(&self.generate_scalar_elementwise(op, total_elements));
        }

        code.push_str("}\n");
        Ok(code)
    }

    /// Generate convolution code
    fn generate_convolution_code(
        &self,
        input_shape: &[usize],
        weight_shape: &[usize],
        stride: &[usize],
        padding: &[usize],
        _dilation: &[usize],
    ) -> Result<String> {
        let mut code = String::new();
        code.push_str("// Optimized Convolution\n");
        code.push_str(
            "void kernel_conv2d(const float* input, const float* weight, float* output) {\n",
        );

        // Generate convolution loops with optimizations
        let n = input_shape[0]; // batch size
        let c_in = input_shape[1]; // input channels
        let h_in = input_shape[2]; // input height
        let w_in = input_shape[3]; // input width

        let c_out = weight_shape[0]; // output channels
        let kh = weight_shape[2]; // kernel height
        let kw = weight_shape[3]; // kernel width

        let h_out = (h_in + 2 * padding[0] - kh) / stride[0] + 1;
        let w_out = (w_in + 2 * padding[1] - kw) / stride[1] + 1;

        if self.codegen_settings.unroll_loops && kh <= 3 && kw <= 3 {
            code.push_str(&self.generate_unrolled_conv(n, c_in, c_out, h_out, w_out, kh, kw));
        } else {
            code.push_str(&self.generate_standard_conv(
                n, c_in, c_out, h_in, w_in, h_out, w_out, kh, kw, stride, padding,
            ));
        }

        code.push_str("}\n");
        Ok(code)
    }

    /// Generate activation function code
    fn generate_activation_code(
        &self,
        activation: &ActivationType,
        input_shape: &[usize],
    ) -> Result<String> {
        let total_elements = input_shape.iter().product::<usize>();
        let mut code = String::new();

        code.push_str(&format!("// Activation: {:?}\n", activation));
        code.push_str("void kernel_activation(const float* input, float* output) {\n");

        if self.codegen_settings.vectorize && self.target_arch_supports_simd() {
            code.push_str(&self.generate_vectorized_activation(activation, total_elements)?);
        } else {
            code.push_str(&self.generate_scalar_activation(activation, total_elements));
        }

        code.push_str("}\n");
        Ok(code)
    }

    /// Generate fused operation code
    fn generate_fused_code(
        &self,
        operations: &[Box<JITOperation>],
        strategy: &FusionStrategy,
    ) -> Result<String> {
        let mut code = String::new();

        code.push_str(&format!(
            "// Fused operations with strategy: {:?}\n",
            strategy
        ));
        code.push_str("void kernel_fused(");

        match strategy {
            FusionStrategy::Vertical => {
                // Generate code that combines operations in sequence
                code.push_str("const float* input, float* output) {\n");
                code.push_str("  // Vertical fusion - pipeline operations\n");
                for (i, op) in operations.iter().enumerate() {
                    code.push_str(&format!("  // Operation {}: {:?}\n", i, op));
                }
            }
            FusionStrategy::Horizontal => {
                // Generate code that combines independent operations
                code.push_str("const float* input, float* output) {\n");
                code.push_str("  // Horizontal fusion - parallel operations\n");
                for (i, op) in operations.iter().enumerate() {
                    code.push_str(&format!("  // Parallel operation {}: {:?}\n", i, op));
                }
            }
            _ => {
                code.push_str("const float* input, float* output) {\n");
                code.push_str("  // Generic fusion\n");
            }
        }

        code.push_str("}\n");
        Ok(code)
    }

    /// Generate generic fallback code
    fn generate_generic_code(&self, operation: &JITOperation) -> String {
        format!("// Generic implementation for: {:?}\nvoid kernel_generic() {{\n  // Fallback implementation\n}}\n", operation)
    }

    /// Check if target architecture supports SIMD
    fn target_arch_supports_simd(&self) -> bool {
        match &self.target_arch {
            TargetArchitecture::X86_64 { avx_level, .. } => *avx_level > 0,
            TargetArchitecture::ARM64 { neon, .. } => *neon,
            TargetArchitecture::WASM { simd, .. } => *simd,
            _ => false,
        }
    }

    /// Generate vectorized matrix multiplication code
    fn generate_vectorized_matmul(&self, m: usize, k: usize, n: usize) -> Result<String> {
        let mut code = String::new();

        match &self.target_arch {
            TargetArchitecture::X86_64 { avx_level, .. } => {
                if *avx_level >= 2 {
                    code.push_str(&format!("  // AVX2 vectorized matmul {}x{}x{}\n", m, k, n));
                    code.push_str("  #pragma omp parallel for\n");
                    code.push_str("  for (int i = 0; i < m; i += 8) {\n");
                    code.push_str("    for (int j = 0; j < n; j += 8) {\n");
                    code.push_str("      __m256 sum = _mm256_setzero_ps();\n");
                    code.push_str("      for (int l = 0; l < k; l++) {\n");
                    code.push_str("        __m256 a_vec = _mm256_broadcast_ss(&A[i*k + l]);\n");
                    code.push_str("        __m256 b_vec = _mm256_loadu_ps(&B[l*n + j]);\n");
                    code.push_str("        sum = _mm256_fmadd_ps(a_vec, b_vec, sum);\n");
                    code.push_str("      }\n");
                    code.push_str("      _mm256_storeu_ps(&C[i*n + j], sum);\n");
                    code.push_str("    }\n");
                    code.push_str("  }\n");
                } else {
                    code.push_str("  // SSE vectorized matmul\n");
                    code.push_str(&self.generate_sse_matmul(m, k, n));
                }
            }
            TargetArchitecture::ARM64 { .. } => {
                code.push_str("  // NEON vectorized matmul\n");
                code.push_str(&self.generate_neon_matmul(m, k, n));
            }
            _ => {
                return Err(NeuralError::ComputationError(
                    "Vectorization not supported for this architecture".to_string(),
                ));
            }
        }

        Ok(code)
    }

    /// Generate scalar matrix multiplication code
    fn generate_scalar_matmul(
        &self,
        m: usize,
        k: usize,
        n: usize,
        transpose_a: bool,
        transpose_b: bool,
    ) -> String {
        let mut code = String::new();

        code.push_str(&format!("  // Scalar matmul {}x{}x{}\n", m, k, n));
        code.push_str("  #pragma omp parallel for\n");
        code.push_str("  for (int i = 0; i < m; i++) {\n");
        code.push_str("    for (int j = 0; j < n; j++) {\n");
        code.push_str("      float sum = 0.0f;\n");
        code.push_str("      for (int l = 0; l < k; l++) {\n");

        if transpose_a && transpose_b {
            code.push_str("        sum += A[l*m + i] * B[j*k + l];\n");
        } else if transpose_a {
            code.push_str("        sum += A[l*m + i] * B[l*n + j];\n");
        } else if transpose_b {
            code.push_str("        sum += A[i*k + l] * B[j*k + l];\n");
        } else {
            code.push_str("        sum += A[i*k + l] * B[l*n + j];\n");
        }

        code.push_str("      }\n");
        code.push_str("      C[i*n + j] = sum;\n");
        code.push_str("    }\n");
        code.push_str("  }\n");

        code
    }

    /// Generate SSE matrix multiplication code
    fn generate_sse_matmul(&self, _m: usize, _k: usize, _n: usize) -> String {
        String::from("  // SSE implementation placeholder\n")
    }

    /// Generate NEON matrix multiplication code
    fn generate_neon_matmul(&self, _m: usize, _k: usize, _n: usize) -> String {
        String::from("  // NEON implementation placeholder\n")
    }

    /// Generate vectorized element-wise code
    fn generate_vectorized_elementwise(
        &self,
        op: &ElementWiseOp,
        total_elements: usize,
    ) -> Result<String> {
        let mut code = String::new();

        code.push_str(&format!(
            "  // Vectorized element-wise operation, {} elements\n",
            total_elements
        ));
        code.push_str("  #pragma omp parallel for\n");
        code.push_str("  for (int i = 0; i < total_elements; i += 8) {\n");

        match op {
            ElementWiseOp::Add => {
                code.push_str("    __m256 a = _mm256_loadu_ps(&input0[i]);\n");
                code.push_str("    __m256 b = _mm256_loadu_ps(&input1[i]);\n");
                code.push_str("    __m256 result = _mm256_add_ps(a, b);\n");
                code.push_str("    _mm256_storeu_ps(&output[i], result);\n");
            }
            ElementWiseOp::Mul => {
                code.push_str("    __m256 a = _mm256_loadu_ps(&input0[i]);\n");
                code.push_str("    __m256 b = _mm256_loadu_ps(&input1[i]);\n");
                code.push_str("    __m256 result = _mm256_mul_ps(a, b);\n");
                code.push_str("    _mm256_storeu_ps(&output[i], result);\n");
            }
            ElementWiseOp::ReLU => {
                code.push_str("    __m256 input_vec = _mm256_loadu_ps(&input0[i]);\n");
                code.push_str("    __m256 zero = _mm256_setzero_ps();\n");
                code.push_str("    __m256 result = _mm256_max_ps(input_vec, zero);\n");
                code.push_str("    _mm256_storeu_ps(&output[i], result);\n");
            }
            _ => {
                code.push_str("    // Generic vectorized operation\n");
            }
        }

        code.push_str("  }\n");
        Ok(code)
    }

    /// Generate scalar element-wise code
    fn generate_scalar_elementwise(&self, op: &ElementWiseOp, total_elements: usize) -> String {
        let mut code = String::new();

        code.push_str(&format!(
            "  // Scalar element-wise operation, {} elements\n",
            total_elements
        ));
        code.push_str("  #pragma omp parallel for\n");
        code.push_str("  for (int i = 0; i < total_elements; i++) {\n");

        match op {
            ElementWiseOp::Add => code.push_str("    output[i] = input0[i] + input1[i];\n"),
            ElementWiseOp::Sub => code.push_str("    output[i] = input0[i] - input1[i];\n"),
            ElementWiseOp::Mul => code.push_str("    output[i] = input0[i] * input1[i];\n"),
            ElementWiseOp::Div => code.push_str("    output[i] = input0[i] / input1[i];\n"),
            ElementWiseOp::Max => code.push_str("    output[i] = fmaxf(input0[i], input1[i]);\n"),
            ElementWiseOp::Min => code.push_str("    output[i] = fminf(input0[i], input1[i]);\n"),
            ElementWiseOp::Abs => code.push_str("    output[i] = fabsf(input0[i]);\n"),
            ElementWiseOp::Sqrt => code.push_str("    output[i] = sqrtf(input0[i]);\n"),
            ElementWiseOp::Exp => code.push_str("    output[i] = expf(input0[i]);\n"),
            ElementWiseOp::Log => code.push_str("    output[i] = logf(input0[i]);\n"),
            ElementWiseOp::Sin => code.push_str("    output[i] = sinf(input0[i]);\n"),
            ElementWiseOp::Cos => code.push_str("    output[i] = cosf(input0[i]);\n"),
            ElementWiseOp::Tanh => code.push_str("    output[i] = tanhf(input0[i]);\n"),
            ElementWiseOp::Custom(expr) => {
                code.push_str(&format!(
                    "    output[i] = {};\n",
                    expr.replace("x", "input0[i]")
                ));
            }
            _ => code.push_str("    output[i] = input0[i]; // fallback\n"),
        }

        code.push_str("  }\n");
        code
    }

    /// Generate unrolled convolution code
    fn generate_unrolled_conv(
        &self,
        n: usize,
        c_in: usize,
        c_out: usize,
        h_out: usize,
        w_out: usize,
        kh: usize,
        kw: usize,
    ) -> String {
        let mut code = String::new();

        code.push_str(&format!("  // Unrolled {}x{} convolution\n", kh, kw));
        code.push_str("  #pragma omp parallel for\n");
        code.push_str(&format!("  for (int n = 0; n < {}; n++) {{\n", n));
        code.push_str(&format!(
            "    for (int c_out = 0; c_out < {}; c_out++) {{\n",
            c_out
        ));
        code.push_str(&format!("      for (int h = 0; h < {}; h++) {{\n", h_out));
        code.push_str(&format!("        for (int w = 0; w < {}; w++) {{\n", w_out));
        code.push_str("          float sum = 0.0f;\n");

        // Unroll the kernel loops
        for kh_i in 0..kh {
            for kw_i in 0..kw {
                code.push_str(&format!(
                    "          for (int c_in = 0; c_in < {}; c_in++) {{\n",
                    c_in
                ));
                code.push_str(&format!(
                    "            sum += input[((n*c_in + c_in)*h_in + h + {})*w_in + w + {}] * weight[((c_out*c_in + c_in)*{} + {})*{} + {}];\n",
                    kh_i, kw_i, kh, kh_i, kw, kw_i
                ));
                code.push_str("          }\n");
            }
        }

        code.push_str("          output[((n*c_out + c_out)*h_out + h)*w_out + w] = sum;\n");
        code.push_str("        }\n");
        code.push_str("      }\n");
        code.push_str("    }\n");
        code.push_str("  }\n");

        code
    }

    /// Generate standard convolution code
    fn generate_standard_conv(
        &self,
        n: usize,
        c_in: usize,
        c_out: usize,
        h_in: usize,
        w_in: usize,
        h_out: usize,
        w_out: usize,
        kh: usize,
        kw: usize,
        stride: &[usize],
        padding: &[usize],
    ) -> String {
        let mut code = String::new();

        code.push_str("  // Standard convolution loops\n");
        code.push_str("  #pragma omp parallel for\n");
        code.push_str(&format!("  for (int n = 0; n < {}; n++) {{\n", n));
        code.push_str(&format!(
            "    for (int c_out = 0; c_out < {}; c_out++) {{\n",
            c_out
        ));
        code.push_str(&format!("      for (int h = 0; h < {}; h++) {{\n", h_out));
        code.push_str(&format!("        for (int w = 0; w < {}; w++) {{\n", w_out));
        code.push_str("          float sum = 0.0f;\n");
        code.push_str(&format!(
            "          for (int c_in = 0; c_in < {}; c_in++) {{\n",
            c_in
        ));
        code.push_str(&format!(
            "            for (int kh = 0; kh < {}; kh++) {{\n",
            kh
        ));
        code.push_str(&format!(
            "              for (int kw = 0; kw < {}; kw++) {{\n",
            kw
        ));
        code.push_str(&format!(
            "                int h_in_idx = h * {} - {} + kh;\n",
            stride[0], padding[0]
        ));
        code.push_str(&format!(
            "                int w_in_idx = w * {} - {} + kw;\n",
            stride[1], padding[1]
        ));
        code.push_str(&format!("                if (h_in_idx >= 0 && h_in_idx < {} && w_in_idx >= 0 && w_in_idx < {}) {{\n", h_in, w_in));
        code.push_str("                  sum += input[((n*c_in + c_in)*h_in + h_in_idx)*w_in + w_in_idx] * weight[((c_out*c_in + c_in)*kh + kh)*kw + kw];\n");
        code.push_str("                }\n");
        code.push_str("              }\n");
        code.push_str("            }\n");
        code.push_str("          }\n");
        code.push_str("          output[((n*c_out + c_out)*h_out + h)*w_out + w] = sum;\n");
        code.push_str("        }\n");
        code.push_str("      }\n");
        code.push_str("    }\n");
        code.push_str("  }\n");

        code
    }

    /// Generate vectorized activation code
    fn generate_vectorized_activation(
        &self,
        activation: &ActivationType,
        total_elements: usize,
    ) -> Result<String> {
        let mut code = String::new();

        code.push_str(&format!("  // Vectorized activation: {:?}\n", activation));
        code.push_str("  #pragma omp parallel for\n");
        code.push_str("  for (int i = 0; i < total_elements; i += 8) {\n");
        code.push_str("    __m256 input_vec = _mm256_loadu_ps(&input[i]);\n");

        match activation {
            ActivationType::ReLU => {
                code.push_str("    __m256 zero = _mm256_setzero_ps();\n");
                code.push_str("    __m256 result = _mm256_max_ps(input_vec, zero);\n");
            }
            ActivationType::Sigmoid => {
                code.push_str("    // Sigmoid approximation\n");
                code.push_str("    __m256 one = _mm256_set1_ps(1.0f);\n");
                code.push_str(
                    "    __m256 neg_input = _mm256_sub_ps(_mm256_setzero_ps(), input_vec);\n",
                );
                code.push_str("    __m256 exp_neg = _mm256_exp_ps(neg_input);\n");
                code.push_str(
                    "    __m256 result = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));\n",
                );
            }
            ActivationType::Tanh => {
                code.push_str("    __m256 result = _mm256_tanh_ps(input_vec);\n");
            }
            _ => {
                code.push_str("    __m256 result = input_vec; // fallback\n");
            }
        }

        code.push_str("    _mm256_storeu_ps(&output[i], result);\n");
        code.push_str("  }\n");

        Ok(code)
    }

    /// Generate scalar activation code
    fn generate_scalar_activation(
        &self,
        activation: &ActivationType,
        total_elements: usize,
    ) -> String {
        let mut code = String::new();

        code.push_str(&format!("  // Scalar activation: {:?}\n", activation));
        code.push_str("  #pragma omp parallel for\n");
        code.push_str("  for (int i = 0; i < total_elements; i++) {\n");

        match activation {
            ActivationType::ReLU => {
                code.push_str("    output[i] = fmaxf(0.0f, input[i]);\n");
            }
            ActivationType::Sigmoid => {
                code.push_str("    output[i] = 1.0f / (1.0f + expf(-input[i]));\n");
            }
            ActivationType::Tanh => {
                code.push_str("    output[i] = tanhf(input[i]);\n");
            }
            ActivationType::GELU => {
                code.push_str("    float x = input[i];\n");
                code.push_str("    output[i] = 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));\n");
            }
            ActivationType::Swish => {
                code.push_str("    float x = input[i];\n");
                code.push_str("    output[i] = x / (1.0f + expf(-x));\n");
            }
            ActivationType::LeakyReLU(alpha) => {
                code.push_str(&format!(
                    "    output[i] = input[i] > 0.0f ? input[i] : {}f * input[i];\n",
                    alpha
                ));
            }
            _ => {
                code.push_str("    output[i] = input[i]; // fallback\n");
            }
        }

        code.push_str("  }\n");
        code
    }

    /// Analyze memory requirements for an operation
    fn analyze_memory_requirements(&self, operation: &JITOperation) -> Result<MemoryRequirements> {
        let (min_memory, optimal_memory, access_pattern) = match operation {
            JITOperation::MatMul {
                a_shape, b_shape, ..
            } => {
                let a_size = a_shape.iter().product::<usize>() * std::mem::size_of::<f32>();
                let b_size = b_shape.iter().product::<usize>() * std::mem::size_of::<f32>();
                let c_size = a_shape[0] * b_shape[1] * std::mem::size_of::<f32>();

                let min_mem = a_size + b_size + c_size;
                let optimal_mem = min_mem * 2; // For blocking optimizations

                (min_mem, optimal_mem, MemoryAccessPattern::Blocked)
            }
            JITOperation::ElementWise { shapes, .. } => {
                let total_size = shapes
                    .iter()
                    .map(|shape| shape.iter().product::<usize>() * std::mem::size_of::<f32>())
                    .sum::<usize>();

                (total_size, total_size, MemoryAccessPattern::Sequential)
            }
            JITOperation::Convolution {
                input_shape,
                weight_shape,
                ..
            } => {
                let input_size = input_shape.iter().product::<usize>() * std::mem::size_of::<f32>();
                let weight_size =
                    weight_shape.iter().product::<usize>() * std::mem::size_of::<f32>();

                // Output size calculation
                let n = input_shape[0];
                let c_out = weight_shape[0];
                let h_out = input_shape[2]; // Simplified
                let w_out = input_shape[3]; // Simplified
                let output_size = n * c_out * h_out * w_out * std::mem::size_of::<f32>();

                let min_mem = input_size + weight_size + output_size;
                let optimal_mem = min_mem + input_size; // For im2col buffer

                (min_mem, optimal_mem, MemoryAccessPattern::Strided)
            }
            _ => {
                // Generic estimation
                (1024, 4096, MemoryAccessPattern::Sequential)
            }
        };

        Ok(MemoryRequirements {
            min_memory,
            optimal_memory,
            access_pattern,
            alignment: 32, // 32-byte alignment for AVX
        })
    }

    /// Estimate performance characteristics
    fn estimate_performance(&self, operation: &JITOperation) -> Result<PerformanceHints> {
        let (flops, memory_bytes, vectorization_factor, parallelization_level) = match operation {
            JITOperation::MatMul {
                a_shape, b_shape, ..
            } => {
                let m = a_shape[0];
                let k = a_shape[1];
                let n = b_shape[1];
                let flops = 2 * m * k * n; // 2 operations per multiply-add
                let memory_bytes = (m * k + k * n + m * n) * std::mem::size_of::<usize>();

                (flops as u64, memory_bytes, 8, 4) // AVX can process 8 floats, good parallelization
            }
            JITOperation::ElementWise { shapes, .. } => {
                let elements = shapes[0].iter().product::<usize>();
                let flops = elements; // 1 operation per element
                let memory_bytes = elements * shapes.len() * std::mem::size_of::<usize>();

                (flops as u64, memory_bytes, 8, 8) // Highly parallel
            }
            JITOperation::Convolution {
                input_shape,
                weight_shape,
                ..
            } => {
                let n = input_shape[0];
                let c_in = input_shape[1];
                let h_in = input_shape[2];
                let w_in = input_shape[3];
                let c_out = weight_shape[0];
                let kh = weight_shape[2];
                let kw = weight_shape[3];

                let flops = n * c_out * h_in * w_in * c_in * kh * kw * 2; // Approximate
                let memory_bytes = (input_shape.iter().product::<usize>()
                    + weight_shape.iter().product::<usize>())
                    * std::mem::size_of::<usize>();

                (flops as u64, memory_bytes, 4, 4) // Moderate vectorization and parallelization
            }
            _ => {
                (1000, 1024, 1, 1) // Conservative estimates
            }
        };

        let compute_intensity = flops as f64 / memory_bytes as f64;
        let memory_bandwidth_util = (compute_intensity / 100.0).min(1.0); // Heuristic

        Ok(PerformanceHints {
            estimated_flops: flops,
            memory_bandwidth_util,
            compute_intensity,
            vectorization_factor,
            parallelization_level,
        })
    }

    /// Validate inputs for kernel execution
    fn validate_kernel_inputs<F: Float + Debug>(
        &self,
        _kernel: &CompiledKernel,
        inputs: &[&ArrayD<F>],
    ) -> Result<()> {
        if inputs.is_empty() {
            return Err(NeuralError::InvalidInput("No inputs provided".to_string()));
        }

        // Additional validation logic would go here
        Ok(())
    }

    /// Placeholder kernel execution (in real implementation, this would call compiled code)
    fn execute_kernel_placeholder<F: Float + Debug>(
        &self,
        inputs: &[&ArrayD<F>],
        output_shape: &[usize],
    ) -> Result<ArrayD<F>> {
        if inputs.is_empty() {
            return Err(NeuralError::InvalidInput("No inputs provided".to_string()));
        }

        // For now, return a zero array with the correct shape
        Ok(Array::zeros(output_shape).into_dyn())
    }

    /// Update cache statistics
    fn update_cache_stats(&self, cache_hit: bool) {
        if let Ok(mut stats) = self.stats.write() {
            let total_requests = stats.kernels_compiled + if cache_hit { 1 } else { 0 };
            if total_requests > 0 {
                let hits = if cache_hit {
                    (stats.cache_hit_rate * (total_requests - 1) as f64) + 1.0
                } else {
                    stats.cache_hit_rate * (total_requests - 1) as f64
                };
                stats.cache_hit_rate = hits / total_requests as f64;
            }
        }
    }

    /// Update compilation statistics
    fn update_compile_stats(&self, compile_time_ms: f64) {
        if let Ok(mut stats) = self.stats.write() {
            stats.kernels_compiled += 1;
            stats.total_compile_time_ms += compile_time_ms;
            stats.avg_compile_time_ms = stats.total_compile_time_ms / stats.kernels_compiled as f64;
        }
    }

    /// Update execution statistics
    fn update_execution_stats(&self, execution_time_ms: f64) {
        if let Ok(mut stats) = self.stats.write() {
            stats.total_execution_time_ms += execution_time_ms;
        }
    }

    /// Get compilation and execution statistics
    pub fn get_statistics(&self) -> JITStatistics {
        if let Ok(stats) = self.stats.read() {
            stats.clone()
        } else {
            JITStatistics::default()
        }
    }

    /// Clear the kernel cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.kernel_cache.write() {
            cache.clear();
        }
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        if let Ok(cache) = self.kernel_cache.read() {
            cache.len()
        } else {
            0
        }
    }

    /// Set optimization level
    pub fn set_optimization_level(&mut self, level: OptimizationLevel) {
        self.optimization_level = level;
    }

    /// Set code generation settings
    pub fn set_codegen_settings(&mut self, settings: CodeGenSettings) {
        self.codegen_settings = settings;
    }
}

impl FusionOptimizer {
    /// Create a new fusion optimizer
    pub fn new() -> Self {
        let mut fusion_rules = HashMap::new();

        // Add common fusion rules
        fusion_rules.insert(
            ("elementwise".to_string(), "elementwise".to_string()),
            FusionRule {
                can_fuse: true,
                performance_gain: 1.5,
                memory_savings: 0,
                strategy: FusionStrategy::Horizontal,
            },
        );

        fusion_rules.insert(
            ("activation".to_string(), "elementwise".to_string()),
            FusionRule {
                can_fuse: true,
                performance_gain: 1.3,
                memory_savings: 0,
                strategy: FusionStrategy::Vertical,
            },
        );

        Self {
            fusion_rules,
            max_fusion_depth: 4,
            memory_threshold: 1024 * 1024, // 1MB
        }
    }

    /// Optimize an operation by applying fusion rules
    pub fn optimize_operation(&self, operation: &JITOperation) -> Result<JITOperation> {
        // For now, return the operation as-is
        // In a full implementation, this would analyze the operation graph
        // and apply fusion optimizations
        Ok(operation.clone())
    }

    /// Check if two operations can be fused
    pub fn can_fuse(&self, op1: &JITOperation, op2: &JITOperation) -> bool {
        let key = (self.operation_type(op1), self.operation_type(op2));
        self.fusion_rules
            .get(&key)
            .map_or(false, |rule| rule.can_fuse)
    }

    /// Get operation type string for fusion rules
    fn operation_type(&self, operation: &JITOperation) -> String {
        match operation {
            JITOperation::MatMul { .. } => "matmul".to_string(),
            JITOperation::ElementWise { .. } => "elementwise".to_string(),
            JITOperation::Convolution { .. } => "convolution".to_string(),
            JITOperation::Activation { .. } => "activation".to_string(),
            JITOperation::Pooling { .. } => "pooling".to_string(),
            JITOperation::Normalization { .. } => "normalization".to_string(),
            JITOperation::Reduction { .. } => "reduction".to_string(),
            JITOperation::FusedOp { .. } => "fused".to_string(),
        }
    }
}

impl Default for FusionOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for CodeGenSettings {
    fn default() -> Self {
        Self {
            vectorize: true,
            unroll_loops: true,
            use_lookup_tables: false,
            aggressive_inlining: true,
            use_intrinsics: true,
            debug_info: false,
            target_features: HashSet::new(),
        }
    }
}

/// Detect AVX support level on x86_64
#[cfg(target_arch = "x86_64")]
fn detect_avx_level() -> u8 {
    if is_x86_feature_detected!("avx512f") {
        return 512;
    }
    if is_x86_feature_detected!("avx2") {
        return 2;
    }
    if is_x86_feature_detected!("avx") {
        return 1;
    }
    0
}

/// Fallback for non-x86_64 architectures
#[cfg(not(target_arch = "x86_64"))]
fn detect_avx_level() -> u8 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_compiler_creation() {
        let target_arch = JITCompiler::detect_target_architecture();
        let compiler = JITCompiler::new(target_arch);
        assert_eq!(compiler.cache_size(), 0);
    }

    #[test]
    fn test_matrix_multiplication_compilation() {
        let target_arch = TargetArchitecture::Generic;
        let compiler = JITCompiler::new(target_arch);

        let operation = JITOperation::MatMul {
            a_shape: vec![16, 32],
            b_shape: vec![32, 64],
            transpose_a: false,
            transpose_b: false,
        };

        let result = compiler.compile_operation(&operation);
        assert!(result.is_ok());

        let kernel = result.unwrap();
        assert!(kernel.code.contains("matmul"));
        assert!(kernel.entry_point.starts_with("kernel_"));
    }

    #[test]
    fn test_element_wise_compilation() {
        let target_arch = TargetArchitecture::Generic;
        let compiler = JITCompiler::new(target_arch);

        let operation = JITOperation::ElementWise {
            op: ElementWiseOp::Add,
            shapes: vec![vec![1024, 512], vec![1024, 512]],
        };

        let result = compiler.compile_operation(&operation);
        assert!(result.is_ok());

        let kernel = result.unwrap();
        assert!(kernel.code.contains("elementwise"));
    }

    #[test]
    fn test_convolution_compilation() {
        let target_arch = TargetArchitecture::Generic;
        let compiler = JITCompiler::new(target_arch);

        let operation = JITOperation::Convolution {
            input_shape: vec![1, 3, 224, 224],
            weight_shape: vec![64, 3, 7, 7],
            stride: vec![2, 2],
            padding: vec![3, 3],
            dilation: vec![1, 1],
        };

        let result = compiler.compile_operation(&operation);
        assert!(result.is_ok());

        let kernel = result.unwrap();
        assert!(kernel.code.contains("conv"));
    }

    #[test]
    fn test_activation_compilation() {
        let target_arch = TargetArchitecture::Generic;
        let compiler = JITCompiler::new(target_arch);

        let operation = JITOperation::Activation {
            activation: ActivationType::ReLU,
            input_shape: vec![32, 128, 56, 56],
        };

        let result = compiler.compile_operation(&operation);
        assert!(result.is_ok());

        let kernel = result.unwrap();
        assert!(kernel.code.contains("activation"));
    }

    #[test]
    fn test_cache_functionality() {
        let target_arch = TargetArchitecture::Generic;
        let compiler = JITCompiler::new(target_arch);

        let operation = JITOperation::ElementWise {
            op: ElementWiseOp::Mul,
            shapes: vec![vec![100, 100]],
        };

        // First compilation
        let result1 = compiler.compile_operation(&operation);
        assert!(result1.is_ok());
        assert_eq!(compiler.cache_size(), 1);

        // Second compilation should hit cache
        let result2 = compiler.compile_operation(&operation);
        assert!(result2.is_ok());
        assert_eq!(compiler.cache_size(), 1);
    }

    #[test]
    fn test_fusion_optimizer() {
        let optimizer = FusionOptimizer::new();

        let op1 = JITOperation::ElementWise {
            op: ElementWiseOp::Add,
            shapes: vec![vec![100, 100]],
        };

        let op2 = JITOperation::ElementWise {
            op: ElementWiseOp::Mul,
            shapes: vec![vec![100, 100]],
        };

        assert!(optimizer.can_fuse(&op1, &op2));
    }

    #[test]
    fn test_memory_requirements_analysis() {
        let target_arch = TargetArchitecture::Generic;
        let compiler = JITCompiler::new(target_arch);

        let operation = JITOperation::MatMul {
            a_shape: vec![100, 200],
            b_shape: vec![200, 300],
            transpose_a: false,
            transpose_b: false,
        };

        let requirements = compiler.analyze_memory_requirements(&operation);
        assert!(requirements.is_ok());

        let mem_req = requirements.unwrap();
        assert!(mem_req.min_memory > 0);
        assert!(mem_req.optimal_memory >= mem_req.min_memory);
    }

    #[test]
    fn test_performance_estimation() {
        let target_arch = TargetArchitecture::Generic;
        let compiler = JITCompiler::new(target_arch);

        let operation = JITOperation::MatMul {
            a_shape: vec![100, 200],
            b_shape: vec![200, 300],
            transpose_a: false,
            transpose_b: false,
        };

        let performance = compiler.estimate_performance(&operation);
        assert!(performance.is_ok());

        let perf_hints = performance.unwrap();
        assert!(perf_hints.estimated_flops > 0);
        assert!(perf_hints.compute_intensity >= 0.0);
    }

    #[test]
    fn test_target_architecture_detection() {
        let target_arch = JITCompiler::detect_target_architecture();

        // Should detect some valid architecture
        match target_arch {
            TargetArchitecture::X86_64 { .. }
            | TargetArchitecture::ARM64 { .. }
            | TargetArchitecture::WASM { .. }
            | TargetArchitecture::Generic => {
                // All valid
            }
            _ => panic!("Unknown architecture detected"),
        }
    }

    #[test]
    fn test_statistics_tracking() {
        let target_arch = TargetArchitecture::Generic;
        let compiler = JITCompiler::new(target_arch);

        let operation = JITOperation::ElementWise {
            op: ElementWiseOp::Add,
            shapes: vec![vec![10, 10]],
        };

        // Compile operation
        let _result = compiler.compile_operation(&operation);

        let stats = compiler.get_statistics();
        assert_eq!(stats.kernels_compiled, 1);
        assert!(stats.total_compile_time_ms >= 0.0);
    }

    #[test]
    fn test_code_generation_settings() {
        let target_arch = TargetArchitecture::Generic;
        let mut compiler = JITCompiler::new(target_arch);

        let mut settings = CodeGenSettings::default();
        settings.vectorize = false;
        settings.unroll_loops = false;

        compiler.set_codegen_settings(settings);

        let operation = JITOperation::ElementWise {
            op: ElementWiseOp::Add,
            shapes: vec![vec![100, 100]],
        };

        let result = compiler.compile_operation(&operation);
        assert!(result.is_ok());
    }

    #[test]
    fn test_optimization_levels() {
        let target_arch = TargetArchitecture::Generic;
        let mut compiler = JITCompiler::new(target_arch);

        // Test different optimization levels
        let levels = vec![
            OptimizationLevel::O0,
            OptimizationLevel::O1,
            OptimizationLevel::O2,
            OptimizationLevel::O3,
            OptimizationLevel::Os,
        ];

        let operation = JITOperation::ElementWise {
            op: ElementWiseOp::Mul,
            shapes: vec![vec![50, 50]],
        };

        for level in levels {
            compiler.set_optimization_level(level);
            let result = compiler.compile_operation(&operation);
            assert!(result.is_ok());
        }
    }
}
