# Migration Guide: Migrating Existing Implementations to Core Unified System

This guide explains the concrete steps to migrate existing SIMD/GPU implementations to the scirs2-core unified system.

## 🎯 Basic Migration Strategy

1. **Incremental Migration**: Don't change everything at once, migrate step by step
2. **Test-Driven**: Maintain existing tests while changing implementation
3. **Performance Verification**: Compare benchmarks before and after migration
4. **Backward Compatibility**: Maintain API as much as possible

## 📋 Preparation

### 1. Current State Analysis
```bash
# Identify SIMD usage in the module
rg "use (wide|packed_simd|std::arch)" --type rust
rg "f32x8|f64x4|__m256|__m128" --type rust

# Identify GPU/CUDA usage
rg "cuda|opencl|metal|gpu" --type rust -i

# Check benchmarks
cargo bench --no-run
```

### 2. Update Dependencies
```toml
# Cargo.toml
[dependencies]
# Add
scirs2-core = { workspace = true, features = ["simd", "parallel", "gpu"] }

# Remove (or move to dev-dependencies)
# wide = "0.7"
# packed_simd = "0.3"
```

## 🔧 SIMD Implementation Migration Steps

### Step 1: Change Import Statements

```rust
// Before
use wide::{f32x8, f64x4};
use std::arch::x86_64::*;

// After
use scirs2_core::simd_ops::{SimdUnifiedOps, PlatformCapabilities, AutoOptimizer};
```

### Step 2: Replace Basic Operations

#### Example 1: Vector Addition
```rust
// Before: Direct SIMD implementation
pub fn simd_add_vectors(a: &[f32], b: &[f32], result: &mut [f32]) {
    let chunks = a.len() / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let a_vec = f32x8::from_slice(&a[idx..]);
        let b_vec = f32x8::from_slice(&b[idx..]);
        let sum = a_vec + b_vec;
        sum.copy_to_slice(&mut result[idx..]);
    }
    // Handle remaining elements...
}

// After: Using Core SIMD ops
pub fn simd_add_vectors(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    f32::simd_add(a, b)
}
```

#### Example 2: Dot Product
```rust
// Before: Custom SIMD implementation
pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum_vec = f32x8::splat(0.0);
    let chunks = a.len() / 8;
    
    for i in 0..chunks {
        let idx = i * 8;
        let a_vec = f32x8::from_slice(&a[idx..]);
        let b_vec = f32x8::from_slice(&b[idx..]);
        sum_vec += a_vec * b_vec;
    }
    
    sum_vec.horizontal_sum() + /* remaining elements */
}

// After: Using Core SIMD ops
pub fn simd_dot_product(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    f32::simd_dot(a, b)
}
```

### Step 3: Migrate Complex Operations

#### Example: Matrix Operations
```rust
// Before: Block-based SIMD GEMM
pub fn simd_gemm(/* complex parameters */) {
    // 100+ lines of custom SIMD implementation
}

// After: Using Core SIMD ops
pub fn simd_gemm(
    alpha: f32,
    a: &ArrayView2<f32>,
    b: &ArrayView2<f32>,
    beta: f32,
    c: &mut Array2<f32>
) {
    f32::simd_gemm(alpha, a, b, beta, c);
}
```

### Step 4: Replace Platform Detection

```rust
// Before: Custom detection
#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

// After: Using Core detection system
fn select_implementation() {
    let caps = PlatformCapabilities::detect();
    
    if caps.simd_available {
        // Use SIMD implementation
    } else {
        // Use scalar implementation
    }
}
```

### Step 5: Leverage Auto-Optimization

```rust
// New feature: Automatic selection based on problem size
pub fn optimized_operation(data: &ArrayView1<f64>) -> f64 {
    let optimizer = AutoOptimizer::new();
    
    if optimizer.should_use_gpu(data.len()) {
        // GPU implementation (future)
        gpu_implementation(data)
    } else if optimizer.should_use_simd(data.len()) {
        // SIMD implementation
        f64::simd_sum(data) / data.len() as f64
    } else {
        // Scalar implementation
        data.mean().unwrap()
    }
}
```

## 🎮 GPU Implementation Migration Steps

### Step 1: Extract and Register Kernels

```rust
// Before: CUDA code inside module
mod cuda {
    pub fn launch_kernel() {
        // Direct CUDA API calls
    }
}

// After: Register kernel
use scirs2_core::gpu_registry::{register_module_kernel, KernelId, KernelSource};

fn register_my_kernels() {
    register_module_kernel(
        KernelId::new("mymodule", "operation", "f32"),
        KernelSource {
            source: include_str!("kernels/operation.cu"),
            backend: GpuBackend::Cuda,
            entry_point: "operation_f32",
            workgroup_size: (256, 1, 1),
            shared_memory: 0,
            uses_tensor_cores: false,
        },
    );
}
```

### Step 2: Change Kernel Usage

```rust
// Before: Direct GPU API usage
fn gpu_compute(data: &[f32]) -> Vec<f32> {
    let device = cuda::Device::new(0);
    let kernel = compile_kernel(KERNEL_SOURCE);
    // ...
}

// After: Via registry
use scirs2_core::gpu_registry::get_kernel;

fn gpu_compute(device: &GpuDevice, data: &Array1<f32>) -> Result<Array1<f32>> {
    let kernel_id = KernelId::new("mymodule", "operation", "f32");
    let kernel = get_kernel(&kernel_id, device)?;
    // ...
}
```

## 🧪 Test Migration

### Maintain Existing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_operation() {
        // Even if API changes, test expectations remain the same
        let a = array![1.0, 2.0, 3.0, 4.0];
        let b = array![5.0, 6.0, 7.0, 8.0];
        
        let result = simd_add_vectors(&a.view(), &b.view());
        
        assert_eq!(result, array![6.0, 8.0, 10.0, 12.0]);
    }
}
```

### Add Performance Tests

```rust
#[bench]
fn bench_compare_implementations(b: &mut Bencher) {
    let data = Array1::random(10000, Uniform::new(0.0, 1.0));
    
    b.iter(|| {
        f32::simd_sum(&data.view())
    });
}
```

## 📊 Migration Checklist

### Per-Module Checklist

- [ ] Update dependencies (Cargo.toml)
- [ ] Change import statements
- [ ] Replace SIMD operations
  - [ ] Basic operations (add, mul, etc.)
  - [ ] Reductions (sum, max, etc.)
  - [ ] Complex operations (GEMM, FFT, etc.)
- [ ] Replace platform detection
- [ ] Migrate GPU implementations (if applicable)
- [ ] Run and verify tests
- [ ] Compare benchmarks
- [ ] Update documentation

### Performance Verification

```bash
# Save benchmark results before migration
cargo bench > benchmark_before.txt

# Do migration work...

# Benchmark after migration
cargo bench > benchmark_after.txt

# Compare results
diff benchmark_before.txt benchmark_after.txt
```

## 🚨 Common Issues and Solutions

### Issue 1: Different Array Formats

```rust
// Problem: Converting from slice to ArrayView
let slice: &[f32] = &vec[..];

// Solution
use ndarray::ArrayView1;
let array_view = ArrayView1::from_shape(slice.len(), slice).unwrap();
```

### Issue 2: Non-contiguous Memory Layout

```rust
// Problem: Transposed matrices, etc.
let transposed = matrix.t();

// Solution: Make contiguous if needed
let contiguous = transposed.as_standard_layout().to_owned();
```

### Issue 3: Loss of Custom Optimizations

```rust
// When special optimizations are needed
#[cfg(feature = "custom_optimization")]
fn special_case_operation() {
    // Module-specific optimization
}

#[cfg(not(feature = "custom_optimization"))]
fn special_case_operation() {
    // Use Core implementation
}
```

## 📈 Phased Migration Strategy

### Phase 1: Basic Functions (1-2 days)
- Migrate simple SIMD operations
- Ensure tests pass

### Phase 2: Complex Functions (3-5 days)
- Matrix operations, FFT, and other complex operations
- Performance testing

### Phase 3: Optimization and Cleanup (1-2 days)
- Remove unnecessary code
- Update documentation
- Final performance verification

## 🎉 After Migration

1. **Remove unnecessary dependencies**
   ```toml
   # Remove from Cargo.toml
   # wide = "0.7"
   ```

2. **Update CI**
   - Add Clippy rules
   - Add lints to enforce Core usage

3. **Update documentation**
   - Update README
   - Document migration

## 💡 Tips

- **Start small**: Begin with the simplest functions
- **Trust tests**: If tests pass, implementation is correct
- **Measure performance**: Don't guess, measure
- **Ask questions**: Ask in issues if unclear

Following these steps, you can safely and incrementally migrate existing implementations to the Core unified system.