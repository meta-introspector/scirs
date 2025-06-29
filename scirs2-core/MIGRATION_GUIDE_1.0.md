# SciRS2 Core 1.0 Migration and Compatibility Guide

## Table of Contents

1. [Migration Overview](#migration-overview)
2. [Compatibility Matrix](#compatibility-matrix)
3. [Breaking Changes](#breaking-changes)
4. [Migration Path from Beta](#migration-path-from-beta)
5. [Migration from Alpha](#migration-from-alpha)
6. [Python SciPy Migration](#python-scipy-migration)
7. [Feature Migration Guide](#feature-migration-guide)
8. [Configuration Migration](#configuration-migration)
9. [Code Migration Examples](#code-migration-examples)
10. [Testing and Validation](#testing-and-validation)
11. [Rollback Procedures](#rollback-procedures)
12. [Post-Migration Optimization](#post-migration-optimization)

## Migration Overview

SciRS2 Core 1.0 represents the first stable release with long-term API compatibility guarantees. This guide provides comprehensive migration instructions for users upgrading from previous versions or migrating from other scientific computing platforms.

### Release Timeline

```
Alpha → Beta → 1.0 Stable
 ↓       ↓       ↓
0.x     0.1.0   1.0.0
```

### Migration Complexity

| Source Version | Target Version | Complexity | Estimated Time | Breaking Changes |
|---------------|----------------|------------|----------------|------------------|
| 0.1.0-beta.1  | 1.0.0         | Low        | 1-2 days       | Minimal         |
| 0.1.0-alpha   | 1.0.0         | Medium     | 1-2 weeks      | Moderate        |
| Python SciPy  | 1.0.0         | High       | 2-4 weeks      | Significant     |
| MATLAB        | 1.0.0         | High       | 3-6 weeks      | Significant     |
| Julia         | 1.0.0         | Medium     | 1-3 weeks      | Moderate        |

## Compatibility Matrix

### API Compatibility

| Feature Category | Beta → 1.0 | Alpha → 1.0 | SciPy → 1.0 |
|-----------------|------------|-------------|-------------|
| Core Arrays     | ✅ Compatible | ⚠️ Minor Changes | 🔄 API Translation |
| Linear Algebra  | ✅ Compatible | ⚠️ Minor Changes | 🔄 API Translation |
| Statistics      | ✅ Compatible | ❌ Breaking Changes | 🔄 API Translation |
| Signal Processing | ✅ Compatible | ⚠️ Minor Changes | 🔄 API Translation |
| Optimization    | ⚠️ Minor Changes | ❌ Breaking Changes | 🔄 API Translation |
| Interpolation   | ✅ Compatible | ✅ Compatible | 🔄 API Translation |
| Integration     | ✅ Compatible | ⚠️ Minor Changes | 🔄 API Translation |
| Sparse Matrices | ✅ Compatible | ❌ Breaking Changes | 🔄 API Translation |

**Legend:**
- ✅ Fully Compatible
- ⚠️ Minor Changes Required
- ❌ Breaking Changes
- 🔄 API Translation Required

### Feature Compatibility

| Feature | 0.1.0-beta.1 | 0.1.0-alpha | 1.0.0 | Migration Notes |
|---------|--------------|-------------|-------|-----------------|
| SIMD Operations | ✅ | ⚠️ | ✅ | API standardized in beta |
| GPU Computing | ✅ | ❌ | ✅ | Complete rewrite from alpha |
| Memory Mapping | ✅ | ⚠️ | ✅ | Enhanced in 1.0 |
| Parallel Processing | ✅ | ⚠️ | ✅ | Thread safety improved |
| Serialization | ⚠️ | ❌ | ✅ | Format changed |
| Configuration | ⚠️ | ❌ | ✅ | TOML format adopted |
| Error Handling | ✅ | ⚠️ | ✅ | Enhanced diagnostics |
| Observability | ✅ | ❌ | ✅ | Added in beta |

## Breaking Changes

### From Beta 1 to 1.0

#### Minimal Breaking Changes

1. **Configuration Format**
   ```rust
   // Beta 1 (DEPRECATED)
   use scirs2_core::config::set_config;
   set_config("threads", "8");
   
   // 1.0 (RECOMMENDED)
   use scirs2_core::config::{Config, ConfigValue};
   let mut config = Config::default();
   config.set("runtime.num_threads", ConfigValue::USize(8))?;
   scirs2_core::set_global_config(config)?;
   ```

2. **Error Types Consolidation**
   ```rust
   // Beta 1 (DEPRECATED)
   use scirs2_core::error::{ComputationError, ValidationError};
   
   // 1.0 (RECOMMENDED)
   use scirs2_core::error::{CoreError, ErrorContext};
   
   // Migration: All specific error types now use CoreError variants
   match result {
       Err(CoreError::ComputationError(ctx)) => { /* handle */ },
       Err(CoreError::ValidationError(ctx)) => { /* handle */ },
       Ok(value) => { /* success */ }
   }
   ```

3. **Feature Flag Reorganization**
   ```toml
   # Beta 1 (DEPRECATED)
   features = ["simd", "parallel", "gpu-cuda"]
   
   # 1.0 (RECOMMENDED)
   features = ["simd", "parallel", "gpu", "cuda"]
   ```

### From Alpha to 1.0

#### Significant Breaking Changes

1. **Array API Redesign**
   ```rust
   // Alpha (REMOVED)
   use scirs2_core::array::ScientificArray;
   let arr = ScientificArray::new(data);
   
   // 1.0 (NEW)
   use scirs2_core::array::{MaskedArray, RecordArray};
   let masked_arr = MaskedArray::new(data, mask);
   ```

2. **GPU API Complete Rewrite**
   ```rust
   // Alpha (REMOVED)
   use scirs2_core::gpu::CudaContext;
   let context = CudaContext::new(0)?;
   
   // 1.0 (NEW)
   use scirs2_core::gpu::{GpuContext, GpuBackend};
   let context = GpuContext::new(GpuBackend::CUDA)?;
   ```

3. **Statistics Module Restructure**
   ```rust
   // Alpha (REMOVED)
   use scirs2_core::stats::distributions::Normal;
   
   // 1.0 (NEW)
   use scirs2_stats::distributions::Normal;
   ```

## Migration Path from Beta

### Step 1: Update Dependencies

```toml
# Before (Beta 1)
[dependencies]
scirs2-core = "0.1.0-beta.1"

# After (1.0)
[dependencies]
scirs2-core = "1.0"
# Optional: Add specific modules as needed
scirs2-linalg = "1.0"
scirs2-stats = "1.0"
```

### Step 2: Update Configuration

```rust
// migration-config.rs

// Old configuration (Beta 1)
fn old_config() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_core::config::set_config;
    set_config("threads", "8");
    set_config("memory_limit", "8192");
    Ok(())
}

// New configuration (1.0)
fn new_config() -> scirs2_core::CoreResult<()> {
    use scirs2_core::config::{Config, ConfigValue};
    
    let config = Config::default()
        .set("runtime.num_threads", ConfigValue::USize(8))?
        .set("runtime.memory_limit_mb", ConfigValue::USize(8192))?;
    
    scirs2_core::set_global_config(config)?;
    Ok(())
}
```

### Step 3: Update Error Handling

```rust
// migration-errors.rs

// Old error handling (Beta 1)
fn old_error_handling() {
    match computation() {
        Ok(result) => println!("Success: {:?}", result),
        Err(e) => {
            // Multiple error types to handle
            match e {
                scirs2_core::error::ComputationError(msg) => eprintln!("Computation: {}", msg),
                scirs2_core::error::ValidationError(msg) => eprintln!("Validation: {}", msg),
                _ => eprintln!("Other error: {}", e),
            }
        }
    }
}

// New error handling (1.0)
fn new_error_handling() {
    use scirs2_core::{CoreResult, CoreError, diagnose_error};
    
    match computation() {
        Ok(result) => println!("Success: {:?}", result),
        Err(error) => {
            // Enhanced error diagnostics
            let diagnostics = diagnose_error(&error);
            eprintln!("Error: {}", diagnostics);
            
            // Pattern matching on unified error type
            match error {
                CoreError::ComputationError(ctx) => {
                    eprintln!("Computation failed: {}", ctx.message());
                },
                CoreError::ValidationError(ctx) => {
                    eprintln!("Validation failed: {}", ctx.message());
                },
                _ => eprintln!("Other error: {:?}", error),
            }
        }
    }
}

fn computation() -> CoreResult<f64> {
    // Your computation logic
    Ok(42.0)
}
```

### Step 4: Update Feature Flags

```toml
# Cargo.toml migration

# Before (Beta 1)
[dependencies]
scirs2-core = { version = "0.1.0-beta.1", features = ["simd", "parallel", "gpu-cuda"] }

# After (1.0)
[dependencies]
scirs2-core = { version = "1.0", features = ["simd", "parallel", "gpu", "cuda"] }
```

### Step 5: Validation and Testing

```rust
// validation-test.rs

#[cfg(test)]
mod migration_tests {
    use super::*;
    
    #[test]
    fn test_configuration_migration() {
        // Test that new configuration works
        assert!(new_config().is_ok());
    }
    
    #[test]
    fn test_error_handling_migration() {
        // Test error handling still works
        new_error_handling();
    }
    
    #[test]
    fn test_feature_compatibility() {
        // Test that features work as expected
        #[cfg(feature = "simd")]
        {
            use scirs2_core::simd_ops::SimdUnifiedOps;
            let a = ndarray::arr1(&[1.0f32, 2.0, 3.0, 4.0]);
            let b = ndarray::arr1(&[5.0f32, 6.0, 7.0, 8.0]);
            let result = f32::simd_add(&a.view(), &b.view());
            assert_eq!(result.len(), 4);
        }
    }
}
```

## Migration from Alpha

### Step 1: Major API Replacements

```rust
// alpha-to-1.0-migration.rs

// Alpha APIs that were removed/changed
mod alpha_apis {
    // These APIs no longer exist in 1.0
    /*
    use scirs2_core::array::ScientificArray;
    use scirs2_core::gpu::CudaContext;
    use scirs2_core::stats::distributions::Normal;
    */
}

// 1.0 replacement APIs
mod new_apis {
    use scirs2_core::array::{MaskedArray, RecordArray};
    use scirs2_core::gpu::{GpuContext, GpuBackend};
    use scirs2_stats::distributions::Normal;
    
    // Example migration for array handling
    pub fn migrate_array_usage() -> scirs2_core::CoreResult<()> {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        
        // Alpha (REMOVED): ScientificArray::new(data)
        // 1.0 (NEW): Use ndarray or specialized arrays
        let array = ndarray::Array1::from(data.clone());
        
        // Or use MaskedArray for advanced features
        let mask = vec![true, true, false, true];
        let masked_array = MaskedArray::new(data, Some(mask))?;
        
        Ok(())
    }
    
    // Example migration for GPU usage
    pub fn migrate_gpu_usage() -> scirs2_core::CoreResult<()> {
        // Alpha (REMOVED): CudaContext::new(0)
        // 1.0 (NEW): GpuContext with backend selection
        let gpu_context = GpuContext::new(GpuBackend::CUDA)?;
        
        Ok(())
    }
}
```

### Step 2: Module Restructuring

```rust
// module-migration.rs

// Alpha module structure (OUTDATED)
/*
use scirs2_core::stats::*;  // Everything was in core
use scirs2_core::linalg::*;
use scirs2_core::signal::*;
*/

// 1.0 module structure (CURRENT)
use scirs2_core::{CoreResult, validation, array};  // Core utilities only
use scirs2_linalg::*;     // Separate crate for linear algebra
use scirs2_stats::*;      // Separate crate for statistics
use scirs2_signal::*;     // Separate crate for signal processing

// Migration function to demonstrate new structure
fn demonstrate_new_structure() -> CoreResult<()> {
    // Core validation (still in scirs2-core)
    let data = vec![1.0, 2.0, 3.0];
    scirs2_core::validation::check_finite(&data, "input data")?;
    
    // Linear algebra (now in scirs2-linalg)
    let matrix = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let eigenvalues = scirs2_linalg::eigen::eigenvalues(&matrix)?;
    
    // Statistics (now in scirs2-stats)
    let mean = scirs2_stats::descriptive::mean(&data)?;
    
    Ok(())
}
```

### Step 3: Configuration System Migration

```rust
// config-migration.rs

// Alpha configuration (REMOVED)
/*
fn alpha_config() {
    scirs2_core::set_global_threads(8);
    scirs2_core::set_memory_limit(8 * 1024 * 1024 * 1024);
}
*/

// 1.0 configuration (NEW)
fn new_config() -> CoreResult<()> {
    use scirs2_core::config::{Config, ConfigValue};
    
    let config = Config::default()
        .set("runtime.num_threads", ConfigValue::USize(8))?
        .set("runtime.memory_limit_mb", ConfigValue::USize(8192))?
        .set("performance.enable_simd", ConfigValue::Bool(true))?
        .set("performance.enable_gpu", ConfigValue::Bool(true))?;
    
    scirs2_core::set_global_config(config)?;
    Ok(())
}
```

## Python SciPy Migration

### API Translation Table

| SciPy Function | SciRS2 Equivalent | Notes |
|----------------|-------------------|-------|
| `numpy.array()` | `ndarray::Array::from()` | Direct equivalent |
| `scipy.linalg.solve()` | `scirs2_linalg::solve()` | Same API pattern |
| `scipy.stats.norm()` | `scirs2_stats::distributions::Normal` | Object-oriented |
| `scipy.optimize.minimize()` | `scirs2_optimize::minimize()` | Similar interface |
| `scipy.signal.fft()` | `scirs2_fft::fft()` | Performance enhanced |
| `scipy.interpolate.interp1d()` | `scirs2_interpolate::interp1d()` | Type-safe |

### Migration Examples

#### Basic Array Operations

```python
# Python SciPy
import numpy as np
import scipy.stats as stats

# Create array
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Statistical operations
mean = np.mean(data)
std = np.std(data)

# Normal distribution
dist = stats.norm(loc=mean, scale=std)
samples = dist.rvs(size=1000)
```

```rust
// Rust SciRS2
use ndarray::Array1;
use scirs2_stats::{descriptive, distributions::Normal};

// Create array
let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

// Statistical operations
let mean = descriptive::mean(&data)?;
let std = descriptive::std(&data)?;

// Normal distribution
let dist = Normal::new(mean, std)?;
let samples = dist.sample(1000)?;
```

#### Linear Algebra

```python
# Python SciPy
import numpy as np
from scipy.linalg import solve, eig

# Linear system
A = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([5.0, 6.0])
x = solve(A, b)

# Eigenvalues
eigenvals, eigenvecs = eig(A)
```

```rust
// Rust SciRS2
use ndarray::arr2;
use scirs2_linalg::{solve, eigen};

// Linear system
let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
let b = arr1(&[5.0, 6.0]);
let x = solve(&a, &b)?;

// Eigenvalues
let (eigenvals, eigenvecs) = eigen::eig(&a)?;
```

#### Signal Processing

```python
# Python SciPy
import numpy as np
from scipy import signal

# Filter design
b, a = signal.butter(4, 0.1)

# Apply filter
filtered = signal.filtfilt(b, a, data)

# FFT
fft_result = np.fft.fft(data)
```

```rust
// Rust SciRS2
use scirs2_signal::{filter, fft};

// Filter design
let (b, a) = filter::butter(4, 0.1)?;

// Apply filter
let filtered = filter::filtfilt(&b, &a, &data)?;

// FFT
let fft_result = fft::fft(&data)?;
```

### Migration Automation Tools

```rust
// scipy-migration-tool.rs

/// Tool to help migrate SciPy code to SciRS2
pub struct SciPyMigrationTool {
    translation_rules: Vec<TranslationRule>,
}

pub struct TranslationRule {
    python_pattern: String,
    rust_replacement: String,
    notes: String,
}

impl SciPyMigrationTool {
    pub fn new() -> Self {
        Self {
            translation_rules: vec![
                TranslationRule {
                    python_pattern: "np.array({data})".to_string(),
                    rust_replacement: "Array1::from({data})".to_string(),
                    notes: "Add: use ndarray::Array1;".to_string(),
                },
                TranslationRule {
                    python_pattern: "np.mean({data})".to_string(),
                    rust_replacement: "descriptive::mean(&{data})?".to_string(),
                    notes: "Add: use scirs2_stats::descriptive;".to_string(),
                },
                // More rules...
            ],
        }
    }
    
    pub fn suggest_migration(&self, python_code: &str) -> Vec<MigrationSuggestion> {
        // Analyze Python code and suggest Rust equivalents
        let mut suggestions = Vec::new();
        
        for rule in &self.translation_rules {
            if python_code.contains(&rule.python_pattern.replace("{data}", "")) {
                suggestions.push(MigrationSuggestion {
                    original: rule.python_pattern.clone(),
                    replacement: rule.rust_replacement.clone(),
                    notes: rule.notes.clone(),
                });
            }
        }
        
        suggestions
    }
}

pub struct MigrationSuggestion {
    pub original: String,
    pub replacement: String,
    pub notes: String,
}
```

## Feature Migration Guide

### SIMD Operations

```rust
// Migration from manual SIMD to unified API

// Before (Alpha/Beta - manual SIMD)
#[cfg(target_arch = "x86_64")]
fn manual_simd_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    // Manual SIMD implementation
    // Complex and error-prone
}

// After (1.0 - unified SIMD API)
fn unified_simd_add(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    use scirs2_core::simd_ops::SimdUnifiedOps;
    f32::simd_add(a, b)  // Automatic SIMD optimization
}
```

### GPU Computing

```rust
// Migration from low-level GPU to high-level API

// Before (Alpha - low-level CUDA)
/*
fn low_level_gpu() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_core::gpu::cuda::*;
    
    let context = CudaContext::new(0)?;
    let stream = CudaStream::new(&context)?;
    let buffer = CudaBuffer::allocate(&context, 1000)?;
    // Manual memory management and kernel launches
    Ok(())
}
*/

// After (1.0 - high-level GPU API)
fn high_level_gpu() -> scirs2_core::CoreResult<()> {
    use scirs2_core::gpu::{GpuContext, GpuBackend};
    
    let gpu = GpuContext::new(GpuBackend::CUDA)?;
    let result = gpu.compute_on_device(&data, |gpu_data| {
        // High-level operations on GPU
        gpu_data.mapv(|x| x * 2.0)
    })?;
    
    Ok(())
}
```

### Memory Management

```rust
// Migration from manual memory management to automatic

// Before (Alpha/Beta - manual management)
fn manual_memory_management() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_core::memory::ManualMemory;
    
    let buffer = ManualMemory::allocate(1024 * 1024)?;
    // Manual lifetime management
    buffer.deallocate();
    Ok(())
}

// After (1.0 - automatic memory management)
fn automatic_memory_management() -> scirs2_core::CoreResult<()> {
    use scirs2_core::memory::{BufferPool, global_buffer_pool};
    
    // Automatic pool management
    let buffer = global_buffer_pool().get_buffer(1024 * 1024)?;
    // Automatic cleanup when buffer goes out of scope
    Ok(())
}
```

### Parallel Processing

```rust
// Migration from low-level threading to high-level parallel operations

// Before (Beta - low-level threading)
fn low_level_parallel(data: &[f64]) -> Vec<f64> {
    use std::thread;
    use std::sync::Arc;
    
    let data = Arc::new(data.to_vec());
    let mut handles = vec![];
    
    for i in 0..4 {
        let data_clone = Arc::clone(&data);
        let handle = thread::spawn(move || {
            // Manual thread management
            let chunk_size = data_clone.len() / 4;
            let start = i * chunk_size;
            let end = (i + 1) * chunk_size;
            data_clone[start..end].iter().map(|x| x * 2.0).collect::<Vec<_>>()
        });
        handles.push(handle);
    }
    
    let mut result = Vec::new();
    for handle in handles {
        result.extend(handle.join().unwrap());
    }
    result
}

// After (1.0 - high-level parallel operations)
fn high_level_parallel(data: &[f64]) -> Vec<f64> {
    use scirs2_core::parallel_ops::par_chunks;
    
    par_chunks(data, 1000)
        .map(|chunk| chunk.iter().map(|x| x * 2.0).collect::<Vec<_>>())
        .flatten()
        .collect()
}
```

## Configuration Migration

### Environment Variables

```bash
# Migration of environment variables

# Before (Alpha/Beta)
export SCIRS2_THREADS=8
export SCIRS2_MEMORY=8192
export SCIRS2_GPU_DEVICE=0

# After (1.0)
export SCIRS2_NUM_THREADS=8
export SCIRS2_MEMORY_LIMIT=8192
export CUDA_VISIBLE_DEVICES=0  # Standard CUDA variable
```

### Configuration Files

```toml
# migration-config.toml

# Before (Beta) - config.json
#{
#  "threads": 8,
#  "memory_limit": 8192,
#  "enable_gpu": true
#}

# After (1.0) - config.toml
[runtime]
num_threads = 8
memory_limit_mb = 8192

[performance]
enable_simd = true
enable_gpu = true
adaptive_optimization = true

[observability]
enable_metrics = true
enable_tracing = true
log_level = "info"

[security]
enable_validation = true
audit_logging = true
```

### Programmatic Configuration

```rust
// config-migration-example.rs

// Before (Beta)
fn beta_config() {
    scirs2_core::set_threads(8);
    scirs2_core::set_memory_limit(8192);
}

// After (1.0)
fn stable_config() -> scirs2_core::CoreResult<()> {
    use scirs2_core::config::{Config, ConfigValue};
    
    let config = Config::from_file("config.toml")?
        .or_default()
        .set("runtime.num_threads", ConfigValue::USize(8))?
        .set("runtime.memory_limit_mb", ConfigValue::USize(8192))?;
    
    scirs2_core::set_global_config(config)?;
    Ok(())
}
```

## Code Migration Examples

### Complete Application Migration

```rust
// complete-migration-example.rs

// Example: Migrating a complete scientific application

// Before (Alpha/Beta)
/*
mod old_application {
    use scirs2_core::*;
    
    fn main() -> Result<(), Box<dyn std::error::Error>> {
        // Old configuration
        set_threads(8);
        
        // Old array creation
        let data = ScientificArray::new(vec![1.0, 2.0, 3.0, 4.0]);
        
        // Old computation
        let result = compute_statistics(&data);
        
        println!("Result: {:?}", result);
        Ok(())
    }
}
*/

// After (1.0)
mod new_application {
    use scirs2_core::{CoreResult, config::{Config, ConfigValue}};
    use scirs2_stats::descriptive;
    use ndarray::Array1;
    
    fn main() -> CoreResult<()> {
        // New configuration
        let config = Config::default()
            .set("runtime.num_threads", ConfigValue::USize(8))?;
        scirs2_core::set_global_config(config)?;
        
        // New array creation
        let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        
        // New computation
        let result = compute_statistics(&data)?;
        
        println!("Result: {:?}", result);
        Ok(())
    }
    
    fn compute_statistics(data: &Array1<f64>) -> CoreResult<StatisticsResult> {
        Ok(StatisticsResult {
            mean: descriptive::mean(data)?,
            std: descriptive::std(data)?,
            min: descriptive::min(data)?,
            max: descriptive::max(data)?,
        })
    }
    
    #[derive(Debug)]
    struct StatisticsResult {
        mean: f64,
        std: f64,
        min: f64,
        max: f64,
    }
}
```

### Library Integration Migration

```rust
// library-integration-migration.rs

// Migration example for library authors

// Before (Beta) - Library exposing old API
/*
pub mod old_library {
    use scirs2_core::ScientificArray;
    
    pub fn process_data(input: &ScientificArray) -> Result<ScientificArray, String> {
        // Old processing logic
        Ok(input.clone())
    }
}
*/

// After (1.0) - Library exposing new API with compatibility
pub mod new_library {
    use scirs2_core::{CoreResult, array::MaskedArray};
    use ndarray::{Array1, ArrayView1};
    
    // New primary API
    pub fn process_data(input: ArrayView1<f64>) -> CoreResult<Array1<f64>> {
        // New processing logic with enhanced error handling
        scirs2_core::validation::check_finite(input.as_slice().unwrap(), "input")?;
        Ok(input.mapv(|x| x * 2.0).to_owned())
    }
    
    // Compatibility wrapper for migration period
    #[deprecated(since = "1.0.0", note = "Use process_data with ArrayView1 instead")]
    pub fn process_data_compat(input: &[f64]) -> CoreResult<Vec<f64>> {
        let array_view = ArrayView1::from(input);
        let result = process_data(array_view)?;
        Ok(result.to_vec())
    }
    
    // Advanced API using MaskedArray
    pub fn process_masked_data(input: &MaskedArray<f64>) -> CoreResult<MaskedArray<f64>> {
        let processed_data = input.data().mapv(|x| x * 2.0);
        Ok(MaskedArray::new(processed_data.to_vec(), input.mask().cloned())?)
    }
}

// Migration helper
pub mod migration_helpers {
    use super::*;
    
    /// Helper function to migrate from old API patterns
    pub fn migrate_from_vec(data: Vec<f64>) -> CoreResult<Array1<f64>> {
        Ok(Array1::from(data))
    }
    
    /// Helper to convert results back to Vec for compatibility
    pub fn to_vec_compat(array: Array1<f64>) -> Vec<f64> {
        array.to_vec()
    }
}
```

## Testing and Validation

### Migration Test Suite

```rust
// migration-tests.rs

#[cfg(test)]
mod migration_tests {
    use super::*;
    use scirs2_core::CoreResult;
    
    /// Test suite to validate migration correctness
    #[test]
    fn test_configuration_migration() -> CoreResult<()> {
        // Test that old and new configurations produce same results
        let config = scirs2_core::config::Config::default()
            .set("runtime.num_threads", scirs2_core::config::ConfigValue::USize(4))?;
        
        scirs2_core::set_global_config(config)?;
        
        // Verify configuration was applied
        let resources = scirs2_core::resource::get_system_resources();
        assert!(resources.cpu_cores >= 1);
        
        Ok(())
    }
    
    #[test]
    fn test_array_api_migration() -> CoreResult<()> {
        // Test that array operations work as expected
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let array = ndarray::Array1::from(data.clone());
        
        // Test basic operations
        let sum = array.sum();
        let expected_sum: f64 = data.iter().sum();
        assert!((sum - expected_sum).abs() < 1e-10);
        
        Ok(())
    }
    
    #[test]
    fn test_error_handling_migration() {
        // Test that new error handling works
        let result: CoreResult<f64> = Err(scirs2_core::CoreError::ValidationError(
            scirs2_core::error::ErrorContext::new("Test error")
        ));
        
        match result {
            Err(scirs2_core::CoreError::ValidationError(ctx)) => {
                assert!(ctx.message().contains("Test error"));
            },
            _ => panic!("Expected ValidationError"),
        }
    }
    
    #[test]
    fn test_performance_regression() -> CoreResult<()> {
        // Test that performance hasn't regressed
        let data = (0..10000).map(|i| i as f64).collect::<Vec<_>>();
        let array = ndarray::Array1::from(data);
        
        let start = std::time::Instant::now();
        let _result = array.mapv(|x| x * 2.0);
        let duration = start.elapsed();
        
        // Ensure operation completes in reasonable time
        assert!(duration < std::time::Duration::from_millis(100));
        
        Ok(())
    }
    
    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_migration() {
        // Test SIMD operations work correctly
        use scirs2_core::simd_ops::SimdUnifiedOps;
        
        let a = ndarray::arr1(&[1.0f32, 2.0, 3.0, 4.0]);
        let b = ndarray::arr1(&[5.0f32, 6.0, 7.0, 8.0]);
        
        let result = f32::simd_add(&a.view(), &b.view());
        let expected = ndarray::arr1(&[6.0f32, 8.0, 10.0, 12.0]);
        
        assert_eq!(result, expected);
    }
    
    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_migration() {
        // Test parallel operations work correctly
        use scirs2_core::parallel_ops::{par_chunks, set_num_threads};
        
        set_num_threads(2);
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        
        let result: Vec<f64> = par_chunks(&data, 100)
            .map(|chunk| chunk.iter().sum::<f64>())
            .collect();
        
        assert_eq!(result.len(), 10);  // 1000 / 100 = 10 chunks
    }
}

/// Integration tests that compare results with reference implementations
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_scipy_compatibility() {
        // Test that results match SciPy (within numerical precision)
        let data = ndarray::arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        
        // Test mean calculation
        let mean = scirs2_stats::descriptive::mean(&data).unwrap();
        let expected_mean = 3.0; // (1+2+3+4+5)/5 = 15/5 = 3
        assert!((mean - expected_mean).abs() < 1e-10);
        
        // Test standard deviation
        let std = scirs2_stats::descriptive::std(&data).unwrap();
        let expected_std = (2.5_f64).sqrt(); // Sample std of [1,2,3,4,5]
        assert!((std - expected_std).abs() < 1e-10);
    }
}
```

### Performance Validation

```rust
// performance-validation.rs

/// Performance validation suite for migration
pub struct PerformanceValidator {
    baseline_metrics: Vec<PerformanceMetric>,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetric {
    operation: String,
    data_size: usize,
    duration: std::time::Duration,
    memory_usage: usize,
}

impl PerformanceValidator {
    pub fn new() -> Self {
        Self {
            baseline_metrics: Vec::new(),
        }
    }
    
    pub fn benchmark_operation<F, T>(&mut self, name: &str, data_size: usize, operation: F) -> CoreResult<T>
    where
        F: FnOnce() -> CoreResult<T>,
    {
        let memory_before = self.get_memory_usage();
        let start_time = std::time::Instant::now();
        
        let result = operation()?;
        
        let duration = start_time.elapsed();
        let memory_after = self.get_memory_usage();
        let memory_usage = memory_after.saturating_sub(memory_before);
        
        self.baseline_metrics.push(PerformanceMetric {
            operation: name.to_string(),
            data_size,
            duration,
            memory_usage,
        });
        
        Ok(result)
    }
    
    pub fn validate_performance(&self, tolerance_percent: f64) -> Result<(), String> {
        // Validate that performance is within acceptable tolerance
        for metric in &self.baseline_metrics {
            if metric.duration > std::time::Duration::from_secs(10) {
                return Err(format!("Operation '{}' took too long: {:?}", metric.operation, metric.duration));
            }
            
            // Additional performance checks...
        }
        
        Ok(())
    }
    
    fn get_memory_usage(&self) -> usize {
        // Implementation depends on platform
        0
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    
    #[test]
    fn test_migration_performance() -> CoreResult<()> {
        let mut validator = PerformanceValidator::new();
        
        // Benchmark array operations
        validator.benchmark_operation("array_creation", 10000, || {
            let data: Vec<f64> = (0..10000).map(|i| i as f64).collect();
            let _array = ndarray::Array1::from(data);
            Ok(())
        })?;
        
        // Benchmark mathematical operations
        validator.benchmark_operation("array_multiplication", 10000, || {
            let array = ndarray::Array1::from((0..10000).map(|i| i as f64).collect::<Vec<_>>());
            let _result = array.mapv(|x| x * 2.0);
            Ok(())
        })?;
        
        // Validate performance is acceptable
        validator.validate_performance(20.0)?; // 20% tolerance
        
        Ok(())
    }
}
```

## Rollback Procedures

### Version Rollback

```toml
# Cargo.toml rollback procedure

# If migration to 1.0 fails, rollback to last known good version
[dependencies]
# scirs2-core = "1.0"  # Comment out problematic version
scirs2-core = "0.1.0-beta.1"  # Rollback to stable beta
```

### Configuration Rollback

```rust
// rollback-procedures.rs

/// Rollback utilities for failed migrations
pub struct RollbackManager {
    backup_config: Option<String>,
    backup_data: Option<Vec<u8>>,
}

impl RollbackManager {
    pub fn new() -> Self {
        Self {
            backup_config: None,
            backup_data: None,
        }
    }
    
    /// Create backup before migration
    pub fn create_backup(&mut self) -> CoreResult<()> {
        // Backup configuration
        if let Ok(config_content) = std::fs::read_to_string("config.toml") {
            self.backup_config = Some(config_content);
        }
        
        // Backup critical data
        if let Ok(data) = std::fs::read("critical_data.bin") {
            self.backup_data = Some(data);
        }
        
        Ok(())
    }
    
    /// Rollback to previous state
    pub fn rollback(&self) -> CoreResult<()> {
        // Restore configuration
        if let Some(ref config) = self.backup_config {
            std::fs::write("config.toml", config)?;
        }
        
        // Restore data
        if let Some(ref data) = self.backup_data {
            std::fs::write("critical_data.bin", data)?;
        }
        
        println!("Rollback completed successfully");
        Ok(())
    }
}

/// Automated rollback on migration failure
pub fn safe_migration<F>(migration_fn: F) -> CoreResult<()>
where
    F: FnOnce() -> CoreResult<()>,
{
    let mut rollback_manager = RollbackManager::new();
    rollback_manager.create_backup()?;
    
    match migration_fn() {
        Ok(()) => {
            println!("Migration completed successfully");
            Ok(())
        },
        Err(error) => {
            eprintln!("Migration failed: {:?}", error);
            eprintln!("Initiating rollback...");
            rollback_manager.rollback()?;
            Err(error)
        }
    }
}
```

### Gradual Migration Strategy

```rust
// gradual-migration.rs

/// Strategy for gradual migration with fallback
pub struct GradualMigrator {
    migration_percentage: f64,
    fallback_enabled: bool,
}

impl GradualMigrator {
    pub fn new() -> Self {
        Self {
            migration_percentage: 0.0,
            fallback_enabled: true,
        }
    }
    
    /// Gradually increase migration percentage
    pub fn increase_migration(&mut self, percentage: f64) {
        self.migration_percentage = (self.migration_percentage + percentage).min(100.0);
        println!("Migration percentage: {:.1}%", self.migration_percentage);
    }
    
    /// Execute operation with gradual migration
    pub fn execute_with_migration<F, G, T>(&self, new_impl: F, old_impl: G) -> CoreResult<T>
    where
        F: FnOnce() -> CoreResult<T>,
        G: FnOnce() -> CoreResult<T>,
    {
        use rand::Rng;
        
        let mut rng = rand::thread_rng();
        let random_percentage = rng.gen_range(0.0..100.0);
        
        if random_percentage < self.migration_percentage {
            // Use new implementation
            match new_impl() {
                Ok(result) => Ok(result),
                Err(error) if self.fallback_enabled => {
                    eprintln!("New implementation failed, falling back: {:?}", error);
                    old_impl()
                },
                Err(error) => Err(error),
            }
        } else {
            // Use old implementation
            old_impl()
        }
    }
}

#[cfg(test)]
mod gradual_migration_tests {
    use super::*;
    
    #[test]
    fn test_gradual_migration() -> CoreResult<()> {
        let mut migrator = GradualMigrator::new();
        
        // Start with 10% migration
        migrator.increase_migration(10.0);
        
        // Execute with migration
        let result = migrator.execute_with_migration(
            || {
                // New implementation
                Ok("new_result".to_string())
            },
            || {
                // Old implementation
                Ok("old_result".to_string())
            }
        )?;
        
        assert!(result == "new_result" || result == "old_result");
        Ok(())
    }
}
```

## Post-Migration Optimization

### Performance Tuning

```rust
// post-migration-optimization.rs

/// Post-migration optimization recommendations
pub struct PostMigrationOptimizer {
    performance_metrics: Vec<PerformanceMetric>,
}

impl PostMigrationOptimizer {
    pub fn new() -> Self {
        Self {
            performance_metrics: Vec::new(),
        }
    }
    
    /// Analyze performance after migration
    pub fn analyze_performance(&mut self) -> CoreResult<OptimizationReport> {
        let mut report = OptimizationReport::new();
        
        // Check SIMD utilization
        if self.is_simd_underutilized()? {
            report.add_recommendation(
                "Enable SIMD features",
                "Add 'simd' feature flag for better performance"
            );
        }
        
        // Check parallel processing
        if self.is_sequential_processing()? {
            report.add_recommendation(
                "Enable parallel processing",
                "Add 'parallel' feature flag and configure thread count"
            );
        }
        
        // Check memory usage
        if self.is_memory_inefficient()? {
            report.add_recommendation(
                "Enable memory-efficient operations",
                "Use memory-mapped arrays for large datasets"
            );
        }
        
        Ok(report)
    }
    
    fn is_simd_underutilized(&self) -> CoreResult<bool> {
        // Check if SIMD could improve performance
        Ok(false) // Placeholder
    }
    
    fn is_sequential_processing(&self) -> CoreResult<bool> {
        // Check if parallel processing would help
        Ok(false) // Placeholder
    }
    
    fn is_memory_inefficient(&self) -> CoreResult<bool> {
        // Check memory usage patterns
        Ok(false) // Placeholder
    }
}

pub struct OptimizationReport {
    recommendations: Vec<OptimizationRecommendation>,
}

impl OptimizationReport {
    pub fn new() -> Self {
        Self {
            recommendations: Vec::new(),
        }
    }
    
    pub fn add_recommendation(&mut self, title: &str, description: &str) {
        self.recommendations.push(OptimizationRecommendation {
            title: title.to_string(),
            description: description.to_string(),
        });
    }
    
    pub fn print_report(&self) {
        println!("=== Post-Migration Optimization Report ===");
        for (i, rec) in self.recommendations.iter().enumerate() {
            println!("{}. {}", i + 1, rec.title);
            println!("   {}", rec.description);
        }
    }
}

#[derive(Debug)]
pub struct OptimizationRecommendation {
    title: String,
    description: String,
}
```

### Configuration Optimization

```rust
// config-optimization.rs

/// Optimize configuration after migration
pub fn optimize_post_migration_config() -> CoreResult<()> {
    use scirs2_core::config::{Config, ConfigValue};
    use scirs2_core::resource::get_system_resources;
    
    let resources = get_system_resources();
    
    let optimized_config = Config::default()
        // Optimize thread count based on system
        .set("runtime.num_threads", ConfigValue::USize(resources.cpu_cores.min(16)))?
        // Optimize memory based on available RAM
        .set("runtime.memory_limit_mb", ConfigValue::USize(resources.total_memory_mb / 2))?
        // Enable performance features
        .set("performance.enable_simd", ConfigValue::Bool(true))?
        .set("performance.enable_gpu", ConfigValue::Bool(resources.gpu_count > 0))?
        .set("performance.adaptive_optimization", ConfigValue::Bool(true))?
        // Enable monitoring for production
        .set("observability.enable_metrics", ConfigValue::Bool(true))?
        .set("observability.enable_tracing", ConfigValue::Bool(true))?;
    
    scirs2_core::set_global_config(optimized_config)?;
    
    println!("Configuration optimized for current system:");
    println!("  CPU cores: {}", resources.cpu_cores);
    println!("  Memory: {} MB", resources.total_memory_mb);
    println!("  GPU count: {}", resources.gpu_count);
    
    Ok(())
}
```

---

## Summary

This comprehensive migration guide provides detailed instructions for upgrading to SciRS2 Core 1.0 from previous versions and migrating from other scientific computing platforms. Key points:

1. **Beta to 1.0**: Minimal breaking changes, mostly configuration updates
2. **Alpha to 1.0**: Significant API changes require code restructuring
3. **SciPy Migration**: Comprehensive API translation with automation tools
4. **Testing**: Extensive validation suite to ensure migration correctness
5. **Rollback**: Safe rollback procedures for failed migrations
6. **Optimization**: Post-migration performance tuning recommendations

For additional migration support, consult the [SciRS2 documentation](https://docs.rs/scirs2-core/1.0) or reach out to the community through [GitHub Discussions](https://github.com/cool-japan/scirs/discussions).

---

**Version**: SciRS2 Core 1.0  
**Last Updated**: 2025-06-29  
**Authors**: SciRS2 Migration Team  
**License**: See LICENSE file for details