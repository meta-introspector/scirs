# Practical Migration Steps

This document provides the actual workflow for migrating existing implementations to use the unified core system.

## Quick Start: How to Actually Do It

### 1. **Start with One Module**
```bash
# Choose a module to migrate (e.g., scirs2-optimize)
cd scirs2-optimize

# Create a branch for migration
git checkout -b migrate-to-core-simd

# Save current performance baseline
cargo bench > baseline_performance.txt
```

### 2. **Update Dependencies First**
```toml
# Edit Cargo.toml
[dependencies]
scirs2-core = { workspace = true, features = ["simd", "parallel"] }

# Remove or comment out direct SIMD dependencies
# wide = "0.7"  # REMOVE
# packed_simd = "0.3"  # REMOVE
```

### 3. **Use IDE Search & Replace**

**Step 1: Replace imports**
- Find: `use wide::{f32x8, f64x4};`
- Replace: `use scirs2_core::simd_ops::SimdUnifiedOps;`

- Find: `use std::arch::x86_64::*;`
- Replace: `use scirs2_core::simd_ops::{SimdUnifiedOps, PlatformCapabilities};`

**Step 2: Replace simple operations**
- Find: `f32x8::splat(0.0)`
- Replace: `/* Will be replaced with unified ops */`

### 4. **Migrate Function by Function**

#### Start with the Simplest Function
```rust
// Pick the simplest SIMD function first
// OLD: src/simd_ops.rs
pub fn add_vectors_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
    // Complex SIMD code...
}

// NEW: Create temporary file src/simd_ops_new.rs
use ndarray::{Array1, ArrayView1};
use scirs2_core::simd_ops::SimdUnifiedOps;

pub fn add_vectors_simd(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    f32::simd_add(a, b)
}
```

#### Test Immediately
```bash
# Add a test to verify behavior is the same
#[test]
fn test_add_vectors_migration() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    
    // Convert to ArrayView for new API
    let a_view = ArrayView1::from(&a);
    let b_view = ArrayView1::from(&b);
    
    let result = add_vectors_simd(&a_view, &b_view);
    
    assert_eq!(result.as_slice().unwrap(), &[6.0, 8.0, 10.0, 12.0]);
}

# Run the specific test
cargo test test_add_vectors_migration
```

### 5. **Handle Common Patterns**

#### Pattern 1: Slice to ArrayView
```rust
// Problem: Function takes slices
pub fn process(data: &[f64]) -> f64 {
    // Need ArrayView for core ops
}

// Solution: Add conversion
pub fn process(data: &[f64]) -> f64 {
    let view = ArrayView1::from(data);
    f64::simd_sum(&view)
}
```

#### Pattern 2: Custom SIMD loops
```rust
// OLD: Manual SIMD loop
for i in (0..n).step_by(8) {
    let vec = f32x8::from_slice(&data[i..]);
    // Process...
}

// NEW: Let core handle it
let result = f32::simd_operation(&data_view);
```

### 6. **Incremental Testing Strategy**

```bash
# After each function migration:
cargo test --lib         # Run unit tests
cargo test --doc        # Run doc tests
cargo bench --no-run    # Ensure benchmarks compile
```

### 7. **Performance Validation**

```bash
# Run benchmarks after migrating a few functions
cargo bench > current_performance.txt

# Compare with baseline
diff baseline_performance.txt current_performance.txt

# If performance degrades significantly (>10%):
# 1. Check if you're using the right core function
# 2. Consider keeping old implementation behind feature flag
# 3. Report issue to core team for optimization
```

### 8. **Gradual Replacement**

```rust
// Use feature flags for gradual migration
#[cfg(feature = "use_core_simd")]
pub use simd_ops_new::*;

#[cfg(not(feature = "use_core_simd"))]
pub use simd_ops_old::*;
```

### 9. **Final Cleanup**

```bash
# Once all tests pass and performance is acceptable:

# Remove old files
git rm src/simd_ops_old.rs

# Remove old dependencies from Cargo.toml
# Update documentation
cargo doc

# Run final checks
cargo clippy
cargo fmt
cargo test --all-features
```

## Real Example: Migrating a Dot Product

### Before (50+ lines):
```rust
use std::arch::x86_64::*;

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum = 0.0;
    
    unsafe {
        if is_x86_feature_detected!("avx2") {
            let mut vsum = _mm256_setzero_ps();
            let chunks = len / 8;
            
            for i in 0..chunks {
                let idx = i * 8;
                let va = _mm256_loadu_ps(a.as_ptr().add(idx));
                let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
                vsum = _mm256_fmadd_ps(va, vb, vsum);
            }
            
            // Horizontal sum...
            // Handle remainder...
        } else {
            // SSE fallback...
        }
    }
    
    sum
}
```

### After (3 lines):
```rust
use scirs2_core::simd_ops::SimdUnifiedOps;

pub fn dot_product(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    f32::simd_dot(a, b)
}
```

## Common Pitfalls and Solutions

### Pitfall 1: Trying to migrate everything at once
**Solution**: Migrate one function at a time, test, commit.

### Pitfall 2: Not checking performance
**Solution**: Benchmark after every few functions.

### Pitfall 3: Fighting the new API
**Solution**: Embrace ArrayView/Array instead of slices.

### Pitfall 4: Keeping unnecessary complexity
**Solution**: Trust that core implementations are optimized.

## Time Estimates

- Simple module (10-20 SIMD functions): 2-3 days
- Complex module (50+ SIMD functions): 1 week
- Module with custom GPU kernels: 1-2 weeks

## When You're Done

1. Delete old SIMD code
2. Remove platform-specific dependencies
3. Update module documentation
4. Submit PR with:
   - Migration summary
   - Performance comparison
   - Any issues encountered

This practical approach minimizes risk and ensures a smooth migration to the unified core system.