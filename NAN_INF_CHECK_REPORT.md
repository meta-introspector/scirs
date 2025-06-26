# NaN and Infinity Check Report for SciRS2

## Summary

This report identifies locations in the SciRS2 codebase where NaN or Infinity values could be produced but are not properly checked, along with patterns of existing validation.

## Current State of Validation

### Existing Infrastructure

1. **Core validation module** (`scirs2-core/src/validation.rs`):
   - `check_finite()` - Validates single floating-point values
   - `check_array_finite()` - Validates all values in an array
   - Good error handling with context and location information

2. **Error messages module** (`scirs2-stats/src/error_messages.rs`):
   - `validation::ensure_positive()` - Used for domain validation
   - Consistent error messages for validation failures

### Areas Missing NaN/Inf Checks

## 1. Division Operations Without Zero Checks

### scirs2-stats/src/correlation.rs
```rust
// Line 89: Division without explicit zero check
let corr = sum_xy / (sum_x2 * sum_y2).sqrt();

// Lines 82-86 check for epsilon but could be more robust:
if sum_x2 <= F::epsilon() || sum_y2 <= F::epsilon() {
    return Err(...);
}
// Should explicitly check for zero or near-zero denominators
```

### scirs2-stats/src/tests/mod.rs
```rust
// Line 175: Division in t-statistic calculation
t_stat = (mean_x - mean_y) / se;

// Line 188: Division in Welch's t-test
t_stat = (mean_x - mean_y) / se;

// Line 195: Division for degrees of freedom
df = numerator / denominator;
```

### scirs2-ndimage/src/segmentation/chan_vese.rs
```rust
// Lines 201-202: Division with max() protection but no NaN check
c1 /= area1.max(1.0);
c2 /= area2.max(1.0);

// Line 76: Division in curvature calculation
let denominator = (phi_x * phi_x + phi_y * phi_y).powf(1.5) + 1e-10;
// Adding epsilon helps but result could still be NaN if phi gradients are NaN

// Line 108: Division in sign calculation
let sign_phi = phi[[i, j]] / (phi[[i, j]].abs() + 1e-10);
```

## 2. Square Root Operations Without Negative Checks

### scirs2-vision/src/feature/ransac.rs
```rust
// Multiple instances of sqrt without checking if sum is negative:
let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
// Should verify sum is non-negative before sqrt
```

### scirs2-vision/src/feature/hough_circle.rs
```rust
// Line: Magnitude calculation without validation
let magnitude = (gx * gx + gy * gy).sqrt();
// Could check that gx, gy are finite first
```

### scirs2-cluster hierarchy modules
```rust
// Multiple sqrt operations on distance calculations:
sum.sqrt()  // Should verify sum >= 0
(factor * avg_dist_sq).sqrt()  // Should check inputs
```

## 3. Logarithm Operations Without Positive Checks

### scirs2-stats/src/distributions/exponential.rs
```rust
// Line 215: Log of (1-p) without explicit validation
let result = -((F::one() - p).ln()) / self.rate;
// p is validated to be in [0,1] but edge case p=1 returns infinity (handled)
```

### scirs2-stats distributions modules
Many distribution implementations use `.ln()`, `.log()` without explicit checks:
- Normal distribution
- Gamma distribution
- Beta distribution
- etc.

Most validate input parameters but could benefit from explicit NaN/Inf checks on results.

## 4. Power Operations That Could Overflow

### scirs2-ndimage/src/segmentation/chan_vese.rs
```rust
// Line 213: Square operation
let f1 = (img[[i, j]] - c1).powi(2);
// Could overflow for large differences
```

## Recommendations

### 1. Add Validation Helpers to Core

Create additional validation functions in `scirs2-core/src/validation.rs`:

```rust
/// Check if a division operation will produce finite result
pub fn check_division_safe<T: Float>(numerator: T, denominator: T, operation: &str) -> CoreResult<T> {
    check_finite(numerator, &format!("{} numerator", operation))?;
    check_finite(denominator, &format!("{} denominator", operation))?;
    
    if denominator.abs() < T::epsilon() {
        return Err(CoreError::ValueError(
            ErrorContext::new(format!("{}: division by zero or near-zero value", operation))
                .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    
    let result = numerator / denominator;
    check_finite(result, &format!("{} result", operation))?;
    Ok(result)
}

/// Check if sqrt operation will produce finite result  
pub fn check_sqrt_safe<T: Float>(value: T, operation: &str) -> CoreResult<T> {
    check_finite(value, &format!("{} input", operation))?;
    
    if value < T::zero() {
        return Err(CoreError::ValueError(
            ErrorContext::new(format!("{}: cannot take sqrt of negative value", operation))
                .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    
    let result = value.sqrt();
    check_finite(result, &format!("{} result", operation))?;
    Ok(result)
}

/// Check if log operation will produce finite result
pub fn check_log_safe<T: Float>(value: T, operation: &str) -> CoreResult<T> {
    check_finite(value, &format!("{} input", operation))?;
    
    if value <= T::zero() {
        return Err(CoreError::ValueError(
            ErrorContext::new(format!("{}: cannot take log of non-positive value", operation))
                .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    
    let result = value.ln();
    check_finite(result, &format!("{} result", operation))?;
    Ok(result)
}
```

### 2. Apply Validation in Critical Sections

#### For correlation calculations:
```rust
// Replace direct division with validated division
let corr = check_division_safe(sum_xy, (sum_x2 * sum_y2).sqrt(), "correlation")?;
```

#### For statistical tests:
```rust
// Validate intermediate calculations
let se = check_sqrt_safe(var_x_over_n_x + var_y_over_n_y, "standard error")?;
let t_stat = check_division_safe(mean_x - mean_y, se, "t-statistic")?;
```

#### For segmentation algorithms:
```rust
// Validate region averages
c1 = check_division_safe(c1, area1.max(1.0), "region average c1")?;
c2 = check_division_safe(c2, area2.max(1.0), "region average c2")?;
```

### 3. Add Debug Assertions

For performance-critical code where full validation might be expensive:

```rust
debug_assert!(value.is_finite(), "Value must be finite");
debug_assert!(denominator != 0.0, "Denominator must not be zero");
```

### 4. Document Numerical Stability Guarantees

Add documentation to functions about:
- Input domain requirements
- Conditions that might produce NaN/Inf
- Numerical stability guarantees
- Error handling behavior

### 5. Consider Using checked_* Operations

For integer operations that could overflow:
```rust
// Instead of: x.powi(2)
// Use: x.checked_pow(2).ok_or_else(|| ...)?
```

## Priority Areas for Immediate Attention

1. **Statistical tests** (`scirs2-stats/src/tests/`) - Critical for correctness
2. **Correlation functions** (`scirs2-stats/src/correlation.rs`) - Widely used
3. **Distribution calculations** (`scirs2-stats/src/distributions/`) - Mathematical accuracy required
4. **Image segmentation** (`scirs2-ndimage/src/segmentation/`) - Numerical stability important
5. **Feature detection** (`scirs2-vision/src/feature/`) - Edge cases common with real images

## Testing Recommendations

1. Add unit tests with edge cases:
   - Zero variance data for correlation
   - Single-element arrays for statistical tests  
   - Constant images for segmentation
   - Near-zero denominators

2. Add property-based tests to verify:
   - No NaN/Inf outputs for valid inputs
   - Graceful error handling for edge cases

3. Add benchmarks to measure validation overhead

## Conclusion

While the codebase has good validation infrastructure in `scirs2-core`, many mathematical operations could benefit from explicit NaN/Inf checking. The recommendations above would improve numerical robustness without significantly impacting performance.