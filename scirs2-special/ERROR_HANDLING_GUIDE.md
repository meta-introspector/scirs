# Error Handling Guide for scirs2-special

This guide documents the consistent error handling patterns implemented across all special functions in the scirs2-special module.

## Overview

The scirs2-special module implements a comprehensive error handling system that provides:

1. **Consistent Error Types**: All functions use `SpecialError` and `SpecialResult<T>` types
2. **Context Tracking**: Detailed error context including function names, parameters, and operations
3. **Validation Patterns**: Standardized input validation across all functions
4. **Recovery Strategies**: Optional error recovery mechanisms
5. **Error Wrappers**: Safe wrappers for all special functions

## Error Types

### SpecialError

The main error enum with the following variants:

```rust
pub enum SpecialError {
    ComputationError(String),    // General computation errors
    DomainError(String),         // Input outside valid domain
    ValueError(String),          // Invalid value errors
    NotImplementedError(String), // Feature not yet implemented
    ConvergenceError(String),    // Algorithm convergence failures
    CoreError(CoreError),        // Errors from scirs2-core
}
```

### Error Context

Enhanced error messages with full context:

```rust
use scirs2_special::error_context::ErrorContext;

let ctx = ErrorContext::new("gamma", "evaluation")
    .with_param("x", -1.0)
    .with_info("Gamma function is undefined at negative integers");
```

## Validation Patterns

### Standard Validation Functions

All special functions use consistent validation from `scirs2_special::validation`:

```rust
// Check if value is positive
check_positive(x, "x")?;

// Check if value is finite
check_finite(x, "x")?;

// Check if value is in range
check_in_bounds(x, 0.0, 1.0, "x")?;

// Check probability values
check_probability(p, "p")?;

// Array validations
check_array_finite(&array, "input")?;
check_not_empty(&array, "input")?;
check_same_shape(&a, "a", &b, "b")?;
```

### Function-Specific Validation

Special functions have domain-specific validation:

```rust
// Bessel function order validation
check_order(n, "n")?;

// Spherical harmonics degree/order validation
check_degree(l, "l")?;
check_order_m(l, m)?;

// Convergence parameters
check_convergence_params(max_iter, tolerance)?;
```

## Error Handling Patterns

### Pattern 1: Direct Function with Result

```rust
use scirs2_special::{SpecialResult, gamma_safe};

pub fn gamma_safe(x: f64) -> SpecialResult<f64> {
    // Validate input
    validation::check_finite(x, "x")?;
    
    // Handle special cases
    if x == 0.0 {
        return Ok(f64::INFINITY); // Expected behavior
    }
    
    if x < 0.0 && x.fract() == 0.0 {
        return Err(special_error!(
            domain: "gamma", "evaluation",
            "x" => x
        ));
    }
    
    // Compute result
    let result = compute_gamma(x);
    
    // Validate output
    if result.is_nan() {
        return Err(special_error!(
            computation: "gamma", "evaluation failed",
            "x" => x
        ));
    }
    
    Ok(result)
}
```

### Pattern 2: Wrapped Functions

```rust
use scirs2_special::error_wrappers::wrapped::gamma_wrapped;

let gamma = gamma_wrapped();
let result = gamma.evaluate(5.0)?; // Returns Result<f64, SpecialError>
```

### Pattern 3: Array Operations

```rust
use scirs2_special::error_wrappers::ArrayWrapper;
use ndarray::arr1;

let arr_gamma = ArrayWrapper::new("gamma_array", |x: &ArrayView1<f64>| {
    x.mapv(gamma)
});

let input = arr1(&[1.0, 2.0, 3.0, 4.0]);
let result = arr_gamma.evaluate(&input)?;
```

## Error Recovery Strategies

### Recovery Options

```rust
pub enum RecoveryStrategy {
    ReturnDefault,      // Return a default value
    ClampToRange,       // Clamp input to valid range
    UseApproximation,   // Use an approximation
    PropagateError,     // Propagate the error (default)
}
```

### Configuring Recovery

```rust
use scirs2_special::error_wrappers::{ErrorConfig, RecoveryStrategy};

let config = ErrorConfig {
    enable_recovery: true,
    default_recovery: RecoveryStrategy::UseApproximation,
    log_errors: true,
    max_iterations: 1000,
    tolerance: 1e-10,
};

let gamma = gamma_wrapped().with_config(config);
```

## Macro Usage

### Error Creation Macros

```rust
// Domain error with context
let err = special_error!(
    domain: "bessel_j", "evaluation",
    "n" => 5,
    "x" => -10.0
);

// Convergence error
let err = special_error!(
    convergence: "hypergeometric", "series evaluation",
    "iterations" => 1000,
    "tolerance" => 1e-10
);

// Computation error
let err = special_error!(
    computation: "elliptic_k", "overflow",
    "m" => 0.9999999
);
```

### Validation Macros

```rust
// Validate with automatic error context
validate_with_context!(
    x > 0.0,
    DomainError,
    "beta",
    "Both arguments must be positive",
    "a" => a,
    "b" => b
);
```

## Best Practices

1. **Always validate inputs** before computation
2. **Use specific error types** (DomainError for domain issues, ConvergenceError for convergence)
3. **Include parameter values** in error messages for debugging
4. **Handle special cases explicitly** (e.g., gamma(0) = infinity)
5. **Validate outputs** when numerical issues might occur
6. **Use error context** for better debugging experience
7. **Document error conditions** in function documentation

## Migration Guide

### Converting Existing Functions

Before:
```rust
pub fn my_special_function(x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    // computation...
}
```

After:
```rust
pub fn my_special_function(x: f64) -> SpecialResult<f64> {
    validation::check_non_negative(x, "x")?;
    
    let result = // computation...
    
    if result.is_nan() {
        return Err(special_error!(
            computation: "my_special_function", "evaluation",
            "x" => x
        ));
    }
    
    Ok(result)
}
```

### Using Wrapped Versions

For backward compatibility, you can provide both versions:

```rust
// Safe version with error handling
pub fn gamma_safe(x: f64) -> SpecialResult<f64> {
    // ... implementation with full error handling
}

// Legacy version for compatibility
pub fn gamma(x: f64) -> f64 {
    gamma_safe(x).unwrap_or(f64::NAN)
}
```

## Testing Error Handling

```rust
#[test]
fn test_gamma_errors() {
    // Test domain error
    let result = gamma_safe(-1.0);
    assert!(matches!(result, Err(SpecialError::DomainError(_))));
    
    // Test finite input validation
    let result = gamma_safe(f64::NAN);
    assert!(matches!(result, Err(SpecialError::DomainError(_))));
    
    // Test valid input
    let result = gamma_safe(5.0);
    assert!(result.is_ok());
    assert!((result.unwrap() - 24.0).abs() < 1e-10);
}
```

## Performance Considerations

1. **Error handling overhead is minimal** for the happy path
2. **Validation is fast** compared to computation
3. **Use batch validation** for array operations
4. **Consider recovery strategies** for performance-critical code
5. **Profile with and without error handling** to measure impact

## Future Enhancements

1. **Structured error types** with specific fields for each error variant
2. **Error recovery database** with function-specific recovery strategies
3. **Automatic error reporting** integration
4. **Performance profiling** of error paths
5. **Machine learning-based** error prediction and prevention