# scirs2-stats Error Recovery Guide

This guide provides comprehensive strategies for handling and recovering from common errors in statistical computations.

## Table of Contents

1. [Error Types Overview](#error-types-overview)
2. [Common Error Patterns](#common-error-patterns)
3. [Recovery Strategies](#recovery-strategies)
4. [Best Practices](#best-practices)
5. [Code Examples](#code-examples)

## Error Types Overview

### StatsError Variants

- **ComputationError**: General computation failures
- **DomainError**: Input values outside valid domain
- **DimensionMismatch**: Incompatible array dimensions
- **InvalidArgument**: Invalid function arguments
- **NotImplementedError**: Feature not yet available
- **CoreError**: Errors from scirs2-core

## Common Error Patterns

### 1. NaN and Infinite Values

**Problem**: Data contains NaN or infinite values that break calculations.

**Detection**:
```rust
use scirs2_stats::error_suggestions::diagnose_error;

// Check data quality
let has_nan = data.iter().any(|&x| x.is_nan());
let has_inf = data.iter().any(|&x| x.is_infinite());
```

**Recovery Strategies**:

```rust
// Strategy 1: Filter out invalid values
let clean_data: Vec<f64> = data.iter()
    .filter(|&&x| x.is_finite())
    .copied()
    .collect();

// Strategy 2: Impute with mean
let valid_mean = data.iter()
    .filter(|&&x| x.is_finite())
    .sum::<f64>() / valid_count as f64;

let imputed = data.mapv(|x| if x.is_finite() { x } else { valid_mean });

// Strategy 3: Use robust statistics
use scirs2_stats::{median, mad};
let robust_center = median(&mut data.view_mut())?; // Median is robust to outliers
```

### 2. Empty Arrays

**Problem**: Functions receive empty input arrays.

**Recovery Strategies**:

```rust
// Always validate input
use scirs2_stats::error_messages::validation::ensure_not_empty;

ensure_not_empty(&data, "input data")?;

// Provide defaults for edge cases
let result = if data.is_empty() {
    Some(default_value)
} else {
    Some(compute_statistic(&data)?)
};

// Chain operations safely
let result = data_source
    .load()?
    .filter(|x| x > threshold)
    .map(|data| {
        if data.is_empty() {
            Err(StatsError::invalid_argument("No data after filtering"))
        } else {
            Ok(process_data(&data))
        }
    })?;
```

### 3. Dimension Mismatches

**Problem**: Arrays have incompatible shapes for the operation.

**Recovery Strategies**:

```rust
// Check dimensions before operations
use scirs2_stats::error_messages::validation::ensure_same_length;

ensure_same_length(&x, &y)?;

// Reshape arrays as needed
let y_reshaped = if x.shape() != y.shape() {
    y.reshape(x.shape())?
} else {
    y
};

// Use broadcasting when appropriate
use ndarray::ArrayBase;
let broadcasted = x.broadcast(y.shape())?;

// Transpose for matrix operations
let result = if a.shape()[1] != b.shape()[0] {
    a.dot(&b.t())  // Try transposed
} else {
    a.dot(&b)
};
```

### 4. Convergence Failures

**Problem**: Iterative algorithms fail to converge.

**Recovery Strategies**:

```rust
// Adaptive parameter adjustment
let mut max_iter = 1000;
let mut tolerance = 1e-8;
let mut result = Err(StatsError::computation("Not converged"));

for attempt in 0..3 {
    result = optimize_with_params(data, max_iter, tolerance);
    
    if result.is_ok() {
        break;
    }
    
    // Relax parameters
    max_iter *= 2;
    tolerance *= 10.0;
}

// Preprocessing for better conditioning
let scaled_data = standardize_data(&data)?;
let result = optimize(&scaled_data)?;
let final_result = unscale_result(result, &original_scale);

// Try alternative algorithms
let result = match primary_algorithm(&data) {
    Ok(res) => res,
    Err(_) => {
        eprintln!("Primary algorithm failed, trying fallback");
        fallback_algorithm(&data)?
    }
};
```

### 5. Singular Matrix Errors

**Problem**: Matrix is singular or near-singular.

**Recovery Strategies**:

```rust
// Add regularization
use ndarray::Array2;
let lambda = 1e-6;
let regularized = &matrix + lambda * Array2::eye(matrix.nrows());

// Use pseudo-inverse for singular matrices
use scirs2_linalg::pinv;
let pseudo_inverse = pinv(&matrix, 1e-10)?;

// Remove collinear features
use scirs2_stats::corrcoef;
let corr = corrcoef(&features.t(), "pearson")?;
let independent_features = remove_correlated_features(&features, &corr, 0.95);

// Use SVD for numerical stability
use scirs2_linalg::svd;
let (u, s, vt) = svd(&matrix, true, true)?;
let rank = s.iter().filter(|&&x| x > 1e-10).count();
```

### 6. Numerical Overflow

**Problem**: Calculations produce values too large to represent.

**Recovery Strategies**:

```rust
// Work in log space
let log_probs = values.mapv(|x| x.ln());
let log_sum = log_sum_exp(&log_probs);
let result = log_sum.exp();

// Scale data before operations
let scale = data.max()? - data.min()?;
let scaled = &data / scale;
let result = compute(&scaled)? * scale;

// Use stable algorithms
// Instead of: exp(a) / (exp(a) + exp(b))
// Use: 1.0 / (1.0 + exp(b - a))

// Clip extreme values
let clipped = data.mapv(|x| x.max(-1e100).min(1e100));
```

## Best Practices

### 1. Defensive Programming

```rust
use scirs2_stats::{StatsResult, StatsError};

pub fn robust_mean(data: &Array1<f64>) -> StatsResult<f64> {
    // Input validation
    if data.is_empty() {
        return Err(StatsError::invalid_argument("Cannot compute mean of empty array"));
    }
    
    // Check for special cases
    let finite_count = data.iter().filter(|&&x| x.is_finite()).count();
    if finite_count == 0 {
        return Err(StatsError::invalid_argument("No finite values in array"));
    }
    
    // Compute with fallback
    let sum = data.iter()
        .filter(|&&x| x.is_finite())
        .fold(0.0, |acc, &x| acc + x);
    
    Ok(sum / finite_count as f64)
}
```

### 2. Error Context

```rust
use scirs2_stats::error_context::EnhancedError;

pub fn complex_operation(data: &Array2<f64>) -> StatsResult<Array1<f64>> {
    let step1 = compute_step1(data)
        .map_err(|e| EnhancedError::new(e, "Failed in step 1")
            .with_suggestion("Check input data format")
            .into_error())?;
    
    let step2 = compute_step2(&step1)
        .map_err(|e| EnhancedError::new(e, "Failed in step 2")
            .with_suggestion("Try preprocessing data")
            .into_error())?;
    
    Ok(step2)
}
```

### 3. Progressive Fallbacks

```rust
pub fn compute_with_fallbacks(data: &Array1<f64>) -> StatsResult<f64> {
    // Try optimal algorithm first
    if let Ok(result) = optimal_algorithm(data) {
        return Ok(result);
    }
    
    // Fall back to robust but slower algorithm
    if let Ok(result) = robust_algorithm(data) {
        eprintln!("Warning: Using slower robust algorithm");
        return Ok(result);
    }
    
    // Last resort: simple approximation
    simple_approximation(data)
        .map_err(|_| StatsError::computation("All algorithms failed"))
}
```

### 4. Error Logging and Monitoring

```rust
use scirs2_stats::error_suggestions::{ErrorFormatter, diagnose_error};

pub fn monitored_computation(data: &Array1<f64>) -> StatsResult<f64> {
    match perform_computation(data) {
        Ok(result) => Ok(result),
        Err(e) => {
            // Diagnose the error
            let diagnosis = diagnose_error(&e);
            
            // Log detailed information
            eprintln!("Error Type: {:?}", diagnosis.error_type);
            eprintln!("Severity: {:?}", diagnosis.severity);
            eprintln!("Likely causes: {:?}", diagnosis.likely_causes);
            
            // Format with suggestions
            let formatter = ErrorFormatter::new();
            eprintln!("{}", formatter.format_error(e.clone(), Some("computation")));
            
            Err(e)
        }
    }
}
```

## Code Examples

### Complete Error Handling Example

```rust
use scirs2_stats::{
    StatsResult, StatsError, 
    error_suggestions::{ErrorFormatter, diagnose_error},
    error_context::EnhancedError,
    mean, std, correlation
};
use ndarray::Array1;

pub struct RobustAnalysis;

impl RobustAnalysis {
    pub fn analyze(data: &Array1<f64>) -> StatsResult<AnalysisResult> {
        // Step 1: Validate input
        Self::validate_input(data)?;
        
        // Step 2: Clean data
        let clean_data = Self::clean_data(data)?;
        
        // Step 3: Compute statistics with error handling
        let stats = Self::compute_statistics(&clean_data)?;
        
        Ok(AnalysisResult {
            mean: stats.0,
            std: stats.1,
            n_valid: clean_data.len(),
            n_removed: data.len() - clean_data.len(),
        })
    }
    
    fn validate_input(data: &Array1<f64>) -> StatsResult<()> {
        if data.is_empty() {
            return Err(EnhancedError::new(
                StatsError::invalid_argument("Empty input"),
                "Input validation failed"
            )
            .with_suggestions(vec![
                "Check data loading process",
                "Verify file exists and is readable",
                "Ensure filters aren't too restrictive",
            ])
            .into_error());
        }
        
        Ok(())
    }
    
    fn clean_data(data: &Array1<f64>) -> StatsResult<Array1<f64>> {
        let valid_data: Vec<f64> = data.iter()
            .filter(|&&x| x.is_finite())
            .copied()
            .collect();
        
        if valid_data.is_empty() {
            return Err(EnhancedError::new(
                StatsError::invalid_argument("No valid data after cleaning"),
                "Data cleaning resulted in empty dataset"
            )
            .with_suggestions(vec![
                "Check for data corruption",
                "Review data collection process",
                "Consider imputation strategies",
            ])
            .into_error());
        }
        
        if valid_data.len() < data.len() / 2 {
            eprintln!("Warning: Removed {} invalid values ({}% of data)",
                data.len() - valid_data.len(),
                ((data.len() - valid_data.len()) * 100) / data.len()
            );
        }
        
        Ok(Array1::from_vec(valid_data))
    }
    
    fn compute_statistics(data: &Array1<f64>) -> StatsResult<(f64, f64)> {
        let mean_val = mean(&data.view())?;
        let std_val = std(&data.view(), 1)?;
        
        // Validate results
        if !mean_val.is_finite() || !std_val.is_finite() {
            return Err(StatsError::computation(
                "Numerical error in statistics computation"
            ));
        }
        
        Ok((mean_val, std_val))
    }
}

pub struct AnalysisResult {
    pub mean: f64,
    pub std: f64,
    pub n_valid: usize,
    pub n_removed: usize,
}

// Usage example with comprehensive error handling
fn main() {
    let data = Array1::from_vec(vec![1.0, 2.0, f64::NAN, 4.0, 5.0]);
    
    match RobustAnalysis::analyze(&data) {
        Ok(result) => {
            println!("Analysis successful:");
            println!("  Mean: {:.2}", result.mean);
            println!("  Std: {:.2}", result.std);
            println!("  Valid samples: {}", result.n_valid);
            println!("  Removed samples: {}", result.n_removed);
        }
        Err(e) => {
            // Use error formatter for detailed output
            let formatter = ErrorFormatter::new();
            eprintln!("{}", formatter.format_error(e, Some("analysis")));
            
            // Could also implement retry logic here
        }
    }
}
```

## Summary

Effective error handling in statistical computing requires:

1. **Proactive validation** - Check inputs before computation
2. **Graceful degradation** - Provide fallbacks when possible
3. **Clear communication** - Give users actionable error messages
4. **Defensive programming** - Anticipate edge cases
5. **Recovery strategies** - Implement robust alternatives

By following these patterns and using the provided error handling infrastructure, you can build robust statistical applications that handle errors gracefully and provide users with helpful guidance for resolution.