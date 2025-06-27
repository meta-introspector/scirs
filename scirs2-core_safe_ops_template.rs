// Template for scirs2-core/src/safe_ops.rs

use crate::error::{Scirs2_CoreError, Scirs2_CoreResult};
use num_traits::{Float, Zero};
use std::fmt::Display;

/// Safe division with zero checking
pub fn safe_divide<T: Float + Display>(num: T, denom: T) -> Scirs2_CoreResult<T> {
    if denom.abs() < T::epsilon() {
        return Err(Scirs2_CoreError::DomainError(
            format!("Division by zero or near-zero: {} / {}", num, denom)
        ));
    }
    
    let result = num / denom;
    if !result.is_finite() {
        return Err(Scirs2_CoreError::ComputationError(
            format!("Division produced non-finite result: {:?}", result)
        ));
    }
    
    Ok(result)
}

/// Safe square root with domain checking
pub fn safe_sqrt<T: Float + Display>(value: T) -> Scirs2_CoreResult<T> {
    if value < T::zero() {
        return Err(Scirs2_CoreError::DomainError(
            format!("Cannot compute sqrt of negative value: {}", value)
        ));
    }
    
    Ok(value.sqrt())
}

// Add more safe operations as needed...
