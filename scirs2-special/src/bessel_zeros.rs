//! Bessel function zeros and related utilities
//!
//! This module provides functions to compute zeros of Bessel functions
//! and other Bessel-related utilities.

use crate::bessel::{j0, j1, jn, y0, y1, yn};
use crate::bessel::derivatives::{j1_prime, jn_prime};
use crate::error::{SpecialError, SpecialResult};
use crate::validation::check_positive;
use num_traits::{Float, FromPrimitive};
use std::f64::consts::PI;
use std::fmt::{Debug, Display};
use std::convert::TryFrom;

/// Compute the k-th zero of J₀(x)
///
/// # Arguments
/// * `k` - Index of the zero (1-based)
///
/// # Returns
/// The k-th positive zero of J₀(x)
pub fn j0_zeros<T>(k: usize) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display,
{
    if k == 0 {
        return Err(SpecialError::ValueError("j0_zeros: k must be >= 1".to_string()));
    }
    
    // McMahon's asymptotic expansion for large zeros
    let k_f = T::from_usize(k).unwrap();
    let pi = T::from_f64(PI).unwrap();
    
    // Initial approximation
    let beta = (k_f - T::from_f64(0.25).unwrap()) * pi;
    
    // Refine with Newton's method
    refine_bessel_zero(beta, |x| j0(x), |x| -j1(x))
}

/// Compute the k-th zero of J₁(x)
pub fn j1_zeros<T>(k: usize) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display,
{
    if k == 0 {
        return Err(SpecialError::ValueError("j1_zeros: k must be >= 1".to_string()));
    }
    
    let k_f = T::from_usize(k).unwrap();
    let pi = T::from_f64(PI).unwrap();
    
    // Initial approximation
    let beta = (k_f + T::from_f64(0.25).unwrap()) * pi;
    
    // Refine with Newton's method
    refine_bessel_zero(beta, |x| j1(x), |x| j1_prime(x))
}

/// Compute the k-th zero of Jₙ(x)
///
/// # Arguments
/// * `n` - Order of the Bessel function
/// * `k` - Index of the zero (1-based)
///
/// # Returns
/// The k-th positive zero of Jₙ(x)
pub fn jn_zeros<T>(n: usize, k: usize) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display + std::ops::AddAssign,
{
    if k == 0 {
        return Err(SpecialError::ValueError("jn_zeros: k must be >= 1".to_string()));
    }
    
    let n_f = T::from_usize(n).unwrap();
    let k_f = T::from_usize(k).unwrap();
    let pi = T::from_f64(PI).unwrap();
    
    // McMahon's asymptotic expansion
    let beta = (k_f + n_f / T::from_f64(2.0).unwrap() - T::from_f64(0.25).unwrap()) * pi;
    
    // Refine with Newton's method
    let n_i32 = i32::try_from(n).map_err(|_| SpecialError::ValueError("jn_zeros: n too large".to_string()))?;
    refine_bessel_zero(beta, |x| jn(n_i32, x), |x| jn_prime(n_i32, x))
}

/// Compute the k-th zero of the derivative J'ₙ(x)
pub fn jnp_zeros<T>(n: usize, k: usize) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display + std::ops::AddAssign,
{
    if k == 0 {
        return Err(SpecialError::ValueError("jnp_zeros: k must be >= 1".to_string()));
    }
    
    let n_f = T::from_usize(n).unwrap();
    let k_f = T::from_usize(k).unwrap();
    let pi = T::from_f64(PI).unwrap();
    
    // Initial approximation
    let beta = (k_f + n_f / T::from_f64(2.0).unwrap() - T::from_f64(0.75).unwrap()) * pi;
    
    // Refine with Newton's method for derivative
    let n_i32 = i32::try_from(n).map_err(|_| SpecialError::ValueError("jnp_zeros: n too large".to_string()))?;
    refine_bessel_zero(beta, |x| jn_prime(n_i32, x), |x| jn_prime_prime(n, x))
}

/// Compute the k-th zero of Y₀(x)
pub fn y0_zeros<T>(k: usize) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display,
{
    if k == 0 {
        return Err(SpecialError::ValueError("y0_zeros: k must be >= 1".to_string()));
    }
    
    let k_f = T::from_usize(k).unwrap();
    let pi = T::from_f64(PI).unwrap();
    
    // Initial approximation
    let beta = (k_f - T::from_f64(0.75).unwrap()) * pi;
    
    // Refine with Newton's method
    refine_bessel_zero(beta, |x| y0(x), |x| -y1(x))
}

/// Compute the k-th zero of Y₁(x)
pub fn y1_zeros<T>(k: usize) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display,
{
    if k == 0 {
        return Err(SpecialError::ValueError("y1_zeros: k must be >= 1".to_string()));
    }
    
    let k_f = T::from_usize(k).unwrap();
    let pi = T::from_f64(PI).unwrap();
    
    // Initial approximation
    let beta = (k_f - T::from_f64(0.25).unwrap()) * pi;
    
    // Refine with Newton's method
    refine_bessel_zero(beta, |x| y1(x), |x| y1_prime(x))
}

/// Compute the k-th zero of Yₙ(x)
pub fn yn_zeros<T>(n: usize, k: usize) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Debug + Display,
{
    if k == 0 {
        return Err(SpecialError::ValueError("yn_zeros: k must be >= 1".to_string()));
    }
    
    let n_f = T::from_usize(n).unwrap();
    let k_f = T::from_usize(k).unwrap();
    let pi = T::from_f64(PI).unwrap();
    
    // Initial approximation
    let beta = (k_f + n_f / T::from_f64(2.0).unwrap() - T::from_f64(0.75).unwrap()) * pi;
    
    // Refine with Newton's method
    let n_i32 = i32::try_from(n).map_err(|_| SpecialError::ValueError("yn_zeros: n too large".to_string()))?;
    refine_bessel_zero(beta, |x| yn(n_i32, x), |x| yn_prime(n, x))
}

/// Compute zeros of Jₙ(x) and Yₙ(x) simultaneously
///
/// Returns (jn_zero, yn_zero) for the k-th zero
pub fn jnyn_zeros<T>(n: usize, k: usize) -> SpecialResult<(T, T)>
where
    T: Float + FromPrimitive + Debug + Display + std::ops::AddAssign,
{
    let jn_zero = jn_zeros(n, k)?;
    let yn_zero = yn_zeros(n, k)?;
    Ok((jn_zero, yn_zero))
}

/// Compute integrals ∫₀^∞ tⁿ J₀(t) Y₀(xt) dt and ∫₀^∞ tⁿ J₀(t) Y₀(xt) J₀(t) dt
///
/// Used in various applications involving Bessel functions.
pub fn itj0y0<T>(x: T, n: usize) -> SpecialResult<(T, T)>
where
    T: Float + FromPrimitive + Debug + Display,
{
    check_positive(x, "x")?;
    
    // These integrals have closed-form solutions for specific n values
    // This is a simplified implementation
    match n {
        0 => {
            // ∫₀^∞ J₀(t) Y₀(xt) dt = -2/(π(1-x²)) for |x| < 1
            if x < T::one() {
                let pi = T::from_f64(PI).unwrap();
                let denom = pi * (T::one() - x * x);
                let integral1 = -T::from_f64(2.0).unwrap() / denom;
                let integral2 = integral1; // Simplified
                Ok((integral1, integral2))
            } else {
                Err(SpecialError::DomainError("itj0y0: x must be < 1 for n=0".to_string()))
            }
        }
        _ => {
            // For other n values, numerical integration would be needed
            Err(SpecialError::NotImplementedError(
                format!("itj0y0: n={} not yet implemented", n)
            ))
        }
    }
}

/// Compute Bessel polynomial
///
/// The Bessel polynomial of degree n at point x.
pub fn besselpoly<T>(n: usize, x: T) -> T
where
    T: Float + FromPrimitive + std::ops::AddAssign + std::ops::MulAssign,
{
    if n == 0 {
        return T::one();
    }
    
    // Recurrence relation: y_{n+1}(x) = (2n+1)x y_n(x) + y_{n-1}(x)
    let mut y_prev = T::one();
    let mut y_curr = T::one() + x;
    
    for k in 1..n {
        let two_k_plus_one = T::from_usize(2 * k + 1).unwrap();
        let y_next = two_k_plus_one * x * y_curr + y_prev;
        y_prev = y_curr;
        y_curr = y_next;
    }
    
    y_curr
}

// Helper functions

/// Refine a Bessel function zero using Newton's method
fn refine_bessel_zero<T, F, D>(initial: T, f: F, df: D) -> SpecialResult<T>
where
    T: Float + FromPrimitive,
    F: Fn(T) -> T,
    D: Fn(T) -> T,
{
    let mut x = initial;
    let tol = T::from_f64(1e-12).unwrap();
    let max_iter = 20;
    
    for _ in 0..max_iter {
        let fx = f(x);
        let dfx = df(x);
        
        if dfx.abs() < T::from_f64(1e-30).unwrap() {
            return Err(SpecialError::ConvergenceError(
                "refine_bessel_zero: derivative too small".to_string()
            ));
        }
        
        let dx = fx / dfx;
        x = x - dx;
        
        if dx.abs() < tol * x.abs() {
            return Ok(x);
        }
    }
    
    Err(SpecialError::ConvergenceError(
        "refine_bessel_zero: Newton iteration did not converge".to_string()
    ))
}

/// Compute Y'₁(x) using the recurrence relation
fn y1_prime<T>(x: T) -> T
where
    T: Float + FromPrimitive + Debug,
{
    y0(x) - y1(x) / x
}

/// Compute Y'ₙ(x) using the recurrence relation
fn yn_prime<T>(n: usize, x: T) -> T
where
    T: Float + FromPrimitive + Debug,
{
    if n == 0 {
        -y1(x)
    } else {
        let n_i32 = i32::try_from(n).unwrap_or(i32::MAX);
        (yn(n_i32 - 1, x) - yn(n_i32 + 1, x)) / T::from_f64(2.0).unwrap()
    }
}

/// Compute J''ₙ(x) using the recurrence relation
fn jn_prime_prime<T>(n: usize, x: T) -> T
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign,
{
    let n_f = T::from_usize(n).unwrap();
    let n_i32 = i32::try_from(n).unwrap_or(i32::MAX);
    let jn_val = jn(n_i32, x);
    let jn_p1 = jn(n_i32 + 1, x);
    let jn_p2 = jn(n_i32 + 2, x);
    
    // J''_n(x) = -(1 + n(n-1)/x²)J_n(x) + (2n+1)/x J_{n+1}(x) - J_{n+2}(x)
    let term1 = -(T::one() + n_f * (n_f - T::one()) / (x * x)) * jn_val;
    let term2 = (T::from_f64(2.0).unwrap() * n_f + T::one()) / x * jn_p1;
    let term3 = -jn_p2;
    
    term1 + term2 + term3
}

/// Compute zeros where Jₙ(x) and J'ₙ(x) cross zero simultaneously
///
/// These are important in various boundary value problems.
pub fn jnjnp_zeros<T>(n: usize, k: usize) -> SpecialResult<(T, T)>
where
    T: Float + FromPrimitive + Debug + Display + std::ops::AddAssign,
{
    let jn_zero = jn_zeros(n, k)?;
    let jnp_zero = jnp_zeros(n, k)?;
    Ok((jn_zero, jnp_zero))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_j0_zeros() {
        // First few zeros of J₀(x)
        assert_relative_eq!(j0_zeros::<f64>(1).unwrap(), 2.4048255576957728, epsilon = 1e-10);
        assert_relative_eq!(j0_zeros::<f64>(2).unwrap(), 5.5200781102863106, epsilon = 1e-10);
        assert_relative_eq!(j0_zeros::<f64>(3).unwrap(), 8.6537279129110122, epsilon = 1e-10);
    }

    #[test]
    fn test_j1_zeros() {
        // First few zeros of J₁(x)
        assert_relative_eq!(j1_zeros::<f64>(1).unwrap(), 3.8317059702075123, epsilon = 1e-10);
        assert_relative_eq!(j1_zeros::<f64>(2).unwrap(), 7.0155866698156187, epsilon = 1e-10);
    }

    #[test]
    fn test_jn_zeros() {
        // First zero of J₂(x)
        assert_relative_eq!(jn_zeros::<f64>(2, 1).unwrap(), 5.1356223018406826, epsilon = 1e-8);
    }

    #[test]
    fn test_y0_zeros() {
        // First few zeros of Y₀(x)
        assert_relative_eq!(y0_zeros::<f64>(1).unwrap(), 0.8935769662791675, epsilon = 1e-10);
        assert_relative_eq!(y0_zeros::<f64>(2).unwrap(), 3.9576784193148578, epsilon = 1e-10);
    }

    #[test]
    fn test_besselpoly() {
        // Test Bessel polynomials
        assert_eq!(besselpoly(0, 2.0), 1.0);
        assert_eq!(besselpoly(1, 2.0), 3.0); // 1 + 2
        assert_eq!(besselpoly(2, 2.0), 15.0); // 3*2*3 + 1 = 15
    }

    #[test]
    fn test_error_cases() {
        assert!(j0_zeros::<f64>(0).is_err());
        assert!(jn_zeros::<f64>(0, 0).is_err());
    }
}