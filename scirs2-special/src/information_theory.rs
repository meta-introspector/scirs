//! Information theory functions
//!
//! This module provides functions related to information theory, including
//! entropy, Kullback-Leibler divergence, and Huber loss functions.

use crate::error::{SpecialError, SpecialResult};
use crate::validation::{check_finite, check_non_negative};
use ndarray::{Array1, ArrayView1, ArrayViewMut1};
use num_traits::{Float, FromPrimitive, Zero};
use std::fmt::{Debug, Display};

/// Shannon entropy function
///
/// Computes -x * log(x) for x > 0, and 0 for x = 0.
/// This is the entropy contribution of a single probability.
///
/// # Arguments
/// * `x` - Input value (typically a probability)
///
/// # Returns
/// The entropy value -x * log(x)
///
/// # Examples
/// ```
/// use scirs2_special::information_theory::entr;
/// 
/// let h = entr(0.5);
/// assert!((h - 0.34657359027997264).abs() < 1e-10);
/// ```
pub fn entr<T>(x: T) -> T
where
    T: Float + FromPrimitive + Zero,
{
    if x.is_zero() {
        T::zero()
    } else if x < T::zero() {
        -T::infinity()
    } else {
        -x * x.ln()
    }
}

/// Relative entropy (Kullback-Leibler divergence term)
///
/// Computes x * log(x/y) for x > 0, y > 0.
/// Special cases:
/// - If x = 0, returns 0
/// - If y = 0 and x > 0, returns infinity
///
/// # Arguments
/// * `x` - First probability
/// * `y` - Second probability
///
/// # Returns
/// The relative entropy term x * log(x/y)
pub fn rel_entr<T>(x: T, y: T) -> T
where
    T: Float + FromPrimitive + Zero,
{
    if x.is_zero() {
        T::zero()
    } else if y.is_zero() {
        T::infinity()
    } else {
        x * (x / y).ln()
    }
}

/// Kullback-Leibler divergence
///
/// Computes the KL divergence: x * log(x/y) - x + y
/// This is a symmetrized version that's more numerically stable.
///
/// # Arguments
/// * `x` - First value
/// * `y` - Second value
///
/// # Returns
/// The KL divergence value
pub fn kl_div<T>(x: T, y: T) -> T
where
    T: Float + FromPrimitive + Zero,
{
    if x.is_zero() {
        y
    } else if y.is_zero() {
        T::infinity()
    } else {
        x * (x / y).ln() - x + y
    }
}

/// Huber loss function
///
/// The Huber loss is a robust loss function that behaves like squared error
/// for small errors and like absolute error for large errors.
///
/// huber(δ, r) = { r²/2           if |r| <= δ
///               { δ|r| - δ²/2    if |r| > δ
///
/// # Arguments
/// * `delta` - Threshold parameter (must be positive)
/// * `r` - Residual value
///
/// # Returns
/// The Huber loss value
pub fn huber<T>(delta: T, r: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display,
{
    check_finite(delta, "delta")?;
    check_finite(r, "r")?;
    
    if delta <= T::zero() {
        return Err(SpecialError::DomainError("huber: delta must be positive".to_string()));
    }
    
    let abs_r = r.abs();
    
    if abs_r <= delta {
        Ok(r * r / T::from_f64(2.0).unwrap())
    } else {
        Ok(delta * abs_r - delta * delta / T::from_f64(2.0).unwrap())
    }
}

/// Pseudo-Huber loss function
///
/// A smooth approximation to the Huber loss function:
/// pseudo_huber(δ, r) = δ² * (sqrt(1 + (r/δ)²) - 1)
///
/// # Arguments
/// * `delta` - Scale parameter (must be positive)
/// * `r` - Residual value
///
/// # Returns
/// The pseudo-Huber loss value
pub fn pseudo_huber<T>(delta: T, r: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display,
{
    check_finite(delta, "delta")?;
    check_finite(r, "r")?;
    
    if delta <= T::zero() {
        return Err(SpecialError::DomainError("pseudo_huber: delta must be positive".to_string()));
    }
    
    let r_over_delta = r / delta;
    let delta_squared = delta * delta;
    
    Ok(delta_squared * ((T::one() + r_over_delta * r_over_delta).sqrt() - T::one()))
}

/// Apply entropy function to array
///
/// Computes -x * log(x) element-wise for an array.
pub fn entr_array<T>(x: &ArrayView1<T>) -> Array1<T>
where
    T: Float + FromPrimitive + Zero + Send + Sync,
{
    x.mapv(entr)
}

/// Compute total entropy of a probability distribution
///
/// Computes H(p) = -Σ p_i * log(p_i)
///
/// # Arguments
/// * `p` - Probability distribution (must sum to 1)
///
/// # Returns
/// The Shannon entropy of the distribution
pub fn entropy<T>(p: &ArrayView1<T>) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Zero + Display + Debug,
{
    // Check that probabilities are non-negative
    for &pi in p.iter() {
        check_non_negative(pi, "probability")?;
    }
    
    // Compute entropy
    let mut h = T::zero();
    for &pi in p.iter() {
        h = h + entr(pi);
    }
    
    Ok(h)
}

/// Compute KL divergence between two probability distributions
///
/// Computes D_KL(p || q) = Σ p_i * log(p_i / q_i)
///
/// # Arguments
/// * `p` - First probability distribution
/// * `q` - Second probability distribution
///
/// # Returns
/// The KL divergence from q to p
pub fn kl_divergence<T>(p: &ArrayView1<T>, q: &ArrayView1<T>) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Zero + Display + Debug,
{
    if p.len() != q.len() {
        return Err(SpecialError::ValueError(
            "kl_divergence: arrays must have the same length".to_string()
        ));
    }
    
    let mut kl = T::zero();
    for i in 0..p.len() {
        kl = kl + rel_entr(p[i], q[i]);
    }
    
    Ok(kl)
}

/// Apply Huber loss to arrays of predictions and targets
///
/// # Arguments
/// * `delta` - Threshold parameter
/// * `predictions` - Predicted values
/// * `targets` - True values
/// * `output` - Output array for losses
pub fn huber_loss<T>(
    delta: T,
    predictions: &ArrayView1<T>,
    targets: &ArrayView1<T>,
    output: &mut ArrayViewMut1<T>,
) -> SpecialResult<()>
where
    T: Float + FromPrimitive + Display + Debug,
{
    if predictions.len() != targets.len() || predictions.len() != output.len() {
        return Err(SpecialError::ValueError(
            "huber_loss: all arrays must have the same length".to_string()
        ));
    }
    
    for i in 0..predictions.len() {
        let residual = predictions[i] - targets[i];
        output[i] = huber(delta, residual)?;
    }
    
    Ok(())
}

/// Binary entropy function
///
/// Computes the binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)
///
/// # Arguments
/// * `p` - Probability value in [0, 1]
///
/// # Returns
/// The binary entropy
pub fn binary_entropy<T>(p: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display,
{
    crate::validation::check_probability(p, "p")?;
    
    if p.is_zero() || p == T::one() {
        return Ok(T::zero());
    }
    
    Ok(entr(p) + entr(T::one() - p))
}

/// Cross entropy between two probability distributions
///
/// Computes H(p, q) = -Σ p_i * log(q_i)
///
/// # Arguments
/// * `p` - True probability distribution
/// * `q` - Predicted probability distribution
///
/// # Returns
/// The cross entropy
pub fn cross_entropy<T>(p: &ArrayView1<T>, q: &ArrayView1<T>) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Zero + Display + Debug,
{
    if p.len() != q.len() {
        return Err(SpecialError::ValueError(
            "cross_entropy: arrays must have the same length".to_string()
        ));
    }
    
    let mut ce = T::zero();
    for i in 0..p.len() {
        if p[i] > T::zero() {
            if q[i].is_zero() {
                return Ok(T::infinity());
            }
            ce = ce - p[i] * q[i].ln();
        }
    }
    
    Ok(ce)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_entr() {
        assert_eq!(entr(0.0), 0.0);
        assert_relative_eq!(entr(0.5), 0.34657359027997264, epsilon = 1e-10);
        assert_relative_eq!(entr(1.0), 0.0, epsilon = 1e-10);
        assert!(entr(-1.0).is_infinite() && entr(-1.0) < 0.0);
    }

    #[test]
    fn test_rel_entr() {
        assert_eq!(rel_entr(0.0, 1.0), 0.0);
        assert!(rel_entr(1.0, 0.0).is_infinite());
        assert_relative_eq!(rel_entr(0.5, 0.5), 0.0, epsilon = 1e-10);
        assert_relative_eq!(rel_entr(0.7, 0.3), 0.5995732273553991, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_div() {
        assert_eq!(kl_div(0.0, 1.0), 1.0);
        assert!(kl_div(1.0, 0.0).is_infinite());
        assert_relative_eq!(kl_div(0.5, 0.5), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_huber() {
        let delta = 1.0;
        
        // Small residuals (quadratic region)
        assert_relative_eq!(huber(delta, 0.5).unwrap(), 0.125, epsilon = 1e-10);
        assert_relative_eq!(huber(delta, -0.5).unwrap(), 0.125, epsilon = 1e-10);
        
        // Large residuals (linear region)
        assert_relative_eq!(huber(delta, 2.0).unwrap(), 1.5, epsilon = 1e-10);
        assert_relative_eq!(huber(delta, -2.0).unwrap(), 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_pseudo_huber() {
        let delta = 1.0;
        
        assert_relative_eq!(pseudo_huber(delta, 0.0).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(pseudo_huber(delta, 1.0).unwrap(), 0.41421356237309515, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy() {
        let uniform = arr1(&[0.25, 0.25, 0.25, 0.25]);
        let h = entropy(&uniform.view()).unwrap();
        assert_relative_eq!(h, 1.3862943611198906, epsilon = 1e-10); // log(4)
        
        let certain = arr1(&[1.0, 0.0, 0.0, 0.0]);
        let h = entropy(&certain.view()).unwrap();
        assert_relative_eq!(h, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_divergence() {
        let p = arr1(&[0.5, 0.5]);
        let q = arr1(&[0.9, 0.1]);
        let kl = kl_divergence(&p.view(), &q.view()).unwrap();
        assert!(kl > 0.0); // KL divergence is always non-negative
    }

    #[test]
    fn test_binary_entropy() {
        assert_eq!(binary_entropy(0.0).unwrap(), 0.0);
        assert_eq!(binary_entropy(1.0).unwrap(), 0.0);
        assert_relative_eq!(binary_entropy(0.5).unwrap(), 0.6931471805599453, epsilon = 1e-10); // log(2)
    }
}