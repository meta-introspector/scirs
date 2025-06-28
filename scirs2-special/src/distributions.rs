//! Statistical distribution functions
//!
//! This module provides cumulative distribution functions (CDFs), their complements,
//! and their inverses for various statistical distributions, matching SciPy's special module.

use crate::error::{SpecialError, SpecialResult};
use crate::gamma::{betainc_regularized, gamma};
use crate::validation::{check_finite, check_probability};
use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::f64::consts::PI;
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, SubAssign, MulAssign};


// Normal distribution functions

/// Normal cumulative distribution function
///
/// Computes the cumulative distribution function of the standard normal distribution.
///
/// # Arguments
/// * `x` - The point at which to evaluate the CDF
///
/// # Returns
/// The probability P(X <= x) where X ~ N(0, 1)
///
/// # Examples
/// ```
/// use scirs2_special::distributions::ndtr;
/// 
/// let p = ndtr(0.0);
/// assert!((p - 0.5).abs() < 1e-10);
/// ```
pub fn ndtr<T: Float + FromPrimitive>(x: T) -> T {
    // Use the error function: ndtr(x) = 0.5 * (1 + erf(x/sqrt(2)))
    let sqrt2 = T::from_f64(std::f64::consts::SQRT_2).unwrap();
    let half = T::from_f64(0.5).unwrap();
    let one = T::one();
    
    half * (one + crate::erf::erf(x / sqrt2))
}

/// Log of normal cumulative distribution function
///
/// Computes log(ndtr(x)) accurately for negative x values.
///
/// # Arguments
/// * `x` - The point at which to evaluate log(CDF)
///
/// # Returns
/// log(P(X <= x)) where X ~ N(0, 1)
pub fn log_ndtr<T: Float + FromPrimitive>(x: T) -> T {
    if x >= T::zero() {
        ndtr(x).ln()
    } else {
        // For negative x, use log(ndtr(x)) = log(erfc(-x/sqrt(2))) - log(2)
        let sqrt2 = T::from_f64(std::f64::consts::SQRT_2).unwrap();
        let log2 = T::from_f64(std::f64::consts::LN_2).unwrap();
        
        crate::erf::erfc(-x / sqrt2).ln() - log2
    }
}

/// Inverse of normal cumulative distribution function
///
/// Computes the inverse of the standard normal CDF (quantile function).
///
/// # Arguments
/// * `p` - Probability value in [0, 1]
///
/// # Returns
/// The value x such that P(X <= x) = p where X ~ N(0, 1)
///
/// # Errors
/// Returns an error if p is not in [0, 1]
pub fn ndtri<T: Float + FromPrimitive + Display>(p: T) -> SpecialResult<T> {
    check_probability(p, "p")?;
    
    // Use the inverse error function: ndtri(p) = sqrt(2) * erfinv(2*p - 1)
    let sqrt2 = T::from_f64(std::f64::consts::SQRT_2).unwrap();
    let two = T::from_f64(2.0).unwrap();
    let one = T::one();
    
    Ok(sqrt2 * crate::erf::erfinv(two * p - one))
}

/// Exponentially scaled inverse normal CDF
///
/// Computes ndtri(exp(y)) for y in [-infinity, 0], useful for log-probability calculations.
pub fn ndtri_exp<T: Float + FromPrimitive + Display>(y: T) -> SpecialResult<T> {
    if y > T::zero() {
        return Err(SpecialError::DomainError("ndtri_exp: y must be <= 0".to_string()));
    }
    
    let p = y.exp();
    ndtri(p)
}

// Binomial distribution functions

/// Binomial cumulative distribution function
///
/// Computes the cumulative distribution function of the binomial distribution.
///
/// # Arguments
/// * `k` - Number of successes (0 <= k <= n)
/// * `n` - Number of trials
/// * `p` - Probability of success in each trial
///
/// # Returns
/// P(X <= k) where X ~ Binomial(n, p)
pub fn bdtr<T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign>(k: usize, n: usize, p: T) -> SpecialResult<T> {
    check_probability(p, "p")?;
    
    if k >= n {
        return Ok(T::one());
    }
    
    // Use the regularized incomplete beta function
    // P(X <= k) = I_{1-p}(n-k, k+1)
    let n_minus_k = T::from_usize(n - k).unwrap();
    let k_plus_1 = T::from_usize(k + 1).unwrap();
    let one_minus_p = T::one() - p;
    
    betainc_regularized(n_minus_k, k_plus_1, one_minus_p)
}

/// Binomial survival function (complement of CDF)
///
/// # Arguments
/// * `k` - Number of successes
/// * `n` - Number of trials  
/// * `p` - Probability of success
///
/// # Returns
/// P(X > k) where X ~ Binomial(n, p)
pub fn bdtrc<T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign>(k: usize, n: usize, p: T) -> SpecialResult<T> {
    check_probability(p, "p")?;
    
    if k >= n {
        return Ok(T::zero());
    }
    
    // P(X > k) = I_p(k+1, n-k)
    let k_plus_1 = T::from_usize(k + 1).unwrap();
    let n_minus_k = T::from_usize(n - k).unwrap();
    
    betainc_regularized(k_plus_1, n_minus_k, p)
}

/// Inverse of binomial CDF with respect to k
///
/// Find k such that bdtr(k, n, p) = y
pub fn bdtri<T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign>(n: usize, p: T, y: T) -> SpecialResult<usize> {
    check_probability(p, "p")?;
    check_probability(y, "y")?;
    
    // Binary search for k
    let mut low = 0;
    let mut high = n;
    
    while low < high {
        let mid = (low + high) / 2;
        let cdf = bdtr(mid, n, p)?;
        
        if cdf < y {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    
    Ok(low)
}

// Poisson distribution functions

/// Poisson cumulative distribution function
///
/// # Arguments
/// * `k` - Number of events
/// * `lambda` - Rate parameter (mean)
///
/// # Returns
/// P(X <= k) where X ~ Poisson(lambda)
pub fn pdtr<T: Float + FromPrimitive + Display + Debug + AddAssign>(k: usize, lambda: T) -> SpecialResult<T> {
    check_finite(lambda, "lambda")?;
    if lambda <= T::zero() {
        return Err(SpecialError::DomainError("pdtr: lambda must be positive".to_string()));
    }
    
    // Use the regularized incomplete gamma function
    // P(X <= k) = Q(k+1, lambda) = gammaincc(k+1, lambda)
    let k_plus_1 = T::from_usize(k + 1).unwrap();
    
    // For now, use the complementary relationship with gamma
    // This should be replaced with proper gammaincc when implemented
    let gamma_full = gamma(k_plus_1);
    let gamma_inc = gamma_incomplete_lower(k_plus_1, lambda)?;
    
    Ok(gamma_inc / gamma_full)
}

/// Poisson survival function
///
/// # Returns
/// P(X > k) where X ~ Poisson(lambda)
pub fn pdtrc<T: Float + FromPrimitive + Display + Debug + AddAssign>(k: usize, lambda: T) -> SpecialResult<T> {
    check_finite(lambda, "lambda")?;
    if lambda <= T::zero() {
        return Err(SpecialError::DomainError("pdtrc: lambda must be positive".to_string()));
    }
    
    // P(X > k) = P(k+1, lambda) = gammainc(k+1, lambda)
    let k_plus_1 = T::from_usize(k + 1).unwrap();
    let gamma_full = gamma(k_plus_1);
    let gamma_inc = gamma_incomplete_upper(k_plus_1, lambda)?;
    
    Ok(gamma_inc / gamma_full)
}

// Chi-square distribution functions

/// Chi-square cumulative distribution function
///
/// # Arguments
/// * `df` - Degrees of freedom
/// * `x` - Point at which to evaluate
///
/// # Returns
/// P(X <= x) where X ~ Chi-square(df)
pub fn chdtr<T: Float + FromPrimitive + Display + Debug + AddAssign>(df: T, x: T) -> SpecialResult<T> {
    check_finite(df, "df")?;
    check_finite(x, "x")?;
    
    if df <= T::zero() {
        return Err(SpecialError::DomainError("chdtr: df must be positive".to_string()));
    }
    
    if x <= T::zero() {
        return Ok(T::zero());
    }
    
    // Chi-square CDF = gammainc(df/2, x/2)
    let half_df = df / T::from_f64(2.0).unwrap();
    let half_x = x / T::from_f64(2.0).unwrap();
    
    let gamma_full = gamma(half_df);
    let gamma_inc = gamma_incomplete_lower(half_df, half_x)?;
    
    Ok(gamma_inc / gamma_full)
}

/// Chi-square survival function
pub fn chdtrc<T: Float + FromPrimitive + Display + Debug + AddAssign>(df: T, x: T) -> SpecialResult<T> {
    let cdf = chdtr(df, x)?;
    Ok(T::one() - cdf)
}

// Student's t distribution functions

/// Student's t cumulative distribution function
///
/// # Arguments
/// * `df` - Degrees of freedom
/// * `t` - Point at which to evaluate
///
/// # Returns
/// P(X <= t) where X ~ t(df)
pub fn stdtr<T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign>(df: T, t: T) -> SpecialResult<T> {
    check_finite(df, "df")?;
    check_finite(t, "t")?;
    
    if df <= T::zero() {
        return Err(SpecialError::DomainError("stdtr: df must be positive".to_string()));
    }
    
    // Use the relationship with incomplete beta function
    let x = df / (df + t * t);
    let half = T::from_f64(0.5).unwrap();
    
    if t < T::zero() {
        Ok(half * betainc_regularized(half * df, half, x)?)
    } else {
        Ok(T::one() - half * betainc_regularized(half * df, half, x)?)
    }
}

// F distribution functions

/// F cumulative distribution function
///
/// # Arguments
/// * `dfn` - Numerator degrees of freedom
/// * `dfd` - Denominator degrees of freedom
/// * `x` - Point at which to evaluate
///
/// # Returns
/// P(X <= x) where X ~ F(dfn, dfd)
pub fn fdtr<T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign>(dfn: T, dfd: T, x: T) -> SpecialResult<T> {
    check_finite(dfn, "dfn")?;
    check_finite(dfd, "dfd")?;
    check_finite(x, "x")?;
    
    if dfn <= T::zero() || dfd <= T::zero() {
        return Err(SpecialError::DomainError("fdtr: degrees of freedom must be positive".to_string()));
    }
    
    if x <= T::zero() {
        return Ok(T::zero());
    }
    
    // Use the relationship with incomplete beta function
    let half_dfn = dfn / T::from_f64(2.0).unwrap();
    let half_dfd = dfd / T::from_f64(2.0).unwrap();
    let y = (dfn * x) / (dfn * x + dfd);
    
    betainc_regularized(half_dfn, half_dfd, y)
}

/// F survival function
pub fn fdtrc<T: Float + FromPrimitive + Display + Debug + AddAssign + SubAssign + MulAssign>(dfn: T, dfd: T, x: T) -> SpecialResult<T> {
    let cdf = fdtr(dfn, dfd, x)?;
    Ok(T::one() - cdf)
}

// Gamma distribution functions

/// Gamma cumulative distribution function
///
/// # Arguments
/// * `a` - Shape parameter
/// * `x` - Point at which to evaluate
///
/// # Returns
/// P(X <= x) where X ~ Gamma(a, 1)
pub fn gdtr<T: Float + FromPrimitive + Display + Debug + AddAssign>(a: T, x: T) -> SpecialResult<T> {
    check_finite(a, "a")?;
    check_finite(x, "x")?;
    
    if a <= T::zero() {
        return Err(SpecialError::DomainError("gdtr: shape parameter must be positive".to_string()));
    }
    
    if x <= T::zero() {
        return Ok(T::zero());
    }
    
    // Gamma CDF = gammainc(a, x)
    let gamma_full = gamma(a);
    let gamma_inc = gamma_incomplete_lower(a, x)?;
    
    Ok(gamma_inc / gamma_full)
}

/// Gamma survival function  
pub fn gdtrc<T: Float + FromPrimitive + Display + Debug + AddAssign>(a: T, x: T) -> SpecialResult<T> {
    let cdf = gdtr(a, x)?;
    Ok(T::one() - cdf)
}

// Kolmogorov-Smirnov distribution functions

/// Kolmogorov distribution CDF
///
/// Computes the CDF of the Kolmogorov distribution (supremum of Brownian bridge).
///
/// # Arguments
/// * `x` - Point at which to evaluate
///
/// # Returns
/// P(D_n * sqrt(n) <= x) where D_n is the Kolmogorov-Smirnov statistic
pub fn kolmogorov<T: Float + FromPrimitive>(x: T) -> T {
    if x <= T::zero() {
        return T::zero();
    }
    
    if x >= T::from_f64(6.0).unwrap() {
        return T::one();
    }
    
    // Use the alternating series representation
    let pi = T::from_f64(PI).unwrap();
    let mut sum = T::zero();
    let mut k = T::one();
    let tol = T::from_f64(1e-12).unwrap();
    
    loop {
        let term = T::from_f64(2.0).unwrap() * 
                  (-(T::from_f64(2.0).unwrap() * k * k * x * x)).exp() *
                  ((T::from_f64(2.0).unwrap() * k * k * x * x - T::one()) * 
                   T::from_f64(2.0).unwrap()).exp();
        
        sum = sum + if k.to_isize().unwrap() % 2 == 0 { -term } else { term };
        
        if term.abs() < tol {
            break;
        }
        
        k = k + T::one();
    }
    
    (T::from_f64(8.0).unwrap() * x / pi.sqrt()) * sum
}

/// Inverse of Kolmogorov distribution
pub fn kolmogi<T: Float + FromPrimitive + Display>(p: T) -> SpecialResult<T> {
    check_probability(p, "p")?;
    
    // Use Newton's method or bisection to find x such that kolmogorov(x) = p
    // This is a simplified implementation
    let mut low = T::zero();
    let mut high = T::from_f64(6.0).unwrap();
    let tol = T::from_f64(1e-8).unwrap();
    
    while (high - low) > tol {
        let mid = (low + high) / T::from_f64(2.0).unwrap();
        let cdf = kolmogorov(mid);
        
        if cdf < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    
    Ok((low + high) / T::from_f64(2.0).unwrap())
}

// Helper functions for incomplete gamma (simplified versions)
// These should be replaced with proper implementations

fn gamma_incomplete_lower<T: Float + FromPrimitive + Debug + AddAssign>(a: T, x: T) -> SpecialResult<T> {
    // Simplified implementation using series expansion for small x
    if x <= T::zero() {
        return Ok(T::zero());
    }
    
    if x < a + T::one() {
        // Series expansion
        let mut sum = T::one() / a;
        let mut term = T::one() / a;
        let mut n = T::one();
        
        while term.abs() > T::from_f64(1e-12).unwrap() * sum.abs() {
            term = term * x / (a + n);
            sum += term;
            n += T::one();
        }
        
        Ok(x.powf(a) * (-x).exp() * sum)
    } else {
        // Use complement
        let gamma_full = gamma(a);
        let gamma_upper = gamma_incomplete_upper(a, x)?;
        Ok(gamma_full - gamma_upper)
    }
}

fn gamma_incomplete_upper<T: Float + FromPrimitive + Debug + AddAssign>(a: T, x: T) -> SpecialResult<T> {
    // Simplified implementation using continued fraction for large x
    if x <= T::zero() {
        return Ok(gamma(a));
    }
    
    if x >= a + T::one() {
        // Continued fraction expansion
        let mut b = x + T::one() - a;
        let mut c = T::from_f64(1e30).unwrap();
        let mut d = T::one() / b;
        let mut h = d;
        
        for i in 1..100 {
            let an = -T::from_usize(i).unwrap() * (T::from_usize(i).unwrap() - a);
            b += T::from_f64(2.0).unwrap();
            d = an * d + b;
            if d.abs() < T::from_f64(1e-30).unwrap() {
                d = T::from_f64(1e-30).unwrap();
            }
            c = b + an / c;
            if c.abs() < T::from_f64(1e-30).unwrap() {
                c = T::from_f64(1e-30).unwrap();
            }
            d = T::one() / d;
            let delta = d * c;
            h = h * delta;
            
            if (delta - T::one()).abs() < T::from_f64(1e-10).unwrap() {
                break;
            }
        }
        
        Ok(x.powf(a) * (-x).exp() * h)
    } else {
        // Use complement
        let gamma_full = gamma(a);
        let gamma_lower = gamma_incomplete_lower(a, x)?;
        Ok(gamma_full - gamma_lower)
    }
}

// Array operations for distribution functions

/// Apply normal CDF to array
pub fn ndtr_array<T>(x: &ArrayView1<T>) -> Array1<T>
where
    T: Float + FromPrimitive + Send + Sync + Debug,
{
    #[cfg(feature = "parallel")]
    {
        if x.len() > 1000 {
            // Use parallel processing for large arrays
            use scirs2_core::parallel_ops::*;
            let vec: Vec<T> = x.as_slice()
                .unwrap()
                .par_iter()
                .map(|&val| ndtr(val))
                .collect();
            Array1::from_vec(vec)
        } else {
            x.mapv(ndtr)
        }
    }
    
    #[cfg(not(feature = "parallel"))]
    {
        x.mapv(ndtr)
    }
}

/// Apply binomial CDF to arrays
pub fn bdtr_array<T>(k: &[usize], n: usize, p: T) -> SpecialResult<Array1<T>>
where
    T: Float + FromPrimitive + Send + Sync + Debug + Display + AddAssign + SubAssign + MulAssign,
{
    check_probability(p, "p")?;
    
    let results: Result<Vec<T>, _> = k.iter()
        .map(|&ki| bdtr(ki, n, p))
        .collect();
    
    Ok(Array1::from_vec(results?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_normal_distribution() {
        // Test standard normal CDF
        assert_relative_eq!(ndtr(0.0), 0.5, epsilon = 1e-10);
        assert_relative_eq!(ndtr(1.0), 0.8413447460685429, epsilon = 1e-10);
        assert_relative_eq!(ndtr(-1.0), 0.15865525393145707, epsilon = 1e-10);
        
        // Test inverse
        let p = 0.95;
        let x = ndtri(p).unwrap();
        assert_relative_eq!(ndtr(x), p, epsilon = 1e-10);
    }

    #[test]
    fn test_binomial_distribution() {
        // Test binomial CDF
        let cdf = bdtr(2, 5, 0.5).unwrap();
        assert_relative_eq!(cdf, 0.5, epsilon = 1e-10);
        
        // Test complement
        let surv = bdtrc(2, 5, 0.5).unwrap();
        assert_relative_eq!(cdf + surv, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_chi_square_distribution() {
        // Test chi-square CDF
        let cdf = chdtr(2.0, 2.0).unwrap();
        assert_relative_eq!(cdf, 0.6321205588285577, epsilon = 1e-8);
    }

    #[test]
    fn test_student_t_distribution() {
        // Test t distribution CDF
        let cdf = stdtr(10.0, 0.0).unwrap();
        assert_relative_eq!(cdf, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_f_distribution() {
        // Test F distribution CDF  
        let cdf = fdtr(5.0, 10.0, 1.0).unwrap();
        assert_relative_eq!(cdf, 0.5417926019448583, epsilon = 1e-8);
    }
}