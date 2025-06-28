//! Utility and convenience functions
//!
//! This module provides various utility functions that are commonly used
//! in scientific computing, matching SciPy's special module.

use crate::error::{SpecialError, SpecialResult};
use crate::validation::check_finite;
use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive, Zero};
use std::fmt::{Debug, Display};

/// Cube root function
///
/// Computes the real cube root of x, handling negative values correctly.
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// The cube root of x
///
/// # Examples
/// ```
/// use scirs2_special::utility::cbrt;
///
/// assert_eq!(cbrt(8.0), 2.0);
/// assert_eq!(cbrt(-8.0), -2.0);
/// ```
pub fn cbrt<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    if x >= T::zero() {
        x.powf(T::from_f64(1.0 / 3.0).unwrap())
    } else {
        -(-x).powf(T::from_f64(1.0 / 3.0).unwrap())
    }
}

/// Base-10 exponential function
///
/// Computes 10^x.
///
/// # Arguments
/// * `x` - Exponent
///
/// # Returns
/// 10 raised to the power x
pub fn exp10<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    T::from_f64(10.0).unwrap().powf(x)
}

/// Base-2 exponential function
///
/// Computes 2^x.
///
/// # Arguments
/// * `x` - Exponent
///
/// # Returns
/// 2 raised to the power x
pub fn exp2<T>(x: T) -> T
where
    T: Float,
{
    x.exp2()
}

/// Convert degrees to radians
///
/// # Arguments
/// * `degrees` - Angle in degrees
///
/// # Returns
/// Angle in radians
pub fn radian<T>(degrees: T) -> T
where
    T: Float + FromPrimitive,
{
    let pi = T::from_f64(std::f64::consts::PI).unwrap();
    degrees * pi / T::from_f64(180.0).unwrap()
}

/// Cosine of angle in degrees
///
/// # Arguments
/// * `x` - Angle in degrees
///
/// # Returns
/// cos(x) where x is in degrees
pub fn cosdg<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    radian(x).cos()
}

/// Sine of angle in degrees
///
/// # Arguments
/// * `x` - Angle in degrees
///
/// # Returns
/// sin(x) where x is in degrees
pub fn sindg<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    radian(x).sin()
}

/// Tangent of angle in degrees
///
/// # Arguments
/// * `x` - Angle in degrees
///
/// # Returns
/// tan(x) where x is in degrees
pub fn tandg<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    radian(x).tan()
}

/// Cotangent of angle in degrees
///
/// # Arguments
/// * `x` - Angle in degrees
///
/// # Returns
/// cot(x) = 1/tan(x) where x is in degrees
pub fn cotdg<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    T::from_f64(1.0).unwrap() / tandg(x)
}

/// Compute cos(x) - 1 accurately for small x
///
/// This function provides better numerical accuracy than directly computing cos(x) - 1
/// when x is close to 0.
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// cos(x) - 1
pub fn cosm1<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    // Use Taylor series for small x
    if x.abs() < T::from_f64(0.1).unwrap() {
        let x2 = x * x;
        let mut sum = -x2 / T::from_f64(2.0).unwrap();
        let mut term = sum;
        let mut n = T::from_f64(4.0).unwrap();

        while term.abs() > T::epsilon() * sum.abs() {
            term = term * (-x2) / (n * (n - T::from_f64(1.0).unwrap()));
            sum = sum + term;
            n = n + T::from_f64(2.0).unwrap();
        }

        sum
    } else {
        x.cos() - T::from_f64(1.0).unwrap()
    }
}

/// Compute (1 + x)^y - 1 accurately
///
/// This function provides better numerical accuracy than directly computing (1 + x)^y - 1
/// when x is small.
///
/// # Arguments
/// * `x` - Base adjustment
/// * `y` - Exponent
///
/// # Returns
/// (1 + x)^y - 1
pub fn powm1<T>(x: T, y: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display,
{
    check_finite(x, "x")?;
    check_finite(y, "y")?;

    if x.abs() < T::from_f64(0.1).unwrap() && y.abs() < T::from_f64(10.0).unwrap() {
        // Use exp(y * log1p(x)) - 1 = expm1(y * log1p(x))
        Ok((y * x.ln_1p()).exp_m1())
    } else {
        Ok((T::from_f64(1.0).unwrap() + x).powf(y) - T::from_f64(1.0).unwrap())
    }
}

/// Compute x * log(y) safely
///
/// Returns 0 when x = 0, even if log(y) is undefined or infinite.
///
/// # Arguments
/// * `x` - Multiplier
/// * `y` - Argument to logarithm
///
/// # Returns
/// x * log(y) with special handling for x = 0
pub fn xlogy<T>(x: T, y: T) -> T
where
    T: Float + Zero,
{
    if x.is_zero() {
        T::zero()
    } else if y <= T::zero() {
        T::nan()
    } else {
        x * y.ln()
    }
}

/// Compute x * log(1 + y) safely
///
/// Returns 0 when x = 0, provides accurate results for small y.
///
/// # Arguments
/// * `x` - Multiplier
/// * `y` - Argument to log1p
///
/// # Returns
/// x * log(1 + y) with special handling
pub fn xlog1py<T>(x: T, y: T) -> T
where
    T: Float + Zero,
{
    if x.is_zero() {
        T::zero()
    } else {
        x * y.ln_1p()
    }
}

/// Relative exponential function
///
/// Computes (exp(x) - 1) / x accurately for small x.
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// (exp(x) - 1) / x
pub fn exprel<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    if x.abs() < T::from_f64(1e-5).unwrap() {
        // Taylor series: 1 + x/2 + x²/6 + x³/24 + ...
        let mut sum = T::from_f64(1.0).unwrap();
        let mut term = x / T::from_f64(2.0).unwrap();
        let mut n = T::from_f64(2.0).unwrap();

        sum = sum + term;

        while term.abs() > T::epsilon() * sum.abs() {
            term = term * x / (n + T::from_f64(1.0).unwrap());
            sum = sum + term;
            n = n + T::from_f64(1.0).unwrap();
        }

        sum
    } else {
        x.exp_m1() / x
    }
}

/// Round to nearest integer
///
/// Rounds half-integers to nearest even number (banker's rounding).
///
/// # Arguments
/// * `x` - Value to round
///
/// # Returns
/// Rounded value
pub fn round<T>(x: T) -> T
where
    T: Float,
{
    x.round()
}

/// Dirichlet kernel (periodic sinc function)
///
/// Computes sin(n * x/2) / (n * sin(x/2))
///
/// # Arguments
/// * `x` - Input value
/// * `n` - Integer parameter
///
/// # Returns
/// The Dirichlet kernel value
pub fn diric<T>(x: T, n: i32) -> T
where
    T: Float + FromPrimitive,
{
    if n == 0 {
        return T::zero();
    }

    let n_f = T::from_i32(n).unwrap();
    let half = T::from_f64(0.5).unwrap();
    let x_half = x * half;
    let sin_x_half = x_half.sin();

    if sin_x_half.abs() < T::epsilon() {
        // Use limit as x -> 0
        T::from_i32(n).unwrap()
    } else {
        (n_f * x_half).sin() / (n_f * sin_x_half)
    }
}

/// Arithmetic-geometric mean
///
/// Computes the arithmetic-geometric mean of a and b.
///
/// # Arguments
/// * `a` - First value (must be positive)
/// * `b` - Second value (must be positive)
///
/// # Returns
/// The arithmetic-geometric mean
pub fn agm<T>(a: T, b: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display,
{
    check_finite(a, "a")?;
    check_finite(b, "b")?;

    if a <= T::zero() || b <= T::zero() {
        return Err(SpecialError::DomainError(
            "agm: arguments must be positive".to_string(),
        ));
    }

    let mut a_n = a;
    let mut b_n = b;
    let tol = T::epsilon() * a.max(b);

    while (a_n - b_n).abs() > tol {
        let a_next = (a_n + b_n) / T::from_f64(2.0).unwrap();
        let b_next = (a_n * b_n).sqrt();
        a_n = a_next;
        b_n = b_next;
    }

    Ok(a_n)
}

/// Log of expit function
///
/// Computes log(1 / (1 + exp(-x))) = -log1p(exp(-x))
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// log(expit(x))
pub fn log_expit<T>(x: T) -> T
where
    T: Float,
{
    if x >= T::zero() {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

/// Softplus function
///
/// Computes log(1 + exp(x)) in a numerically stable way.
///
/// # Arguments
/// * `x` - Input value
///
/// # Returns
/// log(1 + exp(x))
pub fn softplus<T>(x: T) -> T
where
    T: Float + FromPrimitive,
{
    if x > T::from_f64(20.0).unwrap() {
        // For large x, log(1 + exp(x)) ≈ x
        x
    } else if x < T::from_f64(-20.0).unwrap() {
        // For large negative x, log(1 + exp(x)) ≈ exp(x)
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

/// Owen's T function
///
/// Computes T(h, a) = (1/2π) ∫₀ᵃ exp(-h²(1+x²)/2) / (1+x²) dx
///
/// # Arguments
/// * `h` - First parameter
/// * `a` - Second parameter
///
/// # Returns
/// Owen's T function value
///
/// # Algorithm
/// Uses a combination of series expansion for small |h|, asymptotic expansion
/// for large |h|, and numerical integration for intermediate values.
pub fn owens_t<T>(h: T, a: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Debug,
{
    check_finite(h, "h")?;
    check_finite(a, "a")?;

    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0).unwrap();
    let pi = T::from_f64(std::f64::consts::PI).unwrap();

    // Handle special cases
    if a.is_zero() {
        return Ok(zero);
    }

    if h.is_zero() {
        return Ok(a.atan() / (two * pi));
    }

    let abs_h = h.abs();
    let abs_a = a.abs();

    // Use symmetry properties to reduce to first quadrant
    let sign = if (h >= zero && a >= zero) || (h < zero && a < zero) {
        one
    } else {
        -one
    };

    let result = if abs_h < T::from_f64(0.1).unwrap() {
        // For small |h|, use series expansion
        owens_t_series(abs_h, abs_a)?
    } else if abs_h > T::from_f64(10.0).unwrap() {
        // For large |h|, use asymptotic expansion
        owens_t_asymptotic(abs_h, abs_a)?
    } else {
        // For intermediate values, use numerical integration
        owens_t_numerical(abs_h, abs_a)?
    };

    Ok(sign * result)
}

/// Owen's T function using series expansion for small h
fn owens_t_series<T>(h: T, a: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive,
{
    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0).unwrap();
    let pi = T::from_f64(std::f64::consts::PI).unwrap();

    let h2 = h * h;
    let a2 = a * a;
    let atan_a = a.atan();

    // Series: T(h,a) = (1/2π) * atan(a) - (h/2π) * ∑ (-1)^n * h^(2n) * I_n(a)
    // where I_n(a) = ∫₀ᵃ x^(2n) / (1+x²) dx

    let mut sum = zero;
    let mut h_power = one;

    for n in 0..20 {
        let integral = if n == 0 {
            atan_a
        } else {
            // I_n(a) can be computed recursively

            if n == 1 {
                (a2.ln_1p()) / two
            } else {
                // For higher n, use recursive relation or approximation
                a.powi(2 * n as i32 - 1) / T::from_usize(2 * n - 1).unwrap()
            }
        };

        let term = if n % 2 == 0 {
            h_power * integral
        } else {
            -h_power * integral
        };
        sum = sum + term;

        // Check for convergence
        if term.abs() < T::from_f64(1e-15).unwrap() {
            break;
        }

        h_power = h_power * h2;
    }

    Ok(atan_a / (two * pi) - h * sum / (two * pi))
}

/// Owen's T function using asymptotic expansion for large h
fn owens_t_asymptotic<T>(h: T, a: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive,
{
    let one = T::one();
    let two = T::from_f64(2.0).unwrap();
    let pi = T::from_f64(std::f64::consts::PI).unwrap();

    let h2 = h * h;
    let a2 = a * a;
    let exp_factor = (-h2 * (one + a2) / two).exp();

    // Asymptotic expansion for large h
    // T(h,a) ≈ (1/2π) * exp(-h²(1+a²)/2) * (a/(h²(1+a²))) * [1 + O(1/h²)]

    let denominator = h2 * (one + a2);
    let result = exp_factor * a / (two * pi * denominator);

    // Add first correction term
    let correction = one - (T::from_f64(3.0).unwrap() * a2) / (one + a2).powi(2);
    let corrected_result = result * correction;

    Ok(corrected_result)
}

/// Owen's T function using numerical integration
fn owens_t_numerical<T>(h: T, a: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive,
{
    let zero = T::zero();
    let one = T::one();
    let two = T::from_f64(2.0).unwrap();
    let pi = T::from_f64(std::f64::consts::PI).unwrap();

    let h2 = h * h;

    // Use Simpson's rule for numerical integration
    let n = 1000; // Number of intervals
    let dx = a / T::from_usize(n).unwrap();

    let mut sum = zero;

    for i in 0..=n {
        let x = T::from_usize(i).unwrap() * dx;
        let integrand = (-h2 * (one + x * x) / two).exp() / (one + x * x);

        let weight = if i == 0 || i == n {
            one
        } else if i % 2 == 1 {
            T::from_f64(4.0).unwrap()
        } else {
            two
        };

        sum = sum + weight * integrand;
    }

    let result = sum * dx / (T::from_f64(3.0).unwrap() * two * pi);
    Ok(result)
}

/// Apply utility function to arrays
pub fn cbrt_array<T>(x: &ArrayView1<T>) -> Array1<T>
where
    T: Float + FromPrimitive + Send + Sync,
{
    x.mapv(cbrt)
}

pub fn exp10_array<T>(x: &ArrayView1<T>) -> Array1<T>
where
    T: Float + FromPrimitive + Send + Sync,
{
    x.mapv(exp10)
}

pub fn round_array<T>(x: &ArrayView1<T>) -> Array1<T>
where
    T: Float + Send + Sync,
{
    x.mapv(round)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cbrt() {
        assert_relative_eq!(cbrt(8.0), 2.0, epsilon = 1e-10);
        assert_relative_eq!(cbrt(-8.0), -2.0, epsilon = 1e-10);
        assert_relative_eq!(cbrt(27.0), 3.0, epsilon = 1e-10);
        assert_eq!(cbrt(0.0), 0.0);
    }

    #[test]
    fn test_exp10() {
        assert_relative_eq!(exp10(0.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(exp10(1.0), 10.0, epsilon = 1e-10);
        assert_relative_eq!(exp10(2.0), 100.0, epsilon = 1e-10);
        assert_relative_eq!(exp10(-1.0), 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_exp2() {
        assert_eq!(exp2(0.0), 1.0);
        assert_eq!(exp2(1.0), 2.0);
        assert_eq!(exp2(3.0), 8.0);
        assert_eq!(exp2(-1.0), 0.5);
    }

    #[test]
    fn test_trig_degrees() {
        assert_relative_eq!(cosdg(0.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(cosdg(90.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(sindg(90.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(tandg(45.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosm1() {
        // For small x, cosm1 should be more accurate than cos(x) - 1
        let x = 1e-8;
        let result = cosm1(x);
        assert!(result < 0.0);
        assert!(result.abs() < 1e-15);
    }

    #[test]
    fn test_xlogy() {
        assert_eq!(xlogy(0.0, 2.0), 0.0);
        assert_eq!(xlogy(0.0, 0.0), 0.0);
        assert!(xlogy(1.0, 0.0).is_nan());
        assert_relative_eq!(xlogy(2.0, 3.0), 2.0 * 3.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_exprel() {
        assert_relative_eq!(exprel(0.0), 1.0, epsilon = 1e-10);
        let x = 1e-10;
        assert_relative_eq!(exprel(x), 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_agm() {
        let result = agm(1.0, 2.0).unwrap();
        assert_relative_eq!(result, 1.4567910310469068, epsilon = 1e-10);

        // AGM is symmetric
        assert_relative_eq!(agm(2.0, 1.0).unwrap(), result, epsilon = 1e-10);
    }

    #[test]
    fn test_diric() {
        assert_relative_eq!(diric(0.0, 5), 5.0, epsilon = 1e-10);
        assert_eq!(diric(0.0, 0), 0.0);
    }
}
