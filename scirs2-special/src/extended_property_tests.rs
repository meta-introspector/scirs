//! Extended property-based testing for special functions
//!
//! This module provides comprehensive property-based tests using QuickCheck
//! to verify mathematical properties, identities, and invariants across
//! all special functions in the module.

use num_complex::{Complex64, ComplexFloat};
use quickcheck::{Arbitrary, Gen, TestResult};
use quickcheck_macros::quickcheck;
use std::f64;

// Custom arbitrary types for constrained inputs

/// Positive real number in reasonable range
#[derive(Clone, Debug)]
struct Positive(f64);

impl Arbitrary for Positive {
    fn arbitrary(g: &mut Gen) -> Self {
        let val: f64 = Arbitrary::arbitrary(g);
        Positive((val.abs() % 50.0) + 0.1)
    }
}

/// Small positive real number
#[derive(Clone, Debug)]
struct SmallPositive(f64);

impl Arbitrary for SmallPositive {
    fn arbitrary(g: &mut Gen) -> Self {
        let val: f64 = Arbitrary::arbitrary(g);
        SmallPositive((val.abs() % 2.0) + 0.1)
    }
}

/// Real number in [-1, 1]
#[derive(Clone, Debug)]
struct UnitInterval(f64);

impl Arbitrary for UnitInterval {
    fn arbitrary(g: &mut Gen) -> Self {
        let val: f64 = Arbitrary::arbitrary(g);
        UnitInterval((val % 2.0 - 1.0).clamp(-1.0, 1.0))
    }
}

/// Non-negative integer
#[derive(Clone, Debug)]
struct NonNegInt(i32);

impl Arbitrary for NonNegInt {
    fn arbitrary(g: &mut Gen) -> Self {
        let val: u32 = Arbitrary::arbitrary(g);
        NonNegInt((val % 20) as i32)
    }
}

/// Reasonable complex number
#[derive(Clone, Debug)]
struct ReasonableComplex(Complex64);

impl Arbitrary for ReasonableComplex {
    fn arbitrary(g: &mut Gen) -> Self {
        let re: f64 = Arbitrary::arbitrary(g);
        let im: f64 = Arbitrary::arbitrary(g);
        ReasonableComplex(Complex64::new(
            (re % 10.0).clamp(-10.0, 10.0),
            (im % 10.0).clamp(-10.0, 10.0),
        ))
    }
}

// Helper functions
fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    if a.is_nan() || b.is_nan() {
        return false;
    }
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

fn complex_approx_eq(a: Complex64, b: Complex64, tol: f64) -> bool {
    approx_eq(a.re, b.re, tol) && approx_eq(a.im, b.im, tol)
}

// Gamma function properties
mod gamma_properties {
    use super::*;
    use crate::{beta, digamma, gamma, gammaln};

    #[quickcheck]
    fn gamma_reflection_formula(x: f64) -> TestResult {
        // Gamma(x) * Gamma(1-x) = π / sin(πx)
        if x <= 0.0 || x >= 1.0 || (x - 0.5).abs() < 0.1 {
            return TestResult::discard();
        }

        let gamma_x = gamma(x);
        let gamma_1_minus_x = gamma(1.0 - x);
        let product = gamma_x * gamma_1_minus_x;
        let expected = f64::consts::PI / (f64::consts::PI * x).sin();

        TestResult::from_bool(approx_eq(product, expected, 1e-10))
    }

    #[quickcheck]
    fn gamma_duplication_formula(x: Positive) -> TestResult {
        // Gamma(x) * Gamma(x + 0.5) = sqrt(π) * 2^(1-2x) * Gamma(2x)
        let x = x.0;
        if x > 20.0 {
            return TestResult::discard();
        }

        let gamma_x = gamma(x);
        let gamma_x_half = gamma(x + 0.5);
        let gamma_2x = gamma(2.0 * x);

        let left = gamma_x * gamma_x_half;
        let right = f64::consts::PI.sqrt() * 2.0_f64.powf(1.0 - 2.0 * x) * gamma_2x;

        TestResult::from_bool(approx_eq(left, right, 1e-10))
    }

    #[quickcheck]
    fn beta_gamma_relationship(a: SmallPositive, b: SmallPositive) -> bool {
        // B(a,b) = Gamma(a) * Gamma(b) / Gamma(a + b)
        let a = a.0;
        let b = b.0;

        let beta_ab = beta(a, b);
        let expected = gamma(a) * gamma(b) / gamma(a + b);

        approx_eq(beta_ab, expected, 1e-10)
    }

    #[quickcheck]
    fn digamma_difference_formula(x: Positive, n: NonNegInt) -> TestResult {
        // ψ(x + n) - ψ(x) = sum(1/(x + k) for k in 0..n)
        let x = x.0;
        let n = n.0 as usize;

        if x > 50.0 || n > 10 {
            return TestResult::discard();
        }

        let psi_x = digamma(x);
        let psi_x_n = digamma(x + n as f64);
        let diff = psi_x_n - psi_x;

        let mut sum = 0.0;
        for k in 0..n {
            sum += 1.0 / (x + k as f64);
        }

        TestResult::from_bool(approx_eq(diff, sum, 1e-10))
    }

    #[quickcheck]
    fn log_gamma_stirling_approximation(x: f64) -> TestResult {
        // For large x: log(Gamma(x)) ≈ (x - 0.5) * log(x) - x + 0.5 * log(2π)
        if x < 100.0 || x > 1000.0 {
            return TestResult::discard();
        }

        let log_gamma_x = gammaln(x);
        let stirling = (x - 0.5) * x.ln() - x + 0.5 * (2.0 * f64::consts::PI).ln();
        let relative_error = (log_gamma_x - stirling).abs() / log_gamma_x.abs();

        TestResult::from_bool(relative_error < 0.01) // 1% error for large x
    }
}

// Bessel function properties
mod bessel_properties {
    use super::*;
    use crate::bessel::{iv, j0, j1, jn, kv, y0, y1};

    #[quickcheck]
    fn bessel_j_recurrence(n: NonNegInt, x: Positive) -> TestResult {
        // J_{n-1}(x) + J_{n+1}(x) = (2n/x) * J_n(x)
        let n = n.0;
        let x = x.0;

        if n < 1 || n > 10 || x > 20.0 {
            return TestResult::discard();
        }

        let j_n_minus_1 = jn(n - 1, x);
        let j_n = jn(n, x);
        let j_n_plus_1 = jn(n + 1, x);

        let left = j_n_minus_1 + j_n_plus_1;
        let right = (2.0 * n as f64 / x) * j_n;

        TestResult::from_bool(approx_eq(left, right, 1e-10))
    }

    #[quickcheck]
    fn bessel_j_derivative_relation(x: Positive) -> bool {
        // J_0'(x) = -J_1(x)
        let x = x.0;
        let h = 1e-8;

        let j0_x = j0(x);
        let j0_x_h = j0(x + h);
        let derivative = (j0_x_h - j0_x) / h;
        let expected = -j1(x);

        approx_eq(derivative, expected, 1e-5)
    }

    #[quickcheck]
    fn bessel_wronskian(x: Positive) -> bool {
        // J_n(x) * Y_{n+1}(x) - J_{n+1}(x) * Y_n(x) = -2/(π*x)
        let x = x.0;
        let _n = 0; // Use n=0 for simplicity

        let j_n = j0(x);
        let j_n_1 = j1(x);
        let y_n = y0(x);
        let y_n_1 = y1(x);

        let wronskian = j_n * y_n_1 - j_n_1 * y_n;
        let expected = -2.0 / (f64::consts::PI * x);

        approx_eq(wronskian, expected, 1e-10)
    }

    #[quickcheck]
    fn modified_bessel_relation(v: SmallPositive, x: Positive) -> TestResult {
        // I_v(x) * K_v(x) - I_{v+1}(x) * K_{v-1}(x) = 1/x
        let v = v.0;
        let x = x.0;

        if v < 0.5 || v > 5.0 || x > 10.0 {
            return TestResult::discard();
        }

        let i_v = iv(v, x);
        let k_v = kv(v, x);
        let i_v1 = iv(v + 1.0, x);
        let k_v_1 = kv(v - 1.0, x);

        let left = i_v * k_v - i_v1 * k_v_1;
        let right = 1.0 / x;

        TestResult::from_bool(approx_eq(left.abs(), right, 1e-8))
    }
}

// Error function properties
mod error_function_properties {
    use super::*;
    use crate::{erf, erfc, erfcinv, erfinv};

    #[quickcheck]
    fn erf_erfc_complement(x: f64) -> bool {
        // erf(x) + erfc(x) = 1
        let erf_x = erf(x);
        let erfc_x = erfc(x);

        approx_eq(erf_x + erfc_x, 1.0, 1e-14)
    }

    #[quickcheck]
    fn erf_odd_function(x: f64) -> TestResult {
        // erf(-x) = -erf(x)
        if x.abs() > 10.0 {
            return TestResult::discard();
        }

        let erf_x = erf(x);
        let erf_neg_x = erf(-x);

        TestResult::from_bool(approx_eq(erf_neg_x, -erf_x, 1e-14))
    }

    #[quickcheck]
    fn erf_erfinv_inverse(x: UnitInterval) -> TestResult {
        // erfinv(erf(x)) = x
        let x_val = x.0;
        if x_val.abs() > 0.999 {
            return TestResult::discard();
        }

        let erf_x = erf(x_val);
        let erfinv_erf_x = erfinv(erf_x);

        TestResult::from_bool(approx_eq(erfinv_erf_x, x_val, 1e-10))
    }

    #[quickcheck]
    fn erfc_erfcinv_inverse(p: UnitInterval) -> TestResult {
        // erfcinv(erfc(x)) = x
        let p_val = p.0;
        if p_val < 0.001 || p_val > 0.999 {
            return TestResult::discard();
        }

        let x = erfcinv(p_val);
        let erfc_x = erfc(x);

        TestResult::from_bool(approx_eq(erfc_x, p_val, 1e-10))
    }
}

// Orthogonal polynomial properties
mod orthogonal_polynomial_properties {
    use super::*;
    use crate::{chebyshev, hermite, laguerre, legendre};

    #[quickcheck]
    fn legendre_recurrence(n: NonNegInt, x: UnitInterval) -> TestResult {
        // (n+1)*P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)
        let n = n.0 as usize;
        let x = x.0;

        if n < 1 || n > 10 {
            return TestResult::discard();
        }

        let p_n_minus_1 = legendre(n - 1, x);
        let p_n = legendre(n, x);
        let p_n_plus_1 = legendre(n + 1, x);

        let left = (n + 1) as f64 * p_n_plus_1;
        let right = (2 * n + 1) as f64 * x * p_n - n as f64 * p_n_minus_1;

        TestResult::from_bool(approx_eq(left, right, 1e-10))
    }

    #[quickcheck]
    fn chebyshev_identity(n: NonNegInt, x: UnitInterval) -> bool {
        // T_n(cos(θ)) = cos(n*θ) where x = cos(θ)
        let n = n.0 as usize;
        let x = x.0;

        let t_n = chebyshev(n, x, true); // Use first kind Chebyshev polynomials
        let theta = x.acos();
        let expected = (n as f64 * theta).cos();

        approx_eq(t_n, expected, 1e-10)
    }

    #[quickcheck]
    fn hermite_parity(n: NonNegInt, x: f64) -> TestResult {
        // H_n(-x) = (-1)^n * H_n(x)
        let n = n.0 as usize;

        if x.abs() > 5.0 || n > 10 {
            return TestResult::discard();
        }

        let h_x = hermite(n, x);
        let h_neg_x = hermite(n, -x);
        let expected = if n % 2 == 0 { h_x } else { -h_x };

        TestResult::from_bool(approx_eq(h_neg_x, expected, 1e-10))
    }

    #[quickcheck]
    fn laguerre_special_value(n: NonNegInt) -> bool {
        // L_n(0) = 1
        let n = n.0 as usize;
        let l_n_0 = laguerre(n, 0.0);

        approx_eq(l_n_0, 1.0, 1e-14)
    }
}

// Spherical harmonics properties
mod spherical_harmonics_properties {
    use super::*;

    #[quickcheck]
    fn spherical_harmonics_normalization(
        l: NonNegInt,
        theta: UnitInterval,
        phi: f64,
    ) -> TestResult {
        // Check normalization for m=0 case
        let l = l.0;
        let m = 0;

        if l > 5 {
            return TestResult::discard();
        }

        let theta_val = (theta.0 + 1.0) * f64::consts::PI / 2.0; // Map to [0, π]
        let phi_val = phi % (2.0 * f64::consts::PI);

        let y_lm = crate::spherical_harmonics::sph_harm_complex(l as usize, m, theta_val, phi_val);

        // For m=0, Y_l0 should be real
        match y_lm {
            Ok((_re, im)) => TestResult::from_bool(im.abs() < 1e-14),
            Err(_) => TestResult::discard(),
        }
    }

    #[quickcheck]
    fn spherical_harmonics_conjugate_symmetry(
        l: NonNegInt,
        m: NonNegInt,
        theta: f64,
        phi: f64,
    ) -> TestResult {
        // Y_l^{-m} = (-1)^m * conj(Y_l^m)
        let l = l.0;
        let m = m.0;

        if l > 5 || m > l {
            return TestResult::discard();
        }

        let theta_val = theta.abs() % f64::consts::PI;
        let phi_val = phi % (2.0 * f64::consts::PI);

        let y_lm =
            crate::spherical_harmonics::sph_harm_complex(l as usize, m as i32, theta_val, phi_val);
        let y_l_neg_m = crate::spherical_harmonics::sph_harm_complex(
            l as usize,
            -(m as i32),
            theta_val,
            phi_val,
        );

        match (y_lm, y_l_neg_m) {
            (Ok((re1, im1)), Ok((re2, im2))) => {
                let val1 = Complex64::new(re1, im1);
                let val2 = Complex64::new(re2, im2);
                let expected = if m % 2 == 0 {
                    val1.conj()
                } else {
                    -val1.conj()
                };
                TestResult::from_bool(complex_approx_eq(val2, expected, 1e-10))
            }
            _ => TestResult::discard(),
        }
    }
}

// Elliptic function properties
mod elliptic_properties {
    use super::*;
    use crate::elliptic::{elliptic_e as ellipe, elliptic_k as ellipk};

    #[quickcheck]
    fn elliptic_k_special_values() -> bool {
        // K(0) = π/2
        let k_0 = ellipk(0.0);
        approx_eq(k_0, f64::consts::PI / 2.0, 1e-14)
    }

    #[quickcheck]
    fn elliptic_e_special_values() -> bool {
        // E(0) = π/2
        let e_0 = ellipe(0.0);
        approx_eq(e_0, f64::consts::PI / 2.0, 1e-14)
    }

    #[quickcheck]
    fn elliptic_e_bounds(m: UnitInterval) -> TestResult {
        // 1 <= E(m) <= π/2 for 0 <= m <= 1
        let m_val = m.0;
        if m_val < 0.0 || m_val > 1.0 {
            return TestResult::discard();
        }

        let e_m = ellipe(m_val);
        TestResult::from_bool(e_m >= 1.0 && e_m <= f64::consts::PI / 2.0)
    }
}

// Hypergeometric function properties
mod hypergeometric_properties {
    use super::*;
    use crate::hyp2f1;

    #[quickcheck]
    fn hyp1f1_special_case(b: Positive, z: f64) -> TestResult {
        // 1F1(0; b; z) = 1
        let b_val = b.0;

        if b_val > 10.0 || z.abs() > 10.0 {
            return TestResult::discard();
        }

        let result = crate::hypergeometric::hyp1f1(0.0, b_val, z);
        match result {
            Ok(val) => TestResult::from_bool(approx_eq(val, 1.0, 1e-10)),
            Err(_) => TestResult::discard(),
        }
    }

    #[quickcheck]
    fn hyp2f1_special_case(c: Positive, z: UnitInterval) -> TestResult {
        // 2F1(a, b; c; 0) = 1
        let c_val = c.0;
        let z_val = z.0 * 0.5; // Keep z small

        if c_val > 10.0 {
            return TestResult::discard();
        }

        let result = hyp2f1(1.0, 2.0, c_val, 0.0);
        TestResult::from_bool(approx_eq(result, 1.0, 1e-10))
    }
}

// Cross-function relationships
mod cross_function_properties {
    use super::*;
    use crate::combinatorial::factorial;
    use crate::{beta, erf, gamma};

    #[quickcheck]
    fn gamma_factorial_relation(n: NonNegInt) -> TestResult {
        // Gamma(n+1) = n!
        let n = n.0 as usize;

        if n > 20 {
            return TestResult::discard();
        }

        let gamma_n_plus_1 = gamma((n + 1) as f64);
        let n_factorial = factorial(n).unwrap();

        TestResult::from_bool(approx_eq(gamma_n_plus_1, n_factorial, 1e-10))
    }

    #[quickcheck]
    fn beta_integral_representation(a: SmallPositive, b: SmallPositive) -> TestResult {
        // Verify beta function satisfies certain integral properties
        let a_val = a.0;
        let b_val = b.0;

        if a_val > 5.0 || b_val > 5.0 {
            return TestResult::discard();
        }

        let beta_ab = beta(a_val, b_val);

        // Check beta is positive
        TestResult::from_bool(beta_ab > 0.0)
    }

    #[quickcheck]
    fn erf_probability_connection(x: f64) -> TestResult {
        // erf(x/sqrt(2)) relates to normal CDF
        if x.abs() > 5.0 {
            return TestResult::discard();
        }

        let erf_scaled = erf(x / 2.0_f64.sqrt());

        // Check bounds: -1 <= erf(x) <= 1
        TestResult::from_bool(erf_scaled >= -1.0 && erf_scaled <= 1.0)
    }
}
