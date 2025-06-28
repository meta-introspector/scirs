//! Property-based tests for mathematical invariants
//!
//! This module contains property-based tests that verify mathematical invariants
//! and properties that should hold for all valid inputs to statistical functions.

use approx::assert_relative_eq;
use ndarray::{Array1, Array2};
use quickcheck::{Arbitrary, Gen, QuickCheck, TestResult};
use quickcheck_macros::quickcheck;
use scirs2_stats::{
    corrcoef,
    distributions::{norm, uniform},
    kurtosis, mean, median, pearson_r, skew, std,
    traits::Distribution,
    var,
};

/// Helper function to generate valid statistical data
fn generate_valid_data(size: usize, gen: &mut Gen) -> Array1<f64> {
    let mut data = Array1::zeros(size);
    for i in 0..size {
        data[i] = f64::arbitrary(gen).abs().min(1e6); // Bounded, positive
    }
    data
}

/// Helper function to generate valid correlation data (finite, not all equal)
fn generate_correlation_data(size: usize, gen: &mut Gen) -> (Array1<f64>, Array1<f64>) {
    let mut x = Array1::zeros(size);
    let mut y = Array1::zeros(size);

    for i in 0..size {
        x[i] = f64::arbitrary(gen).abs().min(1e6);
        y[i] = f64::arbitrary(gen).abs().min(1e6);
    }

    // Ensure some variance
    if x.var(0).unwrap() < 1e-10 {
        x[0] += 1.0;
    }
    if y.var(0).unwrap() < 1e-10 {
        y[0] += 1.0;
    }

    (x, y)
}

#[cfg(test)]
mod descriptive_stats_properties {
    use super::*;

    #[quickcheck]
    fn mean_bounds_property(data: Vec<f64>) -> TestResult {
        if data.is_empty() || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let mean_val = mean(&arr.view()).unwrap();
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        TestResult::from_bool(mean_val >= min_val && mean_val <= max_val && mean_val.is_finite())
    }

    #[quickcheck]
    fn variance_non_negative_property(data: Vec<f64>) -> TestResult {
        if data.len() < 2 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let variance = var(&arr.view(), 0).unwrap();

        TestResult::from_bool(variance >= 0.0 && variance.is_finite())
    }

    #[quickcheck]
    fn std_variance_relation_property(data: Vec<f64>) -> TestResult {
        if data.len() < 2 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let variance = var(&arr.view(), 0).unwrap();
        let std_dev = std(&arr.view(), 0).unwrap();

        TestResult::from_bool((std_dev * std_dev - variance).abs() < 1e-10 && std_dev >= 0.0)
    }

    #[quickcheck]
    fn mean_linearity_property(data: Vec<f64>, a: f64, b: f64) -> TestResult {
        if data.is_empty()
            || data.iter().any(|x| !x.is_finite())
            || !a.is_finite()
            || !b.is_finite()
        {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let original_mean = mean(&arr.view()).unwrap();

        // Transform: y = a*x + b
        let transformed = arr.mapv(|x| a * x + b);
        let transformed_mean = mean(&transformed.view()).unwrap();

        let expected_mean = a * original_mean + b;

        TestResult::from_bool((transformed_mean - expected_mean).abs() < 1e-10)
    }

    #[quickcheck]
    fn variance_scaling_property(data: Vec<f64>, a: f64) -> TestResult {
        if data.len() < 2
            || data.iter().any(|x| !x.is_finite())
            || !a.is_finite()
            || a.abs() > 1000.0
        {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data.clone());
        let original_var = var(&arr.view(), 0).unwrap();

        // Transform: y = a*x
        let transformed = arr.mapv(|x| a * x);
        let transformed_var = var(&transformed.view(), 0).unwrap();

        let expected_var = a * a * original_var;

        TestResult::from_bool(
            (transformed_var - expected_var).abs() < 1e-8 * expected_var.abs().max(1.0),
        )
    }

    #[quickcheck]
    fn skewness_bounds_property(data: Vec<f64>) -> TestResult {
        if data.len() < 3 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        // Check that all values aren't the same (would give NaN)
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if (max_val - min_val).abs() < 1e-10 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let skewness = skew(&arr.view(), false).unwrap();

        // Skewness should be finite for valid data
        TestResult::from_bool(skewness.is_finite())
    }

    #[quickcheck]
    fn kurtosis_minimum_property(data: Vec<f64>) -> TestResult {
        if data.len() < 4 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        // Check that all values aren't the same
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if (max_val - min_val).abs() < 1e-10 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);
        let kurt = kurtosis(&arr.view(), true, false).unwrap(); // Fisher's definition

        // Fisher kurtosis should be >= -2 (theoretical minimum)
        TestResult::from_bool(kurt >= -2.0 && kurt.is_finite())
    }
}

#[cfg(test)]
mod correlation_properties {
    use super::*;

    #[quickcheck]
    fn correlation_bounds_property(x_data: Vec<f64>, y_data: Vec<f64>) -> TestResult {
        if x_data.len() != y_data.len() || x_data.len() < 2 {
            return TestResult::discard();
        }

        if x_data.iter().any(|x| !x.is_finite()) || y_data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let x = Array1::from_vec(x_data);
        let y = Array1::from_vec(y_data);

        // Check for zero variance (would cause division by zero)
        let x_var = var(&x.view(), 0).unwrap();
        let y_var = var(&y.view(), 0).unwrap();
        if x_var < 1e-10 || y_var < 1e-10 {
            return TestResult::discard();
        }

        let correlation = pearson_r(&x.view(), &y.view()).unwrap();

        TestResult::from_bool(correlation >= -1.0 && correlation <= 1.0 && correlation.is_finite())
    }

    #[quickcheck]
    fn correlation_symmetry_property(x_data: Vec<f64>, y_data: Vec<f64>) -> TestResult {
        if x_data.len() != y_data.len() || x_data.len() < 2 {
            return TestResult::discard();
        }

        if x_data.iter().any(|x| !x.is_finite()) || y_data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let x = Array1::from_vec(x_data);
        let y = Array1::from_vec(y_data);

        // Check for zero variance
        let x_var = var(&x.view(), 0).unwrap();
        let y_var = var(&y.view(), 0).unwrap();
        if x_var < 1e-10 || y_var < 1e-10 {
            return TestResult::discard();
        }

        let corr_xy = pearson_r(&x.view(), &y.view()).unwrap();
        let corr_yx = pearson_r(&y.view(), &x.view()).unwrap();

        TestResult::from_bool((corr_xy - corr_yx).abs() < 1e-10)
    }

    #[quickcheck]
    fn perfect_correlation_property(data: Vec<f64>, a: f64, b: f64) -> TestResult {
        if data.len() < 2 || data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        if !a.is_finite() || !b.is_finite() || a.abs() < 1e-10 {
            return TestResult::discard();
        }

        let x = Array1::from_vec(data);
        let y = x.mapv(|val| a * val + b);

        // Check for zero variance in x
        let x_var = var(&x.view(), 0).unwrap();
        if x_var < 1e-10 {
            return TestResult::discard();
        }

        let correlation = pearson_r(&x.view(), &y.view()).unwrap();
        let expected = if a > 0.0 { 1.0 } else { -1.0 };

        TestResult::from_bool((correlation - expected).abs() < 1e-10)
    }

    #[quickcheck]
    fn correlation_matrix_diagonal_property(data: Vec<Vec<f64>>) -> TestResult {
        if data.len() < 2 || data.iter().any(|row| row.len() < 2) {
            return TestResult::discard();
        }

        // Ensure all rows have the same length
        let n_cols = data[0].len();
        if !data.iter().all(|row| row.len() == n_cols) {
            return TestResult::discard();
        }

        // Check for finite values and non-zero variance
        let mut matrix_data = Vec::new();
        for row in &data {
            if row.iter().any(|x| !x.is_finite()) {
                return TestResult::discard();
            }
            matrix_data.extend_from_slice(row);
        }

        let matrix = Array2::from_shape_vec((data.len(), n_cols), matrix_data).unwrap();

        // Check each column has non-zero variance
        for j in 0..n_cols {
            let col = matrix.column(j);
            let col_var = var(&col, 0).unwrap();
            if col_var < 1e-10 {
                return TestResult::discard();
            }
        }

        let corr_matrix = corrcoef(&matrix.view(), "pearson").unwrap();

        // Check that diagonal elements are 1.0
        let mut diagonal_ok = true;
        for i in 0..corr_matrix.nrows() {
            if (corr_matrix[[i, i]] - 1.0).abs() > 1e-10 {
                diagonal_ok = false;
                break;
            }
        }

        TestResult::from_bool(diagonal_ok)
    }
}

#[cfg(test)]
mod distribution_properties {
    use super::*;

    #[quickcheck]
    fn normal_pdf_non_negative_property(mu: f64, sigma: f64, x: f64) -> TestResult {
        if !mu.is_finite() || !sigma.is_finite() || !x.is_finite() || sigma <= 0.0 {
            return TestResult::discard();
        }

        if sigma > 1000.0 || mu.abs() > 1000.0 || x.abs() > 1000.0 {
            return TestResult::discard();
        }

        let normal = norm(mu, sigma).unwrap();
        let pdf_value = normal.pdf(x);

        TestResult::from_bool(pdf_value >= 0.0 && pdf_value.is_finite())
    }

    #[quickcheck]
    fn normal_cdf_monotonic_property(mu: f64, sigma: f64, x1: f64, x2: f64) -> TestResult {
        if !mu.is_finite() || !sigma.is_finite() || !x1.is_finite() || !x2.is_finite() {
            return TestResult::discard();
        }

        if sigma <= 0.0 || sigma > 1000.0 || mu.abs() > 1000.0 {
            return TestResult::discard();
        }

        if x1.abs() > 1000.0 || x2.abs() > 1000.0 {
            return TestResult::discard();
        }

        let normal = norm(mu, sigma).unwrap();
        let cdf1 = normal.cdf(x1);
        let cdf2 = normal.cdf(x2);

        TestResult::from_bool(if x1 <= x2 {
            cdf1 <= cdf2 && cdf1.is_finite() && cdf2.is_finite()
        } else {
            cdf1 >= cdf2 && cdf1.is_finite() && cdf2.is_finite()
        })
    }

    #[quickcheck]
    fn normal_cdf_bounds_property(mu: f64, sigma: f64, x: f64) -> TestResult {
        if !mu.is_finite() || !sigma.is_finite() || !x.is_finite() || sigma <= 0.0 {
            return TestResult::discard();
        }

        if sigma > 1000.0 || mu.abs() > 1000.0 || x.abs() > 1000.0 {
            return TestResult::discard();
        }

        let normal = norm(mu, sigma).unwrap();
        let cdf_value = normal.cdf(x);

        TestResult::from_bool(cdf_value >= 0.0 && cdf_value <= 1.0 && cdf_value.is_finite())
    }

    #[quickcheck]
    fn uniform_pdf_bounds_property(low: f64, high: f64, x: f64) -> TestResult {
        if !low.is_finite() || !high.is_finite() || !x.is_finite() || low >= high {
            return TestResult::discard();
        }

        if low.abs() > 1000.0 || high.abs() > 1000.0 || x.abs() > 1000.0 {
            return TestResult::discard();
        }

        let unif = uniform(low, high).unwrap();
        let pdf_value = unif.pdf(x);

        let expected_pdf = if x >= low && x < high {
            1.0 / (high - low)
        } else {
            0.0
        };

        TestResult::from_bool(
            pdf_value >= 0.0 && pdf_value.is_finite() && (pdf_value - expected_pdf).abs() < 1e-10,
        )
    }

    #[quickcheck]
    fn distribution_mean_variance_finite_property(mu: f64, sigma: f64) -> TestResult {
        if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return TestResult::discard();
        }

        if sigma > 1000.0 || mu.abs() > 1000.0 {
            return TestResult::discard();
        }

        let normal = norm(mu, sigma).unwrap();
        let dist_mean = normal.mean();
        let dist_var = normal.var();

        TestResult::from_bool(dist_mean.is_finite() && dist_var.is_finite() && dist_var >= 0.0)
    }
}

/// Run all property-based tests
#[cfg(test)]
mod test_runner {
    use super::*;

    #[test]
    fn run_comprehensive_property_tests() {
        // Configure QuickCheck for comprehensive testing
        let mut qc = QuickCheck::new()
            .tests(1000)  // Run 1000 test cases per property
            .max_tests(10000); // Allow up to 10000 attempts to find valid inputs

        println!("Running comprehensive property-based tests...");

        // Test a few key properties manually to ensure they work

        // Test mean bounds property
        qc.clone().quickcheck(
            descriptive_stats_properties::mean_bounds_property as fn(Vec<f64>) -> TestResult,
        );

        // Test variance non-negative property
        qc.clone().quickcheck(
            descriptive_stats_properties::variance_non_negative_property
                as fn(Vec<f64>) -> TestResult,
        );

        // Test correlation bounds property
        qc.clone().quickcheck(
            correlation_properties::correlation_bounds_property
                as fn(Vec<f64>, Vec<f64>) -> TestResult,
        );

        println!("All property-based tests passed!");
    }
}
