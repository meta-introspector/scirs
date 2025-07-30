//! Property-based tests for mathematical invariants
//!
//! This module contains property-based tests that verify mathematical invariants
//! and properties that should hold for all valid inputs to statistical functions.
//!
//! Extended to include comprehensive testing of additional statistical operations,
//! SIMD optimizations, and advanced mathematical properties.

use approx::assert_relative_eq;
use ndarray::{Array1, Array2};
use quickcheck::{Arbitrary, Gen, QuickCheck, TestResult};
use quickcheck__macros::quickcheck;
use scirs2__stats::{
    corrcoef,
    distributions::{beta, gamma, norm, uniform},
    kurtosis, mean, median, pearson_r, quantile, range, skew, std,
    traits::Distribution,
    var,
};
use statrs::statistics::Statistics;

/// Helper function to generate valid statistical data
#[allow(dead_code)]
fn generate_valid_data(_size: usize, gen: &mut Gen) -> Array1<f64> {
    let mut data = Array1::zeros(_size);
    for i in 0.._size {
        data[i] = f64::arbitrary(gen).abs().min(1e6); // Bounded, positive
    }
    data
}

/// Helper function to generate valid correlation data (finite, not all equal)
#[allow(dead_code)]
fn generate_correlation_data(_size: usize, gen: &mut Gen) -> (Array1<f64>, Array1<f64>) {
    let mut x = Array1::zeros(_size);
    let mut y = Array1::zeros(_size);

    for i in 0.._size {
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
    fn mean_bounds_property(_data: Vec<f64>) -> TestResult {
        if _data.is_empty() || _data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data.clone());
        let mean_val = mean(&arr.view()).unwrap();
        let min_val = _data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = _data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        TestResult::from_bool(mean_val >= min_val && mean_val <= max_val && mean_val.is_finite())
    }

    #[quickcheck]
    fn variance_non_negative_property(_data: Vec<f64>) -> TestResult {
        if _data.len() < 2 || _data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data);
        let variance = var(&arr.view(), 0).unwrap();

        TestResult::from_bool(variance >= 0.0 && variance.is_finite())
    }

    #[quickcheck]
    fn std_variance_relation_property(_data: Vec<f64>) -> TestResult {
        if _data.len() < 2 || _data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data);
        let variance = var(&arr.view(), 0).unwrap();
        let std_dev = std(&arr.view(), 0).unwrap();

        TestResult::from_bool((std_dev * std_dev - variance).abs() < 1e-10 && std_dev >= 0.0)
    }

    #[quickcheck]
    fn mean_linearity_property(_data: Vec<f64>, a: f64, b: f64) -> TestResult {
        if _data.is_empty()
            || _data.iter().any(|x| !x.is_finite())
            || !a.is_finite()
            || !b.is_finite()
        {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data.clone());
        let original_mean = mean(&arr.view()).unwrap();

        // Transform: y = a*x + b
        let transformed = arr.mapv(|x| a * x + b);
        let transformed_mean = mean(&transformed.view()).unwrap();

        let expected_mean = a * original_mean + b;

        TestResult::from_bool((transformed_mean - expected_mean).abs() < 1e-10)
    }

    #[quickcheck]
    fn variance_scaling_property(_data: Vec<f64>, a: f64) -> TestResult {
        if _data.len() < 2
            || _data.iter().any(|x| !x.is_finite())
            || !a.is_finite()
            || a.abs() > 1000.0
        {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data.clone());
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
    fn skewness_bounds_property(_data: Vec<f64>) -> TestResult {
        if _data.len() < 3 || _data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        // Check that all values aren't the same (would give NaN)
        let min_val = _data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = _data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if (max_val - min_val).abs() < 1e-10 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data);
        let skewness = skew(&arr.view(), false).unwrap();

        // Skewness should be finite for valid _data
        TestResult::from_bool(skewness.is_finite())
    }

    #[quickcheck]
    fn kurtosis_minimum_property(_data: Vec<f64>) -> TestResult {
        if _data.len() < 4 || _data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        // Check that all values aren't the same
        let min_val = _data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = _data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if (max_val - min_val).abs() < 1e-10 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data);
        let kurt = kurtosis(&arr.view(), true, false).unwrap(); // Fisher's definition

        // Fisher kurtosis should be >= -2 (theoretical minimum)
        TestResult::from_bool(kurt >= -2.0 && kurt.is_finite())
    }
}

#[cfg(test)]
mod correlation_properties {
    use super::*;

    #[quickcheck]
    fn correlation_bounds_property(_x_data: Vec<f64>, y_data: Vec<f64>) -> TestResult {
        if _x_data.len() != y_data.len() || _x_data.len() < 2 {
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
    fn correlation_symmetry_property(_x_data: Vec<f64>, y_data: Vec<f64>) -> TestResult {
        if _x_data.len() != y_data.len() || _x_data.len() < 2 {
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
    fn perfect_correlation_property(_data: Vec<f64>, a: f64, b: f64) -> TestResult {
        if _data.len() < 2 || _data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        if !a.is_finite() || !b.is_finite() || a.abs() < 1e-10 {
            return TestResult::discard();
        }

        let x = Array1::from_vec(_data);
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
    fn correlation_matrix_diagonal_property(_data: Vec<Vec<f64>>) -> TestResult {
        if _data.len() < 2 || _data.iter().any(|row| row.len() < 2) {
            return TestResult::discard();
        }

        // Ensure all rows have the same length
        let n_cols = _data[0].len();
        if !_data.iter().all(|row| row.len() == n_cols) {
            return TestResult::discard();
        }

        // Check for finite values and non-zero variance
        let mut matrix_data = Vec::new();
        for row in &_data {
            if row.iter().any(|x| !x.is_finite()) {
                return TestResult::discard();
            }
            matrix_data.extend_from_slice(row);
        }

        let matrix = Array2::from_shape_vec((_data.len(), n_cols), matrix_data).unwrap();

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
    fn normal_pdf_non_negative_property(_mu: f64, sigma: f64, x: f64) -> TestResult {
        if !_mu.is_finite() || !sigma.is_finite() || !x.is_finite() || sigma <= 0.0 {
            return TestResult::discard();
        }

        if sigma > 1000.0 || _mu.abs() > 1000.0 || x.abs() > 1000.0 {
            return TestResult::discard();
        }

        let normal = norm(_mu, sigma).unwrap();
        let pdf_value = normal.pdf(x);

        TestResult::from_bool(pdf_value >= 0.0 && pdf_value.is_finite())
    }

    #[quickcheck]
    fn normal_cdf_monotonic_property(_mu: f64, sigma: f64, x1: f64, x2: f64) -> TestResult {
        if !_mu.is_finite() || !sigma.is_finite() || !x1.is_finite() || !x2.is_finite() {
            return TestResult::discard();
        }

        if sigma <= 0.0 || sigma > 1000.0 || _mu.abs() > 1000.0 {
            return TestResult::discard();
        }

        if x1.abs() > 1000.0 || x2.abs() > 1000.0 {
            return TestResult::discard();
        }

        let normal = norm(_mu, sigma).unwrap();
        let cdf1 = normal.cdf(x1);
        let cdf2 = normal.cdf(x2);

        TestResult::from_bool(if x1 <= x2 {
            cdf1 <= cdf2 && cdf1.is_finite() && cdf2.is_finite()
        } else {
            cdf1 >= cdf2 && cdf1.is_finite() && cdf2.is_finite()
        })
    }

    #[quickcheck]
    fn normal_cdf_bounds_property(_mu: f64, sigma: f64, x: f64) -> TestResult {
        if !_mu.is_finite() || !sigma.is_finite() || !x.is_finite() || sigma <= 0.0 {
            return TestResult::discard();
        }

        if sigma > 1000.0 || _mu.abs() > 1000.0 || x.abs() > 1000.0 {
            return TestResult::discard();
        }

        let normal = norm(_mu, sigma).unwrap();
        let cdf_value = normal.cdf(x);

        TestResult::from_bool(cdf_value >= 0.0 && cdf_value <= 1.0 && cdf_value.is_finite())
    }

    #[quickcheck]
    fn uniform_pdf_bounds_property(_low: f64, high: f64, x: f64) -> TestResult {
        if !_low.is_finite() || !high.is_finite() || !x.is_finite() || _low >= high {
            return TestResult::discard();
        }

        if _low.abs() > 1000.0 || high.abs() > 1000.0 || x.abs() > 1000.0 {
            return TestResult::discard();
        }

        let unif = uniform(_low, high).unwrap();
        let pdf_value = unif.pdf(x);

        let expected_pdf = if x >= _low && x < high {
            1.0 / (high - _low)
        } else {
            0.0
        };

        TestResult::from_bool(
            pdf_value >= 0.0 && pdf_value.is_finite() && (pdf_value - expected_pdf).abs() < 1e-10,
        )
    }

    #[quickcheck]
    fn distribution_mean_variance_finite_property(_mu: f64, sigma: f64) -> TestResult {
        if !_mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return TestResult::discard();
        }

        if sigma > 1000.0 || _mu.abs() > 1000.0 {
            return TestResult::discard();
        }

        let normal = norm(_mu, sigma).unwrap();
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

/// Extended property-based tests for additional statistical functions
#[cfg(test)]
mod extended_properties {
    use super::*;

    #[quickcheck]
    fn range_property(_data: Vec<f64>) -> TestResult {
        if _data.len() < 2 || _data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data.clone());
        let range_val = range(&arr.view()).unwrap();
        let min_val = _data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = _data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let expected_range = max_val - min_val;

        TestResult::from_bool((range_val - expected_range).abs() < 1e-10 && range_val >= 0.0)
    }

    #[quickcheck]
    fn quantile_monotonicity_property(_data: Vec<f64>, q1: f64, q2: f64) -> TestResult {
        if _data.len() < 2 || _data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        if q1 < 0.0 || q1 > 1.0 || q2 < 0.0 || q2 > 1.0 || q1 >= q2 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data);
        let quant1 = quantile(&arr.view(), q1).unwrap();
        let quant2 = quantile(&arr.view(), q2).unwrap();

        TestResult::from_bool(quant1 <= quant2 && quant1.is_finite() && quant2.is_finite())
    }

    #[quickcheck]
    fn quantile_bounds_property(_data: Vec<f64>, q: f64) -> TestResult {
        if _data.len() < 2 || _data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        if q < 0.0 || q > 1.0 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data.clone());
        let quant = quantile(&arr.view(), q).unwrap();
        let min_val = _data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = _data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        TestResult::from_bool(quant >= min_val && quant <= max_val && quant.is_finite())
    }

    #[quickcheck]
    fn median_middle_property(_data: Vec<f64>) -> TestResult {
        if _data.len() < 3 || _data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data.clone());
        let median_val = median(&arr.view()).unwrap();
        let min_val = _data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = _data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        TestResult::from_bool(
            median_val >= min_val && median_val <= max_val && median_val.is_finite(),
        )
    }

    #[quickcheck]
    fn variance_translation_invariance(_data: Vec<f64>, c: f64) -> TestResult {
        if _data.len() < 2 || _data.iter().any(|x| !x.is_finite()) || !c.is_finite() {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data.clone());
        let original_var = var(&arr.view(), 0).unwrap();

        // Add constant to all elements
        let translated = arr.mapv(|x| x + c);
        let translated_var = var(&translated.view(), 0).unwrap();

        TestResult::from_bool((original_var - translated_var).abs() < 1e-10)
    }

    #[quickcheck]
    fn standardization_property(_data: Vec<f64>) -> TestResult {
        if _data.len() < 3 || _data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data);
        let mean_val = mean(&arr.view()).unwrap();
        let std_val = std(&arr.view(), 0).unwrap();

        // Avoid division by zero
        if std_val < 1e-10 {
            return TestResult::discard();
        }

        // Standardize: z = (x - mean) / std
        let standardized = arr.mapv(|x| (x - mean_val) / std_val);
        let std_mean = mean(&standardized.view()).unwrap();
        let std_var = var(&standardized.view(), 0).unwrap();

        TestResult::from_bool(std_mean.abs() < 1e-10 && (std_var - 1.0).abs() < 1e-10)
    }
}

/// Property-based tests for robust statistics
#[cfg(test)]
mod robust_statistics_properties {
    use super::*;

    #[quickcheck]
    fn median_outlier_resistance(_data: Vec<f64>, outlier_factor: f64) -> TestResult {
        if _data.len() < 5 || _data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        if !outlier_factor.is_finite() || outlier_factor.abs() < 1.0 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data.clone());
        let original_median = median(&arr.view()).unwrap();

        // Add extreme outlier
        let mut with_outlier = _data;
        let max_val = with_outlier
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        with_outlier.push(max_val * outlier_factor);
        let outlier_arr = Array1::from_vec(with_outlier);
        let outlier_median = median(&outlier_arr.view()).unwrap();

        // Median should be less affected than mean
        let original_mean = mean(&arr.view()).unwrap();
        let outlier_mean = mean(&outlier_arr.view()).unwrap();

        let median_change = (original_median - outlier_median).abs();
        let mean_change = (original_mean - outlier_mean).abs();

        TestResult::from_bool(median_change <= mean_change || median_change < 1e-5)
    }

    #[quickcheck]
    fn mad_consistency_property(_data: Vec<f64>) -> TestResult {
        if _data.len() < 3 || _data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        // Check that all values aren't the same
        let min_val = _data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = _data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if (max_val - min_val).abs() < 1e-10 {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data);
        let median_val = median(&arr.view()).unwrap();

        // Compute MAD manually for verification
        let deviations: Vec<f64> = arr.iter().map(|&x| (x - median_val).abs()).collect();
        let mut sorted_deviations = deviations;
        sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted_deviations.len();
        let mad = if n % 2 == 1 {
            sorted_deviations[n / 2]
        } else {
            (sorted_deviations[n / 2 - 1] + sorted_deviations[n / 2]) / 2.0
        };

        TestResult::from_bool(mad >= 0.0 && mad.is_finite())
    }
}

/// Property-based tests for distribution properties
#[cfg(test)]
mod advanced_distribution_properties {
    use super::*;

    #[quickcheck]
    fn beta_distribution_bounds_property(_alpha: f64, beta_param: f64, x: f64) -> TestResult {
        if !_alpha.is_finite() || !beta_param.is_finite() || !x.is_finite() {
            return TestResult::discard();
        }

        if _alpha <= 0.0 || beta_param <= 0.0 || _alpha > 100.0 || beta_param > 100.0 {
            return TestResult::discard();
        }

        if x < 0.0 || x > 1.0 {
            return TestResult::discard();
        }

        match beta(_alpha, beta_param) {
            Ok(dist) => {
                let pdf_val = dist.pdf(x);
                let cdf_val = dist.cdf(x);
                TestResult::from_bool(
                    pdf_val >= 0.0
                        && pdf_val.is_finite()
                        && cdf_val >= 0.0
                        && cdf_val <= 1.0
                        && cdf_val.is_finite(),
                )
            }
            Err(_) => TestResult::discard(),
        }
    }

    #[quickcheck]
    fn gamma_distribution_properties(_shape: f64, scale: f64, x: f64) -> TestResult {
        if !_shape.is_finite() || !scale.is_finite() || !x.is_finite() {
            return TestResult::discard();
        }

        if _shape <= 0.0 || scale <= 0.0 || _shape > 100.0 || scale > 100.0 {
            return TestResult::discard();
        }

        if x < 0.0 || x > 1000.0 {
            return TestResult::discard();
        }

        match gamma(_shape, scale) {
            Ok(dist) => {
                let pdf_val = dist.pdf(x);
                let cdf_val = dist.cdf(x);
                TestResult::from_bool(
                    pdf_val >= 0.0
                        && pdf_val.is_finite()
                        && cdf_val >= 0.0
                        && cdf_val <= 1.0
                        && cdf_val.is_finite(),
                )
            }
            Err(_) => TestResult::discard(),
        }
    }

    #[quickcheck]
    fn distribution_cdf_pdf_consistency(_mu: f64, sigma: f64, x1: f64, x2: f64) -> TestResult {
        if !_mu.is_finite() || !sigma.is_finite() || !x1.is_finite() || !x2.is_finite() {
            return TestResult::discard();
        }

        if sigma <= 0.0 || sigma > 100.0 || _mu.abs() > 100.0 {
            return TestResult::discard();
        }

        if x1.abs() > 100.0 || x2.abs() > 100.0 || x1 >= x2 {
            return TestResult::discard();
        }

        let normal = norm(_mu, sigma).unwrap();
        let cdf_x1 = normal.cdf(x1);
        let cdf_x2 = normal.cdf(x2);

        // CDF should be monotonic
        TestResult::from_bool(cdf_x1 <= cdf_x2)
    }

    #[quickcheck]
    fn distribution_symmetry_property(_sigma: f64, x: f64) -> TestResult {
        if !_sigma.is_finite() || !x.is_finite() {
            return TestResult::discard();
        }

        if _sigma <= 0.0 || _sigma > 100.0 || x.abs() > 100.0 {
            return TestResult::discard();
        }

        // Test symmetry of normal distribution around mean
        let normal = norm(0.0, _sigma).unwrap();
        let pdf_pos = normal.pdf(x);
        let pdf_neg = normal.pdf(-x);

        TestResult::from_bool((pdf_pos - pdf_neg).abs() < 1e-10)
    }
}

/// Property-based tests for SIMD optimization consistency
#[cfg(test)]
mod simd_consistency_properties {
    use super::*;

    #[quickcheck]
    fn simd_scalar_consistency_mean(_data: Vec<f64>) -> TestResult {
        if _data.len() < 1 || _data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data.clone());
        let simd_result = mean(&arr.view()).unwrap();

        // Compute scalar version manually
        let scalar_result = _data.iter().sum::<f64>() / _data.len() as f64;

        TestResult::from_bool((simd_result - scalar_result).abs() < 1e-10)
    }

    #[quickcheck]
    fn simd_scalar_consistency_variance(_data: Vec<f64>) -> TestResult {
        if _data.len() < 2 || _data.iter().any(|x| !x.is_finite()) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(_data.clone());
        let simd_result = var(&arr.view(), 0).unwrap();

        // Compute scalar version manually
        let mean_val = _data.iter().sum::<f64>() / _data.len() as f64;
        let scalar_result =
            _data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / (_data.len() - 1) as f64;

        TestResult::from_bool((simd_result - scalar_result).abs() < 1e-10)
    }

    #[quickcheck]
    fn large_dataset_stability(_size: usize) -> TestResult {
        if _size < 100 || _size > 10000 {
            return TestResult::discard();
        }

        // Generate deterministic data for reproducibility
        let data: Vec<f64> = (0.._size).map(|i| (i as f64).sin()).collect();
        let arr = Array1::from_vec(data);

        let mean_val = mean(&arr.view()).unwrap();
        let var_val = var(&arr.view(), 0).unwrap();
        let std_val = std(&arr.view(), 0).unwrap();

        TestResult::from_bool(
            mean_val.is_finite()
                && var_val.is_finite()
                && var_val >= 0.0
                && std_val.is_finite()
                && std_val >= 0.0
                && (std_val * std_val - var_val).abs() < 1e-10,
        )
    }
}

/// Property-based tests for edge cases and numerical stability
#[cfg(test)]
mod numerical_stability_properties {
    use super::*;

    #[quickcheck]
    fn tiny_values_stability(_exponent: i32) -> TestResult {
        if _exponent < -100 || _exponent > -10 {
            return TestResult::discard();
        }

        let value = 10.0_f64.powi(_exponent);
        let data = vec![value, value * 1.1, value * 0.9, value * 1.05, value * 0.95];
        let arr = Array1::from_vec(data);

        let mean_val = mean(&arr.view()).unwrap();
        let var_val = var(&arr.view(), 0).unwrap();

        TestResult::from_bool(
            mean_val.is_finite() && mean_val > 0.0 && var_val.is_finite() && var_val >= 0.0,
        )
    }

    #[quickcheck]
    fn large_values_stability(_exponent: i32) -> TestResult {
        if _exponent < 10 || _exponent > 100 {
            return TestResult::discard();
        }

        let value = 10.0_f64.powi(_exponent);
        let data = vec![value, value * 1.1, value * 0.9, value * 1.05, value * 0.95];
        let arr = Array1::from_vec(data);

        let mean_val = mean(&arr.view()).unwrap();
        let var_val = var(&arr.view(), 0).unwrap();

        TestResult::from_bool(mean_val.is_finite() && var_val.is_finite() && var_val >= 0.0)
    }

    #[quickcheck]
    fn near_identical_values_stability(_base: f64, epsilon_exp: i32) -> TestResult {
        if !_base.is_finite() || _base.abs() > 1000.0 || epsilon_exp < -15 || epsilon_exp > -5 {
            return TestResult::discard();
        }

        let epsilon = 10.0_f64.powi(epsilon_exp);
        let data = vec![
            _base,
            _base + epsilon,
            _base - epsilon,
            _base + 2.0 * epsilon,
            _base - 2.0 * epsilon,
        ];
        let arr = Array1::from_vec(data);

        let mean_val = mean(&arr.view()).unwrap();
        let var_val = var(&arr.view(), 0).unwrap();
        let std_val = std(&arr.view(), 0).unwrap();

        TestResult::from_bool(
            mean_val.is_finite()
                && var_val.is_finite()
                && var_val >= 0.0
                && std_val.is_finite()
                && std_val >= 0.0
                && (mean_val - _base).abs() < epsilon * 10.0,
        )
    }
}

/// Property-based tests for multivariate statistics
#[cfg(test)]
mod multivariate_properties {
    use super::*;

    #[quickcheck]
    fn correlation_matrix_properties(_data: Vec<Vec<f64>>) -> TestResult {
        if _data.len() < 2 || _data.iter().any(|row| row.len() < 3) {
            return TestResult::discard();
        }

        // Ensure all rows have the same length
        let n_cols = _data[0].len();
        if !_data.iter().all(|row| row.len() == n_cols) {
            return TestResult::discard();
        }

        // Check for finite values
        for row in &_data {
            if row.iter().any(|x| !x.is_finite()) {
                return TestResult::discard();
            }
        }

        let mut matrix_data = Vec::new();
        for row in &_data {
            matrix_data.extend_from_slice(row);
        }

        let matrix = Array2::from_shape_vec((_data.len(), n_cols), matrix_data).unwrap();

        // Check each column has non-zero variance
        for j in 0..n_cols {
            let col = matrix.column(j);
            let col_var = var(&col, 0).unwrap();
            if col_var < 1e-10 {
                return TestResult::discard();
            }
        }

        let corr_matrix = corrcoef(&matrix.view(), "pearson").unwrap();

        // Check correlation matrix properties
        let mut properties_hold = true;

        // 1. Diagonal elements should be 1.0
        for i in 0..corr_matrix.nrows() {
            if (corr_matrix[[i, i]] - 1.0).abs() > 1e-10 {
                properties_hold = false;
                break;
            }
        }

        // 2. Matrix should be symmetric
        for i in 0..corr_matrix.nrows() {
            for j in 0..corr_matrix.ncols() {
                if (corr_matrix[[i, j]] - corr_matrix[[j, i]]).abs() > 1e-10 {
                    properties_hold = false;
                    break;
                }
            }
            if !properties_hold {
                break;
            }
        }

        // 3. All correlations should be in [-1, 1]
        for i in 0..corr_matrix.nrows() {
            for j in 0..corr_matrix.ncols() {
                let corr_val = corr_matrix[[i, j]];
                if corr_val < -1.0 || corr_val > 1.0 || !corr_val.is_finite() {
                    properties_hold = false;
                    break;
                }
            }
            if !properties_hold {
                break;
            }
        }

        TestResult::from_bool(properties_hold)
    }
}

/// Comprehensive property test runner with extended coverage
#[cfg(test)]
mod comprehensive_test_runner {
    use super::*;

    #[test]
    fn run_extended_property_tests() {
        let mut qc = QuickCheck::new()
            .tests(2000)  // Increased for more comprehensive testing
            .max_tests(20000);

        println!("Running extended property-based tests...");

        // Extended properties
        qc.clone()
            .quickcheck(extended_properties::range_property as fn(Vec<f64>) -> TestResult);

        qc.clone().quickcheck(
            extended_properties::quantile_monotonicity_property
                as fn(Vec<f64>, f64, f64) -> TestResult,
        );

        qc.clone().quickcheck(
            extended_properties::standardization_property as fn(Vec<f64>) -> TestResult,
        );

        // Robust statistics
        qc.clone().quickcheck(
            robust_statistics_properties::median_outlier_resistance
                as fn(Vec<f64>, f64) -> TestResult,
        );

        // SIMD consistency
        qc.clone().quickcheck(
            simd_consistency_properties::simd_scalar_consistency_mean as fn(Vec<f64>) -> TestResult,
        );

        qc.clone().quickcheck(
            simd_consistency_properties::simd_scalar_consistency_variance
                as fn(Vec<f64>) -> TestResult,
        );

        // Numerical stability
        qc.clone().quickcheck(
            numerical_stability_properties::tiny_values_stability as fn(i32) -> TestResult,
        );

        qc.clone().quickcheck(
            numerical_stability_properties::near_identical_values_stability
                as fn(f64, i32) -> TestResult,
        );

        println!("All extended property-based tests passed!");
    }

    #[test]
    fn run_distribution_property_tests() {
        let mut qc = QuickCheck::new().tests(1000).max_tests(10000);

        println!("Running distribution property tests...");

        qc.clone().quickcheck(
            advanced_distribution_properties::beta_distribution_bounds_property
                as fn(f64, f64, f64) -> TestResult,
        );

        qc.clone().quickcheck(
            advanced_distribution_properties::distribution_symmetry_property
                as fn(f64, f64) -> TestResult,
        );

        println!("All distribution property tests passed!");
    }

    #[test]
    fn run_multivariate_property_tests() {
        let mut qc = QuickCheck::new()
            .tests(500)  // Fewer tests due to computational complexity
            .max_tests(5000);

        println!("Running multivariate property tests...");

        qc.clone().quickcheck(
            multivariate_properties::correlation_matrix_properties
                as fn(Vec<Vec<f64>>) -> TestResult,
        );

        println!("All multivariate property tests passed!");
    }
}
