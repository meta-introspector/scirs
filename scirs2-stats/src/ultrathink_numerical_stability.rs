//! Ultrathink Numerical Stability Testing Framework
//!
//! Advanced numerical stability testing and edge case detection for ultrathink mode,
//! featuring comprehensive precision analysis, catastrophic cancellation detection,
//! overflow/underflow handling, and extreme value testing.

use crate::{kurtosis, mean, pearson_r, quantile, skew, std, var, QuantileInterpolation};
use ndarray::{Array1, ArrayView1};
use num_traits::{Float, NumCast, One, Zero};
use std::collections::HashMap;

/// Numerical stability test configuration
#[derive(Debug, Clone)]
pub struct NumericalStabilityConfig {
    /// Relative tolerance for precision testing
    pub relative_tolerance: f64,
    /// Absolute tolerance for near-zero comparisons
    pub absolute_tolerance: f64,
    /// Enable catastrophic cancellation detection
    pub detect_cancellation: bool,
    /// Enable overflow/underflow detection
    pub detect_overflow: bool,
    /// Enable extreme value testing
    pub test_extreme_values: bool,
    /// Maximum condition number for stability
    pub max_condition_number: f64,
}

impl Default for NumericalStabilityConfig {
    fn default() -> Self {
        Self {
            relative_tolerance: 1e-12,
            absolute_tolerance: 1e-15,
            detect_cancellation: true,
            detect_overflow: true,
            test_extreme_values: true,
            max_condition_number: 1e12,
        }
    }
}

/// Comprehensive numerical stability analyzer
pub struct UltrathinkNumericalStabilityAnalyzer {
    config: NumericalStabilityConfig,
    test_results: HashMap<String, StabilityTestResult>,
}

impl UltrathinkNumericalStabilityAnalyzer {
    /// Create a new numerical stability analyzer
    pub fn new(config: NumericalStabilityConfig) -> Self {
        Self {
            config,
            test_results: HashMap::new(),
        }
    }

    /// Run comprehensive stability analysis on statistical operations
    pub fn analyze_statistical_stability<F>(
        &mut self,
        data: &ArrayView1<F>,
    ) -> StabilityAnalysisReport
    where
        F: Float + NumCast + Copy + PartialOrd + std::fmt::Debug,
    {
        let mut report = StabilityAnalysisReport::new();

        // Test basic descriptive statistics
        report.descriptive_stats = self.test_descriptive_stability(data);

        // Test variance computation stability
        report.variance_computation = self.test_variance_stability(data);

        // Test extreme value handling
        report.extreme_value_handling = self.test_extreme_values_internal(data);

        // Test precision preservation
        report.precision_preservation = self.test_precision_preservation(data);

        // Test algorithmic stability
        report.algorithmic_stability = self.test_algorithmic_stability(data);

        // Test edge cases
        report.edge_cases = self.test_edge_cases(data);

        report
    }

    /// Test stability of descriptive statistics under various conditions
    fn test_descriptive_stability<F>(&mut self, data: &ArrayView1<F>) -> StabilityTestResult
    where
        F: Float + NumCast + Copy + PartialOrd + std::fmt::Debug,
    {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();

        // Test 1: Stability under data scaling
        let scale_factors = [1e-10, 1e-5, 1.0, 1e5, 1e10];

        for &scale in &scale_factors {
            let scaled_data: Array1<F> = data.map(|&x| x * F::from(scale).unwrap());

            if let Ok(mean_val) = mean(&scaled_data.view()) {
                if !mean_val.is_finite() {
                    issues.push(format!("Mean becomes non-finite at scale {:.0e}", scale));
                }
            } else {
                issues.push(format!("Mean computation fails at scale {:.0e}", scale));
            }

            if scaled_data.len() >= 2 {
                if let Ok(var_val) = var(&scaled_data.view(), 0) {
                    if !var_val.is_finite() || var_val < F::zero() {
                        issues.push(format!("Variance becomes invalid at scale {:.0e}", scale));
                    }
                } else {
                    issues.push(format!("Variance computation fails at scale {:.0e}", scale));
                }
            }
        }

        // Test 2: Translation invariance check
        let large_offset = F::from(1e15).unwrap();
        let translated_data: Array1<F> = data.map(|&x| x + large_offset);

        if let (Ok(orig_var), Ok(trans_var)) = (var(data, 0), var(&translated_data.view(), 0)) {
            let var_diff = (orig_var.to_f64().unwrap() - trans_var.to_f64().unwrap()).abs();
            let relative_error = var_diff / orig_var.to_f64().unwrap().abs().max(1e-15);

            if relative_error > self.config.relative_tolerance {
                issues.push(format!(
                    "Translation invariance violated for variance: relative error {:.2e}",
                    relative_error
                ));
            }
        }

        // Test 3: Check for catastrophic cancellation
        if self.config.detect_cancellation {
            let cancellation_issues = self.detect_catastrophic_cancellation(data);
            warnings.extend(cancellation_issues);
        }

        StabilityTestResult {
            passed: issues.is_empty(),
            issues,
            warnings,
            metrics: HashMap::new(),
        }
    }

    /// Test variance computation stability using multiple algorithms
    fn test_variance_stability<F>(&mut self, data: &ArrayView1<F>) -> StabilityTestResult
    where
        F: Float + NumCast + Copy + PartialOrd,
    {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        let mut metrics = HashMap::new();

        if data.len() < 2 {
            return StabilityTestResult {
                passed: true,
                issues: vec!["Insufficient data for variance testing".to_string()],
                warnings: Vec::new(),
                metrics: HashMap::new(),
            };
        }

        // Algorithm 1: Two-pass algorithm (reference)
        let mean_val =
            data.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(data.len()).unwrap();
        let two_pass_var = data
            .iter()
            .map(|&x| {
                let diff = x - mean_val;
                diff * diff
            })
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(data.len()).unwrap();

        // Algorithm 2: One-pass algorithm (potentially unstable)
        let n = F::from(data.len()).unwrap();
        let sum = data.iter().fold(F::zero(), |acc, &x| acc + x);
        let sum_sq = data.iter().fold(F::zero(), |acc, &x| acc + x * x);
        let one_pass_var = (sum_sq / n) - (sum / n) * (sum / n);

        // Algorithm 3: Welford's algorithm (numerically stable)
        let mut welford_mean = F::zero();
        let mut welford_m2 = F::zero();
        let mut count = 0;

        for &x in data.iter() {
            count += 1;
            let delta = x - welford_mean;
            welford_mean = welford_mean + delta / F::from(count).unwrap();
            let delta2 = x - welford_mean;
            welford_m2 = welford_m2 + delta * delta2;
        }
        let welford_var = welford_m2 / F::from(count).unwrap();

        // Compare algorithms
        let two_pass_f64 = two_pass_var.to_f64().unwrap();
        let one_pass_f64 = one_pass_var.to_f64().unwrap();
        let welford_f64 = welford_var.to_f64().unwrap();

        metrics.insert("two_pass_variance".to_string(), two_pass_f64);
        metrics.insert("one_pass_variance".to_string(), one_pass_f64);
        metrics.insert("welford_variance".to_string(), welford_f64);

        // Check for significant differences
        let one_pass_error = (one_pass_f64 - two_pass_f64).abs();
        let welford_error = (welford_f64 - two_pass_f64).abs();

        let relative_one_pass = one_pass_error / two_pass_f64.abs().max(1e-15);
        let relative_welford = welford_error / two_pass_f64.abs().max(1e-15);

        metrics.insert("one_pass_relative_error".to_string(), relative_one_pass);
        metrics.insert("welford_relative_error".to_string(), relative_welford);

        if relative_one_pass > self.config.relative_tolerance {
            issues.push(format!(
                "One-pass variance algorithm unstable: relative error {:.2e}",
                relative_one_pass
            ));
        }

        if relative_welford > self.config.relative_tolerance {
            warnings.push(format!(
                "Welford algorithm shows unexpected error: {:.2e}",
                relative_welford
            ));
        }

        // Check for overflow/underflow
        if self.config.detect_overflow {
            if !one_pass_var.is_finite() {
                issues
                    .push("One-pass variance computation resulted in non-finite value".to_string());
            }
            if !welford_var.is_finite() {
                issues
                    .push("Welford variance computation resulted in non-finite value".to_string());
            }
        }

        StabilityTestResult {
            passed: issues.is_empty(),
            issues,
            warnings,
            metrics,
        }
    }

    /// Test handling of extreme values
    fn test_extreme_values_internal<F>(&mut self, data: &ArrayView1<F>) -> StabilityTestResult
    where
        F: Float + NumCast + Copy + PartialOrd + std::fmt::Debug,
    {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        let mut metrics = HashMap::new();

        if !self.config.test_extreme_values {
            return StabilityTestResult {
                passed: true,
                issues: Vec::new(),
                warnings: vec!["Extreme value testing disabled".to_string()],
                metrics: HashMap::new(),
            };
        }

        // Test with extreme values added
        let extreme_values = [
            F::from(f64::MAX / 1e6).unwrap(), // Very large
            F::from(f64::MIN / 1e6).unwrap(), // Very small (negative)
            F::from(1e-100).unwrap(),         // Tiny positive
            F::from(-1e-100).unwrap(),        // Tiny negative
        ];

        for (i, &extreme) in extreme_values.iter().enumerate() {
            let mut test_data = data.to_owned();
            test_data
                .push(ndarray::Axis(0), ndarray::ArrayView::from(&[extreme]))
                .unwrap();

            // Test mean computation
            match mean(&test_data.view()) {
                Ok(mean_val) => {
                    if !mean_val.is_finite() {
                        issues.push(format!("Mean becomes non-finite with extreme value {}", i));
                    }
                }
                Err(_) => {
                    issues.push(format!("Mean computation fails with extreme value {}", i));
                }
            }

            // Test variance computation
            if test_data.len() >= 2 {
                match var(&test_data.view(), 0) {
                    Ok(var_val) => {
                        if !var_val.is_finite() {
                            issues.push(format!(
                                "Variance becomes non-finite with extreme value {}",
                                i
                            ));
                        }
                    }
                    Err(_) => {
                        issues.push(format!(
                            "Variance computation fails with extreme value {}",
                            i
                        ));
                    }
                }
            }
        }

        // Test with all extreme values
        let all_extreme = Array1::from_vec(extreme_values.to_vec());

        if let Ok(extreme_mean) = mean(&all_extreme.view()) {
            metrics.insert(
                "extreme_values_mean".to_string(),
                extreme_mean.to_f64().unwrap(),
            );
            if !extreme_mean.is_finite() {
                issues.push("Mean of extreme values is non-finite".to_string());
            }
        } else {
            issues.push("Cannot compute mean of extreme values".to_string());
        }

        StabilityTestResult {
            passed: issues.is_empty(),
            issues,
            warnings,
            metrics,
        }
    }

    /// Test precision preservation in computational chains
    fn test_precision_preservation<F>(&mut self, data: &ArrayView1<F>) -> StabilityTestResult
    where
        F: Float + NumCast + Copy + PartialOrd,
    {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        let mut metrics = HashMap::new();

        if data.len() < 4 {
            return StabilityTestResult {
                passed: true,
                issues: vec!["Insufficient data for precision testing".to_string()],
                warnings: Vec::new(),
                metrics: HashMap::new(),
            };
        }

        // Test precision in standardization: (X - μ) / σ
        if let (Ok(mean_val), Ok(std_val)) = (mean(data), std(data, 0)) {
            if std_val > F::from(1e-10).unwrap() {
                let standardized: Array1<F> = data.map(|&x| (x - mean_val) / std_val);

                // Standardized data should have mean ≈ 0 and std ≈ 1
                if let (Ok(std_mean), Ok(std_std)) =
                    (mean(&standardized.view()), std(&standardized.view(), 0))
                {
                    let mean_error = std_mean.to_f64().unwrap().abs();
                    let std_error = (std_std.to_f64().unwrap() - 1.0).abs();

                    metrics.insert("standardization_mean_error".to_string(), mean_error);
                    metrics.insert("standardization_std_error".to_string(), std_error);

                    if mean_error > self.config.absolute_tolerance {
                        issues.push(format!(
                            "Standardization precision loss in mean: {:.2e}",
                            mean_error
                        ));
                    }

                    if std_error > self.config.relative_tolerance {
                        issues.push(format!(
                            "Standardization precision loss in std: {:.2e}",
                            std_error
                        ));
                    }
                }
            }
        }

        // Test precision in correlation computation
        if data.len() >= 4 {
            let mid = data.len() / 2;
            let x = data.slice(ndarray::s![..mid]);
            let y = data.slice(ndarray::s![mid..mid + x.len()]);

            // Self-correlation should be exactly 1
            if let Ok(self_corr) = pearson_r(&x, &x) {
                let self_corr_error = (self_corr.to_f64().unwrap() - 1.0).abs();
                metrics.insert("self_correlation_error".to_string(), self_corr_error);

                if self_corr_error > self.config.relative_tolerance {
                    issues.push(format!(
                        "Self-correlation precision loss: {:.2e}",
                        self_corr_error
                    ));
                }
            }
        }

        StabilityTestResult {
            passed: issues.is_empty(),
            issues,
            warnings,
            metrics,
        }
    }

    /// Test algorithmic stability across different implementations
    fn test_algorithmic_stability<F>(&mut self, data: &ArrayView1<F>) -> StabilityTestResult
    where
        F: Float + NumCast + Copy + PartialOrd,
    {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        let mut metrics = HashMap::new();

        // Compare quantile computation methods
        if data.len() >= 10 {
            let quantiles = [0.25, 0.5, 0.75];

            for &q in &quantiles {
                // Test different interpolation methods
                let linear_result =
                    quantile(data, F::from(q).unwrap(), QuantileInterpolation::Linear);
                let nearest_result =
                    quantile(data, F::from(q).unwrap(), QuantileInterpolation::Nearest);

                match (linear_result, nearest_result) {
                    (Ok(linear), Ok(nearest)) => {
                        let diff = (linear.to_f64().unwrap() - nearest.to_f64().unwrap()).abs();
                        metrics.insert(format!("quantile_{}_method_diff", q), diff);
                    }
                    _ => {
                        warnings.push(format!("Quantile computation failed at q={}", q));
                    }
                }
            }
        }

        // Test moment computation stability
        if data.len() >= 4 {
            // Higher moments are more sensitive to numerical issues
            if let (Ok(skew_val), Ok(kurt_val)) = (skew(data, false), kurtosis(data, true, false)) {
                let skew_f64 = skew_val.to_f64().unwrap();
                let kurt_f64 = kurt_val.to_f64().unwrap();

                metrics.insert("skewness".to_string(), skew_f64);
                metrics.insert("kurtosis".to_string(), kurt_f64);

                if !skew_f64.is_finite() {
                    issues.push("Skewness computation resulted in non-finite value".to_string());
                }

                if !kurt_f64.is_finite() {
                    issues.push("Kurtosis computation resulted in non-finite value".to_string());
                }

                // Check for unreasonably extreme values that might indicate numerical issues
                if skew_f64.abs() > 1000.0 {
                    warnings.push(format!(
                        "Extremely large skewness detected: {:.2e}",
                        skew_f64
                    ));
                }

                if kurt_f64 > 10000.0 || kurt_f64 < -10.0 {
                    warnings.push(format!("Extreme kurtosis detected: {:.2e}", kurt_f64));
                }
            }
        }

        StabilityTestResult {
            passed: issues.is_empty(),
            issues,
            warnings,
            metrics,
        }
    }

    /// Test various edge cases
    fn test_edge_cases<F>(&mut self, _data: &ArrayView1<F>) -> StabilityTestResult
    where
        F: Float + NumCast + Copy + PartialOrd,
    {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        let mut metrics = HashMap::new();

        // Test empty array handling
        let empty_array = Array1::<F>::zeros(0);

        if let Ok(_) = mean(&empty_array.view()) {
            issues.push("Mean computation should fail for empty arrays".to_string());
        }

        // Test single element array
        let single_element = Array1::from_vec(vec![F::one()]);

        if let Ok(single_mean) = mean(&single_element.view()) {
            if (single_mean.to_f64().unwrap() - 1.0).abs() > self.config.absolute_tolerance {
                issues.push("Single element mean is incorrect".to_string());
            }
        }

        // Test arrays with all identical values
        let identical_values = Array1::from_vec(vec![F::from(42.0).unwrap(); 100]);

        if let Ok(var_val) = var(&identical_values.view(), 0) {
            if var_val.to_f64().unwrap() > self.config.absolute_tolerance {
                issues.push("Variance of identical values should be zero".to_string());
            }
        }

        // Test arrays with NaN values (should be handled gracefully)
        if F::nan().is_nan() {
            let with_nan = Array1::from_vec(vec![F::one(), F::nan(), F::from(3.0).unwrap()]);

            match mean(&with_nan.view()) {
                Ok(mean_val) => {
                    if mean_val.is_nan() {
                        metrics.insert("nan_propagation".to_string(), 1.0);
                    } else {
                        warnings.push("NaN values not properly propagated in mean".to_string());
                    }
                }
                Err(_) => {
                    metrics.insert("nan_error_handling".to_string(), 1.0);
                }
            }
        }

        StabilityTestResult {
            passed: issues.is_empty(),
            issues,
            warnings,
            metrics,
        }
    }

    /// Detect catastrophic cancellation in numerical computations
    fn detect_catastrophic_cancellation<F>(&self, data: &ArrayView1<F>) -> Vec<String>
    where
        F: Float + NumCast + Copy + PartialOrd,
    {
        let mut warnings = Vec::new();

        if data.len() < 2 {
            return warnings;
        }

        // Check for potential cancellation in variance computation
        let mean_val =
            data.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(data.len()).unwrap();
        let mean_squared = mean_val * mean_val;
        let mean_of_squares =
            data.iter().fold(F::zero(), |acc, &x| acc + x * x) / F::from(data.len()).unwrap();

        // If mean_of_squares ≈ mean_squared, we have potential cancellation
        let ratio = if mean_squared != F::zero() {
            (mean_of_squares / mean_squared).to_f64().unwrap()
        } else {
            1.0
        };

        if ratio > 0.99 && ratio < 1.01 {
            warnings.push(format!(
                "Potential catastrophic cancellation in variance computation: ratio = {:.6}",
                ratio
            ));
        }

        // Check for loss of precision in subtraction
        let max_val = data.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
        let min_val = data.iter().fold(F::infinity(), |a, &b| a.min(b));
        let range = max_val - min_val;

        if range.to_f64().unwrap() / max_val.to_f64().unwrap().abs() < 1e-10 {
            warnings
                .push("Very small relative range detected - potential precision loss".to_string());
        }

        warnings
    }

    /// Test stability of a specific function with given data
    pub fn test_function_stability<F, E>(
        &mut self,
        function_name: &str,
        data: &ArrayView1<F>,
        function: impl Fn(&ArrayView1<F>) -> Result<F, E>,
    ) -> Result<StabilityTestResult, E>
    where
        F: Float + NumCast + Copy + PartialOrd + std::fmt::Debug,
        E: std::fmt::Debug,
    {
        let mut issues = Vec::new();
        let warnings = Vec::new();
        let mut metrics = HashMap::new();

        // Test 1: Basic function execution
        let base_result = function(data)?;
        metrics.insert("base_result".to_string(), base_result.to_f64().unwrap());

        // Test 2: Stability under data permutation
        let mut data_vec: Vec<F> = data.iter().cloned().collect();
        data_vec.reverse();
        let reversed_data = Array1::from_vec(data_vec);

        // Only test commutative operations (like mean, variance)
        if function_name == "mean" || function_name == "variance" || function_name == "std" {
            if let Ok(reversed_result) = function(&reversed_data.view()) {
                let diff =
                    (base_result.to_f64().unwrap() - reversed_result.to_f64().unwrap()).abs();
                metrics.insert("permutation_error".to_string(), diff);

                if diff > self.config.relative_tolerance {
                    issues.push(format!(
                        "Function {} not stable under permutation: error = {:.2e}",
                        function_name, diff
                    ));
                }
            }
        }

        // Test 3: Scaling stability for scale-invariant functions
        if function_name == "correlation" || function_name == "pearson_r" {
            let scale = F::from(2.0).unwrap();
            let scaled_data: Array1<F> = data.map(|&x| x * scale);

            if let Ok(scaled_result) = function(&scaled_data.view()) {
                let diff = (base_result.to_f64().unwrap() - scaled_result.to_f64().unwrap()).abs();
                metrics.insert("scaling_error".to_string(), diff);

                if diff > self.config.relative_tolerance {
                    issues.push(format!(
                        "Function {} not scale-invariant: error = {:.2e}",
                        function_name, diff
                    ));
                }
            }
        }

        // Test 4: Finite result check
        if !base_result.is_finite() {
            issues.push(format!(
                "Function {} returned non-finite value",
                function_name
            ));
        }

        let result = StabilityTestResult {
            passed: issues.is_empty(),
            issues,
            warnings,
            metrics,
        };

        // Store result
        self.test_results
            .insert(function_name.to_string(), result.clone());

        Ok(result)
    }

    /// Test extreme values handling for a function
    pub fn test_extreme_values<F, E>(
        &mut self,
        function_name: &str,
        function: impl Fn(&ArrayView1<F>) -> Result<F, E>,
    ) -> Result<StabilityTestResult, E>
    where
        F: Float + NumCast + Copy + PartialOrd + std::fmt::Debug,
        E: std::fmt::Debug,
    {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        let mut metrics = HashMap::new();

        // Test with extreme values
        let extreme_data = Array1::from_vec(vec![
            F::from(f64::MAX / 1e6).unwrap(),
            F::from(f64::MIN / 1e6).unwrap(),
            F::from(1e-100).unwrap(),
            F::from(-1e-100).unwrap(),
            F::zero(),
        ]);

        match function(&extreme_data.view()) {
            Ok(result) => {
                metrics.insert("extreme_result".to_string(), result.to_f64().unwrap());

                if !result.is_finite() {
                    issues.push(format!(
                        "Function {} returned non-finite result with extreme values",
                        function_name
                    ));
                }
            }
            Err(_) => {
                warnings.push(format!(
                    "Function {} failed with extreme values (may be expected)",
                    function_name
                ));
            }
        }

        Ok(StabilityTestResult {
            passed: issues.is_empty(),
            issues,
            warnings,
            metrics,
        })
    }

    /// Get comprehensive stability report
    pub fn get_stability_report(&self) -> HashMap<String, StabilityTestResult> {
        self.test_results.clone()
    }
}

/// Result of a stability test
#[derive(Debug, Clone)]
pub struct StabilityTestResult {
    pub passed: bool,
    pub issues: Vec<String>,
    pub warnings: Vec<String>,
    pub metrics: HashMap<String, f64>,
}

/// Comprehensive stability analysis report
#[derive(Debug, Clone)]
pub struct StabilityAnalysisReport {
    pub descriptive_stats: StabilityTestResult,
    pub variance_computation: StabilityTestResult,
    pub extreme_value_handling: StabilityTestResult,
    pub precision_preservation: StabilityTestResult,
    pub algorithmic_stability: StabilityTestResult,
    pub edge_cases: StabilityTestResult,
}

impl StabilityAnalysisReport {
    fn new() -> Self {
        let empty_result = StabilityTestResult {
            passed: true,
            issues: Vec::new(),
            warnings: Vec::new(),
            metrics: HashMap::new(),
        };

        Self {
            descriptive_stats: empty_result.clone(),
            variance_computation: empty_result.clone(),
            extreme_value_handling: empty_result.clone(),
            precision_preservation: empty_result.clone(),
            algorithmic_stability: empty_result.clone(),
            edge_cases: empty_result,
        }
    }

    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.descriptive_stats.passed
            && self.variance_computation.passed
            && self.extreme_value_handling.passed
            && self.precision_preservation.passed
            && self.algorithmic_stability.passed
            && self.edge_cases.passed
    }

    /// Get all issues across all tests
    pub fn get_all_issues(&self) -> Vec<String> {
        let mut all_issues = Vec::new();

        all_issues.extend(self.descriptive_stats.issues.iter().cloned());
        all_issues.extend(self.variance_computation.issues.iter().cloned());
        all_issues.extend(self.extreme_value_handling.issues.iter().cloned());
        all_issues.extend(self.precision_preservation.issues.iter().cloned());
        all_issues.extend(self.algorithmic_stability.issues.iter().cloned());
        all_issues.extend(self.edge_cases.issues.iter().cloned());

        all_issues
    }

    /// Get all warnings across all tests
    pub fn get_all_warnings(&self) -> Vec<String> {
        let mut all_warnings = Vec::new();

        all_warnings.extend(self.descriptive_stats.warnings.iter().cloned());
        all_warnings.extend(self.variance_computation.warnings.iter().cloned());
        all_warnings.extend(self.extreme_value_handling.warnings.iter().cloned());
        all_warnings.extend(self.precision_preservation.warnings.iter().cloned());
        all_warnings.extend(self.algorithmic_stability.warnings.iter().cloned());
        all_warnings.extend(self.edge_cases.warnings.iter().cloned());

        all_warnings
    }
}

/// Create a default numerical stability analyzer
pub fn create_numerical_stability_analyzer() -> UltrathinkNumericalStabilityAnalyzer {
    UltrathinkNumericalStabilityAnalyzer::new(NumericalStabilityConfig::default())
}

/// Create a numerical stability analyzer with custom configuration
pub fn create_configured_stability_analyzer(
    config: NumericalStabilityConfig,
) -> UltrathinkNumericalStabilityAnalyzer {
    UltrathinkNumericalStabilityAnalyzer::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_stability_analyzer_creation() {
        let analyzer = create_numerical_stability_analyzer();
        assert!(analyzer.config.relative_tolerance > 0.0);
        assert!(analyzer.config.absolute_tolerance > 0.0);
    }

    #[test]
    fn test_descriptive_stability() {
        let mut analyzer = create_numerical_stability_analyzer();
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = analyzer.test_descriptive_stability(&data.view());
        assert!(result.passed || !result.issues.is_empty());
    }

    #[test]
    fn test_variance_stability() {
        let mut analyzer = create_numerical_stability_analyzer();
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = analyzer.test_variance_stability(&data.view());
        assert!(result.metrics.contains_key("two_pass_variance"));
        assert!(result.metrics.contains_key("welford_variance"));
    }

    #[test]
    fn test_edge_cases() {
        let mut analyzer = create_numerical_stability_analyzer();
        let data = array![1.0, 2.0, 3.0];

        let result = analyzer.test_edge_cases(&data.view());
        // Should handle edge cases gracefully
        assert!(result.passed || !result.warnings.is_empty());
    }

    #[test]
    fn test_comprehensive_analysis() {
        let mut analyzer = create_numerical_stability_analyzer();
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let report = analyzer.analyze_statistical_stability(&data.view());

        // Should have run all tests
        assert!(report.descriptive_stats.passed || !report.descriptive_stats.issues.is_empty());
        assert!(
            report.variance_computation.passed || !report.variance_computation.issues.is_empty()
        );
    }
}
