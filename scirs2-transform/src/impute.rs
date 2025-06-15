//! Missing value imputation utilities
//!
//! This module provides methods for handling missing values in datasets,
//! which is a crucial preprocessing step for machine learning.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use num_traits::{Float, NumCast};

use crate::error::{Result, TransformError};

/// Strategy for imputing missing values
#[derive(Debug, Clone, PartialEq)]
pub enum ImputeStrategy {
    /// Replace missing values with mean of the feature
    Mean,
    /// Replace missing values with median of the feature
    Median,
    /// Replace missing values with most frequent value
    MostFrequent,
    /// Replace missing values with a constant value
    Constant(f64),
}

/// SimpleImputer for filling missing values
///
/// This transformer fills missing values using simple strategies like mean,
/// median, most frequent value, or a constant value.
pub struct SimpleImputer {
    /// Strategy for imputation
    strategy: ImputeStrategy,
    /// Missing value indicator (what value is considered missing)
    missing_values: f64,
    /// Values used for imputation (computed during fit)
    statistics_: Option<Array1<f64>>,
}

impl SimpleImputer {
    /// Creates a new SimpleImputer
    ///
    /// # Arguments
    /// * `strategy` - The imputation strategy to use
    /// * `missing_values` - The value that represents missing data (default: NaN)
    ///
    /// # Returns
    /// * A new SimpleImputer instance
    pub fn new(strategy: ImputeStrategy, missing_values: f64) -> Self {
        SimpleImputer {
            strategy,
            missing_values,
            statistics_: None,
        }
    }

    /// Creates a SimpleImputer with NaN as missing value indicator
    ///
    /// # Arguments
    /// * `strategy` - The imputation strategy to use
    ///
    /// # Returns
    /// * A new SimpleImputer instance
    pub fn with_strategy(strategy: ImputeStrategy) -> Self {
        Self::new(strategy, f64::NAN)
    }

    /// Fits the SimpleImputer to the input data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        let mut statistics = Array1::zeros(n_features);

        for j in 0..n_features {
            // Extract non-missing values for this feature
            let feature_data: Vec<f64> = x_f64
                .column(j)
                .iter()
                .filter(|&&val| !self.is_missing(val))
                .copied()
                .collect();

            if feature_data.is_empty() {
                return Err(TransformError::InvalidInput(format!(
                    "All values are missing in feature {}",
                    j
                )));
            }

            statistics[j] = match &self.strategy {
                ImputeStrategy::Mean => {
                    feature_data.iter().sum::<f64>() / feature_data.len() as f64
                }
                ImputeStrategy::Median => {
                    let mut sorted_data = feature_data.clone();
                    sorted_data
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let n = sorted_data.len();
                    if n % 2 == 0 {
                        (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0
                    } else {
                        sorted_data[n / 2]
                    }
                }
                ImputeStrategy::MostFrequent => {
                    // For numerical data, we'll find the value that appears most frequently
                    // This is a simplified implementation
                    let mut counts = std::collections::HashMap::new();
                    for &val in &feature_data {
                        *counts.entry(val.to_bits()).or_insert(0) += 1;
                    }

                    let most_frequent_bits = counts
                        .into_iter()
                        .max_by_key(|(_, count)| *count)
                        .map(|(bits, _)| bits)
                        .unwrap_or(0);

                    f64::from_bits(most_frequent_bits)
                }
                ImputeStrategy::Constant(value) => *value,
            };
        }

        self.statistics_ = Some(statistics);
        Ok(())
    }

    /// Transforms the input data using the fitted SimpleImputer
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data with imputed values
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if self.statistics_.is_none() {
            return Err(TransformError::TransformationError(
                "SimpleImputer has not been fitted".to_string(),
            ));
        }

        let statistics = self.statistics_.as_ref().unwrap();

        if n_features != statistics.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but SimpleImputer was fitted with {} features",
                n_features,
                statistics.len()
            )));
        }

        let mut transformed = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                let value = x_f64[[i, j]];
                if self.is_missing(value) {
                    transformed[[i, j]] = statistics[j];
                } else {
                    transformed[[i, j]] = value;
                }
            }
        }

        Ok(transformed)
    }

    /// Fits the SimpleImputer to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data with imputed values
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the statistics computed during fitting
    ///
    /// # Returns
    /// * `Option<&Array1<f64>>` - The statistics for each feature
    pub fn statistics(&self) -> Option<&Array1<f64>> {
        self.statistics_.as_ref()
    }

    /// Checks if a value is considered missing
    ///
    /// # Arguments
    /// * `value` - The value to check
    ///
    /// # Returns
    /// * `bool` - True if the value is missing, false otherwise
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

/// Indicator for missing values
///
/// This transformer creates a binary indicator matrix that shows where
/// missing values were located in the original data.
pub struct MissingIndicator {
    /// Missing value indicator (what value is considered missing)
    missing_values: f64,
    /// Features that have missing values (computed during fit)
    features_: Option<Vec<usize>>,
}

impl MissingIndicator {
    /// Creates a new MissingIndicator
    ///
    /// # Arguments
    /// * `missing_values` - The value that represents missing data (default: NaN)
    ///
    /// # Returns
    /// * A new MissingIndicator instance
    pub fn new(missing_values: f64) -> Self {
        MissingIndicator {
            missing_values,
            features_: None,
        }
    }

    /// Creates a MissingIndicator with NaN as missing value indicator
    pub fn with_nan() -> Self {
        Self::new(f64::NAN)
    }

    /// Fits the MissingIndicator to the input data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_features = x_f64.shape()[1];
        let mut features_with_missing = Vec::new();

        for j in 0..n_features {
            let has_missing = x_f64.column(j).iter().any(|&val| self.is_missing(val));
            if has_missing {
                features_with_missing.push(j);
            }
        }

        self.features_ = Some(features_with_missing);
        Ok(())
    }

    /// Transforms the input data to create missing value indicators
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - Binary indicator matrix, shape (n_samples, n_features_with_missing)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];

        if self.features_.is_none() {
            return Err(TransformError::TransformationError(
                "MissingIndicator has not been fitted".to_string(),
            ));
        }

        let features_with_missing = self.features_.as_ref().unwrap();
        let n_output_features = features_with_missing.len();

        let mut indicators = Array2::zeros((n_samples, n_output_features));

        for i in 0..n_samples {
            for (out_j, &orig_j) in features_with_missing.iter().enumerate() {
                if self.is_missing(x_f64[[i, orig_j]]) {
                    indicators[[i, out_j]] = 1.0;
                }
            }
        }

        Ok(indicators)
    }

    /// Fits the MissingIndicator to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - Binary indicator matrix
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the features that have missing values
    ///
    /// # Returns
    /// * `Option<&Vec<usize>>` - Indices of features with missing values
    pub fn features(&self) -> Option<&Vec<usize>> {
        self.features_.as_ref()
    }

    /// Checks if a value is considered missing
    ///
    /// # Arguments
    /// * `value` - The value to check
    ///
    /// # Returns
    /// * `bool` - True if the value is missing, false otherwise
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    #[test]
    fn test_simple_imputer_mean() {
        // Create test data with NaN values
        let data = Array::from_shape_vec(
            (4, 3),
            vec![
                1.0,
                2.0,
                3.0,
                f64::NAN,
                5.0,
                6.0,
                7.0,
                f64::NAN,
                9.0,
                10.0,
                11.0,
                f64::NAN,
            ],
        )
        .unwrap();

        let mut imputer = SimpleImputer::with_strategy(ImputeStrategy::Mean);
        let transformed = imputer.fit_transform(&data).unwrap();

        // Check shape is preserved
        assert_eq!(transformed.shape(), &[4, 3]);

        // Check that mean values were used for imputation
        // Column 0: mean of [1.0, 7.0, 10.0] = 6.0
        // Column 1: mean of [2.0, 5.0, 11.0] = 6.0
        // Column 2: mean of [3.0, 6.0, 9.0] = 6.0

        assert_abs_diff_eq!(transformed[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 0]], 6.0, epsilon = 1e-10); // Imputed
        assert_abs_diff_eq!(transformed[[2, 0]], 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[3, 0]], 10.0, epsilon = 1e-10);

        assert_abs_diff_eq!(transformed[[0, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 1]], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 1]], 6.0, epsilon = 1e-10); // Imputed
        assert_abs_diff_eq!(transformed[[3, 1]], 11.0, epsilon = 1e-10);

        assert_abs_diff_eq!(transformed[[0, 2]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 2]], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 2]], 9.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[3, 2]], 6.0, epsilon = 1e-10); // Imputed
    }

    #[test]
    fn test_simple_imputer_median() {
        // Create test data with NaN values
        let data = Array::from_shape_vec(
            (5, 2),
            vec![
                1.0,
                10.0,
                f64::NAN,
                20.0,
                3.0,
                f64::NAN,
                4.0,
                40.0,
                5.0,
                50.0,
            ],
        )
        .unwrap();

        let mut imputer = SimpleImputer::with_strategy(ImputeStrategy::Median);
        let transformed = imputer.fit_transform(&data).unwrap();

        // Check shape is preserved
        assert_eq!(transformed.shape(), &[5, 2]);

        // Column 0: median of [1.0, 3.0, 4.0, 5.0] = 3.5
        // Column 1: median of [10.0, 20.0, 40.0, 50.0] = 30.0

        assert_abs_diff_eq!(transformed[[1, 0]], 3.5, epsilon = 1e-10); // Imputed
        assert_abs_diff_eq!(transformed[[2, 1]], 30.0, epsilon = 1e-10); // Imputed
    }

    #[test]
    fn test_simple_imputer_constant() {
        // Create test data with NaN values
        let data =
            Array::from_shape_vec((3, 2), vec![1.0, f64::NAN, f64::NAN, 3.0, 4.0, 5.0]).unwrap();

        let mut imputer = SimpleImputer::with_strategy(ImputeStrategy::Constant(99.0));
        let transformed = imputer.fit_transform(&data).unwrap();

        // Check that constant value was used for imputation
        assert_abs_diff_eq!(transformed[[0, 1]], 99.0, epsilon = 1e-10); // Imputed
        assert_abs_diff_eq!(transformed[[1, 0]], 99.0, epsilon = 1e-10); // Imputed

        // Non-missing values should remain unchanged
        assert_abs_diff_eq!(transformed[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[1, 1]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 0]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed[[2, 1]], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_missing_indicator() {
        // Create test data with NaN values
        let data = Array::from_shape_vec(
            (3, 4),
            vec![
                1.0,
                f64::NAN,
                3.0,
                4.0,
                f64::NAN,
                6.0,
                f64::NAN,
                8.0,
                9.0,
                10.0,
                11.0,
                f64::NAN,
            ],
        )
        .unwrap();

        let mut indicator = MissingIndicator::with_nan();
        let indicators = indicator.fit_transform(&data).unwrap();

        // All features have missing values, so output shape should be (3, 4)
        assert_eq!(indicators.shape(), &[3, 4]);

        // Check indicators
        assert_abs_diff_eq!(indicators[[0, 0]], 0.0, epsilon = 1e-10); // Not missing
        assert_abs_diff_eq!(indicators[[0, 1]], 1.0, epsilon = 1e-10); // Missing
        assert_abs_diff_eq!(indicators[[0, 2]], 0.0, epsilon = 1e-10); // Not missing
        assert_abs_diff_eq!(indicators[[0, 3]], 0.0, epsilon = 1e-10); // Not missing

        assert_abs_diff_eq!(indicators[[1, 0]], 1.0, epsilon = 1e-10); // Missing
        assert_abs_diff_eq!(indicators[[1, 1]], 0.0, epsilon = 1e-10); // Not missing
        assert_abs_diff_eq!(indicators[[1, 2]], 1.0, epsilon = 1e-10); // Missing
        assert_abs_diff_eq!(indicators[[1, 3]], 0.0, epsilon = 1e-10); // Not missing

        assert_abs_diff_eq!(indicators[[2, 0]], 0.0, epsilon = 1e-10); // Not missing
        assert_abs_diff_eq!(indicators[[2, 1]], 0.0, epsilon = 1e-10); // Not missing
        assert_abs_diff_eq!(indicators[[2, 2]], 0.0, epsilon = 1e-10); // Not missing
        assert_abs_diff_eq!(indicators[[2, 3]], 1.0, epsilon = 1e-10); // Missing
    }

    #[test]
    fn test_imputer_errors() {
        // Test error when all values are missing in a feature
        let data = Array::from_shape_vec((2, 2), vec![f64::NAN, 1.0, f64::NAN, 2.0]).unwrap();

        let mut imputer = SimpleImputer::with_strategy(ImputeStrategy::Mean);
        assert!(imputer.fit(&data).is_err());
    }
}
