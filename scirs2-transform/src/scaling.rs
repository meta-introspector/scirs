//! Advanced scaling and transformation methods
//!
//! This module provides sophisticated scaling methods that go beyond basic normalization,
//! including quantile transformations and robust scaling methods.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use num_traits::{Float, NumCast};

use crate::error::{Result, TransformError};

// Define a small value to use for comparison with zero
const EPSILON: f64 = 1e-10;

/// QuantileTransformer for non-linear transformations
///
/// This transformer transforms features to follow a uniform or normal distribution
/// using quantiles information. This method reduces the impact of outliers.
pub struct QuantileTransformer {
    /// Number of quantiles to estimate
    n_quantiles: usize,
    /// Output distribution ('uniform' or 'normal')
    output_distribution: String,
    /// Whether to clip transformed values to bounds [0, 1] for uniform distribution
    clip: bool,
    /// The quantiles for each feature
    quantiles: Option<Array2<f64>>,
    /// References values for each quantile
    references: Option<Array1<f64>>,
}

impl QuantileTransformer {
    /// Creates a new QuantileTransformer
    ///
    /// # Arguments
    /// * `n_quantiles` - Number of quantiles to estimate (default: 1000)
    /// * `output_distribution` - Target distribution ('uniform' or 'normal')
    /// * `clip` - Whether to clip transformed values
    ///
    /// # Returns
    /// * A new QuantileTransformer instance
    pub fn new(n_quantiles: usize, output_distribution: &str, clip: bool) -> Result<Self> {
        if n_quantiles < 2 {
            return Err(TransformError::InvalidInput(
                "n_quantiles must be at least 2".to_string(),
            ));
        }

        if output_distribution != "uniform" && output_distribution != "normal" {
            return Err(TransformError::InvalidInput(
                "output_distribution must be 'uniform' or 'normal'".to_string(),
            ));
        }

        Ok(QuantileTransformer {
            n_quantiles,
            output_distribution: output_distribution.to_string(),
            clip,
            quantiles: None,
            references: None,
        })
    }

    /// Fits the QuantileTransformer to the input data
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

        if self.n_quantiles > n_samples {
            return Err(TransformError::InvalidInput(format!(
                "n_quantiles ({}) cannot be greater than n_samples ({})",
                self.n_quantiles, n_samples
            )));
        }

        // Compute quantiles for each feature
        let mut quantiles = Array2::zeros((n_features, self.n_quantiles));

        for j in 0..n_features {
            // Extract feature data and sort it
            let mut feature_data: Vec<f64> = x_f64.column(j).to_vec();
            feature_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Compute quantiles
            for i in 0..self.n_quantiles {
                let q = i as f64 / (self.n_quantiles - 1) as f64;
                let idx = (q * (feature_data.len() - 1) as f64).round() as usize;
                quantiles[[j, i]] = feature_data[idx];
            }
        }

        // Generate reference distribution
        let references = if self.output_distribution == "uniform" {
            // Uniform distribution references
            Array1::from_shape_fn(self.n_quantiles, |i| {
                i as f64 / (self.n_quantiles - 1) as f64
            })
        } else {
            // Normal distribution references (using inverse normal CDF approximation)
            Array1::from_shape_fn(self.n_quantiles, |i| {
                let u = i as f64 / (self.n_quantiles - 1) as f64;
                // Clamp u to avoid extreme values
                let u_clamped = u.max(1e-7).min(1.0 - 1e-7);
                inverse_normal_cdf(u_clamped)
            })
        };

        self.quantiles = Some(quantiles);
        self.references = Some(references);

        Ok(())
    }

    /// Transforms the input data using the fitted QuantileTransformer
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        if self.quantiles.is_none() || self.references.is_none() {
            return Err(TransformError::TransformationError(
                "QuantileTransformer has not been fitted".to_string(),
            ));
        }

        let quantiles = self.quantiles.as_ref().unwrap();
        let references = self.references.as_ref().unwrap();

        if n_features != quantiles.shape()[0] {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but QuantileTransformer was fitted with {} features",
                n_features,
                quantiles.shape()[0]
            )));
        }

        let mut transformed = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                let value = x_f64[[i, j]];

                // Find the position of the value in the quantiles
                let feature_quantiles = quantiles.row(j);

                // Find the index where value would be inserted
                let mut lower_idx = 0;
                let mut upper_idx = self.n_quantiles - 1;

                // Handle edge cases
                if value <= feature_quantiles[0] {
                    transformed[[i, j]] = references[0];
                    continue;
                }
                if value >= feature_quantiles[self.n_quantiles - 1] {
                    transformed[[i, j]] = references[self.n_quantiles - 1];
                    continue;
                }

                // Binary search to find the interval
                while upper_idx - lower_idx > 1 {
                    let mid = (lower_idx + upper_idx) / 2;
                    if value <= feature_quantiles[mid] {
                        upper_idx = mid;
                    } else {
                        lower_idx = mid;
                    }
                }

                // Linear interpolation between reference values
                let lower_quantile = feature_quantiles[lower_idx];
                let upper_quantile = feature_quantiles[upper_idx];
                let lower_ref = references[lower_idx];
                let upper_ref = references[upper_idx];

                if (upper_quantile - lower_quantile).abs() < EPSILON {
                    transformed[[i, j]] = lower_ref;
                } else {
                    let ratio = (value - lower_quantile) / (upper_quantile - lower_quantile);
                    transformed[[i, j]] = lower_ref + ratio * (upper_ref - lower_ref);
                }
            }
        }

        // Apply clipping if requested and output distribution is uniform
        if self.clip && self.output_distribution == "uniform" {
            for i in 0..n_samples {
                for j in 0..n_features {
                    transformed[[i, j]] = transformed[[i, j]].max(0.0).min(1.0);
                }
            }
        }

        Ok(transformed)
    }

    /// Fits the QuantileTransformer to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the quantiles for each feature
    ///
    /// # Returns
    /// * `Option<&Array2<f64>>` - The quantiles, shape (n_features, n_quantiles)
    pub fn quantiles(&self) -> Option<&Array2<f64>> {
        self.quantiles.as_ref()
    }
}

/// Approximation of the inverse normal cumulative distribution function
///
/// This uses the Beasley-Springer-Moro algorithm for approximating the inverse normal CDF
fn inverse_normal_cdf(u: f64) -> f64 {
    // Constants for the Beasley-Springer-Moro algorithm
    const A0: f64 = 2.50662823884;
    const A1: f64 = -18.61500062529;
    const A2: f64 = 41.39119773534;
    const A3: f64 = -25.44106049637;
    const B1: f64 = -8.47351093090;
    const B2: f64 = 23.08336743743;
    const B3: f64 = -21.06224101826;
    const B4: f64 = 3.13082909833;
    const C0: f64 = 0.3374754822726147;
    const C1: f64 = 0.9761690190917186;
    const C2: f64 = 0.1607979714918209;
    const C3: f64 = 0.0276438810333863;
    const C4: f64 = 0.0038405729373609;
    const C5: f64 = 0.0003951896511919;
    const C6: f64 = 0.0000321767881768;
    const C7: f64 = 0.0000002888167364;
    const C8: f64 = 0.0000003960315187;

    let y = u - 0.5;

    if y.abs() < 0.42 {
        // Central region
        let r = y * y;
        y * (((A3 * r + A2) * r + A1) * r + A0) / ((((B4 * r + B3) * r + B2) * r + B1) * r + 1.0)
    } else {
        // Tail region
        let r = if y > 0.0 { 1.0 - u } else { u };
        let r = (-r.ln()).ln();

        let result = C0
            + r * (C1 + r * (C2 + r * (C3 + r * (C4 + r * (C5 + r * (C6 + r * (C7 + r * C8)))))));

        if y < 0.0 {
            -result
        } else {
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    #[test]
    fn test_quantile_transformer_uniform() {
        // Create test data with different distributions
        let data = Array::from_shape_vec(
            (6, 2),
            vec![
                1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 100.0, 1000.0,
            ], // Last row has outliers
        )
        .unwrap();

        let mut transformer = QuantileTransformer::new(5, "uniform", true).unwrap();
        let transformed = transformer.fit_transform(&data).unwrap();

        // Check that the shape is preserved
        assert_eq!(transformed.shape(), &[6, 2]);

        // For uniform distribution, values should be between 0 and 1
        for i in 0..6 {
            for j in 0..2 {
                assert!(
                    transformed[[i, j]] >= 0.0 && transformed[[i, j]] <= 1.0,
                    "Value at [{}, {}] = {} is not in [0, 1]",
                    i,
                    j,
                    transformed[[i, j]]
                );
            }
        }

        // The smallest value should map to 0 and largest to 1
        assert_abs_diff_eq!(transformed[[0, 0]], 0.0, epsilon = 1e-10); // min of column 0
        assert_abs_diff_eq!(transformed[[5, 0]], 1.0, epsilon = 1e-10); // max of column 0
        assert_abs_diff_eq!(transformed[[0, 1]], 0.0, epsilon = 1e-10); // min of column 1
        assert_abs_diff_eq!(transformed[[5, 1]], 1.0, epsilon = 1e-10); // max of column 1
    }

    #[test]
    fn test_quantile_transformer_normal() {
        // Create test data
        let data = Array::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        let mut transformer = QuantileTransformer::new(5, "normal", false).unwrap();
        let transformed = transformer.fit_transform(&data).unwrap();

        // Check that the shape is preserved
        assert_eq!(transformed.shape(), &[5, 1]);

        // The middle value should be close to 0 (median of normal distribution)
        assert_abs_diff_eq!(transformed[[2, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quantile_transformer_errors() {
        // Test invalid n_quantiles
        assert!(QuantileTransformer::new(1, "uniform", true).is_err());

        // Test invalid output_distribution
        assert!(QuantileTransformer::new(100, "invalid", true).is_err());

        // Test fitting with insufficient data
        let small_data = Array::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let mut transformer = QuantileTransformer::new(10, "uniform", true).unwrap();
        assert!(transformer.fit(&small_data).is_err());
    }

    #[test]
    fn test_inverse_normal_cdf() {
        // Test some known values
        assert_abs_diff_eq!(inverse_normal_cdf(0.5), 0.0, epsilon = 1e-6);
        assert!(inverse_normal_cdf(0.1) < 0.0); // Should be negative
        assert!(inverse_normal_cdf(0.9) > 0.0); // Should be positive
    }
}
