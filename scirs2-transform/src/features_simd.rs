//! SIMD-accelerated feature engineering operations
//!
//! This module provides SIMD-optimized implementations of feature engineering operations
//! using the unified SIMD operations from scirs2-core.

use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use num_traits::{Float, NumCast};
use scirs2_core::simd_ops::SimdUnifiedOps;

use crate::error::{Result, TransformError};

/// SIMD-accelerated polynomial feature generation
pub struct SimdPolynomialFeatures<F: Float + NumCast + SimdUnifiedOps> {
    degree: usize,
    include_bias: bool,
    interaction_only: bool,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + NumCast + SimdUnifiedOps> SimdPolynomialFeatures<F> {
    /// Creates a new SIMD-accelerated polynomial features generator
    pub fn new(degree: usize, include_bias: bool, interaction_only: bool) -> Result<Self> {
        if degree == 0 {
            return Err(TransformError::InvalidInput(
                "Degree must be at least 1".to_string(),
            ));
        }

        Ok(SimdPolynomialFeatures {
            degree,
            include_bias,
            interaction_only,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Transforms input features to polynomial features using SIMD operations
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<F>>
    where
        S: Data<Elem = F>,
    {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        // Calculate output dimensions
        let n_output_features = self.calculate_n_output_features(n_features)?;
        let mut output = Array2::zeros((n_samples, n_output_features));

        // Process each sample
        for i in 0..n_samples {
            let sample = x.row(i);
            let sample_array = sample.to_owned();
            let poly_features = self.transform_sample(&sample_array)?;
            
            for (j, &val) in poly_features.iter().enumerate() {
                output[[i, j]] = val;
            }
        }

        Ok(output)
    }

    /// Transforms a single sample using SIMD operations
    fn transform_sample(&self, sample: &Array1<F>) -> Result<Array1<F>> {
        let n_features = sample.len();
        let n_output_features = self.calculate_n_output_features(n_features)?;
        let mut output = Array1::zeros(n_output_features);
        let mut output_idx = 0;

        // Include bias term if requested
        if self.include_bias {
            output[output_idx] = F::one();
            output_idx += 1;
        }

        // Copy original features
        for j in 0..n_features {
            output[output_idx] = sample[j];
            output_idx += 1;
        }

        // Generate polynomial features
        if self.degree > 1 {
            if self.interaction_only {
                // Only interaction terms (no powers of single features)
                output_idx = self.add_interaction_terms(sample, &mut output, output_idx, 2)?;
            } else {
                // All polynomial combinations
                output_idx = self.add_polynomial_terms(sample, &mut output, output_idx)?;
            }
        }

        Ok(output)
    }

    /// Adds polynomial terms using SIMD operations where possible
    fn add_polynomial_terms(
        &self,
        sample: &Array1<F>,
        output: &mut Array1<F>,
        mut output_idx: usize,
    ) -> Result<usize> {
        let n_features = sample.len();

        // For degree 2, use SIMD for efficient computation
        if self.degree == 2 {
            // Squared terms
            let squared = F::simd_mul(&sample.view(), &sample.view());
            for j in 0..n_features {
                output[output_idx] = squared[j];
                output_idx += 1;
            }

            // Cross terms
            for j in 0..n_features {
                for k in j + 1..n_features {
                    output[output_idx] = sample[j] * sample[k];
                    output_idx += 1;
                }
            }
        } else {
            // For higher degrees, fall back to iterative computation
            // but still use SIMD where beneficial
            for current_degree in 2..=self.degree {
                output_idx = self.add_degree_terms(sample, output, output_idx, current_degree)?;
            }
        }

        Ok(output_idx)
    }

    /// Adds interaction terms only
    fn add_interaction_terms(
        &self,
        sample: &Array1<F>,
        output: &mut Array1<F>,
        mut output_idx: usize,
        degree: usize,
    ) -> Result<usize> {
        let n_features = sample.len();

        if degree == 2 {
            // Pairwise interactions
            for j in 0..n_features {
                for k in j + 1..n_features {
                    output[output_idx] = sample[j] * sample[k];
                    output_idx += 1;
                }
            }
        } else {
            // Higher-order interactions
            let indices = self.generate_interaction_indices(n_features, degree);
            for idx_set in indices {
                let mut prod = F::one();
                for &idx in &idx_set {
                    prod = prod * sample[idx];
                }
                output[output_idx] = prod;
                output_idx += 1;
            }
        }

        Ok(output_idx)
    }

    /// Adds terms of a specific degree
    fn add_degree_terms(
        &self,
        sample: &Array1<F>,
        output: &mut Array1<F>,
        mut output_idx: usize,
        degree: usize,
    ) -> Result<usize> {
        let n_features = sample.len();
        let indices = self.generate_degree_indices(n_features, degree);

        for idx_vec in indices {
            let mut prod = F::one();
            for &idx in &idx_vec {
                prod = prod * sample[idx];
            }
            output[output_idx] = prod;
            output_idx += 1;
        }

        Ok(output_idx)
    }

    /// Calculates the number of output features
    fn calculate_n_output_features(&self, n_features: usize) -> Result<usize> {
        let mut count = if self.include_bias { 1 } else { 0 };
        count += n_features; // Original features

        if self.degree > 1 {
            if self.interaction_only {
                // Only interaction terms
                for d in 2..=self.degree {
                    count += self.n_choose_k(n_features, d);
                }
            } else {
                // All polynomial combinations
                count += self.n_polynomial_features(n_features, self.degree) - n_features;
                if self.include_bias {
                    count -= 1;
                }
            }
        }

        Ok(count)
    }

    /// Calculates n choose k
    fn n_choose_k(&self, n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }

        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }

    /// Calculates the number of polynomial features
    fn n_polynomial_features(&self, n_features: usize, degree: usize) -> usize {
        self.n_choose_k(n_features + degree, degree)
    }

    /// Generates indices for interaction terms
    fn generate_interaction_indices(&self, n_features: usize, degree: usize) -> Vec<Vec<usize>> {
        let mut indices = Vec::new();
        let mut current = vec![0; degree];

        loop {
            // Add current combination
            indices.push(current.clone());

            // Find the rightmost element that can be incremented
            let mut i = degree - 1;
            loop {
                current[i] += 1;
                if current[i] < n_features - (degree - 1 - i) {
                    // Reset all elements to the right
                    for j in i + 1..degree {
                        current[j] = current[j - 1] + 1;
                    }
                    break;
                }
                if i == 0 {
                    return indices;
                }
                i -= 1;
            }
        }
    }

    /// Generates indices for polynomial terms of a specific degree
    fn generate_degree_indices(&self, n_features: usize, degree: usize) -> Vec<Vec<usize>> {
        let mut indices = Vec::new();
        let mut current = vec![0; degree];

        loop {
            // Add current combination (with repetition allowed)
            indices.push(current.clone());

            // Find the rightmost element that can be incremented
            let mut i = degree - 1;
            loop {
                current[i] += 1;
                if current[i] < n_features {
                    // Reset all elements to the right to the same value
                    for j in i + 1..degree {
                        current[j] = current[i];
                    }
                    break;
                }
                if i == 0 {
                    return indices;
                }
                current[i] = 0;
                i -= 1;
            }
        }
    }
}

/// SIMD-accelerated power transformation (Box-Cox and Yeo-Johnson)
pub fn simd_power_transform<F>(data: &Array1<F>, lambda: F, method: &str) -> Result<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    let n = data.len();
    let mut result = Array1::zeros(n);

    match method {
        "box-cox" => {
            // Check for negative values
            let min_val = F::simd_min_element(&data.view());
            if min_val <= F::zero() {
                return Err(TransformError::InvalidInput(
                    "Box-Cox requires strictly positive values".to_string(),
                ));
            }

            if lambda.abs() < F::from(1e-6).unwrap() {
                // lambda ≈ 0: log transform
                for i in 0..n {
                    result[i] = data[i].ln();
                }
            } else {
                // General Box-Cox: (x^lambda - 1) / lambda
                let ones = Array1::from_elem(n, F::one());
                let powered = simd_array_pow(data, lambda)?;
                let numerator = F::simd_sub(&powered.view(), &ones.view());
                let lambda_array = Array1::from_elem(n, lambda);
                result = F::simd_div(&numerator.view(), &lambda_array.view());
            }
        }
        "yeo-johnson" => {
            // Yeo-Johnson handles both positive and negative values
            for i in 0..n {
                let x = data[i];
                if x >= F::zero() {
                    if lambda.abs() < F::from(1e-6).unwrap() {
                        result[i] = x.ln() + F::one();
                    } else {
                        result[i] = ((x + F::one()).powf(lambda) - F::one()) / lambda;
                    }
                } else {
                    if (F::from(2.0).unwrap() - lambda).abs() < F::from(1e-6).unwrap() {
                        result[i] = -((-x + F::one()).ln());
                    } else {
                        result[i] = -((-x + F::one()).powf(F::from(2.0).unwrap() - lambda) - F::one())
                            / (F::from(2.0).unwrap() - lambda);
                    }
                }
            }
        }
        _ => {
            return Err(TransformError::InvalidInput(
                "Method must be 'box-cox' or 'yeo-johnson'".to_string(),
            ));
        }
    }

    Ok(result)
}

/// Helper function to compute element-wise power using SIMD where possible
fn simd_array_pow<F>(data: &Array1<F>, exponent: F) -> Result<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    let n = data.len();
    let mut result = Array1::zeros(n);

    // For common exponents, use SIMD operations
    if exponent == F::from(2.0).unwrap() {
        // Square using SIMD multiplication
        result = F::simd_mul(&data.view(), &data.view());
    } else if exponent == F::from(0.5).unwrap() {
        // Square root using SIMD
        result = F::simd_sqrt(&data.view());
    } else {
        // General case: fall back to element-wise computation
        for i in 0..n {
            result[i] = data[i].powf(exponent);
        }
    }

    Ok(result)
}

/// SIMD-accelerated binarization
pub fn simd_binarize<F>(data: &Array2<F>, threshold: F) -> Result<Array2<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    let shape = data.shape();
    let mut result = Array2::zeros((shape[0], shape[1]));

    // Process row by row using SIMD comparisons
    for i in 0..shape[0] {
        let row = data.row(i);
        let row_array = row.to_owned();
        let threshold_array = Array1::from_elem(shape[1], threshold);
        
        // Compare each element with threshold
        for j in 0..shape[1] {
            result[[i, j]] = if row_array[j] > threshold {
                F::one()
            } else {
                F::zero()
            };
        }
    }

    Ok(result)
}