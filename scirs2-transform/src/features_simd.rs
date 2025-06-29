//! SIMD-accelerated feature engineering operations
//!
//! This module provides SIMD-optimized implementations of feature engineering operations
//! using the unified SIMD operations from scirs2-core.

use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Axis, Data, Ix2};
use num_traits::{Float, NumCast};
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::validation::{check_finite, check_not_empty, check_positive};

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
        // Validate input using scirs2-core validation
        check_not_empty(x, "x")?;
        check_finite(x, "x")?;

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if n_features > 1000 {
            return Err(TransformError::InvalidInput(
                "Too many features for polynomial expansion (>1000)".to_string(),
            ));
        }

        // Calculate output dimensions with overflow check
        let n_output_features = self.calculate_n_output_features(n_features)?;

        // Check for memory constraints
        if n_samples > 100_000 && n_output_features > 10_000 {
            return Err(TransformError::ComputationError(
                "Output matrix would be too large (>1B elements)".to_string(),
            ));
        }

        let mut output = Array2::zeros((n_samples, n_output_features));

        // Process samples in batches for better cache locality
        const BATCH_SIZE: usize = 256;
        for batch_start in (0..n_samples).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(n_samples);

            for i in batch_start..batch_end {
                let sample = x.row(i);
                let poly_features = self.transform_sample_simd(&sample)?;

                // Use SIMD copy if available
                if poly_features.len() == n_output_features {
                    let mut output_row = output.row_mut(i);
                    for (j, &val) in poly_features.iter().enumerate() {
                        output_row[j] = val;
                    }
                } else {
                    return Err(TransformError::ComputationError(
                        "Feature count mismatch in polynomial expansion".to_string(),
                    ));
                }
            }
        }

        Ok(output)
    }

    /// Transforms a single sample using SIMD operations
    fn transform_sample_simd(&self, sample: &ArrayView1<F>) -> Result<Array1<F>> {
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
            // Squared terms using SIMD
            let squared = F::simd_mul(&sample.view(), &sample.view());
            for j in 0..n_features {
                output[output_idx] = squared[j];
                output_idx += 1;
            }

            // Cross terms with vectorized operations where possible
            for j in 0..n_features {
                let remaining_features = n_features - j - 1;
                if remaining_features > 0 {
                    // Use SIMD for remaining cross products
                    let sample_j = sample[j];
                    let remaining_slice = sample.slice(ndarray::s![j + 1..]);

                    // Create a vector filled with sample[j]
                    let sample_j_vec = Array1::from_elem(remaining_features, sample_j);
                    let cross_products = F::simd_mul(&sample_j_vec.view(), &remaining_slice);

                    for &val in cross_products.iter() {
                        output[output_idx] = val;
                        output_idx += 1;
                    }
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
            // Pairwise interactions with SIMD optimization
            for j in 0..n_features {
                let remaining_features = n_features - j - 1;
                if remaining_features > 0 {
                    let sample_j = sample[j];
                    let remaining_slice = sample.slice(ndarray::s![j + 1..]);

                    // Use SIMD for batch processing of interactions
                    let sample_j_vec = Array1::from_elem(remaining_features, sample_j);
                    let interactions = F::simd_mul(&sample_j_vec.view(), &remaining_slice);

                    for &val in interactions.iter() {
                        output[output_idx] = val;
                        output_idx += 1;
                    }
                } else {
                    // Fallback for remaining elements
                    for k in j + 1..n_features {
                        output[output_idx] = sample[j] * sample[k];
                        output_idx += 1;
                    }
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
                        result[i] = -((-x + F::one()).powf(F::from(2.0).unwrap() - lambda)
                            - F::one())
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

    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    if !exponent.is_finite() {
        return Err(TransformError::InvalidInput(
            "Exponent must be finite".to_string(),
        ));
    }

    let mut result = Array1::zeros(n);

    // For common exponents, use optimized SIMD operations
    if (exponent - F::from(2.0).unwrap()).abs() < F::from(1e-10).unwrap() {
        // Square using SIMD multiplication
        result = F::simd_mul(&data.view(), &data.view());
    } else if (exponent - F::from(0.5).unwrap()).abs() < F::from(1e-10).unwrap() {
        // Square root using SIMD - check for non-negative values first
        for &val in data.iter() {
            if val < F::zero() {
                return Err(TransformError::ComputationError(
                    "Cannot compute square root of negative values".to_string(),
                ));
            }
        }
        result = F::simd_sqrt(&data.view());
    } else if (exponent - F::from(3.0).unwrap()).abs() < F::from(1e-10).unwrap() {
        // Cube: x^3 = x * x * x
        let squared = F::simd_mul(&data.view(), &data.view());
        result = F::simd_mul(&squared.view(), &data.view());
    } else if (exponent - F::from(1.0).unwrap()).abs() < F::from(1e-10).unwrap() {
        // Identity: x^1 = x
        result = data.clone();
    } else if (exponent - F::from(0.0).unwrap()).abs() < F::from(1e-10).unwrap() {
        // Constant: x^0 = 1
        result.fill(F::one());
    } else {
        // General case: use vectorized exponentiation
        let exponent_array = Array1::from_elem(n, exponent);
        result = F::simd_pow(&data.view(), &exponent_array.view());

        // Validate results
        for &val in result.iter() {
            if !val.is_finite() {
                return Err(TransformError::ComputationError(
                    "Power operation produced non-finite values".to_string(),
                ));
            }
        }
    }

    Ok(result)
}

/// SIMD-accelerated binarization with validation
pub fn simd_binarize<F>(data: &Array2<F>, threshold: F) -> Result<Array2<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    check_not_empty(data, "data")?;
    check_finite(data, "data")?;

    if !threshold.is_finite() {
        return Err(TransformError::InvalidInput(
            "Threshold must be finite".to_string(),
        ));
    }

    let shape = data.shape();
    let mut result = Array2::zeros((shape[0], shape[1]));

    // Process in chunks for better cache locality
    const CHUNK_SIZE: usize = 64;

    for i in 0..shape[0] {
        let row = data.row(i);
        let row_array = row.to_owned();

        // Process row in chunks using SIMD
        for chunk_start in (0..shape[1]).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(shape[1]);
            let chunk_size = chunk_end - chunk_start;

            let chunk_slice = row_array.slice(ndarray::s![chunk_start..chunk_end]);
            let threshold_array = Array1::from_elem(chunk_size, threshold);

            // Use SIMD comparison where available
            let comparison_result = F::simd_greater_than(&chunk_slice, &threshold_array.view());

            for (j, &cmp_result) in comparison_result.iter().enumerate() {
                result[[i, chunk_start + j]] = if cmp_result > F::zero() {
                    F::one()
                } else {
                    F::zero()
                };
            }
        }
    }

    Ok(result)
}
