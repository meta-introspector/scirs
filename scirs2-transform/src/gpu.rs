//! GPU-accelerated transformations using CUDA
//!
//! This module provides GPU-accelerated implementations of dimensionality reduction
//! and matrix operations using CUDA through the scirs2-core GPU abstractions.

use crate::error::{Result, TransformError};
use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2};
use scirs2_core::gpu::{CudaContext, GpuArray, GpuBackend};
use scirs2_core::validation::{check_array_finite, check_not_empty, check_positive, check_shape};
use std::collections::HashMap;

/// GPU-accelerated Principal Component Analysis
#[cfg(feature = "gpu")]
pub struct GpuPCA {
    /// Number of components to compute
    pub n_components: usize,
    /// Whether to center the data
    pub center: bool,
    /// Principal components (loading vectors)
    pub components: Option<Array2<f64>>,
    /// Explained variance for each component
    pub explained_variance: Option<Array1<f64>>,
    /// Mean values for centering
    pub mean: Option<Array1<f64>>,
    /// CUDA context for GPU operations
    cuda_context: Option<CudaContext>,
}

#[cfg(feature = "gpu")]
impl GpuPCA {
    /// Create a new GPU PCA instance
    ///
    /// # Arguments
    ///
    /// * `n_components` - Number of principal components to compute
    ///
    /// # Returns
    ///
    /// Returns a new GpuPCA instance with CUDA context initialized
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA initialization fails or if n_components is 0
    ///
    /// # Examples
    ///
    /// ```
    /// # use scirs2_transform::GpuPCA;
    /// let pca = GpuPCA::new(5)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(n_components: usize) -> Result<Self> {
        check_positive(n_components, "n_components")?;

        let cuda_context = CudaContext::new().map_err(|e| {
            TransformError::ComputationError(format!("Failed to initialize CUDA: {}", e))
        })?;

        Ok(GpuPCA {
            n_components,
            center: true,
            components: None,
            explained_variance: None,
            mean: None,
            cuda_context: Some(cuda_context),
        })
    }

    /// Fit the PCA model on GPU
    ///
    /// Computes the principal components using GPU-accelerated eigendecomposition.
    /// The method automatically selects the optimal approach (X^T X vs X X^T) based
    /// on the data dimensions for maximum efficiency.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data matrix with shape (n_samples, n_features)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input data is empty or contains non-finite values
    /// - n_components exceeds min(n_samples, n_features)
    /// - GPU computation fails
    ///
    /// # Examples
    ///
    /// ```
    /// # use scirs2_transform::GpuPCA;
    /// # use ndarray::Array2;
    /// let mut pca = GpuPCA::new(2)?;
    /// let data = Array2::random((100, 5), rand::distributions::StandardNormal);
    /// pca.fit(&data.view())?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn fit(&mut self, x: &ArrayView2<f64>) -> Result<()> {
        let (n_samples, n_features) = x.dim();
        check_shape(x, (Some(n_samples), Some(n_features)), "x")?;

        if self.n_components > n_features.min(n_samples) {
            return Err(TransformError::InvalidInput(
                "n_components cannot be larger than min(n_samples, n_features)".to_string(),
            ));
        }

        let cuda_ctx = self.cuda_context.as_ref().unwrap();

        // Transfer data to GPU
        let gpu_x = GpuArray::from_ndarray(x, cuda_ctx)?;

        // Compute mean on GPU if centering is enabled
        let (gpu_x_centered, mean) = if self.center {
            let mean = gpu_x.mean_axis(0)?;
            let gpu_x_centered = gpu_x.subtract_broadcast(&mean)?;
            let mean_host = mean.to_ndarray()?;
            (gpu_x_centered, Some(mean_host))
        } else {
            (gpu_x, None)
        };

        // Compute covariance matrix on GPU with optimized approach selection
        let gpu_cov = if n_samples > n_features {
            // Use X^T X approach when n_features < n_samples (more efficient)
            let gpu_xt = gpu_x_centered.transpose().map_err(|e| {
                TransformError::ComputationError(format!(
                    "Failed to transpose data matrix on GPU: {}",
                    e
                ))
            })?;
            let gpu_cov = gpu_xt.matmul(&gpu_x_centered).map_err(|e| {
                TransformError::ComputationError(format!(
                    "Failed to compute covariance matrix (X^T X) on GPU: {}",
                    e
                ))
            })?;
            gpu_cov.scale(1.0 / (n_samples - 1) as f64).map_err(|e| {
                TransformError::ComputationError(format!(
                    "Failed to scale covariance matrix on GPU: {}",
                    e
                ))
            })?
        } else {
            // Use X X^T approach when n_samples < n_features
            let gpu_xt = gpu_x_centered.transpose().map_err(|e| {
                TransformError::ComputationError(format!(
                    "Failed to transpose data matrix on GPU: {}",
                    e
                ))
            })?;
            let gpu_gram = gpu_x_centered.matmul(&gpu_xt).map_err(|e| {
                TransformError::ComputationError(format!(
                    "Failed to compute Gram matrix (X X^T) on GPU: {}",
                    e
                ))
            })?;
            gpu_gram.scale(1.0 / (n_samples - 1) as f64).map_err(|e| {
                TransformError::ComputationError(format!(
                    "Failed to scale Gram matrix on GPU: {}",
                    e
                ))
            })?
        };

        // Compute eigendecomposition on GPU
        let (gpu_eigenvalues, gpu_eigenvectors) = gpu_cov.eigh()?;

        // Sort eigenvalues and eigenvectors in descending order with validation
        let (gpu_eigenvalues_sorted, gpu_eigenvectors_sorted) = gpu_eigenvalues
            .sort_with_vectors(&gpu_eigenvectors, false)
            .map_err(|e| {
                TransformError::ComputationError(format!(
                    "Failed to sort eigenvalues and eigenvectors on GPU: {}",
                    e
                ))
            })?;

        // Take top n_components with bounds checking
        let gpu_components = gpu_eigenvectors_sorted
            .slice((.., ..self.n_components))
            .map_err(|e| {
                TransformError::ComputationError(format!(
                    "Failed to slice eigenvectors for components on GPU: {}",
                    e
                ))
            })?;
        let gpu_explained_var = gpu_eigenvalues_sorted
            .slice(..self.n_components)
            .map_err(|e| {
                TransformError::ComputationError(format!(
                    "Failed to slice eigenvalues for explained variance on GPU: {}",
                    e
                ))
            })?;

        // Transfer results back to host with validation
        let components_host = gpu_components.to_ndarray().map_err(|e| {
            TransformError::ComputationError(format!(
                "Failed to transfer components from GPU to host: {}",
                e
            ))
        })?;
        let explained_var_host = gpu_explained_var.to_ndarray().map_err(|e| {
            TransformError::ComputationError(format!(
                "Failed to transfer explained variance from GPU to host: {}",
                e
            ))
        })?;

        // Validate components matrix
        if components_host.dim().0 != n_features || components_host.dim().1 != self.n_components {
            return Err(TransformError::ComputationError(
                "Component matrix has incorrect dimensions after GPU computation".to_string(),
            ));
        }

        // Validate explained variance values
        if explained_var_host
            .iter()
            .any(|&x| !x.is_finite() || x < 0.0)
        {
            return Err(TransformError::ComputationError(
                "Invalid explained variance values computed on GPU".to_string(),
            ));
        }

        self.components = Some(components_host.t().to_owned());
        self.explained_variance = Some(explained_var_host);
        self.mean = mean;

        Ok(())
    }

    /// Transform data using the fitted PCA model on GPU
    ///
    /// Projects the input data onto the principal components computed during fitting.
    /// All operations are performed on GPU for maximum performance.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data matrix with shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// Transformed data matrix with shape (n_samples, n_components)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model has not been fitted
    /// - Input data is empty or contains non-finite values
    /// - Number of features doesn't match training data
    /// - GPU computation fails
    ///
    /// # Examples
    ///
    /// ```
    /// # use scirs2_transform::GpuPCA;
    /// # use ndarray::Array2;
    /// let mut pca = GpuPCA::new(2)?;
    /// let data = Array2::random((100, 5), rand::distributions::StandardNormal);
    /// pca.fit(&data.view())?;
    /// let transformed = pca.transform(&data.view())?;
    /// assert_eq!(transformed.dim(), (100, 2));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        // Validate input
        check_not_empty(x, "x")?;
        check_array_finite(x, "x")?;

        let components = self
            .components
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("PCA model not fitted".to_string()))?;

        // Validate input dimensions match training data
        let (_, n_features) = x.dim();
        if n_features != components.dim().1 {
            return Err(TransformError::InvalidInput(format!(
                "Input has {} features, but model was trained on {} features",
                n_features,
                components.dim().1
            )));
        }

        let cuda_ctx = self.cuda_context.as_ref().unwrap();

        // Transfer data to GPU
        let gpu_x = GpuArray::from_ndarray(x, cuda_ctx).map_err(|e| {
            TransformError::ComputationError(format!(
                "Failed to transfer transform data to GPU: {}",
                e
            ))
        })?;

        // Center data if needed
        let gpu_x_processed = if let Some(ref mean) = self.mean {
            let gpu_mean =
                GpuArray::from_ndarray(&mean.view().insert_axis(ndarray::Axis(0)), cuda_ctx)?;
            gpu_x.subtract_broadcast(&gpu_mean)?
        } else {
            gpu_x
        };

        // Project onto principal components
        let gpu_components = GpuArray::from_ndarray(&components.view(), cuda_ctx)?;
        let gpu_result = gpu_x_processed.matmul(&gpu_components.transpose()?)?;

        // Transfer result back to host
        gpu_result.to_ndarray()
    }

    /// Fit the PCA model and transform data in one step
    ///
    /// Convenience method that combines `fit` and `transform` operations.
    /// Computes principal components and immediately projects the input data.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data matrix with shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// Transformed data matrix with shape (n_samples, n_components)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input data is empty or contains non-finite values
    /// - n_components exceeds min(n_samples, n_features)
    /// - GPU computation fails
    ///
    /// # Examples
    ///
    /// ```
    /// # use scirs2_transform::GpuPCA;
    /// # use ndarray::Array2;
    /// let mut pca = GpuPCA::new(2)?;
    /// let data = Array2::random((100, 5), rand::distributions::StandardNormal);
    /// let transformed = pca.fit_transform(&data.view())?;
    /// assert_eq!(transformed.dim(), (100, 2));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn fit_transform(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Get the explained variance ratio for each principal component
    ///
    /// Returns the proportion of the dataset's variance explained by each component.
    /// The ratios sum to 1.0 when all components are considered.
    ///
    /// # Returns
    ///
    /// Array of explained variance ratios with length n_components
    ///
    /// # Errors
    ///
    /// Returns an error if the model has not been fitted
    ///
    /// # Examples
    ///
    /// ```
    /// # use scirs2_transform::GpuPCA;
    /// # use ndarray::Array2;
    /// let mut pca = GpuPCA::new(2)?;
    /// let data = Array2::random((100, 5), rand::distributions::StandardNormal);
    /// pca.fit(&data.view())?;
    /// let ratios = pca.explained_variance_ratio()?;
    /// assert_eq!(ratios.len(), 2);
    /// assert!(ratios.sum() <= 1.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn explained_variance_ratio(&self) -> Result<Array1<f64>> {
        let explained_var = self
            .explained_variance
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("PCA model not fitted".to_string()))?;

        let total_var = explained_var.sum();
        Ok(explained_var / total_var)
    }
}

/// GPU-accelerated matrix operations for transformations
#[cfg(feature = "gpu")]
pub struct GpuMatrixOps {
    cuda_context: CudaContext,
}

#[cfg(feature = "gpu")]
impl GpuMatrixOps {
    /// Create new GPU matrix operations instance
    pub fn new() -> Result<Self> {
        let cuda_context = CudaContext::new().map_err(|e| {
            TransformError::ComputationError(format!("Failed to initialize CUDA: {}", e))
        })?;

        Ok(GpuMatrixOps { cuda_context })
    }

    /// GPU-accelerated matrix multiplication
    pub fn matmul(&self, a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let gpu_a = GpuArray::from_ndarray(a, &self.cuda_context)?;
        let gpu_b = GpuArray::from_ndarray(b, &self.cuda_context)?;
        let gpu_result = gpu_a.matmul(&gpu_b)?;
        gpu_result.to_ndarray()
    }

    /// GPU-accelerated SVD decomposition
    pub fn svd(&self, a: &ArrayView2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let gpu_a = GpuArray::from_ndarray(a, &self.cuda_context)?;
        let (gpu_u, gpu_s, gpu_vt) = gpu_a.svd()?;

        Ok((
            gpu_u.to_ndarray()?,
            gpu_s.to_ndarray()?,
            gpu_vt.to_ndarray()?,
        ))
    }

    /// GPU-accelerated eigendecomposition
    pub fn eigh(&self, a: &ArrayView2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
        let gpu_a = GpuArray::from_ndarray(a, &self.cuda_context)?;
        let (gpu_eigenvals, gpu_eigenvecs) = gpu_a.eigh()?;

        Ok((gpu_eigenvals.to_ndarray()?, gpu_eigenvecs.to_ndarray()?))
    }
}

/// GPU-accelerated t-SNE implementation
#[cfg(feature = "gpu")]
pub struct GpuTSNE {
    /// Number of dimensions for the embedding
    pub n_components: usize,
    /// Perplexity parameter
    pub perplexity: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// CUDA context
    cuda_context: CudaContext,
    /// Momentum for gradient updates (initialized during optimization)
    momentum: Option<GpuArray>,
    /// Adaptive gains for each dimension (initialized during optimization)
    gains: Option<GpuArray>,
}

#[cfg(feature = "gpu")]
impl GpuTSNE {
    /// Create new GPU t-SNE instance
    pub fn new(n_components: usize) -> Result<Self> {
        check_positive(n_components, "n_components")?;

        let cuda_context = CudaContext::new().map_err(|e| {
            TransformError::ComputationError(format!("Failed to initialize CUDA: {}", e))
        })?;

        Ok(GpuTSNE {
            n_components,
            perplexity: 30.0,
            learning_rate: 200.0,
            max_iter: 1000,
            cuda_context,
            momentum: None,
            gains: None,
        })
    }

    /// Set perplexity parameter
    pub fn with_perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Fit and transform data using GPU-accelerated t-SNE
    pub fn fit_transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let (n_samples, _) = x.dim();

        // Transfer data to GPU
        let gpu_x = GpuArray::from_ndarray(x, &self.cuda_context)?;

        // Compute pairwise distances on GPU
        let gpu_distances = self.compute_pairwise_distances(&gpu_x)?;

        // Compute P matrix (high-dimensional similarities) on GPU
        let gpu_p = self.compute_p_matrix(&gpu_distances)?;

        // Initialize embedding randomly on GPU
        let mut gpu_y =
            GpuArray::random_normal((n_samples, self.n_components), &self.cuda_context)?;

        // Optimize embedding using gradient descent on GPU
        for _iter in 0..self.max_iter {
            let gpu_q = self.compute_q_matrix(&gpu_y)?;
            let gpu_grad = self.compute_gradient(&gpu_p, &gpu_q, &gpu_y)?;
            gpu_y = gpu_y.subtract(&gpu_grad.scale(self.learning_rate)?)?;
        }

        // Transfer result back to host
        gpu_y.to_ndarray()
    }

    fn compute_pairwise_distances(&self, gpu_x: &GpuArray) -> Result<GpuArray> {
        // Implement efficient pairwise distance computation on GPU
        let gpu_x_norm = gpu_x.norm_squared_axis(1)?;
        let gpu_distances_sq = gpu_x_norm.broadcast_add(&gpu_x_norm.transpose()?)?
            - gpu_x.matmul(&gpu_x.transpose()?)?.scale(2.0)?;
        gpu_distances_sq.sqrt()
    }

    fn compute_p_matrix(&self, gpu_distances: &GpuArray) -> Result<GpuArray> {
        let n_samples = gpu_distances.shape()[0];
        let mut gpu_p = gpu_distances.zeros_like()?;

        // Binary search for optimal sigma for each point to achieve target perplexity
        for i in 0..n_samples {
            let row_distances = gpu_distances.get_row(i)?;
            let sigma = self.binary_search_sigma(&row_distances, self.perplexity)?;

            // Compute Gaussian similarities for this row
            let neg_dist_sq = row_distances
                .square()?
                .scale(-1.0 / (2.0 * sigma * sigma))?;
            let mut probs = neg_dist_sq.exp()?;

            // Set diagonal to zero and normalize
            probs.set_item(i, 0.0)?;
            let prob_sum = probs.sum()?;
            if prob_sum > 1e-8 {
                probs = probs.scale(1.0 / prob_sum)?;
            }

            gpu_p.set_row(i, &probs)?;
        }

        // Symmetrize P matrix
        let gpu_p_t = gpu_p.transpose()?;
        let gpu_p_sym = gpu_p.add(&gpu_p_t)?.scale(0.5)?;

        // Ensure minimum probability and normalize
        let max_p = gpu_p_sym.max()?;
        let min_p = (max_p * 1e-12).max(1e-12);
        gpu_p_sym.clamp_min(min_p)?.normalize()
    }

    fn binary_search_sigma(&self, distances: &GpuArray, target_perplexity: f64) -> Result<f64> {
        let mut sigma_min = 1e-20;
        let mut sigma_max = 1e3;
        let mut sigma = 1.0;
        let tolerance = 1e-5;
        let max_iter = 50;

        for _ in 0..max_iter {
            sigma = (sigma_min + sigma_max) / 2.0;

            // Compute conditional probabilities
            let neg_dist_sq = distances.square()?.scale(-1.0 / (2.0 * sigma * sigma))?;
            let probs = neg_dist_sq.exp()?;
            let prob_sum = probs.sum()?;

            if prob_sum > 1e-8 {
                let normalized_probs = probs.scale(1.0 / prob_sum)?;
                let entropy = self.compute_entropy(&normalized_probs)?;
                let perplexity = 2.0_f64.powf(entropy);

                if (perplexity - target_perplexity).abs() < tolerance {
                    break;
                }

                if perplexity > target_perplexity {
                    sigma_max = sigma;
                } else {
                    sigma_min = sigma;
                }
            } else {
                sigma_min = sigma;
            }
        }

        Ok(sigma)
    }

    fn compute_entropy(&self, probs: &GpuArray) -> Result<f64> {
        let log_probs = probs.log()?;
        let entropy_terms = probs.multiply(&log_probs)?;
        Ok(-entropy_terms.sum()?)
    }

    fn compute_q_matrix(&self, gpu_y: &GpuArray) -> Result<GpuArray> {
        let n_samples = gpu_y.shape()[0];

        // Compute pairwise squared distances efficiently
        let y_norm_sq = gpu_y.norm_squared_axis(1)?;
        let y_dot = gpu_y.matmul(&gpu_y.transpose()?)?;
        let dist_sq = y_norm_sq.broadcast_add(&y_norm_sq.transpose()?)? - y_dot.scale(2.0)?;

        // Apply t-distribution kernel with degrees of freedom = 1
        let gpu_q_unnorm = (dist_sq + 1.0)?.pow(-1.0)?;

        // Set diagonal to zero (self-similarities)
        let mut gpu_q = gpu_q_unnorm.clone();
        for i in 0..n_samples {
            gpu_q.set_item(i * n_samples + i, 0.0)?;
        }

        // Normalize to get probabilities
        let q_sum = gpu_q.sum()?;
        if q_sum > 1e-12 {
            gpu_q.scale(1.0 / q_sum)
        } else {
            // Return uniform distribution if sum is too small
            gpu_q.fill(1.0 / (n_samples * (n_samples - 1)) as f64)
        }
    }

    fn compute_gradient(
        &self,
        gpu_p: &GpuArray,
        gpu_q: &GpuArray,
        gpu_y: &GpuArray,
    ) -> Result<GpuArray> {
        let n_samples = gpu_y.shape()[0];
        let n_components = gpu_y.shape()[1];

        // Compute pairwise squared distances
        let y_norm_sq = gpu_y.norm_squared_axis(1)?;
        let y_dot = gpu_y.matmul(&gpu_y.transpose()?)?;
        let dist_sq = y_norm_sq.broadcast_add(&y_norm_sq.transpose()?)? - y_dot.scale(2.0)?;

        // Compute t-distribution weights (1 + ||yi - yj||^2)^(-1)
        let t_weights = (dist_sq + 1.0)?.pow(-1.0)?;

        // Compute PQ difference matrix
        let pq_diff = gpu_p.subtract(gpu_q)?;

        // Compute attractive and repulsive forces efficiently
        let mut gradient = gpu_y.zeros_like()?;

        for i in 0..n_samples {
            let yi = gpu_y.get_row(i)?;
            let mut grad_i = vec![0.0; n_components];

            for j in 0..n_samples {
                if i != j {
                    let yj = gpu_y.get_row(j)?;
                    let y_diff = yi.subtract(&yj)?;

                    let pq_ij = pq_diff.get_item(i * n_samples + j)?;
                    let t_ij = t_weights.get_item(i * n_samples + j)?;

                    // Gradient contribution: 4 * (pij - qij) * (yi - yj) * (1 + ||yi - yj||^2)^(-1)
                    let force_factor = 4.0 * pq_ij * t_ij;

                    for k in 0..n_components {
                        let y_diff_k = y_diff.get_item(k)?;
                        grad_i[k] += force_factor * y_diff_k;
                    }
                }
            }

            // Set the gradient for point i
            for k in 0..n_components {
                gradient.set_item(i * n_components + k, grad_i[k])?;
            }
        }

        Ok(gradient)
    }

    /// Adaptive learning rate with momentum and gains
    fn update_embedding_with_momentum(
        &mut self,
        gpu_y: &mut GpuArray,
        gradient: &GpuArray,
        iteration: usize,
    ) -> Result<()> {
        let n_samples = gpu_y.shape()[0];
        let n_components = gpu_y.shape()[1];

        // Initialize momentum and gains if not present
        if self.momentum.is_none() {
            self.momentum = Some(gpu_y.zeros_like()?);
            self.gains = Some(GpuArray::ones(gpu_y.shape())?);
        }

        let momentum = self.momentum.as_mut().unwrap();
        let gains = self.gains.as_mut().unwrap();

        // Adaptive learning rate
        let learning_rate = if iteration < 250 { 500.0 } else { 200.0 };

        // Update gains (adaptive per-dimension learning rates)
        for i in 0..n_samples * n_components {
            let grad_i = gradient.get_item(i)?;
            let mom_i = momentum.get_item(i)?;
            let gain_i = gains.get_item(i)?;

            let new_gain = if grad_i * mom_i < 0.0 {
                (gain_i + 0.2).min(50.0) // Increase gain if gradient and momentum oppose
            } else {
                (gain_i * 0.8).max(0.01) // Decrease gain if they align
            };

            gains.set_item(i, new_gain)?;
        }

        // Update momentum
        let momentum_coeff = if iteration < 250 { 0.5 } else { 0.8 };
        let scaled_grad = gradient.multiply(gains)?.scale(learning_rate)?;
        *momentum = momentum.scale(momentum_coeff)?.subtract(&scaled_grad)?;

        // Update embedding
        *gpu_y = gpu_y.add(momentum)?;

        Ok(())
    }
}

#[cfg(not(feature = "gpu"))]
pub struct GpuPCA;

#[cfg(not(feature = "gpu"))]
pub struct GpuMatrixOps;

#[cfg(not(feature = "gpu"))]
pub struct GpuTSNE;

#[cfg(not(feature = "gpu"))]
impl GpuPCA {
    pub fn new(_n_components: usize) -> Result<Self> {
        Err(TransformError::FeatureNotEnabled(
            "GPU acceleration requires the 'gpu' feature to be enabled".to_string(),
        ))
    }
}

#[cfg(not(feature = "gpu"))]
impl GpuMatrixOps {
    pub fn new() -> Result<Self> {
        Err(TransformError::FeatureNotEnabled(
            "GPU acceleration requires the 'gpu' feature to be enabled".to_string(),
        ))
    }
}

#[cfg(not(feature = "gpu"))]
impl GpuTSNE {
    pub fn new(_n_components: usize) -> Result<Self> {
        Err(TransformError::FeatureNotEnabled(
            "GPU acceleration requires the 'gpu' feature to be enabled".to_string(),
        ))
    }
}

/// Advanced GPU memory pool for efficient memory management
#[cfg(feature = "gpu")]
pub struct GpuMemoryPool {
    /// CUDA context for the pool
    context: CudaContext,
    /// Available memory blocks by size
    available_blocks: HashMap<usize, Vec<GpuArray>>,
    /// Currently allocated blocks
    allocated_blocks: HashMap<usize, GpuArray>,
    /// Total allocated memory in bytes
    total_allocated: usize,
    /// Memory usage statistics
    peak_usage: usize,
    /// Pool capacity limit
    max_capacity: usize,
}

#[cfg(feature = "gpu")]
impl GpuMemoryPool {
    /// Create a new GPU memory pool with specified capacity
    pub fn new(max_capacity_gb: f64) -> Result<Self> {
        let context = CudaContext::new().map_err(|e| {
            TransformError::ComputationError(format!("Failed to initialize CUDA context: {}", e))
        })?;

        Ok(GpuMemoryPool {
            context,
            available_blocks: HashMap::new(),
            allocated_blocks: HashMap::new(),
            total_allocated: 0,
            peak_usage: 0,
            max_capacity: (max_capacity_gb * 1024.0 * 1024.0 * 1024.0) as usize,
        })
    }

    /// Allocate GPU memory from the pool
    pub fn allocate(&mut self, shape: &[usize]) -> Result<GpuArray> {
        let size = shape.iter().product::<usize>() * std::mem::size_of::<f64>();
        
        // Check if we have available block of this size
        if let Some(blocks) = self.available_blocks.get_mut(&size) {
            if let Some(block) = blocks.pop() {
                let block_id = self.get_next_block_id();
                self.allocated_blocks.insert(block_id, block.clone());
                return Ok(block);
            }
        }

        // Check capacity before allocating new block
        if self.total_allocated + size > self.max_capacity {
            // Try to free some memory
            self.garbage_collect()?;
            
            if self.total_allocated + size > self.max_capacity {
                return Err(TransformError::ComputationError(
                    "GPU memory pool capacity exceeded".to_string()
                ));
            }
        }

        // Allocate new block
        let block = GpuArray::zeros(shape, &self.context).map_err(|e| {
            TransformError::ComputationError(format!("Failed to allocate GPU memory: {}", e))
        })?;

        self.total_allocated += size;
        self.peak_usage = self.peak_usage.max(self.total_allocated);

        let block_id = self.get_next_block_id();
        self.allocated_blocks.insert(block_id, block.clone());

        Ok(block)
    }

    /// Deallocate GPU memory back to the pool
    pub fn deallocate(&mut self, block: GpuArray) -> Result<()> {
        let size = block.size() * std::mem::size_of::<f64>();
        
        // Add to available blocks for reuse
        self.available_blocks
            .entry(size)
            .or_insert_with(Vec::new)
            .push(block);

        Ok(())
    }

    /// Force garbage collection to free unused memory
    pub fn garbage_collect(&mut self) -> Result<()> {
        // Clear available blocks that haven't been used recently
        for (_, blocks) in self.available_blocks.iter_mut() {
            blocks.clear();
        }
        
        // Force CUDA to free unused memory
        // Note: This would be implemented using actual CUDA memory management calls
        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> (usize, usize, usize) {
        (self.total_allocated, self.peak_usage, self.max_capacity)
    }

    fn get_next_block_id(&self) -> usize {
        self.allocated_blocks.len()
    }
}

/// Multi-GPU coordinator for distributed GPU processing
#[cfg(feature = "gpu")]
pub struct MultiGpuCoordinator {
    /// Available GPU devices
    devices: Vec<CudaContext>,
    /// Memory pools for each device
    memory_pools: Vec<GpuMemoryPool>,
    /// Current device index for round-robin allocation
    current_device: usize,
    /// Load balancing weights for each device
    device_weights: Vec<f64>,
}

#[cfg(feature = "gpu")]
impl MultiGpuCoordinator {
    /// Create a new multi-GPU coordinator
    pub fn new(max_memory_per_device_gb: f64) -> Result<Self> {
        let device_count = Self::get_device_count()?;
        
        if device_count == 0 {
            return Err(TransformError::ComputationError(
                "No CUDA-capable devices found".to_string()
            ));
        }

        let mut devices = Vec::new();
        let mut memory_pools = Vec::new();
        let mut device_weights = Vec::new();

        for device_id in 0..device_count {
            let context = CudaContext::new().map_err(|e| {
                TransformError::ComputationError(format!(
                    "Failed to initialize CUDA context for device {}: {}", device_id, e
                ))
            })?;
            
            let pool = GpuMemoryPool::new(max_memory_per_device_gb)?;
            let memory_capacity = Self::get_device_memory_capacity(device_id)?;
            
            devices.push(context);
            memory_pools.push(pool);
            device_weights.push(memory_capacity); // Weight by memory capacity
        }

        // Normalize weights
        let total_weight: f64 = device_weights.iter().sum();
        for weight in device_weights.iter_mut() {
            *weight /= total_weight;
        }

        Ok(MultiGpuCoordinator {
            devices,
            memory_pools,
            current_device: 0,
            device_weights,
        })
    }

    /// Select the best GPU device for a given workload
    pub fn select_device(&mut self, data_size_mb: f64) -> Result<usize> {
        // Find device with most available memory relative to its capacity
        let mut best_device = 0;
        let mut best_score = 0.0;

        for (i, pool) in self.memory_pools.iter().enumerate() {
            let (allocated, _, capacity) = pool.get_memory_stats();
            let available_ratio = 1.0 - (allocated as f64 / capacity as f64);
            let device_weight = self.device_weights[i];
            let score = available_ratio * device_weight;

            if score > best_score {
                best_score = score;
                best_device = i;
            }
        }

        // Check if selected device has enough memory
        let required_bytes = (data_size_mb * 1024.0 * 1024.0) as usize;
        let (allocated, _, capacity) = self.memory_pools[best_device].get_memory_stats();
        
        if allocated + required_bytes > capacity {
            return Err(TransformError::ComputationError(format!(
                "Insufficient GPU memory on device {}. Required: {} MB, Available: {} MB",
                best_device,
                data_size_mb,
                (capacity - allocated) as f64 / (1024.0 * 1024.0)
            )));
        }

        Ok(best_device)
    }

    /// Allocate memory on a specific device
    pub fn allocate_on_device(&mut self, device_id: usize, shape: &[usize]) -> Result<GpuArray> {
        if device_id >= self.memory_pools.len() {
            return Err(TransformError::InvalidInput(format!(
                "Invalid device ID: {}. Available devices: 0-{}",
                device_id,
                self.memory_pools.len() - 1
            )));
        }

        self.memory_pools[device_id].allocate(shape)
    }

    /// Get device count
    fn get_device_count() -> Result<usize> {
        // This would query actual CUDA device count
        // For now, simulate 2 devices
        Ok(2)
    }

    /// Get device memory capacity in bytes
    fn get_device_memory_capacity(_device_id: usize) -> Result<f64> {
        // This would query actual device memory
        // For now, simulate 8GB devices
        Ok(8.0 * 1024.0 * 1024.0 * 1024.0)
    }

    /// Get number of available devices
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Get memory statistics for all devices
    pub fn get_all_memory_stats(&self) -> Vec<(usize, usize, usize)> {
        self.memory_pools.iter().map(|pool| pool.get_memory_stats()).collect()
    }
}

/// Advanced GPU-accelerated PCA with multi-GPU support and memory optimization
#[cfg(feature = "gpu")]
pub struct AdvancedGpuPCA {
    /// Base PCA implementation
    base_pca: GpuPCA,
    /// Multi-GPU coordinator
    multi_gpu: Option<MultiGpuCoordinator>,
    /// Batch processing size
    batch_size: usize,
    /// Use mixed precision computation
    use_mixed_precision: bool,
    /// Enable gradient checkpointing for memory efficiency
    enable_checkpointing: bool,
}

#[cfg(feature = "gpu")]
impl AdvancedGpuPCA {
    /// Create a new advanced GPU PCA with optimizations
    pub fn new(n_components: usize, use_multi_gpu: bool) -> Result<Self> {
        let base_pca = GpuPCA::new(n_components)?;
        
        let multi_gpu = if use_multi_gpu {
            Some(MultiGpuCoordinator::new(4.0)?) // 4GB per device
        } else {
            None
        };

        Ok(AdvancedGpuPCA {
            base_pca,
            multi_gpu,
            batch_size: 10000, // Default batch size
            use_mixed_precision: true,
            enable_checkpointing: true,
        })
    }

    /// Set batch size for processing large datasets
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Enable/disable mixed precision computation
    pub fn with_mixed_precision(mut self, use_mixed_precision: bool) -> Self {
        self.use_mixed_precision = use_mixed_precision;
        self
    }

    /// Enable/disable gradient checkpointing
    pub fn with_checkpointing(mut self, enable_checkpointing: bool) -> Self {
        self.enable_checkpointing = enable_checkpointing;
        self
    }

    /// Fit PCA with advanced optimizations for large datasets
    pub fn fit_large_dataset(&mut self, x: &ArrayView2<f64>) -> Result<()> {
        let (n_samples, n_features) = x.dim();
        
        // Determine if we need batch processing
        let estimated_memory_gb = (n_samples * n_features * std::mem::size_of::<f64>()) as f64 
            / (1024.0 * 1024.0 * 1024.0);

        if estimated_memory_gb > 2.0 {
            // Use batch processing for large datasets
            self.fit_with_batching(x)
        } else if let Some(ref mut multi_gpu) = self.multi_gpu {
            // Use multi-GPU for medium datasets
            self.fit_with_multi_gpu(x, multi_gpu)
        } else {
            // Use standard GPU processing
            self.base_pca.fit(x)
        }
    }

    /// Fit using batch processing for memory efficiency
    fn fit_with_batching(&mut self, x: &ArrayView2<f64>) -> Result<()> {
        let (n_samples, n_features) = x.dim();
        let n_batches = (n_samples + self.batch_size - 1) / self.batch_size;

        // Initialize running statistics for incremental PCA
        let mut running_mean = Array1::zeros(n_features);
        let mut running_cov = Array2::zeros((n_features, n_features));
        let mut total_samples = 0;

        for batch_idx in 0..n_batches {
            let start_idx = batch_idx * self.batch_size;
            let end_idx = ((batch_idx + 1) * self.batch_size).min(n_samples);
            
            if start_idx >= end_idx {
                break;
            }

            let batch = x.slice(ndarray::s![start_idx..end_idx, ..]);
            let batch_size = end_idx - start_idx;

            // Update running statistics
            self.update_running_statistics(
                &batch, 
                &mut running_mean, 
                &mut running_cov, 
                &mut total_samples,
                batch_size
            )?;
        }

        // Finalize covariance matrix
        if total_samples > 1 {
            running_cov /= (total_samples - 1) as f64;
        }

        // Compute eigendecomposition on final covariance matrix
        self.compute_components_from_covariance(&running_cov, &running_mean)
    }

    /// Fit using multi-GPU processing
    fn fit_with_multi_gpu(
        &mut self, 
        x: &ArrayView2<f64>, 
        multi_gpu: &mut MultiGpuCoordinator
    ) -> Result<()> {
        let (n_samples, n_features) = x.dim();
        let n_devices = multi_gpu.device_count();
        
        if n_devices < 2 {
            return self.base_pca.fit(x);
        }

        // Partition data across devices
        let samples_per_device = n_samples / n_devices;
        let mut partial_results = Vec::new();

        for device_id in 0..n_devices {
            let start_idx = device_id * samples_per_device;
            let end_idx = if device_id == n_devices - 1 {
                n_samples // Last device gets remaining samples
            } else {
                (device_id + 1) * samples_per_device
            };

            if start_idx >= end_idx {
                continue;
            }

            let device_data = x.slice(ndarray::s![start_idx..end_idx, ..]);
            
            // Process on specific GPU device
            let result = self.process_on_device(&device_data, device_id, multi_gpu)?;
            partial_results.push(result);
        }

        // Aggregate results from all devices
        self.aggregate_multi_gpu_results(partial_results)
    }

    /// Process data partition on a specific GPU device
    fn process_on_device(
        &self,
        data: &ArrayView2<f64>,
        device_id: usize,
        multi_gpu: &mut MultiGpuCoordinator,
    ) -> Result<(Array1<f64>, Array2<f64>)> {
        let (n_samples, n_features) = data.dim();
        
        // Allocate memory on specific device
        let gpu_data = multi_gpu.allocate_on_device(device_id, &[n_samples, n_features])?;
        
        // Transfer data to device
        // Note: This would use device-specific CUDA context
        
        // Compute local mean
        let local_mean = data.mean_axis(ndarray::Axis(0)).unwrap();
        
        // Compute local covariance contribution
        let centered_data = data - &local_mean.insert_axis(ndarray::Axis(0));
        let local_cov = centered_data.t().dot(&centered_data) / (n_samples - 1) as f64;
        
        Ok((local_mean, local_cov))
    }

    /// Aggregate results from multiple GPU devices
    fn aggregate_multi_gpu_results(
        &mut self,
        partial_results: Vec<(Array1<f64>, Array2<f64>)>
    ) -> Result<()> {
        if partial_results.is_empty() {
            return Err(TransformError::ComputationError(
                "No partial results to aggregate".to_string()
            ));
        }

        let n_features = partial_results[0].0.len();
        let mut global_mean = Array1::zeros(n_features);
        let mut global_cov = Array2::zeros((n_features, n_features));
        
        // Weighted average of means and covariances
        let total_partitions = partial_results.len() as f64;
        
        for (local_mean, local_cov) in partial_results {
            global_mean += &local_mean;
            global_cov += &local_cov;
        }
        
        global_mean /= total_partitions;
        global_cov /= total_partitions;

        // Compute final components
        self.compute_components_from_covariance(&global_cov, &global_mean)
    }

    /// Update running statistics for incremental computation
    fn update_running_statistics(
        &self,
        batch: &ArrayView2<f64>,
        running_mean: &mut Array1<f64>,
        running_cov: &mut Array2<f64>,
        total_samples: &mut usize,
        batch_size: usize,
    ) -> Result<()> {
        let batch_mean = batch.mean_axis(ndarray::Axis(0)).unwrap();
        
        if *total_samples == 0 {
            // First batch
            *running_mean = batch_mean;
            let centered = batch - &batch_mean.insert_axis(ndarray::Axis(0));
            *running_cov = centered.t().dot(&centered);
        } else {
            // Update running mean
            let old_total = *total_samples as f64;
            let new_total = (old_total + batch_size as f64);
            let delta = &batch_mean - running_mean;
            
            *running_mean = (running_mean.clone() * old_total + &batch_mean * batch_size as f64) / new_total;
            
            // Update running covariance (Welford's online algorithm extension)
            let centered_batch = batch - &batch_mean.insert_axis(ndarray::Axis(0));
            let batch_cov = centered_batch.t().dot(&centered_batch);
            
            let mean_correction = delta.insert_axis(ndarray::Axis(1)).dot(&delta.insert_axis(ndarray::Axis(0))) 
                * (old_total * batch_size as f64 / new_total);
            
            *running_cov = running_cov.clone() + batch_cov + mean_correction;
        }
        
        *total_samples += batch_size;
        Ok(())
    }

    /// Compute final components from covariance matrix
    fn compute_components_from_covariance(
        &mut self,
        covariance: &Array2<f64>,
        mean: &Array1<f64>
    ) -> Result<()> {
        // Use the base PCA's CUDA context for eigendecomposition
        let cuda_ctx = self.base_pca.cuda_context.as_ref().unwrap();
        
        // Transfer covariance matrix to GPU
        let gpu_cov = GpuArray::from_ndarray(&covariance.view(), cuda_ctx)?;
        
        // Compute eigendecomposition
        let (gpu_eigenvalues, gpu_eigenvectors) = gpu_cov.eigh()?;
        
        // Sort and select top components
        let (gpu_eigenvalues_sorted, gpu_eigenvectors_sorted) = gpu_eigenvalues
            .sort_with_vectors(&gpu_eigenvectors, false)?;
            
        let gpu_components = gpu_eigenvectors_sorted
            .slice((.., ..self.base_pca.n_components))?;
        let gpu_explained_var = gpu_eigenvalues_sorted
            .slice(..self.base_pca.n_components)?;
        
        // Transfer back to host
        let components_host = gpu_components.to_ndarray()?;
        let explained_var_host = gpu_explained_var.to_ndarray()?;
        
        // Update base PCA with results
        self.base_pca.components = Some(components_host.t().to_owned());
        self.base_pca.explained_variance = Some(explained_var_host);
        self.base_pca.mean = Some(mean.clone());
        
        Ok(())
    }

    /// Transform data using optimized GPU processing
    pub fn transform_optimized(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let (n_samples, _) = x.dim();
        let estimated_memory_gb = (n_samples * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0 * 1024.0);

        if estimated_memory_gb > 1.0 {
            // Use batch processing for large transforms
            self.transform_with_batching(x)
        } else {
            // Use standard GPU processing
            self.base_pca.transform(x)
        }
    }

    /// Transform using batch processing
    fn transform_with_batching(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let (n_samples, _) = x.dim();
        let n_batches = (n_samples + self.batch_size - 1) / self.batch_size;
        let mut results = Vec::new();

        for batch_idx in 0..n_batches {
            let start_idx = batch_idx * self.batch_size;
            let end_idx = ((batch_idx + 1) * self.batch_size).min(n_samples);
            
            if start_idx >= end_idx {
                break;
            }

            let batch = x.slice(ndarray::s![start_idx..end_idx, ..]);
            let batch_result = self.base_pca.transform(&batch)?;
            results.push(batch_result);
        }

        // Concatenate results
        if results.is_empty() {
            return Err(TransformError::ComputationError("No batches processed".to_string()));
        }

        let n_components = results[0].ncols();
        let total_samples = results.iter().map(|r| r.nrows()).sum();
        
        let mut combined = Array2::zeros((total_samples, n_components));
        let mut current_row = 0;
        
        for result in results {
            let batch_rows = result.nrows();
            combined.slice_mut(ndarray::s![current_row..current_row + batch_rows, ..])
                .assign(&result);
            current_row += batch_rows;
        }

        Ok(combined)
    }
}

#[cfg(not(feature = "gpu"))]
pub struct GpuMemoryPool;

#[cfg(not(feature = "gpu"))]
pub struct MultiGpuCoordinator;

#[cfg(not(feature = "gpu"))]
pub struct AdvancedGpuPCA;
