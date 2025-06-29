//! GPU-accelerated transformations using CUDA
//!
//! This module provides GPU-accelerated implementations of dimensionality reduction
//! and matrix operations using CUDA through the scirs2-core GPU abstractions.

use crate::error::{Result, TransformError};
use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2};
use scirs2_core::gpu::{CudaContext, GpuArray, GpuBackend};
use scirs2_core::validation::{check_positive, check_shape, check_not_empty, check_array_finite};

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
            let gpu_xt = gpu_x_centered.transpose()
                .context("Failed to transpose data matrix on GPU")?;
            let gpu_cov = gpu_xt.matmul(&gpu_x_centered)
                .context("Failed to compute covariance matrix (X^T X) on GPU")?;
            gpu_cov.scale(1.0 / (n_samples - 1) as f64)
                .context("Failed to scale covariance matrix on GPU")?
        } else {
            // Use X X^T approach when n_samples < n_features
            let gpu_xt = gpu_x_centered.transpose()
                .context("Failed to transpose data matrix on GPU")?;
            let gpu_gram = gpu_x_centered.matmul(&gpu_xt)
                .context("Failed to compute Gram matrix (X X^T) on GPU")?;
            gpu_gram.scale(1.0 / (n_samples - 1) as f64)
                .context("Failed to scale Gram matrix on GPU")?
        };

        // Compute eigendecomposition on GPU
        let (gpu_eigenvalues, gpu_eigenvectors) = gpu_cov.eigh()?;

        // Sort eigenvalues and eigenvectors in descending order with validation
        let (gpu_eigenvalues_sorted, gpu_eigenvectors_sorted) =
            gpu_eigenvalues.sort_with_vectors(&gpu_eigenvectors, false)
                .context("Failed to sort eigenvalues and eigenvectors on GPU")?;

        // Take top n_components with bounds checking
        let gpu_components = gpu_eigenvectors_sorted.slice((.., ..self.n_components))
            .context("Failed to slice eigenvectors for components on GPU")?;
        let gpu_explained_var = gpu_eigenvalues_sorted.slice(..self.n_components)
            .context("Failed to slice eigenvalues for explained variance on GPU")?;

        // Transfer results back to host with validation
        let components_host = gpu_components.to_ndarray()
            .context("Failed to transfer components from GPU to host")?;
        let explained_var_host = gpu_explained_var.to_ndarray()
            .context("Failed to transfer explained variance from GPU to host")?;
            
        // Validate components matrix
        if components_host.dim().0 != n_features || components_host.dim().1 != self.n_components {
            return Err(TransformError::ComputationError(
                "Component matrix has incorrect dimensions after GPU computation".to_string(),
            ));
        }
        
        // Validate explained variance values
        if explained_var_host.iter().any(|&x| !x.is_finite() || x < 0.0) {
            return Err(TransformError::ComputationError(
                "Invalid explained variance values computed on GPU".to_string(),
            ));
        }
        
        self.components = Some(components_host.t().to_owned());
        self.explained_variance = Some(explained_var_host);
        self.mean = mean;

        Ok(())
    }

    /// Transform data using the fitted PCA model on GPU with enhanced validation
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
            return Err(TransformError::InvalidInput(
                format!("Input has {} features, but model was trained on {} features", 
                    n_features, components.dim().1)
            ));
        }

        let cuda_ctx = self.cuda_context.as_ref().unwrap();

        // Transfer data to GPU
        let gpu_x = GpuArray::from_ndarray(x, cuda_ctx)
            .context("Failed to transfer transform data to GPU")?;

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

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Get the explained variance ratio
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
            let neg_dist_sq = row_distances.square()?.scale(-1.0 / (2.0 * sigma * sigma))?;
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
        let learning_rate = if iteration < 250 {
            500.0
        } else {
            200.0
        };
        
        // Update gains (adaptive per-dimension learning rates)
        for i in 0..n_samples * n_components {
            let grad_i = gradient.get_item(i)?;
            let mom_i = momentum.get_item(i)?;
            let gain_i = gains.get_item(i)?;
            
            let new_gain = if grad_i * mom_i < 0.0 {
                (gain_i + 0.2).min(50.0)  // Increase gain if gradient and momentum oppose
            } else {
                (gain_i * 0.8).max(0.01)  // Decrease gain if they align
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
