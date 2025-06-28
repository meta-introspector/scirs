//! GPU-accelerated transformations using CUDA
//!
//! This module provides GPU-accelerated implementations of dimensionality reduction
//! and matrix operations using CUDA through the scirs2-core GPU abstractions.

use crate::error::{Result, TransformError};
use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2};
use scirs2_core::gpu::{CudaContext, GpuArray, GpuBackend};
use scirs2_core::validation::{check_positive, check_shape};

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

        // Compute covariance matrix on GPU
        let gpu_cov = if n_samples > n_features {
            // Use X^T X approach when n_features < n_samples
            let gpu_xt = gpu_x_centered.transpose()?;
            let gpu_cov = gpu_xt.matmul(&gpu_x_centered)?;
            gpu_cov.scale(1.0 / (n_samples - 1) as f64)?
        } else {
            // Use X X^T approach when n_samples < n_features
            let gpu_gram = gpu_x_centered.matmul(&gpu_x_centered.transpose()?)?;
            gpu_gram.scale(1.0 / (n_samples - 1) as f64)?
        };

        // Compute eigendecomposition on GPU
        let (gpu_eigenvalues, gpu_eigenvectors) = gpu_cov.eigh()?;

        // Sort eigenvalues and eigenvectors in descending order
        let (gpu_eigenvalues_sorted, gpu_eigenvectors_sorted) = 
            gpu_eigenvalues.sort_with_vectors(&gpu_eigenvectors, false)?;

        // Take top n_components
        let gpu_components = gpu_eigenvectors_sorted.slice((.., ..self.n_components))?;
        let gpu_explained_var = gpu_eigenvalues_sorted.slice(..self.n_components)?;

        // Transfer results back to host
        self.components = Some(gpu_components.to_ndarray()?.t().to_owned());
        self.explained_variance = Some(gpu_explained_var.to_ndarray()?);
        self.mean = mean;

        Ok(())
    }

    /// Transform data using the fitted PCA model on GPU
    pub fn transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let components = self.components.as_ref()
            .ok_or_else(|| TransformError::NotFitted("PCA model not fitted".to_string()))?;

        let cuda_ctx = self.cuda_context.as_ref().unwrap();

        // Transfer data to GPU
        let gpu_x = GpuArray::from_ndarray(x, cuda_ctx)?;

        // Center data if needed
        let gpu_x_processed = if let Some(ref mean) = self.mean {
            let gpu_mean = GpuArray::from_ndarray(&mean.view().insert_axis(ndarray::Axis(0)), cuda_ctx)?;
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
        let explained_var = self.explained_variance.as_ref()
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
        
        Ok((
            gpu_eigenvals.to_ndarray()?,
            gpu_eigenvecs.to_ndarray()?,
        ))
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
        let mut gpu_y = GpuArray::random_normal(
            (n_samples, self.n_components),
            &self.cuda_context,
        )?;

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
        // Compute high-dimensional similarities using Gaussian kernel
        let sigma = (2.0 * self.perplexity).sqrt();
        let gpu_p = gpu_distances.scale(-1.0 / (2.0 * sigma * sigma))?.exp()?;
        let gpu_p_sum = gpu_p.sum_axis(1)?;
        gpu_p.divide_broadcast(&gpu_p_sum)
    }

    fn compute_q_matrix(&self, gpu_y: &GpuArray) -> Result<GpuArray> {
        // Compute low-dimensional similarities using t-distribution
        let gpu_distances = self.compute_pairwise_distances(gpu_y)?;
        let gpu_q = (gpu_distances.square()? + 1.0)?.pow(-1.0)?;
        let gpu_q_sum = gpu_q.sum()?;
        gpu_q.scale(1.0)?.divide_scalar(gpu_q_sum)
    }

    fn compute_gradient(&self, gpu_p: &GpuArray, gpu_q: &GpuArray, gpu_y: &GpuArray) -> Result<GpuArray> {
        // Compute gradient of KL divergence
        let gpu_pq_diff = gpu_p.subtract(gpu_q)?;
        let gpu_y_diff = gpu_y.unsqueeze(1)?.subtract(&gpu_y.unsqueeze(0)?)?;
        let gpu_distances = self.compute_pairwise_distances(gpu_y)?;
        let gpu_weights = (gpu_distances.square()? + 1.0)?.pow(-1.0)?;
        
        gpu_pq_diff.unsqueeze(-1)?
            .multiply(&gpu_y_diff)?
            .multiply(&gpu_weights.unsqueeze(-1)?)?
            .sum_axis(1)?
            .scale(4.0)
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