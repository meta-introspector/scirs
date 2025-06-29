//! Enhanced visualization capabilities for clustering results
//!
//! This module provides comprehensive visualization tools for clustering algorithms,
//! including scatter plots, 3D visualizations, dimensionality reduction plots,
//! and interactive exploration tools for high-dimensional data.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{ClusteringError, Result};

/// Configuration for clustering visualizations
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Color scheme for clusters
    pub color_scheme: ColorScheme,
    /// Point size for scatter plots
    pub point_size: f32,
    /// Point opacity (0.0 to 1.0)
    pub point_opacity: f32,
    /// Show cluster centroids
    pub show_centroids: bool,
    /// Show cluster boundaries (convex hull or ellipse)
    pub show_boundaries: bool,
    /// Boundary type
    pub boundary_type: BoundaryType,
    /// Enable interactive features
    pub interactive: bool,
    /// Animation settings
    pub animation: Option<AnimationConfig>,
    /// Dimensionality reduction method for high-dimensional data
    pub dimensionality_reduction: DimensionalityReduction,
}

/// Color schemes for cluster visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorScheme {
    /// Default bright colors
    Default,
    /// Colorblind-friendly palette
    ColorblindFriendly,
    /// High contrast colors
    HighContrast,
    /// Pastel colors
    Pastel,
    /// Viridis colormap
    Viridis,
    /// Plasma colormap
    Plasma,
    /// Custom colors (user-defined)
    Custom,
}

/// Cluster boundary visualization types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryType {
    /// Convex hull around points
    ConvexHull,
    /// Ellipse based on covariance
    Ellipse,
    /// Alpha shapes for non-convex boundaries
    AlphaShape,
    /// No boundaries
    None,
}

/// Dimensionality reduction methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimensionalityReduction {
    /// Principal Component Analysis
    PCA,
    /// t-Distributed Stochastic Neighbor Embedding
    TSNE,
    /// Uniform Manifold Approximation and Projection
    UMAP,
    /// Multidimensional Scaling
    MDS,
    /// Use first two dimensions
    First2D,
    /// Use first three dimensions
    First3D,
    /// No reduction (error if >3D)
    None,
}

/// Animation configuration for visualizations
#[derive(Debug, Clone)]
pub struct AnimationConfig {
    /// Animation duration in milliseconds
    pub duration_ms: u32,
    /// Number of animation frames
    pub frames: u32,
    /// Easing function
    pub easing: EasingFunction,
    /// Whether to loop the animation
    pub loop_animation: bool,
}

/// Easing functions for animations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EasingFunction {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Bounce,
    Elastic,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            color_scheme: ColorScheme::Default,
            point_size: 5.0,
            point_opacity: 0.8,
            show_centroids: true,
            show_boundaries: false,
            boundary_type: BoundaryType::ConvexHull,
            interactive: true,
            animation: None,
            dimensionality_reduction: DimensionalityReduction::PCA,
        }
    }
}

/// 2D scatter plot visualization data
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScatterPlot2D {
    /// Point coordinates
    pub points: Array2<f64>,
    /// Cluster labels for each point
    pub labels: Array1<i32>,
    /// Cluster centroids (if available)
    pub centroids: Option<Array2<f64>>,
    /// Point colors (hex format)
    pub colors: Vec<String>,
    /// Point sizes
    pub sizes: Vec<f32>,
    /// Point labels (optional)
    pub point_labels: Option<Vec<String>>,
    /// Plot boundaries (min_x, max_x, min_y, max_y)
    pub bounds: (f64, f64, f64, f64),
    /// Legend information
    pub legend: Vec<LegendEntry>,
}

/// 3D scatter plot visualization data
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScatterPlot3D {
    /// Point coordinates (x, y, z)
    pub points: Array2<f64>,
    /// Cluster labels for each point
    pub labels: Array1<i32>,
    /// Cluster centroids (if available)
    pub centroids: Option<Array2<f64>>,
    /// Point colors (hex format)
    pub colors: Vec<String>,
    /// Point sizes
    pub sizes: Vec<f32>,
    /// Point labels (optional)
    pub point_labels: Option<Vec<String>>,
    /// Plot boundaries (min_x, max_x, min_y, max_y, min_z, max_z)
    pub bounds: (f64, f64, f64, f64, f64, f64),
    /// Legend information
    pub legend: Vec<LegendEntry>,
}

/// Legend entry for visualizations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LegendEntry {
    /// Cluster ID
    pub cluster_id: i32,
    /// Color hex code
    pub color: String,
    /// Cluster label/name
    pub label: String,
    /// Number of points in cluster
    pub count: usize,
}

/// Cluster boundary representation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClusterBoundary {
    /// Cluster ID
    pub cluster_id: i32,
    /// Boundary points
    pub boundary_points: Array2<f64>,
    /// Boundary type
    pub boundary_type: String,
    /// Color for the boundary
    pub color: String,
}

/// Create 2D scatter plot visualization
///
/// # Arguments
///
/// * `data` - Input data matrix (samples x features)
/// * `labels` - Cluster labels for each sample
/// * `centroids` - Optional cluster centroids
/// * `config` - Visualization configuration
///
/// # Returns
///
/// * `Result<ScatterPlot2D>` - 2D scatter plot data
pub fn create_scatter_plot_2d<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    labels: &Array1<i32>,
    centroids: Option<&Array2<F>>,
    config: &VisualizationConfig,
) -> Result<ScatterPlot2D> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if labels.len() != n_samples {
        return Err(ClusteringError::InvalidInput(
            "Number of labels must match number of samples".to_string(),
        ));
    }

    // Reduce dimensionality if needed
    let plot_data = if n_features == 2 && config.dimensionality_reduction == DimensionalityReduction::None {
        data.mapv(|x| x.to_f64().unwrap_or(0.0))
    } else {
        apply_dimensionality_reduction_2d(data, config.dimensionality_reduction)?
    };

    // Convert centroids if provided
    let plot_centroids = if let Some(cents) = centroids {
        if cents.ncols() == 2 && config.dimensionality_reduction == DimensionalityReduction::None {
            Some(cents.mapv(|x| x.to_f64().unwrap_or(0.0)))
        } else {
            Some(apply_dimensionality_reduction_2d(cents.view(), config.dimensionality_reduction)?)
        }
    } else {
        None
    };

    // Generate colors for clusters
    let unique_labels: Vec<i32> = {
        let mut labels_vec: Vec<i32> = labels.iter().cloned().collect();
        labels_vec.sort_unstable();
        labels_vec.dedup();
        labels_vec
    };
    
    let cluster_colors = generate_cluster_colors(&unique_labels, config.color_scheme);
    let point_colors = labels.iter().map(|&label| {
        cluster_colors.get(&label).cloned().unwrap_or_else(|| "#000000".to_string())
    }).collect();

    // Generate point sizes
    let sizes = vec![config.point_size; n_samples];

    // Calculate plot bounds
    let bounds = calculate_2d_bounds(&plot_data);

    // Create legend
    let legend = create_legend(&unique_labels, &cluster_colors, labels);

    Ok(ScatterPlot2D {
        points: plot_data,
        labels: labels.clone(),
        centroids: plot_centroids,
        colors: point_colors,
        sizes,
        point_labels: None,
        bounds,
        legend,
    })
}

/// Create 3D scatter plot visualization
///
/// # Arguments
///
/// * `data` - Input data matrix (samples x features)
/// * `labels` - Cluster labels for each sample
/// * `centroids` - Optional cluster centroids
/// * `config` - Visualization configuration
///
/// # Returns
///
/// * `Result<ScatterPlot3D>` - 3D scatter plot data
pub fn create_scatter_plot_3d<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    labels: &Array1<i32>,
    centroids: Option<&Array2<F>>,
    config: &VisualizationConfig,
) -> Result<ScatterPlot3D> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if labels.len() != n_samples {
        return Err(ClusteringError::InvalidInput(
            "Number of labels must match number of samples".to_string(),
        ));
    }

    // Reduce dimensionality if needed
    let plot_data = if n_features == 3 && config.dimensionality_reduction == DimensionalityReduction::None {
        data.mapv(|x| x.to_f64().unwrap_or(0.0))
    } else {
        apply_dimensionality_reduction_3d(data, config.dimensionality_reduction)?
    };

    // Convert centroids if provided
    let plot_centroids = if let Some(cents) = centroids {
        if cents.ncols() == 3 && config.dimensionality_reduction == DimensionalityReduction::None {
            Some(cents.mapv(|x| x.to_f64().unwrap_or(0.0)))
        } else {
            Some(apply_dimensionality_reduction_3d(cents.view(), config.dimensionality_reduction)?)
        }
    } else {
        None
    };

    // Generate colors for clusters
    let unique_labels: Vec<i32> = {
        let mut labels_vec: Vec<i32> = labels.iter().cloned().collect();
        labels_vec.sort_unstable();
        labels_vec.dedup();
        labels_vec
    };
    
    let cluster_colors = generate_cluster_colors(&unique_labels, config.color_scheme);
    let point_colors = labels.iter().map(|&label| {
        cluster_colors.get(&label).cloned().unwrap_or_else(|| "#000000".to_string())
    }).collect();

    // Generate point sizes
    let sizes = vec![config.point_size; n_samples];

    // Calculate plot bounds
    let bounds = calculate_3d_bounds(&plot_data);

    // Create legend
    let legend = create_legend(&unique_labels, &cluster_colors, labels);

    Ok(ScatterPlot3D {
        points: plot_data,
        labels: labels.clone(),
        centroids: plot_centroids,
        colors: point_colors,
        sizes,
        point_labels: None,
        bounds,
        legend,
    })
}

/// Apply dimensionality reduction for 2D visualization
fn apply_dimensionality_reduction_2d<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    method: DimensionalityReduction,
) -> Result<Array2<f64>> {
    let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));
    
    match method {
        DimensionalityReduction::PCA => apply_pca_2d(&data_f64),
        DimensionalityReduction::First2D => {
            if data_f64.ncols() >= 2 {
                Ok(data_f64.slice(s![.., 0..2]).to_owned())
            } else {
                Err(ClusteringError::InvalidInput(
                    "Data must have at least 2 dimensions for First2D".to_string(),
                ))
            }
        }
        DimensionalityReduction::TSNE => apply_tsne_2d(&data_f64),
        DimensionalityReduction::UMAP => apply_umap_2d(&data_f64),
        DimensionalityReduction::MDS => apply_mds_2d(&data_f64),
        DimensionalityReduction::None => {
            if data_f64.ncols() == 2 {
                Ok(data_f64)
            } else {
                Err(ClusteringError::InvalidInput(
                    "Data must be 2D when no dimensionality reduction is specified".to_string(),
                ))
            }
        }
        _ => apply_pca_2d(&data_f64), // Default to PCA
    }
}

/// Apply dimensionality reduction for 3D visualization
fn apply_dimensionality_reduction_3d<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    method: DimensionalityReduction,
) -> Result<Array2<f64>> {
    let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));
    
    match method {
        DimensionalityReduction::PCA => apply_pca_3d(&data_f64),
        DimensionalityReduction::First3D => {
            if data_f64.ncols() >= 3 {
                Ok(data_f64.slice(s![.., 0..3]).to_owned())
            } else {
                Err(ClusteringError::InvalidInput(
                    "Data must have at least 3 dimensions for First3D".to_string(),
                ))
            }
        }
        DimensionalityReduction::None => {
            if data_f64.ncols() == 3 {
                Ok(data_f64)
            } else {
                Err(ClusteringError::InvalidInput(
                    "Data must be 3D when no dimensionality reduction is specified".to_string(),
                ))
            }
        }
        _ => apply_pca_3d(&data_f64), // Default to PCA
    }
}

/// Apply PCA for 2D visualization
fn apply_pca_2d(data: &Array2<f64>) -> Result<Array2<f64>> {
    let n_samples = data.nrows();
    let n_features = data.ncols();
    
    if n_features < 2 {
        return Err(ClusteringError::InvalidInput(
            "Need at least 2 features for PCA".to_string(),
        ));
    }

    // Center the data
    let mean = data.mean_axis(Axis(0)).unwrap();
    let centered = data - &mean;

    // Compute covariance matrix
    let cov = centered.t().dot(&centered) / (n_samples - 1) as f64;

    // Simplified eigenvalue/eigenvector computation (using power iteration for largest eigenvalues)
    let (eigenvectors, _) = compute_top_eigenvectors(&cov, 2)?;

    // Project data onto first 2 principal components
    let projected = centered.dot(&eigenvectors);
    
    Ok(projected)
}

/// Apply PCA for 3D visualization
fn apply_pca_3d(data: &Array2<f64>) -> Result<Array2<f64>> {
    let n_samples = data.nrows();
    let n_features = data.ncols();
    
    if n_features < 3 {
        return Err(ClusteringError::InvalidInput(
            "Need at least 3 features for 3D PCA".to_string(),
        ));
    }

    // Center the data
    let mean = data.mean_axis(Axis(0)).unwrap();
    let centered = data - &mean;

    // Compute covariance matrix
    let cov = centered.t().dot(&centered) / (n_samples - 1) as f64;

    // Compute top 3 eigenvectors
    let (eigenvectors, _) = compute_top_eigenvectors(&cov, 3)?;

    // Project data onto first 3 principal components
    let projected = centered.dot(&eigenvectors);
    
    Ok(projected)
}

/// t-SNE implementation for 2D visualization
fn apply_tsne_2d(data: &Array2<f64>) -> Result<Array2<f64>> {
    let n_samples = data.nrows();
    let n_features = data.ncols();
    
    if n_samples < 4 {
        // Fall back to PCA for very small datasets
        return apply_pca_2d(data);
    }

    // t-SNE parameters
    let perplexity = (n_samples as f64 / 4.0).min(30.0).max(5.0);
    let learning_rate = 200.0;
    let n_iter = 1000;
    let early_exaggeration = 12.0;
    let early_exaggeration_iter = 250;

    // Step 1: Compute pairwise squared distances in high-dimensional space
    let distances_sq = compute_pairwise_distances_squared(data);
    
    // Step 2: Compute conditional probabilities P_j|i
    let p_conditional = compute_conditional_probabilities(&distances_sq, perplexity)?;
    
    // Step 3: Compute joint probabilities P_ij = (P_j|i + P_i|j) / (2n)
    let p_joint = compute_joint_probabilities(&p_conditional);
    
    // Step 4: Initialize low-dimensional embedding randomly
    let mut y = Array2::random((n_samples, 2), rand_distr::StandardNormal);
    y *= 1e-4; // Small random initialization
    
    // Step 5: Gradient descent optimization
    let mut momentum = Array2::zeros((n_samples, 2));
    let momentum_factor = 0.5;
    let final_momentum = 0.8;
    
    for iter in 0..n_iter {
        // Compute Q matrix (probabilities in low-dimensional space)
        let q_matrix = compute_low_dim_probabilities(&y);
        
        // Apply early exaggeration
        let current_p = if iter < early_exaggeration_iter {
            &p_joint * early_exaggeration
        } else {
            p_joint.clone()
        };
        
        // Compute gradient
        let gradient = compute_tsne_gradient(&current_p, &q_matrix, &y);
        
        // Update momentum
        let current_momentum = if iter < early_exaggeration_iter {
            momentum_factor
        } else {
            final_momentum
        };
        
        momentum = momentum * current_momentum - &gradient * learning_rate;
        y = y + &momentum;
        
        // Center the embedding
        let mean = y.mean_axis(Axis(0)).unwrap();
        y = y - &mean;
        
        // Early stopping based on gradient norm
        if iter > 50 && iter % 50 == 0 {
            let grad_norm = gradient.mapv(|x| x * x).sum().sqrt();
            if grad_norm < 1e-7 {
                break;
            }
        }
    }
    
    Ok(y)
}

/// UMAP implementation for 2D visualization 
fn apply_umap_2d(data: &Array2<f64>) -> Result<Array2<f64>> {
    let n_samples = data.nrows();
    let n_features = data.ncols();
    
    if n_samples < 4 {
        return apply_pca_2d(data);
    }

    // UMAP parameters
    let n_neighbors = (n_samples as f64).sqrt().floor() as usize + 1;
    let n_neighbors = n_neighbors.min(n_samples - 1).max(2);
    let min_dist = 0.1;
    let spread = 1.0;
    let n_epochs = 200;
    let learning_rate = 1.0;

    // Step 1: Find k-nearest neighbors
    let distances = compute_pairwise_distances(data);
    let neighbors = find_k_nearest_neighbors(&distances, n_neighbors);
    
    // Step 2: Compute local connectivity (σ values)
    let sigmas = compute_local_connectivity(&distances, &neighbors);
    
    // Step 3: Compute high-dimensional graph weights
    let weights = compute_umap_weights(&distances, &neighbors, &sigmas);
    
    // Step 4: Initialize low-dimensional embedding
    let mut embedding = if n_features >= 2 {
        // Initialize with first two principal components
        apply_pca_2d(data)?
    } else {
        Array2::random((n_samples, 2), rand_distr::Uniform::new(-10.0, 10.0))
    };
    
    // Normalize initial embedding
    let embedding_std = embedding.std_axis(Axis(0), 0.0);
    for j in 0..2 {
        if embedding_std[j] > 0.0 {
            for i in 0..n_samples {
                embedding[[i, j]] /= embedding_std[j];
            }
        }
    }
    
    // Step 5: Optimize embedding with SGD
    for epoch in 0..n_epochs {
        let alpha = learning_rate * (1.0 - epoch as f64 / n_epochs as f64);
        
        for i in 0..n_samples {
            for &j in &neighbors[i] {
                if i != j {
                    let weight = weights[[i, j]];
                    if weight > 0.0 {
                        // Attractive force
                        let diff = embedding.row(j) - embedding.row(i);
                        let dist_sq = diff.mapv(|x| x * x).sum();
                        let dist = (dist_sq + 1e-7).sqrt();
                        
                        let grad_coeff = -2.0 * weight / (1.0 + dist_sq);
                        let grad = diff * grad_coeff;
                        
                        embedding.row_mut(i).scaled_add(-alpha, &grad);
                        embedding.row_mut(j).scaled_add(alpha, &grad);
                    }
                }
            }
            
            // Sample negative examples
            for _ in 0..5 {
                let j = rand::random::<usize>() % n_samples;
                if i != j {
                    let diff = embedding.row(j) - embedding.row(i);
                    let dist_sq = diff.mapv(|x| x * x).sum();
                    
                    // Repulsive force
                    if dist_sq < spread * spread {
                        let grad_coeff = 2.0 / ((0.001 + dist_sq) * (1.0 + dist_sq));
                        let grad = diff * grad_coeff;
                        
                        embedding.row_mut(i).scaled_add(-alpha, &grad);
                    }
                }
            }
        }
    }
    
    Ok(embedding)
}

/// Classical MDS (Multidimensional Scaling) implementation
fn apply_mds_2d(data: &Array2<f64>) -> Result<Array2<f64>> {
    let n_samples = data.nrows();
    
    if n_samples < 3 {
        return apply_pca_2d(data);
    }

    // Step 1: Compute pairwise squared distances
    let distances_sq = compute_pairwise_distances_squared(data);
    
    // Step 2: Apply double centering to get Gram matrix
    // B = -0.5 * H * D^2 * H where H = I - (1/n) * 1 * 1^T
    let mut b_matrix = Array2::zeros((n_samples, n_samples));
    
    // Compute row means
    let row_means: Array1<f64> = distances_sq.mean_axis(Axis(1)).unwrap();
    
    // Compute overall mean
    let overall_mean = distances_sq.mean().unwrap();
    
    // Apply double centering
    for i in 0..n_samples {
        for j in 0..n_samples {
            b_matrix[[i, j]] = -0.5 * (distances_sq[[i, j]] - row_means[i] - row_means[j] + overall_mean);
        }
    }
    
    // Step 3: Eigenvalue decomposition of Gram matrix
    let (eigenvectors, eigenvalues) = compute_top_eigenvectors(&b_matrix, 2)?;
    
    // Step 4: Construct 2D embedding
    let mut embedding = Array2::zeros((n_samples, 2));
    for i in 0..n_samples {
        for j in 0..2 {
            if eigenvalues[j] > 0.0 {
                embedding[[i, j]] = eigenvectors[[i, j]] * eigenvalues[j].sqrt();
            }
        }
    }
    
    Ok(embedding)
}

/// Compute pairwise squared Euclidean distances
fn compute_pairwise_distances_squared(data: &Array2<f64>) -> Array2<f64> {
    let n_samples = data.nrows();
    let mut distances = Array2::zeros((n_samples, n_samples));
    
    for i in 0..n_samples {
        for j in i..n_samples {
            let mut dist_sq = 0.0;
            for k in 0..data.ncols() {
                let diff = data[[i, k]] - data[[j, k]];
                dist_sq += diff * diff;
            }
            distances[[i, j]] = dist_sq;
            distances[[j, i]] = dist_sq;
        }
    }
    
    distances
}

/// Compute pairwise Euclidean distances
fn compute_pairwise_distances(data: &Array2<f64>) -> Array2<f64> {
    let distances_sq = compute_pairwise_distances_squared(data);
    distances_sq.mapv(|x| x.sqrt())
}

/// Compute conditional probabilities for t-SNE
fn compute_conditional_probabilities(distances_sq: &Array2<f64>, perplexity: f64) -> Result<Array2<f64>> {
    let n_samples = distances_sq.nrows();
    let mut p_conditional = Array2::zeros((n_samples, n_samples));
    let target_entropy = perplexity.ln();
    
    for i in 0..n_samples {
        // Binary search for optimal sigma
        let mut sigma = 1.0;
        let mut sigma_min = 1e-20;
        let mut sigma_max = 1e20;
        
        for _ in 0..50 { // Max 50 iterations for binary search
            // Compute probabilities with current sigma
            let mut prob_sum = 0.0;
            let mut entropy = 0.0;
            
            for j in 0..n_samples {
                if i != j {
                    let prob = (-distances_sq[[i, j]] / (2.0 * sigma * sigma)).exp();
                    p_conditional[[i, j]] = prob;
                    prob_sum += prob;
                }
            }
            
            // Normalize probabilities
            if prob_sum > 0.0 {
                for j in 0..n_samples {
                    if i != j {
                        p_conditional[[i, j]] /= prob_sum;
                        if p_conditional[[i, j]] > 1e-12 {
                            entropy -= p_conditional[[i, j]] * p_conditional[[i, j]].ln();
                        }
                    }
                }
            }
            
            // Check convergence
            let entropy_diff = entropy - target_entropy;
            if entropy_diff.abs() < 1e-5 {
                break;
            }
            
            // Update sigma bounds
            if entropy_diff > 0.0 {
                sigma_min = sigma;
                sigma = if sigma_max == 1e20 { sigma * 2.0 } else { (sigma + sigma_max) / 2.0 };
            } else {
                sigma_max = sigma;
                sigma = (sigma + sigma_min) / 2.0;
            }
        }
    }
    
    Ok(p_conditional)
}

/// Compute joint probabilities for t-SNE
fn compute_joint_probabilities(p_conditional: &Array2<f64>) -> Array2<f64> {
    let n_samples = p_conditional.nrows();
    let mut p_joint = Array2::zeros((n_samples, n_samples));
    
    for i in 0..n_samples {
        for j in 0..n_samples {
            if i != j {
                p_joint[[i, j]] = (p_conditional[[i, j]] + p_conditional[[j, i]]) / (2.0 * n_samples as f64);
            }
        }
    }
    
    p_joint
}

/// Compute probabilities in low-dimensional space for t-SNE
fn compute_low_dim_probabilities(y: &Array2<f64>) -> Array2<f64> {
    let n_samples = y.nrows();
    let mut q_matrix = Array2::zeros((n_samples, n_samples));
    let mut sum_q = 0.0;
    
    // Compute pairwise probabilities using Student t-distribution
    for i in 0..n_samples {
        for j in i + 1..n_samples {
            let mut dist_sq = 0.0;
            for k in 0..2 {
                let diff = y[[i, k]] - y[[j, k]];
                dist_sq += diff * diff;
            }
            
            let q_val = 1.0 / (1.0 + dist_sq);
            q_matrix[[i, j]] = q_val;
            q_matrix[[j, i]] = q_val;
            sum_q += 2.0 * q_val;
        }
    }
    
    // Normalize
    if sum_q > 0.0 {
        q_matrix /= sum_q;
    }
    
    // Add small constant to avoid numerical issues
    q_matrix.mapv_inplace(|x| (x + 1e-12).max(1e-12));
    
    q_matrix
}

/// Compute gradient for t-SNE optimization
fn compute_tsne_gradient(p_matrix: &Array2<f64>, q_matrix: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    let n_samples = y.nrows();
    let mut gradient = Array2::zeros((n_samples, 2));
    
    for i in 0..n_samples {
        for j in 0..n_samples {
            if i != j {
                let p_ij = p_matrix[[i, j]];
                let q_ij = q_matrix[[i, j]];
                
                let mut dist_sq = 0.0;
                for k in 0..2 {
                    let diff = y[[i, k]] - y[[j, k]];
                    dist_sq += diff * diff;
                }
                
                let multiplier = 4.0 * (p_ij - q_ij) / (1.0 + dist_sq);
                
                for k in 0..2 {
                    let diff = y[[i, k]] - y[[j, k]];
                    gradient[[i, k]] += multiplier * diff;
                }
            }
        }
    }
    
    gradient
}

/// Find k-nearest neighbors for UMAP
fn find_k_nearest_neighbors(distances: &Array2<f64>, k: usize) -> Vec<Vec<usize>> {
    let n_samples = distances.nrows();
    let mut neighbors = Vec::with_capacity(n_samples);
    
    for i in 0..n_samples {
        let mut indices_distances: Vec<(usize, f64)> = (0..n_samples)
            .filter(|&j| i != j)
            .map(|j| (j, distances[[i, j]]))
            .collect();
        
        // Sort by distance and take k nearest
        indices_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let nearest_k: Vec<usize> = indices_distances.into_iter()
            .take(k)
            .map(|(idx, _)| idx)
            .collect();
        
        neighbors.push(nearest_k);
    }
    
    neighbors
}

/// Compute local connectivity for UMAP
fn compute_local_connectivity(distances: &Array2<f64>, neighbors: &[Vec<usize>]) -> Vec<f64> {
    let n_samples = distances.nrows();
    let mut sigmas = vec![1.0; n_samples];
    
    for i in 0..n_samples {
        if !neighbors[i].is_empty() {
            // Use distance to nearest neighbor as a starting point
            let nearest_dist = distances[[i, neighbors[i][0]]];
            sigmas[i] = nearest_dist.max(1e-3);
        }
    }
    
    sigmas
}

/// Compute UMAP edge weights
fn compute_umap_weights(distances: &Array2<f64>, neighbors: &[Vec<usize>], sigmas: &[f64]) -> Array2<f64> {
    let n_samples = distances.nrows();
    let mut weights = Array2::zeros((n_samples, n_samples));
    
    for i in 0..n_samples {
        for &j in &neighbors[i] {
            let dist = distances[[i, j]];
            let weight = (-((dist - sigmas[i]).max(0.0) / sigmas[i])).exp();
            weights[[i, j]] = weight;
        }
    }
    
    // Symmetrize weights
    for i in 0..n_samples {
        for j in 0..n_samples {
            let symmetric_weight = weights[[i, j]] + weights[[j, i]] - weights[[i, j]] * weights[[j, i]];
            weights[[i, j]] = symmetric_weight;
            weights[[j, i]] = symmetric_weight;
        }
    }
    
    weights
}

/// Compute top k eigenvectors using power iteration
fn compute_top_eigenvectors(matrix: &Array2<f64>, k: usize) -> Result<(Array2<f64>, Array1<f64>)> {
    let n = matrix.nrows();
    let max_iter = 100;
    let tolerance = 1e-8;
    
    let mut eigenvectors = Array2::zeros((n, k));
    let mut eigenvalues = Array1::zeros(k);
    let mut deflated_matrix = matrix.clone();
    
    for i in 0..k {
        // Random initialization
        let mut v = Array1::from_shape_fn(n, |_| rand::random::<f64>() - 0.5);
        v /= (v.dot(&v)).sqrt();
        
        // Power iteration
        for _ in 0..max_iter {
            let v_new = deflated_matrix.dot(&v);
            let norm = (v_new.dot(&v_new)).sqrt();
            
            if norm < tolerance {
                break;
            }
            
            let v_normalized = &v_new / norm;
            let convergence = (&v_normalized - &v).dot(&(&v_normalized - &v)).sqrt();
            
            v = v_normalized;
            
            if convergence < tolerance {
                break;
            }
        }
        
        // Store eigenvector and eigenvalue
        eigenvectors.column_mut(i).assign(&v);
        let eigenvalue = v.dot(&deflated_matrix.dot(&v));
        eigenvalues[i] = eigenvalue;
        
        // Deflate matrix (remove contribution of this eigenvector)
        let vv_t = v.view().insert_axis(Axis(1)).dot(&v.view().insert_axis(Axis(0)));
        deflated_matrix = deflated_matrix - eigenvalue * vv_t;
    }
    
    Ok((eigenvectors, eigenvalues))
}

/// Generate colors for clusters
fn generate_cluster_colors(cluster_ids: &[i32], scheme: ColorScheme) -> HashMap<i32, String> {
    let colors = match scheme {
        ColorScheme::Default => get_default_color_palette(),
        ColorScheme::ColorblindFriendly => get_colorblind_friendly_palette(),
        ColorScheme::HighContrast => get_high_contrast_palette(),
        ColorScheme::Pastel => get_pastel_palette(),
        ColorScheme::Viridis => get_viridis_palette(),
        ColorScheme::Plasma => get_plasma_palette(),
        ColorScheme::Custom => get_default_color_palette(), // Fallback
    };

    let mut color_map = HashMap::new();
    for (i, &cluster_id) in cluster_ids.iter().enumerate() {
        let color = colors[i % colors.len()].to_string();
        color_map.insert(cluster_id, color);
    }
    
    color_map
}

/// Default color palette
fn get_default_color_palette() -> Vec<&'static str> {
    vec![
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]
}

/// Colorblind-friendly palette
fn get_colorblind_friendly_palette() -> Vec<&'static str> {
    vec![
        "#0173b2", "#de8f05", "#029e73", "#cc78bc", "#ca9161",
        "#fbafe4", "#949494", "#ece133", "#56b4e9", "#f0e442",
    ]
}

/// High contrast palette
fn get_high_contrast_palette() -> Vec<&'static str> {
    vec![
        "#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff",
        "#ffff00", "#ff00ff", "#00ffff", "#800000", "#008000",
    ]
}

/// Pastel color palette
fn get_pastel_palette() -> Vec<&'static str> {
    vec![
        "#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6",
        "#ffffcc", "#e5d8bd", "#fddaec", "#f2f2f2", "#b3e2cd",
    ]
}

/// Viridis color palette
fn get_viridis_palette() -> Vec<&'static str> {
    vec![
        "#440154", "#482777", "#3f4a8a", "#31678e", "#26838f",
        "#1f9d8a", "#6cce5a", "#b6de2b", "#fee825", "#fff200",
    ]
}

/// Plasma color palette
fn get_plasma_palette() -> Vec<&'static str> {
    vec![
        "#0c0887", "#5c01a6", "#900da4", "#bf3984", "#e16462",
        "#f89441", "#fdc328", "#f0f921", "#fcffa4", "#ffffff",
    ]
}

/// Calculate 2D plot bounds
fn calculate_2d_bounds(data: &Array2<f64>) -> (f64, f64, f64, f64) {
    let x_values = data.column(0);
    let y_values = data.column(1);
    
    let min_x = x_values.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
    let max_x = x_values.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    let min_y = y_values.iter().fold(f64::INFINITY, |acc, &y| acc.min(y));
    let max_y = y_values.iter().fold(f64::NEG_INFINITY, |acc, &y| acc.max(y));
    
    // Add some padding
    let x_range = max_x - min_x;
    let y_range = max_y - min_y;
    let padding_x = x_range * 0.05;
    let padding_y = y_range * 0.05;
    
    (min_x - padding_x, max_x + padding_x, min_y - padding_y, max_y + padding_y)
}

/// Calculate 3D plot bounds
fn calculate_3d_bounds(data: &Array2<f64>) -> (f64, f64, f64, f64, f64, f64) {
    let x_values = data.column(0);
    let y_values = data.column(1);
    let z_values = data.column(2);
    
    let min_x = x_values.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
    let max_x = x_values.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    let min_y = y_values.iter().fold(f64::INFINITY, |acc, &y| acc.min(y));
    let max_y = y_values.iter().fold(f64::NEG_INFINITY, |acc, &y| acc.max(y));
    let min_z = z_values.iter().fold(f64::INFINITY, |acc, &z| acc.min(z));
    let max_z = z_values.iter().fold(f64::NEG_INFINITY, |acc, &z| acc.max(z));
    
    // Add some padding
    let x_range = max_x - min_x;
    let y_range = max_y - min_y;
    let z_range = max_z - min_z;
    let padding_x = x_range * 0.05;
    let padding_y = y_range * 0.05;
    let padding_z = z_range * 0.05;
    
    (
        min_x - padding_x, max_x + padding_x,
        min_y - padding_y, max_y + padding_y,
        min_z - padding_z, max_z + padding_z,
    )
}

/// Create legend entries
fn create_legend(cluster_ids: &[i32], colors: &HashMap<i32, String>, labels: &Array1<i32>) -> Vec<LegendEntry> {
    cluster_ids.iter().map(|&cluster_id| {
        let color = colors.get(&cluster_id).cloned().unwrap_or_else(|| "#000000".to_string());
        let count = labels.iter().filter(|&&label| label == cluster_id).count();
        
        LegendEntry {
            cluster_id,
            color,
            label: if cluster_id == -1 {
                "Noise".to_string()
            } else {
                format!("Cluster {}", cluster_id)
            },
            count,
        }
    }).collect()
}

/// Export visualization to various formats
pub mod export {
    use super::*;
    use std::io::Write;

    /// Export format for clustering visualizations
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ExportFormat {
        /// HTML with interactive JavaScript
        Html,
        /// SVG vector graphics
        Svg,
        /// JSON data format
        Json,
        /// CSV data format
        Csv,
        /// Plotly JSON format
        Plotly,
    }

    /// Export 2D scatter plot to HTML
    pub fn export_scatter_2d_to_html(plot: &ScatterPlot2D) -> Result<String> {
        let mut html = String::new();
        
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>2D Cluster Visualization</title>\n");
        html.push_str("<script src=\"https://d3js.org/d3.v7.min.js\"></script>\n");
        html.push_str("<style>\n");
        html.push_str("body { font-family: Arial, sans-serif; margin: 20px; }\n");
        html.push_str(".point { stroke: #000; stroke-width: 0.5; }\n");
        html.push_str(".centroid { stroke: #000; stroke-width: 2; fill-opacity: 0.8; }\n");
        html.push_str(".legend { font-size: 12px; }\n");
        html.push_str(".tooltip { position: absolute; background: #f9f9f9; border: 1px solid #ddd; padding: 5px; border-radius: 3px; pointer-events: none; }\n");
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");
        html.push_str("<h1>2D Cluster Visualization</h1>\n");
        html.push_str("<div id=\"plot\"></div>\n");
        html.push_str("<div id=\"legend\"></div>\n");

        // Add JavaScript for D3 visualization
        html.push_str("<script>\n");
        html.push_str("const margin = {top: 20, right: 20, bottom: 40, left: 40};\n");
        html.push_str("const width = 800 - margin.left - margin.right;\n");
        html.push_str("const height = 600 - margin.top - margin.bottom;\n");

        // Create SVG
        html.push_str("const svg = d3.select('#plot').append('svg')\n");
        html.push_str("  .attr('width', width + margin.left + margin.right)\n");
        html.push_str("  .attr('height', height + margin.top + margin.bottom)\n");
        html.push_str("  .append('g')\n");
        html.push_str("  .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');\n");

        // Add scales
        html.push_str(&format!(
            "const xScale = d3.scaleLinear().domain([{}, {}]).range([0, width]);\n",
            plot.bounds.0, plot.bounds.1
        ));
        html.push_str(&format!(
            "const yScale = d3.scaleLinear().domain([{}, {}]).range([height, 0]);\n",
            plot.bounds.2, plot.bounds.3
        ));

        // Add axes
        html.push_str("svg.append('g').attr('transform', 'translate(0,' + height + ')').call(d3.axisBottom(xScale));\n");
        html.push_str("svg.append('g').call(d3.axisLeft(yScale));\n");

        // Add data points
        html.push_str("const points = [\n");
        for (i, point) in plot.points.rows().into_iter().enumerate() {
            html.push_str(&format!(
                "  {{x: {}, y: {}, color: '{}', label: {}, size: {}}},\n",
                point[0], point[1], plot.colors[i], plot.labels[i], plot.sizes[i]
            ));
        }
        html.push_str("];\n");

        // Draw points
        html.push_str("svg.selectAll('.point').data(points).enter().append('circle')\n");
        html.push_str("  .attr('class', 'point')\n");
        html.push_str("  .attr('cx', d => xScale(d.x))\n");
        html.push_str("  .attr('cy', d => yScale(d.y))\n");
        html.push_str("  .attr('r', d => d.size)\n");
        html.push_str("  .attr('fill', d => d.color)\n");
        html.push_str("  .attr('opacity', 0.8);\n");

        // Add centroids if available
        if let Some(ref centroids) = plot.centroids {
            html.push_str("const centroids = [\n");
            for (i, centroid) in centroids.rows().into_iter().enumerate() {
                let color = if i < plot.legend.len() {
                    &plot.legend[i].color
                } else {
                    "#000000"
                };
                html.push_str(&format!(
                    "  {{x: {}, y: {}, color: '{}'}},\n",
                    centroid[0], centroid[1], color
                ));
            }
            html.push_str("];\n");

            html.push_str("svg.selectAll('.centroid').data(centroids).enter().append('circle')\n");
            html.push_str("  .attr('class', 'centroid')\n");
            html.push_str("  .attr('cx', d => xScale(d.x))\n");
            html.push_str("  .attr('cy', d => yScale(d.y))\n");
            html.push_str("  .attr('r', 8)\n");
            html.push_str("  .attr('fill', d => d.color);\n");
        }

        // Add legend
        html.push_str("const legend = d3.select('#legend').append('svg')\n");
        html.push_str("  .attr('width', 200).attr('height', 300);\n");
        
        html.push_str("const legendData = [\n");
        for entry in &plot.legend {
            html.push_str(&format!(
                "  {{color: '{}', label: '{}', count: {}}},\n",
                entry.color, entry.label, entry.count
            ));
        }
        html.push_str("];\n");

        html.push_str("legend.selectAll('.legend-item').data(legendData).enter().append('g')\n");
        html.push_str("  .attr('class', 'legend-item')\n");
        html.push_str("  .each(function(d, i) {\n");
        html.push_str("    const g = d3.select(this);\n");
        html.push_str("    g.append('circle')\n");
        html.push_str("      .attr('cx', 10)\n");
        html.push_str("      .attr('cy', i * 25 + 15)\n");
        html.push_str("      .attr('r', 5)\n");
        html.push_str("      .attr('fill', d.color);\n");
        html.push_str("    g.append('text')\n");
        html.push_str("      .attr('x', 25)\n");
        html.push_str("      .attr('y', i * 25 + 20)\n");
        html.push_str("      .text(d.label + ' (' + d.count + ')');\n");
        html.push_str("  });\n");

        html.push_str("</script>\n");
        html.push_str("</body>\n</html>");

        Ok(html)
    }

    /// Export 3D scatter plot to HTML with Three.js
    pub fn export_scatter_3d_to_html(plot: &ScatterPlot3D) -> Result<String> {
        let mut html = String::new();
        
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>3D Cluster Visualization</title>\n");
        html.push_str("<script src=\"https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js\"></script>\n");
        html.push_str("<script src=\"https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js\"></script>\n");
        html.push_str("<style>\n");
        html.push_str("body { margin: 0; padding: 0; background: #f0f0f0; font-family: Arial, sans-serif; }\n");
        html.push_str("#container { width: 100vw; height: 100vh; }\n");
        html.push_str("#info { position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.8); padding: 10px; border-radius: 5px; }\n");
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");
        html.push_str("<div id=\"container\"></div>\n");
        html.push_str("<div id=\"info\">\n");
        html.push_str("<h3>3D Cluster Visualization</h3>\n");
        html.push_str("<p>Use mouse to rotate, zoom, and pan</p>\n");
        html.push_str("</div>\n");

        // Add JavaScript for Three.js visualization
        html.push_str("<script>\n");
        html.push_str("let scene, camera, renderer, controls;\n");
        html.push_str("let points = [];\n");

        html.push_str("function init() {\n");
        html.push_str("  const container = document.getElementById('container');\n");
        html.push_str("  scene = new THREE.Scene();\n");
        html.push_str("  scene.background = new THREE.Color(0xffffff);\n");

        html.push_str("  camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);\n");
        html.push_str("  camera.position.set(10, 10, 10);\n");

        html.push_str("  renderer = new THREE.WebGLRenderer({ antialias: true });\n");
        html.push_str("  renderer.setSize(window.innerWidth, window.innerHeight);\n");
        html.push_str("  container.appendChild(renderer.domElement);\n");

        html.push_str("  controls = new THREE.OrbitControls(camera, renderer.domElement);\n");
        html.push_str("  controls.enableDamping = true;\n");

        // Add coordinate axes
        html.push_str("  const axesHelper = new THREE.AxesHelper(5);\n");
        html.push_str("  scene.add(axesHelper);\n");

        // Add data points
        html.push_str("  const pointsData = [\n");
        for (i, point) in plot.points.rows().into_iter().enumerate() {
            let color = &plot.colors[i];
            // Convert hex color to integer
            let color_int = u32::from_str_radix(&color[1..], 16).unwrap_or(0x000000);
            html.push_str(&format!(
                "    {{x: {}, y: {}, z: {}, color: 0x{:06x}, label: {}}},\n",
                point[0], point[1], point[2], color_int, plot.labels[i]
            ));
        }
        html.push_str("  ];\n");

        // Create point geometries and materials
        html.push_str("  pointsData.forEach(point => {\n");
        html.push_str("    const geometry = new THREE.SphereGeometry(0.1, 16, 16);\n");
        html.push_str("    const material = new THREE.MeshBasicMaterial({ color: point.color });\n");
        html.push_str("    const sphere = new THREE.Mesh(geometry, material);\n");
        html.push_str("    sphere.position.set(point.x, point.y, point.z);\n");
        html.push_str("    scene.add(sphere);\n");
        html.push_str("  });\n");

        // Add centroids if available
        if let Some(ref centroids) = plot.centroids {
            html.push_str("  const centroidsData = [\n");
            for (i, centroid) in centroids.rows().into_iter().enumerate() {
                let color = if i < plot.legend.len() {
                    &plot.legend[i].color
                } else {
                    "#000000"
                };
                let color_int = u32::from_str_radix(&color[1..], 16).unwrap_or(0x000000);
                html.push_str(&format!(
                    "    {{x: {}, y: {}, z: {}, color: 0x{:06x}}},\n",
                    centroid[0], centroid[1], centroid[2], color_int
                ));
            }
            html.push_str("  ];\n");

            html.push_str("  centroidsData.forEach(centroid => {\n");
            html.push_str("    const geometry = new THREE.SphereGeometry(0.2, 16, 16);\n");
            html.push_str("    const material = new THREE.MeshBasicMaterial({ color: centroid.color });\n");
            html.push_str("    const sphere = new THREE.Mesh(geometry, material);\n");
            html.push_str("    sphere.position.set(centroid.x, centroid.y, centroid.z);\n");
            html.push_str("    scene.add(sphere);\n");
            html.push_str("  });\n");
        }

        html.push_str("}\n");

        html.push_str("function animate() {\n");
        html.push_str("  requestAnimationFrame(animate);\n");
        html.push_str("  controls.update();\n");
        html.push_str("  renderer.render(scene, camera);\n");
        html.push_str("}\n");

        html.push_str("window.addEventListener('resize', function() {\n");
        html.push_str("  camera.aspect = window.innerWidth / window.innerHeight;\n");
        html.push_str("  camera.updateProjectionMatrix();\n");
        html.push_str("  renderer.setSize(window.innerWidth, window.innerHeight);\n");
        html.push_str("});\n");

        html.push_str("init();\n");
        html.push_str("animate();\n");
        html.push_str("</script>\n");
        html.push_str("</body>\n</html>");

        Ok(html)
    }

    /// Export plot to JSON format
    pub fn export_scatter_2d_to_json(plot: &ScatterPlot2D) -> Result<String> {
        serde_json::to_string_pretty(plot).map_err(|e| {
            ClusteringError::InvalidInput(format!("JSON serialization failed: {}", e))
        })
    }

    /// Export plot to JSON format
    pub fn export_scatter_3d_to_json(plot: &ScatterPlot3D) -> Result<String> {
        serde_json::to_string_pretty(plot).map_err(|e| {
            ClusteringError::InvalidInput(format!("JSON serialization failed: {}", e))
        })
    }

    /// Save visualization to file
    pub fn save_visualization_to_file<P: AsRef<std::path::Path>>(
        content: &str,
        path: P,
    ) -> Result<()> {
        let mut file = std::fs::File::create(path).map_err(|e| {
            ClusteringError::InvalidInput(format!("Failed to create file: {}", e))
        })?;
        file.write_all(content.as_bytes()).map_err(|e| {
            ClusteringError::InvalidInput(format!("Failed to write file: {}", e))
        })?;
        Ok(())
    }
}

/// Animation support for iterative algorithms and real-time streaming
pub mod animation {
    use super::*;
    use std::collections::VecDeque;
    use std::time::{Duration, Instant};

    /// Animation frame representing a single state in the clustering process
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct AnimationFrame {
        /// Frame number/timestamp
        pub frame_id: u32,
        /// Data points at this frame
        pub points: Array2<f64>,
        /// Cluster labels at this frame
        pub labels: Array1<i32>,
        /// Centroids at this frame (if available)
        pub centroids: Option<Array2<f64>>,
        /// Frame timestamp (milliseconds since animation start)
        pub timestamp_ms: u64,
        /// Algorithm iteration number (if applicable)
        pub iteration: Option<usize>,
        /// Convergence metrics (if available)
        pub convergence_info: Option<ConvergenceInfo>,
    }

    /// Convergence information for iterative algorithms
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct ConvergenceInfo {
        /// Current inertia (within-cluster sum of squares)
        pub inertia: f64,
        /// Change in inertia from previous iteration
        pub inertia_change: f64,
        /// Number of points that changed cluster assignment
        pub points_changed: usize,
        /// Maximum centroid movement distance
        pub max_centroid_movement: f64,
        /// Whether the algorithm has converged
        pub converged: bool,
    }

    /// Configuration for algorithm animations
    #[derive(Debug, Clone)]
    pub struct IterativeAnimationConfig {
        /// Animation speed (frames per second)
        pub fps: u32,
        /// Whether to capture intermediate iterations
        pub capture_intermediate: bool,
        /// Minimum time between captures (milliseconds)
        pub min_capture_interval_ms: u64,
        /// Maximum number of frames to store
        pub max_frames: usize,
        /// Whether to show convergence metrics
        pub show_convergence: bool,
        /// Centroid trail length (number of previous positions to show)
        pub centroid_trail_length: usize,
        /// Animation easing function
        pub easing: EasingFunction,
    }

    impl Default for IterativeAnimationConfig {
        fn default() -> Self {
            Self {
                fps: 30,
                capture_intermediate: true,
                min_capture_interval_ms: 100,
                max_frames: 1000,
                show_convergence: true,
                centroid_trail_length: 5,
                easing: EasingFunction::EaseInOut,
            }
        }
    }

    /// Animation recorder for iterative clustering algorithms
    #[derive(Debug)]
    pub struct IterativeAnimationRecorder {
        /// Animation configuration
        pub config: IterativeAnimationConfig,
        /// Recorded animation frames
        pub frames: Vec<AnimationFrame>,
        /// Animation start time
        start_time: Option<Instant>,
        /// Last capture time
        last_capture: Option<Instant>,
        /// Current frame counter
        frame_counter: u32,
    }

    impl IterativeAnimationRecorder {
        /// Create new animation recorder
        pub fn new(config: IterativeAnimationConfig) -> Self {
            Self {
                config,
                frames: Vec::new(),
                start_time: None,
                last_capture: None,
                frame_counter: 0,
            }
        }

        /// Start recording animation
        pub fn start_recording(&mut self) {
            self.start_time = Some(Instant::now());
            self.last_capture = None;
            self.frame_counter = 0;
            self.frames.clear();
        }

        /// Capture current algorithm state
        pub fn capture_frame(
            &mut self,
            points: ArrayView2<f64>,
            labels: &Array1<i32>,
            centroids: Option<&Array2<f64>>,
            iteration: Option<usize>,
            convergence_info: Option<ConvergenceInfo>,
        ) -> Result<()> {
            let now = Instant::now();
            
            // Check if we should capture this frame
            if let Some(last) = self.last_capture {
                let elapsed = now.duration_since(last).as_millis() as u64;
                if elapsed < self.config.min_capture_interval_ms {
                    return Ok(());
                }
            }

            // Calculate timestamp
            let timestamp_ms = if let Some(start) = self.start_time {
                now.duration_since(start).as_millis() as u64
            } else {
                0
            };

            // Create animation frame
            let frame = AnimationFrame {
                frame_id: self.frame_counter,
                points: points.to_owned(),
                labels: labels.clone(),
                centroids: centroids.map(|c| c.to_owned()),
                timestamp_ms,
                iteration,
                convergence_info,
            };

            // Add frame to collection
            self.frames.push(frame);
            self.frame_counter += 1;
            self.last_capture = Some(now);

            // Limit frame count
            if self.frames.len() > self.config.max_frames {
                self.frames.remove(0);
            }

            Ok(())
        }

        /// Get all recorded frames
        pub fn get_frames(&self) -> &[AnimationFrame] {
            &self.frames
        }

        /// Export animation to HTML with JavaScript playback controls
        pub fn export_to_html(&self, vis_config: &VisualizationConfig) -> Result<String> {
            if self.frames.is_empty() {
                return Err(ClusteringError::InvalidInput(
                    "No animation frames to export".to_string(),
                ));
            }

            let mut html = String::new();
            
            html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
            html.push_str("<title>Iterative Clustering Animation</title>\n");
            html.push_str("<script src=\"https://d3js.org/d3.v7.min.js\"></script>\n");
            html.push_str("<style>\n");
            html.push_str("body { font-family: Arial, sans-serif; margin: 20px; }\n");
            html.push_str(".controls { margin-bottom: 20px; }\n");
            html.push_str(".controls button { margin-right: 10px; padding: 5px 15px; }\n");
            html.push_str(".info-panel { margin-left: 20px; padding: 10px; background: #f9f9f9; border-radius: 5px; }\n");
            html.push_str(".point { stroke: #000; stroke-width: 0.5; }\n");
            html.push_str(".centroid { stroke: #000; stroke-width: 2; fill-opacity: 0.8; }\n");
            html.push_str(".centroid-trail { fill: none; stroke-width: 1; opacity: 0.5; }\n");
            html.push_str("#main-container { display: flex; }\n");
            html.push_str("#plot-container { flex: 1; }\n");
            html.push_str("#info-container { width: 250px; }\n");
            html.push_str("</style>\n");
            html.push_str("</head>\n<body>\n");
            
            html.push_str("<h1>Iterative Clustering Animation</h1>\n");
            html.push_str("<div class=\"controls\">\n");
            html.push_str("  <button id=\"play-btn\">Play</button>\n");
            html.push_str("  <button id=\"pause-btn\">Pause</button>\n");
            html.push_str("  <button id=\"reset-btn\">Reset</button>\n");
            html.push_str("  <button id=\"step-btn\">Step</button>\n");
            html.push_str("  <label>Speed: <input id=\"speed-slider\" type=\"range\" min=\"1\" max=\"10\" value=\"5\"></label>\n");
            html.push_str("  <label>Frame: <input id=\"frame-slider\" type=\"range\" min=\"0\" max=\"\" value=\"0\"></label>\n");
            html.push_str("</div>\n");
            
            html.push_str("<div id=\"main-container\">\n");
            html.push_str("  <div id=\"plot-container\"></div>\n");
            html.push_str("  <div id=\"info-container\">\n");
            html.push_str("    <div class=\"info-panel\">\n");
            html.push_str("      <h3>Animation Info</h3>\n");
            html.push_str("      <div id=\"frame-info\"></div>\n");
            html.push_str("      <div id=\"convergence-info\"></div>\n");
            html.push_str("    </div>\n");
            html.push_str("  </div>\n");
            html.push_str("</div>\n");

            // Add JavaScript animation engine
            html.push_str("<script>\n");
            
            // Embed frame data
            html.push_str("const animationFrames = [\n");
            for frame in &self.frames {
                html.push_str("  {\n");
                html.push_str(&format!("    frameId: {},\n", frame.frame_id));
                html.push_str(&format!("    timestampMs: {},\n", frame.timestamp_ms));
                html.push_str(&format!("    iteration: {},\n", frame.iteration.unwrap_or(0)));
                
                // Points data
                html.push_str("    points: [\n");
                for (i, point) in frame.points.rows().into_iter().enumerate() {
                    html.push_str(&format!(
                        "      {{x: {}, y: {}, label: {}}},\n",
                        point[0], point[1], frame.labels[i]
                    ));
                }
                html.push_str("    ],\n");
                
                // Centroids data
                if let Some(ref centroids) = frame.centroids {
                    html.push_str("    centroids: [\n");
                    for centroid in centroids.rows() {
                        html.push_str(&format!("      {{x: {}, y: {}}},\n", centroid[0], centroid[1]));
                    }
                    html.push_str("    ],\n");
                } else {
                    html.push_str("    centroids: [],\n");
                }
                
                // Convergence info
                if let Some(ref conv) = frame.convergence_info {
                    html.push_str(&format!(
                        "    convergence: {{ inertia: {}, inertiaChange: {}, pointsChanged: {}, converged: {} }},\n",
                        conv.inertia, conv.inertia_change, conv.points_changed, conv.converged
                    ));
                } else {
                    html.push_str("    convergence: null,\n");
                }
                
                html.push_str("  },\n");
            }
            html.push_str("];\n");

            // Animation control logic
            html.push_str(&format!("const maxFrames = {};\n", self.frames.len()));
            html.push_str("let currentFrame = 0;\n");
            html.push_str("let isPlaying = false;\n");
            html.push_str("let animationSpeed = 5;\n");
            html.push_str("let animationTimer = null;\n");
            
            // Create SVG
            html.push_str("const margin = {top: 20, right: 20, bottom: 40, left: 40};\n");
            html.push_str("const width = 600 - margin.left - margin.right;\n");
            html.push_str("const height = 400 - margin.top - margin.bottom;\n");

            html.push_str("const svg = d3.select('#plot-container').append('svg')\n");
            html.push_str("  .attr('width', width + margin.left + margin.right)\n");
            html.push_str("  .attr('height', height + margin.top + margin.bottom)\n");
            html.push_str("  .append('g')\n");
            html.push_str("  .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');\n");

            // Calculate bounds from all frames
            html.push_str("let allPoints = [];\n");
            html.push_str("animationFrames.forEach(frame => {\n");
            html.push_str("  allPoints = allPoints.concat(frame.points);\n");
            html.push_str("  if (frame.centroids) allPoints = allPoints.concat(frame.centroids);\n");
            html.push_str("});\n");
            
            html.push_str("const xExtent = d3.extent(allPoints, d => d.x);\n");
            html.push_str("const yExtent = d3.extent(allPoints, d => d.y);\n");
            html.push_str("const xScale = d3.scaleLinear().domain(xExtent).range([0, width]);\n");
            html.push_str("const yScale = d3.scaleLinear().domain(yExtent).range([height, 0]);\n");

            // Add axes
            html.push_str("svg.append('g').attr('transform', 'translate(0,' + height + ')').call(d3.axisBottom(xScale));\n");
            html.push_str("svg.append('g').call(d3.axisLeft(yScale));\n");

            // Update frame function
            html.push_str("function updateFrame() {\n");
            html.push_str("  const frame = animationFrames[currentFrame];\n");
            html.push_str("  if (!frame) return;\n");

            // Update points
            html.push_str("  const points = svg.selectAll('.point').data(frame.points, d => d.x + ',' + d.y);\n");
            html.push_str("  points.exit().remove();\n");
            html.push_str("  points.enter().append('circle').attr('class', 'point')\n");
            html.push_str("    .merge(points)\n");
            html.push_str("    .transition().duration(100)\n");
            html.push_str("    .attr('cx', d => xScale(d.x))\n");
            html.push_str("    .attr('cy', d => yScale(d.y))\n");
            html.push_str("    .attr('r', 4)\n");
            html.push_str("    .attr('fill', d => d.label === 0 ? '#1f77b4' : d.label === 1 ? '#ff7f0e' : '#2ca02c')\n");
            html.push_str("    .attr('opacity', 0.8);\n");

            // Update centroids
            html.push_str("  const centroids = svg.selectAll('.centroid').data(frame.centroids || []);\n");
            html.push_str("  centroids.exit().remove();\n");
            html.push_str("  centroids.enter().append('circle').attr('class', 'centroid')\n");
            html.push_str("    .merge(centroids)\n");
            html.push_str("    .transition().duration(200)\n");
            html.push_str("    .attr('cx', d => xScale(d.x))\n");
            html.push_str("    .attr('cy', d => yScale(d.y))\n");
            html.push_str("    .attr('r', 8)\n");
            html.push_str("    .attr('fill', 'red')\n");
            html.push_str("    .attr('opacity', 0.9);\n");

            // Update info panel
            html.push_str("  d3.select('#frame-info').html(\n");
            html.push_str("    `<p><strong>Frame:</strong> ${frame.frameId + 1} / ${maxFrames}</p>` +\n");
            html.push_str("    `<p><strong>Iteration:</strong> ${frame.iteration || 'N/A'}</p>` +\n");
            html.push_str("    `<p><strong>Timestamp:</strong> ${frame.timestampMs}ms</p>`\n");
            html.push_str("  );\n");

            html.push_str("  if (frame.convergence) {\n");
            html.push_str("    d3.select('#convergence-info').html(\n");
            html.push_str("      `<h4>Convergence</h4>` +\n");
            html.push_str("      `<p><strong>Inertia:</strong> ${frame.convergence.inertia.toFixed(2)}</p>` +\n");
            html.push_str("      `<p><strong>Change:</strong> ${frame.convergence.inertiaChange.toFixed(4)}</p>` +\n");
            html.push_str("      `<p><strong>Points Changed:</strong> ${frame.convergence.pointsChanged}</p>` +\n");
            html.push_str("      `<p><strong>Converged:</strong> ${frame.convergence.converged ? 'Yes' : 'No'}</p>`\n");
            html.push_str("    );\n");
            html.push_str("  }\n");

            // Update frame slider
            html.push_str("  d3.select('#frame-slider').property('value', currentFrame);\n");
            html.push_str("}\n");

            // Animation control functions
            html.push_str("function play() {\n");
            html.push_str("  if (isPlaying) return;\n");
            html.push_str("  isPlaying = true;\n");
            html.push_str("  const interval = 1000 / (animationSpeed * 2);\n");
            html.push_str("  animationTimer = setInterval(() => {\n");
            html.push_str("    currentFrame = (currentFrame + 1) % maxFrames;\n");
            html.push_str("    updateFrame();\n");
            html.push_str("  }, interval);\n");
            html.push_str("}\n");

            html.push_str("function pause() {\n");
            html.push_str("  isPlaying = false;\n");
            html.push_str("  if (animationTimer) clearInterval(animationTimer);\n");
            html.push_str("}\n");

            html.push_str("function reset() {\n");
            html.push_str("  pause();\n");
            html.push_str("  currentFrame = 0;\n");
            html.push_str("  updateFrame();\n");
            html.push_str("}\n");

            html.push_str("function step() {\n");
            html.push_str("  pause();\n");
            html.push_str("  currentFrame = (currentFrame + 1) % maxFrames;\n");
            html.push_str("  updateFrame();\n");
            html.push_str("}\n");

            // Event listeners
            html.push_str("d3.select('#play-btn').on('click', play);\n");
            html.push_str("d3.select('#pause-btn').on('click', pause);\n");
            html.push_str("d3.select('#reset-btn').on('click', reset);\n");
            html.push_str("d3.select('#step-btn').on('click', step);\n");
            
            html.push_str("d3.select('#speed-slider').on('input', function() {\n");
            html.push_str("  animationSpeed = +this.value;\n");
            html.push_str("  if (isPlaying) { pause(); play(); }\n");
            html.push_str("});\n");

            html.push_str("d3.select('#frame-slider')\n");
            html.push_str("  .attr('max', maxFrames - 1)\n");
            html.push_str("  .on('input', function() {\n");
            html.push_str("    pause();\n");
            html.push_str("    currentFrame = +this.value;\n");
            html.push_str("    updateFrame();\n");
            html.push_str("  });\n");

            // Initialize
            html.push_str("updateFrame();\n");
            
            html.push_str("</script>\n");
            html.push_str("</body>\n</html>");

            Ok(html)
        }
    }

    /// Real-time streaming visualization
    #[derive(Debug)]
    pub struct StreamingVisualizer {
        /// Streaming configuration
        pub config: StreamingConfig,
        /// Current data buffer
        pub data_buffer: VecDeque<Array1<f64>>,
        /// Current labels buffer
        pub labels_buffer: VecDeque<i32>,
        /// Animation recorder
        pub recorder: Option<IterativeAnimationRecorder>,
        /// Last update time
        last_update: Option<Instant>,
    }

    /// Configuration for streaming visualization
    #[derive(Debug, Clone)]
    pub struct StreamingConfig {
        /// Maximum number of points to keep in buffer
        pub max_buffer_size: usize,
        /// Update frequency (milliseconds)
        pub update_interval_ms: u64,
        /// Whether to record streaming animation
        pub record_animation: bool,
        /// Fade out old points
        pub fade_old_points: bool,
        /// Number of recent points to highlight
        pub highlight_recent: usize,
    }

    impl Default for StreamingConfig {
        fn default() -> Self {
            Self {
                max_buffer_size: 10000,
                update_interval_ms: 100,
                record_animation: true,
                fade_old_points: true,
                highlight_recent: 50,
            }
        }
    }

    impl StreamingVisualizer {
        /// Create new streaming visualizer
        pub fn new(config: StreamingConfig) -> Self {
            let recorder = if config.record_animation {
                Some(IterativeAnimationRecorder::new(IterativeAnimationConfig::default()))
            } else {
                None
            };

            Self {
                config,
                data_buffer: VecDeque::new(),
                labels_buffer: VecDeque::new(),
                recorder,
                last_update: None,
            }
        }

        /// Add new data point to the stream
        pub fn add_point(&mut self, point: Array1<f64>, label: i32) -> Result<bool> {
            self.data_buffer.push_back(point);
            self.labels_buffer.push_back(label);

            // Maintain buffer size
            if self.data_buffer.len() > self.config.max_buffer_size {
                self.data_buffer.pop_front();
                self.labels_buffer.pop_front();
            }

            // Check if we should update visualization
            let should_update = if let Some(last) = self.last_update {
                let elapsed = last.elapsed().as_millis() as u64;
                elapsed >= self.config.update_interval_ms
            } else {
                true
            };

            if should_update {
                self.last_update = Some(Instant::now());
                
                // Record frame if animation recording is enabled
                if let Some(ref mut recorder) = self.recorder {
                    let data_matrix = self.get_data_matrix()?;
                    let labels_array = Array1::from(self.labels_buffer.iter().cloned().collect::<Vec<_>>());
                    recorder.capture_frame(data_matrix.view(), &labels_array, None, None, None)?;
                }
            }

            Ok(should_update)
        }

        /// Get current data as matrix
        pub fn get_data_matrix(&self) -> Result<Array2<f64>> {
            if self.data_buffer.is_empty() {
                return Err(ClusteringError::InvalidInput("No data in buffer".to_string()));
            }

            let n_points = self.data_buffer.len();
            let n_features = self.data_buffer[0].len();
            
            let mut data = Array2::zeros((n_points, n_features));
            for (i, point) in self.data_buffer.iter().enumerate() {
                data.row_mut(i).assign(point);
            }

            Ok(data)
        }

        /// Get current labels as array
        pub fn get_labels_array(&self) -> Array1<i32> {
            Array1::from(self.labels_buffer.iter().cloned().collect::<Vec<_>>())
        }

        /// Export streaming animation
        pub fn export_streaming_animation(&self, vis_config: &VisualizationConfig) -> Result<String> {
            if let Some(ref recorder) = self.recorder {
                recorder.export_to_html(vis_config)
            } else {
                Err(ClusteringError::InvalidInput(
                    "Animation recording is not enabled".to_string(),
                ))
            }
        }
    }
}

/// Interactive visualization features
pub mod interactive {
    use super::*;
    use std::collections::BTreeMap;

    /// Interactive visualization configuration
    #[derive(Debug, Clone)]
    pub struct InteractiveConfig {
        /// Enable point selection
        pub enable_selection: bool,
        /// Enable zooming and panning
        pub enable_zoom_pan: bool,
        /// Enable clustering on-the-fly
        pub enable_live_clustering: bool,
        /// Show cluster statistics on hover
        pub show_statistics: bool,
        /// Enable brushing (rectangular selection)
        pub enable_brushing: bool,
        /// Update frequency for live features (ms)
        pub update_frequency_ms: u32,
    }

    impl Default for InteractiveConfig {
        fn default() -> Self {
            Self {
                enable_selection: true,
                enable_zoom_pan: true,
                enable_live_clustering: false,
                show_statistics: true,
                enable_brushing: true,
                update_frequency_ms: 100,
            }
        }
    }

    /// Interactive visualization state
    #[derive(Debug, Clone)]
    pub struct InteractiveState {
        /// Currently selected points
        pub selected_points: Vec<usize>,
        /// Current zoom level
        pub zoom_level: f64,
        /// Current pan offset (x, y)
        pub pan_offset: (f64, f64),
        /// Brush selection area (x1, y1, x2, y2)
        pub brush_area: Option<(f64, f64, f64, f64)>,
        /// Current cluster statistics
        pub cluster_stats: BTreeMap<i32, ClusterStats>,
    }

    /// Statistics for a cluster
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct ClusterStats {
        /// Cluster ID
        pub cluster_id: i32,
        /// Number of points
        pub point_count: usize,
        /// Centroid coordinates
        pub centroid: Vec<f64>,
        /// Cluster radius (average distance to centroid)
        pub radius: f64,
        /// Cluster density
        pub density: f64,
        /// Inertia (within-cluster sum of squares)
        pub inertia: f64,
    }

    impl Default for InteractiveState {
        fn default() -> Self {
            Self {
                selected_points: Vec::new(),
                zoom_level: 1.0,
                pan_offset: (0.0, 0.0),
                brush_area: None,
                cluster_stats: BTreeMap::new(),
            }
        }
    }

    /// Interactive visualization manager
    #[derive(Debug)]
    pub struct InteractiveVisualizer {
        /// Current visualization state
        pub state: InteractiveState,
        /// Configuration
        pub config: InteractiveConfig,
        /// Cached plot data
        pub plot_2d: Option<ScatterPlot2D>,
        /// Cached plot data
        pub plot_3d: Option<ScatterPlot3D>,
    }

    impl InteractiveVisualizer {
        /// Create new interactive visualizer
        pub fn new(config: InteractiveConfig) -> Self {
            Self {
                state: InteractiveState::default(),
                config,
                plot_2d: None,
                plot_3d: None,
            }
        }

        /// Update with new 2D plot data
        pub fn update_plot_2d(&mut self, plot: ScatterPlot2D) {
            self.plot_2d = Some(plot);
            self.update_cluster_statistics();
        }

        /// Update with new 3D plot data
        pub fn update_plot_3d(&mut self, plot: ScatterPlot3D) {
            self.plot_3d = Some(plot);
            self.update_cluster_statistics();
        }

        /// Update cluster statistics
        fn update_cluster_statistics(&mut self) {
            self.state.cluster_stats.clear();

            if let Some(ref plot) = self.plot_2d {
                self.calculate_2d_cluster_stats(plot);
            } else if let Some(ref plot) = self.plot_3d {
                self.calculate_3d_cluster_stats(plot);
            }
        }

        /// Calculate cluster statistics for 2D plot
        fn calculate_2d_cluster_stats(&mut self, plot: &ScatterPlot2D) {
            let unique_labels: std::collections::HashSet<i32> = plot.labels.iter().cloned().collect();

            for &cluster_id in &unique_labels {
                let cluster_points: Vec<_> = plot.points.rows()
                    .into_iter()
                    .zip(plot.labels.iter())
                    .filter(|(_, &label)| label == cluster_id)
                    .map(|(point, _)| point)
                    .collect();

                if cluster_points.is_empty() {
                    continue;
                }

                // Calculate centroid
                let centroid = if cluster_points.len() == 1 {
                    vec![cluster_points[0][0], cluster_points[0][1]]
                } else {
                    let sum_x: f64 = cluster_points.iter().map(|p| p[0]).sum();
                    let sum_y: f64 = cluster_points.iter().map(|p| p[1]).sum();
                    vec![sum_x / cluster_points.len() as f64, sum_y / cluster_points.len() as f64]
                };

                // Calculate radius and inertia
                let mut total_distance = 0.0;
                let mut inertia = 0.0;
                for point in &cluster_points {
                    let dx = point[0] - centroid[0];
                    let dy = point[1] - centroid[1];
                    let distance = (dx * dx + dy * dy).sqrt();
                    total_distance += distance;
                    inertia += dx * dx + dy * dy;
                }

                let radius = total_distance / cluster_points.len() as f64;
                let density = cluster_points.len() as f64 / (std::f64::consts::PI * radius * radius);

                let stats = ClusterStats {
                    cluster_id,
                    point_count: cluster_points.len(),
                    centroid,
                    radius,
                    density,
                    inertia,
                };

                self.state.cluster_stats.insert(cluster_id, stats);
            }
        }

        /// Calculate cluster statistics for 3D plot
        fn calculate_3d_cluster_stats(&mut self, plot: &ScatterPlot3D) {
            let unique_labels: std::collections::HashSet<i32> = plot.labels.iter().cloned().collect();

            for &cluster_id in &unique_labels {
                let cluster_points: Vec<_> = plot.points.rows()
                    .into_iter()
                    .zip(plot.labels.iter())
                    .filter(|(_, &label)| label == cluster_id)
                    .map(|(point, _)| point)
                    .collect();

                if cluster_points.is_empty() {
                    continue;
                }

                // Calculate centroid
                let centroid = if cluster_points.len() == 1 {
                    vec![cluster_points[0][0], cluster_points[0][1], cluster_points[0][2]]
                } else {
                    let sum_x: f64 = cluster_points.iter().map(|p| p[0]).sum();
                    let sum_y: f64 = cluster_points.iter().map(|p| p[1]).sum();
                    let sum_z: f64 = cluster_points.iter().map(|p| p[2]).sum();
                    vec![
                        sum_x / cluster_points.len() as f64,
                        sum_y / cluster_points.len() as f64,
                        sum_z / cluster_points.len() as f64,
                    ]
                };

                // Calculate radius and inertia
                let mut total_distance = 0.0;
                let mut inertia = 0.0;
                for point in &cluster_points {
                    let dx = point[0] - centroid[0];
                    let dy = point[1] - centroid[1];
                    let dz = point[2] - centroid[2];
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    total_distance += distance;
                    inertia += dx * dx + dy * dy + dz * dz;
                }

                let radius = total_distance / cluster_points.len() as f64;
                let volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
                let density = cluster_points.len() as f64 / volume;

                let stats = ClusterStats {
                    cluster_id,
                    point_count: cluster_points.len(),
                    centroid,
                    radius,
                    density,
                    inertia,
                };

                self.state.cluster_stats.insert(cluster_id, stats);
            }
        }

        /// Get statistics for a specific cluster
        pub fn get_cluster_stats(&self, cluster_id: i32) -> Option<&ClusterStats> {
            self.state.cluster_stats.get(&cluster_id)
        }

        /// Get all cluster statistics
        pub fn get_all_cluster_stats(&self) -> &BTreeMap<i32, ClusterStats> {
            &self.state.cluster_stats
        }

        /// Export interactive state to JSON
        pub fn export_state_to_json(&self) -> Result<String> {
            #[derive(serde::Serialize)]
            struct ExportState {
                selected_points: Vec<usize>,
                zoom_level: f64,
                pan_offset: (f64, f64),
                brush_area: Option<(f64, f64, f64, f64)>,
                cluster_stats: BTreeMap<i32, ClusterStats>,
            }

            let export_state = ExportState {
                selected_points: self.state.selected_points.clone(),
                zoom_level: self.state.zoom_level,
                pan_offset: self.state.pan_offset,
                brush_area: self.state.brush_area,
                cluster_stats: self.state.cluster_stats.clone(),
            };

            serde_json::to_string_pretty(&export_state).map_err(|e| {
                ClusteringError::InvalidInput(format!("JSON serialization failed: {}", e))
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_create_scatter_plot_2d() {
        let data = Array2::from_shape_vec(
            (6, 4),
            vec![
                1.0, 2.0, 3.0, 4.0,
                2.0, 3.0, 4.0, 5.0,
                1.5, 2.5, 3.5, 4.5,
                10.0, 11.0, 12.0, 13.0,
                11.0, 12.0, 13.0, 14.0,
                10.5, 11.5, 12.5, 13.5,
            ],
        ).unwrap();

        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);
        let config = VisualizationConfig::default();

        let plot = create_scatter_plot_2d(data.view(), &labels, None, &config).unwrap();

        assert_eq!(plot.points.nrows(), 6);
        assert_eq!(plot.points.ncols(), 2);
        assert_eq!(plot.labels.len(), 6);
        assert_eq!(plot.colors.len(), 6);
        assert_eq!(plot.legend.len(), 2);
    }

    #[test]
    fn test_create_scatter_plot_3d() {
        let data = Array2::from_shape_vec(
            (4, 5),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0,
                2.0, 3.0, 4.0, 5.0, 6.0,
                10.0, 11.0, 12.0, 13.0, 14.0,
                11.0, 12.0, 13.0, 14.0, 15.0,
            ],
        ).unwrap();

        let labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let config = VisualizationConfig::default();

        let plot = create_scatter_plot_3d(data.view(), &labels, None, &config).unwrap();

        assert_eq!(plot.points.nrows(), 4);
        assert_eq!(plot.points.ncols(), 3);
        assert_eq!(plot.labels.len(), 4);
        assert_eq!(plot.colors.len(), 4);
        assert_eq!(plot.legend.len(), 2);
    }

    #[test]
    fn test_color_schemes() {
        let cluster_ids = vec![0, 1, 2, -1];
        
        let schemes = vec![
            ColorScheme::Default,
            ColorScheme::ColorblindFriendly,
            ColorScheme::HighContrast,
            ColorScheme::Pastel,
            ColorScheme::Viridis,
            ColorScheme::Plasma,
        ];

        for scheme in schemes {
            let colors = generate_cluster_colors(&cluster_ids, scheme);
            assert_eq!(colors.len(), 4);
            
            for color in colors.values() {
                assert!(color.starts_with('#'));
                assert_eq!(color.len(), 7);
            }
        }
    }

    #[test]
    fn test_dimensionality_reduction_first2d() {
        let data = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
            ],
        ).unwrap();

        let reduced = apply_dimensionality_reduction_2d(
            data.view(),
            DimensionalityReduction::First2D,
        ).unwrap();

        assert_eq!(reduced.shape(), &[3, 2]);
        assert_eq!(reduced[[0, 0]], 1.0);
        assert_eq!(reduced[[0, 1]], 2.0);
        assert_eq!(reduced[[1, 0]], 5.0);
        assert_eq!(reduced[[1, 1]], 6.0);
    }

    #[test]
    fn test_pca_computation() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
                10.0, 11.0, 12.0,
            ],
        ).unwrap();

        let result = apply_pca_2d(&data);
        assert!(result.is_ok());
        
        let pca_result = result.unwrap();
        assert_eq!(pca_result.shape(), &[4, 2]);
    }

    #[test]
    fn test_bounds_calculation() {
        let data = Array2::from_shape_vec(
            (3, 2),
            vec![0.0, 0.0, 5.0, 5.0, 2.5, 2.5],
        ).unwrap();

        let bounds = calculate_2d_bounds(&data);
        
        // Should have some padding around the actual data bounds
        assert!(bounds.0 < 0.0); // min_x with padding
        assert!(bounds.1 > 5.0); // max_x with padding
        assert!(bounds.2 < 0.0); // min_y with padding
        assert!(bounds.3 > 5.0); // max_y with padding
    }

    #[test]
    fn test_interactive_state() {
        let config = interactive::InteractiveConfig::default();
        let mut visualizer = interactive::InteractiveVisualizer::new(config);

        // Test that initial state is correct
        assert!(visualizer.state.selected_points.is_empty());
        assert_eq!(visualizer.state.zoom_level, 1.0);
        assert_eq!(visualizer.state.pan_offset, (0.0, 0.0));
        assert!(visualizer.state.cluster_stats.is_empty());

        // Test updating with plot data
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0],
        ).unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let config = VisualizationConfig::default();
        let plot = create_scatter_plot_2d(data.view(), &labels, None, &config).unwrap();

        visualizer.update_plot_2d(plot);

        // Should have calculated cluster statistics
        assert_eq!(visualizer.state.cluster_stats.len(), 2);
        assert!(visualizer.state.cluster_stats.contains_key(&0));
        assert!(visualizer.state.cluster_stats.contains_key(&1));
    }
}