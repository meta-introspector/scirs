//! Isomap (Isometric Feature Mapping) for non-linear dimensionality reduction
//!
//! Isomap is a non-linear dimensionality reduction method that preserves geodesic 
//! distances between all points. It extends MDS by using geodesic distances instead
//! of Euclidean distances.

use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use num_traits::{Float, NumCast};
use scirs2_linalg::{eigh, svd};
use std::collections::BinaryHeap;
use std::f64;

use crate::error::{Result, TransformError};

/// Isomap (Isometric Feature Mapping) dimensionality reduction
///
/// Isomap seeks a lower-dimensional embedding that maintains geodesic distances
/// between all points. It uses graph distances to approximate geodesic distances
/// on the manifold.
#[derive(Debug, Clone)]
pub struct Isomap {
    /// Number of neighbors to use for graph construction
    n_neighbors: usize,
    /// Number of components for dimensionality reduction
    n_components: usize,
    /// Whether to use k-neighbors or epsilon-ball for graph construction
    neighbor_mode: String,
    /// Epsilon for epsilon-ball graph construction
    epsilon: f64,
    /// The embedding vectors
    embedding: Option<Array2<f64>>,
    /// Training data for out-of-sample extension
    training_data: Option<Array2<f64>>,
    /// Geodesic distances from training data
    geodesic_distances: Option<Array2<f64>>,
}

impl Isomap {
    /// Creates a new Isomap instance
    ///
    /// # Arguments
    /// * `n_neighbors` - Number of neighbors for graph construction
    /// * `n_components` - Number of dimensions in the embedding space
    pub fn new(n_neighbors: usize, n_components: usize) -> Self {
        Isomap {
            n_neighbors,
            n_components,
            neighbor_mode: "knn".to_string(),
            epsilon: 0.0,
            embedding: None,
            training_data: None,
            geodesic_distances: None,
        }
    }

    /// Use epsilon-ball instead of k-nearest neighbors
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.neighbor_mode = "epsilon".to_string();
        self.epsilon = epsilon;
        self
    }

    /// Compute pairwise Euclidean distances
    fn compute_distances<S>(&self, x: &ArrayBase<S, Ix2>) -> Array2<f64>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.shape()[0];
        let mut distances = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let mut dist = 0.0;
                for k in 0..x.shape()[1] {
                    let diff = num_traits::cast::<S::Elem, f64>(x[[i, k]]).unwrap_or(0.0)
                        - num_traits::cast::<S::Elem, f64>(x[[j, k]]).unwrap_or(0.0);
                    dist += diff * diff;
                }
                dist = dist.sqrt();
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        distances
    }

    /// Construct the neighborhood graph
    fn construct_graph(&self, distances: &Array2<f64>) -> Array2<f64> {
        let n_samples = distances.shape()[0];
        let mut graph = Array2::from_elem((n_samples, n_samples), f64::INFINITY);

        // Set diagonal to 0
        for i in 0..n_samples {
            graph[[i, i]] = 0.0;
        }

        if self.neighbor_mode == "knn" {
            // K-nearest neighbors graph
            for i in 0..n_samples {
                // Find k nearest neighbors
                let mut heap: BinaryHeap<(std::cmp::Reverse<i64>, usize)> = BinaryHeap::new();

                for j in 0..n_samples {
                    if i != j {
                        let dist_fixed = (distances[[i, j]] * 1e9) as i64;
                        heap.push((std::cmp::Reverse(dist_fixed), j));
                    }
                }

                // Connect to k nearest neighbors
                for _ in 0..self.n_neighbors {
                    if let Some((_, j)) = heap.pop() {
                        graph[[i, j]] = distances[[i, j]];
                        graph[[j, i]] = distances[[j, i]]; // Make symmetric
                    }
                }
            }
        } else {
            // Epsilon-ball graph
            for i in 0..n_samples {
                for j in i + 1..n_samples {
                    if distances[[i, j]] <= self.epsilon {
                        graph[[i, j]] = distances[[i, j]];
                        graph[[j, i]] = distances[[j, i]];
                    }
                }
            }
        }

        graph
    }

    /// Compute shortest paths using Floyd-Warshall algorithm
    fn compute_shortest_paths(&self, graph: &Array2<f64>) -> Result<Array2<f64>> {
        let n = graph.shape()[0];
        let mut dist = graph.clone();

        // Floyd-Warshall algorithm
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if dist[[i, k]] + dist[[k, j]] < dist[[i, j]] {
                        dist[[i, j]] = dist[[i, k]] + dist[[k, j]];
                    }
                }
            }
        }

        // Check if graph is connected
        for i in 0..n {
            for j in 0..n {
                if dist[[i, j]].is_infinite() {
                    return Err(TransformError::InvalidInput(
                        "Graph is not connected. Try increasing n_neighbors or epsilon.".to_string(),
                    ));
                }
            }
        }

        Ok(dist)
    }

    /// Apply classical MDS to the geodesic distance matrix
    fn classical_mds(&self, distances: &Array2<f64>) -> Result<Array2<f64>> {
        let n = distances.shape()[0];
        
        // Double center the squared distance matrix
        let mut squared_distances = distances.mapv(|d| d * d);
        
        // Row means
        let row_means = squared_distances.mean_axis(Axis(1)).unwrap();
        
        // Column means
        let col_means = squared_distances.mean_axis(Axis(0)).unwrap();
        
        // Grand mean
        let grand_mean = row_means.mean().unwrap();
        
        // Double centering
        let mut gram = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                gram[[i, j]] = -0.5 * (squared_distances[[i, j]] - row_means[i] - col_means[j] + grand_mean);
            }
        }
        
        // Eigendecomposition
        let (eigenvalues, eigenvectors) = match eigh(&gram.view()) {
            Ok(result) => result,
            Err(e) => return Err(TransformError::LinalgError(e)),
        };
        
        // Sort eigenvalues and eigenvectors in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());
        
        // Extract the top n_components eigenvectors
        let mut embedding = Array2::zeros((n, self.n_components));
        for j in 0..self.n_components {
            let idx = indices[j];
            let scale = eigenvalues[idx].max(0.0).sqrt();
            
            for i in 0..n {
                embedding[[i, j]] = eigenvectors[[i, idx]] * scale;
            }
        }
        
        Ok(embedding)
    }

    /// Fits the Isomap model to the input data
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
        let n_samples = x.shape()[0];
        
        if n_samples < self.n_neighbors {
            return Err(TransformError::InvalidInput(format!(
                "n_neighbors={} must be <= n_samples={}",
                self.n_neighbors, n_samples
            )));
        }
        
        if self.n_components >= n_samples {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} must be < n_samples={}",
                self.n_components, n_samples
            )));
        }
        
        // Convert input to f64
        let x_f64 = x.mapv(|v| num_traits::cast::<S::Elem, f64>(v).unwrap_or(0.0));
        
        // Step 1: Compute pairwise distances
        let distances = self.compute_distances(&x_f64.view());
        
        // Step 2: Construct neighborhood graph
        let graph = self.construct_graph(&distances);
        
        // Step 3: Compute shortest paths (geodesic distances)
        let geodesic_distances = self.compute_shortest_paths(&graph)?;
        
        // Step 4: Apply classical MDS
        let embedding = self.classical_mds(&geodesic_distances)?;
        
        self.embedding = Some(embedding);
        self.training_data = Some(x_f64);
        self.geodesic_distances = Some(geodesic_distances);
        
        Ok(())
    }

    /// Transforms the input data using the fitted Isomap model
    ///
    /// For new points, this uses the Landmark MDS approach
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data, shape (n_samples, n_components)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if self.embedding.is_none() {
            return Err(TransformError::TransformationError(
                "Isomap model has not been fitted".to_string(),
            ));
        }
        
        // For simplicity, just return the fitted embedding for training data
        // TODO: Implement proper out-of-sample extension using Landmark MDS
        if let Some(ref training_data) = self.training_data {
            let x_f64 = x.mapv(|v| num_traits::cast::<S::Elem, f64>(v).unwrap_or(0.0));
            
            // Check if this is the training data
            if x_f64.shape() == training_data.shape() {
                let mut is_same = true;
                for i in 0..x_f64.shape()[0] {
                    for j in 0..x_f64.shape()[1] {
                        if (x_f64[[i, j]] - training_data[[i, j]]).abs() > 1e-10 {
                            is_same = false;
                            break;
                        }
                    }
                    if !is_same {
                        break;
                    }
                }
                
                if is_same {
                    return Ok(self.embedding.as_ref().unwrap().clone());
                }
            }
        }
        
        // For new data, we'd need to implement Landmark MDS
        Err(TransformError::TransformationError(
            "Out-of-sample extension not yet implemented".to_string(),
        ))
    }

    /// Fits the Isomap model and transforms the data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data, shape (n_samples, n_components)
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the embedding
    pub fn embedding(&self) -> Option<&Array2<f64>> {
        self.embedding.as_ref()
    }

    /// Returns the geodesic distances computed during fitting
    pub fn geodesic_distances(&self) -> Option<&Array2<f64>> {
        self.geodesic_distances.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_isomap_basic() {
        // Create a simple S-curve dataset
        let n_points = 20;
        let mut data = Vec::new();
        
        for i in 0..n_points {
            let t = i as f64 / n_points as f64 * 3.0 * std::f64::consts::PI;
            let x = t.sin();
            let y = 2.0 * (i as f64 / n_points as f64);
            let z = t.cos();
            data.extend_from_slice(&[x, y, z]);
        }
        
        let x = Array::from_shape_vec((n_points, 3), data).unwrap();
        
        // Fit Isomap
        let mut isomap = Isomap::new(5, 2);
        let embedding = isomap.fit_transform(&x).unwrap();
        
        // Check shape
        assert_eq!(embedding.shape(), &[n_points, 2]);
        
        // Check that values are finite
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_isomap_epsilon_ball() {
        let x = Array::eye(5);
        
        let mut isomap = Isomap::new(3, 2).with_epsilon(1.5);
        let result = isomap.fit_transform(&x);
        
        // This should work as the identity matrix forms a connected graph with epsilon=1.5
        assert!(result.is_ok());
        
        let embedding = result.unwrap();
        assert_eq!(embedding.shape(), &[5, 2]);
    }

    #[test]
    fn test_isomap_disconnected_graph() {
        let x = Array::eye(5);
        
        // With only 1 neighbor and identity matrix, graph will be disconnected
        let mut isomap = Isomap::new(1, 2);
        let result = isomap.fit(&x);
        
        // Should fail due to disconnected graph
        assert!(result.is_err());
    }
}