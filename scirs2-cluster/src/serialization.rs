//! Serialization and deserialization support for clustering models
//!
//! This module provides functionality to save and load clustering models,
//! including support for various formats and model types.

use crate::error::{ClusteringError, Result};
use crate::leader::{LeaderNode, LeaderTree};
use ndarray::{Array1, Array2, ArrayView2};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Trait for clustering models that can be serialized
pub trait SerializableModel: Serialize + for<'de> Deserialize<'de> {
    /// Save the model to a file
    fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path).map_err(|e| {
            ClusteringError::InvalidInput(format!("Failed to create file: {}", e))
        })?;
        self.save_to_writer(file)
    }

    /// Save the model to a writer
    fn save_to_writer<W: Write>(&self, writer: W) -> Result<()> {
        serde_json::to_writer_pretty(writer, self).map_err(|e| {
            ClusteringError::InvalidInput(format!("Failed to serialize model: {}", e))
        })
    }

    /// Load the model from a file
    fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path).map_err(|e| {
            ClusteringError::InvalidInput(format!("Failed to open file: {}", e))
        })?;
        Self::load_from_reader(&mut file)
    }

    /// Load the model from a reader
    fn load_from_reader<R: Read>(reader: R) -> Result<Self> {
        serde_json::from_reader(reader).map_err(|e| {
            ClusteringError::InvalidInput(format!("Failed to deserialize model: {}", e))
        })
    }
}

/// K-means model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KMeansModel {
    /// Cluster centroids
    pub centroids: Array2<f64>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Sum of squared distances
    pub inertia: f64,
    /// Cluster labels for training data (optional)
    pub labels: Option<Array1<usize>>,
}

impl SerializableModel for KMeansModel {}

impl KMeansModel {
    /// Create a new K-means model
    pub fn new(
        centroids: Array2<f64>,
        n_clusters: usize,
        n_iter: usize,
        inertia: f64,
        labels: Option<Array1<usize>>,
    ) -> Self {
        Self {
            centroids,
            n_clusters,
            n_iter,
            inertia,
            labels,
        }
    }

    /// Predict cluster labels for new data
    pub fn predict(&self, data: ArrayView2<f64>) -> Result<Array1<usize>> {
        let n_samples = data.nrows();
        let mut labels = Array1::zeros(n_samples);
        
        for (i, sample) in data.rows().into_iter().enumerate() {
            let mut min_distance = f64::INFINITY;
            let mut closest_cluster = 0;
            
            for (j, centroid) in self.centroids.rows().into_iter().enumerate() {
                // Calculate Euclidean distance
                let distance = sample.iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                    
                if distance < min_distance {
                    min_distance = distance;
                    closest_cluster = j;
                }
            }
            
            labels[i] = closest_cluster;
        }
        
        Ok(labels)
    }
}

/// Hierarchical clustering result that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HierarchicalModel {
    /// Linkage matrix
    pub linkage: Array2<f64>,
    /// Number of original observations
    pub n_observations: usize,
    /// Method used for linkage
    pub method: String,
    /// Dendrogram labels (optional)
    pub labels: Option<Vec<String>>,
}

impl SerializableModel for HierarchicalModel {}

impl HierarchicalModel {
    /// Create a new hierarchical clustering model
    pub fn new(
        linkage: Array2<f64>,
        n_observations: usize,
        method: String,
        labels: Option<Vec<String>>,
    ) -> Self {
        Self {
            linkage,
            n_observations,
            method,
            labels,
        }
    }

    /// Export dendrogram to Newick format
    pub fn to_newick(&self) -> Result<String> {
        let mut newick = String::new();
        let n_nodes = self.linkage.nrows();
        
        if n_nodes == 0 {
            return Ok("();".to_string());
        }
        
        // Build the tree structure
        self.build_newick_recursive(
            n_nodes + self.n_observations - 1,
            &mut newick,
        )?;
        
        newick.push_str(";");
        Ok(newick)
    }

    fn build_newick_recursive(&self, node_idx: usize, newick: &mut String) -> Result<()> {
        if node_idx < self.n_observations {
            // Leaf node
            if let Some(ref labels) = self.labels {
                newick.push_str(&labels[node_idx]);
            } else {
                newick.push_str(&node_idx.to_string());
            }
        } else {
            // Internal node
            let row_idx = node_idx - self.n_observations;
            if row_idx >= self.linkage.nrows() {
                return Err(ClusteringError::InvalidInput(
                    "Invalid node index".to_string(),
                ));
            }
            
            let left = self.linkage[[row_idx, 0]] as usize;
            let right = self.linkage[[row_idx, 1]] as usize;
            let distance = self.linkage[[row_idx, 2]];
            
            newick.push('(');
            self.build_newick_recursive(left, newick)?;
            newick.push(':');
            newick.push_str(&format!("{:.6}", distance / 2.0));
            newick.push(',');
            self.build_newick_recursive(right, newick)?;
            newick.push(':');
            newick.push_str(&format!("{:.6}", distance / 2.0));
            newick.push(')');
        }
        
        Ok(())
    }

    /// Export dendrogram to JSON format
    pub fn to_json_tree(&self) -> Result<serde_json::Value> {
        use serde_json::json;
        
        let n_nodes = self.linkage.nrows();
        if n_nodes == 0 {
            return Ok(json!({}));
        }
        
        self.build_json_recursive(n_nodes + self.n_observations - 1)
    }

    fn build_json_recursive(&self, node_idx: usize) -> Result<serde_json::Value> {
        use serde_json::json;
        
        if node_idx < self.n_observations {
            // Leaf node
            let name = if let Some(ref labels) = self.labels {
                labels[node_idx].clone()
            } else {
                node_idx.to_string()
            };
            
            Ok(json!({
                "name": name,
                "type": "leaf",
                "index": node_idx
            }))
        } else {
            // Internal node
            let row_idx = node_idx - self.n_observations;
            if row_idx >= self.linkage.nrows() {
                return Err(ClusteringError::InvalidInput(
                    "Invalid node index".to_string(),
                ));
            }
            
            let left = self.linkage[[row_idx, 0]] as usize;
            let right = self.linkage[[row_idx, 1]] as usize;
            let distance = self.linkage[[row_idx, 2]];
            
            let left_child = self.build_json_recursive(left)?;
            let right_child = self.build_json_recursive(right)?;
            
            Ok(json!({
                "type": "internal",
                "distance": distance,
                "children": [left_child, right_child]
            }))
        }
    }
}

/// DBSCAN model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DBSCANModel {
    /// Core sample indices
    pub core_sample_indices: Array1<usize>,
    /// Cluster labels
    pub labels: Array1<i32>,
    /// Epsilon parameter
    pub eps: f64,
    /// Min samples parameter
    pub min_samples: usize,
}

impl SerializableModel for DBSCANModel {}

impl DBSCANModel {
    /// Create a new DBSCAN model
    pub fn new(
        core_sample_indices: Array1<usize>,
        labels: Array1<i32>,
        eps: f64,
        min_samples: usize,
    ) -> Self {
        Self {
            core_sample_indices,
            labels,
            eps,
            min_samples,
        }
    }
}

/// Mean Shift model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MeanShiftModel {
    /// Cluster centers
    pub cluster_centers: Array2<f64>,
    /// Bandwidth parameter
    pub bandwidth: f64,
    /// Cluster labels (optional)
    pub labels: Option<Array1<usize>>,
}

impl SerializableModel for MeanShiftModel {}

impl MeanShiftModel {
    /// Create a new Mean Shift model
    pub fn new(
        cluster_centers: Array2<f64>,
        bandwidth: f64,
        labels: Option<Array1<usize>>,
    ) -> Self {
        Self {
            cluster_centers,
            bandwidth,
            labels,
        }
    }
}

/// Leader algorithm model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LeaderModel {
    /// Cluster leaders
    pub leaders: Array2<f64>,
    /// Distance threshold
    pub threshold: f64,
    /// Cluster labels (optional)
    pub labels: Option<Array1<usize>>,
}

impl SerializableModel for LeaderModel {}

impl LeaderModel {
    /// Create a new Leader model
    pub fn new(
        leaders: Array2<f64>,
        threshold: f64,
        labels: Option<Array1<usize>>,
    ) -> Self {
        Self {
            leaders,
            threshold,
            labels,
        }
    }
    
    /// Predict cluster labels for new data using Euclidean distance
    pub fn predict(&self, data: ArrayView2<f64>) -> Result<Array1<usize>> {
        let n_samples = data.nrows();
        let mut labels = Array1::zeros(n_samples);
        
        for (i, sample) in data.rows().into_iter().enumerate() {
            let mut min_distance = f64::INFINITY;
            let mut closest_leader = 0;
            
            for (j, leader) in self.leaders.rows().into_iter().enumerate() {
                // Calculate Euclidean distance
                let distance = sample.iter()
                    .zip(leader.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                    
                if distance < min_distance {
                    min_distance = distance;
                    closest_leader = j;
                }
            }
            
            labels[i] = closest_leader;
        }
        
        Ok(labels)
    }
}

/// Leader tree model that can be serialized (hierarchical leader algorithm)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LeaderTreeModel {
    /// Root nodes of the tree
    pub roots: Vec<LeaderNodeModel>,
    /// Distance threshold for this level
    pub threshold: f64,
    /// Thresholds used for hierarchical clustering
    pub thresholds: Vec<f64>,
}

/// Affinity Propagation model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AffinityPropagationModel {
    /// Indices of exemplars (cluster centers)
    pub exemplars: Vec<usize>,
    /// Cluster labels
    pub labels: Array1<i32>,
    /// Damping factor used
    pub damping: f64,
    /// Preference value used
    pub preference: Option<f64>,
    /// Number of iterations performed
    pub n_iter: usize,
}

/// BIRCH model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BirchModel {
    /// Cluster centroids
    pub centroids: Array2<f64>,
    /// Cluster labels
    pub labels: Array1<i32>,
    /// Threshold parameter
    pub threshold: f64,
    /// Branching factor
    pub branching_factor: usize,
    /// Number of leaf clusters
    pub n_clusters: usize,
}

/// Gaussian Mixture Model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GMMModel {
    /// Component weights
    pub weights: Array1<f64>,
    /// Component means
    pub means: Array2<f64>,
    /// Component covariances
    pub covariances: Vec<Array2<f64>>,
    /// Cluster labels (optional, from prediction)
    pub labels: Option<Array1<i32>>,
    /// Number of components
    pub n_components: usize,
    /// Covariance type
    pub covariance_type: String,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Log-likelihood of the model
    pub log_likelihood: f64,
}

/// Spectral Clustering model that can be serialized
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SpectralClusteringModel {
    /// Spectral embeddings
    pub embeddings: Array2<f64>,
    /// Cluster labels
    pub labels: Array1<usize>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Affinity mode used
    pub affinity_mode: String,
    /// Gamma parameter (for RBF affinity)
    pub gamma: Option<f64>,
}

impl SerializableModel for AffinityPropagationModel {}

impl SerializableModel for BirchModel {}

impl SerializableModel for GMMModel {}

impl SerializableModel for SpectralClusteringModel {}

/// Serializable version of LeaderNode
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LeaderNodeModel {
    /// The leader vector
    pub leader: Array1<f64>,
    /// Child nodes
    pub children: Vec<LeaderNodeModel>,
    /// Indices of data points in this cluster
    pub members: Vec<usize>,
}

impl SerializableModel for LeaderTreeModel {}

impl LeaderTreeModel {
    /// Create a new Leader tree model
    pub fn new(roots: Vec<LeaderNodeModel>, threshold: f64, thresholds: Vec<f64>) -> Self {
        Self {
            roots,
            threshold,
            thresholds,
        }
    }
    
    /// Get the total number of nodes in the tree
    pub fn node_count(&self) -> usize {
        self.roots.iter().map(|root| Self::count_nodes(root)).sum()
    }
    
    fn count_nodes(node: &LeaderNodeModel) -> usize {
        1 + node.children.iter().map(|child| Self::count_nodes(child)).sum::<usize>()
    }
    
    /// Convert from LeaderTree
    pub fn from_leader_tree<F: num_traits::Float>(tree: &LeaderTree<F>) -> Self 
    where 
        f64: From<F>
    {
        let roots = tree.roots.iter()
            .map(|node| Self::convert_node(node))
            .collect();
            
        Self {
            roots,
            threshold: tree.threshold.to_f64().unwrap_or(0.0),
            thresholds: vec![tree.threshold.to_f64().unwrap_or(0.0)],
        }
    }
    
    fn convert_node<F: num_traits::Float>(node: &LeaderNode<F>) -> LeaderNodeModel 
    where 
        f64: From<F>
    {
        LeaderNodeModel {
            leader: node.leader.mapv(|x| x.to_f64().unwrap_or(0.0)),
            children: node.children.iter().map(|child| Self::convert_node(child)).collect(),
            members: node.members.clone(),
        }
    }
}

/// Convert K-means output to a serializable model
pub fn kmeans_to_model(
    centroids: Array2<f64>,
    labels: Array1<usize>,
    n_iter: usize,
) -> KMeansModel {
    // Calculate inertia (sum of squared distances)
    let inertia = calculate_inertia(&centroids, &labels);
    let n_clusters = centroids.nrows();
    
    KMeansModel::new(centroids, n_clusters, n_iter, inertia, Some(labels))
}

/// Convert hierarchical clustering output to a serializable model
pub fn hierarchy_to_model(
    linkage: Array2<f64>,
    n_observations: usize,
    method: &str,
    labels: Option<Vec<String>>,
) -> HierarchicalModel {
    HierarchicalModel::new(linkage, n_observations, method.to_string(), labels)
}

/// Convert DBSCAN output to a serializable model
pub fn dbscan_to_model(
    core_sample_indices: Array1<usize>,
    labels: Array1<i32>,
    eps: f64,
    min_samples: usize,
) -> DBSCANModel {
    DBSCANModel::new(core_sample_indices, labels, eps, min_samples)
}

/// Convert Mean Shift output to a serializable model
pub fn meanshift_to_model(
    cluster_centers: Array2<f64>,
    bandwidth: f64,
    labels: Option<Array1<usize>>,
) -> MeanShiftModel {
    MeanShiftModel::new(cluster_centers, bandwidth, labels)
}

/// Convert Leader algorithm output to a serializable model
pub fn leader_to_model(
    leaders: Array2<f64>,
    threshold: f64,
    labels: Option<Array1<usize>>,
) -> LeaderModel {
    LeaderModel::new(leaders, threshold, labels)
}

/// Convert Leader tree to a serializable model
pub fn leader_tree_to_model<F: num_traits::Float>(
    tree: &LeaderTree<F>,
) -> LeaderTreeModel 
where 
    f64: From<F>
{
    LeaderTreeModel::from_leader_tree(tree)
}

/// Convert Affinity Propagation output to a serializable model
pub fn affinity_propagation_to_model(
    exemplars: Vec<usize>,
    labels: Array1<i32>,
    damping: f64,
    preference: Option<f64>,
    n_iter: usize,
) -> AffinityPropagationModel {
    AffinityPropagationModel {
        exemplars,
        labels,
        damping,
        preference,
        n_iter,
    }
}

/// Convert BIRCH output to a serializable model
pub fn birch_to_model(
    centroids: Array2<f64>,
    labels: Array1<i32>,
    threshold: f64,
    branching_factor: usize,
) -> BirchModel {
    BirchModel {
        centroids: centroids.clone(),
        labels,
        threshold,
        branching_factor,
        n_clusters: centroids.nrows(),
    }
}

/// Convert GMM output to a serializable model
pub fn gmm_to_model(
    weights: Array1<f64>,
    means: Array2<f64>,
    covariances: Vec<Array2<f64>>,
    labels: Option<Array1<i32>>,
    covariance_type: String,
    n_iter: usize,
    log_likelihood: f64,
) -> GMMModel {
    GMMModel {
        weights,
        means,
        covariances,
        labels,
        n_components: weights.len(),
        covariance_type,
        n_iter,
        log_likelihood,
    }
}

/// Convert Spectral Clustering output to a serializable model
pub fn spectral_clustering_to_model(
    embeddings: Array2<f64>,
    labels: Array1<usize>,
    n_clusters: usize,
    affinity_mode: String,
    gamma: Option<f64>,
) -> SpectralClusteringModel {
    SpectralClusteringModel {
        embeddings,
        labels,
        n_clusters,
        affinity_mode,
        gamma,
    }
}

fn calculate_inertia(_centroids: &Array2<f64>, _labels: &Array1<usize>) -> f64 {
    // This is a placeholder - in practice, we'd need the original data
    // to calculate the actual inertia
    0.0
}

/// Convenience function to save K-means results directly
pub fn save_kmeans<P: AsRef<Path>>(
    path: P,
    centroids: Array2<f64>,
    labels: Array1<usize>,
    n_iter: usize,
) -> Result<()> {
    let model = kmeans_to_model(centroids, labels, n_iter);
    model.save_to_file(path)
}

/// Convenience function to save hierarchical clustering results directly
pub fn save_hierarchy<P: AsRef<Path>>(
    path: P,
    linkage: Array2<f64>,
    n_observations: usize,
    method: &str,
    labels: Option<Vec<String>>,
) -> Result<()> {
    let model = hierarchy_to_model(linkage, n_observations, method, labels);
    model.save_to_file(path)
}

/// Convenience function to save Leader algorithm results directly
pub fn save_leader<P: AsRef<Path>>(
    path: P,
    leaders: Array2<f64>,
    threshold: f64,
    labels: Option<Array1<usize>>,
) -> Result<()> {
    let model = leader_to_model(leaders, threshold, labels);
    model.save_to_file(path)
}

/// Convenience function to save Leader tree results directly
pub fn save_leader_tree<P: AsRef<Path>, F: num_traits::Float>(
    path: P,
    tree: &LeaderTree<F>,
) -> Result<()>
where 
    f64: From<F>
{
    let model = leader_tree_to_model(tree);
    model.save_to_file(path)
}

/// Convenience function to save Affinity Propagation results directly
pub fn save_affinity_propagation<P: AsRef<Path>>(
    path: P,
    exemplars: Vec<usize>,
    labels: Array1<i32>,
    damping: f64,
    preference: Option<f64>,
    n_iter: usize,
) -> Result<()> {
    let model = affinity_propagation_to_model(exemplars, labels, damping, preference, n_iter);
    model.save_to_file(path)
}

/// Convenience function to save BIRCH results directly
pub fn save_birch<P: AsRef<Path>>(
    path: P,
    centroids: Array2<f64>,
    labels: Array1<i32>,
    threshold: f64,
    branching_factor: usize,
) -> Result<()> {
    let model = birch_to_model(centroids, labels, threshold, branching_factor);
    model.save_to_file(path)
}

/// Convenience function to save GMM results directly
pub fn save_gmm<P: AsRef<Path>>(
    path: P,
    weights: Array1<f64>,
    means: Array2<f64>,
    covariances: Vec<Array2<f64>>,
    labels: Option<Array1<i32>>,
    covariance_type: String,
    n_iter: usize,
    log_likelihood: f64,
) -> Result<()> {
    let model = gmm_to_model(
        weights,
        means,
        covariances,
        labels,
        covariance_type,
        n_iter,
        log_likelihood,
    );
    model.save_to_file(path)
}

/// Convenience function to save Spectral Clustering results directly
pub fn save_spectral_clustering<P: AsRef<Path>>(
    path: P,
    embeddings: Array2<f64>,
    labels: Array1<usize>,
    n_clusters: usize,
    affinity_mode: String,
    gamma: Option<f64>,
) -> Result<()> {
    let model = spectral_clustering_to_model(embeddings, labels, n_clusters, affinity_mode, gamma);
    model.save_to_file(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tempfile::NamedTempFile;

    #[test]
    fn test_kmeans_serialization() {
        let centroids = array![[1.0, 2.0], [3.0, 4.0]];
        let labels = array![0, 1, 0, 1];
        let model = KMeansModel::new(centroids.clone(), 2, 10, 5.0, Some(labels.clone()));

        // Test JSON serialization
        let json = serde_json::to_string(&model).unwrap();
        let deserialized: KMeansModel = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.n_clusters, 2);
        assert_eq!(deserialized.n_iter, 10);
        assert_eq!(deserialized.inertia, 5.0);
        assert_eq!(deserialized.centroids, centroids);
        assert_eq!(deserialized.labels, Some(labels));
    }

    #[test]
    fn test_kmeans_save_load_file() {
        let centroids = array![[1.0, 2.0], [3.0, 4.0]];
        let model = KMeansModel::new(centroids.clone(), 2, 10, 5.0, None);

        let temp_file = NamedTempFile::new().unwrap();
        model.save_to_file(temp_file.path()).unwrap();
        
        let loaded_model = KMeansModel::load_from_file(temp_file.path()).unwrap();
        assert_eq!(loaded_model.n_clusters, model.n_clusters);
        assert_eq!(loaded_model.centroids, model.centroids);
    }

    #[test]
    fn test_hierarchical_to_newick() {
        // Simple linkage matrix
        let linkage = array![
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 3.0, 2.0, 2.0],
            [4.0, 5.0, 3.0, 4.0]
        ];
        
        let model = HierarchicalModel::new(linkage, 4, "single".to_string(), None);
        let newick = model.to_newick().unwrap();
        
        assert!(newick.ends_with(';'));
        assert!(newick.contains('('));
        assert!(newick.contains(')'));
    }

    #[test]
    fn test_hierarchical_to_json() {
        let linkage = array![
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 3.0, 2.0, 2.0]
        ];
        
        let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let model = HierarchicalModel::new(linkage, 3, "average".to_string(), Some(labels));
        
        let json_tree = model.to_json_tree().unwrap();
        
        assert!(json_tree.is_object());
        assert_eq!(json_tree["type"], "internal");
        assert!(json_tree["children"].is_array());
    }

    #[test]
    fn test_kmeans_predict() {
        let centroids = array![[0.0, 0.0], [5.0, 5.0]];
        let model = KMeansModel::new(centroids, 2, 10, 5.0, None);
        
        let test_data = array![[1.0, 1.0], [4.0, 4.0], [0.5, 0.5], [5.5, 5.5]];
        let predictions = model.predict(test_data.view()).unwrap();
        
        assert_eq!(predictions, array![0, 1, 0, 1]);
    }
}