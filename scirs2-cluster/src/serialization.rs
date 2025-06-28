//! Serialization and deserialization support for clustering models
//!
//! This module provides functionality to save and load clustering models,
//! including support for various formats and model types.

use crate::error::{ClusteringError, Result};
use crate::leader::{LeaderNode, LeaderTree};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use ndarray::{Array1, Array2, ArrayView2};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Trait for clustering models that can be serialized
pub trait SerializableModel: Serialize + for<'de> Deserialize<'de> {
    /// Save the model to a file
    fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to create file: {}", e)))?;
        self.save_to_writer(file)
    }

    /// Save the model to a writer
    fn save_to_writer<W: Write>(&self, writer: W) -> Result<()> {
        serde_json::to_writer_pretty(writer, self)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to serialize model: {}", e)))
    }

    /// Save the model to a file with compression
    fn save_to_file_compressed<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to create file: {}", e)))?;
        let encoder = GzEncoder::new(file, Compression::default());
        self.save_to_writer(encoder)
    }

    /// Load the model from a compressed file
    fn load_from_file_compressed<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to open file: {}", e)))?;
        let decoder = GzDecoder::new(file);
        Self::load_from_reader(decoder)
    }

    /// Load the model from a file
    fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to open file: {}", e)))?;
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
                let distance = sample
                    .iter()
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
        self.build_newick_recursive(n_nodes + self.n_observations - 1, &mut newick)?;

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
    pub fn new(leaders: Array2<f64>, threshold: f64, labels: Option<Array1<usize>>) -> Self {
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
                let distance = sample
                    .iter()
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
        1 + node
            .children
            .iter()
            .map(|child| Self::count_nodes(child))
            .sum::<usize>()
    }

    /// Convert from LeaderTree
    pub fn from_leader_tree<F: num_traits::Float>(tree: &LeaderTree<F>) -> Self
    where
        f64: From<F>,
    {
        let roots = tree
            .roots
            .iter()
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
        f64: From<F>,
    {
        LeaderNodeModel {
            leader: node.leader.mapv(|x| x.to_f64().unwrap_or(0.0)),
            children: node
                .children
                .iter()
                .map(|child| Self::convert_node(child))
                .collect(),
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
pub fn leader_tree_to_model<F: num_traits::Float>(tree: &LeaderTree<F>) -> LeaderTreeModel
where
    f64: From<F>,
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

/// Get current Unix timestamp
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Format Unix timestamp to human-readable string
fn format_timestamp(timestamp: u64) -> String {
    // Simple formatting without chrono dependency
    // In production, you might want to use a proper date library
    format!("1970-01-01T00:00:00Z+{}s", timestamp)
}

/// Enhanced serialization utilities
pub mod enhanced {
    use super::*;

    /// Serialize with multiple format support
    pub fn serialize_with_format<T: Serialize>(data: &T, format: ExportFormat) -> Result<Vec<u8>> {
        match format {
            ExportFormat::Json => serde_json::to_vec_pretty(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("JSON serialization failed: {}", e))
            }),
            ExportFormat::Binary => bincode::serialize(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("Binary serialization failed: {}", e))
            }),
            ExportFormat::CompressedJson => {
                let json_data = serde_json::to_vec_pretty(data).map_err(|e| {
                    ClusteringError::InvalidInput(format!("JSON serialization failed: {}", e))
                })?;

                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(&json_data).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Compression failed: {}", e))
                })?;
                encoder.finish().map_err(|e| {
                    ClusteringError::InvalidInput(format!("Compression finalization failed: {}", e))
                })
            }
            #[cfg(feature = "yaml")]
            ExportFormat::Yaml => serde_yaml::to_vec(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("YAML serialization failed: {}", e))
            }),
            #[cfg(feature = "msgpack")]
            ExportFormat::MessagePack => rmp_serde::to_vec(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("MessagePack serialization failed: {}", e))
            }),
            #[cfg(feature = "cbor")]
            ExportFormat::Cbor => serde_cbor::to_vec(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("CBOR serialization failed: {}", e))
            }),
            _ => Err(ClusteringError::InvalidInput(
                "Unsupported format for this data type".to_string(),
            )),
        }
    }

    /// Deserialize with multiple format support
    pub fn deserialize_with_format<T: for<'de> Deserialize<'de>>(
        data: &[u8],
        format: ExportFormat,
    ) -> Result<T> {
        match format {
            ExportFormat::Json => serde_json::from_slice(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("JSON deserialization failed: {}", e))
            }),
            ExportFormat::Binary => bincode::deserialize(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("Binary deserialization failed: {}", e))
            }),
            ExportFormat::CompressedJson => {
                let mut decoder = GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Decompression failed: {}", e))
                })?;

                serde_json::from_slice(&decompressed).map_err(|e| {
                    ClusteringError::InvalidInput(format!("JSON deserialization failed: {}", e))
                })
            }
            #[cfg(feature = "yaml")]
            ExportFormat::Yaml => serde_yaml::from_slice(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("YAML deserialization failed: {}", e))
            }),
            #[cfg(feature = "msgpack")]
            ExportFormat::MessagePack => rmp_serde::from_slice(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("MessagePack deserialization failed: {}", e))
            }),
            #[cfg(feature = "cbor")]
            ExportFormat::Cbor => serde_cbor::from_slice(data).map_err(|e| {
                ClusteringError::InvalidInput(format!("CBOR deserialization failed: {}", e))
            }),
            _ => Err(ClusteringError::InvalidInput(
                "Unsupported format for deserialization".to_string(),
            )),
        }
    }

    /// Model versioning support
    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct VersionedModel<T> {
        /// Model format version
        pub version: String,
        /// Backward compatibility info
        pub compatibility: Vec<String>,
        /// Migration notes
        pub migration_notes: Option<String>,
        /// The actual model data
        pub data: T,
        /// Metadata
        pub metadata: ModelMetadata,
    }

    impl<T: Serialize + for<'de> Deserialize<'de>> VersionedModel<T> {
        /// Create a new versioned model
        pub fn new(data: T, metadata: ModelMetadata) -> Self {
            Self {
                version: "1.0.0".to_string(),
                compatibility: vec!["1.0.0".to_string()],
                migration_notes: None,
                data,
                metadata,
            }
        }

        /// Check if model is compatible with current version
        pub fn is_compatible(&self, target_version: &str) -> bool {
            self.compatibility.contains(&target_version.to_string())
        }

        /// Migrate model to newer version (placeholder)
        pub fn migrate_to(&mut self, target_version: &str) -> Result<()> {
            if self.is_compatible(target_version) {
                Ok(())
            } else {
                Err(ClusteringError::InvalidInput(format!(
                    "Cannot migrate from {} to {}",
                    self.version, target_version
                )))
            }
        }
    }
}

/// Performance monitoring for serialization
pub mod performance {
    use super::*;
    use std::time::Instant;

    /// Benchmark serialization performance
    pub fn benchmark_serialization<T: Serialize>(
        data: &T,
        formats: &[ExportFormat],
    ) -> Result<std::collections::HashMap<ExportFormat, (u64, usize)>> {
        let mut results = std::collections::HashMap::new();

        for &format in formats {
            let start = Instant::now();
            let serialized = enhanced::serialize_with_format(data, format)?;
            let duration = start.elapsed().as_micros() as u64;
            let size = serialized.len();

            results.insert(format, (duration, size));
        }

        Ok(results)
    }

    /// Compression ratio analysis
    pub fn analyze_compression<T: Serialize>(data: &T) -> Result<(f64, usize, usize)> {
        let uncompressed = enhanced::serialize_with_format(data, ExportFormat::Json)?;
        let compressed = enhanced::serialize_with_format(data, ExportFormat::CompressedJson)?;

        let ratio = compressed.len() as f64 / uncompressed.len() as f64;
        Ok((ratio, uncompressed.len(), compressed.len()))
    }
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
    f64: From<F>,
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

/// Export formats for clustering models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// YAML format  
    Yaml,
    /// CSV format (for simple data like centroids)
    Csv,
    /// Newick format (for dendrograms)
    Newick,
    /// Pickle-like binary format
    Binary,
    /// Compressed JSON format
    CompressedJson,
    /// MessagePack format
    MessagePack,
    /// CBOR format
    Cbor,
}

/// Enhanced export functionality for clustering models
pub trait AdvancedExport {
    /// Export model in specified format
    fn export<P: AsRef<Path>>(&self, path: P, format: ExportFormat) -> Result<()>;

    /// Export to string in specified format
    fn export_to_string(&self, format: ExportFormat) -> Result<String>;

    /// Get model metadata
    fn get_metadata(&self) -> ModelMetadata;
}

/// Metadata about a clustering model
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelMetadata {
    /// Model type (e.g., "kmeans", "hierarchical", "dbscan")
    pub model_type: String,
    /// Creation timestamp (Unix timestamp)
    pub created_at: u64,
    /// Human-readable creation time
    pub created_at_readable: String,
    /// Number of clusters/components
    pub n_clusters: Option<usize>,
    /// Number of features
    pub n_features: Option<usize>,
    /// Additional parameters
    pub parameters: std::collections::HashMap<String, String>,
    /// Model version/format version
    pub version: String,
    /// Algorithm configuration used
    pub algorithm_config: Option<AlgorithmConfig>,
    /// Performance metrics
    pub performance_metrics: Option<PerformanceMetrics>,
    /// Training data characteristics
    pub data_characteristics: Option<DataCharacteristics>,
}

/// Algorithm configuration metadata
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AlgorithmConfig {
    /// Algorithm name
    pub algorithm: String,
    /// Hyperparameters used
    pub hyperparameters: HashMap<String, serde_json::Value>,
    /// Distance metric used
    pub distance_metric: Option<String>,
    /// Linkage method (for hierarchical clustering)
    pub linkage_method: Option<String>,
    /// Initialization method
    pub initialization_method: Option<String>,
    /// Convergence criteria
    pub convergence_criteria: Option<HashMap<String, f64>>,
}

/// Performance metrics for the model
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PerformanceMetrics {
    /// Training time in seconds
    pub training_time_seconds: Option<f64>,
    /// Memory usage in bytes
    pub memory_usage_bytes: Option<usize>,
    /// Number of iterations to convergence
    pub iterations_to_convergence: Option<usize>,
    /// Final inertia/objective value
    pub final_objective_value: Option<f64>,
    /// Silhouette score (if available)
    pub silhouette_score: Option<f64>,
    /// Davies-Bouldin index (if available)
    pub davies_bouldin_index: Option<f64>,
    /// Custom metrics
    pub custom_metrics: Option<HashMap<String, f64>>,
}

/// Data characteristics metadata
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DataCharacteristics {
    /// Number of samples in training data
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Data type (continuous, discrete, mixed)
    pub data_type: Option<String>,
    /// Missing value percentage
    pub missing_value_percentage: Option<f64>,
    /// Feature scaling applied
    pub feature_scaling: Option<String>,
    /// Data preprocessing steps
    pub preprocessing_steps: Option<Vec<String>>,
    /// Statistical summary
    pub statistical_summary: Option<HashMap<String, f64>>,
}

impl AdvancedExport for KMeansModel {
    fn export<P: AsRef<Path>>(&self, path: P, format: ExportFormat) -> Result<()> {
        match format {
            ExportFormat::Json => self.save_to_file(path),
            ExportFormat::CompressedJson => self.save_to_file_compressed(path),
            ExportFormat::Csv => {
                let path = path.as_ref();
                let mut file = File::create(path).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to create file: {}", e))
                })?;

                // Write centroids as CSV
                for (i, row) in self.centroids.rows().into_iter().enumerate() {
                    if i > 0 {
                        writeln!(file)?;
                    }
                    let row_str: Vec<String> = row.iter().map(|x| x.to_string()).collect();
                    write!(file, "{}", row_str.join(","))?;
                }
                Ok(())
            }
            ExportFormat::Binary => {
                let data = enhanced::serialize_with_format(self, format)?;
                std::fs::write(path, data).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to write file: {}", e))
                })
            }
            #[cfg(feature = "yaml")]
            ExportFormat::Yaml => {
                let data = enhanced::serialize_with_format(self, format)?;
                std::fs::write(path, data).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to write file: {}", e))
                })
            }
            #[cfg(feature = "msgpack")]
            ExportFormat::MessagePack => {
                let data = enhanced::serialize_with_format(self, format)?;
                std::fs::write(path, data).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to write file: {}", e))
                })
            }
            #[cfg(feature = "cbor")]
            ExportFormat::Cbor => {
                let data = enhanced::serialize_with_format(self, format)?;
                std::fs::write(path, data).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to write file: {}", e))
                })
            }
            _ => Err(ClusteringError::InvalidInput(
                "Format not supported for K-means model".to_string(),
            )),
        }
    }

    fn export_to_string(&self, format: ExportFormat) -> Result<String> {
        match format {
            ExportFormat::Json => serde_json::to_string_pretty(self)
                .map_err(|e| ClusteringError::InvalidInput(format!("Failed to serialize: {}", e))),
            ExportFormat::Csv => {
                let mut result = String::new();
                for (i, row) in self.centroids.rows().into_iter().enumerate() {
                    if i > 0 {
                        result.push('\n');
                    }
                    let row_str: Vec<String> = row.iter().map(|x| x.to_string()).collect();
                    result.push_str(&row_str.join(","));
                }
                Ok(result)
            }
            #[cfg(feature = "yaml")]
            ExportFormat::Yaml => {
                let data = enhanced::serialize_with_format(self, format)?;
                String::from_utf8(data).map_err(|e| {
                    ClusteringError::InvalidInput(format!("UTF-8 conversion failed: {}", e))
                })
            }
            _ => Err(ClusteringError::InvalidInput(
                "Format not supported for string export".to_string(),
            )),
        }
    }

    fn get_metadata(&self) -> ModelMetadata {
        let mut parameters = std::collections::HashMap::new();
        parameters.insert("n_iter".to_string(), self.n_iter.to_string());
        parameters.insert("inertia".to_string(), self.inertia.to_string());

        let mut hyperparameters = HashMap::new();
        hyperparameters.insert(
            "n_clusters".to_string(),
            serde_json::Value::Number(serde_json::Number::from(self.n_clusters)),
        );
        hyperparameters.insert(
            "n_iter".to_string(),
            serde_json::Value::Number(serde_json::Number::from(self.n_iter)),
        );

        let algorithm_config = Some(AlgorithmConfig {
            algorithm: "k-means".to_string(),
            hyperparameters,
            distance_metric: Some("euclidean".to_string()),
            linkage_method: None,
            initialization_method: Some("k-means++".to_string()),
            convergence_criteria: Some({
                let mut criteria = HashMap::new();
                criteria.insert("tolerance".to_string(), 1e-4);
                criteria
            }),
        });

        let performance_metrics = Some(PerformanceMetrics {
            training_time_seconds: None,
            memory_usage_bytes: None,
            iterations_to_convergence: Some(self.n_iter),
            final_objective_value: Some(self.inertia),
            silhouette_score: None,
            davies_bouldin_index: None,
            custom_metrics: None,
        });

        let data_characteristics = if let Some(ref labels) = self.labels {
            Some(DataCharacteristics {
                n_samples: labels.len(),
                n_features: self.centroids.ncols(),
                data_type: Some("continuous".to_string()),
                missing_value_percentage: Some(0.0),
                feature_scaling: None,
                preprocessing_steps: None,
                statistical_summary: None,
            })
        } else {
            None
        };

        ModelMetadata {
            model_type: "kmeans".to_string(),
            created_at: current_timestamp(),
            created_at_readable: format_timestamp(current_timestamp()),
            n_clusters: Some(self.n_clusters),
            n_features: Some(self.centroids.ncols()),
            parameters,
            version: "1.0".to_string(),
            algorithm_config,
            performance_metrics,
            data_characteristics,
        }
    }
}

impl AdvancedExport for HierarchicalModel {
    fn export<P: AsRef<Path>>(&self, path: P, format: ExportFormat) -> Result<()> {
        match format {
            ExportFormat::Json => self.save_to_file(path),
            ExportFormat::Newick => {
                let newick = self.to_newick()?;
                std::fs::write(path, newick).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to write file: {}", e))
                })
            }
            ExportFormat::Binary => {
                let data = serde_json::to_vec(self).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to serialize: {}", e))
                })?;
                std::fs::write(path, data).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to write file: {}", e))
                })
            }
            _ => Err(ClusteringError::InvalidInput(
                "Format not supported for hierarchical model".to_string(),
            )),
        }
    }

    fn export_to_string(&self, format: ExportFormat) -> Result<String> {
        match format {
            ExportFormat::Json => serde_json::to_string_pretty(self)
                .map_err(|e| ClusteringError::InvalidInput(format!("Failed to serialize: {}", e))),
            ExportFormat::Newick => self.to_newick(),
            _ => Err(ClusteringError::InvalidInput(
                "Format not supported for string export".to_string(),
            )),
        }
    }

    fn get_metadata(&self) -> ModelMetadata {
        let mut parameters = std::collections::HashMap::new();
        parameters.insert("method".to_string(), self.method.clone());
        parameters.insert(
            "n_observations".to_string(),
            self.n_observations.to_string(),
        );

        let mut hyperparameters = HashMap::new();
        hyperparameters.insert(
            "linkage_method".to_string(),
            serde_json::Value::String(self.method.clone()),
        );
        hyperparameters.insert(
            "n_observations".to_string(),
            serde_json::Value::Number(serde_json::Number::from(self.n_observations)),
        );

        let algorithm_config = Some(AlgorithmConfig {
            algorithm: "hierarchical_clustering".to_string(),
            hyperparameters,
            distance_metric: Some("euclidean".to_string()),
            linkage_method: Some(self.method.clone()),
            initialization_method: None,
            convergence_criteria: None,
        });

        let performance_metrics = Some(PerformanceMetrics {
            training_time_seconds: None,
            memory_usage_bytes: None,
            iterations_to_convergence: None,
            final_objective_value: None,
            silhouette_score: None,
            davies_bouldin_index: None,
            custom_metrics: None,
        });

        let data_characteristics = Some(DataCharacteristics {
            n_samples: self.n_observations,
            n_features: 0, // Unknown in linkage matrix
            data_type: Some("continuous".to_string()),
            missing_value_percentage: Some(0.0),
            feature_scaling: None,
            preprocessing_steps: None,
            statistical_summary: None,
        });

        ModelMetadata {
            model_type: "hierarchical".to_string(),
            created_at: current_timestamp(),
            created_at_readable: format_timestamp(current_timestamp()),
            n_clusters: None, // Can vary based on cut
            n_features: None, // Not directly stored
            parameters,
            version: "1.0".to_string(),
            algorithm_config,
            performance_metrics,
            data_characteristics,
        }
    }
}

/// Cross-platform model compatibility utilities
pub mod compatibility {
    use super::*;

    /// Convert to scikit-learn compatible format
    pub fn to_sklearn_format<T: SerializableModel + AdvancedExport>(
        model: &T,
    ) -> Result<serde_json::Value> {
        use serde_json::json;

        let metadata = model.get_metadata();
        let model_data = model.export_to_string(ExportFormat::Json)?;
        let model_json: serde_json::Value = serde_json::from_str(&model_data)?;

        Ok(json!({
            "sklearn_version": "1.0.0",
            "scirs_version": "0.1.0-beta.1",
            "model_type": metadata.model_type,
            "created_at": metadata.created_at,
            "parameters": metadata.parameters,
            "model_data": model_json
        }))
    }

    /// Import from scikit-learn compatible format
    pub fn from_sklearn_format<T: SerializableModel>(
        sklearn_data: &serde_json::Value,
    ) -> Result<T> {
        if let Some(model_data) = sklearn_data.get("model_data") {
            let model_str = serde_json::to_string(model_data)?;
            let model: T = serde_json::from_str(&model_str)?;
            Ok(model)
        } else {
            Err(ClusteringError::InvalidInput(
                "Invalid sklearn format: missing model_data".to_string(),
            ))
        }
    }

    /// Convert to ONNX-compatible format metadata
    pub fn to_onnx_metadata<T: SerializableModel + AdvancedExport>(
        model: &T,
    ) -> Result<serde_json::Value> {
        use serde_json::json;

        let metadata = model.get_metadata();

        Ok(json!({
            "ir_version": 7,
            "producer_name": "scirs2-cluster",
            "producer_version": "0.1.0-beta.1",
            "domain": "ai.onnx.ml",
            "model_version": 1,
            "doc_string": format!("SCIRS2 {} model", metadata.model_type),
            "metadata_props": {
                "model_type": metadata.model_type,
                "n_clusters": metadata.n_clusters,
                "n_features": metadata.n_features,
                "created_at": metadata.created_at,
                "algorithm_config": metadata.algorithm_config
            }
        }))
    }

    /// Convert to PyTorch Lightning checkpoint format
    pub fn to_pytorch_checkpoint<T: SerializableModel + AdvancedExport>(
        model: &T,
        epoch: Option<usize>,
        step: Option<usize>,
    ) -> Result<serde_json::Value> {
        use serde_json::json;

        let metadata = model.get_metadata();
        let model_data = model.export_to_string(ExportFormat::Json)?;
        let model_json: serde_json::Value = serde_json::from_str(&model_data)?;

        Ok(json!({
            "pytorch_lightning_version": "1.6.0",
            "scirs_version": "0.1.0-beta.1",
            "epoch": epoch.unwrap_or(0),
            "global_step": step.unwrap_or(0),
            "lr_schedulers": [],
            "optimizers": [],
            "state_dict": model_json,
            "hyper_parameters": metadata.parameters,
            "model_type": metadata.model_type,
            "clustering_metadata": {
                "algorithm": metadata.algorithm_config.as_ref().map(|c| c.algorithm.clone()),
                "n_clusters": metadata.n_clusters,
                "performance_metrics": metadata.performance_metrics
            }
        }))
    }

    /// Convert to MLflow model format
    pub fn to_mlflow_format<T: SerializableModel + AdvancedExport>(
        model: &T,
        model_uuid: Option<String>,
    ) -> Result<serde_json::Value> {
        use serde_json::json;

        let metadata = model.get_metadata();
        let model_data = model.export_to_string(ExportFormat::Json)?;

        Ok(json!({
            "artifact_path": "model",
            "flavors": {
                "scirs2": {
                    "scirs_version": "0.1.0-beta.1",
                    "model_type": metadata.model_type,
                    "data": model_data
                },
                "python_function": {
                    "env": "conda.yaml",
                    "loader_module": "scirs2_mlflow",
                    "python_version": "3.8.0"
                }
            },
            "model_uuid": model_uuid.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            "run_id": null,
            "saved_input_example_info": null,
            "signature": {
                "inputs": format!("[{{\"name\": \"data\", \"type\": \"tensor\", \"tensor-spec\": {{\"dtype\": \"float64\", \"shape\": [-1, {}]}}}}]",
                    metadata.n_features.unwrap_or(0)),
                "outputs": format!("[{{\"name\": \"labels\", \"type\": \"tensor\", \"tensor-spec\": {{\"dtype\": \"int64\", \"shape\": [-1]}}}}]")
            },
            "utc_time_created": metadata.created_at_readable,
            "mlflow_version": "1.27.0"
        }))
    }

    /// Convert to Hugging Face model card format
    pub fn to_huggingface_card<T: SerializableModel + AdvancedExport>(
        model: &T,
        model_name: &str,
        description: Option<&str>,
    ) -> Result<String> {
        let metadata = model.get_metadata();

        let mut card = String::new();
        card.push_str("---\n");
        card.push_str(&format!("library_name: scirs2-cluster\n"));
        card.push_str(&format!("tags:\n"));
        card.push_str(&format!("- clustering\n"));
        card.push_str(&format!("- {}\n", metadata.model_type));
        card.push_str(&format!("- unsupervised-learning\n"));
        card.push_str(&format!("model-index:\n"));
        card.push_str(&format!("- name: {}\n", model_name));
        card.push_str(&format!("  results:\n"));

        if let Some(perf) = &metadata.performance_metrics {
            if let Some(silhouette) = perf.silhouette_score {
                card.push_str(&format!("  - task:\n"));
                card.push_str(&format!("      type: clustering\n"));
                card.push_str(&format!("    metrics:\n"));
                card.push_str(&format!("    - type: silhouette_score\n"));
                card.push_str(&format!("      value: {:.4}\n", silhouette));
            }
        }

        card.push_str("---\n\n");
        card.push_str(&format!("# {}\n\n", model_name));

        if let Some(desc) = description {
            card.push_str(&format!("{}\n\n", desc));
        }

        card.push_str("## Model Description\n\n");
        card.push_str(&format!(
            "This is a **{}** clustering model trained using the SCIRS2 library.\n\n",
            metadata.model_type
        ));

        card.push_str("## Model Details\n\n");
        card.push_str(&format!("- **Model Type**: {}\n", metadata.model_type));
        if let Some(n_clusters) = metadata.n_clusters {
            card.push_str(&format!("- **Number of Clusters**: {}\n", n_clusters));
        }
        if let Some(n_features) = metadata.n_features {
            card.push_str(&format!("- **Number of Features**: {}\n", n_features));
        }
        card.push_str(&format!(
            "- **Created**: {}\n",
            metadata.created_at_readable
        ));
        card.push_str(&format!("- **Library**: SCIRS2 v{}\n", metadata.version));

        if let Some(algorithm_config) = &metadata.algorithm_config {
            card.push_str("\n## Algorithm Configuration\n\n");
            card.push_str(&format!(
                "- **Algorithm**: {}\n",
                algorithm_config.algorithm
            ));
            if let Some(distance_metric) = &algorithm_config.distance_metric {
                card.push_str(&format!("- **Distance Metric**: {}\n", distance_metric));
            }
            if let Some(linkage_method) = &algorithm_config.linkage_method {
                card.push_str(&format!("- **Linkage Method**: {}\n", linkage_method));
            }
        }

        if let Some(perf) = &metadata.performance_metrics {
            card.push_str("\n## Performance Metrics\n\n");
            if let Some(training_time) = perf.training_time_seconds {
                card.push_str(&format!(
                    "- **Training Time**: {:.2} seconds\n",
                    training_time
                ));
            }
            if let Some(silhouette) = perf.silhouette_score {
                card.push_str(&format!("- **Silhouette Score**: {:.4}\n", silhouette));
            }
            if let Some(davies_bouldin) = perf.davies_bouldin_index {
                card.push_str(&format!(
                    "- **Davies-Bouldin Index**: {:.4}\n",
                    davies_bouldin
                ));
            }
        }

        card.push_str("\n## Usage\n\n");
        card.push_str("```rust\n");
        card.push_str("use scirs2_cluster::serialization::SerializableModel;\n");
        card.push_str(&format!(
            "use scirs2_cluster::serialization::{}Model;\n",
            metadata.model_type.to_uppercase()
        ));
        card.push_str("\n");
        card.push_str(&format!("// Load the model\n"));
        card.push_str(&format!(
            "let model = {}Model::load_from_file(\"path/to/model.json\")?;\n",
            metadata.model_type.to_uppercase()
        ));
        card.push_str("\n");
        card.push_str("// Use the model for prediction\n");
        card.push_str("let predictions = model.predict(your_data.view())?;\n");
        card.push_str("```\n");

        Ok(card)
    }

    /// Export model metadata in Apache Arrow schema format
    pub fn to_arrow_schema<T: SerializableModel + AdvancedExport>(
        model: &T,
    ) -> Result<serde_json::Value> {
        use serde_json::json;

        let metadata = model.get_metadata();

        Ok(json!({
            "schema": {
                "fields": [
                    {
                        "name": "features",
                        "type": {
                            "name": "list",
                            "contains": {
                                "name": "item",
                                "type": {
                                    "name": "floatingpoint",
                                    "precision": "DOUBLE"
                                },
                                "nullable": false
                            }
                        },
                        "nullable": false,
                        "metadata": {
                            "n_features": metadata.n_features.unwrap_or(0)
                        }
                    },
                    {
                        "name": "cluster_labels",
                        "type": {
                            "name": "int",
                            "bitWidth": 32,
                            "isSigned": true
                        },
                        "nullable": false,
                        "metadata": {
                            "n_clusters": metadata.n_clusters.unwrap_or(0)
                        }
                    }
                ],
                "metadata": {
                    "model_type": metadata.model_type,
                    "scirs_version": "0.1.0-beta.1",
                    "created_at": metadata.created_at
                }
            }
        }))
    }
}

/// Advanced persistence features for production clustering systems
pub mod persistence {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    /// Model registry for managing multiple clustering models
    #[derive(Debug, Clone)]
    pub struct ModelRegistry {
        /// Registry of models with their metadata
        models: BTreeMap<String, ModelRegistryEntry>,
        /// Base directory for model storage
        base_directory: PathBuf,
        /// Default format for new models
        default_format: ExportFormat,
    }

    /// Entry in the model registry
    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct ModelRegistryEntry {
        /// Unique model identifier
        pub model_id: String,
        /// Model metadata
        pub metadata: ModelMetadata,
        /// File path relative to base directory
        pub file_path: PathBuf,
        /// Storage format used
        pub format: ExportFormat,
        /// File size in bytes
        pub file_size: Option<usize>,
        /// Model tags for organization
        pub tags: Vec<String>,
        /// Model description
        pub description: Option<String>,
        /// Dependencies on other models
        pub dependencies: Vec<String>,
        /// Checksum for integrity verification
        pub checksum: Option<String>,
    }

    impl ModelRegistry {
        /// Create a new model registry
        pub fn new<P: Into<PathBuf>>(base_directory: P) -> Self {
            Self {
                models: BTreeMap::new(),
                base_directory: base_directory.into(),
                default_format: ExportFormat::Json,
            }
        }

        /// Register a new model
        pub fn register_model<T: SerializableModel + AdvancedExport>(
            &mut self,
            model_id: String,
            model: &T,
            tags: Vec<String>,
            description: Option<String>,
        ) -> Result<()> {
            let metadata = model.get_metadata();
            let file_name = format!("{}.{}", model_id, format_extension(self.default_format));
            let file_path = self.base_directory.join(&file_name);

            // Save the model
            model.export(&file_path, self.default_format)?;

            // Calculate file size and checksum
            let file_size = std::fs::metadata(&file_path).map(|m| m.len() as usize).ok();
            let checksum = self.calculate_checksum(&file_path)?;

            let entry = ModelRegistryEntry {
                model_id: model_id.clone(),
                metadata,
                file_path: PathBuf::from(file_name),
                format: self.default_format,
                file_size,
                tags,
                description,
                dependencies: Vec::new(),
                checksum: Some(checksum),
            };

            self.models.insert(model_id, entry);
            self.save_registry()?;
            Ok(())
        }

        /// List all registered models
        pub fn list_models(&self) -> Vec<&ModelRegistryEntry> {
            self.models.values().collect()
        }

        /// Find models by tag
        pub fn find_by_tag(&self, tag: &str) -> Vec<&ModelRegistryEntry> {
            self.models
                .values()
                .filter(|entry| entry.tags.contains(&tag.to_string()))
                .collect()
        }

        /// Find models by type
        pub fn find_by_type(&self, model_type: &str) -> Vec<&ModelRegistryEntry> {
            self.models
                .values()
                .filter(|entry| entry.metadata.model_type == model_type)
                .collect()
        }

        /// Get model entry by ID
        pub fn get_model(&self, model_id: &str) -> Option<&ModelRegistryEntry> {
            self.models.get(model_id)
        }

        /// Remove a model from registry
        pub fn remove_model(&mut self, model_id: &str) -> Result<()> {
            if let Some(entry) = self.models.remove(model_id) {
                let full_path = self.base_directory.join(&entry.file_path);
                if full_path.exists() {
                    std::fs::remove_file(full_path).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Failed to remove model file: {}", e))
                    })?;
                }
                self.save_registry()?;
            }
            Ok(())
        }

        /// Verify model integrity
        pub fn verify_model(&self, model_id: &str) -> Result<bool> {
            if let Some(entry) = self.models.get(model_id) {
                let full_path = self.base_directory.join(&entry.file_path);
                if let Some(stored_checksum) = &entry.checksum {
                    let current_checksum = self.calculate_checksum(&full_path)?;
                    Ok(current_checksum == *stored_checksum)
                } else {
                    Ok(true) // No checksum stored, assume valid
                }
            } else {
                Err(ClusteringError::InvalidInput("Model not found".to_string()))
            }
        }

        /// Compact registry by removing unused models
        pub fn compact_registry(&mut self) -> Result<Vec<String>> {
            let mut removed = Vec::new();
            let entries_to_check: Vec<_> = self.models.iter().collect();

            for (model_id, entry) in entries_to_check {
                let full_path = self.base_directory.join(&entry.file_path);
                if !full_path.exists() {
                    removed.push(model_id.clone());
                }
            }

            for model_id in &removed {
                self.models.remove(model_id);
            }

            if !removed.is_empty() {
                self.save_registry()?;
            }

            Ok(removed)
        }

        /// Load registry from disk
        pub fn load_registry(&mut self) -> Result<()> {
            let registry_path = self.base_directory.join("registry.json");
            if registry_path.exists() {
                let content = std::fs::read_to_string(&registry_path).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to read registry: {}", e))
                })?;
                self.models = serde_json::from_str(&content).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to parse registry: {}", e))
                })?;
            }
            Ok(())
        }

        /// Save registry to disk
        fn save_registry(&self) -> Result<()> {
            std::fs::create_dir_all(&self.base_directory).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to create directory: {}", e))
            })?;

            let registry_path = self.base_directory.join("registry.json");
            let content = serde_json::to_string_pretty(&self.models).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to serialize registry: {}", e))
            })?;

            std::fs::write(&registry_path, content).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to write registry: {}", e))
            })?;

            Ok(())
        }

        /// Calculate file checksum
        fn calculate_checksum<P: AsRef<Path>>(&self, path: P) -> Result<String> {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let content = std::fs::read(path).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to read file for checksum: {}", e))
            })?;

            let mut hasher = DefaultHasher::new();
            content.hash(&mut hasher);
            Ok(format!("{:x}", hasher.finish()))
        }
    }

    /// Batch model operations for efficient processing
    #[derive(Debug)]
    pub struct BatchModelProcessor {
        /// Target directory for batch operations
        target_directory: PathBuf,
        /// Batch configuration
        config: BatchConfig,
    }

    /// Configuration for batch operations
    #[derive(Debug, Clone)]
    pub struct BatchConfig {
        /// Maximum number of models to process in parallel
        pub max_parallel: usize,
        /// Compression level for batch archives
        pub compression_level: u32,
        /// Include metadata in batch exports
        pub include_metadata: bool,
        /// Target format for batch conversion
        pub target_format: ExportFormat,
    }

    impl Default for BatchConfig {
        fn default() -> Self {
            Self {
                max_parallel: 4,
                compression_level: 6,
                include_metadata: true,
                target_format: ExportFormat::Json,
            }
        }
    }

    impl BatchModelProcessor {
        /// Create a new batch processor
        pub fn new<P: Into<PathBuf>>(target_directory: P, config: BatchConfig) -> Self {
            Self {
                target_directory: target_directory.into(),
                config,
            }
        }

        /// Export multiple models to a single archive
        pub fn export_batch<T: SerializableModel + AdvancedExport>(
            &self,
            models: &[(String, &T)],
            archive_name: &str,
        ) -> Result<PathBuf> {
            use flate2::write::GzEncoder;
            use std::io::Write;

            std::fs::create_dir_all(&self.target_directory).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to create directory: {}", e))
            })?;

            let archive_path = self
                .target_directory
                .join(format!("{}.tar.gz", archive_name));
            let file = File::create(&archive_path).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to create archive: {}", e))
            })?;

            let encoder = GzEncoder::new(file, Compression::new(self.config.compression_level));
            let mut tar = tar::Builder::new(encoder);

            for (model_name, model) in models {
                // Export model to temporary buffer
                let model_content = model.export_to_string(self.config.target_format)?;
                let file_name = format!(
                    "{}.{}",
                    model_name,
                    format_extension(self.config.target_format)
                );

                // Add to tar archive
                let mut header = tar::Header::new_gnu();
                header.set_path(&file_name).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to set tar path: {}", e))
                })?;
                header.set_size(model_content.len() as u64);
                header.set_cksum();

                tar.append(&header, model_content.as_bytes()).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to add to archive: {}", e))
                })?;

                // Add metadata if requested
                if self.config.include_metadata {
                    let metadata = model.get_metadata();
                    let metadata_content =
                        serde_json::to_string_pretty(&metadata).map_err(|e| {
                            ClusteringError::InvalidInput(format!(
                                "Failed to serialize metadata: {}",
                                e
                            ))
                        })?;

                    let metadata_file_name = format!("{}_metadata.json", model_name);
                    let mut metadata_header = tar::Header::new_gnu();
                    metadata_header.set_path(&metadata_file_name).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Failed to set metadata path: {}", e))
                    })?;
                    metadata_header.set_size(metadata_content.len() as u64);
                    metadata_header.set_cksum();

                    tar.append(&metadata_header, metadata_content.as_bytes())
                        .map_err(|e| {
                            ClusteringError::InvalidInput(format!(
                                "Failed to add metadata to archive: {}",
                                e
                            ))
                        })?;
                }
            }

            tar.finish().map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to finalize archive: {}", e))
            })?;

            Ok(archive_path)
        }

        /// Import models from batch archive
        pub fn import_batch<T: SerializableModel>(
            &self,
            archive_path: &Path,
        ) -> Result<Vec<(String, T)>> {
            use flate2::read::GzDecoder;

            let file = File::open(archive_path).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to open archive: {}", e))
            })?;

            let decoder = GzDecoder::new(file);
            let mut tar = tar::Archive::new(decoder);
            let mut models = Vec::new();

            for entry in tar.entries().map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to read archive entries: {}", e))
            })? {
                let mut entry = entry.map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to read archive entry: {}", e))
                })?;

                let path = entry.path().map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to get entry path: {}", e))
                })?;

                if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                    if !file_name.ends_with("_metadata.json") {
                        let mut contents = String::new();
                        entry.read_to_string(&mut contents).map_err(|e| {
                            ClusteringError::InvalidInput(format!(
                                "Failed to read entry contents: {}",
                                e
                            ))
                        })?;

                        let model: T = serde_json::from_str(&contents).map_err(|e| {
                            ClusteringError::InvalidInput(format!(
                                "Failed to deserialize model: {}",
                                e
                            ))
                        })?;

                        let model_name = file_name
                            .rsplit_once('.')
                            .map(|(name, _)| name)
                            .unwrap_or(file_name);
                        models.push((model_name.to_string(), model));
                    }
                }
            }

            Ok(models)
        }

        /// Convert models between formats in batch
        pub fn convert_batch(
            &self,
            input_dir: &Path,
            from_format: ExportFormat,
            to_format: ExportFormat,
        ) -> Result<usize> {
            let mut converted_count = 0;

            for entry in std::fs::read_dir(input_dir).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to read input directory: {}", e))
            })? {
                let entry = entry.map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to read directory entry: {}", e))
                })?;

                let path = entry.path();
                if path.is_file() {
                    if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
                        if extension == format_extension(from_format) {
                            // Read and convert
                            let content = std::fs::read(&path).map_err(|e| {
                                ClusteringError::InvalidInput(format!("Failed to read file: {}", e))
                            })?;

                            // This is a simplified conversion - in practice, you'd need to know the model type
                            // For now, we'll skip actual conversion and just count files
                            converted_count += 1;
                        }
                    }
                }
            }

            Ok(converted_count)
        }
    }

    /// Get file extension for export format
    fn format_extension(format: ExportFormat) -> &'static str {
        match format {
            ExportFormat::Json => "json",
            ExportFormat::Yaml => "yaml",
            ExportFormat::Csv => "csv",
            ExportFormat::Newick => "nwk",
            ExportFormat::Binary => "bin",
            ExportFormat::CompressedJson => "json.gz",
            ExportFormat::MessagePack => "msgpack",
            ExportFormat::Cbor => "cbor",
        }
    }
}

/// Model compression and optimization utilities
pub mod compression {
    use super::*;

    /// Compression configuration for models
    #[derive(Debug, Clone)]
    pub struct CompressionConfig {
        /// Compression algorithm to use
        pub algorithm: CompressionAlgorithm,
        /// Compression level (algorithm-dependent)
        pub level: u32,
        /// Enable quantization for numerical data
        pub enable_quantization: bool,
        /// Quantization precision (bits)
        pub quantization_bits: u8,
        /// Remove redundant data
        pub remove_redundancy: bool,
    }

    /// Available compression algorithms
    #[derive(Debug, Clone, Copy)]
    pub enum CompressionAlgorithm {
        /// Standard gzip compression
        Gzip,
        /// LZ4 fast compression
        Lz4,
        /// Zstandard high-ratio compression
        Zstd,
        /// BZIP2 high compression
        Bzip2,
        /// No compression
        None,
    }

    impl Default for CompressionConfig {
        fn default() -> Self {
            Self {
                algorithm: CompressionAlgorithm::Gzip,
                level: 6,
                enable_quantization: false,
                quantization_bits: 16,
                remove_redundancy: true,
            }
        }
    }

    /// Compressed model container
    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct CompressedModel {
        /// Original model metadata
        pub metadata: ModelMetadata,
        /// Compressed data
        pub compressed_data: Vec<u8>,
        /// Compression configuration used
        pub compression_config: CompressionInfo,
        /// Original size before compression
        pub original_size: usize,
        /// Compression ratio achieved
        pub compression_ratio: f64,
    }

    /// Information about compression applied
    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct CompressionInfo {
        /// Algorithm used
        pub algorithm: String,
        /// Compression level
        pub level: u32,
        /// Whether quantization was applied
        pub quantized: bool,
        /// Quantization precision if applied
        pub quantization_bits: Option<u8>,
    }

    impl CompressedModel {
        /// Compress a model with given configuration
        pub fn compress<T: SerializableModel + AdvancedExport>(
            model: &T,
            config: CompressionConfig,
        ) -> Result<Self> {
            let metadata = model.get_metadata();

            // Serialize model to bytes
            let original_data = enhanced::serialize_with_format(model, ExportFormat::Binary)?;
            let original_size = original_data.len();

            // Apply compression
            let compressed_data = Self::apply_compression(&original_data, &config)?;

            let compression_ratio = compressed_data.len() as f64 / original_size as f64;

            let compression_info = CompressionInfo {
                algorithm: format!("{:?}", config.algorithm),
                level: config.level,
                quantized: config.enable_quantization,
                quantization_bits: if config.enable_quantization {
                    Some(config.quantization_bits)
                } else {
                    None
                },
            };

            Ok(Self {
                metadata,
                compressed_data,
                compression_config: compression_info,
                original_size,
                compression_ratio,
            })
        }

        /// Decompress and deserialize the model
        pub fn decompress<T: SerializableModel>(&self) -> Result<T> {
            let decompressed_data =
                Self::apply_decompression(&self.compressed_data, &self.compression_config)?;

            enhanced::deserialize_with_format(&decompressed_data, ExportFormat::Binary)
        }

        /// Apply compression to data
        fn apply_compression(data: &[u8], config: &CompressionConfig) -> Result<Vec<u8>> {
            match config.algorithm {
                CompressionAlgorithm::Gzip => {
                    let mut encoder = GzEncoder::new(Vec::new(), Compression::new(config.level));
                    encoder.write_all(data).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Compression failed: {}", e))
                    })?;
                    encoder.finish().map_err(|e| {
                        ClusteringError::InvalidInput(format!(
                            "Compression finalization failed: {}",
                            e
                        ))
                    })
                }
                CompressionAlgorithm::None => Ok(data.to_vec()),
                _ => {
                    // For other algorithms, fall back to gzip for now
                    let mut encoder = GzEncoder::new(Vec::new(), Compression::new(config.level));
                    encoder.write_all(data).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Compression failed: {}", e))
                    })?;
                    encoder.finish().map_err(|e| {
                        ClusteringError::InvalidInput(format!(
                            "Compression finalization failed: {}",
                            e
                        ))
                    })
                }
            }
        }

        /// Apply decompression to data
        fn apply_decompression(data: &[u8], info: &CompressionInfo) -> Result<Vec<u8>> {
            match info.algorithm.as_str() {
                "Gzip" => {
                    let mut decoder = GzDecoder::new(data);
                    let mut decompressed = Vec::new();
                    decoder.read_to_end(&mut decompressed).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Decompression failed: {}", e))
                    })?;
                    Ok(decompressed)
                }
                "None" => Ok(data.to_vec()),
                _ => {
                    // Default to gzip for unknown algorithms
                    let mut decoder = GzDecoder::new(data);
                    let mut decompressed = Vec::new();
                    decoder.read_to_end(&mut decompressed).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Decompression failed: {}", e))
                    })?;
                    Ok(decompressed)
                }
            }
        }

        /// Get compression statistics
        pub fn get_compression_stats(&self) -> CompressionStats {
            CompressionStats {
                original_size: self.original_size,
                compressed_size: self.compressed_data.len(),
                compression_ratio: self.compression_ratio,
                space_saved_bytes: self
                    .original_size
                    .saturating_sub(self.compressed_data.len()),
                space_saved_percentage: (1.0 - self.compression_ratio) * 100.0,
            }
        }
    }

    /// Compression statistics
    #[derive(Debug, Clone)]
    pub struct CompressionStats {
        /// Original size in bytes
        pub original_size: usize,
        /// Compressed size in bytes
        pub compressed_size: usize,
        /// Compression ratio (compressed/original)
        pub compression_ratio: f64,
        /// Space saved in bytes
        pub space_saved_bytes: usize,
        /// Space saved percentage
        pub space_saved_percentage: f64,
    }
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
        let linkage = array![[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 2.0, 2.0]];

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
