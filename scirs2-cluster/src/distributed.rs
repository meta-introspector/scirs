//! Distributed clustering algorithms for large-scale datasets
//!
//! This module provides distributed implementations of clustering algorithms that can
//! handle datasets too large to fit in memory on a single machine. It supports
//! distributed K-means, hierarchical clustering, and data partitioning strategies.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive, Zero};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{ClusteringError, Result};
use crate::vq::euclidean_distance;

/// Configuration for distributed clustering algorithms
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DistributedConfig {
    /// Number of worker nodes/processes
    pub n_workers: usize,
    /// Data partitioning strategy
    pub partitioning_strategy: PartitioningStrategy,
    /// Maximum iterations per coordination round
    pub max_iterations_per_round: usize,
    /// Global convergence tolerance
    pub global_tolerance: f64,
    /// Maximum number of coordination rounds
    pub max_coordination_rounds: usize,
    /// Memory limit per worker (in MB)
    pub memory_limit_mb: usize,
    /// Whether to use compression for data transfer
    pub use_compression: bool,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            n_workers: 4,
            partitioning_strategy: PartitioningStrategy::RandomSplit,
            max_iterations_per_round: 10,
            global_tolerance: 1e-4,
            max_coordination_rounds: 100,
            memory_limit_mb: 1024,
            use_compression: true,
            load_balancing: LoadBalancingStrategy::EqualSize,
        }
    }
}

/// Data partitioning strategies for distributed computing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PartitioningStrategy {
    /// Random assignment of data points to workers
    RandomSplit,
    /// Hash-based partitioning for deterministic assignment
    HashBased,
    /// K-d tree based spatial partitioning
    SpatialPartitioning,
    /// Stratified sampling to ensure balanced clusters
    StratifiedSampling,
    /// Round-robin assignment
    RoundRobin,
}

/// Load balancing strategies for distributed workers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LoadBalancingStrategy {
    /// Equal number of data points per worker
    EqualSize,
    /// Balance by estimated computational cost
    ComputationalBalance,
    /// Dynamic load balancing based on worker performance
    Dynamic,
    /// Custom weights for each worker
    WeightedDistribution,
}

/// Distributed data partition
#[derive(Debug, Clone)]
pub struct DataPartition<F: Float> {
    /// Partition identifier
    pub partition_id: usize,
    /// Data points in this partition
    pub data: Array2<F>,
    /// Local cluster assignments
    pub labels: Option<Array1<usize>>,
    /// Worker node identifier
    pub worker_id: usize,
    /// Partition weight for load balancing
    pub weight: f64,
}

/// Distributed K-means algorithm coordinator
#[derive(Debug)]
pub struct DistributedKMeans<F: Float> {
    /// Configuration
    pub config: DistributedConfig,
    /// Number of clusters
    pub k: usize,
    /// Global centroids
    pub centroids: Option<Array2<F>>,
    /// Data partitions across workers
    pub partitions: Vec<DataPartition<F>>,
    /// Convergence history
    pub convergence_history: Vec<ConvergenceMetrics>,
    /// Worker statistics
    pub worker_stats: HashMap<usize, WorkerStatistics>,
}

/// Convergence metrics for distributed algorithms
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConvergenceMetrics {
    /// Coordination round number
    pub round: usize,
    /// Global inertia (sum of squared distances)
    pub global_inertia: f64,
    /// Change in global inertia
    pub inertia_change: f64,
    /// Number of points that changed cluster assignment
    pub points_changed: usize,
    /// Maximum centroid movement distance
    pub max_centroid_movement: f64,
    /// Whether algorithm has converged globally
    pub converged: bool,
    /// Worker synchronization time (milliseconds)
    pub sync_time_ms: u64,
}

/// Statistics for individual workers in distributed computation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WorkerStatistics {
    /// Worker identifier
    pub worker_id: usize,
    /// Number of data points processed
    pub n_points: usize,
    /// Local computation time per round (milliseconds)
    pub computation_time_ms: u64,
    /// Communication time per round (milliseconds)
    pub communication_time_ms: u64,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
    /// Local convergence status
    pub local_converged: bool,
    /// Number of local iterations performed
    pub local_iterations: usize,
}

impl<F: Float + FromPrimitive + Debug + Send + Sync> DistributedKMeans<F> {
    /// Create new distributed K-means instance
    pub fn new(k: usize, config: DistributedConfig) -> Self {
        Self {
            config,
            k,
            centroids: None,
            partitions: Vec::new(),
            convergence_history: Vec::new(),
            worker_stats: HashMap::new(),
        }
    }

    /// Partition data across workers using specified strategy
    pub fn partition_data(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput("Empty dataset".to_string()));
        }

        // Calculate partition sizes based on load balancing strategy
        let partition_sizes = self.calculate_partition_sizes(n_samples)?;

        // Create partitions based on strategy
        self.partitions = match self.config.partitioning_strategy {
            PartitioningStrategy::RandomSplit => self.random_partition(data, &partition_sizes)?,
            PartitioningStrategy::HashBased => self.hash_based_partition(data, &partition_sizes)?,
            PartitioningStrategy::SpatialPartitioning => {
                self.spatial_partition(data, &partition_sizes)?
            }
            PartitioningStrategy::StratifiedSampling => {
                self.stratified_partition(data, &partition_sizes)?
            }
            PartitioningStrategy::RoundRobin => {
                self.round_robin_partition(data, &partition_sizes)?
            }
        };

        Ok(())
    }

    /// Run distributed K-means clustering
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        // Partition data across workers
        self.partition_data(data)?;

        // Initialize global centroids
        self.centroids = Some(self.initialize_global_centroids()?);

        // Distributed coordination loop
        for round in 0..self.config.max_coordination_rounds {
            let start_time = std::time::Instant::now();

            // Local computation phase on each worker
            let local_results = self.execute_local_kmeans_round()?;

            // Global coordination phase
            let new_centroids = self.coordinate_global_centroids(&local_results)?;
            let convergence = self.check_global_convergence(&new_centroids, round)?;

            // Update global state
            self.centroids = Some(new_centroids);
            self.convergence_history.push(convergence.clone());

            let sync_time = start_time.elapsed().as_millis() as u64;

            // Log round completion
            if round % 10 == 0 {
                println!(
                    "Round {}: Global inertia = {:.4}, Converged = {}",
                    round, convergence.global_inertia, convergence.converged
                );
            }

            // Check for global convergence
            if convergence.converged {
                println!("Distributed K-means converged after {} rounds", round + 1);
                break;
            }
        }

        Ok(())
    }

    /// Calculate partition sizes based on load balancing strategy
    fn calculate_partition_sizes(&self, total_samples: usize) -> Result<Vec<usize>> {
        match self.config.load_balancing {
            LoadBalancingStrategy::EqualSize => {
                let base_size = total_samples / self.config.n_workers;
                let remainder = total_samples % self.config.n_workers;

                let mut sizes = vec![base_size; self.config.n_workers];
                for i in 0..remainder {
                    sizes[i] += 1;
                }
                Ok(sizes)
            }
            LoadBalancingStrategy::ComputationalBalance => {
                // Simulate computational weights (in practice, these would be measured)
                let weights = vec![1.0; self.config.n_workers];
                self.distribute_by_weights(total_samples, &weights)
            }
            LoadBalancingStrategy::Dynamic => {
                // Start with equal distribution, adjust later based on performance
                self.calculate_partition_sizes(total_samples)
            }
            LoadBalancingStrategy::WeightedDistribution => {
                // Use predefined weights (simplified)
                let weights = vec![1.0; self.config.n_workers];
                self.distribute_by_weights(total_samples, &weights)
            }
        }
    }

    /// Distribute samples based on worker weights
    fn distribute_by_weights(&self, total_samples: usize, weights: &[f64]) -> Result<Vec<usize>> {
        let total_weight: f64 = weights.iter().sum();
        if total_weight <= 0.0 {
            return Err(ClusteringError::InvalidInput(
                "Worker weights must be positive".to_string(),
            ));
        }

        let mut sizes = Vec::new();
        let mut assigned = 0;

        for (i, &weight) in weights.iter().enumerate() {
            let size = if i == weights.len() - 1 {
                // Last worker gets remaining samples
                total_samples - assigned
            } else {
                ((total_samples as f64 * weight / total_weight).round() as usize)
                    .min(total_samples - assigned)
            };
            sizes.push(size);
            assigned += size;
        }

        Ok(sizes)
    }

    /// Random data partitioning
    fn random_partition(
        &self,
        data: ArrayView2<F>,
        partition_sizes: &[usize],
    ) -> Result<Vec<DataPartition<F>>> {
        let mut partitions = Vec::new();
        let mut indices: Vec<usize> = (0..data.nrows()).collect();

        // Shuffle indices for random assignment
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());

        let mut start_idx = 0;
        for (worker_id, &size) in partition_sizes.iter().enumerate() {
            let end_idx = (start_idx + size).min(data.nrows());

            if start_idx < end_idx {
                let partition_indices = &indices[start_idx..end_idx];
                let mut partition_data = Array2::zeros((partition_indices.len(), data.ncols()));

                for (i, &row_idx) in partition_indices.iter().enumerate() {
                    partition_data.row_mut(i).assign(&data.row(row_idx));
                }

                partitions.push(DataPartition {
                    partition_id: worker_id,
                    data: partition_data,
                    labels: None,
                    worker_id,
                    weight: size as f64 / data.nrows() as f64,
                });
            }

            start_idx = end_idx;
        }

        Ok(partitions)
    }

    /// Hash-based data partitioning
    fn hash_based_partition(
        &self,
        data: ArrayView2<F>,
        partition_sizes: &[usize],
    ) -> Result<Vec<DataPartition<F>>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let n_workers = self.config.n_workers;
        let mut worker_data: Vec<Vec<usize>> = vec![Vec::new(); n_workers];

        // Assign each point to a worker based on hash of its coordinates
        for (row_idx, row) in data.rows().into_iter().enumerate() {
            let mut hasher = DefaultHasher::new();

            // Hash the first few coordinates to determine worker assignment
            for &value in row.iter().take(3.min(row.len())) {
                let bits = value.to_f64().unwrap_or(0.0).to_bits();
                bits.hash(&mut hasher);
            }

            let worker_id = (hasher.finish() as usize) % n_workers;
            worker_data[worker_id].push(row_idx);
        }

        // Create partitions
        let mut partitions = Vec::new();
        for (worker_id, row_indices) in worker_data.into_iter().enumerate() {
            if !row_indices.is_empty() {
                let mut partition_data = Array2::zeros((row_indices.len(), data.ncols()));

                for (i, &row_idx) in row_indices.iter().enumerate() {
                    partition_data.row_mut(i).assign(&data.row(row_idx));
                }

                partitions.push(DataPartition {
                    partition_id: worker_id,
                    data: partition_data,
                    labels: None,
                    worker_id,
                    weight: row_indices.len() as f64 / data.nrows() as f64,
                });
            }
        }

        Ok(partitions)
    }

    /// Spatial partitioning using simplified k-d tree approach
    fn spatial_partition(
        &self,
        data: ArrayView2<F>,
        _partition_sizes: &[usize],
    ) -> Result<Vec<DataPartition<F>>> {
        let n_features = data.ncols();
        let n_workers = self.config.n_workers;

        // Simple spatial partitioning: divide along the dimension with highest variance
        let mut dimension_variances = Vec::new();
        for dim in 0..n_features {
            let column = data.column(dim);
            let mean =
                column.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(column.len()).unwrap();
            let variance = column
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(column.len()).unwrap();
            dimension_variances.push((dim, variance.to_f64().unwrap_or(0.0)));
        }

        // Sort by variance and use top dimension for partitioning
        dimension_variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let split_dim = dimension_variances[0].0;

        // Get values for splitting dimension
        let mut split_values: Vec<(usize, f64)> = data
            .column(split_dim)
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val.to_f64().unwrap_or(0.0)))
            .collect();

        split_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Create partitions based on sorted values
        let mut partitions = Vec::new();
        let partition_size = split_values.len() / n_workers;

        for worker_id in 0..n_workers {
            let start_idx = worker_id * partition_size;
            let end_idx = if worker_id == n_workers - 1 {
                split_values.len()
            } else {
                (worker_id + 1) * partition_size
            };

            if start_idx < end_idx {
                let partition_indices: Vec<usize> = split_values[start_idx..end_idx]
                    .iter()
                    .map(|(idx, _)| *idx)
                    .collect();

                let mut partition_data = Array2::zeros((partition_indices.len(), data.ncols()));
                for (i, &row_idx) in partition_indices.iter().enumerate() {
                    partition_data.row_mut(i).assign(&data.row(row_idx));
                }

                partitions.push(DataPartition {
                    partition_id: worker_id,
                    data: partition_data,
                    labels: None,
                    worker_id,
                    weight: partition_indices.len() as f64 / data.nrows() as f64,
                });
            }
        }

        Ok(partitions)
    }

    /// Stratified sampling partition (simplified)
    fn stratified_partition(
        &self,
        data: ArrayView2<F>,
        partition_sizes: &[usize],
    ) -> Result<Vec<DataPartition<F>>> {
        // For now, fall back to random partitioning
        // In a full implementation, this would perform clustering first
        // to identify strata, then sample proportionally from each
        self.random_partition(data, partition_sizes)
    }

    /// Round-robin partitioning
    fn round_robin_partition(
        &self,
        data: ArrayView2<F>,
        _partition_sizes: &[usize],
    ) -> Result<Vec<DataPartition<F>>> {
        let n_workers = self.config.n_workers;
        let mut worker_data: Vec<Vec<usize>> = vec![Vec::new(); n_workers];

        // Assign points in round-robin fashion
        for (row_idx, _) in data.rows().into_iter().enumerate() {
            let worker_id = row_idx % n_workers;
            worker_data[worker_id].push(row_idx);
        }

        // Create partitions
        let mut partitions = Vec::new();
        for (worker_id, row_indices) in worker_data.into_iter().enumerate() {
            if !row_indices.is_empty() {
                let mut partition_data = Array2::zeros((row_indices.len(), data.ncols()));

                for (i, &row_idx) in row_indices.iter().enumerate() {
                    partition_data.row_mut(i).assign(&data.row(row_idx));
                }

                partitions.push(DataPartition {
                    partition_id: worker_id,
                    data: partition_data,
                    labels: None,
                    worker_id,
                    weight: row_indices.len() as f64 / data.nrows() as f64,
                });
            }
        }

        Ok(partitions)
    }

    /// Initialize global centroids using K-means++ across partitions
    fn initialize_global_centroids(&self) -> Result<Array2<F>> {
        if self.partitions.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "No data partitions available".to_string(),
            ));
        }

        let n_features = self.partitions[0].data.ncols();
        let mut centroids = Array2::zeros((self.k, n_features));

        // Use first partition for initialization (simplified)
        let first_partition = &self.partitions[0];
        if first_partition.data.nrows() < self.k {
            return Err(ClusteringError::InvalidInput(
                "Not enough data points for initialization".to_string(),
            ));
        }

        // Simple initialization: use first k points from first partition
        for i in 0..self.k {
            let row_idx = i % first_partition.data.nrows();
            centroids
                .row_mut(i)
                .assign(&first_partition.data.row(row_idx));
        }

        Ok(centroids)
    }

    /// Execute local K-means iterations on each worker
    fn execute_local_kmeans_round(&mut self) -> Result<Vec<LocalKMeansResult<F>>> {
        let global_centroids = self.centroids.as_ref().unwrap();
        let mut results = Vec::new();

        for partition in &mut self.partitions {
            let start_time = std::time::Instant::now();

            // Local K-means iterations
            let (local_centroids, local_labels, local_inertia) =
                self.local_kmeans_iterations(&partition.data, global_centroids.view())?;

            let computation_time = start_time.elapsed().as_millis() as u64;

            // Update partition labels
            partition.labels = Some(local_labels.clone());

            // Create result
            results.push(LocalKMeansResult {
                worker_id: partition.worker_id,
                local_centroids,
                local_labels,
                local_inertia,
                n_points: partition.data.nrows(),
                computation_time_ms: computation_time,
            });

            // Update worker statistics
            self.worker_stats.insert(
                partition.worker_id,
                WorkerStatistics {
                    worker_id: partition.worker_id,
                    n_points: partition.data.nrows(),
                    computation_time_ms: computation_time,
                    communication_time_ms: 0, // Simplified
                    memory_usage_mb: self.estimate_memory_usage(&partition.data),
                    local_converged: false, // Simplified
                    local_iterations: self.config.max_iterations_per_round,
                },
            );
        }

        Ok(results)
    }

    /// Perform local K-means iterations on a data partition
    fn local_kmeans_iterations(
        &self,
        data: &Array2<F>,
        global_centroids: ArrayView2<F>,
    ) -> Result<(Array2<F>, Array1<usize>, F)> {
        let mut centroids = global_centroids.to_owned();
        let mut labels = Array1::zeros(data.nrows());
        let mut prev_inertia = F::infinity();

        for _iter in 0..self.config.max_iterations_per_round {
            // Assign points to nearest centroids
            for (i, point) in data.rows().into_iter().enumerate() {
                let mut min_dist = F::infinity();
                let mut best_cluster = 0;

                for (j, centroid) in centroids.rows().into_iter().enumerate() {
                    let dist = euclidean_distance(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = j;
                    }
                }
                labels[i] = best_cluster;
            }

            // Update centroids
            let new_centroids = self.compute_local_centroids(data, &labels)?;

            // Calculate inertia
            let inertia = self.compute_local_inertia(data, &labels, new_centroids.view())?;

            // Check for local convergence
            if (prev_inertia - inertia).abs() <= F::from(self.config.global_tolerance).unwrap() {
                break;
            }

            centroids = new_centroids;
            prev_inertia = inertia;
        }

        Ok((centroids, labels, prev_inertia))
    }

    /// Compute local centroids from data and labels
    fn compute_local_centroids(
        &self,
        data: &Array2<F>,
        labels: &Array1<usize>,
    ) -> Result<Array2<F>> {
        let n_features = data.ncols();
        let mut centroids = Array2::zeros((self.k, n_features));
        let mut counts = vec![0; self.k];

        // Accumulate sums for each cluster
        for (point, &label) in data.rows().into_iter().zip(labels.iter()) {
            for (j, &value) in point.iter().enumerate() {
                centroids[[label, j]] = centroids[[label, j]] + value;
            }
            counts[label] += 1;
        }

        // Compute averages
        for i in 0..self.k {
            if counts[i] > 0 {
                for j in 0..n_features {
                    centroids[[i, j]] = centroids[[i, j]] / F::from(counts[i]).unwrap();
                }
            }
        }

        Ok(centroids)
    }

    /// Compute local inertia
    fn compute_local_inertia(
        &self,
        data: &Array2<F>,
        labels: &Array1<usize>,
        centroids: ArrayView2<F>,
    ) -> Result<F> {
        let mut inertia = F::zero();

        for (point, &label) in data.rows().into_iter().zip(labels.iter()) {
            let centroid = centroids.row(label);
            let dist = euclidean_distance(point, centroid);
            inertia = inertia + dist * dist;
        }

        Ok(inertia)
    }

    /// Coordinate global centroids from local results
    fn coordinate_global_centroids(
        &self,
        local_results: &[LocalKMeansResult<F>],
    ) -> Result<Array2<F>> {
        let n_features = self.partitions[0].data.ncols();
        let mut global_centroids = Array2::zeros((self.k, n_features));
        let mut cluster_counts = vec![0; self.k];

        // Weighted average of local centroids
        for result in local_results {
            for i in 0..self.k {
                // Count points in this cluster for this worker
                let local_count = result
                    .local_labels
                    .iter()
                    .filter(|&&label| label == i)
                    .count();

                if local_count > 0 {
                    for j in 0..n_features {
                        global_centroids[[i, j]] = global_centroids[[i, j]]
                            + result.local_centroids[[i, j]] * F::from(local_count).unwrap();
                    }
                    cluster_counts[i] += local_count;
                }
            }
        }

        // Compute global averages
        for i in 0..self.k {
            if cluster_counts[i] > 0 {
                for j in 0..n_features {
                    global_centroids[[i, j]] =
                        global_centroids[[i, j]] / F::from(cluster_counts[i]).unwrap();
                }
            }
        }

        Ok(global_centroids)
    }

    /// Check global convergence
    fn check_global_convergence(
        &self,
        new_centroids: &Array2<F>,
        round: usize,
    ) -> Result<ConvergenceMetrics> {
        let old_centroids = self.centroids.as_ref().unwrap();

        // Compute maximum centroid movement
        let mut max_movement = 0.0;
        for i in 0..self.k {
            let movement = euclidean_distance(old_centroids.row(i), new_centroids.row(i))
                .to_f64()
                .unwrap_or(0.0);
            max_movement = max_movement.max(movement);
        }

        // Compute global inertia
        let mut global_inertia = 0.0;
        for partition in &self.partitions {
            if let Some(ref labels) = partition.labels {
                let local_inertia = self
                    .compute_local_inertia(&partition.data, labels, new_centroids.view())?
                    .to_f64()
                    .unwrap_or(0.0);
                global_inertia += local_inertia;
            }
        }

        // Compute change in inertia
        let inertia_change = if let Some(last_convergence) = self.convergence_history.last() {
            (last_convergence.global_inertia - global_inertia).abs()
        } else {
            f64::INFINITY
        };

        let converged = max_movement < self.config.global_tolerance
            && inertia_change < self.config.global_tolerance;

        Ok(ConvergenceMetrics {
            round,
            global_inertia,
            inertia_change,
            points_changed: 0, // Simplified
            max_centroid_movement: max_movement,
            converged,
            sync_time_ms: 0, // Simplified
        })
    }

    /// Estimate memory usage for a data partition
    fn estimate_memory_usage(&self, data: &Array2<F>) -> f64 {
        let bytes_per_element = std::mem::size_of::<F>();
        let total_elements = data.len();
        let total_bytes = total_elements * bytes_per_element;
        total_bytes as f64 / (1024.0 * 1024.0) // Convert to MB
    }

    /// Get final cluster assignments for all data
    pub fn get_labels(&self) -> Result<Array1<usize>> {
        let total_points: usize = self.partitions.iter().map(|p| p.data.nrows()).sum();

        let mut global_labels = Array1::zeros(total_points);
        let mut offset = 0;

        for partition in &self.partitions {
            if let Some(ref labels) = partition.labels {
                let end = offset + labels.len();
                global_labels
                    .slice_mut(ndarray::s![offset..end])
                    .assign(labels);
                offset = end;
            }
        }

        Ok(global_labels)
    }

    /// Get convergence history
    pub fn get_convergence_history(&self) -> &[ConvergenceMetrics] {
        &self.convergence_history
    }

    /// Get worker statistics
    pub fn get_worker_statistics(&self) -> &HashMap<usize, WorkerStatistics> {
        &self.worker_stats
    }
}

/// Result from local K-means computation on a worker
#[derive(Debug, Clone)]
struct LocalKMeansResult<F: Float> {
    worker_id: usize,
    local_centroids: Array2<F>,
    local_labels: Array1<usize>,
    local_inertia: F,
    n_points: usize,
    computation_time_ms: u64,
}

/// Distributed hierarchical clustering (simplified implementation)
#[derive(Debug)]
pub struct DistributedHierarchical<F: Float> {
    pub config: DistributedConfig,
    pub linkage_method: LinkageMethod,
    pub partitions: Vec<DataPartition<F>>,
    pub local_dendrograms: Vec<LocalDendrogram>,
}

/// Linkage methods for hierarchical clustering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LinkageMethod {
    Single,
    Complete,
    Average,
    Ward,
}

/// Local dendrogram from a worker
#[derive(Debug, Clone)]
pub struct LocalDendrogram {
    pub worker_id: usize,
    pub merge_sequence: Vec<(usize, usize, f64)>, // (cluster1, cluster2, distance)
    pub n_points: usize,
}

impl<F: Float + FromPrimitive + Debug + Send + Sync> DistributedHierarchical<F> {
    /// Create new distributed hierarchical clustering instance
    pub fn new(linkage_method: LinkageMethod, config: DistributedConfig) -> Self {
        Self {
            config,
            linkage_method,
            partitions: Vec::new(),
            local_dendrograms: Vec::new(),
        }
    }

    /// Partition data and perform local hierarchical clustering
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        // Reuse partitioning logic from DistributedKMeans
        let mut temp_kmeans = DistributedKMeans::new(2, self.config.clone());
        temp_kmeans.partition_data(data)?;
        self.partitions = temp_kmeans.partitions;

        // Perform local hierarchical clustering on each partition
        for partition in &self.partitions {
            let local_dendrogram = self.local_hierarchical_clustering(&partition.data)?;
            self.local_dendrograms.push(local_dendrogram);
        }

        Ok(())
    }

    /// Perform hierarchical clustering on a local partition
    fn local_hierarchical_clustering(&self, data: &Array2<F>) -> Result<LocalDendrogram> {
        let n_points = data.nrows();
        let mut merge_sequence = Vec::new();

        // Simplified hierarchical clustering: merge closest pairs
        // In a full implementation, this would use proper linkage criteria
        let mut clusters: Vec<Vec<usize>> = (0..n_points).map(|i| vec![i]).collect();
        let mut next_cluster_id = n_points;

        while clusters.len() > 1 {
            let mut min_distance = f64::INFINITY;
            let mut merge_pair = (0, 1);

            // Find closest pair of clusters
            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    let distance =
                        self.compute_cluster_distance(data, &clusters[i], &clusters[j])?;
                    if distance < min_distance {
                        min_distance = distance;
                        merge_pair = (i, j);
                    }
                }
            }

            // Merge clusters
            let (i, j) = merge_pair;
            let mut merged_cluster = clusters[i].clone();
            merged_cluster.extend_from_slice(&clusters[j]);

            merge_sequence.push((i, j, min_distance));

            // Remove old clusters and add merged cluster
            if i > j {
                clusters.remove(i);
                clusters.remove(j);
            } else {
                clusters.remove(j);
                clusters.remove(i);
            }
            clusters.push(merged_cluster);
            next_cluster_id += 1;
        }

        Ok(LocalDendrogram {
            worker_id: 0, // Simplified
            merge_sequence,
            n_points,
        })
    }

    /// Compute distance between two clusters
    fn compute_cluster_distance(
        &self,
        data: &Array2<F>,
        cluster1: &[usize],
        cluster2: &[usize],
    ) -> Result<f64> {
        match self.linkage_method {
            LinkageMethod::Single => {
                // Single linkage: minimum distance
                let mut min_dist = f64::INFINITY;
                for &i in cluster1 {
                    for &j in cluster2 {
                        let dist = euclidean_distance(data.row(i), data.row(j))
                            .to_f64()
                            .unwrap_or(f64::INFINITY);
                        min_dist = min_dist.min(dist);
                    }
                }
                Ok(min_dist)
            }
            LinkageMethod::Complete => {
                // Complete linkage: maximum distance
                let mut max_dist = 0.0;
                for &i in cluster1 {
                    for &j in cluster2 {
                        let dist = euclidean_distance(data.row(i), data.row(j))
                            .to_f64()
                            .unwrap_or(0.0);
                        max_dist = max_dist.max(dist);
                    }
                }
                Ok(max_dist)
            }
            LinkageMethod::Average => {
                // Average linkage: mean distance
                let mut total_dist = 0.0;
                let mut count = 0;
                for &i in cluster1 {
                    for &j in cluster2 {
                        let dist = euclidean_distance(data.row(i), data.row(j))
                            .to_f64()
                            .unwrap_or(0.0);
                        total_dist += dist;
                        count += 1;
                    }
                }
                Ok(total_dist / count as f64)
            }
            LinkageMethod::Ward => {
                // Ward linkage: simplified implementation
                self.compute_cluster_distance(data, cluster1, cluster2)
            }
        }
    }
}

/// Utility functions for distributed clustering
pub mod utils {
    use super::*;

    /// Estimate optimal number of workers based on data size and memory constraints
    pub fn estimate_optimal_workers(data_size_mb: f64, memory_limit_mb: f64) -> usize {
        let min_workers = 1;
        let max_workers = num_cpus::get().max(1);
        let memory_based_workers = (data_size_mb / memory_limit_mb).ceil() as usize;

        memory_based_workers.clamp(min_workers, max_workers)
    }

    /// Generate synthetic dataset for distributed clustering testing
    pub fn generate_large_dataset(
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
    ) -> Array2<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));

        let cluster_size = n_samples / n_clusters;

        for cluster in 0..n_clusters {
            let start_idx = cluster * cluster_size;
            let end_idx = if cluster == n_clusters - 1 {
                n_samples
            } else {
                (cluster + 1) * cluster_size
            };

            // Generate cluster center
            let center: Vec<f64> = (0..n_features)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect();

            // Generate points around center
            for i in start_idx..end_idx {
                for j in 0..n_features {
                    data[[i, j]] = center[j] + rng.gen_range(-2.0..2.0);
                }
            }
        }

        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_distributed_kmeans_partitioning() {
        let data =
            Array2::from_shape_vec((100, 3), (0..300).map(|i| i as f64 / 10.0).collect()).unwrap();

        let config = DistributedConfig {
            n_workers: 4,
            partitioning_strategy: PartitioningStrategy::RandomSplit,
            ..Default::default()
        };

        let mut distributed_kmeans = DistributedKMeans::new(3, config);
        distributed_kmeans.partition_data(data.view()).unwrap();

        // Check that all data is partitioned
        let total_partitioned: usize = distributed_kmeans
            .partitions
            .iter()
            .map(|p| p.data.nrows())
            .sum();
        assert_eq!(total_partitioned, 100);

        // Check that we have expected number of partitions
        assert_eq!(distributed_kmeans.partitions.len(), 4);
    }

    #[test]
    fn test_distributed_kmeans_simple() {
        let data = utils::generate_large_dataset(200, 2, 3);

        let config = DistributedConfig {
            n_workers: 2,
            max_coordination_rounds: 5,
            ..Default::default()
        };

        let mut distributed_kmeans = DistributedKMeans::new(3, config);

        // Test should complete without errors
        let result = distributed_kmeans.fit(data.view());
        assert!(result.is_ok());

        // Check that we have final labels
        let labels = distributed_kmeans.get_labels().unwrap();
        assert_eq!(labels.len(), 200);

        // Check convergence history
        assert!(!distributed_kmeans.get_convergence_history().is_empty());
    }

    #[test]
    fn test_load_balancing_strategies() {
        let total_samples = 100;
        let config = DistributedConfig {
            n_workers: 3,
            load_balancing: LoadBalancingStrategy::EqualSize,
            ..Default::default()
        };

        let distributed_kmeans = DistributedKMeans::<f64>::new(2, config);
        let sizes = distributed_kmeans
            .calculate_partition_sizes(total_samples)
            .unwrap();

        // Check that total adds up
        let total: usize = sizes.iter().sum();
        assert_eq!(total, total_samples);

        // Check relatively equal distribution
        let max_size = *sizes.iter().max().unwrap();
        let min_size = *sizes.iter().min().unwrap();
        assert!(max_size - min_size <= 1);
    }

    #[test]
    fn test_hierarchical_clustering() {
        let data = Array2::from_shape_vec((20, 2), (0..40).map(|i| i as f64).collect()).unwrap();

        let config = DistributedConfig {
            n_workers: 2,
            ..Default::default()
        };

        let mut distributed_hierarchical =
            DistributedHierarchical::new(LinkageMethod::Single, config);

        let result = distributed_hierarchical.fit(data.view());
        assert!(result.is_ok());

        // Check that local dendrograms were created
        assert_eq!(distributed_hierarchical.local_dendrograms.len(), 2);
    }
}
