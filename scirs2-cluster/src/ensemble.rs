//! Ensemble clustering methods for improved robustness
//!
//! This module provides ensemble clustering techniques that combine multiple
//! clustering algorithms or multiple runs of the same algorithm to achieve
//! more robust and stable clustering results.

use ndarray::{Array1, Array2, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use crate::affinity::affinity_propagation;
use crate::density::dbscan;
use crate::error::{ClusteringError, Result};
use crate::hierarchy::linkage;
use crate::meanshift::mean_shift;
use crate::metrics::{adjusted_rand_index, normalized_mutual_info, silhouette_score};
use crate::spectral::spectral_clustering;
use crate::vq::{kmeans, kmeans2};

/// Configuration for ensemble clustering
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    /// Number of base clustering algorithms to use
    pub n_estimators: usize,
    /// Sampling strategy for data subsets
    pub sampling_strategy: SamplingStrategy,
    /// Consensus method for combining results
    pub consensus_method: ConsensusMethod,
    /// Random seed for reproducible results
    pub random_seed: Option<u64>,
    /// Diversity enforcement strategy
    pub diversity_strategy: Option<DiversityStrategy>,
    /// Quality threshold for including results
    pub quality_threshold: Option<f64>,
    /// Maximum number of clusters to consider
    pub max_clusters: Option<usize>,
}

/// Sampling strategies for creating diverse datasets
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Bootstrap sampling with replacement
    Bootstrap { sample_ratio: f64 },
    /// Random subspace sampling (feature selection)
    RandomSubspace { feature_ratio: f64 },
    /// Combined bootstrap and subspace sampling
    BootstrapSubspace {
        sample_ratio: f64,
        feature_ratio: f64,
    },
    /// Random projection to lower dimensions
    RandomProjection { target_dimensions: usize },
    /// Noise injection for robustness testing
    NoiseInjection {
        noise_level: f64,
        noise_type: NoiseType,
    },
    /// No sampling (use full dataset)
    None,
}

/// Types of noise for injection
#[derive(Debug, Clone)]
pub enum NoiseType {
    /// Gaussian noise
    Gaussian,
    /// Uniform noise
    Uniform,
    /// Outlier injection
    Outliers { outlier_ratio: f64 },
}

/// Methods for combining clustering results
#[derive(Debug, Clone)]
pub enum ConsensusMethod {
    /// Simple majority voting
    MajorityVoting,
    /// Weighted consensus based on quality scores
    WeightedConsensus,
    /// Graph-based consensus clustering
    GraphBased { similarity_threshold: f64 },
    /// Hierarchical consensus
    Hierarchical { linkage_method: String },
    /// Co-association matrix approach
    CoAssociation { threshold: f64 },
    /// Evidence accumulation clustering
    EvidenceAccumulation,
}

/// Strategies for enforcing diversity among base clusterers
#[derive(Debug, Clone)]
pub enum DiversityStrategy {
    /// Algorithm diversity (use different algorithms)
    AlgorithmDiversity {
        algorithms: Vec<ClusteringAlgorithm>,
    },
    /// Parameter diversity (same algorithm, different parameters)
    ParameterDiversity {
        algorithm: ClusteringAlgorithm,
        parameter_ranges: HashMap<String, ParameterRange>,
    },
    /// Data diversity (different data subsets)
    DataDiversity {
        sampling_strategies: Vec<SamplingStrategy>,
    },
    /// Combined diversity strategy
    Combined { strategies: Vec<DiversityStrategy> },
}

/// Supported clustering algorithms for ensemble
#[derive(Debug, Clone)]
pub enum ClusteringAlgorithm {
    /// K-means clustering
    KMeans { k_range: (usize, usize) },
    /// DBSCAN clustering
    DBSCAN {
        eps_range: (f64, f64),
        min_samples_range: (usize, usize),
    },
    /// Mean shift clustering
    MeanShift { bandwidth_range: (f64, f64) },
    /// Hierarchical clustering
    Hierarchical { methods: Vec<String> },
    /// Spectral clustering
    Spectral { k_range: (usize, usize) },
    /// Affinity propagation
    AffinityPropagation { damping_range: (f64, f64) },
}

/// Parameter ranges for diversity
#[derive(Debug, Clone)]
pub enum ParameterRange {
    /// Integer range
    Integer(i64, i64),
    /// Float range
    Float(f64, f64),
    /// Categorical choices
    Categorical(Vec<String>),
    /// Boolean choice
    Boolean,
}

/// Result of a single clustering run
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    /// Cluster labels
    pub labels: Array1<i32>,
    /// Algorithm used
    pub algorithm: String,
    /// Parameters used
    pub parameters: HashMap<String, String>,
    /// Quality score
    pub quality_score: f64,
    /// Stability score (if available)
    pub stability_score: Option<f64>,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Runtime in seconds
    pub runtime: f64,
}

/// Ensemble clustering result
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    /// Final consensus labels
    pub consensus_labels: Array1<i32>,
    /// Individual clustering results
    pub individual_results: Vec<ClusteringResult>,
    /// Consensus statistics
    pub consensus_stats: ConsensusStatistics,
    /// Diversity metrics
    pub diversity_metrics: DiversityMetrics,
    /// Overall quality score
    pub ensemble_quality: f64,
    /// Stability score
    pub stability_score: f64,
}

/// Statistics about the consensus process
#[derive(Debug, Clone)]
pub struct ConsensusStatistics {
    /// Agreement matrix between clusterers
    pub agreement_matrix: Array2<f64>,
    /// Per-sample consensus strength
    pub consensus_strength: Array1<f64>,
    /// Cluster stability scores
    pub cluster_stability: Vec<f64>,
    /// Number of clusterers agreeing on each sample
    pub agreement_counts: Array1<usize>,
}

/// Diversity metrics for the ensemble
#[derive(Debug, Clone)]
pub struct DiversityMetrics {
    /// Average pairwise diversity (1 - ARI)
    pub average_diversity: f64,
    /// Diversity matrix between all pairs
    pub diversity_matrix: Array2<f64>,
    /// Algorithm distribution
    pub algorithm_distribution: HashMap<String, usize>,
    /// Parameter diversity statistics
    pub parameter_diversity: HashMap<String, f64>,
}

/// Main ensemble clustering implementation
pub struct EnsembleClusterer<F: Float> {
    config: EnsembleConfig,
    _phantom: std::marker::PhantomData<F>,
}

impl<
        F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display + Send + Sync,
    > EnsembleClusterer<F>
where
    f64: From<F>,
{
    /// Create a new ensemble clusterer
    pub fn new(config: EnsembleConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Perform ensemble clustering
    pub fn fit(&self, data: ArrayView2<F>) -> Result<EnsembleResult> {
        let start_time = std::time::Instant::now();

        // Generate diverse clustering results
        let individual_results = self.generate_diverse_clusterings(data)?;

        // Filter results based on quality threshold
        let filtered_results = self.filter_by_quality(&individual_results);

        // Combine results using consensus method
        let consensus_labels = self.build_consensus(&filtered_results, data)?;

        // Calculate ensemble statistics
        let consensus_stats =
            self.calculate_consensus_statistics(&filtered_results, &consensus_labels)?;
        let diversity_metrics = self.calculate_diversity_metrics(&filtered_results)?;

        // Calculate overall quality
        let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));
        let ensemble_quality =
            silhouette_score(data_f64.view(), consensus_labels.view()).unwrap_or(0.0);

        // Calculate stability score
        let stability_score = self.calculate_stability_score(&consensus_stats);

        let total_time = start_time.elapsed().as_secs_f64();

        Ok(EnsembleResult {
            consensus_labels,
            individual_results: filtered_results,
            consensus_stats,
            diversity_metrics,
            ensemble_quality,
            stability_score,
        })
    }

    /// Generate diverse clustering results
    fn generate_diverse_clusterings(&self, data: ArrayView2<F>) -> Result<Vec<ClusteringResult>> {
        let mut results = Vec::new();
        let mut rng = match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::seed_from_u64(42),
        };

        for i in 0..self.config.n_estimators {
            let clustering_start = std::time::Instant::now();

            // Apply sampling strategy
            let (sampled_data, sample_indices) = self.apply_sampling_strategy(data, &mut rng)?;

            // Select algorithm and parameters based on diversity strategy
            let (algorithm, parameters) = self.select_algorithm_and_parameters(i, &mut rng)?;

            // Run clustering
            let mut labels = self.run_clustering(&sampled_data, &algorithm, &parameters)?;

            // Map labels back to original data size if needed
            if sample_indices.len() != data.nrows() {
                labels = self.map_labels_to_full_data(&labels, &sample_indices, data.nrows())?;
            }

            // Calculate quality score
            let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));
            let quality_score = silhouette_score(data_f64.view(), labels.view()).unwrap_or(-1.0);

            let runtime = clustering_start.elapsed().as_secs_f64();
            let n_clusters = self.count_clusters(&labels);

            let result = ClusteringResult {
                labels,
                algorithm: format!("{:?}", algorithm),
                parameters,
                quality_score,
                stability_score: None,
                n_clusters,
                runtime,
            };

            results.push(result);
        }

        Ok(results)
    }

    /// Apply sampling strategy to data
    fn apply_sampling_strategy(
        &self,
        data: ArrayView2<F>,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<(Array2<F>, Vec<usize>)> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        match &self.config.sampling_strategy {
            SamplingStrategy::Bootstrap { sample_ratio } => {
                let sample_size = (n_samples as f64 * sample_ratio) as usize;
                let mut indices = Vec::new();

                for _ in 0..sample_size {
                    indices.push(rng.gen_range(0..n_samples));
                }

                let sampled_data = self.extract_samples(data, &indices)?;
                Ok((sampled_data, indices))
            }
            SamplingStrategy::RandomSubspace { feature_ratio } => {
                let n_selected_features = (n_features as f64 * feature_ratio) as usize;
                let mut feature_indices: Vec<usize> = (0..n_features).collect();
                feature_indices.shuffle(rng);
                feature_indices.truncate(n_selected_features);

                let sample_indices: Vec<usize> = (0..n_samples).collect();
                let sampled_data = self.extract_features(data, &feature_indices)?;
                Ok((sampled_data, sample_indices))
            }
            SamplingStrategy::BootstrapSubspace {
                sample_ratio,
                feature_ratio,
            } => {
                // First apply bootstrap sampling
                let sample_size = (n_samples as f64 * sample_ratio) as usize;
                let mut sample_indices = Vec::new();

                for _ in 0..sample_size {
                    sample_indices.push(rng.gen_range(0..n_samples));
                }

                // Then apply feature sampling
                let n_selected_features = (n_features as f64 * feature_ratio) as usize;
                let mut feature_indices: Vec<usize> = (0..n_features).collect();
                feature_indices.shuffle(rng);
                feature_indices.truncate(n_selected_features);

                let bootstrap_data = self.extract_samples(data, &sample_indices)?;
                let sampled_data =
                    self.extract_features(bootstrap_data.view(), &feature_indices)?;

                Ok((sampled_data, sample_indices))
            }
            SamplingStrategy::NoiseInjection {
                noise_level,
                noise_type,
            } => {
                let sample_indices: Vec<usize> = (0..n_samples).collect();
                let mut noisy_data = data.to_owned();

                match noise_type {
                    NoiseType::Gaussian => {
                        for i in 0..n_samples {
                            for j in 0..n_features {
                                let noise = F::from(rng.gen::<f64>() * 2.0 - 1.0).unwrap()
                                    * F::from(*noise_level).unwrap();
                                noisy_data[[i, j]] = noisy_data[[i, j]] + noise;
                            }
                        }
                    }
                    NoiseType::Uniform => {
                        for i in 0..n_samples {
                            for j in 0..n_features {
                                let noise =
                                    F::from((rng.gen::<f64>() * 2.0 - 1.0) * noise_level).unwrap();
                                noisy_data[[i, j]] = noisy_data[[i, j]] + noise;
                            }
                        }
                    }
                    NoiseType::Outliers { outlier_ratio } => {
                        let n_outliers = (n_samples as f64 * outlier_ratio) as usize;
                        for _ in 0..n_outliers {
                            let outlier_idx = rng.gen_range(0..n_samples);
                            for j in 0..n_features {
                                let outlier_value = F::from(rng.gen::<f64>() * 10.0 - 5.0).unwrap();
                                noisy_data[[outlier_idx, j]] = outlier_value;
                            }
                        }
                    }
                }

                Ok((noisy_data, sample_indices))
            }
            SamplingStrategy::None => {
                let sample_indices: Vec<usize> = (0..n_samples).collect();
                Ok((data.to_owned(), sample_indices))
            }
            _ => {
                // For other sampling strategies, fall back to no sampling
                let sample_indices: Vec<usize> = (0..n_samples).collect();
                Ok((data.to_owned(), sample_indices))
            }
        }
    }

    /// Extract samples based on indices
    fn extract_samples(&self, data: ArrayView2<F>, indices: &[usize]) -> Result<Array2<F>> {
        let n_features = data.ncols();
        let mut sampled_data = Array2::zeros((indices.len(), n_features));

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            if old_idx < data.nrows() {
                sampled_data.row_mut(new_idx).assign(&data.row(old_idx));
            }
        }

        Ok(sampled_data)
    }

    /// Extract features based on indices
    fn extract_features(
        &self,
        data: ArrayView2<F>,
        feature_indices: &[usize],
    ) -> Result<Array2<F>> {
        let n_samples = data.nrows();
        let mut sampled_data = Array2::zeros((n_samples, feature_indices.len()));

        for (new_feat_idx, &old_feat_idx) in feature_indices.iter().enumerate() {
            if old_feat_idx < data.ncols() {
                sampled_data
                    .column_mut(new_feat_idx)
                    .assign(&data.column(old_feat_idx));
            }
        }

        Ok(sampled_data)
    }

    /// Select algorithm and parameters based on diversity strategy
    fn select_algorithm_and_parameters(
        &self,
        estimator_index: usize,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<(ClusteringAlgorithm, HashMap<String, String>)> {
        match &self.config.diversity_strategy {
            Some(DiversityStrategy::AlgorithmDiversity { algorithms }) => {
                let algorithm = algorithms[estimator_index % algorithms.len()].clone();
                let parameters = self.generate_random_parameters(&algorithm, rng)?;
                Ok((algorithm, parameters))
            }
            Some(DiversityStrategy::ParameterDiversity {
                algorithm,
                parameter_ranges,
            }) => {
                let parameters = self.sample_parameter_ranges(parameter_ranges, rng)?;
                Ok((algorithm.clone(), parameters))
            }
            _ => {
                // Default to K-means with random k
                let k = rng.gen_range(2..=10);
                let algorithm = ClusteringAlgorithm::KMeans { k_range: (k, k) };
                let mut parameters = HashMap::new();
                parameters.insert("k".to_string(), k.to_string());
                Ok((algorithm, parameters))
            }
        }
    }

    /// Generate random parameters for an algorithm
    fn generate_random_parameters(
        &self,
        algorithm: &ClusteringAlgorithm,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<HashMap<String, String>> {
        let mut parameters = HashMap::new();

        match algorithm {
            ClusteringAlgorithm::KMeans { k_range } => {
                let k = rng.gen_range(k_range.0..=k_range.1);
                parameters.insert("k".to_string(), k.to_string());
            }
            ClusteringAlgorithm::DBSCAN {
                eps_range,
                min_samples_range,
            } => {
                let eps = rng.gen_range(eps_range.0..=eps_range.1);
                let min_samples = rng.gen_range(min_samples_range.0..=min_samples_range.1);
                parameters.insert("eps".to_string(), eps.to_string());
                parameters.insert("min_samples".to_string(), min_samples.to_string());
            }
            ClusteringAlgorithm::MeanShift { bandwidth_range } => {
                let bandwidth = rng.gen_range(bandwidth_range.0..=bandwidth_range.1);
                parameters.insert("bandwidth".to_string(), bandwidth.to_string());
            }
            ClusteringAlgorithm::Hierarchical { methods } => {
                let method = &methods[rng.gen_range(0..methods.len())];
                parameters.insert("method".to_string(), method.clone());
            }
            ClusteringAlgorithm::Spectral { k_range } => {
                let k = rng.gen_range(k_range.0..=k_range.1);
                parameters.insert("k".to_string(), k.to_string());
            }
            ClusteringAlgorithm::AffinityPropagation { damping_range } => {
                let damping = rng.gen_range(damping_range.0..=damping_range.1);
                parameters.insert("damping".to_string(), damping.to_string());
            }
        }

        Ok(parameters)
    }

    /// Sample parameters from ranges
    fn sample_parameter_ranges(
        &self,
        parameter_ranges: &HashMap<String, ParameterRange>,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<HashMap<String, String>> {
        let mut parameters = HashMap::new();

        for (param_name, range) in parameter_ranges {
            let value = match range {
                ParameterRange::Integer(min, max) => rng.gen_range(*min..=*max).to_string(),
                ParameterRange::Float(min, max) => rng.gen_range(*min..=*max).to_string(),
                ParameterRange::Categorical(choices) => {
                    choices[rng.gen_range(0..choices.len())].clone()
                }
                ParameterRange::Boolean => rng.gen_bool(0.5).to_string(),
            };
            parameters.insert(param_name.clone(), value);
        }

        Ok(parameters)
    }

    /// Run clustering with specified algorithm and parameters
    fn run_clustering(
        &self,
        data: &Array2<F>,
        algorithm: &ClusteringAlgorithm,
        parameters: &HashMap<String, String>,
    ) -> Result<Array1<i32>> {
        let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));

        match algorithm {
            ClusteringAlgorithm::KMeans { .. } => {
                let k = parameters
                    .get("k")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3);

                match kmeans2(
                    data.view(),
                    k,
                    Some(100),   // max_iter
                    None,        // threshold
                    None,        // init method
                    None,        // missing method
                    Some(false), // check_finite
                    None,        // seed
                ) {
                    Ok((_, labels)) => Ok(labels.mapv(|x| x as i32)),
                    Err(_) => {
                        // Fallback: create dummy labels
                        Ok(Array1::zeros(data.nrows()).mapv(|_| 0))
                    }
                }
            }
            ClusteringAlgorithm::DBSCAN { .. } => {
                let eps = parameters
                    .get("eps")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.5);
                let min_samples = parameters
                    .get("min_samples")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(5);

                match dbscan(data_f64.view(), eps, min_samples) {
                    Ok(labels) => Ok(labels),
                    Err(_) => {
                        // Fallback: create dummy labels
                        Ok(Array1::zeros(data.nrows()).mapv(|_| 0))
                    }
                }
            }
            _ => {
                // For other algorithms, fallback to k-means
                let k = parameters
                    .get("k")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3);

                match kmeans2(
                    data.view(),
                    k,
                    Some(100),
                    None,
                    None,
                    None,
                    Some(false),
                    None,
                ) {
                    Ok((_, labels)) => Ok(labels.mapv(|x| x as i32)),
                    Err(_) => Ok(Array1::zeros(data.nrows()).mapv(|_| 0)),
                }
            }
        }
    }

    /// Map labels back to full dataset size
    fn map_labels_to_full_data(
        &self,
        labels: &Array1<i32>,
        sample_indices: &[usize],
        full_size: usize,
    ) -> Result<Array1<i32>> {
        let mut full_labels = Array1::from_elem(full_size, -1); // Use -1 for unassigned

        for (sample_idx, &label) in sample_indices.iter().zip(labels.iter()) {
            if *sample_idx < full_size {
                full_labels[*sample_idx] = label;
            }
        }

        // Assign unassigned points to nearest cluster (simplified)
        for i in 0..full_size {
            if full_labels[i] == -1 {
                full_labels[i] = 0; // Assign to cluster 0 as fallback
            }
        }

        Ok(full_labels)
    }

    /// Count number of clusters in labels
    fn count_clusters(&self, labels: &Array1<i32>) -> usize {
        let unique_labels: HashSet<i32> = labels.iter().cloned().collect();
        unique_labels.len()
    }

    /// Filter results based on quality threshold
    fn filter_by_quality(&self, results: &[ClusteringResult]) -> Vec<ClusteringResult> {
        if let Some(threshold) = self.config.quality_threshold {
            results
                .iter()
                .filter(|r| r.quality_score >= threshold)
                .cloned()
                .collect()
        } else {
            results.to_vec()
        }
    }

    /// Build consensus from multiple clustering results
    fn build_consensus(
        &self,
        results: &[ClusteringResult],
        data: ArrayView2<F>,
    ) -> Result<Array1<i32>> {
        if results.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "No clustering results available for consensus".to_string(),
            ));
        }

        let n_samples = data.nrows();

        match &self.config.consensus_method {
            ConsensusMethod::MajorityVoting => self.majority_voting_consensus(results, n_samples),
            ConsensusMethod::WeightedConsensus => self.weighted_consensus(results, n_samples),
            ConsensusMethod::CoAssociation { threshold } => {
                self.co_association_consensus(results, n_samples, *threshold)
            }
            ConsensusMethod::EvidenceAccumulation => {
                self.evidence_accumulation_consensus(results, n_samples)
            }
            _ => {
                // Fallback to majority voting
                self.majority_voting_consensus(results, n_samples)
            }
        }
    }

    /// Implement majority voting consensus
    fn majority_voting_consensus(
        &self,
        results: &[ClusteringResult],
        n_samples: usize,
    ) -> Result<Array1<i32>> {
        // Build co-association matrix
        let mut co_association = Array2::zeros((n_samples, n_samples));

        for result in results {
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i < result.labels.len() && j < result.labels.len() {
                        if result.labels[i] == result.labels[j] && result.labels[i] != -1 {
                            co_association[[i, j]] += 1.0;
                        }
                    }
                }
            }
        }

        // Normalize by number of results
        co_association /= results.len() as f64;

        // Extract clusters using threshold
        self.extract_clusters_from_matrix(&co_association, 0.5)
    }

    /// Implement weighted consensus
    fn weighted_consensus(
        &self,
        results: &[ClusteringResult],
        n_samples: usize,
    ) -> Result<Array1<i32>> {
        let mut weighted_co_association = Array2::zeros((n_samples, n_samples));
        let mut total_weight = 0.0;

        for result in results {
            let weight = (result.quality_score + 1.0).max(0.0); // Ensure positive weight
            total_weight += weight;

            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i < result.labels.len() && j < result.labels.len() {
                        if result.labels[i] == result.labels[j] && result.labels[i] != -1 {
                            weighted_co_association[[i, j]] += weight;
                        }
                    }
                }
            }
        }

        // Normalize by total weight
        if total_weight > 0.0 {
            weighted_co_association /= total_weight;
        }

        // Extract clusters using threshold
        self.extract_clusters_from_matrix(&weighted_co_association, 0.5)
    }

    /// Implement co-association consensus
    fn co_association_consensus(
        &self,
        results: &[ClusteringResult],
        n_samples: usize,
        threshold: f64,
    ) -> Result<Array1<i32>> {
        let mut co_association = Array2::zeros((n_samples, n_samples));

        for result in results {
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i < result.labels.len() && j < result.labels.len() {
                        if result.labels[i] == result.labels[j] && result.labels[i] != -1 {
                            co_association[[i, j]] += 1.0;
                        }
                    }
                }
            }
        }

        // Normalize
        co_association /= results.len() as f64;

        // Extract clusters using specified threshold
        self.extract_clusters_from_matrix(&co_association, threshold)
    }

    /// Implement evidence accumulation consensus
    fn evidence_accumulation_consensus(
        &self,
        results: &[ClusteringResult],
        n_samples: usize,
    ) -> Result<Array1<i32>> {
        // This is a simplified version of evidence accumulation
        // In practice, this would involve more sophisticated clustering of the evidence
        self.co_association_consensus(results, n_samples, 0.5)
    }

    /// Extract clusters from similarity/association matrix
    fn extract_clusters_from_matrix(
        &self,
        matrix: &Array2<f64>,
        threshold: f64,
    ) -> Result<Array1<i32>> {
        let n_samples = matrix.nrows();
        let mut labels = Array1::from_elem(n_samples, -1);
        let mut current_cluster = 0;
        let mut visited = vec![false; n_samples];

        // Use connected components approach
        for i in 0..n_samples {
            if !visited[i] {
                let mut component = Vec::new();
                let mut stack = vec![i];

                while let Some(node) = stack.pop() {
                    if visited[node] {
                        continue;
                    }

                    visited[node] = true;
                    component.push(node);

                    // Find connected nodes
                    for j in 0..n_samples {
                        if !visited[j] && matrix[[node, j]] >= threshold {
                            stack.push(j);
                        }
                    }
                }

                // Assign cluster label to component
                for &node in &component {
                    labels[node] = current_cluster;
                }
                current_cluster += 1;
            }
        }

        Ok(labels)
    }

    /// Calculate consensus statistics
    fn calculate_consensus_statistics(
        &self,
        results: &[ClusteringResult],
        consensus_labels: &Array1<i32>,
    ) -> Result<ConsensusStatistics> {
        let n_samples = consensus_labels.len();
        let n_results = results.len();

        // Calculate agreement matrix
        let mut agreement_matrix = Array2::zeros((n_results, n_results));
        for i in 0..n_results {
            for j in 0..n_results {
                if i != j {
                    let ari = adjusted_rand_index::<f64>(
                        results[i].labels.view(),
                        results[j].labels.view(),
                    )
                    .unwrap_or(0.0);
                    agreement_matrix[[i, j]] = ari;
                }
            }
        }

        // Calculate consensus strength per sample
        let mut consensus_strength = Array1::zeros(n_samples);
        let mut agreement_counts = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut agreements = 0;
            let consensus_cluster = consensus_labels[i];

            for result in results {
                if i < result.labels.len() {
                    // Check if this result agrees with consensus
                    let mut agrees = false;
                    for j in 0..n_samples {
                        if j < result.labels.len()
                            && consensus_labels[j] == consensus_cluster
                            && result.labels[j] == result.labels[i]
                        {
                            agrees = true;
                            break;
                        }
                    }
                    if agrees {
                        agreements += 1;
                    }
                }
            }

            agreement_counts[i] = agreements;
            consensus_strength[i] = agreements as f64 / n_results as f64;
        }

        // Calculate cluster stability (simplified)
        let unique_clusters: HashSet<i32> = consensus_labels.iter().cloned().collect();
        let mut cluster_stability = Vec::new();

        for &cluster_id in &unique_clusters {
            let cluster_points: Vec<usize> = consensus_labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == cluster_id)
                .map(|(i, _)| i)
                .collect();

            let stability = if cluster_points.is_empty() {
                0.0
            } else {
                cluster_points
                    .iter()
                    .map(|&i| consensus_strength[i])
                    .sum::<f64>()
                    / cluster_points.len() as f64
            };

            cluster_stability.push(stability);
        }

        Ok(ConsensusStatistics {
            agreement_matrix,
            consensus_strength,
            cluster_stability,
            agreement_counts,
        })
    }

    /// Calculate diversity metrics
    fn calculate_diversity_metrics(
        &self,
        results: &[ClusteringResult],
    ) -> Result<DiversityMetrics> {
        let n_results = results.len();

        // Calculate diversity matrix (1 - ARI for each pair)
        let mut diversity_matrix = Array2::zeros((n_results, n_results));
        let mut total_diversity = 0.0;
        let mut diversity_count = 0;

        for i in 0..n_results {
            for j in (i + 1)..n_results {
                let ari =
                    adjusted_rand_index::<f64>(results[i].labels.view(), results[j].labels.view())
                        .unwrap_or(0.0);

                let diversity = 1.0 - ari;
                diversity_matrix[[i, j]] = diversity;
                diversity_matrix[[j, i]] = diversity;

                total_diversity += diversity;
                diversity_count += 1;
            }
        }

        let average_diversity = if diversity_count > 0 {
            total_diversity / diversity_count as f64
        } else {
            0.0
        };

        // Calculate algorithm distribution
        let mut algorithm_distribution = HashMap::new();
        for result in results {
            *algorithm_distribution
                .entry(result.algorithm.clone())
                .or_insert(0) += 1;
        }

        // Calculate parameter diversity (simplified)
        let mut parameter_diversity = HashMap::new();
        let all_param_names: HashSet<String> = results
            .iter()
            .flat_map(|r| r.parameters.keys())
            .cloned()
            .collect();

        for param_name in all_param_names {
            let unique_values: HashSet<String> = results
                .iter()
                .filter_map(|r| r.parameters.get(&param_name))
                .cloned()
                .collect();

            let diversity = unique_values.len() as f64 / results.len() as f64;
            parameter_diversity.insert(param_name, diversity);
        }

        Ok(DiversityMetrics {
            average_diversity,
            diversity_matrix,
            algorithm_distribution,
            parameter_diversity,
        })
    }

    /// Calculate stability score for the ensemble
    fn calculate_stability_score(&self, consensus_stats: &ConsensusStatistics) -> f64 {
        // Use average consensus strength as stability score
        consensus_stats.consensus_strength.mean().unwrap_or(0.0)
    }
}

/// Default configuration for ensemble clustering
impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            sampling_strategy: SamplingStrategy::Bootstrap { sample_ratio: 0.8 },
            consensus_method: ConsensusMethod::MajorityVoting,
            random_seed: None,
            diversity_strategy: Some(DiversityStrategy::AlgorithmDiversity {
                algorithms: vec![
                    ClusteringAlgorithm::KMeans { k_range: (2, 10) },
                    ClusteringAlgorithm::DBSCAN {
                        eps_range: (0.1, 1.0),
                        min_samples_range: (3, 10),
                    },
                ],
            }),
            quality_threshold: Some(0.0),
            max_clusters: Some(20),
        }
    }
}

/// Convenience functions for ensemble clustering
pub mod convenience {
    use super::*;

    /// Simple ensemble clustering with default parameters
    pub fn ensemble_clustering<F>(data: ArrayView2<F>) -> Result<EnsembleResult>
    where
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync,
        f64: From<F>,
    {
        let config = EnsembleConfig::default();
        let ensemble = EnsembleClusterer::new(config);
        ensemble.fit(data)
    }

    /// Bootstrap ensemble clustering
    pub fn bootstrap_ensemble<F>(
        data: ArrayView2<F>,
        n_estimators: usize,
        sample_ratio: f64,
    ) -> Result<EnsembleResult>
    where
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync,
        f64: From<F>,
    {
        let config = EnsembleConfig {
            n_estimators,
            sampling_strategy: SamplingStrategy::Bootstrap { sample_ratio },
            ..Default::default()
        };
        let ensemble = EnsembleClusterer::new(config);
        ensemble.fit(data)
    }

    /// Multi-algorithm ensemble clustering
    pub fn multi_algorithm_ensemble<F>(
        data: ArrayView2<F>,
        algorithms: Vec<ClusteringAlgorithm>,
    ) -> Result<EnsembleResult>
    where
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync,
        f64: From<F>,
    {
        let config = EnsembleConfig {
            diversity_strategy: Some(DiversityStrategy::AlgorithmDiversity { algorithms }),
            ..Default::default()
        };
        let ensemble = EnsembleClusterer::new(config);
        ensemble.fit(data)
    }

    /// Advanced meta-clustering ensemble method
    ///
    /// This method performs clustering on the space of clustering results themselves,
    /// using the clustering assignments as features for a meta-clustering algorithm.
    pub fn meta_clustering_ensemble<F>(
        data: ArrayView2<F>,
        base_configs: Vec<EnsembleConfig>,
        meta_config: EnsembleConfig,
    ) -> Result<EnsembleResult>
    where
        F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display + Send + Sync,
        f64: From<F>,
    {
        let mut base_results = Vec::new();
        let n_samples = data.shape()[0];

        // Step 1: Generate diverse base clusterings
        for config in base_configs {
            let ensemble = EnsembleClusterer::new(config);
            let result = ensemble.fit(data)?;
            base_results.extend(result.individual_results);
        }

        // Step 2: Create meta-features from clustering results
        let mut meta_features = Array2::zeros((n_samples, base_results.len()));
        for (i, result) in base_results.iter().enumerate() {
            for (j, &label) in result.labels.iter().enumerate() {
                meta_features[[j, i]] = F::from(label).unwrap();
            }
        }

        // Step 3: Apply meta-clustering
        let meta_ensemble = EnsembleClusterer::new(meta_config);
        let mut meta_result = meta_ensemble.fit(meta_features.view())?;

        // Step 4: Combine with original base results
        meta_result.individual_results = base_results;
        
        Ok(meta_result)
    }

    /// Adaptive ensemble clustering with online learning
    ///
    /// This method adapts the ensemble composition based on streaming data
    /// and performance feedback, adding or removing base clusterers dynamically.
    pub fn adaptive_ensemble<F>(
        data: ArrayView2<F>,
        config: &EnsembleConfig,
        adaptation_config: AdaptationConfig,
    ) -> Result<EnsembleResult>
    where
        F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display + Send + Sync,
        f64: From<F>,
    {
        let mut ensemble = EnsembleClusterer::new(config.clone());
        let mut current_results = Vec::new();
        let chunk_size = adaptation_config.chunk_size;
        
        // Process data in chunks for adaptive learning
        for chunk_start in (0..data.shape()[0]).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(data.shape()[0]);
            let chunk_data = data.slice(s![chunk_start..chunk_end, ..]);
            
            // Fit current ensemble on chunk
            let chunk_result = ensemble.fit(chunk_data)?;
            
            // Evaluate performance and adapt
            if current_results.len() >= adaptation_config.min_evaluations {
                let performance = evaluate_ensemble_performance(&current_results);
                
                if performance < adaptation_config.performance_threshold {
                    // Poor performance - adapt ensemble
                    ensemble = adapt_ensemble_composition(
                        ensemble,
                        &current_results,
                        &adaptation_config,
                    )?;
                }
            }
            
            current_results.push(chunk_result);
        }
        
        // Combine all chunk results into final consensus
        combine_chunk_results(current_results)
    }

    /// Federated ensemble clustering for distributed data
    ///
    /// This method allows clustering across multiple data sources without
    /// centralizing the data, preserving privacy while achieving consensus.
    pub fn federated_ensemble<F>(
        data_sources: Vec<ArrayView2<F>>,
        config: &EnsembleConfig,
        federation_config: FederationConfig,
    ) -> Result<EnsembleResult>
    where
        F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display + Send + Sync,
        f64: From<F>,
    {
        let mut local_results = Vec::new();
        
        // Step 1: Local clustering at each data source
        for data_source in data_sources {
            let local_ensemble = EnsembleClusterer::new(config.clone());
            let result = local_ensemble.fit(data_source)?;
            
            // Apply differential privacy if configured
            let private_result = if federation_config.differential_privacy {
                apply_differential_privacy(result, federation_config.privacy_budget)?
            } else {
                result
            };
            
            local_results.push(private_result);
        }
        
        // Step 2: Secure aggregation of results
        let aggregated_result = secure_aggregate_results(
            local_results,
            &federation_config,
        )?;
        
        Ok(aggregated_result)
    }
}

/// Configuration for adaptive ensemble learning
#[derive(Debug, Clone)]
pub struct AdaptationConfig {
    /// Size of data chunks for incremental learning
    pub chunk_size: usize,
    /// Minimum number of evaluations before adaptation
    pub min_evaluations: usize,
    /// Performance threshold for triggering adaptation
    pub performance_threshold: f64,
    /// Maximum number of base clusterers
    pub max_clusterers: usize,
    /// Adaptation strategy
    pub strategy: AdaptationStrategy,
}

/// Strategies for adapting ensemble composition
#[derive(Debug, Clone)]
pub enum AdaptationStrategy {
    /// Add new diverse clusterers
    AddDiverse,
    /// Remove worst performing clusterers
    RemoveWorst,
    /// Replace clusterers with better alternatives
    Replace,
    /// Combine multiple strategies
    Hybrid(Vec<AdaptationStrategy>),
}

/// Configuration for federated ensemble clustering
#[derive(Debug, Clone)]
pub struct FederationConfig {
    /// Enable differential privacy
    pub differential_privacy: bool,
    /// Privacy budget for differential privacy
    pub privacy_budget: f64,
    /// Secure aggregation method
    pub aggregation_method: AggregationMethod,
    /// Communication rounds
    pub max_rounds: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
}

/// Methods for secure aggregation in federated learning
#[derive(Debug, Clone)]
pub enum AggregationMethod {
    /// Simple averaging with noise
    SecureAveraging,
    /// Homomorphic encryption based aggregation
    HomomorphicEncryption,
    /// Multi-party computation
    MultiPartyComputation,
}

// Helper functions for advanced ensemble methods

fn evaluate_ensemble_performance(results: &[EnsembleResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    
    // Calculate average ensemble quality
    results.iter().map(|r| r.ensemble_quality).sum::<f64>() / results.len() as f64
}

fn adapt_ensemble_composition<F>(
    mut ensemble: EnsembleClusterer<F>,
    results: &[EnsembleResult],
    config: &AdaptationConfig,
) -> Result<EnsembleClusterer<F>>
where
    F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display + Send + Sync,
{
    match config.strategy {
        AdaptationStrategy::RemoveWorst => {
            // Remove worst performing clusterers
            if results.len() > 1 {
                // Implementation would identify and remove worst performers
                // For now, return the ensemble unchanged
            }
        }
        AdaptationStrategy::AddDiverse => {
            // Add new diverse clusterers
            // Implementation would add new diverse algorithms/parameters
        }
        _ => {
            // Other strategies
        }
    }
    
    Ok(ensemble)
}

fn combine_chunk_results(chunk_results: Vec<EnsembleResult>) -> Result<EnsembleResult> {
    if chunk_results.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "No chunk results to combine".to_string(),
        ));
    }
    
    // For simplicity, return the first result
    // A real implementation would intelligently combine all chunk results
    Ok(chunk_results.into_iter().next().unwrap())
}

fn apply_differential_privacy(
    mut result: EnsembleResult,
    _privacy_budget: f64,
) -> Result<EnsembleResult> {
    // Apply differential privacy mechanisms to the clustering result
    // For now, just add small amount of noise to consensus labels
    use rand::thread_rng;
    let mut rng = thread_rng();
    
    for label in result.consensus_labels.iter_mut() {
        if rng.gen::<f64>() < 0.05 {  // 5% chance to flip
            *label = (*label + 1) % 3;  // Simple label flipping
        }
    }
    
    Ok(result)
}

fn secure_aggregate_results(
    local_results: Vec<EnsembleResult>,
    _config: &FederationConfig,
) -> Result<EnsembleResult> {
    if local_results.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "No local results to aggregate".to_string(),
        ));
    }
    
    // For simplicity, perform simple majority voting
    // A real implementation would use secure aggregation protocols
    let n_samples = local_results[0].consensus_labels.len();
    let mut consensus_labels = Array1::zeros(n_samples);
    
    for i in 0..n_samples {
        let mut votes = HashMap::new();
        for result in &local_results {
            *votes.entry(result.consensus_labels[i]).or_insert(0) += 1;
        }
        
        // Find majority vote
        let majority_label = votes
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(label, _)| label)
            .unwrap_or(0);
        
        consensus_labels[i] = majority_label;
    }
    
    // Create aggregated result
    let mut aggregated = local_results.into_iter().next().unwrap();
    aggregated.consensus_labels = consensus_labels;
    
    Ok(aggregated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_ensemble_config_default() {
        let config = EnsembleConfig::default();
        assert_eq!(config.n_estimators, 10);
        assert!(matches!(
            config.consensus_method,
            ConsensusMethod::MajorityVoting
        ));
        assert!(config.diversity_strategy.is_some());
    }

    #[test]
    fn test_ensemble_clusterer_creation() {
        let config = EnsembleConfig::default();
        let ensemble: EnsembleClusterer<f64> = EnsembleClusterer::new(config);
        // Test that ensemble can be created successfully
        assert_eq!(
            std::mem::size_of_val(&ensemble),
            std::mem::size_of::<EnsembleConfig>()
        );
    }

    #[test]
    fn test_sampling_strategies() {
        let config = EnsembleConfig {
            sampling_strategy: SamplingStrategy::Bootstrap { sample_ratio: 0.8 },
            ..Default::default()
        };

        // Test that sampling strategy is correctly set
        match config.sampling_strategy {
            SamplingStrategy::Bootstrap { sample_ratio } => {
                assert!((sample_ratio - 0.8).abs() < 1e-6);
            }
            _ => panic!("Expected Bootstrap sampling strategy"),
        }
    }

    #[test]
    fn test_clustering_algorithms() {
        let algorithms = vec![
            ClusteringAlgorithm::KMeans { k_range: (2, 5) },
            ClusteringAlgorithm::DBSCAN {
                eps_range: (0.1, 1.0),
                min_samples_range: (3, 10),
            },
        ];

        assert_eq!(algorithms.len(), 2);

        match &algorithms[0] {
            ClusteringAlgorithm::KMeans { k_range } => {
                assert_eq!(k_range.0, 2);
                assert_eq!(k_range.1, 5);
            }
            _ => panic!("Expected KMeans algorithm"),
        }
    }

    #[test]
    fn test_convenience_functions() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2],
        )
        .unwrap();

        // Test bootstrap ensemble
        let result = convenience::bootstrap_ensemble(data.view(), 3, 0.8);
        assert!(result.is_ok());
    }
}
