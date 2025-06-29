//! Cluster stability assessment tools
//!
//! This module provides various methods for assessing the stability and
//! quality of clustering results, including bootstrap validation,
//! consensus clustering, and stability indices.

use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, FromPrimitive};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};
use crate::metrics::adjusted_rand_index;
use crate::vq::kmeans2;

/// Configuration for stability assessment
#[derive(Debug, Clone)]
pub struct StabilityConfig {
    /// Number of bootstrap iterations
    pub n_bootstrap: usize,
    /// Fraction of data to sample in each bootstrap iteration
    pub subsample_ratio: f64,
    /// Random seed for reproducible results
    pub random_seed: Option<u64>,
    /// Number of clustering algorithm runs per bootstrap
    pub n_runs_per_bootstrap: usize,
    /// Range of cluster numbers to test (for optimal k selection)
    pub k_range: Option<(usize, usize)>,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            n_bootstrap: 100,
            subsample_ratio: 0.8,
            random_seed: None,
            n_runs_per_bootstrap: 10,
            k_range: None,
        }
    }
}

/// Results of stability assessment
#[derive(Debug, Clone)]
pub struct StabilityResult<F: Float> {
    /// Stability scores for each tested configuration
    pub stability_scores: Vec<F>,
    /// Consensus clustering result
    pub consensus_labels: Option<Array1<usize>>,
    /// Optimal number of clusters (if k_range was provided)
    pub optimal_k: Option<usize>,
    /// Mean stability score across all bootstrap iterations
    pub mean_stability: F,
    /// Standard deviation of stability scores
    pub std_stability: F,
    /// Bootstrap stability matrix
    pub bootstrap_matrix: Array2<F>,
}

/// Bootstrap validation for clustering stability
///
/// This method assesses the stability of clustering by running the algorithm
/// on multiple bootstrap samples of the data and measuring the consistency
/// of the results.
pub struct BootstrapValidator<F: Float> {
    config: StabilityConfig,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display>
    BootstrapValidator<F>
{
    /// Create a new bootstrap validator
    pub fn new(config: StabilityConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Assess K-means clustering stability
    pub fn assess_kmeans_stability(
        &self,
        data: ArrayView2<F>,
        k: usize,
    ) -> Result<StabilityResult<F>> {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        if n_samples < 2 {
            return Err(ClusteringError::InvalidInput(
                "Need at least 2 samples for stability assessment".into(),
            ));
        }

        let subsample_size = ((n_samples as f64) * self.config.subsample_ratio) as usize;
        if subsample_size < k {
            return Err(ClusteringError::InvalidInput(
                "Subsample size must be at least k".into(),
            ));
        }

        let mut rng = match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => {
                // Use a default seed when no seed is provided
                rand::rngs::StdRng::seed_from_u64(42)
            }
        };

        let mut bootstrap_results = Vec::new();

        // Perform bootstrap iterations
        for _iteration in 0..self.config.n_bootstrap {
            // Create bootstrap sample
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            indices.truncate(subsample_size);

            let mut bootstrap_data = Array2::zeros((subsample_size, n_features));
            for (new_idx, &old_idx) in indices.iter().enumerate() {
                bootstrap_data.row_mut(new_idx).assign(&data.row(old_idx));
            }

            // Run clustering multiple times on this bootstrap sample
            let mut run_labels = Vec::new();
            for _run in 0..self.config.n_runs_per_bootstrap {
                let seed = rng.random::<u64>();

                match kmeans2(
                    bootstrap_data.view(),
                    k,
                    Some(100),   // max_iter
                    None,        // threshold
                    None,        // init method
                    None,        // missing method
                    Some(false), // check_finite
                    Some(seed),
                ) {
                    Ok((_, labels)) => {
                        let labels_usize: Array1<usize> = labels.mapv(|x| x);
                        run_labels.push(labels_usize);
                    }
                    Err(_) => {
                        // If clustering fails, create a dummy result
                        let dummy_labels = Array1::zeros(subsample_size);
                        run_labels.push(dummy_labels);
                    }
                }
            }

            bootstrap_results.push((indices, run_labels));
        }

        // Calculate stability metrics
        let stability_scores = self.calculate_stability_scores(&bootstrap_results)?;
        let mean_stability = stability_scores
            .iter()
            .copied()
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(stability_scores.len()).unwrap();

        let variance = stability_scores
            .iter()
            .map(|&x| {
                let diff = x - mean_stability;
                diff * diff
            })
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(stability_scores.len()).unwrap();
        let std_stability = variance.sqrt();

        // Create bootstrap stability matrix
        let bootstrap_matrix = self.create_bootstrap_matrix(&bootstrap_results, n_samples)?;

        Ok(StabilityResult {
            stability_scores,
            consensus_labels: None, // Would need consensus clustering implementation
            optimal_k: None,
            mean_stability,
            std_stability,
            bootstrap_matrix,
        })
    }

    /// Calculate stability scores from bootstrap results
    fn calculate_stability_scores(
        &self,
        bootstrap_results: &[(Vec<usize>, Vec<Array1<usize>>)],
    ) -> Result<Vec<F>> {
        let mut scores = Vec::new();

        for (_, run_labels) in bootstrap_results {
            if run_labels.len() < 2 {
                continue;
            }

            // Calculate pairwise ARI between runs
            let mut pairwise_aris = Vec::new();
            for i in 0..run_labels.len() {
                for j in (i + 1)..run_labels.len() {
                    let labels1 = run_labels[i].mapv(|x| x as i32);
                    let labels2 = run_labels[j].mapv(|x| x as i32);

                    match adjusted_rand_index::<F>(labels1.view(), labels2.view()) {
                        Ok(ari) => pairwise_aris.push(ari),
                        Err(_) => pairwise_aris.push(F::zero()),
                    }
                }
            }

            if !pairwise_aris.is_empty() {
                let mean_ari = pairwise_aris
                    .iter()
                    .copied()
                    .fold(F::zero(), |acc, x| acc + x)
                    / F::from(pairwise_aris.len()).unwrap();
                scores.push(mean_ari);
            }
        }

        Ok(scores)
    }

    /// Create bootstrap stability matrix
    fn create_bootstrap_matrix(
        &self,
        bootstrap_results: &[(Vec<usize>, Vec<Array1<usize>>)],
        n_samples: usize,
    ) -> Result<Array2<F>> {
        let mut co_occurrence_matrix: Array2<F> = Array2::zeros((n_samples, n_samples));
        let mut count_matrix: Array2<F> = Array2::zeros((n_samples, n_samples));

        for (indices, run_labels) in bootstrap_results {
            if run_labels.is_empty() {
                continue;
            }

            // Use the first run's labels for this bootstrap
            let labels = &run_labels[0];

            // Update co-occurrence matrix
            for (i, &idx_i) in indices.iter().enumerate() {
                for (j, &idx_j) in indices.iter().enumerate() {
                    if i != j {
                        count_matrix[[idx_i, idx_j]] = count_matrix[[idx_i, idx_j]] + F::one();

                        if labels[i] == labels[j] {
                            co_occurrence_matrix[[idx_i, idx_j]] =
                                co_occurrence_matrix[[idx_i, idx_j]] + F::one();
                        }
                    }
                }
            }
        }

        // Convert to probabilities
        let mut stability_matrix = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                if count_matrix[[i, j]] > F::zero() {
                    stability_matrix[[i, j]] = co_occurrence_matrix[[i, j]] / count_matrix[[i, j]];
                }
            }
        }

        Ok(stability_matrix)
    }
}

/// Consensus clustering for robust cluster identification
///
/// This method combines multiple clustering results to identify
/// stable cluster structures.
pub struct ConsensusClusterer<F: Float> {
    config: StabilityConfig,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + FromPrimitive + Debug + std::iter::Sum + std::fmt::Display> ConsensusClusterer<F> {
    /// Create a new consensus clusterer
    pub fn new(config: StabilityConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Find consensus clusters using multiple algorithm runs
    pub fn find_consensus_clusters(&self, data: ArrayView2<F>, k: usize) -> Result<Array1<usize>> {
        let n_samples = data.shape()[0];

        if n_samples < 2 {
            return Err(ClusteringError::InvalidInput(
                "Need at least 2 samples for consensus clustering".into(),
            ));
        }

        let mut rng = match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => {
                // Use a default seed when no seed is provided
                rand::rngs::StdRng::seed_from_u64(42)
            }
        };

        let mut all_labels = Vec::new();

        // Run clustering multiple times with different initializations
        for _run in 0..self.config.n_bootstrap {
            let seed = rng.random::<u64>();

            match kmeans2(
                data,
                k,
                Some(100),   // max_iter
                None,        // threshold
                None,        // init method
                None,        // missing method
                Some(false), // check_finite
                Some(seed),
            ) {
                Ok((_, labels)) => {
                    let labels_usize: Array1<usize> = labels.mapv(|x| x);
                    all_labels.push(labels_usize);
                }
                Err(_) => {
                    // Skip failed runs
                    continue;
                }
            }
        }

        if all_labels.is_empty() {
            return Err(ClusteringError::ComputationError(
                "All clustering runs failed".into(),
            ));
        }

        // Build consensus matrix
        let mut consensus_matrix = Array2::zeros((n_samples, n_samples));

        for labels in &all_labels {
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if labels[i] == labels[j] {
                        consensus_matrix[[i, j]] = consensus_matrix[[i, j]] + F::one();
                    }
                }
            }
        }

        // Normalize by number of runs
        let n_runs = F::from(all_labels.len()).unwrap();
        consensus_matrix.mapv_inplace(|x| x / n_runs);

        // Extract consensus clusters using threshold
        let threshold = F::from(0.5).unwrap();
        self.extract_consensus_clusters(&consensus_matrix, threshold, k)
    }

    /// Extract clusters from consensus matrix
    fn extract_consensus_clusters(
        &self,
        consensus_matrix: &Array2<F>,
        threshold: F,
        k: usize,
    ) -> Result<Array1<usize>> {
        let n_samples = consensus_matrix.shape()[0];
        let mut labels = Array1::from_elem(n_samples, usize::MAX); // Unassigned
        let mut current_cluster = 0;

        // Use a greedy approach to find dense consensus regions
        let mut unassigned: HashSet<usize> = (0..n_samples).collect();

        while current_cluster < k && !unassigned.is_empty() {
            // Find the pair with highest consensus that includes an unassigned point
            let mut best_consensus = F::zero();
            let mut best_seed = None;

            for &i in &unassigned {
                for &j in &unassigned {
                    if i != j && consensus_matrix[[i, j]] > best_consensus {
                        best_consensus = consensus_matrix[[i, j]];
                        best_seed = Some(i);
                    }
                }
            }

            if let Some(seed) = best_seed {
                // Grow cluster from seed
                let mut cluster_members = Vec::new();
                cluster_members.push(seed);

                // Add all points with high consensus to the seed
                for &candidate in &unassigned {
                    if candidate != seed && consensus_matrix[[seed, candidate]] >= threshold {
                        cluster_members.push(candidate);
                    }
                }

                // Assign cluster label
                for &member in &cluster_members {
                    labels[member] = current_cluster;
                    unassigned.remove(&member);
                }

                current_cluster += 1;
            } else {
                // No more good consensus pairs, assign remaining points to nearest cluster
                break;
            }
        }

        // Assign remaining unassigned points to the nearest existing cluster
        for &unassigned_point in &unassigned {
            let mut best_cluster = 0;
            let mut best_avg_consensus = F::zero();

            for cluster_id in 0..current_cluster {
                let mut total_consensus = F::zero();
                let mut count = 0;

                for i in 0..n_samples {
                    if labels[i] == cluster_id {
                        total_consensus = total_consensus + consensus_matrix[[unassigned_point, i]];
                        count += 1;
                    }
                }

                if count > 0 {
                    let avg_consensus = total_consensus / F::from(count).unwrap();
                    if avg_consensus > best_avg_consensus {
                        best_avg_consensus = avg_consensus;
                        best_cluster = cluster_id;
                    }
                }
            }

            labels[unassigned_point] = best_cluster;
        }

        Ok(labels)
    }
}

/// Optimal cluster number selection using stability criteria
pub struct OptimalKSelector<F: Float> {
    config: StabilityConfig,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display>
    OptimalKSelector<F>
{
    /// Create a new optimal k selector
    pub fn new(config: StabilityConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Find optimal number of clusters using stability gap statistic
    pub fn find_optimal_k(&self, data: ArrayView2<F>) -> Result<(usize, Vec<F>)> {
        let (k_min, k_max) = self.config.k_range.unwrap_or((2, 10));
        let mut stability_scores = Vec::new();

        for k in k_min..=k_max {
            let validator = BootstrapValidator::new(self.config.clone());
            match validator.assess_kmeans_stability(data, k) {
                Ok(result) => stability_scores.push(result.mean_stability),
                Err(_) => stability_scores.push(F::zero()),
            }
        }

        // Find k with maximum stability
        let mut best_k = k_min;
        let mut best_score = F::neg_infinity();

        for (i, &score) in stability_scores.iter().enumerate() {
            if score > best_score {
                best_score = score;
                best_k = k_min + i;
            }
        }

        Ok((best_k, stability_scores))
    }

    /// Find optimal k using gap statistic with reference distribution
    pub fn gap_statistic(&self, data: ArrayView2<F>) -> Result<(usize, Vec<F>)> {
        let (k_min, k_max) = self.config.k_range.unwrap_or((2, 10));
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        let mut gap_scores = Vec::new();

        // Find data bounds for reference distribution
        let mut min_vals = Array1::from_elem(n_features, F::infinity());
        let mut max_vals = Array1::from_elem(n_features, F::neg_infinity());

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = data[[i, j]];
                if val < min_vals[j] {
                    min_vals[j] = val;
                }
                if val > max_vals[j] {
                    max_vals[j] = val;
                }
            }
        }

        for k in k_min..=k_max {
            // Calculate log(W_k) for original data
            let original_wk = self.calculate_within_cluster_dispersion(data, k)?;
            let log_wk = original_wk.ln();

            // Calculate expected log(W_k) from reference distribution
            let mut reference_log_wks = Vec::new();
            let mut rng = match self.config.random_seed {
                Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
                None => {
                    // Use a default seed when no seed is provided
                    rand::rngs::StdRng::seed_from_u64(42)
                }
            };

            for _b in 0..self.config.n_bootstrap {
                // Generate reference data
                let mut reference_data = Array2::zeros((n_samples, n_features));
                for i in 0..n_samples {
                    for j in 0..n_features {
                        let range = max_vals[j] - min_vals[j];
                        let random_val =
                            min_vals[j] + range * F::from(rng.random::<f64>()).unwrap();
                        reference_data[[i, j]] = random_val;
                    }
                }

                let reference_wk =
                    self.calculate_within_cluster_dispersion(reference_data.view(), k)?;
                reference_log_wks.push(reference_wk.ln());
            }

            // Calculate gap statistic
            let expected_log_wk = reference_log_wks
                .iter()
                .copied()
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(reference_log_wks.len()).unwrap();
            let gap = expected_log_wk - log_wk;
            gap_scores.push(gap);
        }

        // Find optimal k (first k where gap(k) >= gap(k+1) - s_{k+1})
        let mut optimal_k = k_min;
        for i in 0..(gap_scores.len() - 1) {
            if gap_scores[i] >= gap_scores[i + 1] {
                optimal_k = k_min + i;
                break;
            }
        }

        Ok((optimal_k, gap_scores))
    }

    /// Calculate within-cluster dispersion W_k
    fn calculate_within_cluster_dispersion(&self, data: ArrayView2<F>, k: usize) -> Result<F> {
        // Run K-means clustering
        match kmeans2(
            data,
            k,
            Some(100),   // max_iter
            None,        // threshold
            None,        // init method
            None,        // missing method
            Some(false), // check_finite
            self.config.random_seed,
        ) {
            Ok((centroids, labels)) => {
                let mut total_dispersion = F::zero();

                for cluster_id in 0..k {
                    let mut cluster_dispersion = F::zero();
                    let mut cluster_size = 0;

                    // Calculate sum of squared distances within cluster
                    for i in 0..data.shape()[0] {
                        if labels[i] == cluster_id {
                            let mut sq_dist = F::zero();
                            for j in 0..data.shape()[1] {
                                let diff = data[[i, j]] - centroids[[cluster_id, j]];
                                sq_dist = sq_dist + diff * diff;
                            }
                            cluster_dispersion = cluster_dispersion + sq_dist;
                            cluster_size += 1;
                        }
                    }

                    // Normalize by cluster size
                    if cluster_size > 1 {
                        total_dispersion =
                            total_dispersion + cluster_dispersion / F::from(cluster_size).unwrap();
                    }
                }

                Ok(total_dispersion)
            }
            Err(e) => Err(e),
        }
    }
}

/// Advanced stability assessment methods
pub mod advanced {
    use super::*;
    use crate::metrics::{silhouette_score, mutual_info_score};
    use crate::ensemble::{EnsembleClusterer, EnsembleConfig};

    /// Cross-validation based stability assessment
    ///
    /// This method uses k-fold cross-validation to assess clustering stability
    /// by training on different subsets and testing on held-out data.
    pub struct CrossValidationStability<F: Float> {
        config: StabilityConfig,
        n_folds: usize,
        _phantom: std::marker::PhantomData<F>,
    }

    impl<F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display>
        CrossValidationStability<F>
    {
        /// Create a new cross-validation stability assessor
        pub fn new(config: StabilityConfig, n_folds: usize) -> Self {
            Self {
                config,
                n_folds,
                _phantom: std::marker::PhantomData,
            }
        }

        /// Assess clustering stability using cross-validation
        pub fn assess_stability(
            &self,
            data: ArrayView2<F>,
            k: usize,
        ) -> Result<StabilityResult<F>> {
            let n_samples = data.shape()[0];
            let fold_size = n_samples / self.n_folds;
            let mut stability_scores = Vec::new();
            let mut bootstrap_matrix = Array2::zeros((self.n_folds, self.n_folds));

            // Perform k-fold cross-validation
            for fold in 0..self.n_folds {
                let start_idx = fold * fold_size;
                let end_idx = if fold == self.n_folds - 1 {
                    n_samples
                } else {
                    (fold + 1) * fold_size
                };

                // Create training set (excluding current fold)
                let mut train_indices = Vec::new();
                for i in 0..n_samples {
                    if i < start_idx || i >= end_idx {
                        train_indices.push(i);
                    }
                }

                // Create training data
                let train_data = Array2::from_shape_fn(
                    (train_indices.len(), data.shape()[1]),
                    |(i, j)| data[[train_indices[i], j]]
                );

                // Run clustering on training data
                let (train_centroids, train_labels) = kmeans2(
                    train_data.view(),
                    k,
                    100,     // max_iter
                    F::from(1e-6).unwrap(), // tol
                    Some(42), // seed
                )?;

                // Assign test data to nearest centroids
                let test_labels = Array1::from_shape_fn(end_idx - start_idx, |i| {
                    let test_point = data.row(start_idx + i);
                    let mut min_dist = F::infinity();
                    let mut closest_cluster = 0;

                    for (cluster_id, centroid) in train_centroids.outer_iter().enumerate() {
                        let dist = test_point
                            .iter()
                            .zip(centroid.iter())
                            .map(|(a, b)| (*a - *b) * (*a - *b))
                            .sum::<F>()
                            .sqrt();

                        if dist < min_dist {
                            min_dist = dist;
                            closest_cluster = cluster_id;
                        }
                    }
                    closest_cluster
                });

                // Calculate stability score for this fold
                let stability = self.calculate_fold_stability(&test_labels, k)?;
                stability_scores.push(stability);
            }

            // Calculate mean and standard deviation
            let mean_stability = stability_scores.iter().sum::<F>() / F::from(stability_scores.len()).unwrap();
            let variance = stability_scores
                .iter()
                .map(|&s| (s - mean_stability) * (s - mean_stability))
                .sum::<F>() / F::from(stability_scores.len()).unwrap();
            let std_stability = variance.sqrt();

            Ok(StabilityResult {
                stability_scores,
                consensus_labels: None,
                optimal_k: None,
                mean_stability,
                std_stability,
                bootstrap_matrix,
            })
        }

        fn calculate_fold_stability(&self, labels: &Array1<usize>, k: usize) -> Result<F> {
            // Calculate intra-cluster cohesion
            let mut cluster_cohesion = F::zero();
            let mut total_pairs = 0;

            for cluster_id in 0..k {
                let cluster_members: Vec<_> = labels
                    .iter()
                    .enumerate()
                    .filter(|(_, &label)| label == cluster_id)
                    .map(|(idx, _)| idx)
                    .collect();

                let cluster_size = cluster_members.len();
                if cluster_size > 1 {
                    let pairs = cluster_size * (cluster_size - 1) / 2;
                    cluster_cohesion = cluster_cohesion + F::from(pairs).unwrap();
                    total_pairs += pairs;
                }
            }

            if total_pairs == 0 {
                Ok(F::zero())
            } else {
                Ok(cluster_cohesion / F::from(total_pairs).unwrap())
            }
        }
    }

    /// Perturbation-based stability assessment
    ///
    /// This method assesses stability by introducing controlled perturbations
    /// to the data and measuring how much the clustering results change.
    pub struct PerturbationStability<F: Float> {
        config: StabilityConfig,
        perturbation_types: Vec<PerturbationType>,
        _phantom: std::marker::PhantomData<F>,
    }

    /// Types of perturbations for stability testing
    #[derive(Debug, Clone)]
    pub enum PerturbationType {
        /// Add Gaussian noise
        GaussianNoise { std_dev: f64 },
        /// Remove random samples
        SampleRemoval { removal_rate: f64 },
        /// Add random features
        FeatureNoise { noise_level: f64 },
        /// Outlier injection
        OutlierInjection { outlier_rate: f64, outlier_magnitude: f64 },
    }

    impl<F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display>
        PerturbationStability<F>
    {
        /// Create a new perturbation stability assessor
        pub fn new(config: StabilityConfig, perturbation_types: Vec<PerturbationType>) -> Self {
            Self {
                config,
                perturbation_types,
                _phantom: std::marker::PhantomData,
            }
        }

        /// Assess clustering stability under perturbations
        pub fn assess_stability(
            &self,
            data: ArrayView2<F>,
            k: usize,
        ) -> Result<StabilityResult<F>> {
            let mut all_stability_scores = Vec::new();
            let mut rng = rand::thread_rng();

            // Get baseline clustering
            let (baseline_centroids, baseline_labels) = kmeans2(
                data,
                k,
                100,     // max_iter
                F::from(1e-6).unwrap(), // tol
                Some(42), // seed
            )?;

            // Test each perturbation type
            for perturbation in &self.perturbation_types {
                let mut perturbation_scores = Vec::new();

                for _ in 0..self.config.n_bootstrap {
                    // Apply perturbation
                    let perturbed_data = self.apply_perturbation(data, perturbation, &mut rng)?;

                    // Run clustering on perturbed data
                    let (_, perturbed_labels) = kmeans2(
                        perturbed_data.view(),
                        k,
                        100,     // max_iter
                        F::from(1e-6).unwrap(), // tol
                        None,    // random seed
                    )?;

                    // Calculate similarity to baseline
                    let similarity = self.calculate_label_similarity(&baseline_labels, &perturbed_labels)?;
                    perturbation_scores.push(similarity);
                }

                all_stability_scores.extend(perturbation_scores);
            }

            // Calculate overall statistics
            let mean_stability = all_stability_scores.iter().sum::<F>() / F::from(all_stability_scores.len()).unwrap();
            let variance = all_stability_scores
                .iter()
                .map(|&s| (s - mean_stability) * (s - mean_stability))
                .sum::<F>() / F::from(all_stability_scores.len()).unwrap();
            let std_stability = variance.sqrt();

            let bootstrap_matrix = Array2::zeros((self.config.n_bootstrap, self.perturbation_types.len()));

            Ok(StabilityResult {
                stability_scores: all_stability_scores,
                consensus_labels: None,
                optimal_k: None,
                mean_stability,
                std_stability,
                bootstrap_matrix,
            })
        }

        fn apply_perturbation(
            &self,
            data: ArrayView2<F>,
            perturbation: &PerturbationType,
            rng: &mut impl Rng,
        ) -> Result<Array2<F>> {
            let mut perturbed = data.to_owned();

            match perturbation {
                PerturbationType::GaussianNoise { std_dev } => {
                    for elem in perturbed.iter_mut() {
                        let noise = rng.gen::<f64>() * std_dev;
                        *elem = *elem + F::from(noise).unwrap();
                    }
                }
                PerturbationType::SampleRemoval { removal_rate } => {
                    let n_samples = data.shape()[0];
                    let n_remove = (n_samples as f64 * removal_rate) as usize;
                    let mut indices: Vec<_> = (0..n_samples).collect();
                    indices.shuffle(rng);
                    indices.truncate(n_samples - n_remove);
                    indices.sort();

                    let mut new_data = Array2::zeros((indices.len(), data.shape()[1]));
                    for (new_i, &old_i) in indices.iter().enumerate() {
                        new_data.row_mut(new_i).assign(&data.row(old_i));
                    }
                    perturbed = new_data;
                }
                PerturbationType::FeatureNoise { noise_level } => {
                    for elem in perturbed.iter_mut() {
                        let noise = (rng.gen::<f64>() - 0.5) * 2.0 * noise_level;
                        *elem = *elem + F::from(noise).unwrap();
                    }
                }
                PerturbationType::OutlierInjection { outlier_rate, outlier_magnitude } => {
                    let n_samples = data.shape()[0];
                    let n_outliers = (n_samples as f64 * outlier_rate) as usize;
                    
                    for _ in 0..n_outliers {
                        let sample_idx = rng.gen_range(0..n_samples);
                        let feature_idx = rng.gen_range(0..data.shape()[1]);
                        let outlier_value = rng.gen::<f64>() * outlier_magnitude;
                        perturbed[[sample_idx, feature_idx]] = F::from(outlier_value).unwrap();
                    }
                }
            }

            Ok(perturbed)
        }

        fn calculate_label_similarity(
            &self,
            labels1: &Array1<usize>,
            labels2: &Array1<usize>,
        ) -> Result<F> {
            if labels1.len() != labels2.len() {
                return Ok(F::zero());
            }

            // Convert to i32 for ARI calculation
            let labels1_i32: Array1<i32> = labels1.mapv(|x| x as i32);
            let labels2_i32: Array1<i32> = labels2.mapv(|x| x as i32);

            // Use adjusted rand index as similarity measure
            let ari: f64 = adjusted_rand_index(labels1_i32.view(), labels2_i32.view())?;
            Ok(F::from(ari).unwrap())
        }
    }

    /// Multi-scale stability assessment
    ///
    /// This method assesses stability across different data scales and resolutions
    /// to understand how clustering behaves at different granularities.
    pub struct MultiScaleStability<F: Float> {
        config: StabilityConfig,
        scale_factors: Vec<f64>,
        _phantom: std::marker::PhantomData<F>,
    }

    impl<F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display>
        MultiScaleStability<F>
    {
        /// Create a new multi-scale stability assessor
        pub fn new(config: StabilityConfig, scale_factors: Vec<f64>) -> Self {
            Self {
                config,
                scale_factors,
                _phantom: std::marker::PhantomData,
            }
        }

        /// Assess clustering stability across multiple scales
        pub fn assess_stability(
            &self,
            data: ArrayView2<F>,
            k_range: (usize, usize),
        ) -> Result<Vec<StabilityResult<F>>> {
            let mut results = Vec::new();

            for &scale_factor in &self.scale_factors {
                // Scale the data
                let scaled_data = data.mapv(|x| x * F::from(scale_factor).unwrap());

                // Assess stability at this scale for different k values
                for k in k_range.0..=k_range.1 {
                    let validator = BootstrapValidator::new(self.config.clone());
                    let stability_result = validator.assess_kmeans_stability(scaled_data.view(), k)?;
                    results.push(stability_result);
                }
            }

            Ok(results)
        }

        /// Find the most stable scale and cluster count combination
        pub fn find_optimal_scale_and_k(
            &self,
            data: ArrayView2<F>,
            k_range: (usize, usize),
        ) -> Result<(f64, usize, F)> {
            let results = self.assess_stability(data, k_range)?;
            
            let mut best_scale = self.scale_factors[0];
            let mut best_k = k_range.0;
            let mut best_stability = F::neg_infinity();

            let mut result_idx = 0;
            for &scale_factor in &self.scale_factors {
                for k in k_range.0..=k_range.1 {
                    if result_idx < results.len() {
                        let stability = results[result_idx].mean_stability;
                        if stability > best_stability {
                            best_stability = stability;
                            best_scale = scale_factor;
                            best_k = k;
                        }
                        result_idx += 1;
                    }
                }
            }

            Ok((best_scale, best_k, best_stability))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_stability_config_default() {
        let config = StabilityConfig::default();
        assert_eq!(config.n_bootstrap, 100);
        assert_eq!(config.subsample_ratio, 0.8);
        assert_eq!(config.n_runs_per_bootstrap, 10);
        assert!(config.random_seed.is_none());
    }

    #[test]
    fn test_bootstrap_validator() {
        let data =
            Array2::from_shape_vec((20, 2), (0..40).map(|i| i as f64 / 10.0).collect()).unwrap();

        let config = StabilityConfig {
            n_bootstrap: 5,
            subsample_ratio: 0.8,
            n_runs_per_bootstrap: 3,
            random_seed: Some(42),
            k_range: None,
        };

        let validator = BootstrapValidator::new(config);
        let result = validator.assess_kmeans_stability(data.view(), 2);

        assert!(result.is_ok());
        let stability_result = result.unwrap();
        assert!(stability_result.mean_stability >= 0.0);
        assert!(stability_result.mean_stability <= 1.0);
        assert_eq!(stability_result.bootstrap_matrix.shape(), &[20, 20]);
    }

    #[test]
    fn test_consensus_clusterer() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2],
        )
        .unwrap();

        let config = StabilityConfig {
            n_bootstrap: 10,
            random_seed: Some(42),
            ..Default::default()
        };

        let consensus = ConsensusClusterer::new(config);
        let result = consensus.find_consensus_clusters(data.view(), 2);

        assert!(result.is_ok());
        let labels = result.unwrap();
        assert_eq!(labels.len(), 6);

        // Check that we have exactly 2 clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().copied().collect();
        assert_eq!(unique_labels.len(), 2);
    }

    #[test]
    fn test_optimal_k_selector() {
        let data = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.2, // Cluster 1
                5.0, 5.0, 5.1, 5.1, 5.2, 5.2, // Cluster 2
                10.0, 10.0, 10.1, 10.1, 10.2, 10.2, // Cluster 3
                15.0, 15.0, 15.1, 15.1, 15.2, 15.2, // Cluster 4
            ],
        )
        .unwrap();

        let config = StabilityConfig {
            k_range: Some((2, 5)),
            n_bootstrap: 5,
            random_seed: Some(42),
            ..Default::default()
        };

        let selector = OptimalKSelector::new(config);
        let result = selector.find_optimal_k(data.view());

        assert!(result.is_ok());
        let (optimal_k, scores) = result.unwrap();
        assert!((2..=5).contains(&optimal_k));
        assert_eq!(scores.len(), 4); // k=2,3,4,5
    }

    #[test]
    fn test_gap_statistic() {
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2, 5.3, 5.3,
            ],
        )
        .unwrap();

        let config = StabilityConfig {
            k_range: Some((2, 4)),
            n_bootstrap: 5,
            random_seed: Some(42),
            ..Default::default()
        };

        let selector = OptimalKSelector::new(config);
        let result = selector.gap_statistic(data.view());

        assert!(result.is_ok());
        let (optimal_k, gap_scores) = result.unwrap();
        assert!((2..=4).contains(&optimal_k));
        assert_eq!(gap_scores.len(), 3); // k=2,3,4
    }
}
