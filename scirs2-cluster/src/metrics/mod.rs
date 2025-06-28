//! Clustering evaluation metrics
//!
//! This module provides metrics for evaluating clustering algorithms performance:
//! - Silhouette coefficient for measuring cluster cohesion and separation
//! - Davies-Bouldin index for evaluating cluster separation
//! - Calinski-Harabasz index for measuring between-cluster vs within-cluster variance

mod silhouette;
pub use silhouette::{silhouette_samples, silhouette_score};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Davies-Bouldin score for clustering evaluation.
///
/// The Davies-Bouldin index measures the average similarity between clusters,
/// where similarity is the ratio of within-cluster and between-cluster distances.
/// A lower score indicates better clustering.
///
/// # Arguments
///
/// * `data` - Input data (n_samples x n_features)
/// * `labels` - Cluster labels for each sample
///
/// # Returns
///
/// The Davies-Bouldin score (lower is better)
///
/// # Example
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_cluster::metrics::davies_bouldin_score;
///
/// let data = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,
///     0.1, 0.1,
///     5.0, 5.0,
///     5.1, 5.1,
/// ]).unwrap();
/// let labels = Array1::from_vec(vec![0, 0, 1, 1]);
///
/// let score = davies_bouldin_score(data.view(), labels.view()).unwrap();
/// assert!(score < 0.5);  // Should be low for well-separated clusters
/// ```
pub fn davies_bouldin_score<F>(data: ArrayView2<F>, labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + PartialOrd + 'static,
{
    if data.shape()[0] != labels.shape()[0] {
        return Err(ClusteringError::InvalidInput(
            "Data and labels must have the same number of samples".to_string(),
        ));
    }

    // Find unique cluster labels
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if label >= 0 && !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    let n_clusters = unique_labels.len();

    if n_clusters < 2 {
        return Err(ClusteringError::InvalidInput(
            "Davies-Bouldin score requires at least 2 clusters".to_string(),
        ));
    }

    // Compute cluster centers
    let mut centers = Array2::<F>::zeros((n_clusters, data.shape()[1]));
    let mut cluster_sizes = vec![0; n_clusters];

    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            let cluster_idx = unique_labels.iter().position(|&l| l == label).unwrap();
            centers
                .row_mut(cluster_idx)
                .scaled_add(F::one(), &data.row(i));
            cluster_sizes[cluster_idx] += 1;
        }
    }

    // Normalize to get averages
    for (i, &size) in cluster_sizes.iter().enumerate() {
        if size > 0 {
            centers
                .row_mut(i)
                .mapv_inplace(|x| x / F::from(size).unwrap());
        }
    }

    // Compute within-cluster scatter
    let mut scatter = vec![F::zero(); n_clusters];
    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            let cluster_idx = unique_labels.iter().position(|&l| l == label).unwrap();
            let center = centers.row(cluster_idx);
            let diff = &data.row(i) - &center;
            let distance = diff.dot(&diff).sqrt();
            scatter[cluster_idx] = scatter[cluster_idx] + distance;
        }
    }

    // Compute average within-cluster scatter
    for (i, &size) in cluster_sizes.iter().enumerate() {
        if size > 0 {
            scatter[i] = scatter[i] / F::from(size).unwrap();
        }
    }

    // Compute Davies-Bouldin index
    let mut db_index = F::zero();

    for i in 0..n_clusters {
        let mut max_ratio = F::zero();

        for j in 0..n_clusters {
            if i != j {
                let between_distance = (&centers.row(i) - &centers.row(j))
                    .mapv(|x| x * x)
                    .sum()
                    .sqrt();

                if between_distance > F::zero() {
                    let ratio = (scatter[i] + scatter[j]) / between_distance;
                    if ratio > max_ratio {
                        max_ratio = ratio;
                    }
                }
            }
        }

        db_index = db_index + max_ratio;
    }

    db_index = db_index / F::from(n_clusters).unwrap();
    Ok(db_index)
}

/// Calinski-Harabasz score for clustering evaluation.
///
/// Also known as the Variance Ratio Criterion, this score computes the ratio of
/// the sum of between-clusters dispersion to the within-cluster dispersion.
/// A higher score indicates better defined clusters.
///
/// # Arguments
///
/// * `data` - Input data (n_samples x n_features)
/// * `labels` - Cluster labels for each sample
///
/// # Returns
///
/// The Calinski-Harabasz score (higher is better)
///
/// # Example
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_cluster::metrics::calinski_harabasz_score;
///
/// let data = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,
///     0.1, 0.1,
///     5.0, 5.0,
///     5.1, 5.1,
/// ]).unwrap();
/// let labels = Array1::from_vec(vec![0, 0, 1, 1]);
///
/// let score = calinski_harabasz_score(data.view(), labels.view()).unwrap();
/// assert!(score > 50.0);  // Should be high for well-separated clusters
/// ```
pub fn calinski_harabasz_score<F>(data: ArrayView2<F>, labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + PartialOrd + 'static,
{
    if data.shape()[0] != labels.shape()[0] {
        return Err(ClusteringError::InvalidInput(
            "Data and labels must have the same number of samples".to_string(),
        ));
    }

    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    // Find unique cluster labels
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if label >= 0 && !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    let n_clusters = unique_labels.len();

    if n_clusters < 2 {
        return Err(ClusteringError::InvalidInput(
            "Calinski-Harabasz score requires at least 2 clusters".to_string(),
        ));
    }

    if n_clusters >= n_samples {
        return Err(ClusteringError::InvalidInput(
            "Number of clusters must be less than number of samples".to_string(),
        ));
    }

    // Compute overall mean
    let mut overall_mean = Array1::<F>::zeros(n_features);
    let mut valid_samples = 0;

    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            overall_mean.scaled_add(F::one(), &data.row(i));
            valid_samples += 1;
        }
    }

    overall_mean.mapv_inplace(|x| x / F::from(valid_samples).unwrap());

    // Compute cluster centers and sizes
    let mut centers = Array2::<F>::zeros((n_clusters, n_features));
    let mut cluster_sizes = vec![0; n_clusters];

    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            let cluster_idx = unique_labels.iter().position(|&l| l == label).unwrap();
            centers
                .row_mut(cluster_idx)
                .scaled_add(F::one(), &data.row(i));
            cluster_sizes[cluster_idx] += 1;
        }
    }

    // Normalize to get averages
    for (i, &size) in cluster_sizes.iter().enumerate() {
        if size > 0 {
            centers
                .row_mut(i)
                .mapv_inplace(|x| x / F::from(size).unwrap());
        }
    }

    // Compute between-group sum of squares (SSB)
    let mut ssb = F::zero();
    for (i, &size) in cluster_sizes.iter().enumerate() {
        if size > 0 {
            let diff = &centers.row(i) - &overall_mean;
            ssb = ssb + F::from(size).unwrap() * diff.dot(&diff);
        }
    }

    // Compute within-group sum of squares (SSW)
    let mut ssw = F::zero();
    for (i, &label) in labels.iter().enumerate() {
        if label >= 0 {
            let cluster_idx = unique_labels.iter().position(|&l| l == label).unwrap();
            let diff = &data.row(i) - &centers.row(cluster_idx);
            ssw = ssw + diff.dot(&diff);
        }
    }

    // Calculate score
    if ssw == F::zero() {
        return Ok(F::infinity());
    }

    let score = (ssb / ssw) * F::from(valid_samples - n_clusters).unwrap()
        / F::from(n_clusters - 1).unwrap();

    Ok(score)
}

/// Mean silhouette coefficient over all samples.
///
/// Convenience wrapper that computes the mean silhouette coefficient using
/// Euclidean distance metric by default.
///
/// # Example
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_cluster::metrics::mean_silhouette_score;
///
/// let data = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,
///     0.1, 0.1,
///     5.0, 5.0,
///     5.1, 5.1,
/// ]).unwrap();
/// let labels = Array1::from_vec(vec![0, 0, 1, 1]);
///
/// let score = mean_silhouette_score(data.view(), labels.view()).unwrap();
/// ```
pub fn mean_silhouette_score<F>(data: ArrayView2<F>, labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + 'static,
{
    silhouette_score(data, labels)
}

/// Adjusted Rand Index for comparing two clusterings.
///
/// The Adjusted Rand Index (ARI) is a measure of the similarity between two data clusterings,
/// adjusted for chance. It has a value between -1 and 1, where:
/// - 1 indicates perfect agreement
/// - 0 indicates agreement no better than random chance
/// - Negative values indicate agreement worse than random chance
///
/// # Arguments
///
/// * `labels_true` - Ground truth cluster labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
///
/// The Adjusted Rand Index score
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_cluster::metrics::adjusted_rand_index;
///
/// let labels_true = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
/// let labels_pred = Array1::from_vec(vec![0, 0, 2, 2, 1, 1]);
///
/// let ari: f64 = adjusted_rand_index(labels_true.view(), labels_pred.view()).unwrap();
/// assert!(ari > 0.0);  // Should be positive for similar clusterings
/// ```
pub fn adjusted_rand_index<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "Labels arrays must have the same length".to_string(),
        ));
    }

    let n = labels_true.len();
    if n == 0 {
        return Err(ClusteringError::InvalidInput(
            "Empty labels arrays".to_string(),
        ));
    }

    // Build contingency table
    let mut true_labels = std::collections::HashSet::new();
    let mut pred_labels = std::collections::HashSet::new();

    for &label in labels_true.iter() {
        true_labels.insert(label);
    }
    for &label in labels_pred.iter() {
        pred_labels.insert(label);
    }

    let n_true = true_labels.len();
    let n_pred = pred_labels.len();

    // Create mapping from labels to indices
    let true_label_map: std::collections::HashMap<_, _> = true_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();
    let pred_label_map: std::collections::HashMap<_, _> = pred_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    // Build contingency table
    let mut contingency = Array2::<usize>::zeros((n_true, n_pred));
    for i in 0..n {
        let true_idx = true_label_map[&labels_true[i]];
        let pred_idx = pred_label_map[&labels_pred[i]];
        contingency[[true_idx, pred_idx]] += 1;
    }

    // Calculate sums
    let sum_comb_c = contingency
        .iter()
        .map(|&n_ij| {
            if n_ij >= 2 {
                (n_ij * (n_ij - 1)) / 2
            } else {
                0
            }
        })
        .sum::<usize>();

    let sum_a = contingency
        .sum_axis(Axis(1))
        .iter()
        .map(|&n_i| if n_i >= 2 { (n_i * (n_i - 1)) / 2 } else { 0 })
        .sum::<usize>();

    let sum_b = contingency
        .sum_axis(Axis(0))
        .iter()
        .map(|&n_j| if n_j >= 2 { (n_j * (n_j - 1)) / 2 } else { 0 })
        .sum::<usize>();

    let n_choose_2 = if n >= 2 { (n * (n - 1)) / 2 } else { 0 };

    // Calculate expected index
    let expected_index =
        F::from(sum_a).unwrap() * F::from(sum_b).unwrap() / F::from(n_choose_2).unwrap();
    let max_index = (F::from(sum_a).unwrap() + F::from(sum_b).unwrap()) / F::from(2.0).unwrap();
    let index = F::from(sum_comb_c).unwrap();

    // Handle edge cases
    if max_index == expected_index {
        return Ok(F::zero());
    }

    // Calculate ARI
    let ari = (index - expected_index) / (max_index - expected_index);
    Ok(ari)
}

/// Normalized Mutual Information (NMI) for comparing two clusterings.
///
/// Normalized Mutual Information is a normalization of the Mutual Information (MI) score
/// to scale the results between 0 (no mutual information) and 1 (perfect correlation).
///
/// # Arguments
///
/// * `labels_true` - Ground truth cluster labels
/// * `labels_pred` - Predicted cluster labels
/// * `average_method` - Method to compute the normalizer ('geometric', 'arithmetic', 'min', 'max')
///
/// # Returns
///
/// The Normalized Mutual Information score
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_cluster::metrics::normalized_mutual_info;
///
/// let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
/// let labels_pred = Array1::from_vec(vec![0, 0, 1, 1]);
///
/// let nmi: f64 = normalized_mutual_info(labels_true.view(), labels_pred.view(), "arithmetic").unwrap();
/// assert!((nmi - 1.0).abs() < 1e-6);  // Perfect agreement
/// ```
pub fn normalized_mutual_info<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
    average_method: &str,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "Labels arrays must have the same length".to_string(),
        ));
    }

    let n = labels_true.len();
    if n == 0 {
        return Ok(F::one());
    }

    // Compute mutual information
    let mi = mutual_info::<F>(labels_true, labels_pred)?;

    // Compute entropies
    let h_true = entropy::<F>(labels_true)?;
    let h_pred = entropy::<F>(labels_pred)?;

    // Handle edge cases
    if h_true == F::zero() && h_pred == F::zero() {
        return Ok(F::one());
    }

    // Compute normalization
    let normalizer = match average_method {
        "arithmetic" => (h_true + h_pred) / F::from(2.0).unwrap(),
        "geometric" => (h_true * h_pred).sqrt(),
        "min" => h_true.min(h_pred),
        "max" => h_true.max(h_pred),
        _ => {
            return Err(ClusteringError::InvalidInput(
                "Invalid average method. Use 'arithmetic', 'geometric', 'min', or 'max'"
                    .to_string(),
            ))
        }
    };

    if normalizer == F::zero() {
        return Ok(F::zero());
    }

    Ok(mi / normalizer)
}

/// Compute mutual information between two label assignments.
fn mutual_info<F>(labels_true: ArrayView1<i32>, labels_pred: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = labels_true.len() as f64;
    let contingency = build_contingency_matrix(labels_true, labels_pred)?;

    let mut mi = F::zero();
    let n_rows = contingency.shape()[0];
    let n_cols = contingency.shape()[1];

    // Compute marginal sums
    let row_sums = contingency.sum_axis(Axis(1));
    let col_sums = contingency.sum_axis(Axis(0));

    for i in 0..n_rows {
        for j in 0..n_cols {
            let n_ij = contingency[[i, j]] as f64;
            if n_ij > 0.0 {
                let n_i = row_sums[i] as f64;
                let n_j = col_sums[j] as f64;
                let term = n_ij / n * (n_ij / (n_i * n_j / n)).ln();
                mi = mi + F::from(term).unwrap();
            }
        }
    }

    Ok(mi)
}

/// Compute entropy of a label assignment.
fn entropy<F>(labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = labels.len() as f64;
    let mut label_counts = std::collections::HashMap::new();

    for &label in labels.iter() {
        *label_counts.entry(label).or_insert(0) += 1;
    }

    let mut h = F::zero();
    for &count in label_counts.values() {
        if count > 0 {
            let p = count as f64 / n;
            h = h - F::from(p * p.ln()).unwrap();
        }
    }

    Ok(h)
}

/// Build contingency matrix for two label assignments.
fn build_contingency_matrix(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<Array2<usize>> {
    let mut true_labels = std::collections::BTreeSet::new();
    let mut pred_labels = std::collections::BTreeSet::new();

    for &label in labels_true.iter() {
        true_labels.insert(label);
    }
    for &label in labels_pred.iter() {
        pred_labels.insert(label);
    }

    let true_label_map: std::collections::HashMap<_, _> = true_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();
    let pred_label_map: std::collections::HashMap<_, _> = pred_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    let mut contingency = Array2::<usize>::zeros((true_labels.len(), pred_labels.len()));
    for i in 0..labels_true.len() {
        let true_idx = true_label_map[&labels_true[i]];
        let pred_idx = pred_label_map[&labels_pred[i]];
        contingency[[true_idx, pred_idx]] += 1;
    }

    Ok(contingency)
}

/// Homogeneity, completeness and V-measure metrics for clustering evaluation.
///
/// These metrics are useful to evaluate the quality of clustering when ground truth is available.
/// - Homogeneity: each cluster contains only members of a single class.
/// - Completeness: all members of a given class are assigned to the same cluster.
/// - V-measure: harmonic mean of homogeneity and completeness.
///
/// All scores are between 0 and 1, where 1 indicates perfect clustering.
///
/// # Arguments
///
/// * `labels_true` - Ground truth cluster labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
///
/// Tuple of (homogeneity, completeness, v_measure)
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_cluster::metrics::homogeneity_completeness_v_measure;
///
/// let labels_true = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
/// let labels_pred = Array1::from_vec(vec![0, 0, 1, 1, 1, 1]);
///
/// let (h, c, v): (f64, f64, f64) = homogeneity_completeness_v_measure(labels_true.view(), labels_pred.view()).unwrap();
/// assert!(h > 0.5);  // Good homogeneity
/// assert!(c > 0.9);  // High completeness (all members of each class in single clusters)
/// ```
pub fn homogeneity_completeness_v_measure<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<(F, F, F)>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "Labels arrays must have the same length".to_string(),
        ));
    }

    let n = labels_true.len();
    if n == 0 {
        return Ok((F::one(), F::one(), F::one()));
    }

    // Compute entropies
    let h_true = entropy::<F>(labels_true)?;
    let h_pred = entropy::<F>(labels_pred)?;

    // Edge cases
    if h_true == F::zero() {
        return Ok((F::one(), F::one(), F::one()));
    }
    if h_pred == F::zero() {
        return Ok((F::one(), F::one(), F::one()));
    }

    // Compute conditional entropies
    let h_true_given_pred = conditional_entropy::<F>(labels_true, labels_pred)?;
    let h_pred_given_true = conditional_entropy::<F>(labels_pred, labels_true)?;

    // Compute homogeneity
    let homogeneity = if h_pred == F::zero() {
        F::one()
    } else {
        F::one() - h_true_given_pred / h_true
    };

    // Compute completeness
    let completeness = if h_true == F::zero() {
        F::one()
    } else {
        F::one() - h_pred_given_true / h_pred
    };

    // Compute V-measure
    let v_measure = if homogeneity + completeness == F::zero() {
        F::zero()
    } else {
        F::from(2.0).unwrap() * homogeneity * completeness / (homogeneity + completeness)
    };

    Ok((homogeneity, completeness, v_measure))
}

/// Compute conditional entropy H(X|Y).
fn conditional_entropy<F>(labels_x: ArrayView1<i32>, labels_y: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = labels_x.len() as f64;
    let contingency = build_contingency_matrix(labels_x, labels_y)?;

    let mut h_xy = F::zero();
    let col_sums = contingency.sum_axis(Axis(0));

    for j in 0..contingency.shape()[1] {
        let n_j = col_sums[j] as f64;
        if n_j > 0.0 {
            for i in 0..contingency.shape()[0] {
                let n_ij = contingency[[i, j]] as f64;
                if n_ij > 0.0 {
                    let term = n_ij / n * (n_ij / n_j).ln();
                    h_xy = h_xy - F::from(term).unwrap();
                }
            }
        }
    }

    Ok(h_xy)
}

/// Information-theoretic clustering metrics
pub mod information_theory {
    use super::*;
    use std::collections::HashMap;

    /// Jensen-Shannon divergence between two clusterings.
    ///
    /// The Jensen-Shannon divergence is a symmetric measure based on the Kullback-Leibler divergence.
    /// It's useful for comparing the probability distributions of two clusterings.
    ///
    /// # Arguments
    ///
    /// * `labels_true` - Ground truth cluster labels
    /// * `labels_pred` - Predicted cluster labels
    ///
    /// # Returns
    ///
    /// The Jensen-Shannon divergence score (0 = identical, 1 = completely different)
    pub fn jensen_shannon_divergence<F>(
        labels_true: ArrayView1<i32>,
        labels_pred: ArrayView1<i32>,
    ) -> Result<F>
    where
        F: Float + FromPrimitive + Debug + 'static,
    {
        if labels_true.len() != labels_pred.len() {
            return Err(ClusteringError::InvalidInput(
                "Labels arrays must have the same length".to_string(),
            ));
        }

        let n = labels_true.len();
        if n == 0 {
            return Ok(F::zero());
        }

        // Convert labels to probability distributions
        let p = label_distribution::<F>(labels_true)?;
        let q = label_distribution::<F>(labels_pred)?;

        // Compute average distribution
        let mut m = HashMap::new();
        for (label, &prob) in &p {
            *m.entry(*label).or_insert(F::zero()) += prob / F::from(2.0).unwrap();
        }
        for (label, &prob) in &q {
            *m.entry(*label).or_insert(F::zero()) += prob / F::from(2.0).unwrap();
        }

        // Compute KL divergences
        let kl_pm = kl_divergence(&p, &m)?;
        let kl_qm = kl_divergence(&q, &m)?;

        // Jensen-Shannon divergence
        let js = (kl_pm + kl_qm) / F::from(2.0).unwrap();
        Ok(js.sqrt()) // Return the Jensen-Shannon distance
    }

    /// Variation of Information (VI) between two clusterings.
    ///
    /// The Variation of Information is a symmetric measure that equals the sum of
    /// conditional entropies H(X|Y) + H(Y|X).
    ///
    /// # Arguments
    ///
    /// * `labels_true` - Ground truth cluster labels
    /// * `labels_pred` - Predicted cluster labels
    ///
    /// # Returns
    ///
    /// The Variation of Information score (lower is better)
    pub fn variation_of_information<F>(
        labels_true: ArrayView1<i32>,
        labels_pred: ArrayView1<i32>,
    ) -> Result<F>
    where
        F: Float + FromPrimitive + Debug + 'static,
    {
        if labels_true.len() != labels_pred.len() {
            return Err(ClusteringError::InvalidInput(
                "Labels arrays must have the same length".to_string(),
            ));
        }

        let h_true_given_pred = conditional_entropy::<F>(labels_true, labels_pred)?;
        let h_pred_given_true = conditional_entropy::<F>(labels_pred, labels_true)?;

        Ok(h_true_given_pred + h_pred_given_true)
    }

    /// Information-theoretic cluster quality measure.
    ///
    /// This measure combines intra-cluster and inter-cluster information
    /// to evaluate clustering quality without ground truth.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data (n_samples x n_features)
    /// * `labels` - Cluster labels
    ///
    /// # Returns
    ///
    /// The information-theoretic quality score (higher is better)
    pub fn information_cluster_quality<F>(
        data: ArrayView2<F>,
        labels: ArrayView1<i32>,
    ) -> Result<F>
    where
        F: Float + FromPrimitive + Debug + PartialOrd + 'static,
    {
        if data.shape()[0] != labels.shape()[0] {
            return Err(ClusteringError::InvalidInput(
                "Data and labels must have the same number of samples".to_string(),
            ));
        }

        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        // Find unique cluster labels
        let mut unique_labels = Vec::new();
        for &label in labels.iter() {
            if label >= 0 && !unique_labels.contains(&label) {
                unique_labels.push(label);
            }
        }

        let n_clusters = unique_labels.len();
        if n_clusters < 2 {
            return Ok(F::zero());
        }

        // Compute cluster entropies (within-cluster information)
        let mut total_within_cluster_entropy = F::zero();
        let mut valid_samples = 0;

        for &cluster_label in &unique_labels {
            let cluster_data: Vec<_> = data
                .rows()
                .into_iter()
                .zip(labels.iter())
                .filter_map(|(row, &label)| if label == cluster_label { Some(row) } else { None })
                .collect();

            if cluster_data.len() > 1 {
                let cluster_entropy = compute_data_entropy(&cluster_data)?;
                let cluster_size = cluster_data.len();
                total_within_cluster_entropy += F::from(cluster_size).unwrap() * cluster_entropy;
                valid_samples += cluster_size;
            }
        }

        if valid_samples > 0 {
            total_within_cluster_entropy /= F::from(valid_samples).unwrap();
        }

        // Compute overall data entropy
        let all_data: Vec<_> = data.rows().into_iter().collect();
        let overall_entropy = compute_data_entropy(&all_data)?;

        // Information gain (reduction in entropy due to clustering)
        let information_gain = overall_entropy - total_within_cluster_entropy;

        Ok(information_gain)
    }

    /// Compute entropy of a dataset based on feature variance.
    fn compute_data_entropy<F>(data: &[ndarray::ArrayView1<F>]) -> Result<F>
    where
        F: Float + FromPrimitive + Debug + 'static,
    {
        if data.is_empty() {
            return Ok(F::zero());
        }

        let n_samples = data.len();
        let n_features = data[0].len();
        
        let mut entropy = F::zero();

        // For each feature, compute entropy based on value distribution
        for feature_idx in 0..n_features {
            let mut values: Vec<F> = data.iter().map(|row| row[feature_idx]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Discretize continuous values into bins for entropy calculation
            let n_bins = (n_samples as f64).sqrt().ceil() as usize;
            let min_val = values[0];
            let max_val = values[n_samples - 1];
            
            if max_val == min_val {
                continue; // No entropy for constant features
            }

            let bin_width = (max_val - min_val) / F::from(n_bins).unwrap();
            let mut bin_counts = vec![0; n_bins];

            for &value in &values {
                let bin_idx = if value == max_val {
                    n_bins - 1
                } else {
                    ((value - min_val) / bin_width).to_usize().unwrap_or(0).min(n_bins - 1)
                };
                bin_counts[bin_idx] += 1;
            }

            // Compute entropy for this feature
            let mut feature_entropy = F::zero();
            for &count in &bin_counts {
                if count > 0 {
                    let p = F::from(count).unwrap() / F::from(n_samples).unwrap();
                    feature_entropy -= p * p.ln();
                }
            }

            entropy += feature_entropy;
        }

        Ok(entropy / F::from(n_features).unwrap())
    }

    /// Convert label array to probability distribution.
    fn label_distribution<F>(labels: ArrayView1<i32>) -> Result<HashMap<i32, F>>
    where
        F: Float + FromPrimitive + Debug + 'static,
    {
        let mut counts = HashMap::new();
        let mut total = 0;

        for &label in labels.iter() {
            if label >= 0 {
                *counts.entry(label).or_insert(0) += 1;
                total += 1;
            }
        }

        if total == 0 {
            return Ok(HashMap::new());
        }

        let mut distribution = HashMap::new();
        for (label, count) in counts {
            distribution.insert(label, F::from(count).unwrap() / F::from(total).unwrap());
        }

        Ok(distribution)
    }

    /// Compute Kullback-Leibler divergence between two probability distributions.
    fn kl_divergence<F>(p: &HashMap<i32, F>, q: &HashMap<i32, F>) -> Result<F>
    where
        F: Float + FromPrimitive + Debug + 'static,
    {
        let mut kl = F::zero();

        for (label, &p_val) in p {
            if p_val > F::zero() {
                let q_val = q.get(label).cloned().unwrap_or(F::from(1e-10).unwrap()); // Smoothing
                if q_val > F::zero() {
                    kl += p_val * (p_val / q_val).ln();
                }
            }
        }

        Ok(kl)
    }
}

/// Stability-based clustering validation methods
pub mod stability {
    use super::*;
    use crate::vq::{kmeans2, MinitMethod};

    /// Cluster stability analysis using bootstrap resampling.
    ///
    /// This function evaluates clustering stability by performing clustering
    /// on multiple bootstrap samples and measuring the consistency of results.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data (n_samples x n_features)
    /// * `n_clusters` - Number of clusters
    /// * `n_bootstrap` - Number of bootstrap iterations
    /// * `subsample_ratio` - Fraction of data to use in each bootstrap sample
    ///
    /// # Returns
    ///
    /// Stability score (0 = unstable, 1 = perfectly stable)
    pub fn cluster_stability_bootstrap<F>(
        data: ArrayView2<F>,
        n_clusters: usize,
        n_bootstrap: usize,
        subsample_ratio: f64,
    ) -> Result<F>
    where
        F: Float + FromPrimitive + Debug + PartialOrd + Copy + 'static,
    {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        if subsample_ratio <= 0.0 || subsample_ratio > 1.0 {
            return Err(ClusteringError::InvalidInput(
                "Subsample ratio must be between 0 and 1".to_string(),
            ));
        }

        let n_samples = data.shape()[0];
        let subsample_size = (n_samples as f64 * subsample_ratio) as usize;
        
        if subsample_size < n_clusters {
            return Err(ClusteringError::InvalidInput(
                "Subsample size must be at least as large as number of clusters".to_string(),
            ));
        }

        let mut rng = thread_rng();
        let mut all_labels = Vec::new();
        let mut sample_indices_list = Vec::new();

        // Perform bootstrap clustering
        for _iter in 0..n_bootstrap {
            // Create bootstrap sample
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            indices.truncate(subsample_size);
            indices.sort();

            // Extract bootstrap sample
            let bootstrap_data = data.select(ndarray::Axis(0), &indices);

            // Perform clustering (using k-means as default)
            let max_iter = 100;
            let thresh = F::from(1e-6).unwrap();
            
            match kmeans2(bootstrap_data.view(), n_clusters, Some(max_iter), Some(thresh), Some(MinitMethod::PlusPlus), None, Some(true), None) {
                Ok((_, labels)) => {
                    all_labels.push(labels);
                    sample_indices_list.push(indices);
                }
                Err(_) => continue, // Skip failed clusterings
            }
        }

        if all_labels.len() < 2 {
            return Err(ClusteringError::ComputationError(
                "Too few successful clustering iterations".to_string(),
            ));
        }

        // Compute pairwise stability scores
        let mut stability_scores = Vec::new();

        for i in 0..all_labels.len() {
            for j in (i + 1)..all_labels.len() {
                // Find common samples between bootstrap iterations
                let common_indices = find_common_indices(&sample_indices_list[i], &sample_indices_list[j]);
                
                if common_indices.len() < n_clusters {
                    continue;
                }

                // Extract labels for common samples
                let labels_i = extract_common_labels(&all_labels[i], &sample_indices_list[i], &common_indices);
                let labels_j = extract_common_labels(&all_labels[j], &sample_indices_list[j], &common_indices);

                // Compute agreement (using ARI)
                if let Ok(ari) = adjusted_rand_index::<F>(
                    ndarray::Array1::from_vec(labels_i).view(),
                    ndarray::Array1::from_vec(labels_j).view(),
                ) {
                    stability_scores.push(ari);
                }
            }
        }

        if stability_scores.is_empty() {
            return Ok(F::zero());
        }

        // Return mean stability score
        let mean_stability = stability_scores.iter().fold(F::zero(), |acc, &x| acc + x) 
            / F::from(stability_scores.len()).unwrap();

        Ok(mean_stability.max(F::zero()).min(F::one()))
    }

    /// Find indices that appear in both bootstrap samples.
    fn find_common_indices(indices_a: &[usize], indices_b: &[usize]) -> Vec<usize> {
        let mut common = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < indices_a.len() && j < indices_b.len() {
            if indices_a[i] == indices_b[j] {
                common.push(indices_a[i]);
                i += 1;
                j += 1;
            } else if indices_a[i] < indices_b[j] {
                i += 1;
            } else {
                j += 1;
            }
        }

        common
    }

    /// Extract labels for common sample indices.
    fn extract_common_labels(
        labels: &Array1<usize>,
        sample_indices: &[usize],
        common_indices: &[usize],
    ) -> Vec<i32> {
        let mut common_labels = Vec::new();
        
        for &common_idx in common_indices {
            if let Some(pos) = sample_indices.iter().position(|&x| x == common_idx) {
                if pos < labels.len() {
                    common_labels.push(labels[pos] as i32);
                }
            }
        }

        common_labels
    }

    /// Stability-based selection of optimal number of clusters.
    ///
    /// This function tests multiple values of k and returns the one with
    /// the highest stability score.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data
    /// * `k_range` - Range of k values to test
    /// * `n_bootstrap` - Number of bootstrap iterations per k
    /// * `subsample_ratio` - Bootstrap subsample ratio
    ///
    /// # Returns
    ///
    /// Tuple of (optimal_k, stability_scores_for_each_k)
    pub fn optimal_clusters_stability<F>(
        data: ArrayView2<F>,
        k_range: std::ops::Range<usize>,
        n_bootstrap: usize,
        subsample_ratio: f64,
    ) -> Result<(usize, Vec<F>)>
    where
        F: Float + FromPrimitive + Debug + PartialOrd + Copy + 'static,
    {
        let mut stability_scores = Vec::new();
        let mut best_k = k_range.start;
        let mut best_score = F::neg_infinity();

        for k in k_range.clone() {
            if k >= data.shape()[0] {
                break; // Can't have more clusters than samples
            }

            match cluster_stability_bootstrap(data, k, n_bootstrap, subsample_ratio) {
                Ok(score) => {
                    stability_scores.push(score);
                    if score > best_score {
                        best_score = score;
                        best_k = k;
                    }
                }
                Err(_) => {
                    stability_scores.push(F::zero());
                }
            }
        }

        Ok((best_k, stability_scores))
    }
}

/// Advanced clustering evaluation metrics
pub mod advanced {
    use super::*;

    /// Dunn index for cluster validation.
    ///
    /// The Dunn index is the ratio of the smallest distance between observations not
    /// in the same cluster to the largest intra-cluster distance. Higher values indicate
    /// better clustering.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data (n_samples x n_features)
    /// * `labels` - Cluster labels
    ///
    /// # Returns
    ///
    /// The Dunn index (higher is better)
    pub fn dunn_index<F>(data: ArrayView2<F>, labels: ArrayView1<i32>) -> Result<F>
    where
        F: Float + FromPrimitive + Debug + PartialOrd + 'static,
    {
        if data.shape()[0] != labels.shape()[0] {
            return Err(ClusteringError::InvalidInput(
                "Data and labels must have the same number of samples".to_string(),
            ));
        }

        let n_samples = data.shape()[0];
        if n_samples < 2 {
            return Ok(F::zero());
        }

        // Find unique cluster labels
        let mut unique_labels = Vec::new();
        for &label in labels.iter() {
            if label >= 0 && !unique_labels.contains(&label) {
                unique_labels.push(label);
            }
        }

        if unique_labels.len() < 2 {
            return Ok(F::zero());
        }

        // Compute minimum inter-cluster distance
        let mut min_inter_cluster = F::infinity();
        
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                if labels[i] >= 0 && labels[j] >= 0 && labels[i] != labels[j] {
                    let distance = euclidean_distance(data.row(i), data.row(j));
                    if distance < min_inter_cluster {
                        min_inter_cluster = distance;
                    }
                }
            }
        }

        // Compute maximum intra-cluster distance
        let mut max_intra_cluster = F::zero();
        
        for &cluster_label in &unique_labels {
            let cluster_indices: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == cluster_label { Some(i) } else { None })
                .collect();

            for i in 0..cluster_indices.len() {
                for j in (i + 1)..cluster_indices.len() {
                    let distance = euclidean_distance(
                        data.row(cluster_indices[i]),
                        data.row(cluster_indices[j]),
                    );
                    if distance > max_intra_cluster {
                        max_intra_cluster = distance;
                    }
                }
            }
        }

        if max_intra_cluster == F::zero() {
            return Ok(F::infinity());
        }

        Ok(min_inter_cluster / max_intra_cluster)
    }

    /// Compute Euclidean distance between two points.
    fn euclidean_distance<F>(point1: ndarray::ArrayView1<F>, point2: ndarray::ArrayView1<F>) -> F
    where
        F: Float + FromPrimitive + Debug + 'static,
    {
        let diff = &point1 - &point2;
        diff.dot(&diff).sqrt()
    }

    /// Bayesian Information Criterion (BIC) for model selection.
    ///
    /// Estimates the BIC for a clustering result, useful for determining
    /// the optimal number of clusters.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data
    /// * `labels` - Cluster labels
    ///
    /// # Returns
    ///
    /// The BIC score (lower is better)
    pub fn bic_score<F>(data: ArrayView2<F>, labels: ArrayView1<i32>) -> Result<F>
    where
        F: Float + FromPrimitive + Debug + PartialOrd + 'static,
    {
        if data.shape()[0] != labels.shape()[0] {
            return Err(ClusteringError::InvalidInput(
                "Data and labels must have the same number of samples".to_string(),
            ));
        }

        let n_samples = data.shape()[0] as f64;
        let n_features = data.shape()[1] as f64;

        // Find unique cluster labels
        let mut unique_labels = Vec::new();
        for &label in labels.iter() {
            if label >= 0 && !unique_labels.contains(&label) {
                unique_labels.push(label);
            }
        }

        let n_clusters = unique_labels.len() as f64;
        if n_clusters < 1.0 {
            return Ok(F::infinity());
        }

        // Compute log-likelihood (simplified Gaussian assumption)
        let mut log_likelihood = 0.0;
        let mut total_variance = 0.0;

        for &cluster_label in &unique_labels {
            let cluster_data: Vec<_> = data
                .rows()
                .into_iter()
                .zip(labels.iter())
                .filter_map(|(row, &label)| if label == cluster_label { Some(row) } else { None })
                .collect();

            if cluster_data.len() > 1 {
                // Compute cluster variance
                let n_cluster = cluster_data.len() as f64;
                let mean = compute_cluster_mean(&cluster_data);
                
                let mut cluster_variance = 0.0;
                for row in &cluster_data {
                    let diff = row.to_owned() - &mean;
                    cluster_variance += diff.dot(&diff).to_f64().unwrap();
                }
                cluster_variance /= n_cluster;
                
                if cluster_variance > 0.0 {
                    log_likelihood -= n_cluster * (cluster_variance.ln() + n_features * (2.0 * std::f64::consts::PI).ln()) / 2.0;
                    total_variance += cluster_variance;
                }
            }
        }

        // Number of parameters: cluster centers + covariance parameters
        let n_params = n_clusters * n_features + n_clusters;

        // BIC = -2 * log_likelihood + k * ln(n)
        let bic = -2.0 * log_likelihood + n_params * n_samples.ln();

        Ok(F::from(bic).unwrap())
    }

    /// Compute mean of cluster data points.
    fn compute_cluster_mean<F>(cluster_data: &[ndarray::ArrayView1<F>]) -> ndarray::Array1<F>
    where
        F: Float + FromPrimitive + Debug + 'static,
    {
        if cluster_data.is_empty() {
            return ndarray::Array1::zeros(0);
        }

        let n_features = cluster_data[0].len();
        let mut mean = ndarray::Array1::zeros(n_features);
        
        for row in cluster_data {
            mean = mean + row;
        }
        
        mean / F::from(cluster_data.len()).unwrap()
    }
}

/// Ensemble validation methods for clustering evaluation
pub mod ensemble {
    use super::*;
    use crate::vq::{kmeans2, MinitMethod, euclidean_distance};
    use rand::Rng;
    use std::collections::HashMap;

    /// Consensus clustering score using multiple clustering algorithms.
    ///
    /// This function evaluates clustering quality by running multiple clustering
    /// algorithms and measuring their agreement. Higher consensus indicates
    /// more stable clustering structure.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data (n_samples x n_features)
    /// * `n_clusters` - Number of clusters
    /// * `n_algorithms` - Number of different initializations to try
    ///
    /// # Returns
    ///
    /// Consensus score (0 = no agreement, 1 = perfect agreement)
    pub fn consensus_clustering_score<F>(
        data: ArrayView2<F>,
        n_clusters: usize,
        n_algorithms: usize,
    ) -> Result<F>
    where
        F: Float + FromPrimitive + Debug + PartialOrd + Copy + 'static,
    {
        if n_algorithms < 2 {
            return Err(ClusteringError::InvalidInput(
                "Need at least 2 algorithms for consensus".to_string(),
            ));
        }

        let mut all_labels = Vec::new();

        // Run clustering with different random initializations
        for seed in 0..n_algorithms {
            // Use different random seeds for initialization
            let max_iter = 100;
            let thresh = F::from(1e-6).unwrap();
            
            match kmeans2(data, n_clusters, Some(max_iter), Some(thresh), Some(MinitMethod::PlusPlus), None, Some(true), Some(seed as u64)) {
                Ok((_, labels)) => {
                    // Convert to i32 for consistency with other functions
                    let labels_i32: Vec<i32> = labels.iter().map(|&x| x as i32).collect();
                    all_labels.push(Array1::from_vec(labels_i32));
                }
                Err(_) => {
                    // If clustering fails, try with different parameters
                    continue;
                }
            }
        }

        if all_labels.len() < 2 {
            return Err(ClusteringError::ComputationError(
                "Not enough successful clustering runs".to_string(),
            ));
        }

        // Compute pairwise agreement using Adjusted Rand Index
        let mut agreements = Vec::new();
        for i in 0..all_labels.len() {
            for j in (i + 1)..all_labels.len() {
                if let Ok(ari) = adjusted_rand_index::<F>(
                    all_labels[i].view(),
                    all_labels[j].view(),
                ) {
                    agreements.push(ari);
                }
            }
        }

        if agreements.is_empty() {
            return Ok(F::zero());
        }

        // Return mean agreement
        let mean_agreement = agreements.iter().fold(F::zero(), |acc, &x| acc + x) 
            / F::from(agreements.len()).unwrap();

        Ok(mean_agreement.max(F::zero()).min(F::one()))
    }

    /// Multi-criterion validation combining multiple metrics.
    ///
    /// This function combines multiple internal validation metrics to provide
    /// a comprehensive assessment of clustering quality.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data
    /// * `labels` - Cluster labels
    ///
    /// # Returns
    ///
    /// Composite score combining multiple metrics
    pub fn multi_criterion_validation<F>(
        data: ArrayView2<F>,
        labels: ArrayView1<i32>,
    ) -> Result<F>
    where
        F: Float + FromPrimitive + Debug + PartialOrd + 'static,
    {
        // Compute individual metrics
        let silhouette = silhouette_score(data, labels)?;
        let davies_bouldin = davies_bouldin_score(data, labels)?;
        let calinski_harabasz = calinski_harabasz_score(data, labels)?;

        // Normalize Davies-Bouldin score (lower is better -> higher is better)
        let db_normalized = F::one() / (F::one() + davies_bouldin);

        // Normalize Calinski-Harabasz score to [0, 1] range
        let ch_normalized = calinski_harabasz / (F::one() + calinski_harabasz);

        // Combine metrics with equal weights
        let composite_score = (silhouette + db_normalized + ch_normalized) / F::from(3.0).unwrap();

        Ok(composite_score)
    }

    /// Cross-validation score for clustering stability.
    ///
    /// This function evaluates clustering stability using k-fold cross-validation,
    /// training on k-1 folds and evaluating on the remaining fold.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data
    /// * `n_clusters` - Number of clusters
    /// * `k_folds` - Number of cross-validation folds
    ///
    /// # Returns
    ///
    /// Cross-validation stability score
    pub fn cross_validation_score<F>(
        data: ArrayView2<F>,
        n_clusters: usize,
        k_folds: usize,
    ) -> Result<F>
    where
        F: Float + FromPrimitive + Debug + PartialOrd + Copy + 'static,
    {
        if k_folds < 2 {
            return Err(ClusteringError::InvalidInput(
                "Need at least 2 folds for cross-validation".to_string(),
            ));
        }

        let n_samples = data.shape()[0];
        if n_samples < k_folds {
            return Err(ClusteringError::InvalidInput(
                "Number of samples must be at least equal to number of folds".to_string(),
            ));
        }

        let fold_size = n_samples / k_folds;
        let mut cv_scores = Vec::new();

        for fold in 0..k_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == k_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create training data (exclude current fold)
            let mut train_indices = Vec::new();
            for i in 0..n_samples {
                if i < start_idx || i >= end_idx {
                    train_indices.push(i);
                }
            }

            if train_indices.len() < n_clusters {
                continue;
            }

            // Extract training data
            let train_data = data.select(ndarray::Axis(0), &train_indices);

            // Perform clustering on training data
            let max_iter = 100;
            let thresh = F::from(1e-6).unwrap();
            
            match kmeans2(train_data.view(), n_clusters, Some(max_iter), Some(thresh), Some(MinitMethod::PlusPlus), None, Some(true), None) {
                Ok((centers, _)) => {
                    // Predict labels for test fold
                    let mut test_labels = Vec::new();
                    for i in start_idx..end_idx {
                        let mut min_dist = F::infinity();
                        let mut closest_cluster = 0;
                        
                        for (cluster_idx, center_row) in centers.rows().into_iter().enumerate() {
                            let dist = euclidean_distance(data.row(i), center_row);
                            if dist < min_dist {
                                min_dist = dist;
                                closest_cluster = cluster_idx;
                            }
                        }
                        test_labels.push(closest_cluster as i32);
                    }

                    // Extract test data
                    let test_indices: Vec<usize> = (start_idx..end_idx).collect();
                    let test_data = data.select(ndarray::Axis(0), &test_indices);
                    let test_labels_array = Array1::from_vec(test_labels);

                    // Compute silhouette score for test fold
                    if let Ok(score) = silhouette_score(test_data.view(), test_labels_array.view()) {
                        cv_scores.push(score);
                    }
                }
                Err(_) => continue,
            }
        }

        if cv_scores.is_empty() {
            return Ok(F::zero());
        }

        // Return mean cross-validation score
        let mean_score = cv_scores.iter().fold(F::zero(), |acc, &x| acc + x) 
            / F::from(cv_scores.len()).unwrap();

        Ok(mean_score)
    }

    /// Robust validation using multiple evaluation criteria.
    ///
    /// This function provides a robust assessment by combining stability analysis,
    /// internal validation metrics, and consensus measurements.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data
    /// * `n_clusters` - Number of clusters
    /// * `n_bootstrap` - Number of bootstrap iterations
    /// * `n_consensus` - Number of consensus clustering runs
    ///
    /// # Returns
    ///
    /// Robust validation score
    pub fn robust_validation<F>(
        data: ArrayView2<F>,
        n_clusters: usize,
        n_bootstrap: usize,
        n_consensus: usize,
    ) -> Result<F>
    where
        F: Float + FromPrimitive + Debug + PartialOrd + Copy + 'static,
    {
        // Perform standard clustering
        let max_iter = 100;
        let thresh = F::from(1e-6).unwrap();
        
        let (_, labels) = kmeans2(data, n_clusters, Some(max_iter), Some(thresh), Some(MinitMethod::PlusPlus), None, Some(true), None)?;
        let labels_i32: Vec<i32> = labels.iter().map(|&x| x as i32).collect();
        let labels_array = Array1::from_vec(labels_i32);

        // 1. Internal validation metrics
        let multi_criterion = multi_criterion_validation(data, labels_array.view())?;

        // 2. Stability analysis
        let stability = super::stability::cluster_stability_bootstrap(
            data, n_clusters, n_bootstrap, 0.8
        ).unwrap_or(F::zero());

        // 3. Consensus clustering
        let consensus = consensus_clustering_score(data, n_clusters, n_consensus)
            .unwrap_or(F::zero());

        // 4. Cross-validation score
        let cv_score = cross_validation_score(data, n_clusters, 5)
            .unwrap_or(F::zero());

        // Combine all scores with weights
        let weights = [F::from(0.3).unwrap(), F::from(0.3).unwrap(), F::from(0.2).unwrap(), F::from(0.2).unwrap()];
        let scores = [multi_criterion, stability, consensus, cv_score];

        let robust_score = weights.iter().zip(scores.iter())
            .map(|(&w, &s)| w * s)
            .fold(F::zero(), |acc, x| acc + x);

        Ok(robust_score)
    }

    /// Confidence interval estimation for clustering metrics using bootstrap.
    ///
    /// This function estimates confidence intervals for clustering quality metrics
    /// using bootstrap resampling.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data
    /// * `labels` - Cluster labels
    /// * `confidence_level` - Confidence level (e.g., 0.95 for 95% CI)
    /// * `n_bootstrap` - Number of bootstrap samples
    ///
    /// # Returns
    ///
    /// Tuple of (lower_bound, mean_score, upper_bound)
    pub fn bootstrap_confidence_interval<F>(
        data: ArrayView2<F>,
        labels: ArrayView1<i32>,
        confidence_level: f64,
        n_bootstrap: usize,
    ) -> Result<(F, F, F)>
    where
        F: Float + FromPrimitive + Debug + PartialOrd + Copy + 'static,
    {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(ClusteringError::InvalidInput(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }

        let n_samples = data.shape()[0];
        let mut rng = thread_rng();
        let mut bootstrap_scores = Vec::new();

        // Perform bootstrap resampling
        for _iter in 0..n_bootstrap {
            // Create bootstrap sample
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            
            // Sample with replacement
            let bootstrap_indices: Vec<usize> = (0..n_samples)
                .map(|_| indices[rng.gen_range(0..n_samples)])
                .collect();

            // Extract bootstrap sample
            let bootstrap_data = data.select(ndarray::Axis(0), &bootstrap_indices);
            let bootstrap_labels: Vec<i32> = bootstrap_indices.iter()
                .map(|&i| labels[i])
                .collect();
            let bootstrap_labels_array = Array1::from_vec(bootstrap_labels);

            // Compute metric (using silhouette score as example)
            if let Ok(score) = silhouette_score(bootstrap_data.view(), bootstrap_labels_array.view()) {
                bootstrap_scores.push(score);
            }
        }

        if bootstrap_scores.is_empty() {
            return Err(ClusteringError::ComputationError(
                "No successful bootstrap iterations".to_string(),
            ));
        }

        // Sort scores for percentile calculation
        bootstrap_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate confidence interval
        let alpha = 1.0 - confidence_level;
        let lower_percentile = alpha / 2.0;
        let upper_percentile = 1.0 - alpha / 2.0;

        let lower_idx = (bootstrap_scores.len() as f64 * lower_percentile) as usize;
        let upper_idx = (bootstrap_scores.len() as f64 * upper_percentile) as usize;

        let lower_bound = bootstrap_scores[lower_idx.min(bootstrap_scores.len() - 1)];
        let upper_bound = bootstrap_scores[upper_idx.min(bootstrap_scores.len() - 1)];

        // Calculate mean score
        let mean_score = bootstrap_scores.iter().fold(F::zero(), |acc, &x| acc + x) 
            / F::from(bootstrap_scores.len()).unwrap();

        Ok((lower_bound, mean_score, upper_bound))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_davies_bouldin_score() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2],
        )
        .unwrap();
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let score = davies_bouldin_score(data.view(), labels.view()).unwrap();
        assert!(score < 0.5); // Should be low for well-separated clusters
    }

    #[test]
    fn test_calinski_harabasz_score() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2],
        )
        .unwrap();
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let score = calinski_harabasz_score(data.view(), labels.view()).unwrap();
        assert!(score > 50.0); // Should be high for well-separated clusters
    }

    #[test]
    fn test_adjusted_rand_index() {
        // Perfect agreement
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let ari: f64 = adjusted_rand_index(labels_true.view(), labels_pred.view()).unwrap();
        assert!((ari - 1.0).abs() < 1e-6);

        // Label permutation (still perfect agreement)
        let labels_pred2 = Array1::from_vec(vec![1, 1, 2, 2, 0, 0]);
        let ari2: f64 = adjusted_rand_index(labels_true.view(), labels_pred2.view()).unwrap();
        assert!((ari2 - 1.0).abs() < 1e-6);

        // Partial agreement
        let labels_pred3 = Array1::from_vec(vec![0, 0, 1, 1, 1, 1]);
        let ari3: f64 = adjusted_rand_index(labels_true.view(), labels_pred3.view()).unwrap();
        assert!(ari3 > 0.0 && ari3 < 1.0);
    }

    #[test]
    fn test_normalized_mutual_info() {
        // Perfect agreement
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 1]);

        let nmi: f64 =
            normalized_mutual_info(labels_true.view(), labels_pred.view(), "arithmetic").unwrap();
        assert!((nmi - 1.0).abs() < 1e-6);

        // Test different normalizations
        let labels_true2 = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let labels_pred2 = Array1::from_vec(vec![0, 0, 1, 1, 1, 1]);

        let nmi_arith: f64 =
            normalized_mutual_info(labels_true2.view(), labels_pred2.view(), "arithmetic").unwrap();
        let nmi_geom: f64 =
            normalized_mutual_info(labels_true2.view(), labels_pred2.view(), "geometric").unwrap();

        assert!(nmi_arith > 0.0 && nmi_arith < 1.0);
        assert!(nmi_geom > 0.0 && nmi_geom < 1.0);
    }

    #[test]
    fn test_homogeneity_completeness_v_measure() {
        // Perfect clustering
        let labels_true = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let labels_pred = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);

        let (h, c, v): (f64, f64, f64) =
            homogeneity_completeness_v_measure(labels_true.view(), labels_pred.view()).unwrap();
        assert!((h - 1.0).abs() < 1e-6);
        assert!((c - 1.0).abs() < 1e-6);
        assert!((v - 1.0).abs() < 1e-6);

        // Imperfect clustering - classes 1 and 2 merged
        let labels_pred2 = Array1::from_vec(vec![0, 0, 1, 1, 1, 1]);
        let (h2, c2, v2): (f64, f64, f64) =
            homogeneity_completeness_v_measure(labels_true.view(), labels_pred2.view()).unwrap();

        // When classes are merged, completeness is actually perfect (1.0) because
        // all members of each true class are contained within single predicted clusters
        // Homogeneity is lower because predicted clusters contain multiple true classes
        assert!(h2 > 0.0 && h2 < 1.0); // Reduced homogeneity
        assert!(c2 > 0.9); // High completeness
        assert!(v2 > 0.0 && v2 < 1.0); // V-measure between 0 and 1
    }
}
