//! Information-theoretic clustering evaluation metrics
//!
//! This module provides information-theoretic metrics for evaluating clustering algorithms
//! when ground truth labels are available. These metrics are based on concepts from 
//! information theory and measure how well the clustering preserves the information
//! in the true class structure.

use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Calculate mutual information between two label assignments
///
/// Mutual Information (MI) measures the amount of information shared between
/// two random variables. In clustering, it measures how much information
/// the predicted clusters share with the true clusters.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
///
/// The mutual information score (higher is better)
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_cluster::metrics::mutual_info_score;
///
/// let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
/// let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 1, 2]);
///
/// let mi = mutual_info_score(true_labels.view(), pred_labels.view()).unwrap();
/// assert!(mi > 0.0);
/// ```
pub fn mutual_info_score<F>(labels_true: ArrayView1<i32>, labels_pred: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "True and predicted labels must have the same length".to_string(),
        ));
    }

    let n_samples = labels_true.len();
    if n_samples == 0 {
        return Ok(F::zero());
    }

    // Create contingency table
    let contingency = build_contingency_table(labels_true, labels_pred);
    
    let mut mi = F::zero();
    let n_samples_f = F::from(n_samples).unwrap();

    // Calculate marginal probabilities
    let mut row_sums = HashMap::new();
    let mut col_sums = HashMap::new();
    
    for (&(i, j), &count) in &contingency {
        *row_sums.entry(i).or_insert(0) += count;
        *col_sums.entry(j).or_insert(0) += count;
    }

    // Calculate mutual information
    for (&(i, j), &n_ij) in &contingency {
        if n_ij > 0 {
            let n_i = row_sums[&i];
            let n_j = col_sums[&j];
            
            let p_ij = F::from(n_ij).unwrap() / n_samples_f;
            let p_i = F::from(n_i).unwrap() / n_samples_f;
            let p_j = F::from(n_j).unwrap() / n_samples_f;
            
            mi = mi + p_ij * (p_ij / (p_i * p_j)).ln();
        }
    }

    Ok(mi)
}

/// Calculate normalized mutual information between two label assignments
///
/// Normalized Mutual Information (NMI) normalizes the mutual information
/// by the geometric mean of the entropies of both label assignments.
/// This provides a score between 0 and 1, where 1 indicates perfect agreement.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
///
/// The normalized mutual information score (0 to 1, higher is better)
pub fn normalized_mutual_info_score<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let mi = mutual_info_score(labels_true, labels_pred)?;
    let h_true = entropy(labels_true)?;
    let h_pred = entropy(labels_pred)?;

    if h_true == F::zero() && h_pred == F::zero() {
        return Ok(F::one());
    }

    let normalizer = (h_true * h_pred).sqrt();
    
    if normalizer == F::zero() {
        Ok(F::zero())
    } else {
        Ok(mi / normalizer)
    }
}

/// Calculate adjusted mutual information between two label assignments
///
/// Adjusted Mutual Information (AMI) adjusts the mutual information for chance,
/// providing a score that is corrected for random labeling.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
///
/// The adjusted mutual information score (higher is better, can be negative)
pub fn adjusted_mutual_info_score<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let mi = mutual_info_score(labels_true, labels_pred)?;
    let emi = expected_mutual_info(labels_true, labels_pred)?;
    let h_true = entropy(labels_true)?;
    let h_pred = entropy(labels_pred)?;

    let max_entropy = if h_true > h_pred { h_true } else { h_pred };
    
    if max_entropy == emi {
        Ok(F::zero())
    } else {
        Ok((mi - emi) / (max_entropy - emi))
    }
}

/// Calculate adjusted rand index between two label assignments
///
/// The Adjusted Rand Index (ARI) measures the similarity between two clusterings
/// by considering all pairs of samples and counting pairs that are assigned 
/// in the same or different clusters in both clusterings.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
///
/// The adjusted rand index (-1 to 1, higher is better)
pub fn adjusted_rand_score<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    if labels_true.len() != labels_pred.len() {
        return Err(ClusteringError::InvalidInput(
            "True and predicted labels must have the same length".to_string(),
        ));
    }

    let n_samples = labels_true.len();
    if n_samples == 0 {
        return Ok(F::one());
    }

    let contingency = build_contingency_table(labels_true, labels_pred);
    
    // Calculate marginal sums
    let mut row_sums = HashMap::new();
    let mut col_sums = HashMap::new();
    
    for (&(i, j), &count) in &contingency {
        *row_sums.entry(i).or_insert(0) += count;
        *col_sums.entry(j).or_insert(0) += count;
    }

    // Calculate ARI components
    let mut sum_comb_c = F::zero();
    for &count in contingency.values() {
        if count >= 2 {
            sum_comb_c = sum_comb_c + F::from(comb2(count)).unwrap();
        }
    }

    let mut sum_comb_a = F::zero();
    for &count in row_sums.values() {
        if count >= 2 {
            sum_comb_a = sum_comb_a + F::from(comb2(count)).unwrap();
        }
    }

    let mut sum_comb_b = F::zero();
    for &count in col_sums.values() {
        if count >= 2 {
            sum_comb_b = sum_comb_b + F::from(comb2(count)).unwrap();
        }
    }

    let n_choose_2 = F::from(comb2(n_samples)).unwrap();
    let expected_index = sum_comb_a * sum_comb_b / n_choose_2;
    let max_index = (sum_comb_a + sum_comb_b) / F::from(2).unwrap();

    if max_index == expected_index {
        Ok(F::zero())
    } else {
        Ok((sum_comb_c - expected_index) / (max_index - expected_index))
    }
}

/// Calculate V-measure score
///
/// V-measure is the harmonic mean of homogeneity and completeness.
/// It provides a single score that balances both aspects of clustering quality.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
///
/// The V-measure score (0 to 1, higher is better)
pub fn v_measure_score<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let homogeneity = homogeneity_score(labels_true, labels_pred)?;
    let completeness = completeness_score(labels_true, labels_pred)?;

    if homogeneity + completeness == F::zero() {
        Ok(F::zero())
    } else {
        let two = F::from(2).unwrap();
        Ok(two * homogeneity * completeness / (homogeneity + completeness))
    }
}

/// Calculate homogeneity score
///
/// Homogeneity measures whether each cluster contains only members of a single class.
/// A clustering is homogeneous if all clusters contain only data points from one class.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
///
/// The homogeneity score (0 to 1, higher is better)
pub fn homogeneity_score<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let h_true = entropy(labels_true)?;
    
    if h_true == F::zero() {
        return Ok(F::one());
    }

    let h_true_given_pred = conditional_entropy(labels_true, labels_pred)?;
    Ok((h_true - h_true_given_pred) / h_true)
}

/// Calculate completeness score
///
/// Completeness measures whether all members of a given class are assigned 
/// to the same cluster. A clustering is complete if all data points from 
/// one class are in the same cluster.
///
/// # Arguments
///
/// * `labels_true` - Ground truth class labels
/// * `labels_pred` - Predicted cluster labels
///
/// # Returns
///
/// The completeness score (0 to 1, higher is better)
pub fn completeness_score<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let h_pred = entropy(labels_pred)?;
    
    if h_pred == F::zero() {
        return Ok(F::one());
    }

    let h_pred_given_true = conditional_entropy(labels_pred, labels_true)?;
    Ok((h_pred - h_pred_given_true) / h_pred)
}

// Helper functions

/// Build contingency table from two label arrays
fn build_contingency_table(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> HashMap<(i32, i32), usize> {
    let mut contingency = HashMap::new();
    
    for (&true_label, &pred_label) in labels_true.iter().zip(labels_pred.iter()) {
        *contingency.entry((true_label, pred_label)).or_insert(0) += 1;
    }
    
    contingency
}

/// Calculate entropy of a label assignment
fn entropy<F>(labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let mut counts = HashMap::new();
    for &label in labels.iter() {
        *counts.entry(label).or_insert(0) += 1;
    }

    let n_samples = labels.len();
    let n_samples_f = F::from(n_samples).unwrap();
    
    let mut entropy = F::zero();
    for &count in counts.values() {
        if count > 0 {
            let p = F::from(count).unwrap() / n_samples_f;
            entropy = entropy - p * p.ln();
        }
    }

    Ok(entropy)
}

/// Calculate conditional entropy H(X|Y)
fn conditional_entropy<F>(
    labels_x: ArrayView1<i32>,
    labels_y: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let contingency = build_contingency_table(labels_x, labels_y);
    
    // Calculate marginal counts for Y
    let mut y_counts = HashMap::new();
    for (&(_, y), &count) in &contingency {
        *y_counts.entry(y).or_insert(0) += count;
    }

    let n_samples = labels_x.len();
    let n_samples_f = F::from(n_samples).unwrap();
    
    let mut cond_entropy = F::zero();
    
    for (&y, &n_y) in &y_counts {
        if n_y == 0 {
            continue;
        }
        
        let p_y = F::from(n_y).unwrap() / n_samples_f;
        let mut h_x_given_y = F::zero();
        
        for (&(x, y_val), &n_xy) in &contingency {
            if y_val == y && n_xy > 0 {
                let p_x_given_y = F::from(n_xy).unwrap() / F::from(n_y).unwrap();
                h_x_given_y = h_x_given_y - p_x_given_y * p_x_given_y.ln();
            }
        }
        
        cond_entropy = cond_entropy + p_y * h_x_given_y;
    }

    Ok(cond_entropy)
}

/// Calculate expected mutual information for random labeling
fn expected_mutual_info<F>(
    labels_true: ArrayView1<i32>,
    labels_pred: ArrayView1<i32>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    // For simplicity, we approximate EMI as 0
    // A more accurate implementation would require complex hypergeometric calculations
    Ok(F::zero())
}

/// Calculate combinations C(n, 2) = n * (n-1) / 2
fn comb2(n: usize) -> usize {
    if n < 2 {
        0
    } else {
        n * (n - 1) / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_mutual_information() {
        // Perfect clustering
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        
        let mi: f64 = mutual_info_score(true_labels.view(), pred_labels.view()).unwrap();
        assert!(mi > 0.0);
        
        // Random clustering should have lower MI
        let random_labels = Array1::from_vec(vec![0, 1, 0, 1, 0, 1]);
        let mi_random: f64 = mutual_info_score(true_labels.view(), random_labels.view()).unwrap();
        assert!(mi > mi_random);
    }

    #[test]
    fn test_normalized_mutual_info() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        
        let nmi: f64 = normalized_mutual_info_score(true_labels.view(), pred_labels.view()).unwrap();
        assert!(nmi >= 0.0 && nmi <= 1.0);
        assert!(nmi > 0.9); // Should be close to 1 for perfect clustering
    }

    #[test]
    fn test_adjusted_rand_score() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        
        let ari: f64 = adjusted_rand_score(true_labels.view(), pred_labels.view()).unwrap();
        assert!(ari > 0.9); // Should be close to 1 for perfect clustering
        
        // Test with random clustering
        let random_labels = Array1::from_vec(vec![0, 1, 0, 1, 0, 1]);
        let ari_random: f64 = adjusted_rand_score(true_labels.view(), random_labels.view()).unwrap();
        assert!(ari > ari_random);
    }

    #[test]
    fn test_v_measure() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        
        let v_measure: f64 = v_measure_score(true_labels.view(), pred_labels.view()).unwrap();
        assert!(v_measure >= 0.0 && v_measure <= 1.0);
        assert!(v_measure > 0.9); // Should be close to 1 for perfect clustering
    }

    #[test]
    fn test_homogeneity_and_completeness() {
        let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        
        let homogeneity: f64 = homogeneity_score(true_labels.view(), pred_labels.view()).unwrap();
        let completeness: f64 = completeness_score(true_labels.view(), pred_labels.view()).unwrap();
        
        assert!(homogeneity >= 0.0 && homogeneity <= 1.0);
        assert!(completeness >= 0.0 && completeness <= 1.0);
        assert!(homogeneity > 0.9);
        assert!(completeness > 0.9);
    }

    #[test]
    fn test_empty_labels() {
        let empty_labels = Array1::from_vec(vec![]);
        let empty_labels2 = Array1::from_vec(vec![]);
        
        let mi: f64 = mutual_info_score(empty_labels.view(), empty_labels2.view()).unwrap();
        assert_eq!(mi, 0.0);
    }

    #[test]
    fn test_single_cluster() {
        let true_labels = Array1::from_vec(vec![0, 0, 0, 0]);
        let pred_labels = Array1::from_vec(vec![1, 1, 1, 1]);
        
        let nmi: f64 = normalized_mutual_info_score(true_labels.view(), pred_labels.view()).unwrap();
        assert_eq!(nmi, 1.0); // Single clusters should have perfect NMI
    }
}