//! Feature importance analysis for model explainability

use crate::error::{MetricsError, Result};
use ndarray::{Array1, Array2, ArrayView2};
use num_traits::Float;
use std::collections::HashMap;

/// Feature importance calculator
pub struct FeatureImportanceCalculator<F: Float> {
    /// Number of permutations for permutation importance
    pub n_permutations: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + num_traits::FromPrimitive + std::iter::Sum> Default
    for FeatureImportanceCalculator<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + num_traits::FromPrimitive + std::iter::Sum> FeatureImportanceCalculator<F> {
    /// Create new feature importance calculator
    pub fn new() -> Self {
        Self {
            n_permutations: 100,
            random_seed: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set number of permutations
    pub fn with_permutations(mut self, n: usize) -> Self {
        self.n_permutations = n;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Compute permutation importance
    pub fn permutation_importance<M, S>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        y_test: &Array1<F>,
        score_fn: S,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
        S: Fn(&Array1<F>, &Array1<F>) -> F,
    {
        // Baseline score
        let baseline_predictions = model(&x_test.view());
        let baseline_score = score_fn(&y_test, &baseline_predictions);

        let mut importance_scores = HashMap::new();
        let n_features = x_test.ncols();

        for (feature_idx, feature_name) in feature_names.iter().enumerate() {
            if feature_idx >= n_features {
                continue;
            }

            let mut permutation_scores = Vec::new();

            for _ in 0..self.n_permutations {
                let mut x_permuted = x_test.clone();
                self.permute_column(&mut x_permuted, feature_idx)?;

                let permuted_predictions = model(&x_permuted.view());
                let permuted_score = score_fn(&y_test, &permuted_predictions);

                let importance = baseline_score - permuted_score;
                permutation_scores.push(importance);
            }

            let mean_importance = permutation_scores.iter().cloned().sum::<F>()
                / F::from(permutation_scores.len()).unwrap();

            importance_scores.insert(feature_name.clone(), mean_importance);
        }

        Ok(importance_scores)
    }

    /// Compute feature importance using drop column method
    pub fn drop_column_importance<M, S>(
        &self,
        model: &M,
        x_test: &Array2<F>,
        y_test: &Array1<F>,
        score_fn: S,
        feature_names: &[String],
    ) -> Result<HashMap<String, F>>
    where
        M: Fn(&ArrayView2<F>) -> Array1<F>,
        S: Fn(&Array1<F>, &Array1<F>) -> F,
    {
        // Baseline score with all features
        let baseline_predictions = model(&x_test.view());
        let baseline_score = score_fn(&y_test, &baseline_predictions);

        let mut importance_scores = HashMap::new();
        let n_features = x_test.ncols();

        for (feature_idx, feature_name) in feature_names.iter().enumerate() {
            if feature_idx >= n_features {
                continue;
            }

            // Create dataset without this feature
            let _x_without_feature = self.drop_column(x_test, feature_idx)?;

            // Note: In practice, you'd need a model that can handle different input sizes
            // For this example, we'll set the dropped feature to zero instead
            let mut x_zeroed = x_test.clone();
            for i in 0..x_zeroed.nrows() {
                x_zeroed[[i, feature_idx]] = F::zero();
            }

            let reduced_predictions = model(&x_zeroed.view());
            let reduced_score = score_fn(&y_test, &reduced_predictions);

            let importance = baseline_score - reduced_score;
            importance_scores.insert(feature_name.clone(), importance);
        }

        Ok(importance_scores)
    }

    /// Compute feature importance statistics
    pub fn compute_importance_statistics(
        &self,
        importance_scores: &HashMap<String, F>,
    ) -> FeatureImportanceStats<F> {
        let values: Vec<F> = importance_scores.values().cloned().collect();

        if values.is_empty() {
            return FeatureImportanceStats::default();
        }

        let mean = values.iter().cloned().sum::<F>() / F::from(values.len()).unwrap();

        let variance = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<F>()
            / F::from(values.len()).unwrap();

        let std_dev = variance.sqrt();

        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if sorted_values.len() % 2 == 0 {
            let mid = sorted_values.len() / 2;
            (sorted_values[mid - 1] + sorted_values[mid]) / F::from(2).unwrap()
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        let min = sorted_values[0];
        let max = sorted_values[sorted_values.len() - 1];

        FeatureImportanceStats {
            mean,
            std_dev,
            median,
            min,
            max,
            n_features: values.len(),
        }
    }

    /// Get top-k most important features
    pub fn get_top_features(
        &self,
        importance_scores: &HashMap<String, F>,
        k: usize,
    ) -> Vec<(String, F)> {
        let mut sorted_features: Vec<(String, F)> = importance_scores
            .iter()
            .map(|(name, &score)| (name.clone(), score))
            .collect();

        sorted_features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_features.truncate(k);
        sorted_features
    }

    /// Filter features by importance threshold
    pub fn filter_by_threshold(
        &self,
        importance_scores: &HashMap<String, F>,
        threshold: F,
    ) -> HashMap<String, F> {
        importance_scores
            .iter()
            .filter(|(_, &score)| score >= threshold)
            .map(|(name, &score)| (name.clone(), score))
            .collect()
    }

    // Helper methods

    fn permute_column(&self, data: &mut Array2<F>, column_idx: usize) -> Result<()> {
        if column_idx >= data.ncols() {
            return Err(MetricsError::InvalidInput(
                "Column index out of bounds".to_string(),
            ));
        }

        let mut column_values: Vec<F> = data.column(column_idx).to_vec();

        // Simple permutation (in practice, would use proper random shuffle)
        for i in (1..column_values.len()).rev() {
            let j = match self.random_seed {
                Some(seed) => (seed as usize + i) % (i + 1),
                None => i % (i + 1),
            };
            column_values.swap(i, j);
        }

        for (i, &value) in column_values.iter().enumerate() {
            data[[i, column_idx]] = value;
        }

        Ok(())
    }

    fn drop_column(&self, data: &Array2<F>, column_idx: usize) -> Result<Array2<F>> {
        if column_idx >= data.ncols() {
            return Err(MetricsError::InvalidInput(
                "Column index out of bounds".to_string(),
            ));
        }

        let n_rows = data.nrows();
        let n_cols = data.ncols() - 1;
        let mut result = Array2::zeros((n_rows, n_cols));

        for i in 0..n_rows {
            let mut result_col = 0;
            for j in 0..data.ncols() {
                if j != column_idx {
                    result[[i, result_col]] = data[[i, j]];
                    result_col += 1;
                }
            }
        }

        Ok(result)
    }
}

/// Feature importance statistics
#[derive(Debug, Clone)]
pub struct FeatureImportanceStats<F: Float> {
    pub mean: F,
    pub std_dev: F,
    pub median: F,
    pub min: F,
    pub max: F,
    pub n_features: usize,
}

impl<F: Float> Default for FeatureImportanceStats<F> {
    fn default() -> Self {
        Self {
            mean: F::zero(),
            std_dev: F::zero(),
            median: F::zero(),
            min: F::zero(),
            max: F::zero(),
            n_features: 0,
        }
    }
}

/// Compute mutual information based feature importance
pub fn mutual_information_importance<F: Float + num_traits::FromPrimitive + std::iter::Sum>(
    x: &Array2<F>,
    y: &Array1<F>,
    feature_names: &[String],
) -> Result<HashMap<String, F>> {
    let mut importance_scores = HashMap::new();

    for (i, feature_name) in feature_names.iter().enumerate() {
        if i >= x.ncols() {
            continue;
        }

        let feature_column = x.column(i);
        let mi_score = compute_mutual_information(&feature_column, y)?;
        importance_scores.insert(feature_name.clone(), mi_score);
    }

    Ok(importance_scores)
}

/// Compute mutual information between two variables (simplified)
fn compute_mutual_information<F: Float + num_traits::FromPrimitive + std::iter::Sum>(
    x: &ndarray::ArrayView1<F>,
    y: &Array1<F>,
) -> Result<F> {
    if x.len() != y.len() {
        return Err(MetricsError::InvalidInput(
            "Variables must have the same length".to_string(),
        ));
    }

    // Simplified mutual information calculation
    // In practice, this would involve binning continuous variables and computing entropy

    let correlation = compute_correlation(x, y)?;
    let mi_approx = -F::from(0.5).unwrap() * (F::one() - correlation * correlation).ln();

    Ok(mi_approx.max(F::zero()))
}

/// Compute correlation coefficient
fn compute_correlation<F: Float + std::iter::Sum>(
    x: &ndarray::ArrayView1<F>,
    y: &Array1<F>,
) -> Result<F> {
    let n = F::from(x.len()).unwrap();

    let mean_x = x.iter().cloned().sum::<F>() / n;
    let mean_y = y.iter().cloned().sum::<F>() / n;

    let numerator: F = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    let sum_sq_x: F = x.iter().map(|&xi| (xi - mean_x) * (xi - mean_x)).sum();
    let sum_sq_y: F = y.iter().map(|&yi| (yi - mean_y) * (yi - mean_y)).sum();

    let denominator = (sum_sq_x * sum_sq_y).sqrt();

    if denominator == F::zero() {
        Ok(F::zero())
    } else {
        Ok(numerator / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_feature_importance_calculator() {
        let calculator = FeatureImportanceCalculator::<f64>::new()
            .with_permutations(10)
            .with_seed(42);

        assert_eq!(calculator.n_permutations, 10);
        assert_eq!(calculator.random_seed, Some(42));
    }

    #[test]
    fn test_drop_column() {
        let calculator = FeatureImportanceCalculator::<f64>::new();
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let result = calculator.drop_column(&data, 1).unwrap();

        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 1]], 3.0);
        assert_eq!(result[[1, 0]], 4.0);
        assert_eq!(result[[1, 1]], 6.0);
    }

    #[test]
    fn test_importance_statistics() {
        let calculator = FeatureImportanceCalculator::<f64>::new();
        let mut scores = HashMap::new();
        scores.insert("feature1".to_string(), 0.5);
        scores.insert("feature2".to_string(), 0.3);
        scores.insert("feature3".to_string(), 0.8);
        scores.insert("feature4".to_string(), 0.1);

        let stats = calculator.compute_importance_statistics(&scores);

        assert_eq!(stats.n_features, 4);
        assert!((stats.mean - 0.425).abs() < 1e-10);
        assert_eq!(stats.min, 0.1);
        assert_eq!(stats.max, 0.8);
    }

    #[test]
    fn test_top_features() {
        let calculator = FeatureImportanceCalculator::<f64>::new();
        let mut scores = HashMap::new();
        scores.insert("feature1".to_string(), 0.5);
        scores.insert("feature2".to_string(), 0.3);
        scores.insert("feature3".to_string(), 0.8);
        scores.insert("feature4".to_string(), 0.1);

        let top_features = calculator.get_top_features(&scores, 2);

        assert_eq!(top_features.len(), 2);
        assert_eq!(top_features[0].0, "feature3");
        assert_eq!(top_features[0].1, 0.8);
        assert_eq!(top_features[1].0, "feature1");
        assert_eq!(top_features[1].1, 0.5);
    }

    #[test]
    fn test_threshold_filtering() {
        let calculator = FeatureImportanceCalculator::<f64>::new();
        let mut scores = HashMap::new();
        scores.insert("feature1".to_string(), 0.5);
        scores.insert("feature2".to_string(), 0.3);
        scores.insert("feature3".to_string(), 0.8);
        scores.insert("feature4".to_string(), 0.1);

        let filtered = calculator.filter_by_threshold(&scores, 0.4);

        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains_key("feature1"));
        assert!(filtered.contains_key("feature3"));
    }

    #[test]
    fn test_correlation_computation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let correlation = compute_correlation(&x.view(), &y).unwrap();
        assert!((correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information_importance() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];
        let feature_names = vec!["feature1".to_string(), "feature2".to_string()];

        let importance = mutual_information_importance(&x, &y, &feature_names).unwrap();

        assert_eq!(importance.len(), 2);
        assert!(importance.contains_key("feature1"));
        assert!(importance.contains_key("feature2"));
    }
}
