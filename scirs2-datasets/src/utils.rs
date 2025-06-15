//! Utility functions and data structures for datasets

use crate::error::{DatasetsError, Result};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Helper module for serializing ndarray types with serde
mod serde_array {
    use ndarray::{Array1, Array2};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::vec::Vec;

    /// Serialize a 2D array to a format compatible with serde
    pub fn serialize_array2<S>(array: &Array2<f64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let shape = array.shape();
        let mut vec = Vec::with_capacity(shape[0] * shape[1] + 2);

        // Store shape at the beginning
        vec.push(shape[0] as f64);
        vec.push(shape[1] as f64);

        // Store data
        vec.extend(array.iter().cloned());

        vec.serialize(serializer)
    }

    /// Deserialize a 2D array from a serde-compatible format
    pub fn deserialize_array2<'de, D>(deserializer: D) -> Result<Array2<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec = Vec::<f64>::deserialize(deserializer)?;
        if vec.len() < 2 {
            return Err(serde::de::Error::custom("Invalid array2 serialization"));
        }

        let nrows = vec[0] as usize;
        let ncols = vec[1] as usize;

        if vec.len() != nrows * ncols + 2 {
            return Err(serde::de::Error::custom("Invalid array2 serialization"));
        }

        let data = vec[2..].to_vec();
        match Array2::from_shape_vec((nrows, ncols), data) {
            Ok(array) => Ok(array),
            Err(_) => Err(serde::de::Error::custom("Failed to reshape array2")),
        }
    }

    /// Serialize a 1D array to a format compatible with serde
    #[allow(dead_code)]
    pub fn serialize_array1<S>(array: &Array1<f64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let vec = array.to_vec();
        vec.serialize(serializer)
    }

    /// Deserialize a 1D array from a serde-compatible format
    pub fn deserialize_array1<'de, D>(deserializer: D) -> Result<Array1<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec = Vec::<f64>::deserialize(deserializer)?;
        Ok(Array1::from(vec))
    }
}

/// Represents a dataset with features, optional targets, and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    /// Features/data matrix (n_samples, n_features)
    #[serde(
        serialize_with = "serde_array::serialize_array2",
        deserialize_with = "serde_array::deserialize_array2"
    )]
    pub data: Array2<f64>,

    /// Optional target values
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<Array1<f64>>,

    /// Optional target names for classification problems
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_names: Option<Vec<String>>,

    /// Optional feature names
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_names: Option<Vec<String>>,

    /// Optional descriptions for each feature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_descriptions: Option<Vec<String>>,

    /// Optional dataset description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Optional dataset metadata
    pub metadata: HashMap<String, String>,
}

/// Helper module for serializing Option<Array1<f64>>
mod optional_array1 {
    use super::serde_array;
    use ndarray::Array1;
    use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

    #[allow(dead_code)]
    pub fn serialize<S>(array_opt: &Option<Array1<f64>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match array_opt {
            Some(array) => {
                #[derive(Serialize)]
                struct Helper<'a>(&'a Array1<f64>);

                #[derive(Serialize)]
                struct Wrapper<'a> {
                    #[serde(
                        serialize_with = "serde_array::serialize_array1",
                        deserialize_with = "serde_array::deserialize_array1"
                    )]
                    value: &'a Array1<f64>,
                }

                Wrapper { value: array }.serialize(serializer)
            }
            None => serializer.serialize_none(),
        }
    }

    #[allow(dead_code)]
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Array1<f64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wrapper {
            #[serde(
                serialize_with = "serde_array::serialize_array1",
                deserialize_with = "serde_array::deserialize_array1"
            )]
            #[allow(dead_code)]
            value: Array1<f64>,
        }

        Option::<Wrapper>::deserialize(deserializer).map(|opt_wrapper| opt_wrapper.map(|w| w.value))
    }
}

impl Dataset {
    /// Create a new dataset with the given data and target
    pub fn new(data: Array2<f64>, target: Option<Array1<f64>>) -> Self {
        Dataset {
            data,
            target,
            target_names: None,
            feature_names: None,
            feature_descriptions: None,
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Add target names to the dataset
    pub fn with_target_names(mut self, target_names: Vec<String>) -> Self {
        self.target_names = Some(target_names);
        self
    }

    /// Add feature names to the dataset
    pub fn with_feature_names(mut self, feature_names: Vec<String>) -> Self {
        self.feature_names = Some(feature_names);
        self
    }

    /// Add feature descriptions to the dataset
    pub fn with_feature_descriptions(mut self, feature_descriptions: Vec<String>) -> Self {
        self.feature_descriptions = Some(feature_descriptions);
        self
    }

    /// Add a description to the dataset
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Add metadata to the dataset
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Get the number of samples in the dataset
    pub fn n_samples(&self) -> usize {
        self.data.nrows()
    }

    /// Get the number of features in the dataset
    pub fn n_features(&self) -> usize {
        self.data.ncols()
    }

    /// Split the dataset into training and test sets
    pub fn train_test_split(
        &self,
        test_size: f64,
        random_seed: Option<u64>,
    ) -> Result<(Dataset, Dataset)> {
        if test_size <= 0.0 || test_size >= 1.0 {
            return Err(DatasetsError::InvalidFormat(
                "test_size must be between 0 and 1".to_string(),
            ));
        }

        let n_samples = self.n_samples();
        let n_test = (n_samples as f64 * test_size).round() as usize;
        let n_train = n_samples - n_test;

        if n_train == 0 || n_test == 0 {
            return Err(DatasetsError::InvalidFormat(
                "Both train and test sets must have at least one sample".to_string(),
            ));
        }

        // Create shuffled indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = match random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut r = rng();
                StdRng::seed_from_u64(r.next_u64())
            }
        };
        indices.shuffle(&mut rng);

        let train_indices = &indices[0..n_train];
        let test_indices = &indices[n_train..];

        // Create training dataset
        let train_data = self.data.select(ndarray::Axis(0), train_indices);
        let train_target = self
            .target
            .as_ref()
            .map(|t| t.select(ndarray::Axis(0), train_indices));

        let mut train_dataset = Dataset::new(train_data, train_target);
        if let Some(feature_names) = &self.feature_names {
            train_dataset = train_dataset.with_feature_names(feature_names.clone());
        }
        if let Some(description) = &self.description {
            train_dataset = train_dataset.with_description(description.clone());
        }

        // Create test dataset
        let test_data = self.data.select(ndarray::Axis(0), test_indices);
        let test_target = self
            .target
            .as_ref()
            .map(|t| t.select(ndarray::Axis(0), test_indices));

        let mut test_dataset = Dataset::new(test_data, test_target);
        if let Some(feature_names) = &self.feature_names {
            test_dataset = test_dataset.with_feature_names(feature_names.clone());
        }
        if let Some(description) = &self.description {
            test_dataset = test_dataset.with_description(description.clone());
        }

        Ok((train_dataset, test_dataset))
    }
}

/// Helper function to normalize data (zero mean, unit variance)
///
/// This function normalizes each feature (column) in the dataset to have zero mean
/// and unit variance. This is commonly used as a preprocessing step for machine learning.
///
/// # Arguments
///
/// * `data` - A mutable reference to the data array to normalize in-place
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use scirs2_datasets::utils::normalize;
///
/// let mut data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// normalize(&mut data);
/// // data is now normalized with zero mean and unit variance for each feature
/// ```
pub fn normalize(data: &mut Array2<f64>) {
    let n_features = data.ncols();

    for j in 0..n_features {
        let mut column = data.column_mut(j);

        // Calculate mean and std
        let mean = column.mean().unwrap_or(0.0);
        let std = column.std(0.0);

        // Avoid division by zero
        if std > 1e-10 {
            column.mapv_inplace(|x| (x - mean) / std);
        }
    }
}

/// Cross-validation fold indices
pub type CrossValidationFolds = Vec<(Vec<usize>, Vec<usize>)>;

/// Performs K-fold cross-validation splitting
///
/// Splits the dataset into k consecutive folds. Each fold is used once as a validation
/// set while the remaining k-1 folds form the training set.
///
/// # Arguments
///
/// * `n_samples` - Number of samples in the dataset
/// * `n_folds` - Number of folds (must be >= 2 and <= n_samples)
/// * `shuffle` - Whether to shuffle the data before splitting
/// * `random_seed` - Optional random seed for reproducible shuffling
///
/// # Returns
///
/// A vector of (train_indices, validation_indices) tuples for each fold
///
/// # Example
///
/// ```
/// use scirs2_datasets::utils::k_fold_split;
///
/// let folds = k_fold_split(10, 3, true, Some(42)).unwrap();
/// assert_eq!(folds.len(), 3);
/// ```
pub fn k_fold_split(
    n_samples: usize,
    n_folds: usize,
    shuffle: bool,
    random_seed: Option<u64>,
) -> Result<CrossValidationFolds> {
    if n_folds < 2 {
        return Err(DatasetsError::InvalidFormat(
            "Number of folds must be at least 2".to_string(),
        ));
    }

    if n_folds > n_samples {
        return Err(DatasetsError::InvalidFormat(
            "Number of folds cannot exceed number of samples".to_string(),
        ));
    }

    let mut indices: Vec<usize> = (0..n_samples).collect();

    if shuffle {
        let mut rng = match random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut r = rng();
                StdRng::seed_from_u64(r.next_u64())
            }
        };
        indices.shuffle(&mut rng);
    }

    let mut folds = Vec::new();
    let fold_size = n_samples / n_folds;
    let remainder = n_samples % n_folds;

    for i in 0..n_folds {
        let start = i * fold_size + i.min(remainder);
        let end = start + fold_size + if i < remainder { 1 } else { 0 };

        let validation_indices = indices[start..end].to_vec();
        let mut train_indices = Vec::new();
        train_indices.extend(&indices[0..start]);
        train_indices.extend(&indices[end..]);

        folds.push((train_indices, validation_indices));
    }

    Ok(folds)
}

/// Performs stratified K-fold cross-validation splitting
///
/// Splits the dataset into k folds while preserving the percentage of samples
/// for each target class in each fold.
///
/// # Arguments
///
/// * `targets` - Target values for stratification
/// * `n_folds` - Number of folds (must be >= 2)
/// * `shuffle` - Whether to shuffle the data before splitting
/// * `random_seed` - Optional random seed for reproducible shuffling
///
/// # Returns
///
/// A vector of (train_indices, validation_indices) tuples for each fold
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_datasets::utils::stratified_k_fold_split;
///
/// let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
/// let folds = stratified_k_fold_split(&targets, 2, true, Some(42)).unwrap();
/// assert_eq!(folds.len(), 2);
/// ```
pub fn stratified_k_fold_split(
    targets: &Array1<f64>,
    n_folds: usize,
    shuffle: bool,
    random_seed: Option<u64>,
) -> Result<CrossValidationFolds> {
    if n_folds < 2 {
        return Err(DatasetsError::InvalidFormat(
            "Number of folds must be at least 2".to_string(),
        ));
    }

    let n_samples = targets.len();
    if n_folds > n_samples {
        return Err(DatasetsError::InvalidFormat(
            "Number of folds cannot exceed number of samples".to_string(),
        ));
    }

    // Group indices by target class
    let mut class_indices: std::collections::HashMap<i64, Vec<usize>> =
        std::collections::HashMap::new();

    for (i, &target) in targets.iter().enumerate() {
        let class = target.round() as i64;
        class_indices.entry(class).or_default().push(i);
    }

    // Shuffle indices within each class if requested
    if shuffle {
        let mut rng = match random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut r = rng();
                StdRng::seed_from_u64(r.next_u64())
            }
        };

        for indices in class_indices.values_mut() {
            indices.shuffle(&mut rng);
        }
    }

    // Create folds while maintaining class proportions
    let mut folds = vec![Vec::new(); n_folds];

    for (_, indices) in class_indices {
        let class_size = indices.len();
        let fold_size = class_size / n_folds;
        let remainder = class_size % n_folds;

        for (i, fold) in folds.iter_mut().enumerate() {
            let start = i * fold_size + i.min(remainder);
            let end = start + fold_size + if i < remainder { 1 } else { 0 };
            fold.extend(&indices[start..end]);
        }
    }

    // Convert to (train, validation) pairs
    let cv_folds = (0..n_folds)
        .map(|i| {
            let validation_indices = folds[i].clone();
            let mut train_indices = Vec::new();
            for (j, fold) in folds.iter().enumerate() {
                if i != j {
                    train_indices.extend(fold);
                }
            }
            (train_indices, validation_indices)
        })
        .collect();

    Ok(cv_folds)
}

/// Performs time series cross-validation splitting
///
/// Creates splits suitable for time series data where future observations
/// should not be used to predict past observations. Each training set contains
/// all observations up to a certain point, and the validation set contains
/// the next `n_test_samples` observations.
///
/// # Arguments
///
/// * `n_samples` - Number of samples in the dataset
/// * `n_splits` - Number of splits to create
/// * `n_test_samples` - Number of samples in each test set
/// * `gap` - Number of samples to skip between train and test sets (default: 0)
///
/// # Returns
///
/// A vector of (train_indices, validation_indices) tuples for each split
///
/// # Example
///
/// ```
/// use scirs2_datasets::utils::time_series_split;
///
/// let folds = time_series_split(100, 5, 10, 0).unwrap();
/// assert_eq!(folds.len(), 5);
/// ```
pub fn time_series_split(
    n_samples: usize,
    n_splits: usize,
    n_test_samples: usize,
    gap: usize,
) -> Result<CrossValidationFolds> {
    if n_splits < 1 {
        return Err(DatasetsError::InvalidFormat(
            "Number of splits must be at least 1".to_string(),
        ));
    }

    if n_test_samples < 1 {
        return Err(DatasetsError::InvalidFormat(
            "Number of test samples must be at least 1".to_string(),
        ));
    }

    // Calculate minimum samples needed
    let min_samples_needed = n_test_samples + gap + n_splits;
    if n_samples < min_samples_needed {
        return Err(DatasetsError::InvalidFormat(format!(
            "Not enough samples for time series split. Need at least {}, got {}",
            min_samples_needed, n_samples
        )));
    }

    let mut folds = Vec::new();
    let test_starts = (0..n_splits)
        .map(|i| {
            let split_size = (n_samples - n_test_samples - gap) / n_splits;
            split_size * (i + 1) + gap
        })
        .collect::<Vec<_>>();

    for &test_start in &test_starts {
        let train_end = test_start - gap;
        let test_end = test_start + n_test_samples;

        if test_end > n_samples {
            break;
        }

        let train_indices = (0..train_end).collect();
        let test_indices = (test_start..test_end).collect();

        folds.push((train_indices, test_indices));
    }

    if folds.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "Could not create any valid time series splits".to_string(),
        ));
    }

    Ok(folds)
}

/// Performs random sampling with or without replacement
///
/// # Arguments
///
/// * `n_samples` - Total number of samples in the dataset
/// * `sample_size` - Number of samples to draw
/// * `replace` - Whether to sample with replacement (bootstrap)
/// * `random_seed` - Optional random seed for reproducible sampling
///
/// # Returns
///
/// A vector of indices representing the sampled data points
///
/// # Example
///
/// ```
/// use scirs2_datasets::utils::random_sample;
///
/// // Sample 5 indices from 10 total samples without replacement
/// let indices = random_sample(10, 5, false, Some(42)).unwrap();
/// assert_eq!(indices.len(), 5);
///
/// // Bootstrap sampling (with replacement)
/// let bootstrap_indices = random_sample(10, 15, true, Some(42)).unwrap();
/// assert_eq!(bootstrap_indices.len(), 15);
/// ```
pub fn random_sample(
    n_samples: usize,
    sample_size: usize,
    replace: bool,
    random_seed: Option<u64>,
) -> Result<Vec<usize>> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Number of samples must be > 0".to_string(),
        ));
    }

    if sample_size == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Sample size must be > 0".to_string(),
        ));
    }

    if !replace && sample_size > n_samples {
        return Err(DatasetsError::InvalidFormat(format!(
            "Cannot sample {} items from {} without replacement",
            sample_size, n_samples
        )));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut indices = Vec::with_capacity(sample_size);

    if replace {
        // Bootstrap sampling (with replacement)
        for _ in 0..sample_size {
            indices.push(rng.random_range(0..n_samples));
        }
    } else {
        // Sampling without replacement
        let mut available: Vec<usize> = (0..n_samples).collect();
        available.shuffle(&mut rng);
        indices.extend_from_slice(&available[0..sample_size]);
    }

    Ok(indices)
}

/// Performs stratified random sampling
///
/// Maintains the same class distribution in the sample as in the original dataset.
///
/// # Arguments
///
/// * `targets` - Target values for stratification
/// * `sample_size` - Number of samples to draw
/// * `random_seed` - Optional random seed for reproducible sampling
///
/// # Returns
///
/// A vector of indices representing the stratified sample
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_datasets::utils::stratified_sample;
///
/// let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
/// let indices = stratified_sample(&targets, 6, Some(42)).unwrap();
/// assert_eq!(indices.len(), 6);
/// ```
pub fn stratified_sample(
    targets: &Array1<f64>,
    sample_size: usize,
    random_seed: Option<u64>,
) -> Result<Vec<usize>> {
    if targets.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "Targets array cannot be empty".to_string(),
        ));
    }

    if sample_size == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Sample size must be > 0".to_string(),
        ));
    }

    if sample_size > targets.len() {
        return Err(DatasetsError::InvalidFormat(format!(
            "Cannot sample {} items from {} total samples",
            sample_size,
            targets.len()
        )));
    }

    // Group indices by target class
    let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &target) in targets.iter().enumerate() {
        let class = target.round() as i64;
        class_indices.entry(class).or_default().push(i);
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut stratified_indices = Vec::new();
    let n_classes = class_indices.len();
    let base_samples_per_class = sample_size / n_classes;
    let remainder = sample_size % n_classes;

    let mut class_list: Vec<_> = class_indices.keys().cloned().collect();
    class_list.sort();

    for (i, &class) in class_list.iter().enumerate() {
        let class_samples = class_indices.get(&class).unwrap();
        let samples_for_this_class = if i < remainder {
            base_samples_per_class + 1
        } else {
            base_samples_per_class
        };

        if samples_for_this_class > class_samples.len() {
            return Err(DatasetsError::InvalidFormat(format!(
                "Class {} has only {} samples but needs {} for stratified sampling",
                class,
                class_samples.len(),
                samples_for_this_class
            )));
        }

        // Sample from this class
        let sampled_indices = random_sample(
            class_samples.len(),
            samples_for_this_class,
            false,
            Some(rng.next_u64()),
        )?;

        for &idx in &sampled_indices {
            stratified_indices.push(class_samples[idx]);
        }
    }

    stratified_indices.shuffle(&mut rng);
    Ok(stratified_indices)
}

/// Performs importance sampling based on provided weights
///
/// Samples indices according to the provided probability weights. Higher weights
/// increase the probability of selection.
///
/// # Arguments
///
/// * `weights` - Probability weights for each sample (must be non-negative)
/// * `sample_size` - Number of samples to draw
/// * `replace` - Whether to sample with replacement
/// * `random_seed` - Optional random seed for reproducible sampling
///
/// # Returns
///
/// A vector of indices representing the importance-weighted sample
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_datasets::utils::importance_sample;
///
/// // Give higher weights to the last few samples
/// let weights = Array1::from(vec![0.1, 0.1, 0.1, 0.8, 0.9, 1.0]);
/// let indices = importance_sample(&weights, 3, false, Some(42)).unwrap();
/// assert_eq!(indices.len(), 3);
/// ```
pub fn importance_sample(
    weights: &Array1<f64>,
    sample_size: usize,
    replace: bool,
    random_seed: Option<u64>,
) -> Result<Vec<usize>> {
    if weights.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "Weights array cannot be empty".to_string(),
        ));
    }

    if sample_size == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Sample size must be > 0".to_string(),
        ));
    }

    if !replace && sample_size > weights.len() {
        return Err(DatasetsError::InvalidFormat(format!(
            "Cannot sample {} items from {} without replacement",
            sample_size,
            weights.len()
        )));
    }

    // Check for negative weights
    for &weight in weights.iter() {
        if weight < 0.0 {
            return Err(DatasetsError::InvalidFormat(
                "All weights must be non-negative".to_string(),
            ));
        }
    }

    let weight_sum: f64 = weights.sum();
    if weight_sum <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "Sum of weights must be positive".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut indices = Vec::with_capacity(sample_size);
    let mut available_weights = weights.clone();
    let mut available_indices: Vec<usize> = (0..weights.len()).collect();

    for _ in 0..sample_size {
        let current_sum = available_weights.sum();
        if current_sum <= 0.0 {
            break;
        }

        // Generate random number between 0 and current_sum
        let random_value = rng.random_range(0.0..current_sum);

        // Find the index corresponding to this random value
        let mut cumulative_weight = 0.0;
        let mut selected_idx = 0;

        for (i, &weight) in available_weights.iter().enumerate() {
            cumulative_weight += weight;
            if random_value <= cumulative_weight {
                selected_idx = i;
                break;
            }
        }

        let original_idx = available_indices[selected_idx];
        indices.push(original_idx);

        if !replace {
            // Remove the selected item for sampling without replacement
            available_weights = Array1::from_iter(
                available_weights
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != selected_idx)
                    .map(|(_, &w)| w),
            );
            available_indices.remove(selected_idx);
        }
    }

    Ok(indices)
}

/// Performs random oversampling to balance class distribution
///
/// Duplicates samples from minority classes to match the majority class size.
/// This is useful for handling imbalanced datasets in classification problems.
///
/// # Arguments
///
/// * `data` - Feature matrix (n_samples, n_features)
/// * `targets` - Target values for each sample
/// * `random_seed` - Optional random seed for reproducible sampling
///
/// # Returns
///
/// A tuple containing the resampled (data, targets) arrays
///
/// # Example
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_datasets::utils::random_oversample;
///
/// let data = Array2::from_shape_vec((6, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
/// let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]); // Imbalanced: 2 vs 4
/// let (balanced_data, balanced_targets) = random_oversample(&data, &targets, Some(42)).unwrap();
/// // Now both classes have 4 samples each
/// ```
pub fn random_oversample(
    data: &Array2<f64>,
    targets: &Array1<f64>,
    random_seed: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if data.nrows() != targets.len() {
        return Err(DatasetsError::InvalidFormat(
            "Data rows and targets length must match".to_string(),
        ));
    }

    if data.is_empty() || targets.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "Data and targets cannot be empty".to_string(),
        ));
    }

    // Group indices by class
    let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &target) in targets.iter().enumerate() {
        let class = target.round() as i64;
        class_indices.entry(class).or_default().push(i);
    }

    // Find the majority class size
    let max_class_size = class_indices.values().map(|v| v.len()).max().unwrap();

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Collect all resampled indices
    let mut resampled_indices = Vec::new();

    for (_, indices) in class_indices {
        let class_size = indices.len();

        // Add all original samples
        resampled_indices.extend(&indices);

        // Oversample if this class is smaller than the majority class
        if class_size < max_class_size {
            let samples_needed = max_class_size - class_size;
            for _ in 0..samples_needed {
                let random_idx = rng.random_range(0..class_size);
                resampled_indices.push(indices[random_idx]);
            }
        }
    }

    // Create resampled data and targets
    let resampled_data = data.select(ndarray::Axis(0), &resampled_indices);
    let resampled_targets = targets.select(ndarray::Axis(0), &resampled_indices);

    Ok((resampled_data, resampled_targets))
}

/// Performs random undersampling to balance class distribution
///
/// Randomly removes samples from majority classes to match the minority class size.
/// This reduces the overall dataset size but maintains balance.
///
/// # Arguments
///
/// * `data` - Feature matrix (n_samples, n_features)
/// * `targets` - Target values for each sample
/// * `random_seed` - Optional random seed for reproducible sampling
///
/// # Returns
///
/// A tuple containing the undersampled (data, targets) arrays
///
/// # Example
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_datasets::utils::random_undersample;
///
/// let data = Array2::from_shape_vec((6, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
/// let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]); // Imbalanced: 2 vs 4
/// let (balanced_data, balanced_targets) = random_undersample(&data, &targets, Some(42)).unwrap();
/// // Now both classes have 2 samples each
/// ```
pub fn random_undersample(
    data: &Array2<f64>,
    targets: &Array1<f64>,
    random_seed: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if data.nrows() != targets.len() {
        return Err(DatasetsError::InvalidFormat(
            "Data rows and targets length must match".to_string(),
        ));
    }

    if data.is_empty() || targets.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "Data and targets cannot be empty".to_string(),
        ));
    }

    // Group indices by class
    let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &target) in targets.iter().enumerate() {
        let class = target.round() as i64;
        class_indices.entry(class).or_default().push(i);
    }

    // Find the minority class size
    let min_class_size = class_indices.values().map(|v| v.len()).min().unwrap();

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Collect undersampled indices
    let mut undersampled_indices = Vec::new();

    for (_, mut indices) in class_indices {
        if indices.len() > min_class_size {
            // Randomly sample down to minority class size
            indices.shuffle(&mut rng);
            undersampled_indices.extend(&indices[0..min_class_size]);
        } else {
            // Use all samples if already at or below minority class size
            undersampled_indices.extend(&indices);
        }
    }

    // Create undersampled data and targets
    let undersampled_data = data.select(ndarray::Axis(0), &undersampled_indices);
    let undersampled_targets = targets.select(ndarray::Axis(0), &undersampled_indices);

    Ok((undersampled_data, undersampled_targets))
}

/// Generates synthetic samples using SMOTE-like interpolation
///
/// Creates synthetic samples by interpolating between existing samples within each class.
/// This is useful for oversampling minority classes without simple duplication.
///
/// # Arguments
///
/// * `data` - Feature matrix (n_samples, n_features)
/// * `targets` - Target values for each sample
/// * `target_class` - The class to generate synthetic samples for
/// * `n_synthetic` - Number of synthetic samples to generate
/// * `k_neighbors` - Number of nearest neighbors to consider for interpolation
/// * `random_seed` - Optional random seed for reproducible generation
///
/// # Returns
///
/// A tuple containing the synthetic (data, targets) arrays
///
/// # Example
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_datasets::utils::generate_synthetic_samples;
///
/// let data = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 1.5, 1.5, 2.5, 2.5]).unwrap();
/// let targets = Array1::from(vec![0.0, 0.0, 0.0, 1.0]);
/// let (synthetic_data, synthetic_targets) = generate_synthetic_samples(&data, &targets, 0.0, 2, 2, Some(42)).unwrap();
/// assert_eq!(synthetic_data.nrows(), 2);
/// assert_eq!(synthetic_targets.len(), 2);
/// ```
pub fn generate_synthetic_samples(
    data: &Array2<f64>,
    targets: &Array1<f64>,
    target_class: f64,
    n_synthetic: usize,
    k_neighbors: usize,
    random_seed: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if data.nrows() != targets.len() {
        return Err(DatasetsError::InvalidFormat(
            "Data rows and targets length must match".to_string(),
        ));
    }

    if n_synthetic == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Number of synthetic samples must be > 0".to_string(),
        ));
    }

    if k_neighbors == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Number of neighbors must be > 0".to_string(),
        ));
    }

    // Find samples belonging to the target class
    let class_indices: Vec<usize> = targets
        .iter()
        .enumerate()
        .filter(|(_, &target)| (target - target_class).abs() < 1e-10)
        .map(|(i, _)| i)
        .collect();

    if class_indices.len() < 2 {
        return Err(DatasetsError::InvalidFormat(
            "Need at least 2 samples of the target class for synthetic generation".to_string(),
        ));
    }

    if k_neighbors >= class_indices.len() {
        return Err(DatasetsError::InvalidFormat(
            "k_neighbors must be less than the number of samples in the target class".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let n_features = data.ncols();
    let mut synthetic_data = Array2::zeros((n_synthetic, n_features));
    let synthetic_targets = Array1::from_elem(n_synthetic, target_class);

    for i in 0..n_synthetic {
        // Randomly select a sample from the target class
        let base_idx = class_indices[rng.random_range(0..class_indices.len())];
        let base_sample = data.row(base_idx);

        // Find k nearest neighbors within the same class
        let mut distances: Vec<(usize, f64)> = class_indices
            .iter()
            .filter(|&&idx| idx != base_idx)
            .map(|&idx| {
                let neighbor = data.row(idx);
                let distance: f64 = base_sample
                    .iter()
                    .zip(neighbor.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (idx, distance)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let k_nearest = &distances[0..k_neighbors.min(distances.len())];

        // Select a random neighbor from the k nearest
        let neighbor_idx = k_nearest[rng.random_range(0..k_nearest.len())].0;
        let neighbor_sample = data.row(neighbor_idx);

        // Generate synthetic sample by interpolation
        let alpha = rng.random_range(0.0..1.0);
        for (j, synthetic_feature) in synthetic_data.row_mut(i).iter_mut().enumerate() {
            *synthetic_feature = base_sample[j] + alpha * (neighbor_sample[j] - base_sample[j]);
        }
    }

    Ok((synthetic_data, synthetic_targets))
}

/// Creates a balanced dataset using the specified balancing strategy
///
/// Automatically balances the dataset by applying oversampling, undersampling,
/// or synthetic sample generation based on the specified strategy.
///
/// # Arguments
///
/// * `data` - Feature matrix (n_samples, n_features)
/// * `targets` - Target values for each sample
/// * `strategy` - Balancing strategy to use
/// * `random_seed` - Optional random seed for reproducible balancing
///
/// # Returns
///
/// A tuple containing the balanced (data, targets) arrays
///
/// # Example
///
/// ```
/// use ndarray::{Array1, Array2};
/// use scirs2_datasets::utils::{create_balanced_dataset, BalancingStrategy};
///
/// let data = Array2::from_shape_vec((6, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
/// let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
/// let (balanced_data, balanced_targets) = create_balanced_dataset(&data, &targets, BalancingStrategy::RandomOversample, Some(42)).unwrap();
/// ```
pub fn create_balanced_dataset(
    data: &Array2<f64>,
    targets: &Array1<f64>,
    strategy: BalancingStrategy,
    random_seed: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    match strategy {
        BalancingStrategy::RandomOversample => random_oversample(data, targets, random_seed),
        BalancingStrategy::RandomUndersample => random_undersample(data, targets, random_seed),
        BalancingStrategy::SMOTE { k_neighbors } => {
            // Apply SMOTE to minority classes
            let mut class_counts: HashMap<i64, usize> = HashMap::new();
            for &target in targets.iter() {
                let class = target.round() as i64;
                *class_counts.entry(class).or_default() += 1;
            }

            let max_count = *class_counts.values().max().unwrap();
            let mut combined_data = data.clone();
            let mut combined_targets = targets.clone();

            for (&class, &count) in &class_counts {
                if count < max_count {
                    let samples_needed = max_count - count;
                    let (synthetic_data, synthetic_targets) = generate_synthetic_samples(
                        data,
                        targets,
                        class as f64,
                        samples_needed,
                        k_neighbors,
                        random_seed,
                    )?;

                    // Concatenate with existing data
                    combined_data = ndarray::concatenate(
                        ndarray::Axis(0),
                        &[combined_data.view(), synthetic_data.view()],
                    )
                    .map_err(|_| {
                        DatasetsError::InvalidFormat("Failed to concatenate data".to_string())
                    })?;

                    combined_targets = ndarray::concatenate(
                        ndarray::Axis(0),
                        &[combined_targets.view(), synthetic_targets.view()],
                    )
                    .map_err(|_| {
                        DatasetsError::InvalidFormat("Failed to concatenate targets".to_string())
                    })?;
                }
            }

            Ok((combined_data, combined_targets))
        }
    }
}

/// Balancing strategies for handling imbalanced datasets
#[derive(Debug, Clone, Copy)]
pub enum BalancingStrategy {
    /// Random oversampling - duplicates minority class samples
    RandomOversample,
    /// Random undersampling - removes majority class samples
    RandomUndersample,
    /// SMOTE (Synthetic Minority Oversampling Technique) with specified k_neighbors
    SMOTE {
        /// Number of nearest neighbors to consider for synthetic sample generation
        k_neighbors: usize,
    },
}

/// Performs Min-Max scaling to scale features to a specified range
///
/// Transforms features by scaling each feature to a given range, typically [0, 1].
/// The transformation is: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
///
/// # Arguments
///
/// * `data` - Feature matrix to scale in-place (n_samples, n_features)
/// * `feature_range` - Target range as (min, max) tuple
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use scirs2_datasets::utils::min_max_scale;
///
/// let mut data = Array2::from_shape_vec((3, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap();
/// min_max_scale(&mut data, (0.0, 1.0));
/// // Features are now scaled to [0, 1] range
/// ```
pub fn min_max_scale(data: &mut Array2<f64>, feature_range: (f64, f64)) {
    let (range_min, range_max) = feature_range;
    let range_size = range_max - range_min;

    for j in 0..data.ncols() {
        let mut column = data.column_mut(j);

        // Find min and max values in the column
        let col_min = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let col_max = column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Avoid division by zero
        if (col_max - col_min).abs() > 1e-10 {
            column.mapv_inplace(|x| (x - col_min) / (col_max - col_min) * range_size + range_min);
        } else {
            // If all values are the same, set to the middle of the range
            column.fill(range_min + range_size / 2.0);
        }
    }
}

/// Performs robust scaling using median and interquartile range
///
/// Scales features using statistics that are robust to outliers. Each feature is
/// scaled by: X_scaled = (X - median) / IQR, where IQR is the interquartile range.
///
/// # Arguments
///
/// * `data` - Feature matrix to scale in-place (n_samples, n_features)
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use scirs2_datasets::utils::robust_scale;
///
/// let mut data = Array2::from_shape_vec((5, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 100.0, 500.0]).unwrap();
/// robust_scale(&mut data);
/// // Features are now robustly scaled using median and IQR
/// ```
pub fn robust_scale(data: &mut Array2<f64>) {
    for j in 0..data.ncols() {
        let mut column_values: Vec<f64> = data.column(j).to_vec();
        column_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = column_values.len();
        if n == 0 {
            continue;
        }

        // Calculate median
        let median = if n % 2 == 0 {
            (column_values[n / 2 - 1] + column_values[n / 2]) / 2.0
        } else {
            column_values[n / 2]
        };

        // Calculate Q1 and Q3
        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;
        let q1 = column_values[q1_idx];
        let q3 = column_values[q3_idx];
        let iqr = q3 - q1;

        // Scale the column
        let mut column = data.column_mut(j);
        if iqr > 1e-10 {
            column.mapv_inplace(|x| (x - median) / iqr);
        } else {
            // If IQR is zero, center around median but don't scale
            column.mapv_inplace(|x| x - median);
        }
    }
}

/// Generates polynomial features up to a specified degree
///
/// Creates polynomial combinations of features up to the specified degree.
/// For example, with degree=2 and features [a, b], generates [1, a, b, a², ab, b²].
///
/// # Arguments
///
/// * `data` - Input feature matrix (n_samples, n_features)
/// * `degree` - Maximum polynomial degree (must be >= 1)
/// * `include_bias` - Whether to include the bias column (all ones)
///
/// # Returns
///
/// A new array with polynomial features
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use scirs2_datasets::utils::polynomial_features;
///
/// let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let poly_features = polynomial_features(&data, 2, true).unwrap();
/// // Result includes: [1, x1, x2, x1², x1*x2, x2²]
/// ```
pub fn polynomial_features(
    data: &Array2<f64>,
    degree: usize,
    include_bias: bool,
) -> Result<Array2<f64>> {
    if degree == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Polynomial degree must be at least 1".to_string(),
        ));
    }

    let n_samples = data.nrows();
    let n_features = data.ncols();

    // Calculate number of polynomial features
    let mut n_output_features = 0;
    if include_bias {
        n_output_features += 1;
    }

    // Count features for each degree
    for d in 1..=degree {
        // Number of multivariate polynomials of degree d with n_features variables
        // This uses the formula for combinations with repetition: C(n+k-1, k)
        let mut combinations = 1;
        for i in 0..d {
            combinations = combinations * (n_features + i) / (i + 1);
        }
        n_output_features += combinations;
    }

    let mut output = Array2::zeros((n_samples, n_output_features));
    let mut col_idx = 0;

    // Add bias column if requested
    if include_bias {
        output.column_mut(col_idx).fill(1.0);
    }

    // Generate polynomial features
    for sample_idx in 0..n_samples {
        let sample = data.row(sample_idx);
        col_idx = if include_bias { 1 } else { 0 };

        // Degree 1 features (original features)
        for &feature_val in sample.iter() {
            output[[sample_idx, col_idx]] = feature_val;
            col_idx += 1;
        }

        // Higher degree features
        for deg in 2..=degree {
            generate_polynomial_combinations(
                &sample.to_owned(),
                deg,
                sample_idx,
                &mut output,
                &mut col_idx,
            );
        }
    }

    Ok(output)
}

/// Helper function to generate polynomial combinations recursively
fn generate_polynomial_combinations(
    features: &Array1<f64>,
    degree: usize,
    sample_idx: usize,
    output: &mut Array2<f64>,
    col_idx: &mut usize,
) {
    fn combinations_recursive(
        features: &Array1<f64>,
        degree: usize,
        start_idx: usize,
        current_product: f64,
        sample_idx: usize,
        output: &mut Array2<f64>,
        col_idx: &mut usize,
    ) {
        if degree == 0 {
            output[[sample_idx, *col_idx]] = current_product;
            *col_idx += 1;
            return;
        }

        for i in start_idx..features.len() {
            combinations_recursive(
                features,
                degree - 1,
                i, // Allow repetition by using i instead of i+1
                current_product * features[i],
                sample_idx,
                output,
                col_idx,
            );
        }
    }

    combinations_recursive(features, degree, 0, 1.0, sample_idx, output, col_idx);
}

/// Extracts statistical features from the data
///
/// Computes various statistical measures for each feature, including central tendency,
/// dispersion, and shape statistics.
///
/// # Arguments
///
/// * `data` - Input feature matrix (n_samples, n_features)
///
/// # Returns
///
/// A new array with statistical features: [mean, std, min, max, median, q25, q75, skewness, kurtosis] for each original feature
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use scirs2_datasets::utils::statistical_features;
///
/// let data = Array2::from_shape_vec((5, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0]).unwrap();
/// let stats_features = statistical_features(&data).unwrap();
/// // Result includes 9 statistical measures for each of the 2 original features
/// ```
pub fn statistical_features(data: &Array2<f64>) -> Result<Array2<f64>> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if n_samples == 0 || n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Data cannot be empty for statistical feature extraction".to_string(),
        ));
    }

    // 9 statistical features per original feature
    let n_stat_features = 9;
    let mut stats = Array2::zeros((n_samples, n_features * n_stat_features));

    for sample_idx in 0..n_samples {
        for feature_idx in 0..n_features {
            let feature_values = data.column(feature_idx);

            // Calculate basic statistics
            let mean = feature_values.mean().unwrap_or(0.0);
            let std = feature_values.std(0.0);
            let min_val = feature_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = feature_values
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            // Calculate quantiles
            let mut sorted_values: Vec<f64> = feature_values.to_vec();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let median = calculate_quantile(&sorted_values, 0.5);
            let q25 = calculate_quantile(&sorted_values, 0.25);
            let q75 = calculate_quantile(&sorted_values, 0.75);

            // Calculate skewness and kurtosis
            let skewness = calculate_skewness(&feature_values, mean, std);
            let kurtosis = calculate_kurtosis(&feature_values, mean, std);

            // Store statistical features
            let base_idx = feature_idx * n_stat_features;
            stats[[sample_idx, base_idx]] = mean;
            stats[[sample_idx, base_idx + 1]] = std;
            stats[[sample_idx, base_idx + 2]] = min_val;
            stats[[sample_idx, base_idx + 3]] = max_val;
            stats[[sample_idx, base_idx + 4]] = median;
            stats[[sample_idx, base_idx + 5]] = q25;
            stats[[sample_idx, base_idx + 6]] = q75;
            stats[[sample_idx, base_idx + 7]] = skewness;
            stats[[sample_idx, base_idx + 8]] = kurtosis;
        }
    }

    Ok(stats)
}

/// Calculates a specific quantile from sorted data
fn calculate_quantile(sorted_data: &[f64], quantile: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }

    let n = sorted_data.len();
    let index = quantile * (n - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        sorted_data[lower]
    } else {
        let weight = index - lower as f64;
        sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
    }
}

/// Calculates skewness (third moment)
fn calculate_skewness(data: &ndarray::ArrayView1<f64>, mean: f64, std: f64) -> f64 {
    if std <= 1e-10 {
        return 0.0;
    }

    let n = data.len() as f64;
    let sum_cubed_deviations: f64 = data.iter().map(|&x| ((x - mean) / std).powi(3)).sum();

    sum_cubed_deviations / n
}

/// Calculates kurtosis (fourth moment)
fn calculate_kurtosis(data: &ndarray::ArrayView1<f64>, mean: f64, std: f64) -> f64 {
    if std <= 1e-10 {
        return 0.0;
    }

    let n = data.len() as f64;
    let sum_fourth_deviations: f64 = data.iter().map(|&x| ((x - mean) / std).powi(4)).sum();

    (sum_fourth_deviations / n) - 3.0 // Excess kurtosis (subtract 3 for normal distribution)
}

/// Creates binned (discretized) features from continuous features
///
/// Transforms continuous features into categorical features by binning values
/// into specified ranges. This can be useful for creating non-linear features
/// or reducing the impact of outliers.
///
/// # Arguments
///
/// * `data` - Input feature matrix (n_samples, n_features)
/// * `n_bins` - Number of bins per feature
/// * `strategy` - Binning strategy to use
///
/// # Returns
///
/// A new array with binned features (encoded as bin indices)
///
/// # Example
///
/// ```
/// use ndarray::Array2;
/// use scirs2_datasets::utils::{create_binned_features, BinningStrategy};
///
/// let data = Array2::from_shape_vec((5, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0]).unwrap();
/// let binned = create_binned_features(&data, 3, BinningStrategy::Uniform).unwrap();
/// // Each feature is now discretized into 3 bins (values 0, 1, 2)
/// ```
pub fn create_binned_features(
    data: &Array2<f64>,
    n_bins: usize,
    strategy: BinningStrategy,
) -> Result<Array2<f64>> {
    if n_bins < 2 {
        return Err(DatasetsError::InvalidFormat(
            "Number of bins must be at least 2".to_string(),
        ));
    }

    let n_samples = data.nrows();
    let n_features = data.ncols();
    let mut binned = Array2::zeros((n_samples, n_features));

    for j in 0..n_features {
        let column = data.column(j);
        let bin_edges = calculate_bin_edges(&column, n_bins, &strategy)?;

        for i in 0..n_samples {
            let value = column[i];
            let bin_idx = find_bin_index(value, &bin_edges);
            binned[[i, j]] = bin_idx as f64;
        }
    }

    Ok(binned)
}

/// Binning strategies for discretization
#[derive(Debug, Clone, Copy)]
pub enum BinningStrategy {
    /// Uniform-width bins based on min-max range
    Uniform,
    /// Quantile-based bins (equal frequency)
    Quantile,
}

/// Calculate bin edges based on the specified strategy
fn calculate_bin_edges(
    data: &ndarray::ArrayView1<f64>,
    n_bins: usize,
    strategy: &BinningStrategy,
) -> Result<Vec<f64>> {
    match strategy {
        BinningStrategy::Uniform => {
            let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            if (max_val - min_val).abs() <= 1e-10 {
                return Ok(vec![min_val, min_val + 1e-10]);
            }

            let bin_width = (max_val - min_val) / n_bins as f64;
            let mut edges = Vec::with_capacity(n_bins + 1);

            for i in 0..=n_bins {
                edges.push(min_val + i as f64 * bin_width);
            }

            Ok(edges)
        }
        BinningStrategy::Quantile => {
            let mut sorted_data: Vec<f64> = data.to_vec();
            sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut edges = Vec::with_capacity(n_bins + 1);
            edges.push(sorted_data[0]);

            for i in 1..n_bins {
                let quantile = i as f64 / n_bins as f64;
                let edge = calculate_quantile(&sorted_data, quantile);
                edges.push(edge);
            }

            edges.push(sorted_data[sorted_data.len() - 1]);

            Ok(edges)
        }
    }
}

/// Find the bin index for a given value
fn find_bin_index(value: f64, bin_edges: &[f64]) -> usize {
    for (i, &edge) in bin_edges.iter().enumerate().skip(1) {
        if value <= edge {
            return i - 1;
        }
    }
    bin_edges.len() - 2 // Last bin
}

/// Trait extension for Array2 to calculate mean and standard deviation
#[allow(dead_code)]
trait StatsExt {
    fn mean(&self) -> Option<f64>;
    fn std(&self, ddof: f64) -> f64;
}

impl StatsExt for ndarray::ArrayView1<'_, f64> {
    fn mean(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }

        let sum: f64 = self.sum();
        Some(sum / self.len() as f64)
    }

    fn std(&self, ddof: f64) -> f64 {
        if self.is_empty() {
            return 0.0;
        }

        let n = self.len() as f64;
        let mean = self.mean().unwrap_or(0.0);

        let mut sum_sq = 0.0;
        for &x in self.iter() {
            let diff = x - mean;
            sum_sq += diff * diff;
        }

        let divisor = n - ddof;
        if divisor <= 0.0 {
            return 0.0;
        }

        (sum_sq / divisor).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_dataset_creation() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let target = Array1::from(vec![1.0, 0.0, 1.0]);

        let dataset = Dataset::new(data.clone(), Some(target.clone()));

        assert_eq!(dataset.n_samples(), 3);
        assert_eq!(dataset.n_features(), 2);
        assert_eq!(dataset.data, data);
        assert_eq!(dataset.target.unwrap(), target);
    }

    #[test]
    fn test_dataset_with_metadata() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let dataset = Dataset::new(data, None)
            .with_feature_names(vec!["feature1".to_string(), "feature2".to_string()])
            .with_description("Test dataset".to_string())
            .with_metadata("source", "test");

        assert_eq!(dataset.feature_names.as_ref().unwrap().len(), 2);
        assert_eq!(dataset.description.as_ref().unwrap(), "Test dataset");
        assert_eq!(dataset.metadata.get("source").unwrap(), "test");
    }

    #[test]
    fn test_train_test_split() {
        let data = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let target = Array1::from((0..10).map(|x| x as f64).collect::<Vec<_>>());

        let dataset = Dataset::new(data, Some(target));
        let (train, test) = dataset.train_test_split(0.3, Some(42)).unwrap();

        assert_eq!(train.n_samples(), 7);
        assert_eq!(test.n_samples(), 3);
        assert_eq!(train.n_features(), 2);
        assert_eq!(test.n_features(), 2);
    }

    #[test]
    fn test_train_test_split_invalid_size() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let dataset = Dataset::new(data, None);

        assert!(dataset.train_test_split(0.0, None).is_err());
        assert!(dataset.train_test_split(1.0, None).is_err());
        assert!(dataset.train_test_split(-0.1, None).is_err());
        assert!(dataset.train_test_split(1.1, None).is_err());
    }

    #[test]
    fn test_normalize() {
        let mut data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        normalize(&mut data);

        // Check that each column has approximately zero mean
        for j in 0..data.ncols() {
            let column = data.column(j);
            let mean = column.mean().unwrap();
            assert!(mean.abs() < 1e-10);
        }
    }

    #[test]
    fn test_dataset_serialization() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let target = Array1::from(vec![1.0, 0.0]);

        let dataset = Dataset::new(data, Some(target)).with_description("Test dataset".to_string());

        // Test that serialization doesn't panic
        let serialized = serde_json::to_string(&dataset);
        assert!(serialized.is_ok());

        // Test that deserialization works
        let deserialized = serde_json::from_str::<Dataset>(&serialized.unwrap());
        assert!(deserialized.is_ok());

        let deserialized_dataset = deserialized.unwrap();
        assert_eq!(deserialized_dataset.n_samples(), 2);
        assert_eq!(deserialized_dataset.n_features(), 2);
        assert_eq!(
            deserialized_dataset.description.as_ref().unwrap(),
            "Test dataset"
        );
    }

    #[test]
    fn test_k_fold_split() {
        let folds = k_fold_split(10, 3, false, Some(42)).unwrap();
        assert_eq!(folds.len(), 3);

        // Check that all samples are used exactly once as validation
        let mut all_validation_indices: Vec<usize> = Vec::new();
        for (_, validation_indices) in &folds {
            all_validation_indices.extend(validation_indices);
        }
        all_validation_indices.sort();
        assert_eq!(all_validation_indices, (0..10).collect::<Vec<usize>>());

        // Check that train and validation sets don't overlap
        for (train_indices, validation_indices) in &folds {
            for &val_idx in validation_indices {
                assert!(!train_indices.contains(&val_idx));
            }
        }
    }

    #[test]
    fn test_k_fold_split_invalid_params() {
        assert!(k_fold_split(10, 1, false, None).is_err());
        assert!(k_fold_split(5, 10, false, None).is_err());
    }

    #[test]
    fn test_stratified_k_fold_split() {
        let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let folds = stratified_k_fold_split(&targets, 2, false, Some(42)).unwrap();
        assert_eq!(folds.len(), 2);

        // Check that class proportions are maintained
        for (_, validation_indices) in &folds {
            let val_targets: Vec<f64> = validation_indices.iter().map(|&i| targets[i]).collect();

            let class_0_count = val_targets.iter().filter(|&&x| x == 0.0).count();
            let class_1_count = val_targets.iter().filter(|&&x| x == 1.0).count();

            // Should have roughly equal proportions (allowing for rounding)
            assert!((class_0_count as i32 - class_1_count as i32).abs() <= 1);
        }
    }

    #[test]
    fn test_stratified_k_fold_split_invalid_params() {
        let targets = Array1::from(vec![0.0, 1.0]);
        assert!(stratified_k_fold_split(&targets, 1, false, None).is_err());
        assert!(stratified_k_fold_split(&targets, 5, false, None).is_err());
    }

    #[test]
    fn test_time_series_split() {
        let folds = time_series_split(100, 5, 10, 0).unwrap();
        assert_eq!(folds.len(), 5);

        // Check that training sets are always before validation sets
        for (train_indices, validation_indices) in &folds {
            let max_train = train_indices.iter().max().unwrap_or(&0);
            let min_validation = validation_indices.iter().min().unwrap_or(&0);
            assert!(max_train < min_validation);
        }

        // Check that validation sets have the correct size
        for (_, validation_indices) in &folds {
            assert_eq!(validation_indices.len(), 10);
        }
    }

    #[test]
    fn test_time_series_split_with_gap() {
        let folds = time_series_split(50, 3, 5, 2).unwrap();
        assert_eq!(folds.len(), 3);

        // Check that there's a gap between train and validation sets
        for (train_indices, validation_indices) in &folds {
            let max_train = train_indices.iter().max().unwrap_or(&0);
            let min_validation = validation_indices.iter().min().unwrap_or(&0);
            assert!(min_validation - max_train >= 3); // gap of 2 + 1 for exclusive range
        }
    }

    #[test]
    fn test_time_series_split_invalid_params() {
        assert!(time_series_split(10, 0, 5, 0).is_err());
        assert!(time_series_split(10, 5, 0, 0).is_err());
        assert!(time_series_split(5, 3, 5, 0).is_err()); // Not enough samples
    }

    #[test]
    fn test_random_sample_without_replacement() {
        let indices = random_sample(10, 5, false, Some(42)).unwrap();
        assert_eq!(indices.len(), 5);

        // Check all indices are unique
        let mut sorted_indices = indices.clone();
        sorted_indices.sort();
        sorted_indices.dedup();
        assert_eq!(sorted_indices.len(), 5);

        // Check all indices are valid
        for &idx in &indices {
            assert!(idx < 10);
        }
    }

    #[test]
    fn test_random_sample_with_replacement() {
        let indices = random_sample(5, 10, true, Some(42)).unwrap();
        assert_eq!(indices.len(), 10);

        // Check all indices are valid (may have duplicates)
        for &idx in &indices {
            assert!(idx < 5);
        }
    }

    #[test]
    fn test_random_sample_invalid_params() {
        // Zero n_samples
        assert!(random_sample(0, 5, false, None).is_err());

        // Zero sample_size
        assert!(random_sample(10, 0, false, None).is_err());

        // Sample size > n_samples without replacement
        assert!(random_sample(5, 10, false, None).is_err());

        // This should work with replacement
        assert!(random_sample(5, 10, true, None).is_ok());
    }

    #[test]
    fn test_stratified_sample() {
        let targets = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0]);
        let indices = stratified_sample(&targets, 6, Some(42)).unwrap();
        assert_eq!(indices.len(), 6);

        // Check that we maintain class proportions
        let sampled_targets: Vec<f64> = indices.iter().map(|&i| targets[i]).collect();
        let class_0_count = sampled_targets.iter().filter(|&&x| x == 0.0).count();
        let class_1_count = sampled_targets.iter().filter(|&&x| x == 1.0).count();
        let class_2_count = sampled_targets.iter().filter(|&&x| x == 2.0).count();

        // Should have 2 from each class
        assert_eq!(class_0_count, 2);
        assert_eq!(class_1_count, 2);
        assert_eq!(class_2_count, 2);
    }

    #[test]
    fn test_stratified_sample_invalid_params() {
        let targets = Array1::from(vec![0.0, 1.0, 2.0]);

        // Empty targets
        let empty_targets = Array1::from(vec![]);
        assert!(stratified_sample(&empty_targets, 1, None).is_err());

        // Zero sample size
        assert!(stratified_sample(&targets, 0, None).is_err());

        // Sample size > number of samples
        assert!(stratified_sample(&targets, 5, None).is_err());

        // Sample size requiring more samples from a class than available
        let unbalanced_targets = Array1::from(vec![0.0, 1.0, 1.0, 1.0]);
        assert!(stratified_sample(&unbalanced_targets, 4, None).is_err()); // Would need 1 from class 0, but only has 1
    }

    #[test]
    fn test_importance_sample() {
        // Create weights that heavily favor the last few indices
        let weights = Array1::from(vec![0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0]);
        let indices = importance_sample(&weights, 3, false, Some(42)).unwrap();
        assert_eq!(indices.len(), 3);

        // Most selected indices should be from the high-weight region (4, 5, 6)
        let high_weight_count = indices.iter().filter(|&&idx| idx >= 4).count();
        assert!(high_weight_count >= 2); // Should favor high-weight indices
    }

    #[test]
    fn test_importance_sample_with_replacement() {
        let weights = Array1::from(vec![0.1, 0.1, 1.0, 1.0]);
        let indices = importance_sample(&weights, 6, true, Some(42)).unwrap();
        assert_eq!(indices.len(), 6);

        // Check that all indices are valid
        for &idx in &indices {
            assert!(idx < 4);
        }
    }

    #[test]
    fn test_importance_sample_uniform_weights() {
        // With uniform weights, should behave like random sampling
        let weights = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        let indices = importance_sample(&weights, 3, false, Some(42)).unwrap();
        assert_eq!(indices.len(), 3);

        // Check all indices are unique and valid
        let mut sorted_indices = indices.clone();
        sorted_indices.sort();
        sorted_indices.dedup();
        assert_eq!(sorted_indices.len(), 3);
    }

    #[test]
    fn test_importance_sample_invalid_params() {
        let weights = Array1::from(vec![0.1, 0.2, 0.3]);

        // Empty weights
        let empty_weights = Array1::from(vec![]);
        assert!(importance_sample(&empty_weights, 1, false, None).is_err());

        // Zero sample size
        assert!(importance_sample(&weights, 0, false, None).is_err());

        // Sample size > weights.len() without replacement
        assert!(importance_sample(&weights, 5, false, None).is_err());

        // Negative weights
        let negative_weights = Array1::from(vec![0.1, -0.1, 0.3]);
        assert!(importance_sample(&negative_weights, 2, false, None).is_err());

        // All zero weights
        let zero_weights = Array1::from(vec![0.0, 0.0, 0.0]);
        assert!(importance_sample(&zero_weights, 1, false, None).is_err());
    }

    #[test]
    fn test_random_oversample() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]); // Imbalanced: 2 vs 4

        let (balanced_data, balanced_targets) =
            random_oversample(&data, &targets, Some(42)).unwrap();

        // Check that we now have equal number of each class
        let class_0_count = balanced_targets.iter().filter(|&&x| x == 0.0).count();
        let class_1_count = balanced_targets.iter().filter(|&&x| x == 1.0).count();
        assert_eq!(class_0_count, 4); // Should be oversampled to match majority class
        assert_eq!(class_1_count, 4);

        // Check that total samples increased
        assert_eq!(balanced_data.nrows(), 8);
        assert_eq!(balanced_targets.len(), 8);

        // Check that data dimensions are preserved
        assert_eq!(balanced_data.ncols(), 2);
    }

    #[test]
    fn test_random_oversample_invalid_params() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let targets = Array1::from(vec![0.0, 1.0]);

        // Mismatched data and targets
        assert!(random_oversample(&data, &targets, None).is_err());

        // Empty data
        let empty_data = Array2::zeros((0, 2));
        let empty_targets = Array1::from(vec![]);
        assert!(random_oversample(&empty_data, &empty_targets, None).is_err());
    }

    #[test]
    fn test_random_undersample() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]); // Imbalanced: 2 vs 4

        let (balanced_data, balanced_targets) =
            random_undersample(&data, &targets, Some(42)).unwrap();

        // Check that we now have equal number of each class (minimum)
        let class_0_count = balanced_targets.iter().filter(|&&x| x == 0.0).count();
        let class_1_count = balanced_targets.iter().filter(|&&x| x == 1.0).count();
        assert_eq!(class_0_count, 2); // Should match minority class
        assert_eq!(class_1_count, 2); // Should be undersampled to match minority class

        // Check that total samples decreased
        assert_eq!(balanced_data.nrows(), 4);
        assert_eq!(balanced_targets.len(), 4);

        // Check that data dimensions are preserved
        assert_eq!(balanced_data.ncols(), 2);
    }

    #[test]
    fn test_random_undersample_invalid_params() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let targets = Array1::from(vec![0.0, 1.0]);

        // Mismatched data and targets
        assert!(random_undersample(&data, &targets, None).is_err());

        // Empty data
        let empty_data = Array2::zeros((0, 2));
        let empty_targets = Array1::from(vec![]);
        assert!(random_undersample(&empty_data, &empty_targets, None).is_err());
    }

    #[test]
    fn test_generate_synthetic_samples() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 1.5, 1.5, 2.5, 2.5]).unwrap();
        let targets = Array1::from(vec![0.0, 0.0, 0.0, 1.0]);

        let (synthetic_data, synthetic_targets) =
            generate_synthetic_samples(&data, &targets, 0.0, 2, 2, Some(42)).unwrap();

        // Check that we generated the correct number of synthetic samples
        assert_eq!(synthetic_data.nrows(), 2);
        assert_eq!(synthetic_targets.len(), 2);

        // Check that all synthetic targets are the correct class
        for &target in synthetic_targets.iter() {
            assert_eq!(target, 0.0);
        }

        // Check that data dimensions are preserved
        assert_eq!(synthetic_data.ncols(), 2);

        // Check that synthetic samples are interpolations (should be within reasonable bounds)
        for i in 0..synthetic_data.nrows() {
            for j in 0..synthetic_data.ncols() {
                let value = synthetic_data[[i, j]];
                assert!(value >= 0.5 && value <= 2.5); // Should be within range of class 0 samples
            }
        }
    }

    #[test]
    fn test_generate_synthetic_samples_invalid_params() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 1.5, 1.5, 2.5, 2.5]).unwrap();
        let targets = Array1::from(vec![0.0, 0.0, 0.0, 1.0]);

        // Mismatched data and targets
        let bad_targets = Array1::from(vec![0.0, 1.0]);
        assert!(generate_synthetic_samples(&data, &bad_targets, 0.0, 2, 2, None).is_err());

        // Zero synthetic samples
        assert!(generate_synthetic_samples(&data, &targets, 0.0, 0, 2, None).is_err());

        // Zero neighbors
        assert!(generate_synthetic_samples(&data, &targets, 0.0, 2, 0, None).is_err());

        // Too few samples of target class (only 1 sample of class 1.0)
        assert!(generate_synthetic_samples(&data, &targets, 1.0, 2, 2, None).is_err());

        // k_neighbors >= number of samples in class
        assert!(generate_synthetic_samples(&data, &targets, 0.0, 2, 3, None).is_err());
    }

    #[test]
    fn test_create_balanced_dataset_random_oversample() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let (balanced_data, balanced_targets) = create_balanced_dataset(
            &data,
            &targets,
            BalancingStrategy::RandomOversample,
            Some(42),
        )
        .unwrap();

        // Check that classes are balanced
        let class_0_count = balanced_targets.iter().filter(|&&x| x == 0.0).count();
        let class_1_count = balanced_targets.iter().filter(|&&x| x == 1.0).count();
        assert_eq!(class_0_count, class_1_count);
        assert_eq!(balanced_data.nrows(), balanced_targets.len());
    }

    #[test]
    fn test_create_balanced_dataset_random_undersample() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let (balanced_data, balanced_targets) = create_balanced_dataset(
            &data,
            &targets,
            BalancingStrategy::RandomUndersample,
            Some(42),
        )
        .unwrap();

        // Check that classes are balanced
        let class_0_count = balanced_targets.iter().filter(|&&x| x == 0.0).count();
        let class_1_count = balanced_targets.iter().filter(|&&x| x == 1.0).count();
        assert_eq!(class_0_count, class_1_count);
        assert_eq!(balanced_data.nrows(), balanced_targets.len());
    }

    #[test]
    fn test_create_balanced_dataset_smote() {
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0,
            ],
        )
        .unwrap();
        let targets = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]); // Already balanced for easier testing

        let (balanced_data, balanced_targets) = create_balanced_dataset(
            &data,
            &targets,
            BalancingStrategy::SMOTE { k_neighbors: 2 },
            Some(42),
        )
        .unwrap();

        // Check that classes remain balanced
        let class_0_count = balanced_targets.iter().filter(|&&x| x == 0.0).count();
        let class_1_count = balanced_targets.iter().filter(|&&x| x == 1.0).count();
        assert_eq!(class_0_count, class_1_count);
        assert_eq!(balanced_data.nrows(), balanced_targets.len());
    }

    #[test]
    fn test_balancing_strategy_with_multiple_classes() {
        // Test with 3 classes of different sizes
        let data = Array2::from_shape_vec((9, 2), (0..18).map(|x| x as f64).collect()).unwrap();
        let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        // Class distribution: 0 (2 samples), 1 (4 samples), 2 (3 samples)

        // Test oversampling
        let (_over_data, over_targets) = create_balanced_dataset(
            &data,
            &targets,
            BalancingStrategy::RandomOversample,
            Some(42),
        )
        .unwrap();

        let over_class_0_count = over_targets.iter().filter(|&&x| x == 0.0).count();
        let over_class_1_count = over_targets.iter().filter(|&&x| x == 1.0).count();
        let over_class_2_count = over_targets.iter().filter(|&&x| x == 2.0).count();

        // All classes should have 4 samples (majority class size)
        assert_eq!(over_class_0_count, 4);
        assert_eq!(over_class_1_count, 4);
        assert_eq!(over_class_2_count, 4);

        // Test undersampling
        let (_under_data, under_targets) = create_balanced_dataset(
            &data,
            &targets,
            BalancingStrategy::RandomUndersample,
            Some(42),
        )
        .unwrap();

        let under_class_0_count = under_targets.iter().filter(|&&x| x == 0.0).count();
        let under_class_1_count = under_targets.iter().filter(|&&x| x == 1.0).count();
        let under_class_2_count = under_targets.iter().filter(|&&x| x == 2.0).count();

        // All classes should have 2 samples (minority class size)
        assert_eq!(under_class_0_count, 2);
        assert_eq!(under_class_1_count, 2);
        assert_eq!(under_class_2_count, 2);
    }

    #[test]
    fn test_min_max_scale() {
        let mut data =
            Array2::from_shape_vec((3, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]).unwrap();
        min_max_scale(&mut data, (0.0, 1.0));

        // Check that values are scaled to [0, 1]
        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                let value = data[[i, j]];
                assert!(value >= 0.0 && value <= 1.0);
            }
        }

        // Check specific scaling: first column should be [0, 0.5, 1]
        assert!((data[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((data[[1, 0]] - 0.5).abs() < 1e-10);
        assert!((data[[2, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_max_scale_custom_range() {
        let mut data = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        min_max_scale(&mut data, (-1.0, 1.0));

        // Check that values are scaled to [-1, 1]
        assert!((data[[0, 0]] - (-1.0)).abs() < 1e-10);
        assert!((data[[1, 0]] - 0.0).abs() < 1e-10);
        assert!((data[[2, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_max_scale_constant_values() {
        let mut data = Array2::from_shape_vec((3, 1), vec![5.0, 5.0, 5.0]).unwrap();
        min_max_scale(&mut data, (0.0, 1.0));

        // All values should be 0.5 (middle of range) when all values are the same
        for i in 0..data.nrows() {
            assert!((data[[i, 0]] - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_robust_scale() {
        let mut data = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 100.0, 500.0],
        )
        .unwrap(); // Last row has outliers

        robust_scale(&mut data);

        // Check that the scaling was applied (data should have different values than original)
        // and that extreme outliers have limited influence
        let col1_values: Vec<f64> = data.column(0).to_vec();
        let col2_values: Vec<f64> = data.column(1).to_vec();

        // Verify that the data has been transformed (not all values are the same)
        let col1_range = col1_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - col1_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let col2_range = col2_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - col2_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        // After robust scaling, the range should be reasonable (not infinite)
        assert!(col1_range.is_finite());
        assert!(col2_range.is_finite());
        assert!(col1_range > 0.0); // Some variation should remain
        assert!(col2_range > 0.0); // Some variation should remain
    }

    #[test]
    fn test_polynomial_features_degree_2() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let poly = polynomial_features(&data, 2, true).unwrap();

        // Should have: [bias, x1, x2, x1², x1*x2, x2²] = 6 features
        assert_eq!(poly.ncols(), 6);
        assert_eq!(poly.nrows(), 2);

        // Check first sample: [1, 1, 2, 1, 2, 4]
        assert!((poly[[0, 0]] - 1.0).abs() < 1e-10); // bias
        assert!((poly[[0, 1]] - 1.0).abs() < 1e-10); // x1
        assert!((poly[[0, 2]] - 2.0).abs() < 1e-10); // x2
        assert!((poly[[0, 3]] - 1.0).abs() < 1e-10); // x1²
        assert!((poly[[0, 4]] - 2.0).abs() < 1e-10); // x1*x2
        assert!((poly[[0, 5]] - 4.0).abs() < 1e-10); // x2²
    }

    #[test]
    fn test_polynomial_features_no_bias() {
        let data = Array2::from_shape_vec((1, 2), vec![2.0, 3.0]).unwrap();
        let poly = polynomial_features(&data, 2, false).unwrap();

        // Should have: [x1, x2, x1², x1*x2, x2²] = 5 features (no bias)
        assert_eq!(poly.ncols(), 5);

        // Check values: [2, 3, 4, 6, 9]
        assert!((poly[[0, 0]] - 2.0).abs() < 1e-10); // x1
        assert!((poly[[0, 1]] - 3.0).abs() < 1e-10); // x2
        assert!((poly[[0, 2]] - 4.0).abs() < 1e-10); // x1²
        assert!((poly[[0, 3]] - 6.0).abs() < 1e-10); // x1*x2
        assert!((poly[[0, 4]] - 9.0).abs() < 1e-10); // x2²
    }

    #[test]
    fn test_polynomial_features_invalid_degree() {
        let data = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        assert!(polynomial_features(&data, 0, true).is_err());
    }

    #[test]
    fn test_statistical_features() {
        let data = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let stats = statistical_features(&data).unwrap();

        // Should have 9 statistical features for 1 original feature
        assert_eq!(stats.ncols(), 9);
        assert_eq!(stats.nrows(), 5);

        // All samples should have the same statistical features (global statistics)
        for i in 0..stats.nrows() {
            assert!((stats[[i, 0]] - 3.0).abs() < 1e-10); // mean
            assert!((stats[[i, 2]] - 1.0).abs() < 1e-10); // min
            assert!((stats[[i, 3]] - 5.0).abs() < 1e-10); // max
            assert!((stats[[i, 4]] - 3.0).abs() < 1e-10); // median
        }
    }

    #[test]
    fn test_statistical_features_empty_data() {
        let data = Array2::zeros((0, 1));
        assert!(statistical_features(&data).is_err());
    }

    #[test]
    fn test_create_binned_features_uniform() {
        let data = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let binned = create_binned_features(&data, 3, BinningStrategy::Uniform).unwrap();

        assert_eq!(binned.nrows(), 5);
        assert_eq!(binned.ncols(), 1);

        // Check that all values are valid bin indices (0, 1, or 2)
        for i in 0..binned.nrows() {
            let bin_val = binned[[i, 0]] as usize;
            assert!(bin_val < 3);
        }
    }

    #[test]
    fn test_create_binned_features_quantile() {
        let data = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let binned = create_binned_features(&data, 3, BinningStrategy::Quantile).unwrap();

        assert_eq!(binned.nrows(), 6);
        assert_eq!(binned.ncols(), 1);

        // With quantile binning, each bin should have roughly equal number of samples
        let mut bin_counts = vec![0; 3];
        for i in 0..binned.nrows() {
            let bin_val = binned[[i, 0]] as usize;
            bin_counts[bin_val] += 1;
        }

        // Each bin should have 2 samples (6 samples / 3 bins)
        for &count in &bin_counts {
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn test_create_binned_features_invalid_bins() {
        let data = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(create_binned_features(&data, 1, BinningStrategy::Uniform).is_err());
        assert!(create_binned_features(&data, 0, BinningStrategy::Uniform).is_err());
    }

    #[test]
    fn test_feature_extraction_pipeline() {
        // Test a complete feature extraction pipeline
        let mut data =
            Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0])
                .unwrap();

        // Step 1: Min-max scaling
        min_max_scale(&mut data, (0.0, 1.0));

        // Step 2: Generate polynomial features
        let poly_data = polynomial_features(&data, 2, false).unwrap();

        // Step 3: Create binned features
        let binned_data = create_binned_features(&poly_data, 2, BinningStrategy::Uniform).unwrap();

        // Verify pipeline produces expected shapes
        assert_eq!(data.ncols(), 2); // Original scaled features
        assert_eq!(poly_data.ncols(), 5); // [x1, x2, x1², x1*x2, x2²]
        assert_eq!(binned_data.ncols(), 5); // Same number of features, but binned
        assert_eq!(binned_data.nrows(), 4); // Same number of samples
    }

    #[test]
    fn test_robust_vs_standard_scaling() {
        // Create data with outliers
        let mut data_robust = Array2::from_shape_vec(
            (5, 1),
            vec![1.0, 2.0, 3.0, 4.0, 100.0], // 100.0 is an outlier
        )
        .unwrap();
        let mut data_standard = data_robust.clone();

        // Apply different scaling methods
        robust_scale(&mut data_robust);
        normalize(&mut data_standard); // Standard z-score normalization

        // Both scaling methods should produce finite, transformed data
        let robust_range = data_robust.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - data_robust.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let standard_range = data_standard
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - data_standard.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        // Both scaling methods should produce finite ranges
        assert!(robust_range.is_finite());
        assert!(standard_range.is_finite());
        assert!(robust_range > 0.0);
        assert!(standard_range > 0.0);

        // The scaled data should be different from the original
        assert!(data_robust[[0, 0]] != 1.0); // First value should be transformed
        assert!(data_standard[[0, 0]] != 1.0); // First value should be transformed
    }
}
