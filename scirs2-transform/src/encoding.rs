//! Categorical data encoding utilities
//!
//! This module provides methods for encoding categorical data into numerical
//! formats suitable for machine learning algorithms.

use ndarray::{Array2, ArrayBase, Data, Ix2};
use num_traits::{Float, NumCast};
use std::collections::HashMap;

use crate::error::{Result, TransformError};

/// OneHotEncoder for converting categorical features to binary features
///
/// This transformer converts categorical features into a one-hot encoded representation,
/// where each category is represented by a binary feature.
pub struct OneHotEncoder {
    /// Categories for each feature (learned during fit)
    categories_: Option<Vec<Vec<u64>>>,
    /// Whether to drop one category per feature to avoid collinearity
    drop: Option<String>,
    /// Whether to handle unknown categories
    handle_unknown: String,
    /// Sparse output (not implemented yet)
    #[allow(dead_code)]
    sparse: bool,
}

impl OneHotEncoder {
    /// Creates a new OneHotEncoder
    ///
    /// # Arguments
    /// * `drop` - Strategy for dropping categories ('first', 'if_binary', or None)
    /// * `handle_unknown` - How to handle unknown categories ('error' or 'ignore')
    /// * `sparse` - Whether to return sparse arrays (not implemented)
    ///
    /// # Returns
    /// * A new OneHotEncoder instance
    pub fn new(drop: Option<String>, handle_unknown: &str, sparse: bool) -> Result<Self> {
        if let Some(ref drop_strategy) = drop {
            if drop_strategy != "first" && drop_strategy != "if_binary" {
                return Err(TransformError::InvalidInput(
                    "drop must be 'first', 'if_binary', or None".to_string(),
                ));
            }
        }

        if handle_unknown != "error" && handle_unknown != "ignore" {
            return Err(TransformError::InvalidInput(
                "handle_unknown must be 'error' or 'ignore'".to_string(),
            ));
        }

        if sparse {
            return Err(TransformError::InvalidInput(
                "Sparse output is not yet implemented".to_string(),
            ));
        }

        Ok(OneHotEncoder {
            categories_: None,
            drop,
            handle_unknown: handle_unknown.to_string(),
            sparse,
        })
    }

    /// Creates a OneHotEncoder with default settings
    pub fn default() -> Self {
        Self::new(None, "error", false).unwrap()
    }

    /// Fits the OneHotEncoder to the input data
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        let mut categories = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Collect unique values for this feature
            let mut unique_values: Vec<u64> = x_u64.column(j).to_vec();
            unique_values.sort_unstable();
            unique_values.dedup();

            categories.push(unique_values);
        }

        self.categories_ = Some(categories);
        Ok(())
    }

    /// Transforms the input data using the fitted OneHotEncoder
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The one-hot encoded data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if self.categories_.is_none() {
            return Err(TransformError::TransformationError(
                "OneHotEncoder has not been fitted".to_string(),
            ));
        }

        let categories = self.categories_.as_ref().unwrap();

        if n_features != categories.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but OneHotEncoder was fitted with {} features",
                n_features,
                categories.len()
            )));
        }

        // Calculate total number of output features
        let mut total_features = 0;
        for (j, feature_categories) in categories.iter().enumerate() {
            let n_cats = feature_categories.len();

            // Apply drop strategy
            let n_output_cats = match &self.drop {
                Some(strategy) if strategy == "first" => n_cats.saturating_sub(1),
                Some(strategy) if strategy == "if_binary" && n_cats == 2 => 1,
                _ => n_cats,
            };

            if n_output_cats == 0 {
                return Err(TransformError::InvalidInput(format!(
                    "Feature {} has only one category after dropping",
                    j
                )));
            }

            total_features += n_output_cats;
        }

        let mut transformed = Array2::zeros((n_samples, total_features));

        // Create mappings from category values to column indices
        let mut category_mappings = Vec::new();
        let mut current_col = 0;

        for (j, feature_categories) in categories.iter().enumerate() {
            let mut mapping = HashMap::new();
            let n_cats = feature_categories.len();

            // Determine how many categories to keep
            let (start_idx, n_output_cats) = match &self.drop {
                Some(strategy) if strategy == "first" => (1, n_cats.saturating_sub(1)),
                Some(strategy) if strategy == "if_binary" && n_cats == 2 => (0, 1),
                _ => (0, n_cats),
            };

            for (cat_idx, &category) in feature_categories.iter().enumerate() {
                if cat_idx >= start_idx && cat_idx < start_idx + n_output_cats {
                    mapping.insert(category, current_col + cat_idx - start_idx);
                }
            }

            category_mappings.push(mapping);
            current_col += n_output_cats;
        }

        // Fill the transformed array
        for i in 0..n_samples {
            for j in 0..n_features {
                let value = x_u64[[i, j]];

                if let Some(&col_idx) = category_mappings[j].get(&value) {
                    transformed[[i, col_idx]] = 1.0;
                } else {
                    // Check if this is a dropped category (which should be represented as all zeros)
                    let feature_categories = &categories[j];
                    let is_dropped_category = match &self.drop {
                        Some(strategy) if strategy == "first" => {
                            // If it's the first category in the sorted list, it was dropped
                            !feature_categories.is_empty() && value == feature_categories[0]
                        }
                        Some(strategy) if strategy == "if_binary" && feature_categories.len() == 2 => {
                            // If it's the second category (index 1) in a binary feature, it was dropped
                            feature_categories.len() == 2 && value == feature_categories[1]
                        }
                        _ => false,
                    };
                    
                    if !is_dropped_category && self.handle_unknown == "error" {
                        return Err(TransformError::InvalidInput(format!(
                            "Found unknown category {} in feature {}",
                            value, j
                        )));
                    }
                    // If it's a dropped category or handle_unknown == "ignore", we just leave it as 0
                }
            }
        }

        Ok(transformed)
    }

    /// Fits the OneHotEncoder to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The one-hot encoded data
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the categories for each feature
    ///
    /// # Returns
    /// * `Option<&Vec<Vec<u64>>>` - The categories for each feature
    pub fn categories(&self) -> Option<&Vec<Vec<u64>>> {
        self.categories_.as_ref()
    }

    /// Gets the feature names for the transformed output
    ///
    /// # Arguments
    /// * `input_features` - Names of input features
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - Names of output features
    pub fn get_feature_names(&self, input_features: Option<&[String]>) -> Result<Vec<String>> {
        if self.categories_.is_none() {
            return Err(TransformError::TransformationError(
                "OneHotEncoder has not been fitted".to_string(),
            ));
        }

        let categories = self.categories_.as_ref().unwrap();
        let mut feature_names = Vec::new();

        for (j, feature_categories) in categories.iter().enumerate() {
            let feature_name = if let Some(names) = input_features {
                if j < names.len() {
                    names[j].clone()
                } else {
                    format!("x{}", j)
                }
            } else {
                format!("x{}", j)
            };

            let n_cats = feature_categories.len();

            // Determine which categories to include based on drop strategy
            let (start_idx, n_output_cats) = match &self.drop {
                Some(strategy) if strategy == "first" => (1, n_cats.saturating_sub(1)),
                Some(strategy) if strategy == "if_binary" && n_cats == 2 => (0, 1),
                _ => (0, n_cats),
            };

            for cat_idx in start_idx..start_idx + n_output_cats {
                let category = feature_categories[cat_idx];
                feature_names.push(format!("{}_cat_{}", feature_name, category));
            }
        }

        Ok(feature_names)
    }
}

/// OrdinalEncoder for converting categorical features to ordinal integers
///
/// This transformer converts categorical features into ordinal integers,
/// where each category is assigned a unique integer.
pub struct OrdinalEncoder {
    /// Categories for each feature (learned during fit)
    categories_: Option<Vec<Vec<u64>>>,
    /// How to handle unknown categories
    handle_unknown: String,
    /// Value to use for unknown categories
    unknown_value: Option<f64>,
}

impl OrdinalEncoder {
    /// Creates a new OrdinalEncoder
    ///
    /// # Arguments
    /// * `handle_unknown` - How to handle unknown categories ('error' or 'use_encoded_value')
    /// * `unknown_value` - Value to use for unknown categories (when handle_unknown='use_encoded_value')
    ///
    /// # Returns
    /// * A new OrdinalEncoder instance
    pub fn new(handle_unknown: &str, unknown_value: Option<f64>) -> Result<Self> {
        if handle_unknown != "error" && handle_unknown != "use_encoded_value" {
            return Err(TransformError::InvalidInput(
                "handle_unknown must be 'error' or 'use_encoded_value'".to_string(),
            ));
        }

        if handle_unknown == "use_encoded_value" && unknown_value.is_none() {
            return Err(TransformError::InvalidInput(
                "unknown_value must be specified when handle_unknown='use_encoded_value'"
                    .to_string(),
            ));
        }

        Ok(OrdinalEncoder {
            categories_: None,
            handle_unknown: handle_unknown.to_string(),
            unknown_value,
        })
    }

    /// Creates an OrdinalEncoder with default settings
    pub fn default() -> Self {
        Self::new("error", None).unwrap()
    }

    /// Fits the OrdinalEncoder to the input data
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        let mut categories = Vec::with_capacity(n_features);

        for j in 0..n_features {
            // Collect unique values for this feature
            let mut unique_values: Vec<u64> = x_u64.column(j).to_vec();
            unique_values.sort_unstable();
            unique_values.dedup();

            categories.push(unique_values);
        }

        self.categories_ = Some(categories);
        Ok(())
    }

    /// Transforms the input data using the fitted OrdinalEncoder
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The ordinally encoded data
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_u64 = x.mapv(|x| {
            let val_f64 = num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0);
            val_f64 as u64
        });

        let n_samples = x_u64.shape()[0];
        let n_features = x_u64.shape()[1];

        if self.categories_.is_none() {
            return Err(TransformError::TransformationError(
                "OrdinalEncoder has not been fitted".to_string(),
            ));
        }

        let categories = self.categories_.as_ref().unwrap();

        if n_features != categories.len() {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, but OrdinalEncoder was fitted with {} features",
                n_features,
                categories.len()
            )));
        }

        let mut transformed = Array2::zeros((n_samples, n_features));

        // Create mappings from category values to ordinal values
        let mut category_mappings = Vec::new();
        for feature_categories in categories {
            let mut mapping = HashMap::new();
            for (ordinal, &category) in feature_categories.iter().enumerate() {
                mapping.insert(category, ordinal as f64);
            }
            category_mappings.push(mapping);
        }

        // Fill the transformed array
        for i in 0..n_samples {
            for j in 0..n_features {
                let value = x_u64[[i, j]];

                if let Some(&ordinal_value) = category_mappings[j].get(&value) {
                    transformed[[i, j]] = ordinal_value;
                } else if self.handle_unknown == "error" {
                    return Err(TransformError::InvalidInput(format!(
                        "Found unknown category {} in feature {}",
                        value, j
                    )));
                } else {
                    // handle_unknown == "use_encoded_value"
                    transformed[[i, j]] = self.unknown_value.unwrap();
                }
            }
        }

        Ok(transformed)
    }

    /// Fits the OrdinalEncoder to the input data and transforms it
    ///
    /// # Arguments
    /// * `x` - The input categorical data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The ordinally encoded data
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the categories for each feature
    ///
    /// # Returns
    /// * `Option<&Vec<Vec<u64>>>` - The categories for each feature
    pub fn categories(&self) -> Option<&Vec<Vec<u64>>> {
        self.categories_.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    #[test]
    fn test_one_hot_encoder_basic() {
        // Create test data with categorical values
        let data = Array::from_shape_vec(
            (4, 2),
            vec![
                0.0, 1.0, // categories: [0, 1, 2] and [1, 2, 3]
                1.0, 2.0, 2.0, 3.0, 0.0, 1.0,
            ],
        )
        .unwrap();

        let mut encoder = OneHotEncoder::default();
        let encoded = encoder.fit_transform(&data).unwrap();

        // Should have 3 + 3 = 6 output features
        assert_eq!(encoded.shape(), &[4, 6]);

        // Check first row: category 0 in feature 0, category 1 in feature 1
        assert_abs_diff_eq!(encoded[[0, 0]], 1.0, epsilon = 1e-10); // cat 0, feature 0
        assert_abs_diff_eq!(encoded[[0, 1]], 0.0, epsilon = 1e-10); // cat 1, feature 0
        assert_abs_diff_eq!(encoded[[0, 2]], 0.0, epsilon = 1e-10); // cat 2, feature 0
        assert_abs_diff_eq!(encoded[[0, 3]], 1.0, epsilon = 1e-10); // cat 1, feature 1
        assert_abs_diff_eq!(encoded[[0, 4]], 0.0, epsilon = 1e-10); // cat 2, feature 1
        assert_abs_diff_eq!(encoded[[0, 5]], 0.0, epsilon = 1e-10); // cat 3, feature 1

        // Check second row: category 1 in feature 0, category 2 in feature 1
        assert_abs_diff_eq!(encoded[[1, 0]], 0.0, epsilon = 1e-10); // cat 0, feature 0
        assert_abs_diff_eq!(encoded[[1, 1]], 1.0, epsilon = 1e-10); // cat 1, feature 0
        assert_abs_diff_eq!(encoded[[1, 2]], 0.0, epsilon = 1e-10); // cat 2, feature 0
        assert_abs_diff_eq!(encoded[[1, 3]], 0.0, epsilon = 1e-10); // cat 1, feature 1
        assert_abs_diff_eq!(encoded[[1, 4]], 1.0, epsilon = 1e-10); // cat 2, feature 1
        assert_abs_diff_eq!(encoded[[1, 5]], 0.0, epsilon = 1e-10); // cat 3, feature 1
    }

    #[test]
    fn test_one_hot_encoder_drop_first() {
        // Create test data with categorical values
        let data = Array::from_shape_vec((3, 2), vec![0.0, 1.0, 1.0, 2.0, 2.0, 1.0]).unwrap();

        let mut encoder = OneHotEncoder::new(Some("first".to_string()), "error", false).unwrap();
        let encoded = encoder.fit_transform(&data).unwrap();

        // Should have (3-1) + (2-1) = 3 output features (dropped first category of each)
        assert_eq!(encoded.shape(), &[3, 3]);

        // Categories: feature 0: [0, 1, 2] -> keep [1, 2]
        //            feature 1: [1, 2] -> keep [2]

        // First row: category 0 (dropped), category 1 (dropped)
        assert_abs_diff_eq!(encoded[[0, 0]], 0.0, epsilon = 1e-10); // cat 1, feature 0
        assert_abs_diff_eq!(encoded[[0, 1]], 0.0, epsilon = 1e-10); // cat 2, feature 0
        assert_abs_diff_eq!(encoded[[0, 2]], 0.0, epsilon = 1e-10); // cat 2, feature 1

        // Second row: category 1, category 2
        assert_abs_diff_eq!(encoded[[1, 0]], 1.0, epsilon = 1e-10); // cat 1, feature 0
        assert_abs_diff_eq!(encoded[[1, 1]], 0.0, epsilon = 1e-10); // cat 2, feature 0
        assert_abs_diff_eq!(encoded[[1, 2]], 1.0, epsilon = 1e-10); // cat 2, feature 1
    }

    #[test]
    fn test_ordinal_encoder() {
        // Create test data with categorical values
        let data = Array::from_shape_vec(
            (4, 2),
            vec![
                2.0, 10.0, // categories will be mapped to ordinals
                1.0, 20.0, 3.0, 10.0, 2.0, 30.0,
            ],
        )
        .unwrap();

        let mut encoder = OrdinalEncoder::default();
        let encoded = encoder.fit_transform(&data).unwrap();

        // Should preserve shape
        assert_eq!(encoded.shape(), &[4, 2]);

        // Categories for feature 0: [1, 2, 3] -> ordinals [0, 1, 2]
        // Categories for feature 1: [10, 20, 30] -> ordinals [0, 1, 2]

        // Check mappings
        assert_abs_diff_eq!(encoded[[0, 0]], 1.0, epsilon = 1e-10); // 2 -> ordinal 1
        assert_abs_diff_eq!(encoded[[0, 1]], 0.0, epsilon = 1e-10); // 10 -> ordinal 0
        assert_abs_diff_eq!(encoded[[1, 0]], 0.0, epsilon = 1e-10); // 1 -> ordinal 0
        assert_abs_diff_eq!(encoded[[1, 1]], 1.0, epsilon = 1e-10); // 20 -> ordinal 1
        assert_abs_diff_eq!(encoded[[2, 0]], 2.0, epsilon = 1e-10); // 3 -> ordinal 2
        assert_abs_diff_eq!(encoded[[2, 1]], 0.0, epsilon = 1e-10); // 10 -> ordinal 0
        assert_abs_diff_eq!(encoded[[3, 0]], 1.0, epsilon = 1e-10); // 2 -> ordinal 1
        assert_abs_diff_eq!(encoded[[3, 1]], 2.0, epsilon = 1e-10); // 30 -> ordinal 2
    }

    #[test]
    fn test_unknown_category_handling() {
        let train_data = Array::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();

        let test_data = Array::from_shape_vec(
            (1, 1),
            vec![3.0], // Unknown category
        )
        .unwrap();

        // Test error handling
        let mut encoder = OneHotEncoder::default(); // default is handle_unknown="error"
        encoder.fit(&train_data).unwrap();
        assert!(encoder.transform(&test_data).is_err());

        // Test ignore handling
        let mut encoder = OneHotEncoder::new(None, "ignore", false).unwrap();
        encoder.fit(&train_data).unwrap();
        let encoded = encoder.transform(&test_data).unwrap();

        // Should be all zeros (ignored unknown category)
        assert_eq!(encoded.shape(), &[1, 2]);
        assert_abs_diff_eq!(encoded[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(encoded[[0, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ordinal_encoder_unknown_value() {
        let train_data = Array::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();

        let test_data = Array::from_shape_vec(
            (1, 1),
            vec![3.0], // Unknown category
        )
        .unwrap();

        let mut encoder = OrdinalEncoder::new("use_encoded_value", Some(-1.0)).unwrap();
        encoder.fit(&train_data).unwrap();
        let encoded = encoder.transform(&test_data).unwrap();

        // Should use the specified unknown value
        assert_eq!(encoded.shape(), &[1, 1]);
        assert_abs_diff_eq!(encoded[[0, 0]], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_get_feature_names() {
        let data = Array::from_shape_vec((2, 2), vec![1.0, 10.0, 2.0, 20.0]).unwrap();

        let mut encoder = OneHotEncoder::default();
        encoder.fit(&data).unwrap();

        let feature_names = encoder.get_feature_names(None).unwrap();
        assert_eq!(feature_names.len(), 4); // 2 cats per feature * 2 features

        let custom_names = vec!["feat_a".to_string(), "feat_b".to_string()];
        let feature_names = encoder.get_feature_names(Some(&custom_names)).unwrap();
        assert!(feature_names[0].starts_with("feat_a_cat_"));
        assert!(feature_names[2].starts_with("feat_b_cat_"));
    }
}
