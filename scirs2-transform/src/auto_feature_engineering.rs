//! Automated feature engineering with meta-learning
//!
//! This module provides automated feature engineering capabilities that use
//! meta-learning to select optimal transformations for given datasets.

use crate::error::Result;
use ndarray::{Array1, ArrayView2};
use std::collections::HashMap;

#[cfg(feature = "auto-feature-engineering")]
use tch::{nn, Device, Tensor};

/// Meta-features extracted from datasets for transformation selection
#[derive(Debug, Clone)]
pub struct DatasetMetaFeatures {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Sparsity ratio (fraction of zero values)
    pub sparsity: f64,
    /// Mean of feature correlations
    pub mean_correlation: f64,
    /// Standard deviation of feature correlations
    pub std_correlation: f64,
    /// Skewness statistics
    pub mean_skewness: f64,
    /// Kurtosis statistics
    pub mean_kurtosis: f64,
    /// Number of missing values
    pub missing_ratio: f64,
    /// Feature variance statistics
    pub variance_ratio: f64,
    /// Outlier ratio
    pub outlier_ratio: f64,
}

/// Available transformation types for automated selection
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TransformationType {
    /// Standardization (Z-score normalization)
    StandardScaler,
    /// Min-max scaling
    MinMaxScaler,
    /// Robust scaling using median and IQR
    RobustScaler,
    /// Power transformation (Box-Cox/Yeo-Johnson)
    PowerTransformer,
    /// Polynomial feature generation
    PolynomialFeatures,
    /// Principal Component Analysis
    PCA,
    /// Feature selection based on variance
    VarianceThreshold,
    /// Quantile transformation
    QuantileTransformer,
    /// Binary encoding for categorical features
    BinaryEncoder,
    /// Target encoding
    TargetEncoder,
}

/// Configuration for a transformation with its parameters
#[derive(Debug, Clone)]
pub struct TransformationConfig {
    /// Type of transformation to apply
    pub transformation_type: TransformationType,
    /// Parameters for the transformation
    pub parameters: HashMap<String, f64>,
    /// Expected performance score for this transformation
    pub expected_performance: f64,
}

/// Meta-learning model for transformation selection
#[cfg(feature = "auto-feature-engineering")]
pub struct MetaLearningModel {
    /// Neural network for predicting transformation performance
    model: nn::Sequential,
    /// Device for computation (CPU/GPU)
    device: Device,
    /// Training data cache
    training_cache: Vec<(DatasetMetaFeatures, Vec<TransformationConfig>)>,
}

#[cfg(feature = "auto-feature-engineering")]
impl MetaLearningModel {
    /// Create a new meta-learning model
    pub fn new() -> Result<Self> {
        let device = Device::cuda_if_available();
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        // Build neural network architecture
        let model = nn::seq()
            .add(nn::linear(&root / "layer1", 10, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&root / "layer2", 64, 32, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&root / "layer3", 32, 16, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&root / "output", 16, 10, Default::default()))
            .add_fn(|xs| xs.softmax(-1, tch::Kind::Float));

        Ok(MetaLearningModel {
            model,
            device,
            training_cache: Vec::new(),
        })
    }

    /// Train the meta-learning model on historical transformation performance data
    pub fn train(&mut self, training_data: Vec<(DatasetMetaFeatures, Vec<TransformationConfig>)>) -> Result<()> {
        self.training_cache.extend(training_data.clone());

        // Convert training data to tensors
        let (input_features, target_scores) = self.prepare_training_data(&training_data)?;

        // Training loop
        let mut opt = nn::Adam::default().build(&self.model.vs, 1e-3).unwrap();
        
        for epoch in 0..100 {
            let predicted = self.model.forward(&input_features);
            let loss = predicted.mse_loss(&target_scores, tch::Reduction::Mean);
            
            opt.zero_grad();
            loss.backward();
            opt.step();

            if epoch % 20 == 0 {
                println!("Epoch {}: Loss = {:.4}", epoch, f64::from(loss));
            }
        }

        Ok(())
    }

    /// Predict optimal transformations for a given dataset
    pub fn predict_transformations(&self, meta_features: &DatasetMetaFeatures) -> Result<Vec<TransformationConfig>> {
        let input_tensor = self.meta_features_to_tensor(meta_features)?;
        let prediction = self.model.forward(&input_tensor);
        
        // Convert prediction to transformation recommendations
        self.tensor_to_transformations(&prediction)
    }

    fn prepare_training_data(
        &self, 
        training_data: &[(DatasetMetaFeatures, Vec<TransformationConfig>)]
    ) -> Result<(Tensor, Tensor)> {
        let n_samples = training_data.len();
        let mut input_features = Vec::new();
        let mut target_scores = Vec::new();

        for (meta_features, transformations) in training_data {
            let features = vec![
                meta_features.n_samples as f64,
                meta_features.n_features as f64,
                meta_features.sparsity,
                meta_features.mean_correlation,
                meta_features.std_correlation,
                meta_features.mean_skewness,
                meta_features.mean_kurtosis,
                meta_features.missing_ratio,
                meta_features.variance_ratio,
                meta_features.outlier_ratio,
            ];
            input_features.extend(features);

            // Create target vector (transformation type scores)
            let mut scores = vec![0.0; 10]; // Number of transformation types
            for config in transformations {
                let idx = self.transformation_type_to_index(&config.transformation_type);
                scores[idx] = config.expected_performance;
            }
            target_scores.extend(scores);
        }

        let input_tensor = Tensor::of_slice(&input_features)
            .reshape(&[n_samples as i64, 10])
            .to_device(self.device);
        let target_tensor = Tensor::of_slice(&target_scores)
            .reshape(&[n_samples as i64, 10])
            .to_device(self.device);

        Ok((input_tensor, target_tensor))
    }

    fn meta_features_to_tensor(&self, meta_features: &DatasetMetaFeatures) -> Result<Tensor> {
        let features = vec![
            meta_features.n_samples as f64,
            meta_features.n_features as f64,
            meta_features.sparsity,
            meta_features.mean_correlation,
            meta_features.std_correlation,
            meta_features.mean_skewness,
            meta_features.mean_kurtosis,
            meta_features.missing_ratio,
            meta_features.variance_ratio,
            meta_features.outlier_ratio,
        ];

        Ok(Tensor::of_slice(&features)
            .reshape(&[1, 10])
            .to_device(self.device))
    }

    fn tensor_to_transformations(&self, prediction: &Tensor) -> Result<Vec<TransformationConfig>> {
        let scores: Vec<f64> = prediction.double_value(&[0]).into();
        let mut transformations = Vec::new();

        for (i, &score) in scores.iter().enumerate() {
            if score > 0.5 { // Threshold for recommendation
                let transformation_type = self.index_to_transformation_type(i);
                let config = TransformationConfig {
                    transformation_type,
                    parameters: self.get_default_parameters_for_type(&self.index_to_transformation_type(i)),
                    expected_performance: score,
                };
                transformations.push(config);
            }
        }

        // Sort by expected performance
        transformations.sort_by(|a, b| b.expected_performance.partial_cmp(&a.expected_performance).unwrap());
        
        Ok(transformations)
    }

    fn transformation_type_to_index(&self, t_type: &TransformationType) -> usize {
        match t_type {
            TransformationType::StandardScaler => 0,
            TransformationType::MinMaxScaler => 1,
            TransformationType::RobustScaler => 2,
            TransformationType::PowerTransformer => 3,
            TransformationType::PolynomialFeatures => 4,
            TransformationType::PCA => 5,
            TransformationType::VarianceThreshold => 6,
            TransformationType::QuantileTransformer => 7,
            TransformationType::BinaryEncoder => 8,
            TransformationType::TargetEncoder => 9,
        }
    }

    fn index_to_transformation_type(&self, index: usize) -> TransformationType {
        match index {
            0 => TransformationType::StandardScaler,
            1 => TransformationType::MinMaxScaler,
            2 => TransformationType::RobustScaler,
            3 => TransformationType::PowerTransformer,
            4 => TransformationType::PolynomialFeatures,
            5 => TransformationType::PCA,
            6 => TransformationType::VarianceThreshold,
            7 => TransformationType::QuantileTransformer,
            8 => TransformationType::BinaryEncoder,
            9 => TransformationType::TargetEncoder,
            _ => TransformationType::StandardScaler,
        }
    }

    fn get_default_parameters_for_type(&self, t_type: &TransformationType) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        match t_type {
            TransformationType::PCA => {
                params.insert("n_components".to_string(), 0.95); // Keep 95% variance
            },
            TransformationType::PolynomialFeatures => {
                params.insert("degree".to_string(), 2.0);
                params.insert("include_bias".to_string(), 0.0);
            },
            TransformationType::VarianceThreshold => {
                params.insert("threshold".to_string(), 0.01);
            },
            _ => {}, // Use defaults for other transformations
        }
        params
    }
}

/// Automated feature engineering pipeline
pub struct AutoFeatureEngineer {
    #[cfg(feature = "auto-feature-engineering")]
    meta_model: MetaLearningModel,
    #[allow(dead_code)]
    transformation_history: Vec<(DatasetMetaFeatures, Vec<TransformationConfig>, f64)>,
}

impl AutoFeatureEngineer {
    /// Create a new automated feature engineer
    pub fn new() -> Result<Self> {
        #[cfg(feature = "auto-feature-engineering")]
        let meta_model = MetaLearningModel::new()?;

        Ok(AutoFeatureEngineer {
            #[cfg(feature = "auto-feature-engineering")]
            meta_model,
            transformation_history: Vec::new(),
        })
    }

    /// Extract meta-features from a dataset
    pub fn extract_meta_features(&self, x: &ArrayView2<f64>) -> Result<DatasetMetaFeatures> {
        let (n_samples, n_features) = x.dim();
        
        // Calculate sparsity
        let zeros = x.iter().filter(|&&val| val == 0.0).count();
        let sparsity = zeros as f64 / (n_samples * n_features) as f64;

        // Calculate correlation statistics
        let correlations = self.compute_feature_correlations(x)?;
        let mean_correlation = correlations.mean().unwrap_or(0.0);
        let std_correlation = correlations.std(0.0);

        // Calculate skewness and kurtosis
        let (mean_skewness, mean_kurtosis) = self.compute_distribution_stats(x)?;

        // Calculate missing values (assuming NaN represents missing)
        let missing_count = x.iter().filter(|val| val.is_nan()).count();
        let missing_ratio = missing_count as f64 / (n_samples * n_features) as f64;

        // Calculate variance statistics
        let variances: Array1<f64> = x.var_axis(ndarray::Axis(0), 0.0);
        let variance_ratio = variances.std(0.0) / variances.mean().unwrap_or(1.0);

        // Calculate outlier ratio (using IQR method)
        let outlier_ratio = self.compute_outlier_ratio(x)?;

        Ok(DatasetMetaFeatures {
            n_samples,
            n_features,
            sparsity,
            mean_correlation,
            std_correlation,
            mean_skewness,
            mean_kurtosis,
            missing_ratio,
            variance_ratio,
            outlier_ratio,
        })
    }

    /// Recommend optimal transformations for a dataset
    #[cfg(feature = "auto-feature-engineering")]
    pub fn recommend_transformations(&self, x: &ArrayView2<f64>) -> Result<Vec<TransformationConfig>> {
        let meta_features = self.extract_meta_features(x)?;
        self.meta_model.predict_transformations(&meta_features)
    }

    /// Recommend optimal transformations for a dataset (fallback implementation)
    #[cfg(not(feature = "auto-feature-engineering"))]
    pub fn recommend_transformations(&self, x: &ArrayView2<f64>) -> Result<Vec<TransformationConfig>> {
        // Fallback to rule-based recommendations
        self.rule_based_recommendations(x)
    }

    /// Rule-based transformation recommendations (fallback)
    fn rule_based_recommendations(&self, x: &ArrayView2<f64>) -> Result<Vec<TransformationConfig>> {
        let meta_features = self.extract_meta_features(x)?;
        let mut recommendations = Vec::new();

        // Rule 1: High skewness -> Power transformation
        if meta_features.mean_skewness.abs() > 1.0 {
            recommendations.push(TransformationConfig {
                transformation_type: TransformationType::PowerTransformer,
                parameters: HashMap::new(),
                expected_performance: 0.8,
            });
        }

        // Rule 2: High dimensionality -> PCA
        if meta_features.n_features > 100 {
            let mut params = HashMap::new();
            params.insert("n_components".to_string(), 0.95);
            recommendations.push(TransformationConfig {
                transformation_type: TransformationType::PCA,
                parameters: params,
                expected_performance: 0.75,
            });
        }

        // Rule 3: Different scales -> StandardScaler
        if meta_features.variance_ratio > 1.0 {
            recommendations.push(TransformationConfig {
                transformation_type: TransformationType::StandardScaler,
                parameters: HashMap::new(),
                expected_performance: 0.9,
            });
        }

        // Rule 4: High outlier ratio -> RobustScaler
        if meta_features.outlier_ratio > 0.1 {
            recommendations.push(TransformationConfig {
                transformation_type: TransformationType::RobustScaler,
                parameters: HashMap::new(),
                expected_performance: 0.85,
            });
        }

        // Sort by expected performance
        recommendations.sort_by(|a, b| b.expected_performance.partial_cmp(&a.expected_performance).unwrap());

        Ok(recommendations)
    }

    /// Train the meta-learning model with new data
    #[cfg(feature = "auto-feature-engineering")]
    pub fn update_model(&mut self, 
        meta_features: DatasetMetaFeatures, 
        transformations: Vec<TransformationConfig>,
        performance: f64
    ) -> Result<()> {
        self.transformation_history.push((meta_features.clone(), transformations.clone(), performance));
        
        // Retrain every 10 new examples
        if self.transformation_history.len() % 10 == 0 {
            let training_data: Vec<_> = self.transformation_history.iter()
                .map(|(meta, trans, _perf)| (meta.clone(), trans.clone()))
                .collect();
            self.meta_model.train(training_data)?;
        }

        Ok(())
    }

    fn compute_feature_correlations(&self, x: &ArrayView2<f64>) -> Result<Array1<f64>> {
        let n_features = x.ncols();
        let mut correlations = Vec::new();

        for i in 0..n_features {
            for j in i+1..n_features {
                let col_i = x.column(i);
                let col_j = x.column(j);
                let correlation = self.pearson_correlation(&col_i, &col_j);
                correlations.push(correlation);
            }
        }

        Ok(Array1::from_vec(correlations))
    }

    fn pearson_correlation(&self, x: &ndarray::ArrayView1<f64>, y: &ndarray::ArrayView1<f64>) -> f64 {
        let _n = x.len() as f64;
        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let numerator: f64 = x.iter().zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn compute_distribution_stats(&self, x: &ArrayView2<f64>) -> Result<(f64, f64)> {
        let mut skewness_values = Vec::new();
        let mut kurtosis_values = Vec::new();

        for col in x.columns() {
            let mean = col.mean().unwrap_or(0.0);
            let variance = col.var(0.0);
            let std = variance.sqrt();

            if std > 1e-10 {
                let n = col.len() as f64;
                
                // Skewness
                let skew: f64 = col.iter()
                    .map(|&val| ((val - mean) / std).powi(3))
                    .sum::<f64>() / n;
                skewness_values.push(skew);

                // Kurtosis
                let kurt: f64 = col.iter()
                    .map(|&val| ((val - mean) / std).powi(4))
                    .sum::<f64>() / n - 3.0; // Excess kurtosis
                kurtosis_values.push(kurt);
            }
        }

        let mean_skewness = skewness_values.iter().sum::<f64>() / skewness_values.len() as f64;
        let mean_kurtosis = kurtosis_values.iter().sum::<f64>() / kurtosis_values.len() as f64;

        Ok((mean_skewness, mean_kurtosis))
    }

    fn compute_outlier_ratio(&self, x: &ArrayView2<f64>) -> Result<f64> {
        let mut total_outliers = 0;
        let mut total_values = 0;

        for col in x.columns() {
            let mut sorted_col = col.to_vec();
            sorted_col.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let n = sorted_col.len();
            if n < 4 { continue; }

            let q1 = sorted_col[n / 4];
            let q3 = sorted_col[3 * n / 4];
            let iqr = q3 - q1;
            let lower_bound = q1 - 1.5 * iqr;
            let upper_bound = q3 + 1.5 * iqr;

            let outliers = col.iter()
                .filter(|&&val| val < lower_bound || val > upper_bound)
                .count();

            total_outliers += outliers;
            total_values += n;
        }

        Ok(total_outliers as f64 / total_values as f64)
    }
}