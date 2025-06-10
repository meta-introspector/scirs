//! Feature Selection Demo for Time Series Data
//!
//! This example demonstrates comprehensive feature selection methods specifically
//! designed for time series data. It shows how to extract features from time series
//! and then select the most relevant ones using various techniques.

use ndarray::{Array1, Array2};
use scirs2_series::feature_selection::{
    EmbeddedMethods, FeatureSelectionConfig, FeatureSelector, FilterMethods, TimeSeriesMethods,
    WrapperMethods,
};
use scirs2_series::features::{extract_features, FeatureExtractionOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Time Series Feature Selection Demo");
    println!("=====================================\n");

    // Create a realistic time series with patterns and noise
    let n = 200;
    let mut ts1 = Array1::zeros(n);
    let mut ts2 = Array1::zeros(n);
    let mut ts3 = Array1::zeros(n);
    let mut target = Array1::zeros(n);

    // Generate time series with different characteristics
    for i in 0..n {
        let t = i as f64;

        // First series: seasonal pattern with trend
        ts1[i] = 10.0 + 0.02 * t + 3.0 * (t / 12.0).sin() + 0.5 * (t / 50.0).cos();

        // Second series: high-frequency oscillation
        ts2[i] = 5.0 + 2.0 * (t / 3.0).sin() + 0.8 * (t / 7.0).cos();

        // Third series: noise with some correlation to target
        ts3[i] = 2.0 + 0.5 * (t / 20.0).sin() + 0.3 * rand::random::<f64>();

        // Target: combination of first two series with noise
        target[i] = 0.6 * ts1[i] + 0.3 * ts2[i] + 0.1 * ts3[i] + 0.2 * rand::random::<f64>();
    }

    println!("📊 Generated time series data:");
    println!("   - {} observations", n);
    println!("   - 3 feature time series");
    println!("   - 1 target variable\n");

    // Extract comprehensive features from each time series
    let mut options = FeatureExtractionOptions::default();
    options.calculate_complexity = true;
    options.calculate_frequency_features = true;
    options.detect_temporal_patterns = true;
    options.motif_length = Some(10);

    println!("🔧 Extracting features from time series...");

    let features1 = extract_features(&ts1, &options)?;
    let features2 = extract_features(&ts2, &options)?;
    let features3 = extract_features(&ts3, &options)?;

    // Create feature matrix
    let features_vec = vec![
        features1.mean,
        features1.std_dev,
        features1.skewness,
        features1.kurtosis,
        features1.trend_strength,
        features1.acf1,
        features1.cv,
        features1.complexity_features.approximate_entropy,
        features1.complexity_features.sample_entropy,
        features1.complexity_features.turning_points as f64,
        features1.frequency_features.spectral_centroid,
        features1.frequency_features.spectral_entropy,
        features2.mean,
        features2.std_dev,
        features2.skewness,
        features2.kurtosis,
        features2.trend_strength,
        features2.acf1,
        features2.cv,
        features2.complexity_features.approximate_entropy,
        features2.complexity_features.sample_entropy,
        features2.complexity_features.turning_points as f64,
        features2.frequency_features.spectral_centroid,
        features2.frequency_features.spectral_entropy,
        features3.mean,
        features3.std_dev,
        features3.skewness,
        features3.kurtosis,
        features3.trend_strength,
        features3.acf1,
        features3.cv,
        features3.complexity_features.approximate_entropy,
        features3.complexity_features.sample_entropy,
        features3.complexity_features.turning_points as f64,
        features3.frequency_features.spectral_centroid,
        features3.frequency_features.spectral_entropy,
    ];

    let n_features = features_vec.len();
    let feature_matrix = Array2::from_shape_vec((1, n_features), features_vec)?;

    println!("   ✓ Extracted {} features per observation\n", n_features);

    // For a more realistic demo, create multiple observations
    let mut full_feature_matrix = Array2::zeros((50, n_features));
    let mut full_target = Array1::zeros(50);

    // Generate multiple feature observations with slight variations
    for i in 0..50 {
        for j in 0..n_features {
            // Add small random variations to features
            full_feature_matrix[[i, j]] = feature_matrix[[0, j]]
                + 0.1 * feature_matrix[[0, j]] * (rand::random::<f64>() - 0.5);
        }
        full_target[i] = 100.0 + 10.0 * (i as f64 / 10.0).sin() + 5.0 * rand::random::<f64>();
    }

    println!(
        "🎯 Created dataset with {} observations and {} features\n",
        50, n_features
    );

    // Demonstrate Filter Methods
    println!("=== Filter Methods ===");

    // Variance threshold
    println!("🔍 1. Variance Threshold Selection:");
    let variance_result = FilterMethods::variance_threshold(&full_feature_matrix, 0.01)?;
    println!(
        "   Selected {} features based on variance > 0.01",
        variance_result.selected_features.len()
    );
    println!(
        "   Features: {:?}",
        variance_result
            .selected_features
            .iter()
            .take(5)
            .collect::<Vec<_>>()
    );

    // Correlation-based selection
    println!("\n🔍 2. Correlation-based Selection:");
    let correlation_result =
        FilterMethods::correlation_selection(&full_feature_matrix, &full_target, 0.1)?;
    println!(
        "   Selected {} features with |correlation| > 0.1",
        correlation_result.selected_features.len()
    );
    println!(
        "   Top features: {:?}",
        correlation_result
            .selected_features
            .iter()
            .take(5)
            .collect::<Vec<_>>()
    );

    // Mutual information
    println!("\n🔍 3. Mutual Information Selection:");
    let mi_result = FilterMethods::mutual_information_selection(
        &full_feature_matrix,
        &full_target,
        5,
        Some(8),
    )?;
    println!(
        "   Selected {} features based on mutual information",
        mi_result.selected_features.len()
    );
    println!(
        "   Features: {:?}",
        mi_result
            .selected_features
            .iter()
            .take(5)
            .collect::<Vec<_>>()
    );

    // Demonstrate Wrapper Methods
    println!("\n=== Wrapper Methods ===");

    let mut config = FeatureSelectionConfig::default();
    config.n_features = Some(8);

    // Forward selection
    println!("🔍 4. Forward Feature Selection:");
    let forward_result =
        WrapperMethods::forward_selection(&full_feature_matrix, &full_target, &config)?;
    println!(
        "   Selected {} features using forward selection",
        forward_result.selected_features.len()
    );
    println!("   Features: {:?}", forward_result.selected_features);

    // Backward elimination
    println!("\n🔍 5. Backward Feature Elimination:");
    let backward_result =
        WrapperMethods::backward_elimination(&full_feature_matrix, &full_target, &config)?;
    println!(
        "   Selected {} features using backward elimination",
        backward_result.selected_features.len()
    );
    println!("   Features: {:?}", backward_result.selected_features);

    // Demonstrate Embedded Methods
    println!("\n=== Embedded Methods ===");

    // LASSO selection
    println!("🔍 6. LASSO Feature Selection:");
    let lasso_result =
        EmbeddedMethods::lasso_selection(&full_feature_matrix, &full_target, 0.1, 100)?;
    println!(
        "   Selected {} features using LASSO regularization",
        lasso_result.selected_features.len()
    );
    println!(
        "   Features: {:?}",
        lasso_result
            .selected_features
            .iter()
            .take(8)
            .collect::<Vec<_>>()
    );

    // Tree-based importance
    println!("\n🔍 7. Tree-based Feature Selection:");
    let tree_result =
        EmbeddedMethods::tree_based_selection(&full_feature_matrix, &full_target, Some(5))?;
    println!(
        "   Selected {} most important features",
        tree_result.selected_features.len()
    );
    println!("   Features: {:?}", tree_result.selected_features);

    // Demonstrate Time Series Specific Methods
    println!("\n=== Time Series Specific Methods ===");

    // Lag-based selection
    println!("🔍 8. Lag-based Feature Selection:");
    let lag_result =
        TimeSeriesMethods::lag_based_selection(&full_feature_matrix, &full_target, 3, Some(5))?;
    println!(
        "   Selected {} features based on predictive power at different lags",
        lag_result.selected_features.len()
    );
    println!("   Features: {:?}", lag_result.selected_features);

    // Cross-correlation selection
    println!("\n🔍 9. Cross-correlation Feature Selection:");
    let ccf_result =
        TimeSeriesMethods::cross_correlation_selection(&full_feature_matrix, &full_target, 5, 0.1)?;
    println!(
        "   Selected {} features based on cross-correlation",
        ccf_result.selected_features.len()
    );
    println!(
        "   Features: {:?}",
        ccf_result
            .selected_features
            .iter()
            .take(5)
            .collect::<Vec<_>>()
    );

    // Demonstrate Automatic Feature Selection
    println!("\n=== Automatic Feature Selection ===");

    println!("🤖 10. Automatic Feature Selection (Ensemble):");
    let auto_config = FeatureSelectionConfig {
        n_features: Some(10),
        cv_folds: 3,
        scoring_method: scirs2_series::feature_selection::ScoringMethod::MeanSquaredError,
        ..Default::default()
    };

    let auto_result =
        FeatureSelector::auto_select(&full_feature_matrix, Some(&full_target), &auto_config)?;
    println!(
        "   Selected {} features using ensemble voting",
        auto_result.selected_features.len()
    );
    println!("   Features: {:?}", auto_result.selected_features);
    println!(
        "   Feature scores: {:?}",
        auto_result
            .feature_scores
            .iter()
            .enumerate()
            .filter(|(_, &score)| score > 0.0)
            .take(5)
            .collect::<Vec<_>>()
    );

    // Performance comparison
    println!("\n=== Performance Comparison ===");
    println!("📊 Method Comparison:");
    println!(
        "   Filter (Variance):      {} features",
        variance_result.selected_features.len()
    );
    println!(
        "   Filter (Correlation):   {} features",
        correlation_result.selected_features.len()
    );
    println!(
        "   Filter (Mutual Info):   {} features",
        mi_result.selected_features.len()
    );
    println!(
        "   Wrapper (Forward):      {} features",
        forward_result.selected_features.len()
    );
    println!(
        "   Wrapper (Backward):     {} features",
        backward_result.selected_features.len()
    );
    println!(
        "   Embedded (LASSO):       {} features",
        lasso_result.selected_features.len()
    );
    println!(
        "   Embedded (Tree):        {} features",
        tree_result.selected_features.len()
    );
    println!(
        "   Time Series (Lag):      {} features",
        lag_result.selected_features.len()
    );
    println!(
        "   Time Series (CCF):      {} features",
        ccf_result.selected_features.len()
    );
    println!(
        "   Automatic (Ensemble):   {} features",
        auto_result.selected_features.len()
    );

    println!("\n✅ Feature selection demo completed successfully!");
    println!("\n💡 Key Insights:");
    println!("   • Different methods select different numbers of features");
    println!("   • Each method has its own strengths for different types of data");
    println!("   • Time series-specific methods consider temporal relationships");
    println!("   • Automatic selection provides robust feature sets using ensemble voting");
    println!("   • Feature selection is crucial for building effective time series models");

    Ok(())
}
