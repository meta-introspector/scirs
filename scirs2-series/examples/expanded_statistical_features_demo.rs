//! Expanded Statistical Features Demo
//!
//! This example demonstrates comprehensive expanded statistical feature extraction for time series data.
//! It shows how to extract advanced statistical measures including higher-order moments, robust statistics,
//! distribution characteristics, tail measures, normality tests, and concentration measures.

use ndarray::Array1;
use scirs2_series::features::{
    extract_features, ExpandedStatisticalConfig, FeatureExtractionOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Expanded Statistical Features Demo");
    println!("=====================================\n");

    // Create multiple synthetic time series with different characteristics
    let n = 500;

    println!("🏗️  Generating synthetic time series with different characteristics:");
    println!("   - Normal-like distribution");
    println!("   - Skewed distribution");
    println!("   - Heavy-tailed distribution with outliers");
    println!("   - Multi-modal distribution");
    println!("   - Trend with heteroscedasticity\n");

    // 1. Normal-like distribution
    println!("=== Normal-like Distribution Analysis ===");
    let mut normal_ts = Array1::zeros(n);
    for i in 0..n {
        // Approximate normal using Box-Muller transform
        let u1 = rand::random::<f64>();
        let u2 = rand::random::<f64>();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        normal_ts[i] = 10.0 + 2.0 * z0; // N(10, 2^2)
    }

    let mut normal_options = FeatureExtractionOptions::default();
    normal_options.calculate_expanded_statistical_features = true;

    let normal_config = ExpandedStatisticalConfig {
        calculate_higher_order_moments: true,
        calculate_robust_statistics: true,
        calculate_percentiles: true,
        calculate_distribution_characteristics: true,
        calculate_tail_statistics: true,
        calculate_central_tendency_variations: true,
        calculate_variability_measures: true,
        calculate_normality_tests: true,
        calculate_advanced_shape_measures: true,
        calculate_count_statistics: true,
        calculate_concentration_measures: true,
        trimming_fraction_10: 0.1,
        trimming_fraction_20: 0.2,
        winsorizing_fraction: 0.05,
        mode_bins: Some(20),
        normality_alpha: 0.05,
        use_fast_approximations: false,
    };
    normal_options.expanded_statistical_config = Some(normal_config);

    let normal_features = extract_features(&normal_ts, &normal_options)?;
    let normal_expanded = &normal_features.expanded_statistical_features;

    display_expanded_features("Normal-like", normal_expanded);

    // 2. Skewed distribution
    println!("\n=== Skewed Distribution Analysis ===");
    let mut skewed_ts = Array1::zeros(n);
    for i in 0..n {
        // Create right-skewed data using exponential-like transformation
        let uniform = rand::random::<f64>();
        skewed_ts[i] = 5.0 + (-2.0 * uniform.ln()); // Exponential-like
    }

    let skewed_features = extract_features(&skewed_ts, &normal_options)?;
    let skewed_expanded = &skewed_features.expanded_statistical_features;

    display_expanded_features("Skewed", skewed_expanded);

    // 3. Heavy-tailed with outliers
    println!("\n=== Heavy-tailed with Outliers Analysis ===");
    let mut heavy_ts = Array1::zeros(n);
    for i in 0..n {
        let base_value = 10.0 + 2.0 * (rand::random::<f64>() - 0.5);
        // Add occasional extreme outliers
        if rand::random::<f64>() < 0.05 {
            // 5% outliers
            heavy_ts[i] = base_value + 20.0 * (rand::random::<f64>() - 0.5);
        } else {
            heavy_ts[i] = base_value;
        }
    }

    let heavy_features = extract_features(&heavy_ts, &normal_options)?;
    let heavy_expanded = &heavy_features.expanded_statistical_features;

    display_expanded_features("Heavy-tailed", heavy_expanded);

    // 4. Multi-modal distribution
    println!("\n=== Multi-modal Distribution Analysis ===");
    let mut multimodal_ts = Array1::zeros(n);
    for i in 0..n {
        // Create bimodal distribution
        if rand::random::<f64>() < 0.4 {
            multimodal_ts[i] = 5.0 + (rand::random::<f64>() - 0.5); // First mode
        } else if rand::random::<f64>() < 0.8 {
            multimodal_ts[i] = 15.0 + (rand::random::<f64>() - 0.5); // Second mode
        } else {
            multimodal_ts[i] = 10.0 + 3.0 * (rand::random::<f64>() - 0.5); // Bridge values
        }
    }

    let multimodal_features = extract_features(&multimodal_ts, &normal_options)?;
    let multimodal_expanded = &multimodal_features.expanded_statistical_features;

    display_expanded_features("Multi-modal", multimodal_expanded);

    // 5. Trend with heteroscedasticity
    println!("\n=== Trend with Heteroscedasticity Analysis ===");
    let mut hetero_ts = Array1::zeros(n);
    for i in 0..n {
        let t = i as f64 / n as f64;
        let trend = 5.0 + 10.0 * t; // Linear trend
        let volatility = 0.5 + 2.0 * t; // Increasing variance
        let noise = volatility * (rand::random::<f64>() - 0.5);
        hetero_ts[i] = trend + noise;
    }

    let hetero_features = extract_features(&hetero_ts, &normal_options)?;
    let hetero_expanded = &hetero_features.expanded_statistical_features;

    display_expanded_features("Heteroscedastic", hetero_expanded);

    // Comparison Analysis
    println!("\n=== Comparative Analysis ===");
    compare_distributions_analysis(&[
        ("Normal-like", normal_expanded),
        ("Skewed", skewed_expanded),
        ("Heavy-tailed", heavy_expanded),
        ("Multi-modal", multimodal_expanded),
        ("Heteroscedastic", hetero_expanded),
    ]);

    // Configuration demonstration
    println!("\n=== Configuration Options Demo ===");
    demonstrate_configuration_options(&normal_ts)?;

    // Practical applications
    println!("\n=== Practical Applications & Insights ===");
    println!("💡 Feature Categories & Applications:");
    println!("   • Higher-order moments: Detect asymmetry and tail behavior");
    println!("   • Robust statistics: Handle outliers and non-normal data");
    println!("   • Percentile measures: Understand distribution shape");
    println!("   • Tail statistics: Identify extreme value patterns");
    println!("   • Normality tests: Assess distributional assumptions");
    println!("   • Concentration measures: Analyze data diversity");

    println!("\n🔧 Configuration Guidelines:");
    println!("   • Enable all categories for comprehensive analysis");
    println!("   • Adjust trimming fractions based on outlier sensitivity");
    println!("   • Use fast approximations for large datasets");
    println!("   • Set appropriate mode bins for categorical-like data");

    println!("\n📊 Interpretation Tips:");
    println!("   • High fifth/sixth moments indicate extreme asymmetry/kurtosis");
    println!("   • Large MAD suggests outlier presence");
    println!("   • Extreme percentile ratios indicate skewness");
    println!("   • Low normality scores suggest non-Gaussian behavior");
    println!("   • High concentration measures indicate low diversity");

    println!("\n✅ Expanded statistical features demo completed successfully!");
    println!("   Comprehensive statistical characterization provides rich insights");
    println!("   for anomaly detection, distribution modeling, and feature engineering.");

    Ok(())
}

fn display_expanded_features(
    name: &str,
    features: &scirs2_series::features::ExpandedStatisticalFeatures<f64>,
) {
    println!("📊 {} Distribution Features:", name);

    // Higher-order moments
    println!("   Higher-order Moments:");
    println!("     Fifth moment: {:.4}", features.fifth_moment);
    println!("     Sixth moment: {:.4}", features.sixth_moment);
    println!("     Excess kurtosis: {:.4}", features.excess_kurtosis);

    // Robust statistics
    println!("   Robust Statistics:");
    println!("     Trimmed mean (10%): {:.4}", features.trimmed_mean_10);
    println!("     Trimmed mean (20%): {:.4}", features.trimmed_mean_20);
    println!(
        "     Winsorized mean (5%): {:.4}",
        features.winsorized_mean_5
    );
    println!(
        "     Median absolute deviation: {:.4}",
        features.median_absolute_deviation
    );
    println!(
        "     Interquartile mean: {:.4}",
        features.interquartile_mean
    );

    // Percentile measures
    println!("   Percentile Analysis:");
    println!(
        "     P5: {:.3}, P10: {:.3}, P90: {:.3}, P95: {:.3}, P99: {:.3}",
        features.p5, features.p10, features.p90, features.p95, features.p99
    );
    println!("     P90/P10 ratio: {:.3}", features.percentile_ratio_90_10);
    println!("     P95/P5 ratio: {:.3}", features.percentile_ratio_95_5);

    // Shape measures
    println!("   Shape & Distribution:");
    println!("     Gini coefficient: {:.4}", features.gini_coefficient);
    println!(
        "     Index of dispersion: {:.4}",
        features.index_of_dispersion
    );
    println!("     Bowley skewness: {:.4}", features.bowley_skewness);
    println!("     Kelly skewness: {:.4}", features.kelly_skewness);

    // Tail statistics
    println!("   Tail Analysis:");
    println!("     Lower tail ratio: {:.4}", features.lower_tail_ratio);
    println!("     Upper tail ratio: {:.4}", features.upper_tail_ratio);
    println!("     Outlier ratio: {:.4}", features.outlier_ratio);
    println!(
        "     Lower outliers: {}, Upper outliers: {}",
        features.lower_outlier_count, features.upper_outlier_count
    );

    // Central tendency variations
    println!("   Central Tendency Variations:");
    println!("     Harmonic mean: {:.4}", features.harmonic_mean);
    println!("     Geometric mean: {:.4}", features.geometric_mean);
    println!("     Quadratic mean: {:.4}", features.quadratic_mean);
    println!(
        "     Mode approximation: {:.4}",
        features.mode_approximation
    );

    // Normality indicators
    println!("   Normality Assessment:");
    println!(
        "     Jarque-Bera statistic: {:.4}",
        features.jarque_bera_statistic
    );
    println!(
        "     Anderson-Darling statistic: {:.4}",
        features.anderson_darling_statistic
    );
    println!("     Normality score: {:.4}", features.normality_score);

    // Advanced shape measures
    println!("   Advanced Shape Measures:");
    println!(
        "     Biweight midvariance: {:.4}",
        features.biweight_midvariance
    );
    println!("     Qn estimator: {:.4}", features.qn_estimator);
    println!("     Sn estimator: {:.4}", features.sn_estimator);

    // Count statistics
    println!("   Count-based Statistics:");
    println!("     Zero crossings: {}", features.zero_crossings);
    println!(
        "     Positive count: {}, Negative count: {}",
        features.positive_count, features.negative_count
    );
    println!(
        "     Local maxima: {}, Local minima: {}",
        features.local_maxima_count, features.local_minima_count
    );

    // Concentration measures
    println!("   Concentration & Diversity:");
    println!("     Herfindahl index: {:.4}", features.herfindahl_index);
    println!("     Shannon diversity: {:.4}", features.shannon_diversity);
    println!("     Simpson diversity: {:.4}", features.simpson_diversity);
}

fn compare_distributions_analysis(
    distributions: &[(
        &str,
        &scirs2_series::features::ExpandedStatisticalFeatures<f64>,
    )],
) {
    println!("🔍 Distribution Comparison:");

    // Compare key characteristics
    println!("   Skewness Comparison (Bowley):");
    for (name, features) in distributions {
        println!("     {}: {:.4}", name, features.bowley_skewness);
    }

    println!("   Tail Behavior (Outlier Ratio):");
    for (name, features) in distributions {
        println!(
            "     {}: {:.4} ({:.1}%)",
            name,
            features.outlier_ratio,
            features.outlier_ratio * 100.0
        );
    }

    println!("   Normality Assessment:");
    for (name, features) in distributions {
        println!("     {}: {:.4}", name, features.normality_score);
    }

    println!("   Concentration (Gini Coefficient):");
    for (name, features) in distributions {
        println!("     {}: {:.4}", name, features.gini_coefficient);
    }

    println!("   Higher-order Behavior (Excess Kurtosis):");
    for (name, features) in distributions {
        println!("     {}: {:.4}", name, features.excess_kurtosis);
    }
}

fn demonstrate_configuration_options(ts: &Array1<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Configuration Options Demonstration:");

    // Fast approximation configuration
    println!("   1. Fast approximation mode:");
    let mut fast_options = FeatureExtractionOptions::default();
    fast_options.calculate_expanded_statistical_features = true;

    let fast_config = ExpandedStatisticalConfig {
        calculate_higher_order_moments: true,
        calculate_robust_statistics: false, // Disable expensive calculations
        calculate_percentiles: true,
        calculate_distribution_characteristics: false,
        calculate_tail_statistics: true,
        calculate_central_tendency_variations: false,
        calculate_variability_measures: true,
        calculate_normality_tests: false, // Disable expensive tests
        calculate_advanced_shape_measures: false,
        calculate_count_statistics: true,
        calculate_concentration_measures: false,
        use_fast_approximations: true,
        ..Default::default()
    };
    fast_options.expanded_statistical_config = Some(fast_config);

    let fast_features = extract_features(ts, &fast_options)?;
    println!("     Features calculated with reduced computational cost");
    println!(
        "     Outlier ratio: {:.4}",
        fast_features.expanded_statistical_features.outlier_ratio
    );

    // High-precision configuration
    println!("   2. High-precision mode:");
    let mut precise_options = FeatureExtractionOptions::default();
    precise_options.calculate_expanded_statistical_features = true;

    let precise_config = ExpandedStatisticalConfig {
        calculate_higher_order_moments: true,
        calculate_robust_statistics: true,
        calculate_percentiles: true,
        calculate_distribution_characteristics: true,
        calculate_tail_statistics: true,
        calculate_central_tendency_variations: true,
        calculate_variability_measures: true,
        calculate_normality_tests: true,
        calculate_advanced_shape_measures: true,
        calculate_count_statistics: true,
        calculate_concentration_measures: true,
        trimming_fraction_10: 0.05, // More conservative trimming
        trimming_fraction_20: 0.1,
        winsorizing_fraction: 0.025,
        mode_bins: Some(50),   // Higher resolution
        normality_alpha: 0.01, // Stricter significance
        use_fast_approximations: false,
    };
    precise_options.expanded_statistical_config = Some(precise_config);

    let precise_features = extract_features(ts, &precise_options)?;
    println!("     Features calculated with maximum precision");
    println!(
        "     Normality score: {:.6}",
        precise_features
            .expanded_statistical_features
            .normality_score
    );

    Ok(())
}
