//! Turning Points Analysis Demo
//!
//! This example demonstrates comprehensive turning points analysis for time series data.
//! It shows how to extract various turning point features including local extrema detection,
//! directional changes, momentum patterns, trend reversals, and advanced chart patterns.

use ndarray::Array1;
use scirs2_series::features::{extract_features, FeatureExtractionOptions, TurningPointsConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Turning Points Analysis Demo");
    println!("==============================\n");

    // Create diverse synthetic time series with different turning point characteristics
    let n = 200;

    println!("🏗️  Generating synthetic time series with different patterns:");
    println!("   - Financial-like signal with peaks and valleys");
    println!("   - Oscillating signal with regular turning points");
    println!("   - Trending signal with occasional reversals");
    println!("   - Noisy signal with irregular turning points");
    println!("   - Complex multi-frequency signal\n");

    // 1. Financial-like signal with peaks and valleys
    println!("=== Financial-like Signal Analysis ===");
    let mut financial_ts = Array1::zeros(n);
    for i in 0..n {
        let t = i as f64;
        // Simulate stock-like movement with trend, volatility, and jumps
        let trend = 0.02 * t;
        let volatility = 5.0 * (t / 50.0).sin();
        let jump = if i == 100 {
            15.0
        } else if i == 150 {
            -10.0
        } else {
            0.0
        };
        let noise = 2.0 * (rand::random::<f64>() - 0.5);
        financial_ts[i] = 100.0 + trend + volatility + jump + noise;
    }

    let mut options = FeatureExtractionOptions::default();
    options.calculate_turning_points_features = true;

    let config = TurningPointsConfig {
        min_turning_point_threshold: 0.02, // 2% threshold
        extrema_window_size: 5,
        major_reversal_threshold: 0.1, // 10% for major reversals
        detect_advanced_patterns: true,
        calculate_temporal_patterns: true,
        analyze_clustering: true,
        multiscale_analysis: true,
        smoothing_windows: vec![3, 7, 14, 21], // Different timeframes
        ..Default::default()
    };
    options.turning_points_config = Some(config.clone());

    let financial_features = extract_features(&financial_ts, &options)?;
    let financial_tp = &financial_features.turning_points_features;

    display_turning_points_features("Financial-like", financial_tp);

    // 2. Oscillating signal with regular turning points
    println!("\n=== Oscillating Signal Analysis ===");
    let mut oscillating_ts = Array1::zeros(n);
    for i in 0..n {
        let t = i as f64;
        // Multiple oscillations with different frequencies
        oscillating_ts[i] = 10.0 * (t / 20.0).sin() + 5.0 * (t / 8.0).sin() + 2.0 * (t / 3.0).sin();
    }

    let oscillating_features = extract_features(&oscillating_ts, &options)?;
    let oscillating_tp = &oscillating_features.turning_points_features;

    display_turning_points_features("Oscillating", oscillating_tp);

    // 3. Trending signal with occasional reversals
    println!("\n=== Trending Signal Analysis ===");
    let mut trending_ts = Array1::zeros(n);
    for i in 0..n {
        let t = i as f64;
        let base_trend = if i < 50 {
            0.1 * t
        } else if i < 120 {
            5.0 - 0.05 * (t - 50.0)
        } else {
            0.1 * (t - 120.0) + 1.5
        };
        let noise = 0.5 * (rand::random::<f64>() - 0.5);
        trending_ts[i] = base_trend + noise;
    }

    let trending_features = extract_features(&trending_ts, &options)?;
    let trending_tp = &trending_features.turning_points_features;

    display_turning_points_features("Trending", trending_tp);

    // 4. Noisy signal with irregular turning points
    println!("\n=== Noisy Signal Analysis ===");
    let mut noisy_ts = Array1::zeros(n);
    for i in 0..n {
        let t = i as f64;
        let signal = (t / 30.0).sin();
        let noise = 3.0 * (rand::random::<f64>() - 0.5);
        noisy_ts[i] = signal + noise;
    }

    let noisy_features = extract_features(&noisy_ts, &options)?;
    let noisy_tp = &noisy_features.turning_points_features;

    display_turning_points_features("Noisy", noisy_tp);

    // 5. Complex multi-frequency signal
    println!("\n=== Complex Multi-frequency Signal Analysis ===");
    let mut complex_ts = Array1::zeros(n);
    for i in 0..n {
        let t = i as f64;
        // Multiple components with different scales
        let low_freq = 8.0 * (t / 50.0).sin();
        let mid_freq = 4.0 * (t / 15.0).sin();
        let high_freq = 2.0 * (t / 5.0).sin();
        let trend = 0.05 * t;
        complex_ts[i] = trend + low_freq + mid_freq + high_freq;
    }

    let complex_features = extract_features(&complex_ts, &options)?;
    let complex_tp = &complex_features.turning_points_features;

    display_turning_points_features("Complex", complex_tp);

    // Comparative analysis
    println!("\n=== Comparative Turning Points Analysis ===");
    compare_turning_points_characteristics(&[
        ("Financial", financial_tp),
        ("Oscillating", oscillating_tp),
        ("Trending", trending_tp),
        ("Noisy", noisy_tp),
        ("Complex", complex_tp),
    ]);

    // Configuration demonstration
    println!("\n=== Configuration Options Demo ===");
    demonstrate_configurations(&financial_ts)?;

    // Advanced pattern analysis
    println!("\n=== Advanced Pattern Detection ===");
    demonstrate_pattern_detection()?;

    // Applications and insights
    println!("\n=== Practical Applications & Insights ===");
    println!("💡 Turning Points Analysis Applications:");
    println!("   • Financial Markets: Support/resistance levels, trend reversals");
    println!("   • Technical Analysis: Chart pattern recognition (M/W, H&S)");
    println!("   • Signal Processing: Peak detection, signal segmentation");
    println!("   • Quality Control: Process monitoring, anomaly detection");
    println!("   • Economic Analysis: Business cycle turning points");
    println!("   • Biomedical: Heart rate variability, EEG analysis");

    println!("\n🔧 Configuration Guidelines:");
    println!("   • Lower thresholds: More sensitive, more turning points");
    println!("   • Larger windows: Smoother detection, fewer false positives");
    println!("   • Major reversal threshold: Distinguish significant changes");
    println!("   • Advanced patterns: Enable for financial/technical analysis");
    println!("   • Multiscale analysis: Capture patterns at different timescales");
    println!("   • Temporal patterns: Understand regularity and clustering");

    println!("\n📊 Interpretation Tips:");
    println!("   • High turning point frequency → High volatility/noise");
    println!("   • Low stability index → Unstable, frequently changing");
    println!("   • High momentum persistence → Strong trending behavior");
    println!("   • Clustered turning points → Regime changes or volatility periods");
    println!("   • Regular turning point intervals → Cyclical behavior");
    println!("   • Advanced patterns → Specific market/signal characteristics");

    println!("\n⚠️  Analysis Considerations:");
    println!("   • Threshold selection affects sensitivity vs specificity");
    println!("   • Window size balances noise reduction vs responsiveness");
    println!("   • Advanced patterns may have low occurrence rates");
    println!("   • Temporal patterns require sufficient data for reliability");
    println!("   • Multiscale analysis helps understand hierarchical structure");

    println!("\n🎯 Domain-Specific Usage:");
    println!("   • Finance: Focus on reversal patterns, momentum, stability");
    println!("   • Engineering: Emphasize extrema characterization, noise ratio");
    println!("   • Biology: Analyze periodicity, clustering, temporal patterns");
    println!("   • Economics: Track major reversals, trend consistency");
    println!("   • Signal Processing: Multi-scale analysis, pattern recognition");

    println!("\n✅ Turning points analysis demo completed successfully!");
    println!("   Comprehensive analysis provides insights into signal dynamics,");
    println!("   trend changes, and pattern characteristics across multiple scales.");

    Ok(())
}

fn display_turning_points_features(
    name: &str,
    features: &scirs2_series::features::TurningPointsFeatures<f64>,
) {
    println!("🔄 {} Signal Turning Points Features:", name);

    // Basic counts
    println!("   Basic Turning Point Statistics:");
    println!(
        "     Total turning points: {}",
        features.total_turning_points
    );
    println!("     Local maxima: {}", features.local_maxima_count);
    println!("     Local minima: {}", features.local_minima_count);
    println!("     Peak/valley ratio: {:.3}", features.peak_valley_ratio);
    println!(
        "     Average distance: {:.2}",
        features.average_turning_point_distance
    );

    // Directional changes
    println!("   Directional Change Analysis:");
    println!("     Upward changes: {}", features.upward_changes);
    println!("     Downward changes: {}", features.downward_changes);
    println!(
        "     Change ratio: {:.3}",
        features.directional_change_ratio
    );
    println!(
        "     Avg upward magnitude: {:.3}",
        features.average_upward_magnitude
    );
    println!(
        "     Avg downward magnitude: {:.3}",
        features.average_downward_magnitude
    );

    // Momentum and persistence
    println!("   Momentum & Persistence:");
    println!(
        "     Longest upward sequence: {}",
        features.longest_upward_sequence
    );
    println!(
        "     Longest downward sequence: {}",
        features.longest_downward_sequence
    );
    println!(
        "     Avg upward sequence: {:.2}",
        features.average_upward_sequence_length
    );
    println!(
        "     Avg downward sequence: {:.2}",
        features.average_downward_sequence_length
    );
    println!(
        "     Momentum persistence: {:.3}",
        features.momentum_persistence_ratio
    );

    // Extrema characteristics
    println!("   Extrema Characteristics:");
    println!(
        "     Average peak amplitude: {:.3}",
        features.average_peak_amplitude
    );
    println!(
        "     Average valley amplitude: {:.3}",
        features.average_valley_amplitude
    );
    println!(
        "     Peak amplitude std: {:.3}",
        features.peak_amplitude_std
    );
    println!(
        "     Valley amplitude std: {:.3}",
        features.valley_amplitude_std
    );
    println!("     Extrema asymmetry: {:.3}", features.extrema_asymmetry);

    // Trend reversals
    println!("   Trend Reversal Analysis:");
    println!("     Major reversals: {}", features.major_trend_reversals);
    println!("     Minor reversals: {}", features.minor_trend_reversals);
    println!(
        "     Reversal frequency: {:.4}",
        features.trend_reversal_frequency
    );
    println!(
        "     Reversal strength: {:.3}",
        features.reversal_strength_index
    );

    // Stability measures
    println!("   Stability & Volatility:");
    println!(
        "     Turning point volatility: {:.3}",
        features.turning_point_volatility
    );
    println!("     Stability index: {:.2}", features.stability_index);
    println!(
        "     Noise/signal ratio: {:.3}",
        features.noise_signal_ratio
    );
    println!("     Trend consistency: {:.3}", features.trend_consistency);

    // Advanced patterns
    println!("   Advanced Patterns:");
    println!("     Double peaks (M): {}", features.double_peak_count);
    println!("     Double bottoms (W): {}", features.double_bottom_count);
    println!("     Head & shoulders: {}", features.head_shoulders_count);
    println!(
        "     Triangular patterns: {}",
        features.triangular_pattern_count
    );

    // Temporal patterns
    println!("   Temporal Patterns:");
    println!("     Regularity: {:.3}", features.turning_point_regularity);
    println!("     Clustering: {:.3}", features.turning_point_clustering);
    println!(
        "     Periodicity: {:.3}",
        features.turning_point_periodicity
    );
    println!(
        "     Autocorrelation: {:.3}",
        features.turning_point_autocorrelation
    );

    // Position analysis
    println!("   Position Analysis:");
    println!(
        "     Upper half points: {:.2}%",
        features.upper_half_turning_points * 100.0
    );
    println!(
        "     Lower half points: {:.2}%",
        features.lower_half_turning_points * 100.0
    );
    println!(
        "     Position skewness: {:.3}",
        features.turning_point_position_skewness
    );
    println!(
        "     Position kurtosis: {:.3}",
        features.turning_point_position_kurtosis
    );

    // Multi-scale analysis
    if !features.multiscale_turning_points.is_empty() {
        println!("   Multi-scale Analysis:");
        println!(
            "     Scales analyzed: {}",
            features.multiscale_turning_points.len()
        );
        for (i, &count) in features.multiscale_turning_points.iter().enumerate() {
            println!("       Scale {}: {} turning points", i + 1, count);
        }
        println!(
            "     Scale ratio: {:.3}",
            features.scale_turning_point_ratio
        );
        println!(
            "     Cross-scale consistency: {:.3}",
            features.cross_scale_consistency
        );
        println!(
            "     Hierarchical structure: {:.3}",
            features.hierarchical_structure_index
        );
    }
}

fn compare_turning_points_characteristics(
    signals: &[(&str, &scirs2_series::features::TurningPointsFeatures<f64>)],
) {
    println!("🔍 Turning Points Comparison Across Signals:");

    // Compare basic statistics
    println!("   Total Turning Points:");
    for (name, features) in signals {
        println!("     {}: {}", name, features.total_turning_points);
    }

    println!("   Stability Index (higher = more stable):");
    for (name, features) in signals {
        println!("     {}: {:.2}", name, features.stability_index);
    }

    println!("   Momentum Persistence (higher = more trending):");
    for (name, features) in signals {
        println!("     {}: {:.3}", name, features.momentum_persistence_ratio);
    }

    println!("   Trend Consistency (higher = more consistent):");
    for (name, features) in signals {
        println!("     {}: {:.3}", name, features.trend_consistency);
    }

    println!("   Advanced Pattern Counts:");
    for (name, features) in signals {
        let total_patterns = features.double_peak_count
            + features.double_bottom_count
            + features.head_shoulders_count
            + features.triangular_pattern_count;
        println!("     {}: {} patterns", name, total_patterns);
    }

    // Characterize signal types
    println!("\n📊 Signal Characteristics Summary:");
    for (name, features) in signals {
        let volatility_level = if features.stability_index > 100.0 {
            "Low"
        } else if features.stability_index > 20.0 {
            "Medium"
        } else {
            "High"
        };

        let trend_strength = if features.momentum_persistence_ratio > 0.7 {
            "Strong Trending"
        } else if features.momentum_persistence_ratio > 0.3 {
            "Moderate Trending"
        } else {
            "Non-trending"
        };

        let pattern_richness = if features.total_turning_points > 20 {
            "Rich"
        } else if features.total_turning_points > 10 {
            "Moderate"
        } else {
            "Simple"
        };

        println!(
            "     {}: Volatility = {}, Trend = {}, Patterns = {}",
            name, volatility_level, trend_strength, pattern_richness
        );
    }
}

fn demonstrate_configurations(ts: &ndarray::Array1<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Turning Points Configuration Options:");

    // Conservative configuration (fewer, more significant turning points)
    println!("   1. Conservative configuration (high thresholds):");
    let mut conservative_options = FeatureExtractionOptions::default();
    conservative_options.calculate_turning_points_features = true;

    let conservative_config = TurningPointsConfig {
        min_turning_point_threshold: 0.05, // 5% threshold
        extrema_window_size: 7,
        major_reversal_threshold: 0.15, // 15% for major reversals
        detect_advanced_patterns: false,
        calculate_temporal_patterns: false,
        analyze_clustering: false,
        multiscale_analysis: false,
        ..Default::default()
    };
    conservative_options.turning_points_config = Some(conservative_config);

    let conservative_features = extract_features(ts, &conservative_options)?;
    println!(
        "     Turning points detected: {}",
        conservative_features
            .turning_points_features
            .total_turning_points
    );
    println!(
        "     Stability index: {:.2}",
        conservative_features
            .turning_points_features
            .stability_index
    );

    // Sensitive configuration (more turning points, detailed analysis)
    println!("   2. Sensitive configuration (low thresholds, full analysis):");
    let mut sensitive_options = FeatureExtractionOptions::default();
    sensitive_options.calculate_turning_points_features = true;

    let sensitive_config = TurningPointsConfig {
        min_turning_point_threshold: 0.005, // 0.5% threshold
        extrema_window_size: 3,
        major_reversal_threshold: 0.02, // 2% for major reversals
        detect_advanced_patterns: true,
        calculate_temporal_patterns: true,
        analyze_clustering: true,
        multiscale_analysis: true,
        smoothing_windows: vec![3, 5, 7, 10, 15, 20],
        max_autocorr_lag: 30,
        ..Default::default()
    };
    sensitive_options.turning_points_config = Some(sensitive_config);

    let sensitive_features = extract_features(ts, &sensitive_options)?;
    println!(
        "     Turning points detected: {}",
        sensitive_features
            .turning_points_features
            .total_turning_points
    );
    println!(
        "     Advanced patterns found: {}",
        sensitive_features.turning_points_features.double_peak_count
            + sensitive_features
                .turning_points_features
                .double_bottom_count
            + sensitive_features
                .turning_points_features
                .head_shoulders_count
            + sensitive_features
                .turning_points_features
                .triangular_pattern_count
    );

    // Balanced configuration
    println!("   3. Balanced configuration (default with customizations):");
    let mut balanced_options = FeatureExtractionOptions::default();
    balanced_options.calculate_turning_points_features = true;

    let balanced_config = TurningPointsConfig {
        min_turning_point_threshold: 0.02, // 2% threshold
        extrema_window_size: 5,
        major_reversal_threshold: 0.08, // 8% for major reversals
        detect_advanced_patterns: true,
        calculate_temporal_patterns: true,
        analyze_clustering: true,
        multiscale_analysis: true,
        ..Default::default()
    };
    balanced_options.turning_points_config = Some(balanced_config);

    let balanced_features = extract_features(ts, &balanced_options)?;
    println!(
        "     Turning points detected: {}",
        balanced_features
            .turning_points_features
            .total_turning_points
    );
    println!(
        "     Temporal regularity: {:.3}",
        balanced_features
            .turning_points_features
            .turning_point_regularity
    );

    Ok(())
}

fn demonstrate_pattern_detection() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Advanced Pattern Detection Examples:");

    // Create signal with double peak pattern
    let double_peak_signal = ndarray::Array1::from_vec(vec![
        1.0, 2.0, 5.0, 3.0, 4.9, 2.0, 1.0, // Double peak
        0.5, 2.0, 0.7, 2.1, 0.5, // Double bottom
        1.0, 4.0, 7.0, 4.2, 1.0, // Head and shoulders
        3.0, 4.0, 3.5, 3.7, 3.2, 3.4, 3.0, // Triangular convergence
    ]);

    let mut options = FeatureExtractionOptions::default();
    options.calculate_turning_points_features = true;

    let pattern_config = TurningPointsConfig {
        min_turning_point_threshold: 0.05,
        extrema_window_size: 2,
        detect_advanced_patterns: true,
        ..Default::default()
    };
    options.turning_points_config = Some(pattern_config);

    let pattern_features = extract_features(&double_peak_signal, &options)?;
    let pattern_tp = &pattern_features.turning_points_features;

    println!("   Pattern Detection Results:");
    println!(
        "     Double peaks (M patterns): {}",
        pattern_tp.double_peak_count
    );
    println!(
        "     Double bottoms (W patterns): {}",
        pattern_tp.double_bottom_count
    );
    println!("     Head & shoulders: {}", pattern_tp.head_shoulders_count);
    println!(
        "     Triangular patterns: {}",
        pattern_tp.triangular_pattern_count
    );

    // Create signal with regular turning points for temporal analysis
    let regular_signal: ndarray::Array1<f64> = ndarray::Array1::from_vec(
        (0..40)
            .map(|i| if i % 6 == 0 || i % 6 == 1 { 5.0 } else { 1.0 })
            .collect(),
    );

    let temporal_config = TurningPointsConfig {
        calculate_temporal_patterns: true,
        analyze_clustering: true,
        max_autocorr_lag: 15,
        ..Default::default()
    };
    options.turning_points_config = Some(temporal_config);

    let temporal_features = extract_features(&regular_signal, &options)?;
    let temporal_tp = &temporal_features.turning_points_features;

    println!("\n   Temporal Pattern Analysis:");
    println!(
        "     Regularity: {:.3}",
        temporal_tp.turning_point_regularity
    );
    println!(
        "     Clustering: {:.3}",
        temporal_tp.turning_point_clustering
    );
    println!(
        "     Periodicity: {:.3}",
        temporal_tp.turning_point_periodicity
    );
    println!(
        "     Autocorrelation: {:.3}",
        temporal_tp.turning_point_autocorrelation
    );

    Ok(())
}
