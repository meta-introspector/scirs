//! Window-based Aggregation Features Demo
//!
//! This example demonstrates comprehensive window-based feature extraction for time series data.
//! It shows how to extract multi-scale features using sliding windows, including rolling statistics,
//! cross-window correlations, change detection, and advanced financial indicators.

use ndarray::Array1;
use scirs2_series::features::{extract_features, FeatureExtractionOptions, WindowConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Window-based Aggregation Features Demo");
    println!("========================================\n");

    // Create complex synthetic time series with multiple patterns
    let n = 300;
    let mut ts = Array1::zeros(n);

    println!("🏗️  Generating synthetic time series with complex patterns:");
    println!("   - Linear trend component");
    println!("   - Multiple seasonal cycles");
    println!("   - Structural breaks and regime changes");
    println!("   - Non-stationary variance");
    println!("   - Random noise\n");

    for i in 0..n {
        let t = i as f64;

        // Base trend with structural break
        let trend = if i < 150 { 0.02 * t } else { 0.02 * t + 3.0 };

        // Multiple seasonal components
        let annual = 2.0 * (t / 52.0 * 2.0 * std::f64::consts::PI).sin();
        let quarterly = 1.0 * (t / 13.0 * 2.0 * std::f64::consts::PI).sin();
        let weekly = 0.5 * (t / 7.0 * 2.0 * std::f64::consts::PI).sin();

        // Regime-dependent volatility
        let volatility = if i > 100 && i < 200 { 2.0 } else { 1.0 };
        let noise = volatility * 0.3 * (rand::random::<f64>() - 0.5);

        // Financial-like patterns
        let jump = if i == 180 { 5.0 } else { 0.0 }; // Market shock

        ts[i] = 10.0 + trend + annual + quarterly + weekly + noise + jump;
    }

    // Configure comprehensive window-based feature extraction
    let mut options = FeatureExtractionOptions::default();
    options.calculate_window_features = true;

    // Custom window configuration for detailed analysis
    let window_config = WindowConfig {
        small_window_size: 7,   // Weekly patterns
        medium_window_size: 30, // Monthly patterns
        large_window_size: 90,  // Quarterly patterns
        calculate_cross_correlations: true,
        detect_changes: true,
        calculate_bollinger_bands: true,
        calculate_macd: true,
        calculate_rsi: true,
        rsi_period: 14,
        macd_fast_period: 12,
        macd_slow_period: 26,
        macd_signal_period: 9,
        bollinger_std_dev: 2.0,
        ewma_alpha: 0.1,
        change_threshold: 2.0,
    };
    options.window_config = Some(window_config);

    println!("🔧 Window-based Configuration:");
    println!("   - Small windows: 7 observations (weekly patterns)");
    println!("   - Medium windows: 30 observations (monthly patterns)");
    println!("   - Large windows: 90 observations (quarterly patterns)");
    println!("   - Cross-correlations: Enabled");
    println!("   - Change detection: Enabled (threshold: 2.0)");
    println!("   - Bollinger bands: Enabled (2σ)");
    println!("   - MACD: Enabled (12,26,9)");
    println!("   - RSI: Enabled (14-period)\n");

    // Extract comprehensive features including window-based
    println!("🧮 Extracting window-based aggregation features...");
    let features = extract_features(&ts, &options)?;
    let window_features = &features.window_based_features;

    // Display multi-scale window analysis
    println!("\n=== Multi-scale Window Analysis ===");

    println!("🔍 Small Window Features (7-period):");
    let small = &window_features.small_window_features;
    println!(
        "   Number of rolling windows: {}",
        small.rolling_means.len()
    );
    if !small.rolling_means.is_empty() {
        println!(
            "   Rolling means range: {:.3} to {:.3}",
            small
                .rolling_mins
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b)),
            small
                .rolling_maxs
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        );
        println!(
            "   Average rolling std: {:.3}",
            small.rolling_stds.iter().sum::<f64>() / small.rolling_stds.len() as f64
        );
        println!(
            "   Mean coefficient of variation: {:.3}",
            small.rolling_cv.iter().sum::<f64>() / small.rolling_cv.len() as f64
        );
    }

    println!("\n🔍 Medium Window Features (30-period):");
    let medium = &window_features.medium_window_features;
    println!(
        "   Number of rolling windows: {}",
        medium.rolling_means.len()
    );
    if !medium.rolling_means.is_empty() {
        println!(
            "   Rolling means range: {:.3} to {:.3}",
            medium
                .rolling_mins
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b)),
            medium
                .rolling_maxs
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        );
        println!(
            "   Average rolling std: {:.3}",
            medium.rolling_stds.iter().sum::<f64>() / medium.rolling_stds.len() as f64
        );
    }

    println!("\n🔍 Large Window Features (90-period):");
    let large = &window_features.large_window_features;
    println!(
        "   Number of rolling windows: {}",
        large.rolling_means.len()
    );
    if !large.rolling_means.is_empty() {
        println!(
            "   Rolling means range: {:.3} to {:.3}",
            large
                .rolling_mins
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b)),
            large
                .rolling_maxs
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        );
        println!(
            "   Average rolling std: {:.3}",
            large.rolling_stds.iter().sum::<f64>() / large.rolling_stds.len() as f64
        );
    }

    // Display summary statistics for each window size
    println!("\n=== Window Summary Statistics ===");

    println!("📊 Small Window Summary:");
    let small_summary = &small.summary_stats;
    println!("   Mean of means: {:.3}", small_summary.mean_of_means);
    println!("   Std of means: {:.3}", small_summary.std_of_means);
    println!("   Trend in means: {:.6}", small_summary.trend_in_means);
    println!(
        "   Variability index: {:.3}",
        small_summary.variability_index
    );

    println!("\n📊 Medium Window Summary:");
    let medium_summary = &medium.summary_stats;
    println!("   Mean of means: {:.3}", medium_summary.mean_of_means);
    println!("   Std of means: {:.3}", medium_summary.std_of_means);
    println!("   Trend in means: {:.6}", medium_summary.trend_in_means);
    println!(
        "   Variability index: {:.3}",
        medium_summary.variability_index
    );

    println!("\n📊 Large Window Summary:");
    let large_summary = &large.summary_stats;
    println!("   Mean of means: {:.3}", large_summary.mean_of_means);
    println!("   Std of means: {:.3}", large_summary.std_of_means);
    println!("   Trend in means: {:.6}", large_summary.trend_in_means);
    println!(
        "   Variability index: {:.3}",
        large_summary.variability_index
    );

    // Display multi-scale variance and trend analysis
    println!("\n=== Multi-scale Variance & Trend Analysis ===");
    println!("🔄 Multi-scale Variance:");
    for (i, &variance) in window_features.multi_scale_variance.iter().enumerate() {
        let window_name = match i {
            0 => "Small (7)",
            1 => "Medium (30)",
            2 => "Large (90)",
            _ => "Unknown",
        };
        println!("   {} window variance: {:.4}", window_name, variance);
    }

    println!("\n📈 Multi-scale Trends:");
    for (i, &trend) in window_features.multi_scale_trends.iter().enumerate() {
        let window_name = match i {
            0 => "Small (7)",
            1 => "Medium (30)",
            2 => "Large (90)",
            _ => "Unknown",
        };
        println!("   {} window trend: {:.6}", window_name, trend);
    }

    // Display cross-window correlations
    println!("\n=== Cross-window Correlation Analysis ===");
    let cross_corr = &window_features.cross_window_correlations;
    println!("🔗 Cross-scale Correlations:");
    println!(
        "   Small-Medium correlation: {:.4}",
        cross_corr.small_medium_correlation
    );
    println!(
        "   Medium-Large correlation: {:.4}",
        cross_corr.medium_large_correlation
    );
    println!(
        "   Small-Large correlation: {:.4}",
        cross_corr.small_large_correlation
    );
    println!(
        "   Cross-scale consistency: {:.4}",
        cross_corr.cross_scale_consistency
    );
    println!(
        "   Multi-scale coherence: {:.4}",
        cross_corr.multi_scale_coherence
    );

    // Display change detection results
    println!("\n=== Change Detection Analysis ===");
    let change_det = &window_features.change_detection_features;
    println!("🚨 Detected Changes:");
    println!("   Mean change points: {}", change_det.mean_change_points);
    println!(
        "   Variance change points: {}",
        change_det.variance_change_points
    );
    println!("   Max CUSUM mean: {:.4}", change_det.max_cusum_mean);
    println!(
        "   Max CUSUM variance: {:.4}",
        change_det.max_cusum_variance
    );
    println!("   Stability measure: {:.4}", change_det.stability_measure);
    println!(
        "   Relative change magnitude: {:.4}",
        change_det.relative_change_magnitude
    );

    // Display rolling statistics
    println!("\n=== Advanced Rolling Statistics ===");
    let rolling = &window_features.rolling_statistics;

    println!("📈 EWMA Analysis:");
    if !rolling.ewma.is_empty() {
        let final_ewma = rolling.ewma[rolling.ewma.len() - 1];
        let initial_ewma = rolling.ewma[0];
        println!("   Initial EWMA: {:.3}", initial_ewma);
        println!("   Final EWMA: {:.3}", final_ewma);
        println!("   EWMA change: {:.3}", final_ewma - initial_ewma);
    }

    // Display Bollinger band analysis
    println!("\n🎯 Bollinger Band Analysis:");
    let bollinger = &rolling.bollinger_bands;
    if !bollinger.upper_band.is_empty() {
        println!("   Band entries: {}", bollinger.upper_band.len());
        println!(
            "   Percent above upper band: {:.2}%",
            bollinger.percent_above_upper * 100.0
        );
        println!(
            "   Percent below lower band: {:.2}%",
            bollinger.percent_below_lower * 100.0
        );
        println!("   Mean band width: {:.3}", bollinger.mean_band_width);
        println!("   Squeeze periods: {}", bollinger.squeeze_periods);
    }

    // Display MACD analysis
    println!("\n📊 MACD Analysis:");
    let macd = &rolling.macd_features;
    if !macd.macd_line.is_empty() {
        println!("   MACD entries: {}", macd.macd_line.len());
        println!("   Bullish crossovers: {}", macd.bullish_crossovers);
        println!("   Bearish crossovers: {}", macd.bearish_crossovers);
        println!("   Mean histogram: {:.4}", macd.mean_histogram);
        println!("   Divergence measure: {:.4}", macd.divergence_measure);
    }

    // Display RSI analysis
    println!("\n⚖️  RSI Analysis:");
    if !rolling.rsi_values.is_empty() {
        let current_rsi = rolling.rsi_values[rolling.rsi_values.len() - 1];
        let avg_rsi = rolling.rsi_values.iter().sum::<f64>() / rolling.rsi_values.len() as f64;
        let overbought = rolling.rsi_values.iter().filter(|&&x| x > 70.0).count();
        let oversold = rolling.rsi_values.iter().filter(|&&x| x < 30.0).count();

        println!("   Current RSI: {:.2}", current_rsi);
        println!("   Average RSI: {:.2}", avg_rsi);
        println!("   Overbought periods (>70): {}", overbought);
        println!("   Oversold periods (<30): {}", oversold);
    }

    // Display normalized features and outlier detection
    println!("\n=== Outlier Detection & Normalization ===");
    let normalized = &rolling.normalized_features;
    if !normalized.outlier_scores.is_empty() {
        println!("🔍 Outlier Analysis:");
        println!("   Total outliers detected: {}", normalized.outlier_count);
        println!(
            "   Outlier rate: {:.2}%",
            (normalized.outlier_count as f64 / n as f64) * 100.0
        );

        let max_outlier_score = normalized
            .outlier_scores
            .iter()
            .fold(0.0f64, |a, &b| a.max(b));
        println!("   Maximum outlier score: {:.3}", max_outlier_score);
    }

    // Performance and practical insights
    println!("\n=== Performance & Practical Insights ===");
    println!("💡 Feature Dimensionality:");
    println!(
        "   Small window features: ~{} dimensions",
        small.rolling_means.len() + small.rolling_stds.len() + small.rolling_ranges.len()
    );
    println!(
        "   Medium window features: ~{} dimensions",
        medium.rolling_means.len() + medium.rolling_stds.len() + medium.rolling_ranges.len()
    );
    println!(
        "   Large window features: ~{} dimensions",
        large.rolling_means.len() + large.rolling_stds.len() + large.rolling_ranges.len()
    );

    let total_features = if !rolling.ewma.is_empty() && !rolling.rsi_values.is_empty() {
        rolling.ewma.len()
            + rolling.rsi_values.len()
            + bollinger.upper_band.len()
            + macd.macd_line.len()
    } else {
        0
    };
    println!("   Total rolling features: ~{} dimensions", total_features);

    println!("\n📋 Applications & Use Cases:");
    println!("   • Financial market analysis and trading signals");
    println!("   • Quality control and process monitoring");
    println!("   • Anomaly detection in sensor data");
    println!("   • Regime change detection in economic data");
    println!("   • Multi-resolution analysis of complex signals");
    println!("   • Feature engineering for machine learning models");

    println!("\n🔧 Configuration Guidelines:");
    println!("   • Small windows: Capture short-term patterns and noise");
    println!("   • Medium windows: Balance between detail and stability");
    println!("   • Large windows: Identify long-term trends and cycles");
    println!("   • Enable cross-correlations for scale relationships");
    println!("   • Use change detection for regime identification");
    println!("   • Financial indicators for trading applications");

    println!("\n⚠️  Interpretation Notes:");
    println!("   • Higher cross-scale correlations indicate consistent patterns");
    println!("   • Change points reveal structural breaks in data");
    println!("   • Bollinger band violations suggest volatility regimes");
    println!("   • MACD crossovers indicate momentum changes");
    println!("   • RSI extremes (>70, <30) suggest overbought/oversold conditions");
    println!("   • High variability index indicates non-stationary behavior");

    println!("\n✅ Window-based aggregation features demo completed successfully!");
    println!("   Comprehensive multi-scale analysis provides rich feature set");
    println!("   for time series classification, forecasting, and anomaly detection.");

    Ok(())
}
