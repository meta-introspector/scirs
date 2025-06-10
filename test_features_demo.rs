use ndarray::Array1;
use scirs2_series::features::{extract_features, FeatureExtractionOptions};

fn main() {
    // Create a test time series with some interesting complexity patterns
    let ts: Array1<f64> = Array1::from_vec(
        (0..100)
            .map(|i| {
                let t = i as f64 / 10.0;
                // Mix of trend, seasonality, and noise
                let trend = 0.1 * t;
                let seasonal = 2.0 * (2.0 * std::f64::consts::PI * t / 12.0).sin();
                let noise = 0.5 * (t * 1.5).sin() * (t * 0.3).cos();
                trend + seasonal + noise
            })
            .collect(),
    );

    let mut options = FeatureExtractionOptions::default();
    options.calculate_complexity = true;
    options.calculate_frequency_features = true;
    options.frequency_bands = 8;

    match extract_features(&ts, &options) {
        Ok(features) => {
            println!("Time Series Feature Extraction Results:");
            println!("=====================================");
            println!("Basic Features:");
            println!("  Mean: {:.4}", features.mean);
            println!("  Std Dev: {:.4}", features.std_dev);
            println!("  Skewness: {:.4}", features.skewness);
            println!("  Kurtosis: {:.4}", features.kurtosis);
            println!("  Trend Strength: {:.4}", features.trend_strength);
            
            println!("\nComplexity Features:");
            println!("  Approximate Entropy: {:.4}", features.complexity_features.approximate_entropy);
            println!("  Sample Entropy: {:.4}", features.complexity_features.sample_entropy);
            println!("  Permutation Entropy: {:.4}", features.complexity_features.permutation_entropy);
            println!("  Lempel-Ziv Complexity: {:.4}", features.complexity_features.lempel_ziv_complexity);
            println!("  Fractal Dimension: {:.4}", features.complexity_features.fractal_dimension);
            println!("  Hurst Exponent: {:.4}", features.complexity_features.hurst_exponent);
            println!("  DFA Exponent: {:.4}", features.complexity_features.dfa_exponent);
            println!("  Turning Points: {}", features.complexity_features.turning_points);
            println!("  Longest Strike: {}", features.complexity_features.longest_strike);
            
            println!("\nFrequency Domain Features:");
            println!("  Spectral Centroid: {:.4}", features.frequency_features.spectral_centroid);
            println!("  Spectral Spread: {:.4}", features.frequency_features.spectral_spread);
            println!("  Spectral Entropy: {:.4}", features.frequency_features.spectral_entropy);
            println!("  Dominant Frequency: {:.4}", features.frequency_features.dominant_frequency);
            println!("  Spectral Peaks: {}", features.frequency_features.spectral_peaks);
            println!("  Frequency Bands: {:?}", features.frequency_features.frequency_bands.iter()
                .map(|&x| format!("{:.4}", x))
                .collect::<Vec<_>>());
        }
        Err(e) => {
            eprintln!("Error extracting features: {:?}", e);
        }
    }
}