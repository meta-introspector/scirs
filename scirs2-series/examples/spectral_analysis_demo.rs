//! Spectral Analysis Features Demo
//!
//! This example demonstrates comprehensive spectral analysis feature extraction for time series data.
//! It shows how to extract various spectral measures including power spectral density estimation,
//! spectral peak detection, frequency band analysis, spectral entropy measures, and multi-scale analysis.

use ndarray::Array1;
use scirs2_series::features::{extract_features, FeatureExtractionOptions, SpectralAnalysisConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌈 Spectral Analysis Features Demo");
    println!("=================================\n");

    // Create diverse synthetic time series with different spectral characteristics
    let n = 400;

    println!("🏗️  Generating synthetic time series with different spectral characteristics:");
    println!("   - Pure sinusoidal signal (narrow spectrum)");
    println!("   - Multi-frequency signal (multiple spectral peaks)");
    println!("   - Chirp signal (varying frequency)");
    println!("   - White noise (broad spectrum)");
    println!("   - AR signal (colored noise spectrum)\n");

    // 1. Pure sinusoidal signal (narrow spectrum)
    println!("=== Pure Sinusoidal Signal Analysis ===");
    let mut sinusoidal_ts = Array1::zeros(n);
    for i in 0..n {
        let t = i as f64;
        // Pure sine wave at normalized frequency 0.1
        sinusoidal_ts[i] = 2.0 * (t / 10.0).sin();
    }

    let mut options = FeatureExtractionOptions::default();
    options.calculate_frequency_features = true;

    let spectral_config = SpectralAnalysisConfig {
        calculate_welch_psd: true,
        detect_spectral_peaks: true,
        calculate_eeg_bands: true,
        calculate_spectral_shannon_entropy: true,
        calculate_spectral_flatness: true,
        calculate_harmonic_analysis: true,
        enable_multiscale_analysis: true,
        multiscale_scales: 3,
        welch_window_length_factor: 0.5,
        welch_overlap_factor: 0.5,
        min_peak_height: 0.1,
        peak_prominence_threshold: 0.05,
        ..Default::default()
    };
    options.spectral_analysis_config = Some(spectral_config.clone());

    let sinusoidal_features = extract_features(&sinusoidal_ts, &options)?;

    // Display spectral analysis features
    display_spectral_features(
        "Pure Sinusoidal",
        &sinusoidal_features.frequency_features.spectral_analysis,
    );

    // 2. Multi-frequency signal (multiple spectral peaks)
    println!("\n=== Multi-frequency Signal Analysis ===");
    let mut multifreq_ts = Array1::zeros(n);
    for i in 0..n {
        let t = i as f64;
        // Multiple sine waves at different frequencies
        multifreq_ts[i] = 1.5 * (t / 15.0).sin() +     // Low frequency
            1.0 * (t / 8.0).sin() +      // Medium frequency
            0.5 * (t / 4.0).sin(); // High frequency
    }

    let multifreq_features = extract_features(&multifreq_ts, &options)?;

    display_spectral_features(
        "Multi-frequency",
        &multifreq_features.frequency_features.spectral_analysis,
    );

    // 3. Chirp signal (varying frequency)
    println!("\n=== Chirp Signal Analysis ===");
    let mut chirp_ts = Array1::zeros(n);
    for i in 0..n {
        let t = i as f64;
        let freq = 5.0 + 10.0 * t / n as f64; // Linearly increasing frequency
        chirp_ts[i] = (t / freq).sin();
    }

    let chirp_features = extract_features(&chirp_ts, &options)?;

    display_spectral_features(
        "Chirp",
        &chirp_features.frequency_features.spectral_analysis,
    );

    // 4. White noise (broad spectrum)
    println!("\n=== White Noise Analysis ===");
    let mut noise_ts = Array1::zeros(n);
    for i in 0..n {
        noise_ts[i] = rand::random::<f64>() - 0.5;
    }

    let noise_features = extract_features(&noise_ts, &options)?;

    display_spectral_features(
        "White Noise",
        &noise_features.frequency_features.spectral_analysis,
    );

    // 5. AR signal (colored noise spectrum)
    println!("\n=== AR Signal Analysis ===");
    let mut ar_ts = Array1::zeros(n);
    ar_ts[0] = rand::random::<f64>() - 0.5;
    for i in 1..n {
        // AR(1) process with coefficient 0.8
        ar_ts[i] = 0.8 * ar_ts[i - 1] + 0.3 * (rand::random::<f64>() - 0.5);
    }

    let ar_features = extract_features(&ar_ts, &options)?;

    display_spectral_features(
        "AR Signal",
        &ar_features.frequency_features.spectral_analysis,
    );

    // Configuration demonstration
    println!("\n=== Configuration Options Demo ===");
    demonstrate_spectral_configurations(&sinusoidal_ts)?;

    // Practical applications and insights
    println!("\n=== Practical Applications & Insights ===");
    println!("💡 Spectral Analysis Applications:");
    println!("   • Signal Processing: Filter design, noise characterization");
    println!("   • Audio Analysis: Music information retrieval, speech processing");
    println!("   • Biomedical: EEG/ECG frequency analysis, heart rate variability");
    println!("   • Vibration Analysis: Machine health monitoring, fault detection");
    println!("   • Communications: Channel characterization, interference analysis");
    println!("   • Geophysics: Seismic data analysis, earthquake detection");

    println!("\n🔧 Configuration Guidelines:");
    println!("   • Welch window length: Balance frequency resolution vs variance");
    println!("   • Overlap factor: Higher overlap reduces variance, increases computation");
    println!("   • Peak detection thresholds: Adjust based on signal-to-noise ratio");
    println!("   • Multiscale analysis: Capture patterns at different resolutions");
    println!("   • Frequency bands: Customize for domain-specific applications");

    println!("\n📊 Interpretation Tips:");
    println!("   • Spectral centroid: Average frequency (brightness measure)");
    println!("   • Spectral spread: Frequency dispersion around centroid");
    println!("   • Spectral entropy: Measure of spectral complexity/randomness");
    println!("   • Peak characteristics: Dominant frequencies and their properties");
    println!("   • Frequency band ratios: Relative power distribution");

    println!("\n⚠️  Analysis Considerations:");
    println!("   • Window effects: Consider windowing artifacts in PSD estimation");
    println!("   • Frequency resolution: Limited by signal length and windowing");
    println!("   • Spectral leakage: May affect peak detection accuracy");
    println!("   • Stationarity assumption: Spectral methods assume stationary signals");
    println!("   • Computational cost: Consider efficiency for real-time applications");

    println!("\n🎯 Domain-Specific Usage:");
    println!("   • Audio: Focus on harmonic content, spectral centroid, rolloff");
    println!("   • Biomedical: Emphasize frequency bands, spectral entropy");
    println!("   • Machinery: Peak detection, harmonic analysis, power concentration");
    println!("   • Communications: Spectral efficiency, bandwidth utilization");
    println!("   • Finance: Spectral density estimation for cyclical patterns");

    println!("\n✅ Spectral analysis demo completed successfully!");
    println!("   Comprehensive spectral characterization provides insights into");
    println!("   frequency domain properties, periodicity, and spectral complexity.");

    Ok(())
}

fn display_spectral_features(
    name: &str,
    features: &scirs2_series::features::SpectralAnalysisFeatures<f64>,
) {
    println!("🌈 {} Signal Spectral Features:", name);

    // Power Spectral Density features
    println!("   Power Spectral Density:");
    println!("     Total power: {:.6}", features.total_power);
    if features.welch_psd.len() > 0 {
        println!("     Welch PSD points: {}", features.welch_psd.len());
        let max_psd = features.welch_psd.iter().fold(0.0f64, |a, &b| a.max(b));
        println!("     Max Welch PSD: {:.6}", max_psd);
    }
    if features.periodogram_psd.len() > 0 {
        println!(
            "     Periodogram PSD points: {}",
            features.periodogram_psd.len()
        );
    }
    if features.ar_psd.len() > 0 {
        println!("     AR PSD points: {}", features.ar_psd.len());
    }

    // Spectral peak features
    println!("   Spectral Peaks:");
    println!("     Peak count: {}", features.significant_peaks_count);
    if features.peak_frequencies.len() > 0 {
        println!(
            "     Peak frequencies: {:?}",
            &features.peak_frequencies[..features.peak_frequencies.len().min(3)]
        );
        println!(
            "     Peak magnitudes: {:?}",
            &features.peak_magnitudes[..features.peak_magnitudes.len().min(3)]
        );
        println!("     Peak density: {:.4}", features.peak_density);
        if features.average_peak_spacing.is_finite() {
            println!(
                "     Average peak spacing: {:.4}",
                features.average_peak_spacing
            );
        }
    }

    // Frequency band analysis
    println!("   Frequency Bands:");
    println!("     Delta power: {:.6}", features.delta_power);
    println!("     Theta power: {:.6}", features.theta_power);
    println!("     Alpha power: {:.6}", features.alpha_power);
    println!("     Beta power: {:.6}", features.beta_power);
    println!("     Gamma power: {:.6}", features.gamma_power);

    // Spectral entropy measures
    println!("   Spectral Entropy:");
    println!(
        "     Shannon entropy: {:.4}",
        features.spectral_shannon_entropy
    );
    println!("     Renyi entropy: {:.4}", features.spectral_renyi_entropy);
    println!(
        "     Permutation entropy: {:.4}",
        features.spectral_permutation_entropy
    );
    println!(
        "     Sample entropy: {:.4}",
        features.spectral_sample_entropy
    );

    // Spectral shape measures
    println!("   Spectral Shape:");
    println!("     Spectral flatness: {:.4}", features.spectral_flatness);
    println!(
        "     Spectral crest factor: {:.4}",
        features.spectral_crest_factor
    );
    println!(
        "     Spectral irregularity: {:.4}",
        features.spectral_irregularity
    );
    println!(
        "     Spectral smoothness: {:.4}",
        features.spectral_smoothness
    );
    println!("     Spectral slope: {:.4}", features.spectral_slope);
    println!(
        "     Spectral brightness: {:.4}",
        features.spectral_brightness
    );

    // Harmonic analysis
    println!("   Harmonic Analysis:");
    println!("     Spectral purity: {:.4}", features.spectral_purity);
    println!(
        "     Harmonic-to-noise ratio: {:.4}",
        features.harmonic_noise_ratio
    );
    println!(
        "     Spectral inharmonicity: {:.4}",
        features.spectral_inharmonicity
    );
    println!(
        "     Frequency stability: {:.4}",
        features.frequency_stability
    );

    // Multi-scale analysis
    if features.multiscale_spectral_entropy.len() > 1 {
        println!("   Multi-scale Analysis:");
        println!(
            "     Scales analyzed: {}",
            features.multiscale_spectral_entropy.len()
        );
        println!(
            "     Cross-scale correlations: {}",
            features.cross_scale_spectral_correlations.len()
        );
        println!(
            "     Hierarchical spectral index: {:.4}",
            features.hierarchical_spectral_index
        );
        println!(
            "     Scale features: {}",
            features.scale_spectral_features.len()
        );
    }
}

fn demonstrate_spectral_configurations(ts: &Array1<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Spectral Analysis Configuration Options:");

    // Basic configuration (fast)
    println!("   1. Basic spectral configuration (fast):");
    let mut basic_options = FeatureExtractionOptions::default();
    basic_options.calculate_frequency_features = true;

    let basic_config = SpectralAnalysisConfig {
        calculate_welch_psd: true,
        detect_spectral_peaks: true,
        calculate_eeg_bands: false,
        calculate_spectral_shannon_entropy: false,
        calculate_spectral_flatness: true,
        calculate_harmonic_analysis: false,
        enable_multiscale_analysis: false,
        welch_window_length_factor: 0.25,
        max_peaks: 5,
        ..Default::default()
    };
    basic_options.spectral_analysis_config = Some(basic_config);

    let basic_features = extract_features(ts, &basic_options)?;
    let spectral_features = &basic_features.frequency_features.spectral_analysis;
    println!("     Total power: {:.6}", spectral_features.total_power);
    println!(
        "     Peaks detected: {}",
        spectral_features.significant_peaks_count
    );
    println!(
        "     Spectral flatness: {:.4}",
        spectral_features.spectral_flatness
    );

    // Comprehensive configuration (detailed)
    println!("   2. Comprehensive spectral configuration (detailed):");
    let mut comprehensive_options = FeatureExtractionOptions::default();
    comprehensive_options.calculate_frequency_features = true;

    let comprehensive_config = SpectralAnalysisConfig {
        calculate_welch_psd: true,
        calculate_periodogram_psd: true,
        calculate_ar_psd: true,
        detect_spectral_peaks: true,
        calculate_eeg_bands: true,
        calculate_spectral_shannon_entropy: true,
        calculate_spectral_flatness: true,
        calculate_harmonic_analysis: true,
        enable_multiscale_analysis: true,
        multiscale_scales: 5,
        welch_window_length_factor: 0.5,
        welch_overlap_factor: 0.75,
        max_peaks: 20,
        peak_prominence_threshold: 0.01,
        ..Default::default()
    };
    comprehensive_options.spectral_analysis_config = Some(comprehensive_config);

    let comprehensive_features = extract_features(ts, &comprehensive_options)?;
    let spectral_features = &comprehensive_features.frequency_features.spectral_analysis;
    println!("     Total power: {:.6}", spectral_features.total_power);
    println!(
        "     Peaks detected: {}",
        spectral_features.significant_peaks_count
    );
    println!(
        "     Shannon entropy: {:.4}",
        spectral_features.spectral_shannon_entropy
    );
    println!(
        "     Harmonic-to-noise ratio: {:.4}",
        spectral_features.harmonic_noise_ratio
    );
    println!(
        "     Multi-scale entropy scales: {}",
        spectral_features.multiscale_spectral_entropy.len()
    );

    Ok(())
}
