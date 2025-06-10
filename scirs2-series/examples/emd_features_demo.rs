//! EMD (Hilbert-Huang Transform) Features Demo
//!
//! This example demonstrates comprehensive EMD-based feature extraction for time series data.
//! It shows how to extract Intrinsic Mode Functions (IMFs), analyze their characteristics,
//! and compute various EMD-based features including Hilbert spectral analysis.

use ndarray::Array1;
use scirs2_series::features::{
    extract_features, EMDConfig, EdgeMethod, FeatureExtractionOptions, InterpolationMethod,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌊 EMD (Hilbert-Huang Transform) Features Demo");
    println!("==============================================\n");

    // Create complex synthetic time series with multiple components
    let n = 200;
    let mut ts = Array1::zeros(n);

    println!("📊 Generating synthetic time series with multiple components:");
    println!("   - Low-frequency trend");
    println!("   - Medium-frequency oscillation");
    println!("   - High-frequency component");
    println!("   - Random noise\n");

    for i in 0..n {
        let t = i as f64;

        // Multiple frequency components
        let trend = 0.01 * t + 5.0; // Linear trend
        let low_freq = 2.0 * (t / 50.0).sin(); // Low frequency
        let mid_freq = 1.5 * (t / 15.0).sin(); // Medium frequency
        let high_freq = 0.8 * (t / 5.0).sin(); // High frequency
        let noise = 0.3 * rand::random::<f64>(); // Random noise

        ts[i] = trend + low_freq + mid_freq + high_freq + noise;
    }

    // Configure EMD feature extraction
    let mut options = FeatureExtractionOptions::default();
    options.calculate_emd_features = true;

    // Custom EMD configuration for comprehensive analysis
    let emd_config = EMDConfig {
        max_imfs: 8,
        sifting_tolerance: 0.2,
        max_sifting_iterations: 100,
        calculate_hilbert_spectrum: true,
        calculate_instantaneous: true,
        calculate_emd_entropies: true,
        interpolation_method: InterpolationMethod::CubicSpline,
        edge_method: EdgeMethod::Mirror,
    };
    options.emd_config = Some(emd_config);

    println!("🔧 EMD Configuration:");
    println!("   - Maximum IMFs: 8");
    println!("   - Sifting tolerance: 0.2");
    println!("   - Interpolation: Cubic spline");
    println!("   - Edge handling: Mirror reflection");
    println!("   - Hilbert spectrum: Enabled");
    println!("   - Instantaneous features: Enabled");
    println!("   - EMD entropies: Enabled\n");

    // Extract comprehensive features including EMD
    println!("🧮 Extracting EMD features...");
    let features = extract_features(&ts, &options)?;
    let emd_features = &features.frequency_features.emd_features;

    // Display basic EMD decomposition results
    println!("\n=== EMD Decomposition Results ===");
    println!("📋 Number of extracted IMFs: {}", emd_features.num_imfs);

    if !emd_features.imf_energies.is_empty() {
        println!("\n🔋 IMF Energy Distribution:");
        for (i, &energy) in emd_features.imf_energies.iter().enumerate() {
            let relative_energy = if i < emd_features.imf_relative_energies.len() {
                emd_features.imf_relative_energies[i]
            } else {
                0.0
            };
            println!(
                "   IMF {}: Energy = {:.4}, Relative = {:.2}%",
                i + 1,
                energy,
                relative_energy * 100.0
            );
        }
    }

    // Display frequency characteristics
    if !emd_features.imf_mean_frequencies.is_empty() {
        println!("\n📊 IMF Frequency Characteristics:");
        for (i, &freq) in emd_features.imf_mean_frequencies.iter().enumerate() {
            let bandwidth = if i < emd_features.imf_frequency_bandwidths.len() {
                emd_features.imf_frequency_bandwidths[i]
            } else {
                0.0
            };
            let complexity = if i < emd_features.imf_complexities.len() {
                emd_features.imf_complexities[i]
            } else {
                0.0
            };
            println!(
                "   IMF {}: Freq = {:.4}, Bandwidth = {:.4}, Complexity = {:.4}",
                i + 1,
                freq,
                bandwidth,
                complexity
            );
        }
    }

    // Display orthogonality and residue analysis
    println!("\n📈 EMD Quality Metrics:");
    println!(
        "   Orthogonality index: {:.4}",
        emd_features.orthogonality_index
    );
    println!("   (Higher values indicate better orthogonality between IMFs)");

    println!("\n🔍 Residue (Trend) Analysis:");
    let residue = &emd_features.residue_features;
    println!("   Mean: {:.4}", residue.mean);
    println!("   Trend slope: {:.6}", residue.trend_slope);
    println!("   Variance: {:.4}", residue.variance);
    println!("   Monotonicity: {:.4}", residue.monotonicity);
    println!("   Smoothness index: {:.4}", residue.smoothness_index);

    // Display Hilbert spectral analysis
    let hilbert = &emd_features.hilbert_spectral_features;
    if !hilbert.marginal_spectrum.is_empty() {
        println!("\n🌈 Hilbert Spectral Analysis:");
        println!(
            "   Marginal spectrum entries: {}",
            hilbert.marginal_spectrum.len()
        );
        println!(
            "   Hilbert spectral entropy: {:.4}",
            hilbert.hilbert_spectral_entropy
        );
        println!(
            "   Non-stationarity index: {:.4}",
            hilbert.nonstationarity_index
        );
        println!(
            "   Frequency resolution: {:.6}",
            hilbert.frequency_resolution
        );

        if !hilbert.instantaneous_energy.is_empty() {
            let avg_inst_energy: f64 = hilbert.instantaneous_energy.iter().sum::<f64>()
                / hilbert.instantaneous_energy.len() as f64;
            println!("   Average instantaneous energy: {:.4}", avg_inst_energy);
        }
    }

    // Display instantaneous characteristics
    let instant = &emd_features.instantaneous_features;
    println!("\n⚡ Instantaneous Features:");
    println!(
        "   Mean instantaneous frequency: {:.4}",
        instant.mean_instantaneous_freq
    );
    println!(
        "   Instantaneous freq. variance: {:.4}",
        instant.instantaneous_freq_variance
    );
    println!(
        "   Mean instantaneous amplitude: {:.4}",
        instant.mean_instantaneous_amplitude
    );
    println!(
        "   Instantaneous amp. variance: {:.4}",
        instant.instantaneous_amplitude_variance
    );
    println!(
        "   Frequency modulation index: {:.4}",
        instant.frequency_modulation_index
    );
    println!(
        "   Amplitude modulation index: {:.4}",
        instant.amplitude_modulation_index
    );

    // Display phase characteristics
    let phase = &instant.phase_features;
    println!("\n🌀 Phase Analysis:");
    println!("   Phase coherence: {:.4}", phase.phase_coherence);
    println!("   Phase coupling: {:.4}", phase.phase_coupling);
    println!("   Phase synchrony: {:.4}", phase.phase_synchrony);
    println!("   Phase entropy: {:.4}", phase.phase_entropy);

    // Display entropy-based features if calculated
    let entropy = &emd_features.emd_entropy_features;
    if !entropy.multiscale_entropy.is_empty() {
        println!("\n🔀 EMD-based Entropy Features:");
        println!(
            "   Multiscale entropy entries: {}",
            entropy.multiscale_entropy.len()
        );
        println!("   Composite entropy: {:.4}", entropy.composite_entropy);

        if !entropy.imf_permutation_entropies.is_empty() {
            let avg_perm_entropy: f64 = entropy.imf_permutation_entropies.iter().sum::<f64>()
                / entropy.imf_permutation_entropies.len() as f64;
            println!("   Average permutation entropy: {:.4}", avg_perm_entropy);
        }

        if !entropy.imf_sample_entropies.is_empty() {
            let avg_sample_entropy: f64 = entropy.imf_sample_entropies.iter().sum::<f64>()
                / entropy.imf_sample_entropies.len() as f64;
            println!("   Average sample entropy: {:.4}", avg_sample_entropy);
        }

        if !entropy.imf_cross_entropies.is_empty() {
            let avg_cross_entropy: f64 = entropy.imf_cross_entropies.iter().sum::<f64>()
                / entropy.imf_cross_entropies.len() as f64;
            println!("   Average cross-entropy: {:.4}", avg_cross_entropy);
        }
    }

    // Demonstrate different EMD configurations
    println!("\n=== Different EMD Configuration Demo ===");

    // Fast EMD configuration (fewer IMFs, basic analysis)
    let fast_config = EMDConfig {
        max_imfs: 4,
        sifting_tolerance: 0.5,
        max_sifting_iterations: 50,
        calculate_hilbert_spectrum: false,
        calculate_instantaneous: false,
        calculate_emd_entropies: false,
        interpolation_method: InterpolationMethod::Linear,
        edge_method: EdgeMethod::ZeroPadding,
    };

    let mut fast_options = FeatureExtractionOptions::default();
    fast_options.calculate_emd_features = true;
    fast_options.emd_config = Some(fast_config);

    println!("🚀 Fast EMD Configuration:");
    println!("   - Maximum IMFs: 4");
    println!("   - Simplified analysis (no Hilbert spectrum or entropies)");

    let fast_features = extract_features(&ts, &fast_options)?;
    let fast_emd = &fast_features.frequency_features.emd_features;

    println!("   - Extracted IMFs: {}", fast_emd.num_imfs);
    println!(
        "   - Orthogonality index: {:.4}",
        fast_emd.orthogonality_index
    );

    // Performance comparison
    println!("\n=== Configuration Comparison ===");
    println!("📊 Full EMD vs Fast EMD:");
    println!(
        "   Full: {} IMFs, Orthogonality: {:.4}",
        emd_features.num_imfs, emd_features.orthogonality_index
    );
    println!(
        "   Fast: {} IMFs, Orthogonality: {:.4}",
        fast_emd.num_imfs, fast_emd.orthogonality_index
    );

    // Application insights
    println!("\n=== Practical Applications ===");
    println!("💡 EMD features are particularly useful for:");
    println!("   • Non-stationary signal analysis");
    println!("   • Multi-scale decomposition of complex signals");
    println!("   • Identification of intrinsic oscillatory modes");
    println!("   • Adaptive filtering and denoising");
    println!("   • Instantaneous frequency analysis");
    println!("   • Biomedical signal processing (ECG, EEG, etc.)");
    println!("   • Financial time series analysis");
    println!("   • Geophysical and climate data analysis");

    println!("\n🔧 Configuration Guidelines:");
    println!("   • Use more IMFs for complex, multi-component signals");
    println!("   • Lower sifting tolerance for better convergence");
    println!("   • Enable Hilbert spectrum for frequency-time analysis");
    println!("   • Enable entropies for complexity characterization");
    println!("   • Use cubic spline interpolation for smoother envelopes");

    println!("\n✅ EMD feature extraction demo completed successfully!");
    println!(
        "   Extracted {} unique EMD-based features",
        if emd_features.num_imfs > 0 {
            8 + emd_features.num_imfs * 4 // Rough count of feature dimensions
        } else {
            0
        }
    );

    Ok(())
}
