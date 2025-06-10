//! Entropy Features Demo
//!
//! This example demonstrates comprehensive entropy-based feature extraction for time series data.
//! It shows how to extract various entropy measures including Shannon, Rényi, Tsallis entropies,
//! complexity measures, spectral entropies, and cross-scale entropy analysis.

use ndarray::Array1;
use scirs2_series::features::{extract_features, EntropyConfig, FeatureExtractionOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌀 Entropy Features Demo");
    println!("========================\n");

    // Create diverse synthetic time series with different entropy characteristics
    let n = 400;

    println!("🏗️  Generating synthetic time series with different entropy characteristics:");
    println!("   - Highly regular periodic signal (low entropy)");
    println!("   - Chaotic signal (medium entropy)");
    println!("   - Random noise (high entropy)");
    println!("   - Mixed signal with regime changes");
    println!("   - Fractal-like signal\n");

    // 1. Highly regular periodic signal (low entropy)
    println!("=== Regular Periodic Signal Analysis ===");
    let mut regular_ts = Array1::zeros(n);
    for i in 0..n {
        let t = i as f64;
        // Pure sinusoidal with harmonics
        regular_ts[i] = 2.0 * (t / 20.0).sin() + 0.5 * (t / 10.0).sin() + 0.1 * (t / 5.0).sin();
    }

    let mut regular_options = FeatureExtractionOptions::default();
    regular_options.calculate_entropy_features = true;

    let entropy_config = EntropyConfig {
        calculate_classical_entropy: true,
        calculate_differential_entropy: true,
        calculate_multiscale_entropy: true,
        calculate_conditional_entropy: true,
        calculate_spectral_entropy: true,
        calculate_timefrequency_entropy: true,
        calculate_symbolic_entropy: true,
        calculate_distribution_entropy: true,
        calculate_complexity_measures: true,
        calculate_fractal_entropy: true,
        calculate_crossscale_entropy: true,
        n_bins: 10,
        embedding_dimension: 3,
        tolerance_fraction: 0.2,
        permutation_order: 3,
        max_lag: 5,
        n_scales: 5,
        renyi_alpha: 2.0,
        tsallis_q: 2.0,
        n_symbols: 3,
        use_fast_approximations: false,
        instantaneous_window_size: 50,
        instantaneous_overlap: 0.5,
    };
    regular_options.entropy_config = Some(entropy_config.clone());

    let regular_features = extract_features(&regular_ts, &regular_options)?;
    let regular_entropy = &regular_features.entropy_features;

    display_entropy_features("Regular Periodic", regular_entropy);

    // 2. Chaotic signal (Logistic map)
    println!("\n=== Chaotic Signal Analysis ===");
    let mut chaotic_ts = Array1::zeros(n);
    let mut x = 0.5; // Initial condition
    let r = 3.8; // Chaotic regime
    for i in 0..n {
        x = r * x * (1.0 - x); // Logistic map
        chaotic_ts[i] = x;
    }

    let chaotic_features = extract_features(&chaotic_ts, &regular_options)?;
    let chaotic_entropy = &chaotic_features.entropy_features;

    display_entropy_features("Chaotic (Logistic)", chaotic_entropy);

    // 3. Random noise (high entropy)
    println!("\n=== Random Noise Analysis ===");
    let mut random_ts = Array1::zeros(n);
    for i in 0..n {
        random_ts[i] = rand::random::<f64>() - 0.5;
    }

    let random_features = extract_features(&random_ts, &regular_options)?;
    let random_entropy = &random_features.entropy_features;

    display_entropy_features("Random Noise", random_entropy);

    // 4. Mixed signal with regime changes
    println!("\n=== Mixed Signal with Regime Changes ===");
    let mut mixed_ts = Array1::zeros(n);
    for i in 0..n {
        let t = i as f64;
        if i < n / 3 {
            // Regular oscillation
            mixed_ts[i] = (t / 15.0).sin();
        } else if i < 2 * n / 3 {
            // Chaotic behavior
            let x = ((t - n as f64 / 3.0) / 100.0) % 1.0;
            mixed_ts[i] = 3.9 * x * (1.0 - x);
        } else {
            // Random walk
            mixed_ts[i] =
                if i > 0 { mixed_ts[i - 1] } else { 0.0 } + 0.1 * (rand::random::<f64>() - 0.5);
        }
    }

    let mixed_features = extract_features(&mixed_ts, &regular_options)?;
    let mixed_entropy = &mixed_features.entropy_features;

    display_entropy_features("Mixed Regime", mixed_entropy);

    // 5. Fractal-like signal (Brownian motion)
    println!("\n=== Fractal-like Signal Analysis ===");
    let mut fractal_ts = Array1::zeros(n);
    fractal_ts[0] = 0.0;
    for i in 1..n {
        // Fractional Brownian motion approximation
        let increment = (rand::random::<f64>() - 0.5) * (i as f64).powf(-0.3);
        fractal_ts[i] = fractal_ts[i - 1] + increment;
    }

    let fractal_features = extract_features(&fractal_ts, &regular_options)?;
    let fractal_entropy = &fractal_features.entropy_features;

    display_entropy_features("Fractal-like", fractal_entropy);

    // Comparative analysis
    println!("\n=== Comparative Entropy Analysis ===");
    compare_entropy_characteristics(&[
        ("Regular", regular_entropy),
        ("Chaotic", chaotic_entropy),
        ("Random", random_entropy),
        ("Mixed", mixed_entropy),
        ("Fractal", fractal_entropy),
    ]);

    // Configuration demonstration
    println!("\n=== Configuration Options Demo ===");
    demonstrate_entropy_configurations(&regular_ts)?;

    // Multiscale analysis demonstration
    println!("\n=== Multiscale Entropy Analysis ===");
    demonstrate_multiscale_analysis(&chaotic_ts)?;

    // Applications and insights
    println!("\n=== Practical Applications & Insights ===");
    println!("💡 Entropy Feature Categories & Applications:");
    println!("   • Basic entropies: Overall signal predictability and randomness");
    println!("   • Differential entropies: Local complexity and regularity patterns");
    println!("   • Conditional entropies: Temporal dependencies and memory effects");
    println!("   • Spectral entropies: Frequency domain complexity and distribution");
    println!("   • Complexity entropies: Algorithmic complexity and compressibility");
    println!("   • Fractal entropies: Self-similarity and scaling properties");
    println!("   • Cross-scale entropies: Multi-resolution complexity analysis");

    println!("\n🔧 Configuration Guidelines:");
    println!("   • Higher permutation orders capture more complex patterns");
    println!("   • Adjust tolerance parameters based on signal noise level");
    println!("   • Use multiscale analysis for signals with multiple timescales");
    println!("   • Enable normalization for comparative analysis");
    println!("   • Increase conditional bins for higher resolution dependency analysis");

    println!("\n📊 Interpretation Tips:");
    println!("   • Low Shannon entropy → highly predictable, regular patterns");
    println!("   • High Rényi entropy → presence of extreme values or outliers");
    println!("   • Low permutation entropy → deterministic dynamics");
    println!("   • High spectral entropy → broad frequency distribution");
    println!("   • Low Lempel-Ziv complexity → repetitive patterns");
    println!("   • High differential entropy → irregular temporal structure");

    println!("\n⚠️  Analysis Considerations:");
    println!("   • Entropy measures are sensitive to signal length");
    println!("   • Some measures require sufficient data for reliable estimation");
    println!("   • Normalization helps with cross-signal comparisons");
    println!("   • Parameter selection affects sensitivity and specificity");
    println!("   • Consider computational cost for real-time applications");

    println!("\n🎯 Domain-Specific Applications:");
    println!("   • Biomedical: EEG/ECG complexity analysis, disease detection");
    println!("   • Finance: Market efficiency, volatility clustering analysis");
    println!("   • Engineering: System health monitoring, fault detection");
    println!("   • Climate: Weather pattern complexity, extreme event analysis");
    println!("   • Communications: Signal quality, information content analysis");

    println!("\n✅ Entropy features demo completed successfully!");
    println!("   Comprehensive entropy characterization provides deep insights");
    println!("   into signal complexity, predictability, and information content.");

    Ok(())
}

fn display_entropy_features(name: &str, features: &scirs2_series::features::EntropyFeatures<f64>) {
    println!("🌀 {} Signal Entropy Features:", name);

    // Basic entropies
    println!("   Basic Information Entropies:");
    println!("     Shannon entropy: {:.4}", features.shannon_entropy);
    println!("     Rényi entropy (α=2): {:.4}", features.renyi_entropy_2);
    println!(
        "     Tsallis entropy (q=2): {:.4}",
        features.tsallis_entropy
    );
    println!(
        "     Permutation entropy: {:.4}",
        features.permutation_entropy
    );

    // Differential entropies
    println!("   Differential & Temporal Entropies:");
    println!(
        "     Weighted permutation entropy: {:.4}",
        features.weighted_permutation_entropy
    );
    if !features.multiscale_entropy.is_empty() {
        let avg_multiscale = features.multiscale_entropy.iter().sum::<f64>()
            / features.multiscale_entropy.len() as f64;
        println!("     Average multiscale entropy: {:.4}", avg_multiscale);
    }

    // Conditional entropies
    println!("   Conditional & Dependency Entropies:");
    println!(
        "     Conditional entropy: {:.4}",
        features.conditional_entropy
    );
    println!("     Excess entropy: {:.4}", features.excess_entropy);
    println!(
        "     Mutual information: {:.4}",
        features.mutual_information
    );
    println!("     Transfer entropy: {:.4}", features.transfer_entropy);

    // Spectral entropies
    println!("   Spectral & Frequency Entropies:");
    println!("     Spectral entropy: {:.4}", features.spectral_entropy);
    println!("     Wavelet entropy: {:.4}", features.wavelet_entropy);
    println!(
        "     Normalized spectral entropy: {:.4}",
        features.normalized_spectral_entropy
    );

    // Complexity entropies
    println!("   Complexity & Regularity Entropies:");
    println!("     Sample entropy: {:.4}", features.sample_entropy);
    println!(
        "     Approximate entropy: {:.4}",
        features.approximate_entropy
    );
    println!(
        "     Lempel-Ziv complexity: {:.4}",
        features.lempel_ziv_complexity
    );
    println!(
        "     Effective complexity: {:.4}",
        features.effective_complexity
    );

    // Fractal entropies
    println!("   Fractal & Scaling Entropies:");
    println!("     Fractal entropy: {:.4}", features.fractal_entropy);
    println!("     DFA entropy: {:.4}", features.dfa_entropy);
    println!(
        "     Multifractal entropy width: {:.4}",
        features.multifractal_entropy_width
    );

    // Cross-scale entropies
    println!("   Cross-scale & Multi-resolution Entropies:");
    if !features.cross_scale_entropy.is_empty() {
        let avg_cross_scale = features.cross_scale_entropy.iter().sum::<f64>()
            / features.cross_scale_entropy.len() as f64;
        println!("     Average cross-scale entropy: {:.4}", avg_cross_scale);
    }
    if !features.hierarchical_entropy.is_empty() {
        let avg_hierarchical = features.hierarchical_entropy.iter().sum::<f64>()
            / features.hierarchical_entropy.len() as f64;
        println!("     Average hierarchical entropy: {:.4}", avg_hierarchical);
    }
    println!(
        "     Scale entropy ratio: {:.4}",
        features.scale_entropy_ratio
    );
}

fn compare_entropy_characteristics(
    signals: &[(&str, &scirs2_series::features::EntropyFeatures<f64>)],
) {
    println!("🔍 Entropy Comparison Across Signals:");

    // Compare basic entropy measures
    println!("   Shannon Entropy (Predictability):");
    for (name, features) in signals {
        println!("     {}: {:.4}", name, features.shannon_entropy);
    }

    println!("   Permutation Entropy (Ordinal Patterns):");
    for (name, features) in signals {
        println!("     {}: {:.4}", name, features.permutation_entropy);
    }

    println!("   Sample Entropy (Regularity):");
    for (name, features) in signals {
        println!("     {}: {:.4}", name, features.sample_entropy);
    }

    println!("   Lempel-Ziv Complexity (Algorithmic Complexity):");
    for (name, features) in signals {
        println!("     {}: {:.4}", name, features.lempel_ziv_complexity);
    }

    println!("   Spectral Entropy (Frequency Dispersion):");
    for (name, features) in signals {
        println!("     {}: {:.4}", name, features.spectral_entropy);
    }

    // Identify characteristics
    println!("\n📊 Signal Characteristics Summary:");
    for (name, features) in signals {
        let complexity_level = if features.shannon_entropy < 0.3 {
            "Very Low"
        } else if features.shannon_entropy < 0.6 {
            "Low"
        } else if features.shannon_entropy < 0.8 {
            "Medium"
        } else {
            "High"
        };

        let predictability = if features.sample_entropy < 0.2 {
            "Highly Predictable"
        } else if features.sample_entropy < 0.5 {
            "Moderately Predictable"
        } else {
            "Unpredictable"
        };

        println!(
            "     {}: Complexity = {}, Predictability = {}",
            name, complexity_level, predictability
        );
    }
}

fn demonstrate_entropy_configurations(ts: &Array1<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Entropy Configuration Options:");

    // Basic configuration (fast)
    println!("   1. Basic entropy configuration (fast):");
    let mut basic_options = FeatureExtractionOptions::default();
    basic_options.calculate_entropy_features = true;

    let basic_config = EntropyConfig {
        calculate_classical_entropy: true,
        calculate_differential_entropy: false,
        calculate_conditional_entropy: false,
        calculate_spectral_entropy: true,
        calculate_complexity_measures: true,
        calculate_fractal_entropy: false,
        calculate_crossscale_entropy: false,
        permutation_order: 3,
        n_scales: 3,
        ..Default::default()
    };
    basic_options.entropy_config = Some(basic_config);

    let basic_features = extract_features(ts, &basic_options)?;
    println!(
        "     Shannon entropy: {:.4}",
        basic_features.entropy_features.shannon_entropy
    );
    println!(
        "     Sample entropy: {:.4}",
        basic_features.entropy_features.sample_entropy
    );

    // Comprehensive configuration (detailed)
    println!("   2. Comprehensive entropy configuration (detailed):");
    let mut comprehensive_options = FeatureExtractionOptions::default();
    comprehensive_options.calculate_entropy_features = true;

    let comprehensive_config = EntropyConfig {
        calculate_classical_entropy: true,
        calculate_differential_entropy: true,
        calculate_multiscale_entropy: true,
        calculate_conditional_entropy: true,
        calculate_spectral_entropy: true,
        calculate_timefrequency_entropy: true,
        calculate_symbolic_entropy: true,
        calculate_distribution_entropy: true,
        calculate_complexity_measures: true,
        calculate_fractal_entropy: true,
        calculate_crossscale_entropy: true,
        permutation_order: 5,     // Higher order for more detail
        n_bins: 20,               // Higher resolution
        n_scales: 8,              // More scales
        tolerance_fraction: 0.15, // Tighter tolerance
        ..Default::default()
    };
    comprehensive_options.entropy_config = Some(comprehensive_config);

    let comprehensive_features = extract_features(ts, &comprehensive_options)?;
    println!(
        "     Shannon entropy: {:.4}",
        comprehensive_features.entropy_features.shannon_entropy
    );
    println!(
        "     Cross-scale entropy: {:.4}",
        comprehensive_features
            .entropy_features
            .cross_scale_entropy
            .iter()
            .sum::<f64>()
            / comprehensive_features
                .entropy_features
                .cross_scale_entropy
                .len() as f64
    );
    println!(
        "     Fractal entropy: {:.4}",
        comprehensive_features.entropy_features.fractal_entropy
    );

    Ok(())
}

fn demonstrate_multiscale_analysis(ts: &Array1<f64>) -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Multiscale Entropy Analysis:");

    let mut options = FeatureExtractionOptions::default();
    options.calculate_entropy_features = true;

    let config = EntropyConfig {
        calculate_differential_entropy: true,
        calculate_multiscale_entropy: true,
        n_scales: 10,
        ..Default::default()
    };
    options.entropy_config = Some(config);

    let features = extract_features(ts, &options)?;
    let entropy_features = &features.entropy_features;

    if !entropy_features.multiscale_entropy.is_empty() {
        println!("   Multiscale Entropy by Scale:");
        for (scale, &entropy_value) in entropy_features.multiscale_entropy.iter().enumerate() {
            println!("     Scale {}: {:.4}", scale + 1, entropy_value);
        }

        // Analyze trend
        let first_half_avg = entropy_features.multiscale_entropy
            [0..entropy_features.multiscale_entropy.len() / 2]
            .iter()
            .sum::<f64>()
            / (entropy_features.multiscale_entropy.len() / 2) as f64;
        let second_half_avg = entropy_features.multiscale_entropy
            [entropy_features.multiscale_entropy.len() / 2..]
            .iter()
            .sum::<f64>()
            / (entropy_features.multiscale_entropy.len()
                - entropy_features.multiscale_entropy.len() / 2) as f64;

        println!("\n   Multiscale Analysis:");
        println!(
            "     Fine scales (1-{}): {:.4}",
            entropy_features.multiscale_entropy.len() / 2,
            first_half_avg
        );
        println!(
            "     Coarse scales ({}-{}): {:.4}",
            entropy_features.multiscale_entropy.len() / 2 + 1,
            entropy_features.multiscale_entropy.len(),
            second_half_avg
        );

        if second_half_avg > first_half_avg {
            println!("     → Signal shows more complexity at coarse scales");
        } else {
            println!("     → Signal shows more complexity at fine scales");
        }
    }

    Ok(())
}
