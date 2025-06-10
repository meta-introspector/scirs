use ndarray::Array1;
use scirs2_series::features::{
    calculate_discord_scores, discover_motifs, extract_features, extract_shapelets,
    time_series_to_sax, FeatureExtractionOptions,
};

fn main() {
    println!("=== Temporal Pattern Mining Demo ===\n");

    // Create a time series with repetitive patterns and anomalies
    let ts: Array1<f64> = Array1::from_vec(
        (0..60)
            .map(|i| {
                let t = i as f64 / 10.0;
                let base_pattern = 2.0 * (2.0 * std::f64::consts::PI * t / 6.0).sin();
                let noise = 0.2 * (t * 1.5).sin();

                // Add some anomalies
                if i == 25 || i == 45 {
                    base_pattern + 5.0 + noise // Strong anomaly
                } else if i >= 30 && i <= 35 {
                    base_pattern + 2.0 + noise // Moderate anomaly region
                } else {
                    base_pattern + noise
                }
            })
            .collect(),
    );

    println!("Time series length: {}", ts.len());
    println!(
        "Sample values: {:?}",
        &ts.slice(ndarray::s![0..10]).to_vec()
    );

    // 1. Motif Discovery
    println!("\n1. MOTIF DISCOVERY");
    println!("==================");

    match discover_motifs(&ts, 6, 3) {
        Ok(motifs) => {
            println!("Found {} motifs:", motifs.len());
            for (i, motif) in motifs.iter().enumerate() {
                println!(
                    "  Motif {}: length={}, frequency={}, avg_distance={:.4}",
                    i + 1,
                    motif.length,
                    motif.frequency,
                    motif.avg_distance
                );
                println!("    Positions: {:?}", motif.positions);
            }
        }
        Err(e) => println!("Error in motif discovery: {:?}", e),
    }

    // 2. Discord Detection
    println!("\n2. DISCORD DETECTION");
    println!("====================");

    match calculate_discord_scores(&ts, 5, 3) {
        Ok(discord_scores) => {
            println!(
                "Discord scores calculated for {} positions",
                discord_scores.len()
            );

            // Find top 5 discord positions
            let mut indexed_scores: Vec<(usize, f64)> = discord_scores
                .iter()
                .enumerate()
                .map(|(i, &score)| (i, score))
                .collect();
            indexed_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            println!("Top 5 discord positions:");
            for (i, (pos, score)) in indexed_scores.iter().take(5).enumerate() {
                println!("  {}: Position {} with score {:.4}", i + 1, pos, score);
            }
        }
        Err(e) => println!("Error in discord detection: {:?}", e),
    }

    // 3. SAX (Symbolic Aggregate approXimation)
    println!("\n3. SAX REPRESENTATION");
    println!("=====================");

    match time_series_to_sax(&ts, 12, 4) {
        Ok(sax_symbols) => {
            println!("SAX representation (word_length=12, alphabet_size=4):");
            let sax_string: String = sax_symbols.iter().collect();
            println!("  SAX string: {}", sax_string);

            // Show symbol frequencies
            let mut symbol_counts = std::collections::HashMap::new();
            for &symbol in &sax_symbols {
                *symbol_counts.entry(symbol).or_insert(0) += 1;
            }
            println!("  Symbol frequencies: {:?}", symbol_counts);
        }
        Err(e) => println!("Error in SAX conversion: {:?}", e),
    }

    // 4. Shapelet Extraction (requires two classes)
    println!("\n4. SHAPELET EXTRACTION");
    println!("======================");

    // Create two classes of time series for demonstration
    let ts_class1 = vec![
        // Normal patterns
        ts.slice(ndarray::s![0..20]).to_owned(),
        ts.slice(ndarray::s![10..30]).to_owned(),
        ts.slice(ndarray::s![40..60]).to_owned(),
    ];

    let ts_class2 = vec![
        // Anomalous patterns (shifted windows around anomalies)
        ts.slice(ndarray::s![20..40]).to_owned(),
        ts.slice(ndarray::s![25..45]).to_owned(),
    ];

    match extract_shapelets(&ts_class1, &ts_class2, 3, 8, 5) {
        Ok(shapelets) => {
            println!("Found {} shapelets:", shapelets.len());
            for (i, shapelet) in shapelets.iter().enumerate() {
                println!(
                    "  Shapelet {}: length={}, info_gain={:.4}, position={}",
                    i + 1,
                    shapelet.length,
                    shapelet.information_gain,
                    shapelet.position
                );
                println!(
                    "    Pattern: {:?}",
                    shapelet
                        .pattern
                        .slice(ndarray::s![..shapelet.length.min(5)])
                        .to_vec()
                );
            }
        }
        Err(e) => println!("Error in shapelet extraction: {:?}", e),
    }

    // 5. Comprehensive Feature Extraction
    println!("\n5. COMPREHENSIVE FEATURE EXTRACTION");
    println!("===================================");

    let mut options = FeatureExtractionOptions::default();
    options.calculate_complexity = true;
    options.calculate_frequency_features = true;
    options.detect_temporal_patterns = true;
    options.motif_length = Some(6);

    match extract_features(&ts, &options) {
        Ok(features) => {
            println!("Basic Statistics:");
            println!("  Mean: {:.4}", features.mean);
            println!("  Std Dev: {:.4}", features.std_dev);
            println!("  Skewness: {:.4}", features.skewness);
            println!("  Kurtosis: {:.4}", features.kurtosis);
            println!("  Trend Strength: {:.4}", features.trend_strength);

            println!("\nComplexity Features:");
            println!(
                "  Approximate Entropy: {:.4}",
                features.complexity_features.approximate_entropy
            );
            println!(
                "  Sample Entropy: {:.4}",
                features.complexity_features.sample_entropy
            );
            println!(
                "  Permutation Entropy: {:.4}",
                features.complexity_features.permutation_entropy
            );
            println!(
                "  Lempel-Ziv Complexity: {:.4}",
                features.complexity_features.lempel_ziv_complexity
            );
            println!(
                "  Fractal Dimension: {:.4}",
                features.complexity_features.fractal_dimension
            );
            println!(
                "  Hurst Exponent: {:.4}",
                features.complexity_features.hurst_exponent
            );
            println!(
                "  DFA Exponent: {:.4}",
                features.complexity_features.dfa_exponent
            );
            println!(
                "  Turning Points: {}",
                features.complexity_features.turning_points
            );
            println!(
                "  Longest Strike: {}",
                features.complexity_features.longest_strike
            );

            println!("\nFrequency Domain Features:");
            println!(
                "  Spectral Centroid: {:.4}",
                features.frequency_features.spectral_centroid
            );
            println!(
                "  Spectral Spread: {:.4}",
                features.frequency_features.spectral_spread
            );
            println!(
                "  Spectral Entropy: {:.4}",
                features.frequency_features.spectral_entropy
            );
            println!(
                "  Dominant Frequency: {:.4}",
                features.frequency_features.dominant_frequency
            );
            println!(
                "  Spectral Peaks: {}",
                features.frequency_features.spectral_peaks
            );

            println!("\nTemporal Pattern Features:");
            println!(
                "  Motifs found: {}",
                features.temporal_pattern_features.motifs.len()
            );
            println!(
                "  Discord scores: {} values",
                features.temporal_pattern_features.discord_scores.len()
            );
            println!(
                "  SAX symbols: {} symbols",
                features.temporal_pattern_features.sax_symbols.len()
            );
            println!(
                "  SAX string: {}",
                features
                    .temporal_pattern_features
                    .sax_symbols
                    .iter()
                    .collect::<String>()
            );

            if !features.temporal_pattern_features.motifs.is_empty() {
                let motif = &features.temporal_pattern_features.motifs[0];
                println!(
                    "  Best motif: frequency={}, avg_distance={:.4}",
                    motif.frequency, motif.avg_distance
                );
            }
        }
        Err(e) => println!("Error in feature extraction: {:?}", e),
    }

    println!("\n=== Demo Complete ===");
}
