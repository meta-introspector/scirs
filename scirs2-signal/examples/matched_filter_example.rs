//! Example of matched filter design and signal detection
//!
//! This example demonstrates matched filtering, which is optimal for detecting
//! known signals in additive white Gaussian noise. Applications include radar,
//! communications, pattern recognition, and correlation-based signal processing.

use scirs2_signal::filter::{
    correlate, detect_peaks, matched_filter, matched_filter_bank, matched_filter_bank_detect,
    matched_filter_detect,
};

fn main() {
    println!("Matched Filter Example");
    println!("=====================\n");

    // Example 1: Basic matched filter design
    println!("1. Basic Matched Filter Design");
    println!("-----------------------------");

    let template = vec![1.0, 2.0, 3.0, 2.0, 1.0];
    println!("Template signal: {:?}", template);

    // Design matched filter (unnormalized)
    let mf = matched_filter(&template, false).unwrap();
    println!("Matched filter (time-reversed): {:?}", mf);

    // Design normalized matched filter
    let mf_norm = matched_filter(&template, true).unwrap();
    println!("Normalized matched filter: {:?}", mf_norm);

    // Verify energy normalization
    let energy: f64 = mf_norm.iter().map(|&x| x * x).sum();
    println!("Normalized filter energy: {:.6}", energy);

    // Example 2: Template detection in a signal
    println!("\n\n2. Template Detection in Signal");
    println!("------------------------------");

    // Create a signal with the template embedded at different locations
    let mut signal = vec![0.1, -0.2, 0.15, 0.0]; // Noise at beginning
    signal.extend_from_slice(&template); // Template at position 4-8
    signal.extend_from_slice(&[0.05, -0.1, 0.2, 0.1]); // More noise
    signal.extend_from_slice(&template); // Template again at position 13-17
    signal.extend_from_slice(&[-0.05, 0.3, -0.1]); // Noise at end

    println!("Signal length: {}", signal.len());
    println!("Template embedded at positions: 4-8 and 13-17");

    // Apply matched filter detection
    let correlation = matched_filter_detect(&signal, &template, "same").unwrap();

    println!("\nCorrelation output (first 10 values):");
    for (i, &val) in correlation.iter().take(10).enumerate() {
        println!("  [{}]: {:8.4}", i, val);
    }

    // Find peaks in the correlation
    let peaks = detect_peaks(&correlation, 0.5, 3).unwrap();
    println!("\nDetected peaks (threshold = 0.5, min_distance = 3):");
    for (idx, val) in &peaks {
        println!("  Peak at index {}: value = {:.4}", idx, val);
    }

    // Example 3: Cross-correlation modes
    println!("\n\n3. Cross-Correlation Modes");
    println!("--------------------------");

    let signal_a = vec![1.0, 2.0, 3.0, 4.0];
    let signal_b = vec![1.0, 1.0];

    // Full correlation
    let corr_full = correlate(&signal_a, &signal_b, "full").unwrap();
    println!("Signal A: {:?}", signal_a);
    println!("Signal B: {:?}", signal_b);
    println!("Full correlation: {:?}", corr_full);

    // Same correlation
    let corr_same = correlate(&signal_a, &signal_b, "same").unwrap();
    println!("Same correlation: {:?}", corr_same);

    // Valid correlation
    let corr_valid = correlate(&signal_a, &signal_b, "valid").unwrap();
    println!("Valid correlation: {:?}", corr_valid);

    // Example 4: Noisy signal detection
    println!("\n\n4. Detection in Noisy Signal");
    println!("----------------------------");

    // Create a clean template
    let clean_template = vec![1.0, -1.0, 1.0, -1.0];
    println!("Clean template: {:?}", clean_template);

    // Create a noisy signal with the template buried in noise
    let noisy_signal = vec![
        0.2, -0.3, 0.15, 0.1, // Noise
        1.1, -0.9, 1.05, -1.1, // Template with noise (position 4-7)
        0.1, 0.2, -0.25, 0.05, // More noise
        0.9, -1.15, 0.95, -0.85, // Template with noise (position 12-15)
        -0.1, 0.3, 0.0, // Final noise
    ];

    println!("Noisy signal length: {}", noisy_signal.len());

    // Detect the template in noise
    let noisy_correlation = matched_filter_detect(&noisy_signal, &clean_template, "same").unwrap();

    // Find significant peaks
    let noisy_peaks = detect_peaks(&noisy_correlation, 0.3, 2).unwrap();
    println!("Detected templates in noisy signal:");
    for (idx, val) in &noisy_peaks {
        println!(
            "  Template detected at index {}: correlation = {:.4}",
            idx, val
        );
    }

    // Calculate SNR improvement
    let template_energy: f64 = clean_template.iter().map(|&x| x * x).sum();
    let snr_improvement = 10.0 * template_energy.log10();
    println!("Theoretical SNR improvement: {:.2} dB", snr_improvement);

    // Example 5: Multiple template detection (filter bank)
    println!("\n\n5. Multiple Template Detection (Filter Bank)");
    println!("-------------------------------------------");

    // Define multiple templates
    let templates = vec![
        vec![1.0, 1.0, 1.0],  // Rectangle
        vec![1.0, 2.0, 1.0],  // Triangle
        vec![1.0, -1.0, 1.0], // Alternating
        vec![1.0, 0.0, 1.0],  // Spaced pulse
    ];

    println!("Template bank:");
    for (i, template) in templates.iter().enumerate() {
        println!("  Template {}: {:?}", i, template);
    }

    // Create filter bank
    let filter_bank = matched_filter_bank(&templates, true).unwrap();
    println!("Created filter bank with {} filters", filter_bank.len());

    // Create a test signal with different templates
    let test_signal = vec![
        0.1, 0.0, -0.1, // Noise
        1.0, 1.0, 1.0, // Template 0 (rectangle)
        0.0, 0.1, -0.05, // Noise
        1.0, 2.0, 1.0, // Template 1 (triangle)
        0.05, 0.0, // Noise
        1.0, -1.0, 1.0, // Template 2 (alternating)
        -0.1, 0.2, // Final noise
    ];

    println!("Test signal length: {}", test_signal.len());

    // Apply filter bank
    let bank_results = matched_filter_bank_detect(&test_signal, &filter_bank, "same").unwrap();

    println!("\nFilter bank detection results:");
    for (filter_idx, correlation) in bank_results.iter().enumerate() {
        let peaks = detect_peaks(correlation, 0.3, 2).unwrap();
        println!(
            "  Filter {} (template {:?}):",
            filter_idx, templates[filter_idx]
        );
        if peaks.is_empty() {
            println!("    No significant detections");
        } else {
            for (peak_idx, peak_val) in &peaks {
                println!("    Detection at index {}: {:.4}", peak_idx, peak_val);
            }
        }
    }

    // Example 6: Pulse compression (chirp matching)
    println!("\n\n6. Pulse Compression Example");
    println!("---------------------------");

    // Create a simple linear chirp-like signal
    let chirp_length = 8;
    let chirp_template: Vec<f64> = (0..chirp_length)
        .map(|i| (2.0 * std::f64::consts::PI * i as f64 * i as f64 / chirp_length as f64).sin())
        .collect();

    println!("Chirp template (length {}):", chirp_length);
    for (i, &val) in chirp_template.iter().enumerate() {
        println!("  [{}]: {:6.3}", i, val);
    }

    // Create a signal with the chirp at a specific location
    let mut chirp_signal = vec![0.0; 20];
    for (i, &val) in chirp_template.iter().enumerate() {
        chirp_signal[5 + i] = val; // Place chirp at position 5
    }

    // Add some noise
    for val in &mut chirp_signal {
        *val += 0.1 * (fastrand::f64() - 0.5); // Simple noise
    }

    println!(
        "Signal with embedded chirp (position 5-{}):",
        5 + chirp_length - 1
    );

    // Perform pulse compression
    let compressed = matched_filter_detect(&chirp_signal, &chirp_template, "same").unwrap();

    // Find the compressed pulse
    let chirp_peaks = detect_peaks(&compressed, 0.2, 1).unwrap();
    println!("Pulse compression results:");
    for (idx, val) in &chirp_peaks {
        println!(
            "  Compressed pulse at index {}: amplitude = {:.4}",
            idx, val
        );
    }

    // Calculate compression ratio
    let compression_ratio = chirp_length as f64;
    println!("Compression ratio: {:.1}:1", compression_ratio);

    // Example 7: Performance analysis
    println!("\n\n7. Performance Characteristics");
    println!("-----------------------------");

    println!("Matched filter properties:");
    println!("- Maximizes Signal-to-Noise Ratio (SNR) for known signals in AWGN");
    println!("- Provides optimal detection performance (Neyman-Pearson criterion)");
    println!("- SNR improvement = 10*log10(signal_energy) dB");
    println!("- Impulse response = time-reversed template");
    println!("- Output peak occurs when template aligns with signal");

    println!("\nApplications:");
    println!("- Radar pulse compression");
    println!("- Digital communications (symbol detection)");
    println!("- Sonar signal processing");
    println!("- Pattern recognition");
    println!("- Correlation-based measurements");

    println!("\nComputational complexity:");
    println!("- Direct correlation: O(N*M) where N=signal length, M=template length");
    println!("- FFT-based correlation: O(N*log(N)) for large signals");
    println!("- Filter bank: O(K*N*M) where K=number of templates");

    println!("\n\nMatched filtering provides optimal signal detection in noise,");
    println!("making it fundamental to many signal processing applications.");
}

// Simple random number generation for noise (avoiding external dependencies)
mod fastrand {
    use std::cell::Cell;

    thread_local! {
        static RNG: Cell<u64> = const { Cell::new(1) };
    }

    pub fn f64() -> f64 {
        RNG.with(|rng| {
            let mut x = rng.get();
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            rng.set(x);
            (x as f64) / (u64::MAX as f64)
        })
    }
}
