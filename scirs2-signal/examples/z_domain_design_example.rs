//! Example demonstrating Z-domain filter design methods
//!
//! This example shows how to use the new Z-domain filter design functions
//! to create digital filters directly in the Z-domain.

use scirs2_signal::filter::{
    butter_bandpass_bandstop, z_domain_chebyshev1, z_domain_iir_design, FilterType,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Z-Domain Filter Design Examples");
    println!("===============================\n");

    // Example 1: Bandpass Butterworth filter
    println!("1. Bandpass Butterworth Filter Design");
    let order = 4;
    let low_freq = 0.1;
    let high_freq = 0.4;
    let (b_bp, a_bp) = butter_bandpass_bandstop(order, low_freq, high_freq, FilterType::Bandpass)?;

    println!("   Order: {}", order);
    println!("   Low cutoff: {:.2}", low_freq);
    println!("   High cutoff: {:.2}", high_freq);
    println!(
        "   Numerator coefficients: {:?}",
        &b_bp[..3.min(b_bp.len())]
    );
    println!(
        "   Denominator coefficients: {:?}",
        &a_bp[..3.min(a_bp.len())]
    );
    println!(
        "   Total coefficients: {} num, {} den\n",
        b_bp.len(),
        a_bp.len()
    );

    // Example 2: Bandstop Butterworth filter
    println!("2. Bandstop Butterworth Filter Design");
    let (b_bs, a_bs) = butter_bandpass_bandstop(order, low_freq, high_freq, FilterType::Bandstop)?;

    println!("   Order: {}", order);
    println!("   Low cutoff: {:.2}", low_freq);
    println!("   High cutoff: {:.2}", high_freq);
    println!(
        "   Numerator coefficients: {:?}",
        &b_bs[..3.min(b_bs.len())]
    );
    println!(
        "   Denominator coefficients: {:?}",
        &a_bs[..3.min(a_bs.len())]
    );
    println!(
        "   Total coefficients: {} num, {} den\n",
        b_bs.len(),
        a_bs.len()
    );

    // Example 3: Direct Z-domain Chebyshev Type I filter
    println!("3. Z-Domain Chebyshev Type I Filter Design");
    let cheby_order = 3;
    let ripple_db = 1.0;
    let cutoff = 0.3;
    let (b_cheby, a_cheby) =
        z_domain_chebyshev1(cheby_order, ripple_db, cutoff, FilterType::Lowpass)?;

    println!("   Order: {}", cheby_order);
    println!("   Ripple: {:.1} dB", ripple_db);
    println!("   Cutoff: {:.2}", cutoff);
    println!("   Numerator coefficients: {:?}", b_cheby);
    println!("   Denominator coefficients: {:?}", a_cheby);
    println!();

    // Example 4: Direct Z-domain IIR optimization design
    println!("4. Direct Z-Domain IIR Optimization Design");
    let frequencies = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let desired = vec![1.0, 1.0, 1.0, 0.7, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001];
    let weights = vec![1.0; frequencies.len()];

    let opt_order = 3;
    let (b_opt, a_opt) = z_domain_iir_design(opt_order, &desired, &frequencies, Some(&weights))?;

    println!("   Order: {}", opt_order);
    println!("   Frequency points: {}", frequencies.len());
    println!("   Numerator coefficients: {:?}", b_opt);
    println!("   Denominator coefficients: {:?}", a_opt);
    println!();

    // Demonstrate filter evaluation at a test frequency
    println!("5. Filter Response Evaluation");
    let test_freq = 0.25; // Normalized frequency
    let omega = std::f64::consts::PI * test_freq;
    let z = num_complex::Complex64::new(omega.cos(), omega.sin());

    // Evaluate Chebyshev filter response
    let mut h_num = num_complex::Complex64::new(0.0, 0.0);
    let mut h_den = num_complex::Complex64::new(0.0, 0.0);

    for (i, &coeff) in b_cheby.iter().enumerate() {
        h_num += coeff * z.powf(-(i as f64));
    }
    for (i, &coeff) in a_cheby.iter().enumerate() {
        h_den += coeff * z.powf(-(i as f64));
    }

    let response = h_num / h_den;
    let magnitude_db = 20.0 * response.norm().log10();

    println!("   Test frequency: {:.2}", test_freq);
    println!("   Chebyshev filter magnitude: {:.2} dB", magnitude_db);
    println!("   Chebyshev filter phase: {:.2} radians", response.arg());

    println!("\nZ-domain filter design examples completed successfully!");
    Ok(())
}
