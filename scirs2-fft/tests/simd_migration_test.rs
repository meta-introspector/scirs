//! Test to verify migration from custom SIMD to core SIMD produces identical results

use num_complex::Complex64;
use scirs2_fft::simd_fft::{apply_simd_normalization, NormMode};
use scirs2_fft::simd_fft_new::{
    apply_simd_normalization as apply_simd_normalization_new, NormMode as NormModeNew,
};

#[test]
fn test_normalization_migration() {
    // Test various sizes including edge cases
    let test_sizes = vec![1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 1000, 1023, 1024];
    let test_scales = vec![1.0, 0.5, 0.25, 2.0, 0.125, 1.0 / 1024.0];

    for size in test_sizes {
        for &scale in &test_scales {
            // Create test data
            let mut data_old = Vec::with_capacity(size);
            let mut data_new = Vec::with_capacity(size);

            for i in 0..size {
                let c = Complex64::new(i as f64, (i as f64) * 0.5);
                data_old.push(c);
                data_new.push(c);
            }

            // Apply old normalization
            apply_simd_normalization(&mut data_old, scale);

            // Apply new normalization
            apply_simd_normalization_new(&mut data_new, scale);

            // Compare results
            for (i, (old, new)) in data_old.iter().zip(data_new.iter()).enumerate() {
                let re_diff = (old.re - new.re).abs();
                let im_diff = (old.im - new.im).abs();

                // Allow for small floating point differences
                assert!(
                    re_diff < 1e-10,
                    "Real part mismatch at index {} for size {} scale {}: old={}, new={}, diff={}",
                    i,
                    size,
                    scale,
                    old.re,
                    new.re,
                    re_diff
                );
                assert!(
                    im_diff < 1e-10,
                    "Imaginary part mismatch at index {} for size {} scale {}: old={}, new={}, diff={}",
                    i, size, scale, old.im, new.im, im_diff
                );
            }
        }
    }
}

#[test]
fn test_simd_detection_migration() {
    use scirs2_fft::simd_fft::simd_support_available;
    use scirs2_fft::simd_fft_new::simd_support_available as simd_support_available_new;

    // Both should report the same SIMD availability
    let old_support = simd_support_available();
    let new_support = simd_support_available_new();

    println!("Old SIMD detection: {}", old_support);
    println!("New SIMD detection: {}", new_support);

    // They might differ since old checks for specific instructions (AVX2, SSE4.1)
    // while new uses generic platform capabilities
    // But both should at least agree on whether some form of SIMD is available
    if old_support {
        assert!(
            new_support,
            "Old implementation detected SIMD but new implementation did not"
        );
    }
}

#[test]
fn test_f64_to_complex_migration() {
    use scirs2_fft::simd_fft_new::simd_f64_to_complex;

    let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = simd_f64_to_complex(&test_data);

    assert_eq!(result.len(), test_data.len());
    for (i, &val) in test_data.iter().enumerate() {
        assert_eq!(result[i].re, val);
        assert_eq!(result[i].im, 0.0);
    }
}
