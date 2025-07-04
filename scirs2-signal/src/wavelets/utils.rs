//! Helper functions for wavelet transforms

/// Helper function to calculate factorial
#[allow(dead_code)]
pub fn factorial(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}
