// GPU compute shader for error function (erf) calculation
// Uses Abramowitz and Stegun approximation

@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

// Constants for error function approximation
const A1: f32 = 0.254829592;
const A2: f32 = -0.284496736;
const A3: f32 = 1.421413741;
const A4: f32 = -1.453152027;
const A5: f32 = 1.061405429;
const P: f32 = 0.3275911;

// Error function approximation using Abramowitz and Stegun formula (optimized)
fn erf_approx(x: f32) -> f32 {
    // Save the sign of x
    let sign = select(-1.0, 1.0, x >= 0.0);
    let ax = abs(x);
    
    // Handle large values
    if (ax > 6.0) {
        return sign;
    }
    
    // A&S formula 7.1.26 with optimized polynomial evaluation using Horner's method
    let t = 1.0 / fma(P, ax, 1.0);
    let t2 = t * t;
    
    // Use Horner's method for better numerical stability and performance
    let poly = fma(fma(fma(fma(A5, t, A4), t, A3), t, A2), t, A1) * t;
    
    // Use fma for the final computation
    let ax2 = ax * ax;
    let y = fma(-poly, exp(-ax2), 1.0);
    
    return sign * y;
}

// More accurate error function for small values (optimized Taylor series)
fn erf_taylor(x: f32) -> f32 {
    // Taylor series for small |x| with optimized computation
    if (abs(x) < 0.5) {
        let x2 = x * x;
        
        // Use Horner's method for Taylor series: erf(x) ≈ (2/√π) * x * (1 - x²/3 + x⁴/10 - x⁶/42 + x⁸/216)
        let series = fma(fma(fma(fma(x2, 0.004629629629629629, -0.023809523809523808), x2, 0.1), x2, -0.3333333333333333), x2, 1.0);
        
        let two_over_sqrt_pi = 1.1283791670955126;
        return two_over_sqrt_pi * x * series;
    }
    
    return erf_approx(x);
}

// Main error function with comprehensive optimizations
fn erf_optimized(x: f32) -> f32 {
    // Handle special cases efficiently
    if (x != x) { // NaN check
        return x;
    }
    
    let ax = abs(x);
    
    if (ax == 0.0) {
        return 0.0;
    }
    
    // Early exit for very large values
    if (ax > 10.0) {
        return select(-1.0, 1.0, x > 0.0);
    }
    
    // Use branch-free selection for better GPU performance
    // For very small values (< 1e-6), use simple approximation: erf(x) ≈ (2/√π) * x
    if (ax < 1e-6) {
        return 1.1283791670955126 * x;
    }
    
    // Use Taylor series for small values, approximation for larger
    return erf_taylor(x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    output_data[index] = erf_optimized(x);
}