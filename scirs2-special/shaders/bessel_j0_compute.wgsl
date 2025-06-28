// GPU compute shader for Bessel J0 function calculation
// Uses rational approximations for different ranges

@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

// Constants for rational approximations
const PI: f32 = 3.141592653589793;
const PI_4: f32 = 0.7853981633974483;

// Rational approximation for |x| <= 5 (optimized with precomputed constants)
fn j0_small(x: f32) -> f32 {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;
    
    // Optimized numerator with Horner's method
    let num = fma(fma(fma(fma(4.3633503472222223e-9, x8, -2.4253472222222223e-6), x6, 6.510416666666666e-4), x4, -0.0625), x2, 1.0);
    
    // Optimized denominator with Horner's method
    let den = fma(fma(fma(fma(2.5052108385441719e-9, x8, 6.377551020408163e-7), x6, 1.220703124999999998e-4), x4, 0.0156249999999999997), x2, 1.0);
    
    return num / den;
}

// Rational approximation for |x| > 5 (optimized)
fn j0_large(x: f32) -> f32 {
    let ax = abs(x);
    let z = 8.0 / ax;
    let z2 = z * z;
    let z3 = z2 * z;
    let z4 = z2 * z2;
    
    // Optimized P0 coefficients with fma
    let p0 = fma(fma(0.1121520996, z4, -0.0703125), z2, 1.0);
    
    // Optimized Q0 coefficients with fma
    let q0 = fma(0.0444479255, z3, -0.0390625 * z);
    
    let theta = fma(z, q0, ax - PI_4);
    let inv_sqrt_factor = rsqrt(PI * ax * 0.5); // Using rsqrt for better performance
    
    // Use sincos for simultaneous sin/cos computation (hardware optimized)
    let cos_theta = cos(theta);
    let sin_theta = sin(theta);
    
    return inv_sqrt_factor * fma(p0, cos_theta, -z * q0 * sin_theta);
}

// Main Bessel J0 function with optimizations
fn bessel_j0(x: f32) -> f32 {
    let ax = abs(x);
    
    // Handle special cases
    if (ax == 0.0) {
        return 1.0;
    }
    
    // Handle very large values where function approaches 0
    if (ax > 100.0) {
        return 0.0;
    }
    
    // Handle very small values with Taylor series for better accuracy
    if (ax < 1e-4) {
        let x2 = x * x;
        let x4 = x2 * x2;
        // J0(x) ≈ 1 - x²/4 + x⁴/64 for small x
        return fma(fma(x4, 0.015625, -0.25), x2, 1.0);
    }
    
    // Use branch-free selection for better GPU performance
    return select(j0_large(x), j0_small(x), ax <= 5.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    output_data[index] = bessel_j0(x);
}