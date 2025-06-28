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

// Error function approximation using Abramowitz and Stegun formula
fn erf_approx(x: f32) -> f32 {
    // Save the sign of x
    let sign = select(-1.0, 1.0, x >= 0.0);
    let ax = abs(x);
    
    // Handle large values
    if (ax > 6.0) {
        return sign;
    }
    
    // A&S formula 7.1.26
    let t = 1.0 / (1.0 + P * ax);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t2 * t2;
    let t5 = t4 * t;
    
    let y = 1.0 - (A1 * t + A2 * t2 + A3 * t3 + A4 * t4 + A5 * t5) * exp(-ax * ax);
    
    return sign * y;
}

// More accurate error function for small values
fn erf_taylor(x: f32) -> f32 {
    // Taylor series for small |x|
    if (abs(x) < 0.5) {
        let x2 = x * x;
        let x3 = x * x2;
        let x5 = x3 * x2;
        let x7 = x5 * x2;
        let x9 = x7 * x2;
        
        let two_sqrt_pi = 1.1283791670955126;
        return (2.0 / two_sqrt_pi) * (x - x3 / 3.0 + x5 / 10.0 - x7 / 42.0 + x9 / 216.0);
    }
    
    return erf_approx(x);
}

// Main error function
fn erf_optimized(x: f32) -> f32 {
    // Handle special cases
    if (x != x) { // NaN check
        return x;
    }
    
    if (x == 0.0) {
        return 0.0;
    }
    
    if (abs(x) > 10.0) {
        return select(-1.0, 1.0, x > 0.0);
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