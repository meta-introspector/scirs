// GPU compute shader for Bessel J0 function calculation
// Uses rational approximations for different ranges

@group(0) @binding(0)
var<storage, read> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

// Constants for rational approximations
const PI: f32 = 3.141592653589793;
const PI_4: f32 = 0.7853981633974483;

// Rational approximation for |x| <= 5
fn j0_small(x: f32) -> f32 {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;
    
    // Numerator coefficients
    let num = 1.0
        - 0.0625 * x2
        + 0.0006510416666666666 * x4
        - 0.0000024253472222222223 * x6
        + 0.0000000043633503472222223 * x8;
    
    // Denominator coefficients  
    let den = 1.0
        + 0.0156249999999999997 * x2
        + 0.0001220703124999999998 * x4
        + 0.0000006377551020408163 * x6
        + 0.0000000025052108385441719 * x8;
    
    return num / den;
}

// Rational approximation for |x| > 5
fn j0_large(x: f32) -> f32 {
    let ax = abs(x);
    let z = 8.0 / ax;
    let z2 = z * z;
    let z3 = z2 * z;
    let z4 = z2 * z2;
    
    // P0 coefficients
    let p0 = 1.0
        - 0.0703125 * z2
        + 0.1121520996 * z4;
    
    // Q0 coefficients  
    let q0 = -0.0390625 * z
        + 0.0444479255 * z3;
    
    let theta = ax - PI_4 + z * q0;
    let factor = sqrt(2.0 / (PI * ax));
    
    return factor * (p0 * cos(theta) - z * q0 * sin(theta));
}

// Main Bessel J0 function
fn bessel_j0(x: f32) -> f32 {
    let ax = abs(x);
    
    if (ax <= 5.0) {
        return j0_small(x);
    } else {
        return j0_large(x);
    }
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