// Test script to verify polygamma sign fix
use scirs2_special::gamma::polygamma;

#[allow(dead_code)]
fn main() {
    println!("Testing polygamma sign fix...");
    
    // Test polygamma(1, 1) = π²/6 (should be positive)
    let pi_squared_over_6 = std::f64::consts::PI.powi(2) / 6.0;
    let result = polygamma(1, 1.0);
    
    println!("polygamma(1, 1.0) = {:.16}", result);
    println!("Expected π²/6 = {:.16}", pi_squared_over_6);
    println!("Ratio: {:.6}", result / pi_squared_over_6);
    
    if result > 0.0 && (result / pi_squared_over_6 - 1.0).abs() < 0.01 {
        println!("✅ PASS: polygamma sign fix successful");
    } else {
        println!("❌ FAIL: polygamma sign fix failed");
    }
    
    // Test trigamma monotonicity
    println!("\nTesting trigamma monotonicity:");
    let x_vals = [1.0, 2.0, 3.0, 4.0, 5.0];
    for x in x_vals {
        let psi1 = polygamma(1, x);
        println!("polygamma(1, {:.1}) = {:.6}", x, psi1);
    }
}
