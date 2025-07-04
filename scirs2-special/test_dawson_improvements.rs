// Test script to verify Dawson function accuracy improvements
use scirs2_special::erf::dawsn;

#[allow(dead_code)]
fn main() {
    println!("Testing Dawson function accuracy improvements...");
    
    // Test cases with known values
    let test_cases = [
        (0.0, 0.0),                    // D(0) = 0
        (0.1, 0.099335326418),         // Small x
        (0.5, 0.480453394613),         // Small x  
        (1.0, 0.538079506913),         // Boundary
        (2.0, 0.301341486081),         // Moderate x
        (3.0, 0.177891584421),         // Moderate x
        (4.0, 0.129348205085),         // Intermediate x (where we fixed the bug)
        (5.0, 0.101000590205),         // Large x boundary
        (10.0, 0.050001248855),        // Large x
    ];
    
    println!("Testing against reference values:");
    for (x, expected) in test_cases {
        let result = dawsn(x);
        let error = (result - expected).abs();
        let relative_error = if expected != 0.0 { error / expected.abs() } else { error };
        
        println!("dawsn({:.1}) = {:.12}, expected = {:.12}, rel_err = {:.2e}", 
                 x, result, expected, relative_error);
                 
        if relative_error < 1e-6 {
            println!("  ✅ PASS");
        } else {
            println!("  ❌ FAIL (relative error too large)");
        }
    }
    
    // Test odd symmetry: D(-x) = -D(x)
    println!("\nTesting odd symmetry D(-x) = -D(x):");
    let x_vals = [0.5, 1.0, 2.0, 3.5, 5.0];
    for x in x_vals {
        let pos = dawsn(x);
        let neg = dawsn(-x);
        let symmetry_error = (pos + neg).abs();
        
        println!("D({:.1}) = {:.8}, D(-{:.1}) = {:.8}, |D(x) + D(-x)| = {:.2e}", 
                 x, pos, x, neg, symmetry_error);
                 
        if symmetry_error < 1e-10 {
            println!("  ✅ PASS");
        } else {
            println!("  ❌ FAIL (symmetry violated)");
        }
    }
    
    println!("\nDawson function accuracy improvements test completed.");
}
