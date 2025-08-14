/// Quick test file to validate new Advanced mode functions
use scirs2_special::{sici, shichi, spence, erfcx, erfi, wofz, expit, logit, expit_array, logit_array};
use ndarray::array;

#[allow(dead_code)]
fn main() {
    println!("Testing new SciPy parity functions (advanced mode)...");
    
    // Test sici function
    if let Ok((si_val, ci_val)) = sici(1.0) {
        println!("sici(1.0) = ({:.6}, {:.6})", si_val, ci_val);
        assert!((si_val - 0.946083).abs() < 1e-5, "Si(1) incorrect");
        assert!((ci_val - 0.337404).abs() < 1e-5, "Ci(1) incorrect");
        println!("✓ sici tests passed");
    } else {
        panic!("sici(1.0) failed");
    }
    
    // Test shichi function
    if let Ok((shi_val, chi_val)) = shichi(1.0) {
        println!("shichi(1.0) = ({:.6}, {:.6})", shi_val, chi_val);
        assert!((shi_val - 1.057251).abs() < 1e-5, "Shi(1) incorrect");
        assert!((chi_val - 0.837866).abs() < 1e-5, "Chi(1) incorrect");
        println!("✓ shichi tests passed");
    } else {
        panic!("shichi(1.0) failed");
    }
    
    // Test spence function
    if let Ok(spence_val) = spence(0.0) {
        let pi_sq_6 = std::f64::consts::PI.powi(2) / 6.0;
        println!("spence(0.0) = {:.10}, expected π²/6 = {:.10}", spence_val, pi_sq_6);
        assert!((spence_val - pi_sq_6).abs() < 1e-10, "spence(0) should be π²/6");
    } else {
        panic!("spence(0.0) failed");
    }
    
    if let Ok(spence_1) = spence(1.0) {
        println!("spence(1.0) = {:.10}", spence_1);
        assert!(spence_1.abs() < 1e-10, "spence(1) should be 0");
        println!("✓ spence tests passed");
    } else {
        panic!("spence(1.0) failed");
    }
    
    // Test erfcx function
    let erfcx_val = erfcx(0.0);
    println!("erfcx(0.0) = {:.10}", erfcx_val);
    assert!((erfcx_val - 1.0).abs() < 1e-10, "erfcx(0) should be 1");
    println!("✓ erfcx tests passed");
    
    // Test erfi function
    let erfi_val = erfi(0.0);
    println!("erfi(0.0) = {:.10}", erfi_val);
    assert!(erfi_val.abs() < 1e-10, "erfi(0) should be 0");
    
    let erfi_1 = erfi(1.0);
    let erfi_neg1 = erfi(-1.0);
    println!("erfi(1.0) = {:.6}, erfi(-1.0) = {:.6}", erfi_1, erfi_neg1);
    assert!((erfi_1 + erfi_neg1).abs() < 1e-10, "erfi should be odd function");
    println!("✓ erfi tests passed");
    
    // Test wofz function
    let wofz_val = wofz(0.0);
    println!("wofz(0.0) = {:.10}", wofz_val);
    assert!((wofz_val - 1.0).abs() < 1e-10, "wofz(0) should be 1");
    println!("✓ wofz tests passed");

    // Test existing functions for regression
    println!("\nTesting existing functions for regression:");
    
    // Test expit function
    println!("Testing expit function:");
    assert!((expit(0.0) - 0.5).abs() < 1e-10);
    assert!(expit(10.0) > 0.99);
    assert!(expit(-10.0) < 0.01);
    println!("✓ expit tests passed");

    // Test logit function
    println!("Testing logit function:");
    assert!((logit(0.5).unwrap() - 0.0).abs() < 1e-10);
    assert!(logit(0.9).unwrap() > 0.0);
    assert!(logit(0.1).unwrap() < 0.0);
    assert!(logit(0.0).is_err());
    assert!(logit(1.0).is_err());
    println!("✓ logit tests passed");

    // Test array functions
    println!("Testing array functions:");
    
    // Test expit_array
    let input = array![0.0, 1.0, -1.0];
    let result = expit_array(&input.view());
    assert!((result[0] - 0.5).abs() < 1e-10);
    assert!(result[1] > 0.7);
    assert!(result[2] < 0.3);
    
    // Test logit_array
    let probinput = array![0.1, 0.5, 0.9];
    let logit_result = logit_array(&probinput.view());
    assert!((logit_result[1] - 0.0).abs() < 1e-10);
    assert!(logit_result[0] < 0.0);
    assert!(logit_result[2] > 0.0);
    
    println!("✓ array function tests passed");
    
    println!("\n🎉 All advanced mode function tests passed successfully!");
    println!("NEW HIGH-PRIORITY FUNCTIONS:");
    println!("✓ sici (sine and cosine integrals)");
    println!("✓ shichi (hyperbolic sine and cosine integrals)");
    println!("✓ spence (dilogarithm)");
    println!("✓ erfcx (scaled complementary error function)");
    println!("✓ erfi (imaginary error function)");
    println!("✓ wofz (Faddeeva function)");
    println!("\nEXISTING FUNCTIONS (regression tested):");
    println!("✓ expit (logistic function)");
    println!("✓ logit (inverse logistic)"); 
    println!("✓ expit_array");
    println!("✓ logit_array");
    println!("\nadvanced mode SciPy parity implementation completed! 🚀");
}
