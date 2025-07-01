use scirs2_special::{expit, logit, expit_array, logit_array};
use ndarray::array;

fn main() {
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

    // Test inverse relationship
    println!("Testing expit/logit inverse relationship:");
    let values = [0.1, 0.3, 0.5, 0.7, 0.9];
    for &val in &values {
        let logit_val = logit(val).unwrap();
        let back = expit(logit_val);
        assert!((back - val).abs() < 1e-10);
    }
    println!("✓ inverse relationship tests passed");

    // Test array functions
    println!("Testing array functions:");
    
    // Test expit_array
    let input = array![0.0, 1.0, -1.0];
    let result = expit_array(&input.view());
    assert!((result[0] - 0.5).abs() < 1e-10);
    assert!(result[1] > 0.7);
    assert!(result[2] < 0.3);
    
    // Test logit_array
    let prob_input = array![0.1, 0.5, 0.9];
    let logit_result = logit_array(&prob_input.view());
    assert!((logit_result[1] - 0.0).abs() < 1e-10);
    assert!(logit_result[0] < 0.0);
    assert!(logit_result[2] > 0.0);
    
    println!("✓ array function tests passed");
    
    println!("\n🎉 All new function tests passed successfully!");
    println!("✓ expit (logistic function)");
    println!("✓ logit (inverse logistic)"); 
    println!("✓ expit_array");
    println!("✓ logit_array");
    println!("\nSciPy parity improvements completed!");
}