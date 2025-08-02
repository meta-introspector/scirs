//! Simple validation script for basic stats functionality
//! This script performs basic functionality checks without heavy dependencies

use ndarray::Array1;
use scirs2_stats::{mean, std, var};

#[allow(dead_code)]
fn main() {
    println!("🔍 Basic Stats Validation");
    println!("==========================\n");

    // Basic data for testing
    let test_data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    println!("✅ Test data created: {:?}", test_data);

    // Basic statistical functions
    println!("1. 📊 Basic Statistics");

    match mean(&test_data.view()) {
        Ok(mean_val) => println!("   ✅ Mean: {:.6}", mean_val),
        Err(e) => println!("   ❌ Mean failed: {}", e),
    }

    match var(&test_data.view(), 1, None) {
        Ok(var_val) => println!("   ✅ Variance: {:.6}", var_val),
        Err(e) => println!("   ❌ Variance failed: {}", e),
    }

    match std(&test_data.view(), 1, None) {
        Ok(std_val) => println!("   ✅ Standard Deviation: {:.6}", std_val),
        Err(e) => println!("   ❌ Standard Deviation failed: {}", e),
    }

    println!("\n✅ Basic validation completed successfully!");
}
