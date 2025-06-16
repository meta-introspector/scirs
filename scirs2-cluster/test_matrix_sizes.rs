use ndarray::{Array2, array};
use scirs2_linalg::eigh;

fn main() {
    println!("Testing eigh function with different matrix sizes:");
    
    // Test 1x1 matrix
    println!("\n=== Testing 1x1 matrix ===");
    let a1 = array![[5.0_f64]];
    match eigh(&a1.view(), None) {
        Ok((eigenvals, eigenvecs)) => {
            println!("1x1 matrix: SUCCESS");
            println!("Eigenvalues: {:?}", eigenvals);
            println!("Eigenvectors: {:?}", eigenvecs);
        }
        Err(e) => println!("1x1 matrix: FAILED - {:?}", e),
    }
    
    // Test 2x2 matrix
    println!("\n=== Testing 2x2 matrix ===");
    let a2 = array![[1.0_f64, 0.5], [0.5, 2.0]];
    match eigh(&a2.view(), None) {
        Ok((eigenvals, eigenvecs)) => {
            println!("2x2 matrix: SUCCESS");
            println!("Eigenvalues: {:?}", eigenvals);
            println!("Eigenvectors: {:?}", eigenvecs);
        }
        Err(e) => println!("2x2 matrix: FAILED - {:?}", e),
    }
    
    // Test 3x3 matrix
    println!("\n=== Testing 3x3 matrix ===");
    let a3 = array![
        [1.0_f64, 0.5, 0.0],
        [0.5, 2.0, 0.5],
        [0.0, 0.5, 3.0]
    ];
    match eigh(&a3.view(), None) {
        Ok((eigenvals, eigenvecs)) => {
            println!("3x3 matrix: SUCCESS");
            println!("Eigenvalues: {:?}", eigenvals);
            println!("Eigenvectors: {:?}", eigenvecs);
        }
        Err(e) => println!("3x3 matrix: FAILED - {:?}", e),
    }
    
    // Test 4x4 matrix - This should fail based on the code analysis
    println!("\n=== Testing 4x4 matrix ===");
    let a4 = array![
        [1.0_f64, 0.5, 0.0, 0.0],
        [0.5, 2.0, 0.5, 0.0],
        [0.0, 0.5, 3.0, 0.5],
        [0.0, 0.0, 0.5, 4.0]
    ];
    match eigh(&a4.view(), None) {
        Ok((eigenvals, eigenvecs)) => {
            println!("4x4 matrix: SUCCESS");
            println!("Eigenvalues: {:?}", eigenvals);
            println!("Eigenvectors: {:?}", eigenvecs);
        }
        Err(e) => println!("4x4 matrix: FAILED - {:?}", e),
    }
    
    // Test 5x5 matrix
    println!("\n=== Testing 5x5 matrix ===");
    let mut a5 = Array2::eye(5);
    a5[[0, 1]] = 0.1;
    a5[[1, 0]] = 0.1;
    a5[[1, 2]] = 0.1;
    a5[[2, 1]] = 0.1;
    match eigh(&a5.view(), None) {
        Ok((eigenvals, eigenvecs)) => {
            println!("5x5 matrix: SUCCESS");
            println!("Eigenvalues: {:?}", eigenvals);
            println!("Eigenvectors: {:?}", eigenvecs);
        }
        Err(e) => println!("5x5 matrix: FAILED - {:?}", e),
    }
    
    // Test 10x10 matrix
    println!("\n=== Testing 10x10 matrix ===");
    let a10 = Array2::eye(10);
    match eigh(&a10.view(), None) {
        Ok((eigenvals, eigenvecs)) => {
            println!("10x10 matrix: SUCCESS");
            println!("Eigenvalues: {:?}", eigenvals);
            println!("Eigenvectors: {:?}", eigenvecs);
        }
        Err(e) => println!("10x10 matrix: FAILED - {:?}", e),
    }
}