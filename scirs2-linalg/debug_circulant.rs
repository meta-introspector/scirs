use ndarray::{Array1, array};
use scirs2_linalg::circulant_toeplitz::CirculantMatrix;

fn main() {
    // Simple debug test for circulant matrix issues
    let first_row = array![1.0, 2.0, 3.0];
    let mut circ = CirculantMatrix::new(first_row.clone()).unwrap();
    
    println!("First row: {:?}", first_row);
    
    // Check the dense representation
    let dense = circ.to_dense();
    println!("Dense matrix:");
    for i in 0..3 {
        println!("{:?}", dense.row(i));
    }
    
    // Test vector
    let v = array![1.0, 0.0, 0.0];
    println!("Test vector: {:?}", v);
    
    // Expected result (first column)
    let expected = array![1.0, 3.0, 2.0];
    println!("Expected result: {:?}", expected);
    
    // Manual computation
    let manual_result = dense.dot(&v);
    println!("Manual result (dense * v): {:?}", manual_result);
    
    // FFT-based computation
    match circ.matvec(&v.view()) {
        Ok(result) => {
            println!("FFT result: {:?}", result);
            for i in 0..3 {
                println!("  result[{}] = {}, expected[{}] = {}, diff = {}", 
                        i, result[i], i, expected[i], (result[i] - expected[i]).abs());
            }
        }
        Err(e) => println!("Error in matvec: {:?}", e),
    }
    
    // Check eigenvalues
    if let Ok(eigenvals) = circ.compute_eigenvalues() {
        println!("Eigenvalues: {:?}", eigenvals);
    }
}