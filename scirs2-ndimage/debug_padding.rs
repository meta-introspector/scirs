// Debug script to test padding behavior
use ndarray::{array, Array2};

// Simulating the exact same padding logic as in utils.rs
fn debug_reflect_padding() {
    let input = array![[1.0, 2.0], [3.0, 4.0]];
    println!("Original input: {:?}", input);
    
    // Simulate BorderMode::Reflect with pad_width [(1,1), (1,1)]
    let pad_width = [(1, 1), (1, 1)];
    let new_shape = [4, 4]; // 2+1+1, 2+1+1
    
    // Create padded array filled with zeros initially 
    let mut padded = Array2::<f64>::zeros((4, 4));
    
    // Copy input to center (starting at (1,1))
    let start_i = 1;
    let start_j = 1;
    for i in 0..2 {
        for j in 0..2 {
            padded[[start_i + i, start_j + j]] = input[[i, j]];
        }
    }
    
    println!("After copying input to center: {:?}", padded);
    
    // Reflect padding for rows (simplified version)
    // Top row padding 
    for i in 0..1 {
        let src_i = 1 - i; // This should be 1 for i=0
        for j in 0..2 {
            padded[[i, start_j + j]] = input[[src_i, j]];
        }
    }
    
    println!("After top row reflection: {:?}", padded);
    
    // Now let's see what neighborhood extraction gives us
    for pos_i in 0..2 {
        for pos_j in 0..2 {
            let mut neighborhood = Vec::new();
            for di in 0..2 {
                for dj in 0..2 {
                    let pi = pos_i + di;
                    let pj = pos_j + dj;
                    if pi < 4 && pj < 4 {
                        neighborhood.push(padded[[pi, pj]]);
                    }
                }
            }
            println!("Position ({}, {}): neighborhood = {:?}", pos_i, pos_j, neighborhood);
            let max_val = neighborhood.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            println!("  Max value: {}", max_val);
        }
    }
}

fn main() {
    debug_reflect_padding();
}