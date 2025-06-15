use ndarray::{array, Array2};
use scirs2_ndimage::filters::generic_filter;

fn debug_max_func(values: &[f64]) -> f64 {
    println!("Neighborhood values: {:?}", values);
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    println!("Max value: {}", max_val);
    max_val
}

fn main() {
    let input = array![[1.0, 2.0], [3.0, 4.0]];
    println!("Input array: {:?}", input);
    
    let result = generic_filter(&input, debug_max_func, &[2, 2], None, None).unwrap();
    println!("Result: {:?}", result);
    
    // Check what each position gets
    for i in 0..2 {
        for j in 0..2 {
            println!("Position ({}, {}): {}", i, j, result[[i, j]]);
        }
    }
}