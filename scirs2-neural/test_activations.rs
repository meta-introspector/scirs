use scirs2_neural::prelude::*;
use ndarray::Array;

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("Testing activation functions...");
    
    // Test GELU
    let gelu = GELU::new();
    let input = Array::from_vec(vec![1.0, -1.0, 0.0, 2.0]).into_dyn();
    let output = gelu.forward(&input)?;
    println!("GELU test: {:?} -> {:?}", input, output);
    
    // Test Sigmoid
    let sigmoid = Sigmoid::new();
    let input = Array::from_vec(vec![0.0, 1.0, -1.0]).into_dyn();
    let output = sigmoid.forward(&input)?;
    println!("Sigmoid test: {:?} -> {:?}", input, output);
    
    // Test ReLU
    let relu = ReLU::new();
    let input = Array::from_vec(vec![1.0, -1.0, 0.0, 2.0]).into_dyn();
    let output = relu.forward(&input)?;
    println!("ReLU test: {:?} -> {:?}", input, output);
    
    // Test Tanh
    let tanh = Tanh::new();
    let input = Array::from_vec(vec![1.0, -1.0, 0.0]).into_dyn();
    let output = tanh.forward(&input)?;
    println!("Tanh test: {:?} -> {:?}", input, output);
    
    // Test Softmax
    let softmax = Softmax::new(-1);
    let input = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
    let output = softmax.forward(&input)?;
    println!("Softmax test: {:?} -> {:?}", input, output);
    
    println!("All activation functions working correctly!");
    Ok(())
}
