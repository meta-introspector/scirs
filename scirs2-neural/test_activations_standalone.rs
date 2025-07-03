//! Standalone test of activation functions without cargo build
//! Run with: rustc --edition 2021 test_activations_standalone.rs && ./test_activations_standalone

// Minimal error type for testing
#[derive(Debug)]
pub enum TestError {
    ComputationError(String),
}

impl std::fmt::Display for TestError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TestError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
        }
    }
}

impl std::error::Error for TestError {}

pub type Result<T> = std::result::Result<T, TestError>;

// Simple GELU implementation for testing
pub struct GELU {
    fast: bool,
}

impl GELU {
    pub fn new() -> Self {
        Self { fast: false }
    }
    
    pub fn fast() -> Self {
        Self { fast: true }
    }
    
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>> {
        let mut output = Vec::with_capacity(input.len());
        
        if self.fast {
            let sqrt_2_over_pi = 0.7978845608028654;
            let coeff = 0.044715;
            
            for &x in input {
                let x3 = x * x * x;
                let inner = sqrt_2_over_pi * (x + coeff * x3);
                let result = 0.5 * x * (1.0 + inner.tanh());
                output.push(result);
            }
        } else {
            let sqrt_pi_over_2 = 1.2533141373155;
            let coeff = 0.044715;
            
            for &x in input {
                let x2 = x * x;
                let inner = sqrt_pi_over_2 * x * (1.0 + coeff * x2);
                let result = 0.5 * x * (1.0 + inner.tanh());
                output.push(result);
            }
        }
        
        Ok(output)
    }
}

fn main() -> Result<()> {
    println!("Testing GELU activation function...");
    
    let gelu = GELU::new();
    let input = vec![1.0, -1.0, 2.0, 0.0, -2.0];
    
    println!("Input: {:?}", input);
    
    let output = gelu.forward(&input)?;
    println!("GELU output: {:?}", output);
    
    let gelu_fast = GELU::fast();
    let output_fast = gelu_fast.forward(&input)?;
    println!("GELU fast output: {:?}", output_fast);
    
    // Basic validation
    assert!(output[0] > 0.0, "GELU(1.0) should be positive");
    assert!(output[1] < 0.0, "GELU(-1.0) should be negative"); 
    assert_eq!(output[3], 0.0, "GELU(0.0) should be 0.0");
    
    println!("✅ All GELU tests passed!");
    Ok(())
}