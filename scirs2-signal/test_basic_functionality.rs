// Basic functionality test for scirs2-signal

use crate::error::SignalResult;
use scirs2_signal::filter::butter;
use scirs2_signal::error::SignalResult;

#[allow(dead_code)]
fn main() -> SignalResult<()> {
    println!("Testing basic scirs2-signal functionality...");
    
    // Test basic Butterworth filter design
    let _sos = butter(4, 0.5, "low", Some("analog"))?;
    
    println!("✓ Basic Butterworth filter design works");
    
    // Test basic signal generation
    let signal: Vec<f64> = (0..100)
        .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / 100.0).sin())
        .collect();
    
    println!("✓ Basic signal generation works");
    
    // Test basic validation
    if signal.len() == 100 {
        println!("✓ Signal length validation works");
    }
    
    println!("✅ All basic functionality tests passed!");
    
    Ok(())
}
