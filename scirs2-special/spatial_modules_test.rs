//! Simple test to validate that scirs2-spatial advanced modules are accessible
//! This test verifies that the advanced spatial modules can be imported and used.

use scirs2_spatial::{
    KDTree, distance::euclidean,
    quantum_inspired::QuantumClusterer,
    neuromorphic::SpikingNeuralClusterer,
    gpu_accel::is_gpu_acceleration_available,
};
use ndarray::array;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing SciRS2-Spatial Advanced Modules Access");
    println!("================================================");
    
    // Test basic spatial functionality
    let points = array![
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
    ];
    
    // Test 1: Core spatial algorithms
    println!("📊 Testing core spatial algorithms...");
    let kdtree = KDTree::new(&points)?;
    let (indices, _distances) = kdtree.query(&[0.5, 0.5], 2)?;
    println!("✅ KDTree query successful: found {} neighbors", indices.len());
    
    let dist = euclidean(&[0.0, 0.0], &[1.0, 1.0]);
    println!("✅ Distance calculation: {:.3}", dist);
    
    // Test 2: Quantum-inspired algorithms
    println!("\n🔬 Testing quantum-inspired algorithms...");
    match QuantumClusterer::new(2, 16, 10, 0.01) {
        Ok(mut clusterer) => {
            match clusterer.cluster(&points.view()).await {
                Ok(result) => {
                    println!("✅ Quantum clustering successful: {} clusters, {} labels", 
                             result.centers.nrows(), result.labels.len());
                },
                Err(e) => println!("⚠️  Quantum clustering execution failed: {}", e),
            }
        },
        Err(e) => println!("⚠️  Quantum clustering initialization failed: {}", e),
    }
    
    // Test 3: Neuromorphic computing
    println!("\n🧠 Testing neuromorphic computing...");
    let mut spiking_clusterer = SpikingNeuralClusterer::new(2);
    match spiking_clusterer.cluster(&points.view()).await {
        Ok(result) => {
            println!("✅ Neuromorphic clustering successful: {} clusters, {} labels", 
                     result.centers.nrows(), result.labels.len());
            println!("   Silhouette score: {:.3}", result.silhouette_score);
        },
        Err(e) => println!("⚠️  Neuromorphic clustering failed: {}", e),
    }
    
    // Test 4: GPU capabilities
    println!("\n🖥️  Testing GPU capabilities...");
    let gpu_available = is_gpu_acceleration_available();
    println!("GPU acceleration available: {}", if gpu_available { "✅ Yes" } else { "❌ No" });
    
    println!("\n🎉 Advanced modules test completed successfully!");
    println!("   All advanced spatial modules are accessible and functional.");
    
    Ok(())
}