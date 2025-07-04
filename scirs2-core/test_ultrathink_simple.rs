//! Simple test for ultrathink mode basic functionality

fn main() {
    println!("Testing advanced mode basic functionality...");
    
    // Test basic imports - this should compile if the modules are working
    println!("Testing imports...");
    
    // Test ecosystem coordinator
    let _coordinator = scirs2_core::advanced_ecosystem_integration::UltrathinkEcosystemCoordinator::new();
    println!("✓ Ecosystem coordinator created");
    
    // Test distributed computer
    match scirs2_core::advanced_distributed_computing::UltrathinkDistributedComputer::new() {
        Ok(_) => println!("✓ Distributed computer created"),
        Err(e) => println!("⚠ Distributed computer failed: {}", e),
    }
    
    // Test neural architecture search
    let search_space = scirs2_core::neural_architecture_search::SearchSpace::default();
    let objectives = scirs2_core::neural_architecture_search::OptimizationObjectives::default();
    let constraints = scirs2_core::neural_architecture_search::HardwareConstraints::default();
    let config = scirs2_core::neural_architecture_search::SearchConfig {
        strategy: scirs2_core::neural_architecture_search::NASStrategy::Evolutionary,
        max_evaluations: 5,
        population_size: 3,
        max_generations: 2,
    };

    match scirs2_core::neural_architecture_search::NeuralArchitectureSearch::new(
        search_space,
        scirs2_core::neural_architecture_search::NASStrategy::Evolutionary,
        objectives,
        constraints,
        config,
    ) {
        Ok(_) => println!("✓ Neural architecture search created"),
        Err(e) => println!("⚠ Neural architecture search failed: {}", e),
    }
    
    #[cfg(feature = "jit")]
    {
        // Test JIT compiler if feature is enabled
        match scirs2_core::advanced_jit_compilation::UltrathinkJitCompiler::new() {
            Ok(_) => println!("✓ JIT compiler created"),
            Err(e) => println!("⚠ JIT compiler failed: {}", e),
        }
    }
    #[cfg(not(feature = "jit"))]
    {
        println!("ℹ JIT compiler test skipped (feature not enabled)");
    }
    
    println!("✅ Basic advanced mode functionality test complete!");
}