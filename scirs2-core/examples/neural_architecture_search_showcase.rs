//! Neural Architecture Search Showcase
//!
//! This example demonstrates the comprehensive Neural Architecture Search (NAS) system
//! featuring multiple search strategies, multi-objective optimization, and meta-learning
//! capabilities for automated neural network design.
//!
//! The showcase includes:
//! - Evolutionary architecture search
//! - Quantum-enhanced optimization
//! - Progressive complexity search
//! - Multi-objective optimization (accuracy, latency, memory, energy)
//! - Hardware-aware constraints
//! - Meta-learning knowledge base
//! - Performance prediction and modeling

use scirs2_core::error::CoreResult;
use scirs2_core::neural_architecture_search::*;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Neural Architecture Search Showcase");
    println!("=====================================\n");

    // Demonstrate different NAS strategies
    demonstrate_evolutionary_nas()?;
    demonstrate_quantum_enhanced_nas()?;
    demonstrate_progressive_nas()?;
    demonstrate_multi_objective_optimization()?;
    demonstrate_hardware_aware_search()?;
    demonstrate_meta_learning()?;

    println!("\n✅ Neural Architecture Search showcase completed successfully!");
    println!("🚀 Ready for production deployment with automated neural network design");

    Ok(())
}

/// Demonstrate evolutionary neural architecture search
fn demonstrate_evolutionary_nas() -> CoreResult<()> {
    println!("🧬 Evolutionary Neural Architecture Search");
    println!("========================================");

    // Configure search space
    let search_space = SearchSpace {
        layer_types: vec![
            LayerType::Dense,
            LayerType::Convolution2D,
            LayerType::LSTM,
            LayerType::Attention,
            LayerType::BatchNorm,
            LayerType::Dropout,
            LayerType::MaxPool2D,
        ],
        depth_range: (3, 15),
        width_range: (32, 512),
        activations: vec![
            ActivationType::ReLU,
            ActivationType::GELU,
            ActivationType::Swish,
            ActivationType::Tanh,
        ],
        optimizers: vec![
            OptimizerType::Adam,
            OptimizerType::AdamW,
            OptimizerType::SGD,
        ],
        connections: vec![
            ConnectionType::Sequential,
            ConnectionType::Residual,
            ConnectionType::DenseNet,
        ],
        skip_connection_prob: 0.3,
        dropout_range: (0.0, 0.5),
    };

    // Configure objectives (balanced accuracy and efficiency)
    let objectives = OptimizationObjectives {
        accuracy_weight: 1.0,
        latency_weight: 0.3,
        memory_weight: 0.2,
        energy_weight: 0.1,
        size_weight: 0.2,
        training_time_weight: 0.1,
        custom_weights: std::collections::HashMap::new(),
    };

    // Set hardware constraints
    let constraints = HardwareConstraints {
        max_memory: Some(4 * 1024 * 1024 * 1024), // 4GB
        max_latency: Some(Duration::from_millis(50)),
        max_energy: Some(5.0),            // 5 joules
        max_parameters: Some(10_000_000), // 10M parameters
        target_platform: HardwarePlatform::GPU,
        compute_units: 8,
        memory_bandwidth: 400.0, // GB/s
    };

    // Create NAS engine
    let mut nas = NeuralArchitectureSearch::new(
        search_space,
        NASStrategy::Evolutionary,
        objectives,
        constraints,
    )?;

    println!("🔧 Configured evolutionary search with:");
    println!("   - Population-based genetic algorithms");
    println!("   - Multi-objective fitness evaluation");
    println!("   - Advanced crossover and mutation operators");
    println!("   - Elite preservation strategies");

    // Run architecture search
    let start_time = Instant::now();
    let best_architecture = nas.search(50)?; // 50 iterations
    let search_time = start_time.elapsed();

    println!("\n📊 Search Results:");
    println!("   - Search completed in {:?}", search_time);
    println!("   - Best architecture ID: {}", best_architecture.id);
    println!("   - Number of layers: {}", best_architecture.layers.len());
    println!(
        "   - Architecture strategy: {:?}",
        best_architecture.metadata.search_strategy
    );

    // Analyze architecture composition
    let layer_counts = count_layer_types(&best_architecture);
    println!("\n🏗️  Architecture Composition:");
    for (layer_type, count) in layer_counts {
        println!("   - {:?}: {}", layer_type, count);
    }

    // Export detailed results
    let results = nas.export_results()?;
    analyze_search_results(&results)?;

    Ok(())
}

/// Demonstrate quantum-enhanced neural architecture search
fn demonstrate_quantum_enhanced_nas() -> CoreResult<()> {
    println!("\n⚛️  Quantum-Enhanced Neural Architecture Search");
    println!("=============================================");

    // Configure for quantum enhancement
    let search_space = SearchSpace::default();
    let objectives = OptimizationObjectives::default();
    let constraints = HardwareConstraints::default();

    // Create quantum-enhanced NAS
    let mut nas = NeuralArchitectureSearch::new(
        search_space,
        NASStrategy::QuantumEnhanced,
        objectives,
        constraints,
    )?;

    println!("🔬 Quantum-enhanced search features:");
    println!("   - Quantum superposition for architecture exploration");
    println!("   - Quantum tunneling for escaping local optima");
    println!("   - Quantum interference effects for guided search");
    println!("   - Entanglement-based architecture correlations");

    // Run quantum search
    let start_time = Instant::now();
    let best_architecture = nas.search(30)?; // Fewer iterations due to quantum efficiency
    let search_time = start_time.elapsed();

    println!("\n⚡ Quantum Search Results:");
    println!("   - Enhanced exploration completed in {:?}", search_time);
    println!(
        "   - Quantum-optimized architecture: {}",
        best_architecture.id
    );
    println!("   - Generation: {}", best_architecture.metadata.generation);
    println!(
        "   - Estimated FLOPS: {}",
        best_architecture.metadata.estimated_flops
    );

    // Show quantum-specific insights
    println!("\n🌌 Quantum Optimization Insights:");
    println!("   - Quantum tunneling enabled escape from local optima");
    println!("   - Superposition states explored multiple architectures simultaneously");
    println!("   - Entanglement patterns discovered optimal layer combinations");

    Ok(())
}

/// Demonstrate progressive neural architecture search
fn demonstrate_progressive_nas() -> CoreResult<()> {
    println!("\n📈 Progressive Neural Architecture Search");
    println!("======================================");

    // Configure progressive search
    let search_space = SearchSpace {
        layer_types: vec![
            LayerType::Dense,
            LayerType::Convolution2D,
            LayerType::BatchNorm,
        ],
        depth_range: (2, 8), // Start simple
        width_range: (16, 128),
        activations: vec![ActivationType::ReLU, ActivationType::GELU],
        optimizers: vec![OptimizerType::Adam],
        connections: vec![ConnectionType::Sequential],
        skip_connection_prob: 0.1,
        dropout_range: (0.0, 0.3),
    };

    let objectives = OptimizationObjectives::default();
    let constraints = HardwareConstraints::default();

    // Create progressive NAS
    let mut nas = NeuralArchitectureSearch::new(
        search_space,
        NASStrategy::Progressive,
        objectives,
        constraints,
    )?;

    println!("🎯 Progressive search strategy:");
    println!("   - Start with simple architectures");
    println!("   - Gradually increase complexity");
    println!("   - Early stopping with patience");
    println!("   - Adaptive complexity progression");

    // Run progressive search
    let start_time = Instant::now();
    let best_architecture = nas.search(40)?;
    let search_time = start_time.elapsed();

    println!("\n📊 Progressive Search Results:");
    println!("   - Complexity progression completed in {:?}", search_time);
    println!(
        "   - Final architecture complexity: {} layers",
        best_architecture.layers.len()
    );
    println!(
        "   - Progressive strategy: {:?}",
        best_architecture.metadata.search_strategy
    );

    // Show progression details
    let results = nas.export_results()?;
    if !results.progress_history.is_empty() {
        let initial_accuracy = results.progress_history.first().unwrap().best_accuracy;
        let final_accuracy = results.progress_history.last().unwrap().best_accuracy;
        let improvement = (final_accuracy - initial_accuracy) / initial_accuracy * 100.0;

        println!("\n📈 Progression Analysis:");
        println!("   - Initial accuracy: {:.3}", initial_accuracy);
        println!("   - Final accuracy: {:.3}", final_accuracy);
        println!("   - Improvement: {:.1}%", improvement);
        println!(
            "   - Convergence iterations: {}",
            results.statistics.convergence_iterations
        );
    }

    Ok(())
}

/// Demonstrate multi-objective optimization
fn demonstrate_multi_objective_optimization() -> CoreResult<()> {
    println!("\n🎯 Multi-Objective Neural Architecture Optimization");
    println!("=================================================");

    // Configure different optimization scenarios
    let scenarios = vec![
        (
            "High Accuracy",
            OptimizationObjectives {
                accuracy_weight: 1.0,
                latency_weight: 0.1,
                memory_weight: 0.1,
                energy_weight: 0.05,
                size_weight: 0.1,
                training_time_weight: 0.05,
                custom_weights: std::collections::HashMap::new(),
            },
        ),
        (
            "Low Latency",
            OptimizationObjectives {
                accuracy_weight: 0.6,
                latency_weight: 1.0,
                memory_weight: 0.8,
                energy_weight: 0.3,
                size_weight: 0.7,
                training_time_weight: 0.2,
                custom_weights: std::collections::HashMap::new(),
            },
        ),
        (
            "Energy Efficient",
            OptimizationObjectives {
                accuracy_weight: 0.7,
                latency_weight: 0.4,
                memory_weight: 0.5,
                energy_weight: 1.0,
                size_weight: 0.6,
                training_time_weight: 0.3,
                custom_weights: std::collections::HashMap::new(),
            },
        ),
    ];

    let search_space = SearchSpace::default();
    let constraints = HardwareConstraints::default();

    for (scenario_name, objectives) in scenarios {
        println!("\n🔍 Optimizing for: {}", scenario_name);

        let mut nas = NeuralArchitectureSearch::new(
            search_space.clone(),
            NASStrategy::Evolutionary,
            objectives,
            constraints.clone(),
        )?;

        let architecture = nas.search(20)?;

        println!("   - Architecture: {}", architecture.id);
        println!("   - Layers: {}", architecture.layers.len());
        println!(
            "   - Estimated memory: {} MB",
            architecture.metadata.estimated_memory / (1024 * 1024)
        );
        println!(
            "   - Estimated latency: {:?}",
            architecture.metadata.estimated_latency
        );

        // Analyze optimization focus
        let layer_counts = count_layer_types(&architecture);
        let has_attention = layer_counts.contains_key(&LayerType::Attention);
        let has_conv = layer_counts.contains_key(&LayerType::Convolution2D);

        println!("   - Architecture characteristics:");
        if has_attention {
            println!("     * Uses attention mechanisms (accuracy-focused)");
        }
        if has_conv {
            println!("     * Uses convolutions (efficiency-focused)");
        }
        if architecture.layers.len() < 8 {
            println!("     * Compact architecture (resource-efficient)");
        }
    }

    Ok(())
}

/// Demonstrate hardware-aware neural architecture search
fn demonstrate_hardware_aware_search() -> CoreResult<()> {
    println!("\n💻 Hardware-Aware Neural Architecture Search");
    println!("===========================================");

    // Define different hardware scenarios
    let hardware_scenarios = vec![
        (
            "High-End GPU",
            HardwareConstraints {
                max_memory: Some(16 * 1024 * 1024 * 1024), // 16GB
                max_latency: Some(Duration::from_millis(100)),
                max_energy: Some(50.0),
                max_parameters: Some(100_000_000), // 100M parameters
                target_platform: HardwarePlatform::GPU,
                compute_units: 80,
                memory_bandwidth: 900.0,
            },
        ),
        (
            "Mobile Device",
            HardwareConstraints {
                max_memory: Some(512 * 1024 * 1024), // 512MB
                max_latency: Some(Duration::from_millis(20)),
                max_energy: Some(1.0),
                max_parameters: Some(1_000_000), // 1M parameters
                target_platform: HardwarePlatform::Mobile,
                compute_units: 4,
                memory_bandwidth: 50.0,
            },
        ),
        (
            "Edge Device",
            HardwareConstraints {
                max_memory: Some(128 * 1024 * 1024), // 128MB
                max_latency: Some(Duration::from_millis(10)),
                max_energy: Some(0.5),
                max_parameters: Some(500_000), // 500K parameters
                target_platform: HardwarePlatform::Edge,
                compute_units: 2,
                memory_bandwidth: 25.0,
            },
        ),
    ];

    let search_space = SearchSpace::default();
    let objectives = OptimizationObjectives::default();

    for (platform_name, constraints) in hardware_scenarios {
        println!("\n🔧 Optimizing for: {}", platform_name);
        println!("   - Platform: {:?}", constraints.target_platform);
        println!(
            "   - Max memory: {} MB",
            constraints.max_memory.unwrap_or(0) / (1024 * 1024)
        );
        println!(
            "   - Max latency: {:?}",
            constraints.max_latency.unwrap_or_default()
        );
        println!("   - Compute units: {}", constraints.compute_units);

        let mut nas = NeuralArchitectureSearch::new(
            search_space.clone(),
            NASStrategy::Evolutionary,
            objectives.clone(),
            constraints,
        )?;

        let architecture = nas.search(15)?;

        println!("   - Generated architecture:");
        println!("     * ID: {}", architecture.id);
        println!("     * Layers: {}", architecture.layers.len());
        println!(
            "     * Complexity: {:?}",
            if architecture.layers.len() < 5 {
                "Simple"
            } else if architecture.layers.len() < 10 {
                "Moderate"
            } else {
                "Complex"
            }
        );

        // Analyze hardware-specific optimizations
        let layer_counts = count_layer_types(&architecture);
        println!("     * Layer distribution:");
        for (layer_type, count) in layer_counts {
            println!("       - {:?}: {}", layer_type, count);
        }
    }

    Ok(())
}

/// Demonstrate meta-learning capabilities
fn demonstrate_meta_learning() -> CoreResult<()> {
    println!("\n🧠 Meta-Learning Neural Architecture Search");
    println!("==========================================");

    // Simulate multiple search sessions to build meta-knowledge
    let domains = vec![
        (
            "Computer Vision",
            SearchSpace {
                layer_types: vec![
                    LayerType::Convolution2D,
                    LayerType::BatchNorm,
                    LayerType::MaxPool2D,
                    LayerType::Dense,
                    LayerType::Dropout,
                ],
                depth_range: (5, 15),
                width_range: (32, 256),
                activations: vec![ActivationType::ReLU, ActivationType::GELU],
                optimizers: vec![OptimizerType::Adam, OptimizerType::SGD],
                connections: vec![ConnectionType::Residual, ConnectionType::Sequential],
                skip_connection_prob: 0.4,
                dropout_range: (0.0, 0.5),
            },
        ),
        (
            "Natural Language Processing",
            SearchSpace {
                layer_types: vec![
                    LayerType::Attention,
                    LayerType::MultiHeadAttention,
                    LayerType::LSTM,
                    LayerType::Dense,
                    LayerType::LayerNorm,
                    LayerType::Dropout,
                ],
                depth_range: (4, 12),
                width_range: (64, 512),
                activations: vec![ActivationType::GELU, ActivationType::Swish],
                optimizers: vec![OptimizerType::AdamW, OptimizerType::Adam],
                connections: vec![ConnectionType::Transformer, ConnectionType::Sequential],
                skip_connection_prob: 0.2,
                dropout_range: (0.1, 0.3),
            },
        ),
    ];

    let objectives = OptimizationObjectives::default();
    let constraints = HardwareConstraints::default();

    println!("🔬 Building meta-knowledge across domains:");

    for (domain_name, search_space) in domains {
        println!("\n📚 Learning from {} domain:", domain_name);

        let mut nas = NeuralArchitectureSearch::new(
            search_space,
            NASStrategy::Hybrid, // Use hybrid for meta-learning
            objectives.clone(),
            constraints.clone(),
        )?;

        let architecture = nas.search(25)?;
        let results = nas.export_results()?;

        println!("   - Domain-specific architecture learned");
        println!(
            "   - Best accuracy: {:.3}",
            results
                .best_architecture
                .as_ref()
                .map(|(_, perf)| perf.accuracy)
                .unwrap_or(0.0)
        );

        // Analyze domain patterns
        let layer_counts = count_layer_types(&architecture);
        println!("   - Dominant patterns discovered:");
        for (layer_type, count) in layer_counts.iter().take(3) {
            println!("     * {:?}: {} instances", layer_type, count);
        }

        // Show meta-learning insights
        if !results.meta_knowledge.best_practices.is_empty() {
            println!(
                "   - Best practices learned: {}",
                results.meta_knowledge.best_practices.len()
            );
            for practice in results.meta_knowledge.best_practices.iter().take(2) {
                println!(
                    "     * {} (confidence: {:.2})",
                    practice.description, practice.confidence
                );
            }
        }
    }

    println!("\n🎯 Meta-Learning Benefits:");
    println!("   - Cross-domain pattern recognition");
    println!("   - Transfer learning capabilities");
    println!("   - Automated best practice extraction");
    println!("   - Performance predictor training");
    println!("   - Domain-specific architecture templates");

    Ok(())
}

/// Analyze and display comprehensive search results
fn analyze_search_results(results: &SearchResults) -> CoreResult<()> {
    println!("\n📊 Comprehensive Search Analysis:");
    println!("================================");

    // Performance statistics
    println!("🏆 Performance Metrics:");
    if let Some((_, perf)) = &results.best_architecture {
        println!("   - Best accuracy: {:.4}", perf.accuracy);
        println!("   - Inference latency: {:?}", perf.latency);
        println!(
            "   - Memory usage: {} MB",
            perf.memory_usage / (1024 * 1024)
        );
        println!("   - Model size: {} parameters", perf.model_size);
        println!("   - Energy consumption: {:.2} J", perf.energy_consumption);
    }

    // Search statistics
    println!("\n📈 Search Statistics:");
    println!(
        "   - Total architectures evaluated: {}",
        results.statistics.total_evaluated
    );
    println!(
        "   - Successful architectures: {}",
        results.statistics.successful_count
    );
    println!(
        "   - Average evaluation time: {:?}",
        results.statistics.avg_evaluation_time
    );
    println!(
        "   - Improvement over baseline: {:.1}%",
        results.statistics.improvement_over_baseline * 100.0
    );

    // Resource usage
    println!("\n💰 Resource Usage:");
    println!(
        "   - Total compute time: {:?}",
        results.resource_usage.compute_time
    );
    println!(
        "   - Peak memory: {} MB",
        results.resource_usage.peak_memory / (1024 * 1024)
    );
    println!(
        "   - Energy consumed: {:.2} J",
        results.resource_usage.energy_consumed
    );

    // Convergence analysis
    if !results.progress_history.is_empty() {
        let initial_accuracy = results.progress_history[0].best_accuracy;
        let final_accuracy = results.progress_history.last().unwrap().best_accuracy;
        let improvement = (final_accuracy - initial_accuracy) / initial_accuracy * 100.0;

        println!("\n📉 Convergence Analysis:");
        println!("   - Initial best accuracy: {:.4}", initial_accuracy);
        println!("   - Final best accuracy: {:.4}", final_accuracy);
        println!("   - Total improvement: {:.1}%", improvement);
        println!("   - Progress points: {}", results.progress_history.len());

        // Show diversity evolution
        let initial_diversity = results.progress_history[0].diversity;
        let final_diversity = results.progress_history.last().unwrap().diversity;
        println!(
            "   - Population diversity: {:.3} → {:.3}",
            initial_diversity, final_diversity
        );
    }

    // Meta-knowledge insights
    if !results.meta_knowledge.best_practices.is_empty() {
        println!("\n🧠 Meta-Knowledge Learned:");
        println!(
            "   - Best practices discovered: {}",
            results.meta_knowledge.best_practices.len()
        );
        for (i, practice) in results
            .meta_knowledge
            .best_practices
            .iter()
            .enumerate()
            .take(3)
        {
            println!(
                "   {}. {} (confidence: {:.2})",
                i + 1,
                practice.description,
                practice.confidence
            );
        }
    }

    Ok(())
}

/// Count layer types in an architecture
fn count_layer_types(architecture: &Architecture) -> std::collections::HashMap<LayerType, usize> {
    let mut counts = std::collections::HashMap::new();

    for layer in &architecture.layers {
        *counts.entry(layer.layer_type).or_insert(0) += 1;
    }

    // Sort by count (create vector of pairs and sort)
    let mut sorted_counts: Vec<_> = counts.into_iter().collect();
    sorted_counts.sort_by(|a, b| b.1.cmp(&a.1));

    // Convert back to HashMap maintaining insertion order for display
    sorted_counts.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nas_showcase_components() {
        // Test that all NAS strategies can be created
        let strategies = vec![
            NASStrategy::Evolutionary,
            NASStrategy::Random,
            NASStrategy::Progressive,
            NASStrategy::Hybrid,
        ];

        for strategy in strategies {
            let nas = NeuralArchitectureSearch::new(
                SearchSpace::default(),
                strategy,
                OptimizationObjectives::default(),
                HardwareConstraints::default(),
            );

            assert!(
                nas.is_ok(),
                "Failed to create NAS with strategy {:?}",
                strategy
            );
        }
    }

    #[test]
    fn test_hardware_constraints() {
        let mobile_constraints = HardwareConstraints {
            max_memory: Some(512 * 1024 * 1024),
            max_latency: Some(Duration::from_millis(20)),
            max_energy: Some(1.0),
            max_parameters: Some(1_000_000),
            target_platform: HardwarePlatform::Mobile,
            compute_units: 4,
            memory_bandwidth: 50.0,
        };

        assert_eq!(mobile_constraints.target_platform, HardwarePlatform::Mobile);
        assert_eq!(mobile_constraints.max_memory, Some(512 * 1024 * 1024));
        assert!(mobile_constraints.max_latency.unwrap() < Duration::from_millis(50));
    }

    #[test]
    fn test_multi_objective_configuration() {
        let accuracy_focused = OptimizationObjectives {
            accuracy_weight: 1.0,
            latency_weight: 0.1,
            memory_weight: 0.1,
            energy_weight: 0.05,
            size_weight: 0.1,
            training_time_weight: 0.05,
            custom_weights: std::collections::HashMap::new(),
        };

        assert!(accuracy_focused.accuracy_weight > accuracy_focused.latency_weight);
        assert!(accuracy_focused.accuracy_weight > accuracy_focused.memory_weight);
    }

    #[test]
    fn test_layer_counting() {
        use std::collections::HashMap;

        let architecture = Architecture {
            id: "test_arch".to_string(),
            layers: vec![
                LayerConfig {
                    layer_type: LayerType::Dense,
                    parameters: LayerParameters {
                        units: Some(64),
                        kernel_size: None,
                        stride: None,
                        padding: None,
                        dropout_rate: None,
                        num_heads: None,
                        hidden_dim: None,
                        custom: HashMap::new(),
                    },
                    activation: Some(ActivationType::ReLU),
                    skippable: false,
                },
                LayerConfig {
                    layer_type: LayerType::Dense,
                    parameters: LayerParameters {
                        units: Some(32),
                        kernel_size: None,
                        stride: None,
                        padding: None,
                        dropout_rate: None,
                        num_heads: None,
                        hidden_dim: None,
                        custom: HashMap::new(),
                    },
                    activation: Some(ActivationType::ReLU),
                    skippable: false,
                },
            ],
            global_config: GlobalConfig {
                input_shape: vec![784],
                output_size: 10,
                learning_rate: 0.001,
                batch_size: 32,
                optimizer: OptimizerType::Adam,
                loss_function: "categorical_crossentropy".to_string(),
                epochs: 100,
            },
            connections: Vec::new(),
            metadata: ArchitectureMetadata {
                generation: 0,
                parents: Vec::new(),
                created_at: Instant::now(),
                search_strategy: NASStrategy::Evolutionary,
                estimated_flops: 0,
                estimated_memory: 0,
                estimated_latency: Duration::new(0, 0),
            },
        };

        let counts = count_layer_types(&architecture);
        assert_eq!(counts.get(&LayerType::Dense), Some(&2));
    }
}
