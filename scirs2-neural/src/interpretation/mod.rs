//! Model interpretation utilities for neural networks
//!
//! This module provides comprehensive tools for understanding neural network decisions including:
//! - Gradient-based attribution methods (Saliency, Integrated Gradients, GradCAM)
//! - Feature visualization and analysis
//! - Layer activation analysis and statistics
//! - Decision explanation tools (LIME, SHAP, counterfactuals)
//! - Attention visualization for transformer models
//! - Comprehensive reporting and analysis
//!
//! # Module Organization
//!
//! - [`core`] - Main orchestrator (`ModelInterpreter`) and shared utilities
//! - [`attribution`] - Attribution methods and baseline handling
//! - [`analysis`] - Layer analysis and statistical evaluation
//! - [`explanations`] - Advanced explanation techniques (LIME, counterfactuals, concepts)
//! - [`visualization`] - Attention and feature visualization
//! - [`reporting`] - Report generation and unified interfaces
//!
//! # Basic Usage
//!
//! ```rust
//! use scirs2_neural::interpretation::{ModelInterpreter, AttributionMethod, BaselineMethod};
//! use ndarray::Array;
//!
//! // Create interpreter
//! let mut interpreter = ModelInterpreter::<f64>::new();
//!
//! // Add attribution methods
//! interpreter.add_attribution_method(AttributionMethod::Saliency);
//! interpreter.add_attribution_method(AttributionMethod::IntegratedGradients {
//!     baseline: BaselineMethod::Zero,
//!     num_steps: 50,
//! });
//!
//! // Cache gradients (normally computed by your model)
//! let gradients = Array::ones((3, 32, 32)).into_dyn();
//! interpreter.cache_gradients("input_gradient".to_string(), gradients);
//!
//! // Compute attributions
//! let input = Array::ones((3, 32, 32)).into_dyn();
//! let attribution = interpreter.compute_attribution(
//!     &AttributionMethod::Saliency,
//!     &input,
//!     Some(0), // target class
//! );
//!
//! // Generate comprehensive report
//! let report = interpreter.generate_report(&input);
//! ```

pub mod core;
pub mod attribution;
pub mod analysis;
pub mod explanations;
pub mod visualization;
pub mod reporting;

// Re-export main types and functions for backward compatibility
pub use core::ModelInterpreter;

// Attribution types and methods
pub use attribution::{
    AttributionMethod, BaselineMethod, LRPRule,
    compute_saliency_attribution, compute_integrated_gradients, compute_gradcam_attribution,
    compute_guided_backprop_attribution, compute_deeplift_attribution, compute_shap_attribution,
    compute_lrp_attribution, create_baseline,
};

// Analysis types and functions
pub use analysis::{
    LayerAnalysisStats, AttributionStatistics, InterpretationSummary,
    analyze_layer_activations, compute_layer_statistics, compute_attribution_statistics,
    generate_interpretation_summary, find_most_important_features, compute_interpretation_confidence,
    compute_correlation, analyze_activation_distribution, compare_layer_statistics,
};

// Explanation types and functions
pub use explanations::{
    CounterfactualGenerator, ConceptActivationVector, LIMEExplainer, AdversarialExplanation,
    DistanceMetric, generate_adversarial_explanation, compute_concept_vectors,
};

// Visualization types and functions
pub use visualization::{
    AttentionVisualizer, VisualizationMethod, AttentionAggregation, VisualizationResult,
    NetworkDissectionResult, generate_feature_visualization, perform_network_dissection,
    create_attention_heatmap,
};

// Reporting types and functions
pub use reporting::{
    InterpretationReport, ComprehensiveInterpretationReport, ConfidenceEstimates, RobustnessAnalysis,
    generate_comprehensive_report, generate_basic_report, generate_report_summary, export_report_data,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, ArrayD};

    #[test]
    fn test_interpretation_module_integration() {
        // Test that all modules work together
        let mut interpreter = ModelInterpreter::<f64>::new();
        
        // Add attribution methods
        interpreter.add_attribution_method(AttributionMethod::Saliency);
        interpreter.add_attribution_method(AttributionMethod::IntegratedGradients {
            baseline: BaselineMethod::Zero,
            num_steps: 10,
        });
        
        // Cache some test data
        let gradients = Array::ones((3, 4, 4)).into_dyn();
        interpreter.cache_gradients("input_gradient".to_string(), gradients);
        
        let activations = Array::from_elem((3, 4, 4), 0.5).into_dyn();
        interpreter.cache_activations("test_layer".to_string(), activations);
        
        // Test attribution computation
        let input = Array::ones((3, 4, 4)).into_dyn();
        let saliency_result = interpreter.compute_attribution(
            &AttributionMethod::Saliency,
            &input,
            Some(0),
        );
        assert!(saliency_result.is_ok());
        
        // Test layer analysis
        let analysis_result = interpreter.analyze_layer_activations("test_layer");
        assert!(analysis_result.is_ok());
        
        // Test report generation
        let report = interpreter.generate_report(&input);
        assert!(report.is_ok());
    }

    #[test]
    fn test_counterfactual_integration() {
        let mut interpreter = ModelInterpreter::<f64>::new();
        
        // Set up counterfactual generator
        let cf_generator = CounterfactualGenerator::new(
            5,
            0.01,
            50,
            DistanceMetric::L2,
        );
        interpreter.set_counterfactual_generator(cf_generator);
        
        assert!(interpreter.counterfactual_generator().is_some());
    }

    #[test]
    fn test_attention_integration() {
        let mut interpreter = ModelInterpreter::<f64>::new();
        
        // Set up attention visualizer
        let attention_viz = AttentionVisualizer::new(
            8,
            512,
            AttentionAggregation::Average,
            vec!["layer1".to_string()],
        );
        interpreter.set_attention_visualizer(attention_viz);
        
        assert!(interpreter.attention_visualizer().is_some());
    }

    #[test]
    fn test_lime_integration() {
        let mut interpreter = ModelInterpreter::<f64>::new();
        
        // Set up LIME explainer
        let lime_explainer = LIMEExplainer::new(
            100,
            0.1,
            0.01,
            42,
        );
        interpreter.set_lime_explainer(lime_explainer);
        
        assert!(interpreter.lime_explainer().is_some());
    }

    #[test]
    fn test_concept_vectors() {
        let mut interpreter = ModelInterpreter::<f64>::new();
        
        // Add concept activation vector
        let activation_vector = Array::ones((10,)).into_dyn();
        let concept = ConceptActivationVector::new(
            "test_concept".to_string(),
            "conv1".to_string(),
            activation_vector,
            0.9,
            100,
        );
        
        interpreter.add_concept_vector("test_concept".to_string(), concept);
        assert!(interpreter.get_concept_vector("test_concept").is_some());
    }

    #[test]
    fn test_comprehensive_workflow() {
        // Test a comprehensive interpretation workflow
        let mut interpreter = ModelInterpreter::<f64>::new();
        
        // Set up multiple attribution methods
        interpreter.add_attribution_method(AttributionMethod::Saliency);
        interpreter.add_attribution_method(AttributionMethod::GradCAM {
            target_layer: "conv5".to_string(),
        });
        
        // Cache data for different layers
        let input_grad = Array::from_elem((3, 32, 32), 0.1).into_dyn();
        interpreter.cache_gradients("input_gradient".to_string(), input_grad);
        
        let conv5_activations = Array::from_elem((8, 16, 16), 0.5).into_dyn();
        let conv5_gradients = Array::from_elem((8, 16, 16), 0.2).into_dyn();
        interpreter.cache_activations("conv5".to_string(), conv5_activations);
        interpreter.cache_gradients("conv5".to_string(), conv5_gradients);
        
        // Test input
        let input = Array::ones((3, 32, 32)).into_dyn();
        
        // Compute multiple attributions
        let saliency = interpreter.compute_attribution(
            &AttributionMethod::Saliency,
            &input,
            Some(1),
        );
        assert!(saliency.is_ok());
        
        let gradcam = interpreter.compute_attribution(
            &AttributionMethod::GradCAM {
                target_layer: "conv5".to_string(),
            },
            &input,
            Some(1),
        );
        assert!(gradcam.is_ok());
        
        // Analyze layers
        let layer_analysis = interpreter.analyze_layer_activations("conv5");
        assert!(layer_analysis.is_ok());
        
        // Generate comprehensive report
        let report = interpreter.generate_report(&input);
        assert!(report.is_ok());
        
        let comp_report = report.unwrap();
        assert!(comp_report.basic_report.attributions.len() >= 2);
        assert!(comp_report.basic_report.layer_statistics.contains_key("conv5"));
        assert!(comp_report.confidence_estimates.overall_confidence > 0.0);
    }

    #[test]
    fn test_error_handling() {
        let interpreter = ModelInterpreter::<f64>::new();
        let input = Array::ones((3, 32, 32)).into_dyn();
        
        // Test attribution without cached gradients
        let result = interpreter.compute_attribution(
            &AttributionMethod::Saliency,
            &input,
            Some(0),
        );
        assert!(result.is_ok()); // Should return placeholder attribution
        
        // Test GradCAM without cached data
        let gradcam_result = interpreter.compute_attribution(
            &AttributionMethod::GradCAM {
                target_layer: "nonexistent_layer".to_string(),
            },
            &input,
            Some(0),
        );
        assert!(gradcam_result.is_err()); // Should error for missing layer data
    }

    #[test]
    fn test_baseline_methods() {
        // Test different baseline methods
        let input: ArrayD<f64> = Array::ones((2, 3, 3)).into_dyn();
        
        let zero_baseline = create_baseline(&input, &BaselineMethod::Zero);
        assert!(zero_baseline.is_ok());
        assert_eq!(zero_baseline.unwrap().sum(), 0.0);
        
        let random_baseline = create_baseline(&input, &BaselineMethod::Random { seed: 42 });
        assert!(random_baseline.is_ok());
        
        let custom_data = Array::from_elem((2, 3, 3), 0.5_f32).into_dyn();
        let custom_baseline = create_baseline(&input, &BaselineMethod::Custom(custom_data));
        assert!(custom_baseline.is_ok());
    }

    #[test]
    fn test_visualization_methods() {
        // Test feature visualization
        let method = VisualizationMethod::ActivationMaximization {
            target_layer: "conv1".to_string(),
            target_unit: Some(5),
            num_iterations: 10,
            learning_rate: 0.01,
        };
        
        let result = generate_feature_visualization::<f64>(&method, &[3, 16, 16]);
        assert!(result.is_ok());
        
        let viz_result = result.unwrap();
        assert_eq!(viz_result.visualization_data.shape(), &[3, 16, 16]);
        assert!(viz_result.metadata.contains_key("target_layer"));
    }
}