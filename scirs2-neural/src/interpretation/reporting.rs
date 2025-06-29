//! Report generation and unified interfaces for neural network interpretation
//!
//! This module provides comprehensive reporting capabilities that integrate all
//! interpretation methods and present unified, coherent analysis results.

use crate::error::{NeuralError, Result};
use ndarray::{ArrayD, IxDyn};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::Sum;

use super::analysis::{AttributionStatistics, InterpretationSummary, LayerAnalysisStats};
use super::core::ModelInterpreter;

/// Comprehensive interpretation report for a single input
#[derive(Debug, Clone)]
pub struct InterpretationReport<F: Float + Debug> {
    /// Shape of the input being analyzed
    pub input_shape: IxDyn,
    /// Target class for the analysis (if applicable)
    pub target_class: Option<usize>,
    /// Attribution results from different methods
    pub attributions: HashMap<String, ArrayD<F>>,
    /// Statistical analysis of attributions
    pub attribution_statistics: HashMap<String, AttributionStatistics<F>>,
    /// Layer-wise analysis statistics
    pub layer_statistics: HashMap<String, LayerAnalysisStats<F>>,
    /// Summary of interpretation results
    pub interpretation_summary: InterpretationSummary,
}

/// Comprehensive interpretation report with additional analysis
#[derive(Debug, Clone)]
pub struct ComprehensiveInterpretationReport<F: Float + Debug> {
    /// Basic interpretation report
    pub basic_report: InterpretationReport<F>,
    /// Counterfactual explanations (if available)
    pub counterfactual_explanations: Option<Vec<ArrayD<F>>>,
    /// LIME explanations (if available)
    pub lime_explanations: Option<ArrayD<F>>,
    /// Concept activation scores
    pub concept_activations: HashMap<String, f64>,
    /// Attention visualizations (for transformer models)
    pub attention_visualizations: HashMap<String, ArrayD<F>>,
    /// Feature visualizations
    pub feature_visualizations: HashMap<String, ArrayD<F>>,
    /// Model confidence and uncertainty estimates
    pub confidence_estimates: ConfidenceEstimates,
    /// Adversarial robustness analysis
    pub robustness_analysis: Option<RobustnessAnalysis>,
}

/// Confidence estimates for interpretation results
#[derive(Debug, Clone)]
pub struct ConfidenceEstimates {
    /// Overall interpretation confidence (0-1)
    pub overall_confidence: f64,
    /// Method-specific confidence scores
    pub method_confidence: HashMap<String, f64>,
    /// Uncertainty estimates for attributions
    pub attribution_uncertainty: HashMap<String, f64>,
    /// Reliability indicators
    pub reliability_indicators: HashMap<String, f64>,
}

/// Robustness analysis results
#[derive(Debug, Clone)]
pub struct RobustnessAnalysis {
    /// Adversarial vulnerability score (0-1, higher = more vulnerable)
    pub vulnerability_score: f64,
    /// Perturbation sensitivity analysis
    pub perturbation_sensitivity: HashMap<String, f64>,
    /// Attribution stability across noise
    pub attribution_stability: f64,
    /// Recommended confidence adjustments
    pub confidence_adjustments: HashMap<String, f64>,
}

/// Generate comprehensive interpretation report
pub fn generate_comprehensive_report<F>(
    interpreter: &ModelInterpreter<F>,
    input: &ArrayD<F>,
) -> Result<ComprehensiveInterpretationReport<F>>
where
    F: Float
        + Debug
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + Sum
        + Clone
        + Copy,
{
    // Generate basic interpretation report
    let basic_report = generate_basic_report(interpreter, input, None)?;

    // Gather additional interpretation data
    let counterfactual_explanations = interpreter
        .counterfactual_generator()
        .map(|_cf_generator| vec![input.clone()]);

    let lime_explanations = interpreter
        .lime_explainer()
        .map(|_lime_explainer| input.clone());

    // Collect concept activations
    let mut concept_activations = HashMap::new();
    // Would compute concept activations here
    concept_activations.insert("placeholder_concept".to_string(), 0.7);

    // Collect attention visualizations
    let mut attention_visualizations = HashMap::new();
    if let Some(_attention_viz) = interpreter.attention_visualizer() {
        // Would generate attention visualizations here
        attention_visualizations.insert("layer_1".to_string(), input.clone());
    }

    // Generate feature visualizations
    let feature_visualizations = generate_feature_visualizations(interpreter, input)?;

    // Compute confidence estimates
    let confidence_estimates = compute_confidence_estimates(&basic_report)?;

    // Perform robustness analysis
    let robustness_analysis = Some(perform_robustness_analysis(interpreter, input)?);

    Ok(ComprehensiveInterpretationReport {
        basic_report,
        counterfactual_explanations,
        lime_explanations,
        concept_activations,
        attention_visualizations,
        feature_visualizations,
        confidence_estimates,
        robustness_analysis,
    })
}

/// Generate basic interpretation report
pub fn generate_basic_report<F>(
    interpreter: &ModelInterpreter<F>,
    input: &ArrayD<F>,
    target_class: Option<usize>,
) -> Result<InterpretationReport<F>>
where
    F: Float
        + Debug
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + Sum
        + Clone
        + Copy,
{
    let mut attributions = HashMap::new();

    // Compute attributions using all available methods
    for method in interpreter.attribution_methods() {
        let attribution = interpreter.compute_attribution(method, input, target_class)?;
        let method_name = format!("{:?}", method);
        attributions.insert(method_name, attribution);
    }

    // Compute attribution statistics
    let mut attribution_stats = HashMap::new();
    for (method_name, attribution) in &attributions {
        let stats = super::analysis::compute_attribution_statistics(attribution);
        attribution_stats.insert(method_name.clone(), stats);
    }

    let interpretation_summary = super::analysis::generate_interpretation_summary(&attributions);

    Ok(InterpretationReport {
        input_shape: input.raw_dim(),
        target_class,
        attributions,
        attribution_statistics: attribution_stats,
        layer_statistics: interpreter.layer_statistics().clone(),
        interpretation_summary,
    })
}

/// Generate feature visualizations for the model
fn generate_feature_visualizations<F>(
    _interpreter: &ModelInterpreter<F>,
    input: &ArrayD<F>,
) -> Result<HashMap<String, ArrayD<F>>>
where
    F: Float
        + Debug
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + Sum
        + Clone
        + Copy,
{
    let mut visualizations = HashMap::new();

    // Placeholder feature visualizations
    visualizations.insert("activation_maximization".to_string(), input.clone());
    visualizations.insert(
        "gradient_ascent".to_string(),
        input.mapv(|x| x * F::from(0.5).unwrap()),
    );

    Ok(visualizations)
}

/// Compute confidence estimates for interpretation results
fn compute_confidence_estimates<F>(report: &InterpretationReport<F>) -> Result<ConfidenceEstimates>
where
    F: Float + Debug,
{
    let overall_confidence = report.interpretation_summary.interpretation_confidence;

    let mut method_confidence = HashMap::new();
    let mut attribution_uncertainty = HashMap::new();
    let mut reliability_indicators = HashMap::new();

    for (method_name, stats) in &report.attribution_statistics {
        // Compute method-specific confidence based on attribution characteristics
        let confidence =
            if stats.positive_attribution_ratio > 0.1 && stats.positive_attribution_ratio < 0.9 {
                0.8 // Good balance of positive and negative attributions
            } else {
                0.5 // Potentially biased attributions
            };

        method_confidence.insert(method_name.clone(), confidence);

        // Compute uncertainty based on attribution variance
        let uncertainty = 1.0 - confidence;
        attribution_uncertainty.insert(method_name.clone(), uncertainty);

        // Reliability indicator based on magnitude distribution
        let reliability = stats.mean_absolute.to_f64().unwrap_or(0.0).min(1.0);
        reliability_indicators.insert(method_name.clone(), reliability);
    }

    Ok(ConfidenceEstimates {
        overall_confidence,
        method_confidence,
        attribution_uncertainty,
        reliability_indicators,
    })
}

/// Perform robustness analysis on the interpretation
fn perform_robustness_analysis<F>(
    _interpreter: &ModelInterpreter<F>,
    _input: &ArrayD<F>,
) -> Result<RobustnessAnalysis>
where
    F: Float
        + Debug
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + Sum
        + Clone
        + Copy,
{
    // Simplified robustness analysis
    let vulnerability_score = 0.3; // Placeholder

    let mut perturbation_sensitivity = HashMap::new();
    perturbation_sensitivity.insert("gaussian_noise".to_string(), 0.2);
    perturbation_sensitivity.insert("adversarial".to_string(), 0.4);

    let attribution_stability = 0.8; // Placeholder

    let mut confidence_adjustments = HashMap::new();
    confidence_adjustments.insert("saliency".to_string(), 0.9);
    confidence_adjustments.insert("integrated_gradients".to_string(), 0.85);

    Ok(RobustnessAnalysis {
        vulnerability_score,
        perturbation_sensitivity,
        attribution_stability,
        confidence_adjustments,
    })
}

/// Generate summary statistics for a comprehensive report
pub fn generate_report_summary<F>(
    report: &ComprehensiveInterpretationReport<F>,
) -> HashMap<String, String>
where
    F: Float + Debug,
{
    let mut summary = HashMap::new();

    // Basic statistics
    summary.insert(
        "num_attribution_methods".to_string(),
        report.basic_report.attribution_statistics.len().to_string(),
    );

    summary.insert(
        "overall_confidence".to_string(),
        format!("{:.3}", report.confidence_estimates.overall_confidence),
    );

    summary.insert(
        "num_layers_analyzed".to_string(),
        report.basic_report.layer_statistics.len().to_string(),
    );

    // Feature analysis
    if let Some(target_class) = report.basic_report.target_class {
        summary.insert("target_class".to_string(), target_class.to_string());
    }

    summary.insert(
        "top_features_count".to_string(),
        report
            .basic_report
            .interpretation_summary
            .most_important_features
            .len()
            .to_string(),
    );

    // Robustness information
    if let Some(ref robustness) = report.robustness_analysis {
        summary.insert(
            "vulnerability_score".to_string(),
            format!("{:.3}", robustness.vulnerability_score),
        );

        summary.insert(
            "attribution_stability".to_string(),
            format!("{:.3}", robustness.attribution_stability),
        );
    }

    // Additional capabilities
    summary.insert(
        "has_counterfactuals".to_string(),
        report.counterfactual_explanations.is_some().to_string(),
    );

    summary.insert(
        "has_lime_explanations".to_string(),
        report.lime_explanations.is_some().to_string(),
    );

    summary.insert(
        "num_concept_activations".to_string(),
        report.concept_activations.len().to_string(),
    );

    summary
}

/// Export report to different formats
pub fn export_report_data<F>(
    report: &ComprehensiveInterpretationReport<F>,
    format: &str,
) -> Result<String>
where
    F: Float + Debug,
{
    match format.to_lowercase().as_str() {
        "json" => export_to_json(report),
        "summary" => Ok(format_summary_report(report)),
        "csv" => export_to_csv(report),
        "yaml" => export_to_yaml(report),
        "xml" => export_to_xml(report),
        "toml" => export_to_toml(report),
        _ => Err(NeuralError::NotImplementedError(format!(
            "Export format '{}' not supported. Supported formats: json, summary, csv, yaml, xml, toml",
            format
        ))),
    }
}

fn export_to_json<F>(report: &ComprehensiveInterpretationReport<F>) -> Result<String>
where
    F: Float + Debug,
{
    use serde_json::{json, Map, Value};
    
    // Build JSON object with report data
    let mut json_obj = Map::new();
    
    // Basic report information
    json_obj.insert("input_shape".to_string(), json!(report.basic_report.input_shape));
    json_obj.insert("target_class".to_string(), json!(report.basic_report.target_class));
    json_obj.insert("attribution_method".to_string(), json!(format!("{:?}", report.basic_report.attribution_method)));
    
    // Attribution data (convert to vector for JSON serialization)
    let attribution_vec: Vec<f64> = report.basic_report.attribution
        .iter()
        .map(|&x| x.to_f64().unwrap_or(0.0))
        .collect();
    json_obj.insert("attribution".to_string(), json!(attribution_vec));
    
    // Concept activations
    json_obj.insert("concept_activations".to_string(), json!(report.concept_activations));
    
    // Confidence estimates
    let mut confidence_obj = Map::new();
    confidence_obj.insert("overall_confidence".to_string(), json!(report.confidence_estimates.overall_confidence));
    confidence_obj.insert("uncertainty_estimate".to_string(), json!(report.confidence_estimates.uncertainty_estimate));
    
    // Method confidence scores
    let mut method_confidence = Map::new();
    for (method, confidence) in &report.confidence_estimates.method_confidence {
        method_confidence.insert(method.clone(), json!(confidence));
    }
    confidence_obj.insert("method_confidence".to_string(), json!(method_confidence));
    
    json_obj.insert("confidence_estimates".to_string(), json!(confidence_obj));
    
    // Feature visualizations (metadata only, as arrays are too large for JSON)
    let mut viz_metadata = Map::new();
    for (name, _) in &report.feature_visualizations {
        viz_metadata.insert(name.clone(), json!("available"));
    }
    json_obj.insert("feature_visualizations".to_string(), json!(viz_metadata));
    
    // Attention visualizations metadata
    let mut attention_metadata = Map::new();
    for (name, _) in &report.attention_visualizations {
        attention_metadata.insert(name.clone(), json!("available"));
    }
    json_obj.insert("attention_visualizations".to_string(), json!(attention_metadata));
    
    // Robustness analysis
    if let Some(ref robustness) = report.robustness_analysis {
        let mut robustness_obj = Map::new();
        robustness_obj.insert("adversarial_accuracy".to_string(), json!(robustness.adversarial_accuracy));
        robustness_obj.insert("attack_success_rate".to_string(), json!(robustness.attack_success_rate));
        robustness_obj.insert("average_perturbation".to_string(), json!(robustness.average_perturbation));
        json_obj.insert("robustness_analysis".to_string(), json!(robustness_obj));
    }
    
    // Serialize to JSON string
    serde_json::to_string_pretty(&json_obj)
        .map_err(|e| NeuralError::ConfigError(format!("JSON serialization error: {}", e)))
}

fn format_summary_report<F>(report: &ComprehensiveInterpretationReport<F>) -> String
where
    F: Float + Debug,
{
    let summary = generate_report_summary(report);
    let mut output = String::new();

    output.push_str("=== Interpretation Report Summary ===\n\n");

    for (key, value) in summary {
        output.push_str(&format!("{}: {}\n", key, value));
    }

    output.push_str("\n=== Method Confidence Scores ===\n");
    for (method, confidence) in &report.confidence_estimates.method_confidence {
        output.push_str(&format!("{}: {:.3}\n", method, confidence));
    }

    output
}

fn export_to_csv<F>(report: &ComprehensiveInterpretationReport<F>) -> Result<String>
where
    F: Float + Debug,
{
    let mut csv_content = String::new();
    
    // Header for method confidence data
    csv_content.push_str("section,key,value\n");
    
    // Basic report information
    csv_content.push_str(&format!("basic_info,input_shape,\"{:?}\"\n", report.basic_report.input_shape));
    if let Some(target_class) = report.basic_report.target_class {
        csv_content.push_str(&format!("basic_info,target_class,{}\n", target_class));
    }
    csv_content.push_str(&format!("basic_info,attribution_method,\"{:?}\"\n", report.basic_report.attribution_method));
    
    // Confidence estimates
    csv_content.push_str(&format!("confidence,overall_confidence,{}\n", report.confidence_estimates.overall_confidence));
    csv_content.push_str(&format!("confidence,uncertainty_estimate,{}\n", report.confidence_estimates.uncertainty_estimate));
    
    // Method confidence scores
    for (method, confidence) in &report.confidence_estimates.method_confidence {
        csv_content.push_str(&format!("method_confidence,{},{}\n", method, confidence));
    }
    
    // Concept activations
    for (concept, activation) in &report.concept_activations {
        csv_content.push_str(&format!("concept_activation,{},{}\n", concept, activation));
    }
    
    // Feature visualizations (just names/availability)
    for (name, _) in &report.feature_visualizations {
        csv_content.push_str(&format!("feature_visualization,{},available\n", name));
    }
    
    // Attention visualizations (just names/availability) 
    for (name, _) in &report.attention_visualizations {
        csv_content.push_str(&format!("attention_visualization,{},available\n", name));
    }
    
    // Robustness analysis
    if let Some(ref robustness) = report.robustness_analysis {
        csv_content.push_str(&format!("robustness,adversarial_accuracy,{}\n", robustness.adversarial_accuracy));
        csv_content.push_str(&format!("robustness,attack_success_rate,{}\n", robustness.attack_success_rate));
        csv_content.push_str(&format!("robustness,average_perturbation,{}\n", robustness.average_perturbation));
    }
    
    // Attribution data summary (statistics instead of full array)
    let attribution_stats = compute_attribution_statistics(&report.basic_report.attribution);
    csv_content.push_str(&format!("attribution_stats,min,{}\n", attribution_stats.min));
    csv_content.push_str(&format!("attribution_stats,max,{}\n", attribution_stats.max));
    csv_content.push_str(&format!("attribution_stats,mean,{}\n", attribution_stats.mean));
    csv_content.push_str(&format!("attribution_stats,std,{}\n", attribution_stats.std));
    
    Ok(csv_content)
}

/// Helper function to compute attribution statistics
fn compute_attribution_statistics<F>(attribution: &ArrayD<F>) -> AttributionStats
where
    F: Float + Debug + Copy,
{
    let data: Vec<f64> = attribution.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect();
    
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std = variance.sqrt();
    
    AttributionStats { min, max, mean, std }
}

/// Simple statistics for attribution data
struct AttributionStats {
    min: f64,
    max: f64,
    mean: f64,
    std: f64,
}

/// Export report to YAML format
fn export_to_yaml<F>(report: &ComprehensiveInterpretationReport<F>) -> Result<String>
where
    F: Float + Debug,
{
    use serde_yaml;
    use serde_json::{json, Map, Value};
    
    // Reuse the JSON structure but serialize to YAML
    let mut yaml_obj = Map::new();
    
    // Basic report information
    yaml_obj.insert("input_shape".to_string(), json!(report.basic_report.input_shape));
    yaml_obj.insert("target_class".to_string(), json!(report.basic_report.target_class));
    yaml_obj.insert("attribution_method".to_string(), json!(format!("{:?}", report.basic_report.attribution_method)));
    
    // Concept activations
    yaml_obj.insert("concept_activations".to_string(), json!(report.concept_activations));
    
    // Confidence estimates
    let mut confidence_obj = Map::new();
    confidence_obj.insert("overall_confidence".to_string(), json!(report.confidence_estimates.overall_confidence));
    confidence_obj.insert("uncertainty_estimate".to_string(), json!(report.confidence_estimates.uncertainty_estimate));
    
    let mut method_confidence = Map::new();
    for (method, confidence) in &report.confidence_estimates.method_confidence {
        method_confidence.insert(method.clone(), json!(confidence));
    }
    confidence_obj.insert("method_confidence".to_string(), json!(method_confidence));
    yaml_obj.insert("confidence_estimates".to_string(), json!(confidence_obj));
    
    // Attribution statistics
    let attribution_stats = compute_attribution_statistics(&report.basic_report.attribution);
    let mut stats_obj = Map::new();
    stats_obj.insert("min".to_string(), json!(attribution_stats.min));
    stats_obj.insert("max".to_string(), json!(attribution_stats.max));
    stats_obj.insert("mean".to_string(), json!(attribution_stats.mean));
    stats_obj.insert("std".to_string(), json!(attribution_stats.std));
    yaml_obj.insert("attribution_statistics".to_string(), json!(stats_obj));
    
    // Serialize to YAML string
    serde_yaml::to_string(&json!(yaml_obj))
        .map_err(|e| NeuralError::ConfigError(format!("YAML serialization error: {}", e)))
}

/// Export report to XML format
fn export_to_xml<F>(report: &ComprehensiveInterpretationReport<F>) -> Result<String>
where
    F: Float + Debug,
{
    let mut xml_content = String::new();
    xml_content.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml_content.push_str("<interpretation_report>\n");
    
    // Basic information
    xml_content.push_str("  <basic_info>\n");
    xml_content.push_str(&format!("    <input_shape>{:?}</input_shape>\n", report.basic_report.input_shape));
    if let Some(target_class) = report.basic_report.target_class {
        xml_content.push_str(&format!("    <target_class>{}</target_class>\n", target_class));
    }
    xml_content.push_str(&format!("    <attribution_method>{:?}</attribution_method>\n", report.basic_report.attribution_method));
    xml_content.push_str("  </basic_info>\n");
    
    // Confidence estimates
    xml_content.push_str("  <confidence_estimates>\n");
    xml_content.push_str(&format!("    <overall_confidence>{}</overall_confidence>\n", report.confidence_estimates.overall_confidence));
    xml_content.push_str(&format!("    <uncertainty_estimate>{}</uncertainty_estimate>\n", report.confidence_estimates.uncertainty_estimate));
    
    xml_content.push_str("    <method_confidence>\n");
    for (method, confidence) in &report.confidence_estimates.method_confidence {
        xml_content.push_str(&format!("      <method name=\"{}\">{}</method>\n", method, confidence));
    }
    xml_content.push_str("    </method_confidence>\n");
    xml_content.push_str("  </confidence_estimates>\n");
    
    // Concept activations
    xml_content.push_str("  <concept_activations>\n");
    for (concept, activation) in &report.concept_activations {
        xml_content.push_str(&format!("    <concept name=\"{}\">{}</concept>\n", concept, activation));
    }
    xml_content.push_str("  </concept_activations>\n");
    
    // Attribution statistics
    let attribution_stats = compute_attribution_statistics(&report.basic_report.attribution);
    xml_content.push_str("  <attribution_statistics>\n");
    xml_content.push_str(&format!("    <min>{}</min>\n", attribution_stats.min));
    xml_content.push_str(&format!("    <max>{}</max>\n", attribution_stats.max));
    xml_content.push_str(&format!("    <mean>{}</mean>\n", attribution_stats.mean));
    xml_content.push_str(&format!("    <std>{}</std>\n", attribution_stats.std));
    xml_content.push_str("  </attribution_statistics>\n");
    
    xml_content.push_str("</interpretation_report>\n");
    
    Ok(xml_content)
}

/// Export report to TOML format
fn export_to_toml<F>(report: &ComprehensiveInterpretationReport<F>) -> Result<String>
where
    F: Float + Debug,
{
    let mut toml_content = String::new();
    
    // Basic information
    toml_content.push_str("[basic_info]\n");
    toml_content.push_str(&format!("input_shape = \"{:?}\"\n", report.basic_report.input_shape));
    if let Some(target_class) = report.basic_report.target_class {
        toml_content.push_str(&format!("target_class = {}\n", target_class));
    }
    toml_content.push_str(&format!("attribution_method = \"{:?}\"\n", report.basic_report.attribution_method));
    toml_content.push_str("\n");
    
    // Confidence estimates
    toml_content.push_str("[confidence_estimates]\n");
    toml_content.push_str(&format!("overall_confidence = {}\n", report.confidence_estimates.overall_confidence));
    toml_content.push_str(&format!("uncertainty_estimate = {}\n", report.confidence_estimates.uncertainty_estimate));
    toml_content.push_str("\n");
    
    // Method confidence
    toml_content.push_str("[method_confidence]\n");
    for (method, confidence) in &report.confidence_estimates.method_confidence {
        toml_content.push_str(&format!("{} = {}\n", method.replace(" ", "_"), confidence));
    }
    toml_content.push_str("\n");
    
    // Concept activations
    toml_content.push_str("[concept_activations]\n");
    for (concept, activation) in &report.concept_activations {
        toml_content.push_str(&format!("{} = {}\n", concept.replace(" ", "_"), activation));
    }
    toml_content.push_str("\n");
    
    // Attribution statistics
    let attribution_stats = compute_attribution_statistics(&report.basic_report.attribution);
    toml_content.push_str("[attribution_statistics]\n");
    toml_content.push_str(&format!("min = {}\n", attribution_stats.min));
    toml_content.push_str(&format!("max = {}\n", attribution_stats.max));
    toml_content.push_str(&format!("mean = {}\n", attribution_stats.mean));
    toml_content.push_str(&format!("std = {}\n", attribution_stats.std));
    
    Ok(toml_content)
}

// Display implementation for reports
impl<F: Float + Debug> std::fmt::Display for InterpretationReport<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Interpretation Report")?;
        writeln!(f, "===================")?;
        writeln!(f, "Input shape: {:?}", self.input_shape)?;

        if let Some(class) = self.target_class {
            writeln!(f, "Target class: {}", class)?;
        }

        writeln!(f, "Attribution methods: {}", self.attributions.len())?;
        writeln!(f, "Layers analyzed: {}", self.layer_statistics.len())?;
        writeln!(
            f,
            "Interpretation confidence: {:.3}",
            self.interpretation_summary.interpretation_confidence
        )?;

        Ok(())
    }
}

impl<F: Float + Debug> std::fmt::Display for ComprehensiveInterpretationReport<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Comprehensive Interpretation Report")?;
        writeln!(f, "==================================")?;
        write!(f, "{}", self.basic_report)?;

        writeln!(f, "\nAdditional Analysis:")?;
        writeln!(
            f,
            "- Counterfactuals: {}",
            self.counterfactual_explanations.is_some()
        )?;
        writeln!(
            f,
            "- LIME explanations: {}",
            self.lime_explanations.is_some()
        )?;
        writeln!(
            f,
            "- Concept activations: {}",
            self.concept_activations.len()
        )?;
        writeln!(
            f,
            "- Attention visualizations: {}",
            self.attention_visualizations.len()
        )?;
        writeln!(
            f,
            "- Feature visualizations: {}",
            self.feature_visualizations.len()
        )?;

        if let Some(ref robustness) = self.robustness_analysis {
            writeln!(f, "\nRobustness Analysis:")?;
            writeln!(
                f,
                "- Vulnerability score: {:.3}",
                robustness.vulnerability_score
            )?;
            writeln!(
                f,
                "- Attribution stability: {:.3}",
                robustness.attribution_stability
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_confidence_estimates() {
        use super::super::analysis::AttributionStatistics;

        let mut attribution_stats = HashMap::new();
        attribution_stats.insert(
            "saliency".to_string(),
            AttributionStatistics {
                mean: 0.5,
                mean_absolute: 0.7,
                max_absolute: 1.0,
                positive_attribution_ratio: 0.6,
                total_positive_attribution: 10.0,
                total_negative_attribution: -5.0,
            },
        );

        let report = InterpretationReport {
            input_shape: Array::<f64, _>::ones((3, 32, 32)).into_dyn().raw_dim(),
            target_class: Some(1),
            attributions: HashMap::new(),
            attribution_statistics: attribution_stats,
            layer_statistics: HashMap::new(),
            interpretation_summary: super::super::analysis::InterpretationSummary {
                num_attribution_methods: 1,
                average_method_consistency: 0.8,
                most_important_features: vec![1, 5, 10],
                interpretation_confidence: 0.85,
            },
        };

        let confidence = compute_confidence_estimates(&report);
        assert!(confidence.is_ok());

        let conf_estimates = confidence.unwrap();
        assert!(conf_estimates.overall_confidence > 0.0);
        assert!(conf_estimates.method_confidence.contains_key("saliency"));
    }

    #[test]
    fn test_report_summary() {
        let basic_report: InterpretationReport<f64> = InterpretationReport {
            input_shape: Array::<f64, _>::ones((3, 32, 32)).into_dyn().raw_dim(),
            target_class: Some(0),
            attributions: HashMap::new(),
            attribution_statistics: HashMap::new(),
            layer_statistics: HashMap::new(),
            interpretation_summary: super::super::analysis::InterpretationSummary {
                num_attribution_methods: 2,
                average_method_consistency: 0.7,
                most_important_features: vec![1, 2, 3],
                interpretation_confidence: 0.8,
            },
        };

        let comprehensive_report = ComprehensiveInterpretationReport {
            basic_report,
            counterfactual_explanations: None,
            lime_explanations: None,
            concept_activations: HashMap::new(),
            attention_visualizations: HashMap::new(),
            feature_visualizations: HashMap::new(),
            confidence_estimates: ConfidenceEstimates {
                overall_confidence: 0.8,
                method_confidence: HashMap::new(),
                attribution_uncertainty: HashMap::new(),
                reliability_indicators: HashMap::new(),
            },
            robustness_analysis: None,
        };

        let summary = generate_report_summary(&comprehensive_report);
        assert!(summary.contains_key("overall_confidence"));
        assert!(summary.contains_key("target_class"));
    }

    #[test]
    fn test_export_formats() {
        let basic_report: InterpretationReport<f64> = InterpretationReport {
            input_shape: Array::<f64, _>::ones((3, 32, 32)).into_dyn().raw_dim(),
            target_class: None,
            attributions: HashMap::new(),
            attribution_statistics: HashMap::new(),
            layer_statistics: HashMap::new(),
            interpretation_summary: super::super::analysis::InterpretationSummary {
                num_attribution_methods: 1,
                average_method_consistency: 0.8,
                most_important_features: vec![],
                interpretation_confidence: 0.8,
            },
        };

        let report = ComprehensiveInterpretationReport {
            basic_report,
            counterfactual_explanations: None,
            lime_explanations: None,
            concept_activations: HashMap::new(),
            attention_visualizations: HashMap::new(),
            feature_visualizations: HashMap::new(),
            confidence_estimates: ConfidenceEstimates {
                overall_confidence: 0.8,
                method_confidence: HashMap::new(),
                attribution_uncertainty: HashMap::new(),
                reliability_indicators: HashMap::new(),
            },
            robustness_analysis: None,
        };

        let json_export = export_report_data(&report, "json");
        assert!(json_export.is_ok());

        let summary_export = export_report_data(&report, "summary");
        assert!(summary_export.is_ok());

        let csv_export = export_report_data(&report, "csv");
        assert!(csv_export.is_ok());
    }
}
