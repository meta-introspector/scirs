//! Model validation and compatibility checking between frameworks

use crate::error::Result;
use crate::ml_framework::{DataType, MLFramework, MLModel};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Model validator for checking compatibility between frameworks
pub struct ModelValidator {
    source_framework: MLFramework,
    target_framework: MLFramework,
    validation_config: ValidationConfig,
}

#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub check_data_types: bool,
    pub check_tensor_shapes: bool,
    pub check_operations: bool,
    pub check_metadata: bool,
    pub strict_mode: bool,
    pub allow_type_conversion: bool,
    pub max_shape_dimension: Option<usize>,
    pub supported_dtypes: Option<HashSet<DataType>>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            check_data_types: true,
            check_tensor_shapes: true,
            check_operations: true,
            check_metadata: true,
            strict_mode: false,
            allow_type_conversion: true,
            max_shape_dimension: Some(8), // Most frameworks support up to 8D tensors
            supported_dtypes: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub is_compatible: bool,
    pub compatibility_score: f32, // 0.0 to 1.0
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub recommendations: Vec<ValidationRecommendation>,
    pub conversion_path: Option<ConversionPath>,
}

#[derive(Debug, Clone)]
pub struct ValidationError {
    pub category: ErrorCategory,
    pub severity: ErrorSeverity,
    pub message: String,
    pub location: Option<String>, // e.g., tensor name, operation name
    pub fix_suggestion: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub category: WarningCategory,
    pub message: String,
    pub location: Option<String>,
    pub impact: WarningImpact,
}

#[derive(Debug, Clone)]
pub struct ValidationRecommendation {
    pub category: RecommendationCategory,
    pub message: String,
    pub priority: RecommendationPriority,
    pub estimated_effort: EstimatedEffort,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorCategory {
    DataType,
    Shape,
    Operation,
    Metadata,
    Framework,
    Version,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorSeverity {
    Critical, // Blocks conversion
    High,     // Likely to cause runtime errors
    Medium,   // May cause issues
    Low,      // Minor issues
}

#[derive(Debug, Clone, PartialEq)]
pub enum WarningCategory {
    Performance,
    Precision,
    Compatibility,
    BestPractice,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WarningImpact {
    High,   // Significant impact on model behavior
    Medium, // Moderate impact
    Low,    // Minor impact
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationCategory {
    Optimization,
    Conversion,
    Preprocessing,
    Alternative,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecommendationPriority {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone)]
pub enum EstimatedEffort {
    Minimal,  // < 1 hour
    Low,      // 1-4 hours
    Medium,   // 1-2 days
    High,     // 1 week
    VeryHigh, // > 1 week
}

#[derive(Debug, Clone)]
pub struct ConversionPath {
    pub steps: Vec<ConversionStep>,
    pub estimated_accuracy_loss: f32,      // 0.0 to 1.0
    pub estimated_performance_impact: f32, // Relative performance change
    pub complexity: ConversionComplexity,
}

#[derive(Debug, Clone)]
pub struct ConversionStep {
    pub operation: ConversionOperation,
    pub description: String,
    pub required_tools: Vec<String>,
    pub estimated_time: EstimatedEffort,
}

#[derive(Debug, Clone)]
pub enum ConversionOperation {
    DirectConversion,
    TypeConversion,
    ShapeReshaping,
    OperationMapping,
    ManualIntervention,
    AlternativeImplementation,
}

#[derive(Debug, Clone)]
pub enum ConversionComplexity {
    Trivial,     // Direct conversion possible
    Simple,      // Minor adjustments needed
    Moderate,    // Some manual work required
    Complex,     // Significant effort required
    VeryComplex, // Major rewrite needed
}

impl ModelValidator {
    pub fn new(source: MLFramework, target: MLFramework, config: ValidationConfig) -> Self {
        Self {
            source_framework: source,
            target_framework: target,
            validation_config: config,
        }
    }

    /// Validate model compatibility
    pub fn validate(&self, model: &MLModel) -> Result<ValidationReport> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut recommendations = Vec::new();

        // Check framework compatibility
        let framework_compatibility = self.check_framework_compatibility(model);
        if let Some(error) = framework_compatibility.error {
            errors.push(error);
        }
        warnings.extend(framework_compatibility.warnings);
        recommendations.extend(framework_compatibility.recommendations);

        // Check data types
        if self.validation_config.check_data_types {
            let dtype_check = self.check_data_types(model);
            errors.extend(dtype_check.errors);
            warnings.extend(dtype_check.warnings);
            recommendations.extend(dtype_check.recommendations);
        }

        // Check tensor shapes
        if self.validation_config.check_tensor_shapes {
            let shape_check = self.check_tensor_shapes(model);
            errors.extend(shape_check.errors);
            warnings.extend(shape_check.warnings);
            recommendations.extend(shape_check.recommendations);
        }

        // Check operations (if applicable)
        if self.validation_config.check_operations {
            let ops_check = self.check_operations(model);
            errors.extend(ops_check.errors);
            warnings.extend(ops_check.warnings);
            recommendations.extend(ops_check.recommendations);
        }

        // Check metadata
        if self.validation_config.check_metadata {
            let metadata_check = self.check_metadata(model);
            errors.extend(metadata_check.errors);
            warnings.extend(metadata_check.warnings);
            recommendations.extend(metadata_check.recommendations);
        }

        // Calculate compatibility score
        let compatibility_score = self.calculate_compatibility_score(&errors, &warnings);
        let is_compatible = compatibility_score > 0.7
            && errors.iter().all(|e| e.severity != ErrorSeverity::Critical);

        // Generate conversion path if compatible
        let conversion_path = if is_compatible {
            Some(self.generate_conversion_path(model, &errors, &warnings)?)
        } else {
            None
        };

        Ok(ValidationReport {
            is_compatible,
            compatibility_score,
            errors,
            warnings,
            recommendations,
            conversion_path,
        })
    }

    // Private helper methods would continue here...
    // (Implementation details omitted for brevity but would include all the check_* methods)
}

/// Batch validation for multiple models
pub struct BatchValidator {
    validators: Vec<ModelValidator>,
    #[allow(dead_code)]
    parallel: bool,
}

impl Default for BatchValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchValidator {
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
            parallel: true,
        }
    }

    pub fn add_validation(
        &mut self,
        source: MLFramework,
        target: MLFramework,
        config: ValidationConfig,
    ) {
        self.validators
            .push(ModelValidator::new(source, target, config));
    }

    pub fn validate_all(&self, models: &[MLModel]) -> Result<Vec<ValidationReport>> {
        let mut reports = Vec::new();

        for model in models {
            for validator in &self.validators {
                reports.push(validator.validate(model)?);
            }
        }

        Ok(reports)
    }
}

/// Validation utilities
pub mod utils {
    use super::*;

    /// Quick compatibility check
    pub fn quick_compatibility_check(source: MLFramework, target: MLFramework) -> f32 {
        // Simplified compatibility check
        if source == target {
            1.0
        } else if matches!((source, target), 
            (MLFramework::PyTorch, MLFramework::ONNX) |
            (MLFramework::TensorFlow, MLFramework::ONNX) |
            (MLFramework::ONNX, MLFramework::PyTorch) |
            (MLFramework::ONNX, MLFramework::TensorFlow)) {
            0.9
        } else {
            0.5
        }
    }

    /// Generate compatibility matrix for all frameworks
    pub fn generate_compatibility_matrix() -> BTreeMap<String, BTreeMap<String, f32>> {
        let frameworks = [
            MLFramework::PyTorch,
            MLFramework::TensorFlow,
            MLFramework::ONNX,
            MLFramework::SafeTensors,
            MLFramework::JAX,
            MLFramework::MXNet,
            MLFramework::CoreML,
            MLFramework::HuggingFace,
        ];

        let mut matrix = BTreeMap::new();

        for source in &frameworks {
            let mut row = BTreeMap::new();
            for target in &frameworks {
                let score = quick_compatibility_check(*source, *target);
                row.insert(format!("{:?}", target), score);
            }
            matrix.insert(format!("{:?}", source), row);
        }

        matrix
    }

    /// Find best conversion path between frameworks
    pub fn find_best_conversion_path(
        source: MLFramework,
        target: MLFramework,
    ) -> Vec<MLFramework> {
        // Simple pathfinding - in practice could use more sophisticated algorithms
        if source == target {
            return vec![source];
        }

        // Try direct conversion first
        if quick_compatibility_check(source, target) > 0.7 {
            return vec![source, target];
        }

        // Try via ONNX as intermediate
        if quick_compatibility_check(source, MLFramework::ONNX) > 0.7
            && quick_compatibility_check(MLFramework::ONNX, target) > 0.7
        {
            return vec![source, MLFramework::ONNX, target];
        }

        // Fallback to direct conversion
        vec![source, target]
    }
}

// Supporting structures
#[derive(Debug, Clone)]
struct FrameworkCompatibilityResult {
    error: Option<ValidationError>,
    warnings: Vec<ValidationWarning>,
    recommendations: Vec<ValidationRecommendation>,
}

#[derive(Debug, Clone)]
struct ValidationCheckResult {
    errors: Vec<ValidationError>,
    warnings: Vec<ValidationWarning>,
    recommendations: Vec<ValidationRecommendation>,
}

#[derive(Debug, Clone)]
struct FrameworkCompatibility {
    level: CompatibilityLevel,
    recommendations: Vec<ValidationRecommendation>,
}

#[derive(Debug, Clone)]
enum CompatibilityLevel {
    FullyCompatible,
    MostlyCompatible,
    PartiallyCompatible,
    #[allow(dead_code)]
    Incompatible,
}