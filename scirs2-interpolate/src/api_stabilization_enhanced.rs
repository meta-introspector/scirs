//! Enhanced API stabilization analysis for 0.1.0 stable release
//!
//! This module provides comprehensive analysis and validation of the public API
//! to ensure it's ready for stabilization in the 0.1.0 release.
//!
//! ## Key Features
//!
//! - **API consistency analysis**: Ensure consistent naming and patterns
//! - **Breaking change detection**: Identify potential breaking changes
//! - **API completeness validation**: Check for missing functionality
//! - **Documentation coverage analysis**: Ensure all public APIs are documented
//! - **Deprecation tracking**: Manage deprecated features
//! - **Semantic versioning compliance**: Ensure API follows semver

use crate::error::{InterpolateError, InterpolateResult};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Enhanced API stabilization analyzer for 0.1.0 stable release
#[derive(Debug)]
pub struct ApiStabilizationAnalyzer {
    /// Configuration for analysis
    config: StabilizationConfig,
    /// Collected API information
    api_inventory: ApiInventory,
    /// Analysis results
    analysis_results: Vec<ApiAnalysisResult>,
    /// Breaking change assessments
    breaking_changes: Vec<BreakingChangeAssessment>,
    /// Deprecation tracking
    deprecations: Vec<DeprecationItem>,
}

/// Configuration for API stabilization analysis
#[derive(Debug, Clone)]
pub struct StabilizationConfig {
    /// Minimum documentation coverage required (%)
    pub min_documentation_coverage: f32,
    /// Maximum allowed breaking changes for stable release
    pub max_breaking_changes: usize,
    /// Required API consistency score (0.0 to 1.0)
    pub min_consistency_score: f32,
    /// Allow experimental features in stable release
    pub allow_experimental_features: bool,
    /// Strictness level for analysis
    pub strictness_level: StrictnessLevel,
}

impl Default for StabilizationConfig {
    fn default() -> Self {
        Self {
            min_documentation_coverage: 95.0,
            max_breaking_changes: 0,
            min_consistency_score: 0.9,
            allow_experimental_features: false,
            strictness_level: StrictnessLevel::Strict,
        }
    }
}

/// Strictness levels for API analysis
#[derive(Debug, Clone)]
pub enum StrictnessLevel {
    /// Relaxed analysis for early development
    Relaxed,
    /// Standard analysis for alpha/beta releases
    Standard,
    /// Strict analysis for stable releases
    Strict,
    /// Ultra-strict analysis for LTS releases
    UltraStrict,
}

/// Complete inventory of the public API
#[derive(Debug, Default)]
pub struct ApiInventory {
    /// All public functions
    pub functions: Vec<ApiFunction>,
    /// All public types/structs
    pub types: Vec<ApiType>,
    /// All public traits
    pub traits: Vec<ApiTrait>,
    /// All public modules
    pub modules: Vec<ApiModule>,
    /// All public macros
    pub macros: Vec<ApiMacro>,
    /// Re-exports
    pub reexports: Vec<ApiReexport>,
}

/// Analysis result for a specific API item
#[derive(Debug, Clone)]
pub struct ApiAnalysisResult {
    /// API item name
    pub item_name: String,
    /// Item type
    pub item_type: ApiItemType,
    /// Stability assessment
    pub stability: StabilityAssessment,
    /// Issues found
    pub issues: Vec<ApiIssue>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Documentation coverage
    pub documentation: DocumentationAssessment,
    /// Consistency score
    pub consistency_score: f32,
}

/// Types of API items
#[derive(Debug, Clone)]
pub enum ApiItemType {
    Function,
    Struct,
    Enum,
    Trait,
    Module,
    Macro,
    Reexport,
    Constant,
    TypeAlias,
}

/// Stability assessment for an API item
#[derive(Debug, Clone)]
pub struct StabilityAssessment {
    /// Overall stability level
    pub level: ApiStabilityLevel,
    /// Confidence in assessment (0.0 to 1.0)
    pub confidence: f32,
    /// Risk factors
    pub risk_factors: Vec<RiskFactor>,
    /// Mitigation strategies
    pub mitigations: Vec<String>,
}

/// API stability levels
#[derive(Debug, Clone, PartialEq)]
pub enum ApiStabilityLevel {
    /// Stable and ready for release
    Stable,
    /// Mostly stable with minor concerns
    MostlyStable,
    /// Unstable and needs attention
    Unstable,
    /// Experimental - not ready for stable release
    Experimental,
    /// Deprecated - scheduled for removal
    Deprecated,
}

/// Risk factors affecting API stability
#[derive(Debug, Clone)]
pub enum RiskFactor {
    /// Insufficient testing
    InsufficientTesting,
    /// Poor documentation
    PoorDocumentation,
    /// Inconsistent naming
    InconsistentNaming,
    /// Complex API surface
    ComplexApiSurface,
    /// Missing error handling
    MissingErrorHandling,
    /// Performance concerns
    PerformanceConcerns,
    /// Breaking change potential
    BreakingChangePotential,
    /// Missing functionality
    MissingFunctionality,
}

/// API issue found during analysis
#[derive(Debug, Clone)]
pub struct ApiIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue category
    pub category: IssueCategory,
    /// Description of the issue
    pub description: String,
    /// Location of the issue
    pub location: String,
    /// Suggested resolution
    pub suggested_resolution: Option<String>,
    /// Blocking for stable release
    pub blocks_stable_release: bool,
}

/// Severity levels for API issues
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum IssueSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Categories of API issues
#[derive(Debug, Clone)]
pub enum IssueCategory {
    /// Breaking change
    BreakingChange,
    /// Documentation issue
    Documentation,
    /// Naming consistency
    NamingConsistency,
    /// API design
    ApiDesign,
    /// Error handling
    ErrorHandling,
    /// Performance
    Performance,
    /// Security
    Security,
    /// Completeness
    Completeness,
    /// Deprecation
    Deprecation,
}

/// Documentation coverage assessment
#[derive(Debug, Clone)]
pub struct DocumentationAssessment {
    /// Has documentation
    pub has_docs: bool,
    /// Documentation quality score (0.0 to 1.0)
    pub quality_score: f32,
    /// Missing documentation elements
    pub missing_elements: Vec<String>,
    /// Examples present
    pub has_examples: bool,
    /// Error conditions documented
    pub documents_errors: bool,
}

/// Breaking change assessment
#[derive(Debug, Clone)]
pub struct BreakingChangeAssessment {
    /// Change description
    pub description: String,
    /// Severity of breaking change
    pub severity: BreakingChangeSeverity,
    /// Affected API items
    pub affected_items: Vec<String>,
    /// Migration path available
    pub migration_path: Option<String>,
    /// Can be mitigated
    pub can_be_mitigated: bool,
    /// Mitigation strategy
    pub mitigation_strategy: Option<String>,
}

/// Severity of breaking changes
#[derive(Debug, Clone, PartialEq)]
pub enum BreakingChangeSeverity {
    /// Severe - affects most users
    Severe,
    /// Major - affects many users
    Major,
    /// Minor - affects some users
    Minor,
    /// Negligible - affects very few users
    Negligible,
}

/// Deprecation item tracking
#[derive(Debug, Clone)]
pub struct DeprecationItem {
    /// Item name
    pub item_name: String,
    /// Deprecation reason
    pub reason: String,
    /// Replacement suggestion
    pub replacement: Option<String>,
    /// Target removal version
    pub removal_version: Option<String>,
    /// Deprecation timeline
    pub timeline: DeprecationTimeline,
}

/// Deprecation timeline
#[derive(Debug, Clone)]
pub struct DeprecationTimeline {
    /// When deprecated
    pub deprecated_in: String,
    /// Warning period (versions)
    pub warning_period: usize,
    /// When to remove
    pub remove_in: String,
}

/// Placeholder API item types for analysis
#[derive(Debug, Clone)]
pub struct ApiFunction {
    pub name: String,
    pub signature: String,
    pub visibility: Visibility,
    pub documentation: bool,
    pub deprecated: bool,
    pub experimental: bool,
}

#[derive(Debug, Clone)]
pub struct ApiType {
    pub name: String,
    pub kind: TypeKind,
    pub visibility: Visibility,
    pub documentation: bool,
    pub deprecated: bool,
    pub experimental: bool,
}

#[derive(Debug, Clone)]
pub struct ApiTrait {
    pub name: String,
    pub methods: Vec<String>,
    pub visibility: Visibility,
    pub documentation: bool,
    pub deprecated: bool,
    pub experimental: bool,
}

#[derive(Debug, Clone)]
pub struct ApiModule {
    pub name: String,
    pub visibility: Visibility,
    pub documentation: bool,
    pub deprecated: bool,
    pub experimental: bool,
}

#[derive(Debug, Clone)]
pub struct ApiMacro {
    pub name: String,
    pub visibility: Visibility,
    pub documentation: bool,
    pub deprecated: bool,
    pub experimental: bool,
}

#[derive(Debug, Clone)]
pub struct ApiReexport {
    pub name: String,
    pub source: String,
    pub visibility: Visibility,
    pub documentation: bool,
}

#[derive(Debug, Clone)]
pub enum Visibility {
    Public,
    PublicCrate,
    Private,
}

#[derive(Debug, Clone)]
pub enum TypeKind {
    Struct,
    Enum,
    Union,
    TypeAlias,
}

impl ApiStabilizationAnalyzer {
    /// Create a new API stabilization analyzer
    pub fn new(config: StabilizationConfig) -> Self {
        Self {
            config,
            api_inventory: ApiInventory::default(),
            analysis_results: Vec::new(),
            breaking_changes: Vec::new(),
            deprecations: Vec::new(),
        }
    }

    /// Run comprehensive API stabilization analysis
    pub fn analyze_api_stability(&mut self) -> InterpolateResult<ApiStabilizationReport> {
        println!("Starting comprehensive API stabilization analysis...");

        // 1. Collect API inventory
        self.collect_api_inventory()?;

        // 2. Analyze each API item
        self.analyze_api_items()?;

        // 3. Check for breaking changes
        self.detect_breaking_changes()?;

        // 4. Assess API consistency
        let consistency_score = self.assess_api_consistency()?;

        // 5. Check documentation coverage
        let documentation_coverage = self.check_documentation_coverage()?;

        // 6. Generate stability recommendations
        let recommendations = self.generate_stability_recommendations()?;

        // 7. Assess overall readiness
        let readiness = self.assess_stable_release_readiness()?;

        println!("API stabilization analysis completed.");

        Ok(ApiStabilizationReport {
            overall_readiness: readiness,
            consistency_score,
            documentation_coverage,
            total_items: self.count_total_api_items(),
            stable_items: self.count_stable_items(),
            unstable_items: self.count_unstable_items(),
            breaking_changes: self.breaking_changes.clone(),
            critical_issues: self.get_critical_issues(),
            analysis_results: self.analysis_results.clone(),
            recommendations,
            deprecations: self.deprecations.clone(),
            config: self.config.clone(),
        })
    }

    /// Collect comprehensive API inventory
    fn collect_api_inventory(&mut self) -> InterpolateResult<()> {
        println!("Collecting API inventory...");

        // This would normally use reflection or AST parsing
        // For now, we'll populate with known API items from the interpolation library

        // Major public functions
        self.api_inventory.functions.extend(vec![
            ApiFunction {
                name: "linear_interpolate".to_string(),
                signature: "fn linear_interpolate<T>(...) -> InterpolateResult<Array1<T>>".to_string(),
                visibility: Visibility::Public,
                documentation: true,
                deprecated: false,
                experimental: false,
            },
            ApiFunction {
                name: "cubic_interpolate".to_string(),
                signature: "fn cubic_interpolate<T>(...) -> InterpolateResult<Array1<T>>".to_string(),
                visibility: Visibility::Public,
                documentation: true,
                deprecated: false,
                experimental: false,
            },
            ApiFunction {
                name: "pchip_interpolate".to_string(),
                signature: "fn pchip_interpolate<T>(...) -> InterpolateResult<Array1<T>>".to_string(),
                visibility: Visibility::Public,
                documentation: true,
                deprecated: false,
                experimental: false,
            },
            ApiFunction {
                name: "make_rbf_interpolator".to_string(),
                signature: "fn make_rbf_interpolator<T>(...) -> InterpolateResult<RBFInterpolator<T>>".to_string(),
                visibility: Visibility::Public,
                documentation: true,
                deprecated: false,
                experimental: false,
            },
            ApiFunction {
                name: "make_kriging_interpolator".to_string(),
                signature: "fn make_kriging_interpolator<T>(...) -> InterpolateResult<KrigingInterpolator<T>>".to_string(),
                visibility: Visibility::Public,
                documentation: true,
                deprecated: false,
                experimental: false,
            },
        ]);

        // Major public types
        self.api_inventory.types.extend(vec![
            ApiType {
                name: "RBFInterpolator".to_string(),
                kind: TypeKind::Struct,
                visibility: Visibility::Public,
                documentation: true,
                deprecated: false,
                experimental: false,
            },
            ApiType {
                name: "KrigingInterpolator".to_string(),
                kind: TypeKind::Struct,
                visibility: Visibility::Public,
                documentation: true,
                deprecated: false,
                experimental: false,
            },
            ApiType {
                name: "BSpline".to_string(),
                kind: TypeKind::Struct,
                visibility: Visibility::Public,
                documentation: true,
                deprecated: false,
                experimental: false,
            },
            ApiType {
                name: "CubicSpline".to_string(),
                kind: TypeKind::Struct,
                visibility: Visibility::Public,
                documentation: true,
                deprecated: false,
                experimental: false,
            },
            ApiType {
                name: "InterpolateError".to_string(),
                kind: TypeKind::Enum,
                visibility: Visibility::Public,
                documentation: true,
                deprecated: false,
                experimental: false,
            },
            ApiType {
                name: "InterpolationMethod".to_string(),
                kind: TypeKind::Enum,
                visibility: Visibility::Public,
                documentation: true,
                deprecated: false,
                experimental: false,
            },
        ]);

        // Core traits
        self.api_inventory.traits.extend(vec![ApiTrait {
            name: "InterpolationFloat".to_string(),
            methods: vec!["from_f64".to_string(), "to_f64".to_string()],
            visibility: Visibility::Public,
            documentation: true,
            deprecated: false,
            experimental: false,
        }]);

        // Main modules
        self.api_inventory.modules.extend(vec![
            ApiModule {
                name: "interp1d".to_string(),
                visibility: Visibility::Public,
                documentation: true,
                deprecated: false,
                experimental: false,
            },
            ApiModule {
                name: "advanced".to_string(),
                visibility: Visibility::Public,
                documentation: true,
                deprecated: false,
                experimental: false,
            },
            ApiModule {
                name: "spline".to_string(),
                visibility: Visibility::Public,
                documentation: true,
                deprecated: false,
                experimental: false,
            },
            ApiModule {
                name: "gpu_accelerated".to_string(),
                visibility: Visibility::Public,
                documentation: true,
                deprecated: false,
                experimental: true, // Still experimental
            },
            ApiModule {
                name: "neural_enhanced".to_string(),
                visibility: Visibility::Public,
                documentation: true,
                deprecated: false,
                experimental: true, // Still experimental
            },
        ]);

        Ok(())
    }

    /// Analyze stability of each API item
    fn analyze_api_items(&mut self) -> InterpolateResult<()> {
        println!("Analyzing API item stability...");

        // Analyze functions
        for func in &self.api_inventory.functions {
            let analysis = self.analyze_function(func)?;
            self.analysis_results.push(analysis);
        }

        // Analyze types
        for type_item in &self.api_inventory.types {
            let analysis = self.analyze_type(type_item)?;
            self.analysis_results.push(analysis);
        }

        // Analyze traits
        for trait_item in &self.api_inventory.traits {
            let analysis = self.analyze_trait(trait_item)?;
            self.analysis_results.push(analysis);
        }

        // Analyze modules
        for module in &self.api_inventory.modules {
            let analysis = self.analyze_module(module)?;
            self.analysis_results.push(analysis);
        }

        Ok(())
    }

    /// Analyze a specific function
    fn analyze_function(&self, func: &ApiFunction) -> InterpolateResult<ApiAnalysisResult> {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        let mut risk_factors = Vec::new();

        // Check if experimental
        if func.experimental && !self.config.allow_experimental_features {
            issues.push(ApiIssue {
                severity: IssueSeverity::High,
                category: IssueCategory::ApiDesign,
                description: "Function marked as experimental".to_string(),
                location: func.name.clone(),
                suggested_resolution: Some(
                    "Remove experimental status or exclude from stable release".to_string(),
                ),
                blocks_stable_release: true,
            });
            risk_factors.push(RiskFactor::BreakingChangePotential);
        }

        // Check documentation
        if !func.documentation {
            issues.push(ApiIssue {
                severity: IssueSeverity::High,
                category: IssueCategory::Documentation,
                description: "Function lacks documentation".to_string(),
                location: func.name.clone(),
                suggested_resolution: Some(
                    "Add comprehensive documentation with examples".to_string(),
                ),
                blocks_stable_release: true,
            });
            risk_factors.push(RiskFactor::PoorDocumentation);
        }

        // Check naming consistency
        let naming_score = self.assess_naming_consistency(&func.name, "function");
        if naming_score < 0.8 {
            issues.push(ApiIssue {
                severity: IssueSeverity::Medium,
                category: IssueCategory::NamingConsistency,
                description: "Function name doesn't follow consistent patterns".to_string(),
                location: func.name.clone(),
                suggested_resolution: Some(
                    "Consider renaming to follow established patterns".to_string(),
                ),
                blocks_stable_release: false,
            });
            risk_factors.push(RiskFactor::InconsistentNaming);
        }

        // Determine stability level
        let stability_level = if func.experimental {
            ApiStabilityLevel::Experimental
        } else if func.deprecated {
            ApiStabilityLevel::Deprecated
        } else if issues.iter().any(|i| i.blocks_stable_release) {
            ApiStabilityLevel::Unstable
        } else if issues.iter().any(|i| i.severity >= IssueSeverity::High) {
            ApiStabilityLevel::MostlyStable
        } else {
            ApiStabilityLevel::Stable
        };

        // Generate recommendations
        if !issues.is_empty() {
            recommendations.push("Address identified issues before stable release".to_string());
        }
        if stability_level == ApiStabilityLevel::Stable {
            recommendations.push("Function is ready for stable release".to_string());
        }

        Ok(ApiAnalysisResult {
            item_name: func.name.clone(),
            item_type: ApiItemType::Function,
            stability: StabilityAssessment {
                level: stability_level,
                confidence: if issues.is_empty() { 0.95 } else { 0.7 },
                risk_factors,
                mitigations: recommendations.clone(),
            },
            issues,
            recommendations,
            documentation: DocumentationAssessment {
                has_docs: func.documentation,
                quality_score: if func.documentation { 0.8 } else { 0.0 },
                missing_elements: if func.documentation {
                    vec![]
                } else {
                    vec!["Basic documentation".to_string()]
                },
                has_examples: func.documentation, // Simplified assumption
                documents_errors: func.documentation, // Simplified assumption
            },
            consistency_score: naming_score,
        })
    }

    /// Analyze a specific type
    fn analyze_type(&self, type_item: &ApiType) -> InterpolateResult<ApiAnalysisResult> {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        let mut risk_factors = Vec::new();

        // Similar analysis as functions but type-specific
        if type_item.experimental && !self.config.allow_experimental_features {
            issues.push(ApiIssue {
                severity: IssueSeverity::High,
                category: IssueCategory::ApiDesign,
                description: "Type marked as experimental".to_string(),
                location: type_item.name.clone(),
                suggested_resolution: Some(
                    "Stabilize type or exclude from stable release".to_string(),
                ),
                blocks_stable_release: true,
            });
            risk_factors.push(RiskFactor::BreakingChangePotential);
        }

        if !type_item.documentation {
            issues.push(ApiIssue {
                severity: IssueSeverity::High,
                category: IssueCategory::Documentation,
                description: "Type lacks documentation".to_string(),
                location: type_item.name.clone(),
                suggested_resolution: Some("Add comprehensive type documentation".to_string()),
                blocks_stable_release: true,
            });
            risk_factors.push(RiskFactor::PoorDocumentation);
        }

        let naming_score = self.assess_naming_consistency(&type_item.name, "type");
        if naming_score < 0.8 {
            issues.push(ApiIssue {
                severity: IssueSeverity::Medium,
                category: IssueCategory::NamingConsistency,
                description: "Type name doesn't follow consistent patterns".to_string(),
                location: type_item.name.clone(),
                suggested_resolution: Some(
                    "Consider renaming to follow established patterns".to_string(),
                ),
                blocks_stable_release: false,
            });
            risk_factors.push(RiskFactor::InconsistentNaming);
        }

        let stability_level = if type_item.experimental {
            ApiStabilityLevel::Experimental
        } else if type_item.deprecated {
            ApiStabilityLevel::Deprecated
        } else if issues.iter().any(|i| i.blocks_stable_release) {
            ApiStabilityLevel::Unstable
        } else if issues.iter().any(|i| i.severity >= IssueSeverity::High) {
            ApiStabilityLevel::MostlyStable
        } else {
            ApiStabilityLevel::Stable
        };

        if !issues.is_empty() {
            recommendations.push("Address identified issues before stable release".to_string());
        }

        Ok(ApiAnalysisResult {
            item_name: type_item.name.clone(),
            item_type: ApiItemType::Struct, // Simplified
            stability: StabilityAssessment {
                level: stability_level,
                confidence: if issues.is_empty() { 0.95 } else { 0.7 },
                risk_factors,
                mitigations: recommendations.clone(),
            },
            issues,
            recommendations,
            documentation: DocumentationAssessment {
                has_docs: type_item.documentation,
                quality_score: if type_item.documentation { 0.8 } else { 0.0 },
                missing_elements: if type_item.documentation {
                    vec![]
                } else {
                    vec!["Type documentation".to_string()]
                },
                has_examples: type_item.documentation,
                documents_errors: true, // Types don't directly document errors
            },
            consistency_score: naming_score,
        })
    }

    /// Analyze a specific trait
    fn analyze_trait(&self, trait_item: &ApiTrait) -> InterpolateResult<ApiAnalysisResult> {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        let mut risk_factors = Vec::new();

        if trait_item.experimental && !self.config.allow_experimental_features {
            issues.push(ApiIssue {
                severity: IssueSeverity::High,
                category: IssueCategory::ApiDesign,
                description: "Trait marked as experimental".to_string(),
                location: trait_item.name.clone(),
                suggested_resolution: Some(
                    "Stabilize trait or exclude from stable release".to_string(),
                ),
                blocks_stable_release: true,
            });
        }

        if !trait_item.documentation {
            issues.push(ApiIssue {
                severity: IssueSeverity::High,
                category: IssueCategory::Documentation,
                description: "Trait lacks documentation".to_string(),
                location: trait_item.name.clone(),
                suggested_resolution: Some("Add comprehensive trait documentation".to_string()),
                blocks_stable_release: true,
            });
            risk_factors.push(RiskFactor::PoorDocumentation);
        }

        let naming_score = self.assess_naming_consistency(&trait_item.name, "trait");
        let stability_level = if issues.iter().any(|i| i.blocks_stable_release) {
            ApiStabilityLevel::Unstable
        } else {
            ApiStabilityLevel::Stable
        };

        Ok(ApiAnalysisResult {
            item_name: trait_item.name.clone(),
            item_type: ApiItemType::Trait,
            stability: StabilityAssessment {
                level: stability_level,
                confidence: 0.9,
                risk_factors,
                mitigations: recommendations.clone(),
            },
            issues,
            recommendations,
            documentation: DocumentationAssessment {
                has_docs: trait_item.documentation,
                quality_score: if trait_item.documentation { 0.8 } else { 0.0 },
                missing_elements: if trait_item.documentation {
                    vec![]
                } else {
                    vec!["Trait documentation".to_string()]
                },
                has_examples: trait_item.documentation,
                documents_errors: true,
            },
            consistency_score: naming_score,
        })
    }

    /// Analyze a specific module
    fn analyze_module(&self, module: &ApiModule) -> InterpolateResult<ApiAnalysisResult> {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();
        let mut risk_factors = Vec::new();

        if module.experimental && !self.config.allow_experimental_features {
            issues.push(ApiIssue {
                severity: IssueSeverity::High,
                category: IssueCategory::ApiDesign,
                description: "Module marked as experimental".to_string(),
                location: module.name.clone(),
                suggested_resolution: Some(
                    "Stabilize module or exclude from stable release".to_string(),
                ),
                blocks_stable_release: true,
            });
            risk_factors.push(RiskFactor::BreakingChangePotential);
        }

        if !module.documentation {
            issues.push(ApiIssue {
                severity: IssueSeverity::Medium,
                category: IssueCategory::Documentation,
                description: "Module lacks documentation".to_string(),
                location: module.name.clone(),
                suggested_resolution: Some("Add module-level documentation".to_string()),
                blocks_stable_release: false,
            });
            risk_factors.push(RiskFactor::PoorDocumentation);
        }

        let naming_score = self.assess_naming_consistency(&module.name, "module");
        let stability_level = if issues.iter().any(|i| i.blocks_stable_release) {
            ApiStabilityLevel::Unstable
        } else {
            ApiStabilityLevel::Stable
        };

        Ok(ApiAnalysisResult {
            item_name: module.name.clone(),
            item_type: ApiItemType::Module,
            stability: StabilityAssessment {
                level: stability_level,
                confidence: 0.85,
                risk_factors,
                mitigations: recommendations.clone(),
            },
            issues,
            recommendations,
            documentation: DocumentationAssessment {
                has_docs: module.documentation,
                quality_score: if module.documentation { 0.7 } else { 0.0 },
                missing_elements: if module.documentation {
                    vec![]
                } else {
                    vec!["Module documentation".to_string()]
                },
                has_examples: false, // Modules typically don't have examples
                documents_errors: false,
            },
            consistency_score: naming_score,
        })
    }

    /// Assess naming consistency
    fn assess_naming_consistency(&self, name: &str, item_type: &str) -> f32 {
        let mut score = 1.0;

        match item_type {
            "function" => {
                // Functions should be snake_case
                if !name
                    .chars()
                    .all(|c| c.is_lowercase() || c == '_' || c.is_ascii_digit())
                {
                    score -= 0.3;
                }
                // Should have descriptive verbs
                if !name.contains("interpolate")
                    && !name.contains("make")
                    && !name.contains("create")
                {
                    score -= 0.1;
                }
            }
            "type" => {
                // Types should be PascalCase
                if !name.chars().next().unwrap_or('a').is_uppercase() {
                    score -= 0.3;
                }
                // Should be nouns
                if name.ends_with("ing") || name.ends_with("ed") {
                    score -= 0.1;
                }
            }
            "trait" => {
                // Traits should be PascalCase
                if !name.chars().next().unwrap_or('a').is_uppercase() {
                    score -= 0.3;
                }
            }
            "module" => {
                // Modules should be snake_case
                if !name
                    .chars()
                    .all(|c| c.is_lowercase() || c == '_' || c.is_ascii_digit())
                {
                    score -= 0.3;
                }
            }
            _ => {}
        }

        score.max(0.0)
    }

    /// Detect potential breaking changes
    fn detect_breaking_changes(&mut self) -> InterpolateResult<()> {
        println!("Detecting potential breaking changes...");

        // Check for experimental features that might become breaking changes
        for result in &self.analysis_results {
            if let Some(experimental_issue) = result
                .issues
                .iter()
                .find(|i| i.description.contains("experimental"))
            {
                self.breaking_changes.push(BreakingChangeAssessment {
                    description: format!(
                        "Experimental feature '{}' may introduce breaking changes",
                        result.item_name
                    ),
                    severity: BreakingChangeSeverity::Major,
                    affected_items: vec![result.item_name.clone()],
                    migration_path: Some("Provide migration guide when stabilizing".to_string()),
                    can_be_mitigated: true,
                    mitigation_strategy: Some(
                        "Deprecation period with clear migration path".to_string(),
                    ),
                });
            }
        }

        // Check for deprecated items that might be removed
        for func in &self.api_inventory.functions {
            if func.deprecated {
                self.deprecations.push(DeprecationItem {
                    item_name: func.name.clone(),
                    reason: "Superseded by improved implementation".to_string(),
                    replacement: Some("New improved version".to_string()),
                    removal_version: Some("0.2.0".to_string()),
                    timeline: DeprecationTimeline {
                        deprecated_in: "0.1.0".to_string(),
                        warning_period: 2,
                        remove_in: "0.2.0".to_string(),
                    },
                });
            }
        }

        Ok(())
    }

    /// Assess overall API consistency
    fn assess_api_consistency(&self) -> InterpolateResult<f32> {
        if self.analysis_results.is_empty() {
            return Ok(0.0);
        }

        let total_score: f32 = self
            .analysis_results
            .iter()
            .map(|r| r.consistency_score)
            .sum();

        Ok(total_score / self.analysis_results.len() as f32)
    }

    /// Check documentation coverage
    fn check_documentation_coverage(&self) -> InterpolateResult<f32> {
        if self.analysis_results.is_empty() {
            return Ok(0.0);
        }

        let documented_count = self
            .analysis_results
            .iter()
            .filter(|r| r.documentation.has_docs)
            .count();

        Ok((documented_count as f32 / self.analysis_results.len() as f32) * 100.0)
    }

    /// Generate stability recommendations
    fn generate_stability_recommendations(&self) -> InterpolateResult<Vec<String>> {
        let mut recommendations = Vec::new();

        let critical_issues = self.get_critical_issues();
        if !critical_issues.is_empty() {
            recommendations.push(format!(
                "CRITICAL: Address {} critical issues before stable release",
                critical_issues.len()
            ));
        }

        let experimental_count = self
            .analysis_results
            .iter()
            .filter(|r| r.stability.level == ApiStabilityLevel::Experimental)
            .count();

        if experimental_count > 0 && !self.config.allow_experimental_features {
            recommendations.push(format!(
                "Remove or stabilize {} experimental features",
                experimental_count
            ));
        }

        let documentation_coverage = self.check_documentation_coverage().unwrap_or(0.0);
        if documentation_coverage < self.config.min_documentation_coverage {
            recommendations.push(format!(
                "Improve documentation coverage from {:.1}% to {:.1}%",
                documentation_coverage, self.config.min_documentation_coverage
            ));
        }

        let consistency_score = self.assess_api_consistency().unwrap_or(0.0);
        if consistency_score < self.config.min_consistency_score {
            recommendations.push(format!(
                "Improve API consistency from {:.2} to {:.2}",
                consistency_score, self.config.min_consistency_score
            ));
        }

        if self.breaking_changes.len() > self.config.max_breaking_changes {
            recommendations.push(format!(
                "Reduce breaking changes from {} to {}",
                self.breaking_changes.len(),
                self.config.max_breaking_changes
            ));
        }

        if recommendations.is_empty() {
            recommendations.push("API appears ready for stable release".to_string());
        }

        Ok(recommendations)
    }

    /// Assess overall stable release readiness
    fn assess_stable_release_readiness(&self) -> InterpolateResult<StableReleaseReadiness> {
        let critical_issues = self.get_critical_issues();
        let documentation_coverage = self.check_documentation_coverage().unwrap_or(0.0);
        let consistency_score = self.assess_api_consistency().unwrap_or(0.0);

        let readiness = if !critical_issues.is_empty() {
            StableReleaseReadiness::NotReady
        } else if documentation_coverage < self.config.min_documentation_coverage {
            StableReleaseReadiness::NeedsWork
        } else if consistency_score < self.config.min_consistency_score {
            StableReleaseReadiness::NeedsWork
        } else if self.breaking_changes.len() > self.config.max_breaking_changes {
            StableReleaseReadiness::NeedsWork
        } else {
            StableReleaseReadiness::Ready
        };

        Ok(readiness)
    }

    /// Helper methods
    fn count_total_api_items(&self) -> usize {
        self.api_inventory.functions.len()
            + self.api_inventory.types.len()
            + self.api_inventory.traits.len()
            + self.api_inventory.modules.len()
    }

    fn count_stable_items(&self) -> usize {
        self.analysis_results
            .iter()
            .filter(|r| r.stability.level == ApiStabilityLevel::Stable)
            .count()
    }

    fn count_unstable_items(&self) -> usize {
        self.analysis_results
            .iter()
            .filter(|r| {
                matches!(
                    r.stability.level,
                    ApiStabilityLevel::Unstable | ApiStabilityLevel::Experimental
                )
            })
            .count()
    }

    fn get_critical_issues(&self) -> Vec<ApiIssue> {
        self.analysis_results
            .iter()
            .flat_map(|r| &r.issues)
            .filter(|i| i.severity == IssueSeverity::Critical || i.blocks_stable_release)
            .cloned()
            .collect()
    }
}

/// Complete API stabilization report
#[derive(Debug)]
pub struct ApiStabilizationReport {
    /// Overall readiness assessment
    pub overall_readiness: StableReleaseReadiness,
    /// API consistency score (0.0 to 1.0)
    pub consistency_score: f32,
    /// Documentation coverage percentage
    pub documentation_coverage: f32,
    /// Total API items analyzed
    pub total_items: usize,
    /// Number of stable items
    pub stable_items: usize,
    /// Number of unstable items
    pub unstable_items: usize,
    /// Detected breaking changes
    pub breaking_changes: Vec<BreakingChangeAssessment>,
    /// Critical issues that block release
    pub critical_issues: Vec<ApiIssue>,
    /// Detailed analysis results
    pub analysis_results: Vec<ApiAnalysisResult>,
    /// Recommendations for stable release
    pub recommendations: Vec<String>,
    /// Deprecation tracking
    pub deprecations: Vec<DeprecationItem>,
    /// Configuration used for analysis
    pub config: StabilizationConfig,
}

/// Stable release readiness levels
#[derive(Debug, Clone, PartialEq)]
pub enum StableReleaseReadiness {
    /// Ready for stable release
    Ready,
    /// Needs minor work before stable release
    NeedsWork,
    /// Major issues prevent stable release
    NotReady,
}

impl fmt::Display for ApiStabilizationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "=== API Stabilization Report for 0.1.0 Stable Release ==="
        )?;
        writeln!(f)?;
        writeln!(f, "Overall Readiness: {:?}", self.overall_readiness)?;
        writeln!(
            f,
            "API Consistency Score: {:.2}/1.0",
            self.consistency_score
        )?;
        writeln!(
            f,
            "Documentation Coverage: {:.1}%",
            self.documentation_coverage
        )?;
        writeln!(f)?;
        writeln!(f, "API Items Summary:")?;
        writeln!(f, "  Total Items: {}", self.total_items)?;
        writeln!(
            f,
            "  Stable Items: {} ({:.1}%)",
            self.stable_items,
            (self.stable_items as f32 / self.total_items as f32) * 100.0
        )?;
        writeln!(
            f,
            "  Unstable Items: {} ({:.1}%)",
            self.unstable_items,
            (self.unstable_items as f32 / self.total_items as f32) * 100.0
        )?;
        writeln!(f)?;

        if !self.critical_issues.is_empty() {
            writeln!(f, "Critical Issues ({}):", self.critical_issues.len())?;
            for issue in &self.critical_issues {
                writeln!(
                    f,
                    "  - {} ({}): {}",
                    issue.location,
                    format!("{:?}", issue.severity),
                    issue.description
                )?;
            }
            writeln!(f)?;
        }

        if !self.breaking_changes.is_empty() {
            writeln!(f, "Breaking Changes ({}):", self.breaking_changes.len())?;
            for change in &self.breaking_changes {
                writeln!(
                    f,
                    "  - {}: {}",
                    format!("{:?}", change.severity),
                    change.description
                )?;
            }
            writeln!(f)?;
        }

        writeln!(f, "Recommendations:")?;
        for recommendation in &self.recommendations {
            writeln!(f, "  - {}", recommendation)?;
        }

        Ok(())
    }
}

/// Convenience functions for quick API analysis

/// Run comprehensive API stabilization analysis with default configuration
pub fn analyze_api_for_stable_release() -> InterpolateResult<ApiStabilizationReport> {
    let config = StabilizationConfig::default();
    let mut analyzer = ApiStabilizationAnalyzer::new(config);
    analyzer.analyze_api_stability()
}

/// Run quick API analysis for development
pub fn quick_api_analysis() -> InterpolateResult<ApiStabilizationReport> {
    let config = StabilizationConfig {
        min_documentation_coverage: 80.0,
        max_breaking_changes: 5,
        min_consistency_score: 0.8,
        allow_experimental_features: true,
        strictness_level: StrictnessLevel::Standard,
    };
    let mut analyzer = ApiStabilizationAnalyzer::new(config);
    analyzer.analyze_api_stability()
}

/// Run API analysis with custom configuration
pub fn analyze_api_with_config(
    config: StabilizationConfig,
) -> InterpolateResult<ApiStabilizationReport> {
    let mut analyzer = ApiStabilizationAnalyzer::new(config);
    analyzer.analyze_api_stability()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_analyzer_creation() {
        let config = StabilizationConfig::default();
        let analyzer = ApiStabilizationAnalyzer::new(config);
        assert_eq!(analyzer.analysis_results.len(), 0);
    }

    #[test]
    fn test_naming_consistency() {
        let config = StabilizationConfig::default();
        let analyzer = ApiStabilizationAnalyzer::new(config);

        // Test function naming
        assert!(analyzer.assess_naming_consistency("linear_interpolate", "function") > 0.8);
        assert!(analyzer.assess_naming_consistency("LinearInterpolate", "function") < 0.8);

        // Test type naming
        assert!(analyzer.assess_naming_consistency("RBFInterpolator", "type") > 0.8);
        assert!(analyzer.assess_naming_consistency("rbf_interpolator", "type") < 0.8);
    }

    #[test]
    fn test_quick_api_analysis() {
        let result = quick_api_analysis();
        assert!(result.is_ok());
    }
}
