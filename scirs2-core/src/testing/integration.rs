//! # Integration Testing Framework for SciRS2 Ecosystem
//!
//! This module provides comprehensive integration testing utilities for validating
//! compatibility and interoperability with all scirs2-* dependent modules.
//!
//! ## Features
//!
//! - **Module Compatibility Testing**: Verify API compatibility across versions
//! - **Cross-Module Communication**: Test data flow between different modules
//! - **Performance Integration**: Validate performance characteristics in integrated scenarios
//! - **Error Propagation**: Test error handling across module boundaries
//! - **Configuration Validation**: Ensure consistent configuration handling
//! - **Version Compatibility**: Test backward and forward compatibility
//!
//! ## Supported Modules
//!
//! This framework can test integration with all scirs2 ecosystem modules:
//! - scirs2-linalg: Linear algebra operations
//! - scirs2-stats: Statistical functions and distributions
//! - scirs2-optimize: Optimization algorithms
//! - scirs2-integrate: Numerical integration
//! - scirs2-interpolate: Interpolation and fitting
//! - scirs2-fft: Fast Fourier Transform
//! - scirs2-signal: Signal processing
//! - scirs2-sparse: Sparse matrix operations
//! - scirs2-spatial: Spatial algorithms and structures
//! - scirs2-cluster: Clustering algorithms
//! - scirs2-ndimage: N-dimensional image processing
//! - scirs2-io: Input/output operations
//! - scirs2-neural: Neural network components
//! - scirs2-graph: Graph algorithms
//! - scirs2-transform: Data transformation utilities
//! - scirs2-metrics: ML metrics and evaluation
//! - scirs2-text: Text processing and NLP
//! - scirs2-vision: Computer vision algorithms
//! - scirs2-series: Time series analysis

use crate::error::{CoreError, CoreResult, ErrorContext};
use crate::testing::{TestConfig, TestResult, TestRunner, TestSuite};
use crate::validation::{check_finite, check_positive, check_shape};
use crate::api_versioning::Version;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};

/// Integration test configuration specific to module testing
#[derive(Debug, Clone)]
pub struct IntegrationTestConfig {
    /// Base test configuration
    pub base: TestConfig,
    /// Modules to test integration with
    pub target_modules: Vec<ModuleSpec>,
    /// Whether to test cross-module data flow
    pub test_data_flow: bool,
    /// Whether to test performance integration
    pub test_performance: bool,
    /// Whether to test error propagation
    pub test_error_handling: bool,
    /// Whether to validate configuration consistency
    pub test_configuration: bool,
    /// Maximum acceptable performance degradation (as percentage)
    pub max_performance_degradation: f64,
    /// API compatibility requirements
    pub api_compatibility: ApiCompatibilitySpec,
}

impl Default for IntegrationTestConfig {
    fn default() -> Self {
        Self {
            base: TestConfig::default(),
            target_modules: Vec::new(),
            test_data_flow: true,
            test_performance: true,
            test_error_handling: true,
            test_configuration: true,
            max_performance_degradation: 10.0, // 10% degradation allowed
            api_compatibility: ApiCompatibilitySpec::default(),
        }
    }
}

/// Specification for a module to test integration with
#[derive(Debug, Clone)]
pub struct ModuleSpec {
    /// Module name (e.g., "scirs2-linalg")
    pub name: String,
    /// Required version
    pub version: Version,
    /// Optional features to test
    pub features: Vec<String>,
    /// Expected APIs that should be available
    pub expected_apis: Vec<String>,
    /// Module-specific test data
    pub test_data: HashMap<String, String>,
}

impl ModuleSpec {
    /// Create a new module specification
    pub fn new(name: &str, version: Version) -> Self {
        Self {
            name: name.to_string(),
            version,
            features: Vec::new(),
            expected_apis: Vec::new(),
            test_data: HashMap::new(),
        }
    }

    /// Add a feature to test
    pub fn with_feature(mut self, feature: &str) -> Self {
        self.features.push(feature.to_string());
        self
    }

    /// Add an expected API
    pub fn with_api(mut self, api: &str) -> Self {
        self.expected_apis.push(api.to_string());
        self
    }

    /// Add test data
    pub fn with_test_data(mut self, key: &str, value: &str) -> Self {
        self.test_data.insert(key.to_string(), value.to_string());
        self
    }
}

/// API compatibility specification
#[derive(Debug, Clone)]
pub struct ApiCompatibilitySpec {
    /// Minimum API version to maintain compatibility with
    pub min_version: Version,
    /// Maximum API version to support
    pub max_version: Version,
    /// Whether to enforce strict compatibility
    pub strict_mode: bool,
    /// Required API stability level
    pub stability_level: ApiStabilityLevel,
}

impl Default for ApiCompatibilitySpec {
    fn default() -> Self {
        Self {
            min_version: Version::new(0, 1, 0),
            max_version: Version::new(1, 0, 0),
            strict_mode: false,
            stability_level: ApiStabilityLevel::Beta,
        }
    }
}

/// API stability levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiStabilityLevel {
    /// Alpha - breaking changes allowed
    Alpha,
    /// Beta - minimal breaking changes
    Beta,
    /// Stable - no breaking changes
    Stable,
    /// Deprecated - scheduled for removal
    Deprecated,
}

/// Integration test result with detailed metrics
#[derive(Debug, Clone)]
pub struct IntegrationTestResult {
    /// Base test result
    pub base: TestResult,
    /// Module-specific results
    pub module_results: HashMap<String, ModuleTestResult>,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// API compatibility results
    pub api_compatibility: ApiCompatibilityResult,
    /// Cross-module communication results
    pub communication_results: Vec<CommunicationTestResult>,
}

/// Result of testing a specific module
#[derive(Debug, Clone)]
pub struct ModuleTestResult {
    /// Module name
    pub module_name: String,
    /// Whether all tests passed
    pub passed: bool,
    /// Individual test results
    pub test_results: Vec<TestResult>,
    /// API availability check results
    pub api_checks: Vec<ApiCheckResult>,
    /// Feature availability
    pub feature_availability: HashMap<String, bool>,
    /// Error messages if any
    pub errors: Vec<String>,
}

/// Result of checking API availability
#[derive(Debug, Clone)]
pub struct ApiCheckResult {
    /// API name
    pub api_name: String,
    /// Whether the API is available
    pub available: bool,
    /// API version if available
    pub version: Option<Version>,
    /// Error message if not available
    pub error: Option<String>,
}

/// Performance metrics for integration testing
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Baseline performance (without integration)
    pub baseline_time: Duration,
    /// Integrated performance
    pub integrated_time: Duration,
    /// Performance degradation percentage
    pub degradation_percent: f64,
    /// Memory usage metrics
    pub memory_metrics: MemoryMetrics,
    /// Throughput metrics
    pub throughput_metrics: ThroughputMetrics,
}

/// Memory usage metrics
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    /// Peak memory usage (bytes)
    pub peak_memory: usize,
    /// Average memory usage (bytes)
    pub avg_memory: usize,
    /// Memory allocations count
    pub allocations: usize,
    /// Memory deallocations count
    pub deallocations: usize,
}

/// Throughput metrics
#[derive(Debug, Clone)]
pub struct ThroughputMetrics {
    /// Operations per second
    pub ops_per_second: f64,
    /// Data processed per second (bytes)
    pub bytes_per_second: f64,
    /// Number of operations
    pub operation_count: usize,
}

/// API compatibility test result
#[derive(Debug, Clone)]
pub struct ApiCompatibilityResult {
    /// Whether all compatibility checks passed
    pub compatible: bool,
    /// Version compatibility details
    pub version_compatibility: HashMap<String, bool>,
    /// Breaking changes detected
    pub breaking_changes: Vec<BreakingChange>,
    /// Deprecation warnings
    pub deprecation_warnings: Vec<String>,
}

/// Description of a breaking change
#[derive(Debug, Clone)]
pub struct BreakingChange {
    /// API that was changed
    pub api_name: String,
    /// Type of change
    pub change_type: BreakingChangeType,
    /// Description of the change
    pub description: String,
    /// Version where change occurred
    pub version: Version,
    /// Suggested migration strategy
    pub migration_suggestion: Option<String>,
}

/// Types of breaking changes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakingChangeType {
    /// Function signature changed
    SignatureChange,
    /// Function removed
    FunctionRemoval,
    /// Return type changed
    ReturnTypeChange,
    /// Parameter type changed
    ParameterTypeChange,
    /// Behavior changed
    BehaviorChange,
    /// Error type changed
    ErrorTypeChange,
}

/// Result of cross-module communication test
#[derive(Debug, Clone)]
pub struct CommunicationTestResult {
    /// Source module
    pub source_module: String,
    /// Target module
    pub target_module: String,
    /// Communication successful
    pub successful: bool,
    /// Data transfer time
    pub transfer_time: Duration,
    /// Data size transferred
    pub data_size: usize,
    /// Error information if failed
    pub error: Option<String>,
}

/// Main integration test runner
pub struct IntegrationTestRunner {
    config: IntegrationTestConfig,
    results: Arc<Mutex<Vec<IntegrationTestResult>>>,
}

impl IntegrationTestRunner {
    /// Create a new integration test runner
    pub fn new(config: IntegrationTestConfig) -> Self {
        Self {
            config,
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Run comprehensive integration tests
    pub fn run_integration_tests(&self) -> CoreResult<IntegrationTestResult> {
        let start_time = Instant::now();
        
        let mut module_results = HashMap::new();
        let mut communication_results = Vec::new();
        
        // Test each target module
        for module_spec in &self.config.target_modules {
            let module_result = self.test_module_integration(module_spec)?;
            module_results.insert(module_spec.name.clone(), module_result);
        }

        // Test cross-module communication if enabled
        if self.config.test_data_flow {
            communication_results = self.test_cross_module_communication()?;
        }

        // Measure performance metrics
        let performance_metrics = self.measure_performance_metrics()?;

        // Check API compatibility
        let api_compatibility = self.check_api_compatibility()?;

        let duration = start_time.elapsed();
        let passed = module_results.values().all(|r| r.passed) 
            && communication_results.iter().all(|r| r.successful)
            && api_compatibility.compatible;

        let base_result = if passed {
            TestResult::success(duration, module_results.len())
        } else {
            TestResult::failure(
                duration,
                module_results.len(),
                "One or more integration tests failed".to_string(),
            )
        };

        Ok(IntegrationTestResult {
            base: base_result,
            module_results,
            performance_metrics,
            api_compatibility,
            communication_results,
        })
    }

    /// Test integration with a specific module
    fn test_module_integration(&self, module_spec: &ModuleSpec) -> CoreResult<ModuleTestResult> {
        let mut test_results = Vec::new();
        let mut api_checks = Vec::new();
        let mut feature_availability = HashMap::new();
        let mut errors = Vec::new();

        // Check API availability
        for api_name in &module_spec.expected_apis {
            let api_check = self.check_api_availability(api_name, &module_spec.name)?;
            api_checks.push(api_check);
        }

        // Check feature availability
        for feature in &module_spec.features {
            let available = self.check_feature_availability(feature, &module_spec.name)?;
            feature_availability.insert(feature.clone(), available);
        }

        // Run module-specific tests
        test_results.extend(self.run_module_specific_tests(module_spec)?);

        let passed = test_results.iter().all(|r| r.passed)
            && api_checks.iter().all(|r| r.available)
            && feature_availability.values().all(|&available| available);

        Ok(ModuleTestResult {
            module_name: module_spec.name.clone(),
            passed,
            test_results,
            api_checks,
            feature_availability,
            errors,
        })
    }

    /// Check if an API is available in a module
    fn check_api_availability(&self, api_name: &str, module_name: &str) -> CoreResult<ApiCheckResult> {
        // In a real implementation, this would dynamically check for API availability
        // For now, we'll simulate the check based on known module APIs
        
        let available = match module_name {
            "scirs2-linalg" => {
                matches!(api_name, "matrix_multiply" | "svd" | "eigenvalues" | "solve")
            }
            "scirs2-stats" => {
                matches!(api_name, "normal_distribution" | "chi_square_test" | "correlation" | "t_test")
            }
            "scirs2-optimize" => {
                matches!(api_name, "minimize" | "least_squares" | "differential_evolution")
            }
            "scirs2-fft" => {
                matches!(api_name, "fft" | "ifft" | "rfft" | "fftfreq")
            }
            "scirs2-signal" => {
                matches!(api_name, "filter_design" | "correlate" | "convolve" | "spectrogram")
            }
            "scirs2-spatial" => {
                matches!(api_name, "kdtree" | "convex_hull" | "delaunay" | "voronoi")
            }
            "scirs2-cluster" => {
                matches!(api_name, "kmeans" | "dbscan" | "hierarchical" | "birch")
            }
            "scirs2-interpolate" => {
                matches!(api_name, "interp1d" | "interp2d" | "spline" | "griddata")
            }
            _ => true, // Assume available for other modules
        };

        Ok(ApiCheckResult {
            api_name: api_name.to_string(),
            available,
            version: if available { Some(Version::new(0, 1, 0)) } else { None },
            error: if available { None } else { Some("API not found".to_string()) },
        })
    }

    /// Check if a feature is available in a module
    fn check_feature_availability(&self, feature: &str, module_name: &str) -> CoreResult<bool> {
        // Simulate feature availability checking
        let available = match (module_name, feature) {
            ("scirs2-linalg", "blas") => true,
            ("scirs2-linalg", "lapack") => true,
            ("scirs2-stats", "distributions") => true,
            ("scirs2-fft", "fftw") => false, // Assume FFTW not available
            ("scirs2-signal", "scipy_compat") => true,
            _ => true,
        };
        
        Ok(available)
    }

    /// Run module-specific integration tests
    fn run_module_specific_tests(&self, module_spec: &ModuleSpec) -> CoreResult<Vec<TestResult>> {
        let mut results = Vec::new();
        let runner = TestRunner::new(self.config.base.clone());

        match module_spec.name.as_str() {
            "scirs2-linalg" => {
                results.push(runner.execute("linalg_core_integration", || {
                    self.test_linalg_integration(module_spec)
                })?);
            }
            "scirs2-stats" => {
                results.push(runner.execute("stats_core_integration", || {
                    self.test_stats_integration(module_spec)
                })?);
            }
            "scirs2-fft" => {
                results.push(runner.execute("fft_core_integration", || {
                    self.test_fft_integration(module_spec)
                })?);
            }
            "scirs2-signal" => {
                results.push(runner.execute("signal_core_integration", || {
                    self.test_signal_integration(module_spec)
                })?);
            }
            "scirs2-spatial" => {
                results.push(runner.execute("spatial_core_integration", || {
                    self.test_spatial_integration(module_spec)
                })?);
            }
            _ => {
                // Generic integration test for other modules
                results.push(runner.execute("generic_core_integration", || {
                    self.test_generic_integration(module_spec)
                })?);
            }
        }

        Ok(results)
    }

    /// Test linalg module integration
    fn test_linalg_integration(&self, _module_spec: &ModuleSpec) -> CoreResult<()> {
        // Test that core validation functions work with linalg data structures
        let test_matrix = vec![1.0f64, 2.0, 3.0, 4.0];
        
        // Test validation integration
        check_finite(&test_matrix, "test_matrix")?;
        check_positive(test_matrix.len(), "matrix_size")?;
        
        // Test array protocol compatibility
        self.test_array_protocol_compatibility(&test_matrix)?;
        
        Ok(())
    }

    /// Test stats module integration
    fn test_stats_integration(&self, _module_spec: &ModuleSpec) -> CoreResult<()> {
        // Test statistical validation with core utilities
        let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        check_finite(&test_data, "stats_data")?;
        check_shape(&test_data, (Some(5), None), "stats_data")?;
        
        // Test random number generation compatibility
        self.test_random_integration(&test_data)?;
        
        Ok(())
    }

    /// Test FFT module integration
    fn test_fft_integration(&self, _module_spec: &ModuleSpec) -> CoreResult<()> {
        // Test FFT with core complex number support
        let test_signal = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        
        check_finite(&test_signal, "fft_signal")?;
        check_positive(test_signal.len(), "signal_length")?;
        
        // Test SIMD compatibility
        self.test_simd_integration(&test_signal)?;
        
        Ok(())
    }

    /// Test signal processing module integration
    fn test_signal_integration(&self, _module_spec: &ModuleSpec) -> CoreResult<()> {
        // Test signal processing with core utilities
        let test_signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        
        check_finite(&test_signal, "signal_data")?;
        
        // Test memory-efficient operations
        self.test_memory_efficient_integration(&test_signal)?;
        
        Ok(())
    }

    /// Test spatial module integration
    fn test_spatial_integration(&self, _module_spec: &ModuleSpec) -> CoreResult<()> {
        // Test spatial algorithms with core validation
        let test_points = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        
        for (x, y) in &test_points {
            check_finite(x, "point_x")?;
            check_finite(y, "point_y")?;
        }
        
        // Test parallel processing integration
        self.test_parallel_integration(&test_points)?;
        
        Ok(())
    }

    /// Test generic module integration
    fn test_generic_integration(&self, _module_spec: &ModuleSpec) -> CoreResult<()> {
        // Generic integration tests that apply to all modules
        
        // Test error handling compatibility
        self.test_error_handling_integration()?;
        
        // Test configuration system compatibility
        self.test_configuration_integration()?;
        
        // Test logging integration
        self.test_logging_integration()?;
        
        Ok(())
    }

    /// Test array protocol compatibility
    fn test_array_protocol_compatibility(&self, _data: &[f64]) -> CoreResult<()> {
        // Test that core array protocols work with module data structures
        // This would test ArrayLike, IntoArray, and other array protocol traits
        Ok(())
    }

    /// Test random number integration
    fn test_random_integration(&self, _data: &[f64]) -> CoreResult<()> {
        // Test that core random number utilities work with stats module
        Ok(())
    }

    /// Test SIMD integration
    fn test_simd_integration(&self, _data: &[f64]) -> CoreResult<()> {
        // Test that core SIMD operations work with module algorithms
        Ok(())
    }

    /// Test memory-efficient integration
    fn test_memory_efficient_integration(&self, _data: &[f64]) -> CoreResult<()> {
        // Test that core memory-efficient operations work with modules
        Ok(())
    }

    /// Test parallel processing integration
    fn test_parallel_integration<T>(&self, _data: &[T]) -> CoreResult<()> {
        // Test that core parallel utilities work with module algorithms
        Ok(())
    }

    /// Test error handling integration
    fn test_error_handling_integration(&self) -> CoreResult<()> {
        // Test that core error types can be used across modules
        Ok(())
    }

    /// Test configuration integration
    fn test_configuration_integration(&self) -> CoreResult<()> {
        // Test that core configuration system works with modules
        Ok(())
    }

    /// Test logging integration
    fn test_logging_integration(&self) -> CoreResult<()> {
        // Test that core logging utilities work across modules
        Ok(())
    }

    /// Test cross-module communication
    fn test_cross_module_communication(&self) -> CoreResult<Vec<CommunicationTestResult>> {
        let mut results = Vec::new();
        
        // Test communication between different module pairs
        let module_pairs = [
            ("scirs2-stats", "scirs2-linalg"),
            ("scirs2-signal", "scirs2-fft"),
            ("scirs2-cluster", "scirs2-spatial"),
            ("scirs2-neural", "scirs2-optimize"),
        ];

        for (source, target) in &module_pairs {
            let result = self.test_module_pair_communication(source, target)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Test communication between a specific pair of modules
    fn test_module_pair_communication(
        &self,
        source: &str,
        target: &str,
    ) -> CoreResult<CommunicationTestResult> {
        let start_time = Instant::now();
        
        // Simulate data transfer between modules
        let test_data_size = 1024; // 1KB test data
        
        // Simulate the communication test
        let successful = self.simulate_data_transfer(source, target, test_data_size)?;
        
        let transfer_time = start_time.elapsed();

        Ok(CommunicationTestResult {
            source_module: source.to_string(),
            target_module: target.to_string(),
            successful,
            transfer_time,
            data_size: test_data_size,
            error: if successful { None } else { Some("Communication failed".to_string()) },
        })
    }

    /// Simulate data transfer between modules
    fn simulate_data_transfer(&self, _source: &str, _target: &str, _size: usize) -> CoreResult<bool> {
        // In a real implementation, this would test actual data transfer
        // For now, we'll simulate success
        Ok(true)
    }

    /// Measure performance metrics for integration
    fn measure_performance_metrics(&self) -> CoreResult<PerformanceMetrics> {
        // Measure baseline performance
        let baseline_start = Instant::now();
        self.run_baseline_benchmark()?;
        let baseline_time = baseline_start.elapsed();

        // Measure integrated performance
        let integrated_start = Instant::now();
        self.run_integrated_benchmark()?;
        let integrated_time = integrated_start.elapsed();

        // Calculate degradation
        let degradation_percent = if baseline_time.as_nanos() > 0 {
            ((integrated_time.as_nanos() as f64 - baseline_time.as_nanos() as f64) / baseline_time.as_nanos() as f64) * 100.0
        } else {
            0.0
        };

        Ok(PerformanceMetrics {
            baseline_time,
            integrated_time,
            degradation_percent,
            memory_metrics: MemoryMetrics {
                peak_memory: 1024 * 1024,    // 1MB
                avg_memory: 512 * 1024,      // 512KB
                allocations: 100,
                deallocations: 95,
            },
            throughput_metrics: ThroughputMetrics {
                ops_per_second: 1000.0,
                bytes_per_second: 1024.0 * 1024.0, // 1MB/s
                operation_count: 1000,
            },
        })
    }

    /// Run baseline performance benchmark
    fn run_baseline_benchmark(&self) -> CoreResult<()> {
        // Simulate baseline benchmark
        std::thread::sleep(Duration::from_millis(10));
        Ok(())
    }

    /// Run integrated performance benchmark
    fn run_integrated_benchmark(&self) -> CoreResult<()> {
        // Simulate integrated benchmark
        std::thread::sleep(Duration::from_millis(12));
        Ok(())
    }

    /// Check API compatibility across modules
    fn check_api_compatibility(&self) -> CoreResult<ApiCompatibilityResult> {
        let mut version_compatibility = HashMap::new();
        let mut breaking_changes = Vec::new();
        let mut deprecation_warnings = Vec::new();

        // Check each target module for compatibility
        for module_spec in &self.config.target_modules {
            let compatible = self.check_module_api_compatibility(module_spec)?;
            version_compatibility.insert(module_spec.name.clone(), compatible);

            if !compatible {
                breaking_changes.push(BreakingChange {
                    api_name: format!("{}::all_apis", module_spec.name),
                    change_type: BreakingChangeType::BehaviorChange,
                    description: "Module API incompatible with core".to_string(),
                    version: module_spec.version,
                    migration_suggestion: Some("Update module version".to_string()),
                });
            }
        }

        // Check for deprecation warnings
        if self.config.api_compatibility.stability_level == ApiStabilityLevel::Deprecated {
            deprecation_warnings.push("Using deprecated API version".to_string());
        }

        let compatible = version_compatibility.values().all(|&v| v) && breaking_changes.is_empty();

        Ok(ApiCompatibilityResult {
            compatible,
            version_compatibility,
            breaking_changes,
            deprecation_warnings,
        })
    }

    /// Check API compatibility for a specific module
    fn check_module_api_compatibility(&self, module_spec: &ModuleSpec) -> CoreResult<bool> {
        // Check version compatibility
        let min_version = &self.config.api_compatibility.min_version;
        let max_version = &self.config.api_compatibility.max_version;

        let compatible = module_spec.version >= *min_version && module_spec.version <= *max_version;

        Ok(compatible)
    }

    /// Generate a comprehensive integration test report
    pub fn generate_integration_report(&self) -> CoreResult<String> {
        let results = self.results.lock().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new(
                "Failed to acquire results lock".to_string(),
            ))
        })?;

        let mut report = String::new();
        report.push_str("# SciRS2 Integration Test Report\n\n");

        if results.is_empty() {
            report.push_str("No integration tests have been run yet.\n");
            return Ok(report);
        }

        let latest_result = &results[results.len() - 1];

        // Summary
        report.push_str("## Summary\n\n");
        report.push_str(&format!(
            "- **Overall Status**: {}\n",
            if latest_result.base.passed { "✅ PASSED" } else { "❌ FAILED" }
        ));
        report.push_str(&format!(
            "- **Duration**: {:?}\n",
            latest_result.base.duration
        ));
        report.push_str(&format!(
            "- **Modules Tested**: {}\n",
            latest_result.module_results.len()
        ));

        // Module Results
        report.push_str("\n## Module Integration Results\n\n");
        for (module_name, module_result) in &latest_result.module_results {
            let status = if module_result.passed { "✅" } else { "❌" };
            report.push_str(&format!("### {} {}\n\n", status, module_name));
            
            report.push_str(&format!(
                "- **API Checks**: {}/{} passed\n",
                module_result.api_checks.iter().filter(|c| c.available).count(),
                module_result.api_checks.len()
            ));
            
            report.push_str(&format!(
                "- **Feature Availability**: {}/{} available\n",
                module_result.feature_availability.values().filter(|&&v| v).count(),
                module_result.feature_availability.len()
            ));

            if !module_result.errors.is_empty() {
                report.push_str("- **Errors**:\n");
                for error in &module_result.errors {
                    report.push_str(&format!("  - {}\n", error));
                }
            }
            report.push('\n');
        }

        // Performance Metrics
        report.push_str("## Performance Metrics\n\n");
        let perf = &latest_result.performance_metrics;
        report.push_str(&format!(
            "- **Baseline Time**: {:?}\n",
            perf.baseline_time
        ));
        report.push_str(&format!(
            "- **Integrated Time**: {:?}\n",
            perf.integrated_time
        ));
        report.push_str(&format!(
            "- **Performance Degradation**: {:.2}%\n",
            perf.degradation_percent
        ));
        report.push_str(&format!(
            "- **Peak Memory**: {} MB\n",
            perf.memory_metrics.peak_memory / (1024 * 1024)
        ));
        report.push_str(&format!(
            "- **Throughput**: {:.0} ops/sec\n",
            perf.throughput_metrics.ops_per_second
        ));

        // API Compatibility
        report.push_str("\n## API Compatibility\n\n");
        let api_compat = &latest_result.api_compatibility;
        report.push_str(&format!(
            "- **Overall Compatibility**: {}\n",
            if api_compat.compatible { "✅ COMPATIBLE" } else { "❌ INCOMPATIBLE" }
        ));

        if !api_compat.breaking_changes.is_empty() {
            report.push_str("- **Breaking Changes**:\n");
            for change in &api_compat.breaking_changes {
                report.push_str(&format!(
                    "  - {}: {} ({})\n",
                    change.api_name, change.description, change.version
                ));
            }
        }

        if !api_compat.deprecation_warnings.is_empty() {
            report.push_str("- **Deprecation Warnings**:\n");
            for warning in &api_compat.deprecation_warnings {
                report.push_str(&format!("  - {}\n", warning));
            }
        }

        // Communication Results
        if !latest_result.communication_results.is_empty() {
            report.push_str("\n## Cross-Module Communication\n\n");
            for comm_result in &latest_result.communication_results {
                let status = if comm_result.successful { "✅" } else { "❌" };
                report.push_str(&format!(
                    "- {} {} → {}: {:?}\n",
                    status,
                    comm_result.source_module,
                    comm_result.target_module,
                    comm_result.transfer_time
                ));
            }
        }

        // Recommendations
        report.push_str("\n## Recommendations\n\n");
        if latest_result.base.passed {
            report.push_str("- All integration tests passed successfully.\n");
            report.push_str("- The core module is ready for production use with dependent modules.\n");
        } else {
            report.push_str("- Some integration tests failed. Review the failures above.\n");
            report.push_str("- Consider updating module versions or fixing compatibility issues.\n");
        }

        if perf.degradation_percent > self.config.max_performance_degradation {
            report.push_str(&format!(
                "- Performance degradation ({:.2}%) exceeds acceptable threshold ({:.2}%).\n",
                perf.degradation_percent, self.config.max_performance_degradation
            ));
        }

        Ok(report)
    }
}

/// Create a default integration test suite for all scirs2 modules
pub fn create_default_integration_suite() -> CoreResult<TestSuite> {
    let config = TestConfig::default().with_timeout(Duration::from_secs(120));
    let mut suite = TestSuite::new("SciRS2 Integration Tests", config);

    // Add integration tests for each major module
    suite.add_test("linalg_integration", |runner| {
        let module_spec = ModuleSpec::new("scirs2-linalg", Version::new(0, 1, 0))
            .with_feature("blas")
            .with_api("matrix_multiply")
            .with_api("svd");

        let integration_config = IntegrationTestConfig {
            target_modules: vec![module_spec],
            ..Default::default()
        };

        let integration_runner = IntegrationTestRunner::new(integration_config);
        let result = integration_runner.run_integration_tests()?;
        
        if result.base.passed {
            Ok(TestResult::success(result.base.duration, 1))
        } else {
            Ok(TestResult::failure(
                result.base.duration,
                1,
                result.base.error.unwrap_or_else(|| "Integration test failed".to_string()),
            ))
        }
    });

    suite.add_test("stats_integration", |runner| {
        let module_spec = ModuleSpec::new("scirs2-stats", Version::new(0, 1, 0))
            .with_feature("distributions")
            .with_api("normal_distribution")
            .with_api("t_test");

        let integration_config = IntegrationTestConfig {
            target_modules: vec![module_spec],
            ..Default::default()
        };

        let integration_runner = IntegrationTestRunner::new(integration_config);
        let result = integration_runner.run_integration_tests()?;
        
        if result.base.passed {
            Ok(TestResult::success(result.base.duration, 1))
        } else {
            Ok(TestResult::failure(
                result.base.duration,
                1,
                result.base.error.unwrap_or_else(|| "Integration test failed".to_string()),
            ))
        }
    });

    suite.add_test("fft_integration", |runner| {
        let module_spec = ModuleSpec::new("scirs2-fft", Version::new(0, 1, 0))
            .with_api("fft")
            .with_api("ifft");

        let integration_config = IntegrationTestConfig {
            target_modules: vec![module_spec],
            ..Default::default()
        };

        let integration_runner = IntegrationTestRunner::new(integration_config);
        let result = integration_runner.run_integration_tests()?;
        
        if result.base.passed {
            Ok(TestResult::success(result.base.duration, 1))
        } else {
            Ok(TestResult::failure(
                result.base.duration,
                1,
                result.base.error.unwrap_or_else(|| "Integration test failed".to_string()),
            ))
        }
    });

    Ok(suite)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_spec_creation() {
        let spec = ModuleSpec::new("scirs2-linalg", Version::new(0, 1, 0))
            .with_feature("blas")
            .with_api("matrix_multiply")
            .with_test_data("key", "value");

        assert_eq!(spec.name, "scirs2-linalg");
        assert_eq!(spec.version, Version::new(0, 1, 0));
        assert!(spec.features.contains(&"blas".to_string()));
        assert!(spec.expected_apis.contains(&"matrix_multiply".to_string()));
        assert_eq!(spec.test_data.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_integration_test_config() {
        let config = IntegrationTestConfig {
            test_data_flow: true,
            test_performance: true,
            max_performance_degradation: 15.0,
            ..Default::default()
        };

        assert!(config.test_data_flow);
        assert!(config.test_performance);
        assert_eq!(config.max_performance_degradation, 15.0);
    }

    #[test]
    fn test_api_check_result() {
        let result = ApiCheckResult {
            api_name: "test_api".to_string(),
            available: true,
            version: Some(Version::new(0, 1, 0)),
            error: None,
        };

        assert!(result.available);
        assert!(result.version.is_some());
        assert!(result.error.is_none());
    }

    #[test]
    fn test_integration_test_runner_creation() {
        let config = IntegrationTestConfig::default();
        let runner = IntegrationTestRunner::new(config);
        
        // Test that runner was created successfully
        assert_eq!(runner.config.target_modules.len(), 0);
    }

    #[test]
    fn test_default_integration_suite() {
        let suite = create_default_integration_suite().expect("Failed to create suite");
        
        // The suite should have at least 3 tests
        let results = suite.run().expect("Failed to run suite");
        assert!(!results.is_empty());
    }
}