//! Performance regression testing binary for CI/CD integration
//!
//! This binary provides command-line interface for running comprehensive
//! performance regression tests and generating CI-compatible reports.

use clap::{Arg, ArgMatches, Command};
use ndarray::Array1;
use scirs2_optim::benchmarking::regression_tester::{
    CiReportFormat, RegressionConfig, RegressionTester,
};
use scirs2_optim::benchmarking::{BenchmarkResult, OptimizerBenchmark};
use scirs2_optim::error::{OptimError, Result};
use scirs2_optim::optimizers::*;
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[allow(dead_code)]
fn main() -> Result<()> {
    let matches = Command::new("performance_regression_tester")
        .version("0.1.0")
        .author("SCIRS2 Team")
        .about("Performance regression testing for CI/CD integration")
        .arg(
            Arg::new("baseline-dir")
                .long("baseline-dir")
                .value_name("DIR")
                .help("Directory containing performance baselines")
                .required(true),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .value_name("FILE")
                .help("Output file for regression report")
                .required(true),
        )
        .arg(
            Arg::new("format")
                .long("format")
                .value_name("FORMAT")
                .help("Output format (json, junit-xml, markdown, github-actions)")
                .default_value("json"),
        )
        .arg(
            Arg::new("degradation-threshold")
                .long("degradation-threshold")
                .value_name("PERCENT")
                .help("Performance degradation threshold percentage")
                .default_value("5.0"),
        )
        .arg(
            Arg::new("memory-threshold")
                .long("memory-threshold")
                .value_name("PERCENT")
                .help("Memory regression threshold percentage")
                .default_value("10.0"),
        )
        .arg(
            Arg::new("significance-threshold")
                .long("significance-threshold")
                .value_name("ALPHA")
                .help("Statistical significance threshold")
                .default_value("0.05"),
        )
        .arg(
            Arg::new("update-baseline")
                .long("update-baseline")
                .help("Update baseline after successful tests")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose output")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    // Parse configuration from command line arguments
    let config = parse_config(&matches)?;

    if matches.get_flag("verbose") {
        println!(
            "Starting performance regression testing with config: {:#?}",
            config
        );
    }

    // Create regression tester
    let mut tester = RegressionTester::<f64>::new(config.clone())?;

    // Load existing baselines
    if matches.get_flag("verbose") {
        println!(
            "Loading performance baselines from: {}",
            config.baseline_dir.display()
        );
    }
    tester.load_baselines()?;

    // Run comprehensive benchmarks
    if matches.get_flag("verbose") {
        println!("Running comprehensive optimizer benchmarks...");
    }
    let benchmark_results = run_comprehensive_benchmarks(matches.get_flag("verbose"))?;

    // Perform regression analysis
    if matches.get_flag("verbose") {
        println!("Performing regression analysis...");
    }
    let regression_results = tester.analyze_regressions(&benchmark_results)?;

    // Generate CI report
    let output_path = PathBuf::from(matches.get_one::<String>("output").unwrap());
    let report_format = parse_report_format(matches.get_one::<String>("format").unwrap())?;

    if matches.get_flag("verbose") {
        println!(
            "Generating {} report to: {}",
            format!("{:?}", report_format),
            output_path.display()
        );
    }

    let ci_report = generate_ci_report(&regression_results, &benchmark_results, report_format)?;
    fs::write(&output_path, ci_report)?;

    // Update baselines if requested and no regressions detected
    if matches.get_flag("update-baseline") && !has_critical_regressions(&regression_results) {
        if matches.get_flag("verbose") {
            println!("Updating performance baselines...");
        }
        tester.update_baselines(&benchmark_results)?;
        tester.save_baselines()?;
    }

    // Exit with error code if critical regressions detected
    if has_critical_regressions(&regression_results) {
        eprintln!("Critical performance regressions detected!");
        std::process::exit(1);
    }

    if matches.get_flag("verbose") {
        println!("Performance regression testing completed successfully!");
    }

    Ok(())
}

#[allow(dead_code)]
fn parse_config(matches: &ArgMatches) -> Result<RegressionConfig> {
    let baseline_dir = PathBuf::from(matches.get_one::<String>("baseline-dir").unwrap());

    let degradation_threshold: f64 = matches
        .get_one::<String>("degradation-threshold")
        .unwrap()
        .parse()
        .map_err(|_| OptimError::InvalidConfig("Invalid degradation threshold".to_string()))?;

    let memory_threshold: f64 = matches
        .get_one::<String>("memory-threshold")
        .unwrap()
        .parse()
        .map_err(|_| OptimError::InvalidConfig("Invalid memory threshold".to_string()))?;

    let significance_threshold: f64 = matches
        .get_one::<String>("significance-threshold")
        .unwrap()
        .parse()
        .map_err(|_| OptimError::InvalidConfig("Invalid significance threshold".to_string()))?;

    Ok(RegressionConfig {
        baseline_dir,
        max_history_length: 1000,
        min_baseline_samples: 10,
        significance_threshold,
        degradation_threshold,
        memory_threshold,
        enable_ci_integration: true,
        enable_alerts: true,
        outlier_sensitivity: 2.0,
        detection_algorithms: vec![
            "statistical_test".to_string(),
            "sliding_window".to_string(),
            "change_point".to_string(),
        ],
        ci_report_format: parse_report_format(matches.get_one::<String>("format").unwrap())
            .unwrap_or(CiReportFormat::Json),
    })
}

#[allow(dead_code)]
fn parse_report_format(format_str: &str) -> Result<CiReportFormat> {
    match format_str.to_lowercase().as_str() {
        "json" => Ok(CiReportFormat::Json),
        "junit-xml" => Ok(CiReportFormat::JunitXml),
        "markdown" => Ok(CiReportFormat::Markdown),
        "github-actions" => Ok(CiReportFormat::GitHubActions),
        _ => Err(OptimError::InvalidConfig(format!(
            "Unknown format: {}",
            format_str
        ))),
    }
}

#[allow(dead_code)]
fn run_comprehensive_benchmarks(verbose: bool) -> Result<Vec<BenchmarkResult<f64>>> {
    let mut benchmark = OptimizerBenchmark::new();
    benchmark.add_standard_test_functions();

    let mut all_results = Vec::new();

    // Test different optimizers
    let optimizers = vec![
        ("SGD", create_sgd_step_function()),
        ("Adam", create_adam_step_function()),
        ("AdaGrad", create_adagrad_step_function()),
        ("RMSprop", create_rmsprop_step_function()),
        ("LAMB", create_lamb_step_function()),
    ];

    for (name, mut step_fn) in optimizers {
        if verbose {
            println!("  Benchmarking {} optimizer...", name);
        }

        let results = benchmark.run_benchmark(
            name.to_string(),
            &mut step_fn,
            1000, // max iterations
            1e-6, // tolerance
        )?;

        all_results.extend(results);
    }

    Ok(all_results)
}

// Helper functions to create optimizer step functions
#[allow(dead_code)]
fn create_sgd_step_function() -> Box<dyn FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>> {
    let learning_rate = 0.01;
    Box::new(move |x: &Array1<f64>, grad: &Array1<f64>| x - &(grad * learning_rate))
}

#[allow(dead_code)]
fn create_adam_step_function() -> Box<dyn FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>> {
    let mut m = Array1::zeros(0);
    let mut v = Array1::zeros(0);
    let mut t = 0;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;
    let learning_rate = 0.001;

    Box::new(move |x: &Array1<f64>, grad: &Array1<f64>| {
        if m.len() != x.len() {
            m = Array1::zeros(x.len());
            v = Array1::zeros(x.len());
        }

        t += 1;

        // Update biased first moment estimate
        m = &m * beta1 + &(grad * (1.0 - beta1));

        // Update biased second raw moment estimate
        let grad_squared = grad.mapv(|g| g * g);
        v = &v * beta2 + &(&grad_squared * (1.0 - beta2));

        // Compute bias-corrected first moment estimate
        let m_hat = &m / (1.0 - beta1.powi(t));

        // Compute bias-corrected second raw moment estimate
        let v_hat = &v / (1.0 - beta2.powi(t));

        // Update parameters
        let denominator = v_hat.mapv(|v| v.sqrt() + epsilon);
        x - &(&m_hat / &denominator * learning_rate)
    })
}

#[allow(dead_code)]
fn create_adagrad_step_function() -> Box<dyn FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>> {
    let mut g = Array1::zeros(0);
    let learning_rate = 0.01;
    let epsilon = 1e-8;

    Box::new(move |x: &Array1<f64>, grad: &Array1<f64>| {
        if g.len() != x.len() {
            g = Array1::zeros(x.len());
        }

        // Accumulate squared gradients
        let grad_squared = grad.mapv(|g| g * g);
        g = &g + &grad_squared;

        // Update parameters
        let denominator = g.mapv(|g| g.sqrt() + epsilon);
        x - &(grad / &denominator * learning_rate)
    })
}

#[allow(dead_code)]
fn create_rmsprop_step_function() -> Box<dyn FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>> {
    let mut s = Array1::zeros(0);
    let learning_rate = 0.001;
    let epsilon = 1e-8;
    let decay_rate = 0.9;

    Box::new(move |x: &Array1<f64>, grad: &Array1<f64>| {
        if s.len() != x.len() {
            s = Array1::zeros(x.len());
        }

        // Update moving average of squared gradients
        let grad_squared = grad.mapv(|g| g * g);
        s = &s * decay_rate + &(&grad_squared * (1.0 - decay_rate));

        // Update parameters
        let denominator = s.mapv(|s| s.sqrt() + epsilon);
        x - &(grad / &denominator * learning_rate)
    })
}

#[allow(dead_code)]
fn create_lamb_step_function() -> Box<dyn FnMut(&Array1<f64>, &Array1<f64>) -> Array1<f64>> {
    let mut m = Array1::zeros(0);
    let mut v = Array1::zeros(0);
    let mut t = 0;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-6;
    let learning_rate = 0.001;

    Box::new(move |x: &Array1<f64>, grad: &Array1<f64>| {
        if m.len() != x.len() {
            m = Array1::zeros(x.len());
            v = Array1::zeros(x.len());
        }

        t += 1;

        // Update biased first moment estimate
        m = &m * beta1 + &(grad * (1.0 - beta1));

        // Update biased second raw moment estimate
        let grad_squared = grad.mapv(|g| g * g);
        v = &v * beta2 + &(&grad_squared * (1.0 - beta2));

        // Compute bias-corrected first moment estimate
        let m_hat = &m / (1.0 - beta1.powi(t));

        // Compute bias-corrected second raw moment estimate
        let v_hat = &v / (1.0 - beta2.powi(t));

        // Compute update
        let denominator = v_hat.mapv(|v| v.sqrt() + epsilon);
        let update = &m_hat / &denominator;

        // Compute trust ratio (simplified)
        let param_norm = x.mapv(|p| p * p).sum().sqrt();
        let update_norm = update.mapv(|u| u * u).sum().sqrt();

        let trust_ratio = if update_norm > 0.0 {
            (param_norm / update_norm).min(1.0)
        } else {
            1.0
        };

        // Apply update with trust ratio
        x - &(&update * (learning_rate * trust_ratio))
    })
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct RegressionTestResult {
    pub optimizer_name: String,
    pub test_name: String,
    pub status: String,
    pub baseline_performance: Option<f64>,
    pub current_performance: f64,
    pub performance_change_percent: f64,
    pub is_regression: bool,
    pub regression_severity: String,
    pub statistical_significance: f64,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct CiRegressionReport {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub regression_count: usize,
    pub test_results: Vec<RegressionTestResult>,
    pub summary: String,
    pub timestamp: u64,
    pub environment_info: HashMap<String, String>,
}

#[allow(dead_code)]
fn has_critical_regressions(results: &[RegressionTestResult]) -> bool {
    results
        .iter()
        .any(|r| r.is_regression && r.regression_severity == "critical")
}

#[allow(dead_code)]
fn generate_ci_report(
    regression_results: &[RegressionTestResult],
    _benchmark_results: &[BenchmarkResult<f64>],
    format: CiReportFormat,
) -> Result<String> {
    let passed_tests = regression_results
        .iter()
        .filter(|r| !r.is_regression)
        .count();
    let failed_tests = regression_results
        .iter()
        .filter(|r| r.is_regression)
        .count();
    let regression_count = failed_tests;

    let mut env_info = HashMap::new();
    env_info.insert("os".to_string(), std::env::consts::OS.to_string());
    env_info.insert("arch".to_string(), std::env::consts::ARCH.to_string());

    let report = CiRegressionReport {
        total_tests: regression_results.len(),
        passed_tests,
        failed_tests,
        regression_count,
        test_results: regression_results.to_vec(),
        summary: if failed_tests > 0 {
            format!(
                "{} regression(s) detected across {} test(s)",
                regression_count, failed_tests
            )
        } else {
            "All performance tests passed successfully".to_string()
        },
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        environment_info: env_info,
    };

    match format {
        CiReportFormat::Json => Ok(serde_json::to_string_pretty(&report)
            .map_err(|e| OptimError::SerializationError(e.to_string()))?),
        CiReportFormat::JunitXml => generate_junit_xml_report(&report),
        CiReportFormat::Markdown => generate_markdown_report(&report),
        CiReportFormat::GitHubActions => generate_github_actions_report(&report),
    }
}

#[allow(dead_code)]
fn generate_junit_xml_report(report: &CiRegressionReport) -> Result<String> {
    let mut xml = String::new();
    xml.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    xml.push('\n');
    xml.push_str(&format!(
        r#"<testsuite name="performance_regression" tests="{}" failures="{}" time="0">"#,
        report.total_tests, report.failed_tests
    ));
    xml.push('\n');

    for result in &report.test_results {
        let test_name = format!("{}::{}", result.optimizer_name, result.test_name);
        xml.push_str(&format!(
            r#"  <testcase name="{}" classname="performance">"#,
            test_name
        ));
        xml.push('\n');

        if result.is_regression {
            xml.push_str("    <failure type=\"performance_regression\">");
            xml.push_str(&format!(
                "Performance regression detected: {:.2}% degradation",
                result.performance_change_percent
            ));
            xml.push_str("</failure>");
            xml.push('\n');
        }

        xml.push_str("  </testcase>");
        xml.push('\n');
    }

    xml.push_str("</testsuite>");
    xml.push('\n');

    Ok(xml)
}

#[allow(dead_code)]
fn generate_markdown_report(report: &CiRegressionReport) -> Result<String> {
    let mut md = String::new();

    md.push_str("# Performance Regression Test Report\n\n");
    md.push_str(&format!("**Generated**: <t:{}:F>\n", report.timestamp));
    md.push_str(&format!("**Total Tests**: {}\n", report.total_tests));
    md.push_str(&format!("**Passed**: {} ✅\n", report.passed_tests));
    md.push_str(&format!("**Failed**: {} ❌\n\n", report.failed_tests));

    if report.failed_tests > 0 {
        md.push_str("## 🚨 Regressions Detected\n\n");
        md.push_str("| Optimizer | Test | Performance Change | Severity |\n");
        md.push_str("|-----------|------|-------------------|----------|\n");

        for result in &report.test_results {
            if result.is_regression {
                let change_icon = if result.performance_change_percent > 0.0 {
                    "📈"
                } else {
                    "📉"
                };
                md.push_str(&format!(
                    "| {} | {} | {}{:.2}% | {} |\n",
                    result.optimizer_name,
                    result.test_name,
                    change_icon,
                    result.performance_change_percent.abs(),
                    result.regression_severity
                ));
            }
        }
    } else {
        md.push_str("## ✅ All Tests Passed\n\nNo performance regressions detected!\n");
    }

    Ok(md)
}

#[allow(dead_code)]
fn generate_github_actions_report(report: &CiRegressionReport) -> Result<String> {
    // GitHub Actions format is JSON with additional workflow commands
    let json_report = serde_json::to_string_pretty(&report)
        .map_err(|e| OptimError::SerializationError(e.to_string()))?;

    let mut output = String::new();

    // Add GitHub Actions workflow commands
    if report.failed_tests > 0 {
        output.push_str(&format!(
            "::error::Performance regression detected! {} test(s) failed.\n",
            report.failed_tests
        ));

        for result in &report.test_results {
            if result.is_regression {
                output.push_str(&format!(
                    "::warning::{}::{} - {:.2}% performance degradation\n",
                    result.optimizer_name, result.test_name, result.performance_change_percent
                ));
            }
        }
    } else {
        output.push_str("::notice::All performance tests passed successfully!\n");
    }

    // Add the JSON report
    output.push('\n');
    output.push_str(&json_report);

    Ok(output)
}
