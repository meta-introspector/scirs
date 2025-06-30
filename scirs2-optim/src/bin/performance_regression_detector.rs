//! Performance Regression Detection Binary
//!
//! This binary provides a command-line interface for detecting performance regressions
//! in benchmark results using statistical analysis and configurable thresholds.

use clap::{Arg, Command};
use scirs2_optim::benchmarking::performance_regression_detector::{
    AlertThresholds, BaselineStrategy, CiCdConfig, MetricType, PerformanceRegressionDetector,
    RegressionConfig, RegressionSensitivity, StatisticalTest,
};
use scirs2_optim::error::{OptimError, Result};
use serde_json;
use std::fs;
use std::path::PathBuf;
use std::process;

fn main() {
    let matches = Command::new("performance-regression-detector")
        .version("0.1.0")
        .author("SciRS2 Development Team")
        .about("Advanced performance regression detection for continuous integration")
        .arg(
            Arg::new("benchmark-results")
                .long("benchmark-results")
                .value_name("FILE")
                .help("Path to benchmark results JSON file")
                .required(true),
        )
        .arg(
            Arg::new("baseline-dir")
                .long("baseline-dir")
                .value_name("DIR")
                .help("Directory containing baseline performance data")
                .required(true),
        )
        .arg(
            Arg::new("output-report")
                .long("output-report")
                .value_name("FILE")
                .help("Output file for regression analysis report")
                .required(true),
        )
        .arg(
            Arg::new("confidence-threshold")
                .long("confidence-threshold")
                .value_name("FLOAT")
                .help("Statistical confidence threshold (0.0-1.0)")
                .default_value("0.95"),
        )
        .arg(
            Arg::new("degradation-threshold")
                .long("degradation-threshold")
                .value_name("FLOAT")
                .help("Performance degradation threshold (e.g., 0.05 = 5%)")
                .default_value("0.05"),
        )
        .arg(
            Arg::new("sensitivity")
                .long("sensitivity")
                .value_name("LEVEL")
                .help("Regression detection sensitivity")
                .value_parser(["low", "medium", "high"])
                .default_value("medium"),
        )
        .arg(
            Arg::new("features")
                .long("features")
                .value_name("STRING")
                .help("Feature set being tested")
                .required(true),
        )
        .arg(
            Arg::new("statistical-test")
                .long("statistical-test")
                .value_name("TEST")
                .help("Statistical test to use for regression detection")
                .value_parser(["mann-whitney", "t-test", "wilcoxon", "kolmogorov-smirnov"])
                .default_value("mann-whitney"),
        )
        .arg(
            Arg::new("min-samples")
                .long("min-samples")
                .value_name("NUMBER")
                .help("Minimum number of samples required for analysis")
                .default_value("5"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose output")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("fail-on-regression")
                .long("fail-on-regression")
                .help("Exit with error code if regressions are detected")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    // Parse command line arguments
    let benchmark_results = matches.get_one::<String>("benchmark-results").unwrap();
    let baseline_dir = matches.get_one::<String>("baseline-dir").unwrap();
    let output_report = matches.get_one::<String>("output-report").unwrap();
    let confidence_threshold: f64 = matches
        .get_one::<String>("confidence-threshold")
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("Error: Invalid confidence threshold");
            process::exit(1);
        });
    let degradation_threshold: f64 = matches
        .get_one::<String>("degradation-threshold")
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("Error: Invalid degradation threshold");
            process::exit(1);
        });
    let sensitivity_str = matches.get_one::<String>("sensitivity").unwrap();
    let features = matches.get_one::<String>("features").unwrap();
    let statistical_test_str = matches.get_one::<String>("statistical-test").unwrap();
    let min_samples: usize = matches
        .get_one::<String>("min-samples")
        .unwrap()
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("Error: Invalid min-samples value");
            process::exit(1);
        });
    let verbose = matches.get_flag("verbose");
    let fail_on_regression = matches.get_flag("fail-on-regression");

    // Convert string arguments to enum values
    let sensitivity = match sensitivity_str.as_str() {
        "low" => RegressionSensitivity::Low,
        "medium" => RegressionSensitivity::Medium,
        "high" => RegressionSensitivity::High,
        _ => {
            eprintln!("Error: Invalid sensitivity level");
            process::exit(1);
        }
    };

    let statistical_test = match statistical_test_str.as_str() {
        "mann-whitney" => StatisticalTest::MannWhitneyU,
        "t-test" => StatisticalTest::StudentTTest,
        "wilcoxon" => StatisticalTest::WilcoxonSignedRank,
        "kolmogorov-smirnov" => StatisticalTest::KolmogorovSmirnov,
        _ => {
            eprintln!("Error: Invalid statistical test");
            process::exit(1);
        }
    };

    // Create regression detection configuration
    let config = RegressionConfig {
        enable_detection: true,
        confidence_threshold,
        degradation_threshold,
        min_samples,
        max_history_size: 1000,
        tracked_metrics: vec![
            MetricType::ExecutionTime,
            MetricType::MemoryUsage,
            MetricType::Throughput,
            MetricType::ConvergenceRate,
        ],
        statistical_test,
        sensitivity,
        baseline_strategy: BaselineStrategy::RollingAverage { window_size: 10 },
        alert_thresholds: AlertThresholds {
            warning_threshold: degradation_threshold,
            critical_threshold: degradation_threshold * 2.0,
            memory_threshold: 0.1, // 10% memory increase
        },
        ci_cd_config: CiCdConfig {
            enable_integration: true,
            fail_on_critical_regression: fail_on_regression,
            slack_webhook: None,
            email_notifications: vec![],
            custom_webhooks: vec![],
        },
    };

    if verbose {
        println!("🔍 Performance Regression Detection Configuration:");
        println!("  Benchmark Results: {}", benchmark_results);
        println!("  Baseline Directory: {}", baseline_dir);
        println!("  Output Report: {}", output_report);
        println!("  Confidence Threshold: {:.2}", confidence_threshold);
        println!(
            "  Degradation Threshold: {:.1}%",
            degradation_threshold * 100.0
        );
        println!("  Sensitivity: {:?}", sensitivity);
        println!("  Statistical Test: {:?}", statistical_test);
        println!("  Features: {}", features);
        println!("  Min Samples: {}", min_samples);
        println!();
    }

    // Run regression detection
    match run_regression_detection(
        benchmark_results,
        baseline_dir,
        output_report,
        features,
        config,
        verbose,
    ) {
        Ok(has_regressions) => {
            if has_regressions && fail_on_regression {
                println!("❌ Performance regressions detected - failing build");
                process::exit(1);
            } else if has_regressions {
                println!("⚠️  Performance regressions detected - check report for details");
                process::exit(0);
            } else {
                println!("✅ No performance regressions detected");
                process::exit(0);
            }
        }
        Err(e) => {
            eprintln!("❌ Error running regression detection: {}", e);
            process::exit(1);
        }
    }
}

fn run_regression_detection(
    benchmark_results: &str,
    baseline_dir: &str,
    output_report: &str,
    features: &str,
    config: RegressionConfig,
    verbose: bool,
) -> Result<bool> {
    if verbose {
        println!("📊 Loading benchmark results...");
    }

    // Load benchmark results
    let benchmark_data = load_benchmark_results(benchmark_results)?;

    if verbose {
        println!("📈 Initializing regression detector...");
    }

    // Create regression detector
    let mut detector = PerformanceRegressionDetector::new(config)?;

    // Load baseline data
    let baseline_path = PathBuf::from(baseline_dir).join(format!("baseline_{}.json", features));
    if baseline_path.exists() {
        if verbose {
            println!("📋 Loading baseline data from: {}", baseline_path.display());
        }
        detector.load_baseline(&baseline_path)?;
    } else {
        if verbose {
            println!("⚠️  No baseline data found - will establish new baseline");
        }
    }

    if verbose {
        println!("🔬 Analyzing performance data...");
    }

    // Analyze benchmark results for regressions
    let analysis_result = detector.analyze_benchmarks(&benchmark_data)?;

    if verbose {
        println!("📝 Generating regression report...");
    }

    // Generate detailed report
    let report = detector.generate_report(&analysis_result)?;

    // Save report
    let output_dir = PathBuf::from(output_report).parent().unwrap();
    fs::create_dir_all(output_dir)
        .map_err(|e| OptimError::IoError(format!("Failed to create output directory: {}", e)))?;

    let report_json = serde_json::to_string_pretty(&report).map_err(|e| {
        OptimError::SerializationError(format!("Failed to serialize report: {}", e))
    })?;

    fs::write(output_report, report_json)
        .map_err(|e| OptimError::IoError(format!("Failed to write report: {}", e)))?;

    // Check if regressions were detected
    let has_regressions = analysis_result.regression_count > 0;

    if verbose {
        println!("📊 Analysis Summary:");
        println!("  Total Benchmarks: {}", analysis_result.total_benchmarks);
        println!(
            "  Regressions Detected: {}",
            analysis_result.regression_count
        );
        println!(
            "  Critical Regressions: {}",
            analysis_result.critical_regressions
        );
        println!(
            "  Status: {}",
            if has_regressions {
                "⚠️  REGRESSIONS FOUND"
            } else {
                "✅ PASSED"
            }
        );
        println!();
        println!("📄 Report saved to: {}", output_report);
    }

    Ok(has_regressions)
}

fn load_benchmark_results(path: &str) -> Result<serde_json::Value> {
    let content = fs::read_to_string(path)
        .map_err(|e| OptimError::IoError(format!("Failed to read benchmark results: {}", e)))?;

    let data: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
        OptimError::SerializationError(format!("Failed to parse benchmark results: {}", e))
    })?;

    Ok(data)
}
