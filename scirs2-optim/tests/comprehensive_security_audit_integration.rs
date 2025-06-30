//! Integration tests for the comprehensive security auditor
//!
//! This test suite demonstrates how to use the security auditor in practice
//! and validates that it can detect various security issues.

use scirs2_optim::benchmarking::comprehensive_security_auditor::*;
use std::path::PathBuf;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_comprehensive_security_audit_integration() {
    // Create a temporary directory for testing
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let project_path = temp_dir.path();

    // Create a sample Cargo.toml with dependencies
    let cargo_toml_content = r#"
[package]
name = "test-project"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1.0"
tokio = "1.0"
old-time = "0.1.0"  # This might be vulnerable
chrono = "0.4"

[dev-dependencies]
tempfile = "3.0"
"#;

    fs::write(project_path.join("Cargo.toml"), cargo_toml_content)
        .expect("Failed to write Cargo.toml");

    // Create sample Rust source files with various security issues
    let src_dir = project_path.join("src");
    fs::create_dir_all(&src_dir).expect("Failed to create src directory");

    // Main file with some security issues
    let main_rs_content = r#"
use std::process::Command;

fn main() {
    println!("Hello, world!");
    
    // Potential command injection vulnerability
    let user_input = "ls";
    let output = Command::new(user_input).output().unwrap();
    
    // Hardcoded secret (should be detected)
    let api_key = "sk_test_abc123def456ghi789jkl";
    
    // Unsafe code block
    unsafe {
        let ptr = 0x12345678 as *const i32;
        let value = *ptr;
        println!("Value: {}", value);
    }
    
    // Use of weak cryptography
    use md5::Md5;
    let hash = md5::compute(b"hello world");
    
    // Panic-prone code
    let data = vec![1, 2, 3];
    let value = data[10]; // Index out of bounds
    
    println!("Output: {:?}", output);
}
"#;

    fs::write(src_dir.join("main.rs"), main_rs_content)
        .expect("Failed to write main.rs");

    // Lib file with more issues
    let lib_rs_content = r#"
//! Test library with security issues

use std::env;

pub struct Config {
    pub database_url: String,
    pub secret_key: String,
}

impl Config {
    pub fn new() -> Self {
        Self {
            // Another hardcoded secret
            database_url: "postgresql://user:password123@localhost/db".to_string(),
            secret_key: "super_secret_key_123".to_string(),
        }
    }
    
    pub fn from_env() -> Self {
        Self {
            database_url: env::var("DATABASE_URL").unwrap(),
            secret_key: env::var("SECRET_KEY").unwrap(),
        }
    }
}

// Function with unsafe operations
pub unsafe fn dangerous_operation(ptr: *mut u8, len: usize) {
    // Multiple unsafe operations in one block (should be flagged)
    let slice = std::slice::from_raw_parts_mut(ptr, len);
    slice[0] = 42;
    std::ptr::write(ptr, 100);
}

// Function that uses expect (panic-prone)
pub fn risky_function(data: Option<String>) -> String {
    data.expect("Data should always be Some")
}

// Weak crypto usage
pub fn weak_hash(input: &str) -> String {
    use sha1::{Sha1, Digest};
    let mut hasher = Sha1::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config = Config::new();
        assert!(!config.database_url.is_empty());
        assert!(!config.secret_key.is_empty());
    }
    
    #[test]
    fn test_risky_function() {
        let result = risky_function(Some("test".to_string()));
        assert_eq!(result, "test");
    }
}
"#;

    fs::write(src_dir.join("lib.rs"), lib_rs_content)
        .expect("Failed to write lib.rs");

    // Create a configuration file with potential issues
    let config_toml_content = r#"
[server]
host = "0.0.0.0"  # Potentially insecure binding to all interfaces
port = 8080
debug = true      # Debug mode in production?

[database]
host = "localhost"
port = 5432
username = "admin"
password = "admin123"  # Hardcoded password

[api]
secret_key = "my_secret_api_key_12345"  # Another secret
rate_limit = 1000
"#;

    fs::write(project_path.join("config.toml"), config_toml_content)
        .expect("Failed to write config.toml");

    // Create custom security rules for testing
    let custom_rules = vec![
        CustomSecurityRule {
            id: "CUSTOM001".to_string(),
            name: "Database Password Detection".to_string(),
            description: "Detect potential database passwords in configuration".to_string(),
            pattern: "password.*=.*\".*\"".to_string(),
            severity: SecuritySeverity::High,
            file_types: vec!["toml".to_string(), "yml".to_string(), "yaml".to_string()],
            remediation: Some("Use environment variables for sensitive configuration".to_string()),
        },
        CustomSecurityRule {
            id: "CUSTOM002".to_string(),
            name: "Debug Mode Detection".to_string(),
            description: "Detect debug mode enabled in configuration".to_string(),
            pattern: "debug.*=.*true".to_string(),
            severity: SecuritySeverity::Medium,
            file_types: vec!["toml".to_string()],
            remediation: Some("Disable debug mode in production".to_string()),
        },
    ];

    // Create security audit configuration
    let mut config = SecurityAuditConfig::default();
    config.custom_rules = custom_rules;
    config.enable_dependency_scanning = true;
    config.enable_static_analysis = true;
    config.enable_secret_detection = true;
    config.alert_threshold = SecuritySeverity::Medium;

    // Create and run the security auditor
    let mut auditor = ComprehensiveSecurityAuditor::new(config);
    let audit_result = auditor.audit_project(project_path)
        .expect("Security audit should complete successfully");

    // Validate audit results
    println!("Security Audit Results:");
    println!("======================");
    println!("Security Score: {:.2}", audit_result.security_score);
    println!("Duration: {:?}", audit_result.duration);

    // Check dependency scan results
    println!("\nDependency Analysis:");
    println!("Total dependencies: {}", audit_result.dependency_results.total_dependencies);
    println!("Vulnerable dependencies: {}", audit_result.dependency_results.vulnerable_dependencies.len());
    
    for vuln_dep in &audit_result.dependency_results.vulnerable_dependencies {
        println!("  - {}: {} ({})", vuln_dep.name, vuln_dep.current_version, 
                 format!("{:?}", vuln_dep.severity));
    }

    // Check static analysis results
    println!("\nStatic Analysis:");
    println!("Files scanned: {}", audit_result.static_analysis_results.files_scanned);
    println!("Security issues found: {}", audit_result.static_analysis_results.security_issues.len());
    
    let mut unsafe_issues = 0;
    let mut secret_issues = 0;
    let mut crypto_issues = 0;
    let mut command_injection_issues = 0;

    for issue in &audit_result.static_analysis_results.security_issues {
        println!("  - {}: {} ({}:{}) - {:?}", 
                 issue.rule_id, issue.description, 
                 issue.file.display(), issue.line, issue.severity);
        
        match issue.issue_type {
            SecurityIssueType::UnsafeCode => unsafe_issues += 1,
            SecurityIssueType::HardcodedSecret => secret_issues += 1,
            SecurityIssueType::WeakCryptography => crypto_issues += 1,
            SecurityIssueType::CommandInjection => command_injection_issues += 1,
            _ => {}
        }
    }

    // Check secret detection results
    println!("\nSecret Detection:");
    println!("Secrets found: {}", audit_result.secret_detection_results.secrets_found.len());

    // Check risk assessment
    println!("\nRisk Assessment:");
    println!("Overall risk: {:?}", audit_result.risk_assessment.overall_risk);
    println!("Risk score: {:.2}", audit_result.risk_assessment.risk_score);
    println!("Risk factors: {}", audit_result.risk_assessment.risk_factors.len());

    for factor in &audit_result.risk_assessment.risk_factors {
        println!("  - {}: {:.2} impact", factor.name, factor.impact);
    }

    // Check remediation suggestions
    println!("\nRemediation Suggestions: {}", audit_result.remediation_suggestions.len());
    for suggestion in &audit_result.remediation_suggestions {
        println!("  - {} ({}): {}", 
                 suggestion.title, 
                 format!("{:?}", suggestion.priority), 
                 suggestion.description);
    }

    // Validate that security issues were detected
    assert!(audit_result.static_analysis_results.files_scanned > 0, 
            "Should have scanned some files");
    
    assert!(audit_result.static_analysis_results.security_issues.len() > 0,
            "Should have detected security issues");
    
    assert!(unsafe_issues > 0, "Should have detected unsafe code");
    assert!(secret_issues > 0, "Should have detected hardcoded secrets");
    assert!(crypto_issues > 0, "Should have detected weak cryptography");
    assert!(command_injection_issues > 0, "Should have detected command injection risk");

    // Validate that secrets were detected
    assert!(audit_result.secret_detection_results.secrets_found.len() > 0,
            "Should have detected secrets in source code");

    // Validate security score is reasonable (should be low due to issues)
    assert!(audit_result.security_score < 0.8, 
            "Security score should be low due to detected issues");

    // Validate risk assessment
    assert!(audit_result.risk_assessment.risk_score > 0.0,
            "Should have non-zero risk score");
    
    assert!(!audit_result.risk_assessment.risk_factors.is_empty(),
            "Should have identified risk factors");

    // Validate remediation suggestions were generated
    assert!(!audit_result.remediation_suggestions.is_empty(),
            "Should have generated remediation suggestions");

    // Test report generation
    let report = auditor.generate_report(&audit_result)
        .expect("Should be able to generate report");
    
    assert!(report.contains("Security Audit Report"), "Report should contain title");
    assert!(report.contains("Security Score"), "Report should contain security score");

    println!("\n✅ Comprehensive security audit integration test passed!");
    println!("📊 Generated {} remediation suggestions", audit_result.remediation_suggestions.len());
    println!("🔍 Detected {} security issues across {} files", 
             audit_result.static_analysis_results.security_issues.len(),
             audit_result.static_analysis_results.files_scanned);
}

#[test]
fn test_dependency_parsing() {
    let cargo_content = r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
serde = "1.0"
tokio = { version = "1.0", features = ["full"] }
log = "0.4"

[dev-dependencies]
tempfile = "3.0"
"#;

    let config = SecurityAuditConfig::default();
    let auditor = ComprehensiveSecurityAuditor::new(config);
    
    let deps = auditor.parse_cargo_dependencies(cargo_content)
        .expect("Should parse dependencies successfully");

    assert!(deps.len() >= 2, "Should find at least 2 dependencies");
    
    let serde_dep = deps.iter().find(|(name, _)| name == "serde");
    assert!(serde_dep.is_some(), "Should find serde dependency");
    assert_eq!(serde_dep.unwrap().1, "1.0");

    let log_dep = deps.iter().find(|(name, _)| name == "log");
    assert!(log_dep.is_some(), "Should find log dependency");
    assert_eq!(log_dep.unwrap().1, "0.4");
}

#[test]
fn test_secret_detection_patterns() {
    let config = SecurityAuditConfig::default();
    let auditor = ComprehensiveSecurityAuditor::new(config);

    // Test cases for secret detection
    let test_cases = vec![
        ("let password = \"secret123\";", true),
        ("let api_key = 'sk_test_abc123';", true),
        ("let token = \"bearer_xyz789\";", true),
        ("let private_key = \"-----BEGIN PRIVATE KEY-----\";", true),
        ("let x = 5;", false),
        ("let message = \"hello world\";", false),
        ("let config_file = \"config.toml\";", false),
    ];

    for (code, should_detect) in test_cases {
        let detected = auditor.contains_potential_secret(code);
        assert_eq!(detected, should_detect, 
                   "Secret detection failed for: {}", code);
    }
}

#[test]
fn test_weak_crypto_detection() {
    let config = SecurityAuditConfig::default();
    let auditor = ComprehensiveSecurityAuditor::new(config);

    let test_cases = vec![
        ("use md5::Md5;", true),
        ("let hash = sha1::compute(data);", true),
        ("use des::Des;", true),
        ("encrypt_with_rc4(data);", true),
        ("use sha256::Sha256;", false),
        ("use aes::Aes256;", false),
        ("use argon2::Argon2;", false),
    ];

    for (code, should_detect) in test_cases {
        let detected = auditor.uses_weak_crypto(code);
        assert_eq!(detected, should_detect,
                   "Weak crypto detection failed for: {}", code);
    }
}

#[test]
fn test_security_score_calculation() {
    let config = SecurityAuditConfig::default();
    let auditor = ComprehensiveSecurityAuditor::new(config);

    // Create audit result with known issues
    let mut audit_result = SecurityAuditResult {
        timestamp: std::time::SystemTime::now(),
        duration: std::time::Duration::from_secs(10),
        security_score: 0.0,
        dependency_results: DependencyScanResult {
            vulnerable_dependencies: vec![
                VulnerableDependency {
                    name: "test-dep".to_string(),
                    current_version: "1.0.0".to_string(),
                    vulnerabilities: Vec::new(),
                    affected_versions: "<= 1.0.0".to_string(),
                    fixed_version: Some("1.0.1".to_string()),
                    severity: SecuritySeverity::Critical,
                    cve_ids: vec!["CVE-2024-12345".to_string()],
                }
            ],
            ..Default::default()
        },
        static_analysis_results: StaticAnalysisResult {
            security_issues: vec![
                SecurityIssue {
                    id: "test_issue".to_string(),
                    issue_type: SecurityIssueType::HardcodedSecret,
                    severity: SecuritySeverity::High,
                    file: PathBuf::from("test.rs"),
                    line: 10,
                    column: None,
                    description: "Test security issue".to_string(),
                    code_snippet: None,
                    remediation: None,
                    rule_id: "TEST001".to_string(),
                }
            ],
            ..Default::default()
        },
        secret_detection_results: SecretDetectionResult {
            secrets_found: vec![
                DetectedSecret {
                    id: "secret1".to_string(),
                    secret_type: "api_key".to_string(),
                    file: PathBuf::from("test.rs"),
                    line: 5,
                    severity: SecuritySeverity::High,
                }
            ],
        },
        license_compliance_results: LicenseComplianceResult {
            violations: vec![
                LicenseViolation {
                    package: "test-package".to_string(),
                    license: "GPL-3.0".to_string(),
                    reason: "Non-permissive license".to_string(),
                }
            ],
        },
        supply_chain_results: SupplyChainAnalysisResult::default(),
        config_security_results: ConfigSecurityResult::default(),
        policy_compliance_results: PolicyComplianceResult::default(),
        remediation_suggestions: Vec::new(),
        risk_assessment: RiskAssessment::default(),
    };

    let score = auditor.calculate_security_score(&audit_result);
    
    // Score should be low due to critical vulnerability and multiple issues
    assert!(score < 0.7, "Security score should be low with critical issues");
    assert!(score >= 0.0, "Security score should not be negative");

    // Test with no issues
    audit_result.dependency_results.vulnerable_dependencies.clear();
    audit_result.static_analysis_results.security_issues.clear();
    audit_result.secret_detection_results.secrets_found.clear();
    audit_result.license_compliance_results.violations.clear();

    let clean_score = auditor.calculate_security_score(&audit_result);
    assert!(clean_score >= 0.9, "Clean audit should have high security score");
}

#[test]
fn test_risk_assessment() {
    let config = SecurityAuditConfig::default();
    let auditor = ComprehensiveSecurityAuditor::new(config);

    let audit_result = SecurityAuditResult {
        timestamp: std::time::SystemTime::now(),
        duration: std::time::Duration::from_secs(10),
        security_score: 0.5,
        dependency_results: DependencyScanResult {
            vulnerable_dependencies: vec![
                VulnerableDependency {
                    name: "vuln-dep".to_string(),
                    current_version: "1.0.0".to_string(),
                    vulnerabilities: Vec::new(),
                    affected_versions: "<= 1.0.0".to_string(),
                    fixed_version: Some("1.0.1".to_string()),
                    severity: SecuritySeverity::High,
                    cve_ids: Vec::new(),
                }
            ],
            ..Default::default()
        },
        static_analysis_results: StaticAnalysisResult {
            security_issues: vec![
                SecurityIssue {
                    id: "issue1".to_string(),
                    issue_type: SecurityIssueType::UnsafeCode,
                    severity: SecuritySeverity::Medium,
                    file: PathBuf::from("test.rs"),
                    line: 1,
                    column: None,
                    description: "Unsafe code detected".to_string(),
                    code_snippet: None,
                    remediation: None,
                    rule_id: "SEC001".to_string(),
                }
            ],
            ..Default::default()
        },
        secret_detection_results: SecretDetectionResult {
            secrets_found: vec![
                DetectedSecret {
                    id: "secret1".to_string(),
                    secret_type: "password".to_string(),
                    file: PathBuf::from("config.rs"),
                    line: 10,
                    severity: SecuritySeverity::High,
                }
            ],
        },
        ..Default::default()
    };

    let risk_assessment = auditor.assess_risk(&audit_result)
        .expect("Risk assessment should complete successfully");

    assert!(risk_assessment.risk_score > 0.0, "Should have non-zero risk score");
    assert!(!risk_assessment.risk_factors.is_empty(), "Should identify risk factors");
    assert!(!risk_assessment.recommendations.is_empty(), "Should provide recommendations");
    assert!(!risk_assessment.mitigation_strategies.is_empty(), "Should provide mitigation strategies");

    // Check that we have the expected risk factors
    let factor_names: Vec<_> = risk_assessment.risk_factors.iter()
        .map(|f| f.name.as_str())
        .collect();
    
    assert!(factor_names.contains(&"Dependency Vulnerabilities"));
    assert!(factor_names.contains(&"Static Analysis Issues"));
    assert!(factor_names.contains(&"Exposed Secrets"));
}

#[test]
fn test_remediation_suggestions() {
    let config = SecurityAuditConfig::default();
    let auditor = ComprehensiveSecurityAuditor::new(config);

    let audit_result = SecurityAuditResult {
        timestamp: std::time::SystemTime::now(),
        duration: std::time::Duration::from_secs(10),
        security_score: 0.6,
        dependency_results: DependencyScanResult {
            vulnerable_dependencies: vec![
                VulnerableDependency {
                    name: "old-dep".to_string(),
                    current_version: "1.0.0".to_string(),
                    vulnerabilities: Vec::new(),
                    affected_versions: "<= 1.0.0".to_string(),
                    fixed_version: Some("2.0.0".to_string()),
                    severity: SecuritySeverity::High,
                    cve_ids: Vec::new(),
                }
            ],
            ..Default::default()
        },
        static_analysis_results: StaticAnalysisResult {
            security_issues: vec![
                SecurityIssue {
                    id: "hardcoded_secret".to_string(),
                    issue_type: SecurityIssueType::HardcodedSecret,
                    severity: SecuritySeverity::Critical,
                    file: PathBuf::from("main.rs"),
                    line: 15,
                    column: None,
                    description: "Hardcoded API key detected".to_string(),
                    code_snippet: Some("api_key = \"sk_test_123\"".to_string()),
                    remediation: Some("Move to environment variable".to_string()),
                    rule_id: "SEC002".to_string(),
                }
            ],
            ..Default::default()
        },
        secret_detection_results: SecretDetectionResult {
            secrets_found: vec![
                DetectedSecret {
                    id: "secret1".to_string(),
                    secret_type: "api_key".to_string(),
                    file: PathBuf::from("main.rs"),
                    line: 15,
                    severity: SecuritySeverity::Critical,
                }
            ],
        },
        ..Default::default()
    };

    let suggestions = auditor.generate_remediation_suggestions(&audit_result)
        .expect("Should generate remediation suggestions");

    assert!(!suggestions.is_empty(), "Should generate remediation suggestions");

    // Check for dependency update suggestion
    let dep_suggestion = suggestions.iter()
        .find(|s| s.title.contains("old-dep"));
    assert!(dep_suggestion.is_some(), "Should suggest updating vulnerable dependency");

    // Check for secret remediation suggestion
    let secret_suggestion = suggestions.iter()
        .find(|s| s.title.contains("secret"));
    assert!(secret_suggestion.is_some(), "Should suggest removing hardcoded secrets");

    // Validate suggestion structure
    for suggestion in &suggestions {
        assert!(!suggestion.id.is_empty(), "Suggestion should have ID");
        assert!(!suggestion.title.is_empty(), "Suggestion should have title");
        assert!(!suggestion.description.is_empty(), "Suggestion should have description");
        assert!(!suggestion.steps.is_empty(), "Suggestion should have steps");
    }
}