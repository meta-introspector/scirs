//! Plugin loader for dynamic loading and management of optimizer plugins
//!
//! This module provides functionality for loading plugins from various sources,
//! including compiled libraries, configuration files, and remote repositories.

use super::core::*;
use super::registry::*;
use crate::error::{OptimError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "crypto")]
use rsa::{PaddingScheme, PublicKey, RsaPublicKey};
#[cfg(feature = "crypto")]
use sha2::{Digest, Sha256};
#[cfg(feature = "crypto")]
use x509_parser::prelude::*;

/// Plugin loader for managing plugin loading and unloading
#[derive(Debug)]
pub struct PluginLoader {
    /// Loader configuration
    config: LoaderConfig,
    /// Loaded plugins
    loaded_plugins: HashMap<String, LoadedPlugin>,
    /// Plugin dependencies
    dependency_graph: DependencyGraph,
    /// Security manager
    security_manager: SecurityManager,
}

/// Configuration for plugin loader
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Enable dynamic loading
    pub enable_dynamic_loading: bool,
    /// Plugin directories to scan
    pub plugin_directories: Vec<PathBuf>,
    /// Maximum plugins to load
    pub max_plugins: usize,
    /// Load timeout
    pub load_timeout: std::time::Duration,
    /// Enable plugin sandboxing
    pub enable_sandboxing: bool,
    /// Allowed plugin sources
    pub allowed_sources: Vec<PluginSource>,
    /// Security policy
    pub security_policy: SecurityPolicy,
}

/// Loaded plugin information
#[derive(Debug)]
pub struct LoadedPlugin {
    /// Plugin info
    pub info: PluginInfo,
    /// Load source
    pub source: PluginSource,
    /// Load timestamp
    pub loaded_at: std::time::SystemTime,
    /// Plugin handle (for dynamic libraries)
    pub handle: Option<PluginHandle>,
    /// Initialization status
    pub initialized: bool,
    /// Dependencies
    pub dependencies: Vec<String>,
}

/// Plugin handle for managing dynamic libraries
#[derive(Debug)]
pub struct PluginHandle {
    /// Library path
    pub library_path: PathBuf,
    /// Entry point function name
    pub entry_point: String,
    /// Plugin metadata
    pub metadata: PluginMetadata,
}

/// Plugin metadata from manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Plugin manifest version
    pub manifest_version: String,
    /// Plugin information
    pub plugin: PluginManifest,
    /// Build information
    pub build: BuildInfo,
    /// Runtime requirements
    pub runtime: RuntimeRequirements,
}

/// Plugin manifest information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Plugin author
    pub author: String,
    /// Plugin license
    pub license: String,
    /// Plugin homepage
    pub homepage: Option<String>,
    /// Plugin entry point
    pub entry_point: String,
    /// Plugin dependencies
    pub dependencies: Vec<PluginDependency>,
    /// Supported platforms
    pub platforms: Vec<String>,
    /// Required permissions
    pub permissions: Vec<Permission>,
}

/// Build information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInfo {
    /// Rust version used
    pub rust_version: String,
    /// Target triple
    pub target: String,
    /// Build profile (debug/release)
    pub profile: String,
    /// Build timestamp
    pub timestamp: String,
    /// Compiler flags
    pub compiler_flags: Vec<String>,
}

/// Runtime requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeRequirements {
    /// Minimum Rust version
    pub min_rust_version: String,
    /// Required system libraries
    pub system_libraries: Vec<String>,
    /// Environment variables
    pub environment_variables: Vec<String>,
    /// Memory requirements (MB)
    pub memory_mb: Option<usize>,
    /// CPU requirements
    pub cpu_requirements: CpuRequirements,
}

/// CPU requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuRequirements {
    /// Minimum cores
    pub min_cores: Option<usize>,
    /// Required instruction sets
    pub instruction_sets: Vec<String>,
    /// Architecture requirements
    pub architectures: Vec<String>,
}

/// Security permissions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Permission {
    /// File system access
    FileSystem(String),
    /// Network access
    Network(String),
    /// Process execution
    ProcessExecution,
    /// System information access
    SystemInfo,
    /// Hardware access
    Hardware(String),
    /// Custom permission
    Custom(String),
}

/// Dependency graph for managing plugin dependencies
#[derive(Debug)]
pub struct DependencyGraph {
    /// Node dependencies
    dependencies: HashMap<String, Vec<String>>,
    /// Reverse dependencies
    dependents: HashMap<String, Vec<String>>,
}

/// Security manager for plugin validation
#[derive(Debug)]
pub struct SecurityManager {
    /// Security policy
    policy: SecurityPolicy,
    /// Permission validator
    permission_validator: PermissionValidator,
    /// Code scanner
    code_scanner: CodeScanner,
    /// Cryptographic validator
    crypto_validator: CryptographicValidator,
}

/// Cryptographic validator for signature verification
#[derive(Debug)]
pub struct CryptographicValidator {
    /// Trusted CAs
    trusted_cas: Vec<TrustedCA>,
    /// Signature verification configuration
    config: SignatureVerificationConfig,
}

/// Security policy configuration
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    /// Allow unsigned plugins
    pub allow_unsigned: bool,
    /// Require specific permissions
    pub required_permissions: Vec<Permission>,
    /// Forbidden permissions
    pub forbidden_permissions: Vec<Permission>,
    /// Maximum plugin size (bytes)
    pub max_plugin_size: usize,
    /// Enable code scanning
    pub enable_code_scanning: bool,
    /// Sandbox configuration
    pub sandbox_config: SandboxConfig,
    /// Cryptographic signature verification
    pub signature_verification: SignatureVerificationConfig,
    /// Trusted certificate authorities
    pub trusted_cas: Vec<TrustedCA>,
    /// Plugin allowlist (hashes of approved plugins)
    pub plugin_allowlist: Vec<String>,
    /// Enable plugin integrity monitoring
    pub integrity_monitoring: bool,
}

/// Sandbox configuration
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Enable process isolation
    pub process_isolation: bool,
    /// Memory limit (bytes)
    pub memory_limit: usize,
    /// CPU time limit (seconds)
    pub cpu_time_limit: f64,
    /// Network access allowed
    pub network_access: bool,
    /// File system access paths
    pub filesystem_access: Vec<PathBuf>,
}

/// Permission validator
#[derive(Debug)]
pub struct PermissionValidator {
    /// Validation rules
    rules: Vec<ValidationRule>,
}

/// Validation rule for permissions
#[derive(Debug)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Permission pattern
    pub permission_pattern: String,
    /// Validation function
    pub validator: fn(&Permission) -> bool,
}

/// Code scanner for malware detection
#[derive(Debug)]
pub struct CodeScanner {
    /// Scanning rules
    rules: Vec<ScanningRule>,
    /// Signature database
    signatures: Vec<MalwareSignature>,
}

/// Code scanning rule
#[derive(Debug)]
pub struct ScanningRule {
    /// Rule name
    pub name: String,
    /// Pattern to match
    pub pattern: String,
    /// Severity level
    pub severity: ScanSeverity,
}

/// Malware signature
#[derive(Debug)]
pub struct MalwareSignature {
    /// Signature name
    pub name: String,
    /// Signature hash
    pub hash: String,
    /// Description
    pub description: String,
}

/// Scan severity levels
#[derive(Debug, Clone)]
pub enum ScanSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Cryptographic signature verification configuration
#[derive(Debug, Clone)]
pub struct SignatureVerificationConfig {
    /// Enable signature verification
    pub enabled: bool,
    /// Required signature algorithm
    pub required_algorithm: SignatureAlgorithm,
    /// Minimum key size (bits)
    pub min_key_size: usize,
    /// Allow self-signed certificates
    pub allow_self_signed: bool,
    /// Certificate chain validation depth
    pub max_chain_depth: usize,
    /// Certificate revocation checking
    pub check_revocation: bool,
    /// Signature validation timeout
    pub validation_timeout: std::time::Duration,
}

/// Signature algorithms
#[derive(Debug, Clone, Copy)]
pub enum SignatureAlgorithm {
    RSA2048_SHA256,
    RSA3072_SHA256,
    RSA4096_SHA256,
    ECDSA_P256_SHA256,
    ECDSA_P384_SHA384,
    Ed25519,
}

/// Trusted Certificate Authority
#[derive(Debug, Clone)]
pub struct TrustedCA {
    /// CA name
    pub name: String,
    /// CA public key (PEM format)
    pub public_key: String,
    /// CA certificate (PEM format)
    pub certificate: String,
    /// Key usage constraints
    pub key_usage: Vec<KeyUsage>,
    /// Valid from date
    pub valid_from: std::time::SystemTime,
    /// Valid until date
    pub valid_until: std::time::SystemTime,
}

/// Key usage types
#[derive(Debug, Clone, Copy)]
pub enum KeyUsage {
    DigitalSignature,
    ContentCommitment,
    KeyEncipherment,
    DataEncipherment,
    KeyAgreement,
    KeyCertSign,
    CRLSign,
    CodeSigning,
}

/// Plugin signature information
#[derive(Debug, Clone)]
pub struct PluginSignature {
    /// Signature algorithm used
    pub algorithm: SignatureAlgorithm,
    /// Signature bytes
    pub signature: Vec<u8>,
    /// Signing certificate chain
    pub certificate_chain: Vec<String>,
    /// Signature timestamp
    pub timestamp: std::time::SystemTime,
    /// Signer information
    pub signer_info: SignerInfo,
}

/// Signer information
#[derive(Debug, Clone)]
pub struct SignerInfo {
    /// Signer name
    pub name: String,
    /// Signer email
    pub email: String,
    /// Organization
    pub organization: String,
    /// Country
    pub country: String,
}

/// Signature verification result
#[derive(Debug, Clone)]
pub struct SignatureVerificationResult {
    /// Signature is valid
    pub valid: bool,
    /// Verification errors
    pub errors: Vec<String>,
    /// Verification warnings  
    pub warnings: Vec<String>,
    /// Certificate chain validation result
    pub chain_valid: bool,
    /// Signer information
    pub signer_info: Option<SignerInfo>,
    /// Signature algorithm used
    pub algorithm: Option<SignatureAlgorithm>,
}

/// Plugin load result
#[derive(Debug)]
pub struct PluginLoadResult {
    /// Whether loading was successful
    pub success: bool,
    /// Loaded plugin information
    pub plugin_info: Option<PluginInfo>,
    /// Load errors
    pub errors: Vec<String>,
    /// Load warnings
    pub warnings: Vec<String>,
    /// Load time
    pub load_time: std::time::Duration,
    /// Security scan results
    pub security_results: SecurityScanResult,
}

/// Security scan result
#[derive(Debug, Clone)]
pub struct SecurityScanResult {
    /// Scan successful
    pub scan_successful: bool,
    /// Security threats found
    pub threats: Vec<SecurityThreat>,
    /// Permission violations
    pub permission_violations: Vec<String>,
    /// Overall security score (0.0 to 1.0)
    pub security_score: f64,
    /// Signature verification result
    pub signature_verification: Option<SignatureVerificationResult>,
    /// Plugin hash (for allowlist checking)
    pub plugin_hash: String,
    /// Integrity check result
    pub integrity_valid: bool,
}

impl Default for SecurityScanResult {
    fn default() -> Self {
        Self {
            scan_successful: false,
            threats: Vec::new(),
            permission_violations: Vec::new(),
            security_score: 0.0,
            signature_verification: None,
            plugin_hash: String::new(),
            integrity_valid: false,
        }
    }
}

/// Security threat information
#[derive(Debug, Clone)]
pub struct SecurityThreat {
    /// Threat type
    pub threat_type: ThreatType,
    /// Threat description
    pub description: String,
    /// Severity level
    pub severity: ScanSeverity,
    /// Location in code
    pub location: Option<String>,
}

/// Types of security threats
#[derive(Debug, Clone)]
pub enum ThreatType {
    /// Suspicious function call
    SuspiciousFunction,
    /// Unsafe code block
    UnsafeCode,
    /// Network access
    NetworkAccess,
    /// File system access
    FileSystemAccess,
    /// Process execution
    ProcessExecution,
    /// Invalid cryptographic signature
    InvalidSignature,
    /// Unsigned plugin when signature required
    UnsignedPlugin,
    /// Plugin not in allowlist
    UnauthorizedPlugin,
    /// Expired certificate
    ExpiredCertificate,
    /// Revoked certificate
    RevokedCertificate,
    /// Weak cryptographic algorithms
    WeakCryptography,
    /// File integrity violation
    IntegrityViolation,
    /// Unknown/custom threat
    Unknown(String),
}

impl PluginLoader {
    /// Create a new plugin loader
    pub fn new(config: LoaderConfig) -> Self {
        Self {
            config,
            loaded_plugins: HashMap::new(),
            dependency_graph: DependencyGraph::new(),
            security_manager: SecurityManager::new(config.security_policy.clone()),
        }
    }

    /// Load plugin from file
    pub fn load_plugin_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<PluginLoadResult> {
        let start_time = std::time::Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        let path = path.as_ref();
        if !path.exists() {
            return Ok(PluginLoadResult {
                success: false,
                plugin_info: None,
                errors: vec![format!("Plugin file not found: {}", path.display())],
                warnings,
                load_time: start_time.elapsed(),
                security_results: SecurityScanResult::default(),
            });
        }

        // Load plugin metadata
        let metadata = match self.load_plugin_metadata(path) {
            Ok(metadata) => metadata,
            Err(e) => {
                return Ok(PluginLoadResult {
                    success: false,
                    plugin_info: None,
                    errors: vec![format!("Failed to load metadata: {}", e)],
                    warnings,
                    load_time: start_time.elapsed(),
                    security_results: SecurityScanResult::default(),
                });
            }
        };

        // Perform security scan
        let security_results = self.security_manager.scan_plugin(path, &metadata)?;

        if !security_results.scan_successful || security_results.security_score < 0.5 {
            errors.push("Plugin failed security scan".to_string());
            return Ok(PluginLoadResult {
                success: false,
                plugin_info: None,
                errors,
                warnings,
                load_time: start_time.elapsed(),
                security_results,
            });
        }

        // Check dependencies
        if let Err(e) = self.check_dependencies(&metadata.plugin.dependencies) {
            errors.push(format!("Dependency check failed: {}", e));
        }

        // Load the plugin
        let plugin_info = PluginInfo {
            name: metadata.plugin.name.clone(),
            version: metadata.plugin.version.clone(),
            author: metadata.plugin.author.clone(),
            description: metadata.plugin.description.clone(),
            homepage: metadata.plugin.homepage.clone(),
            license: metadata.plugin.license.clone(),
            supported_types: vec![DataType::F32, DataType::F64], // Default
            category: PluginCategory::FirstOrder,                // Default
            tags: Vec::new(),
            min_sdk_version: metadata.runtime.min_rust_version.clone(),
            dependencies: metadata.plugin.dependencies.clone(),
        };

        // Create loaded plugin entry
        let loaded_plugin = LoadedPlugin {
            info: plugin_info.clone(),
            source: PluginSource::Local(path.to_path_buf()),
            loaded_at: std::time::SystemTime::now(),
            handle: Some(PluginHandle {
                library_path: path.to_path_buf(),
                entry_point: metadata.plugin.entry_point.clone(),
                metadata: metadata.clone(),
            }),
            initialized: false,
            dependencies: metadata
                .plugin
                .dependencies
                .iter()
                .map(|dep| dep.name.clone())
                .collect(),
        };

        // Add to loaded plugins
        self.loaded_plugins
            .insert(metadata.plugin.name.clone(), loaded_plugin);

        // Update dependency graph
        self.dependency_graph.add_plugin(
            &metadata.plugin.name,
            &metadata
                .plugin
                .dependencies
                .iter()
                .map(|dep| dep.name.clone())
                .collect::<Vec<_>>(),
        );

        Ok(PluginLoadResult {
            success: errors.is_empty(),
            plugin_info: Some(plugin_info),
            errors,
            warnings,
            load_time: start_time.elapsed(),
            security_results,
        })
    }

    /// Load plugin from configuration
    pub fn load_plugin_from_config(&mut self, config: PluginConfig) -> Result<PluginLoadResult> {
        let start_time = std::time::Instant::now();
        let mut warnings = Vec::new();

        // Check if already loaded
        if let Some(loaded) = self.loaded_plugins.get(&config.name) {
            warnings.push(format!("Plugin '{}' is already loaded", config.name));
            return Ok(PluginLoadResult {
                success: true,
                plugin_info: Some(loaded.info.clone()),
                errors: Vec::new(),
                warnings,
                load_time: start_time.elapsed(),
                security_results: SecurityScanResult::default(),
            });
        }

        // Load from appropriate source
        let result = match &config.source {
            PluginSourceConfig::File(path) => self.load_plugin_from_file(path),
            PluginSourceConfig::Git { url, branch } => {
                self.load_plugin_from_git(url, branch.as_deref(), &config)
            }
            PluginSourceConfig::Registry { name, version } => {
                self.load_plugin_from_registry(name, version.as_deref(), &config)
            }
            PluginSourceConfig::Http(url) => self.load_plugin_from_http(url, &config),
        }?;

        // Apply additional configuration if loading was successful
        if result.success {
            if let Some(plugin_info) = &result.plugin_info {
                // Store configuration parameters
                if let Some(loaded) = self.loaded_plugins.get_mut(&plugin_info.name) {
                    // Apply configuration parameters (placeholder for future enhancement)
                    loaded.initialized = true;
                }
            }
        }

        Ok(result)
    }

    /// Unload plugin
    pub fn unload_plugin(&mut self, name: &str) -> Result<()> {
        // Check dependencies before unloading
        if let Some(dependents) = self.dependency_graph.get_dependents(name) {
            if !dependents.is_empty() {
                return Err(OptimError::PluginStillInUse(format!(
                    "Plugin '{}' is still used by: {}",
                    name,
                    dependents.join(", ")
                )));
            }
        }

        // Remove from loaded plugins
        self.loaded_plugins.remove(name);

        // Update dependency graph
        self.dependency_graph.remove_plugin(name);

        Ok(())
    }

    /// List loaded plugins
    pub fn list_loaded_plugins(&self) -> Vec<&PluginInfo> {
        self.loaded_plugins.values().map(|p| &p.info).collect()
    }

    /// Get plugin load status
    pub fn get_load_status(&self, name: &str) -> Option<&LoadedPlugin> {
        self.loaded_plugins.get(name)
    }

    /// Discover plugins in configured directories
    pub fn discover_plugins(&mut self) -> Result<Vec<PluginLoadResult>> {
        let mut results = Vec::new();

        for directory in &self.config.plugin_directories {
            if directory.exists() && directory.is_dir() {
                let discovered = self.discover_plugins_in_directory(directory)?;
                results.extend(discovered);
            }
        }

        Ok(results)
    }

    // Private helper methods

    /// Load plugin from Git repository
    fn load_plugin_from_git(
        &mut self,
        url: &str,
        branch: Option<&str>,
        config: &PluginConfig,
    ) -> Result<PluginLoadResult> {
        let start_time = std::time::Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Create temporary directory for clone
        let temp_dir = std::env::temp_dir().join(format!(
            "plugin_{}_{}",
            config.name,
            start_time.elapsed().as_nanos()
        ));

        // Clone repository (simplified implementation - would need git2 crate)
        // This is a placeholder implementation
        errors.push(
            "Git plugin loading not yet fully implemented - requires git2 dependency".to_string(),
        );
        warnings.push("Git cloning functionality requires additional dependencies".to_string());

        // TODO: Implement actual Git cloning:
        // 1. git2::Repository::clone(url, &temp_dir)
        // 2. If branch specified, checkout branch
        // 3. Find plugin files in repository
        // 4. Load plugin from found files
        // 5. Clean up temporary directory

        let security_results = SecurityScanResult::default();

        Ok(PluginLoadResult {
            success: false,
            plugin_info: None,
            errors,
            warnings,
            load_time: start_time.elapsed(),
            security_results,
        })
    }

    /// Load plugin from package registry
    fn load_plugin_from_registry(
        &mut self,
        name: &str,
        version: Option<&str>,
        config: &PluginConfig,
    ) -> Result<PluginLoadResult> {
        let start_time = std::time::Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // TODO: Implement registry plugin loading:
        // 1. Query registry API for plugin information
        // 2. Download plugin package
        // 3. Verify package signature if required
        // 4. Extract package to temporary directory
        // 5. Load plugin from extracted files
        // 6. Clean up temporary files

        errors.push("Registry plugin loading not yet fully implemented".to_string());
        warnings.push("Registry loading functionality is under development".to_string());

        let security_results = SecurityScanResult::default();

        Ok(PluginLoadResult {
            success: false,
            plugin_info: None,
            errors,
            warnings,
            load_time: start_time.elapsed(),
            security_results,
        })
    }

    /// Load plugin from HTTP URL
    fn load_plugin_from_http(
        &mut self,
        url: &str,
        config: &PluginConfig,
    ) -> Result<PluginLoadResult> {
        let start_time = std::time::Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // TODO: Implement HTTP plugin loading:
        // 1. Download plugin from URL (with reqwest or similar)
        // 2. Verify content-type and size limits
        // 3. Perform security scanning on downloaded content
        // 4. Save to temporary file
        // 5. Load plugin from temporary file
        // 6. Clean up temporary file

        errors.push(
            "HTTP plugin loading not yet fully implemented - requires HTTP client dependency"
                .to_string(),
        );
        warnings
            .push("HTTP downloading functionality requires additional dependencies".to_string());

        let security_results = SecurityScanResult::default();

        Ok(PluginLoadResult {
            success: false,
            plugin_info: None,
            errors,
            warnings,
            load_time: start_time.elapsed(),
            security_results,
        })
    }

    fn load_plugin_metadata(&self, path: &Path) -> Result<PluginMetadata> {
        // Look for plugin.toml or plugin.json in the same directory
        let manifest_path = path
            .parent()
            .map(|p| p.join("plugin.toml"))
            .unwrap_or_else(|| PathBuf::from("plugin.toml"));

        if manifest_path.exists() {
            let content = std::fs::read_to_string(&manifest_path)?;
            // TODO: Implement TOML parsing when toml crate is available
            // For now, return a default metadata with a warning
            let _content = content; // Suppress unused variable warning
            Ok(PluginMetadata::default_for_path(path))
        } else {
            // Create default metadata
            Ok(PluginMetadata::default_for_path(path))
        }
    }

    fn check_dependencies(&self, dependencies: &[PluginDependency]) -> Result<()> {
        for dep in dependencies {
            if !dep.optional && !self.is_dependency_satisfied(dep) {
                return Err(OptimError::MissingDependency(dep.name.clone()));
            }
        }
        Ok(())
    }

    fn is_dependency_satisfied(&self, dependency: &PluginDependency) -> bool {
        match dependency.dependency_type {
            DependencyType::Plugin => self.loaded_plugins.contains_key(&dependency.name),
            DependencyType::SystemLibrary => {
                // Check if system library is available
                // This would use platform-specific detection
                true // Simplified
            }
            DependencyType::Crate => {
                // Check if Rust crate is available
                // This would check the cargo metadata
                true // Simplified
            }
            DependencyType::Runtime => {
                // Check runtime requirements
                true // Simplified
            }
        }
    }

    fn discover_plugins_in_directory(&mut self, directory: &Path) -> Result<Vec<PluginLoadResult>> {
        let mut results = Vec::new();

        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();

            if self.is_plugin_file(&path) {
                let result = self.load_plugin_from_file(&path)?;
                results.push(result);
            }
        }

        Ok(results)
    }

    fn is_plugin_file(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension() {
            match extension.to_str() {
                Some("so") | Some("dylib") | Some("dll") => true,
                Some("toml") if path.file_stem().and_then(|s| s.to_str()) == Some("plugin") => true,
                _ => false,
            }
        } else {
            false
        }
    }
}

/// Plugin configuration for loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Plugin source
    pub source: PluginSourceConfig,
    /// Plugin name
    pub name: String,
    /// Plugin version requirement
    pub version: Option<String>,
    /// Configuration parameters
    pub config: HashMap<String, serde_json::Value>,
    /// Enable automatic updates
    pub auto_update: bool,
}

/// Plugin source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginSourceConfig {
    /// Local file path
    File(PathBuf),
    /// Git repository
    Git { url: String, branch: Option<String> },
    /// Package registry
    Registry {
        name: String,
        version: Option<String>,
    },
    /// HTTP/HTTPS URL
    Http(String),
}

// Implementations for supporting structures

impl DependencyGraph {
    fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
            dependents: HashMap::new(),
        }
    }

    fn add_plugin(&mut self, name: &str, dependencies: &[String]) {
        self.dependencies
            .insert(name.to_string(), dependencies.to_vec());

        for dep in dependencies {
            self.dependents
                .entry(dep.clone())
                .or_insert_with(Vec::new)
                .push(name.to_string());
        }
    }

    fn remove_plugin(&mut self, name: &str) {
        if let Some(dependencies) = self.dependencies.remove(name) {
            for dep in dependencies {
                if let Some(dependents) = self.dependents.get_mut(&dep) {
                    dependents.retain(|x| x != name);
                }
            }
        }
        self.dependents.remove(name);
    }

    fn get_dependents(&self, name: &str) -> Option<&Vec<String>> {
        self.dependents.get(name)
    }
}

impl SecurityManager {
    fn new(policy: SecurityPolicy) -> Self {
        let crypto_validator = CryptographicValidator::new(
            policy.trusted_cas.clone(),
            policy.signature_verification.clone(),
        );

        Self {
            policy,
            permission_validator: PermissionValidator::new(),
            code_scanner: CodeScanner::new(),
            crypto_validator,
        }
    }

    fn scan_plugin(&self, path: &Path, metadata: &PluginMetadata) -> Result<SecurityScanResult> {
        let mut threats = Vec::new();
        let mut permission_violations = Vec::new();

        // Calculate plugin hash for integrity checking
        let plugin_hash = self.calculate_plugin_hash(path)?;

        // Check if plugin is in allowlist
        let mut integrity_valid = true;
        if !self.policy.plugin_allowlist.is_empty() {
            integrity_valid = self.policy.plugin_allowlist.contains(&plugin_hash);
            if !integrity_valid {
                threats.push(SecurityThreat {
                    threat_type: ThreatType::UnauthorizedPlugin,
                    description: "Plugin not in approved allowlist".to_string(),
                    severity: ScanSeverity::Critical,
                });
            }
        }

        // Perform cryptographic signature verification
        let signature_verification = if self.policy.signature_verification.enabled {
            Some(
                self.crypto_validator
                    .verify_plugin_signature(path, metadata)?,
            )
        } else {
            None
        };

        // Check if unsigned plugins are allowed
        if !self.policy.allow_unsigned {
            match &signature_verification {
                Some(sig_result) if !sig_result.valid => {
                    threats.push(SecurityThreat {
                        threat_type: ThreatType::InvalidSignature,
                        description: "Plugin signature verification failed".to_string(),
                        severity: ScanSeverity::Critical,
                    });
                }
                None => {
                    threats.push(SecurityThreat {
                        threat_type: ThreatType::UnsignedPlugin,
                        description: "Plugin is unsigned but policy requires signatures"
                            .to_string(),
                        severity: ScanSeverity::Critical,
                    });
                }
                _ => {}
            }
        }

        // Check permissions
        for permission in &metadata.plugin.permissions {
            if !self.permission_validator.validate_permission(permission) {
                permission_violations.push(format!("Invalid permission: {:?}", permission));
            }

            if self.policy.forbidden_permissions.contains(permission) {
                permission_violations.push(format!("Forbidden permission: {:?}", permission));
            }
        }

        // Perform code scanning if enabled
        if self.policy.enable_code_scanning {
            let scan_threats = self.code_scanner.scan_code(path)?;
            threats.extend(scan_threats);
        }

        // Calculate comprehensive security score
        let security_score = self.calculate_comprehensive_security_score(
            &threats,
            &permission_violations,
            &signature_verification,
            integrity_valid,
        );

        let scan_successful = permission_violations.is_empty()
            && threats
                .iter()
                .all(|t| !matches!(t.severity, ScanSeverity::Critical))
            && integrity_valid;

        Ok(SecurityScanResult {
            scan_successful,
            threats,
            permission_violations,
            security_score,
            signature_verification,
            plugin_hash,
            integrity_valid,
        })
    }

    fn calculate_security_score(&self, threats: &[SecurityThreat], violations: &[String]) -> f64 {
        let threat_penalty = threats.len() as f64 * 0.1;
        let violation_penalty = violations.len() as f64 * 0.2;
        (1.0 - threat_penalty - violation_penalty).max(0.0)
    }

    fn calculate_comprehensive_security_score(
        &self,
        threats: &[SecurityThreat],
        violations: &[String],
        signature_verification: &Option<SignatureVerificationResult>,
        integrity_valid: bool,
    ) -> f64 {
        let mut score = 1.0;

        // Penalty for threats based on severity
        for threat in threats {
            let penalty = match threat.severity {
                ScanSeverity::Critical => 0.5,
                ScanSeverity::Error => 0.3,
                ScanSeverity::Warning => 0.1,
                ScanSeverity::Info => 0.05,
            };
            score -= penalty;
        }

        // Penalty for permission violations
        score -= violations.len() as f64 * 0.1;

        // Penalty for signature verification failures
        if let Some(sig_result) = signature_verification {
            if !sig_result.valid {
                score -= 0.4;
            }
            if !sig_result.chain_valid {
                score -= 0.2;
            }
        }

        // Penalty for integrity violations
        if !integrity_valid {
            score -= 0.3;
        }

        score.max(0.0).min(1.0)
    }

    fn calculate_plugin_hash(&self, path: &Path) -> Result<String> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        #[cfg(feature = "crypto")]
        {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&buffer);
            Ok(format!("{:x}", hasher.finalize()))
        }
        #[cfg(not(feature = "crypto"))]
        {
            // Fallback hash calculation without crypto feature
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            buffer.hash(&mut hasher);
            Ok(format!("{:x}", hasher.finish()))
        }
    }
}

impl PermissionValidator {
    fn new() -> Self {
        Self { rules: Vec::new() }
    }

    fn validate_permission(&self, permission: &Permission) -> bool {
        // Basic permission validation
        match permission {
            Permission::FileSystem(path) => {
                // Validate file system access
                !path.contains("..") && !path.starts_with('/')
            }
            Permission::Network(addr) => {
                // Validate network access
                !addr.is_empty()
            }
            _ => true,
        }
    }
}

impl CryptographicValidator {
    fn new(trusted_cas: Vec<TrustedCA>, config: SignatureVerificationConfig) -> Self {
        Self {
            trusted_cas,
            config,
        }
    }

    fn verify_plugin_signature(
        &self,
        path: &Path,
        metadata: &PluginMetadata,
    ) -> Result<SignatureVerificationResult> {
        // Look for signature file (plugin.sig or similar)
        let sig_path = path
            .parent()
            .map(|p| p.join("plugin.sig"))
            .unwrap_or_else(|| PathBuf::from("plugin.sig"));

        if !self.config.enabled {
            return Ok(SignatureVerificationResult {
                valid: true,
                errors: Vec::new(),
                warnings: vec!["Signature verification disabled".to_string()],
                chain_valid: true,
                signer_info: None,
                algorithm: None,
            });
        }

        if !sig_path.exists() {
            return Ok(SignatureVerificationResult {
                valid: false,
                errors: vec!["No signature file found".to_string()],
                warnings: Vec::new(),
                chain_valid: false,
                signer_info: None,
                algorithm: None,
            });
        }

        // In a real implementation, this would:
        // 1. Load the signature file
        // 2. Parse the signature and certificate chain
        // 3. Verify the signature against the plugin file
        // 4. Validate the certificate chain against trusted CAs
        // 5. Check certificate validity dates and revocation status

        #[cfg(feature = "crypto")]
        {
            // Placeholder for actual cryptographic verification
            // This would use RSA/ECDSA verification with x509 certificate validation
            Ok(SignatureVerificationResult {
                valid: false,
                errors: vec![
                    "Cryptographic signature verification not yet fully implemented".to_string(),
                ],
                warnings: vec![
                    "Full crypto implementation requires additional dependencies".to_string(),
                ],
                chain_valid: false,
                signer_info: None,
                algorithm: Some(self.config.required_algorithm),
            })
        }
        #[cfg(not(feature = "crypto"))]
        {
            Ok(SignatureVerificationResult {
                valid: false,
                errors: vec!["Cryptographic features not enabled".to_string()],
                warnings: vec![
                    "Build with --features crypto for signature verification".to_string()
                ],
                chain_valid: false,
                signer_info: None,
                algorithm: None,
            })
        }
    }
}

impl CodeScanner {
    fn new() -> Self {
        Self {
            rules: Vec::new(),
            signatures: Vec::new(),
        }
    }

    fn scan_code(&self, _path: &Path) -> Result<Vec<SecurityThreat>> {
        // In a real implementation, this would scan the plugin code
        // for suspicious patterns and known malware signatures
        Ok(Vec::new())
    }
}

impl PluginMetadata {
    fn default_for_path(path: &Path) -> Self {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        Self {
            manifest_version: "1.0".to_string(),
            plugin: PluginManifest {
                name: name.clone(),
                version: "0.1.0".to_string(),
                description: "Auto-generated plugin manifest".to_string(),
                author: "Unknown".to_string(),
                license: "MIT".to_string(),
                homepage: None,
                entry_point: "plugin_main".to_string(),
                dependencies: Vec::new(),
                platforms: vec!["*".to_string()],
                permissions: Vec::new(),
            },
            build: BuildInfo {
                rust_version: "1.70.0".to_string(),
                target: std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()),
                profile: "release".to_string(),
                timestamp: format!("{:?}", std::time::SystemTime::now()),
                compiler_flags: Vec::new(),
            },
            runtime: RuntimeRequirements {
                min_rust_version: "1.70.0".to_string(),
                system_libraries: Vec::new(),
                environment_variables: Vec::new(),
                memory_mb: None,
                cpu_requirements: CpuRequirements {
                    min_cores: None,
                    instruction_sets: Vec::new(),
                    architectures: Vec::new(),
                },
            },
        }
    }
}

impl Default for SecurityScanResult {
    fn default() -> Self {
        Self {
            scan_successful: true,
            threats: Vec::new(),
            permission_violations: Vec::new(),
            security_score: 1.0,
        }
    }
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            enable_dynamic_loading: true,
            plugin_directories: vec![PathBuf::from("./plugins")],
            max_plugins: 100,
            load_timeout: std::time::Duration::from_secs(30),
            enable_sandboxing: false,
            allowed_sources: vec![PluginSource::Local(PathBuf::from("./plugins"))],
            security_policy: SecurityPolicy::default(),
        }
    }
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            allow_unsigned: true,
            required_permissions: Vec::new(),
            forbidden_permissions: vec![Permission::ProcessExecution, Permission::SystemInfo],
            max_plugin_size: 100 * 1024 * 1024, // 100MB
            enable_code_scanning: false,
            sandbox_config: SandboxConfig::default(),
            signature_verification: SignatureVerificationConfig::default(),
            trusted_cas: Vec::new(),
            plugin_allowlist: Vec::new(),
            integrity_monitoring: false,
        }
    }
}

impl Default for SignatureVerificationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            required_algorithm: SignatureAlgorithm::RSA2048_SHA256,
            min_key_size: 2048,
            allow_self_signed: false,
            max_chain_depth: 5,
            check_revocation: false,
            validation_timeout: std::time::Duration::from_secs(30),
        }
    }
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            process_isolation: false,
            memory_limit: 512 * 1024 * 1024, // 512MB
            cpu_time_limit: 60.0,            // 60 seconds
            network_access: false,
            filesystem_access: vec![PathBuf::from("./tmp")],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_loader_creation() {
        let config = LoaderConfig::default();
        let loader = PluginLoader::new(config);
        assert_eq!(loader.loaded_plugins.len(), 0);
    }

    #[test]
    fn test_dependency_graph() {
        let mut graph = DependencyGraph::new();
        graph.add_plugin("plugin_a", &["dep1".to_string(), "dep2".to_string()]);

        let dependents = graph.get_dependents("dep1");
        assert!(dependents.is_some());
        assert_eq!(dependents.unwrap().len(), 1);
        assert_eq!(dependents.unwrap()[0], "plugin_a");
    }

    #[test]
    fn test_security_scan_result() {
        let result = SecurityScanResult::default();
        assert!(result.scan_successful);
        assert_eq!(result.security_score, 1.0);
    }
}
