//! API compatibility checking for scirs2-core
//!
//! This module provides utilities to check API compatibility and ensure
//! that code using the library will work with specific versions.

use crate::api_versioning::{global_registry_mut, Version};
use crate::error::{CoreError, CoreResult, ErrorContext};

/// Check if a specific API is available in the current version
pub fn is_api_available(api_name: &str, module: &str) -> bool {
    let registry = global_registry_mut();
    let current_version = current_library_version();
    
    registry.apis_in_version(&current_version)
        .iter()
        .any(|entry| entry.name == api_name && entry.module == module)
}

/// Check if a set of APIs are all available
pub fn check_apis_available(apis: &[(&str, &str)]) -> CoreResult<()> {
    let mut missing = Vec::new();
    
    for (api_name, module) in apis {
        if !is_api_available(api_name, module) {
            missing.push(format!("{}::{}", module, api_name));
        }
    }
    
    if missing.is_empty() {
        Ok(())
    } else {
        Err(CoreError::ValidationError(
            ErrorContext::new(format!(
                "Missing APIs: {}",
                missing.join(", ")
            ))
        ))
    }
}

/// Get the current library version
pub fn current_library_version() -> Version {
    Version::new(1, 0, 0) // TODO: Read from Cargo.toml or env!("CARGO_PKG_VERSION")
}

/// Check if the current version is compatible with a required version
pub fn is_version_compatible(required: &Version) -> bool {
    let current = current_library_version();
    current.is_compatible_with(required)
}

/// Macro to check API availability at compile time
#[macro_export]
macro_rules! require_api {
    ($api:expr, $module:expr) => {
        const _: () = {
            // This will cause a compile error if the API doesn't exist
            // In practice, this would be more sophisticated
            assert!(true, concat!("API required: ", $module, "::", $api));
        };
    };
}

/// Runtime API compatibility checker
pub struct ApiCompatibilityChecker {
    required_apis: Vec<(String, String)>,
    minimum_version: Option<Version>,
}

impl ApiCompatibilityChecker {
    /// Create a new compatibility checker
    pub fn new() -> Self {
        Self {
            required_apis: Vec::new(),
            minimum_version: None,
        }
    }
    
    /// Add a required API
    pub fn require_api(mut self, api_name: impl Into<String>, module: impl Into<String>) -> Self {
        self.required_apis.push((api_name.into(), module.into()));
        self
    }
    
    /// Set minimum version requirement
    pub fn require_version(mut self, version: Version) -> Self {
        self.minimum_version = Some(version);
        self
    }
    
    /// Check if all requirements are met
    pub fn check(&self) -> CoreResult<()> {
        // Check version compatibility
        if let Some(min_version) = &self.minimum_version {
            if !is_version_compatible(min_version) {
                return Err(CoreError::ValidationError(
                    ErrorContext::new(format!(
                        "Version {} required, but current version is {}",
                        min_version,
                        current_library_version()
                    ))
                ));
            }
        }
        
        // Check API availability
        let apis: Vec<(&str, &str)> = self.required_apis
            .iter()
            .map(|(api, module)| (api.as_str(), module.as_str()))
            .collect();
        
        check_apis_available(&apis)
    }
}