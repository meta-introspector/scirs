//! API versioning system for backward compatibility
//!
//! This module provides version management for scirs2-core APIs,
//! ensuring smooth transitions between versions and maintaining
//! backward compatibility.

use std::fmt;
use std::sync::{Mutex, OnceLock};

/// Represents a semantic version number
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version {
    /// Major version (breaking changes)
    pub major: u32,
    /// Minor version (new features, backward compatible)
    pub minor: u32,
    /// Patch version (bug fixes)
    pub patch: u32,
}

impl Version {
    /// Create a new version
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse a version string like "1.2.3" or "1.2.3-beta.1"
    pub fn parse(version_str: &str) -> Result<Self, String> {
        // Split on '-' to handle version suffixes like "-beta.1"
        let base_version = version_str.split('-').next().unwrap_or(version_str);

        // Split the base version on '.'
        let parts: Vec<&str> = base_version.split('.').collect();

        if parts.len() < 3 {
            return Err(format!("Invalid version format: {}", version_str));
        }

        let major = parts[0]
            .parse::<u32>()
            .map_err(|_| format!("Invalid major version: {}", parts[0]))?;
        let minor = parts[1]
            .parse::<u32>()
            .map_err(|_| format!("Invalid minor version: {}", parts[1]))?;
        let patch = parts[2]
            .parse::<u32>()
            .map_err(|_| format!("Invalid patch version: {}", parts[2]))?;

        Ok(Self::new(major, minor, patch))
    }

    /// Current version of scirs2-core (Beta 1)
    pub const CURRENT: Self = Self::new(0, 1, 0);

    /// Current beta release identifier
    pub const CURRENT_BETA: &'static str = "0.1.0-beta.1";

    /// Check if this version is compatible with another
    pub fn is_compatible_with(&self, other: &Version) -> bool {
        // Same major version and greater or equal minor/patch
        self.major == other.major
            && (self.minor > other.minor
                || (self.minor == other.minor && self.patch >= other.patch))
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Trait for versioned APIs
pub trait Versioned {
    /// Get the version this API was introduced in
    fn since_version() -> Version;

    /// Get the version this API was deprecated in (if any)
    fn deprecated_version() -> Option<Version> {
        None
    }

    /// Check if this API is available in a given version
    fn is_available_in(version: &Version) -> bool {
        version.is_compatible_with(&Self::since_version())
            && Self::deprecated_version()
                .map(|dep| version < &dep)
                .unwrap_or(true)
    }
}

/// Macro to mark APIs with version information
#[macro_export]
macro_rules! since_version {
    ($major:expr, $minor:expr, $patch:expr) => {
        fn since_version() -> $crate::api_versioning::Version {
            $crate::api_versioning::Version::new($major, $minor, $patch)
        }
    };
}

/// Macro to mark deprecated APIs
#[macro_export]
macro_rules! deprecated_in {
    ($major:expr, $minor:expr, $patch:expr) => {
        fn deprecated_version() -> Option<$crate::api_versioning::Version> {
            Some($crate::api_versioning::Version::new($major, $minor, $patch))
        }
    };
}

/// Version registry for tracking API changes
pub struct VersionRegistry {
    entries: Vec<ApiEntry>,
}

#[derive(Debug, Clone)]
pub struct ApiEntry {
    pub name: String,
    pub module: String,
    pub since: Version,
    pub deprecated: Option<Version>,
    pub replacement: Option<String>,
    pub breaking_changes: Vec<BreakingChange>,
    pub example_usage: Option<String>,
    pub migration_example: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BreakingChange {
    pub version: Version,
    pub description: String,
    pub impact: BreakingChangeImpact,
    pub mitigation: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BreakingChangeImpact {
    Low,      // Minor API signature changes
    Medium,   // Functionality changes
    High,     // Major restructuring
    Critical, // Complete removal
}

impl VersionRegistry {
    /// Create a new version registry
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register a new API
    pub fn register_api(
        &mut self,
        name: impl Into<String>,
        module: impl Into<String>,
        since: Version,
    ) -> &mut Self {
        let name_str = name.into();
        let module_str = module.into();

        // Check if the API is already registered
        if !self
            .entries
            .iter()
            .any(|e| e.name == name_str && e.module == module_str)
        {
            self.entries.push(ApiEntry {
                name: name_str,
                module: module_str,
                since,
                deprecated: None,
                replacement: None,
                breaking_changes: Vec::new(),
                example_usage: None,
                migration_example: None,
            });
        }
        self
    }

    /// Register an API with usage example
    pub fn register_api_with_example(
        &mut self,
        name: impl Into<String>,
        module: impl Into<String>,
        since: Version,
        example: impl Into<String>,
    ) -> &mut Self {
        let name_str = name.into();
        let module_str = module.into();

        if !self
            .entries
            .iter()
            .any(|e| e.name == name_str && e.module == module_str)
        {
            self.entries.push(ApiEntry {
                name: name_str,
                module: module_str,
                since,
                deprecated: None,
                replacement: None,
                breaking_changes: Vec::new(),
                example_usage: Some(example.into()),
                migration_example: None,
            });
        }
        self
    }

    /// Mark an API as deprecated
    pub fn deprecate_api(
        &mut self,
        name: &str,
        version: Version,
        replacement: Option<String>,
    ) -> Result<(), String> {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.name == name) {
            entry.deprecated = Some(version);
            entry.replacement = replacement;
            Ok(())
        } else {
            Err(format!("API '{}' not found in registry", name))
        }
    }

    /// Mark an API as deprecated with migration example
    pub fn deprecate_api_with_migration(
        &mut self,
        name: &str,
        version: Version,
        replacement: Option<String>,
        migration_example: impl Into<String>,
    ) -> Result<(), String> {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.name == name) {
            entry.deprecated = Some(version);
            entry.replacement = replacement;
            entry.migration_example = Some(migration_example.into());
            Ok(())
        } else {
            Err(format!("API '{}' not found in registry", name))
        }
    }

    /// Add a breaking change to an API
    pub fn add_breaking_change(
        &mut self,
        api_name: &str,
        change: BreakingChange,
    ) -> Result<(), String> {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.name == api_name) {
            entry.breaking_changes.push(change);
            Ok(())
        } else {
            Err(format!("API '{}' not found in registry", api_name))
        }
    }

    /// Get all APIs available in a specific version
    pub fn apis_in_version(&self, version: &Version) -> Vec<&ApiEntry> {
        self.entries
            .iter()
            .filter(|entry| {
                version.is_compatible_with(&entry.since)
                    && entry.deprecated.map(|dep| version < &dep).unwrap_or(true)
            })
            .collect()
    }

    /// Get all deprecated APIs in a version
    pub fn deprecated_apis(&self, version: &Version) -> Vec<&ApiEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.deprecated.map(|dep| version >= &dep).unwrap_or(false))
            .collect()
    }

    /// Generate migration guide between versions
    pub fn migration_guide(&self, from: &Version, to: &Version) -> String {
        let mut guide = format!("# Migration Guide: {} → {}\n\n", from, to);

        guide.push_str(&format!(
            "This guide helps you upgrade from scirs2-core {} to {}.\n\n",
            from, to
        ));

        // Breaking changes analysis
        let breaking_changes: Vec<_> = self
            .entries
            .iter()
            .filter_map(|e| {
                if !e.breaking_changes.is_empty() {
                    let relevant_changes: Vec<_> = e
                        .breaking_changes
                        .iter()
                        .filter(|bc| bc.version > *from && bc.version <= *to)
                        .collect();
                    if !relevant_changes.is_empty() {
                        Some((e, relevant_changes))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        if !breaking_changes.is_empty() {
            guide.push_str("## ⚠️ Breaking Changes\n\n");

            for (api, changes) in breaking_changes {
                guide.push_str(&format!("### {} ({})\n\n", api.name, api.module));

                for change in changes {
                    let impact_icon = match change.impact {
                        BreakingChangeImpact::Low => "🟡",
                        BreakingChangeImpact::Medium => "🟠",
                        BreakingChangeImpact::High => "🔴",
                        BreakingChangeImpact::Critical => "💥",
                    };

                    guide.push_str(&format!(
                        "{} **{}**: {}\n\n**Mitigation**: {}\n\n",
                        impact_icon, change.version, change.description, change.mitigation
                    ));
                }
            }
        }

        // Find removed APIs
        let removed: Vec<_> = self
            .entries
            .iter()
            .filter(|e| {
                from.is_compatible_with(&e.since)
                    && e.deprecated.map(|d| to >= &d && from < &d).unwrap_or(false)
            })
            .collect();

        if !removed.is_empty() {
            guide.push_str("## 🗑️ Removed APIs\n\n");

            for api in removed {
                guide.push_str(&format!("### {} ({})\n\n", api.name, api.module));

                if let Some(ref replacement) = api.replacement {
                    guide.push_str(&format!("**Replacement**: {}\n\n", replacement));
                }

                if let Some(ref migration_example) = api.migration_example {
                    guide.push_str("**Migration Example**:\n\n");
                    guide.push_str("```rust\n");
                    guide.push_str(migration_example);
                    guide.push_str("\n```\n\n");
                }
            }
        }

        // Find new APIs
        let new_apis: Vec<_> = self
            .entries
            .iter()
            .filter(|e| to.is_compatible_with(&e.since) && !from.is_compatible_with(&e.since))
            .collect();

        if !new_apis.is_empty() {
            guide.push_str("## ✨ New APIs\n\n");

            for api in new_apis {
                guide.push_str(&format!(
                    "### {} ({}) - Since {}\n\n",
                    api.name, api.module, api.since
                ));

                if let Some(ref example) = api.example_usage {
                    guide.push_str("**Usage Example**:\n\n");
                    guide.push_str("```rust\n");
                    guide.push_str(example);
                    guide.push_str("\n```\n\n");
                }
            }
        }

        // Migration checklist
        guide.push_str("## 📋 Migration Checklist\n\n");
        guide.push_str("- [ ] Update Cargo.toml dependencies\n");
        guide.push_str("- [ ] Fix compilation errors\n");
        guide.push_str("- [ ] Update deprecated API usage\n");
        guide.push_str("- [ ] Run test suite\n");
        guide.push_str("- [ ] Update documentation\n");
        guide.push_str("- [ ] Performance testing\n\n");

        guide.push_str("## 📚 Additional Resources\n\n");
        guide.push_str(&format!(
            "- [API Documentation](https://docs.rs/scirs2-core/{})\n",
            to
        ));
        guide.push_str(
            "- [Changelog](https://github.com/cool-japan/scirs/blob/main/CHANGELOG.md)\n",
        );
        guide.push_str("- [Examples](https://github.com/cool-japan/scirs/tree/main/examples)\n");

        guide
    }

    /// Generate a deprecation timeline
    pub fn deprecation_timeline(&self) -> String {
        let mut timeline = String::from("# API Deprecation Timeline\n\n");

        let mut deprecated_apis: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.deprecated.is_some())
            .collect();

        deprecated_apis.sort_by_key(|e| e.deprecated.unwrap());

        let mut current_version: Option<Version> = None;

        for api in deprecated_apis {
            let dep_version = api.deprecated.unwrap();

            if current_version != Some(dep_version) {
                timeline.push_str(&format!("\n## Version {}\n\n", dep_version));
                current_version = Some(dep_version);
            }

            timeline.push_str(&format!("- **{}** ({})", api.name, api.module));

            if let Some(ref replacement) = api.replacement {
                timeline.push_str(&format!(" → {}", replacement));
            }

            timeline.push('\n');
        }

        timeline
    }
}

impl Default for VersionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global version registry
static REGISTRY: OnceLock<Mutex<VersionRegistry>> = OnceLock::new();

/// Get the global registry for modification
pub fn global_registry_mut() -> std::sync::MutexGuard<'static, VersionRegistry> {
    REGISTRY
        .get_or_init(|| Mutex::new(VersionRegistry::new()))
        .lock()
        .unwrap()
}

/// Get the global registry for reading
pub fn global_registry() -> &'static Mutex<VersionRegistry> {
    REGISTRY.get_or_init(|| Mutex::new(VersionRegistry::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parse() {
        assert_eq!(Version::parse("1.2.3").unwrap(), Version::new(1, 2, 3));
        assert_eq!(
            Version::parse("0.1.0-beta.1").unwrap(),
            Version::new(0, 1, 0)
        );
        assert_eq!(
            Version::parse("10.20.30").unwrap(),
            Version::new(10, 20, 30)
        );

        assert!(Version::parse("1.2").is_err());
        assert!(Version::parse("a.b.c").is_err());
        assert!(Version::parse("").is_err());
    }

    #[test]
    fn test_version_compatibility() {
        let v1_0_0 = Version::new(1, 0, 0);
        let v1_1_0 = Version::new(1, 1, 0);
        let v1_0_1 = Version::new(1, 0, 1);
        let v2_0_0 = Version::new(2, 0, 0);

        assert!(v1_1_0.is_compatible_with(&v1_0_0));
        assert!(v1_0_1.is_compatible_with(&v1_0_0));
        assert!(!v2_0_0.is_compatible_with(&v1_0_0));
        assert!(!v1_0_0.is_compatible_with(&v1_1_0));
    }

    #[test]
    fn test_version_registry() {
        let mut registry = VersionRegistry::new();

        registry
            .register_api("Array", "core", Version::new(0, 1, 0))
            .register_api("Matrix", "linalg", Version::new(0, 1, 0))
            .register_api("OldArray", "core", Version::new(0, 1, 0));

        registry
            .deprecate_api("OldArray", Version::new(0, 2, 0), Some("Array".to_string()))
            .unwrap();

        let v0_1_0 = Version::new(0, 1, 0);
        let v0_2_0 = Version::new(0, 2, 0);

        // Check APIs in v0.1.0
        let apis_v1 = registry.apis_in_version(&v0_1_0);
        assert_eq!(apis_v1.len(), 3);

        // Check APIs in v0.2.0
        let apis_v2 = registry.apis_in_version(&v0_2_0);
        assert_eq!(apis_v2.len(), 2); // OldArray is deprecated

        // Check deprecated APIs
        let deprecated = registry.deprecated_apis(&v0_2_0);
        assert_eq!(deprecated.len(), 1);
        assert_eq!(deprecated[0].name, "OldArray");
    }

    #[test]
    fn test_migration_guide() {
        let mut registry = VersionRegistry::new();

        registry
            .register_api("Feature1", "module1", Version::new(0, 1, 0))
            .register_api("Feature2", "module2", Version::new(0, 2, 0))
            .register_api("OldFeature", "module1", Version::new(0, 1, 0));

        registry
            .deprecate_api(
                "OldFeature",
                Version::new(0, 2, 0),
                Some("Feature2".to_string()),
            )
            .unwrap();

        let guide = registry.migration_guide(&Version::new(0, 1, 0), &Version::new(0, 2, 0));

        assert!(guide.contains("Removed APIs"));
        assert!(guide.contains("OldFeature"));
        assert!(guide.contains("Replacement"));
        assert!(guide.contains("Feature2"));
        assert!(guide.contains("New APIs"));
        assert!(guide.contains("Migration Checklist"));
    }
}
