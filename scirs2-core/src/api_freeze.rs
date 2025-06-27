//! API freeze for scirs2-core 1.0
//!
//! This module contains the frozen API surface for scirs2-core 1.0.
//! These APIs are guaranteed to remain stable throughout the 1.x version series.

use crate::api_versioning::{global_registry_mut, Version};

/// Initialize the API registry with all frozen APIs for 1.0
pub fn initialize_api_freeze() {
    let mut registry = global_registry_mut();
    let v1_0_0 = Version::new(1, 0, 0);
    
    // Core error types and traits
    registry
        .register_api("CoreError", "error", v1_0_0)
        .register_api("CoreResult", "error", v1_0_0)
        .register_api("ErrorContext", "error", v1_0_0)
        .register_api("ErrorKind", "error", v1_0_0)
        .register_api("ErrorContextWithDetails", "error", v1_0_0)
        .register_api("PreciseError", "error", v1_0_0);
    
    // Validation functions
    registry
        .register_api("check_finite", "validation", v1_0_0)
        .register_api("check_array_finite", "validation", v1_0_0)
        .register_api("check_positive", "validation", v1_0_0)
        .register_api("check_shape", "validation", v1_0_0)
        .register_api("check_in_bounds", "validation", v1_0_0);
    
    // Numeric operations
    registry
        .register_api("NumericOps", "numeric", v1_0_0)
        .register_api("Complex", "numeric", v1_0_0)
        .register_api("Precision", "numeric", v1_0_0)
        .register_api("StableAlgorithms", "numeric", v1_0_0);
    
    // Array protocol
    registry
        .register_api("ArrayProtocol", "array_protocol", v1_0_0)
        .register_api("ArrayLike", "array_protocol", v1_0_0)
        .register_api("IntoArray", "array_protocol", v1_0_0)
        .register_api("ArrayView", "array_protocol", v1_0_0)
        .register_api("ArrayViewMut", "array_protocol", v1_0_0);
    
    // Memory efficient operations (conditional)
    #[cfg(feature = "memory_efficient")]
    {
        registry
            .register_api("MemoryMappedArray", "memory_efficient", v1_0_0)
            .register_api("ChunkedArray", "memory_efficient", v1_0_0)
            .register_api("LazyArray", "memory_efficient", v1_0_0)
            .register_api("OutOfCoreArray", "memory_efficient", v1_0_0)
            .register_api("create_mmap", "memory_efficient", v1_0_0)
            .register_api("open_mmap", "memory_efficient", v1_0_0)
            .register_api("chunk_wise_op", "memory_efficient", v1_0_0)
            .register_api("chunk_wise_reduce", "memory_efficient", v1_0_0);
    }
    
    // SIMD operations (conditional)
    #[cfg(feature = "simd")]
    {
        registry
            .register_api("SimdOps", "simd_ops", v1_0_0)
            .register_api("SimdUnifiedOps", "simd_ops", v1_0_0)
            .register_api("PlatformCapabilities", "simd_ops", v1_0_0);
    }
    
    // Parallel operations (conditional)
    #[cfg(feature = "parallel")]
    {
        registry
            .register_api("par_chunks", "parallel_ops", v1_0_0)
            .register_api("par_chunks_mut", "parallel_ops", v1_0_0)
            .register_api("par_range", "parallel_ops", v1_0_0)
            .register_api("par_join", "parallel_ops", v1_0_0)
            .register_api("par_scope", "parallel_ops", v1_0_0)
            .register_api("num_threads", "parallel_ops", v1_0_0)
            .register_api("is_parallel_enabled", "parallel_ops", v1_0_0);
    }
    
    // GPU operations (conditional)
    #[cfg(feature = "gpu")]
    {
        registry
            .register_api("GpuContext", "gpu", v1_0_0)
            .register_api("GpuBuffer", "gpu", v1_0_0)
            .register_api("GpuBackend", "gpu", v1_0_0)
            .register_api("GpuCompute", "gpu", v1_0_0);
    }
    
    // Caching (conditional)
    #[cfg(feature = "cache")]
    {
        registry
            .register_api("Cache", "cache", v1_0_0)
            .register_api("CacheConfig", "cache", v1_0_0)
            .register_api("LruCache", "cache", v1_0_0)
            .register_api("Cacheable", "cache", v1_0_0);
    }
    
    // Types and conversions (conditional)
    #[cfg(feature = "types")]
    {
        registry
            .register_api("ComplexExt", "types", v1_0_0)
            .register_api("ComplexOps", "types", v1_0_0)
            .register_api("ComplexConversionError", "types", v1_0_0)
            .register_api("BatchConverter", "batch_conversions", v1_0_0);
    }
    
    // Array types (conditional)
    #[cfg(feature = "array")]
    {
        registry
            .register_api("MaskedArray", "array", v1_0_0)
            .register_api("RecordArray", "array", v1_0_0)
            .register_api("masked_where", "array", v1_0_0)
            .register_api("masked_invalid", "array", v1_0_0)
            .register_api("record_array_from_arrays", "array", v1_0_0);
    }
    
    // Configuration system
    registry
        .register_api("Config", "config", v1_0_0)
        .register_api("ConfigValue", "config", v1_0_0)
        .register_api("get_config", "config", v1_0_0)
        .register_api("set_config_value", "config", v1_0_0);
    
    // Constants
    registry
        .register_api("math::PI", "constants", v1_0_0)
        .register_api("math::E", "constants", v1_0_0)
        .register_api("physical::C", "constants", v1_0_0)
        .register_api("physical::G", "constants", v1_0_0);
    
    // IO utilities
    registry
        .register_api("read_file", "io", v1_0_0)
        .register_api("write_file", "io", v1_0_0);
    
    // Resource discovery
    registry
        .register_api("SystemResources", "resource", v1_0_0)
        .register_api("get_system_resources", "resource", v1_0_0)
        .register_api("get_available_memory", "resource", v1_0_0)
        .register_api("is_gpu_available", "resource", v1_0_0);
    
    // Units system
    registry
        .register_api("UnitValue", "units", v1_0_0)
        .register_api("UnitRegistry", "units", v1_0_0)
        .register_api("unit_value", "units", v1_0_0)
        .register_api("convert", "units", v1_0_0);
    
    // Versioning
    registry
        .register_api("Version", "versioning", v1_0_0)
        .register_api("ApiVersion", "versioning", v1_0_0)
        .register_api("CompatibilityLevel", "versioning", v1_0_0);
    
    // Metrics
    registry
        .register_api("Counter", "metrics", v1_0_0)
        .register_api("Gauge", "metrics", v1_0_0)
        .register_api("Histogram", "metrics", v1_0_0)
        .register_api("Timer", "metrics", v1_0_0);
}

/// Check if an API is part of the frozen 1.0 API surface
pub fn is_api_frozen(api_name: &str, module: &str) -> bool {
    let registry = global_registry_mut();
    let v1_0_0 = Version::new(1, 0, 0);
    
    registry.apis_in_version(&v1_0_0)
        .iter()
        .any(|entry| entry.name == api_name && entry.module == module)
}

/// Generate a report of all frozen APIs
pub fn generate_frozen_api_report() -> String {
    let registry = global_registry_mut();
    let v1_0_0 = Version::new(1, 0, 0);
    
    let mut report = String::from("# Frozen APIs for scirs2-core 1.0\n\n");
    
    let apis = registry.apis_in_version(&v1_0_0);
    let mut apis_by_module: std::collections::HashMap<&str, Vec<&str>> = std::collections::HashMap::new();
    
    for api in apis {
        apis_by_module.entry(&api.module).or_default().push(&api.name);
    }
    
    let mut modules: Vec<_> = apis_by_module.keys().copied().collect();
    modules.sort();
    
    for module in modules {
        report.push_str(&format!("## Module: {}\n", module));
        if let Some(apis) = apis_by_module.get(module) {
            let mut sorted_apis = apis.clone();
            sorted_apis.sort();
            for api in sorted_apis {
                report.push_str(&format!("- {}\n", api));
            }
        }
        report.push('\n');
    }
    
    report.push_str(&format!("\nTotal frozen APIs: {}\n", apis.len()));
    
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_api_freeze_initialization() {
        initialize_api_freeze();
        
        // Test that core APIs are registered
        assert!(is_api_frozen("CoreError", "error"));
        assert!(is_api_frozen("check_finite", "validation"));
        assert!(is_api_frozen("SystemResources", "resource"));
        
        // Test that the report generates successfully
        let report = generate_frozen_api_report();
        assert!(report.contains("Frozen APIs for scirs2-core 1.0"));
        assert!(report.contains("Module: error"));
        assert!(report.contains("Module: validation"));
    }
}