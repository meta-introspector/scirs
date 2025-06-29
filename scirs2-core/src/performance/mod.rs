//! Performance optimization and monitoring for SciRS2 Core
//!
//! This module provides comprehensive performance optimization capabilities
//! including advanced SIMD operations, cache-aware algorithms, adaptive
//! optimization, and production-ready resource management.

pub mod advanced_optimization;

/// Re-export key performance types and functions
pub use advanced_optimization::{
    cache_aware_matrix_multiply, prefetch, profiling, simd_ops, AdvancedPerformanceOptimizer,
    CacheInfo, OptimizationSettings, PerformanceProfile, PerformanceRecommendation,
    SimdCapabilities, SimdInstructionSet, WorkloadType,
};

/// Initialize the global performance optimizer
pub fn initialize_performance_optimizer() -> crate::error::CoreResult<()> {
    let _optimizer = AdvancedPerformanceOptimizer::global();
    Ok(())
}

/// Get performance recommendations for the current system
pub fn get_system_performance_recommendations() -> Vec<String> {
    let optimizer = AdvancedPerformanceOptimizer::new();
    let profile = optimizer.profile();
    let mut recommendations = Vec::new();

    if profile.simd_capabilities.avx2 {
        recommendations
            .push("AVX2 instruction set detected - enable advanced SIMD optimizations".to_string());
    }

    if profile.cpu_cores > 8 {
        recommendations.push(format!(
            "High core count ({}) detected - consider parallel algorithms",
            profile.cpu_cores
        ));
    }

    if profile.cache_info.l3_cache_size > 16 * 1024 * 1024 {
        recommendations.push(
            "Large L3 cache detected - enable cache-aware blocking for large matrices".to_string(),
        );
    }

    recommendations
}
