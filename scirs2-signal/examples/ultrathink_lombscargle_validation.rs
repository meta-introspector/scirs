//! Ultrathink Enhanced Lomb-Scargle Validation Example
//!
//! This example demonstrates the comprehensive ultrathink validation suite for
//! Lomb-Scargle periodogram, including SciPy comparison, SIMD validation,
//! memory profiling, and statistical validation.

use scirs2_signal::{
    run_ultrathink_lombscargle_validation, generate_ultrathink_lombscargle_report,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Ultrathink Enhanced Lomb-Scargle Validation Suite");
    println!("==================================================");
    
    println!("\n🔬 Initializing comprehensive validation...");
    println!("This validation includes:");
    println!("  📊 Comprehensive accuracy testing");
    println!("  🐍 SciPy reference comparison");
    println!("  ⚡ Complete SIMD validation");
    println!("  💾 Memory profiling");
    println!("  📈 Statistical validation");
    println!("  ⏱️ Performance regression detection");
    
    // Run the comprehensive ultrathink validation
    println!("\n🚀 Running ultrathink validation suite...");
    let validation_result = run_ultrathink_lombscargle_validation()?;
    
    // Generate and display the comprehensive report
    let report = generate_ultrathink_lombscargle_report(&validation_result);
    println!("\n{}", report);
    
    // Detailed analysis with colored output
    println!("\n🎯 **EXECUTIVE SUMMARY**");
    println!("========================");
    
    let quality_icon = if validation_result.quality_score >= 95.0 {
        "🌟"
    } else if validation_result.quality_score >= 85.0 {
        "⭐"
    } else if validation_result.quality_score >= 75.0 {
        "⚠️"
    } else {
        "❌"
    };
    
    println!("{} Overall Quality Score: {:.1}/100", quality_icon, validation_result.quality_score);
    
    // Accuracy Analysis
    println!("\n📊 **ACCURACY ANALYSIS**");
    println!("-------------------------");
    
    let freq_accuracy = validation_result.accuracy_validation.frequency_accuracy.single_tone_accuracy;
    let freq_icon = if freq_accuracy < 0.01 { "✅" } else if freq_accuracy < 0.05 { "⚠️" } else { "❌" };
    println!("{} Frequency Estimation: {:.4} relative error", freq_icon, freq_accuracy);
    
    let power_accuracy = validation_result.accuracy_validation.power_accuracy.amplitude_linearity;
    let power_icon = if power_accuracy > 0.95 { "✅" } else if power_accuracy > 0.9 { "⚠️" } else { "❌" };
    println!("{} Power Estimation: {:.3} linearity", power_icon, power_accuracy);
    
    let phase_coherence = validation_result.accuracy_validation.phase_coherence.phase_preservation;
    let phase_icon = if phase_coherence > 0.95 { "✅" } else if phase_coherence > 0.9 { "⚠️" } else { "❌" };
    println!("{} Phase Coherence: {:.3} preservation", phase_icon, phase_coherence);
    
    // SciPy Comparison Analysis
    println!("\n🐍 **SCIPY COMPATIBILITY**");
    println!("---------------------------");
    
    let scipy_corr = validation_result.scipy_comparison.correlation;
    let scipy_icon = if scipy_corr > 0.99 { "✅" } else if scipy_corr > 0.95 { "⚠️" } else { "❌" };
    println!("{} Correlation: {:.4}", scipy_icon, scipy_corr);
    
    let max_error = validation_result.scipy_comparison.max_relative_error;
    let error_icon = if max_error < 0.001 { "✅" } else if max_error < 0.01 { "⚠️" } else { "❌" };
    println!("{} Max Relative Error: {:.2e}", error_icon, max_error);
    
    let mean_error = validation_result.scipy_comparison.mean_relative_error;
    let mean_icon = if mean_error < 0.0001 { "✅" } else if mean_error < 0.001 { "⚠️" } else { "❌" };
    println!("{} Mean Relative Error: {:.2e}", mean_icon, mean_error);
    
    // SIMD Performance Analysis
    println!("\n⚡ **SIMD OPTIMIZATION**");
    println!("-----------------------");
    
    let simd_speedup = validation_result.simd_validation.performance_improvement;
    let speedup_icon = if simd_speedup > 2.0 { "🚀" } else if simd_speedup > 1.5 { "⚡" } else if simd_speedup > 1.1 { "⚠️" } else { "❌" };
    println!("{} Performance Improvement: {:.1}x", speedup_icon, simd_speedup);
    
    let simd_accuracy = validation_result.simd_validation.accuracy_comparison.correlation_coefficient;
    let simd_acc_icon = if simd_accuracy > 0.9999 { "✅" } else if simd_accuracy > 0.999 { "⚠️" } else { "❌" };
    println!("{} SIMD Accuracy: {:.6} correlation", simd_acc_icon, simd_accuracy);
    
    let platform_util = validation_result.simd_validation.platform_utilization.vector_width_utilization;
    let util_icon = if platform_util > 0.8 { "✅" } else if platform_util > 0.6 { "⚠️" } else { "❌" };
    println!("{} Platform Utilization: {:.1}%", util_icon, platform_util * 100.0);
    
    // Memory Analysis
    println!("\n💾 **MEMORY EFFICIENCY**");
    println!("------------------------");
    
    let peak_memory = validation_result.memory_profiling.peak_memory_mb;
    let memory_icon = if peak_memory < 20.0 { "✅" } else if peak_memory < 50.0 { "⚠️" } else { "❌" };
    println!("{} Peak Memory: {:.1} MB", memory_icon, peak_memory);
    
    let cache_hit = validation_result.memory_profiling.efficiency_metrics.cache_hit_ratio;
    let cache_icon = if cache_hit > 0.9 { "✅" } else if cache_hit > 0.8 { "⚠️" } else { "❌" };
    println!("{} Cache Hit Ratio: {:.1}%", cache_icon, cache_hit * 100.0);
    
    let memory_per_sample = validation_result.memory_profiling.efficiency_metrics.memory_per_sample;
    let mps_icon = if memory_per_sample < 0.1 { "✅" } else if memory_per_sample < 0.5 { "⚠️" } else { "❌" };
    println!("{} Memory per Sample: {:.3} KB", mps_icon, memory_per_sample * 1024.0);
    
    // Statistical Validation
    println!("\n📈 **STATISTICAL VALIDATION**");
    println!("-----------------------------");
    
    let fap_accuracy = validation_result.statistical_validation.false_alarm_validation.fap_accuracy;
    let fap_icon = if fap_accuracy > 0.95 { "✅" } else if fap_accuracy > 0.9 { "⚠️" } else { "❌" };
    println!("{} False Alarm Probability: {:.1}% accuracy", fap_icon, fap_accuracy * 100.0);
    
    let psd_comparison = validation_result.statistical_validation.psd_theoretical_comparison.white_noise_comparison;
    let psd_icon = if psd_comparison > 0.9 { "✅" } else if psd_comparison > 0.8 { "⚠️" } else { "❌" };
    println!("{} PSD Theoretical Match: {:.3}", psd_icon, psd_comparison);
    
    // Performance Regression
    println!("\n⏱️ **PERFORMANCE REGRESSION**");
    println!("-----------------------------");
    
    if validation_result.performance_regression.regression_detected {
        println!("❌ Performance regression detected!");
        println!("   📉 Time trend: {:.2}%", 
                 validation_result.performance_regression.trend_analysis.time_trend_slope * 100.0);
        println!("   📉 Memory trend: {:.2}%", 
                 validation_result.performance_regression.trend_analysis.memory_trend_slope * 100.0);
    } else {
        println!("✅ No performance regression detected");
        println!("   📈 Time improvement: {:.1}%", 
                 validation_result.performance_regression.trend_analysis.time_trend_slope * 100.0);
        println!("   📈 Memory improvement: {:.1}%", 
                 validation_result.performance_regression.trend_analysis.memory_trend_slope * 100.0);
    }
    
    // Critical Issues Summary
    if !validation_result.critical_issues.is_empty() {
        println!("\n🚨 **CRITICAL ISSUES REQUIRING ATTENTION**");
        println!("==========================================");
        for (i, issue) in validation_result.critical_issues.iter().enumerate() {
            println!("{}. ❌ {}", i + 1, issue);
        }
    } else {
        println!("\n✅ **NO CRITICAL ISSUES DETECTED**");
        println!("==================================");
        println!("Implementation passes all critical validation checks!");
    }
    
    // Recommendations Summary
    if !validation_result.recommendations.is_empty() {
        println!("\n💡 **OPTIMIZATION OPPORTUNITIES**");
        println!("=================================");
        for (i, recommendation) in validation_result.recommendations.iter().enumerate() {
            println!("{}. 🔧 {}", i + 1, recommendation);
        }
    }
    
    // Final Assessment
    println!("\n🎯 **FINAL ASSESSMENT**");
    println!("=======================");
    
    match validation_result.quality_score as u32 {
        95..=100 => {
            println!("🌟 **EXCEPTIONAL**: This implementation exceeds industry standards!");
            println!("   Ready for production use in demanding applications.");
        },
        85..=94 => {
            println!("⭐ **EXCELLENT**: High-quality implementation with minor optimization opportunities.");
            println!("   Suitable for most production applications.");
        },
        75..=84 => {
            println!("⚠️ **GOOD**: Functional implementation with room for improvement.");
            println!("   Consider addressing recommendations before production use.");
        },
        0..=74 => {
            println!("❌ **NEEDS WORK**: Significant issues require attention.");
            println!("   Please address critical issues before production deployment.");
        },
        _ => unreachable!(),
    }
    
    println!("\n🏁 Ultrathink validation complete!");
    println!("   📋 Full report generated above");
    println!("   📊 Quality score: {:.1}/100", validation_result.quality_score);
    println!("   🔍 {} tests completed", 
             5 + validation_result.statistical_validation.false_alarm_validation.confidence_level_validation.len());
    
    Ok(())
}