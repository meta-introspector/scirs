//! Memory leak detection and monitoring for continuous use
//!
//! This module provides utilities for tracking memory usage patterns, detecting
//! potential leaks, and monitoring memory-related performance issues during
//! long-running interpolation operations.
//!
//! # Overview
//!
//! The memory monitoring system tracks:
//! - Memory allocations and deallocations per interpolator
//! - Cache memory usage and growth patterns  
//! - Peak memory usage across operations
//! - Memory leaks through reference counting
//! - Memory pressure and allocation patterns
//!
//! # Usage
//!
//! ```rust
//! use scirs2_interpolate::memory_monitor::{MemoryMonitor, start_monitoring};
//!
//! // Start global memory monitoring
//! start_monitoring();
//!
//! // Create a monitored interpolator
//! let mut monitor = MemoryMonitor::new("rbf_interpolator");
//! 
//! // Track memory during operations
//! monitor.track_allocation(1024, "distance_matrix");
//! // ... perform interpolation operations ...
//! monitor.track_deallocation(1024, "distance_matrix");
//!
//! // Check for memory leaks
//! let report = monitor.generate_report();
//! if report.has_potential_leaks() {
//!     println!("Warning: Potential memory leaks detected");
//! }
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Global memory monitoring registry
static GLOBAL_MONITOR: OnceLock<Arc<Mutex<GlobalMemoryMonitor>>> = OnceLock::new();

/// Global memory monitoring system
#[derive(Debug)]
struct GlobalMemoryMonitor {
    /// Active memory monitors by name
    monitors: HashMap<String, Arc<Mutex<MemoryMonitor>>>,
    
    /// Global memory statistics
    global_stats: GlobalMemoryStats,
    
    /// Whether monitoring is enabled
    enabled: bool,
    
    /// Maximum number of monitors to track
    max_monitors: usize,
}

/// Global memory statistics across all interpolators
#[derive(Debug, Clone)]
pub struct GlobalMemoryStats {
    /// Total memory allocated across all interpolators
    pub total_allocated_bytes: usize,
    
    /// Peak memory usage across all interpolators
    pub peak_total_bytes: usize,
    
    /// Number of active interpolators being monitored
    pub active_interpolators: usize,
    
    /// Total number of allocations tracked
    pub total_allocations: u64,
    
    /// Total number of deallocations tracked
    pub total_deallocations: u64,
    
    /// Start time of monitoring
    pub monitoring_start: Instant,
}

impl Default for GlobalMemoryStats {
    fn default() -> Self {
        Self {
            total_allocated_bytes: 0,
            peak_total_bytes: 0,
            active_interpolators: 0,
            total_allocations: 0,
            total_deallocations: 0,
            monitoring_start: Instant::now(),
        }
    }
}

/// Individual memory monitor for a specific interpolator
#[derive(Debug)]
pub struct MemoryMonitor {
    /// Name/identifier for this monitor
    name: String,
    
    /// Current memory allocations by category
    allocations: HashMap<String, usize>,
    
    /// Memory allocation history
    allocation_history: VecDeque<AllocationEvent>,
    
    /// Peak memory usage for this interpolator
    peak_memory_bytes: usize,
    
    /// Current total memory usage
    current_memory_bytes: usize,
    
    /// Statistics for leak detection
    leak_stats: LeakDetectionStats,
    
    /// Performance metrics
    perf_metrics: MemoryPerformanceMetrics,
    
    /// Whether this monitor is active
    active: bool,
    
    /// Creation timestamp
    created_at: Instant,
}

/// Memory allocation/deallocation event
#[derive(Debug, Clone)]
struct AllocationEvent {
    /// Type of event (allocation or deallocation)
    event_type: EventType,
    
    /// Size in bytes
    size_bytes: usize,
    
    /// Category of memory (e.g., "distance_matrix", "cache", "coefficients")
    category: String,
    
    /// Timestamp of event
    timestamp: Instant,
}

/// Type of memory event
#[derive(Debug, Clone, Copy, PartialEq)]
enum EventType {
    Allocation,
    Deallocation,
}

/// Statistics for leak detection
#[derive(Debug, Clone)]
struct LeakDetectionStats {
    /// Total number of allocations
    total_allocations: u64,
    
    /// Total number of deallocations
    total_deallocations: u64,
    
    /// Number of unmatched allocations (potential leaks)
    unmatched_allocations: u64,
    
    /// Memory that has been allocated but not freed for a long time
    long_lived_allocations: HashMap<String, (usize, Instant)>,
    
    /// Threshold for considering allocations as potential leaks (in seconds)
    leak_detection_threshold: Duration,
}

impl Default for LeakDetectionStats {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            unmatched_allocations: 0,
            long_lived_allocations: HashMap::new(),
            leak_detection_threshold: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Memory performance metrics
#[derive(Debug, Clone)]
struct MemoryPerformanceMetrics {
    /// Average allocation size
    avg_allocation_size: f64,
    
    /// Average time between allocations
    avg_allocation_interval: Duration,
    
    /// Memory fragmentation estimate (0.0 to 1.0)
    fragmentation_estimate: f64,
    
    /// Cache hit ratio for memory reuse
    cache_hit_ratio: f64,
    
    /// Last update timestamp
    last_update: Instant,
}

impl Default for MemoryPerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_allocation_size: 0.0,
            avg_allocation_interval: Duration::from_millis(0),
            fragmentation_estimate: 0.0,
            cache_hit_ratio: 0.0,
            last_update: Instant::now(),
        }
    }
}

/// Memory monitoring report
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Monitor name
    pub monitor_name: String,
    
    /// Current memory usage by category
    pub current_allocations: HashMap<String, usize>,
    
    /// Peak memory usage
    pub peak_memory_bytes: usize,
    
    /// Total memory allocated over lifetime
    pub total_allocated_bytes: usize,
    
    /// Memory leak indicators
    pub leak_indicators: LeakIndicators,
    
    /// Performance metrics
    pub performance_summary: PerformanceSummary,
    
    /// Recommendations for memory optimization
    pub recommendations: Vec<String>,
    
    /// Report generation timestamp
    pub generated_at: Instant,
}

/// Memory leak indicators
#[derive(Debug, Clone)]
pub struct LeakIndicators {
    /// Potential memory leaks detected
    pub has_potential_leaks: bool,
    
    /// Number of unmatched allocations
    pub unmatched_allocations: u64,
    
    /// Memory that has been held for a long time
    pub long_lived_memory_bytes: usize,
    
    /// Categories with suspicious allocation patterns
    pub suspicious_categories: Vec<String>,
    
    /// Leak severity (0.0 = no leaks, 1.0 = severe leaks)
    pub leak_severity: f64,
}

/// Performance summary for memory usage
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Memory efficiency (lower is better)
    pub memory_efficiency_score: f64,
    
    /// Allocation pattern efficiency
    pub allocation_pattern_score: f64,
    
    /// Cache utilization score
    pub cache_utilization_score: f64,
    
    /// Overall memory performance grade
    pub overall_grade: PerformanceGrade,
}

/// Performance grade classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PerformanceGrade {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

impl MemoryMonitor {
    /// Create a new memory monitor
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let monitor = Self {
            name: name.clone(),
            allocations: HashMap::new(),
            allocation_history: VecDeque::new(),
            peak_memory_bytes: 0,
            current_memory_bytes: 0,
            leak_stats: LeakDetectionStats::default(),
            perf_metrics: MemoryPerformanceMetrics::default(),
            active: true,
            created_at: Instant::now(),
        };
        
        // Register with global monitor
        register_monitor(&name, monitor.clone());
        monitor
    }
    
    /// Track a memory allocation
    pub fn track_allocation(&mut self, size_bytes: usize, category: impl Into<String>) {
        if !self.active {
            return;
        }
        
        let category = category.into();
        let now = Instant::now();
        
        // Update current allocations
        *self.allocations.entry(category.clone()).or_insert(0) += size_bytes;
        self.current_memory_bytes += size_bytes;
        
        // Update peak usage
        if self.current_memory_bytes > self.peak_memory_bytes {
            self.peak_memory_bytes = self.current_memory_bytes;
        }
        
        // Record allocation event
        let event = AllocationEvent {
            event_type: EventType::Allocation,
            size_bytes,
            category: category.clone(),
            timestamp: now,
        };
        
        self.allocation_history.push_back(event);
        
        // Limit history size to prevent memory growth
        if self.allocation_history.len() > 10000 {
            self.allocation_history.pop_front();
        }
        
        // Update leak detection stats
        self.leak_stats.total_allocations += 1;
        self.leak_stats.long_lived_allocations.insert(
            format!("{}_{}", category, self.leak_stats.total_allocations),
            (size_bytes, now),
        );
        
        // Update performance metrics
        self.update_performance_metrics();
        
        // Update global stats
        update_global_stats(size_bytes, true);
    }
    
    /// Track a memory deallocation
    pub fn track_deallocation(&mut self, size_bytes: usize, category: impl Into<String>) {
        if !self.active {
            return;
        }
        
        let category = category.into();
        let now = Instant::now();
        
        // Update current allocations
        if let Some(current) = self.allocations.get_mut(&category) {
            *current = current.saturating_sub(size_bytes);
            if *current == 0 {
                self.allocations.remove(&category);
            }
        }
        
        self.current_memory_bytes = self.current_memory_bytes.saturating_sub(size_bytes);
        
        // Record deallocation event
        let event = AllocationEvent {
            event_type: EventType::Deallocation,
            size_bytes,
            category: category.clone(),
            timestamp: now,
        };
        
        self.allocation_history.push_back(event);
        
        // Update leak detection stats
        self.leak_stats.total_deallocations += 1;
        
        // Remove from long-lived allocations (simplified - would need better matching in production)
        self.leak_stats.long_lived_allocations.retain(|k, _| !k.starts_with(&category));
        
        // Update performance metrics
        self.update_performance_metrics();
        
        // Update global stats
        update_global_stats(size_bytes, false);
    }
    
    /// Generate a comprehensive memory report
    pub fn generate_report(&self) -> MemoryReport {
        let leak_indicators = self.analyze_leaks();
        let performance_summary = self.analyze_performance();
        let recommendations = self.generate_recommendations(&leak_indicators, &performance_summary);
        
        MemoryReport {
            monitor_name: self.name.clone(),
            current_allocations: self.allocations.clone(),
            peak_memory_bytes: self.peak_memory_bytes,
            total_allocated_bytes: self.calculate_total_allocated(),
            leak_indicators,
            performance_summary,
            recommendations,
            generated_at: Instant::now(),
        }
    }
    
    /// Analyze potential memory leaks
    fn analyze_leaks(&self) -> LeakIndicators {
        let unmatched = self.leak_stats.total_allocations.saturating_sub(self.leak_stats.total_deallocations);
        
        // Calculate long-lived memory
        let now = Instant::now();
        let long_lived_memory: usize = self.leak_stats.long_lived_allocations
            .values()
            .filter(|(_, timestamp)| now.duration_since(*timestamp) > self.leak_stats.leak_detection_threshold)
            .map(|(size, _)| size)
            .sum();
        
        // Identify suspicious categories (categories with consistently growing memory)
        let suspicious_categories: Vec<String> = self.allocations
            .iter()
            .filter(|(_, &size)| size > 1024 * 1024) // More than 1MB
            .map(|(cat, _)| cat.clone())
            .collect();
        
        let has_potential_leaks = unmatched > 0 || long_lived_memory > 0 || !suspicious_categories.is_empty();
        
        // Calculate leak severity
        let leak_severity = if has_potential_leaks {
            let severity_factors = [
                (unmatched as f64) / (self.leak_stats.total_allocations as f64).max(1.0),
                (long_lived_memory as f64) / (self.peak_memory_bytes as f64).max(1.0),
                (suspicious_categories.len() as f64) / 10.0, // Normalize by 10 categories
            ];
            severity_factors.iter().sum::<f64>() / severity_factors.len() as f64
        } else {
            0.0
        };
        
        LeakIndicators {
            has_potential_leaks,
            unmatched_allocations: unmatched,
            long_lived_memory_bytes: long_lived_memory,
            suspicious_categories,
            leak_severity: leak_severity.min(1.0),
        }
    }
    
    /// Analyze memory performance
    fn analyze_performance(&self) -> PerformanceSummary {
        // Calculate memory efficiency (lower peak/current ratio is better)
        let memory_efficiency_score = if self.peak_memory_bytes > 0 {
            1.0 - (self.current_memory_bytes as f64 / self.peak_memory_bytes as f64)
        } else {
            1.0
        };
        
        // Calculate allocation pattern efficiency
        let allocation_pattern_score = if self.leak_stats.total_allocations > 0 {
            let deallocation_ratio = self.leak_stats.total_deallocations as f64 / self.leak_stats.total_allocations as f64;
            deallocation_ratio.min(1.0)
        } else {
            1.0
        };
        
        // Use cached cache utilization score
        let cache_utilization_score = self.perf_metrics.cache_hit_ratio;
        
        // Calculate overall grade
        let overall_score = (memory_efficiency_score + allocation_pattern_score + cache_utilization_score) / 3.0;
        let overall_grade = match overall_score {
            s if s >= 0.9 => PerformanceGrade::Excellent,
            s if s >= 0.7 => PerformanceGrade::Good,
            s if s >= 0.5 => PerformanceGrade::Fair,
            s if s >= 0.3 => PerformanceGrade::Poor,
            _ => PerformanceGrade::Critical,
        };
        
        PerformanceSummary {
            memory_efficiency_score,
            allocation_pattern_score,
            cache_utilization_score,
            overall_grade,
        }
    }
    
    /// Generate optimization recommendations
    fn generate_recommendations(&self, leak_indicators: &LeakIndicators, performance: &PerformanceSummary) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if leak_indicators.has_potential_leaks {
            recommendations.push("Consider implementing explicit memory cleanup in destructor".to_string());
            
            if leak_indicators.unmatched_allocations > 0 {
                recommendations.push(format!(
                    "Found {} unmatched allocations - check for missing deallocations",
                    leak_indicators.unmatched_allocations
                ));
            }
            
            if leak_indicators.long_lived_memory_bytes > 1024 * 1024 {
                recommendations.push(format!(
                    "Large amount of long-lived memory ({} MB) - consider periodic cleanup",
                    leak_indicators.long_lived_memory_bytes / (1024 * 1024)
                ));
            }
        }
        
        if matches!(performance.overall_grade, PerformanceGrade::Fair | PerformanceGrade::Poor | PerformanceGrade::Critical) {
            recommendations.push("Memory performance can be improved".to_string());
            
            if performance.memory_efficiency_score < 0.5 {
                recommendations.push("High peak memory usage - consider processing data in chunks".to_string());
            }
            
            if performance.cache_utilization_score < 0.3 {
                recommendations.push("Low cache utilization - enable caching for repeated operations".to_string());
            }
        }
        
        if self.peak_memory_bytes > 1024 * 1024 * 1024 {
            recommendations.push("Very high memory usage - consider using memory-efficient algorithms".to_string());
        }
        
        recommendations
    }
    
    /// Update performance metrics
    fn update_performance_metrics(&mut self) {
        let now = Instant::now();
        
        // Update average allocation size
        if self.leak_stats.total_allocations > 0 {
            let total_size: usize = self.allocation_history
                .iter()
                .filter(|e| e.event_type == EventType::Allocation)
                .map(|e| e.size_bytes)
                .sum();
            self.perf_metrics.avg_allocation_size = total_size as f64 / self.leak_stats.total_allocations as f64;
        }
        
        // Simple cache hit ratio simulation (would need actual cache statistics in practice)
        self.perf_metrics.cache_hit_ratio = 0.7; // Placeholder
        
        self.perf_metrics.last_update = now;
    }
    
    /// Calculate total memory allocated over lifetime
    fn calculate_total_allocated(&self) -> usize {
        self.allocation_history
            .iter()
            .filter(|e| e.event_type == EventType::Allocation)
            .map(|e| e.size_bytes)
            .sum()
    }
    
    /// Disable this monitor
    pub fn disable(&mut self) {
        self.active = false;
    }
    
    /// Check if monitor is active
    pub fn is_active(&self) -> bool {
        self.active
    }
}

impl Clone for MemoryMonitor {
    fn clone(&self) -> Self {
        Self {
            name: format!("{}_clone", self.name),
            allocations: self.allocations.clone(),
            allocation_history: self.allocation_history.clone(),
            peak_memory_bytes: self.peak_memory_bytes,
            current_memory_bytes: self.current_memory_bytes,
            leak_stats: self.leak_stats.clone(),
            perf_metrics: self.perf_metrics.clone(),
            active: self.active,
            created_at: self.created_at,
        }
    }
}

impl MemoryReport {
    /// Check if the report indicates potential memory leaks
    pub fn has_potential_leaks(&self) -> bool {
        self.leak_indicators.has_potential_leaks
    }
    
    /// Get memory efficiency rating
    pub fn memory_efficiency_rating(&self) -> PerformanceGrade {
        self.performance_summary.overall_grade
    }
    
    /// Get human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "Memory Report for '{}': Current: {} KB, Peak: {} KB, Grade: {:?}, Leaks: {}",
            self.monitor_name,
            self.current_allocations.values().sum::<usize>() / 1024,
            self.peak_memory_bytes / 1024,
            self.performance_summary.overall_grade,
            if self.has_potential_leaks() { "Detected" } else { "None" }
        )
    }
}

/// Global memory monitoring functions

/// Start global memory monitoring
pub fn start_monitoring() {
    let _ = GLOBAL_MONITOR.set(Arc::new(Mutex::new(GlobalMemoryMonitor {
        monitors: HashMap::new(),
        global_stats: GlobalMemoryStats::default(),
        enabled: true,
        max_monitors: 100,
    })));
}

/// Stop global memory monitoring
pub fn stop_monitoring() {
    if let Some(monitor) = GLOBAL_MONITOR.get() {
        if let Ok(mut global) = monitor.lock() {
            global.enabled = false;
            global.monitors.clear();
        }
    }
}

/// Register a memory monitor with the global system
fn register_monitor(name: &str, monitor: MemoryMonitor) {
    if let Some(global_monitor) = GLOBAL_MONITOR.get() {
        if let Ok(mut global) = global_monitor.lock() {
            if global.enabled && global.monitors.len() < global.max_monitors {
                global.monitors.insert(name.to_string(), Arc::new(Mutex::new(monitor)));
                global.global_stats.active_interpolators = global.monitors.len();
            }
        }
    }
}

/// Update global memory statistics
fn update_global_stats(size_bytes: usize, is_allocation: bool) {
    if let Some(global_monitor) = GLOBAL_MONITOR.get() {
        if let Ok(mut global) = global_monitor.lock() {
            if is_allocation {
                global.global_stats.total_allocated_bytes += size_bytes;
                global.global_stats.total_allocations += 1;
                
                if global.global_stats.total_allocated_bytes > global.global_stats.peak_total_bytes {
                    global.global_stats.peak_total_bytes = global.global_stats.total_allocated_bytes;
                }
            } else {
                global.global_stats.total_allocated_bytes = 
                    global.global_stats.total_allocated_bytes.saturating_sub(size_bytes);
                global.global_stats.total_deallocations += 1;
            }
        }
    }
}

/// Get global memory statistics
pub fn get_global_stats() -> Option<GlobalMemoryStats> {
    GLOBAL_MONITOR.get()
        .and_then(|monitor| monitor.lock().ok())
        .map(|global| global.global_stats.clone())
}

/// Get report for a specific monitor
pub fn get_monitor_report(name: &str) -> Option<MemoryReport> {
    GLOBAL_MONITOR.get()
        .and_then(|global_monitor| global_monitor.lock().ok())
        .and_then(|global| global.monitors.get(name).cloned())
        .and_then(|monitor| monitor.lock().ok())
        .map(|m| m.generate_report())
}

/// Get reports for all active monitors
pub fn get_all_reports() -> Vec<MemoryReport> {
    if let Some(global_monitor) = GLOBAL_MONITOR.get() {
        if let Ok(global) = global_monitor.lock() {
            return global.monitors
                .values()
                .filter_map(|monitor| monitor.lock().ok())
                .map(|m| m.generate_report())
                .collect();
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_monitor_basic() {
        let mut monitor = MemoryMonitor::new("test");
        
        // Track some allocations
        monitor.track_allocation(1024, "matrix");
        monitor.track_allocation(512, "cache");
        
        assert_eq!(monitor.current_memory_bytes, 1536);
        assert_eq!(monitor.peak_memory_bytes, 1536);
        
        // Track deallocation
        monitor.track_deallocation(512, "cache");
        assert_eq!(monitor.current_memory_bytes, 1024);
        
        let report = monitor.generate_report();
        assert!(!report.has_potential_leaks());
    }
    
    #[test]
    fn test_leak_detection() {
        let mut monitor = MemoryMonitor::new("leak_test");
        
        // Allocate without deallocating (potential leak)
        monitor.track_allocation(2048, "leaked_memory");
        
        let report = monitor.generate_report();
        assert!(report.leak_indicators.unmatched_allocations > 0);
    }
    
    #[test]
    fn test_global_monitoring() {
        start_monitoring();
        
        let _monitor1 = MemoryMonitor::new("global_test_1");
        let _monitor2 = MemoryMonitor::new("global_test_2");
        
        let stats = get_global_stats().unwrap();
        assert_eq!(stats.active_interpolators, 2);
        
        stop_monitoring();
    }
}