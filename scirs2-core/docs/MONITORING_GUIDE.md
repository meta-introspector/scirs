# SciRS2 Monitoring and Observability Guide

This guide provides comprehensive monitoring, observability, and alerting strategies for SciRS2 in production environments.

## Table of Contents

1. [Monitoring Architecture](#monitoring-architecture)
2. [Key Metrics](#key-metrics)
3. [Alerting Rules](#alerting-rules)
4. [Dashboard Setup](#dashboard-setup)
5. [Log Management](#log-management)
6. [Performance Monitoring](#performance-monitoring)
7. [Security Monitoring](#security-monitoring)
8. [Troubleshooting](#troubleshooting)

## Monitoring Architecture

### Components Overview

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   SciRS2 App    │───▶│  Prometheus  │───▶│     Grafana     │
│                 │    │   (Metrics)  │    │  (Dashboards)   │
└─────────────────┘    └──────────────┘    └─────────────────┘
         │                       │
         │                       ▼
         │              ┌──────────────┐    ┌─────────────────┐
         │              │ AlertManager │───▶│   PagerDuty     │
         │              │              │    │   Slack/Email   │
         │              └──────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│      Logs       │───▶│    Loki      │───▶│     Grafana     │
│   (JSON/Text)   │    │ (Log Aggr.)  │    │  (Log Viewer)   │
└─────────────────┘    └──────────────┘    └─────────────────┘
```

### Integration Code

```rust
use scirs2_core::observability::{
    Counter, Gauge, Histogram, Timer,
    AuditLogger, TracingContext
};
use prometheus::{register_counter, register_histogram, register_gauge};

// Initialize monitoring
fn setup_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    // Setup Prometheus metrics
    prometheus::default_registry().register(Box::new(OPERATIONS_TOTAL.clone()))?;
    prometheus::default_registry().register(Box::new(MEMORY_USAGE_BYTES.clone()))?;
    prometheus::default_registry().register(Box::new(COMPUTATION_DURATION.clone()))?;
    
    // Setup audit logging
    AuditLogger::initialize(AuditLogger::new()
        .with_file_output("/var/log/scirs2/audit.log")
        .with_structured_format()
        .with_rotation_size(100_000_000))?; // 100MB
    
    // Setup tracing
    TracingContext::initialize(TracingContext::new()
        .with_service_name("scirs2-app")
        .with_version(scirs2_core::version()))?;
    
    Ok(())
}
```

## Key Metrics

### Application Metrics

#### Operation Metrics
```rust
use prometheus::{register_counter_vec, register_histogram_vec};

lazy_static::lazy_static! {
    // Operation counters
    static ref OPERATIONS_TOTAL: CounterVec = register_counter_vec!(
        "scirs2_operations_total",
        "Total number of operations performed",
        &["operation", "status"]
    ).unwrap();
    
    // Operation duration
    static ref OPERATION_DURATION: HistogramVec = register_histogram_vec!(
        "scirs2_operation_duration_seconds",
        "Duration of operations in seconds",
        &["operation"],
        vec![0.001, 0.01, 0.1, 1.0, 10.0, 60.0]
    ).unwrap();
    
    // Error rates
    static ref ERRORS_TOTAL: CounterVec = register_counter_vec!(
        "scirs2_errors_total",
        "Total number of errors",
        &["error_type", "module"]
    ).unwrap();
}

// Usage in application code
fn tracked_operation(op_name: &str) -> Result<f64, Box<dyn std::error::Error>> {
    let timer = OPERATION_DURATION.with_label_values(&[op_name]).start_timer();
    
    match perform_operation() {
        Ok(result) => {
            OPERATIONS_TOTAL.with_label_values(&[op_name, "success"]).inc();
            timer.observe_duration();
            Ok(result)
        }
        Err(e) => {
            OPERATIONS_TOTAL.with_label_values(&[op_name, "error"]).inc();
            ERRORS_TOTAL.with_label_values(&["computation", "core"]).inc();
            timer.observe_duration();
            Err(e)
        }
    }
}
```

#### Memory Metrics
```rust
lazy_static::lazy_static! {
    static ref MEMORY_USAGE_BYTES: Gauge = register_gauge!(
        "scirs2_memory_usage_bytes",
        "Current memory usage in bytes"
    ).unwrap();
    
    static ref MEMORY_ALLOCATIONS_TOTAL: Counter = register_counter!(
        "scirs2_memory_allocations_total",
        "Total number of memory allocations"
    ).unwrap();
    
    static ref CACHE_HIT_RATIO: Gauge = register_gauge!(
        "scirs2_cache_hit_ratio",
        "Cache hit ratio (0-1)"
    ).unwrap();
}

fn update_memory_metrics() {
    if let Ok(usage) = get_memory_usage() {
        MEMORY_USAGE_BYTES.set(usage as f64);
    }
    
    if let Ok(ratio) = get_cache_hit_ratio() {
        CACHE_HIT_RATIO.set(ratio);
    }
}
```

#### Performance Metrics
```rust
lazy_static::lazy_static! {
    static ref SIMD_OPERATIONS_TOTAL: Counter = register_counter!(
        "scirs2_simd_operations_total",
        "Total SIMD operations performed"
    ).unwrap();
    
    static ref PARALLEL_EFFICIENCY: Gauge = register_gauge!(
        "scirs2_parallel_efficiency",
        "Parallel execution efficiency (0-1)"
    ).unwrap();
    
    static ref GPU_UTILIZATION: Gauge = register_gauge!(
        "scirs2_gpu_utilization",
        "GPU utilization percentage (0-100)"
    ).unwrap();
}
```

### System Metrics

#### Resource Utilization
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'scirs2-app'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
```

## Alerting Rules

### Critical Alerts

```yaml
# alerts.yml
groups:
  - name: scirs2_critical
    rules:
      - alert: HighErrorRate
        expr: rate(scirs2_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec over the last 5 minutes"

      - alert: MemoryExhaustion
        expr: scirs2_memory_usage_bytes / 1024 / 1024 / 1024 > 28
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Memory usage critically high"
          description: "Memory usage is {{ $value }}GB, approaching system limits"

      - alert: GPUFailure
        expr: scirs2_gpu_utilization == 0 and rate(scirs2_operations_total{operation=~".*gpu.*"}[5m]) > 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "GPU operations failing"
          description: "GPU operations requested but GPU utilization is 0%"
```

### Warning Alerts

```yaml
  - name: scirs2_warnings
    rules:
      - alert: SlowPerformance
        expr: rate(scirs2_operation_duration_seconds_sum[5m]) / rate(scirs2_operation_duration_seconds_count[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Operations taking longer than expected"
          description: "Average operation time is {{ $value }}s over the last 5 minutes"

      - alert: LowCacheHitRatio
        expr: scirs2_cache_hit_ratio < 0.7
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Cache hit ratio is low"
          description: "Cache hit ratio is {{ $value }}, consider increasing cache size"

      - alert: ParallelEfficiencyLow
        expr: scirs2_parallel_efficiency < 0.5
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Poor parallel execution efficiency"
          description: "Parallel efficiency is {{ $value }}, check for contention or load imbalance"
```

## Dashboard Setup

### Grafana Dashboard JSON

```json
{
  "dashboard": {
    "title": "SciRS2 Production Dashboard",
    "panels": [
      {
        "title": "Operation Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(scirs2_operations_total[5m])",
            "legendFormat": "{{ operation }} - {{ status }}"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "singlestat",
        "targets": [
          {
            "expr": "scirs2_memory_usage_bytes / 1024 / 1024 / 1024",
            "legendFormat": "Memory (GB)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(scirs2_errors_total[5m])",
            "legendFormat": "{{ error_type }}"
          }
        ]
      }
    ]
  }
}
```

### Custom Dashboard Code

```rust
use scirs2_core::observability::dashboard::{Dashboard, Panel, Metric};

fn create_custom_dashboard() -> Dashboard {
    Dashboard::new("SciRS2 Custom Dashboard")
        .add_panel(Panel::new("Operations Overview")
            .with_metric(Metric::counter("scirs2_operations_total"))
            .with_time_range("5m"))
        .add_panel(Panel::new("Performance Metrics")
            .with_metric(Metric::histogram("scirs2_operation_duration_seconds"))
            .with_percentiles(vec![50.0, 95.0, 99.0]))
        .add_panel(Panel::new("Resource Utilization")
            .with_metric(Metric::gauge("scirs2_memory_usage_bytes"))
            .with_metric(Metric::gauge("scirs2_gpu_utilization")))
}
```

## Log Management

### Structured Logging Setup

```rust
use scirs2_core::observability::audit::{AuditEvent, AuditLogger};
use serde_json::json;

fn setup_structured_logging() -> Result<(), Box<dyn std::error::Error>> {
    let logger = AuditLogger::new()
        .with_format("json")
        .with_fields(vec!["timestamp", "level", "service", "operation", "duration", "error"])
        .with_output("/var/log/scirs2/app.log");
    
    AuditLogger::set_global(logger)?;
    Ok(())
}

fn log_operation(operation: &str, duration: std::time::Duration, result: &Result<(), String>) {
    let event = match result {
        Ok(_) => json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "level": "info",
            "service": "scirs2",
            "operation": operation,
            "duration_ms": duration.as_millis(),
            "status": "success"
        }),
        Err(error) => json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "level": "error",
            "service": "scirs2",
            "operation": operation,
            "duration_ms": duration.as_millis(),
            "status": "error",
            "error": error
        }),
    };
    
    AuditLogger::global().log_json(&event);
}
```

### Log Aggregation with Loki

```yaml
# promtail.yml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: scirs2
    static_configs:
      - targets:
          - localhost
        labels:
          job: scirs2
          __path__: /var/log/scirs2/*.log
    pipeline_stages:
      - json:
          expressions:
            level: level
            operation: operation
            service: service
      - labels:
          level:
          operation:
          service:
```

## Performance Monitoring

### Continuous Profiling

```rust
use scirs2_core::profiling::{Profiler, ProfileScope, ProfileReport};

fn setup_continuous_profiling() -> Result<(), Box<dyn std::error::Error>> {
    let profiler = Profiler::new()
        .with_sampling_rate(100) // Hz
        .with_output_format("flamegraph")
        .with_output_path("/var/log/scirs2/profiles");
    
    profiler.start_continuous()?;
    Ok(())
}

fn profile_critical_section<F, R>(name: &str, f: F) -> R 
where 
    F: FnOnce() -> R 
{
    let _scope = ProfileScope::new(name);
    f()
}

// Usage
fn critical_computation() -> Vec<f64> {
    profile_critical_section("matrix_multiply", || {
        // Expensive computation here
        vec![1.0, 2.0, 3.0]
    })
}
```

### Performance Benchmarking

```rust
use scirs2_core::performance_optimization::benchmarking::{
    BenchmarkRunner, BenchmarkConfig, presets
};

fn run_continuous_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    let config = presets::array_operations();
    let runner = BenchmarkRunner::new(config);
    
    let results = runner.benchmark_operation("array_addition", |data, strategy| {
        let start = std::time::Instant::now();
        let result: Vec<f64> = data.iter().map(|x| x + 1.0).collect();
        (start.elapsed(), result)
    });
    
    // Report benchmark results to monitoring system
    for measurement in &results.measurements {
        OPERATION_DURATION
            .with_label_values(&["benchmark", &format!("{:?}", measurement.strategy)])
            .observe(measurement.duration.as_secs_f64());
    }
    
    Ok(())
}
```

## Security Monitoring

### Security Event Tracking

```rust
use scirs2_core::observability::audit::{SecurityEvent, SecurityLevel};

lazy_static::lazy_static! {
    static ref SECURITY_EVENTS_TOTAL: CounterVec = register_counter_vec!(
        "scirs2_security_events_total",
        "Total number of security events",
        &["event_type", "severity"]
    ).unwrap();
}

fn track_security_event(event_type: &str, level: SecurityLevel, details: &str) {
    SECURITY_EVENTS_TOTAL
        .with_label_values(&[event_type, &format!("{:?}", level)])
        .inc();
    
    let event = SecurityEvent {
        timestamp: std::time::SystemTime::now(),
        event_type: event_type.to_string(),
        severity: level,
        source_ip: get_client_ip(),
        user_id: get_current_user(),
        details: details.to_string(),
    };
    
    AuditLogger::global().log_security_event(&event);
}

// Usage
fn validate_input(data: &[f64]) -> Result<(), String> {
    if data.len() > 1_000_000 {
        track_security_event(
            "input_validation_failure",
            SecurityLevel::Medium,
            &format!("Input size {} exceeds limit", data.len())
        );
        return Err("Input too large".to_string());
    }
    
    for (i, &value) in data.iter().enumerate() {
        if !value.is_finite() {
            track_security_event(
                "invalid_input_detected",
                SecurityLevel::High,
                &format!("Non-finite value at index {}: {}", i, value)
            );
            return Err("Invalid input detected".to_string());
        }
    }
    
    Ok(())
}
```

### Anomaly Detection

```rust
use scirs2_core::observability::anomaly::{AnomalyDetector, ThresholdConfig};

fn setup_anomaly_detection() -> Result<(), Box<dyn std::error::Error>> {
    let detector = AnomalyDetector::new()
        .with_threshold_config(ThresholdConfig {
            memory_usage_gb: 25.0,
            error_rate_per_sec: 0.1,
            response_time_seconds: 5.0,
            cache_hit_ratio: 0.5,
        })
        .with_detection_window(std::time::Duration::from_minutes(5))
        .with_alert_callback(|anomaly| {
            track_security_event(
                "anomaly_detected",
                SecurityLevel::High,
                &format!("Anomaly: {}", anomaly.description)
            );
        });
    
    detector.start_monitoring()?;
    Ok(())
}
```

## Troubleshooting

### Diagnostic Commands

```bash
# Check application health
curl http://localhost:8080/health

# Get metrics snapshot
curl http://localhost:9090/metrics

# Check memory usage
curl http://localhost:8080/debug/memory

# Get performance profile
curl http://localhost:8080/debug/pprof/profile?seconds=30

# Check GPU status
curl http://localhost:8080/debug/gpu
```

### Common Issues and Solutions

#### High Memory Usage
```rust
fn diagnose_memory_usage() -> String {
    let report = scirs2_core::memory::generate_memory_report();
    format!("Memory diagnosis:\n{}", report)
}

fn remediate_memory_pressure() -> Result<(), Box<dyn std::error::Error>> {
    // Force garbage collection
    scirs2_core::cache::clear_expired_entries();
    
    // Reduce chunk sizes
    scirs2_core::memory_efficient::set_global_chunk_size(1024 * 1024); // 1MB
    
    // Enable aggressive cleanup
    scirs2_core::memory_efficient::enable_aggressive_cleanup();
    
    Ok(())
}
```

#### Performance Degradation
```rust
fn diagnose_performance() -> String {
    let profiler = scirs2_core::profiling::Profiler::global();
    let report = profiler.generate_report().unwrap();
    
    format!("Performance diagnosis:\n{}", report.summary())
}

fn optimize_performance() -> Result<(), Box<dyn std::error::Error>> {
    // Check SIMD availability
    let caps = scirs2_core::simd_ops::PlatformCapabilities::detect();
    if !caps.has_advanced_simd() {
        println!("Warning: Limited SIMD support");
    }
    
    // Optimize thread count
    let optimal_threads = std::cmp::min(num_cpus::get(), 8);
    scirs2_core::parallel_ops::set_num_threads(optimal_threads);
    
    // Enable performance optimizations
    scirs2_core::performance_optimization::enable_all_optimizations();
    
    Ok(())
}
```

### Monitoring Health Checks

```rust
use scirs2_core::observability::health::{HealthCheck, HealthStatus};

fn comprehensive_health_check() -> HealthStatus {
    let mut checks = Vec::new();
    
    // Memory check
    checks.push(HealthCheck {
        name: "memory".to_string(),
        status: if get_memory_usage_ratio() < 0.8 { "healthy" } else { "unhealthy" }.to_string(),
        details: Some(format!("Memory usage: {:.1}%", get_memory_usage_ratio() * 100.0)),
    });
    
    // Performance check
    let avg_response_time = get_average_response_time();
    checks.push(HealthCheck {
        name: "performance".to_string(),
        status: if avg_response_time < 1.0 { "healthy" } else { "degraded" }.to_string(),
        details: Some(format!("Average response time: {:.3}s", avg_response_time)),
    });
    
    // Cache check
    let cache_ratio = get_cache_hit_ratio();
    checks.push(HealthCheck {
        name: "cache".to_string(),
        status: if cache_ratio > 0.7 { "healthy" } else { "degraded" }.to_string(),
        details: Some(format!("Cache hit ratio: {:.1}%", cache_ratio * 100.0)),
    });
    
    HealthStatus { checks }
}

// Helper functions
fn get_memory_usage_ratio() -> f64 { 
    // Implementation to get memory usage as ratio
    0.65 
}

fn get_average_response_time() -> f64 { 
    // Implementation to get average response time
    0.25 
}

fn get_cache_hit_ratio() -> f64 { 
    // Implementation to get cache hit ratio
    0.85 
}
```

This monitoring guide provides comprehensive observability for SciRS2 applications in production, covering metrics collection, alerting, logging, performance monitoring, and security event tracking.