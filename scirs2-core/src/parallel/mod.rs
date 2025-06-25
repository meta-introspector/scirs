//! Advanced parallel processing and scheduling
//!
//! This module provides comprehensive parallel processing capabilities including:
//! - Work-stealing scheduler for efficient thread utilization
//! - Custom partitioning strategies for different data distributions
//! - Nested parallelism with controlled resource usage
//! - Load balancing and adaptive scheduling

mod scheduler;
mod partitioning;
mod nested;

// Re-export scheduler functionality
pub use scheduler::{
    create_work_stealing_scheduler, create_work_stealing_scheduler_with_workers, get_worker_id,
    CloneableTask, ParallelTask, SchedulerConfig, SchedulerConfigBuilder, SchedulerStats,
    SchedulingPolicy, TaskHandle, TaskPriority, TaskStatus, WorkStealingArray,
    WorkStealingScheduler,
};

// Re-export partitioning functionality
pub use partitioning::{
    DataDistribution, PartitionStrategy, PartitionerConfig, DataPartitioner,
    LoadBalancer,
};

// Re-export nested parallelism functionality
pub use nested::{
    ResourceLimits, NestedContext, ResourceManager, ResourceUsageStats,
    NestedScope, NestedPolicy, NestedConfig,
    nested_scope, nested_scope_with_limits, with_nested_policy,
    current_nesting_level, is_nested_parallelism_allowed,
    adaptive_par_for_each, adaptive_par_map,
};

// Note: parallel_map is now provided by parallel_ops module for simpler usage
