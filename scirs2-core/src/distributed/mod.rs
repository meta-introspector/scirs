//! Production-grade distributed computing infrastructure
//!
//! This module provides comprehensive distributed computing capabilities
//! for SciRS2 Core 1.0, including distributed arrays, cluster management,
//! fault tolerance, and scalable computation orchestration.

pub mod array;
pub mod cluster;
pub mod communication;
pub mod fault_tolerance;
pub mod load_balancing;
pub mod orchestration;
pub mod scheduler;

// Array operations
pub use array::{DistributedArray, DistributedArrayManager};

// Cluster management
pub use cluster::{
    ClusterManager, ClusterConfiguration, ClusterState, ClusterHealth, ComputeCapacity,
    NodeInfo as ClusterNodeInfo, TaskId, DistributedTask, ResourceRequirements,
    TaskPriority as ClusterTaskPriority, TaskType, TaskParameters, RetryPolicy, BackoffStrategy,
    NodeCapabilities, NodeType, NodeStatus, NodeMetadata, ClusterEventLog, initialize_cluster_manager
};

// Communication
pub use communication::{
    DistributedMessage, CommunicationEndpoint, MessageHandler, HeartbeatHandler, CommunicationManager
};

// Fault tolerance
pub use fault_tolerance::{
    FaultToleranceManager, NodeHealth as FaultNodeHealth, FaultDetectionStrategy, RecoveryStrategy,
    NodeInfo as FaultNodeInfo, ClusterHealthSummary, initialize_fault_tolerance
};

// Load balancing
pub use load_balancing::{
    LoadBalancer as DistributedLoadBalancer, LoadBalancingStrategy, 
    NodeLoad as LoadBalancerNodeLoad, TaskAssignment as LoadBalancerTaskAssignment, LoadBalancingStats
};

// Orchestration
pub use orchestration::{
    OrchestrationEngine, Task as OrchestrationTask, Workflow, TaskStatus as OrchestrationTaskStatus,
    TaskPriority as OrchestrationTaskPriority, WorkflowStatus, OrchestratorNode, OrchestrationStats
};

// Scheduler
pub use scheduler::{
    DistributedScheduler, TaskQueue, ExecutionTracker, LoadBalancer as SchedulerLoadBalancer,
    SchedulingPolicies, SchedulingAlgorithm, LoadBalancingStrategy as SchedulerLoadBalancingStrategy,
    TaskAssignment as SchedulerTaskAssignment, CompletedTask, FailedTask, NodeLoad as SchedulerNodeLoad,
    initialize_distributed_scheduler
};

/// Initialize distributed computing infrastructure
pub fn initialize_distributed_computing() -> crate::error::CoreResult<()> {
    cluster::initialize_cluster_manager()?;
    scheduler::initialize_distributed_scheduler()?;
    fault_tolerance::initialize_fault_tolerance()?;
    Ok(())
}

/// Get distributed system status
pub fn get_distributed_status() -> crate::error::CoreResult<DistributedSystemStatus> {
    let cluster_manager = cluster::ClusterManager::global()?;
    let scheduler = scheduler::DistributedScheduler::global()?;
    
    Ok(DistributedSystemStatus {
        cluster_health: cluster_manager.get_health()?,
        active_nodes: cluster_manager.get_active_nodes()?.len(),
        pending_tasks: scheduler.get_pending_task_count()?,
        total_capacity: cluster_manager.get_total_capacity()?,
    })
}

#[derive(Debug, Clone)]
pub struct DistributedSystemStatus {
    pub cluster_health: ClusterHealth,
    pub active_nodes: usize,
    pub pending_tasks: usize,
    pub total_capacity: ComputeCapacity,
}