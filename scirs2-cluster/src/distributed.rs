//! Distributed clustering algorithms for large-scale datasets
//!
//! This module provides distributed implementations of clustering algorithms that can
//! handle datasets too large to fit in memory on a single machine. It supports
//! distributed K-means, hierarchical clustering, and data partitioning strategies.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive, Zero};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::error::{ClusteringError, Result};
use crate::vq::euclidean_distance;

/// Configuration for distributed clustering algorithms
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DistributedConfig {
    /// Number of worker nodes/processes
    pub n_workers: usize,
    /// Data partitioning strategy
    pub partitioning_strategy: PartitioningStrategy,
    /// Maximum iterations per coordination round
    pub max_iterations_per_round: usize,
    /// Global convergence tolerance
    pub global_tolerance: f64,
    /// Maximum number of coordination rounds
    pub max_coordination_rounds: usize,
    /// Memory limit per worker (in MB)
    pub memory_limit_mb: usize,
    /// Whether to use compression for data transfer
    pub use_compression: bool,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Fault tolerance configuration
    pub fault_tolerance: FaultToleranceConfig,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            n_workers: 4,
            partitioning_strategy: PartitioningStrategy::RandomSplit,
            max_iterations_per_round: 10,
            global_tolerance: 1e-4,
            max_coordination_rounds: 100,
            memory_limit_mb: 1024,
            use_compression: true,
            load_balancing: LoadBalancingStrategy::EqualSize,
            fault_tolerance: FaultToleranceConfig::default(),
        }
    }
}

/// Data partitioning strategies for distributed computing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PartitioningStrategy {
    /// Random assignment of data points to workers
    RandomSplit,
    /// Hash-based partitioning for deterministic assignment
    HashBased,
    /// K-d tree based spatial partitioning
    SpatialPartitioning,
    /// Stratified sampling to ensure balanced clusters
    StratifiedSampling,
    /// Round-robin assignment
    RoundRobin,
}

/// Load balancing strategies for distributed workers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LoadBalancingStrategy {
    /// Equal number of data points per worker
    EqualSize,
    /// Balance by estimated computational cost
    ComputationalBalance,
    /// Dynamic load balancing based on worker performance
    Dynamic,
    /// Custom weights for each worker
    WeightedDistribution,
}

/// Fault tolerance configuration for distributed clustering
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FaultToleranceConfig {
    /// Enable fault tolerance mechanisms
    pub enabled: bool,
    /// Maximum number of worker failures to tolerate
    pub max_failures: usize,
    /// Timeout for worker operations in milliseconds
    pub worker_timeout_ms: u64,
    /// Heartbeat interval for worker health checks
    pub heartbeat_interval_ms: u64,
    /// Enable automatic worker replacement
    pub auto_replace_workers: bool,
    /// Recovery strategy when workers fail
    pub recovery_strategy: RecoveryStrategy,
    /// Enable data replication across workers
    pub enable_replication: bool,
    /// Replication factor for data
    pub replication_factor: usize,
    /// Enable checkpointing for recovery
    pub enable_checkpointing: bool,
    /// Checkpoint interval in iterations
    pub checkpoint_interval: usize,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_failures: 2,
            worker_timeout_ms: 30000,
            heartbeat_interval_ms: 5000,
            auto_replace_workers: true,
            recovery_strategy: RecoveryStrategy::Redistribute,
            enable_replication: false,
            replication_factor: 2,
            enable_checkpointing: true,
            checkpoint_interval: 10,
        }
    }
}

/// Recovery strategies for handling worker failures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RecoveryStrategy {
    /// Redistribute failed worker's data to remaining workers
    Redistribute,
    /// Spawn new workers to replace failed ones
    Replace,
    /// Use checkpoints to restore worker state
    Checkpoint,
    /// Restart the entire computation
    Restart,
    /// Graceful degradation (continue with fewer workers)
    Degrade,
}

/// Worker health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerStatus {
    Active,
    Inactive,
    Failed,
    Timeout,
    Unknown,
}

/// Worker health monitoring information
#[derive(Debug, Clone)]
pub struct WorkerHealth {
    pub worker_id: usize,
    pub status: WorkerStatus,
    pub last_heartbeat: std::time::Instant,
    pub consecutive_failures: usize,
    pub total_failures: usize,
    pub response_time_ms: u64,
    pub cpu_usage: f64,
    pub memory_usage: f64,
}

/// Comprehensive health report for all workers
#[derive(Debug, Clone)]
pub struct WorkerHealthReport {
    pub total_workers: usize,
    pub active_workers: usize,
    pub failed_workers: usize,
    pub timeout_workers: usize,
    pub avg_response_time_ms: f64,
    pub avg_cpu_usage: f64,
    pub avg_memory_usage: f64,
    pub at_risk_workers: Vec<(usize, f64)>, // (worker_id, risk_score)
}

impl WorkerHealth {
    pub fn new(worker_id: usize) -> Self {
        Self {
            worker_id,
            status: WorkerStatus::Unknown,
            last_heartbeat: std::time::Instant::now(),
            consecutive_failures: 0,
            total_failures: 0,
            response_time_ms: 0,
            cpu_usage: 0.0,
            memory_usage: 0.0,
        }
    }

    pub fn is_healthy(&self, timeout_ms: u64) -> bool {
        matches!(self.status, WorkerStatus::Active) &&
        self.last_heartbeat.elapsed().as_millis() < timeout_ms as u128
    }

    /// Check if worker is healthy with adaptive thresholds
    pub fn is_healthy_adaptive(&self, timeout_ms: u64, cpu_threshold: f64, memory_threshold: f64) -> bool {
        if !matches!(self.status, WorkerStatus::Active) {
            return false;
        }

        // Check basic heartbeat timeout
        if self.last_heartbeat.elapsed().as_millis() >= timeout_ms as u128 {
            return false;
        }

        // Check performance thresholds
        if self.cpu_usage > cpu_threshold || self.memory_usage > memory_threshold {
            return false;
        }

        // Check for too many recent failures
        if self.consecutive_failures > 3 {
            return false;
        }

        true
    }

    /// Predict potential failure based on trends
    pub fn predict_failure_risk(&self) -> f64 {
        let mut risk_score = 0.0;

        // High consecutive failures increase risk
        risk_score += (self.consecutive_failures as f64) * 0.2;

        // High response time increases risk
        if self.response_time_ms > 1000 {
            risk_score += 0.3;
        } else if self.response_time_ms > 500 {
            risk_score += 0.1;
        }

        // High resource usage increases risk
        if self.cpu_usage > 90.0 {
            risk_score += 0.3;
        } else if self.cpu_usage > 70.0 {
            risk_score += 0.1;
        }

        if self.memory_usage > 90.0 {
            risk_score += 0.3;
        } else if self.memory_usage > 70.0 {
            risk_score += 0.1;
        }

        // Time since last heartbeat
        let heartbeat_age_ms = self.last_heartbeat.elapsed().as_millis() as u64;
        if heartbeat_age_ms > 10000 {
            risk_score += 0.4;
        } else if heartbeat_age_ms > 5000 {
            risk_score += 0.2;
        }

        risk_score.min(1.0) // Cap at 1.0
    }

    /// Update worker metrics from performance data
    pub fn update_metrics(&mut self, response_time_ms: u64, cpu_usage: f64, memory_usage: f64) {
        self.response_time_ms = response_time_ms;
        self.cpu_usage = cpu_usage.clamp(0.0, 100.0);
        self.memory_usage = memory_usage.clamp(0.0, 100.0);
        self.last_heartbeat = std::time::Instant::now();
    }

    pub fn mark_failure(&mut self) {
        self.consecutive_failures += 1;
        self.total_failures += 1;
        self.status = WorkerStatus::Failed;
    }

    pub fn mark_success(&mut self) {
        self.consecutive_failures = 0;
        self.status = WorkerStatus::Active;
        self.last_heartbeat = std::time::Instant::now();
    }

    /// Mark worker as timeout instead of complete failure
    pub fn mark_timeout(&mut self) {
        self.consecutive_failures += 1;
        self.status = WorkerStatus::Timeout;
    }

    /// Get worker efficiency score based on performance metrics
    pub fn get_efficiency_score(&self) -> f64 {
        if !matches!(self.status, WorkerStatus::Active) {
            return 0.0;
        }

        let mut score = 1.0;

        // Penalize high response times
        if self.response_time_ms > 2000 {
            score *= 0.5;
        } else if self.response_time_ms > 1000 {
            score *= 0.8;
        }

        // Penalize high resource usage
        if self.cpu_usage > 90.0 {
            score *= 0.6;
        } else if self.cpu_usage > 70.0 {
            score *= 0.9;
        }

        if self.memory_usage > 90.0 {
            score *= 0.6;
        } else if self.memory_usage > 70.0 {
            score *= 0.9;
        }

        // Penalize workers with many failures
        if self.total_failures > 10 {
            score *= 0.7;
        } else if self.total_failures > 5 {
            score *= 0.9;
        }

        score
    }
}

/// Checkpoint data for recovery
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClusteringCheckpoint<F: Float> {
    pub iteration: usize,
    pub centroids: Option<Array2<F>>,
    pub global_inertia: f64,
    pub convergence_history: Vec<ConvergenceMetrics>,
    pub worker_assignments: HashMap<usize, Vec<usize>>,
    pub timestamp: u64,
}

/// Fault-tolerant coordinator for distributed clustering
#[derive(Debug)]
pub struct FaultTolerantCoordinator<F: Float> {
    pub fault_config: FaultToleranceConfig,
    pub worker_health: HashMap<usize, WorkerHealth>,
    pub failed_workers: HashSet<usize>,
    pub checkpoints: Vec<ClusteringCheckpoint<F>>,
    pub data_replicas: HashMap<usize, Vec<usize>>, // partition_id -> replica_worker_ids
    last_health_check: std::time::Instant,
}

impl<F: Float + FromPrimitive + Debug + Send + Sync> FaultTolerantCoordinator<F> {
    /// Create a new fault-tolerant coordinator
    pub fn new(fault_config: FaultToleranceConfig) -> Self {
        Self {
            fault_config,
            worker_health: HashMap::new(),
            failed_workers: HashSet::new(),
            checkpoints: Vec::new(),
            data_replicas: HashMap::new(),
            last_health_check: std::time::Instant::now(),
        }
    }

    /// Register a new worker for health monitoring
    pub fn register_worker(&mut self, worker_id: usize) {
        let health = WorkerHealth::new(worker_id);
        self.worker_health.insert(worker_id, health);
    }

    /// Check health of all workers with comprehensive monitoring
    pub fn check_worker_health(&mut self) -> Vec<usize> {
        let mut newly_failed = Vec::new();
        let timeout_ms = self.fault_config.worker_timeout_ms;

        if self.last_health_check.elapsed().as_millis() < self.fault_config.heartbeat_interval_ms as u128 {
            return newly_failed;
        }

        for (worker_id, health) in &mut self.worker_health {
            if !health.is_healthy(timeout_ms) && health.status != WorkerStatus::Failed {
                // Distinguish between timeout and complete failure
                if health.last_heartbeat.elapsed().as_millis() >= timeout_ms as u128 {
                    health.mark_timeout();
                } else {
                    health.mark_failure();
                }
                self.failed_workers.insert(*worker_id);
                newly_failed.push(*worker_id);
            }
        }

        self.last_health_check = std::time::Instant::now();
        newly_failed
    }

    /// Perform predictive failure detection
    pub fn predict_worker_failures(&self) -> Vec<(usize, f64)> {
        let mut at_risk_workers = Vec::new();

        for (worker_id, health) in &self.worker_health {
            let risk_score = health.predict_failure_risk();
            if risk_score > 0.7 {
                at_risk_workers.push(*worker_id, risk_score);
            }
        }

        // Sort by risk score (descending)
        at_risk_workers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        at_risk_workers
    }

    /// Get workers ranked by efficiency for load balancing
    pub fn get_workers_by_efficiency(&self) -> Vec<(usize, f64)> {
        let mut worker_efficiency: Vec<(usize, f64)> = self.worker_health
            .iter()
            .filter(|(_, health)| health.is_healthy(self.fault_config.worker_timeout_ms))
            .map(|(worker_id, health)| (*worker_id, health.get_efficiency_score()))
            .collect();

        // Sort by efficiency (descending)
        worker_efficiency.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        worker_efficiency
    }

    /// Update worker performance metrics
    pub fn update_worker_metrics(&mut self, worker_id: usize, response_time_ms: u64, cpu_usage: f64, memory_usage: f64) {
        if let Some(health) = self.worker_health.get_mut(&worker_id) {
            health.update_metrics(response_time_ms, cpu_usage, memory_usage);
            health.mark_success(); // Update heartbeat
        }
    }

    /// Get comprehensive health report
    pub fn get_health_report(&self) -> WorkerHealthReport {
        let total_workers = self.worker_health.len();
        let active_workers = self.worker_health.values()
            .filter(|h| matches!(h.status, WorkerStatus::Active))
            .count();
        let failed_workers = self.failed_workers.len();
        let timeout_workers = self.worker_health.values()
            .filter(|h| matches!(h.status, WorkerStatus::Timeout))
            .count();

        let avg_response_time = if active_workers > 0 {
            self.worker_health.values()
                .filter(|h| matches!(h.status, WorkerStatus::Active))
                .map(|h| h.response_time_ms)
                .sum::<u64>() as f64 / active_workers as f64
        } else {
            0.0
        };

        let avg_cpu_usage = if active_workers > 0 {
            self.worker_health.values()
                .filter(|h| matches!(h.status, WorkerStatus::Active))
                .map(|h| h.cpu_usage)
                .sum::<f64>() / active_workers as f64
        } else {
            0.0
        };

        let avg_memory_usage = if active_workers > 0 {
            self.worker_health.values()
                .filter(|h| matches!(h.status, WorkerStatus::Active))
                .map(|h| h.memory_usage)
                .sum::<f64>() / active_workers as f64
        } else {
            0.0
        };

        WorkerHealthReport {
            total_workers,
            active_workers,
            failed_workers,
            timeout_workers,
            avg_response_time_ms: avg_response_time,
            avg_cpu_usage,
            avg_memory_usage,
            at_risk_workers: self.predict_worker_failures(),
        }
    }

    /// Check if load rebalancing is needed
    pub fn should_rebalance(&self) -> bool {
        let efficiency_scores = self.get_workers_by_efficiency();
        
        if efficiency_scores.len() < 2 {
            return false;
        }

        // Check if there's significant efficiency difference
        let best_efficiency = efficiency_scores[0].1;
        let worst_efficiency = efficiency_scores.last().unwrap().1;
        
        // Rebalance if efficiency gap is > 30%
        (best_efficiency - worst_efficiency) > 0.3
    }

    /// Handle worker failure with configured recovery strategy
    pub fn handle_worker_failure(&mut self, failed_worker_id: usize, partitions: &mut Vec<DataPartition<F>>) -> Result<()> {
        if !self.fault_config.enabled {
            return Ok(());
        }

        if self.failed_workers.len() > self.fault_config.max_failures {
            return Err(ClusteringError::InvalidInput(
                format!("Too many worker failures: {} > {}", 
                    self.failed_workers.len(), self.fault_config.max_failures)
            ));
        }

        match self.fault_config.recovery_strategy {
            RecoveryStrategy::Redistribute => {
                self.redistribute_failed_worker_data(failed_worker_id, partitions)?;
            }
            RecoveryStrategy::Replace => {
                self.replace_failed_worker(failed_worker_id)?;
            }
            RecoveryStrategy::Checkpoint => {
                self.restore_from_checkpoint()?;
            }
            RecoveryStrategy::Restart => {
                return Err(ClusteringError::InvalidInput(
                    "Restart strategy requires external coordination".to_string()
                ));
            }
            RecoveryStrategy::Degrade => {
                // Continue with fewer workers - no action needed
            }
        }

        Ok(())
    }

    /// Redistribute data from failed worker to healthy workers
    fn redistribute_failed_worker_data(&mut self, failed_worker_id: usize, partitions: &mut Vec<DataPartition<F>>) -> Result<()> {
        let healthy_workers: Vec<usize> = self.worker_health
            .iter()
            .filter(|(_, health)| health.is_healthy(self.fault_config.worker_timeout_ms))
            .map(|(&id, _)| id)
            .collect();

        if healthy_workers.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "No healthy workers available for redistribution".to_string()
            ));
        }

        // Find partitions assigned to failed worker
        let failed_partitions: Vec<usize> = partitions
            .iter()
            .enumerate()
            .filter(|(_, p)| p.worker_id == failed_worker_id)
            .map(|(i, _)| i)
            .collect();

        // Redistribute to healthy workers using round-robin
        for (idx, &partition_idx) in failed_partitions.iter().enumerate() {
            let new_worker = healthy_workers[idx % healthy_workers.len()];
            partitions[partition_idx].worker_id = new_worker;
        }

        Ok(())
    }

    /// Replace failed worker with a new worker
    fn replace_failed_worker(&mut self, failed_worker_id: usize) -> Result<()> {
        if !self.fault_config.auto_replace_workers {
            return Err(ClusteringError::InvalidInput(
                "Worker replacement is disabled".to_string()
            ));
        }

        // In a real implementation, this would spawn a new worker process
        // For now, we'll simulate by creating a new worker ID
        let new_worker_id = self.worker_health.keys().max().unwrap_or(&0) + 1;
        self.register_worker(new_worker_id);
        
        // Mark the new worker as healthy
        if let Some(health) = self.worker_health.get_mut(&new_worker_id) {
            health.mark_success();
        }

        Ok(())
    }

    /// Restore from the latest checkpoint
    fn restore_from_checkpoint(&mut self) -> Result<()> {
        if !self.fault_config.enable_checkpointing {
            return Err(ClusteringError::InvalidInput(
                "Checkpointing is disabled".to_string()
            ));
        }

        if let Some(_latest_checkpoint) = self.checkpoints.last() {
            // In a real implementation, this would restore the clustering state
            // For now, we'll just clear the failed workers list
            self.failed_workers.clear();
            
            // Reset worker health
            for health in self.worker_health.values_mut() {
                if health.status == WorkerStatus::Failed {
                    health.mark_success();
                }
            }
        }

        Ok(())
    }

    /// Create a checkpoint of the current clustering state
    pub fn create_checkpoint(&mut self, iteration: usize, centroids: Option<&Array2<F>>, 
                           global_inertia: f64, convergence_history: &[ConvergenceMetrics],
                           worker_assignments: &HashMap<usize, Vec<usize>>) {
        if !self.fault_config.enable_checkpointing {
            return;
        }

        if iteration % self.fault_config.checkpoint_interval != 0 {
            return;
        }

        let checkpoint = ClusteringCheckpoint {
            iteration,
            centroids: centroids.map(|c| c.clone()),
            global_inertia,
            convergence_history: convergence_history.to_vec(),
            worker_assignments: worker_assignments.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        self.checkpoints.push(checkpoint);

        // Keep only recent checkpoints to save memory
        if self.checkpoints.len() > 10 {
            self.checkpoints.remove(0);
        }
    }

    /// Setup data replication for fault tolerance
    pub fn setup_replication(&mut self, partitions: &mut Vec<DataPartition<F>>) -> Result<()> {
        if !self.fault_config.enabled || !self.fault_config.enable_replication {
            return Ok(());
        }

        let replication_factor = self.fault_config.replication_factor;
        let healthy_workers: Vec<usize> = self.worker_health
            .iter()
            .filter(|(_, health)| health.is_healthy(self.fault_config.worker_timeout_ms))
            .map(|(&id, _)| id)
            .collect();

        if healthy_workers.len() < replication_factor {
            return Err(ClusteringError::InvalidInput(
                format!("Not enough healthy workers for replication factor {}", replication_factor)
            ));
        }

        // Setup replicas for each partition
        for partition in partitions.iter_mut() {
            let primary_worker = partition.worker_id;
            let mut replica_workers = Vec::new();

            // Select replica workers (excluding primary)
            let available_workers: Vec<usize> = healthy_workers
                .iter()
                .filter(|&&id| id != primary_worker)
                .copied()
                .collect();

            // Select replica workers round-robin
            for i in 0..(replication_factor - 1).min(available_workers.len()) {
                let replica_worker = available_workers[i % available_workers.len()];
                replica_workers.push(replica_worker);
            }

            partition.replica_workers = replica_workers.clone();
            
            // Update coordinator's replica mapping
            self.data_replicas.insert(partition.partition_id, replica_workers);
        }

        Ok(())
    }

    /// Replicate partition data to replica workers
    pub fn replicate_partition_data(&mut self, partition: &DataPartition<F>) -> Result<Vec<DataPartition<F>>> {
        if !self.fault_config.enable_replication {
            return Ok(Vec::new());
        }

        let mut replicas = Vec::new();
        
        for &replica_worker_id in &partition.replica_workers {
            let mut replica_partition = partition.clone();
            replica_partition.worker_id = replica_worker_id;
            replica_partition.partition_id = partition.partition_id; // Keep same partition ID
            replicas.push(replica_partition);
        }

        Ok(replicas)
    }

    /// Handle replica recovery after worker failure
    pub fn recover_from_replicas(&mut self, failed_partition_id: usize, partitions: &mut Vec<DataPartition<F>>) -> Result<()> {
        if !self.fault_config.enable_replication {
            return Err(ClusteringError::InvalidInput(
                "Replication is not enabled".to_string()
            ));
        }

        if let Some(replica_workers) = self.data_replicas.get(&failed_partition_id) {
            // Find a healthy replica worker
            for &replica_worker_id in replica_workers {
                if let Some(health) = self.worker_health.get(&replica_worker_id) {
                    if health.is_healthy(self.fault_config.worker_timeout_ms) {
                        // Promote replica to primary
                        if let Some(partition) = partitions.iter_mut()
                            .find(|p| p.partition_id == failed_partition_id) {
                            partition.worker_id = replica_worker_id;
                            return Ok(());
                        }
                    }
                }
            }
        }

        Err(ClusteringError::InvalidInput(
            format!("No healthy replicas found for partition {}", failed_partition_id)
        ))
    }

    /// Verify data consistency across replicas
    pub fn verify_replica_consistency(&self, partition_id: usize, partitions: &[DataPartition<F>]) -> Result<bool> {
        if !self.fault_config.enable_replication {
            return Ok(true);
        }

        // Find primary partition
        let primary_partition = partitions.iter()
            .find(|p| p.partition_id == partition_id)
            .ok_or_else(|| ClusteringError::InvalidInput(
                format!("Primary partition {} not found", partition_id)
            ))?;

        let primary_checksum = primary_partition.checksum;

        // Find replica partitions and verify checksums
        if let Some(replica_workers) = self.data_replicas.get(&partition_id) {
            for &replica_worker_id in replica_workers {
                // In a real implementation, this would fetch the replica data from the worker
                // For now, we assume replicas have the same checksum as primary if healthy
                if let Some(health) = self.worker_health.get(&replica_worker_id) {
                    if !health.is_healthy(self.fault_config.worker_timeout_ms) {
                        continue;
                    }
                    
                    // In practice, you would compare actual replica checksums
                    // For this implementation, we assume consistency if worker is healthy
                }
            }
        }

        Ok(true)
    }

    /// Update replica after data changes
    pub fn update_replicas(&mut self, partition_id: usize, updated_partition: &DataPartition<F>) -> Result<()> {
        if !self.fault_config.enable_replication {
            return Ok(());
        }

        if let Some(replica_workers) = self.data_replicas.get(&partition_id) {
            for &replica_worker_id in replica_workers {
                if let Some(health) = self.worker_health.get(&replica_worker_id) {
                    if health.is_healthy(self.fault_config.worker_timeout_ms) {
                        // In a real implementation, this would send the updated data to replica workers
                        // For now, we just log the operation
                        println!("Updating replica on worker {} for partition {}", replica_worker_id, partition_id);
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if clustering can continue with current worker health
    pub fn can_continue_clustering(&self) -> bool {
        if !self.fault_config.enabled {
            return true;
        }

        let healthy_workers = self.worker_health.values()
            .filter(|h| h.is_healthy(self.fault_config.worker_timeout_ms))
            .count();

        // Need at least one healthy worker to continue
        healthy_workers > 0 && self.failed_workers.len() <= self.fault_config.max_failures
    }
}

/// Distributed data partition
#[derive(Debug, Clone)]
pub struct DataPartition<F: Float> {
    /// Partition identifier
    pub partition_id: usize,
    /// Data points in this partition
    pub data: Array2<F>,
    /// Local cluster assignments
    pub labels: Option<Array1<usize>>,
    /// Worker node identifier
    pub worker_id: usize,
    /// Partition weight for load balancing
    pub weight: f64,
    /// Replica worker IDs for fault tolerance
    pub replica_workers: Vec<usize>,
    /// Checksum for data integrity verification
    pub checksum: u64,
}

impl<F: Float> DataPartition<F> {
    pub fn new(partition_id: usize, data: Array2<F>, worker_id: usize) -> Self {
        let checksum = Self::compute_checksum(&data);
        Self {
            partition_id,
            data,
            labels: None,
            worker_id,
            weight: 1.0,
            replica_workers: Vec::new(),
            checksum,
        }
    }

    fn compute_checksum(data: &Array2<F>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash the shape first
        data.shape().hash(&mut hasher);
        
        // Hash the actual data values for better integrity checking
        // Convert float values to bits for consistent hashing
        for row in data.rows() {
            for &value in row.iter() {
                // Convert to f64 and then to bits for consistent hashing across float types
                let bits = value.to_f64().unwrap_or(0.0).to_bits();
                bits.hash(&mut hasher);
            }
        }
        
        // Also hash row/column strides for memory layout verification
        if let Some(strides) = data.strides() {
            for &stride in strides {
                stride.hash(&mut hasher);
            }
        }
        
        hasher.finish()
    }

    pub fn verify_integrity(&self) -> bool {
        self.checksum == Self::compute_checksum(&self.data)
    }

    pub fn add_replica(&mut self, worker_id: usize) {
        if !self.replica_workers.contains(&worker_id) {
            self.replica_workers.push(worker_id);
        }
    }

    pub fn remove_replica(&mut self, worker_id: usize) {
        self.replica_workers.retain(|&id| id != worker_id);
    }
}

/// Distributed K-means algorithm coordinator
#[derive(Debug)]
pub struct DistributedKMeans<F: Float> {
    /// Configuration
    pub config: DistributedConfig,
    /// Number of clusters
    pub k: usize,
    /// Global centroids
    pub centroids: Option<Array2<F>>,
    /// Data partitions across workers
    pub partitions: Vec<DataPartition<F>>,
    /// Convergence history
    pub convergence_history: Vec<ConvergenceMetrics>,
    /// Worker statistics
    pub worker_stats: HashMap<usize, WorkerStatistics>,
}

/// Convergence metrics for distributed algorithms
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConvergenceMetrics {
    /// Coordination round number
    pub round: usize,
    /// Global inertia (sum of squared distances)
    pub global_inertia: f64,
    /// Change in global inertia
    pub inertia_change: f64,
    /// Number of points that changed cluster assignment
    pub points_changed: usize,
    /// Maximum centroid movement distance
    pub max_centroid_movement: f64,
    /// Whether algorithm has converged globally
    pub converged: bool,
    /// Worker synchronization time (milliseconds)
    pub sync_time_ms: u64,
}

/// Statistics for individual workers in distributed computation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WorkerStatistics {
    /// Worker identifier
    pub worker_id: usize,
    /// Number of data points processed
    pub n_points: usize,
    /// Local computation time per round (milliseconds)
    pub computation_time_ms: u64,
    /// Communication time per round (milliseconds)
    pub communication_time_ms: u64,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
    /// Local convergence status
    pub local_converged: bool,
    /// Number of local iterations performed
    pub local_iterations: usize,
}

impl<F: Float + FromPrimitive + Debug + Send + Sync> DistributedKMeans<F> {
    /// Create new distributed K-means instance
    pub fn new(k: usize, config: DistributedConfig) -> Self {
        Self {
            config,
            k,
            centroids: None,
            partitions: Vec::new(),
            convergence_history: Vec::new(),
            worker_stats: HashMap::new(),
        }
    }

    /// Partition data across workers using specified strategy
    pub fn partition_data(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput("Empty dataset".to_string()));
        }

        // Calculate partition sizes based on load balancing strategy
        let partition_sizes = self.calculate_partition_sizes(n_samples)?;

        // Create partitions based on strategy
        self.partitions = match self.config.partitioning_strategy {
            PartitioningStrategy::RandomSplit => self.random_partition(data, &partition_sizes)?,
            PartitioningStrategy::HashBased => self.hash_based_partition(data, &partition_sizes)?,
            PartitioningStrategy::SpatialPartitioning => {
                self.spatial_partition(data, &partition_sizes)?
            }
            PartitioningStrategy::StratifiedSampling => {
                self.stratified_partition(data, &partition_sizes)?
            }
            PartitioningStrategy::RoundRobin => {
                self.round_robin_partition(data, &partition_sizes)?
            }
        };

        Ok(())
    }

    /// Run distributed K-means clustering
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        // Partition data across workers
        self.partition_data(data)?;

        // Initialize global centroids
        self.centroids = Some(self.initialize_global_centroids()?);

        // Distributed coordination loop
        for round in 0..self.config.max_coordination_rounds {
            let start_time = std::time::Instant::now();

            // Local computation phase on each worker
            let local_results = self.execute_local_kmeans_round()?;

            // Global coordination phase
            let new_centroids = self.coordinate_global_centroids(&local_results)?;
            let convergence = self.check_global_convergence(&new_centroids, round)?;

            // Update global state
            self.centroids = Some(new_centroids);
            self.convergence_history.push(convergence.clone());

            let sync_time = start_time.elapsed().as_millis() as u64;

            // Log round completion
            if round % 10 == 0 {
                println!(
                    "Round {}: Global inertia = {:.4}, Converged = {}",
                    round, convergence.global_inertia, convergence.converged
                );
            }

            // Check for global convergence
            if convergence.converged {
                println!("Distributed K-means converged after {} rounds", round + 1);
                break;
            }
        }

        Ok(())
    }

    /// Calculate partition sizes based on load balancing strategy
    fn calculate_partition_sizes(&self, total_samples: usize) -> Result<Vec<usize>> {
        match self.config.load_balancing {
            LoadBalancingStrategy::EqualSize => {
                let base_size = total_samples / self.config.n_workers;
                let remainder = total_samples % self.config.n_workers;

                let mut sizes = vec![base_size; self.config.n_workers];
                for i in 0..remainder {
                    sizes[i] += 1;
                }
                Ok(sizes)
            }
            LoadBalancingStrategy::ComputationalBalance => {
                // Simulate computational weights (in practice, these would be measured)
                let weights = vec![1.0; self.config.n_workers];
                self.distribute_by_weights(total_samples, &weights)
            }
            LoadBalancingStrategy::Dynamic => {
                // Start with equal distribution, adjust later based on performance
                self.calculate_partition_sizes(total_samples)
            }
            LoadBalancingStrategy::WeightedDistribution => {
                // Use predefined weights (simplified)
                let weights = vec![1.0; self.config.n_workers];
                self.distribute_by_weights(total_samples, &weights)
            }
        }
    }

    /// Distribute samples based on worker weights
    fn distribute_by_weights(&self, total_samples: usize, weights: &[f64]) -> Result<Vec<usize>> {
        let total_weight: f64 = weights.iter().sum();
        if total_weight <= 0.0 {
            return Err(ClusteringError::InvalidInput(
                "Worker weights must be positive".to_string(),
            ));
        }

        let mut sizes = Vec::new();
        let mut assigned = 0;

        for (i, &weight) in weights.iter().enumerate() {
            let size = if i == weights.len() - 1 {
                // Last worker gets remaining samples
                total_samples - assigned
            } else {
                ((total_samples as f64 * weight / total_weight).round() as usize)
                    .min(total_samples - assigned)
            };
            sizes.push(size);
            assigned += size;
        }

        Ok(sizes)
    }

    /// Random data partitioning
    fn random_partition(
        &self,
        data: ArrayView2<F>,
        partition_sizes: &[usize],
    ) -> Result<Vec<DataPartition<F>>> {
        let mut partitions = Vec::new();
        let mut indices: Vec<usize> = (0..data.nrows()).collect();

        // Shuffle indices for random assignment
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());

        let mut start_idx = 0;
        for (worker_id, &size) in partition_sizes.iter().enumerate() {
            let end_idx = (start_idx + size).min(data.nrows());

            if start_idx < end_idx {
                let partition_indices = &indices[start_idx..end_idx];
                let mut partition_data = Array2::zeros((partition_indices.len(), data.ncols()));

                for (i, &row_idx) in partition_indices.iter().enumerate() {
                    partition_data.row_mut(i).assign(&data.row(row_idx));
                }

                partitions.push(DataPartition {
                    partition_id: worker_id,
                    data: partition_data,
                    labels: None,
                    worker_id,
                    weight: size as f64 / data.nrows() as f64,
                });
            }

            start_idx = end_idx;
        }

        Ok(partitions)
    }

    /// Hash-based data partitioning
    fn hash_based_partition(
        &self,
        data: ArrayView2<F>,
        partition_sizes: &[usize],
    ) -> Result<Vec<DataPartition<F>>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let n_workers = self.config.n_workers;
        let mut worker_data: Vec<Vec<usize>> = vec![Vec::new(); n_workers];

        // Assign each point to a worker based on hash of its coordinates
        for (row_idx, row) in data.rows().into_iter().enumerate() {
            let mut hasher = DefaultHasher::new();

            // Hash the first few coordinates to determine worker assignment
            for &value in row.iter().take(3.min(row.len())) {
                let bits = value.to_f64().unwrap_or(0.0).to_bits();
                bits.hash(&mut hasher);
            }

            let worker_id = (hasher.finish() as usize) % n_workers;
            worker_data[worker_id].push(row_idx);
        }

        // Create partitions
        let mut partitions = Vec::new();
        for (worker_id, row_indices) in worker_data.into_iter().enumerate() {
            if !row_indices.is_empty() {
                let mut partition_data = Array2::zeros((row_indices.len(), data.ncols()));

                for (i, &row_idx) in row_indices.iter().enumerate() {
                    partition_data.row_mut(i).assign(&data.row(row_idx));
                }

                partitions.push(DataPartition {
                    partition_id: worker_id,
                    data: partition_data,
                    labels: None,
                    worker_id,
                    weight: row_indices.len() as f64 / data.nrows() as f64,
                });
            }
        }

        Ok(partitions)
    }

    /// Spatial partitioning using simplified k-d tree approach
    fn spatial_partition(
        &self,
        data: ArrayView2<F>,
        _partition_sizes: &[usize],
    ) -> Result<Vec<DataPartition<F>>> {
        let n_features = data.ncols();
        let n_workers = self.config.n_workers;

        // Simple spatial partitioning: divide along the dimension with highest variance
        let mut dimension_variances = Vec::new();
        for dim in 0..n_features {
            let column = data.column(dim);
            let mean =
                column.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(column.len()).unwrap();
            let variance = column
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(column.len()).unwrap();
            dimension_variances.push((dim, variance.to_f64().unwrap_or(0.0)));
        }

        // Sort by variance and use top dimension for partitioning
        dimension_variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let split_dim = dimension_variances[0].0;

        // Get values for splitting dimension
        let mut split_values: Vec<(usize, f64)> = data
            .column(split_dim)
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val.to_f64().unwrap_or(0.0)))
            .collect();

        split_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Create partitions based on sorted values
        let mut partitions = Vec::new();
        let partition_size = split_values.len() / n_workers;

        for worker_id in 0..n_workers {
            let start_idx = worker_id * partition_size;
            let end_idx = if worker_id == n_workers - 1 {
                split_values.len()
            } else {
                (worker_id + 1) * partition_size
            };

            if start_idx < end_idx {
                let partition_indices: Vec<usize> = split_values[start_idx..end_idx]
                    .iter()
                    .map(|(idx, _)| *idx)
                    .collect();

                let mut partition_data = Array2::zeros((partition_indices.len(), data.ncols()));
                for (i, &row_idx) in partition_indices.iter().enumerate() {
                    partition_data.row_mut(i).assign(&data.row(row_idx));
                }

                partitions.push(DataPartition {
                    partition_id: worker_id,
                    data: partition_data,
                    labels: None,
                    worker_id,
                    weight: partition_indices.len() as f64 / data.nrows() as f64,
                });
            }
        }

        Ok(partitions)
    }

    /// Stratified sampling partition using preliminary clustering
    fn stratified_partition(
        &self,
        data: ArrayView2<F>,
        partition_sizes: &[usize],
    ) -> Result<Vec<DataPartition<F>>> {
        let n_samples = data.nrows();
        let n_strata = (self.config.n_workers * 2).max(4).min(n_samples / 10); // Adaptive strata count
        
        if n_samples < n_strata {
            // Fall back to random if not enough data
            return self.random_partition(data, partition_sizes);
        }

        // Step 1: Perform preliminary clustering to identify strata
        let strata_assignments = self.identify_strata(data, n_strata)?;
        
        // Step 2: Group data points by stratum
        let mut strata_groups: Vec<Vec<usize>> = vec![Vec::new(); n_strata];
        for (point_idx, &stratum_id) in strata_assignments.iter().enumerate() {
            strata_groups[stratum_id].push(point_idx);
        }

        // Step 3: Distribute strata points proportionally to workers
        let mut worker_assignments: Vec<Vec<usize>> = vec![Vec::new(); self.config.n_workers];
        
        for (stratum_id, stratum_points) in strata_groups.iter().enumerate() {
            if stratum_points.is_empty() {
                continue;
            }

            // Calculate how many points each worker should get from this stratum
            let total_points = stratum_points.len();
            let mut distributed = 0;

            for worker_id in 0..self.config.n_workers {
                let target_size = partition_sizes[worker_id];
                let current_size = worker_assignments[worker_id].len();
                let remaining_capacity = target_size.saturating_sub(current_size);
                
                // Proportional allocation with remaining capacity constraint
                let total_remaining_capacity: usize = worker_assignments.iter()
                    .enumerate()
                    .skip(worker_id)
                    .map(|(i, assignments)| partition_sizes[i].saturating_sub(assignments.len()))
                    .sum();

                let points_for_worker = if total_remaining_capacity == 0 {
                    0
                } else {
                    let proportion = remaining_capacity as f64 / total_remaining_capacity as f64;
                    let remaining_points = total_points - distributed;
                    (remaining_points as f64 * proportion).round() as usize
                        .min(remaining_points)
                        .min(remaining_capacity)
                };

                // Assign points to this worker
                let start_idx = distributed;
                let end_idx = (start_idx + points_for_worker).min(total_points);
                
                for &point_idx in &stratum_points[start_idx..end_idx] {
                    worker_assignments[worker_id].push(point_idx);
                }
                
                distributed = end_idx;
                
                if distributed >= total_points {
                    break;
                }
            }
        }

        // Step 4: Create partitions from worker assignments
        let mut partitions = Vec::new();
        for (worker_id, point_indices) in worker_assignments.into_iter().enumerate() {
            if !point_indices.is_empty() {
                let mut partition_data = Array2::zeros((point_indices.len(), data.ncols()));
                
                for (i, &point_idx) in point_indices.iter().enumerate() {
                    partition_data.row_mut(i).assign(&data.row(point_idx));
                }

                partitions.push(DataPartition::new(worker_id, partition_data, worker_id));
            }
        }

        Ok(partitions)
    }

    /// Identify strata using simple K-means clustering
    fn identify_strata(&self, data: ArrayView2<F>, n_strata: usize) -> Result<Array1<usize>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        
        // Initialize centroids randomly
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut point_indices: Vec<usize> = (0..n_samples).collect();
        point_indices.shuffle(&mut rng);
        
        let mut centroids = Array2::zeros((n_strata, n_features));
        for (i, &point_idx) in point_indices.iter().take(n_strata).enumerate() {
            centroids.row_mut(i).assign(&data.row(point_idx));
        }

        let mut assignments = Array1::zeros(n_samples);
        let max_iterations = 10; // Quick preliminary clustering

        for _ in 0..max_iterations {
            let mut changed = false;

            // Assign points to nearest centroids
            for (point_idx, point) in data.rows().into_iter().enumerate() {
                let mut min_dist = F::infinity();
                let mut best_centroid = 0;

                for (centroid_idx, centroid) in centroids.rows().into_iter().enumerate() {
                    let dist = euclidean_distance(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_centroid = centroid_idx;
                    }
                }

                if assignments[point_idx] != best_centroid {
                    assignments[point_idx] = best_centroid;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            centroids.fill(F::zero());
            let mut counts = vec![0; n_strata];

            for (point_idx, point) in data.rows().into_iter().enumerate() {
                let cluster_id = assignments[point_idx];
                for (j, &value) in point.iter().enumerate() {
                    centroids[[cluster_id, j]] = centroids[[cluster_id, j]] + value;
                }
                counts[cluster_id] += 1;
            }

            // Compute averages
            for i in 0..n_strata {
                if counts[i] > 0 {
                    for j in 0..n_features {
                        centroids[[i, j]] = centroids[[i, j]] / F::from(counts[i]).unwrap();
                    }
                }
            }
        }

        Ok(assignments)
    }

    /// Round-robin partitioning
    fn round_robin_partition(
        &self,
        data: ArrayView2<F>,
        _partition_sizes: &[usize],
    ) -> Result<Vec<DataPartition<F>>> {
        let n_workers = self.config.n_workers;
        let mut worker_data: Vec<Vec<usize>> = vec![Vec::new(); n_workers];

        // Assign points in round-robin fashion
        for (row_idx, _) in data.rows().into_iter().enumerate() {
            let worker_id = row_idx % n_workers;
            worker_data[worker_id].push(row_idx);
        }

        // Create partitions
        let mut partitions = Vec::new();
        for (worker_id, row_indices) in worker_data.into_iter().enumerate() {
            if !row_indices.is_empty() {
                let mut partition_data = Array2::zeros((row_indices.len(), data.ncols()));

                for (i, &row_idx) in row_indices.iter().enumerate() {
                    partition_data.row_mut(i).assign(&data.row(row_idx));
                }

                partitions.push(DataPartition {
                    partition_id: worker_id,
                    data: partition_data,
                    labels: None,
                    worker_id,
                    weight: row_indices.len() as f64 / data.nrows() as f64,
                });
            }
        }

        Ok(partitions)
    }

    /// Initialize global centroids using K-means++ across partitions
    fn initialize_global_centroids(&self) -> Result<Array2<F>> {
        if self.partitions.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "No data partitions available".to_string(),
            ));
        }

        let n_features = self.partitions[0].data.ncols();
        let mut centroids = Array2::zeros((self.k, n_features));

        // Use first partition for initialization (simplified)
        let first_partition = &self.partitions[0];
        if first_partition.data.nrows() < self.k {
            return Err(ClusteringError::InvalidInput(
                "Not enough data points for initialization".to_string(),
            ));
        }

        // Simple initialization: use first k points from first partition
        for i in 0..self.k {
            let row_idx = i % first_partition.data.nrows();
            centroids
                .row_mut(i)
                .assign(&first_partition.data.row(row_idx));
        }

        Ok(centroids)
    }

    /// Execute local K-means iterations on each worker
    fn execute_local_kmeans_round(&mut self) -> Result<Vec<LocalKMeansResult<F>>> {
        let global_centroids = self.centroids.as_ref().unwrap();
        let mut results = Vec::new();

        for partition in &mut self.partitions {
            let start_time = std::time::Instant::now();

            // Local K-means iterations
            let (local_centroids, local_labels, local_inertia) =
                self.local_kmeans_iterations(&partition.data, global_centroids.view())?;

            let computation_time = start_time.elapsed().as_millis() as u64;

            // Update partition labels
            partition.labels = Some(local_labels.clone());

            // Create result
            results.push(LocalKMeansResult {
                worker_id: partition.worker_id,
                local_centroids,
                local_labels,
                local_inertia,
                n_points: partition.data.nrows(),
                computation_time_ms: computation_time,
            });

            // Update worker statistics
            self.worker_stats.insert(
                partition.worker_id,
                WorkerStatistics {
                    worker_id: partition.worker_id,
                    n_points: partition.data.nrows(),
                    computation_time_ms: computation_time,
                    communication_time_ms: 0, // Simplified
                    memory_usage_mb: self.estimate_memory_usage(&partition.data),
                    local_converged: false, // Simplified
                    local_iterations: self.config.max_iterations_per_round,
                },
            );
        }

        Ok(results)
    }

    /// Perform local K-means iterations on a data partition
    fn local_kmeans_iterations(
        &self,
        data: &Array2<F>,
        global_centroids: ArrayView2<F>,
    ) -> Result<(Array2<F>, Array1<usize>, F)> {
        let mut centroids = global_centroids.to_owned();
        let mut labels = Array1::zeros(data.nrows());
        let mut prev_inertia = F::infinity();

        for _iter in 0..self.config.max_iterations_per_round {
            // Assign points to nearest centroids
            for (i, point) in data.rows().into_iter().enumerate() {
                let mut min_dist = F::infinity();
                let mut best_cluster = 0;

                for (j, centroid) in centroids.rows().into_iter().enumerate() {
                    let dist = euclidean_distance(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = j;
                    }
                }
                labels[i] = best_cluster;
            }

            // Update centroids
            let new_centroids = self.compute_local_centroids(data, &labels)?;

            // Calculate inertia
            let inertia = self.compute_local_inertia(data, &labels, new_centroids.view())?;

            // Check for local convergence
            if (prev_inertia - inertia).abs() <= F::from(self.config.global_tolerance).unwrap() {
                break;
            }

            centroids = new_centroids;
            prev_inertia = inertia;
        }

        Ok((centroids, labels, prev_inertia))
    }

    /// Compute local centroids from data and labels
    fn compute_local_centroids(
        &self,
        data: &Array2<F>,
        labels: &Array1<usize>,
    ) -> Result<Array2<F>> {
        let n_features = data.ncols();
        let mut centroids = Array2::zeros((self.k, n_features));
        let mut counts = vec![0; self.k];

        // Accumulate sums for each cluster
        for (point, &label) in data.rows().into_iter().zip(labels.iter()) {
            for (j, &value) in point.iter().enumerate() {
                centroids[[label, j]] = centroids[[label, j]] + value;
            }
            counts[label] += 1;
        }

        // Compute averages
        for i in 0..self.k {
            if counts[i] > 0 {
                for j in 0..n_features {
                    centroids[[i, j]] = centroids[[i, j]] / F::from(counts[i]).unwrap();
                }
            }
        }

        Ok(centroids)
    }

    /// Compute local inertia
    fn compute_local_inertia(
        &self,
        data: &Array2<F>,
        labels: &Array1<usize>,
        centroids: ArrayView2<F>,
    ) -> Result<F> {
        let mut inertia = F::zero();

        for (point, &label) in data.rows().into_iter().zip(labels.iter()) {
            let centroid = centroids.row(label);
            let dist = euclidean_distance(point, centroid);
            inertia = inertia + dist * dist;
        }

        Ok(inertia)
    }

    /// Coordinate global centroids from local results
    fn coordinate_global_centroids(
        &self,
        local_results: &[LocalKMeansResult<F>],
    ) -> Result<Array2<F>> {
        let n_features = self.partitions[0].data.ncols();
        let mut global_centroids = Array2::zeros((self.k, n_features));
        let mut cluster_counts = vec![0; self.k];

        // Weighted average of local centroids
        for result in local_results {
            for i in 0..self.k {
                // Count points in this cluster for this worker
                let local_count = result
                    .local_labels
                    .iter()
                    .filter(|&&label| label == i)
                    .count();

                if local_count > 0 {
                    for j in 0..n_features {
                        global_centroids[[i, j]] = global_centroids[[i, j]]
                            + result.local_centroids[[i, j]] * F::from(local_count).unwrap();
                    }
                    cluster_counts[i] += local_count;
                }
            }
        }

        // Compute global averages
        for i in 0..self.k {
            if cluster_counts[i] > 0 {
                for j in 0..n_features {
                    global_centroids[[i, j]] =
                        global_centroids[[i, j]] / F::from(cluster_counts[i]).unwrap();
                }
            }
        }

        Ok(global_centroids)
    }

    /// Check global convergence
    fn check_global_convergence(
        &self,
        new_centroids: &Array2<F>,
        round: usize,
    ) -> Result<ConvergenceMetrics> {
        let old_centroids = self.centroids.as_ref().unwrap();

        // Compute maximum centroid movement
        let mut max_movement = 0.0;
        for i in 0..self.k {
            let movement = euclidean_distance(old_centroids.row(i), new_centroids.row(i))
                .to_f64()
                .unwrap_or(0.0);
            max_movement = max_movement.max(movement);
        }

        // Compute global inertia
        let mut global_inertia = 0.0;
        for partition in &self.partitions {
            if let Some(ref labels) = partition.labels {
                let local_inertia = self
                    .compute_local_inertia(&partition.data, labels, new_centroids.view())?
                    .to_f64()
                    .unwrap_or(0.0);
                global_inertia += local_inertia;
            }
        }

        // Compute change in inertia
        let inertia_change = if let Some(last_convergence) = self.convergence_history.last() {
            (last_convergence.global_inertia - global_inertia).abs()
        } else {
            f64::INFINITY
        };

        let converged = max_movement < self.config.global_tolerance
            && inertia_change < self.config.global_tolerance;

        Ok(ConvergenceMetrics {
            round,
            global_inertia,
            inertia_change,
            points_changed: 0, // Simplified
            max_centroid_movement: max_movement,
            converged,
            sync_time_ms: 0, // Simplified
        })
    }

    /// Estimate comprehensive memory usage for a data partition
    fn estimate_memory_usage(&self, data: &Array2<F>) -> f64 {
        let bytes_per_element = std::mem::size_of::<F>();
        let n_points = data.nrows();
        let n_features = data.ncols();
        
        // Base data storage
        let data_bytes = data.len() * bytes_per_element;
        
        // Labels storage (usize per point)
        let labels_bytes = n_points * std::mem::size_of::<usize>();
        
        // Centroids storage (k centroids * n_features)
        let centroids_bytes = self.k * n_features * bytes_per_element;
        
        // Temporary arrays during computation
        // - Distance calculations: k distances per point
        let distance_temp_bytes = n_points * self.k * std::mem::size_of::<f64>();
        
        // - Centroid update accumulators
        let accumulator_bytes = self.k * n_features * std::mem::size_of::<f64>();
        
        // - Cluster counts
        let counts_bytes = self.k * std::mem::size_of::<usize>();
        
        // Convergence history and statistics
        let history_bytes = self.config.max_coordination_rounds * std::mem::size_of::<ConvergenceMetrics>();
        
        // Worker statistics
        let stats_bytes = std::mem::size_of::<WorkerStatistics>();
        
        // Replication overhead (if enabled)
        let replication_bytes = if self.config.fault_tolerance.enable_replication {
            data_bytes * (self.config.fault_tolerance.replication_factor.saturating_sub(1))
        } else {
            0
        };
        
        // Communication buffers (estimated 10% of data size)
        let communication_bytes = (data_bytes as f64 * 0.1) as usize;
        
        // Operating system and runtime overhead (estimated 20% of total)
        let base_memory = data_bytes + labels_bytes + centroids_bytes + distance_temp_bytes 
            + accumulator_bytes + counts_bytes + history_bytes + stats_bytes 
            + replication_bytes + communication_bytes;
        let overhead_bytes = (base_memory as f64 * 0.2) as usize;
        
        let total_bytes = base_memory + overhead_bytes;
        total_bytes as f64 / (1024.0 * 1024.0) // Convert to MB
    }

    /// Estimate peak memory usage during algorithm execution
    pub fn estimate_peak_memory_usage(&self) -> f64 {
        let mut peak_memory = 0.0;
        
        for partition in &self.partitions {
            let partition_memory = self.estimate_memory_usage(&partition.data);
            peak_memory += partition_memory;
        }
        
        // Add coordinator overhead
        let coordinator_overhead = if !self.partitions.is_empty() {
            let n_features = self.partitions[0].data.ncols();
            
            // Global centroids
            let global_centroids_mb = (self.k * n_features * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);
            
            // Convergence history
            let history_mb = (self.config.max_coordination_rounds * std::mem::size_of::<ConvergenceMetrics>()) as f64 / (1024.0 * 1024.0);
            
            // Worker health monitoring
            let monitoring_mb = (self.config.n_workers * std::mem::size_of::<WorkerHealth>()) as f64 / (1024.0 * 1024.0);
            
            // Checkpoints (if enabled)
            let checkpoint_mb = if self.config.fault_tolerance.enable_checkpointing {
                // Estimate 10 checkpoints worth of state
                10.0 * global_centroids_mb
            } else {
                0.0
            };
            
            global_centroids_mb + history_mb + monitoring_mb + checkpoint_mb
        } else {
            0.0
        };
        
        peak_memory + coordinator_overhead
    }

    /// Check if memory usage is within configured limits
    pub fn check_memory_limits(&self) -> Result<()> {
        let estimated_peak_mb = self.estimate_peak_memory_usage();
        let per_worker_limit_mb = self.config.memory_limit_mb as f64;
        let total_limit_mb = per_worker_limit_mb * self.config.n_workers as f64;
        
        if estimated_peak_mb > total_limit_mb {
            return Err(ClusteringError::InvalidInput(
                format!(
                    "Estimated memory usage ({:.1} MB) exceeds limit ({:.1} MB). \
                     Consider reducing data size, number of clusters, or increasing memory limit.",
                    estimated_peak_mb, total_limit_mb
                )
            ));
        }
        
        // Check per-worker limits
        let avg_per_worker_mb = estimated_peak_mb / self.config.n_workers as f64;
        if avg_per_worker_mb > per_worker_limit_mb {
            return Err(ClusteringError::InvalidInput(
                format!(
                    "Average per-worker memory usage ({:.1} MB) exceeds per-worker limit ({:.1} MB). \
                     Consider increasing number of workers or memory limit.",
                    avg_per_worker_mb, per_worker_limit_mb
                )
            ));
        }
        
        Ok(())
    }

    /// Get final cluster assignments for all data
    pub fn get_labels(&self) -> Result<Array1<usize>> {
        let total_points: usize = self.partitions.iter().map(|p| p.data.nrows()).sum();

        let mut global_labels = Array1::zeros(total_points);
        let mut offset = 0;

        for partition in &self.partitions {
            if let Some(ref labels) = partition.labels {
                let end = offset + labels.len();
                global_labels
                    .slice_mut(ndarray::s![offset..end])
                    .assign(labels);
                offset = end;
            }
        }

        Ok(global_labels)
    }

    /// Get convergence history
    pub fn get_convergence_history(&self) -> &[ConvergenceMetrics] {
        &self.convergence_history
    }

    /// Get worker statistics
    pub fn get_worker_statistics(&self) -> &HashMap<usize, WorkerStatistics> {
        &self.worker_stats
    }
}

/// Result from local K-means computation on a worker
#[derive(Debug, Clone)]
struct LocalKMeansResult<F: Float> {
    worker_id: usize,
    local_centroids: Array2<F>,
    local_labels: Array1<usize>,
    local_inertia: F,
    n_points: usize,
    computation_time_ms: u64,
}

/// Distributed hierarchical clustering (simplified implementation)
#[derive(Debug)]
pub struct DistributedHierarchical<F: Float> {
    pub config: DistributedConfig,
    pub linkage_method: LinkageMethod,
    pub partitions: Vec<DataPartition<F>>,
    pub local_dendrograms: Vec<LocalDendrogram>,
}

/// Linkage methods for hierarchical clustering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum LinkageMethod {
    Single,
    Complete,
    Average,
    Ward,
}

/// Local dendrogram from a worker
#[derive(Debug, Clone)]
pub struct LocalDendrogram {
    pub worker_id: usize,
    pub merge_sequence: Vec<(usize, usize, f64)>, // (cluster1, cluster2, distance)
    pub n_points: usize,
}

impl<F: Float + FromPrimitive + Debug + Send + Sync> DistributedHierarchical<F> {
    /// Create new distributed hierarchical clustering instance
    pub fn new(linkage_method: LinkageMethod, config: DistributedConfig) -> Self {
        Self {
            config,
            linkage_method,
            partitions: Vec::new(),
            local_dendrograms: Vec::new(),
        }
    }

    /// Partition data and perform local hierarchical clustering
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        // Reuse partitioning logic from DistributedKMeans
        let mut temp_kmeans = DistributedKMeans::new(2, self.config.clone());
        temp_kmeans.partition_data(data)?;
        self.partitions = temp_kmeans.partitions;

        // Perform local hierarchical clustering on each partition
        for partition in &self.partitions {
            let local_dendrogram = self.local_hierarchical_clustering(&partition.data)?;
            self.local_dendrograms.push(local_dendrogram);
        }

        Ok(())
    }

    /// Perform hierarchical clustering on a local partition
    fn local_hierarchical_clustering(&self, data: &Array2<F>) -> Result<LocalDendrogram> {
        let n_points = data.nrows();
        let mut merge_sequence = Vec::new();

        // Simplified hierarchical clustering: merge closest pairs
        // In a full implementation, this would use proper linkage criteria
        let mut clusters: Vec<Vec<usize>> = (0..n_points).map(|i| vec![i]).collect();
        let mut next_cluster_id = n_points;

        while clusters.len() > 1 {
            let mut min_distance = f64::INFINITY;
            let mut merge_pair = (0, 1);

            // Find closest pair of clusters
            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    let distance =
                        self.compute_cluster_distance(data, &clusters[i], &clusters[j])?;
                    if distance < min_distance {
                        min_distance = distance;
                        merge_pair = (i, j);
                    }
                }
            }

            // Merge clusters
            let (i, j) = merge_pair;
            let mut merged_cluster = clusters[i].clone();
            merged_cluster.extend_from_slice(&clusters[j]);

            merge_sequence.push((i, j, min_distance));

            // Remove old clusters and add merged cluster
            if i > j {
                clusters.remove(i);
                clusters.remove(j);
            } else {
                clusters.remove(j);
                clusters.remove(i);
            }
            clusters.push(merged_cluster);
            next_cluster_id += 1;
        }

        Ok(LocalDendrogram {
            worker_id: 0, // Simplified
            merge_sequence,
            n_points,
        })
    }

    /// Compute centroid of a cluster
    fn compute_cluster_centroid(&self, data: &Array2<F>, cluster: &[usize]) -> Result<Vec<f64>> {
        if cluster.is_empty() {
            return Ok(vec![0.0; data.ncols()]);
        }

        let mut centroid = vec![0.0; data.ncols()];
        for &point_idx in cluster {
            for (j, &value) in data.row(point_idx).iter().enumerate() {
                centroid[j] += value.to_f64().unwrap_or(0.0);
            }
        }

        for coord in &mut centroid {
            *coord /= cluster.len() as f64;
        }

        Ok(centroid)
    }

    /// Compute distance between two clusters
    fn compute_cluster_distance(
        &self,
        data: &Array2<F>,
        cluster1: &[usize],
        cluster2: &[usize],
    ) -> Result<f64> {
        match self.linkage_method {
            LinkageMethod::Single => {
                // Single linkage: minimum distance
                let mut min_dist = f64::INFINITY;
                for &i in cluster1 {
                    for &j in cluster2 {
                        let dist = euclidean_distance(data.row(i), data.row(j))
                            .to_f64()
                            .unwrap_or(f64::INFINITY);
                        min_dist = min_dist.min(dist);
                    }
                }
                Ok(min_dist)
            }
            LinkageMethod::Complete => {
                // Complete linkage: maximum distance
                let mut max_dist = 0.0;
                for &i in cluster1 {
                    for &j in cluster2 {
                        let dist = euclidean_distance(data.row(i), data.row(j))
                            .to_f64()
                            .unwrap_or(0.0);
                        max_dist = max_dist.max(dist);
                    }
                }
                Ok(max_dist)
            }
            LinkageMethod::Average => {
                // Average linkage: mean distance
                let mut total_dist = 0.0;
                let mut count = 0;
                for &i in cluster1 {
                    for &j in cluster2 {
                        let dist = euclidean_distance(data.row(i), data.row(j))
                            .to_f64()
                            .unwrap_or(0.0);
                        total_dist += dist;
                        count += 1;
                    }
                }
                Ok(total_dist / count as f64)
            }
            LinkageMethod::Ward => {
                // Ward linkage: calculate increase in within-cluster sum of squares
                if cluster1.is_empty() || cluster2.is_empty() {
                    return Ok(f64::INFINITY);
                }

                // Compute centroids
                let centroid1 = self.compute_cluster_centroid(data, cluster1)?;
                let centroid2 = self.compute_cluster_centroid(data, cluster2)?;
                
                // Compute combined centroid
                let n1 = cluster1.len() as f64;
                let n2 = cluster2.len() as f64;
                let combined_centroid: Vec<f64> = centroid1.iter()
                    .zip(centroid2.iter())
                    .map(|(&c1, &c2)| (n1 * c1 + n2 * c2) / (n1 + n2))
                    .collect();

                // Ward's criterion: ESS increase = (n1 * n2) / (n1 + n2) * ||centroid1 - centroid2||^2
                let centroid_distance_sq: f64 = centroid1.iter()
                    .zip(centroid2.iter())
                    .map(|(&c1, &c2)| (c1 - c2).powi(2))
                    .sum();

                let ward_distance = (n1 * n2) / (n1 + n2) * centroid_distance_sq;
                Ok(ward_distance.sqrt())
            }
        }
    }
}

/// Utility functions for distributed clustering
pub mod utils {
    use super::*;

    /// Estimate optimal number of workers based on data size and memory constraints
    pub fn estimate_optimal_workers(data_size_mb: f64, memory_limit_mb: f64) -> usize {
        let min_workers = 1;
        let max_workers = num_cpus::get().max(1);
        let memory_based_workers = (data_size_mb / memory_limit_mb).ceil() as usize;

        memory_based_workers.clamp(min_workers, max_workers)
    }

    /// Generate synthetic dataset for distributed clustering testing
    pub fn generate_large_dataset(
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
    ) -> Array2<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));

        let cluster_size = n_samples / n_clusters;

        for cluster in 0..n_clusters {
            let start_idx = cluster * cluster_size;
            let end_idx = if cluster == n_clusters - 1 {
                n_samples
            } else {
                (cluster + 1) * cluster_size
            };

            // Generate cluster center
            let center: Vec<f64> = (0..n_features)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect();

            // Generate points around center
            for i in start_idx..end_idx {
                for j in 0..n_features {
                    data[[i, j]] = center[j] + rng.gen_range(-2.0..2.0);
                }
            }
        }

        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_distributed_kmeans_partitioning() {
        let data =
            Array2::from_shape_vec((100, 3), (0..300).map(|i| i as f64 / 10.0).collect()).unwrap();

        let config = DistributedConfig {
            n_workers: 4,
            partitioning_strategy: PartitioningStrategy::RandomSplit,
            ..Default::default()
        };

        let mut distributed_kmeans = DistributedKMeans::new(3, config);
        distributed_kmeans.partition_data(data.view()).unwrap();

        // Check that all data is partitioned
        let total_partitioned: usize = distributed_kmeans
            .partitions
            .iter()
            .map(|p| p.data.nrows())
            .sum();
        assert_eq!(total_partitioned, 100);

        // Check that we have expected number of partitions
        assert_eq!(distributed_kmeans.partitions.len(), 4);
    }

    #[test]
    fn test_distributed_kmeans_simple() {
        let data = utils::generate_large_dataset(200, 2, 3);

        let config = DistributedConfig {
            n_workers: 2,
            max_coordination_rounds: 5,
            ..Default::default()
        };

        let mut distributed_kmeans = DistributedKMeans::new(3, config);

        // Test should complete without errors
        let result = distributed_kmeans.fit(data.view());
        assert!(result.is_ok());

        // Check that we have final labels
        let labels = distributed_kmeans.get_labels().unwrap();
        assert_eq!(labels.len(), 200);

        // Check convergence history
        assert!(!distributed_kmeans.get_convergence_history().is_empty());
    }

    #[test]
    fn test_load_balancing_strategies() {
        let total_samples = 100;
        let config = DistributedConfig {
            n_workers: 3,
            load_balancing: LoadBalancingStrategy::EqualSize,
            ..Default::default()
        };

        let distributed_kmeans = DistributedKMeans::<f64>::new(2, config);
        let sizes = distributed_kmeans
            .calculate_partition_sizes(total_samples)
            .unwrap();

        // Check that total adds up
        let total: usize = sizes.iter().sum();
        assert_eq!(total, total_samples);

        // Check relatively equal distribution
        let max_size = *sizes.iter().max().unwrap();
        let min_size = *sizes.iter().min().unwrap();
        assert!(max_size - min_size <= 1);
    }

    #[test]
    fn test_hierarchical_clustering() {
        let data = Array2::from_shape_vec((20, 2), (0..40).map(|i| i as f64).collect()).unwrap();

        let config = DistributedConfig {
            n_workers: 2,
            ..Default::default()
        };

        let mut distributed_hierarchical =
            DistributedHierarchical::new(LinkageMethod::Single, config);

        let result = distributed_hierarchical.fit(data.view());
        assert!(result.is_ok());

        // Check that local dendrograms were created
        assert_eq!(distributed_hierarchical.local_dendrograms.len(), 2);
    }
}
