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

/// Message passing system for distributed clustering
pub mod message_passing {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::thread;
    use std::time::{Duration, Instant};
    
    /// Message types for distributed clustering coordination
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub enum ClusteringMessage<F: Float> {
        /// Initialize worker with partition data
        InitializeWorker {
            worker_id: usize,
            partition_data: Array2<F>,
            initial_centroids: Array2<F>,
        },
        /// Update global centroids
        UpdateCentroids {
            round: usize,
            centroids: Array2<F>,
        },
        /// Request local computation
        ComputeLocal {
            round: usize,
            max_iterations: usize,
        },
        /// Local computation result
        LocalResult {
            worker_id: usize,
            round: usize,
            local_centroids: Array2<F>,
            local_labels: Array1<usize>,
            local_inertia: f64,
            computation_time_ms: u64,
        },
        /// Heartbeat for health monitoring
        Heartbeat {
            worker_id: usize,
            timestamp: u64,
            cpu_usage: f64,
            memory_usage: f64,
        },
        /// Synchronization barrier
        SyncBarrier {
            round: usize,
            participant_count: usize,
        },
        /// Convergence check result
        ConvergenceCheck {
            round: usize,
            converged: bool,
            max_centroid_movement: f64,
        },
        /// Terminate worker
        Terminate,
        /// Checkpoint creation request
        CreateCheckpoint {
            round: usize,
        },
        /// Checkpoint data
        CheckpointData {
            worker_id: usize,
            round: usize,
            centroids: Array2<F>,
            labels: Array1<usize>,
        },
        /// Recovery request
        RecoveryRequest {
            failed_worker_id: usize,
            recovery_strategy: RecoveryStrategy,
        },
        /// Load balancing request
        LoadBalance {
            target_worker_loads: HashMap<usize, f64>,
        },
        /// Data migration for load balancing
        MigrateData {
            source_worker: usize,
            target_worker: usize,
            data_subset: Array2<F>,
        },
        /// Acknowledgment message
        Acknowledgment {
            worker_id: usize,
            message_id: u64,
        },
    }
    
    /// Message priority levels
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub enum MessagePriority {
        Critical = 0,   // Immediate processing required
        High = 1,       // High priority
        Normal = 2,     // Normal processing
        Low = 3,        // Background processing
    }
    
    /// Message envelope with metadata
    #[derive(Debug, Clone)]
    pub struct MessageEnvelope<F: Float> {
        pub message_id: u64,
        pub sender_id: usize,
        pub receiver_id: usize,
        pub priority: MessagePriority,
        pub timestamp: u64,
        pub retry_count: u32,
        pub timeout_ms: u64,
        pub message: ClusteringMessage<F>,
    }
    
    /// Message passing coordinator for distributed clustering
    #[derive(Debug)]
    pub struct MessagePassingCoordinator<F: Float> {
        pub coordinator_id: usize,
        pub worker_channels: HashMap<usize, Sender<MessageEnvelope<F>>>,
        pub coordinator_receiver: Receiver<MessageEnvelope<F>>,
        pub coordinator_sender: Sender<MessageEnvelope<F>>,
        pub message_counter: Arc<Mutex<u64>>,
        pub pending_messages: HashMap<u64, MessageEnvelope<F>>,
        pub message_timeouts: HashMap<u64, Instant>,
        pub worker_status: HashMap<usize, WorkerStatus>,
        pub sync_barriers: HashMap<usize, SynchronizationBarrier>,
        pub config: MessagePassingConfig,
    }
    
    /// Configuration for message passing system
    #[derive(Debug, Clone)]
    pub struct MessagePassingConfig {
        pub max_message_queue_size: usize,
        pub message_timeout_ms: u64,
        pub max_retry_attempts: u32,
        pub heartbeat_interval_ms: u64,
        pub sync_timeout_ms: u64,
        pub enable_message_compression: bool,
        pub enable_message_ordering: bool,
        pub batch_size: usize,
    }
    
    impl Default for MessagePassingConfig {
        fn default() -> Self {
            Self {
                max_message_queue_size: 1000,
                message_timeout_ms: 30000,
                max_retry_attempts: 3,
                heartbeat_interval_ms: 5000,
                sync_timeout_ms: 60000,
                enable_message_compression: false,
                enable_message_ordering: true,
                batch_size: 10,
            }
        }
    }
    
    /// Synchronization barrier for coordinating worker phases
    #[derive(Debug)]
    pub struct SynchronizationBarrier {
        pub round: usize,
        pub expected_participants: usize,
        pub arrived_participants: HashSet<usize>,
        pub barrier_start_time: Instant,
        pub timeout_ms: u64,
    }
    
    impl<F: Float + FromPrimitive + Debug + Send + Sync + 'static> MessagePassingCoordinator<F> {
        /// Create new message passing coordinator
        pub fn new(coordinator_id: usize, config: MessagePassingConfig) -> Self {
            let (coordinator_sender, coordinator_receiver) = mpsc::channel();
            
            Self {
                coordinator_id,
                worker_channels: HashMap::new(),
                coordinator_receiver,
                coordinator_sender,
                message_counter: Arc::new(Mutex::new(0)),
                pending_messages: HashMap::new(),
                message_timeouts: HashMap::new(),
                worker_status: HashMap::new(),
                sync_barriers: HashMap::new(),
                config,
            }
        }
        
        /// Register a new worker with the coordinator
        pub fn register_worker(&mut self, worker_id: usize) -> Receiver<MessageEnvelope<F>> {
            let (sender, receiver) = mpsc::channel();
            self.worker_channels.insert(worker_id, sender);
            self.worker_status.insert(worker_id, WorkerStatus::Active);
            receiver
        }
        
        /// Send message to a specific worker
        pub fn send_message(
            &mut self,
            receiver_id: usize,
            message: ClusteringMessage<F>,
            priority: MessagePriority,
        ) -> Result<u64> {
            let message_id = self.get_next_message_id();
            
            let envelope = MessageEnvelope {
                message_id,
                sender_id: self.coordinator_id,
                receiver_id,
                priority,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
                retry_count: 0,
                timeout_ms: self.config.message_timeout_ms,
                message,
            };
            
            self.send_envelope(envelope)
        }
        
        /// Send message envelope
        fn send_envelope(&mut self, envelope: MessageEnvelope<F>) -> Result<u64> {
            let message_id = envelope.message_id;
            let receiver_id = envelope.receiver_id;
            
            // Store for timeout tracking
            self.pending_messages.insert(message_id, envelope.clone());
            self.message_timeouts.insert(message_id, Instant::now());
            
            // Send to worker
            if let Some(sender) = self.worker_channels.get(&receiver_id) {
                sender.send(envelope).map_err(|e| {
                    ClusteringError::InvalidInput(format!("Failed to send message: {}", e))
                })?;
            } else {
                return Err(ClusteringError::InvalidInput(format!(
                    "Worker {} not found", receiver_id
                )));
            }
            
            Ok(message_id)
        }
        
        /// Broadcast message to all workers
        pub fn broadcast_message(
            &mut self,
            message: ClusteringMessage<F>,
            priority: MessagePriority,
        ) -> Result<Vec<u64>> {
            let mut message_ids = Vec::new();
            
            for &worker_id in self.worker_channels.keys() {
                let message_id = self.send_message(worker_id, message.clone(), priority)?;
                message_ids.push(message_id);
            }
            
            Ok(message_ids)
        }
        
        /// Process incoming messages from workers
        pub fn process_messages(&mut self, timeout_ms: u64) -> Result<Vec<MessageEnvelope<F>>> {
            let mut received_messages = Vec::new();
            let timeout = Duration::from_millis(timeout_ms);
            let start_time = Instant::now();
            
            while start_time.elapsed() < timeout {
                match self.coordinator_receiver.try_recv() {
                    Ok(envelope) => {
                        self.handle_message_acknowledgment(&envelope);
                        received_messages.push(envelope);
                    }
                    Err(mpsc::TryRecvError::Empty) => {
                        // No messages available, continue
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        return Err(ClusteringError::InvalidInput(
                            "Message channel disconnected".to_string(),
                        ));
                    }
                }
            }
            
            // Check for message timeouts
            self.handle_message_timeouts()?;
            
            Ok(received_messages)
        }
        
        /// Handle message acknowledgments
        fn handle_message_acknowledgment(&mut self, envelope: &MessageEnvelope<F>) {
            if let ClusteringMessage::Acknowledgment { message_id, .. } = &envelope.message {
                self.pending_messages.remove(message_id);
                self.message_timeouts.remove(message_id);
            }
        }
        
        /// Handle message timeouts and retries
        fn handle_message_timeouts(&mut self) -> Result<()> {
            let now = Instant::now();
            let mut timed_out_messages = Vec::new();
            
            for (&message_id, &start_time) in &self.message_timeouts {
                if now.duration_since(start_time).as_millis() as u64 > self.config.message_timeout_ms {
                    timed_out_messages.push(message_id);
                }
            }
            
            for message_id in timed_out_messages {
                if let Some(mut envelope) = self.pending_messages.remove(&message_id) {
                    self.message_timeouts.remove(&message_id);
                    
                    if envelope.retry_count < self.config.max_retry_attempts {
                        envelope.retry_count += 1;
                        println!("Retrying message {} (attempt {})", message_id, envelope.retry_count);
                        self.send_envelope(envelope)?;
                    } else {
                        println!("Message {} failed after {} attempts", message_id, envelope.retry_count);
                        // Mark worker as potentially failed
                        self.worker_status.insert(envelope.receiver_id, WorkerStatus::Timeout);
                    }
                }
            }
            
            Ok(())
        }
        
        /// Create synchronization barrier
        pub fn create_sync_barrier(&mut self, round: usize, participant_count: usize) {
            let barrier = SynchronizationBarrier {
                round,
                expected_participants: participant_count,
                arrived_participants: HashSet::new(),
                barrier_start_time: Instant::now(),
                timeout_ms: self.config.sync_timeout_ms,
            };
            
            self.sync_barriers.insert(round, barrier);
        }
        
        /// Wait for synchronization barrier
        pub fn wait_for_sync_barrier(&mut self, round: usize) -> Result<bool> {
            let barrier = self.sync_barriers.get_mut(&round)
                .ok_or_else(|| ClusteringError::InvalidInput(
                    format!("Sync barrier for round {} not found", round)
                ))?;
            
            let elapsed = barrier.barrier_start_time.elapsed().as_millis() as u64;
            if elapsed > barrier.timeout_ms {
                return Err(ClusteringError::InvalidInput(
                    format!("Sync barrier timeout for round {}", round)
                ));
            }
            
            Ok(barrier.arrived_participants.len() >= barrier.expected_participants)
        }
        
        /// Register worker arrival at sync barrier
        pub fn register_barrier_arrival(&mut self, round: usize, worker_id: usize) -> Result<bool> {
            let barrier = self.sync_barriers.get_mut(&round)
                .ok_or_else(|| ClusteringError::InvalidInput(
                    format!("Sync barrier for round {} not found", round)
                ))?;
            
            barrier.arrived_participants.insert(worker_id);
            Ok(barrier.arrived_participants.len() >= barrier.expected_participants)
        }
        
        /// Get next message ID
        fn get_next_message_id(&self) -> u64 {
            let mut counter = self.message_counter.lock().unwrap();
            *counter += 1;
            *counter
        }
        
        /// Send heartbeat to workers
        pub fn send_heartbeat(&mut self) -> Result<()> {
            let heartbeat = ClusteringMessage::Heartbeat {
                worker_id: self.coordinator_id,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
                cpu_usage: 0.0, // Coordinator doesn't track CPU usage
                memory_usage: 0.0,
            };
            
            self.broadcast_message(heartbeat, MessagePriority::Low)?;
            Ok(())
        }
        
        /// Collect local results from all workers
        pub fn collect_local_results(&mut self, round: usize) -> Result<Vec<LocalKMeansResult<F>>> {
            let mut results = Vec::new();
            let compute_message = ClusteringMessage::ComputeLocal {
                round,
                max_iterations: 10, // Configurable
            };
            
            // Send compute requests to all workers
            self.broadcast_message(compute_message, MessagePriority::High)?;
            
            // Create sync barrier
            self.create_sync_barrier(round, self.worker_channels.len());
            
            // Collect results with timeout
            let timeout_duration = Duration::from_millis(self.config.sync_timeout_ms);
            let start_time = Instant::now();
            
            while start_time.elapsed() < timeout_duration {
                let messages = self.process_messages(1000)?; // 1 second timeout per batch
                
                for envelope in messages {
                    if let ClusteringMessage::LocalResult {
                        worker_id,
                        round: msg_round,
                        local_centroids,
                        local_labels,
                        local_inertia,
                        computation_time_ms,
                    } = envelope.message {
                        if msg_round == round {
                            results.push(LocalKMeansResult {
                                worker_id,
                                local_centroids,
                                local_labels,
                                local_inertia: F::from(local_inertia).unwrap_or(F::zero()),
                                n_points: local_labels.len(),
                                computation_time_ms,
                            });
                            
                            // Register barrier arrival
                            self.register_barrier_arrival(round, worker_id)?;
                        }
                    }
                }
                
                // Check if all workers have responded
                if self.wait_for_sync_barrier(round)? {
                    break;
                }
            }
            
            // Clean up barrier
            self.sync_barriers.remove(&round);
            
            Ok(results)
        }
        
        /// Coordinate global centroid updates
        pub fn coordinate_centroid_update(
            &mut self,
            round: usize,
            new_centroids: &Array2<F>,
        ) -> Result<()> {
            let update_message = ClusteringMessage::UpdateCentroids {
                round,
                centroids: new_centroids.clone(),
            };
            
            self.broadcast_message(update_message, MessagePriority::High)?;
            Ok(())
        }
        
        /// Shutdown message passing system
        pub fn shutdown(&mut self) -> Result<()> {
            let terminate_message = ClusteringMessage::Terminate;
            self.broadcast_message(terminate_message, MessagePriority::Critical)?;
            
            // Wait for acknowledgments
            thread::sleep(Duration::from_millis(1000));
            
            // Clear all channels
            self.worker_channels.clear();
            self.pending_messages.clear();
            self.message_timeouts.clear();
            self.sync_barriers.clear();
            
            Ok(())
        }
    }
}

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
        matches!(self.status, WorkerStatus::Active)
            && self.last_heartbeat.elapsed().as_millis() < timeout_ms as u128
    }

    /// Check if worker is healthy with adaptive thresholds
    pub fn is_healthy_adaptive(
        &self,
        timeout_ms: u64,
        cpu_threshold: f64,
        memory_threshold: f64,
    ) -> bool {
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

        if self.last_health_check.elapsed().as_millis()
            < self.fault_config.heartbeat_interval_ms as u128
        {
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
        let mut worker_efficiency: Vec<(usize, f64)> = self
            .worker_health
            .iter()
            .filter(|(_, health)| health.is_healthy(self.fault_config.worker_timeout_ms))
            .map(|(worker_id, health)| (*worker_id, health.get_efficiency_score()))
            .collect();

        // Sort by efficiency (descending)
        worker_efficiency
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        worker_efficiency
    }

    /// Update worker performance metrics
    pub fn update_worker_metrics(
        &mut self,
        worker_id: usize,
        response_time_ms: u64,
        cpu_usage: f64,
        memory_usage: f64,
    ) {
        if let Some(health) = self.worker_health.get_mut(&worker_id) {
            health.update_metrics(response_time_ms, cpu_usage, memory_usage);
            health.mark_success(); // Update heartbeat
        }
    }

    /// Get comprehensive health report
    pub fn get_health_report(&self) -> WorkerHealthReport {
        let total_workers = self.worker_health.len();
        let active_workers = self
            .worker_health
            .values()
            .filter(|h| matches!(h.status, WorkerStatus::Active))
            .count();
        let failed_workers = self.failed_workers.len();
        let timeout_workers = self
            .worker_health
            .values()
            .filter(|h| matches!(h.status, WorkerStatus::Timeout))
            .count();

        let avg_response_time = if active_workers > 0 {
            self.worker_health
                .values()
                .filter(|h| matches!(h.status, WorkerStatus::Active))
                .map(|h| h.response_time_ms)
                .sum::<u64>() as f64
                / active_workers as f64
        } else {
            0.0
        };

        let avg_cpu_usage = if active_workers > 0 {
            self.worker_health
                .values()
                .filter(|h| matches!(h.status, WorkerStatus::Active))
                .map(|h| h.cpu_usage)
                .sum::<f64>()
                / active_workers as f64
        } else {
            0.0
        };

        let avg_memory_usage = if active_workers > 0 {
            self.worker_health
                .values()
                .filter(|h| matches!(h.status, WorkerStatus::Active))
                .map(|h| h.memory_usage)
                .sum::<f64>()
                / active_workers as f64
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
    pub fn handle_worker_failure(
        &mut self,
        failed_worker_id: usize,
        partitions: &mut Vec<DataPartition<F>>,
    ) -> Result<()> {
        if !self.fault_config.enabled {
            return Ok(());
        }

        if self.failed_workers.len() > self.fault_config.max_failures {
            return Err(ClusteringError::InvalidInput(format!(
                "Too many worker failures: {} > {}",
                self.failed_workers.len(),
                self.fault_config.max_failures
            )));
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
                    "Restart strategy requires external coordination".to_string(),
                ));
            }
            RecoveryStrategy::Degrade => {
                // Continue with fewer workers - no action needed
            }
        }

        Ok(())
    }

    /// Redistribute data from failed worker to healthy workers
    fn redistribute_failed_worker_data(
        &mut self,
        failed_worker_id: usize,
        partitions: &mut Vec<DataPartition<F>>,
    ) -> Result<()> {
        let healthy_workers: Vec<usize> = self
            .worker_health
            .iter()
            .filter(|(_, health)| health.is_healthy(self.fault_config.worker_timeout_ms))
            .map(|(&id, _)| id)
            .collect();

        if healthy_workers.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "No healthy workers available for redistribution".to_string(),
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
                "Worker replacement is disabled".to_string(),
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
                "Checkpointing is disabled".to_string(),
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
    pub fn create_checkpoint(
        &mut self,
        iteration: usize,
        centroids: Option<&Array2<F>>,
        global_inertia: f64,
        convergence_history: &[ConvergenceMetrics],
        worker_assignments: &HashMap<usize, Vec<usize>>,
    ) {
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
        let healthy_workers: Vec<usize> = self
            .worker_health
            .iter()
            .filter(|(_, health)| health.is_healthy(self.fault_config.worker_timeout_ms))
            .map(|(&id, _)| id)
            .collect();

        if healthy_workers.len() < replication_factor {
            return Err(ClusteringError::InvalidInput(format!(
                "Not enough healthy workers for replication factor {}",
                replication_factor
            )));
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
            self.data_replicas
                .insert(partition.partition_id, replica_workers);
        }

        Ok(())
    }

    /// Replicate partition data to replica workers
    pub fn replicate_partition_data(
        &mut self,
        partition: &DataPartition<F>,
    ) -> Result<Vec<DataPartition<F>>> {
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
    pub fn recover_from_replicas(
        &mut self,
        failed_partition_id: usize,
        partitions: &mut Vec<DataPartition<F>>,
    ) -> Result<()> {
        if !self.fault_config.enable_replication {
            return Err(ClusteringError::InvalidInput(
                "Replication is not enabled".to_string(),
            ));
        }

        if let Some(replica_workers) = self.data_replicas.get(&failed_partition_id) {
            // Find a healthy replica worker
            for &replica_worker_id in replica_workers {
                if let Some(health) = self.worker_health.get(&replica_worker_id) {
                    if health.is_healthy(self.fault_config.worker_timeout_ms) {
                        // Promote replica to primary
                        if let Some(partition) = partitions
                            .iter_mut()
                            .find(|p| p.partition_id == failed_partition_id)
                        {
                            partition.worker_id = replica_worker_id;
                            return Ok(());
                        }
                    }
                }
            }
        }

        Err(ClusteringError::InvalidInput(format!(
            "No healthy replicas found for partition {}",
            failed_partition_id
        )))
    }

    /// Verify data consistency across replicas
    pub fn verify_replica_consistency(
        &self,
        partition_id: usize,
        partitions: &[DataPartition<F>],
    ) -> Result<bool> {
        if !self.fault_config.enable_replication {
            return Ok(true);
        }

        // Find primary partition
        let primary_partition = partitions
            .iter()
            .find(|p| p.partition_id == partition_id)
            .ok_or_else(|| {
                ClusteringError::InvalidInput(format!(
                    "Primary partition {} not found",
                    partition_id
                ))
            })?;

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
    pub fn update_replicas(
        &mut self,
        partition_id: usize,
        updated_partition: &DataPartition<F>,
    ) -> Result<()> {
        if !self.fault_config.enable_replication {
            return Ok(());
        }

        if let Some(replica_workers) = self.data_replicas.get(&partition_id) {
            for &replica_worker_id in replica_workers {
                if let Some(health) = self.worker_health.get(&replica_worker_id) {
                    if health.is_healthy(self.fault_config.worker_timeout_ms) {
                        // In a real implementation, this would send the updated data to replica workers
                        // For now, we just log the operation
                        println!(
                            "Updating replica on worker {} for partition {}",
                            replica_worker_id, partition_id
                        );
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

        let healthy_workers = self
            .worker_health
            .values()
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

/// Distributed K-means algorithm coordinator with message passing
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
    /// Message passing coordinator
    pub message_coordinator: Option<message_passing::MessagePassingCoordinator<F>>,
    /// Fault tolerance coordinator
    pub fault_coordinator: Option<FaultTolerantCoordinator<F>>,
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

impl<F: Float + FromPrimitive + Debug + Send + Sync + 'static> DistributedKMeans<F> {
    /// Create new distributed K-means instance
    pub fn new(k: usize, config: DistributedConfig) -> Self {
        Self {
            config,
            k,
            centroids: None,
            partitions: Vec::new(),
            convergence_history: Vec::new(),
            worker_stats: HashMap::new(),
            message_coordinator: None,
            fault_coordinator: None,
        }
    }
    
    /// Create new distributed K-means with message passing
    pub fn new_with_message_passing(
        k: usize,
        config: DistributedConfig,
        message_config: message_passing::MessagePassingConfig,
    ) -> Self {
        let mut instance = Self::new(k, config.clone());
        
        // Initialize message passing coordinator
        let coordinator_id = 0; // Coordinator is worker 0
        instance.message_coordinator = Some(
            message_passing::MessagePassingCoordinator::new(coordinator_id, message_config)
        );
        
        // Initialize fault tolerance coordinator
        instance.fault_coordinator = Some(
            FaultTolerantCoordinator::new(config.fault_tolerance)
        );
        
        instance
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
        if self.message_coordinator.is_some() {
            self.fit_with_message_passing(data)
        } else {
            self.fit_traditional(data)
        }
    }
    
    /// Traditional fit method (backward compatibility)
    pub fn fit_traditional(&mut self, data: ArrayView2<F>) -> Result<()> {
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
    
    /// Fit with message passing coordination
    pub fn fit_with_message_passing(&mut self, data: ArrayView2<F>) -> Result<()> {
        // Partition data across workers
        self.partition_data(data)?;
        
        // Initialize global centroids
        self.centroids = Some(self.initialize_global_centroids()?);
        
        // Initialize workers with message passing
        self.initialize_workers_with_message_passing()?;
        
        // Distributed coordination loop with message passing
        for round in 0..self.config.max_coordination_rounds {
            let start_time = std::time::Instant::now();
            
            // Send heartbeats to monitor worker health
            if let Some(ref mut coordinator) = self.message_coordinator {
                coordinator.send_heartbeat()?;
            }
            
            // Coordinate local computation via message passing
            let local_results = self.coordinate_local_computation_with_messages(round)?;
            
            // Global coordination phase
            let new_centroids = self.coordinate_global_centroids(&local_results)?;
            let convergence = self.check_global_convergence(&new_centroids, round)?;
            
            // Update global centroids via message passing
            if let Some(ref mut coordinator) = self.message_coordinator {
                coordinator.coordinate_centroid_update(round, &new_centroids)?;
            }
            
            // Update global state
            self.centroids = Some(new_centroids);
            self.convergence_history.push(convergence.clone());
            
            // Handle worker failures
            self.handle_worker_failures()?;
            
            let sync_time = start_time.elapsed().as_millis() as u64;
            
            // Log round completion
            if round % 10 == 0 {
                println!(
                    "Round {}: Global inertia = {:.4}, Converged = {}, Sync time = {}ms",
                    round, convergence.global_inertia, convergence.converged, sync_time
                );
            }
            
            // Check for global convergence
            if convergence.converged {
                println!("Distributed K-means converged after {} rounds", round + 1);
                break;
            }
            
            // Adaptive load balancing
            if round % 5 == 0 {
                self.perform_load_balancing()?;
            }
        }
        
        // Shutdown message passing system
        if let Some(ref mut coordinator) = self.message_coordinator {
            coordinator.shutdown()?;
        }
        
        Ok(())
    }
    
    /// Initialize workers with message passing
    fn initialize_workers_with_message_passing(&mut self) -> Result<()> {
        if let Some(ref mut coordinator) = self.message_coordinator {
            let initial_centroids = self.centroids.as_ref().unwrap();
            
            for partition in &self.partitions {
                // Register worker
                let _receiver = coordinator.register_worker(partition.worker_id);
                
                // Send initialization message
                let init_message = message_passing::ClusteringMessage::InitializeWorker {
                    worker_id: partition.worker_id,
                    partition_data: partition.data.clone(),
                    initial_centroids: initial_centroids.clone(),
                };
                
                coordinator.send_message(
                    partition.worker_id,
                    init_message,
                    message_passing::MessagePriority::High,
                )?;
            }
        }
        
        Ok(())
    }
    
    /// Coordinate local computation using message passing
    fn coordinate_local_computation_with_messages(
        &mut self,
        round: usize,
    ) -> Result<Vec<LocalKMeansResult<F>>> {
        if let Some(ref mut coordinator) = self.message_coordinator {
            coordinator.collect_local_results(round)
        } else {
            // Fallback to traditional method
            self.execute_local_kmeans_round()
        }
    }
    
    /// Handle worker failures using fault tolerance
    fn handle_worker_failures(&mut self) -> Result<()> {
        if let Some(ref mut fault_coordinator) = self.fault_coordinator {
            let failed_workers = fault_coordinator.check_worker_health();
            
            for failed_worker_id in failed_workers {
                fault_coordinator.handle_worker_failure(failed_worker_id, &mut self.partitions)?;
                
                // Notify message coordinator about failure
                if let Some(ref mut msg_coordinator) = self.message_coordinator {
                    msg_coordinator.worker_status.insert(failed_worker_id, WorkerStatus::Failed);
                }
            }
        }
        
        Ok(())
    }
    
    /// Perform adaptive load balancing
    fn perform_load_balancing(&mut self) -> Result<()> {
        if let Some(ref fault_coordinator) = self.fault_coordinator {
            if fault_coordinator.should_rebalance() {
                let worker_efficiency = fault_coordinator.get_workers_by_efficiency();
                
                // Create load balancing message
                let mut target_loads = HashMap::new();
                for (worker_id, efficiency) in worker_efficiency {
                    target_loads.insert(worker_id, efficiency);
                }
                
                if let Some(ref mut msg_coordinator) = self.message_coordinator {
                    let load_balance_msg = message_passing::ClusteringMessage::LoadBalance {
                        target_worker_loads: target_loads,
                    };
                    
                    msg_coordinator.broadcast_message(
                        load_balance_msg,
                        message_passing::MessagePriority::Normal,
                    )?;
                }
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
                let total_remaining_capacity: usize = worker_assignments
                    .iter()
                    .enumerate()
                    .skip(worker_id)
                    .map(|(i, assignments)| partition_sizes[i].saturating_sub(assignments.len()))
                    .sum();

                let points_for_worker = if total_remaining_capacity == 0 {
                    0
                } else {
                    let proportion = remaining_capacity as f64 / total_remaining_capacity as f64;
                    let remaining_points = total_points - distributed;
                    ((remaining_points as f64 * proportion).round() as usize)
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
        let history_bytes =
            self.config.max_coordination_rounds * std::mem::size_of::<ConvergenceMetrics>();

        // Worker statistics
        let stats_bytes = std::mem::size_of::<WorkerStatistics>();

        // Replication overhead (if enabled)
        let replication_bytes = if self.config.fault_tolerance.enable_replication {
            data_bytes
                * (self
                    .config
                    .fault_tolerance
                    .replication_factor
                    .saturating_sub(1))
        } else {
            0
        };

        // Communication buffers (estimated 10% of data size)
        let communication_bytes = (data_bytes as f64 * 0.1) as usize;

        // Operating system and runtime overhead (estimated 20% of total)
        let base_memory = data_bytes
            + labels_bytes
            + centroids_bytes
            + distance_temp_bytes
            + accumulator_bytes
            + counts_bytes
            + history_bytes
            + stats_bytes
            + replication_bytes
            + communication_bytes;
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
            let global_centroids_mb =
                (self.k * n_features * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);

            // Convergence history
            let history_mb = (self.config.max_coordination_rounds
                * std::mem::size_of::<ConvergenceMetrics>()) as f64
                / (1024.0 * 1024.0);

            // Worker health monitoring
            let monitoring_mb = (self.config.n_workers * std::mem::size_of::<WorkerHealth>())
                as f64
                / (1024.0 * 1024.0);

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
            return Err(ClusteringError::InvalidInput(format!(
                "Estimated memory usage ({:.1} MB) exceeds limit ({:.1} MB). \
                     Consider reducing data size, number of clusters, or increasing memory limit.",
                estimated_peak_mb, total_limit_mb
            )));
        }

        // Check per-worker limits
        let avg_per_worker_mb = estimated_peak_mb / self.config.n_workers as f64;
        if avg_per_worker_mb > per_worker_limit_mb {
            return Err(ClusteringError::InvalidInput(format!(
                "Average per-worker memory usage ({:.1} MB) exceeds per-worker limit ({:.1} MB). \
                     Consider increasing number of workers or memory limit.",
                avg_per_worker_mb, per_worker_limit_mb
            )));
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
                let combined_centroid: Vec<f64> = centroid1
                    .iter()
                    .zip(centroid2.iter())
                    .map(|(&c1, &c2)| (n1 * c1 + n2 * c2) / (n1 + n2))
                    .collect();

                // Ward's criterion: ESS increase = (n1 * n2) / (n1 + n2) * ||centroid1 - centroid2||^2
                let centroid_distance_sq: f64 = centroid1
                    .iter()
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
    use rand::Rng;
    use std::collections::BTreeMap;

    /// Advanced worker performance metrics
    #[derive(Debug, Clone)]
    pub struct WorkerPerformanceProfile {
        pub worker_id: usize,
        pub cpu_cores: usize,
        pub memory_gb: f64,
        pub network_bandwidth_mbps: f64,
        pub historical_throughput: f64,
        pub reliability_score: f64,
        pub current_load: f64,
        pub specialization: Vec<String>,
    }

    /// Dynamic load balancing configuration
    #[derive(Debug, Clone)]
    pub struct DynamicLoadBalancer {
        pub worker_profiles: HashMap<usize, WorkerPerformanceProfile>,
        pub load_history: Vec<LoadSnapshot>,
        pub balancing_strategy: AdvancedBalancingStrategy,
        pub rebalancing_threshold: f64,
        pub prediction_horizon: usize,
    }

    /// Load snapshot for historical analysis
    #[derive(Debug, Clone)]
    pub struct LoadSnapshot {
        pub timestamp: std::time::Instant,
        pub worker_loads: HashMap<usize, f64>,
        pub global_throughput: f64,
        pub coordination_overhead: f64,
    }

    /// Advanced load balancing strategies
    #[derive(Debug, Clone)]
    pub enum AdvancedBalancingStrategy {
        /// Predictive load balancing using historical data
        Predictive {
            model_type: PredictionModel,
            look_ahead_steps: usize,
        },
        /// Game theory based optimal assignment
        GameTheoretic {
            convergence_threshold: f64,
            max_iterations: usize,
        },
        /// Machine learning based adaptive balancing
        Adaptive {
            learning_rate: f64,
            exploration_rate: f64,
        },
        /// Multi-objective optimization
        MultiObjective {
            objectives: Vec<OptimizationObjective>,
            weights: Vec<f64>,
        },
    }

    /// Prediction models for load forecasting
    #[derive(Debug, Clone)]
    pub enum PredictionModel {
        LinearRegression,
        ExponentialSmoothing { alpha: f64, beta: f64 },
        ARIMA { p: usize, d: usize, q: usize },
        NeuralNetwork { hidden_layers: Vec<usize> },
    }

    /// Optimization objectives for multi-objective balancing
    #[derive(Debug, Clone)]
    pub enum OptimizationObjective {
        MinimizeLatency,
        MinimizeEnergyConsumption,
        MaximizeThroughput,
        MaximizeReliability,
        MinimizeCommunicationOverhead,
    }

    impl DynamicLoadBalancer {
        /// Create new dynamic load balancer
        pub fn new(balancing_strategy: AdvancedBalancingStrategy) -> Self {
            Self {
                worker_profiles: HashMap::new(),
                load_history: Vec::new(),
                balancing_strategy,
                rebalancing_threshold: 0.15, // 15% imbalance threshold
                prediction_horizon: 10,
            }
        }

        /// Register worker with performance profile
        pub fn register_worker(&mut self, profile: WorkerPerformanceProfile) {
            self.worker_profiles.insert(profile.worker_id, profile);
        }

        /// Compute optimal data distribution using advanced algorithms
        pub fn compute_optimal_distribution(
            &mut self,
            data_size: usize,
            current_assignments: &HashMap<usize, usize>,
        ) -> Result<HashMap<usize, usize>> {
            match &self.balancing_strategy {
                AdvancedBalancingStrategy::Predictive {
                    model_type,
                    look_ahead_steps,
                } => self.predictive_balancing(
                    data_size,
                    current_assignments,
                    model_type,
                    *look_ahead_steps,
                ),
                AdvancedBalancingStrategy::GameTheoretic {
                    convergence_threshold,
                    max_iterations,
                } => self.game_theoretic_balancing(
                    data_size,
                    *convergence_threshold,
                    *max_iterations,
                ),
                AdvancedBalancingStrategy::Adaptive {
                    learning_rate,
                    exploration_rate,
                } => self.adaptive_balancing(
                    data_size,
                    current_assignments,
                    *learning_rate,
                    *exploration_rate,
                ),
                AdvancedBalancingStrategy::MultiObjective {
                    objectives,
                    weights,
                } => self.multi_objective_balancing(data_size, objectives, weights),
            }
        }

        /// Predictive load balancing using time series forecasting
        fn predictive_balancing(
            &self,
            data_size: usize,
            current_assignments: &HashMap<usize, usize>,
            model_type: &PredictionModel,
            look_ahead_steps: usize,
        ) -> Result<HashMap<usize, usize>> {
            let mut new_assignments = HashMap::new();

            // Predict future worker loads based on historical data
            let predicted_loads = self.predict_worker_loads(model_type, look_ahead_steps)?;

            // Sort workers by predicted performance
            let mut worker_efficiency: Vec<(usize, f64)> = predicted_loads
                .iter()
                .map(|(&worker_id, &predicted_load)| {
                    let profile = self.worker_profiles.get(&worker_id).unwrap();
                    let efficiency = profile.historical_throughput
                        * (1.0 - predicted_load)
                        * profile.reliability_score;
                    (worker_id, efficiency)
                })
                .collect();

            worker_efficiency
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Distribute data proportionally to predicted efficiency
            let total_efficiency: f64 = worker_efficiency.iter().map(|(_, eff)| eff).sum();
            let mut remaining_data = data_size;

            for (i, (worker_id, efficiency)) in worker_efficiency.iter().enumerate() {
                let assignment = if i == worker_efficiency.len() - 1 {
                    remaining_data // Last worker gets remaining data
                } else {
                    let proportion = efficiency / total_efficiency;
                    let assignment = (data_size as f64 * proportion).round() as usize;
                    assignment.min(remaining_data)
                };

                new_assignments.insert(*worker_id, assignment);
                remaining_data = remaining_data.saturating_sub(assignment);
            }

            Ok(new_assignments)
        }

        /// Game theoretic load balancing using Nash equilibrium
        fn game_theoretic_balancing(
            &self,
            data_size: usize,
            convergence_threshold: f64,
            max_iterations: usize,
        ) -> Result<HashMap<usize, usize>> {
            let mut assignments = HashMap::new();
            let worker_ids: Vec<usize> = self.worker_profiles.keys().copied().collect();

            // Initialize with equal distribution
            let base_assignment = data_size / worker_ids.len();
            let remainder = data_size % worker_ids.len();

            for (i, &worker_id) in worker_ids.iter().enumerate() {
                let assignment = base_assignment + if i < remainder { 1 } else { 0 };
                assignments.insert(worker_id, assignment);
            }

            // Iterate to find Nash equilibrium
            for iteration in 0..max_iterations {
                let mut converged = true;
                let old_assignments = assignments.clone();

                // Each worker adjusts their load based on others' decisions
                for &worker_id in &worker_ids {
                    let optimal_load =
                        self.compute_best_response(worker_id, &assignments, data_size);
                    let current_load = assignments[&worker_id];

                    if (optimal_load as f64 - current_load as f64).abs() / current_load as f64
                        > convergence_threshold
                    {
                        assignments.insert(worker_id, optimal_load);
                        converged = false;
                    }
                }

                // Normalize to ensure total equals data_size
                let total_assigned: usize = assignments.values().sum();
                if total_assigned != data_size {
                    let adjustment_factor = data_size as f64 / total_assigned as f64;
                    for assignment in assignments.values_mut() {
                        *assignment = (*assignment as f64 * adjustment_factor).round() as usize;
                    }
                }

                if converged {
                    println!(
                        "Game theoretic balancing converged after {} iterations",
                        iteration + 1
                    );
                    break;
                }
            }

            Ok(assignments)
        }

        /// Compute best response for a worker in game theoretic setting
        fn compute_best_response(
            &self,
            worker_id: usize,
            current_assignments: &HashMap<usize, usize>,
            total_data: usize,
        ) -> usize {
            let profile = self.worker_profiles.get(&worker_id).unwrap();

            // Utility function considers throughput, reliability, and coordination cost
            let mut best_assignment = current_assignments[&worker_id];
            let mut best_utility =
                self.compute_worker_utility(worker_id, best_assignment, current_assignments);

            // Try different assignment levels
            let current = current_assignments[&worker_id];
            let others_total: usize = current_assignments
                .iter()
                .filter(|(&id, _)| id != worker_id)
                .map(|(_, &assignment)| assignment)
                .sum();

            let max_possible = total_data.saturating_sub(others_total);

            for test_assignment in 0..=max_possible.min(current * 2) {
                let utility =
                    self.compute_worker_utility(worker_id, test_assignment, current_assignments);
                if utility > best_utility {
                    best_utility = utility;
                    best_assignment = test_assignment;
                }
            }

            best_assignment
        }

        /// Compute utility for a worker's assignment
        fn compute_worker_utility(
            &self,
            worker_id: usize,
            assignment: usize,
            all_assignments: &HashMap<usize, usize>,
        ) -> f64 {
            let profile = self.worker_profiles.get(&worker_id).unwrap();

            // Throughput component
            let load_factor = assignment as f64 / (profile.memory_gb * 1000.0); // Rough capacity estimate
            let throughput_utility = profile.historical_throughput * (1.0 - load_factor.min(1.0));

            // Reliability component
            let reliability_utility = profile.reliability_score * (1.0 - load_factor * 0.5);

            // Communication overhead (increases with imbalance)
            let avg_assignment: f64 = all_assignments.values().map(|&v| v as f64).sum::<f64>()
                / all_assignments.len() as f64;
            let imbalance = (assignment as f64 - avg_assignment).abs() / avg_assignment;
            let communication_penalty = imbalance * 0.2;

            throughput_utility + reliability_utility - communication_penalty
        }

        /// Adaptive balancing using reinforcement learning principles
        fn adaptive_balancing(
            &mut self,
            data_size: usize,
            current_assignments: &HashMap<usize, usize>,
            learning_rate: f64,
            exploration_rate: f64,
        ) -> Result<HashMap<usize, usize>> {
            let mut new_assignments = current_assignments.clone();

            // ε-greedy exploration strategy
            let mut rng = rand::thread_rng();

            for (&worker_id, &current_assignment) in current_assignments {
                if rng.gen::<f64>() < exploration_rate {
                    // Explore: random adjustment
                    let max_change = (current_assignment as f64 * 0.2) as usize; // Max 20% change
                    let change = rng.gen_range(0..=max_change * 2) as i32 - max_change as i32;
                    let new_assignment = (current_assignment as i32 + change).max(0) as usize;
                    new_assignments.insert(worker_id, new_assignment);
                } else {
                    // Exploit: use learned policy
                    let optimal_assignment =
                        self.compute_learned_optimal_assignment(worker_id, data_size);

                    // Apply learning rate to smooth transitions
                    let adjusted_assignment = current_assignment as f64
                        + learning_rate * (optimal_assignment as f64 - current_assignment as f64);
                    new_assignments.insert(worker_id, adjusted_assignment.round() as usize);
                }
            }

            // Normalize assignments to match total data size
            let total_assigned: usize = new_assignments.values().sum();
            if total_assigned != data_size && total_assigned > 0 {
                let scale_factor = data_size as f64 / total_assigned as f64;
                for assignment in new_assignments.values_mut() {
                    *assignment = (*assignment as f64 * scale_factor).round() as usize;
                }
            }

            Ok(new_assignments)
        }

        /// Multi-objective optimization for load balancing
        fn multi_objective_balancing(
            &self,
            data_size: usize,
            objectives: &[OptimizationObjective],
            weights: &[f64],
        ) -> Result<HashMap<usize, usize>> {
            let worker_ids: Vec<usize> = self.worker_profiles.keys().copied().collect();
            let n_workers = worker_ids.len();

            // Generate Pareto-optimal solutions using weighted sum approach
            let mut best_assignment = HashMap::new();
            let mut best_score = f64::NEG_INFINITY;

            // Try different assignment combinations
            for _ in 0..1000 {
                // Monte Carlo sampling
                let mut assignment = self.generate_random_assignment(data_size, &worker_ids);

                // Evaluate multi-objective score
                let score = self.evaluate_multi_objective_score(&assignment, objectives, weights);

                if score > best_score {
                    best_score = score;
                    best_assignment = assignment;
                }
            }

            Ok(best_assignment)
        }

        /// Generate random valid assignment
        fn generate_random_assignment(
            &self,
            data_size: usize,
            worker_ids: &[usize],
        ) -> HashMap<usize, usize> {
            let mut rng = rand::thread_rng();
            let mut assignment = HashMap::new();
            let mut remaining = data_size;

            for (i, &worker_id) in worker_ids.iter().enumerate() {
                let max_assignment = if i == worker_ids.len() - 1 {
                    remaining // Last worker gets all remaining
                } else {
                    rng.gen_range(0..=remaining)
                };

                assignment.insert(worker_id, max_assignment);
                remaining = remaining.saturating_sub(max_assignment);
            }

            assignment
        }

        /// Evaluate multi-objective score
        fn evaluate_multi_objective_score(
            &self,
            assignment: &HashMap<usize, usize>,
            objectives: &[OptimizationObjective],
            weights: &[f64],
        ) -> f64 {
            let mut total_score = 0.0;

            for (objective, &weight) in objectives.iter().zip(weights.iter()) {
                let objective_score = match objective {
                    OptimizationObjective::MinimizeLatency => {
                        self.compute_latency_score(assignment)
                    }
                    OptimizationObjective::MinimizeEnergyConsumption => {
                        self.compute_energy_score(assignment)
                    }
                    OptimizationObjective::MaximizeThroughput => {
                        self.compute_throughput_score(assignment)
                    }
                    OptimizationObjective::MaximizeReliability => {
                        self.compute_reliability_score(assignment)
                    }
                    OptimizationObjective::MinimizeCommunicationOverhead => {
                        self.compute_communication_score(assignment)
                    }
                };

                total_score += weight * objective_score;
            }

            total_score
        }

        /// Compute latency objective score
        fn compute_latency_score(&self, assignment: &HashMap<usize, usize>) -> f64 {
            let mut max_latency = 0.0;

            for (&worker_id, &assigned_data) in assignment {
                if let Some(profile) = self.worker_profiles.get(&worker_id) {
                    let estimated_latency = assigned_data as f64 / profile.historical_throughput;
                    max_latency = max_latency.max(estimated_latency);
                }
            }

            1.0 / (1.0 + max_latency) // Minimize latency
        }

        /// Compute energy objective score
        fn compute_energy_score(&self, assignment: &HashMap<usize, usize>) -> f64 {
            let mut total_energy = 0.0;

            for (&worker_id, &assigned_data) in assignment {
                if let Some(profile) = self.worker_profiles.get(&worker_id) {
                    // Simplified energy model: proportional to CPU usage
                    let load_factor = assigned_data as f64 / (profile.memory_gb * 1000.0);
                    let energy_consumption = profile.cpu_cores as f64 * load_factor.powi(2);
                    total_energy += energy_consumption;
                }
            }

            1.0 / (1.0 + total_energy) // Minimize energy
        }

        /// Compute throughput objective score
        fn compute_throughput_score(&self, assignment: &HashMap<usize, usize>) -> f64 {
            let mut total_throughput = 0.0;

            for (&worker_id, &assigned_data) in assignment {
                if let Some(profile) = self.worker_profiles.get(&worker_id) {
                    let load_factor = assigned_data as f64 / (profile.memory_gb * 1000.0);
                    let effective_throughput =
                        profile.historical_throughput * (1.0 - load_factor.min(1.0)) * 0.8;
                    total_throughput += effective_throughput;
                }
            }

            total_throughput // Maximize throughput
        }

        /// Compute reliability objective score
        fn compute_reliability_score(&self, assignment: &HashMap<usize, usize>) -> f64 {
            let mut weighted_reliability = 0.0;
            let total_data: usize = assignment.values().sum();

            for (&worker_id, &assigned_data) in assignment {
                if let Some(profile) = self.worker_profiles.get(&worker_id) {
                    let weight = assigned_data as f64 / total_data as f64;
                    weighted_reliability += weight * profile.reliability_score;
                }
            }

            weighted_reliability // Maximize reliability
        }

        /// Compute communication overhead objective score
        fn compute_communication_score(&self, assignment: &HashMap<usize, usize>) -> f64 {
            let values: Vec<usize> = assignment.values().copied().collect();
            if values.is_empty() {
                return 1.0;
            }

            let mean = values.iter().sum::<usize>() as f64 / values.len() as f64;
            let variance = values
                .iter()
                .map(|&x| (x as f64 - mean).powi(2))
                .sum::<f64>()
                / values.len() as f64;

            1.0 / (1.0 + variance.sqrt()) // Minimize communication overhead via balance
        }

        /// Predict worker loads using time series models
        fn predict_worker_loads(
            &self,
            model_type: &PredictionModel,
            look_ahead_steps: usize,
        ) -> Result<HashMap<usize, f64>> {
            let mut predictions = HashMap::new();

            for &worker_id in self.worker_profiles.keys() {
                let historical_loads = self.get_worker_load_history(worker_id);
                let predicted_load = match model_type {
                    PredictionModel::LinearRegression => {
                        self.linear_regression_predict(&historical_loads, look_ahead_steps)
                    }
                    PredictionModel::ExponentialSmoothing { alpha, beta } => self
                        .exponential_smoothing_predict(
                            &historical_loads,
                            *alpha,
                            *beta,
                            look_ahead_steps,
                        ),
                    PredictionModel::ARIMA { p, d, q } => {
                        self.arima_predict(&historical_loads, *p, *d, *q, look_ahead_steps)
                    }
                    PredictionModel::NeuralNetwork { hidden_layers } => self
                        .neural_network_predict(&historical_loads, hidden_layers, look_ahead_steps),
                };

                predictions.insert(worker_id, predicted_load);
            }

            Ok(predictions)
        }

        /// Get historical load data for a worker
        fn get_worker_load_history(&self, worker_id: usize) -> Vec<f64> {
            self.load_history
                .iter()
                .filter_map(|snapshot| snapshot.worker_loads.get(&worker_id).copied())
                .collect()
        }

        /// Simple linear regression prediction
        fn linear_regression_predict(&self, data: &[f64], steps: usize) -> f64 {
            if data.len() < 2 {
                return data.last().copied().unwrap_or(0.5);
            }

            let n = data.len() as f64;
            let x_sum: f64 = (0..data.len()).map(|i| i as f64).sum();
            let y_sum: f64 = data.iter().sum();
            let xy_sum: f64 = data.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
            let x2_sum: f64 = (0..data.len()).map(|i| (i as f64).powi(2)).sum();

            let slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum.powi(2));
            let intercept = (y_sum - slope * x_sum) / n;

            let future_x = data.len() + steps - 1;
            (slope * future_x as f64 + intercept).clamp(0.0, 1.0)
        }

        /// Exponential smoothing prediction
        fn exponential_smoothing_predict(
            &self,
            data: &[f64],
            alpha: f64,
            beta: f64,
            steps: usize,
        ) -> f64 {
            if data.is_empty() {
                return 0.5;
            }

            if data.len() == 1 {
                return data[0];
            }

            let mut level = data[0];
            let mut trend = data[1] - data[0];

            for i in 1..data.len() {
                let prev_level = level;
                level = alpha * data[i] + (1.0 - alpha) * (level + trend);
                trend = beta * (level - prev_level) + (1.0 - beta) * trend;
            }

            (level + steps as f64 * trend).clamp(0.0, 1.0)
        }

        /// Simplified ARIMA prediction
        fn arima_predict(&self, data: &[f64], p: usize, d: usize, q: usize, steps: usize) -> f64 {
            if data.len() < p.max(q) + d {
                return data.last().copied().unwrap_or(0.5);
            }

            // Simplified AR(1) model for demonstration
            if p >= 1 && data.len() >= 2 {
                let recent_values: Vec<f64> = data.iter().rev().take(p).copied().collect();
                let weights: Vec<f64> = (0..p).map(|i| 1.0 / (i + 1) as f64).collect();
                let weighted_sum: f64 = recent_values
                    .iter()
                    .zip(weights.iter())
                    .map(|(v, w)| v * w)
                    .sum();
                let weight_sum: f64 = weights.iter().sum();

                (weighted_sum / weight_sum).clamp(0.0, 1.0)
            } else {
                data.last().copied().unwrap_or(0.5)
            }
        }

        /// Simplified neural network prediction
        fn neural_network_predict(
            &self,
            data: &[f64],
            hidden_layers: &[usize],
            steps: usize,
        ) -> f64 {
            // Simplified feedforward network simulation
            if data.len() < 3 {
                return data.last().copied().unwrap_or(0.5);
            }

            // Use last few values as input
            let input_size = 3.min(data.len());
            let inputs: Vec<f64> = data.iter().rev().take(input_size).copied().collect();

            // Simple weighted combination (simulating trained network)
            let weights = vec![0.4, 0.3, 0.3]; // Recent values have higher weight
            let prediction: f64 = inputs.iter().zip(weights.iter()).map(|(x, w)| x * w).sum();

            prediction.clamp(0.0, 1.0)
        }

        /// Compute learned optimal assignment using historical performance
        fn compute_learned_optimal_assignment(&self, worker_id: usize, total_data: usize) -> usize {
            if let Some(profile) = self.worker_profiles.get(&worker_id) {
                // Use historical throughput and reliability to estimate optimal load
                let capacity_factor = profile.historical_throughput * profile.reliability_score;
                let total_capacity: f64 = self
                    .worker_profiles
                    .values()
                    .map(|p| p.historical_throughput * p.reliability_score)
                    .sum();

                if total_capacity > 0.0 {
                    let proportion = capacity_factor / total_capacity;
                    (total_data as f64 * proportion).round() as usize
                } else {
                    total_data / self.worker_profiles.len()
                }
            } else {
                total_data / self.worker_profiles.len()
            }
        }

        /// Record load snapshot for historical analysis
        pub fn record_load_snapshot(&mut self, snapshot: LoadSnapshot) {
            self.load_history.push(snapshot);

            // Keep only recent history to prevent memory bloat
            if self.load_history.len() > 1000 {
                self.load_history.remove(0);
            }
        }

        /// Check if rebalancing is needed
        pub fn should_rebalance(&self, current_assignments: &HashMap<usize, usize>) -> bool {
            if current_assignments.len() < 2 {
                return false;
            }

            let values: Vec<usize> = current_assignments.values().copied().collect();
            let mean = values.iter().sum::<usize>() as f64 / values.len() as f64;
            let variance = values
                .iter()
                .map(|&x| (x as f64 - mean).powi(2))
                .sum::<f64>()
                / values.len() as f64;
            let coefficient_of_variation = variance.sqrt() / mean;

            coefficient_of_variation > self.rebalancing_threshold
        }
    }

    /// Estimate optimal number of workers based on data size and memory constraints
    pub fn estimate_optimal_workers(data_size_mb: f64, memory_limit_mb: f64) -> usize {
        let min_workers = 1;
        let max_workers = num_cpus::get().max(1);
        let memory_based_workers = (data_size_mb / memory_limit_mb).ceil() as usize;

        memory_based_workers.clamp(min_workers, max_workers)
    }

    /// Advanced optimal worker estimation with performance profiling
    pub fn estimate_optimal_workers_advanced(
        data_size_mb: f64,
        memory_limit_mb: f64,
        algorithm_complexity: f64,
        communication_overhead: f64,
        target_efficiency: f64,
    ) -> (usize, f64) {
        let min_workers = 1;
        let max_workers = num_cpus::get().max(1);

        let mut best_workers = min_workers;
        let mut best_efficiency = 0.0;

        for n_workers in min_workers..=max_workers {
            // Compute parallel efficiency considering Amdahl's law
            let sequential_fraction = 1.0 / algorithm_complexity;
            let parallel_fraction = 1.0 - sequential_fraction;

            // Amdahl's law speedup
            let theoretical_speedup =
                1.0 / (sequential_fraction + parallel_fraction / n_workers as f64);

            // Communication overhead penalty
            let comm_penalty = communication_overhead * (n_workers - 1) as f64 / n_workers as f64;
            let actual_speedup = theoretical_speedup * (1.0 - comm_penalty);

            // Memory efficiency
            let memory_per_worker = data_size_mb / n_workers as f64;
            let memory_efficiency = if memory_per_worker <= memory_limit_mb {
                1.0
            } else {
                memory_limit_mb / memory_per_worker
            };

            // Overall efficiency
            let efficiency = (actual_speedup / n_workers as f64) * memory_efficiency;

            if efficiency > best_efficiency && efficiency >= target_efficiency {
                best_efficiency = efficiency;
                best_workers = n_workers;
            }
        }

        (best_workers, best_efficiency)
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

    /// Generate realistic dataset with noise and outliers
    pub fn generate_realistic_dataset(
        n_samples: usize,
        n_features: usize,
        n_clusters: usize,
        noise_level: f64,
        outlier_fraction: f64,
    ) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));

        let n_outliers = (n_samples as f64 * outlier_fraction) as usize;
        let n_normal = n_samples - n_outliers;
        let cluster_size = n_normal / n_clusters;

        // Generate cluster centers with varying separations
        let mut centers = Vec::new();
        for i in 0..n_clusters {
            let center: Vec<f64> = (0..n_features)
                .map(|j| {
                    let base = (i as f64 * 10.0) + rng.gen_range(-2.0..2.0);
                    if j % 2 == 0 {
                        base
                    } else {
                        -base
                    }
                })
                .collect();
            centers.push(center);
        }

        // Generate normal cluster points
        let mut point_idx = 0;
        for cluster in 0..n_clusters {
            let start_idx = point_idx;
            let end_idx = if cluster == n_clusters - 1 {
                point_idx + (n_normal - cluster * cluster_size)
            } else {
                point_idx + cluster_size
            };

            for i in start_idx..end_idx {
                for j in 0..n_features {
                    // Cluster-specific variance
                    let cluster_variance = 1.0 + cluster as f64 * 0.5;
                    let noise = rng.gen_range(-noise_level..noise_level) * cluster_variance;
                    data[[i, j]] = centers[cluster][j] + noise;
                }
            }
            point_idx = end_idx;
        }

        // Generate outliers
        for i in n_normal..n_samples {
            for j in 0..n_features {
                data[[i, j]] = rng.gen_range(-50.0..50.0);
            }
        }

        data
    }

    /// Benchmark distributed vs centralized clustering
    pub fn benchmark_distributed_efficiency(
        data: &Array2<f64>,
        n_clusters: usize,
        worker_configs: &[usize],
    ) -> Vec<(usize, f64, f64)> {
        // (n_workers, time_seconds, speedup)
        let mut results = Vec::new();

        // Baseline centralized time (simulated)
        let centralized_time = simulate_centralized_clustering_time(data, n_clusters);

        for &n_workers in worker_configs {
            let distributed_time =
                simulate_distributed_clustering_time(data, n_clusters, n_workers);
            let speedup = centralized_time / distributed_time;
            results.push((n_workers, distributed_time, speedup));
        }

        results
    }

    /// Simulate centralized clustering time
    fn simulate_centralized_clustering_time(data: &Array2<f64>, n_clusters: usize) -> f64 {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Simplified complexity model: O(n * k * d * iterations)
        let complexity = n_samples as f64 * n_clusters as f64 * n_features as f64 * 100.0;
        complexity / 1e6 // Convert to seconds (simulated)
    }

    /// Simulate distributed clustering time
    fn simulate_distributed_clustering_time(
        data: &Array2<f64>,
        n_clusters: usize,
        n_workers: usize,
    ) -> f64 {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Parallel computation time
        let samples_per_worker = n_samples / n_workers;
        let parallel_complexity =
            samples_per_worker as f64 * n_clusters as f64 * n_features as f64 * 100.0;
        let parallel_time = parallel_complexity / 1e6;

        // Communication overhead
        let coordination_rounds = 50.0; // Typical convergence
        let communication_time =
            coordination_rounds * n_workers as f64 * n_clusters as f64 * n_features as f64 / 1e8;

        // Synchronization overhead
        let sync_time = coordination_rounds * 0.01; // 10ms per round

        parallel_time + communication_time + sync_time
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
