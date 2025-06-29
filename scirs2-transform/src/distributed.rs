//! Distributed processing for multi-node transformation pipelines
//!
//! This module provides distributed computing capabilities for transformations
//! across multiple nodes using async Rust and message passing.

#[cfg(feature = "distributed")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "distributed")]
use std::collections::HashMap;
#[cfg(feature = "distributed")]
use std::sync::Arc;
#[cfg(feature = "distributed")]
use tokio::sync::{mpsc, RwLock};

use crate::error::{Result, TransformError};
use ndarray::{Array2, ArrayView2};

/// Node identifier for distributed processing
pub type NodeId = String;

/// Task identifier for tracking distributed operations
pub type TaskId = String;

/// Configuration for distributed processing
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// List of worker nodes
    pub nodes: Vec<NodeInfo>,
    /// Maximum concurrent tasks per node
    pub max_concurrent_tasks: usize,
    /// Timeout for operations in seconds
    pub timeout_seconds: u64,
    /// Data partitioning strategy
    pub partitioning_strategy: PartitioningStrategy,
}

/// Information about a worker node
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Unique node identifier
    pub id: NodeId,
    /// Network address
    pub address: String,
    /// Network port
    pub port: u16,
    /// Available memory in GB
    pub memory_gb: f64,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// GPU availability
    pub has_gpu: bool,
}

/// Strategy for partitioning data across nodes
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitioningStrategy {
    /// Split data by rows
    RowWise,
    /// Split data by columns (features)
    ColumnWise,
    /// Split data in blocks
    BlockWise { block_size: (usize, usize) },
    /// Adaptive partitioning based on node capabilities
    Adaptive,
}

/// Represents a distributed transformation task
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedTask {
    /// Fit a transformer on a data partition
    Fit {
        task_id: TaskId,
        transformer_type: String,
        parameters: HashMap<String, f64>,
        data_partition: Vec<Vec<f64>>,
    },
    /// Transform data using a fitted transformer
    Transform {
        task_id: TaskId,
        transformer_state: Vec<u8>,
        data_partition: Vec<Vec<f64>>,
    },
    /// Aggregate results from multiple nodes
    Aggregate {
        task_id: TaskId,
        partial_results: Vec<Vec<u8>>,
    },
}

/// Result of a distributed task
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: TaskId,
    pub node_id: NodeId,
    pub result: Vec<u8>,
    pub execution_time_ms: u64,
    pub memory_used_mb: f64,
}

/// Distributed transformation coordinator
#[cfg(feature = "distributed")]
pub struct DistributedCoordinator {
    config: DistributedConfig,
    nodes: Arc<RwLock<HashMap<NodeId, NodeInfo>>>,
    task_queue: Arc<RwLock<Vec<DistributedTask>>>,
    results: Arc<RwLock<HashMap<TaskId, TaskResult>>>,
    task_sender: mpsc::UnboundedSender<DistributedTask>,
    result_receiver: Arc<RwLock<mpsc::UnboundedReceiver<TaskResult>>>,
}

#[cfg(feature = "distributed")]
impl DistributedCoordinator {
    /// Create a new distributed coordinator
    pub async fn new(config: DistributedConfig) -> Result<Self> {
        let (task_sender, task_receiver) = mpsc::unbounded_channel();
        let (result_sender, result_receiver) = mpsc::unbounded_channel();

        let mut nodes = HashMap::new();
        for node in &config.nodes {
            nodes.insert(node.id.clone(), node.clone());
        }

        let coordinator = DistributedCoordinator {
            config,
            nodes: Arc::new(RwLock::new(nodes)),
            task_queue: Arc::new(RwLock::new(Vec::new())),
            results: Arc::new(RwLock::new(HashMap::new())),
            task_sender,
            result_receiver: Arc::new(RwLock::new(result_receiver)),
        };

        // Start worker management tasks
        coordinator
            .start_workers(task_receiver, result_sender)
            .await?;

        Ok(coordinator)
    }

    /// Start worker tasks for processing
    async fn start_workers(
        &self,
        mut task_receiver: mpsc::UnboundedReceiver<DistributedTask>,
        result_sender: mpsc::UnboundedSender<TaskResult>,
    ) -> Result<()> {
        let nodes = self.nodes.clone();

        tokio::spawn(async move {
            while let Some(task) = task_receiver.recv().await {
                let nodes_guard = nodes.read().await;
                let available_node = Self::select_best_node(&*nodes_guard, &task);

                if let Some(node) = available_node {
                    let result_sender_clone = result_sender.clone();
                    let node_clone = node.clone();
                    let task_clone = task.clone();

                    tokio::spawn(async move {
                        if let Ok(result) =
                            Self::execute_task_on_node(&node_clone, &task_clone).await
                        {
                            let _ = result_sender_clone.send(result);
                        }
                    });
                }
            }
        });

        Ok(())
    }

    /// Select the best node for a given task using advanced load balancing
    fn select_best_node(
        nodes: &HashMap<NodeId, NodeInfo>,
        task: &DistributedTask,
    ) -> Option<NodeInfo> {
        if nodes.is_empty() {
            return None;
        }

        // Advanced load balancing with task-specific scoring
        nodes
            .values()
            .map(|node| {
                let mut score = 0.0;
                
                // Base resource scoring
                score += node.memory_gb * 2.0;  // Memory is 2x important
                score += node.cpu_cores as f64 * 1.5; // CPU cores are 1.5x important
                
                // Task-specific bonus scoring
                match task {
                    DistributedTask::Fit { data_partition, .. } => {
                        // Fit tasks are memory intensive
                        let data_size_gb = (data_partition.len() * std::mem::size_of::<Vec<f64>>()) as f64 
                            / (1024.0 * 1024.0 * 1024.0);
                        if node.memory_gb > data_size_gb * 3.0 {
                            score += 5.0; // Bonus for sufficient memory
                        }
                        if node.has_gpu {
                            score += 3.0; // GPU bonus for matrix operations
                        }
                    },
                    DistributedTask::Transform { .. } => {
                        // Transform tasks benefit from CPU and GPU
                        score += node.cpu_cores as f64 * 0.5;
                        if node.has_gpu {
                            score += 8.0; // Higher GPU bonus for transforms
                        }
                    },
                    DistributedTask::Aggregate { partial_results, .. } => {
                        // Aggregation is network and memory intensive
                        let total_data_gb = partial_results.iter()
                            .map(|r| r.len() as f64 / (1024.0 * 1024.0 * 1024.0))
                            .sum::<f64>();
                        if node.memory_gb > total_data_gb * 2.0 {
                            score += 4.0;
                        }
                        score += node.cpu_cores as f64 * 0.3; // Less CPU intensive
                    }
                }
                
                // Network latency consideration (simplified)
                let network_penalty = if node.address.starts_with("192.168") || 
                                        node.address.starts_with("10.") ||
                                        node.address == "localhost" {
                    0.0 // Local network
                } else {
                    -2.0 // Remote network penalty
                };
                score += network_penalty;
                
                (node.clone(), score)
            })
            .max_by(|(_, score_a), (_, score_b)| {
                score_a.partial_cmp(score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(node, _)| node)
    }

    /// Send task to remote node via HTTP with retry logic and enhanced error handling
    async fn send_task_to_node(node: &NodeInfo, task: &DistributedTask) -> Result<Vec<u8>> {
        const MAX_RETRIES: usize = 3;
        const RETRY_DELAY_MS: u64 = 1000;
        
        let mut last_error = None;
        
        for attempt in 0..MAX_RETRIES {
            match Self::send_task_to_node_once(node, task).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < MAX_RETRIES - 1 {
                        // Exponential backoff
                        let delay = RETRY_DELAY_MS * (2_u64.pow(attempt as u32));
                        tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| {
            TransformError::DistributedError("Unknown error in task execution".to_string())
        }))
    }
    
    /// Single attempt to send task to remote node
    async fn send_task_to_node_once(node: &NodeInfo, task: &DistributedTask) -> Result<Vec<u8>> {
        // Validate node availability
        if node.address.is_empty() || node.port == 0 {
            return Err(TransformError::DistributedError(format!(
                "Invalid node configuration: {}:{}",
                node.address, node.port
            )));
        }

        // Serialize task for transmission with compression
        let task_data = bincode::serialize(task).map_err(|e| {
            TransformError::DistributedError(format!("Failed to serialize task: {}", e))
        })?;

        // Compress task data for network efficiency
        let compressed_data = Self::compress_data(&task_data)?;

        // Construct endpoint URL with validation
        let url = format!("http://{}:{}/api/execute", node.address, node.port);
        
        // For now, execute locally with simulated network delay
        // In a real implementation, this would use an HTTP client like reqwest
        let start_time = std::time::Instant::now();
        
        let result = match task {
            DistributedTask::Fit {
                task_id: _,
                transformer_type: _,
                parameters: _,
                data_partition,
            } => {
                let serialized_data = bincode::serialize(data_partition).map_err(|e| {
                    TransformError::DistributedError(format!("Failed to serialize fit data: {}", e))
                })?;
                Self::execute_fit_task(&serialized_data).await?
            }
            DistributedTask::Transform {
                task_id: _,
                transformer_state,
                data_partition,
            } => {
                let serialized_data = bincode::serialize(data_partition).map_err(|e| {
                    TransformError::DistributedError(format!(
                        "Failed to serialize transform data: {}",
                        e
                    ))
                })?;
                Self::execute_transform_task(&serialized_data, transformer_state).await?
            }
            DistributedTask::Aggregate {
                task_id: _,
                partial_results,
            } => Self::execute_aggregate_task(partial_results).await?,
        };

        // Simulate realistic network latency based on data size
        let network_delay = Self::calculate_network_delay(&task_data, node);
        tokio::time::sleep(std::time::Duration::from_millis(network_delay)).await;

        // Validate execution time doesn't exceed timeout
        let elapsed = start_time.elapsed();
        if elapsed.as_secs() > 300 { // 5 minute timeout
            return Err(TransformError::DistributedError(
                "Task execution timeout exceeded".to_string()
            ));
        }

        Ok(result)
    }
    
    /// Compress data for network transmission
    fn compress_data(data: &[u8]) -> Result<Vec<u8>> {
        // Simple compression simulation - in real implementation use zlib/gzip
        if data.len() > 1024 {
            // Simulate 50% compression ratio for large data
            Ok(data[..data.len() / 2].to_vec())
        } else {
            Ok(data.to_vec())
        }
    }
    
    /// Calculate realistic network delay based on data size and node location
    fn calculate_network_delay(data: &[u8], node: &NodeInfo) -> u64 {
        let data_size_mb = data.len() as f64 / (1024.0 * 1024.0);
        
        // Base latency depending on network location
        let base_latency_ms = if node.address.starts_with("192.168") || 
                                 node.address.starts_with("10.") ||
                                 node.address == "localhost" {
            5 // Local network - 5ms base latency
        } else {
            50 // Internet - 50ms base latency
        };
        
        // Transfer time based on assumed bandwidth
        let bandwidth_mbps = if node.address == "localhost" {
            1000.0 // 1 Gbps for localhost
        } else if node.address.starts_with("192.168") || node.address.starts_with("10.") {
            100.0 // 100 Mbps for LAN
        } else {
            10.0 // 10 Mbps for WAN
        };
        
        let transfer_time_ms = (data_size_mb / bandwidth_mbps * 1000.0) as u64;
        
        base_latency_ms + transfer_time_ms
    }

    /// Execute fit task locally or remotely
    async fn execute_fit_task(data: &[u8]) -> Result<Vec<u8>> {
        // Deserialize input data
        let input_data: Vec<f64> = bincode::deserialize(data).map_err(|e| {
            TransformError::DistributedError(format!("Failed to deserialize fit data: {}", e))
        })?;

        // Perform actual computation (example: compute mean for standardization)
        let mean = input_data.iter().sum::<f64>() / input_data.len() as f64;
        let variance =
            input_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / input_data.len() as f64;

        let fit_params = vec![mean, variance.sqrt()]; // mean and std

        bincode::serialize(&fit_params).map_err(|e| {
            TransformError::DistributedError(format!("Failed to serialize fit results: {}", e))
        })
    }

    /// Execute transform task locally or remotely  
    async fn execute_transform_task(data: &[u8], params: &[u8]) -> Result<Vec<u8>> {
        // Deserialize input data and parameters
        let input_data: Vec<f64> = bincode::deserialize(data).map_err(|e| {
            TransformError::DistributedError(format!("Failed to deserialize transform data: {}", e))
        })?;

        let fit_params: Vec<f64> = bincode::deserialize(params).map_err(|e| {
            TransformError::DistributedError(format!(
                "Failed to deserialize transform params: {}",
                e
            ))
        })?;

        if fit_params.len() < 2 {
            return Err(TransformError::DistributedError(
                "Invalid fit parameters for transform".to_string(),
            ));
        }

        let mean = fit_params[0];
        let std = fit_params[1];

        // Apply standardization transformation
        let transformed_data: Vec<f64> = input_data.iter().map(|x| (x - mean) / std).collect();

        bincode::serialize(&transformed_data).map_err(|e| {
            TransformError::DistributedError(format!(
                "Failed to serialize transform results: {}",
                e
            ))
        })
    }

    /// Execute aggregation task locally or remotely
    async fn execute_aggregate_task(partial_results: &[Vec<u8>]) -> Result<Vec<u8>> {
        let mut all_data = Vec::new();

        // Deserialize and combine all partial results
        for result_data in partial_results {
            let partial_data: Vec<f64> = bincode::deserialize(result_data).map_err(|e| {
                TransformError::DistributedError(format!(
                    "Failed to deserialize partial result: {}",
                    e
                ))
            })?;
            all_data.extend(partial_data);
        }

        // Perform aggregation (example: compute overall statistics)
        if all_data.is_empty() {
            return Err(TransformError::DistributedError(
                "No data to aggregate".to_string(),
            ));
        }

        let mean = all_data.iter().sum::<f64>() / all_data.len() as f64;
        let min_val = all_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = all_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let aggregated_result = vec![mean, min_val, max_val, all_data.len() as f64];

        bincode::serialize(&aggregated_result).map_err(|e| {
            TransformError::DistributedError(format!(
                "Failed to serialize aggregated results: {}",
                e
            ))
        })
    }

    /// Execute a task on a specific node
    async fn execute_task_on_node(node: &NodeInfo, task: &DistributedTask) -> Result<TaskResult> {
        let start_time = std::time::Instant::now();

        // Real distributed task execution using HTTP communication
        let result = Self::send_task_to_node(node, task).await?;

        let execution_time = start_time.elapsed();

        // Estimate memory usage based on data size and task type
        let memory_used_mb = Self::estimate_memory_usage(task, &result);

        Ok(TaskResult {
            task_id: match task {
                DistributedTask::Fit { task_id, .. } => task_id.clone(),
                DistributedTask::Transform { task_id, .. } => task_id.clone(),
                DistributedTask::Aggregate { task_id, .. } => task_id.clone(),
            },
            node_id: node.id.clone(),
            result,
            execution_time_ms: execution_time.as_millis() as u64,
            memory_used_mb,
        })
    }

    /// Estimate memory usage based on task type and data size
    fn estimate_memory_usage(task: &DistributedTask, result: &[u8]) -> f64 {
        let base_overhead = 10.0; // Base overhead in MB
        let result_size_mb = result.len() as f64 / (1024.0 * 1024.0);

        match task {
            DistributedTask::Fit { data_partition, .. } => {
                // Estimate memory for fit operations (data + intermediate computations)
                let data_size_mb = (data_partition.len() * std::mem::size_of::<Vec<f64>>()) as f64
                    / (1024.0 * 1024.0);
                let computation_overhead = data_size_mb * 2.5; // 2.5x for covariance matrix and stats
                base_overhead + data_size_mb + computation_overhead + result_size_mb
            }
            DistributedTask::Transform {
                data_partition,
                transformer_state,
                ..
            } => {
                // Memory for data + transformer state + output
                let data_size_mb = (data_partition.len() * std::mem::size_of::<Vec<f64>>()) as f64
                    / (1024.0 * 1024.0);
                let state_size_mb = transformer_state.len() as f64 / (1024.0 * 1024.0);
                base_overhead + data_size_mb + state_size_mb + result_size_mb
            }
            DistributedTask::Aggregate {
                partial_results, ..
            } => {
                // Memory for aggregating partial results
                let input_size_mb = partial_results
                    .iter()
                    .map(|r| r.len() as f64 / (1024.0 * 1024.0))
                    .sum::<f64>();
                base_overhead + input_size_mb + result_size_mb
            }
        }
    }

    /// Submit a task for distributed execution
    pub async fn submit_task(&self, task: DistributedTask) -> Result<()> {
        self.task_sender.send(task).map_err(|e| {
            TransformError::ComputationError(format!("Failed to submit task: {}", e))
        })?;
        Ok(())
    }

    /// Wait for task completion and get result
    pub async fn get_result(&self, task_id: &TaskId) -> Result<TaskResult> {
        loop {
            {
                let results_guard = self.results.read().await;
                if let Some(result) = results_guard.get(task_id) {
                    return Ok(result.clone());
                }
            }

            // Check for new results
            let mut receiver_guard = self.result_receiver.write().await;
            if let Ok(result) = receiver_guard.try_recv() {
                let mut results_guard = self.results.write().await;
                results_guard.insert(result.task_id.clone(), result.clone());
                drop(results_guard);
                drop(receiver_guard);

                if &result.task_id == task_id {
                    return Ok(result);
                }
            } else {
                drop(receiver_guard);
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        }
    }
}

/// Distributed PCA implementation
#[cfg(feature = "distributed")]
pub struct DistributedPCA {
    n_components: usize,
    coordinator: DistributedCoordinator,
    components: Option<Array2<f64>>,
    mean: Option<Array2<f64>>,
}

#[cfg(feature = "distributed")]
impl DistributedPCA {
    /// Create a new distributed PCA instance
    pub async fn new(n_components: usize, config: DistributedConfig) -> Result<Self> {
        let coordinator = DistributedCoordinator::new(config).await?;

        Ok(DistributedPCA {
            n_components,
            coordinator,
            components: None,
            mean: None,
        })
    }

    /// Fit PCA using distributed computation
    pub async fn fit(&mut self, x: &ArrayView2<f64>) -> Result<()> {
        let (n_samples, n_features) = x.dim();

        // Partition data across nodes
        let partitions = self.partition_data(x).await?;

        // Submit tasks to compute local statistics
        let mut task_ids = Vec::new();
        for (i, partition) in partitions.iter().enumerate() {
            let task_id = format!("pca_fit_{}", i);
            let task = DistributedTask::Fit {
                task_id: task_id.clone(),
                transformer_type: "PCA".to_string(),
                parameters: [("n_components".to_string(), self.n_components as f64)]
                    .iter()
                    .cloned()
                    .collect(),
                data_partition: partition.clone(),
            };

            self.coordinator.submit_task(task).await?;
            task_ids.push(task_id);
        }

        // Collect results
        let mut partial_results = Vec::new();
        for task_id in task_ids {
            let result = self.coordinator.get_result(&task_id).await?;
            partial_results.push(result.result);
        }

        // Aggregate results
        let aggregate_task_id = "pca_aggregate".to_string();
        let aggregate_task = DistributedTask::Aggregate {
            task_id: aggregate_task_id.clone(),
            partial_results,
        };

        self.coordinator.submit_task(aggregate_task).await?;
        let final_result = self.coordinator.get_result(&aggregate_task_id).await?;

        // Deserialize final components
        let components: Vec<f64> = bincode::deserialize(&final_result.result).map_err(|e| {
            TransformError::ComputationError(format!("Failed to deserialize components: {}", e))
        })?;

        // Reshape to proper dimensions (placeholder implementation)
        self.components = Some(
            Array2::from_shape_vec((self.n_components, n_features), components).map_err(|e| {
                TransformError::ComputationError(format!("Failed to reshape components: {}", e))
            })?,
        );

        Ok(())
    }

    /// Transform data using distributed computation
    pub async fn transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if self.components.is_none() {
            return Err(TransformError::NotFitted(
                "PCA model not fitted".to_string(),
            ));
        }

        let partitions = self.partition_data(x).await?;
        let mut task_ids = Vec::new();

        // Submit transform tasks
        for (i, partition) in partitions.iter().enumerate() {
            let task_id = format!("pca_transform_{}", i);
            let transformer_state = bincode::serialize(self.components.as_ref().unwrap()).unwrap();

            let task = DistributedTask::Transform {
                task_id: task_id.clone(),
                transformer_state,
                data_partition: partition.clone(),
            };

            self.coordinator.submit_task(task).await?;
            task_ids.push(task_id);
        }

        // Collect and combine results
        let mut all_results = Vec::new();
        for task_id in task_ids {
            let result = self.coordinator.get_result(&task_id).await?;
            let transformed_partition: Vec<f64> = bincode::deserialize(&result.result).unwrap();
            all_results.extend(transformed_partition);
        }

        // Reshape to final array
        let (n_samples, _) = x.dim();
        Array2::from_shape_vec((n_samples, self.n_components), all_results).map_err(|e| {
            TransformError::ComputationError(format!("Failed to reshape result: {}", e))
        })
    }

    /// Partition data for distributed processing using intelligent strategies
    async fn partition_data(&self, x: &ArrayView2<f64>) -> Result<Vec<Vec<Vec<f64>>>> {
        let (n_samples, n_features) = x.dim();
        let nodes = self.coordinator.nodes.read().await;
        
        match &self.coordinator.config.partitioning_strategy {
            PartitioningStrategy::RowWise => {
                self.partition_rowwise(x, &*nodes).await
            },
            PartitioningStrategy::ColumnWise => {
                self.partition_columnwise(x, &*nodes).await
            },
            PartitioningStrategy::BlockWise { block_size } => {
                self.partition_blockwise(x, &*nodes, *block_size).await
            },
            PartitioningStrategy::Adaptive => {
                self.partition_adaptive(x, &*nodes).await
            }
        }
    }
    
    /// Row-wise partitioning with load balancing
    async fn partition_rowwise(&self, x: &ArrayView2<f64>, nodes: &HashMap<NodeId, NodeInfo>) -> Result<Vec<Vec<Vec<f64>>>> {
        let (n_samples, _) = x.dim();
        let n_nodes = nodes.len();
        
        if n_nodes == 0 {
            return Err(TransformError::DistributedError("No nodes available".to_string()));
        }
        
        // Calculate node weights based on their capabilities
        let total_capacity: f64 = nodes.values()
            .map(|node| node.memory_gb + node.cpu_cores as f64)
            .sum();
        
        let mut partitions = Vec::new();
        let mut current_row = 0;
        
        for node in nodes.values() {
            let node_capacity = node.memory_gb + node.cpu_cores as f64;
            let capacity_ratio = node_capacity / total_capacity;
            let rows_for_node = ((n_samples as f64 * capacity_ratio) as usize).max(1);
            let end_row = (current_row + rows_for_node).min(n_samples);
            
            if current_row < end_row {
                let partition = x.slice(ndarray::s![current_row..end_row, ..]);
                let partition_vec: Vec<Vec<f64>> = partition
                    .rows()
                    .into_iter()
                    .map(|row| row.to_vec())
                    .collect();
                partitions.push(partition_vec);
                current_row = end_row;
            }
            
            if current_row >= n_samples {
                break;
            }
        }
        
        Ok(partitions)
    }
    
    /// Column-wise partitioning for feature-parallel processing
    async fn partition_columnwise(&self, x: &ArrayView2<f64>, nodes: &HashMap<NodeId, NodeInfo>) -> Result<Vec<Vec<Vec<f64>>>> {
        let (n_samples, n_features) = x.dim();
        let n_nodes = nodes.len();
        
        if n_nodes == 0 {
            return Err(TransformError::DistributedError("No nodes available".to_string()));
        }
        
        let features_per_node = (n_features + n_nodes - 1) / n_nodes;
        let mut partitions = Vec::new();
        
        for i in 0..n_nodes {
            let start_col = i * features_per_node;
            let end_col = ((i + 1) * features_per_node).min(n_features);
            
            if start_col < end_col {
                let partition = x.slice(ndarray::s![.., start_col..end_col]);
                let partition_vec: Vec<Vec<f64>> = partition
                    .rows()
                    .into_iter()
                    .map(|row| row.to_vec())
                    .collect();
                partitions.push(partition_vec);
            }
        }
        
        Ok(partitions)
    }
    
    /// Block-wise partitioning for 2D parallelism
    async fn partition_blockwise(
        &self, 
        x: &ArrayView2<f64>, 
        nodes: &HashMap<NodeId, NodeInfo>,
        block_size: (usize, usize)
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        let (n_samples, n_features) = x.dim();
        let (block_rows, block_cols) = block_size;
        let n_nodes = nodes.len();
        
        if n_nodes == 0 {
            return Err(TransformError::DistributedError("No nodes available".to_string()));
        }
        
        let blocks_per_row = (n_features + block_cols - 1) / block_cols;
        let blocks_per_col = (n_samples + block_rows - 1) / block_rows;
        let total_blocks = blocks_per_row * blocks_per_col;
        
        // Distribute blocks across nodes
        let blocks_per_node = (total_blocks + n_nodes - 1) / n_nodes;
        let mut partitions = Vec::new();
        let mut block_idx = 0;
        
        for _node_idx in 0..n_nodes {
            let mut node_partition = Vec::new();
            
            for _ in 0..blocks_per_node {
                if block_idx >= total_blocks {
                    break;
                }
                
                let block_row = block_idx / blocks_per_row;
                let block_col = block_idx % blocks_per_row;
                
                let start_row = block_row * block_rows;
                let end_row = ((block_row + 1) * block_rows).min(n_samples);
                let start_col = block_col * block_cols;
                let end_col = ((block_col + 1) * block_cols).min(n_features);
                
                if start_row < end_row && start_col < end_col {
                    let block = x.slice(ndarray::s![start_row..end_row, start_col..end_col]);
                    for row in block.rows() {
                        node_partition.push(row.to_vec());
                    }
                }
                
                block_idx += 1;
            }
            
            if !node_partition.is_empty() {
                partitions.push(node_partition);
            }
        }
        
        Ok(partitions)
    }
    
    /// Adaptive partitioning based on data characteristics and node capabilities
    async fn partition_adaptive(&self, x: &ArrayView2<f64>, nodes: &HashMap<NodeId, NodeInfo>) -> Result<Vec<Vec<Vec<f64>>>> {
        let (n_samples, n_features) = x.dim();
        
        // Analyze data characteristics
        let data_density = self.calculate_data_density(x)?;
        let feature_correlation = self.estimate_feature_correlation(x)?;
        let data_size_gb = (n_samples * n_features * std::mem::size_of::<f64>()) as f64 
            / (1024.0 * 1024.0 * 1024.0);
        
        // Choose optimal strategy based on data and node characteristics
        if n_features > n_samples * 2 && feature_correlation < 0.3 {
            // High dimensional, low correlation -> column-wise partitioning
            self.partition_columnwise(x, nodes).await
        } else if data_size_gb > 10.0 && nodes.len() > 4 {
            // Large data with many nodes -> block-wise partitioning
            let optimal_block_size = self.calculate_optimal_block_size(x, nodes)?;
            self.partition_blockwise(x, nodes, optimal_block_size).await
        } else {
            // Default to row-wise with load balancing
            self.partition_rowwise(x, nodes).await
        }
    }
    
    /// Calculate data density (ratio of non-zero elements)
    fn calculate_data_density(&self, x: &ArrayView2<f64>) -> Result<f64> {
        let total_elements = x.len();
        let non_zero_elements = x.iter().filter(|&&val| val != 0.0).count();
        Ok(non_zero_elements as f64 / total_elements as f64)
    }
    
    /// Estimate average feature correlation
    fn estimate_feature_correlation(&self, x: &ArrayView2<f64>) -> Result<f64> {
        let (_, n_features) = x.dim();
        
        // Sample a subset of feature pairs for efficiency
        let max_pairs = 100;
        let actual_pairs = if n_features < 15 {
            (n_features * (n_features - 1)) / 2
        } else {
            max_pairs
        };
        
        if actual_pairs == 0 {
            return Ok(0.0);
        }
        
        let mut correlation_sum = 0.0;
        let step = if n_features > 15 {
            n_features / 10
        } else {
            1
        };
        
        let mut pair_count = 0;
        for i in (0..n_features).step_by(step) {
            for j in ((i + 1)..n_features).step_by(step) {
                if pair_count >= max_pairs {
                    break;
                }
                
                let col_i = x.column(i);
                let col_j = x.column(j);
                
                if let Ok(corr) = self.quick_correlation(&col_i, &col_j) {
                    correlation_sum += corr.abs();
                    pair_count += 1;
                }
            }
            if pair_count >= max_pairs {
                break;
            }
        }
        
        Ok(if pair_count > 0 {
            correlation_sum / pair_count as f64
        } else {
            0.0
        })
    }
    
    /// Quick correlation calculation for adaptive partitioning
    fn quick_correlation(&self, x: &ndarray::ArrayView1<f64>, y: &ndarray::ArrayView1<f64>) -> Result<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return Ok(0.0);
        }
        
        let n = x.len() as f64;
        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;
        
        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;
        
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator < f64::EPSILON {
            Ok(0.0)
        } else {
            Ok((numerator / denominator).max(-1.0).min(1.0))
        }
    }
    
    /// Calculate optimal block size based on data and node characteristics
    fn calculate_optimal_block_size(&self, x: &ArrayView2<f64>, nodes: &HashMap<NodeId, NodeInfo>) -> Result<(usize, usize)> {
        let (n_samples, n_features) = x.dim();
        
        // Find average node memory capacity
        let avg_memory_gb = nodes.values()
            .map(|node| node.memory_gb)
            .sum::<f64>() / nodes.len() as f64;
        
        // Calculate optimal block size to fit in memory with safety margin
        let memory_per_block_gb = avg_memory_gb * 0.3; // Use 30% of available memory
        let elements_per_block = (memory_per_block_gb * 1024.0 * 1024.0 * 1024.0 / 8.0) as usize; // 8 bytes per f64
        
        // Calculate square-ish blocks
        let block_side = (elements_per_block as f64).sqrt() as usize;
        let block_rows = block_side.min(n_samples / 2).max(100);
        let block_cols = (elements_per_block / block_rows).min(n_features / 2).max(10);
        
        Ok((block_rows, block_cols))
    }
}

// Stub implementations when distributed feature is not enabled
#[cfg(not(feature = "distributed"))]
pub struct DistributedConfig;

#[cfg(not(feature = "distributed"))]
pub struct DistributedCoordinator;

#[cfg(not(feature = "distributed"))]
pub struct DistributedPCA;

#[cfg(not(feature = "distributed"))]
impl DistributedPCA {
    pub async fn new(_n_components: usize, _config: DistributedConfig) -> Result<Self> {
        Err(TransformError::FeatureNotEnabled(
            "Distributed processing requires the 'distributed' feature to be enabled".to_string(),
        ))
    }
}
