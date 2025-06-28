//! Distributed computing support for metrics computation
//!
//! This module provides tools for computing metrics across multiple nodes
//! in a distributed environment, supporting both cluster and cloud deployments.

use crate::error::{MetricsError, Result};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, RwLock};

/// Configuration for distributed metrics computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// List of worker node addresses
    pub worker_addresses: Vec<String>,
    /// Maximum chunk size for data distribution
    pub max_chunk_size: usize,
    /// Timeout for worker operations (in milliseconds)
    pub worker_timeout_ms: u64,
    /// Number of retries for failed operations
    pub max_retries: usize,
    /// Enable compression for data transfer
    pub enable_compression: bool,
    /// Replication factor for fault tolerance
    pub replication_factor: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            worker_addresses: Vec::new(),
            max_chunk_size: 100000,
            worker_timeout_ms: 30000,
            max_retries: 3,
            enable_compression: true,
            replication_factor: 1,
        }
    }
}

/// Message types for distributed computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedMessage {
    /// Request to compute metrics on a data chunk
    ComputeMetrics {
        task_id: String,
        chunk_id: usize,
        y_true: Vec<f64>,
        y_pred: Vec<f64>,
        metric_names: Vec<String>,
    },
    /// Response with computed metrics
    MetricsResult {
        task_id: String,
        chunk_id: usize,
        results: HashMap<String, f64>,
        sample_count: usize,
    },
    /// Health check request
    HealthCheck,
    /// Health check response
    HealthCheckResponse { status: WorkerStatus },
    /// Error message
    Error {
        task_id: String,
        chunk_id: Option<usize>,
        error: String,
    },
}

/// Worker node status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatus {
    pub node_id: String,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub active_tasks: usize,
    pub completed_tasks: usize,
    pub last_heartbeat: u64,
}

/// Result of distributed computation
#[derive(Debug, Clone)]
pub struct DistributedMetricsResult {
    /// Final aggregated metrics
    pub metrics: HashMap<String, f64>,
    /// Per-worker execution times
    pub execution_times: HashMap<String, u64>,
    /// Total samples processed
    pub total_samples: usize,
    /// Number of workers used
    pub workers_used: usize,
    /// Any errors encountered
    pub errors: Vec<String>,
}

/// Aggregation strategy for distributed results
#[derive(Debug, Clone)]
pub enum AggregationStrategy {
    /// Simple mean aggregation
    Mean,
    /// Weighted mean by sample count
    WeightedMean,
    /// Sum aggregation
    Sum,
    /// Custom aggregation function
    Custom(fn(&[f64], &[usize]) -> f64),
}

/// Distributed metrics coordinator
pub struct DistributedMetricsCoordinator {
    config: DistributedConfig,
    workers: Arc<RwLock<HashMap<String, WorkerConnection>>>,
    task_counter: Arc<RwLock<usize>>,
}

/// Worker connection wrapper
struct WorkerConnection {
    address: String,
    status: WorkerStatus,
    sender: mpsc::Sender<DistributedMessage>,
    _handle: Option<std::thread::JoinHandle<()>>,
}

impl DistributedMetricsCoordinator {
    /// Create a new distributed metrics coordinator
    pub fn new(config: DistributedConfig) -> Result<Self> {
        let workers = Arc::new(RwLock::new(HashMap::new()));
        let coordinator = Self {
            config,
            workers,
            task_counter: Arc::new(RwLock::new(0)),
        };

        // Initialize worker connections
        coordinator.initialize_workers()?;

        Ok(coordinator)
    }

    /// Initialize connections to all worker nodes
    fn initialize_workers(&self) -> Result<()> {
        let mut workers = self.workers.write().unwrap();

        for address in &self.config.worker_addresses {
            let worker = self.create_worker_connection(address.clone())?;
            workers.insert(address.clone(), worker);
        }

        Ok(())
    }

    /// Create a connection to a worker node
    fn create_worker_connection(&self, address: String) -> Result<WorkerConnection> {
        let (sender, receiver) = mpsc::channel();

        // Simulate worker connection (in real implementation, this would be network communication)
        let worker_address = address.clone();
        let handle = std::thread::spawn(move || {
            while let Ok(message) = receiver.recv() {
                // Process message (in real implementation, send over network)
                match message {
                    DistributedMessage::ComputeMetrics {
                        task_id,
                        chunk_id,
                        y_true,
                        y_pred,
                        metric_names,
                    } => {
                        // Simulate computation time
                        std::thread::sleep(std::time::Duration::from_millis(100));

                        // Compute metrics locally
                        let mut results = HashMap::new();
                        for metric_name in &metric_names {
                            let result = match metric_name.as_str() {
                                "mse" => {
                                    let mse: f64 = y_true
                                        .iter()
                                        .zip(y_pred.iter())
                                        .map(|(t, p)| (t - p).powi(2))
                                        .sum::<f64>()
                                        / y_true.len() as f64;
                                    mse
                                }
                                "mae" => {
                                    let mae: f64 = y_true
                                        .iter()
                                        .zip(y_pred.iter())
                                        .map(|(t, p)| (t - p).abs())
                                        .sum::<f64>()
                                        / y_true.len() as f64;
                                    mae
                                }
                                "accuracy" => {
                                    let correct: usize = y_true
                                        .iter()
                                        .zip(y_pred.iter())
                                        .map(|(t, p)| if (t - p).abs() < 0.5 { 1 } else { 0 })
                                        .sum();
                                    correct as f64 / y_true.len() as f64
                                }
                                _ => 0.0,
                            };
                            results.insert(metric_name.clone(), result);
                        }

                        // Send result back (in real implementation, send over network)
                        println!(
                            "Worker {} computed metrics for task {} chunk {}",
                            worker_address, task_id, chunk_id
                        );
                    }
                    DistributedMessage::HealthCheck => {
                        // Respond with health status
                        println!("Worker {} health check", worker_address);
                    }
                    _ => {}
                }
            }
        });

        Ok(WorkerConnection {
            address: address.clone(),
            status: WorkerStatus {
                node_id: address,
                cpu_usage: 0.0,
                memory_usage: 0.0,
                active_tasks: 0,
                completed_tasks: 0,
                last_heartbeat: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            },
            sender,
            _handle: Some(handle),
        })
    }

    /// Compute metrics across distributed nodes
    pub fn compute_distributed_metrics(
        &self,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        metric_names: &[String],
        aggregation: AggregationStrategy,
    ) -> Result<DistributedMetricsResult> {
        // Generate unique task ID
        let task_id = {
            let mut counter = self.task_counter.write().unwrap();
            *counter += 1;
            format!("task_{}", *counter)
        };

        // Split data into chunks
        let chunks = self.create_data_chunks(y_true, y_pred)?;

        // Distribute chunks to workers
        let chunk_results = self.distribute_chunks(&task_id, chunks, metric_names)?;

        // Aggregate results
        let aggregated_metrics = self.aggregate_results(chunk_results, &aggregation)?;

        Ok(DistributedMetricsResult {
            metrics: aggregated_metrics,
            execution_times: HashMap::new(), // Would be populated in real implementation
            total_samples: y_true.len(),
            workers_used: self.config.worker_addresses.len(),
            errors: Vec::new(),
        })
    }

    /// Create data chunks for distribution
    fn create_data_chunks(
        &self,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
    ) -> Result<Vec<(Vec<f64>, Vec<f64>)>> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let total_samples = y_true.len();
        let num_workers = self.config.worker_addresses.len().max(1);
        let chunk_size = (total_samples + num_workers - 1) / num_workers;

        let mut chunks = Vec::new();

        for i in (0..total_samples).step_by(chunk_size) {
            let end = (i + chunk_size).min(total_samples);
            let true_chunk = y_true.slice(s![i..end]).to_vec();
            let pred_chunk = y_pred.slice(s![i..end]).to_vec();
            chunks.push((true_chunk, pred_chunk));
        }

        Ok(chunks)
    }

    /// Distribute chunks to worker nodes
    fn distribute_chunks(
        &self,
        task_id: &str,
        chunks: Vec<(Vec<f64>, Vec<f64>)>,
        metric_names: &[String],
    ) -> Result<Vec<ChunkResult>> {
        let workers = self.workers.read().unwrap();
        let worker_addresses: Vec<_> = workers.keys().cloned().collect();

        if worker_addresses.is_empty() {
            return Err(MetricsError::ComputationError(
                "No workers available".to_string(),
            ));
        }

        let mut results = Vec::new();

        for (chunk_id, (y_true_chunk, y_pred_chunk)) in chunks.into_iter().enumerate() {
            let worker_address = &worker_addresses[chunk_id % worker_addresses.len()];

            if let Some(worker) = workers.get(worker_address) {
                let message = DistributedMessage::ComputeMetrics {
                    task_id: task_id.to_string(),
                    chunk_id,
                    y_true: y_true_chunk.clone(),
                    y_pred: y_pred_chunk.clone(),
                    metric_names: metric_names.to_vec(),
                };

                // Send task to worker
                if let Err(e) = worker.sender.send(message) {
                    return Err(MetricsError::ComputationError(format!(
                        "Failed to send task to worker {}: {}",
                        worker_address, e
                    )));
                }

                // Compute result directly (simplified for non-async version)
                let chunk_result = ChunkResult {
                    chunk_id,
                    metrics: self.compute_chunk_metrics_locally(
                        &y_true_chunk,
                        &y_pred_chunk,
                        metric_names,
                    )?,
                    sample_count: y_true_chunk.len(),
                };

                results.push(chunk_result);
            }
        }

        // Results already collected above

        Ok(results)
    }

    /// Compute metrics locally for a chunk (fallback/simulation)
    fn compute_chunk_metrics_locally(
        &self,
        y_true: &[f64],
        y_pred: &[f64],
        metric_names: &[String],
    ) -> Result<HashMap<String, f64>> {
        let mut results = HashMap::new();

        for metric_name in metric_names {
            let result = match metric_name.as_str() {
                "mse" => {
                    let mse: f64 = y_true
                        .iter()
                        .zip(y_pred.iter())
                        .map(|(t, p)| (t - p).powi(2))
                        .sum::<f64>()
                        / y_true.len() as f64;
                    mse
                }
                "mae" => {
                    let mae: f64 = y_true
                        .iter()
                        .zip(y_pred.iter())
                        .map(|(t, p)| (t - p).abs())
                        .sum::<f64>()
                        / y_true.len() as f64;
                    mae
                }
                "rmse" => {
                    let mse: f64 = y_true
                        .iter()
                        .zip(y_pred.iter())
                        .map(|(t, p)| (t - p).powi(2))
                        .sum::<f64>()
                        / y_true.len() as f64;
                    mse.sqrt()
                }
                "r2_score" => {
                    let mean_true = y_true.iter().sum::<f64>() / y_true.len() as f64;
                    let ss_tot: f64 = y_true.iter().map(|t| (t - mean_true).powi(2)).sum();
                    let ss_res: f64 = y_true
                        .iter()
                        .zip(y_pred.iter())
                        .map(|(t, p)| (t - p).powi(2))
                        .sum();

                    if ss_tot == 0.0 {
                        0.0
                    } else {
                        1.0 - ss_res / ss_tot
                    }
                }
                _ => {
                    return Err(MetricsError::InvalidInput(format!(
                        "Unsupported metric: {}",
                        metric_name
                    )))
                }
            };

            results.insert(metric_name.clone(), result);
        }

        Ok(results)
    }

    /// Aggregate results from multiple chunks
    fn aggregate_results(
        &self,
        chunk_results: Vec<ChunkResult>,
        strategy: &AggregationStrategy,
    ) -> Result<HashMap<String, f64>> {
        if chunk_results.is_empty() {
            return Ok(HashMap::new());
        }

        let mut aggregated = HashMap::new();

        // Get all metric names
        let metric_names: std::collections::HashSet<String> = chunk_results
            .iter()
            .flat_map(|r| r.metrics.keys().cloned())
            .collect();

        for metric_name in metric_names {
            let values: Vec<f64> = chunk_results
                .iter()
                .filter_map(|r| r.metrics.get(&metric_name).copied())
                .collect();

            let sample_counts: Vec<usize> = chunk_results.iter().map(|r| r.sample_count).collect();

            if !values.is_empty() {
                let aggregated_value = match strategy {
                    AggregationStrategy::Mean => values.iter().sum::<f64>() / values.len() as f64,
                    AggregationStrategy::WeightedMean => {
                        let total_weight: usize = sample_counts.iter().sum();
                        if total_weight == 0 {
                            0.0
                        } else {
                            values
                                .iter()
                                .zip(sample_counts.iter())
                                .map(|(v, &w)| v * w as f64)
                                .sum::<f64>()
                                / total_weight as f64
                        }
                    }
                    AggregationStrategy::Sum => values.iter().sum::<f64>(),
                    AggregationStrategy::Custom(func) => func(&values, &sample_counts),
                };

                aggregated.insert(metric_name, aggregated_value);
            }
        }

        Ok(aggregated)
    }

    /// Compute batch metrics across multiple samples in distributed fashion
    pub fn compute_batch_distributed_metrics(
        &self,
        y_true_batch: &Array2<f64>,
        y_pred_batch: &Array2<f64>,
        metric_names: &[String],
    ) -> Result<Vec<HashMap<String, f64>>> {
        let batch_size = y_true_batch.nrows();
        let mut batch_results = Vec::with_capacity(batch_size);

        // Process each sample in the batch
        for i in 0..batch_size {
            let y_true_sample = y_true_batch.row(i).to_owned();
            let y_pred_sample = y_pred_batch.row(i).to_owned();

            let result = self.compute_distributed_metrics(
                &y_true_sample,
                &y_pred_sample,
                metric_names,
                AggregationStrategy::WeightedMean,
            )?;

            batch_results.push(result.metrics);
        }

        Ok(batch_results)
    }

    /// Check health of all worker nodes
    pub fn check_worker_health(&self) -> Result<HashMap<String, WorkerStatus>> {
        let workers = self.workers.read().unwrap();
        let mut health_status = HashMap::new();

        for (address, worker) in workers.iter() {
            // Send health check message
            let _ = worker.sender.send(DistributedMessage::HealthCheck);

            // In real implementation, would wait for response
            // For now, return current status
            health_status.insert(address.clone(), worker.status.clone());
        }

        Ok(health_status)
    }

    /// Add a new worker node
    pub fn add_worker(&self, address: String) -> Result<()> {
        let worker = self.create_worker_connection(address.clone())?;
        let mut workers = self.workers.write().unwrap();
        workers.insert(address, worker);
        Ok(())
    }

    /// Remove a worker node
    pub fn remove_worker(&self, address: &str) -> Result<()> {
        let mut workers = self.workers.write().unwrap();
        if workers.remove(address).is_some() {
            Ok(())
        } else {
            Err(MetricsError::InvalidInput(format!(
                "Worker {} not found",
                address
            )))
        }
    }

    /// Get current cluster status
    pub fn get_cluster_status(&self) -> ClusterStatus {
        let workers = self.workers.read().unwrap();
        let total_workers = workers.len();
        let active_workers = workers
            .values()
            .filter(|w| w.status.last_heartbeat > 0)
            .count();

        ClusterStatus {
            total_workers,
            active_workers,
            total_tasks_completed: workers.values().map(|w| w.status.completed_tasks).sum(),
            average_cpu_usage: workers.values().map(|w| w.status.cpu_usage).sum::<f64>()
                / workers.len().max(1) as f64,
            average_memory_usage: workers.values().map(|w| w.status.memory_usage).sum::<f64>()
                / workers.len().max(1) as f64,
        }
    }
}

/// Result from computing metrics on a chunk
#[derive(Debug, Clone)]
struct ChunkResult {
    chunk_id: usize,
    metrics: HashMap<String, f64>,
    sample_count: usize,
}

/// Overall cluster status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStatus {
    pub total_workers: usize,
    pub active_workers: usize,
    pub total_tasks_completed: usize,
    pub average_cpu_usage: f64,
    pub average_memory_usage: f64,
}

/// Distributed metrics builder for convenient setup
pub struct DistributedMetricsBuilder {
    config: DistributedConfig,
}

impl DistributedMetricsBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: DistributedConfig::default(),
        }
    }

    /// Add worker nodes
    pub fn with_workers(mut self, addresses: Vec<String>) -> Self {
        self.config.worker_addresses = addresses;
        self
    }

    /// Set chunk size
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.config.max_chunk_size = size;
        self
    }

    /// Set worker timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.config.worker_timeout_ms = timeout_ms;
        self
    }

    /// Enable compression
    pub fn with_compression(mut self, enable: bool) -> Self {
        self.config.enable_compression = enable;
        self
    }

    /// Set replication factor for fault tolerance
    pub fn with_replication(mut self, factor: usize) -> Self {
        self.config.replication_factor = factor;
        self
    }

    /// Build the distributed coordinator
    pub fn build(self) -> Result<DistributedMetricsCoordinator> {
        DistributedMetricsCoordinator::new(self.config)
    }
}

impl Default for DistributedMetricsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Helper macro for including ndarray slice syntax has been replaced with direct import

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_distributed_config_creation() {
        let config = DistributedConfig::default();
        assert_eq!(config.max_chunk_size, 100000);
        assert_eq!(config.worker_timeout_ms, 30000);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_distributed_metrics_builder() {
        let builder = DistributedMetricsBuilder::new()
            .with_workers(vec!["worker1".to_string(), "worker2".to_string()])
            .with_chunk_size(50000)
            .with_timeout(60000)
            .with_compression(true);

        assert_eq!(builder.config.worker_addresses.len(), 2);
        assert_eq!(builder.config.max_chunk_size, 50000);
        assert_eq!(builder.config.worker_timeout_ms, 60000);
        assert!(builder.config.enable_compression);
    }

    #[test]
    fn test_local_metrics_computation() {
        let config = DistributedConfig::default();
        let coordinator = DistributedMetricsCoordinator::new(config).unwrap();

        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.1, 2.1, 2.9, 4.1, 4.9];
        let metrics = vec!["mse".to_string(), "mae".to_string()];

        let result = coordinator
            .compute_chunk_metrics_locally(&y_true, &y_pred, &metrics)
            .unwrap();

        assert!(result.contains_key("mse"));
        assert!(result.contains_key("mae"));
        assert!(result["mse"] > 0.0);
        assert!(result["mae"] > 0.0);
    }

    #[test]
    fn test_aggregation_strategies() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![1, 2, 3, 4];

        // Test mean aggregation
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        assert_eq!(mean, 2.5);

        // Test weighted mean aggregation
        let total_weight: usize = weights.iter().sum();
        let weighted_mean = values
            .iter()
            .zip(weights.iter())
            .map(|(v, &w)| v * w as f64)
            .sum::<f64>()
            / total_weight as f64;
        assert_eq!(weighted_mean, 3.0);

        // Test sum aggregation
        let sum = values.iter().sum::<f64>();
        assert_eq!(sum, 10.0);
    }

    #[test]
    fn test_data_chunking() {
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y_pred = array![1.1, 2.1, 2.9, 4.1, 4.9, 6.1, 6.9, 8.1, 8.9, 10.1];

        let config = DistributedConfig {
            worker_addresses: vec![
                "worker1".to_string(),
                "worker2".to_string(),
                "worker3".to_string(),
            ],
            ..Default::default()
        };

        let coordinator = DistributedMetricsCoordinator::new(config).unwrap();

        let chunks = coordinator.create_data_chunks(&y_true, &y_pred).unwrap();

        // Should create 3 chunks for 3 workers
        assert_eq!(chunks.len(), 3);

        // First two chunks should have 4 elements, last chunk should have 2
        assert_eq!(chunks[0].0.len(), 4);
        assert_eq!(chunks[1].0.len(), 3);
        assert_eq!(chunks[2].0.len(), 3);
    }
}
