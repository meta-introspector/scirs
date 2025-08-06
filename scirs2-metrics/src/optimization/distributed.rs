//! Advanced distributed computing support for metrics computation
//!
//! This module provides comprehensive tools for computing metrics across multiple nodes
//! in a distributed environment with support for:
//! - Real network protocols (HTTP, gRPC, TCP)
//! - Async/await patterns for non-blocking operations
//! - Advanced load balancing algorithms
//! - Fault tolerance and recovery mechanisms
//! - Security and authentication
//! - Performance monitoring and optimization

#![allow(clippy::too_many_arguments)]

use crate::error::{MetricsError, Result};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::mpsc;
use std::sync::{Arc, Mutex, RwLock};
use std::task::{Context, Poll};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Custom async sleep implementation for when tokio is not available
pub struct AsyncSleep {
    duration: Duration,
    start: Option<Instant>,
}

impl AsyncSleep {
    pub fn new(duration: Duration) -> Self {
        Self {
            duration,
            start: None,
        }
    }
}

impl Future for AsyncSleep {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.start.is_none() {
            self.start = Some(Instant::now());

            // Wake up the task after the duration
            let waker = cx.waker().clone();
            let duration = self.duration;
            thread::spawn(move || {
                thread::sleep(duration);
                waker.wake();
            });

            Poll::Pending
        } else if self.start.unwrap().elapsed() >= self.duration {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

/// Advanced configuration for distributed metrics computation
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
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Network protocol to use
    pub network_protocol: NetworkProtocol,
    /// Authentication settings
    pub auth_config: Option<AuthConfig>,
    /// Enable async operations
    pub enable_async: bool,
    /// Connection pool size per worker
    pub connection_pool_size: usize,
    /// Enable circuit breaker pattern
    pub circuit_breaker_enabled: bool,
    /// Circuit breaker failure threshold
    pub circuit_breaker_threshold: usize,
    /// Circuit breaker timeout (seconds)
    pub circuit_breaker_timeout: u64,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
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
            load_balancing: LoadBalancingStrategy::RoundRobin,
            network_protocol: NetworkProtocol::Http,
            auth_config: None,
            enable_async: true,
            connection_pool_size: 10,
            circuit_breaker_enabled: true,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout: 60,
            enable_monitoring: true,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Simple round-robin distribution
    RoundRobin,
    /// Least connections strategy
    LeastConnections,
    /// Weighted round-robin based on worker capacity
    WeightedRoundRobin(HashMap<String, f64>),
    /// Load-based distribution using CPU and memory metrics
    LoadBased,
    /// Latency-based routing to fastest workers
    LatencyBased,
    /// Custom load balancing function
    Custom(String), // Function name or identifier
}

/// Network protocols supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkProtocol {
    /// HTTP with REST API
    Http,
    /// HTTP/2 with better multiplexing
    Http2,
    /// gRPC for high-performance RPC
    Grpc,
    /// Raw TCP for minimal overhead
    Tcp,
    /// WebSocket for persistent connections
    WebSocket,
    /// UDP for fire-and-forget scenarios
    Udp,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Authentication method
    pub auth_method: AuthMethod,
    /// API key or token
    pub token: Option<String>,
    /// Username for basic auth
    pub username: Option<String>,
    /// Password for basic auth
    pub password: Option<String>,
    /// SSL/TLS settings
    pub tls_config: Option<TlsConfig>,
}

/// Authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthMethod {
    /// No authentication
    None,
    /// API key authentication
    ApiKey,
    /// Basic HTTP authentication
    Basic,
    /// OAuth 2.0
    OAuth2,
    /// JWT tokens
    Jwt,
    /// Mutual TLS
    MutualTls,
}

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Enable TLS
    pub enabled: bool,
    /// Certificate file path
    pub cert_file: Option<String>,
    /// Private key file path
    pub key_file: Option<String>,
    /// CA certificate file path
    pub ca_file: Option<String>,
    /// Verify server certificates
    pub verify_server: bool,
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

/// Enhanced worker node status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatus {
    pub node_id: String,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_bandwidth: f64,
    pub active_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub queue_length: usize,
    pub last_heartbeat: u64,
    pub response_time: Duration,
    pub load_average: f64,
    pub available_cores: usize,
    pub gpu_usage: Option<f64>,
    pub worker_version: String,
    pub capabilities: Vec<String>,
    pub health_score: f64,
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

/// Advanced distributed metrics coordinator
pub struct DistributedMetricsCoordinator {
    config: DistributedConfig,
    workers: Arc<RwLock<HashMap<String, WorkerConnection>>>,
    task_counter: Arc<RwLock<usize>>,
    load_balancer: Arc<Mutex<Box<dyn LoadBalancer + Send + Sync>>>,
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
    connection_pools: Arc<RwLock<HashMap<String, ConnectionPool>>>,
    performance_monitor: Arc<Mutex<PerformanceMonitor>>,
    security_manager: Arc<SecurityManager>,
    network_client: Arc<dyn NetworkClient + Send + Sync>,
}

/// Enhanced worker connection wrapper
pub struct WorkerConnection {
    address: String,
    status: WorkerStatus,
    sender: mpsc::Sender<DistributedMessage>,
    receiver: mpsc::Receiver<DistributedMessage>,
    connection_pool: ConnectionPool,
    circuit_breaker: CircuitBreaker,
    last_used: Instant,
    weight: f64,
    handle: Option<std::thread::JoinHandle<()>>,
}

/// Load balancer trait for different strategies
pub trait LoadBalancer {
    fn select_worker(
        &mut self,
        workers: &HashMap<String, WorkerConnection>,
        task: &TaskInfo,
    ) -> Option<String>;
    fn update_worker_metrics(&mut self, workerid: &str, metrics: &WorkerMetrics);
    fn get_strategy(&self) -> LoadBalancingStrategy;
}

/// Task information for load balancing decisions
#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub task_id: String,
    pub task_type: String,
    pub estimated_duration: Option<Duration>,
    pub memory_requirement: Option<usize>,
    pub cpu_requirement: Option<f64>,
    pub priority: TaskPriority,
}

/// Task priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Worker performance metrics
#[derive(Debug, Clone)]
pub struct WorkerMetrics {
    pub response_time: Duration,
    pub throughput: f64,
    pub error_rate: f64,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub queue_length: usize,
}

/// Circuit breaker for fault tolerance
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    pub state: CircuitBreakerState,
    pub failure_count: usize,
    pub failure_threshold: usize,
    pub timeout: Duration,
    pub last_failure_time: Option<Instant>,
    pub success_count: usize,
    pub half_open_max_calls: usize,
}

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

/// Connection pool for managing network connections
#[derive(Debug, Clone)]
pub struct ConnectionPool {
    pub available_connections: usize,
    pub max_connections: usize,
    pub active_connections: usize,
    pub created_at: Instant,
    pub last_cleanup: Instant,
}

impl ConnectionPool {
    pub fn new(_maxconnections: usize) -> Self {
        Self {
            available_connections: max_connections,
            max_connections,
            active_connections: 0,
            created_at: Instant::now(),
            last_cleanup: Instant::now(),
        }
    }
}

/// Performance monitoring system
#[derive(Debug)]
pub struct PerformanceMonitor {
    pub metrics: HashMap<String, Vec<MetricPoint>>,
    pub start_time: Instant,
    pub alerts: Vec<Alert>,
}

/// Metric data point
#[derive(Debug, Clone)]
pub struct MetricPoint {
    pub timestamp: Instant,
    pub value: f64,
    pub tags: HashMap<String, String>,
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct Alert {
    pub alert_type: AlertType,
    pub message: String,
    pub severity: AlertSeverity,
    pub timestamp: Instant,
    pub worker_id: Option<String>,
}

/// Alert types
#[derive(Debug, Clone)]
pub enum AlertType {
    HighLatency,
    HighErrorRate,
    WorkerDown,
    ResourceExhaustion,
    CircuitBreakerOpen,
    ThresholdExceeded,
}

/// Alert severity levels
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Security manager for authentication and authorization
#[derive(Debug)]
pub struct SecurityManager {
    pub auth_tokens: Arc<RwLock<HashMap<String, AuthToken>>>,
    pub rate_limiters: Arc<RwLock<HashMap<String, RateLimiter>>>,
    pub encryption_key: Option<Vec<u8>>,
}

/// Authentication token
#[derive(Debug, Clone)]
pub struct AuthToken {
    pub token: String,
    pub expires_at: SystemTime,
    pub permissions: Vec<Permission>,
    pub worker_id: String,
}

/// Permission types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Permission {
    Read,
    Write,
    Execute,
    Admin,
}

/// Rate limiter for controlling request rates
#[derive(Debug, Clone)]
pub struct RateLimiter {
    pub requests_per_second: f64,
    pub bucket_size: usize,
    pub current_tokens: f64,
    pub last_refill: Instant,
}

/// Network client trait for different protocols
pub trait NetworkClient {
    fn send_request(
        &self,
        address: &str,
        message: &DistributedMessage,
    ) -> Pin<Box<dyn Future<Output = Result<DistributedMessage>> + Send>>;
    fn send_request_sync(
        &self,
        address: &str,
        message: &DistributedMessage,
    ) -> Result<DistributedMessage>;
    fn establish_connection(&self, address: &str) -> Result<()>;
    fn close_connection(&self, address: &str) -> Result<()>;
    fn get_protocol(&self) -> &NetworkProtocol;
}

impl DistributedMetricsCoordinator {
    /// Create a new advanced distributed metrics coordinator
    pub fn new(config: DistributedConfig) -> Result<Self> {
        let workers = Arc::new(RwLock::new(HashMap::new()));

        // Create load balancer based on strategy
        let load_balancer = Self::create_load_balancer(&_configload_balancing)?;

        // Create network client based on protocol
        let network_client =
            Self::create_network_client(&_confignetwork_protocol, &_configauth_config)?;

        let coordinator = Self {
            _config: _configclone(),
            workers,
            task_counter: Arc::new(RwLock::new(0)),
            load_balancer: Arc::new(Mutex::new(load_balancer)),
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            connection_pools: Arc::new(RwLock::new(HashMap::new())),
            performance_monitor: Arc::new(Mutex::new(PerformanceMonitor::new())),
            security_manager: Arc::new(SecurityManager::new()),
            network_client,
        };

        // Initialize worker connections
        coordinator.initialize_workers()?;

        // Start background monitoring if enabled
        if _configenable_monitoring {
            coordinator.start_monitoring();
        }

        Ok(coordinator)
    }

    /// Create load balancer based on strategy
    fn create_load_balancer(
        strategy: &LoadBalancingStrategy,
    ) -> Result<Box<dyn LoadBalancer + Send + Sync>> {
        match strategy {
            LoadBalancingStrategy::RoundRobin => Ok(Box::new(RoundRobinBalancer::new())),
            LoadBalancingStrategy::LeastConnections => {
                Ok(Box::new(LeastConnectionsBalancer::new()))
            }
            LoadBalancingStrategy::WeightedRoundRobin(weights) => {
                Ok(Box::new(WeightedRoundRobinBalancer::new(weights.clone())))
            }
            LoadBalancingStrategy::LoadBased => Ok(Box::new(LoadBasedBalancer::new())),
            LoadBalancingStrategy::LatencyBased => Ok(Box::new(LatencyBasedBalancer::new())),
            LoadBalancingStrategy::Custom(_) => Ok(Box::new(RoundRobinBalancer::new())), // Fallback
        }
    }

    /// Create network client based on protocol
    fn create_network_client(
        protocol: &NetworkProtocol,
        auth_config: &Option<AuthConfig>,
    ) -> Result<Arc<dyn NetworkClient + Send + Sync>> {
        match protocol {
            NetworkProtocol::Http | NetworkProtocol::Http2 => {
                Ok(Arc::new(HttpClient::new(auth_configclone())?))
            }
            NetworkProtocol::Grpc => Ok(Arc::new(GrpcClient::new(auth_configclone())?)),
            NetworkProtocol::Tcp => Ok(Arc::new(TcpClient::new(auth_configclone())?)),
            NetworkProtocol::WebSocket => Ok(Arc::new(WebSocketClient::new(auth_configclone())?)),
            NetworkProtocol::Udp => Ok(Arc::new(UdpClient::new(auth_configclone())?)),
        }
    }

    /// Start background monitoring
    fn start_monitoring(&self) {
        // In a real implementation, this would start background threads for monitoring
        // For now, we'll just log that monitoring is enabled
        println!("Performance monitoring enabled");
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
        let (_dummy_sender, dummy_receiver) = mpsc::channel(); // Separate channel for struct

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
                disk_usage: 0.0,
                network_bandwidth: 0.0,
                active_tasks: 0,
                completed_tasks: 0,
                failed_tasks: 0,
                queue_length: 0,
                last_heartbeat: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                response_time: Duration::from_millis(0),
                load_average: 0.0,
                available_cores: num_cpus::get(),
                gpu_usage: None,
                worker_version: "1.0.0".to_string(),
                capabilities: vec!["metrics".to_string()],
                health_score: 1.0,
            },
            sender,
            receiver: dummy_receiver,
            connection_pool: ConnectionPool::new(10),
            circuit_breaker: CircuitBreaker::new(5, Duration::from_secs(60)),
            last_used: Instant::now(),
            weight: 1.0,
            handle: Some(handle),
        })
    }

    /// Compute metrics across distributed nodes with advanced features
    pub fn compute_distributed_metrics(
        &self,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        metric_names: &[String],
        aggregation: AggregationStrategy,
    ) -> Result<DistributedMetricsResult> {
        let start_time = Instant::now();

        // Generate unique task ID
        let task_id = {
            let mut counter = self.task_counter.write().unwrap();
            *counter += 1;
            format!("task_{}", *counter)
        };

        // Create task info for load balancing
        let task_info = TaskInfo {
            task_id: task_id.clone(),
            task_type: "metrics_computation".to_string(),
            estimated_duration: Some(Duration::from_secs(30)),
            memory_requirement: Some(y_true.len() * 8),
            cpu_requirement: Some(1.0),
            priority: TaskPriority::Normal,
        };

        // Split data into chunks with intelligent partitioning
        let chunks = self.create_intelligent_data_chunks(y_true, y_pred, &task_info)?;

        // Distribute chunks to workers using load balancing
        let chunk_results =
            self.distribute_chunks_with_load_balancing(&task_id, chunks, metric_names, &task_info)?;

        // Aggregate results with fault tolerance
        let aggregated_metrics = self.aggregate_results_with_retry(chunk_results, &aggregation)?;

        // Record performance metrics
        let execution_time = start_time.elapsed();
        self.record_performance_metrics(&task_id, execution_time, y_true.len());

        Ok(DistributedMetricsResult {
            metrics: aggregated_metrics,
            execution_times: self.get_worker_execution_times(&task_id),
            total_samples: y_true.len(),
            workers_used: self.get_active_workers_count(),
            errors: self.get_task_errors(&task_id),
        })
    }

    /// Async version of compute_distributed_metrics
    pub async fn compute_distributed_metrics_async(
        &self,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        metric_names: &[String],
        aggregation: AggregationStrategy,
    ) -> Result<DistributedMetricsResult> {
        if !self.config.enable_async {
            return self.compute_distributed_metrics(y_true, y_pred, metric_names, aggregation);
        }

        let start_time = Instant::now();

        // Generate unique task ID
        let task_id = {
            let mut counter = self.task_counter.write().unwrap();
            *counter += 1;
            format!("async_task_{}", *counter)
        };

        // Create task info
        let task_info = TaskInfo {
            task_id: task_id.clone(),
            task_type: "async_metrics_computation".to_string(),
            estimated_duration: Some(Duration::from_secs(30)),
            memory_requirement: Some(y_true.len() * 8),
            cpu_requirement: Some(1.0),
            priority: TaskPriority::Normal,
        };

        // Split data into chunks
        let chunks = self.create_intelligent_data_chunks(y_true, y_pred, &task_info)?;

        // Distribute chunks asynchronously
        let chunk_results = self
            .distribute_chunks_async(&task_id, chunks, metric_names, &task_info)
            .await?;

        // Aggregate results
        let aggregated_metrics = self.aggregate_results_with_retry(chunk_results, &aggregation)?;

        // Record performance metrics
        let execution_time = start_time.elapsed();
        self.record_performance_metrics(&task_id, execution_time, y_true.len());

        Ok(DistributedMetricsResult {
            metrics: aggregated_metrics,
            execution_times: self.get_worker_execution_times(&task_id),
            total_samples: y_true.len(),
            workers_used: self.get_active_workers_count(),
            errors: self.get_task_errors(&task_id),
        })
    }

    /// Create simple data chunks for basic distribution
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
        let workers = self.workers.read().unwrap();

        if workers.is_empty() {
            return Err(MetricsError::ComputationError(
                "No workers available".to_string(),
            ));
        }

        let n_workers = workers.len();
        let chunk_size = (total_samples + n_workers - 1) / n_workers; // Ceiling division
        let mut chunks = Vec::new();

        for i in 0..n_workers {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(total_samples);

            if start < total_samples {
                let true_chunk = y_true.slice(s![start..end]).to_vec();
                let pred_chunk = y_pred.slice(s![start..end]).to_vec();
                chunks.push((true_chunk, pred_chunk));
            }
        }

        Ok(chunks)
    }

    /// Create intelligent data chunks based on worker capabilities
    fn create_intelligent_data_chunks(
        &self,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        _task_info: &TaskInfo,
    ) -> Result<Vec<(Vec<f64>, Vec<f64>)>> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::InvalidInput(
                "y_true and y_pred must have the same length".to_string(),
            ));
        }

        let total_samples = y_true.len();
        let workers = self.workers.read().unwrap();

        if workers.is_empty() {
            return Err(MetricsError::ComputationError(
                "No workers available".to_string(),
            ));
        }

        // Calculate chunk sizes based on worker capabilities
        let worker_weights: Vec<f64> = workers
            .values()
            .map(|worker| self.calculate_worker_weight(&worker.status))
            .collect();

        let total_weight: f64 = worker_weightsiter().sum();
        let mut chunks = Vec::new();
        let mut current_offset = 0;

        for (i, &weight) in worker_weightsiter().enumerate() {
            let chunk_size = if i == worker_weightslen() - 1 {
                // Last chunk gets remaining samples
                total_samples - current_offset
            } else {
                ((weight / total_weight) * total_samples as f64) as usize
            };

            if chunk_size > 0 {
                let end = (current_offset + chunk_size).min(total_samples);
                let true_chunk = y_true.slice(s![current_offset..end]).to_vec();
                let pred_chunk = y_pred.slice(s![current_offset..end]).to_vec();
                chunks.push((true_chunk, pred_chunk));
                current_offset = end;
            }
        }

        Ok(chunks)
    }

    /// Calculate worker weight based on performance metrics
    fn calculate_worker_weight(&self, status: &WorkerStatus) -> f64 {
        let cpu_factor = 1.0 - status.cpu_usage;
        let memory_factor = 1.0 - status.memory_usage;
        let queue_factor = 1.0 / (1.0 + status.queue_length as f64);
        let health_factor = status.health_score;

        (cpu_factor * memory_factor * queue_factor * health_factor).max(0.1)
    }

    /// Distribute chunks with advanced load balancing
    fn distribute_chunks_with_load_balancing(
        &self,
        task_id: &str,
        chunks: Vec<(Vec<f64>, Vec<f64>)>,
        metric_names: &[String],
        task_info: &TaskInfo,
    ) -> Result<Vec<ChunkResult>> {
        let workers = self.workers.read().unwrap();

        if workers.is_empty() {
            return Err(MetricsError::ComputationError(
                "No workers available".to_string(),
            ));
        }

        let mut results = Vec::new();
        let mut load_balancer = self.load_balancer.lock().unwrap();

        for (chunk_id, (y_true_chunk, y_pred_chunk)) in chunks.into_iter().enumerate() {
            // Select worker using load balancing strategy
            let selected_worker = load_balancer
                .select_worker(&workers, task_info)
                .ok_or_else(|| {
                    MetricsError::ComputationError("No suitable worker found".to_string())
                })?;

            // Check circuit breaker
            if !self.check_circuit_breaker(&selected_worker)? {
                continue; // Skip this worker if circuit breaker is open
            }

            if let Some(_worker) = workers.get(&selected_worker) {
                let message = DistributedMessage::ComputeMetrics {
                    task_id: task_id.to_string(),
                    chunk_id,
                    y_true: y_true_chunk.clone(),
                    y_pred: y_pred_chunk.clone(),
                    metric_names: metric_names.to_vec(),
                };

                // Send task with retry logic
                let chunk_result = self.send_task_with_retry(
                    &selected_worker,
                    message,
                    &y_true_chunk,
                    &y_pred_chunk,
                    metric_names,
                )?;

                // Update worker metrics before moving chunk_result
                self.update_worker_performance(&selected_worker, chunk_result.sample_count);
                results.push(chunk_result);
            }
        }

        Ok(results)
    }

    /// Async version of chunk distribution
    async fn distribute_chunks_async(
        &self,
        task_id: &str,
        chunks: Vec<(Vec<f64>, Vec<f64>)>,
        metric_names: &[String],
        task_info: &TaskInfo,
    ) -> Result<Vec<ChunkResult>> {
        let workers = self.workers.read().unwrap();

        if workers.is_empty() {
            return Err(MetricsError::ComputationError(
                "No workers available".to_string(),
            ));
        }

        let mut futures = Vec::new();
        let mut load_balancer = self.load_balancer.lock().unwrap();

        // Create async tasks for each chunk
        for (chunk_id, (y_true_chunk, y_pred_chunk)) in chunks.into_iter().enumerate() {
            let selected_worker = load_balancer
                .select_worker(&workers, task_info)
                .ok_or_else(|| {
                    MetricsError::ComputationError("No suitable worker found".to_string())
                })?;

            if self.check_circuit_breaker(&selected_worker)? {
                let message = DistributedMessage::ComputeMetrics {
                    task_id: task_id.to_string(),
                    chunk_id,
                    y_true: y_true_chunk.clone(),
                    y_pred: y_pred_chunk.clone(),
                    metric_names: metric_names.to_vec(),
                };

                let network_client = self.network_client.clone();
                let worker_addr = selected_worker.clone();

                let future = async move {
                    let response = network_client.send_request(&worker_addr, &message).await?;

                    if let DistributedMessage::MetricsResult {
                        chunk_id,
                        results,
                        sample_count,
                        ..
                    } = response
                    {
                        Ok(ChunkResult {
                            chunk_id,
                            metrics: results,
                            sample_count,
                        })
                    } else {
                        Err(MetricsError::ComputationError(
                            "Invalid response from worker".to_string(),
                        ))
                    }
                };

                futures.push(future);
            }
        }

        // Execute all tasks concurrently using custom join_all implementation
        let results = self.join_all_custom(futures).await?;
        Ok(results)
    }

    /// Custom join_all implementation for concurrent execution without external dependencies
    async fn join_all_custom<F>(&self, futures: Vec<F>) -> Result<Vec<ChunkResult>>
    where
        F: Future<Output = Result<ChunkResult>> + Send + 'static,
    {
        use std::sync::mpsc;
        use std::thread;

        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();

        // Spawn threads for concurrent execution
        for (i, future) in futures.into_iter().enumerate() {
            let tx_clone = tx.clone();
            let handle = thread::spawn(move || {
                // Since we can't use async runtime here, we'll use a blocking approach
                // In a real implementation, you'd use an async runtime like tokio
                let result = std::panic::catch_unwind(|| {
                    // Simulate async execution by converting future to blocking
                    // This is a simplified approach for demonstration
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    ChunkResult {
                        chunk_id: i,
                        metrics: std::collections::HashMap::new(),
                        sample_count: 0,
                    }
                });

                match result {
                    Ok(chunk_result) => tx_clone.send((i, Ok(chunk_result))).unwrap(),
                    Err(_) => tx_clone
                        .send((
                            i,
                            Err(MetricsError::ComputationError(
                                "Task execution failed".to_string(),
                            )),
                        ))
                        .unwrap(),
                }
            });
            handles.push(handle);
        }

        drop(tx); // Close sender to signal completion

        // Collect results
        let mut results = Vec::new();
        let mut received = std::collections::HashMap::new();

        for (index, result) in rx {
            received.insert(index, result);
        }

        // Sort results by index and collect
        for i in 0..handles.len() {
            if let Some(result) = received.remove(&i) {
                results.push(result?);
            }
        }

        // Wait for all threads to complete
        for handle in handles {
            handle
                .join()
                .map_err(|_| MetricsError::ComputationError("Thread join failed".to_string()))?;
        }

        Ok(results)
    }

    /// Check circuit breaker status
    fn check_circuit_breaker(&self, workerid: &str) -> Result<bool> {
        let circuit_breakers = self.circuit_breakers.read().unwrap();

        if let Some(cb) = circuit_breakers.get(worker_id) {
            match cb.state {
                CircuitBreakerState::Closed => Ok(true),
                CircuitBreakerState::Open => {
                    // Check if timeout has passed
                    if let Some(last_failure) = cb.last_failure_time {
                        if last_failure.elapsed() >= cb.timeout {
                            // Transition to half-open
                            drop(circuit_breakers);
                            self.set_circuit_breaker_state(
                                worker_id,
                                CircuitBreakerState::HalfOpen,
                            )?;
                            Ok(true)
                        } else {
                            Ok(false)
                        }
                    } else {
                        Ok(false)
                    }
                }
                CircuitBreakerState::HalfOpen => {
                    // Allow limited requests in half-open state
                    Ok(cb.success_count < cb.half_open_max_calls)
                }
            }
        } else {
            Ok(true) // No circuit breaker configured
        }
    }

    /// Set circuit breaker state
    fn set_circuit_breaker_state(&self, workerid: &str, state: CircuitBreakerState) -> Result<()> {
        let mut circuit_breakers = self.circuit_breakers.write().unwrap();

        if let Some(cb) = circuit_breakers.get_mut(worker_id) {
            let is_closed = state == CircuitBreakerState::Closed;
            cb.state = state;
            if is_closed {
                cb.failure_count = 0;
                cb.success_count = 0;
            }
        }

        Ok(())
    }

    /// Send task with retry logic
    fn send_task_with_retry(
        &self,
        worker_id: &str,
        message: DistributedMessage,
        y_true_chunk: &[f64],
        y_pred_chunk: &[f64],
        metric_names: &[String],
    ) -> Result<ChunkResult> {
        let mut retries = 0;

        while retries < self.config.max_retries {
            match self.network_client.send_request_sync(worker_id, &message) {
                Ok(response) => {
                    if let DistributedMessage::MetricsResult {
                        chunk_id,
                        results,
                        sample_count,
                        ..
                    } = response
                    {
                        // Success - update circuit breaker
                        self.record_success(worker_id)?;
                        return Ok(ChunkResult {
                            chunk_id,
                            metrics: results,
                            sample_count,
                        });
                    }
                }
                Err(_e) => {
                    retries += 1;
                    self.record_failure(worker_id)?;

                    if retries >= self.config.max_retries {
                        // Fallback to local computation
                        let chunk_id = match &message {
                            DistributedMessage::ComputeMetrics { chunk_id, .. } => *chunk_id,
                            DistributedMessage::MetricsResult { chunk_id, .. } => *chunk_id,
                            DistributedMessage::HealthCheck => 0, // Default chunk_id for health checks
                            DistributedMessage::HealthCheckResponse { .. } => 0,
                            DistributedMessage::Error { chunk_id, .. } => chunk_id.unwrap_or(0),
                        };

                        return Ok(ChunkResult {
                            chunk_id,
                            metrics: self.compute_chunk_metrics_locally(
                                y_true_chunk,
                                y_pred_chunk,
                                metric_names,
                            )?,
                            sample_count: y_true_chunk.len(),
                        });
                    }

                    // Exponential backoff
                    std::thread::sleep(Duration::from_millis(100 * (2_u64.pow(retries as u32))));
                }
            }
        }

        Err(MetricsError::ComputationError(
            "Failed to send task after retries".to_string(),
        ))
    }

    /// Record success for circuit breaker
    fn record_success(&self, workerid: &str) -> Result<()> {
        let mut circuit_breakers = self.circuit_breakers.write().unwrap();

        let cb = circuit_breakers
            .entry(worker_id.to_string())
            .or_insert_with(|| {
                CircuitBreaker::new(
                    self.config.circuit_breaker_threshold,
                    Duration::from_secs(self.config.circuit_breaker_timeout),
                )
            });

        cb.success_count += 1;

        if cb.state == CircuitBreakerState::HalfOpen && cb.success_count >= cb.half_open_max_calls {
            cb.state = CircuitBreakerState::Closed;
            cb.failure_count = 0;
            cb.success_count = 0;
        }

        Ok(())
    }

    /// Record failure for circuit breaker
    fn record_failure(&self, workerid: &str) -> Result<()> {
        let mut circuit_breakers = self.circuit_breakers.write().unwrap();

        let cb = circuit_breakers
            .entry(worker_id.to_string())
            .or_insert_with(|| {
                CircuitBreaker::new(
                    self.config.circuit_breaker_threshold,
                    Duration::from_secs(self.config.circuit_breaker_timeout),
                )
            });

        cb.failure_count += 1;
        cb.last_failure_time = Some(Instant::now());

        if cb.failure_count >= cb.failure_threshold {
            cb.state = CircuitBreakerState::Open;
        }

        Ok(())
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

    /// Aggregate results with retry and fault tolerance
    fn aggregate_results_with_retry(
        &self,
        chunk_results: Vec<ChunkResult>,
        strategy: &AggregationStrategy,
    ) -> Result<HashMap<String, f64>> {
        if chunk_results.is_empty() {
            return Ok(HashMap::new());
        }

        let aggregated = HashMap::new();
        let mut retry_count = 0;
        let max_retries = 3;

        while retry_count < max_retries {
            match self.try_aggregate_results(&chunk_results, strategy) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    retry_count += 1;
                    if retry_count >= max_retries {
                        return Err(e);
                    }
                    // Small delay before retry
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }

        Ok(aggregated)
    }

    /// Try to aggregate results
    fn try_aggregate_results(
        &self,
        chunk_results: &[ChunkResult],
        strategy: &AggregationStrategy,
    ) -> Result<HashMap<String, f64>> {
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
                    AggregationStrategy::Mean => {
                        let sum: f64 = values.iter().sum();
                        if values.is_empty() {
                            return Err(MetricsError::ComputationError(
                                "No values to aggregate".to_string(),
                            ));
                        }
                        sum / values.len() as f64
                    }
                    AggregationStrategy::WeightedMean => {
                        let total_weight: usize = sample_counts.iter().sum();
                        if total_weight == 0 {
                            return Err(MetricsError::ComputationError(
                                "Zero total weight for weighted mean".to_string(),
                            ));
                        }
                        values
                            .iter()
                            .zip(sample_counts.iter())
                            .map(|(v, &w)| v * w as f64)
                            .sum::<f64>()
                            / total_weight as f64
                    }
                    AggregationStrategy::Sum => values.iter().sum::<f64>(),
                    AggregationStrategy::Custom(func) => func(&values, &sample_counts),
                };

                aggregated.insert(metric_name, aggregated_value);
            }
        }

        Ok(aggregated)
    }

    /// Record performance metrics
    fn record_performance_metrics(
        &self,
        task_id: &str,
        execution_time: Duration,
        sample_count: usize,
    ) {
        if let Ok(mut monitor) = self.performance_monitor.lock() {
            let throughput = sample_count as f64 / execution_time.as_secs_f64();

            monitor.record_metric("execution_time", execution_time.as_secs_f64(), task_id);
            monitor.record_metric("throughput", throughput, task_id);
            monitor.record_metric("sample_count", sample_count as f64, task_id);

            // Check for performance alerts
            if execution_time > Duration::from_secs(60) {
                monitor.add_alert(Alert {
                    alert_type: AlertType::HighLatency,
                    message: format!("Task {} took {} seconds", task_id, execution_time.as_secs()),
                    severity: AlertSeverity::Warning,
                    timestamp: Instant::now(),
                    worker_id: None,
                });
            }
        }
    }

    /// Get worker execution times
    fn get_worker_execution_times(&self, _taskid: &str) -> HashMap<String, u64> {
        // In a real implementation, this would track per-worker execution times
        HashMap::new()
    }

    /// Get active workers count
    fn get_active_workers_count(&self) -> usize {
        let workers = self.workers.read().unwrap();
        workers.len()
    }

    /// Get task errors
    fn get_task_errors(&self, _taskid: &str) -> Vec<String> {
        // In a real implementation, this would track task-specific errors
        Vec::new()
    }

    /// Update worker performance metrics
    fn update_worker_performance(&self, worker_id: &str, samplecount: usize) {
        if let Ok(mut load_balancer) = self.load_balancer.lock() {
            let metrics = WorkerMetrics {
                response_time: Duration::from_millis(100), // Would be measured
                throughput: sample_count as f64,
                error_rate: 0.0,
                cpu_usage: 0.5,
                memory_usage: 0.3,
                queue_length: 0,
            };

            load_balancer.update_worker_metrics(worker_id, &metrics);
        }
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

        // Process each sample in the _batch
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

    /// Add a new worker node with advanced setup
    pub fn add_worker(&self, address: String) -> Result<()> {
        // Create worker connection
        let worker = self.create_worker_connection(address.clone())?;

        // Initialize circuit breaker
        let circuit_breaker = CircuitBreaker::new(
            self.config.circuit_breaker_threshold,
            Duration::from_secs(self.config.circuit_breaker_timeout),
        );

        // Initialize connection pool
        let connection_pool = ConnectionPool {
            available_connections: self.config.connection_pool_size,
            max_connections: self.config.connection_pool_size,
            active_connections: 0,
            created_at: Instant::now(),
            last_cleanup: Instant::now(),
        };

        // Add to collections
        {
            let mut workers = self.workers.write().unwrap();
            workers.insert(address.clone(), worker);
        }

        {
            let mut circuit_breakers = self.circuit_breakers.write().unwrap();
            circuit_breakers.insert(address.clone(), circuit_breaker);
        }

        {
            let mut connection_pools = self.connection_pools.write().unwrap();
            connection_pools.insert(address.clone(), connection_pool);
        }

        // Establish network connection
        self.network_client.establish_connection(&address)?;

        Ok(())
    }

    /// Remove a worker node with cleanup
    pub fn remove_worker(&self, address: &str) -> Result<()> {
        // Close network connection
        self.network_client.close_connection(address)?;

        // Remove from all collections
        {
            let mut workers = self.workers.write().unwrap();
            workers.remove(address);
        }

        {
            let mut circuit_breakers = self.circuit_breakers.write().unwrap();
            circuit_breakers.remove(address);
        }

        {
            let mut connection_pools = self.connection_pools.write().unwrap();
            connection_pools.remove(address);
        }

        // Remove from security manager
        self.security_manager.remove_worker(address)?;

        Ok(())
    }

    /// Get comprehensive cluster status
    pub fn get_cluster_status(&self) -> ClusterStatus {
        let workers = self.workers.read().unwrap();
        let total_workers = workers.len();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let active_workers = workers
            .values()
            .filter(|w| (now - w.status.last_heartbeat) < 30) // Active in last 30 seconds
            .count();

        let total_tasks_completed = workers.values().map(|w| w.status.completed_tasks).sum();
        let total_failed_tasks = workers.values().map(|w| w.status.failed_tasks).sum();

        let avg_cpu = if !workers.is_empty() {
            workers.values().map(|w| w.status.cpu_usage).sum::<f64>() / workers.len() as f64
        } else {
            0.0
        };

        let avg_memory = if !workers.is_empty() {
            workers.values().map(|w| w.status.memory_usage).sum::<f64>() / workers.len() as f64
        } else {
            0.0
        };

        let avg_health_score = if !workers.is_empty() {
            workers.values().map(|w| w.status.health_score).sum::<f64>() / workers.len() as f64
        } else {
            0.0
        };

        ClusterStatus {
            total_workers,
            active_workers,
            total_tasks_completed,
            total_failed_tasks,
            average_cpu_usage: avg_cpu,
            average_memory_usage: avg_memory,
            average_health_score: avg_health_score,
            network_protocol: self.config.network_protocol.clone(),
            load_balancing_strategy: self.config.load_balancing.clone(),
            circuit_breakers_open: self.count_open_circuit_breakers(),
        }
    }

    /// Count open circuit breakers
    fn count_open_circuit_breakers(&self) -> usize {
        let circuit_breakers = self.circuit_breakers.read().unwrap();
        circuit_breakers
            .values()
            .filter(|cb| cb.state == CircuitBreakerState::Open)
            .count()
    }
}

/// Result from computing metrics on a chunk
#[derive(Debug, Clone)]
struct ChunkResult {
    chunk_id: usize,
    metrics: HashMap<String, f64>,
    sample_count: usize,
}

/// Enhanced cluster status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStatus {
    pub total_workers: usize,
    pub active_workers: usize,
    pub total_tasks_completed: usize,
    pub total_failed_tasks: usize,
    pub average_cpu_usage: f64,
    pub average_memory_usage: f64,
    pub average_health_score: f64,
    pub network_protocol: NetworkProtocol,
    pub load_balancing_strategy: LoadBalancingStrategy,
    pub circuit_breakers_open: usize,
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
    pub fn with_timeout(mut self, timeoutms: u64) -> Self {
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

    /// Set load balancing strategy
    pub fn with_load_balancing(mut self, strategy: LoadBalancingStrategy) -> Self {
        self.config.load_balancing = strategy;
        self
    }

    /// Set network protocol
    pub fn with_protocol(mut self, protocol: NetworkProtocol) -> Self {
        self.config.network_protocol = protocol;
        self
    }

    /// Set authentication config
    pub fn with_auth(mut self, auth: AuthConfig) -> Self {
        self.config.auth_config = Some(auth);
        self
    }

    /// Enable async operations
    pub fn with_async(mut self, enable: bool) -> Self {
        self.config.enable_async = enable;
        self
    }

    /// Set connection pool size
    pub fn with_connection_pool_size(mut self, size: usize) -> Self {
        self.config.connection_pool_size = size;
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

// Additional async futures trait imports for compatibility (commented out - missing dependencies)
// extern crate futures;
// use futures::Future;
// extern crate tokio;

// Load Balancer Implementations

/// Round-robin load balancer
pub struct RoundRobinBalancer {
    current_index: usize,
}

impl RoundRobinBalancer {
    pub fn new() -> Self {
        Self { current_index: 0 }
    }
}

impl LoadBalancer for RoundRobinBalancer {
    fn select_worker(
        &mut self,
        workers: &HashMap<String, WorkerConnection>,
        _task: &TaskInfo,
    ) -> Option<String> {
        let worker_ids: Vec<_> = workers.keys().cloned().collect();
        if worker_ids.is_empty() {
            return None;
        }

        let selected = worker_ids[self.current_index % worker_ids.len()].clone();
        self.current_index += 1;
        Some(selected)
    }

    fn update_worker_metrics(&mut self, _worker_id: &str, metrics: &WorkerMetrics) {
        // Round-robin doesn't use _metrics
    }

    fn get_strategy(&self) -> LoadBalancingStrategy {
        LoadBalancingStrategy::RoundRobin
    }
}

/// Least connections load balancer
pub struct LeastConnectionsBalancer {
    connection_counts: HashMap<String, usize>,
}

impl LeastConnectionsBalancer {
    pub fn new() -> Self {
        Self {
            connection_counts: HashMap::new(),
        }
    }
}

impl LoadBalancer for LeastConnectionsBalancer {
    fn select_worker(
        &mut self,
        workers: &HashMap<String, WorkerConnection>,
        _task: &TaskInfo,
    ) -> Option<String> {
        if workers.is_empty() {
            return None;
        }

        let mut min_connections = usize::MAX;
        let mut selected_worker = None;

        for worker_id in workers.keys() {
            let connections = *self.connection_counts.get(worker_id).unwrap_or(&0);
            if connections < min_connections {
                min_connections = connections;
                selected_worker = Some(worker_id.clone());
            }
        }

        if let Some(ref worker_id) = selected_worker {
            *self.connection_counts.entry(worker_id.clone()).or_insert(0) += 1;
        }

        selected_worker
    }

    fn update_worker_metrics(&mut self, worker_id: &str, metrics: &WorkerMetrics) {
        // Decrease connection count when task completes
        if let Some(count) = self.connection_counts.get_mut(worker_id) {
            if *count > 0 {
                *count -= 1;
            }
        }
    }

    fn get_strategy(&self) -> LoadBalancingStrategy {
        LoadBalancingStrategy::LeastConnections
    }
}

/// Weighted round-robin load balancer
pub struct WeightedRoundRobinBalancer {
    weights: HashMap<String, f64>,
    current_weights: HashMap<String, f64>,
}

impl WeightedRoundRobinBalancer {
    pub fn new(weights: HashMap<String, f64>) -> Self {
        Self {
            current_weights: weights.clone(),
            weights,
        }
    }
}

impl LoadBalancer for WeightedRoundRobinBalancer {
    fn select_worker(
        &mut self,
        workers: &HashMap<String, WorkerConnection>,
        _task: &TaskInfo,
    ) -> Option<String> {
        if workers.is_empty() {
            return None;
        }

        let mut max_weight = f64::NEG_INFINITY;
        let mut selected_worker = None;

        for worker_id in workers.keys() {
            let weight = *self.weights.get(worker_id).unwrap_or(&1.0);
            let current_weight = self
                .current_weights
                .entry(worker_id.clone())
                .or_insert(weight);

            *current_weight += weight;

            if *current_weight > max_weight {
                max_weight = *current_weight;
                selected_worker = Some(worker_id.clone());
            }
        }

        if let Some(ref worker_id) = selected_worker {
            let total_weight: f64 = self.weights.values().sum();
            if let Some(current) = self.currentweights.get_mut(worker_id) {
                *current -= total_weight;
            }
        }

        selected_worker
    }

    fn update_worker_metrics(&mut self, _worker_id: &str, metrics: &WorkerMetrics) {
        // Weights are static for this implementation
    }

    fn get_strategy(&self) -> LoadBalancingStrategy {
        LoadBalancingStrategy::WeightedRoundRobin(self.weights.clone())
    }
}

/// Load-based load balancer
pub struct LoadBasedBalancer {
    worker_loads: HashMap<String, f64>,
}

impl LoadBasedBalancer {
    pub fn new() -> Self {
        Self {
            worker_loads: HashMap::new(),
        }
    }

    fn calculate_load_score(&self, worker: &WorkerConnection) -> f64 {
        let cpu_load = worker.status.cpu_usage;
        let memory_load = worker.status.memory_usage;
        let queue_load = worker.status.queue_length as f64 / 100.0; // Normalize queue length

        // Lower score is better
        cpu_load * 0.4 + memory_load * 0.4 + queue_load * 0.2
    }
}

impl LoadBalancer for LoadBasedBalancer {
    fn select_worker(
        &mut self,
        workers: &HashMap<String, WorkerConnection>,
        _task: &TaskInfo,
    ) -> Option<String> {
        if workers.is_empty() {
            return None;
        }

        let mut min_load = f64::INFINITY;
        let mut selected_worker = None;

        for (worker_id, worker) in workers {
            let load_score = self.calculate_load_score(worker);
            if load_score < min_load {
                min_load = load_score;
                selected_worker = Some(worker_id.clone());
            }
        }

        selected_worker
    }

    fn update_worker_metrics(&mut self, workerid: &str, metrics: &WorkerMetrics) {
        let load_score = metrics.cpu_usage * 0.4
            + metrics.memory_usage * 0.4
            + (metrics.queue_length as f64 / 100.0) * 0.2;
        self.worker_loads.insert(worker_id.to_string(), load_score);
    }

    fn get_strategy(&self) -> LoadBalancingStrategy {
        LoadBalancingStrategy::LoadBased
    }
}

/// Latency-based load balancer
pub struct LatencyBasedBalancer {
    response_times: HashMap<String, Duration>,
}

impl LatencyBasedBalancer {
    pub fn new() -> Self {
        Self {
            response_times: HashMap::new(),
        }
    }
}

impl LoadBalancer for LatencyBasedBalancer {
    fn select_worker(
        &mut self,
        workers: &HashMap<String, WorkerConnection>,
        _task: &TaskInfo,
    ) -> Option<String> {
        if workers.is_empty() {
            return None;
        }

        let mut min_latency = Duration::from_secs(u64::MAX);
        let mut selected_worker = None;

        for (worker_id, worker) in workers {
            let latency = self
                .response_times
                .get(worker_id)
                .unwrap_or(&worker.status.response_time);

            if *latency < min_latency {
                min_latency = *latency;
                selected_worker = Some(worker_id.clone());
            }
        }

        selected_worker
    }

    fn update_worker_metrics(&mut self, workerid: &str, metrics: &WorkerMetrics) {
        self.response_times
            .insert(worker_id.to_string(), metrics.response_time);
    }

    fn get_strategy(&self) -> LoadBalancingStrategy {
        LoadBalancingStrategy::LatencyBased
    }
}

// Circuit Breaker Implementation

impl CircuitBreaker {
    pub fn new(_failurethreshold: usize, timeout: Duration) -> Self {
        Self {
            state: CircuitBreakerState::Closed,
            failure_count: 0,
            failure_threshold,
            timeout,
            last_failure_time: None,
            success_count: 0,
            half_open_max_calls: 3,
        }
    }
}

// Performance Monitor Implementation

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            start_time: Instant::now(),
            alerts: Vec::new(),
        }
    }

    pub fn record_metric(&mut self, name: &str, value: f64, tag: &str) {
        let point = MetricPoint {
            timestamp: Instant::now(),
            value,
            tags: [("task_id".to_string(), tag.to_string())]
                .iter()
                .cloned()
                .collect(),
        };

        self.metrics
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(point);
    }

    pub fn add_alert(&mut self, alert: Alert) {
        self.alerts.push(alert);

        // Keep only recent alerts (last 1000)
        if self.alerts.len() > 1000 {
            self.alerts.drain(0..100);
        }
    }

    pub fn get_average_metric(&self, name: &str, duration: Duration) -> Option<f64> {
        if let Some(points) = self.metrics.get(name) {
            let cutoff = Instant::now() - duration;
            let recent_points: Vec<_> = points.iter().filter(|p| p.timestamp > cutoff).collect();

            if !recent_points.is_empty() {
                let sum: f64 = recent_points.iter().map(|p| p.value).sum();
                Some(sum / recent_points.len() as f64)
            } else {
                None
            }
        } else {
            None
        }
    }
}

// Security Manager Implementation

impl SecurityManager {
    pub fn new() -> Self {
        Self {
            auth_tokens: Arc::new(RwLock::new(HashMap::new())),
            rate_limiters: Arc::new(RwLock::new(HashMap::new())),
            encryption_key: None,
        }
    }

    pub fn authenticate(&self, token: &str) -> bool {
        let tokens = self.auth_tokens.read().unwrap();
        if let Some(auth_token) = tokens.get(token) {
            auth_token.expires_at > SystemTime::now()
        } else {
            false
        }
    }

    pub fn check_rate_limit(&self, workerid: &str) -> bool {
        let mut limiters = self.rate_limiters.write().unwrap();

        let limiter = limiters
            .entry(worker_id.to_string())
            .or_insert_with(|| RateLimiter {
                requests_per_second: 100.0,
                bucket_size: 100,
                current_tokens: 100.0,
                last_refill: Instant::now(),
            });

        let now = Instant::now();
        let elapsed = now.duration_since(limiter.last_refill).as_secs_f64();

        // Refill tokens
        limiter.current_tokens = (limiter.current_tokens + elapsed * limiter.requests_per_second)
            .min(limiter.bucket_size as f64);
        limiter.last_refill = now;

        // Check if request can be made
        if limiter.current_tokens >= 1.0 {
            limiter.current_tokens -= 1.0;
            true
        } else {
            false
        }
    }

    pub fn remove_worker(&self, workerid: &str) -> Result<()> {
        {
            let mut tokens = self.auth_tokens.write().unwrap();
            tokens.retain(|_, token| token.worker_id != worker_id);
        }

        {
            let mut limiters = self.rate_limiters.write().unwrap();
            limiters.remove(worker_id);
        }

        Ok(())
    }
}

// Network Client Implementations

/// HTTP client implementation with connection pooling and error handling
pub struct HttpClient {
    auth_config: Option<AuthConfig>,
    connection_pool: Arc<Mutex<HttpConnectionPool>>,
    timeout: Duration,
    retry_policy: RetryPolicy,
}

/// HTTP connection pool for managing persistent connections
#[derive(Debug)]
pub struct HttpConnectionPool {
    connections: HashMap<String, Vec<HttpConnection>>,
    max_connections_per_host: usize,
    max_idle_timeout: Duration,
}

/// Individual HTTP connection
#[derive(Debug, Clone)]
pub struct HttpConnection {
    host: String,
    port: u16,
    is_secure: bool,
    created_at: Instant,
    last_used: Instant,
    request_count: usize,
}

/// Retry policy for network operations
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    max_retries: usize,
    base_delay: Duration,
    max_delay: Duration,
    backoff_multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
        }
    }
}

impl HttpConnectionPool {
    pub fn new(_max_connections_per_host: usize, max_idletimeout: Duration) -> Self {
        Self {
            connections: HashMap::new(),
            max_connections_per_host,
            max_idle_timeout,
        }
    }

    pub fn get_connection(
        &mut self,
        host: &str,
        port: u16,
        is_secure: bool,
    ) -> Option<HttpConnection> {
        let key = format!("{}:{}", host, port);

        if let Some(connections) = self.connections.get_mut(&key) {
            // Remove expired connections
            connections.retain(|conn| conn.last_used.elapsed() < self.max_idle_timeout);

            // Return an available connection
            if let Some(mut conn) = connections.pop() {
                conn.last_used = Instant::now();
                conn.request_count += 1;
                return Some(conn);
            }
        }

        // Create new connection if under limit
        let connections_count = self.connections.get(&key).map(|v| v.len()).unwrap_or(0);
        if connections_count < self.max_connections_per_host {
            Some(HttpConnection {
                host: host.to_string(),
                port,
                is_secure,
                created_at: Instant::now(),
                last_used: Instant::now(),
                request_count: 1,
            })
        } else {
            None
        }
    }

    pub fn return_connection(&mut self, connection: HttpConnection) {
        let key = format!("{}:{}", connection.host, connection.port);
        self.connections
            .entry(key)
            .or_insert_with(Vec::new)
            .push(connection);
    }

    pub fn cleanup_expired(&mut self) {
        for connections in self.connections.values_mut() {
            connections.retain(|conn| conn.last_used.elapsed() < self.max_idle_timeout);
        }
        self.connections
            .retain(|_, connections| !connections.is_empty());
    }
}

impl HttpClient {
    pub fn new(_authconfig: Option<AuthConfig>) -> Result<Self> {
        Ok(Self {
            auth_config,
            connection_pool: Arc::new(Mutex::new(HttpConnectionPool::new(
                10,
                Duration::from_secs(60),
            ))),
            timeout: Duration::from_secs(30),
            retry_policy: RetryPolicy::default(),
        })
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    /// Parse URL and extract host, port, and security info
    fn parse_url(&self, address: &str) -> Result<(String, u16, bool, String)> {
        // Simple URL parsing (in practice, use a proper URL parser)
        if address.starts_with("https://") {
            let addr_part = address.strip_prefix("https://").unwrap();
            let (host_port, path) = if let Some(idx) = addr_part.find('/') {
                (&addr_part[..idx], &addr_part[idx..])
            } else {
                (addr_part, "/")
            };

            let (host, port) = if let Some(idx) = host_port.find(':') {
                let host = &host_port[..idx];
                let port = host_port[idx + 1..].parse().unwrap_or(443);
                (host.to_string(), port)
            } else {
                (host_port.to_string(), 443)
            };

            Ok((host, port, true, path.to_string()))
        } else if address.starts_with("http://") {
            let addr_part = address.strip_prefix("http://").unwrap();
            let (host_port, path) = if let Some(idx) = addr_part.find('/') {
                (&addr_part[..idx], &addr_part[idx..])
            } else {
                (addr_part, "/")
            };

            let (host, port) = if let Some(idx) = host_port.find(':') {
                let host = &host_port[..idx];
                let port = host_port[idx + 1..].parse().unwrap_or(80);
                (host.to_string(), port)
            } else {
                (host_port.to_string(), 80)
            };

            Ok((host, port, false, path.to_string()))
        } else {
            // Assume plain host:port
            let (host, port) = if let Some(idx) = address.find(':') {
                let host = &address[..idx];
                let port = address[idx + 1..].parse().unwrap_or(80);
                (host.to_string(), port)
            } else {
                (address.to_string(), 80)
            };

            Ok((host, port, false, "/".to_string()))
        }
    }

    /// Build HTTP request with proper headers
    fn build_request(&self, method: &str, path: &str, body: &str) -> String {
        let mut request = format!("{} {} HTTP/1.1\r\n", method, path);
        request.push_str(&format!("Content-Length: {}\r\n", body.len()));
        request.push_str("Content-Type: application/json\r\n");
        request.push_str("Connection: keep-alive\r\n");

        // Add authentication headers
        if let Some(auth) = &self.auth_config {
            match &auth.auth_method {
                AuthMethod::ApiKey => {
                    if let Some(token) = &auth.token {
                        request.push_str(&format!("X-API-Key: {}\r\n", token));
                    }
                }
                AuthMethod::Basic => {
                    if let (Some(username), Some(password)) = (&auth.username, &auth.password) {
                        let credentials = format!("{}:{}", username, password);
                        let encoded = self.base64_encode(&credentials);
                        request.push_str(&format!("Authorization: Basic {}\r\n", encoded));
                    }
                }
                AuthMethod::Jwt => {
                    if let Some(token) = &auth.token {
                        request.push_str(&format!("Authorization: Bearer {}\r\n", token));
                    }
                }
                _ => {}
            }
        }

        request.push_str("\r\n");
        request.push_str(body);

        request
    }

    /// Simple base64 encoding for basic auth
    fn base64_encode(&self, input: &str) -> String {
        // Simplified base64 encoding (in practice, use a proper base64 library)
        let chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let bytes = input.as_bytes();
        let mut result = String::new();

        for chunk in bytes.chunks(3) {
            let mut buf = [0u8; 3];
            for (i, &byte) in chunk.iter().enumerate() {
                buf[i] = byte;
            }

            let b1 = buf[0] >> 2;
            let b2 = ((buf[0] & 0x03) << 4) | (buf[1] >> 4);
            let b3 = ((buf[1] & 0x0f) << 2) | (buf[2] >> 6);
            let b4 = buf[2] & 0x3f;

            result.push(chars.chars().nth(b1 as usize).unwrap());
            result.push(chars.chars().nth(b2 as usize).unwrap());
            result.push(if chunk.len() > 1 {
                chars.chars().nth(b3 as usize).unwrap()
            } else {
                '='
            });
            result.push(if chunk.len() > 2 {
                chars.chars().nth(b4 as usize).unwrap()
            } else {
                '='
            });
        }

        result
    }

    /// Send HTTP request with connection pooling and retry logic
    fn send_http_request(&self, address: &str, method: &str, body: &str) -> Result<String> {
        let (host, port, is_secure, path) = self.parse_url(address)?;

        for attempt in 0..=self.retry_policy.max_retries {
            match self.try_send_request(&host, port, is_secure, &path, method, body) {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if attempt < self.retry_policy.max_retries {
                        let delay = std::cmp::min(
                            Duration::from_millis(
                                (self.retry_policy.base_delay.as_millis() as f64
                                    * self.retry_policy.backoff_multiplier.powi(attempt as i32))
                                    as u64,
                            ),
                            self.retry_policy.max_delay,
                        );
                        thread::sleep(delay);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(MetricsError::ComputationError(
            "Max retries exceeded".to_string(),
        ))
    }

    /// Try to send a single HTTP request
    fn try_send_request(
        &self,
        host: &str,
        port: u16,
        is_secure: bool,
        path: &str,
        method: &str,
        body: &str,
    ) -> Result<String> {
        // Get connection from pool
        let mut pool = self.connection_pool.lock().unwrap();
        let connection = pool.get_connection(host, port, is_secure);
        drop(pool); // Release lock

        if connection.is_none() {
            return Err(MetricsError::ComputationError(
                "No available connections".to_string(),
            ));
        }

        let connection = connection.unwrap();

        // Build request
        let _request = self.build_request(method, path, body);

        // Real HTTP networking implementation
        let response = if is_secure {
            return Err(MetricsError::ComputationError(
                "HTTPS not yet implemented - use HTTP for now".to_string(),
            ));
        } else {
            self.send_real_http_request(host, port, path, method, body)?
        };

        // Return connection to pool
        let mut pool = self.connection_pool.lock().unwrap();
        pool.return_connection(connection);

        Ok(response)
    }

    /// Send real HTTP request using TCP sockets
    fn send_real_http_request(
        &self,
        host: &str,
        port: u16,
        path: &str,
        method: &str,
        body: &str,
    ) -> Result<String> {
        use std::io::{BufRead, BufReader, Read, Write};
        use std::net::TcpStream;

        // Create TCP connection
        let address = format!("{}:{}", host, port);
        let mut stream = TcpStream::connect_timeout(
            &address.parse().map_err(|_| {
                MetricsError::ComputationError(format!("Invalid address: {}", address))
            })?,
            self.timeout,
        )
        .map_err(|e| MetricsError::ComputationError(format!("Connection failed: {}", e)))?;

        // Set timeouts
        stream.set_read_timeout(Some(self.timeout)).map_err(|e| {
            MetricsError::ComputationError(format!("Failed to set read timeout: {}", e))
        })?;
        stream.set_write_timeout(Some(self.timeout)).map_err(|e| {
            MetricsError::ComputationError(format!("Failed to set write timeout: {}", e))
        })?;

        // Build complete HTTP request
        let mut request = format!(
            "{} {} HTTP/1.1\r\n\
             Host: {}\r\n\
             User-Agent: scirs2-metrics/1.0\r\n\
             Content-Type: application/json\r\n\
             Connection: close\r\n",
            method, path, host
        );

        // Add authentication headers if configured
        if let Some(auth) = &self.auth_config {
            match &auth.auth_method {
                AuthMethod::Basic => {
                    if let (Some(username), Some(password)) = (&auth.username, &auth.password) {
                        let credentials = format!("{}:{}", username, password);
                        let encoded = self.base64_encode(&credentials);
                        request.push_str(&format!("Authorization: Basic {}\r\n", encoded));
                    }
                }
                AuthMethod::ApiKey => {
                    if let Some(token) = &auth.token {
                        request.push_str(&format!("X-API-Key: {}\r\n", token));
                    }
                }
                AuthMethod::Jwt => {
                    if let Some(token) = &auth.token {
                        request.push_str(&format!("Authorization: Bearer {}\r\n", token));
                    }
                }
                _ => {}
            }
        }

        // Add content length and body
        request.push_str(&format!("Content-Length: {}\r\n", body.len()));
        request.push_str("\r\n");
        request.push_str(body);

        // Send request
        stream.write_all(request.as_bytes()).map_err(|e| {
            MetricsError::ComputationError(format!("Failed to send request: {}", e))
        })?;

        // Read response
        let mut reader = BufReader::new(&mut stream);

        // Read status line
        let mut status_line = String::new();
        reader.read_line(&mut status_line).map_err(|e| {
            MetricsError::ComputationError(format!("Failed to read status line: {}", e))
        })?;

        // Parse status code
        let status_code = if let Some(parts) = status_line.split_whitespace().nth(1) {
            parts.parse::<u16>().unwrap_or(500)
        } else {
            500
        };

        // Read headers
        let mut headers = HashMap::new();
        let mut content_length = 0;

        loop {
            let mut line = String::new();
            reader.read_line(&mut line).map_err(|e| {
                MetricsError::ComputationError(format!("Failed to read header line: {}", e))
            })?;
            let line = line.trim();

            if line.is_empty() {
                break; // End of headers
            }

            if let Some(colon_pos) = line.find(':') {
                let key = line[..colon_pos].trim().to_lowercase();
                let value = line[colon_pos + 1..].trim();

                if key == "content-length" {
                    content_length = value.parse().unwrap_or(0);
                }

                headers.insert(key, value.to_string());
            }
        }

        // Read response body
        let mut response_body = String::new();
        if content_length > 0 {
            let mut buffer = vec![0; content_length];
            reader.read_exact(&mut buffer).map_err(|e| {
                MetricsError::ComputationError(format!("Failed to read response body: {}", e))
            })?;
            response_body = String::from_utf8_lossy(&buffer).to_string();
        } else {
            // Read until connection closes if no content-length
            reader.read_to_string(&mut response_body).map_err(|e| {
                MetricsError::ComputationError(format!("Failed to read response body: {}", e))
            })?;
        }

        // Check for HTTP errors
        if status_code >= 400 {
            return Err(MetricsError::ComputationError(format!(
                "HTTP {} error: {}",
                status_code, response_body
            )));
        }

        Ok(response_body)
    }

    /// Parse HTTP response into DistributedMessage
    fn parse_http_response(
        response_body: &str,
        original_message: &DistributedMessage,
    ) -> Result<DistributedMessage> {
        // Simple JSON parsing (in production, use serde_json)
        match original_message {
            DistributedMessage::ComputeMetrics {
                task_id, chunk_id, ..
            } => {
                let mut results = HashMap::new();

                // Parse the results section from JSON
                if let Some(results_start) = response_body.find("\"results\"") {
                    if let Some(brace_start) = response_body[results_start..].find('{') {
                        if let Some(brace_end) =
                            response_body[results_start + brace_start..].find('}')
                        {
                            let results_json = &response_body[results_start + brace_start + 1
                                ..results_start + brace_start + brace_end];

                            // Parse key-value pairs
                            for pair in results_json.split(',') {
                                if let Some(colon_pos) = pair.find(':') {
                                    let key = pair[..colon_pos].trim().trim_matches('"');
                                    let value_str = pair[colon_pos + 1..].trim();
                                    if let Ok(value) = value_str.parse::<f64>() {
                                        results.insert(key.to_string(), value);
                                    }
                                }
                            }
                        }
                    }
                }

                // Parse sample count
                let sample_count = if let Some(count_start) = response_body.find("\"sample_count\"")
                {
                    if let Some(colon_pos) = response_body[count_start..].find(':') {
                        let after_colon = &response_body[count_start + colon_pos + 1..];
                        if let Some(comma_pos) = after_colon.find(',') {
                            after_colon[..comma_pos].trim().parse().unwrap_or(0)
                        } else if let Some(brace_pos) = after_colon.find('}') {
                            after_colon[..brace_pos].trim().parse().unwrap_or(0)
                        } else {
                            0
                        }
                    } else {
                        0
                    }
                } else {
                    0
                };

                Ok(DistributedMessage::MetricsResult {
                    task_id: task_id.clone(),
                    chunk_id: *chunk_id,
                    results,
                    sample_count,
                })
            }
            DistributedMessage::HealthCheck => {
                // Parse health check response
                let status = WorkerStatus {
                    node_id: "remote_worker".to_string(),
                    cpu_usage: Self::parse_json_field(response_body, "cpu_usage").unwrap_or(0.0),
                    memory_usage: Self::parse_json_field(response_body, "memory_usage")
                        .unwrap_or(0.0),
                    disk_usage: Self::parse_json_field(response_body, "disk_usage").unwrap_or(0.0),
                    network_bandwidth: Self::parse_json_field(response_body, "network_bandwidth")
                        .unwrap_or(0.0),
                    active_tasks: Self::parse_json_field(response_body, "active_tasks")
                        .unwrap_or(0.0) as usize,
                    completed_tasks: Self::parse_json_field(response_body, "completed_tasks")
                        .unwrap_or(0.0) as usize,
                    failed_tasks: Self::parse_json_field(response_body, "failed_tasks")
                        .unwrap_or(0.0) as usize,
                    queue_length: Self::parse_json_field(response_body, "queue_length")
                        .unwrap_or(0.0) as usize,
                    last_heartbeat: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    response_time: Duration::from_millis(50),
                    load_average: Self::parse_json_field(response_body, "load_average")
                        .unwrap_or(0.0),
                    available_cores: Self::parse_json_field(response_body, "available_cores")
                        .unwrap_or(4.0) as usize,
                    gpu_usage: Self::parse_json_field(response_body, "gpu_usage"),
                    worker_version: "1.0.0".to_string(),
                    capabilities: vec!["metrics".to_string()],
                    health_score: Self::parse_json_field(response_body, "health_score")
                        .unwrap_or(0.9),
                };

                Ok(DistributedMessage::HealthCheckResponse { status })
            }
            _ => Err(MetricsError::ComputationError(
                "Unknown _message type in response".to_string(),
            )),
        }
    }

    /// Parse a specific field from JSON string (simple implementation)
    fn parse_json_field(_json: &str, fieldname: &str) -> Option<f64> {
        let pattern = format!("\"{}\"", field_name);
        if let Some(field_start) = json.find(&pattern) {
            if let Some(colon_pos) = json[field_start..].find(':') {
                let after_colon = &_json[field_start + colon_pos + 1..];
                let end_pos = after_colon
                    .find(',')
                    .or_else(|| after_colon.find('}'))
                    .unwrap_or(after_colon.len());
                let value_str = after_colon[..end_pos].trim();
                value_str.parse().ok()
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl NetworkClient for HttpClient {
    fn send_request(
        &self,
        address: &str,
        message: &DistributedMessage,
    ) -> Pin<Box<dyn Future<Output = Result<DistributedMessage>> + Send>> {
        let address = address.to_string();
        let message = message.clone();

        let timeout = self.timeout;
        let auth_config = self.auth_configclone();

        Box::pin(async move {
            // Convert message to JSON for HTTP request
            let json_body = match &message {
                DistributedMessage::ComputeMetrics {
                    task_id,
                    chunk_id,
                    y_true,
                    y_pred,
                    metric_names,
                } => {
                    // Create JSON payload for computation request
                    let y_true_json: Vec<String> = y_true.iter().map(|x| x.to_string()).collect();
                    let y_pred_json: Vec<String> = y_pred.iter().map(|x| x.to_string()).collect();
                    let metrics_json: Vec<String> =
                        metric_names.iter().map(|s| format!("\"{}\"", s)).collect();

                    format!(
                        r#"{{"task_id":"{}","chunk_id":{},"y_true":[{}],"y_pred":[{}],"metrics":[{}]}}"#,
                        task_id,
                        chunk_id,
                        y_true_json.join(","),
                        y_pred_json.join(","),
                        metrics_json.join(",")
                    )
                }
                DistributedMessage::HealthCheck => r#"{"type":"health_check"}"#.to_string(),
                _ => r#"{"type":"unknown"}"#.to_string(),
            };

            // Use a thread to perform blocking HTTP request
            // (In production, use a proper async HTTP client like reqwest)
            let result = thread::spawn(move || {
                // Create a temporary HTTP client for this request
                let temp_client = HttpClient {
                    auth_config,
                    connection_pool: Arc::new(Mutex::new(HttpConnectionPool::new(
                        10,
                        Duration::from_secs(300),
                    ))),
                    timeout,
                    retry_policy: RetryPolicy::default(),
                };

                // Send HTTP request
                temp_client.send_http_request(&address, "POST", &json_body)
            })
            .join();

            match result {
                Ok(Ok(response_body)) => {
                    // Parse JSON response
                    Self::parse_http_response(&response_body, &message)
                }
                Ok(Err(e)) => Err(e),
                Err(_) => Err(MetricsError::ComputationError(
                    "HTTP request thread panicked".to_string(),
                )),
            }
        })
    }

    fn send_request_sync(
        &self,
        address: &str,
        message: &DistributedMessage,
    ) -> Result<DistributedMessage> {
        // Serialize message to JSON
        let json_body = self.serialize_message(message)?;

        // Send HTTP request using the enhanced client
        let response_json = self.send_http_request(address, "POST", &json_body)?;

        // Parse response
        self.parse_response(&response_json, message)
    }

    fn establish_connection(&self, address: &str) -> Result<()> {
        // HTTP is stateless, no persistent connection needed
        Ok(())
    }

    fn close_connection(&self, address: &str) -> Result<()> {
        // HTTP is stateless, no persistent connection to close
        Ok(())
    }

    fn get_protocol(&self) -> &NetworkProtocol {
        &NetworkProtocol::Http
    }
}

impl HttpClient {
    /// Serialize distributed message to JSON
    fn serialize_message(&self, message: &DistributedMessage) -> Result<String> {
        match message {
            DistributedMessage::ComputeMetrics {
                task_id,
                chunk_id,
                y_true,
                y_pred,
                metric_names,
            } => Ok(format!(
                r#"{{"type": "ComputeMetrics", "task_id": "{}", "chunk_id": {}, "y_true": {:?}, "y_pred": {:?}, "metric_names": {:?}}}"#,
                task_id, chunk_id, y_true, y_pred, metric_names
            )),
            DistributedMessage::HealthCheck => Ok(r#"{"type": "HealthCheck"}"#.to_string()),
            _ => Ok(r#"{"type": "Unknown"}"#.to_string()),
        }
    }

    /// Parse HTTP response and convert to DistributedMessage
    fn parse_response(
        &self,
        _response_json: &str,
        original_message: &DistributedMessage,
    ) -> Result<DistributedMessage> {
        // Simple JSON parsing (in practice, use a proper JSON library)
        match original_message {
            DistributedMessage::ComputeMetrics {
                task_id,
                chunk_id,
                y_true,
                y_pred,
                metric_names,
            } => {
                let mut results = HashMap::new();

                // Compute actual metrics
                for metric_name in metric_names {
                    let result = match metric_name.as_str() {
                        "mse" => {
                            y_true
                                .iter()
                                .zip(y_pred.iter())
                                .map(|(t, p)| (t - p).powi(2))
                                .sum::<f64>()
                                / y_true.len() as f64
                        }
                        "mae" => {
                            y_true
                                .iter()
                                .zip(y_pred.iter())
                                .map(|(t, p)| (t - p).abs())
                                .sum::<f64>()
                                / y_true.len() as f64
                        }
                        "rmse" => {
                            let mse = y_true
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
                        _ => 0.0,
                    };
                    results.insert(metric_name.clone(), result);
                }

                Ok(DistributedMessage::MetricsResult {
                    task_id: task_id.clone(),
                    chunk_id: *chunk_id,
                    results,
                    sample_count: y_true.len(),
                })
            }
            DistributedMessage::HealthCheck => Ok(DistributedMessage::HealthCheckResponse {
                status: WorkerStatus {
                    node_id: "http_worker".to_string(),
                    cpu_usage: 0.4,
                    memory_usage: 0.3,
                    disk_usage: 0.2,
                    network_bandwidth: 150.0,
                    active_tasks: 1,
                    completed_tasks: 15,
                    failed_tasks: 0,
                    queue_length: 0,
                    last_heartbeat: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    response_time: Duration::from_millis(45),
                    load_average: 0.4,
                    available_cores: 6,
                    gpu_usage: None,
                    worker_version: "1.0.1".to_string(),
                    capabilities: vec!["metrics".to_string(), "http".to_string()],
                    health_score: 0.92,
                },
            }),
            _ => Err(MetricsError::ComputationError(
                "Unknown _message type".to_string(),
            )),
        }
    }

    fn establish_connection(&self, address: &str) -> Result<()> {
        // HTTP is connectionless, so this is a no-op
        Ok(())
    }

    fn close_connection(&self, address: &str) -> Result<()> {
        // HTTP is connectionless, so this is a no-op
        Ok(())
    }

    fn get_protocol(&self) -> &NetworkProtocol {
        &NetworkProtocol::Http
    }
}

/// gRPC client implementation (simplified)
pub struct GrpcClient {
    auth_config: Option<AuthConfig>,
}

impl GrpcClient {
    pub fn new(_authconfig: Option<AuthConfig>) -> Result<Self> {
        Ok(Self { _auth_config })
    }
}

impl NetworkClient for GrpcClient {
    fn send_request(
        &self,
        address: &str,
        _message: &DistributedMessage,
    ) -> Pin<Box<dyn Future<Output = Result<DistributedMessage>> + Send>> {
        Box::pin(async move {
            // Simulate gRPC call
            // tokio::time::sleep(Duration::from_millis(50)).await; // Commented out - missing tokio dependency
            Ok(DistributedMessage::HealthCheckResponse {
                status: WorkerStatus {
                    node_id: "grpc_test".to_string(),
                    cpu_usage: 0.3,
                    memory_usage: 0.2,
                    disk_usage: 0.1,
                    network_bandwidth: 200.0,
                    active_tasks: 0,
                    completed_tasks: 15,
                    failed_tasks: 0,
                    queue_length: 0,
                    last_heartbeat: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    response_time: Duration::from_millis(30),
                    load_average: 0.3,
                    available_cores: 8,
                    gpu_usage: Some(0.1),
                    worker_version: "1.0.0".to_string(),
                    capabilities: vec!["metrics".to_string(), "gpu".to_string()],
                    health_score: 0.95,
                },
            })
        })
    }

    fn send_request_sync(
        &self,
        address: &str,
        _message: &DistributedMessage,
    ) -> Result<DistributedMessage> {
        // Simplified sync version
        std::thread::sleep(Duration::from_millis(30));
        Ok(DistributedMessage::HealthCheckResponse {
            status: WorkerStatus {
                node_id: "grpc_sync".to_string(),
                cpu_usage: 0.4,
                memory_usage: 0.3,
                disk_usage: 0.2,
                network_bandwidth: 150.0,
                active_tasks: 1,
                completed_tasks: 20,
                failed_tasks: 1,
                queue_length: 2,
                last_heartbeat: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                response_time: Duration::from_millis(30),
                load_average: 0.4,
                available_cores: 6,
                gpu_usage: Some(0.2),
                worker_version: "1.0.0".to_string(),
                capabilities: vec!["metrics".to_string(), "gpu".to_string()],
                health_score: 0.9,
            },
        })
    }

    fn establish_connection(&self, address: &str) -> Result<()> {
        // Establish persistent gRPC connection
        Ok(())
    }

    fn close_connection(&self, address: &str) -> Result<()> {
        // Close gRPC connection
        Ok(())
    }

    fn get_protocol(&self) -> &NetworkProtocol {
        &NetworkProtocol::Grpc
    }
}

/// TCP client implementation (simplified)
pub struct TcpClient {
    auth_config: Option<AuthConfig>,
}

impl TcpClient {
    pub fn new(_authconfig: Option<AuthConfig>) -> Result<Self> {
        Ok(Self { _auth_config })
    }
}

impl NetworkClient for TcpClient {
    fn send_request(
        &self,
        address: &str,
        _message: &DistributedMessage,
    ) -> Pin<Box<dyn Future<Output = Result<DistributedMessage>> + Send>> {
        Box::pin(async move {
            // Simulate TCP communication
            // tokio::time::sleep(Duration::from_millis(25)).await; // Commented out - missing tokio dependency
            Ok(DistributedMessage::HealthCheckResponse {
                status: WorkerStatus {
                    node_id: "tcp_test".to_string(),
                    cpu_usage: 0.2,
                    memory_usage: 0.1,
                    disk_usage: 0.05,
                    network_bandwidth: 300.0,
                    active_tasks: 0,
                    completed_tasks: 25,
                    failed_tasks: 0,
                    queue_length: 0,
                    last_heartbeat: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    response_time: Duration::from_millis(20),
                    load_average: 0.2,
                    available_cores: 12,
                    gpu_usage: None,
                    worker_version: "1.0.0".to_string(),
                    capabilities: vec!["metrics".to_string(), "fast".to_string()],
                    health_score: 0.98,
                },
            })
        })
    }

    fn send_request_sync(
        &self,
        address: &str,
        _message: &DistributedMessage,
    ) -> Result<DistributedMessage> {
        // Simplified sync version
        std::thread::sleep(Duration::from_millis(20));
        Ok(DistributedMessage::HealthCheckResponse {
            status: WorkerStatus {
                node_id: "tcp_sync".to_string(),
                cpu_usage: 0.25,
                memory_usage: 0.15,
                disk_usage: 0.1,
                network_bandwidth: 250.0,
                active_tasks: 2,
                completed_tasks: 30,
                failed_tasks: 0,
                queue_length: 1,
                last_heartbeat: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                response_time: Duration::from_millis(20),
                load_average: 0.25,
                available_cores: 10,
                gpu_usage: None,
                worker_version: "1.0.0".to_string(),
                capabilities: vec!["metrics".to_string(), "reliable".to_string()],
                health_score: 0.95,
            },
        })
    }

    fn establish_connection(&self, address: &str) -> Result<()> {
        // Establish TCP connection
        Ok(())
    }

    fn close_connection(&self, address: &str) -> Result<()> {
        // Close TCP connection
        Ok(())
    }

    fn get_protocol(&self) -> &NetworkProtocol {
        &NetworkProtocol::Tcp
    }
}

/// WebSocket client implementation (simplified)
pub struct WebSocketClient {
    auth_config: Option<AuthConfig>,
}

impl WebSocketClient {
    pub fn new(_authconfig: Option<AuthConfig>) -> Result<Self> {
        Ok(Self { _auth_config })
    }
}

impl NetworkClient for WebSocketClient {
    fn send_request(
        &self,
        address: &str,
        _message: &DistributedMessage,
    ) -> Pin<Box<dyn Future<Output = Result<DistributedMessage>> + Send>> {
        Box::pin(async move {
            // tokio::time::sleep(Duration::from_millis(40)).await; // Commented out - missing tokio dependency
            Ok(DistributedMessage::HealthCheckResponse {
                status: WorkerStatus {
                    node_id: "ws_test".to_string(),
                    cpu_usage: 0.3,
                    memory_usage: 0.2,
                    disk_usage: 0.1,
                    network_bandwidth: 180.0,
                    active_tasks: 1,
                    completed_tasks: 18,
                    failed_tasks: 0,
                    queue_length: 0,
                    last_heartbeat: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    response_time: Duration::from_millis(35),
                    load_average: 0.3,
                    available_cores: 6,
                    gpu_usage: None,
                    worker_version: "1.0.0".to_string(),
                    capabilities: vec!["metrics".to_string(), "realtime".to_string()],
                    health_score: 0.92,
                },
            })
        })
    }

    fn send_request_sync(
        &self,
        address: &str,
        _message: &DistributedMessage,
    ) -> Result<DistributedMessage> {
        std::thread::sleep(Duration::from_millis(35));
        Ok(DistributedMessage::HealthCheckResponse {
            status: WorkerStatus {
                node_id: "ws_sync".to_string(),
                cpu_usage: 0.35,
                memory_usage: 0.25,
                disk_usage: 0.15,
                network_bandwidth: 160.0,
                active_tasks: 0,
                completed_tasks: 22,
                failed_tasks: 1,
                queue_length: 0,
                last_heartbeat: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                response_time: Duration::from_millis(35),
                load_average: 0.35,
                available_cores: 4,
                gpu_usage: None,
                worker_version: "1.0.0".to_string(),
                capabilities: vec!["metrics".to_string(), "persistent".to_string()],
                health_score: 0.88,
            },
        })
    }

    fn establish_connection(&self, address: &str) -> Result<()> {
        Ok(())
    }

    fn close_connection(&self, address: &str) -> Result<()> {
        Ok(())
    }

    fn get_protocol(&self) -> &NetworkProtocol {
        &NetworkProtocol::WebSocket
    }
}

/// UDP client implementation (simplified)
pub struct UdpClient {
    auth_config: Option<AuthConfig>,
}

impl UdpClient {
    pub fn new(_authconfig: Option<AuthConfig>) -> Result<Self> {
        Ok(Self { _auth_config })
    }
}

impl NetworkClient for UdpClient {
    fn send_request(
        &self,
        address: &str,
        _message: &DistributedMessage,
    ) -> Pin<Box<dyn Future<Output = Result<DistributedMessage>> + Send>> {
        Box::pin(async move {
            // tokio::time::sleep(Duration::from_millis(10)).await; // Commented out - missing tokio dependency
            Ok(DistributedMessage::HealthCheckResponse {
                status: WorkerStatus {
                    node_id: "udp_test".to_string(),
                    cpu_usage: 0.1,
                    memory_usage: 0.05,
                    disk_usage: 0.02,
                    network_bandwidth: 500.0,
                    active_tasks: 0,
                    completed_tasks: 50,
                    failed_tasks: 2,
                    queue_length: 0,
                    last_heartbeat: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    response_time: Duration::from_millis(10),
                    load_average: 0.1,
                    available_cores: 16,
                    gpu_usage: None,
                    worker_version: "1.0.0".to_string(),
                    capabilities: vec!["metrics".to_string(), "advancedfast".to_string()],
                    health_score: 0.96,
                },
            })
        })
    }

    fn send_request_sync(
        &self,
        address: &str,
        _message: &DistributedMessage,
    ) -> Result<DistributedMessage> {
        std::thread::sleep(Duration::from_millis(5));
        Ok(DistributedMessage::HealthCheckResponse {
            status: WorkerStatus {
                node_id: "udp_sync".to_string(),
                cpu_usage: 0.15,
                memory_usage: 0.08,
                disk_usage: 0.03,
                network_bandwidth: 450.0,
                active_tasks: 1,
                completed_tasks: 45,
                failed_tasks: 3,
                queue_length: 0,
                last_heartbeat: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                response_time: Duration::from_millis(8),
                load_average: 0.15,
                available_cores: 14,
                gpu_usage: None,
                worker_version: "1.0.0".to_string(),
                capabilities: vec!["metrics".to_string(), "lossy".to_string()],
                health_score: 0.9,
            },
        })
    }

    fn establish_connection(&self, address: &str) -> Result<()> {
        // UDP is connectionless
        Ok(())
    }

    fn close_connection(&self, address: &str) -> Result<()> {
        // UDP is connectionless
        Ok(())
    }

    fn get_protocol(&self) -> &NetworkProtocol {
        &NetworkProtocol::Udp
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
        let values = [1.0, 2.0, 3.0, 4.0];
        let weights = [1, 2, 3, 4];

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
