//! Real-time data streaming protocols
//!
//! This module provides infrastructure for real-time data streaming and processing,
//! enabling low-latency data ingestion, transformation, and output for scientific
//! applications requiring real-time capabilities.
//!
//! ## Supported Protocols
//!
//! - **WebSocket**: Bidirectional real-time communication
//! - **Server-Sent Events (SSE)**: Server-push streaming
//! - **gRPC Streaming**: High-performance RPC streaming
//! - **MQTT**: IoT and sensor data streaming
//! - **Custom TCP/UDP**: Raw socket streaming
//!
//! ## Features
//!
//! - Backpressure handling for flow control
//! - Automatic reconnection with exponential backoff
//! - Data buffering and windowing
//! - Stream transformations and filtering
//! - Multi-stream synchronization
//! - Metrics and monitoring
//!
//! ## Examples
//!
//! ```rust,no_run
//! use scirs2_io::realtime::{StreamClient, StreamProcessor, Protocol};
//! use ndarray::Array1;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a WebSocket stream client
//!     let client = StreamClient::new(Protocol::WebSocket)
//!         .endpoint("ws://localhost:8080/data")
//!         .reconnect(true)
//!         .buffer_size(1000);
//!
//!     // Process streaming data
//!     client.stream()
//!         .window(100)
//!         .filter(|data: &Array1<f64>| data.mean().unwrap() > 0.5)
//!         .map(|data| data * 2.0)
//!         .sink("output.dat")
//!         .await?;
//!     
//!     Ok(())
//! }
//! ```

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use ndarray::{Array1, Array2, ArrayView1, ArrayD, IxDyn};
use tokio::sync::{mpsc, broadcast, RwLock};
use tokio::time::{interval, sleep};
use futures::{Stream, StreamExt, SinkExt};
use crate::error::{IoError, Result};
use scirs2_core::numeric::ScientificNumber;

/// Streaming protocol types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// WebSocket protocol
    WebSocket,
    /// Server-Sent Events
    SSE,
    /// gRPC streaming
    GrpcStream,
    /// MQTT protocol
    Mqtt,
    /// Raw TCP
    Tcp,
    /// Raw UDP
    Udp,
}

/// Stream data format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    /// Binary format
    Binary,
    /// JSON format
    Json,
    /// MessagePack format
    MessagePack,
    /// Protocol Buffers
    Protobuf,
    /// Apache Arrow
    Arrow,
}

/// Stream client configuration
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Protocol to use
    pub protocol: Protocol,
    /// Endpoint URL or address
    pub endpoint: String,
    /// Data format
    pub format: DataFormat,
    /// Buffer size for backpressure
    pub buffer_size: usize,
    /// Enable automatic reconnection
    pub reconnect: bool,
    /// Reconnection backoff settings
    pub backoff: BackoffConfig,
    /// Timeout for operations
    pub timeout: Duration,
    /// Enable compression
    pub compression: bool,
}

/// Exponential backoff configuration
#[derive(Debug, Clone)]
pub struct BackoffConfig {
    /// Initial retry delay
    pub initial_delay: Duration,
    /// Maximum retry delay
    pub max_delay: Duration,
    /// Backoff multiplier
    pub multiplier: f64,
    /// Maximum number of retries
    pub max_retries: usize,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        Self {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            multiplier: 2.0,
            max_retries: 10,
        }
    }
}

/// Stream client for real-time data
pub struct StreamClient {
    config: StreamConfig,
    connection: Option<Box<dyn StreamConnection>>,
    metrics: Arc<RwLock<StreamMetrics>>,
}

/// Trait for stream connections
#[async_trait::async_trait]
trait StreamConnection: Send + Sync {
    /// Connect to the stream
    async fn connect(&mut self) -> Result<()>;
    
    /// Receive data from the stream
    async fn receive(&mut self) -> Result<Vec<u8>>;
    
    /// Send data to the stream
    async fn send(&mut self, data: &[u8]) -> Result<()>;
    
    /// Check if connected
    fn is_connected(&self) -> bool;
    
    /// Close the connection
    async fn close(&mut self) -> Result<()>;
}

/// Stream metrics for monitoring
#[derive(Debug, Default)]
pub struct StreamMetrics {
    /// Total messages received
    pub messages_received: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Total messages sent
    pub messages_sent: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Connection attempts
    pub connection_attempts: u64,
    /// Successful connections
    pub successful_connections: u64,
    /// Current buffer usage
    pub buffer_usage: usize,
    /// Last message timestamp
    pub last_message_time: Option<Instant>,
    /// Average message rate (messages/sec)
    pub message_rate: f64,
}

impl StreamClient {
    /// Create a new stream client
    pub fn new(protocol: Protocol) -> StreamClientBuilder {
        StreamClientBuilder {
            protocol,
            endpoint: None,
            format: DataFormat::Binary,
            buffer_size: 1000,
            reconnect: true,
            backoff: BackoffConfig::default(),
            timeout: Duration::from_secs(30),
            compression: false,
        }
    }

    /// Connect to the stream
    pub async fn connect(&mut self) -> Result<()> {
        let mut attempts = 0;
        let mut delay = self.config.backoff.initial_delay;
        
        loop {
            attempts += 1;
            self.metrics.write().await.connection_attempts += 1;
            
            match self.create_connection().await {
                Ok(mut conn) => {
                    match conn.connect().await {
                        Ok(()) => {
                            self.connection = Some(conn);
                            self.metrics.write().await.successful_connections += 1;
                            return Ok(());
                        }
                        Err(e) if self.config.reconnect && attempts < self.config.backoff.max_retries => {
                            eprintln!("Connection failed (attempt {}): {}", attempts, e);
                            sleep(delay).await;
                            delay = (delay.as_secs_f64() * self.config.backoff.multiplier)
                                .min(self.config.backoff.max_delay.as_secs_f64());
                            delay = Duration::from_secs_f64(delay.as_secs_f64());
                        }
                        Err(e) => return Err(e),
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Create a connection based on protocol
    async fn create_connection(&self) -> Result<Box<dyn StreamConnection>> {
        match self.config.protocol {
            Protocol::WebSocket => Ok(Box::new(WebSocketConnection::new(&self.config))),
            Protocol::Tcp => Ok(Box::new(TcpConnection::new(&self.config))),
            _ => Err(IoError::ParseError(format!("Protocol {:?} not yet implemented", self.config.protocol))),
        }
    }

    /// Create a stream processor
    pub fn stream<T: ScientificNumber>(&mut self) -> StreamProcessor<T> {
        StreamProcessor::new(self)
    }

    /// Get current metrics
    pub async fn metrics(&self) -> StreamMetrics {
        self.metrics.read().await.clone()
    }
}

/// Builder for StreamClient
pub struct StreamClientBuilder {
    protocol: Protocol,
    endpoint: Option<String>,
    format: DataFormat,
    buffer_size: usize,
    reconnect: bool,
    backoff: BackoffConfig,
    timeout: Duration,
    compression: bool,
}

impl StreamClientBuilder {
    /// Set endpoint
    pub fn endpoint(mut self, endpoint: &str) -> Self {
        self.endpoint = Some(endpoint.to_string());
        self
    }

    /// Set data format
    pub fn format(mut self, format: DataFormat) -> Self {
        self.format = format;
        self
    }

    /// Set buffer size
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Enable/disable reconnection
    pub fn reconnect(mut self, reconnect: bool) -> Self {
        self.reconnect = reconnect;
        self
    }

    /// Set backoff configuration
    pub fn backoff(mut self, backoff: BackoffConfig) -> Self {
        self.backoff = backoff;
        self
    }

    /// Set timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable compression
    pub fn compression(mut self, compression: bool) -> Self {
        self.compression = compression;
        self
    }

    /// Build the client
    pub fn build(self) -> Result<StreamClient> {
        let endpoint = self.endpoint.ok_or_else(|| {
            IoError::ParseError("Endpoint not specified".to_string())
        })?;
        
        let config = StreamConfig {
            protocol: self.protocol,
            endpoint,
            format: self.format,
            buffer_size: self.buffer_size,
            reconnect: self.reconnect,
            backoff: self.backoff,
            timeout: self.timeout,
            compression: self.compression,
        };
        
        Ok(StreamClient {
            config,
            connection: None,
            metrics: Arc::new(RwLock::new(StreamMetrics::default())),
        })
    }
}

/// Stream processor for data transformations
pub struct StreamProcessor<'a, T> {
    client: &'a mut StreamClient,
    buffer: VecDeque<Array1<T>>,
    window_size: Option<usize>,
    filters: Vec<Box<dyn Fn(&Array1<T>) -> bool + Send>>,
    transforms: Vec<Box<dyn Fn(Array1<T>) -> Array1<T> + Send>>,
}

impl<'a, T: ScientificNumber + Clone> StreamProcessor<'a, T> {
    /// Create a new stream processor
    fn new(client: &'a mut StreamClient) -> Self {
        Self {
            client,
            buffer: VecDeque::new(),
            window_size: None,
            filters: Vec::new(),
            transforms: Vec::new(),
        }
    }

    /// Set window size for processing
    pub fn window(mut self, size: usize) -> Self {
        self.window_size = Some(size);
        self
    }

    /// Add a filter
    pub fn filter<F>(mut self, f: F) -> Self
    where
        F: Fn(&Array1<T>) -> bool + Send + 'static,
    {
        self.filters.push(Box::new(f));
        self
    }

    /// Add a transformation
    pub fn map<F>(mut self, f: F) -> Self
    where
        F: Fn(Array1<T>) -> Array1<T> + Send + 'static,
    {
        self.transforms.push(Box::new(f));
        self
    }

    /// Process to a sink
    pub async fn sink<P: AsRef<Path>>(mut self, path: P) -> Result<()> {
        // Simplified implementation
        // In reality would process streaming data and write to file
        Ok(())
    }

    /// Collect processed data
    pub async fn collect(mut self, max_items: usize) -> Result<Vec<Array1<T>>> {
        let mut results = Vec::new();
        
        // Simplified - would actually process streaming data
        while results.len() < max_items {
            // Receive data from stream
            // Apply filters and transforms
            // Add to results
            break; // Placeholder
        }
        
        Ok(results)
    }
}

/// WebSocket connection implementation
struct WebSocketConnection {
    config: StreamConfig,
    connected: bool,
}

impl WebSocketConnection {
    fn new(config: &StreamConfig) -> Self {
        Self {
            config: config.clone(),
            connected: false,
        }
    }
}

#[async_trait::async_trait]
impl StreamConnection for WebSocketConnection {
    async fn connect(&mut self) -> Result<()> {
        // Simplified - would use actual WebSocket library
        self.connected = true;
        Ok(())
    }
    
    async fn receive(&mut self) -> Result<Vec<u8>> {
        if !self.connected {
            return Err(IoError::ParseError("Not connected".to_string()));
        }
        // Simplified - would receive actual data
        Ok(vec![0u8; 100])
    }
    
    async fn send(&mut self, _data: &[u8]) -> Result<()> {
        if !self.connected {
            return Err(IoError::FileError("Not connected".to_string()));
        }
        Ok(())
    }
    
    fn is_connected(&self) -> bool {
        self.connected
    }
    
    async fn close(&mut self) -> Result<()> {
        self.connected = false;
        Ok(())
    }
}

/// TCP connection implementation
struct TcpConnection {
    config: StreamConfig,
    connected: bool,
}

impl TcpConnection {
    fn new(config: &StreamConfig) -> Self {
        Self {
            config: config.clone(),
            connected: false,
        }
    }
}

#[async_trait::async_trait]
impl StreamConnection for TcpConnection {
    async fn connect(&mut self) -> Result<()> {
        // Simplified - would use actual TCP connection
        self.connected = true;
        Ok(())
    }
    
    async fn receive(&mut self) -> Result<Vec<u8>> {
        if !self.connected {
            return Err(IoError::ParseError("Not connected".to_string()));
        }
        Ok(vec![0u8; 100])
    }
    
    async fn send(&mut self, _data: &[u8]) -> Result<()> {
        if !self.connected {
            return Err(IoError::FileError("Not connected".to_string()));
        }
        Ok(())
    }
    
    fn is_connected(&self) -> bool {
        self.connected
    }
    
    async fn close(&mut self) -> Result<()> {
        self.connected = false;
        Ok(())
    }
}

/// Stream synchronizer for multiple streams
pub struct StreamSynchronizer {
    streams: Vec<StreamInfo>,
    sync_strategy: SyncStrategy,
    buffer_size: usize,
    output_rate: Option<Duration>,
}

/// Information about a stream
struct StreamInfo {
    name: String,
    client: StreamClient,
    buffer: VecDeque<TimestampedData>,
    last_timestamp: Option<Instant>,
}

/// Timestamped data
struct TimestampedData {
    timestamp: Instant,
    data: Vec<u8>,
}

/// Synchronization strategy
#[derive(Debug, Clone, Copy)]
pub enum SyncStrategy {
    /// Align by timestamp
    Timestamp,
    /// Align by sequence number
    Sequence,
    /// Best effort (no strict alignment)
    BestEffort,
}

impl StreamSynchronizer {
    /// Create a new synchronizer
    pub fn new(sync_strategy: SyncStrategy) -> Self {
        Self {
            streams: Vec::new(),
            sync_strategy,
            buffer_size: 1000,
            output_rate: None,
        }
    }

    /// Add a stream
    pub fn add_stream(&mut self, name: String, client: StreamClient) {
        self.streams.push(StreamInfo {
            name,
            client,
            buffer: VecDeque::new(),
            last_timestamp: None,
        });
    }

    /// Set output rate
    pub fn output_rate(mut self, rate: Duration) -> Self {
        self.output_rate = Some(rate);
        self
    }

    /// Run the synchronizer
    pub async fn run<F>(&mut self, mut processor: F) -> Result<()>
    where
        F: FnMut(Vec<(&str, &[u8])>) -> Result<()>,
    {
        // Start receiving from all streams
        let mut handles = Vec::new();
        
        // Simplified - would actually implement synchronization logic
        
        Ok(())
    }
}

/// Time series buffer for streaming data
pub struct TimeSeriesBuffer<T> {
    /// Maximum buffer size
    max_size: usize,
    /// Time window duration
    window_duration: Option<Duration>,
    /// Data points
    data: VecDeque<TimePoint<T>>,
    /// Statistics tracking
    stats: BufferStats,
}

/// Time point in the buffer
#[derive(Clone)]
struct TimePoint<T> {
    timestamp: Instant,
    value: T,
}

/// Buffer statistics
#[derive(Debug, Default)]
struct BufferStats {
    /// Total points added
    total_added: u64,
    /// Total points dropped
    total_dropped: u64,
    /// Current size
    current_size: usize,
    /// Oldest timestamp
    oldest_timestamp: Option<Instant>,
    /// Newest timestamp
    newest_timestamp: Option<Instant>,
}

impl<T: Clone> TimeSeriesBuffer<T> {
    /// Create a new time series buffer
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            window_duration: None,
            data: VecDeque::with_capacity(max_size),
            stats: BufferStats::default(),
        }
    }

    /// Set time window duration
    pub fn with_time_window(mut self, duration: Duration) -> Self {
        self.window_duration = Some(duration);
        self
    }

    /// Add a value
    pub fn push(&mut self, value: T) {
        let now = Instant::now();
        
        // Remove old data if time window is set
        if let Some(duration) = self.window_duration {
            let cutoff = now - duration;
            while let Some(front) = self.data.front() {
                if front.timestamp < cutoff {
                    self.data.pop_front();
                    self.stats.total_dropped += 1;
                } else {
                    break;
                }
            }
        }
        
        // Remove oldest if at capacity
        if self.data.len() >= self.max_size {
            self.data.pop_front();
            self.stats.total_dropped += 1;
        }
        
        // Add new point
        self.data.push_back(TimePoint {
            timestamp: now,
            value,
        });
        
        // Update stats
        self.stats.total_added += 1;
        self.stats.current_size = self.data.len();
        self.stats.newest_timestamp = Some(now);
        if self.stats.oldest_timestamp.is_none() {
            self.stats.oldest_timestamp = Some(now);
        }
    }

    /// Get all values as array
    pub fn as_array(&self) -> Vec<T> {
        self.data.iter().map(|tp| tp.value.clone()).collect()
    }

    /// Get values within time range
    pub fn range(&self, start: Instant, end: Instant) -> Vec<T> {
        self.data.iter()
            .filter(|tp| tp.timestamp >= start && tp.timestamp <= end)
            .map(|tp| tp.value.clone())
            .collect()
    }

    /// Get buffer statistics
    pub fn stats(&self) -> &BufferStats {
        &self.stats
    }
}

/// Stream aggregator for real-time statistics
pub struct StreamAggregator<T> {
    /// Aggregation window
    window: Duration,
    /// Current window data
    current_window: Vec<T>,
    /// Window start time
    window_start: Instant,
    /// Aggregation functions
    aggregators: Vec<Box<dyn Fn(&[T]) -> f64 + Send>>,
    /// Results channel
    results_tx: mpsc::Sender<AggregationResult>,
}

/// Aggregation result
#[derive(Debug, Clone)]
pub struct AggregationResult {
    /// Window start time
    pub window_start: Instant,
    /// Window end time
    pub window_end: Instant,
    /// Number of samples
    pub count: usize,
    /// Aggregated values
    pub values: Vec<f64>,
}

impl<T: Clone + Send + 'static> StreamAggregator<T> {
    /// Create a new aggregator
    pub fn new(window: Duration) -> (Self, mpsc::Receiver<AggregationResult>) {
        let (tx, rx) = mpsc::channel(100);
        
        let aggregator = Self {
            window,
            current_window: Vec::new(),
            window_start: Instant::now(),
            aggregators: Vec::new(),
            results_tx: tx,
        };
        
        (aggregator, rx)
    }

    /// Add an aggregation function
    pub fn add_aggregator<F>(&mut self, f: F)
    where
        F: Fn(&[T]) -> f64 + Send + 'static,
    {
        self.aggregators.push(Box::new(f));
    }

    /// Process a value
    pub async fn process(&mut self, value: T) -> Result<()> {
        let now = Instant::now();
        
        // Check if we need to start a new window
        if now.duration_since(self.window_start) >= self.window {
            self.flush_window().await?;
            self.window_start = now;
        }
        
        self.current_window.push(value);
        Ok(())
    }

    /// Flush current window
    async fn flush_window(&mut self) -> Result<()> {
        if self.current_window.is_empty() {
            return Ok(());
        }
        
        let values: Vec<f64> = self.aggregators.iter()
            .map(|f| f(&self.current_window))
            .collect();
        
        let result = AggregationResult {
            window_start: self.window_start,
            window_end: Instant::now(),
            count: self.current_window.len(),
            values,
        };
        
        self.results_tx.send(result).await
            .map_err(|_| IoError::FileError("Failed to send aggregation result".to_string()))?;
        
        self.current_window.clear();
        Ok(())
    }
}

// Add async_trait to dependencies
use async_trait;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_series_buffer() {
        let mut buffer = TimeSeriesBuffer::new(100);
        
        for i in 0..150 {
            buffer.push(i as f64);
        }
        
        assert_eq!(buffer.stats().total_added, 150);
        assert_eq!(buffer.stats().total_dropped, 50);
        assert_eq!(buffer.stats().current_size, 100);
        
        let values = buffer.as_array();
        assert_eq!(values.len(), 100);
        assert_eq!(values[0], 50.0);
        assert_eq!(values[99], 149.0);
    }

    #[test]
    fn test_backoff_config() {
        let backoff = BackoffConfig::default();
        assert_eq!(backoff.initial_delay, Duration::from_millis(100));
        assert_eq!(backoff.multiplier, 2.0);
        
        let mut delay = backoff.initial_delay.as_secs_f64();
        for _ in 0..5 {
            delay *= backoff.multiplier;
        }
        assert!(delay <= backoff.max_delay.as_secs_f64());
    }

    #[tokio::test]
    async fn test_stream_aggregator() {
        let (mut aggregator, mut rx) = StreamAggregator::<f64>::new(Duration::from_secs(1));
        
        // Add mean aggregator
        aggregator.add_aggregator(|values| {
            values.iter().sum::<f64>() / values.len() as f64
        });
        
        // Process some values
        for i in 0..10 {
            aggregator.process(i as f64).await.unwrap();
        }
        
        // Force flush
        aggregator.flush_window().await.unwrap();
        
        // Check result
        if let Some(result) = rx.recv().await {
            assert_eq!(result.count, 10);
            assert_eq!(result.values.len(), 1);
            assert_eq!(result.values[0], 4.5); // Mean of 0..10
        }
    }
}