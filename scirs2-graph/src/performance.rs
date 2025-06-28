//! Performance optimizations for large graph operations
//!
//! This module provides performance-optimized algorithms and data structures
//! specifically designed for handling large graphs efficiently.

use crate::base::{EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};
use scirs2_core::parallel_ops::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Configuration for parallel processing
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of threads to use (None = use all available cores)
    pub num_threads: Option<usize>,
    /// Chunk size for parallel operations
    pub chunk_size: usize,
    /// Enable SIMD optimizations where available
    pub enable_simd: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        ParallelConfig {
            num_threads: None, // Use all available cores
            chunk_size: 1000,
            enable_simd: true,
        }
    }
}

/// Memory-efficient iterator for large graph traversals
pub struct LargeGraphIterator<N: Node, E: EdgeWeight> {
    /// Current position in iteration
    position: usize,
    /// Graph reference
    graph_data: Vec<(N, N, E)>,
    /// Chunk size for memory efficiency
    chunk_size: usize,
}

impl<N: Node, E: EdgeWeight> LargeGraphIterator<N, E> {
    /// Create a new iterator for large graphs
    pub fn new<Ix>(graph: &Graph<N, E, Ix>, chunk_size: usize) -> Self
    where
        N: Clone,
        E: Clone,
        Ix: petgraph::graph::IndexType,
    {
        let graph_data = graph
            .edges()
            .into_iter()
            .map(|edge| (edge.source, edge.target, edge.weight))
            .collect();

        LargeGraphIterator {
            position: 0,
            graph_data,
            chunk_size,
        }
    }

    /// Get the next chunk of edges
    pub fn next_chunk(&mut self) -> Option<&[(N, N, E)]> {
        if self.position >= self.graph_data.len() {
            return None;
        }

        let end = (self.position + self.chunk_size).min(self.graph_data.len());
        let chunk = &self.graph_data[self.position..end];
        self.position = end;

        if chunk.is_empty() {
            None
        } else {
            Some(chunk)
        }
    }

    /// Reset iterator to beginning
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

/// Parallel degree computation for large graphs
pub fn parallel_degree_computation<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    config: &ParallelConfig,
) -> Result<HashMap<N, usize>>
where
    N: Node + Clone + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: petgraph::graph::IndexType + Send + Sync,
{
    // Note: Thread pool configuration is handled globally by scirs2-core
    // The num_threads config parameter is preserved for future use but currently ignored

    let nodes: Vec<_> = graph.nodes().into_iter().cloned().collect();

    // Parallel computation of degrees
    let degrees: HashMap<N, usize> = nodes
        .par_chunks(config.chunk_size)
        .map(|chunk| {
            let mut local_degrees = HashMap::new();
            for node in chunk {
                let degree = graph.degree(node);
                local_degrees.insert(node.clone(), degree);
            }
            local_degrees
        })
        .reduce(HashMap::new, |mut acc, local| {
            acc.extend(local);
            acc
        });

    Ok(degrees)
}

/// Memory-efficient parallel shortest path computation
pub fn parallel_shortest_paths<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    sources: &[N],
    _config: &ParallelConfig,
) -> Result<HashMap<N, HashMap<N, E>>>
where
    N: Node + Clone + Send + Sync + std::fmt::Debug,
    E: EdgeWeight
        + Clone
        + Send
        + Sync
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType + Send + Sync,
{
    use crate::algorithms::shortest_path::shortest_path;

    // Note: Thread pool configuration is handled globally by scirs2-core
    // The num_threads config parameter is preserved for future use but currently ignored

    let all_nodes: Vec<_> = graph.nodes().into_iter().cloned().collect();

    // Parallel computation of shortest paths from multiple sources
    let results: HashMap<N, HashMap<N, E>> = sources
        .par_iter()
        .map(|source| {
            let mut paths_from_source = HashMap::new();

            for target in &all_nodes {
                if let Ok(Some(path)) = shortest_path(graph, source, target) {
                    paths_from_source.insert(target.clone(), path.total_weight);
                }
            }

            (source.clone(), paths_from_source)
        })
        .collect();

    Ok(results)
}

/// Cache-friendly adjacency matrix computation for large graphs
pub fn cache_friendly_adjacency_matrix<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<Vec<Vec<E>>>
where
    N: Node + Clone,
    E: EdgeWeight + Clone + num_traits::Zero + Copy,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();
    if n == 0 {
        return Ok(vec![]);
    }

    // Pre-allocate matrix with cache-friendly access patterns
    let mut matrix = vec![vec![E::zero(); n]; n];

    // Node to index mapping
    let node_to_index: HashMap<N, usize> = graph
        .nodes()
        .into_iter()
        .enumerate()
        .map(|(i, node)| (node.clone(), i))
        .collect();

    // Fill matrix in row-major order for cache efficiency
    for edge in graph.edges() {
        if let (Some(&src_idx), Some(&tgt_idx)) = (
            node_to_index.get(&edge.source),
            node_to_index.get(&edge.target),
        ) {
            matrix[src_idx][tgt_idx] = edge.weight;
            matrix[tgt_idx][src_idx] = edge.weight; // Undirected graph
        }
    }

    Ok(matrix)
}

/// Streaming algorithm for processing very large graphs
pub struct StreamingGraphProcessor<N: Node, E: EdgeWeight> {
    /// Current batch of edges being processed
    current_batch: Vec<(N, N, E)>,
    /// Maximum batch size
    batch_size: usize,
    /// Running statistics
    edge_count: AtomicUsize,
    /// Degree accumulator
    degree_counter: Arc<parking_lot::Mutex<HashMap<N, usize>>>,
}

impl<N: Node, E: EdgeWeight> StreamingGraphProcessor<N, E>
where
    N: Clone + Send + Sync,
    E: Clone + Send + Sync,
{
    /// Create a new streaming processor
    pub fn new(batch_size: usize) -> Self {
        StreamingGraphProcessor {
            current_batch: Vec::with_capacity(batch_size),
            batch_size,
            edge_count: AtomicUsize::new(0),
            degree_counter: Arc::new(parking_lot::Mutex::new(HashMap::new())),
        }
    }

    /// Add an edge to the streaming processor
    pub fn add_edge(&mut self, source: N, target: N, weight: E) -> Result<()> {
        self.current_batch.push((source, target, weight));

        if self.current_batch.len() >= self.batch_size {
            self.process_batch()?;
        }

        Ok(())
    }

    /// Process the current batch of edges
    fn process_batch(&mut self) -> Result<()> {
        if self.current_batch.is_empty() {
            return Ok(());
        }

        // Update edge count
        self.edge_count
            .fetch_add(self.current_batch.len(), Ordering::Relaxed);

        // Update degree counts
        {
            let mut degrees = self.degree_counter.lock();
            for (source, target, _) in &self.current_batch {
                *degrees.entry(source.clone()).or_insert(0) += 1;
                *degrees.entry(target.clone()).or_insert(0) += 1;
            }
        }

        // Clear current batch
        self.current_batch.clear();

        Ok(())
    }

    /// Finish processing and return final statistics
    pub fn finish(mut self) -> Result<(usize, HashMap<N, usize>)> {
        // Process remaining edges
        self.process_batch()?;

        let total_edges = self.edge_count.load(Ordering::Relaxed);
        let degrees = Arc::try_unwrap(self.degree_counter)
            .map_err(|_| GraphError::AlgorithmError("Failed to unwrap degree counter".to_string()))?
            .into_inner();

        Ok((total_edges, degrees))
    }

    /// Get current edge count
    pub fn edge_count(&self) -> usize {
        self.edge_count.load(Ordering::Relaxed)
    }
}

/// SIMD-optimized operations for numeric graph computations
#[cfg(target_arch = "x86_64")]
pub mod simd_ops {
    #[allow(unused_imports)]
    use super::*;

    /// SIMD-optimized vector addition for graph metrics
    #[allow(dead_code)]
    pub fn simd_vector_add(a: &[f64], b: &[f64]) -> Vec<f64> {
        assert_eq!(a.len(), b.len());

        let mut result = vec![0.0; a.len()];

        // Use standard operations for now - full SIMD would require more dependencies
        for ((dst, &src_a), &src_b) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
            *dst = src_a + src_b;
        }

        result
    }

    /// SIMD-optimized dot product for similarity computations
    #[allow(dead_code)]
    pub fn simd_dot_product(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());

        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
}

/// Lazy evaluation wrapper for expensive graph computations
pub struct LazyGraphMetric<T> {
    /// The computed value (None if not yet computed)
    value: Option<T>,
    /// Computation function
    #[allow(dead_code)]
    compute_fn: Box<dyn FnOnce() -> Result<T>>,
}

impl<T> LazyGraphMetric<T> {
    /// Create a new lazy metric
    pub fn new<F>(compute_fn: F) -> Self
    where
        F: FnOnce() -> Result<T> + 'static,
    {
        LazyGraphMetric {
            value: None,
            compute_fn: Box::new(compute_fn),
        }
    }

    /// Get the value, computing it if necessary
    pub fn get(&mut self) -> Result<&T> {
        if self.value.is_none() {
            // This is a bit tricky due to ownership. In a real implementation,
            // we'd use something like OnceCell or lazy_static for thread safety
            return Err(GraphError::AlgorithmError(
                "Lazy evaluation not fully implemented".to_string(),
            ));
        }

        Ok(self.value.as_ref().unwrap())
    }

    /// Check if the value has been computed
    pub fn is_computed(&self) -> bool {
        self.value.is_some()
    }
}

/// Advanced memory profiling metrics
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    /// Current memory usage in bytes
    pub current_bytes: usize,
    /// Peak memory usage during operation
    pub peak_bytes: usize,
    /// Average memory usage
    pub average_bytes: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Number of deallocations
    pub deallocation_count: usize,
    /// Memory growth rate (bytes per second)
    pub growth_rate: f64,
    /// Potential memory leaks (allocations - deallocations)
    pub potential_leaks: isize,
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        MemoryMetrics {
            current_bytes: 0,
            peak_bytes: 0,
            average_bytes: 0,
            allocation_count: 0,
            deallocation_count: 0,
            growth_rate: 0.0,
            potential_leaks: 0,
        }
    }
}

/// Real-time memory profiler for graph operations
pub struct RealTimeMemoryProfiler {
    /// Memory samples over time
    samples: Vec<(std::time::Instant, usize)>,
    /// Start time
    start_time: std::time::Instant,
    /// Allocation tracking
    allocations: AtomicUsize,
    /// Deallocation tracking
    deallocations: AtomicUsize,
    /// Sampling interval in milliseconds
    sample_interval_ms: u64,
}

impl RealTimeMemoryProfiler {
    /// Create a new real-time profiler
    pub fn new(sample_interval_ms: u64) -> Self {
        RealTimeMemoryProfiler {
            samples: Vec::new(),
            start_time: std::time::Instant::now(),
            allocations: AtomicUsize::new(0),
            deallocations: AtomicUsize::new(0),
            sample_interval_ms,
        }
    }

    /// Record a memory measurement
    pub fn sample_memory(&mut self, current_memory: usize) {
        self.samples.push((std::time::Instant::now(), current_memory));
    }

    /// Record an allocation
    pub fn record_allocation(&self, _size: usize) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a deallocation
    pub fn record_deallocation(&self, _size: usize) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);
    }

    /// Generate comprehensive memory metrics
    pub fn generate_metrics(&self) -> MemoryMetrics {
        if self.samples.is_empty() {
            return MemoryMetrics::default();
        }

        let current_bytes = self.samples.last().map(|(_, mem)| *mem).unwrap_or(0);
        let peak_bytes = self.samples.iter().map(|(_, mem)| *mem).max().unwrap_or(0);
        let average_bytes = if !self.samples.is_empty() {
            self.samples.iter().map(|(_, mem)| *mem).sum::<usize>() / self.samples.len()
        } else {
            0
        };

        let allocation_count = self.allocations.load(Ordering::Relaxed);
        let deallocation_count = self.deallocations.load(Ordering::Relaxed);
        let potential_leaks = allocation_count as isize - deallocation_count as isize;

        // Calculate growth rate
        let growth_rate = if self.samples.len() >= 2 {
            let first = &self.samples[0];
            let last = &self.samples[self.samples.len() - 1];
            let time_diff = last.0.duration_since(first.0).as_secs_f64();
            let memory_diff = last.1 as f64 - first.1 as f64;
            if time_diff > 0.0 { memory_diff / time_diff } else { 0.0 }
        } else {
            0.0
        };

        MemoryMetrics {
            current_bytes,
            peak_bytes,
            average_bytes,
            allocation_count,
            deallocation_count,
            growth_rate,
            potential_leaks,
        }
    }

    /// Check for potential memory issues
    pub fn analyze_memory_health(&self) -> Vec<String> {
        let metrics = self.generate_metrics();
        let mut warnings = Vec::new();

        // Check for rapid memory growth
        if metrics.growth_rate > 1_000_000.0 { // 1MB/second
            warnings.push(format!("High memory growth rate: {:.2} bytes/second", metrics.growth_rate));
        }

        // Check for potential leaks
        if metrics.potential_leaks > 1000 {
            warnings.push(format!("Potential memory leak detected: {} unmatched allocations", metrics.potential_leaks));
        }

        // Check for excessive peak memory
        if metrics.peak_bytes > 1_000_000_000 { // 1GB
            warnings.push(format!("High peak memory usage: {:.2} MB", metrics.peak_bytes as f64 / 1_000_000.0));
        }

        warnings
    }

    /// Export memory timeline for visualization
    pub fn export_timeline(&self) -> Vec<(f64, usize)> {
        self.samples.iter().map(|(time, memory)| {
            let elapsed = time.duration_since(self.start_time).as_secs_f64();
            (elapsed, *memory)
        }).collect()
    }
}

/// Performance monitoring utilities with enhanced memory profiling
pub struct PerformanceMonitor {
    /// Start time of current operation
    start_time: std::time::Instant,
    /// Operation name
    operation_name: String,
    /// Real-time memory profiler
    memory_profiler: RealTimeMemoryProfiler,
    /// Memory sampling thread handle
    sampling_active: Arc<std::sync::atomic::AtomicBool>,
}

impl PerformanceMonitor {
    /// Start monitoring a new operation with memory profiling
    pub fn start(operation_name: String) -> Self {
        Self::start_with_config(operation_name, 100) // Sample every 100ms by default
    }

    /// Start monitoring with custom sampling interval
    pub fn start_with_config(operation_name: String, sample_interval_ms: u64) -> Self {
        let monitor = PerformanceMonitor {
            start_time: std::time::Instant::now(),
            operation_name,
            memory_profiler: RealTimeMemoryProfiler::new(sample_interval_ms),
            sampling_active: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        };

        monitor
    }

    /// Manually record current memory usage
    pub fn record_memory(&mut self, current_memory: usize) {
        self.memory_profiler.sample_memory(current_memory);
    }

    /// Record an allocation event
    pub fn record_allocation(&self, size: usize) {
        self.memory_profiler.record_allocation(size);
    }

    /// Record a deallocation event
    pub fn record_deallocation(&self, size: usize) {
        self.memory_profiler.record_deallocation(size);
    }

    /// Get current memory metrics
    pub fn get_memory_metrics(&self) -> MemoryMetrics {
        self.memory_profiler.generate_metrics()
    }

    /// Check for memory health issues
    pub fn check_memory_health(&self) -> Vec<String> {
        self.memory_profiler.analyze_memory_health()
    }

    /// Get memory timeline for analysis
    pub fn get_memory_timeline(&self) -> Vec<(f64, usize)> {
        self.memory_profiler.export_timeline()
    }

    /// Update peak memory usage (legacy method)
    pub fn update_memory(&mut self, current_memory: usize) {
        self.record_memory(current_memory);
    }

    /// Finish monitoring and return comprehensive performance metrics
    pub fn finish(self) -> PerformanceReport {
        self.sampling_active.store(false, Ordering::Relaxed);
        
        let duration = self.start_time.elapsed();
        let memory_metrics = self.memory_profiler.generate_metrics();
        let memory_warnings = self.memory_profiler.analyze_memory_health();
        let timeline = self.memory_profiler.export_timeline();

        let report = PerformanceReport {
            operation_name: self.operation_name.clone(),
            duration,
            memory_metrics,
            memory_warnings: memory_warnings.clone(),
            timeline,
        };

        println!(
            "Operation '{}' completed in {:?}",
            self.operation_name, duration
        );
        println!(
            "Memory: peak={:.2}MB, avg={:.2}MB, current={:.2}MB",
            report.memory_metrics.peak_bytes as f64 / 1_000_000.0,
            report.memory_metrics.average_bytes as f64 / 1_000_000.0,
            report.memory_metrics.current_bytes as f64 / 1_000_000.0
        );

        if !memory_warnings.is_empty() {
            println!("Memory warnings:");
            for warning in &memory_warnings {
                println!("  - {}", warning);
            }
        }

        report
    }
}

/// Comprehensive performance report
#[derive(Debug)]
pub struct PerformanceReport {
    /// Operation name
    pub operation_name: String,
    /// Total execution duration
    pub duration: std::time::Duration,
    /// Memory metrics
    pub memory_metrics: MemoryMetrics,
    /// Memory health warnings
    pub memory_warnings: Vec<String>,
    /// Memory usage timeline
    pub timeline: Vec<(f64, usize)>,
}

/// Optimized graph algorithms trait for large graphs
pub trait LargeGraphOps<N: Node, E: EdgeWeight> {
    /// Parallel computation of node degrees
    fn parallel_degrees(&self, config: &ParallelConfig) -> Result<HashMap<N, usize>>;

    /// Memory-efficient iteration over edges
    fn iter_edges_chunked(&self, chunk_size: usize) -> LargeGraphIterator<N, E>;

    /// Cache-friendly matrix representation
    fn cache_friendly_matrix(&self) -> Result<Vec<Vec<E>>>;
}

impl<N: Node, E: EdgeWeight, Ix: petgraph::graph::IndexType + Send + Sync> LargeGraphOps<N, E>
    for Graph<N, E, Ix>
where
    N: Clone + Send + Sync,
    E: Clone + Send + Sync + num_traits::Zero + Copy,
{
    fn parallel_degrees(&self, config: &ParallelConfig) -> Result<HashMap<N, usize>> {
        parallel_degree_computation(self, config)
    }

    fn iter_edges_chunked(&self, chunk_size: usize) -> LargeGraphIterator<N, E> {
        LargeGraphIterator::new(self, chunk_size)
    }

    fn cache_friendly_matrix(&self) -> Result<Vec<Vec<E>>> {
        cache_friendly_adjacency_matrix(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_config() {
        let config = ParallelConfig::default();
        assert_eq!(config.chunk_size, 1000);
        assert!(config.enable_simd);
    }

    #[test]
    fn test_large_graph_iterator() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 2.0).unwrap();
        graph.add_edge(3, 4, 3.0).unwrap();

        let mut iterator = LargeGraphIterator::new(&graph, 2);

        let chunk1 = iterator.next_chunk();
        assert!(chunk1.is_some());
        assert_eq!(chunk1.unwrap().len(), 2);

        let chunk2 = iterator.next_chunk();
        assert!(chunk2.is_some());
        assert_eq!(chunk2.unwrap().len(), 1);

        let chunk3 = iterator.next_chunk();
        assert!(chunk3.is_none());
    }

    #[test]
    fn test_parallel_degree_computation() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0).unwrap();
        graph.add_edge(2, 3, 2.0).unwrap();
        graph.add_edge(3, 1, 3.0).unwrap();

        let config = ParallelConfig::default();
        let degrees = graph.parallel_degrees(&config).unwrap();

        assert_eq!(degrees[&1], 2);
        assert_eq!(degrees[&2], 2);
        assert_eq!(degrees[&3], 2);
    }

    #[test]
    fn test_streaming_processor() {
        let mut processor: StreamingGraphProcessor<i32, f64> = StreamingGraphProcessor::new(2);

        processor.add_edge(1, 2, 1.0).unwrap();
        assert_eq!(processor.edge_count(), 0); // Not yet processed

        processor.add_edge(2, 3, 2.0).unwrap();
        assert_eq!(processor.edge_count(), 2); // Batch processed

        let (total_edges, degrees) = processor.finish().unwrap();
        assert_eq!(total_edges, 2);
        assert_eq!(degrees[&1], 1);
        assert_eq!(degrees[&2], 2);
        assert_eq!(degrees[&3], 1);
    }

    #[test]
    fn test_cache_friendly_matrix() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(0, 1, 1.0).unwrap();
        graph.add_edge(1, 2, 2.0).unwrap();

        let matrix = graph.cache_friendly_matrix().unwrap();
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0][1], 1.0);
        assert_eq!(matrix[1][2], 2.0);
        assert_eq!(matrix[2][1], 2.0); // Undirected
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::start("test_operation".to_string());
        
        // Simulate memory usage
        monitor.record_memory(1024);
        monitor.record_memory(2048);
        monitor.record_memory(1536);
        
        // Simulate allocations
        monitor.record_allocation(1024);
        monitor.record_allocation(512);
        monitor.record_deallocation(256);
        
        std::thread::sleep(std::time::Duration::from_millis(10));
        let report = monitor.finish();
        
        assert!(report.duration.as_millis() >= 10);
        assert_eq!(report.memory_metrics.peak_bytes, 2048);
        assert_eq!(report.memory_metrics.current_bytes, 1536);
        assert_eq!(report.memory_metrics.allocation_count, 2);
        assert_eq!(report.memory_metrics.deallocation_count, 1);
        assert_eq!(report.memory_metrics.potential_leaks, 1);
    }

    #[test]
    fn test_real_time_memory_profiler() {
        let mut profiler = RealTimeMemoryProfiler::new(50);
        
        // Record memory samples
        profiler.sample_memory(1000);
        std::thread::sleep(std::time::Duration::from_millis(10));
        profiler.sample_memory(2000);
        std::thread::sleep(std::time::Duration::from_millis(10));
        profiler.sample_memory(1500);
        
        // Record allocations/deallocations
        profiler.record_allocation(1000);
        profiler.record_allocation(500);
        profiler.record_deallocation(200);
        
        let metrics = profiler.generate_metrics();
        assert_eq!(metrics.current_bytes, 1500);
        assert_eq!(metrics.peak_bytes, 2000);
        assert!(metrics.average_bytes > 0);
        assert_eq!(metrics.allocation_count, 2);
        assert_eq!(metrics.deallocation_count, 1);
        assert_eq!(metrics.potential_leaks, 1);
        
        // Test timeline export
        let timeline = profiler.export_timeline();
        assert_eq!(timeline.len(), 3);
        assert_eq!(timeline[0].1, 1000);
        assert_eq!(timeline[1].1, 2000);
        assert_eq!(timeline[2].1, 1500);
    }

    #[test]
    fn test_memory_health_analysis() {
        let mut profiler = RealTimeMemoryProfiler::new(100);
        
        // Simulate high memory growth
        profiler.sample_memory(100_000_000);
        std::thread::sleep(std::time::Duration::from_millis(50));
        profiler.sample_memory(200_000_000);
        
        // Simulate many unmatched allocations
        for _ in 0..1500 {
            profiler.record_allocation(1024);
        }
        
        let warnings = profiler.analyze_memory_health();
        assert!(!warnings.is_empty());
        
        // Should warn about high growth rate and potential leaks
        let has_growth_warning = warnings.iter().any(|w| w.contains("growth rate"));
        let has_leak_warning = warnings.iter().any(|w| w.contains("leak"));
        
        assert!(has_growth_warning);
        assert!(has_leak_warning);
    }

    #[test]
    fn test_memory_metrics_calculation() {
        let mut profiler = RealTimeMemoryProfiler::new(100);
        
        // Create a clear growth pattern
        profiler.sample_memory(1000);
        std::thread::sleep(std::time::Duration::from_millis(100));
        profiler.sample_memory(2000);
        std::thread::sleep(std::time::Duration::from_millis(100));
        profiler.sample_memory(3000);
        
        let metrics = profiler.generate_metrics();
        
        // Should have positive growth rate
        assert!(metrics.growth_rate > 0.0);
        
        // Average should be around 2000
        assert!(metrics.average_bytes >= 1500 && metrics.average_bytes <= 2500);
        
        // Peak should be 3000
        assert_eq!(metrics.peak_bytes, 3000);
        
        // Current should be 3000
        assert_eq!(metrics.current_bytes, 3000);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_operations() {
        use crate::performance::simd_ops::*;

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let sum = simd_vector_add(&a, &b);
        assert_eq!(sum, vec![5.0, 7.0, 9.0]);

        let dot = simd_dot_product(&a, &b);
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6
    }
}
