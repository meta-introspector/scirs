//! CUDA memory pool management for efficient GPU memory allocation
//!
//! This module provides memory pooling to reduce allocation overhead
//! and improve performance for repeated GPU operations.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::ptr;
use std::sync::{Arc, Mutex};

use crate::gpu::GpuOptimizerError;

#[cfg(feature = "gpu")]
use scirs2_core::gpu::GpuContext;

/// Memory allocation statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total allocated memory
    pub total_allocated: usize,

    /// Currently used memory
    pub current_used: usize,

    /// Peak memory usage
    pub peak_usage: usize,

    /// Number of allocations
    pub allocation_count: usize,

    /// Number of deallocations
    pub deallocation_count: usize,

    /// Number of cache hits
    pub cache_hits: usize,

    /// Number of cache misses
    pub cache_misses: usize,
}

/// Memory block metadata
#[derive(Debug)]
struct MemoryBlock {
    /// Pointer to GPU memory
    ptr: *mut u8,

    /// Size of the block
    size: usize,

    /// Whether block is currently in use
    in_use: bool,

    /// Allocation timestamp
    allocated_at: std::time::Instant,

    /// Last used timestamp
    last_used: std::time::Instant,
}

impl MemoryBlock {
    fn new(ptr: *mut u8, size: usize) -> Self {
        let now = std::time::Instant::now();
        Self {
            ptr,
            size,
            in_use: true,
            allocated_at: now,
            last_used: now,
        }
    }

    fn mark_used(&mut self) {
        self.in_use = true;
        self.last_used = std::time::Instant::now();
    }

    fn mark_free(&mut self) {
        self.in_use = false;
    }
}

/// CUDA memory pool
pub struct CudaMemoryPool {
    /// Free blocks organized by size
    free_blocks: HashMap<usize, VecDeque<MemoryBlock>>,

    /// All allocated blocks
    all_blocks: Vec<MemoryBlock>,

    /// Memory statistics
    stats: MemoryStats,

    /// Maximum pool size
    max_pool_size: usize,

    /// Minimum block size to pool
    min_block_size: usize,

    /// Enable memory defragmentation
    enable_defrag: bool,

    /// GPU context
    gpu_context: Option<Arc<GpuContext>>,

    /// Large batch optimization settings
    large_batch_config: LargeBatchConfig,

    /// Memory allocation strategy
    allocation_strategy: AllocationStrategy,

    /// Adaptive sizing based on usage patterns
    adaptive_sizing: AdaptiveSizing,

    /// Memory pressure monitoring
    pressure_monitor: MemoryPressureMonitor,

    /// Pre-allocated large buffers for batch operations
    batch_buffers: Vec<BatchBuffer>,
}

/// Large batch optimization configuration
#[derive(Debug, Clone)]
pub struct LargeBatchConfig {
    /// Minimum batch size to consider for optimization
    pub min_batch_size: usize,
    /// Maximum number of pre-allocated batch buffers
    pub max_batch_buffers: usize,
    /// Buffer size growth factor
    pub growth_factor: f32,
    /// Enable batch buffer coalescing
    pub enable_coalescing: bool,
    /// Pre-allocation threshold (percentage of max pool size)
    pub preallocation_threshold: f32,
    /// Batch buffer lifetime (seconds)
    pub buffer_lifetime: u64,
}

impl Default for LargeBatchConfig {
    fn default() -> Self {
        Self {
            min_batch_size: 1024 * 1024, // 1MB
            max_batch_buffers: 16,
            growth_factor: 1.5,
            enable_coalescing: true,
            preallocation_threshold: 0.8,
            buffer_lifetime: 300, // 5 minutes
        }
    }
}

/// Memory allocation strategies for different workload patterns
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AllocationStrategy {
    /// First-fit allocation (fastest)
    FirstFit,
    /// Best-fit allocation (memory efficient)
    BestFit,
    /// Worst-fit allocation (reduces fragmentation)
    WorstFit,
    /// Buddy system allocation (power-of-2 sizes)
    BuddySystem,
    /// Segregated list allocation (size-based pools)
    SegregatedList,
    /// Adaptive strategy based on workload
    Adaptive,
}

impl Default for AllocationStrategy {
    fn default() -> Self {
        AllocationStrategy::Adaptive
    }
}

/// Adaptive memory sizing based on usage patterns
#[derive(Debug, Clone)]
pub struct AdaptiveSizing {
    /// Enable adaptive pool resizing
    pub enable_adaptive_resize: bool,
    /// Allocation history for pattern analysis
    pub allocation_history: VecDeque<AllocationEvent>,
    /// Maximum history size
    pub max_history_size: usize,
    /// Resize threshold (utilization percentage)
    pub resize_threshold: f32,
    /// Minimum pool size (bytes)
    pub min_pool_size: usize,
    /// Pool size growth factor
    pub growth_factor: f32,
    /// Pool size shrink factor
    pub shrink_factor: f32,
    /// Analysis window size (number of allocations)
    pub analysis_window: usize,
}

impl Default for AdaptiveSizing {
    fn default() -> Self {
        Self {
            enable_adaptive_resize: true,
            allocation_history: VecDeque::new(),
            max_history_size: 10000,
            resize_threshold: 0.85,
            min_pool_size: 256 * 1024 * 1024, // 256MB
            growth_factor: 1.5,
            shrink_factor: 0.75,
            analysis_window: 1000,
        }
    }
}

/// Allocation event for pattern analysis
#[derive(Debug, Clone)]
pub struct AllocationEvent {
    /// Size of allocation
    pub size: usize,
    /// Timestamp of allocation
    pub timestamp: std::time::Instant,
    /// Whether allocation was satisfied from cache
    pub cache_hit: bool,
    /// Allocation latency (microseconds)
    pub latency_us: u64,
}

/// Memory pressure monitoring
#[derive(Debug, Clone)]
pub struct MemoryPressureMonitor {
    /// Enable pressure monitoring
    pub enable_monitoring: bool,
    /// Memory pressure threshold (0.0-1.0)
    pub pressure_threshold: f32,
    /// Monitoring interval (milliseconds)
    pub monitor_interval_ms: u64,
    /// Current memory pressure level
    pub current_pressure: f32,
    /// Pressure history
    pub pressure_history: VecDeque<PressureReading>,
    /// Maximum history size
    pub max_history_size: usize,
    /// Enable automatic cleanup under pressure
    pub auto_cleanup: bool,
    /// Cleanup threshold (pressure level)
    pub cleanup_threshold: f32,
}

impl Default for MemoryPressureMonitor {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            pressure_threshold: 0.9,
            monitor_interval_ms: 1000,
            current_pressure: 0.0,
            pressure_history: VecDeque::new(),
            max_history_size: 3600, // 1 hour at 1s intervals
            auto_cleanup: true,
            cleanup_threshold: 0.95,
        }
    }
}

/// Memory pressure reading
#[derive(Debug, Clone)]
pub struct PressureReading {
    /// Timestamp of reading
    pub timestamp: std::time::Instant,
    /// Memory pressure level (0.0-1.0)
    pub pressure: f32,
    /// Available memory (bytes)
    pub available_memory: usize,
    /// Total allocated memory (bytes)
    pub allocated_memory: usize,
}

/// Pre-allocated batch buffer for large operations
#[derive(Debug)]
pub struct BatchBuffer {
    /// Buffer pointer
    pub ptr: *mut u8,
    /// Buffer size
    pub size: usize,
    /// Whether buffer is currently in use
    pub in_use: bool,
    /// Creation timestamp
    pub created_at: std::time::Instant,
    /// Last used timestamp
    pub last_used: std::time::Instant,
    /// Usage count
    pub usage_count: usize,
    /// Buffer type/category
    pub buffer_type: BatchBufferType,
}

/// Types of batch buffers for different use cases
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BatchBufferType {
    /// General purpose batch buffer
    General,
    /// Gradient accumulation buffer
    GradientAccumulation,
    /// Parameter update buffer
    ParameterUpdate,
    /// Communication buffer for multi-GPU
    Communication,
    /// Temporary computation buffer
    Temporary,
}

impl BatchBuffer {
    /// Create a new batch buffer
    pub fn new(ptr: *mut u8, size: usize, buffer_type: BatchBufferType) -> Self {
        let now = std::time::Instant::now();
        Self {
            ptr,
            size,
            in_use: false,
            created_at: now,
            last_used: now,
            usage_count: 0,
            buffer_type,
        }
    }

    /// Mark buffer as used
    pub fn mark_used(&mut self) {
        self.in_use = true;
        self.last_used = std::time::Instant::now();
        self.usage_count += 1;
    }

    /// Mark buffer as free
    pub fn mark_free(&mut self) {
        self.in_use = false;
    }

    /// Check if buffer has expired
    pub fn is_expired(&self, lifetime_secs: u64) -> bool {
        self.last_used.elapsed().as_secs() > lifetime_secs
    }
}

impl CudaMemoryPool {
    /// Create a new memory pool
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            free_blocks: HashMap::new(),
            all_blocks: Vec::new(),
            stats: MemoryStats::default(),
            max_pool_size,
            min_block_size: 256, // Don't pool allocations smaller than 256 bytes
            enable_defrag: true,
            gpu_context: None,
            large_batch_config: LargeBatchConfig::default(),
            allocation_strategy: AllocationStrategy::default(),
            adaptive_sizing: AdaptiveSizing::default(),
            pressure_monitor: MemoryPressureMonitor::default(),
            batch_buffers: Vec::new(),
        }
    }

    /// Create a new memory pool with custom configuration
    pub fn with_large_batch_config(max_pool_size: usize, config: LargeBatchConfig) -> Self {
        Self {
            free_blocks: HashMap::new(),
            all_blocks: Vec::new(),
            stats: MemoryStats::default(),
            max_pool_size,
            min_block_size: 256,
            enable_defrag: true,
            gpu_context: None,
            large_batch_config: config,
            allocation_strategy: AllocationStrategy::default(),
            adaptive_sizing: AdaptiveSizing::default(),
            pressure_monitor: MemoryPressureMonitor::default(),
            batch_buffers: Vec::new(),
        }
    }

    /// Set GPU context
    pub fn set_gpu_context(&mut self, context: Arc<GpuContext>) {
        self.gpu_context = Some(context);
    }

    /// Allocate memory from pool
    pub fn allocate(&mut self, size: usize) -> Result<*mut u8, GpuOptimizerError> {
        // Round up to nearest power of 2 for better reuse
        let aligned_size = size.next_power_of_two();

        // Try to find a free block
        if let Some(blocks) = self.free_blocks.get_mut(&aligned_size) {
            if let Some(mut block) = blocks.pop_front() {
                block.mark_used();
                self.stats.current_used += block.size;
                self.stats.cache_hits += 1;
                return Ok(block.ptr);
            }
        }

        // Try to find a larger block that can be reused
        for (&block_size, blocks) in self.free_blocks.iter_mut() {
            if block_size >= aligned_size {
                if let Some(mut block) = blocks.pop_front() {
                    block.mark_used();
                    self.stats.current_used += aligned_size;
                    self.stats.cache_hits += 1;
                    return Ok(block.ptr);
                }
            }
        }

        // Allocate new block if within limits
        if self.stats.total_allocated + aligned_size <= self.max_pool_size {
            let ptr = self.allocate_gpu_memory(aligned_size)?;
            let block = MemoryBlock::new(ptr, aligned_size);

            self.all_blocks.push(block);
            self.stats.total_allocated += aligned_size;
            self.stats.current_used += aligned_size;
            self.stats.allocation_count += 1;
            self.stats.cache_misses += 1;

            // Update peak usage
            if self.stats.current_used > self.stats.peak_usage {
                self.stats.peak_usage = self.stats.current_used;
            }

            Ok(ptr)
        } else {
            // Try defragmentation if enabled
            if self.enable_defrag {
                self.defragment()?;
                return self.allocate(size);
            }

            Err(GpuOptimizerError::InvalidState(format!(
                "Memory pool limit exceeded: requested {}, available {}",
                aligned_size,
                self.max_pool_size - self.stats.total_allocated
            )))
        }
    }

    /// Deallocate memory back to pool
    pub fn deallocate(&mut self, ptr: *mut u8) {
        // Find the block
        for block in &mut self.all_blocks {
            if block.ptr == ptr && block.in_use {
                block.mark_free();
                let size = block.size;

                // Add to free list
                self.free_blocks
                    .entry(size)
                    .or_insert_with(VecDeque::new)
                    .push_back(MemoryBlock {
                        ptr: block.ptr,
                        size: block.size,
                        in_use: false,
                        allocated_at: block.allocated_at,
                        last_used: block.last_used,
                    });

                self.stats.current_used -= size;
                self.stats.deallocation_count += 1;
                return;
            }
        }
    }

    /// Defragment memory pool
    pub fn defragment(&mut self) -> Result<(), GpuOptimizerError> {
        // Remove blocks that haven't been used recently
        let cutoff = std::time::Instant::now() - std::time::Duration::from_secs(60);

        let mut freed_memory = 0;
        for blocks in self.free_blocks.values_mut() {
            blocks.retain(|block| {
                if block.last_used < cutoff {
                    // Free the actual GPU memory
                    if let Err(e) = self.free_gpu_memory(block.ptr) {
                        eprintln!("Failed to free GPU memory: {}", e);
                        true // Keep the block if freeing failed
                    } else {
                        freed_memory += block.size;
                        false // Remove the block
                    }
                } else {
                    true // Keep recently used blocks
                }
            });
        }

        self.stats.total_allocated -= freed_memory;

        // Remove empty entries
        self.free_blocks.retain(|_, blocks| !blocks.is_empty());

        Ok(())
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Allocate a large batch buffer
    pub fn allocate_batch_buffer(
        &mut self,
        size: usize,
        buffer_type: BatchBufferType,
    ) -> Result<*mut u8, GpuOptimizerError> {
        if size < self.large_batch_config.min_batch_size {
            // Use regular allocation for small buffers
            return self.allocate(size);
        }

        // Check for available batch buffer
        for buffer in &mut self.batch_buffers {
            if !buffer.in_use && buffer.size >= size && buffer.buffer_type == buffer_type {
                buffer.mark_used();
                return Ok(buffer.ptr);
            }
        }

        // Pre-allocate new batch buffer if under limit
        if self.batch_buffers.len() < self.large_batch_config.max_batch_buffers {
            let buffer_size = if self.large_batch_config.enable_coalescing {
                (size as f32 * self.large_batch_config.growth_factor) as usize
            } else {
                size
            };

            let ptr = self.allocate_gpu_memory(buffer_size)?;
            let mut buffer = BatchBuffer::new(ptr, buffer_size, buffer_type);
            buffer.mark_used();
            self.batch_buffers.push(buffer);

            self.stats.total_allocated += buffer_size;
            self.stats.current_used += buffer_size;
            self.stats.allocation_count += 1;

            return Ok(ptr);
        }

        // Fallback to regular allocation
        self.allocate(size)
    }

    /// Release a batch buffer
    pub fn release_batch_buffer(&mut self, ptr: *mut u8) {
        for buffer in &mut self.batch_buffers {
            if buffer.ptr == ptr && buffer.in_use {
                buffer.mark_free();
                self.stats.current_used -= buffer.size;
                self.stats.deallocation_count += 1;
                return;
            }
        }

        // Fallback to regular deallocation
        self.deallocate(ptr);
    }

    /// Clean up expired batch buffers
    pub fn cleanup_expired_buffers(&mut self) -> Result<(), GpuOptimizerError> {
        let lifetime = self.large_batch_config.buffer_lifetime;
        let mut freed_memory = 0;

        self.batch_buffers.retain(|buffer| {
            if !buffer.in_use && buffer.is_expired(lifetime) {
                // Free the GPU memory
                if let Err(e) = self.free_gpu_memory(buffer.ptr) {
                    eprintln!("Failed to free batch buffer: {}", e);
                    true // Keep the buffer if freeing failed
                } else {
                    freed_memory += buffer.size;
                    false // Remove the buffer
                }
            } else {
                true // Keep active and recent buffers
            }
        });

        self.stats.total_allocated -= freed_memory;
        Ok(())
    }

    /// Update memory pressure monitoring
    pub fn update_memory_pressure(&mut self) {
        if !self.pressure_monitor.enable_monitoring {
            return;
        }

        let utilization = self.stats.current_used as f32 / self.max_pool_size as f32;
        self.pressure_monitor.current_pressure = utilization;

        let reading = PressureReading {
            timestamp: std::time::Instant::now(),
            pressure: utilization,
            available_memory: self.max_pool_size - self.stats.current_used,
            allocated_memory: self.stats.current_used,
        };

        self.pressure_monitor.pressure_history.push_back(reading);

        // Limit history size
        while self.pressure_monitor.pressure_history.len() > self.pressure_monitor.max_history_size
        {
            self.pressure_monitor.pressure_history.pop_front();
        }

        // Trigger cleanup if pressure is too high
        if self.pressure_monitor.auto_cleanup
            && utilization > self.pressure_monitor.cleanup_threshold
        {
            let _ = self.cleanup_expired_buffers();
            let _ = self.defragment();
        }
    }

    /// Adaptive allocation using strategy selection
    pub fn allocate_adaptive(&mut self, size: usize) -> Result<*mut u8, GpuOptimizerError> {
        self.update_memory_pressure();

        // Record allocation event
        let start_time = std::time::Instant::now();

        let result = match self.allocation_strategy {
            AllocationStrategy::FirstFit => self.allocate_first_fit(size),
            AllocationStrategy::BestFit => self.allocate_best_fit(size),
            AllocationStrategy::WorstFit => self.allocate_worst_fit(size),
            AllocationStrategy::BuddySystem => self.allocate_buddy_system(size),
            AllocationStrategy::SegregatedList => self.allocate_segregated_list(size),
            AllocationStrategy::Adaptive => self.allocate_adaptive_strategy(size),
        };

        // Record allocation event for analysis
        let latency = start_time.elapsed().as_micros() as u64;
        let cache_hit = result.is_ok() && latency < 100; // Assume cache hit if very fast

        let event = AllocationEvent {
            size,
            timestamp: std::time::Instant::now(),
            cache_hit,
            latency_us: latency,
        };

        self.adaptive_sizing.allocation_history.push_back(event);

        // Limit history size
        while self.adaptive_sizing.allocation_history.len() > self.adaptive_sizing.max_history_size
        {
            self.adaptive_sizing.allocation_history.pop_front();
        }

        result
    }

    /// First-fit allocation strategy
    fn allocate_first_fit(&mut self, size: usize) -> Result<*mut u8, GpuOptimizerError> {
        let aligned_size = size.next_power_of_two();

        // Find first available block that fits
        for (&block_size, blocks) in self.free_blocks.iter_mut() {
            if block_size >= aligned_size {
                if let Some(mut block) = blocks.pop_front() {
                    block.mark_used();
                    self.stats.current_used += aligned_size;
                    self.stats.cache_hits += 1;
                    return Ok(block.ptr);
                }
            }
        }

        // Allocate new block
        self.allocate_new_block(aligned_size)
    }

    /// Best-fit allocation strategy
    fn allocate_best_fit(&mut self, size: usize) -> Result<*mut u8, GpuOptimizerError> {
        let aligned_size = size.next_power_of_two();
        let mut best_fit_size = None;
        let mut min_waste = usize::MAX;

        // Find the smallest block that fits
        for &block_size in self.free_blocks.keys() {
            if block_size >= aligned_size {
                let waste = block_size - aligned_size;
                if waste < min_waste {
                    min_waste = waste;
                    best_fit_size = Some(block_size);
                }
            }
        }

        if let Some(block_size) = best_fit_size {
            if let Some(blocks) = self.free_blocks.get_mut(&block_size) {
                if let Some(mut block) = blocks.pop_front() {
                    block.mark_used();
                    self.stats.current_used += aligned_size;
                    self.stats.cache_hits += 1;
                    return Ok(block.ptr);
                }
            }
        }

        // Allocate new block
        self.allocate_new_block(aligned_size)
    }

    /// Worst-fit allocation strategy
    fn allocate_worst_fit(&mut self, size: usize) -> Result<*mut u8, GpuOptimizerError> {
        let aligned_size = size.next_power_of_two();
        let mut worst_fit_size = None;
        let mut max_waste = 0;

        // Find the largest block that fits
        for &block_size in self.free_blocks.keys() {
            if block_size >= aligned_size {
                let waste = block_size - aligned_size;
                if waste > max_waste {
                    max_waste = waste;
                    worst_fit_size = Some(block_size);
                }
            }
        }

        if let Some(block_size) = worst_fit_size {
            if let Some(blocks) = self.free_blocks.get_mut(&block_size) {
                if let Some(mut block) = blocks.pop_front() {
                    block.mark_used();
                    self.stats.current_used += aligned_size;
                    self.stats.cache_hits += 1;
                    return Ok(block.ptr);
                }
            }
        }

        // Allocate new block
        self.allocate_new_block(aligned_size)
    }

    /// Buddy system allocation strategy
    fn allocate_buddy_system(&mut self, size: usize) -> Result<*mut u8, GpuOptimizerError> {
        // Round up to next power of 2 for buddy system
        let buddy_size = size.next_power_of_two();
        self.allocate_first_fit(buddy_size)
    }

    /// Segregated list allocation strategy
    fn allocate_segregated_list(&mut self, size: usize) -> Result<*mut u8, GpuOptimizerError> {
        // Use size classes for segregated allocation
        let size_class = self.get_size_class(size);
        self.allocate_first_fit(size_class)
    }

    /// Adaptive allocation strategy selection
    fn allocate_adaptive_strategy(&mut self, size: usize) -> Result<*mut u8, GpuOptimizerError> {
        // Analyze recent allocation patterns to choose best strategy
        let recent_history: Vec<_> = self
            .adaptive_sizing
            .allocation_history
            .iter()
            .rev()
            .take(self.adaptive_sizing.analysis_window)
            .collect();

        if recent_history.is_empty() {
            return self.allocate_first_fit(size);
        }

        // Analyze patterns
        let avg_latency: f64 = recent_history
            .iter()
            .map(|event| event.latency_us as f64)
            .sum::<f64>()
            / recent_history.len() as f64;

        let cache_hit_rate = recent_history
            .iter()
            .filter(|event| event.cache_hit)
            .count() as f32
            / recent_history.len() as f32;

        // Choose strategy based on performance metrics
        if cache_hit_rate > 0.8 {
            // High cache hit rate, use first-fit for speed
            self.allocate_first_fit(size)
        } else if avg_latency > 1000.0 {
            // High latency, use best-fit to reduce fragmentation
            self.allocate_best_fit(size)
        } else if self.pressure_monitor.current_pressure > 0.8 {
            // High memory pressure, use best-fit to conserve memory
            self.allocate_best_fit(size)
        } else {
            // Balanced approach
            self.allocate_first_fit(size)
        }
    }

    /// Get size class for segregated allocation
    fn get_size_class(&self, size: usize) -> usize {
        // Define size classes (powers of 2)
        let classes = [
            256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
        ];

        for &class_size in &classes {
            if size <= class_size {
                return class_size;
            }
        }

        // For very large allocations, round up to next MB
        ((size + 1048575) / 1048576) * 1048576
    }

    /// Allocate new memory block
    fn allocate_new_block(&mut self, size: usize) -> Result<*mut u8, GpuOptimizerError> {
        if self.stats.total_allocated + size <= self.max_pool_size {
            let ptr = self.allocate_gpu_memory(size)?;
            let block = MemoryBlock::new(ptr, size);

            self.all_blocks.push(block);
            self.stats.total_allocated += size;
            self.stats.current_used += size;
            self.stats.allocation_count += 1;
            self.stats.cache_misses += 1;

            // Update peak usage
            if self.stats.current_used > self.stats.peak_usage {
                self.stats.peak_usage = self.stats.current_used;
            }

            Ok(ptr)
        } else {
            // Try defragmentation if enabled
            if self.enable_defrag {
                self.defragment()?;
                return self.allocate_new_block(size);
            }

            Err(GpuOptimizerError::InvalidState(format!(
                "Memory pool limit exceeded: requested {}, available {}",
                size,
                self.max_pool_size - self.stats.total_allocated
            )))
        }
    }

    /// Get current memory pressure level
    pub fn get_memory_pressure(&self) -> f32 {
        self.pressure_monitor.current_pressure
    }

    /// Get allocation statistics for analysis
    pub fn get_allocation_analytics(&self) -> AllocationAnalytics {
        let recent_history: Vec<_> = self.adaptive_sizing.allocation_history
            .iter()
            .rev()
            .take(1000) // Last 1000 allocations
            .collect();

        if recent_history.is_empty() {
            return AllocationAnalytics::default();
        }

        let total_allocations = recent_history.len();
        let cache_hits = recent_history.iter().filter(|e| e.cache_hit).count();
        let avg_latency = recent_history
            .iter()
            .map(|e| e.latency_us as f64)
            .sum::<f64>()
            / total_allocations as f64;

        let avg_size =
            recent_history.iter().map(|e| e.size as f64).sum::<f64>() / total_allocations as f64;

        AllocationAnalytics {
            total_allocations,
            cache_hit_rate: cache_hits as f32 / total_allocations as f32,
            average_latency_us: avg_latency,
            average_allocation_size: avg_size as usize,
            memory_efficiency: self.stats.current_used as f32 / self.stats.total_allocated as f32,
            fragmentation_ratio: self.calculate_fragmentation_ratio(),
        }
    }

    /// Calculate memory fragmentation ratio
    fn calculate_fragmentation_ratio(&self) -> f32 {
        if self.free_blocks.is_empty() {
            return 0.0;
        }

        let total_free_blocks: usize = self.free_blocks.values().map(|blocks| blocks.len()).sum();

        let total_free_memory: usize = self
            .free_blocks
            .iter()
            .map(|(size, blocks)| size * blocks.len())
            .sum();

        if total_free_memory == 0 {
            return 0.0;
        }

        // Fragmentation ratio: more blocks with less total memory = higher fragmentation
        total_free_blocks as f32 / (total_free_memory as f32 / 1024.0) // Normalize by KB
    }

    /// Clear all cached memory
    pub fn clear(&mut self) -> Result<(), GpuOptimizerError> {
        // Free all GPU memory
        for block in &self.all_blocks {
            self.free_gpu_memory(block.ptr)?;
        }

        self.free_blocks.clear();
        self.all_blocks.clear();
        self.stats = MemoryStats::default();

        Ok(())
    }

    /// Allocate GPU memory (platform-specific)
    fn allocate_gpu_memory(&self, size: usize) -> Result<*mut u8, GpuOptimizerError> {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref context) = self.gpu_context {
                // Use context to allocate
                // In real implementation, would use cudaMalloc or hipMalloc
                Ok(ptr::null_mut()) // Placeholder
            } else {
                Err(GpuOptimizerError::NotInitialized)
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            Err(GpuOptimizerError::UnsupportedOperation(
                "GPU feature not enabled".to_string(),
            ))
        }
    }

    /// Free GPU memory (platform-specific)
    fn free_gpu_memory(&self, ptr: *mut u8) -> Result<(), GpuOptimizerError> {
        #[cfg(feature = "gpu")]
        {
            if let Some(ref _context) = self.gpu_context {
                // Use context to free
                // In real implementation, would use cudaFree or hipFree
                Ok(())
            } else {
                Err(GpuOptimizerError::NotInitialized)
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            Err(GpuOptimizerError::UnsupportedOperation(
                "GPU feature not enabled".to_string(),
            ))
        }
    }

    /// Generate comprehensive memory pool analytics report
    pub fn generate_memory_analytics_report(&self) -> String {
        let stats = &self.stats;
        let analytics = self.get_allocation_analytics();
        let pressure = self.get_memory_pressure();

        format!(
            "Memory Pool Analytics Report\n\
             ============================\n\
             \n\
             Memory Usage:\n\
               Total Allocated: {:.2} MB\n\
               Current Used: {:.2} MB\n\
               Peak Usage: {:.2} MB\n\
               Utilization: {:.1}%\n\
             \n\
             Allocation Performance:\n\
               Total Allocations: {}\n\
               Cache Hit Rate: {:.1}%\n\
               Average Latency: {:.2} μs\n\
               Average Size: {:.2} KB\n\
             \n\
             Memory Efficiency:\n\
               Memory Efficiency: {:.1}%\n\
               Fragmentation Ratio: {:.3}\n\
               Memory Pressure: {:.1}%\n\
             \n\
             Batch Buffers:\n\
               Active Buffers: {}\n\
               Total Buffer Memory: {:.2} MB\n\
             \n\
             Allocation Strategy: {:?}\n\
             Defragmentation: {}\n\
             Auto Cleanup: {}\n",
            stats.total_allocated as f64 / (1024.0 * 1024.0),
            stats.current_used as f64 / (1024.0 * 1024.0),
            stats.peak_usage as f64 / (1024.0 * 1024.0),
            if self.max_pool_size > 0 {
                100.0 * stats.current_used as f64 / self.max_pool_size as f64
            } else {
                0.0
            },
            analytics.total_allocations,
            analytics.cache_hit_rate * 100.0,
            analytics.average_latency_us,
            analytics.average_allocation_size as f64 / 1024.0,
            analytics.memory_efficiency * 100.0,
            analytics.fragmentation_ratio,
            pressure * 100.0,
            self.batch_buffers.iter().filter(|b| b.in_use).count(),
            self.batch_buffers.iter().map(|b| b.size).sum::<usize>() as f64 / (1024.0 * 1024.0),
            self.allocation_strategy,
            if self.enable_defrag { "Enabled" } else { "Disabled" },
            if self.pressure_monitor.auto_cleanup { "Enabled" } else { "Disabled" }
        )
    }

    /// Export detailed metrics for external monitoring systems
    pub fn export_metrics_json(&self) -> String {
        let stats = &self.stats;
        let analytics = self.get_allocation_analytics();
        let pressure_history: Vec<_> = self.pressure_monitor.pressure_history
            .iter()
            .map(|reading| {
                format!(
                    "{{\"timestamp\":\"{:?}\",\"pressure\":{:.3},\"available\":{},\"allocated\":{}}}",
                    reading.timestamp, reading.pressure, reading.available_memory, reading.allocated_memory
                )
            })
            .collect();

        format!(
            "{{\
                \"memory_stats\": {{\
                    \"total_allocated\": {},\
                    \"current_used\": {},\
                    \"peak_usage\": {},\
                    \"allocation_count\": {},\
                    \"deallocation_count\": {},\
                    \"cache_hits\": {},\
                    \"cache_misses\": {}\
                }},\
                \"analytics\": {{\
                    \"total_allocations\": {},\
                    \"cache_hit_rate\": {:.3},\
                    \"average_latency_us\": {:.2},\
                    \"average_allocation_size\": {},\
                    \"memory_efficiency\": {:.3},\
                    \"fragmentation_ratio\": {:.3}\
                }},\
                \"pressure_monitor\": {{\
                    \"current_pressure\": {:.3},\
                    \"threshold\": {:.3},\
                    \"auto_cleanup\": {},\
                    \"pressure_history\": [{}]\
                }},\
                \"batch_buffers\": {{\
                    \"total_buffers\": {},\
                    \"active_buffers\": {},\
                    \"total_memory\": {}\
                }},\
                \"configuration\": {{\
                    \"max_pool_size\": {},\
                    \"allocation_strategy\": \"{:?}\",\
                    \"defragmentation_enabled\": {},\
                    \"monitoring_enabled\": {}\
                }}\
            }}",
            stats.total_allocated,
            stats.current_used,
            stats.peak_usage,
            stats.allocation_count,
            stats.deallocation_count,
            stats.cache_hits,
            stats.cache_misses,
            analytics.total_allocations,
            analytics.cache_hit_rate,
            analytics.average_latency_us,
            analytics.average_allocation_size,
            analytics.memory_efficiency,
            analytics.fragmentation_ratio,
            self.pressure_monitor.current_pressure,
            self.pressure_monitor.pressure_threshold,
            self.pressure_monitor.auto_cleanup,
            pressure_history.join(","),
            self.batch_buffers.len(),
            self.batch_buffers.iter().filter(|b| b.in_use).count(),
            self.batch_buffers.iter().map(|b| b.size).sum::<usize>(),
            self.max_pool_size,
            self.allocation_strategy,
            self.enable_defrag,
            self.pressure_monitor.enable_monitoring
        )
    }
}

/// Thread-safe memory pool wrapper
pub struct ThreadSafeMemoryPool {
    pool: Arc<Mutex<CudaMemoryPool>>,
}

impl ThreadSafeMemoryPool {
    /// Create a new thread-safe memory pool
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pool: Arc::new(Mutex::new(CudaMemoryPool::new(max_pool_size))),
        }
    }

    /// Allocate memory
    pub fn allocate(&self, size: usize) -> Result<PooledMemory, GpuOptimizerError> {
        let ptr = self.pool.lock().unwrap().allocate(size)?;
        Ok(PooledMemory {
            ptr,
            size,
            pool: Arc::clone(&self.pool),
        })
    }

    /// Get statistics
    pub fn get_stats(&self) -> MemoryStats {
        self.pool.lock().unwrap().get_stats().clone()
    }

    /// Clear pool
    pub fn clear(&self) -> Result<(), GpuOptimizerError> {
        self.pool.lock().unwrap().clear()
    }
}

/// RAII wrapper for pooled memory
pub struct PooledMemory {
    ptr: *mut u8,
    size: usize,
    pool: Arc<Mutex<CudaMemoryPool>>,
}

impl PooledMemory {
    /// Get raw pointer
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Get size
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for PooledMemory {
    fn drop(&mut self) {
        self.pool.lock().unwrap().deallocate(self.ptr);
    }
}

unsafe impl Send for PooledMemory {}
unsafe impl Sync for PooledMemory {}

#[allow(dead_code)]
struct CudaKernel;

#[cfg(feature = "gpu")]
type CudaStream = scirs2_core::gpu::CudaStream;

#[cfg(not(feature = "gpu"))]
struct CudaStream;

/// Allocation analytics for performance monitoring
#[derive(Debug, Clone, Default)]
pub struct AllocationAnalytics {
    /// Total number of allocations analyzed
    pub total_allocations: usize,
    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f32,
    /// Average allocation latency (microseconds)
    pub average_latency_us: f64,
    /// Average allocation size (bytes)
    pub average_allocation_size: usize,
    /// Memory efficiency (used/allocated ratio)
    pub memory_efficiency: f32,
    /// Memory fragmentation ratio
    pub fragmentation_ratio: f32,
}

/// Memory pool configuration builder
pub struct MemoryPoolConfig {
    max_pool_size: usize,
    min_block_size: usize,
    enable_defrag: bool,
    defrag_interval: std::time::Duration,
    large_batch_config: LargeBatchConfig,
    allocation_strategy: AllocationStrategy,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 4 * 1024 * 1024 * 1024, // 4GB
            min_block_size: 256,
            enable_defrag: true,
            defrag_interval: std::time::Duration::from_secs(300), // 5 minutes
            large_batch_config: LargeBatchConfig::default(),
            allocation_strategy: AllocationStrategy::default(),
        }
    }
}

impl MemoryPoolConfig {
    /// Set maximum pool size
    pub fn max_size(mut self, size: usize) -> Self {
        self.max_pool_size = size;
        self
    }

    /// Set minimum block size
    pub fn min_block_size(mut self, size: usize) -> Self {
        self.min_block_size = size;
        self
    }

    /// Enable or disable defragmentation
    pub fn enable_defrag(mut self, enable: bool) -> Self {
        self.enable_defrag = enable;
        self
    }

    /// Set large batch configuration
    pub fn large_batch_config(mut self, config: LargeBatchConfig) -> Self {
        self.large_batch_config = config;
        self
    }

    /// Set allocation strategy
    pub fn allocation_strategy(mut self, strategy: AllocationStrategy) -> Self {
        self.allocation_strategy = strategy;
        self
    }

    /// Build the memory pool
    pub fn build(self) -> CudaMemoryPool {
        let mut pool = CudaMemoryPool::new(self.max_pool_size);
        pool.min_block_size = self.min_block_size;
        pool.enable_defrag = self.enable_defrag;
        pool.large_batch_config = self.large_batch_config;
        pool.allocation_strategy = self.allocation_strategy;
        pool
    }
}

impl fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemoryStats {{\n")?;
        write!(
            f,
            "  Total Allocated: {} MB\n",
            self.total_allocated / (1024 * 1024)
        )?;
        write!(
            f,
            "  Current Used: {} MB\n",
            self.current_used / (1024 * 1024)
        )?;
        write!(f, "  Peak Usage: {} MB\n", self.peak_usage / (1024 * 1024))?;
        write!(f, "  Allocations: {}\n", self.allocation_count)?;
        write!(f, "  Deallocations: {}\n", self.deallocation_count)?;
        write!(
            f,
            "  Cache Hit Rate: {:.2}%\n",
            if self.cache_hits + self.cache_misses > 0 {
                100.0 * self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
            } else {
                0.0
            }
        )?;
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let pool = CudaMemoryPool::new(1024 * 1024 * 1024);
        assert_eq!(pool.stats.total_allocated, 0);
        assert_eq!(pool.stats.current_used, 0);
    }

    #[test]
    fn test_memory_stats_display() {
        let mut stats = MemoryStats::default();
        stats.total_allocated = 1024 * 1024 * 100;
        stats.current_used = 1024 * 1024 * 50;
        stats.cache_hits = 80;
        stats.cache_misses = 20;

        let display = format!("{}", stats);
        assert!(display.contains("100 MB"));
        assert!(display.contains("50 MB"));
        assert!(display.contains("80.00%"));
    }

    #[test]
    fn test_thread_safe_pool() {
        let pool = ThreadSafeMemoryPool::new(1024 * 1024);
        let stats = pool.get_stats();
        assert_eq!(stats.total_allocated, 0);
    }

    #[test]
    fn test_memory_pool_config() {
        let config = MemoryPoolConfig::default()
            .max_size(2 * 1024 * 1024 * 1024)
            .min_block_size(512)
            .enable_defrag(false)
            .allocation_strategy(AllocationStrategy::BestFit);

        let pool = config.build();
        assert_eq!(pool.max_pool_size, 2 * 1024 * 1024 * 1024);
        assert_eq!(pool.min_block_size, 512);
        assert!(!pool.enable_defrag);
        assert_eq!(pool.allocation_strategy, AllocationStrategy::BestFit);
    }

    #[test]
    fn test_large_batch_config() {
        let config = LargeBatchConfig {
            min_batch_size: 2 * 1024 * 1024,
            max_batch_buffers: 8,
            growth_factor: 2.0,
            enable_coalescing: false,
            preallocation_threshold: 0.9,
            buffer_lifetime: 600,
        };

        assert_eq!(config.min_batch_size, 2 * 1024 * 1024);
        assert_eq!(config.max_batch_buffers, 8);
        assert_eq!(config.growth_factor, 2.0);
        assert!(!config.enable_coalescing);
        assert_eq!(config.preallocation_threshold, 0.9);
        assert_eq!(config.buffer_lifetime, 600);
    }

    #[test]
    fn test_batch_buffer() {
        let ptr = std::ptr::null_mut();
        let mut buffer = BatchBuffer::new(ptr, 1024, BatchBufferType::General);

        assert_eq!(buffer.size, 1024);
        assert!(!buffer.in_use);
        assert_eq!(buffer.usage_count, 0);
        assert_eq!(buffer.buffer_type, BatchBufferType::General);

        buffer.mark_used();
        assert!(buffer.in_use);
        assert_eq!(buffer.usage_count, 1);

        buffer.mark_free();
        assert!(!buffer.in_use);
    }

    #[test]
    fn test_allocation_strategies() {
        let strategies = [
            AllocationStrategy::FirstFit,
            AllocationStrategy::BestFit,
            AllocationStrategy::WorstFit,
            AllocationStrategy::BuddySystem,
            AllocationStrategy::SegregatedList,
            AllocationStrategy::Adaptive,
        ];

        for strategy in &strategies {
            let mut pool = CudaMemoryPool::new(1024 * 1024);
            pool.allocation_strategy = *strategy;
            assert_eq!(pool.allocation_strategy, *strategy);
        }
    }

    #[test]
    fn test_memory_pressure_monitor() {
        let mut monitor = MemoryPressureMonitor::default();
        assert!(monitor.enable_monitoring);
        assert_eq!(monitor.pressure_threshold, 0.9);
        assert_eq!(monitor.current_pressure, 0.0);
        assert!(monitor.pressure_history.is_empty());
    }

    #[test]
    fn test_adaptive_sizing() {
        let mut sizing = AdaptiveSizing::default();
        assert!(sizing.enable_adaptive_resize);
        assert_eq!(sizing.resize_threshold, 0.85);
        assert!(sizing.allocation_history.is_empty());

        let event = AllocationEvent {
            size: 1024,
            timestamp: std::time::Instant::now(),
            cache_hit: true,
            latency_us: 50,
        };

        sizing.allocation_history.push_back(event);
        assert_eq!(sizing.allocation_history.len(), 1);
    }

    #[test]
    fn test_allocation_analytics() {
        let analytics = AllocationAnalytics::default();
        assert_eq!(analytics.total_allocations, 0);
        assert_eq!(analytics.cache_hit_rate, 0.0);
        assert_eq!(analytics.average_latency_us, 0.0);
        assert_eq!(analytics.average_allocation_size, 0);
        assert_eq!(analytics.memory_efficiency, 0.0);
        assert_eq!(analytics.fragmentation_ratio, 0.0);
    }

    #[test]
    fn test_buffer_types() {
        let types = [
            BatchBufferType::General,
            BatchBufferType::GradientAccumulation,
            BatchBufferType::ParameterUpdate,
            BatchBufferType::Communication,
            BatchBufferType::Temporary,
        ];

        for buffer_type in &types {
            let buffer = BatchBuffer::new(std::ptr::null_mut(), 1024, *buffer_type);
            assert_eq!(buffer.buffer_type, *buffer_type);
        }
    }

    #[test]
    fn test_size_class_calculation() {
        let pool = CudaMemoryPool::new(1024 * 1024);

        assert_eq!(pool.get_size_class(100), 256);
        assert_eq!(pool.get_size_class(300), 512);
        assert_eq!(pool.get_size_class(1000), 1024);
        assert_eq!(pool.get_size_class(2000), 2048);
        assert_eq!(pool.get_size_class(1_000_000), 1048576);
        assert_eq!(pool.get_size_class(2_000_000), 2097152); // Next MB boundary
    }

    #[test]
    fn test_memory_pool_with_large_batch() {
        let config = LargeBatchConfig {
            min_batch_size: 1024,
            max_batch_buffers: 4,
            growth_factor: 1.5,
            enable_coalescing: true,
            preallocation_threshold: 0.8,
            buffer_lifetime: 300,
        };

        let pool = CudaMemoryPool::with_large_batch_config(1024 * 1024, config);
        assert_eq!(pool.large_batch_config.min_batch_size, 1024);
        assert_eq!(pool.large_batch_config.max_batch_buffers, 4);
        assert_eq!(pool.large_batch_config.growth_factor, 1.5);
        assert!(pool.large_batch_config.enable_coalescing);
    }

    #[test]
    fn test_pressure_reading() {
        let reading = PressureReading {
            timestamp: std::time::Instant::now(),
            pressure: 0.75,
            available_memory: 256 * 1024 * 1024,
            allocated_memory: 768 * 1024 * 1024,
        };

        assert_eq!(reading.pressure, 0.75);
        assert_eq!(reading.available_memory, 256 * 1024 * 1024);
        assert_eq!(reading.allocated_memory, 768 * 1024 * 1024);
    }
}
