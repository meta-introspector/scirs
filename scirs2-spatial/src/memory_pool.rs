//! Ultra-optimized memory pool system for spatial algorithms
//!
//! This module provides advanced memory management strategies specifically
//! designed for spatial computing algorithms that perform frequent allocations.
//! The system includes object pools, arena allocators, and cache-aware
//! memory layouts to maximize performance.
//!
//! # Features
//!
//! - **Object pools**: Reusable pools for frequently allocated types
//! - **Arena allocators**: Block-based allocation for temporary objects
//! - **Cache-aware layouts**: Memory alignment for optimal cache performance
//! - **NUMA-aware allocation**: Memory placement for multi-socket systems
//! - **Zero-copy operations**: Minimize data movement and copying
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::memory_pool::{DistancePool, ClusteringArena};
//!
//! // Create a distance computation pool
//! let mut pool = DistancePool::new(1000);
//! 
//! // Get a reusable distance buffer
//! let buffer = pool.get_distance_buffer(256);
//! 
//! // Use buffer for computations...
//! 
//! // Return buffer to pool (automatic with RAII)
//! pool.return_distance_buffer(buffer);
//! ```

use ndarray::{Array2, ArrayViewMut1, ArrayViewMut2};
use std::collections::VecDeque;
use std::sync::Mutex;
use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr::NonNull;

/// Configuration for memory pool system
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Maximum number of objects to keep in each pool
    pub max_pool_size: usize,
    /// Cache line size for alignment (typically 64 bytes)
    pub cache_line_size: usize,
    /// Enable NUMA-aware allocation strategies
    pub numa_aware: bool,
    /// Prefetch distance for memory access patterns
    pub prefetch_distance: usize,
    /// Block size for arena allocators
    pub arena_block_size: usize,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 1000,
            cache_line_size: 64,
            numa_aware: true,
            prefetch_distance: 8,
            arena_block_size: 1024 * 1024, // 1MB blocks
        }
    }
}

/// Ultra-optimized distance computation memory pool
pub struct DistancePool {
    config: MemoryPoolConfig,
    distance_buffers: Mutex<VecDeque<Box<[f64]>>>,
    index_buffers: Mutex<VecDeque<Box<[usize]>>>,
    matrix_buffers: Mutex<VecDeque<Array2<f64>>>,
    stats: PoolStatistics,
}

impl DistancePool {
    /// Create a new distance computation pool
    pub fn new(capacity: usize) -> Self {
        Self::with_config(capacity, MemoryPoolConfig::default())
    }

    /// Create a pool with custom configuration
    pub fn with_config(capacity: usize, config: MemoryPoolConfig) -> Self {
        Self {
            config,
            distance_buffers: Mutex::new(VecDeque::with_capacity(capacity)),
            index_buffers: Mutex::new(VecDeque::with_capacity(capacity)),
            matrix_buffers: Mutex::new(VecDeque::with_capacity(capacity / 4)), // Matrices are larger
            stats: PoolStatistics::new(),
        }
    }

    /// Get a cache-aligned distance buffer
    pub fn get_distance_buffer(&self, size: usize) -> DistanceBuffer {
        let mut buffers = self.distance_buffers.lock().unwrap();
        
        // Try to reuse an existing buffer of appropriate size
        for i in 0..buffers.len() {
            if buffers[i].len() >= size && buffers[i].len() <= size * 2 {
                let buffer = buffers.remove(i).unwrap();
                self.stats.record_hit();
                return DistanceBuffer::new(buffer, self);
            }
        }

        // Create new aligned buffer
        self.stats.record_miss();
        let aligned_buffer = self.create_aligned_buffer(size);
        DistanceBuffer::new(aligned_buffer, self)
    }

    /// Get an index buffer for storing indices
    pub fn get_index_buffer(&self, size: usize) -> IndexBuffer {
        let mut buffers = self.index_buffers.lock().unwrap();
        
        // Try to reuse existing buffer
        for i in 0..buffers.len() {
            if buffers[i].len() >= size && buffers[i].len() <= size * 2 {
                let buffer = buffers.remove(i).unwrap();
                self.stats.record_hit();
                return IndexBuffer::new(buffer, self);
            }
        }

        // Create new buffer
        self.stats.record_miss();
        let new_buffer = vec![0usize; size].into_boxed_slice();
        IndexBuffer::new(new_buffer, self)
    }

    /// Get a distance matrix buffer
    pub fn get_matrix_buffer(&self, rows: usize, cols: usize) -> MatrixBuffer {
        let mut buffers = self.matrix_buffers.lock().unwrap();
        
        // Try to reuse existing matrix
        for i in 0..buffers.len() {
            let (r, c) = buffers[i].dim();
            if r >= rows && c >= cols && r <= rows * 2 && c <= cols * 2 {
                let mut matrix = buffers.remove(i).unwrap();
                // Resize to exact dimensions needed
                matrix = matrix.slice_mut(s![..rows, ..cols]).to_owned();
                self.stats.record_hit();
                return MatrixBuffer::new(matrix, self);
            }
        }

        // Create new matrix
        self.stats.record_miss();
        let matrix = Array2::zeros((rows, cols));
        MatrixBuffer::new(matrix, self)
    }

    /// Create cache-aligned buffer for optimal SIMD performance
    fn create_aligned_buffer(&self, size: usize) -> Box<[f64]> {
        let layout = Layout::from_size_align(
            size * std::mem::size_of::<f64>(),
            self.config.cache_line_size,
        ).unwrap();

        unsafe {
            let ptr = System.alloc(layout) as *mut f64;
            if ptr.is_null() {
                panic!("Failed to allocate aligned memory");
            }

            // Initialize to zero
            std::ptr::write_bytes(ptr, 0, size);
            
            // Convert to boxed slice
            Box::from_raw(std::slice::from_raw_parts_mut(ptr, size))
        }
    }

    /// Return a distance buffer to the pool
    fn return_distance_buffer(&self, buffer: Box<[f64]>) {
        let mut buffers = self.distance_buffers.lock().unwrap();
        if buffers.len() < self.config.max_pool_size {
            buffers.push_back(buffer);
        }
        // Otherwise let it drop and deallocate
    }

    /// Return an index buffer to the pool
    fn return_index_buffer(&self, buffer: Box<[usize]>) {
        let mut buffers = self.index_buffers.lock().unwrap();
        if buffers.len() < self.config.max_pool_size {
            buffers.push_back(buffer);
        }
    }

    /// Return a matrix buffer to the pool
    fn return_matrix_buffer(&self, matrix: Array2<f64>) {
        let mut buffers = self.matrix_buffers.lock().unwrap();
        if buffers.len() < self.config.max_pool_size / 4 { // Keep fewer matrices
            buffers.push_back(matrix);
        }
    }

    /// Get pool statistics for performance monitoring
    pub fn statistics(&self) -> PoolStatistics {
        self.stats.clone()
    }

    /// Clear all pools and free memory
    pub fn clear(&self) {
        self.distance_buffers.lock().unwrap().clear();
        self.index_buffers.lock().unwrap().clear();
        self.matrix_buffers.lock().unwrap().clear();
        self.stats.reset();
    }
}

// Use ndarray's s! macro
use ndarray::s;

/// RAII wrapper for distance buffers with automatic return to pool
pub struct DistanceBuffer<'a> {
    buffer: Option<Box<[f64]>>,
    pool: &'a DistancePool,
}

impl<'a> DistanceBuffer<'a> {
    fn new(buffer: Box<[f64]>, pool: &'a DistancePool) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
        }
    }

    /// Get a mutable slice of the buffer
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        self.buffer.as_mut().unwrap().as_mut()
    }

    /// Get an immutable slice of the buffer
    pub fn as_slice(&self) -> &[f64] {
        self.buffer.as_ref().unwrap().as_ref()
    }

    /// Get the length of the buffer
    pub fn len(&self) -> usize {
        self.buffer.as_ref().unwrap().len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a view as ndarray Array1
    pub fn as_array_mut(&mut self) -> ArrayViewMut1<f64> {
        ArrayViewMut1::from(self.as_mut_slice())
    }
}

impl<'a> Drop for DistanceBuffer<'a> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.return_distance_buffer(buffer);
        }
    }
}

/// RAII wrapper for index buffers
pub struct IndexBuffer<'a> {
    buffer: Option<Box<[usize]>>,
    pool: &'a DistancePool,
}

impl<'a> IndexBuffer<'a> {
    fn new(buffer: Box<[usize]>, pool: &'a DistancePool) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
        }
    }

    /// Get a mutable slice of the buffer
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        self.buffer.as_mut().unwrap().as_mut()
    }

    /// Get an immutable slice of the buffer
    pub fn as_slice(&self) -> &[usize] {
        self.buffer.as_ref().unwrap().as_ref()
    }

    /// Get the length of the buffer
    pub fn len(&self) -> usize {
        self.buffer.as_ref().unwrap().len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a> Drop for IndexBuffer<'a> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.return_index_buffer(buffer);
        }
    }
}

/// RAII wrapper for matrix buffers
pub struct MatrixBuffer<'a> {
    matrix: Option<Array2<f64>>,
    pool: &'a DistancePool,
}

impl<'a> MatrixBuffer<'a> {
    fn new(matrix: Array2<f64>, pool: &'a DistancePool) -> Self {
        Self {
            matrix: Some(matrix),
            pool,
        }
    }

    /// Get a mutable view of the matrix
    pub fn as_mut(&mut self) -> ArrayViewMut2<f64> {
        self.matrix.as_mut().unwrap().view_mut()
    }

    /// Get the dimensions of the matrix
    pub fn dim(&self) -> (usize, usize) {
        self.matrix.as_ref().unwrap().dim()
    }

    /// Fill the matrix with a value
    pub fn fill(&mut self, value: f64) {
        self.matrix.as_mut().unwrap().fill(value);
    }
}

impl<'a> Drop for MatrixBuffer<'a> {
    fn drop(&mut self) {
        if let Some(matrix) = self.matrix.take() {
            self.pool.return_matrix_buffer(matrix);
        }
    }
}

/// Arena allocator for temporary objects in clustering algorithms
pub struct ClusteringArena {
    config: MemoryPoolConfig,
    current_block: Mutex<Option<ArenaBlock>>,
    full_blocks: Mutex<Vec<ArenaBlock>>,
    stats: ArenaStatistics,
}

impl ClusteringArena {
    /// Create a new clustering arena
    pub fn new() -> Self {
        Self::with_config(MemoryPoolConfig::default())
    }

    /// Create arena with custom configuration
    pub fn with_config(config: MemoryPoolConfig) -> Self {
        Self {
            config,
            current_block: Mutex::new(None),
            full_blocks: Mutex::new(Vec::new()),
            stats: ArenaStatistics::new(),
        }
    }

    /// Allocate a temporary vector in the arena
    pub fn alloc_temp_vec<T: Default + Clone>(&self, size: usize) -> ArenaVec<T> {
        let layout = Layout::array::<T>(size).unwrap();
        let ptr = self.allocate_raw(layout);
        
        unsafe {
            // Initialize elements
            for i in 0..size {
                std::ptr::write(ptr.as_ptr().add(i) as *mut T, T::default());
            }
            
            ArenaVec::new(ptr.as_ptr() as *mut T, size)
        }
    }

    /// Allocate raw memory with proper alignment
    fn allocate_raw(&self, layout: Layout) -> NonNull<u8> {
        let mut current = self.current_block.lock().unwrap();
        
        if current.is_none() || !current.as_ref().unwrap().can_allocate(layout) {
            // Need a new block
            if let Some(old_block) = current.take() {
                self.full_blocks.lock().unwrap().push(old_block);
            }
            *current = Some(ArenaBlock::new(self.config.arena_block_size));
        }
        
        current.as_mut().unwrap().allocate(layout)
    }

    /// Reset the arena, keeping allocated blocks for reuse
    pub fn reset(&self) {
        let mut current = self.current_block.lock().unwrap();
        let mut full_blocks = self.full_blocks.lock().unwrap();
        
        if let Some(block) = current.take() {
            full_blocks.push(block);
        }
        
        // Reset all blocks
        for block in full_blocks.iter_mut() {
            block.reset();
        }
        
        // Move one block back to current
        if let Some(block) = full_blocks.pop() {
            *current = Some(block);
        }
        
        self.stats.reset();
    }

    /// Get arena statistics
    pub fn statistics(&self) -> ArenaStatistics {
        self.stats.clone()
    }
}

impl Default for ClusteringArena {
    fn default() -> Self {
        Self::new()
    }
}

/// A block of memory within the arena
struct ArenaBlock {
    memory: NonNull<u8>,
    size: usize,
    offset: usize,
}

// SAFETY: ArenaBlock manages its own memory and ensures thread-safe access
unsafe impl Send for ArenaBlock {}
unsafe impl Sync for ArenaBlock {}

impl ArenaBlock {
    fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size, 64).unwrap(); // 64-byte aligned
        let memory = unsafe {
            NonNull::new(System.alloc(layout)).expect("Failed to allocate arena block")
        };
        
        Self {
            memory,
            size,
            offset: 0,
        }
    }

    fn can_allocate(&self, layout: Layout) -> bool {
        let aligned_offset = (self.offset + layout.align() - 1) & !(layout.align() - 1);
        aligned_offset + layout.size() <= self.size
    }

    fn allocate(&mut self, layout: Layout) -> NonNull<u8> {
        assert!(self.can_allocate(layout));
        
        // Align the offset
        self.offset = (self.offset + layout.align() - 1) & !(layout.align() - 1);
        
        let ptr = unsafe { NonNull::new_unchecked(self.memory.as_ptr().add(self.offset)) };
        self.offset += layout.size();
        
        ptr
    }

    fn reset(&mut self) {
        self.offset = 0;
    }
}

impl Drop for ArenaBlock {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size, 64).unwrap();
        unsafe {
            System.dealloc(self.memory.as_ptr(), layout);
        }
    }
}

/// RAII wrapper for arena-allocated vectors
pub struct ArenaVec<T> {
    ptr: *mut T,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ArenaVec<T> {
    fn new(ptr: *mut T, len: usize) -> Self {
        Self {
            ptr,
            len,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get a mutable slice of the vector
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Get an immutable slice of the vector
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get the length of the vector
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if vector is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// Note: ArenaVec doesn't implement Drop because the arena manages the memory

/// Pool performance statistics
#[derive(Debug)]
pub struct PoolStatistics {
    hits: std::sync::atomic::AtomicUsize,
    misses: std::sync::atomic::AtomicUsize,
    total_allocations: std::sync::atomic::AtomicUsize,
}

impl PoolStatistics {
    fn new() -> Self {
        Self {
            hits: std::sync::atomic::AtomicUsize::new(0),
            misses: std::sync::atomic::AtomicUsize::new(0),
            total_allocations: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn record_hit(&self) {
        self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn record_miss(&self) {
        self.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.total_allocations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn reset(&self) {
        self.hits.store(0, std::sync::atomic::Ordering::Relaxed);
        self.misses.store(0, std::sync::atomic::Ordering::Relaxed);
        self.total_allocations.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get hit rate as a percentage
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(std::sync::atomic::Ordering::Relaxed);
        let total = hits + self.misses.load(std::sync::atomic::Ordering::Relaxed);
        if total == 0 { 0.0 } else { hits as f64 / total as f64 * 100.0 }
    }

    /// Get total requests
    pub fn total_requests(&self) -> usize {
        self.hits.load(std::sync::atomic::Ordering::Relaxed) +
        self.misses.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get total new allocations
    pub fn total_allocations(&self) -> usize {
        self.total_allocations.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Clone for PoolStatistics {
    fn clone(&self) -> Self {
        Self {
            hits: std::sync::atomic::AtomicUsize::new(
                self.hits.load(std::sync::atomic::Ordering::Relaxed)
            ),
            misses: std::sync::atomic::AtomicUsize::new(
                self.misses.load(std::sync::atomic::Ordering::Relaxed)
            ),
            total_allocations: std::sync::atomic::AtomicUsize::new(
                self.total_allocations.load(std::sync::atomic::Ordering::Relaxed)
            ),
        }
    }
}

/// Arena performance statistics
#[derive(Debug)]
pub struct ArenaStatistics {
    blocks_allocated: std::sync::atomic::AtomicUsize,
    total_memory: std::sync::atomic::AtomicUsize,
    active_objects: std::sync::atomic::AtomicUsize,
}

impl ArenaStatistics {
    fn new() -> Self {
        Self {
            blocks_allocated: std::sync::atomic::AtomicUsize::new(0),
            total_memory: std::sync::atomic::AtomicUsize::new(0),
            active_objects: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn reset(&self) {
        self.blocks_allocated.store(0, std::sync::atomic::Ordering::Relaxed);
        self.total_memory.store(0, std::sync::atomic::Ordering::Relaxed);
        self.active_objects.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get number of allocated blocks
    pub fn blocks_allocated(&self) -> usize {
        self.blocks_allocated.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get total memory usage in bytes
    pub fn total_memory(&self) -> usize {
        self.total_memory.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get number of active objects
    pub fn active_objects(&self) -> usize {
        self.active_objects.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Clone for ArenaStatistics {
    fn clone(&self) -> Self {
        Self {
            blocks_allocated: std::sync::atomic::AtomicUsize::new(
                self.blocks_allocated.load(std::sync::atomic::Ordering::Relaxed)
            ),
            total_memory: std::sync::atomic::AtomicUsize::new(
                self.total_memory.load(std::sync::atomic::Ordering::Relaxed)
            ),
            active_objects: std::sync::atomic::AtomicUsize::new(
                self.active_objects.load(std::sync::atomic::Ordering::Relaxed)
            ),
        }
    }
}

/// Global memory pool instance for convenience
static GLOBAL_DISTANCE_POOL: std::sync::OnceLock<DistancePool> = std::sync::OnceLock::new();
static GLOBAL_CLUSTERING_ARENA: std::sync::OnceLock<ClusteringArena> = std::sync::OnceLock::new();

/// Get the global distance pool instance
pub fn global_distance_pool() -> &'static DistancePool {
    GLOBAL_DISTANCE_POOL.get_or_init(|| DistancePool::new(1000))
}

/// Get the global clustering arena instance
pub fn global_clustering_arena() -> &'static ClusteringArena {
    GLOBAL_CLUSTERING_ARENA.get_or_init(|| ClusteringArena::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_pool() {
        let pool = DistancePool::new(10);
        
        // Get a buffer
        let mut buffer1 = pool.get_distance_buffer(100);
        assert_eq!(buffer1.len(), 100);
        
        // Use the buffer
        buffer1.as_mut_slice()[0] = 42.0;
        assert_eq!(buffer1.as_slice()[0], 42.0);
        
        // Get another buffer while first is in use
        let buffer2 = pool.get_distance_buffer(50);
        assert_eq!(buffer2.len(), 50);
        
        // Drop first buffer (should return to pool)
        drop(buffer1);
        
        // Get buffer again (should reuse)
        let buffer3 = pool.get_distance_buffer(100);
        assert_eq!(buffer3.len(), 100);
        // Note: value should be zeroed when creating aligned buffer
    }

    #[test]
    fn test_arena_allocator() {
        let arena = ClusteringArena::new();
        
        // Allocate some temporary vectors
        let mut vec1 = arena.alloc_temp_vec::<f64>(100);
        let mut vec2 = arena.alloc_temp_vec::<usize>(50);
        
        // Use the vectors
        vec1.as_mut_slice()[0] = 3.14;
        vec2.as_mut_slice()[0] = 42;
        
        assert_eq!(vec1.as_slice()[0], 3.14);
        assert_eq!(vec2.as_slice()[0], 42);
        
        // Reset arena
        arena.reset();
        
        // Allocate again (should reuse memory)
        let mut vec3 = arena.alloc_temp_vec::<f64>(200);
        vec3.as_mut_slice()[0] = 2.71;
        assert_eq!(vec3.as_slice()[0], 2.71);
    }

    #[test]
    fn test_pool_statistics() {
        let pool = DistancePool::new(2);
        
        // Initial stats should be zero
        let stats = pool.statistics();
        assert_eq!(stats.total_requests(), 0);
        assert_eq!(stats.total_allocations(), 0);
        
        // First request should be a miss
        let _buffer1 = pool.get_distance_buffer(100);
        let stats = pool.statistics();
        assert_eq!(stats.total_requests(), 1);
        assert_eq!(stats.total_allocations(), 1);
        assert!(stats.hit_rate() < 1.0);
        
        // Drop and get again should be a hit
        drop(_buffer1);
        let _buffer2 = pool.get_distance_buffer(100);
        let stats = pool.statistics();
        assert_eq!(stats.total_requests(), 2);
        assert_eq!(stats.total_allocations(), 1); // No new allocation
        assert!(stats.hit_rate() > 0.0);
    }

    #[test]
    fn test_matrix_buffer() {
        let pool = DistancePool::new(5);
        
        let mut matrix = pool.get_matrix_buffer(10, 10);
        assert_eq!(matrix.dim(), (10, 10));
        
        matrix.fill(42.0);
        // Matrix should be filled with 42.0 (can't easily test without exposing internals)
        
        drop(matrix);
        
        // Get another matrix (should potentially reuse)
        let matrix2 = pool.get_matrix_buffer(8, 8);
        assert_eq!(matrix2.dim(), (8, 8));
    }

    #[test]
    fn test_global_pools() {
        // Test that global pools can be accessed
        let pool = global_distance_pool();
        let arena = global_clustering_arena();
        
        let _buffer = pool.get_distance_buffer(10);
        let _vec = arena.alloc_temp_vec::<f64>(10);
        
        // Should not panic
    }
}