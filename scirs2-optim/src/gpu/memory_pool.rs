//! CUDA memory pool management for efficient GPU memory allocation
//! 
//! This module provides memory pooling to reduce allocation overhead
//! and improve performance for repeated GPU operations.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::ptr;
use std::fmt;

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
            
            Err(GpuOptimizerError::InvalidState(
                format!("Memory pool limit exceeded: requested {}, available {}", 
                        aligned_size, self.max_pool_size - self.stats.total_allocated)
            ))
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
                "GPU feature not enabled".to_string()
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
                "GPU feature not enabled".to_string()
            ))
        }
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

/// Memory pool configuration builder
pub struct MemoryPoolConfig {
    max_pool_size: usize,
    min_block_size: usize,
    enable_defrag: bool,
    defrag_interval: std::time::Duration,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 4 * 1024 * 1024 * 1024, // 4GB
            min_block_size: 256,
            enable_defrag: true,
            defrag_interval: std::time::Duration::from_secs(300), // 5 minutes
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
    
    /// Build the memory pool
    pub fn build(self) -> CudaMemoryPool {
        let mut pool = CudaMemoryPool::new(self.max_pool_size);
        pool.min_block_size = self.min_block_size;
        pool.enable_defrag = self.enable_defrag;
        pool
    }
}

impl fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MemoryStats {{\n")?;
        write!(f, "  Total Allocated: {} MB\n", self.total_allocated / (1024 * 1024))?;
        write!(f, "  Current Used: {} MB\n", self.current_used / (1024 * 1024))?;
        write!(f, "  Peak Usage: {} MB\n", self.peak_usage / (1024 * 1024))?;
        write!(f, "  Allocations: {}\n", self.allocation_count)?;
        write!(f, "  Deallocations: {}\n", self.deallocation_count)?;
        write!(f, "  Cache Hit Rate: {:.2}%\n", 
               if self.cache_hits + self.cache_misses > 0 {
                   100.0 * self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
               } else {
                   0.0
               })?;
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
            .enable_defrag(false);
        
        let pool = config.build();
        assert_eq!(pool.max_pool_size, 2 * 1024 * 1024 * 1024);
        assert_eq!(pool.min_block_size, 512);
        assert!(!pool.enable_defrag);
    }
}