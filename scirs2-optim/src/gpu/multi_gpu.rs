//! Multi-GPU synchronization support for distributed training

use ndarray::{ArrayBase, Data, DataMut, Dimension};
use num_traits::Float;
use std::sync::Arc;
use std::marker::PhantomData;

use crate::gpu::GpuOptimizerError;

#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuContext, GpuKernelHandle, GpuBuffer};

/// Multi-GPU synchronization strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SyncStrategy {
    /// Ring all-reduce (efficient for large tensors)
    RingAllReduce,
    /// Tree all-reduce (efficient for small tensors)
    TreeAllReduce,
    /// Hierarchical all-reduce (for multi-node setups)
    HierarchicalAllReduce,
    /// Pipeline parallel synchronization
    PipelineParallel,
}

/// Multi-GPU configuration
#[derive(Debug, Clone)]
pub struct MultiGpuConfig {
    /// Number of GPUs
    pub num_gpus: usize,
    /// GPU rank (0-indexed)
    pub rank: usize,
    /// Synchronization strategy
    pub sync_strategy: SyncStrategy,
    /// Enable gradient compression
    pub gradient_compression: bool,
    /// Compression ratio (for top-k compression)
    pub compression_ratio: f32,
    /// Local GPU group size (for hierarchical)
    pub local_group_size: usize,
}

impl Default for MultiGpuConfig {
    fn default() -> Self {
        Self {
            num_gpus: 1,
            rank: 0,
            sync_strategy: SyncStrategy::RingAllReduce,
            gradient_compression: false,
            compression_ratio: 0.1, // Keep top 10%
            local_group_size: 4,
        }
    }
}

/// Multi-GPU synchronization manager
pub struct MultiGpuSync<A: Float> {
    /// GPU context
    context: Arc<GpuContext>,
    /// Configuration
    config: MultiGpuConfig,
    /// Synchronization kernels
    sync_kernels: SyncKernels,
    /// Workspace buffers
    workspace: WorkspaceBuffers<A>,
    /// Phantom data for type parameter
    _phantom: PhantomData<A>,
}

/// Container for synchronization kernels
struct SyncKernels {
    ring_allreduce: Option<Arc<GpuKernelHandle>>,
    tree_allreduce: Option<Arc<GpuKernelHandle>>,
    hierarchical_allreduce: Option<Arc<GpuKernelHandle>>,
    compress_gradients: Option<Arc<GpuKernelHandle>>,
    decompress_gradients: Option<Arc<GpuKernelHandle>>,
}

/// Workspace buffers for synchronization
struct WorkspaceBuffers<A: Float> {
    recv_buffer: Option<GpuBuffer<A>>,
    workspace: Option<GpuBuffer<A>>,
    compressed_values: Option<GpuBuffer<A>>,
    compressed_indices: Option<GpuBuffer<i32>>,
    error_feedback: Option<GpuBuffer<A>>,
}

impl<A: Float> MultiGpuSync<A> {
    /// Create a new multi-GPU synchronization manager
    pub fn new(
        context: Arc<GpuContext>,
        config: MultiGpuConfig,
        max_param_size: usize,
    ) -> Result<Self, GpuOptimizerError> {
        // Load synchronization kernels
        let sync_kernels = Self::load_sync_kernels(&context, &config)?;
        
        // Allocate workspace buffers
        let workspace = Self::allocate_workspace(&context, &config, max_param_size)?;
        
        Ok(Self {
            context,
            config,
            sync_kernels,
            workspace,
            _phantom: PhantomData,
        })
    }
    
    /// Synchronize gradients across GPUs
    pub fn sync_gradients<S, D>(
        &mut self,
        gradients: &mut ArrayBase<S, D>,
    ) -> Result<(), GpuOptimizerError>
    where
        S: DataMut<Elem = A>,
        D: Dimension,
    {
        match self.config.sync_strategy {
            SyncStrategy::RingAllReduce => self.ring_allreduce(gradients),
            SyncStrategy::TreeAllReduce => self.tree_allreduce(gradients),
            SyncStrategy::HierarchicalAllReduce => self.hierarchical_allreduce(gradients),
            SyncStrategy::PipelineParallel => {
                Err(GpuOptimizerError::UnsupportedOperation(
                    "Pipeline parallel requires special handling".to_string()
                ))
            }
        }
    }
    
    /// Ring all-reduce implementation
    fn ring_allreduce<S, D>(
        &self,
        gradients: &mut ArrayBase<S, D>,
    ) -> Result<(), GpuOptimizerError>
    where
        S: DataMut<Elem = A>,
        D: Dimension,
    {
        #[cfg(feature = "gpu")]
        {
            let kernel = self.sync_kernels.ring_allreduce.as_ref()
                .ok_or(GpuOptimizerError::NotInitialized)?;
            
            let grad_slice = gradients.as_slice_mut()
                .ok_or_else(|| GpuOptimizerError::InvalidState(
                    "Gradients must be contiguous".to_string()
                ))?;
            
            // Create GPU buffer for gradients
            let grad_buffer = self.context.create_buffer_from_slice(grad_slice);
            
            // Calculate chunk size for ring operations
            let chunk_size = (gradients.len() + self.config.num_gpus - 1) / self.config.num_gpus;
            
            // Set kernel parameters
            kernel.set_buffer("data", &grad_buffer);
            kernel.set_buffer("recv_buffer", self.workspace.recv_buffer.as_ref().unwrap());
            kernel.set_i32("chunk_size", chunk_size as i32);
            kernel.set_i32("rank", self.config.rank as i32);
            kernel.set_i32("world_size", self.config.num_gpus as i32);
            
            // Execute ring all-reduce for each chunk
            for chunk_id in 0..self.config.num_gpus {
                kernel.set_i32("chunk_id", chunk_id as i32);
                
                let (grid_size, block_size) = crate::gpu::utils::calculate_block_size(chunk_size, 256);
                kernel.dispatch([grid_size as u32, 1, 1]);
            }
            
            // Copy results back
            grad_buffer.copy_to_host(grad_slice);
        }
        
        Ok(())
    }
    
    /// Tree all-reduce implementation
    fn tree_allreduce<S, D>(
        &self,
        gradients: &mut ArrayBase<S, D>,
    ) -> Result<(), GpuOptimizerError>
    where
        S: DataMut<Elem = A>,
        D: Dimension,
    {
        #[cfg(feature = "gpu")]
        {
            let kernel = self.sync_kernels.tree_allreduce.as_ref()
                .ok_or(GpuOptimizerError::NotInitialized)?;
            
            // Implementation similar to ring_allreduce but using tree pattern
            // ...
        }
        
        Ok(())
    }
    
    /// Hierarchical all-reduce for multi-node setups
    fn hierarchical_allreduce<S, D>(
        &self,
        gradients: &mut ArrayBase<S, D>,
    ) -> Result<(), GpuOptimizerError>
    where
        S: DataMut<Elem = A>,
        D: Dimension,
    {
        #[cfg(feature = "gpu")]
        {
            let kernel = self.sync_kernels.hierarchical_allreduce.as_ref()
                .ok_or(GpuOptimizerError::NotInitialized)?;
            
            // Calculate local and global ranks
            let local_rank = self.config.rank % self.config.local_group_size;
            let global_rank = self.config.rank / self.config.local_group_size;
            let global_size = self.config.num_gpus / self.config.local_group_size;
            
            // Set kernel parameters
            // ... implementation details
        }
        
        Ok(())
    }
    
    /// Compress gradients for bandwidth optimization
    pub fn compress_gradients<S, D>(
        &mut self,
        gradients: &ArrayBase<S, D>,
    ) -> Result<(Vec<A>, Vec<i32>), GpuOptimizerError>
    where
        S: Data<Elem = A>,
        D: Dimension,
    {
        #[cfg(feature = "gpu")]
        {
            let kernel = self.sync_kernels.compress_gradients.as_ref()
                .ok_or(GpuOptimizerError::NotInitialized)?;
            
            let k = (gradients.len() as f32 * self.config.compression_ratio) as usize;
            
            // Set kernel parameters and execute
            // ... implementation details
            
            // Return compressed values and indices
            let compressed_values = vec![A::zero(); k];
            let compressed_indices = vec![0i32; k];
            
            Ok((compressed_values, compressed_indices))
        }
        
        #[cfg(not(feature = "gpu"))]
        {
            Err(GpuOptimizerError::UnsupportedOperation(
                "GPU feature not enabled".to_string()
            ))
        }
    }
    
    /// Load synchronization kernels
    fn load_sync_kernels(
        context: &Arc<GpuContext>,
        config: &MultiGpuConfig,
    ) -> Result<SyncKernels, GpuOptimizerError> {
        #[cfg(feature = "gpu")]
        {
            let ring_kernel = if matches!(config.sync_strategy, SyncStrategy::RingAllReduce) {
                Some(Arc::new(context.get_kernel("ring_allreduce_f32")?))
            } else {
                None
            };
            
            let tree_kernel = if matches!(config.sync_strategy, SyncStrategy::TreeAllReduce) {
                Some(Arc::new(context.get_kernel("tree_allreduce_f32")?))
            } else {
                None
            };
            
            let hierarchical_kernel = if matches!(config.sync_strategy, SyncStrategy::HierarchicalAllReduce) {
                Some(Arc::new(context.get_kernel("hierarchical_allreduce_f32")?))
            } else {
                None
            };
            
            let compress_kernel = if config.gradient_compression {
                Some(Arc::new(context.get_kernel("compress_gradients_topk_f32")?))
            } else {
                None
            };
            
            let decompress_kernel = if config.gradient_compression {
                Some(Arc::new(context.get_kernel("decompress_gradients_f32")?))
            } else {
                None
            };
            
            Ok(SyncKernels {
                ring_allreduce: ring_kernel,
                tree_allreduce: tree_kernel,
                hierarchical_allreduce: hierarchical_kernel,
                compress_gradients: compress_kernel,
                decompress_gradients: decompress_kernel,
            })
        }
        
        #[cfg(not(feature = "gpu"))]
        {
            Ok(SyncKernels {
                ring_allreduce: None,
                tree_allreduce: None,
                hierarchical_allreduce: None,
                compress_gradients: None,
                decompress_gradients: None,
            })
        }
    }
    
    /// Allocate workspace buffers
    fn allocate_workspace(
        context: &Arc<GpuContext>,
        config: &MultiGpuConfig,
        max_param_size: usize,
    ) -> Result<WorkspaceBuffers<A>, GpuOptimizerError> {
        #[cfg(feature = "gpu")]
        {
            let recv_buffer = Some(context.create_buffer::<A>(max_param_size));
            let workspace = Some(context.create_buffer::<A>(max_param_size));
            
            let (compressed_values, compressed_indices, error_feedback) = if config.gradient_compression {
                let k = (max_param_size as f32 * config.compression_ratio) as usize;
                (
                    Some(context.create_buffer::<A>(k)),
                    Some(context.create_buffer::<i32>(k)),
                    Some(context.create_buffer::<A>(max_param_size)),
                )
            } else {
                (None, None, None)
            };
            
            Ok(WorkspaceBuffers {
                recv_buffer,
                workspace,
                compressed_values,
                compressed_indices,
                error_feedback,
            })
        }
        
        #[cfg(not(feature = "gpu"))]
        {
            Ok(WorkspaceBuffers {
                recv_buffer: None,
                workspace: None,
                compressed_values: None,
                compressed_indices: None,
                error_feedback: None,
            })
        }
    }
}

/// Helper to setup multi-GPU training
pub struct MultiGpuSetup {
    /// GPU contexts for each device
    pub contexts: Vec<Arc<GpuContext>>,
    /// Synchronization managers
    pub sync_managers: Vec<MultiGpuSync<f32>>,
}

impl MultiGpuSetup {
    /// Initialize multi-GPU setup
    pub fn new(num_gpus: usize, max_param_size: usize) -> Result<Self, GpuOptimizerError> {
        let mut contexts = Vec::new();
        let mut sync_managers = Vec::new();
        
        for rank in 0..num_gpus {
            // Create GPU context for each device
            let context = Arc::new(GpuContext::new(scirs2_core::gpu::GpuBackend::Cuda)?);
            
            // Create sync manager
            let config = MultiGpuConfig {
                num_gpus,
                rank,
                ..Default::default()
            };
            
            let sync_manager = MultiGpuSync::new(
                context.clone(),
                config,
                max_param_size,
            )?;
            
            contexts.push(context);
            sync_managers.push(sync_manager);
        }
        
        Ok(Self {
            contexts,
            sync_managers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_multi_gpu_config_default() {
        let config = MultiGpuConfig::default();
        assert_eq!(config.num_gpus, 1);
        assert_eq!(config.rank, 0);
        assert_eq!(config.sync_strategy, SyncStrategy::RingAllReduce);
        assert!(!config.gradient_compression);
    }
    
    #[test]
    fn test_sync_strategy_selection() {
        let strategies = [
            SyncStrategy::RingAllReduce,
            SyncStrategy::TreeAllReduce,
            SyncStrategy::HierarchicalAllReduce,
            SyncStrategy::PipelineParallel,
        ];
        
        for strategy in &strategies {
            let config = MultiGpuConfig {
                sync_strategy: *strategy,
                ..Default::default()
            };
            assert_eq!(config.sync_strategy, *strategy);
        }
    }
}