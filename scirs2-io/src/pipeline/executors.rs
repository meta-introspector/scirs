//! Pipeline execution strategies and executors

use super::*;
use crate::error::Result;
use crate::streaming::StreamingConfig;
#[cfg(feature = "async")]
use futures::stream::{self, StreamExt};
use std::sync::mpsc;
use std::thread;
#[cfg(feature = "async")]
use tokio::runtime::Runtime;

/// Trait for pipeline executors
pub trait PipelineExecutor<I, O> {
    /// Execute the pipeline with the given input
    fn execute(&self, pipeline: &Pipeline<I, O>, input: I) -> Result<O>;
    
    /// Get executor name
    fn name(&self) -> &str;
}

/// Sequential executor - executes stages one after another
pub struct SequentialExecutor;

impl<I, O> PipelineExecutor<I, O> for SequentialExecutor
where
    I: 'static + Send + Sync,
    O: 'static + Send + Sync,
{
    fn execute(&self, pipeline: &Pipeline<I, O>, input: I) -> Result<O> {
        pipeline.execute(input)
    }
    
    fn name(&self) -> &str {
        "sequential"
    }
}

/// Streaming executor - processes data in chunks
pub struct StreamingExecutor {
    pub chunk_size: usize,
}

impl StreamingExecutor {
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }
}

impl<I, O> PipelineExecutor<Vec<I>, Vec<O>> for StreamingExecutor
where
    I: 'static + Send + Sync + Clone,
    O: 'static + Send + Sync,
{
    fn execute(&self, pipeline: &Pipeline<Vec<I>, Vec<O>>, input: Vec<I>) -> Result<Vec<O>> {
        let chunks: Vec<Vec<I>> = input
            .chunks(self.chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        let mut results = Vec::new();
        
        for chunk in chunks {
            let chunk_result = pipeline.execute(chunk)?;
            results.extend(chunk_result);
        }
        
        Ok(results)
    }
    
    fn name(&self) -> &str {
        "streaming"
    }
}

/// Async executor - executes pipeline asynchronously
#[cfg(feature = "async")]
pub struct AsyncExecutor {
    runtime: Runtime,
}

#[cfg(feature = "async")]
impl AsyncExecutor {
    pub fn new() -> Self {
        Self {
            runtime: Runtime::new().unwrap(),
        }
    }
}

#[cfg(feature = "async")]
impl<I, O> PipelineExecutor<I, O> for AsyncExecutor
where
    I: 'static + Send + Sync,
    O: 'static + Send + Sync,
{
    fn execute(&self, pipeline: &Pipeline<I, O>, input: I) -> Result<O> {
        self.runtime.block_on(async {
            // Execute pipeline in async context
            tokio::task::spawn_blocking(move || pipeline.execute(input))
                .await
                .map_err(|e| IoError::Other(format!("Async execution error: {}", e)))?
        })
    }
    
    fn name(&self) -> &str {
        "async"
    }
}

/// Cached executor - caches intermediate results
pub struct CachedExecutor {
    cache_dir: PathBuf,
}

impl CachedExecutor {
    pub fn new(cache_dir: impl AsRef<Path>) -> Self {
        Self {
            cache_dir: cache_dir.as_ref().to_path_buf(),
        }
    }
    
    fn cache_key<T>(&self, stage_name: &str, input: &T) -> String
    where
        T: std::fmt::Debug,
    {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        format!("{:?}", input).hash(&mut hasher);
        format!("{}_{:x}", stage_name, hasher.finish())
    }
}

impl<I, O> PipelineExecutor<I, O> for CachedExecutor
where
    I: 'static + Send + Sync + std::fmt::Debug,
    O: 'static + Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
{
    fn execute(&self, pipeline: &Pipeline<I, O>, input: I) -> Result<O> {
        // Check cache first
        let cache_key = self.cache_key("pipeline", &input);
        let cache_path = self.cache_dir.join(format!("{}.cache", cache_key));
        
        if cache_path.exists() {
            // Try to load from cache
            if let Ok(cached_data) = std::fs::read(&cache_path) {
                if let Ok(result) = bincode::deserialize::<O>(&cached_data) {
                    return Ok(result);
                }
            }
        }
        
        // Execute pipeline
        let result = pipeline.execute(input)?;
        
        // Save to cache
        if let Ok(serialized) = bincode::serialize(&result) {
            let _ = std::fs::create_dir_all(&self.cache_dir);
            let _ = std::fs::write(&cache_path, serialized);
        }
        
        Ok(result)
    }
    
    fn name(&self) -> &str {
        "cached"
    }
}

/// Distributed executor - distributes work across multiple workers
pub struct DistributedExecutor {
    num_workers: usize,
}

impl DistributedExecutor {
    pub fn new(num_workers: usize) -> Self {
        Self { num_workers }
    }
}

impl<I, O> PipelineExecutor<Vec<I>, Vec<O>> for DistributedExecutor
where
    I: 'static + Send + Sync,
    O: 'static + Send + Sync,
{
    fn execute(&self, pipeline: &Pipeline<Vec<I>, Vec<O>>, input: Vec<I>) -> Result<Vec<O>> {
        let chunk_size = (input.len() + self.num_workers - 1) / self.num_workers;
        let chunks: Vec<Vec<I>> = input
            .chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();
        
        // Spawn workers
        for (i, chunk) in chunks.into_iter().enumerate() {
            let tx = tx.clone();
            let pipeline_clone = Pipeline::<Vec<I>, Vec<O>>::new(); // Note: would need proper cloning
            
            let handle = thread::spawn(move || {
                let result = pipeline_clone.execute(chunk);
                tx.send((i, result)).unwrap();
            });
            
            handles.push(handle);
        }
        
        // Collect results
        let mut results = vec![None; self.num_workers];
        for _ in 0..self.num_workers {
            let (index, result) = rx.recv().unwrap();
            results[index] = Some(result?);
        }
        
        // Combine results
        let combined: Vec<O> = results
            .into_iter()
            .filter_map(|r| r)
            .flatten()
            .collect();
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        Ok(combined)
    }
    
    fn name(&self) -> &str {
        "distributed"
    }
}

/// Checkpointed executor - saves progress at intervals
pub struct CheckpointedExecutor {
    checkpoint_dir: PathBuf,
    checkpoint_interval: usize,
}

impl CheckpointedExecutor {
    pub fn new(checkpoint_dir: impl AsRef<Path>, interval: usize) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.as_ref().to_path_buf(),
            checkpoint_interval: interval,
        }
    }
}

impl<I, O> PipelineExecutor<I, O> for CheckpointedExecutor
where
    I: 'static + Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
    O: 'static + Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
{
    fn execute(&self, pipeline: &Pipeline<I, O>, input: I) -> Result<O> {
        // Create checkpoint directory
        std::fs::create_dir_all(&self.checkpoint_dir)
            .map_err(|e| IoError::Io(e))?;
        
        // Execute with checkpointing logic
        // Note: This is simplified - real implementation would checkpoint at each stage
        let result = pipeline.execute(input)?;
        
        // Save final checkpoint
        let checkpoint_path = self.checkpoint_dir.join("final.checkpoint");
        let serialized = bincode::serialize(&result)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;
        std::fs::write(&checkpoint_path, serialized)
            .map_err(|e| IoError::Io(e))?;
        
        Ok(result)
    }
    
    fn name(&self) -> &str {
        "checkpointed"
    }
}

/// Factory for creating executors
pub struct ExecutorFactory;

impl ExecutorFactory {
    /// Create a sequential executor
    pub fn sequential() -> Box<dyn PipelineExecutor<Vec<i32>, Vec<i32>>> {
        Box::new(SequentialExecutor)
    }
    
    /// Create a streaming executor
    pub fn streaming(chunk_size: usize) -> Box<dyn PipelineExecutor<Vec<i32>, Vec<i32>>> {
        Box::new(StreamingExecutor::new(chunk_size))
    }
    
    /// Create an async executor
    #[cfg(feature = "async")]
    pub fn async_executor() -> Box<dyn PipelineExecutor<Vec<i32>, Vec<i32>>> {
        Box::new(AsyncExecutor::new())
    }
    
    /// Create a cached executor
    pub fn cached(cache_dir: impl AsRef<Path>) -> Box<dyn PipelineExecutor<Vec<i32>, Vec<i32>>> {
        Box::new(CachedExecutor::new(cache_dir))
    }
    
    /// Create a distributed executor
    pub fn distributed(num_workers: usize) -> Box<dyn PipelineExecutor<Vec<i32>, Vec<i32>>> {
        Box::new(DistributedExecutor::new(num_workers))
    }
    
    /// Create a checkpointed executor
    pub fn checkpointed(checkpoint_dir: impl AsRef<Path>, interval: usize) -> Box<dyn PipelineExecutor<Vec<i32>, Vec<i32>>> {
        Box::new(CheckpointedExecutor::new(checkpoint_dir, interval))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_executor() {
        let pipeline: Pipeline<i32, i32> = Pipeline::new()
            .add_stage(function_stage("double", |x: i32| Ok(x * 2)));
        
        let executor = SequentialExecutor;
        let result = executor.execute(&pipeline, 21).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_streaming_executor() {
        let pipeline: Pipeline<Vec<i32>, Vec<i32>> = Pipeline::new()
            .add_stage(function_stage("double_all", |nums: Vec<i32>| {
                Ok(nums.into_iter().map(|x| x * 2).collect())
            }));
        
        let executor = StreamingExecutor::new(2);
        let result = executor.execute(&pipeline, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(result, vec![2, 4, 6, 8]);
    }
}