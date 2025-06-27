//! Data pipeline APIs for building complex data processing workflows
//!
//! Provides a flexible framework for constructing data processing pipelines with:
//! - Composable pipeline stages
//! - Error handling and recovery
//! - Progress tracking and monitoring
//! - Parallel and streaming execution
//! - Caching and checkpointing

use crate::error::{IoError, Result};
use crate::metadata::{Metadata, ProcessingHistoryEntry};
use ndarray::{ArrayBase, DataMut, Dimension};
use scirs2_core::parallel_ops::*;
use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

mod builders;
mod executors;
mod stages;
mod transforms;

pub use builders::*;
pub use executors::*;
pub use stages::*;
pub use transforms::*;

/// Pipeline data wrapper that carries data and metadata through stages
#[derive(Debug, Clone)]
pub struct PipelineData<T> {
    /// The actual data
    pub data: T,
    /// Associated metadata
    pub metadata: Metadata,
    /// Pipeline execution context
    pub context: PipelineContext,
}

impl<T> PipelineData<T> {
    /// Create new pipeline data
    pub fn new(data: T) -> Self {
        Self {
            data,
            metadata: Metadata::new(),
            context: PipelineContext::new(),
        }
    }

    /// Create with metadata
    pub fn with_metadata(data: T, metadata: Metadata) -> Self {
        Self {
            data,
            metadata,
            context: PipelineContext::new(),
        }
    }

    /// Transform the data while preserving metadata
    pub fn map<U, F>(self, f: F) -> PipelineData<U>
    where
        F: FnOnce(T) -> U,
    {
        PipelineData {
            data: f(self.data),
            metadata: self.metadata,
            context: self.context,
        }
    }

    /// Transform the data with potential failure
    pub fn try_map<U, F>(self, f: F) -> Result<PipelineData<U>>
    where
        F: FnOnce(T) -> Result<U>,
    {
        Ok(PipelineData {
            data: f(self.data)?,
            metadata: self.metadata,
            context: self.context,
        })
    }
}

/// Pipeline execution context
#[derive(Debug, Clone)]
pub struct PipelineContext {
    /// Shared state between pipeline stages
    pub state: Arc<Mutex<HashMap<String, Box<dyn Any + Send + Sync>>>>,
    /// Execution statistics
    pub stats: Arc<Mutex<PipelineStats>>,
    /// Configuration parameters
    pub config: PipelineConfig,
}

impl PipelineContext {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(PipelineStats::new())),
            config: PipelineConfig::default(),
        }
    }

    /// Store a value in the context
    pub fn set<T: Any + Send + Sync + 'static>(&self, key: &str, value: T) {
        let mut state = self.state.lock().unwrap();
        state.insert(key.to_string(), Box::new(value));
    }

    /// Retrieve a value from the context
    pub fn get<T: Any + Send + Sync + 'static>(&self, key: &str) -> Option<T>
    where
        T: Clone,
    {
        let state = self.state.lock().unwrap();
        state.get(key).and_then(|v| v.downcast_ref::<T>()).cloned()
    }
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Enable parallel execution where possible
    pub parallel: bool,
    /// Number of worker threads for parallel execution
    pub num_threads: Option<usize>,
    /// Enable progress tracking
    pub track_progress: bool,
    /// Enable caching of intermediate results
    pub enable_cache: bool,
    /// Cache directory
    pub cache_dir: Option<PathBuf>,
    /// Maximum memory usage (bytes)
    pub max_memory: Option<usize>,
    /// Enable checkpointing
    pub checkpoint: bool,
    /// Checkpoint interval
    pub checkpoint_interval: Duration,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            num_threads: None,
            track_progress: true,
            enable_cache: false,
            cache_dir: None,
            max_memory: None,
            checkpoint: false,
            checkpoint_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Pipeline execution statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Total execution time
    pub total_time: Duration,
    /// Execution time per stage
    pub stage_times: HashMap<String, Duration>,
    /// Memory usage per stage
    pub memory_usage: HashMap<String, usize>,
    /// Number of items processed
    pub items_processed: usize,
    /// Number of errors
    pub errors: usize,
}

impl PipelineStats {
    pub fn new() -> Self {
        Self {
            total_time: Duration::from_secs(0),
            stage_times: HashMap::new(),
            memory_usage: HashMap::new(),
            items_processed: 0,
            errors: 0,
        }
    }
}

/// Main pipeline structure
pub struct Pipeline<I, O> {
    /// Pipeline stages
    stages: Vec<Box<dyn PipelineStage>>,
    /// Pipeline configuration
    config: PipelineConfig,
    /// Input type marker
    _input: PhantomData<I>,
    /// Output type marker
    _output: PhantomData<O>,
}

impl<I, O> Pipeline<I, O> {
    /// Create a new pipeline
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            config: PipelineConfig::default(),
            _input: PhantomData,
            _output: PhantomData,
        }
    }

    /// Set pipeline configuration
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a stage to the pipeline
    pub fn add_stage(mut self, stage: Box<dyn PipelineStage>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Execute the pipeline
    pub fn execute(&self, input: I) -> Result<O>
    where
        I: 'static + Send + Sync,
        O: 'static + Send + Sync,
    {
        let start_time = Instant::now();
        let mut data = PipelineData::new(Box::new(input) as Box<dyn Any + Send + Sync>);
        data.context.config = self.config.clone();

        // Execute each stage
        for (i, stage) in self.stages.iter().enumerate() {
            let stage_start = Instant::now();
            
            // Update metadata with processing history
            let entry = ProcessingHistoryEntry::new(stage.name())
                .with_parameter("stage_index", i as i64)
                .with_parameter("stage_type", stage.stage_type());
            data.metadata.add_processing_history(entry);

            // Execute stage
            data = stage.execute(data)?;

            // Update statistics
            let mut stats = data.context.stats.lock().unwrap();
            stats.stage_times.insert(stage.name(), stage_start.elapsed());
            stats.items_processed += 1;
        }

        // Update total execution time
        {
            let mut stats = data.context.stats.lock().unwrap();
            stats.total_time = start_time.elapsed();
        }

        // Extract output
        data.data
            .downcast::<O>()
            .map(|boxed| *boxed)
            .map_err(|_| IoError::Other("Pipeline output type mismatch".to_string()))
    }

    /// Execute the pipeline with progress tracking
    pub fn execute_with_progress<F>(&self, input: I, progress_callback: F) -> Result<O>
    where
        I: 'static + Send + Sync,
        O: 'static + Send + Sync,
        F: Fn(usize, usize, &str),
    {
        let start_time = Instant::now();
        let mut data = PipelineData::new(Box::new(input) as Box<dyn Any + Send + Sync>);
        data.context.config = self.config.clone();
        
        let total_stages = self.stages.len();

        // Execute each stage with progress
        for (i, stage) in self.stages.iter().enumerate() {
            progress_callback(i + 1, total_stages, stage.name());
            
            let stage_start = Instant::now();
            
            // Update metadata
            let entry = ProcessingHistoryEntry::new(stage.name())
                .with_parameter("stage_index", i as i64)
                .with_parameter("stage_type", stage.stage_type());
            data.metadata.add_processing_history(entry);

            // Execute stage
            data = stage.execute(data)?;

            // Update statistics
            let mut stats = data.context.stats.lock().unwrap();
            stats.stage_times.insert(stage.name(), stage_start.elapsed());
            stats.items_processed += 1;
        }

        // Update total execution time
        {
            let mut stats = data.context.stats.lock().unwrap();
            stats.total_time = start_time.elapsed();
        }

        // Extract output
        data.data
            .downcast::<O>()
            .map(|boxed| *boxed)
            .map_err(|_| IoError::Other("Pipeline output type mismatch".to_string()))
    }

    /// Get pipeline statistics after execution
    pub fn get_stats(&self, context: &PipelineContext) -> PipelineStats {
        context.stats.lock().unwrap().clone()
    }
}

/// Trait for pipeline stages
pub trait PipelineStage: Send + Sync {
    /// Execute the stage
    fn execute(&self, input: PipelineData<Box<dyn Any + Send + Sync>>) -> Result<PipelineData<Box<dyn Any + Send + Sync>>>;
    
    /// Get stage name
    fn name(&self) -> String;
    
    /// Get stage type
    fn stage_type(&self) -> String {
        "generic".to_string()
    }
    
    /// Check if stage can handle the input type
    fn can_handle(&self, input_type: &str) -> bool {
        true
    }
}

/// Result type for pipeline operations
pub type PipelineResult<T> = std::result::Result<T, PipelineError>;

/// Pipeline-specific error type
#[derive(Debug)]
pub enum PipelineError {
    /// Stage execution error
    StageError { stage: String, error: String },
    /// Type mismatch error
    TypeMismatch { expected: String, actual: String },
    /// Configuration error
    ConfigError(String),
    /// IO error
    IoError(IoError),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StageError { stage, error } => write!(f, "Stage '{}' error: {}", stage, error),
            Self::TypeMismatch { expected, actual } => {
                write!(f, "Type mismatch: expected {}, got {}", expected, actual)
            }
            Self::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            Self::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<IoError> for PipelineError {
    fn from(error: IoError) -> Self {
        PipelineError::IoError(error)
    }
}

/// Create a simple function-based pipeline stage
pub fn function_stage<F, I, O>(name: &str, f: F) -> Box<dyn PipelineStage>
where
    F: Fn(I) -> Result<O> + Send + Sync + 'static,
    I: 'static + Send + Sync,
    O: 'static + Send + Sync,
{
    Box::new(FunctionStage {
        name: name.to_string(),
        function: Box::new(move |input: Box<dyn Any + Send + Sync>| {
            let typed_input = input.downcast::<I>()
                .map_err(|_| IoError::Other("Type mismatch in function stage".to_string()))?;
            let output = f(*typed_input)?;
            Ok(Box::new(output) as Box<dyn Any + Send + Sync>)
        }),
    })
}

struct FunctionStage {
    name: String,
    function: Box<dyn Fn(Box<dyn Any + Send + Sync>) -> Result<Box<dyn Any + Send + Sync>> + Send + Sync>,
}

impl PipelineStage for FunctionStage {
    fn execute(&self, mut input: PipelineData<Box<dyn Any + Send + Sync>>) -> Result<PipelineData<Box<dyn Any + Send + Sync>>> {
        input.data = (self.function)(input.data)?;
        Ok(input)
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn stage_type(&self) -> String {
        "function".to_string()
    }
}