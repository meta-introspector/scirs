//! Workflow automation tools for scientific data processing
//!
//! Provides a framework for building and executing automated data processing
//! workflows with dependency management, scheduling, and monitoring capabilities.

use crate::error::{IoError, Result};
use crate::metadata::Metadata;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Workflow definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub tasks: Vec<Task>,
    pub dependencies: HashMap<String, Vec<String>>,
    pub config: WorkflowConfig,
    pub metadata: Metadata,
}

/// Workflow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    pub max_parallel_tasks: usize,
    pub retry_policy: RetryPolicy,
    pub timeout: Option<Duration>,
    pub checkpoint_dir: Option<PathBuf>,
    pub notifications: NotificationConfig,
    pub scheduling: Option<ScheduleConfig>,
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            max_parallel_tasks: 4,
            retry_policy: RetryPolicy::default(),
            timeout: None,
            checkpoint_dir: None,
            notifications: NotificationConfig::default(),
            scheduling: None,
        }
    }
}

/// Task definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub name: String,
    pub task_type: TaskType,
    pub config: serde_json::Value,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub resources: ResourceRequirements,
}

/// Task types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    /// Data ingestion from files or databases
    DataIngestion,
    /// Data transformation using pipeline
    Transform,
    /// Data validation
    Validation,
    /// Machine learning training
    MLTraining,
    /// Model inference
    MLInference,
    /// Data export
    Export,
    /// Custom script execution
    Script,
    /// Sub-workflow execution
    SubWorkflow,
    /// Conditional execution
    Conditional,
    /// Parallel execution
    Parallel,
}

/// Resource requirements for tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: Option<usize>,
    pub memory_gb: Option<f64>,
    pub gpu: Option<GpuRequirement>,
    pub disk_space_gb: Option<f64>,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: None,
            memory_gb: None,
            gpu: None,
            disk_space_gb: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRequirement {
    pub count: usize,
    pub memory_gb: Option<f64>,
    pub compute_capability: Option<String>,
}

/// Retry policy for failed tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_retries: usize,
    pub backoff_seconds: u64,
    pub exponential_backoff: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff_seconds: 60,
            exponential_backoff: true,
        }
    }
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    pub on_success: bool,
    pub on_failure: bool,
    pub on_start: bool,
    pub channels: Vec<NotificationChannel>,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            on_success: false,
            on_failure: true,
            on_start: false,
            channels: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotificationChannel {
    Email { to: Vec<String> },
    Webhook { url: String },
    File { path: PathBuf },
}

/// Schedule configuration for periodic execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleConfig {
    pub cron: Option<String>,
    pub interval: Option<Duration>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
}

/// Workflow builder
pub struct WorkflowBuilder {
    workflow: Workflow,
}

impl WorkflowBuilder {
    /// Create a new workflow builder
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            workflow: Workflow {
                id: id.into(),
                name: name.into(),
                description: None,
                tasks: Vec::new(),
                dependencies: HashMap::new(),
                config: WorkflowConfig::default(),
                metadata: Metadata::new(),
            },
        }
    }

    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.workflow.description = Some(desc.into());
        self
    }

    /// Add a task
    pub fn add_task(mut self, task: Task) -> Self {
        self.workflow.tasks.push(task);
        self
    }

    /// Add a dependency
    pub fn add_dependency(mut self, task_id: impl Into<String>, depends_on: impl Into<String>) -> Self {
        let task_id = task_id.into();
        let depends_on = depends_on.into();
        
        self.workflow.dependencies
            .entry(task_id)
            .or_insert_with(Vec::new)
            .push(depends_on);
        
        self
    }

    /// Configure workflow
    pub fn configure<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut WorkflowConfig),
    {
        f(&mut self.workflow.config);
        self
    }

    /// Build the workflow
    pub fn build(self) -> Result<Workflow> {
        // Validate workflow
        self.validate()?;
        Ok(self.workflow)
    }

    fn validate(&self) -> Result<()> {
        // Check for cycles in dependencies
        if self.has_cycles() {
            return Err(IoError::ValidationError("Workflow contains dependency cycles".to_string()));
        }
        
        // Check all task IDs are unique
        let mut ids = HashSet::new();
        for task in &self.workflow.tasks {
            if !ids.insert(&task.id) {
                return Err(IoError::ValidationError(format!("Duplicate task ID: {}", task.id)));
            }
        }
        
        // Check all dependencies reference existing tasks
        for (task_id, deps) in &self.workflow.dependencies {
            if !ids.contains(&task_id) {
                return Err(IoError::ValidationError(format!("Unknown task in dependencies: {}", task_id)));
            }
            for dep in deps {
                if !ids.contains(&dep) {
                    return Err(IoError::ValidationError(format!("Unknown dependency: {}", dep)));
                }
            }
        }
        
        Ok(())
    }

    fn has_cycles(&self) -> bool {
        // Simple cycle detection using DFS
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        
        for task in &self.workflow.tasks {
            if !visited.contains(&task.id) {
                if self.has_cycle_dfs(&task.id, &mut visited, &mut rec_stack) {
                    return true;
                }
            }
        }
        
        false
    }

    fn has_cycle_dfs(&self, node: &str, visited: &mut HashSet<String>, rec_stack: &mut HashSet<String>) -> bool {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());
        
        if let Some(deps) = self.workflow.dependencies.get(node) {
            for dep in deps {
                if !visited.contains(dep) {
                    if self.has_cycle_dfs(dep, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(dep) {
                    return true;
                }
            }
        }
        
        rec_stack.remove(node);
        false
    }
}

/// Workflow execution state
#[derive(Debug, Clone)]
pub struct WorkflowState {
    pub workflow_id: String,
    pub execution_id: String,
    pub status: WorkflowStatus,
    pub task_states: HashMap<String, TaskState>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkflowStatus {
    Pending,
    Running,
    Success,
    Failed,
    Cancelled,
    Paused,
}

#[derive(Debug, Clone)]
pub struct TaskState {
    pub status: TaskStatus,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub attempts: usize,
    pub error: Option<String>,
    pub outputs: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    Running,
    Success,
    Failed,
    Skipped,
    Retrying,
}

/// Workflow executor
pub struct WorkflowExecutor {
    config: ExecutorConfig,
    state: Arc<Mutex<HashMap<String, WorkflowState>>>,
}

#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub max_concurrent_workflows: usize,
    pub task_timeout: Duration,
    pub checkpoint_interval: Duration,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_workflows: 10,
            task_timeout: Duration::hours(1),
            checkpoint_interval: Duration::minutes(5),
        }
    }
}

impl WorkflowExecutor {
    /// Create a new workflow executor
    pub fn new(config: ExecutorConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Execute a workflow
    pub fn execute(&self, workflow: &Workflow) -> Result<String> {
        let execution_id = format!("{}-{}", workflow.id, Utc::now().timestamp());
        
        let mut state = WorkflowState {
            workflow_id: workflow.id.clone(),
            execution_id: execution_id.clone(),
            status: WorkflowStatus::Pending,
            task_states: HashMap::new(),
            start_time: None,
            end_time: None,
            error: None,
        };
        
        // Initialize task states
        for task in &workflow.tasks {
            state.task_states.insert(task.id.clone(), TaskState {
                status: TaskStatus::Pending,
                start_time: None,
                end_time: None,
                attempts: 0,
                error: None,
                outputs: HashMap::new(),
            });
        }
        
        // Store state
        self.state.lock().unwrap().insert(execution_id.clone(), state);
        
        // In a real implementation, this would start async execution
        // For now, we'll just return the execution ID
        Ok(execution_id)
    }

    /// Get workflow state
    pub fn get_state(&self, execution_id: &str) -> Option<WorkflowState> {
        self.state.lock().unwrap().get(execution_id).cloned()
    }

    /// Cancel a workflow execution
    pub fn cancel(&self, execution_id: &str) -> Result<()> {
        let mut states = self.state.lock().unwrap();
        if let Some(state) = states.get_mut(execution_id) {
            state.status = WorkflowStatus::Cancelled;
            state.end_time = Some(Utc::now());
            Ok(())
        } else {
            Err(IoError::Other(format!("Execution {} not found", execution_id)))
        }
    }
}

/// Task builders for common operations
pub mod tasks {
    use super::*;
    
    /// Create a data ingestion task
    pub fn data_ingestion(id: impl Into<String>, name: impl Into<String>) -> TaskBuilder {
        TaskBuilder::new(id, name, TaskType::DataIngestion)
    }
    
    /// Create a transformation task
    pub fn transform(id: impl Into<String>, name: impl Into<String>) -> TaskBuilder {
        TaskBuilder::new(id, name, TaskType::Transform)
    }
    
    /// Create a validation task
    pub fn validation(id: impl Into<String>, name: impl Into<String>) -> TaskBuilder {
        TaskBuilder::new(id, name, TaskType::Validation)
    }
    
    /// Create an export task
    pub fn export(id: impl Into<String>, name: impl Into<String>) -> TaskBuilder {
        TaskBuilder::new(id, name, TaskType::Export)
    }
    
    /// Task builder
    pub struct TaskBuilder {
        task: Task,
    }
    
    impl TaskBuilder {
        pub fn new(id: impl Into<String>, name: impl Into<String>, task_type: TaskType) -> Self {
            Self {
                task: Task {
                    id: id.into(),
                    name: name.into(),
                    task_type,
                    config: serde_json::json!({}),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    resources: ResourceRequirements::default(),
                },
            }
        }
        
        pub fn config(mut self, config: serde_json::Value) -> Self {
            self.task.config = config;
            self
        }
        
        pub fn input(mut self, input: impl Into<String>) -> Self {
            self.task.inputs.push(input.into());
            self
        }
        
        pub fn output(mut self, output: impl Into<String>) -> Self {
            self.task.outputs.push(output.into());
            self
        }
        
        pub fn resources(mut self, cpu: usize, memory_gb: f64) -> Self {
            self.task.resources.cpu_cores = Some(cpu);
            self.task.resources.memory_gb = Some(memory_gb);
            self
        }
        
        pub fn build(self) -> Task {
            self.task
        }
    }
}

/// Workflow templates for common patterns
pub mod templates {
    use super::*;
    
    /// Create an ETL (Extract-Transform-Load) workflow
    pub fn etl_workflow(name: impl Into<String>) -> WorkflowBuilder {
        let name = name.into();
        let id = format!("etl_{}", Utc::now().timestamp());
        
        WorkflowBuilder::new(&id, &name)
            .description("Standard ETL workflow template")
            .add_task(
                tasks::data_ingestion("extract", "Extract Data")
                    .config(serde_json::json!({
                        "source": "database",
                        "query": "SELECT * FROM raw_data"
                    }))
                    .output("raw_data")
                    .build()
            )
            .add_task(
                tasks::transform("transform", "Transform Data")
                    .input("raw_data")
                    .output("transformed_data")
                    .config(serde_json::json!({
                        "operations": ["normalize", "aggregate", "filter"]
                    }))
                    .build()
            )
            .add_task(
                tasks::validation("validate", "Validate Data")
                    .input("transformed_data")
                    .output("validated_data")
                    .build()
            )
            .add_task(
                tasks::export("load", "Load Data")
                    .input("validated_data")
                    .config(serde_json::json!({
                        "destination": "warehouse",
                        "table": "processed_data"
                    }))
                    .build()
            )
            .add_dependency("transform", "extract")
            .add_dependency("validate", "transform")
            .add_dependency("load", "validate")
    }
    
    /// Create a batch processing workflow
    pub fn batch_processing(name: impl Into<String>, batch_size: usize) -> WorkflowBuilder {
        let name = name.into();
        let id = format!("batch_{}", Utc::now().timestamp());
        
        WorkflowBuilder::new(&id, &name)
            .description("Batch processing workflow template")
            .configure(|config| {
                config.max_parallel_tasks = 8;
                config.scheduling = Some(ScheduleConfig {
                    cron: Some("0 2 * * *".to_string()), // Daily at 2 AM
                    interval: None,
                    start_time: None,
                    end_time: None,
                });
            })
    }
}

/// Workflow monitoring and metrics
pub mod monitoring {
    use super::*;
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WorkflowMetrics {
        pub total_executions: usize,
        pub successful_executions: usize,
        pub failed_executions: usize,
        pub average_duration: Duration,
        pub task_metrics: HashMap<String, TaskMetrics>,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TaskMetrics {
        pub total_runs: usize,
        pub success_rate: f64,
        pub average_duration: Duration,
        pub retry_rate: f64,
    }
    
    /// Collect metrics for a workflow
    pub fn collect_metrics(states: &[WorkflowState]) -> WorkflowMetrics {
        let total = states.len();
        let successful = states.iter().filter(|s| s.status == WorkflowStatus::Success).count();
        let failed = states.iter().filter(|s| s.status == WorkflowStatus::Failed).count();
        
        let durations: Vec<Duration> = states.iter()
            .filter_map(|s| match (s.start_time, s.end_time) {
                (Some(start), Some(end)) => Some(end - start),
                _ => None,
            })
            .collect();
        
        let avg_duration = if durations.is_empty() {
            Duration::seconds(0)
        } else {
            let total_secs: i64 = durations.iter().map(|d| d.num_seconds()).sum();
            Duration::seconds(total_secs / durations.len() as i64)
        };
        
        WorkflowMetrics {
            total_executions: total,
            successful_executions: successful,
            failed_executions: failed,
            average_duration: avg_duration,
            task_metrics: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_builder() {
        let workflow = WorkflowBuilder::new("test_wf", "Test Workflow")
            .description("A test workflow")
            .add_task(
                tasks::data_ingestion("task1", "Load Data")
                    .output("data.csv")
                    .build()
            )
            .add_task(
                tasks::transform("task2", "Process Data")
                    .input("data.csv")
                    .output("processed.csv")
                    .build()
            )
            .add_dependency("task2", "task1")
            .build()
            .unwrap();
        
        assert_eq!(workflow.tasks.len(), 2);
        assert_eq!(workflow.dependencies.get("task2").unwrap(), &vec!["task1".to_string()]);
    }

    #[test]
    fn test_cycle_detection() {
        let result = WorkflowBuilder::new("cyclic", "Cyclic Workflow")
            .add_task(tasks::transform("a", "Task A").build())
            .add_task(tasks::transform("b", "Task B").build())
            .add_task(tasks::transform("c", "Task C").build())
            .add_dependency("a", "b")
            .add_dependency("b", "c")
            .add_dependency("c", "a") // Creates cycle
            .build();
        
        assert!(result.is_err());
    }

    #[test]
    fn test_etl_template() {
        let workflow = templates::etl_workflow("My ETL Pipeline")
            .build()
            .unwrap();
        
        assert_eq!(workflow.tasks.len(), 4);
        assert!(workflow.tasks.iter().any(|t| t.id == "extract"));
        assert!(workflow.tasks.iter().any(|t| t.id == "transform"));
        assert!(workflow.tasks.iter().any(|t| t.id == "validate"));
        assert!(workflow.tasks.iter().any(|t| t.id == "load"));
    }
}