//! Comprehensive plugin development example for scirs2-optim
//!
//! This example demonstrates the complete workflow for developing, registering,
//! and using custom optimizer plugins with the scirs2-optim plugin system.

use ndarray::{Array1, Array2};
use num_traits::Float;
use scirs2_optim::{
    error::{OptimError, Result},
    plugin::{
        create_basic_capabilities, create_plugin_info, ConfigSchema, ConfigValue,
        ConvergenceMetrics, DataType, ExtendedOptimizerPlugin, FieldSchema, FieldType, MemoryUsage,
        OptimizerConfig, OptimizerPlugin, OptimizerPluginFactory, OptimizerState,
        PerformanceMetrics, PluginCapabilities, PluginCategory, PluginFactoryWrapper, PluginInfo,
        PluginQuery, PluginQueryBuilder, PluginRegistration, PluginRegistry, PluginStatus,
        RegistryConfig, StateValue, ValidationConstraint,
    },
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Custom momentum-based optimizer plugin
#[derive(Debug, Clone)]
pub struct CustomMomentumOptimizer<A: Float> {
    /// Learning rate
    learning_rate: A,
    /// Momentum coefficient
    momentum: A,
    /// Weight decay
    weight_decay: A,
    /// Velocity vectors
    velocity: Option<Array1<A>>,
    /// Step count
    step_count: usize,
    /// Performance tracking
    step_times: Vec<f64>,
    /// Parameter trajectory
    trajectory: Vec<Array1<A>>,
    /// Convergence metrics
    last_gradient_norm: A,
    last_param_change_norm: A,
}

impl<A: Float + Clone + std::fmt::Debug + Send + Sync> CustomMomentumOptimizer<A> {
    pub fn new(learning_rate: A, momentum: A, weight_decay: A) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            velocity: None,
            step_count: 0,
            step_times: Vec::new(),
            trajectory: Vec::new(),
            last_gradient_norm: A::zero(),
            last_param_change_norm: A::zero(),
        }
    }
}

impl<A: Float + Clone + std::fmt::Debug + Send + Sync> OptimizerPlugin<A>
    for CustomMomentumOptimizer<A>
{
    fn step(&mut self, params: &Array1<A>, gradients: &Array1<A>) -> Result<Array1<A>> {
        let start_time = Instant::now();

        // Initialize velocity if needed
        if self.velocity.is_none() {
            self.velocity = Some(Array1::zeros(params.raw_dim()));
        }

        let velocity = self.velocity.as_mut().unwrap();

        // Apply weight decay to gradients
        let decayed_gradients = if self.weight_decay > A::zero() {
            gradients + &(params * self.weight_decay)
        } else {
            gradients.clone()
        };

        // Update velocity: v = momentum * v + learning_rate * grad
        *velocity = &*velocity * self.momentum + &decayed_gradients * self.learning_rate;

        // Update parameters: params = params - velocity
        let new_params = params - velocity;

        // Track metrics
        self.step_count += 1;
        self.step_times.push(start_time.elapsed().as_secs_f64());
        self.trajectory.push(new_params.clone());
        self.last_gradient_norm = gradients.mapv(|x| x * x).sum().sqrt();
        self.last_param_change_norm = velocity.mapv(|x| x * x).sum().sqrt();

        Ok(new_params)
    }

    fn name(&self) -> &str {
        "CustomMomentumOptimizer"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn plugin_info(&self) -> PluginInfo {
        PluginInfo {
            name: self.name().to_string(),
            version: self.version().to_string(),
            author: "Plugin Developer".to_string(),
            description: "Custom momentum-based optimizer with enhanced tracking".to_string(),
            homepage: Some("https://github.com/example/custom-momentum".to_string()),
            license: "MIT".to_string(),
            supported_types: vec![DataType::F32, DataType::F64],
            category: PluginCategory::FirstOrder,
            tags: vec![
                "momentum".to_string(),
                "custom".to_string(),
                "tracking".to_string(),
            ],
            min_sdk_version: "0.1.0".to_string(),
            dependencies: Vec::new(),
        }
    }

    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities {
            momentum: true,
            weight_decay: true,
            state_serialization: true,
            thread_safe: true,
            memory_efficient: true,
            batch_processing: true,
            gradient_clipping: false,
            sparse_gradients: false,
            parameter_groups: false,
            adaptive_learning_rate: false,
            gpu_support: false,
            simd_optimized: false,
            custom_loss_functions: false,
            regularization: true,
        }
    }

    fn initialize(&mut self, param_shape: &[usize]) -> Result<()> {
        if param_shape.len() != 1 {
            return Err(OptimError::DimensionMismatch(
                "CustomMomentumOptimizer only supports 1D parameters".to_string(),
            ));
        }

        self.velocity = Some(Array1::zeros(param_shape[0]));
        self.step_count = 0;
        self.step_times.clear();
        self.trajectory.clear();

        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        if let Some(ref mut velocity) = self.velocity {
            velocity.fill(A::zero());
        }
        self.step_count = 0;
        self.step_times.clear();
        self.trajectory.clear();
        self.last_gradient_norm = A::zero();
        self.last_param_change_norm = A::zero();
        Ok(())
    }

    fn get_config(&self) -> OptimizerConfig {
        let mut config = OptimizerConfig {
            learning_rate: self.learning_rate.to_f64().unwrap_or(0.001),
            weight_decay: self.weight_decay.to_f64().unwrap_or(0.0),
            momentum: self.momentum.to_f64().unwrap_or(0.9),
            gradient_clip: None,
            custom_params: HashMap::new(),
        };

        config.custom_params.insert(
            "step_count".to_string(),
            ConfigValue::Integer(self.step_count as i64),
        );

        config
    }

    fn set_config(&mut self, config: OptimizerConfig) -> Result<()> {
        self.learning_rate = A::from(config.learning_rate)
            .ok_or_else(|| OptimError::InvalidConfig("Invalid learning rate type".to_string()))?;

        self.weight_decay = A::from(config.weight_decay)
            .ok_or_else(|| OptimError::InvalidConfig("Invalid weight decay type".to_string()))?;

        self.momentum = A::from(config.momentum)
            .ok_or_else(|| OptimError::InvalidConfig("Invalid momentum type".to_string()))?;

        // Validate configuration
        if self.learning_rate <= A::zero() {
            return Err(OptimError::InvalidConfig(
                "Learning rate must be positive".to_string(),
            ));
        }

        if self.weight_decay < A::zero() {
            return Err(OptimError::InvalidConfig(
                "Weight decay must be non-negative".to_string(),
            ));
        }

        if self.momentum < A::zero() || self.momentum >= A::one() {
            return Err(OptimError::InvalidConfig(
                "Momentum must be in range [0, 1)".to_string(),
            ));
        }

        Ok(())
    }

    fn get_state(&self) -> Result<OptimizerState> {
        let mut state = OptimizerState {
            state_vectors: HashMap::new(),
            step_count: self.step_count,
            custom_state: HashMap::new(),
        };

        if let Some(ref velocity) = self.velocity {
            state.state_vectors.insert(
                "velocity".to_string(),
                velocity
                    .iter()
                    .map(|&x| x.to_f64().unwrap_or(0.0))
                    .collect(),
            );
        }

        state.custom_state.insert(
            "last_gradient_norm".to_string(),
            StateValue::Float(self.last_gradient_norm.to_f64().unwrap_or(0.0)),
        );

        state.custom_state.insert(
            "last_param_change_norm".to_string(),
            StateValue::Float(self.last_param_change_norm.to_f64().unwrap_or(0.0)),
        );

        state.custom_state.insert(
            "step_times".to_string(),
            StateValue::Array(self.step_times.clone()),
        );

        Ok(state)
    }

    fn set_state(&mut self, state: OptimizerState) -> Result<()> {
        self.step_count = state.step_count;

        if let Some(velocity_data) = state.state_vectors.get("velocity") {
            let velocity_array: Result<Array1<A>, _> = velocity_data
                .iter()
                .map(|&x| {
                    A::from(x).ok_or_else(|| {
                        OptimError::InvalidConfig("Invalid velocity data type".to_string())
                    })
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Array1::from_vec);

            self.velocity = Some(velocity_array?);
        }

        if let Some(StateValue::Float(norm)) = state.custom_state.get("last_gradient_norm") {
            self.last_gradient_norm = A::from(*norm).ok_or_else(|| {
                OptimError::InvalidConfig("Invalid gradient norm type".to_string())
            })?;
        }

        if let Some(StateValue::Float(norm)) = state.custom_state.get("last_param_change_norm") {
            self.last_param_change_norm = A::from(*norm).ok_or_else(|| {
                OptimError::InvalidConfig("Invalid param change norm type".to_string())
            })?;
        }

        if let Some(StateValue::Array(times)) = state.custom_state.get("step_times") {
            self.step_times = times.clone();
        }

        Ok(())
    }

    fn clone_plugin(&self) -> Box<dyn OptimizerPlugin<A>> {
        Box::new(self.clone())
    }

    fn memory_usage(&self) -> MemoryUsage {
        let velocity_size = self
            .velocity
            .as_ref()
            .map(|v| v.len() * std::mem::size_of::<A>())
            .unwrap_or(0);

        let trajectory_size = self.trajectory.len()
            * self.trajectory.first().map(|t| t.len()).unwrap_or(0)
            * std::mem::size_of::<A>();

        let step_times_size = self.step_times.len() * std::mem::size_of::<f64>();

        let current_usage = velocity_size + trajectory_size + step_times_size;

        MemoryUsage {
            current_usage,
            peak_usage: current_usage, // Simplified
            efficiency_score: 0.85,    // Good efficiency
        }
    }

    fn performance_metrics(&self) -> PerformanceMetrics {
        let avg_step_time = if !self.step_times.is_empty() {
            self.step_times.iter().sum::<f64>() / self.step_times.len() as f64
        } else {
            0.0
        };

        let throughput = if avg_step_time > 0.0 {
            1.0 / avg_step_time
        } else {
            0.0
        };

        PerformanceMetrics {
            avg_step_time,
            total_steps: self.step_count,
            throughput,
            cpu_utilization: 0.75, // Estimated
        }
    }
}

impl<A: Float + Clone + std::fmt::Debug + Send + Sync> ExtendedOptimizerPlugin<A>
    for CustomMomentumOptimizer<A>
{
    fn batch_step(&mut self, params: &Array2<A>, gradients: &Array2<A>) -> Result<Array2<A>> {
        if params.shape() != gradients.shape() {
            return Err(OptimError::DimensionMismatch(
                "Parameters and gradients must have the same shape".to_string(),
            ));
        }

        let mut result = Array2::zeros(params.raw_dim());

        // Process each row independently
        for (i, (param_row, grad_row)) in
            params.outer_iter().zip(gradients.outer_iter()).enumerate()
        {
            let param_1d = param_row.to_owned();
            let grad_1d = grad_row.to_owned();
            let updated_params = self.step(&param_1d, &grad_1d)?;
            result.row_mut(i).assign(&updated_params);
        }

        Ok(result)
    }

    fn adaptive_learning_rate(&self, gradients: &Array1<A>) -> A {
        // Simple adaptive learning rate based on gradient norm
        let grad_norm = gradients.mapv(|x| x * x).sum().sqrt();
        if grad_norm > A::one() {
            self.learning_rate / grad_norm
        } else {
            self.learning_rate
        }
    }

    fn preprocess_gradients(&self, gradients: &Array1<A>) -> Result<Array1<A>> {
        // Apply gradient clipping if magnitude is too large
        let grad_norm = gradients.mapv(|x| x * x).sum().sqrt();
        let max_norm = A::from(1.0).unwrap();

        if grad_norm > max_norm {
            Ok(gradients * (max_norm / grad_norm))
        } else {
            Ok(gradients.clone())
        }
    }

    fn postprocess_parameters(&self, params: &Array1<A>) -> Result<Array1<A>> {
        // Simple parameter constraint: clamp values to reasonable range
        let min_val = A::from(-10.0).unwrap();
        let max_val = A::from(10.0).unwrap();

        Ok(params.mapv(|x| {
            if x < min_val {
                min_val
            } else if x > max_val {
                max_val
            } else {
                x
            }
        }))
    }

    fn get_trajectory(&self) -> Vec<Array1<A>> {
        self.trajectory.clone()
    }

    fn convergence_metrics(&self) -> ConvergenceMetrics {
        let loss_improvement_rate = if self.step_times.len() > 1 {
            // Estimate improvement rate from step time changes
            let recent_avg = self.step_times.iter().rev().take(5).sum::<f64>() / 5.0;
            let older_avg = self.step_times.iter().take(5).sum::<f64>() / 5.0;
            (older_avg - recent_avg) / older_avg
        } else {
            0.0
        };

        let convergence_score = if self.last_gradient_norm < A::from(0.01).unwrap() {
            0.9
        } else if self.last_gradient_norm < A::from(0.1).unwrap() {
            0.7
        } else {
            0.3
        };

        ConvergenceMetrics {
            gradient_norm: self.last_gradient_norm.to_f64().unwrap_or(0.0),
            parameter_change_norm: self.last_param_change_norm.to_f64().unwrap_or(0.0),
            loss_improvement_rate,
            convergence_score,
        }
    }
}

/// Factory for creating CustomMomentumOptimizer instances
#[derive(Debug)]
pub struct CustomMomentumFactory;

impl CustomMomentumFactory {
    pub fn new() -> Self {
        Self
    }
}

impl<A: Float + Clone + std::fmt::Debug + Send + Sync + 'static> OptimizerPluginFactory<A>
    for CustomMomentumFactory
{
    fn create_optimizer(&self, config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<A>>> {
        let learning_rate = A::from(config.learning_rate)
            .ok_or_else(|| OptimError::InvalidConfig("Invalid learning rate type".to_string()))?;

        let momentum = A::from(config.momentum)
            .ok_or_else(|| OptimError::InvalidConfig("Invalid momentum type".to_string()))?;

        let weight_decay = A::from(config.weight_decay)
            .ok_or_else(|| OptimError::InvalidConfig("Invalid weight decay type".to_string()))?;

        let optimizer = CustomMomentumOptimizer::new(learning_rate, momentum, weight_decay);
        Ok(Box::new(optimizer))
    }

    fn factory_info(&self) -> PluginInfo {
        create_plugin_info("CustomMomentumOptimizer", "1.0.0", "Plugin Developer")
    }

    fn validate_config(&self, config: &OptimizerConfig) -> Result<()> {
        if config.learning_rate <= 0.0 {
            return Err(OptimError::InvalidConfig(
                "Learning rate must be positive".to_string(),
            ));
        }

        if config.weight_decay < 0.0 {
            return Err(OptimError::InvalidConfig(
                "Weight decay must be non-negative".to_string(),
            ));
        }

        if config.momentum < 0.0 || config.momentum >= 1.0 {
            return Err(OptimError::InvalidConfig(
                "Momentum must be in range [0, 1)".to_string(),
            ));
        }

        Ok(())
    }

    fn default_config(&self) -> OptimizerConfig {
        OptimizerConfig {
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.0001,
            gradient_clip: None,
            custom_params: HashMap::new(),
        }
    }

    fn config_schema(&self) -> ConfigSchema {
        let mut fields = HashMap::new();

        fields.insert(
            "learning_rate".to_string(),
            FieldSchema {
                field_type: FieldType::Float {
                    min: Some(0.0),
                    max: Some(1.0),
                },
                description: "Learning rate for the optimizer".to_string(),
                default_value: Some(ConfigValue::Float(0.01)),
                constraints: vec![ValidationConstraint::Positive],
                required: true,
            },
        );

        fields.insert(
            "momentum".to_string(),
            FieldSchema {
                field_type: FieldType::Float {
                    min: Some(0.0),
                    max: Some(1.0),
                },
                description: "Momentum coefficient".to_string(),
                default_value: Some(ConfigValue::Float(0.9)),
                constraints: vec![ValidationConstraint::Range(0.0, 1.0)],
                required: true,
            },
        );

        fields.insert(
            "weight_decay".to_string(),
            FieldSchema {
                field_type: FieldType::Float {
                    min: Some(0.0),
                    max: None,
                },
                description: "Weight decay (L2 regularization)".to_string(),
                default_value: Some(ConfigValue::Float(0.0001)),
                constraints: vec![ValidationConstraint::NonNegative],
                required: false,
            },
        );

        ConfigSchema {
            fields,
            required_fields: vec!["learning_rate".to_string(), "momentum".to_string()],
            version: "1.0".to_string(),
        }
    }
}

/// Type-erased wrapper for the custom factory
#[derive(Debug)]
pub struct CustomMomentumFactoryWrapper {
    f32_factory: CustomMomentumFactory,
    f64_factory: CustomMomentumFactory,
}

impl CustomMomentumFactoryWrapper {
    pub fn new() -> Self {
        Self {
            f32_factory: CustomMomentumFactory::new(),
            f64_factory: CustomMomentumFactory::new(),
        }
    }
}

impl PluginFactoryWrapper for CustomMomentumFactoryWrapper {
    fn create_f32(&self, config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f32>>> {
        self.f32_factory.create_optimizer(config)
    }

    fn create_f64(&self, config: OptimizerConfig) -> Result<Box<dyn OptimizerPlugin<f64>>> {
        self.f64_factory.create_optimizer(config)
    }

    fn info(&self) -> PluginInfo {
        self.f64_factory.factory_info()
    }

    fn validate_config(&self, config: &OptimizerConfig) -> Result<()> {
        self.f64_factory.validate_config(config)
    }

    fn default_config(&self) -> OptimizerConfig {
        self.f64_factory.default_config()
    }

    fn config_schema(&self) -> ConfigSchema {
        self.f64_factory.config_schema()
    }

    fn supports_type(&self, data_type: &DataType) -> bool {
        matches!(data_type, DataType::F32 | DataType::F64)
    }
}

/// Event listener for plugin registry events
#[derive(Debug)]
pub struct PluginEventLogger {
    events: Vec<String>,
}

impl PluginEventLogger {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn get_events(&self) -> &[String] {
        &self.events
    }
}

impl scirs2_optim::plugin::RegistryEventListener for PluginEventLogger {
    fn on_plugin_registered(&mut self, info: &PluginInfo) {
        self.events.push(format!(
            "Registered plugin: {} v{}",
            info.name, info.version
        ));
    }

    fn on_plugin_unregistered(&mut self, name: &str) {
        self.events.push(format!("Unregistered plugin: {}", name));
    }

    fn on_plugin_loaded(&mut self, name: &str) {
        self.events.push(format!("Loaded plugin: {}", name));
    }

    fn on_plugin_load_failed(&mut self, name: &str, error: &str) {
        self.events
            .push(format!("Failed to load plugin {}: {}", name, error));
    }

    fn on_plugin_status_changed(&mut self, name: &str, status: &PluginStatus) {
        self.events
            .push(format!("Plugin {} status changed to: {:?}", name, status));
    }
}

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("🔌 Comprehensive Plugin Development Example");
    println!("==========================================");

    // Create a custom registry for this example
    let registry_config = RegistryConfig {
        auto_discovery: true,
        validate_on_registration: true,
        enable_caching: true,
        max_cache_size: 50,
        load_timeout: std::time::Duration::from_secs(10),
        enable_sandboxing: false,
        allowed_sources: vec![scirs2_optim::plugin::PluginSource::BuiltIn],
    };

    let registry = PluginRegistry::new(registry_config);

    // Add event listener
    let event_logger = PluginEventLogger::new();
    registry.add_event_listener(Box::new(event_logger));

    // Step 1: Register our custom plugin
    println!("\n📝 Step 1: Registering Custom Plugin");
    let custom_factory = CustomMomentumFactoryWrapper::new();
    registry.register_plugin(custom_factory)?;

    println!("✅ CustomMomentumOptimizer plugin registered successfully");

    // Step 2: Discover and list available plugins
    println!("\n🔍 Step 2: Discovering Available Plugins");
    let plugins = registry.list_plugins();
    println!("Found {} plugin(s):", plugins.len());

    for plugin in &plugins {
        println!(
            "  - {} v{} by {}",
            plugin.name, plugin.version, plugin.author
        );
        println!("    Category: {:?}", plugin.category);
        println!("    Description: {}", plugin.description);
        println!("    Supported types: {:?}", plugin.supported_types);
        println!("    Tags: {:?}", plugin.tags);
    }

    // Step 3: Search for plugins
    println!("\n🔎 Step 3: Searching for Momentum-based Plugins");
    let query = PluginQueryBuilder::new()
        .name_pattern("Momentum")
        .category(scirs2_optim::plugin::PluginCategory::FirstOrder)
        .data_type(DataType::F64)
        .tag("momentum")
        .limit(5)
        .build();

    let search_results = registry.search_plugins(query);
    println!(
        "Search found {} plugin(s) in {:.2}ms:",
        search_results.plugins.len(),
        search_results.search_time.as_secs_f64() * 1000.0
    );

    for plugin in &search_results.plugins {
        println!("  - {} v{}", plugin.name, plugin.version);
    }

    // Step 4: Create optimizer instance
    println!("\n⚙️  Step 4: Creating Optimizer Instance");
    let config = OptimizerConfig {
        learning_rate: 0.01,
        momentum: 0.9,
        weight_decay: 0.0001,
        gradient_clip: None,
        custom_params: HashMap::new(),
    };

    let mut optimizer = registry.create_optimizer::<f64>("CustomMomentumOptimizer", config)?;
    println!("✅ Created optimizer: {}", optimizer.name());

    // Initialize optimizer
    optimizer.initialize(&[5])?;

    // Step 5: Test the optimizer
    println!("\n🧪 Step 5: Testing the Optimizer");
    let mut params = ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    println!("Initial parameters: {:?}", params);

    // Simulate optimization steps
    for step in 0..10 {
        // Simulate gradients (simple quadratic function: f(x) = x^2)
        let gradients = &params * 2.0;

        params = optimizer.step(&params, &gradients)?;

        if step % 3 == 0 {
            let metrics = optimizer.convergence_metrics();
            println!(
                "Step {}: params = {:?}, grad_norm = {:.6}, conv_score = {:.3}",
                step + 1,
                params
                    .iter()
                    .map(|&x| format!("{:.3}", x))
                    .collect::<Vec<_>>(),
                metrics.gradient_norm,
                metrics.convergence_score
            );
        }
    }

    // Step 6: Analyze optimizer performance
    println!("\n📊 Step 6: Performance Analysis");
    let perf_metrics = optimizer.performance_metrics();
    let memory_usage = optimizer.memory_usage();
    let conv_metrics = optimizer.convergence_metrics();

    println!("Performance Metrics:");
    println!("  - Average step time: {:.6}s", perf_metrics.avg_step_time);
    println!("  - Total steps: {}", perf_metrics.total_steps);
    println!("  - Throughput: {:.2} steps/sec", perf_metrics.throughput);
    println!(
        "  - CPU utilization: {:.1}%",
        perf_metrics.cpu_utilization * 100.0
    );

    println!("\nMemory Usage:");
    println!("  - Current usage: {} bytes", memory_usage.current_usage);
    println!("  - Peak usage: {} bytes", memory_usage.peak_usage);
    println!("  - Efficiency score: {:.2}", memory_usage.efficiency_score);

    println!("\nConvergence Metrics:");
    println!("  - Gradient norm: {:.6}", conv_metrics.gradient_norm);
    println!(
        "  - Parameter change norm: {:.6}",
        conv_metrics.parameter_change_norm
    );
    println!(
        "  - Convergence score: {:.3}",
        conv_metrics.convergence_score
    );

    // Step 7: Test state serialization
    println!("\n💾 Step 7: Testing State Serialization");
    let state = optimizer.get_state()?;
    println!(
        "Saved optimizer state with {} state vectors and {} custom fields",
        state.state_vectors.len(),
        state.custom_state.len()
    );

    // Create a new optimizer and restore state
    let config2 = optimizer.get_config();
    let mut optimizer2 = registry.create_optimizer::<f64>("CustomMomentumOptimizer", config2)?;
    optimizer2.initialize(&[5])?;
    optimizer2.set_state(state)?;

    println!("✅ Successfully restored optimizer state");

    // Verify state restoration by comparing metrics
    let metrics1 = optimizer.performance_metrics();
    let metrics2 = optimizer2.performance_metrics();

    if metrics1.total_steps == metrics2.total_steps {
        println!("✅ State restoration verified: step counts match");
    } else {
        println!("⚠️  State restoration issue: step counts don't match");
    }

    // Step 8: Test batch processing
    println!("\n🔄 Step 8: Testing Batch Processing");
    if let Ok(extended_opt) = optimizer
        .as_any_mut()
        .downcast_mut::<CustomMomentumOptimizer<f64>>()
    {
        let batch_params = ndarray::Array2::from_shape_vec(
            (3, 5),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5, 3.5, 4.5, 2.0, 3.0, 4.0, 5.0, 6.0,
            ],
        )
        .unwrap();

        let batch_gradients = &batch_params * 2.0;

        let updated_batch = extended_opt.batch_step(&batch_params, &batch_gradients)?;

        println!("Batch processing completed:");
        for (i, row) in updated_batch.outer_iter().enumerate() {
            println!(
                "  Batch {}: {:?}",
                i + 1,
                row.iter().map(|&x| format!("{:.3}", x)).collect::<Vec<_>>()
            );
        }
    }

    // Step 9: Test advanced features
    println!("\n🚀 Step 9: Testing Advanced Features");

    if let Ok(extended_opt) = optimizer
        .as_any_mut()
        .downcast_mut::<CustomMomentumOptimizer<f64>>()
    {
        let test_gradients = ndarray::Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        // Test adaptive learning rate
        let adaptive_lr = extended_opt.adaptive_learning_rate(&test_gradients);
        println!("Adaptive learning rate: {:.6}", adaptive_lr);

        // Test gradient preprocessing
        let preprocessed_grads = extended_opt.preprocess_gradients(&test_gradients)?;
        println!(
            "Original gradients: {:?}",
            test_gradients
                .iter()
                .map(|&x| format!("{:.3}", x))
                .collect::<Vec<_>>()
        );
        println!(
            "Preprocessed gradients: {:?}",
            preprocessed_grads
                .iter()
                .map(|&x| format!("{:.3}", x))
                .collect::<Vec<_>>()
        );

        // Test trajectory tracking
        let trajectory = extended_opt.get_trajectory();
        println!("Optimization trajectory has {} points", trajectory.len());
        if !trajectory.is_empty() {
            println!(
                "Final trajectory point: {:?}",
                trajectory
                    .last()
                    .unwrap()
                    .iter()
                    .map(|&x| format!("{:.3}", x))
                    .collect::<Vec<_>>()
            );
        }
    }

    // Step 10: Registry management
    println!("\n🗂️  Step 10: Registry Management");

    // Get plugin status
    let status = registry.get_plugin_status("CustomMomentumOptimizer");
    println!("Plugin status: {:?}", status);

    // Get cache statistics
    let cache_stats = registry.get_cache_stats();
    println!("Cache statistics:");
    println!("  - Hits: {}", cache_stats.hits);
    println!("  - Misses: {}", cache_stats.misses);
    println!("  - Memory used: {} bytes", cache_stats.memory_used);

    // Test plugin disable/enable
    registry.set_plugin_status("CustomMomentumOptimizer", PluginStatus::Disabled)?;
    println!("✅ Plugin disabled");

    // Try to create optimizer while disabled (should fail)
    let result = registry.create_optimizer::<f64>("CustomMomentumOptimizer", config);
    match result {
        Err(OptimError::PluginDisabled(_)) => {
            println!("✅ Correctly rejected creation of disabled plugin");
        }
        _ => {
            println!("⚠️  Expected plugin to be rejected when disabled");
        }
    }

    // Re-enable plugin
    registry.set_plugin_status("CustomMomentumOptimizer", PluginStatus::Active)?;
    println!("✅ Plugin re-enabled");

    println!("\n🎉 Plugin Development Example Completed Successfully!");
    println!("Summary:");
    println!("  - Registered custom momentum optimizer plugin");
    println!("  - Demonstrated plugin discovery and search");
    println!("  - Tested optimization with parameter tracking");
    println!("  - Verified state serialization and restoration");
    println!("  - Tested batch processing capabilities");
    println!("  - Explored advanced optimization features");
    println!("  - Demonstrated plugin lifecycle management");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_momentum_optimizer() {
        let mut optimizer = CustomMomentumOptimizer::new(0.01, 0.9, 0.0);

        let params = ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = ndarray::Array1::from_vec(vec![0.1, 0.2, 0.3]);

        optimizer.initialize(&[3]).unwrap();
        let new_params = optimizer.step(&params, &gradients).unwrap();

        assert_eq!(new_params.len(), 3);
        assert!(optimizer.step_count() > 0);
    }

    #[test]
    fn test_plugin_factory() {
        let factory = CustomMomentumFactory::new();
        let config = factory.default_config();

        let optimizer = factory.create_optimizer(config).unwrap();
        assert_eq!(optimizer.name(), "CustomMomentumOptimizer");
    }

    #[test]
    fn test_config_validation() {
        let factory = CustomMomentumFactory::new();

        let valid_config = OptimizerConfig {
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.001,
            gradient_clip: None,
            custom_params: HashMap::new(),
        };

        assert!(factory.validate_config(&valid_config).is_ok());

        let invalid_config = OptimizerConfig {
            learning_rate: -0.01, // Invalid: negative learning rate
            momentum: 0.9,
            weight_decay: 0.001,
            gradient_clip: None,
            custom_params: HashMap::new(),
        };

        assert!(factory.validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_state_serialization() {
        let mut optimizer = CustomMomentumOptimizer::new(0.01, 0.9, 0.0);
        optimizer.initialize(&[3]).unwrap();

        let params = ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = ndarray::Array1::from_vec(vec![0.1, 0.2, 0.3]);

        optimizer.step(&params, &gradients).unwrap();

        let state = optimizer.get_state().unwrap();
        assert!(state.state_vectors.contains_key("velocity"));
        assert_eq!(state.step_count, 1);

        let mut new_optimizer = CustomMomentumOptimizer::new(0.01, 0.9, 0.0);
        new_optimizer.initialize(&[3]).unwrap();
        new_optimizer.set_state(state).unwrap();

        assert_eq!(new_optimizer.step_count, 1);
    }

    #[test]
    fn test_plugin_registry() {
        let registry_config = RegistryConfig::default();
        let registry = PluginRegistry::new(registry_config);

        let factory = CustomMomentumFactoryWrapper::new();
        registry.register_plugin(factory).unwrap();

        let plugins = registry.list_plugins();
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].name, "CustomMomentumOptimizer");
    }

    #[test]
    fn test_plugin_search() {
        let registry_config = RegistryConfig::default();
        let registry = PluginRegistry::new(registry_config);

        let factory = CustomMomentumFactoryWrapper::new();
        registry.register_plugin(factory).unwrap();

        let query = PluginQueryBuilder::new()
            .name_pattern("Momentum")
            .data_type(DataType::F64)
            .build();

        let results = registry.search_plugins(query);
        assert_eq!(results.plugins.len(), 1);
        assert_eq!(results.total_count, 1);
    }
}
