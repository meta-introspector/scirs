//! Cloud deployment utilities for distributed time series processing
//!
//! This module provides utilities for deploying time series analysis workloads
//! across major cloud platforms (AWS, GCP, Azure) with automatic scaling,
//! fault tolerance, and cost optimization.

use crate::error::{Result, TimeSeriesError};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Cloud deployment specific errors
#[derive(Error, Debug)]
pub enum CloudError {
    #[error("Authentication failed: {0}")]
    Authentication(String),
    #[error("Resource allocation failed: {0}")]
    ResourceAllocation(String),
    #[error("Network configuration error: {0}")]
    Network(String),
    #[error("Storage error: {0}")]
    Storage(String),
    #[error("Scaling operation failed: {0}")]
    Scaling(String),
    #[error("Monitoring setup failed: {0}")]
    Monitoring(String),
}

/// Supported cloud platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CloudPlatform {
    AWS,
    GCP,
    Azure,
}

/// Cloud resource configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CloudResourceConfig {
    pub platform: CloudPlatform,
    pub region: String,
    pub instance_type: String,
    pub min_instances: usize,
    pub max_instances: usize,
    pub storage_type: String,
    pub storage_size_gb: usize,
    pub auto_scaling_enabled: bool,
    pub cost_optimization_enabled: bool,
}

impl Default for CloudResourceConfig {
    fn default() -> Self {
        CloudResourceConfig {
            platform: CloudPlatform::AWS,
            region: "us-west-2".to_string(),
            instance_type: "c5.large".to_string(),
            min_instances: 1,
            max_instances: 10,
            storage_type: "gp3".to_string(),
            storage_size_gb: 100,
            auto_scaling_enabled: true,
            cost_optimization_enabled: true,
        }
    }
}

/// Deployment environment configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeploymentConfig {
    pub environment: String,
    pub resources: CloudResourceConfig,
    pub network_config: NetworkConfig,
    pub security_config: SecurityConfig,
    pub monitoring_config: MonitoringConfig,
    pub backup_config: BackupConfig,
}

/// Network configuration for cloud deployment
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NetworkConfig {
    pub vpc_cidr: String,
    pub subnet_cidrs: Vec<String>,
    pub load_balancer_enabled: bool,
    pub ssl_enabled: bool,
    pub firewall_rules: Vec<FirewallRule>,
}

/// Security configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SecurityConfig {
    pub encryption_at_rest: bool,
    pub encryption_in_transit: bool,
    pub access_control_enabled: bool,
    pub audit_logging_enabled: bool,
    pub key_management_service: String,
}

/// Monitoring and observability configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MonitoringConfig {
    pub metrics_enabled: bool,
    pub logging_enabled: bool,
    pub alerting_enabled: bool,
    pub dashboard_enabled: bool,
    pub retention_days: usize,
}

/// Backup and disaster recovery configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BackupConfig {
    pub backup_enabled: bool,
    pub backup_frequency: String,
    pub retention_policy: String,
    pub cross_region_replication: bool,
    pub point_in_time_recovery: bool,
}

/// Firewall rule definition
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FirewallRule {
    pub direction: String,
    pub protocol: String,
    pub port_range: String,
    pub source_cidrs: Vec<String>,
    pub action: String,
}

/// Cloud deployment orchestrator
#[derive(Debug)]
pub struct CloudDeploymentOrchestrator {
    config: DeploymentConfig,
    deployment_state: DeploymentState,
    cost_tracker: CostTracker,
    health_monitor: HealthMonitor,
}

/// Current deployment state
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeploymentState {
    pub status: DeploymentStatus,
    pub active_instances: Vec<InstanceInfo>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub last_scaling_event: Option<Instant>,
    pub total_processed_jobs: usize,
    pub error_count: usize,
}

/// Deployment status enumeration
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DeploymentStatus {
    Initializing,
    Deploying,
    Running,
    Scaling,
    Stopping,
    Stopped,
    Error(String),
}

/// Instance information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InstanceInfo {
    pub instance_id: String,
    pub instance_type: String,
    pub status: String,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub network_throughput: f64,
    #[serde(skip, default = "Instant::now")]
    pub start_time: Instant,
    pub cost_per_hour: f64,
}

/// Cost tracking and optimization
#[derive(Debug)]
pub struct CostTracker {
    pub total_cost: f64,
    pub cost_by_service: HashMap<String, f64>,
    pub cost_optimization_suggestions: Vec<String>,
    pub budget_limit: Option<f64>,
    pub cost_alerts_enabled: bool,
}

/// Health monitoring and alerting
#[derive(Debug)]
pub struct HealthMonitor {
    pub metrics: HashMap<String, f64>,
    pub alerts: Vec<Alert>,
    pub health_checks: Vec<HealthCheck>,
    pub sla_targets: HashMap<String, f64>,
}

/// Alert definition
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Alert {
    pub alert_id: String,
    pub severity: AlertSeverity,
    pub message: String,
    #[cfg_attr(feature = "serde", serde(skip, default = "Instant::now"))]
    pub timestamp: Instant,
    pub resolved: bool,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Health check configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HealthCheck {
    pub check_id: String,
    pub endpoint: String,
    pub interval: Duration,
    pub timeout: Duration,
    pub healthy_threshold: usize,
    pub unhealthy_threshold: usize,
}

/// Time series processing job for cloud execution
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CloudTimeSeriesJob {
    pub job_id: String,
    pub job_type: TimeSeriesJobType,
    pub input_data: Vec<f64>,
    pub parameters: HashMap<String, serde_json::Value>,
    pub priority: JobPriority,
    pub estimated_duration: Duration,
    pub resource_requirements: ResourceRequirements,
}

/// Type of time series analysis job
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TimeSeriesJobType {
    Forecasting,
    AnomalyDetection,
    Decomposition,
    FeatureExtraction,
    Clustering,
    ChangePointDetection,
    NeuralTraining,
}

/// Job priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum JobPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Resource requirements for a job
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResourceRequirements {
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub gpu_required: bool,
    pub storage_gb: f64,
    pub network_bandwidth_mbps: f64,
}

impl CloudDeploymentOrchestrator {
    /// Create a new cloud deployment orchestrator
    pub fn new(config: DeploymentConfig) -> Self {
        let deployment_state = DeploymentState {
            status: DeploymentStatus::Initializing,
            active_instances: Vec::new(),
            last_scaling_event: None,
            total_processed_jobs: 0,
            error_count: 0,
        };

        let cost_tracker = CostTracker {
            total_cost: 0.0,
            cost_by_service: HashMap::new(),
            cost_optimization_suggestions: Vec::new(),
            budget_limit: None,
            cost_alerts_enabled: false,
        };

        let health_monitor = HealthMonitor {
            metrics: HashMap::new(),
            alerts: Vec::new(),
            health_checks: Vec::new(),
            sla_targets: HashMap::new(),
        };

        CloudDeploymentOrchestrator {
            config,
            deployment_state,
            cost_tracker,
            health_monitor,
        }
    }

    /// Deploy the time series analysis infrastructure
    pub fn deploy(&mut self) -> Result<()> {
        self.deployment_state.status = DeploymentStatus::Deploying;

        // Deploy based on platform
        match self.config.resources.platform {
            CloudPlatform::AWS => self.deploy_aws()?,
            CloudPlatform::GCP => self.deploy_gcp()?,
            CloudPlatform::Azure => self.deploy_azure()?,
        }

        self.deployment_state.status = DeploymentStatus::Running;
        Ok(())
    }

    /// Deploy on AWS platform
    fn deploy_aws(&mut self) -> Result<()> {
        println!(
            "🚀 Deploying on AWS in region: {}",
            self.config.resources.region
        );

        // Simulate AWS deployment steps
        self.create_vpc()?;
        self.create_security_groups()?;
        self.launch_instances()?;
        self.setup_load_balancer()?;
        self.configure_auto_scaling()?;
        self.setup_monitoring()?;
        self.configure_storage()?;

        println!("✅ AWS deployment completed successfully");
        Ok(())
    }

    /// Deploy on GCP platform
    fn deploy_gcp(&mut self) -> Result<()> {
        println!(
            "🚀 Deploying on GCP in region: {}",
            self.config.resources.region
        );

        // Simulate GCP deployment steps
        self.create_vpc()?;
        self.create_firewall_rules()?;
        self.launch_instances()?;
        self.setup_load_balancer()?;
        self.configure_auto_scaling()?;
        self.setup_monitoring()?;
        self.configure_storage()?;

        println!("✅ GCP deployment completed successfully");
        Ok(())
    }

    /// Deploy on Azure platform
    fn deploy_azure(&mut self) -> Result<()> {
        println!(
            "🚀 Deploying on Azure in region: {}",
            self.config.resources.region
        );

        // Simulate Azure deployment steps
        self.create_resource_group()?;
        self.create_virtual_network()?;
        self.create_network_security_groups()?;
        self.launch_instances()?;
        self.setup_load_balancer()?;
        self.configure_auto_scaling()?;
        self.setup_monitoring()?;
        self.configure_storage()?;

        println!("✅ Azure deployment completed successfully");
        Ok(())
    }

    /// Create VPC/Virtual Network
    fn create_vpc(&self) -> Result<()> {
        println!(
            "🌐 Creating VPC with CIDR: {}",
            self.config.network_config.vpc_cidr
        );
        // Simulate VPC creation
        std::thread::sleep(Duration::from_millis(100));
        Ok(())
    }

    /// Create resource group (Azure specific)
    fn create_resource_group(&self) -> Result<()> {
        println!("📦 Creating Azure resource group");
        std::thread::sleep(Duration::from_millis(50));
        Ok(())
    }

    /// Create virtual network (Azure specific)
    fn create_virtual_network(&self) -> Result<()> {
        println!("🌐 Creating Azure virtual network");
        std::thread::sleep(Duration::from_millis(100));
        Ok(())
    }

    /// Create security groups
    fn create_security_groups(&self) -> Result<()> {
        println!("🔒 Creating security groups");
        for rule in &self.config.network_config.firewall_rules {
            println!(
                "  Adding rule: {} {} {}",
                rule.direction, rule.protocol, rule.port_range
            );
        }
        std::thread::sleep(Duration::from_millis(200));
        Ok(())
    }

    /// Create firewall rules (GCP specific)
    fn create_firewall_rules(&self) -> Result<()> {
        println!("🔥 Creating GCP firewall rules");
        std::thread::sleep(Duration::from_millis(150));
        Ok(())
    }

    /// Create network security groups (Azure specific)
    fn create_network_security_groups(&self) -> Result<()> {
        println!("🛡️ Creating Azure network security groups");
        std::thread::sleep(Duration::from_millis(150));
        Ok(())
    }

    /// Launch compute instances
    fn launch_instances(&mut self) -> Result<()> {
        println!(
            "🖥️ Launching {} instances of type {}",
            self.config.resources.min_instances, self.config.resources.instance_type
        );

        for i in 0..self.config.resources.min_instances {
            let instance = InstanceInfo {
                instance_id: format!("ts-instance-{:03}", i + 1),
                instance_type: self.config.resources.instance_type.clone(),
                status: "running".to_string(),
                cpu_utilization: 10.0 + (i as f64) * 5.0,
                memory_utilization: 15.0 + (i as f64) * 3.0,
                network_throughput: 100.0,
                start_time: Instant::now(),
                cost_per_hour: self.get_instance_cost_per_hour(),
            };
            self.deployment_state.active_instances.push(instance);
        }

        std::thread::sleep(Duration::from_millis(500));
        Ok(())
    }

    /// Setup load balancer
    fn setup_load_balancer(&self) -> Result<()> {
        if self.config.network_config.load_balancer_enabled {
            println!("⚖️ Setting up load balancer");
            std::thread::sleep(Duration::from_millis(200));
        }
        Ok(())
    }

    /// Configure auto scaling
    fn configure_auto_scaling(&self) -> Result<()> {
        if self.config.resources.auto_scaling_enabled {
            println!(
                "📈 Configuring auto scaling (min: {}, max: {})",
                self.config.resources.min_instances, self.config.resources.max_instances
            );
            std::thread::sleep(Duration::from_millis(150));
        }
        Ok(())
    }

    /// Setup monitoring and alerting
    fn setup_monitoring(&mut self) -> Result<()> {
        if self.config.monitoring_config.metrics_enabled {
            println!("📊 Setting up monitoring and alerting");

            // Add default health checks
            self.health_monitor.health_checks.push(HealthCheck {
                check_id: "cpu-utilization".to_string(),
                endpoint: "/health/cpu".to_string(),
                interval: Duration::from_secs(30),
                timeout: Duration::from_secs(5),
                healthy_threshold: 2,
                unhealthy_threshold: 3,
            });

            // Set default SLA targets
            self.health_monitor
                .sla_targets
                .insert("availability".to_string(), 99.9);
            self.health_monitor
                .sla_targets
                .insert("response_time_ms".to_string(), 500.0);

            std::thread::sleep(Duration::from_millis(100));
        }
        Ok(())
    }

    /// Configure storage
    fn configure_storage(&self) -> Result<()> {
        println!(
            "💾 Configuring {} storage ({} GB)",
            self.config.resources.storage_type, self.config.resources.storage_size_gb
        );
        std::thread::sleep(Duration::from_millis(100));
        Ok(())
    }

    /// Submit a time series analysis job
    pub fn submit_job(&mut self, job: CloudTimeSeriesJob) -> Result<String> {
        if self.deployment_state.status != DeploymentStatus::Running {
            return Err(TimeSeriesError::InvalidOperation(
                "Deployment not in running state".to_string(),
            ));
        }

        println!(
            "📤 Submitting job: {} (type: {:?}, priority: {:?})",
            job.job_id, job.job_type, job.priority
        );

        // Find best instance for the job
        let instance = self.select_best_instance(&job)?;
        println!("🎯 Assigned to instance: {}", instance.instance_id);

        // Execute the job
        self.execute_job(&job, instance)?;

        self.deployment_state.total_processed_jobs += 1;
        Ok(job.job_id)
    }

    /// Select the best instance for a job based on resource requirements
    fn select_best_instance(&self, _job: &CloudTimeSeriesJob) -> Result<&InstanceInfo> {
        // Simple selection based on lowest CPU utilization
        self.deployment_state
            .active_instances
            .iter()
            .min_by(|a, b| a.cpu_utilization.partial_cmp(&b.cpu_utilization).unwrap())
            .ok_or_else(|| TimeSeriesError::InvalidOperation("No active instances".to_string()))
    }

    /// Execute a time series job
    fn execute_job(&self, job: &CloudTimeSeriesJob, instance: &InstanceInfo) -> Result<()> {
        match job.job_type {
            TimeSeriesJobType::Forecasting => self.execute_forecasting_job(job, instance),
            TimeSeriesJobType::AnomalyDetection => {
                self.execute_anomaly_detection_job(job, instance)
            }
            TimeSeriesJobType::Decomposition => self.execute_decomposition_job(job, instance),
            TimeSeriesJobType::FeatureExtraction => {
                self.execute_feature_extraction_job(job, instance)
            }
            TimeSeriesJobType::Clustering => self.execute_clustering_job(job, instance),
            TimeSeriesJobType::ChangePointDetection => self.execute_changepoint_job(job, instance),
            TimeSeriesJobType::NeuralTraining => self.execute_neural_training_job(job, instance),
        }
    }

    /// Execute forecasting job
    fn execute_forecasting_job(
        &self,
        _job: &CloudTimeSeriesJob,
        instance: &InstanceInfo,
    ) -> Result<()> {
        println!("🔮 Executing forecasting job on {}", instance.instance_id);
        // Simulate job execution
        std::thread::sleep(Duration::from_millis(200));
        Ok(())
    }

    /// Execute anomaly detection job
    fn execute_anomaly_detection_job(
        &self,
        _job: &CloudTimeSeriesJob,
        instance: &InstanceInfo,
    ) -> Result<()> {
        println!(
            "🕵️ Executing anomaly detection job on {}",
            instance.instance_id
        );
        std::thread::sleep(Duration::from_millis(150));
        Ok(())
    }

    /// Execute decomposition job
    fn execute_decomposition_job(
        &self,
        _job: &CloudTimeSeriesJob,
        instance: &InstanceInfo,
    ) -> Result<()> {
        println!("🔍 Executing decomposition job on {}", instance.instance_id);
        std::thread::sleep(Duration::from_millis(100));
        Ok(())
    }

    /// Execute feature extraction job
    fn execute_feature_extraction_job(
        &self,
        _job: &CloudTimeSeriesJob,
        instance: &InstanceInfo,
    ) -> Result<()> {
        println!(
            "⚙️ Executing feature extraction job on {}",
            instance.instance_id
        );
        std::thread::sleep(Duration::from_millis(180));
        Ok(())
    }

    /// Execute clustering job
    fn execute_clustering_job(
        &self,
        _job: &CloudTimeSeriesJob,
        instance: &InstanceInfo,
    ) -> Result<()> {
        println!("🎯 Executing clustering job on {}", instance.instance_id);
        std::thread::sleep(Duration::from_millis(250));
        Ok(())
    }

    /// Execute change point detection job
    fn execute_changepoint_job(
        &self,
        _job: &CloudTimeSeriesJob,
        instance: &InstanceInfo,
    ) -> Result<()> {
        println!(
            "📊 Executing change point detection job on {}",
            instance.instance_id
        );
        std::thread::sleep(Duration::from_millis(120));
        Ok(())
    }

    /// Execute neural training job
    fn execute_neural_training_job(
        &self,
        _job: &CloudTimeSeriesJob,
        instance: &InstanceInfo,
    ) -> Result<()> {
        println!(
            "🧠 Executing neural training job on {}",
            instance.instance_id
        );
        std::thread::sleep(Duration::from_millis(500));
        Ok(())
    }

    /// Scale the deployment based on current load
    pub fn auto_scale(&mut self) -> Result<()> {
        if !self.config.resources.auto_scaling_enabled {
            return Ok(());
        }

        let avg_cpu = self.get_average_cpu_utilization();
        let current_instances = self.deployment_state.active_instances.len();

        println!("📊 Auto-scaling check: {current_instances} instances, {avg_cpu:.1}% avg CPU");

        // Scale up if CPU utilization is high
        if avg_cpu > 80.0 && current_instances < self.config.resources.max_instances {
            println!("📈 Scaling up: adding 1 instance");
            self.add_instance()?;
            self.deployment_state.last_scaling_event = Some(Instant::now());
        }
        // Scale down if CPU utilization is low
        else if avg_cpu < 20.0 && current_instances > self.config.resources.min_instances {
            println!("📉 Scaling down: removing 1 instance");
            self.remove_instance()?;
            self.deployment_state.last_scaling_event = Some(Instant::now());
        }

        Ok(())
    }

    /// Add a new instance to the deployment
    fn add_instance(&mut self) -> Result<()> {
        let instance_id = format!(
            "ts-instance-{:03}",
            self.deployment_state.active_instances.len() + 1
        );
        let instance = InstanceInfo {
            instance_id: instance_id.clone(),
            instance_type: self.config.resources.instance_type.clone(),
            status: "running".to_string(),
            cpu_utilization: 10.0,
            memory_utilization: 15.0,
            network_throughput: 100.0,
            start_time: Instant::now(),
            cost_per_hour: self.get_instance_cost_per_hour(),
        };

        self.deployment_state.active_instances.push(instance);
        println!("✅ Added instance: {instance_id}");
        Ok(())
    }

    /// Remove an instance from the deployment
    fn remove_instance(&mut self) -> Result<()> {
        if let Some(instance) = self.deployment_state.active_instances.pop() {
            println!("✅ Removed instance: {}", instance.instance_id);
        }
        Ok(())
    }

    /// Get average CPU utilization across all instances
    fn get_average_cpu_utilization(&self) -> f64 {
        if self.deployment_state.active_instances.is_empty() {
            return 0.0;
        }

        let total: f64 = self
            .deployment_state
            .active_instances
            .iter()
            .map(|i| i.cpu_utilization)
            .sum();

        total / self.deployment_state.active_instances.len() as f64
    }

    /// Get instance cost per hour based on type and platform
    fn get_instance_cost_per_hour(&self) -> f64 {
        match self.config.resources.platform {
            CloudPlatform::AWS => match self.config.resources.instance_type.as_str() {
                "t3.micro" => 0.0104,
                "t3.small" => 0.0208,
                "c5.large" => 0.085,
                "c5.xlarge" => 0.17,
                _ => 0.1,
            },
            CloudPlatform::GCP => match self.config.resources.instance_type.as_str() {
                "e2-micro" => 0.006,
                "e2-small" => 0.012,
                "n1-standard-1" => 0.0475,
                "n1-standard-2" => 0.095,
                _ => 0.08,
            },
            CloudPlatform::Azure => match self.config.resources.instance_type.as_str() {
                "B1s" => 0.0052,
                "B2s" => 0.0208,
                "D2s_v3" => 0.096,
                "D4s_v3" => 0.192,
                _ => 0.09,
            },
        }
    }

    /// Update cost tracking
    pub fn update_costs(&mut self) {
        let hourly_cost: f64 = self
            .deployment_state
            .active_instances
            .iter()
            .map(|i| i.cost_per_hour)
            .sum();

        self.cost_tracker.total_cost += hourly_cost / 3600.0; // Convert to per-second cost

        // Update cost by service
        self.cost_tracker
            .cost_by_service
            .insert("compute".to_string(), hourly_cost);

        // Generate cost optimization suggestions
        if hourly_cost > 1.0 {
            self.cost_tracker.cost_optimization_suggestions.clear();
            self.cost_tracker
                .cost_optimization_suggestions
                .push("Consider using spot instances for non-critical workloads".to_string());
            self.cost_tracker
                .cost_optimization_suggestions
                .push("Review instance types for better price-performance ratio".to_string());
        }
    }

    /// Get deployment status
    pub fn get_status(&self) -> &DeploymentStatus {
        &self.deployment_state.status
    }

    /// Get deployment metrics
    pub fn get_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        metrics.insert(
            "active_instances".to_string(),
            self.deployment_state.active_instances.len() as f64,
        );
        metrics.insert(
            "avg_cpu_utilization".to_string(),
            self.get_average_cpu_utilization(),
        );
        metrics.insert(
            "total_jobs_processed".to_string(),
            self.deployment_state.total_processed_jobs as f64,
        );
        metrics.insert(
            "error_count".to_string(),
            self.deployment_state.error_count as f64,
        );
        metrics.insert("total_cost".to_string(), self.cost_tracker.total_cost);

        metrics
    }

    /// Terminate the deployment
    pub fn terminate(&mut self) -> Result<()> {
        println!("🛑 Terminating deployment...");

        self.deployment_state.status = DeploymentStatus::Stopping;

        // Stop all instances
        for instance in &self.deployment_state.active_instances {
            println!("🔌 Stopping instance: {}", instance.instance_id);
        }

        self.deployment_state.active_instances.clear();
        self.deployment_state.status = DeploymentStatus::Stopped;

        println!("✅ Deployment terminated successfully");
        println!("💰 Total cost: ${:.4}", self.cost_tracker.total_cost);
        println!(
            "📊 Total jobs processed: {}",
            self.deployment_state.total_processed_jobs
        );

        Ok(())
    }
}

/// Default deployment configurations for different scenarios
impl DeploymentConfig {
    /// Development environment configuration
    pub fn development() -> Self {
        DeploymentConfig {
            environment: "development".to_string(),
            resources: CloudResourceConfig {
                min_instances: 1,
                max_instances: 2,
                instance_type: "t3.small".to_string(),
                ..Default::default()
            },
            network_config: NetworkConfig {
                vpc_cidr: "10.0.0.0/16".to_string(),
                subnet_cidrs: vec!["10.0.1.0/24".to_string()],
                load_balancer_enabled: false,
                ssl_enabled: false,
                firewall_rules: vec![FirewallRule {
                    direction: "inbound".to_string(),
                    protocol: "tcp".to_string(),
                    port_range: "22".to_string(),
                    source_cidrs: vec!["0.0.0.0/0".to_string()],
                    action: "allow".to_string(),
                }],
            },
            security_config: SecurityConfig {
                encryption_at_rest: false,
                encryption_in_transit: false,
                access_control_enabled: true,
                audit_logging_enabled: false,
                key_management_service: "none".to_string(),
            },
            monitoring_config: MonitoringConfig {
                metrics_enabled: true,
                logging_enabled: true,
                alerting_enabled: false,
                dashboard_enabled: false,
                retention_days: 7,
            },
            backup_config: BackupConfig {
                backup_enabled: false,
                backup_frequency: "daily".to_string(),
                retention_policy: "7 days".to_string(),
                cross_region_replication: false,
                point_in_time_recovery: false,
            },
        }
    }

    /// Production environment configuration
    pub fn production() -> Self {
        DeploymentConfig {
            environment: "production".to_string(),
            resources: CloudResourceConfig {
                min_instances: 3,
                max_instances: 20,
                instance_type: "c5.xlarge".to_string(),
                auto_scaling_enabled: true,
                cost_optimization_enabled: true,
                ..Default::default()
            },
            network_config: NetworkConfig {
                vpc_cidr: "10.0.0.0/16".to_string(),
                subnet_cidrs: vec![
                    "10.0.1.0/24".to_string(),
                    "10.0.2.0/24".to_string(),
                    "10.0.3.0/24".to_string(),
                ],
                load_balancer_enabled: true,
                ssl_enabled: true,
                firewall_rules: vec![
                    FirewallRule {
                        direction: "inbound".to_string(),
                        protocol: "tcp".to_string(),
                        port_range: "443".to_string(),
                        source_cidrs: vec!["0.0.0.0/0".to_string()],
                        action: "allow".to_string(),
                    },
                    FirewallRule {
                        direction: "inbound".to_string(),
                        protocol: "tcp".to_string(),
                        port_range: "22".to_string(),
                        source_cidrs: vec!["10.0.0.0/16".to_string()],
                        action: "allow".to_string(),
                    },
                ],
            },
            security_config: SecurityConfig {
                encryption_at_rest: true,
                encryption_in_transit: true,
                access_control_enabled: true,
                audit_logging_enabled: true,
                key_management_service: "aws-kms".to_string(),
            },
            monitoring_config: MonitoringConfig {
                metrics_enabled: true,
                logging_enabled: true,
                alerting_enabled: true,
                dashboard_enabled: true,
                retention_days: 90,
            },
            backup_config: BackupConfig {
                backup_enabled: true,
                backup_frequency: "hourly".to_string(),
                retention_policy: "30 days".to_string(),
                cross_region_replication: true,
                point_in_time_recovery: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deployment_config_creation() {
        let config = DeploymentConfig::development();
        assert_eq!(config.environment, "development");
        assert_eq!(config.resources.min_instances, 1);
        assert_eq!(config.resources.max_instances, 2);
    }

    #[test]
    fn test_cloud_orchestrator_creation() {
        let config = DeploymentConfig::development();
        let orchestrator = CloudDeploymentOrchestrator::new(config);
        assert!(matches!(
            orchestrator.deployment_state.status,
            DeploymentStatus::Initializing
        ));
    }

    #[test]
    fn test_job_creation() {
        let job = CloudTimeSeriesJob {
            job_id: "test-job-001".to_string(),
            job_type: TimeSeriesJobType::Forecasting,
            input_data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            parameters: HashMap::new(),
            priority: JobPriority::Normal,
            estimated_duration: Duration::from_secs(300),
            resource_requirements: ResourceRequirements {
                cpu_cores: 2,
                memory_gb: 4.0,
                gpu_required: false,
                storage_gb: 10.0,
                network_bandwidth_mbps: 100.0,
            },
        };

        assert_eq!(job.job_id, "test-job-001");
        assert!(matches!(job.job_type, TimeSeriesJobType::Forecasting));
    }
}
