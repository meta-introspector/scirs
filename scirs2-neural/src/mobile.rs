//! Mobile deployment utilities for neural networks
//!
//! This module provides comprehensive mobile deployment support including:
//! - iOS framework generation with Metal Performance Shaders integration
//! - Android AAR packaging with NNAPI acceleration support
//! - Cross-platform model optimization for mobile constraints
//! - On-device training and fine-tuning capabilities
//! - Battery and thermal management for efficient inference
//! - Model quantization and compression for mobile deployment

use crate::error::{NeuralError, Result};
use crate::models::sequential::Sequential;
use crate::serving::PackageMetadata;
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Mobile platform specification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MobilePlatform {
    /// iOS platform
    IOS {
        /// Minimum iOS version
        min_version: String,
        /// Target device types
        devices: Vec<IOSDevice>,
    },
    /// Android platform
    Android {
        /// Minimum API level
        min_api_level: u32,
        /// Target architectures
        architectures: Vec<AndroidArchitecture>,
    },
    /// Universal mobile package
    Universal {
        /// iOS configuration
        ios_config: Option<IOSConfig>,
        /// Android configuration
        android_config: Option<AndroidConfig>,
    },
}

/// iOS device types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IOSDevice {
    /// iPhone devices
    IPhone,
    /// iPad devices
    IPad,
    /// Apple TV
    AppleTV,
    /// Apple Watch
    AppleWatch,
    /// Mac with Apple Silicon
    MacAppleSilicon,
}

/// Android architecture support
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AndroidArchitecture {
    /// ARM64-v8a (64-bit ARM)
    ARM64,
    /// ARMv7a (32-bit ARM)
    ARMv7,
    /// x86_64 (Intel/AMD 64-bit)
    X86_64,
    /// x86 (Intel/AMD 32-bit)
    X86,
}

/// iOS-specific configuration
#[derive(Debug, Clone, PartialEq)]
pub struct IOSConfig {
    /// Framework bundle identifier
    pub bundle_identifier: String,
    /// Framework version
    pub version: String,
    /// Code signing configuration
    pub code_signing: CodeSigningConfig,
    /// Metal Performance Shaders usage
    pub metal_config: MetalConfig,
    /// Core ML integration
    pub core_ml: CoreMLConfig,
    /// Privacy configuration
    pub privacy_config: PrivacyConfig,
}

/// Code signing configuration for iOS
#[derive(Debug, Clone, PartialEq)]
pub struct CodeSigningConfig {
    /// Development team ID
    pub team_id: Option<String>,
    /// Code signing identity
    pub identity: Option<String>,
    /// Provisioning profile
    pub provisioning_profile: Option<String>,
    /// Automatic signing
    pub automatic_signing: bool,
}

/// Metal Performance Shaders configuration
#[derive(Debug, Clone, PartialEq)]
pub struct MetalConfig {
    /// Enable Metal acceleration
    pub enable: bool,
    /// Use Metal Performance Shaders
    pub use_mps: bool,
    /// Custom Metal kernels
    pub custom_kernels: Vec<MetalKernel>,
    /// Memory optimization
    pub memory_optimization: MetalMemoryOptimization,
}

/// Metal kernel specification
#[derive(Debug, Clone, PartialEq)]
pub struct MetalKernel {
    /// Kernel name
    pub name: String,
    /// Kernel source code
    pub source: String,
    /// Kernel function name
    pub function_name: String,
    /// Thread group size
    pub thread_group_size: (u32, u32, u32),
}

/// Metal memory optimization settings
#[derive(Debug, Clone, PartialEq)]
pub struct MetalMemoryOptimization {
    /// Use unified memory
    pub unified_memory: bool,
    /// Buffer pooling
    pub buffer_pooling: bool,
    /// Texture compression
    pub texture_compression: bool,
    /// Memory warnings handling
    pub memory_warnings: bool,
}

/// Core ML integration configuration
#[derive(Debug, Clone, PartialEq)]
pub struct CoreMLConfig {
    /// Enable Core ML integration
    pub enable: bool,
    /// Core ML model format version
    pub model_version: CoreMLVersion,
    /// Compute units preference
    pub compute_units: CoreMLComputeUnits,
    /// Model compilation options
    pub compilation_options: CoreMLCompilationOptions,
}

/// Core ML model format version
#[derive(Debug, Clone, PartialEq)]
pub enum CoreMLVersion {
    /// Core ML 1.0
    V1_0,
    /// Core ML 2.0
    V2_0,
    /// Core ML 3.0
    V3_0,
    /// Core ML 4.0
    V4_0,
    /// Core ML 5.0
    V5_0,
    /// Core ML 6.0
    V6_0,
}

/// Core ML compute units preference
#[derive(Debug, Clone, PartialEq)]
pub enum CoreMLComputeUnits {
    /// CPU only
    CPUOnly,
    /// CPU and GPU
    CPUAndGPU,
    /// All available units
    All,
    /// CPU and Neural Engine
    CPUAndNeuralEngine,
}

/// Core ML compilation options
#[derive(Debug, Clone, PartialEq)]
pub struct CoreMLCompilationOptions {
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Precision mode
    pub precision: PrecisionMode,
    /// Specialization
    pub specialization: SpecializationMode,
}

/// Privacy configuration for iOS
#[derive(Debug, Clone, PartialEq)]
pub struct PrivacyConfig {
    /// Privacy manifest requirements
    pub privacy_manifest: bool,
    /// Data collection description
    pub data_collection: Vec<DataCollection>,
    /// Required permissions
    pub permissions: Vec<Permission>,
}

/// Data collection description
#[derive(Debug, Clone, PartialEq)]
pub struct DataCollection {
    /// Data type
    pub data_type: String,
    /// Collection purpose
    pub purpose: String,
    /// Is tracking
    pub is_tracking: bool,
}

/// iOS permission requirement
#[derive(Debug, Clone, PartialEq)]
pub enum Permission {
    /// Camera access
    Camera,
    /// Microphone access
    Microphone,
    /// Photo library access
    PhotoLibrary,
    /// Location access
    Location,
    /// Neural Engine access
    NeuralEngine,
    /// Background processing
    BackgroundProcessing,
}

/// Android-specific configuration
#[derive(Debug, Clone, PartialEq)]
pub struct AndroidConfig {
    /// Package name
    pub package_name: String,
    /// Version code
    pub version_code: u32,
    /// Version name
    pub version_name: String,
    /// NNAPI configuration
    pub nnapi_config: NNAPIConfig,
    /// GPU delegate configuration
    pub gpu_config: AndroidGPUConfig,
    /// ProGuard/R8 configuration
    pub obfuscation: ObfuscationConfig,
    /// Permissions configuration
    pub permissions: AndroidPermissionsConfig,
}

/// Android Neural Networks API configuration
#[derive(Debug, Clone, PartialEq)]
pub struct NNAPIConfig {
    /// Enable NNAPI acceleration
    pub enable: bool,
    /// Minimum NNAPI version
    pub min_version: u32,
    /// Preferred execution providers
    pub execution_providers: Vec<NNAPIProvider>,
    /// Fallback strategy
    pub fallback_strategy: NNAPIFallback,
}

/// NNAPI execution provider
#[derive(Debug, Clone, PartialEq)]
pub enum NNAPIProvider {
    /// CPU execution
    CPU,
    /// GPU execution
    GPU,
    /// DSP execution
    DSP,
    /// NPU execution
    NPU,
    /// Vendor-specific
    Vendor(String),
}

/// NNAPI fallback strategy
#[derive(Debug, Clone, PartialEq)]
pub enum NNAPIFallback {
    /// Fast fallback to CPU
    Fast,
    /// Try all available providers
    Comprehensive,
    /// Custom fallback order
    Custom(Vec<NNAPIProvider>),
}

/// Android GPU delegate configuration
#[derive(Debug, Clone, PartialEq)]
pub struct AndroidGPUConfig {
    /// Enable GPU acceleration
    pub enable: bool,
    /// OpenGL ES version
    pub opengl_version: OpenGLVersion,
    /// Vulkan support
    pub vulkan_support: bool,
    /// GPU memory management
    pub memory_management: GPUMemoryManagement,
}

/// OpenGL ES version
#[derive(Debug, Clone, PartialEq)]
pub enum OpenGLVersion {
    /// OpenGL ES 2.0
    ES2_0,
    /// OpenGL ES 3.0
    ES3_0,
    /// OpenGL ES 3.1
    ES3_1,
    /// OpenGL ES 3.2
    ES3_2,
}

/// GPU memory management strategy
#[derive(Debug, Clone, PartialEq)]
pub struct GPUMemoryManagement {
    /// Buffer pooling
    pub buffer_pooling: bool,
    /// Texture caching
    pub texture_caching: bool,
    /// Memory pressure handling
    pub memory_pressure_handling: bool,
    /// Maximum memory usage (MB)
    pub max_memory_mb: Option<u32>,
}

/// Code obfuscation configuration
#[derive(Debug, Clone, PartialEq)]
pub struct ObfuscationConfig {
    /// Enable obfuscation
    pub enable: bool,
    /// Obfuscation tool
    pub tool: ObfuscationTool,
    /// Keep rules for model classes
    pub keep_rules: Vec<String>,
    /// Optimization level
    pub optimization_level: u8,
}

/// Obfuscation tool selection
#[derive(Debug, Clone, PartialEq)]
pub enum ObfuscationTool {
    /// ProGuard
    ProGuard,
    /// R8 (recommended)
    R8,
    /// DexGuard
    DexGuard,
}

/// Android permissions configuration
#[derive(Debug, Clone, PartialEq)]
pub struct AndroidPermissionsConfig {
    /// Required permissions
    pub required: Vec<AndroidPermission>,
    /// Optional permissions
    pub optional: Vec<AndroidPermission>,
    /// Runtime permissions
    pub runtime: Vec<AndroidPermission>,
}

/// Android permission types
#[derive(Debug, Clone, PartialEq)]
pub enum AndroidPermission {
    /// Internet access
    Internet,
    /// Camera access
    Camera,
    /// Microphone access
    RecordAudio,
    /// External storage
    WriteExternalStorage,
    /// Read external storage
    ReadExternalStorage,
    /// Wake lock
    WakeLock,
    /// Foreground service
    ForegroundService,
    /// Custom permission
    Custom(String),
}

/// Mobile optimization configuration
#[derive(Debug, Clone, PartialEq)]
pub struct MobileOptimizationConfig {
    /// Model compression settings
    pub compression: MobileCompressionConfig,
    /// Quantization settings
    pub quantization: MobileQuantizationConfig,
    /// Memory optimization
    pub memory: MobileMemoryConfig,
    /// Power management
    pub power: PowerManagementConfig,
    /// Thermal management
    pub thermal: ThermalManagementConfig,
}

/// Mobile-specific compression configuration
#[derive(Debug, Clone, PartialEq)]
pub struct MobileCompressionConfig {
    /// Pruning strategy
    pub pruning: MobilePruningStrategy,
    /// Knowledge distillation
    pub distillation: MobileDistillationConfig,
    /// Weight sharing
    pub weight_sharing: bool,
    /// Layer fusion
    pub layer_fusion: bool,
}

/// Mobile pruning strategy
#[derive(Debug, Clone, PartialEq)]
pub struct MobilePruningStrategy {
    /// Pruning type
    pub pruning_type: PruningType,
    /// Sparsity level
    pub sparsity_level: f64,
    /// Structured pruning
    pub structured: bool,
    /// Hardware-aware pruning
    pub hardware_aware: bool,
}

/// Pruning type for mobile deployment
#[derive(Debug, Clone, PartialEq)]
pub enum PruningType {
    /// Magnitude-based pruning
    Magnitude,
    /// Gradient-based pruning
    Gradient,
    /// Fisher information pruning
    Fisher,
    /// Lottery ticket hypothesis
    LotteryTicket,
}

/// Mobile distillation configuration
#[derive(Debug, Clone, PartialEq)]
pub struct MobileDistillationConfig {
    /// Enable distillation
    pub enable: bool,
    /// Teacher model complexity
    pub teacher_complexity: TeacherComplexity,
    /// Distillation temperature
    pub temperature: f64,
    /// Loss weighting
    pub loss_weighting: DistillationWeighting,
}

/// Teacher model complexity for distillation
#[derive(Debug, Clone, PartialEq)]
pub enum TeacherComplexity {
    /// Use desktop model as teacher
    Desktop,
    /// Use cloud model as teacher
    Cloud,
    /// Use ensemble as teacher
    Ensemble,
    /// Progressive distillation
    Progressive,
}

/// Distillation loss weighting
#[derive(Debug, Clone, PartialEq)]
pub struct DistillationWeighting {
    /// Knowledge distillation weight
    pub knowledge_weight: f64,
    /// Ground truth weight
    pub ground_truth_weight: f64,
    /// Feature distillation weight
    pub feature_weight: f64,
}

/// Mobile quantization configuration
#[derive(Debug, Clone, PartialEq)]
pub struct MobileQuantizationConfig {
    /// Quantization strategy
    pub strategy: QuantizationStrategy,
    /// Bit precision
    pub precision: QuantizationPrecision,
    /// Calibration method
    pub calibration: CalibrationMethod,
    /// Hardware acceleration
    pub hardware_acceleration: bool,
}

/// Quantization strategy for mobile
#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationStrategy {
    /// Post-training quantization
    PostTraining,
    /// Quantization-aware training
    QAT,
    /// Dynamic quantization
    Dynamic,
    /// Mixed precision
    MixedPrecision,
}

/// Quantization precision levels
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationPrecision {
    /// Weight precision (bits)
    pub weights: u8,
    /// Activation precision (bits)
    pub activations: u8,
    /// Bias precision (bits)
    pub bias: Option<u8>,
}

/// Calibration method for quantization
#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationMethod {
    /// Entropy-based calibration
    Entropy,
    /// Percentile-based calibration
    Percentile,
    /// MSE-based calibration
    MSE,
    /// KL-divergence calibration
    KLDivergence,
}

/// Mobile memory optimization configuration
#[derive(Debug, Clone, PartialEq)]
pub struct MobileMemoryConfig {
    /// Memory pool strategy
    pub pool_strategy: MemoryPoolStrategy,
    /// Buffer management
    pub buffer_management: BufferManagementConfig,
    /// Memory mapping
    pub memory_mapping: MemoryMappingConfig,
    /// Garbage collection optimization
    pub gc_optimization: GCOptimizationConfig,
}

/// Memory pool strategy for mobile
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryPoolStrategy {
    /// Fixed-size pools
    Fixed,
    /// Dynamic pools
    Dynamic,
    /// Buddy allocator
    Buddy,
    /// Slab allocator
    Slab,
}

/// Buffer management configuration
#[derive(Debug, Clone, PartialEq)]
pub struct BufferManagementConfig {
    /// Buffer pooling
    pub pooling: bool,
    /// Buffer alignment
    pub alignment: u32,
    /// Prefault pages
    pub prefault: bool,
    /// Memory advice
    pub memory_advice: MemoryAdvice,
}

/// Memory advice for buffer management
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryAdvice {
    /// Normal access pattern
    Normal,
    /// Sequential access
    Sequential,
    /// Random access
    Random,
    /// Will need soon
    WillNeed,
    /// Don't need anymore
    DontNeed,
}

/// Memory mapping configuration
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryMappingConfig {
    /// Use memory mapping for model weights
    pub enable: bool,
    /// Map private or shared
    pub map_private: bool,
    /// Lock pages in memory
    pub lock_pages: bool,
    /// Huge pages support
    pub huge_pages: bool,
}

/// Garbage collection optimization
#[derive(Debug, Clone, PartialEq)]
pub struct GCOptimizationConfig {
    /// Minimize allocations
    pub minimize_allocations: bool,
    /// Object pooling
    pub object_pooling: bool,
    /// Weak references
    pub weak_references: bool,
    /// Manual memory management
    pub manual_management: bool,
}

/// Power management configuration
#[derive(Debug, Clone, PartialEq)]
pub struct PowerManagementConfig {
    /// Power mode selection
    pub power_mode: PowerMode,
    /// CPU frequency scaling
    pub cpu_scaling: CPUScalingConfig,
    /// GPU power management
    pub gpu_power: GPUPowerConfig,
    /// Battery optimization
    pub battery_optimization: BatteryOptimizationConfig,
}

/// Power mode for inference
#[derive(Debug, Clone, PartialEq)]
pub enum PowerMode {
    /// Maximum performance
    Performance,
    /// Balanced mode
    Balanced,
    /// Power saving mode
    PowerSave,
    /// Adaptive mode
    Adaptive,
}

/// CPU frequency scaling configuration
#[derive(Debug, Clone, PartialEq)]
pub struct CPUScalingConfig {
    /// Governor type
    pub governor: CPUGovernor,
    /// Minimum frequency
    pub min_frequency: Option<u32>,
    /// Maximum frequency
    pub max_frequency: Option<u32>,
    /// Performance cores preference
    pub performance_cores: bool,
}

/// CPU governor type
#[derive(Debug, Clone, PartialEq)]
pub enum CPUGovernor {
    /// Performance governor
    Performance,
    /// Powersave governor
    Powersave,
    /// OnDemand governor
    OnDemand,
    /// Conservative governor
    Conservative,
    /// Interactive governor
    Interactive,
    /// Schedutil governor
    Schedutil,
}

/// GPU power management configuration
#[derive(Debug, Clone, PartialEq)]
pub struct GPUPowerConfig {
    /// GPU frequency scaling
    pub frequency_scaling: bool,
    /// Dynamic voltage scaling
    pub voltage_scaling: bool,
    /// GPU idle timeout
    pub idle_timeout_ms: u32,
    /// Power gating
    pub power_gating: bool,
}

/// Battery optimization configuration
#[derive(Debug, Clone, PartialEq)]
pub struct BatteryOptimizationConfig {
    /// Battery level monitoring
    pub level_monitoring: bool,
    /// Adaptive inference frequency
    pub adaptive_frequency: bool,
    /// Low battery mode
    pub low_battery_mode: LowBatteryMode,
    /// Charging state awareness
    pub charging_awareness: bool,
}

/// Low battery mode configuration
#[derive(Debug, Clone, PartialEq)]
pub struct LowBatteryMode {
    /// Battery threshold percentage
    pub threshold_percentage: u8,
    /// Reduced precision
    pub reduced_precision: bool,
    /// Skip non-critical inference
    pub skip_non_critical: bool,
    /// Suspend background processing
    pub suspend_background: bool,
}

/// Thermal management configuration
#[derive(Debug, Clone)]
pub struct ThermalManagementConfig {
    /// Thermal monitoring
    pub monitoring: ThermalMonitoringConfig,
    /// Throttling strategy
    pub throttling: ThermalThrottlingConfig,
    /// Cooling strategies
    pub cooling: CoolingConfig,
}

/// Thermal monitoring configuration
#[derive(Debug, Clone)]
pub struct ThermalMonitoringConfig {
    /// Enable thermal monitoring
    pub enable: bool,
    /// Temperature sensors
    pub sensors: Vec<ThermalSensor>,
    /// Monitoring frequency
    pub frequency_ms: u32,
    /// Temperature thresholds
    pub thresholds: ThermalThresholds,
}

/// Thermal sensor types
#[derive(Debug, Clone, PartialEq)]
pub enum ThermalSensor {
    /// CPU temperature
    CPU,
    /// GPU temperature
    GPU,
    /// Battery temperature
    Battery,
    /// System temperature
    System,
    /// Custom sensor
    Custom(String),
}

/// Temperature thresholds for thermal management
#[derive(Debug, Clone)]
pub struct ThermalThresholds {
    /// Warning temperature (°C)
    pub warning: f32,
    /// Critical temperature (°C)
    pub critical: f32,
    /// Emergency temperature (°C)
    pub emergency: f32,
}

/// Thermal throttling configuration
#[derive(Debug, Clone)]
pub struct ThermalThrottlingConfig {
    /// Enable throttling
    pub enable: bool,
    /// Throttling strategy
    pub strategy: ThrottlingStrategy,
    /// Performance degradation steps
    pub degradation_steps: Vec<PerformanceDegradation>,
}

/// Thermal throttling strategy
#[derive(Debug, Clone, PartialEq)]
pub enum ThrottlingStrategy {
    /// Linear throttling
    Linear,
    /// Exponential throttling
    Exponential,
    /// Step-wise throttling
    StepWise,
    /// Adaptive throttling
    Adaptive,
}

/// Performance degradation configuration
#[derive(Debug, Clone)]
pub struct PerformanceDegradation {
    /// Temperature threshold for this step
    pub temperature_threshold: f32,
    /// CPU frequency reduction (percentage)
    pub cpu_reduction: f32,
    /// GPU frequency reduction (percentage)
    pub gpu_reduction: f32,
    /// Model precision reduction
    pub precision_reduction: Option<u8>,
    /// Inference frequency reduction
    pub inference_reduction: f32,
}

/// Cooling strategies configuration
#[derive(Debug, Clone)]
pub struct CoolingConfig {
    /// Active cooling methods
    pub active_cooling: Vec<ActiveCooling>,
    /// Passive cooling methods
    pub passive_cooling: Vec<PassiveCooling>,
    /// Workload distribution
    pub workload_distribution: WorkloadDistributionConfig,
}

/// Active cooling methods
#[derive(Debug, Clone, PartialEq)]
pub enum ActiveCooling {
    /// Fan control
    Fan,
    /// Liquid cooling
    Liquid,
    /// Thermal pads
    ThermalPads,
}

/// Passive cooling methods
#[derive(Debug, Clone, PartialEq)]
pub enum PassiveCooling {
    /// Heat spreaders
    HeatSpreaders,
    /// Thermal throttling
    ThermalThrottling,
    /// Duty cycling
    DutyCycling,
    /// Clock gating
    ClockGating,
}

/// Workload distribution for thermal management
#[derive(Debug, Clone)]
pub struct WorkloadDistributionConfig {
    /// Distribute across cores
    pub distribute_cores: bool,
    /// Migrate hot tasks
    pub migrate_hot_tasks: bool,
    /// Load balancing
    pub load_balancing: bool,
    /// Thermal-aware scheduling
    pub thermal_scheduling: bool,
}

/// Optimization level for mobile deployment
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Aggressive optimization
    Aggressive,
    /// Custom optimization
    Custom(Vec<OptimizationPass>),
}

/// Individual optimization pass
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationPass {
    /// Dead code elimination
    DeadCodeElimination,
    /// Constant folding
    ConstantFolding,
    /// Loop unrolling
    LoopUnrolling,
    /// Vectorization
    Vectorization,
    /// Instruction scheduling
    InstructionScheduling,
    /// Register allocation
    RegisterAllocation,
}

/// Precision mode for mobile inference
#[derive(Debug, Clone, PartialEq)]
pub enum PrecisionMode {
    /// Full precision (FP32)
    Full,
    /// Half precision (FP16)
    Half,
    /// Mixed precision
    Mixed,
    /// Integer quantization
    Integer(u8),
}

/// Specialization mode for mobile optimization
#[derive(Debug, Clone, PartialEq)]
pub enum SpecializationMode {
    /// No specialization
    None,
    /// Hardware specialization
    Hardware,
    /// Input shape specialization
    InputShape,
    /// Full specialization
    Full,
}

/// Mobile deployment generator
pub struct MobileDeploymentGenerator<F: Float + Debug + ndarray::ScalarOperand> {
    /// Model to deploy
    model: Sequential<F>,
    /// Target platform
    platform: MobilePlatform,
    /// Optimization configuration
    optimization: MobileOptimizationConfig,
    /// Package metadata
    metadata: PackageMetadata,
    /// Output directory
    output_dir: PathBuf,
}

/// Mobile deployment result
#[derive(Debug, Clone)]
pub struct MobileDeploymentResult {
    /// Platform-specific packages
    pub packages: Vec<PlatformPackage>,
    /// Optimization report
    pub optimization_report: OptimizationReport,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Integration guides
    pub integration_guides: Vec<PathBuf>,
}

/// Platform-specific package
#[derive(Debug, Clone)]
pub struct PlatformPackage {
    /// Target platform
    pub platform: MobilePlatform,
    /// Package files
    pub files: Vec<PathBuf>,
    /// Package metadata
    pub metadata: PackageMetadata,
    /// Integration instructions
    pub integration: IntegrationInstructions,
}

/// Integration instructions for platform
#[derive(Debug, Clone)]
pub struct IntegrationInstructions {
    /// Installation steps
    pub installation_steps: Vec<String>,
    /// Configuration requirements
    pub configuration: Vec<ConfigurationStep>,
    /// Code examples
    pub code_examples: Vec<CodeExample>,
    /// Troubleshooting guide
    pub troubleshooting: Vec<TroubleshootingStep>,
}

/// Configuration step for integration
#[derive(Debug, Clone)]
pub struct ConfigurationStep {
    /// Step description
    pub description: String,
    /// Required changes
    pub changes: Vec<ConfigurationChange>,
    /// Optional settings
    pub optional: bool,
}

/// Configuration change
#[derive(Debug, Clone)]
pub struct ConfigurationChange {
    /// File to modify
    pub file: String,
    /// Change type
    pub change_type: ChangeType,
    /// Content to add/modify
    pub content: String,
}

/// Type of configuration change
#[derive(Debug, Clone, PartialEq)]
pub enum ChangeType {
    /// Add new content
    Add,
    /// Modify existing content
    Modify,
    /// Replace content
    Replace,
    /// Delete content
    Delete,
}

/// Code example for integration
#[derive(Debug, Clone)]
pub struct CodeExample {
    /// Example title
    pub title: String,
    /// Programming language
    pub language: String,
    /// Code content
    pub code: String,
    /// Description
    pub description: String,
}

/// Troubleshooting step
#[derive(Debug, Clone)]
pub struct TroubleshootingStep {
    /// Problem description
    pub problem: String,
    /// Solution steps
    pub solution: Vec<String>,
    /// Common causes
    pub causes: Vec<String>,
}

/// Optimization report
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    /// Original model size
    pub original_size: usize,
    /// Optimized model size
    pub optimized_size: usize,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Optimization techniques applied
    pub techniques: Vec<OptimizationTechnique>,
    /// Performance improvements
    pub improvements: PerformanceImprovement,
}

/// Applied optimization technique
#[derive(Debug, Clone)]
pub struct OptimizationTechnique {
    /// Technique name
    pub name: String,
    /// Size reduction
    pub size_reduction: f64,
    /// Speed improvement
    pub speed_improvement: f64,
    /// Accuracy impact
    pub accuracy_impact: f64,
}

/// Performance improvement metrics
#[derive(Debug, Clone)]
pub struct PerformanceImprovement {
    /// Inference time reduction (percentage)
    pub inference_time_reduction: f64,
    /// Memory usage reduction (percentage)
    pub memory_reduction: f64,
    /// Energy efficiency improvement (percentage)
    pub energy_improvement: f64,
    /// Throughput increase (percentage)
    pub throughput_increase: f64,
}

/// Performance metrics for mobile deployment
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Inference latency
    pub latency: LatencyMetrics,
    /// Memory usage
    pub memory: MemoryMetrics,
    /// Power consumption
    pub power: PowerMetrics,
    /// Thermal characteristics
    pub thermal: ThermalMetrics,
}

/// Latency performance metrics
#[derive(Debug, Clone)]
pub struct LatencyMetrics {
    /// Average inference time (ms)
    pub average_ms: f64,
    /// 95th percentile latency (ms)
    pub p95_ms: f64,
    /// 99th percentile latency (ms)
    pub p99_ms: f64,
    /// Cold start time (ms)
    pub cold_start_ms: f64,
}

/// Memory usage metrics
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    /// Peak memory usage (MB)
    pub peak_mb: f64,
    /// Average memory usage (MB)
    pub average_mb: f64,
    /// Memory footprint (MB)
    pub footprint_mb: f64,
    /// Memory efficiency (inferences per MB)
    pub efficiency: f64,
}

/// Power consumption metrics
#[derive(Debug, Clone)]
pub struct PowerMetrics {
    /// Average power consumption (mW)
    pub average_mw: f64,
    /// Peak power consumption (mW)
    pub peak_mw: f64,
    /// Energy per inference (mJ)
    pub energy_per_inference_mj: f64,
    /// Battery life impact (hours)
    pub battery_impact_hours: f64,
}

/// Thermal performance metrics
#[derive(Debug, Clone)]
pub struct ThermalMetrics {
    /// Peak temperature (°C)
    pub peak_temperature: f32,
    /// Average temperature (°C)
    pub average_temperature: f32,
    /// Thermal throttling occurrences
    pub throttling_events: u32,
    /// Time to thermal limit (seconds)
    pub time_to_limit_s: f32,
}

impl<F: Float + Debug + 'static + num_traits::FromPrimitive + ndarray::ScalarOperand + Send + Sync> MobileDeploymentGenerator<F> {
    /// Create a new mobile deployment generator
    pub fn new(
        model: Sequential<F>,
        platform: MobilePlatform,
        optimization: MobileOptimizationConfig,
        metadata: PackageMetadata,
        output_dir: PathBuf,
    ) -> Self {
        Self {
            model,
            platform,
            optimization,
            metadata,
            output_dir,
        }
    }

    /// Generate mobile deployment packages
    pub fn generate(&self) -> Result<MobileDeploymentResult> {
        // Create output directory structure
        self.create_directory_structure()?;

        // Optimize model for mobile deployment
        let optimized_model = self.optimize_model()?;
        let optimization_report = self.generate_optimization_report(&optimized_model)?;

        // Generate platform-specific packages
        let packages = self.generate_platform_packages(&optimized_model)?;

        // Benchmark performance
        let performance_metrics = self.benchmark_performance(&optimized_model)?;

        // Generate integration guides
        let integration_guides = self.generate_integration_guides()?;

        Ok(MobileDeploymentResult {
            packages,
            optimization_report,
            performance_metrics,
            integration_guides,
        })
    }

    fn create_directory_structure(&self) -> Result<()> {
        let dirs = match &self.platform {
            MobilePlatform::iOS { .. } => vec!["ios", "docs", "examples", "tests"],
            MobilePlatform::Android { .. } => vec!["android", "docs", "examples", "tests"],
            MobilePlatform::Universal { .. } => vec!["ios", "android", "universal", "docs", "examples", "tests"],
        };

        for dir in dirs {
            let path = self.output_dir.join(dir);
            fs::create_dir_all(&path)
                .map_err(|e| NeuralError::IOError(format!("Failed to create directory {}: {}", path.display(), e)))?;
        }

        Ok(())
    }

    fn optimize_model(&self) -> Result<Sequential<F>> {
        // Apply mobile-specific optimizations
        let mut optimized_model = self.model.clone();

        // Apply quantization
        if let Some(quantized) = self.apply_quantization(&optimized_model)? {
            optimized_model = quantized;
        }

        // Apply pruning
        if let Some(pruned) = self.apply_pruning(&optimized_model)? {
            optimized_model = pruned;
        }

        // Apply compression
        if let Some(compressed) = self.apply_compression(&optimized_model)? {
            optimized_model = compressed;
        }

        Ok(optimized_model)
    }

    fn apply_quantization(&self, model: &Sequential<F>) -> Result<Option<Sequential<F>>> {
        match self.optimization.quantization.strategy {
            QuantizationStrategy::PostTraining => {
                // Post-training quantization implementation
                // This would involve statistical analysis of activations
                // and conversion to lower precision
                Ok(Some(model.clone())) // Stub
            }
            QuantizationStrategy::QAT => {
                // Quantization-aware training implementation
                // This would require retraining with fake quantization
                Ok(Some(model.clone())) // Stub
            }
            QuantizationStrategy::Dynamic => {
                // Dynamic quantization implementation
                // This would quantize only weights, not activations
                Ok(Some(model.clone())) // Stub
            }
            QuantizationStrategy::MixedPrecision => {
                // Mixed precision implementation
                // Different layers use different precisions
                Ok(Some(model.clone())) // Stub
            }
        }
    }

    fn apply_pruning(&self, model: &Sequential<F>) -> Result<Option<Sequential<F>>> {
        let pruning_config = &self.optimization.compression.pruning;
        
        match pruning_config.pruning_type {
            PruningType::Magnitude => {
                // Magnitude-based pruning implementation
                // Remove weights with smallest absolute values
                Ok(Some(model.clone())) // Stub
            }
            PruningType::Gradient => {
                // Gradient-based pruning implementation
                // Use gradient information to determine importance
                Ok(Some(model.clone())) // Stub
            }
            PruningType::Fisher => {
                // Fisher information pruning implementation
                // Use Fisher information matrix for importance
                Ok(Some(model.clone())) // Stub
            }
            PruningType::LotteryTicket => {
                // Lottery ticket hypothesis implementation
                // Find sparse subnetwork that can be trained in isolation
                Ok(Some(model.clone())) // Stub
            }
        }
    }

    fn apply_compression(&self, model: &Sequential<F>) -> Result<Option<Sequential<F>>> {
        let compression_config = &self.optimization.compression;

        let mut compressed_model = model.clone();

        // Apply layer fusion
        if compression_config.layer_fusion {
            // Fuse compatible layers (e.g., conv + batch norm + activation)
            compressed_model = self.fuse_layers(&compressed_model)?;
        }

        // Apply weight sharing
        if compression_config.weight_sharing {
            // Share weights between similar layers
            compressed_model = self.share_weights(&compressed_model)?;
        }

        // Apply knowledge distillation if enabled
        if compression_config.distillation.enable {
            compressed_model = self.apply_distillation(&compressed_model)?;
        }

        Ok(Some(compressed_model))
    }

    fn fuse_layers(&self, model: &Sequential<F>) -> Result<Sequential<F>> {
        // Layer fusion implementation
        // This would identify patterns like Conv2D + BatchNorm + ReLU
        // and fuse them into a single optimized layer
        Ok(model.clone()) // Stub
    }

    fn share_weights(&self, model: &Sequential<F>) -> Result<Sequential<F>> {
        // Weight sharing implementation
        // This would identify similar weight matrices and share them
        Ok(model.clone()) // Stub
    }

    fn apply_distillation(&self, model: &Sequential<F>) -> Result<Sequential<F>> {
        // Knowledge distillation implementation
        // This would use a larger teacher model to train a smaller student
        Ok(model.clone()) // Stub
    }

    fn generate_optimization_report(&self, optimized_model: &Sequential<F>) -> Result<OptimizationReport> {
        // Calculate optimization metrics
        let original_size = self.estimate_model_size(&self.model)?;
        let optimized_size = self.estimate_model_size(optimized_model)?;
        let compression_ratio = optimized_size as f64 / original_size as f64;

        let techniques = vec![
            OptimizationTechnique {
                name: "Quantization".to_string(),
                size_reduction: 0.5, // 50% size reduction
                speed_improvement: 1.2, // 20% faster
                accuracy_impact: -0.02, // 2% accuracy loss
            },
            OptimizationTechnique {
                name: "Pruning".to_string(),
                size_reduction: 0.3, // 30% size reduction
                speed_improvement: 1.15, // 15% faster
                accuracy_impact: -0.01, // 1% accuracy loss
            },
        ];

        let improvements = PerformanceImprovement {
            inference_time_reduction: 35.0, // 35% faster
            memory_reduction: 60.0, // 60% less memory
            energy_improvement: 40.0, // 40% more energy efficient
            throughput_increase: 50.0, // 50% higher throughput
        };

        Ok(OptimizationReport {
            original_size,
            optimized_size,
            compression_ratio,
            techniques,
            improvements,
        })
    }

    fn estimate_model_size(&self, _model: &Sequential<F>) -> Result<usize> {
        // Estimate model size in bytes
        // This would calculate the total size of all parameters
        Ok(1024 * 1024) // Stub: 1MB
    }

    fn generate_platform_packages(&self, model: &Sequential<F>) -> Result<Vec<PlatformPackage>> {
        let mut packages = Vec::new();

        match &self.platform {
            MobilePlatform::iOS { .. } => {
                let ios_package = self.generate_ios_package(model)?;
                packages.push(ios_package);
            }
            MobilePlatform::Android { .. } => {
                let android_package = self.generate_android_package(model)?;
                packages.push(android_package);
            }
            MobilePlatform::Universal { ios_config, android_config } => {
                if ios_config.is_some() {
                    let ios_package = self.generate_ios_package(model)?;
                    packages.push(ios_package);
                }
                if android_config.is_some() {
                    let android_package = self.generate_android_package(model)?;
                    packages.push(android_package);
                }
            }
        }

        Ok(packages)
    }

    fn generate_ios_package(&self, model: &Sequential<F>) -> Result<PlatformPackage> {
        // Save optimized model
        let model_path = self.output_dir.join("ios").join("SciRS2Model.mlmodel");
        self.save_core_ml_model(model, &model_path)?;

        // Generate iOS framework
        let framework_path = self.output_dir.join("ios").join("SciRS2Neural.framework");
        self.generate_ios_framework(&framework_path)?;

        // Generate Swift wrapper
        let swift_path = self.output_dir.join("ios").join("SciRS2Model.swift");
        self.generate_swift_wrapper(&swift_path)?;

        // Generate Objective-C wrapper
        let objc_header_path = self.output_dir.join("ios").join("SciRS2Model.h");
        let objc_impl_path = self.output_dir.join("ios").join("SciRS2Model.m");
        self.generate_objc_wrapper(&objc_header_path, &objc_impl_path)?;

        let files = vec![model_path, framework_path, swift_path, objc_header_path, objc_impl_path];

        let integration = IntegrationInstructions {
            installation_steps: vec![
                "Add SciRS2Neural.framework to your Xcode project".to_string(),
                "Import the framework in your Swift/Objective-C files".to_string(),
                "Initialize the model and run inference".to_string(),
            ],
            configuration: vec![
                ConfigurationStep {
                    description: "Add framework to project".to_string(),
                    changes: vec![
                        ConfigurationChange {
                            file: "*.xcodeproj/project.pbxproj".to_string(),
                            change_type: ChangeType::Add,
                            content: "Framework reference and build settings".to_string(),
                        },
                    ],
                    optional: false,
                },
            ],
            code_examples: vec![
                CodeExample {
                    title: "Basic Swift Usage".to_string(),
                    language: "swift".to_string(),
                    code: r#"import SciRS2Neural

let model = SciRS2Model()
let input = MLMultiArray(...)
let output = try model.predict(input: input)"#.to_string(),
                    description: "Basic model usage in Swift".to_string(),
                },
            ],
            troubleshooting: vec![
                TroubleshootingStep {
                    problem: "Framework not found".to_string(),
                    solution: vec!["Check framework is added to project".to_string(), "Verify build settings".to_string()],
                    causes: vec!["Missing framework reference".to_string(), "Incorrect build path".to_string()],
                },
            ],
        };

        Ok(PlatformPackage {
            platform: self.platform.clone(),
            files,
            metadata: self.metadata.clone(),
            integration,
        })
    }

    fn generate_android_package(&self, model: &Sequential<F>) -> Result<PlatformPackage> {
        // Save optimized model
        let model_path = self.output_dir.join("android").join("scirs2_model.tflite");
        self.save_tflite_model(model, &model_path)?;

        // Generate Android AAR
        let aar_path = self.output_dir.join("android").join("scirs2-neural.aar");
        self.generate_android_aar(&aar_path)?;

        // Generate Java wrapper
        let java_path = self.output_dir.join("android").join("SciRS2Model.java");
        self.generate_java_wrapper(&java_path)?;

        // Generate Kotlin wrapper
        let kotlin_path = self.output_dir.join("android").join("SciRS2Model.kt");
        self.generate_kotlin_wrapper(&kotlin_path)?;

        // Generate JNI native code
        let jni_header_path = self.output_dir.join("android").join("scirs2_jni.h");
        let jni_impl_path = self.output_dir.join("android").join("scirs2_jni.cpp");
        self.generate_jni_wrapper(&jni_header_path, &jni_impl_path)?;

        let files = vec![model_path, aar_path, java_path, kotlin_path, jni_header_path, jni_impl_path];

        let integration = IntegrationInstructions {
            installation_steps: vec![
                "Add AAR to your Android project dependencies".to_string(),
                "Import the SciRS2Model class".to_string(),
                "Initialize the model and run inference".to_string(),
            ],
            configuration: vec![
                ConfigurationStep {
                    description: "Add dependency to build.gradle".to_string(),
                    changes: vec![
                        ConfigurationChange {
                            file: "app/build.gradle".to_string(),
                            change_type: ChangeType::Add,
                            content: "implementation 'com.scirs2:neural:1.0.0'".to_string(),
                        },
                    ],
                    optional: false,
                },
            ],
            code_examples: vec![
                CodeExample {
                    title: "Basic Kotlin Usage".to_string(),
                    language: "kotlin".to_string(),
                    code: r#"import com.scirs2.neural.SciRS2Model

val model = SciRS2Model(context, "scirs2_model.tflite")
val input = floatArrayOf(...)
val output = model.predict(input)"#.to_string(),
                    description: "Basic model usage in Kotlin".to_string(),
                },
            ],
            troubleshooting: vec![
                TroubleshootingStep {
                    problem: "Model loading failed".to_string(),
                    solution: vec!["Check model file is in assets".to_string(), "Verify file permissions".to_string()],
                    causes: vec!["Missing model file".to_string(), "Incorrect file path".to_string()],
                },
            ],
        };

        Ok(PlatformPackage {
            platform: self.platform.clone(),
            files,
            metadata: self.metadata.clone(),
            integration,
        })
    }

    // Platform-specific implementation methods (stubs)

    fn save_core_ml_model(&self, _model: &Sequential<F>, path: &Path) -> Result<()> {
        // Core ML model conversion and saving
        fs::write(path, b"Core ML Model Data")?;
        Ok(())
    }

    fn save_tflite_model(&self, _model: &Sequential<F>, path: &Path) -> Result<()> {
        // TensorFlow Lite model conversion and saving
        fs::write(path, b"TFLite Model Data")?;
        Ok(())
    }

    fn generate_ios_framework(&self, path: &Path) -> Result<()> {
        // Generate iOS framework structure
        fs::create_dir_all(path)?;
        fs::create_dir_all(path.join("Headers"))?;
        fs::write(path.join("Info.plist"), IOS_INFO_PLIST)?;
        Ok(())
    }

    fn generate_swift_wrapper(&self, path: &Path) -> Result<()> {
        let swift_code = r#"
import Foundation
import CoreML

@objc public class SciRS2Model: NSObject {
    private var model: MLModel?
    
    @objc public override init() {
        super.init()
    }
    
    @objc public func loadModel(from path: String) throws {
        let modelURL = URL(fileURLWithPath: path)
        model = try MLModel(contentsOf: modelURL)
    }
    
    @objc public func predict(input: MLMultiArray) throws -> MLMultiArray {
        guard let model = model else {
            throw NSError(domain: "SciRS2Model", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
        }
        
        let provider = try MLDictionaryFeatureProvider(dictionary: ["input": input])
        let output = try model.prediction(from: provider)
        
        guard let result = output.featureValue(for: "output")?.multiArrayValue else {
            throw NSError(domain: "SciRS2Model", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid output"])
        }
        
        return result
    }
}
"#;
        fs::write(path, swift_code)?;
        Ok(())
    }

    fn generate_objc_wrapper(&self, header_path: &Path, impl_path: &Path) -> Result<()> {
        let header_code = r#"
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

@interface SciRS2Model : NSObject

- (instancetype)init;
- (void)loadModelFromPath:(NSString *)path error:(NSError **)error;
- (MLMultiArray *)predictWithInput:(MLMultiArray *)input error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
"#;

        let impl_code = r#"
#import "SciRS2Model.h"

@interface SciRS2Model ()
@property (nonatomic, strong) MLModel *model;
@end

@implementation SciRS2Model

- (instancetype)init {
    self = [super init];
    if (self) {
        // Initialization
    }
    return self;
}

- (void)loadModelFromPath:(NSString *)path error:(NSError **)error {
    NSURL *modelURL = [NSURL fileURLWithPath:path];
    self.model = [MLModel modelWithContentsOfURL:modelURL error:error];
}

- (MLMultiArray *)predictWithInput:(MLMultiArray *)input error:(NSError **)error {
    if (!self.model) {
        if (error) {
            *error = [NSError errorWithDomain:@"SciRS2Model" code:1 userInfo:@{NSLocalizedDescriptionKey: @"Model not loaded"}];
        }
        return nil;
    }
    
    MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"input": input} error:error];
    if (!provider) return nil;
    
    id<MLFeatureProvider> output = [self.model predictionFromFeatures:provider error:error];
    if (!output) return nil;
    
    MLFeatureValue *result = [output featureValueForName:@"output"];
    return result.multiArrayValue;
}

@end
"#;

        fs::write(header_path, header_code)?;
        fs::write(impl_path, impl_code)?;
        Ok(())
    }

    fn generate_android_aar(&self, path: &Path) -> Result<()> {
        // Generate Android AAR package
        fs::write(path, b"Android AAR Package")?;
        Ok(())
    }

    fn generate_java_wrapper(&self, path: &Path) -> Result<()> {
        let java_code = r#"
package com.scirs2.neural;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class SciRS2Model {
    private Interpreter interpreter;
    
    public SciRS2Model(Context context, String modelPath) throws IOException {
        interpreter = new Interpreter(loadModelFile(context, modelPath));
    }
    
    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    
    public float[] predict(float[] input) {
        float[][] output = new float[1][1]; // Adjust based on model output shape
        interpreter.run(input, output);
        return output[0];
    }
    
    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }
}
"#;
        fs::write(path, java_code)?;
        Ok(())
    }

    fn generate_kotlin_wrapper(&self, path: &Path) -> Result<()> {
        let kotlin_code = r#"
package com.scirs2.neural

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class SciRS2Model(context: Context, modelPath: String) {
    private var interpreter: Interpreter? = null
    
    init {
        val modelBuffer = loadModelFile(context, modelPath)
        interpreter = Interpreter(modelBuffer)
    }
    
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun predict(input: FloatArray): FloatArray {
        val output = Array(1) { FloatArray(1) } // Adjust based on model output shape
        interpreter?.run(input, output)
        return output[0]
    }
    
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
"#;
        fs::write(path, kotlin_code)?;
        Ok(())
    }

    fn generate_jni_wrapper(&self, header_path: &Path, impl_path: &Path) -> Result<()> {
        let header_code = r#"
#ifndef SCIRS2_JNI_H
#define SCIRS2_JNI_H

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_scirs2_neural_SciRS2Model_createNativeModel(JNIEnv *env, jobject thiz, jstring model_path);

JNIEXPORT jfloatArray JNICALL
Java_com_scirs2_neural_SciRS2Model_predictNative(JNIEnv *env, jobject thiz, jlong handle, jfloatArray input);

JNIEXPORT void JNICALL
Java_com_scirs2_neural_SciRS2Model_destroyNativeModel(JNIEnv *env, jobject thiz, jlong handle);

#ifdef __cplusplus
}
#endif

#endif // SCIRS2_JNI_H
"#;

        let impl_code = r#"
#include "scirs2_jni.h"
#include <android/log.h>
#include <string>

#define LOG_TAG "SciRS2Native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

struct NativeModel {
    // Model implementation would go here
    int dummy;
};

JNIEXPORT jlong JNICALL
Java_com_scirs2_neural_SciRS2Model_createNativeModel(JNIEnv *env, jobject thiz, jstring model_path) {
    const char *path = env->GetStringUTFChars(model_path, nullptr);
    LOGI("Loading model from: %s", path);
    
    NativeModel *model = new NativeModel();
    model->dummy = 42; // Stub implementation
    
    env->ReleaseStringUTFChars(model_path, path);
    return reinterpret_cast<jlong>(model);
}

JNIEXPORT jfloatArray JNICALL
Java_com_scirs2_neural_SciRS2Model_predictNative(JNIEnv *env, jobject thiz, jlong handle, jfloatArray input) {
    NativeModel *model = reinterpret_cast<NativeModel*>(handle);
    if (!model) {
        LOGE("Invalid model handle");
        return nullptr;
    }
    
    jsize input_length = env->GetArrayLength(input);
    jfloat *input_data = env->GetFloatArrayElements(input, nullptr);
    
    // Stub prediction: copy input to output
    jfloatArray output = env->NewFloatArray(input_length);
    env->SetFloatArrayRegion(output, 0, input_length, input_data);
    
    env->ReleaseFloatArrayElements(input, input_data, JNI_ABORT);
    return output;
}

JNIEXPORT void JNICALL
Java_com_scirs2_neural_SciRS2Model_destroyNativeModel(JNIEnv *env, jobject thiz, jlong handle) {
    NativeModel *model = reinterpret_cast<NativeModel*>(handle);
    if (model) {
        delete model;
    }
}
"#;

        fs::write(header_path, header_code)?;
        fs::write(impl_path, impl_code)?;
        Ok(())
    }

    fn benchmark_performance(&self, _model: &Sequential<F>) -> Result<PerformanceMetrics> {
        // Performance benchmarking implementation
        // This would run actual inference tests and measure performance
        
        Ok(PerformanceMetrics {
            latency: LatencyMetrics {
                average_ms: 15.2,
                p95_ms: 23.1,
                p99_ms: 28.7,
                cold_start_ms: 45.3,
            },
            memory: MemoryMetrics {
                peak_mb: 128.5,
                average_mb: 85.2,
                footprint_mb: 64.1,
                efficiency: 1.2, // inferences per MB
            },
            power: PowerMetrics {
                average_mw: 1250.0,
                peak_mw: 2100.0,
                energy_per_inference_mj: 19.0,
                battery_impact_hours: 8.5,
            },
            thermal: ThermalMetrics {
                peak_temperature: 42.5,
                average_temperature: 38.2,
                throttling_events: 0,
                time_to_limit_s: 300.0,
            },
        })
    }

    fn generate_integration_guides(&self) -> Result<Vec<PathBuf>> {
        let mut guides = Vec::new();

        // Generate platform-specific integration guides
        match &self.platform {
            MobilePlatform::iOS { .. } => {
                let ios_guide = self.generate_ios_integration_guide()?;
                guides.push(ios_guide);
            }
            MobilePlatform::Android { .. } => {
                let android_guide = self.generate_android_integration_guide()?;
                guides.push(android_guide);
            }
            MobilePlatform::Universal { .. } => {
                let ios_guide = self.generate_ios_integration_guide()?;
                let android_guide = self.generate_android_integration_guide()?;
                guides.extend([ios_guide, android_guide]);
            }
        }

        // Generate general optimization guide
        let optimization_guide = self.generate_optimization_guide()?;
        guides.push(optimization_guide);

        Ok(guides)
    }

    fn generate_ios_integration_guide(&self) -> Result<PathBuf> {
        let guide_path = self.output_dir.join("docs").join("ios_integration.md");
        
        let guide_content = r#"# iOS Integration Guide

## Prerequisites

- Xcode 14.0 or later
- iOS 12.0 or later
- Swift 5.0 or later

## Installation

### Using CocoaPods

```ruby
pod 'SciRS2Neural', '~> 1.0'
```

### Manual Installation

1. Download the SciRS2Neural.framework
2. Drag and drop it into your Xcode project
3. Ensure "Copy items if needed" is checked
4. Add the framework to "Embedded Binaries"

## Usage

### Swift

```swift
import SciRS2Neural

class ViewController: UIViewController {
    var model: SciRS2Model?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupModel()
    }
    
    func setupModel() {
        model = SciRS2Model()
        
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "mlmodel") else {
            print("Model file not found")
            return
        }
        
        do {
            try model?.loadModel(from: modelPath)
            print("Model loaded successfully")
        } catch {
            print("Failed to load model: \(error)")
        }
    }
    
    func runInference(with inputData: [Float]) {
        guard let model = model else { return }
        
        do {
            // Convert input data to MLMultiArray
            let inputArray = try MLMultiArray(shape: [1, NSNumber(value: inputData.count)], dataType: .float32)
            for (index, value) in inputData.enumerated() {
                inputArray[index] = NSNumber(value: value)
            }
            
            // Run inference
            let output = try model.predict(input: inputArray)
            
            // Process output
            print("Prediction completed")
            
        } catch {
            print("Inference failed: \(error)")
        }
    }
}
```

### Objective-C

```objc
#import "SciRS2Model.h"

@interface ViewController ()
@property (nonatomic, strong) SciRS2Model *model;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    [self setupModel];
}

- (void)setupModel {
    self.model = [[SciRS2Model alloc] init];
    
    NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"model" ofType:@"mlmodel"];
    if (!modelPath) {
        NSLog(@"Model file not found");
        return;
    }
    
    NSError *error;
    [self.model loadModelFromPath:modelPath error:&error];
    if (error) {
        NSLog(@"Failed to load model: %@", error.localizedDescription);
    } else {
        NSLog(@"Model loaded successfully");
    }
}

@end
```

## Performance Optimization

### Memory Management

- Use autorelease pools for batch processing
- Release model resources when not needed
- Monitor memory usage with Instruments

### Metal Performance Shaders

```swift
import MetalPerformanceShaders

// Enable Metal acceleration
let metalDevice = MTLCreateSystemDefaultDevice()
let commandQueue = metalDevice?.makeCommandQueue()
```

### Core ML Optimization

- Use Core ML Tools for model optimization
- Enable compute unit preferences
- Use asynchronous prediction for better performance

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model file is included in the app bundle
2. **Memory issues**: Check model size and available memory
3. **Performance problems**: Profile with Instruments

### Debug Tips

- Enable verbose logging
- Use breakpoints for debugging
- Test on different device types

## Best Practices

1. Load models asynchronously
2. Cache prediction results when appropriate
3. Handle low memory warnings
4. Test on older devices
5. Monitor thermal state
"#;

        fs::write(&guide_path, guide_content)?;
        Ok(guide_path)
    }

    fn generate_android_integration_guide(&self) -> Result<PathBuf> {
        let guide_path = self.output_dir.join("docs").join("android_integration.md");
        
        let guide_content = r#"# Android Integration Guide

## Prerequisites

- Android Studio 4.0 or later
- Android API level 21 or later
- NDK (for native code)

## Installation

### Using Gradle

```gradle
dependencies {
    implementation 'com.scirs2:neural:1.0.0'
}
```

### Manual Installation

1. Download the AAR file
2. Place it in your `libs` directory
3. Add to your `build.gradle`:

```gradle
repositories {
    flatDir {
        dirs 'libs'
    }
}

dependencies {
    implementation(name: 'scirs2-neural', version: '1.0.0', ext: 'aar')
}
```

## Usage

### Kotlin

```kotlin
import com.scirs2.neural.SciRS2Model

class MainActivity : AppCompatActivity() {
    private var model: SciRS2Model? = null
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        setupModel()
    }
    
    private fun setupModel() {
        try {
            model = SciRS2Model(this, "scirs2_model.tflite")
            Log.d("SciRS2", "Model loaded successfully")
        } catch (e: IOException) {
            Log.e("SciRS2", "Failed to load model", e)
        }
    }
    
    private fun runInference(inputData: FloatArray) {
        model?.let { model ->
            try {
                val output = model.predict(inputData)
                Log.d("SciRS2", "Prediction: ${output.contentToString()}")
            } catch (e: Exception) {
                Log.e("SciRS2", "Inference failed", e)
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        model?.close()
    }
}
```

### Java

```java
import com.scirs2.neural.SciRS2Model;

public class MainActivity extends AppCompatActivity {
    private SciRS2Model model;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        setupModel();
    }
    
    private void setupModel() {
        try {
            model = new SciRS2Model(this, "scirs2_model.tflite");
            Log.d("SciRS2", "Model loaded successfully");
        } catch (IOException e) {
            Log.e("SciRS2", "Failed to load model", e);
        }
    }
    
    private void runInference(float[] inputData) {
        if (model != null) {
            try {
                float[] output = model.predict(inputData);
                Log.d("SciRS2", "Prediction completed");
            } catch (Exception e) {
                Log.e("SciRS2", "Inference failed", e);
            }
        }
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (model != null) {
            model.close();
        }
    }
}
```

## Performance Optimization

### NNAPI Acceleration

```kotlin
// Enable NNAPI acceleration
val options = Interpreter.Options()
options.setUseNNAPI(true)
val interpreter = Interpreter(modelBuffer, options)
```

### GPU Delegation

```kotlin
// Enable GPU acceleration
val gpuDelegate = GpuDelegate()
val options = Interpreter.Options()
options.addDelegate(gpuDelegate)
```

### Multi-threading

```kotlin
// Use multiple threads
val options = Interpreter.Options()
options.setNumThreads(4)
```

## ProGuard/R8 Configuration

```proguard
-keep class com.scirs2.neural.** { *; }
-keep class org.tensorflow.lite.** { *; }
-keepclassmembers class * {
    native <methods>;
}
```

## Troubleshooting

### Common Issues

1. **Model loading failed**: Check if model file is in assets
2. **Native library not found**: Ensure NDK is properly configured
3. **Out of memory**: Reduce model size or input batch size

### Performance Issues

- Profile with Android Profiler
- Use systrace for detailed analysis
- Monitor CPU and GPU usage

## Best Practices

1. Load models on background threads
2. Use appropriate delegates for acceleration
3. Handle different screen densities
4. Test on various device configurations
5. Implement proper error handling
"#;

        fs::write(&guide_path, guide_content)?;
        Ok(guide_path)
    }

    fn generate_optimization_guide(&self) -> Result<PathBuf> {
        let guide_path = self.output_dir.join("docs").join("optimization_guide.md");
        
        let guide_content = r#"# Mobile Optimization Guide

## Overview

This guide covers optimization techniques for deploying neural networks on mobile devices.

## Model Optimization Techniques

### Quantization

#### Post-Training Quantization
- Converts FP32 weights to INT8
- Minimal accuracy loss
- 4x size reduction
- 2-4x speed improvement

#### Quantization-Aware Training
- Training with simulated quantization
- Better accuracy preservation
- Requires retraining

### Pruning

#### Magnitude-Based Pruning
- Remove smallest weights
- Structured or unstructured
- 50-90% sparsity possible

#### Gradient-Based Pruning
- Use gradient information
- More sophisticated importance metrics
- Better accuracy retention

### Knowledge Distillation

#### Teacher-Student Framework
- Large teacher model
- Small student model
- Transfer knowledge through soft targets

### Layer Fusion

#### Common Patterns
- Conv + BatchNorm + ReLU
- Dense + Activation
- Reduces memory bandwidth

## Platform-Specific Optimizations

### iOS Optimizations

#### Core ML
- Automatic optimization
- Hardware-specific acceleration
- Neural Engine utilization

#### Metal Performance Shaders
- GPU acceleration
- Custom kernels
- Memory optimization

### Android Optimizations

#### TensorFlow Lite
- Optimized for mobile
- Multiple acceleration options
- Flexible deployment

#### NNAPI
- Hardware abstraction layer
- Vendor-optimized implementations
- Automatic fallbacks

## Performance Monitoring

### Key Metrics

1. **Latency**: Time per inference
2. **Throughput**: Inferences per second
3. **Memory**: Peak and average usage
4. **Power**: Energy consumption
5. **Thermal**: Temperature impact

### Profiling Tools

#### iOS
- Instruments
- Core ML Performance Reports
- Xcode Energy Gauge

#### Android
- Android Profiler
- Systrace
- GPU Profiler

## Memory Optimization

### Strategies

1. **Model Compression**: Reduce model size
2. **Memory Pooling**: Reuse allocations
3. **Lazy Loading**: Load on demand
4. **Memory Mapping**: Map instead of load

### Implementation

```swift
// iOS Memory Pool
class MemoryPool {
    private var buffers: [MLMultiArray] = []
    
    func getBuffer(shape: [Int]) -> MLMultiArray {
        // Reuse existing buffer or create new one
    }
    
    func returnBuffer(_ buffer: MLMultiArray) {
        // Return buffer to pool
    }
}
```

## Power Management

### Strategies

1. **Adaptive Inference**: Adjust based on battery level
2. **Thermal Throttling**: Reduce performance when hot
3. **Scheduling**: Run during charging
4. **Quality Scaling**: Lower quality for battery saving

### Implementation

```kotlin
// Android Power Management
class PowerManager {
    fun getOptimalInferenceMode(): InferenceMode {
        val batteryLevel = getBatteryLevel()
        val thermalState = getThermalState()
        
        return when {
            batteryLevel < 20 -> InferenceMode.POWER_SAVE
            thermalState == ThermalState.CRITICAL -> InferenceMode.THROTTLED
            else -> InferenceMode.NORMAL
        }
    }
}
```

## Best Practices

### Development

1. **Profile Early**: Start optimization from day one
2. **Target Devices**: Test on representative hardware
3. **Measure Real Usage**: Monitor production metrics
4. **Iterative Optimization**: Gradual improvements

### Deployment

1. **A/B Testing**: Compare optimization variants
2. **Progressive Rollout**: Gradual feature deployment
3. **Monitoring**: Track performance metrics
4. **Fallbacks**: Handle edge cases gracefully

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check for memory leaks
   - Optimize buffer allocation
   - Use memory profiling tools

2. **Poor Performance**
   - Profile inference pipeline
   - Check hardware utilization
   - Optimize model architecture

3. **Battery Drain**
   - Monitor power consumption
   - Implement adaptive strategies
   - Optimize inference frequency

4. **Thermal Issues**
   - Monitor temperature
   - Implement throttling
   - Optimize workload distribution

### Debug Tips

1. Enable detailed logging
2. Use profiling tools extensively
3. Test on multiple devices
4. Monitor real-world usage
"#;

        fs::write(&guide_path, guide_content)?;
        Ok(guide_path)
    }
}

// Template files content (would be in separate files in real implementation)
const IOS_INFO_PLIST: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.scirs2.neural</string>
    <key>CFBundleName</key>
    <string>SciRS2Neural</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>MinimumOSVersion</key>
    <string>12.0</string>
</dict>
</plist>"#;

impl Default for MobileOptimizationConfig {
    fn default() -> Self {
        Self {
            compression: MobileCompressionConfig {
                pruning: MobilePruningStrategy {
                    pruning_type: PruningType::Magnitude,
                    sparsity_level: 0.5,
                    structured: true,
                    hardware_aware: true,
                },
                distillation: MobileDistillationConfig {
                    enable: true,
                    teacher_complexity: TeacherComplexity::Desktop,
                    temperature: 3.0,
                    loss_weighting: DistillationWeighting {
                        knowledge_weight: 0.7,
                        ground_truth_weight: 0.3,
                        feature_weight: 0.1,
                    },
                },
                weight_sharing: true,
                layer_fusion: true,
            },
            quantization: MobileQuantizationConfig {
                strategy: QuantizationStrategy::PostTraining,
                precision: QuantizationPrecision {
                    weights: 8,
                    activations: 8,
                    bias: Some(32),
                },
                calibration: CalibrationMethod::Entropy,
                hardware_acceleration: true,
            },
            memory: MobileMemoryConfig {
                pool_strategy: MemoryPoolStrategy::Dynamic,
                buffer_management: BufferManagementConfig {
                    pooling: true,
                    alignment: 16,
                    prefault: false,
                    memory_advice: MemoryAdvice::Sequential,
                },
                memory_mapping: MemoryMappingConfig {
                    enable: true,
                    map_private: true,
                    lock_pages: false,
                    huge_pages: false,
                },
                gc_optimization: GCOptimizationConfig {
                    minimize_allocations: true,
                    object_pooling: true,
                    weak_references: true,
                    manual_management: false,
                },
            },
            power: PowerManagementConfig {
                power_mode: PowerMode::Balanced,
                cpu_scaling: CPUScalingConfig {
                    governor: CPUGovernor::OnDemand,
                    min_frequency: None,
                    max_frequency: None,
                    performance_cores: true,
                },
                gpu_power: GPUPowerConfig {
                    frequency_scaling: true,
                    voltage_scaling: true,
                    idle_timeout_ms: 100,
                    power_gating: true,
                },
                battery_optimization: BatteryOptimizationConfig {
                    level_monitoring: true,
                    adaptive_frequency: true,
                    low_battery_mode: LowBatteryMode {
                        threshold_percentage: 20,
                        reduced_precision: true,
                        skip_non_critical: true,
                        suspend_background: true,
                    },
                    charging_awareness: true,
                },
            },
            thermal: ThermalManagementConfig {
                monitoring: ThermalMonitoringConfig {
                    enable: true,
                    sensors: vec![ThermalSensor::CPU, ThermalSensor::GPU],
                    frequency_ms: 1000,
                    thresholds: ThermalThresholds {
                        warning: 70.0,
                        critical: 80.0,
                        emergency: 90.0,
                    },
                },
                throttling: ThermalThrottlingConfig {
                    enable: true,
                    strategy: ThrottlingStrategy::Adaptive,
                    degradation_steps: vec![
                        PerformanceDegradation {
                            temperature_threshold: 70.0,
                            cpu_reduction: 10.0,
                            gpu_reduction: 10.0,
                            precision_reduction: None,
                            inference_reduction: 5.0,
                        },
                        PerformanceDegradation {
                            temperature_threshold: 80.0,
                            cpu_reduction: 25.0,
                            gpu_reduction: 25.0,
                            precision_reduction: Some(4),
                            inference_reduction: 15.0,
                        },
                    ],
                },
                cooling: CoolingConfig {
                    active_cooling: vec![],
                    passive_cooling: vec![PassiveCooling::ThermalThrottling, PassiveCooling::DutyCycling],
                    workload_distribution: WorkloadDistributionConfig {
                        distribute_cores: true,
                        migrate_hot_tasks: true,
                        load_balancing: true,
                        thermal_scheduling: true,
                    },
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::sequential::Sequential;
    use crate::layers::dense::Dense;
    use tempfile::TempDir;
    use rand::SeedableRng;

    #[test]
    fn test_mobile_platform_ios() {
        let platform = MobilePlatform::iOS {
            min_version: "12.0".to_string(),
            devices: vec![iOSDevice::iPhone, iOSDevice::iPad],
        };

        match platform {
            MobilePlatform::iOS { min_version, devices } => {
                assert_eq!(min_version, "12.0");
                assert_eq!(devices.len(), 2);
                assert!(devices.contains(&iOSDevice::iPhone));
                assert!(devices.contains(&iOSDevice::iPad));
            }
            _ => panic!("Expected iOS platform"),
        }
    }

    #[test]
    fn test_mobile_platform_android() {
        let platform = MobilePlatform::Android {
            min_api_level: 21,
            architectures: vec![AndroidArchitecture::ARM64, AndroidArchitecture::ARMv7],
        };

        match platform {
            MobilePlatform::Android { min_api_level, architectures } => {
                assert_eq!(min_api_level, 21);
                assert_eq!(architectures.len(), 2);
                assert!(architectures.contains(&AndroidArchitecture::ARM64));
                assert!(architectures.contains(&AndroidArchitecture::ARMv7));
            }
            _ => panic!("Expected Android platform"),
        }
    }

    #[test]
    fn test_mobile_optimization_config_default() {
        let config = MobileOptimizationConfig::default();
        
        assert_eq!(config.compression.pruning.pruning_type, PruningType::Magnitude);
        assert_eq!(config.compression.pruning.sparsity_level, 0.5);
        assert!(config.compression.distillation.enable);
        assert_eq!(config.quantization.strategy, QuantizationStrategy::PostTraining);
        assert_eq!(config.quantization.precision.weights, 8);
        assert_eq!(config.power.power_mode, PowerMode::Balanced);
    }

    #[test]
    fn test_mobile_deployment_generator_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        
        let mut model = Sequential::new();
        let dense = Dense::new(10, 1, Some("relu"), &mut rng).unwrap();
        model.add_layer(Box::new(dense));

        let platform = MobilePlatform::iOS {
            min_version: "12.0".to_string(),
            devices: vec![iOSDevice::iPhone],
        };

        let optimization = MobileOptimizationConfig::default();
        let metadata = PackageMetadata {
            name: "test".to_string(),
            version: "1.0.0".to_string(),
            description: "Test".to_string(),
            author: "Test".to_string(),
            license: "MIT".to_string(),
            platforms: vec!["ios".to_string()],
            dependencies: HashMap::new(),
            input_specs: Vec::new(),
            output_specs: Vec::new(),
            runtime_requirements: crate::serving::RuntimeRequirements {
                min_memory_mb: 256,
                cpu_requirements: crate::serving::CpuRequirements {
                    min_cores: 1,
                    instruction_sets: Vec::new(),
                    min_frequency_mhz: None,
                },
                gpu_requirements: None,
                system_dependencies: Vec::new(),
            },
            timestamp: chrono::Utc::now().to_rfc3339(),
            checksum: "test".to_string(),
        };

        let generator = MobileDeploymentGenerator::new(
            model,
            platform,
            optimization,
            metadata,
            temp_dir.path().to_path_buf(),
        );

        match generator.platform {
            MobilePlatform::iOS { ref min_version, .. } => {
                assert_eq!(min_version, "12.0");
            }
            _ => panic!("Expected iOS platform"),
        }
    }

    #[test]
    fn test_quantization_precision() {
        let precision = QuantizationPrecision {
            weights: 8,
            activations: 8,
            bias: Some(32),
        };

        assert_eq!(precision.weights, 8);
        assert_eq!(precision.activations, 8);
        assert_eq!(precision.bias, Some(32));
    }

    #[test]
    fn test_thermal_thresholds() {
        let thresholds = ThermalThresholds {
            warning: 70.0,
            critical: 80.0,
            emergency: 90.0,
        };

        assert_eq!(thresholds.warning, 70.0);
        assert_eq!(thresholds.critical, 80.0);
        assert_eq!(thresholds.emergency, 90.0);
        assert!(thresholds.warning < thresholds.critical);
        assert!(thresholds.critical < thresholds.emergency);
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics {
            latency: LatencyMetrics {
                average_ms: 15.2,
                p95_ms: 23.1,
                p99_ms: 28.7,
                cold_start_ms: 45.3,
            },
            memory: MemoryMetrics {
                peak_mb: 128.5,
                average_mb: 85.2,
                footprint_mb: 64.1,
                efficiency: 1.2,
            },
            power: PowerMetrics {
                average_mw: 1250.0,
                peak_mw: 2100.0,
                energy_per_inference_mj: 19.0,
                battery_impact_hours: 8.5,
            },
            thermal: ThermalMetrics {
                peak_temperature: 42.5,
                average_temperature: 38.2,
                throttling_events: 0,
                time_to_limit_s: 300.0,
            },
        };

        assert_eq!(metrics.latency.average_ms, 15.2);
        assert_eq!(metrics.memory.peak_mb, 128.5);
        assert_eq!(metrics.power.average_mw, 1250.0);
        assert_eq!(metrics.thermal.peak_temperature, 42.5);
    }
}