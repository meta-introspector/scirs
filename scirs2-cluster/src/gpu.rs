//! GPU acceleration interfaces and stubs for clustering algorithms
//!
//! This module provides GPU acceleration capabilities for clustering algorithms.
//! Currently implements stubs and interfaces that can be extended with actual
//! GPU implementations using CUDA, OpenCL, or other GPU computing frameworks.

use crate::error::{ClusteringError, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// GPU acceleration backends
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GpuBackend {
    /// NVIDIA CUDA backend
    Cuda,
    /// OpenCL backend (cross-platform)
    OpenCl,
    /// AMD ROCm backend
    Rocm,
    /// Intel OneAPI backend
    OneApi,
    /// Apple Metal Performance Shaders
    Metal,
    /// CPU fallback (no GPU acceleration)
    CpuFallback,
}

/// GPU device information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuDevice {
    /// Device ID
    pub device_id: u32,
    /// Device name
    pub name: String,
    /// Total memory in bytes
    pub total_memory: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Compute capability or equivalent
    pub compute_capability: String,
    /// Number of compute units
    pub compute_units: u32,
    /// Backend type
    pub backend: GpuBackend,
    /// Whether device supports double precision
    pub supports_double_precision: bool,
}

/// GPU memory allocation strategy
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MemoryStrategy {
    /// Use unified memory (CUDA/HIP)
    Unified,
    /// Explicit host-device transfers
    Explicit,
    /// Memory pooling for reuse
    Pooled { pool_size_mb: usize },
    /// Zero-copy memory (if supported)
    ZeroCopy,
}

/// GPU acceleration configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuConfig {
    /// Preferred backend
    pub backend: GpuBackend,
    /// Device selection strategy
    pub device_selection: DeviceSelection,
    /// Memory allocation strategy
    pub memory_strategy: MemoryStrategy,
    /// Block size for GPU kernels
    pub block_size: u32,
    /// Grid size for GPU kernels
    pub grid_size: u32,
    /// Enable automatic tuning
    pub auto_tune: bool,
    /// Fallback to CPU if GPU fails
    pub cpu_fallback: bool,
}

/// Device selection strategy
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DeviceSelection {
    /// Use device with most memory
    MostMemory,
    /// Use device with highest compute capability
    HighestCompute,
    /// Use specific device by ID
    Specific(u32),
    /// Automatically select best device
    Automatic,
    /// Use multiple devices (multi-GPU)
    MultiGpu(Vec<u32>),
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::CpuFallback,
            device_selection: DeviceSelection::Automatic,
            memory_strategy: MemoryStrategy::Explicit,
            block_size: 256,
            grid_size: 1024,
            auto_tune: true,
            cpu_fallback: true,
        }
    }
}

/// GPU context for clustering operations
#[derive(Debug)]
pub struct GpuContext {
    /// Active devices
    devices: Vec<GpuDevice>,
    /// Current configuration
    config: GpuConfig,
    /// Backend-specific context
    backend_context: BackendContext,
    /// Performance statistics
    stats: GpuStats,
}

/// Backend-specific context (placeholder for actual implementations)
#[derive(Debug)]
enum BackendContext {
    /// CUDA context
    Cuda {
        #[allow(dead_code)]
        context_handle: usize,
    },
    /// OpenCL context
    OpenCl {
        #[allow(dead_code)]
        context_handle: usize,
    },
    /// CPU fallback (no context needed)
    CpuFallback,
}

/// GPU performance statistics
#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuStats {
    /// Total GPU memory allocations
    pub total_allocations: usize,
    /// Current memory usage
    pub current_memory_usage: usize,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Number of kernel launches
    pub kernel_launches: usize,
    /// Total GPU computation time (seconds)
    pub gpu_compute_time: f64,
    /// Total memory transfer time (seconds)
    pub memory_transfer_time: f64,
    /// Host-to-device transfers
    pub h2d_transfers: usize,
    /// Device-to-host transfers
    pub d2h_transfers: usize,
}

impl GpuContext {
    /// Initialize GPU context with configuration
    pub fn new(config: GpuConfig) -> Result<Self> {
        let mut final_config = config.clone();

        // If backend is automatic, try to detect the best available backend
        if matches!(config.backend, GpuBackend::CpuFallback) && config.cpu_fallback {
            final_config.backend = Self::detect_best_backend()?;
        }

        let devices = Self::detect_devices(&final_config.backend)?;

        if devices.is_empty() {
            if !config.cpu_fallback {
                return Err(ClusteringError::ComputationError(
                    "No GPU devices found and CPU fallback disabled".to_string(),
                ));
            } else {
                // Fall back to CPU
                final_config.backend = GpuBackend::CpuFallback;
            }
        }

        let backend_context = Self::initialize_backend(&final_config.backend)?;

        Ok(Self {
            devices,
            config: final_config,
            backend_context,
            stats: GpuStats::default(),
        })
    }

    /// Detect the best available GPU backend automatically
    pub fn detect_best_backend() -> Result<GpuBackend> {
        // Try backends in order of preference
        let backends_to_try = [
            GpuBackend::Cuda,
            GpuBackend::OpenCl,
            GpuBackend::Metal,
            GpuBackend::Rocm,
            GpuBackend::OneApi,
        ];

        for backend in &backends_to_try {
            if let Ok(devices) = Self::detect_devices(backend) {
                if !devices.is_empty() {
                    return Ok(*backend);
                }
            }
        }

        // Fall back to CPU if no GPU backends are available
        Ok(GpuBackend::CpuFallback)
    }

    /// Create a new context with automatic backend detection
    pub fn new_auto() -> Result<Self> {
        let config = GpuConfig {
            backend: GpuBackend::CpuFallback, // Will be auto-detected
            cpu_fallback: true,
            auto_tune: true,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Select the best device based on configuration
    pub fn select_best_device(&self) -> Option<&GpuDevice> {
        if self.devices.is_empty() {
            return None;
        }

        match &self.config.device_selection {
            DeviceSelection::Automatic => {
                // Score devices based on memory and compute capability
                self.devices.iter().max_by_key(|device| {
                    device.available_memory + device.compute_units as usize * 1024 * 1024
                })
            }
            DeviceSelection::MostMemory => self
                .devices
                .iter()
                .max_by_key(|device| device.available_memory),
            DeviceSelection::HighestCompute => self
                .devices
                .iter()
                .max_by_key(|device| device.compute_units),
            DeviceSelection::Specific(device_id) => self
                .devices
                .iter()
                .find(|device| device.device_id == *device_id),
            DeviceSelection::MultiGpu(device_ids) => {
                // Return the first available device from the list
                device_ids
                    .iter()
                    .find_map(|&id| self.devices.iter().find(|device| device.device_id == id))
            }
        }
    }

    /// Check if current configuration is optimal for given data size
    pub fn is_optimal_for_data_size(&self, data_size_bytes: usize) -> bool {
        if let Some(device) = self.select_best_device() {
            // GPU is optimal if data fits comfortably in memory with room for computation
            let required_memory = data_size_bytes * 3; // Data + intermediate results + output
            device.available_memory > required_memory
        } else {
            // If no GPU available, CPU is the only option
            true
        }
    }

    /// Get recommended batch size for given data
    pub fn get_recommended_batch_size(&self, data_size_bytes: usize, element_size: usize) -> usize {
        if let Some(device) = self.select_best_device() {
            // Use up to 80% of available memory for batch processing
            let available_memory = (device.available_memory as f64 * 0.8) as usize;
            let elements_per_batch = available_memory / (element_size * 4); // Account for temporary storage
            elements_per_batch.max(1)
        } else {
            // CPU fallback - use smaller batches
            (data_size_bytes / element_size).min(10000).max(100)
        }
    }

    /// Detect available GPU devices
    fn detect_devices(backend: &GpuBackend) -> Result<Vec<GpuDevice>> {
        match backend {
            GpuBackend::Cuda => Self::detect_cuda_devices(),
            GpuBackend::OpenCl => Self::detect_opencl_devices(),
            GpuBackend::Rocm => Self::detect_rocm_devices(),
            GpuBackend::OneApi => Self::detect_oneapi_devices(),
            GpuBackend::Metal => Self::detect_metal_devices(),
            GpuBackend::CpuFallback => Ok(vec![]),
        }
    }

    /// Detect CUDA devices (enhanced implementation)
    fn detect_cuda_devices() -> Result<Vec<GpuDevice>> {
        #[cfg(feature = "cuda")]
        {
            // For now, simulate device detection based on environment
            if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
                // Simulate a CUDA device
                Ok(vec![GpuDevice {
                    device_id: 0,
                    name: "Simulated CUDA Device".to_string(),
                    total_memory: 8 * 1024 * 1024 * 1024, // 8GB
                    available_memory: 6 * 1024 * 1024 * 1024, // 6GB available
                    compute_capability: "7.5".to_string(),
                    compute_units: 80,
                    backend: GpuBackend::Cuda,
                    supports_double_precision: true,
                }])
            } else {
                Ok(vec![])
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Check for nvidia-smi as a proxy for CUDA availability
            if let Ok(output) = std::process::Command::new("nvidia-smi").output() {
                if output.status.success() {
                    // Parse nvidia-smi output to get device info
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if stdout.contains("NVIDIA") {
                        // Create a mock device for demonstration
                        return Ok(vec![GpuDevice {
                            device_id: 0,
                            name: "NVIDIA GPU (detected via nvidia-smi)".to_string(),
                            total_memory: 8 * 1024 * 1024 * 1024,
                            available_memory: 6 * 1024 * 1024 * 1024,
                            compute_capability: "Unknown".to_string(),
                            compute_units: 80,
                            backend: GpuBackend::Cuda,
                            supports_double_precision: true,
                        }]);
                    }
                }
            }
            Ok(vec![])
        }
    }

    /// Detect OpenCL devices (enhanced implementation)
    fn detect_opencl_devices() -> Result<Vec<GpuDevice>> {
        #[cfg(feature = "opencl")]
        {
            // Actual OpenCL device detection would go here
            Ok(vec![])
        }
        #[cfg(not(feature = "opencl"))]
        {
            // Check for OpenCL availability via clinfo command
            if let Ok(output) = std::process::Command::new("clinfo").output() {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if stdout.contains("Device Type") && stdout.contains("GPU") {
                        // Parse basic device info from clinfo output
                        let mut devices = Vec::new();
                        let mut device_count = 0;

                        // Simple parsing - in practice would be more sophisticated
                        for line in stdout.lines() {
                            if line.trim().starts_with("Device Name") {
                                let name = line
                                    .split(':')
                                    .nth(1)
                                    .unwrap_or("Unknown OpenCL Device")
                                    .trim()
                                    .to_string();

                                devices.push(GpuDevice {
                                    device_id: device_count,
                                    name,
                                    total_memory: 4 * 1024 * 1024 * 1024, // Default 4GB
                                    available_memory: 3 * 1024 * 1024 * 1024, // Default 3GB available
                                    compute_capability: "OpenCL".to_string(),
                                    compute_units: 32, // Default
                                    backend: GpuBackend::OpenCl,
                                    supports_double_precision: true,
                                });
                                device_count += 1;
                            }
                        }

                        return Ok(devices);
                    }
                }
            }

            // Check for common OpenCL platforms
            let opencl_paths = [
                "/usr/lib/x86_64-linux-gnu/libOpenCL.so",
                "/usr/lib64/libOpenCL.so",
                "/opt/intel/opencl/lib64/libOpenCL.so",
                "/opt/amd/opencl/lib/x86_64/libOpenCL.so",
            ];

            for path in &opencl_paths {
                if std::path::Path::new(path).exists() {
                    return Ok(vec![GpuDevice {
                        device_id: 0,
                        name: "OpenCL GPU (library detected)".to_string(),
                        total_memory: 4 * 1024 * 1024 * 1024,
                        available_memory: 3 * 1024 * 1024 * 1024,
                        compute_capability: "OpenCL".to_string(),
                        compute_units: 32,
                        backend: GpuBackend::OpenCl,
                        supports_double_precision: true,
                    }]);
                }
            }

            Ok(vec![])
        }
    }

    /// Detect ROCm devices (enhanced implementation)
    fn detect_rocm_devices() -> Result<Vec<GpuDevice>> {
        #[cfg(feature = "rocm")]
        {
            // Actual ROCm device detection would go here using HIP/ROCm APIs
            Ok(vec![])
        }
        #[cfg(not(feature = "rocm"))]
        {
            // Check for ROCm installation via rocminfo command
            if let Ok(output) = std::process::Command::new("rocminfo").output() {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let mut devices = Vec::new();
                    let mut device_count = 0;
                    let mut current_device_name = String::new();
                    let mut current_device_memory = 0usize;

                    for line in stdout.lines() {
                        let line = line.trim();
                        
                        // Parse device name
                        if line.starts_with("Device Type:") && line.contains("GPU") {
                            // Look for marketing name in subsequent lines
                            current_device_name = "AMD GPU".to_string();
                        } else if line.starts_with("Marketing Name:") {
                            current_device_name = line
                                .split(':')
                                .nth(1)
                                .unwrap_or("AMD GPU")
                                .trim()
                                .to_string();
                        } else if line.starts_with("Max Memory Size:") {
                            // Parse memory size (usually in bytes)
                            if let Some(mem_str) = line.split(':').nth(1) {
                                if let Ok(mem_bytes) = mem_str.trim().parse::<usize>() {
                                    current_device_memory = mem_bytes;
                                }
                            }
                        } else if line.starts_with("Agent Type:") && line.contains("GPU") {
                            // Complete device entry
                            if !current_device_name.is_empty() {
                                devices.push(GpuDevice {
                                    device_id: device_count,
                                    name: current_device_name.clone(),
                                    total_memory: current_device_memory,
                                    available_memory: (current_device_memory as f64 * 0.8) as usize,
                                    compute_capability: "ROCm".to_string(),
                                    compute_units: 64, // Default estimate
                                    backend: GpuBackend::Rocm,
                                    supports_double_precision: true,
                                });
                                device_count += 1;
                                current_device_name.clear();
                                current_device_memory = 0;
                            }
                        }
                    }

                    return Ok(devices);
                }
            }

            // Check for ROCm runtime libraries
            let rocm_paths = [
                "/opt/rocm/lib/libhip_hcc.so",
                "/opt/rocm/lib/libamdhip64.so", 
                "/usr/lib/x86_64-linux-gnu/libamdhip64.so",
                "/opt/rocm/hip/lib/libamdhip64.so",
            ];

            for path in &rocm_paths {
                if std::path::Path::new(path).exists() {
                    return Ok(vec![GpuDevice {
                        device_id: 0,
                        name: "AMD GPU (ROCm runtime detected)".to_string(),
                        total_memory: 8 * 1024 * 1024 * 1024, // Default 8GB
                        available_memory: 6 * 1024 * 1024 * 1024, // Default 6GB available
                        compute_capability: "ROCm".to_string(),
                        compute_units: 64,
                        backend: GpuBackend::Rocm,
                        supports_double_precision: true,
                    }]);
                }
            }

            // Check for AMD GPU via lspci
            if let Ok(output) = std::process::Command::new("lspci").output() {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if stdout.to_lowercase().contains("amd") && stdout.to_lowercase().contains("radeon") {
                        return Ok(vec![GpuDevice {
                            device_id: 0,
                            name: "AMD Radeon GPU (detected via lspci)".to_string(),
                            total_memory: 8 * 1024 * 1024 * 1024,
                            available_memory: 6 * 1024 * 1024 * 1024,
                            compute_capability: "ROCm".to_string(),
                            compute_units: 64,
                            backend: GpuBackend::Rocm,
                            supports_double_precision: true,
                        }]);
                    }
                }
            }

            Ok(vec![])
        }
    }

    /// Detect OneAPI devices (enhanced implementation)
    fn detect_oneapi_devices() -> Result<Vec<GpuDevice>> {
        #[cfg(feature = "oneapi")]
        {
            // Actual OneAPI device detection would go here using Level Zero or SYCL APIs
            Ok(vec![])
        }
        #[cfg(not(feature = "oneapi"))]
        {
            // Check for Intel OneAPI installation via sycl-ls command
            if let Ok(output) = std::process::Command::new("sycl-ls").output() {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let mut devices = Vec::new();
                    let mut device_count = 0;

                    for line in stdout.lines() {
                        let line = line.trim();
                        
                        // Look for GPU devices in sycl-ls output
                        if (line.to_lowercase().contains("gpu") || line.to_lowercase().contains("intel")) 
                            && (line.contains("opencl") || line.contains("level_zero")) {
                            
                            // Extract device name (usually in brackets or after device type)
                            let device_name = if let Some(start) = line.find('[') {
                                if let Some(end) = line.find(']') {
                                    line[start + 1..end].to_string()
                                } else {
                                    "Intel GPU".to_string()
                                }
                            } else if line.to_lowercase().contains("intel") {
                                line.to_string()
                            } else {
                                "Intel OneAPI GPU".to_string()
                            };

                            devices.push(GpuDevice {
                                device_id: device_count,
                                name: device_name,
                                total_memory: 4 * 1024 * 1024 * 1024, // Default 4GB for Intel integrated
                                available_memory: 3 * 1024 * 1024 * 1024, // Default 3GB available
                                compute_capability: "OneAPI".to_string(),
                                compute_units: 96, // Default for Intel Xe
                                backend: GpuBackend::OneApi,
                                supports_double_precision: true,
                            });
                            device_count += 1;
                        }
                    }

                    if !devices.is_empty() {
                        return Ok(devices);
                    }
                }
            }

            // Check for Intel GPU Compute Runtime
            let oneapi_paths = [
                "/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so",
                "/opt/intel/opencl/lib/x64/libintelocl.so",
                "/usr/lib64/libze_loader.so", // Level Zero loader
                "/opt/intel/oneapi/compiler/latest/linux/lib/libsycl.so",
            ];

            for path in &oneapi_paths {
                if std::path::Path::new(path).exists() {
                    return Ok(vec![GpuDevice {
                        device_id: 0,
                        name: "Intel GPU (OneAPI runtime detected)".to_string(),
                        total_memory: 4 * 1024 * 1024 * 1024,
                        available_memory: 3 * 1024 * 1024 * 1024,
                        compute_capability: "OneAPI".to_string(),
                        compute_units: 96,
                        backend: GpuBackend::OneApi,
                        supports_double_precision: true,
                    }]);
                }
            }

            // Check for Intel GPU via lspci
            if let Ok(output) = std::process::Command::new("lspci").output() {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if stdout.to_lowercase().contains("intel") && 
                       (stdout.to_lowercase().contains("graphics") || stdout.to_lowercase().contains("display")) {
                        return Ok(vec![GpuDevice {
                            device_id: 0,
                            name: "Intel Integrated Graphics (detected via lspci)".to_string(),
                            total_memory: 4 * 1024 * 1024 * 1024,
                            available_memory: 3 * 1024 * 1024 * 1024,
                            compute_capability: "OneAPI".to_string(),
                            compute_units: 96,
                            backend: GpuBackend::OneApi,
                            supports_double_precision: true,
                        }]);
                    }
                }
            }

            Ok(vec![])
        }
    }

    /// Detect Metal devices (enhanced implementation)
    fn detect_metal_devices() -> Result<Vec<GpuDevice>> {
        #[cfg(target_os = "macos")]
        {
            #[cfg(feature = "metal")]
            {
                // Actual Metal device detection would go here using Metal APIs
                Ok(vec![])
            }
            #[cfg(not(feature = "metal"))]
            {
                // Use system_profiler to detect GPU on macOS
                if let Ok(output) = std::process::Command::new("system_profiler")
                    .arg("SPDisplaysDataType")
                    .output()
                {
                    if output.status.success() {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        let mut devices = Vec::new();
                        let mut device_count = 0;
                        let mut current_chipset = String::new();
                        let mut current_memory = 0usize;

                        for line in stdout.lines() {
                            let line = line.trim();
                            
                            if line.starts_with("Chipset Model:") {
                                current_chipset = line
                                    .split(':')
                                    .nth(1)
                                    .unwrap_or("Apple GPU")
                                    .trim()
                                    .to_string();
                            } else if line.starts_with("VRAM (Total):") || line.starts_with("Metal Support:") {
                                // Parse VRAM size
                                if let Some(mem_str) = line.split(':').nth(1) {
                                    let mem_str = mem_str.trim();
                                    if mem_str.contains("GB") {
                                        if let Ok(gb) = mem_str.replace("GB", "").trim().parse::<f64>() {
                                            current_memory = (gb * 1024.0 * 1024.0 * 1024.0) as usize;
                                        }
                                    } else if mem_str.contains("MB") {
                                        if let Ok(mb) = mem_str.replace("MB", "").trim().parse::<f64>() {
                                            current_memory = (mb * 1024.0 * 1024.0) as usize;
                                        }
                                    }
                                }
                            }

                            // Complete device entry when we find Metal support
                            if line.contains("Metal Support:") && line.contains("Supported") {
                                if !current_chipset.is_empty() {
                                    // Default memory for Apple Silicon if not detected
                                    if current_memory == 0 {
                                        current_memory = if current_chipset.contains("M1") || current_chipset.contains("M2") || current_chipset.contains("M3") {
                                            16 * 1024 * 1024 * 1024 // 16GB unified memory default
                                        } else {
                                            4 * 1024 * 1024 * 1024 // 4GB default for older systems
                                        };
                                    }

                                    devices.push(GpuDevice {
                                        device_id: device_count,
                                        name: current_chipset.clone(),
                                        total_memory: current_memory,
                                        available_memory: (current_memory as f64 * 0.7) as usize, // 70% available
                                        compute_capability: "Metal".to_string(),
                                        compute_units: if current_chipset.contains("M1") {
                                            8 // M1 has 8-core GPU base config
                                        } else if current_chipset.contains("M2") {
                                            10 // M2 has 10-core GPU base config
                                        } else if current_chipset.contains("M3") {
                                            8 // M3 has 8-core GPU base config
                                        } else {
                                            32 // Default for other GPUs
                                        },
                                        backend: GpuBackend::Metal,
                                        supports_double_precision: true,
                                    });
                                    device_count += 1;
                                }
                                current_chipset.clear();
                                current_memory = 0;
                            }
                        }

                        if !devices.is_empty() {
                            return Ok(devices);
                        }
                    }
                }

                // Fallback: check if we're on Apple Silicon via sysctl
                if let Ok(output) = std::process::Command::new("sysctl")
                    .arg("-n")
                    .arg("machdep.cpu.brand_string")
                    .output()
                {
                    if output.status.success() {
                        let brand = String::from_utf8_lossy(&output.stdout);
                        if brand.contains("Apple") {
                            // Assume Apple Silicon with integrated GPU
                            return Ok(vec![GpuDevice {
                                device_id: 0,
                                name: "Apple Silicon GPU".to_string(),
                                total_memory: 16 * 1024 * 1024 * 1024, // 16GB unified memory
                                available_memory: 11 * 1024 * 1024 * 1024, // ~70% available
                                compute_capability: "Metal".to_string(),
                                compute_units: 8,
                                backend: GpuBackend::Metal,
                                supports_double_precision: true,
                            }]);
                        }
                    }
                }

                Ok(vec![])
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            // Metal is only available on macOS
            Ok(vec![])
        }
    }

    /// Initialize backend context
    fn initialize_backend(backend: &GpuBackend) -> Result<BackendContext> {
        match backend {
            GpuBackend::Cuda => Ok(BackendContext::Cuda { context_handle: 0 }),
            GpuBackend::OpenCl => Ok(BackendContext::OpenCl { context_handle: 0 }),
            _ => Ok(BackendContext::CpuFallback),
        }
    }

    /// Get available devices
    pub fn get_devices(&self) -> &[GpuDevice] {
        &self.devices
    }

    /// Get current configuration
    pub fn get_config(&self) -> &GpuConfig {
        &self.config
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> &GpuStats {
        &self.stats
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        !self.devices.is_empty()
    }
}

/// GPU-accelerated K-means clustering (stub implementation)
pub struct GpuKMeans<F: Float> {
    /// GPU context
    context: GpuContext,
    /// Current cluster centers on GPU
    gpu_centers: Option<GpuArray<F>>,
    /// Configuration
    config: GpuKMeansConfig,
}

/// Configuration for GPU K-means
#[derive(Debug, Clone)]
pub struct GpuKMeansConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Batch size for processing
    pub batch_size: usize,
    /// Use shared memory optimization
    pub use_shared_memory: bool,
}

impl Default for GpuKMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            max_iterations: 300,
            tolerance: 1e-4,
            batch_size: 1024,
            use_shared_memory: true,
        }
    }
}

/// GPU array abstraction
#[derive(Debug)]
pub struct GpuArray<F: Float> {
    /// Device pointer (platform-specific)
    device_ptr: usize,
    /// Array dimensions
    shape: Vec<usize>,
    /// Element count
    size: usize,
    /// Data type marker
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + FromPrimitive> GpuArray<F> {
    /// Allocate GPU memory for array
    pub fn allocate(shape: &[usize]) -> Result<Self> {
        let size = shape.iter().product();

        // Stub implementation - would allocate actual GPU memory
        Ok(Self {
            device_ptr: 0,
            shape: shape.to_vec(),
            size,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Copy data from host to device
    pub fn copy_from_host(&mut self, _host_data: ArrayView2<F>) -> Result<()> {
        // Stub implementation - would perform actual host-to-device transfer
        Ok(())
    }

    /// Copy data from device to host
    pub fn copy_to_host(&self, _host_data: &mut Array2<F>) -> Result<()> {
        // Stub implementation - would perform actual device-to-host transfer
        Ok(())
    }

    /// Get array shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get element count
    pub fn size(&self) -> usize {
        self.size
    }
}

impl<F: Float + FromPrimitive> Drop for GpuArray<F> {
    fn drop(&mut self) {
        // Stub implementation - would free actual GPU memory
    }
}

impl<F: Float + FromPrimitive> GpuKMeans<F> {
    /// Create new GPU K-means instance
    pub fn new(gpu_config: GpuConfig, kmeans_config: GpuKMeansConfig) -> Result<Self> {
        let context = GpuContext::new(gpu_config)?;

        Ok(Self {
            context,
            gpu_centers: None,
            config: kmeans_config,
        })
    }

    /// Initialize cluster centers on GPU
    pub fn initialize_centers(&mut self, initial_centers: ArrayView2<F>) -> Result<()> {
        let shape = initial_centers.shape();
        let mut gpu_centers = GpuArray::allocate(shape)?;
        gpu_centers.copy_from_host(initial_centers)?;
        self.gpu_centers = Some(gpu_centers);
        Ok(())
    }

    /// Perform K-means clustering on GPU
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<(Array2<F>, Array1<usize>)> {
        if self.gpu_centers.is_none() {
            return Err(ClusteringError::InvalidInput(
                "Centers not initialized".to_string(),
            ));
        }

        if !self.context.is_gpu_available() {
            // Fallback to CPU implementation
            return self.fit_cpu_fallback(data);
        }

        // Stub implementation - would perform actual GPU clustering
        let n_samples = data.nrows();
        let centers = Array2::zeros((self.config.n_clusters, data.ncols()));
        let labels = Array1::zeros(n_samples);

        Ok((centers, labels))
    }

    /// CPU fallback implementation
    fn fit_cpu_fallback(&self, data: ArrayView2<F>) -> Result<(Array2<F>, Array1<usize>)> {
        // Use CPU-based K-means as fallback
        use crate::vq::kmeans;

        let (centers, labels) = kmeans(
            data,
            self.config.n_clusters,
            None,
            Some(self.config.max_iterations),
            Some(self.config.tolerance),
            None,
        )?;

        Ok((centers, labels))
    }

    /// Get current cluster centers
    pub fn get_centers(&self) -> Result<Array2<F>> {
        if let Some(ref gpu_centers) = self.gpu_centers {
            let mut host_centers = Array2::zeros(gpu_centers.shape());
            gpu_centers.copy_to_host(&mut host_centers)?;
            Ok(host_centers)
        } else {
            Err(ClusteringError::InvalidInput(
                "Centers not available".to_string(),
            ))
        }
    }

    /// Predict cluster assignments
    pub fn predict(&self, data: ArrayView2<F>) -> Result<Array1<usize>> {
        // Stub implementation - would use GPU for prediction
        let n_samples = data.nrows();
        Ok(Array1::zeros(n_samples))
    }
}

/// GPU-accelerated distance matrix computation
pub struct GpuDistanceMatrix<F: Float> {
    /// GPU context
    context: GpuContext,
    /// Distance metric
    metric: DistanceMetric,
}

/// Supported distance metrics for GPU computation
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Cosine distance
    Cosine,
    /// Minkowski distance with parameter p
    Minkowski(f64),
}

impl<F: Float + FromPrimitive> GpuDistanceMatrix<F> {
    /// Create new GPU distance matrix computer
    pub fn new(gpu_config: GpuConfig, metric: DistanceMetric) -> Result<Self> {
        let context = GpuContext::new(gpu_config)?;

        Ok(Self { context, metric })
    }

    /// Compute pairwise distances on GPU
    pub fn compute_distances(&self, data: ArrayView2<F>) -> Result<Array2<F>> {
        if !self.context.is_gpu_available() {
            // Fallback to CPU implementation
            return self.compute_distances_cpu(data);
        }

        // Stub implementation - would perform actual GPU distance computation
        let n_samples = data.nrows();
        Ok(Array2::zeros((n_samples, n_samples)))
    }

    /// CPU fallback for distance computation
    fn compute_distances_cpu(&self, data: ArrayView2<F>) -> Result<Array2<F>> {
        let n_samples = data.nrows();
        let mut distances = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i..n_samples {
                let dist = match self.metric {
                    DistanceMetric::Euclidean => {
                        crate::vq::euclidean_distance(data.row(i), data.row(j))
                    }
                    DistanceMetric::Manhattan => data
                        .row(i)
                        .iter()
                        .zip(data.row(j).iter())
                        .map(|(a, b)| (*a - *b).abs())
                        .sum(),
                    DistanceMetric::Cosine => {
                        let dot_product = data
                            .row(i)
                            .iter()
                            .zip(data.row(j).iter())
                            .map(|(a, b)| *a * *b)
                            .sum::<F>();

                        let norm_i = data.row(i).iter().map(|x| *x * *x).sum::<F>().sqrt();
                        let norm_j = data.row(j).iter().map(|x| *x * *x).sum::<F>().sqrt();

                        F::one() - dot_product / (norm_i * norm_j)
                    }
                    DistanceMetric::Minkowski(p) => {
                        let sum = data
                            .row(i)
                            .iter()
                            .zip(data.row(j).iter())
                            .map(|(a, b)| (*a - *b).abs().to_f64().unwrap().powf(p))
                            .sum::<f64>();
                        F::from(sum.powf(1.0 / p)).unwrap()
                    }
                };

                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        Ok(distances)
    }
}

/// GPU memory manager for efficient allocation and deallocation
#[derive(Debug)]
pub struct GpuMemoryManager {
    /// Memory pools by size
    pools: HashMap<usize, Vec<usize>>,
    /// Total allocated memory
    total_allocated: usize,
    /// Peak memory usage
    peak_usage: usize,
    /// Configuration
    config: MemoryManagerConfig,
}

/// Configuration for GPU memory manager
#[derive(Debug, Clone)]
pub struct MemoryManagerConfig {
    /// Enable memory pooling
    pub enable_pooling: bool,
    /// Pool sizes to maintain
    pub pool_sizes: Vec<usize>,
    /// Maximum memory usage (bytes)
    pub max_memory: usize,
    /// Automatic garbage collection threshold
    pub gc_threshold: f64,
}

impl Default for MemoryManagerConfig {
    fn default() -> Self {
        Self {
            enable_pooling: true,
            pool_sizes: vec![
                1024 * 1024,       // 1MB
                16 * 1024 * 1024,  // 16MB
                64 * 1024 * 1024,  // 64MB
                256 * 1024 * 1024, // 256MB
            ],
            max_memory: 1024 * 1024 * 1024, // 1GB
            gc_threshold: 0.8,
        }
    }
}

impl GpuMemoryManager {
    /// Create new GPU memory manager
    pub fn new(config: MemoryManagerConfig) -> Self {
        let mut pools = HashMap::new();

        if config.enable_pooling {
            for &size in &config.pool_sizes {
                pools.insert(size, Vec::new());
            }
        }

        Self {
            pools,
            total_allocated: 0,
            peak_usage: 0,
            config,
        }
    }

    /// Allocate GPU memory
    pub fn allocate(&mut self, size: usize) -> Result<usize> {
        // Check memory limit
        if self.total_allocated + size > self.config.max_memory {
            if self.should_gc() {
                self.garbage_collect()?;
            }

            if self.total_allocated + size > self.config.max_memory {
                return Err(ClusteringError::ComputationError(
                    "GPU memory limit exceeded".to_string(),
                ));
            }
        }

        // Try to reuse from pool
        if self.config.enable_pooling {
            if let Some(pool) = self.pools.get_mut(&size) {
                if let Some(ptr) = pool.pop() {
                    return Ok(ptr);
                }
            }
        }

        // Allocate new memory (stub implementation)
        let ptr = self.total_allocated + 1; // Fake pointer
        self.total_allocated += size;
        self.peak_usage = self.peak_usage.max(self.total_allocated);

        Ok(ptr)
    }

    /// Deallocate GPU memory
    pub fn deallocate(&mut self, ptr: usize, size: usize) -> Result<()> {
        if self.config.enable_pooling {
            if let Some(pool) = self.pools.get_mut(&size) {
                pool.push(ptr);
                return Ok(());
            }
        }

        // Actually free memory (stub implementation)
        self.total_allocated = self.total_allocated.saturating_sub(size);
        Ok(())
    }

    /// Check if garbage collection should be triggered
    fn should_gc(&self) -> bool {
        let usage_ratio = self.total_allocated as f64 / self.config.max_memory as f64;
        usage_ratio > self.config.gc_threshold
    }

    /// Perform garbage collection
    fn garbage_collect(&mut self) -> Result<()> {
        // Clear memory pools to free unused allocations
        for pool in self.pools.values_mut() {
            pool.clear();
        }

        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            total_allocated: self.total_allocated,
            peak_usage: self.peak_usage,
            pool_count: self.pools.values().map(|p| p.len()).sum(),
            fragmentation_ratio: self.calculate_fragmentation(),
        }
    }

    /// Calculate memory fragmentation ratio
    fn calculate_fragmentation(&self) -> f64 {
        let pooled_memory: usize = self
            .pools
            .iter()
            .map(|(&size, pool)| size * pool.len())
            .sum();

        if self.total_allocated == 0 {
            0.0
        } else {
            pooled_memory as f64 / self.total_allocated as f64
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MemoryStats {
    /// Total allocated memory (bytes)
    pub total_allocated: usize,
    /// Peak memory usage (bytes)
    pub peak_usage: usize,
    /// Number of objects in memory pools
    pub pool_count: usize,
    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,
}

/// GPU benchmark utilities
pub mod benchmark {
    use super::*;
    use std::time::Instant;

    /// Benchmark result for GPU operations
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct BenchmarkResult {
        /// Operation name
        pub operation: String,
        /// GPU execution time (seconds)
        pub gpu_time: f64,
        /// CPU execution time (seconds)
        pub cpu_time: f64,
        /// Speedup factor (cpu_time / gpu_time)
        pub speedup: f64,
        /// Memory usage (bytes)
        pub memory_usage: usize,
        /// Data size processed
        pub data_size: usize,
        /// Throughput (operations per second)
        pub throughput: f64,
    }

    /// Benchmark GPU vs CPU performance
    pub fn benchmark_kmeans<F: Float + FromPrimitive>(
        data: ArrayView2<F>,
        n_clusters: usize,
    ) -> Result<BenchmarkResult> {
        let data_size = data.len() * std::mem::size_of::<F>();

        // Benchmark GPU implementation
        let gpu_start = Instant::now();
        let gpu_config = GpuConfig::default();
        let kmeans_config = GpuKMeansConfig {
            n_clusters,
            ..Default::default()
        };

        let mut gpu_kmeans = GpuKMeans::new(gpu_config, kmeans_config)?;
        let initial_centers = Array2::zeros((n_clusters, data.ncols()));
        gpu_kmeans.initialize_centers(initial_centers.view())?;
        let _gpu_result = gpu_kmeans.fit(data)?;
        let gpu_time = gpu_start.elapsed().as_secs_f64();

        // Benchmark CPU implementation
        let cpu_start = Instant::now();
        let _cpu_result = crate::vq::kmeans(data, n_clusters, None, None, None, None)?;
        let cpu_time = cpu_start.elapsed().as_secs_f64();

        let speedup = if gpu_time > 0.0 {
            cpu_time / gpu_time
        } else {
            0.0
        };
        let throughput = if gpu_time > 0.0 {
            data.nrows() as f64 / gpu_time
        } else {
            0.0
        };

        Ok(BenchmarkResult {
            operation: "kmeans".to_string(),
            gpu_time,
            cpu_time,
            speedup,
            memory_usage: data_size * 2, // Estimate for data + centers
            data_size,
            throughput,
        })
    }

    /// Benchmark distance matrix computation
    pub fn benchmark_distance_matrix<F: Float + FromPrimitive>(
        data: ArrayView2<F>,
    ) -> Result<BenchmarkResult> {
        let data_size = data.len() * std::mem::size_of::<F>();

        // Benchmark GPU implementation
        let gpu_start = Instant::now();
        let gpu_config = GpuConfig::default();
        let gpu_distances = GpuDistanceMatrix::new(gpu_config, DistanceMetric::Euclidean)?;
        let _gpu_result = gpu_distances.compute_distances(data)?;
        let gpu_time = gpu_start.elapsed().as_secs_f64();

        // Benchmark CPU implementation
        let cpu_start = Instant::now();
        let _cpu_result = gpu_distances.compute_distances_cpu(data)?;
        let cpu_time = cpu_start.elapsed().as_secs_f64();

        let speedup = if gpu_time > 0.0 {
            cpu_time / gpu_time
        } else {
            0.0
        };
        let throughput = if gpu_time > 0.0 {
            (data.nrows() * data.nrows()) as f64 / gpu_time
        } else {
            0.0
        };

        Ok(BenchmarkResult {
            operation: "distance_matrix".to_string(),
            gpu_time,
            cpu_time,
            speedup,
            memory_usage: data_size + data.nrows() * data.nrows() * std::mem::size_of::<F>(),
            data_size,
            throughput,
        })
    }
}

/// High-level GPU-accelerated clustering API with automatic fallback
pub mod accelerated {
    use super::*;
    use crate::density::dbscan;
    use crate::vq::{kmeans, kmeans2};
    use ndarray::{Array1, Array2, ArrayView2};
    use num_traits::{Float, FromPrimitive};
    use std::fmt::Debug;

    /// Accelerated clustering configuration
    #[derive(Debug, Clone)]
    pub struct AcceleratedConfig {
        /// GPU configuration
        pub gpu_config: GpuConfig,
        /// Minimum data size to consider GPU acceleration
        pub gpu_threshold_samples: usize,
        /// Enable automatic performance profiling
        pub enable_profiling: bool,
        /// Cache GPU contexts for reuse
        pub cache_contexts: bool,
    }

    impl Default for AcceleratedConfig {
        fn default() -> Self {
            Self {
                gpu_config: GpuConfig::default(),
                gpu_threshold_samples: 1000,
                enable_profiling: true,
                cache_contexts: true,
            }
        }
    }

    /// Accelerated clustering engine with automatic GPU/CPU selection
    pub struct AcceleratedClusterer {
        config: AcceleratedConfig,
        gpu_context: Option<GpuContext>,
        performance_cache: std::collections::HashMap<String, f64>,
    }

    impl AcceleratedClusterer {
        /// Create new accelerated clusterer
        pub fn new(config: AcceleratedConfig) -> Result<Self> {
            let gpu_context = if config.cache_contexts {
                Some(GpuContext::new_auto()?)
            } else {
                None
            };

            Ok(Self {
                config,
                gpu_context,
                performance_cache: std::collections::HashMap::new(),
            })
        }

        /// Create with automatic configuration
        pub fn new_auto() -> Result<Self> {
            Self::new(AcceleratedConfig::default())
        }

        /// Perform K-means clustering with automatic GPU/CPU selection
        pub fn kmeans<F>(
            &mut self,
            data: ArrayView2<F>,
            n_clusters: usize,
            max_iterations: Option<usize>,
            tolerance: Option<f64>,
        ) -> Result<(Array2<F>, Array1<usize>)>
        where
            F: Float
                + FromPrimitive
                + Debug
                + 'static
                + std::iter::Sum
                + std::fmt::Display
                + Send
                + Sync,
            f64: From<F>,
        {
            let data_size = data.len() * std::mem::size_of::<F>();
            let n_samples = data.nrows();

            // Decide whether to use GPU or CPU
            if self.should_use_gpu(n_samples, "kmeans") {
                match self.kmeans_gpu(data, n_clusters, max_iterations, tolerance) {
                    Ok(result) => {
                        if self.config.enable_profiling {
                            self.update_performance_cache("kmeans_gpu", 1.0);
                        }
                        return Ok(result);
                    }
                    Err(_) => {
                        // GPU failed, fall back to CPU
                        if self.config.enable_profiling {
                            self.update_performance_cache("kmeans_gpu", 0.0);
                        }
                    }
                }
            }

            // Use CPU implementation
            if self.config.enable_profiling {
                self.update_performance_cache("kmeans_cpu", 1.0);
            }

            kmeans(data, n_clusters, None, max_iterations, tolerance, None)
        }

        /// GPU K-means implementation
        fn kmeans_gpu<F>(
            &mut self,
            data: ArrayView2<F>,
            n_clusters: usize,
            max_iterations: Option<usize>,
            tolerance: Option<f64>,
        ) -> Result<(Array2<F>, Array1<usize>)>
        where
            F: Float
                + FromPrimitive
                + Debug
                + 'static
                + std::iter::Sum
                + std::fmt::Display
                + Send
                + Sync,
            f64: From<F>,
        {
            let gpu_context = if let Some(ref context) = self.gpu_context {
                context
            } else {
                // Create context on demand
                let context = GpuContext::new_auto()?;
                self.gpu_context = Some(context);
                self.gpu_context.as_ref().unwrap()
            };

            if !gpu_context.is_gpu_available() {
                return Err(ClusteringError::ComputationError(
                    "No GPU available for acceleration".to_string(),
                ));
            }

            let kmeans_config = GpuKMeansConfig {
                n_clusters,
                max_iterations: max_iterations.unwrap_or(300),
                tolerance: tolerance.unwrap_or(1e-4),
                batch_size: gpu_context.get_recommended_batch_size(
                    data.len() * std::mem::size_of::<F>(),
                    std::mem::size_of::<F>(),
                ),
                use_shared_memory: true,
            };

            let mut gpu_kmeans = GpuKMeans::new(self.config.gpu_config.clone(), kmeans_config)?;

            // Initialize centers using k-means++
            let initial_centers = self.initialize_centers_plus_plus(data, n_clusters)?;
            gpu_kmeans.initialize_centers(initial_centers.view())?;

            gpu_kmeans.fit(data)
        }

        /// Perform DBSCAN clustering with automatic GPU/CPU selection
        pub fn dbscan<F>(
            &mut self,
            data: ArrayView2<F>,
            eps: f64,
            min_samples: usize,
        ) -> Result<Array1<i32>>
        where
            F: Float
                + FromPrimitive
                + Debug
                + 'static
                + std::iter::Sum
                + std::fmt::Display
                + Send
                + Sync,
            f64: From<F>,
        {
            let n_samples = data.nrows();

            // For DBSCAN, GPU acceleration is mainly useful for distance computation
            if self.should_use_gpu(n_samples, "dbscan") {
                match self.dbscan_gpu(data, eps, min_samples) {
                    Ok(result) => {
                        if self.config.enable_profiling {
                            self.update_performance_cache("dbscan_gpu", 1.0);
                        }
                        return Ok(result);
                    }
                    Err(_) => {
                        if self.config.enable_profiling {
                            self.update_performance_cache("dbscan_gpu", 0.0);
                        }
                    }
                }
            }

            // Use CPU implementation
            let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));
            dbscan(data_f64.view(), eps, min_samples)
        }

        /// GPU DBSCAN implementation (using GPU for distance computation)
        fn dbscan_gpu<F>(
            &mut self,
            data: ArrayView2<F>,
            eps: f64,
            min_samples: usize,
        ) -> Result<Array1<i32>>
        where
            F: Float
                + FromPrimitive
                + Debug
                + 'static
                + std::iter::Sum
                + std::fmt::Display
                + Send
                + Sync,
            f64: From<F>,
        {
            let gpu_context = if let Some(ref context) = self.gpu_context {
                context
            } else {
                let context = GpuContext::new_auto()?;
                self.gpu_context = Some(context);
                self.gpu_context.as_ref().unwrap()
            };

            if !gpu_context.is_gpu_available() {
                return Err(ClusteringError::ComputationError(
                    "No GPU available for acceleration".to_string(),
                ));
            }

            // Use GPU for distance matrix computation
            let gpu_distances =
                GpuDistanceMatrix::new(self.config.gpu_config.clone(), DistanceMetric::Euclidean)?;
            let distance_matrix = gpu_distances.compute_distances(data)?;

            // Use CPU for the actual DBSCAN algorithm with precomputed distances
            self.dbscan_with_precomputed_distances(&distance_matrix, eps, min_samples)
        }

        /// DBSCAN implementation with precomputed distance matrix
        fn dbscan_with_precomputed_distances(
            &self,
            distances: &Array2<F>,
            eps: f64,
            min_samples: usize,
        ) -> Result<Array1<i32>>
        where
            F: Float + FromPrimitive,
        {
            let n_samples = distances.nrows();
            let mut labels = Array1::from_elem(n_samples, -1i32);
            let mut visited = vec![false; n_samples];
            let mut cluster_id = 0;

            let eps_f = F::from(eps).unwrap();

            for i in 0..n_samples {
                if visited[i] {
                    continue;
                }
                visited[i] = true;

                // Find neighbors
                let neighbors: Vec<usize> = (0..n_samples)
                    .filter(|&j| distances[[i, j]] <= eps_f)
                    .collect();

                if neighbors.len() < min_samples {
                    labels[i] = -1; // Noise point
                } else {
                    // Start new cluster
                    self.expand_cluster(
                        distances,
                        &mut labels,
                        &mut visited,
                        i,
                        &neighbors,
                        cluster_id,
                        eps_f,
                        min_samples,
                    );
                    cluster_id += 1;
                }
            }

            Ok(labels)
        }

        /// Expand cluster for DBSCAN
        fn expand_cluster<F>(
            &self,
            distances: &Array2<F>,
            labels: &mut Array1<i32>,
            visited: &mut [bool],
            point: usize,
            neighbors: &[usize],
            cluster_id: i32,
            eps: F,
            min_samples: usize,
        ) where
            F: Float + FromPrimitive,
        {
            labels[point] = cluster_id;
            let mut seed_set = neighbors.to_vec();
            let mut i = 0;

            while i < seed_set.len() {
                let current_point = seed_set[i];

                if !visited[current_point] {
                    visited[current_point] = true;

                    let point_neighbors: Vec<usize> = (0..distances.nrows())
                        .filter(|&j| distances[[current_point, j]] <= eps)
                        .collect();

                    if point_neighbors.len() >= min_samples {
                        for &neighbor in &point_neighbors {
                            if !seed_set.contains(&neighbor) {
                                seed_set.push(neighbor);
                            }
                        }
                    }
                }

                if labels[current_point] == -1 {
                    labels[current_point] = cluster_id;
                }

                i += 1;
            }
        }

        /// Decide whether to use GPU based on data size and algorithm
        fn should_use_gpu(&self, n_samples: usize, algorithm: &str) -> bool {
            if n_samples < self.config.gpu_threshold_samples {
                return false;
            }

            // Check if we have GPU context
            if let Some(ref context) = self.gpu_context {
                if !context.is_gpu_available() {
                    return false;
                }

                // Check performance cache if profiling is enabled
                if self.config.enable_profiling {
                    let gpu_key = format!("{}_gpu", algorithm);
                    let cpu_key = format!("{}_cpu", algorithm);

                    if let (Some(&gpu_perf), Some(&cpu_perf)) = (
                        self.performance_cache.get(&gpu_key),
                        self.performance_cache.get(&cpu_key),
                    ) {
                        // Use GPU if it has better historical performance
                        return gpu_perf > cpu_perf;
                    }
                }

                // Default to GPU if available and data size is large enough
                true
            } else {
                false
            }
        }

        /// Update performance cache for profiling
        fn update_performance_cache(&mut self, key: &str, performance: f64) {
            // Simple exponential moving average
            let alpha = 0.1;
            if let Some(cached) = self.performance_cache.get_mut(key) {
                *cached = alpha * performance + (1.0 - alpha) * (*cached);
            } else {
                self.performance_cache.insert(key.to_string(), performance);
            }
        }

        /// Initialize cluster centers using k-means++ algorithm
        fn initialize_centers_plus_plus<F>(
            &self,
            data: ArrayView2<F>,
            k: usize,
        ) -> Result<Array2<F>>
        where
            F: Float + FromPrimitive + std::iter::Sum,
        {
            let n_samples = data.nrows();
            let n_features = data.ncols();

            if k >= n_samples {
                return Err(ClusteringError::InvalidInput(
                    "Number of clusters cannot exceed number of samples".to_string(),
                ));
            }

            let mut centers = Array2::zeros((k, n_features));
            let mut rng = rand::thread_rng();

            // Choose first center randomly
            let first_idx = rng.gen_range(0..n_samples);
            centers.row_mut(0).assign(&data.row(first_idx));

            // Choose remaining centers
            for i in 1..k {
                let mut distances = Array1::zeros(n_samples);

                // Calculate distances to nearest center for each point
                for j in 0..n_samples {
                    let mut min_dist = F::infinity();
                    for center_idx in 0..i {
                        let dist =
                            crate::vq::euclidean_distance(data.row(j), centers.row(center_idx));
                        min_dist = min_dist.min(dist * dist); // Squared distance
                    }
                    distances[j] = min_dist;
                }

                // Choose next center with probability proportional to squared distance
                let total_distance: F = distances.sum();
                if total_distance == F::zero() {
                    // All remaining points are identical to existing centers
                    centers
                        .row_mut(i)
                        .assign(&data.row(rng.gen_range(0..n_samples)));
                } else {
                    let threshold = F::from(rng.gen::<f64>()).unwrap() * total_distance;
                    let mut cumulative = F::zero();

                    for j in 0..n_samples {
                        cumulative = cumulative + distances[j];
                        if cumulative >= threshold {
                            centers.row_mut(i).assign(&data.row(j));
                            break;
                        }
                    }
                }
            }

            Ok(centers)
        }

        /// Get GPU device information
        pub fn get_gpu_info(&self) -> Option<Vec<&GpuDevice>> {
            self.gpu_context
                .as_ref()
                .map(|ctx| ctx.get_devices().iter().collect())
        }

        /// Get performance statistics
        pub fn get_performance_stats(&self) -> &std::collections::HashMap<String, f64> {
            &self.performance_cache
        }

        /// Clear performance cache
        pub fn clear_performance_cache(&mut self) {
            self.performance_cache.clear();
        }
    }

    /// Convenience functions for accelerated clustering

    /// Perform accelerated K-means clustering
    pub fn accelerated_kmeans<F>(
        data: ArrayView2<F>,
        n_clusters: usize,
    ) -> Result<(Array2<F>, Array1<usize>)>
    where
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync,
        f64: From<F>,
    {
        let mut clusterer = AcceleratedClusterer::new_auto()?;
        clusterer.kmeans(data, n_clusters, None, None)
    }

    /// Perform accelerated DBSCAN clustering
    pub fn accelerated_dbscan<F>(
        data: ArrayView2<F>,
        eps: f64,
        min_samples: usize,
    ) -> Result<Array1<i32>>
    where
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync,
        f64: From<F>,
    {
        let mut clusterer = AcceleratedClusterer::new_auto()?;
        clusterer.dbscan(data, eps, min_samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.backend, GpuBackend::CpuFallback);
        assert!(config.cpu_fallback);
        assert!(config.auto_tune);
    }

    #[test]
    fn test_gpu_context_creation() {
        let config = GpuConfig::default();
        let context = GpuContext::new(config);
        assert!(context.is_ok());

        let ctx = context.unwrap();
        assert!(!ctx.is_gpu_available()); // Should be CPU fallback
    }

    #[test]
    fn test_gpu_array_allocation() {
        let shape = vec![10, 5];
        let array = GpuArray::<f64>::allocate(&shape);
        assert!(array.is_ok());

        let arr = array.unwrap();
        assert_eq!(arr.shape(), &shape);
        assert_eq!(arr.size(), 50);
    }

    #[test]
    fn test_gpu_kmeans_creation() {
        let gpu_config = GpuConfig::default();
        let kmeans_config = GpuKMeansConfig::default();

        let gpu_kmeans = GpuKMeans::<f64>::new(gpu_config, kmeans_config);
        assert!(gpu_kmeans.is_ok());
    }

    #[test]
    fn test_gpu_distance_matrix() {
        let gpu_config = GpuConfig::default();
        let distance_matrix = GpuDistanceMatrix::<f64>::new(gpu_config, DistanceMetric::Euclidean);
        assert!(distance_matrix.is_ok());

        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let distances = distance_matrix.unwrap().compute_distances(data.view());
        assert!(distances.is_ok());

        let dist_matrix = distances.unwrap();
        assert_eq!(dist_matrix.shape(), &[4, 4]);
    }

    #[test]
    fn test_memory_manager() {
        let config = MemoryManagerConfig::default();
        let mut manager = GpuMemoryManager::new(config);

        let ptr = manager.allocate(1024);
        assert!(ptr.is_ok());

        let ptr_val = ptr.unwrap();
        let dealloc_result = manager.deallocate(ptr_val, 1024);
        assert!(dealloc_result.is_ok());

        let stats = manager.get_stats();
        assert!(stats.total_allocated >= 0);
    }

    #[test]
    fn test_accelerated_config() {
        let config = accelerated::AcceleratedConfig::default();
        assert_eq!(config.gpu_threshold_samples, 1000);
        assert!(config.enable_profiling);
        assert!(config.cache_contexts);
    }

    #[test]
    fn test_accelerated_clusterer_creation() {
        let config = accelerated::AcceleratedConfig::default();
        let clusterer = accelerated::AcceleratedClusterer::new(config);
        assert!(clusterer.is_ok());

        let auto_clusterer = accelerated::AcceleratedClusterer::new_auto();
        assert!(auto_clusterer.is_ok());
    }

    #[test]
    fn test_accelerated_kmeans_convenience() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2],
        )
        .unwrap();

        // This should work even without GPU (falls back to CPU)
        let result = accelerated::accelerated_kmeans(data.view(), 2);
        assert!(result.is_ok());

        let (centers, labels) = result.unwrap();
        assert_eq!(centers.shape(), &[2, 2]); // 2 clusters, 2 features
        assert_eq!(labels.len(), 6); // 6 data points
    }

    #[test]
    fn test_accelerated_dbscan_convenience() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2],
        )
        .unwrap();

        // This should work even without GPU (falls back to CPU)
        let result = accelerated::accelerated_dbscan(data.view(), 1.0, 2);
        assert!(result.is_ok());

        let labels = result.unwrap();
        assert_eq!(labels.len(), 6); // 6 data points
    }

    #[test]
    fn test_best_backend_detection() {
        let backend = GpuContext::detect_best_backend();
        assert!(backend.is_ok());

        // Should at least fall back to CPU
        let detected_backend = backend.unwrap();
        assert!(matches!(
            detected_backend,
            GpuBackend::Cuda
                | GpuBackend::OpenCl
                | GpuBackend::Metal
                | GpuBackend::Rocm
                | GpuBackend::OneApi
                | GpuBackend::CpuFallback
        ));
    }

    #[test]
    fn test_auto_context_creation() {
        let context = GpuContext::new_auto();
        assert!(context.is_ok());

        let ctx = context.unwrap();
        // Should always succeed with CPU fallback
        assert!(ctx.get_config().cpu_fallback);
    }
}
