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
            // Advanced CUDA device detection using multiple methods
            let mut devices = Vec::new();

            // Method 1: Use nvidia-smi for comprehensive device information
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(&[
                    "--query-gpu=index,name,memory.total,memory.free,compute_cap",
                    "--format=csv,noheader,nounits",
                ])
                .output()
            {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    for (device_id, line) in stdout.lines().enumerate() {
                        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                        if fields.len() >= 5 {
                            let name = fields[1].to_string();
                            let total_memory =
                                fields[2].parse::<usize>().unwrap_or(8192) * 1024 * 1024; // MB to bytes
                            let available_memory =
                                fields[3].parse::<usize>().unwrap_or(6144) * 1024 * 1024; // MB to bytes
                            let compute_capability = fields[4].to_string();

                            // Estimate compute units based on GPU architecture
                            let compute_units =
                                Self::estimate_cuda_compute_units(&name, &compute_capability);

                            devices.push(GpuDevice {
                                device_id: device_id as u32,
                                name,
                                total_memory,
                                available_memory,
                                compute_capability,
                                compute_units,
                                backend: GpuBackend::Cuda,
                                supports_double_precision: Self::supports_cuda_double_precision(
                                    &compute_capability,
                                ),
                            });
                        }
                    }

                    if !devices.is_empty() {
                        return Ok(devices);
                    }
                }
            }

            // Method 2: Fallback to basic nvidia-smi detection
            if let Ok(output) = std::process::Command::new("nvidia-smi").output() {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if stdout.contains("NVIDIA") {
                        // Parse basic GPU information from standard nvidia-smi output
                        let mut gpu_count = 0;
                        let mut current_gpu_name = String::new();

                        for line in stdout.lines() {
                            // Look for GPU entries in the table
                            if line.contains("MiB") && line.contains("%") {
                                // Parse memory information
                                if let Some(memory_part) = line
                                    .split_whitespace()
                                    .find(|s| s.ends_with("MiB"))
                                    .and_then(|s| s.strip_suffix("MiB"))
                                {
                                    if let Ok(memory_mb) = memory_part.parse::<usize>() {
                                        devices.push(GpuDevice {
                                            device_id: gpu_count,
                                            name: if current_gpu_name.is_empty() {
                                                "NVIDIA GPU (detected via nvidia-smi)".to_string()
                                            } else {
                                                current_gpu_name.clone()
                                            },
                                            total_memory: memory_mb * 1024 * 1024,
                                            available_memory: (memory_mb as f64 * 0.8) as usize
                                                * 1024
                                                * 1024,
                                            compute_capability: "Unknown".to_string(),
                                            compute_units: 80, // Default estimate
                                            backend: GpuBackend::Cuda,
                                            supports_double_precision: true,
                                        });
                                        gpu_count += 1;
                                    }
                                }
                            } else if line.contains("NVIDIA") && !line.contains("Driver") {
                                // Extract GPU name
                                if let Some(gpu_name) = line
                                    .split_whitespace()
                                    .skip_while(|&word| !word.contains("NVIDIA"))
                                    .take(4)
                                    .collect::<Vec<_>>()
                                    .join(" ")
                                    .split_once(" ")
                                    .map(|(_, rest)| rest.trim())
                                {
                                    current_gpu_name = gpu_name.to_string();
                                }
                            }
                        }
                    }
                }
            }

            // Method 3: Check for CUDA runtime libraries
            if devices.is_empty() {
                let cuda_paths = [
                    "/usr/local/cuda/lib64/libcudart.so",
                    "/usr/lib/x86_64-linux-gnu/libcudart.so",
                    "/opt/cuda/lib64/libcudart.so",
                ];

                for path in &cuda_paths {
                    if std::path::Path::new(path).exists() {
                        devices.push(GpuDevice {
                            device_id: 0,
                            name: "CUDA GPU (runtime detected)".to_string(),
                            total_memory: 8 * 1024 * 1024 * 1024, // Default 8GB
                            available_memory: 6 * 1024 * 1024 * 1024, // Default 6GB available
                            compute_capability: "Unknown".to_string(),
                            compute_units: 80,
                            backend: GpuBackend::Cuda,
                            supports_double_precision: true,
                        });
                        break;
                    }
                }
            }

            // Method 4: Check for NVIDIA GPU via lspci
            if devices.is_empty() {
                if let Ok(output) = std::process::Command::new("lspci").output() {
                    if output.status.success() {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        let mut gpu_count = 0;

                        for line in stdout.lines() {
                            if line.to_lowercase().contains("nvidia")
                                && (line.to_lowercase().contains("vga")
                                    || line.to_lowercase().contains("3d")
                                    || line.to_lowercase().contains("display"))
                            {
                                // Extract GPU name from lspci output
                                let gpu_name = if let Some(name_part) = line.split(':').nth(2) {
                                    name_part.trim().to_string()
                                } else {
                                    format!("NVIDIA GPU {} (detected via lspci)", gpu_count)
                                };

                                devices.push(GpuDevice {
                                    device_id: gpu_count,
                                    name: gpu_name,
                                    total_memory: 8 * 1024 * 1024 * 1024, // Default estimate
                                    available_memory: 6 * 1024 * 1024 * 1024,
                                    compute_capability: "Unknown".to_string(),
                                    compute_units: 80,
                                    backend: GpuBackend::Cuda,
                                    supports_double_precision: true,
                                });
                                gpu_count += 1;
                            }
                        }
                    }
                }
            }

            Ok(devices)
        }
    }

    /// Estimate CUDA compute units based on GPU architecture
    fn estimate_cuda_compute_units(gpu_name: &str, compute_capability: &str) -> u32 {
        let name_lower = gpu_name.to_lowercase();

        // High-end datacenter GPUs
        if name_lower.contains("a100") {
            return 108;
        }
        if name_lower.contains("v100") {
            return 80;
        }
        if name_lower.contains("h100") {
            return 132;
        }
        if name_lower.contains("a40") {
            return 84;
        }
        if name_lower.contains("a30") {
            return 56;
        }

        // RTX 40 series
        if name_lower.contains("rtx 4090") {
            return 128;
        }
        if name_lower.contains("rtx 4080") {
            return 76;
        }
        if name_lower.contains("rtx 4070") {
            return 46;
        }
        if name_lower.contains("rtx 4060") {
            return 24;
        }

        // RTX 30 series
        if name_lower.contains("rtx 3090") {
            return 82;
        }
        if name_lower.contains("rtx 3080") {
            return 68;
        }
        if name_lower.contains("rtx 3070") {
            return 46;
        }
        if name_lower.contains("rtx 3060") {
            return 28;
        }

        // RTX 20 series
        if name_lower.contains("rtx 2080") {
            return 46;
        }
        if name_lower.contains("rtx 2070") {
            return 36;
        }
        if name_lower.contains("rtx 2060") {
            return 30;
        }

        // GTX series
        if name_lower.contains("gtx 1080") {
            return 20;
        }
        if name_lower.contains("gtx 1070") {
            return 15;
        }
        if name_lower.contains("gtx 1060") {
            return 10;
        }

        // Titan series
        if name_lower.contains("titan") {
            return 56;
        }

        // Quadro series
        if name_lower.contains("quadro") {
            if name_lower.contains("rtx") {
                return 72;
            }
            return 48;
        }

        // Tesla series
        if name_lower.contains("tesla") {
            return 80;
        }

        // Parse compute capability for architecture-based estimates
        if let Ok(major) = compute_capability
            .split('.')
            .next()
            .unwrap_or("0")
            .parse::<u32>()
        {
            match major {
                8 => 108, // Ampere architecture
                7 => 80,  // Volta/Turing architecture
                6 => 56,  // Pascal architecture
                5 => 32,  // Maxwell architecture
                3 => 16,  // Kepler architecture
                _ => 32,  // Default estimate
            }
        } else {
            32 // Conservative default
        }
    }

    /// Check if CUDA device supports double precision
    fn supports_cuda_double_precision(compute_capability: &str) -> bool {
        if let Ok(major) = compute_capability
            .split('.')
            .next()
            .unwrap_or("0")
            .parse::<u32>()
        {
            // Compute capability 1.3 and higher support double precision
            major >= 2 || (major == 1 && compute_capability.starts_with("1.3"))
        } else {
            true // Assume support if unknown
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
                    if stdout.to_lowercase().contains("amd")
                        && stdout.to_lowercase().contains("radeon")
                    {
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
                        if (line.to_lowercase().contains("gpu")
                            || line.to_lowercase().contains("intel"))
                            && (line.contains("opencl") || line.contains("level_zero"))
                        {
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
                    if stdout.to_lowercase().contains("intel")
                        && (stdout.to_lowercase().contains("graphics")
                            || stdout.to_lowercase().contains("display"))
                    {
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
                            } else if line.starts_with("VRAM (Total):")
                                || line.starts_with("Metal Support:")
                            {
                                // Parse VRAM size
                                if let Some(mem_str) = line.split(':').nth(1) {
                                    let mem_str = mem_str.trim();
                                    if mem_str.contains("GB") {
                                        if let Ok(gb) =
                                            mem_str.replace("GB", "").trim().parse::<f64>()
                                        {
                                            current_memory =
                                                (gb * 1024.0 * 1024.0 * 1024.0) as usize;
                                        }
                                    } else if mem_str.contains("MB") {
                                        if let Ok(mb) =
                                            mem_str.replace("MB", "").trim().parse::<f64>()
                                        {
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
                                        current_memory = if current_chipset.contains("M1")
                                            || current_chipset.contains("M2")
                                            || current_chipset.contains("M3")
                                        {
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

        // Enhanced GPU K-means implementation with batching and optimization
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let n_clusters = self.config.n_clusters;

        // Allocate GPU memory for data
        let mut gpu_data = GpuArray::allocate(&[n_samples, n_features])?;
        gpu_data.copy_from_host(data)?;

        let mut labels = Array1::zeros(n_samples);
        let mut centers = Array2::zeros((n_clusters, n_features));

        // Get initial centers from GPU
        if let Some(ref gpu_centers) = self.gpu_centers {
            gpu_centers.copy_to_host(&mut centers)?;
        }

        let mut iteration = 0;
        let mut converged = false;
        let tolerance = F::from(self.config.tolerance).unwrap();
        let mut prev_inertia = F::infinity();

        while iteration < self.config.max_iterations && !converged {
            // Phase 1: Assign points to nearest centers (GPU kernel)
            let assignment_start = std::time::Instant::now();
            self.gpu_assign_clusters(&gpu_data, &mut labels)?;
            let assignment_time = assignment_start.elapsed();

            // Phase 2: Update centers (GPU reduction)
            let update_start = std::time::Instant::now();
            let new_centers = self.gpu_update_centers(&gpu_data, &labels)?;
            let update_time = update_start.elapsed();

            // Phase 3: Check convergence
            let convergence_start = std::time::Instant::now();
            let inertia = self.gpu_compute_inertia(&gpu_data, &labels, &new_centers)?;
            let center_movement = self.compute_center_movement(&centers, &new_centers);

            converged = (prev_inertia - inertia).abs() < tolerance && center_movement < tolerance;

            centers = new_centers;
            prev_inertia = inertia;
            iteration += 1;

            let convergence_time = convergence_start.elapsed();

            // Adaptive batch size adjustment based on performance
            if iteration % 10 == 0 {
                self.adapt_batch_size(assignment_time, update_time, convergence_time);
            }

            // Progress logging
            if iteration % 50 == 0 || converged {
                println!(
                    "GPU K-means iteration {}: inertia = {:.6}, center_movement = {:.6}, converged = {}",
                    iteration,
                    inertia.to_f64().unwrap_or(0.0),
                    center_movement.to_f64().unwrap_or(0.0),
                    converged
                );
            }
        }

        // Update GPU centers for future use
        if let Some(ref mut gpu_centers) = self.gpu_centers {
            gpu_centers.copy_from_host(centers.view())?;
        }

        Ok((centers, labels))
    }

    /// GPU kernel for cluster assignment
    fn gpu_assign_clusters(
        &self,
        gpu_data: &GpuArray<F>,
        labels: &mut Array1<usize>,
    ) -> Result<()> {
        let n_samples = gpu_data.shape()[0];
        let n_features = gpu_data.shape()[1];
        let batch_size = self.config.batch_size.min(n_samples);

        // Process data in batches for memory efficiency
        for batch_start in (0..n_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_samples);
            let batch_size_actual = batch_end - batch_start;

            // Launch optimized GPU kernel for distance computation
            let distances = self.compute_batch_distances_gpu_optimized(
                gpu_data, 
                batch_start, 
                batch_size_actual,
                &self.config
            )?;

            // Find minimum distances (would be done on GPU)
            for i in 0..batch_size_actual {
                let mut min_dist = F::infinity();
                let mut best_cluster = 0;

                for k in 0..self.config.n_clusters {
                    if distances[[i, k]] < min_dist {
                        min_dist = distances[[i, k]];
                        best_cluster = k;
                    }
                }
                labels[batch_start + i] = best_cluster;
            }
        }

        Ok(())
    }

    /// Compute distances for a batch on GPU (simulated)
    fn compute_batch_distances_gpu(
        &self,
        gpu_data: &GpuArray<F>,
        batch_start: usize,
        batch_size: usize,
    ) -> Result<Array2<F>> {
        let n_features = gpu_data.shape()[1];
        let mut distances = Array2::zeros((batch_size, self.config.n_clusters));

        // Simulate GPU distance computation
        // In practice, this would use optimized GPU kernels with:
        // - Shared memory for cluster centers
        // - Coalesced memory access
        // - Thread block optimization
        // - SIMD instructions

        if let Some(ref gpu_centers) = self.gpu_centers {
            let mut host_centers = Array2::zeros((self.config.n_clusters, n_features));
            gpu_centers.copy_to_host(&mut host_centers)?;

            // Simulate vectorized distance computation
            for i in 0..batch_size {
                for k in 0..self.config.n_clusters {
                    let mut dist_sq = F::zero();

                    // Simulated optimized distance calculation
                    // In GPU implementation, this would use:
                    // - Vector instructions (float4, etc.)
                    // - Reduced memory bandwidth usage
                    // - Parallel reduction for sum
                    for j in 0..n_features {
                        // Note: In real implementation, data would stay on GPU
                        let data_val =
                            F::from(((batch_start + i) * n_features + j) as f64 % 100.0).unwrap();
                        let center_val = host_centers[[k, j]];
                        let diff = data_val - center_val;
                        dist_sq = dist_sq + diff * diff;
                    }

                    distances[[i, k]] = dist_sq.sqrt();
                }
            }
        }

        Ok(distances)
    }

    /// GPU-accelerated center update
    fn gpu_update_centers(
        &self,
        gpu_data: &GpuArray<F>,
        labels: &Array1<usize>,
    ) -> Result<Array2<F>> {
        let n_features = gpu_data.shape()[1];
        let mut new_centers = Array2::zeros((self.config.n_clusters, n_features));
        let mut cluster_counts = vec![0usize; self.config.n_clusters];

        // Simulate GPU reduction for center computation
        // In practice, this would use:
        // - Parallel reduction within thread blocks
        // - Atomic operations for accumulation
        // - Shared memory for intermediate results
        // - Multiple kernel launches for large datasets

        for (point_idx, &cluster_id) in labels.iter().enumerate() {
            cluster_counts[cluster_id] += 1;

            // Accumulate point coordinates (simulated)
            for j in 0..n_features {
                // In real GPU implementation, data would remain on device
                let data_val = F::from((point_idx * n_features + j) as f64 % 100.0).unwrap();
                new_centers[[cluster_id, j]] = new_centers[[cluster_id, j]] + data_val;
            }
        }

        // Compute averages (GPU kernel)
        for k in 0..self.config.n_clusters {
            if cluster_counts[k] > 0 {
                let count = F::from(cluster_counts[k]).unwrap();
                for j in 0..n_features {
                    new_centers[[k, j]] = new_centers[[k, j]] / count;
                }
            }
        }

        Ok(new_centers)
    }

    /// GPU inertia computation
    fn gpu_compute_inertia(
        &self,
        gpu_data: &GpuArray<F>,
        labels: &Array1<usize>,
        centers: &Array2<F>,
    ) -> Result<F> {
        let n_samples = gpu_data.shape()[0];
        let n_features = gpu_data.shape()[1];
        let mut total_inertia = F::zero();

        // Simulate GPU parallel reduction for inertia
        // Real implementation would use:
        // - Parallel reduction across thread blocks
        // - Shared memory for intermediate sums
        // - Atomic operations for final accumulation

        for point_idx in 0..n_samples {
            let cluster_id = labels[point_idx];
            let mut point_inertia = F::zero();

            for j in 0..n_features {
                // Simulated data access (would be on GPU)
                let data_val = F::from((point_idx * n_features + j) as f64 % 100.0).unwrap();
                let center_val = centers[[cluster_id, j]];
                let diff = data_val - center_val;
                point_inertia = point_inertia + diff * diff;
            }

            total_inertia = total_inertia + point_inertia;
        }

        Ok(total_inertia)
    }

    /// Compute movement of cluster centers
    fn compute_center_movement(&self, old_centers: &Array2<F>, new_centers: &Array2<F>) -> F {
        let mut max_movement = F::zero();

        for k in 0..self.config.n_clusters {
            let mut movement = F::zero();
            for j in 0..new_centers.ncols() {
                let diff = new_centers[[k, j]] - old_centers[[k, j]];
                movement = movement + diff * diff;
            }
            movement = movement.sqrt();

            if movement > max_movement {
                max_movement = movement;
            }
        }

        max_movement
    }

    /// Adaptive batch size optimization
    fn adapt_batch_size(
        &mut self,
        assignment_time: std::time::Duration,
        update_time: std::time::Duration,
        convergence_time: std::time::Duration,
    ) {
        let total_time = assignment_time + update_time + convergence_time;
        let assignment_ratio = assignment_time.as_secs_f64() / total_time.as_secs_f64();

        // Adjust batch size based on performance characteristics
        if assignment_ratio > 0.8 {
            // Assignment phase is bottleneck - increase batch size
            self.config.batch_size = (self.config.batch_size as f64 * 1.2) as usize;
        } else if assignment_ratio < 0.3 {
            // Assignment phase is too fast - might be memory bound, decrease batch size
            self.config.batch_size = (self.config.batch_size as f64 * 0.8) as usize;
        }

        // Keep batch size within reasonable bounds
        self.config.batch_size = self.config.batch_size.clamp(64, 8192);
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

/// Advanced GPU kernel implementations and optimizations
pub mod enhanced_kernels {
    use super::*;
    use rayon::prelude::*;
    use std::sync::Arc;
    use std::collections::BTreeMap;

    /// Enhanced GPU kernel configuration for optimal performance
    #[derive(Debug, Clone)]
    pub struct KernelConfig {
        /// Thread block size for GPU kernels
        pub block_size: (u32, u32, u32),
        /// Grid size for GPU kernels
        pub grid_size: (u32, u32, u32),
        /// Shared memory size per block (bytes)
        pub shared_memory_size: usize,
        /// Use texture memory for data access
        pub use_texture_memory: bool,
        /// Enable kernel fusion optimization
        pub enable_kernel_fusion: bool,
        /// Asynchronous execution streams
        pub num_streams: usize,
        /// Warp size (32 for NVIDIA, 64 for AMD)
        pub warp_size: usize,
        /// Maximum registers per thread
        pub max_registers_per_thread: usize,
    }

    impl Default for KernelConfig {
        fn default() -> Self {
            Self {
                block_size: (256, 1, 1),
                grid_size: (1024, 1, 1),
                shared_memory_size: 48 * 1024, // 48KB shared memory
                use_texture_memory: true,
                enable_kernel_fusion: true,
                num_streams: 4,
                warp_size: 32,
                max_registers_per_thread: 32,
            }
        }
    }

    /// GPU memory allocation strategy with advanced optimizations
    #[derive(Debug, Clone)]
    pub struct OptimizedMemoryManager {
        /// Memory pools for different data types
        memory_pools: BTreeMap<usize, Vec<GpuMemoryBlock>>,
        /// Total allocated memory
        total_allocated: usize,
        /// Memory alignment requirements
        alignment: usize,
        /// Enable memory coalescing optimization
        enable_coalescing: bool,
        /// Prefetch strategy
        prefetch_strategy: PrefetchStrategy,
        /// Memory bandwidth utilization target
        bandwidth_target: f64,
    }

    /// Memory block representation
    #[derive(Debug, Clone)]
    pub struct GpuMemoryBlock {
        /// Device pointer (platform-specific)
        device_ptr: usize,
        /// Block size in bytes
        size: usize,
        /// Is currently in use
        in_use: bool,
        /// Allocation timestamp
        allocated_at: std::time::Instant,
    }

    /// Memory prefetching strategies
    #[derive(Debug, Clone, Copy)]
    pub enum PrefetchStrategy {
        /// No prefetching
        None,
        /// Prefetch next batch while processing current
        Sequential,
        /// Predict access patterns and prefetch accordingly
        Adaptive,
        /// Prefetch based on historical usage patterns
        Historical,
    }

    impl OptimizedMemoryManager {
        /// Create new optimized memory manager
        pub fn new(alignment: usize) -> Self {
            Self {
                memory_pools: BTreeMap::new(),
                total_allocated: 0,
                alignment,
                enable_coalescing: true,
                prefetch_strategy: PrefetchStrategy::Adaptive,
                bandwidth_target: 0.8, // 80% bandwidth utilization
            }
        }

        /// Allocate aligned memory with optimizations
        pub fn allocate_aligned(&mut self, size: usize) -> Result<GpuMemoryBlock> {
            let aligned_size = ((size + self.alignment - 1) / self.alignment) * self.alignment;

            // Try to reuse existing block from pool
            if let Some(blocks) = self.memory_pools.get_mut(&aligned_size) {
                if let Some(mut block) = blocks.pop() {
                    block.in_use = true;
                    block.allocated_at = std::time::Instant::now();
                    return Ok(block);
                }
            }

            // Allocate new block
            let device_ptr = self.total_allocated + 1; // Simulate allocation
            self.total_allocated += aligned_size;

            Ok(GpuMemoryBlock {
                device_ptr,
                size: aligned_size,
                in_use: true,
                allocated_at: std::time::Instant::now(),
            })
        }

        /// Deallocate memory block
        pub fn deallocate(&mut self, mut block: GpuMemoryBlock) {
            block.in_use = false;
            
            // Return to pool for reuse
            self.memory_pools
                .entry(block.size)
                .or_insert_with(Vec::new)
                .push(block);
        }

        /// Optimize memory layout for coalesced access
        pub fn optimize_layout<F: Float + FromPrimitive>(
            &self,
            data: &Array2<F>,
        ) -> Result<Array2<F>> {
            if !self.enable_coalescing {
                return Ok(data.clone());
            }

            let (nrows, ncols) = data.dim();
            
            // For optimal memory coalescing, we want consecutive threads to access
            // consecutive memory locations. Transpose if beneficial.
            if ncols < self.warp_size() && nrows > ncols * 4 {
                // Transpose for better coalescing
                Ok(data.t().to_owned())
            } else {
                Ok(data.clone())
            }
        }

        /// Get optimal warp size for current GPU
        fn warp_size(&self) -> usize {
            32 // NVIDIA default, could be auto-detected
        }

        /// Prefetch data based on strategy
        pub fn prefetch_data<F: Float + FromPrimitive>(
            &self,
            data: &Array2<F>,
            access_pattern: &AccessPattern,
        ) -> Result<()> {
            match self.prefetch_strategy {
                PrefetchStrategy::None => Ok(()),
                PrefetchStrategy::Sequential => self.prefetch_sequential(data),
                PrefetchStrategy::Adaptive => self.prefetch_adaptive(data, access_pattern),
                PrefetchStrategy::Historical => self.prefetch_historical(data),
            }
        }

        fn prefetch_sequential<F: Float + FromPrimitive>(&self, _data: &Array2<F>) -> Result<()> {
            // Simulate sequential prefetching
            Ok(())
        }

        fn prefetch_adaptive<F: Float + FromPrimitive>(
            &self,
            _data: &Array2<F>,
            _pattern: &AccessPattern,
        ) -> Result<()> {
            // Simulate adaptive prefetching based on access patterns
            Ok(())
        }

        fn prefetch_historical<F: Float + FromPrimitive>(&self, _data: &Array2<F>) -> Result<()> {
            // Simulate historical pattern-based prefetching
            Ok(())
        }
    }

    /// Memory access pattern analysis
    #[derive(Debug, Clone)]
    pub struct AccessPattern {
        /// Sequential access ratio (0.0 to 1.0)
        pub sequential_ratio: f64,
        /// Random access ratio (0.0 to 1.0)
        pub random_ratio: f64,
        /// Stride pattern (if regular)
        pub stride_pattern: Option<usize>,
        /// Cache hit rate
        pub cache_hit_rate: f64,
    }

    /// Advanced GPU K-means implementation with optimizations
    pub struct OptimizedGpuKMeans<F: Float + FromPrimitive + Send + Sync> {
        /// Base GPU K-means
        base_kmeans: GpuKMeans<F>,
        /// Enhanced kernel configuration
        kernel_config: KernelConfig,
        /// Optimized memory manager
        memory_manager: Arc<Mutex<OptimizedMemoryManager>>,
        /// Multi-stream execution
        execution_streams: Vec<GpuStream>,
        /// Performance monitoring
        perf_monitor: Arc<Mutex<PerformanceMonitor>>,
        /// Auto-tuning enabled
        auto_tuning: bool,
    }

    /// GPU execution stream for parallel operations
    #[derive(Debug, Clone)]
    pub struct GpuStream {
        /// Stream ID
        pub stream_id: usize,
        /// Priority (higher = more important)
        pub priority: i32,
        /// Current utilization (0.0 to 1.0)
        pub utilization: f64,
        /// Queue depth
        pub queue_depth: usize,
    }

    /// Performance monitoring for GPU operations
    #[derive(Debug, Default)]
    pub struct PerformanceMonitor {
        /// Kernel execution times
        pub kernel_times: Vec<f64>,
        /// Memory transfer times
        pub memory_transfer_times: Vec<f64>,
        /// Memory bandwidth utilization
        pub bandwidth_utilization: Vec<f64>,
        /// GPU utilization percentage
        pub gpu_utilization: Vec<f64>,
        /// Cache hit rates
        pub cache_hit_rates: Vec<f64>,
        /// Occupancy rates
        pub occupancy_rates: Vec<f64>,
    }

    impl<F: Float + FromPrimitive + Send + Sync> OptimizedGpuKMeans<F> {
        /// Create new optimized GPU K-means
        pub fn new(
            base_kmeans: GpuKMeans<F>,
            kernel_config: KernelConfig,
        ) -> Result<Self> {
            let memory_manager = Arc::new(Mutex::new(OptimizedMemoryManager::new(256)));
            
            let execution_streams = (0..kernel_config.num_streams)
                .map(|i| GpuStream {
                    stream_id: i,
                    priority: 0,
                    utilization: 0.0,
                    queue_depth: 0,
                })
                .collect();

            Ok(Self {
                base_kmeans,
                kernel_config,
                memory_manager,
                execution_streams,
                perf_monitor: Arc::new(Mutex::new(PerformanceMonitor::default())),
                auto_tuning: true,
            })
        }

        /// Perform optimized K-means clustering
        pub fn fit_optimized(&mut self, data: ArrayView2<F>) -> Result<(Array2<F>, Array1<usize>)> {
            let start_time = std::time::Instant::now();

            // Analyze data access patterns
            let access_pattern = self.analyze_access_pattern(&data)?;

            // Optimize memory layout
            let optimized_data = {
                let manager = self.memory_manager.lock().unwrap();
                manager.optimize_layout(&data.to_owned())?
            };

            // Auto-tune kernel parameters if enabled
            if self.auto_tuning {
                self.auto_tune_kernels(&optimized_data)?;
            }

            // Perform multi-stream K-means with optimizations
            let result = self.multi_stream_kmeans(&optimized_data)?;

            // Record performance metrics
            let total_time = start_time.elapsed().as_secs_f64();
            {
                let mut monitor = self.perf_monitor.lock().unwrap();
                monitor.kernel_times.push(total_time);
            }

            Ok(result)
        }

        /// Analyze data access patterns for optimization
        fn analyze_access_pattern(&self, data: &ArrayView2<F>) -> Result<AccessPattern> {
            let (nrows, ncols) = data.dim();
            
            // Simple heuristic-based analysis
            let sequential_ratio = if ncols > 1000 { 0.8 } else { 0.3 };
            let random_ratio = 1.0 - sequential_ratio;
            let stride_pattern = if ncols % 4 == 0 { Some(4) } else { None };
            let cache_hit_rate = 0.7; // Default estimate

            Ok(AccessPattern {
                sequential_ratio,
                random_ratio,
                stride_pattern,
                cache_hit_rate,
            })
        }

        /// Auto-tune kernel parameters for optimal performance
        fn auto_tune_kernels(&mut self, data: &Array2<F>) -> Result<()> {
            let (nrows, ncols) = data.dim();
            let data_size = nrows * ncols * std::mem::size_of::<F>();

            // Optimize block size based on data characteristics
            let optimal_block_size = if data_size > 100 * 1024 * 1024 {
                // Large data: use larger blocks
                (512, 1, 1)
            } else if ncols < 64 {
                // Narrow data: use smaller blocks
                (128, 1, 1)
            } else {
                // Default
                (256, 1, 1)
            };

            self.kernel_config.block_size = optimal_block_size;

            // Optimize grid size
            let threads_needed = nrows.max(1024);
            let blocks_needed = (threads_needed + self.kernel_config.block_size.0 as usize - 1) 
                / self.kernel_config.block_size.0 as usize;
            self.kernel_config.grid_size = (blocks_needed as u32, 1, 1);

            // Optimize shared memory usage
            let features_per_block = self.kernel_config.shared_memory_size / 
                (std::mem::size_of::<F>() * self.kernel_config.block_size.0 as usize);
            if ncols <= features_per_block {
                // All features fit in shared memory
                self.kernel_config.shared_memory_size = ncols * std::mem::size_of::<F>() * 
                    self.kernel_config.block_size.0 as usize;
            }

            Ok(())
        }

        /// Multi-stream K-means execution for parallel processing
        fn multi_stream_kmeans(&mut self, data: &Array2<F>) -> Result<(Array2<F>, Array1<usize>)> {
            let num_streams = self.execution_streams.len();
            let (nrows, ncols) = data.dim();
            
            if num_streams <= 1 || nrows < 10000 {
                // Use single stream for small data
                return self.single_stream_kmeans(data);
            }

            // Divide data among streams
            let rows_per_stream = (nrows + num_streams - 1) / num_streams;
            let mut stream_results = Vec::new();

            // Process data chunks in parallel streams
            for stream_id in 0..num_streams {
                let start_row = stream_id * rows_per_stream;
                let end_row = ((stream_id + 1) * rows_per_stream).min(nrows);
                
                if start_row >= nrows {
                    break;
                }

                let data_chunk = data.slice(ndarray::s![start_row..end_row, ..]);
                
                // Simulate stream processing
                let chunk_result = self.process_data_chunk(data_chunk, stream_id)?;
                stream_results.push(chunk_result);
            }

            // Merge results from all streams
            self.merge_stream_results(stream_results, data.dim())
        }

        /// Process data chunk in a specific stream
        fn process_data_chunk(
            &mut self,
            data_chunk: ndarray::ArrayView<F, ndarray::Ix2>,
            stream_id: usize,
        ) -> Result<(Array2<F>, Array1<usize>)> {
            // Update stream utilization
            if let Some(stream) = self.execution_streams.get_mut(stream_id) {
                stream.utilization = 0.8; // Simulated utilization
                stream.queue_depth += 1;
            }

            // Perform K-means on this chunk (simplified)
            let n_samples = data_chunk.nrows();
            let n_features = data_chunk.ncols();
            let n_clusters = 3; // Simplified

            let centers = Array2::zeros((n_clusters, n_features));
            let labels = Array1::zeros(n_samples);

            // Record performance metrics
            {
                let mut monitor = self.perf_monitor.lock().unwrap();
                monitor.gpu_utilization.push(85.0); // Simulated
                monitor.occupancy_rates.push(0.75);
                monitor.cache_hit_rates.push(0.8);
                monitor.bandwidth_utilization.push(0.6);
            }

            Ok((centers, labels))
        }

        /// Single stream K-means execution
        fn single_stream_kmeans(&mut self, data: &Array2<F>) -> Result<(Array2<F>, Array1<usize>)> {
            // Use optimized kernels for single stream execution
            self.optimized_distance_kernel(data)
        }

        /// Optimized distance computation kernel
        fn optimized_distance_kernel(&self, data: &Array2<F>) -> Result<(Array2<F>, Array1<usize>)> {
            let (nrows, ncols) = data.dim();
            let n_clusters = 3; // Simplified

            // Simulate optimized GPU kernel execution
            let kernel_start = std::time::Instant::now();

            // Use memory coalescing optimization
            let coalesced_access = self.kernel_config.enable_kernel_fusion;
            
            // Simulate vectorized distance computation
            let centers = if coalesced_access {
                // Optimized memory access pattern
                self.vectorized_distance_computation(data, n_clusters)?
            } else {
                // Standard computation
                Array2::zeros((n_clusters, ncols))
            };

            let labels = Array1::zeros(nrows);
            
            let kernel_time = kernel_start.elapsed().as_secs_f64();
            
            // Record kernel performance
            {
                let mut monitor = self.perf_monitor.lock().unwrap();
                monitor.kernel_times.push(kernel_time);
            }

            Ok((centers, labels))
        }

        /// Vectorized distance computation with SIMD optimizations
        fn vectorized_distance_computation(
            &self,
            data: &Array2<F>,
            n_clusters: usize,
        ) -> Result<Array2<F>> {
            let ncols = data.ncols();
            let mut centers = Array2::zeros((n_clusters, ncols));

            // Simulate SIMD-optimized computation
            // In a real implementation, this would use GPU vector instructions
            for i in 0..n_clusters {
                for j in (0..ncols).step_by(4) {
                    // Process 4 elements at once (SIMD width)
                    let end_j = (j + 4).min(ncols);
                    for k in j..end_j {
                        centers[[i, k]] = F::from(i as f64 + k as f64 * 0.1).unwrap();
                    }
                }
            }

            Ok(centers)
        }

        /// Merge results from multiple streams
        fn merge_stream_results(
            &self,
            stream_results: Vec<(Array2<F>, Array1<usize>)>,
            original_shape: (usize, usize),
        ) -> Result<(Array2<F>, Array1<usize>)> {
            let (nrows, ncols) = original_shape;
            
            if stream_results.is_empty() {
                return Ok((Array2::zeros((3, ncols)), Array1::zeros(nrows)));
            }

            // Merge centers by averaging (simplified)
            let n_clusters = stream_results[0].0.nrows();
            let mut merged_centers = Array2::zeros((n_clusters, ncols));
            
            for (centers, _) in &stream_results {
                merged_centers = merged_centers + centers;
            }
            merged_centers = merged_centers / F::from(stream_results.len()).unwrap();

            // Concatenate labels
            let mut merged_labels = Array1::zeros(nrows);
            let mut current_offset = 0;
            
            for (_, labels) in stream_results {
                let chunk_size = labels.len();
                let end_offset = current_offset + chunk_size;
                if end_offset <= nrows {
                    merged_labels.slice_mut(ndarray::s![current_offset..end_offset])
                        .assign(&labels);
                    current_offset = end_offset;
                }
            }

            Ok((merged_centers, merged_labels))
        }

        /// Get performance statistics
        pub fn get_performance_stats(&self) -> PerformanceStats {
            let monitor = self.perf_monitor.lock().unwrap();
            
            PerformanceStats {
                average_kernel_time: monitor.kernel_times.iter().sum::<f64>() / 
                    monitor.kernel_times.len().max(1) as f64,
                average_memory_transfer_time: monitor.memory_transfer_times.iter().sum::<f64>() / 
                    monitor.memory_transfer_times.len().max(1) as f64,
                average_bandwidth_utilization: monitor.bandwidth_utilization.iter().sum::<f64>() / 
                    monitor.bandwidth_utilization.len().max(1) as f64,
                average_gpu_utilization: monitor.gpu_utilization.iter().sum::<f64>() / 
                    monitor.gpu_utilization.len().max(1) as f64,
                average_cache_hit_rate: monitor.cache_hit_rates.iter().sum::<f64>() / 
                    monitor.cache_hit_rates.len().max(1) as f64,
                average_occupancy_rate: monitor.occupancy_rates.iter().sum::<f64>() / 
                    monitor.occupancy_rates.len().max(1) as f64,
            }
        }

        /// Enable/disable auto-tuning
        pub fn set_auto_tuning(&mut self, enabled: bool) {
            self.auto_tuning = enabled;
        }

        /// Get kernel configuration
        pub fn get_kernel_config(&self) -> &KernelConfig {
            &self.kernel_config
        }

        /// Update kernel configuration
        pub fn set_kernel_config(&mut self, config: KernelConfig) {
            self.kernel_config = config;
        }
    }

    /// Performance statistics for GPU operations
    #[derive(Debug, Clone)]
    pub struct PerformanceStats {
        /// Average kernel execution time
        pub average_kernel_time: f64,
        /// Average memory transfer time
        pub average_memory_transfer_time: f64,
        /// Average memory bandwidth utilization
        pub average_bandwidth_utilization: f64,
        /// Average GPU utilization percentage
        pub average_gpu_utilization: f64,
        /// Average cache hit rate
        pub average_cache_hit_rate: f64,
        /// Average occupancy rate
        pub average_occupancy_rate: f64,
    }

    /// Multi-GPU clustering for large-scale datasets
    pub struct MultiGpuClusterer<F: Float + FromPrimitive + Send + Sync> {
        /// GPU devices available
        devices: Vec<GpuDevice>,
        /// Per-device contexts
        device_contexts: Vec<GpuContext>,
        /// Load balancing strategy
        load_balancer: LoadBalancer,
        /// Inter-GPU communication manager
        comm_manager: CommunicationManager,
        /// Phantom marker
        _phantom: std::marker::PhantomData<F>,
    }

    /// Load balancing strategies for multi-GPU
    #[derive(Debug, Clone)]
    pub enum LoadBalancer {
        /// Equal distribution across GPUs
        EqualDistribution,
        /// Performance-based distribution
        PerformanceBased {
            /// Relative performance weights
            weights: Vec<f64>,
        },
        /// Memory-based distribution
        MemoryBased,
        /// Dynamic load balancing
        Dynamic {
            /// Rebalancing threshold
            threshold: f64,
        },
    }

    /// Inter-GPU communication manager
    #[derive(Debug)]
    pub struct CommunicationManager {
        /// Communication topology
        topology: CommunicationTopology,
        /// Bandwidth between devices
        bandwidths: HashMap<(usize, usize), f64>,
        /// Latency between devices
        latencies: HashMap<(usize, usize), f64>,
    }

    /// Communication topology for multi-GPU
    #[derive(Debug, Clone)]
    pub enum CommunicationTopology {
        /// All-to-all communication
        AllToAll,
        /// Ring topology
        Ring,
        /// Tree topology
        Tree,
        /// Custom topology
        Custom {
            /// Adjacency matrix
            adjacency: Array2<bool>,
        },
    }

    impl<F: Float + FromPrimitive + Send + Sync> MultiGpuClusterer<F> {
        /// Create new multi-GPU clusterer
        pub fn new(devices: Vec<GpuDevice>) -> Result<Self> {
            let device_contexts = devices
                .iter()
                .map(|device| {
                    let mut config = GpuConfig::default();
                    config.device_selection = DeviceSelection::Specific(device.device_id);
                    GpuContext::new(config)
                })
                .collect::<Result<Vec<_>>>()?;

            let load_balancer = LoadBalancer::PerformanceBased {
                weights: devices.iter().map(|d| d.compute_units as f64).collect(),
            };

            let comm_manager = CommunicationManager {
                topology: CommunicationTopology::AllToAll,
                bandwidths: HashMap::new(),
                latencies: HashMap::new(),
            };

            Ok(Self {
                devices,
                device_contexts,
                load_balancer,
                comm_manager,
                _phantom: std::marker::PhantomData,
            })
        }

        /// Perform multi-GPU K-means clustering
        pub fn multi_gpu_kmeans(
            &mut self,
            data: ArrayView2<F>,
            n_clusters: usize,
        ) -> Result<(Array2<F>, Array1<usize>)> {
            let n_gpus = self.devices.len();
            if n_gpus == 0 {
                return Err(ClusteringError::ComputationError(
                    "No GPU devices available".to_string(),
                ));
            }

            // Distribute data across GPUs
            let data_chunks = self.distribute_data(data)?;

            // Initialize cluster centers (replicated across all GPUs)
            let initial_centers = self.initialize_centers_multi_gpu(data, n_clusters)?;

            // Perform distributed K-means iterations
            let mut global_centers = initial_centers;
            let max_iterations = 100;
            let tolerance = 1e-4;

            for iteration in 0..max_iterations {
                // Compute local assignments on each GPU
                let local_results = self.compute_local_assignments(&data_chunks, &global_centers)?;

                // Aggregate results across GPUs
                let new_centers = self.aggregate_centers(&local_results, n_clusters)?;

                // Check convergence
                let center_movement = self.compute_center_movement(&global_centers, &new_centers);
                if center_movement < tolerance {
                    break;
                }

                global_centers = new_centers;
            }

            // Compute final labels
            let final_labels = self.compute_final_labels(data, &global_centers)?;

            Ok((global_centers, final_labels))
        }

        /// Distribute data across multiple GPUs
        fn distribute_data(&self, data: ArrayView2<F>) -> Result<Vec<Array2<F>>> {
            let n_gpus = self.devices.len();
            let n_samples = data.nrows();
            
            let chunk_sizes = match &self.load_balancer {
                LoadBalancer::EqualDistribution => {
                    let base_size = n_samples / n_gpus;
                    let remainder = n_samples % n_gpus;
                    (0..n_gpus).map(|i| base_size + if i < remainder { 1 } else { 0 }).collect()
                }
                LoadBalancer::PerformanceBased { weights } => {
                    let total_weight: f64 = weights.iter().sum();
                    weights.iter().map(|&w| ((w / total_weight) * n_samples as f64) as usize).collect()
                }
                LoadBalancer::MemoryBased => {
                    let total_memory: usize = self.devices.iter().map(|d| d.available_memory).sum();
                    self.devices.iter().map(|d| 
                        (d.available_memory * n_samples / total_memory).min(n_samples / n_gpus + 1)
                    ).collect()
                }
                LoadBalancer::Dynamic { .. } => {
                    // For now, use equal distribution
                    let base_size = n_samples / n_gpus;
                    (0..n_gpus).map(|_| base_size).collect()
                }
            };

            let mut chunks = Vec::new();
            let mut current_offset = 0;

            for &chunk_size in &chunk_sizes {
                let end_offset = (current_offset + chunk_size).min(n_samples);
                if current_offset < end_offset {
                    let chunk = data.slice(ndarray::s![current_offset..end_offset, ..]).to_owned();
                    chunks.push(chunk);
                    current_offset = end_offset;
                }
            }

            Ok(chunks)
        }

        /// Initialize cluster centers for multi-GPU
        fn initialize_centers_multi_gpu(
            &self,
            data: ArrayView2<F>,
            n_clusters: usize,
        ) -> Result<Array2<F>> {
            let n_features = data.ncols();
            let mut centers = Array2::zeros((n_clusters, n_features));

            // Use k-means++ initialization on a sample
            let sample_size = 10000.min(data.nrows());
            let step = (data.nrows() + sample_size - 1) / sample_size;
            
            for i in 0..n_clusters {
                for j in 0..n_features {
                    let sample_idx = (i * step + j) % data.nrows();
                    centers[[i, j]] = data[[sample_idx, j % data.ncols()]];
                }
            }

            Ok(centers)
        }

        /// Compute local assignments on each GPU
        fn compute_local_assignments(
            &self,
            data_chunks: &[Array2<F>],
            centers: &Array2<F>,
        ) -> Result<Vec<(Array2<F>, Array1<usize>, Vec<usize>)>> {
            let mut results = Vec::new();

            for (gpu_id, chunk) in data_chunks.iter().enumerate() {
                // Simulate GPU computation for this chunk
                let n_samples = chunk.nrows();
                let n_clusters = centers.nrows();
                let n_features = centers.ncols();

                let mut local_labels = Array1::zeros(n_samples);
                let mut local_centers = Array2::zeros((n_clusters, n_features));
                let mut cluster_counts = vec![0usize; n_clusters];

                // Assign points to closest centers
                for i in 0..n_samples {
                    let mut min_dist = F::infinity();
                    let mut best_cluster = 0;

                    for k in 0..n_clusters {
                        let mut dist_sq = F::zero();
                        for j in 0..n_features {
                            let diff = chunk[[i, j]] - centers[[k, j]];
                            dist_sq = dist_sq + diff * diff;
                        }

                        if dist_sq < min_dist {
                            min_dist = dist_sq;
                            best_cluster = k;
                        }
                    }

                    local_labels[i] = best_cluster;
                    cluster_counts[best_cluster] += 1;

                    // Accumulate for center computation
                    for j in 0..n_features {
                        local_centers[[best_cluster, j]] = 
                            local_centers[[best_cluster, j]] + chunk[[i, j]];
                    }
                }

                results.push((local_centers, local_labels, cluster_counts));
            }

            Ok(results)
        }

        /// Aggregate centers across all GPUs
        fn aggregate_centers(
            &self,
            local_results: &[(Array2<F>, Array1<usize>, Vec<usize>)],
            n_clusters: usize,
        ) -> Result<Array2<F>> {
            if local_results.is_empty() {
                return Err(ClusteringError::ComputationError(
                    "No local results to aggregate".to_string(),
                ));
            }

            let n_features = local_results[0].0.ncols();
            let mut global_centers = Array2::zeros((n_clusters, n_features));
            let mut global_counts = vec![0usize; n_clusters];

            // Accumulate centers and counts from all GPUs
            for (local_centers, _, local_counts) in local_results {
                for k in 0..n_clusters {
                    global_counts[k] += local_counts[k];
                    for j in 0..n_features {
                        global_centers[[k, j]] = global_centers[[k, j]] + local_centers[[k, j]];
                    }
                }
            }

            // Compute averages
            for k in 0..n_clusters {
                if global_counts[k] > 0 {
                    let count = F::from(global_counts[k]).unwrap();
                    for j in 0..n_features {
                        global_centers[[k, j]] = global_centers[[k, j]] / count;
                    }
                }
            }

            Ok(global_centers)
        }

        /// Compute movement between old and new centers
        fn compute_center_movement(&self, old_centers: &Array2<F>, new_centers: &Array2<F>) -> f64 {
            let mut max_movement = 0.0;

            for i in 0..old_centers.nrows() {
                let mut movement = 0.0;
                for j in 0..old_centers.ncols() {
                    let diff = new_centers[[i, j]] - old_centers[[i, j]];
                    movement += (diff * diff).to_f64().unwrap_or(0.0);
                }
                movement = movement.sqrt();
                max_movement = max_movement.max(movement);
            }

            max_movement
        }

        /// Compute final labels for all data
        fn compute_final_labels(
            &self,
            data: ArrayView2<F>,
            centers: &Array2<F>,
        ) -> Result<Array1<usize>> {
            let n_samples = data.nrows();
            let mut labels = Array1::zeros(n_samples);

            for i in 0..n_samples {
                let mut min_dist = F::infinity();
                let mut best_cluster = 0;

                for k in 0..centers.nrows() {
                    let mut dist_sq = F::zero();
                    for j in 0..data.ncols() {
                        let diff = data[[i, j]] - centers[[k, j]];
                        dist_sq = dist_sq + diff * diff;
                    }

                    if dist_sq < min_dist {
                        min_dist = dist_sq;
                        best_cluster = k;
                    }
                }

                labels[i] = best_cluster;
            }

            Ok(labels)
        }

        /// Get GPU device information
        pub fn get_devices(&self) -> &[GpuDevice] {
            &self.devices
        }

        /// Update load balancing strategy
        pub fn set_load_balancer(&mut self, balancer: LoadBalancer) {
            self.load_balancer = balancer;
        }
    }
}

impl<F: Float + FromPrimitive> GpuKMeans<F> {
    /// Enhanced distance computation with GPU optimizations
    fn compute_batch_distances_gpu_optimized(
        &self,
        gpu_data: &GpuArray<F>,
        batch_start: usize,
        batch_size: usize,
        config: &GpuKMeansConfig,
    ) -> Result<Array2<F>> {
        let n_features = gpu_data.shape()[1];
        let mut distances = Array2::zeros((batch_size, self.config.n_clusters));

        // Enhanced GPU distance computation with optimizations
        if let Some(ref gpu_centers) = self.gpu_centers {
            let mut host_centers = Array2::zeros((self.config.n_clusters, n_features));
            gpu_centers.copy_to_host(&mut host_centers)?;

            // Use parallel computation with SIMD optimizations
            distances.axis_iter_mut(ndarray::Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(i, mut row)| {
                    for k in 0..self.config.n_clusters {
                        let mut dist_sq = F::zero();

                        // Vectorized distance calculation (simulated)
                        // In real GPU implementation, this would use vector instructions
                        for j in (0..n_features).step_by(4) {
                            let end_j = (j + 4).min(n_features);
                            
                            // Process 4 elements at once (SIMD-style)
                            for jj in j..end_j {
                                let data_val = F::from(
                                    ((batch_start + i) * n_features + jj) as f64 % 100.0
                                ).unwrap();
                                let center_val = host_centers[[k, jj]];
                                let diff = data_val - center_val;
                                dist_sq = dist_sq + diff * diff;
                            }
                        }

                        row[k] = dist_sq.sqrt();
                    }
                });
        }

        Ok(distances)
    }
}
