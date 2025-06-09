//! GPU backend detection and initialization utilities
//!
//! This module provides utilities for detecting available GPU backends
//! and initializing them appropriately.

use crate::gpu::{GpuBackend, GpuError};
use std::process::Command;

/// Information about available GPU hardware
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// The GPU backend type
    pub backend: GpuBackend,
    /// Device name
    pub device_name: String,
    /// Available memory in bytes
    pub memory_bytes: Option<u64>,
    /// Compute capability or equivalent
    pub compute_capability: Option<String>,
    /// Whether the device supports tensor operations
    pub supports_tensors: bool,
}

/// Detection results for all available GPU backends
#[derive(Debug, Clone)]
pub struct GpuDetectionResult {
    /// Available GPU devices
    pub devices: Vec<GpuInfo>,
    /// Recommended backend for scientific computing
    pub recommended_backend: GpuBackend,
}

/// Detect available GPU backends and devices
pub fn detect_gpu_backends() -> GpuDetectionResult {
    let mut devices = Vec::new();

    // Detect CUDA devices
    if let Ok(cuda_devices) = detect_cuda_devices() {
        devices.extend(cuda_devices);
    }

    // Detect Metal devices (macOS)
    #[cfg(target_os = "macos")]
    if let Ok(metal_devices) = detect_metal_devices() {
        devices.extend(metal_devices);
    }

    // Detect OpenCL devices
    if let Ok(opencl_devices) = detect_opencl_devices() {
        devices.extend(opencl_devices);
    }

    // Determine recommended backend
    let recommended_backend = if devices.iter().any(|d| d.backend == GpuBackend::Cuda) {
        GpuBackend::Cuda
    } else if devices.iter().any(|d| d.backend == GpuBackend::Metal) {
        GpuBackend::Metal
    } else if devices.iter().any(|d| d.backend == GpuBackend::OpenCL) {
        GpuBackend::OpenCL
    } else {
        GpuBackend::Cpu
    };

    // Always add CPU fallback
    devices.push(GpuInfo {
        backend: GpuBackend::Cpu,
        device_name: "CPU".to_string(),
        memory_bytes: None,
        compute_capability: None,
        supports_tensors: false,
    });

    GpuDetectionResult {
        devices,
        recommended_backend,
    }
}

/// Detect CUDA devices using nvidia-ml-py or nvidia-smi
fn detect_cuda_devices() -> Result<Vec<GpuInfo>, GpuError> {
    let mut devices = Vec::new();

    // Try to run nvidia-smi to detect CUDA devices
    match Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total,compute_capability.major,compute_capability.minor")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        Ok(output) if output.status.success() => {
            let output_str = String::from_utf8_lossy(&output.stdout);

            for line in output_str.lines() {
                if line.trim().is_empty() {
                    continue;
                }

                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 4 {
                    let device_name = parts[0].to_string();
                    let memory_mb = parts[1].parse::<u64>().unwrap_or(0) * 1024 * 1024; // Convert MB to bytes
                    let compute_major = parts[2].parse::<u32>().unwrap_or(0);
                    let compute_minor = parts[3].parse::<u32>().unwrap_or(0);

                    devices.push(GpuInfo {
                        backend: GpuBackend::Cuda,
                        device_name,
                        memory_bytes: Some(memory_mb),
                        compute_capability: Some(format!("{}.{}", compute_major, compute_minor)),
                        supports_tensors: compute_major >= 7, // Tensor cores available on Volta+ (7.0+)
                    });
                }
            }
        }
        _ => {
            // nvidia-smi not available or failed
            // In a real implementation, we could try other methods like:
            // - Direct CUDA runtime API calls
            // - nvidia-ml-py if available
            // - /proc/driver/nvidia/gpus/ on Linux
        }
    }

    if devices.is_empty() {
        Err(GpuError::BackendNotAvailable("CUDA".to_string()))
    } else {
        Ok(devices)
    }
}

/// Detect Metal devices (macOS only)
#[cfg(target_os = "macos")]
fn detect_metal_devices() -> Result<Vec<GpuInfo>, GpuError> {
    let mut devices = Vec::new();

    // Try to detect Metal devices using system_profiler
    match Command::new("system_profiler")
        .arg("SPDisplaysDataType")
        .arg("-xml")
        .output()
    {
        Ok(output) if output.status.success() => {
            // In a real implementation, we would parse the XML output
            // to extract GPU information. For now, just add a generic Metal device.
            devices.push(GpuInfo {
                backend: GpuBackend::Metal,
                device_name: "Metal GPU".to_string(),
                memory_bytes: None,
                compute_capability: None,
                supports_tensors: true, // Modern Apple GPUs support tensor operations
            });
        }
        _ => {
            return Err(GpuError::BackendNotAvailable("Metal".to_string()));
        }
    }

    Ok(devices)
}

/// Detect Metal devices (non-macOS - not available)
#[cfg(not(target_os = "macos"))]
fn detect_metal_devices() -> Result<Vec<GpuInfo>, GpuError> {
    Err(GpuError::BackendNotAvailable(
        "Metal (not macOS)".to_string(),
    ))
}

/// Detect OpenCL devices
fn detect_opencl_devices() -> Result<Vec<GpuInfo>, GpuError> {
    let mut devices = Vec::new();

    // Try to detect OpenCL devices using clinfo
    match Command::new("clinfo").arg("--list").output() {
        Ok(output) if output.status.success() => {
            let output_str = String::from_utf8_lossy(&output.stdout);

            for line in output_str.lines() {
                if line.trim().starts_with("Platform") || line.trim().starts_with("Device") {
                    // In a real implementation, we would parse clinfo output properly
                    // For now, just add a generic OpenCL device
                    devices.push(GpuInfo {
                        backend: GpuBackend::OpenCL,
                        device_name: "OpenCL Device".to_string(),
                        memory_bytes: None,
                        compute_capability: None,
                        supports_tensors: false,
                    });
                    break; // Just add one for demo
                }
            }
        }
        _ => {
            return Err(GpuError::BackendNotAvailable("OpenCL".to_string()));
        }
    }

    if devices.is_empty() {
        Err(GpuError::BackendNotAvailable("OpenCL".to_string()))
    } else {
        Ok(devices)
    }
}

/// Check if a specific backend is properly installed and functional
pub fn check_backend_installation(backend: GpuBackend) -> Result<bool, GpuError> {
    match backend {
        GpuBackend::Cuda => {
            // Check for CUDA installation
            match Command::new("nvcc").arg("--version").output() {
                Ok(output) if output.status.success() => Ok(true),
                _ => Ok(false),
            }
        }
        GpuBackend::Metal => {
            #[cfg(target_os = "macos")]
            {
                // Metal is always available on macOS
                Ok(true)
            }
            #[cfg(not(target_os = "macos"))]
            {
                Ok(false)
            }
        }
        GpuBackend::OpenCL => {
            // Check for OpenCL installation
            match Command::new("clinfo").output() {
                Ok(output) if output.status.success() => Ok(true),
                _ => Ok(false),
            }
        }
        GpuBackend::Wgpu => {
            // WebGPU is always available through wgpu crate
            Ok(true)
        }
        GpuBackend::Cpu => Ok(true),
    }
}

/// Get detailed information about a specific GPU device
pub fn get_device_info(backend: GpuBackend, device_id: usize) -> Result<GpuInfo, GpuError> {
    let detection_result = detect_gpu_backends();

    detection_result
        .devices
        .into_iter()
        .filter(|d| d.backend == backend)
        .nth(device_id)
        .ok_or_else(|| {
            GpuError::InvalidParameter(format!(
                "Device {} not found for backend {}",
                device_id, backend
            ))
        })
}

/// Initialize the optimal GPU backend for the current system
pub fn initialize_optimal_backend() -> Result<GpuBackend, GpuError> {
    let detection_result = detect_gpu_backends();

    // Try backends in order of preference for scientific computing
    let preference_order = [
        GpuBackend::Cuda,   // Best for scientific computing
        GpuBackend::Metal,  // Good on Apple hardware
        GpuBackend::OpenCL, // Widely compatible
        GpuBackend::Wgpu,   // Modern cross-platform
        GpuBackend::Cpu,    // Always available fallback
    ];

    for backend in preference_order.iter() {
        if detection_result
            .devices
            .iter()
            .any(|d| d.backend == *backend)
        {
            return Ok(*backend);
        }
    }

    // Should never reach here since CPU is always available
    Ok(GpuBackend::Cpu)
}
