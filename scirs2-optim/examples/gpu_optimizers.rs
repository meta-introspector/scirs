//! Example demonstrating GPU-accelerated optimizers
//!
//! This example shows how to use GPU-accelerated Adam and LAMB optimizers
//! for training large models with improved performance.

use ndarray::{Array1, Array2};
use scirs2_optim::{AdamGpu, LAMBGpu, GpuOptimizer, GpuOptimizerConfig, Optimizer};
use scirs2_core::gpu::GpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GPU-Accelerated Optimizers Example");
    println!("==================================\n");

    // Check available GPU backends
    check_gpu_availability();

    // Example 1: GPU-accelerated Adam optimizer
    println!("\n1. GPU-Accelerated Adam Optimizer");
    println!("---------------------------------");
    gpu_adam_example()?;

    // Example 2: GPU-accelerated LAMB optimizer
    println!("\n2. GPU-Accelerated LAMB Optimizer");
    println!("---------------------------------");
    gpu_lamb_example()?;

    // Example 3: Mixed precision training
    println!("\n3. Mixed Precision Training with GPU");
    println!("------------------------------------");
    mixed_precision_example()?;

    // Example 4: Multi-GPU training simulation
    println!("\n4. Multi-GPU Training Simulation");
    println!("--------------------------------");
    multi_gpu_example()?;

    Ok(())
}

/// Check available GPU backends
fn check_gpu_availability() {
    println!("Checking GPU availability...");
    
    let backends = [
        GpuBackend::Cuda,
        GpuBackend::Metal,
        GpuBackend::Rocm,
        GpuBackend::Wgpu,
    ];
    
    for backend in &backends {
        let available = backend.is_available();
        println!("  {}: {}", backend, if available { "Available" } else { "Not available" });
    }
    
    let preferred = GpuBackend::preferred();
    println!("\nPreferred backend: {}", preferred);
}

/// Example using GPU-accelerated Adam optimizer
fn gpu_adam_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create model parameters and gradients
    let param_size = 10_000; // Large parameter vector
    let mut params = Array1::from_elem(param_size, 1.0f32);
    let gradients = Array1::from_elem(param_size, 0.01f32);
    
    // Create GPU-accelerated Adam optimizer
    let mut optimizer = AdamGpu::new(0.001);
    
    // Configure GPU settings
    let gpu_config = GpuOptimizerConfig {
        backend: GpuBackend::preferred(),
        mixed_precision: false,
        memory_pool_size: 256 * 1024 * 1024, // 256MB
        ..Default::default()
    };
    
    // Initialize GPU resources
    match optimizer.initialize_gpu(param_size, gpu_config) {
        Ok(()) => {
            println!("Successfully initialized GPU for Adam optimizer");
            
            // Move optimizer to GPU
            optimizer.to_gpu()?;
            
            // Perform optimization steps
            println!("Running 10 optimization steps on GPU...");
            for step in 0..10 {
                // In real training, gradients would be computed from loss
                let updated = optimizer.step(&params, &gradients)?;
                params = updated;
                
                if step % 5 == 0 {
                    println!("  Step {}: param mean = {:.6}", step, params.mean().unwrap());
                }
            }
            
            // Move back to CPU if needed
            optimizer.to_cpu()?;
        }
        Err(e) => {
            println!("GPU initialization failed: {}", e);
            println!("Falling back to CPU execution...");
            
            // CPU fallback
            for step in 0..10 {
                let updated = optimizer.step(&params, &gradients)?;
                params = updated;
                
                if step % 5 == 0 {
                    println!("  Step {} (CPU): param mean = {:.6}", step, params.mean().unwrap());
                }
            }
        }
    }
    
    Ok(())
}

/// Example using GPU-accelerated LAMB optimizer
fn gpu_lamb_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create larger model parameters for LAMB (suitable for large batch training)
    let param_size = 50_000;
    let mut params = Array1::linspace(0.1f32, 2.0, param_size);
    let gradients = Array1::from_elem(param_size, 0.001f32);
    
    // Create GPU-accelerated LAMB optimizer
    let mut optimizer = LAMBGpu::new_with_config(
        0.001,  // learning rate
        0.9,    // beta1
        0.999,  // beta2
        1e-6,   // epsilon
        0.01,   // weight decay
    );
    
    // Configure GPU settings
    let gpu_config = GpuOptimizerConfig {
        backend: GpuBackend::preferred(),
        gradient_clipping: true,
        max_grad_norm: 1.0,
        ..Default::default()
    };
    
    // Initialize GPU
    match optimizer.initialize_gpu(param_size, gpu_config) {
        Ok(()) => {
            println!("Successfully initialized GPU for LAMB optimizer");
            
            optimizer.to_gpu()?;
            
            // Simulate large batch training
            println!("Simulating large batch training with LAMB...");
            for epoch in 0..3 {
                // Simulate gradient accumulation from large batch
                let large_batch_grad = gradients.clone() * (epoch as f32 + 1.0);
                
                let updated = optimizer.step(&params, &large_batch_grad)?;
                params = updated;
                
                println!("  Epoch {}: param norm = {:.6}", epoch, params.dot(&params).sqrt());
            }
        }
        Err(e) => {
            println!("GPU initialization failed: {}", e);
            println!("Using CPU fallback for LAMB");
        }
    }
    
    Ok(())
}

/// Example of mixed precision training
fn mixed_precision_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("Mixed precision training simulation");
    
    // Model with different parameter groups
    let layer1_size = 1024 * 768;  // Transformer-like layer
    let layer2_size = 768 * 3072;  // Feed-forward layer
    
    let mut layer1_params = Array1::from_elem(layer1_size, 0.02f32);
    let mut layer2_params = Array1::from_elem(layer2_size, 0.01f32);
    
    // Gradients (would come from backprop in real training)
    let layer1_grads = Array1::from_elem(layer1_size, 1e-4f32);
    let layer2_grads = Array1::from_elem(layer2_size, 5e-5f32);
    
    // Create optimizer with mixed precision config
    let mut optimizer1 = AdamGpu::new(0.001);
    let mut optimizer2 = AdamGpu::new(0.001);
    
    let mp_config = GpuOptimizerConfig {
        backend: GpuBackend::preferred(),
        mixed_precision: true,
        loss_scale: 1024.0,
        dynamic_loss_scaling: true,
        ..Default::default()
    };
    
    // Initialize both optimizers
    if optimizer1.initialize_gpu(layer1_size, mp_config.clone()).is_ok() &&
       optimizer2.initialize_gpu(layer2_size, mp_config).is_ok() {
        
        optimizer1.to_gpu()?;
        optimizer2.to_gpu()?;
        
        println!("Running mixed precision training step...");
        
        // Update both layers
        layer1_params = optimizer1.step(&layer1_params, &layer1_grads)?;
        layer2_params = optimizer2.step(&layer2_params, &layer2_grads)?;
        
        println!("  Layer 1 param mean: {:.8}", layer1_params.mean().unwrap());
        println!("  Layer 2 param mean: {:.8}", layer2_params.mean().unwrap());
        println!("  Loss scale: 1024.0 (dynamic scaling enabled)");
    } else {
        println!("Mixed precision GPU setup failed, using CPU");
    }
    
    Ok(())
}

/// Example simulating multi-GPU training
fn multi_gpu_example() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_optim::gpu::lamb_gpu::BatchLAMBGpu;
    
    println!("Simulating multi-GPU training with 4 GPUs");
    
    // Create parameter groups (simulating model parallel training)
    let num_groups = 4;
    let group_size = 100_000;
    
    let mut param_groups: Vec<Array1<f32>> = (0..num_groups)
        .map(|i| Array1::from_elem(group_size, 0.1 * (i as f32 + 1.0)))
        .collect();
    
    let grad_groups: Vec<Array1<f32>> = (0..num_groups)
        .map(|_| Array1::from_elem(group_size, 1e-3))
        .collect();
    
    // Create batch LAMB optimizer for multi-GPU
    let mut batch_optimizer = BatchLAMBGpu::new(0.001, num_groups);
    
    // Initialize for each GPU
    let group_sizes: Vec<usize> = vec![group_size; num_groups];
    
    match batch_optimizer.initialize_gpu(&group_sizes) {
        Ok(()) => {
            println!("Successfully initialized {} GPUs", num_groups);
            
            // Simulate distributed training step
            println!("Running distributed optimization step...");
            
            // Convert to mutable slices
            let mut param_refs: Vec<_> = param_groups.iter_mut().collect();
            let grad_refs: Vec<_> = grad_groups.iter().collect();
            
            // Update all parameter groups
            if let Err(e) = batch_optimizer.step_all(&mut param_refs, &grad_refs) {
                println!("Multi-GPU step failed: {}", e);
            } else {
                // Show results
                for (i, params) in param_groups.iter().enumerate() {
                    println!("  GPU {}: param norm = {:.6}", i, params.dot(&params).sqrt());
                }
            }
        }
        Err(e) => {
            println!("Multi-GPU initialization failed: {}", e);
        }
    }
    
    Ok(())
}

/// Benchmark GPU vs CPU performance
#[allow(dead_code)]
fn benchmark_gpu_vs_cpu() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;
    
    let sizes = vec![1000, 10_000, 100_000, 1_000_000];
    
    println!("\nGPU vs CPU Performance Comparison");
    println!("=================================");
    println!("Size      | CPU Time  | GPU Time  | Speedup");
    println!("----------|-----------|-----------|--------");
    
    for size in sizes {
        let params = Array1::from_elem(size, 1.0f32);
        let grads = Array1::from_elem(size, 0.01f32);
        
        // CPU timing
        let mut cpu_opt = scirs2_optim::Adam::new(0.001);
        let cpu_start = Instant::now();
        for _ in 0..100 {
            let _ = cpu_opt.step(&params, &grads)?;
        }
        let cpu_time = cpu_start.elapsed();
        
        // GPU timing
        let mut gpu_opt = AdamGpu::new(0.001);
        let gpu_config = GpuOptimizerConfig::default();
        
        let gpu_time = if gpu_opt.initialize_gpu(size, gpu_config).is_ok() {
            gpu_opt.to_gpu()?;
            let gpu_start = Instant::now();
            for _ in 0..100 {
                let _ = gpu_opt.step(&params, &grads)?;
            }
            gpu_start.elapsed()
        } else {
            cpu_time // Fallback to CPU time if GPU not available
        };
        
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        
        println!("{:9} | {:9.3}ms | {:9.3}ms | {:.2}x",
            size,
            cpu_time.as_secs_f64() * 1000.0,
            gpu_time.as_secs_f64() * 1000.0,
            speedup
        );
    }
    
    Ok(())
}