//! Example demonstrating advanced features added to scirs2-ndimage
//!
//! This example showcases:
//! - GPU-accelerated operations
//! - SIMD-optimized specialized filters
//! - Performance profiling and optimization
//! - SciPy-compatible API

use ndarray::{Array2, array};
use scirs2_ndimage::{
    backend::{Backend, BackendBuilder},
    filters::simd_specialized::{
        simd_bilateral_filter, simd_guided_filter, simd_adaptive_median_filter,
    },
    profiling::{
        enable_profiling, disable_profiling, display_performance_report,
        enable_memory_profiling, get_memory_report, OptimizationAdvisor,
    },
    scipy_compat::ndimage::{gaussian_filter, zoom, rotate, laplace},
    error::NdimageResult,
};

fn main() -> NdimageResult<()> {
    println!("=== Advanced scirs2-ndimage Features Demo ===\n");
    
    // Enable profiling
    enable_profiling();
    enable_memory_profiling();
    
    // Create test image
    let test_image = create_test_image();
    println!("Created test image: {}x{}", test_image.nrows(), test_image.ncols());
    
    // 1. Demonstrate GPU acceleration (if available)
    #[cfg(feature = "cuda")]
    {
        println!("\n--- GPU Acceleration Demo ---");
        demo_gpu_acceleration(&test_image)?;
    }
    
    // 2. Demonstrate SIMD-optimized filters
    println!("\n--- SIMD-Optimized Filters Demo ---");
    demo_simd_filters(&test_image)?;
    
    // 3. Demonstrate SciPy-compatible API
    println!("\n--- SciPy-Compatible API Demo ---");
    demo_scipy_compat(&test_image)?;
    
    // 4. Display performance analysis
    println!("\n--- Performance Analysis ---");
    display_performance_report();
    
    // 5. Display memory usage
    let memory_report = get_memory_report();
    memory_report.display();
    
    // 6. Get optimization recommendations
    println!("\n--- Optimization Recommendations ---");
    let profiler = scirs2_ndimage::profiling::get_performance_report();
    let mut advisor = OptimizationAdvisor::new();
    let opt_report = advisor.analyze(profiler.operation_breakdown.values()
        .flat_map(|s| std::iter::repeat(&s.mean_time).take(s.count))
        .enumerate()
        .map(|(i, duration)| scirs2_ndimage::profiling::OperationMetrics {
            name: format!("operation_{}", i),
            duration: *duration,
            memory_allocated: 0,
            memory_deallocated: 0,
            array_shape: vec![test_image.nrows(), test_image.ncols()],
            backend: Backend::Cpu,
            thread_count: 1,
            timestamp: std::time::Instant::now(),
        })
        .collect::<Vec<_>>()
        .as_slice()
    );
    opt_report.display();
    
    disable_profiling();
    
    Ok(())
}

fn create_test_image() -> Array2<f64> {
    let size = 256;
    let mut image = Array2::zeros((size, size));
    
    // Create a pattern with edges and noise
    for i in 0..size {
        for j in 0..size {
            let x = i as f64 / size as f64;
            let y = j as f64 / size as f64;
            
            // Base pattern
            image[(i, j)] = (x * std::f64::consts::PI * 4.0).sin() * 
                           (y * std::f64::consts::PI * 4.0).cos();
            
            // Add some edges
            if (i as i32 - size as i32 / 2).abs() < 10 || 
               (j as i32 - size as i32 / 2).abs() < 10 {
                image[(i, j)] = 1.0;
            }
            
            // Add noise
            if (i * j) % 7 == 0 {
                image[(i, j)] += 0.5;
            }
        }
    }
    
    image
}

#[cfg(feature = "cuda")]
fn demo_gpu_acceleration(image: &Array2<f64>) -> NdimageResult<()> {
    use scirs2_ndimage::backend::cuda::CudaOperations;
    
    println!("Setting up GPU backend...");
    
    // Create backend executor with GPU preference
    let executor = BackendBuilder::new()
        .backend(Backend::Cuda)
        .gpu_threshold(1000)
        .allow_fallback(true)
        .build()?;
    
    // Create CUDA operations handler
    let cuda_ops = CudaOperations::new(None)?;
    
    // Perform Gaussian filter on GPU
    let sigma = [2.0, 2.0];
    let gpu_result = cuda_ops.gaussian_filter_2d(&image.view(), sigma)?;
    
    println!("GPU Gaussian filter completed: {}x{}", 
             gpu_result.nrows(), gpu_result.ncols());
    
    // Compare with CPU version
    let cpu_result = gaussian_filter(image, sigma.to_vec(), None, None, None, None)?;
    
    // Calculate difference
    let diff = (&gpu_result - &cpu_result).mapv(|x| x.abs()).sum();
    println!("GPU vs CPU difference: {:.6}", diff);
    
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn demo_gpu_acceleration(_image: &Array2<f64>) -> NdimageResult<()> {
    println!("GPU acceleration not available (compile with --features cuda)");
    Ok(())
}

fn demo_simd_filters(image: &Array2<f64>) -> NdimageResult<()> {
    // 1. Bilateral filter - edge-preserving smoothing
    println!("Running SIMD bilateral filter...");
    let bilateral_result = simd_bilateral_filter(
        image.view(),
        5.0,    // spatial sigma
        10.0,   // range sigma
        Some(7) // window size
    )?;
    println!("Bilateral filter completed");
    
    // 2. Guided filter - structure-preserving smoothing
    println!("Running SIMD guided filter...");
    let guided_result = simd_guided_filter(
        image.view(),
        image.view(), // using image as its own guide
        5,           // radius
        0.01         // epsilon
    )?;
    println!("Guided filter completed");
    
    // 3. Adaptive median filter - impulse noise removal
    println!("Running SIMD adaptive median filter...");
    let adaptive_median_result = simd_adaptive_median_filter(
        image.view(),
        7 // max window size
    )?;
    println!("Adaptive median filter completed");
    
    Ok(())
}

fn demo_scipy_compat(image: &Array2<f64>) -> NdimageResult<()> {
    // 1. Gaussian filter with SciPy-style API
    println!("Running Gaussian filter (SciPy API)...");
    let gaussian_result = gaussian_filter(
        image,
        vec![2.0, 2.0], // sigma
        Some(0),        // order (0 = Gaussian)
        Some("reflect"), // mode
        None,           // cval
        Some(4.0)       // truncate
    )?;
    
    // 2. Zoom (resize) operation
    println!("Running zoom operation...");
    let zoomed = zoom(
        image,
        vec![2.0, 2.0], // zoom factors
        Some(3),        // spline order
        Some("constant"), // mode
        Some(0.0),      // cval
        Some(true)      // prefilter
    )?;
    println!("Zoomed image size: {}x{}", zoomed.nrows(), zoomed.ncols());
    
    // 3. Rotate operation
    println!("Running rotation...");
    let rotated = rotate(
        &image.view(),
        45.0,          // angle in degrees
        None,          // axes
        Some(false),   // reshape
        Some(1),       // order
        Some("constant"), // mode
        Some(0.0)      // cval
    )?;
    
    // 4. Laplacian edge detection
    println!("Running Laplacian filter...");
    let edges = laplace(
        image,
        Some("reflect"), // mode
        None            // cval
    )?;
    
    println!("All SciPy-compatible operations completed successfully");
    
    Ok(())
}