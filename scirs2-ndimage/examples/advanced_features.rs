//! Example demonstrating advanced features added to scirs2-ndimage
//!
//! This example showcases:
//! - GPU-accelerated operations
//! - SIMD-optimized specialized filters
//! - Advanced-advanced SIMD extensions (wavelets, LBP, edge detection)
//! - Performance profiling and optimization
//! - SciPy-compatible API

use ndarray::{array, Array2};
use scirs2_ndimage::{
    backend::{Backend, BackendBuilder},
    error::NdimageResult,
    filters::simd_specialized::{
        simd_adaptive_median_filter, simd_bilateral_filter, simd_guided_filter,
    },
    profiling::{
        disable_profiling, display_performance_report, enable_memory_profiling, enable_profiling,
        get_memory_report, OptimizationAdvisor,
    },
    scipy_compat::ndimage::{gaussian_filter, laplace, rotate, zoom},
};

// Import advanced SIMD extensions
#[cfg(feature = "simd")]
use scirs2_ndimage::filters::{
    advanced_simd_advanced_edge_detection, advanced_simd_multi_scale_lbp,
    advanced_simd_wavelet_pyramid, WaveletType,
};

#[allow(dead_code)]
fn main() -> NdimageResult<()> {
    println!("=== Advanced scirs2-ndimage Features Demo ===\n");

    // Enable profiling
    enable_profiling();
    enable_memory_profiling();

    // Create test image
    let test_image = create_test_image();
    println!(
        "Created test image: {}x{}",
        test_image.nrows(),
        test_image.ncols()
    );

    // 1. Demonstrate GPU acceleration (if available)
    #[cfg(feature = "cuda")]
    {
        println!("\n--- GPU Acceleration Demo ---");
        demo_gpu_acceleration(&test_image)?;
    }

    // 2. Demonstrate SIMD-optimized filters
    println!("\n--- SIMD-Optimized Filters Demo ---");
    demo_simd_filters(&test_image)?;

    // 3. Demonstrate advanced SIMD extensions
    #[cfg(feature = "simd")]
    {
        println!("\n--- Advanced SIMD Extensions Demo ---");
        demo_advanced_simd_extensions(&test_image)?;
    }

    // 4. Demonstrate SciPy-compatible API
    println!("\n--- SciPy-Compatible API Demo ---");
    demo_scipy_compat(&test_image)?;

    // 5. Display performance analysis
    println!("\n--- Performance Analysis ---");
    display_performance_report();

    // 6. Display memory usage
    let memory_report = get_memory_report();
    memory_report.display();

    // 7. Get optimization recommendations
    println!("\n--- Optimization Recommendations ---");
    let profiler = scirs2_ndimage::profiling::get_performance_report();
    let mut advisor = OptimizationAdvisor::new();
    let opt_report = advisor.analyze(
        profiler
            .operation_breakdown
            .values()
            .flat_map(|s| std::iter::repeat(&s.mean_time).take(s.count))
            .enumerate()
            .map(
                |(i, duration)| scirs2_ndimage::profiling::OperationMetrics {
                    name: format!("operation_{}", i),
                    duration: *duration,
                    memory_allocated: 0,
                    memory_deallocated: 0,
                    array_shape: vec![test_image.nrows(), test_image.ncols()],
                    backend: Backend::Cpu,
                    thread_count: 1,
                    timestamp: std::time::Instant::now(),
                },
            )
            .collect::<Vec<_>>()
            .as_slice(),
    );
    opt_report.display();

    disable_profiling();

    Ok(())
}

#[allow(dead_code)]
fn create_test_image() -> Array2<f64> {
    let size = 256;
    let mut image = Array2::zeros((size, size));

    // Create a pattern with edges and noise
    for i in 0..size {
        for j in 0..size {
            let x = i as f64 / size as f64;
            let y = j as f64 / size as f64;

            // Base pattern
            image[(i, j)] =
                (x * std::f64::consts::PI * 4.0).sin() * (y * std::f64::consts::PI * 4.0).cos();

            // Add some edges
            if (i as i32 - size as i32 / 2).abs() < 10 || (j as i32 - size as i32 / 2).abs() < 10 {
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
#[allow(dead_code)]
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

    println!(
        "GPU Gaussian filter completed: {}x{}",
        gpu_result.nrows(),
        gpu_result.ncols()
    );

    // Compare with CPU version
    let cpu_result = gaussian_filter(image, sigma.to_vec(), None, None, None, None)?;

    // Calculate difference
    let diff = (&gpu_result - &cpu_result).mapv(|x| x.abs()).sum();
    println!("GPU vs CPU difference: {:.6}", diff);

    Ok(())
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
fn demo_gpu_acceleration(_image: &Array2<f64>) -> NdimageResult<()> {
    println!("GPU acceleration not available (compile with --features cuda)");
    Ok(())
}

#[allow(dead_code)]
fn demo_simd_filters(image: &Array2<f64>) -> NdimageResult<()> {
    // 1. Bilateral filter - edge-preserving smoothing
    println!("Running SIMD bilateral filter...");
    let bilateral_result = simd_bilateral_filter(
        image.view(),
        5.0,     // spatial sigma
        10.0,    // range sigma
        Some(7), // window size
    )?;
    println!("Bilateral filter completed");

    // 2. Guided filter - structure-preserving smoothing
    println!("Running SIMD guided filter...");
    let guided_result = simd_guided_filter(
        image.view(),
        image.view(), // using image as its own guide
        5,            // radius
        0.01,         // epsilon
    )?;
    println!("Guided filter completed");

    // 3. Adaptive median filter - impulse noise removal
    println!("Running SIMD adaptive median filter...");
    let adaptive_median_result = simd_adaptive_median_filter(
        image.view(),
        7, // max window size
    )?;
    println!("Adaptive median filter completed");

    Ok(())
}

#[allow(dead_code)]
fn demo_scipy_compat(image: &Array2<f64>) -> NdimageResult<()> {
    // 1. Gaussian filter with SciPy-style API
    println!("Running Gaussian filter (SciPy API)...");
    let gaussian_result = gaussian_filter(
        image,
        vec![2.0, 2.0],  // sigma
        Some(0),         // order (0 = Gaussian)
        Some("reflect"), // mode
        None,            // cval
        Some(4.0),       // truncate
    )?;

    // 2. Zoom (resize) operation
    println!("Running zoom operation...");
    let zoomed = zoom(
        image,
        vec![2.0, 2.0],   // zoom factors
        Some(3),          // spline order
        Some("constant"), // mode
        Some(0.0),        // cval
        Some(true),       // prefilter
    )?;
    println!("Zoomed image size: {}x{}", zoomed.nrows(), zoomed.ncols());

    // 3. Rotate operation
    println!("Running rotation...");
    let rotated = rotate(
        &image.view(),
        45.0,             // angle in degrees
        None,             // axes
        Some(false),      // reshape
        Some(1),          // order
        Some("constant"), // mode
        Some(0.0),        // cval
    )?;

    // 4. Laplacian edge detection
    println!("Running Laplacian filter...");
    let edges = laplace(
        image,
        Some("reflect"), // mode
        None,            // cval
    )?;

    println!("All SciPy-compatible operations completed successfully");

    Ok(())
}

#[cfg(feature = "simd")]
#[allow(dead_code)]
fn demo_advanced_simd_extensions(image: &Array2<f64>) -> NdimageResult<()> {
    println!("Demonstrating advanced SIMD extensions...");

    // 1. Wavelet pyramid decomposition
    println!("Running advanced-SIMD wavelet pyramid...");
    let pyramid = advanced_simd_wavelet_pyramid(
        image.view(),
        3,                 // levels
        WaveletType::Haar, // wavelet type
    )?;
    println!(
        "Wavelet pyramid completed with {} levels",
        pyramid.levels.len()
    );

    // Print information about each level
    for (i, level) in pyramid.levels.iter().enumerate() {
        println!(
            "  Level {}: LL {}x{}, LH {}x{}, HL {}x{}, HH {}x{}",
            i,
            level.ll.nrows(),
            level.ll.ncols(),
            level.lh.nrows(),
            level.lh.ncols(),
            level.hl.nrows(),
            level.hl.ncols(),
            level.hh.nrows(),
            level.hh.ncols()
        );
    }

    // 2. Multi-scale Local Binary Patterns
    println!("Running advanced-SIMD multi-scale LBP...");
    let radii = [1, 2, 3];
    let sample_points = [8, 16, 24];

    let lbp_result = advanced_simd_multi_scale_lbp(image.view(), &radii, &sample_points)?;

    println!(
        "Multi-scale LBP completed: {}x{}",
        lbp_result.nrows(),
        lbp_result.ncols()
    );

    // Print some statistics about the LBP codes
    let max_code = lbp_result.iter().max().unwrap_or(&0);
    let min_code = lbp_result.iter().min().unwrap_or(&0);
    let unique_codes = {
        let mut codes: Vec<_> = lbp_result.iter().collect();
        codes.sort();
        codes.dedup();
        codes.len()
    };
    println!(
        "  LBP code range: {} to {}, {} unique patterns",
        min_code, max_code, unique_codes
    );

    // 3. Advanced edge detection
    println!("Running advanced-SIMD advanced edge detection...");
    let edges = advanced_simd_advanced_edge_detection(
        image.view(),
        1.0, // sigma for Gaussian smoothing
        0.1, // low threshold factor
        0.3, // high threshold factor
    )?;

    println!(
        "Advanced edge detection completed: {}x{}",
        edges.nrows(),
        edges.ncols()
    );

    // Print edge statistics
    let edge_pixels = edges.iter().filter(|&&x| x > 0.0).count();
    let total_pixels = edges.len();
    let edge_percentage = (edge_pixels as f64 / total_pixels as f64) * 100.0;
    println!(
        "  Detected edges: {} pixels ({:.2}% of image)",
        edge_pixels, edge_percentage
    );

    // 4. Compare with different wavelet types
    println!("Comparing wavelet types...");
    for (name, wavelet_type) in [
        ("Daubechies-4", WaveletType::Daubechies4),
        ("Biorthogonal", WaveletType::Biorthogonal),
    ] {
        let pyramid = advanced_simd_wavelet_pyramid(
            image.view(),
            2, // levels
            wavelet_type,
        )?;
        println!("  {}: {} levels generated", name, pyramid.levels.len());
    }

    println!("Advanced-advanced SIMD extensions demo completed successfully");

    Ok(())
}

#[cfg(not(feature = "simd"))]
#[allow(dead_code)]
fn demo_advanced_simd_extensions(_image: &Array2<f64>) -> NdimageResult<()> {
    println!("Advanced-advanced SIMD extensions not available (compile with --features simd)");
    Ok(())
}
