# Unwrap() Usage Report

Total unwrap() calls and unsafe operations found: 3151

## Summary by Type

- Replace with ? operator or .ok_or(): 1910 occurrences
- Division without zero check - use safe_divide(): 1027 occurrences
- Mathematical operation .sqrt() without validation: 90 occurrences
- Use .get().ok_or(Error::IndexOutOfBounds)?: 37 occurrences
- Mathematical operation .ln() without validation: 36 occurrences
- Mathematical operation .powf( without validation: 21 occurrences
- Use .get() with proper bounds checking: 18 occurrences
- Handle array creation errors properly: 12 occurrences

## Detailed Findings


### benches/array_protocol_bench.rs

4 issues found:

- Line 27: `bench.iter(|| matmul(&wrapped_a, &wrapped_b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 50: `bench.iter(|| add(&wrapped_a, &wrapped_b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 69: `bench.iter(|| transpose(&wrapped_a).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 94: `bench.iter(|| matmul(&gpu_a, &gpu_b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()

### benches/numpy_scipy_comparison_bench.rs

19 issues found:

- Line 97: `let result = a.mapv(|x| x.sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 142: `let result = a.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 150: `let mean = a.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 151: `let variance = a.mapv(|x| (x - mean).powi(2)).mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 152: `let std = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 237: `let cols = size / rows;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 241: `let result = a.clone().into_shape_with_order((rows, cols)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 256: `let result = ndarray::concatenate(Axis(0), &views).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 265: `let mid = size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 275: `vec.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 298: `let mean = a.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 299: `let variance = a.mapv(|x| (x - mean).powi(2)).mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 310: `let mean1 = a1.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 311: `let mean2 = a2.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 317: `/ (size as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 327: `sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 329: `(sorted[size / 2 - 1] + sorted[size / 2]) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 331: `sorted[size / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 372: `arrays.push(Array1::<f64>::zeros(s / 10));`
  - **Fix**: Division without zero check - use safe_divide()

### benches/pattern_recognition_bench.rs

3 issues found:

- Line 367: `for i in kernel_size / 2..rows - kernel_size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 368: `for j in kernel_size / 2..cols - kernel_size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 373: `(i - kernel_size / 2 + ki) * cols + (j - kernel_size / 2 + kj);`
  - **Fix**: Division without zero check - use safe_divide()

### benches/validation_bench.rs

7 issues found:

- Line 19: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 49: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 89: `let validator = Validator::new(config.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 112: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 164: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 197: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 251: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/adaptive_optimization_demo.rs

10 issues found:

- Line 111: `.map(|m| m / (1024 * 1024 * 1024))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 134: `custom_config.history_retention.as_secs() / 3600`
  - **Fix**: Division without zero check - use safe_divide()
- Line 164: `compute_workload.data_size / (1024 * 1024)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 194: `memory_workload.data_size / (1024 * 1024)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 239: `compute_hints.preferred_chunk_size.map(|s| s / 1024)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 258: `memory_hints.preferred_chunk_size.map(|s| s / 1024)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 583: `let base_throughput = 1000.0 / load_multiplier;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 585: `let base_memory = 512.0 * load_multiplier.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 649: `(final_stats.uptime.as_secs() / 2).max(1)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 816: `stats.uptime.as_secs_f64() / 60.0`
  - **Fix**: Division without zero check - use safe_divide()

### examples/advanced_error_handling.rs

2 issues found:

- Line 40: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 437: `max_iterations / 2`
  - **Fix**: Division without zero check - use safe_divide()

### examples/advanced_indexing_example.rs

26 issues found:

- Line 26: `let selected = indexing::boolean_mask_2d(a.view(), mask.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 35: `indexing::fancy_index_2d(a.view(), row_indices.view(), col_indices.view()).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 40: `let main_diag = indexing::diagonal(a.view(), 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 45: `let condition_result = indexing::where_2d(a.view(), |&x| x > 5.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 55: `println!("- Global: {:?}", stats::mean(&a.view(), None).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 58: `stats::mean(&a.view(), Some(Axis(0))).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 62: `stats::mean(&a.view(), Some(Axis(1))).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 66: `println!("- Global: {:?}", stats::median(&a.view(), None).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 69: `stats::median(&a.view(), Some(Axis(0))).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 73: `stats::median(&a.view(), Some(Axis(1))).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 79: `stats::std_dev(&a.view(), None, 1).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 83: `stats::std_dev(&a.view(), Some(Axis(0)), 1).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 87: `stats::std_dev(&a.view(), Some(Axis(1)), 1).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 91: `println!("- Min: {:?}", stats::min(&a.view(), None).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 92: `println!("- Max: {:?}", stats::max(&a.view(), None).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 95: `stats::min(&a.view(), Some(Axis(0))).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 99: `stats::max(&a.view(), Some(Axis(1))).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 105: `stats::percentile(&a.view(), 25.0, None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 109: `stats::percentile(&a.view(), 50.0, None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 113: `stats::percentile(&a.view(), 75.0, None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 117: `stats::percentile(&a.view(), 75.0, Some(Axis(0))).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 131: `println!("- Mean: {:?}", stats::mean(&z_scores.view(), None).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 134: `stats::std_dev(&z_scores.view(), None, 0).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 141: `let mean = stats::mean(&array.view(), None).unwrap()[0];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 142: `let std_dev = stats::std_dev(&array.view(), None, 0).unwrap()[0];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 145: `array.mapv(|x| (x - mean) / std_dev)`
  - **Fix**: Division without zero check - use safe_divide()

### examples/advanced_ndarray_example.rs

9 issues found:

- Line 46: `let angles_1d = array![0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 47: `let angles = array![[0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 60: `let tr = trace(array![[1.0, 2.0], [3.0, 4.0]].view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 77: `let toep = toeplitz(array![1.0, 2.0, 3.0].view(), array![1.0, 4.0, 7.0].view())....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 86: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 116: `let concat_h = concatenate_2d(&[a.view(), b.view()], 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 120: `let vstacked = vstack_1d(&[c.view(), array![40.0, 50.0, 60.0].view()]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 237: `standardized[[i, j]] = (measurements[[i, j]] - column_means[j]) / column_std[j];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 261: `corr[[i, j]] = sum_product / (standardized.shape()[0] as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()

### examples/advanced_parallel_processing.rs

5 issues found:

- Line 60: `.map(|i| if i < 1000 { 1.0 } else { (i as f64).powf(2.0) })`
  - **Fix**: Mathematical operation .powf( without validation
- Line 64: `let x = (i as f64 - 5000.0) / 1000.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 65: `(-x * x / 2.0).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 342: `let avg_size = sizes.iter().sum::<usize>() / sizes.len();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 351: `max_size as f64 / min_size as f64`
  - **Fix**: Division without zero check - use safe_divide()

### examples/advanced_prefetching_example.rs

12 issues found:

- Line 49: `((i + j) % 100) as f64 / 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 133: `no_prefetch_time.as_secs_f64() / basic_prefetch_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 183: `basic_prefetch_time.as_secs_f64() / adaptive_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 302: `return Some(vec![i, j / 2]);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 318: `return Some(vec![i / 5, j / 5, 0]);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 343: `if j / 2 < weights_cols {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 344: `let weights_val = weights_cmm.get(&[i, j / 2])?;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 349: `if i / 5 < tensor_size && j / 5 < tensor_size {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 350: `let tensor_val = tensor_cmm.get(&[i / 5, j / 5, 0])?;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 381: `if j / 2 < weights_cols {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 388: `if i / 5 < tensor_size && j / 5 < tensor_size {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 400: `normal_time.as_secs_f64() / cross_file_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()

### examples/arbitrary_precision_example.rs

13 issues found:

- Line 107: `ArbitraryFloat::from_f64_prec(1.0, prec)? / ArbitraryFloat::from_f64_prec(3.0, p...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 126: `println!("ln(exp(x)) = {}", x.exp().ln()?);`
  - **Fix**: Mathematical operation .ln() without validation
- Line 184: `println!("z1 / z2 = {}", z1.clone() / z2.clone());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 209: `(prec as f64 / 3.32) as u32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 231: `let phi_check = (one + five.sqrt()?) / two;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 231: `let phi_check = (one + five.sqrt()?) / two;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 232: `println!("Verification: (1 + √5) / 2 = {}", phi_check);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 258: `Ok(pi_squared / six)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 268: `let term = ArbitraryFloat::from_f64_prec(1.0, 384)? / (n_float.clone() * n_float...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 287: `(x.clone() + two.clone() / x.clone()) / ArbitraryFloat::from_f64_prec(2.0, prec)...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 292: `let actual_sqrt2 = two.sqrt()?;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 299: `let x = pi / four;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 305: `let derivative = (f_x_plus_h - f_x) / h;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/array_protocol_backpropagation.rs

3 issues found:

- Line 186: `let avg_loss = total_loss / inputs.shape()[0] as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 431: `let avg_batch_loss = batch_loss / inputs.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 583: `let result = ndarray.mapv(|x| 1.0 / (1.0 + (-x).exp()));`
  - **Fix**: Division without zero check - use safe_divide()

### examples/array_protocol_distributed_training.rs

3 issues found:

- Line 366: `let temp_dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 468: `if (batch + 1) % (num_batches / 10).max(1) == 0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 470: `1.0 - (epoch as f64 * 0.1 + batch as f64 * 0.01 / num_batches as f64);`
  - **Fix**: Division without zero check - use safe_divide()

### examples/array_protocol_gpu.rs

1 issues found:

- Line 163: `cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()

### examples/batch_conversions_example.rs

2 issues found:

- Line 153: `let speedup = sequential_time.as_nanos() as f64 / simd_time.as_nanos() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 213: `let speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f6...`
  - **Fix**: Division without zero check - use safe_divide()

### examples/core_memory_efficient_example.rs

2 issues found:

- Line 17: `let data = Array2::from_shape_fn((n, n), |(i, j)| (i as f64 + j as f64) / (n as ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 20: `println!("Array memory usage: ~{} MB\n", n * n * 8 / (1024 * 1024));`
  - **Fix**: Division without zero check - use safe_divide()

### examples/coverage_analysis_demo.rs

3 issues found:

- Line 187: `report.performance_impact.memory_overhead_bytes as f64 / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 204: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 430: `for line in 1..=(line_count as u32 * 85 / 100) {`
  - **Fix**: Division without zero check - use safe_divide()

### examples/custom_array_protocol.rs

2 issues found:

- Line 86: `self.nnz() as f64 / total as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 197: `SparseArray::from_dense(&result.into_dimensionality().unwrap());`
  - **Fix**: Handle array creation errors properly

### examples/distributed_training_example.rs

1 issues found:

- Line 131: `let temp_dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/enhanced_memory_metrics_example.rs

10 issues found:

- Line 75: `let dealloc_size = size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 280: `let dealloc_size = size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 332: `result.session.peak_memory_usage as f64 / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 560: `*peak_size as f64 / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 711: `allocation_count as f64 / total_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 739: `result.session.peak_memory_usage as f64 / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 749: `stats.current_usage as f64 / 1024.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 751: `println!("    Peak usage: {:.2} KB", stats.peak_usage as f64 / 1024.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 760: `stats.total_allocated as f64 / stats.peak_usage as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 767: `let alloc_rate = stats.allocation_count as f64 / total_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()

### examples/gpu_detection_example.rs

1 issues found:

- Line 69: `let memory_gb = memory as f64 / (1024.0 * 1024.0 * 1024.0);`
  - **Fix**: Division without zero check - use safe_divide()

### examples/gpu_heavy_stress_test.rs

8 issues found:

- Line 75: `n_elements * 4 / (1024 * 1024)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 112: `(n_elements as f64 * 4.0 * 2.0 * iterations as f64) / (1024.0 * 1024.0 * 1024.0)...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 113: `let throughput = gb_processed / compute_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 144: `let upload_bandwidth = (size_mb as f64) / upload_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 145: `let download_bandwidth = (size_mb as f64) / download_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 163: `println!("Data size: {} elements ({} MB)", n, n * 4 / (1024 * 1024));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 169: `x.sin() * x.cos() + x.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 195: `let gflops = total_flops / compute_time.as_secs_f64() / 1e9;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/gpu_kernel_comprehensive_example.rs

9 issues found:

- Line 130: `let gflops = flops / duration.as_secs_f64() / 1e9;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 215: `let cpu_mean = cpu_sum / data.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 389: `let output_length = (signal_length + 2 * padding - kernel_length) / stride + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 434: `let output_height = (input_height + 2 * padding_y - kernel_height) / stride_y + ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 435: `let output_width = (input_width + 2 * padding_x - kernel_width) / stride_x + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 661: `let cpu_gflops = flops / cpu_time.as_secs_f64() / 1e9;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 662: `let gpu_gflops = flops / gpu_time.as_secs_f64() / 1e9;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 663: `let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 699: `let bandwidth_gb_s = total_bytes as f64 / gpu_time.as_secs_f64() / 1e9;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/gpu_kernel_library.rs

14 issues found:

- Line 67: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 70: `Array2::from_shape_vec((4, 2), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])....`
  - **Fix**: Handle array creation errors properly
- Line 100: `let a_buffer = ctx.create_buffer_from_slice(a.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 101: `let b_buffer = ctx.create_buffer_from_slice(b.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 123: `let result = Array2::from_shape_vec((a.shape()[0], b.shape()[1]), result_vec).un...`
  - **Fix**: Handle array creation errors properly
- Line 165: `let x_buffer = ctx.create_buffer_from_slice(x.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 166: `let y_buffer = ctx.create_buffer_from_slice(y.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 182: `y_buffer.copy_to_host(y.as_slice_mut().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 213: `let input_buffer = ctx.create_buffer_from_slice(x.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 253: `let input_buffer = ctx.create_buffer_from_slice(x.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 266: `let norm = sum_squares.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 271: `let expected: f32 = x.dot(&x).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 292: `let input_buffer = ctx.create_buffer_from_slice(x.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 334: `let expected_sigmoid: Vec<f32> = x.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).col...`
  - **Fix**: Division without zero check - use safe_divide()

### examples/gpu_kernel_library_example.rs

2 issues found:

- Line 297: `let t = (i as f32) / (size as f32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 421: `let speedup = cpu_time / gpu_time;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/gpu_stress_example.rs

1 issues found:

- Line 46: `println!("Allocating {} MB on GPU", N * 4 / (1024 * 1024));`
  - **Fix**: Division without zero check - use safe_divide()

### examples/integrated_features.rs

5 issues found:

- Line 61: `Profiler::global().lock().unwrap().start();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 72: `let normal_distribution = Normal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 157: `Profiler::global().lock().unwrap().print_report();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 162: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 178: `Profiler::global().lock().unwrap().stop();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/large_dataset_processing.rs

4 issues found:

- Line 24: `normalized[[i, col]] = (val - mean) / std_dev;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 80: `corr[[i, j]] = cov / (std_i * std_j);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 98: `let n_chunks = total_samples / chunk_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 169: `100.0 * outlier_count as f64 / mask.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()

### examples/leak_detection_demo.rs

2 issues found:

- Line 31: `config.growth_threshold_bytes / (1024 * 1024)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 51: `checkpoint.memory_usage.rss_bytes / 1024`
  - **Fix**: Division without zero check - use safe_divide()

### examples/memory_management.rs

2 issues found:

- Line 119: `let mut buffer = f32_pool.lock().unwrap().acquire_vec(5);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 129: `f32_pool.lock().unwrap().release_vec(buffer);`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/memory_mapped_adaptive.rs

5 issues found:

- Line 68: `for chunk_idx in 0..(size_1d / CHUNK_SIZE) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 111: `for chunk_idx in 0..(size_large / CHUNK_SIZE) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 303: `let seq_result = [data.iter().map(|&x| (x * x).sqrt()).sum::<f64>()];`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 340: `let result = [data.iter().map(|&x| (x * x).sqrt()).sum::<f64>()];`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 346: `let speedup = baseline_ms / parallel_ms;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/memory_mapped_array.rs

4 issues found:

- Line 82: `let size_mb = (rows * cols * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 158: `let size_mb = (rows * cols * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 208: `mmap_creation_time.as_secs_f64() / in_memory_creation_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 212: `mmap_read_time.as_secs_f64() / in_memory_read_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()

### examples/memory_mapped_chunks.rs

22 issues found:

- Line 123: `let mean = sum as f64 / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 139: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 144: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 153: `let global_mean = global_sum as f64 / global_count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 191: `.min_by(|a, b| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 192: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 195: `.max_by(|a, b| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 196: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 218: `.min_by(|a, b| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 219: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 222: `.max_by(|a, b| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 223: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 231: `.min_by(|a, b| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 232: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 236: `.max_by(|a, b| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 237: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 261: `.min_by(|a, b| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 262: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 265: `.max_by(|a, b| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 266: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 301: `time_chunks.as_secs_f64() / time_no_chunks.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 306: `time_iter.as_secs_f64() / time_no_chunks.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()

### examples/memory_mapped_compressed.rs

5 issues found:

- Line 28: `for chunk_idx in 0..(size / chunk_size) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 96: `let compression_ratio = raw_file_size as f64 / compressed_size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 103: `compressed_size as f64 / (1024.0 * 1024.0),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 287: `let block_idx = (i * 1000) / cmm.metadata().block_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 297: `let improvement = no_preload_time.as_millis() as f64 / with_preload_time.as_mill...`
  - **Fix**: Division without zero check - use safe_divide()

### examples/memory_mapped_mutation.rs

2 issues found:

- Line 162: `*item = (global_idx / 1000) as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 179: `let expected = (pos / 1000) as i32;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/memory_mapped_parallel.rs

3 issues found:

- Line 30: `let temp_file = tempfile::NamedTempFile::new().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 35: `let mut mmap = create_mmap(&data, temp_path, AccessMode::ReadWrite, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 84: `let speedup = sequential_time.as_secs_f64() / parallel_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()

### examples/memory_mapped_running_stats.rs

13 issues found:

- Line 61: `self.mean += delta / self.count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 76: `self.m2 / (self.count - 1) as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 81: `self.variance().sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 101: `(self.mean * self.count as f64 + other.mean * other.count as f64) / total_count ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 106: `+ delta * delta * (self.count * other.count) as f64 / total_count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 164: `(x / 1000.0).sin() * 10.0 + (x / 5000.0).cos() * 5.0 + (i % 100) as f64 / 10.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 184: `let value = (x / 1000.0).sin() * 10.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 185: `+ (x / 5000.0).cos() * 5.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 186: `+ (global_idx % 100) as f64 / 10.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 217: `(size * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 251: `let chunk = ArrayView1::from_shape(chunk_data.len(), chunk_data).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 325: `*value = (*value - mean) / stddev;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 377: `let stride = (mmap.size / sample_size).max(1);`
  - **Fix**: Division without zero check - use safe_divide()

### examples/memory_mapped_zerocopy.rs

7 issues found:

- Line 28: `for chunk_idx in 0..(size / chunk_size) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 118: `let squared_loaded_mean = squared_loaded.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 134: `for chunk_idx in 0..(size / chunk_size) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 273: `total_time as f64 / n_runs as f64,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 287: `total_time as f64 / n_runs as f64,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 308: `total_time as f64 / n_runs as f64,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 334: `total_time as f64 / n_runs as f64,`
  - **Fix**: Division without zero check - use safe_divide()

### examples/memory_mapped_zerocopy_serialization.rs

14 issues found:

- Line 37: `(self.real * self.real + self.imag * self.imag).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 111: `let distance = ((i as f64 - size as f64 / 2.0).powi(2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 112: `+ (j as f64 - size as f64 / 2.0).powi(2))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 113: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 115: `((j as f64 - size as f64 / 2.0) / (i as f64 - size as f64 / 2.0 + 0.001)).atan()...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 464: `memory_size as f64 / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 523: `zero_copy_path.metadata()?.len() as f64 / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 527: `traditional_path.metadata()?.len() as f64 / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 542: `zero_copy_save_time.as_micros() as f64 / traditional_save_time.as_micros() as f6...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 546: `1.0 / ser_ratio`
  - **Fix**: Division without zero check - use safe_divide()
- Line 557: `zero_copy_load_time.as_micros() as f64 / traditional_load_time.as_micros() as f6...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 561: `1.0 / deser_ratio`
  - **Fix**: Division without zero check - use safe_divide()
- Line 572: `zero_copy_access_time.as_micros() as f64 / traditional_access_time.as_micros() a...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 576: `1.0 / access_ratio`
  - **Fix**: Division without zero check - use safe_divide()

### examples/memory_metrics_chunking.rs

2 issues found:

- Line 78: `let average = sum / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 153: `shared_data.total_sum / shared_data.total_count as f64`
  - **Fix**: Division without zero check - use safe_divide()

### examples/memory_metrics_snapshots.rs

2 issues found:

- Line 61: `let diff = compare_snapshots("baseline", "allocated").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 89: `let leak_diff = compare_snapshots("before_leak", "after_leak").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/ndarray_advanced_operations.rs

14 issues found:

- Line 28: `let (x_grid, y_grid) = meshgrid(x.view(), y.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 44: `let unique_values = unique(b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 55: `let min_indices_cols = argmin(c.view(), Some(0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 59: `let min_indices_rows = argmin(c.view(), Some(1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 63: `let min_index = argmin(c.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 68: `let max_indices_cols = argmax(c.view(), Some(0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 72: `let max_indices_rows = argmax(c.view(), Some(1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 76: `let max_index = argmax(c.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 86: `let (grad_y, grad_x) = gradient(a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 93: `let (grad_y, grad_x) = gradient(a.view(), Some((2.0, 0.5))).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 142: `let (grad_y, grad_x) = gradient(img.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 156: `magnitude[[i, j]] = (gy * gy + gx * gx).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 164: `let max_val_idx = argmax(magnitude.view(), None).unwrap()[0];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 166: `let edge_row = max_val_idx / rows;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/ndarray_correlation_binning.rs

10 issues found:

- Line 17: `let counts = bincount(data.view(), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 30: `let indices = digitize(values.view(), bins.view(), true, "indices").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 38: `let hist_result = histogram(values.view(), 5, Some((0.0, 10.0)), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 69: `let corr_pos = corrcoef(x.view(), y_pos.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 70: `let corr_neg = corrcoef(x.view(), y_neg.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 71: `let corr_none = corrcoef(x.view(), y_none.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 87: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 90: `let cov_matrix = cov(data.view(), 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 114: `let quartiles = quantile(data.view(), array![0.25, 0.5, 0.75].view(), None).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 164: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/ndarray_statistical_operations.rs

19 issues found:

- Line 25: `let global_mean = mean_2d(&data.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 26: `let col_means = mean_2d(&data.view(), Some(Axis(0))).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 27: `let row_means = mean_2d(&data.view(), Some(Axis(1))).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 35: `let global_median = median_2d(&data.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 36: `let col_medians = median_2d(&data.view(), Some(Axis(0))).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 37: `let row_medians = median_2d(&data.view(), Some(Axis(1))).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 45: `let global_var = variance_2d(&data.view(), None, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 46: `let global_std = std_dev_2d(&data.view(), None, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 53: `let global_min = min_2d(&data.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 54: `let global_max = max_2d(&data.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 61: `let global_sum = sum_2d(&data.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 62: `let col_sums = sum_2d(&data.view(), Some(Axis(0))).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 71: `let global_p = percentile_2d(&data.view(), *p, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 85: `let (hist, bin_edges) = histogram(data_1d.view(), 5, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 108: `let results = quantile(data_1d.view(), quantiles.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 126: `let x_val = (i as f64) / 10.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 133: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 137: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 145: `histogram2d(x_arr.view(), y_arr.view(), Some((6, 6)), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/ndarray_ufuncs_example.rs

5 issues found:

- Line 22: `let reshaped = reshape_2d(a.view(), (6, 1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 27: `let stacked = stack_2d(&[a.view(), b.view()], 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 37: `let masked = mask_select(a.view(), mask.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 55: `let add_broadcast = binary2d::add(&a.view(), &c.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 59: `let mul_broadcast = binary2d::multiply(&a.view(), &c.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/neural_network_training.rs

3 issues found:

- Line 196: `if (batch + 1) % (num_batches / 10).max(1) == 0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 198: `1.0 - (epoch as f64 * 0.1 + batch as f64 * 0.01 / num_batches as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 262: `let probs = exp_outputs / sum_exp.insert_axis(Axis(1));`
  - **Fix**: Division without zero check - use safe_divide()

### examples/numerical_stability_example.rs

9 issues found:

- Line 120: `let value = mean + std * ((i as f64 / n as f64) - 0.5);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 125: `let naive_mean: f64 = data.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 127: `data.iter().map(|&x| (x - naive_mean).powi(2)).sum::<f64>() / (n - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 146: `welford.variance().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 175: `println!("Naive log(sum(exp)): {}", naive_sum.ln());`
  - **Fix**: Mathematical operation .ln() without validation
- Line 202: `hilbert[[i, j]] = 1.0 / ((i + j + 2) as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 376: `let naive = (1.0_f64 + x).ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 386: `((naive - true_value) / true_value).abs()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 390: `((stable - true_value) / true_value).abs()`
  - **Fix**: Division without zero check - use safe_divide()

### examples/performance_dashboards_demo.rs

3 issues found:

- Line 76: `production_config.retention_period.as_secs() / 86400`
  - **Fix**: Division without zero check - use safe_divide()
- Line 95: `dev_config.retention_period.as_secs() / 86400`
  - **Fix**: Division without zero check - use safe_divide()
- Line 259: `let y = (i / 3) as u32 * 3;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/production_profiling_demo.rs

6 issues found:

- Line 69: `production_config.max_memory_usage / (1024 * 1024)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 93: `dev_config.max_memory_usage / (1024 * 1024)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 177: `report.resource_utilization.memory_bytes as f64 / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 378: `resource_usage.memory_bytes as f64 / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 387: `resource_usage.network_bytes_per_sec / 1024.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 412: `let _result: f64 = (0..50).map(|i| (i as f64).sqrt()).sum();`
  - **Fix**: Mathematical operation .sqrt() without validation

### examples/profiling_example.rs

3 issues found:

- Line 12: `Profiler::global().lock().unwrap().start();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 28: `Profiler::global().lock().unwrap().print_report();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 31: `Profiler::global().lock().unwrap().stop();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/random_example.rs

6 issues found:

- Line 64: `let uniform = Uniform::new(0.0, 10.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 69: `let normal = Normal::new(5.0, 2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 74: `let bernoulli = Bernoulli::new(0.7).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 88: `let uniform = Uniform::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 93: `let normal = Normal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 99: `let dist = Uniform::new(1, 100).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/type_conversion_example.rs

6 issues found:

- Line 39: `let int_result: i32 = float_value.to_numeric().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 44: `let float_result: f64 = int_value.to_numeric().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 49: `let medium_int: i32 = large_int.to_numeric().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 138: `let z_rotated = z1.rotate(std::f64::consts::PI / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 152: `let z32 = z64.convert_complex::<f32>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 156: `let z64_back = z32.convert_complex::<f64>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/versioning_demo.rs

1 issues found:

- Line 120: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples_disabled/scientific_arrays_example.rs

2 issues found:

- Line 59: `let div = &masked_a / &masked_b;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 60: `println!("masked_a / masked_b = {:?}", div);`
  - **Fix**: Division without zero check - use safe_divide()

### src/api_versioning.rs

3 issues found:

- Line 222: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 258: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 292: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/array/masked_array.rs

12 issues found:

- Line 519: `Some(sum / A::from(count).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 546: `Some(sum_sq_diff / A::from(count - ddof).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 907: `let data = &self.data / &rhs.data;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 929: `let ma = MaskedArray::new(data.clone(), Some(mask.clone()), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 942: `let ma = MaskedArray::new(data, Some(mask), Some(999.0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 985: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 992: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1010: `let f = &a / &b;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1020: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1027: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1029: `let i = &g / &h;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1040: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/array/record_array.rs

4 issues found:

- Line 370: `.map(|record| record.get_field(field_name).unwrap().clone())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 650: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 818: `let half = max_records_to_show / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 829: `let half = max_records_to_show / 2;`
  - **Fix**: Division without zero check - use safe_divide()

### src/array_protocol/auto_device.rs

1 issues found:

- Line 314: `self.device_array.as_ref().unwrap().as_ref()`
  - **Fix**: Replace with ? operator or .ok_or()

### src/array_protocol/distributed_impl.rs

3 issues found:

- Line 257: `results.into_iter().reduce(reduce_fn).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 310: `let mean = sum / count;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 634: `let result = dist_array.to_array().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/array_protocol/distributed_training.rs

2 issues found:

- Line 391: `let samples_per_worker = num_samples / num_workers;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 721: `let (input, target) = dist_dataset.get(0).unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?

### src/array_protocol/gpu/cuda_operations.rs

20 issues found:

- Line 65: `let a_cpu = a.to_cpu().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 66: `let b_cpu = b.to_cpu().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 68: `let a_array = a_cpu.downcast_ref::<NdarrayWrapper<f64, _>>().unwrap().as_array()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 69: `let b_array = b_cpu.downcast_ref::<NdarrayWrapper<f64, _>>().unwrap().as_array()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 103: `let a_cpu = a.to_cpu().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 104: `let b_cpu = b.to_cpu().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 106: `let a_array = a_cpu.downcast_ref::<NdarrayWrapper<f64, _>>().unwrap().as_array()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 107: `let b_array = b_cpu.downcast_ref::<NdarrayWrapper<f64, _>>().unwrap().as_array()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 134: `let a_cpu = a.to_cpu().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 135: `let b_cpu = b.to_cpu().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 137: `let a_array = a_cpu.downcast_ref::<NdarrayWrapper<f64, _>>().unwrap().as_array()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 138: `let b_array = b_cpu.downcast_ref::<NdarrayWrapper<f64, _>>().unwrap().as_array()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 162: `let a_cpu = a.to_cpu().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 163: `let a_array = a_cpu.downcast_ref::<NdarrayWrapper<f64, _>>().unwrap().as_array()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 182: `let a_cpu = a.to_cpu().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 183: `let a_array = a_cpu.downcast_ref::<NdarrayWrapper<f64, _>>().unwrap().as_array()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 207: `let a_cpu = a.to_cpu().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 208: `let a_array = a_cpu.downcast_ref::<NdarrayWrapper<f64, _>>().unwrap().as_array()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 245: `let h_out = (input_shape[0] - kernel.shape()[0] + 2 * padding.0) / stride.0 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 246: `let w_out = (input_shape[1] - kernel.shape()[1] + 2 * padding.1) / stride.1 + 1;`
  - **Fix**: Division without zero check - use safe_divide()

### src/array_protocol/gpu_impl.rs

6 issues found:

- Line 235: `let mean = sum / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 646: `assert_eq!(info.get("backend").unwrap(), "CUDA");`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 647: `assert_eq!(info.get("device_id").unwrap(), "0");`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 648: `assert_eq!(info.get("on_gpu").unwrap(), "true");`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 680: `let result = kernels::add(&gpu_a, &gpu_b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 691: `let result = kernels::multiply(&gpu_a, &gpu_b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/array_protocol/grad.rs

10 issues found:

- Line 381: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 393: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 452: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 464: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 648: `multiply_by_scalar(node_grad.as_ref(), 1.0 / n_elements)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 785: `let result = array.mapv(|x| 1.0 / (1.0 + (-x).exp()));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1066: `multiply_by_scalar(m.as_ref(), 1.0 / (1.0 - self.beta1.powi(self.t as i32)))?;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1068: `multiply_by_scalar(v.as_ref(), 1.0 / (1.0 - self.beta2.powi(self.t as i32)))?;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1158: `let result = a_array.as_array().mapv(|x| x.sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1187: `let result = a_array.as_array() / b_array.as_array();`
  - **Fix**: Division without zero check - use safe_divide()

### src/array_protocol/jit_impl.rs

8 issues found:

- Line 437: `let jit_manager = jit_manager.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 446: `let jit_manager = jit_manager.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 463: `let jit_manager = jit_manager.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 537: `let jit_function = factory.create(expression, array_type_id).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 542: `assert_eq!(compile_info.get("backend").unwrap(), "LLVM");`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 559: `let jit_function = jit_manager.compile(expression, array_type_id).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 576: `let mut jit_manager = JITManager::global().write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 585: `let jit_function = jit_array.compile(expression).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/array_protocol/ml_ops.rs

12 issues found:

- Line 57: `ActivationFunc::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 73: `result.assign(&(exp_x / sum_exp));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 82: `slice.assign(&(exp_x / sum_exp));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 179: `let out_height = (input_height - filter_height + 2 * padding.0) / stride.0 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 180: `let out_width = (input_width - filter_width + 2 * padding.1) / stride.1 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 244: `let out_height = (input_height - kernel_size.0 + 2 * padding.0) / stride.0 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 245: `let out_width = (input_width - kernel_size.1 + 2 * padding.1) / stride.1 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 438: `softmax[[i, j]] = val / sum_exp;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 450: `loss -= l_val * (s_val + 1e-10).ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 459: `let mean = sample_losses.sum() / sample_losses.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 546: `let scale = 1.0 / (1.0 - rate);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 637: `let _scale_factor = scale.unwrap_or((d_k as f64).sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/array_protocol/mod.rs

17 issues found:

- Line 316: `let to_remove = self.dispatch_cache.len() / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1159: `let mut jit_manager = JITManager::global().write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1251: `let mut reg = registry.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1257: `let reg = registry.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1258: `let registered_func = reg.get("scirs2::test::test_func").unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 1269: `assert_eq!(info.get("type").unwrap(), "mock_distributed");`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 1270: `assert_eq!(info.get("chunks").unwrap(), "3");`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 1279: `assert_eq!(info.get("device").unwrap(), "cuda:0");`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 1280: `assert_eq!(info.get("type").unwrap(), "mock_gpu");`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 1334: `let result = dist_array.to_array().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1366: `assert_eq!(info.get("backend").unwrap(), "CUDA");`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 1394: `let jit_function = jit_array.compile(expression).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1401: `assert_eq!(info.get("supports_jit").unwrap(), "true");`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 1474: `let mut reg = registry.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1480: `let reg = registry.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1481: `let registered_func = reg.get(func_name).unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 1636: `let sum = *value.downcast_ref::<f64>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/array_protocol/neural.rs

7 issues found:

- Line 96: `let scale = (6.0 / (in_features + out_features) as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 96: `let scale = (6.0 / (in_features + out_features) as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 234: `let scale = (2.0 / fan_in as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 234: `let scale = (2.0 / fan_in as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 707: `let scale = (1.0 / d_model as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 707: `let scale = (1.0 / d_model as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 761: `Some((self.d_model / self.num_heads) as f64),`
  - **Fix**: Division without zero check - use safe_divide()

### src/array_protocol/operations.rs

1 issues found:

- Line 859: `let expected = c.clone().into_shape_with_order(6).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/array_protocol/serialization.rs

6 issues found:

- Line 171: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 267: `let linear = layer.as_any().downcast_ref::<Linear>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 296: `let conv = layer.as_any().downcast_ref::<Conv2D>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 720: `Ok((model, optimizer.unwrap(), epoch, metrics))`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 777: `let (loaded_model, loaded_optimizer) = serializer.load_model("test_model", "v1")...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 836: `let (loaded_model, _loaded_optimizer, loaded_epoch, loaded_metrics) = result.unw...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/array_protocol/training.rs

18 issues found:

- Line 321: `let mean = array.as_array().mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 398: `let log_preds = preds.mapv(|x| x.max(1e-10).ln());`
  - **Fix**: Mathematical operation .ln() without validation
- Line 408: `let mean = losses.mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 479: `Some(sum / self.losses.len() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 490: `Some(sum / accuracies.len() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 596: `if self.verbose && (batch + 1) % (num_batches / 10).max(1) == 0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 743: `let (inputs, targets) = data_loader.next_batch().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 803: `let batch_loss = batch_loss / inputs.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 830: `let (inputs, targets) = data_loader.next_batch().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 895: `let batch_loss = batch_loss / inputs.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 897: `batch_correct as f64 / batch_total as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 946: `let (input, target) = dataset.get(0).unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 971: `let (batch1_inputs, batch1_targets) = loader.next_batch().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 975: `let (batch2_inputs, batch2_targets) = loader.next_batch().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 979: `let (batch3_inputs, batch3_targets) = loader.next_batch().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 985: `let (batch1_inputs, batch1_targets) = loader.next_batch().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1040: `assert_eq!(metrics.mean_loss().unwrap(), 2.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1041: `assert_eq!(metrics.mean_accuracy().unwrap(), 0.6);`
  - **Fix**: Replace with ? operator or .ok_or()

### src/batch_conversions.rs

6 issues found:

- Line 821: `let result: Vec<f32> = converter.convert_slice(&data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 860: `let result: Vec<num_complex::Complex32> = converter.convert_complex_slice(&data)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 871: `let result: Vec<f32> = converter.convert_slice(&data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 899: `let result: Vec<f32> = converter.convert_slice(&data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 906: `let f32_result = utils::f64_to_f32_batch(&f64_data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 910: `let f64_result = utils::f32_to_f64_batch(&f32_data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/benchmarking/mod.rs

19 issues found:

- Line 231: `operations_per_iteration as f64 / avg_time_seconds`
  - **Fix**: Division without zero check - use safe_divide()
- Line 239: `let memory_mb = self.statistics.mean_memory_usage as f64 / (1024.0 * 1024.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 240: `operations_per_iteration as f64 / memory_mb`
  - **Fix**: Division without zero check - use safe_divide()
- Line 288: `/ execution_times.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 292: `let mid = execution_times.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 294: `((execution_times[mid - 1].as_nanos() + execution_times[mid].as_nanos()) / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 298: `execution_times[execution_times.len() / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 308: `/ execution_times.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 309: `let std_dev_execution_time = Duration::from_nanos(variance.sqrt() as u64);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 315: `(variance.sqrt()) / mean_nanos`
  - **Fix**: Division without zero check - use safe_divide()
- Line 315: `(variance.sqrt()) / mean_nanos`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 322: `let standard_error = variance.sqrt() / (execution_times.len() as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 322: `let standard_error = variance.sqrt() / (execution_times.len() as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 334: `/ measurements.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 342: `/ measurements.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 353: `std_dev_memory_usage: memory_variance.sqrt() as usize,`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 376: `let index = (percentile / 100.0 * (times.len() - 1) as f64).round() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 697: `let stats = BenchmarkStatistics::from_measurements(&measurements).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 718: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/benchmarking/performance.rs

13 issues found:

- Line 75: `(self.target_time.as_nanos() as f64 * input_scale.powf(self.scaling_factor)) as ...`
  - **Fix**: Mathematical operation .powf( without validation
- Line 85: `let actual_throughput = 1.0 / result.statistics.mean_execution_time.as_secs_f64(...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 127: `(target.target_time.as_nanos() as f64 * input_scale.powf(target.scaling_factor))...`
  - **Fix**: Mathematical operation .powf( without validation
- Line 134: `/ scaled_target_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 267: `let scale = *size as f64 / input_sizes[0] as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 296: `data_size as f64 / 1000.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 306: `data_size as f64 / 1000.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 316: `/ simd_result`
  - **Fix**: Division without zero check - use safe_divide()
- Line 365: `/ parallel_result`
  - **Fix**: Division without zero check - use safe_divide()
- Line 370: `let efficiency = actual_speedup / theoretical_speedup;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 389: `sum += (i as f64).sin().cos().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 569: `result.finalize().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 595: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/benchmarking/regression.rs

22 issues found:

- Line 216: `/ baseline.mean_execution_time_nanos as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 340: `let recent_count = (historical_results.len() / 3).max(self.config.min_historical...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 350: `let mid = execution_times.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 351: `(execution_times[mid - 1] + execution_times[mid]) / 2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 353: `execution_times[execution_times.len() / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 357: `let mut baseline = recent_results[recent_results.len() / 2].clone();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 379: `let historical_mean = historical_times.iter().sum::<f64>() / historical_times.le...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 384: `/ (historical_times.len() - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 385: `let historical_std = historical_variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 390: `(current_time - historical_mean) / (historical_std / (historical.len() as f64).s...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 390: `(current_time - historical_mean) / (historical_std / (historical.len() as f64).s...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 394: `0.5 * (1.0 - erf(z_score / std::f64::consts::SQRT_2))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 396: `0.5 * (1.0 + erf(-z_score / std::f64::consts::SQRT_2))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 427: `let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - sum_x.powi(2));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 447: `let sample_size_factor = (historical_results.len() as f64 / 10.0).min(1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 451: `(0.1 / current.coefficient_of_variation).min(1.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 564: `let t = 1.0 / (1.0 + p * x);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 596: `result.finalize().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 607: `let temp_dir = TempDir::new().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 620: `result.finalize().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 623: `detector.store_result(&result).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 624: `let analysis = detector.analyze_regression(&result).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/config.rs

23 issues found:

- Line 258: `let config_lock = config.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 265: `None => GLOBAL_CONFIG.read().unwrap().clone(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 271: `let mut global_config = GLOBAL_CONFIG.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 278: `let mut config_lock = thread_config.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 286: `let mut config_lock = thread_config.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 293: `let mut global_config = GLOBAL_CONFIG.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 307: `let mut config_lock = thread_config.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 311: `let global_config = GLOBAL_CONFIG.read().unwrap().clone();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 358: `assert!(config.get_bool("test_bool").unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 359: `assert_eq!(config.get_int("test_int").unwrap(), 42);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 360: `assert_eq!(config.get_uint("test_uint").unwrap(), 100);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 361: `assert_eq!(config.get_float("test_float").unwrap(), 3.5);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 362: `assert_eq!(config.get_string("test_string").unwrap(), "hello");`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 365: `assert_eq!(config.get_float("test_int").unwrap(), 42.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 366: `assert_eq!(config.get_int("test_uint").unwrap(), 100);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 379: `let original_value = GLOBAL_CONFIG.read().unwrap().values.get(&test_key).cloned(...`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 383: `let mut global_config = GLOBAL_CONFIG.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 390: `assert_eq!(config.get_string(&test_key).unwrap(), "test_value");`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 395: `let mut global_config = GLOBAL_CONFIG.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 411: `let mut global_config = GLOBAL_CONFIG.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 422: `assert_eq!(config.get_string(test_key).unwrap(), "thread-local");`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 429: `let locked = config.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 438: `let mut final_config = GLOBAL_CONFIG.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/config/production.rs

10 issues found:

- Line 300: `let config_key = key.strip_prefix("SCIRS_").unwrap().to_lowercase();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 848: `config.set("test_key", "test_value").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 850: `config.get("test_key").unwrap(),`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 855: `config.set("test_number", "42").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 856: `assert_eq!(config.get_typed::<i32>("test_number").unwrap(), Some(42));`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 859: `assert_eq!(config.get_or_default("missing_key", 100i32).unwrap(), 100);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 868: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 873: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 900: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 902: `let entry = config.get_entry("test_config").unwrap().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/constants.rs

26 issues found:

- Line 325: `pub const DEG_TO_RAD: f64 = PI / 180.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 328: `pub const RAD_TO_DEG: f64 = 180.0 / PI;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 334: `pub const ARCMIN: f64 = DEGREE / 60.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 340: `pub const ARCSEC: f64 = ARCMIN / 60.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 391: `pub const MIL: f64 = INCH / 1000.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 394: `pub const POINT: f64 = INCH / 72.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 400: `pub const SURVEY_FOOT: f64 = 1200.0 / 3937.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 445: `pub const BLOB: f64 = POUND * 9.80665 / 0.0254;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 451: `pub const SLUG: f64 = BLOB / 12.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 454: `pub const OUNCE: f64 = POUND / 16.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 496: `pub const TORR: f64 = ATMOSPHERE / 760.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 502: `pub const PSI: f64 = POUND_FORCE / (INCH * INCH);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 546: `pub const FLUID_OUNCE_US: f64 = GALLON_US / 128.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 552: `pub const FLUID_OUNCE_IMP: f64 = GALLON_IMP / 160.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 573: `pub const KMH: f64 = 1e3 / 3600.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 576: `pub const MPH: f64 = MILE / HOUR;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 585: `pub const KNOT: f64 = NAUTICAL_MILE / HOUR;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 597: `pub const DEGREE_FAHRENHEIT: f64 = 1.0 / 1.8;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 629: `"fahrenheit" | "f" => (value - 32.0) * 5.0 / 9.0 + ZERO_CELSIUS,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 630: `"rankine" | "r" => value * 5.0 / 9.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 638: `"fahrenheit" | "f" => (kelvin - ZERO_CELSIUS) * 9.0 / 5.0 + 32.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 639: `"rankine" | "r" => kelvin * 9.0 / 5.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 673: `pub const BTU_IT: f64 = POUND * DEGREE_FAHRENHEIT * CALORIE_IT / 1e-3;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 679: `pub const BTU_TH: f64 = POUND * DEGREE_FAHRENHEIT * CALORIE_TH / 1e-3;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 761: `SPEED_OF_LIGHT / wavelength`
  - **Fix**: Division without zero check - use safe_divide()
- Line 786: `SPEED_OF_LIGHT / frequency`
  - **Fix**: Division without zero check - use safe_divide()

### src/error/async_handling.rs

17 issues found:

- Line 68: `Err(last_error.unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 90: `Err(last_error.unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 114: `Err(last_error.unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 242: `let mut completed = self.completed_steps.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 248: `let mut errors = self.errors.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 254: `let completed = *self.completed_steps.lock().unwrap() as f64;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 255: `completed / self.total_steps as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 268: `let total_estimated = elapsed.as_secs_f64() / progress;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 278: `self.errors.lock().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 283: `!self.errors.lock().unwrap().is_empty()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 288: `let completed = *self.completed_steps.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 291: `let error_count = self.errors.lock().unwrap().len();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 336: `let mut errors = self.errors.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 354: `!self.errors.lock().unwrap().is_empty()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 359: `self.errors.lock().unwrap().len()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 364: `self.errors.lock().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 553: `assert_eq!(result.unwrap(), 42);`
  - **Fix**: Replace with ? operator or .ok_or()

### src/error/circuit_breaker.rs

8 issues found:

- Line 760: `let status = cb.status().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 770: `let status = cb.status().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 798: `assert_eq!(execute_result.unwrap(), "success");`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 807: `let result = fallback.execute(&error).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 825: `assert_eq!(result.unwrap(), "fallback");`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 830: `let cb1 = get_circuit_breaker("test1").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 831: `let cb2 = get_circuit_breaker("test1").unwrap(); // Should get same instance`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 835: `let cb3 = get_circuit_breaker("test2").unwrap(); // Different instance`
  - **Fix**: Replace with ? operator or .ok_or()

### src/error/diagnostics.rs

5 issues found:

- Line 230: `let mut history = self.error_history.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 293: `let history = self.error_history.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 369: `mem as f64 / 1_000_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 500: `let history = self.error_history.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 891: `memory as f64 / 1_000_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()

### src/error/mod.rs

3 issues found:

- Line 264: `assert_eq!(result.unwrap(), 42);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 303: `let status = breaker.status().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 360: `assert_eq!(result.unwrap(), 42);`
  - **Fix**: Replace with ? operator or .ok_or()

### src/error/recovery.rs

8 issues found:

- Line 493: `let mut state = self.state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 514: `let mut state = self.state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 520: `let mut state = self.state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 531: `let state = self.state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 606: `Err(last_error.unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 628: `Err(last_error.unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 652: `Err(last_error.unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 888: `assert_eq!(result.unwrap(), 42);`
  - **Fix**: Replace with ? operator or .ok_or()

### src/gpu/async_execution.rs

27 issues found:

- Line 111: `*self.state.lock().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 153: `*self.duration.lock().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 161: `self.callbacks.lock().unwrap().push(Box::new(callback));`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 175: `*self.duration.lock().unwrap() = Some(duration);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 176: `*self.state.lock().unwrap() = EventState::Completed;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 179: `let callbacks = std::mem::take(&mut *self.callbacks.lock().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 188: `*self.state.lock().unwrap() = EventState::Failed;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 194: `*self.state.lock().unwrap() = EventState::Cancelled;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 214: `&format!("{} callbacks", self.callbacks.lock().unwrap().len()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 279: `self.events.lock().unwrap().push(Arc::downgrade(event));`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 280: `*self.operations_count.lock().unwrap() += 1;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 285: `let events = self.events.lock().unwrap().clone();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 296: `*self.operations_count.lock().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 301: `let events = self.events.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 312: `let mut events = self.events.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 386: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 398: `self.streams.lock().unwrap().get(&id).cloned()`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 407: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 425: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 433: `if let Some(event) = self.events.lock().unwrap().get(&event_id).cloned() {`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 447: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 460: `let stream_ids: Vec<_> = self.streams.lock().unwrap().keys().cloned().collect();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 462: `if let Some(stream) = self.streams.lock().unwrap().get(&stream_id).cloned() {`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 468: `let mut events = self.events.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 474: `let streams = self.streams.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 475: `let events = self.events.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 497: `let events = self.events.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/gpu/auto_tuning.rs

8 issues found:

- Line 309: `if let Some(cached_result) = self.tuning_cache.lock().unwrap().get(&cache_key) {`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 337: `|| metrics.throughput > best_performance.as_ref().unwrap().throughput`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 376: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 384: `self.tuning_cache.lock().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 389: `self.tuning_cache.lock().unwrap().clear();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 501: `let avg_time = execution_times.iter().sum::<Duration>() / execution_times.len() ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 505: `let throughput = total_ops / avg_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 669: `let info = device_info.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/gpu/backends/metal.rs

8 issues found:

- Line 276: `let cache = self.pipeline_cache.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 306: `let mut cache = self.pipeline_cache.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 397: `let mut params = self.parameters.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 402: `let mut params = self.parameters.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 409: `let mut params = self.parameters.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 416: `let mut params = self.parameters.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 423: `let mut params = self.parameters.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 443: `let params = self.parameters.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/gpu/backends/mod.rs

6 issues found:

- Line 563: `let result = check_backend_installation(GpuBackend::Cpu).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 570: `let result = check_backend_installation(GpuBackend::Wgpu).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 576: `let result = check_backend_installation(GpuBackend::Metal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 585: `let backend = initialize_optimal_backend().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 616: `let info = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 715: `let optimal = initialize_optimal_backend().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/gpu/benchmarks.rs

11 issues found:

- Line 395: `let avg_time = execution_times.iter().sum::<Duration>() / execution_times.len() ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 443: `let avg_time = execution_times.iter().sum::<Duration>() / execution_times.len() ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 572: `let speedup = cpu_time.as_secs_f64() / result.execution_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 605: `.max_by(|a, b| a.1.partial_cmp(b.1).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 650: `/ (times.len() - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 652: `Duration::from_secs_f64(variance.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 673: `ops as f64 / time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 783: `let recommendation = if gpu_wins > category_comps.len() / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 841: `cpu_times.iter().sum::<Duration>() / cpu_times.len() as u32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 847: `gpu_times.iter().sum::<Duration>() / gpu_times.len() as u32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 853: `avg_cpu_time.as_secs_f64() / avg_gpu_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()

### src/gpu/heterogeneous.rs

6 issues found:

- Line 234: `(workload.problem_size as f64) / (self.peak_gflops * 1e9 * performance_factor);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 236: `let memory_time = (workload.memory_requirement as f64) / (self.memory_bandwidth ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 374: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 382: `let history = self.performance_history.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 387: `total_time / total_executions as u32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 418: `let history = self.performance_history.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/gpu/kernels/blas/gemm.rs

3 issues found:

- Line 268: `for (var t = 0u; t < (uniforms.k + block_size - 1u) / block_size; t = t + 1u) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 345: `for (uint t = 0; t < (k + block_size - 1) / block_size; t++) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 415: `for (int t = 0; t < (k + block_size - 1) / block_size; t++) {`
  - **Fix**: Division without zero check - use safe_divide()

### src/gpu/kernels/complex.rs

2 issues found:

- Line 372: `for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 497: `let source = kernel.source_for_backend(GpuBackend::Metal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/gpu/kernels/ml/activation.rs

4 issues found:

- Line 176: `output[i] = 1.0f / (1.0f + expf(-input[i]));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 199: `output[i] = 1.0 / (1.0 + exp(-input[i]));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 216: `output[gid] = 1.0f / (1.0f + exp(-input[gid]));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 230: `output[i] = 1.0f / (1.0f + exp(-input[i]));`
  - **Fix**: Division without zero check - use safe_divide()

### src/gpu/kernels/ml/pooling.rs

4 issues found:

- Line 127: `let out_x = global_id.y / uniforms.channels;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 180: `uint out_x = global_id.y / channels;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 229: `int out_x = get_global_id(1) / channels;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 387: `output[output_idx] = sum / (float)count;`
  - **Fix**: Division without zero check - use safe_divide()

### src/gpu/kernels/ml/softmax.rs

16 issues found:

- Line 85: `for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 127: `for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 153: `output[i] = expf(input[i] - max_val[batch_idx]) / sum_val[batch_idx];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 198: `var s = 256u / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 203: `s = s / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 235: `var s = 256u / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 240: `s = s / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 258: `output[i] = exp(input[i] - max_vals[batch_idx]) / sum_vals[batch_idx];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 296: `for (uint s = 256 / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 320: `uint batch_idx = group_id / 32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 336: `for (uint s = 256 / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 358: `uint batch_idx = group_id / 32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 362: `output[i] = exp(input[i] - max_vals[batch_idx]) / sum_vals[batch_idx];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 394: `for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 431: `for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 455: `output[i] = exp(input[i] - max_vals[batch_idx]) / sum_vals[batch_idx];`
  - **Fix**: Division without zero check - use safe_divide()

### src/gpu/kernels/reduction/mean.rs

9 issues found:

- Line 76: `for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 107: `output[0] = total_sum / (float)total_elements;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 152: `var s = 256u / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 158: `s = s / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 181: `output[0] = total_sum / f32(uniforms.total_elements);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 221: `for (uint s = 256 / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 251: `output[0] = total_sum / float(total_elements);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 285: `for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 316: `output[0] = total_sum / (float)total_elements;`
  - **Fix**: Division without zero check - use safe_divide()

### src/gpu/kernels/reduction/min_max.rs

10 issues found:

- Line 74: `for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 125: `var s = 256u / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 131: `s = s / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 176: `for (uint s = 256 / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 219: `for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 345: `for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 396: `var s = 256u / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 402: `s = s / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 447: `for (uint s = 256 / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 490: `for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()

### src/gpu/kernels/reduction/norm.rs

5 issues found:

- Line 101: `for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 153: `var s = 256u / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 159: `s = s / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 205: `for (uint s = 256 / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 249: `for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()

### src/gpu/kernels/reduction/std_dev.rs

13 issues found:

- Line 71: `for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 109: `for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 178: `var s = 256u / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 184: `s = s / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 216: `var s = 256u / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 222: `s = s / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 242: `let variance = total_variance / f32(uniforms.total_elements - 1u);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 279: `for (uint s = 256 / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 320: `for (uint s = 256 / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 347: `float variance = total_variance / float(total_elements - 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 378: `for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 416: `for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 444: `float variance = total_variance / (float)(total_elements - 1);`
  - **Fix**: Division without zero check - use safe_divide()

### src/gpu/kernels/reduction/sum.rs

5 issues found:

- Line 75: `for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 127: `var s = 256u / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 133: `s = s / 2u;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 179: `for (uint s = 256 / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 223: `for (unsigned int s = get_local_size(0) / 2; s > 0; s >>= 1) {`
  - **Fix**: Division without zero check - use safe_divide()

### src/gpu/kernels/transform/convolution.rs

1 issues found:

- Line 361: `let out_x = global_id.y / uniforms.out_channels;`
  - **Fix**: Division without zero check - use safe_divide()

### src/gpu/mod.rs

5 issues found:

- Line 963: `let context = GpuContext::new(GpuBackend::Cpu).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 974: `let context = GpuContext::new(GpuBackend::Cpu).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1012: `let kernel = compiler.compile("dummy kernel source").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1022: `let context = GpuContext::new(GpuBackend::Cpu).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1031: `let context = GpuContext::new(GpuBackend::Cpu).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/gpu_registry.rs

4 issues found:

- Line 325: `let mut registry = registry.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 333: `let mut registry = registry.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 340: `let registry = registry.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 347: `let registry = registry.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/io.rs

25 issues found:

- Line 229: `format!("{:.2} TB", size as f64 / TB as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 231: `format!("{:.2} GB", size as f64 / GB as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 233: `format!("{:.2} MB", size as f64 / MB as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 235: `format!("{:.2} KB", size as f64 / KB as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 249: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 253: `write_string(&file_path, test_str).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 255: `let contents = read_to_string(&file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 261: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 265: `write_bytes(&file_path, &test_bytes).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 267: `let contents = read_to_bytes(&file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 273: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 277: `let mut file = File::create(&file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 278: `writeln!(file, "Line 1").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 279: `writeln!(file, "Line 2").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 280: `writeln!(file, "Line 3").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 288: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 295: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 300: `File::create(&file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 307: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 312: `std::fs::create_dir(&dir_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 319: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 324: `create_directory(&dir_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 331: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 335: `write_string(&file_path, test_str).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 337: `let size = file_size(&file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/logging.rs

45 issues found:

- Line 140: `let mut global_config = LOGGER_CONFIG.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 146: `let mut config = LOGGER_CONFIG.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 152: `let mut config = LOGGER_CONFIG.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 270: `let mut handlers = LOG_HANDLERS.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 276: `let mut handlers = LOG_HANDLERS.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 282: `let mut handlers = LOG_HANDLERS.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 331: `let config = LOGGER_CONFIG.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 352: `let handlers = LOG_HANDLERS.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 494: `let percent = (self.current as f64 / self.total as f64) * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 497: `let time_per_item = elapsed.as_secs_f64() / self.current as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 532: `(self.current as f64 / self.total as f64) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 547: `let time_per_item = elapsed.as_secs_f64() / self.current as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 557: `let handlers = LOG_HANDLERS.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 666: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 680: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 721: `let mut entries = self.entries.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 722: `let mut stats = self.stats.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 732: `let removed = entries.pop_front().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 769: `self.entries.read().unwrap().iter().cloned().collect()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 776: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 787: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 796: `self.stats.read().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 801: `self.entries.write().unwrap().clear();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 802: `*self.stats.write().unwrap() = AggregationStats::default();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 842: `let mut last_reset = self.last_reset.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 847: `let actual_rate = count as f64 / elapsed.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 849: `let mut current_rate = self.current_rate.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 867: `let current_rate = count as f64 / elapsed_secs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 868: `let max_rate = *self.max_rate.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 876: `let mut max_rate = self.max_rate.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 892: `let current_rate = *self.current_rate.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 893: `let max_rate = *self.max_rate.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 904: `*self.current_rate.lock().unwrap() = 0.0;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 905: `*self.last_reset.lock().unwrap() = Instant::now();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 965: `let rate_limiters = self.rate_limiters.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 972: `let mut rate_limiters = self.rate_limiters.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1028: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1048: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1075: `let rate_limiters = self.rate_limiters.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1110: `let mut nodes = self.nodes.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1116: `let mut nodes = self.nodes.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1135: `let nodes_guard = nodes.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1174: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1179: `"nodes_count": self.nodes.read().unwrap().len(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1313: `let json_export = logger.export_logs_json().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/logging/distributed.rs

22 issues found:

- Line 227: `let mut running = self.running.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 242: `self.nodes.lock().unwrap().insert(node.id.clone(), node);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 254: `*self.running.lock().unwrap() = false;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 266: `self.buffer.lock().unwrap().push(entry);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 269: `if self.buffer.lock().unwrap().len() >= self.config.buffer_size {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 290: `self.buffer.lock().unwrap().push(entry);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 295: `let mut buffer = self.buffer.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 322: `let nodes = self.nodes.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 334: `let nodes = self.nodes.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 352: `let nodes = self.nodes.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 377: `while *running.lock().unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 381: `let buffer_size = buffer.lock().unwrap().len();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 396: `while *running.lock().unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 400: `let mut nodes_guard = nodes.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 415: `let buffer_size = self.buffer.lock().unwrap().len();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 416: `let nodes = self.nodes.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 481: `*self.running.lock().unwrap() = true;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 491: `*self.running.lock().unwrap() = false;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 496: `let mut logs = self.collected_logs.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 511: `let logs = self.collected_logs.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 533: `let logs = self.collected_logs.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 631: `assert_eq!(*logger.running.lock().unwrap(), false);`
  - **Fix**: Replace with ? operator or .ok_or()

### src/logging/progress/adaptive.rs

15 issues found:

- Line 55: `let base_interval = Duration::from_secs_f64(1.0 / self.target_frequency);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 91: `self.processing_speeds.iter().sum::<f64>() / self.processing_speeds.len() as f64...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 102: `/ self.processing_speeds.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 104: `let std_dev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 119: `self.processing_speeds.iter().sum::<f64>() / self.processing_speeds.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 128: `let recent_half = self.processing_speeds.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 130: `self.processing_speeds.iter().take(recent_half).sum::<f64>() / recent_half as f6...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 133: `/ (self.processing_speeds.len() - recent_half) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 136: `late_speeds / early_speeds`
  - **Fix**: Division without zero check - use safe_divide()
- Line 240: `let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 243: `let remaining_time = remaining as f64 / slope;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 265: `Duration::from_secs_f64(remaining as f64 / smoothed_speed)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 277: `let recent_count = (self.speed_history.len() / 2).clamp(1, 10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 281: `recent_speeds.iter().map(|&&s| s).sum::<f64>() / recent_speeds.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 284: `Duration::from_secs_f64(remaining as f64 / avg_speed)`
  - **Fix**: Division without zero check - use safe_divide()

### src/logging/progress/formats.rs

2 issues found:

- Line 327: `std::time::Duration::from_secs_f64(1.0 / self.fps)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 398: `let spec = spec.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/logging/progress/multi.rs

1 issues found:

- Line 133: `total_progress / self.trackers.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()

### src/logging/progress/renderer.rs

4 issues found:

- Line 59: `let filled_width = ((percentage / 100.0) * width as f64) as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 114: `let filled_width = ((percentage / 100.0) * width as f64) as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 200: `let filled_width = ((percentage / 100.0) * width as f64) as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 227: `let progress = percentage / 100.0 * width as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### src/logging/progress/statistics.rs

14 issues found:

- Line 62: `self.percentage = (self.processed as f64 / self.total as f64) * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 75: `let speed = items_diff as f64 / time_diff.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 85: `self.recent_speeds.iter().sum::<f64>() / self.recent_speeds.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 90: `self.items_per_second = self.processed as f64 / self.elapsed.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 96: `let remaining_seconds = remaining_items as f64 / self.items_per_second;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 114: `self.processed as f64 / self.elapsed.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 143: `self.recent_speeds.iter().sum::<f64>() / self.recent_speeds.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 147: `let remaining_seconds = remaining_items as f64 / recent_avg;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 158: `self.items_per_second / avg_rate`
  - **Fix**: Division without zero check - use safe_divide()
- Line 173: `let mins = total_secs / 60;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 180: `let hours = mins / 60;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 187: `let days = hours / 24;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 196: `format!("{:.1}M it/s", rate / 1000000.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 198: `format!("{:.1}k it/s", rate / 1000.0)`
  - **Fix**: Division without zero check - use safe_divide()

### src/logging/progress/tracker.rs

8 issues found:

- Line 213: `let mut stats = self.stats.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 236: `let stats = self.stats.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 253: `let mut stats = self.stats.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 284: `self.stats.lock().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 289: `let stats = self.stats.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 296: `let stats = self.stats.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 308: `stats.processed as f64 / stats.total as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 334: `let stats = self.stats.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/logging/rate_limiting.rs

7 issues found:

- Line 310: `let time_to_token = Duration::from_secs_f64((1.0 - self.tokens) / refill_rate);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 361: `(base_max_events as f64 * (2.0 - current_load / load_threshold)).max(1.0) as u32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 390: `(recent_events as f64 / 100.0).min(1.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 840: `let error_decision = limiter.should_allow(&error_event).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 844: `let debug_decision = limiter.should_allow(&debug_event).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 848: `let stats = limiter.get_stats().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 885: `let decision = limiter.should_allow(&critical_event).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory.rs

9 issues found:

- Line 105: `let n_chunks = (array_shape[i] + chunk_shape[i] - 1) / chunk_shape[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 181: `let n_chunks = (array_shape[i] + chunk_shape[i] - 1) / chunk_shape[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 339: `let mut pools = self.pools.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 361: `let mut pools = self.pools.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 439: `let mut allocations = self.allocations.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 445: `let mut allocations = self.allocations.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 453: `let allocations = self.allocations.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 459: `let allocations = self.allocations.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 465: `let mut allocations = self.allocations.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory/compressed_buffers.rs

9 issues found:

- Line 119: `self.original_size as f64 / self.compressed_data.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 191: `array.as_slice().unwrap().to_vec()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 294: `self.total_original_size as f64 / self.total_compressed_size as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 349: `self.total_original_size as f64 / 1024.0 / 1024.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 351: `self.total_compressed_size as f64 / 1024.0 / 1024.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 354: `self.memory_saved as f64 / 1024.0 / 1024.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 393: `data.len() as f64 / compressed.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 402: `data.len() as f64 / compressed.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 553: `let compression_ratio = buffer.compressed_size() as f64 / buffer.original_size()...`
  - **Fix**: Division without zero check - use safe_divide()

### src/memory/cross_device.rs

19 issues found:

- Line 411: `let mut devices = self.devices.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 415: `let mut default_device = self.default_device.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 425: `let devices = self.devices.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 430: `let mut default_device = self.default_device.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 438: `self.default_device.read().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 447: `let devices = self.devices.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 464: `let mut allocations = self.allocations.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 494: `let devices = self.devices.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 543: `let devices = self.devices.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 552: `let allocations = self.allocations.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 553: `let devices = self.devices.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 603: `let mut allocations = self.allocations.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 623: `let mut counter = self.allocation_counter.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 633: `let mut allocations = self.allocations.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 643: `let mut allocations = self.allocations.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 733: `let devices = self.manager.devices.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 753: `let devices = self.manager.devices.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 775: `let mut allocations = self.manager.allocations.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 830: `(self.allocated_bytes as f64 / self.total_bytes as f64) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()

### src/memory/leak_detection.rs

14 issues found:

- Line 784: `leaks.iter().map(|leak| leak.confidence).sum::<f64>() / total_leaks as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1017: `let detector = LeakDetector::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1019: `assert!(!*detector.monitoring_active.lock().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1025: `let detector = LeakDetector::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1027: `let checkpoint = detector.create_checkpoint("test").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1035: `let detector = LeakDetector::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1037: `detector.track_allocation(1024, 0x12345678).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1038: `detector.track_allocation(2048, 0x87654321).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1040: `let count = detector.get_active_allocation_count().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1043: `detector.track_deallocation(0x12345678).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1044: `let count = detector.get_active_allocation_count().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1051: `let detector = LeakDetector::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1054: `let _guard = LeakCheckGuard::new(&detector, "test_guard").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1056: `detector.track_allocation(1024 * 1024, 0x12345678).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory/metrics/analytics.rs

16 issues found:

- Line 338: `let start_time = history.front().unwrap().0;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 364: `let mean_x = sum_x / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 365: `let mean_y = sum_y / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 374: `let slope = numerator / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 389: `1.0 - (ss_res / ss_tot).max(0.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 469: `for cycle_len in 3..values.len() / 3 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 475: `let avg = (values[i] + values[i - cycle_len]) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 477: `correlation += 1.0 - (diff / avg).min(1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 569: `(values[i] as f64 - values[i - 1] as f64).abs() / values[i - 1] as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 641: `total_allocated as f64 / current_usage as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 659: `let allocation_frequency = allocation_count as f64 / duration.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 663: `duration / allocation_count as u32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 704: `let mean = allocation_sizes.iter().sum::<usize>() as f64 / allocation_sizes.len(...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 712: `/ allocation_sizes.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 714: `let std_dev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 717: `(std_dev / mean).min(1.0)`
  - **Fix**: Division without zero check - use safe_divide()

### src/memory/metrics/collector.rs

33 issues found:

- Line 156: `let mut rng = self.rng.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 169: `let mut events = self.events.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 184: `let mut current_usage = self.current_usage.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 189: `let mut peak_usage = self.peak_usage.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 194: `let mut allocation_count = self.allocation_count.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 199: `let mut total_allocated = self.total_allocated.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 204: `let mut avg_allocation_size = self.avg_allocation_size.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 208: `*avg = (*avg * (*count as f64 - 1.0) + event.size as f64) / *count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 212: `let mut current_usage = self.current_usage.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 225: `let mut current_usage = self.current_usage.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 235: `let mut peak_usage = self.peak_usage.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 248: `let current_usage = self.current_usage.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 254: `let peak_usage = self.peak_usage.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 260: `let current_usage = self.current_usage.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 266: `let peak_usage = self.peak_usage.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 278: `let allocation_count = self.allocation_count.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 281: `let total_allocated = self.total_allocated.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 284: `let avg_allocation_size = self.avg_allocation_size.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 287: `let peak_usage = self.peak_usage.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 300: `let current_usage = self.current_usage.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 301: `let peak_usage = self.peak_usage.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 302: `let allocation_count = self.allocation_count.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 303: `let total_allocated = self.total_allocated.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 304: `let avg_allocation_size = self.avg_allocation_size.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 339: `let mut events = self.events.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 342: `let mut current_usage = self.current_usage.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 345: `let mut peak_usage = self.peak_usage.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 348: `let mut allocation_count = self.allocation_count.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 351: `let mut total_allocated = self.total_allocated.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 354: `let mut avg_allocation_size = self.avg_allocation_size.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 360: `let events = self.events.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 474: `let comp1_stats = collector.get_allocation_stats("Component1").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 485: `let comp1_report = report.component_stats.get("Component1").unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?

### src/memory/metrics/mod.rs

1 issues found:

- Line 332: `let test_comp = report.component_stats.get("TestComponent").unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?

### src/memory/metrics/profiler.rs

20 issues found:

- Line 215: `let mut current = self.current_session.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 221: `self.analytics.lock().unwrap().clear();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 229: `let mut current = self.current_session.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 235: `let analytics = self.analytics.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 269: `let mut history = self.results_history.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 294: `self.analytics.lock().unwrap().record_event(event);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 319: `let analytics_guard = analytics.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 339: `memory_report.total_current_usage / (1024 * 1024)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 392: `memory_report.total_allocated_bytes as f64 / duration.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 399: `(bytes_per_second / (100.0 * 1024.0 * 1024.0 * 1024.0)).min(1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 406: `/ pattern_analysis.len().max(1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 430: `let total_memory_mb = memory_report.total_current_usage / (1024 * 1024);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 462: `/ pattern_analysis.len().max(1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 550: `self.results_history.read().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 555: `self.current_session.lock().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 561: `let analytics = self.analytics.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 572: `self.analytics.lock().unwrap().clear();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 573: `self.results_history.write().unwrap().clear();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 574: `*self.current_session.lock().unwrap() = None;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 616: `let result = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory/metrics/reporter.rs

11 issues found:

- Line 18: `format!("{:.2} GB", bytes as f64 / GB as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 20: `format!("{:.2} MB", bytes as f64 / MB as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 22: `format!("{:.2} KB", bytes as f64 / KB as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 36: `let mins = total_secs / 60;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 43: `let hours = mins / 60;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 106: `let reuse_ratio = stats.total_allocated as f64 / stats.peak_usage as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 112: `let alloc_per_sec = stats.allocation_count as f64 / self.duration.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 127: `stats.total_allocated as f64 / stats.peak_usage as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 133: `stats.allocation_count as f64 / self.duration.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 210: `(stats.peak_usage as f64 / max_usage as f64 * CHART_WIDTH as f64) as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 212: `(stats.current_usage as f64 / max_usage as f64 * CHART_WIDTH as f64) as usize;`
  - **Fix**: Division without zero check - use safe_divide()

### src/memory/metrics/snapshot.rs

5 issues found:

- Line 937: `retrieved.unwrap().id,`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1115: `thread1.join().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1116: `thread2.join().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1221: `thread1.join().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1222: `thread2.join().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory/out_of_core.rs

49 issues found:

- Line 231: `let chunks = self.chunks.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 259: `let mut chunks = self.chunks.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 260: `let mut metadata_map = self.metadata.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 261: `let mut access_order = self.access_order.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 262: `let mut current_memory = self.current_memory.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 286: `let mut chunks = self.chunks.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 287: `let mut metadata_map = self.metadata.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 288: `let mut access_order = self.access_order.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 289: `let mut current_memory = self.current_memory.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 306: `let mut metadata_map = self.metadata.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 312: `let mut access_order = self.access_order.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 326: `let current_memory = *self.current_memory.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 327: `let current_count = self.chunks.read().unwrap().len();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 356: `if let Some(metadata) = self.metadata.read().unwrap().get(&chunk_id) {`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 375: `let access_order = self.access_order.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 376: `let metadata_map = self.metadata.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 415: `let chunks = self.chunks.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 416: `let metadata_map = self.metadata.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 417: `let current_memory = *self.current_memory.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 431: `let metadata_map = self.metadata.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 443: `let mut metadata_map = self.metadata.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 451: `let mut metadata_map = self.metadata.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 511: `let mut handles = self.file_handles.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 534: `let mut file = file_handle.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 545: `let mut file = file_handle.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 555: `let file = file_handle.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 562: `let mut registry = self.chunk_registry.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 569: `let mut registry = self.chunk_registry.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 575: `let handles = self.file_handles.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 577: `let mut file = handle.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 653: `.map(|(&idx, &chunk_size)| idx / chunk_size)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 663: `let chunk_map = self.chunk_map.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 689: `let chunk_map = self.chunk_map.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 706: `let chunk_map = self.chunk_map.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 770: `let mut chunk_map = self.chunk_map.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 908: `let chunk_map = self.chunk_map.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 938: `let chunk_map = self.chunk_map.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 992: `let chunk_start = start / chunk_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 993: `let chunk_end = (end - 1) / chunk_size + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1087: `let mut backends = self.storage_backends.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1102: `let storage_backends = self.storage_backends.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1121: `let mut arrays = self.arrays.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1132: `let arrays = self.arrays.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1141: `let mut arrays = self.arrays.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1147: `let arrays = self.arrays.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1153: `let arrays = self.arrays.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1260: `let storage = Arc::new(FileStorageBackend::new(data_path.parent().unwrap())?);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1265: `data_path.file_stem().unwrap().to_string_lossy().to_string(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1460: `let temp_dir = TempDir::new().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory/safety.rs

20 issues found:

- Line 138: `let pressure = current as f64 / limit as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 198: `current / limit`
  - **Fix**: Division without zero check - use safe_divide()
- Line 237: `let allocations = self.allocations.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 241: `total_size / total_allocations`
  - **Fix**: Division without zero check - use safe_divide()
- Line 363: `"Arithmetic error in division: {} / {}",`
  - **Fix**: Division without zero check - use safe_divide()
- Line 630: `assert_eq!(SafeArithmetic::safe_add(5u32, 10u32).unwrap(), 15u32);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 634: `assert_eq!(SafeArithmetic::safe_sub(10u32, 5u32).unwrap(), 5u32);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 638: `assert_eq!(SafeArithmetic::safe_mul(5u32, 10u32).unwrap(), 50u32);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 642: `assert_eq!(SafeArithmetic::safe_div(10u32, 2u32).unwrap(), 5u32);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 651: `assert_eq!(*SafeArrayOps::safe_index(&array, 2).unwrap(), 3);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 655: `let slice = SafeArrayOps::safe_slice(&array, 1, 4).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 668: `*cleanup_called_clone.lock().unwrap() = true;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 672: `assert!(*cleanup_called.lock().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 678: `assert_eq!(safe_op!(add 5u32, 10u32).unwrap(), 15u32);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 679: `assert_eq!(safe_op!(sub 10u32, 5u32).unwrap(), 5u32);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 680: `assert_eq!(safe_op!(mul 5u32, 10u32).unwrap(), 50u32);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 681: `assert_eq!(safe_op!(div 10u32, 2u32).unwrap(), 5u32);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 685: `assert_eq!(*safe_get!(&array, 2).unwrap(), 3);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 695: `tracker.track_allocation(ptr1, 1024, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 696: `tracker.track_allocation(ptr2, 2048, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory_efficient/adaptive_chunking.rs

23 issues found:

- Line 369: `let new_size = (chunk_size / row_length)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 411: `let new_size = (chunk_size / plane_size)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 423: `let new_size = (chunk_size / row_length)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 459: `total_elements / target_num_chunks`
  - **Fix**: Division without zero check - use safe_divide()
- Line 476: `let actual_chunks = total_elements / chunk_size`
  - **Fix**: Division without zero check - use safe_divide()
- Line 612: `let work_share = (total_work as f64 * performance / total_performance) as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 630: `let performance = work_amount as f64 / execution_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 735: `((size / CACHE_LINE_SIZE) + 1) * CACHE_LINE_SIZE`
  - **Fix**: Division without zero check - use safe_divide()
- Line 750: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 755: `let mut file = File::create(&file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 757: `file.write_all(&val.to_ne_bytes()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 762: `let mmap = MemoryMappedArray::<f64>::open(&file_path, &[100_000]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 773: `let result = mmap.adaptive_chunking(params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 798: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 807: `let mut file = File::create(&file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 809: `file.write_all(&val.to_ne_bytes()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 814: `let mmap = MemoryMappedArray::<f64>::open(&file_path, &[rows, cols]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 824: `let result = mmap.adaptive_chunking(params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 849: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 854: `let mut file = File::create(&file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 856: `file.write_all(&val.to_ne_bytes()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 861: `let mmap = MemoryMappedArray::<f64>::open(&file_path, &[1_000_000]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 871: `let result = mmap.adaptive_chunking(params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory_efficient/adaptive_prefetch.rs

10 issues found:

- Line 246: `.max_by(|a, b| a.q_value.partial_cmp(&b.q_value).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 505: `let row_major_pct = row_major_matches as f64 / total_pairs as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 506: `let col_major_pct = col_major_matches as f64 / total_pairs as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 566: `let expected_changes = indices.len() / row_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 567: `direction_changes >= expected_changes / 2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 601: `for window_size in 2..=std::cmp::min(indices.len() / 2, 10) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 606: `Some(s) => s / window_size,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 618: `let threshold = (indices.len() - window_size) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 661: `let latest = self.history.back().unwrap().0;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 715: `for i in 1..=count / 2 {`
  - **Fix**: Division without zero check - use safe_divide()

### src/memory_efficient/chunked.rs

2 issues found:

- Line 60: `let chunk_size = chunk_size_bytes / elem_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 69: `let elements = bytes / elem_size;`
  - **Fix**: Division without zero check - use safe_divide()

### src/memory_efficient/compressed_memmap.rs

35 issues found:

- Line 552: `let num_elements = uncompressed_size / std::mem::size_of::<A>();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 593: `let slice = &mut result.as_slice_mut().unwrap()[start..end];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 646: `let block_idx = flat_index / self.metadata.block_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 737: `let block_idx = source_flat_idx / self.metadata.block_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 763: `let result_slice = result.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 980: `let start_block = start / self.metadata.block_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 981: `let end_block = (end - 1) / self.metadata.block_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1063: `let cache = self.cache.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1090: `let mut cache = self.cache.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1121: `let mut cache = self.cache.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1148: `let mut cache = self.cache.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1155: `let cache = self.cache.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1162: `let cache = self.cache.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1176: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1191: `let cmm = builder.create_from_raw(&data, &[1000], &file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1200: `let val = cmm.get(&[i * 100]).unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 1205: `let slice = cmm.slice(&[(200, 300)]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1214: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1219: `let array = cmm.readonly_array().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1229: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1244: `let cmm = builder.create(&data, &file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1254: `let val = cmm.get(&[i, j]).unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 1260: `let slice = cmm.slice(&[(2, 5), (3, 7)]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1272: `let array = cmm.readonly_array().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1284: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1305: `let cmm = builder.create_from_raw(&data, &[1000], &file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1308: `let array = cmm.readonly_array().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1318: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1328: `.create_from_raw(&data, &[1000], &file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1333: `small_cache.preload_block(i).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1341: `let val = small_cache.get(&[0]).unwrap(); // Block 0`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 1351: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1361: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1364: `cmm.preload_block(5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1370: `let val = cmm.get(&[550]).unwrap(); // In block 5`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?

### src/memory_efficient/cross_device.rs

13 issues found:

- Line 378: `let mut cache = self.cache.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 542: `let mut cache = self.cache.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 573: `let entry = cache.remove(&key).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 604: `let mut cache = self.cache.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 888: `let mut free_buffers = self.free_buffers.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 937: `let mut free_buffers = self.free_buffers.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 947: `let mut free_buffers = self.free_buffers.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1014: `let first = *host_array.iter().next().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1103: `let manager = self.memory_managers.get(&device).unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 1204: `let pool = self.memory_pools.get(&device).unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 1216: `let pool = self.memory_pools.get(&device).unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 1233: `let mut active_transfers = self.active_transfers.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1239: `let mut active_transfers = self.active_transfers.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory_efficient/cross_file_prefetch.rs

16 issues found:

- Line 344: `for i in 1..=max_count / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 382: `let scale = (r2 as f64 - r1 as f64) / (p2 as f64 - p1 as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 560: `let related_datasets = self.correlations.get(&access.dataset).unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 564: `correlations.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 612: `result.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 621: `.max_by(|(_, a), (_, b)| a.strength.partial_cmp(&b.strength).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 675: `INSTANCE.as_ref().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 792: `let block_idx = idx / block_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1118: `manager.record_access(access1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1119: `manager.record_access(access2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1152: `let mut prefetched = self.prefetched_indices.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1158: `let mut prefetched_all = self.prefetched_all.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1210: `manager.record_access(access1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1211: `manager.record_access(access2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1223: `manager.record_access(access).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1226: `let prefetched = prefetcher2.prefetched_indices.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory_efficient/fusion.rs

2 issues found:

- Line 153: `let mut registry = FUSION_REGISTRY.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 165: `let registry = FUSION_REGISTRY.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory_efficient/lazy_array.rs

1 issues found:

- Line 334: `self.zip_with(other, |a, b| a.clone() / b.clone())`
  - **Fix**: Division without zero check - use safe_divide()

### src/memory_efficient/memmap.rs

8 issues found:

- Line 216: `if element_size > 0 && self.size > isize::MAX as usize / element_size {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 703: `if element_size > 0 && self.size > isize::MAX as usize / element_size {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 953: `let total_elements = file_size / element_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 970: `let total_elements = file_size / element_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 987: `let total_elements = file_size / element_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1003: `let total_elements = file_size / element_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1023: `let total_elements = file_size / element_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1037: `let total_elements = file_size / element_size;`
  - **Fix**: Division without zero check - use safe_divide()

### src/memory_efficient/memmap_chunks.rs

13 issues found:

- Line 374: `ChunkingStrategy::Auto => (self.array.size / 100).max(1),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 377: `let elements_per_chunk = bytes / element_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 460: `let optimal_chunk_size = (total_elements / 100).max(1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 466: `let elements_per_chunk = bytes / element_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 488: `(total_elements / 100).max(1)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 492: `let elements_per_chunk = bytes / element_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 528: `(total_elements / 100).max(1)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 531: `let elements_per_chunk = bytes / element_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 548: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 606: `(total_elements / 100).max(1)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 610: `let elements_per_chunk = bytes / element_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 658: `(total_elements / 100).max(1)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 661: `let elements_per_chunk = bytes / element_size;`
  - **Fix**: Division without zero check - use safe_divide()

### src/memory_efficient/memmap_slice.rs

16 issues found:

- Line 590: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 595: `let mut file = File::create(&file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 597: `file.write_all(&val.to_ne_bytes()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 602: `let mmap = MemoryMappedArray::<f64>::open(&file_path, &[100]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 605: `let slice = mmap.slice_1d(10..20).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 608: `let array = slice.load().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 620: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 627: `MemoryMappedArray::<f64>::save_array(&data, &file_path, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 631: `MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 638: `let orig_array = mmap.as_array::<ndarray::Ix2>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 653: `let slice = mmap.slice_2d(2..5, 3..7).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 660: `let array = slice.load().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 693: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 700: `let mmap = create_mmap::<f64, _, _>(&data, &file_path, AccessMode::Write, 0).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 704: `let slice = mmap.slice(s![2..5, 3..7]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 707: `let array: ndarray::Array2<f64> = slice.load().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory_efficient/memory_layout.rs

21 issues found:

- Line 291: `Ok((linear_idx / self.element_size as isize) as usize)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 330: `indices[i] = (remaining as isize / stride) as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 781: `assert_eq!(LayoutOrder::parse("C").unwrap(), LayoutOrder::C);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 782: `assert_eq!(LayoutOrder::parse("F").unwrap(), LayoutOrder::Fortran);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 783: `assert_eq!(LayoutOrder::parse("fortran").unwrap(), LayoutOrder::Fortran);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 784: `assert_eq!(LayoutOrder::parse("A").unwrap(), LayoutOrder::Any);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 785: `assert_eq!(LayoutOrder::parse("K").unwrap(), LayoutOrder::Keep);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 823: `assert_eq!(layout.linear_index(&[0, 0]).unwrap(), 0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 824: `assert_eq!(layout.linear_index(&[0, 1]).unwrap(), 1);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 825: `assert_eq!(layout.linear_index(&[1, 0]).unwrap(), 4);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 826: `assert_eq!(layout.linear_index(&[2, 3]).unwrap(), 11);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 837: `assert_eq!(layout.multi_index(0).unwrap(), vec![0, 0]);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 838: `assert_eq!(layout.multi_index(1).unwrap(), vec![0, 1]);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 839: `assert_eq!(layout.multi_index(4).unwrap(), vec![1, 0]);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 840: `assert_eq!(layout.multi_index(11).unwrap(), vec![2, 3]);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 849: `let f_layout = c_layout.to_order(LayoutOrder::Fortran).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 859: `let transposed = layout.transpose(Some(&[2, 0, 1])).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 864: `let default_transposed = layout.transpose(None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 871: `let reshaped = layout.reshape(&[3, 8]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 897: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 902: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory_efficient/out_of_core.rs

7 issues found:

- Line 181: `let first_dim_size = self.size / chunk_shape.iter().skip(1).product::<usize>().m...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 183: `actual_chunk_size / chunk_shape.iter().skip(1).product::<usize>().max(1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 214: `ChunkingStrategy::Auto => OPTIMAL_CHUNK_SIZE / std::mem::size_of::<A>(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 216: `ChunkingStrategy::FixedBytes(bytes) => bytes / std::mem::size_of::<A>(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 224: `ChunkingStrategy::Auto => OPTIMAL_CHUNK_SIZE / std::mem::size_of::<A>(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 226: `ChunkingStrategy::FixedBytes(bytes) => bytes / std::mem::size_of::<A>(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 414: `index_map.push(sorted_indices.iter().position(|&x| x == idx).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory_efficient/pattern_recognition.rs

25 issues found:

- Line 278: `if sequential_count >= indices.len() * 3 / 4 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 321: `if best_stride_count >= indices.len() * 2 / 3 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 370: `let row = idx / cols;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 452: `let prev_row = prev_idx / cols;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 455: `let curr_row = curr_idx / cols;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 468: `if (diagonal_matches >= expected_transitions / 3 || diagonal_matches >= 3)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 493: `let prev_row = prev_idx / cols;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 496: `let curr_row = curr_idx / cols;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 508: `if (anti_diagonal_matches >= expected_transitions / 3 || anti_diagonal_matches >...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 558: `let row = idx / cols;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 561: `let block_row = row / block_height;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 562: `let block_col = col / block_width;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 609: `if count >= indices.len() / 3 && stride > 1 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 660: `let _center_row = center_idx / cols;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 721: `let density = unique_indices as f64 / (max_idx + 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 855: `let row = current_idx / cols;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 923: `let row = current_idx / cols;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 927: `let block_row = row / *block_height;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 928: `let block_col = col / *block_width;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 959: `let next_block_row = if block_col + 1 < cols / *block_width {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 965: `let next_block_col = if block_col + 1 < cols / *block_width {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 973: `let row_offset = i / *block_width;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 994: `let blocks_advanced = offset / block_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1012: `let row = current_idx / cols;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1053: `for i in 1..=prefetch_count / 2 {`
  - **Fix**: Division without zero check - use safe_divide()

### src/memory_efficient/prefetch.rs

9 issues found:

- Line 187: `let mut prev = *self.history.front().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 204: `let stride = self.history.get(1).unwrap() - self.history.front().unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 205: `prev = *self.history.front().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 250: `let latest = *self.history.back().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 368: `self.stats.hit_rate = self.stats.prefetch_hits as f64 / total as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 823: `let block_idx = flat_index / self.metadata().block_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 995: `let block_idx = flat_index / block_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1002: `let min_block = *blocks.iter().min().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1003: `let max_block = *blocks.iter().max().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory_efficient/resource_aware.rs

11 issues found:

- Line 75: `self.memory_usage as f64 / (self.memory_usage + self.memory_available) as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 312: `let latest = self.snapshots.back().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 349: `let latest = self.snapshots.back().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 426: `let avg_cpu = cpu_sum / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 427: `let avg_memory_pressure = memory_pressure_sum / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 428: `let avg_io_bytes = io_bytes_sum / count as u64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 433: `let newest = self.snapshots.back().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 444: `combined_pressure: self.snapshots.back().unwrap().combined_pressure(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 519: `.map(|cpu| cpu.cpu_usage() as f64 / 100.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 521: `cpu_usage / system.cpus().len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 535: `return (loadavg[0] / num_cpus).min(1.0);`
  - **Fix**: Division without zero check - use safe_divide()

### src/memory_efficient/streaming.rs

67 issues found:

- Line 293: `let mut guard = self.mutex.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 305: `guard = self.condvar.wait(guard).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 327: `let mut guard = self.mutex.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 339: `guard = self.condvar.wait(guard).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 361: `let mut guard = self.mutex.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 395: `guard = self.condvar.wait(guard).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 428: `let _guard = self.mutex.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 434: `let _guard = self.mutex.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 440: `let _guard = self.mutex.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 448: `let _guard = self.mutex.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 454: `let _guard = self.mutex.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 517: `let mut state = self.state.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 531: `let mut start_time = self.start_time.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 579: `let current_state = state.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 589: `Duration::from_secs_f64(config.min_batch_size as f64 / rate_limit as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 604: `let stats_guard = stats.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 612: `(config.max_batch_size + config.min_batch_size) / 2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 624: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 632: `let mut current_state = state.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 642: `let mut stats_guard = stats.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 695: `let mut stats_guard = stats.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 702: `stats_guard.avg_batch_size = total_items as f64 / total_batches as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 708: `/ total_batches as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 711: `if let Some(start) = *start_time.read().unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 714: `stats_guard.avg_throughput = total_items as f64 / uptime_seconds;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 718: `let buffer_len = input_buffer.lock().unwrap().len();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 731: `match output_buffer.lock().unwrap().push_batch(output_batch) {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 735: `let mut stats_guard = stats.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 744: `let mut stats_guard = stats.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 757: `let mut state = self.state.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 771: `self.input_buffer.lock().unwrap().close();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 792: `let state = self.state.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 801: `self.input_buffer.lock().unwrap().push(data)`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 807: `let state = self.state.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 816: `self.input_buffer.lock().unwrap().push_batch(data)`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 822: `let state = self.state.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 834: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 850: `let state = self.state.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 861: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 867: `*self.state.read().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 872: `self.stats.read().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 877: `self.input_buffer.lock().unwrap().is_empty()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 878: `&& self.output_buffer.lock().unwrap().is_empty()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 884: `let state = self.state.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 893: `self.input_buffer.lock().unwrap().clear();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 894: `self.output_buffer.lock().unwrap().clear();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 903: `if *self.state.read().unwrap() == StreamState::Running {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 951: `self.processor.lock().unwrap().start()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 956: `self.processor.lock().unwrap().stop()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 961: `self.processor.lock().unwrap().state()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 966: `self.processor.lock().unwrap().stats()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1106: `let mut state = self.state.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1163: `let current_state = state.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1181: `let mut error_context_guard = error_context.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1194: `let mut current_state = state.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1218: `let mut error_context_guard = error_context.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1230: `let mut current_state = state.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1246: `let mut state = self.state.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1282: `*self.state.read().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1326: `stats.overall_throughput = stats.total_items as f64 / max_uptime;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1334: `self.error_context.read().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1356: `if *self.state.read().unwrap() == StreamState::Running {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1400: `self.stage.processor.lock().unwrap().is_empty()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1423: `self.stage.processor.lock().unwrap().push_batch(input)`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1427: `let output = self.stage.processor.lock().unwrap().pop_batch(100)?;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1512: `chunk_results.into_iter().next().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1517: `chunk_results.into_iter().next().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory_efficient/work_stealing.rs

19 issues found:

- Line 107: `self.memory_used.load(Ordering::Relaxed) as f64 / self.memory_size as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 767: `let mut global_queue = self.global_queue.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 816: `self.stats.read().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 838: `let global_pending = self.global_queue.lock().unwrap().len();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 842: `.map(|w| w.local_queue.lock().unwrap().len())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 887: `stats_guard.throughput = stats_guard.tasks_completed as f64 / elapsed;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1027: `let mut scheduler = create_work_stealing_scheduler().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1028: `scheduler.start().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1031: `scheduler.submit(TaskPriority::Normal, || 42).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1040: `scheduler.stop().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1045: `let mut scheduler = create_work_stealing_scheduler().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1046: `scheduler.start().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1057: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1066: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1074: `scheduler.stop().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1079: `let mut scheduler = create_work_stealing_scheduler().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1080: `scheduler.start().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1086: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1095: `scheduler.stop().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory_efficient/zero_copy_interface.rs

42 issues found:

- Line 478: `let mut named = self.named_data.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 490: `let mut id_map = self.id_data.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 496: `let mut type_map = self.type_data.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 503: `let mut stats = self.stats.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 516: `let named = self.named_data.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 548: `let id_map = self.id_data.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 579: `let type_map = self.type_data.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 602: `let mut stats = self.stats.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 611: `self.named_data.read().unwrap().contains_key(name)`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 616: `self.id_data.read().unwrap().contains_key(&id)`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 621: `let mut named = self.named_data.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 627: `let mut id_map = self.id_data.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 632: `let mut stats = self.stats.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 647: `self.named_data.read().unwrap().keys().cloned().collect()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 652: `self.id_data.read().unwrap().keys().cloned().collect()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 657: `let named = self.named_data.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 671: `self.stats.read().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 676: `let mut named = self.named_data.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 677: `let mut id_map = self.id_data.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 678: `let mut type_map = self.type_data.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 685: `let mut stats = self.stats.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 691: `let mut stats = self.stats.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 789: `let zero_copy = ZeroCopyData::new(data.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 801: `let zero_copy1 = ZeroCopyData::new(data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 814: `let zero_copy = ZeroCopyData::new(data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 815: `let view = zero_copy.view(1, 3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 821: `let subview = view.subview(1, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 829: `let zero_copy = ZeroCopyData::new(data.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 832: `interface.register_data("test_data", zero_copy).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 836: `let retrieved = interface.get_data::<f64>("test_data").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 840: `let view = interface.borrow_data::<f64>("test_data").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 844: `let metadata = interface.get_metadata("test_data").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 853: `let zero_copy = ZeroCopyData::new(data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 855: `interface.register_data("int_data", zero_copy).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 865: `let zero_copy = ZeroCopyData::new(data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 871: `let upgraded = weak_ref.upgrade().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 885: `let zero_copy = ZeroCopyData::new(data.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 887: `register_global_data("global_test", zero_copy).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 889: `let retrieved = get_global_data::<f64>("global_test").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 896: `let zero_copy = data.clone().into_zero_copy().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 900: `let zero_copy2 = slice.into_zero_copy().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 907: `let zero_copy = ZeroCopyData::new(data.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory_efficient/zero_copy_streaming.rs

22 issues found:

- Line 337: `let item = node.data.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 593: `let mut stats = self.stats.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 601: `let mut stats = self.stats.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 620: `let stats = self.stats.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 624: `let mut stats = self.stats.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 634: `self.stats.read().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 684: `max_queue_size / num_workers.max(1),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 880: `let mut state = self.state.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 936: `let current_state = state.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 957: `let mut stats_guard = stats.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 964: `/ stats_guard.items_processed as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 997: `let mut stats = self.stats.read().unwrap().clone();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1003: `/ (buffer_stats.pool_hits + buffer_stats.pool_misses) as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1019: `let mut state = self.state.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1075: `let buffer = ZeroCopyBuffer::new(1024, None, 64).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1106: `let pool = BufferPool::new(4, 1024, false, 64).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1109: `let buffer1 = pool.get_buffer().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1116: `let buffer2 = pool.get_buffer().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1131: `let mut processor = ZeroCopyStreamProcessor::new(config, |x: i32| Ok(x * 2)).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1133: `processor.start().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1137: `processor.push(i).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1149: `processor.stop().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory_efficient/zero_serialization.rs

59 issues found:

- Line 1126: `(self.real * self.real + self.imag * self.imag).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1171: `let deserialized = Complex64::from_bytes(bytes).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1181: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1197: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1206: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1213: `let loaded_array = loaded.readonly_array::<ndarray::Ix2>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1225: `let loaded_metadata = MemoryMappedArray::<Complex64>::read_metadata(&file_path)....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1249: `let deserialized = i32::from_bytes(bytes).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1257: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1269: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1277: `MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1284: `let loaded_array = loaded.readonly_array::<ndarray::Ix1>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1292: `let loaded_metadata = MemoryMappedArray::<f64>::read_metadata(&file_path).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1300: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1307: `let array = MemoryMappedArray::<f32>::save_array(&data, &file_path, None).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1315: `MemoryMappedArray::<f32>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1322: `let loaded_array = loaded.readonly_array::<ndarray::Ix2>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1335: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1351: `MemoryMappedArray::<i32>::save_array(&data, &file_path, Some(metadata)).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1359: `MemoryMappedArray::<i32>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1366: `let loaded_array = loaded.readonly_array::<ndarray::Ix3>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1381: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1410: `MemoryMappedArray::<f64>::save_array(&data, &file_path, Some(metadata)).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1418: `MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1425: `let loaded_array = loaded.readonly_array::<IxDyn>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1443: `let data_slice = data_standard.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1455: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1470: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1475: `MemoryMappedArray::<u32>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1476: `let loaded_array = loaded.readonly_array::<ndarray::Ix1>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1483: `let loaded_metadata = MemoryMappedArray::<u32>::read_metadata(&file_path).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1500: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1505: `MemoryMappedArray::<i64>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1506: `let loaded_array = loaded.readonly_array::<ndarray::Ix2>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1515: `let loaded_metadata = MemoryMappedArray::<i64>::read_metadata(&file_path).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1533: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1538: `MemoryMappedArray::<f32>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1539: `let loaded_array = loaded.readonly_array::<ndarray::Ix3>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1550: `let loaded_metadata = MemoryMappedArray::<f32>::read_metadata(&file_path).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1558: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1569: `MemoryMappedArray::<f64>::save_array(&data, &file_path, Some(initial_metadata))....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1577: `MemoryMappedArray::<f64>::update_metadata(&file_path, updated_metadata.clone())....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1580: `let loaded_metadata = MemoryMappedArray::<f64>::read_metadata(&file_path).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1585: `MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1586: `let loaded_array = loaded.readonly_array::<ndarray::Ix1>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1597: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1604: `MemoryMappedArray::<f32>::save_array(&data, &file_path, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1608: `MemoryMappedArray::<f32>::open_zero_copy(&file_path, AccessMode::ReadWrite).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1612: `let mut array = mmap.as_array_mut::<ndarray::Ix2>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1617: `mmap.flush().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1621: `MemoryMappedArray::<f32>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1622: `let loaded_array = loaded.readonly_array::<ndarray::Ix2>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1639: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1646: `MemoryMappedArray::<f64>::save_array(&data, &file_path, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1650: `MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::CopyOnWrite).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1654: `let mut array_view = cow_mmap.as_array_mut::<ndarray::Ix2>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1663: `MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1664: `let original_array = original.readonly_array::<ndarray::Ix2>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1674: `let cow_array = cow_mmap.as_array::<ndarray::Ix2>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory_efficient/zerocopy.rs

58 issues found:

- Line 256: `let chunk_size = (self.size / rayon::current_num_threads()).max(1024);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 271: `let chunk = &array.as_slice().unwrap()[start..end];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 277: `let out_slice = &mut out_array.as_slice_mut().unwrap()[start..end];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 299: `let chunk = &array.as_slice().unwrap()[start..end];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 307: `let out_slice = &mut out_array.as_slice_mut().unwrap()[start..end];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 337: `let chunk = &array.as_slice().unwrap()[start..end];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 413: `let self_chunk = &self_array.as_slice().unwrap()[start..end];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 414: `let other_chunk = &other_array.as_slice().unwrap()[start..end];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 424: `let out_slice = &mut out_array.as_slice_mut().unwrap()[start..end];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 448: `let slice = &array.as_slice().unwrap()[start..end];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 478: `array.as_slice().unwrap()[0]`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 499: `array.as_slice().unwrap()[0]`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 552: `Ok(sum / count)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 784: `self.combine_zero_copy(other, |a, b| a / b)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 799: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 804: `MemoryMappedArray::<f64>::save_array(&data, &file_path, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 808: `MemoryMappedArray::<f64>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 811: `let result = mmap.map_zero_copy(|x| x * 2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 814: `let result_array = result.readonly_array::<ndarray::Ix1>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 823: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 828: `let mut file = File::create(&file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 830: `file.write_all(&val.to_ne_bytes()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 835: `let mmap = MemoryMappedArray::<f64>::open(&file_path, &[1000]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 838: `let sum = mmap.reduce_zero_copy(0.0, |acc, x| acc + x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 847: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 855: `MemoryMappedArray::<f64>::save_array(&data1, &file_path1, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 856: `MemoryMappedArray::<f64>::save_array(&data2, &file_path2, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 860: `MemoryMappedArray::<f64>::open_zero_copy(&file_path1, AccessMode::ReadOnly).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 862: `MemoryMappedArray::<f64>::open_zero_copy(&file_path2, AccessMode::ReadOnly).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 865: `let result = mmap1.combine_zero_copy(&mmap2, |a, b| a + b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 868: `let result_array = result.readonly_array::<ndarray::Ix1>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 877: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 882: `let mut file = File::create(&file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 884: `file.write_all(&val.to_ne_bytes()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 889: `let mmap = MemoryMappedArray::<f64>::open(&file_path, &[1000]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 892: `let even_numbers = mmap.filter_zero_copy(|&x| (x as usize) % 2 == 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 904: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 912: `MemoryMappedArray::<f64>::save_array(&data1, &file_path1, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 913: `MemoryMappedArray::<f64>::save_array(&data2, &file_path2, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 917: `MemoryMappedArray::<f64>::open_zero_copy(&file_path1, AccessMode::ReadOnly).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 919: `MemoryMappedArray::<f64>::open_zero_copy(&file_path2, AccessMode::ReadOnly).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 922: `let add_result = mmap1.add(&mmap2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 923: `let add_array = add_result.readonly_array::<ndarray::Ix1>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 929: `let sub_result = mmap1.sub(&mmap2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 930: `let sub_array = sub_result.readonly_array::<ndarray::Ix1>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 936: `let mul_result = mmap1.mul(&mmap2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 937: `let mul_array = mul_result.readonly_array::<ndarray::Ix1>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 944: `.div(&mmap1.map_zero_copy(|x| x + 1.0).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 945: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 946: `let div_array = div_result.readonly_array::<ndarray::Ix1>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 948: `assert_eq!(div_array[i], ((i + 5) as f64) / ((i + 1) as f64));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 955: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 964: `MemoryMappedArray::<f64>::save_array(&data1, &file_path1, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 965: `MemoryMappedArray::<f64>::save_array(&data2, &file_path2, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 969: `MemoryMappedArray::<f64>::open_zero_copy(&file_path1, AccessMode::ReadOnly).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 971: `MemoryMappedArray::<f64>::open_zero_copy(&file_path2, AccessMode::ReadOnly).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 974: `let result = mmap1.broadcast_op(&mmap2, |a, b| a * b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 977: `let result_array = result.readonly_array::<ndarray::Ix2>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/metrics.rs

7 issues found:

- Line 264: `let mean = if count > 0 { sum / count as f64 } else { 0.0 };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 846: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 848: `let metrics = registry.get_all_metrics().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 859: `monitor.register_check(memory_check).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 861: `let results = monitor.check_all().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 874: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 876: `let prometheus_output = registry.export_prometheus().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/ndarray_ext/broadcasting.rs

3 issues found:

- Line 293: `let (a_broad, b_broad) = broadcast_arrays(a.view(), b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 308: `let result = broadcast_apply(a.view(), b.view(), |x, y| x + y).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 318: `let result = broadcast_apply(a.view(), b.view(), |x, y| x * y).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/ndarray_ext/indexing.rs

14 issues found:

- Line 607: `let result = boolean_mask_1d(a.view(), mask.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 619: `let result = boolean_mask_2d(a.view(), mask.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 631: `let result = take_1d(a.view(), indices.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 644: `let result = take_2d(a.view(), indices.view(), 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 654: `let result = take_2d(a.view(), indices.view(), 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 670: `let result = fancy_index_2d(a.view(), row_indices.view(), col_indices.view()).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 681: `let main_diag = diagonal(a.view(), 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 688: `let upper_diag = diagonal(a.view(), 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 694: `let lower_diag = diagonal(a.view(), -1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 704: `let result = where_1d(a.view(), |&x| x > 3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 714: `let result = where_2d(a.view(), |&x| x > 3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 725: `let result = indices_where_1d(a.view(), |&x| x > 30).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 735: `let (rows, cols) = indices_where_2d(a.view(), |&x| x > 5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 751: `let result = take_along_axis(a.view(), indices.view(), 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/ndarray_ext/manipulation.rs

25 issues found:

- Line 1050: `grad_y[[0, j]] = (array[[1, j]] - array[[0, j]]) / dy;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1056: `grad_y[[i, j]] = (array[[i + 1, j]] - array[[i - 1, j]]) / (dy + dy);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1062: `grad_y[[rows - 1, j]] = (array[[rows - 1, j]] - array[[rows - 2, j]]) / dy;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1073: `grad_x[[i, 0]] = (array[[i, 1]] - array[[i, 0]]) / dx;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1077: `grad_x[[i, j]] = (array[[i, j + 1]] - array[[i, j - 1]]) / (dx + dx);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1081: `grad_x[[i, cols - 1]] = (array[[i, cols - 1]] - array[[i, cols - 2]]) / dx;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1189: `let swapped_rows = swap_axes_2d(a.view(), 0, 2, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1193: `let swapped_cols = swap_axes_2d(a.view(), 0, 2, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1197: `let swapped_same = swap_axes_2d(a.view(), 1, 1, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1234: `let vertical = concatenate_2d(&[a.view(), b.view()], 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1239: `let horizontal = concatenate_2d(&[a.view(), b.view()], 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1262: `let stacked = vstack_1d(&[a.view(), b.view(), c.view()]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1281: `let stacked = hstack_1d(&[a.view(), b.view()]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1300: `let squeezed_a = squeeze_2d(a.view(), 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1305: `let squeezed_b = squeeze_2d(b.view(), 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1323: `let (x_grid, y_grid) = meshgrid(x.view(), y.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1338: `let result = unique(a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1351: `let result = argmin(a.view(), Some(0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1355: `let result = argmin(a.view(), Some(1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1359: `let result = argmin(a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1375: `let result = argmax(a.view(), Some(0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1379: `let result = argmax(a.view(), Some(1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1383: `let result = argmax(a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1399: `let (grad_y, grad_x) = gradient(a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1423: `let (grad_y, grad_x) = gradient(a.view(), Some((2.0, 0.5))).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/ndarray_ext/matrix.rs

11 issues found:

- Line 741: `power = power.clone() * (T::one() / x[i].clone());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 807: `let main_diag = diagonal(a.view(), 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 811: `let super_diag = diagonal(a.view(), 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 815: `let sub_diag = diagonal(a.view(), -1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 869: `let result = toeplitz(first_row.view(), first_col.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 908: `let result = tridiagonal(diag.view(), lower.view(), upper.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 922: `let result = hankel(first_col.view(), last_row.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 934: `let tr = trace(a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 947: `let v1 = vander(x.view(), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 957: `let v2 = vander(x.view(), None, Some(true)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 967: `let v3 = vander(x.view(), Some(4), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/ndarray_ext/mod.rs

14 issues found:

- Line 70: `let r = i / cols;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 744: `let b = reshape_2d(a.view(), (4, 1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 762: `let c = stack_2d(&[a.view(), b.view()], 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 770: `let d = stack_2d(&[a.view(), b.view()], 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 794: `let result = split_2d(a.view(), &[2], 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 808: `let result = split_2d(a.view(), &[1], 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 820: `let result = take_2d(a.view(), indices.view(), 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 833: `let result = mask_select(a.view(), mask.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 846: `let result = fancy_index_2d(a.view(), row_indices.view(), col_indices.view()).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 855: `let result = where_condition(a.view(), |&x| x > 3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 867: `let b = broadcast_1d_to_2d(a.view(), 2, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 875: `let c = broadcast_1d_to_2d(a.view(), 2, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 888: `let result = broadcast_apply(a.view(), b.view(), |x, y| x + y).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 897: `let result = broadcast_apply(a.view(), b.view(), |x, y| x * y).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/ndarray_ext/ops.rs

6 issues found:

- Line 265: `let b = reshape(a.view(), (4, 1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 283: `let c = stack(&[a.view(), b.view()], Axis(0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 289: `let d = stack(&[a.view(), b.view()], Axis(1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 298: `let b = swapaxes(a.view(), 0, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 311: `let result = split(a.view(), &[2], Axis(1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 321: `let result = split(a.view(), &[1], Axis(0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/ndarray_ext/stats/correlation.rs

8 issues found:

- Line 46: `let n = T::from_usize(x.len()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 55: `let mean_x = sum_x / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 56: `let mean_y = sum_y / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 76: `Ok(cov_xy / (var_x * var_y).sqrt())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 76: `Ok(cov_xy / (var_x * var_y).sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 129: `feature_means[j] = sum / T::from_usize(n_samples).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 134: `let scale = T::from_usize(n_samples - ddof).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 146: `cov_ij = cov_ij / scale;`
  - **Fix**: Division without zero check - use safe_divide()

### src/ndarray_ext/stats/descriptive.rs

42 issues found:

- Line 65: `let n = T::from_usize(rows).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 72: `result[j] = sum / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 80: `let n = T::from_usize(cols).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 87: `result[i] = sum / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 104: `Ok(Array::from_elem(1, sum / count))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 175: `let mid = column_values.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 176: `(column_values[mid - 1] + column_values[mid]) / T::from_f64(2.0).unwrap()`
  - **Fix**: Use .get() with proper bounds checking
- Line 178: `column_values[column_values.len() / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 200: `let mid = row_values.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 201: `(row_values[mid - 1] + row_values[mid]) / T::from_f64(2.0).unwrap()`
  - **Fix**: Use .get() with proper bounds checking
- Line 203: `row_values[row_values.len() / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 219: `let mid = values.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 220: `(values[mid - 1] + values[mid]) / T::from_f64(2.0).unwrap()`
  - **Fix**: Use .get() with proper bounds checking
- Line 222: `values[values.len() / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 274: `Ok(var_result.mapv(|x| x.sqrt()))`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 346: `let divisor = T::from_usize(rows - ddof).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 347: `result[j] = sum_sq_diff / divisor;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 369: `let divisor = T::from_usize(cols - ddof).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 370: `result[i] = sum_sq_diff / divisor;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 395: `let divisor = T::from_usize(total_elements - ddof).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 397: `Ok(Array::from_elem(1, sum_sq_diff / divisor))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 764: `let pos = (q / 100.0) * (column_values.len() as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 774: `result[j] = column_values[idx_low] * T::from_f64(weight_low).unwrap()`
  - **Fix**: Use .get() with proper bounds checking
- Line 775: `+ column_values[idx_high] * T::from_f64(weight_high).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 795: `let pos = (q / 100.0) * (row_values.len() as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 805: `result[i] = row_values[idx_low] * T::from_f64(weight_low).unwrap()`
  - **Fix**: Use .get() with proper bounds checking
- Line 806: `+ row_values[idx_high] * T::from_f64(weight_high).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 821: `let pos = (q / 100.0) * (values.len() as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 831: `values[idx_low] * T::from_f64(weight_low).unwrap()`
  - **Fix**: Use .get() with proper bounds checking
- Line 832: `+ values[idx_high] * T::from_f64(weight_high).unwrap()`
  - **Fix**: Use .get() with proper bounds checking
- Line 888: `Ok(Array::from_elem(1, sum / count))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 935: `let mid = values.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 936: `(values[mid - 1] + values[mid]) / T::from_f64(2.0).unwrap()`
  - **Fix**: Use .get() with proper bounds checking
- Line 938: `values[values.len() / 2]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1002: `let divisor = T::from_usize(total_elements - ddof).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1004: `Ok(Array::from_elem(1, sum_sq_diff / divisor))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1038: `Ok(var_result.mapv(|x| x.sqrt()))`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1072: `let mut min_val = *array.iter().next().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1116: `let mut max_val = *array.iter().next().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1177: `let pos = (q / 100.0) * (values.len() as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1187: `values[idx_low] * T::from_f64(weight_low).unwrap()`
  - **Fix**: Use .get() with proper bounds checking
- Line 1188: `+ values[idx_high] * T::from_f64(weight_high).unwrap()`
  - **Fix**: Use .get() with proper bounds checking

### src/ndarray_ext/stats/distribution.rs

19 issues found:

- Line 83: `let bin_width = (max_val - min_val) / T::from_usize(bins).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 86: `bin_edges[i] = min_val + bin_width * T::from_usize(i).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 115: `let scaled_val = (val - min_val) / bin_width;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 138: `let scaled_val = (val - min_val) / bin_width;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 239: `let x_bin_width = (x_max - x_min) / T::from_usize(x_bins).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 240: `let y_bin_width = (y_max - y_min) / T::from_usize(y_bins).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 243: `x_edges[i] = x_min + x_bin_width * T::from_usize(i).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 247: `y_edges[i] = y_min + y_bin_width * T::from_usize(i).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 271: `let x_scaled = (x_val - x_min) / x_bin_width;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 272: `let y_scaled = (y_val - y_min) / y_bin_width;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 303: `let x_scaled = (x_val - x_min) / x_bin_width;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 304: `let y_scaled = (y_val - y_min) / y_bin_width;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 377: `if val < T::from_f64(0.0).unwrap() || val > T::from_f64(1.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 393: `if q_val == T::from_f64(0.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 398: `if q_val == T::from_f64(1.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 404: `let h = T::from_usize(n - 1).unwrap() * q_val;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 412: `result[i] = sorted[idx_low] * (T::from_f64(1.0).unwrap() - weight) + sorted[idx_...`
  - **Fix**: Use .get() with proper bounds checking
- Line 421: `result[i] = (sorted[idx_low] + sorted[idx_high]) / T::from_f64(2.0).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 425: `if weight < T::from_f64(0.5).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()

### src/ndarray_ext/views.rs

2 issues found:

- Line 217: `let view = strided_view(a.view(), &[2, 2]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 248: `let b = broadcast_to(a.view(), shape).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/numeric.rs

25 issues found:

- Line 242: `self.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 281: `self.ln()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 333: `self.powf(n)`
  - **Fix**: Mathematical operation .powf( without validation
- Line 348: `return SQRT_TWO_PI * self.powf(self + 0.5) * (-self).exp();`
  - **Fix**: Mathematical operation .powf( without validation
- Line 368: `self.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 403: `self.ln()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 455: `self.powf(n)`
  - **Fix**: Mathematical operation .powf( without validation
- Line 470: `return SQRT_TWO_PI * self.powf(self + 0.5) * (-self).exp();`
  - **Fix**: Mathematical operation .powf( without validation
- Line 490: `(self as f64).sqrt() as i32`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 547: `(self / gcd) * other`
  - **Fix**: Division without zero check - use safe_divide()
- Line 622: `result = result * (self - i) / (i + 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 672: `let pi = T::from_f64(std::f64::consts::PI).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 673: `self * pi / T::from_f64(180.0).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 677: `let pi = T::from_f64(std::f64::consts::PI).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 678: `self * T::from_f64(180.0).unwrap() / pi`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 732: `Scalar(self.0 / other.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 770: `assert_eq!(a.sqrt(), (3.0_f32).sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 785: `assert_eq!(a.sqrt(), (3.0_f64).sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 800: `assert_eq!(a.ln(), (3.0_f32).ln());`
  - **Fix**: Mathematical operation .ln() without validation
- Line 803: `assert_eq!(a.powf(2.0), 9.0);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 812: `assert_eq!(a.ln(), (3.0_f64).ln());`
  - **Fix**: Mathematical operation .ln() without validation
- Line 815: `assert_eq!(a.powf(2.0), 9.0);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 837: `let b: i32 = a.try_convert().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 855: `let radians: f64 = std::f64::consts::PI / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 868: `assert_eq!((a / b).value(), 0.75);`
  - **Fix**: Division without zero check - use safe_divide()

### src/numeric/arbitrary_precision.rs

27 issues found:

- Line 38: `*DEFAULT_PRECISION.read().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 44: `*DEFAULT_PRECISION.write().unwrap() = prec;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 172: `BigInt::from_str(&self.value.to_string()).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 218: `value: product / gcd.value,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 234: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 301: `value: self.value / rhs.value,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 385: `(self.value.prec() as f64 / 3.32) as u32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 641: `value: self.value / rhs.value,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 805: `value: self.value / rhs.value,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 916: `let ln_self = self.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 998: `value: self.value / rhs.value,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1110: `self.context.float_precision as f64 / 3.32,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1163: `two.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1170: `let sqrt5 = five.sqrt()?;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1172: `Ok((one + sqrt5) / two)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1211: `let a = ArbitraryFloat::from_f64_prec(1.0, 128).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1212: `let b = ArbitraryFloat::from_f64_prec(3.0, 128).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1213: `let c = a / b;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1223: `let r = ArbitraryRational::from_ratio(22, 7).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1226: `let a = ArbitraryRational::from_ratio(1, 3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1227: `let b = ArbitraryRational::from_ratio(1, 6).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1255: `let pi = utils::pi(256).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1259: `let e = utils::e(256).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1287: `let x = ArbitraryFloat::from_f64_prec(0.5, 128).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1296: `let ln_x = x.ln().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1309: `assert!(neg.sqrt().is_err());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1312: `assert!(neg.ln().is_err());`
  - **Fix**: Mathematical operation .ln() without validation

### src/numeric/precision_tracking.rs

15 issues found:

- Line 214: `severity: if self.precision < min_acceptable / 2.0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 429: `(self.value - other.value).abs() / self.value.abs().max(other.value.abs());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 469: `let result_value = self.value / other.value;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 500: `let result_value = self.value.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 519: `let result_value = self.value.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 546: `(f64::EPSILON * (a.abs() + b.abs())) / result.abs()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 557: `(rel_error_a + rel_error_b) * (a / b).abs()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 770: `let result = a.div(&b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 781: `let warning = warning.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 791: `registry.register_computation("test", context).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 793: `let retrieved = registry.get_computation_context("test").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 795: `assert_eq!(retrieved.unwrap().precision, 10.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 801: `let result = a.sqrt().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 809: `let result = a.ln().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 843: `assert_eq!(context.condition_number.unwrap(), 1e15);`
  - **Fix**: Replace with ? operator or .ok_or()

### src/numeric/scientific_types.rs

26 issues found:

- Line 90: `Self::new(self.value / rhs)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 323: `radians: degrees * std::f64::consts::PI / 180.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 334: `self.radians * 180.0 / std::f64::consts::PI`
  - **Fix**: Division without zero check - use safe_divide()
- Line 406: `Self::from_radians(self.radians / rhs)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 441: `(self.re * self.re + self.im * self.im).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 466: `re: self.magnitude().ln(),`
  - **Fix**: Mathematical operation .ln() without validation
- Line 473: `(self.ln() * exp).exp()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 478: `let magnitude = self.magnitude().powf(exp);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 485: `self.powf(0.5)`
  - **Fix**: Mathematical operation .powf( without validation
- Line 538: `re: (self.re * rhs.re + self.im * rhs.im) / denominator,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 539: `im: (self.im * rhs.re - self.re * rhs.im) / denominator,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 636: `x: self.x / mag,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 637: `y: self.y / mag,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 638: `z: self.z / mag,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 655: `RealNumber::acos(dot / mag_product)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 712: `x: self.x / rhs,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 713: `y: self.y / rhs,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 714: `z: self.z / rhs,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 750: `pub const HBAR: f64 = PLANCK / (2.0 * std::f64::consts::PI);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 872: `let divided = length1.clone() / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 879: `let angle2 = Angle64::from_radians(std::f64::consts::PI / 4.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 881: `assert!((angle1.radians() - std::f64::consts::PI / 2.0).abs() < 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 900: `assert!((z1.phase() - (4.0_f64 / 3.0).atan()).abs() < 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 914: `let polar = Complex64::from_polar(5.0, std::f64::consts::PI / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 938: `assert!((magnitude - (14.0_f64).sqrt()).abs() < 1e-10);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 943: `assert!((angle - std::f64::consts::PI / 2.0).abs() < 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()

### src/numeric/stability.rs

53 issues found:

- Line 160: `let mid = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 178: `Ok(neumaier_sum(values) / n)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 204: `self.mean = self.mean + delta / cast::<usize, T>(self.count).unwrap_or(T::one())...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 221: `Some(self.m2 / cast::<usize, T>(self.count - 1).unwrap_or(T::one()))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 230: `Some(self.m2 / cast::<usize, T>(self.count).unwrap_or(T::one()))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 238: `self.variance().map(|v| v.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 277: `Ok(sum_sq / divisor)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 301: `max_val + sum.ln()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 327: `*exp_val = *exp_val / sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 341: `x - x2 / cast::<f64, T>(2.0).unwrap_or(T::one())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 342: `+ x3 / cast::<f64, T>(3.0).unwrap_or(T::one())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 343: `- x4 / cast::<f64, T>(4.0).unwrap_or(T::one())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 345: `(T::one() + x).ln()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 356: `x + x2 / cast::<f64, T>(2.0).unwrap_or(T::one())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 357: `+ x3 / cast::<f64, T>(6.0).unwrap_or(T::one())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 358: `+ x4 / cast::<f64, T>(24.0).unwrap_or(T::one())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 373: `let ratio = y_abs / x_abs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 374: `x_abs * (T::one() + ratio * ratio).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 379: `let ratio = x_abs / y_abs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 380: `y_abs * (T::one() + ratio * ratio).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 410: `check_positive(m.to_f64().unwrap(), "modulus")?;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 421: `if a_mod.abs() > max_val / b_mod.abs() {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 451: `T::one() / (T::one() + exp_neg_x)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 454: `exp_x / (T::one() + exp_x)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 482: `- (*target * pred_clipped.ln() + (T::one() - *target) * (T::one() - pred_clipped...`
  - **Fix**: Mathematical operation .ln() without validation
- Line 485: `Ok(loss / cast::<usize, T>(predictions.len()).unwrap_or(T::one()))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 506: `Ok(sum.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 576: `let scaled = value / max_abs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 580: `max_abs * sum.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 599: `let mut x = Array1::from_elem(n, T::one() / cast::<usize, T>(n).unwrap_or(T::one...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 619: `y[i] = y[i] / y_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 660: `log_result += ((n - i) as f64).ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 661: `log_result -= ((i + 1) as f64).ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 689: `n_f64 * n_f64.ln() - n_f64 + 0.5 * (2.0 * std::f64::consts::PI * n_f64).ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 771: `assert_relative_eq!(welford.mean().unwrap(), 3.0, epsilon = 1e-10);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 772: `assert_relative_eq!(welford.variance().unwrap(), 2.5, epsilon = 1e-10);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 773: `assert_relative_eq!(welford.std_dev().unwrap(), 2.5_f64.sqrt(), epsilon = 1e-10)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 780: `let expected = 1000.0 + 3.0_f64.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 794: `assert_relative_eq!(p, 1.0 / 3.0, epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 807: `let expected = 2.0_f64.sqrt() * 1e200;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 843: `assert_relative_eq!(reduce_angle(PI / 2.0), PI / 2.0, epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 851: `let expected = 3.0_f64.sqrt() * 1e200;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 857: `let expected_small = 3.0_f64.sqrt() * 1e-200;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 864: `assert_eq!(binomial_stable(5, 2).unwrap(), 10.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 865: `assert_eq!(binomial_stable(10, 3).unwrap(), 120.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 868: `assert_eq!(binomial_stable(5, 0).unwrap(), 1.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 869: `assert_eq!(binomial_stable(5, 5).unwrap(), 1.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 870: `assert_eq!(binomial_stable(5, 6).unwrap(), 0.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 873: `let large_result = binomial_stable(100, 50).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 880: `assert_eq!(factorial_stable(0).unwrap(), 1.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 881: `assert_eq!(factorial_stable(1).unwrap(), 1.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 882: `assert_eq!(factorial_stable(5).unwrap(), 120.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 885: `assert_relative_eq!(factorial_stable(10).unwrap(), 3628800.0, epsilon = 1e-10);`
  - **Fix**: Replace with ? operator or .ok_or()

### src/numeric/stable_algorithms.rs

37 issues found:

- Line 114: `let factor = aug[[i, k]] / aug[[k, k]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 137: `x[i] = sum / aug[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 179: `x[i] = x[i] / norm_v;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 242: `l[[i, i]] = sum.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 251: `l[[j, i]] = sum / l[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 299: `let _initial_residual = r_norm_sq.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 326: `residual: r_norm_sq.sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 332: `let alpha = r_norm_sq / p_ap;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 365: `let beta = r_norm_sq_new / r_norm_sq;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 377: `residual: r_norm_sq.sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 460: `v[0][i] = r[i] / r_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 499: `v[j + 1][k] = w[k] / w_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 519: `y[i] = sum / h[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 549: `y[i] = sum / h[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 606: `d[[i, 0]] = (f_plus - f_minus) / (cast::<f64, T>(2.0).unwrap_or(T::one()) * h_cu...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 609: `h_curr = h_curr / cast::<f64, T>(2.0).unwrap_or(T::one());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 616: `d[[i, j]] = (factor * d[[i, j - 1]] - d[[i - 1, j - 1]]) / (factor - T::one());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 629: `check_finite(a.to_f64().unwrap(), "Lower limit")?;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 630: `check_finite(b.to_f64().unwrap(), "Upper limit")?;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 639: `let h = (b - a) / cast::<f64, T>(6.0).unwrap_or(T::one());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 640: `let mid = (a + b) / cast::<f64, T>(2.0).unwrap_or(T::one());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 657: `let mid = (a + b) / cast::<f64, T>(2.0).unwrap_or(T::one());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 665: `combined + diff / cast::<f64, T>(15.0).unwrap_or(T::one())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 667: `let half_tol = tolerance / cast::<f64, T>(2.0).unwrap_or(T::one());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 708: `scaled_norm = scaled_norm / cast::<f64, T>(2.0).unwrap_or(T::one());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 717: `a_scaled[[i, j]] = a[[i, j]] / scale;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 743: `result[[i, j]] = result[[i, j]] + term[[i, j]] / factorial;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 775: `let x = gaussian_elimination_stable(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 787: `let (q, r) = qr_decomposition_stable(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 816: `let l = cholesky_stable(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 834: `let result = conjugate_gradient(&a.view(), &b.view(), None, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 851: `let derivative = richardson_derivative(f, 2.0, 0.1, 3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 858: `let derivative = richardson_derivative(g, 0.0, 0.01, 4).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 868: `let integral = adaptive_simpson(f, 0.0, 1.0, 1e-10, 10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 869: `assert_relative_eq!(integral, 1.0 / 3.0, epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 875: `let integral = adaptive_simpson(g, 0.0, std::f64::consts::PI, 1e-10, 10).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 884: `let exp_a = matrix_exp_stable(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/observability/tracing.rs

3 issues found:

- Line 1107: `let current_rate = total as f64 / elapsed;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1108: `let adjustment_factor = self.target_rate_per_second / current_rate;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1122: `sampled as f64 / total as f64`
  - **Fix**: Division without zero check - use safe_divide()

### src/parallel/nested.rs

10 issues found:

- Line 57: `threads_per_level: vec![num_cpus, num_cpus / 2, 1],`
  - **Fix**: Division without zero check - use safe_divide()
- Line 259: `cpu_usage: *self.cpu_usage.read().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 260: `active_contexts_per_level: self.active_contexts.read().unwrap().clone(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 509: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 532: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 535: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 536: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 537: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 555: `assert_eq!(result.unwrap(), 42);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 572: `assert_eq!(result.unwrap(), 45);`
  - **Fix**: Replace with ? operator or .ok_or()

### src/parallel/partitioning.rs

30 issues found:

- Line 157: `let mean = values.iter().copied().sum::<f64>() / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 167: `/ n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 168: `let std_dev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 175: `let z = (x - mean) / std_dev;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 179: `sum_cubed / n`
  - **Fix**: Division without zero check - use safe_divide()
- Line 189: `let z = (x - mean) / std_dev;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 193: `sum_fourth / n - 3.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 265: `let initial_sizes = vec![data_size / num_partitions; num_partitions];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 323: `let chunk_size = (data.len() + self.config.num_partitions - 1) / self.config.num...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 354: `let size = ((weight / total_weight) * data.len() as f64) as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 408: `let chunk_size = (data.len() + num_leaves - 1) / num_leaves;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 472: `let base = 1.0 + skewness.abs() / 10.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 477: `base.powf((num_partitions - i - 1) as f64)`
  - **Fix**: Mathematical operation .powf( without validation
- Line 480: `base.powf(i as f64)`
  - **Fix**: Mathematical operation .powf( without validation
- Line 500: `let quantile = i as f64 / num_partitions as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 507: `-((1.0 - 2.0 * quantile).ln() * 2.0).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 507: `-((1.0 - 2.0 * quantile).ln() * 2.0).sqrt()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 510: `((2.0 * quantile - 1.0).ln() * 2.0).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 510: `((2.0 * quantile - 1.0).ln() * 2.0).sqrt()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 529: `let weight = ((i + 1) as f64).powf(-alpha);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 553: `let boundary = mean1 - range1 * 0.5 + (range1 / partitions_mode1 as f64) * i as ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 558: `boundaries.push((mean1 + mean2) / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 562: `let boundary = mean2 - range1 * 0.5 + (range1 / partitions_mode2 as f64) * i as ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 613: `let avg = sum.as_secs_f64() / times.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 620: `let mean_time = total_avg / avg_times.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 627: `} else if avg_time < mean_time / self.target_imbalance {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 647: `times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / times.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 654: `max_time / min_time`
  - **Fix**: Division without zero check - use safe_divide()
- Line 704: `let partitions = partitioner.partition_equal_size(&data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 720: `let partitions = partitioner.partition_weighted(&data, &weights).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/parallel/scheduler.rs

43 issues found:

- Line 266: `*self.status.lock().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 272: `let completed = lock.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 276: `let _unused = cvar.wait(completed).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 285: `let completed = lock.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 289: `let result = cvar.wait_timeout(completed, timeout).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 304: `let mut status = self.status.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 311: `let mut completed = lock.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 384: `let mut status = self.status.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 393: `let mut status = self.status.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 402: `let mut completed = lock.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 560: `let mut last_active = self.last_active.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 566: `let last_active = self.last_active.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 587: `tasks_stolen as f64 / steal_attempts as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 599: `(current_batch_size / 2).max(config.min_batch_size)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 781: `while let SchedulerState::Running = *state.read().unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 798: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 817: `let mut status = task.status.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 855: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 871: `if *self.state.read().unwrap() != SchedulerState::Running {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 887: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 893: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1007: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1014: `if let Some(notify) = self.task_completion.lock().unwrap().get(&id) {`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 1016: `let completed = lock.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1033: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1049: `if let Some(notify) = self.task_completion.lock().unwrap().get(&id) {`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 1051: `let completed = lock.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1054: `let result = cvar.wait_timeout(completed, remaining).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1082: `let submissions = self.task_submissions.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1083: `let executions = self.task_executions.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1100: `stats.avg_task_latency_ms = total_latency.as_millis() as f64 / completed_tasks a...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1102: `total_execution.as_millis() as f64 / completed_tasks as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1119: `tasks_processed as f64 / stats.tasks_submitted as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1127: `stats.avg_queue_size = total_queue_size as f64 / self.worker_states.len() as f64...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1136: `stats.tasks_per_second = stats.tasks_completed as f64 / stats.uptime_seconds;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1146: `let mut state = self.state.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1162: `let mut state = self.state.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1192: `if *self.state.read().unwrap() != SchedulerState::ShutDown {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1369: `results_clone.lock().unwrap().push((i, result));`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1399: `let results_guard = results.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1483: `let chunk_size = std::cmp::max(1, items_owned.len() / num_chunks);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1542: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1543: `let flat = flat_view.to_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/performance_optimization.rs

7 issues found:

- Line 113: `let ops_per_ns = size as f64 / duration_ns as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 118: `.fetch_add(size / 10, Ordering::Relaxed);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 121: `self.simd_threshold.fetch_add(size / 10, Ordering::Relaxed);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 130: `let elements_per_cache_line = self.cache_line_size / element_size.max(1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 184: `let chunks = len / 8;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 288: `strides.push(stride / std::mem::size_of::<T>() as isize);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 347: `fast_paths::add_f64_arrays(&a, &b, &mut result).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/profiling.rs

87 issues found:

- Line 318: `self.total_duration / self.calls as u32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 357: `self.total_delta as f64 / self.allocations as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 530: `entry.total_delta as f64 / 1024.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 531: `entry.max_delta as f64 / 1024.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 543: `writeln!(report, "No profiling data collected.").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 548: `writeln!(report, "\n=== Timing Report ===").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 554: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 555: `writeln!(report, "{}", "-".repeat(90)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 571: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 576: `writeln!(report, "\n=== Memory Report ===").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 582: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 583: `writeln!(report, "{}", "-".repeat(75)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 595: `entry.total_delta as f64 / 1024.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 596: `entry.max_delta as f64 / 1024.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 598: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 984: `/ self.config.min_execution_threshold.as_secs_f64())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1014: `let severity = (stats.calls as f64 / 10000.0).min(1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1045: `memory_entry.total_delta as f64 / memory_entry.allocations as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1049: `/ (self.config.memory_threshold as f64 * 2.0))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1069: `memory_entry.max_delta as f64 / 1024.0 / 1024.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1070: `self.config.memory_threshold as f64 / 1024.0 / 1024.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1128: `report.stats.avg_memory / 1024.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1132: `report.stats.max_memory as f64 / 1024.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1263: `let cpu_hist = self.cpu_history.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1264: `let memory_hist = self.memory_history.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1265: `let network_hist = self.network_history.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1270: `cpu_hist.iter().sum::<f64>() / cpu_hist.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1278: `memory_hist.iter().sum::<usize>() / memory_hist.len()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1479: `diff.baseline_avg / 1024.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1480: `diff.current_avg / 1024.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1517: `/ baseline_avg.as_nanos() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1546: `baseline.total_delta as f64 / baseline.allocations as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1552: `current.total_delta as f64 / current.allocations as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1558: `((current_avg - baseline_avg) / baseline_avg.abs()) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1595: `/ timing_diffs.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1605: `/ memory_diffs.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1737: `Profiler::global().lock().unwrap().start();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1745: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1749: `let (calls, total, avg, max) = stats.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1773: `Profiler::global().lock().unwrap().start();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1787: `let reports = detector.analyze(&Profiler::global().lock().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1799: `Profiler::global().lock().unwrap().start();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1808: `&Profiler::global().lock().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1818: `&Profiler::global().lock().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1825: `let report = report.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1940: `self.app_profiler.lock().unwrap().start();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1951: `self.app_profiler.lock().unwrap().stop();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1984: `let app_report = self.app_profiler.lock().unwrap().get_report();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1992: `bottleneck_reports = detector.analyze(&self.app_profiler.lock().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2119: `writeln!(report, "=== {} ===", self.session_name).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2125: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2126: `writeln!(report, "Generated At: {:?}", self.generated_at).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2127: `writeln!(report).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2130: `writeln!(report, "=== Application Performance ===").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2131: `writeln!(report, "{}", self.application_report).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2135: `writeln!(report, "=== System Resource Summary ===").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2137: `/ self.system_metrics.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2143: `/ self.system_metrics.len();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2156: `writeln!(report, "Average CPU Usage: {:.1}%", avg_cpu).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2157: `writeln!(report, "Maximum CPU Usage: {:.1}%", max_cpu).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2161: `avg_memory as f64 / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2163: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2167: `max_memory as f64 / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2169: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2170: `writeln!(report).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2175: `writeln!(report, "=== System Alerts ({}) ===", self.alerts.len()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2177: `writeln!(report, "[{:?}] {}", alert.severity, alert.message).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2179: `writeln!(report).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2189: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2191: `writeln!(report, "Operation: {}", bottleneck.operation).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2192: `writeln!(report, "Type: {:?}", bottleneck.bottleneck_type).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2193: `writeln!(report, "Severity: {:.2}", bottleneck.severity).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2194: `writeln!(report, "Description: {}", bottleneck.description).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2196: `writeln!(report, "Suggestions:").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2198: `writeln!(report, "  - {}", suggestion).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2201: `writeln!(report).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2214: `writeln!(json, "{{").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2215: `writeln!(json, "  \"session_name\": \"{}\",", self.session_name).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2221: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2222: `writeln!(json, "  \"alert_count\": {},", self.alerts.len()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2228: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2234: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2238: `/ self.system_metrics.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2244: `writeln!(json, "  \"average_cpu_usage\": {},", avg_cpu).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2245: `writeln!(json, "  \"maximum_cpu_usage\": {}", max_cpu).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2248: `writeln!(json, "}}").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2279: `profiler.start().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/profiling/adaptive.rs

8 issues found:

- Line 429: `MemoryPattern::Sequential => Some((self.data_size / 1000).max(1024)),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1018: `recent_values[5..].iter().sum::<f64>() / (recent_values.len() - 5) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1019: `let avg_new = recent_values[..5].iter().sum::<f64>() / 5.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1021: `let change_percent = (avg_new - avg_old) / avg_old * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1170: `let optimizer = optimizer.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1221: `let mut optimizer = AdaptiveOptimizer::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1237: `let mut optimizer = AdaptiveOptimizer::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1246: `let optimizer = AdaptiveOptimizer::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/profiling/continuous_monitoring.rs

33 issues found:

- Line 392: `let mut running = self.running.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 421: `let mut running = self.running.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 427: `let mut monitor = system_monitor.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 476: `while *running.lock().unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 488: `let mut history = metrics_history.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 540: `monitor.lock().unwrap().get_current_metrics().ok()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 549: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 605: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 614: `(sys_metrics.memory_usage as f64 / sys_metrics.memory_total as f64) * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 646: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 678: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 705: `let history = metrics_history.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 731: `Some((m.memory_usage as f64 / m.memory_total as f64) * 100.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 748: `*trend_analysis.write().unwrap() = analysis_results;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 776: `let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 777: `let intercept = (sum_y - slope * sum_x) / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 793: `let last_value = values.last().unwrap().1;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 813: `let _history = metrics_history.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 814: `let trends = trend_analysis.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 863: `let mut recs = recommendations.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 877: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 885: `self.trend_analysis.read().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 890: `self.recommendations.read().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 895: `let history = self.metrics_history.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 972: `(sys_metrics.memory_usage as f64 / sys_metrics.memory_total as f64) * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1027: `let mut metrics = self.metrics.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1040: `self.metrics.lock().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1046: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1075: `let mut monitor = monitor.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1082: `let mut monitor = monitor.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1089: `let monitor = monitor.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1101: `let monitor = monitor.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1220: `assert!(!*monitor.running.lock().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()

### src/profiling/coverage.rs

32 issues found:

- Line 274: `(self.covered_lines as f64 / self.total_lines as f64) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 284: `(covered_branches as f64 / self.branches.len() as f64) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 298: `(covered_functions as f64 / self.functions.len() as f64) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 407: `let execution_factor = (self.execution_count as f64).ln().min(5.0) / 5.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 407: `let execution_factor = (self.execution_count as f64).ln().min(5.0) / 5.0;`
  - **Fix**: Mathematical operation .ln() without validation
- Line 408: `let complexity_factor = 1.0 / (1.0 + self.complexity as f64 / 10.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 833: `file_coverage: self.file_coverage.read().unwrap().clone(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 992: `let coverage = self.file_coverage.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1035: `(stats.covered_lines as f64 / stats.total_lines as f64) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1041: `(stats.covered_branches as f64 / stats.total_branches as f64) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1047: `(stats.covered_functions as f64 / stats.total_functions as f64) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1053: `(stats.covered_integrations as f64 / stats.total_integrations as f64) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1145: `let coverage = self.file_coverage.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1207: `let history = self.history.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1216: `let first = recent_points.last().unwrap().coverage_percentage;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1217: `let last = recent_points.first().unwrap().coverage_percentage;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1233: `let first = recent_points.last().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1234: `let last = recent_points.first().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1246: `coverage_diff / time_diff`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1256: `let last_coverage = recent_points.first().unwrap().coverage_percentage;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1563: `report.overall_stats.line_coverage_percentage / 100.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1566: `report.overall_stats.branch_coverage_percentage / 100.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1584: `cov.line_coverage_percentage() / 100.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1585: `cov.branch_coverage_percentage() / 100.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1806: `let coverage = self.file_coverage.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1982: `let analyzer = CoverageAnalyzer::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2012: `let analyzer = CoverageAnalyzer::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2030: `let recommendations = analyzer.generate_recommendations(&stats).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2048: `let analyzer = CoverageAnalyzer::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2051: `let mut history = analyzer.history.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2080: `let trends = analyzer.calculate_trends().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2083: `let trends = trends.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/profiling/dashboards.rs

13 issues found:

- Line 514: `Some(values.iter().sum::<f64>() / values.len() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 722: `let metrics = self.metrics.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 723: `let alerts = self.alerts.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 985: `let dashboard = dashboard.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 993: `let mut dashboard = PerformanceDashboard::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1001: `let widget_id = dashboard.add_widget(widget).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1009: `let mut dashboard = PerformanceDashboard::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1014: `dashboard.add_widget(widget).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1021: `let metrics = dashboard.get_metrics().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1025: `let cpu_metric = cpu_metric.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1053: `let alert_config = widget.alert_config.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1062: `let mut dashboard = PerformanceDashboard::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1067: `dashboard.add_widget(widget).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/profiling/flame_graph_svg.rs

20 issues found:

- Line 85: `let r = (255.0 * heat.sqrt()) as u8;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 128: `let h = h / 360.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 129: `let s = s / 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 130: `let l = l / 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 134: `let m = l - c / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 136: `let (r, g, b) = if h < 1.0 / 6.0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 138: `} else if h < 2.0 / 6.0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 140: `} else if h < 3.0 / 6.0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 142: `} else if h < 4.0 / 6.0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 144: `} else if h < 5.0 / 6.0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 206: `self.config.width / 2,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 216: `self.config.width / 2,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 268: `node_time / total_time`
  - **Fix**: Division without zero check - use safe_divide()
- Line 277: `format!("{:.2}%", (node_time / total_time) * 100.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 306: `(child_time / node_time) * width`
  - **Fix**: Division without zero check - use safe_divide()
- Line 340: `let max_chars = ((width - 8.0) / char_width) as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 558: `let chart_x = x + (time.as_secs_f64() / max_time) * width;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 559: `let chart_y = y + height - (cpu / max_cpu) * height;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 603: `let chart_x = x + (time.as_secs_f64() / max_time) * width;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 604: `let chart_y = y + height - ((*memory as f64) / (max_memory as f64)) * height;`
  - **Fix**: Division without zero check - use safe_divide()

### src/profiling/hardware_counters.rs

24 issues found:

- Line 304: `let mut counters = self.active_counters.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 314: `let mut counters = self.active_counters.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 325: `let counters = self.active_counters.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 352: `let counters = self.active_counters.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 362: `let counters = self.active_counters.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 424: `let mut counters = self.active_counters.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 433: `let mut counters = self.active_counters.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 442: `let counters = self.active_counters.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 526: `let mut counters = self.active_counters.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 535: `let mut counters = self.active_counters.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 544: `let counters = self.active_counters.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 636: `let mut sessions = self.session_counters.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 644: `let mut sessions = self.session_counters.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 662: `let sessions = self.session_counters.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 674: `let mut history = self.counter_history.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 697: `let history = self.counter_history.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 714: `metrics.instructions_per_cycle = instructions.value as f64 / cycles.value as f64...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 724: `metrics.cache_hit_rate = 1.0 - (misses.value as f64 / references.value as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 735: `1.0 - (misses.value as f64 / instructions.value as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 750: `let sessions = self.session_counters.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 921: `let manager = manager.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 936: `let manager = manager.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 953: `let manager = manager.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 960: `let manager = manager.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/profiling/performance_hints.rs

9 issues found:

- Line 319: `self.average_duration = self.total_duration / self.total_calls as u32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 606: `let retrieved = registry.get_hints("test_function").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 608: `assert_eq!(retrieved.unwrap().complexity, ComplexityClass::Linear);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 616: `let stats = registry.get_stats("test_function").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 618: `assert_eq!(stats.unwrap().total_calls, 1);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 632: `registry.register("test_function", hints).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 636: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 649: `let stats = global_registry().get_stats("test_tracker").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 651: `let stats = stats.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/profiling/production.rs

16 issues found:

- Line 323: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 336: `regressions.sort_by(|a, b| b.significance.partial_cmp(&a.significance).unwrap())...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 481: `self.cpu_samples.iter().sum::<f64>() / self.cpu_samples.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 487: `self.memory_samples.iter().sum::<usize>() / self.memory_samples.len()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 493: `self.thread_samples.iter().sum::<usize>() / self.thread_samples.len()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 633: `let _session = session.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 747: `historical_times.iter().sum::<Duration>() / historical_times.len() as u32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 752: `/ baseline.as_millis() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 784: `std_deviation.as_millis() as f64 / mean_time.as_millis() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 904: `bottlenecks.iter().map(|b| b.confidence).sum::<f64>() / bottlenecks.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 984: `let mut profiler = ProductionProfiler::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 996: `let report = report.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1004: `let profiler = ProductionProfiler::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1006: `let bottlenecks = profiler.identify_bottlenecks("test_workload").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1031: `let profiler = ProductionProfiler::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1033: `let stats = profiler.calculate_statistics("test_workload").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/profiling/system_monitor.rs

20 issues found:

- Line 131: `let mut running = self.running.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 168: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 176: `let history = self.metrics_history.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 184: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 202: `let avg_cpu = metrics.iter().map(|m| m.cpu_usage).sum::<f64>() / count;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 204: `(metrics.iter().map(|m| m.memory_usage).sum::<usize>() as f64 / count) as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 206: `(metrics.iter().map(|m| m.disk_read_bps).sum::<u64>() as f64 / count) as u64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 208: `(metrics.iter().map(|m| m.disk_write_bps).sum::<u64>() as f64 / count) as u64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 210: `(metrics.iter().map(|m| m.network_rx_bps).sum::<u64>() as f64 / count) as u64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 212: `(metrics.iter().map(|m| m.network_tx_bps).sum::<u64>() as f64 / count) as u64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 214: `(metrics.iter().map(|m| m.process_count).sum::<usize>() as f64 / count) as usize...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 215: `let avg_load = metrics.iter().map(|m| m.load_average).sum::<f64>() / count;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 238: `while *running.lock().unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 240: `let mut history = metrics_history.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 430: `let usage = 100.0 - (idle_diff as f64 / total_diff as f64) * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 613: `(metrics.memory_usage as f64 / metrics.memory_total as f64) * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 633: `total_disk_io as f64 / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 647: `total_network_io as f64 / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 723: `assert!(!*monitor.running.lock().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 748: `let mut history = monitor.metrics_history.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/random.rs

58 issues found:

- Line 79: `self.sample(rand_distr::Uniform::new(min, max).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 84: `let dist = rand_distr::Bernoulli::new(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 90: `let dist = rand_distr::Bernoulli::new(prob).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 118: `Array::from_shape_vec(shape, values).unwrap()`
  - **Fix**: Handle array creation errors properly
- Line 176: `Uniform::new(0.0_f64, 1.0_f64).unwrap().sample(&mut rng.rng)`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 182: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 189: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 196: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 202: `rdistr::Exp::new(lambda).unwrap().sample(&mut rng.rng)`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 215: `rng.sample_array(Uniform::new_inclusive(min, max).unwrap(), shape)`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 228: `rng.sample_array(Uniform::new(min, max).unwrap(), shape)`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 237: `let dist = Uniform::new(0, data_size).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 286: `(x as f64) / (u64::MAX as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 307: `Array::from_shape_vec(shape, values).unwrap()`
  - **Fix**: Handle array creation errors properly
- Line 363: `*point_val = value as f64 / (1u64 << 32) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 384: `Array::from_shape_vec((count, self.dimensions), data).unwrap()`
  - **Fix**: Handle array creation errors properly
- Line 441: `Array::from_shape_vec((count, self.dimensions), data).unwrap()`
  - **Fix**: Handle array creation errors properly
- Line 451: `let mut f = 1.0 / base as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 484: `(i as f64 + self.rng.sample(Uniform::new(0.0, 1.0).unwrap())) / count as f64`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 577: `.map(|_| self.sample(Uniform::new(0u8, 255u8).unwrap()))`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 588: `self.sample(Uniform::new(0.0, 1.0).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 596: `self.sample(Uniform::new(min, max).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 626: `.map(|_| self.rng.sample(Uniform::new(0.0, 1.0).unwrap()))`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 643: `let stratum_start = i as f64 / strata as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 644: `let stratum_end = (i + 1) as f64 / strata as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 647: `let uniform_in_stratum = self.rng.sample(Uniform::new(0.0, 1.0).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 685: `let target_mean = target_samples.iter().sum::<f64>() / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 686: `let control_sample_mean = control_samples.iter().sum::<f64>() / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 700: `self.optimal_coefficient = Some(numerator / denominator);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 757: `let weight = target_pdf(sample) / proposal_pdf(sample);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 791: `weighted_sum / weight_sum`
  - **Fix**: Division without zero check - use safe_divide()
- Line 812: `initial_samples / 2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 814: `let normal_dist = Normal::new(proposal_mean, proposal_std).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 823: `let normal_log_pdf = -0.5 * ((sample - proposal_mean) / proposal_std).powi(2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 824: `- 0.5 * (2.0 * std::f64::consts::PI).ln()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 825: `- proposal_std.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 836: `weights.iter().map(|w| w / weight_sum).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 850: `proposal_std = variance.sqrt().max(0.1); // Prevent collapse`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 956: `Array::from_shape_vec(shape, samples).unwrap()`
  - **Fix**: Handle array creation errors properly
- Line 1026: `let standard_normal = Normal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1056: `Array::from_shape_vec((count, self.dimensions), data).unwrap()`
  - **Fix**: Handle array creation errors properly
- Line 1071: `l[i][j] = diagonal.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1074: `l[i][j] = (matrix[i][j] - sum) / l[j][j];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1099: `.map(|&alpha| Gamma::new(alpha, 1.0).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1117: `gamma_samples.into_iter().map(|x| x / sum).collect()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1134: `Array::from_shape_vec((count, dimensions), data).unwrap()`
  - **Fix**: Handle array creation errors properly
- Line 1159: `return rng.sample(Uniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1163: `let uniform = Uniform::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1174: `/ (2.0 * std::f64::consts::PI * self.bessel_i0(self.kappa));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1191: `let y = (x / 3.75).powi(2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1197: `let y = 3.75 / ax;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1198: `(ax.exp() / ax.sqrt())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1198: `(ax.exp() / ax.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1280: `let proposal_sampler = |rng: &mut Random<_>| rng.sample(Uniform::new(-3.0, 3.0)....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1295: `let mvn = specialized_distributions::MultivariateNormal::new(mean, cov).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1305: `let dirichlet = specialized_distributions::Dirichlet::new(alphas).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1317: `let von_mises = specialized_distributions::VonMises::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1331: `let result = pool.with_rng(|rng| rng.sample(Uniform::new(0.0, 1.0).unwrap()));`
  - **Fix**: Replace with ? operator or .ok_or()

### src/random/gpu.rs

21 issues found:

- Line 196: `output[gid] = (float)(seed >> 11) * (1.0f / 9007199254740992.0f);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 222: `if (gid >= count / 2) return;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 241: `float u1 = (float)(seed1 >> 11) * (1.0f / 9007199254740992.0f);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 242: `float u2 = (float)(seed2 >> 11) * (1.0f / 9007199254740992.0f);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 292: `output[gid] = (float)(seed >> 32) * (1.0f / 4294967296.0f);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 354: `output[gid] = (float)(c0 >> 8) * (1.0f / 16777216.0f);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 411: `let state = self.state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 437: `let mut state = self.state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 461: `let state = self.state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 477: `let num_work_groups = ((count / 2).div_ceil(self.work_group_size)) as u32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 487: `let mut state = self.state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 516: `.map(|&u| -(-u.ln()) / lambda)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 516: `.map(|&u| -(-u.ln()) / lambda)`
  - **Fix**: Mathematical operation .ln() without validation
- Line 543: `let r = (-2.0 * (u1 + 1e-7_f32).ln()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 543: `let r = (-2.0 * (u1 + 1e-7_f32).ln()).sqrt();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 576: `let state = self.state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 589: `let state = self.state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 597: `let mut state = self.state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 608: `let mut state = self.state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 776: `.map(|_| cpu_rng.sample(rand_distr::Uniform::new(0.0, 1.0).unwrap()))`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 784: `speedup: cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64(),`
  - **Fix**: Division without zero check - use safe_divide()

### src/random/qmc.rs

33 issues found:

- Line 194: `let base_discrepancy = (self.dimension as f64).ln() / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 194: `let base_discrepancy = (self.dimension as f64).ln() / (n as f64);`
  - **Fix**: Mathematical operation .ln() without validation
- Line 221: `.map(|&x| (x as f64) / (1u64 << 32) as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 293: `let limit = (n as f64).sqrt() as u32 + 1;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 308: `result += (n % base as u64) as f64 / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 391: `let lh_value = (perm_val as f64 + uniform_sample) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 440: `.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 532: `result += scrambled_digit as f64 / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 617: `let mean = sum / n_points as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 618: `let variance = (sum_sq / n_points as f64) - (mean * mean);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 620: `let error = volume * (variance / n_points as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 620: `let error = volume * (variance / n_points as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 623: `let convergence_rate = (dimension as f64 * (n_points as f64).ln()) / n_points as...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 623: `let convergence_rate = (dimension as f64 * (n_points as f64).ln()) / n_points as...`
  - **Fix**: Mathematical operation .ln() without validation
- Line 645: `let points_per_thread = n_points / n_threads;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 658: `let mut generator = create_generator(sequence_type, dimension).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 678: `results_clone.lock().unwrap().push((sum, sum_sq));`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 685: `handle.join().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 688: `let results = results.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 694: `let mean = total_sum / n_points as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 695: `let variance = (total_sum_sq / n_points as f64) - (mean * mean);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 697: `let error = volume * (variance / n_points as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 697: `let error = volume * (variance / n_points as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 698: `let convergence_rate = (dimension as f64 * (n_points as f64).ln()) / n_points as...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 698: `let convergence_rate = (dimension as f64 * (n_points as f64).ln()) / n_points as...`
  - **Fix**: Mathematical operation .ln() without validation
- Line 755: `let mut sobol = SobolGenerator::new(2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 791: `let halton = HaltonGenerator::with_prime_bases(3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 798: `let points = lhs.sample(10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 807: `sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 818: `let mut faure = FaureGenerator::new(2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 847: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 856: `let mut sobol = SobolGenerator::new(2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 868: `let sobol = SobolGenerator::new(2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/resource/cpu.rs

8 issues found:

- Line 325: `(core_score + freq_score + cache_score + simd_score) / 4.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 454: `self.vector_width_bytes() / 4`
  - **Fix**: Division without zero check - use safe_divide()
- Line 459: `self.vector_width_bytes() / 8`
  - **Fix**: Division without zero check - use safe_divide()
- Line 526: `let cpu = cpu_info.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 579: `assert_eq!(CpuInfo::parse_cache_size("32K").unwrap(), 32);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 580: `assert_eq!(CpuInfo::parse_cache_size("256k").unwrap(), 256);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 581: `assert_eq!(CpuInfo::parse_cache_size("8M").unwrap(), 8192);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 582: `assert_eq!(CpuInfo::parse_cache_size("1024").unwrap(), 1024);`
  - **Fix**: Replace with ? operator or .ok_or()

### src/resource/gpu.rs

2 issues found:

- Line 223: `(memory_score + compute_score + bandwidth_score + efficiency_score) / 4.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 374: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()

### src/resource/memory.rs

5 issues found:

- Line 240: `let availability_score = self.available_memory as f64 / self.total_memory as f64...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 242: `(capacity_score + bandwidth_score + latency_score + availability_score) / 4.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 248: `let available_mb = self.available_memory / (1024 * 1024);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 300: `(self.used as f64 / self.total as f64) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 337: `let memory = memory_info.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/resource/mod.rs

8 issues found:

- Line 107: `let combined_score = (cpu_score + memory_score + gpu_score) / 3.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 160: `self.memory.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 164: `self.memory.available_memory as f64 / (1024.0 * 1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 168: `self.memory.page_size / 1024`
  - **Fix**: Division without zero check - use safe_divide()
- Line 177: `gpu.memory_total as f64 / (1024.0 * 1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 190: `self.optimization_params.chunk_size / 1024`
  - **Fix**: Division without zero check - use safe_divide()
- Line 592: `assert!(thread_count.unwrap() > 0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 596: `assert!(chunk_size.unwrap() > 0);`
  - **Fix**: Replace with ? operator or .ok_or()

### src/resource/optimization.rs

8 issues found:

- Line 91: `(cpu.logical_cores - cpu.physical_cores) / 2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 124: `.powf(1.0 / 3.0)) as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 124: `.powf(1.0 / 3.0)) as usize;`
  - **Fix**: Mathematical operation .powf( without validation
- Line 152: `let ratio = problem_size as f64 / base_size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 154: `ratio.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 221: `let prefetch_distance = (cpu.cache_l1_kb * 1024 / 16).clamp(64, 1024);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 224: `let tile_size = (cpu.cache_l1_kb * 1024 / 8).clamp(64, 4096);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 358: `let params = params.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/safe_ops.rs

22 issues found:

- Line 20: `"Division by zero: {} / 0",`
  - **Fix**: Division without zero check - use safe_divide()
- Line 29: `"Division by near-zero value: {} / {} (threshold: {})",`
  - **Fix**: Division without zero check - use safe_divide()
- Line 34: `let result = numerator / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 39: `"Division produced non-finite result: {} / {} = {:?}",`
  - **Fix**: Division without zero check - use safe_divide()
- Line 60: `let result = value.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 86: `let result = value.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 144: `let result = base.powf(exponent);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 257: `assert_eq!(safe_divide(10.0, 2.0).unwrap(), 5.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 258: `assert_eq!(safe_divide(-10.0, 2.0).unwrap(), -5.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 271: `assert_eq!(safe_sqrt(4.0).unwrap(), 2.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 272: `assert_eq!(safe_sqrt(0.0).unwrap(), 0.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 282: `assert!((safe_log(std::f64::consts::E).unwrap() - 1.0).abs() < 1e-10);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 283: `assert_eq!(safe_log(1.0).unwrap(), 0.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 293: `assert_eq!(safe_pow(2.0, 3.0).unwrap(), 8.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 294: `assert_eq!(safe_pow(4.0, 0.5).unwrap(), 2.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 307: `assert!((safe_exp(1.0).unwrap() - std::f64::consts::E).abs() < 1e-10);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 308: `assert_eq!(safe_exp(0.0).unwrap(), 1.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 317: `assert_eq!(safe_mean(&[1.0, 2.0, 3.0]).unwrap(), 2.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 323: `assert_eq!(safe_mean(&[5.0]).unwrap(), 5.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 331: `assert!((safe_variance(&values, mean).unwrap() - 2.5).abs() < 1e-10);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 341: `assert_eq!(safe_normalize(3.0, 4.0).unwrap(), 0.75);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 347: `assert_eq!(safe_normalize(0.0, 0.0).unwrap(), 0.0);`
  - **Fix**: Replace with ? operator or .ok_or()

### src/simd.rs

42 issues found:

- Line 88: `let a_slice = a.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 89: `let b_slice = b.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 90: `let result_slice = result.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 166: `let a_slice = a.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 167: `let b_slice = b.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 168: `let result_slice = result.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 226: `let a_slice = a.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 227: `let b_slice = b.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 228: `let result_slice = result.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 304: `let a_slice = a.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 305: `let b_slice = b.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 306: `let result_slice = result.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 359: `let step = (end - start) / (num as f32 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 360: `let result_slice = result.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 413: `let step = (end - start) / (num as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 414: `let result_slice = result.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 472: `let a_slice = a.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 473: `let b_slice = b.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 474: `let result_slice = result.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 535: `let a_slice = a.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 536: `let b_slice = b.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 537: `let result_slice = result.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 575: `let a_slice = a.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 576: `let b_slice = b.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 577: `let result_slice = result.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 633: `let a_slice = a.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 634: `let b_slice = b.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 635: `let result_slice = result.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 779: `let a_slice = a.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 780: `let result_slice = result.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 820: `let a_slice = a.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 821: `let result_slice = result.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 863: `let a_slice = a.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 864: `let b_slice = b.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 865: `let result_slice = result.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 979: `let input_slice = input.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 980: `let result_slice = result.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1006: `let input_slice = input.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1098: `let a_slice = a.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1099: `let b_slice = b.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1100: `let c_slice = c.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1101: `let result_slice = result.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/simd_ops.rs

8 issues found:

- Line 117: `a / b`
  - **Fix**: Division without zero check - use safe_divide()
- Line 183: `Self::simd_dot(a, a).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 238: `Self::simd_sum(a) / (a.len() as f32)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 273: `a.mapv(|x| x.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 314: `a / b`
  - **Fix**: Division without zero check - use safe_divide()
- Line 380: `Self::simd_dot(a, a).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 429: `Self::simd_sum(a) / (a.len() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 458: `a.mapv(|x| x.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/testing/fuzzing.rs

3 issues found:

- Line 270: `(self.min_value + self.max_value) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 359: `let size = (self.min_size + self.max_size) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 480: `Ok(1.0 / x)`
  - **Fix**: Division without zero check - use safe_divide()

### src/testing/large_scale.rs

34 issues found:

- Line 226: `println!("Generating {} MB numeric dataset...", size / (1024 * 1024));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 239: `let _num_elements_per_chunk = chunk_size / std::mem::size_of::<f64>();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 245: `let elements_in_chunk = current_chunk_size / std::mem::size_of::<f64>();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 249: `.map(|i| (bytes_written / std::mem::size_of::<f64>() + i) as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 266: `let progress = (bytes_written * 100) / size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 285: `size / (1024 * 1024),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 299: `let _num_elements_per_chunk = chunk_size / std::mem::size_of::<f64>();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 308: `let elements_in_chunk = current_chunk_size / std::mem::size_of::<f64>();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 324: `if (bytes_written / std::mem::size_of::<f64>()) % (1.0 / density) as usize`
  - **Fix**: Division without zero check - use safe_divide()
- Line 409: `file_size / (1024 * 1024)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 426: `let _elements_per_chunk = chunk_size / std::mem::size_of::<f64>();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 431: `let elements_in_chunk = current_chunk_size / std::mem::size_of::<f64>();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 452: `let progress = (bytes_processed * 100) / file_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 458: `let throughput = file_size as f64 / duration.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 472: `throughput / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 502: `let num_elements = file_size / std::mem::size_of::<f64>();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 505: `println!("Memory-mapping {} MB dataset...", file_size / (1024 * 1024));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 518: `let chunk_size = self.config.chunk_size / std::mem::size_of::<f64>();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 534: `let progress = (chunk_start * 100) / num_elements;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 540: `let throughput = file_size as f64 / duration.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 554: `throughput / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 620: `let elements = bytes_read / std::mem::size_of::<f64>();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 668: `Ok(chunk.iter().sum::<f64>() / chunk.len() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 759: `let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 761: `chunk.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / chunk.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 809: `let generator = LargeDatasetGenerator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 810: `let dataset_path = generator.generate_numeric_dataset(1024).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 814: `let metadata = fs::metadata(&dataset_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 821: `let generator = LargeDatasetGenerator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 823: `let dataset_path = generator.generate_sparse_dataset(1024, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 826: `let metadata = fs::metadata(&dataset_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 834: `let generator = LargeDatasetGenerator::new(config.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 837: `let dataset_path = generator.generate_numeric_dataset(1024).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 841: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/testing/property_based.rs

7 issues found:

- Line 295: `(self.min_value + self.max_value) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 306: `(min + max) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 769: `let result = property.test(&inputs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 778: `let result = property.test(&inputs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 787: `let result = property.test(&inputs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 796: `let result = property.test(&inputs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 805: `let result = property.test(&inputs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/testing/security.rs

1 issues found:

- Line 465: `memory_growth / (1024 * 1024)`
  - **Fix**: Division without zero check - use safe_divide()

### src/testing/stress.rs

15 issues found:

- Line 255: `.with_total_operations(current_memory / self.config.memory_step);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 293: `let ops_per_second = operations as f64 / start_time.elapsed().as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 376: `let ops_per_second = total_operations as f64 / start_time.elapsed().as_secs_f64(...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 406: `Self::cpu_intensive_computation_static(config.cpu_intensity / 10)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 433: `let ops_per_second = total_operations as f64 / start_time.elapsed().as_secs_f64(...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 540: `let ops_per_second = total_operations as f64 / start_time.elapsed().as_secs_f64(...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 610: `let ops_per_second = total_operations as f64 / start_time.elapsed().as_secs_f64(...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 648: `result.error.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 667: `result.error.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 687: `result.error.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 706: `result.error.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 726: `result.error.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 745: `result.error.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 802: `let result = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 818: `let result = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/types.rs

24 issues found:

- Line 333: `(self.re * self.re + self.im * self.im).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 350: `Complex::new(self.re / mag, self.im / mag)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 513: `assert_eq!(42.0f64.to_numeric::<i32>().unwrap(), 42);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 514: `assert_eq!((-42.0f64).to_numeric::<i32>().unwrap(), -42);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 551: `let z_rot = z1.rotate(std::f64::consts::PI / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 567: `let z2 = z1.convert_complex::<f32>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 905: `value / self.scale_factor - self.offset`
  - **Fix**: Division without zero check - use safe_divide()
- Line 986: `self.unit.scale_factor / other.unit.scale_factor,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 989: `Quantity::new(self.value / other.value, new_unit)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1099: `5.0 / 9.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1177: `self.raw as f64 / Self::FACTOR as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1202: `raw: (self.raw * other.raw) / Self::FACTOR,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1209: `raw: (self.raw * Self::FACTOR) / other.raw,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1257: `remainder = 1.0 / remainder;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1301: `self.numerator as f64 / self.denominator as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1394: `(self.lower + self.upper) / 2.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1456: `self.lower / other.lower,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1457: `self.lower / other.upper,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1458: `self.upper / other.lower,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1459: `self.upper / other.upper,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1480: `let y = x.convert_tracked::<f32>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1492: `let distance = registry.quantity(1000.0, "m").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1493: `let km_unit = registry.get_unit("km").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1494: `let converted = distance.convert_to(km_unit).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/types/dynamic_dispatch.rs

3 issues found:

- Line 768: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 828: `assert_eq!(grouped.get(&TypeCategory::Scalar).unwrap().len(), 2);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 829: `assert_eq!(grouped.get(&TypeCategory::String).unwrap().len(), 1);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?

### src/ufuncs/binary.rs

4 issues found:

- Line 128: `x / y`
  - **Fix**: Division without zero check - use safe_divide()
- Line 155: `apply_binary(inputs[0], inputs[1], output, |&x: &f64, &y: &f64| x.powf(y))`
  - **Fix**: Mathematical operation .powf( without validation
- Line 380: `x / y`
  - **Fix**: Division without zero check - use safe_divide()
- Line 439: `broadcast_apply(a_view, b_view, |x, y| x.powf(*y)).unwrap_or_else(|_| {`
  - **Fix**: Mathematical operation .powf( without validation

### src/ufuncs/core.rs

16 issues found:

- Line 62: `let mut registry = UFUNC_REGISTRY.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 74: `let registry = UFUNC_REGISTRY.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 108: `let registry = UFUNC_REGISTRY.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 141: `let input_slice = input.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 142: `let output_slice = output.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 186: `let input1_slice = input1.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 187: `let input2_slice = input2.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 188: `let output_slice = output.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 276: `let mut acc = initial.clone().unwrap_or_else(|| iter.next().unwrap().clone());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 351: `register_ufunc(ufunc).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 354: `let ufunc = get_ufunc("test_unary").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 364: `apply_unary(&input, &mut output, |&x: &f64| x * x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 375: `apply_binary(&input1, &input2, &mut output, |&x: &f64, &y: &f64| x + y).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 386: `apply_reduction(&input, &mut output, Some(0), Some(0.0), |acc, &x| acc + x).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 391: `apply_reduction(&input, &mut output, Some(1), Some(0.0), |acc, &x| acc + x).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 396: `apply_reduction(&input, &mut output, None, Some(0.0), |acc, &x| acc + x).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/ufuncs/math.rs

9 issues found:

- Line 149: `apply_unary(inputs[0], output, |&x: &f64| x.ln())`
  - **Fix**: Mathematical operation .ln() without validation
- Line 174: `apply_unary(inputs[0], output, |&x: &f64| x.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 240: `sin_ufunc.apply(&[array], &mut result).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 280: `cos_ufunc.apply(&[array], &mut result).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 319: `tan_ufunc.apply(&[array], &mut result).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 357: `exp_ufunc.apply(&[array], &mut result).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 395: `log_ufunc.apply(&[array], &mut result).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 432: `sqrt_ufunc.apply(&[array], &mut result).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 469: `abs_ufunc.apply(&[array], &mut result).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/ufuncs/mod.rs

33 issues found:

- Line 52: `array.mapv(|x| x.ln())`
  - **Fix**: Mathematical operation .ln() without validation
- Line 60: `array.mapv(|x| x.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 188: `array.mapv(|x| x * T::PI() / T::from(180.0).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 196: `array.mapv(|x| x * T::from(180.0).unwrap() / T::PI())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 220: `array.mapv(|x| T::one() / x)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 266: `array.mapv(|x| x.ln())`
  - **Fix**: Mathematical operation .ln() without validation
- Line 274: `array.mapv(|x| x.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 402: `array.mapv(|x| x * T::PI() / T::from(180.0).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 410: `array.mapv(|x| x * T::from(180.0).unwrap() / T::PI())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 434: `array.mapv(|x| T::one() / x)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 521: `.map(|(a_elem, b_elem)| a_elem.clone() / b_elem.clone())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 542: `.map(|(a_elem, b_elem)| a_elem.powf(*b_elem))`
  - **Fix**: Mathematical operation .powf( without validation
- Line 596: `ndarray_ext::broadcast_apply(*a, *b, |x, y| x.clone() / y.clone())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 607: `ndarray_ext::broadcast_apply(*a, *b, |x, y| x.powf(*y))`
  - **Fix**: Mathematical operation .powf( without validation
- Line 737: `let n = T::from_usize(rows).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 744: `result[j] = sum / n.clone();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 752: `let n = T::from_usize(cols).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 759: `result[i] = sum / n.clone();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 770: `let n = T::from_usize(rows * cols).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 777: `Array::from_vec(vec![sum / n])`
  - **Fix**: Division without zero check - use safe_divide()
- Line 804: `let n = T::from_usize(rows).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 812: `result[j] = sum_sq_diff / n.clone();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 821: `let n = T::from_usize(cols).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 829: `result[i] = sum_sq_diff / n.clone();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 840: `let n = T::from_usize(rows * cols).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 852: `Array::from_vec(vec![sum_sq_diff / n])`
  - **Fix**: Division without zero check - use safe_divide()
- Line 873: `variances.mapv(|x| x.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1016: `assert_eq!(result, array![1.0, 2.0, 9.0_f64.sqrt()]);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1018: `let a = array![0.0, PI / 2.0, PI];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1030: `let result = binary::add(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1033: `let result = binary::multiply(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1042: `let result = binary2d::add(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1045: `let result = binary2d::multiply(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/ufuncs/reduction.rs

16 issues found:

- Line 144: `output1d[0] = sum / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 191: `let mean = sum / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 192: `let variance = sum_sq / count as f64 - mean * mean;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 194: `output1d[0] = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 241: `let mean = sum / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 242: `let variance = sum_sq / count as f64 - mean * mean;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 421: `|acc, &x| acc + x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 425: `|acc, &x| acc + x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 477: `|acc, &x| acc * x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 481: `|acc, &x| acc * x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 532: `sum_result.map(|&x| x / axis_len)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 537: `Array::from_vec(vec![sum_result[0] / total_elements])`
  - **Fix**: Division without zero check - use safe_divide()
- Line 576: `var_result.map(|&x| x.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 641: `output[j] = sum_sq_diff / axis_len;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 654: `output[i] = sum_sq_diff / axis_len;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 673: `output[0] = sum_sq_diff / total_elements;`
  - **Fix**: Division without zero check - use safe_divide()

### src/units.rs

26 issues found:

- Line 146: `(value - self.si_offset) / self.si_factor`
  - **Fix**: Division without zero check - use safe_divide()
- Line 592: `5.0 / 9.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 593: `459.67 * 5.0 / 9.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 603: `std::f64::consts::PI / 180.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 749: `celsius * 9.0 / 5.0 + 32.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 753: `(fahrenheit - 32.0) * 5.0 / 9.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 766: `meters / 0.3048`
  - **Fix**: Division without zero check - use safe_divide()
- Line 778: `cm / 2.54`
  - **Fix**: Division without zero check - use safe_divide()
- Line 783: `kg / 0.453_592_37`
  - **Fix**: Division without zero check - use safe_divide()
- Line 792: `degrees * std::f64::consts::PI / 180.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 796: `radians * 180.0 / std::f64::consts::PI`
  - **Fix**: Division without zero check - use safe_divide()
- Line 801: `joules / 4.184`
  - **Fix**: Division without zero check - use safe_divide()
- Line 813: `joules / 1.602_176_634e-19`
  - **Fix**: Division without zero check - use safe_divide()
- Line 854: `let result = registry.convert(1000.0, "m", "km").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 858: `let result = registry.convert(1.0, "m", "ft").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 862: `let result = registry.convert(1.0, "in", "cm").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 871: `let result = registry.convert(0.0, "°C", "K").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 875: `let result = registry.convert(32.0, "°F", "°C").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 879: `let result = registry.convert(32.0, "°F", "K").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 888: `let result = registry.convert(180.0, "°", "rad").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 893: `.convert(std::f64::consts::PI / 2.0, "rad", "°")`
  - **Fix**: Division without zero check - use safe_divide()
- Line 894: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 903: `let result = registry.convert(4.184, "J", "cal").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 907: `let result = registry.convert(1.0, "eV", "J").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 927: `let converted: UnitValue<f64> = registry.convert_value(&value, "km").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 971: `let result = convert(1000.0, "m", "km").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/utils.rs

41 issues found:

- Line 278: `let step = (end - start) / F::from(num - 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 286: `start + step * F::from(i).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 298: `let step = (end - start) / F::from(num - 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 302: `let value = start + step * F::from(i).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 333: `let base = base.unwrap_or_else(|| F::from(10.0).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 339: `linear.mapv(|x| base.powf(x))`
  - **Fix**: Mathematical operation .powf( without validation
- Line 407: `let b_val = b.iter().nth(i).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 482: `let b_val = b.iter().nth(i).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 544: `let scale = 1.0 / sum_of_squares.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 544: `let scale = 1.0 / sum_of_squares.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 557: `let scale = 1.0 / peak;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 570: `let scale = 1.0 / sum;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 583: `let scale = 1.0 / max_val;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 761: `let mean_val = sum / T::from_usize(input_len).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 832: `0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 840: `0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 847: `let w = 0.42 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).co...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 848: `+ 0.08 * (4.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 854: `let m = (n - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 856: `let w = 1.0 - ((i as f64 - m) / m).abs();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 866: `let m = (length - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 868: `let w = 1.0 - ((i as f64 - m) / (m + 1.0)).abs();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 916: `let derivative = (f_plus - f_minus) / (F::from(2.0).unwrap() * h);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 967: `let h = (b - a) / F::from_usize(n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 974: `let x_i = a + F::from_usize(i).unwrap() * h;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 976: `+ F::from(2.0).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 985: `let x_i = a + F::from_usize(i).unwrap() * h;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 987: `+ F::from(4.0).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 993: `let integral = h * sum / F::from(3.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1154: `let normalized = normalize(&signal, "energy").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1162: `let normalized = normalize(&signal, "peak").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1171: `let normalized = normalize(&signal, "sum").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1182: `let padded = pad_array(&arr, &[(1, 2)], "constant", Some(0.0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1189: `let padded = pad_array(&arr, &[(2, 2)], "edge", None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1196: `let padded = pad_array(&arr, &[(1, 1)], "maximum", None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1205: `let window = get_window("hamming", 5, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1212: `let window = get_window("hann", 5, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1219: `let window = get_window("rectangular", 5, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1230: `let derivative = differentiate(3.0, 0.001, f).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1234: `let integral = integrate(0.0, 1.0, 100, f).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1238: `let integral = integrate(0.0, 2.0, 100, f).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/validation.rs

1 issues found:

- Line 446: `num_traits::cast(eps).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()

### src/validation/data/array_validation.rs

9 issues found:

- Line 260: `let std_dev = array.std(num_traits::cast(1.0).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 473: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 487: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 494: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 514: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 537: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 547: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 562: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 579: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/validation/data/constraints.rs

2 issues found:

- Line 360: `1 => self.constraints.into_iter().next().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 369: `1 => self.constraints.into_iter().next().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()

### src/validation/data/mod.rs

9 issues found:

- Line 166: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 175: `let validator = Validator::new(config.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 179: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 185: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 195: `let validator = Validator::new(config.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 198: `Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unw...`
  - **Fix**: Handle array creation errors properly
- Line 207: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 228: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 235: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/validation/data/quality.rs

27 issues found:

- Line 111: `(total_elements - nan_count) as f64 / total_elements as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 139: `(total_elements - nan_count - inf_count) as f64 / total_elements as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 186: `let quality_score = (completeness + validity + consistency + accuracy) / 4.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 233: `/ num_traits::cast(finite_values.len()).unwrap_or(T::one());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 242: `/ num_traits::cast(finite_values.len()).unwrap_or(T::one());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 244: `let std_dev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 254: `sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 280: `let q1_index = sorted_values.len() / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 281: `let q3_index = 3 * sorted_values.len() / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 305: `/ num_traits::cast(sorted_values.len()).unwrap_or(T::one());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 314: `/ num_traits::cast(sorted_values.len()).unwrap_or(T::one());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 316: `let std_dev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 322: `let diff = (x - mean) / std_dev;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 326: `/ num_traits::cast(sorted_values.len()).unwrap_or(T::one());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 357: `let outlier_percentage = (stats.outliers as f64 / stats.count as f64) * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 412: `let mean_diff = diff_scores.iter().sum::<f64>() / diff_scores.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 417: `/ diff_scores.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 421: `for period in 2..((values.len() / 2).min(10)) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 433: `let current_score = matches as f64 / comparisons as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 441: `(-variance.ln()).exp().min(1.0).max(0.0)`
  - **Fix**: Mathematical operation .ln() without validation
- Line 599: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 614: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 630: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 645: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 648: `let stats = report.metrics.statistical_summary.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 662: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 678: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/validation/data/validator.rs

53 issues found:

- Line 810: `let mean = numeric_values.iter().sum::<f64>() / count;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 817: `/ count;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 818: `let std_dev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1495: `Ok(total_hits as f64 / total_entries as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1501: `Self::new(ValidationConfig::default()).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1513: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1523: `let validator = Validator::new(config.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1533: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1540: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1545: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1554: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1560: `let (size, hit_rate) = validator.get_cache_stats().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1569: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1581: `let result = validator.validate(&valid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1589: `let result = validator.validate(&invalid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1598: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1615: `let result = validator.validate(&valid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1621: `let result = validator.validate(&invalid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1629: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1639: `let result = validator.validate(&valid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1645: `let result = validator.validate(&invalid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1653: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1663: `let result = validator.validate(&valid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1669: `let result = validator.validate(&invalid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1677: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1693: `let result = validator.validate(&valid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1699: `let result = validator.validate(&invalid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1706: `let validator = Validator::new(ValidationConfig::default()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1725: `let result = validator.validate(&valid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1731: `let result = validator.validate(&invalid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1748: `let result = validator.validate(&valid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1754: `let result = validator.validate(&invalid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1768: `let result = validator.validate(&valid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1774: `let result = validator.validate(&invalid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1797: `let result = validator.validate(&valid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1804: `let validator = Validator::new(ValidationConfig::default()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1812: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1820: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1847: `let result = validator.validate(&valid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1852: `let result = validator.validate(&invalid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1882: `let result = validator.validate(&valid_high, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1889: `let result = validator.validate(&valid_low, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1896: `let result = validator.validate(&invalid_high, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1903: `let result = validator.validate(&invalid_low, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1919: `let result = validator.validate(&valid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1923: `let result = validator.validate(&invalid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1930: `let validator = Validator::new(ValidationConfig::default()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1944: `let result = validator.validate(&invalid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1970: `let result = validator.validate(&invalid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1980: `let validator = Validator::new(ValidationConfig::default()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1996: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2009: `let result = validator.validate(&valid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2013: `let result = validator.validate(&invalid_data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/validation/production.rs

6 issues found:

- Line 660: `let average_duration = total_duration / total_validations as u32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 664: `let success_rate = successful_validations as f64 / total_validations as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 744: `if dim > usize::MAX / 1024 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 760: `const MAX_TOTAL_SIZE: usize = 1024 * 1024 * 1024 / 8;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 925: `let metrics = validator.get_metrics().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 927: `assert_eq!(metrics.success_rate, 2.0 / 3.0);`
  - **Fix**: Division without zero check - use safe_divide()

### src/versioning/compatibility.rs

29 issues found:

- Line 570: `let v1 = ApiVersionBuilder::new(Version::parse("1.0.0").unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 573: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 574: `let v2 = ApiVersionBuilder::new(Version::parse("1.1.0").unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 579: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 581: `checker.register_version(&v1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 582: `checker.register_version(&v2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 586: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 594: `let v1 = ApiVersionBuilder::new(Version::parse("1.0.0").unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 597: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 598: `let v2 = ApiVersionBuilder::new(Version::parse("2.0.0").unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 601: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 603: `checker.register_version(&v1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 604: `checker.register_version(&v2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 608: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 617: `let v1 = ApiVersionBuilder::new(Version::parse("1.0.0").unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 619: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 620: `let v2 = ApiVersionBuilder::new(Version::parse("2.0.0").unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 623: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 625: `checker.register_version(&v1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 626: `checker.register_version(&v2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 630: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 632: `assert!(report.estimated_migration_effort.unwrap() > 0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 639: `let v1 = ApiVersionBuilder::new(Version::parse("1.0.0").unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 643: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 644: `let v2 = ApiVersionBuilder::new(Version::parse("1.1.0").unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 647: `.build().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 649: `checker.register_version(&v1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 650: `checker.register_version(&v2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 654: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/versioning/deprecation.rs

8 issues found:

- Line 504: `let latest_major = major_keys.last().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 702: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 703: `manager.register_version(&api_version).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 713: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 720: `let status = manager.get_deprecation_status(&version).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 742: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 743: `manager.register_version(&api_version).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 752: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/versioning/migration.rs

7 issues found:

- Line 341: `from_version: path.first().unwrap().clone(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 342: `to_version: path.last().unwrap().clone(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 590: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 605: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 617: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 636: `manager.start_migration(plan, execution_id.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 640: `assert_eq!(status.unwrap().status, ExecutionStatus::NotStarted);`
  - **Fix**: Replace with ? operator or .ok_or()

### src/versioning/mod.rs

15 issues found:

- Line 687: `let version = Version::parse("1.0.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 693: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 704: `let version = Version::parse("1.0.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 705: `let api_version = ApiVersionBuilder::new(version.clone()).build().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 707: `manager.register_version(api_version).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 715: `let version = Version::parse("1.0.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 716: `let api_version = ApiVersionBuilder::new(version.clone()).build().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 718: `manager.register_version(api_version).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 719: `manager.set_current_version(version.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 745: `let v1 = ApiVersionBuilder::new(Version::parse("1.0.0").unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 748: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 749: `let v2 = ApiVersionBuilder::new(Version::parse("2.0.0").unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 753: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 755: `manager.register_version(v1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 756: `manager.register_version(v2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/versioning/negotiation.rs

13 issues found:

- Line 245: `Ok(versions.into_iter().next().unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 261: `Ok(versions.into_iter().next().unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 275: `Ok(versions.into_iter().next().unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 281: `Ok(versions.into_iter().next().unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 290: `Ok(versions.into_iter().next().unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 297: `Ok(versions.into_iter().next().unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 444: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 471: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 483: `Version::parse("2.0.0-alpha").unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 489: `let v2 = Version::parse("2.0.0-alpha").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 495: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 515: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 533: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/versioning/semantic.rs

35 issues found:

- Line 548: `let version = Version::parse("1.2.3").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 551: `let version = Version::parse("1.2.3-alpha.1").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 556: `let version = Version::parse("1.2.3+build.123").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 559: `let version = Version::parse("v1.2.3").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 565: `let v1_0_0 = Version::parse("1.0.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 566: `let v1_0_1 = Version::parse("1.0.1").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 567: `let v1_1_0 = Version::parse("1.1.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 568: `let v2_0_0 = Version::parse("2.0.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 569: `let v1_0_0_alpha = Version::parse("1.0.0-alpha").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 579: `let mut version = Version::parse("1.2.3-alpha+build").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 593: `let constraint = VersionConstraint::parse(">=1.2.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 594: `let version = Version::parse("1.2.3").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 597: `let constraint = VersionConstraint::parse("^1.2.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 598: `let version = Version::parse("1.5.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 600: `let version = Version::parse("2.0.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 603: `let constraint = VersionConstraint::parse("~1.2.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 604: `let version = Version::parse("1.2.5").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 606: `let version = Version::parse("1.3.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 612: `let min = Version::parse("1.0.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 613: `let max = Version::parse("2.0.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 616: `let version = Version::parse("1.5.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 619: `let version = Version::parse("0.9.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 622: `let version = Version::parse("2.0.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 625: `let version = Version::parse("1.5.0-alpha").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 644: `let v1_0_0 = Version::parse("1.0.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 645: `let v1_2_0 = Version::parse("1.2.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 646: `let v2_0_0 = Version::parse("2.0.0").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 657: `Version::parse("1.0.0-alpha").unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 658: `Version::parse("1.0.0-alpha.1").unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 659: `Version::parse("1.0.0-alpha.beta").unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 660: `Version::parse("1.0.0-beta").unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 661: `Version::parse("1.0.0-beta.2").unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 662: `Version::parse("1.0.0-beta.11").unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 663: `Version::parse("1.0.0-rc.1").unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 664: `Version::parse("1.0.0").unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/array_protocol_tests.rs

11 issues found:

- Line 135: `let result = dist_array.to_array().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 161: `let jit_function = jit_array.compile(expression).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 168: `assert_eq!(info.get("supports_jit").unwrap(), "true");`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 197: `let mut registry_write = registry.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 219: `let registry = registry.read().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 282: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 287: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 321: `let mut registry_write = registry.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 536: `let mut registry_write = registry.write().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 705: `Ok(result) => Ok(*result.downcast_ref::<f64>().unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 720: `assert_eq!(sum.unwrap(), 42.0);`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/chunked_map_tests.rs

1 issues found:

- Line 51: `let results = chunked.map(|chunk| chunk.mean().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/gpu_metal_tests.rs

5 issues found:

- Line 36: `println!("  Memory: {} GB", memory / (1024 * 1024 * 1024));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 296: `let huge_size = usize::MAX / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 343: `let size_mb = (size * 4) as f64 / (1024.0 * 1024.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 348: `size_mb / 1024.0 / h2d_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 353: `size_mb / 1024.0 / d2h_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()

### tests/memmap_chunks_tests.rs

16 issues found:

- Line 16: `let temp_dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 18: `let mmap = create_mmap(&data, &file_path, AccessMode::Write, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 37: `let temp_dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 39: `let mmap = create_mmap(&data, &file_path, AccessMode::Write, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 66: `let temp_dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 68: `let mut mmap = create_mmap(&data, &file_path, AccessMode::Write, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 77: `let array = mmap.as_array::<ndarray::Ix1>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 90: `let temp_dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 94: `MemoryMappedArray::<i32>::save_array(&data, &file_path, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 98: `MemoryMappedArray::<i32>::open_zero_copy(&file_path, AccessMode::ReadWrite).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 101: `let original = mmap.as_array::<ndarray::Ix1>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 116: `mmap.flush().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 124: `MemoryMappedArray::<i32>::open_zero_copy(&file_path, AccessMode::ReadOnly).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 125: `let modified = reopened_mmap.as_array::<ndarray::Ix1>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 154: `let temp_dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 156: `let mmap = create_mmap(&data, &file_path, AccessMode::Write, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/memory_efficient_chunked_tests.rs

6 issues found:

- Line 50: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 69: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 87: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 101: `let n = ((OPTIMAL_CHUNK_SIZE / std::mem::size_of::<f64>()) as f64).sqrt() as usi...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 101: `let n = ((OPTIMAL_CHUNK_SIZE / std::mem::size_of::<f64>()) as f64).sqrt() as usi...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 109: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/memory_efficient_fusion_tests.rs

14 issues found:

- Line 91: `Ok(Box::new(x.sqrt()))`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 151: `fusion.add_op(square_op).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 164: `fusion.add_op(square_op).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 208: `fusion.add_op(square_op).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 209: `fusion.add_op(sqrt_op).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 215: `fusion.optimize().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 227: `fusion.add_op(square_op).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 231: `let result = fusion.apply(input).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 234: `let output = result.downcast_ref::<f64>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 242: `register_fusion::<f64>(square_op).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 253: `fusion.optimize().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 263: `fusion.add_op(square_op).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 266: `fusion.optimize().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 276: `fusion.add_op(square_op).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/memory_efficient_integration_tests.rs

15 issues found:

- Line 24: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 33: `let evaluated = evaluate(&lazy_result).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 36: `let transposed = transpose_view(&evaluated).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 39: `let diagonal = diagonal_view(&evaluated).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 43: `let masked = mask_array(evaluated.clone(), Some(mask), Some(0.0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 46: `let temp_file = NamedTempFile::new().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 53: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 56: `let loaded = disk_array.load().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 115: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 122: `let temp_file = NamedTempFile::new().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 127: `let evaluated = evaluate(&lazy_doubled).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 135: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 138: `let loaded = disk_array.load().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 179: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 182: `let doubled_masked = mask_array(result, Some(masked.mask.clone()), Some(0.0)).un...`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/memory_efficient_lazy_tests.rs

1 issues found:

- Line 48: `let result = evaluate(&lazy).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/memory_efficient_out_of_core_tests.rs

15 issues found:

- Line 16: `let temp_file = NamedTempFile::new().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 24: `let array = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 43: `let array = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 65: `let temp_file = NamedTempFile::new().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 69: `let array = OutOfCoreArray::new(&data, file_path, ChunkingStrategy::Fixed(2)).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 72: `let loaded = array.load().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 90: `let temp_file = NamedTempFile::new().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 103: `let array = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 110: `let loaded = array.load().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 136: `let array = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 159: `let array1 = OutOfCoreArray::new_temp(&data, ChunkingStrategy::Fixed(chunk_size)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 168: `OutOfCoreArray::new_temp(&data, ChunkingStrategy::NumChunks(num_chunks)).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 173: `let array3 = OutOfCoreArray::new_temp(&data, ChunkingStrategy::Auto).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 189: `let array = OutOfCoreArray::new_temp(&data, ChunkingStrategy::Fixed(5)).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 192: `let _: Vec<f64> = array.map(|_| 0.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/memory_efficient_views_tests.rs

3 issues found:

- Line 12: `let view = transpose_view(&data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 34: `let view = diagonal_view(&data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 93: `let _: ArrayView<u8, _> = scirs2_core::memory_efficient::view_as(&data).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/memory_mapped_array_tests.rs

27 issues found:

- Line 12: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 19: `let mmap = create_mmap::<f64, _, _>(&data, &file_path, AccessMode::Write, 0).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 32: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 39: `let mut mmap = create_mmap::<f32, _, _>(&data, &file_path, AccessMode::Write, 0)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 42: `mmap.flush().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 45: `let loaded = mmap.as_array::<Ix2>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 57: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 64: `let mut file = fs::File::create(&file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 71: `file.write_all(bytes).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 72: `file.flush().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 76: `let file = fs::File::open(&file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 77: `let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 88: `let dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 95: `let mut file = fs::File::create(&file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 96: `file.write_all(&data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 97: `file.flush().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 105: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 106: `let mut mmap = unsafe { memmap2::MmapMut::map_mut(&file).unwrap() };`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 114: `mmap.flush().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 117: `let file = fs::File::open(&file_path).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 118: `let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 131: `let mmap = create_temp_mmap::<f64, _, _>(&data, AccessMode::ReadWrite, 0).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 137: `let loaded = mmap.as_array::<Ix1>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 150: `assert_eq!("r".parse::<AccessMode>().unwrap(), AccessMode::ReadOnly);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 151: `assert_eq!("r+".parse::<AccessMode>().unwrap(), AccessMode::ReadWrite);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 152: `assert_eq!("w+".parse::<AccessMode>().unwrap(), AccessMode::Write);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 153: `assert_eq!("c".parse::<AccessMode>().unwrap(), AccessMode::CopyOnWrite);`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/shape_validation_tests.rs

20 issues found:

- Line 15: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 29: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 40: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 54: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 70: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 91: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 102: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 123: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 136: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 159: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 170: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 190: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 206: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 220: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 231: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 245: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 261: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 275: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 291: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 313: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/statistical_validation_tests.rs

23 issues found:

- Line 15: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 29: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 37: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 51: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 64: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 78: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 91: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 105: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 116: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 130: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 146: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 160: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 176: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 190: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 206: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 220: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 236: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 250: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 261: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 279: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 286: `let mean = values.iter().sum::<f64>() / values.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 287: `let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.l...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 288: `let std = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation

### tests/temporal_validation_tests.rs

18 issues found:

- Line 18: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 34: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 45: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 59: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 75: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 89: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 105: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 119: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 132: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 146: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 159: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 173: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 184: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 198: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 214: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 228: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 244: `let validator = Validator::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 262: `let result = validator.validate(&data, &schema).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/type_conversions_tests.rs

10 issues found:

- Line 10: `assert_eq!(42.0f64.to_numeric::<i32>().unwrap(), 42);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 11: `assert_eq!((-42.0f64).to_numeric::<i32>().unwrap(), -42);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 12: `assert_eq!(42i32.to_numeric::<f64>().unwrap(), 42.0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 13: `assert_eq!(42u8.to_numeric::<u16>().unwrap(), 42u16);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 67: `let z_rot = z1.rotate(std::f64::consts::PI / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 84: `let z2 = z1.convert_complex::<f32>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 89: `let z3 = z2.convert_complex::<f64>().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 109: `let int_values = convert::slice_to_numeric::<_, i32>(&float_values).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 122: `let converted = convert::complex_slice_to_complex::<_, f32>(&complex_values).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 131: `let complex = convert::real_to_complex::<_, f64>(&real_values).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()