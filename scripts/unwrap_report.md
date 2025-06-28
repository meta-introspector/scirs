# Unwrap() Usage Report

Total unwrap() calls and unsafe operations found: 1768

## Summary by Type

- Division without zero check - use safe_divide(): 868 occurrences
- Replace with ? operator or .ok_or(): 750 occurrences
- Mathematical operation .sqrt() without validation: 115 occurrences
- Mathematical operation .powf( without validation: 13 occurrences
- Mathematical operation .ln() without validation: 12 occurrences
- Handle array creation errors properly: 8 occurrences
- Use .get() with proper bounds checking: 2 occurrences

## Detailed Findings


### benches/acceleration_benchmarks.rs

19 issues found:

- Line 39: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 61: `let sparsity = signal_size / 64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 75: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 91: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 111: `let sparsity = signal_size / 64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 128: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 145: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 162: `let sparsity = signal_size / 64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 176: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 190: `let sparsity = signal_size / 64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 206: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 233: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 249: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 268: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 305: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 322: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 341: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 371: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 388: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()

### benches/batch_processing_benchmarks.rs

13 issues found:

- Line 12: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 57: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 80: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 103: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 134: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 162: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 186: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 210: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 235: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 247: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 277: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 309: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 322: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()

### benches/cuda_sparse_fft_benchmarks.rs

11 issues found:

- Line 19: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 36: `let normal = Normal::new(0.0, noise_level).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 55: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 72: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 89: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 108: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 126: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 144: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 162: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 180: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 198: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### benches/fft_benchmarks.rs

10 issues found:

- Line 22: `.map(|i| (2.0 * PI * 10.0 * i as f64 / size as f64).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 49: `let spectrum = rfft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 68: `let x = (i % size) as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 69: `let y = (i / size) as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 74: `let data_2d = data.into_shape_with_order((size, size)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 109: `.map(|i| (2.0 * PI * 10.0 * i as f64 / size as f64).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 139: `let x = (2.0 * PI * 10.0 * i as f64 / size as f64).sin();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 169: `let x = (i % size) as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 170: `let y = (i / size) as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 174: `let data_2d = data.into_shape_with_order((size, size)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### benches/gpu_kernel_benchmarks.rs

9 issues found:

- Line 15: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 53: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 77: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 101: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 139: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 168: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 190: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 205: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 249: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()

### benches/gpu_sparse_fft_bench.rs

11 issues found:

- Line 15: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 56: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 73: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 90: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 109: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 127: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 145: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 164: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 189: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 208: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 226: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### benches/planning_benchmarks.rs

23 issues found:

- Line 18: `let phase = 2.0 * std::f64::consts::PI * (i as f64) / (size as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 42: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 47: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 77: `let plan = planner.plan_fft(&[size], true, Default::default()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 81: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 96: `let temp_dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 107: `serialized_db_path: Some(db_path.to_str().unwrap().to_string()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 114: `let _ = planner.plan_fft(&[size], true, Default::default()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 115: `planner.save_plans().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 123: `let plan = planner.plan_fft(&[size], true, Default::default()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 127: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 146: `let temp_dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 159: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 164: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 179: `let plan = planner.plan_fft(&[size], true, Default::default()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 183: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 193: `serialized_db_path: Some(db_path.to_str().unwrap().to_string()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 200: `let _ = planner.plan_fft(&[size], true, Default::default()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 201: `planner.save_plans().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 206: `let plan = planner.plan_fft(&[size], true, Default::default()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 210: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 246: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 250: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### benches/scipy_comparison.rs

7 issues found:

- Line 42: `let t = i as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 57: `let x = i as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 58: `let y = j as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 91: `let spectrum = rfft(&real_signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 125: `let data_nd = data.into_shape_with_order(shape.as_slice()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 258: `signal = np.sin(2 * np.pi * 10 * np.arange(size) / size)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 292: `data = np.sin(2 * np.pi * (5 * x / size + 3 * y / size))`
  - **Fix**: Division without zero check - use safe_divide()

### examples/adaptive_planning_example.rs

4 issues found:

- Line 28: `let phase = 2.0 * std::f64::consts::PI * (i as f64) / 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 41: `adaptive_executor.execute(&input, &mut output).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 54: `println!("  Average time: {:?}", total_time / i as u32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 82: `total_time / iterations as u32`
  - **Fix**: Division without zero check - use safe_divide()

### examples/advanced_fft.rs

8 issues found:

- Line 18: `let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 57: `diff * fs / (2.0 * PI)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 73: `envelope.iter().sum::<f64>() / envelope.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 87: `inst_freq.iter().sum::<f64>() / inst_freq.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 203: `let freq = if i <= n_modes / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 213: `freq_magnitudes.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 255: `fft_duration.as_nanos() as f64 / rfft_duration.as_nanos() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 259: `100.0 * rfft_result.len() as f64 / fft_complex.len() as f64`
  - **Fix**: Division without zero check - use safe_divide()

### examples/advanced_planning_strategies.rs

6 issues found:

- Line 14: `array[[size / 4, size / 4]] = Complex64::new(1.0, 0.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 59: `let avg_exec_time = total_exec_time / iterations as u32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 124: `let _ = fft2(&test_array, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 128: `let avg_time = total_time / iterations as u32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 192: `let relative = time.as_secs_f64() / baseline.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 201: `let planner_guard = planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/auto_padding_example.rs

1 issues found:

- Line 58: `let speedup = time_no_pad.as_secs_f64() / time_padded.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()

### examples/auto_tuning_example.rs

7 issues found:

- Line 70: `input.push(Complex64::new(i as f64 / size as f64, 0.0));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 79: `let _ = scirs2_fft::fft(&input, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 84: `let standard_avg = standard_total / iterations;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 91: `let _ = tuner.run_optimal_fft(&input, None, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 96: `let tuned_avg = tuned_total / iterations;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 103: `let improvement = (standard_avg as f64 / tuned_avg as f64 - 1.0) * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 106: `let overhead = (tuned_avg as f64 / standard_avg as f64 - 1.0) * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/backend_example.rs

7 issues found:

- Line 33: `let spectrum = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 37: `let recovered = ifft(&spectrum, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 49: `let _ctx = BackendContext::new("rustfft").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 51: `let _ = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 84: `let result1 = fft(&vec_input, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 85: `let result2 = fft(ndarray_input.as_slice().unwrap(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 106: `let gpu_result = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/benchmark_analysis.rs

2 issues found:

- Line 117: `memory.push(item.memory_kb.unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 167: `accuracy.push(item.accuracy.unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/benchmark_simd_vs_standard.rs

19 issues found:

- Line 69: `let speedup = result.standard_time.as_secs_f64() / result.simd_time.as_secs_f64(...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 84: `let t = i as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 95: `let t = i as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 126: `let standard_time = start.elapsed() / iterations as u32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 133: `let simd_time = start.elapsed() / iterations as u32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 136: `let speedup = standard_time.as_secs_f64() / simd_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 160: `let standard_fn = || fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 161: `let simd_fn = || fft_simd(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 178: `let standard_fn = || fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 179: `let simd_fn = || fft_simd(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 201: `let array = ndarray::Array::from_shape_vec((size, size), signal.clone()).unwrap(...`
  - **Fix**: Handle array creation errors properly
- Line 202: `let result = fft2(&array, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 207: `let result = fft2_simd(&signal, Some((size, size)), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 237: `ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), signal.clone()).unwrap()...`
  - **Fix**: Handle array creation errors properly
- Line 240: `let result = fftn(&array, shape_vec, axes, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 245: `let result = fftn_simd(&signal, Some(&shape), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 273: `ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), signal.clone()).unwrap()...`
  - **Fix**: Handle array creation errors properly
- Line 276: `let result = fftn(&array, shape_vec, axes, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 281: `let result = fftn_simd(&signal, Some(&shape), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/benchmark_simple.rs

9 issues found:

- Line 20: `.map(|i| (2.0 * PI * 10.0 * i as f64 / size as f64).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 29: `let _ = fft(&complex_signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 35: `let _ = rfft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 41: `let _ = frft(&signal, 0.5, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 55: `let x = (2.0 * PI * 10.0 * i as f64 / size as f64).sin();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 63: `let _ = fft(&signal_copy, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 76: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 80: `let speedup = regular_time.as_secs_f64() / inplace_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 84: `println!("\nRegular is {:.2}x faster", 1.0 / speedup);`
  - **Fix**: Division without zero check - use safe_divide()

### examples/comprehensive_acceleration_showcase.rs

4 issues found:

- Line 124: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 204: `let sparsity = (signal.len() / 64).max(4).min(32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 227: `let sparsity = (signal.len() / 64).max(4).min(32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 340: `/ info.capabilities.power_consumption_watts`
  - **Fix**: Division without zero check - use safe_divide()

### examples/context_example.rs

12 issues found:

- Line 35: `let spectrum = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 39: `let _ = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 47: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 59: `fft(&signal, None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 61: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 73: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 88: `let spectrum = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 96: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 115: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 122: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 128: `let original_spectrum = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 130: `let context_spectrum = without_cache(|| fft(&signal, None).unwrap()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/cuda_sparse_fft_example.rs

14 issues found:

- Line 25: `let devices = get_cuda_devices().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 41: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 55: `let cpu_result = sparse_fft(&signal, 6, Some(SparseFFTAlgorithm::Sublinear), Non...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 75: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 87: `cpu_elapsed.as_secs_f64() / cuda_elapsed.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 92: `cuda_elapsed.as_secs_f64() / cpu_elapsed.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 167: `sparse_fft(&large_signal, 6, Some(SparseFFTAlgorithm::Sublinear), None).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 179: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 188: `cpu_elapsed.as_secs_f64() / cuda_elapsed.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 193: `cuda_elapsed.as_secs_f64() / cpu_elapsed.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 212: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 248: `let full_spectrum = scirs2_fft::fft(&signal_complex, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 269: `let pos = cpu_result.indices.iter().position(|&i| i == idx).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 288: `let pos = cuda_result.indices.iter().position(|&i| i == idx).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/cuda_sparse_fft_improved.rs

2 issues found:

- Line 36: `let t = i as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 86: `(result.indices.len() as f64 / n as f64) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()

### examples/cuda_spectral_flatness_example.rs

3 issues found:

- Line 13: `let angle = 2.0 * std::f64::consts::PI * (freq as f64) * (i as f64) / (n as f64)...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 156: `let speedup_cpu = duration_cpu.as_secs_f64() / duration_cuda.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 157: `let speedup_spectral = duration_spectral.as_secs_f64() / duration_cuda.as_secs_f...`
  - **Fix**: Division without zero check - use safe_divide()

### examples/czt_example.rs

1 issues found:

- Line 102: `let circle_points: Vec<f64> = (0..=100).map(|i| 2.0 * PI * i as f64 / 100.0).col...`
  - **Fix**: Division without zero check - use safe_divide()

### examples/fft_tutorial.rs

48 issues found:

- Line 59: `let spectrum = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 73: `let recovered = ifft(&spectrum, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 90: `let spectrum = rfft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 107: `let recovered = irfft(&spectrum, Some(signal.len())).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 135: `let spectrum_2d = fft2(&data.to_owned(), None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 159: `let recovered_2d = ifft2(&spectrum_2d.to_owned(), None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 177: `.map(|i| (2.0 * PI * freq * i as f64 / n as f64).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 183: `let spectrum = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 186: `let padded_spectrum = fft(&signal, Some(4 * n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 216: `let freqs_vec = freqs.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 217: `let padded_freqs_vec = padded_freqs.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 237: `.map(|i| (2.0 * PI * freq * i as f64 / n as f64).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 241: `let spectrum_no_window = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 244: `let hann_window = get_window(Window::Hann, n, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 245: `let blackman_window = get_window(Window::Blackman, n, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 261: `let spectrum_hann = fft(&windowed_signal_hann, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 262: `let spectrum_blackman = fft(&windowed_signal_blackman, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 268: `(total_energy - peak_energy) / total_energy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 276: `.max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 278: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 319: `let spectrum = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 322: `let power_spectrum: Vec<f64> = spectrum.iter().map(|c| c.norm_sqr() / (n as f64)...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 325: `let freq_axis = fftfreq(n, dt).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 329: `for i in 1..n / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 336: `peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 344: `power.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 385: `let mut spectrum = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 388: `let freq_axis = fftfreq(n, dt).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 403: `let filtered_signal = ifft(&spectrum, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 408: `let spectrum = fft(s, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 409: `let freq = fftfreq(s.len(), dt).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 415: `let power = spectrum[i].norm_sqr() / (s.len() as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 423: `10.0 * (signal_power / noise_power).log10()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 446: `let dt = 1.0 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 457: `let analytic = hilbert(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 475: `inst_freq[i] = phase_diff / (2.0 * PI * dt);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 479: `let avg_amplitude: f64 = amplitude.iter().sum::<f64>() / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 480: `let avg_freq: f64 = inst_freq.iter().sum::<f64>() / ((n - 1) as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 502: `let x = (i as f64 - n as f64 / 2.0) / (n as f64 / 8.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 503: `(-x * x / 2.0).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 510: `let _spectrum = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 517: `let frft_result = frft(&signal, alpha, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 522: `let energy_ratio = output_energy / input_energy;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 557: `let dt = 1.0 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 564: `let rate = (end_freq - start_freq) / t[n - 1];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 593: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 618: `if freq > 0.0 && freq < fs / 2.0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 624: `let mean_error = error_sum / count as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/fht_example.rs

22 issues found:

- Line 14: `(x.sin() / x) * (1.0 - x * x / 6.0 + x.powi(4) / 120.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 51: `.map(|&ri| (-ri * ri / (2.0 * sigma * sigma)).exp())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 55: `let f_transform = fht(&f, dln, mu, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 60: `println!("Transform: Should be Gaussian with σ' ≈ {}", 1.0 / sigma);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 66: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 70: `let f_recovered = ifht(&f_transform, dln, mu, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 78: `/ n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 102: `let f_transform = fht(&f, dln, mu, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 109: `let norm: f64 = f_transform.iter().map(|x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 126: `let f: Vec<f64> = r.iter().map(|&ri| ri.powf(-alpha)).collect();`
  - **Fix**: Mathematical operation .powf( without validation
- Line 129: `let f_unbiased = fht(&f, dln, mu, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 132: `let f_biased = fht(&f, dln, mu, None, Some(alpha)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 135: `let norm_unbiased: f64 = f_unbiased.iter().map(|x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 136: `let norm_biased: f64 = f_biased.iter().map(|x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 143: `norm_biased / norm_unbiased`
  - **Fix**: Division without zero check - use safe_divide()
- Line 159: `let offset = fhtoffset(dln, mu, None, Some(bias)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 175: `let f_transform = fht(&f, dln, mu, Some(offset), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 176: `let norm: f64 = f_transform.iter().map(|x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 201: `(1.0 - x) * (-x / 2.0).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 206: `let f_transform = fht(&f, dln, mu, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 217: `.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 219: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/frft_comparison.rs

33 issues found:

- Line 33: `(2.0 * PI * 5.0 * i as f64 / n as f64).sin()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 34: `+ (2.0 * PI * 12.0 * i as f64 / n as f64).sin()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 44: `let direct_orig = frft(&signal, alpha1 + alpha2, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 45: `let temp_orig = frft(&signal, alpha2, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 51: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 55: `let ratio_orig = energy_direct_orig / energy_sequential_orig;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 58: `let direct_ozaktas = frft_stable(&signal, alpha1 + alpha2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 59: `let temp_ozaktas = frft_stable(&signal, alpha2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 64: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 68: `let ratio_ozaktas = energy_direct_ozaktas / energy_sequential_ozaktas;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 71: `let direct_dft = frft_dft(&signal, alpha1 + alpha2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 72: `let temp_dft = frft_dft(&signal, alpha2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 74: `frft_dft(&temp_dft.iter().map(|&c| c.re).collect::<Vec<_>>(), alpha1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 78: `let ratio_dft = energy_direct_dft / energy_sequential_dft;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 106: `.map(|i| (-((i as f64 - n as f64 / 2.0).powi(2)) / 100.0).exp())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 118: `let result_orig = frft(&signal, alpha, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 120: `let deviation_orig = ((energy_orig - input_energy) / input_energy * 100.0).abs()...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 122: `let result_ozaktas = frft_stable(&signal, alpha).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 124: `let deviation_ozaktas = ((energy_ozaktas - input_energy) / input_energy * 100.0)...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 126: `let result_dft = frft_dft(&signal, alpha).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 128: `let deviation_dft = ((energy_dft - input_energy) / input_energy * 100.0).abs();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 150: `let signal: Vec<f64> = (0..n).map(|i| if i == n / 4 { 1.0 } else { 0.0 }).collec...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 173: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 181: `let temp = frft_stable(&result_ozaktas, alpha).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 184: `let final_ozaktas = frft_stable(&result_ozaktas, 0.0).unwrap(); // Convert to co...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 188: `let direct_orig = frft(&signal, alpha * num_iterations as f64, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 191: `let direct_ozaktas = frft_stable(&signal, alpha * num_iterations as f64).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 199: `energy_orig / energy_direct_orig`
  - **Fix**: Division without zero check - use safe_divide()
- Line 207: `energy_ozaktas / energy_direct_ozaktas`
  - **Fix**: Division without zero check - use safe_divide()
- Line 214: `.max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 216: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 221: `.max_by(|(_, a), (_, b)| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 223: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/frft_example.rs

3 issues found:

- Line 12: `.map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 26: `let t = i as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 57: `((energy1 - energy2) / energy1).abs()`
  - **Fix**: Division without zero check - use safe_divide()

### examples/gpu_memory_optimization_example.rs

10 issues found:

- Line 20: `max_memory / (1024 * 1024)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 28: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 55: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 80: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 97: `let manager = get_global_memory_manager().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 98: `let manager = manager.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 101: `manager.memory_limit() / (1024 * 1024)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 105: `manager.current_memory_usage() / 1024`
  - **Fix**: Division without zero check - use safe_divide()
- Line 118: `let result = memory_efficient_gpu_sparse_fft(&large_signal, max_memory).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 143: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()

### examples/gpu_sparse_fft_example.rs

6 issues found:

- Line 27: `let cpu_result = sparse_fft(&signal, 6, Some(SparseFFTAlgorithm::Sublinear), Non...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 58: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 117: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 153: `let full_spectrum = scirs2_fft::fft(&signal_complex, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 174: `let pos = cpu_result.indices.iter().position(|&i| i == idx).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 193: `let pos = gpu_result.indices.iter().position(|&i| i == idx).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/hartley_example.rs

1 issues found:

- Line 33: `let t = i as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/hermitian_fft2_example.rs

3 issues found:

- Line 12: `let x = i as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 13: `let y = j as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 27: `let spectrum = ihfft2(&signal.view(), None, None, Some("backward")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/hermitian_fft_example.rs

5 issues found:

- Line 10: `.map(|i| (2.0 * PI * i as f64 / n as f64).cos())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 19: `let spectrum = ihfft(&signal, None, Some("backward")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 38: `for i in 1..spectrum.len() / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 61: `let n_freq = n / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 83: `.map(|i| (2.0 * PI * i as f64 / n as f64).cos())`
  - **Fix**: Division without zero check - use safe_divide()

### examples/iterative_sparse_fft_example.rs

25 issues found:

- Line 25: `let normal = Normal::new(0.0, noise_level).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 31: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 63: `let tolerance = std::cmp::max(1, n / 1000);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 78: `let precision = true_positives as f64 / result.indices.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 79: `let recall = true_positives as f64 / true_frequencies.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 141: `scirs2_fft::sparse_fft::sparse_fft(&signal, sparsity, Some(algorithm), None).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 170: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 199: `let max_freq_to_plot = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 228: `components.sort_by(|a, b| b.1.norm().partial_cmp(&a.1.norm()).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 317: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 379: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 443: `let result = execute_cuda_iterative_sparse_fft(&signal, sparsity, iterations).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 456: `let reconstructed = reconstruct_time_domain(&result, n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 462: `error = (error / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 462: `error = (error / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 476: `if iterations == 1 || iterations == *iterations_to_test.last().unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 487: `components.sort_by(|a, b| b.1.norm().partial_cmp(&a.1.norm()).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 493: `let tolerance = std::cmp::max(1, n / 1000);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 538: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 551: `let reconstructed = reconstruct_time_domain(&sublinear_result, n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 557: `error = (error / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 557: `error = (error / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 571: `components.sort_by(|a, b| b.1.norm().partial_cmp(&a.1.norm()).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 577: `let tolerance = std::cmp::max(1, n / 1000);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 604: `let devices = get_cuda_devices().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/mdct_example.rs

13 issues found:

- Line 36: `long_signal[i] = (2.0 * PI * 440.0 * i as f64 / 8000.0).sin();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 41: `let hop_size = block_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 49: `let num_blocks = (signal_len - block_size) / hop_size + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 67: `let start_idx = block_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 68: `let end_idx = signal_len - block_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 97: `test_signal[i] = (2.0 * PI * i as f64 / 16.0).cos() + (4.0 * PI * i as f64 / 16....`
  - **Fix**: Division without zero check - use safe_divide()
- Line 126: `let t = i as f64 / sample_rate;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 132: `let hop_size = block_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 135: `let num_blocks = (signal_len - block_size) / hop_size + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 145: `let quantized: Array1<f64> = mdct_block.mapv(|x| (x * 100.0).round() / 100.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 154: `let signal_power: f64 = audio_signal.mapv(|x| x * x).sum() / signal_len as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 157: `let noise_power: f64 = noise.mapv(|x| x * x).sum() / noise.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 158: `let snr_db = 10.0 * (signal_power / noise_power).log10();`
  - **Fix**: Division without zero check - use safe_divide()

### examples/memory_efficient_fft.rs

16 issues found:

- Line 53: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 67: `let standard_fft = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 93: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 115: `let x = i as f64 / 1000.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 128: `let _result = fft_streaming(&signal, None, FftMode::Forward, Some(chunk_size)).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 142: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 161: `let standard_result = fft(subset, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 165: `let streaming_subset = fft_streaming(subset, None, FftMode::Forward, None).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 205: `let spectrum_2d = fft2_efficient(&data.view(), None, FftMode::Forward, false).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 217: `let standard_2d = scirs2_fft::fft2(&data.to_owned(), None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 236: `(spectrum_2d[[i, j]] - standard_2d[[i, j]]).norm() / mag1.max(mag2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 257: `let recovered = fft2_efficient(&spectrum_2d.view(), None, FftMode::Inverse, true...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 296: `.map(|i| (2.0 * PI * (i as f64) / 1024.0).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 301: `let _ = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 306: `let _ = fft_streaming(&signal, None, FftMode::Forward, Some(4096)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 321: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/memory_profiling_example.rs

11 issues found:

- Line 38: `(size as f64 * std::mem::size_of::<Complex64>() as f64 * 3.0) / (1024.0 * 1024.0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 42: `(size as f64 * std::mem::size_of::<Complex64>() as f64 * 1.75) / (1024.0 * 1024....`
  - **Fix**: Division without zero check - use safe_divide()
- Line 46: `(size as f64 * std::mem::size_of::<Complex64>() as f64 * 2.2) / (1024.0 * 1024.0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 50: `(size as f64 * std::mem::size_of::<f64>() as f64 * 2.0) / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 54: `(size as f64 * std::mem::size_of::<Complex64>() as f64 * 4.0) / (1024.0 * 1024.0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 58: `(size as f64 * std::mem::size_of::<Complex64>() as f64 * 3.5) / (1024.0 * 1024.0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 62: `(size as f64 * std::mem::size_of::<Complex64>() as f64 * 2.0) / (1024.0 * 1024.0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 66: `(size as f64 * std::mem::size_of::<Complex64>() as f64 * 2.3) / (1024.0 * 1024.0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 87: `.map(|i| (2.0 * PI * 10.0 * i as f64 / size as f64).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 132: `let x = i as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 133: `let y = j as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/memory_usage_benchmarking.rs

33 issues found:

- Line 46: `let _ = fft(&complex_signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 47: `let _ = optimized_fft(&complex_signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 50: `let _ = fft(&complex_signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 56: `let _ = fft(&complex_signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 64: `let _ = optimized_fft(&complex_signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 73: `let _ = fft(&complex_signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 80: `let _ = fft(&complex_signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 85: `let avg_standard = total_standard.as_secs_f64() * 1000.0 / iterations as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 86: `let avg_efficient = total_efficient.as_secs_f64() * 1000.0 / iterations as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 87: `let avg_planned = total_planned.as_secs_f64() * 1000.0 / iterations as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 92: `avg_standard / min_time`
  - **Fix**: Division without zero check - use safe_divide()
- Line 108: `first_plan_time.as_secs_f64() * 1000.0 / avg_planned`
  - **Fix**: Division without zero check - use safe_divide()
- Line 137: `let _ = fft2(&signal, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 138: `let _ = optimized_fft2(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 140: `let _ = fft2(&signal, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 146: `let _ = fft2(&signal, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 154: `let _ = optimized_fft2(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 163: `let _ = fft2(&signal, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 170: `let _ = fft2(&signal, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 175: `let avg_standard = total_standard.as_secs_f64() * 1000.0 / iterations as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 176: `let avg_efficient = total_efficient.as_secs_f64() * 1000.0 / iterations as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 177: `let avg_planned = total_planned.as_secs_f64() * 1000.0 / iterations as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 182: `avg_standard / min_time`
  - **Fix**: Division without zero check - use safe_divide()
- Line 198: `first_plan_time.as_secs_f64() * 1000.0 / avg_planned`
  - **Fix**: Division without zero check - use safe_divide()
- Line 219: `size as f64 * std::mem::size_of::<Complex64>() as f64 * 3.5 / (1024.0 * 1024.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 221: `size as f64 * std::mem::size_of::<Complex64>() as f64 * 2.0 / (1024.0 * 1024.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 222: `let reduction = 100.0 * (std_mem - eff_mem) / std_mem;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 235: `let x = i as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 247: `let x = i as f64 / rows as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 248: `let y = j as f64 / cols as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 361: `format!("{:.2} KB", size as f64 / 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 363: `format!("{:.2} MB", size as f64 / (1024.0 * 1024.0))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 365: `format!("{:.2} GB", size as f64 / (1024.0 * 1024.0 * 1024.0))`
  - **Fix**: Division without zero check - use safe_divide()

### examples/multi_gpu_device_example.rs

6 issues found:

- Line 48: `device.memory_total as f64 / (1024.0 * 1024.0 * 1024.0),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 49: `device.memory_free as f64 / (1024.0 * 1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 182: `let throughput = signal_size as f64 / elapsed.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 250: `let avg_time = times.iter().sum::<f64>() / times.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 282: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 319: `total_memory as f64 / (1024.0 * 1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()

### examples/ndim_optimized_example.rs

9 issues found:

- Line 23: `((i as f64).sin() + (j as f64).cos() + (k as f64).tan()) / 3.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 28: `let _result_std = fftn(&array.to_owned().into_dyn(), None, None, None, None, Non...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 33: `let _result_opt = fftn_optimized(&array.view(), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 37: `let speedup = time_standard.as_secs_f64() / time_optimized.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 54: `(large_array.len() * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0 * 102...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 87: `let _result = fftn_optimized(&asymmetric.view(), None, Some(axes)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 101: `let _result = fftn_optimized(&chunk_test.view(), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 128: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 129: `let result_optimized = fftn_optimized(&test_array.view(), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/optimized_kernel_example.rs

1 issues found:

- Line 67: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()

### examples/parallel_planning_example.rs

8 issues found:

- Line 37: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 49: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 66: `let results = planner.plan_multiple(&sizes).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 87: `sequential_time.as_secs_f64() / total_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 102: `executor.execute(&input, &mut output).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 130: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 145: `/ times.len() as u32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 153: `let phase = 2.0 * std::f64::consts::PI * (i as f64) / (size as f64);`
  - **Fix**: Division without zero check - use safe_divide()

### examples/parallel_planning_simple.rs

3 issues found:

- Line 46: `let results = planner.plan_multiple(&plan_specs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 71: `executor.execute(&input, &mut output).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 82: `let x = i as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/plan_cache_example.rs

4 issues found:

- Line 32: `let _ = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 39: `let _ = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 44: `let speedup = cold_duration.as_secs_f64() / warm_duration.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 82: `let _ = fft(&signal_128, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/plan_optimization_example.rs

10 issues found:

- Line 29: `plan_ahead_of_time(&common_sizes, Some("./fft_plans.json")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 37: `test_array[[n / 4, n / 4]] = Complex64::new(1.0, 0.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 42: `let result1 = fft2(&test_array, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 56: `let plan = builder.build().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 70: `executor.execute(&input_flat, &mut result2_data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 82: `let plan = builder.build().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 93: `executor.execute(&input_flat, &mut result3_data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 104: `let mut planner_guard = planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 107: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 117: `executor.execute(&input_flat, &mut result4_data).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/plan_serialization_example.rs

1 issues found:

- Line 99: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/signal_processing_example.rs

13 issues found:

- Line 37: `.map(|i| i as f64 / sample_rate)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 55: `let smoothed = convolve(&signal, &kernel).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 65: `let correlation = cross_correlate(&signal, pattern).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 73: `.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 75: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 94: `let filter_coeffs = design_fir_filter(&spec).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 102: `let filtered_signal = fir_filter(&signal, &filter_coeffs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 117: `let freq = i as f64 * sample_rate / n_freq as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 124: `let freq_filtered = frequency_filter(&signal, &filter_response).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 133: `let original_fft = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 136: `let filtered_fft = fft(&filtered_signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 141: `let freq = i as f64 * sample_rate / n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 147: `freq_magnitude.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/simd_fft2_image_processing.rs

11 issues found:

- Line 81: `let x_norm = x as f64 / width as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 82: `let y_norm = y as f64 / height as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 105: `let spectrum = fft2_adaptive(image, Some((height, width)), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 111: `let center_x = width / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 112: `let center_y = height / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 122: `let distance = ((freq_x.pow(2) + freq_y.pow(2)) as f64) / max_distance;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 132: `0.5 * (1.0 + (PI * (distance - cutoff) / cutoff).cos())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 143: `0.5 * (1.0 - (PI * (distance - cutoff) / cutoff).cos())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 155: `0.5 * (1.0 - (PI * (distance - low_cutoff) / low_cutoff).cos())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 159: `0.5 * (1.0 + (PI * (distance - high_cutoff) / high_cutoff).cos())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 180: `ifft2_adaptive(&filtered_spectrum, Some((height, width)), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/simd_fft_example.rs

12 issues found:

- Line 24: `let t = i as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 33: `let standard_fft = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 40: `let simd_fft = fft_simd(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 56: `let speedup = standard_time.as_secs_f64() / simd_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 62: `let _adaptive_fft = fft_adaptive(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 78: `let t = i as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 87: `let _ = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 93: `let _ = fft_adaptive(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 100: `let _ = fft_simd(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 105: `let speedup = standard_time.as_secs_f64() / simd_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 111: `let _ = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 117: `let _ = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/simd_fftn_volumetric_data.rs

18 issues found:

- Line 47: `let mean_diff = sum_diff / total_voxels as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 67: `let ops_per_sec = test_total as f64 / time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 79: `let x_norm = x as f64 / width as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 80: `let y_norm = y as f64 / height as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 81: `let z_norm = z as f64 / depth as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 114: `let spectrum = fftn_adaptive(volume, Some(shape), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 120: `let center_x = shape[0] / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 121: `let center_y = shape[1] / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 122: `let center_z = shape[2] / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 150: `((freq_x.pow(2) + freq_y.pow(2) + freq_z.pow(2)) as f64) / max_distance;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 161: `0.5 * (1.0 + (PI * (distance - cutoff) / cutoff).cos())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 173: `0.5 * (1.0 - (PI * (distance - cutoff) / cutoff).cos())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 186: `0.5 * (1.0 - (PI * (distance - low_cutoff) / low_cutoff).cos())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 191: `0.5 * (1.0 + (PI * (distance - high_cutoff) / high_cutoff).cos())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 215: `ifftn_adaptive(&filtered_spectrum, Some(shape), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 244: `for z in [0, shape[2] / 2, shape[2] - 1] {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 245: `for y in [0, shape[1] / 2, shape[1] - 1] {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 246: `for x in [0, shape[0] / 2, shape[0] - 1] {`
  - **Fix**: Division without zero check - use safe_divide()

### examples/simd_rfft_example.rs

10 issues found:

- Line 48: `let speedup = standard_time.as_secs_f64() / simd_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 51: `let slowdown = simd_time.as_secs_f64() / standard_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 102: `let t = i as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 115: `let spectrum = rfft(signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 124: `let spectrum = rfft_adaptive(signal, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 137: `.map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 153: `.max_by(|a, b| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 154: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 161: `peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 166: `let freq = *bin as f64 * n as f64 / spectrum.len() as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/sparse_fft_algorithm_comparison.rs

19 issues found:

- Line 21: `let normal = Normal::new(0.0, noise_level).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 27: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 55: `(detected_freq as f64 - true_freq as f64).abs() / (true_freq as f64).max(1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 70: `min_error_sum / found_count as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 113: `(size / 100, 1.0),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 114: `(size / 40, 0.5),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 115: `(size / 20, 0.25),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 116: `(size / 10, 0.15),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 117: `(size / 5, 0.1),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 133: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 158: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 227: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 346: `(size / 100, 1.0),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 347: `(size / 40, 0.5),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 348: `(size / 20, 0.25),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 349: `(size / 10, 0.15),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 350: `(size / 5, 0.1),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 367: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 426: `let devices = get_cuda_devices().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/sparse_fft_batch_processing.rs

13 issues found:

- Line 18: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 100: `sequential_time.as_secs_f64() / cpu_batch_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 143: `sequential_time.as_secs_f64() / gpu_batch_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 147: `cpu_batch_time.as_secs_f64() / gpu_batch_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 234: `100.0 * match_count as f64 / total as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 240: `100.0 * partial_match_count as f64 / total as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 244: `100.0 * (match_count + partial_match_count) as f64 / total as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 259: `/ sequential_results.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 264: `/ parallel_results.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 269: `/ spectral_results.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 281: `/ sequential_results.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 286: `/ parallel_results.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 291: `/ spectral_results.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/sparse_fft_example.rs

22 issues found:

- Line 25: `let full_fft_result = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 41: `let sparse_result = sparse_fft(&signal, 6, Some(alg), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 87: `let reconstructed_spectrum = reconstruct_spectrum(&sparse_result, n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 90: `let reconstructed_signal = scirs2_fft::ifft(&reconstructed_spectrum, None).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 103: `let adaptive_result = adaptive_sparse_fft(&signal, 0.1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 115: `let pruning_result = frequency_pruning_sparse_fft(&signal, 2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 158: `let reconstructed_spectrum = reconstruct_spectrum(&pruning_result, n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 159: `let reconstructed_signal = scirs2_fft::ifft(&reconstructed_spectrum, None).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 171: `noisy_signal[i] += 0.05 * ((i % 64) as f64 / 64.0 - 0.5);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 175: `let flatness_result = spectral_flatness_sparse_fft(&noisy_signal, 0.3, 32).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 218: `let reconstructed_spectrum = reconstruct_spectrum(&flatness_result, n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 219: `let reconstructed_signal = scirs2_fft::ifft(&reconstructed_spectrum, None).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 250: `sparse_fft2(&signal_2d_matrix, 8, Some(SparseFFTAlgorithm::Sublinear)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 273: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 288: `let x = 2.0 * PI * (i as f64) / (rows as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 289: `let y = 2.0 * PI * (j as f64) / (cols as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 312: `1.0 / orig_energy.sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 312: `1.0 / orig_energy.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 317: `1.0 / recon_energy.sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 317: `1.0 / recon_energy.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 332: `(error_sum / (2.0 * len as f64)).sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 332: `(error_sum / (2.0 * len as f64)).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation

### examples/sparse_fft_performance_visualization.rs

14 issues found:

- Line 19: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 74: `(size / 100, 1.0),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 75: `(size / 50, 0.5),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 76: `(size / 20, 0.25),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 77: `(size / 10, 0.15),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 78: `(size / 5, 0.1),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 90: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 104: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 110: `cpu_time / gpu_time`
  - **Fix**: Division without zero check - use safe_divide()
- Line 141: `let size_labels: Vec<String> = sizes.iter().map(|&s| format!("{}K", s / 1024)).c...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 244: `let normal = Normal::new(0.0, noise_level).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 258: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 273: `let accuracy = found_count as f64 / frequencies.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 326: `let devices = get_cuda_devices().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/sparse_reconstruction_example.rs

15 issues found:

- Line 41: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 52: `let reconstructed = reconstruct_time_domain(&sparse_result, n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 72: `let high_res = reconstruct_high_resolution(&sparse_result, n, target_length).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 82: `let nyquist = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 96: `let lowpass_signal = reconstruct_filtered(&sparse_result, n, lowpass).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 103: `let nyquist = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 118: `let bandpass_signal = reconstruct_filtered(&sparse_result, n, bandpass).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 138: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 165: `let orig_scale = 1.0 / orig_energy.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 165: `let orig_scale = 1.0 / orig_energy.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 166: `let recon_scale = 1.0 / recon_energy.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 166: `let recon_scale = 1.0 / recon_energy.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 178: `(error_sum / (2.0 * original.len() as f64)).sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 178: `(error_sum / (2.0 * original.len() as f64)).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 266: `.map(|i| hires_slice_start as f64 + i as f64 / 2.0)`
  - **Fix**: Division without zero check - use safe_divide()

### examples/specialized_hardware_example.rs

4 issues found:

- Line 131: `let throughput = signal_size as f64 / elapsed.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 172: `let throughput = signal_size as f64 / elapsed.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 213: `let efficiency = throughput / power;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 268: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()

### examples/spectral_analysis_simd.rs

26 issues found:

- Line 62: `let energy_ratio = filtered_energy / original_energy;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 105: `let ops_per_sec = size as f64 * (size as f64).log2() / elapsed.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 112: `let step = (end - start) / (num - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 140: `let t = i as f64 / 1000.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 163: `let normal = (-2.0 * x.ln()).sqrt() * (2.0 * PI * y).cos();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 163: `let normal = (-2.0 * x.ln()).sqrt() * (2.0 * PI * y).cos();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 170: `let window_func = window::get_window(window::Window::Hann, signal.len(), true).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 179: `let spectrum = fft_adaptive(&windowed_signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 182: `let freqs = fftfreq(signal.len(), 1.0 / sample_rate).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 188: `let half_len = signal.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 206: `.map(|c| (c.re * c.re + c.im * c.im) / signal.len() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 229: `peaks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 242: `let mut spectrum = fft_adaptive(signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 245: `let freq_resolution = sample_rate / signal.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 250: `let freq = if i <= spectrum.len() / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 263: `let filtered_signal = ifft_adaptive(&spectrum, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 280: `let num_frames = (signal.len() - overlap) / hop_size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 283: `let window_func = window::get_window(window::Window::Hann, window_size, true).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 286: `let mut spectrogram = Vec::with_capacity(num_frames * (window_size / 2 + 1));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 304: `let spectrum = fft_adaptive(&windowed_frame, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 307: `for i in 0..=window_size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 308: `let power = spectrum[i].norm_sqr() / window_size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 315: `.map(|i| i as f64 * hop_size as f64 / sample_rate)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 319: `let nyquist = sample_rate / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 320: `let freq_bins = (0..=window_size / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 321: `.map(|i| i as f64 * nyquist / (window_size / 2) as f64)`
  - **Fix**: Division without zero check - use safe_divide()

### examples/spectrogram/spectral_analysis.rs

7 issues found:

- Line 21: `let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 30: `let progress = i as f64 / n_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 62: `let freqs = fftfreq(n_samples, 1.0 / fs);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 79: `for (i, &freq) in freqs.iter().enumerate().take(n_samples / 2) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 98: `.map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 120: `inst_freq.push(phase_diff * fs / (2.0 * PI));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 198: `let enbw = window.iter().map(|x| x.powi(2)).sum::<f64>() /`
  - **Fix**: Division without zero check - use safe_divide()

### examples/spectrogram/visualization.rs

10 issues found:

- Line 23: `let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 30: `let freq = 10.0 + (200.0 - 10.0) * (ti / t_max);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 31: `let phase = 2.0 * PI * (10.0 * ti + 0.5 * (200.0 - 10.0) / t_max * ti.powi(2));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 37: `let ti = i as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 46: `let window = 0.5 * (1.0 - ((ti - 5.0) * PI / 2.0).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 107: `let test_amplitudes: Vec<f64> = (0..n_test).map(|i| i as f64 / (n_test as f64 - ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 162: `let quarter_idx = times.len() / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 163: `let mid_idx = times.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 164: `let three_quarter_idx = 3 * times.len() / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 187: `freq_powers.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/strided_fft_example.rs

7 issues found:

- Line 18: `arr[[i, j]] = (i * j) as f64 / 1000.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 35: `let result_strided = fft_strided(&arr, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 54: `let result_strided = fft_strided(&arr, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 74: `let fwd = fft_strided_complex(&complex_arr, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 77: `let inv = ifft_strided(&fwd, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 99: `let fft_result = fft(&column, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 118: `let fft_result = fft(&row, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/worker_pool_example.rs

3 issues found:

- Line 39: `fft2(&signal.to_owned(), None, None, None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 53: `let _result = fft2(&signal.to_owned(), None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 74: `fft2(&signal.to_owned(), None, None, None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()

### src/arm_fft_test.rs

16 issues found:

- Line 22: `.map(|i| (2.0 * PI * i as f64 / 128.0).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 30: `let magnitude = |c: &Complex64| -> f64 { (c.re.powi(2) + c.im.powi(2)).sqrt() };`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 35: `if magnitude(&spectrum[i]) > n as f64 / 4.0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 79: `let x = i as f64 / n_rows as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 80: `let y = j as f64 / n_cols as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 91: `let magnitude = |c: &Complex64| -> f64 { (c.re.powi(2) + c.im.powi(2)).sqrt() };`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 101: `if magnitude(&spectrum[[i, j]]) > (n_rows * n_cols) as f64 / 8.0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 130: `let x = i as f64 / shape[0] as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 131: `let y = j as f64 / shape[1] as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 132: `let z = k as f64 / shape[2] as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 157: `.map(|i| (2.0 * PI * i as f64 / 256.0).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 176: `let magnitude = |c: &Complex64| -> f64 { (c.re.powi(2) + c.im.powi(2)).sqrt() };`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 181: `if magnitude(&output[i]) > n as f64 / 4.0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 213: `.map(|i| Complex64::new((2.0 * PI * i as f64 / 512.0).sin(), 0.0))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 228: `let magnitude = |c: &Complex64| -> f64 { (c.re.powi(2) + c.im.powi(2)).sqrt() };`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 233: `if magnitude(&spectrum[i]) > n as f64 / 4.0 {`
  - **Fix**: Division without zero check - use safe_divide()

### src/auto_tuning.rs

5 issues found:

- Line 416: `let avg_time = times.iter().sum::<u64>() / times.len() as u64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 427: `/ times.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 428: `let std_dev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 573: `let scale = 1.0 / (actual_size as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 690: `let temp_dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/backend.rs

13 issues found:

- Line 103: `let mut planner = self.planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 136: `let mut planner = self.planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 149: `let scale = 1.0 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 185: `let mut backends = self.backends.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 198: `let backends = self.backends.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 204: `let backends = self.backends.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 222: `*self.current_backend.lock().unwrap() = name.to_string();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 228: `self.current_backend.lock().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 233: `let current_name = self.current_backend.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 234: `let backends = self.backends.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 243: `let backends = self.backends.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 370: `let info = manager.get_backend_info("rustfft").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 380: `let _ctx = BackendContext::new("rustfft").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/bin/accuracy_comparison.rs

36 issues found:

- Line 42: `let mean_error = sum_error / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 43: `let rms_error = (sum_squared_error / n).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 43: `let rms_error = (sum_squared_error / n).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 45: `sum_error / sum_magnitude`
  - **Fix**: Division without zero check - use safe_divide()
- Line 71: `let mean_error = sum_error / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 72: `let rms_error = (sum_squared_error / n).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 72: `let rms_error = (sum_squared_error / n).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 74: `sum_error / sum_magnitude`
  - **Fix**: Division without zero check - use safe_divide()
- Line 91: `let t = i as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 96: `let spectrum = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 104: `expected[freq_index] = Complex64::new(0.0, -(size as f64) / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 105: `expected[conj_index] = Complex64::new(0.0, (size as f64) / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 125: `let t = i as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 130: `let spectrum = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 136: `let freq_energy: f64 = spectrum.iter().map(|x| x.norm_sqr()).sum::<f64>() / size...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 139: `let relative_error = energy_error / time_energy;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 170: `let spectrum = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 171: `let reconstructed = ifft(&spectrum, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 198: `let t = i as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 204: `let spectrum = rfft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 207: `let reconstructed = irfft(&spectrum, Some(size)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 233: `let x = i as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 234: `let y = j as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 242: `let spectrum = fft2(&complex_signal.to_owned(), None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 246: `expected[(3, 2)] = Complex64::new(0.0, -(size as f64) * (size as f64) / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 247: `expected[(size - 3, size - 2)] = Complex64::new(0.0, (size as f64) * (size as f6...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 279: `let t = i as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 284: `let dct_result = dct(&signal, Some(scirs2_fft::DCTType::Type2), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 290: `.max_by(|a, b| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 291: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 295: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 305: `relative_error: frequency_error / frequency,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 321: `let t = i as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 331: `let frft1 = frft(&signal, alpha1, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 332: `let frft2 = frft_complex(&frft1, alpha2, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 335: `let frft_direct = frft(&signal, alpha_sum, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/context.rs

5 issues found:

- Line 227: `let context = builder.build().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 239: `assert_eq!(result.unwrap(), 42);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 249: `assert_eq!(result.unwrap(), 84);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 259: `assert_eq!(result.unwrap(), 168);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 275: `assert_eq!(result.unwrap(), 336);`
  - **Fix**: Replace with ? operator or .ok_or()

### src/czt.rs

20 issues found:

- Line 34: `k.mapv(|ki| a * w.powf(-ki))`
  - **Fix**: Mathematical operation .powf( without validation
- Line 37: `k.mapv(|ki| a * (Complex::new(0.0, 2.0 * PI * ki / m as f64)).exp())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 85: `let wk2 = k.mapv(|ki| w.powf(ki * ki / 2.0));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 85: `let wk2 = k.mapv(|ki| w.powf(ki * ki / 2.0));`
  - **Fix**: Mathematical operation .powf( without validation
- Line 89: `let w = (-2.0 * PI * Complex::<f64>::i() / m as f64).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 92: `let phase = -(PI * ((ki_i64 * ki_i64) % (2 * m as i64)) as f64) / m as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 102: `let awk2: Array1<Complex<f64>> = (0..n).map(|k| a.powf(-(k as f64)) * wk2[k]).co...`
  - **Fix**: Mathematical operation .powf( without validation
- Line 109: `chirp_vec[n - 1 - i] = Complex::new(1.0, 0.0) / wk2[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 112: `chirp_vec[n - 1 + i] = Complex::new(1.0, 0.0) / wk2[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 184: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 186: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 197: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 344: `let step = (k1_float - k0_float) / (m - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 346: `let phi = 2.0 * PI * k0_float / (n as f64 * oversampling);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 349: `let theta = -2.0 * PI * step / (n as f64 * oversampling);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 385: `let czt_result = czt(&x.view(), None, None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 389: `let czt_result_1d: Array1<Complex<f64>> = czt_result.into_dimensionality().unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 391: `let fft_result_vec = crate::fft::fft(&x.to_vec(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 413: `let zoom_result = zoom_fft(&x.view(), m, 0.0, 0.5, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 417: `let zoom_result_1d: Array1<Complex<f64>> = zoom_result.into_dimensionality().unw...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/dct.rs

52 issues found:

- Line 442: `let angle = PI * k_f * i_f / (n - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 457: `let norm_factor = (2.0 / (n - 1) as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 457: `let norm_factor = (2.0 / (n - 1) as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 458: `let endpoints_factor = 1.0 / 2.0_f64.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 458: `let endpoints_factor = 1.0 / 2.0_f64.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 491: `let norm_factor = ((n - 1) as f64 / 2.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 491: `let norm_factor = ((n - 1) as f64 / 2.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 492: `let endpoints_factor = 2.0_f64.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 511: `let angle = PI * k_f * i_f / (n - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 515: `sum *= 2.0 / (n - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 540: `let angle = PI * (i_f + 0.5) * k_f / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 550: `let norm_factor = (2.0 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 550: `let norm_factor = (2.0 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 551: `let first_factor = 1.0 / 2.0_f64.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 551: `let first_factor = 1.0 / 2.0_f64.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 576: `let norm_factor = (n as f64 / 2.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 576: `let norm_factor = (n as f64 / 2.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 577: `let first_factor = 2.0_f64.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 593: `let angle = PI * k_f * (i_f + 0.5) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 597: `sum *= 2.0 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 618: `let norm_factor = (n as f64 / 2.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 618: `let norm_factor = (n as f64 / 2.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 619: `let first_factor = 1.0 / 2.0_f64.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 619: `let first_factor = 1.0 / 2.0_f64.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 635: `let angle = PI * i_f * (k_f + 0.5) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 639: `sum *= 2.0 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 660: `let norm_factor = (2.0 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 660: `let norm_factor = (2.0 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 661: `let first_factor = 2.0_f64.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 677: `let angle = PI * (i_f + 0.5) * k_f / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 705: `let angle = PI * (i_f + 0.5) * (k_f + 0.5) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 714: `let norm_factor = (2.0 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 714: `let norm_factor = (2.0 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 737: `let norm_factor = (n as f64 / 2.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 737: `let norm_factor = (n as f64 / 2.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 744: `*val *= 2.0 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 763: `let dct_coeffs = dct(&signal, Some(DCTType::Type2), Some("ortho")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 766: `let recovered = idct(&dct_coeffs, Some(DCTType::Type2), Some("ortho")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 780: `let dct1_coeffs = dct(&signal, Some(DCTType::Type1), Some("ortho")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 781: `let recovered = idct(&dct1_coeffs, Some(DCTType::Type1), Some("ortho")).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 787: `let dct2_coeffs = dct(&signal, Some(DCTType::Type2), Some("ortho")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 788: `let recovered = idct(&dct2_coeffs, Some(DCTType::Type2), Some("ortho")).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 794: `let dct3_coeffs = dct(&signal, Some(DCTType::Type3), Some("ortho")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 801: `let recovered = idct(&dct3_coeffs, Some(DCTType::Type3), Some("ortho")).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 808: `let recovered = idct(&dct3_coeffs, Some(DCTType::Type3), Some("ortho")).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 815: `let dct4_coeffs = dct(&signal, Some(DCTType::Type4), Some("ortho")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 819: `let recovered = idct(&dct4_coeffs, Some(DCTType::Type4), Some("ortho")).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 821: `let original_ratio = signal[3] / signal[0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 824: `let recovered = idct(&dct4_coeffs, Some(DCTType::Type4), Some("ortho")).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 837: `let dct2_coeffs = dct2(&arr.view(), Some(DCTType::Type2), Some("ortho")).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 840: `let recovered = idct2(&dct2_coeffs.view(), Some(DCTType::Type2), Some("ortho"))....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 856: `let dct_coeffs = dct(&signal, Some(DCTType::Type2), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/distributed.rs

19 issues found:

- Line 299: `let flat_output = output.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 308: `let flat_output = output.as_slice_mut().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 312: `flat_output[i] = *exchanged_data.iter().nth(i).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 467: `let my_row = self.config.rank / p2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 581: `let my_plane = self.config.rank / (p2 * p3);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 583: `let my_row = remainder / p3;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 672: `let sqrt_nodes = (self.config.node_count as f64).sqrt().floor() as usize;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 673: `config.process_grid = vec![sqrt_nodes, self.config.node_count / sqrt_nodes];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 689: `let remaining = self.config.node_count / cbrt_nodes;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 690: `let sqrt_remaining = (remaining as f64).sqrt().floor() as usize;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 691: `config.process_grid = vec![cbrt_nodes, sqrt_remaining, remaining / sqrt_remainin...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 875: `assert_eq!(result.unwrap().len(), 2);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 884: `assert_eq!(result.unwrap(), data);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 909: `let local_data = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 929: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 934: `let local_data = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 955: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 960: `let local_data = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 997: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()

### src/dst.rs

41 issues found:

- Line 236: `return Ok(Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap());`
  - **Fix**: Handle array creation errors properly
- Line 426: `let angle = PI * k_f * m_f / (n as f64 + 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 435: `let norm_factor = (2.0 / (n as f64 + 1.0)).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 435: `let norm_factor = (2.0 / (n as f64 + 1.0)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 442: `*val *= 2.0 / (n as f64 + 1.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 442: `*val *= 2.0 / (n as f64 + 1.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 468: `let norm_factor = (n as f64 + 1.0).sqrt() / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 468: `let norm_factor = (n as f64 + 1.0).sqrt() / 2.0;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 475: `*val *= (n as f64 + 1.0).sqrt() / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 475: `*val *= (n as f64 + 1.0).sqrt() / 2.0;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 501: `let angle = PI * k_f * (m_f + 0.5) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 510: `let norm_factor = (2.0 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 510: `let norm_factor = (2.0 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 538: `let norm_factor = (n as f64 / 2.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 538: `let norm_factor = (n as f64 / 2.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 572: `let angle = PI * m_f * (k_f + 0.5) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 581: `let norm_factor = (2.0 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 581: `let norm_factor = (2.0 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 583: `*val *= norm_factor / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 614: `let norm_factor = (n as f64 / 2.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 614: `let norm_factor = (n as f64 / 2.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 647: `let angle = PI * (m_f + 0.5) * (k_f + 0.5) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 656: `let norm_factor = (2.0 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 656: `let norm_factor = (2.0 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 689: `let norm_factor = (n as f64 / 2.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 689: `let norm_factor = (n as f64 / 2.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 696: `*val *= 1.0 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 716: `let dst_coeffs = dst(&signal, Some(DSTType::Type2), Some("ortho")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 719: `let recovered = idst(&dst_coeffs, Some(DSTType::Type2), Some("ortho")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 733: `let dst1_coeffs = dst(&signal, Some(DSTType::Type1), Some("ortho")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 734: `let recovered = idst(&dst1_coeffs, Some(DSTType::Type1), Some("ortho")).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 740: `let dst2_coeffs = dst(&signal, Some(DSTType::Type2), Some("ortho")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 741: `let recovered = idst(&dst2_coeffs, Some(DSTType::Type2), Some("ortho")).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 747: `let dst3_coeffs = dst(&signal, Some(DSTType::Type3), Some("ortho")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 748: `let recovered = idst(&dst3_coeffs, Some(DSTType::Type3), Some("ortho")).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 754: `let dst4_coeffs = dst(&signal, Some(DSTType::Type4), Some("ortho")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 755: `let recovered = idst(&dst4_coeffs, Some(DSTType::Type4), Some("ortho")).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 767: `let dst2_coeffs = dst2(&arr.view(), Some(DSTType::Type2), Some("ortho")).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 770: `let recovered = idst2(&dst2_coeffs.view(), Some(DSTType::Type2), Some("ortho"))....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 786: `let dst2_coeffs = dst(&signal, Some(DSTType::Type2), Some("ortho")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 789: `let recovered = idst(&dst2_coeffs, Some(DSTType::Type2), Some("ortho")).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/example_utils.rs

1 issues found:

- Line 8: `let flat_view = arr.view().into_shape(arr.len()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/fft/algorithms.rs

18 issues found:

- Line 56: `let scale = 1.0 / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 60: `let scale = 1.0 / (n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 60: `let scale = 1.0 / (n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 64: `let scale = 1.0 / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 388: `NormMode::Backward => 1.0 / (total_elements as f64),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 389: `NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 389: `NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 390: `NormMode::Forward => 1.0 / (total_elements as f64),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 530: `NormMode::Backward => 1.0 / (total_elements as f64),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 531: `NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 531: `NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 696: `NormMode::Backward => 1.0 / (total_elements as f64),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 697: `NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 697: `NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 698: `NormMode::Forward => 1.0 / (total_elements as f64),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 878: `NormMode::Backward => 1.0 / (total_elements as f64),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 879: `NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 879: `NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/fft/planning.rs

7 issues found:

- Line 191: `NormMode::Backward => 1.0 / (total_elements as f64),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 192: `NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 192: `NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 193: `NormMode::Forward => 1.0 / (total_elements as f64),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 376: `NormMode::Backward => 1.0 / (total_elements as f64),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 377: `NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 377: `NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/fft/windowing.rs

19 issues found:

- Line 81: `let x = i as f64 / (n - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 88: `let x = i as f64 / (n - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 95: `let x = i as f64 / (n - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 102: `let x = i as f64 / (n - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 110: `let x = i as f64 / (n - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 120: `let x = i as f64 / (n - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 127: `let x = i as f64 / (n - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 139: `let x = i as f64 / (n - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 144: `let alpha_n = alpha * (n - 1.0) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 148: `*w = 0.5 * (1.0 - (PI * x / alpha_n).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 152: `*w = 0.5 * (1.0 - (PI * (x - (n - 1.0) + alpha_n) / alpha_n).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 165: `term *= (x / 2.0).powi(2) / (k_f * k_f);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 176: `let x = 2.0 * i as f64 / (n - 1.0) - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 177: `let arg = beta * (1.0 - x * x).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 178: `*w = bessel_i0(arg) / beta_i0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 183: `let center = (n - 1.0) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 186: `*w = (-0.5 * (x / (sigma * center)).powi(2)).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 261: `let processing_gain = coherent_gain.powi(2) / (sum_squared / n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 262: `let equivalent_noise_bandwidth = n as f64 * sum_squared / (sum * sum);`
  - **Fix**: Division without zero check - use safe_divide()

### src/fht.rs

14 issues found:

- Line 124: `let m = i as f64 - n as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 125: `let k = 2.0 * PI * m / (n as f64 * dln);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 128: `let basic_coeff = k.powf(mu) * (-(k * k) / 4.0).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 128: `let basic_coeff = k.powf(mu) * (-(k * k) / 4.0).exp();`
  - **Fix**: Mathematical operation .powf( without validation
- Line 132: `basic_coeff * (1.0 + bias * k * k).powf(-bias / 2.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 132: `basic_coeff * (1.0 + bias * k * k).powf(-bias / 2.0)`
  - **Fix**: Mathematical operation .powf( without validation
- Line 161: `.map(|i| ((i as f64 - n as f64 / 2.0) * dln + offset).exp())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 178: `.map(|i| ((i as f64 - n as f64 / 2.0) * dln).exp())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 182: `let y = fht(&x, dln, mu, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 186: `let x_recovered = ifht(&y, dln, mu, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 196: `let offset1 = fhtoffset(dln, mu, None, Some(0.0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 200: `let offset2 = fhtoffset(dln, mu, Some(0.5), Some(1.0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 215: `let ratio = points[i] / points[i - 1];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 216: `assert_relative_eq!(ratio.ln(), dln, epsilon = 1e-10);`
  - **Fix**: Mathematical operation .ln() without validation

### src/frft.rs

42 issues found:

- Line 171: `let cot_alpha = 1.0 / alpha.tan();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 172: `let scale = (1.0 - Complex64::i() * cot_alpha).sqrt() / (2.0 * PI).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 172: `let scale = (1.0 - Complex64::i() * cot_alpha).sqrt() / (2.0 * PI).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 177: `padded[i + n / 2] = x[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 183: `let t = (i as f64 - n_padded as f64 / 2.0) * d;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 194: `let u = (i as f64 - n as f64 / 2.0) * 2.0 * PI / (n_padded as f64 * d);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 197: `let idx = (i + n_padded / 4) % n_padded;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 212: `(0.0, 0.5 * PI, alpha / (0.5 * PI))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 215: `(0.5 * PI, PI, (alpha - 0.5 * PI) / (0.5 * PI))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 218: `let base = (alpha / PI).floor() * PI;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 219: `(base, base + 0.5 * PI, (alpha - base) / (0.5 * PI))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 337: `let alpha = alpha * PI / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 417: `let result = frft(&signal, 0.0, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 429: `let frft_result = frft(&signal, 1.0, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 430: `let fft_result = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 442: `let result = frft(&signal, 2.0, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 462: `let frft_result = frft_complex(&signal_vec, 3.0, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 463: `let ifft_result = ifft(&signal_vec, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 480: `.map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 493: `let orig_result1 = frft_complex(&signal_complex, alpha1 + alpha2, None).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 494: `let orig_temp = frft_complex(&signal_complex, alpha2, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 495: `let orig_result2 = frft_complex(&orig_temp, alpha1, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 500: `let orig_energy_ratio = orig_energy1 / orig_energy2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 510: `let ozaktas_result1 = frft_stable(&signal, alpha1 + alpha2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 511: `let ozaktas_temp = frft_stable(&signal, alpha2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 515: `let ozaktas_result2 = frft_stable(&real_temp, alpha1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 520: `let ozaktas_energy_ratio = ozaktas_energy1 / ozaktas_energy2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 535: `let dft_result1 = frft_dft(&signal, alpha1 + alpha2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 536: `let dft_temp = frft_dft(&signal, alpha2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 540: `let dft_result2 = frft_dft(&dft_real_temp, alpha1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 545: `let dft_energy_ratio = dft_energy1 / dft_energy2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 574: `.map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 577: `.map(|i| (2.0 * PI * 10.0 * i as f64 / n as f64).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 591: `let frft1 = frft_complex(&signal1_complex, alpha, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 592: `let frft2 = frft_complex(&signal2_complex, alpha, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 605: `let combined2 = frft_complex(&combined_signal, alpha, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 610: `for i in n / 4..3 * n / 4 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 615: `let relative_error = ((norm1 - norm2) / norm1).abs();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 634: `let t = i as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 639: `let result = frft_complex(&signal_complex, 0.5, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 645: `let result2 = frft_complex(&result, 0.5, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 649: `let result4 = frft_complex(&signal_complex, 4.0, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/frft_dft.rs

12 issues found:

- Line 46: `let _angle = alpha * PI / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 87: `let x = (j as f64 - n_f64 / 2.0) / (n_f64 / 4.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 87: `let x = (j as f64 - n_f64 / 2.0) / (n_f64 / 4.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 89: `let phase = Complex64::new(0.0, -PI * j as f64 * k as f64 / n_f64).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 99: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 152: `let gaussian = (-x * x / 2.0).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 164: `let result = frft_dft(&signal, 0.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 179: `let result = frft_dft(&signal, *alpha).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 188: `let result = frft_dft(&signal, *alpha).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 192: `let ratio = output_energy / input_energy;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 204: `let result = frft_dft(&signal, *alpha).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 207: `let ratio = output_energy / input_energy;`
  - **Fix**: Division without zero check - use safe_divide()

### src/frft_ozaktas.rs

17 issues found:

- Line 44: `let phi = alpha * PI / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 54: `let tan_phi_2 = (phi / 2.0).tan();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 57: `let scale = (1.0 - sin_phi).abs().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 107: `let k_centered = k as f64 - n_f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 109: `let arg = PI * param * k_centered * k_centered / n_f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 124: `let ratio = i as f64 / taper_len as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 168: `let k = (phi / PI).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 190: `let phase = alpha * PI / 4.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 208: `let result = frft_ozaktas(&signal, 0.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 219: `let frft_result = frft_ozaktas(&signal, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 227: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 235: `let scale = (fft_norm / frft_norm).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 235: `let scale = (fft_norm / frft_norm).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 251: `let direct = frft_ozaktas(&signal, alpha1 + alpha2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 254: `let intermediate = frft_ozaktas(&signal, alpha1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 259: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 267: `let energy_ratio = direct_energy / sequential_energy;`
  - **Fix**: Division without zero check - use safe_divide()

### src/hartley.rs

5 issues found:

- Line 106: `let scale = 1.0 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 219: `let h = dht(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 222: `let x_recovered = idht(&h).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 238: `let h = dht(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 249: `let h = dht2(&x, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/helper.rs

34 issues found:

- Line 39: `let val = 1.0 / (n as f64 * d);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 43: `for i in 0..n / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 47: `for i in 1..n / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 48: `freq.push((-((n / 2 - i) as i64) as f64) * val);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 56: `1.0 / 7.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 57: `2.0 / 7.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 58: `-3.0 / 7.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 59: `-2.0 / 7.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 60: `-1.0 / 7.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 67: `for i in 0..=(n - 1) / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 70: `for i in 1..=(n - 1) / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 71: `let idx = (n - 1) / 2 - i + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 107: `let val = 1.0 / (n as f64 * d);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 108: `let results = (0..=n / 2).map(|i| i as f64 * val).collect::<Vec<_>>();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 234: `fftfreq(n, 1.0 / fs)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 378: `let freq = fftfreq(8, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 386: `let freq = fftfreq(7, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 403: `let freq = fftfreq(4, 0.1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 413: `let freq = rfftfreq(8, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 421: `let freq = rfftfreq(7, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 429: `let freq = rfftfreq(4, 0.1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 440: `let shifted = fftshift(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 446: `let shifted = fftshift(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 451: `let x = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 2.0, 3.0]).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 452: `let shifted = fftshift(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 453: `let expected = Array2::from_shape_vec((2, 2), vec![3.0, 2.0, 1.0, 0.0]).unwrap()...`
  - **Fix**: Handle array creation errors properly
- Line 461: `let shifted = fftshift(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 462: `let unshifted = ifftshift(&shifted).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 467: `let shifted = fftshift(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 468: `let unshifted = ifftshift(&shifted).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 472: `let x = Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwra...`
  - **Fix**: Handle array creation errors properly
- Line 473: `let shifted = fftshift(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 474: `let unshifted = ifftshift(&shifted).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 480: `let bins = freq_bins(8, 16000.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/hfft/complex_to_real.rs

1 issues found:

- Line 469: `*idx_val = (i / stride) % array.shape()[dim];`
  - **Fix**: Division without zero check - use safe_divide()

### src/hfft/real_to_complex.rs

1 issues found:

- Line 138: `let mid = (n_fft + 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()

### src/hfft/symmetric.rs

15 issues found:

- Line 35: `for j in 1..cols / 2 + (cols % 2).not() {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 41: `for i in 1..rows / 2 + (rows % 2).not() {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 48: `array[[rows / 2, 0]] = Complex64::new(array[[rows / 2, 0]].re, 0.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 52: `array[[0, cols / 2]] = Complex64::new(array[[0, cols / 2]].re, 0.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 89: `for i in 1..n / 2 + 1 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 99: `slice[n / 2] = Complex64::new(slice[n / 2].re, 0.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 111: `let flat = view2.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 177: `let dc_val = &array.as_slice().unwrap()[0];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 190: `let data = array.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 191: `for i in 1..n / 2 + 1 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 212: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 215: `for j in 1..cols / 2 + 1 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 227: `for i in 1..rows / 2 + 1 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 272: `for (_i, &amp) in amplitudes.iter().enumerate().skip(1).take(n / 2 - 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 290: `result.push(Complex64::new(amplitudes[n / 2], 0.0));`
  - **Fix**: Division without zero check - use safe_divide()

### src/higher_order_dct_dst.rs

42 issues found:

- Line 45: `let scale = (2.0 / (2.0 * n as f64)).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 45: `let scale = (2.0 / (2.0 * n as f64)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 48: `let phase = PI * (2 * k + 1) as f64 / (4.0 * n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 68: `let scale = (2.0_f64 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 68: `let scale = (2.0_f64 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 73: `let angle = PI * (2 * i + 1) as f64 * (2 * k + 1) as f64 / (4.0 * n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 130: `let scale = 1.0 / (x.len() as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 149: `let scale = (2.0 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 149: `let scale = (2.0 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 154: `let angle = PI * k as f64 * (n_i as f64 + 0.5) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 171: `let scale = (2.0_f64 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 171: `let scale = (2.0_f64 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 176: `let angle = PI * k as f64 * (i as f64 + 0.5) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 200: `let scale = 2.0 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 205: `let angle = PI * (k as f64 + 0.5) * (n_i as f64 + 0.5) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 212: `result[k] *= 0.5_f64.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 254: `let scale = (2.0 / (2.0 * n as f64)).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 254: `let scale = (2.0 / (2.0 * n as f64)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 257: `let phase = PI * (2 * k + 1) as f64 / (4.0 * n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 276: `let scale = (2.0_f64 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 276: `let scale = (2.0_f64 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 281: `let angle = PI * (2 * i + 1) as f64 * (2 * k + 1) as f64 / (4.0 * n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 304: `let scale = (2.0 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 304: `let scale = (2.0 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 309: `let angle = PI * (k as f64 + 0.5) * (n_i as f64 + 1.0) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 341: `let scale = (2.0 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 341: `let scale = (2.0 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 346: `let angle = PI * (k as f64 + 1.0) * (n_i as f64 + 0.5) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 377: `let scale = 2.0 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 382: `let angle = PI * (k as f64 + 0.5) * (n_i as f64 + 0.5) / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 408: `let dct_v_result = dct_v(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 409: `let idct_v_result = idct_v(&dct_v_result).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 443: `let dst_v_result = dst_v(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 444: `let idst_v_result = idst_v(&dst_v_result).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 467: `let _ = dct_v(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 468: `let _ = dct_vi(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 469: `let _ = dct_vii(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 470: `let _ = dct_viii(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 473: `let _ = dst_v(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 474: `let _ = dst_vi(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 475: `let _ = dst_vii(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 476: `let _ = dst_viii(&x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/lib.rs

8 issues found:

- Line 482: `h.iter_mut().take(n / 2).skip(1).for_each(|val| {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 487: `h.iter_mut().skip(n / 2 + 1).for_each(|val| {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 546: `let min = -(n_i32 / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 580: `let dt = 1.0 / sample_rate;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 587: `let analytic = hilbert(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 592: `let start_idx = n / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 593: `let end_idx = 3 * n / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 596: `let magnitude = (analytic[i].re.powi(2) + analytic[i].im.powi(2)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/mdct.rs

19 issues found:

- Line 59: `let half_n = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 74: `let angle = PI / n as f64 * (n_idx as f64 + 0.5 + half_n as f64) * (k as f64 + 0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 77: `result[k] = sum * (2.0 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 77: `result[k] = sum * (2.0 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 121: `let angle = PI / n as f64 * (n_idx as f64 + 0.5 + half_n as f64) * (k as f64 + 0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 124: `result[n_idx] = sum * (2.0 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 124: `result[n_idx] = sum * (2.0 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 170: `let half_n = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 185: `let angle = PI / n as f64 * (n_idx as f64 + 0.5 + half_n as f64) * (k as f64 + 0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 188: `result[k] = sum * (2.0 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 188: `result[k] = sum * (2.0 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 218: `let angle = PI / n as f64 * (n_idx as f64 + 0.5 + half_n as f64) * (k as f64 + 0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 221: `result[n_idx] = sum * (2.0 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 221: `result[n_idx] = sum * (2.0 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 285: `let mdct_result = mdct(&signal, 8, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 297: `let mdct_coeffs = mdct(&signal, 8, window.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 300: `let reconstructed = imdct(&mdct_coeffs, window).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 310: `let mdst_result = mdst(&signal, 8, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 323: `let result = mdct_overlap_add(&blocks, Some(Window::Hann), 4).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory_efficient.rs

17 issues found:

- Line 152: `let scale = if normalize { 1.0 / (n as f64) } else { 1.0 };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 208: `let chunk_size_nz = NonZeroUsize::new(chunk_size).unwrap_or(NonZeroUsize::new(1)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 286: `let mut buffer = complex_input.as_slice_mut().unwrap().to_vec();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 363: `1.0 / ((n_rows_out * n_cols_out) as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 482: `1.0 / (n_val as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 496: `let chunk_size_nz = NonZeroUsize::new(chunk_size_val).unwrap_or(NonZeroUsize::ne...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 556: `1.0 / (chunk_size as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 573: `let full_scale = 1.0 / (n_val as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 574: `let chunk_scale = 1.0 / (chunk_size_val as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 575: `let scale_adjustment = full_scale / chunk_scale;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 604: `fft_inplace(&mut input, &mut output, FftMode::Forward, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 610: `fft_inplace(&mut input, &mut output, FftMode::Inverse, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 625: `let spectrum_2d = fft2_efficient(&arr.view(), None, FftMode::Forward, false).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 631: `let recovered = fft2_efficient(&spectrum_2d.view(), None, FftMode::Inverse, true...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 646: `let result = fft_streaming(&signal, None, FftMode::Forward, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 652: `let inverse = fft_streaming(&result, None, FftMode::Inverse, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 662: `fft_streaming(&signal, None, FftMode::Forward, Some(signal.len())).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/memory_efficient_v2.rs

14 issues found:

- Line 139: `let mut planner = planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 149: `NormMode::Forward => 1.0 / (fft_size as f64),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 151: `NormMode::Ortho => 1.0 / (fft_size as f64).sqrt(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 151: `NormMode::Ortho => 1.0 / (fft_size as f64).sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 242: `let mut planner = planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 253: `NormMode::Backward => 1.0 / (fft_size as f64),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 254: `NormMode::Ortho => 1.0 / (fft_size as f64).sqrt(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 254: `NormMode::Ortho => 1.0 / (fft_size as f64).sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 393: `NormMode::Forward => 1.0 / (n_rows_out * n_cols_out) as f64,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 395: `NormMode::Ortho => 1.0 / ((n_rows_out * n_cols_out) as f64).sqrt(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 395: `NormMode::Ortho => 1.0 / ((n_rows_out * n_cols_out) as f64).sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 508: `NormMode::Backward => 1.0 / (n_rows_out * n_cols_out) as f64,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 509: `NormMode::Ortho => 1.0 / ((n_rows_out * n_cols_out) as f64).sqrt(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 509: `NormMode::Ortho => 1.0 / ((n_rows_out * n_cols_out) as f64).sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/ndim_optimized.rs

3 issues found:

- Line 38: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 189: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 281: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/nufft.rs

27 issues found:

- Line 121: `InterpolationType::Gaussian => 2.0 * (-epsilon.ln()).sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 121: `InterpolationType::Gaussian => 2.0 * (-epsilon.ln()).sqrt(),`
  - **Fix**: Mathematical operation .ln() without validation
- Line 126: `let width = (sigma * sigma * (-epsilon.ln()) / PI).ceil() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 126: `let width = (sigma * sigma * (-epsilon.ln()) / PI).ceil() as usize;`
  - **Fix**: Mathematical operation .ln() without validation
- Line 130: `let h_grid = 2.0 * PI / n_grid as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 138: `let x_grid = (xi + PI) / h_grid;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 144: `let kernel_arg = (x_grid - (i_grid + j) as f64) / sigma;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 170: `if i <= m / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 265: `InterpolationType::Gaussian => 2.0 * (-epsilon.ln()).sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 265: `InterpolationType::Gaussian => 2.0 * (-epsilon.ln()).sqrt(),`
  - **Fix**: Mathematical operation .ln() without validation
- Line 270: `let width = (sigma * sigma * (-epsilon.ln()) / PI).ceil() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 270: `let width = (sigma * sigma * (-epsilon.ln()) / PI).ceil() as usize;`
  - **Fix**: Mathematical operation .ln() without validation
- Line 278: `if i <= m / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 291: `let h_grid = 2.0 * PI / n_grid as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 298: `let x_grid = (xi + PI) / h_grid;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 304: `let kernel_arg = (x_grid - (i_grid + j) as f64) / sigma;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 362: `let scale = 1.0 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 379: `.map(|i| -PI + 1.8 * PI * i as f64 / (n as f64))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 384: `let real = (-xi.powi(2) / 2.0).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 391: `let result = nufft_type1(&x, &samples, m, InterpolationType::Gaussian, 1e-6).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 421: `spectrum[m / 2] = Complex64::new(1.0, 0.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 426: `.map(|i| -PI + 1.8 * PI * i as f64 / (n as f64))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 430: `let result = nufft_type2(&spectrum, &x, InterpolationType::Gaussian, 1e-6).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 436: `let avg_magnitude: f64 = result.iter().map(|c| c.norm()).sum::<f64>() / n as f64...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 447: `.map(|i| -PI + 1.8 * PI * i as f64 / (n as f64))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 453: `let result = nufft_type1(&x, &samples, m, InterpolationType::Linear, 1e-6).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 464: `magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()

### src/optimized_fft.rs

12 issues found:

- Line 144: `self.total_time_ns / self.operation_count as u64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 153: `self.total_flops as f64 / (self.total_time_ns as f64 / 1_000_000_000.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 211: `let mut twiddles = Vec::with_capacity(size / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 212: `let factor = -2.0 * std::f64::consts::PI / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 214: `for k in 0..size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 292: `let mflops = op_count / duration.as_secs_f64() / 1_000_000.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 335: `let mflops = op_count / duration.as_secs_f64() / 1_000_000.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 460: `let mflops = op_count / duration.as_secs_f64() / 1_000_000.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 598: `let rows = shape[0].min(self.config.max_fft_size / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 599: `let cols = shape[1].min(self.config.max_fft_size / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 734: `let output = fft.fft(&input, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 752: `let _ = fft.fft(&input, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/padding.rs

15 issues found:

- Line 132: `(padded_size - n) / 2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 268: `(padded_size - n) / 2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 302: `let t = i as f64 / start_idx as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 310: `let t = 1.0 - (i as f64 / pad_len as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 340: `(padded_size - original_size) / 2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 394: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 401: `(padded_shape[0] - shape[0]) / 2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 409: `(padded_shape[0] - shape[0]) / 2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 414: `(padded_shape[1] - shape[1]) / 2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 420: `.assign(&x_dyn.view().into_dimensionality::<ndarray::Ix2>().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 459: `let padded = auto_pad_complex(&x.mapv(|v| Complex::new(v, 0.0)), &config).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 475: `let padded = auto_pad_complex(&x.mapv(|v| Complex::new(v, 0.0)), &config).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 488: `assert_eq!(unpadded.as_slice().unwrap(), &[0.0, 1.0, 2.0, 3.0]);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 498: `let padded = auto_pad_complex(&x.mapv(|v| Complex::new(v, 0.0)), &config).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 502: `let start = (padded.len() - 3) / 2;`
  - **Fix**: Division without zero check - use safe_divide()

### src/plan_cache.rs

8 issues found:

- Line 64: `*self.enabled.lock().unwrap() = enabled;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 69: `*self.enabled.lock().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 81: `let hit_count = *self.hit_count.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 82: `let miss_count = *self.miss_count.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 85: `hit_count as f64 / total_requests as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 108: `if !*self.enabled.lock().unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 125: `*self.hit_count.lock().unwrap() += 1;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 135: `*self.miss_count.lock().unwrap() += 1;`
  - **Fix**: Replace with ? operator or .ok_or()

### src/plan_serialization.rs

12 issues found:

- Line 223: `let db = self.database.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 236: `let mut db = self.database.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 256: `/ metrics.usage_count as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 277: `let db = self.database.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 305: `let db = self.database.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 356: `let temp_dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 366: `manager.record_plan_usage(&plan_info, 5000).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 372: `manager.save_database().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 387: `let temp_dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 403: `manager.record_plan_usage(&plan_info1, time1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 404: `manager.record_plan_usage(&plan_info2, time2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 410: `let (_, metrics) = best.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/planning.rs

13 issues found:

- Line 709: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 727: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 746: `executor.execute(&input, &mut output).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 752: `let magnitude = (val.re.powi(2) + val.im.powi(2)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 765: `let plan = builder.build().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 773: `let temp_dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 778: `config.serialized_db_path = Some(db_path.to_str().unwrap().to_string());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 787: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 790: `planner.save_plans().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 802: `let mut planner_guard = planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 806: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 814: `let temp_dir = tempdir().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 819: `plan_ahead_of_time(&sizes, Some(db_path.to_str().unwrap())).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/planning_adaptive.rs

8 issues found:

- Line 81: `Duration::from_nanos((self.total_time.as_nanos() / self.count as u128) as u64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 215: `best_time.as_nanos() as f64 / metrics.avg_time.as_nanos() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 279: `let mut planner = self.planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 293: `let mut planner = self.planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 302: `let planner = self.planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 308: `let planner = self.planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 329: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 347: `executor.execute(&input, &mut output).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/planning_parallel.rs

11 issues found:

- Line 124: `let mut planner = self.base_planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 134: `let mut planner = planner_clone.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 162: `let mut planner = self.base_planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 194: `let mut planner_guard = planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 229: `let planner = self.base_planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 235: `let planner = self.base_planner.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 419: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 430: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 439: `executor.execute(&input, &mut output).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 450: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 465: `let times = executor.execute_batch(&inputs, &mut outputs).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/rfft.rs

19 issues found:

- Line 51: `let n_output = n_val / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 228: `let n_rows_result = n_rows_out / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 290: `return Ok(Array2::from_shape_vec((2, 2), vec![3.0, 6.0, 9.0, 12.0]).unwrap());`
  - **Fix**: Handle array creation errors properly
- Line 347: `let scale_factor = (n_rows_out * n_cols_out) as f64 / (n_rows * n_cols) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 510: `out_shape[last_axis] = out_shape[last_axis] / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 853: `if idx[axis] == 0 || (shape[axis] % 2 == 0 && idx[axis] == shape[axis] / 2) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 930: `let spectrum = rfft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 933: `assert_eq!(spectrum.len(), signal.len() / 2 + 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 939: `let recovered = irfft(&spectrum, Some(signal.len())).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 951: `let padded_spectrum = rfft(&signal, Some(8)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 954: `assert_eq!(padded_spectrum.len(), 8 / 2 + 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 960: `let recovered = irfft(&padded_spectrum, Some(4)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 974: `let spectrum_2d = rfft2(&arr.view(), None, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 977: `assert_eq!(spectrum_2d.dim(), (arr.dim().0 / 2 + 1, arr.dim().1));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 983: `let recovered_2d = irfft2(&spectrum_2d.view(), Some((2, 2)), None, None).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1000: `.map(|i| (2.0 * PI * freq * i as f64 / n as f64).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1004: `let spectrum = rfft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1008: `let expected_peak = n as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1019: `let recovered = irfft(&spectrum, Some(n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/signal_processing.rs

29 issues found:

- Line 154: `for i in 0..=cutoff_idx.min(size / 2) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 156: `if i > 0 && i < size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 162: `for i in cutoff_idx..=size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 164: `if i > 0 && i < size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 170: `for i in cutoff_idx..=cutoff_high_idx.min(size / 2) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 172: `if i > 0 && i < size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 178: `for i in 0..=size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 181: `if i > 0 && i < size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 218: `0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / size as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 227: `0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / size as f64).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 235: `let x = 2.0 * std::f64::consts::PI * i as f64 / size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 246: `let x = 2.0 * i as f64 / size as f64 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 247: `let window_val = bessel_i0(beta * (1.0 - x * x).sqrt()) / bessel_i0(beta);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 247: `let window_val = bessel_i0(beta * (1.0 - x * x).sqrt()) / bessel_i0(beta);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 269: `let y = (x / 3.75).powi(2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 275: `let y = 3.75 / ax;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 276: `(ax.exp() / ax.sqrt())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 276: `(ax.exp() / ax.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 444: `let half_order = adjusted_order / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 462: `* (2.0 * std::f64::consts::PI * i as f64 / (adjusted_order - 1) as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 470: `- (2.0 * std::f64::consts::PI * i as f64 / (adjusted_order - 1) as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 476: `let x = 2.0 * std::f64::consts::PI * i as f64 / (adjusted_order - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 483: `let x = 2.0 * i as f64 / (adjusted_order - 1) as f64 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 484: `window[i] = bessel_i0(beta * (1.0 - x * x).sqrt()) / bessel_i0(beta);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 484: `window[i] = bessel_i0(beta * (1.0 - x * x).sqrt()) / bessel_i0(beta);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 536: `signal[i] += (2.0 * std::f64::consts::PI * 2.0 * i as f64 / n as f64).sin();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 541: `signal[i] += 0.5 * (2.0 * std::f64::consts::PI * 10.0 * i as f64 / n as f64).sin...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 554: `let filtered = frequency_filter(&signal, &filter_spec).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 569: `let result = convolve(&signal, &kernel).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/simd_fft.rs

1 issues found:

- Line 66: `let size = (len as f64).sqrt() as usize;`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/simd_rfft.rs

15 issues found:

- Line 57: `let scale = 1.0 / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 61: `let scale = 1.0 / (n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 61: `let scale = 1.0 / (n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 65: `let scale = 1.0 / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 123: `let scale = 1.0 / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 127: `let scale = 1.0 / (n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 127: `let scale = 1.0 / (n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 131: `let scale = 1.0 / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 194: `let spectrum = rfft_simd(&signal, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 197: `assert_eq!(spectrum.len(), signal.len() / 2 + 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 209: `let spectrum = rfft_simd(&signal, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 212: `let recovered = irfft_simd(&spectrum, Some(signal.len()), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 227: `let spectrum = rfft_adaptive(&signal, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 228: `assert_eq!(spectrum.len(), signal.len() / 2 + 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 230: `let recovered = irfft_adaptive(&spectrum, Some(signal.len()), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/sparse_fft/algorithms.rs

10 issues found:

- Line 75: `let log_sum: f64 = magnitudes.iter().map(|&x| (x + epsilon).ln()).sum::<f64>();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 76: `let geometric_mean = (log_sum / magnitudes.len() as f64).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 79: `let arithmetic_mean: f64 = magnitudes.iter().sum::<f64>() / magnitudes.len() as ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 86: `let flatness = geometric_mean / arithmetic_mean;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 356: `2.0 * std::f64::consts::PI * (best_idx as f64) * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 357: `let component = best_value * Complex64::new(phase.cos(), phase.sin()) / (n as f6...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 405: `let mean: f64 = magnitudes.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 406: `let variance: f64 = magnitudes.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 407: `let std_dev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 467: `for start in (0..n).step_by(window_size / 2) {`
  - **Fix**: Division without zero check - use safe_divide()

### src/sparse_fft/estimation.rs

17 issues found:

- Line 117: `let window_size = (n / 16).max(3).min(n);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 120: `let start = if i >= window_size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 121: `i - window_size / 2`
  - **Fix**: Division without zero check - use safe_divide()
- Line 125: `let end = (i + window_size / 2 + 1).min(n);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 128: `let mean = window_mags.iter().sum::<f64>() / window_mags.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 130: `window_mags.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window_mags.len()...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 136: `let mean_variance = local_variances.iter().sum::<f64>() / local_variances.len() ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 166: `let step_size = window_size / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 182: `.map(|&x| x.ln())`
  - **Fix**: Mathematical operation .ln() without validation
- Line 186: `(log_sum / count).exp()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 193: `let arithmetic_mean = window_power.iter().sum::<f64>() / window_power.len() as f...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 197: `geometric_mean / arithmetic_mean`
  - **Fix**: Division without zero check - use safe_divide()
- Line 223: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 238: `let result = estimate_sparsity_threshold(&signal, 0.1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 249: `let result = estimate_sparsity_adaptive(&signal, 0.25, 10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 260: `let result = estimate_sparsity_frequency_pruning(&signal, 2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 270: `let result = estimate_sparsity_spectral_flatness(&signal, 0.3, 8).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/sparse_fft/reconstruction.rs

14 issues found:

- Line 175: `let original_nyquist = original_length / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 176: `let target_nyquist = target_length / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 186: `((i as f64) * (target_nyquist as f64) / (original_nyquist as f64)).round() as us...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 302: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 311: `let sparse_result = sparse_fft(&signal, 4, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 312: `let spectrum = reconstruct_spectrum(&sparse_result, 64).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 339: `let sparse_result = sparse_fft(&signal, 4, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 340: `let reconstructed = reconstruct_time_domain(&sparse_result, 64).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 348: `let sparse_result = sparse_fft(&signal, 4, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 349: `let high_res = reconstruct_high_resolution(&sparse_result, 32, 64).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 357: `let sparse_result = sparse_fft(&signal, 4, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 366: `let sparse_result = sparse_fft(&signal, 6, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 370: `if freq_index <= n / 8 || freq_index >= 7 * n / 8 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 377: `let filtered = reconstruct_filtered(&sparse_result, 64, filter).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/sparse_fft/tests.rs

28 issues found:

- Line 13: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 30: `let result = sparse_fft(&signal, 6, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 53: `let estimated_k = processor.estimate_sparsity(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 77: `let result = processor.sparse_fft(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 83: `let result2 = frequency_pruning_sparse_fft(&signal, 2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 97: `noisy_signal[i] += 0.1 * (i as f64 / n as f64 - 0.5);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 112: `let result = processor.sparse_fft(&noisy_signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 118: `let result2 = spectral_flatness_sparse_fft(&noisy_signal, 0.5, 16).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 127: `let result = windowing::apply_window(&signal, config::WindowFunction::Hann, 14.0...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 134: `let result = windowing::apply_window(&signal, config::WindowFunction::Hamming, 1...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 141: `let result = windowing::apply_window(&signal, config::WindowFunction::None, 14.0...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 154: `let sparse_result = sparse_fft(&signal, 4, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 157: `let spectrum = reconstruction::reconstruct_spectrum(&sparse_result, n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 161: `let reconstructed = reconstruction::reconstruct_time_domain(&sparse_result, n).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 165: `let high_res = reconstruction::reconstruct_high_resolution(&sparse_result, n, 2 ...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 176: `let result = adaptive_sparse_fft(&signal, 0.1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 198: `let result = sparse_fft(&signal, 4, Some(algorithm), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 212: `let result = sparse_fft(&signal, 4, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 227: `let result = sparse_fft(&single_sample, 1, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 232: `let result = sparse_fft(&small_signal, 10, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 243: `let estimated = estimation::estimate_sparsity_threshold(&signal, 0.1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 247: `let estimated = estimation::estimate_sparsity_adaptive(&signal, 0.25, 10).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 251: `let estimated = estimation::estimate_sparsity_frequency_pruning(&signal, 2.0).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 255: `let estimated = estimation::estimate_sparsity_spectral_flatness(&signal, 0.3, 8)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 293: `let result = processor.sparse_fft(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 318: `let sparse_result = sparse_fft(&signal, 6, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 322: `if freq_idx <= n / 8 || freq_idx >= 7 * n / 8 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 329: `let filtered = reconstruction::reconstruct_filtered(&sparse_result, n, low_pass)...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/sparse_fft/windowing.rs

13 issues found:

- Line 48: `let window_val = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 57: `let window_val = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 66: `let angle = 2.0 * PI * i as f64 / (n - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 76: `let angle = 2.0 * PI * i as f64 / (n - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 89: `let alpha = (n - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 93: `let x = beta * (1.0 - ((i as f64 - alpha) / alpha).powi(2)).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 93: `let x = beta * (1.0 - ((i as f64 - alpha) / alpha).powi(2)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 94: `let window_val = modified_bessel_i0(x) / i0_beta;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 109: `let half_x = x / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 112: `term *= (half_x / k as f64).powi(2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 129: `let result = apply_window(&signal, WindowFunction::None, 14.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 139: `let result = apply_window(&signal, WindowFunction::Hann, 14.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 150: `let result = apply_window(&signal, WindowFunction::Hamming, 14.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/sparse_fft_batch.rs

4 issues found:

- Line 438: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 485: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 528: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 574: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/sparse_fft_gpu.rs

3 issues found:

- Line 289: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 314: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 340: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/sparse_fft_gpu_cuda.rs

5 issues found:

- Line 525: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 546: `let devices = get_cuda_devices().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 550: `let context = GpuContext::new(0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 571: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 592: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/sparse_fft_gpu_kernels.rs

8 issues found:

- Line 680: `let idx = i * (signal.len() / sparsity);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 681: `let val = Complex64::new(1.0 / (i + 1) as f64, 0.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 703: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 724: `let kernel = factory.create_fft_kernel(1024, 0x10000, 0x20000).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 743: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 765: `let (input_address, output_address) = launcher.allocate_fft_memory(1024).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 772: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 802: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/sparse_fft_gpu_memory.rs

14 issues found:

- Line 867: `let mut global = GLOBAL_MEMORY_MANAGER.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 879: `let global = GLOBAL_MEMORY_MANAGER.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 904: `let _manager = manager.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 941: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 950: `manager.release_buffer(buffer).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 972: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 975: `manager.release_buffer(buffer1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 983: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 989: `manager.clear_cache().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1003: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1006: `let manager = get_global_memory_manager().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1007: `let mut manager = manager.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1012: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1015: `manager.release_buffer(buffer).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/sparse_fft_gpu_performance.rs

11 issues found:

- Line 133: `.min_by(|a, b| a.stats.execution_time_ms.partial_cmp(&b.stats.execution_time_ms)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 400: `SparseFFTAlgorithm::Sublinear => { /* Default is fine */ },`
  - **Fix**: Division without zero check - use safe_divide()
- Line 409: `SparseFFTAlgorithm::Deterministic => { /* Default is fine */ },`
  - **Fix**: Division without zero check - use safe_divide()
- Line 410: `SparseFFTAlgorithm::FrequencyPruning => { /* Default is fine */ },`
  - **Fix**: Division without zero check - use safe_divide()
- Line 509: `let mean = signal_f64.iter().sum::<f64>() / signal_f64.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 510: `let variance = signal_f64.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / sig...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 511: `let std_dev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 515: `let snr_estimate = if std_dev > 0.0 { peak / std_dev } else { f64::INFINITY };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 709: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 749: `medium_snr[i] += 0.05 * (i as f64 / 1024.0 - 0.5);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 756: `low_snr[i] += 0.2 * (i as f64 / 1024.0 - 0.5);`
  - **Fix**: Division without zero check - use safe_divide()

### src/sparse_fft_multi_gpu.rs

14 issues found:

- Line 336: `let performance_history = self.performance_history.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 350: `1.0 / (times.iter().sum::<f64>() / times.len() as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 501: `let base_size = signal_len / num_devices;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 528: `let ratio = device_compute / total_compute;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 552: `let ratio = device.memory_free as f32 / total_memory as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 579: `let size = (signal_len as f32 * ratio / total_ratio) as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 633: `return Ok(chunk_results.into_iter().next().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 696: `self.performance_history.lock().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 701: `self.performance_history.lock().unwrap().clear();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 741: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 765: `processor.initialize().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 789: `let result = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 805: `let chunk_sizes = processor.calculate_chunk_sizes(1000, 3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 816: `let chunks = processor.split_signal(&signal, &chunk_sizes).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/sparse_fft_original.rs

84 issues found:

- Line 207: `- (2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0)).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 221: `* (2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0)).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 233: `let x = 2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 254: `let x = 2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 274: `let y = (x / 3.75).powi(2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 282: `let y = 3.75 / x.abs();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 283: `(x.abs().exp() / x.abs().sqrt())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 283: `(x.abs().exp() / x.abs().sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 302: `let x = 2.0 * i as f64 / (n as f64 - 1.0) - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 303: `let arg = beta * (1.0 - x * x).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 304: `let window_val = i0(arg) / i0_beta;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 378: `let mean: f64 = magnitudes.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 381: `magnitudes.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 383: `let std_dev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 466: `let log_sum: f64 = magnitudes.iter().map(|&x| (x + epsilon).ln()).sum::<f64>();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 467: `let geometric_mean = (log_sum / magnitudes.len() as f64).exp();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 470: `let arithmetic_mean: f64 = magnitudes.iter().sum::<f64>() / magnitudes.len() as ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 477: `let flatness = geometric_mean / arithmetic_mean;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 616: `let original_nyquist = original_length / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 617: `let target_nyquist = target_length / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 627: `((i as f64) * (target_nyquist as f64) / (original_nyquist as f64)).round() as us...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 733: `let mean: f64 = magnitudes.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 734: `let variance: f64 = magnitudes.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 735: `let std_dev = variance.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 776: `if idx != 0 && (n % 2 == 0 && idx != n / 2) && selected_indices.len() < k {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 853: `let segment_score = (1.0 - local_flatness) * segment_energy.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 904: `if idx != 0 && (n % 2 == 0 && idx != n / 2) && selected_indices.len() < k {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 919: `let global_mean = magnitudes.iter().sum::<f64>() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 934: `if i != 0 && (n % 2 == 0 && i != n / 2) && selected_indices.len() < k {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1018: `if idx != 0 && (n % 2 == 0 && idx != n / 2) && selected_indices.len() < k {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1111: `if idx != 0 && (n % 2 == 0 && idx != n / 2) && selected_indices.len() < k {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1204: `if max_idx != 0 && (n % 2 == 0 && max_idx != n / 2) && indices.len() < k {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1236: `if residual_energy / original_energy < 1e-10 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1309: `if idx != 0 && (n % 2 == 0 && idx != n / 2) && selected_indices.len() < k {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2009: `let t = 2.0 * PI * (i as f64) / (n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2026: `let result = sparse_fft(&signal, 6, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2049: `let estimated_k = processor.estimate_sparsity(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2073: `let result = processor.sparse_fft(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2079: `let result2 = frequency_pruning_sparse_fft(&signal, 2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2093: `noisy_signal[i] += 0.1 * (i as f64 / n as f64 - 0.5);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2111: `let windowed_signal = processor.apply_window(&signal_complex).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2112: `let spectrum = fft(&windowed_signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2126: `let result = processor.sparse_fft(&noisy_signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2134: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2138: `let reconstructed_spectrum = reconstruct_spectrum(&result2, n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2139: `let reconstructed_signal = ifft(&reconstructed_spectrum, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2150: `let signal_scale = 1.0 / signal_energy.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2150: `let signal_scale = 1.0 / signal_energy.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 2151: `let recon_scale = 1.0 / recon_energy.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2151: `let recon_scale = 1.0 / recon_energy.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 2163: `let relative_error = (error_sum / (2.0 * n as f64)).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2163: `let relative_error = (error_sum / (2.0 * n as f64)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 2185: `spectral_flatness_sparse_fft(&noisy_signal, 0.3, 32, Some(*window)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2200: `let sparse_result = sparse_fft(&signal, 4, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2203: `let reconstructed = reconstruct_spectrum(&sparse_result, n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2214: `let signal_scale = 1.0 / signal_energy.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2214: `let signal_scale = 1.0 / signal_energy.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 2215: `let recon_scale = 1.0 / recon_energy.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2215: `let recon_scale = 1.0 / recon_energy.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 2227: `let relative_error = (error_sum / (2.0 * n as f64)).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2227: `let relative_error = (error_sum / (2.0 * n as f64)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 2271: `let result = processor.sparse_fft(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2289: `let result = processor.sparse_fft(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2304: `let result = adaptive_sparse_fft(&signal, 0.1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2321: `let x = 2.0 * PI * (i as f64) / (rows as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2322: `let y = 2.0 * PI * (j as f64) / (cols as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2337: `let result = sparse_fft2(&signal, (rows, cols), 4, Some(*window)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2359: `1.0 / orig_energy.sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2359: `1.0 / orig_energy.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 2364: `1.0 / recon_energy.sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2364: `1.0 / recon_energy.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 2379: `(error_sum / (2.0 * len as f64)).sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2379: `(error_sum / (2.0 * len as f64)).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 2390: `let sparse_result = sparse_fft(&signal, 6, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2393: `let reconstructed = reconstruct_time_domain(&sparse_result, n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2410: `let high_res = reconstruct_high_resolution(&sparse_result, n, target_length).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2421: `let energy_ratio = high_res_energy / orig_energy;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2430: `let nyquist = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2443: `let filtered = reconstruct_filtered(&sparse_result, n, lowpass).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2487: `let windowed_signal = processor.apply_window(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2500: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2501: `let windowed_fft = fft(&windowed_signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2539: `sparse_fft(&signal, 2, Some(SparseFFTAlgorithm::Sublinear), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2551: `let windowed_result = processor.sparse_fft(&signal).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/sparse_fft_specialized_hardware.rs

11 issues found:

- Line 290: `data.len() as f64 / (self.info.capabilities.memory_bandwidth_gb_s * 1000.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 295: `data.len() as f64 / (1024.0 * 1024.0 * 1024.0) / (transfer_time_us / 1_000_000.0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 310: `data.len() as f64 / (self.info.capabilities.memory_bandwidth_gb_s * 1000.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 318: `data.len() as f64 / (1024.0 * 1024.0 * 1024.0) / (transfer_time_us / 1_000_000.0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 360: `/ (elapsed.as_secs_f64() * 1e9),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 471: `let transfer_time_ns = data.len() as f64 / self.info.capabilities.memory_bandwid...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 477: `let transfer_time_ns = data.len() as f64 / self.info.capabilities.memory_bandwid...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 636: `let accelerator = self.accelerators.get_mut(&best_accelerator).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 698: `score += 50.0 / info.capabilities.power_consumption_watts;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 791: `let discovered = manager.discover_accelerators().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 813: `let result = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/spectrogram.rs

26 issues found:

- Line 116: `let noverlap = noverlap.unwrap_or(nperseg / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 143: `let mut num_frames = 1 + (x_f64.len() - nperseg) / step;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 164: `num_frames = 1 + (padded.len() - nperseg) / step;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 189: `num_frames = 1 + (padded.len() - nperseg) / step;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 195: `let freq_len = if return_onesided { nfft / 2 + 1 } else { nfft };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 196: `let frequencies: Vec<f64> = (0..freq_len).map(|i| i as f64 * fs / nfft as f64).c...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 200: `.map(|i| (i * step + nperseg / 2) as f64 / fs)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 217: `let mean = detrended.iter().sum::<f64>() / detrended.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 350: `"density" => 1.0 / (fs * win_sum_sq),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 351: `"spectrum" => 1.0 / win_sum_sq,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 378: `magnitude[[i, j]] = val.norm() * scale_factor.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 391: `phase[[i, j]] = phase[[i, j]] * 180.0 / PI;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 503: `10.0 * (val / max_val).log10()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 515: `spec_norm[[i, j]] = (val + db_range).max(0.0).min(db_range) / db_range;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 530: `.map(|i| (2.0 * PI * freq * (i as f64 / fs)).sin())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 554: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 557: `let expected_num_freqs = nperseg / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 558: `let expected_num_frames = 1 + (signal.len() - nperseg) / (nperseg - noverlap);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 585: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 591: `.min_by(|(_, &a), (_, &b)| (a - freq).abs().partial_cmp(&(b - freq).abs()).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 592: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 603: `let avg_power = total_power / zxx.shape()[0] as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 614: `let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 632: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 666: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 694: `spectrogram_normalized(&signal, Some(fs), Some(128), Some(64), Some(80.0)).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/strided_fft.rs

6 issues found:

- Line 195: `let scale = 1.0 / (axis_len as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 256: `let result = fft_strided(&input, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 274: `let result1 = fft_strided(&input, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 278: `let result2 = fft_strided(&input, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 292: `let forward = fft_strided_complex(&input, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 293: `let inverse = ifft_strided(&forward, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/time_frequency.rs

35 issues found:

- Line 179: `let hop_size = config.hop_size.min(window_size / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 194: `let num_frames = ((signal.len() - window_size) / hop_size) + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 197: `let num_frames = num_frames.min(config.max_size / window_size);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 200: `let num_bins = padded_size / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 213: `time / fs`
  - **Fix**: Division without zero check - use safe_divide()
- Line 221: `let freq = k as f64 / padded_size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 269: `hop_size as f64 / sample_rate.unwrap_or(1.0),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 273: `sample_rate.unwrap_or(1.0) / padded_size as f64,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 297: `let num_freqs = config.frequency_bins.min(config.max_size / 4);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 300: `let log_min = min_freq.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 301: `let log_max = max_freq.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 302: `let log_step = (log_max - log_min) / (num_freqs as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 315: `time / fs`
  - **Fix**: Division without zero check - use safe_divide()
- Line 394: `let dt = 1.0 / sample_rate;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 395: `let scale = 1.0 / scale_freq;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 400: `let freq = if k <= n / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 401: `k as f64 / (n as f64 * dt)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 403: `-((n - k) as f64) / (n as f64 * dt)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 421: `wavelet_fft[k] = Complex64::new(exp_term * scale.sqrt(), 0.0);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 432: `Complex64::new(exp_term * norm_freq.powi(2) * scale.sqrt(), 0.0);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 447: `Complex64::new(h * scale.sqrt() * norm_freq.powi(m) * exp_term, 0.0);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 460: `let real_part = exp_term * norm_freq.powi(m) * scale.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 494: `let max_frames = num_frames.min(config.max_size / num_bins);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 495: `let max_bins = num_bins.min(config.max_size / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 517: `.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 594: `/ 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 597: `let inst_freq = phase_diff / (2.0 * PI) * sample_rate.unwrap_or(1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 606: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 723: `let t = i as f64 / sample_rate;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 739: `let result = compute_stft(&signal, &config, Some(sample_rate)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 754: `let mid_frame = result.times.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 777: `let t = i as f64 / sample_rate;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 792: `let result = compute_cwt(&signal, &config, Some(sample_rate)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 798: `config.frequency_bins.min(config.max_size / 4)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 806: `let mid_time = result.times.len() / 2;`
  - **Fix**: Division without zero check - use safe_divide()

### src/waterfall.rs

20 issues found:

- Line 87: `let noverlap = noverlap.unwrap_or(nperseg / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 198: `let noverlap = noverlap.unwrap_or(nperseg / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 316: `let noverlap = noverlap.unwrap_or(nperseg / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 351: `.map(|i| i * (n_times - 1) / (n_lines - 1))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 450: `let noverlap = noverlap.unwrap_or(nperseg / 2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 596: `(amp - 0.35) / 0.31`
  - **Fix**: Division without zero check - use safe_divide()
- Line 605: `(amp - 0.125) / 0.25`
  - **Fix**: Division without zero check - use safe_divide()
- Line 609: `(1.0 - amp) / 0.125`
  - **Fix**: Division without zero check - use safe_divide()
- Line 618: `(0.625 - amp) / 0.25`
  - **Fix**: Division without zero check - use safe_divide()
- Line 628: `colors[[i, 0]] = amp.powf(1.5) * 0.9; // Red increases nonlinearly`
  - **Fix**: Mathematical operation .powf( without validation
- Line 629: `colors[[i, 1]] = amp.powf(0.8) * 0.9; // Green increases faster`
  - **Fix**: Mathematical operation .powf( without validation
- Line 640: `colors[[i, 0]] = 0.05 + amp.powf(0.7) * 0.95; // Red increases quickly`
  - **Fix**: Mathematical operation .powf( without validation
- Line 641: `colors[[i, 1]] = amp.powf(2.0) * 0.9; // Green increases slowly then faster`
  - **Fix**: Mathematical operation .powf( without validation
- Line 684: `let t: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 706: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 745: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 756: `assert!(freq_mesh.iter().fold(f64::MIN, |a, &b| a.max(b)) <= fs / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 783: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 835: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 868: `let colors = apply_colormap(&amplitudes, cmap).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/window.rs

43 issues found:

- Line 219: `if x < (n as f64) / 2.0 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 220: `w[i] = 2.0 * x / (n as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 222: `w[i] = 2.0 - 2.0 * x / (n as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 267: `let half_n = (n as f64) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 270: `let x = (i as f64 - half_n + 0.5).abs() / half_n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 297: `let half_n = (n as f64 - 1.0) / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 300: `let x = ((i as f64) - half_n).abs() / half_n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 302: `w[i] = (1.0 - x) * (PI * x).cos() + (PI * x).sin() / PI;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 343: `let fac = 1.0 / (n as f64 - 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 370: `(0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 372: `(0..n).map(|i| i as f64 / n as f64).collect()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 400: `let center = if sym { (n as f64 - 1.0) / 2.0 } else { 0.0 };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 405: `let x = (i as f64 - center).abs() / (tau * (n as f64));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 443: `let width = (alpha * (n as f64 - 1.0) / 2.0).floor() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 447: `let x = 0.5 * (1.0 + ((PI * i as f64) / width as f64).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 454: `let x = 0.5 * (1.0 + ((PI * i as f64) / width as f64).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 491: `let x = beta * (1.0 - ((i as f64 - alpha) / alpha).powi(2)).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 491: `let x = beta * (1.0 - ((i as f64 - alpha) / alpha).powi(2)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 492: `w[i] = bessel_i0(x) / i0_beta;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 523: `let center = if sym { (n as f64 - 1.0) / 2.0 } else { 0.0 };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 526: `let x = (i as f64 - center) / (std * (n as f64 - 1.0) / 2.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 551: `2.0 * PI / (n as f64 - 1.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 553: `2.0 * PI / n as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 579: `let y = (x / 3.75).powi(2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 589: `let y = 3.75 / ax;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 590: `let exp_term = (ax).exp() / (ax).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 590: `let exp_term = (ax).exp() / (ax).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 643: `result[i] = x[i] * F::from_f64(win[i]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 680: `Ok(n_f64 * sum_squared / square_sum)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 690: `let win = rectangular(5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 699: `let win = hann(5, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 708: `let win = hamming(5, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 717: `let win = blackman(5, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 726: `assert_eq!(Window::from_str("hann").unwrap(), Window::Hann);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 727: `assert_eq!(Window::from_str("hamming").unwrap(), Window::Hamming);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 728: `assert_eq!(Window::from_str("blackman").unwrap(), Window::Blackman);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 730: `Window::from_str("rectangular").unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 738: `let win1 = get_window(Window::Hann, 5, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 739: `let win2 = get_window("hann", 5, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 749: `let win = apply_window(&signal.view(), Window::Hann).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 760: `let rect_enbw = enbw(Window::Rectangular, 1024).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 763: `let hann_enbw = enbw(Window::Hann, 1024).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 766: `let hamming_enbw = enbw(Window::Hamming, 1024).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/window_extended.rs

51 issues found:

- Line 94: `let r = 10.0_f64.powf(attenuation_db / 20.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 94: `let r = 10.0_f64.powf(attenuation_db / 20.0);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 95: `let beta = (r + (r * r - 1.0).sqrt()).ln() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 95: `let beta = (r + (r * r - 1.0).sqrt()).ln() / n as f64;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 95: `let beta = (r + (r * r - 1.0).sqrt()).ln() / n as f64;`
  - **Fix**: Mathematical operation .ln() without validation
- Line 98: `let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 99: `w[i] = ((n as f64 - 1.0) * beta * (1.0 - x * x).sqrt().acos()).cosh() / r.cosh()...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 99: `w[i] = ((n as f64 - 1.0) * beta * (1.0 - x * x).sqrt().acos()).cosh() / r.cosh()...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 116: `let t = 2.0 * i as f64 / (n - 1) as f64 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 117: `w[i] = (PI * width * n as f64 * t).sin() / (PI * t);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 125: `w.mapv_inplace(|x| x / sum * n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 134: `let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 138: `(PI * x).sin() / (PI * x)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 153: `let t = i as f64 / (n - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 155: `1.0 / ((-epsilon / (t * (epsilon - t))).exp() + 1.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 157: `1.0 / ((-epsilon / ((1.0 - t) * (t - 1.0 + epsilon))).exp() + 1.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 174: `w.mapv_inplace(|x| x / sum * n);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 182: `let half_n = n as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 185: `let t = (i as f64 - half_n).abs() / half_n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 195: `let hann_part = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 196: `let poisson_part = (-alpha * (n as f64 / 2.0 - i as f64).abs() / (n as f64 / 2.0...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 204: `let center = (n - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 207: `let t = (i as f64 - center) / center;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 208: `w[i] = 1.0 / (1.0 + (alpha * t).powi(2));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 223: `let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 225: `w[i] = (1.0 - (x / x0).powi(2)).powf(mu);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 225: `w[i] = (1.0 - (x / x0).powi(2)).powf(mu);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 248: `let a = sidelobe_level_db.abs() / 20.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 249: `let r = 10.0_f64.powf(a);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 254: `let x = 2.0 * PI * k as f64 * (i as f64 / (n - 1) as f64 - 0.5);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 255: `sum += (-1.0_f64).powi(k as i32 + 1) * x.cos() / k as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 262: `w.mapv_inplace(|x| x / max_val);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 295: `let coherent_gain = window.sum() / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 299: `let processing_gain = window.sum().powi(2) / (n as f64 * sum_squared);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 302: `let enbw = n as f64 * sum_squared / window.sum().powi(2);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 323: `.map(|&m| 20.0 * (m / max_mag).log10())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 328: `for (i, &val) in mag_db.iter().enumerate().take(n_fft / 2).skip(1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 330: `main_lobe_width = 2.0 * i as f64 * fs / n_fft as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 337: `let main_lobe_end = (main_lobe_width * n_fft as f64 / fs).ceil() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 338: `for &val in mag_db.iter().take(n_fft / 2).skip(main_lobe_end) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 345: `let bin_edge_response = magnitude[n_fft / (2 * n)];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 346: `let scalloping_loss_db = -20.0 * (bin_edge_response / max_mag).log10();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 386: `.map(|&m| 20.0 * (m / max_mag).max(1e-100).log10())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 429: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 433: `let lanczos = get_extended_window(ExtendedWindow::Lanczos, n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 437: `let poisson = get_extended_window(ExtendedWindow::Poisson { alpha: 2.0 }, n).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 444: `let window = crate::window::get_window(Window::Hann, n, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 446: `let props = analyze_window(&window, Some(1000.0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 461: `crate::window::get_window(Window::Hann, n, true).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 465: `crate::window::get_window(Window::Hamming, n, true).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 469: `let comparison = compare_windows(&windows).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/worker_pool.rs

6 issues found:

- Line 75: `self.config.lock().unwrap().num_workers`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 85: `let mut config = self.config.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 92: `self.config.lock().unwrap().enabled`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 97: `self.config.lock().unwrap().enabled = enabled;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 126: `let config = self.config.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 249: `let pool = WorkerPool::with_config(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/simd_fft_test.rs

2 issues found:

- Line 18: `let standard_result = fft(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 21: `let simd_result = fft_simd(&signal, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()