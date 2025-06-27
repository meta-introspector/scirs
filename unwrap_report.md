# Unwrap() Usage Report

Total unwrap() calls and unsafe operations found: 3632

## Summary by Type

- Replace with ? operator or .ok_or(): 2092 occurrences
- Division without zero check - use safe_divide(): 937 occurrences
- Mathematical operation .sqrt() without validation: 332 occurrences
- Use .get().ok_or(Error::IndexOutOfBounds)?: 124 occurrences
- Use .get() with proper bounds checking: 94 occurrences
- Mathematical operation .ln() without validation: 32 occurrences
- Handle array creation errors properly: 16 occurrences
- Mathematical operation .powf( without validation: 5 occurrences

## Detailed Findings


### benches/attention_bench.rs

11 issues found:

- Line 31: `let scale = 1.0 / f32::sqrt(d_model as f32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 40: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 49: `let scale = 1.0 / f32::sqrt(d_model as f32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 51: `causal_attention(&query.view(), &key.view(), &value.view(), scale).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 59: `let scale = 1.0 / f32::sqrt(d_model as f32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 69: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 77: `let scale = 1.0 / f32::sqrt(d_model as f32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 79: `linear_attention(&query.view(), &key.view(), &value.view(), scale).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 88: `let head_dim = d_model / num_heads;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 99: `scale: Some(1.0 / f32::sqrt(head_dim as f32)),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 115: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()

### benches/comprehensive_decomposition_bench.rs

36 issues found:

- Line 65: `b.iter(|| lu(black_box(&m.view()), None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 75: `lu(black_box(&m.view()), None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 88: `solve(black_box(&m.view()), black_box(&r.view()), None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 114: `|b, m| b.iter(|| qr(black_box(&m.view()), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 119: `b.iter(|| qr(black_box(&m.view()), None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 124: `b.iter(|| qr(black_box(&m.view()), None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 131: `|b, m| b.iter(|| qr(black_box(&m.view()), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 138: `|b, m| b.iter(|| rank_revealing_qr(black_box(&m.view()), 1e-12).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 163: `|b, m| b.iter(|| svd(black_box(&m.view()), false, None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 170: `|b, m| b.iter(|| svd(black_box(&m.view()), true, None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 175: `b.iter(|| svd(black_box(&m.view()), true, None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 180: `b.iter(|| svd(black_box(&m.view()), true, None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 187: `|b, m| b.iter(|| svd(black_box(&m.view()), true, None).unwrap().1),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 208: `|b, m| b.iter(|| cholesky(black_box(&m.view()), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 218: `let l = cholesky(black_box(&m.view()), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 219: `solve_triangular(&l.view(), black_box(&r.view()), false, false).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 259: `|b, m| b.iter(|| eigvals(black_box(&m.view()), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 266: `|b, m| b.iter(|| eig(black_box(&m.view()), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 273: `|b, m| b.iter(|| eigvalsh(black_box(&m.view()), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 280: `|b, m| b.iter(|| eigh(black_box(&m.view()), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 291: `smallest_k_eigh(black_box(&m.view()), 10, size, 1e-10).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 314: `b.iter(|| schur(black_box(&m.view())).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 321: `schur(black_box(&m.view())).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 329: `schur(black_box(&m.view())).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 351: `let (u, p) = advanced_polar_decomposition(black_box(&m.view()), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 360: `let (u, p) = advanced_polar_decomposition(black_box(&m.view()), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 369: `let (u, _) = advanced_polar_decomposition(black_box(&m.view()), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 410: `eigvals_gen(black_box(&a.view()), black_box(&matrix_b.view()), None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 422: `eig_gen(black_box(&a.view()), black_box(&matrix_b.view()), None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 445: `|b, m| b.iter(|| complex_lu(black_box(&m.view())).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 452: `|b, m| b.iter(|| complex_qr(black_box(&m.view())).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 460: `|b, m| b.iter(|| complex_svd(black_box(&m.view()), true).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 468: `|b, m| b.iter(|| complex_eig(black_box(&m.view())).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 536: `lu(&m_copy.view(), None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 543: `|b, m| b.iter(|| lu(black_box(&m.view()), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 563: `|b, m| b.iter(|| cholesky(black_box(&m.view()), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()

### benches/linalg_bench.rs

6 issues found:

- Line 196: `let toeplitz = ToeplitzMatrix::new(r.view(), r.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 197: `solve_toeplitz(r.view(), r.view(), black_box(rhs.view())).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 208: `let circulant = CirculantMatrix::new(r.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 209: `solve_circulant(r.view(), black_box(rhs.view())).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 391: `let target_dim = size / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 401: `gaussian_random_matrix(black_box(*d), black_box(m.ncols())).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### benches/matrix_functions_bench.rs

42 issues found:

- Line 41: `matrix[[i, j]] = 0.1 / ((j - i) as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 52: `let t = i as f64 / (n - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 64: `let (q, _) = qr(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 87: `|b, m| b.iter(|| matrix_functions::expm(black_box(&m.view()), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 94: `|b, m| b.iter(|| matrix_functions::expm(black_box(&m.view()), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 101: `|b, m| b.iter(|| matrix_functions::expm(black_box(&m.view()), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 108: `|b, m| b.iter(|| matrix_functions::expm(black_box(&m.view()), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 118: `matrix_functions::expm(black_box(&m.view()), None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 130: `matrix_functions::expm(black_box(&m.view()), None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 154: `b.iter(|| logm(black_box(&m.view())).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 161: `|b, m| b.iter(|| logm(black_box(&m.view())).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 168: `matrix_functions::logm(black_box(&m.view())).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 179: `matrix_functions::logm(black_box(&m.view())).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 207: `matrix_power(black_box(&m.view()), *p as f64).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 223: `matrix_power(black_box(&m.view()), *p as f64).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 226: `matrix_power(black_box(&m.view()), *p).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 254: `matrix_power(black_box(&m.view()), *p).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 276: `b.iter(|| sqrtm(black_box(&m.view()), 100, 1e-12).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 283: `|b, m| b.iter(|| sqrtm(black_box(&m.view()), 100, 1e-12).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 293: `sqrtm(black_box(&m.view()), 100, 1e-12).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 305: `sqrtm(black_box(&m.view()), 100, 1e-12).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 317: `sqrtm(black_box(&m.view()), 100, 1e-12).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 341: `matrix_functions::signm(black_box(&m.view())).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 349: `|b, m| b.iter(|| signm(black_box(&m.view())).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 356: `matrix_functions::signm(black_box(&m.view())).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 364: `matrix_functions::signm(black_box(&m.view())).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 384: `b.iter(|| matrix_functions::cosm(black_box(&m.view())).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 389: `b.iter(|| sinm(black_box(&m.view())).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 394: `b.iter(|| tanm(black_box(&m.view())).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 399: `b.iter(|| matrix_functions::coshm(black_box(&m.view())).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 404: `b.iter(|| sinhm(black_box(&m.view())).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 409: `b.iter(|| tanhm(black_box(&m.view())).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 432: `matrix_functions::asinm(black_box(&m.view())).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 440: `matrix_functions::acosm(black_box(&m.view())).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 451: `matrix_functions::atanm(black_box(&m.view())).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 506: `matrix_functions::expm(black_box(&m.view()), None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 515: `matrix_functions::expm(black_box(&m.view()), None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 534: `sqrtm(black_box(&m.view()), 100, 1e-12).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 561: `matrix_functions::expm(black_box(&m.view()), None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 572: `|b, (m, iter)| b.iter(|| sqrtm(black_box(&m.view()), black_box(*iter), 1e-12).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 595: `|b, m| b.iter(|| matrix_functions::expm(black_box(&m.view()), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 601: `|b, m| b.iter(|| sqrtm(black_box(&m.view()), 100, 1e-12).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()

### benches/perf_opt_bench.rs

6 issues found:

- Line 37: `black_box(blocked_matmul(&a.view(), &b.view(), &config_blocked).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 45: `black_box(blocked_matmul(&a.view(), &b.view(), &config_adaptive).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 72: `inplace_add(&mut a_copy.view_mut(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 117: `let _result = black_box(optimized_transpose(&a.view()).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 142: `black_box(blocked_matmul(&a.view(), &b.view(), &config_serial).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 155: `black_box(blocked_matmul(&a.view(), &b.view(), &config_parallel).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()

### benches/quantization_bench.rs

7 issues found:

- Line 13: `Array2::from_shape_fn((rows, cols), |(i, j)| ((i * cols + j) % 100) as f32 / 100...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 17: `Array1::from_iter((0..size).map(|i| (i % 100) as f32 / 100.0))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 128: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 146: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 195: `b.iter(|| black_box(calibrate_matrix(&matrix.view(), 8, &minmax_config_clone).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 200: `black_box(calibrate_matrix(&matrix.view(), 8, &percentile_config_clone).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 205: `b.iter(|| black_box(calibrate_matrix(&matrix.view(), 8, &ema_config_clone).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()

### benches/quantized_solvers_bench.rs

25 issues found:

- Line 71: `bench.iter(|| black_box(conjugate_gradient(&standard_op, &b, 100, 1e-6).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 95: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 103: `black_box(conjugate_gradient(&quantized_linear_op, &b, 100, 1e-6).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 128: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 135: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 161: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 168: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 191: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 196: `let precond = quantized_jacobi_preconditioner(&quantized_op).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 208: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 239: `bench.iter(|| black_box(gmres(&standard_op, &b, 100, 1e-6, Some(20)).unwrap()))`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 263: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 269: `black_box(gmres(&quantized_linear_op, &b, 100, 1e-6, Some(20)).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 294: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 298: `quantized_gmres(&quantized_op, &b, 100, 1e-6, Some(20), false).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 324: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 328: `quantized_gmres(&quantized_op, &b, 100, 1e-6, Some(20), true).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 374: `bench.iter(|| black_box(conjugate_gradient(&standard_op, &b, 100, 1e-6).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 398: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 404: `quantized_conjugate_gradient(&quantized_op, &b, 100, 1e-6, false).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 430: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 435: `let precond = quantized_jacobi_preconditioner(&quantized_op).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 447: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 481: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 487: `quantized_conjugate_gradient(&quantized_op, &b, 50, 1e-5, false).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()

### benches/scipy_compat_benchmarks.rs

51 issues found:

- Line 50: `b.iter(|| compat::det(&m.view(), false, true).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 54: `b.iter(|| det(&m.view(), None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 59: `b.iter(|| compat::inv(&m.view(), false, true).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 63: `b.iter(|| inv(&m.view(), None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 83: `|b, m| b.iter(|| compat::norm(&m.view(), Some("fro"), None, false, true).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 89: `|b, m| b.iter(|| matrix_norm(&m.view(), "frobenius", None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 94: `b.iter(|| compat::norm(&m.view(), Some("1"), None, false, true).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 98: `b.iter(|| matrix_norm(&m.view(), "1", None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 105: `|b, m| b.iter(|| compat::norm(&m.view(), Some("inf"), None, false, true).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 109: `b.iter(|| matrix_norm(&m.view(), "inf", None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 129: `|b, v| b.iter(|| compat::vector_norm(&v.view(), Some(2.0), true).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 135: `|b, v| b.iter(|| vector_norm(&v.view(), 2).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 142: `|b, v| b.iter(|| compat::vector_norm(&v.view(), Some(1.0), true).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 148: `|b, v| b.iter(|| vector_norm(&v.view(), 1).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 155: `|b, v| b.iter(|| compat::vector_norm(&v.view(), Some(f64::INFINITY), true).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 161: `|b, v| b.iter(|| vector_norm(&v.view(), 0).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 182: `b.iter(|| compat::lu(&m.view(), false, false, true, false).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 186: `b.iter(|| lu(&m.view(), None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 191: `b.iter(|| compat::qr(&m.view(), false, None, "full", false, true).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 195: `b.iter(|| qr(&m.view(), None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 201: `b.iter(|| compat::svd(&m.view(), true, true, false, true, "gesdd").unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 205: `b.iter(|| svd(&m.view(), false, None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 213: `|b, m| b.iter(|| compat::cholesky(&m.view(), true, false, true).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 219: `|b, m| b.iter(|| cholesky(&m.view(), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 256: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 264: `|b, m| b.iter(|| eigen::eigvalsh(&m.view(), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 287: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 293: `b.iter(|| eigen::eigh(&m.view(), None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 329: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 337: `|b, (m, r)| b.iter(|| solve(&m.view(), &r.view(), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 350: `compat::lstsq(&m.view(), &r.view(), None, false, false, true, None).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 358: `|b, (m, r)| b.iter(|| lstsq(&m.view(), &r.view(), None).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 380: `b.iter(|| compat::expm(&m.view(), None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 384: `b.iter(|| expm(&m.view(), None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 389: `b.iter(|| compat::sqrtm(&m.view(), Some(true)).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 393: `b.iter(|| sqrtm(&m.view(), 100, 1e-12).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 400: `b.iter(|| compat::logm(&m.view()).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 404: `b.iter(|| logm(&m.view()).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 423: `b.iter(|| compat::cond(&m.view(), Some("2")).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 427: `b.iter(|| cond(&m.view(), None, None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 432: `b.iter(|| compat::matrix_rank(&m.view(), None, false, true).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 436: `b.iter(|| matrix_rank(&m.view(), None, None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 442: `b.iter(|| compat::pinv(&m.view(), None, false, true).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 462: `b.iter(|| compat::rq(&m.view(), false, None, "full", true).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 467: `b.iter(|| compat::polar(&m.view(), "right").unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 471: `b.iter(|| compat::polar(&m.view(), "left").unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 495: `|b, blocks| b.iter(|| compat::block_diag(&blocks).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 522: `compat::det(&m.view(), false, true).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 545: `b.iter(|| compat::det(&m.view(), false, true).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 550: `b.iter(|| compat::norm(&m.view(), Some("fro"), None, false, true).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 556: `b.iter(|| compat::lu(&m.view(), false, false, true, false).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()

### benches/simd_bench.rs

8 issues found:

- Line 53: `Array1::from_iter((0..size).map(|i| (i % 100) as f32 / 100.0))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 57: `Array2::from_shape_fn((rows, cols), |(i, j)| ((i * cols + j) % 100) as f32 / 100...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 79: `simd_matvec_f32(&black_box(matrix.view()), &black_box(vector.view())).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 94: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 123: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 135: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 162: `black_box(simd_dot_f32(&black_box(vec_a.view()), &black_box(vec_b.view())).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 170: `.unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/advanced_features_showcase.rs

3 issues found:

- Line 76: `let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 175: `let blockwise_norm = norm_accumulator.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 316: `let sparsity = 1.0 - (non_zeros as f64) / (size * size) as f64;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/attention_example.rs

12 issues found:

- Line 15: `let scale = 1.0 / (d_model as f32).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 15: `let scale = 1.0 / (d_model as f32).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 26: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 33: `let causal_output = causal_attention(&query.view(), &key.view(), &value.view(), ...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 49: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 72: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 79: `let linear_output = linear_attention(&query.view(), &key.view(), &value.view(), ...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 88: `let head_dim = d_model / num_heads;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 102: `scale: Some(1.0 / (head_dim as f32).sqrt()),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 102: `scale: Some(1.0 / (head_dim as f32).sqrt()),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 116: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 124: `let rope_output = rotary_embedding(&query.view(), freq_base).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/autograd_linalg_example.rs

4 issues found:

- Line 165: `println!("dL/dX shape: {:?}", results[0].as_ref().unwrap().shape());`
  - **Fix**: Use .get() with proper bounds checking
- Line 166: `println!("dL/dW shape: {:?}", results[1].as_ref().unwrap().shape());`
  - **Fix**: Use .get() with proper bounds checking
- Line 167: `println!("dL/db shape: {:?}", results[2].as_ref().unwrap().shape());`
  - **Fix**: Use .get() with proper bounds checking
- Line 171: `let eye_val = eye_3.eval(ctx).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/batch_attention_example.rs

8 issues found:

- Line 15: `let scale = 1.0 / (d_model as f32).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 15: `let scale = 1.0 / (d_model as f32).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 33: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 54: `let head_dim = d_model / num_heads;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 71: `scale: Some(1.0 / (head_dim as f32).sqrt()),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 71: `scale: Some(1.0 / (head_dim as f32).sqrt()),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 86: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 122: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/broadcast_example.rs

6 issues found:

- Line 30: `let c = broadcast_matmul_3d(&a, &b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 41: `let y = broadcast_matvec(&a_dyn, &x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 44: `println!("First batch result: {:?}", &y.as_slice().unwrap()[0..2]);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 45: `println!("Second batch result: {:?}", &y.as_slice().unwrap()[2..4]);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 56: `let c = broadcast_matmul(&a, &b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 72: `let c = broadcast_matmul_3d(&a, &b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/calibration_benchmark.rs

19 issues found:

- Line 73: `let uniform = Uniform::new(-1.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 88: `let normal = Normal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 103: `let lognormal = LogNormal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 118: `let normal1 = Normal::new(-2.0, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 119: `let normal2 = Normal::new(2.0, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 143: `let region_size = size / 4;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 179: `let cauchy = Cauchy::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 224: `let params = calibrate_matrix(&data.view(), BITS, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 233: `let mse = diff.mapv(|x| x * x).sum() / data.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 238: `let avg_time = elapsed.as_millis() as f32 / NUM_ITERATIONS as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 239: `let avg_mse = total_mse / NUM_ITERATIONS as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 243: `let size_reduction = (1.0 - (BITS as f32 / fp32_size as f32)) * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 282: `let params = calibrate_matrix(&data.view(), bits, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 290: `let mse = diff.mapv(|x| x * x).sum() / data.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 294: `let rel_error = diff_abs.sum() / data.mapv(|x| x.abs()).sum() * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 298: `let size_reduction = (1.0 - (bits as f32 / fp32_size as f32)) * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 349: `let mse = diff.mapv(|x| x * x).sum() / data.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 353: `let rel_error = diff_abs.sum() / data.mapv(|x| x.abs()).sum() * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 357: `let size_reduction = (1.0 - (bits as f32 / fp32_size as f32)) * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()

### examples/comprehensive_core_linalg.rs

1 issues found:

- Line 314: `let y = Array1::from_shape_fn(size, |i| (i as f64 + 1.0).sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation

### examples/dynamic_quantization_example.rs

17 issues found:

- Line 47: `let normal = Normal::new(mean, std_dev).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 90: `let static_params = calibrate_matrix(&data_sequence[0].view(), bits, &static_con...`
  - **Fix**: Use .get() with proper bounds checking
- Line 100: `calibrate_matrix(&data_sequence[0].view(), bits, &dynamic_config).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 110: `let static_mse = (data - &static_dequantized).mapv(|x| x * x).sum() / data.len()...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 115: `dynamic_params = calibrate_matrix(&data.view(), bits, &dynamic_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 120: `let dynamic_mse = (data - &dynamic_dequantized).mapv(|x| x * x).sum() / data.len...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 124: `((static_mse - dynamic_mse) / static_mse) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 139: `let avg_static_mse = total_static_mse / data_sequence.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 140: `let avg_dynamic_mse = total_dynamic_mse / data_sequence.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 141: `let overall_improvement = ((avg_static_mse - avg_dynamic_mse) / avg_static_mse) ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 177: `calibrate_matrix(&data_sequence[0].view(), bits, configs.last().unwrap()).unwrap...`
  - **Fix**: Use .get() with proper bounds checking
- Line 188: `params_list[j] = calibrate_matrix(&data.view(), bits, &configs[j]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 193: `let mse = (data - &dequantized).mapv(|x| x * x).sum() / data.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 235: `let mut params = calibrate_matrix(&initial_data.view(), bits, &dynamic_config).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 265: `let error = (&data - &dequantized).mapv(|x| x.abs()).mean().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 279: `params = calibrate_matrix(&data.view(), bits, &dynamic_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 298: `let normal = Normal::new(drift, amplitude).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/extended_precision_example.rs

6 issues found:

- Line 23: `hilbert[[i, j]] = 1.0 / ((i + j + 1) as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 85: `let a_f32 = Array2::from_shape_fn((3, 3), |(i, j)| 1.0 / ((i + j + 1) as f32));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 149: `hilbert_det[[i, j]] = 1.0 / ((i + j + 1) as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 209: `println!("Improvement factor: {:.2}x", error_std / error_ext);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 300: `sym_matrix[[i, j]] = 1.0 / ((i + j + 1) as f32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 401: `max_residual_std / max_residual_ext`
  - **Fix**: Division without zero check - use safe_divide()

### examples/fft_spectral_analysis_example.rs

27 issues found:

- Line 39: `Complex64::new((2.0 * PI * 3.0 * i as f64 / n_power2 as f64).sin(), 0.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 60: `Complex64::new((2.0 * PI * 2.0 * i as f64 / n_arbitrary as f64).cos(), 0.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 75: `(2.0 * PI * 5.0 * i as f64 / n_real as f64).sin()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 76: `+ 0.5 * (2.0 * PI * 10.0 * i as f64 / n_real as f64).cos()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 86: `100 - (rfft_result.len() * 100 / n_real)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 97: `reconstruction_error = (reconstruction_error / n_real as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 97: `reconstruction_error = (reconstruction_error / n_real as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 111: `let x = 2.0 * PI * (i as f64) / (rows as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 112: `let y = 2.0 * PI * (j as f64) / (cols as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 126: `error_2d = (error_2d / (rows * cols) as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 126: `error_2d = (error_2d / (rows * cols) as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 147: `error_3d = (error_3d / (depth * height * width) as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 147: `error_3d = (error_3d / (depth * height * width) as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 196: `let leakage_ratio = 1.0 - main_lobe_energy / total_energy;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 226: `dct_error = (dct_error / dct_signal.len() as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 226: `dct_error = (dct_error / dct_signal.len() as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 231: `sorted_coeffs.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 236: `100.0 * top_8_energy / total_energy`
  - **Fix**: Division without zero check - use safe_divide()
- Line 273: `.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 274: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 288: `let t = i as f64 / fs;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 313: `peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 333: `n_samples / (segment_length / 2) - 1,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 347: `welch_peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 390: `Complex64::new((2.0 * PI * i as f64 / size as f64).sin(), 0.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 399: `let theoretical_speedup = naive_operations as f64 / fft_operations as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 404: `fft_time.as_nanos() as f64 / 1000.0,`
  - **Fix**: Division without zero check - use safe_divide()

### examples/generic_example.rs

9 issues found:

- Line 27: `let c = gemm(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 31: `let det = gdet(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 35: `let a_inv = ginv(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 39: `let svd = gsvd(&a.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 43: `let qr = gqr(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 47: `let eigen = geig(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 52: `let x = gsolve(&a.view(), &b_vec.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 63: `let c = gemm(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 67: `let det = gdet(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/hierarchical_matrix_example.rs

11 issues found:

- Line 28: `let dist = ((i as f64 - j as f64).powi(2) + 1.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 29: `1.0 / (1.0 + dist)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 58: `/ memory_info.original_size as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 91: `dense_time.as_nanos() as f64 / h_time.as_nanos() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 140: `hss_dense_time.as_nanos() as f64 / hss_time.as_nanos() as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 153: `let r1 = (i as f64 / block_size as f64) * (j as f64 / block_size as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 154: `let r2 = ((i + j) as f64 / (2.0 * block_size as f64)).sin() * 0.5;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 155: `let r3 = ((i as f64 - j as f64).abs() / block_size as f64).exp() * 0.1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 165: `let compression = (block_size * block_size) as f64 / (u.len() + v.len()) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 175: `let rmse = (error / (block_size * block_size) as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 175: `let rmse = (error / (block_size * block_size) as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation

### examples/kfac_neural_optimization_example.rs

6 issues found:

- Line 51: `let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 51: `let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 107: `Some((previous_loss - current_loss) / previous_loss)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 187: `let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 187: `let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 220: `let kfac_improvement = sgd_losses.last().unwrap_or(&1.0) / kfac_losses.last().un...`
  - **Fix**: Division without zero check - use safe_divide()

### examples/matrix_dynamics_example.rs

18 issues found:

- Line 47: `initial_vector.as_slice().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 63: `exp_time.as_nanos() as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 67: `evolved_vector.as_slice().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 96: `lyapunov_time.as_nanos() as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 105: `let residual_norm = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 137: `riccati_time.as_nanos() as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 143: `let input_cost_inv = 1.0 / input_cost[[0, 0]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 188: `ode_time.as_nanos() as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 200: `let mid_point = n_points / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 245: `let sqrt_half = (0.5_f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 252: `println!("   |ψ(0)⟩ = {:?}", initial_state.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 255: `let evolution_times = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0, 2.0 * PI];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 274: `t / PI,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 279: `quantum_time.as_nanos() as f64 / 1_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 322: `println!("   Eigenvalues: {:?}", eigenvalues.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 371: `let test_vector = Array2::from_shape_fn((n, 1), |(i, _)| (i as f64 + 1.0) / n as...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 386: `elapsed.as_nanos() as f64 / 1_000_000.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 387: `memory_estimate as f64 / 1024.0`
  - **Fix**: Division without zero check - use safe_divide()

### examples/matrix_transformations_example.rs

2 issues found:

- Line 19: `let angle_data = array![std::f64::consts::PI / 4.0].into_dyn();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 136: `example::run().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/matrixfree_example.rs

10 issues found:

- Line 25: `let y = spd_op.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 39: `let solution = matrix_free_conjugate_gradient(&spd_op, &b, 10, 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 46: `let ax = spd_op.apply(&solution.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 66: `let solution2 = matrix_free_gmres(&nonsym_op, &b2, 10, 1e-10, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 73: `let ax2 = nonsym_op.apply(&solution2.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 82: `let precond = jacobi_preconditioner(&spd_op).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 86: `matrix_free_preconditioned_conjugate_gradient(&spd_op, &precond, &b, 10, 1e-10)....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 93: `let ax3 = spd_op.apply(&solution3.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 120: `let y5 = block_op.apply(&x5.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 135: `sum.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation

### examples/neural_network_quantization_example.rs

20 issues found:

- Line 59: `let std_dev1 = (2.0 / 32.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 59: `let std_dev1 = (2.0 / 32.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 63: `weights1[[i, j]] = Normal::new(0.0, std_dev1).unwrap().sample(&mut rng);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 69: `let std_dev2 = (2.0 / 64.0).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 69: `let std_dev2 = (2.0 / 64.0).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 73: `weights2[[i, j]] = Normal::new(0.0, std_dev2).unwrap().sample(&mut rng);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 79: `biases2[[i, 0]] = 0.01 * Normal::new(0.0, 1.0).unwrap().sample(&mut rng);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 104: `input[[i, j]] = Normal::new(0.0, 1.0).unwrap().sample(&mut rng);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 161: `calibrate_matrix(&layer.weights.view(), bits, &weights_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 167: `let bias_params = calibrate_matrix(&layer.biases.view(), bits, &bias_config).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 208: `let act_params = calibrate_matrix(&input.view(), 8, &activation_config).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 236: `let hidden_params = calibrate_matrix(&hidden_activated.view(), 8, &hidden_config...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 265: `let mse = (full_precision - quantized).mapv(|x| x * x).sum() / full_precision.le...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 274: `/ full_precision.mapv(|x| x.abs()).sum()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 311: `(top1_matches as f32 / batch_size as f32) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 352: `calibrate_matrix(&layer.weights.view(), w_bits, &weights_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 357: `let bias_params = calibrate_matrix(&layer.biases.view(), 8, &bias_config).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 381: `let act_params = calibrate_matrix(&input.view(), a_bits0, &activation_config).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 408: `calibrate_matrix(&hidden_activated.view(), a_bits1, &hidden_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 446: `let memory_reduction = (1.0 - (mixed_weight_size as f32 / fp32_weight_size as f3...`
  - **Fix**: Division without zero check - use safe_divide()

### examples/out_of_core_example.rs

27 issues found:

- Line 18: `println!("Using temporary file: {}", file_path.to_str().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 48: `file_path.to_str().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 50: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 60: `let y = chunked.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 71: `(y[i] - expected[i]).abs() / expected[i].abs()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 93: `file_path.to_str().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 95: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 111: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 118: `let residual_norm = (r.dot(&r)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 119: `let b_norm = (b.dot(&b)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 120: `let relative_residual = residual_norm / b_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 151: `file_path_8bit.to_str().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 153: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 165: `file_path_4bit.to_str().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 167: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 184: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 189: `let residual_norm_8bit = (r_8bit.dot(&r_8bit)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 190: `let b_norm = (b.dot(&b)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 191: `let relative_residual_8bit = residual_norm_8bit / b_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 199: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 204: `let residual_norm_4bit = (r_4bit.dot(&r_4bit)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 205: `let relative_residual_4bit = residual_norm_4bit / b_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 210: `.map(|m| m.len() as f64 / 1024.0 / 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 213: `.map(|m| m.len() as f64 / 1024.0 / 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 221: `(1.0 - file_size_4bit / file_size_8bit) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 229: `(elapsed_8bit.as_secs_f64() / elapsed_4bit.as_secs_f64() - 1.0) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 237: `relative_residual_4bit / relative_residual_8bit`
  - **Fix**: Division without zero check - use safe_divide()

### examples/per_channel_quantization_example.rs

2 issues found:

- Line 102: `println!("  Improvement: {:.2}x\n", std_total_err / perchan_total_err);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 187: `std_asym_total_err / perchan_asym_total_err`
  - **Fix**: Division without zero check - use safe_divide()

### examples/perf_opt_example.rs

6 issues found:

- Line 71: `time_standard.as_secs_f64() / time_blocked.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 104: `time_standard.as_secs_f64() / time_inplace.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 108: `memory_standard as f64 / (1024.0 * 1024.0)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 131: `time_standard.as_secs_f64() / time_inplace.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 159: `time_standard.as_secs_f64() / time_optimized.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 225: `time_serial.as_secs_f64() / time_parallel.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()

### examples/preconditioners_example.rs

23 issues found:

- Line 56: `println!("   Test vector: {:?}", x_test.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 59: `preconditioned_result.as_slice().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 86: `unpreconditioned_time.as_nanos() as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 90: `preconditioned_time.as_nanos() as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 93: `let speedup = unpreconditioned_time.as_nanos() as f64 / preconditioned_time.as_n...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 136: `ilu_time.as_nanos() as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 141: `let residual_norm_ilu = (residual_ilu.iter().map(|&x| x * x).sum::<f64>()).sqrt(...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 143: `println!("   Solution: {:?}", solution_ilu.as_slice().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 184: `ic_time.as_nanos() as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 188: `let residual_norm_ic = (residual_ic.iter().map(|&x| x * x).sum::<f64>()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 197: `let block_i = i / 3;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 198: `let block_j = j / 3;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 203: `} else if (i as i32 - j as i32).abs() == 1 && i / 3 == j / 3 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 239: `bj_time.as_nanos() as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 243: `let residual_norm_bj = (residual_bj.iter().map(|&x| x * x).sum::<f64>()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 284: `poly_time.as_nanos() as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 288: `let residual_norm_poly = (residual_poly.iter().map(|&x| x * x).sum::<f64>()).sqr...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 354: `adaptive_time.as_nanos() as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 358: `let residual_norm_adaptive = (residual_adaptive.iter().map(|&x| x * x).sum::<f64...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 422: `setup_time.as_nanos() as f64 / 1_000_000.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 423: `apply_time.as_nanos() as f64 / 1_000.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 424: `analysis.memory_usage_bytes as f64 / 1024.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 512: `zero_elements as f64 / total_elements as f64`
  - **Fix**: Division without zero check - use safe_divide()

### examples/quantization_calibration_example.rs

16 issues found:

- Line 47: `let uniform = Uniform::new(-1.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 62: `let normal = Normal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 77: `let normal1 = Normal::new(-2.0, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 78: `let normal2 = Normal::new(2.0, 0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 153: `let params = calibrate_matrix(&data.view(), bits, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 160: `let mse = (data - &dequantized).mapv(|x| x * x).sum() / data.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 184: `let params_std = calibrate_matrix(&data.view(), bits, &config_std).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 188: `let mse_std = (data - &dequantized_std).mapv(|x| x * x).sum() / data.len() as f3...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 201: `let params_pc = calibrate_matrix(&data.view(), bits, &config_pc).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 213: `let mse_pc = (data - &dequantized_pc).mapv(|x| x * x).sum() / data.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 216: `println!("  Improvement: {:.2}x", mse_std / mse_pc);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 232: `let col_mse_std = (&col_data - &col_std).mapv(|x| x * x).sum() / col_data.len() ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 234: `let col_mse_pc = (&col_data - &col_pc).mapv(|x| x * x).sum() / col_data.len() as...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 260: `let params = calibrate_matrix(&data.view(), bit, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 267: `let mse = (data - &dequantized).mapv(|x| x * x).sum() / data.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 271: `(data - &dequantized).mapv(|x| x.abs()).sum() / data.mapv(|x| x.abs()).sum();`
  - **Fix**: Division without zero check - use safe_divide()

### examples/quantization_example.rs

24 issues found:

- Line 23: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 82: `Array2::from_shape_vec((2, 4), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])...`
  - **Fix**: Handle array creation errors properly
- Line 138: `Array2::from_shape_vec((3, 3), vec![1.2, 2.5, 3.7, 4.2, 5.0, 6.1, 7.3, 8.4, 9.5]...`
  - **Fix**: Handle array creation errors properly
- Line 140: `let b = Array2::from_shape_vec((3, 2), vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5]).unwra...`
  - **Fix**: Handle array creation errors properly
- Line 142: `let x = Array1::from_shape_vec(3, vec![0.1, 0.2, 0.3]).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 240: `let c_q = quantized_matmul(&a_q_symmetric, &a_params_symmetric, &b_q, &b_params)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 247: `let rel_error = (&c - &c_q).mapv(|x| x.abs()).sum() / c.sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 261: `let y_q = quantized_matvec(&a_q_symmetric, &a_params_symmetric, &x_q, &x_params)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 268: `let rel_error = (&y - &y_q).mapv(|x| x.abs()).sum() / y.sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 281: `let dot_q = quantized_dot(&x_q, &x_params, &x_q, &x_params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 288: `let rel_error = (dot - dot_q).abs() / dot;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 308: `let mse = (&a - &a_fake_q).mapv(|x| x * x).sum() / a.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 364: `let mse = (&a - &a_fake_q).mapv(|x| x * x).sum() / a.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 390: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 412: `100.0 * (orig - dequant).abs() / orig.abs()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 441: `100.0 * (orig - dequant).abs() / orig.abs()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 458: `let a_for_f16 = Array2::from_shape_vec((2, 3), vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6...`
  - **Fix**: Handle array creation errors properly
- Line 459: `let b_for_f16 = Array2::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6...`
  - **Fix**: Handle array creation errors properly
- Line 471: `let c_f16 = quantized_matmul(&a_f16, &a_f16_params, &b_f16, &b_f16_params).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 476: `let rel_error_f16 = (&c_full - &c_f16).mapv(|x| x.abs()).sum() / c_full.sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 505: `100.0 * int8_size as f32 / original_size as f32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 510: `100.0 * int4_size as f32 / original_size as f32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 515: `100.0 * f16_size as f32 / original_size as f32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 520: `100.0 * bf16_size as f32 / original_size as f32`
  - **Fix**: Division without zero check - use safe_divide()

### examples/quantization_ml_example.rs

19 issues found:

- Line 40: `let normal = Normal::new(0.0, 0.1).unwrap(); // Small standard deviation typical...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 60: `let val = Normal::new(0.0, 1.0).unwrap().sample(&mut rng);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 117: `let weights_params = calibrate_matrix(&weights.view(), bits, &config_weights).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 123: `(weights - &dequantized_weights).mapv(|x| x * x).sum() / weights.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 127: `calibrate_matrix(&activations.view(), bits, &config_activations).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 137: `/ activations.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 158: `/ reference_result.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 164: `/ reference_result.mapv(|x| x.abs()).sum()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 203: `let weights_params = calibrate_matrix(&weights.view(), bits, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 213: `let activations_params = calibrate_matrix(&activations.view(), bits, &config_act...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 238: `/ reference_result.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 244: `/ reference_result.mapv(|x| x.abs()).sum()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 249: `let memory_savings = (1.0 - (bits as f32 / fp32_size as f32)) * 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 297: `calibrate_matrix(&weights.view(), weight_bits, &weights_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 303: `calibrate_matrix(&activations.view(), act_bits, &activations_config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 315: `/ reference_result.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 319: `/ reference_result.mapv(|x| x.abs()).sum()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 325: `let weight_savings = 1.0 - (weight_bits as f32 / fp32_size as f32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 326: `let act_savings = 1.0 - (act_bits as f32 / fp32_size as f32);`
  - **Fix**: Division without zero check - use safe_divide()

### examples/quantized_solvers_example.rs

20 issues found:

- Line 58: `let x_standard = conjugate_gradient(&standard_op, &b, 10, 1e-6).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 64: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 69: `let x_quantized = quantized_conjugate_gradient(&quantized_op, &b, 10, 1e-6, fals...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 75: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 81: `quantized_conjugate_gradient(&quantized_op_4bit, &b, 10, 1e-6, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 115: `let x_standard = gmres(&standard_op, &b, 10, 1e-6, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 121: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 124: `let x_quantized = quantized_gmres(&quantized_op, &b, 10, 1e-6, None, false).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 130: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 133: `let x_quantized_4bit = quantized_gmres(&quantized_op_4bit, &b, 10, 1e-6, None, f...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 165: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 170: `let precond = quantized_jacobi_preconditioner(&quantized_op).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 174: `quantized_conjugate_gradient(&quantized_op, &b, 10, 1e-6, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 183: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 220: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 225: `let x_banded = quantized_conjugate_gradient(&banded_op, &b, 20, 1e-6, false).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 257: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 262: `let x_standard = quantized_conjugate_gradient(&quantized_op, &b, 50, 1e-5, false...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 266: `let x_adaptive = quantized_conjugate_gradient(&quantized_op, &b, 50, 1e-5, true)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 279: `v.dot(v).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation

### examples/random_matrices_example.rs

2 issues found:

- Line 230: `let actual_density = nnz as f64 / total as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 233: `"\nNon-zero elements: {} / {} (density: {:.3})",`
  - **Fix**: Division without zero check - use safe_divide()

### examples/scalable_algorithms_example.rs

22 issues found:

- Line 55: `let freq1 = 2.0 * std::f64::consts::PI * (i as f64) / 100.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 56: `let freq2 = 2.0 * std::f64::consts::PI * (j as f64) / 10.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 57: `freq1.sin() + 0.5 * freq2.cos() + 0.1 * (i + j) as f64 / 1000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 64: `m_tall as f64 / n_tall as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 75: `tsqr_time.as_nanos() as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 117: `let signal = (j as f64 / 50.0).sin() * (i as f64 + 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 118: `signal + 0.01 * (i * j) as f64 / 10000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 125: `m_short as f64 / n_short as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 134: `lq_time.as_nanos() as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 199: `result.memory_estimate as f64 / 1024.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 237: `let matrix_a = Array2::from_shape_fn(a_size, |(i, j)| ((i + j + 1) as f64).sin()...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 238: `let matrix_b = Array2::from_shape_fn(b_size, |(i, j)| ((i * j + 1) as f64).cos()...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 258: `blocked_time.as_nanos() as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 262: `standard_time.as_nanos() as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 285: `Array2::from_shape_fn((300, rank_true), |(i, j)| ((i + j + 1) as f64).sin() / 10...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 290: `Array2::from_shape_fn((rank_true, 200), |(i, j)| ((i * j + 1) as f64).cos() / 10...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 314: `randomized_time.as_nanos() as f64 / 1_000_000.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 337: `approximation_error = approximation_error.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 339: `let matrix_norm = low_rank_matrix.iter().map(|&x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 340: `let relative_error = approximation_error / matrix_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 363: `Array2::from_shape_fn((m, n), |(i, j)| (i + j + 1) as f64 / (m + n) as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 389: `elapsed.as_nanos() as f64 / 1_000_000.0,`
  - **Fix**: Division without zero check - use safe_divide()

### examples/scipy_compat_example.rs

3 issues found:

- Line 32: `println!("Q =\n{}", q.unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 38: `println!("U =\n{}", u.unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 40: `println!("Vt =\n{}", vt.unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/scipy_compat_showcase.rs

13 issues found:

- Line 245: `let det_result = compat::det(&a.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 249: `let norm_result = compat::norm(&a.view(), Some("fro"), None, false, true).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 253: `let rank = compat::matrix_rank(&a.view(), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 262: `let (p, l, u) = compat::lu(&a.view(), false, false, true, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 268: `let (q_opt, r) = compat::qr(&a.view(), false, None, "full", false, true).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 273: `let (u_opt, s, vt_opt) = compat::svd(&a.view(), true, true, false, true, "gesdd"...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 284: `let exp_result = compat::expm(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 288: `let sqrt_result = compat::sqrtm(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 292: `let pinv_result = compat::pinv(&a.view(), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 301: `let norm_2 = compat::vector_norm(&v.view(), Some(2.0), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 305: `let norm_1 = compat::vector_norm(&v.view(), Some(1.0), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 309: `let norm_inf = compat::vector_norm(&v.view(), Some(f64::INFINITY), true).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 319: `let block_diag = compat::block_diag(&blocks).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/scipy_migration_guide.rs

4 issues found:

- Line 211: `let residual_norm = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 245: `let residual_norm_lstsq = residual_lstsq.iter().map(|&r| r * r).sum::<f64>().sqr...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 283: `let error = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 420: `println!("            Ok(result) => {{ /* use result */ }}");`
  - **Fix**: Division without zero check - use safe_divide()

### examples/simd_quantization_example.rs

9 issues found:

- Line 39: `let c_q_simd = simd_quantized_matmul(&a_q, &a_params, &b_q, &b_params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 47: `let rel_error = max_error / c_ref.fold(0.0_f32, |acc, &x| acc.max(x.abs()));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 52: `ref_time.as_secs_f64() / simd_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 67: `let r_q_simd = simd_quantized_matvec(&a_q, &a_params, &v.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 75: `let rel_error = max_error / r_ref.fold(0.0_f32, |acc, &x| acc.max(x.abs()));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 80: `ref_time.as_secs_f64() / simd_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 165: `let c_q = simd_quantized_matmul(&a_q, &a_params, &b_q, &b_params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 172: `let rel_error = abs_error / max_abs_val;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 175: `let speedup = ref_time.as_secs_f64() / q_time.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()

### examples/sparse_dense_example.rs

19 issues found:

- Line 36: `let sparse_a = sparse_from_ndarray(&dense_a.view(), 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 45: `(sparse_a.nnz() as f64 / (sparse_a.rows * sparse_a.cols) as f64) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 50: `let result_ab = sparse_dense_matmul(&sparse_a, &dense_b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 58: `let error_ab = (&result_ab - &expected_ab).mapv(|x: f64| x.abs()).sum() / expect...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 63: `let result_ac = sparse_dense_matvec(&sparse_a, &vec_c.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 71: `let error_ac = (&result_ac - &expected_ac).mapv(|x: f64| x.abs()).sum() / expect...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 82: `let result_ad_add = sparse_dense_add(&sparse_a, &dense_d.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 93: `/ expected_ad_add.sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 98: `let result_ad_mul = sparse_dense_elementwise_mul(&sparse_a, &dense_d.view()).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 110: `/ expected_ad_mul.sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 115: `let sparse_a_t = sparse_transpose(&sparse_a).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 127: `/ expected_a_t.sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 143: `100.0 / (k as f64)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 154: `dense_large[[i, j]] = (i * j) as f64 / (n * m) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 163: `(nnz as f64 / (n * m) as f64) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 167: `let sparse_large = sparse_from_ndarray(&dense_large.view(), 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 178: `let _ = sparse_dense_matvec(&sparse_large, &dense_vec.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 194: `dense_time.as_secs_f64() / sparse_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 199: `sparse_time.as_secs_f64() / dense_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()

### examples/sparse_eigen_solver.rs

7 issues found:

- Line 29: `let (eigenvalues, eigenvectors) = largest_k_eigh(&a.view(), 5, 1000, 1e-8).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 44: `let residual = (&av - &lambda_v).mapv(|x| x * x).sum().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 51: `let (eigenvalues, eigenvectors) = smallest_k_eigh(&a.view(), 5, 1000, 1e-8).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 66: `let residual = (&av - &lambda_v).mapv(|x| x * x).sum().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 73: `let (all_eigenvalues, _) = eigh(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 78: `sorted_eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 84: `sorted_eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/structured_matrices_example.rs

21 issues found:

- Line 18: `let toeplitz = ToeplitzMatrix::new(first_row.view(), first_col.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 21: `let dense_toeplitz = toeplitz.to_dense().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 27: `let y = toeplitz.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 35: `let toeplitz_sym = ToeplitzMatrix::new_symmetric(first_row_sym.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 38: `println!("{}", toeplitz_sym.to_dense().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 45: `let circulant = CirculantMatrix::new(first_row.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 48: `let dense_circulant = circulant.to_dense().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 53: `let y = circulant.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 66: `let hankel = HankelMatrix::new(first_col.view(), last_row.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 69: `let dense_hankel = hankel.to_dense().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 75: `let y = hankel.matvec(&x_hankel.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 83: `let hankel_seq = HankelMatrix::from_sequence(sequence.view(), 3, 3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 86: `println!("{}", hankel_seq.to_dense().unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 95: `let y_op = toeplitz_op.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 102: `let direct_y = toeplitz.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 119: `let x = solve_toeplitz(c.view(), r.view(), b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 128: `let toeplitz = ToeplitzMatrix::new(c.view(), r.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 129: `let tx = toeplitz.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 143: `let x = solve_circulant(c.view(), b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 151: `let circulant = CirculantMatrix::new(c.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 152: `let cx = circulant.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/tensor_contraction_example.rs

8 issues found:

- Line 18: `let c = contract(&a.view(), &b.view(), &[1], &[0]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 38: `let batch_c = batch_matmul(&batch_a.view(), &batch_b.view(), 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 51: `let d = einsum("ij,jk->ik", &[&a.view().into_dyn(), &b.view().into_dyn()]).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 62: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 70: `let trace = einsum("ii->", &[&matrix.view().into_dyn()]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 84: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 100: `let (core, factors) = hosvd(&tensor.view(), &[2, 2, 2]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 125: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### examples/tensor_train_example.rs

9 issues found:

- Line 22: `let tt = tensor_train_decomposition(&tensor.view(), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 31: `let reconstructed = tt.to_full().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 61: `let tt_truncated = tensor_train_decomposition(&tensor4d.view(), Some(2), None).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 70: `let reconstructed4d = tt_truncated.to_full().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 84: `let value = tt.get(&indices).unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 97: `let rounded_tt = tt.round(eps).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 103: `let reconstructed = rounded_tt.to_full().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 123: `(diff_sum / orig_sum).sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 123: `(diff_sum / orig_sum).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation

### examples/tensor_train_high_dimensional_example.rs

11 issues found:

- Line 98: `dense_tensor.len() * 8 / 1024`
  - **Fix**: Division without zero check - use safe_divide()
- Line 116: `(1.0 - tt_result.storage_size() as f64 / dense_tensor.len() as f64) * 100.0`
  - **Fix**: Division without zero check - use safe_divide()
- Line 127: `(sample / 4) % 4,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 128: `(sample / 16) % 4,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 129: `(sample / 64) % 4,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 142: `total_error / num_samples as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 156: `let compression = full_size as f64 / estimated_tt_size as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 162: `full_size as f64 * 8.0 / 1e9`
  - **Fix**: Division without zero check - use safe_divide()
- Line 167: `estimated_tt_size as f64 * 8.0 / 1e6`
  - **Fix**: Division without zero check - use safe_divide()
- Line 174: `full_size as f64 * 8.0 / 1e12`
  - **Fix**: Division without zero check - use safe_divide()
- Line 265: `high_rank_tensor.storage_size() as f64 / rounded.storage_size() as f64`
  - **Fix**: Division without zero check - use safe_divide()

### examples/tutorial_advanced.rs

2 issues found:

- Line 152: `let scale = 1.0 / (d_k as f32).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 152: `let scale = 1.0 / (d_k as f32).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/attention/mod.rs

33 issues found:

- Line 603: `Some(s) => F::from(s).unwrap_or_else(|| F::from(1.0 / (head_dim as f64).sqrt())....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 604: `None => F::from(1.0 / (head_dim as f64).sqrt()).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 844: `let scale_factor = (m_prev - m_new).exp() / l_block[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1120: `let pos_diff = F::from((i as isize - j as isize).abs() as f64).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1212: `let half_dim = d_model / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1216: `let freq = F::one() / (freq_base.powf(F::from(2.0 * i as f64 / d_model as f64).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1231: `let theta = F::from(pos as f64).unwrap() * freqs[i];`
  - **Fix**: Use .get() with proper bounds checking
- Line 1342: `result[[b, i, j]] = sum / z[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1522: `let heads_per_kv = num_heads / num_kv_heads;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1523: `let head_dim = d_model / num_heads;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1583: `let kv_head_idx = h / heads_per_kv;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1677: `let pos_i = F::from(i as f64 + 1.0).unwrap(); // 1-indexed position`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1680: `let dim_factor = F::from(j as f64 / d_model as f64).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1681: `let scale_factor = F::one() / pos_i.powf(dim_factor);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1681: `let scale_factor = F::one() / pos_i.powf(dim_factor);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 1687: `let pos_i = F::from(i as f64 + 1.0).unwrap(); // 1-indexed position`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1690: `let dim_factor = F::from(j as f64 / d_model as f64).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1691: `let scale_factor = F::one() / pos_i.powf(dim_factor);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1691: `let scale_factor = F::one() / pos_i.powf(dim_factor);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 1715: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1718: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1721: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1724: `let scale = 1.0 / (2.0_f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1724: `let scale = 1.0 / (2.0_f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1726: `let result = attention(&query.view(), &key.view(), &value.view(), None, scale).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1733: `let expected_first_pos = [(5.0 + 7.0) / 2.0, (6.0 + 8.0) / 2.0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1734: `let expected_second_pos = [(5.0 + 7.0) / 2.0, (6.0 + 8.0) / 2.0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1748: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1751: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1754: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1756: `let scale = 1.0 / (2.0_f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1756: `let scale = 1.0 / (2.0_f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1758: `let result = causal_attention(&query.view(), &key.view(), &value.view(), scale)....`
  - **Fix**: Replace with ? operator or .ok_or()

### src/autograd/batch.rs

13 issues found:

- Line 98: `let grad_3d = grad.clone().into_shape((batch_size, n, p)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 99: `let b_3d = b_data.clone().into_shape((batch_size, m, p)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 131: `let grad_3d = grad.clone().into_shape((batch_size, n, p)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 132: `let a_3d = a_data.clone().into_shape((batch_size, n, m)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 262: `let grad_2d = grad.clone().into_shape((batch_size, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 263: `let x_2d = x_data.clone().into_shape((batch_size, m)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 291: `let grad_2d = grad.clone().into_shape((batch_size, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 292: `let a_3d = a_data.clone().into_shape((batch_size, n, m)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 398: `let inv_det = F::one() / det_val;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 401: `result_data[[batch_idx, 0, 0]] = F::one() / matrix[[0, 0]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 422: `let grad_3d = grad.clone().into_shape((batch_size, n, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 423: `let inv_3d = inv_data.clone().into_shape((batch_size, n, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 545: `let grad_2d = grad.clone().into_shape((batch_size, 1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/autograd/factorizations.rs

19 issues found:

- Line 54: `let mut u = a.data.clone().into_shape((n, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 80: `l[[1, 0]] = u[[1, 0]] / u[[0, 0]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 106: `let grad_u_2d = grad_u.clone().into_shape((n, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 171: `let mut r = a.data.clone().into_shape((m, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 176: `let alpha = -x[0_usize].signum() * x.mapv(|v| v * v).sum().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 179: `let u_norm = u.mapv(|v| v * v).sum().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 182: `u.mapv_inplace(|v| v / u_norm);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 192: `r[[i, j]] = r[[i, j]] - F::from(2.0).unwrap() * u[i] * dot_product;`
  - **Fix**: Use .get() with proper bounds checking
- Line 200: `q[[i, j]] = identity - F::from(2.0).unwrap() * u[i] * u[j];`
  - **Fix**: Use .get() with proper bounds checking
- Line 226: `let grad_r_2d = grad_r.clone().into_shape((m, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 227: `let q_2d = q_data_clone.clone().into_shape((m, m)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 331: `l[[0, 0]] = a.data[[0, 0]].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 333: `l[[0, 0]] = a.data[[0, 0]].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 334: `l[[1, 0]] = a.data[[1, 0]] / l[[0, 0]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 335: `l[[1, 1]] = (a.data[[1, 1]] - l[[1, 0]] * l[[1, 0]]).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 352: `let grad_l_2d = grad_l.clone().into_shape((n, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 353: `let l_2d = l_data_clone.clone().into_shape((n, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 367: `grad_a[[j, j]] = (grad_l_2d[[j, j]] - sum) / (F::from(2.0).unwrap() * l_2d[[j, j...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 374: `grad_a[[i, j]] = (grad_l_2d[[i, j]] - sum) / l_2d[[j, j]];`
  - **Fix**: Division without zero check - use safe_divide()

### src/autograd/matrix_calculus.rs

22 issues found:

- Line 178: `let result = x.mapv(|elem| elem / norm_val);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 290: `let result = (term1 + term2) * F::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 360: `/ (F::from(2.0).unwrap() * spacing);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 362: `/ (F::from(2.0).unwrap() * spacing);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 402: `/ (F::from(2.0).unwrap() * spacing);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 404: `/ (F::from(2.0).unwrap() * spacing);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 409: `/ (F::from(2.0).unwrap() * spacing);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 411: `/ (F::from(2.0).unwrap() * spacing);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 416: `/ (F::from(2.0).unwrap() * spacing);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 418: `/ (F::from(2.0).unwrap() * spacing);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 423: `/ (F::from(2.0).unwrap() * spacing);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 425: `/ (F::from(2.0).unwrap() * spacing);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 459: `- F::from(2.0).unwrap() * field[[i, j, comp]]`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 461: `/ spacing_sq;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 463: `- F::from(2.0).unwrap() * field[[i, j, comp]]`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 465: `/ spacing_sq;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 499: `/ (F::from(2.0).unwrap() * spacing);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 504: `/ (F::from(2.0).unwrap() * spacing);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 637: `let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 656: `grad[[i, j]] = (f_plus - f_minus) / (F::from(2.0).unwrap() * eps);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 680: `let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 703: `(f_plus[[p_idx, q_idx]] - f_x[[p_idx, q_idx]]) / eps;`
  - **Fix**: Division without zero check - use safe_divide()

### src/autograd/mod.rs

2 issues found:

- Line 95: `ag::ndarray::Array2::from_shape_vec((n, n), eye_data).unwrap(),`
  - **Fix**: Handle array creation errors properly
- Line 114: `ag::ndarray::Array2::from_shape_vec((n, n), eye_data).unwrap(),`
  - **Fix**: Handle array creation errors properly

### src/autograd/special.rs

48 issues found:

- Line 85: `let discriminant = trace * trace - F::from(4.0).unwrap() * det;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 93: `let sqrt_disc = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 94: `s_squared[0] = (trace + sqrt_disc) / F::from(2.0).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 96: `s_squared[1] = (trace - sqrt_disc) / F::from(2.0).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 130: `let norm = norm_sq.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 134: `v[[i, j]] = v[[i, j]] / norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 143: `s[i] = s_squared[i].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 156: `u[[i, j]] = sum / s[j];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 167: `let default_rcond = F::from(1e-15).unwrap().sqrt();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 176: `s_inv[i] = F::one() / s[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 210: `let grad_2d = grad.clone().into_shape((n, m)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 211: `let pinv_2d = pinv_data.clone().into_shape((n, m)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 294: `result[[0, 0]] = a.data[[0, 0]].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 308: `let discriminant = trace * trace - F::from(4.0).unwrap() * det;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 316: `let sqrt_disc = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 317: `let lambda1 = (trace + sqrt_disc) / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 318: `let lambda2 = (trace - sqrt_disc) / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 349: `let norm1 = (v[[0, 0]] * v[[0, 0]] + v[[1, 0]] * v[[1, 0]]).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 350: `let norm2 = (v[[0, 1]] * v[[0, 1]] + v[[1, 1]] * v[[1, 1]]).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 353: `v[[0, 0]] = v[[0, 0]] / norm1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 354: `v[[1, 0]] = v[[1, 0]] / norm1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 358: `v[[0, 1]] = v[[0, 1]] / norm2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 359: `v[[1, 1]] = v[[1, 1]] / norm2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 371: `let inv_det_v = F::one() / det_v;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 380: `d_sqrt[[0, 0]] = lambda1.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 381: `d_sqrt[[1, 1]] = lambda2.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 416: `let grad_2d = grad.clone().into_shape((n, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 417: `let sqrtm_2d = sqrtm_data.clone().into_shape((n, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 422: `inv[[0, 0]] = F::one() / sqrtm_2d[[0, 0]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 433: `let inv_det = F::one() / det;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 450: `q[[i, j]] = sum / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 540: `result[[0, 0]] = a.data[[0, 0]].ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 554: `let discriminant = trace * trace - F::from(4.0).unwrap() * det;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 562: `let sqrt_disc = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 563: `let lambda1 = (trace + sqrt_disc) / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 564: `let lambda2 = (trace - sqrt_disc) / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 596: `let norm1 = (v[[0, 0]] * v[[0, 0]] + v[[1, 0]] * v[[1, 0]]).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 597: `let norm2 = (v[[0, 1]] * v[[0, 1]] + v[[1, 1]] * v[[1, 1]]).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 600: `v[[0, 0]] = v[[0, 0]] / norm1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 601: `v[[1, 0]] = v[[1, 0]] / norm1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 605: `v[[0, 1]] = v[[0, 1]] / norm2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 606: `v[[1, 1]] = v[[1, 1]] / norm2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 618: `let inv_det_v = F::one() / det_v;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 627: `d_log[[0, 0]] = lambda1.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 628: `d_log[[1, 1]] = lambda2.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 663: `let grad_2d = grad.clone().into_shape((n, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 668: `inv[[0, 0]] = F::one() / a_data[[0, 0]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 679: `let inv_det = F::one() / det;`
  - **Fix**: Division without zero check - use safe_divide()

### src/autograd/tensor_algebra.rs

8 issues found:

- Line 113: `let grad_2d = grad.clone().into_shape((m, p)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 114: `let b_2d = b_data.clone().into_shape((n, p)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 141: `let grad_2d = grad.clone().into_shape((m, p)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 142: `let a_2d = a_data.clone().into_shape((m, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 308: `let grad_2d = grad.clone().into_shape((m, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 334: `let grad_2d = grad.clone().into_shape((m, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 450: `let grad_1d = grad.clone().into_shape(m).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 472: `let grad_1d = grad.clone().into_shape(m).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/autograd/transformations.rs

16 issues found:

- Line 64: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 65: `let a_data_2d = a.data.clone().into_shape((a_shape[0], a_shape[1])).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 92: `result[[0, 0]] = F::one() / result[[0, 0]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 106: `let inv_det = F::one() / det;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 126: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 127: `let x_data_1d = x.data.clone().into_shape(a_shape[0]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 146: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 147: `let a_t_x_data_1d = a_t_x.data.clone().into_shape(a_shape[1]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 162: `let a_data_2d = a.data.clone().into_shape((a_shape[0], a_shape[1])).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 163: `let temp_data_1d = temp.data.clone().into_shape(a_shape[1]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 273: `let grad_2d = grad.clone().into_shape((2, 2)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 345: `let grad_2d = grad.clone().into_shape((n, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 406: `let norm = norm_squared.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 407: `let unit_normal = normal.data.mapv(|x| x / norm);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 415: `result_data[[i, j]] - F::from(2.0).unwrap() * unit_normal[i] * unit_normal[j];`
  - **Fix**: Use .get() with proper bounds checking
- Line 500: `let grad_2d = grad.clone().into_shape((n, n)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/basic.rs

17 issues found:

- Line 142: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 144: `Some((norm_a / det_val.abs()).to_f64().unwrap_or(1e16))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 156: `let inv_det = F::one() / det_val;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 356: `let d = det(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 360: `let d = det(&b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 367: `let d = det(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 371: `let d = det(&b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 378: `let a_inv = inv(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 385: `let b_inv = inv(&b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 396: `let a_inv = inv(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 418: `let b_inv = inv(&b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 420: `assert_relative_eq!(b_inv[[1, 1]], 1.0 / 3.0, epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 434: `let a_0 = matrix_power(&a.view(), 0, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 441: `let a_1 = matrix_power(&a.view(), 1, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 457: `let d = det(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 468: `let d = det(&b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 478: `let d = det(&c.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/batch/attention.rs

28 issues found:

- Line 227: `F::from(1.0 / (head_dim as f64).sqrt())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 227: `F::from(1.0 / (head_dim as f64).sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 228: `.unwrap_or_else(|| F::one() / F::from(head_dim).unwrap_or(F::one()).sqrt())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 228: `.unwrap_or_else(|| F::one() / F::from(head_dim).unwrap_or(F::one()).sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 232: `F::from(1.0 / (head_dim as f64).sqrt())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 232: `F::from(1.0 / (head_dim as f64).sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 233: `.unwrap_or_else(|| F::one() / F::from(head_dim).unwrap_or(F::one()).sqrt())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 233: `.unwrap_or_else(|| F::one() / F::from(head_dim).unwrap_or(F::one()).sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 541: `let scale_factor = (m_prev - m_new).exp() / l_block[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 605: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 608: `let key = Array::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 611: `let value = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 614: `let scale = 1.0 / (2.0f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 614: `let scale = 1.0 / (2.0f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 624: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 648: `let head_dim = d_model / num_heads;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 667: `scale: Some(1.0 / (head_dim as f32).sqrt()),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 667: `scale: Some(1.0 / (head_dim as f32).sqrt()),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 682: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 717: `1.0 / (d_model as f64).sqrt(),`
  - **Fix**: Division without zero check - use safe_divide()
- Line 717: `1.0 / (d_model as f64).sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 720: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 750: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 761: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 772: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 775: `let scale = 1.0 / (2.0f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 775: `let scale = 1.0 / (2.0f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 788: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/batch/mod.rs

13 issues found:

- Line 351: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 357: `let result = batch_matmul(&batch_a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 385: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 391: `let result = batch_matvec(&batch_a.view(), &x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 415: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 421: `let result = batch_add(&batch_a.view(), &v.view(), 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 449: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 455: `let result = batch_add(&batch_a.view(), &v.view(), 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 483: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 508: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 530: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 552: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 574: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/blas.rs

1 issues found:

- Line 76: `result.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/blas_accelerated.rs

9 issues found:

- Line 363: `let factor = aug[[j, i]] / aug[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 379: `x[i] = sum / aug[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 501: `let result = dot(&x.view(), &y.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 508: `let result = norm(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 517: `let result = gemv(1.0, &a.view(), &x.view(), 0.0, &y.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 527: `let result = gemm(1.0, &a.view(), &b.view(), 0.0, &c.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 538: `let result = matmul(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 549: `let x = solve(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 562: `let a_inv = inv(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/broadcast.rs

7 issues found:

- Line 223: `let n_batch = output.len() / (a_rows * b_cols);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 353: `let n_batch = output.len() / a_rows;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 448: `let shape = a.broadcast_shape(&b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 458: `let c = broadcast_matmul_3d(&a, &b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 479: `let c = broadcast_matmul(&a, &b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 500: `let y = broadcast_matvec(&a, &x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 527: `let c = broadcast_matmul_3d(&a, &b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/circulant_toeplitz.rs

36 issues found:

- Line 135: `Ok(self.eigenvalues.as_ref().unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 146: `let angle = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 236: `x_fft[i] = b_fft[i] / eigenvals[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 330: `Ok(max_abs / min_abs)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 530: `x[0] = b[0] / self.first_row[0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 546: `y[0] = b[0] / self.first_column[0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 554: `let mut alpha = -self.first_column[1] / self.first_column[0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 563: `beta = (b[k] - beta) / self.first_column[0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 577: `gamma = -gamma / self.first_column[0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 658: `let circ = CirculantMatrix::new(first_row.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 667: `let circ = CirculantMatrix::new(first_row).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 683: `let mut circ = CirculantMatrix::new(first_row).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 686: `let x = circ.solve(&b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 689: `let result = circ.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 699: `let mut circ = CirculantMatrix::new(first_row.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 718: `let result = circ.matvec(&v.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 722: `let eigenvals = circ.compute_eigenvalues().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 742: `let toep = ToeplitzMatrix::new(first_row, first_col).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 751: `let toep = ToeplitzMatrix::symmetric(diag.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 762: `let toep = ToeplitzMatrix::new(first_row, first_col).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 778: `let toep = ToeplitzMatrix::symmetric(diag).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 781: `let x = toep.solve(&b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 784: `let result = toep.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 795: `let toep = ToeplitzMatrix::symmetric(diag).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 798: `let x = toep.solve_levinson(&b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 801: `let result = toep.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 812: `let toep = ToeplitzMatrix::new(first_row, first_col).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 815: `let result = toep.matvec(&v.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 827: `let mut circ = CirculantMatrix::new(first_row).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 829: `let cond = circ.condition_number().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 843: `let mut circ = CirculantMatrix::new(first_row).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 846: `let x = circ.solve(&b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 847: `let result = circ.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 858: `let toep = ToeplitzMatrix::symmetric(diag).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 862: `let x_fft = toep.solve(&b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 863: `let x_levinson = toep.solve_levinson(&b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/compat.rs

8 issues found:

- Line 540: `matrix_functions::sqrtm(a, 100, F::from(1e-12).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 630: `Ok(F::from(count).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 636: `.map(|&x| x.abs().powf(F::from(p).unwrap()))`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 638: `Ok(sum.powf(F::one() / F::from(p).unwrap()))`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 678: `max_singular_value * F::from(1e-15).unwrap() * F::from(a.dim().0.max(a.dim().1))...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 684: `F::one() / val`
  - **Fix**: Division without zero check - use safe_divide()
- Line 825: `max_sv * F::from(1e-15).unwrap() * F::from(a.dim().0.max(a.dim().1)).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1017: `"sqrt" => matrix_functions::sqrtm(a, 100, F::from(1e-12).unwrap()),`
  - **Fix**: Replace with ? operator or .ok_or()

### src/complex/decompositions.rs

42 issues found:

- Line 100: `lu[[i, k]] = lu[[i, k]] / lu[[k, k]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 172: `let norm = norm_sq.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 177: `q[[i, j]] = q_j[i] / Complex::<F>::new(norm, F::zero());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 244: `indices.sort_by(|&i, &j| eigenvalues[j].re.partial_cmp(&eigenvalues[i].re).unwra...`
  - **Fix**: Use .get() with proper bounds checking
- Line 252: `s[new_idx] = eigenvalues[old_idx].re.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 273: `u[[i, j]] = sum / Complex::<F>::new(s[j], F::zero());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 299: `let tolerance = F::from(1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 316: `let norm = norm_sq.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 319: `first_elem / Complex::<F>::new(first_elem.norm(), F::zero())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 335: `Complex::<F>::new(F::from(2.0).unwrap() / v_norm_sq, F::zero()) * sum;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 349: `Complex::<F>::new(F::from(2.0).unwrap() / v_norm_sq, F::zero()) * sum;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 387: `trace * trace - Complex::<F>::new(F::from(4.0).unwrap(), F::zero()) * det;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 388: `let sqrt_disc = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 391: `(trace + sqrt_disc) / Complex::<F>::new(F::from(2.0).unwrap(), F::zero());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 393: `(trace - sqrt_disc) / Complex::<F>::new(F::from(2.0).unwrap(), F::zero());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 410: `h = crate::complex::complex_matmul(&qr.r.view(), &qr.q.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 417: `q_total = crate::complex::complex_matmul(&q_total.view(), &qr.q.view()).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 444: `let tolerance = F::from(1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 475: `let norm = norm_sq.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 478: `first_elem / Complex::<F>::new(first_elem.norm(), F::zero())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 501: `let factor = Complex::<F>::new(F::from(2.0).unwrap() / v_norm_sq, F::zero());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 577: `let d = (diagonal[n - 2] - diagonal[n - 1]) / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 581: `/ (d + sign * (d.powi(2) + subdiagonal[n - 2].powi(2)).sqrt());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 581: `/ (d + sign * (d.powi(2) + subdiagonal[n - 2].powi(2)).sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 589: `let r = (g * g + s * s).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 591: `let c = g / r;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 592: `let sn = s / r;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 604: `diagonal[k] = c * c * d1 + F::from(2.0).unwrap() * c * sn * e + sn * sn * d2;`
  - **Fix**: Use .get() with proper bounds checking
- Line 605: `diagonal[k + 1] = sn * sn * d1 - F::from(2.0).unwrap() * c * sn * e + c * c * d2...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 653: `let tolerance = F::from(1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 685: `l[[i, j]] = Complex::<F>::new(diag.sqrt(), F::zero());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 692: `l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 735: `let lu_result = complex_lu(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 764: `let qr_result = complex_qr(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 776: `let should_be_i = crate::complex::complex_matmul(&qh.view(), &qr_result.q.view()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 798: `let qr = crate::complex::complex_matmul(&qr_result.q.view(), &qr_result.r.view()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 834: `let l = complex_cholesky(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 838: `let llh = crate::complex::complex_matmul(&l.view(), &lh.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 856: `let svd = complex_svd(&a.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 867: `let svd_full = complex_svd(&a.view(), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 882: `let svd_rect = complex_svd(&b.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 901: `let svd_wide = complex_svd(&c.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/complex/decompositions_backup.rs

38 issues found:

- Line 100: `lu[[i, k]] = lu[[i, k]] / lu[[k, k]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 139: `let norm = norm_sq.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 143: `q[[i, j]] = r[[i, j]] / Complex::<f64>::new(norm, F::zero());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 210: `eigenvalues[j].re.partial_cmp(&eigenvalues[i].re).unwrap()`
  - **Fix**: Use .get() with proper bounds checking
- Line 218: `s[new_idx] = eigenvalues[old_idx].re.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 238: `u[[i, j]] = sum / Complex::<f64>::new(s[j], F::zero());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 263: `let tolerance = F::from(1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 280: `let norm = norm_sq.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 283: `first_elem / Complex::<f64>::new(first_elem.norm(), F::zero())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 298: `let factor = Complex::<f64>::new(F::from(2.0).unwrap() / v_norm_sq, F::zero()) *...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 311: `let factor = Complex::<f64>::new(F::from(2.0).unwrap() / v_norm_sq, F::zero()) *...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 348: `let discriminant = trace * trace - Complex::<f64>::new(F::from(4.0).unwrap(), F:...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 349: `let sqrt_disc = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 351: `let eig1 = (trace + sqrt_disc) / Complex::<f64>::new(F::from(2.0).unwrap(), F::z...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 352: `let eig2 = (trace - sqrt_disc) / Complex::<f64>::new(F::from(2.0).unwrap(), F::z...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 365: `h = crate::complex::complex_matmul(&qr.r.view(), &qr.q.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 372: `q_total = crate::complex::complex_matmul(&q_total.view(), &qr.q.view()).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 399: `let tolerance = F::from(1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 430: `let norm = norm_sq.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 433: `first_elem / Complex::<f64>::new(first_elem.norm(), F::zero())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 456: `let factor = Complex::<f64>::new(F::from(2.0).unwrap() / v_norm_sq, F::zero());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 531: `let d = (diagonal[n - 2] - diagonal[n - 1]) / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 533: `let shift = diagonal[n - 1] - subdiagonal[n - 2].powi(2) / (d + sign * (d.powi(2...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 533: `let shift = diagonal[n - 1] - subdiagonal[n - 2].powi(2) / (d + sign * (d.powi(2...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 541: `let r = (g * g + s * s).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 543: `let c = g / r;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 544: `let sn = s / r;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 556: `diagonal[k] = c * c * d1 + F::from(2.0).unwrap() * c * sn * e + sn * sn * d2;`
  - **Fix**: Use .get() with proper bounds checking
- Line 557: `diagonal[k + 1] = sn * sn * d1 - F::from(2.0).unwrap() * c * sn * e + c * c * d2...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 603: `let tolerance = F::from(1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 635: `l[[i, j]] = Complex::<f64>::new(diag.sqrt(), F::zero());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 642: `l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 679: `let lu_result = complex_lu(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 708: `let qr_result = complex_qr(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 712: `let should_be_i = crate::complex::complex_matmul(&qh.view(), &qr_result.q.view()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 726: `let qr = crate::complex::complex_matmul(&qr_result.q.view(), &qr_result.r.view()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 744: `let l = complex_cholesky(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 748: `let llh = crate::complex::complex_matmul(&l.view(), &lh.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/complex/enhanced_ops.rs

29 issues found:

- Line 144: `let factor = lu[[i, k]] / lu[[k, k]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 243: `y[0] = Complex::new(F::from(3.0).unwrap(), F::zero());`
  - **Fix**: Use .get() with proper bounds checking
- Line 244: `y[1] = Complex::new(F::from(9.0).unwrap(), F::from(-3.0).unwrap());`
  - **Fix**: Use .get() with proper bounds checking
- Line 310: `let five = F::from(5.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 311: `let six = F::from(6.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 320: `F::from(-18.0).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 321: `F::from(-8.0).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 516: `norm = norm.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 518: `v[i] = v[i] / Complex::new(norm, F::zero());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 544: `norm = norm.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 546: `v[i] = v[i] / Complex::new(norm, F::zero());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 624: `norm = norm.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 632: `q[[i, k]] = r[[i, k]] / Complex::new(norm, F::zero());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 697: `(a[[i, j]] + ah[[i, j]]) * Complex::new(F::from(0.5).unwrap(), F::zero());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 752: `(a[[i, j]] - ah[[i, j]]) * Complex::new(F::from(0.5).unwrap(), F::zero());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 801: `Ok(sum.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 878: `(x[[i, j]] + x_inv_h[[i, j]]) * Complex::new(F::from(0.5).unwrap(), F::zero());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 889: `diff_norm = diff_norm.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 919: `factorial = factorial * F::from(p - j).unwrap() / F::from((p + q - j) * (j + 1))...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1120: `norm = norm.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1123: `q_iter[[i, k]] = r[[i, k]] / Complex::new(norm, F::zero());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1173: `let tr = trace(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1185: `let d = det(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1200: `let y = matvec(&a.view(), &x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1216: `let ip = inner_product(&x.view(), &y.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1239: `assert!(is_hermitian(&h.view(), 1e-10).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1240: `assert!(!is_hermitian(&nh.view(), 1e-10).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1257: `assert!(is_unitary(&u.view(), 1e-10).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1258: `assert!(!is_unitary(&nu.view(), 1e-10).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()

### src/complex/mod.rs

2 issues found:

- Line 104: `sum.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 163: `augmented[[i, j]] = augmented[[i, j]] / pivot;`
  - **Fix**: Division without zero check - use safe_divide()

### src/convolution/mod.rs

26 issues found:

- Line 72: `let output_h = ((height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / str...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 73: `let output_w = ((width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stri...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 208: `let output_h = ((height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / str...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 209: `let output_w = ((width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stri...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 286: `output[[batch_idx, channel_idx, h, w]] /= F::from(count).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 348: `let output_h = ((height + 2 * padding_h - pool_h) / stride_h) + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 349: `let output_w = ((width + 2 * padding_w - pool_w) / stride_w) + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 495: `let input_h = index / width;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 556: `let output_h = ((height + 2 * padding_h - kernel_h) / stride_h) + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 557: `let output_w = ((width + 2 * padding_w - kernel_w) / stride_w) + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 721: `let output_h = ((height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / str...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 722: `let output_w = ((width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stri...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1223: `let cols = im2col(&input.view(), (2, 2), (1, 1), (0, 0), (1, 1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1251: `let cols = im2col(&input.view(), (3, 3), (1, 1), (1, 1), (1, 1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1277: `let cols = im2col(&input.view(), (2, 2), (1, 1), (0, 0), (1, 1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1280: `let output = col2im(&cols.view(), (1, 1, 3, 3), (2, 2), (1, 1), (0, 0), (1, 1))....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1308: `let (output, indices) = max_pool2d(&input.view(), (2, 2), (2, 2), (0, 0)).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1337: `let (_output, indices) = max_pool2d(&input.view(), (2, 2), (2, 2), (0, 0)).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1344: `max_pool2d_backward(&grad_output.view(), &indices.view(), (1, 1, 4, 4)).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1382: `conv2d_im2col(&input.view(), &kernel.view(), None, (1, 1), (0, 0), (1, 1)).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1423: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1449: `conv2d_im2col(&input.view(), &kernel.view(), None, (1, 1), (0, 0), (1, 1)).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1463: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1496: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1524: `let grad_bias = conv2d_backward_bias(&grad_output.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1555: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/decomposition.rs

16 issues found:

- Line 562: `*norm = col.iter().fold(F::zero(), |acc, &x| acc + x * x).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 611: `let x_norm = x.iter().fold(F::zero(), |acc, &val| acc + val * val).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 618: `let v_norm = v.iter().fold(F::zero(), |acc, &val| acc + val * val).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 639: `a_copy[[i, j]] -= F::from(2.0).unwrap() * v[i - k] * dot_product;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 654: `q_sub[[i, j]] -= F::from(2.0).unwrap() * v[i - k] * v[j - k];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 670: `col_norms[j] = col_norms[j].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 693: `let x_norm = x.iter().fold(F::zero(), |acc, &val| acc + val * val).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 701: `let v_norm = v.iter().fold(F::zero(), |acc, &val| acc + val * val).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 720: `a_copy[[i, j]] -= F::from(2.0).unwrap() * v[j - k] * dot_product;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 735: `p_sub[[i, j]] -= F::from(2.0).unwrap() * v[i - k] * v[j - k];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 760: `let l = cholesky(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 781: `let (p, l, u) = lu(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 796: `let (q, r) = qr(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 819: `let (z, t) = schur(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 846: `let (q, _a_decomp, b_decomp, z) = qz(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 879: `let (q, r, p) = complete_orthogonal_decomposition(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/decomposition_advanced.rs

10 issues found:

- Line 88: `A::from(std::f64::consts::PI / 4.0).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 90: `((apq + aqp) / (app - aqq)).atan() * A::from(0.5).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 137: `indices.sort_by(|&i, &j| s[j].partial_cmp(&s[i]).unwrap());`
  - **Fix**: Use .get() with proper bounds checking
- Line 261: `s_pinv[i] = A::one() / s_inv[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 269: `let u_new = (&u + &ut_inv) * A::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 351: `let (u, s, vt) = jacobi_svd(&a.view(), 100, 1e-14).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 389: `let (u, p_opt) = polar_decomposition(&a.view(), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 390: `let p = p_opt.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 413: `let (u, p) = polar_decomposition_newton(&a.view(), 10, 1e-12).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 437: `let (q, _r, p, rank) = qr_with_column_pivoting(&a.view(), 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/diagnostics.rs

21 issues found:

- Line 133: `let zero_threshold = F::epsilon() * F::from(1000.0).unwrap(); // More generous z...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 150: `diagnostics.frobenius_norm = frobenius_sum.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 152: `F::from(near_zero_count).unwrap() / F::from(total_elements).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 190: `if cond > F::from(1e12).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 194: `} else if cond > F::from(1e6).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 206: `if det_val.abs() < F::from(1e-8).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 222: `if diagnostics.max_abs_value / diagnostics.min_abs_value > F::from(1e15).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 230: `if diagnostics.sparsity_ratio > F::from(0.5).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 243: `} else if max_diag / min_diag > F::from(1e12).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 268: `} else if det_val.abs() < F::from(1e-10).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 288: `if (a[[i, j]] - a[[j, i]]).abs() > F::epsilon() * F::from(100.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 320: `let condition_est = norm_a * norm_a / det_a.abs();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 321: `if condition_est > F::from(1e12).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 345: `Ok((max_diag / min_diag) * norm_a)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 432: `let threshold = F::epsilon() * F::from(100.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 475: `if cond > F::from(1e14).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 483: `} else if cond > F::from(1e10).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 495: `let scale_ratio = diagnostics.max_abs_value / diagnostics.min_abs_value;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 509: `if diagnostics.sparsity_ratio > F::from(0.7).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 550: `if a[[i, i]].abs() < F::epsilon() * F::from(1000.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 559: `if det_val.abs() < F::epsilon() * F::from(1000.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()

### src/distributed/communication.rs

14 issues found:

- Line 280: `let rows_per_node = rows / self.size;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 334: `self.message_buffer.lock().unwrap().clear();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 349: `self.stats.lock().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 355: `let mut counter = self.sequence_counter.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 379: `let mut buffer = self.message_buffer.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 386: `let mut buffer = self.message_buffer.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 440: `let mut stats = self.stats.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 447: `let mut stats = self.stats.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 475: `self.bytes_sent as f64 / self.total_send_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 484: `self.bytes_received as f64 / self.total_recv_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 494: `let avg_time = (self.total_send_time + self.total_recv_time).as_secs_f64() / tot...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 539: `let comm = DistributedCommunicator::new(&config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 543: `let serialized = comm.serialize_matrix(&matrix.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 544: `let deserialized: Array2<f64> = comm.deserialize_matrix(&serialized).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/distributed/computation.rs

14 issues found:

- Line 91: `let mut load_balancer = self.load_balancer.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 105: `let mut load_balancer = self.load_balancer.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 196: `self.metrics.lock().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 201: `let mut metrics = self.metrics.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 223: `if communication_cost < computation_cost / 10 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 272: `let mut metrics = self.metrics.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 279: `let mut metrics = self.metrics.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 323: `self.total_computation_time / self.operation_count as u32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 332: `self.operation_count as f64 / self.total_computation_time.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 341: `self.operation_count as f64 / self.peak_memory_usage as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 389: `let mut queue = self.operation_queue.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 420: `let mut queue = self.operation_queue.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 489: `let engine = DistributedComputationEngine::new(config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 531: `let next = scheduler.next_operation().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/distributed/coordination.rs

16 issues found:

- Line 84: `let mut state = self.sync_state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 112: `let mut state = self.sync_state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 137: `let state = self.sync_state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 151: `let mut state = self.sync_state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 167: `state = self.sync_state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 248: `let mut state = self.state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 278: `let mut state = self.state.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 343: `let mut arrived = self.arrived_nodes.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 358: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 371: `let mut arrived = self.arrived_nodes.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 448: `Some((node - 1) / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 527: `self.total_sync_time / (self.barrier_count + self.checkpoint_count) as u32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 537: `self.active_nodes as f64 / total_nodes as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 553: `let coordinator = DistributedCoordinator::new(&config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 562: `let mut lock = DistributedLock::new("test_lock".to_string(), 0, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 588: `let reduction = ReductionCoordination::new(ReductionOperation::Sum, 0, 4).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/distributed/decomposition.rs

7 issues found:

- Line 429: `let tolerance = T::from(1e-12).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 493: `max_singular_value * T::from(1e-12).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 518: `Ok(max_sv / min_sv)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 531: `let dist_matrix = DistributedMatrix::from_local(matrix, config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 550: `let dist_matrix = DistributedMatrix::from_local(matrix, config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 565: `let mut dist_matrix = DistributedMatrix::from_local(matrix, config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 580: `let dist_matrix = DistributedMatrix::from_local(matrix, config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/distributed/distribution.rs

22 issues found:

- Line 91: `let rows_per_node = global_rows / num_nodes;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 150: `let cols_per_node = global_cols / num_nodes;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 213: `let grid_rows = (global_rows + block_rows - 1) / block_rows;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 214: `let grid_cols = (global_cols + block_cols - 1) / block_cols;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 217: `let proc_grid_rows = (num_nodes as f64).sqrt() as usize;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 218: `let proc_grid_cols = (num_nodes + proc_grid_rows - 1) / proc_grid_rows;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 220: `let proc_row = node_rank / proc_grid_cols;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 376: `let history = self.performance_history.get_mut(&node_rank).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 382: `let avg_performance = history.iter().sum::<f64>() / history.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 396: `let work_fraction = capability / total_capability;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 411: `let avg_workload = workloads.iter().sum::<f64>() / workloads.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 415: `let imbalance_ratio = (max_workload - min_workload) / avg_workload;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 454: `let avg_workload = workloads.iter().sum::<f64>() / workloads.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 463: `.sum::<f64>() / workloads.len() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 465: `let coefficient_of_variation = variance.sqrt() / avg_workload;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 465: `let coefficient_of_variation = variance.sqrt() / avg_workload;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 468: `1.0 / (1.0 + coefficient_of_variation)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 571: `let distribution = DataDistribution::row_wise((100, 50), 4, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 584: `let distribution = DataDistribution::column_wise((100, 50), 4, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 599: `let mut balancer = LoadBalancer::new(&config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 617: `let distribution = DataDistribution::row_wise((10, 8), 2, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 619: `let partition = MatrixPartitioner::partition(&matrix.view(), &distribution).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/distributed/matrix.rs

6 issues found:

- Line 443: `let elements_per_node = global_length / num_nodes;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 613: `let dist_matrix = DistributedMatrix::from_local(matrix.clone(), config).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 627: `let dist_vector = DistributedVector::from_local(vector, config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 650: `let dist1 = DistributedMatrix::from_local(matrix1, config.clone()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 651: `let dist2 = DistributedMatrix::from_local(matrix2, config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 654: `let result = dist1.add(&dist2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/distributed/mod.rs

2 issues found:

- Line 268: `self.comm_time_ms as f64 / self.compute_time_ms as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 276: `self.bytes_transferred as f64 / self.comm_time_ms as f64`
  - **Fix**: Division without zero check - use safe_divide()

### src/distributed/solvers.rs

32 issues found:

- Line 39: `distributed_conjugate_gradient(a, b, 1000, T::from(1e-6).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 79: `let alpha = rsold / p_ap;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 91: `if rsnew.sqrt() < tolerance {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 96: `let beta = rsnew / rsold;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 103: `println!("CG iteration {}: residual norm = {:e}", iteration, rsnew.sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 138: `let beta = (r.dot(&r)?).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 146: `v.push(scale_vector(&r, T::one() / beta)?);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 173: `h[[j + 1, j]] = (w.dot(&w)?).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 180: `v.push(scale_vector(&w, T::one() / h[[j + 1, j]])?);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 190: `let r_norm = (h[[j, j]] * h[[j, j]] + h[[j + 1, j]] * h[[j + 1, j]]).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 191: `c[j] = h[[j, j]] / r_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 192: `s[j] = h[[j + 1, j]] / r_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 212: `y[i] = (g[i] - sum) / h[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 233: `y[i] = (g[i] - sum) / h[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 295: `let beta = (rho_new / rho) * (alpha / omega);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 305: `alpha = rho_new / r_hat.dot(&v)?;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 311: `let s_norm = (s.dot(&s)?).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 319: `omega = t.dot(&s)? / t.dot(&t)?;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 329: `let r_norm = (r.dot(&r)?).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 414: `let alpha = rzold / p.dot(&ap)?;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 425: `let r_norm = (r.dot(&r)?).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 434: `let beta = rznew / rzold;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 496: `.map(|(&xi, &di)| xi / di)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 512: `let dist_vector = DistributedVector::from_local(vector, config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 514: `let scaled = scale_vector(&dist_vector, 2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 524: `let dist_matrix = DistributedMatrix::from_local(matrix, config.clone()).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 526: `let preconditioner = JacobiPreconditioner::new(&dist_matrix).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 529: `let dist_x = DistributedVector::from_local(x, config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 531: `let result = preconditioner.apply(&dist_x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 540: `let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 544: `let dist_matrix = DistributedMatrix::from_local(matrix, config.clone()).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 545: `let dist_vector = DistributedVector::from_local(vector, config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/eigen/generalized.rs

17 issues found:

- Line 108: `let eigenvalue = Complex::new(a_val / b_val, F::zero());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 128: `eigenvalues[i] = Complex::new(alpha / beta, F::zero());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 241: `let avg = (transformed_a[[i, j]] + transformed_a[[j, i]]) / F::from(2.0).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 262: `let norm = x.dot(&bx).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 264: `let normalized_x = x.mapv(|val| val / norm);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 354: `if (b[[i, j]] - expected).abs() > F::epsilon() * F::from(10.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 396: `let (w_gen, _v_gen) = eig_gen(&a.view(), &b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 397: `let (w_std, _v_std) = eig(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 401: `w_gen_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 404: `w_std_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 420: `let (w, _v) = eig_gen(&a.view(), &b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 424: `eigenvals.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 445: `let (w, v) = eigh_gen(&a.view(), &b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 486: `let w_full = eig_gen(&a.view(), &b.view(), None).unwrap().0;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 487: `let w_vals_only = eigvals_gen(&a.view(), &b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 501: `let w_full = eigh_gen(&a.view(), &b.view(), None).unwrap().0;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 502: `let w_vals_only = eigvalsh_gen(&a.view(), &b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/eigen/mod.rs

43 issues found:

- Line 165: `let discriminant = trace * trace - F::from(4.0).unwrap() * det;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 168: `let sqrt_disc = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 169: `let lambda1 = (trace + sqrt_disc) / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 170: `let lambda2 = (trace - sqrt_disc) / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 179: `let norm1 = (v1_1 * v1_1 + v1_2 * v1_2).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 180: `eigenvectors[[0, 0]] = v1_1 / norm1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 181: `eigenvectors[[1, 0]] = v1_2 / norm1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 191: `let norm2 = (v2_1 * v2_1 + v2_2 * v2_2).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 192: `eigenvectors[[0, 1]] = v2_1 / norm2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 193: `eigenvectors[[1, 1]] = v2_2 / norm2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 236: `let residual_norm = residual.dot(&residual).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 246: `eigenvalues[i] = vt_av / vt_v;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 261: `let norm = vi_new.dot(&vi_new).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 269: `let norm = vi.dot(&vi).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 272: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 338: `F::from(1e12).unwrap() // Matrix is singular or nearly singular`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 340: `max_sv / min_sv`
  - **Fix**: Division without zero check - use safe_divide()
- Line 349: `let n_f = F::from(n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 350: `(norm_2 * norm_1) / n_f`
  - **Fix**: Division without zero check - use safe_divide()
- Line 367: `F::from(1e12).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 369: `max_diag / min_diag`
  - **Fix**: Division without zero check - use safe_divide()
- Line 403: `let base_tol = F::epsilon() * F::from(100.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 406: `if condition_number > F::from(1e12).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 407: `base_tol * F::from(1000.0).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 408: `} else if condition_number > F::from(1e8).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 409: `base_tol * F::from(100.0).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 410: `} else if condition_number > F::from(1e4).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 411: `base_tol * F::from(10.0).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 429: `let (w1, v1) = eig(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 430: `let (w2, v2) = standard::eig(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 437: `let (w1, v1) = eigh(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 438: `let (w2, v2) = standard::eigh(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 444: `let w1 = eigvals(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 445: `let w2 = standard::eigvals(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 450: `let w1 = eigvalsh(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 451: `let (w2, _) = eigh(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 464: `let (w1, v1) = eig_gen(&a.view(), &b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 465: `let (w2, v2) = generalized::eig_gen(&a.view(), &b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 470: `let (w1, v1) = eigh_gen(&a.view(), &b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 471: `let (w2, v2) = generalized::eigh_gen(&a.view(), &b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 485: `let (w, v) = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 521: `let _ = standard::eig(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 525: `let _ = generalized::eig_gen(&a.view(), &b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/eigen/standard.rs

29 issues found:

- Line 172: `b.mapv_inplace(|x| x / norm_b);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 197: `b[i] = b_new[i] / norm_b_new;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 302: `let discriminant = trace * trace - F::from(4.0).unwrap() * det;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 310: `let sqrt_discriminant = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 311: `let lambda1 = (trace + sqrt_discriminant) / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 312: `let lambda2 = (trace - sqrt_discriminant) / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 344: `eigenvector.mapv_inplace(|x| x / norm);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 356: `let real_part = trace / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 357: `let imag_part = (-discriminant).sqrt() / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 385: `let norm = norm_sq.re.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 413: `let discriminant = trace * trace - F::from(4.0).unwrap() * det;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 414: `let sqrt_discriminant = discriminant.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 417: `let lambda1 = (trace + sqrt_discriminant) / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 418: `let lambda2 = (trace - sqrt_discriminant) / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 455: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 482: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 505: `let tol = F::from(1e-12).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 534: `indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());`
  - **Fix**: Use .get() with proper bounds checking
- Line 562: `let tol = F::from(1e-12).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 595: `indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());`
  - **Fix**: Use .get() with proper bounds checking
- Line 619: `let tol = F::epsilon() * F::from(100.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 694: `let (eigenvalues, eigenvectors) = eig(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 706: `let (eigenvalues, _eigenvectors) = eig(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 713: `let (eigenvalues, _) = eigh(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 726: `let (eigenvalues, eigenvectors) = eigh(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 745: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 748: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 764: `let (eigenvalue, eigenvector) = power_iteration(&a.view(), 100, 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 770: `let norm = (eigenvector[0] * eigenvector[0] + eigenvector[1] * eigenvector[1]).s...`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/eigen/tridiagonal.rs

10 issues found:

- Line 56: `let r = (a * a + b * b).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 60: `(a / r, -b / r)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 65: `let tol = F::epsilon().sqrt() * eigenvalues.iter()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 98: `let mut g = (eigenvalues[l] - shift) / (F::from(2.0).unwrap() * e[l]);`
  - **Fix**: Use .get() with proper bounds checking
- Line 99: `let mut r = (F::one() + g * g).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 104: `g = eigenvalues[l] - shift + e[l] / (g + r);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 124: `r = (eigenvalues[i] - g) * s + F::from(2.0).unwrap() * c * b;`
  - **Fix**: Use .get() with proper bounds checking
- Line 146: `sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 220: `if i != j && temp[[i, j]].abs() > F::epsilon() * F::from(100.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 242: `indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());`
  - **Fix**: Use .get() with proper bounds checking

### src/eigen_specialized.rs

23 issues found:

- Line 225: `let mut g = (d[l + 1] - d[l]) / (F::from(2.0).unwrap() * e[l]);`
  - **Fix**: Use .get() with proper bounds checking
- Line 226: `let mut r = (g * g + F::one()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 230: `g = d[m] - d[l] + e[l] / (g + r);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 241: `r = (f * f + g * g).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 250: `s = f / r;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 251: `c = g / r;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 253: `r = (d[i] - g) * s + F::from(2.0).unwrap() * c * b;`
  - **Fix**: Use .get() with proper bounds checking
- Line 392: `let theta = F::from(-2.0 * std::f64::consts::PI).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 393: `* F::from(k).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 394: `* F::from(j).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 395: `/ F::from(n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 440: `if (matrix[[i, j]] - matrix[[j, i]]).abs() > F::epsilon() * F::from(1000.0).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 491: `let tol = tol.unwrap_or(F::epsilon() * F::from(1000.0).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 502: `q_matrix[[i, 0]] = F::from(rng.random_range(-1.0..=1.0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 510: `norm = norm.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 547: `beta[j] = beta[j].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 552: `q_matrix[[i, j + 1]] = w[i] / beta[j];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 664: `let x_norm = x.iter().fold(F::zero(), |acc, &val| acc + val * val).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 670: `let v_norm = v.iter().fold(F::zero(), |acc, &val| acc + val * val).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 713: `tridiagonal_eigen(&diagonal.view(), &sub_diagonal.view(), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 730: `tridiagonal_eigen(&diagonal.view(), &sub_diagonal.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 740: `let eigenvals = circulant_eigenvalues(&first_col.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 755: `partial_eigen(&matrix.view(), 2, "largest", None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/eigen_specialized/banded.rs

4 issues found:

- Line 52: `if diff > F::epsilon() * F::from(10.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 101: `let r = (x * x + y * y).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 102: `let c = x / r;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 103: `let s = -y / r;`
  - **Fix**: Division without zero check - use safe_divide()

### src/eigen_specialized/sparse.rs

10 issues found:

- Line 271: `b.mapv_inplace(|x| x / norm_b);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 293: `b_new.mapv_inplace(|x| x / norm_b_new);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 343: `y[i] = (b_perm[i] - sum) / l[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 353: `x[i] = (y[i] - sum) / u[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 368: `let (eigenvalues, eigenvectors) = largest_k_eigh(&a.view(), 2, 100, 1e-10).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 406: `let (eigenvalues, eigenvectors) = smallest_k_eigh(&a.view(), 2, 100, 1e-10).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 447: `power_iteration_with_convergence(&a.view(), 100, 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 453: `let norm = vector_norm(&eigenvector.view(), 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 458: `let expected_val = 1.0 / 2.0_f64.sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 458: `let expected_val = 1.0 / 2.0_f64.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/eigen_specialized/symmetric.rs

6 issues found:

- Line 41: `if diff > F::epsilon() * F::from(10.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 84: `if diff > F::epsilon() * F::from(10.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 128: `alpha = alpha.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 156: `let vnorm = v.iter().map(|&x| x * x).sum::<F>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 184: `- F::from(2.0).unwrap() * (v[j] * w[k] + w[j] * v[k])`
  - **Fix**: Use .get() with proper bounds checking
- Line 185: `+ F::from(4.0).unwrap() * z * v[j] * v[k];`
  - **Fix**: Use .get() with proper bounds checking

### src/eigen_specialized/tridiagonal.rs

10 issues found:

- Line 57: `let r = (a * a + b * b).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 61: `(a / r, -b / r)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 66: `let tol = F::epsilon().sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 101: `let mut g = (eigenvalues[l] - shift) / (F::from(2.0).unwrap() * e[l]);`
  - **Fix**: Use .get() with proper bounds checking
- Line 102: `let mut r = (F::one() + g * g).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 107: `g = eigenvalues[l] - shift + e[l] / (g + r);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 127: `r = (eigenvalues[i] - g) * s + F::from(2.0).unwrap() * c * b;`
  - **Fix**: Use .get() with proper bounds checking
- Line 149: `sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 223: `if i != j && temp[[i, j]].abs() > F::epsilon() * F::from(100.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 245: `indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());`
  - **Fix**: Use .get() with proper bounds checking

### src/error.rs

1 issues found:

- Line 175: `if residual / tolerance < 10.0 {`
  - **Fix**: Division without zero check - use safe_divide()

### src/extended_precision/eigen.rs

39 issues found:

- Line 106: `let tol = tol.unwrap_or(A::epsilon().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 120: `let eigenvalues_high = qr_algorithm(a_high, max_iter, I::from(tol.promote()).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 275: `if (a[[i, j]] - a[[j, i]]).abs() > A::epsilon() * A::from(10.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 284: `let tol = tol.unwrap_or(A::epsilon().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 299: `qr_algorithm_symmetric(a_high, max_iter, I::from(tol.promote()).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 386: `if (a[[i, j]] - a[[j, i]]).abs() > A::epsilon() * A::from(10.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 395: `let tol = tol.unwrap_or(A::epsilon().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 410: `qr_algorithm_symmetric_with_vectors(a_tri, q, max_iter, I::from(tol.promote()).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 447: `a[[i, k]] = a[[i, k]] / scale;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 452: `let g = if f >= I::zero() { -h.sqrt() } else { h.sqrt() };`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 462: `f = f / h;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 474: `f = f / h;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 516: `let g = if f >= I::zero() { -h.sqrt() } else { h.sqrt() };`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 588: `v[i] = a_copy[[i + k + 1, k]] / scale;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 597: `let g = if f >= I::zero() { -h.sqrt() } else { h.sqrt() };`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 694: `let r_norm = (alpha * alpha + beta * beta).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 696: `let c = alpha / r_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 697: `let s = -beta / r_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 797: `let g = (d[l + 1] - d[l]) / (I::from(2.0).unwrap() * e[l]);`
  - **Fix**: Use .get() with proper bounds checking
- Line 798: `let mut r = (g * g + I::one()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 799: `let mut g = d[m] - d[l] + e[l] / (g + if g >= I::zero() { r } else { -r });`
  - **Fix**: Division without zero check - use safe_divide()
- Line 811: `c = g / f;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 812: `r = (c * c + I::one()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 816: `s = I::one() / r;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 819: `s = f / g;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 820: `r = (s * s + I::one()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 824: `c = I::one() / r;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 829: `r = (d[i] - g) * s + I::from(2.0).unwrap() * c * b;`
  - **Fix**: Use .get() with proper bounds checking
- Line 939: `I::from(2.0).unwrap() * e[i] / h`
  - **Fix**: Use .get() with proper bounds checking
- Line 942: `let r = (t * t + I::one()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 943: `let c = I::one() / r;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 961: `d[i] = c2 * temp_i + s2 * temp_ip1 - I::from(2.0).unwrap() * cs * e[i];`
  - **Fix**: Use .get() with proper bounds checking
- Line 962: `d[i + 1] = s2 * temp_i + c2 * temp_ip1 + I::from(2.0).unwrap() * cs * e[i];`
  - **Fix**: Use .get() with proper bounds checking
- Line 1003: `norm = norm.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1030: `let eigenvalues = extended_eigvalsh::<_, f64>(&a.view(), Some(1000), Some(1e-10)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1048: `let (eigenvalues, eigenvectors) = extended_eigh::<_, f64>(&a.view(), None, None)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1052: `sorted_indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwr...`
  - **Fix**: Use .get() with proper bounds checking
- Line 1111: `let (eigenvalues, eigenvectors) = extended_eigh::<_, f64>(&a.view(), None, None)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1133: `eigenvalues_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()

### src/extended_precision/factorizations.rs

24 issues found:

- Line 115: `a_high[[i, k]] = a_high[[i, k]] / a_high[[k, k]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 267: `let norm_x = x.iter().map(|&val| val * val).sum::<I>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 277: `let norm_v = v.iter().map(|&val| val * val).sum::<I>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 282: `v[i] = v[i] / norm_v;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 293: `a_high[[i + k, j]] -= I::from(2.0).unwrap() * dot_product * v[i];`
  - **Fix**: Use .get() with proper bounds checking
- Line 305: `q_high[[i + k, j]] -= I::from(2.0).unwrap() * dot_product * v[i];`
  - **Fix**: Use .get() with proper bounds checking
- Line 407: `if (a[[i, j]] - a[[j, i]]).abs() > A::epsilon() * A::from(10.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 440: `l_high[[j, j]] = d.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 449: `l_high[[i, j]] = s / l_high[[j, j]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 531: `let tol = tol.unwrap_or(A::epsilon().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 587: `if i != j && ata[[i, j]].abs() > I::from(tol.promote()).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 605: `s_high[i] = ata[[i, i]].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 610: `indices.sort_by(|&i, &j| s_high[j].partial_cmp(&s_high[i]).unwrap());`
  - **Fix**: Use .get() with proper bounds checking
- Line 629: `u[[i, j]] += a_high[[i, l]] * sorted_v_high[[l, j]] / sorted_s_high[j];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 647: `vh[[j, i]] += a_high[[l, i]] * sorted_v_high[[l, j]] / sorted_s_high[j];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 690: `let norm = v.iter().map(|&x| x * x).sum::<I>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 693: `u_full[[i, j]] = v[i] / norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 716: `let norm = v.iter().map(|&x| x * x).sum::<I>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 719: `vh_full[[i, j]] = v[j] / norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 779: `let norm_x = x.iter().map(|&val| val * val).sum::<I>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 789: `let norm_v = v.iter().map(|&val| val * val).sum::<I>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 794: `v[i] = v[i] / norm_v;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 805: `r[[i + k, j]] -= I::from(2.0).unwrap() * dot_product * v[i];`
  - **Fix**: Use .get() with proper bounds checking
- Line 817: `q[[j, i + k]] -= I::from(2.0).unwrap() * dot_product * v[i];`
  - **Fix**: Use .get() with proper bounds checking

### src/extended_precision/mod.rs

11 issues found:

- Line 366: `let factor = a_high[[i, k]] / a_high[[k, k]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 386: `x_high[i] = (b_high[i] - sum) / a_high[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 408: `let y = extended_matvec::<_, f64>(&a.view(), &x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 420: `let c = extended_matmul::<_, f64>(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 437: `let det = extended_det::<_, f64>(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 442: `let det = extended_det::<_, f64>(&c.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 447: `let det = extended_det::<_, f64>(&d.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 458: `hilbert[[i, j]] = 1.0 / ((i + j + 1) as f32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 463: `let std_det = crate::basic::det(&hilbert.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 466: `let ext_det = extended_det::<_, f64>(&hilbert.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 486: `let x = extended_solve::<_, f64>(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/fft.rs

77 issues found:

- Line 165: `let two_pi = F::from(2.0).unwrap() * F::PI();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 168: `let angle = -two_pi * F::from(k).unwrap() / F::from(size).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 281: `let half_length = length / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 282: `let step = size / length;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 306: `let scale = Complex64::new(1.0 / size as f64, 0.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 343: `PI / n as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 345: `-PI / n as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 383: `let scale = Complex64::new(1.0 / n as f64, 0.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 454: `let output_size = n / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 483: `let expected_input_size = output_size / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 502: `for i in 1..output_size / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 671: `let factor = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 677: `let factor = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 683: `let x = 2.0 * PI * i as f64 / (n - 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 692: `let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 693: `let factor = modified_bessel_i0(beta * (1.0 - x * x).sqrt()) / i0_beta;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 693: `let factor = modified_bessel_i0(beta * (1.0 - x * x).sqrt()) / i0_beta;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 698: `let taper_len = ((alpha * n as f64) / 2.0) as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 701: `0.5 * (1.0 + (PI * i as f64 / taper_len as f64 - PI).cos())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 703: `0.5 * (1.0 + (PI * (n - 1 - i) as f64 / taper_len as f64 - PI).cos())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 711: `let center = (n - 1) as f64 / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 713: `let x = (i as f64 - center) / sigma;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 730: `term *= (x / 2.0) * (x / 2.0) / (k * k);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 756: `let angle = PI * k as f64 * (2.0 * i as f64 + 1.0) / (2.0 * n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 761: `(1.0 / n as f64).sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 761: `(1.0 / n as f64).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 763: `(2.0 / n as f64).sqrt()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 763: `(2.0 / n as f64).sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 781: `sum += input[0] * (1.0 / n as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 781: `sum += input[0] * (1.0 / n as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 785: `let angle = PI * k as f64 * (2.0 * i as f64 + 1.0) / (2.0 * n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 786: `sum += input[k] * (2.0 / n as f64).sqrt() * angle.cos();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 786: `sum += input[k] * (2.0 / n as f64).sqrt() * angle.cos();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 803: `let angle = PI * (k + 1) as f64 * (i + 1) as f64 / (n + 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 807: `result[k] = sum * (2.0 / (n + 1) as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 807: `result[k] = sum * (2.0 / (n + 1) as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 903: `let normalization = 1.0 / (fft_size as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 955: `(n - nperseg) / step + 1`
  - **Fix**: Division without zero check - use safe_divide()
- Line 967: `let output_size = fft_size / 2 + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1057: `let scale = 1.0 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1153: `let scale = 1.0 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1172: `let output_size = if real_fft { n / 2 + 1 } else { n };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1175: `let df = sample_rate / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1183: `freqs[i] = if i <= n / 2 {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1202: `let plan = FFTPlan::<f64>::new(8, FFTAlgorithm::CooleyTukey, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1217: `let result = fft_1d(&input.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1234: `let fft_result = fft_1d(&input.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1235: `let ifft_result = fft_1d(&fft_result.view(), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1246: `let result = rfft_1d(&input.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1256: `let fft_result = rfft_1d(&input.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1257: `let reconstructed = irfft_1d(&fft_result.view(), 4).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1268: `let result = fft_2d(&input.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1271: `let reconstructed = fft_2d(&result.view(), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1286: `let rect = apply_window(&signal.view(), WindowFunction::Rectangular).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1292: `let hann = apply_window(&signal.view(), WindowFunction::Hann).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1302: `let dct_result = dct_1d(&input.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1303: `let idct_result = idct_1d(&dct_result.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1313: `let dst_result = dst_1d(&input.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1323: `let result = fft_convolve(&signal1.view(), &signal2.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1332: `let signal = Array1::from_shape_fn(16, |i| (2.0 * PI * i as f64 / 16.0).sin());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1333: `let psd = periodogram_psd(&signal.view(), WindowFunction::Rectangular, None).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1341: `let signal = Array1::from_shape_fn(64, |i| (2.0 * PI * i as f64 / 8.0).sin());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1342: `let psd = welch_psd(&signal.view(), 16, 0.5, WindowFunction::Hann).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1368: `let result = bluestein_fft(&input.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1369: `let reconstructed = bluestein_fft(&result.view(), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1383: `let result = fft_3d(&input.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1386: `let reconstructed = fft_3d(&result.view(), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1409: `let windowed = apply_window(&signal.view(), WindowFunction::Kaiser(2.0)).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1420: `let windowed = apply_window(&signal.view(), WindowFunction::Tukey(0.5)).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1430: `let result = hadamard_transform(&input.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1439: `let reconstructed = hadamard_transform(&result.view(), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1448: `let result = walsh_hadamard_transform(&input.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1454: `let reconstructed = walsh_hadamard_transform(&result.view(), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1463: `let result = fast_walsh_transform(&input.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1469: `let reconstructed = fast_walsh_transform(&result.view(), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1479: `let transformed = hadamard_transform(&input.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1480: `let twice_transformed = hadamard_transform(&transformed.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/generic.rs

11 issues found:

- Line 297: `let c = gemm(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 308: `let y = gemv(&a.view(), &x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 316: `let det = gdet(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 323: `let a_inv = ginv(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 331: `let norm = gnorm(&a.view(), "fro").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 332: `let expected = (1.0 + 4.0 + 9.0 + 16.0_f32).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 339: `let svd = gsvd(&a.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 354: `let qr = gqr(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 377: `let eigen = geig(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 382: `eigenvalues_real.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 396: `let x = gsolve(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/gradient/mod.rs

26 issues found:

- Line 62: `let n = F::from(predictions.len()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 63: `let two = F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 67: `let scale = two / n;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 144: `let eps = F::from(1e-15).unwrap(); // Small epsilon to prevent division by zero`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 153: `-t / (p + eps)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 158: `(one - t) / (one - p + eps)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 230: `if (row_sum - F::one()).abs() > F::from(1e-5).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 241: `if (row_sum - F::one()).abs() > F::from(1e-6).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 251: `if (val - F::one()).abs() < F::from(1e-6).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 260: `} else if val > F::from(1e-6).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 277: `let batch_size_f = F::from(batch_size).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 346: `let two_epsilon = F::from(2.0).unwrap() * epsilon;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 365: `jacobian[[i, j]] = (f_forward[i] - f_backward[i]) / two_epsilon;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 422: `let two = F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 445: `let h_ii = (f_plus - two * f_x + f_minus) / epsilon_squared;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 478: `let four = F::from(4.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 480: `/ (four * epsilon_squared);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 503: `let gradient = mse_gradient(&predictions.view(), &targets.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 509: `assert_relative_eq!(gradient[0], 1.0 / 3.0, epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 510: `assert_relative_eq!(gradient[1], 1.0 / 3.0, epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 520: `let gradient = binary_crossentropy_gradient(&predictions.view(), &targets.view()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 538: `softmax_crossentropy_gradient(&softmax_output.view(), &targets.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 567: `let jac = jacobian(&f, &x, epsilon).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 594: `let hess = hessian(&f, &x, epsilon).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 613: `let hess = hessian(&f, &x, epsilon).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 655: `let hess = hessian(&f, &x, epsilon).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/hierarchical.rs

25 issues found:

- Line 233: `let target_rank = (max_rank.min(min_dim / 2)).max(1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 241: `if sigma / max_singular_value > tolerance {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 255: `u_scaled[[i, j]] = u[[i, j]] * s[j].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 261: `v_scaled[[i, j]] = vt[[j, i]] * s[j].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 281: `let mid = (row_cluster.start + row_cluster.end) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 288: `*row_cluster.left.clone().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 289: `*row_cluster.right.clone().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 294: `let mid = (col_cluster.start + col_cluster.end) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 301: `*col_cluster.left.clone().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 302: `*col_cluster.right.clone().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 413: `(self.size * self.size) as f64 / (total_dense_elements + total_lowrank_elements)...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 535: `let mid = (start + end) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 559: `u_gen[[i, j]] = u1[[i, j]] * s1[j].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 565: `v_gen[[i, j]] = vt1[[j, i]] * s1[j].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 662: `let mid = (start + end) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 711: `if sigma / max_sv > tolerance {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 756: `let left = tree.left.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 757: `let right = tree.right.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 793: `let h_matrix = HMatrix::from_dense(&matrix.view(), 1e-6, 2, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 798: `let y_h = h_matrix.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 808: `1.0 / (1.0 + (i as f64 - j as f64).abs())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 811: `let h_matrix = HMatrix::from_dense(&matrix.view(), 1e-4, 16, 16).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 830: `let hss_matrix = HSSMatrix::from_dense(&matrix.view(), 1e-6).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 835: `let y_hss = hss_matrix.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 852: `let h_matrix = HMatrix::from_dense(&square_matrix.view(), 1e-6, 2, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/iterative_solvers.rs

41 issues found:

- Line 68: `if (a[[i, j]] - a[[j, i]]).abs() > F::epsilon() * F::from(10.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 103: `if rsold.sqrt() < tol * b_norm {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 124: `let alpha = rsold / pap;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 142: `let current_residual = rsnew.sqrt() / b_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 142: `let current_residual = rsnew.sqrt() / b_norm;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 151: `let beta = rsnew / rsold;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 254: `x_new[i] = (b[i] - sum) / a[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 263: `let relative_residual = diff_norm / b_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 380: `x[i] = (b[i] - sum1 - sum2) / a[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 389: `let relative_residual = diff_norm / b_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 461: `if omega <= F::zero() || omega >= F::from(2.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 508: `let gauss_seidel_update = (b[i] - sum1 - sum2) / a[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 652: `let quarter = F::from(0.25).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 653: `let half = F::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 677: `p[[i_fine + 1, i]] = F::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 678: `p[[i_fine + 1, i + 1]] = F::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 947: `let beta = (rho / rho_prev) * (alpha / omega);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 968: `alpha = rho / r_hat_dot_v;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1004: `omega = t_dot_s / t_dot_t;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1094: `if (a[[i, j]] - a[[j, i]]).abs() > F::epsilon() * F::from(10.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1133: `v[0][i] = r[i] / beta[1];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1191: `let delta = (gamma[0] * gamma[0] + beta[2] * beta[2]).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1196: `c[0] = gamma[0] / delta;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1197: `s[0] = -beta[2] / delta;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1289: `x_new[i] = (b[i] - sum) / a[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1359: `let x = conjugate_gradient(&a.view(), &b.view(), 10, 1e-10, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1370: `let x = conjugate_gradient(&a.view(), &b.view(), 10, 1e-10, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1381: `let x = jacobi_method(&a.view(), &b.view(), 100, 1e-10, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1392: `let x = gauss_seidel(&a.view(), &b.view(), 100, 1e-10, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1403: `let x = successive_over_relaxation(&a.view(), &b.view(), 1.5, 100, 1e-10, None)....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1430: `let x_cg = conjugate_gradient(&a.view(), &b.view(), 100, 1e-10, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1431: `let x_jacobi = jacobi_method(&a.view(), &b.view(), 100, 1e-10, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1432: `let x_gs = gauss_seidel(&a.view(), &b.view(), 100, 1e-10, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1434: `successive_over_relaxation(&a.view(), &b.view(), 1.5, 100, 1e-10, None).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1473: `let x_mg = geometric_multigrid(&a.view(), &b.view(), 2, 5, 2, 2, 1e-6, None).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1480: `let residual_norm = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1494: `let x = minres(&a.view(), &b.view(), 100, 1e-10, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1526: `let x_indef = minres(&a_indef.view(), &b_indef.view(), 100, 1e-6, None).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1556: `let x_large = minres(&a_large.view(), &b_large.view(), 100, 1e-10, None).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1573: `let x = bicgstab(&a.view(), &b.view(), 100, 1e-10, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1597: `let x_large = bicgstab(&a_large.view(), &b_large.view(), 100, 1e-10, None).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/kronecker/mod.rs

51 issues found:

- Line 281: `let p_rows = total_rows / m_rows;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 282: `let q_cols = total_cols / n_cols;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 321: `a[[i, j]] = sum / count;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 326: `let a_norm = a.iter().map(|&x| x * x).sum::<F>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 355: `a[[i, j]] *= scaling_factor.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 361: `b[[k, l]] /= scaling_factor.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 430: `let damping_factor = damping.unwrap_or_else(|| F::from(1e-4).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 452: `a_cov[[i, j]] = sum / F::from(batch_size).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 472: `s_cov[[i, j]] = sum / F::from(batch_size).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 637: `let decay = decay_factor.unwrap_or_else(|| F::from(0.95).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 638: `let damping = base_damping.unwrap_or_else(|| F::from(1e-4).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 644: `min_damping: damping / F::from(10.0).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 645: `max_damping: damping * F::from(100.0).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 706: `let mut corrected_input = self.input_cov_avg.as_ref().unwrap().clone();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 707: `let mut corrected_output = self.output_cov_avg.as_ref().unwrap().clone();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 761: `if ratio > F::from(0.75).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 764: `(self.adaptive_damping / F::from(3.0).unwrap()).max(self.min_damping);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 765: `} else if ratio > F::from(0.25).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 768: `(self.adaptive_damping / F::from(2.0).unwrap()).max(self.min_damping);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 773: `(self.adaptive_damping / F::from(1.5).unwrap()).max(self.min_damping);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 778: `(self.adaptive_damping * F::from(2.0).unwrap()).min(self.max_damping);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 917: `y[i] = (b[i] - sum) / l[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 927: `x[i] = (y[i] - sum) / l[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 942: `regularized[[i, i]] += self.damping * F::from(10.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 951: `inv[[i, i]] = F::one() / (regularized[[i, i]] + self.damping);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1003: `extended_grads[[i, j]] = sum / F::from(batch_size).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1012: `extended_grads[[input_dim, j]] = sum / F::from(batch_size).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1050: `original_elements as f64 / (total_elements + total_inverse_elements) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1125: `let scale_factor = clip_threshold / grad_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1176: `y[i] = (b[i] - sum) / l[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1186: `x[i] = (y[i] - sum) / l[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1200: `let diag_val = matrix[[i, i]] + damping * F::from(100.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1201: `inv[[i, i]] = F::one() / diag_val;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1219: `let c = kron(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1253: `let y = kron_matvec(&a.view(), &b.view(), &x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1256: `let ab = kron(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1275: `let y = kron_matmul(&a.view(), &b.view(), &x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1278: `let ab = kron(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1297: `let ab = kron(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1300: `let (a_hat, b_hat) = kron_factorize(&ab.view(), 2, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1303: `let ab_hat = kron(&a_hat.view(), &b_hat.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1327: `kfac_factorization(&input_acts.view(), &output_grads.view(), Some(0.01)).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1364: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1390: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1399: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1452: `fisher.update_fisher(&activations, &gradients).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1459: `let preconditioned = fisher.precondition_gradients(&grad_matrices).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1494: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1521: `let inv = stable_matrix_inverse(&matrix.view(), damping).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1550: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1579: `fisher.update_fisher(&activations, &gradients).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/lapack.rs

25 issues found:

- Line 110: `Some((F::one() / max_val).to_f64().unwrap_or(1e16))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 138: `lu[[i, k]] = lu[[i, k]] / lu[[k, k]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 207: `let x_norm = x.iter().map(|&xi| xi * xi).sum::<F>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 214: `let v_norm = v.iter().map(|&vi| vi * vi).sum::<F>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 230: `r[[i, j]] -= F::from(2.0).unwrap() * v[i - k] * dot_product;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 242: `q[[i, j]] -= F::from(2.0).unwrap() * v[i - k] * dot_product;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 350: `s[new_idx] = eigenvalues[old_idx].abs().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 366: `if s[i] > F::from(1e-14).unwrap() {`
  - **Fix**: Use .get() with proper bounds checking
- Line 369: `let norm = av_col.dot(&av_col).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 370: `if norm > F::from(1e-14).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 371: `u.column_mut(i).assign(&(&av_col / norm));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 395: `if s[i] > F::from(1e-14).unwrap() {`
  - **Fix**: Use .get() with proper bounds checking
- Line 397: `let norm = atv_col.dot(&atv_col).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 398: `if norm > F::from(1e-14).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 399: `v.column_mut(i).assign(&(&atv_col / norm));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 430: `sort_indices.sort_by(|&i, &j| s[j].partial_cmp(&s[i]).unwrap());`
  - **Fix**: Use .get() with proper bounds checking
- Line 482: `let norm = col_i.dot(&col_i).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 484: `if norm > F::from(1e-14).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 534: `let norm = new_vec.dot(&new_vec).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 535: `if norm > F::from(1e-14).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 671: `l[[i, j]] = val.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 673: `l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 690: `let result = lu_factor(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 739: `let l = cholesky(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 759: `let result = qr_factor(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/lapack_accelerated.rs

29 issues found:

- Line 130: `l[[i, k]] = u[[i, k]] / u[[k, k]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 236: `r_jj = r_jj.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 247: `q[[i, j]] = a_j[i] / r_jj;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 344: `norm = norm.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 365: `norm = norm.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 372: `v[j] = new_v[j] / norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 388: `sigma = sigma.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 396: `u[[j, i]] = u_i[j] / sigma;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 509: `let disc = (trace * trace - F::from(4.0).unwrap() * det).sqrt();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 510: `let lambda1 = (trace + disc) / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 511: `let lambda2 = (trace - disc) / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 534: `let norm1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 553: `let norm2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 628: `if Float::abs(a[[i, j]] - a[[j, i]]) > F::epsilon() * F::from(100.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 659: `norm = norm.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 689: `lambda = numerator / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 692: `if (lambda - prev_lambda).abs() < F::epsilon() * F::from(10.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 702: `norm = norm.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 709: `v[i] = av[i] / norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 735: `norm = norm.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 812: `if Float::abs(a[[i, j]] - a[[j, i]]) > F::epsilon() * F::from(100.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 846: `l[[j, j]] = diag_val.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 859: `l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 876: `let (p, l, u) = lu(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 896: `let (q, r) = qr(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 927: `let (u, s, vt) = svd(&a.view(), false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 952: `let (eigenvalues, eigenvectors) = eig(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 976: `let (eigenvalues, eigenvectors) = eigh(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1001: `let l = cholesky(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/large_scale.rs

23 issues found:

- Line 64: `let normal = Normal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 67: `let scale = A::from(1.0 / (sketch_size as f64).sqrt()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 71: `s[[i, j]] = A::from(normal.sample(&mut rng)).unwrap() * scale;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 144: `let normal = Normal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 152: `v[i] = A::from(normal.sample(&mut rng)).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 158: `v.mapv_inplace(|x| x / vnorm);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 170: `v.mapv_inplace(|x| x / vnorm);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 177: `let norm = norm_sq.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 191: `let scale = A::from(total_entries / sample_size as f64).unwrap_or(A::one());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 205: `Ok((sum_sq * scale).sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 459: `let normal = Normal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 464: `q[[i, j]] = A::from(normal.sample(&mut rng)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 477: `for iter in 0..max_iterations.min(total_size / block_size) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 565: `indices.sort_by(|&i, &j| eigvals[j].partial_cmp(&eigvals[i]).unwrap());`
  - **Fix**: Use .get() with proper bounds checking
- Line 613: `let x = randomized_least_squares(&a.view(), &b.view(), 2, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 625: `let spec_norm = randomized_norm(&a.view(), "2", 20, 3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 629: `let fro_norm = randomized_norm(&a.view(), "fro", 100, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 643: `let (u, s, vt) = crate::decomposition::svd(&initial.view(), false, None).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 651: `incremental_svd(&u.view(), &s.view(), &vt.view(), &new_cols.view(), 6).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 668: `let x = block_krylov_solve(&a.view(), &b.view(), 1, 10, 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 682: `let x = ca_gmres(&a.view(), &b.view(), 2, 100, 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 686: `let res_norm = residual.dot(&residual).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 707: `let (eigvals, eigvecs) = randomized_block_lanczos(&a.view(), k, 2, 2, 10).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/lowrank.rs

17 issues found:

- Line 249: `let mean = data.column(j).sum() / F::from(n_samples).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 256: `let scale = F::from(n_samples - 1).unwrap().sqrt();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 257: `centered_data.mapv_inplace(|x| x / scale);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 277: `explained_variance.mapv(|x| x / total_variance)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 313: `let tolerance = tolerance.unwrap_or_else(|| F::from(1e-6).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 350: `w[[i, j]] = F::from(rng.random::<f64>()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 356: `h[[i, j]] = F::from(rng.random::<f64>()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 360: `let mut prev_error = F::from(f64::INFINITY).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 373: `h[[i, j]] = h[[i, j]] * numerator / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 386: `w[[i, j]] = w[[i, j]] * numerator / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 515: `col_leverage_scores.mapv_inplace(|x| x / total_leverage);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 525: `let r_f = F::from(r).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 578: `row_leverage_scores.mapv_inplace(|x| x / total_row_leverage);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 586: `let r_f = F::from(r).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 639: `s_inv[i] = F::one() / s_w[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 703: `let (components, explained_var, explained_var_ratio) = pca(&data.view(), 1, None...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 719: `let (w, h) = nmf(&a.view(), 2, Some(50), Some(1e-4), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/matrix_calculus/enhanced.rs

38 issues found:

- Line 72: `let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 89: `jvp[i] = (f_x_plus[i] - f_x[i]) / eps;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 158: `let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 176: `vjp[j] += v[i] * (f_x_plus[i] - f_x[i]) / eps;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 244: `if (x[0] - F::from(3.0).unwrap()).abs() < F::epsilon()`
  - **Fix**: Use .get() with proper bounds checking
- Line 245: `&& (x[1] - F::from(2.0).unwrap()).abs() < F::epsilon()`
  - **Fix**: Use .get() with proper bounds checking
- Line 251: `result[0] = F::from(2.0).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 252: `result[1] = F::from(4.0).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 312: `let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 328: `grad[[i, j]] = (f_x_plus - f_x) / eps;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 393: `let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 412: `jac[[k, i, j]] = (f_x_plus[k] - f_x[k]) / eps;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 482: `&& (y[0] - F::from(1.1).unwrap()).abs() < F::epsilon()`
  - **Fix**: Use .get() with proper bounds checking
- Line 483: `&& (y[1] - F::from(1.2).unwrap()).abs() < F::epsilon()`
  - **Fix**: Use .get() with proper bounds checking
- Line 486: `return Ok(F::from(2.0).unwrap()); // f(1,1) = 1^2 + 2*1^2 = 3`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 488: `return Ok(F::from(2.6).unwrap()); // First-order approx`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 490: `return Ok(F::from(2.65).unwrap()); // Second-order approx`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 500: `let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 540: `hessian[[i, j]] = (grad_plus[j] - grad[j]) / eps;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 551: `quadratic_term = quadratic_term * F::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 613: `&& (domain[[0, 0]] + F::from(3.0).unwrap()).abs() < F::epsilon()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 614: `&& (domain[[0, 1]] - F::from(3.0).unwrap()).abs() < F::epsilon()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 615: `&& (domain[[1, 0]] + F::from(5.0).unwrap()).abs() < F::epsilon()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 616: `&& (domain[[1, 1]] - F::from(1.0).unwrap()).abs() < F::epsilon()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 622: `min_point[0] = F::from(1.0).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 623: `min_point[1] = F::from(-2.0).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 656: `let t = F::from(j as f64 / (grid_points - 1) as f64).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 691: `let grad_mag = grad.iter().fold(F::zero(), |acc, &g| acc + g * g).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 717: `let jvp = jacobian_vector_product(f, &x.view(), &v.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 734: `let vjp = vector_jacobian_product(f, &x.view(), &v.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 748: `let hvp = hessian_vector_product(f, &x.view(), &v.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 764: `let grad = matrix_gradient(f, &x.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 783: `let jac = matrix_jacobian(f, &x.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 807: `let approx0 = taylor_approximation(f, &x.view(), &y.view(), 0, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 811: `let approx1 = taylor_approximation(f, &x.view(), &y.view(), 1, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 816: `let approx2 = taylor_approximation(f, &x.view(), &y.view(), 2, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 821: `let actual = f(&y.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 838: `find_critical_points(f, &domain.view(), grid_points, threshold).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/matrix_calculus/matrix_derivatives.rs

29 issues found:

- Line 259: `factorial *= F::from(k).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 284: `result = result + inner_sum * (F::one() / factorial);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 349: `if (x[[i, j]] - x[[j, i]]).abs() > F::epsilon() * F::from(100.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 359: `let eps = F::epsilon().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 371: `derivatives[i] = (eig_x_pert[i] - eig_x[i]) / eps;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 424: `Ok(x.to_owned() / norm_val)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 586: `let d_det = det_derivative(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 599: `let d_trace = trace_derivative::<f64>(None, (3, 3)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 613: `let d_trace_a = trace_derivative(Some(&a.view()), (2, 2)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 626: `let d_norm = norm_derivative(&x.view(), "fro").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 630: `assert_abs_diff_eq!(d_norm[[0, 0]], 3.0 / 5.0, epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 631: `assert_abs_diff_eq!(d_norm[[0, 1]], 4.0 / 5.0, epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 642: `let d_ab = matmul_derivative(&a.view(), &b.view(), Some(&va.view()), None).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 681: `(field[[0, 0, k + 1]] - field[[0, 0, k - 1]]) / (F::from(2.0).unwrap() * spacing...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 685: `(field[[0, 1, k + 1]] - field[[0, 1, k - 1]]) / (F::from(2.0).unwrap() * spacing...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 689: `(field[[1, 0, k + 1]] - field[[1, 0, k - 1]]) / (F::from(2.0).unwrap() * spacing...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 693: `(field[[1, 1, k + 1]] - field[[1, 1, k - 1]]) / (F::from(2.0).unwrap() * spacing...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 732: `(field[[0, 1, k + 1]] - field[[0, 1, k - 1]]) / (F::from(2.0).unwrap() * spacing...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 736: `(field[[0, 0, k + 1]] - field[[0, 0, k - 1]]) / (F::from(2.0).unwrap() * spacing...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 740: `(field[[1, 1, k + 1]] - field[[1, 1, k - 1]]) / (F::from(2.0).unwrap() * spacing...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 744: `(field[[1, 0, k + 1]] - field[[1, 0, k - 1]]) / (F::from(2.0).unwrap() * spacing...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 784: `- F::from(2.0).unwrap() * field[[i, j, k]]`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 786: `/ h_sq;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 828: `/ (F::from(2.0).unwrap() * spacing);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 833: `gradient[[i, j, 0]] = (field[[i, j, 1]] - field[[i, j, 0]]) / spacing;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 835: `(field[[i, j, n_points - 1]] - field[[i, j, n_points - 2]]) / spacing;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 862: `let div = matrix_divergence(&field, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 880: `let laplacian = matrix_laplacian(&field, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 898: `let gradient = matrix_gradient(&field, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/matrix_calculus/mod.rs

8 issues found:

- Line 92: `let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 106: `jac[[i, j]] = (f_x_plus[i] - f_x[i]) / eps;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 137: `let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 150: `grad[i] = (f_x_plus - f_x) / eps;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 179: `let eps = epsilon.unwrap_or_else(|| F::epsilon().sqrt().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 210: `let h_ij = (f_pp - f_pm - f_mp + f_mm) / (F::from(4.0).unwrap() * eps * eps);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 251: `let v_norm = v.iter().fold(F::zero(), |acc, &val| acc + val * val).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 259: `let unit_v = Array1::from_iter(v.iter().map(|&val| val / v_norm));`
  - **Fix**: Division without zero check - use safe_divide()

### src/matrix_calculus/optimization.rs

19 issues found:

- Line 33: `gradient_tolerance: F::from(1e-6).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 34: `initial_step_size: F::from(1.0).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 35: `backtrack_factor: F::from(0.5).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 112: `let grad_norm = grad.iter().fold(F::zero(), |acc, &g| acc + g * g).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 139: `f_val - f_new >= F::from(1e-4).unwrap() * step_size * grad_norm * grad_norm;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 154: `let grad_norm = grad.iter().fold(F::zero(), |acc, &g| acc + g * g).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 222: `let grad_norm = grad.iter().fold(F::zero(), |acc, &g| acc + g * g).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 249: `config.initial_step_size * F::from(0.1).unwrap() / f_change`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 263: `let grad_norm = grad.iter().fold(F::zero(), |acc, &g| acc + g * g).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 381: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 404: `let grad_norm = grad.iter().fold(F::zero(), |acc, &g| acc + g * g).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 436: `let eps = F::epsilon().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 447: `grad[[i, j]] = (f_pert - f_x) / eps;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 507: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 511: `*elem = *elem / norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 561: `let sym = (x + &x.t()) / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 565: `let reg = F::from(1e-6).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 602: `let result = matrix_gradient_descent(&objective, &x0.view(), &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 636: `let result = projected_gradient_descent(&objective, project, &x0.view(), &config...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/matrix_dynamics.rs

47 issues found:

- Line 234: `if a00.abs() < F::from(1e-14).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 235: `&& a11.abs() < F::from(1e-14).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 236: `&& (a01 + a10).abs() < F::from(1e-14).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 277: `let tolerance = F::from(1e-15).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 286: `scaled_a = &scaled_a / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 292: `term = term.dot(&scaled_a) / F::from(k).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 329: `if beta < F::from(config.atol).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 334: `let mut v = b_col.to_owned() / beta;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 356: `if j_krylov < m - 1 && norm_w > F::from(config.atol).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 358: `v = w / norm_w;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 521: `if residual_norm < F::from(config.rtol).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 526: `let step = (&ax_plus_xa + c) * F::from(-config.dt_initial).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 721: `let mut dt = F::from(config.dt_initial).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 731: `let dt_min = F::from(config.dt_min).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 732: `let dt_max = F::from(config.dt_max).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 733: `let safety = F::from(config.safety_factor).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 734: `let rtol = F::from(config.rtol).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 735: `let atol = F::from(config.atol).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 751: `let x_temp = &x + &k1 * (dt / F::from(2.0).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 752: `let k2 = f(t + dt / F::from(2.0).unwrap(), &x_temp.view()).map_err(|_| {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 756: `let x_temp = &x + &k2 * (dt / F::from(2.0).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 757: `let k3 = f(t + dt / F::from(2.0).unwrap(), &x_temp.view()).map_err(|_| {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 767: `let factor = dt / F::from(6.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 769: `+ &k2.mapv(|x| x * F::from(2.0).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 770: `+ &k3.mapv(|x| x * F::from(2.0).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 776: `let k_embedded_sum = &k1.mapv(|x| x * F::from(2.0).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 777: `+ &k2.mapv(|x| x * F::from(3.0).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 778: `+ &k3.mapv(|x| x * F::from(3.0).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 779: `let embedded_factor = dt / F::from(8.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 791: `dt = (safety * dt * (tolerance / error).powf(F::from(0.2).unwrap())).max(dt_min)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 793: `} else if error < tolerance / F::from(10.0).unwrap() && dt < dt_max {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 795: `dt = (safety * dt * (tolerance / error).powf(F::from(0.25).unwrap())).min(dt_max...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 889: `if hamiltonian[[0, 1]].abs() < F::from(1e-10).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 890: `&& hamiltonian[[1, 0]].abs() < F::from(1e-10).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 908: `if (e1 - F::one()).abs() < F::from(1e-10).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 909: `&& (e2 + F::one()).abs() < F::from(1e-10).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 910: `&& (t - F::from(std::f64::consts::PI).unwrap()).abs() < F::from(1e-10).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 992: `let result = matrix_exp_action(&a.view(), &b.view(), t, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1015: `let x = lyapunov_solve(&a.view(), &c.view(), &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1019: `let residual_norm = matrix_norm(&residual.view(), "fro", None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1032: `let x = riccati_solve(&a.view(), &b.view(), &q.view(), &r.view(), &config).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1037: `let expected = (2.0_f64).sqrt() - 1.0;`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1049: `let psi_t = quantum_evolution(&h.view(), &psi.view(), t, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1060: `let (stable, _eigs, margin) = stability_analysis(&a_stable.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1066: `let (stable, _eigs, margin) = stability_analysis(&a_unstable.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1080: `let result = matrix_ode_solve(f, &x0.view(), [0.0, 1.0], &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1087: `let final_norm = matrix_norm(&final_state.view(), "fro", None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/matrix_equations.rs

7 issues found:

- Line 333: `Ok((x.clone() + x.t()) * A::from(0.5).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 388: `let tolerance = A::from(1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 420: `return Ok((x.clone() + x.t()) * A::from(0.5).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 505: `let x = solve_sylvester(&a.view(), &b.view(), &c.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 533: `let x = solve_stein(&a.view(), &q.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 555: `let x = solve_continuous_riccati(&a.view(), &b.view(), &q.view(), &r.view()).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 569: `.dot(&crate::inv(&r.view(), None).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()

### src/matrix_factorization.rs

34 issues found:

- Line 89: `check_positive(F::from(rank).unwrap(), "rank")?;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 112: `let epsilon = F::from(1e-5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 119: `w[[i, j]] = F::from(rand::random::<f64>()).unwrap() + epsilon;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 125: `h[[i, j]] = F::from(rand::random::<f64>()).unwrap() + epsilon;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 146: `h[[i, j]] = h[[i, j]] * numerator / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 164: `w[[i, j]] = w[[i, j]] * numerator / denominator;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 180: `error = error.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 275: `let norm = col.iter().fold(F::zero(), |acc, &x| acc + x * x).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 301: `let pivot_norm = pivot.iter().fold(F::zero(), |acc, &x| acc + x * x).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 310: `/ pivot_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 314: `a_copy[[row, j]] - dot_product * a_copy[[row, i]] / pivot_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 357: `*si = F::one() / *si;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 441: `s_inv_diag[[i, i]] = F::one() / s_k[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 607: `s_w_inv[[i, i]] = F::one() / s_w[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 626: `F::from(rand::random::<f64>() * 2.0 - 1.0).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 660: `s_inv[[i, i]] = F::one() / s_k[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 693: `let rand_val = F::from(rand::random::<f64>()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 710: `let rand_val = F::from(rand::random::<f64>()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 730: `let scale = F::one() / (F::from(c_samples).unwrap() * col_leverage[col]).sqrt();`
  - **Fix**: Use .get() with proper bounds checking
- Line 737: `let scale = F::one() / (F::from(r_samples).unwrap() * row_leverage[row]).sqrt();`
  - **Fix**: Use .get() with proper bounds checking
- Line 759: `c_s_inv[[i, i]] = F::one() / c_s[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 762: `r_s_inv[[i, i]] = F::one() / r_s[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 893: `if max_norm.sqrt() <= tol {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 924: `let x_norm = x.iter().fold(F::zero(), |acc, &val| acc + val * val).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 933: `let v_norm = v.iter().fold(F::zero(), |acc, &val| acc + val * val).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 955: `r[[i, j]] -= F::from(2.0).unwrap() * v[i - k] * dot_product;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 976: `q[[i, j]] -= F::from(2.0).unwrap() * dot_product * v[j - k];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1178: `let (w, h) = nmf(&a.view(), 2, 100, 1e-4).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1205: `error = error.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1208: `assert!(error / 9.0 < 1.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1244: `let (c, u, r) = cur_decomposition(&a.view(), 2, Some(2), Some(2), "uniform").unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1267: `let (q, r, p) = rank_revealing_qr(&a.view(), 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1311: `assert!(r[[2, 2]].abs() < 1e-6 || r[[2, 2]].abs() / r[[0, 0]].abs() < 1e-6);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1320: `let (u, t, v) = utv_decomposition(&a.view(), "urv", 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/matrix_functions.rs

62 issues found:

- Line 101: `F::from(1.0).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 102: `F::from(1.0 / 2.0).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 103: `F::from(1.0 / 6.0).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 104: `F::from(1.0 / 24.0).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 105: `F::from(1.0 / 120.0).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 106: `F::from(1.0 / 720.0).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 302: `result[[0, 0]] = val.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 333: `result[[i, i]] = a[[i, i]].ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 370: `result[[0, 0]] = a00.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 371: `result[[1, 1]] = a11.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 392: `if max_diff > F::from(0.5).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 416: `if max_scaled_diff <= F::from(0.2).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 421: `match sqrtm(&a_scaled.view(), 20, F::from(1e-12).unwrap()) {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 499: `let half = F::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 500: `let third = F::from(1.0 / 3.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 501: `let fourth = F::from(0.25).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 502: `let fifth = F::from(0.2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 503: `let sixth = F::from(1.0 / 6.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 515: `let scale_factor = F::from(2.0_f64.powi(scaling_k)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 582: `let half = F::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 583: `let third = F::from(1.0 / 3.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 584: `let fourth = F::from(0.25).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 585: `let fifth = F::from(0.2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 586: `let sixth = F::from(1.0 / 6.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 653: `result[[0, 0]] = val.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 669: `result[[0, 0]] = a00.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 670: `result[[1, 1]] = a11.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 712: `let half = F::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 837: `factorial *= F::from(2 * k - 1).unwrap() * F::from(2 * k).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 845: `result[[i, j]] += sign * a_power[[i, j]] / factorial;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 943: `factorial *= F::from(2 * k).unwrap() * F::from(2 * k + 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 951: `result[[i, j]] += sign * a_power[[i, j]] / factorial;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1063: `result[[0, 0]] = a[[0, 0]].powf(p);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 1088: `let is_int_power = (p - F::from(p_int).unwrap()).abs() < F::epsilon() && p_int >...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1148: `let exp_a = expm(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1160: `let log_a = logm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1172: `let log_a = logm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1200: `let exp_a = expm(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1236: `let exp_a = expm(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1265: `let sqrt_a = sqrtm(&a.view(), 20, 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1277: `let sqrt_a = sqrtm(&a.view(), 20, 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1289: `let a_pow = matrix_power(&a.view(), 3.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1303: `let a_squared = matrix_power(&a.view(), 2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1314: `let cos_a = cosm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1326: `let cos_a = cosm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1339: `let sin_a = sinm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1351: `let sin_a = sinm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1364: `let tan_a = tanm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1376: `let tan_a = tanm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1390: `let sin_a = sinm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1391: `let cos_a = cosm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1459: `let half = F::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1514: `let half = F::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1621: `let tol = F::epsilon() * F::from(100.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1628: `let x_new = (&x + &x_inv) * F::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1746: `let cosh_a = coshm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1758: `let sinh_a = sinhm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1770: `let tanh_a = tanhm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1783: `let sinh_a = sinhm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1784: `let cosh_a = coshm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1804: `let sign_a = signm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1816: `let sign_a = signm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/matrixfree/mod.rs

30 issues found:

- Line 333: `let result_block = block.apply(&x_block.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 427: `if rsold.sqrt() < tol * b_norm {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 437: `let alpha = rsold / pap;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 449: `if rsnew.sqrt() < tol * b_norm {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 454: `let beta = rsnew / rsold;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 533: `v[i] = r[i] / beta;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 577: `new_v[j] = w[j] / h[[i + 1, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 620: `y[j] = sum / h[[j, j]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 650: `let t = a / b;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 651: `let s = F::one() / (F::one() + t * t).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 651: `let s = F::one() / (F::one() + t * t).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 655: `let t = b / a;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 656: `let c = F::one() / (F::one() + t * t).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 656: `let c = F::one() / (F::one() + t * t).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 705: `diag[i] = F::one() / diag[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 806: `let alpha = rz_old / pap;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 826: `let beta = rz_new / rz_old;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 851: `let ax = a.apply(x).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 857: `let diff_norm = vector_norm(&diff.view(), 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 858: `let b_norm = vector_norm(b, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 870: `let y = identity.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 885: `let y = diag_op.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 906: `let y = block_op.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 932: `let x = conjugate_gradient(&spd_op, &b, 10, 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 954: `let x = gmres(&op, &b, 10, 1e-10, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 977: `let precond = jacobi_preconditioner(&op).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 981: `let y = precond.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 988: `assert_relative_eq!(y[1], 2.0 / 3.0, epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1006: `let diag = array![1.0 / 4.0, 1.0 / 3.0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1013: `let x = preconditioned_conjugate_gradient(&spd_op, &precond, &b, 10, 1e-10).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/mixed_precision/adaptive.rs

17 issues found:

- Line 124: `let factor = aug[[j, i]] / aug[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 141: `x_high[i] = (aug[[i, n]] - sum) / aug[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 218: `None => s_max / s_min,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 302: `let tol = tol.unwrap_or(NumCast::from(1e-8).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 453: `norm_x = norm_x.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 456: `if norm_x <= NumCast::from(1e-15).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 477: `let v_norm = v.iter().fold(H::zero(), |sum, &x| sum + x * x).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 478: `if v_norm > NumCast::from(1e-15).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 492: `r_h[[k + i, j]] -= H::from(2.0).unwrap() * v[i] * dot_product;`
  - **Fix**: Use .get() with proper bounds checking
- Line 504: `q_h[[i, k + j]] -= H::from(2.0).unwrap() * dot_product * v[j];`
  - **Fix**: Use .get() with proper bounds checking
- Line 522: `col_norm = col_norm.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 524: `if col_norm > H::from(1e-15).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 599: `let x = mixed_precision_solve::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 612: `let cond = mixed_precision_cond::<f32, f32, f64>(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 617: `let cond_b = mixed_precision_cond::<f32, f32, f64>(&b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 625: `let (q, r) = mixed_precision_qr::<f32, f32, f64>(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 654: `let (u, s, vt) = mixed_precision_svd::<f32, f32, f64>(&a.view(), false).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/mixed_precision/f32_ops.rs

6 issues found:

- Line 298: `mixed_precision_matvec_f32::<f32, f32, f32, f64>(&a.view(), &x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 311: `mixed_precision_matmul_f32_basic::<f32, f32, f32, f64>(&a.view(), &b.view()).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 328: `mixed_precision_matmul_f32_basic::<f32, f32, f32, f64>(&a.view(), &b.view()).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 344: `let result = mixed_precision_dot_f32::<f32, f32, f32, f64>(&a.view(), &b.view())...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 356: `let result = mixed_precision_dot_f32::<f32, f32, f32, f64>(&a.view(), &b.view())...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 368: `let result = mixed_precision_dot_f32::<f32, f32, f32, f64>(&a.view(), &b.view())...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/mixed_precision/f64_ops.rs

5 issues found:

- Line 312: `mixed_precision_matmul_f64::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 329: `mixed_precision_matmul_f64::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 348: `mixed_precision_matmul_f64::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 367: `mixed_precision_matmul_f64::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 400: `mixed_precision_matmul_f64::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/mixed_precision/mod.rs

12 issues found:

- Line 409: `let factor = a_high[[j, i]] / a_high[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 433: `let result = mixed_precision_matvec::<f32, f32, f32, f64>(&a.view(), &x.view())....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 445: `let result = mixed_precision_matmul::<f32, f32, f32, f64>(&a.view(), &b.view())....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 459: `let result = mixed_precision_dot::<f32, f32, f32, f64>(&a.view(), &b.view()).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 469: `let result = mixed_precision_inv::<f32, f32, f64>(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 491: `let det = mixed_precision_det::<f32, f32, f64>(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 497: `let det_id = mixed_precision_det::<f32, f32, f64>(&identity.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 502: `let det_sing = mixed_precision_det::<f32, f32, f64>(&singular.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 515: `mixed_precision_matmul::<f32, f32, f32, f64>(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 520: `mixed_precision_matvec::<f32, f32, f32, f64>(&a.view(), &x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 525: `mixed_precision_solve::<f32, f32, f32, f64>(&a.view(), &x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 529: `let (q, r) = mixed_precision_qr::<f32, f32, f64>(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/mixed_precision/simd.rs

6 issues found:

- Line 456: `simd_mixed_precision_matvec_f32_f64::<f32>(&mat.view(), &vec.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 458: `simd_mixed_precision_matvec_f32_f64::<f64>(&mat.view(), &vec.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 493: `let result_f32 = simd_mixed_precision_matmul_f32_f64::<f32>(&a.view(), &b.view()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 494: `let result_f64 = simd_mixed_precision_matmul_f32_f64::<f64>(&a.view(), &b.view()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 532: `let result_f32 = simd_mixed_precision_dot_f32_f64::<f32>(&a.view(), &b.view()).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 533: `let result_f64 = simd_mixed_precision_dot_f32_f64::<f64>(&a.view(), &b.view()).u...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/norm.rs

13 issues found:

- Line 57: `Ok(sum_sq.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 141: `Ok(sum_sq.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 215: `if val > F::epsilon() * F::from(100).unwrap() * sigma_max {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 224: `Ok(sigma_max / sigma_min)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 241: `if val > F::epsilon() * F::from(100).unwrap() * sigma_max {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 250: `Ok(sigma_max / sigma_min)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 320: `F::from(max_dim).unwrap() * eps * sigma_max`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 373: `let norm = matrix_norm(&a.view(), "fro", None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 381: `let norm = matrix_norm(&a.view(), "1", None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 389: `let norm = matrix_norm(&a.view(), "inf", None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 397: `let norm = vector_norm(&x.view(), 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 405: `let norm = vector_norm(&x.view(), 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 413: `let norm = vector_norm(&x.view(), usize::MAX).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/optim/mod.rs

12 issues found:

- Line 224: `return standard_matmul(a, b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 227: `let half_n = n / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 407: `let result = block_matmul(&a.view(), &b.view(), Some(1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 421: `let result = block_matmul(&a.view(), &b.view(), Some(2)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 435: `let result = strassen_matmul(&a.view(), &b.view(), Some(1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 449: `let result = strassen_matmul(&a.view(), &b.view(), Some(2)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 467: `let result = tiled_matmul(&a.view(), &b.view(), Some(1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 481: `let result = tiled_matmul(&a.view(), &b.view(), Some(2)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 506: `let result_standard = standard_matmul(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 507: `let result_block = block_matmul(&a.view(), &b.view(), Some(4)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 508: `let result_strassen = strassen_matmul(&a.view(), &b.view(), Some(8)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 509: `let result_tiled = tiled_matmul(&a.view(), &b.view(), Some(4)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/parallel.rs

50 issues found:

- Line 357: `let mut results_guard = results.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 367: `handle.join().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 382: `let items_per_worker = total_items / self.num_workers;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 386: `std::cmp::max(1, items_per_worker / 4)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 389: `items_per_worker / 8`
  - **Fix**: Division without zero check - use safe_divide()
- Line 392: `std::cmp::min(self.chunk_size, items_per_worker / 16)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 456: `let mut global_results = results_vec.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 464: `handle.join().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 561: `let mut results_guard = results.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 567: `let mut stats = timing_stats.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 578: `handle.join().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 590: `let stats = self.timing_stats.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 592: `stats.total_time_ms as f64 / stats.total_items as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 600: `let stats = self.timing_stats.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 602: `(stats.max_time_ms - stats.min_time_ms) as f64 / stats.min_time_ms as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 655: `.map(|n| std::cmp::max(1, n.get() / 2))`
  - **Fix**: Division without zero check - use safe_divide()
- Line 661: `.map(|n| n.get() + n.get() / 2)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 774: `GLOBAL_POOL.as_ref().unwrap().clone()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 781: `let mut manager = pool.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 812: `let mut current = self.current_threads.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 813: `let mut cpu_util = self.cpu_utilization.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 834: `*self.current_threads.lock().unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 927: `let throughput = total_work / elapsed.as_secs_f64();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 955: `.max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1057: `let norm = v.iter().map(|&x| x * x).sum::<F>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1074: `let norm = new_v.iter().map(|&x| x * x).sum::<F>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1080: `let normalized_v = new_v / norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1137: `return Ok(x.iter().map(|&xi| xi * xi).sum::<F>().sqrt());`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1144: `Ok(sum_squares.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1270: `-x.iter().map(|&xi| xi * xi).sum::<F>().sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1272: `x.iter().map(|&xi| xi * xi).sum::<F>().sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1297: `let factor = F::from(2.0).unwrap() * dot_product / v_norm_sq;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1313: `let factor = F::from(2.0).unwrap() * dot_product / v_norm_sq;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1369: `l[[i, i]] = aii.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1377: `l[[j, i]] = (matrix[[j, i]] - sum) / l[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1389: `l[[i, j]] = (matrix[[i, j]] - sum) / l[[j, j]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1453: `let multiplier = a[[i, k]] / pivot;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1535: `let alpha = rsold / vector_ops::parallel_dot(&p.view(), &ap.view(), config)?;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1542: `if rsnew.sqrt() < tolerance {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1546: `let beta = rsnew / rsold;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1650: `let mut v = vec![r / beta];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1671: `v.push(w_new / h[[j + 1, j]]);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1687: `y[i] = sum / h_sub[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1764: `let beta = (rho_new / rho) * (alpha / omega);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1773: `alpha = rho_new / vector_ops::parallel_dot(&r_hat.view(), &v.view(), config)?;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1788: `omega = vector_ops::parallel_dot(&t.view(), &s.view(), config)? /`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1870: `sum / diag[i]`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1919: `if omega <= F::zero() || omega >= F::from(2.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1951: `let x_gs = sum / matrix[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1973: `let x_gs = sum / matrix[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()

### src/perf_opt.rs

14 issues found:

- Line 519: `time_standard.as_secs_f64() / time_optimized.as_secs_f64()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 595: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 612: `let scale = F::one() / u1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 736: `.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 749: `let scale = F::one() / beta;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 764: `let tau = dot_product * F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 808: `let c = blocked_matmul(&a.view(), &b.view(), &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 819: `let a_t = optimized_transpose(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 836: `let y = parallel_matvec(&a.view(), &x.view(), &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 847: `inplace::add_assign(&mut a, &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 854: `inplace::scalar_mul_assign(&mut a, 2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 866: `inplace::transpose_square(&mut a).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 879: `let c = adaptive_matmul(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 901: `let c = blocked_matmul(&a.view(), &b.view(), &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/preconditioners.rs

51 issues found:

- Line 225: `inverse_diagonal[i] = F::one() / diag_elem;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 292: `if working_matrix[[i, k]].abs() > F::from(config.drop_tolerance).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 293: `l_factor[[i, k]] = working_matrix[[i, k]] / pivot;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 296: `if working_matrix[[k, j]].abs() > F::from(config.drop_tolerance).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 298: `> F::from(config.drop_tolerance).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 309: `if working_matrix[[k, j]].abs() > F::from(config.drop_tolerance).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 346: `z[i] = (y[i] - sum) / self.u_factor[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 407: `l_factor[[k, k]] = diag_elem.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 411: `if working_matrix[[i, k]].abs() > F::from(config.drop_tolerance).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 412: `l_factor[[i, k]] = working_matrix[[i, k]] / l_factor[[k, k]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 419: `if l_factor[[i, k]].abs() > F::from(config.drop_tolerance).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 420: `&& l_factor[[j, k]].abs() > F::from(config.drop_tolerance).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 440: `y[i] = (x[i] - sum) / self.l_factor[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 455: `z[i] = (y[i] - sum) / self.l_factor[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 544: `x[i] = (y[i] - sum) / u[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 621: `let scaling = F::one() / matrix_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 693: `if condition_estimate < F::from(10.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 723: `let norm_x = (x.iter().map(|&val| val * val).sum::<F>()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 729: `let norm_y = (y.iter().map(|&val| val * val).sum::<F>()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 730: `x = y / norm_y;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 737: `Ok(lambda_max * F::from(n as f64).unwrap() / lambda_max.max(F::epsilon()))`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 747: `let tolerance = F::from(1e-12).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 777: `let tolerance = F::from(1e-14).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 780: `zero_elements as f64 / total_elements as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 910: `let alpha = rsold / p.dot(&ap);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 916: `let residual_norm = (r.iter().map(|&val| val * val).sum::<F>()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 925: `let beta = rsnew / rsold;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 959: `for outer_iter in 0..(max_iterations / restart + 1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 962: `let r_norm = (r.iter().map(|&val| val * val).sum::<F>()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 974: `v[0] = preconditioner.apply(&r.view())? / r_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 992: `let w_norm = (w_prec.iter().map(|&val| val * val).sum::<F>()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 996: `v[j + 1] = w_prec / w_norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1009: `x = &x + &v[0] * (tolerance * F::from(0.1).unwrap());`
  - **Fix**: Use .get() with proper bounds checking
- Line 1082: `let preconditioner = DiagonalPreconditioner::new(&matrix.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1085: `let result = preconditioner.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1089: `assert_relative_eq!(result[1], 2.0 / 3.0, epsilon = 1e-10);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1097: `let preconditioner = IncompleteLUPreconditioner::new(&matrix.view(), &config).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1100: `let result = preconditioner.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1113: `IncompleteCholeskyPreconditioner::new(&matrix.view(), &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1116: `let result = preconditioner.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1133: `let preconditioner = BlockJacobiPreconditioner::new(&matrix.view(), &config).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1136: `let result = preconditioner.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1148: `let preconditioner = PolynomialPreconditioner::new(&matrix.view(), &config).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1151: `let result = preconditioner.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1163: `let preconditioner = AdaptivePreconditioner::new(&matrix.view(), &config).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1166: `let result = preconditioner.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1179: `let preconditioner = create_preconditioner(&matrix.view(), &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1189: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1193: `let residual_norm = (residual.iter().map(|&val| val * val).sum::<f64>()).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1202: `let preconditioner = create_preconditioner(&matrix.view(), &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1204: `let analysis = analyze_preconditioner(&matrix.view(), preconditioner.as_ref()).u...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/projection/mod.rs

28 issues found:

- Line 65: `let scale = F::from(1.0 / (n_components as f64).sqrt()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 75: `let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 75: `let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 77: `let value = F::from(z).unwrap() * scale;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 134: `let scale = F::from(1.0 / (density * n_components as f64).sqrt()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 138: `let prob_neg = density / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 199: `let s = (n_features as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 200: `let prob_nonzero = 1.0 / s;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 201: `let prob_neg = prob_nonzero / 2.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 204: `let scale = F::from((s / n_components as f64).sqrt()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 213: `F::from(-1.0).unwrap() * scale`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 215: `F::from(1.0).unwrap() * scale`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 373: `let denominator = eps.powi(2) / 2.0 - eps.powi(3) / 3.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 374: `let min_dim = (4.0 * (n_samples as f64).ln() / denominator).ceil() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 374: `let min_dim = (4.0 * (n_samples as f64).ln() / denominator).ceil() as usize;`
  - **Fix**: Mathematical operation .ln() without validation
- Line 390: `let components = gaussian_random_matrix::<f64>(n_components, n_features).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 407: `let components = sparse_random_matrix::<f64>(n_components, n_features, density)....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 415: `let actual_density = non_zeros as f64 / total_elements as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 434: `let components = very_sparse_random_matrix::<f64>(n_components, n_features).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 442: `let expected_density = 1.0 / (n_features as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 442: `let expected_density = 1.0 / (n_features as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 443: `let actual_density = non_zeros as f64 / total_elements as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 464: `let components = gaussian_random_matrix::<f64>(n_components, n_features).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 467: `let x_projected = project(&x.view(), &components.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 487: `let (x_projected, components) = johnson_lindenstrauss_transform(&x.view(), eps)....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 493: `let min_dim = johnson_lindenstrauss_min_dim(n_samples, eps).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 497: `let expected_proj = project(&x.view(), &components.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 504: `assert!(johnson_lindenstrauss_min_dim(10000, 0.1).unwrap() >= 500);`
  - **Fix**: Replace with ? operator or .ok_or()

### src/quantization/calibration.rs

79 issues found:

- Line 393: `let min_val = *values.first().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 394: `let max_val = *values.last().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 398: `let min_val = values.iter().take(window_size).sum::<f32>() / window_size as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 399: `let max_val = values.iter().rev().take(window_size).sum::<f32>() / window_size a...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 563: `let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 566: `let scale = (col_max - col_min) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 567: `let zero_point = (-col_min / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 653: `(*values.first().unwrap(), *values.last().unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 656: `let min_val = values.iter().take(window_size).sum::<f32>() / window_size as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 657: `let max_val = values.iter().rev().take(window_size).sum::<f32>() / window_size a...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 664: `let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 667: `let scale = (col_max - col_min) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 668: `let zero_point = (-col_min / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 768: `let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 771: `let scale = (col_max - col_min) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 772: `let zero_point = (-col_min / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 843: `let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 846: `let scale = (opt_max - opt_min) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 847: `let zero_point = (-opt_min / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 909: `abs_max / ((1 << (bits - 1)) - 1) as f32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 911: `(col_max - col_min) / ((1 << bits) - 1) as f32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 917: `(-col_min / base_scale).round() as i32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1033: `let min_val = *values.first().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1034: `let max_val = *values.last().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1038: `let min_val = values.iter().take(window_size).sum::<f32>() / window_size as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1039: `let max_val = values.iter().rev().take(window_size).sum::<f32>() / window_size a...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1225: `let bin_width = (max_val - min_val) / num_bins as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1229: `histogram[num_bins / 2] = matrix.len();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1236: `let bin_idx = ((val_f32 - min_val) / bin_width).floor() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1256: `let bin_width = (max_val - min_val) / num_bins as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1260: `histogram[num_bins / 2] = vector.len();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1267: `let bin_idx = ((val_f32 - min_val) / bin_width).floor() as usize;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1285: `let bin_width = (max_val - min_val) / num_bins as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1291: `.map(|&count| count as f32 / total_count)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1308: `let step = (best_abs_max / 20.0).max(1e-6);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1316: `let quantization_step = abs_max / (levels - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1343: `let min_step = (max_val - min_val) / 40.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1357: `let quantization_step = (trial_max - trial_min) / (levels - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1407: `(orig_val / quantization_step).round() * quantization_step`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1411: `let new_bin_idx = ((quantized_val - min_val) / bin_width).floor() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1423: `kl += p * (p / q).ln();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1423: `kl += p * (p / q).ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 1456: `let steps = ((orig_val - quant_min) / quantization_step).round();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1461: `let new_bin_idx = ((quantized_val - min_val) / bin_width).floor() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1473: `kl += p * (p / q).ln();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1473: `kl += p * (p / q).ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 1489: `let factor = 0.5 + 1.5 * (i as f32 / (num_trials - 1) as f32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1523: `let quantized = (x / scale)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1530: `let mse = (&matrix_f32 - &dequantized).mapv(|x| x * x).sum() / matrix.len() as f...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1550: `let factor = 0.5 + 1.5 * (i as f32 / (num_trials - 1) as f32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1584: `let quantized = (x / scale)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1591: `let mse = (&vector_f32 - &dequantized).mapv(|x| x * x).sum() / vector.len() as f...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1618: `let factor = 0.8 + 0.4 * (i as f32 / (num_scale_trials - 1) as f32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1668: `let quantized = ((x / scale) + zero_point as f32)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1675: `let mse = (&matrix_f32 - &dequantized).mapv(|x| x * x).sum() / matrix.len() as f...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1704: `let factor = 0.8 + 0.4 * (i as f32 / (num_scale_trials - 1) as f32);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1754: `let quantized = ((x / scale) + zero_point as f32)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1761: `let mse = (&vector_f32 - &dequantized).mapv(|x| x * x).sum() / vector.len() as f...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1783: `let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1787: `let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1788: `let zero_point = (-min_val / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1829: `let params = calibrate_matrix_minmax(&matrix.view(), 8, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1834: `assert_relative_eq!(params.scale, 9.0 / 127.0, epsilon = 1e-6);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1838: `let params = calibrate_matrix_minmax(&matrix.view(), 8, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1843: `assert_relative_eq!(params.scale, (9.0 - 1.0) / 255.0, epsilon = 1e-6);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1846: `(-params.min_val / params.scale).round() as i32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1859: `let params = calibrate_matrix_percentile(&matrix.view(), 8, 0.8, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1863: `let params = calibrate_matrix_percentile(&matrix.view(), 8, 1.0, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1872: `let params = calibrate_vector_minmax(&vector.view(), 8, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1877: `assert_relative_eq!(params.scale, 5.0 / 127.0, epsilon = 1e-6);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1881: `let params = calibrate_vector_minmax(&vector.view(), 8, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1886: `assert_relative_eq!(params.scale, (5.0 - 1.0) / 255.0, epsilon = 1e-6);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1889: `(-params.min_val / params.scale).round() as i32`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1903: `let params = calibrate_matrix_per_channel_minmax(&matrix.view(), 8, true).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1908: `let scales = params.channel_scales.as_ref().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1916: `assert_relative_eq!(scales[0], 0.3 / 127.0, epsilon = 1e-5);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1917: `assert_relative_eq!(scales[1], 30.0 / 127.0, epsilon = 1e-5);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1918: `assert_relative_eq!(scales[2], 300.0 / 127.0, epsilon = 1e-5);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1934: `let params = calibrate_matrix(&matrix.view(), 8, &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/quantization/calibration_ema.rs

28 issues found:

- Line 16: `F::from_f32(val).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 87: `let mse = (&matrix_f32 - &dequantized).mapv(|x| x * x).sum() / matrix_f32.len() ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 155: `1.0 + (negative_errors / neg_count)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 160: `1.0 + (positive_errors / pos_count)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 255: `let scale_f32 = abs_max_f32 / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 260: `let scale_f32 = (max_val_f32 - min_val_f32) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 261: `let zero_point = (-min_val_f32 / scale_f32).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 280: `/ column_f32_view.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 300: `/ column_f32_view.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 350: `1.0 + (negative_errors / neg_count)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 355: `1.0 + (positive_errors / pos_count)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 379: `let scale_f32 = abs_max_f32 / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 384: `let scale_f32 = (max_val_f32 - min_val_f32) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 385: `let zero_point = (-min_val_f32 / scale_f32).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 470: `let scale_f32 = abs_max_f32 / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 475: `let scale_f32 = (max_val_f32 - min_val_f32) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 476: `let zero_point = (-min_val_f32 / scale_f32).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 495: `/ vector_f32.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 518: `/ vector_f32.len() as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 567: `1.0 + (negative_errors / neg_count)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 572: `1.0 + (positive_errors / pos_count)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 616: `let quantized = (val / scale).round().clamp(clamp_min, clamp_max);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 625: `let quantized = ((val / scale) + zero_point).round().clamp(0.0, clamp_max);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 656: `let quantized = (x / scale).round().clamp(clamp_min, clamp_max);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 666: `let quantized = ((x / scale) + zero_point).round().clamp(0.0, clamp_max);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 686: `let quantized = (val / scale).round().clamp(clamp_min, clamp_max);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 693: `params.channel_zero_points.as_ref().unwrap()[col_idx] as f32;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 697: `((val / scale) + zero_point).round().clamp(0.0, clamp_max);`
  - **Fix**: Division without zero check - use safe_divide()

### src/quantization/fusion.rs

11 issues found:

- Line 99: `.map(|m| get_quantized_matrix_2d_i8(m).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 107: `let (_, cols) = matrices.last().unwrap().shape;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 156: `let last_mat = int8_matrices.last().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 214: `if matrices.last().unwrap().shape.1 != vector_len {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 217: `matrices.last().unwrap().shape.1,`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 259: `Ok(result_f32.mapv(|x| num_traits::FromPrimitive::from_f32(x).unwrap()))`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 262: `Ok(result_f32.mapv(|x| num_traits::FromPrimitive::from_f32(x).unwrap()))`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 287: `let result_f = result_1d_f32.mapv(|x| num_traits::FromPrimitive::from_f32(x).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 302: `.map(|m| get_quantized_matrix_2d_i8(m).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 414: `let result = fused_quantized_matmul_chain(&matrices, &params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 442: `let result = fused_quantized_matvec_sequence(&matrices, &params, &x.view(), fals...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/quantization/mod.rs

103 issues found:

- Line 345: `let byte_idx = idx / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 347: `let byte = arr.as_slice().unwrap()[byte_idx];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 359: `let byte_idx = idx / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 361: `let byte = arr.as_slice().unwrap()[byte_idx];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 454: `let byte_idx = idx / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 467: `let byte_idx = idx / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 549: `f16_data.as_slice_mut().unwrap()[i] = f16::from_f32(val_f32);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 572: `bf16_data.as_slice_mut().unwrap()[i] = bf16::from_f32(val_f32);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 607: `let scale = (max_val - min_val) / ((1 << effective_bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 614: `let scale = abs_max / ((1 << (effective_bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 619: `let scale = (max_val - min_val) / ((1 << effective_bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 620: `let zero_point = (-min_val / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 626: `let ideal_scale = range / ((1 << effective_bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 628: `let scale = 2.0_f32.powf(exponent);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 642: `let zero_point = (-min_val / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 670: `let val_f32: f32 = matrix.as_slice().unwrap()[i].as_();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 672: `let q_val = ((val_f32 / scale).round() as i8).clamp(-8, 7);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 674: `let byte_idx = i / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 677: `packed_data.as_slice_mut().unwrap()[byte_idx] = q_val << 4;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 680: `packed_data.as_slice_mut().unwrap()[byte_idx] |= q_val & 0x0F;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 688: `let packed_reshaped = packed_data.into_shape_with_order(packed_shape).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 701: `let val_f32: f32 = matrix.as_slice().unwrap()[i].as_();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 703: `let ival = ((val_f32 - min_val) / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 706: `let byte_idx = i / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 709: `packed_data.as_slice_mut().unwrap()[byte_idx] = q_val << 4;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 712: `packed_data.as_slice_mut().unwrap()[byte_idx] |= q_val & 0x0F;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 720: `let packed_reshaped = packed_data.into_shape_with_order(packed_shape).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 733: `let q_val = ((val_f32 - min_val) / scale).round() as i8;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 734: `quantized.as_slice_mut().unwrap()[i] = q_val;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 742: `let q_val = (val_f32 / scale).round() as i8;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 743: `quantized.as_slice_mut().unwrap()[i] = q_val;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 751: `let q_val = ((val_f32 / scale) + zero_point as f32).round() as i8;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 752: `quantized.as_slice_mut().unwrap()[i] = q_val;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 760: `let q_val = ((val_f32 - min_val) / scale).round() as i8;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 761: `quantized.as_slice_mut().unwrap()[i] = q_val;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 856: `channel_scales[col] = abs_max / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 864: `(channel_max_vals[col] - channel_min_vals[col]) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 866: `(-channel_min_vals[col] / channel_scales[col]).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 874: `let scale = channel_scales.iter().sum::<f32>() / num_channels as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 876: `(channel_zero_points.iter().sum::<i32>() as f32 / num_channels as f32).round() a...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 907: `(val_f32 / scale)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 914: `((val_f32 / scale) + zero_point as f32)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 951: `dequantized.as_slice_mut().unwrap()[i] = val.to_f32();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 957: `dequantized.as_slice_mut().unwrap()[i] = val.to_f32();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 968: `let row = i / shape.1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1034: `dequantized.as_slice_mut().unwrap()[i] = val;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1040: `dequantized.as_slice_mut().unwrap()[i] = val;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1046: `dequantized.as_slice_mut().unwrap()[i] = val;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1052: `dequantized.as_slice_mut().unwrap()[i] = val;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1168: `let scale = (max_val - min_val) / ((1 << effective_bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1175: `let scale = abs_max / ((1 << (effective_bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1180: `let scale = (max_val - min_val) / ((1 << effective_bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1181: `let zero_point = (-min_val / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1187: `let ideal_scale = range / ((1 << effective_bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1189: `let scale = 2.0_f32.powf(exponent);`
  - **Fix**: Mathematical operation .powf( without validation
- Line 1203: `let zero_point = (-min_val / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1232: `let q_val = ((val_f32 / scale).round() as i8).clamp(-8, 7);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1234: `let byte_idx = i / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1257: `let ival = ((val_f32 - min_val) / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1260: `let byte_idx = i / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1282: `let q_val = ((val_f32 - min_val) / scale).round() as i8;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1291: `let q_val = (val_f32 / scale).round() as i8;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1300: `let q_val = ((val_f32 / scale) + zero_point as f32).round() as i8;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1309: `let q_val = ((val_f32 - min_val) / scale).round() as i8;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1775: `result.as_slice_mut().unwrap()[i] = F::from_f32(val).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1808: `result[i] = F::from_f32(val).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 1931: `let c_q = quantized_matmul(&a_q, &a_params, &b_q, &b_params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1937: `let rel_error = (&c - &c_q).mapv(|x| x.abs()).sum() / c.sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1951: `let c_q = quantized_matmul(&a_q, &a_params, &b_q, &b_params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1957: `let rel_error = (&c - &c_q).mapv(|x| x.abs()).sum() / c.sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1972: `let y_q = quantized_matvec(&a_q, &a_params, &x_q, &x_params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1978: `let rel_error = (&y - &y_q).mapv(|x| x.abs()).sum() / y.sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1992: `let y_q = quantized_matvec(&a_q, &a_params, &x_q, &x_params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1998: `let rel_error = (&y - &y_q).mapv(|x| x.abs()).sum() / y.sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2013: `let dot_q = quantized_dot(&a_q, &a_params, &b_q, &b_params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2019: `let rel_error = (dot - dot_q).abs() / dot;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2033: `let dot_q = quantized_dot(&a_q, &a_params, &b_q, &b_params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2039: `let rel_error = (dot - dot_q).abs() / dot;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2129: `let matrix = Array2::from_shape_vec((rows, cols), data).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 2198: `let matrix = Array2::from_shape_vec((rows, cols), data).unwrap();`
  - **Fix**: Handle array creation errors properly
- Line 2238: `let channel_scales = params.channel_scales.as_ref().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2245: `let zero_points = params.channel_zero_points.as_ref().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2262: `/ col_original`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2295: `let channel_scales = params.channel_scales.as_ref().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2296: `let zero_points = params.channel_zero_points.as_ref().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2320: `/ col_original`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2380: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2384: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2391: `perchan_small_error < reg_small_error / 2.0,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2412: `let c_q = quantized_matmul(&a_q, &a_params, &b_q, &b_params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2415: `let rel_error = (&c_true - &c_q).mapv(|x| x.abs()).sum() / c_true.sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2441: `let y_q = quantized_matvec(&a_q, &a_params, &x_q, &x_params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2444: `let rel_error = (&y_true - &y_q).mapv(|x| x.abs()).sum() / y_true.sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2467: `let c_q = quantized_matmul(&a_q, &a_params, &b_q, &b_params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2471: `let y_q = quantized_matvec(&a_q, &a_params, &x_q, &x_params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2475: `let matmul_rel_error = (&c - &c_q).mapv(|x| x.abs()).sum() / c.sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2476: `let matvec_rel_error = (&y - &y_q).mapv(|x| x.abs()).sum() / y.sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2496: `let dot_q = quantized_dot(&a_q, &a_params, &b_q, &b_params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2502: `let rel_error = (dot - dot_q).abs() / dot;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2521: `let c_mixed = quantized_matmul(&a_f16, &a_f16_params, &b_bf16, &b_bf16_params).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2522: `let y_mixed = quantized_matvec(&a_f16, &a_f16_params, &x_i8, &x_i8_params).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 2529: `let matmul_rel_error = (&c - &c_mixed).mapv(|x| x.abs()).sum() / c.sum();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 2530: `let matvec_rel_error = (&y - &y_mixed).mapv(|x| x.abs()).sum() / y.sum();`
  - **Fix**: Division without zero check - use safe_divide()

### src/quantization/out_of_core.rs

24 issues found:

- Line 126: `let scale = global_max / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 129: `let scale = (global_max - global_min) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 130: `let zero_point = (-global_min / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 499: `let b_norm = (b.dot(b)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 500: `if b_norm < F::epsilon() || rsold.sqrt() < tol * b_norm {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 520: `let alpha = rsold / pap;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 532: `if rsnew.sqrt() < tol * b_norm {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 539: `let ratio = rsnew / previous_residual;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 542: `if ratio > F::from(0.9).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 562: `if rsnew.sqrt() < tol * b_norm {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 571: `let beta = rsnew / rsold;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 741: `((val / params.scale).round()).clamp(min_val as f32, max_val as f32) as i8`
  - **Fix**: Division without zero check - use safe_divide()
- Line 744: `((val / params.scale + params.zero_point as f32).round())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 783: `file_path.to_str().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 785: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 789: `let y = chunked.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 817: `file_path.to_str().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 819: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 822: `let loaded = ChunkedQuantizedMatrix::from_file(file_path.to_str().unwrap()).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 826: `let y = loaded.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 854: `file_path.to_str().unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 856: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 866: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 871: `let residual_norm = (residual.dot(&residual)).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/quantization/quantized_matrixfree.rs

37 issues found:

- Line 90: `let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 93: `let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 94: `let zero_point = (-min_val / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 157: `let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 160: `let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 161: `let zero_point = (-min_val / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 186: `(val / scale)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 191: `((val / scale) + zero_point as f32)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 326: `let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 329: `let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 330: `let zero_point = (-min_val / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 339: `(val / scale)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 344: `((val / scale) + zero_point as f32)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 437: `let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 440: `let scale = (global_max_val - global_min_val) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 441: `let zero_point = (-global_min_val / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 529: `let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 532: `let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 533: `let zero_point = (-min_val / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 542: `(val / scale)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 547: `((val / scale) + zero_point as f32)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 677: `let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 680: `let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 681: `let zero_point = (-min_val / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 690: `(val / scale)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 695: `((val / scale) + zero_point as f32)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 956: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 960: `let y = op.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 985: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 989: `let y = op.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1020: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1024: `let y = op.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1055: `let op = QuantizedMatrixFreeOp::banded(3, bands, 8, QuantizationMethod::Symmetri...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1059: `let y = op.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1081: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1099: `let y_quantized = quantized_op.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1100: `let y_linear = linear_op.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/quantization/simd.rs

15 issues found:

- Line 47: `let vec_slice = vector.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 80: `let row_slice = row.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 120: `let row_slice = row.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 173: `let row_slice = row.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 356: `let byte_idx = k / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 389: `let byte_idx = k / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 568: `let byte_idx = i / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 595: `let byte_idx = i / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 628: `let a_slice = a_data.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 629: `let b_slice = b_data.as_slice().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 718: `let result = simd_quantized_matvec(&qmat, &qparams, &vec.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 741: `let result = simd_quantized_matmul(&qa, &qa_params, &qb, &qb_params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 764: `let result = simd_quantized_dot(&qa, &qa_params, &qb, &qb_params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 784: `let result = simd_quantized_matvec(&qmat, &qparams, &vec.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 812: `let result = simd_quantized_matvec(&qmat, &qparams, &vec.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/quantization/solvers.rs

39 issues found:

- Line 92: `if rsold.sqrt() < tol * b_norm {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 125: `let alpha = rsold / pap;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 137: `if rsnew.sqrt() < tol * b_norm {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 144: `let ratio = rsnew / previous_residual;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 147: `if ratio > F::from(0.9).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 167: `if rsnew.sqrt() < tol * b_norm {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 176: `let beta = rsnew / rsold;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 262: `v[i] = r[i] / beta;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 316: `new_v[j] = w[j] / h[[i + 1, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 351: `let progress_ratio = residual / g[i].abs();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 354: `if progress_ratio > F::from(0.8).unwrap() && reorth_step > 1 {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 355: `reorth_step = reorth_step.max(1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 358: `else if progress_ratio < F::from(0.5).unwrap() && reorth_step < restart_iter {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 373: `y[j] = sum / h[[j, j]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 413: `let t = a / b;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 414: `let s = F::one() / (F::one() + t * t).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 414: `let s = F::one() / (F::one() + t * t).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 418: `let t = b / a;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 419: `let c = F::one() / (F::one() + t * t).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 419: `let c = F::one() / (F::one() + t * t).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 481: `diag[i] = F::one() / diag[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 624: `let alpha = rz_old / pap;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 643: `let ratio = r_squared / previous_residual;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 646: `if ratio > F::from(0.9).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 678: `let beta = rz_new / rz_old;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 707: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 715: `let x = quantized_conjugate_gradient(&op, &b, 10, 1e-6, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 735: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 741: `let x = quantized_gmres(&op, &b, 10, 1e-6, None, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 761: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 766: `let precond = quantized_jacobi_preconditioner(&op).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 773: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 793: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 796: `let precond = quantized_jacobi_preconditioner(&op).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 800: `let y = precond.apply(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 806: `let expected = array![0.25f32, 2.0 / 3.0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 822: `.unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 830: `let x_adaptive = quantized_conjugate_gradient(&op, &b, 10, 1e-6, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 833: `let x_standard = quantized_conjugate_gradient(&op, &b, 10, 1e-6, false).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/quantization/stability.rs

22 issues found:

- Line 119: `let scale = abs_max / ((1 << (bits - 1)) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 122: `let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 123: `let zero_point = (-min_val / scale).round() as i32;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 133: `let quantized = (x / scale).round().clamp(clamp_min, clamp_max);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 140: `let quantized = ((x / scale) + zero_point as f32)`
  - **Fix**: Division without zero check - use safe_divide()
- Line 164: `let mse = sum_squared_error / num_elements;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 165: `let rmse = mse.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 166: `let mae = sum_abs_error / num_elements;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 169: `let signal_power = sum_squared_signal / num_elements;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 171: `signal_power / mse`
  - **Fix**: Division without zero check - use safe_divide()
- Line 180: `20.0 * (data_range / 2.0).log10() - 10.0 * mse.log10()`
  - **Fix**: Division without zero check - use safe_divide()
- Line 186: `let dynamic_range = (max_val / min_val.abs().max(1e-6)).abs().log2().ceil();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 219: `let is_asymmetric_data = min_val.abs() < max_val / 10.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 253: `let zero_ratio = count_near_zero_values(&matrix_f32, scale / 2.0) as f32 / num_e...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 362: `let is_asymmetric = min_val.abs() < max_val / 5.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 468: `max_range / min_range`
  - **Fix**: Division without zero check - use safe_divide()
- Line 501: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 525: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 537: `analyze_quantization_stability(&matrix.view(), 8, QuantizationMethod::Affine).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 554: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 570: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 586: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/random.rs

25 issues found:

- Line 102: `let val = low + F::from_f64(r).unwrap() * range;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 176: `let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 176: `let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 179: `let val = mean + F::from_f64(z0).unwrap() * std;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 231: `let (q, _) = qr(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 297: `diag_values[i] = min_eigenval + F::from_f64(r).unwrap() * range;`
  - **Fix**: Use .get() with proper bounds checking
- Line 354: `diag[i] = low + F::from_f64(r).unwrap() * range;`
  - **Fix**: Use .get() with proper bounds checking
- Line 426: `result[[i, j]] = low + F::from_f64(r).unwrap() * range;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 497: `result[[i, j]] = low + F::from_f64(r).unwrap() * range;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 554: `first_row[i] = low + F::from_f64(r).unwrap() * range;`
  - **Fix**: Use .get() with proper bounds checking
- Line 562: `first_col[i] = low + F::from_f64(r).unwrap() * range;`
  - **Fix**: Use .get() with proper bounds checking
- Line 633: `let min_eigenval = F::one() / condition_number;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 635: `let log_min = min_eigenval.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 636: `let log_max = F::one().ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 639: `let t = F::from_f64((i as f64) / ((n - 1) as f64)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 746: `let value = F::one() / F::from_f64((i + j + 1) as f64).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 856: `let k = (n / 2) + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 868: `corr[[i, j]] = result[[i, j]] / (result[[i, i]] * result[[j, j]]).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 868: `corr[[i, j]] = result[[i, j]] / (result[[i, i]] * result[[j, j]]).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 969: `let scaling = F::from_f64(1.0).unwrap(); // All singular values are 1.0`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1365: `let computed_eigenvalues = eigvals(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1376: `real_computed.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1379: `sorted_expected.sort_by(|a, b| a.partial_cmp(b).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1400: `let expected = 1.0 / (i + j + 1) as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1472: `let eigenvalues = eigvals(&c.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/random_matrices.rs

23 issues found:

- Line 109: `*elem = F::from(uniform.sample(rng)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 117: `*elem = F::from(normal.sample(rng)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 121: `let normal = Normal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 123: `*elem = F::from(normal.sample(rng)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 146: `let avg = (matrix[[i, j]] + matrix[[j, i]]) / F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 182: `eigenvalues[i] = F::from(uniform.sample(rng)).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 230: `let denom = prod.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 267: `if F::from(uniform.sample(rng)).unwrap() < F::from(density).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 270: `F::from(Uniform::new(a, b).unwrap().sample(rng)).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 273: `F::from(Normal::new(mean, std_dev).unwrap().sample(rng)).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 276: `F::from(Normal::new(0.0, 1.0).unwrap().sample(rng)).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 304: `matrix[[i, i]] = F::from(uniform.sample(rng)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 312: `matrix[[i, i]] = F::from(normal.sample(rng)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 316: `let normal = Normal::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 318: `matrix[[i, i]] = F::from(normal.sample(rng)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 346: `F::from(Uniform::new(a, b).unwrap().sample(rng)).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 349: `F::from(Normal::new(mean, std_dev).unwrap().sample(rng)).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 352: `F::from(Normal::new(0.0, 1.0).unwrap().sample(rng)).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 404: `(matrix[[i, j]] + matrix[[j, i]].conj()) / Complex::from(F::from(2.0).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 431: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 444: `let q = random_matrix::<f64, _>(4, 4, MatrixType::Orthogonal, &mut rng).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 470: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 488: `let matrix = random_matrix::<f64, _>(4, 4, MatrixType::Correlation, &mut rng).un...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/random_new.rs

63 issues found:

- Line 112: `).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 123: `result[[i, j]] = F::from_f64(flat_array[idx]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 172: `).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 183: `result[[i, j]] = F::from_f64(flat_array[idx]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 296: `let (q, _) = qr(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 371: `let norm = norm.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 373: `q[[i, 0]] = q[[i, 0]] / norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 400: `let norm = norm.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 402: `q[[i, j]] = q[[i, j]] / norm;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 466: `).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 472: `result[[i, i]] += F::from_f64(diag_values[i]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 546: `).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 552: `result[[i, i]] = result[[i, i]] + Complex::new(F::from_f64(diag_values[i]).unwra...`
  - **Fix**: Use .get() with proper bounds checking
- Line 600: `).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 608: `result[[i, i]] = F::from_f64(diag_values[i]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 665: `).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 679: `result[[i, j]] = F::from_f64(band_values[idx]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 737: `).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 755: `let i = pos / cols;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 757: `result[[i, j]] = F::from_f64(values[idx]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 805: `).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 812: `.map(|i| F::from_f64(all_values[i]).unwrap())`
  - **Fix**: Use .get() with proper bounds checking
- Line 817: `.map(|i| F::from_f64(all_values[n + i]).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 887: `let min_eigenval = F::one() / condition_number;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 889: `let log_min = min_eigenval.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 890: `let log_max = F::one().ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 893: `let t = F::from_f64((i as f64) / ((n - 1) as f64)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1000: `let value = F::one() / F::from_f64((i + j + 1) as f64).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1110: `let k = (n / 2) + 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1122: `corr[[i, j]] = result[[i, j]] / (result[[i, i]] * result[[j, j]]).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1122: `corr[[i, j]] = result[[i, j]] / (result[[i, i]] * result[[j, j]]).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1318: `let upper_size = n * (n + 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1334: `let normal_dist = NormalDist::new(0.0, 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1338: `let val = F::from_f64(normal_dist.sample(&mut rng.rng)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1349: `).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1354: `upper[[i, i]] += F::from_f64(diag_values[i]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 1401: `let normalized_coeffs: Vec<F> = coeffs.iter().map(|&c| c / leading_coeff).collec...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1482: `).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1487: `).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1499: `result[[i, i]] = F::from_f64(diag_values[i]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 1504: `result[[i, i-1]] = F::from_f64(subdiag_values[i-1]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1509: `result[[i, i+1]] = F::from_f64(superdiag_values[i]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 1582: `).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1587: `).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1598: `result[[i, i]] = F::from_f64(diag_values[i]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 1603: `let val = F::from_f64(offdiag_values[i]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 1659: `let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1659: `let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1661: `let dist = UniformDist::new(-limit, limit).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1668: `result[[i, j]] = F::from_f64(values[idx]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 1677: `let std_dev = (2.0 / fan_in as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1677: `let std_dev = (2.0 / fan_in as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1679: `let dist = NormalDist::new(0.0, std_dev).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1686: `result[[i, j]] = F::from_f64(values[idx]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 1695: `let std_dev = (1.0 / fan_in as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1695: `let std_dev = (1.0 / fan_in as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1697: `let dist = NormalDist::new(0.0, std_dev).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1704: `result[[i, j]] = F::from_f64(values[idx]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 1732: `let std_dev = (1.0 / embedding_dim as f64).sqrt();`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1732: `let std_dev = (1.0 / embedding_dim as f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 1734: `let dist = NormalDist::new(0.0, std_dev).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1741: `result[[i, j]] = F::from_f64(values[idx]).unwrap();`
  - **Fix**: Use .get() with proper bounds checking
- Line 1749: `uniform(rows, cols, F::from(-0.1).unwrap(), F::from(0.1).unwrap(), seed)`
  - **Fix**: Replace with ? operator or .ok_or()

### src/scalable.rs

14 issues found:

- Line 134: `let ratio = m as f64 / n as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 138: `} else if ratio < 1.0 / threshold {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 226: `omega[[i, j]] = F::from(rand::random::<f64>() * 2.0 - 1.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 490: `2 * m * n * n + n * n * n / 3`
  - **Fix**: Division without zero check - use safe_divide()
- Line 495: `2 * m * m * n - 2 * m * m * m / 3`
  - **Fix**: Division without zero check - use safe_divide()
- Line 500: `2 * m * n * n - 2 * n * n * n / 3`
  - **Fix**: Division without zero check - use safe_divide()
- Line 581: `let (q, r) = tsqr(&matrix.view(), &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 607: `let (l, q) = lq_decomposition(&matrix.view(), &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 633: `let result = adaptive_decomposition(&tall_matrix.view(), &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 663: `matrix[[i, j]] = 3.0 * (i as f64 / m as f64) * (j as f64 / n as f64);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 664: `matrix[[i, j]] += 2.0 * ((i + 1) as f64 / m as f64) * ((n - j) as f64 / n as f64...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 683: `let (u_approx, s_approx, vt_approx) = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 700: `error / ((m * n) as f64) < 0.1,`
  - **Fix**: Division without zero check - use safe_divide()
- Line 712: `let result_blocked = blocked_matmul(&a.view(), &b.view(), &config).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/simd_ops/elementwise.rs

3 issues found:

- Line 169: `let result = simd_matrix_add_f32(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 183: `let result = simd_matrix_scale_f32(&a.view(), scalar).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 197: `let result = simd_matrix_mul_elementwise_f32(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/simd_ops/gemm.rs

6 issues found:

- Line 333: `simd_gemm_f32(1.0, &a.view(), &b.view(), 0.0, &mut c, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 353: `simd_gemm_f64(1.0, &a.view(), &b.view(), 0.0, &mut c, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 373: `simd_gemm_f32(alpha, &a.view(), &b.view(), beta, &mut c, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 391: `let c = simd_matmul_optimized_f32(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 410: `simd_gemv_f32(alpha, &a.view(), &x.view(), beta, &mut y).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 439: `simd_gemm_f32(1.0, &a.view(), &b.view(), 0.0, &mut c, Some(block_sizes)).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/simd_ops/norms.rs

5 issues found:

- Line 34: `sum_sq.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 59: `sum_sq.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 260: `let expected = (9.0 + 16.0 + 144.0 + 25.0f32).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 273: `let expected = (9.0 + 16.0 + 0.0 + 144.0 + 25.0f32).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 355: `let expected = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/simd_ops/transpose.rs

4 issues found:

- Line 141: `let result = simd_transpose_f32(&matrix.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 161: `let result = simd_transpose_f64(&matrix.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 176: `simd_transpose_inplace_f32(&mut matrix).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 200: `let result = simd_transpose_f32(&matrix.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/solve.rs

12 issues found:

- Line 176: `x[i] = sum / a[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 196: `x[i] = sum / a[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 299: `let threshold = s[0] * F::from(a.nrows().max(a.ncols())).unwrap() * F::epsilon()...`
  - **Fix**: Use .get() with proper bounds checking
- Line 316: `let s_inv = F::one() / s[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 463: `let x = solve(&a.view(), &b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 470: `let x = solve(&a.view(), &b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 480: `let x = solve_triangular(&a.view(), &b.view(), true, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 482: `assert_relative_eq!(x[1], 4.0 / 3.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 487: `let x = solve_triangular(&a.view(), &b.view(), true, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 497: `let x = solve_triangular(&a.view(), &b.view(), false, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 498: `assert_relative_eq!(x[0], 4.0 / 3.0);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 504: `let x = solve_triangular(&a.view(), &b.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/solvers/iterative.rs

39 issues found:

- Line 28: `tolerance: A::from(1e-10).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 111: `if r_norm_sq.sqrt() < options.tolerance {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 115: `residual_norm: r_norm_sq.sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 135: `let alpha = r_norm_sq / p_ap;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 150: `r_norm_sq_new.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 159: `residual_norm: r_norm_sq_new.sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 165: `let beta = r_norm_sq_new / r_norm_sq;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 178: `residual_norm: r_norm_sq.sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 254: `let alpha = rz_old / p_ap;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 269: `r_norm_sq.sqrt()`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 277: `residual_norm: r_norm_sq.sqrt(),`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 289: `let beta = rz_new / rz_old;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 299: `let r_norm = r.dot(&r).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 359: `for _outer in 0..(options.max_iterations / restart).max(1) {`
  - **Fix**: Division without zero check - use safe_divide()
- Line 362: `let beta = r.dot(&r).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 376: `v[0] = &r / beta;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 395: `h[[j + 1, j]] = w_orth.dot(&w_orth).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 404: `v[j + 1] = &w_orth / h[[j + 1, j]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 421: `let residual_norm = r_final.dot(&r_final).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 442: `let residual_norm = r_final.dot(&r_final).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 499: `let r_norm_init = r.dot(&r).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 531: `let beta = (rho / rho_old) * (alpha / omega);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 539: `alpha = rho / r_hat.dot(&v);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 545: `let s_norm = s.dot(&s).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 559: `omega = t.dot(&s) / t.dot(&t);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 569: `let r_norm = r.dot(&r).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 593: `let residual_norm = r_final.dot(&r_final).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 627: `let r = (a * a + b * b).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 628: `let c = a / r;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 629: `let s = b / r;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 660: `y[i] = sum / h_copy[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 679: `let result = conjugate_gradient(&a.view(), &b.view(), None, &options).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 686: `assert!(residual.dot(&residual).sqrt() < 1e-10);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 696: `let result = gmres(&a.view(), &b.view(), None, &options).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 712: `let result = bicgstab(&a.view(), &b.view(), None, &options).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 718: `assert!(residual.dot(&residual).sqrt() < 1e-10);`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 729: `|r: &ArrayView1<f64>| -> Array1<f64> { array![r[0] / 4.0, r[1] / 3.0] };`
  - **Fix**: Division without zero check - use safe_divide()
- Line 734: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 741: `assert!(residual.dot(&residual).sqrt() < 1e-10);`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/sparse_dense/mod.rs

32 issues found:

- Line 329: `col_sums.get_mut(&col).unwrap()[k] = col_sums[&col][k] + dense_row[i] * val;`
  - **Fix**: Use .get() with proper bounds checking
- Line 651: `let sparsity_ratio = sparse.nnz() as f64 / (sparse.nrows() * sparse.ncols()) as ...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 652: `let avg_nnz_per_row = sparse.nnz() as f64 / sparse.nrows() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 701: `let b_norm = b.dot(b).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 719: `let alpha = rsold / pap;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 728: `if rsnew.sqrt() < tolerance * b_norm {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 733: `let beta = rsnew / rsold;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 777: `let m_inv = diag.mapv(|x| T::one() / x);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 786: `let b_norm = b.dot(b).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 798: `let alpha = rzold / pap;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 804: `let r_norm = r.dot(&r).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 812: `let beta = rznew / rzold;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 838: `let sparsity_ratio = sparse.nnz() as f64 / total_elements as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 839: `let avg_nnz_per_row = sparse.nnz() as f64 / sparse.nrows() as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 901: `let tolerance = T::epsilon() * T::from(100.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 905: `} else if val_ij.abs() > T::epsilon() * T::from(100.0).unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1003: `T::epsilon() * T::from(1000.0).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1041: `let threshold = T::epsilon() * T::from(1000.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1049: `let sparsity_ratio = zero_count as f64 / total_elements as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 1109: `let sparse = sparse_from_ndarray(&dense.view(), 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1130: `let sparse_a = sparse_from_ndarray(&dense_a.view(), 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1133: `let result = sparse_dense_matmul(&sparse_a, &dense_b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1152: `let sparse_b = sparse_from_ndarray(&dense_b.view(), 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1155: `let result = dense_sparse_matmul(&dense_a.view(), &sparse_b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1174: `let sparse_a = sparse_from_ndarray(&dense_a.view(), 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1177: `let result = sparse_dense_matvec(&sparse_a, &vec_b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1194: `let sparse_a = sparse_from_ndarray(&dense_a.view(), 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1197: `let result = sparse_dense_add(&sparse_a, &dense_b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1216: `let sparse_a = sparse_from_ndarray(&dense_a.view(), 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1219: `let result = sparse_dense_elementwise_mul(&sparse_a, &dense_b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1239: `let sparse_a = sparse_from_ndarray(&dense_a.view(), 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1242: `let transposed = sparse_transpose(&sparse_a).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/sparse_dense/sparse_eigen.rs

9 issues found:

- Line 67: `v[[i, 0]] = T::from(rand::Rng::random_range(&mut rng, -0.5..0.5)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 210: `v[[i, 0]] = T::from(rand::Rng::random_range(&mut rng, -0.5..0.5)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 300: `let half_k = k / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 445: `let shift = (min_val + max_val) / (T::one() + T::one());`
  - **Fix**: Division without zero check - use safe_divide()
- Line 451: `let k = std::cmp::min(std::cmp::max(10, n / 10), n - 1);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 497: `let sparse = sparse_from_ndarray(&dense.view(), 1e-12).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 503: `let (eigenvals, _eigenvecs) = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 520: `let sparse = sparse_from_ndarray(&dense.view(), 1e-12).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 526: `let (eigenvals, eigenvecs) = result.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/special.rs

9 issues found:

- Line 163: `matrix_functions::sqrtm(a, 20, F::from(1e-10).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 242: `let half = F::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 283: `let result = block_diag(&[&a.view(), &b.view()]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 322: `let sqrt_a = sqrtm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 328: `let log_id = logm(&id.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 334: `let exp_zero = expm(&zero.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 345: `let sign_a = signm(&a.view(), 20, 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 353: `let sign_b = signm(&b.view(), 20, 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 361: `let sign_c = signm(&c.view(), 20, 1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/specialized/banded.rs

42 issues found:

- Line 257: `x[0] = b[0] / a;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 314: `let factor = augmented[[j, i]] / augmented[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 331: `x[i] = (augmented[[i, self.ncols]] - sum) / augmented[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 393: `d_prime[0] = b[0] / diag[0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 394: `c_prime[0] = upper[0] / diag[0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 397: `let m = lower[i - 1] / diag[i - 1];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 407: `d_prime[i] = (b[i] - lower[i - 1] * d_prime[i - 1]) / new_diag;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 408: `c_prime[i] = upper[i] / new_diag;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 416: `let m = lower[n - 2] / diag[n - 2];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 426: `d_prime[n - 1] = (b[n - 1] - lower[n - 2] * d_prime[n - 2]) / new_diag;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 623: `let band = BandedMatrix::new(data.view(), 1, 2, 5, 5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 632: `assert_relative_eq!(band.get(0, 0).unwrap(), 5.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 633: `assert_relative_eq!(band.get(0, 1).unwrap(), 10.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 634: `assert_relative_eq!(band.get(0, 2).unwrap(), 14.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 635: `assert_relative_eq!(band.get(1, 0).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 636: `assert_relative_eq!(band.get(1, 1).unwrap(), 6.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 637: `assert_relative_eq!(band.get(1, 2).unwrap(), 11.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 638: `assert_relative_eq!(band.get(1, 3).unwrap(), 15.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 641: `assert_relative_eq!(band.get(0, 3).unwrap(), 0.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 642: `assert_relative_eq!(band.get(0, 4).unwrap(), 0.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 643: `assert_relative_eq!(band.get(3, 0).unwrap(), 0.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 658: `let band = BandedMatrix::from_matrix(&a.view(), 1, 2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 667: `assert_relative_eq!(band.get(0, 0).unwrap(), 5.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 668: `assert_relative_eq!(band.get(0, 1).unwrap(), 10.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 669: `assert_relative_eq!(band.get(0, 2).unwrap(), 14.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 670: `assert_relative_eq!(band.get(1, 0).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 671: `assert_relative_eq!(band.get(1, 1).unwrap(), 6.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 672: `assert_relative_eq!(band.get(1, 2).unwrap(), 11.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 673: `assert_relative_eq!(band.get(1, 3).unwrap(), 15.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 676: `assert_relative_eq!(band.get(0, 3).unwrap(), 0.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 677: `assert_relative_eq!(band.get(0, 4).unwrap(), 0.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 678: `assert_relative_eq!(band.get(3, 0).unwrap(), 0.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 702: `let band = BandedMatrix::new(data.view(), 1, 1, 4, 4).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 711: `let y = band.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 749: `let band = BandedMatrix::new(data.view(), 1, 1, 4, 4).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 758: `let y = band.matvec_transpose(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 793: `let band = BandedMatrix::new(data.view(), 1, 1, 3, 3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 795: `let dense = band.to_dense().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 819: `let band = BandedMatrix::from_matrix(&a.view(), 1, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 828: `let x = band.solve_tridiagonal(&b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 836: `let ax = band.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 844: `let x2 = band.solve(&b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/specialized/block_tridiagonal.rs

17 issues found:

- Line 509: `BlockTridiagonalMatrix::new(vec![a1, a2, a3], vec![b1, b2], vec![c1, c2]).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 527: `assert_eq!(matrix.get(0, 0).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 528: `assert_eq!(matrix.get(0, 1).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 529: `assert_eq!(matrix.get(2, 2).unwrap(), 5.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 530: `assert_eq!(matrix.get(5, 5).unwrap(), 12.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 533: `assert_eq!(matrix.get(0, 2).unwrap(), 13.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 534: `assert_eq!(matrix.get(1, 3).unwrap(), 16.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 535: `assert_eq!(matrix.get(2, 4).unwrap(), 17.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 538: `assert_eq!(matrix.get(2, 0).unwrap(), 21.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 539: `assert_eq!(matrix.get(4, 2).unwrap(), 25.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 542: `assert_eq!(matrix.get(0, 4).unwrap(), 0.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 544: `assert_eq!(matrix.get(5, 2).unwrap(), 27.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 550: `let dense = matrix.to_dense().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 574: `let y = matrix.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 577: `let dense = matrix.to_dense().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 594: `let y = matrix.matvec_transpose(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 597: `let dense = matrix.to_dense().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/specialized/symmetric.rs

24 issues found:

- Line 172: `l[[j, j]] = diag_val.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 179: `l[[i, j]] = (self.data[[i, j]] - sum) / l[[j, j]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 216: `y[i] = (b[i] - sum) / l[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 227: `x[i] = (y[i] - sum) / l[[i, i]];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 326: `let sym = SymmetricMatrix::from_matrix(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 332: `assert_relative_eq!(sym.get(0, 0).unwrap(), 1.0, epsilon = 1e-10);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 333: `assert_relative_eq!(sym.get(0, 1).unwrap(), 2.0, epsilon = 1e-10);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 334: `assert_relative_eq!(sym.get(1, 0).unwrap(), 2.0, epsilon = 1e-10);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 335: `assert_relative_eq!(sym.get(1, 1).unwrap(), 4.0, epsilon = 1e-10);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 336: `assert_relative_eq!(sym.get(1, 2).unwrap(), 5.0, epsilon = 1e-10);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 337: `assert_relative_eq!(sym.get(2, 1).unwrap(), 5.0, epsilon = 1e-10);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 338: `assert_relative_eq!(sym.get(2, 2).unwrap(), 6.0, epsilon = 1e-10);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 355: `let sym = SymmetricMatrix::from_matrix(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 358: `let y = sym.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 378: `let sym = SymmetricMatrix::from_matrix(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 381: `let y1 = sym.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 382: `let y2 = sym.matvec_transpose(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 396: `let sym = SymmetricMatrix::from_matrix(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 398: `let dense = sym.to_dense().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 414: `let sym = SymmetricMatrix::from_matrix(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 416: `let l = sym.cholesky().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 444: `let sym = SymmetricMatrix::from_matrix(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 450: `let x = sym.solve(&b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 453: `let ax = sym.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/specialized/tridiagonal.rs

25 issues found:

- Line 184: `x[0] = b[0] / self.diag[0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 193: `c_prime[0] = self.superdiag[0] / self.diag[0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 194: `d_prime[0] = b[0] / self.diag[0];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 206: `c_prime[i] = self.superdiag[i] / m;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 209: `d_prime[i] = (b[i] - self.subdiag[i - 1] * d_prime[i - 1]) / m;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 363: `let tri = TridiagonalMatrix::new(diag.view(), superdiag.view(), subdiag.view())....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 369: `assert_relative_eq!(tri.get(0, 0).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 370: `assert_relative_eq!(tri.get(0, 1).unwrap(), 5.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 371: `assert_relative_eq!(tri.get(1, 0).unwrap(), 8.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 372: `assert_relative_eq!(tri.get(1, 1).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 373: `assert_relative_eq!(tri.get(1, 2).unwrap(), 6.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 374: `assert_relative_eq!(tri.get(2, 1).unwrap(), 9.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 377: `assert_relative_eq!(tri.get(0, 2).unwrap(), 0.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 378: `assert_relative_eq!(tri.get(0, 3).unwrap(), 0.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 379: `assert_relative_eq!(tri.get(2, 0).unwrap(), 0.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 391: `let tri = TridiagonalMatrix::from_matrix(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 417: `let tri = TridiagonalMatrix::new(diag.view(), superdiag.view(), subdiag.view())....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 420: `let y = tri.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 443: `let tri = TridiagonalMatrix::new(diag.view(), superdiag.view(), subdiag.view())....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 446: `let y = tri.matvec_transpose(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 469: `let tri = TridiagonalMatrix::new(diag.view(), superdiag.view(), subdiag.view())....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 471: `let dense = tri.to_dense().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 491: `let tri = TridiagonalMatrix::new(diag.view(), superdiag.view(), subdiag.view())....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 497: `let x = tri.solve(&b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 500: `let ax = tri.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/stats/covariance.rs

6 issues found:

- Line 46: `let mean = data.mean_axis(Axis(0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 53: `let normalizer = F::from(n_samples - ddof).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 62: `let val = sum / normalizer;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 95: `std_devs[i] = cov[[i, i]].sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 104: `corr[[i, j]] = cov[[i, j]] / (std_devs[i] * std_devs[j]);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 161: `Ok(dist_sq.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation

### src/stats/distributions.rs

56 issues found:

- Line 102: `let min_dof = F::from(p).unwrap() - F::one();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 151: `let log_det_u = det(&params.row_cov.view(), None)?.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 152: `let log_det_v = det(&params.col_cov.view(), None)?.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 164: `let log_2pi = F::from(2.0 * PI).unwrap().ln();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 165: `let normalizer = -F::from(m * n).unwrap() * F::from(0.5).unwrap() * log_2pi`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 166: `- F::from(n).unwrap() * F::from(0.5).unwrap() * log_det_u`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 167: `- F::from(m).unwrap() * F::from(0.5).unwrap() * log_det_v;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 169: `Ok(normalizer - F::from(0.5).unwrap() * quad_form)`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 212: `let log_det_x = det(x, None)?.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 213: `let log_det_v = det(&params.scale.view(), None)?.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 221: `let log_2 = F::from(2.0).unwrap().ln();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 223: `let log_normalizer = params.dof * F::from(p).unwrap() * F::from(0.5).unwrap() * ...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 224: `+ F::from(0.25).unwrap() * F::from(p * (p - 1)).unwrap() * F::from(PI).unwrap()....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 226: `+ params.dof * F::from(0.5).unwrap() * log_det_v;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 229: `let main_term = (params.dof - F::from(p + 1).unwrap()) * F::from(0.5).unwrap() *...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 230: `- F::from(0.5).unwrap() * trace_term;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 319: `let chi_approx = z[[i, j]].abs() * (params.dof - F::from(i).unwrap()).sqrt();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 345: `let log_pi = F::from(PI).unwrap().ln();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 346: `let mut result = F::from(p * (p - 1)).unwrap() * F::from(0.25).unwrap() * log_pi...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 349: `let arg = x + (F::one() - F::from(j).unwrap()) * F::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 352: `(arg - F::from(0.5).unwrap()) * arg.ln() - arg`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 353: `+ F::from(0.5).unwrap() * F::from(2.0 * PI).unwrap().ln()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 375: `let params = MatrixNormalParams::new(mean, row_cov, col_cov).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 386: `let params = WishartParams::new(scale, dof).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 398: `let params = MatrixNormalParams::new(mean, row_cov, col_cov).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 399: `let logpdf = matrix_normal_logpdf(&x.view(), &params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 411: `let params = MatrixNormalParams::new(mean, row_cov, col_cov).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 412: `let sample = sample_matrix_normal(&params, Some(42)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 447: `let p = F::from(scale.nrows()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 487: `let p = F::from(x.nrows()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 491: `let log_det_x = det(x, None)?.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 499: `let log_det_psi = det(&params.scale.view(), None)?.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 514: `let half = F::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 515: `let two = F::from(2.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 516: `let pi = F::from(PI).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 520: `- half * nu * p * two.ln()`
  - **Fix**: Mathematical operation .ln() without validation
- Line 521: `- F::from(0.25).unwrap() * p * (p - F::one()) * pi.ln();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 526: `for j in 0..p.to_usize().unwrap() {`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 527: `let arg = half * (nu - F::from(j).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 530: `let ln_2pi = F::from(2.0 * PI).unwrap().ln();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 531: `log_gamma_p += (arg - half) * arg.ln() - arg + half * ln_2pi;`
  - **Fix**: Mathematical operation .ln() without validation
- Line 630: `let log_det_u = det(&params.scale_u.view(), None)?.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 631: `let log_det_v = det(&params.scale_v.view(), None)?.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 634: `let half = F::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 635: `let pi = F::from(PI).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 636: `let n_f = F::from(n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 637: `let p_f = F::from(p).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 641: `let log_norm = -half * n_f * log_det_u - half * p_f * log_det_v - half * n_f * p...`
  - **Fix**: Mathematical operation .ln() without validation
- Line 645: `log_norm - half * (nu + n_f + p_f - F::one()) * (F::one() + quadratic_form / nu)...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 645: `log_norm - half * (nu + n_f + p_f - F::one()) * (F::one() + quadratic_form / nu)...`
  - **Fix**: Mathematical operation .ln() without validation
- Line 660: `let params = InverseWishartParams::new(scale, dof).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 675: `let params = MatrixTParams::new(location, scale_u, scale_v, dof).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 692: `let params = InverseWishartParams::new(scale, 5.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 694: `let logpdf = inverse_wishart_logpdf(&x.view(), &params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 704: `let params = MatrixTParams::new(location, scale_u, scale_v, 3.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 706: `let logpdf = matrix_t_logpdf(&x.view(), &params).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/stats/sampling.rs

10 issues found:

- Line 260: `let scale_factor = (dof / (dof + chi_approx[[0, 0]] * chi_approx[[0, 0]])).sqrt(...`
  - **Fix**: Division without zero check - use safe_divide()
- Line 260: `let scale_factor = (dof / (dof + chi_approx[[0, 0]] * chi_approx[[0, 0]])).sqrt(...`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 433: `if log_alpha > uniform.ln() {`
  - **Fix**: Mathematical operation .ln() without validation
- Line 446: `let acceptance_rate = accepted as f64 / total_samples as f64;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 468: `let samples = sample_multivariate_normal(&mean.view(), &cov.view(), 100, Some(42...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 474: `let sample_mean = samples.mean_axis(ndarray::Axis(0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 483: `let bootstrap_samples = bootstrap_sample(&data.view(), 10, Some(2), Some(42)).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 496: `let permuted_samples = permutation_sample(&data.view(), 5, Some(42)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 511: `let params = MatrixNormalParams::new(mean, row_cov, col_cov).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 512: `let samples = sample_matrix_normal_multiple(&params, 5, Some(42)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/stats/tests.rs

40 issues found:

- Line 112: `let reg_factor = F::from(1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 128: `let log_det_i = det_i.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 138: `let weight = F::from(sample_sizes[i] - 1).unwrap() / F::from(total_dof).unwrap()...`
  - **Fix**: Use .get() with proper bounds checking
- Line 143: `let reg_factor = F::from(1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 158: `let log_det_pooled = det_pooled.ln();`
  - **Fix**: Mathematical operation .ln() without validation
- Line 161: `let mut m_statistic = F::from(total_dof).unwrap() * log_det_pooled;`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 164: `m_statistic -= F::from(sample_sizes[i] - 1).unwrap() * log_det_i;`
  - **Fix**: Use .get() with proper bounds checking
- Line 179: `let df = (k - 1) * p * (p + 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 240: `.powf(F::one() / F::from(p).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 241: `let arithmetic_mean = eigenvals.sum() / F::from(p).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 243: `let w_statistic = (geometric_mean / arithmetic_mean).powf(F::from(p).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 246: `let n_f = F::from(n - 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 247: `let _f = F::from(p * (p + 1) / 2 - 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 249: `-(n_f - F::from(2 * p * p + p + 2).unwrap() / F::from(6 * p).unwrap()) * w_stati...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 251: `let df = p * (p + 1) / 2 - 1;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 302: `let mean = data.mean_axis(Axis(0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 306: `let reg_factor = F::from(1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 339: `let skewness_stat = skewness_sum / (F::from(n).unwrap().powi(2));`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 343: `let kurtosis_stat = kurtosis_sum / F::from(n).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 346: `let skewness_chi2 = F::from(n).unwrap() * skewness_stat / F::from(6.0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 347: `let kurtosis_z = (kurtosis_stat - F::from(p * (p + 2)).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 348: `/ (F::from(8 * p * (p + 2)).unwrap() / F::from(n).unwrap()).sqrt();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 351: `let skewness_df = p * (p + 1) * (p + 2) / 6;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 356: `F::from(2.0).unwrap() * standard_normal_survival_function(kurtosis_z.abs());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 412: `let sample_mean = data.mean_axis(Axis(0)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 436: `let reg_factor = F::from(1e-10).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 444: `let t2_stat = F::from(n).unwrap() * diff.dot(&cov_inv).dot(&diff);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 448: `t2_stat * F::from(n - p).unwrap() / (F::from(n - 1).unwrap() * F::from(p).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 476: `sum_inv = sum_inv + F::one() / F::from(n_i - 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 480: `let inv_total = F::one() / F::from(total_dof).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 482: `let c1 = (F::from(2 * p * p + 3 * p - 1).unwrap() / F::from(6 * (p + 1) * (k - 1...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 500: `let z = (x - F::from(df).unwrap()) / F::from(2 * df).unwrap().sqrt();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 505: `let approx = (-x / F::from(2.0).unwrap()).exp();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 519: `let approx = F::one() / (F::one() + x);`
  - **Fix**: Division without zero check - use safe_divide()
- Line 529: `return F::from(0.5).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 533: `let approx = (-z * z / F::from(2.0).unwrap()).exp() / (z * F::from(2.0 * PI).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 534: `approx.min(F::from(0.5).unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 553: `let result = hotelling_t2_test(&data.view(), None, 0.05).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 597: `let (skewness_result, kurtosis_result) = mardia_normality_test(&data.view(), 0.0...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 627: `let result = box_m_test(&groups, 0.05).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/structured/circulant.rs

24 issues found:

- Line 196: `let circulant = CirculantMatrix::new(first_row.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 208: `assert_relative_eq!(circulant.get(0, 0).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 209: `assert_relative_eq!(circulant.get(0, 1).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 210: `assert_relative_eq!(circulant.get(0, 2).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 211: `assert_relative_eq!(circulant.get(0, 3).unwrap(), 4.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 213: `assert_relative_eq!(circulant.get(1, 0).unwrap(), 4.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 214: `assert_relative_eq!(circulant.get(1, 1).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 215: `assert_relative_eq!(circulant.get(1, 2).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 216: `assert_relative_eq!(circulant.get(1, 3).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 218: `assert_relative_eq!(circulant.get(2, 0).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 219: `assert_relative_eq!(circulant.get(2, 1).unwrap(), 4.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 220: `assert_relative_eq!(circulant.get(2, 2).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 221: `assert_relative_eq!(circulant.get(2, 3).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 223: `assert_relative_eq!(circulant.get(3, 0).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 224: `assert_relative_eq!(circulant.get(3, 1).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 225: `assert_relative_eq!(circulant.get(3, 2).unwrap(), 4.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 226: `assert_relative_eq!(circulant.get(3, 3).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 234: `let circulant = CirculantMatrix::from_kernel(kernel.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 249: `let circulant = CirculantMatrix::new(first_row.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 258: `let y = circulant.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 273: `let circulant = CirculantMatrix::new(first_row.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 282: `let y = circulant.matvec_transpose(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 297: `let circulant = CirculantMatrix::new(first_row.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 299: `let dense = circulant.to_dense().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/structured/hankel.rs

41 issues found:

- Line 176: `result[i] += self.get(i, j).unwrap() * x[j];`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 196: `result[j] += self.get(i, j).unwrap() * x[i];`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 215: `let hankel = HankelMatrix::new(first_col.view(), last_row.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 226: `assert_relative_eq!(hankel.get(0, 0).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 227: `assert_relative_eq!(hankel.get(0, 1).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 228: `assert_relative_eq!(hankel.get(0, 2).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 229: `assert_relative_eq!(hankel.get(1, 0).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 230: `assert_relative_eq!(hankel.get(1, 1).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 231: `assert_relative_eq!(hankel.get(1, 2).unwrap(), 4.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 232: `assert_relative_eq!(hankel.get(2, 0).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 233: `assert_relative_eq!(hankel.get(2, 1).unwrap(), 4.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 234: `assert_relative_eq!(hankel.get(2, 2).unwrap(), 5.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 241: `let hankel = HankelMatrix::from_sequence(sequence.view(), 3, 3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 252: `assert_relative_eq!(hankel.get(0, 0).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 253: `assert_relative_eq!(hankel.get(0, 1).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 254: `assert_relative_eq!(hankel.get(0, 2).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 255: `assert_relative_eq!(hankel.get(1, 0).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 256: `assert_relative_eq!(hankel.get(1, 1).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 257: `assert_relative_eq!(hankel.get(1, 2).unwrap(), 4.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 258: `assert_relative_eq!(hankel.get(2, 0).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 259: `assert_relative_eq!(hankel.get(2, 1).unwrap(), 4.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 260: `assert_relative_eq!(hankel.get(2, 2).unwrap(), 5.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 269: `let hankel = HankelMatrix::new(first_col.view(), last_row.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 281: `assert_relative_eq!(hankel.get(0, 0).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 282: `assert_relative_eq!(hankel.get(0, 1).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 283: `assert_relative_eq!(hankel.get(0, 2).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 285: `assert_relative_eq!(hankel.get(1, 0).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 286: `assert_relative_eq!(hankel.get(1, 1).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 287: `assert_relative_eq!(hankel.get(1, 2).unwrap(), 4.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 289: `assert_relative_eq!(hankel.get(2, 0).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 290: `assert_relative_eq!(hankel.get(2, 1).unwrap(), 4.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 291: `assert_relative_eq!(hankel.get(2, 2).unwrap(), 5.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 293: `assert_relative_eq!(hankel.get(3, 0).unwrap(), 4.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 294: `assert_relative_eq!(hankel.get(3, 1).unwrap(), 5.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 295: `assert_relative_eq!(hankel.get(3, 2).unwrap(), 6.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 303: `let hankel = HankelMatrix::new(first_col.view(), last_row.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 311: `let y = hankel.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 326: `let hankel = HankelMatrix::new(first_col.view(), last_row.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 334: `let y = hankel.matvec_transpose(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 349: `let hankel = HankelMatrix::new(first_col.view(), last_row.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 351: `let dense = hankel.to_dense().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/structured/mod.rs

2 issues found:

- Line 107: `matrix.matvec(x).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 122: `matrix_clone.matvec(x).unwrap()`
  - **Fix**: Replace with ? operator or .ok_or()

### src/structured/toeplitz.rs

36 issues found:

- Line 277: `let toeplitz = ToeplitzMatrix::new(first_row.view(), first_col.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 283: `assert_relative_eq!(toeplitz.get(0, 0).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 284: `assert_relative_eq!(toeplitz.get(0, 1).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 285: `assert_relative_eq!(toeplitz.get(0, 2).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 286: `assert_relative_eq!(toeplitz.get(1, 0).unwrap(), 4.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 287: `assert_relative_eq!(toeplitz.get(1, 1).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 288: `assert_relative_eq!(toeplitz.get(1, 2).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 289: `assert_relative_eq!(toeplitz.get(2, 0).unwrap(), 5.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 290: `assert_relative_eq!(toeplitz.get(2, 1).unwrap(), 4.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 291: `assert_relative_eq!(toeplitz.get(2, 2).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 298: `let toeplitz = ToeplitzMatrix::new_symmetric(first_row.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 304: `assert_relative_eq!(toeplitz.get(0, 0).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 305: `assert_relative_eq!(toeplitz.get(0, 1).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 306: `assert_relative_eq!(toeplitz.get(0, 2).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 307: `assert_relative_eq!(toeplitz.get(1, 0).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 308: `assert_relative_eq!(toeplitz.get(1, 1).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 309: `assert_relative_eq!(toeplitz.get(1, 2).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 310: `assert_relative_eq!(toeplitz.get(2, 0).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 311: `assert_relative_eq!(toeplitz.get(2, 1).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 312: `assert_relative_eq!(toeplitz.get(2, 2).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 321: `let toeplitz = ToeplitzMatrix::from_parts(c, r.view(), l.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 327: `assert_relative_eq!(toeplitz.get(0, 0).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 328: `assert_relative_eq!(toeplitz.get(0, 1).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 329: `assert_relative_eq!(toeplitz.get(0, 2).unwrap(), 3.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 330: `assert_relative_eq!(toeplitz.get(1, 0).unwrap(), 4.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 331: `assert_relative_eq!(toeplitz.get(1, 1).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 332: `assert_relative_eq!(toeplitz.get(1, 2).unwrap(), 2.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 333: `assert_relative_eq!(toeplitz.get(2, 0).unwrap(), 5.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 334: `assert_relative_eq!(toeplitz.get(2, 1).unwrap(), 4.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 335: `assert_relative_eq!(toeplitz.get(2, 2).unwrap(), 1.0);`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 343: `let toeplitz = ToeplitzMatrix::new(first_row.view(), first_col.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 351: `let y = toeplitz.matvec(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 366: `let toeplitz = ToeplitzMatrix::new(first_row.view(), first_col.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 374: `let y = toeplitz.matvec_transpose(&x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 389: `let toeplitz = ToeplitzMatrix::new(first_row.view(), first_col.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 391: `let dense = toeplitz.to_dense().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/structured/utils.rs

8 issues found:

- Line 83: `let pad = (nb - 1) / 2;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 273: `let result = convolution(a.view(), b.view(), "full").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 288: `let result = convolution(a.view(), b.view(), "same").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 303: `let result = convolution(a.view(), b.view(), "valid").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 317: `let result = circular_convolution(a.view(), b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 343: `assert_eq!(result.unwrap().len(), 0);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 360: `let x = solve_toeplitz(c.view(), r.view(), b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 387: `let x = solve_circulant(c.view(), b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/tensor_contraction/cp.rs

24 issues found:

- Line 227: `Ok((diff_squared_sum / orig_squared_sum).sqrt())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 227: `Ok((diff_squared_sum / orig_squared_sum).sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 356: `factor[[i, j]] = A::from(((i + 1) * (j + 1)) % 10).unwrap() / A::from(10).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 399: `let rel_improvement = (prev_error - error) / prev_error;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 535: `result_rows = result.as_ref().unwrap().shape()[0];`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 593: `if s[i] > A::epsilon() * A::from(10.0).unwrap() {`
  - **Fix**: Use .get() with proper bounds checking
- Line 594: `s_inv[[i, i]] = A::one() / s[i];`
  - **Fix**: Division without zero check - use safe_divide()
- Line 627: `let norm = norm_sq.sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 659: `let cp = cp_als(&tensor.view(), 2, 50, 1e-4, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 669: `assert_eq!(cp.weights.as_ref().unwrap().len(), 2);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 672: `let _reconstructed = cp.to_full().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 675: `let error = cp.reconstruction_error(&tensor.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 688: `let cp = cp_als(&tensor.view(), 2, 50, 1e-4, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 697: `let _reconstructed = cp.to_full().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 700: `let error = cp.reconstruction_error(&tensor.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 715: `let cp = cp_als(&tensor.view(), 4, 50, 1e-4, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 718: `let compressed = cp.compress(2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 728: `assert_eq!(compressed.weights.as_ref().unwrap().len(), 2);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 731: `let error_orig = cp.reconstruction_error(&tensor.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 732: `let error_comp = compressed.reconstruction_error(&tensor.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 761: `let cp = CanonicalPolyadic::new(factors, None, Some(vec![2, 3, 2])).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 764: `let reconstructed = cp.to_full().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 790: `let kr = khatri_rao_product(&factors, 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 801: `let kr = khatri_rao_product(&factors, 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/tensor_contraction/mod.rs

18 issues found:

- Line 229: `let mut result_tensor = result.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 233: `Ok(Arc::try_unwrap(result).unwrap().into_inner().unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 405: `let mut result_tensor = result.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 416: `Ok(Arc::try_unwrap(result).unwrap().into_inner().unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 599: `let mut result_tensor = result.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 605: `Ok(Arc::try_unwrap(result).unwrap().into_inner().unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 850: `let mut result_tensor = result.lock().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 854: `Ok(Arc::try_unwrap(result).unwrap().into_inner().unwrap())`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 941: `let unfolded = unfold(&tensor_dyn, *mode).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 944: `let (u, _, _) = svd_truncated(&unfolded, rank[*mode]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1100: `let result = contract(&a.view(), &b.view(), &[1], &[0]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1128: `let result = batch_matmul(&a.view(), &b.view(), 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1156: `let result = einsum("ij,jk->ik", &[&a_view, &b_view]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1178: `let result = einsum("i,i->", &[&a_view, &b_view]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1185: `assert_abs_diff_eq!(result.iter().next().unwrap(), &expected, epsilon = 1e-10);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1201: `let result = mode_n_product(&tensor.view(), &matrix.view(), 0).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1218: `let (core, factors) = hosvd(&tensor.view(), &[2, 2, 2]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1231: `reconstructed = mode_n_product(&reconstructed.view(), &factor.view(), mode).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()

### src/tensor_contraction/tensor_network.rs

31 issues found:

- Line 675: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 678: `let node = TensorNode::new(data, indices).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 690: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 693: `let node = TensorNode::new(data, indices).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 696: `let transposed = node.transpose(&["j".to_string(), "i".to_string()]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 715: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 717: `let node1 = TensorNode::new(data1, indices1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 723: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 725: `let node2 = TensorNode::new(data2, indices2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 728: `let result = node1.contract(&node2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 744: `let data1 = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0, 2.0]).unwrap(...`
  - **Fix**: Handle array creation errors properly
- Line 746: `let node1 = TensorNode::new(data1, indices1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 748: `let data2 = ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![3.0, 4.0, 5.0]).un...`
  - **Fix**: Handle array creation errors properly
- Line 750: `let node2 = TensorNode::new(data2, indices2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 753: `let result = node1.outer_product(&node2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 772: `ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap...`
  - **Fix**: Handle array creation errors properly
- Line 774: `let node = TensorNode::new(data, indices).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 777: `let result = node.trace("i", "j").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 792: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 794: `let node = TensorNode::new(data, indices).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 797: `let result = node.add_dummy_index("k", 1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 820: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 822: `let node = TensorNode::new(data, indices).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 825: `let result = node.remove_index("j").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 841: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 843: `let node1 = TensorNode::new(data1, indices1).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 851: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 853: `let node2 = TensorNode::new(data2, indices2).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 859: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 861: `let node3 = TensorNode::new(data3, indices3).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 867: `let result = network.contract_all().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/tensor_contraction/tensor_train.rs

13 issues found:

- Line 682: `s.mapv(|x| x / s[0])`
  - **Fix**: Division without zero check - use safe_divide()
- Line 742: `s.mapv(|x| x / s[0])`
  - **Fix**: Division without zero check - use safe_divide()
- Line 802: `let tt = tensor_train_decomposition(&tensor.view(), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 822: `let reconstructed = tt.to_full().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 857: `let tt = tensor_train_decomposition(&tensor.view(), Some(2), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 865: `let reconstructed = tt.to_full().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 892: `let tt = tensor_train_decomposition(&tensor.view(), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 898: `let value = tt.get(&[i, j, k]).unwrap();`
  - **Fix**: Use .get().ok_or(Error::IndexOutOfBounds)?
- Line 924: `let tt = tensor_train_decomposition(&tensor.view(), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 928: `let rounded_tt = tt.round(*epsilon).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 931: `let reconstructed = rounded_tt.to_full().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 935: `let norm = tensor.mapv(|x| x * x).sum().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 949: `let relative_error = max_error / norm;`
  - **Fix**: Division without zero check - use safe_divide()

### src/tensor_contraction/tucker.rs

18 issues found:

- Line 200: `Ok((diff_squared_sum / orig_squared_sum).sqrt())`
  - **Fix**: Division without zero check - use safe_divide()
- Line 200: `Ok((diff_squared_sum / orig_squared_sum).sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 257: `s.mapv(|v| v / s[[0, 0]])`
  - **Fix**: Division without zero check - use safe_divide()
- Line 291: `s.mapv(|v| v / s[[0, 0]])`
  - **Fix**: Division without zero check - use safe_divide()
- Line 547: `let rel_improvement = (prev_error - error) / prev_error;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 663: `let tucker = tucker_decomposition(&tensor.view(), &[2, 3, 2]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 673: `let reconstructed = tucker.to_full().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 699: `let tucker = tucker_decomposition(&tensor.view(), &[2, 2, 2]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 709: `let _reconstructed = tucker.to_full().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 712: `let error = tucker.reconstruction_error(&tensor.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 727: `let tucker = tucker_als(&tensor.view(), &[2, 2, 2], 10, 1e-4).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 737: `let _reconstructed = tucker.to_full().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 740: `let hosvd_tucker = tucker_decomposition(&tensor.view(), &[2, 2, 2]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 741: `let als_error = tucker.reconstruction_error(&tensor.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 742: `let hosvd_error = hosvd_tucker.reconstruction_error(&tensor.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 758: `let tucker = tucker_decomposition(&tensor.view(), &[2, 3, 2]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 761: `let compressed = tucker.compress(Some(vec![2, 2, 2]), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 770: `let compressed_eps = tucker.compress(None, Some(0.1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### src/tensor_train.rs

35 issues found:

- Line 166: `full_size as f64 / tt_size as f64`
  - **Fix**: Division without zero check - use safe_divide()
- Line 274: `Ok(norm_squared.sqrt())`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 317: `if energy.sqrt() > abs_tolerance {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 428: `let frobenius_norm = tensor.iter().map(|&x| x * x).sum::<F>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 430: `let abs_tolerance = tolerance * frobenius_norm / F::from(d - 1).unwrap().sqrt();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 469: `if error_estimate.sqrt() > abs_tolerance {`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 513: `let remaining_size = expected_elements / r_prev;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 704: `let tt_tensor = TTTensor::new(vec![core1, core2]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 755: `let tt_tensor = TTTensor::new(vec![core1, core2]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 759: `tt_tensor.get_element(&[0, 0]).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 764: `tt_tensor.get_element(&[0, 1]).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 769: `tt_tensor.get_element(&[1, 0]).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 774: `tt_tensor.get_element(&[1, 1]).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 785: `let tt_tensor = tt_decomposition(&tensor.view(), 1e-12, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 792: `let reconstructed = tt_tensor.get_element(&[i, j, k]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 803: `let tt_tensor = TTTensor::new(vec![core1]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 805: `let norm = tt_tensor.frobenius_norm().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 811: `let expected_norm = (1.0 + 4.0_f64).sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 815: `let dense = tt_tensor.to_dense().unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 816: `let dense_norm = dense.iter().map(|&x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 824: `let tt_a = TTTensor::new(vec![core1_a]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 827: `let tt_b = TTTensor::new(vec![core1_b]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 829: `let tt_sum = tt_add(&tt_a, &tt_b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 832: `assert_relative_eq!(tt_sum.get_element(&[0]).unwrap(), 3.0, epsilon = 1e-10); //...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 833: `assert_relative_eq!(tt_sum.get_element(&[1]).unwrap(), 5.0, epsilon = 1e-10);`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 841: `let tt_a = TTTensor::new(vec![core1_a]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 844: `let tt_b = TTTensor::new(vec![core1_b]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 846: `let tt_product = tt_hadamard(&tt_a, &tt_b).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 849: `assert_relative_eq!(tt_product.get_element(&[0]).unwrap(), 2.0, epsilon = 1e-10)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 850: `assert_relative_eq!(tt_product.get_element(&[1]).unwrap(), 6.0, epsilon = 1e-10)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 858: `let tt_tensor = tt_decomposition(&tensor.view(), 1e-12, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 861: `let rounded = tt_tensor.round(1e-1, Some(2)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 867: `let original = tt_tensor.get_element(&[i, j, k]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 868: `let rounded_val = rounded.get_element(&[i, j, k]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 879: `let tt_tensor = TTTensor::new(vec![core1]).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/basic_extended_precision_tests.rs

3 issues found:

- Line 14: `let c = extended_matmul::<_, f64>(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 35: `let y = extended_matvec::<_, f64>(&a.view(), &x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 53: `let x = extended_solve::<_, f64>(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/basic_extended_tests.rs

3 issues found:

- Line 14: `let c = extended_matmul::<_, f64>(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 35: `let y = extended_matvec::<_, f64>(&a.view(), &x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 53: `let x = extended_solve::<_, f64>(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/debug_3x3_eigentest.rs

1 issues found:

- Line 12: `let (eigenvalues, eigenvectors) = eigh(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/debug_matrix_rank.rs

9 issues found:

- Line 16: `let default_rank = matrix_rank(&matrix.view(), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 19: `let tight_tol_rank = matrix_rank(&matrix.view(), Some(1e-14), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 22: `let loose_tol_rank = matrix_rank(&matrix.view(), Some(1e-12), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 26: `let (_, s, _) = svd(&matrix.view(), false, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 54: `let rank2 = matrix_rank(&matrix2.view(), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 57: `let (_, s2, _) = svd(&matrix2.view(), false, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 82: `let rank3 = matrix_rank(&matrix3.view(), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 85: `let (_, s3, _) = svd(&matrix3.view(), false, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 101: `let strict_rank3 = matrix_rank(&matrix3.view(), Some(1e-12), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/eigen_sparse_tests.rs

8 issues found:

- Line 16: `let (eigenvalues, eigenvectors) = largest_k_eigh(&a.view(), 2, 100, 1e-10).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 45: `let (eigenvalues, eigenvectors) = smallest_k_eigh(&a.view(), 2, 100, 1e-10).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 74: `let (eigenvalues, eigenvectors) = largest_k_eigh(&a.view(), 2, 100, 1e-10).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 104: `let (eigenvalues, eigenvectors) = smallest_k_eigh(&a.view(), 2, 100, 1e-10).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 128: `let (eigenvalues, eigenvectors) = largest_k_eigh(&a.view(), 0, 100, 1e-10).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 133: `let (eigenvalues, eigenvectors) = smallest_k_eigh(&a.view(), 0, 100, 1e-10).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 149: `let (eigenvalues, eigenvectors) = largest_k_eigh(&a.view(), 4, 100, 1e-10).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 160: `let (eigenvalues, eigenvectors) = smallest_k_eigh(&a.view(), 4, 100, 1e-10).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/enhanced_op_tests.rs

16 issues found:

- Line 33: `let d = det(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 40: `let y = matvec(&a.view(), &x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 54: `let ip = inner_product(&v1.view(), &v2.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 70: `assert!(is_hermitian(&h.view(), 1e-10).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 71: `assert!(!is_hermitian(&nh.view(), 1e-10).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 84: `assert!(is_unitary(&u.view(), 1e-10).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 85: `assert!(!is_unitary(&nu.view(), 1e-10).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 93: `let h_part = hermitian_part(&mixed.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 94: `assert!(is_hermitian(&h_part.view(), 1e-10).unwrap());`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 107: `let exp_zero = matrix_exp(&zero.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 145: `let exp_non_trivial = matrix_exp(&non_trivial.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 234: `let vjp = vector_jacobian_product(g, &x.view(), &v.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 241: `let hvp = hessian_vector_product(f, &x.view(), &v.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 256: `let grad = matrix_gradient(matrix_f, &matrix_x.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 269: `let approx2 = taylor_approximation(f, &taylor_x.view(), &taylor_y.view(), 2, Non...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 272: `let actual = f(&taylor_y.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/extended_precision_tests.rs

7 issues found:

- Line 19: `let c = extended_matmul::<_, f64>(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 37: `let (p, l, u) = extended_lu::<_, f64>(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 59: `let (q, r) = extended_qr::<_, f64>(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 90: `let l = extended_cholesky::<_, f64>(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 108: `let eigvals = extended_eigvalsh::<_, f64>(&a.view(), None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 123: `let x = extended_solve::<_, f64>(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 138: `let (u, s, vh) = extended_svd::<_, f64>(&a.view(), true, None, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/minimal_extended_precision_tests.rs

3 issues found:

- Line 14: `let c = extended_matmul::<_, f64>(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 35: `let y = extended_matvec::<_, f64>(&a.view(), &x.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 53: `let x = extended_solve::<_, f64>(&a.view(), &b.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/scipy_compat_api_stability.rs

56 issues found:

- Line 165: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 180: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 207: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 222: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 524: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 549: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 684: `let det_2x2: f64 = compat::det(&identity_2x2.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 685: `let det_3x3: f64 = compat::det(&identity_3x3.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 686: `let det_5x5: f64 = compat::det(&identity_5x5.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 694: `let known_det: f64 = compat::det(&known_det_matrix.view(), false, true).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 700: `let computed_inverse = compat::inv(&known_inv_matrix.view(), false, true).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 722: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 734: `compat::norm(&norm_test_matrix.view(), Some("fro"), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 738: `compat::norm(&norm_test_matrix.view(), Some("1"), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 742: `compat::norm(&norm_test_matrix.view(), Some("inf"), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 755: `let det = compat::det(&rotation.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 759: `let cond = compat::cond(&rotation.view(), Some("2")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 763: `let (q_opt, r) = compat::qr(&rotation.view(), false, None, "full", false, true)....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 766: `let reconstruction_error = (&rotation - &q).mapv(|x| x * x).sum().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 767: `let sign_flip_error = (&rotation + &q).mapv(|x| x * x).sum().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 797: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 818: `let det = compat::det(&symmetric.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 833: `let rank = compat::matrix_rank(&nearly_singular.view(), Some(1e-7), false, true)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 837: `let cond = compat::cond(&nearly_singular.view(), Some("2")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 844: `let det_small: f64 = compat::det(&small_matrix.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 875: `let det1: f64 = compat::det(&test_matrix.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 876: `let det2: f64 = compat::det(&test_matrix.view(), false, true).unwrap(); // Same ...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 881: `let norm_default: f64 = compat::norm(&test_matrix.view(), None, None, false, tru...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 883: `compat::norm(&test_matrix.view(), Some("fro"), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 889: `let vnorm_default: f64 = compat::vector_norm(&vector.view(), None, true).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 890: `let vnorm_explicit: f64 = compat::vector_norm(&vector.view(), Some(2.0), true).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 915: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 931: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 936: `let pinv1 = compat::pinv(&test_matrix.view(), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 937: `let pinv2 = compat::pinv(&test_matrix.view(), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 951: `let _det: f64 = compat::det(&f64_matrix.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 952: `let _inv: Array2<f64> = compat::inv(&f64_matrix.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 953: `let _norm: f64 = compat::norm(&f64_matrix.view(), Some("fro"), None, false, true...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 954: `let _vnorm: f64 = compat::vector_norm(&f64_vector.view(), Some(2.0), true).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 958: `compat::lu(&f64_matrix.view(), false, false, true, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 960: `compat::qr(&f64_matrix.view(), false, None, "full", false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 974: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1011: `let det = compat::det(&base_matrix.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1012: `let cond = compat::cond(&base_matrix.view(), Some("2")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1013: `let rank = compat::matrix_rank(&base_matrix.view(), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1014: `let norm = compat::norm(&base_matrix.view(), Some("fro"), None, false, true).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1022: `let (_, l, u) = compat::lu(&base_matrix.view(), false, false, true, false).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1023: `let (_, r) = compat::qr(&base_matrix.view(), false, None, "full", false, true).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1037: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1041: `compat::norm(&l.view(), Some("fro"), None, false, true).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1045: `compat::norm(&u.view(), Some("fro"), None, false, true).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1049: `compat::norm(&r.view(), Some("fro"), None, false, true).unwrap(),`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1065: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1067: `compat::norm(&solution.view(), Some("fro"), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1071: `let matrix_exp = compat::expm(&(base_matrix.clone() * 0.1).view(), None).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1073: `compat::norm(&matrix_exp.view(), Some("fro"), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/scipy_compat_comprehensive_tests.rs

99 issues found:

- Line 52: `let det_result = compat::det(&identity_2x2.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 57: `let det_result = compat::det(&identity_3x3.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 62: `let det_result = compat::det(&upper_triangular.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 67: `let det_result = compat::det(&singular.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 72: `let det_result = compat::det(&well_conditioned.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 83: `let inv_result = compat::inv(&a_2x2.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 94: `let inv_result = compat::inv(&a_3x3.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 107: `let fro_norm = compat::norm(&test_matrix.view(), Some("fro"), None, false, true)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 108: `assert!(scalars_close(fro_norm, 30.0_f64.sqrt(), TEST_TOL));`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 111: `let norm_1 = compat::norm(&test_matrix.view(), Some("1"), None, false, true).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 115: `let norm_inf = compat::norm(&test_matrix.view(), Some("inf"), None, false, true)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 120: `let zero_norm = compat::norm(&zeros.view(), Some("fro"), None, false, true).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 124: `let ones_fro_norm = compat::norm(&ones.view(), Some("fro"), None, false, true).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 133: `let norm_2 = compat::vector_norm(&test_vector.view(), Some(2.0), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 134: `assert!(scalars_close(norm_2, (50.0_f64).sqrt(), TEST_TOL));`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 137: `let norm_1 = compat::vector_norm(&test_vector.view(), Some(1.0), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 141: `let norm_inf = compat::vector_norm(&test_vector.view(), Some(f64::INFINITY), tru...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 145: `let norm_0 = compat::vector_norm(&test_vector.view(), Some(0.0), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 149: `let norm_3 = compat::vector_norm(&test_vector.view(), Some(3.0), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 159: `let pinv_result = compat::pinv(&square_full_rank.view(), None, false, true).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 160: `let inv_result = compat::inv(&square_full_rank.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 165: `let pinv_tall = compat::pinv(&tall_matrix.view(), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 174: `let pinv_wide = compat::pinv(&wide_matrix.view(), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 183: `let pinv_rank_def = compat::pinv(&rank_deficient.view(), None, false, true).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 195: `let cond_result = compat::cond(&well_conditioned.view(), Some("2")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 200: `let cond_result = compat::cond(&moderate.view(), Some("2")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 206: `let cond_1 = compat::cond(&test_matrix.view(), Some("1")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 207: `let cond_inf = compat::cond(&test_matrix.view(), Some("inf")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 208: `let cond_fro = compat::cond(&test_matrix.view(), Some("fro")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 220: `let rank = compat::matrix_rank(&full_rank_2x2.view(), None, false, true).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 224: `let rank = compat::matrix_rank(&full_rank_3x3.view(), None, false, true).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 229: `let rank = compat::matrix_rank(&rank_1.view(), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 234: `let rank = compat::matrix_rank(&zero_matrix.view(), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 239: `let rank = compat::matrix_rank(&rect_full_rank.view(), None, false, true).unwrap...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 258: `let (p, l, u) = compat::lu(&matrix.view(), false, false, true, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 291: `let (q_opt, r) = compat::qr(&matrix.view(), false, None, "full", false, true).un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 293: `let q = q_opt.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 323: `compat::svd(&matrix.view(), true, true, false, true, "gesdd").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 325: `let u = u_opt.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 326: `let vt = vt_opt.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 369: `let l = compat::cholesky(&matrix.view(), true, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 383: `let u = compat::cholesky(&matrix.view(), false, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 400: `let (u_right, p_right) = compat::polar(&matrix.view(), "right").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 412: `let (p_left, u_left) = compat::polar(&matrix.view(), "left").unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 428: `let (r, q) = compat::rq(&matrix.view(), false, None, "full", true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 454: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 472: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 491: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 512: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 549: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 575: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 587: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 629: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 656: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 690: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 715: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 725: `let det = compat::det(&pos_def.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 744: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 763: `let exp_zero = compat::expm(&zero_matrix.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 769: `let exp_diag = compat::expm(&diag_matrix.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 775: `let exp_nilpotent = compat::expm(&nilpotent.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 781: `let exp_antisym = compat::expm(&antisymmetric.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 788: `let det_exp = compat::det(&exp_antisym.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 798: `let log_identity = compat::logm(&identity.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 804: `let log_pos_def = compat::logm(&pos_def.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 807: `let exp_log = compat::expm(&log_pos_def.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 812: `let log_diag = compat::logm(&diag_matrix.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 813: `let expected = array![[2.0_f64.ln(), 0.0], [0.0, 3.0_f64.ln()]];`
  - **Fix**: Mathematical operation .ln() without validation
- Line 823: `let sqrt_identity = compat::sqrtm(&identity.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 828: `let sqrt_pos_def = compat::sqrtm(&pos_def.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 836: `let sqrt_diag = compat::sqrtm(&diag_matrix.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 842: `let sqrt_zero = compat::sqrtm(&zero_matrix.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 852: `let exp_via_funm = compat::funm(&test_matrix.view(), "exp", false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 853: `let exp_direct = compat::expm(&test_matrix.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 857: `let log_via_funm = compat::funm(&test_matrix.view(), "log", false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 858: `let log_direct = compat::logm(&test_matrix.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 862: `let sqrt_via_funm = compat::funm(&test_matrix.view(), "sqrt", false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 863: `let sqrt_direct = compat::sqrtm(&test_matrix.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 867: `let cos_via_funm = compat::funm(&test_matrix.view(), "cos", false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 868: `let cos_direct = compat::cosm(&test_matrix.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 871: `let sin_via_funm = compat::funm(&test_matrix.view(), "sin", false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 872: `let sin_direct = compat::sinm(&test_matrix.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 875: `let tan_via_funm = compat::funm(&test_matrix.view(), "tan", false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 876: `let tan_direct = compat::tanm(&test_matrix.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 897: `let block_diag = compat::block_diag(&blocks).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 917: `let mixed_diag = compat::block_diag(&mixed_blocks).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 922: `let single_diag = compat::block_diag(&single_block).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1048: `let det_a = compat::det(&a.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1051: `let cond_a = compat::cond(&a.view(), Some("2")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1054: `let rank_a = compat::matrix_rank(&a.view(), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1059: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1066: `let (p, l, u) = compat::lu(&a.view(), false, false, true, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1067: `let (q_opt, r) = compat::qr(&a.view(), false, None, "full", false, true).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1081: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1094: `let exp_a = compat::expm(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1095: `let log_exp_a = compat::logm(&exp_a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1100: `let pinv_a_rect = compat::pinv(&a_rect.view(), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1128: `let _det = compat::det(&matrix.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 1129: `let _norm = compat::norm(&matrix.view(), Some("fro"), None, false, true).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/scipy_compat_validation.rs

76 issues found:

- Line 36: `let det1 = compat::det(&a1.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 41: `let det2 = compat::det(&a2.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 46: `let det3 = compat::det(&a3.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 55: `let inv_a = compat::inv(&a.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 61: `let inv_magic = compat::inv(&magic.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 79: `let fro_norm = compat::norm(&test_matrix.view(), Some("fro"), None, false, true)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 82: `let norm_1 = compat::norm(&test_matrix.view(), Some("1"), None, false, true).unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 85: `let norm_inf = compat::norm(&test_matrix.view(), Some("inf"), None, false, true)...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 91: `let vec_norm_2 = compat::vector_norm(&test_vector.view(), Some(2.0), true).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 94: `let vec_norm_1 = compat::vector_norm(&test_vector.view(), Some(1.0), true).unwra...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 98: `compat::vector_norm(&test_vector.view(), Some(f64::INFINITY), true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 119: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 141: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 161: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 171: `let exp_zero = compat::expm(&zero_2x2.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 177: `let exp_diag = compat::expm(&diag_matrix.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 182: `let sqrt_identity = compat::sqrtm(&identity.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 187: `let sqrt_diag = compat::sqrtm(&diag_squares.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 197: `compat::qr(&simple_matrix.view(), false, None, "full", false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 212: `let (_, s, _) = compat::svd(&rank1.view(), true, true, false, true, "gesdd").unw...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 230: `let det_a = compat::det(&a.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 231: `let det_b = compat::det(&b.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 232: `let det_ab = compat::det(&a.dot(&b).view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 237: `let det_at = compat::det(&a.t().view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 243: `let det_ka = compat::det(&ka.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 250: `let inv_a = compat::inv(&a.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 253: `let inv_inv_a = compat::inv(&inv_a.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 257: `let inv_at = compat::inv(&a.t().view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 262: `let det_a = compat::det(&a.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 263: `let det_inv_a = compat::det(&inv_a.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 264: `assert!(close_f64(det_inv_a, 1.0 / det_a, VALIDATION_TOL));`
  - **Fix**: Division without zero check - use safe_divide()
- Line 273: `let norm_a = compat::norm(&a.view(), Some("fro"), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 274: `let norm_b = compat::norm(&b.view(), Some("fro"), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 275: `let norm_sum = compat::norm(&(&a + &b).view(), Some("fro"), None, false, true).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 282: `let norm_ka = compat::norm(&ka.view(), Some("fro"), None, false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 287: `let norm_zero = compat::norm(&zero_matrix.view(), Some("fro"), None, false, true...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 308: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 316: `let det = compat::det(&a.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 328: `let error = diff.iter().map(|&x| x * x).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 339: `let log_a = compat::logm(&a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 340: `let exp_log_a = compat::expm(&log_a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 345: `let exp_small_a = compat::expm(&small_a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 346: `let log_exp_small_a = compat::logm(&exp_small_a.view()).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 350: `let sqrt_a = compat::sqrtm(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 358: `let exp_sum = compat::expm(&(&diag_a + &diag_b).view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 359: `let exp_a = compat::expm(&diag_a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 360: `let exp_b = compat::expm(&diag_b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 371: `let (p, l, u) = compat::lu(&a.view(), false, false, true, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 377: `let (q_opt, r) = compat::qr(&a.view(), false, None, "full", false, true).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 389: `let (u_opt, s, vt_opt) = compat::svd(&a.view(), true, true, false, true, "gesdd"...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 408: `let cond_identity = compat::cond(&identity.view(), Some("2")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 413: `let cond_random = compat::cond(&random_matrix.view(), Some("2")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 419: `let cond_scaled = compat::cond(&scaled_matrix.view(), Some("2")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 427: `let rank_identity = compat::matrix_rank(&identity_3.view(), None, false, true).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 434: `let rank_original = compat::matrix_rank(&test_matrix.view(), None, false, true)....`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 436: `let rank_transformed = compat::matrix_rank(&transformed.view(), None, false, tru...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 444: `let rank_outer = compat::matrix_rank(&outer_product.view(), None, false, true).u...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 471: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 473: `let residual_norm = residual.iter().map(|&r| r * r).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 478: `let cond_num = compat::cond(&well_conditioned.view(), Some("2")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 485: `let angle = PI / 4.0;`
  - **Fix**: Division without zero check - use safe_divide()
- Line 489: `let det = compat::det(&rotation.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 493: `let cond_num = compat::cond(&rotation.view(), Some("2")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 526: `.unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 554: `let (_, s, _) = compat::svd(&matrix.view(), true, true, false, true, "gesdd").un...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 565: `let norm_2 = compat::norm(&matrix.view(), Some("2"), None, false, true).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 569: `let fro_norm = compat::norm(&matrix.view(), Some("fro"), None, false, true).unwr...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 570: `let s_fro: f64 = s.iter().map(|&sigma| sigma * sigma).sum::<f64>().sqrt();`
  - **Fix**: Mathematical operation .sqrt() without validation
- Line 581: `let det_small = compat::det(&small_matrix.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 586: `let cond_wide = compat::cond(&wide_range.view(), Some("2")).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 592: `let det_nearly: f64 = compat::det(&nearly_singular.view(), false, true).unwrap()...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 596: `compat::matrix_rank(&nearly_singular.view(), Some(1e-12), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 632: `let _det = compat::det(&matrix.view(), false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 633: `let _norm = compat::norm(&matrix.view(), Some("fro"), None, false, true).unwrap(...`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 673: `let _lu = compat::lu(&matrix.view(), false, false, true, false).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 679: `let _qr = compat::qr(&matrix.view(), false, None, "full", false, true).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/scipy_validation.rs

16 issues found:

- Line 20: `let actual_det = det(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 29: `let actual_det = det(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 44: `let actual_inv = inv(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 54: `let (p, l, u) = lu(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 82: `let (q, r) = qr(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 108: `let (u, s, vt) = svd(&a.view(), false, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 140: `let l = cholesky(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 162: `let actual_frobenius = matrix_norm(&a.view(), "fro", None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 168: `let actual_1norm = matrix_norm(&a.view(), "1", None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 174: `let actual_infnorm = matrix_norm(&a.view(), "inf", None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 186: `let actual_x = solve(&a.view(), &b.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 196: `let det_result = det(&singular.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 203: `let identity_det = det(&identity.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 207: `let identity_inv = inv(&identity.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 217: `let det_result = det(&a.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 223: `let small_det = det(&small.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/svd_implementation_tests.rs

7 issues found:

- Line 8: `let (u, s, vt) = svd(&a.view(), false, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 38: `let (u, s, vt) = svd(&a.view(), false, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 62: `let (u, s, vt) = svd(&a.view(), false, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 98: `let (u, s, vt) = svd(&a.view(), true, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 129: `let a_1x1 = a_view.view().into_shape_with_order((1, 1)).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 130: `let (u, s, vt) = svd(&a_1x1, false, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()
- Line 148: `let (u, s, vt) = svd(&a.view(), false, None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()

### tests/test_eigenvalue_precision.rs

1 issues found:

- Line 8: `let (eigenvals, eigenvecs) = eigh(&symmetric_matrix.view(), None).unwrap();`
  - **Fix**: Replace with ? operator or .ok_or()